//! The deblocking filter (H.264 clause 8.7), run over the whole picture
//! once every slice is decoded: boundary strength per edge segment, then
//! the luma and chroma edge filters, macroblock by macroblock in raster
//! order — vertical edges first, then horizontal, luma then chroma.

use crate::dsp::h264::H264Dsp;
use crate::sample::Sample;

use super::frame::Frame;
use super::mb::{MbKind, PicInfo};
use super::tables::{ALPHA, BETA, TC0};

/// The per-slice deblocking parameters the filter needs at each macroblock.
#[derive(Debug, Clone, Copy, Default)]
pub struct DeblockParams {
    /// `disable_deblocking_filter_idc`.
    pub disable_idc: u32,
    /// `FilterOffsetA`.
    pub offset_a: i32,
    /// `FilterOffsetB`.
    pub offset_b: i32,
}

/// The 16-bit "has coefficients" mask of a macroblock's 4x4 luma blocks
/// (raster), with an 8x8 transform's four blocks all set when any is.
#[inline]
fn nz_mask(info: &PicInfo, addr: usize) -> u16 {
    let base = addr * 16;
    let mut m = 0u16;
    for b in 0..16 {
        m |= ((info.luma_nz[base + b] != 0) as u16) << b;
    }
    if info.mbs[addr].transform_8x8 {
        // Spread each 8x8's bit over its four 4x4s.
        let q = |bits: u16| -> u16 { if bits != 0 { 0x33 } else { 0 } };
        let tl = q(m & 0x0033);
        let tr = q(m & 0x00cc) << 2;
        let bl = q(m & 0x3300) << 8;
        let br = q(m & 0xcc00) << 10;
        m = tl | tr | bl | br;
    }
    m
}

/// bS 0 or 1 from motion alone (8.7.2.1, the last three rules), for two
/// blocks of inter macroblocks with no coefficients. `mvy_limit` is 4 for
/// frame macroblocks and 2 for field macroblocks (their vertical vectors
/// are in quarter field samples; NOTE 3 of 8.7.2.1).
#[inline]
fn motion_bs<S: Sample>(
    frame: &Frame<S>,
    pa: usize,
    p_blk: usize,
    qa: usize,
    q_blk: usize,
    mvy_limit: i16,
) -> u8 {
    use super::frame::{BlockMotion, Mv};
    let p0 = &frame.motion[0][pa * 16 + p_blk];
    let p1 = &frame.motion[1][pa * 16 + p_blk];
    let q0 = &frame.motion[0][qa * 16 + q_blk];
    let q1 = &frame.motion[1][qa * 16 + q_blk];
    let (pn, qn) = (
        (p0.ref_idx >= 0) as u32 + (p1.ref_idx >= 0) as u32,
        (q0.ref_idx >= 0) as u32 + (q1.ref_idx >= 0) as u32,
    );
    if pn != qn {
        return 1;
    }
    let mv_far =
        |a: Mv, b: Mv| -> bool { (a.x - b.x).abs() >= 4 || (a.y - b.y).abs() >= mvy_limit };
    #[inline(always)]
    fn same_pic(a: &BlockMotion, b: &BlockMotion) -> bool {
        a.same_ref(b)
    }
    if pn == 1 {
        let pp = if p0.ref_idx >= 0 { p0 } else { p1 };
        let qq = if q0.ref_idx >= 0 { q0 } else { q1 };
        if !same_pic(pp, qq) {
            return 1;
        }
        return mv_far(pp.mv, qq.mv) as u8;
    }
    let straight = same_pic(p0, q0) && same_pic(p1, q1);
    let crossed = same_pic(p0, q1) && same_pic(p1, q0);
    if !straight && !crossed {
        return 1;
    }
    if same_pic(p0, p1) {
        let pair_a = mv_far(p0.mv, q0.mv) || mv_far(p1.mv, q1.mv);
        let pair_b = mv_far(p0.mv, q1.mv) || mv_far(p1.mv, q0.mv);
        return (pair_a && pair_b) as u8;
    }
    if straight {
        return (mv_far(p0.mv, q0.mv) || mv_far(p1.mv, q1.mv)) as u8;
    }
    (mv_far(p0.mv, q1.mv) || mv_far(p1.mv, q0.mv)) as u8
}

#[inline(always)]
fn clip3(lo: i32, hi: i32, v: i32) -> i32 {
    v.clamp(lo, hi)
}

/// tC0 per 4-line segment for a bS < 4 edge (−1 = bS 0, leave alone).
#[inline]
fn tc0_of(bs: &[u8; 4], index_a: i32, bd_shift: u32) -> [i16; 4] {
    let mut t = [-1i16; 4];
    for k in 0..4 {
        if bs[k] != 0 {
            t[k] = (TC0[index_a as usize][(bs[k] - 1).min(2) as usize] as i16) << bd_shift;
        }
    }
    t
}

/// Deblock the whole picture in place.
pub fn deblock_picture<S: Sample>(
    dsp: &H264Dsp<S>,
    frame: &mut Frame<S>,
    info: &PicInfo,
    params: &[DeblockParams],
) {
    deblock_mb_rows(dsp, frame, info, params, 0, info.mb_height);
}

/// Deblock macroblock rows `r0..r1` in raster order (each row's top edges
/// reach three lines into the row above). Rows must be filtered in order,
/// and a row only after the row below it is decoded (intra prediction
/// reads unfiltered neighbours), which is how the picture-level filter
/// order is preserved when rows are filtered as decoding proceeds.
pub fn deblock_mb_rows<S: Sample>(
    dsp: &H264Dsp<S>,
    frame: &mut Frame<S>,
    info: &PicInfo,
    params: &[DeblockParams],
    r0: usize,
    r1: usize,
) {
    if frame.mbaff {
        // Pair by pair, in the macroblocks' own frame / field geometry.
        deblock_mbaff_pairs(dsp, frame, info, params, r0 / 2, r1.div_ceil(2));
        return;
    }
    let mbw = info.mb_width;
    let mbh = info.mb_height;
    // Thresholds scale with the bit depth (8.7.2.2): alpha, beta and tC0 are
    // multiplied by 2^(BitDepth − 8).
    let bd_shift = frame.bit_depth - 8;
    let max = (1i32 << frame.bit_depth) - 1;
    // A field picture: every macroblock is a field macroblock — intra
    // horizontal edges get bS 3 (only vertical ones 4), and vertical vector
    // differences count in field units.
    let field_pic = frame.field_coded;
    let mvy_limit: i16 = if field_pic { 2 } else { 4 };
    for mby in r0..r1.min(mbh) {
        for mbx in 0..mbw {
            let addr = mby * mbw + mbx;
            let m = &info.mbs[addr];
            if !m.decoded {
                continue;
            }
            let par = params[m.slice as usize];
            if par.disable_idc == 1 {
                continue;
            }
            let left = if mbx > 0 { Some(addr - 1) } else { None };
            let above = if mby > 0 { Some(addr - mbw) } else { None };
            let across_slices = par.disable_idc != 2;
            let filter_left = left.is_some_and(|l| {
                info.mbs[l].decoded && (across_slices || info.mbs[l].slice == m.slice)
            });
            let filter_top = above.is_some_and(|a| {
                info.mbs[a].decoded && (across_slices || info.mbs[a].slice == m.slice)
            });

            // Boundary strengths for the vertical edges (4 edges x 4 rows)
            // and horizontal edges (4 edges x 4 columns).
            let mut bs_v = [[0u8; 4]; 4];
            let mut bs_h = [[0u8; 4]; 4];
            // The odd internal edges are no luma transform edges under the
            // 8x8 transform (luma skips them below), but 4:2:2 chroma still
            // filters its edges there with the strength they would have.
            let internal_odd = !m.transform_8x8;
            if m.kind.is_intra() {
                for e in 0..4 {
                    let v = if e == 0 { 4 } else { 3 };
                    if e != 0 || filter_left {
                        bs_v[e] = [v; 4];
                    }
                    if e != 0 || filter_top {
                        bs_h[e] = [if e == 0 && field_pic { 3 } else { v }; 4];
                    }
                }
            } else {
                let nz = nz_mask(info, addr);
                // Internal edges: coefficients, else motion.
                for e in 1..4 {
                    for k in 0..4 {
                        let (pb, qb) = (k * 4 + e - 1, k * 4 + e);
                        bs_v[e][k] = if (nz >> pb | nz >> qb) & 1 != 0 {
                            2
                        } else {
                            motion_bs(frame, addr, pb, addr, qb, mvy_limit)
                        };
                        let (pb, qb) = ((e - 1) * 4 + k, e * 4 + k);
                        bs_h[e][k] = if (nz >> pb | nz >> qb) & 1 != 0 {
                            2
                        } else {
                            motion_bs(frame, addr, pb, addr, qb, mvy_limit)
                        };
                    }
                }
                if filter_left {
                    let l = left.unwrap();
                    if info.mbs[l].kind.is_intra() {
                        bs_v[0] = [4; 4];
                    } else {
                        let nzl = nz_mask(info, l);
                        for k in 0..4 {
                            let (pb, qb) = (k * 4 + 3, k * 4);
                            bs_v[0][k] = if (nzl >> pb | nz >> qb) & 1 != 0 {
                                2
                            } else {
                                motion_bs(frame, l, pb, addr, qb, mvy_limit)
                            };
                        }
                    }
                }
                if filter_top {
                    let a = above.unwrap();
                    if info.mbs[a].kind.is_intra() {
                        bs_h[0] = [if field_pic { 3 } else { 4 }; 4];
                    } else {
                        let nza = nz_mask(info, a);
                        for k in 0..4 {
                            let (pb, qb) = (12 + k, k);
                            bs_h[0][k] = if (nza >> pb | nz >> qb) & 1 != 0 {
                                2
                            } else {
                                motion_bs(frame, a, pb, addr, qb, mvy_limit)
                            };
                        }
                    }
                }
            }

            let (x0, y0) = (mbx * 16, mby * 16);
            // Luma-style filtering: the luma plane, and in 4:4:4 the chroma
            // planes too (chromaStyleFilteringFlag is 0 there: the luma
            // filters and edge set, at the chroma QP).
            let luma_style_planes = if frame.chroma == crate::picture::ChromaFormat::Yuv444 {
                3
            } else {
                1
            };
            for p in 0..luma_style_planes {
                let qp_of = |mb: &super::mb::MbInfo| -> i32 {
                    if p == 0 {
                        mb.qp as i32
                    } else {
                        mb.qpc[p - 1] as i32
                    }
                };
                let qp_cur = qp_of(m);
                let plane = match p {
                    0 => &mut frame.y,
                    1 => &mut frame.cb,
                    _ => &mut frame.cr,
                };
                let stride = plane.stride;
                // Vertical edges.
                for e in 0..4 {
                    if bs_v[e].iter().all(|&b| b == 0) || (e % 2 == 1 && !internal_odd) {
                        continue;
                    }
                    let qp_p = if e == 0 {
                        qp_of(&info.mbs[left.unwrap()])
                    } else {
                        qp_cur
                    };
                    let qp_av = (qp_p + qp_cur + 1) >> 1;
                    let index_a = clip3(0, 51, qp_av + par.offset_a);
                    let index_b = clip3(0, 51, qp_av + par.offset_b);
                    let (alpha, beta) = (
                        (ALPHA[index_a as usize] as i32) << bd_shift,
                        (BETA[index_b as usize] as i32) << bd_shift,
                    );
                    let off = plane.offset((x0 + e * 4) as isize, y0 as isize);
                    if bs_v[e][0] == 4 {
                        (dsp.deblock_luma_v_intra)(&mut plane.data, off, stride, alpha, beta, max);
                    } else {
                        (dsp.deblock_luma_v)(
                            &mut plane.data,
                            off,
                            stride,
                            alpha,
                            beta,
                            &tc0_of(&bs_v[e], index_a, bd_shift),
                            max,
                        );
                    }
                }
                // Horizontal edges.
                for e in 0..4 {
                    if bs_h[e].iter().all(|&b| b == 0) || (e % 2 == 1 && !internal_odd) {
                        continue;
                    }
                    let qp_p = if e == 0 {
                        qp_of(&info.mbs[above.unwrap()])
                    } else {
                        qp_cur
                    };
                    let qp_av = (qp_p + qp_cur + 1) >> 1;
                    let index_a = clip3(0, 51, qp_av + par.offset_a);
                    let index_b = clip3(0, 51, qp_av + par.offset_b);
                    let (alpha, beta) = (
                        (ALPHA[index_a as usize] as i32) << bd_shift,
                        (BETA[index_b as usize] as i32) << bd_shift,
                    );
                    let off = plane.offset(x0 as isize, (y0 + e * 4) as isize);
                    if bs_h[e][0] == 4 {
                        (dsp.deblock_luma_h_intra)(&mut plane.data, off, stride, alpha, beta, max);
                    } else {
                        (dsp.deblock_luma_h)(
                            &mut plane.data,
                            off,
                            stride,
                            alpha,
                            beta,
                            &tc0_of(&bs_h[e], index_a, bd_shift),
                            max,
                        );
                    }
                }
            }
            // Chroma. Vertical edges at chroma x = 0 and 4 (luma edges 0
            // and 2); horizontal edges at chroma y = 0 and 4 for 4:2:0
            // (luma edges 0, 2) and at 0, 4, 8, 12 for 4:2:2 (all four luma
            // edge rows). A kernel call covers eight chroma lines with
            // `tc0[i / 2]`: for 4:2:0 that is the four luma segments (two
            // chroma lines each); for 4:2:2 a vertical edge is sixteen lines,
            // two calls of two luma segments (four chroma lines each).
            let c422 = frame.chroma == crate::picture::ChromaFormat::Yuv422;
            if frame.chroma == crate::picture::ChromaFormat::Yuv420 || c422 {
                let mbh_c = if c422 { 16 } else { 8 };
                for comp in 0..2 {
                    for &e in &[0usize, 2] {
                        if bs_v[e].iter().all(|&b| b == 0) {
                            continue;
                        }
                        let (qpc_p, qpc_q) = if e == 0 {
                            (info.mbs[left.unwrap()].qpc[comp] as i32, m.qpc[comp] as i32)
                        } else {
                            (m.qpc[comp] as i32, m.qpc[comp] as i32)
                        };
                        let qp_av = (qpc_p + qpc_q + 1) >> 1;
                        let index_a = clip3(0, 51, qp_av + par.offset_a);
                        let index_b = clip3(0, 51, qp_av + par.offset_b);
                        let (alpha, beta) = (
                            (ALPHA[index_a as usize] as i32) << bd_shift,
                            (BETA[index_b as usize] as i32) << bd_shift,
                        );
                        let plane = if comp == 0 {
                            &mut frame.cb
                        } else {
                            &mut frame.cr
                        };
                        let stride = plane.stride;
                        let tc = tc0_of(&bs_v[e], index_a, bd_shift);
                        if !c422 {
                            let off =
                                plane.offset((mbx * 8 + e * 2) as isize, (mby * mbh_c) as isize);
                            if bs_v[e][0] == 4 {
                                (dsp.deblock_chroma_v_intra)(
                                    &mut plane.data,
                                    off,
                                    stride,
                                    alpha,
                                    beta,
                                    max,
                                );
                            } else {
                                (dsp.deblock_chroma_v)(
                                    &mut plane.data,
                                    off,
                                    stride,
                                    alpha,
                                    beta,
                                    &tc,
                                    max,
                                );
                            }
                        } else {
                            for half in 0..2 {
                                let off = plane.offset(
                                    (mbx * 8 + e * 2) as isize,
                                    (mby * mbh_c + half * 8) as isize,
                                );
                                let t = [
                                    tc[2 * half],
                                    tc[2 * half],
                                    tc[2 * half + 1],
                                    tc[2 * half + 1],
                                ];
                                if bs_v[e][2 * half] == 4 {
                                    (dsp.deblock_chroma_v_intra)(
                                        &mut plane.data,
                                        off,
                                        stride,
                                        alpha,
                                        beta,
                                        max,
                                    );
                                } else if t.iter().any(|&v| v >= 0) {
                                    (dsp.deblock_chroma_v)(
                                        &mut plane.data,
                                        off,
                                        stride,
                                        alpha,
                                        beta,
                                        &t,
                                        max,
                                    );
                                }
                            }
                        }
                    }
                    let h_edges: &[usize] = if c422 { &[0, 1, 2, 3] } else { &[0, 2] };
                    for &e in h_edges {
                        if bs_h[e].iter().all(|&b| b == 0) {
                            continue;
                        }
                        let (qpc_p, qpc_q) = if e == 0 {
                            (
                                info.mbs[above.unwrap()].qpc[comp] as i32,
                                m.qpc[comp] as i32,
                            )
                        } else {
                            (m.qpc[comp] as i32, m.qpc[comp] as i32)
                        };
                        let qp_av = (qpc_p + qpc_q + 1) >> 1;
                        let index_a = clip3(0, 51, qp_av + par.offset_a);
                        let index_b = clip3(0, 51, qp_av + par.offset_b);
                        let (alpha, beta) = (
                            (ALPHA[index_a as usize] as i32) << bd_shift,
                            (BETA[index_b as usize] as i32) << bd_shift,
                        );
                        let plane = if comp == 0 {
                            &mut frame.cb
                        } else {
                            &mut frame.cr
                        };
                        // Chroma row of the edge: 4:2:0 e*2, 4:2:2 e*4.
                        let cy = if c422 { e * 4 } else { e * 2 };
                        let off = plane.offset((mbx * 8) as isize, (mby * mbh_c + cy) as isize);
                        let stride = plane.stride;
                        if bs_h[e][0] == 4 {
                            (dsp.deblock_chroma_h_intra)(
                                &mut plane.data,
                                off,
                                stride,
                                alpha,
                                beta,
                                max,
                            );
                        } else {
                            (dsp.deblock_chroma_h)(
                                &mut plane.data,
                                off,
                                stride,
                                alpha,
                                beta,
                                &tc0_of(&bs_h[e], index_a, bd_shift),
                                max,
                            );
                        }
                    }
                }
            }
        }
    }
    let _ = MbKind::PSkip;
}

// ----------------------------------------------------------------------
// MBAFF frames (8.7 with MbaffFrameFlag = 1)
// ----------------------------------------------------------------------

/// One side of an edge: the macroblock and the 4x4 block (raster) whose
/// samples sit on that side of a given line.
#[derive(Clone, Copy)]
struct Side {
    addr: usize,
    blk: usize,
}

/// bS for one set of samples across an edge (8.7.2.1) in an MBAFF frame:
/// `mixed` is mixedModeEdgeFlag, `vertical` verticalEdgeFlag, and both
/// sides carry the macroblock and 4x4 block holding p0 / q0.
#[allow(clippy::too_many_arguments)]
fn bs_mbaff<S: Sample>(
    frame: &Frame<S>,
    info: &PicInfo,
    p: Side,
    q: Side,
    mixed: bool,
    vertical: bool,
    nz_p: u16,
    nz_q: u16,
) -> u8 {
    let mp = &info.mbs[p.addr];
    let mq = &info.mbs[q.addr];
    let intra = mp.kind.is_intra() || mq.kind.is_intra();
    if intra {
        // Macroblock edges: 4 on vertical ones, and on horizontal ones
        // between two frame macroblocks; 3 otherwise (mixed, fields) and on
        // internal edges.
        let mb_edge = p.addr != q.addr;
        return if mb_edge && (vertical || (!mixed && !mp.field && !mq.field)) {
            4
        } else {
            3
        };
    }
    if (nz_p >> p.blk) & 1 != 0 || (nz_q >> q.blk) & 1 != 0 {
        return 2;
    }
    if mixed {
        return 1;
    }
    let mvy_limit: i16 = if mq.field { 2 } else { 4 };
    motion_bs(frame, p.addr, p.blk, q.addr, q.blk, mvy_limit)
}

/// The luma filter thresholds for a p / q macroblock pair.
#[inline]
fn thresholds(qp_p: i32, qp_q: i32, par: &DeblockParams, bd_shift: u32) -> (i32, i32, i32) {
    let qp_av = (qp_p + qp_q + 1) >> 1;
    let index_a = clip3(0, 51, qp_av + par.offset_a);
    let index_b = clip3(0, 51, qp_av + par.offset_b);
    (
        index_a,
        (ALPHA[index_a as usize] as i32) << bd_shift,
        (BETA[index_b as usize] as i32) << bd_shift,
    )
}

/// tC0 for one line of a bS < 4 edge (bS 0: no filtering, `None`).
#[inline]
fn tc0_line(bs: u8, index_a: i32, bd_shift: u32) -> Option<Option<i32>> {
    match bs {
        0 => None,
        4 => Some(None),
        _ => Some(Some(
            (TC0[index_a as usize][(bs - 1).min(2) as usize] as i32) << bd_shift,
        )),
    }
}

/// Filter one line of luma or chroma samples across an edge at `pos`
/// (the q0 sample), `across` being the sample step across the edge.
#[inline]
fn filter_line<S: Sample>(
    data: &mut [S],
    pos: usize,
    across: usize,
    tc0: Option<i32>,
    alpha: i32,
    beta: i32,
    chroma: bool,
    max: i32,
) {
    let mut p = [0i32; 4];
    let mut q = [0i32; 4];
    let taps = if chroma { 2 } else { 4 };
    for k in 0..taps {
        p[k] = data[pos - (k + 1) * across].to_i32();
        q[k] = data[pos + k * across].to_i32();
    }
    crate::dsp::h264::deblock_line(&mut p, &mut q, tc0, alpha, beta, chroma, max);
    let writes = if chroma { 1 } else { 3 };
    for k in 0..writes {
        data[pos - (k + 1) * across] = S::from_i32(p[k]);
        data[pos + k * across] = S::from_i32(q[k]);
    }
}

/// Deblock the macroblock pairs of pair rows `pr0..pr1` of an MBAFF frame,
/// pair by pair (top macroblock then bottom), each macroblock's edges in
/// its own geometry — a field macroblock's on its own field lines — with
/// the mixed frame / field edges of 8.7 filtered line by line.
fn deblock_mbaff_pairs<S: Sample>(
    dsp: &H264Dsp<S>,
    frame: &mut Frame<S>,
    info: &PicInfo,
    params: &[DeblockParams],
    pr0: usize,
    pr1: usize,
) {
    let mbw = info.mb_width;
    let pairs_h = info.mb_height / 2;
    let bd_shift = frame.bit_depth - 8;
    let max = (1i32 << frame.bit_depth) - 1;
    let chroma420 = frame.chroma == crate::picture::ChromaFormat::Yuv420;
    let chroma422 = frame.chroma == crate::picture::ChromaFormat::Yuv422;
    // 4:2:0 and 4:2:2 chroma; 4:4:4 MBAFF (luma-style chroma) is not
    // handled here (no such streams are in circulation) — luma only.
    let do_chroma = chroma420 || chroma422;
    let mbh_c = if chroma422 { 16 } else { 8 };
    for pr in pr0..pr1.min(pairs_h) {
        for mbx in 0..mbw {
            let top_addr = (2 * pr) * mbw + mbx;
            for bottom in 0..2usize {
                let sa = top_addr + bottom * mbw;
                let m = &info.mbs[sa];
                if !m.decoded {
                    continue;
                }
                let par = params[m.slice as usize];
                if par.disable_idc == 1 {
                    continue;
                }
                let mf = m.field;
                let dy = if mf { 2 } else { 1 };
                let across_slices = par.disable_idc != 2;
                // Neighbouring pairs (top macroblock storage addresses).
                let left_top = if mbx > 0 { Some(top_addr - 1) } else { None };
                let above_top = if pr > 0 {
                    Some(top_addr - 2 * mbw)
                } else {
                    None
                };
                let avail = |a: Option<usize>| {
                    a.is_some_and(|t| {
                        info.mbs[t].decoded && (across_slices || info.mbs[t].slice == m.slice)
                    })
                };
                let filter_left = avail(left_top);
                let filter_top = if mf {
                    avail(above_top)
                } else if bottom == 1 {
                    true
                } else {
                    avail(above_top)
                };
                // Luma geometry.
                let (x0, y_p) = (mbx * 16, pr * 32 + if mf { bottom } else { 16 * bottom });
                let ystride = frame.y.stride;
                let cstride = frame.cb.stride;
                let (xc0, yc_p) = (
                    mbx * 8,
                    pr * 2 * mbh_c + if mf { bottom } else { mbh_c * bottom },
                );
                let nz_q = nz_mask(info, sa);
                let internal_odd = !m.transform_8x8;

                // ---------- vertical edges ----------
                // Left MB edge.
                if filter_left {
                    let lt = left_top.unwrap();
                    let left_field = info.mbs[lt].field;
                    if left_field == mf {
                        // Same kind: one macroblock on the p side, sixteen
                        // lines in the current geometry.
                        let pa = if bottom == 1 { lt + mbw } else { lt };
                        let nz_p = nz_mask(info, pa);
                        let mut bs = [0u8; 4];
                        for k in 0..4 {
                            bs[k] = bs_mbaff(
                                frame,
                                info,
                                Side {
                                    addr: pa,
                                    blk: k * 4 + 3,
                                },
                                Side {
                                    addr: sa,
                                    blk: k * 4,
                                },
                                false,
                                true,
                                nz_p,
                                nz_q,
                            );
                        }
                        let (index_a, alpha, beta) =
                            thresholds(info.mbs[pa].qp as i32, m.qp as i32, &par, bd_shift);
                        let off = frame.y.offset(x0 as isize, y_p as isize);
                        if bs.iter().any(|&b| b != 0) {
                            if bs[0] == 4 {
                                (dsp.deblock_luma_v_intra)(
                                    &mut frame.y.data,
                                    off,
                                    ystride * dy,
                                    alpha,
                                    beta,
                                    max,
                                );
                            } else {
                                (dsp.deblock_luma_v)(
                                    &mut frame.y.data,
                                    off,
                                    ystride * dy,
                                    alpha,
                                    beta,
                                    &tc0_of(&bs, index_a, bd_shift),
                                    max,
                                );
                            }
                        }
                        if do_chroma && bs.iter().any(|&b| b != 0) {
                            for comp in 0..2 {
                                let (index_a, alpha, beta) = thresholds(
                                    info.mbs[pa].qpc[comp] as i32,
                                    m.qpc[comp] as i32,
                                    &par,
                                    bd_shift,
                                );
                                let plane = if comp == 0 {
                                    &mut frame.cb
                                } else {
                                    &mut frame.cr
                                };
                                chroma_v_edge(
                                    dsp, plane, xc0, yc_p, dy, chroma422, &bs, index_a, alpha,
                                    beta, bd_shift, max,
                                );
                            }
                        }
                    } else {
                        // Mixed: line by line — each line's p samples belong
                        // to one of the two left macroblocks.
                        let mut line_bs = [0u8; 16];
                        let mut line_pa = [0usize; 16];
                        for k in 0..16 {
                            // The pair line of this current line and the p
                            // macroblock / block row holding it.
                            let (pa, prow) = if !mf {
                                // Frame MB, field left pair: pair line 16*bottom+k
                                // is field line (16*bottom+k)/2 of the field MB of
                                // that parity.
                                let pl = 16 * bottom + k;
                                (if pl % 2 == 0 { lt } else { lt + mbw }, pl / 2)
                            } else {
                                // Field MB, frame left pair: pair line 2k+bottom
                                // is line (2k+bottom) % 16 of frame MB (2k+bottom)/16.
                                let pl = 2 * k + bottom;
                                (if pl < 16 { lt } else { lt + mbw }, pl % 16)
                            };
                            let nz_p = nz_mask(info, pa);
                            line_bs[k] = bs_mbaff(
                                frame,
                                info,
                                Side {
                                    addr: pa,
                                    blk: (prow / 4) * 4 + 3,
                                },
                                Side {
                                    addr: sa,
                                    blk: (k / 4) * 4,
                                },
                                true,
                                true,
                                nz_p,
                                nz_q,
                            );
                            line_pa[k] = pa;
                        }
                        for k in 0..16 {
                            if line_bs[k] == 0 {
                                continue;
                            }
                            let pa = line_pa[k];
                            let (index_a, alpha, beta) =
                                thresholds(info.mbs[pa].qp as i32, m.qp as i32, &par, bd_shift);
                            let Some(tc) = tc0_line(line_bs[k], index_a, bd_shift) else {
                                continue;
                            };
                            let pos = frame.y.offset(x0 as isize, (y_p + dy * k) as isize);
                            filter_line(&mut frame.y.data, pos, 1, tc, alpha, beta, false, max);
                        }
                        if do_chroma {
                            // Chroma line k takes the bS of the luma line at
                            // its position in the same field: 2k for a field
                            // MB, and for a frame MB 2k on even lines (top
                            // field) but 2k - 1 on odd ones (bottom field) —
                            // and that line's p macroblock.
                            for k in 0..mbh_c {
                                let l = if chroma422 {
                                    k
                                } else if !mf && k % 2 == 1 {
                                    2 * k - 1
                                } else {
                                    2 * k
                                };
                                if line_bs[l] == 0 {
                                    continue;
                                }
                                let pa = line_pa[l];
                                for comp in 0..2 {
                                    let (index_a, alpha, beta) = thresholds(
                                        info.mbs[pa].qpc[comp] as i32,
                                        m.qpc[comp] as i32,
                                        &par,
                                        bd_shift,
                                    );
                                    let Some(tc) = tc0_line(line_bs[l], index_a, bd_shift) else {
                                        continue;
                                    };
                                    let plane = if comp == 0 {
                                        &mut frame.cb
                                    } else {
                                        &mut frame.cr
                                    };
                                    let pos = plane.offset(xc0 as isize, (yc_p + dy * k) as isize);
                                    filter_line(
                                        &mut plane.data,
                                        pos,
                                        1,
                                        tc,
                                        alpha,
                                        beta,
                                        true,
                                        max,
                                    );
                                }
                            }
                        }
                    }
                }
                // Internal vertical edges.
                for e in 1..4 {
                    if e % 2 == 1 && !internal_odd {
                        // Chroma still has its edge at e == 2 only (4:2:0).
                        continue;
                    }
                    let mut bs = [0u8; 4];
                    for k in 0..4 {
                        bs[k] = bs_mbaff(
                            frame,
                            info,
                            Side {
                                addr: sa,
                                blk: k * 4 + e - 1,
                            },
                            Side {
                                addr: sa,
                                blk: k * 4 + e,
                            },
                            false,
                            true,
                            nz_q,
                            nz_q,
                        );
                    }
                    if bs.iter().all(|&b| b == 0) {
                        continue;
                    }
                    let (index_a, alpha, beta) =
                        thresholds(m.qp as i32, m.qp as i32, &par, bd_shift);
                    let off = frame.y.offset((x0 + e * 4) as isize, y_p as isize);
                    if bs[0] == 4 {
                        (dsp.deblock_luma_v_intra)(
                            &mut frame.y.data,
                            off,
                            ystride * dy,
                            alpha,
                            beta,
                            max,
                        );
                    } else {
                        (dsp.deblock_luma_v)(
                            &mut frame.y.data,
                            off,
                            ystride * dy,
                            alpha,
                            beta,
                            &tc0_of(&bs, index_a, bd_shift),
                            max,
                        );
                    }
                    if do_chroma && e == 2 {
                        for comp in 0..2 {
                            let (index_a, alpha, beta) =
                                thresholds(m.qpc[comp] as i32, m.qpc[comp] as i32, &par, bd_shift);
                            let plane = if comp == 0 {
                                &mut frame.cb
                            } else {
                                &mut frame.cr
                            };
                            chroma_v_edge(
                                dsp,
                                plane,
                                xc0 + 4,
                                yc_p,
                                dy,
                                chroma422,
                                &bs,
                                index_a,
                                alpha,
                                beta,
                                bd_shift,
                                max,
                            );
                        }
                    }
                }

                // ---------- horizontal edges ----------
                // Top MB edge.
                if filter_top {
                    if !mf && bottom == 1 {
                        // Bottom frame MB: the edge with the top frame MB of
                        // the pair — two frame macroblocks.
                        let pa = sa - mbw;
                        let nz_p = nz_mask(info, pa);
                        let mut bs = [0u8; 4];
                        for k in 0..4 {
                            bs[k] = bs_mbaff(
                                frame,
                                info,
                                Side {
                                    addr: pa,
                                    blk: 12 + k,
                                },
                                Side { addr: sa, blk: k },
                                false,
                                false,
                                nz_p,
                                nz_q,
                            );
                        }
                        top_edge_plain(
                            dsp, frame, info, &par, sa, pa, &bs, x0, y_p, xc0, yc_p, 1, do_chroma,
                            bd_shift, max,
                        );
                    } else if !mf {
                        // Top frame MB below a pair.
                        let at = above_top.unwrap();
                        if info.mbs[at + mbw].field {
                            // Above pair is field: the edge is filtered as two
                            // field edges, the current MB's even lines against
                            // the above top field MB, its odd lines against the
                            // bottom one (fieldModeInFrameFilteringFlag = 1).
                            for pass in 0..2usize {
                                let pa = at + pass * mbw;
                                let nz_p = nz_mask(info, pa);
                                let mut bs = [0u8; 4];
                                for k in 0..4 {
                                    bs[k] = bs_mbaff(
                                        frame,
                                        info,
                                        Side {
                                            addr: pa,
                                            blk: 12 + k,
                                        },
                                        Side { addr: sa, blk: k },
                                        true,
                                        false,
                                        nz_p,
                                        nz_q,
                                    );
                                }
                                if bs.iter().all(|&b| b == 0) {
                                    continue;
                                }
                                let (index_a, alpha, beta) =
                                    thresholds(info.mbs[pa].qp as i32, m.qp as i32, &par, bd_shift);
                                // q0 at row y_p + pass, step 2 across the edge.
                                let off = frame.y.offset(x0 as isize, (y_p + pass) as isize);
                                if bs[0] == 4 {
                                    (dsp.deblock_luma_h_intra)(
                                        &mut frame.y.data,
                                        off,
                                        ystride * 2,
                                        alpha,
                                        beta,
                                        max,
                                    );
                                } else {
                                    (dsp.deblock_luma_h)(
                                        &mut frame.y.data,
                                        off,
                                        ystride * 2,
                                        alpha,
                                        beta,
                                        &tc0_of(&bs, index_a, bd_shift),
                                        max,
                                    );
                                }
                                if do_chroma {
                                    for comp in 0..2 {
                                        let (index_a, alpha, beta) = thresholds(
                                            info.mbs[pa].qpc[comp] as i32,
                                            m.qpc[comp] as i32,
                                            &par,
                                            bd_shift,
                                        );
                                        let plane = if comp == 0 {
                                            &mut frame.cb
                                        } else {
                                            &mut frame.cr
                                        };
                                        let off =
                                            plane.offset(xc0 as isize, (yc_p + pass) as isize);
                                        if bs[0] == 4 {
                                            (dsp.deblock_chroma_h_intra)(
                                                &mut plane.data,
                                                off,
                                                cstride * 2,
                                                alpha,
                                                beta,
                                                max,
                                            );
                                        } else {
                                            (dsp.deblock_chroma_h)(
                                                &mut plane.data,
                                                off,
                                                cstride * 2,
                                                alpha,
                                                beta,
                                                &tc0_of(&bs, index_a, bd_shift),
                                                max,
                                            );
                                        }
                                    }
                                }
                            }
                        } else {
                            // Above pair is frame: a plain frame edge with its
                            // bottom macroblock.
                            let pa = at + mbw;
                            let nz_p = nz_mask(info, pa);
                            let mut bs = [0u8; 4];
                            for k in 0..4 {
                                bs[k] = bs_mbaff(
                                    frame,
                                    info,
                                    Side {
                                        addr: pa,
                                        blk: 12 + k,
                                    },
                                    Side { addr: sa, blk: k },
                                    false,
                                    false,
                                    nz_p,
                                    nz_q,
                                );
                            }
                            top_edge_plain(
                                dsp, frame, info, &par, sa, pa, &bs, x0, y_p, xc0, yc_p, 1,
                                do_chroma, bd_shift, max,
                            );
                        }
                    } else {
                        // Field MB: its previous field lines are two frame
                        // rows up — in the same-parity field MB of a field
                        // pair above, or the bottom frame MB of a frame pair
                        // (a mixed edge).
                        let at = above_top.unwrap();
                        let above_field = info.mbs[at + mbw].field;
                        let pa = if above_field {
                            at + bottom * mbw
                        } else {
                            at + mbw
                        };
                        let mixed = !above_field;
                        let nz_p = nz_mask(info, pa);
                        let mut bs = [0u8; 4];
                        for k in 0..4 {
                            bs[k] = bs_mbaff(
                                frame,
                                info,
                                Side {
                                    addr: pa,
                                    blk: 12 + k,
                                },
                                Side { addr: sa, blk: k },
                                mixed,
                                false,
                                nz_p,
                                nz_q,
                            );
                        }
                        top_edge_plain(
                            dsp, frame, info, &par, sa, pa, &bs, x0, y_p, xc0, yc_p, 2, do_chroma,
                            bd_shift, max,
                        );
                    }
                }
                // Internal horizontal edges (luma skips the odd ones under the
                // 8x8 transform; 4:2:2 chroma has all three, 4:2:0 only e = 2).
                for e in 1..4 {
                    let luma_edge = e % 2 == 0 || internal_odd;
                    let chroma_edge = do_chroma && (e == 2 || chroma422);
                    if !luma_edge && !chroma_edge {
                        continue;
                    }
                    let mut bs = [0u8; 4];
                    for k in 0..4 {
                        bs[k] = bs_mbaff(
                            frame,
                            info,
                            Side {
                                addr: sa,
                                blk: (e - 1) * 4 + k,
                            },
                            Side {
                                addr: sa,
                                blk: e * 4 + k,
                            },
                            false,
                            false,
                            nz_q,
                            nz_q,
                        );
                    }
                    if bs.iter().all(|&b| b == 0) {
                        continue;
                    }
                    if luma_edge {
                        let (index_a, alpha, beta) =
                            thresholds(m.qp as i32, m.qp as i32, &par, bd_shift);
                        let off = frame.y.offset(x0 as isize, (y_p + dy * e * 4) as isize);
                        if bs[0] == 4 {
                            (dsp.deblock_luma_h_intra)(
                                &mut frame.y.data,
                                off,
                                ystride * dy,
                                alpha,
                                beta,
                                max,
                            );
                        } else {
                            (dsp.deblock_luma_h)(
                                &mut frame.y.data,
                                off,
                                ystride * dy,
                                alpha,
                                beta,
                                &tc0_of(&bs, index_a, bd_shift),
                                max,
                            );
                        }
                    }
                    if chroma_edge {
                        // Chroma row of the edge: 4:2:0 only e = 2 (row 4), 4:2:2 e * 4.
                        let cy = if chroma422 { e * 4 } else { 4 };
                        for comp in 0..2 {
                            let (index_a, alpha, beta) =
                                thresholds(m.qpc[comp] as i32, m.qpc[comp] as i32, &par, bd_shift);
                            let plane = if comp == 0 {
                                &mut frame.cb
                            } else {
                                &mut frame.cr
                            };
                            let off = plane.offset(xc0 as isize, (yc_p + dy * cy) as isize);
                            if bs[0] == 4 {
                                (dsp.deblock_chroma_h_intra)(
                                    &mut plane.data,
                                    off,
                                    cstride * dy,
                                    alpha,
                                    beta,
                                    max,
                                );
                            } else {
                                (dsp.deblock_chroma_h)(
                                    &mut plane.data,
                                    off,
                                    cstride * dy,
                                    alpha,
                                    beta,
                                    &tc0_of(&bs, index_a, bd_shift),
                                    max,
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

/// A vertical chroma edge of one macroblock at chroma column `xc`, rows
/// `yc, yc + dy, ...`: its eight lines for 4:2:0; sixteen for 4:2:2 in
/// two eight-line halves, each line taking the bS of the luma line at
/// its position.
#[allow(clippy::too_many_arguments)]
fn chroma_v_edge<S: Sample>(
    dsp: &H264Dsp<S>,
    plane: &mut super::frame::PaddedPlane<S>,
    xc: usize,
    yc: usize,
    dy: usize,
    c422: bool,
    bs: &[u8; 4],
    index_a: i32,
    alpha: i32,
    beta: i32,
    bd_shift: u32,
    max: i32,
) {
    let stride = plane.stride * dy;
    let tc = tc0_of(bs, index_a, bd_shift);
    if !c422 {
        let off = plane.offset(xc as isize, yc as isize);
        if bs[0] == 4 {
            (dsp.deblock_chroma_v_intra)(&mut plane.data, off, stride, alpha, beta, max);
        } else {
            (dsp.deblock_chroma_v)(&mut plane.data, off, stride, alpha, beta, &tc, max);
        }
        return;
    }
    for half in 0..2 {
        // Chroma lines 8h..8h+7 sit against luma segments 2h and 2h + 1.
        let off = plane.offset(xc as isize, (yc + 8 * half * dy) as isize);
        let t = [
            tc[2 * half],
            tc[2 * half],
            tc[2 * half + 1],
            tc[2 * half + 1],
        ];
        if bs[2 * half] == 4 {
            (dsp.deblock_chroma_v_intra)(&mut plane.data, off, stride, alpha, beta, max);
        } else if bs[2 * half] != 0 || bs[2 * half + 1] != 0 {
            (dsp.deblock_chroma_v)(&mut plane.data, off, stride, alpha, beta, &t, max);
        }
    }
}

/// A top macroblock edge (luma and chroma) against one p macroblock, with
/// the current macroblock's row step `dy` (its lines above the edge being
/// the p macroblock's, `dy` rows apart).
#[allow(clippy::too_many_arguments)]
fn top_edge_plain<S: Sample>(
    dsp: &H264Dsp<S>,
    frame: &mut Frame<S>,
    info: &PicInfo,
    par: &DeblockParams,
    sa: usize,
    pa: usize,
    bs: &[u8; 4],
    x0: usize,
    y_p: usize,
    xc0: usize,
    yc_p: usize,
    dy: usize,
    do_chroma: bool,
    bd_shift: u32,
    max: i32,
) {
    if bs.iter().all(|&b| b == 0) {
        return;
    }
    let m = &info.mbs[sa];
    let (index_a, alpha, beta) = thresholds(info.mbs[pa].qp as i32, m.qp as i32, par, bd_shift);
    let ystride = frame.y.stride;
    let off = frame.y.offset(x0 as isize, y_p as isize);
    if bs[0] == 4 {
        (dsp.deblock_luma_h_intra)(&mut frame.y.data, off, ystride * dy, alpha, beta, max);
    } else {
        (dsp.deblock_luma_h)(
            &mut frame.y.data,
            off,
            ystride * dy,
            alpha,
            beta,
            &tc0_of(bs, index_a, bd_shift),
            max,
        );
    }
    if do_chroma {
        let cstride = frame.cb.stride;
        for comp in 0..2 {
            let (index_a, alpha, beta) = thresholds(
                info.mbs[pa].qpc[comp] as i32,
                m.qpc[comp] as i32,
                par,
                bd_shift,
            );
            let plane = if comp == 0 {
                &mut frame.cb
            } else {
                &mut frame.cr
            };
            let off = plane.offset(xc0 as isize, yc_p as isize);
            if bs[0] == 4 {
                (dsp.deblock_chroma_h_intra)(&mut plane.data, off, cstride * dy, alpha, beta, max);
            } else {
                (dsp.deblock_chroma_h)(
                    &mut plane.data,
                    off,
                    cstride * dy,
                    alpha,
                    beta,
                    &tc0_of(bs, index_a, bd_shift),
                    max,
                );
            }
        }
    }
}
