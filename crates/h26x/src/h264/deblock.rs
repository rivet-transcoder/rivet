//! The deblocking filter (H.264 clause 8.7), run over the whole picture
//! once every slice is decoded: boundary strength per edge segment, then
//! the luma and chroma edge filters, macroblock by macroblock in raster
//! order — vertical edges first, then horizontal, luma then chroma.

use crate::sample::Sample;
use crate::dsp::h264::H264Dsp;

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
/// blocks of inter macroblocks with no coefficients.
#[inline]
fn motion_bs<S: Sample>(frame: &Frame<S>, pa: usize, p_blk: usize, qa: usize, q_blk: usize) -> u8 {
    use super::frame::{BlockMotion, Mv};
    let p0 = &frame.motion[0][pa * 16 + p_blk];
    let p1 = &frame.motion[1][pa * 16 + p_blk];
    let q0 = &frame.motion[0][qa * 16 + q_blk];
    let q1 = &frame.motion[1][qa * 16 + q_blk];
    let (pn, qn) = ((p0.ref_idx >= 0) as u32 + (p1.ref_idx >= 0) as u32, (q0.ref_idx >= 0) as u32 + (q1.ref_idx >= 0) as u32);
    if pn != qn {
        return 1;
    }
    #[inline(always)]
    fn mv_far(a: Mv, b: Mv) -> bool {
        (a.x - b.x).abs() >= 4 || (a.y - b.y).abs() >= 4
    }
    #[inline(always)]
    fn same_pic(a: &BlockMotion, b: &BlockMotion) -> bool {
        a.ref_poc == b.ref_poc && a.ref_long_term == b.ref_long_term
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
pub fn deblock_picture<S: Sample>(dsp: &H264Dsp<S>, frame: &mut Frame<S>, info: &PicInfo, params: &[DeblockParams]) {
    deblock_mb_rows(dsp, frame, info, params, 0, info.mb_height);
}

/// Deblock macroblock rows `r0..r1` in raster order (each row's top edges
/// reach three lines into the row above). Rows must be filtered in order,
/// and a row only after the row below it is decoded (intra prediction
/// reads unfiltered neighbours), which is how the picture-level filter
/// order is preserved when rows are filtered as decoding proceeds.
pub fn deblock_mb_rows<S: Sample>(dsp: &H264Dsp<S>, frame: &mut Frame<S>, info: &PicInfo, params: &[DeblockParams], r0: usize, r1: usize) {
    let mbw = info.mb_width;
    let mbh = info.mb_height;
    // Thresholds scale with the bit depth (8.7.2.2): alpha, beta and tC0 are
    // multiplied by 2^(BitDepth − 8).
    let bd_shift = frame.bit_depth - 8;
    let max = (1i32 << frame.bit_depth) - 1;
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
            let filter_left = left.is_some_and(|l| info.mbs[l].decoded && (across_slices || info.mbs[l].slice == m.slice));
            let filter_top = above.is_some_and(|a| info.mbs[a].decoded && (across_slices || info.mbs[a].slice == m.slice));

            // Boundary strengths for the vertical edges (4 edges x 4 rows)
            // and horizontal edges (4 edges x 4 columns).
            let mut bs_v = [[0u8; 4]; 4];
            let mut bs_h = [[0u8; 4]; 4];
            let internal_odd = !m.transform_8x8;
            if m.kind.is_intra() {
                for e in 0..4 {
                    let v = if e == 0 { 4 } else { 3 };
                    if e == 0 || e == 2 || internal_odd {
                        if e != 0 || filter_left {
                            bs_v[e] = [v; 4];
                        }
                        if e != 0 || filter_top {
                            bs_h[e] = [v; 4];
                        }
                    }
                }
            } else {
                let nz = nz_mask(info, addr);
                // Internal edges: coefficients, else motion.
                for e in 1..4 {
                    if e % 2 == 1 && !internal_odd {
                        continue;
                    }
                    for k in 0..4 {
                        let (pb, qb) = (k * 4 + e - 1, k * 4 + e);
                        bs_v[e][k] = if (nz >> pb | nz >> qb) & 1 != 0 { 2 } else { motion_bs(frame, addr, pb, addr, qb) };
                        let (pb, qb) = ((e - 1) * 4 + k, e * 4 + k);
                        bs_h[e][k] = if (nz >> pb | nz >> qb) & 1 != 0 { 2 } else { motion_bs(frame, addr, pb, addr, qb) };
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
                            bs_v[0][k] = if (nzl >> pb | nz >> qb) & 1 != 0 { 2 } else { motion_bs(frame, l, pb, addr, qb) };
                        }
                    }
                }
                if filter_top {
                    let a = above.unwrap();
                    if info.mbs[a].kind.is_intra() {
                        bs_h[0] = [4; 4];
                    } else {
                        let nza = nz_mask(info, a);
                        for k in 0..4 {
                            let (pb, qb) = (12 + k, k);
                            bs_h[0][k] = if (nza >> pb | nz >> qb) & 1 != 0 { 2 } else { motion_bs(frame, a, pb, addr, qb) };
                        }
                    }
                }
            }

            let qp_cur = m.qp as i32;
            let (x0, y0) = (mbx * 16, mby * 16);
            // Luma vertical edges.
            for e in 0..4 {
                if bs_v[e].iter().all(|&b| b == 0) {
                    continue;
                }
                let qp_p = if e == 0 { info.mbs[left.unwrap()].qp as i32 } else { qp_cur };
                let qp_av = (qp_p + qp_cur + 1) >> 1;
                let index_a = clip3(0, 51, qp_av + par.offset_a);
                let index_b = clip3(0, 51, qp_av + par.offset_b);
                let (alpha, beta) = ((ALPHA[index_a as usize] as i32) << bd_shift, (BETA[index_b as usize] as i32) << bd_shift);
                let off = frame.y.offset((x0 + e * 4) as isize, y0 as isize);
                let stride = frame.y.stride;
                if bs_v[e][0] == 4 {
                    (dsp.deblock_luma_v_intra)(&mut frame.y.data, off, stride, alpha, beta, max);
                } else {
                    (dsp.deblock_luma_v)(&mut frame.y.data, off, stride, alpha, beta, &tc0_of(&bs_v[e], index_a, bd_shift), max);
                }
            }
            // Luma horizontal edges.
            for e in 0..4 {
                if bs_h[e].iter().all(|&b| b == 0) {
                    continue;
                }
                let qp_p = if e == 0 { info.mbs[above.unwrap()].qp as i32 } else { qp_cur };
                let qp_av = (qp_p + qp_cur + 1) >> 1;
                let index_a = clip3(0, 51, qp_av + par.offset_a);
                let index_b = clip3(0, 51, qp_av + par.offset_b);
                let (alpha, beta) = ((ALPHA[index_a as usize] as i32) << bd_shift, (BETA[index_b as usize] as i32) << bd_shift);
                let off = frame.y.offset(x0 as isize, (y0 + e * 4) as isize);
                let stride = frame.y.stride;
                if bs_h[e][0] == 4 {
                    (dsp.deblock_luma_h_intra)(&mut frame.y.data, off, stride, alpha, beta, max);
                } else {
                    (dsp.deblock_luma_h)(&mut frame.y.data, off, stride, alpha, beta, &tc0_of(&bs_h[e], index_a, bd_shift), max);
                }
            }
            // Chroma (4:2:0): edges 0 and 2 (in luma 4x4 units) — chroma
            // sample positions 0 and 4; bS of chroma line k comes from luma
            // bS at row 2k (segment (2k)/4).
            if frame.chroma == crate::picture::ChromaFormat::Yuv420 {
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
                        let (alpha, beta) = ((ALPHA[index_a as usize] as i32) << bd_shift, (BETA[index_b as usize] as i32) << bd_shift);
                        let plane = if comp == 0 { &mut frame.cb } else { &mut frame.cr };
                        let off = plane.offset((mbx * 8 + e * 2) as isize, (mby * 8) as isize);
                        let stride = plane.stride;
                        // 8 chroma rows; row r uses luma bS index (2r)/4 = r/2.
                        if bs_v[e][0] == 4 {
                            (dsp.deblock_chroma_v_intra)(&mut plane.data, off, stride, alpha, beta, max);
                        } else {
                            (dsp.deblock_chroma_v)(&mut plane.data, off, stride, alpha, beta, &tc0_of(&bs_v[e], index_a, bd_shift), max);
                        }
                    }
                    for &e in &[0usize, 2] {
                        if bs_h[e].iter().all(|&b| b == 0) {
                            continue;
                        }
                        let (qpc_p, qpc_q) = if e == 0 {
                            (info.mbs[above.unwrap()].qpc[comp] as i32, m.qpc[comp] as i32)
                        } else {
                            (m.qpc[comp] as i32, m.qpc[comp] as i32)
                        };
                        let qp_av = (qpc_p + qpc_q + 1) >> 1;
                        let index_a = clip3(0, 51, qp_av + par.offset_a);
                        let index_b = clip3(0, 51, qp_av + par.offset_b);
                        let (alpha, beta) = ((ALPHA[index_a as usize] as i32) << bd_shift, (BETA[index_b as usize] as i32) << bd_shift);
                        let plane = if comp == 0 { &mut frame.cb } else { &mut frame.cr };
                        let off = plane.offset((mbx * 8) as isize, (mby * 8 + e * 2) as isize);
                        let stride = plane.stride;
                        if bs_h[e][0] == 4 {
                            (dsp.deblock_chroma_h_intra)(&mut plane.data, off, stride, alpha, beta, max);
                        } else {
                            (dsp.deblock_chroma_h)(&mut plane.data, off, stride, alpha, beta, &tc0_of(&bs_h[e], index_a, bd_shift), max);
                        }
                    }
                }
            }
        }
    }
    let _ = MbKind::PSkip;
}
