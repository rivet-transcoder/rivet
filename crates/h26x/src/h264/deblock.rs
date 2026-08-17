//! The deblocking filter (H.264 clause 8.7), run over the whole picture
//! once every slice is decoded: boundary strength per edge segment, then
//! the luma and chroma edge filters, macroblock by macroblock in raster
//! order — vertical edges first, then horizontal, luma then chroma.

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

/// Boundary strength for the edge between 4x4 blocks `p` (in MB `pa`) and
/// `q` (in MB `qa`) — `p_blk` / `q_blk` are raster 4x4 indices — for a
/// frame picture (8.7.2.1). `mb_edge` says whether the edge is a macroblock
/// edge.
#[allow(clippy::too_many_arguments)]
fn boundary_strength(
    frame: &Frame,
    info: &PicInfo,
    pa: usize,
    p_blk: usize,
    qa: usize,
    q_blk: usize,
    mb_edge: bool,
) -> u8 {
    let pm = &info.mbs[pa];
    let qm = &info.mbs[qa];
    if pm.kind.is_intra() || qm.kind.is_intra() {
        return if mb_edge { 4 } else { 3 };
    }
    // Transform coefficients in either block (the 8x8 block for 8x8 transforms).
    let has_coeffs = |addr: usize, blk: usize| -> bool {
        let m = &info.mbs[addr];
        if m.transform_8x8 {
            let bx8 = (blk % 4) / 2 * 2;
            let by8 = (blk / 4) / 2 * 2;
            let base = addr * 16;
            info.luma_nz[base + by8 * 4 + bx8] != 0
                || info.luma_nz[base + by8 * 4 + bx8 + 1] != 0
                || info.luma_nz[base + (by8 + 1) * 4 + bx8] != 0
                || info.luma_nz[base + (by8 + 1) * 4 + bx8 + 1] != 0
        } else {
            info.luma_nz[addr * 16 + blk] != 0
        }
    };
    if has_coeffs(pa, p_blk) || has_coeffs(qa, q_blk) {
        return 2;
    }
    // Motion: different reference pictures / number of vectors, or a
    // vector component differing by 4 or more quarter samples.
    let p0 = frame.motion[0][pa * 16 + p_blk];
    let p1 = frame.motion[1][pa * 16 + p_blk];
    let q0 = frame.motion[0][qa * 16 + q_blk];
    let q1 = frame.motion[1][qa * 16 + q_blk];
    let (pn, qn) = ((p0.ref_idx >= 0) as u32 + (p1.ref_idx >= 0) as u32, (q0.ref_idx >= 0) as u32 + (q1.ref_idx >= 0) as u32);
    if pn != qn {
        return 1;
    }
    let mv_far = |a: super::frame::Mv, b: super::frame::Mv| (a.x - b.x).abs() >= 4 || (a.y - b.y).abs() >= 4;
    // Reference picture identity: POC plus long-term-ness.
    let same_pic = |a: &super::frame::BlockMotion, b: &super::frame::BlockMotion| {
        a.ref_poc == b.ref_poc && a.ref_long_term == b.ref_long_term
    };
    if pn == 1 {
        let (pp, qq) = (if p0.ref_idx >= 0 { p0 } else { p1 }, if q0.ref_idx >= 0 { q0 } else { q1 });
        if !same_pic(&pp, &qq) {
            return 1;
        }
        return mv_far(pp.mv, qq.mv) as u8;
    }
    // Two vectors each.
    let straight = same_pic(&p0, &q0) && same_pic(&p1, &q1);
    let crossed = same_pic(&p0, &q1) && same_pic(&p1, &q0);
    if !straight && !crossed {
        return 1;
    }
    if same_pic(&p0, &p1) {
        // Both vectors reference the same picture: bS is 1 only if both
        // pairings differ.
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

/// Filter one line of samples across an edge (8.7.2.3 / 8.7.2.4).
/// `p[0..4]` are p0..p3 (p0 nearest the edge), `q[0..4]` are q0..q3.
#[inline]
fn filter_line(p: &mut [i32; 4], q: &mut [i32; 4], bs: u8, alpha: i32, beta: i32, tc0: i32, chroma: bool) {
    let (p0, p1, p2, p3) = (p[0], p[1], p[2], p[3]);
    let (q0, q1, q2, q3) = (q[0], q[1], q[2], q[3]);
    if !((p0 - q0).abs() < alpha && (p1 - p0).abs() < beta && (q1 - q0).abs() < beta) {
        return;
    }
    let ap = (p2 - p0).abs();
    let aq = (q2 - q0).abs();
    if bs < 4 {
        let tc = if chroma { tc0 + 1 } else { tc0 + (ap < beta) as i32 + (aq < beta) as i32 };
        let delta = clip3(-tc, tc, (((q0 - p0) << 2) + (p1 - q1) + 4) >> 3);
        p[0] = clip3(0, 255, p0 + delta);
        q[0] = clip3(0, 255, q0 - delta);
        if !chroma {
            if ap < beta {
                p[1] = p1 + clip3(-tc0, tc0, (p2 + ((p0 + q0 + 1) >> 1) - (p1 << 1)) >> 1);
            }
            if aq < beta {
                q[1] = q1 + clip3(-tc0, tc0, (q2 + ((p0 + q0 + 1) >> 1) - (q1 << 1)) >> 1);
            }
        }
    } else {
        let strong = (p0 - q0).abs() < ((alpha >> 2) + 2);
        if !chroma && ap < beta && strong {
            p[0] = (p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) >> 3;
            p[1] = (p2 + p1 + p0 + q0 + 2) >> 2;
            p[2] = (2 * p3 + 3 * p2 + p1 + p0 + q0 + 4) >> 3;
        } else {
            p[0] = (2 * p1 + p0 + q1 + 2) >> 2;
        }
        if !chroma && aq < beta && strong {
            q[0] = (p1 + 2 * p0 + 2 * q0 + 2 * q1 + q2 + 4) >> 3;
            q[1] = (p0 + q0 + q1 + q2 + 2) >> 2;
            q[2] = (2 * q3 + 3 * q2 + q1 + q0 + p0 + 4) >> 3;
        } else {
            q[0] = (2 * q1 + q0 + p1 + 2) >> 2;
        }
    }
}

/// Filter `len` lines of an edge in a plane. `off` is the offset of the
/// first q0 sample; `step` moves along the edge (stride for a vertical
/// edge, 1 for a horizontal edge), `across` moves across it (1 / stride).
#[allow(clippy::too_many_arguments)]
fn filter_edge(
    data: &mut [u8],
    off: usize,
    step: usize,
    across: usize,
    len: usize,
    bs: &[u8],
    bs_per_line: usize,
    index_a: i32,
    index_b: i32,
    chroma: bool,
) {
    let alpha = ALPHA[index_a as usize] as i32;
    let beta = BETA[index_b as usize] as i32;
    for i in 0..len {
        let s = bs[i / bs_per_line];
        if s == 0 {
            continue;
        }
        let tc0 = TC0[index_a as usize][(s - 1).min(2) as usize] as i32;
        let base = off + i * step;
        let mut p = [0i32; 4];
        let mut q = [0i32; 4];
        let depth = if chroma { 2 } else { 4 };
        for k in 0..depth {
            p[k] = data[base - (k + 1) * across] as i32;
            q[k] = data[base + k * across] as i32;
        }
        filter_line(&mut p, &mut q, s, alpha, beta, tc0, chroma);
        let write = if chroma { 1 } else { 3 };
        for k in 0..write {
            data[base - (k + 1) * across] = p[k] as u8;
            data[base + k * across] = q[k] as u8;
        }
    }
}

/// Deblock the whole picture in place.
pub fn deblock_picture(frame: &mut Frame, info: &PicInfo, params: &[DeblockParams]) {
    let mbw = info.mb_width;
    let mbh = info.mb_height;
    for mby in 0..mbh {
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
            for e in 0..4 {
                for k in 0..4 {
                    // Vertical edge e: p block (col e-1, row k), q block (col e, row k).
                    if e == 0 {
                        if filter_left {
                            let l = left.unwrap();
                            bs_v[0][k] = boundary_strength(frame, info, l, k * 4 + 3, addr, k * 4, true);
                        }
                    } else if !(m.transform_8x8 && e % 2 == 1) {
                        bs_v[e][k] = boundary_strength(frame, info, addr, k * 4 + e - 1, addr, k * 4 + e, false);
                    }
                    // Horizontal edge e: p block (col k, row e-1), q (col k, row e).
                    if e == 0 {
                        if filter_top {
                            let a = above.unwrap();
                            bs_h[0][k] = boundary_strength(frame, info, a, 12 + k, addr, k, true);
                        }
                    } else if !(m.transform_8x8 && e % 2 == 1) {
                        bs_h[e][k] = boundary_strength(frame, info, addr, (e - 1) * 4 + k, addr, e * 4 + k, false);
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
                let off = frame.y.offset((x0 + e * 4) as isize, y0 as isize);
                let stride = frame.y.stride;
                filter_edge(&mut frame.y.data, off, stride, 1, 16, &bs_v[e], 4, index_a, index_b, false);
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
                let off = frame.y.offset(x0 as isize, (y0 + e * 4) as isize);
                let stride = frame.y.stride;
                filter_edge(&mut frame.y.data, off, 1, stride, 16, &bs_h[e], 4, index_a, index_b, false);
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
                        let plane = if comp == 0 { &mut frame.cb } else { &mut frame.cr };
                        let off = plane.offset((mbx * 8 + e * 2) as isize, (mby * 8) as isize);
                        let stride = plane.stride;
                        // 8 chroma rows; row r uses luma bS index (2r)/4 = r/2.
                        filter_edge(&mut plane.data, off, stride, 1, 8, &bs_v[e], 2, index_a, index_b, true);
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
                        let plane = if comp == 0 { &mut frame.cb } else { &mut frame.cr };
                        let off = plane.offset((mbx * 8) as isize, (mby * 8 + e * 2) as isize);
                        let stride = plane.stride;
                        filter_edge(&mut plane.data, off, 1, stride, 8, &bs_h[e], 2, index_a, index_b, true);
                    }
                }
            }
        }
    }
    let _ = MbKind::PSkip;
}
