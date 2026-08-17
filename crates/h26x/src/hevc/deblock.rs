//! Deblocking filter (H.265 8.7.2): boundary strengths from the per-4x4
//! side data, then all vertical edges of the picture followed by all
//! horizontal edges, luma and chroma (4:2:0).

use super::ctu::chroma_qp_420;
use super::frame::{Frame, MotionInfo, Plane16};
use super::pic::PicInfo;
use super::pps::Pps;
use super::tables_gen::{BETA_TABLE, TC_TABLE};

/// Boundary strength from the motion of the two sides (8.7.2.4, the
/// inter-vs-inter cases).
fn motion_bs(a: &MotionInfo, b: &MotionInfo) -> u8 {
    let na = a.uses(0) as u8 + a.uses(1) as u8;
    let nb = b.uses(0) as u8 + b.uses(1) as u8;
    if na != nb {
        return 1;
    }
    let far = |p: super::frame::Mv, q: super::frame::Mv| -> bool { (p.x as i32 - q.x as i32).abs() >= 4 || (p.y as i32 - q.y as i32).abs() >= 4 };
    if na == 1 {
        let la = if a.uses(0) { 0 } else { 1 };
        let lb = if b.uses(0) { 0 } else { 1 };
        if a.ref_poc[la] != b.ref_poc[lb] || a.ref_long_term[la] != b.ref_long_term[lb] {
            return 1;
        }
        return far(a.mv[la], b.mv[lb]) as u8;
    }
    // Two vectors each.
    let (pa0, pa1) = ((a.ref_poc[0], a.ref_long_term[0]), (a.ref_poc[1], a.ref_long_term[1]));
    let (pb0, pb1) = ((b.ref_poc[0], b.ref_long_term[0]), (b.ref_poc[1], b.ref_long_term[1]));
    let same_set = (pa0 == pb0 && pa1 == pb1) || (pa0 == pb1 && pa1 == pb0);
    if !same_set {
        return 1;
    }
    if pa0 != pa1 {
        // Two different pictures: match by picture.
        if pa0 == pb0 {
            (far(a.mv[0], b.mv[0]) || far(a.mv[1], b.mv[1])) as u8
        } else {
            (far(a.mv[0], b.mv[1]) || far(a.mv[1], b.mv[0])) as u8
        }
    } else {
        // The same picture twice on both sides.
        let c1 = far(a.mv[0], b.mv[0]) || far(a.mv[1], b.mv[1]);
        let c2 = far(a.mv[0], b.mv[1]) || far(a.mv[1], b.mv[0]);
        (c1 && c2) as u8
    }
}

/// Compute the boundary strengths of the vertical (`ver`) and horizontal
/// (`hor`) edges at 4x4 granularity (index = 4x4 block whose left / top
/// side the edge is; only the 8x8 luma grid gets nonzero values).
fn boundary_strengths(frame: &Frame, info: &PicInfo, pps: &Pps) -> (Vec<u8>, Vec<u8>) {
    let (w4, h4) = (info.w4, info.h4);
    let mut ver = vec![0u8; w4 * h4];
    let mut hor = vec![0u8; w4 * h4];
    let ctb_mask = (1usize << info.log2_ctb) - 1;
    for by in 0..h4 {
        for bx in 0..w4 {
            let (x, y) = (bx * 4, by * 4);
            let q = by * w4 + bx;
            let sl_idx = info.ctb_slice[info.ctb_of(x, y)];
            if sl_idx == u16::MAX {
                continue;
            }
            let sl = &info.slices[sl_idx as usize];
            if sl.deblocking_disabled {
                continue;
            }
            let edges = info.edges[q];
            let mq = &frame.motion[q];
            let intra_q = info.pred_mode[q] == 1;
            // Vertical edge on the left side of this block.
            if x > 0 && x % 8 == 0 && (edges & 3) != 0 {
                let p = q - 1;
                let mut ok = true;
                if x & ctb_mask == 0 {
                    let cq = info.ctb_of(x, y);
                    let cp = info.ctb_of(x - 1, y);
                    if info.ctb_tile[cq] != info.ctb_tile[cp] && !pps.loop_filter_across_tiles {
                        ok = false;
                    }
                    if info.ctb_slice_addr[cq] != info.ctb_slice_addr[cp] && !sl.loop_filter_across_slices {
                        ok = false;
                    }
                }
                if ok {
                    ver[q] = if intra_q || info.pred_mode[p] == 1 {
                        2
                    } else if (edges & 1) != 0 && (info.cbf_luma[p] != 0 || info.cbf_luma[q] != 0) {
                        1
                    } else {
                        motion_bs(&frame.motion[p], mq)
                    };
                }
            }
            // Horizontal edge on the top side.
            if y > 0 && y % 8 == 0 && (edges & 12) != 0 {
                let p = q - w4;
                let mut ok = true;
                if y & ctb_mask == 0 {
                    let cq = info.ctb_of(x, y);
                    let cp = info.ctb_of(x, y - 1);
                    if info.ctb_tile[cq] != info.ctb_tile[cp] && !pps.loop_filter_across_tiles {
                        ok = false;
                    }
                    if info.ctb_slice_addr[cq] != info.ctb_slice_addr[cp] && !sl.loop_filter_across_slices {
                        ok = false;
                    }
                }
                if ok {
                    hor[q] = if intra_q || info.pred_mode[p] == 1 {
                        2
                    } else if (edges & 4) != 0 && (info.cbf_luma[p] != 0 || info.cbf_luma[q] != 0) {
                        1
                    } else {
                        motion_bs(&frame.motion[p], mq)
                    };
                }
            }
        }
    }
    (ver, hor)
}

/// Filter one 4-line luma edge segment. `pos` is the offset of q0 of the
/// first line, `step` the distance across the edge (1 for vertical edges,
/// stride for horizontal), `along` the distance between lines.
#[allow(clippy::too_many_arguments)]
fn luma_edge(d: &mut [u16], pos: usize, step: usize, along: usize, beta: i32, tc: i32, no_p: bool, no_q: bool, max: i32) {
    let s = |d: &[u16], line: usize, k: isize| -> i32 { d[(pos as isize + (line * along) as isize + k * step as isize) as usize] as i32 };
    let dp0 = (s(d, 0, -3) - 2 * s(d, 0, -2) + s(d, 0, -1)).abs();
    let dp3 = (s(d, 3, -3) - 2 * s(d, 3, -2) + s(d, 3, -1)).abs();
    let dq0 = (s(d, 0, 2) - 2 * s(d, 0, 1) + s(d, 0, 0)).abs();
    let dq3 = (s(d, 3, 2) - 2 * s(d, 3, 1) + s(d, 3, 0)).abs();
    let dpq0 = dp0 + dq0;
    let dpq3 = dp3 + dq3;
    let dp = dp0 + dp3;
    let dq = dq0 + dq3;
    let dd = dpq0 + dpq3;
    if dd >= beta {
        return;
    }
    let dsam = |d: &[u16], line: usize, dpq: i32| -> bool {
        dpq < (beta >> 2)
            && (s(d, line, -4) - s(d, line, -1)).abs() + (s(d, line, 0) - s(d, line, 3)).abs() < (beta >> 3)
            && (s(d, line, -1) - s(d, line, 0)).abs() < ((5 * tc + 1) >> 1)
    };
    let strong = dsam(d, 0, 2 * dpq0) && dsam(d, 3, 2 * dpq3);
    let dep = dp < ((beta + (beta >> 1)) >> 3);
    let deq = dq < ((beta + (beta >> 1)) >> 3);
    for line in 0..4 {
        let base = pos + line * along;
        let at = |k: isize| -> usize { (base as isize + k * step as isize) as usize };
        let p0 = d[at(-1)] as i32;
        let p1 = d[at(-2)] as i32;
        let p2 = d[at(-3)] as i32;
        let p3 = d[at(-4)] as i32;
        let q0 = d[at(0)] as i32;
        let q1 = d[at(1)] as i32;
        let q2 = d[at(2)] as i32;
        let q3 = d[at(3)] as i32;
        if strong {
            if !no_p {
                d[at(-1)] = ((p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) >> 3).clamp(p0 - 2 * tc, p0 + 2 * tc) as u16;
                d[at(-2)] = ((p2 + p1 + p0 + q0 + 2) >> 2).clamp(p1 - 2 * tc, p1 + 2 * tc) as u16;
                d[at(-3)] = ((2 * p3 + 3 * p2 + p1 + p0 + q0 + 4) >> 3).clamp(p2 - 2 * tc, p2 + 2 * tc) as u16;
            }
            if !no_q {
                d[at(0)] = ((p1 + 2 * p0 + 2 * q0 + 2 * q1 + q2 + 4) >> 3).clamp(q0 - 2 * tc, q0 + 2 * tc) as u16;
                d[at(1)] = ((p0 + q0 + q1 + q2 + 2) >> 2).clamp(q1 - 2 * tc, q1 + 2 * tc) as u16;
                d[at(2)] = ((p0 + q0 + q1 + 3 * q2 + 2 * q3 + 4) >> 3).clamp(q2 - 2 * tc, q2 + 2 * tc) as u16;
            }
        } else {
            let mut delta = (9 * (q0 - p0) - 3 * (q1 - p1) + 8) >> 4;
            if delta.abs() < tc * 10 {
                delta = delta.clamp(-tc, tc);
                if !no_p {
                    d[at(-1)] = (p0 + delta).clamp(0, max) as u16;
                }
                if !no_q {
                    d[at(0)] = (q0 - delta).clamp(0, max) as u16;
                }
                if dep && !no_p {
                    let dp = ((((p2 + p0 + 1) >> 1) - p1 + delta) >> 1).clamp(-(tc >> 1), tc >> 1);
                    d[at(-2)] = (p1 + dp).clamp(0, max) as u16;
                }
                if deq && !no_q {
                    let dq = ((((q2 + q0 + 1) >> 1) - q1 - delta) >> 1).clamp(-(tc >> 1), tc >> 1);
                    d[at(1)] = (q1 + dq).clamp(0, max) as u16;
                }
            }
        }
    }
}

/// Filter `n` lines of a chroma edge.
#[allow(clippy::too_many_arguments)]
fn chroma_edge(d: &mut [u16], pos: usize, step: usize, along: usize, n: usize, tc: i32, no_p: bool, no_q: bool, max: i32) {
    for line in 0..n {
        let base = pos + line * along;
        let at = |k: isize| -> usize { (base as isize + k * step as isize) as usize };
        let p0 = d[at(-1)] as i32;
        let p1 = d[at(-2)] as i32;
        let q0 = d[at(0)] as i32;
        let q1 = d[at(1)] as i32;
        let delta = ((((q0 - p0) << 2) + p1 - q1 + 4) >> 3).clamp(-tc, tc);
        if !no_p {
            d[at(-1)] = (p0 + delta).clamp(0, max) as u16;
        }
        if !no_q {
            d[at(0)] = (q0 - delta).clamp(0, max) as u16;
        }
    }
}

/// Deblock the whole picture in place.
pub fn deblock_picture(frame: &mut Frame, info: &PicInfo, pps: &Pps, bit_depth_luma: u32, bit_depth_chroma: u32) {
    let (bs_ver, bs_hor) = boundary_strengths(frame, info, pps);
    let (w4, h4) = (info.w4, info.h4);
    let max_l = (1i32 << bit_depth_luma) - 1;
    let max_c = (1i32 << bit_depth_chroma) - 1;
    let sh_l = bit_depth_luma as i32 - 8;
    let sh_c = bit_depth_chroma as i32 - 8;
    let has_chroma = frame.chroma != crate::picture::ChromaFormat::Monochrome;

    for pass in 0..2 {
        let bs = if pass == 0 { &bs_ver } else { &bs_hor };
        // Luma.
        {
            let stride = frame.y.stride;
            let (step, along) = if pass == 0 { (1, stride) } else { (stride, 1) };
            for by in 0..h4 {
                for bx in 0..w4 {
                    let b = bs[by * w4 + bx];
                    if b == 0 {
                        continue;
                    }
                    let (x, y) = (bx * 4, by * 4);
                    let q = by * w4 + bx;
                    let p = if pass == 0 { q - 1 } else { q - w4 };
                    let sl = &info.slices[info.ctb_slice[info.ctb_of(x, y)] as usize];
                    let qp = (info.qp_y[p] as i32 + info.qp_y[q] as i32 + 1) >> 1;
                    let beta = BETA_TABLE[(qp + sl.beta_offset).clamp(0, 51) as usize] as i32 * (1 << sh_l);
                    let tc = TC_TABLE[(qp + 2 * (b as i32 - 1) + sl.tc_offset).clamp(0, 53) as usize] as i32 * (1 << sh_l);
                    let no_p = info.filter_exempt[p] & 1 != 0;
                    let no_q = info.filter_exempt[q] & 1 != 0;
                    let pos = frame.y.offset(x as isize, y as isize);
                    luma_edge(&mut frame.y.data, pos, step, along, beta, tc, no_p, no_q, max_l);
                }
            }
        }
        // Chroma (4:2:0): bS == 2 edges on the 8x8 chroma grid.
        if has_chroma {
            let stride = frame.cb.stride;
            let (step, along) = if pass == 0 { (1, stride) } else { (stride, 1) };
            for by in 0..h4 {
                for bx in 0..w4 {
                    let q = by * w4 + bx;
                    if bs[q] != 2 {
                        continue;
                    }
                    let (x, y) = (bx * 4, by * 4);
                    if (pass == 0 && x % 16 != 0) || (pass == 1 && y % 16 != 0) {
                        continue;
                    }
                    let p = if pass == 0 { q - 1 } else { q - w4 };
                    let sl = &info.slices[info.ctb_slice[info.ctb_of(x, y)] as usize];
                    let qp_avg = (info.qp_y[p] as i32 + info.qp_y[q] as i32 + 1) >> 1;
                    let no_p = info.filter_exempt[p] & 1 != 0;
                    let no_q = info.filter_exempt[q] & 1 != 0;
                    for (c, plane) in [(0usize, &mut frame.cb), (1, &mut frame.cr)] {
                        let off = if c == 0 { sl.cb_qp_offset } else { sl.cr_qp_offset };
                        let qpi = qp_avg + off;
                        let qpc = if qpi < 0 { qpi } else { chroma_qp_420(qpi) };
                        let tc = TC_TABLE[(qpc + 2 + sl.tc_offset).clamp(0, 53) as usize] as i32 * (1 << sh_c);
                        let pos = plane.offset((x / 2) as isize, (y / 2) as isize);
                        chroma_edge(&mut plane.data, pos, step, along, 2, tc, no_p, no_q, max_c);
                    }
                }
            }
        }
    }
}

// Silence the unused-import lint on builds without chroma paths.
#[allow(dead_code)]
fn _plane_type_check(_p: &Plane16) {}
