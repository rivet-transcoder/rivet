//! Fractional sample interpolation (H.265 8.5.3.3.3) and weighted sample
//! prediction (8.5.3.3.4).

use super::frame::{Frame, Mv, Plane16};
use super::tables::{EPEL_FILTERS, QPEL_FILTERS};

/// The two prediction arrays of a block, at 14-bit intermediate precision.
pub struct PredBlock {
    /// Samples (`w * h`).
    pub data: Vec<i32>,
}

/// Fetch a window with clamping to the picture (the border covers the
/// common case; the slow path handles vectors far outside).
#[inline]
fn fetch(p: &Plane16, x0: i32, y0: i32, w: usize, h: usize, out: &mut [i32]) {
    let pad = p.pad as i32;
    let (pw, ph) = (p.width as i32, p.height as i32);
    if x0 >= -pad && y0 >= -pad && x0 + w as i32 <= pw + pad && y0 + h as i32 <= ph + pad {
        for y in 0..h {
            let off = p.offset(x0 as isize, y0 as isize + y as isize);
            for x in 0..w {
                out[y * w + x] = p.data[off + x] as i32;
            }
        }
    } else {
        for y in 0..h {
            for x in 0..w {
                out[y * w + x] = p.at_clamped(x0 + x as i32, y0 + y as i32) as i32;
            }
        }
    }
}

/// Luma interpolation of a `w x h` block at `(x, y)` with vector `mv`,
/// producing 14-bit-precision samples in `out` (8.5.3.3.3.1).
pub fn luma_mc(reference: &Plane16, x: i32, y: i32, mv: Mv, w: usize, h: usize, bit_depth: u32, out: &mut [i32]) {
    let xi = x + (mv.x as i32 >> 2);
    let yi = y + (mv.y as i32 >> 2);
    let xf = (mv.x & 3) as usize;
    let yf = (mv.y & 3) as usize;
    let shift1 = bit_depth.min(12) as i32 - 8; // Min(4, BitDepth - 8)
    let shift3 = 14 - bit_depth as i32;
    let ww = w + 7;
    let hh = h + 7;
    let mut src = vec![0i32; ww * hh];
    fetch(reference, xi - 3, yi - 3, ww, hh, &mut src);
    let s = |xx: usize, yy: usize| src[(yy + 3) * ww + (xx + 3)];
    if xf == 0 && yf == 0 {
        for yy in 0..h {
            for xx in 0..w {
                out[yy * w + xx] = s(xx, yy) << shift3;
            }
        }
        return;
    }
    let fh = &QPEL_FILTERS[xf][..8];
    let fv = &QPEL_FILTERS[yf][..8];
    if yf == 0 {
        for yy in 0..h {
            for xx in 0..w {
                let mut acc = 0i32;
                for k in 0..8 {
                    acc += fh[k] as i32 * src[(yy + 3) * ww + xx + k];
                }
                out[yy * w + xx] = acc >> shift1;
            }
        }
        return;
    }
    if xf == 0 {
        for yy in 0..h {
            for xx in 0..w {
                let mut acc = 0i32;
                for k in 0..8 {
                    acc += fv[k] as i32 * src[(yy + k) * ww + xx + 3];
                }
                out[yy * w + xx] = acc >> shift1;
            }
        }
        return;
    }
    // Both: horizontal into a temp of (h + 7) rows, then vertical >> 6.
    let mut tmp = vec![0i32; w * hh];
    for yy in 0..hh {
        for xx in 0..w {
            let mut acc = 0i32;
            for k in 0..8 {
                acc += fh[k] as i32 * src[yy * ww + xx + k];
            }
            tmp[yy * w + xx] = acc >> shift1;
        }
    }
    for yy in 0..h {
        for xx in 0..w {
            let mut acc = 0i32;
            for k in 0..8 {
                acc += fv[k] as i32 * tmp[(yy + k) * w + xx];
            }
            out[yy * w + xx] = acc >> 6;
        }
    }
}

/// Chroma interpolation (8.5.3.3.3.2) for 4:2:0: block `w x h` at chroma
/// position `(xc, yc)` with the luma vector `mv` (eighth-sample chroma units).
#[allow(clippy::too_many_arguments)]
pub fn chroma_mc(reference: &Plane16, xc: i32, yc: i32, mv: Mv, w: usize, h: usize, bit_depth: u32, out: &mut [i32]) {
    let xi = xc + (mv.x as i32 >> 3);
    let yi = yc + (mv.y as i32 >> 3);
    let xf = (mv.x & 7) as usize;
    let yf = (mv.y & 7) as usize;
    let shift1 = bit_depth.min(12) as i32 - 8;
    let shift3 = 14 - bit_depth as i32;
    let ww = w + 3;
    let hh = h + 3;
    let mut src = vec![0i32; ww * hh];
    fetch(reference, xi - 1, yi - 1, ww, hh, &mut src);
    if xf == 0 && yf == 0 {
        for yy in 0..h {
            for xx in 0..w {
                out[yy * w + xx] = src[(yy + 1) * ww + xx + 1] << shift3;
            }
        }
        return;
    }
    // Row index is the fraction itself (row 0 is the unused full-sample slot).
    let fh = &EPEL_FILTERS[xf];
    let fv = &EPEL_FILTERS[yf];
    if yf == 0 {
        for yy in 0..h {
            for xx in 0..w {
                let mut acc = 0i32;
                for k in 0..4 {
                    acc += fh[k] as i32 * src[(yy + 1) * ww + xx + k];
                }
                out[yy * w + xx] = acc >> shift1;
            }
        }
        return;
    }
    if xf == 0 {
        for yy in 0..h {
            for xx in 0..w {
                let mut acc = 0i32;
                for k in 0..4 {
                    acc += fv[k] as i32 * src[(yy + k) * ww + xx + 1];
                }
                out[yy * w + xx] = acc >> shift1;
            }
        }
        return;
    }
    let mut tmp = vec![0i32; w * hh];
    for yy in 0..hh {
        for xx in 0..w {
            let mut acc = 0i32;
            for k in 0..4 {
                acc += fh[k] as i32 * src[yy * ww + xx + k];
            }
            tmp[yy * w + xx] = acc >> shift1;
        }
    }
    for yy in 0..h {
        for xx in 0..w {
            let mut acc = 0i32;
            for k in 0..4 {
                acc += fv[k] as i32 * tmp[(yy + k) * w + xx];
            }
            out[yy * w + xx] = acc >> 6;
        }
    }
}

/// How to combine the predictions of a block.
#[derive(Debug, Clone, Copy)]
pub enum Weighting {
    /// Default (8.5.3.3.4.2).
    Default,
    /// Explicit: `log2_wd` (already including shift1), weights and offsets
    /// per list, for one component.
    Explicit {
        /// `log2WD`.
        log2_wd: i32,
        /// `w0, w1`.
        w: [i32; 2],
        /// `o0, o1` (in sample units).
        o: [i32; 2],
    },
}

/// Combine `p0` / `p1` into `dst` (a `w x h` block at `off` in a plane).
#[allow(clippy::too_many_arguments)]
pub fn combine(
    p0: Option<&[i32]>,
    p1: Option<&[i32]>,
    weighting: Weighting,
    bit_depth: u32,
    dst: &mut [u16],
    dst_stride: usize,
    w: usize,
    h: usize,
) {
    let max = (1i32 << bit_depth) - 1;
    match (p0, p1, weighting) {
        (Some(a), None, Weighting::Default) | (None, Some(a), Weighting::Default) => {
            let shift1 = 14 - bit_depth as i32;
            let offset1 = if shift1 > 0 { 1 << (shift1 - 1) } else { 0 };
            for y in 0..h {
                for x in 0..w {
                    dst[y * dst_stride + x] = ((a[y * w + x] + offset1) >> shift1).clamp(0, max) as u16;
                }
            }
        }
        (Some(a), Some(b), Weighting::Default) => {
            let shift2 = 15 - bit_depth as i32;
            let offset2 = 1 << (shift2 - 1);
            for y in 0..h {
                for x in 0..w {
                    dst[y * dst_stride + x] = ((a[y * w + x] + b[y * w + x] + offset2) >> shift2).clamp(0, max) as u16;
                }
            }
        }
        (Some(a), None, Weighting::Explicit { log2_wd, w: wt, o }) => {
            uni_weighted(a, log2_wd, wt[0], o[0], max, dst, dst_stride, w, h);
        }
        (None, Some(b), Weighting::Explicit { log2_wd, w: wt, o }) => {
            uni_weighted(b, log2_wd, wt[1], o[1], max, dst, dst_stride, w, h);
        }
        (Some(a), Some(b), Weighting::Explicit { log2_wd, w: wt, o }) => {
            let round = (o[0] + o[1] + 1) << log2_wd;
            for y in 0..h {
                for x in 0..w {
                    let v = (a[y * w + x] * wt[0] + b[y * w + x] * wt[1] + round) >> (log2_wd + 1);
                    dst[y * dst_stride + x] = v.clamp(0, max) as u16;
                }
            }
        }
        (None, None, _) => {}
    }
}

#[allow(clippy::too_many_arguments)]
fn uni_weighted(a: &[i32], log2_wd: i32, wt: i32, o: i32, max: i32, dst: &mut [u16], dst_stride: usize, w: usize, h: usize) {
    if log2_wd >= 1 {
        let round = 1 << (log2_wd - 1);
        for y in 0..h {
            for x in 0..w {
                let v = ((a[y * w + x] * wt + round) >> log2_wd) + o;
                dst[y * dst_stride + x] = v.clamp(0, max) as u16;
            }
        }
    } else {
        for y in 0..h {
            for x in 0..w {
                dst[y * dst_stride + x] = (a[y * w + x] * wt + o).clamp(0, max) as u16;
            }
        }
    }
}

/// Predict one prediction block of the picture (`w x h` luma at `(x, y)`).
#[allow(clippy::too_many_arguments)]
pub fn predict_block(
    cur: &mut Frame,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    ref0: Option<(&Frame, Mv)>,
    ref1: Option<(&Frame, Mv)>,
    weighting: [Weighting; 3],
) {
    let bd = cur.bit_depth;
    let mut l0 = vec![0i32; w * h];
    let mut l1 = vec![0i32; w * h];
    let (cw, ch) = (w / 2, h / 2);
    let mut c0 = [vec![0i32; cw * ch], vec![0i32; cw * ch]];
    let mut c1 = [vec![0i32; cw * ch], vec![0i32; cw * ch]];
    if let Some((r, mv)) = ref0 {
        luma_mc(&r.y, x as i32, y as i32, mv, w, h, bd, &mut l0);
        chroma_mc(&r.cb, (x / 2) as i32, (y / 2) as i32, mv, cw, ch, bd, &mut c0[0]);
        chroma_mc(&r.cr, (x / 2) as i32, (y / 2) as i32, mv, cw, ch, bd, &mut c0[1]);
    }
    if let Some((r, mv)) = ref1 {
        luma_mc(&r.y, x as i32, y as i32, mv, w, h, bd, &mut l1);
        chroma_mc(&r.cb, (x / 2) as i32, (y / 2) as i32, mv, cw, ch, bd, &mut c1[0]);
        chroma_mc(&r.cr, (x / 2) as i32, (y / 2) as i32, mv, cw, ch, bd, &mut c1[1]);
    }
    let yoff = cur.y.offset(x as isize, y as isize);
    let ystride = cur.y.stride;
    combine(ref0.map(|_| &l0[..]), ref1.map(|_| &l1[..]), weighting[0], bd, &mut cur.y.data[yoff..], ystride, w, h);
    let coff = cur.cb.offset((x / 2) as isize, (y / 2) as isize);
    let cstride = cur.cb.stride;
    combine(ref0.map(|_| &c0[0][..]), ref1.map(|_| &c1[0][..]), weighting[1], bd, &mut cur.cb.data[coff..], cstride, cw, ch);
    combine(ref0.map(|_| &c0[1][..]), ref1.map(|_| &c1[1][..]), weighting[2], bd, &mut cur.cr.data[coff..], cstride, cw, ch);
}
