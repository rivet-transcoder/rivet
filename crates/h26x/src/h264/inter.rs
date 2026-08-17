//! Inter prediction (H.264 clause 8.4.2): fractional sample interpolation
//! for luma (six-tap half-sample filter, averaged quarter samples) and
//! chroma (eighth-sample bilinear), and weighted sample prediction —
//! default, explicit and implicit.

use super::frame::{Frame, Mv, PaddedPlane};

/// How the two prediction lists are combined for a partition (8.4.2.3).
#[derive(Debug, Clone, Copy)]
pub enum Weighting {
    /// Default: plain average for bi-prediction, copy for uni-prediction.
    Default,
    /// Explicit or implicit weights: `(log_wd, w0, o0, w1, o1)` per component
    /// (luma, Cb, Cr). Offsets are already scaled to the sample bit depth.
    Weighted {
        /// logWD per component.
        log_wd: [i32; 3],
        /// Weights `[component][list]`.
        w: [[i32; 2]; 3],
        /// Offsets `[component][list]`.
        o: [[i32; 2]; 3],
    },
}

/// Fetch a `(w) x (h)` window of reference samples whose top-left is at
/// `(x0, y0)` (may be outside the picture: samples are clamped to the picture
/// edge, which the replicated border makes free for the common case).
#[inline]
fn fetch(p: &PaddedPlane, x0: i32, y0: i32, w: usize, h: usize, out: &mut [i32]) {
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
            let yy = (y0 + y as i32).clamp(0, ph - 1);
            for x in 0..w {
                let xx = (x0 + x as i32).clamp(0, pw - 1);
                out[y * w + x] = p.at(xx as isize, yy as isize) as i32;
            }
        }
    }
}

#[inline(always)]
fn tap6(a: i32, b: i32, c: i32, d: i32, e: i32, f: i32) -> i32 {
    a - 5 * b + 20 * c + 20 * d - 5 * e + f
}

#[inline(always)]
fn clip8(v: i32) -> i32 {
    v.clamp(0, 255)
}

/// Luma motion compensation (8.4.2.2.1) of a `w x h` block at integer
/// position `(x, y)` in the reference plus quarter-sample vector `mv`;
/// writes the predicted samples (0..=255) into `out` (row-major, `w` wide).
pub fn mc_luma(reference: &PaddedPlane, x: i32, y: i32, mv: Mv, w: usize, h: usize, out: &mut [i32]) {
    let xi = x + (mv.x as i32 >> 2);
    let yi = y + (mv.y as i32 >> 2);
    let xf = (mv.x & 3) as usize;
    let yf = (mv.y & 3) as usize;
    // Window with a 2-sample margin above/left and 3 below/right.
    let ww = w + 5;
    let hh = h + 5;
    let mut src = vec![0i32; ww * hh];
    fetch(reference, xi - 2, yi - 2, ww, hh, &mut src);
    let s = |xx: usize, yy: usize| src[(yy + 2) * ww + (xx + 2)]; // full-sample G at block coords
    let sraw = |xx: i32, yy: i32| src[((yy + 2) as usize) * ww + (xx + 2) as usize];

    if xf == 0 && yf == 0 {
        for yy in 0..h {
            for xx in 0..w {
                out[yy * w + xx] = s(xx, yy);
            }
        }
        return;
    }
    // Horizontal half-sample intermediates b1 at rows -2..h+3, columns 0..w
    // (b at (x,y) uses E..J = x-2..x+3 in the row).
    let b1 = |xx: i32, yy: i32| -> i32 {
        tap6(sraw(xx - 2, yy), sraw(xx - 1, yy), sraw(xx, yy), sraw(xx + 1, yy), sraw(xx + 2, yy), sraw(xx + 3, yy))
    };
    // Vertical half-sample intermediates h1 at columns -2..w+3.
    let h1 = |xx: i32, yy: i32| -> i32 {
        tap6(sraw(xx, yy - 2), sraw(xx, yy - 1), sraw(xx, yy), sraw(xx, yy + 1), sraw(xx, yy + 2), sraw(xx, yy + 3))
    };
    let b = |xx: i32, yy: i32| clip8((b1(xx, yy) + 16) >> 5);
    let hh_ = |xx: i32, yy: i32| clip8((h1(xx, yy) + 16) >> 5);
    // j from the vertical filter over horizontal intermediates.
    let j = |xx: i32, yy: i32| -> i32 {
        let j1 = tap6(b1(xx, yy - 2), b1(xx, yy - 1), b1(xx, yy), b1(xx, yy + 1), b1(xx, yy + 2), b1(xx, yy + 3));
        clip8((j1 + 512) >> 10)
    };
    for yy in 0..h as i32 {
        for xx in 0..w as i32 {
            let g = sraw(xx, yy);
            let v = match (xf, yf) {
                (1, 0) => (g + b(xx, yy) + 1) >> 1,
                (2, 0) => b(xx, yy),
                (3, 0) => (sraw(xx + 1, yy) + b(xx, yy) + 1) >> 1,
                (0, 1) => (g + hh_(xx, yy) + 1) >> 1,
                (0, 2) => hh_(xx, yy),
                (0, 3) => (sraw(xx, yy + 1) + hh_(xx, yy) + 1) >> 1,
                (2, 2) => j(xx, yy),
                (1, 1) => (b(xx, yy) + hh_(xx, yy) + 1) >> 1,
                (3, 1) => (b(xx, yy) + hh_(xx + 1, yy) + 1) >> 1,
                (1, 3) => (hh_(xx, yy) + b(xx, yy + 1) + 1) >> 1,
                (3, 3) => (hh_(xx + 1, yy) + b(xx, yy + 1) + 1) >> 1,
                (2, 1) => (b(xx, yy) + j(xx, yy) + 1) >> 1,
                (2, 3) => (j(xx, yy) + b(xx, yy + 1) + 1) >> 1,
                (1, 2) => (hh_(xx, yy) + j(xx, yy) + 1) >> 1,
                (3, 2) => (j(xx, yy) + hh_(xx + 1, yy) + 1) >> 1,
                _ => g,
            };
            out[yy as usize * w + xx as usize] = v;
        }
    }
}

/// Chroma motion compensation for 4:2:0 (8.4.2.2.2): block of `w x h`
/// chroma samples at chroma position `(xc, yc)` with the luma vector `mv`
/// (eighth-sample units in chroma).
pub fn mc_chroma_420(reference: &PaddedPlane, xc: i32, yc: i32, mv: Mv, w: usize, h: usize, out: &mut [i32]) {
    let xi = xc + (mv.x as i32 >> 3);
    let yi = yc + (mv.y as i32 >> 3);
    let xf = (mv.x & 7) as i32;
    let yf = (mv.y & 7) as i32;
    let ww = w + 1;
    let hh = h + 1;
    let mut src = vec![0i32; ww * hh];
    fetch(reference, xi, yi, ww, hh, &mut src);
    for yy in 0..h {
        for xx in 0..w {
            let a = src[yy * ww + xx];
            let b = src[yy * ww + xx + 1];
            let c = src[(yy + 1) * ww + xx];
            let d = src[(yy + 1) * ww + xx + 1];
            out[yy * w + xx] = ((8 - xf) * (8 - yf) * a + xf * (8 - yf) * b + (8 - xf) * yf * c + xf * yf * d + 32) >> 6;
        }
    }
}

/// Combine the list-0 / list-1 predictions of one component (8.4.2.3) and
/// write into `dst`.
pub fn combine(
    p0: Option<&[i32]>,
    p1: Option<&[i32]>,
    weighting: Weighting,
    comp: usize,
    dst: &mut [u8],
    dst_stride: usize,
    w: usize,
    h: usize,
) {
    match (p0, p1, weighting) {
        (Some(a), None, Weighting::Default) | (None, Some(a), Weighting::Default) => {
            for y in 0..h {
                for x in 0..w {
                    dst[y * dst_stride + x] = a[y * w + x] as u8;
                }
            }
        }
        (Some(a), Some(b), Weighting::Default) => {
            for y in 0..h {
                for x in 0..w {
                    dst[y * dst_stride + x] = ((a[y * w + x] + b[y * w + x] + 1) >> 1) as u8;
                }
            }
        }
        (Some(a), None, Weighting::Weighted { log_wd, w: wt, o }) => {
            uni_weighted(a, log_wd[comp], wt[comp][0], o[comp][0], dst, dst_stride, w, h);
        }
        (None, Some(b), Weighting::Weighted { log_wd, w: wt, o }) => {
            uni_weighted(b, log_wd[comp], wt[comp][1], o[comp][1], dst, dst_stride, w, h);
        }
        (Some(a), Some(b), Weighting::Weighted { log_wd, w: wt, o }) => {
            let lwd = log_wd[comp];
            let (w0, w1) = (wt[comp][0], wt[comp][1]);
            let off = (o[comp][0] + o[comp][1] + 1) >> 1;
            let round = 1 << lwd;
            for y in 0..h {
                for x in 0..w {
                    let v = ((a[y * w + x] * w0 + b[y * w + x] * w1 + round) >> (lwd + 1)) + off;
                    dst[y * dst_stride + x] = clip8(v) as u8;
                }
            }
        }
        (None, None, _) => {}
    }
}

#[allow(clippy::too_many_arguments)]
fn uni_weighted(a: &[i32], lwd: i32, wt: i32, o: i32, dst: &mut [u8], dst_stride: usize, w: usize, h: usize) {
    if lwd >= 1 {
        let round = 1 << (lwd - 1);
        for y in 0..h {
            for x in 0..w {
                let v = ((a[y * w + x] * wt + round) >> lwd) + o;
                dst[y * dst_stride + x] = clip8(v) as u8;
            }
        }
    } else {
        for y in 0..h {
            for x in 0..w {
                dst[y * dst_stride + x] = clip8(a[y * w + x] * wt + o) as u8;
            }
        }
    }
}

/// Predict one partition (`w x h` luma samples at `(x, y)` in the picture)
/// from up to two references into `cur`.
#[allow(clippy::too_many_arguments)]
pub fn predict_partition(
    cur: &mut Frame,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    ref0: Option<(&Frame, Mv)>,
    ref1: Option<(&Frame, Mv)>,
    weighting: Weighting,
) {
    let mut l0 = [0i32; 256];
    let mut l1 = [0i32; 256];
    let mut c0 = [[0i32; 64]; 2];
    let mut c1 = [[0i32; 64]; 2];
    let (cw, ch) = (w / 2, h / 2);
    if let Some((r, mv)) = ref0 {
        mc_luma(&r.y, x as i32, y as i32, mv, w, h, &mut l0[..w * h]);
        mc_chroma_420(&r.cb, (x / 2) as i32, (y / 2) as i32, mv, cw, ch, &mut c0[0][..cw * ch]);
        mc_chroma_420(&r.cr, (x / 2) as i32, (y / 2) as i32, mv, cw, ch, &mut c0[1][..cw * ch]);
    }
    if let Some((r, mv)) = ref1 {
        mc_luma(&r.y, x as i32, y as i32, mv, w, h, &mut l1[..w * h]);
        mc_chroma_420(&r.cb, (x / 2) as i32, (y / 2) as i32, mv, cw, ch, &mut c1[0][..cw * ch]);
        mc_chroma_420(&r.cr, (x / 2) as i32, (y / 2) as i32, mv, cw, ch, &mut c1[1][..cw * ch]);
    }
    let yoff = cur.y.offset(x as isize, y as isize);
    let ystride = cur.y.stride;
    combine(
        ref0.map(|_| &l0[..w * h]),
        ref1.map(|_| &l1[..w * h]),
        weighting,
        0,
        &mut cur.y.data[yoff..],
        ystride,
        w,
        h,
    );
    let coff = cur.cb.offset((x / 2) as isize, (y / 2) as isize);
    let cstride = cur.cb.stride;
    combine(
        ref0.map(|_| &c0[0][..cw * ch]),
        ref1.map(|_| &c1[0][..cw * ch]),
        weighting,
        1,
        &mut cur.cb.data[coff..],
        cstride,
        cw,
        ch,
    );
    combine(
        ref0.map(|_| &c0[1][..cw * ch]),
        ref1.map(|_| &c1[1][..cw * ch]),
        weighting,
        2,
        &mut cur.cr.data[coff..],
        cstride,
        cw,
        ch,
    );
}
