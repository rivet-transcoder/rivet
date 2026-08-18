//! Inter prediction (H.264 clause 8.4.2): fractional sample interpolation
//! for luma (six-tap half-sample filter, averaged quarter samples) and
//! chroma (eighth-sample bilinear), and weighted sample prediction —
//! default, explicit and implicit — on the [`crate::dsp::h264`] kernels.

use crate::dsp::h264::H264Dsp;

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

/// A gathered source window for vectors that leave the padded plane.
struct Window {
    data: [u8; 32 * 32],
}

/// Run one interpolation of a `w x h` block whose six-tap (luma) / bilinear
/// (chroma) window starts at `(x0, y0)` in `plane`, into `out`.
#[allow(clippy::too_many_arguments)]
fn interp(
    dsp: &H264Dsp,
    plane: &PaddedPlane,
    x0: i32,
    y0: i32,
    ww: usize,
    hh: usize,
    window: &mut Window,
    out: &mut [u8],
    kernel: impl FnOnce(&mut [u8], &[u8], usize),
) {
    let pad = plane.pad as i32;
    let (pw, ph) = (plane.width as i32, plane.height as i32);
    if x0 >= -pad && y0 >= -pad && x0 + ww as i32 <= pw + pad && y0 + hh as i32 <= ph + pad {
        kernel(out, &plane.data[plane.offset(x0 as isize, y0 as isize)..], plane.stride);
    } else {
        // Clamp to the picture, sample by sample (a rare, far-out vector).
        let stride = 32;
        for y in 0..hh {
            let yy = (y0 + y as i32).clamp(0, ph - 1) as isize;
            for x in 0..ww {
                let xx = (x0 + x as i32).clamp(0, pw - 1) as isize;
                window.data[y * stride + x] = plane.at(xx, yy);
            }
        }
        kernel(out, &window.data, stride);
    }
}

/// Predict one partition (`w x h` luma samples at `(x, y)` in the picture)
/// from up to two references into `cur`.
#[allow(clippy::too_many_arguments)]
pub fn predict_partition(
    dsp: &H264Dsp,
    cur: &mut Frame,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    ref0: Option<(&Frame, Mv)>,
    ref1: Option<(&Frame, Mv)>,
    weighting: Weighting,
) {
    // Per list: luma 16x16, chroma 8x8 max (one buffer size for simplicity).
    let mut pred = [[[0u8; 256]; 3]; 2];
    let mut window = Window { data: [0; 32 * 32] };
    let (cw, ch) = (w / 2, h / 2);
    for (list, r) in [ref0, ref1].into_iter().enumerate() {
        let Some((rf, mv)) = r else { continue };
        let [pl, pcb, pcr] = &mut pred[list];
        // Luma: window two left / above the integer position, (w + 5) x (h + 5).
        let xi = x as i32 + (mv.x as i32 >> 2);
        let yi = y as i32 + (mv.y as i32 >> 2);
        let pos = ((mv.y & 3) as usize) * 4 + (mv.x & 3) as usize;
        let k = dsp.qpel[pos];
        interp(dsp, &rf.y, xi - 2, yi - 2, w + 5, h + 5, &mut window, &mut pl[..w * h], |o, s, st| k(o, s, st, w, h));
        // Chroma (4:2:0): eighth-sample bilinear, window (cw + 1) x (ch + 1).
        let xci = (x / 2) as i32 + (mv.x as i32 >> 3);
        let yci = (y / 2) as i32 + (mv.y as i32 >> 3);
        let (xf, yf) = ((mv.x & 7) as i32, (mv.y & 7) as i32);
        let kc = dsp.chroma;
        interp(dsp, &rf.cb, xci, yci, cw + 1, ch + 1, &mut window, &mut pcb[..cw * ch], |o, s, st| kc(o, s, st, cw, ch, xf, yf));
        interp(dsp, &rf.cr, xci, yci, cw + 1, ch + 1, &mut window, &mut pcr[..cw * ch], |o, s, st| kc(o, s, st, cw, ch, xf, yf));
    }
    let both = ref0.is_some() && ref1.is_some();
    let single = if ref0.is_some() { 0 } else { 1 };
    let planes: [(&mut PaddedPlane, usize, usize, usize, usize); 3] =
        [(&mut cur.y, x, y, w, h), (&mut cur.cb, x / 2, y / 2, cw, ch), (&mut cur.cr, x / 2, y / 2, cw, ch)];
    for (c, (plane, px, py, pwid, phei)) in planes.into_iter().enumerate() {
        let off = plane.offset(px as isize, py as isize);
        let stride = plane.stride;
        let dst = &mut plane.data[off..];
        let n = pwid * phei;
        let a = &pred[0][c][..n];
        let b = &pred[1][c][..n];
        match (both, weighting) {
            (false, Weighting::Default) => (dsp.copy)(dst, stride, if single == 0 { a } else { b }, pwid, phei),
            (true, Weighting::Default) => (dsp.avg)(dst, stride, a, b, pwid, phei),
            (false, Weighting::Weighted { log_wd, w: wt, o }) => {
                (dsp.weighted_uni)(dst, stride, if single == 0 { a } else { b }, pwid, phei, log_wd[c], wt[c][single], o[c][single]);
            }
            (true, Weighting::Weighted { log_wd, w: wt, o }) => {
                (dsp.weighted_bi)(dst, stride, a, b, pwid, phei, log_wd[c], wt[c][0], wt[c][1], o[c][0], o[c][1]);
            }
        }
    }
}
