//! Fractional sample interpolation (H.265 8.5.3.3.3) and weighted sample
//! prediction (8.5.3.3.4), on top of the [`crate::dsp::hevc`] kernels.

use crate::dsp::hevc::HevcDsp;

use super::frame::{Frame, Mv, Plane16};

/// Scratch buffers for one prediction block (allocated once per slice).
pub struct McScratch {
    /// 14-bit predictions per list: luma, cb, cr.
    pub pred: [[Vec<i16>; 3]; 2],
    /// Intermediate rows of the separable filter.
    pub tmp: Vec<i16>,
    /// A copy of the reference window when it lies outside the padded plane.
    pub window: Vec<u16>,
}

impl McScratch {
    /// Room for 64x64 blocks.
    pub fn new() -> Self {
        let n = 64 * 64;
        McScratch {
            pred: [[vec![0; n], vec![0; n], vec![0; n]], [vec![0; n], vec![0; n], vec![0; n]]],
            tmp: vec![0; 64 * (64 + 7)],
            window: vec![0; (64 + 7) * (64 + 7)],
        }
    }
}

impl Default for McScratch {
    fn default() -> Self {
        Self::new()
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

/// Interpolate one component block: `taps` (8 luma / 4 chroma) filter with
/// fractions `xf`/`yf` at integer position `(xi, yi)` of `plane`, into `out`
/// (`w * h`, 14-bit precision).
#[allow(clippy::too_many_arguments)]
fn interp(
    dsp: &HevcDsp,
    scratch_tmp: &mut [i16],
    window: &mut [u16],
    plane: &Plane16,
    xi: i32,
    yi: i32,
    xf: usize,
    yf: usize,
    w: usize,
    h: usize,
    luma: bool,
    bit_depth: u32,
    out: &mut [i16],
) {
    let reach: usize = if luma { 3 } else { 1 }; // taps before the sample
    let taps = if luma { 8 } else { 4 };
    let shift1 = bit_depth.min(12) as i32 - 8;
    let shift3 = 14 - bit_depth as i32;
    // The window of source samples the filter touches.
    let x0 = xi - reach as i32;
    let y0 = yi - reach as i32;
    let ww = w + taps - 1;
    let hh = h + taps - 1;
    let pad = plane.pad as i32;
    let (pw, ph) = (plane.width as i32, plane.height as i32);
    let inside = x0 >= -pad && y0 >= -pad && x0 + ww as i32 <= pw + pad && y0 + hh as i32 <= ph + pad;
    // Source slice starting at (x0, y0) with its stride.
    let (src, stride): (&[u16], usize) = if inside {
        (&plane.data[plane.offset(x0 as isize, y0 as isize)..], plane.stride)
    } else {
        // Vectors far outside the picture: gather with clamping.
        for yy in 0..hh {
            for xx in 0..ww {
                window[yy * ww + xx] = plane.at_clamped(x0 + xx as i32, y0 + yy as i32);
            }
        }
        (&window[..], ww)
    };
    // From the window origin to the block's own top-left.
    let at_block = reach * stride + reach;
    match (xf, yf) {
        (0, 0) => (if luma { dsp.qpel_copy } else { dsp.epel_copy })(out, &src[at_block..], stride, w, h, shift3),
        (_, 0) => (if luma { dsp.qpel_h } else { dsp.epel_h })(out, &src[reach * stride..], stride, w, h, xf, shift1),
        (0, _) => (if luma { dsp.qpel_v } else { dsp.epel_v })(out, &src[reach..], stride, w, h, yf, shift1),
        _ => {
            // Horizontal over h + taps - 1 rows, then vertical over the 14-bit rows.
            (if luma { dsp.qpel_h } else { dsp.epel_h })(scratch_tmp, src, stride, w, hh, xf, shift1);
            (if luma { dsp.qpel_v2 } else { dsp.epel_v2 })(out, scratch_tmp, w, w, h, yf);
        }
    }
}

/// Predict one prediction block of the picture (`w x h` luma at `(x, y)`).
#[allow(clippy::too_many_arguments)]
pub fn predict_block(
    dsp: &HevcDsp,
    scratch: &mut McScratch,
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
    let (cw, ch) = (w / 2, h / 2);
    let McScratch { pred, tmp, window } = scratch;
    for (list, r) in [ref0, ref1].into_iter().enumerate() {
        let Some((rf, mv)) = r else { continue };
        let [pl, pcb, pcr] = &mut pred[list];
        // Luma: quarter-sample vectors.
        let xi = x as i32 + (mv.x as i32 >> 2);
        let yi = y as i32 + (mv.y as i32 >> 2);
        interp(dsp, tmp, window, &rf.y, xi, yi, (mv.x & 3) as usize, (mv.y & 3) as usize, w, h, true, bd, pl);
        // Chroma (4:2:0): eighth-sample vectors in chroma units.
        let xci = (x / 2) as i32 + (mv.x as i32 >> 3);
        let yci = (y / 2) as i32 + (mv.y as i32 >> 3);
        let (xcf, ycf) = ((mv.x & 7) as usize, (mv.y & 7) as usize);
        interp(dsp, tmp, window, &rf.cb, xci, yci, xcf, ycf, cw, ch, false, bd, pcb);
        interp(dsp, tmp, window, &rf.cr, xci, yci, xcf, ycf, cw, ch, false, bd, pcr);
    }
    let max = (1i32 << bd) - 1;
    let both = ref0.is_some() && ref1.is_some();
    let planes: [(&mut Plane16, usize, usize, usize, usize); 3] =
        [(&mut cur.y, x, y, w, h), (&mut cur.cb, x / 2, y / 2, cw, ch), (&mut cur.cr, x / 2, y / 2, cw, ch)];
    for (c, (plane, px, py, pwid, phei)) in planes.into_iter().enumerate() {
        let off = plane.offset(px as isize, py as isize);
        let stride = plane.stride;
        let dst = &mut plane.data[off..];
        let a = &pred[0][c];
        let b = &pred[1][c];
        match (both, weighting[c]) {
            (false, Weighting::Default) => {
                let src = if ref0.is_some() { a } else { b };
                (dsp.uni)(dst, stride, src, pwid, phei, 14 - bd as i32, max);
            }
            (true, Weighting::Default) => (dsp.bi)(dst, stride, a, b, pwid, phei, 15 - bd as i32, max),
            (false, Weighting::Explicit { log2_wd, w: wt, o }) => {
                let (src, l) = if ref0.is_some() { (a, 0) } else { (b, 1) };
                (dsp.weighted_uni)(dst, stride, src, pwid, phei, log2_wd, wt[l], o[l], max);
            }
            (true, Weighting::Explicit { log2_wd, w: wt, o }) => {
                (dsp.weighted_bi)(dst, stride, a, b, pwid, phei, log2_wd, wt[0], wt[1], o[0], o[1], max)
            }
        }
    }
}
