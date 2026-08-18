//! Inter prediction (H.264 clause 8.4.2): fractional sample interpolation
//! for luma (six-tap half-sample filter, averaged quarter samples) and
//! chroma (eighth-sample bilinear), and weighted sample prediction —
//! default, explicit and implicit — on the [`crate::dsp::h264`] kernels.

use crate::dsp::h264::{H264Dsp, PRED_STRIDE};
use crate::sample::Sample;

use super::frame::{Frame, Mv, PARITY_FRAME, PaddedPlane};

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
struct Window<S: Sample> {
    data: [S; 32 * 32],
}

/// A read view of a reference plane: the frame, or one of its fields
/// (every other row, half the height). `pad_y` is how far above / below the
/// picture the border can be trusted for this view: a frame's own borders
/// for a frame read of a frame-bordered plane, half the border in field
/// rows for a field read of a field-bordered plane, nothing when the
/// border was built the other way (the window then clamps sample by
/// sample).
struct SrcView<'a, S: Sample> {
    data: &'a [S],
    origin: usize,
    stride: usize,
    width: i32,
    height: i32,
    pad_x: i32,
    pad_y: i32,
}

impl<'a, S: Sample> SrcView<'a, S> {
    fn of(plane: &'a PaddedPlane<S>, parity: u8, field_borders: bool) -> Self {
        let pad = plane.pad as i32;
        if parity == PARITY_FRAME {
            SrcView {
                data: &plane.data,
                origin: plane.origin(),
                stride: plane.stride,
                width: plane.width as i32,
                height: plane.height as i32,
                pad_x: pad,
                pad_y: if field_borders { 0 } else { pad },
            }
        } else {
            SrcView {
                data: &plane.data,
                origin: plane.origin() + parity as usize * plane.stride,
                stride: plane.stride * 2,
                width: plane.width as i32,
                height: (plane.height / 2) as i32,
                pad_x: pad,
                pad_y: if field_borders { pad / 2 } else { 0 },
            }
        }
    }
    #[inline(always)]
    fn offset(&self, x: isize, y: isize) -> usize {
        (self.origin as isize + y * self.stride as isize + x) as usize
    }
}

/// Run one interpolation of a `w x h` block whose six-tap (luma) / bilinear
/// (chroma) window starts at `(x0, y0)` in `src`, into `out`.
#[allow(clippy::too_many_arguments)]
fn interp<S: Sample>(
    src: &SrcView<S>,
    x0: i32,
    y0: i32,
    ww: usize,
    hh: usize,
    window: &mut Window<S>,
    out: &mut [S],
    kernel: impl FnOnce(&mut [S], &[S], usize),
) {
    let (pw, ph) = (src.width, src.height);
    if x0 >= -src.pad_x
        && y0 >= -src.pad_y
        && x0 + ww as i32 <= pw + src.pad_x
        && y0 + hh as i32 <= ph + src.pad_y
    {
        kernel(
            out,
            &src.data[src.offset(x0 as isize, y0 as isize)..],
            src.stride,
        );
    } else {
        // Clamp to the picture, sample by sample (a rare, far-out vector, or
        // a vertical excursion the border was not built for).
        let stride = 32;
        for y in 0..hh {
            let yy = (y0 + y as i32).clamp(0, ph - 1) as isize;
            for x in 0..ww {
                let xx = (x0 + x as i32).clamp(0, pw - 1) as isize;
                window.data[y * stride + x] = src.data[src.offset(xx, yy)];
            }
        }
        kernel(out, &window.data, stride);
    }
}

/// A reference for one list of a partition: the frame, the vector, and
/// which picture of the frame is read (0 / 1 a field, [`PARITY_FRAME`]).
pub type PartRef<'a, S> = (&'a Frame<S>, Mv, u8);

/// Where a macroblock's samples live: `x` and `y_pic` are its top-left in
/// the coordinates motion compensation works in (the frame's, or the
/// field's for a field picture or an MBAFF field macroblock); `y_dst` /
/// `yc_dst` the frame-buffer rows of its first luma / chroma line and
/// `step` the row step there (2 for an MBAFF field macroblock, whose lines
/// alternate with its pair partner's); `parity` which field it is (for the
/// chroma vector offset), [`PARITY_FRAME`] for a frame macroblock.
#[derive(Debug, Clone, Copy)]
pub struct MbGeom {
    /// Luma x of the macroblock in the picture.
    pub x: usize,
    /// Luma y of the macroblock in the picture (frame rows).
    pub y_pic: usize,
    /// Luma row of the macroblock's first line in the destination frame.
    pub y_dst: usize,
    /// Chroma row of the macroblock's first line in the destination frame.
    pub yc_dst: usize,
    /// Row step in the destination: 2 for an MBAFF field macroblock, else 1.
    pub step: usize,
    /// Field parity of the lines written, [`PARITY_FRAME`] for frame lines.
    pub parity: u8,
}

/// Predict one partition (`w x h` luma samples at `(x, y)` within the
/// macroblock `geom`) from up to two references into `cur`.
#[allow(clippy::too_many_arguments)]
pub fn predict_partition<S: Sample>(
    dsp: &H264Dsp<S>,
    cur: &mut Frame<S>,
    geom: MbGeom,
    x_in: usize,
    y_in: usize,
    w: usize,
    h: usize,
    ref0: Option<PartRef<S>>,
    ref1: Option<PartRef<S>>,
    weighting: Weighting,
) {
    let cur_parity = geom.parity;
    let (x, y) = (geom.x + x_in, geom.y_pic + y_in);
    // Per list and component: a 16x16 scratch block (stride PRED_STRIDE), the
    // prediction in its top-left. Chroma uses the same shape.
    let mut pred = [[[S::default(); 16 * PRED_STRIDE]; 3]; 2];
    let mut window = Window {
        data: [S::default(); 32 * 32],
    };
    let (sw, sh) = cur.chroma.subsampling();
    let (sw, sh) = (sw as usize, sh as usize);
    let mono = cur.chroma == crate::picture::ChromaFormat::Monochrome;
    // 4:4:4 chroma is interpolated like luma (8.4.2.2: mvCLX = mvLX).
    let c444 = cur.chroma == crate::picture::ChromaFormat::Yuv444;
    let (cw, ch) = (w / sw, h / sh);
    let max = (1i32 << cur.bit_depth) - 1;
    for (list, r) in [ref0, ref1].into_iter().enumerate() {
        let Some((rf, mv, rpar)) = r else { continue };
        let [pl, pcb, pcr] = &mut pred[list];
        let fb = rf.field_borders;
        // Luma: window two left / above the integer position, (w + 5) x (h + 5).
        let xi = x as i32 + (mv.x as i32 >> 2);
        let yi = y as i32 + (mv.y as i32 >> 2);
        let pos = ((mv.y & 3) as usize) * 4 + (mv.x & 3) as usize;
        let k = dsp.qpel[pos];
        interp(
            &SrcView::of(&rf.y, rpar, fb),
            xi - 2,
            yi - 2,
            w + 5,
            h + 5,
            &mut window,
            pl,
            |o, s, st| k(o, s, st, w, h, max),
        );
        if mono {
            continue;
        }
        if c444 {
            interp(
                &SrcView::of(&rf.cb, rpar, fb),
                xi - 2,
                yi - 2,
                w + 5,
                h + 5,
                &mut window,
                pcb,
                |o, s, st| k(o, s, st, w, h, max),
            );
            interp(
                &SrcView::of(&rf.cr, rpar, fb),
                xi - 2,
                yi - 2,
                w + 5,
                h + 5,
                &mut window,
                pcr,
                |o, s, st| k(o, s, st, w, h, max),
            );
            continue;
        }
        // Chroma: eighth-sample bilinear, window (cw + 1) x (ch + 1). The
        // vertical vector is in quarter chroma samples when chroma is not
        // subsampled vertically (4:2:2): `yFracC = (mv & 3) << 1` (8.4.1.4).
        // A field read of the opposite parity in 4:2:0 shifts the vertical
        // chroma vector by a quarter chroma sample (Table 8-10).
        let mvcy = mv.y as i32
            + if sh == 2 && cur_parity != PARITY_FRAME && rpar != PARITY_FRAME && rpar != cur_parity
            {
                if cur_parity == 1 { 2 } else { -2 }
            } else {
                0
            };
        let xci = (x / sw) as i32 + (mv.x as i32 >> 3);
        let (yci, yf) = if sh == 2 {
            ((y / 2) as i32 + (mvcy >> 3), mvcy & 7)
        } else {
            (y as i32 + (mvcy >> 2), (mvcy & 3) << 1)
        };
        let xf = (mv.x & 7) as i32;
        let kc = dsp.chroma;
        interp(
            &SrcView::of(&rf.cb, rpar, fb),
            xci,
            yci,
            cw + 1,
            ch + 1,
            &mut window,
            pcb,
            |o, s, st| kc(o, s, st, cw, ch, xf, yf),
        );
        interp(
            &SrcView::of(&rf.cr, rpar, fb),
            xci,
            yci,
            cw + 1,
            ch + 1,
            &mut window,
            pcr,
            |o, s, st| kc(o, s, st, cw, ch, xf, yf),
        );
    }
    let both = ref0.is_some() && ref1.is_some();
    let single = if ref0.is_some() { 0 } else { 1 };
    // Destination rows in the frame buffer.
    let (dy, dyc) = (
        geom.y_dst + geom.step * y_in,
        geom.yc_dst + geom.step * (y_in / sh),
    );
    let planes: [(&mut PaddedPlane<S>, usize, usize, usize, usize); 3] = [
        (&mut cur.y, x, dy, w, h),
        (&mut cur.cb, x / sw, dyc, cw, ch),
        (&mut cur.cr, x / sw, dyc, cw, ch),
    ];
    for (c, (plane, px, py, pwid, phei)) in planes.into_iter().enumerate() {
        if c > 0 && mono {
            break;
        }
        let off = plane.offset(px as isize, py as isize);
        let stride = plane.stride * geom.step;
        let dst = &mut plane.data[off..];
        let a = &pred[0][c][..];
        let b = &pred[1][c][..];
        match (both, weighting) {
            (false, Weighting::Default) => {
                (dsp.copy)(dst, stride, if single == 0 { a } else { b }, pwid, phei)
            }
            (true, Weighting::Default) => (dsp.avg)(dst, stride, a, b, pwid, phei),
            (false, Weighting::Weighted { log_wd, w: wt, o }) => {
                (dsp.weighted_uni)(
                    dst,
                    stride,
                    if single == 0 { a } else { b },
                    pwid,
                    phei,
                    log_wd[c],
                    wt[c][single],
                    o[c][single],
                    max,
                );
            }
            (true, Weighting::Weighted { log_wd, w: wt, o }) => {
                (dsp.weighted_bi)(
                    dst, stride, a, b, pwid, phei, log_wd[c], wt[c][0], wt[c][1], o[c][0], o[c][1],
                    max,
                );
            }
        }
    }
}
