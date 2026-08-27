//! `hqdn3d` — high-quality 3D denoise: a **temporal** filter.
//!
//! Each sample is low-passed along its row, then down its column, then
//! against the same sample of the previous *output* frame — three first-order
//! IIR stages whose coefficient is a function of the difference being
//! smoothed, so a small difference (noise) is averaged away and a large one
//! (an edge, motion) passes through. The parameters and the arithmetic follow
//! ffmpeg's `vf_hqdn3d` (`hqdn3d=luma_spatial:chroma_spatial:luma_tmp:chroma_tmp`),
//! including its 16-bit intermediate precision and its coefficient tables, so
//! a command line means the same thing here.
//!
//! The temporal stage is why this filter needs state: a [`State`] per decode
//! stream holds the previous output of every plane. It is created by
//! [`super::super::FilterInstance`] and never shared.

use anyhow::Result;

use super::super::{assemble, planes_8bit};
use crate::frame::VideoFrame;

/// ffmpeg's `LUT_BITS` for 8-bit input.
const LUT_BITS: u32 = 4;
/// Where difference `0` sits in a coefficient table.
const LUT_HALF: usize = 256 << LUT_BITS;
/// Entries in a coefficient table (differences `−LUT_HALF .. LUT_HALF`).
const LUT_LEN: usize = 512 << LUT_BITS;

/// ffmpeg's defaults: `luma_spatial = 4`, and the rest derived from it.
const LUMA_SPATIAL_DEFAULT: f64 = 4.0;
const CHROMA_SPATIAL_DEFAULT: f64 = 3.0;
const LUMA_TMP_DEFAULT: f64 = 6.0;

/// The four strengths, with ffmpeg's derivation applied to any that was
/// omitted (given as `0`): `cs = 3·ls/4`, `lt = 6·ls/4`, `ct = lt·cs/ls`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Strengths {
    pub luma_spatial: f32,
    pub chroma_spatial: f32,
    pub luma_tmp: f32,
    pub chroma_tmp: f32,
}

impl Strengths {
    /// Resolve `hqdn3d=ls:cs:lt:ct` the way ffmpeg's `init` does: a zero
    /// means "derive from the others". Negative values are the caller's to
    /// reject.
    pub fn resolve(ls: f32, cs: f32, lt: f32, ct: f32) -> Strengths {
        let ls = if ls == 0.0 {
            LUMA_SPATIAL_DEFAULT
        } else {
            ls as f64
        };
        let cs = if cs == 0.0 {
            CHROMA_SPATIAL_DEFAULT * ls / LUMA_SPATIAL_DEFAULT
        } else {
            cs as f64
        };
        let lt = if lt == 0.0 {
            LUMA_TMP_DEFAULT * ls / LUMA_SPATIAL_DEFAULT
        } else {
            lt as f64
        };
        let ct = if ct == 0.0 { lt * cs / ls } else { ct as f64 };
        Strengths {
            luma_spatial: ls as f32,
            chroma_spatial: cs as f32,
            luma_tmp: lt as f32,
            chroma_tmp: ct as f32,
        }
    }
}

/// One coefficient table: for a difference `d` (in 1/16ths of an 8-bit step,
/// so ±4096 spans the whole 16-bit range) the correction to add to the
/// current sample, `simil^γ · (prev − cur)` in 16-bit units, where
/// `simil = 1 − |Δ|/255` and `γ` is chosen so a difference of `strength`
/// keeps a quarter of itself. ffmpeg's `precalc_coefs`.
fn precalc_coefs(dist25: f64) -> Vec<i16> {
    let gamma = (0.25f64).ln() / (1.0 - dist25.min(252.0) / 255.0 - 0.00001).ln();
    let mut ct = vec![0i16; LUT_LEN];
    for i in -(LUT_HALF as i64)..(LUT_HALF as i64) {
        // Midpoint of the bin, in 8-bit units.
        let f = ((i * (1 << (9 - LUT_BITS))) + (1 << (8 - LUT_BITS)) - 1) as f64 / 512.0;
        let simil = (1.0 - f.abs() / 255.0).max(0.0);
        let c = simil.powf(gamma) * 256.0 * f;
        ct[(LUT_HALF as i64 + i) as usize] = c.round_ties_even() as i16;
    }
    ct
}

/// The prepared filter: the four coefficient tables, built once per chain.
pub(crate) struct Prepared {
    luma_spatial: Vec<i16>,
    chroma_spatial: Vec<i16>,
    luma_tmp: Vec<i16>,
    chroma_tmp: Vec<i16>,
}

/// Per-stream history: the previous output frame of every plane at 16-bit
/// precision, plus the row scratch. Belongs to one decode stream.
pub(crate) struct State {
    width: u32,
    height: u32,
    /// `frame_ant` per plane (Y, U, V).
    prev: [Vec<u16>; 3],
    /// `line_ant` — one luma row.
    line: Vec<u16>,
}

impl Prepared {
    pub(crate) fn new(strengths: Strengths) -> Self {
        Prepared {
            luma_spatial: precalc_coefs(strengths.luma_spatial as f64),
            chroma_spatial: precalc_coefs(strengths.chroma_spatial as f64),
            luma_tmp: precalc_coefs(strengths.luma_tmp as f64),
            chroma_tmp: precalc_coefs(strengths.chroma_tmp as f64),
        }
    }

    /// Filter one frame against the stream's history, updating it. A missing
    /// history — the first frame, or a frame of a different size — starts
    /// from this frame, as ffmpeg does (the previous frame is taken to be the
    /// source itself).
    pub(crate) fn apply(
        &self,
        state: &mut Option<State>,
        frame: &VideoFrame,
    ) -> Result<VideoFrame> {
        let (yp, up, vp) = planes_8bit(frame, "hqdn3d")?;
        let (w, h) = (frame.width as usize, frame.height as usize);
        let (cw, ch) = (w / 2, h / 2);
        let fresh = match state {
            Some(s) if s.width == frame.width && s.height == frame.height => false,
            _ => true,
        };
        if fresh {
            let load = |p: &[u8]| p.iter().map(|&v| load(v) as u16).collect::<Vec<u16>>();
            *state = Some(State {
                width: frame.width,
                height: frame.height,
                prev: [load(&yp), load(&up), load(&vp)],
                line: vec![0u16; w.max(1)],
            });
        }
        let st = state.as_mut().expect("just set");
        let [py, pu, pv] = &mut st.prev;
        let mut out_y = vec![0u8; w * h];
        let mut out_u = vec![0u8; cw * ch];
        let mut out_v = vec![0u8; cw * ch];
        // The three planes are independent; the row chain inside a plane is
        // not, so a plane is the unit of parallelism.
        std::thread::scope(|scope| {
            let line = &mut st.line;
            let (oy, ou, ov) = (&mut out_y, &mut out_u, &mut out_v);
            scope.spawn(move || plane(&yp, oy, line, py, w, h, &self.luma_spatial, &self.luma_tmp));
            let mut line_u = vec![0u16; cw.max(1)];
            scope.spawn(move || {
                plane(
                    &up,
                    ou,
                    &mut line_u,
                    pu,
                    cw,
                    ch,
                    &self.chroma_spatial,
                    &self.chroma_tmp,
                )
            });
            let mut line_v = vec![0u16; cw.max(1)];
            plane(
                &vp,
                ov,
                &mut line_v,
                pv,
                cw,
                ch,
                &self.chroma_spatial,
                &self.chroma_tmp,
            );
        });
        Ok(assemble(
            frame,
            frame.width,
            frame.height,
            out_y,
            out_u,
            out_v,
        ))
    }
}

/// An 8-bit sample at 16-bit precision, centred in its bin (`LOAD`).
#[inline(always)]
fn load(v: u8) -> i32 {
    ((v as i32) << 8) + ((1 << 8) - 1) / 2
}

/// One IIR step: `cur + coef[(prev − cur) >> 4]`. Everything is `i32`; the
/// caller stores to `u16` (wrapping) and emits `>> 8`, exactly as the C does.
#[inline(always)]
fn lowpass(prev: i32, cur: i32, coef: &[i16]) -> i32 {
    let d = (prev - cur) >> (8 - LUT_BITS);
    cur + coef[(LUT_HALF as i32 + d) as usize] as i32
}

/// ffmpeg's `denoise_spatial`, one plane: the spatial row/column IIR feeding
/// the temporal IIR against `frame_ant`, which is updated in place.
fn plane(
    src: &[u8],
    dst: &mut [u8],
    line_ant: &mut [u16],
    frame_ant: &mut [u16],
    w: usize,
    h: usize,
    spatial: &[i16],
    temporal: &[i16],
) {
    if w == 0 || h == 0 {
        return;
    }
    // First line has no top neighbour: only the left one, and the last frame.
    let mut pixel_ant = load(src[0]);
    for x in 0..w {
        pixel_ant = lowpass(pixel_ant, load(src[x]), spatial);
        line_ant[x] = pixel_ant as u16;
        let tmp = lowpass(frame_ant[x] as i32, pixel_ant, temporal);
        frame_ant[x] = tmp as u16;
        dst[x] = ((tmp as u32) >> 8) as u8;
    }
    for y in 1..h {
        let src = &src[y * w..][..w];
        let dst = &mut dst[y * w..][..w];
        let frame_ant = &mut frame_ant[y * w..][..w];
        let mut pixel_ant = load(src[0]);
        for x in 0..w - 1 {
            let tmp = lowpass(line_ant[x] as i32, pixel_ant, spatial);
            line_ant[x] = tmp as u16;
            pixel_ant = lowpass(pixel_ant, load(src[x + 1]), spatial);
            let tmp = lowpass(frame_ant[x] as i32, tmp, temporal);
            frame_ant[x] = tmp as u16;
            dst[x] = ((tmp as u32) >> 8) as u8;
        }
        let x = w - 1;
        let tmp = lowpass(line_ant[x] as i32, pixel_ant, spatial);
        line_ant[x] = tmp as u16;
        let tmp = lowpass(frame_ant[x] as i32, tmp, temporal);
        frame_ant[x] = tmp as u16;
        dst[x] = ((tmp as u32) >> 8) as u8;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn omitted_strengths_derive_as_ffmpeg_does() {
        assert_eq!(
            Strengths::resolve(0.0, 0.0, 0.0, 0.0),
            Strengths {
                luma_spatial: 4.0,
                chroma_spatial: 3.0,
                luma_tmp: 6.0,
                chroma_tmp: 4.5
            }
        );
        assert_eq!(
            Strengths::resolve(8.0, 0.0, 0.0, 0.0),
            Strengths {
                luma_spatial: 8.0,
                chroma_spatial: 6.0,
                luma_tmp: 12.0,
                chroma_tmp: 9.0
            }
        );
        // An explicit value is kept; only the omitted ones derive.
        assert_eq!(
            Strengths::resolve(4.0, 3.0, 6.0, 4.5),
            Strengths {
                luma_spatial: 4.0,
                chroma_spatial: 3.0,
                luma_tmp: 6.0,
                chroma_tmp: 4.5
            }
        );
        assert_eq!(
            Strengths::resolve(4.0, 1.0, 0.0, 0.0).chroma_tmp,
            6.0 * 1.0 / 4.0
        );
    }

    #[test]
    fn coefficient_tables_match_values_computed_outside_this_code() {
        // `precalc_coefs` evaluated independently (Python, IEEE double,
        // round-half-even) at a spread of differences, per strength.
        let cases: [(f64, &[(i64, i16)]); 4] = [
            (
                4.0,
                &[
                    (0, 7),
                    (1, 23),
                    (-1, -8),
                    (16, 185),
                    (-16, -178),
                    (64, 255),
                    (-64, -257),
                    (128, 125),
                    (512, 0),
                    (-4096, 0),
                    (4095, 0),
                ],
            ),
            (
                3.0,
                &[
                    (0, 7),
                    (16, 164),
                    (-16, -159),
                    (64, 160),
                    (-64, -162),
                    (128, 49),
                    (512, 0),
                ],
            ),
            (
                6.0,
                &[
                    (0, 7),
                    (16, 208),
                    (-16, -199),
                    (64, 408),
                    (-64, -408),
                    (128, 319),
                    (512, 3),
                    (2048, 0),
                ],
            ),
            (
                4.5,
                &[
                    (0, 7),
                    (16, 192),
                    (-16, -185),
                    (64, 299),
                    (-64, -300),
                    (128, 170),
                    (512, 0),
                ],
            ),
        ];
        for (strength, want) in cases {
            let ct = precalc_coefs(strength);
            assert_eq!(ct.len(), LUT_LEN);
            for &(d, c) in want {
                assert_eq!(
                    ct[(LUT_HALF as i64 + d) as usize],
                    c,
                    "strength {strength} d {d}"
                );
            }
        }
    }

    #[test]
    fn a_stronger_setting_smooths_a_given_difference_more() {
        let weak = precalc_coefs(2.0);
        let strong = precalc_coefs(8.0);
        for d in [8i64, 32, 64, 128] {
            let (w, s) = (
                weak[(LUT_HALF as i64 + d) as usize],
                strong[(LUT_HALF as i64 + d) as usize],
            );
            assert!(s > w, "d {d}: strong {s} <= weak {w}");
        }
    }
}
