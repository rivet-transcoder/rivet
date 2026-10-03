//! `hqdn3d` — a high-quality spatio-temporal ("3D") denoiser. Clean-room
//! design (see `docs/filters/hqdn3d.md` for provenance); the option syntax is
//! the familiar `luma_spatial:chroma_spatial:luma_tmp:chroma_tmp`.
//!
//! Every stage is the same edge-preserving first-order recursive low-pass:
//!
//! ```text
//! y = x + k(|y_prev − x|) · (y_prev − x)
//! k(d) = k_max · exp(−(d / τ)²),   k_max = S / (S + KNEE),   τ = EDGE · S
//! ```
//!
//! `x` is the incoming sample, `y_prev` the filter's running state and `S` the
//! stage's strength. A small difference is mostly noise, so the state is kept
//! (up to `k_max`); a difference well beyond `τ` is an edge or motion, so `k`
//! falls to ~0 and the sample passes through. A stronger `S` both keeps more
//! (`k_max` → 1: a longer average) and tolerates larger differences (`τ`).
//!
//! - **Spatial**: the recursion runs left→right then right→left along every
//!   row, then top→bottom then bottom→top down every column. The forward and
//!   backward sweeps cancel each other's lag, so the result is centred (no
//!   smear in one direction).
//! - **Temporal**: the spatially filtered frame is blended, sample by sample,
//!   into the previous output frame with the same recursion — a static area
//!   converges to its long-run average, a moving one passes through.
//!
//! The history is kept in `f32`, so slow convergence is not lost to rounding;
//! the output is rounded once. A frame whose samples are all equal comes out
//! unchanged (every difference is 0).

use anyhow::Result;

use super::super::{assemble, planes_8bit};
use super::for_row_bands;
use crate::frame::VideoFrame;

/// The documented defaults: `luma_spatial = 4`; the rest derive from it.
const LUMA_SPATIAL_DEFAULT: f32 = 4.0;

/// `k_max = S / (S + KNEE)`: the strength at which a flat area keeps half of
/// its running state per step.
const KNEE: f32 = 4.0;
/// `τ = EDGE · S`: the difference (in 8-bit code values) at which the
/// retention has fallen to `k_max / e`.
const EDGE: f32 = 1.5;

/// Retention table resolution: entries per code value; differences span
/// `0..=255`.
const STEPS: f32 = 8.0;
const CURVE_LEN: usize = 256 * STEPS as usize;

/// Below this many rows per band a thread does not pay for itself.
const MIN_BAND_ROWS: usize = 64;

/// The four strengths. An omitted value (given as `0`, or negative) is
/// derived from the others, as the filter's user documentation specifies:
/// `luma_spatial` defaults to 4, `chroma_spatial` to `3·ls/4`, `luma_tmp` to
/// `6·ls/4` and `chroma_tmp` to `lt·cs/ls`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Strengths {
    pub luma_spatial: f32,
    pub chroma_spatial: f32,
    pub luma_tmp: f32,
    pub chroma_tmp: f32,
}

impl Strengths {
    /// Resolve `ls:cs:lt:ct`, deriving any value that is not positive.
    pub fn resolve(ls: f32, cs: f32, lt: f32, ct: f32) -> Strengths {
        let given = |v: f32| v > 0.0 && v.is_finite();
        let luma_spatial = if given(ls) { ls } else { LUMA_SPATIAL_DEFAULT };
        let chroma_spatial = if given(cs) { cs } else { 3.0 * luma_spatial / 4.0 };
        let luma_tmp = if given(lt) { lt } else { 6.0 * luma_spatial / 4.0 };
        let chroma_tmp = if given(ct) { ct } else { luma_tmp * chroma_spatial / luma_spatial };
        Strengths { luma_spatial, chroma_spatial, luma_tmp, chroma_tmp }
    }
}

/// The retention `k(d)` of one stage, tabulated over `d ∈ [0, 256)`.
struct Curve {
    k: Vec<f32>,
}

impl Curve {
    fn new(strength: f32) -> Self {
        if strength <= 0.0 || !strength.is_finite() {
            return Curve { k: vec![0.0; CURVE_LEN] };
        }
        let k_max = strength / (strength + KNEE);
        let tau = EDGE * strength;
        let k = (0..CURVE_LEN)
            .map(|i| {
                let d = (i as f32 + 0.5) / STEPS;
                k_max * (-(d / tau) * (d / tau)).exp()
            })
            .collect();
        Curve { k }
    }

    /// One recursion step: the new state from the running `state` and the
    /// incoming sample `x`.
    #[inline(always)]
    fn step(&self, state: f32, x: f32) -> f32 {
        let diff = state - x;
        // `as usize` saturates; |diff| ≤ 255 keeps it in the table anyway.
        let k = self.k.get((diff.abs() * STEPS) as usize).copied().unwrap_or(0.0);
        x + k * diff
    }
}

/// The per-plane curves, built once per chain ([`super::super::FilterChain`])
/// and shared by every stream's instance.
pub(crate) struct Prepared {
    spatial: [Curve; 2],
    temporal: [Curve; 2],
}

/// One stream's history: the previous output frame, unrounded.
pub(crate) struct State {
    w: usize,
    h: usize,
    planes: [Vec<f32>; 3],
}

impl Prepared {
    pub(crate) fn new(strengths: Strengths) -> Self {
        Prepared {
            spatial: [Curve::new(strengths.luma_spatial), Curve::new(strengths.chroma_spatial)],
            temporal: [Curve::new(strengths.luma_tmp), Curve::new(strengths.chroma_tmp)],
        }
    }

    /// Filter the stream's next frame against `state` (its history), updating
    /// it. No history, or history of another frame size, starts afresh.
    pub(crate) fn apply(&self, state: &mut Option<State>, frame: &VideoFrame) -> Result<VideoFrame> {
        let (yp, up, vp) = planes_8bit(frame, "hqdn3d")?;
        let (w, h) = (frame.width as usize, frame.height as usize);
        let dims = [(w, h), (w / 2, h / 2), (w / 2, h / 2)];
        if state.as_ref().is_some_and(|s| s.w != w || s.h != h) {
            *state = None;
        }
        let mut next: [Vec<f32>; 3] = Default::default();
        let mut out: [Vec<u8>; 3] = Default::default();
        for (i, src) in [yp, up, vp].iter().enumerate() {
            let (pw, ph) = dims[i];
            let c = usize::from(i > 0);
            let mut cur = spatial(src, pw, ph, &self.spatial[c]);
            if let Some(prev) = state.as_ref().map(|s| &s.planes[i]) {
                temporal(&mut cur, prev, pw, &self.temporal[c]);
            }
            out[i] = cur.iter().map(|&v| v.round().clamp(0.0, 255.0) as u8).collect();
            next[i] = cur;
        }
        *state = Some(State { w, h, planes: next });
        let [y, u, v] = out;
        Ok(assemble(frame, frame.width, frame.height, y, u, v))
    }
}

/// The spatial stage: both directions along rows, then both down columns.
fn spatial(src: &[u8], w: usize, h: usize, curve: &Curve) -> Vec<f32> {
    let mut buf: Vec<f32> = src[..w * h].iter().map(|&v| v as f32).collect();
    if w == 0 || h == 0 {
        return buf;
    }
    for_row_bands(&mut buf, w, MIN_BAND_ROWS, |_, rows| {
        for row in rows.chunks_exact_mut(w) {
            sweep_row(row, curve);
        }
    });
    columns(&mut buf, w, h, curve);
    buf
}

/// Forward then backward recursion along one row, in place.
fn sweep_row(row: &mut [f32], curve: &Curve) {
    let mut s = row[0];
    for v in row.iter_mut() {
        s = curve.step(s, *v);
        *v = s;
    }
    let mut s = *row.last().unwrap();
    for v in row.iter_mut().rev() {
        s = curve.step(s, *v);
        *v = s;
    }
}

/// Downward then upward recursion along every column, in place. The state
/// is one row wide, so the sweep walks memory row by row.
fn columns(buf: &mut [f32], w: usize, h: usize, curve: &Curve) {
    let mut s = buf[..w].to_vec();
    for y in 0..h {
        let row = &mut buf[y * w..][..w];
        for (st, v) in s.iter_mut().zip(row.iter_mut()) {
            *st = curve.step(*st, *v);
            *v = *st;
        }
    }
    s.copy_from_slice(&buf[(h - 1) * w..][..w]);
    for y in (0..h).rev() {
        let row = &mut buf[y * w..][..w];
        for (st, v) in s.iter_mut().zip(row.iter_mut()) {
            *st = curve.step(*st, *v);
            *v = *st;
        }
    }
}

/// The temporal stage: blend `cur` into the previous output `prev`.
fn temporal(cur: &mut [f32], prev: &[f32], w: usize, curve: &Curve) {
    if w == 0 {
        return;
    }
    for_row_bands(cur, w, MIN_BAND_ROWS, |y0, rows| {
        for (c, &p) in rows.iter_mut().zip(&prev[y0 * w..]) {
            *c = curve.step(p, *c);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn omitted_strengths_derive_from_the_given_ones() {
        let r = Strengths::resolve;
        let s = |ls, cs, lt, ct| Strengths { luma_spatial: ls, chroma_spatial: cs, luma_tmp: lt, chroma_tmp: ct };
        assert_eq!(r(0.0, 0.0, 0.0, 0.0), s(4.0, 3.0, 6.0, 4.5));
        assert_eq!(r(8.0, 0.0, 0.0, 0.0), s(8.0, 6.0, 12.0, 9.0));
        assert_eq!(r(2.0, 0.0, 10.0, 0.0), s(2.0, 1.5, 10.0, 7.5));
        assert_eq!(r(4.0, 1.0, 0.0, 0.0), s(4.0, 1.0, 6.0, 1.5));
        assert_eq!(r(1.0, 2.0, 3.0, 4.0), s(1.0, 2.0, 3.0, 4.0));
        assert_eq!(r(-1.0, f32::NAN, 0.0, 0.0), s(4.0, 3.0, 6.0, 4.5));
    }

    #[test]
    fn retention_falls_with_the_difference_and_rises_with_the_strength() {
        for st in [1.0f32, 4.0, 10.0] {
            let c = Curve::new(st);
            assert!(c.k.windows(2).all(|p| p[1] <= p[0]), "k must not rise with d");
            assert!(c.k[0] < 1.0, "k must stay below 1 so the state cannot freeze");
            // An edge of 10·S passes essentially untouched.
            assert!(c.step(0.0, 10.0 * st) > 10.0 * st - 1e-3);
        }
        let (weak, strong) = (Curve::new(2.0), Curve::new(8.0));
        for d in [0.5f32, 2.0, 5.0, 12.0] {
            assert!(
                (strong.step(0.0, d) - d).abs() > (weak.step(0.0, d) - d).abs(),
                "a stronger setting must smooth a difference of {d} more"
            );
        }
        // Strength 0 is a pass-through.
        assert_eq!(Curve::new(0.0).step(7.0, 3.0), 3.0);
    }

    #[test]
    fn the_spatial_sweeps_are_symmetric() {
        // A centred impulse spreads the same amount left and right, up and
        // down: forward and backward sweeps cancel each other's lag.
        let (w, h) = (15, 15);
        let mut src = vec![100u8; w * h];
        src[7 * w + 7] = 104;
        let out = spatial(&src, w, h, &Curve::new(6.0));
        for d in 1..7 {
            let (l, r) = (out[7 * w + 7 - d], out[7 * w + 7 + d]);
            let (u, b) = (out[(7 - d) * w + 7], out[(7 + d) * w + 7]);
            assert!((l - r).abs() < 0.05, "row asymmetry at {d}: {l} vs {r}");
            assert!((u - b).abs() < 0.05, "column asymmetry at {d}: {u} vs {b}");
        }
    }
}
