//! `denoise` — spatial denoise with a **selectable algorithm** ([`DenoiseMethod`])
//! and a `strength` (`0.0..=1.0`) that blends the filtered result back with the
//! source. Each method lives in its own file; they share the dispatch + blend
//! here. 8-bit `Yuv420p` only (luma + chroma).
//!
//! `strength` is a uniform "how much" dial: every method runs at a fixed,
//! moderate internal setting and the output is `src·(1−s) + filtered·s`, so the
//! same number means the same amount of denoising regardless of algorithm.

use std::fmt;
use std::sync::OnceLock;

use anyhow::Result;

use super::{assemble, planes_8bit};
use crate::frame::VideoFrame;
use simd::{Simd, Tier, round_clamp_u8, tiered};

mod anisotropic;
mod bilateral;
mod gaussian;
pub(crate) mod hqdn3d;
mod mean;
mod median;
mod nlmeans;
pub(crate) mod simd;

/// Which spatial denoise algorithm [`super::VideoFilter::Denoise`] runs. Each
/// suits a different kind of noise; `strength` then blends the result with the
/// source. (The temporal denoiser, [`super::VideoFilter::Hqdn3d`], is its own
/// filter: it needs frame history, which a per-frame method cannot carry.)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum DenoiseMethod {
    /// Edge-preserving [**bilateral**](bilateral) filter (5×5): smooths flat /
    /// sensor noise while keeping edges sharp. The general-purpose default.
    #[default]
    Bilateral,
    /// [**Gaussian**](gaussian) low-pass blur (separable 5×5): smooths
    /// everything, so it softens fine detail along with the noise.
    Gaussian,
    /// [**Median**](median) filter (3×3): best for salt-and-pepper / impulse
    /// noise; also edge-preserving.
    Median,
    /// [**Mean**](mean) (box) blur over a 3×3 window — the cheapest smoother;
    /// blurs noise and detail equally.
    Mean,
    /// [**Non-local means**](nlmeans): averages samples weighted by how similar
    /// their surrounding patch is, so repeating texture denoises without
    /// blurring. Highest classical quality — and by far the slowest.
    Nlmeans,
    /// [**Anisotropic diffusion**](anisotropic) (Perona–Malik): gradient-gated
    /// diffusion — smooths flat regions but stops at edges. Edge-preserving like
    /// bilateral, different character.
    Anisotropic,
}

impl fmt::Display for DenoiseMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            DenoiseMethod::Bilateral => "bilateral",
            DenoiseMethod::Gaussian => "gaussian",
            DenoiseMethod::Median => "median",
            DenoiseMethod::Mean => "mean",
            DenoiseMethod::Nlmeans => "nlmeans",
            DenoiseMethod::Anisotropic => "anisotropic",
        })
    }
}

/// Serde default for [`super::VideoFilter::Denoise::strength`].
#[cfg(feature = "serde")]
pub(super) fn default_denoise_strength() -> f32 {
    0.5
}

// ── ffmpeg-compatible `nlmeans` defaults (serde) ─────────────────────────────
// The values `ffmpeg -h filter=nlmeans` reports. `pc`/`rc` of 0 mean "same as
// the luma parameter", which [`apply_nlmeans`] resolves.

/// Serde default for [`super::VideoFilter::Nlmeans::s`] — ffmpeg's `s=1.0`.
#[cfg(feature = "serde")]
pub(super) fn default_nlmeans_s() -> f32 {
    1.0
}

/// Serde default for [`super::VideoFilter::Nlmeans::p`] — ffmpeg's `p=7`.
#[cfg(feature = "serde")]
pub(super) fn default_nlmeans_p() -> u32 {
    7
}

/// Serde default for [`super::VideoFilter::Nlmeans::r`] — ffmpeg's `r=15`.
#[cfg(feature = "serde")]
pub(super) fn default_nlmeans_r() -> u32 {
    15
}

/// ffmpeg's `s` range: `1.0..=30.0`.
pub(super) const NLMEANS_SIGMA_RANGE: std::ops::RangeInclusive<f32> = 1.0..=30.0;
/// ffmpeg's `p` / `pc` / `r` / `rc` range: `0..=99` (0 = "same as luma" for the
/// chroma pair; sizes are forced odd by the kernel).
pub(super) const NLMEANS_SIZE_MAX: u32 = 99;

/// **Parameterized non-local means**, matching `ffmpeg -vf nlmeans=s=..:p=..:r=..`.
///
/// Unlike [`apply`] — where every method runs at a fixed internal setting and
/// `strength` merely blends — this exposes the algorithm's real knobs: the patch
/// size that defines "similar surroundings", the research window that bounds how
/// far to look for them, and a σ-style strength. Applied at full weight (no
/// blend), to luma + chroma, with the chroma pair falling back to the luma
/// values when `pc` / `rc` are 0. 8-bit `Yuv420p` only.
pub(super) fn apply_nlmeans(
    frame: &VideoFrame,
    s: f32,
    p: u32,
    pc: u32,
    r: u32,
    rc: u32,
) -> Result<VideoFrame> {
    let (yp, up, vp) = planes_8bit(frame, "nlmeans")?;
    let (w, h) = (frame.width as usize, frame.height as usize);
    let (cw, ch) = (w / 2, h / 2);
    let chroma_patch = if pc == 0 { p } else { pc };
    let chroma_research = if rc == 0 { r } else { rc };
    Ok(assemble(
        frame,
        frame.width,
        frame.height,
        nlmeans::plane_params(&yp, w, h, p, r, s),
        nlmeans::plane_params(&up, cw, ch, chroma_patch, chroma_research, s),
        nlmeans::plane_params(&vp, cw, ch, chroma_patch, chroma_research, s),
    ))
}

/// Denoise luma + chroma with `method`, blending by `strength`.
pub(super) fn apply(frame: &VideoFrame, method: DenoiseMethod, strength: f32) -> Result<VideoFrame> {
    let (yp, up, vp) = planes_8bit(frame, "denoise")?;
    let s = strength.clamp(0.0, 1.0);
    let (w, h) = (frame.width as usize, frame.height as usize);
    let (cw, ch) = (w / 2, h / 2);
    Ok(assemble(
        frame,
        frame.width,
        frame.height,
        plane(method, &yp, w, h, s),
        plane(method, &up, cw, ch, s),
        plane(method, &vp, cw, ch, s),
    ))
}

/// Denoise one 8-bit plane with `method`, then blend the filtered plane back
/// with the source by `strength` (`0` ⇒ source, `1` ⇒ fully filtered). `strength
/// == 0` and degenerate sizes short-circuit to a copy.
fn plane(method: DenoiseMethod, src: &[u8], w: usize, h: usize, strength: f32) -> Vec<u8> {
    if w == 0 || h == 0 || strength <= 0.0 {
        return src.to_vec();
    }
    let filtered = match method {
        DenoiseMethod::Bilateral => bilateral::plane(src, w, h),
        DenoiseMethod::Gaussian => gaussian::plane(src, w, h),
        DenoiseMethod::Median => median::plane(src, w, h),
        DenoiseMethod::Mean => mean::plane(src, w, h),
        DenoiseMethod::Nlmeans => nlmeans::plane(src, w, h),
        DenoiseMethod::Anisotropic => anisotropic::plane(src, w, h),
    };
    if strength >= 1.0 {
        return filtered;
    }
    blend(src, &filtered, w, strength, Tier::detect())
}

/// `src·(1−strength) + filtered·strength`, rounded, per sample.
fn blend(src: &[u8], filtered: &[u8], w: usize, strength: f32, tier: Tier) -> Vec<u8> {
    let mut out = vec![0u8; src.len()];
    for_row_bands(&mut out, w, 512, |y0, rows| {
        let n = rows.len();
        blend_row(tier, &src[y0 * w..][..n], &filtered[y0 * w..][..n], strength, rows);
    });
    out
}

/// The blend, scalar — the reference.
fn blend_scalar(src: &[u8], filtered: &[u8], strength: f32, out: &mut [u8]) {
    let inv = 1.0 - strength;
    for ((o, &s), &f) in out.iter_mut().zip(src).zip(filtered) {
        *o = (s as f32 * inv + f as f32 * strength).round().clamp(0.0, 255.0) as u8;
    }
}

#[cfg_attr(not(any(target_arch = "x86", target_arch = "x86_64")), allow(dead_code))]
#[inline(always)]
unsafe fn blend_body<S: Simd>(src: &[u8], filtered: &[u8], strength: f32, out: &mut [u8]) {
    unsafe {
        let inv = S::set1_f32(1.0 - strength);
        let st = S::set1_f32(strength);
        let mut x = 0;
        while x + S::LANES <= out.len() {
            let s = S::load_u8_f32(src.as_ptr().add(x));
            let f = S::load_u8_f32(filtered.as_ptr().add(x));
            let v = S::add_f32(S::mul_f32(s, inv), S::mul_f32(f, st));
            S::store_f32_u8(out.as_mut_ptr().add(x), round_clamp_u8::<S>(v));
            x += S::LANES;
        }
        blend_scalar(&src[x..], &filtered[x..], strength, &mut out[x..]);
    }
}

tiered!(fn blend_row(src: &[u8], filtered: &[u8], strength: f32, out: &mut [u8]) => blend_body, scalar blend_scalar);

// ── row bands across cores ───────────────────────────────────────────────────

/// How many threads the denoise kernels may use: the host's parallelism,
/// capped by `RIVET_DENOISE_THREADS` (so a timing run can hold threads at one
/// while it varies the SIMD tier, and vice versa). Read once per process.
pub(super) fn max_threads() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| {
        let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
        match std::env::var("RIVET_DENOISE_THREADS").ok().and_then(|v| v.trim().parse::<usize>().ok()) {
            Some(n) if n >= 1 => n,
            _ => cores,
        }
    })
}

/// Run `f(y0, rows)` over horizontal bands of a `w`-wide plane held in
/// `out`, one band per thread, when the plane is tall enough for the split to
/// pay. `min_band_rows` is the caller's statement of that: a thread costs
/// tens of microseconds to start, so a kernel that spends nanoseconds per
/// sample (mean, gaussian) wants hundreds of rows per band where the
/// bilateral wants dozens. Bands write disjoint rows, so there is nothing to
/// synchronise; every caller computes each sample from the source alone, so
/// the split cannot change the result.
pub(super) fn for_row_bands<T: Send>(
    out: &mut [T],
    w: usize,
    min_band_rows: usize,
    f: impl Fn(usize, &mut [T]) + Sync,
) {
    let h = if w == 0 { 0 } else { out.len() / w };
    let bands = max_threads().min(h / min_band_rows.max(1)).max(1);
    if bands == 1 {
        return f(0, out);
    }
    let rows_per_band = h.div_ceil(bands);
    std::thread::scope(|scope| {
        for (i, chunk) in out.chunks_mut(rows_per_band * w).enumerate() {
            let f = &f;
            scope.spawn(move || f(i * rows_per_band, chunk));
        }
    });
}

/// Clamp `v` to `0..hi` (edge-replicate border addressing). Shared by the
/// method kernels that use a clamped window.
pub(super) fn clamp_idx(v: isize, hi: usize) -> usize {
    v.clamp(0, hi as isize - 1) as usize
}

#[cfg(test)]
mod tests {
    use super::test_support::planes;
    use super::*;

    #[test]
    fn the_blend_matches_the_scalar_reference_at_every_tier() {
        for (w, h, src) in planes() {
            let filtered = median::plane(&src, w, h);
            for strength in [0.1f32, 0.25, 0.5, 0.8, 0.99] {
                let want = blend(&src, &filtered, w, strength, Tier::Scalar);
                let mut reference = vec![0u8; src.len()];
                blend_scalar(&src, &filtered, strength, &mut reference);
                assert_eq!(want, reference);
                for tier in Tier::available() {
                    assert_eq!(blend(&src, &filtered, w, strength, tier), want, "{tier:?} diverged at {w}x{h} s={strength}");
                }
            }
        }
    }
}

#[cfg(test)]
pub(super) mod test_support {
    //! Inputs the per-kernel bit-exactness tests share: random planes at
    //! widths on and off every lane multiple (so every scalar tail runs),
    //! degenerate sizes, and the patterns a kernel is most likely to get
    //! wrong at — flat, extremes, impulses, hard edges.

    /// Deterministic pseudo-noise.
    pub(crate) fn noisy(n: usize, seed: u32) -> Vec<u8> {
        let mut s = seed ^ 0x2545_F491;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((s >> 16) & 0xFF) as u8
            })
            .collect()
    }

    /// `(w, h, plane)` cases.
    pub(crate) fn planes() -> Vec<(usize, usize, Vec<u8>)> {
        let sizes = [
            (1, 1), (2, 2), (3, 1), (1, 3), (5, 7), (8, 8), (9, 9), (15, 4), (16, 16), (17, 3),
            (31, 5), (33, 34), (64, 2), (70, 9), (100, 70), (129, 3),
        ];
        let mut out = Vec::new();
        for (i, &(w, h)) in sizes.iter().enumerate() {
            out.push((w, h, noisy(w * h, i as u32)));
        }
        // Patterns.
        let (w, h) = (40, 21);
        out.push((w, h, vec![100u8; w * h]));
        out.push((w, h, vec![0u8; w * h]));
        out.push((w, h, vec![255u8; w * h]));
        out.push((w, h, (0..w * h).map(|i| if (i / w + i % w) % 2 == 0 { 0 } else { 255 }).collect()));
        out.push((w, h, (0..w * h).map(|i| if i % w < w / 2 { 30 } else { 220 }).collect()));
        out.push((w, h, (0..w * h).map(|i| (i % 256) as u8).collect()));
        let mut impulses = vec![128u8; w * h];
        for i in (0..w * h).step_by(7) {
            impulses[i] = if i % 2 == 0 { 255 } else { 0 };
        }
        out.push((w, h, impulses));
        out
    }
}
