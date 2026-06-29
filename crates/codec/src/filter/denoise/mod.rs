//! `denoise` — spatial denoise with a **selectable algorithm** ([`DenoiseMethod`])
//! and a `strength` (`0.0..=1.0`) that blends the filtered result back with the
//! source. Each method lives in its own file; they share the dispatch + blend
//! here. 8-bit `Yuv420p` only (luma + chroma).
//!
//! `strength` is a uniform "how much" dial: every method runs at a fixed,
//! moderate internal setting and the output is `src·(1−s) + filtered·s`, so the
//! same number means the same amount of denoising regardless of algorithm.

use std::fmt;

use anyhow::Result;

use super::{assemble, planes_8bit};
use crate::frame::VideoFrame;

mod anisotropic;
mod bilateral;
mod gaussian;
mod mean;
mod median;
mod nlmeans;

/// Which spatial denoise algorithm [`super::VideoFilter::Denoise`] runs. Each
/// suits a different kind of noise; `strength` then blends the result with the
/// source. (Temporal denoisers — hqdn3d / NLM-temporal — need frame history and
/// don't fit this stateless per-frame filter; a future extension.)
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
    let inv = 1.0 - strength;
    src.iter()
        .zip(&filtered)
        .map(|(&s, &f)| (s as f32 * inv + f as f32 * strength).round().clamp(0.0, 255.0) as u8)
        .collect()
}

/// Clamp `v` to `0..hi` (edge-replicate border addressing). Shared by the
/// method kernels that use a clamped window.
pub(super) fn clamp_idx(v: isize, hi: usize) -> usize {
    v.clamp(0, hi as isize - 1) as usize
}
