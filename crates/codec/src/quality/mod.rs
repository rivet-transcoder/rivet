//! What an encode cost, measured rather than assumed.
//!
//! # Why this is in the crate rather than in a test
//!
//! These implementations lived in `tests/fidelity_psnr_ssim.rs`, where they
//! could gate a regression and nothing else. Every other question — "is this
//! rung worth its bytes", "which quality setting hits 95 on this clip", "did
//! that policy change help" — had to be answered outside the crate, by
//! shelling out to ffmpeg against files already written to a bucket. That
//! works and it is a terrible loop: it needs a whole job to have finished
//! before it can say anything, so the answer arrives minutes later and only
//! for settings somebody already deployed.
//!
//! Scoring belongs next to encoding, because the interesting use is to encode
//! a few frames several ways and compare — which is [`super::bench`].
//!
//! # What these metrics are and are not
//!
//! PSNR and SSIM are pixel-domain. They catch "the encoder regressed by 6 dB",
//! they rank quality settings on one clip reliably, and they are cheap enough
//! to run inside a job. They do **not** model perception: banding, blocking
//! and ringing can be plainly visible at a good score, and a face matters more
//! than the wall behind it to a viewer and not at all to MSE.
//!
//! VMAF is the metric that does model perception, and it is deliberately not
//! here: it is a trained model plus its coefficient file, which means either
//! vendoring libvmaf and its `.json` or reimplementing a neural net. For
//! ranking settings on one piece of content — which is what this module is
//! for — SSIM tracks VMAF closely enough to choose between candidates, and the
//! external harness stays the check on absolute numbers.

use crate::frame::VideoFrame;

/// Mean squared error between two equal-length 8-bit planes.
fn mse(a: &[u8], b: &[u8]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    if a.is_empty() {
        return 0.0;
    }
    let acc: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| {
            let d = f64::from(*x) - f64::from(*y);
            d * d
        })
        .sum();
    acc / a.len() as f64
}

/// Peak signal-to-noise ratio in dB over an 8-bit plane.
///
/// Infinite for identical planes, which is correct and is also why a caller
/// averaging PSNR across frames has to decide what to do with it — see
/// [`Score::psnr`].
pub fn psnr_8bit(a: &[u8], b: &[u8]) -> f64 {
    let m = mse(a, b);
    if m == 0.0 {
        return f64::INFINITY;
    }
    10.0 * (255.0 * 255.0 / m).log10()
}

/// SSIM per Wang et al. 2004, 11x11 Gaussian window, sigma 1.5.
///
/// Constants are the reference ones: `K1 = 0.01`, `K2 = 0.03`, `L = 255`, so
/// `C1 = 6.5025` and `C2 = 58.5225`. The Gaussian is normalised to sum to one,
/// which makes every weighted statistic a true expectation and removes the
/// per-window divide.
///
/// Planes smaller than the window fall back to a single global window. That is
/// a degenerate SSIM and is only meaningful for the tests that exercise this
/// function directly.
pub fn ssim_8bit(a: &[u8], b: &[u8], width: usize, height: usize) -> f64 {
    debug_assert_eq!(a.len(), width * height);
    debug_assert_eq!(b.len(), width * height);

    const WIN: usize = 11;
    if width < WIN || height < WIN {
        return ssim_global(a, b);
    }

    let sigma = 1.5f64;
    let mut kernel = [0f64; WIN];
    let mut total = 0f64;
    for (i, weight) in kernel.iter_mut().enumerate() {
        let x = i as f64 - 5.0;
        *weight = (-(x * x) / (2.0 * sigma * sigma)).exp();
        total += *weight;
    }
    for weight in &mut kernel {
        *weight /= total;
    }

    const C1: f64 = (0.01 * 255.0) * (0.01 * 255.0);
    const C2: f64 = (0.03 * 255.0) * (0.03 * 255.0);

    let rows = height - WIN + 1;
    let cols = width - WIN + 1;
    let mut acc = 0f64;

    for r0 in 0..rows {
        for c0 in 0..cols {
            let (mut mu_a, mut mu_b) = (0f64, 0f64);
            for (dy, ky) in kernel.iter().enumerate() {
                for (dx, kx) in kernel.iter().enumerate() {
                    let w = ky * kx;
                    mu_a += w * f64::from(a[(r0 + dy) * width + c0 + dx]);
                    mu_b += w * f64::from(b[(r0 + dy) * width + c0 + dx]);
                }
            }

            let (mut var_a, mut var_b, mut cov) = (0f64, 0f64, 0f64);
            for (dy, ky) in kernel.iter().enumerate() {
                for (dx, kx) in kernel.iter().enumerate() {
                    let w = ky * kx;
                    let da = f64::from(a[(r0 + dy) * width + c0 + dx]) - mu_a;
                    let db = f64::from(b[(r0 + dy) * width + c0 + dx]) - mu_b;
                    var_a += w * da * da;
                    var_b += w * db * db;
                    cov += w * da * db;
                }
            }

            let num = (2.0 * mu_a * mu_b + C1) * (2.0 * cov + C2);
            let den = (mu_a * mu_a + mu_b * mu_b + C1) * (var_a + var_b + C2);
            acc += num / den;
        }
    }

    acc / (rows * cols) as f64
}

/// Single-window SSIM, for planes too small for the sliding window.
fn ssim_global(a: &[u8], b: &[u8]) -> f64 {
    let n = a.len() as f64;
    if n == 0.0 {
        return 1.0;
    }
    let mu_a = a.iter().map(|v| f64::from(*v)).sum::<f64>() / n;
    let mu_b = b.iter().map(|v| f64::from(*v)).sum::<f64>() / n;

    let (mut var_a, mut var_b, mut cov) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        let da = f64::from(*x) - mu_a;
        let db = f64::from(*y) - mu_b;
        var_a += da * da;
        var_b += db * db;
        cov += da * db;
    }
    var_a /= n;
    var_b /= n;
    cov /= n;

    const C1: f64 = (0.01 * 255.0) * (0.01 * 255.0);
    const C2: f64 = (0.03 * 255.0) * (0.03 * 255.0);
    let num = (2.0 * mu_a * mu_b + C1) * (2.0 * cov + C2);
    let den = (mu_a * mu_a + mu_b * mu_b + C1) * (var_a + var_b + C2);
    num / den
}

/// One comparison's result.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Score {
    /// Luma PSNR in dB.
    ///
    /// `f64::INFINITY` when the planes are identical. A caller averaging over
    /// frames must decide what that means before it poisons the mean — the
    /// usual answer is to cap it, because "this frame was perfect" and "this
    /// clip is perfect" are different claims.
    pub psnr: f64,
    /// Luma SSIM in `[-1, 1]`, and in practice `[0, 1]`.
    pub ssim: f64,
}

/// Score one decoded frame against the reference it was encoded from.
///
/// Luma only, deliberately: chroma is subsampled and its errors are far less
/// visible, so including it flatters the result and blunts exactly the
/// comparison this is used for. `None` when the two frames disagree about
/// their dimensions, which is a caller bug rather than a bad score.
pub fn score_frame(reference: &VideoFrame, decoded: &VideoFrame) -> Option<Score> {
    if reference.width != decoded.width || reference.height != decoded.height {
        return None;
    }
    let (w, h) = (reference.width as usize, reference.height as usize);
    let luma = w.checked_mul(h)?;

    let a = reference.data.get(..luma)?;
    let b = decoded.data.get(..luma)?;

    Some(Score { psnr: psnr_8bit(a, b), ssim: ssim_8bit(a, b, w, h) })
}

#[cfg(test)]
mod tests;
