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

    Some(Score {
        psnr: psnr_8bit(a, b),
        ssim: ssim_8bit(a, b, w, h),
    })
}

/// How much picture a frame actually contains.
///
/// A frame's *content* is a separate question from its fidelity, and the two
/// get confused: a black frame encodes to almost nothing at every setting and
/// scores near-perfectly against itself, so a sweep taken across one reports
/// that the content is free and needs no bits. It is not a measurement of the
/// clip; it is a measurement of the fade-in.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Activity {
    /// Mean luma, `0..=255`.
    pub mean: f64,
    /// Standard deviation of luma — the useful half. A frame can be bright and
    /// still carry nothing (a white title card); what makes content *content*
    /// is variation.
    pub std_dev: f64,
}

/// Luma black in TV range (`16..=235`), which is what almost every decoded
/// frame is in. Full-range black is 0, so testing against 0 finds nothing.
pub const TV_BLACK: f64 = 16.0;

impl Activity {
    /// Whether this frame is too close to blank to measure anything from.
    ///
    /// Deliberately generous on brightness and strict on variation: a dim
    /// night scene is real content and must not be rejected, while a frame
    /// with almost no variation carries no detail for an encoder to spend
    /// bits on however bright it is.
    ///
    /// The thresholds are a default, not a rule — `mean` and `std_dev` are
    /// public so a caller with a different definition of "worth sampling" can
    /// use its own without reimplementing the statistics.
    pub fn looks_blank(&self) -> bool {
        const NEAR_BLACK: f64 = TV_BLACK + 6.0;
        const FLAT: f64 = 3.0;

        self.std_dev < FLAT || (self.mean <= NEAR_BLACK && self.std_dev < FLAT * 2.0)
    }
}

/// Measure how much picture a frame carries.
///
/// Luma only, and over the whole plane: this answers "is there anything here",
/// which does not need windowing.
pub fn luma_activity(frame: &VideoFrame) -> Activity {
    let (w, h) = (frame.width as usize, frame.height as usize);
    let Some(luma) = w.checked_mul(h).and_then(|n| frame.data.get(..n)) else {
        return Activity {
            mean: 0.0,
            std_dev: 0.0,
        };
    };
    if luma.is_empty() {
        return Activity {
            mean: 0.0,
            std_dev: 0.0,
        };
    }

    let n = luma.len() as f64;
    let mean = luma.iter().map(|v| f64::from(*v)).sum::<f64>() / n;
    let variance = luma
        .iter()
        .map(|v| {
            let d = f64::from(*v) - mean;
            d * d
        })
        .sum::<f64>()
        / n;

    Activity {
        mean,
        std_dev: variance.sqrt(),
    }
}

/// Average activity across a window of frames.
///
/// Useful for reporting what a window looked like. It is deliberately **not**
/// the way to decide whether a window is worth sampling — see
/// [`blank_fraction`] for why averaging is the wrong aggregate for that.
/// Returns `None` for an empty window.
pub fn window_activity(frames: &[VideoFrame]) -> Option<Activity> {
    if frames.is_empty() {
        return None;
    }
    let n = frames.len() as f64;
    let (mean, std_dev) = frames.iter().map(luma_activity).fold((0.0, 0.0), |acc, a| {
        (acc.0 + a.mean / n, acc.1 + a.std_dev / n)
    });

    Some(Activity { mean, std_dev })
}

/// What proportion of a window's frames carry no picture, in `0.0..=1.0`.
///
/// # Why this and not the averaged activity
///
/// Averaging conceals the shape of a window. Eight black frames and one busy
/// one average to a respectable standard deviation — the busy frame's variance
/// is spread across the whole window — and the window passes as content while
/// being 89% fade. That is exactly the sample this is meant to reject, and the
/// averaged number says it is fine.
///
/// Counting frames cannot be fooled that way: it asks how much of the window is
/// blank, which is the actual question, and it answers in a unit a caller can
/// set a threshold on without knowing anything about luma.
///
/// `0.0` for an empty window — there are no blank frames in it, and a caller
/// deciding what to do about emptiness has [`window_activity`] returning `None`
/// to work from.
pub fn blank_fraction(frames: &[VideoFrame]) -> f64 {
    if frames.is_empty() {
        return 0.0;
    }
    let blank = frames
        .iter()
        .filter(|frame| luma_activity(frame).looks_blank())
        .count();

    blank as f64 / frames.len() as f64
}

/// Whether a window is too blank to measure an encode against.
///
/// Majority rule: a window more than half fade is not a sample of the content,
/// whatever the other frames contain. The threshold is exposed as
/// [`blank_fraction`] for callers that want a different one.
pub fn window_looks_blank(frames: &[VideoFrame]) -> bool {
    blank_fraction(frames) > 0.5
}

#[cfg(test)]
mod tests;
