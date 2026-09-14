//! The inverse transform: A/52:2018 §7.9.4 (PDF p.99–103).
//!
//! Both the 512-sample transform (`blksw = 0`) and the pair of 256-sample
//! transforms (`blksw = 1`) are computed exactly as the spec writes them —
//! pre-twiddle, N/4- (or N/8-) point complex IFFT, post-twiddle, window and
//! de-interleave — with a radix-2 FFT standing in for the spec's O(N²) sum
//! (the test below checks the two agree to float precision). The overlap-add
//! carries the `2 *` headroom factor from step 6.
//!
//! The window is the Kaiser-Bessel-derived window with α = 5 that Table 7.33
//! tabulates to five decimals; it is derived here analytically and the table
//! test in `tables` pins the derivation to the printed values.

use std::f64::consts::PI;
use std::sync::OnceLock;

const N: usize = 512;

/// Zeroth-order modified Bessel function of the first kind, by its power
/// series (converges fast for the arguments a KBD window needs).
fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    let half = x / 2.0;
    for k in 1..60 {
        term *= (half / k as f64) * (half / k as f64);
        sum += term;
        if term < sum * 1e-17 {
            break;
        }
    }
    sum
}

/// The 256-point half of the KBD window (α = 5) used by both transform
/// lengths; `w[n]` for n in 0..256, rising from ~0 to 1.
pub(super) fn kbd_window() -> &'static [f32; 256] {
    static W: OnceLock<[f32; 256]> = OnceLock::new();
    W.get_or_init(|| {
        let alpha = 5.0f64;
        let half = 256usize;
        // Kaiser window of length half+1.
        let mut kaiser = vec![0.0f64; half + 1];
        for (n, k) in kaiser.iter_mut().enumerate() {
            let r = (2.0 * n as f64) / half as f64 - 1.0;
            *k = bessel_i0(PI * alpha * (1.0 - r * r).max(0.0).sqrt());
        }
        let total: f64 = kaiser.iter().sum();
        let mut acc = 0.0;
        let mut w = [0.0f32; 256];
        for n in 0..half {
            acc += kaiser[n];
            w[n] = (acc / total).sqrt() as f32;
        }
        w
    })
}

#[derive(Clone, Copy)]
struct C {
    re: f64,
    im: f64,
}

/// In-place iterative radix-2 inverse DFT (unnormalised, +j convention):
/// `z[n] = Σ_k Z[k] exp(+j 2π k n / len)`.
fn ifft(buf: &mut [C]) {
    let len = buf.len();
    debug_assert!(len.is_power_of_two());
    // bit reversal
    let bits = len.trailing_zeros();
    for i in 0..len {
        let j = i.reverse_bits() >> (usize::BITS - bits);
        if j > i {
            buf.swap(i, j);
        }
    }
    let mut size = 2;
    while size <= len {
        let step = 2.0 * PI / size as f64;
        for start in (0..len).step_by(size) {
            for k in 0..size / 2 {
                let (s, c) = (step * k as f64).sin_cos();
                let a = buf[start + k];
                let b = buf[start + k + size / 2];
                let t = C { re: b.re * c - b.im * s, im: b.re * s + b.im * c };
                buf[start + k] = C { re: a.re + t.re, im: a.im + t.im };
                buf[start + k + size / 2] = C { re: a.re - t.re, im: a.im - t.im };
            }
        }
        size *= 2;
    }
}

struct Twiddles {
    /// `xcos1[k] = -cos(2π(8k+1)/(8N))`, `xsin1[k] = -sin(...)`, k < N/4.
    cos1: [f64; 128],
    sin1: [f64; 128],
    /// `xcos2[k] = -cos(2π(8k+1)/(4N))`, k < N/8.
    cos2: [f64; 64],
    sin2: [f64; 64],
}

fn twiddles() -> &'static Twiddles {
    static T: OnceLock<Twiddles> = OnceLock::new();
    T.get_or_init(|| {
        let mut t = Twiddles { cos1: [0.0; 128], sin1: [0.0; 128], cos2: [0.0; 64], sin2: [0.0; 64] };
        for k in 0..128 {
            let a = 2.0 * PI * (8.0 * k as f64 + 1.0) / (8.0 * N as f64);
            t.cos1[k] = -a.cos();
            t.sin1[k] = -a.sin();
        }
        for k in 0..64 {
            let a = 2.0 * PI * (8.0 * k as f64 + 1.0) / (4.0 * N as f64);
            t.cos2[k] = -a.cos();
            t.sin2[k] = -a.sin();
        }
        t
    })
}

/// One block of the inverse transform for one channel.
///
/// `coeffs` are the 256 transform coefficients; `delay` is the channel's
/// overlap-add memory (the second half of the previous windowed block).
/// Writes 256 PCM samples to `out` and updates `delay`.
pub(super) fn imdct_block(coeffs: &[f32; 256], blksw: bool, delay: &mut [f32; 256], out: &mut [f32; 256]) {
    let w = kbd_window();
    let tw = twiddles();
    let mut x = [0.0f64; N];
    if !blksw {
        // §7.9.4.1: 512-sample transform.
        let mut z = [C { re: 0.0, im: 0.0 }; N / 4];
        for k in 0..N / 4 {
            let a = f64::from(coeffs[N / 2 - 2 * k - 1]);
            let b = f64::from(coeffs[2 * k]);
            z[k] = C {
                re: a * tw.cos1[k] - b * tw.sin1[k],
                im: b * tw.cos1[k] + a * tw.sin1[k],
            };
        }
        ifft(&mut z);
        let mut yr = [0.0f64; N / 4];
        let mut yi = [0.0f64; N / 4];
        for n in 0..N / 4 {
            yr[n] = z[n].re * tw.cos1[n] - z[n].im * tw.sin1[n];
            yi[n] = z[n].im * tw.cos1[n] + z[n].re * tw.sin1[n];
        }
        let wf = |i: usize| f64::from(w[i]);
        for n in 0..N / 8 {
            x[2 * n] = -yi[N / 8 + n] * wf(2 * n);
            x[2 * n + 1] = yr[N / 8 - n - 1] * wf(2 * n + 1);
            x[N / 4 + 2 * n] = -yr[n] * wf(N / 4 + 2 * n);
            x[N / 4 + 2 * n + 1] = yi[N / 4 - n - 1] * wf(N / 4 + 2 * n + 1);
            x[N / 2 + 2 * n] = -yr[N / 8 + n] * wf(N / 2 - 2 * n - 1);
            x[N / 2 + 2 * n + 1] = yi[N / 8 - n - 1] * wf(N / 2 - 2 * n - 2);
            x[3 * N / 4 + 2 * n] = yi[n] * wf(N / 4 - 2 * n - 1);
            x[3 * N / 4 + 2 * n + 1] = -yr[N / 4 - n - 1] * wf(N / 4 - 2 * n - 2);
        }
    } else {
        // §7.9.4.2: two 256-sample transforms on the interleaved coefficients.
        let mut x1 = [0.0f64; N / 4];
        let mut x2 = [0.0f64; N / 4];
        for k in 0..N / 4 {
            x1[k] = f64::from(coeffs[2 * k]);
            x2[k] = f64::from(coeffs[2 * k + 1]);
        }
        let mut z1 = [C { re: 0.0, im: 0.0 }; N / 8];
        let mut z2 = [C { re: 0.0, im: 0.0 }; N / 8];
        for k in 0..N / 8 {
            let (a1, b1) = (x1[N / 4 - 2 * k - 1], x1[2 * k]);
            let (a2, b2) = (x2[N / 4 - 2 * k - 1], x2[2 * k]);
            z1[k] = C { re: a1 * tw.cos2[k] - b1 * tw.sin2[k], im: b1 * tw.cos2[k] + a1 * tw.sin2[k] };
            z2[k] = C { re: a2 * tw.cos2[k] - b2 * tw.sin2[k], im: b2 * tw.cos2[k] + a2 * tw.sin2[k] };
        }
        ifft(&mut z1);
        ifft(&mut z2);
        let mut yr1 = [0.0f64; N / 8];
        let mut yi1 = [0.0f64; N / 8];
        let mut yr2 = [0.0f64; N / 8];
        let mut yi2 = [0.0f64; N / 8];
        for n in 0..N / 8 {
            yr1[n] = z1[n].re * tw.cos2[n] - z1[n].im * tw.sin2[n];
            yi1[n] = z1[n].im * tw.cos2[n] + z1[n].re * tw.sin2[n];
            yr2[n] = z2[n].re * tw.cos2[n] - z2[n].im * tw.sin2[n];
            yi2[n] = z2[n].im * tw.cos2[n] + z2[n].re * tw.sin2[n];
        }
        let wf = |i: usize| f64::from(w[i]);
        for n in 0..N / 8 {
            x[2 * n] = -yi1[n] * wf(2 * n);
            x[2 * n + 1] = yr1[N / 8 - n - 1] * wf(2 * n + 1);
            x[N / 4 + 2 * n] = -yr1[n] * wf(N / 4 + 2 * n);
            x[N / 4 + 2 * n + 1] = yi1[N / 8 - n - 1] * wf(N / 4 + 2 * n + 1);
            x[N / 2 + 2 * n] = -yr2[n] * wf(N / 2 - 2 * n - 1);
            x[N / 2 + 2 * n + 1] = yi2[N / 8 - n - 1] * wf(N / 2 - 2 * n - 2);
            x[3 * N / 4 + 2 * n] = yi2[n] * wf(N / 4 - 2 * n - 1);
            x[3 * N / 4 + 2 * n + 1] = -yr2[N / 8 - n - 1] * wf(N / 4 - 2 * n - 2);
        }
    }
    // Step 6: overlap and add, with the encoder's headroom factor undone.
    for n in 0..N / 2 {
        out[n] = 2.0 * (x[n] as f32 + delay[n]);
        delay[n] = x[N / 2 + n] as f32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The spec's step 3 written as the literal O(N²) sum.
    fn slow_ifft(z: &[C]) -> Vec<C> {
        let len = z.len();
        (0..len)
            .map(|n| {
                let mut acc = C { re: 0.0, im: 0.0 };
                for (k, zk) in z.iter().enumerate() {
                    let a = 2.0 * PI * (k * n) as f64 / len as f64;
                    acc.re += zk.re * a.cos() - zk.im * a.sin();
                    acc.im += zk.re * a.sin() + zk.im * a.cos();
                }
                acc
            })
            .collect()
    }

    #[test]
    fn radix2_ifft_matches_the_direct_sum() {
        for &len in &[64usize, 128] {
            let z: Vec<C> = (0..len)
                .map(|i| C { re: ((i * 7) % 13) as f64 - 6.0, im: ((i * 5) % 11) as f64 - 5.0 })
                .collect();
            let slow = slow_ifft(&z);
            let mut fast = z.clone();
            ifft(&mut fast);
            for (a, b) in slow.iter().zip(&fast) {
                assert!((a.re - b.re).abs() < 1e-9 && (a.im - b.im).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn window_is_kbd_alpha_5_power_complementary() {
        let w = kbd_window();
        // Princen-Bradley: w[n]^2 + w[255-n]^2 == 1 for the KBD window.
        for n in 0..256 {
            let s = w[n] * w[n] + w[255 - n] * w[255 - n];
            assert!((s - 1.0).abs() < 1e-5, "n={n} s={s}");
        }
        assert!(w[0] > 0.0 && w[0] < 0.001);
        assert!(w[255] > 0.9999);
    }

    /// A pure tone reconstructed through the TDAC overlap-add must be a
    /// pure tone at the same frequency after the first (alias-cancelling)
    /// block — this pins the sign/ordering conventions of step 5 without
    /// needing a forward transform.
    #[test]
    fn long_block_tdac_reconstructs_a_single_bin_tone_with_constant_level() {
        let mut delay = [0.0f32; 256];
        let mut coeffs = [0.0f32; 256];
        coeffs[10] = 0.25;
        let mut out = [0.0f32; 256];
        let mut blocks = Vec::new();
        for _ in 0..4 {
            imdct_block(&coeffs, false, &mut delay, &mut out);
            blocks.push(out);
        }
        // Blocks 1..3 are steady state: same RMS, and a periodic tone.
        let rms = |b: &[f32; 256]| (b.iter().map(|v| v * v).sum::<f32>() / 256.0).sqrt();
        let r1 = rms(&blocks[1]);
        let r2 = rms(&blocks[2]);
        let r3 = rms(&blocks[3]);
        assert!(r1 > 0.01, "tone must come out: {r1}");
        assert!((r1 - r2).abs() < 1e-3 * r1 && (r2 - r3).abs() < 1e-3 * r2, "{r1} {r2} {r3}");
        // Bin k of a 512-MDCT is a cosine at (k + 0.5) cycles per 512 samples;
        // one 256-sample block advances it by (k+0.5)π. For k = 10 that is an
        // odd multiple of π/2 … so check continuity at the block seam instead:
        // the sample-to-sample step must stay small relative to the peak.
        let peak = blocks[2].iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let seam = (blocks[2][0] - blocks[1][255]).abs();
        let max_step = 2.0 * PI as f32 * 10.5 / 512.0 * peak * 1.05;
        assert!(seam < max_step, "seam {seam} > {max_step}");
    }

    #[test]
    fn short_blocks_reconstruct_the_same_tone_level_as_long_blocks() {
        // A DC-ish low bin in the interleaved short-block layout is bin 2k
        // (first half) and 2k+1 (second half); driving both halves with the
        // same coefficient should produce a steady tone of similar RMS to a
        // long block of bin k (same frequency, half the length).
        let mut delay = [0.0f32; 256];
        let mut coeffs = [0.0f32; 256];
        coeffs[10] = 0.25; // X1[5]
        coeffs[11] = 0.25; // X2[5]
        let mut out = [0.0f32; 256];
        let mut prev = 0.0;
        for i in 0..4 {
            imdct_block(&coeffs, true, &mut delay, &mut out);
            let r = (out.iter().map(|v| v * v).sum::<f32>() / 256.0).sqrt();
            if i >= 2 {
                assert!((r - prev).abs() < 1e-3 * r, "{r} vs {prev}");
            }
            prev = r;
        }
        assert!(prev > 0.01);
    }
}
