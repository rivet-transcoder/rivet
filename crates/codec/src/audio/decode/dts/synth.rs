//! Reconstruction filters of the DTS core: the 32-band QMF synthesis bank
//! (ETSI TS 102 114 Annex C.3.6) and the LFE interpolation FIR (C.3.7).
//!
//! Both are written to follow the annex pseudocode line by line, with the
//! per-channel state (`raX`, `raZ`, the LFE history) held in the structs so
//! it persists across subframes and frames as the pseudocode assumes.

use std::sync::LazyLock;

use super::tables;

/// Number of subbands in the core filter bank.
pub const NUM_SUBBANDS: usize = 32;

/// The `rScale` of `QMFInterpolation`, which C.3.6 applies to every output
/// sample but never assigns. The structure as printed — post-scaling terms of
/// `0.25 / (2 cos)` and a prototype whose 512 taps sum to 1 — passes a
/// constant on subband 0 through at 1/(128·√2) of its value, while the
/// subband samples are already on the PCM scale (the D.1 scale factors top
/// out at 2^23, the LFE takes the same tables with no such factor). So the
/// reconstruction is unity-gain only with this factor, and it is, to four
/// decimals, the gain libavcodec's output has over the unscaled structure
/// (`tests/dts_core.rs`).
const RECONSTRUCTION_GAIN: f64 = 128.0 * std::f64::consts::SQRT_2;

/// `PreCalCosMod()` of Annex C.3.6: 16×16 + 16×16 cosine terms followed by
/// the 16 + 16 post-scaling terms, in the order the interpolation consumes
/// them (`raCosMod[j++]`).
static COS_MOD: LazyLock<[f64; 544]> = LazyLock::new(|| {
    use std::f64::consts::PI;
    let mut m = [0.0f64; 544];
    let mut j = 0;
    for k in 0..16 {
        for i in 0..16 {
            m[j] = (((2 * i + 1) * (2 * k + 1)) as f64 * PI / 64.0).cos();
            j += 1;
        }
    }
    for k in 0..16 {
        for i in 0..16 {
            m[j] = ((i * (2 * k + 1)) as f64 * PI / 32.0).cos();
            j += 1;
        }
    }
    for k in 0..16 {
        m[j] = 0.25 / (2.0 * ((2 * k + 1) as f64 * PI / 128.0).cos());
        j += 1;
    }
    for k in 0..16 {
        m[j] = -0.25 / (2.0 * ((2 * k + 1) as f64 * PI / 128.0).sin());
        j += 1;
    }
    m
});

/// One primary channel's QMF synthesis state.
pub struct Qmf {
    /// `raX`: the 512-sample modulated history, newest 32 first.
    x: [f64; 512],
    /// `raZ`: the 64-sample overlap accumulator; the first 32 are output,
    /// the second 32 carry into the next block.
    z: [f64; 64],
}

impl Default for Qmf {
    fn default() -> Self {
        Self::new()
    }
}

impl Qmf {
    pub fn new() -> Self {
        Self {
            x: [0.0; 512],
            z: [0.0; 64],
        }
    }

    /// `QMFInterpolation` for one subband sample vector: 32 subband samples
    /// in, 32 PCM samples out on the same scale (see [`RECONSTRUCTION_GAIN`]).
    /// `perfect` selects the FILTS == 1 prototype.
    pub fn synthesize(&mut self, xin: &[f64; NUM_SUBBANDS], perfect: bool, out: &mut [f64; 32]) {
        let coeff: &[f32; 512] = if perfect {
            &tables::QMF_FIR_PERFECT
        } else {
            &tables::QMF_FIR_NON_PERFECT
        };
        let cm = &*COS_MOD;

        // Cosine modulation → SUM / DIFF.
        let mut a = [0.0f64; 16];
        let mut b = [0.0f64; 16];
        let mut j = 0;
        for ak in a.iter_mut() {
            for i in 0..16 {
                *ak += (xin[2 * i] + xin[2 * i + 1]) * cm[j];
                j += 1;
            }
        }
        for bk in b.iter_mut() {
            for i in 0..16 {
                let v = if i > 0 { xin[2 * i] + xin[2 * i - 1] } else { xin[0] };
                *bk += v * cm[j];
                j += 1;
            }
        }
        // Store history: the new 32 entries of raX.
        for k in 0..16 {
            self.x[k] = cm[j] * (a[k] + b[k]);
            j += 1;
        }
        for k in 0..16 {
            self.x[32 - k - 1] = cm[j] * (a[k] - b[k]);
            j += 1;
        }
        debug_assert_eq!(j, 544);

        // Multiply by the prototype filter (8 taps of 64 per output).
        for i in 0..32 {
            let k = 31 - i;
            let mut acc = 0.0f64;
            let mut acc2 = 0.0f64;
            for jj in (0..512).step_by(64) {
                acc += coeff[i + jj] as f64 * (self.x[i + jj] - self.x[jj + k]);
                acc2 += coeff[32 + i + jj] as f64 * (-self.x[i + jj] - self.x[jj + k]);
            }
            self.z[i] += acc;
            self.z[32 + i] += acc2;
        }
        for (o, z) in out.iter_mut().zip(&self.z[..32]) {
            *o = z * RECONSTRUCTION_GAIN;
        }

        // Update working arrays.
        self.x.copy_within(0..480, 32);
        self.z.copy_within(32..64, 0);
        self.z[32..].fill(0.0);
    }
}

/// The LFE channel's interpolation state: the last `512 / factor - 1`
/// decimated samples, newest last.
pub struct LfeInterp {
    hist: [f64; 8],
}

impl Default for LfeInterp {
    fn default() -> Self {
        Self::new()
    }
}

impl LfeInterp {
    pub fn new() -> Self {
        Self { hist: [0.0; 8] }
    }

    /// `InterpolationFIR`: each decimated sample yields `factor` (64 or 128)
    /// PCM samples, appended to `out`.
    pub fn interpolate(&mut self, decimated: &[f64], factor: usize, out: &mut Vec<f64>) {
        debug_assert!(factor == 64 || factor == 128);
        let coeff: &[f32; 512] = if factor == 128 {
            &tables::LFE_FIR_128X
        } else {
            &tables::LFE_FIR_64X
        };
        let taps = 512 / factor; // 4 or 8 decimated samples per output
        for &s in decimated {
            // hist[7] is the newest sample (rLFE[n]), hist[6] is rLFE[n-1], …
            self.hist.copy_within(1..8, 0);
            self.hist[7] = s;
            for k in 0..factor {
                let mut acc = 0.0f64;
                for j in 0..taps {
                    acc += self.hist[7 - j] * coeff[k + j * factor] as f64;
                }
                out.push(acc);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The two prototypes are low-pass filters of a 32-band bank: a constant
    /// on subband 0 must come out as that constant, once the structure is
    /// scaled by [`RECONSTRUCTION_GAIN`]. This pins the overall gain of the
    /// implementation and would fail for a shifted or transposed table.
    #[test]
    fn dc_on_subband_zero_comes_out_as_dc() {
        for perfect in [false, true] {
            let mut q = Qmf::new();
            let mut xin = [0.0f64; 32];
            xin[0] = 1000.0;
            let mut out = [0.0f64; 32];
            let mut last = Vec::new();
            for _ in 0..40 {
                q.synthesize(&xin, perfect, &mut out);
                last = out.to_vec();
            }
            let mean = last.iter().sum::<f64>() / 32.0;
            let ripple = last.iter().map(|v| (v - mean).abs()).fold(0.0, f64::max);
            assert!(
                (mean - 1000.0).abs() < 1.0,
                "perfect={perfect}: DC gain should be unity on the subband scale, got mean {mean}"
            );
            // A lone constant on subband 0 is not what the analysis bank
            // would produce for DC PCM (the aliasing terms in the other
            // subbands are missing), so the output is not perfectly flat:
            // the non-perfect prototype ripples < 0.1 %, the perfect one
            // 0.33 %. A shifted or transposed table gives tens of percent.
            assert!(ripple < 10.0, "perfect={perfect}: ripple {ripple} on a DC input");
        }
    }

    /// Silence in, silence out — and the accumulator carry does not leak.
    #[test]
    fn zero_input_is_exactly_zero() {
        let mut q = Qmf::new();
        let xin = [0.0f64; 32];
        let mut out = [1.0f64; 32];
        for _ in 0..20 {
            q.synthesize(&xin, true, &mut out);
        }
        assert!(out.iter().all(|v| *v == 0.0));
    }

    /// A constant decimated LFE stream interpolates to that constant: the
    /// polyphase sums of both FIRs are unity (the 64× taps sum to 64).
    #[test]
    fn lfe_dc_gain_is_unity() {
        for factor in [64usize, 128] {
            let mut l = LfeInterp::new();
            let mut out = Vec::new();
            l.interpolate(&[500.0; 16], factor, &mut out);
            assert_eq!(out.len(), 16 * factor);
            let tail = &out[out.len() - factor..];
            for v in tail {
                assert!((v - 500.0).abs() < 0.5, "factor {factor}: {v}");
            }
        }
    }
}
