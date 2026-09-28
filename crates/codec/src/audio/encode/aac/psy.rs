//! Psychoacoustic model and transient detection.
//!
//! The model follows the structure of the informative model in ISO/IEC
//! 13818-7 Annex C.1 — band energies, a tonality estimate, convolution with
//! the Annex's spreading function (C.1.3), a tonality-dependent required SNR
//! (step 8), pre-echo control against the previous long block (step 11,
//! factor 2) and the threshold in quiet — but runs directly on the MDCT
//! spectrum at
//! scalefactor-band resolution instead of on a separate FFT. Tonality comes
//! from the spectral flatness measure of each band, and a tone needs
//! 14.5 + Bark dB of SNR where noise needs 5.5 dB (both after Johnston,
//! "Transform coding of audio signals using perceptual noise criteria",
//! IEEE JSAC 6(2), 1988; the Annex's flat 18 dB for tones is less
//! conservative above ~4 Bark), instead of the Annex's unpredictability
//! measure and 18 / 6 dB. The threshold in quiet is Terhardt's
//! approximation ("Calculating virtual pitch", Hearing Research 1, 1979) with
//! a full-scale sine placed at 96 dB SPL.
//!
//! The transient detector is a plain energy-ratio detector on a high-passed
//! signal, evaluated on 128-sample sub-blocks that coincide with the centres
//! of the eight short windows (see the module docs of `mod.rs` for the
//! timing).

use std::f64::consts::LN_10;

/// Per-band constants for one block length at one sampling rate.
pub(super) struct BandPsy {
    /// Normalised spreading matrix: `spread[src * n + dst]`.
    spread: Vec<f32>,
    /// Centre of each band in Bark.
    bark: Vec<f32>,
    /// Threshold in quiet per band, in MDCT energy units.
    ath: Vec<f32>,
}

/// Bark scale (Zwicker & Terhardt, JASA 68(5), 1980).
fn bark(f: f64) -> f64 {
    13.0 * (0.00076 * f).atan() + 3.5 * (f / 7500.0).powi(2).atan()
}

/// Threshold in quiet in dB SPL (Terhardt 1979), capped at 50 dB so the
/// steep rise above ~16 kHz cannot silence clearly audible content at
/// ordinary listening levels.
fn ath_db(f: f64) -> f64 {
    let k = f.max(20.0) / 1000.0;
    let db = 3.64 * k.powf(-0.8) - 6.5 * (-0.6 * (k - 3.3).powi(2)).exp() + 1e-3 * k.powi(4);
    db.min(50.0)
}

/// The spreading function of Annex C.1.3, from a masker at `i` Bark to a
/// maskee at `j` Bark, as an energy ratio.
fn spreading(i: f64, j: f64) -> f64 {
    let tmpx = if j >= i { 3.0 * (j - i) } else { 1.5 * (j - i) };
    let t = tmpx - 0.5;
    let tmpz = 8.0 * (t * t - 2.0 * t).min(0.0);
    let tmpy = 15.811389 + 7.5 * (tmpx + 0.474) - 17.5 * (1.0 + (tmpx + 0.474).powi(2)).sqrt();
    if tmpy < -100.0 {
        0.0
    } else {
        10f64.powf((tmpz + tmpy) / 10.0)
    }
}

impl BandPsy {
    /// `swb`: band offsets of one window; `m`: its transform length.
    pub fn new(swb: &[u16], m: usize, rate: u32) -> Self {
        let n = swb.len() - 1;
        let hz_per_line = f64::from(rate) / (2.0 * m as f64);
        let barks: Vec<f64> = (0..n)
            .map(|b| bark((f64::from(swb[b]) + f64::from(swb[b + 1])) * 0.5 * hz_per_line))
            .collect();
        let mut spread = vec![0.0f32; n * n];
        for dst in 0..n {
            let norm: f64 = (0..n).map(|src| spreading(barks[src], barks[dst])).sum();
            for src in 0..n {
                spread[src * n + dst] = (spreading(barks[src], barks[dst]) / norm) as f32;
            }
        }
        // A sine of amplitude A (16-bit units) carries about (m * A)^2 of
        // energy in this transform's scale; full scale (A = 32768) is 96 dB SPL.
        let full_scale = (m as f64 * 32768.0).powi(2);
        let ath = (0..n)
            .map(|b| {
                let lo = usize::from(swb[b]);
                let hi = usize::from(swb[b + 1]);
                let db = (lo..hi)
                    .map(|k| ath_db((k as f64 + 0.5) * hz_per_line))
                    .fold(f64::INFINITY, f64::min);
                (full_scale * 10f64.powf((db - 96.0) / 10.0)) as f32
            })
            .collect();
        Self {
            spread,
            bark: barks.iter().map(|&b| b as f32).collect(),
            ath,
        }
    }

    /// The threshold in quiet per band.
    pub fn ath(&self) -> &[f32] {
        &self.ath
    }

    pub fn num_bands(&self) -> usize {
        self.ath.len()
    }

    /// Masking thresholds for one window's spectrum. `prev_nb` is the
    /// previous long block's raw threshold when there is one: the threshold
    /// may rise to at most twice it (pre-echo control, Annex C.1.4 step 11).
    pub fn analyse(&self, swb: &[u16], spec: &[f32], prev_nb: Option<&[f32]>) -> Masking {
        let n = self.num_bands();
        let mut energy = vec![0.0f32; n];
        let mut nb = vec![0.0f32; n];
        let mut thr = vec![0.0f32; n];
        let mut snr = [0.0f32; 64];
        for b in 0..n {
            let band = &spec[usize::from(swb[b])..usize::from(swb[b + 1])];
            let e: f64 = band.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
            energy[b] = e as f32;
            // Spectral flatness in dB: 10*log10(geometric mean / arithmetic mean)
            // of the line energies. Noise sits near -5 dB, a tone far below.
            let w = band.len() as f64;
            let alpha = if e > 1e-3 {
                let floor = e / w * 1e-6;
                let geo = band
                    .iter()
                    .map(|&x| (f64::from(x) * f64::from(x) + floor).ln())
                    .sum::<f64>()
                    / w;
                let sfm_db = 10.0 / LN_10 * (geo - (e / w + floor).ln());
                ((-sfm_db - 6.0) / 24.0).clamp(0.0, 1.0)
            } else {
                0.0
            };
            snr[b] = alpha as f32 * (14.5 + self.bark[b]) + (1.0 - alpha as f32) * 5.5;
        }
        for dst in 0..n {
            let en: f32 = energy
                .iter()
                .zip(self.spread[dst..].iter().step_by(n))
                .map(|(e, s)| e * s)
                .sum();
            let mut raw = en * 10f32.powf(-snr[dst] / 10.0);
            if let Some(prev) = prev_nb {
                raw = raw.min(2.0 * prev[dst]);
            }
            nb[dst] = raw;
            thr[dst] = raw.max(self.ath[dst]);
        }
        Masking { energy, nb, thr }
    }
}

/// One window's band energies and thresholds.
pub(super) struct Masking {
    pub energy: Vec<f32>,
    /// The raw threshold, before the threshold in quiet: what the next
    /// block's pre-echo control compares against.
    pub nb: Vec<f32>,
    /// The masking threshold, floored by the threshold in quiet.
    pub thr: Vec<f32>,
}

/// What the transient detector saw in one 1024-sample zone: the high-passed
/// energy of each 128-sample sub-block, and the first sub-block that rose
/// sharply above the recent level.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Zone {
    pub energy: [f32; 8],
    pub attack: Option<usize>,
}

/// Energy-ratio transient detector.
#[derive(Default)]
pub(super) struct AttackDetector {
    prev_sample: f32,
    /// Decaying peak of recent sub-block energies.
    peak: f32,
}

/// A sub-block whose energy exceeds the decayed recent peak by this factor
/// (10 dB) is an attack.
const ATTACK_RATIO: f32 = 10.0;
/// Per-sub-block decay of the reference peak (3 dB per 128 samples).
const PEAK_DECAY: f32 = 0.5;
/// Attacks quieter than this (high-passed energy per 128 samples, 16-bit
/// scale; a difference signal of about 30 LSB RMS) are not worth short
/// blocks.
const ATTACK_FLOOR: f32 = 128.0 * 30.0 * 30.0;

impl AttackDetector {
    pub fn analyse(&mut self, zone: &[f32]) -> Zone {
        debug_assert_eq!(zone.len(), 1024);
        let mut out = Zone::default();
        for (j, block) in zone.as_chunks::<128>().0.iter().enumerate() {
            let mut e = 0.0f32;
            for &x in block {
                let d = x - self.prev_sample;
                self.prev_sample = x;
                e += d * d;
            }
            out.energy[j] = e;
            if out.attack.is_none() && e > ATTACK_FLOOR && e > ATTACK_RATIO * self.peak {
                out.attack = Some(j);
            }
            self.peak = (self.peak * PEAK_DECAY).max(e);
        }
        out
    }
}

/// Group the eight short windows of an EIGHT_SHORT_SEQUENCE: windows share
/// scalefactors while their energies stay within a factor of eight of the
/// group's first window, and the attack window always opens a group of its
/// own. Returns the window_group_length list.
pub(super) fn group_short_windows(energy: &[f32; 8], attack: Option<usize>) -> Vec<usize> {
    let mut groups = vec![1usize];
    let mut first = energy[0];
    for (w, &e) in energy.iter().enumerate().skip(1) {
        let lo = first.min(e).max(1.0);
        let hi = first.max(e).max(1.0);
        let split = Some(w) == attack || Some(w - 1) == attack || hi > 8.0 * lo;
        if split {
            groups.push(1);
            first = e;
        } else {
            *groups.last_mut().unwrap() += 1;
        }
    }
    groups
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spreading_peaks_at_zero_distance_and_is_narrow() {
        let at0 = spreading(10.0, 10.0);
        assert!((at0 - 1.0).abs() < 0.01, "{at0}");
        // The Annex's function is steep both ways: at least 15 dB down one
        // Bark away, which keeps the model on the conservative side.
        assert!(spreading(10.0, 11.0) < 10f64.powf(-1.5));
        assert!(spreading(10.0, 9.0) < 10f64.powf(-1.5));
    }

    #[test]
    fn detector_flags_a_click_after_silence_but_not_a_steady_tone() {
        let mut d = AttackDetector::default();
        let tone: Vec<f32> = (0..4096)
            .map(|i| 8000.0 * (i as f32 * 0.07).sin())
            .collect();
        let mut hits = 0;
        for z in tone.as_chunks::<1024>().0 {
            hits += d.analyse(z).attack.is_some() as usize;
        }
        // The very first zone rises out of nothing; nothing after it may.
        assert!(hits <= 1, "{hits}");

        let mut d = AttackDetector::default();
        let mut zone = vec![0.0f32; 1024];
        for (i, s) in zone[600..700].iter_mut().enumerate() {
            *s = 20000.0 * (-(i as f32) / 20.0).exp() * if i % 2 == 0 { 1.0 } else { -1.0 };
        }
        assert_eq!(d.analyse(&zone).attack, Some(4));
    }

    #[test]
    fn grouping_isolates_the_attack_window() {
        let e = [1.0, 1.0, 1.0, 1e6, 5e5, 4e5, 3e5, 2e5];
        assert_eq!(group_short_windows(&e, Some(3)), vec![3, 1, 4]);
        assert_eq!(group_short_windows(&[5.0; 8], None), vec![8]);
    }
}
