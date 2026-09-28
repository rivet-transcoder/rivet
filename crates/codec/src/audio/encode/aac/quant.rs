//! Quantization and scalefactor selection for one channel.
//!
//! Every scalefactor band gets an allowed noise energy — the masking
//! threshold scaled by the frame's common offset (see `AacEncoder::search`),
//! but never below the threshold in quiet, which is absolute — and its
//! scalefactor is the coarsest one whose actual quantization error stays
//! within it. The starting point is the high-resolution approximation of the
//! error of the x^(3/4) companded quantizer (the step in the linear domain is
//! d/dy of y^(4/3), so the error energy of a line is about
//! (1/12)(4/3)^2 |x|^(1/2) 2^(3s/8) for scalefactor offset s); the measured
//! error then corrects it by a step or two. Bands that may be dropped
//! entirely (energy below the allowed noise) are zeroed.
//!
//! The quantizer itself is Annex C.7.4's: x_quant = int(|x|^(3/4) *
//! 2^(-3/16 * (sf - 100)) + 0.4054).

use std::sync::OnceLock;

use super::huffman::{self, MAX_QUANT, NUM_CODEBOOKS, Section};
use super::mdct::WindowSequence;

const SF_OFFSET: i32 = 100;
const MAGIC_NUMBER: f32 = 0.4054;
/// Scalefactor differences must stay within the codebook's -60..=60.
const MAX_SF_DELTA: i32 = 60;

fn pow43() -> &'static [f32] {
    static TABLE: OnceLock<Vec<f32>> = OnceLock::new();
    TABLE.get_or_init(|| {
        (0..=MAX_QUANT as usize)
            .map(|q| (q as f64).powf(4.0 / 3.0) as f32)
            .collect()
    })
}

/// How the 1024 coefficients of one channel's frame are split into windows,
/// groups and scalefactor bands. Coefficients are stored interleaved as the
/// bitstream orders them (subclause 8.3.5): group by group, and within a
/// group scalefactor band by band, each band holding that band of every
/// window of the group. So each (group, band) is one contiguous slice.
#[derive(Clone, Debug)]
pub(super) struct Layout {
    pub seq: WindowSequence,
    /// window_group_length; `[1]` for the long sequences.
    pub group_len: Vec<usize>,
    /// Band offsets of one window.
    pub swb: &'static [u16],
}

impl Layout {
    pub fn short(&self) -> bool {
        self.seq.is_short()
    }

    pub fn num_swb(&self) -> usize {
        self.swb.len() - 1
    }

    pub fn num_groups(&self) -> usize {
        self.group_len.len()
    }

    /// Coefficient range of band `sfb` of group `g` in the interleaved order.
    pub fn band(&self, g: usize, sfb: usize) -> std::ops::Range<usize> {
        let window_len = usize::from(*self.swb.last().unwrap());
        let base: usize = self.group_len[..g].iter().sum::<usize>() * window_len;
        let len = self.group_len[g];
        base + usize::from(self.swb[sfb]) * len..base + usize::from(self.swb[sfb + 1]) * len
    }

    /// The 7-bit scale_factor_grouping field: bit `7 - w` set when window
    /// `w` continues the previous window's group (subclause 8.3.4).
    pub fn grouping_bits(&self) -> u32 {
        let mut bits = 0u32;
        let mut w = 0;
        for &len in &self.group_len {
            for i in 0..len {
                if w > 0 && i > 0 {
                    bits |= 1 << (7 - w);
                }
                w += 1;
            }
        }
        bits
    }
}

/// One channel's analysed frame, ready for quantization.
pub(super) struct ChannelFrame {
    pub layout: Layout,
    /// Interleaved MDCT coefficients (after M/S, when used).
    pub coef: Vec<f32>,
    /// Per band, `g * num_swb + sfb`.
    pub energy: Vec<f32>,
    pub thr: Vec<f32>,
    /// Per band: the threshold in quiet, a floor under the allowed noise
    /// however many bits there are.
    pub floor: Vec<f32>,
    /// Bands at or above this index (within each group) are never coded:
    /// the bandwidth limit, or the LFE's twelve lines.
    pub band_limit: usize,
    /// Perceptual entropy estimate, for the bit reservoir.
    pub pe: f32,
    abs: Vec<f32>,
    abs34: Vec<f32>,
    /// Per band: sum of sqrt|x| and the largest |x|.
    sum_sqrt: Vec<f32>,
    max_abs: Vec<f32>,
}

impl ChannelFrame {
    pub fn new(
        layout: Layout,
        coef: Vec<f32>,
        energy: Vec<f32>,
        thr: Vec<f32>,
        floor: Vec<f32>,
        band_limit: usize,
    ) -> Self {
        let n = layout.num_groups() * layout.num_swb();
        let mut cf = Self {
            layout,
            coef,
            energy,
            thr,
            floor,
            band_limit,
            pe: 0.0,
            abs: Vec::new(),
            abs34: Vec::new(),
            sum_sqrt: vec![0.0; n],
            max_abs: vec![0.0; n],
        };
        cf.refresh();
        cf
    }

    /// Recompute the derived per-coefficient and per-band values after the
    /// coefficients changed (M/S).
    pub fn refresh(&mut self) {
        self.abs = self.coef.iter().map(|x| x.abs()).collect();
        self.abs34 = self
            .abs
            .iter()
            .map(|&a| a.sqrt() * a.sqrt().sqrt())
            .collect();
        let nswb = self.layout.num_swb();
        let mut pe = 0.0f32;
        for g in 0..self.layout.num_groups() {
            for sfb in 0..nswb {
                let i = g * nswb + sfb;
                let r = self.layout.band(g, sfb);
                let band = &self.abs[r.clone()];
                self.sum_sqrt[i] = band.iter().map(|a| a.sqrt()).sum();
                self.max_abs[i] = band.iter().fold(0.0, |m, &a| m.max(a));
                self.energy[i] = band.iter().map(|a| a * a).sum();
                if sfb < self.band_limit && self.energy[i] > self.thr[i] {
                    pe += 0.5 * r.len() as f32 * (self.energy[i] / self.thr[i]).log2();
                }
            }
        }
        self.pe = pe;
    }
}

/// A channel's quantized frame and everything needed to write its
/// individual_channel_stream.
#[derive(Clone, Default)]
pub(super) struct Quantized {
    /// Signed quantized values, interleaved like `ChannelFrame::coef`.
    pub q: Vec<i32>,
    /// Scalefactor per band (meaningful where the band is coded).
    pub sf: Vec<i32>,
    /// Whether the band has a non-zero value.
    pub active: Vec<bool>,
    /// 1 + the highest band (in any group) with a non-zero value.
    pub max_sfb: usize,
    pub global_gain: i32,
    /// Per group, set by [`Quantized::finish`].
    pub sections: Vec<Vec<Section>>,
    /// Bits of the ICS body: global_gain, section data, scalefactors, the
    /// three tool flags and the spectral data (ics_info excluded).
    pub body_bits: usize,
}

fn quantize_band(abs34: &[f32], sf: i32, out: &mut [i32]) -> i32 {
    let gain = 2f32.powf(-0.1875 * (sf - SF_OFFSET) as f32);
    let mut max = 0;
    for (o, &a) in out.iter_mut().zip(abs34) {
        let q = (a * gain + MAGIC_NUMBER) as i32;
        *o = q;
        max = max.max(q);
    }
    max
}

fn band_noise(abs: &[f32], q: &[i32], sf: i32) -> f32 {
    let step = 2f32.powf(0.25 * (sf - SF_OFFSET) as f32);
    let p = pow43();
    abs.iter()
        .zip(q)
        .map(|(&a, &q)| {
            let d = a - p[q as usize] * step;
            d * d
        })
        .sum()
}

/// Smallest scalefactor at which the band's largest magnitude still
/// quantizes to at most 8191.
fn min_sf_for(max_abs: f32) -> i32 {
    if max_abs <= 0.0 {
        return 0;
    }
    // (max^(3/4) * 2^(-3s/16)) + 0.4054 <= 8191  <=>  s >= 16/3 * log2(max^(3/4) / 8190.5946)
    let s = 16.0 / 3.0 * (max_abs.powf(0.75) / (MAX_QUANT as f32 - MAGIC_NUMBER)).log2();
    let mut sf = (s.ceil() as i32 + SF_OFFSET).max(0);
    // Guard the float edge: step up until the quantizer really fits.
    let a34 = max_abs.powf(0.75);
    while (a34 * 2f32.powf(-0.1875 * (sf - SF_OFFSET) as f32) + MAGIC_NUMBER) as i32 > MAX_QUANT {
        sf += 1;
    }
    sf
}

impl Quantized {
    /// Quantize every band of `cf` against `allowed[band]` noise energy.
    /// Chooses scalefactors and zeroes bands, but not the sections: those
    /// depend on max_sfb, which a channel pair shares ([`Quantized::finish`]).
    pub fn quantize(cf: &ChannelFrame, noise_scale: f32) -> Self {
        let layout = &cf.layout;
        let nswb = layout.num_swb();
        let nbands = layout.num_groups() * nswb;
        let mut out = Quantized {
            q: vec![0; cf.coef.len()],
            sf: vec![0; nbands],
            active: vec![false; nbands],
            ..Default::default()
        };
        let mut min_sf = vec![0i32; nbands];
        for g in 0..layout.num_groups() {
            for sfb in 0..nswb.min(cf.band_limit) {
                let i = g * nswb + sfb;
                let allowed = (cf.thr[i] * noise_scale).max(cf.floor[i]);
                if cf.energy[i] <= allowed || cf.max_abs[i] <= 0.0 {
                    continue;
                }
                let r = layout.band(g, sfb);
                min_sf[i] = min_sf_for(cf.max_abs[i]);
                let sf = choose_sf(
                    &cf.abs[r.clone()],
                    &cf.abs34[r.clone()],
                    cf.sum_sqrt[i],
                    allowed,
                    min_sf[i],
                    &mut out.q[r],
                );
                out.sf[i] = sf;
                out.active[i] = true;
            }
        }
        out.enforce_sf_deltas(cf);
        for (q, &c) in out.q.iter_mut().zip(&cf.coef) {
            if c < 0.0 {
                *q = -*q;
            }
        }
        out.max_sfb = (0..nbands)
            .filter(|&i| out.active[i])
            .map(|i| i % nswb + 1)
            .max()
            .unwrap_or(0);
        out.global_gain = (0..nbands)
            .find(|&i| out.active[i])
            .map(|i| out.sf[i])
            .unwrap_or(SF_OFFSET);
        out
    }

    /// Scalefactors of consecutive coded bands may differ by at most 60.
    /// Only ever *raise* scalefactors (coarser, so the 8191 limit keeps
    /// holding); a band that quantizes to nothing after the raise drops out
    /// of the chain, which can bring two others next to each other, so
    /// repeat until stable.
    fn enforce_sf_deltas(&mut self, cf: &ChannelFrame) {
        let layout = &cf.layout;
        let nswb = layout.num_swb();
        loop {
            let chain: Vec<usize> = (0..self.active.len()).filter(|&i| self.active[i]).collect();
            let mut target: Vec<i32> = chain.iter().map(|&i| self.sf[i]).collect();
            for k in (1..target.len()).rev() {
                target[k - 1] = target[k - 1].max(target[k] - MAX_SF_DELTA);
            }
            for k in 1..target.len() {
                target[k] = target[k].max(target[k - 1] - MAX_SF_DELTA);
            }
            let mut dropped = false;
            for (k, &i) in chain.iter().enumerate() {
                if target[k] == self.sf[i] {
                    continue;
                }
                let sf = target[k].min(255);
                let r = layout.band(i / nswb, i % nswb);
                let max = quantize_band(&cf.abs34[r.clone()], sf, &mut self.q[r]);
                self.sf[i] = sf;
                if max == 0 {
                    self.active[i] = false;
                    dropped = true;
                }
            }
            if !dropped {
                break;
            }
        }
    }

    /// Choose sections for `max_sfb` bands and total up the ICS body bits.
    pub fn finish(&mut self, cf: &ChannelFrame, max_sfb: usize) {
        let layout = &cf.layout;
        let nswb = layout.num_swb();
        let short = layout.short();
        let mut bits = 8 + 3; // global_gain; pulse, tns and gain-control flags
        self.sections.clear();
        let mut cost = vec![[u32::MAX; NUM_CODEBOOKS]; max_sfb];
        for g in 0..layout.num_groups() {
            for (sfb, row) in cost.iter_mut().enumerate() {
                let i = g * nswb + sfb;
                *row = [u32::MAX; NUM_CODEBOOKS];
                let q = &self.q[layout.band(g, sfb)];
                let max_abs = if self.active[i] {
                    q.iter().map(|v| v.abs()).max().unwrap_or(0)
                } else {
                    0
                };
                for cb in 0..NUM_CODEBOOKS as u8 {
                    // Every codebook covers an empty band.
                    if !huffman::codebook_covers(cb, max_abs) {
                        continue;
                    }
                    let mut c = huffman::band_bits(cb, q);
                    if cb > 0 && max_abs == 0 {
                        // An empty band inside a coded section still sends
                        // a scalefactor: the repeat of the previous one, the
                        // one-bit difference 0.
                        c += 1;
                    }
                    row[usize::from(cb)] = c;
                }
            }
            let (sections, c) = huffman::choose_sections(&cost, short);
            bits += c as usize;
            self.sections.push(sections);
        }
        // Scalefactors of the non-empty bands, as differences along the chain.
        let mut last = self.global_gain;
        for g in 0..layout.num_groups() {
            for sfb in 0..max_sfb {
                let i = g * nswb + sfb;
                if self.active[i] {
                    bits += huffman::sf_bits(self.sf[i] - last) as usize;
                    last = self.sf[i];
                }
            }
        }
        self.max_sfb = max_sfb;
        self.body_bits = bits;
    }

    /// Codebook of band `sfb` of group `g` after sectioning.
    pub fn band_codebook(&self, g: usize, sfb: usize) -> u8 {
        self.sections[g]
            .iter()
            .find(|s| s.start <= sfb && sfb < s.end)
            .map(|s| s.cb)
            .unwrap_or(0)
    }
}

/// Coarsest scalefactor whose measured noise stays within `allowed`.
fn choose_sf(
    abs: &[f32],
    abs34: &[f32],
    sum_sqrt: f32,
    allowed: f32,
    min_sf: i32,
    q: &mut [i32],
) -> i32 {
    // Error energy ~ (4/27) * 2^(3s/8) * sum sqrt|x|  (s = sf - 100).
    let est = 8.0 / 3.0 * (27.0 * allowed / (4.0 * sum_sqrt.max(1e-9))).log2();
    let mut sf = (est.floor() as i32 + SF_OFFSET).clamp(min_sf, 255);
    quantize_band(abs34, sf, q);
    let mut noise = band_noise(abs, q, sf);
    if noise > allowed {
        for _ in 0..8 {
            if sf <= min_sf {
                break;
            }
            sf -= 1;
            quantize_band(abs34, sf, q);
            noise = band_noise(abs, q, sf);
            if noise <= allowed {
                break;
            }
        }
    } else {
        let mut trial = vec![0i32; q.len()];
        for _ in 0..4 {
            if sf >= 255 {
                break;
            }
            quantize_band(abs34, sf + 1, &mut trial);
            if band_noise(abs, &trial, sf + 1) > allowed || trial.iter().all(|&v| v == 0) {
                break;
            }
            sf += 1;
            q.copy_from_slice(&trial);
        }
    }
    sf
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn min_sf_keeps_the_largest_value_codable() {
        for max in [1.0f32, 1234.5, 3.2e7, 1e9] {
            let sf = min_sf_for(max);
            let mut q = [0];
            assert!(
                quantize_band(&[max.powf(0.75)], sf, &mut q) <= MAX_QUANT,
                "{max}"
            );
            if sf > 0 {
                quantize_band(&[max.powf(0.75)], sf - 1, &mut q);
                assert!(
                    q[0] > MAX_QUANT || sf - 1 < 0,
                    "{max}: sf {sf} is not the minimum"
                );
            }
        }
    }

    #[test]
    fn grouping_bits_mark_continuations() {
        let l = Layout {
            seq: WindowSequence::EightShort,
            group_len: vec![3, 1, 4],
            swb: &[0, 4, 128],
        };
        // Windows 1, 2 continue group 0; window 3 starts group 1; 4 starts
        // group 2 and 5, 6, 7 continue it.
        assert_eq!(l.grouping_bits(), 0b110_0111);
        assert_eq!(l.band(1, 1), 3 * 128 + 4..3 * 128 + 128);
        assert_eq!(l.band(2, 0), 4 * 128..4 * 128 + 16);
    }
}
