//! ALAC (Apple Lossless) encoder, clean-room from the published format
//! description and the literature on linear prediction.
//!
//! 4096-sample frames of SCE / CPE / LFE elements in ALAC's channel order
//! for the count (see [`element_layout`]). Each element is coded two ways
//! and the smaller kept:
//! - **predicted**: the adaptive FIR predictor seeded, per channel and
//!   frame, with LPC coefficients from Levinson-Durbin (orders 4 and 8 tried,
//!   in units of 2^-9), its residual in the adaptive Rice code; a channel
//!   pair is first mixed to a weighted mid and a side, with the weight
//!   (none, ¼ … 1) that leaves the smallest first differences;
//! - **escaped**: the samples raw at the stream's bit depth.
//!
//! 20- and 24-bit elements are also tried with their low byte split off into
//! the element's raw shift buffer (32-bit audio always splits off two), and
//! the smaller kept: the Rice parameter is capped at 14 bits, so a noise
//! floor well above that codes better with the bottom byte sent raw.
//!
//! The predictor's arithmetic is checked against the 32-bit range a
//! reference decoder computes in; a coding that would leave it is not an
//! option, and an element with no other goes out escaped, so every frame
//! decodes the same everywhere.

use crate::audio::lossless::alac::{
    AlacConfig, ID_END, RiceParams, element_layout, encode_residuals, native_from_alac, predict, residual_bits,
};
use crate::audio::lossless::bits::BitWriter;
use crate::audio::lossless::{f32_to_int, lpc};
use crate::audio::{AudioEncoder, AudioEncoderConfig, AudioError, AudioFrame, EncodedAudioPacket};

#[cfg(test)]
mod tests;

/// Coefficient scale: units of 2^-DEN_SHIFT.
const DEN_SHIFT: u32 = 9;
/// Rice history multiplier factor, in quarters of the cookie's `pb`.
const PB_FACTOR: u32 = 4;
/// Predictor orders tried per channel.
const ORDERS: [usize; 2] = [4, 8];
/// Pair mixing: the weight is `mix_res / 2^MIX_BITS`.
const MIX_BITS: u32 = 2;

pub struct AlacEncoder {
    config: AlacConfig,
    pending: Vec<i32>,
    samples: u64,
    bytes: u64,
}

impl AlacEncoder {
    pub fn new(sample_rate: u32, channels: u8, bit_depth: u8) -> Result<Self, AudioError> {
        if !(1..=8).contains(&channels) {
            return Err(AudioError::Unsupported(format!("alac: {channels} channels (1–8)")));
        }
        if !matches!(bit_depth, 16 | 20 | 24 | 32) {
            return Err(AudioError::Unsupported(format!("alac: {bit_depth}-bit samples (16, 20, 24 or 32)")));
        }
        if sample_rate == 0 {
            return Err(AudioError::Unsupported("alac: sample rate 0".into()));
        }
        Ok(Self { config: AlacConfig::new(sample_rate, channels, bit_depth), pending: Vec::new(), samples: 0, bytes: 0 })
    }

    /// The magic cookie, with the largest frame and the average bit rate
    /// filled in from what has been encoded.
    pub fn cookie(&self) -> AlacConfig {
        let mut c = self.config.clone();
        if self.samples > 0 {
            c.avg_bit_rate =
                (self.bytes as f64 * 8.0 * f64::from(c.sample_rate) / self.samples as f64).round() as u32;
        }
        c
    }

    /// Encode interleaved integer samples in the pipeline's channel order;
    /// returns the frames completed, each with its sample count.
    pub fn encode_int(&mut self, samples: &[i32]) -> Vec<(Vec<u8>, u32)> {
        self.pending.extend_from_slice(samples);
        let ch = usize::from(self.config.num_channels);
        let len = self.config.frame_length as usize;
        let whole = self.pending.len() / (len * ch);
        let mut out = Vec::with_capacity(whole);
        for i in 0..whole {
            let block = self.pending[i * len * ch..(i + 1) * len * ch].to_vec();
            out.push((self.encode_frame(&block), len as u32));
        }
        self.pending.drain(..whole * len * ch);
        out
    }

    /// Encode what is left as a final, shorter frame.
    pub fn finish(&mut self) -> Vec<(Vec<u8>, u32)> {
        if self.pending.is_empty() {
            return Vec::new();
        }
        let block = std::mem::take(&mut self.pending);
        let n = block.len() / usize::from(self.config.num_channels);
        vec![(self.encode_frame(&block), n as u32)]
    }

    fn encode_frame(&mut self, interleaved: &[i32]) -> Vec<u8> {
        let ch = usize::from(self.config.num_channels);
        let n = interleaved.len() / ch;
        // ALAC channel `a` is the native slot that maps to it.
        let native = native_from_alac(self.config.num_channels);
        let mut alac: Vec<Vec<i64>> = vec![Vec::new(); ch];
        for (slot, &a) in native.iter().enumerate() {
            alac[a] = interleaved.iter().skip(slot).step_by(ch).map(|&s| i64::from(s)).collect();
        }
        let mut bw = BitWriter::with_capacity(interleaved.len() * 3);
        let mut next = 0usize;
        let mut instances = [0u32; 8];
        for &(tag, count) in element_layout(self.config.num_channels) {
            let instance = instances[tag as usize];
            instances[tag as usize] += 1;
            self.write_element(&mut bw, tag, instance, &alac[next..next + count], n);
            next += count;
        }
        bw.write(u64::from(ID_END), 3);
        let frame = bw.into_bytes();
        self.samples += n as u64;
        self.bytes += frame.len() as u64;
        self.config.max_frame_bytes = self.config.max_frame_bytes.max(frame.len() as u32);
        frame
    }

    fn write_element(&self, bw: &mut BitWriter, tag: u32, instance: u32, chans: &[Vec<i64>], n: usize) {
        let depth = u32::from(self.config.bit_depth);
        let partial = n != self.config.frame_length as usize;
        let header_bits = 3 + 4 + 12 + 1 + 2 + 1 + if partial { 32 } else { 0 };
        let escaped_bits = header_bits + n * chans.len() * depth as usize;
        // 32-bit audio always splits off two bytes. At 20 and 24 bits both
        // are tried: the Rice parameter tops out at `kb` (14), so a residual
        // much above 14 bits codes better with its low byte sent raw.
        let shifts: &[u32] = match depth {
            32 => &[2],
            24 | 20 => &[0, 1],
            _ => &[0],
        };
        let compressed = shifts.iter().filter_map(|&s| self.plan_compressed(chans, s)).min_by_key(|p| p.bits);
        let write_header = |bw: &mut BitWriter, shift: u32, escape: bool| {
            bw.write(u64::from(tag), 3);
            bw.write(u64::from(instance), 4);
            bw.write(0, 12);
            bw.write_bit(partial);
            bw.write(u64::from(shift), 2);
            bw.write_bit(escape);
            if partial {
                bw.write(n as u64, 32);
            }
        };
        match compressed {
            Some(plan) if header_bits + plan.bits < escaped_bits => {
                write_header(bw, plan.shift_bytes, false);
                self.write_compressed(bw, &plan);
            }
            _ => {
                write_header(bw, 0, true);
                for i in 0..n {
                    for c in chans {
                        bw.write_signed(c[i], depth);
                    }
                }
            }
        }
    }

    /// The cheapest predicted coding of one element with `shift_bytes` low
    /// bytes split off; `None` when the predictor's arithmetic would leave
    /// the 32-bit range.
    fn plan_compressed(&self, chans: &[Vec<i64>], shift_bytes: u32) -> Option<ElementPlan> {
        let depth = u32::from(self.config.bit_depth);
        let shift = shift_bytes * 8;
        let nch = chans.len();
        let chan_bits = depth - shift + (nch as u32 - 1);
        let high: Vec<Vec<i64>> = chans.iter().map(|c| c.iter().map(|&s| s >> shift).collect()).collect();
        let low: Vec<Vec<i64>> =
            chans.iter().map(|c| c.iter().map(|&s| s & ((1i64 << shift) - 1)).collect()).collect();
        let (mix_res, mixed) = if nch == 2 { best_mix(&high[0], &high[1]) } else { (0, high) };
        let params = RiceParams::new(&self.config, PB_FACTOR);
        let mut channels = Vec::with_capacity(nch);
        let mut bits = 8 + 8 + nch * shift as usize * chans[0].len();
        for x in &mixed {
            let mut best: Option<ChannelPlan> = None;
            for coefs in seed_coefficients(x) {
                let Some(residual) = predict(x, &coefs, DEN_SHIFT, chan_bits) else { continue };
                let b = 4 + 4 + 3 + 5 + 16 * coefs.len() + residual_bits(&params, &residual, chan_bits);
                if best.as_ref().is_none_or(|p| b < p.bits) {
                    best = Some(ChannelPlan { coefs, residual, bits: b });
                }
            }
            let best = best?;
            bits += best.bits;
            channels.push(best);
        }
        Some(ElementPlan { shift_bytes, chan_bits, mix_res, low, channels, bits })
    }

    fn write_compressed(&self, bw: &mut BitWriter, plan: &ElementPlan) {
        let params = RiceParams::new(&self.config, PB_FACTOR);
        bw.write(u64::from(MIX_BITS), 8);
        bw.write_signed(i64::from(plan.mix_res), 8);
        for c in &plan.channels {
            bw.write(0, 4); // mode: the plain predictor
            bw.write(u64::from(DEN_SHIFT), 4);
            bw.write(u64::from(PB_FACTOR), 3);
            bw.write(c.coefs.len() as u64, 5);
            for &k in &c.coefs {
                bw.write_signed(i64::from(k), 16);
            }
        }
        let shift = plan.shift_bytes * 8;
        if shift > 0 {
            for i in 0..plan.low[0].len() {
                for ch in &plan.low {
                    bw.write(ch[i] as u64, shift);
                }
            }
        }
        for c in &plan.channels {
            encode_residuals(bw, &params, &c.residual, plan.chan_bits);
        }
    }
}

struct ChannelPlan {
    coefs: Vec<i16>,
    residual: Vec<i32>,
    bits: usize,
}

struct ElementPlan {
    shift_bytes: u32,
    chan_bits: u32,
    mix_res: i32,
    /// The split-off low bits, per channel.
    low: Vec<Vec<i64>>,
    channels: Vec<ChannelPlan>,
    /// Bits after the element header.
    bits: usize,
}

/// The pair mixing with the smallest first differences: `mix_res` 0 keeps
/// the channels as they are; otherwise the pair becomes
/// `u = (w·L + (1-w)·R)`, `v = L - R` with `w = mix_res / 4`.
fn best_mix(l: &[i64], r: &[i64]) -> (i32, Vec<Vec<i64>>) {
    let roughness = |x: &[i64]| x.windows(2).map(|w| (w[1] - w[0]).unsigned_abs()).sum::<u64>();
    let mut best = (0i32, u64::MAX, Vec::new());
    for mix_res in 0..=(1i32 << MIX_BITS) {
        let (u, v): (Vec<i64>, Vec<i64>) = if mix_res == 0 {
            (l.to_vec(), r.to_vec())
        } else {
            let m = 1i64 << MIX_BITS;
            let w = i64::from(mix_res);
            l.iter().zip(r).map(|(&a, &b)| ((w * a + (m - w) * b) >> MIX_BITS, a - b)).unzip()
        };
        let cost = roughness(&u) + roughness(&v);
        if cost < best.1 {
            best = (mix_res, cost, vec![u, v]);
        }
    }
    (best.0, best.2)
}

/// Starting coefficients for the adaptive predictor: the LPC solutions of
/// each tried order, in units of 2^-DEN_SHIFT.
fn seed_coefficients(x: &[i64]) -> Vec<Vec<i16>> {
    let max = *ORDERS.iter().max().expect("orders");
    if x.len() <= max + 1 {
        return vec![Vec::new()];
    }
    let window = lpc::tukey(x.len(), 0.5);
    let r = lpc::autocorrelation(x, &window, max);
    let (coefs, _) = lpc::levinson(&r, max);
    let mut out: Vec<Vec<i16>> = ORDERS
        .iter()
        .filter_map(|&o| coefs.get(o - 1))
        .map(|c| {
            c.iter()
                .map(|&v| (v * f64::from(1u32 << DEN_SHIFT)).round().clamp(-32_768.0, 32_767.0) as i16)
                .collect()
        })
        .collect();
    if out.is_empty() {
        // Silence or a signal the recursion cannot model: a short, neutral
        // predictor, which the adaptation takes from there.
        out.push(vec![0; ORDERS[0]]);
    }
    out
}

/// ALAC through the [`AudioEncoder`] surface: pipeline f32 samples are
/// taken at `bits_per_sample`.
pub struct AlacAudioEncoder {
    inner: AlacEncoder,
    samples_out: u64,
}

impl AlacAudioEncoder {
    pub fn new(config: &AudioEncoderConfig, bits_per_sample: u8) -> Result<Self, AudioError> {
        Ok(Self { inner: AlacEncoder::new(config.sample_rate, config.channels, bits_per_sample)?, samples_out: 0 })
    }

    fn packets(&mut self, frames: Vec<(Vec<u8>, u32)>) -> Vec<EncodedAudioPacket> {
        let rate = i64::from(self.inner.config.sample_rate);
        frames
            .into_iter()
            .map(|(data, n)| {
                let pts = self.samples_out as i64 * 1_000_000 / rate;
                self.samples_out += u64::from(n);
                EncodedAudioPacket { data, pts, duration: i64::from(n) }
            })
            .collect()
    }
}

impl AudioEncoder for AlacAudioEncoder {
    fn encode(&mut self, frame: &AudioFrame) -> Result<Vec<EncodedAudioPacket>, AudioError> {
        let cfg = &self.inner.config;
        if frame.channels != cfg.num_channels || frame.sample_rate != cfg.sample_rate {
            return Err(AudioError::Encode(format!(
                "alac: a {}-channel {} Hz frame into a {}-channel {} Hz encoder",
                frame.channels, frame.sample_rate, cfg.num_channels, cfg.sample_rate
            )));
        }
        let bits = u32::from(cfg.bit_depth);
        let ints: Vec<i32> = frame.samples.iter().map(|&x| f32_to_int(x, bits)).collect();
        let frames = self.inner.encode_int(&ints);
        Ok(self.packets(frames))
    }

    fn flush(&mut self) -> Result<Vec<EncodedAudioPacket>, AudioError> {
        let frames = self.inner.finish();
        Ok(self.packets(frames))
    }

    fn pre_skip(&self) -> u16 {
        0
    }

    /// The 24-byte magic cookie, with its frame-size and bit-rate fields
    /// final once flushed.
    fn extra_data(&self) -> Vec<u8> {
        self.inner.cookie().to_bytes().to_vec()
    }

    fn sample_rate(&self) -> u32 {
        self.inner.config.sample_rate
    }
}
