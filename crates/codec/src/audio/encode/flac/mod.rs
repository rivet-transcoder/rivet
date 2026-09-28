//! FLAC encoder, clean-room from RFC 9639 and the published literature on
//! linear prediction and Rice coding.
//!
//! A fixed-block-size stream of 4096-sample frames. For every subframe the
//! encoder weighs the forms the format has — constant, verbatim, the fixed
//! polynomial predictors of orders 0–4 and LPC — and keeps the cheapest in
//! exact bits; LPC coefficients come from the autocorrelation of a
//! Tukey-windowed block through Levinson-Durbin, quantised with error
//! feedback. Residuals are Rice coded in partitions whose order and
//! parameters are searched per subframe, with the escape to raw bits where
//! that is smaller. Stereo frames try all four channel assignments
//! (independent, left/side, side/right, mid/side). Low bits that are zero
//! throughout a block ("wasted bits") are shifted out.
//!
//! [`FlacLevel`] trades speed for size: `Fast` stops at the fixed
//! predictors, `Default` adds LPC up to order 8 with the order picked from
//! the Levinson error estimate, `Best` tries every LPC order to 12 exactly.
//!
//! The stream's STREAMINFO — frame size bounds, sample count and the MD5 of
//! the audio — is complete once [`FlacEncoder::flush`] has run, which is
//! when a muxer asks for [`AudioEncoder::extra_data`].

use crate::audio::lossless::bits::BitWriter;
use crate::audio::lossless::flac::{BLOCK_STREAMINFO, StreamInfo, block_header, crc8, crc16, md5_bytes};
use crate::audio::lossless::{f32_to_int, lpc};
use crate::audio::{AudioEncoder, AudioEncoderConfig, AudioError, AudioFrame, EncodedAudioPacket};

#[cfg(test)]
mod tests;

/// Samples per channel in every frame but the last.
pub const BLOCK_SIZE: usize = 4096;

/// Compression effort.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum FlacLevel {
    /// Fixed predictors only, Rice partitions to order 3.
    Fast,
    /// LPC to order 8 (order picked by estimate), partitions to order 6.
    #[default]
    Default,
    /// LPC to order 12 (every order tried), partitions to order 8.
    Best,
}

impl FlacLevel {
    fn max_lpc_order(self) -> usize {
        match self {
            Self::Fast => 0,
            Self::Default => 8,
            Self::Best => 12,
        }
    }

    fn max_partition_order(self) -> u32 {
        match self {
            Self::Fast => 3,
            Self::Default => 6,
            Self::Best => 8,
        }
    }
}

/// What a FLAC encoder is built for.
#[derive(Clone, Copy, Debug)]
pub struct FlacEncoderConfig {
    pub sample_rate: u32,
    /// 1–8, in the pipeline's native order (which is FLAC's).
    pub channels: u8,
    /// 4–32.
    pub bits_per_sample: u8,
    pub level: FlacLevel,
}

pub struct FlacEncoder {
    config: FlacEncoderConfig,
    /// Interleaved samples waiting for a whole frame.
    pending: Vec<i32>,
    frames: u64,
    samples: u64,
    min_frame: u32,
    max_frame: u32,
    /// The size of the only block, while there has been one.
    last_block: usize,
    md5: md5::Context,
    md5_scratch: Vec<u8>,
    md5_digest: Option<[u8; 16]>,
}

impl FlacEncoder {
    pub fn new(config: FlacEncoderConfig) -> Result<Self, AudioError> {
        if !(1..=8).contains(&config.channels) {
            return Err(AudioError::Unsupported(format!("flac: {} channels (1–8)", config.channels)));
        }
        if !(4..=32).contains(&config.bits_per_sample) {
            return Err(AudioError::Unsupported(format!("flac: {}-bit samples (4–32)", config.bits_per_sample)));
        }
        if config.sample_rate == 0 || config.sample_rate >= 1 << 20 {
            return Err(AudioError::Unsupported(format!("flac: sample rate {} Hz", config.sample_rate)));
        }
        Ok(Self {
            config,
            pending: Vec::new(),
            frames: 0,
            samples: 0,
            min_frame: u32::MAX,
            max_frame: 0,
            last_block: 0,
            md5: md5::Context::new(),
            md5_scratch: Vec::new(),
            md5_digest: None,
        })
    }

    /// Encode interleaved integer samples; returns the frames completed, each
    /// with its sample count.
    pub fn encode_int(&mut self, samples: &[i32]) -> Vec<(Vec<u8>, u32)> {
        self.pending.extend_from_slice(samples);
        let ch = usize::from(self.config.channels);
        let whole = self.pending.len() / (BLOCK_SIZE * ch);
        let mut out = Vec::with_capacity(whole);
        for i in 0..whole {
            let block = self.pending[i * BLOCK_SIZE * ch..(i + 1) * BLOCK_SIZE * ch].to_vec();
            out.push((self.encode_block(&block), BLOCK_SIZE as u32));
        }
        self.pending.drain(..whole * BLOCK_SIZE * ch);
        out
    }

    /// Encode what is left as a final, shorter frame and seal the MD5.
    pub fn finish(&mut self) -> Vec<(Vec<u8>, u32)> {
        let mut out = Vec::new();
        if !self.pending.is_empty() {
            let block = std::mem::take(&mut self.pending);
            let n = block.len() / usize::from(self.config.channels);
            out.push((self.encode_block(&block), n as u32));
        }
        if self.md5_digest.is_none() {
            self.md5_digest = Some(self.md5.clone().compute().0);
        }
        out
    }

    /// The stream's STREAMINFO as it stands (complete after [`finish`](Self::finish)).
    pub fn stream_info(&self) -> StreamInfo {
        let block = if self.frames <= 1 { self.last_block.max(16) } else { BLOCK_SIZE } as u16;
        StreamInfo {
            min_block_size: block,
            max_block_size: block,
            min_frame_size: if self.max_frame == 0 { 0 } else { self.min_frame },
            max_frame_size: self.max_frame,
            sample_rate: self.config.sample_rate,
            channels: self.config.channels,
            bits_per_sample: self.config.bits_per_sample,
            total_samples: self.samples,
            md5: self.md5_digest.unwrap_or([0; 16]),
        }
    }

    /// The metadata blocks a container carries (`dfLa` body, Matroska
    /// CodecPrivate after `fLaC`): STREAMINFO alone, flagged last.
    pub fn metadata_blocks(&self) -> Vec<u8> {
        let mut b = block_header(true, BLOCK_STREAMINFO, StreamInfo::LEN).to_vec();
        b.extend_from_slice(&self.stream_info().to_bytes());
        b
    }

    fn encode_block(&mut self, interleaved: &[i32]) -> Vec<u8> {
        let ch = usize::from(self.config.channels);
        let n = interleaved.len() / ch;
        let bps = u32::from(self.config.bits_per_sample);
        self.md5_scratch.clear();
        md5_bytes(interleaved, bps, &mut self.md5_scratch);
        self.md5.consume(&self.md5_scratch);

        let chans: Vec<Vec<i64>> =
            (0..ch).map(|c| interleaved.iter().skip(c).step_by(ch).map(|&s| i64::from(s)).collect()).collect();
        let level = self.config.level;
        let (assignment, subframes) = if ch == 2 {
            let (l, r) = (&chans[0], &chans[1]);
            let mid: Vec<i64> = l.iter().zip(r).map(|(a, b)| (a + b) >> 1).collect();
            let side: Vec<i64> = l.iter().zip(r).map(|(a, b)| a - b).collect();
            let pl = plan_subframe(l, bps, level);
            let pr = plan_subframe(r, bps, level);
            let pm = plan_subframe(&mid, bps, level);
            let ps = plan_subframe(&side, bps + 1, level);
            let options = [
                (1u8, pl.bits + pr.bits),
                (8, pl.bits + ps.bits),
                (9, ps.bits + pr.bits),
                (10, pm.bits + ps.bits),
            ];
            let best = options.iter().min_by_key(|o| o.1).expect("four options").0;
            let pair = match best {
                1 => vec![pl, pr],
                8 => vec![pl, ps],
                9 => vec![ps, pr],
                _ => vec![pm, ps],
            };
            (best, pair)
        } else {
            ((ch - 1) as u8, chans.iter().map(|c| plan_subframe(c, bps, level)).collect())
        };

        let mut bw = BitWriter::with_capacity(n * ch * bps as usize / 8 + 64);
        write_frame_header(&mut bw, self.frames, n, self.config.sample_rate, assignment, bps);
        for s in &subframes {
            write_subframe(&mut bw, s);
        }
        bw.align();
        let crc = crc16(bw.bytes());
        bw.write(u64::from(crc), 16);
        let frame = bw.into_bytes();

        self.frames += 1;
        self.samples += n as u64;
        self.last_block = n;
        self.min_frame = self.min_frame.min(frame.len() as u32);
        self.max_frame = self.max_frame.max(frame.len() as u32);
        frame
    }
}

/// The frame header (§9.1) of a fixed-block-size stream, CRC-8 included.
fn write_frame_header(bw: &mut BitWriter, frame: u64, n: usize, rate: u32, assignment: u8, bps: u32) {
    let start = bw.len_bits();
    debug_assert_eq!(start % 8, 0);
    let (bs_code, bs_extra) = match n {
        192 => (1, None),
        576 | 1152 | 2304 | 4608 => (2 + (n / 576).trailing_zeros(), None),
        256 | 512 | 1024 | 2048 | 4096 | 8192 | 16384 | 32768 => (8 + (n / 256).trailing_zeros(), None),
        n if n <= 256 => (6, Some(((n - 1) as u64, 8))),
        n => (7, Some(((n - 1) as u64, 16))),
    };
    let (sr_code, sr_extra) = match rate {
        88_200 => (1, None),
        176_400 => (2, None),
        192_000 => (3, None),
        8_000 => (4, None),
        16_000 => (5, None),
        22_050 => (6, None),
        24_000 => (7, None),
        32_000 => (8, None),
        44_100 => (9, None),
        48_000 => (10, None),
        96_000 => (11, None),
        r if r % 1000 == 0 && r / 1000 <= 255 => (12, Some((u64::from(r / 1000), 8))),
        r if r <= 0xFFFF => (13, Some((u64::from(r), 16))),
        r if r % 10 == 0 && r / 10 <= 0xFFFF => (14, Some((u64::from(r / 10), 16))),
        // Only in STREAMINFO.
        _ => (0, None),
    };
    let bps_code = match bps {
        8 => 1,
        12 => 2,
        16 => 4,
        20 => 5,
        24 => 6,
        32 => 7,
        _ => 0,
    };
    bw.write(0x3FFE, 14);
    bw.write(0, 1);
    bw.write(0, 1); // fixed block size
    bw.write(u64::from(bs_code), 4);
    bw.write(sr_code, 4);
    bw.write(u64::from(assignment), 4);
    bw.write(bps_code, 3);
    bw.write(0, 1);
    write_coded_number(bw, frame);
    if let Some((v, n)) = bs_extra {
        bw.write(v, n);
    }
    if let Some((v, n)) = sr_extra {
        bw.write(v, n);
    }
    let bytes = bw.bytes();
    let crc = crc8(&bytes[start / 8..]);
    bw.write(u64::from(crc), 8);
}

/// The UTF-8-style coded number (§9.1.5).
fn write_coded_number(bw: &mut BitWriter, v: u64) {
    if v < 0x80 {
        bw.write(v, 8);
        return;
    }
    // Continuation bytes carry 6 bits each; the lead byte what is left.
    let extra = (1..=6).find(|&k| v < 1u64 << (6 * k + 6 - k)).unwrap_or(6);
    let lead_bits = 6 - extra;
    let marker = (0xFFu64 << (7 - extra)) & 0xFF;
    bw.write(marker | (v >> (6 * extra)) & ((1 << lead_bits) - 1), 8);
    for k in (0..extra).rev() {
        bw.write(0x80 | ((v >> (6 * k)) & 0x3F), 8);
    }
}

/// How one subframe will be written.
struct SubframePlan {
    kind: SubKind,
    /// Wasted low bits shifted out.
    wasted: u32,
    /// Bits per sample after the shift.
    bps: u32,
    /// The samples after the shift (warm-up and verbatim come from here).
    samples: Vec<i64>,
    residual: Vec<i64>,
    rice: Option<RicePlan>,
    /// Exact size in bits.
    bits: usize,
}

enum SubKind {
    Constant,
    Verbatim,
    Fixed(usize),
    Lpc { coefs: Vec<i32>, precision: u32, shift: i32 },
}

impl SubKind {
    fn order(&self) -> usize {
        match self {
            SubKind::Fixed(o) => *o,
            SubKind::Lpc { coefs, .. } => coefs.len(),
            _ => 0,
        }
    }
}

/// Rice partitioning of one residual (§9.2.7).
struct RicePlan {
    order: u32,
    /// Per partition: the Rice parameter, or `None` for the escape (with the
    /// raw bit width).
    params: Vec<Result<u32, u32>>,
    /// Coding method 1 (5-bit parameters) when some parameter needs it.
    wide: bool,
    bits: usize,
}

fn plan_subframe(x: &[i64], bps: u32, level: FlacLevel) -> SubframePlan {
    let n = x.len();
    // Constant: one sample.
    if x.iter().all(|&v| v == x[0]) {
        return SubframePlan {
            kind: SubKind::Constant,
            wasted: 0,
            bps,
            samples: vec![x[0]],
            residual: Vec::new(),
            rice: None,
            bits: 8 + bps as usize,
        };
    }
    let or = x.iter().fold(0i64, |a, &v| a | v);
    let wasted = (or.trailing_zeros()).min(bps - 1);
    let samples: Vec<i64> = if wasted > 0 { x.iter().map(|&v| v >> wasted).collect() } else { x.to_vec() };
    let ebps = bps - wasted;
    // Header: padding bit, type, wasted flag, and the wasted count in unary.
    let header = 8 + wasted as usize;
    let mut best = SubframePlan {
        kind: SubKind::Verbatim,
        wasted,
        bps: ebps,
        samples,
        residual: Vec::new(),
        rice: None,
        bits: header + n * ebps as usize,
    };

    // Fixed predictors. `Fast` estimates the order from the residual sums;
    // the others price every order exactly.
    let fixed: Vec<(usize, Vec<i64>)> = (0..=4.min(n.saturating_sub(1))).map(|o| (o, fixed_residual(&best.samples, o))).collect();
    let candidates: Vec<&(usize, Vec<i64>)> = if level == FlacLevel::Fast {
        fixed
            .iter()
            .min_by_key(|(_, r)| r.iter().map(|v| v.unsigned_abs()).sum::<u64>())
            .into_iter()
            .collect()
    } else {
        fixed.iter().collect()
    };
    for (order, res) in candidates {
        consider(&mut best, SubKind::Fixed(*order), res.clone(), header + *order * ebps as usize, level);
    }

    // LPC.
    let max_order = level.max_lpc_order().min(n.saturating_sub(1));
    if max_order > 0 {
        let window = lpc::tukey(n, 0.5);
        let r = lpc::autocorrelation(&best.samples, &window, max_order);
        let (coefs, errors) = lpc::levinson(&r, max_order);
        let precision: u32 = if ebps <= 16 { 13 } else { 15 };
        let orders: Vec<usize> = if level == FlacLevel::Best {
            (1..=coefs.len()).collect()
        } else {
            // The order whose estimated size — residual entropy from the
            // Levinson error, plus the coefficients — is least.
            let est = |k: usize| -> f64 {
                let e = (errors[k - 1] / n as f64).max(1e-9);
                n as f64 * (0.5 * e.log2()).max(0.0) + (k as f64) * f64::from(precision + ebps)
            };
            (1..=coefs.len()).min_by(|&a, &b| est(a).total_cmp(&est(b))).into_iter().collect()
        };
        for order in orders {
            let (q, shift) = lpc::quantize(&coefs[order - 1], precision, 15);
            let Some(res) = lpc_residual(&best.samples, &q, shift) else { continue };
            let head = header + order * ebps as usize + 4 + 5 + order * precision as usize;
            consider(&mut best, SubKind::Lpc { coefs: q, precision, shift }, res, head, level);
        }
    }
    best
}

/// Replace `best` with the predicted form when its residual codes smaller.
fn consider(best: &mut SubframePlan, kind: SubKind, residual: Vec<i64>, head_bits: usize, level: FlacLevel) {
    // Residuals a decoder cannot hold in 32 bits are not an option.
    if residual.iter().any(|&r| r.unsigned_abs() >= 1 << 30) {
        return;
    }
    let n = best.samples.len();
    let rice = plan_rice(&residual, n, kind.order(), level.max_partition_order());
    let bits = head_bits + rice.bits;
    if bits < best.bits {
        best.kind = kind;
        best.residual = residual;
        best.rice = Some(rice);
        best.bits = bits;
    }
}

fn fixed_residual(x: &[i64], order: usize) -> Vec<i64> {
    (order..x.len())
        .map(|i| {
            x[i] - match order {
                0 => 0,
                1 => x[i - 1],
                2 => 2 * x[i - 1] - x[i - 2],
                3 => 3 * x[i - 1] - 3 * x[i - 2] + x[i - 3],
                _ => 4 * x[i - 1] - 6 * x[i - 2] + 4 * x[i - 3] - x[i - 4],
            }
        })
        .collect()
}

fn lpc_residual(x: &[i64], coefs: &[i32], shift: i32) -> Option<Vec<i64>> {
    let order = coefs.len();
    let mut out = Vec::with_capacity(x.len() - order);
    for i in order..x.len() {
        let mut acc: i64 = 0;
        for (j, &c) in coefs.iter().enumerate() {
            acc = acc.checked_add(i64::from(c).checked_mul(x[i - 1 - j])?)?;
        }
        out.push(x[i] - (acc >> shift));
    }
    Some(out)
}

fn zigzag(r: i64) -> u64 {
    ((r << 1) ^ (r >> 63)) as u64
}

/// Choose the partition order and per-partition parameters for a residual
/// of a block of `n` samples with predictor order `order`.
fn plan_rice(residual: &[i64], n: usize, order: usize, max_order: u32) -> RicePlan {
    let u: Vec<u64> = residual.iter().map(|&r| zigzag(r)).collect();
    // Finest partitioning allowed: the block divides evenly and the first
    // partition still holds more than the warm-up.
    let mut top = 0u32;
    while top < max_order.min(15) && n.is_multiple_of(1 << (top + 1)) && (n >> (top + 1)) > order {
        top += 1;
    }
    // Sums per partition at the finest order, merged pairwise going up.
    let parts = 1usize << top;
    let per = n >> top;
    let mut sums = vec![0u64; parts];
    let mut counts = vec![0usize; parts];
    let mut at = 0usize;
    for p in 0..parts {
        let count = per - if p == 0 { order } else { 0 };
        sums[p] = u[at..at + count].iter().sum();
        counts[p] = count;
        at += count;
    }
    let mut best: Option<RicePlan> = None;
    let mut level = top as i32;
    while level >= 0 {
        let mut plan_params = Vec::with_capacity(sums.len());
        let mut bits = 2 + 4;
        let mut wide = false;
        for (&sum, &count) in sums.iter().zip(&counts) {
            let k = best_k(sum, count);
            wide |= k > 14;
            plan_params.push(k);
            bits += count * (k as usize + 1) + (sum >> k) as usize;
        }
        bits += plan_params.len() * if wide { 5 } else { 4 };
        let candidate = RicePlan { order: level as u32, params: plan_params.into_iter().map(Ok).collect(), wide, bits };
        if best.as_ref().is_none_or(|b| candidate.bits < b.bits) {
            best = Some(candidate);
        }
        if level == 0 {
            break;
        }
        sums = sums.chunks(2).map(|c| c.iter().sum()).collect();
        counts = counts.chunks(2).map(|c| c.iter().sum()).collect();
        level -= 1;
    }
    let mut plan = best.expect("partition order 0 always exists");
    exact_rice(&mut plan, &u, n, order);
    plan
}

/// The Rice parameter minimising `count·(k+1) + sum>>k`.
fn best_k(sum: u64, count: usize) -> u32 {
    if count == 0 || sum == 0 {
        return 0;
    }
    let mean = sum / count as u64;
    let guess = if mean == 0 { 0 } else { 63 - mean.leading_zeros() };
    let cost = |k: u32| count as u64 * u64::from(k + 1) + (sum >> k);
    let mut k = guess.min(30);
    while k > 0 && cost(k - 1) <= cost(k) {
        k -= 1;
    }
    while k < 30 && cost(k + 1) < cost(k) {
        k += 1;
    }
    k
}

/// Re-price the chosen partitioning exactly, taking the raw-bits escape for
/// any partition where it is smaller.
fn exact_rice(plan: &mut RicePlan, u: &[u64], n: usize, order: usize) {
    let per = n >> plan.order;
    let mut at = 0usize;
    let mut bits = 2 + 4;
    for (p, param) in plan.params.iter_mut().enumerate() {
        let count = per - if p == 0 { order } else { 0 };
        let part = &u[at..at + count];
        at += count;
        let k = param.expect("planned as Rice");
        let rice: usize = part.iter().map(|&v| (v >> k) as usize + 1 + k as usize).sum();
        // The escape: 5 bits of width, then every value in that many bits.
        let width = part
            .iter()
            .map(|&v| {
                let r = ((v >> 1) as i64) ^ -((v & 1) as i64);
                if r == 0 { 0 } else { 65 - if r < 0 { r.leading_ones() } else { r.leading_zeros() } }
            })
            .max()
            .unwrap_or(0);
        let escape = 5 + count * width as usize;
        let escape_code = if plan.wide { 31 } else { 15 };
        if escape < rice && width <= 31 || k >= escape_code {
            *param = Err(width);
            bits += escape;
        } else {
            bits += rice;
        }
    }
    bits += plan.params.len() * if plan.wide { 5 } else { 4 };
    plan.bits = bits;
}

fn write_subframe(bw: &mut BitWriter, s: &SubframePlan) {
    let kind = match &s.kind {
        SubKind::Constant => 0,
        SubKind::Verbatim => 1,
        SubKind::Fixed(o) => 8 + *o as u64,
        SubKind::Lpc { coefs, .. } => 32 + coefs.len() as u64 - 1,
    };
    bw.write(0, 1);
    bw.write(kind, 6);
    if s.wasted > 0 {
        bw.write(1, 1);
        bw.write_unary_zeros(s.wasted - 1);
    } else {
        bw.write(0, 1);
    }
    match &s.kind {
        SubKind::Constant => bw.write_signed(s.samples[0], s.bps),
        SubKind::Verbatim => {
            for &v in &s.samples {
                bw.write_signed(v, s.bps);
            }
        }
        SubKind::Fixed(order) => {
            for &v in &s.samples[..*order] {
                bw.write_signed(v, s.bps);
            }
            write_residual(bw, s);
        }
        SubKind::Lpc { coefs, precision, shift } => {
            for &v in &s.samples[..coefs.len()] {
                bw.write_signed(v, s.bps);
            }
            bw.write(u64::from(precision - 1), 4);
            bw.write_signed(i64::from(*shift), 5);
            for &c in coefs {
                bw.write_signed(i64::from(c), *precision);
            }
            write_residual(bw, s);
        }
    }
}

fn write_residual(bw: &mut BitWriter, s: &SubframePlan) {
    let rice = s.rice.as_ref().expect("a predicted subframe has a Rice plan");
    let (param_bits, escape) = if rice.wide { (5, 31) } else { (4, 15) };
    bw.write(u64::from(rice.wide), 2);
    bw.write(u64::from(rice.order), 4);
    let n = s.samples.len();
    let order = s.kind.order();
    let per = n >> rice.order;
    let mut at = 0usize;
    for (p, param) in rice.params.iter().enumerate() {
        let count = per - if p == 0 { order } else { 0 };
        let part = &s.residual[at..at + count];
        at += count;
        match *param {
            Ok(k) => {
                bw.write(u64::from(k), param_bits);
                for &r in part {
                    let u = zigzag(r);
                    bw.write_unary_zeros((u >> k) as u32);
                    bw.write(u & ((1u64 << k) - 1), k);
                }
            }
            Err(width) => {
                bw.write(escape, param_bits);
                bw.write(u64::from(width), 5);
                for &r in part {
                    bw.write_signed(r, width);
                }
            }
        }
    }
}

/// FLAC through the [`AudioEncoder`] surface: pipeline f32 samples are
/// taken at `bits_per_sample` (exactly, for audio that came from integers
/// of at most that depth).
pub struct FlacAudioEncoder {
    inner: FlacEncoder,
    samples_out: u64,
}

impl FlacAudioEncoder {
    pub fn new(config: &AudioEncoderConfig, bits_per_sample: u8, level: FlacLevel) -> Result<Self, AudioError> {
        Ok(Self {
            inner: FlacEncoder::new(FlacEncoderConfig {
                sample_rate: config.sample_rate,
                channels: config.channels,
                bits_per_sample,
                level,
            })?,
            samples_out: 0,
        })
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

impl AudioEncoder for FlacAudioEncoder {
    fn encode(&mut self, frame: &AudioFrame) -> Result<Vec<EncodedAudioPacket>, AudioError> {
        if frame.channels != self.inner.config.channels {
            return Err(AudioError::Encode(format!(
                "flac: a {}-channel frame into a {}-channel encoder",
                frame.channels, self.inner.config.channels
            )));
        }
        if frame.sample_rate != self.inner.config.sample_rate {
            return Err(AudioError::Encode(format!(
                "flac: a {} Hz frame into a {} Hz encoder",
                frame.sample_rate, self.inner.config.sample_rate
            )));
        }
        let bits = u32::from(self.inner.config.bits_per_sample);
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

    /// The metadata blocks (STREAMINFO), final once flushed.
    fn extra_data(&self) -> Vec<u8> {
        self.inner.metadata_blocks()
    }

    fn sample_rate(&self) -> u32 {
        self.inner.config.sample_rate
    }
}
