//! ALAC format definitions shared by the decoder and the encoder: the magic
//! cookie (`ALACSpecificConfig`), the adaptive Golomb-Rice entropy coder and
//! the adaptive linear predictor.
//!
//! Written from the published description of the Apple Lossless format —
//! the magic cookie layout, the channel layouts, the frame element syntax
//! and the coding scheme — not from any implementation's source.
//!
//! # The two adaptive stages
//!
//! **Prediction.** A sign-sign LMS FIR filter: each sample is predicted from
//! the previous `order` samples, all taken relative to the oldest sample in
//! the window (`top`), with integer coefficients in units of `2^-den_shift`.
//! After each sample the coefficients step by ±1 toward reducing the error,
//! oldest tap first, until the step has used up the error. The first
//! `order + 1` samples of a block are coded as first differences.
//!
//! **Entropy coding.** A Rice code whose parameter follows a running mean of
//! the coded magnitudes (`history`), with an escape to raw bits for large
//! values and, when the mean falls low, a run-length code for a run of zero
//! samples.
//!
//! Both sides run the predictor in exact integer arithmetic; the encoder
//! rejects (and codes some other way) a block whose arithmetic would leave
//! the 32-bit range a reference decoder works in, so what it writes decodes
//! identically everywhere.

use super::bits::{BitReader, BitWriter};
use super::sign_extend;
use crate::audio::AudioError;

/// Element tags in a frame.
pub const ID_SCE: u32 = 0;
pub const ID_CPE: u32 = 1;
pub const ID_CCE: u32 = 2;
pub const ID_LFE: u32 = 3;
pub const ID_DSE: u32 = 4;
pub const ID_PCE: u32 = 5;
pub const ID_FIL: u32 = 6;
pub const ID_END: u32 = 7;

/// Default samples per frame.
pub const DEFAULT_FRAME_LENGTH: u32 = 4096;
/// Default Rice history multiplier, initial history and parameter limit.
pub const DEFAULT_PB: u8 = 40;
pub const DEFAULT_MB: u8 = 10;
pub const DEFAULT_KB: u8 = 14;
pub const DEFAULT_MAX_RUN: u16 = 255;

/// Unary prefixes at or past this many ones escape to a raw value.
const MAX_PREFIX: u32 = 9;
/// History is kept as a mean in units of 2^-9.
const QB_SHIFT: u32 = 9;
/// Coded values past this clamp the history.
const N_MAX_MEAN_CLAMP: u32 = 0xFFFF;

/// `ALACSpecificConfig`, the 24-byte magic cookie.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AlacConfig {
    pub frame_length: u32,
    pub compatible_version: u8,
    pub bit_depth: u8,
    pub pb: u8,
    pub mb: u8,
    pub kb: u8,
    pub num_channels: u8,
    pub max_run: u16,
    pub max_frame_bytes: u32,
    pub avg_bit_rate: u32,
    pub sample_rate: u32,
}

impl AlacConfig {
    pub const LEN: usize = 24;

    /// A cookie for the given stream, with the default coding parameters.
    pub fn new(sample_rate: u32, channels: u8, bit_depth: u8) -> Self {
        Self {
            frame_length: DEFAULT_FRAME_LENGTH,
            compatible_version: 0,
            bit_depth,
            pb: DEFAULT_PB,
            mb: DEFAULT_MB,
            kb: DEFAULT_KB,
            num_channels: channels,
            max_run: DEFAULT_MAX_RUN,
            max_frame_bytes: 0,
            avg_bit_rate: 0,
            sample_rate,
        }
    }

    /// Parse the cookie out of whatever a container hands over: the bare 24
    /// bytes, the 28-byte `alac` FullBox body (version and flags first), a
    /// whole `alac` atom (size + `alac` + version/flags + config), or a
    /// QuickTime `wave` atom holding one.
    pub fn parse(extra: &[u8]) -> Result<Self, AudioError> {
        let config = if extra.len() == Self::LEN {
            extra
        } else if extra.len() == Self::LEN + 4 && extra[..4] == [0, 0, 0, 0] {
            &extra[4..]
        } else if let Some(i) = extra.windows(4).position(|w| w == b"alac") {
            // `alac` atom: 4-byte size before the type, 4 bytes of
            // version/flags after it.
            extra.get(i + 8..i + 8 + Self::LEN).ok_or_else(|| {
                AudioError::Decode("alac: magic cookie truncated after its atom header".into())
            })?
        } else {
            return Err(AudioError::Decode(format!(
                "alac: {}-byte codec configuration is not an ALACSpecificConfig",
                extra.len()
            )));
        };
        let be32 = |i: usize| u32::from_be_bytes(config[i..i + 4].try_into().expect("4 bytes"));
        let c = Self {
            frame_length: be32(0),
            compatible_version: config[4],
            bit_depth: config[5],
            pb: config[6],
            mb: config[7],
            kb: config[8],
            num_channels: config[9],
            max_run: u16::from_be_bytes([config[10], config[11]]),
            max_frame_bytes: be32(12),
            avg_bit_rate: be32(16),
            sample_rate: be32(20),
        };
        if c.compatible_version != 0 {
            return Err(AudioError::Unsupported(format!(
                "alac: magic cookie compatible version {}",
                c.compatible_version
            )));
        }
        if !matches!(c.bit_depth, 16 | 20 | 24 | 32) {
            return Err(AudioError::Unsupported(format!("alac: bit depth {}", c.bit_depth)));
        }
        if !(1..=8).contains(&c.num_channels) {
            return Err(AudioError::Unsupported(format!("alac: {} channels", c.num_channels)));
        }
        if c.frame_length == 0 || c.frame_length > 1 << 16 {
            return Err(AudioError::Decode(format!("alac: frame length {}", c.frame_length)));
        }
        if c.kb == 0 || c.kb > 31 {
            return Err(AudioError::Decode(format!("alac: Rice limit {}", c.kb)));
        }
        Ok(c)
    }

    pub fn to_bytes(&self) -> [u8; Self::LEN] {
        let mut b = [0u8; Self::LEN];
        b[0..4].copy_from_slice(&self.frame_length.to_be_bytes());
        b[4] = self.compatible_version;
        b[5] = self.bit_depth;
        b[6] = self.pb;
        b[7] = self.mb;
        b[8] = self.kb;
        b[9] = self.num_channels;
        b[10..12].copy_from_slice(&self.max_run.to_be_bytes());
        b[12..16].copy_from_slice(&self.max_frame_bytes.to_be_bytes());
        b[16..20].copy_from_slice(&self.avg_bit_rate.to_be_bytes());
        b[20..24].copy_from_slice(&self.sample_rate.to_be_bytes());
        b
    }
}

/// The elements a frame of `channels` channels is made of, in order, as
/// (tag, channels in it). The channel layouts ALAC defines for each count
/// are: mono; L R; C L R; C L R Cs; C L R Ls Rs; C L R Ls Rs LFE;
/// C L R Ls Rs Cs LFE; C Lc Rc L R Ls Rs LFE.
pub fn element_layout(channels: u8) -> &'static [(u32, usize)] {
    const S: (u32, usize) = (ID_SCE, 1);
    const C: (u32, usize) = (ID_CPE, 2);
    const L: (u32, usize) = (ID_LFE, 1);
    match channels {
        1 => &[S],
        2 => &[C],
        3 => &[S, C],
        4 => &[S, C, S],
        5 => &[S, C, C],
        6 => &[S, C, C, L],
        7 => &[S, C, C, S, L],
        _ => &[S, C, C, C, L],
    }
}

/// For each pipeline (native-order) channel, the ALAC channel it is. The
/// native layouts these land on: 3.0 FL FR FC; 4.0 FL FR FC BC; 5.0 FL FR FC
/// BL BR; 5.1 FL FR FC LFE BL BR; 6.1 FL FR FC LFE BC SL SR; and for eight
/// channels 7.1(wide) FL FR FC LFE BL BR FLC FRC, ALAC's Lc and Rc being
/// front left- and right-of-centre.
pub fn native_from_alac(channels: u8) -> &'static [usize] {
    match channels {
        1 => &[0],
        2 => &[0, 1],
        3 => &[1, 2, 0],
        4 => &[1, 2, 0, 3],
        5 => &[1, 2, 0, 3, 4],
        6 => &[1, 2, 0, 5, 3, 4],
        7 => &[1, 2, 0, 6, 5, 3, 4],
        _ => &[3, 4, 0, 7, 5, 6, 1, 2],
    }
}

/// The adaptive Rice coder's parameters for one channel of one block.
#[derive(Clone, Copy, Debug)]
pub struct RiceParams {
    /// History multiplier: the cookie's `pb` scaled by the channel's
    /// `pbFactor` (in quarters).
    pub pb: u32,
    /// Initial history.
    pub mb: u32,
    /// Parameter limit.
    pub kb: u32,
}

impl RiceParams {
    pub fn new(config: &AlacConfig, pb_factor: u32) -> Self {
        Self {
            pb: u32::from(config.pb) * pb_factor / 4,
            mb: u32::from(config.mb),
            kb: u32::from(config.kb),
        }
    }
}

/// `k` for a sample: log2 of the running mean, limited to `kb`.
fn sample_k(history: u32, kb: u32) -> u32 {
    (31 - ((history >> QB_SHIFT) + 3).leading_zeros()).min(kb)
}

/// `k` for a zero-run length.
fn run_k(history: u32) -> u32 {
    history.leading_zeros() - 24 + ((history + 16) >> 6)
}

fn read_code(br: &mut BitReader<'_>, k: u32, escape_bits: u32) -> Result<u32, AudioError> {
    let prefix = br.read_unary_ones(MAX_PREFIX)?;
    if prefix >= MAX_PREFIX {
        return Ok(br.read(escape_bits)? as u32);
    }
    if k <= 1 {
        return Ok(prefix);
    }
    let m = (1u32 << k) - 1;
    let v = br.read_u32(k)?;
    if v >= 2 {
        Ok(prefix * m + v - 1)
    } else {
        br.unread(1);
        Ok(prefix * m)
    }
}

fn write_code(bw: &mut BitWriter, value: u32, k: u32, escape_bits: u32) {
    let m = (1u32 << k) - 1;
    let q = if k <= 1 { value } else { value / m };
    if q >= MAX_PREFIX {
        bw.write((1u64 << MAX_PREFIX) - 1, MAX_PREFIX);
        bw.write(u64::from(value), escape_bits);
        return;
    }
    // q ones, then a zero.
    bw.write(((1u64 << q) - 1) << 1, q + 1);
    if k <= 1 {
        return;
    }
    let r = value % m;
    if r == 0 {
        bw.write(0, k - 1);
    } else {
        bw.write(u64::from(r + 1), k);
    }
}

/// Decode `n` residuals of up to `sample_bits` bits.
pub(crate) fn decode_residuals(
    br: &mut BitReader<'_>,
    p: &RiceParams,
    n: usize,
    sample_bits: u32,
) -> Result<Vec<i32>, AudioError> {
    let mut out = vec![0i32; n];
    let mut history = p.mb;
    let mut sign_modifier = 0u32;
    let mut i = 0usize;
    while i < n {
        let k = sample_k(history, p.kb);
        let coded = read_code(br, k, sample_bits)?;
        let v = coded.wrapping_add(sign_modifier);
        // Even values are the non-negative residuals, odd ones the negative.
        out[i] = if v & 1 == 1 { -(((v >> 1) + 1) as i64) as i32 } else { (v >> 1) as i32 };
        i += 1;
        sign_modifier = 0;
        history = if coded > N_MAX_MEAN_CLAMP {
            N_MAX_MEAN_CLAMP
        } else {
            history.wrapping_add(coded.wrapping_mul(p.pb)).wrapping_sub((history.wrapping_mul(p.pb)) >> QB_SHIFT)
        };
        if history < 128 && i < n {
            let k = run_k(history).min(p.kb);
            let run = read_code(br, k, 16)? as usize;
            if run > n - i {
                return Err(AudioError::Decode(format!(
                    "alac: a run of {run} zeros overruns the block ({} samples left)",
                    n - i
                )));
            }
            i += run;
            if run < 0xFFFF {
                sign_modifier = 1;
            }
            history = 0;
        }
    }
    Ok(out)
}

/// Encode `residuals` with the same adaptation [`decode_residuals`] follows.
pub(crate) fn encode_residuals(bw: &mut BitWriter, p: &RiceParams, residuals: &[i32], sample_bits: u32) {
    let n = residuals.len();
    let mut history = p.mb;
    let mut sign_modifier = 0u32;
    let mut i = 0usize;
    while i < n {
        let r = residuals[i];
        let v = if r < 0 { (-2 * i64::from(r) - 1) as u32 } else { (2 * i64::from(r)) as u32 };
        let coded = v.wrapping_sub(sign_modifier);
        let k = sample_k(history, p.kb);
        write_code(bw, coded, k, sample_bits);
        i += 1;
        sign_modifier = 0;
        history = if coded > N_MAX_MEAN_CLAMP {
            N_MAX_MEAN_CLAMP
        } else {
            history.wrapping_add(coded.wrapping_mul(p.pb)).wrapping_sub((history.wrapping_mul(p.pb)) >> QB_SHIFT)
        };
        if history < 128 && i < n {
            let k = run_k(history).min(p.kb);
            let mut run = 0usize;
            while i + run < n && residuals[i + run] == 0 && run < 0xFFFF {
                run += 1;
            }
            write_code(bw, run as u32, k, 16);
            i += run;
            if run < 0xFFFF {
                sign_modifier = 1;
            }
            history = 0;
        }
    }
}

/// Bits [`encode_residuals`] would write, without writing them.
pub(crate) fn residual_bits(p: &RiceParams, residuals: &[i32], sample_bits: u32) -> usize {
    let code_bits = |value: u32, k: u32, escape_bits: u32| -> usize {
        let m = (1u32 << k) - 1;
        let q = if k <= 1 { value } else { value / m };
        if q >= MAX_PREFIX {
            return (MAX_PREFIX + escape_bits) as usize;
        }
        let tail = if k <= 1 { 0 } else if value.is_multiple_of(m) { k - 1 } else { k };
        (q + 1 + tail) as usize
    };
    let n = residuals.len();
    let mut bits = 0usize;
    let mut history = p.mb;
    let mut sign_modifier = 0u32;
    let mut i = 0usize;
    while i < n {
        let r = residuals[i];
        let v = if r < 0 { (-2 * i64::from(r) - 1) as u32 } else { (2 * i64::from(r)) as u32 };
        let coded = v.wrapping_sub(sign_modifier);
        bits += code_bits(coded, sample_k(history, p.kb), sample_bits);
        i += 1;
        sign_modifier = 0;
        history = if coded > N_MAX_MEAN_CLAMP {
            N_MAX_MEAN_CLAMP
        } else {
            history.wrapping_add(coded.wrapping_mul(p.pb)).wrapping_sub((history.wrapping_mul(p.pb)) >> QB_SHIFT)
        };
        if history < 128 && i < n {
            let k = run_k(history).min(p.kb);
            let mut run = 0usize;
            while i + run < n && residuals[i + run] == 0 && run < 0xFFFF {
                run += 1;
            }
            bits += code_bits(run as u32, k, 16);
            i += run;
            if run < 0xFFFF {
                sign_modifier = 1;
            }
            history = 0;
        }
    }
    bits
}

fn sign(v: i64) -> i64 {
    v.signum()
}

/// The predictor's state across one block of one channel.
struct Lms {
    coefs: Vec<i64>,
    den_shift: u32,
    /// Whether every product and sum stayed within the 32-bit range.
    in_range: bool,
}

impl Lms {
    /// The prediction for sample `i` of `out` (`i > order`).
    fn predict(&mut self, out: &[i64], i: usize) -> i64 {
        let order = self.coefs.len();
        let top = out[i - order - 1];
        let mut sum: i64 = 0;
        for (j, &c) in self.coefs.iter().enumerate() {
            let term = c * (out[i - 1 - j] - top);
            sum += term;
        }
        if sum > i64::from(i32::MAX) || sum < i64::from(i32::MIN) {
            self.in_range = false;
        }
        let rounding = if self.den_shift > 0 { 1i64 << (self.den_shift - 1) } else { 0 };
        top + ((sum + rounding) >> self.den_shift)
    }

    /// Step the coefficients after sample `i` came out with error `err`.
    fn adapt(&mut self, out: &[i64], i: usize, err: i64) {
        let order = self.coefs.len();
        let top = out[i - order - 1];
        let s = sign(err);
        if s == 0 {
            return;
        }
        let mut left = err;
        for j in (0..order).rev() {
            let d = top - out[i - 1 - j];
            let sd = sign(d);
            self.coefs[j] -= s * sd;
            left -= (order - j) as i64 * ((s * sd * d) >> self.den_shift);
            if s > 0 && left <= 0 || s < 0 && left >= 0 {
                break;
            }
        }
    }
}

/// Undo the predictor in place: `data` holds residuals and becomes samples
/// of `bits` bits. `order` 31 is the plain first-order integrator.
pub fn unpredict(data: &mut [i64], coefs: &[i16], order: usize, den_shift: u32, bits: u32) {
    let n = data.len();
    if n == 0 || order == 0 {
        return;
    }
    if order == 31 {
        for i in 1..n {
            data[i] = sign_extend(data[i] + data[i - 1], bits);
        }
        return;
    }
    let warm = (order + 1).min(n);
    for i in 1..warm {
        data[i] = sign_extend(data[i] + data[i - 1], bits);
    }
    let mut lms = Lms { coefs: coefs.iter().map(|&c| i64::from(c)).collect(), den_shift, in_range: true };
    for i in warm..n {
        let err = data[i];
        let pred = lms.predict(data, i);
        data[i] = sign_extend(pred + err, bits);
        lms.adapt(data, i, err);
    }
}

/// Run the predictor forward over `samples` (of `bits` bits): the
/// residuals [`unpredict`] turns back into them, or `None` when the
/// arithmetic would leave the range a 32-bit decoder computes in.
pub fn predict(samples: &[i64], coefs: &[i16], den_shift: u32, bits: u32) -> Option<Vec<i32>> {
    let n = samples.len();
    let order = coefs.len();
    let mut res: Vec<i64> = Vec::with_capacity(n);
    if n == 0 {
        return Some(Vec::new());
    }
    res.push(samples[0]);
    if order == 0 {
        res.extend_from_slice(&samples[1..]);
    } else {
        let warm = (order + 1).min(n);
        for i in 1..warm {
            res.push(sign_extend(samples[i] - samples[i - 1], bits));
        }
        let mut lms = Lms { coefs: coefs.iter().map(|&c| i64::from(c)).collect(), den_shift, in_range: true };
        for i in warm..n {
            let pred = lms.predict(samples, i);
            let err = sign_extend(samples[i] - pred, bits);
            res.push(err);
            lms.adapt(samples, i, err);
            if !lms.in_range {
                return None;
            }
        }
    }
    res.into_iter().map(|r| i32::try_from(r).ok()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> RiceParams {
        RiceParams::new(&AlacConfig::new(44_100, 2, 16), 4)
    }

    #[test]
    fn cookie_round_trips_in_every_wrapping() {
        let c = AlacConfig { max_frame_bytes: 9000, avg_bit_rate: 800_000, ..AlacConfig::new(96_000, 6, 24) };
        let bare = c.to_bytes();
        assert_eq!(AlacConfig::parse(&bare).unwrap(), c);
        let mut fullbox = vec![0, 0, 0, 0];
        fullbox.extend_from_slice(&bare);
        assert_eq!(AlacConfig::parse(&fullbox).unwrap(), c);
        let mut atom = 36u32.to_be_bytes().to_vec();
        atom.extend_from_slice(b"alac");
        atom.extend_from_slice(&fullbox);
        assert_eq!(AlacConfig::parse(&atom).unwrap(), c);
    }

    #[test]
    fn residuals_round_trip_through_runs_and_escapes() {
        let mut r: Vec<i32> = Vec::new();
        r.extend([0; 300]);
        r.extend([1, -1, 2, -2, 3, 0, 0, 0, 0, 5]);
        r.extend((0..500).map(|i| ((i * 7919) % 2001) - 1000));
        r.extend([32767, -32768, 0, 0, 1]);
        r.extend([0; 70_000]);
        r.push(-3);
        let p = params();
        let mut bw = BitWriter::default();
        encode_residuals(&mut bw, &p, &r, 17);
        let bits = bw.len_bits();
        assert_eq!(bits, residual_bits(&p, &r, 17));
        let bytes = bw.into_bytes();
        let mut br = BitReader::new(&bytes, "alac");
        assert_eq!(decode_residuals(&mut br, &p, r.len(), 17).unwrap(), r);
        assert_eq!(br.pos(), bits);
    }

    #[test]
    fn the_predictor_inverts() {
        let x: Vec<i64> = (0..4096).map(|i| ((i as f64 * 0.05).sin() * 20_000.0) as i64 + (i % 7) as i64).collect();
        for (coefs, shift) in [(vec![], 9), (vec![1000i16, -500, 100, 20], 9), (vec![512; 8], 9)] {
            let res = predict(&x, &coefs, shift, 17).expect("in range");
            let mut back: Vec<i64> = res.iter().map(|&r| i64::from(r)).collect();
            unpredict(&mut back, &coefs, coefs.len(), shift, 17);
            assert_eq!(back, x, "{coefs:?}");
        }
    }
}
