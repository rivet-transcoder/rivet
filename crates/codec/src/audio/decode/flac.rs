//! FLAC decoder, clean-room from RFC 9639.
//!
//! One packet is one FLAC frame (what an MP4 `fLaC` track or a Matroska
//! `A_FLAC` block holds, and what the native-stream reader cuts); a packet
//! holding several whole frames is decoded frame by frame. Everything the
//! format allows is handled: the four subframe types (constant, verbatim,
//! fixed and LPC prediction), wasted bits, the three stereo decorrelation
//! modes, 4–32 bits per sample, 1–8 channels, fixed and variable block
//! sizes. Both CRCs are checked, and when STREAMINFO carries an MD5 the
//! decoded audio is hashed and compared at the end of the stream.
//!
//! FLAC's channel order for every count (§9.1.3) is the pipeline's native
//! order (mono; FL FR; FL FR FC; FL FR BL BR; FL FR FC BL BR; 5.1 FL FR FC
//! LFE BL BR; 6.1 FL FR FC LFE BC SL SR; 7.1 FL FR FC LFE BL BR SL SR), so
//! the channels pass through in place.

use crate::audio::lossless::bits::BitReader;
use crate::audio::lossless::flac::{StreamInfo, crc8, crc16, md5_bytes, stream_info_from_extra};
use crate::audio::lossless::ints_to_f32;
use crate::audio::{AudioDecoder, AudioError, AudioFrame};

fn err(msg: impl Into<String>) -> AudioError {
    AudioError::Decode(format!("flac: {}", msg.into()))
}

/// The decoded header of one frame (§9.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameHeader {
    /// Variable block size stream: `number` counts samples, not frames.
    pub variable_block_size: bool,
    pub block_size: u32,
    pub sample_rate: u32,
    /// Channel assignment code (§9.1.3): 0–7 independent channels, 8
    /// left/side, 9 side/right, 10 mid/side.
    pub assignment: u8,
    pub channels: u8,
    pub bits_per_sample: u32,
    /// Frame number, or first sample number for a variable block size.
    pub number: u64,
    /// Bytes the header took, CRC-8 included.
    pub len: usize,
}

/// Parse the frame header at the start of `data`, checking its CRC-8. The
/// sample rate and bit depth a header defers to STREAMINFO come from `info`.
pub fn parse_frame_header(data: &[u8], info: Option<&StreamInfo>) -> Result<FrameHeader, AudioError> {
    let mut br = BitReader::new(data, "flac");
    if br.read(14)? != 0x3FFE {
        return Err(err("no frame sync code"));
    }
    if br.read_bit()? {
        return Err(err("reserved bit after the sync code is set"));
    }
    let variable_block_size = br.read_bit()?;
    let bs_code = br.read_u32(4)?;
    let sr_code = br.read_u32(4)?;
    let assignment = br.read_u32(4)? as u8;
    let bps_code = br.read_u32(3)?;
    if br.read_bit()? {
        return Err(err("reserved bit in the frame header is set"));
    }
    // The coded number, UTF-8 style (§9.1.5): up to 36 bits in 7 bytes.
    let first = br.read_u32(8)? as u8;
    let (extra, mut number) = match first.leading_ones() {
        0 => (0, u64::from(first)),
        lead @ 2..=7 => (lead - 1, u64::from(first) & (0xFF >> (lead + 1))),
        _ => return Err(err(format!("invalid coded number lead byte {first:#04x}"))),
    };
    for _ in 0..extra {
        let b = br.read_u32(8)?;
        if b & 0xC0 != 0x80 {
            return Err(err("invalid coded number continuation byte"));
        }
        number = (number << 6) | u64::from(b & 0x3F);
    }
    let block_size = match bs_code {
        0 => return Err(err("reserved block size code 0")),
        1 => 192,
        2..=5 => 576 << (bs_code - 2),
        6 => br.read_u32(8)? + 1,
        7 => br.read_u32(16)? + 1,
        _ => 256 << (bs_code - 8),
    };
    let sample_rate = match sr_code {
        0 => info.map(|i| i.sample_rate).ok_or_else(|| err("frame defers its sample rate to a STREAMINFO there is none of"))?,
        1 => 88_200,
        2 => 176_400,
        3 => 192_000,
        4 => 8_000,
        5 => 16_000,
        6 => 22_050,
        7 => 24_000,
        8 => 32_000,
        9 => 44_100,
        10 => 48_000,
        11 => 96_000,
        12 => br.read_u32(8)? * 1000,
        13 => br.read_u32(16)?,
        14 => br.read_u32(16)? * 10,
        _ => return Err(err("invalid sample rate code 15")),
    };
    let bits_per_sample = match bps_code {
        0 => u32::from(
            info.map(|i| i.bits_per_sample)
                .ok_or_else(|| err("frame defers its bit depth to a STREAMINFO there is none of"))?,
        ),
        1 => 8,
        2 => 12,
        4 => 16,
        5 => 20,
        6 => 24,
        7 => 32,
        _ => return Err(err("reserved bit depth code 3")),
    };
    let channels = match assignment {
        0..=7 => assignment + 1,
        8..=10 => 2,
        _ => return Err(err(format!("reserved channel assignment {assignment}"))),
    };
    let header_bytes = br.pos() / 8;
    let crc = br.read_u32(8)? as u8;
    if crc8(&data[..header_bytes]) != crc {
        return Err(err("frame header CRC-8 mismatch"));
    }
    Ok(FrameHeader {
        variable_block_size,
        block_size,
        sample_rate,
        assignment,
        channels,
        bits_per_sample,
        number,
        len: header_bytes + 1,
    })
}

/// One decoded frame: interleaved samples at the frame's bit depth.
#[derive(Clone, Debug)]
pub struct DecodedFrame {
    pub header: FrameHeader,
    /// Interleaved, `block_size × channels`.
    pub samples: Vec<i32>,
    /// Bytes the frame took, CRC-16 included.
    pub len: usize,
}

/// Decode the frame at the start of `data`.
pub fn decode_frame(data: &[u8], info: Option<&StreamInfo>) -> Result<DecodedFrame, AudioError> {
    let header = parse_frame_header(data, info)?;
    let mut br = BitReader::new(data, "flac");
    br.skip(header.len * 8)?;
    let n = header.block_size as usize;
    let bps = header.bits_per_sample;
    let mut chans: Vec<Vec<i64>> = Vec::with_capacity(usize::from(header.channels));
    for c in 0..header.channels {
        // The side channel carries one more bit (§9.2).
        let side = matches!((header.assignment, c), (8, 1) | (9, 0) | (10, 1));
        chans.push(decode_subframe(&mut br, n, bps + u32::from(side))?);
    }
    br.align();
    let body = br.byte_pos();
    let crc = br.read_u32(16)? as u16;
    if crc16(&data[..body]) != crc {
        return Err(err("frame CRC-16 mismatch"));
    }
    match header.assignment {
        8 => {
            let (l, s) = chans.split_at_mut(1);
            for (s, &l) in s[0].iter_mut().zip(&l[0]) {
                *s = l - *s;
            }
        }
        9 => {
            let (s, r) = chans.split_at_mut(1);
            for (s, &r) in s[0].iter_mut().zip(&r[0]) {
                *s += r;
            }
        }
        10 => {
            let (m, s) = chans.split_at_mut(1);
            for (m, s) in m[0].iter_mut().zip(s[0].iter_mut()) {
                let mid = (*m << 1) | (*s & 1);
                let side = *s;
                *m = (mid + side) >> 1;
                *s = (mid - side) >> 1;
            }
        }
        _ => {}
    }
    let mut samples = Vec::with_capacity(n * chans.len());
    for i in 0..n {
        for ch in &chans {
            samples.push(ch[i] as i32);
        }
    }
    Ok(DecodedFrame { header, samples, len: body + 2 })
}

fn decode_subframe(br: &mut BitReader<'_>, n: usize, bps: u32) -> Result<Vec<i64>, AudioError> {
    if br.read_bit()? {
        return Err(err("subframe padding bit is set"));
    }
    let kind = br.read_u32(6)?;
    let wasted = if br.read_bit()? { br.read_unary_zeros()? + 1 } else { 0 };
    if wasted >= bps {
        return Err(err(format!("{wasted} wasted bits of a {bps}-bit subframe")));
    }
    let bps = bps - wasted;
    let mut out = match kind {
        0 => vec![br.read_signed(bps)?; n],
        1 => (0..n).map(|_| br.read_signed(bps)).collect::<Result<_, _>>()?,
        8..=12 => {
            let order = (kind - 8) as usize;
            let mut s = warmup(br, n, order, bps)?;
            decode_residual(br, n, order, &mut s)?;
            restore_fixed(&mut s, order);
            s
        }
        32..=63 => {
            let order = (kind - 31) as usize;
            let mut s = warmup(br, n, order, bps)?;
            let precision = br.read_u32(4)? + 1;
            if precision == 16 {
                return Err(err("invalid LPC coefficient precision"));
            }
            let shift = br.read_signed(5)?;
            if shift < 0 {
                return Err(err(format!("negative LPC shift {shift}")));
            }
            let coefs: Vec<i64> = (0..order).map(|_| br.read_signed(precision)).collect::<Result<_, _>>()?;
            decode_residual(br, n, order, &mut s)?;
            restore_lpc(&mut s, &coefs, shift as u32);
            s
        }
        _ => return Err(err(format!("reserved subframe type {kind}"))),
    };
    if wasted > 0 {
        for s in &mut out {
            *s <<= wasted;
        }
    }
    Ok(out)
}

fn warmup(br: &mut BitReader<'_>, n: usize, order: usize, bps: u32) -> Result<Vec<i64>, AudioError> {
    if order > n {
        return Err(err(format!("predictor order {order} exceeds the block size {n}")));
    }
    let mut s = Vec::with_capacity(n);
    for _ in 0..order {
        s.push(br.read_signed(bps)?);
    }
    Ok(s)
}

/// Append the block's residual (§9.2.7) to `out`, which holds the warm-up.
fn decode_residual(br: &mut BitReader<'_>, n: usize, order: usize, out: &mut Vec<i64>) -> Result<(), AudioError> {
    let (param_bits, escape) = match br.read_u32(2)? {
        0 => (4, 15),
        1 => (5, 31),
        m => return Err(err(format!("reserved residual coding method {m}"))),
    };
    let partition_order = br.read_u32(4)?;
    let partitions = 1usize << partition_order;
    if !n.is_multiple_of(partitions) || n >> partition_order < order {
        return Err(err(format!(
            "partition order {partition_order} does not fit a {n}-sample block of order {order}"
        )));
    }
    for p in 0..partitions {
        let count = (n >> partition_order) - if p == 0 { order } else { 0 };
        let k = br.read_u32(param_bits)?;
        if k == escape {
            let raw = br.read_u32(5)?;
            for _ in 0..count {
                out.push(br.read_signed(raw)?);
            }
        } else {
            for _ in 0..count {
                let q = u64::from(br.read_unary_zeros()?);
                let u = (q << k) | br.read(k)?;
                out.push((u >> 1) as i64 ^ -((u & 1) as i64));
            }
        }
    }
    Ok(())
}

/// The fixed predictors (§9.2.5), undone in place over the residual.
fn restore_fixed(s: &mut [i64], order: usize) {
    for i in order..s.len() {
        let p = match order {
            0 => 0,
            1 => s[i - 1],
            2 => 2 * s[i - 1] - s[i - 2],
            3 => 3 * s[i - 1] - 3 * s[i - 2] + s[i - 3],
            _ => 4 * s[i - 1] - 6 * s[i - 2] + 4 * s[i - 3] - s[i - 4],
        };
        s[i] += p;
    }
}

/// The LPC predictor (§9.2.6), undone in place: `coefs[j]` weighs the
/// sample `j + 1` back.
fn restore_lpc(s: &mut [i64], coefs: &[i64], shift: u32) {
    let order = coefs.len();
    for i in order..s.len() {
        let mut acc: i64 = 0;
        for (j, &c) in coefs.iter().enumerate() {
            acc = acc.wrapping_add(c.wrapping_mul(s[i - 1 - j]));
        }
        s[i] += acc >> shift;
    }
}

/// FLAC through the [`AudioDecoder`] surface.
pub struct FlacDecoder {
    info: Option<StreamInfo>,
    /// Running MD5 of the decoded audio, when STREAMINFO has one to check.
    md5: Option<md5::Context>,
    md5_scratch: Vec<u8>,
    samples_decoded: u64,
    first_pts_us: Option<i64>,
    /// Bit depth and rate of the last frame, for the f32 conversion.
    bits: u32,
    sample_rate: u32,
    channels: u8,
}

impl FlacDecoder {
    /// `extra_data` is the codec configuration: an MP4 `dfLa` body or a
    /// Matroska CodecPrivate (`fLaC` + metadata blocks). Without it, every
    /// frame header has to be self-describing.
    pub fn new(extra_data: Option<&[u8]>, sample_rate: u32, channels: u8) -> Result<Self, AudioError> {
        let info = match extra_data {
            Some(e) if !e.is_empty() => Some(stream_info_from_extra(e)?),
            _ => None,
        };
        let md5 = info.as_ref().filter(|i| i.md5 != [0; 16]).map(|_| md5::Context::new());
        Ok(Self {
            bits: info.as_ref().map_or(16, |i| u32::from(i.bits_per_sample)),
            sample_rate: info.as_ref().map_or(sample_rate, |i| i.sample_rate),
            channels: info.as_ref().map_or(channels, |i| i.channels),
            info,
            md5,
            md5_scratch: Vec::new(),
            samples_decoded: 0,
            first_pts_us: None,
        })
    }

    /// The stream's STREAMINFO, when the configuration carried one.
    pub fn stream_info(&self) -> Option<&StreamInfo> {
        self.info.as_ref()
    }

    /// Decode every frame in `packet` to interleaved integer samples, with
    /// the channel count and bit depth they are at.
    pub fn decode_int(&mut self, packet: &[u8]) -> Result<(Vec<i32>, u8, u32), AudioError> {
        let mut at = 0usize;
        let mut out = Vec::new();
        while packet.len() - at >= 2 && packet[at] == 0xFF && packet[at + 1] & 0xFE == 0xF8 {
            let frame = decode_frame(&packet[at..], self.info.as_ref())?;
            at += frame.len;
            let h = frame.header;
            if !out.is_empty() && (h.channels != self.channels || h.bits_per_sample != self.bits) {
                return Err(err("frames of one packet change channel count or bit depth"));
            }
            self.channels = h.channels;
            self.bits = h.bits_per_sample;
            self.sample_rate = h.sample_rate;
            if let Some(ctx) = self.md5.as_mut() {
                self.md5_scratch.clear();
                md5_bytes(&frame.samples, h.bits_per_sample, &mut self.md5_scratch);
                ctx.consume(&self.md5_scratch);
            }
            self.samples_decoded += u64::from(h.block_size);
            out.extend_from_slice(&frame.samples);
        }
        if at == 0 && !packet.is_empty() {
            return Err(err("packet does not start with a frame"));
        }
        Ok((out, self.channels, self.bits))
    }

    /// Whether the audio decoded so far hashes to STREAMINFO's MD5: `None`
    /// when there is none to compare, or the stream has not been decoded to
    /// its stated end.
    pub fn md5_matches(&self) -> Option<bool> {
        let info = self.info.as_ref()?;
        let ctx = self.md5.as_ref()?;
        if info.total_samples == 0 || info.total_samples != self.samples_decoded {
            return None;
        }
        Some(ctx.clone().compute().0 == info.md5)
    }
}

impl AudioDecoder for FlacDecoder {
    fn decode(&mut self, packet: &[u8], pts: i64) -> Result<Vec<AudioFrame>, AudioError> {
        let first_pts_us = *self.first_pts_us.get_or_insert(pts);
        let before = self.samples_decoded;
        let (samples, channels, bits) = self.decode_int(packet)?;
        if samples.is_empty() {
            return Ok(Vec::new());
        }
        let pts = first_pts_us + (before as i64 * 1_000_000) / i64::from(self.sample_rate.max(1));
        Ok(vec![AudioFrame { samples: ints_to_f32(&samples, bits), sample_rate: self.sample_rate, channels, pts }])
    }

    fn flush(&mut self) -> Result<Vec<AudioFrame>, AudioError> {
        if self.md5_matches() == Some(false) {
            tracing::warn!("flac: the decoded audio does not match the stream's MD5 signature");
        }
        Ok(Vec::new())
    }
}
