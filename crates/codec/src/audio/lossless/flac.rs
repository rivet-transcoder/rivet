//! FLAC format definitions shared by the decoder and the encoder: the
//! STREAMINFO block, the metadata block framing, and the two CRCs.
//!
//! Written from the format specification, RFC 9639 ("Free Lossless Audio
//! Codec"); section numbers below refer to it.

use crate::audio::AudioError;

/// Metadata block types (§8.1).
pub const BLOCK_STREAMINFO: u8 = 0;
pub const BLOCK_PADDING: u8 = 1;
pub const BLOCK_SEEKTABLE: u8 = 3;
pub const BLOCK_VORBIS_COMMENT: u8 = 4;

/// The STREAMINFO block (§8.2): what a decoder needs before the first frame.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StreamInfo {
    pub min_block_size: u16,
    pub max_block_size: u16,
    /// Bytes; 0 = unknown.
    pub min_frame_size: u32,
    pub max_frame_size: u32,
    pub sample_rate: u32,
    pub channels: u8,
    pub bits_per_sample: u8,
    /// Inter-channel samples in the stream; 0 = unknown.
    pub total_samples: u64,
    /// MD5 of the unencoded audio (§8.2): every sample, interleaved, signed
    /// little-endian in the fewest whole bytes that hold the bit depth. All
    /// zeros = not computed.
    pub md5: [u8; 16],
}

impl StreamInfo {
    pub const LEN: usize = 34;

    pub fn parse(b: &[u8]) -> Result<Self, AudioError> {
        if b.len() < Self::LEN {
            return Err(AudioError::Decode(format!(
                "flac: STREAMINFO is {} bytes, needs {}",
                b.len(),
                Self::LEN
            )));
        }
        let be = |r: std::ops::Range<usize>| b[r].iter().fold(0u64, |v, &x| (v << 8) | u64::from(x));
        let packed = be(10..18);
        let info = Self {
            min_block_size: be(0..2) as u16,
            max_block_size: be(2..4) as u16,
            min_frame_size: be(4..7) as u32,
            max_frame_size: be(7..10) as u32,
            sample_rate: (packed >> 44) as u32,
            channels: ((packed >> 41) & 0x7) as u8 + 1,
            bits_per_sample: ((packed >> 36) & 0x1F) as u8 + 1,
            total_samples: packed & 0xF_FFFF_FFFF,
            md5: b[18..34].try_into().expect("16 bytes"),
        };
        if info.sample_rate == 0 {
            return Err(AudioError::Decode("flac: STREAMINFO sample rate is 0".into()));
        }
        if info.bits_per_sample < 4 {
            return Err(AudioError::Decode(format!(
                "flac: STREAMINFO bit depth {} is below the minimum of 4",
                info.bits_per_sample
            )));
        }
        Ok(info)
    }

    pub fn to_bytes(&self) -> [u8; Self::LEN] {
        let mut b = [0u8; Self::LEN];
        b[0..2].copy_from_slice(&self.min_block_size.to_be_bytes());
        b[2..4].copy_from_slice(&self.max_block_size.to_be_bytes());
        b[4..7].copy_from_slice(&self.min_frame_size.to_be_bytes()[1..]);
        b[7..10].copy_from_slice(&self.max_frame_size.to_be_bytes()[1..]);
        let packed = (u64::from(self.sample_rate) << 44)
            | (u64::from(self.channels - 1) << 41)
            | (u64::from(self.bits_per_sample - 1) << 36)
            | (self.total_samples & 0xF_FFFF_FFFF);
        b[10..18].copy_from_slice(&packed.to_be_bytes());
        b[18..34].copy_from_slice(&self.md5);
        b
    }
}

/// One metadata block's 4-byte header (§8.1): last-block flag, type, length.
pub fn block_header(last: bool, kind: u8, len: usize) -> [u8; 4] {
    let len = len as u32;
    [(u8::from(last) << 7) | kind, (len >> 16) as u8, (len >> 8) as u8, len as u8]
}

/// Walk a run of metadata blocks (as they follow the `fLaC` marker, or fill
/// an MP4 `dfLa` box) and return STREAMINFO, which must come first, plus the
/// byte length the blocks took (up to and including the last-flagged one).
pub fn parse_metadata_blocks(b: &[u8]) -> Result<(StreamInfo, usize), AudioError> {
    let mut at = 0usize;
    let mut info = None;
    loop {
        let Some(h) = b.get(at..at + 4) else {
            return Err(AudioError::Decode("flac: metadata ends inside a block header".into()));
        };
        let last = h[0] & 0x80 != 0;
        let kind = h[0] & 0x7F;
        let len = (usize::from(h[1]) << 16) | (usize::from(h[2]) << 8) | usize::from(h[3]);
        let body = b
            .get(at + 4..at + 4 + len)
            .ok_or_else(|| AudioError::Decode(format!("flac: metadata block of type {kind} runs past the data")))?;
        if info.is_none() {
            if kind != BLOCK_STREAMINFO {
                return Err(AudioError::Decode(format!(
                    "flac: the first metadata block is type {kind}, not STREAMINFO"
                )));
            }
            info = Some(StreamInfo::parse(body)?);
        }
        at += 4 + len;
        if last {
            break;
        }
    }
    Ok((info.expect("first block checked"), at))
}

/// STREAMINFO from whatever form a container hands over: the native stream
/// head (`fLaC` + blocks, as Matroska's `A_FLAC` CodecPrivate holds it), the
/// bare blocks (an MP4 `dfLa` body after its version and flags), the `dfLa`
/// body with them, or a bare 34-byte STREAMINFO.
pub fn stream_info_from_extra(extra: &[u8]) -> Result<StreamInfo, AudioError> {
    if let Some(rest) = extra.strip_prefix(b"fLaC") {
        return Ok(parse_metadata_blocks(rest)?.0);
    }
    if extra.len() == StreamInfo::LEN {
        return StreamInfo::parse(extra);
    }
    // A block header opens with type 0 (possibly flagged last) and a 34-byte
    // length; a `dfLa` FullBox body opens with version 0 and zero flags.
    let looks_like_blocks = |b: &[u8]| b.len() >= 4 && b[0] & 0x7F == 0 && b[1..4] == [0, 0, 34];
    if looks_like_blocks(extra) {
        return Ok(parse_metadata_blocks(extra)?.0);
    }
    if extra.len() > 4 && extra[..4] == [0, 0, 0, 0] && looks_like_blocks(&extra[4..]) {
        return Ok(parse_metadata_blocks(&extra[4..])?.0);
    }
    Err(AudioError::Decode(format!(
        "flac: no STREAMINFO in the {}-byte codec configuration",
        extra.len()
    )))
}

/// CRC-8 of a frame header (§9.1.8): polynomial x^8 + x^2 + x + 1, zero
/// initial value, no reflection.
pub fn crc8(data: &[u8]) -> u8 {
    static TABLE: std::sync::OnceLock<[u8; 256]> = std::sync::OnceLock::new();
    let table = TABLE.get_or_init(|| {
        let mut t = [0u8; 256];
        for (i, e) in t.iter_mut().enumerate() {
            let mut c = i as u8;
            for _ in 0..8 {
                c = if c & 0x80 != 0 { (c << 1) ^ 0x07 } else { c << 1 };
            }
            *e = c;
        }
        t
    });
    data.iter().fold(0u8, |c, &b| table[usize::from(c ^ b)])
}

/// CRC-16 of a whole frame (§9.3): polynomial x^16 + x^15 + x^2 + 1, zero
/// initial value, no reflection.
pub fn crc16(data: &[u8]) -> u16 {
    static TABLE: std::sync::OnceLock<[u16; 256]> = std::sync::OnceLock::new();
    let table = TABLE.get_or_init(|| {
        let mut t = [0u16; 256];
        for (i, e) in t.iter_mut().enumerate() {
            let mut c = (i as u16) << 8;
            for _ in 0..8 {
                c = if c & 0x8000 != 0 { (c << 1) ^ 0x8005 } else { c << 1 };
            }
            *e = c;
        }
        t
    });
    data.iter().fold(0u16, |c, &b| (c << 8) ^ table[usize::from((c >> 8) as u8 ^ b)])
}

/// The MD5 input for interleaved samples at `bits` bits (§8.2): each sample
/// signed little-endian in `ceil(bits / 8)` bytes.
pub fn md5_bytes(samples: &[i32], bits: u32, out: &mut Vec<u8>) {
    let width = bits.div_ceil(8) as usize;
    out.reserve(samples.len() * width);
    for &s in samples {
        out.extend_from_slice(&s.to_le_bytes()[..width]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streaminfo_round_trips() {
        let info = StreamInfo {
            min_block_size: 4096,
            max_block_size: 4096,
            min_frame_size: 14,
            max_frame_size: 12_345,
            sample_rate: 96_000,
            channels: 6,
            bits_per_sample: 24,
            total_samples: 0x9_1234_5678,
            md5: [7; 16],
        };
        assert_eq!(StreamInfo::parse(&info.to_bytes()).unwrap(), info);
        let mut native = b"fLaC".to_vec();
        native.extend_from_slice(&block_header(true, BLOCK_STREAMINFO, 34));
        native.extend_from_slice(&info.to_bytes());
        assert_eq!(stream_info_from_extra(&native).unwrap(), info);
        assert_eq!(stream_info_from_extra(&native[4..]).unwrap(), info);
        let mut dfla = vec![0, 0, 0, 0];
        dfla.extend_from_slice(&native[4..]);
        assert_eq!(stream_info_from_extra(&dfla).unwrap(), info);
    }

    #[test]
    fn crcs_match_the_published_check_values() {
        // The "123456789" check values of CRC-8 (poly 0x07) and of
        // CRC-16/UMTS (poly 0x8005, zero init, unreflected).
        assert_eq!(crc8(b"123456789"), 0xF4);
        assert_eq!(crc16(b"123456789"), 0xFEE8);
    }
}
