//! FLAC and ALAC tracks: MP4 sample entries, Matroska codec private data,
//! and the native FLAC stream.
//!
//! The canonical `AudioTrack::codec_private` for each, whichever container
//! it came from:
//! - `flac`: the metadata blocks, STREAMINFO first and the last one flagged
//!   — the body of an MP4 `dfLa` box after its version and flags, a
//!   Matroska `A_FLAC` CodecPrivate after its `fLaC` marker, a native
//!   stream's header after its marker.
//! - `alac`: the 24-byte `ALACSpecificConfig` (magic cookie).
//!
//! Every packet is one frame, and its duration is the frame's own sample
//! count, read from the frame (a FLAC frame header's block size, an ALAC
//! element header's partial-frame count) rather than rounded out of a
//! container clock.

use std::io::Cursor;

use anyhow::{Context, Result, bail};
use mp4::Mp4Reader;

use super::super::AudioTrack;

/// The fields of a FLAC STREAMINFO a demuxer needs: (sample rate,
/// channels, bits per sample, block size).
pub(crate) fn flac_stream_params(blocks: &[u8]) -> Option<(u32, u16, u8, u16)> {
    // Block header (4 bytes), then STREAMINFO; the packed rate / channels /
    // depth sit at bytes 10..14 of it.
    if blocks.len() < 4 + 34 || blocks[0] & 0x7F != 0 {
        return None;
    }
    let si = &blocks[4..];
    let rate = (u32::from(si[10]) << 12) | (u32::from(si[11]) << 4) | (u32::from(si[12]) >> 4);
    let channels = u16::from((si[12] >> 1) & 0x7) + 1;
    let bits = (((si[12] & 1) << 4) | (si[13] >> 4)) + 1;
    let block = u16::from_be_bytes([si[2], si[3]]);
    (rate > 0).then_some((rate, channels, bits, block))
}

/// The ALAC cookie's (sample rate, channels, frame length).
pub(crate) fn alac_stream_params(cookie: &[u8]) -> Option<(u32, u16, u32)> {
    if cookie.len() < 24 {
        return None;
    }
    let frame_length = u32::from_be_bytes(cookie[0..4].try_into().ok()?);
    let rate = u32::from_be_bytes(cookie[20..24].try_into().ok()?);
    Some((rate, u16::from(cookie[9]), frame_length))
}

/// The 24-byte cookie out of the wrappings it travels in: bare, after a
/// FullBox version/flags, or inside a whole `alac` atom.
pub(crate) fn normalize_alac_cookie(raw: &[u8]) -> Option<Vec<u8>> {
    match raw.len() {
        24 => Some(raw.to_vec()),
        28 if raw[..4] == [0, 0, 0, 0] => Some(raw[4..].to_vec()),
        // The cookie followed by its optional channel layout info (a `chan`
        // atom), as Apple's encoder writes it for more than two channels.
        48 if raw[28..32] == *b"chan" => Some(raw[..24].to_vec()),
        _ => {
            let at = raw.windows(4).position(|w| w == b"alac")?;
            raw.get(at + 8..at + 32).map(<[u8]>::to_vec)
        }
    }
}

/// The STREAMINFO block out of a Matroska `A_FLAC` CodecPrivate (`fLaC` +
/// blocks) or a `dfLa` body (version/flags + blocks), flagged last.
///
/// The other blocks go: Vorbis comments, pictures and application data are
/// the source's tags, not what a decoder needs, and a copy of the stream
/// must not carry them into an output.
pub(crate) fn normalize_flac_blocks(raw: &[u8]) -> Option<Vec<u8>> {
    let blocks = raw.strip_prefix(b"fLaC").or_else(|| raw.strip_prefix(&[0, 0, 0, 0])).unwrap_or(raw);
    flac_stream_params(blocks)?;
    let mut streaminfo = blocks.get(..38)?.to_vec();
    streaminfo[..4].copy_from_slice(&[0x80, 0, 0, 34]);
    Some(streaminfo)
}

/// Samples in a FLAC frame, from its header (RFC 9639 §9.1.1).
pub(crate) fn flac_frame_samples(frame: &[u8]) -> Option<u32> {
    if frame.len() < 5 || frame[0] != 0xFF || frame[1] & 0xFE != 0xF8 {
        return None;
    }
    let code = frame[2] >> 4;
    // The coded number follows the fixed 4 bytes; an 8- or 16-bit block
    // size follows it.
    let lead = frame[4].leading_ones() as usize;
    let number_len = match lead {
        0 => 1,
        2..=7 => lead,
        _ => return None,
    };
    let at = 4 + number_len;
    Some(match code {
        0 => return None,
        1 => 192,
        2..=5 => 576 << (code - 2),
        6 => u32::from(*frame.get(at)?) + 1,
        7 => u32::from(u16::from_be_bytes([*frame.get(at)?, *frame.get(at + 1)?])) + 1,
        _ => 256 << (code - 8),
    })
}

/// Samples in an ALAC frame: the partial-frame count of its first element
/// when it has one, else the cookie's frame length.
pub(crate) fn alac_frame_samples(frame: &[u8], frame_length: u32) -> u32 {
    // tag(3) instance(4) unused(12) partial(1) shift(2) escape(1) [count(32)]
    let bit = |i: usize| frame.get(i / 8).map(|b| (b >> (7 - i % 8)) & 1);
    let tag = frame.first().map_or(7, |b| b >> 5);
    if tag > 1 && tag != 3 || bit(19) != Some(1) {
        return frame_length;
    }
    let mut n = 0u32;
    for i in 23..55 {
        match bit(i) {
            Some(b) => n = (n << 1) | u32::from(b),
            None => return frame_length,
        }
    }
    n
}

/// A FLAC or ALAC track in an MP4, when the audio track's sample entry is
/// `fLaC` (with `dfLa`) or `alac` (with its `alac` cookie box).
pub(crate) fn extract_mp4_lossless(data: &[u8]) -> Option<AudioTrack> {
    let (codec, private, rate, channels) =
        if let Some(dfla) = super::ac3::extract_mp4_audio_config_body(data, b"fLaC", b"dfLa") {
            let blocks = normalize_flac_blocks(&dfla).or_else(|| {
                tracing::warn!("MP4 fLaC: dfLa holds no STREAMINFO; dropping audio");
                None
            })?;
            let (rate, channels, _, _) = flac_stream_params(&blocks)?;
            ("flac", blocks, rate, channels)
        } else if let Some(raw) = super::ac3::extract_mp4_audio_config_body(data, b"alac", b"alac") {
            let cookie = normalize_alac_cookie(&raw).or_else(|| {
                tracing::warn!(len = raw.len(), "MP4 alac: magic cookie is not 24 bytes; dropping audio");
                None
            })?;
            let (rate, channels, _) = alac_stream_params(&cookie)?;
            ("alac", cookie, rate, channels)
        } else {
            return None;
        };
    let size = data.len() as u64;
    let mut reader = Mp4Reader::read_header(Cursor::new(data), size).ok()?;
    let (track_id, timescale, sample_count) = reader
        .tracks()
        .values()
        .find(|t| t.track_type().ok() == Some(mp4::TrackType::Audio))
        .map(|t| (t.track_id(), t.timescale(), t.sample_count()))?;
    let mut samples = Vec::with_capacity(sample_count as usize);
    let mut durations = Vec::with_capacity(sample_count as usize);
    for idx in 1..=sample_count {
        match reader.read_sample(track_id, idx) {
            Ok(Some(s)) => {
                durations.push(s.duration);
                samples.push(s.bytes.to_vec());
            }
            Ok(None) => break,
            Err(e) => {
                tracing::warn!(codec, idx, error = %e, "MP4 lossless audio: read_sample failed; keeping what was read");
                break;
            }
        }
    }
    if samples.is_empty() {
        return None;
    }
    Some(AudioTrack {
        codec: codec.into(),
        samples,
        sample_rate: rate,
        channels,
        asc: Vec::new(),
        codec_private: private,
        timescale,
        durations,
    })
}

/// Durations (in samples) for packets whose clock came from a container
/// that rounds: each frame's own sample count.
pub(crate) fn frame_durations(codec: &str, codec_private: &[u8], samples: &[Vec<u8>]) -> Option<Vec<u32>> {
    match codec {
        "flac" => samples.iter().map(|f| flac_frame_samples(f)).collect(),
        "alac" => {
            let (_, _, frame_length) = alac_stream_params(codec_private)?;
            Some(samples.iter().map(|f| alac_frame_samples(f, frame_length)).collect())
        }
        _ => None,
    }
}

/// A native FLAC stream (`fLaC` + metadata blocks + frames), cut into one
/// packet per frame. An ID3v2 tag in front of the marker is skipped.
pub fn read_native_flac(data: &[u8]) -> Result<AudioTrack> {
    let start = native_flac_offset(data).context("not a native FLAC stream (no fLaC marker)")?;
    let body = &data[start + 4..];
    // Walk the metadata blocks to the first frame.
    let mut at = 0usize;
    loop {
        let h = body.get(at..at + 4).context("FLAC metadata ends inside a block header")?;
        let len = (usize::from(h[1]) << 16) | (usize::from(h[2]) << 8) | usize::from(h[3]);
        at += 4 + len;
        if at > body.len() {
            bail!("FLAC metadata block runs past the end of the file");
        }
        if h[0] & 0x80 != 0 {
            break;
        }
    }
    // Keep STREAMINFO alone as the configuration: the seek table, tags and
    // padding describe this file, not the stream a muxer writes.
    let mut blocks = body[..4 + 34].to_vec();
    blocks[0] = 0x80;
    let (rate, channels, _, _) = flac_stream_params(&blocks).context("FLAC stream has no STREAMINFO")?;
    let frames = split_flac_frames(&body[at..]);
    if frames.is_empty() {
        bail!("FLAC stream holds no frames");
    }
    let mut samples = Vec::with_capacity(frames.len());
    let mut durations = Vec::with_capacity(frames.len());
    for f in frames {
        durations.push(flac_frame_samples(f).context("FLAC frame header")?);
        samples.push(f.to_vec());
    }
    Ok(AudioTrack {
        codec: "flac".into(),
        samples,
        sample_rate: rate,
        channels,
        asc: Vec::new(),
        codec_private: blocks,
        timescale: rate,
        durations,
    })
}

/// Where the `fLaC` marker is: at the start, or after an ID3v2 tag.
pub(crate) fn native_flac_offset(data: &[u8]) -> Option<usize> {
    if data.starts_with(b"fLaC") {
        return Some(0);
    }
    if data.len() >= 10 && data.starts_with(b"ID3") {
        // Syncsafe size, plus the 10-byte header and a 10-byte footer when
        // the footer flag is set.
        let size = data[6..10].iter().fold(0usize, |v, &b| (v << 7) | usize::from(b & 0x7F));
        let at = 10 + size + if data[5] & 0x10 != 0 { 10 } else { 0 };
        if data.get(at..at + 4) == Some(b"fLaC") {
            return Some(at);
        }
    }
    None
}

/// Cut a run of FLAC frames at their sync codes. A candidate boundary is a
/// sync code whose header passes its CRC-8, and it is taken when the bytes
/// before it pass the frame CRC-16 — a sync pattern inside frame data does
/// not.
fn split_flac_frames(data: &[u8]) -> Vec<&[u8]> {
    let is_header = |i: usize| -> bool {
        if i + 6 > data.len() || data[i] != 0xFF || data[i + 1] & 0xFE != 0xF8 {
            return false;
        }
        header_len(&data[i..]).is_some_and(|n| crc8(&data[i..i + n - 1]) == data[i + n - 1])
    };
    let mut frames = Vec::new();
    let mut start = match (0..data.len()).find(|&i| is_header(i)) {
        Some(s) => s,
        None => return frames,
    };
    let mut i = start + 2;
    while i < data.len() {
        if is_header(i) && crc16(&data[start..i]) == 0 {
            frames.push(&data[start..i]);
            start = i;
            i += 2;
            continue;
        }
        i += 1;
    }
    // The last frame runs to the end, less any trailing bytes (an ID3v1 tag)
    // that fail its CRC: shrink to the longest prefix that passes.
    let mut crc = 0u16;
    let mut end = None;
    for (e, &b) in data.iter().enumerate().skip(start) {
        crc = crc16_step(crc, b);
        if crc == 0 && e >= start + 2 {
            end = Some(e + 1);
        }
    }
    if let Some(end) = end {
        frames.push(&data[start..end]);
    }
    frames
}

/// Bytes a FLAC frame header takes, CRC-8 included.
fn header_len(h: &[u8]) -> Option<usize> {
    let bs = h[2] >> 4;
    let sr = h[2] & 0xF;
    if bs == 0 || sr == 15 || h[3] & 1 != 0 || (h[3] >> 4) > 10 || (h[3] >> 1) & 7 == 3 {
        return None;
    }
    let lead = h[4].leading_ones() as usize;
    let mut n = 4 + match lead {
        0 => 1,
        2..=7 => lead,
        _ => return None,
    };
    n += match bs {
        6 => 1,
        7 => 2,
        _ => 0,
    };
    n += match sr {
        12 => 1,
        13 | 14 => 2,
        _ => 0,
    };
    (n < h.len()).then_some(n + 1)
}

fn crc8(data: &[u8]) -> u8 {
    data.iter().fold(0u8, |mut c, &b| {
        c ^= b;
        for _ in 0..8 {
            c = if c & 0x80 != 0 { (c << 1) ^ 0x07 } else { c << 1 };
        }
        c
    })
}

fn crc16(data: &[u8]) -> u16 {
    data.iter().fold(0u16, |c, &b| crc16_step(c, b))
}

fn crc16_step(c: u16, b: u8) -> u16 {
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
    (c << 8) ^ table[usize::from((c >> 8) as u8 ^ b)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_sample_counts() {
        // 4096-sample block (code 12), frame number 0.
        assert_eq!(flac_frame_samples(&[0xFF, 0xF8, 0xC9, 0x18, 0x00, 0x00]), Some(4096));
        // 16-bit explicit size (code 7): 1234 + 1, after a 1-byte number.
        assert_eq!(flac_frame_samples(&[0xFF, 0xF8, 0x79, 0x18, 0x05, 0x04, 0xD2, 0]), Some(1235));
        // A two-byte coded number moves the explicit size along.
        assert_eq!(flac_frame_samples(&[0xFF, 0xF9, 0x69, 0x18, 0xC2, 0x80, 0x09, 0]), Some(10));
        // ALAC: a partial frame of 1000 samples.
        let mut alac = vec![0u8; 8];
        // bit 19 set; count 1000 in bits 23..55.
        let mut set = |i: usize| alac[i / 8] |= 1 << (7 - i % 8);
        set(19);
        for b in 0..32 {
            if (1000u32 >> (31 - b)) & 1 == 1 {
                set(23 + b);
            }
        }
        assert_eq!(alac_frame_samples(&alac, 4096), 1000);
        assert_eq!(alac_frame_samples(&[0x20, 0, 0, 0], 4096), 4096);
    }

    #[test]
    fn cookies_and_blocks_normalize() {
        let cookie: Vec<u8> = (0..24).collect();
        assert_eq!(normalize_alac_cookie(&cookie).unwrap(), cookie);
        let mut boxed = vec![0, 0, 0, 36];
        boxed.extend_from_slice(b"alac");
        boxed.extend_from_slice(&[0; 4]);
        boxed.extend_from_slice(&cookie);
        assert_eq!(normalize_alac_cookie(&boxed).unwrap(), cookie);
        let mut blocks = vec![0x80, 0, 0, 34];
        let mut si = [0u8; 34];
        // 44100 Hz, 2 channels, 16 bits.
        si[10] = 0x0A;
        si[11] = 0xC4;
        si[12] = 0x42;
        si[13] = 0xF0;
        blocks.extend_from_slice(&si);
        assert_eq!(flac_stream_params(&blocks), Some((44_100, 2, 16, 0)));
        let mut mkv = b"fLaC".to_vec();
        mkv.extend_from_slice(&blocks);
        assert_eq!(normalize_flac_blocks(&mkv).unwrap(), blocks);
    }

    /// A source's tags, cover art and application data are not the stream's:
    /// only STREAMINFO survives, flagged last, from a Matroska CodecPrivate
    /// or a `dfLa` body alike.
    #[test]
    fn only_streaminfo_is_kept_of_a_tagged_stream() {
        let mut si = [0u8; 34];
        si[10] = 0x0A;
        si[11] = 0xC4;
        si[12] = 0x42;
        si[13] = 0xF0;
        let block = |kind: u8, last: bool, body: &[u8]| {
            let mut b = vec![kind | if last { 0x80 } else { 0 }];
            b.extend_from_slice(&(body.len() as u32).to_be_bytes()[1..]);
            b.extend_from_slice(body);
            b
        };
        let mut comment = Vec::new();
        for field in [&b"Lavf61.7.100"[..], b"TITLE=Secret", b"DATE=2024"] {
            comment.extend_from_slice(&(field.len() as u32).to_le_bytes());
            comment.extend_from_slice(field);
            if field.starts_with(b"Lavf") {
                comment.extend_from_slice(&2u32.to_le_bytes());
            }
        }
        let tagged = [block(0, false, &si), block(4, false, &comment), block(6, false, b"picture"), block(2, true, b"appl")].concat();
        let expect = block(0, true, &si);
        let mut mkv = b"fLaC".to_vec();
        mkv.extend_from_slice(&tagged);
        let mut dfla = vec![0, 0, 0, 0];
        dfla.extend_from_slice(&tagged);
        assert_eq!(normalize_flac_blocks(&mkv).unwrap(), expect);
        assert_eq!(normalize_flac_blocks(&dfla).unwrap(), expect);
    }
}
