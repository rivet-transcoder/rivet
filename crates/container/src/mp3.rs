//! MPEG audio elementary streams (ISO/IEC 11172-3, 13818-3): frame headers,
//! and the bare `.mp3` file — reading one as an audio-only source, writing
//! one as an audio-only output, with the `Xing`/`Info` frame and LAME tag
//! that tell a player the stream's length, where to seek, and (gapless)
//! how many samples at each end are codec delay rather than audio.
//!
//! A `.mp3` file is the frames back to back, optionally behind an ID3v2 tag
//! and followed by an ID3v1 / APE tag. Nothing in the frames says how long
//! the stream is, so a player either scans it or trusts an `Info` frame (a
//! Layer III frame whose main data is the tag, which decodes as silence and
//! which players skip): frame and byte counts, a 100-entry seek table, and
//! LAME's extension carrying the encoder delay and end padding.

use anyhow::{Result, bail};

use crate::demux::AudioTrack;
use crate::edit::AudioEdit;

/// MPEG audio version, from the header's two version bits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Version {
    /// ISO/IEC 11172-3: 32 / 44.1 / 48 kHz.
    Mpeg1,
    /// ISO/IEC 13818-3 low sampling frequencies: 16 / 22.05 / 24 kHz.
    Mpeg2,
    /// The unofficial 8 / 11.025 / 12 kHz extension.
    Mpeg25,
}

/// One frame header (§2.4.1.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameHeader {
    pub version: Version,
    /// 1, 2 or 3.
    pub layer: u8,
    /// Whether a 16-bit CRC follows the header.
    pub protected: bool,
    pub bitrate_kbps: u32,
    pub sample_rate: u32,
    pub padding: bool,
    /// 0 stereo, 1 joint stereo, 2 dual channel, 3 mono.
    pub mode: u8,
}

const BITRATES_V1: [[u32; 15]; 3] = [
    [0, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448],
    [0, 32, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384],
    [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320],
];
const BITRATES_V2: [[u32; 15]; 3] = [
    [0, 32, 48, 56, 64, 80, 96, 112, 128, 144, 160, 176, 192, 224, 256],
    [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160],
    [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160],
];

impl FrameHeader {
    /// The header at the start of `b`, or `None` when `b` does not open with
    /// a valid one (a reserved version, layer, rate or bitrate index, or free
    /// format, whose frame length the header does not give).
    pub fn parse(b: &[u8]) -> Option<Self> {
        if b.len() < 4 || b[0] != 0xFF || b[1] & 0xE0 != 0xE0 {
            return None;
        }
        let version = match (b[1] >> 3) & 3 {
            3 => Version::Mpeg1,
            2 => Version::Mpeg2,
            0 => Version::Mpeg25,
            _ => return None,
        };
        let layer = match (b[1] >> 1) & 3 {
            3 => 1,
            2 => 2,
            1 => 3,
            _ => return None,
        };
        let index = usize::from(b[2] >> 4);
        if index == 0 || index == 15 {
            return None;
        }
        let table = if version == Version::Mpeg1 { &BITRATES_V1 } else { &BITRATES_V2 };
        let base = [44_100, 48_000, 32_000, 0][usize::from((b[2] >> 2) & 3)];
        if base == 0 {
            return None;
        }
        let sample_rate = match version {
            Version::Mpeg1 => base,
            Version::Mpeg2 => base / 2,
            Version::Mpeg25 => base / 4,
        };
        Some(Self {
            version,
            layer,
            protected: b[1] & 1 == 0,
            bitrate_kbps: table[usize::from(layer - 1)][index],
            sample_rate,
            padding: (b[2] >> 1) & 1 == 1,
            mode: b[3] >> 6,
        })
    }

    /// Bytes in the frame, header included (§2.4.3.1).
    pub fn frame_len(&self) -> usize {
        let (br, sr, pad) = (self.bitrate_kbps as usize * 1000, self.sample_rate as usize, usize::from(self.padding));
        match (self.layer, self.version) {
            (1, _) => (12 * br / sr + pad) * 4,
            (3, Version::Mpeg2 | Version::Mpeg25) => 72 * br / sr + pad,
            _ => 144 * br / sr + pad,
        }
    }

    /// PCM samples per channel the frame decodes to.
    pub fn samples(&self) -> u32 {
        match (self.layer, self.version) {
            (1, _) => 384,
            (3, Version::Mpeg2 | Version::Mpeg25) => 576,
            _ => 1152,
        }
    }

    pub fn channels(&self) -> u16 {
        if self.mode == 3 { 1 } else { 2 }
    }

    /// Layer III side information's length, which is where a Xing / Info tag
    /// starts after the header (and CRC).
    fn side_info_len(&self) -> usize {
        match (self.version, self.mode == 3) {
            (Version::Mpeg1, true) => 17,
            (Version::Mpeg1, false) => 32,
            (_, true) => 9,
            (_, false) => 17,
        }
    }

    /// Offset of a Xing / Info tag in a frame with this header.
    fn tag_offset(&self) -> usize {
        4 + if self.protected { 2 } else { 0 } + self.side_info_len()
    }

    /// The pipeline's codec label: `mp3` for Layer III, `mp2` for I and II
    /// (which the same decoder reads, and which no browser plays).
    pub fn codec(&self) -> &'static str {
        if self.layer == 3 { "mp3" } else { "mp2" }
    }

    /// The 4 header bytes.
    fn to_bytes(self, bitrate_index: u8, padding: bool) -> [u8; 4] {
        let v = match self.version {
            Version::Mpeg1 => 3,
            Version::Mpeg2 => 2,
            Version::Mpeg25 => 0,
        };
        let sr = match self.sample_rate {
            44_100 | 22_050 | 11_025 => 0,
            48_000 | 24_000 | 12_000 => 1,
            _ => 2,
        };
        [
            0xFF,
            0xE0 | (v << 3) | ((4 - self.layer) << 1) | 1,
            (bitrate_index << 4) | (sr << 2) | (u8::from(padding) << 1),
            self.mode << 6,
        ]
    }
}

/// Where each whole frame of `es` starts and how long it is, resynchronising
/// past bytes that are not a frame. A frame counts only when the one after
/// it (or the end of the stream) is where its header says: two headers
/// agreeing is how a random `0xFFE` in the data is told from a sync word.
pub fn frames(es: &[u8]) -> Vec<(usize, FrameHeader)> {
    let mut out = Vec::new();
    let mut at = 0usize;
    let mut locked: Option<FrameHeader> = None;
    while at + 4 <= es.len() {
        let Some(h) = FrameHeader::parse(&es[at..]) else {
            at += 1;
            continue;
        };
        let end = at + h.frame_len();
        let same_stream = |n: &FrameHeader| n.version == h.version && n.layer == h.layer && n.sample_rate == h.sample_rate;
        let confirmed = match locked {
            Some(l) => same_stream(&l),
            None => end == es.len() || FrameHeader::parse(&es[end.min(es.len())..]).is_some_and(|n| same_stream(&n)),
        };
        if !confirmed || end > es.len() {
            if locked.is_some() && end > es.len() {
                break; // a truncated last frame
            }
            at += 1;
            continue;
        }
        locked = Some(h);
        out.push((at, h));
        at = end;
    }
    out
}

/// A `Xing` / `Info` tag read from a frame (the first of a file).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct XingTag {
    /// Audio frames in the stream, the tag frame not counted.
    pub frames: Option<u32>,
    /// Bytes in the stream.
    pub bytes: Option<u32>,
    /// LAME's encoder delay and end padding, in samples, when the frame
    /// carries LAME's extension (the decoder's own 529 samples not counted).
    pub delay: Option<(u32, u32)>,
    /// The extension's 9-byte encoder name (`rivetmp3`, `LAME3.100`,
    /// `Lavc61.19`), when `delay` is read.
    pub encoder: Option<String>,
}

impl XingTag {
    /// The tag in `frame`, if it is a Xing / Info frame.
    pub fn parse(frame: &[u8]) -> Option<Self> {
        let h = FrameHeader::parse(frame)?;
        let mut at = h.tag_offset();
        let magic = frame.get(at..at + 4)?;
        if magic != b"Xing" && magic != b"Info" {
            return None;
        }
        let be32 = |at: usize| frame.get(at..at + 4).map(|b| u32::from_be_bytes(b.try_into().unwrap()));
        let flags = be32(at + 4)?;
        at += 8;
        let mut take = |bit: u32, len: usize| {
            let v = (flags & bit != 0).then(|| be32(at)).flatten();
            if flags & bit != 0 {
                at += len;
            }
            v
        };
        let frames = take(1, 4);
        let bytes = take(2, 4);
        take(4, 100);
        take(8, 4);
        // The LAME-style extension: a 9-byte encoder string, then 12 bytes to
        // the 24-bit delay/padding pair, and at its end a CRC of the frame up
        // to it. Taken at its word when the CRC checks out, whoever wrote it
        // (rivet's own encoder signs `rivetmp3`), or when the encoder is one
        // known to write the pair without a valid CRC (LAME and the `Lavf` /
        // `Lavc` muxers).
        let crc_ok = frame.get(at + 34..at + 36).is_some_and(|c| u16::from_be_bytes([c[0], c[1]]) == crc16_lame(&frame[..at + 34]));
        let ext = frame
            .get(at..at + 24)
            .filter(|ext| crc_ok || [b"LAME", b"Lavf", b"Lavc"].iter().any(|m| &ext[..4] == *m));
        let delay = ext.map(|ext| {
            let v = u32::from_be_bytes([0, ext[21], ext[22], ext[23]]);
            (v >> 12, v & 0xFFF)
        });
        let encoder = ext.map(|ext| String::from_utf8_lossy(&ext[..9]).trim_end().to_string());
        Some(Self { frames, bytes, delay, encoder })
    }
}

/// How much of an encoded stream is audio: the encoder's delay at the start
/// (the decoder's 529 samples not counted), and the samples that follow it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Gapless {
    pub encoder_delay: u32,
    pub samples: u64,
}

/// A `.mp3` file: an `Info` frame (a `Xing` frame when the bitrate varies)
/// and then `frames`, each one whole MPEG audio frame.
///
/// With `gapless` and an `encoder` (a name of up to 9 characters, e.g.
/// `rivetmp3`, or the one a source's tag gave), the tag frame also carries
/// the LAME-style extension: the encoder delay and end padding a gapless
/// player trims, and the CRC that marks the extension valid. Without them the tag says only how long
/// the stream is, which a player then plays whole — delay included.
pub fn write_file(frames: &[Vec<u8>], gapless: Option<Gapless>, encoder: Option<&str>) -> Result<Vec<u8>> {
    let Some(first) = frames.first() else {
        bail!("an MP3 file needs at least one frame");
    };
    let Some(h) = FrameHeader::parse(first) else {
        bail!("the first MP3 frame has no valid header");
    };
    let spf = u64::from(h.samples());
    let total_audio: usize = frames.iter().map(Vec::len).sum();
    let lame = match (gapless, encoder) {
        (Some(g), Some(enc)) => {
            let coded = spf * frames.len() as u64;
            let padding = coded.checked_sub(u64::from(g.encoder_delay) + g.samples);
            match padding {
                Some(p) if g.encoder_delay <= 0xFFF && p <= 0xFFF => Some((enc, g.encoder_delay, p as u32)),
                _ => bail!(
                    "gapless info does not fit the stream: {} frames of {spf} samples, delay {}, {} samples",
                    frames.len(),
                    g.encoder_delay,
                    g.samples
                ),
            }
        }
        _ => None,
    };
    // Xing: magic, flags, frames, bytes, TOC, quality. LAME's extension: 36.
    let needed = h.tag_offset() - if h.protected { 2 } else { 0 } + 120 + if lame.is_some() { 36 } else { 0 };
    let table = if h.version == Version::Mpeg1 { &BITRATES_V1 } else { &BITRATES_V2 };
    let ladder = &table[usize::from(h.layer - 1)];
    // The stream's own bitrate when the tag fits in one of its frames (a CBR
    // stream stays uniform), else the smallest one it fits in.
    let own = ladder.iter().position(|&k| k == h.bitrate_kbps).unwrap_or(1);
    let fits = |i: usize| FrameHeader { bitrate_kbps: ladder[i], padding: false, protected: false, ..h }.frame_len() >= needed;
    let Some(index) = std::iter::once(own).chain(1..15).find(|&i| fits(i)) else {
        bail!("no {:?} layer {} bitrate has room for a Xing tag", h.version, h.layer);
    };
    let tag_h = FrameHeader { bitrate_kbps: ladder[index], padding: false, protected: false, ..h };
    let tag_len = tag_h.frame_len();
    let mut tag = vec![0u8; tag_len];
    tag[..4].copy_from_slice(&tag_h.to_bytes(index as u8, false));
    let cbr = frames.iter().all(|f| f.get(2).map(|b| b >> 4) == first.get(2).map(|b| b >> 4));
    let mut at = tag_h.tag_offset();
    tag[at..at + 4].copy_from_slice(if cbr { b"Info" } else { b"Xing" });
    tag[at + 4..at + 8].copy_from_slice(&0x0Fu32.to_be_bytes());
    tag[at + 8..at + 12].copy_from_slice(&(frames.len() as u32).to_be_bytes());
    let file_bytes = (tag_len + total_audio) as u32;
    tag[at + 12..at + 16].copy_from_slice(&file_bytes.to_be_bytes());
    // The seek table: for each percent of the duration, where in the file
    // (in 256ths of its length) that frame starts.
    let mut offsets = Vec::with_capacity(frames.len());
    let mut pos = tag_len as u64;
    for f in frames {
        offsets.push(pos);
        pos += f.len() as u64;
    }
    for (i, slot) in tag[at + 16..at + 116].iter_mut().enumerate() {
        let frame = (i * frames.len() / 100).min(frames.len() - 1);
        *slot = (offsets[frame] * 256 / u64::from(file_bytes)).min(255) as u8;
    }
    // Quality: LAME's scale, 0 = unset.
    at += 120;
    if let Some((enc, delay, padding)) = lame {
        let mut name = [b' '; 9];
        for (d, s) in name.iter_mut().zip(enc.bytes()) {
            *d = s;
        }
        tag[at..at + 9].copy_from_slice(&name);
        tag[at + 9] = if cbr { 1 } else { 0 }; // tag revision 0, VBR method (1 = CBR)
        // Lowpass, peak, two ReplayGains, flags: unset.
        tag[at + 20] = h.bitrate_kbps.min(255) as u8;
        let pair = (delay << 12) | padding;
        tag[at + 21..at + 24].copy_from_slice(&pair.to_be_bytes()[1..]);
        // Misc, MP3 gain, preset: unset. Music length: the whole file.
        tag[at + 28..at + 32].copy_from_slice(&file_bytes.to_be_bytes());
        // Music CRC left 0; the tag CRC covers the frame up to itself.
        let crc = crc16_lame(&tag[..at + 34]);
        tag[at + 34..at + 36].copy_from_slice(&crc.to_be_bytes());
    }
    let mut out = Vec::with_capacity(tag_len + total_audio);
    out.extend_from_slice(&tag);
    for f in frames {
        out.extend_from_slice(f);
    }
    Ok(out)
}

/// The LAME tag's CRC: CRC-16 with the 0x8005 polynomial, bit-reflected
/// (CRC-16/ARC), from zero.
fn crc16_lame(bytes: &[u8]) -> u16 {
    let mut crc = 0u16;
    for &b in bytes {
        crc ^= u16::from(b);
        for _ in 0..8 {
            crc = if crc & 1 != 0 { (crc >> 1) ^ 0xA001 } else { crc >> 1 };
        }
    }
    crc
}

/// Whether `data` looks like a bare MPEG audio file: an ID3v2 tag, or two
/// frame headers that agree where the first says the second is.
pub fn sniff(data: &[u8]) -> bool {
    if data.len() >= 10 && &data[..3] == b"ID3" {
        return true;
    }
    FrameHeader::parse(data).is_some_and(|h| {
        let next = h.frame_len();
        data.len() > next + 4 && FrameHeader::parse(&data[next..]).is_some_and(|n| n.sample_rate == h.sample_rate && n.layer == h.layer)
    })
}

/// A bare `.mp3` / `.mp2` file as a track: its frames, one packet each, and
/// — when its `Info` frame carries LAME's delay and padding — the edit that
/// presents exactly the encoded audio. The track's `codec_private` then
/// holds the encoder name the tag gave, so a passthrough into another
/// `.mp3` can write the same gapless information under the same name.
pub fn read_file(data: &[u8]) -> Result<(AudioTrack, Option<AudioEdit>)> {
    let mut start = 0usize;
    // ID3v2 (id3.org, v2.4 §3.1): "ID3", version, flags, a 28-bit syncsafe
    // size that excludes the 10-byte header and a 10-byte footer if flagged.
    while data.len() >= start + 10 && &data[start..start + 3] == b"ID3" {
        let s = &data[start + 6..start + 10];
        let size = s.iter().fold(0usize, |acc, &b| (acc << 7) | usize::from(b & 0x7F));
        let footer = if data[start + 5] & 0x10 != 0 { 10 } else { 0 };
        start += 10 + size + footer;
    }
    if start >= data.len() {
        bail!("MP3: nothing after the ID3 tag");
    }
    let es = &data[start..];
    let found = frames(es);
    let Some(&(_, first)) = found.first() else {
        bail!("MP3: no MPEG audio frames found");
    };
    let mut packets: Vec<&[u8]> = found.iter().map(|&(at, h)| &es[at..at + h.frame_len()]).collect();
    let tag = XingTag::parse(packets[0]);
    if tag.is_some() || packets[0].get(36..40) == Some(b"VBRI") {
        packets.remove(0);
    }
    if packets.is_empty() {
        bail!("MP3: the stream holds a tag frame and no audio");
    }
    let spf = first.samples();
    let edit = tag.as_ref().and_then(|t| t.delay).and_then(|(delay, padding)| {
        let coded = u64::from(spf) * packets.len() as u64;
        let skip = u64::from(delay) + 529;
        let presented = coded.checked_sub(u64::from(delay) + u64::from(padding))?;
        Some(AudioEdit { delay: 0, media_start: skip, media_end: Some(skip + presented) })
    });
    let track = AudioTrack {
        codec: first.codec().into(),
        samples: packets.iter().map(|p| p.to_vec()).collect(),
        sample_rate: first.sample_rate,
        channels: first.channels(),
        asc: Vec::new(),
        codec_private: tag
            .and_then(|t| t.delay.and(t.encoder))
            .filter(|_| edit.is_some())
            .map(String::into_bytes)
            .unwrap_or_default(),
        timescale: first.sample_rate,
        durations: vec![spf; packets.len()],
    };
    Ok((track, edit))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A syntactically valid MPEG-1 Layer III frame: its header, the rest zero.
    fn frame(kbps_index: u8, rate_index: u8, mono: bool, padding: bool) -> Vec<u8> {
        let h = [0xFF, 0xFB, (kbps_index << 4) | (rate_index << 2) | (u8::from(padding) << 1), if mono { 0xC0 } else { 0x40 }];
        let len = FrameHeader::parse(&h).unwrap().frame_len();
        let mut f = vec![0u8; len];
        f[..4].copy_from_slice(&h);
        f
    }

    #[test]
    fn headers_parse_every_version_and_layer() {
        // MPEG-1 Layer III, 128 kbps, 44.1 kHz, joint stereo, no CRC.
        let h = FrameHeader::parse(&[0xFF, 0xFB, 0x90, 0x44]).unwrap();
        assert_eq!((h.version, h.layer, h.bitrate_kbps, h.sample_rate, h.channels()), (Version::Mpeg1, 3, 128, 44_100, 2));
        assert_eq!((h.frame_len(), h.samples(), h.protected, h.codec()), (417, 1152, false, "mp3"));
        // Padded: one byte more.
        assert_eq!(FrameHeader::parse(&[0xFF, 0xFB, 0x92, 0x44]).unwrap().frame_len(), 418);
        // MPEG-2 Layer III, 64 kbps, 22.05 kHz, mono: 576 samples, 72·br/sr.
        let h = FrameHeader::parse(&[0xFF, 0xF3, 0x80, 0xC0]).unwrap();
        assert_eq!((h.version, h.sample_rate, h.channels(), h.samples(), h.frame_len()), (Version::Mpeg2, 22_050, 1, 576, 208));
        // MPEG-2.5 Layer III at 8 kHz.
        assert_eq!(FrameHeader::parse(&[0xFF, 0xE3, 0x88, 0xC0]).unwrap().sample_rate, 8_000);
        // MPEG-1 Layer II, 192 kbps, 48 kHz: 576 bytes, codec mp2.
        let h = FrameHeader::parse(&[0xFF, 0xFD, 0xA4, 0x00]).unwrap();
        assert_eq!((h.layer, h.bitrate_kbps, h.frame_len(), h.samples(), h.codec()), (2, 192, 576, 1152, "mp2"));
        // Layer I frames are 4-byte slots.
        assert_eq!(FrameHeader::parse(&[0xFF, 0xFF, 0x10, 0x00]).unwrap().frame_len(), 32 * 12 * 1000 / 44_100 * 4);
        // Reserved version / layer / rate, free format, bad index.
        for bad in [[0xFF, 0xEB, 0x90, 0], [0xFF, 0xF9, 0x90, 0], [0xFF, 0xFB, 0x9C, 0], [0xFF, 0xFB, 0x00, 0], [0xFF, 0xFB, 0xF0, 0], [0xFE, 0xFB, 0x90, 0]] {
            assert_eq!(FrameHeader::parse(&bad), None, "{bad:02X?}");
        }
    }

    #[test]
    fn the_frame_walk_resyncs_and_drops_a_truncated_tail() {
        let mut es = vec![0x00, 0xFF, 0xFB, 0x12]; // junk, including a false sync
        let f = frame(9, 0, false, false);
        for _ in 0..3 {
            es.extend_from_slice(&f);
        }
        es.extend_from_slice(&f[..100]);
        let got = frames(&es);
        assert_eq!(got.iter().map(|&(at, _)| at).collect::<Vec<_>>(), vec![4, 4 + 417, 4 + 2 * 417]);
    }

    /// The written file reads back: an `Info` frame of the stream's bitrate,
    /// counts, a monotone seek table, and LAME's delay / padding under the
    /// tag CRC the LAME Info Tag specification defines.
    #[test]
    fn an_info_frame_with_the_lame_extension_round_trips() {
        let frames: Vec<Vec<u8>> = (0..40).map(|i| frame(9, 0, false, i % 3 == 0)).collect();
        let audio: usize = frames.iter().map(Vec::len).sum();
        // 40 frames = 46080 samples: 576 of delay, 44100 of audio, 1404 padding.
        let g = Gapless { encoder_delay: 576, samples: 44_100 };
        let file = write_file(&frames, Some(g), Some("LAME3.100")).unwrap();
        let tag_frame = &file[..417];
        assert_eq!(&file[417..], &frames.concat()[..], "the audio follows verbatim");
        assert_eq!(&tag_frame[36..40], b"Info", "CBR, stereo side info (32 bytes) before it");
        let tag = XingTag::parse(tag_frame).unwrap();
        assert_eq!(tag.frames, Some(40));
        assert_eq!(tag.bytes, Some((417 + audio) as u32));
        assert_eq!(tag.delay, Some((576, 40 * 1152 - 576 - 44_100)));
        assert_eq!(tag.encoder.as_deref(), Some("LAME3.100"));
        let toc = &tag_frame[52..152];
        assert!(toc.windows(2).all(|w| w[0] <= w[1]), "the seek table only moves forward");
        assert_eq!(toc[0], (417 * 256 / (417 + audio)) as u8, "0% is the first audio frame");
        assert_eq!(&tag_frame[156..165], b"LAME3.100");
        // The tag CRC covers the frame's first 190 bytes (stereo MPEG-1).
        assert_eq!(u16::from_be_bytes([tag_frame[190], tag_frame[191]]), crc16_lame(&tag_frame[..190]));
        // And the reader turns it into the edit that presents the audio.
        let (track, edit) = read_file(&file).unwrap();
        assert_eq!((track.codec.as_str(), track.samples.len(), track.sample_rate, track.channels), ("mp3", 40, 44_100, 2));
        assert_eq!(track.codec_private, b"LAME3.100", "the encoder name, for a passthrough to write again");
        assert_eq!(edit, Some(AudioEdit { delay: 0, media_start: 576 + 529, media_end: Some(576 + 529 + 44_100) }));
    }

    #[test]
    fn a_low_bitrate_stream_gets_a_tag_frame_big_enough() {
        // 32 kbps mono at 48 kHz: 96-byte frames, too small for the tag.
        let frames: Vec<Vec<u8>> = (0..10).map(|_| frame(1, 1, true, false)).collect();
        let file = write_file(&frames, Some(Gapless { encoder_delay: 576, samples: 10_000 }), Some("LAME3.100")).unwrap();
        let h = FrameHeader::parse(&file).unwrap();
        assert!(h.frame_len() >= 21 + 156 && h.bitrate_kbps > 32, "{h:?}");
        assert_eq!(&file[21..25], b"Info", "mono side info is 17 bytes");
        assert_eq!(XingTag::parse(&file).unwrap().frames, Some(10));
    }

    #[test]
    fn varying_bitrates_are_tagged_xing_and_plain_passthrough_has_no_lame_extension() {
        let frames = vec![frame(9, 0, false, false), frame(11, 0, false, false), frame(9, 0, false, false)];
        let file = write_file(&frames, None, None).unwrap();
        assert_eq!(&file[36..40], b"Xing");
        let tag = XingTag::parse(&file).unwrap();
        assert_eq!((tag.frames, tag.delay), (Some(3), None));
        let (_, edit) = read_file(&file).unwrap();
        assert_eq!(edit, None, "no delay stated, nothing hidden");
    }

    #[test]
    fn gapless_info_that_does_not_fit_the_frames_is_refused() {
        let frames = vec![frame(9, 0, false, false); 2];
        assert!(write_file(&frames, Some(Gapless { encoder_delay: 576, samples: 5000 }), Some("LAME3.100")).is_err());
    }

    #[test]
    fn id3v2_is_skipped_and_the_file_sniffs() {
        let mut data = b"ID3\x04\x00\x00\x00\x00\x00\x05".to_vec();
        data.extend_from_slice(b"hello");
        let f = frame(9, 0, false, false);
        for _ in 0..3 {
            data.extend_from_slice(&f);
        }
        assert!(sniff(&data));
        assert!(sniff(&data[15..]), "bare frames sniff too");
        assert!(!sniff(&data[16..]), "mid-frame bytes do not");
        let (track, edit) = read_file(&data).unwrap();
        assert_eq!((track.samples.len(), track.durations[0], edit), (3, 1152, None));
    }

    /// Whether an input has video decides audio-only, not whether the video
    /// demuxer took it: a file with a video track it refused is not read as
    /// audio alone.
    #[test]
    fn audio_only_means_no_video_track() {
        let fixture = |p: &str| std::fs::read(format!("{}/{p}", env!("CARGO_MANIFEST_DIR"))).unwrap();
        let mkv = fixture("tests/fixtures/colour/h264_601.mkv");
        let mp4 = fixture("tests/fixtures/colour/h264_601.mp4");
        let mka = fixture("../rivet/tests/data/audio/tones_51_ac3.mka");
        let m4a = fixture("../rivet/tests/data/audio/tones_51_aac.m4a");
        assert!(crate::demux::mkv::has_video_track(&mkv).unwrap());
        assert!(crate::demux::mp4::has_video_track(&mp4).unwrap());
        assert!(!crate::demux::mkv::has_video_track(&mka).unwrap());
        assert!(!crate::demux::mp4::has_video_track(&m4a).unwrap());
        for audio_only in [mka, m4a] {
            let src = crate::streaming::demux_audio(bytes::Bytes::from(audio_only)).unwrap().expect("the audio");
            assert!(!src.has_video);
            assert_eq!(src.track.channels, 6);
        }
    }

    /// A bare MP3 is an audio source with no video, through the same entry
    /// point a video file's audio is read by.
    #[test]
    fn a_bare_mp3_is_an_audio_only_source() {
        let frames: Vec<Vec<u8>> = (0..10).map(|_| frame(9, 1, false, false)).collect();
        let file = write_file(&frames, Some(Gapless { encoder_delay: 576, samples: 10_000 }), Some("LAME3.100")).unwrap();
        let src = crate::streaming::demux_audio(bytes::Bytes::from(file)).unwrap().expect("audio");
        assert!(!src.has_video);
        assert_eq!((src.track.codec.as_str(), src.track.sample_rate, src.track.samples.len()), ("mp3", 48_000, 10));
        assert_eq!(src.edit.map(|e| (e.media_start, e.media_end)), Some((1105, Some(11_105))));
    }
}
