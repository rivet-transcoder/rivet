//! Ogg (RFC 3533) audio files: Ogg Opus (RFC 7845) and Ogg Vorbis (Vorbis I
//! §A), read and written. Audio only: a file's first audio logical stream is
//! the track, and the others are skipped (an Ogg with Theora video is refused
//! by name rather than read as audio alone).
//!
//! The pages are the Vorbis crate's (`vorbis::ogg`, this workspace's own
//! RFC 3533 reader and writer); what is here is the codec mapping on top of
//! them:
//!
//! - **Opus**: an `OpusHead` page, an `OpusTags` page, then the packets.
//!   Granule positions count 48 kHz samples from the start of the decoded
//!   stream, pre-skip included; the last page's position ends the stream
//!   exactly (RFC 7845 §4.4). Read back, the pre-skip and that end are the
//!   track's edit, the same one an MP4 edit list states.
//! - **Vorbis**: the identification header alone on the first page, the
//!   comment and setup headers ending the second, then the packets. Granule
//!   positions count samples of the decoded stream; the last one ends it.
//!
//! [`write_audio`] takes what the job layer's audio pipeline produces (the
//! packets with their durations, and the presentation edit); [`read_audio`]
//! gives back an [`AudioTrack`] with exact per-packet durations and that
//! edit.

use anyhow::{Context, Result, bail};
use vorbis::ogg::{PacketReader, PacketWriter};

use crate::AudioInfo;
use crate::demux::AudioTrack;
use crate::edit::{AudioEdit, TrackEdit};

/// The serial number a written file's one logical stream carries.
const SERIAL: u32 = 0x7269_7674; // "rivt"

/// The vendor string an `OpusTags` / Vorbis comment header names.
pub const VENDOR: &str = "rivet";

/// Whether `data` opens with an Ogg page.
pub fn sniff(data: &[u8]) -> bool {
    data.len() >= 27 && &data[..4] == b"OggS" && data[4] == 0
}

/// An Ogg file holding the audio track `info` describes: `packets` with their
/// durations (ticks of `info.timescale`), presented as `edit` says (its
/// `media_time` is Opus's pre-skip; its `duration`, when given, ends the
/// stream before the last packet's end). Opus and Vorbis.
pub fn write_audio(info: &AudioInfo, packets: &[(Vec<u8>, u32)], edit: TrackEdit) -> Result<Vec<u8>> {
    if packets.is_empty() {
        bail!("an Ogg file needs at least one audio packet");
    }
    if edit.delay != 0 {
        bail!("an Ogg file cannot start its audio late (an edit delay of {} ticks)", edit.delay);
    }
    let mut w = PacketWriter::new(Vec::new(), SERIAL);
    let total: u64 = packets.iter().map(|(_, d)| u64::from(*d)).sum();
    let (start, end) = match info.codec.to_ascii_lowercase().as_str() {
        "opus" => {
            if info.codec_private.len() < 11 || info.timescale != 48_000 {
                bail!("Opus in Ogg needs its OpusHead and 48 kHz timing");
            }
            let mut head = b"OpusHead".to_vec();
            head.extend_from_slice(&info.codec_private);
            head[8] = 1; // the OpusHead version (a `dOps` body says 0)
            let pre_skip = u64::from(u16::from_le_bytes([info.codec_private[2], info.codec_private[3]]));
            if edit.media_time != 0 && edit.media_time != pre_skip {
                bail!("an Opus edit starting at {} is not the stream's pre-skip ({pre_skip})", edit.media_time);
            }
            w.write_packet(&head, 0, true, false)?;
            w.write_packet(&opus_tags(), 0, true, false)?;
            (0, edit.duration.map(|d| pre_skip + d))
        }
        "vorbis" => {
            let headers = vorbis::split_xiph_lacing(&info.codec_private)
                .map_err(|e| anyhow::anyhow!("the Vorbis headers: {e}"))?;
            w.write_packet(&headers[0], 0, true, false)?;
            w.write_packet(&headers[1], 0, false, false)?;
            w.write_packet(&headers[2], 0, true, false)?;
            // Presentation from `media_time` on: a leading trim is a negative
            // start, which the granule positions of Vorbis I §A.2 express.
            (edit.media_time, edit.duration.map(|d| edit.media_time + d))
        }
        other => bail!("an Ogg file holds Opus or Vorbis, not {other}"),
    };
    let end = end.unwrap_or(total).min(total);
    let mut at = 0u64;
    for (i, (data, dur)) in packets.iter().enumerate() {
        at += u64::from(*dur);
        let last = i + 1 == packets.len();
        let granule = if last { end } else { at.min(end) } as i64 - start as i64;
        w.write_packet(data, granule, false, last)?;
    }
    Ok(w.into_inner())
}

/// The `OpusTags` header: the vendor and no comments (RFC 7845 §5.2).
fn opus_tags() -> Vec<u8> {
    let mut v = b"OpusTags".to_vec();
    v.extend_from_slice(&(VENDOR.len() as u32).to_le_bytes());
    v.extend_from_slice(VENDOR.as_bytes());
    v.extend_from_slice(&0u32.to_le_bytes());
    v
}

/// The audio of an Ogg file: its first Opus or Vorbis logical stream, as a
/// track timed by its packets' own durations, and the edit its granule
/// positions state (Opus: the pre-skip and the end; Vorbis: a leading trim
/// and the end). `None` for the edit when it presents every decoded sample.
pub fn read_audio(data: &[u8]) -> Result<(AudioTrack, Option<AudioEdit>)> {
    let mut reader = PacketReader::new(data);
    let mut serial = None;
    let mut kind = None;
    let mut headers: Vec<Vec<u8>> = Vec::new();
    let mut packets: Vec<Vec<u8>> = Vec::new();
    // The granule position at the end of each packet that ends a page.
    let mut granules: Vec<(usize, i64)> = Vec::new();
    while let Some(p) = reader.next_packet().map_err(|e| anyhow::anyhow!("Ogg: {e}"))? {
        if serial.is_none() {
            if !p.bos {
                continue;
            }
            if p.data.starts_with(b"OpusHead") {
                kind = Some(Kind::Opus);
            } else if p.data.starts_with(b"\x01vorbis") {
                kind = Some(Kind::Vorbis);
            } else if p.data.starts_with(b"\x80theora") {
                bail!("an Ogg file with Theora video: rivet reads Ogg audio only");
            } else {
                continue; // another codec's stream (skeleton, FLAC, ...)
            }
            serial = Some(p.serial);
        }
        if Some(p.serial) != serial {
            if p.bos && p.data.starts_with(b"\x80theora") {
                bail!("an Ogg file with Theora video: rivet reads Ogg audio only");
            }
            continue;
        }
        let need = match kind {
            Some(Kind::Opus) => 2,
            _ => 3,
        };
        if headers.len() < need {
            headers.push(p.data);
            continue;
        }
        if p.data.is_empty() {
            // An end-of-stream page that completes no packet: its granule
            // still ends the stream.
            if let Some(g) = p.granule {
                granules.push((packets.len(), g));
            }
            continue;
        }
        packets.push(p.data);
        if let Some(g) = p.granule {
            granules.push((packets.len(), g));
        }
        if p.eos {
            break;
        }
    }
    let kind = kind.context("Ogg: no Opus or Vorbis stream")?;
    if packets.is_empty() {
        bail!("Ogg: the {} stream has no audio packets", kind.name());
    }
    let last_granule = granules.last().map(|&(_, g)| g);
    match kind {
        Kind::Opus => {
            let body = headers[0][8..].to_vec();
            if body.len() < 11 {
                bail!("Ogg Opus: an OpusHead of {} bytes", body.len());
            }
            let channels = u16::from(body[1]);
            let pre_skip = u64::from(u16::from_le_bytes([body[2], body[3]]));
            let input_rate = u32::from_le_bytes([body[4], body[5], body[6], body[7]]);
            let durations: Vec<u32> = packets
                .iter()
                .map(|p| opus_packet_samples(p).context("Ogg Opus: a packet whose TOC does not parse"))
                .collect::<Result<_>>()?;
            let total: u64 = durations.iter().map(|&d| u64::from(d)).sum();
            // A stream that begins mid-way (RFC 7845 §4.5) counts from before
            // its first packet: the difference comes off the end position.
            let offset = first_offset(&granules, &durations).max(0);
            let end = last_granule.map(|g| (g - offset).max(0) as u64);
            let edit = AudioEdit { delay: 0, media_start: pre_skip, media_end: end.filter(|&e| e < total) };
            let track = AudioTrack {
                codec: "opus".into(),
                samples: packets,
                sample_rate: if input_rate == 0 { 48_000 } else { input_rate },
                channels,
                asc: Vec::new(),
                codec_private: body,
                timescale: 48_000,
                durations,
            };
            Ok((track, (!edit.is_identity(total)).then_some(edit)))
        }
        Kind::Vorbis => {
            let refs: [&[u8]; 3] = [&headers[0], &headers[1], &headers[2]];
            let private = vorbis::xiph_lacing(refs);
            let ident = vorbis::Identification::read(&headers[0]).map_err(|e| anyhow::anyhow!("Ogg Vorbis: {e}"))?;
            let durations = crate::demux::audio::vorbis_durations(&private, &packets)
                .context("Ogg Vorbis: a packet names a mode the setup header lacks")?;
            let total: u64 = durations.iter().map(|&d| u64::from(d)).sum();
            let offset = first_offset(&granules, &durations);
            // A negative offset is samples to drop at the start (§A.2).
            let start = (-offset).max(0) as u64;
            let end = last_granule.map(|g| (g - offset).max(0) as u64);
            let edit = AudioEdit { delay: 0, media_start: start, media_end: end.filter(|&e| e < total) };
            let track = AudioTrack {
                codec: "vorbis".into(),
                samples: packets,
                sample_rate: ident.sample_rate,
                channels: u16::from(ident.channels),
                asc: Vec::new(),
                codec_private: private,
                timescale: ident.sample_rate,
                durations,
            };
            Ok((track, (!edit.is_identity(total)).then_some(edit)))
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Opus,
    Vorbis,
}

impl Kind {
    fn name(self) -> &'static str {
        match self {
            Kind::Opus => "Opus",
            Kind::Vorbis => "Vorbis",
        }
    }
}

/// Granule position minus decoded samples at the first page that states
/// one: zero for a stream that starts at its beginning, negative for a
/// Vorbis stream whose first samples are to be dropped.
fn first_offset(granules: &[(usize, i64)], durations: &[u32]) -> i64 {
    let Some(&(n, g)) = granules.first() else {
        return 0;
    };
    let decoded: i64 = durations[..n.min(durations.len())].iter().map(|&d| i64::from(d)).sum();
    // The last page's position is an end trim, not an offset.
    if n == durations.len() && granules.len() == 1 && g <= decoded {
        return 0;
    }
    g - decoded
}

/// The samples (at 48 kHz) an Opus packet decodes to, from its TOC byte and
/// frame count (RFC 6716 §3.1–3.2).
pub fn opus_packet_samples(packet: &[u8]) -> Option<u32> {
    let toc = *packet.first()?;
    let config = toc >> 3;
    // Frame sizes in 48 kHz samples, by configuration (Table 2).
    let frame = match config {
        0..=11 => [480, 960, 1920, 2880][usize::from(config % 4)],
        12..=15 => [480, 960][usize::from(config % 2)],
        _ => [120, 240, 480, 960][usize::from(config % 4)],
    };
    let frames = match toc & 3 {
        0 => 1,
        1 | 2 => 2,
        _ => u32::from(*packet.get(1)? & 0x3F),
    };
    let n = frame * frames;
    (frames > 0 && n <= 5760).then_some(n)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal Opus packet: a CELT-only fullband 20 ms TOC (config 31),
    /// code 0, and a few bytes.
    fn opus_packet() -> Vec<u8> {
        vec![31 << 3, 0xAA, 0xBB, 0xCC]
    }

    fn opus_info(pre_skip: u16) -> AudioInfo {
        let mut head = vec![0u8, 2];
        head.extend_from_slice(&pre_skip.to_le_bytes());
        head.extend_from_slice(&48_000u32.to_le_bytes());
        head.extend_from_slice(&[0, 0, 0]);
        AudioInfo::opus(48_000, 2, head)
    }

    #[test]
    fn toc_durations_follow_rfc_6716() {
        assert_eq!(opus_packet_samples(&[31 << 3]), Some(960));
        assert_eq!(opus_packet_samples(&[(28 << 3) | 1]), Some(240));
        assert_eq!(opus_packet_samples(&[(3 << 3) | 2]), Some(5760));
        assert_eq!(opus_packet_samples(&[(1 << 3) | 3, 6]), Some(5760));
        assert_eq!(opus_packet_samples(&[(1 << 3) | 3, 7]), None, "over 120 ms");
        assert_eq!(opus_packet_samples(&[]), None);
    }

    /// An Opus stream written with a pre-skip and an end trim reads back
    /// packet for packet, with the edit an MP4 edit list would state.
    #[test]
    fn opus_round_trips_with_its_pre_skip_and_end() {
        let packets: Vec<(Vec<u8>, u32)> = (0..50).map(|_| (opus_packet(), 960)).collect();
        let edit = TrackEdit { delay: 0, media_time: 312, duration: Some(47_000) };
        let file = write_audio(&opus_info(312), &packets, edit).unwrap();
        assert!(sniff(&file));
        let (track, read) = read_audio(&file).unwrap();
        assert_eq!(track.codec, "opus");
        assert_eq!(track.samples.len(), 50);
        assert!(track.durations.iter().all(|&d| d == 960));
        assert_eq!(&track.codec_private[1..4], &[2, 0x38, 0x01]);
        assert_eq!(read, Some(AudioEdit { delay: 0, media_start: 312, media_end: Some(312 + 47_000) }));
    }

    #[test]
    fn other_codecs_and_late_starts_are_refused() {
        let packets = vec![(vec![1u8, 2, 3], 1024)];
        assert!(write_audio(&AudioInfo::aac_lc(48_000, 2, vec![0x11, 0x90]), &packets, TrackEdit::default()).is_err());
        let edit = TrackEdit { delay: 10, ..TrackEdit::default() };
        assert!(write_audio(&opus_info(312), &packets, edit).is_err());
        assert!(read_audio(b"not an ogg file at all, not one bit").is_err());
    }
}
