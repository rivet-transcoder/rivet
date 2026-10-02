//! The MPEG program stream demuxer (`.mpg`, `.mpeg`, `.vob`): MPEG-1 system
//! streams (ISO/IEC 11172-1) and MPEG-2 program streams (ISO/IEC 13818-1
//! §2.5), the files DVDs and older capture tools write.
//!
//! A program stream is a run of packs (`00 00 01 BA`), each a pack header and
//! PES packets: video on stream ids `0xE0`-`0xEF`, MPEG audio on
//! `0xC0`-`0xDF`, and on a DVD the AC-3 / DTS / LPCM audio in
//! `private_stream_1` (`0xBD`), each behind a sub-stream id. This reads the
//! first video stream and the first audio stream it can carry — MPEG audio,
//! or AC-3 (DVD sub-streams `0x80`-`0x87`) — and skips the rest (padding, the
//! DVD navigation packets of `private_stream_2`, subpictures, LPCM, DTS).
//!
//! The video is MPEG-1 or MPEG-2 (`mpeg1` when the sequence header has no
//! sequence extension). It is split into one sample per coded frame — a
//! frame picture, or the two field pictures of a field-coded frame — with the
//! sequence and GOP headers riding with the picture they precede, in coded
//! order; the decoder reorders. Samples are timed one frame apart from the
//! frame rate the sequence header states: an MPEG-2 decoder works from the
//! stream, and a program stream's PES timestamps are sparse (one per PES,
//! not per picture). The audio is placed against the video by the first
//! timestamps of each: a later start is kept as a delay, an earlier one is
//! trimmed.
//!
//! The whole file is read at construction, as the transport-stream reader
//! does: a program stream has no index to seek by.

use anyhow::{Result, bail};
use frame::{ColorSpace, PixelFormat, StreamInfo};

use crate::demux::AudioTrack;
use crate::edit::{AudioEdit, rescale_round};
use crate::mpeg_es::{MPEG2_GOP, MPEG2_PICTURE, MPEG2_SEQUENCE_HEADER, start_codes};
use crate::streaming::{DemuxHeader, Sample, StreamingDemuxer};

/// The 90 kHz clock of PES timestamps.
const PTS_HZ: u32 = 90_000;

/// Whether `data` opens like a program stream: a pack start code.
pub fn is_program_stream(data: &[u8]) -> bool {
    data.len() >= 12 && data[..4] == [0, 0, 1, 0xBA]
}

/// What one PES packet carries.
struct Pes<'a> {
    stream_id: u8,
    pts: Option<u64>,
    payload: &'a [u8],
}

/// The PES packets of `data`, in order. Pack headers, system headers and
/// the program end code are stepped over; a packet cut short by the end of
/// the file is dropped.
fn pes_packets(data: &[u8]) -> Vec<Pes<'_>> {
    let mut out = Vec::new();
    let mut pos = 0usize;
    while pos + 4 <= data.len() {
        if data[pos..pos + 3] != [0, 0, 1] {
            // Resynchronise on the next start code prefix.
            match data[pos + 1..].windows(3).position(|w| w == [0, 0, 1]) {
                Some(skip) => {
                    pos += 1 + skip;
                    continue;
                }
                None => break,
            }
        }
        let code = data[pos + 3];
        match code {
            0xBA => {
                // Pack header: MPEG-2 ('01' after the start code) is 14 bytes
                // plus its stuffing; MPEG-1 ('0010') is 12.
                let Some(&b) = data.get(pos + 4) else { break };
                if b >> 6 == 0b01 {
                    let Some(&stuff) = data.get(pos + 13) else { break };
                    pos += 14 + usize::from(stuff & 0x7);
                } else {
                    pos += 12;
                }
            }
            0xB9 => pos += 4, // MPEG_program_end_code
            0xBB..=0xFF => {
                let Some(len) = data.get(pos + 4..pos + 6) else { break };
                let len = usize::from(u16::from_be_bytes([len[0], len[1]]));
                let end = pos + 6 + len;
                if end > data.len() {
                    break;
                }
                if code != 0xBB && code != 0xBE && code != 0xBF && code != 0xBC {
                    if let Some((offset, pts)) = pes_header(&data[pos..end]) {
                        out.push(Pes { stream_id: code, pts, payload: &data[pos + offset..end] });
                    }
                }
                pos = end;
            }
            // Not a system-layer start code here: skip the prefix and resync.
            _ => pos += 3,
        }
    }
    out
}

/// Where a PES packet's payload starts, and its PTS — the MPEG-2 PES header
/// (`'10'` after the length), or the MPEG-1 one (stuffing, an optional STD
/// buffer size, then the timestamps).
fn pes_header(pes: &[u8]) -> Option<(usize, Option<u64>)> {
    let read_pts = |b: &[u8]| -> u64 {
        (u64::from((b[0] >> 1) & 0x07) << 30)
            | (u64::from(b[1]) << 22)
            | (u64::from(b[2] >> 1) << 15)
            | (u64::from(b[3]) << 7)
            | u64::from(b[4] >> 1)
    };
    let first = *pes.get(6)?;
    if first >> 6 == 0b10 {
        let flags = *pes.get(7)?;
        let header_len = usize::from(*pes.get(8)?);
        let start = 9 + header_len;
        if start > pes.len() {
            return None;
        }
        let pts = (flags >> 6 & 0b10 != 0).then(|| pes.get(9..14).map(read_pts)).flatten();
        return Some((start, pts));
    }
    let mut i = 6;
    while pes.get(i) == Some(&0xFF) {
        i += 1;
    }
    if pes.get(i)? >> 6 == 0b01 {
        i += 2;
    }
    let b = *pes.get(i)?;
    match b >> 4 {
        0b0010 => Some((i + 5, pes.get(i..i + 5).map(read_pts))),
        0b0011 => Some((i + 10, pes.get(i..i + 5).map(read_pts))),
        _ if b == 0x0F => Some((i + 1, None)),
        _ => None,
    }
}

/// The frame rate a sequence header's `frame_rate_code` names (H.262 Table
/// 6-4; the MPEG-1 table agrees for 1-8).
fn sequence_frame_rate(es: &[u8]) -> Option<f64> {
    let (o, _) = start_codes(es).into_iter().find(|(_, c)| *c == MPEG2_SEQUENCE_HEADER)?;
    let code = es.get(o + 7)? & 0x0F;
    Some(match code {
        1 => 24_000.0 / 1001.0,
        2 => 24.0,
        3 => 25.0,
        4 => 30_000.0 / 1001.0,
        5 => 30.0,
        6 => 50.0,
        7 => 60_000.0 / 1001.0,
        8 => 60.0,
        _ => return None,
    })
}

/// The `picture_structure` of the picture in `unit` (1 top field, 2 bottom
/// field, 3 frame) from its picture coding extension; 3 when it has none
/// (MPEG-1).
fn picture_structure(unit: &[u8]) -> u8 {
    start_codes(unit)
        .into_iter()
        .find(|&(o, c)| c == 0xB5 && unit.get(o + 4).is_some_and(|b| b >> 4 == 0x8))
        .and_then(|(o, _)| unit.get(o + 6).map(|b| b & 0x3))
        .unwrap_or(3)
}

/// One sample per coded frame: [`crate::mpeg_es::split_mpeg2_pictures`]'s
/// pictures, the second field of a field pair joined to the first.
fn coded_frames(es: &[u8]) -> Vec<Vec<u8>> {
    let mut out: Vec<Vec<u8>> = Vec::new();
    let mut open_field = false;
    for unit in crate::mpeg_es::split_mpeg2_pictures(es) {
        let structure = picture_structure(&unit);
        if structure != 3 && open_field {
            out.last_mut().expect("the first field").extend_from_slice(&unit);
            open_field = false;
        } else {
            open_field = structure != 3;
            out.push(unit);
        }
    }
    out
}

/// The program stream's video as samples and its audio, read whole.
pub struct PsStreamingDemuxer {
    header: DemuxHeader,
    samples: std::vec::IntoIter<Vec<u8>>,
    index: u64,
    frame_ticks: f64,
    audio: Option<AudioTrack>,
    audio_edit: Option<AudioEdit>,
}

pub(crate) fn demux_ps_streaming_init(data: bytes::Bytes) -> Result<PsStreamingDemuxer> {
    let packets = pes_packets(&data);
    let Some(video_id) = packets.iter().map(|p| p.stream_id).find(|id| (0xE0..=0xEF).contains(id)) else {
        bail!("MPEG program stream: no video stream");
    };
    let mut es = Vec::new();
    let mut video_pts: Option<u64> = None;
    for p in packets.iter().filter(|p| p.stream_id == video_id) {
        if let Some(pts) = p.pts {
            video_pts = Some(video_pts.map_or(pts, |v: u64| v.min(pts)));
        }
        es.extend_from_slice(p.payload);
    }
    // From the first sequence header: what comes before it cannot be decoded.
    let Some((start, _)) = start_codes(&es).into_iter().find(|(_, c)| *c == MPEG2_SEQUENCE_HEADER) else {
        bail!("MPEG program stream: the video has no sequence header");
    };
    let es = es.split_off(start);
    let mpeg2 = start_codes(&es)
        .into_iter()
        .take_while(|(_, c)| !matches!(*c, MPEG2_PICTURE | MPEG2_GOP))
        .any(|(o, c)| c == 0xB5 && es.get(o + 4).is_some_and(|b| b >> 4 == 0x1));
    let codec = if mpeg2 { "mpeg2" } else { "mpeg1" }.to_string();
    let frames = coded_frames(&es);
    if frames.is_empty() {
        bail!("MPEG program stream: the video has no picture");
    }
    let (width, height) = frame::pixel_format::detect_dims(&codec, std::slice::from_ref(&frames[0]))
        .ok_or_else(|| anyhow::anyhow!("MPEG program stream: unreadable sequence header"))?;
    let frame_rate = sequence_frame_rate(&es).unwrap_or(25.0);
    let mut info = StreamInfo {
        codec: codec.clone(),
        width,
        height,
        frame_rate,
        duration: frames.len() as f64 / frame_rate,
        pixel_format: PixelFormat::Yuv420p,
        color_space: ColorSpace::Bt709,
        total_frames: frames.len() as u64,
        bitrate: (data.len() as f64 * 8.0 / (frames.len() as f64 / frame_rate)) as u64,
        color_metadata: Default::default(),
    };
    let sample_aspect = crate::demux::aspect::resolve(
        None,
        || crate::demux::aspect::from_bitstream(&codec, &[], Some(&frames[0]), width, height),
        "ps",
    );
    crate::demux::hdr::resolve_source_colour(&mut info, Default::default(), &codec, &[], Some(&frames[0]), "ps");

    // Audio: the first MPEG audio stream, or the first AC-3 DVD sub-stream.
    let audio_choice = packets.iter().find_map(|p| match p.stream_id {
        0xC0..=0xDF => Some((p.stream_id, None)),
        0xBD => p.payload.first().filter(|s| (0x80..=0x87).contains(*s)).map(|&s| (0xBD, Some(s))),
        _ => None,
    });
    let (audio, audio_edit) = match audio_choice {
        None => (None, None),
        Some((id, sub)) => {
            let mut aes = Vec::new();
            let mut audio_pts: Option<u64> = None;
            for p in packets.iter().filter(|p| p.stream_id == id) {
                let payload = match sub {
                    // DVD private_stream_1: sub-stream id, frame count, and the
                    // first access unit's offset (2 bytes) before the frames.
                    Some(s) if p.payload.first() == Some(&s) => p.payload.get(4..).unwrap_or(&[]),
                    Some(_) => continue,
                    None => p.payload,
                };
                if audio_pts.is_none() {
                    audio_pts = p.pts;
                }
                aes.extend_from_slice(payload);
            }
            let track = match sub {
                Some(_) => crate::ts::audio::ac3_from_es(&aes).unwrap_or_else(|e| {
                    tracing::warn!(error = %e, "MPEG program stream: AC-3 audio unreadable; video only");
                    None
                }),
                None => crate::ts::audio::mpeg_audio_from_es(&aes),
            }
            .map(|(t, _)| t);
            let edit = match (&track, audio_pts, video_pts) {
                (Some(t), Some(a), Some(v)) if a != v => {
                    let sr = t.timescale.max(1);
                    Some(if a > v {
                        AudioEdit { delay: rescale_round(a - v, sr, PTS_HZ), media_start: 0, media_end: None }
                    } else {
                        AudioEdit { delay: 0, media_start: rescale_round(v - a, sr, PTS_HZ), media_end: None }
                    })
                }
                _ => None,
            };
            (track, edit)
        }
    };

    Ok(PsStreamingDemuxer {
        header: DemuxHeader { codec, info, timescale: PTS_HZ, rotation_degrees: 0, sample_aspect },
        samples: frames.into_iter(),
        index: 0,
        frame_ticks: f64::from(PTS_HZ) / frame_rate,
        audio,
        audio_edit,
    })
}

impl StreamingDemuxer for PsStreamingDemuxer {
    fn header(&self) -> &DemuxHeader {
        &self.header
    }

    fn next_video_sample(&mut self) -> Result<Option<Sample>> {
        let Some(data) = self.samples.next() else { return Ok(None) };
        let pts_ticks = (self.index as f64 * self.frame_ticks).round() as i64;
        self.index += 1;
        Ok(Some(Sample { data, pts_ticks, duration_ticks: self.frame_ticks.round() as u32 }))
    }

    fn audio(&self) -> Option<&AudioTrack> {
        self.audio.as_ref()
    }

    fn audio_edit(&self) -> Option<AudioEdit> {
        self.audio_edit
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mpeg1_and_mpeg2_pes_headers_read() {
        // MPEG-2: '10' flags, PTS only.
        let pts = 0x1_2345_6789u64 & ((1 << 33) - 1);
        let b = [
            0x21 | (((pts >> 30) & 0x7) << 1) as u8,
            (pts >> 22) as u8,
            0x01 | (((pts >> 15) & 0x7F) << 1) as u8,
            (pts >> 7) as u8,
            0x01 | ((pts & 0x7F) << 1) as u8,
        ];
        let mut m2 = vec![0, 0, 1, 0xE0, 0, 0, 0x80, 0x80, 5];
        m2.extend(b);
        m2.extend([0xAA, 0xBB]);
        assert_eq!(pes_header(&m2), Some((14, Some(pts))));
        // MPEG-1: stuffing, STD buffer, PTS.
        let mut m1 = vec![0, 0, 1, 0xE0, 0, 0, 0xFF, 0xFF, 0x40, 0x20];
        m1.extend(b);
        assert_eq!(pes_header(&m1), Some((15, Some(pts))));
        // MPEG-1 without timestamps.
        assert_eq!(pes_header(&[0, 0, 1, 0xC0, 0, 0, 0x0F, 0x55]), Some((7, None)));
    }

    #[test]
    fn a_pack_sniffs_as_a_program_stream() {
        let mut d = vec![0, 0, 1, 0xBA, 0x44];
        d.resize(16, 0);
        assert!(is_program_stream(&d));
        assert!(!is_program_stream(&[0, 0, 1, 0xB3, 0, 0, 0, 0, 0, 0, 0, 0]));
    }
}
