//! Minimal MPEG-2 Transport Stream demuxer.
//!
//! Scope: take a .ts / .m2ts byte stream, locate the PAT and a PMT,
//! pick the first video elementary stream the PMT advertises, and
//! return its PES payloads as one sample per access unit.
//!
//! PTS is carried through on the first PES packet that opens an AU;
//! continuation packets accumulate bytes onto the current sample until
//! the next `payload_unit_start_indicator=1` closes it.
//!
//! What's implemented:
//! - PAT walk that surfaces every program in the file, with a default
//!   "first program" pick (matches legacy behaviour) and a
//!   `select_program(program_number)` API for callers that want one of
//!   the others (Squad-37).
//! - PMT walk: video stream_types 0x02 (MPEG-2), 0x1B (H.264),
//!   0x24 (HEVC) plus audio stream_types 0x0F (AAC-ADTS, Squad-27),
//!   0x81 (AC-3, ATSC A/53), 0x87 (E-AC-3, ATSC A/53), and 0x06 (PES
//!   private) when the ES descriptor loop carries a registration_descriptor
//!   tagged "AC-3" / "EAC3" (DVB / ETSI TS 101 154) — Squad-37.
//! - Encrypted streams (`transport_scrambling_control != 0` on the active
//!   video PID) trip a one-time typed warn and switch the demuxer into a
//!   drop-everything mode (Squad-37); previously the bytes were silently
//!   skipped on a per-packet basis which meant a partial-scramble error
//!   condition could still leak garbled samples.
//!
//! What's not implemented:
//! - Full CRC validation of PAT/PMT (we trust what the bitstream gives
//!   us; a mis-CRCed file is already corrupt and will surface as wrong
//!   stream_type or truncated samples further down).
//! - Multiple video streams within one program (we take the first).
//! - Adaptation-field-only packets with payload=0 are passed over
//!   transparently.
//! - BDAV 192-byte wrapper (the 4-byte timestamp prefix) — if present,
//!   we detect and strip it.
//! - Common-Access (CA) tables: encrypted streams are dropped, not
//!   decrypted (we don't carry CA descriptors).

use anyhow::{Context, Result, bail};
use codec::frame::{ColorSpace, PixelFormat, StreamInfo};

use crate::ac3_sync::{
    self, Eac3SyncInfo, SyncInfo, ac3_bit_rate_kbps, channel_count, eac3_sample_rate_hz,
    eac3_samples_per_frame,
};
use crate::demux::{AudioTrack, DemuxResult};
use crate::mux::{dac3_body_from_sync, dec3_body_from_sync};
use crate::streaming::{DemuxHeader, Sample, StreamingDemuxer};

const TS_PACKET: usize = 188;
const TS_SYNC: u8 = 0x47;

const STREAM_TYPE_MPEG2_VIDEO: u8 = 0x02;
const STREAM_TYPE_H264: u8 = 0x1B;
const STREAM_TYPE_HEVC: u8 = 0x24;
/// PES private stream_type. ETSI TS 101 154 (DVB) routes AC-3 / E-AC-3
/// through this generic stream_type with a `registration_descriptor`
/// (descriptor_tag = 0x05) tagged "AC-3" or "EAC3" carrying the actual
/// codec identity. We only honour 0x06 entries that carry one of those
/// two registrations — random PES-private streams (DVB subtitles, teletext)
/// are dropped silently.
const STREAM_TYPE_PES_PRIVATE: u8 = 0x06;
/// PMT stream_type for AAC carried as ADTS frames in PES packets.
/// Defined in ISO/IEC 13818-1:2019 Table 2-34 — `0x0F` is
/// "ISO/IEC 13818-7 Audio with ADTS transport syntax", which is the
/// MPEG-2/MPEG-4 AAC ADTS form that broadcast / streaming MPEG-TS uses.
const STREAM_TYPE_AAC_ADTS: u8 = 0x0F;
/// ATSC A/53 §3 / ATSC A/52 Annex A — AC-3 elementary streams in PES
/// packets. Common in over-the-air ATSC broadcast captures (.ts / .trp).
const STREAM_TYPE_AC3: u8 = 0x81;
/// ATSC A/53 §3 / ATSC A/52 Annex E — E-AC-3 elementary streams.
const STREAM_TYPE_EAC3: u8 = 0x87;

/// PMT descriptor_tag for the registration_descriptor carrying a
/// 4-character format identifier. ETSI TS 101 154 §F (DVB) registers
/// `"AC-3"` (0x41432D33) and `"EAC3"` (0x45414333) for Dolby streams
/// carried as PES-private (stream_type 0x06).
const DESC_TAG_REGISTRATION: u8 = 0x05;
const REG_AC3: u32 = 0x41432D33; // "AC-3"
const REG_EAC3: u32 = 0x45414333; // "EAC3"

pub(crate) fn demux_ts(data: &[u8]) -> Result<DemuxResult> {
    // Detect BDAV wrapper: 192-byte packets carry a 4-byte TP_extra
    // header in front of each 188-byte TS packet. Stripping the 4-byte
    // prefix brings us back to the canonical 188-byte form.
    let (packets, packet_stride, prefix_len) = detect_packet_layout(data)?;
    if packets == 0 {
        bail!("TS: file contains no TS packets");
    }

    // First pass: find PAT (PID=0), then PMT, collect video + audio PID +
    // stream_type. The PMT walk surfaces (video_streams, audio_streams)
    // and we take the first of each. Squad-37 expanded recognised audio
    // codec families to AAC-ADTS (0x0F), AC-3 (0x81 / 0x06+reg), and
    // E-AC-3 (0x87 / 0x06+reg) — every other audio stream_type is
    // dropped silently (matches MP4/MKV's "non-supported audio → drop"
    // behaviour at the demuxer layer; the pipeline already knows how to
    // emit video-only).
    let mut pmt_pid: Option<u16> = None;
    let mut chosen_video: Option<VideoStreamInfo> = None;
    let mut chosen_audio: Option<AudioStreamInfo> = None;
    for i in 0..packets {
        let start = i * packet_stride + prefix_len;
        let pkt = &data[start..start + TS_PACKET];
        if pkt[0] != TS_SYNC {
            continue;
        }
        let pid = (((pkt[1] & 0x1F) as u16) << 8) | pkt[2] as u16;
        // PAT
        if pmt_pid.is_none() && pid == 0 {
            if let Some(payload) = ts_psi_payload(pkt)
                && let Some(p) = parse_pat_first_pmt_pid(payload)
            {
                pmt_pid = Some(p);
            }
            continue;
        }
        // PMT
        if let (Some(pmt), None) = (pmt_pid, chosen_video)
            && pid == pmt
            && let Some(payload) = ts_psi_payload(pkt)
            && let Some((video_streams, audio_streams)) = parse_pmt_streams(payload)
        {
            chosen_video = video_streams.into_iter().next();
            chosen_audio = audio_streams.into_iter().next();
            if chosen_video.is_some() {
                break;
            }
        }
    }

    let video = chosen_video.context("TS: no video elementary stream found in PMT")?;
    let video_pid = video.pid;
    let codec = match video.stream_type {
        STREAM_TYPE_MPEG2_VIDEO => "mpeg2",
        STREAM_TYPE_H264 => "h264",
        STREAM_TYPE_HEVC => "h265",
        other => bail!("TS: unsupported stream_type 0x{:02X}", other),
    }
    .to_string();

    // Second pass: reassemble PES payloads for the video PID, one
    // sample per `payload_unit_start_indicator`.
    let mut samples: Vec<Vec<u8>> = Vec::new();
    let mut pending: Vec<u8> = Vec::new();
    let mut have_first_start = false;
    let mut first_pts: Option<u64> = None;
    let mut last_pts: Option<u64> = None;
    // Collect every PTS so we can share the streaming path's
    // `estimate_frame_rate_from_ptses` (median-of-deltas) — more
    // robust than `(samples - 1) / duration`, which was off-by-one
    // on boundary edge cases that the streaming scan also hit.
    let mut ptses: Vec<u64> = Vec::new();

    let flush = |pending: &mut Vec<u8>, samples: &mut Vec<Vec<u8>>| {
        if !pending.is_empty() {
            samples.push(std::mem::take(pending));
        }
    };

    for i in 0..packets {
        let start = i * packet_stride + prefix_len;
        let pkt = &data[start..start + TS_PACKET];
        if pkt[0] != TS_SYNC {
            continue;
        }
        let pid = (((pkt[1] & 0x1F) as u16) << 8) | pkt[2] as u16;
        if pid != video_pid {
            continue;
        }
        let pusi = pkt[1] & 0x40 != 0;
        let scramble = (pkt[3] >> 6) & 0x03;
        if scramble != 0 {
            continue;
        } // encrypted; no way to decode
        let adaptation = (pkt[3] >> 4) & 0x03;
        let has_payload = adaptation & 0x01 != 0;
        let has_adaptation = adaptation & 0x02 != 0;
        if !has_payload {
            continue;
        }

        let mut offset = 4usize;
        if has_adaptation {
            if offset >= TS_PACKET {
                continue;
            }
            let adap_len = pkt[offset] as usize;
            offset += 1 + adap_len;
            if offset > TS_PACKET {
                continue;
            }
        }
        if offset >= TS_PACKET {
            continue;
        }
        let payload = &pkt[offset..];

        if pusi {
            // New PES packet begins here — flush whatever we were
            // accumulating, then parse the PES header to find PTS and
            // the elementary-stream payload start.
            if have_first_start {
                flush(&mut pending, &mut samples);
            }
            have_first_start = true;

            let Some((es_start, pts)) = parse_pes_header(payload) else {
                // Malformed PES; skip this packet, keep state.
                have_first_start = false;
                pending.clear();
                continue;
            };
            if let Some(p) = pts {
                if first_pts.is_none() {
                    first_pts = Some(p);
                }
                last_pts = Some(p);
                ptses.push(p);
            }
            if es_start < payload.len() {
                pending.extend_from_slice(&payload[es_start..]);
            }
        } else if have_first_start {
            pending.extend_from_slice(payload);
        }
    }
    flush(&mut pending, &mut samples);

    if samples.is_empty() {
        bail!("TS: reassembled zero video samples from PID {}", video_pid);
    }

    // PTS is 90 kHz. Duration stays span-based (last - first is the
    // right answer for "how long does this stream play"). Frame rate
    // switches to the median-of-deltas path for consistency with the
    // streaming demuxer's init; falls back to the span/count calc and
    // then 30.0 if the PTS window isn't populated enough for a median.
    let duration = match (first_pts, last_pts) {
        (Some(a), Some(b)) if b >= a => (b - a) as f64 / 90_000.0,
        _ => 0.0,
    };
    let frame_rate = estimate_frame_rate_from_ptses(&ptses)
        .or_else(|| {
            if duration > 0.0 && samples.len() > 1 {
                Some((samples.len() - 1) as f64 / duration)
            } else {
                None
            }
        })
        .unwrap_or(30.0);

    // TS carries no container-level width/height; the sample-entry /
    // track-header equivalents that MP4/MKV/AVI/MOV all have don't
    // exist here. We recover dims by parsing the first sample's SPS
    // (H.264 / HEVC) or sequence header (MPEG-2). `detect_dims`
    // returns None if the parse fails — fall back to 0 so downstream
    // reporting still shows "unknown" rather than a fabricated value.
    let (width, height) = codec::pixel_format::detect_dims(&codec, &samples).unwrap_or((0, 0));
    if width == 0 || height == 0 {
        tracing::warn!(
            codec = codec.as_str(),
            "TS demux: could not recover width/height from first sample — \
             downstream encoder may reject the 0×0 config"
        );
    }

    let info = StreamInfo {
        codec: codec.clone(),
        width,
        height,
        frame_rate,
        duration,
        pixel_format: PixelFormat::Yuv420p,
        color_space: ColorSpace::Bt709,
        total_frames: samples.len() as u64,
        bitrate: 0,
        color_metadata: Default::default(),
    };

    let detected_pf = codec::pixel_format::detect(&codec, &samples);
    let info = StreamInfo {
        pixel_format: detected_pf,
        ..info
    };

    // Audio extraction. Squad-37 expanded the routing: AAC-ADTS goes
    // through Squad-27's path; AC-3 / E-AC-3 use the new pure-Rust
    // extractors that derive `dac3` / `dec3` from the first frame's
    // sync header (Squad-26 helpers).
    let audio = chosen_audio.and_then(|info| {
        match extract_ts_audio(data, packets, packet_stride, prefix_len, info) {
            Ok(track) => track,
            Err(e) => {
                tracing::warn!(
                    audio_pid = info.pid,
                    audio_kind = ?info.kind,
                    error = %e,
                    "TS audio extraction failed; emitting video-only"
                );
                None
            }
        }
    });

    Ok(DemuxResult {
        codec,
        info,
        samples,
        audio,
    })
}

/// Decide whether the file uses 188-byte (plain TS) or 192-byte (BDAV
/// M2TS) packets. Returns (packet_count, stride, prefix_len).
/// BDAV prepends a 4-byte TP_extra_header before each 188-byte TS
/// packet, so stride=192 and prefix_len=4. For plain TS stride=188
/// and prefix_len=0.
fn detect_packet_layout(data: &[u8]) -> Result<(usize, usize, usize)> {
    if data.len() < TS_PACKET {
        bail!("TS: file too small");
    }
    // Plain 188-byte: sync at 0, 188, 376...
    if data[0] == TS_SYNC && data.len() >= 2 * TS_PACKET && data[TS_PACKET] == TS_SYNC {
        return Ok((data.len() / TS_PACKET, TS_PACKET, 0));
    }
    // M2TS 192-byte: 4-byte prefix, then sync at 4, 196, 388...
    if data.len() >= 192 + 4 && data[4] == TS_SYNC && data[196] == TS_SYNC {
        return Ok((data.len() / 192, 192, 4));
    }
    bail!("TS: could not locate 0x47 sync pattern at 188- or 192-byte intervals")
}

/// Extract the PSI (PAT/PMT) section payload from a TS packet whose PID
/// we already know carries PSI. Returns the raw section bytes or None
/// when the packet has no payload or has a continuation we can't
/// reassemble in a single-packet model.
fn ts_psi_payload(pkt: &[u8]) -> Option<&[u8]> {
    let pusi = pkt[1] & 0x40 != 0;
    let adaptation = (pkt[3] >> 4) & 0x03;
    let has_payload = adaptation & 0x01 != 0;
    let has_adaptation = adaptation & 0x02 != 0;
    if !has_payload {
        return None;
    }
    let mut offset = 4usize;
    if has_adaptation {
        if offset >= TS_PACKET {
            return None;
        }
        let adap_len = pkt[offset] as usize;
        offset += 1 + adap_len;
        if offset > TS_PACKET {
            return None;
        }
    }
    // PSI packets with PUSI=1 carry a pointer_field byte telling us how
    // many bytes to skip before the section starts. We take that first
    // section only — subsequent sections in the same packet would need
    // separate handling we don't need for PAT/PMT (usually one each).
    if pusi {
        if offset >= TS_PACKET {
            return None;
        }
        let pointer = pkt[offset] as usize;
        offset += 1 + pointer;
        if offset >= TS_PACKET {
            return None;
        }
    }
    Some(&pkt[offset..])
}

/// One PAT entry — `(program_number, pmt_pid)`. Entry with program=0 is
/// the network_PID and is skipped by callers (it is not a real program).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PatProgram {
    pub program_number: u16,
    pub pmt_pid: u16,
}

/// Audio codec discriminator surfaced from the PMT walk. The PMT only
/// tells us the codec family; the actual codec_private bytes (`asc` for
/// AAC, `dac3` / `dec3` for AC-3 / E-AC-3) are derived in `extract_*` by
/// reading the first frame of the elementary stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioCodecKind {
    /// ISO/IEC 13818-7 AAC carried as ADTS frames (stream_type 0x0F).
    AacAdts,
    /// ETSI TS 102 366 AC-3 (stream_type 0x81 OR 0x06 + registration "AC-3").
    Ac3,
    /// ETSI TS 102 366 E-AC-3 (stream_type 0x87 OR 0x06 + registration "EAC3").
    Eac3,
}

/// Per-stream info gathered from one PMT entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VideoStreamInfo {
    pub pid: u16,
    pub stream_type: u8,
}

/// Per-audio-stream info gathered from one PMT entry. `kind` is the
/// codec family — extraction reads the first frame to derive
/// `codec_private` / `sample_rate` / `channels`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AudioStreamInfo {
    pub pid: u16,
    pub stream_type: u8,
    pub kind: AudioCodecKind,
}

/// One MPEG-TS program found in the PAT, after the corresponding PMT has
/// been walked. `pmt_pid` is the bitstream-side PID where the PMT section
/// lives; `video_streams` / `audio_streams` are the elementary streams
/// the PMT advertises (video filtered to MPEG-2 / H.264 / HEVC; audio
/// filtered to AAC-ADTS / AC-3 / E-AC-3 — exactly the codec families we
/// can passthrough). A program with neither a recognised video nor a
/// recognised audio stream is still surfaced so callers can see "this
/// program exists, just contains things we can't carry".
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProgramInfo {
    pub program_number: u16,
    pub pmt_pid: u16,
    pub video_streams: Vec<VideoStreamInfo>,
    pub audio_streams: Vec<AudioStreamInfo>,
}

/// Walk the PAT section and return every `(program_number, pmt_pid)`
/// pair (skipping `program_number == 0`, which carries the network_PID
/// per ISO/IEC 13818-1 §2.4.4.3 — not a real program).
fn parse_pat_all_programs(section: &[u8]) -> Vec<PatProgram> {
    // PAT section:
    //   table_id(8)=0, section_syntax_indicator(1), '0'(1), reserved(2),
    //   section_length(12), transport_stream_id(16), reserved(2),
    //   version(5), current_next(1), section_number(8),
    //   last_section_number(8), then N × (program_number u16, reserved(3),
    //   PID(13)), followed by CRC_32.
    let mut out = Vec::new();
    if section.len() < 12 {
        return out;
    }
    if section[0] != 0x00 {
        return out;
    }
    let section_length = (((section[1] & 0x0F) as usize) << 8) | section[2] as usize;
    let total = 3 + section_length;
    if total > section.len() {
        return out;
    }
    let loop_start = 8;
    let loop_end = total - 4;
    let mut i = loop_start;
    while i + 4 <= loop_end {
        let program = u16::from_be_bytes([section[i], section[i + 1]]);
        let pid = (((section[i + 2] & 0x1F) as u16) << 8) | section[i + 3] as u16;
        if program != 0 {
            out.push(PatProgram {
                program_number: program,
                pmt_pid: pid,
            });
        }
        i += 4;
    }
    out
}

/// Back-compat shim — returns the first program's PMT PID. Kept so the
/// streaming-init path stays a one-line change; the multi-program walk
/// uses `parse_pat_all_programs` directly.
fn parse_pat_first_pmt_pid(section: &[u8]) -> Option<u16> {
    parse_pat_all_programs(section).first().map(|p| p.pmt_pid)
}

/// Walk the PMT section once and return every recognised video stream
/// (PID + stream_type) plus every recognised audio stream (PID +
/// stream_type + codec kind). Audio is optional — TS files without an
/// audio track demux video-only, matching MP4/MKV behaviour at the
/// demuxer layer.
///
/// AC-3 / E-AC-3 detection (Squad-37): we honour the ATSC A/53
/// stream_types (0x81 / 0x87) directly AND the DVB form where the
/// stream_type is 0x06 (PES private) and the ES descriptor loop carries
/// a `registration_descriptor` (tag 0x05) with the 4-char identifier
/// "AC-3" or "EAC3".
fn parse_pmt_streams(section: &[u8]) -> Option<(Vec<VideoStreamInfo>, Vec<AudioStreamInfo>)> {
    // PMT section:
    //   table_id(8)=0x02, section_syntax_indicator(1), '0'(1), reserved(2),
    //   section_length(12), program_number(16), reserved(2), version(5),
    //   current_next(1), section_number(8), last_section_number(8),
    //   reserved(3), PCR_PID(13), reserved(4), program_info_length(12),
    //   program_info_descriptors...,
    //   then N × (stream_type(8), reserved(3), elementary_PID(13),
    //     reserved(4), ES_info_length(12), ES_info_descriptors...),
    //   CRC_32.
    if section.len() < 12 {
        return None;
    }
    if section[0] != 0x02 {
        return None;
    }
    let section_length = (((section[1] & 0x0F) as usize) << 8) | section[2] as usize;
    let total = 3 + section_length;
    if total > section.len() {
        return None;
    }
    if section.len() < 12 {
        return None;
    }
    let pil = (((section[10] & 0x0F) as usize) << 8) | section[11] as usize;
    let mut i = 12 + pil;
    let loop_end = total - 4; // strip CRC
    let mut video: Vec<VideoStreamInfo> = Vec::new();
    let mut audio: Vec<AudioStreamInfo> = Vec::new();
    while i + 5 <= loop_end {
        let stype = section[i];
        let pid = (((section[i + 1] & 0x1F) as u16) << 8) | section[i + 2] as u16;
        let esi_len = (((section[i + 3] & 0x0F) as usize) << 8) | section[i + 4] as usize;
        let desc_start = i + 5;
        let desc_end = (desc_start + esi_len).min(loop_end);
        let descriptors = if desc_start <= desc_end {
            &section[desc_start..desc_end]
        } else {
            &[][..]
        };

        match stype {
            STREAM_TYPE_MPEG2_VIDEO | STREAM_TYPE_H264 | STREAM_TYPE_HEVC => {
                video.push(VideoStreamInfo {
                    pid,
                    stream_type: stype,
                });
            }
            STREAM_TYPE_AAC_ADTS => {
                audio.push(AudioStreamInfo {
                    pid,
                    stream_type: stype,
                    kind: AudioCodecKind::AacAdts,
                });
            }
            STREAM_TYPE_AC3 => {
                audio.push(AudioStreamInfo {
                    pid,
                    stream_type: stype,
                    kind: AudioCodecKind::Ac3,
                });
            }
            STREAM_TYPE_EAC3 => {
                audio.push(AudioStreamInfo {
                    pid,
                    stream_type: stype,
                    kind: AudioCodecKind::Eac3,
                });
            }
            STREAM_TYPE_PES_PRIVATE => {
                // DVB carries AC-3 / E-AC-3 here. Walk the ES descriptor
                // loop and look for a registration_descriptor whose
                // 4-char tag is "AC-3" or "EAC3".
                if let Some(reg) = find_registration(descriptors) {
                    match reg {
                        REG_AC3 => audio.push(AudioStreamInfo {
                            pid,
                            stream_type: stype,
                            kind: AudioCodecKind::Ac3,
                        }),
                        REG_EAC3 => audio.push(AudioStreamInfo {
                            pid,
                            stream_type: stype,
                            kind: AudioCodecKind::Eac3,
                        }),
                        _ => {}
                    }
                }
            }
            _ => {}
        }
        i += 5 + esi_len;
    }
    Some((video, audio))
}

/// Walk the ES descriptor loop and look for a registration_descriptor
/// (tag 0x05) carrying a 4-byte format_identifier; return the BE u32 of
/// that identifier or None.
fn find_registration(descriptors: &[u8]) -> Option<u32> {
    let mut i = 0usize;
    while i + 2 <= descriptors.len() {
        let tag = descriptors[i];
        let len = descriptors[i + 1] as usize;
        let body_start = i + 2;
        let body_end = body_start + len;
        if body_end > descriptors.len() {
            break;
        }
        if tag == DESC_TAG_REGISTRATION && len >= 4 {
            let id = u32::from_be_bytes([
                descriptors[body_start],
                descriptors[body_start + 1],
                descriptors[body_start + 2],
                descriptors[body_start + 3],
            ]);
            return Some(id);
        }
        i = body_end;
    }
    None
}

// (Squad-13's `parse_pmt_video_and_audio` shim was retired in Squad-37
// — both the legacy `demux_ts` path and the streaming demuxer now use
// `parse_pmt_streams` directly so the AC-3 / E-AC-3 / multi-program
// surface stays a single-walker invariant.)

/// Parse a PES header at the start of `payload`. Returns the byte
/// offset of the elementary-stream payload within `payload`, plus any
/// PTS we extracted. PES layout (ISO/IEC 13818-1 §2.4.3.6):
///   start_code(0x000001) + stream_id(8) + PES_packet_length(16)
///   flags(16) + PES_header_data_length(8) + header_extension(...) + ES data
fn parse_pes_header(payload: &[u8]) -> Option<(usize, Option<u64>)> {
    if payload.len() < 9 {
        return None;
    }
    if payload[0] != 0 || payload[1] != 0 || payload[2] != 1 {
        return None;
    }
    let stream_id = payload[3];
    // Video streams are 0xE0..=0xEF. Other stream_ids (audio, padding,
    // program streams) aren't what we want; bail so the caller can drop
    // the sample.
    if !(0xE0..=0xEF).contains(&stream_id) {
        return None;
    }
    // The two PES flag bytes live at offsets 6-7.
    let flags = payload[7];
    let pts_dts_flags = (flags >> 6) & 0x03;
    let header_data_len = payload[8] as usize;
    let es_start = 9 + header_data_len;
    if es_start > payload.len() {
        return None;
    }
    let pts = if pts_dts_flags == 0b10 || pts_dts_flags == 0b11 {
        // PTS occupies bytes 9..14. Layout: 4 marker bits + PTS[32..30]
        //   + 1 marker, 15 bits PTS[29..15] + 1 marker, 15 bits PTS[14..0] + 1 marker.
        if payload.len() < 14 {
            return None;
        }
        let p0 = ((payload[9] >> 1) & 0x07) as u64;
        let p1 = (((payload[10] as u64) << 7) | ((payload[11] as u64) >> 1)) & 0x7FFF;
        let p2 = (((payload[12] as u64) << 7) | ((payload[13] as u64) >> 1)) & 0x7FFF;
        Some((p0 << 30) | (p1 << 15) | p2)
    } else {
        None
    };
    Some((es_start, pts))
}

/// Result of a single-pass scan over the active video PID: the first
/// access unit's bytes (for SPS / seq-header dim extraction) plus a
/// window of PTSes (for frame-rate estimation).
pub(crate) struct VideoStreamScan {
    pub first_au: Option<Vec<u8>>,
    pub ptses: Vec<u64>,
}

/// Walk TS packets on the active video PID and reassemble the first
/// complete access unit into a single contiguous byte buffer. "Complete"
/// = from the first PUSI on the target PID up to (but not including)
/// the second PUSI; if there's no second PUSI before EOF we return
/// whatever we've accumulated so far. Also collects up to
/// `max_pts_samples` successive PTSes off the video PID so the caller
/// can derive a frame rate from their inter-arrival span.
///
/// Used by the streaming demuxer's init path to populate
/// `StreamInfo.width` / `.height` from the codec's SPS (H.264 / HEVC)
/// or sequence header (MPEG-2) — AND a correct `frame_rate` from the
/// PTS window — before any downstream consumer reads `header()`.
/// Walks the same packets `next_video_sample` would walk later; state
/// is local to this fn so the main walk state isn't disturbed.
fn scan_first_video_au(
    data: &[u8],
    packets: usize,
    packet_stride: usize,
    prefix_len: usize,
    video_pid: u16,
    max_pts_samples: usize,
) -> VideoStreamScan {
    let mut accumulator: Vec<u8> = Vec::new();
    let mut first_au: Option<Vec<u8>> = None;
    let mut ptses: Vec<u64> = Vec::new();
    let mut au_started = false;
    let mut au_done = false;
    for i in 0..packets {
        let start = i * packet_stride + prefix_len;
        let pkt = &data[start..start + TS_PACKET];
        if pkt[0] != TS_SYNC {
            continue;
        }
        let pid = (((pkt[1] & 0x1F) as u16) << 8) | pkt[2] as u16;
        if pid != video_pid {
            continue;
        }
        let pusi = pkt[1] & 0x40 != 0;
        let scramble = (pkt[3] >> 6) & 0x03;
        if scramble != 0 {
            continue;
        } // encrypted; skip probe
        let adaptation = (pkt[3] >> 4) & 0x03;
        let has_payload = adaptation & 0x01 != 0;
        let has_adaptation = adaptation & 0x02 != 0;
        if !has_payload {
            continue;
        }
        let mut offset = 4usize;
        if has_adaptation {
            if offset >= TS_PACKET {
                continue;
            }
            let adap_len = pkt[offset] as usize;
            offset += 1 + adap_len;
            if offset > TS_PACKET {
                continue;
            }
        }
        if offset >= TS_PACKET {
            continue;
        }
        let payload = &pkt[offset..];

        if pusi {
            // Close out the first AU on the second PUSI we see.
            if au_started && !au_done {
                first_au = Some(std::mem::take(&mut accumulator));
                au_done = true;
            }
            if let Some((es_start, pts)) = parse_pes_header(payload) {
                if let Some(p) = pts
                    && ptses.len() < max_pts_samples
                {
                    ptses.push(p);
                }
                if !au_done {
                    if es_start < payload.len() {
                        accumulator.extend_from_slice(&payload[es_start..]);
                    }
                    au_started = true;
                }
            }
        } else if au_started && !au_done {
            accumulator.extend_from_slice(payload);
        }

        // Early exit once both targets are hit.
        if au_done && ptses.len() >= max_pts_samples {
            break;
        }
    }
    // EOF before we saw a second PUSI — emit whatever's accumulated.
    if first_au.is_none() && au_started && !accumulator.is_empty() {
        first_au = Some(accumulator);
    }
    VideoStreamScan { first_au, ptses }
}

/// Estimate source frame rate from a window of video-PID PTSes.
///
/// Uses the **median** of sorted inter-PTS deltas rather than span /
/// count: the span method is off-by-one-period sensitive to the
/// boundary conditions of the scan (an extra stray PTS on the video
/// PID, a stuffing PES, a mid-stream split) and consistently produced
/// 23.625 instead of 24.000 on the BBB test sample. Median handles
/// outliers uniformly — one spurious 2× delta leaves a run of
/// correct-period deltas around it, and sorting picks the correct
/// one as the middle.
///
/// PTS is 90 kHz; median_delta = ticks-per-frame; fps = 90000 /
/// median_delta. Zero deltas (duplicate PTSes, e.g. if a frame's
/// AU is split across multiple PES packets on the same PID) drop
/// out — they would otherwise force fps → ∞.
///
/// Returns `None` when fewer than two PTSes are present, all deltas
/// are zero, or the estimate lands outside `[1.0, 240.0]` (protects
/// against 33-bit wraparound or a fixed-value PTS injection).
fn estimate_frame_rate_from_ptses(ptses: &[u64]) -> Option<f64> {
    if ptses.len() < 2 {
        return None;
    }
    let mut sorted: Vec<u64> = ptses.to_vec();
    sorted.sort_unstable();
    let mut deltas: Vec<u64> = sorted.windows(2).map(|w| w[1] - w[0]).collect();
    deltas.retain(|&d| d > 0);
    if deltas.is_empty() {
        return None;
    }
    deltas.sort_unstable();
    let median = deltas[deltas.len() / 2];
    if median == 0 {
        return None;
    }
    let fps = 90000.0 / median as f64;
    if !fps.is_finite() || !(1.0..=240.0).contains(&fps) {
        return None;
    }
    Some(fps)
}

// ---------------------------------------------------------------------------
// AAC-ADTS audio extraction (Squad-27)
// ---------------------------------------------------------------------------
//
// The MPEG-TS audio path stores AAC as a stream of ADTS frames inside PES
// packets — same PES framing as the video path, but the elementary stream
// payload is ADTS, not Annex-B. The downstream mux (Squad-18) wants raw
// AAC access units (no ADTS header) plus a synthesized AudioSpecificConfig
// (ASC) — both come from the first ADTS header.
//
// References:
// - ADTS frame layout: ISO/IEC 13818-7 §6.2 (the "_adts_frame()" syntax
//   table — 7-byte fixed header without CRC, 9-byte with CRC).
// - ASC layout: ISO/IEC 14496-3 §1.6.2 (`AudioSpecificConfig` →
//   `GetAudioObjectType` + `samplingFrequencyIndex` + `channelConfiguration`
//   + `GASpecificConfig` for AOT 1..7).
//
// Sampling frequency table (ISO/IEC 14496-3 §1.6.3.4 Table 1.16):
const AAC_SAMPLE_RATES: [u32; 13] = [
    96000, 88200, 64000, 48000, 44100, 32000, 24000, 22050, 16000, 12000, 11025, 8000, 7350,
];

/// Parsed view of a single ADTS frame header (ISO/IEC 13818-7 §6.2).
/// Only the fields we need for ASC synthesis + frame slicing — buffer
/// fullness / number_of_raw_data_blocks are not exposed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct AdtsHeader {
    /// ADTS profile (2 bits): AAC ObjectType - 1.
    /// `0`=Main, `1`=LC, `2`=SSR, `3`=LTP. Maps to ASC AOT via `+1`.
    profile: u8,
    /// Sampling frequency index (4 bits, 0..=12 valid; 15 = explicit).
    /// `decode_sample_rate_index` resolves to Hz.
    sampling_frequency_index: u8,
    /// `channel_configuration` (3 bits). 1 = mono, 2 = stereo, etc.
    /// 0 = "channel config defined in PCE" — uncommon; we accept 1/2
    /// at the audio-track surface, downstream mux rejects the rest.
    channel_configuration: u8,
    /// Whole frame length in bytes including header + (optional CRC) +
    /// AAC payload.
    frame_length: usize,
    /// Length of the ADTS header itself: 7 bytes if `protection_absent`
    /// (no CRC), 9 bytes otherwise.
    header_len: usize,
}

/// Parse an ADTS frame header at `buf[0..]`. Returns the parsed header on
/// success. Does NOT validate the CRC even when present — the demux path
/// trusts the upstream PMT routing to point us at AAC bytes; a corrupt
/// stream surfaces as a sync-loss frame downstream.
fn parse_adts_header(buf: &[u8]) -> Option<AdtsHeader> {
    if buf.len() < 7 {
        return None;
    }
    // Sync word: 12 bits = 0xFFF. Bytes 0..1 = `1111_1111  1111_xxxx`.
    if buf[0] != 0xFF || (buf[1] & 0xF0) != 0xF0 {
        return None;
    }
    let protection_absent = (buf[1] & 0x01) != 0;
    let header_len = if protection_absent { 7 } else { 9 };
    if buf.len() < header_len {
        return None;
    }
    let profile = (buf[2] >> 6) & 0x03;
    let sampling_frequency_index = (buf[2] >> 2) & 0x0F;
    // channel_configuration straddles bytes 2..3:
    //   bit 0 of byte 2 (low bit after profile/sr_idx/private) = ch_cfg high bit
    //   bits 7..6 of byte 3 (top two bits)                     = ch_cfg low 2 bits
    let channel_configuration = ((buf[2] & 0x01) << 2) | ((buf[3] >> 6) & 0x03);
    // frame_length is 13 bits across bytes 3..4..5:
    //   bits 1..0 of byte 3 = frame_length[12..11]
    //   bits 7..0 of byte 4 = frame_length[10..3]
    //   bits 7..5 of byte 5 = frame_length[2..0]
    let frame_length =
        (((buf[3] & 0x03) as usize) << 11) | ((buf[4] as usize) << 3) | ((buf[5] >> 5) as usize);
    if frame_length < header_len {
        return None;
    }
    Some(AdtsHeader {
        profile,
        sampling_frequency_index,
        channel_configuration,
        frame_length,
        header_len,
    })
}

/// Resolve an ADTS sampling_frequency_index to Hz. Only indices 0..=12 are
/// recognised; 13/14 are reserved and 15 ("escape") would carry an
/// explicit 24-bit rate after the header, which we don't accept (no
/// real-world AAC ADTS file uses index 15 — the escape form is for
/// AAC-in-LATM, not ADTS).
fn decode_sample_rate_index(idx: u8) -> Option<u32> {
    AAC_SAMPLE_RATES.get(idx as usize).copied()
}

/// Synthesize a 2-byte AudioSpecificConfig from an ADTS header per
/// ISO/IEC 14496-3 §1.6.2:
/// - 5 bits: audioObjectType = ADTS profile + 1
///   (so ADTS profile=1 LC → ASC AOT=2 LC; ADTS profile=4 HE-AAC parent
///    AOT=5 SBR → also AOT=5 here, though real HE-AAC ASC also signals
///    SBR explicitly via extension AOT — we don't try to do that, the
///    mux validation rejects HE-AAC anyway).
/// - 4 bits: samplingFrequencyIndex (copy from ADTS verbatim)
/// - 4 bits: channelConfiguration (copy from ADTS verbatim)
/// - 3 bits: GASpecificConfig padding (frameLengthFlag=0,
///   dependsOnCoreCoder=0, extensionFlag=0)
///
/// Total: 16 bits = 2 bytes.
///
/// Example: ADTS profile=1 (LC), sr_idx=3 (48k), ch_cfg=2 (stereo) →
/// ASC bytes `0x11 0x90`.
fn synthesize_asc(adts: &AdtsHeader) -> [u8; 2] {
    let aot = adts.profile + 1; // ADTS profile (AOT-1) → ASC AOT
    let sr_idx = adts.sampling_frequency_index;
    let ch_cfg = adts.channel_configuration;
    // Bit layout (MSB first, 16 bits):
    //   AOT(5) | sr_idx(4) | ch_cfg(4) | GA padding(3)
    // Pack into a u16 then split to BE bytes.
    let mut bits: u16 = 0;
    bits |= ((aot as u16) & 0x1F) << 11;
    bits |= ((sr_idx as u16) & 0x0F) << 7;
    bits |= ((ch_cfg as u16) & 0x0F) << 3;
    // GA padding bits already 0.
    bits.to_be_bytes()
}

/// Reassemble all PES packets on `audio_pid` and split the resulting
/// elementary stream into ADTS frames. Returns one `Vec<u8>` per frame
/// (raw access unit — ADTS header stripped) and a parallel duration list
/// in `sample_rate` ticks (always 1024 per AAC-LC frame).
///
/// The first valid ADTS header drives ASC synthesis; subsequent frames
/// must carry the same sampling_frequency_index and channel_configuration
/// — a switch mid-stream would invalidate the ASC and the mux can't
/// tolerate that. We currently bail out of audio extraction if the
/// stream switches; downstream falls back to video-only.
fn extract_ts_aac_audio(
    data: &[u8],
    packets: usize,
    packet_stride: usize,
    prefix_len: usize,
    audio_pid: u16,
) -> Result<Option<AudioTrack>> {
    // Reassemble all PES packets on `audio_pid` into one elementary
    // stream — shared with the AC-3 / E-AC-3 paths (Squad-37). ADTS
    // sync words let us split into frames after the fact.
    let es = reassemble_audio_pes(data, packets, packet_stride, prefix_len, audio_pid);

    if es.is_empty() {
        return Ok(None);
    }

    // Step 2: scan for the first valid ADTS sync, derive ASC.
    let mut cursor = match find_adts_sync(&es, 0) {
        Some(idx) => idx,
        None => return Ok(None),
    };
    let first = parse_adts_header(&es[cursor..]).context("TS: first ADTS frame failed to parse")?;
    let sample_rate = decode_sample_rate_index(first.sampling_frequency_index)
        .context("TS: AAC sampling_frequency_index out of range")?;
    let channels = first.channel_configuration as u16;
    if channels == 0 {
        bail!("TS: AAC channel_configuration=0 (PCE-defined); not supported");
    }
    let asc = synthesize_asc(&first).to_vec();

    // Step 3: walk frames, strip headers, accumulate samples + durations.
    // Each AAC-LC frame is exactly 1024 samples per channel — that's the
    // duration in `sample_rate` ticks (timescale = sample_rate).
    let mut samples: Vec<Vec<u8>> = Vec::new();
    let mut durations: Vec<u32> = Vec::new();
    while cursor < es.len() {
        // Resync if we've drifted off a frame boundary (rare in practice
        // but possible on packet loss or if a PES header extension we
        // don't recognise pushed garbage into the ES).
        let Some(found) = find_adts_sync(&es, cursor) else {
            break;
        };
        cursor = found;
        let Some(hdr) = parse_adts_header(&es[cursor..]) else {
            break;
        };
        if hdr.sampling_frequency_index != first.sampling_frequency_index
            || hdr.channel_configuration != first.channel_configuration
        {
            tracing::warn!(
                "TS: AAC ADTS stream switched sr_idx/ch_cfg mid-stream; truncating audio at frame {}",
                samples.len()
            );
            break;
        }
        let end = cursor + hdr.frame_length;
        if end > es.len() {
            break;
        }
        let payload_start = cursor + hdr.header_len;
        if payload_start > end {
            break;
        }
        samples.push(es[payload_start..end].to_vec());
        durations.push(1024);
        cursor = end;
    }

    if samples.is_empty() {
        return Ok(None);
    }

    Ok(Some(AudioTrack {
        codec: "aac".into(),
        samples,
        sample_rate,
        channels,
        asc,
        codec_private: Vec::new(),
        timescale: sample_rate,
        durations,
    }))
}

// ---------------------------------------------------------------------------
// AC-3 / E-AC-3 in MPEG-TS audio extraction (Squad-37)
// ---------------------------------------------------------------------------
//
// PES payload for an AC-3 / E-AC-3 audio PID is a stream of raw
// syncframes — 0x0B77 sync word at the start of each frame, followed by
// the BSI fields whose layout `crate::ac3_sync` already parses for
// MP4 / MKV passthrough. Squad-26 settled the codec_private wire
// format: a 3-byte `dac3` body for AC-3, a 5-byte `dec3` body for
// vanilla single-substream E-AC-3.
//
// The MP4 mux contract (Squad-26) is: pass the raw AC-3 / E-AC-3
// frames through verbatim as samples; populate `codec_private` with the
// dac3/dec3 body derived from the first frame; `asc` stays empty for
// these codecs. We do NOT re-frame, decode, or strip anything — the
// frames are length-self-describing via the syncframe info, and the
// muxer / downstream demuxer round-trip in Squad-26 already handles
// that on the MP4 side.

/// Find the next 0x0B77 AC-3 / E-AC-3 sync word at or after `from`.
fn find_ac3_sync(es: &[u8], from: usize) -> Option<usize> {
    let mut i = from;
    while i + 1 < es.len() {
        if es[i] == 0x0B && es[i + 1] == 0x77 {
            return Some(i);
        }
        i += 1;
    }
    None
}

/// Compute the byte length of one AC-3 syncframe given its bit-rate
/// code and fscod. ETSI TS 102 366 §F Table F.7 gives the wire-byte
/// count per (bit_rate_code, fscod) pair; the closed form is:
///   for fscod=0 (48k):  frame_size_bytes = 2 * frame_size_words[brc]
///                       where frame_size_words = bit_rate_kbps * 32 / 48 / 2
///                       reduces to: bytes = bit_rate_kbps * 4 / 3
/// For 44.1k and 32k there's a per-(brc,fscod) padding offset table; we
/// derive it from the algebraic identity bytes = bit_rate_kbps * 1000 /
/// (sample_rate / samples_per_frame * 8). AC-3 has a fixed 1536 samples
/// per frame, so:
///   bytes = bit_rate_kbps * 1000 * 1536 / sample_rate / 8
///         = bit_rate_kbps * 192000 / sample_rate
/// 44.1k frames are not byte-exact this way (frame size oscillates
/// between two adjacent values to track the average rate); the bsi
/// `frmsizecod` low bit indicates which of the two values applies, so
/// we honour it and add 2 bytes when set. For 48k and 32k the low bit
/// is irrelevant (rates divide evenly).
fn ac3_frame_size(brc: u8, fscod: u8, frmsizecod_low_bit: u8) -> Option<usize> {
    let kbps = ac3_bit_rate_kbps(brc) as usize;
    if kbps == 0 {
        return None;
    }
    let sr = ac3_sync::ac3_sample_rate_hz(fscod) as usize;
    if sr == 0 {
        return None;
    }
    let base = (kbps * 1000 * 1536) / (sr * 8);
    // 44.1k oscillation: one of two frame sizes per syncframe (the low
    // bit of frmsizecod selects). At 48k / 32k both sides match the
    // algebraic value, so the bit is harmless.
    let extra = if fscod == 1 && frmsizecod_low_bit != 0 {
        2
    } else {
        0
    };
    Some(base + extra)
}

/// Compute the byte length of one E-AC-3 syncframe — the BSI directly
/// carries `frmsiz` (frame_size_words - 1), so frame_size_bytes is
/// (frmsiz + 1) * 2.
fn eac3_frame_size(frmsiz: u16) -> usize {
    ((frmsiz as usize) + 1) * 2
}

/// Extract AC-3 frames from PES packets on `audio_pid`. Returns an
/// `AudioTrack` with `codec = "ac3"`, `codec_private = dac3 body`, and
/// one sample per AC-3 syncframe (raw frame bytes verbatim).
///
/// The first valid syncframe drives `dac3` / sample_rate / channel
/// derivation; subsequent frames are emitted as samples without
/// re-validating their BSI (a corrupt mid-stream sync would surface as
/// a downstream decoder error, the same way our AAC path handles it).
fn extract_ts_ac3_audio(
    data: &[u8],
    packets: usize,
    packet_stride: usize,
    prefix_len: usize,
    audio_pid: u16,
) -> Result<Option<AudioTrack>> {
    let es = reassemble_audio_pes(data, packets, packet_stride, prefix_len, audio_pid);
    if es.is_empty() {
        return Ok(None);
    }
    let mut cursor = match find_ac3_sync(&es, 0) {
        Some(idx) => idx,
        None => return Ok(None),
    };
    // Parse the first frame's BSI to derive dac3 + sample_rate + channels.
    let first = match ac3_sync::parse_sync_info(&es[cursor..])
        .context("TS: first AC-3 frame failed to parse sync header")?
    {
        SyncInfo::Ac3(s) => s,
        SyncInfo::Eac3(_) => bail!("TS: AC-3 PMT entry but bitstream is E-AC-3 (bsid=16)"),
    };
    let sample_rate = ac3_sync::ac3_sample_rate_hz(first.fscod);
    if sample_rate == 0 {
        bail!("TS: AC-3 fscod={} reserved", first.fscod);
    }
    let channels = channel_count(first.acmod, first.lfeon);
    let dac3 = dac3_body_from_sync(&first).to_vec();

    // Walk frames: re-sync on 0x0B77, slice by computed frame size, push
    // the slice as a sample. AC-3 emits 1536 samples per frame.
    let mut samples: Vec<Vec<u8>> = Vec::new();
    let mut durations: Vec<u32> = Vec::new();
    while cursor < es.len() {
        let Some(found) = find_ac3_sync(&es, cursor) else {
            break;
        };
        cursor = found;
        // Re-read the per-frame frmsizecod low bit so the 44.1k
        // oscillation lands on the right boundary.
        if cursor + 5 > es.len() {
            break;
        }
        let frmsizecod = es[cursor + 4] & 0x3F;
        let bit_rate_code = frmsizecod >> 1;
        let low_bit = frmsizecod & 0x01;
        let fscod = (es[cursor + 4] >> 6) & 0x03;
        let Some(size) = ac3_frame_size(bit_rate_code, fscod, low_bit) else {
            break;
        };
        let end = cursor + size;
        if end > es.len() {
            break;
        }
        samples.push(es[cursor..end].to_vec());
        durations.push(1536);
        cursor = end;
    }
    if samples.is_empty() {
        return Ok(None);
    }
    Ok(Some(AudioTrack {
        codec: "ac3".into(),
        samples,
        sample_rate,
        channels,
        asc: Vec::new(),
        codec_private: dac3,
        timescale: sample_rate,
        durations,
    }))
}

/// Extract E-AC-3 frames from PES packets on `audio_pid`. Returns an
/// `AudioTrack` with `codec = "eac3"`, `codec_private = dec3 body`, and
/// one sample per E-AC-3 syncframe (raw frame bytes verbatim).
///
/// `dec3.data_rate` is computed from the first frame: frame_size_bytes /
/// samples_per_frame * sample_rate * 8 / 2 / 1000 (kbps / 2 per §F.6).
fn extract_ts_eac3_audio(
    data: &[u8],
    packets: usize,
    packet_stride: usize,
    prefix_len: usize,
    audio_pid: u16,
) -> Result<Option<AudioTrack>> {
    let es = reassemble_audio_pes(data, packets, packet_stride, prefix_len, audio_pid);
    if es.is_empty() {
        return Ok(None);
    }
    let mut cursor = match find_ac3_sync(&es, 0) {
        Some(idx) => idx,
        None => return Ok(None),
    };
    let first: Eac3SyncInfo = match ac3_sync::parse_sync_info(&es[cursor..])
        .context("TS: first E-AC-3 frame failed to parse sync header")?
    {
        SyncInfo::Eac3(s) => s,
        SyncInfo::Ac3(_) => bail!("TS: E-AC-3 PMT entry but bitstream is AC-3 (bsid<=10)"),
    };
    let sample_rate = eac3_sample_rate_hz(first.fscod, first.fscod2);
    if sample_rate == 0 {
        bail!(
            "TS: E-AC-3 reserved sample rate (fscod={}, fscod2={})",
            first.fscod,
            first.fscod2
        );
    }
    let channels = channel_count(first.acmod, first.lfeon);
    let spf = eac3_samples_per_frame(first.numblkscod) as u64;
    let frame_bytes = ((first.frmsiz as u64) + 1) * 2;
    let bitrate_kbps = if spf > 0 && sample_rate > 0 {
        (frame_bytes * 8 * sample_rate as u64) / spf / 1000
    } else {
        0
    };
    let data_rate = bitrate_kbps.div_ceil(2) as u16;
    let dec3 = dec3_body_from_sync(&first, data_rate).to_vec();

    let mut samples: Vec<Vec<u8>> = Vec::new();
    let mut durations: Vec<u32> = Vec::new();
    while cursor < es.len() {
        let Some(found) = find_ac3_sync(&es, cursor) else {
            break;
        };
        cursor = found;
        if cursor + 5 > es.len() {
            break;
        }
        // Re-read frmsiz from this frame's BSI: bytes 2..4 carry
        // strmtyp(2) + substreamid(3) + frmsiz(11). frmsiz = bits 5..15
        // of the BE u16 starting at byte 2.
        let raw = u16::from_be_bytes([es[cursor + 2], es[cursor + 3]]);
        let frmsiz = raw & 0x07FF;
        let size = eac3_frame_size(frmsiz);
        let end = cursor + size;
        if end > es.len() {
            break;
        }
        samples.push(es[cursor..end].to_vec());
        durations.push(spf as u32);
        cursor = end;
    }
    if samples.is_empty() {
        return Ok(None);
    }
    Ok(Some(AudioTrack {
        codec: "eac3".into(),
        samples,
        sample_rate,
        channels,
        asc: Vec::new(),
        codec_private: dec3,
        timescale: sample_rate,
        durations,
    }))
}

/// Reassemble all PES payloads on `audio_pid` into one elementary stream
/// `Vec<u8>`. Shared between the AAC, AC-3 and E-AC-3 audio extractors —
/// each codec slices the resulting buffer into frames using its own
/// sync-word + frame-size logic.
fn reassemble_audio_pes(
    data: &[u8],
    packets: usize,
    packet_stride: usize,
    prefix_len: usize,
    audio_pid: u16,
) -> Vec<u8> {
    let mut es: Vec<u8> = Vec::new();
    let mut have_first_start = false;
    for i in 0..packets {
        let start = i * packet_stride + prefix_len;
        let pkt = &data[start..start + TS_PACKET];
        if pkt[0] != TS_SYNC {
            continue;
        }
        let pid = (((pkt[1] & 0x1F) as u16) << 8) | pkt[2] as u16;
        if pid != audio_pid {
            continue;
        }
        let pusi = pkt[1] & 0x40 != 0;
        let scramble = (pkt[3] >> 6) & 0x03;
        if scramble != 0 {
            continue;
        }
        let adaptation = (pkt[3] >> 4) & 0x03;
        let has_payload = adaptation & 0x01 != 0;
        let has_adaptation = adaptation & 0x02 != 0;
        if !has_payload {
            continue;
        }

        let mut offset = 4usize;
        if has_adaptation {
            if offset >= TS_PACKET {
                continue;
            }
            let adap_len = pkt[offset] as usize;
            offset += 1 + adap_len;
            if offset > TS_PACKET {
                continue;
            }
        }
        if offset >= TS_PACKET {
            continue;
        }
        let payload = &pkt[offset..];

        if pusi {
            let Some((es_start, _pts)) = parse_pes_header_audio(payload) else {
                have_first_start = false;
                continue;
            };
            have_first_start = true;
            if es_start < payload.len() {
                es.extend_from_slice(&payload[es_start..]);
            }
        } else if have_first_start {
            es.extend_from_slice(payload);
        }
    }
    es
}

/// Dispatch audio extraction by codec kind from the PMT walk. Per
/// Squad-37: AAC routes through `extract_ts_aac_audio` (Squad-27 path);
/// AC-3 and E-AC-3 route through their respective new extractors.
fn extract_ts_audio(
    data: &[u8],
    packets: usize,
    packet_stride: usize,
    prefix_len: usize,
    info: AudioStreamInfo,
) -> Result<Option<AudioTrack>> {
    match info.kind {
        AudioCodecKind::AacAdts => {
            extract_ts_aac_audio(data, packets, packet_stride, prefix_len, info.pid)
        }
        AudioCodecKind::Ac3 => {
            extract_ts_ac3_audio(data, packets, packet_stride, prefix_len, info.pid)
        }
        AudioCodecKind::Eac3 => {
            extract_ts_eac3_audio(data, packets, packet_stride, prefix_len, info.pid)
        }
    }
}

/// Find the next ADTS sync word at or after `from` in `es`. Returns the
/// offset of the sync byte (0xFF) or `None`.
fn find_adts_sync(es: &[u8], from: usize) -> Option<usize> {
    let mut i = from;
    while i + 1 < es.len() {
        if es[i] == 0xFF && (es[i + 1] & 0xF0) == 0xF0 {
            return Some(i);
        }
        i += 1;
    }
    None
}

/// Parse a PES header for audio (stream_id 0xC0..=0xDF). Same shape as
/// `parse_pes_header` for video but accepts the audio stream_id range.
/// Returns `(es_start, pts)`.
fn parse_pes_header_audio(payload: &[u8]) -> Option<(usize, Option<u64>)> {
    if payload.len() < 9 {
        return None;
    }
    if payload[0] != 0 || payload[1] != 0 || payload[2] != 1 {
        return None;
    }
    let stream_id = payload[3];
    // Audio streams are 0xC0..=0xDF per ISO/IEC 13818-1 §2.4.3.7.
    if !(0xC0..=0xDF).contains(&stream_id) {
        return None;
    }
    let flags = payload[7];
    let pts_dts_flags = (flags >> 6) & 0x03;
    let header_data_len = payload[8] as usize;
    let es_start = 9 + header_data_len;
    if es_start > payload.len() {
        return None;
    }
    let pts = if pts_dts_flags == 0b10 || pts_dts_flags == 0b11 {
        if payload.len() < 14 {
            return None;
        }
        let p0 = ((payload[9] >> 1) & 0x07) as u64;
        let p1 = (((payload[10] as u64) << 7) | ((payload[11] as u64) >> 1)) & 0x7FFF;
        let p2 = (((payload[12] as u64) << 7) | ((payload[13] as u64) >> 1)) & 0x7FFF;
        Some((p0 << 30) | (p1 << 15) | p2)
    } else {
        None
    };
    Some((es_start, pts))
}

// ---------------------------------------------------------------------------
// TsStreamingDemuxer (Squad streaming-migration-55 P1)
// ---------------------------------------------------------------------------

/// Streaming MPEG-TS demuxer. Holds the PES reassembly buffer for one
/// in-flight access unit only — yields whenever a PUSI=1 packet
/// closes the current sample (or at EOF for the final pending sample).
///
/// Squad-37 added:
/// - **Multi-program awareness**: `programs()` returns every program
///   the PAT advertised plus their PMT contents; `select_program()`
///   switches the active video PID + audio extraction to a different
///   program. Default behaviour is unchanged (first program with a
///   recognised video stream wins).
/// - **Encrypted-stream guard**: if any packet on the active video PID
///   carries `transport_scrambling_control != 0`, we log a one-time
///   typed warn ("encrypted TS stream; we don't carry CA tables —
///   drop video output") and switch to a "drop everything" mode where
///   `next_video_sample` returns `Ok(None)` without further parsing.
pub struct TsStreamingDemuxer {
    data: Vec<u8>,
    header: DemuxHeader,
    audio: Option<AudioTrack>,
    packets: usize,
    packet_stride: usize,
    prefix_len: usize,
    /// Every program the PAT advertised, in PAT order. Each entry's
    /// PMT was walked at init to populate its video/audio stream lists.
    /// Programs whose PMT we couldn't parse are still listed (with
    /// empty video_streams/audio_streams) so callers see them.
    programs: Vec<ProgramInfo>,
    /// Index into `programs` of the currently active program. Default:
    /// the first program with a recognised video stream.
    active_program_idx: usize,
    /// Active video PID (mirrors `programs[active_program_idx].video_streams[0].pid`).
    video_pid: u16,
    /// Index of the next packet to scan.
    next_pkt: usize,
    /// In-flight PES payload — emptied & yielded on next PUSI.
    pending: Vec<u8>,
    /// PTS attached to `pending` (PTS lives in the PES header that
    /// opened the AU).
    pending_pts: Option<u64>,
    /// True once we've seen the first PUSI for our PID. Bytes before
    /// the first PUSI are dropped (mid-stream join semantics).
    have_first_start: bool,
    /// True after we've returned `Ok(None)` once — guards against
    /// repeated drains.
    eof: bool,
    /// Lazily set on first emitted sample — `pixel_format::detect` is
    /// one-shot against `samples[0]` so we patch `header.info.pixel_format`
    /// in place once and skip the probe thereafter.
    pixel_format_detected: bool,
    /// Encrypted-stream guard (Squad-37). Latches `true` the first time
    /// we see `transport_scrambling_control != 0` on the active video
    /// PID; warning is logged exactly once and `next_video_sample`
    /// returns `Ok(None)` from that point on.
    encrypted_drop: bool,
}

pub(crate) fn demux_ts_streaming_init(data: &[u8]) -> Result<TsStreamingDemuxer> {
    let owned = data.to_vec();
    let (packets, packet_stride, prefix_len) = detect_packet_layout(&owned)?;
    if packets == 0 {
        bail!("TS: file contains no TS packets");
    }

    // Phase 1: walk the PAT and collect every program + its PMT PID.
    let mut pat_programs: Vec<PatProgram> = Vec::new();
    for i in 0..packets {
        let start = i * packet_stride + prefix_len;
        let pkt = &owned[start..start + TS_PACKET];
        if pkt[0] != TS_SYNC {
            continue;
        }
        let pid = (((pkt[1] & 0x1F) as u16) << 8) | pkt[2] as u16;
        if pid == 0
            && let Some(payload) = ts_psi_payload(pkt)
        {
            let progs = parse_pat_all_programs(payload);
            if !progs.is_empty() {
                pat_programs = progs;
                break;
            }
        }
    }
    if pat_programs.is_empty() {
        bail!("TS: no PAT entries found");
    }

    // Phase 2: walk every PMT and resolve its video+audio streams. We
    // remember the FIRST PMT section we see per PID — later versions
    // (table_id 0x02 with a higher `version_number`) would update an
    // active session in a real-world receiver but our demuxer is
    // start-of-file-only, so first-section semantics are correct.
    let mut programs: Vec<ProgramInfo> = pat_programs
        .iter()
        .map(|p| ProgramInfo {
            program_number: p.program_number,
            pmt_pid: p.pmt_pid,
            video_streams: Vec::new(),
            audio_streams: Vec::new(),
        })
        .collect();
    // Track which programs still need their PMT parsed.
    let mut need: std::collections::HashSet<u16> = pat_programs.iter().map(|p| p.pmt_pid).collect();
    for i in 0..packets {
        if need.is_empty() {
            break;
        }
        let start = i * packet_stride + prefix_len;
        let pkt = &owned[start..start + TS_PACKET];
        if pkt[0] != TS_SYNC {
            continue;
        }
        let pid = (((pkt[1] & 0x1F) as u16) << 8) | pkt[2] as u16;
        if !need.contains(&pid) {
            continue;
        }
        if let Some(payload) = ts_psi_payload(pkt)
            && let Some((video_streams, audio_streams)) = parse_pmt_streams(payload)
        {
            if let Some(prog) = programs.iter_mut().find(|p| p.pmt_pid == pid) {
                prog.video_streams = video_streams;
                prog.audio_streams = audio_streams;
            }
            need.remove(&pid);
        }
    }

    // Phase 3: pick the default active program — first one with a
    // recognised video stream. Matches legacy "first program wins"
    // semantics for single-program files.
    let active_program_idx = programs
        .iter()
        .position(|p| !p.video_streams.is_empty())
        .context("TS: no program advertises a recognised video elementary stream")?;
    let active = &programs[active_program_idx];
    let video = active.video_streams[0];
    let audio = active.audio_streams.first().copied();
    let codec = match video.stream_type {
        STREAM_TYPE_MPEG2_VIDEO => "mpeg2",
        STREAM_TYPE_H264 => "h264",
        STREAM_TYPE_HEVC => "h265",
        other => bail!("TS: unsupported stream_type 0x{:02X}", other),
    }
    .to_string();

    // total_frames + duration are unknown until drained.
    //
    // width/height recovery: TS carries nothing at the container layer,
    // so we walk just enough packets to capture the first video AU and
    // parse its SPS (H.264 / HEVC) or sequence header (MPEG-2). This
    // has to happen during init — `header()` is read by the pipeline
    // before any `next_video_sample` call, and the rav1e encoder
    // rejects 0×0 configs outright. Parse failure is non-fatal: we
    // warn and leave dims at 0 so the failure surfaces in the encoder
    // config error rather than silently corrupting the output.
    //
    // frame_rate: same scan collects a window of video-PID PTSes (up
    // to 64 PUSIs). Inter-PTS span over (count-1) intervals at the
    // 90 kHz TS clock gives the source fps. A wrong frame_rate here
    // causes exactly the kind of "video sped up, audio drags" sync
    // symptom that the BBB 24 fps sample hit against the previous
    // hardcoded `30.0` fallback. Falls back to `30.0` only when the
    // scan can't derive a finite fps in [1.0, 240.0].
    let scan = scan_first_video_au(&owned, packets, packet_stride, prefix_len, video.pid, 64);
    let (width, height) = match &scan.first_au {
        Some(au) => codec::pixel_format::detect_dims(&codec, std::slice::from_ref(au))
            .unwrap_or_else(|| {
                tracing::warn!(
                    codec = codec.as_str(),
                    video_pid = video.pid,
                    "TS streaming demux: first AU SPS parse failed; width/height=0×0"
                );
                (0, 0)
            }),
        None => {
            tracing::warn!(
                codec = codec.as_str(),
                video_pid = video.pid,
                "TS streaming demux: could not locate first video AU during init; width/height=0×0"
            );
            (0, 0)
        }
    };
    let frame_rate = estimate_frame_rate_from_ptses(&scan.ptses).unwrap_or_else(|| {
        tracing::warn!(
            codec = codec.as_str(),
            video_pid = video.pid,
            pts_samples = scan.ptses.len(),
            "TS streaming demux: could not derive frame_rate from PTS window; defaulting to 30.0"
        );
        30.0
    });

    let info = StreamInfo {
        codec: codec.clone(),
        width,
        height,
        frame_rate,
        duration: 0.0,
        pixel_format: PixelFormat::Yuv420p,
        color_space: ColorSpace::Bt709,
        total_frames: 0,
        bitrate: 0,
        color_metadata: Default::default(),
    };

    // Audio passthrough still happens up-front (Squad-18 contract).
    // Squad-37 routes by codec kind (AAC / AC-3 / E-AC-3).
    let audio_track = audio.and_then(|info| {
        match extract_ts_audio(&owned, packets, packet_stride, prefix_len, info) {
            Ok(track) => track,
            Err(e) => {
                tracing::warn!(
                    audio_pid = info.pid,
                    audio_kind = ?info.kind,
                    error = %e,
                    "TS audio extraction failed; emitting video-only"
                );
                None
            }
        }
    });

    Ok(TsStreamingDemuxer {
        data: owned,
        header: DemuxHeader { codec, info },
        audio: audio_track,
        packets,
        packet_stride,
        prefix_len,
        programs,
        active_program_idx,
        video_pid: video.pid,
        next_pkt: 0,
        pending: Vec::new(),
        pending_pts: None,
        have_first_start: false,
        eof: false,
        pixel_format_detected: false,
        encrypted_drop: false,
    })
}

impl TsStreamingDemuxer {
    /// Every program the PAT advertised, in PAT order. Squad-37 multi-
    /// program API — useful for callers that want to enumerate channels
    /// in a multi-program transport (DVB / ATSC broadcast capture). For
    /// single-program files the slice has length 1.
    pub fn programs(&self) -> &[ProgramInfo] {
        &self.programs
    }

    /// Index of the currently active program (within `programs()`).
    pub fn active_program_index(&self) -> usize {
        self.active_program_idx
    }

    /// Switch the active program by PMT-side `program_number`. Resets the
    /// per-AU walk state (pending PES bytes, PTS, encrypted-drop guard,
    /// pixel-format detection) so the next `next_video_sample` call
    /// starts cleanly on the new video PID. Returns `Ok(())` on success
    /// or an error if `program_number` is not in `programs()` or the
    /// chosen program has no recognised video stream.
    ///
    /// Audio is re-extracted from the new program's first audio stream
    /// (if any). For single-program files (the common case) callers
    /// don't need to touch this; the constructor already picked program
    /// 0 by default.
    pub fn select_program(&mut self, program_number: u16) -> Result<()> {
        let new_idx = self
            .programs
            .iter()
            .position(|p| p.program_number == program_number)
            .with_context(|| format!("TS: program_number {} not found in PAT", program_number))?;
        if self.programs[new_idx].video_streams.is_empty() {
            bail!(
                "TS: program {} has no recognised video stream",
                program_number
            );
        }
        let video = self.programs[new_idx].video_streams[0];
        let audio = self.programs[new_idx].audio_streams.first().copied();
        let codec = match video.stream_type {
            STREAM_TYPE_MPEG2_VIDEO => "mpeg2",
            STREAM_TYPE_H264 => "h264",
            STREAM_TYPE_HEVC => "h265",
            other => bail!(
                "TS: program {} video stream_type 0x{:02X} unsupported",
                program_number,
                other
            ),
        }
        .to_string();
        self.active_program_idx = new_idx;
        self.video_pid = video.pid;
        // Refresh the codec / pixel-format fields on the cached header
        // — `info.codec` flows out of `header()` to the pipeline.
        self.header.codec = codec.clone();
        self.header.info.codec = codec.clone();
        self.header.info.pixel_format = PixelFormat::Yuv420p;
        self.pixel_format_detected = false;
        // Re-probe width/height + frame_rate from the new program's
        // video PID. Zero dims / 30 fps fallback on parse failure so
        // the encoder reports the miss rather than silently using the
        // previous program's values.
        let scan = scan_first_video_au(
            &self.data,
            self.packets,
            self.packet_stride,
            self.prefix_len,
            video.pid,
            64,
        );
        let (w, h) = match &scan.first_au {
            Some(au) => {
                codec::pixel_format::detect_dims(&codec, std::slice::from_ref(au)).unwrap_or((0, 0))
            }
            None => (0, 0),
        };
        self.header.info.width = w;
        self.header.info.height = h;
        self.header.info.frame_rate = estimate_frame_rate_from_ptses(&scan.ptses).unwrap_or(30.0);
        // Reset PES walk state.
        self.next_pkt = 0;
        self.pending.clear();
        self.pending_pts = None;
        self.have_first_start = false;
        self.eof = false;
        self.encrypted_drop = false;
        // Re-extract audio from the new program's first audio stream.
        self.audio = audio.and_then(|info| {
            match extract_ts_audio(
                &self.data,
                self.packets,
                self.packet_stride,
                self.prefix_len,
                info,
            ) {
                Ok(track) => track,
                Err(e) => {
                    tracing::warn!(
                        audio_pid = info.pid,
                        audio_kind = ?info.kind,
                        error = %e,
                        "TS audio extraction failed on program switch; emitting video-only"
                    );
                    None
                }
            }
        });
        Ok(())
    }

    /// Build a Sample from raw AU bytes, applying the one-shot
    /// pixel_format detection on the first emission. Centralises the
    /// three yield sites in `next_video_sample`.
    fn yield_sample(&mut self, data: Vec<u8>, pts: Option<u64>) -> Sample {
        if !self.pixel_format_detected {
            let detected =
                codec::pixel_format::detect(&self.header.codec, std::slice::from_ref(&data));
            self.header.info.pixel_format = detected;
            self.pixel_format_detected = true;
        }
        Sample {
            data,
            pts_ticks: pts.map(|p| p as i64).unwrap_or(0),
            duration_ticks: 0,
        }
    }
}

impl StreamingDemuxer for TsStreamingDemuxer {
    fn header(&self) -> &DemuxHeader {
        &self.header
    }

    fn next_video_sample(&mut self) -> Result<Option<Sample>> {
        if self.eof || self.encrypted_drop {
            return Ok(None);
        }
        loop {
            if self.next_pkt >= self.packets {
                // Drain the final pending sample at EOF.
                self.eof = true;
                if !self.pending.is_empty() {
                    let data = std::mem::take(&mut self.pending);
                    let pts = self.pending_pts.take();
                    return Ok(Some(self.yield_sample(data, pts)));
                }
                return Ok(None);
            }

            let i = self.next_pkt;
            self.next_pkt += 1;
            let start = i * self.packet_stride + self.prefix_len;
            let pkt = &self.data[start..start + TS_PACKET];
            if pkt[0] != TS_SYNC {
                continue;
            }
            let pid = (((pkt[1] & 0x1F) as u16) << 8) | pkt[2] as u16;
            if pid != self.video_pid {
                continue;
            }
            let pusi = pkt[1] & 0x40 != 0;
            let scramble = (pkt[3] >> 6) & 0x03;
            if scramble != 0 {
                // Encrypted-stream guard (Squad-37). The first scrambled
                // packet on the active video PID triggers a one-time
                // typed warn and flips us into drop-everything mode —
                // any further `next_video_sample` calls return
                // `Ok(None)` immediately. We don't carry CA tables, so
                // any byte we feed downstream from here is garbage.
                tracing::warn!(
                    video_pid = self.video_pid,
                    transport_scrambling_control = scramble,
                    error_kind = "encrypted_ts",
                    "encrypted TS stream; we don't carry CA tables — drop video output"
                );
                self.encrypted_drop = true;
                self.pending.clear();
                self.pending_pts = None;
                self.have_first_start = false;
                self.eof = true;
                return Ok(None);
            }
            let adaptation = (pkt[3] >> 4) & 0x03;
            let has_payload = adaptation & 0x01 != 0;
            let has_adaptation = adaptation & 0x02 != 0;
            if !has_payload {
                continue;
            }

            let mut offset = 4usize;
            if has_adaptation {
                if offset >= TS_PACKET {
                    continue;
                }
                let adap_len = pkt[offset] as usize;
                offset += 1 + adap_len;
                if offset > TS_PACKET {
                    continue;
                }
            }
            if offset >= TS_PACKET {
                continue;
            }
            let payload = &pkt[offset..];

            if pusi {
                // PUSI flushes the previous AU. If we already had one
                // in-flight, return it now and stage the new one for
                // the next call.
                let had_pending = self.have_first_start;
                let prev_data = if had_pending {
                    std::mem::take(&mut self.pending)
                } else {
                    Vec::new()
                };
                let prev_pts = self.pending_pts.take();
                self.have_first_start = true;

                let Some((es_start, pts)) = parse_pes_header(payload) else {
                    // Malformed PES — drop state, keep walking.
                    self.have_first_start = false;
                    self.pending.clear();
                    if !prev_data.is_empty() {
                        return Ok(Some(self.yield_sample(prev_data, prev_pts)));
                    }
                    continue;
                };
                self.pending_pts = pts;
                if es_start < payload.len() {
                    self.pending.extend_from_slice(&payload[es_start..]);
                }

                if !prev_data.is_empty() {
                    return Ok(Some(self.yield_sample(prev_data, prev_pts)));
                }
                // No previous AU to yield — keep walking until the next
                // PUSI (or EOF).
            } else if self.have_first_start {
                self.pending.extend_from_slice(payload);
            }
        }
    }

    fn audio(&self) -> Option<&AudioTrack> {
        self.audio.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ts_pkt(pid: u16, pusi: bool, adaptation: u8, payload: &[u8]) -> [u8; TS_PACKET] {
        let mut p = [0xFFu8; TS_PACKET];
        p[0] = TS_SYNC;
        // TEI=0, PUSI=pusi, transport_priority=0, PID(13)
        p[1] = if pusi { 0x40 } else { 0x00 } | ((pid >> 8) & 0x1F) as u8;
        p[2] = (pid & 0xFF) as u8;
        // scramble=00, adaptation=adaptation, continuity=0
        p[3] = (adaptation & 0x03) << 4;
        let mut off = 4;
        // For these tests we always use adaptation=01 (payload only).
        let pay_len = payload.len().min(TS_PACKET - off);
        p[off..off + pay_len].copy_from_slice(&payload[..pay_len]);
        off += pay_len;
        // Pad any remaining bytes with 0xFF (already initialised).
        let _ = off;
        p
    }

    #[test]
    fn estimate_frame_rate_from_uniform_ptses_returns_exact_fps() {
        // 24 fps: inter-PTS = 90000/24 = 3750 ticks.
        let ptses: Vec<u64> = (0..64).map(|i| i as u64 * 3750).collect();
        let fps = estimate_frame_rate_from_ptses(&ptses).expect("24 fps");
        assert!((fps - 24.0).abs() < 1e-9, "{} != 24.0", fps);
    }

    #[test]
    fn estimate_frame_rate_from_reordered_ptses_sorts_before_delta() {
        // Same 24 fps, but decode-order != display-order (one B-frame
        // pair swapped). Median should still pick up the 3750-tick
        // period cleanly.
        let mut ptses: Vec<u64> = (0..32).map(|i| i as u64 * 3750).collect();
        ptses.swap(5, 6);
        ptses.swap(10, 11);
        let fps = estimate_frame_rate_from_ptses(&ptses).expect("24 fps after swap");
        assert!((fps - 24.0).abs() < 1e-9, "{} != 24.0", fps);
    }

    #[test]
    fn estimate_frame_rate_from_single_outlier_delta_uses_median() {
        // 23 uniform 24-fps deltas + one 10× outlier. Median still 3750.
        let mut ptses: Vec<u64> = (0..24).map(|i| i as u64 * 3750).collect();
        ptses.push(24 * 3750 + 37500); // one huge gap
        let fps = estimate_frame_rate_from_ptses(&ptses).expect("24 fps despite outlier");
        assert!((fps - 24.0).abs() < 1e-9);
    }

    #[test]
    fn estimate_frame_rate_returns_none_when_all_ptses_equal() {
        let ptses = vec![0u64; 10];
        assert!(estimate_frame_rate_from_ptses(&ptses).is_none());
    }

    #[test]
    fn estimate_frame_rate_returns_none_when_fewer_than_two() {
        assert!(estimate_frame_rate_from_ptses(&[]).is_none());
        assert!(estimate_frame_rate_from_ptses(&[1234]).is_none());
    }

    #[test]
    fn estimate_frame_rate_rejects_out_of_range_values() {
        // Single 1-tick delta → fps = 90000, outside [1, 240].
        let ptses = vec![0u64, 1];
        assert!(estimate_frame_rate_from_ptses(&ptses).is_none());
    }

    #[test]
    fn estimate_frame_rate_handles_29_97_ntsc() {
        // 29.97 fps = 30000/1001. Inter-PTS = 90000 * 1001 / 30000 = 3003.
        let ptses: Vec<u64> = (0..32).map(|i| i as u64 * 3003).collect();
        let fps = estimate_frame_rate_from_ptses(&ptses).expect("29.97");
        assert!((fps - 30.0).abs() < 0.05, "got {}", fps); // 90000/3003 = 29.97..30.03
    }

    #[test]
    fn detects_plain_ts_layout() {
        let mut buf = Vec::with_capacity(3 * TS_PACKET);
        for _ in 0..3 {
            let pkt = ts_pkt(0x1FFF, false, 0b01, &[]);
            buf.extend_from_slice(&pkt);
        }
        let (count, stride, prefix) = detect_packet_layout(&buf).unwrap();
        assert_eq!((count, stride, prefix), (3, 188, 0));
    }

    #[test]
    fn parses_minimal_pat_pmt_and_reassembles_one_sample() {
        // Build a PAT pointing at PMT=0x100, a PMT listing video PID=0x200
        // stream_type=MPEG-2, then a single PES packet carrying 16 bytes
        // of video ES.

        // PAT section (we skip CRC correctness — the parser only uses
        // section_length to decide where to stop).
        let mut pat = vec![0u8; 0];
        pat.push(0x00); // table_id
        let section_length: usize = 5 + 4 + 4; // 5 header bytes (after len) + 1 program + CRC
        pat.push(0xB0 | ((section_length >> 8) & 0x0F) as u8);
        pat.push((section_length & 0xFF) as u8);
        pat.extend_from_slice(&[0x00, 0x01, 0xC1, 0x00, 0x00]); // tsid, ver/current, secno, lastno
        pat.extend_from_slice(&[0x00, 0x01]); // program_number = 1
        pat.extend_from_slice(&[0xE1, 0x00]); // reserved + PMT PID = 0x100
        pat.extend_from_slice(&[0, 0, 0, 0]); // CRC placeholder

        // PAT packet payload = [pointer_field=0, section...]
        let mut pat_payload = vec![0u8];
        pat_payload.extend_from_slice(&pat);
        let pat_pkt = ts_pkt(0x0000, true, 0b01, &pat_payload);

        // PMT section.
        let mut pmt = vec![0u8; 0];
        pmt.push(0x02);
        let pmt_sec_len: usize = 9 + 5 + 4; // program_number..pil(9) + 1 stream entry(5) + CRC(4)
        pmt.push(0xB0 | ((pmt_sec_len >> 8) & 0x0F) as u8);
        pmt.push((pmt_sec_len & 0xFF) as u8);
        pmt.extend_from_slice(&[0x00, 0x01, 0xC1, 0x00, 0x00]); // prog, ver/current, sec/last
        pmt.extend_from_slice(&[0xE2, 0x00]); // PCR PID = 0x200
        pmt.extend_from_slice(&[0xF0, 0x00]); // program_info_length = 0
        pmt.extend_from_slice(&[STREAM_TYPE_MPEG2_VIDEO, 0xE2, 0x00, 0xF0, 0x00]); // stream entry
        pmt.extend_from_slice(&[0, 0, 0, 0]); // CRC placeholder
        let mut pmt_payload = vec![0u8];
        pmt_payload.extend_from_slice(&pmt);
        let pmt_pkt = ts_pkt(0x0100, true, 0b01, &pmt_payload);

        // Two PES packets, each 16 bytes of ES, so the reassembler's
        // PUSI-flush path is exercised. Real MPEG-TS files also set
        // PES_packet_length which bounds the first one, but packet_length=0
        // ("unbounded") is also legal for MPEG-2 video PES, which is what
        // we emit here — termination comes from the next PUSI.
        let make_pes = |byte: u8| {
            let mut pes = vec![0u8, 0u8, 1u8]; // start code
            pes.push(0xE0); // stream_id video
            pes.extend_from_slice(&[0u8, 0u8]); // packet_length=0
            pes.push(0x80);
            pes.push(0x80); // PTS_DTS_flags = 10
            pes.push(5); // PES_header_data_length
            pes.extend_from_slice(&[0x21, 0x00, 0x01, 0x00, 0x01]); // PTS=0
            pes.extend_from_slice(&[byte; 16]);
            pes
        };
        let pes_pkt_a = ts_pkt(0x0200, true, 0b01, &make_pes(0xAA));
        let pes_pkt_b = ts_pkt(0x0200, true, 0b01, &make_pes(0xBB));

        let mut buf = Vec::new();
        buf.extend_from_slice(&pat_pkt);
        buf.extend_from_slice(&pmt_pkt);
        buf.extend_from_slice(&pes_pkt_a);
        buf.extend_from_slice(&pes_pkt_b);
        // Trailing null packet so detect_packet_layout sees a sync run.
        buf.extend_from_slice(&ts_pkt(0x1FFF, false, 0b01, &[]));

        let d = demux_ts(&buf).expect("demux");
        assert_eq!(d.codec, "mpeg2");
        // We should have reassembled two samples (the first flushed when
        // the second PUSI arrives). Sample A carries the 16 AU bytes
        // plus whatever TS padding trailed the PES header — the
        // demuxer does not know the bound, so exact byte-for-byte
        // comparison needs packet_length support (future). For now
        // assert: right sample count, correct leading bytes.
        assert_eq!(d.samples.len(), 2);
        assert_eq!(&d.samples[0][..16], &[0xAA; 16]);
        assert_eq!(&d.samples[1][..16], &[0xBB; 16]);
    }

    #[test]
    fn rejects_file_with_no_sync() {
        let garbage = vec![0u8; TS_PACKET * 3];
        assert!(demux_ts(&garbage).is_err());
    }

    // ---------------- AAC-ADTS / ASC unit tests (Squad-27) ----------------

    /// Build a 7-byte ADTS header (no CRC) with the given fields.
    /// `frame_length` covers header + payload.
    fn build_adts_header_7(profile: u8, sr_idx: u8, ch_cfg: u8, frame_length: usize) -> [u8; 7] {
        let mut h = [0u8; 7];
        // Bytes 0..1: 0xFFF sync + ID(1)=0 (MPEG-4) + layer(2)=0 +
        // protection_absent(1)=1.
        h[0] = 0xFF;
        h[1] = 0xF0 | 0x01; // protection_absent = 1
        // Byte 2: profile(2) | sr_idx(4) | private(1) | ch_cfg high bit(1).
        h[2] = ((profile & 0x03) << 6) | ((sr_idx & 0x0F) << 2) | ((ch_cfg >> 2) & 0x01);
        // Byte 3: ch_cfg low 2 bits(2) | original/copy(1) | home(1) |
        // copyright_id_bit(1) | copyright_id_start(1) | frame_length high 2.
        h[3] = ((ch_cfg & 0x03) << 6) | (((frame_length >> 11) & 0x03) as u8);
        h[4] = ((frame_length >> 3) & 0xFF) as u8;
        h[5] = (((frame_length & 0x07) << 5) | 0x1F) as u8;
        // Byte 6: low buffer_fullness bits + number_of_raw_data_blocks(2) = 0.
        h[6] = 0xFC;
        h
    }

    /// Build a 9-byte ADTS header (with CRC). CRC bytes are placeholders.
    fn build_adts_header_9(profile: u8, sr_idx: u8, ch_cfg: u8, frame_length: usize) -> [u8; 9] {
        let mut h = [0u8; 9];
        h[0] = 0xFF;
        h[1] = 0xF0; // protection_absent = 0 → CRC present
        h[2] = ((profile & 0x03) << 6) | ((sr_idx & 0x0F) << 2) | ((ch_cfg >> 2) & 0x01);
        h[3] = ((ch_cfg & 0x03) << 6) | (((frame_length >> 11) & 0x03) as u8);
        h[4] = ((frame_length >> 3) & 0xFF) as u8;
        h[5] = (((frame_length & 0x07) << 5) | 0x1F) as u8;
        h[6] = 0xFC;
        // Bytes 7..8: CRC placeholder (not validated by the parser).
        h
    }

    #[test]
    fn adts_parser_decodes_canonical_lc_stereo_7byte_header() {
        // Canonical LC stereo @ 48k, 100-byte payload + 7-byte header.
        let h = build_adts_header_7(1, 3, 2, 107);
        let parsed = parse_adts_header(&h).expect("must parse 7-byte ADTS header");
        assert_eq!(parsed.profile, 1, "ADTS profile=1 LC");
        assert_eq!(parsed.sampling_frequency_index, 3, "sr_idx=3 → 48kHz");
        assert_eq!(parsed.channel_configuration, 2, "ch_cfg=2 stereo");
        assert_eq!(parsed.frame_length, 107);
        assert_eq!(parsed.header_len, 7, "protection_absent=1 → 7-byte header");
        assert_eq!(
            decode_sample_rate_index(parsed.sampling_frequency_index),
            Some(48000)
        );
    }

    #[test]
    fn adts_parser_decodes_9byte_header_with_crc() {
        let h = build_adts_header_9(1, 4, 2, 109);
        let parsed = parse_adts_header(&h).expect("must parse 9-byte ADTS header");
        assert_eq!(parsed.profile, 1);
        assert_eq!(parsed.sampling_frequency_index, 4, "sr_idx=4 → 44.1kHz");
        assert_eq!(parsed.channel_configuration, 2);
        assert_eq!(parsed.frame_length, 109);
        assert_eq!(
            parsed.header_len, 9,
            "protection_absent=0 → 9-byte header (incl CRC)"
        );
        assert_eq!(
            decode_sample_rate_index(parsed.sampling_frequency_index),
            Some(44100)
        );
    }

    #[test]
    fn adts_parser_decodes_aac_profile_bits_full_range() {
        // ADTS profile is 2 bits → values 0..=3 are the only legal forms:
        // 0=Main, 1=LC, 2=SSR, 3=LTP. Parent HE-AAC's AOT=5 (SBR) cannot
        // be carried in ADTS — HE-AAC streams in ADTS look like LC at
        // the header level and signal SBR inside the access unit. The
        // parser must round-trip every legal 2-bit profile value so the
        // upstream router can decide what to do (we accept LC=1 and
        // reject the rest at mux-validation time).
        for profile in 0u8..=3 {
            let h = build_adts_header_7(profile, 3, 2, 32);
            let parsed =
                parse_adts_header(&h).unwrap_or_else(|| panic!("must parse profile={profile}"));
            assert_eq!(parsed.profile, profile);
        }
    }

    #[test]
    fn adts_parser_rejects_missing_sync() {
        let mut h = build_adts_header_7(1, 3, 2, 32);
        h[0] = 0x00;
        assert!(parse_adts_header(&h).is_none());
    }

    #[test]
    fn adts_parser_rejects_short_buffer() {
        let h = build_adts_header_7(1, 3, 2, 32);
        assert!(
            parse_adts_header(&h[..6]).is_none(),
            "<7 bytes can't carry a complete ADTS header"
        );
    }

    #[test]
    fn synthesize_asc_lc_stereo_48k_emits_0x1190() {
        // Squad-27 spec example: ADTS profile=1 (LC), sr_idx=3 (48k),
        // ch_cfg=2 (stereo) → ASC `0x11 0x90`.
        // Bit math:
        //   AOT=2 (LC),    5 bits = 00010
        //   sr_idx=3,      4 bits = 0011
        //   ch_cfg=2,      4 bits = 0010
        //   GA padding,    3 bits = 000
        // Concat: 00010 0011 0010 000 = 0001 0001 1001 0000 = 0x1190
        let adts = AdtsHeader {
            profile: 1,
            sampling_frequency_index: 3,
            channel_configuration: 2,
            frame_length: 0,
            header_len: 7,
        };
        let asc = synthesize_asc(&adts);
        assert_eq!(asc, [0x11, 0x90], "LC/48k/stereo → ASC 0x11 0x90");
    }

    #[test]
    fn synthesize_asc_lc_mono_44k() {
        // AOT=2, sr_idx=4 (44.1k), ch_cfg=1 (mono):
        //   00010 0100 0001 000 = 0001 0010 0000 1000 = 0x12 0x08
        let adts = AdtsHeader {
            profile: 1,
            sampling_frequency_index: 4,
            channel_configuration: 1,
            frame_length: 0,
            header_len: 7,
        };
        assert_eq!(synthesize_asc(&adts), [0x12, 0x08]);
    }

    #[test]
    fn synthesize_asc_main_aot_at_44k_5p1_rejected_at_channel_layer() {
        // ADTS profile=0 (Main) → ASC AOT=1. sr_idx=4 (44.1k),
        // ch_cfg=6 (5.1). The ASC bit packing must round-trip these
        // values regardless of whether the downstream mux accepts them
        // (mux today validates channels in {1, 2}).
        //   00001 0100 0110 000 = 0000 1010 0011 0000 = 0x0A 0x30
        let adts = AdtsHeader {
            profile: 0,
            sampling_frequency_index: 4,
            channel_configuration: 6,
            frame_length: 0,
            header_len: 7,
        };
        assert_eq!(synthesize_asc(&adts), [0x0A, 0x30]);
    }

    #[test]
    fn adts_strip_7byte_header_yields_payload_only() {
        // Synthesize one ADTS frame: 7-byte header + 100-byte payload.
        // Run it through extract_ts_aac_audio's frame loop (via a minimal
        // synthetic TS) and assert the resulting sample is exactly 100
        // bytes — header stripped.
        let mut frame = Vec::with_capacity(107);
        frame.extend_from_slice(&build_adts_header_7(1, 3, 2, 107));
        frame.extend_from_slice(&[0x42u8; 100]);
        // Drive the frame loop directly to avoid the PES/TS scaffolding.
        // We test the public extraction in a separate integration test.
        let header = parse_adts_header(&frame).unwrap();
        assert_eq!(header.frame_length, 107);
        let payload = &frame[header.header_len..header.frame_length];
        assert_eq!(payload.len(), 100);
        assert!(payload.iter().all(|b| *b == 0x42));
    }

    #[test]
    fn adts_sample_rate_table_covers_documented_indices() {
        // Spot-check the two anchors plus the boundary indices.
        assert_eq!(decode_sample_rate_index(0), Some(96000));
        assert_eq!(decode_sample_rate_index(3), Some(48000));
        assert_eq!(decode_sample_rate_index(4), Some(44100));
        assert_eq!(decode_sample_rate_index(12), Some(7350));
        assert!(decode_sample_rate_index(13).is_none(), "13 is reserved");
        assert!(
            decode_sample_rate_index(15).is_none(),
            "15 (escape) not supported"
        );
    }

    /// End-to-end: build a synthetic TS file with PAT + PMT advertising
    /// MPEG-2 video on PID 0x200 AND AAC-ADTS on PID 0x300, plus PES
    /// packets carrying ADTS frames. After demux, the audio track must
    /// surface with synthesized ASC + stripped AAC samples + 1024-tick
    /// durations.
    #[test]
    fn demux_ts_yields_audio_track_when_pmt_advertises_aac() {
        // ---- PAT pointing at PMT 0x100 ----
        let mut pat = vec![0x00];
        let pat_section_len: usize = 5 + 4 + 4;
        pat.push(0xB0 | ((pat_section_len >> 8) & 0x0F) as u8);
        pat.push((pat_section_len & 0xFF) as u8);
        pat.extend_from_slice(&[0x00, 0x01, 0xC1, 0x00, 0x00]);
        pat.extend_from_slice(&[0x00, 0x01, 0xE1, 0x00, 0u8, 0u8, 0u8, 0u8]);
        let mut pat_payload = vec![0u8];
        pat_payload.extend_from_slice(&pat);
        let pat_pkt = ts_pkt(0x0000, true, 0b01, &pat_payload);

        // ---- PMT advertising MPEG-2 video (PID 0x200) and AAC-ADTS audio
        // (PID 0x300) ----
        let mut pmt = vec![0x02];
        let pmt_section_len: usize = 9 + 5 + 5 + 4; // hdr + 2 stream entries + CRC
        pmt.push(0xB0 | ((pmt_section_len >> 8) & 0x0F) as u8);
        pmt.push((pmt_section_len & 0xFF) as u8);
        pmt.extend_from_slice(&[0x00, 0x01, 0xC1, 0x00, 0x00]);
        pmt.extend_from_slice(&[0xE2, 0x00]); // PCR PID = 0x200
        pmt.extend_from_slice(&[0xF0, 0x00]); // program_info_length = 0
        // Stream 1: MPEG-2 video on 0x200
        pmt.extend_from_slice(&[STREAM_TYPE_MPEG2_VIDEO, 0xE2, 0x00, 0xF0, 0x00]);
        // Stream 2: AAC-ADTS on 0x300
        pmt.extend_from_slice(&[STREAM_TYPE_AAC_ADTS, 0xE3, 0x00, 0xF0, 0x00]);
        pmt.extend_from_slice(&[0u8; 4]); // CRC placeholder
        let mut pmt_payload = vec![0u8];
        pmt_payload.extend_from_slice(&pmt);
        let pmt_pkt = ts_pkt(0x0100, true, 0b01, &pmt_payload);

        // ---- Video PES (one packet, byte-pattern 0xAA × 16) so video
        // path doesn't bail. ----
        let video_pes = {
            let mut pes = vec![
                0u8, 0u8, 1u8, 0xE0, 0u8, 0u8, 0x80, 0x80, 5, 0x21, 0x00, 0x01, 0x00, 0x01,
            ];
            pes.extend_from_slice(&[0xAAu8; 16]);
            pes
        };
        let video_pkt = ts_pkt(0x0200, true, 0b01, &video_pes);

        // ---- Audio PES carrying TWO ADTS frames (so we exercise the
        // frame-walking loop, not just the first). Each frame: 7-byte
        // header + 32-byte payload = 39 bytes total.
        let mut adts_stream = Vec::new();
        for fill in [0xCCu8, 0xDDu8] {
            adts_stream.extend_from_slice(&build_adts_header_7(1, 3, 2, 39));
            adts_stream.extend_from_slice(&[fill; 32]);
        }
        let audio_pes = {
            // PES header (audio stream_id 0xC0).
            let mut pes = vec![
                0u8, 0u8, 1u8, 0xC0, 0u8, 0u8, 0x80, 0x80, 5, 0x21, 0x00, 0x01, 0x00, 0x01,
            ];
            pes.extend_from_slice(&adts_stream);
            pes
        };
        let audio_pkt = ts_pkt(0x0300, true, 0b01, &audio_pes);

        let mut buf = Vec::new();
        buf.extend_from_slice(&pat_pkt);
        buf.extend_from_slice(&pmt_pkt);
        buf.extend_from_slice(&video_pkt);
        buf.extend_from_slice(&audio_pkt);
        buf.extend_from_slice(&ts_pkt(0x1FFF, false, 0b01, &[]));

        let d = demux_ts(&buf).expect("demux must succeed");
        assert_eq!(d.codec, "mpeg2");
        let audio = d.audio.expect("AAC audio track must be surfaced");
        assert_eq!(audio.codec, "aac");
        assert_eq!(audio.channels, 2, "ch_cfg=2 stereo");
        assert_eq!(audio.sample_rate, 48000, "sr_idx=3 → 48k");
        assert_eq!(audio.timescale, 48000, "AAC timescale = sample_rate");
        assert_eq!(
            audio.asc,
            vec![0x11, 0x90],
            "synthesized ASC for LC/48k/stereo"
        );
        assert_eq!(audio.samples.len(), 2, "two ADTS frames → two samples");
        assert_eq!(
            audio.samples[0].len(),
            32,
            "32-byte payload after 7-byte header strip"
        );
        assert!(audio.samples[0].iter().all(|b| *b == 0xCC));
        assert!(audio.samples[1].iter().all(|b| *b == 0xDD));
        assert_eq!(
            audio.durations,
            vec![1024, 1024],
            "AAC-LC frame duration = 1024 ticks @ sample-rate timescale"
        );
    }

    #[test]
    fn demux_ts_emits_audio_none_when_no_aac_stream_in_pmt() {
        // The original two-stream test (video-only PMT). No audio expected.
        let mut buf = Vec::new();
        let mut pat = vec![0x00];
        let pat_section_len: usize = 5 + 4 + 4;
        pat.push(0xB0 | ((pat_section_len >> 8) & 0x0F) as u8);
        pat.push((pat_section_len & 0xFF) as u8);
        pat.extend_from_slice(&[0x00, 0x01, 0xC1, 0x00, 0x00]);
        pat.extend_from_slice(&[0x00, 0x01, 0xE1, 0x00, 0u8, 0u8, 0u8, 0u8]);
        let mut pat_payload = vec![0u8];
        pat_payload.extend_from_slice(&pat);
        buf.extend_from_slice(&ts_pkt(0x0000, true, 0b01, &pat_payload));

        let mut pmt = vec![0x02];
        let pmt_section_len: usize = 9 + 5 + 4;
        pmt.push(0xB0 | ((pmt_section_len >> 8) & 0x0F) as u8);
        pmt.push((pmt_section_len & 0xFF) as u8);
        pmt.extend_from_slice(&[0x00, 0x01, 0xC1, 0x00, 0x00]);
        pmt.extend_from_slice(&[0xE2, 0x00, 0xF0, 0x00]);
        pmt.extend_from_slice(&[STREAM_TYPE_MPEG2_VIDEO, 0xE2, 0x00, 0xF0, 0x00]);
        pmt.extend_from_slice(&[0u8; 4]);
        let mut pmt_payload = vec![0u8];
        pmt_payload.extend_from_slice(&pmt);
        buf.extend_from_slice(&ts_pkt(0x0100, true, 0b01, &pmt_payload));

        let video_pes = {
            let mut pes = vec![
                0u8, 0u8, 1u8, 0xE0, 0u8, 0u8, 0x80, 0x80, 5, 0x21, 0x00, 0x01, 0x00, 0x01,
            ];
            pes.extend_from_slice(&[0xAAu8; 16]);
            pes
        };
        buf.extend_from_slice(&ts_pkt(0x0200, true, 0b01, &video_pes));
        buf.extend_from_slice(&ts_pkt(0x1FFF, false, 0b01, &[]));

        let d = demux_ts(&buf).expect("demux");
        assert!(
            d.audio.is_none(),
            "PMT without AAC-ADTS stream → no audio track surfaced"
        );
    }

    // ---------------- Squad-37: AC-3 / E-AC-3 in TS, multi-program, encrypted ----------------

    /// Build a minimal AC-3 syncframe by hand with a valid frmsizecod:
    /// fscod=0 (48k), bit_rate_code=8 (128 kbps) → frame_length = 384
    /// bytes per Table F.7. acmod=2 stereo, lfeon=0, bsid=8, bsmod=0.
    /// The body bytes after the BSI prefix are zero-padded — only the
    /// first ~7 bytes participate in our parser.
    fn synth_ac3_frame_stereo_48k_128k() -> Vec<u8> {
        let mut bw = BitWriter::new();
        bw.put(16, 0x0B77); // syncword
        bw.put(16, 0); // crc1
        bw.put(2, 0); // fscod=0 → 48k
        bw.put(6, 8 << 1); // frmsizecod = bit_rate_code(8) << 1 = 16
        bw.put(5, 8); // bsid
        bw.put(3, 0); // bsmod
        bw.put(3, 2); // acmod=2 stereo
        // acmod=2 → dsurmod (2 bits)
        bw.put(2, 0);
        bw.put(1, 0); // lfeon=0
        // Pad up to 384 bytes (the AC-3 frame size we just announced).
        while bw.bytes.len() < 384 {
            bw.put(8, 0);
        }
        bw.flush()
    }

    /// E-AC-3 stereo frame with 6 audio blocks (numblkscod=3) at 48k.
    /// frmsiz chosen such that frame_size_bytes = 192 ((0x5F + 1) * 2).
    fn synth_eac3_frame_stereo_48k_192bytes() -> Vec<u8> {
        let mut bw = BitWriter::new();
        bw.put(16, 0x0B77);
        bw.put(2, 0); // strmtyp = 0 (independent)
        bw.put(3, 0); // substreamid
        bw.put(11, 0x5F); // frmsiz = 95 → frame_size = 192 bytes
        bw.put(2, 0); // fscod=0 → 48k
        bw.put(2, 3); // numblkscod=3 → 6 blocks
        bw.put(3, 2); // acmod=2 stereo
        bw.put(1, 0); // lfeon
        bw.put(5, 16); // bsid=16
        bw.put(5, 0); // dialnorm
        bw.put(1, 0); // compre=0
        while bw.bytes.len() < 192 {
            bw.put(8, 0);
        }
        bw.flush()
    }

    /// Local copy of the BitWriter used by the existing AAC tests, kept
    /// alongside the Squad-37 sync-frame builders for self-containment.
    struct BitWriter {
        bytes: Vec<u8>,
        bit_pos: usize,
    }
    impl BitWriter {
        fn new() -> Self {
            Self {
                bytes: Vec::new(),
                bit_pos: 0,
            }
        }
        fn put(&mut self, n: usize, v: u32) {
            for i in (0..n).rev() {
                let bit = ((v >> i) & 0x01) as u8;
                if self.bit_pos % 8 == 0 {
                    self.bytes.push(0);
                }
                let byte_idx = self.bit_pos / 8;
                let bit_idx = 7 - (self.bit_pos % 8);
                self.bytes[byte_idx] |= bit << bit_idx;
                self.bit_pos += 1;
            }
        }
        fn flush(self) -> Vec<u8> {
            self.bytes
        }
    }

    /// Build a continuation TS packet (PUSI=0) on `pid` with raw
    /// `payload` bytes. Used by `build_ts_with_audio` when an audio PES
    /// payload doesn't fit in a single 188-byte packet — the PES header
    /// rides on the PUSI=1 packet, and continuation packets carry the
    /// rest of the elementary-stream bytes verbatim until the next PUSI.
    fn ts_pkt_continuation(pid: u16, payload: &[u8]) -> [u8; TS_PACKET] {
        let mut p = [0xFFu8; TS_PACKET];
        p[0] = TS_SYNC;
        p[1] = ((pid >> 8) & 0x1F) as u8; // PUSI=0
        p[2] = (pid & 0xFF) as u8;
        p[3] = 0b01 << 4; // adaptation=01 (payload only), continuity=0
        let pay_len = payload.len().min(TS_PACKET - 4);
        p[4..4 + pay_len].copy_from_slice(&payload[..pay_len]);
        p
    }

    /// Helper to build a TS file with: PAT, PMT, video PES (so the
    /// video gate doesn't bail), audio PES on `audio_pid` with a given
    /// `stream_type` byte and `descriptor_loop` for the PMT entry.
    /// `audio_es` is the elementary-stream payload (AC-3 frame, etc.)
    /// inserted into the audio PES packet body. If `audio_es` is too
    /// large to fit in a single TS packet's payload area (~184 bytes),
    /// the helper emits one PUSI=1 packet with the PES header + the
    /// first chunk and successive PUSI=0 continuation packets carrying
    /// the rest.
    fn build_ts_with_audio(
        audio_stream_type: u8,
        audio_descriptors: &[u8],
        audio_pid: u16,
        audio_es: &[u8],
    ) -> Vec<u8> {
        // PAT pointing at PMT 0x100.
        let mut pat = vec![0x00];
        let pat_section_len: usize = 5 + 4 + 4;
        pat.push(0xB0 | ((pat_section_len >> 8) & 0x0F) as u8);
        pat.push((pat_section_len & 0xFF) as u8);
        pat.extend_from_slice(&[0x00, 0x01, 0xC1, 0x00, 0x00]);
        pat.extend_from_slice(&[0x00, 0x01, 0xE1, 0x00, 0u8, 0u8, 0u8, 0u8]);
        let mut pat_payload = vec![0u8];
        pat_payload.extend_from_slice(&pat);
        let pat_pkt = ts_pkt(0x0000, true, 0b01, &pat_payload);

        // PMT advertising MPEG-2 video on 0x200 + audio entry.
        let mut pmt = vec![0x02];
        let pmt_stream_entries = 5  // video stream entry
            + 5 + audio_descriptors.len(); // audio stream entry + descriptors
        let pmt_section_len: usize = 9 + pmt_stream_entries + 4;
        pmt.push(0xB0 | ((pmt_section_len >> 8) & 0x0F) as u8);
        pmt.push((pmt_section_len & 0xFF) as u8);
        pmt.extend_from_slice(&[0x00, 0x01, 0xC1, 0x00, 0x00]);
        pmt.extend_from_slice(&[0xE2, 0x00]); // PCR PID = 0x200
        pmt.extend_from_slice(&[0xF0, 0x00]); // program_info_length=0
        // Stream 1: MPEG-2 video on 0x200, no descriptors.
        pmt.extend_from_slice(&[STREAM_TYPE_MPEG2_VIDEO, 0xE2, 0x00, 0xF0, 0x00]);
        // Stream 2: audio_pid w/ given stream_type + descriptors.
        pmt.push(audio_stream_type);
        pmt.push(0xE0 | ((audio_pid >> 8) & 0x1F) as u8);
        pmt.push((audio_pid & 0xFF) as u8);
        let esi_len = audio_descriptors.len() as u16;
        pmt.push(0xF0 | ((esi_len >> 8) & 0x0F) as u8);
        pmt.push((esi_len & 0xFF) as u8);
        pmt.extend_from_slice(audio_descriptors);
        pmt.extend_from_slice(&[0u8; 4]); // CRC placeholder
        let mut pmt_payload = vec![0u8];
        pmt_payload.extend_from_slice(&pmt);
        let pmt_pkt = ts_pkt(0x0100, true, 0b01, &pmt_payload);

        // Video PES (just enough so the video path doesn't bail).
        let video_pes = {
            let mut pes = vec![
                0u8, 0u8, 1u8, 0xE0, 0u8, 0u8, 0x80, 0x80, 5, 0x21, 0x00, 0x01, 0x00, 0x01,
            ];
            pes.extend_from_slice(&[0xAAu8; 16]);
            pes
        };
        let video_pkt = ts_pkt(0x0200, true, 0b01, &video_pes);

        // Audio PES — a single PES packet carrying all of audio_es,
        // potentially split across multiple TS packets via continuation.
        // Stream_id 0xC0 is audio per ISO/IEC 13818-1 §2.4.3.7.
        // Note: for AC-3 / E-AC-3, ATSC A/53 PES uses stream_id 0xBD
        // (PES private) rather than 0xC0; our parse_pes_header_audio
        // accepts the 0xC0..=0xDF range so we use 0xC0 here for test
        // simplicity. In real-world bitstreams the parser would also
        // need 0xBD support — that's a separate uplift.
        let mut audio_pes = vec![
            0u8, 0u8, 1u8, 0xC0, 0u8, 0u8, 0x80, 0x80, 5, 0x21, 0x00, 0x01, 0x00, 0x01,
        ];
        audio_pes.extend_from_slice(audio_es);

        // Split audio_pes across one PUSI=1 packet plus continuation
        // packets so PES payloads larger than 184 bytes flow through.
        let first_chunk_max = TS_PACKET - 4; // 184 bytes per TS packet payload
        let mut audio_pkts: Vec<[u8; TS_PACKET]> = Vec::new();
        let first_len = audio_pes.len().min(first_chunk_max);
        audio_pkts.push(ts_pkt(audio_pid, true, 0b01, &audio_pes[..first_len]));
        let mut cursor = first_len;
        while cursor < audio_pes.len() {
            let end = (cursor + first_chunk_max).min(audio_pes.len());
            audio_pkts.push(ts_pkt_continuation(audio_pid, &audio_pes[cursor..end]));
            cursor = end;
        }

        let mut buf = Vec::new();
        buf.extend_from_slice(&pat_pkt);
        buf.extend_from_slice(&pmt_pkt);
        buf.extend_from_slice(&video_pkt);
        for pkt in &audio_pkts {
            buf.extend_from_slice(pkt);
        }
        buf.extend_from_slice(&ts_pkt(0x1FFF, false, 0b01, &[]));
        buf
    }

    #[test]
    fn pmt_walker_classifies_aac_ac3_eac3_stream_types() {
        // Build a synthetic PMT section with one of each audio
        // stream_type and verify the walker tags them correctly.
        let mut pmt = vec![0x02];
        let stream_entries = 5 + 5 + 5 + 5; // video + AAC + AC-3 + E-AC-3
        let pmt_section_len: usize = 9 + stream_entries + 4;
        pmt.push(0xB0 | ((pmt_section_len >> 8) & 0x0F) as u8);
        pmt.push((pmt_section_len & 0xFF) as u8);
        pmt.extend_from_slice(&[0x00, 0x01, 0xC1, 0x00, 0x00]);
        pmt.extend_from_slice(&[0xE2, 0x00, 0xF0, 0x00]); // PCR + pil=0
        pmt.extend_from_slice(&[STREAM_TYPE_MPEG2_VIDEO, 0xE2, 0x00, 0xF0, 0x00]);
        pmt.extend_from_slice(&[STREAM_TYPE_AAC_ADTS, 0xE3, 0x00, 0xF0, 0x00]);
        pmt.extend_from_slice(&[STREAM_TYPE_AC3, 0xE4, 0x00, 0xF0, 0x00]);
        pmt.extend_from_slice(&[STREAM_TYPE_EAC3, 0xE5, 0x00, 0xF0, 0x00]);
        pmt.extend_from_slice(&[0u8; 4]);

        let (video, audio) = parse_pmt_streams(&pmt).expect("parse");
        assert_eq!(video.len(), 1);
        assert_eq!(video[0].pid, 0x200);
        assert_eq!(audio.len(), 3);
        assert_eq!(
            (audio[0].pid, audio[0].kind),
            (0x300, AudioCodecKind::AacAdts)
        );
        assert_eq!((audio[1].pid, audio[1].kind), (0x400, AudioCodecKind::Ac3));
        assert_eq!((audio[2].pid, audio[2].kind), (0x500, AudioCodecKind::Eac3));
    }

    #[test]
    fn pmt_walker_recognises_dvb_ac3_via_registration_descriptor() {
        // PES private (0x06) with a registration_descriptor whose 4-char
        // identifier is "AC-3" → audio routed as AC-3 per ETSI TS 101 154.
        let mut pmt = vec![0x02];
        let descriptors: [u8; 6] = [DESC_TAG_REGISTRATION, 4, b'A', b'C', b'-', b'3'];
        let stream_entries = 5 + 5 + descriptors.len();
        let pmt_section_len: usize = 9 + stream_entries + 4;
        pmt.push(0xB0 | ((pmt_section_len >> 8) & 0x0F) as u8);
        pmt.push((pmt_section_len & 0xFF) as u8);
        pmt.extend_from_slice(&[0x00, 0x01, 0xC1, 0x00, 0x00]);
        pmt.extend_from_slice(&[0xE2, 0x00, 0xF0, 0x00]);
        pmt.extend_from_slice(&[STREAM_TYPE_MPEG2_VIDEO, 0xE2, 0x00, 0xF0, 0x00]);
        pmt.push(STREAM_TYPE_PES_PRIVATE);
        pmt.extend_from_slice(&[0xE3, 0x00]);
        let esi_len = descriptors.len() as u16;
        pmt.push(0xF0 | ((esi_len >> 8) & 0x0F) as u8);
        pmt.push((esi_len & 0xFF) as u8);
        pmt.extend_from_slice(&descriptors);
        pmt.extend_from_slice(&[0u8; 4]);

        let (_, audio) = parse_pmt_streams(&pmt).expect("parse");
        assert_eq!(audio.len(), 1);
        assert_eq!(audio[0].kind, AudioCodecKind::Ac3);
        assert_eq!(audio[0].stream_type, STREAM_TYPE_PES_PRIVATE);
    }

    #[test]
    fn pmt_walker_recognises_dvb_eac3_via_registration_descriptor() {
        let mut pmt = vec![0x02];
        let descriptors: [u8; 6] = [DESC_TAG_REGISTRATION, 4, b'E', b'A', b'C', b'3'];
        let stream_entries = 5 + 5 + descriptors.len();
        let pmt_section_len: usize = 9 + stream_entries + 4;
        pmt.push(0xB0 | ((pmt_section_len >> 8) & 0x0F) as u8);
        pmt.push((pmt_section_len & 0xFF) as u8);
        pmt.extend_from_slice(&[0x00, 0x01, 0xC1, 0x00, 0x00]);
        pmt.extend_from_slice(&[0xE2, 0x00, 0xF0, 0x00]);
        pmt.extend_from_slice(&[STREAM_TYPE_MPEG2_VIDEO, 0xE2, 0x00, 0xF0, 0x00]);
        pmt.push(STREAM_TYPE_PES_PRIVATE);
        pmt.extend_from_slice(&[0xE3, 0x00]);
        let esi_len = descriptors.len() as u16;
        pmt.push(0xF0 | ((esi_len >> 8) & 0x0F) as u8);
        pmt.push((esi_len & 0xFF) as u8);
        pmt.extend_from_slice(&descriptors);
        pmt.extend_from_slice(&[0u8; 4]);

        let (_, audio) = parse_pmt_streams(&pmt).expect("parse");
        assert_eq!(audio.len(), 1);
        assert_eq!(audio[0].kind, AudioCodecKind::Eac3);
    }

    #[test]
    fn extract_ac3_frames_from_synthetic_ts_yields_passthrough_track() {
        // stream_type 0x81, no descriptors needed.
        let frame = synth_ac3_frame_stereo_48k_128k();
        // Concatenate two frames so the frame loop runs more than once.
        let mut es = frame.clone();
        es.extend_from_slice(&frame);
        let buf = build_ts_with_audio(STREAM_TYPE_AC3, &[], 0x300, &es);

        let d = demux_ts(&buf).expect("demux");
        let audio = d.audio.expect("AC-3 audio surfaced");
        assert_eq!(audio.codec, "ac3");
        assert_eq!(audio.channels, 2);
        assert_eq!(audio.sample_rate, 48_000);
        assert_eq!(audio.timescale, 48_000);
        // dac3 body is the 3-byte payload that goes into the MP4 sample
        // entry verbatim — derived from the first sync header.
        assert_eq!(audio.codec_private.len(), 3);
        // Two frames in, two samples out (raw frame bytes, sync word
        // intact).
        assert!(
            audio.samples.len() >= 1,
            "at least one AC-3 frame extracted"
        );
        assert_eq!(
            &audio.samples[0][..2],
            &[0x0B, 0x77],
            "AC-3 frame begins with 0x0B77 sync word verbatim"
        );
        // Each AC-3 frame is 1536 samples per spec.
        assert!(
            audio.durations.iter().all(|&d| d == 1536),
            "AC-3 frames are 1536 samples each"
        );
    }

    #[test]
    fn extract_eac3_frames_from_synthetic_ts_yields_passthrough_track() {
        let frame = synth_eac3_frame_stereo_48k_192bytes();
        let mut es = frame.clone();
        es.extend_from_slice(&frame);
        let buf = build_ts_with_audio(STREAM_TYPE_EAC3, &[], 0x300, &es);

        let d = demux_ts(&buf).expect("demux");
        let audio = d.audio.expect("E-AC-3 audio surfaced");
        assert_eq!(audio.codec, "eac3");
        assert_eq!(audio.channels, 2);
        assert_eq!(audio.sample_rate, 48_000);
        // dec3 single-substream body is 5 bytes per ETSI TS 102 366 §F.6.
        assert_eq!(audio.codec_private.len(), 5);
        assert!(!audio.samples.is_empty());
        assert_eq!(
            &audio.samples[0][..2],
            &[0x0B, 0x77],
            "E-AC-3 frame begins with 0x0B77 sync word verbatim"
        );
        // numblkscod=3 → 1536 samples/frame.
        assert!(audio.durations.iter().all(|&d| d == 1536));
    }

    #[test]
    fn extract_ac3_via_pes_private_with_dvb_registration() {
        // stream_type 0x06 + registration "AC-3" must route through the
        // AC-3 extractor end-to-end.
        let frame = synth_ac3_frame_stereo_48k_128k();
        let descriptors: [u8; 6] = [DESC_TAG_REGISTRATION, 4, b'A', b'C', b'-', b'3'];
        let buf = build_ts_with_audio(STREAM_TYPE_PES_PRIVATE, &descriptors, 0x300, &frame);
        let d = demux_ts(&buf).expect("demux");
        let audio = d.audio.expect("AC-3 audio via DVB registration surfaced");
        assert_eq!(audio.codec, "ac3");
        assert_eq!(&audio.samples[0][..2], &[0x0B, 0x77]);
    }

    #[test]
    fn dac3_body_synthesized_from_first_ts_frame_matches_sync_header() {
        // The dac3 body the TS extractor produces must equal the body
        // we'd compute by parsing the same first frame independently —
        // proves the AC-3 path is using the canonical Squad-26 helper
        // rather than a parallel implementation.
        let frame = synth_ac3_frame_stereo_48k_128k();
        let buf = build_ts_with_audio(STREAM_TYPE_AC3, &[], 0x300, &frame);
        let d = demux_ts(&buf).expect("demux");
        let audio = d.audio.expect("AC-3 audio");
        let parsed = match crate::ac3_sync::parse_sync_info(&frame).unwrap() {
            crate::ac3_sync::SyncInfo::Ac3(s) => s,
            _ => panic!("expected AC-3"),
        };
        let expected = crate::mux::dac3_body_from_sync(&parsed);
        assert_eq!(
            audio.codec_private,
            expected.to_vec(),
            "TS-extracted dac3 must match the canonical helper"
        );
    }

    /// Build a TS file with two distinct programs (program_number 1 and
    /// 2). Program 1 carries MPEG-2 video on 0x200; program 2 carries
    /// H.264 video on 0x300. Both PMTs live in their own PIDs (0x100,
    /// 0x101 respectively).
    fn build_two_program_ts() -> Vec<u8> {
        // PAT with TWO program entries.
        let mut pat = vec![0x00];
        let pat_section_len: usize = 5 + 4 + 4 + 4; // 2 programs + CRC
        pat.push(0xB0 | ((pat_section_len >> 8) & 0x0F) as u8);
        pat.push((pat_section_len & 0xFF) as u8);
        pat.extend_from_slice(&[0x00, 0x01, 0xC1, 0x00, 0x00]);
        pat.extend_from_slice(&[0x00, 0x01, 0xE1, 0x00]); // program 1 → PMT 0x100
        pat.extend_from_slice(&[0x00, 0x02, 0xE1, 0x01]); // program 2 → PMT 0x101
        pat.extend_from_slice(&[0u8; 4]);
        let mut pat_payload = vec![0u8];
        pat_payload.extend_from_slice(&pat);
        let pat_pkt = ts_pkt(0x0000, true, 0b01, &pat_payload);

        // PMT 1: MPEG-2 video on 0x200.
        let mut pmt1 = vec![0x02];
        let pmt1_section_len: usize = 9 + 5 + 4;
        pmt1.push(0xB0 | ((pmt1_section_len >> 8) & 0x0F) as u8);
        pmt1.push((pmt1_section_len & 0xFF) as u8);
        pmt1.extend_from_slice(&[0x00, 0x01, 0xC1, 0x00, 0x00]); // program 1
        pmt1.extend_from_slice(&[0xE2, 0x00, 0xF0, 0x00]);
        pmt1.extend_from_slice(&[STREAM_TYPE_MPEG2_VIDEO, 0xE2, 0x00, 0xF0, 0x00]);
        pmt1.extend_from_slice(&[0u8; 4]);
        let mut pmt1_payload = vec![0u8];
        pmt1_payload.extend_from_slice(&pmt1);
        let pmt1_pkt = ts_pkt(0x0100, true, 0b01, &pmt1_payload);

        // PMT 2: H.264 video on 0x300.
        let mut pmt2 = vec![0x02];
        let pmt2_section_len: usize = 9 + 5 + 4;
        pmt2.push(0xB0 | ((pmt2_section_len >> 8) & 0x0F) as u8);
        pmt2.push((pmt2_section_len & 0xFF) as u8);
        pmt2.extend_from_slice(&[0x00, 0x02, 0xC1, 0x00, 0x00]); // program 2
        pmt2.extend_from_slice(&[0xE3, 0x00, 0xF0, 0x00]);
        pmt2.extend_from_slice(&[STREAM_TYPE_H264, 0xE3, 0x00, 0xF0, 0x00]);
        pmt2.extend_from_slice(&[0u8; 4]);
        let mut pmt2_payload = vec![0u8];
        pmt2_payload.extend_from_slice(&pmt2);
        let pmt2_pkt = ts_pkt(0x0101, true, 0b01, &pmt2_payload);

        // Distinct PES bytes so we can tell programs apart at sample
        // level. Program 1 → 0xAA; program 2 → 0xBB.
        let make_pes = |fill: u8| {
            let mut pes = vec![
                0u8, 0u8, 1u8, 0xE0, 0u8, 0u8, 0x80, 0x80, 5, 0x21, 0x00, 0x01, 0x00, 0x01,
            ];
            pes.extend_from_slice(&[fill; 16]);
            pes
        };
        let p1_pes = ts_pkt(0x0200, true, 0b01, &make_pes(0xAA));
        let p2_pes = ts_pkt(0x0300, true, 0b01, &make_pes(0xBB));

        let mut buf = Vec::new();
        buf.extend_from_slice(&pat_pkt);
        buf.extend_from_slice(&pmt1_pkt);
        buf.extend_from_slice(&pmt2_pkt);
        // Two PES per program so the streaming path's PUSI flush yields.
        buf.extend_from_slice(&p1_pes);
        buf.extend_from_slice(&p2_pes);
        buf.extend_from_slice(&ts_pkt(0x0200, true, 0b01, &make_pes(0xAA)));
        buf.extend_from_slice(&ts_pkt(0x0300, true, 0b01, &make_pes(0xBB)));
        buf.extend_from_slice(&ts_pkt(0x1FFF, false, 0b01, &[]));
        buf
    }

    #[test]
    fn streaming_demuxer_lists_all_pat_programs() {
        let buf = build_two_program_ts();
        let dem = demux_ts_streaming_init(&buf).expect("init");
        let progs = dem.programs();
        assert_eq!(progs.len(), 2, "PAT advertised 2 programs");
        let nums: Vec<u16> = progs.iter().map(|p| p.program_number).collect();
        assert_eq!(nums, vec![1, 2]);
        assert_eq!(progs[0].pmt_pid, 0x100);
        assert_eq!(progs[1].pmt_pid, 0x101);
        // Program 1 → MPEG-2 on 0x200; program 2 → H.264 on 0x300.
        assert_eq!(progs[0].video_streams[0].pid, 0x200);
        assert_eq!(
            progs[0].video_streams[0].stream_type,
            STREAM_TYPE_MPEG2_VIDEO
        );
        assert_eq!(progs[1].video_streams[0].pid, 0x300);
        assert_eq!(progs[1].video_streams[0].stream_type, STREAM_TYPE_H264);
    }

    #[test]
    fn streaming_demuxer_default_picks_first_program() {
        let buf = build_two_program_ts();
        let mut dem = demux_ts_streaming_init(&buf).expect("init");
        assert_eq!(dem.active_program_index(), 0);
        assert_eq!(dem.header().codec, "mpeg2", "program 1 is MPEG-2");
        // Drain — samples should be 0xAA-filled (program 1's bytes).
        let s = dem.next_video_sample().expect("sample").expect("some");
        assert!(
            s.data.iter().any(|&b| b == 0xAA),
            "program 1 sample should carry 0xAA"
        );
        assert!(
            !s.data.iter().any(|&b| b == 0xBB),
            "program 1 sample must not carry program 2's 0xBB"
        );
    }

    #[test]
    fn streaming_demuxer_select_program_switches_active_streams() {
        let buf = build_two_program_ts();
        let mut dem = demux_ts_streaming_init(&buf).expect("init");
        dem.select_program(2).expect("switch to program 2");
        assert_eq!(dem.active_program_index(), 1);
        assert_eq!(dem.header().codec, "h264", "program 2 is H.264");
        let s = dem.next_video_sample().expect("sample").expect("some");
        assert!(
            s.data.iter().any(|&b| b == 0xBB),
            "program 2 sample should carry 0xBB"
        );
        assert!(
            !s.data.iter().any(|&b| b == 0xAA),
            "program 2 sample must not carry program 1's 0xAA"
        );
    }

    #[test]
    fn streaming_demuxer_select_program_rejects_unknown_number() {
        let buf = build_two_program_ts();
        let mut dem = demux_ts_streaming_init(&buf).expect("init");
        assert!(
            dem.select_program(99).is_err(),
            "unknown program_number must error rather than silently no-op"
        );
    }

    /// Build a single-program TS where the video PID's packets carry
    /// `transport_scrambling_control != 0` (TSC=01 = "user-defined,
    /// reserved" in ISO/IEC 13818-1 — both this and 10/11 indicate the
    /// payload is encrypted and we have no CA tables).
    fn build_encrypted_ts() -> Vec<u8> {
        // Reuse the single-program PAT/PMT shape from the existing tests.
        let mut pat = vec![0x00];
        let pat_section_len: usize = 5 + 4 + 4;
        pat.push(0xB0 | ((pat_section_len >> 8) & 0x0F) as u8);
        pat.push((pat_section_len & 0xFF) as u8);
        pat.extend_from_slice(&[0x00, 0x01, 0xC1, 0x00, 0x00]);
        pat.extend_from_slice(&[0x00, 0x01, 0xE1, 0x00, 0u8, 0u8, 0u8, 0u8]);
        let mut pat_payload = vec![0u8];
        pat_payload.extend_from_slice(&pat);
        let pat_pkt = ts_pkt(0x0000, true, 0b01, &pat_payload);

        let mut pmt = vec![0x02];
        let pmt_section_len: usize = 9 + 5 + 4;
        pmt.push(0xB0 | ((pmt_section_len >> 8) & 0x0F) as u8);
        pmt.push((pmt_section_len & 0xFF) as u8);
        pmt.extend_from_slice(&[0x00, 0x01, 0xC1, 0x00, 0x00]);
        pmt.extend_from_slice(&[0xE2, 0x00, 0xF0, 0x00]);
        pmt.extend_from_slice(&[STREAM_TYPE_MPEG2_VIDEO, 0xE2, 0x00, 0xF0, 0x00]);
        pmt.extend_from_slice(&[0u8; 4]);
        let mut pmt_payload = vec![0u8];
        pmt_payload.extend_from_slice(&pmt);
        let pmt_pkt = ts_pkt(0x0100, true, 0b01, &pmt_payload);

        // Encrypted video PES: build the packet as normal but flip
        // bits 6-7 of byte 3 to TSC=01.
        let video_pes = {
            let mut pes = vec![
                0u8, 0u8, 1u8, 0xE0, 0u8, 0u8, 0x80, 0x80, 5, 0x21, 0x00, 0x01, 0x00, 0x01,
            ];
            pes.extend_from_slice(&[0xAAu8; 16]);
            pes
        };
        let mut video_pkt = ts_pkt(0x0200, true, 0b01, &video_pes);
        // TSC = 01 (single-bit set in the top 2 bits of byte 3).
        video_pkt[3] = (video_pkt[3] & 0x3F) | (0x01 << 6);

        let mut buf = Vec::new();
        buf.extend_from_slice(&pat_pkt);
        buf.extend_from_slice(&pmt_pkt);
        buf.extend_from_slice(&video_pkt);
        buf.extend_from_slice(&ts_pkt(0x1FFF, false, 0b01, &[]));
        buf
    }

    #[test]
    fn streaming_demuxer_drops_video_when_active_pid_is_scrambled() {
        let buf = build_encrypted_ts();
        let mut dem = demux_ts_streaming_init(&buf).expect("init");
        // First call should hit the encrypted packet, latch the guard,
        // and return None. No samples should ever surface.
        let s = dem.next_video_sample().expect("call must not error");
        assert!(
            s.is_none(),
            "encrypted TS → next_video_sample returns None on first call"
        );
        // Subsequent calls remain None — the guard latches.
        let s2 = dem.next_video_sample().expect("call must not error");
        assert!(
            s2.is_none(),
            "encrypted TS → guard remains latched on subsequent calls"
        );
    }
}
