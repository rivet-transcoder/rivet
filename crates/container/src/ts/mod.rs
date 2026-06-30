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

mod audio;
mod framerate;
mod pat_pmt;
mod pes;
mod streaming;
#[cfg(test)]
mod tests;

pub use streaming::TsStreamingDemuxer;
pub(crate) use streaming::demux_ts_streaming_init;

use anyhow::{Context, Result, bail};
use codec::frame::{ColorSpace, PixelFormat, StreamInfo};

use crate::demux::DemuxResult;

// ---------------------------------------------------------------------------
// Shared constants — visible to all sub-modules via `super::`.
// ---------------------------------------------------------------------------

pub(super) const TS_PACKET: usize = 188;
pub(super) const TS_SYNC: u8 = 0x47;

pub(super) const STREAM_TYPE_MPEG2_VIDEO: u8 = 0x02;
pub(super) const STREAM_TYPE_H264: u8 = 0x1B;
pub(super) const STREAM_TYPE_HEVC: u8 = 0x24;
/// PES private stream_type. ETSI TS 101 154 (DVB) routes AC-3 / E-AC-3
/// through this generic stream_type with a `registration_descriptor`
/// (descriptor_tag = 0x05) tagged "AC-3" or "EAC3" carrying the actual
/// codec identity. We only honour 0x06 entries that carry one of those
/// two registrations — random PES-private streams (DVB subtitles, teletext)
/// are dropped silently.
pub(super) const STREAM_TYPE_PES_PRIVATE: u8 = 0x06;
/// PMT stream_type for AAC carried as ADTS frames in PES packets.
/// Defined in ISO/IEC 13818-1:2019 Table 2-34 — `0x0F` is
/// "ISO/IEC 13818-7 Audio with ADTS transport syntax", which is the
/// MPEG-2/MPEG-4 AAC ADTS form that broadcast / streaming MPEG-TS uses.
pub(super) const STREAM_TYPE_AAC_ADTS: u8 = 0x0F;
/// ATSC A/53 §3 / ATSC A/52 Annex A — AC-3 elementary streams in PES
/// packets. Common in over-the-air ATSC broadcast captures (.ts / .trp).
pub(super) const STREAM_TYPE_AC3: u8 = 0x81;
/// ATSC A/53 §3 / ATSC A/52 Annex E — E-AC-3 elementary streams.
pub(super) const STREAM_TYPE_EAC3: u8 = 0x87;

/// PMT descriptor_tag for the registration_descriptor carrying a
/// 4-character format identifier. ETSI TS 101 154 §F (DVB) registers
/// `"AC-3"` (0x41432D33) and `"EAC3"` (0x45414333) for Dolby streams
/// carried as PES-private (stream_type 0x06).
pub(super) const DESC_TAG_REGISTRATION: u8 = 0x05;
pub(super) const REG_AC3: u32 = 0x41432D33; // "AC-3"
pub(super) const REG_EAC3: u32 = 0x45414333; // "EAC3"

// ---------------------------------------------------------------------------
// Shared public types
// ---------------------------------------------------------------------------

/// One PAT entry — `(program_number, pmt_pid)`. Entry with program=0 is
/// the network_PID and is skipped by callers (it is not a real program).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PatProgram {
    pub(super) program_number: u16,
    pub(super) pmt_pid: u16,
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

// ---------------------------------------------------------------------------
// Shared private helpers used by both entry points
// ---------------------------------------------------------------------------

/// Decide whether the file uses 188-byte (plain TS) or 192-byte (BDAV
/// M2TS) packets. Returns (packet_count, stride, prefix_len).
/// BDAV prepends a 4-byte TP_extra_header before each 188-byte TS
/// packet, so stride=192 and prefix_len=4. For plain TS stride=188
/// and prefix_len=0.
pub(super) fn detect_packet_layout(data: &[u8]) -> Result<(usize, usize, usize)> {
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
pub(super) fn ts_psi_payload(pkt: &[u8]) -> Option<&[u8]> {
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

// ---------------------------------------------------------------------------
// Public entry point — legacy materialise-all demux
// ---------------------------------------------------------------------------

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
                && let Some(p) = pat_pmt::parse_pat_first_pmt_pid(payload)
            {
                pmt_pid = Some(p);
            }
            continue;
        }
        // PMT
        if let (Some(pmt), None) = (pmt_pid, chosen_video)
            && pid == pmt
            && let Some(payload) = ts_psi_payload(pkt)
            && let Some((video_streams, audio_streams)) = pat_pmt::parse_pmt_streams(payload)
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

            let Some((es_start, pts)) = pes::parse_pes_header(payload) else {
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
    let frame_rate = framerate::estimate_frame_rate_from_ptses(&ptses)
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
        match audio::extract_ts_audio(data, packets, packet_stride, prefix_len, info) {
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
