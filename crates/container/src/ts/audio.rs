//! AAC-ADTS, AC-3, and E-AC-3 audio extraction from MPEG-TS PES streams.
//!
//! # AAC-ADTS (Squad-27)
//!
//! The MPEG-TS audio path stores AAC as a stream of ADTS frames inside PES
//! packets — same PES framing as the video path, but the elementary stream
//! payload is ADTS, not Annex-B. The downstream mux (Squad-18) wants raw
//! AAC access units (no ADTS header) plus a synthesized AudioSpecificConfig
//! (ASC) — both come from the first ADTS header.
//!
//! References:
//! - ADTS frame layout: ISO/IEC 13818-7 §6.2 (the "_adts_frame()" syntax
//!   table — 7-byte fixed header without CRC, 9-byte with CRC).
//! - ASC layout: ISO/IEC 14496-3 §1.6.2 (`AudioSpecificConfig` →
//!   `GetAudioObjectType` + `samplingFrequencyIndex` + `channelConfiguration`
//!   + `GASpecificConfig` for AOT 1..7).
//!
//! # AC-3 / E-AC-3 (Squad-37)
//!
//! PES payload for an AC-3 / E-AC-3 audio PID is a stream of raw
//! syncframes — 0x0B77 sync word at the start of each frame, followed by
//! the BSI fields whose layout `crate::ac3_sync` already parses for
//! MP4 / MKV passthrough. Squad-26 settled the codec_private wire
//! format: a 3-byte `dac3` body for AC-3, a 5-byte `dec3` body for
//! vanilla single-substream E-AC-3.
//!
//! The MP4 mux contract (Squad-26) is: pass the raw AC-3 / E-AC-3
//! frames through verbatim as samples; populate `codec_private` with the
//! dac3/dec3 body derived from the first frame; `asc` stays empty for
//! these codecs. We do NOT re-frame, decode, or strip anything — the
//! frames are length-self-describing via the syncframe info, and the
//! muxer / downstream demuxer round-trip in Squad-26 already handles
//! that on the MP4 side.

use anyhow::{Context, Result, bail};

use crate::ac3_sync::{
    self, Eac3SyncInfo, SyncInfo, ac3_bit_rate_kbps, channel_count, eac3_sample_rate_hz,
    eac3_samples_per_frame,
};
use crate::demux::AudioTrack;
use crate::mux::{dac3_body_from_sync, dec3_body_from_sync};

use super::{AudioCodecKind, AudioStreamInfo, TS_PACKET, TS_SYNC};

// ---------------------------------------------------------------------------
// AAC-ADTS helpers
// ---------------------------------------------------------------------------

// Sampling frequency table (ISO/IEC 14496-3 §1.6.3.4 Table 1.16):
const AAC_SAMPLE_RATES: [u32; 13] = [
    96000, 88200, 64000, 48000, 44100, 32000, 24000, 22050, 16000, 12000, 11025, 8000, 7350,
];

/// Parsed view of a single ADTS frame header (ISO/IEC 13818-7 §6.2).
/// Only the fields we need for ASC synthesis + frame slicing — buffer
/// fullness / number_of_raw_data_blocks are not exposed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct AdtsHeader {
    /// ADTS profile (2 bits): AAC ObjectType - 1.
    /// `0`=Main, `1`=LC, `2`=SSR, `3`=LTP. Maps to ASC AOT via `+1`.
    pub(super) profile: u8,
    /// Sampling frequency index (4 bits, 0..=12 valid; 15 = explicit).
    /// `decode_sample_rate_index` resolves to Hz.
    pub(super) sampling_frequency_index: u8,
    /// `channel_configuration` (3 bits). 1 = mono, 2 = stereo, etc.
    /// 0 = "channel config defined in PCE" — uncommon; we accept 1/2
    /// at the audio-track surface, downstream mux rejects the rest.
    pub(super) channel_configuration: u8,
    /// Whole frame length in bytes including header + (optional CRC) +
    /// AAC payload.
    pub(super) frame_length: usize,
    /// Length of the ADTS header itself: 7 bytes if `protection_absent`
    /// (no CRC), 9 bytes otherwise.
    pub(super) header_len: usize,
}

/// Parse an ADTS frame header at `buf[0..]`. Returns the parsed header on
/// success. Does NOT validate the CRC even when present — the demux path
/// trusts the upstream PMT routing to point us at AAC bytes; a corrupt
/// stream surfaces as a sync-loss frame downstream.
pub(super) fn parse_adts_header(buf: &[u8]) -> Option<AdtsHeader> {
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
pub(super) fn decode_sample_rate_index(idx: u8) -> Option<u32> {
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
pub(super) fn synthesize_asc(adts: &AdtsHeader) -> [u8; 2] {
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
// AC-3 / E-AC-3 helpers
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Shared PES reassembly for audio PIDs
// ---------------------------------------------------------------------------

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
// Dispatch
// ---------------------------------------------------------------------------

/// Dispatch audio extraction by codec kind from the PMT walk. Per
/// Squad-37: AAC routes through `extract_ts_aac_audio` (Squad-27 path);
/// AC-3 and E-AC-3 route through their respective new extractors.
pub(super) fn extract_ts_audio(
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
