//! Streaming MPEG-TS demuxer (`TsStreamingDemuxer`).
//!
//! Holds the PES reassembly buffer for exactly one in-flight access unit;
//! yields a sample whenever a PUSI=1 packet closes the current one (or at
//! EOF for the final pending sample).

use anyhow::{Context, Result, bail};
use codec::frame::{ColorSpace, PixelFormat, StreamInfo};

use crate::demux::AudioTrack;
use crate::streaming::{DemuxHeader, Sample, StreamingDemuxer};

use super::{
    ProgramInfo, PatProgram,
    STREAM_TYPE_H264, STREAM_TYPE_HEVC, STREAM_TYPE_MPEG2_VIDEO,
    TS_PACKET, TS_SYNC,
};
use super::audio::extract_ts_audio;
use super::framerate::estimate_frame_rate_from_ptses;
use super::pat_pmt::{parse_pat_all_programs, parse_pmt_streams};
use super::pes::{parse_pes_header, scan_first_video_au};

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
    let (packets, packet_stride, prefix_len) = super::detect_packet_layout(&owned)?;
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
            && let Some(payload) = super::ts_psi_payload(pkt)
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
    let mut need: std::collections::HashSet<u16> =
        pat_programs.iter().map(|p| p.pmt_pid).collect();
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
        if let Some(payload) = super::ts_psi_payload(pkt)
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
        Some(au) => {
            codec::pixel_format::detect_dims(&codec, std::slice::from_ref(au)).unwrap_or_else(
                || {
                    tracing::warn!(
                        codec = codec.as_str(),
                        video_pid = video.pid,
                        "TS streaming demux: first AU SPS parse failed; width/height=0×0"
                    );
                    (0, 0)
                },
            )
        }
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
                codec::pixel_format::detect_dims(&codec, std::slice::from_ref(au))
                    .unwrap_or((0, 0))
            }
            None => (0, 0),
        };
        self.header.info.width = w;
        self.header.info.height = h;
        self.header.info.frame_rate =
            estimate_frame_rate_from_ptses(&scan.ptses).unwrap_or(30.0);
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
