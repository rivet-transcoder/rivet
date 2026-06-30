/// MP4 / MOV box-tree demux, codec detection, AVC/HEVC config extraction,
/// fragmented-MP4 sample-table builder, and the `Mp4StreamingDemuxer`
/// implementation (Squad streaming-migration-55 P1).
use anyhow::{Context, Result};
use codec::frame::{ColorMetadata, ColorSpace, PixelFormat, StreamInfo};
use mp4::Mp4Reader;
use std::io::Cursor;

use crate::annexb::{
    AvcConfig, HevcConfig, NaluCodec, ParamSetTracker, length_prefixed_to_annexb_tracked,
    parse_avcc, parse_hvcc,
};
use crate::mp4_sanitize::sanitize_isobmff_box_sizes;
use crate::streaming::{DemuxHeader, Sample, StreamingDemuxer};

use super::{AudioTrack, DemuxResult};

// ---------------------------------------------------------------------------
// Public demux entry point
// ---------------------------------------------------------------------------

pub fn demux_mp4(data: &[u8]) -> Result<DemuxResult> {
    // Pre-pass to clamp any over-reported child box sizes (common
    // on iPhone-recorded MP4s where the legacy QuickTime `wave`
    // atom inside `mp4a` exposes child boxes whose advertised size
    // exceeds the parent's remaining payload). The sanitizer is
    // byte-identical on clean files, so this is safe to run
    // unconditionally — only malformed files mutate. See
    // `mp4_sanitize::sanitize_isobmff_box_sizes`.
    let sanitized = sanitize_isobmff_box_sizes(data);
    let data: &[u8] = &sanitized;
    let size = data.len() as u64;
    let cursor = Cursor::new(data);
    let reader = Mp4Reader::read_header(cursor, size).context("reading MP4 header")?;

    let video_track = reader
        .tracks()
        .values()
        .find(|t| t.track_type().ok() == Some(mp4::TrackType::Video))
        .context("no video track in MP4")?;

    let track_id = video_track.track_id();
    let codec_from_mp4 = format_codec(video_track);
    // mp4 0.14 has no av01 sample-entry support: tracks using AV1 come back
    // as "unknown" and the decoder factory would fail. Byte-scan stsd for
    // the av01 fourcc to recover the codec label. Sample iteration still
    // works because stco/stsc/stsz are read independently of the sample
    // entry; AV1-in-MP4 samples are raw OBU streams with no AVCC wrapping.
    let codec = if codec_from_mp4 == "unknown" && has_av01_sample_entry(data) {
        "av1".to_string()
    } else if codec_from_mp4 == "unknown" && hevc_sample_entry_fourcc(data).is_some() {
        // hvc1 sample entry — mp4 0.14 only parses hev1. Same length-
        // prefixed bitstream, different fourcc. We retrieve VPS/SPS/PPS
        // from the hvcC box via byte-scan (below) and convert samples
        // to Annex-B the same way as avc1.
        "h265".to_string()
    } else if codec_from_mp4 == "unknown" && prores_sample_entry_fourcc(data).is_some() {
        // Apple ProRes lives in MOV (which is ISOBMFF, same box tree as
        // MP4) under one of six fourccs — mp4 0.14 returns `unknown` for
        // all of them. Samples are stored as self-contained ProRes frames
        // with no AVCC-style length prefix, so stco/stsc/stsz iteration
        // already reads them correctly — we just need the codec label so
        // downstream decode (legacy-cpu-eng's lane) can dispatch.
        "prores".to_string()
    } else {
        codec_from_mp4
    };
    let width = video_track.width() as u32;
    let height = video_track.height() as u32;
    let sample_count = video_track.sample_count();
    let duration = video_track.duration().as_secs_f64();
    let frame_rate = mp4_frame_rate(video_track, duration);
    let bitrate = video_track.bitrate() as u64;

    // Squad-21: pull `mdcv` and `clli` boxes nested inside the visual
    // sample entry (`stsd > {av01, hvc1, hev1, ...}`) and surface them
    // to ColorMetadata so the muxer can round-trip them. These boxes
    // are an HDR10 / HDR10+ requirement — without them, Apple's player
    // (and many TVs) silently fall back to BT.709 limited even when
    // colr nclx says BT.2020.
    let mp4_color = super::hdr::extract_mp4_visual_color_metadata(data);
    let initial_color_metadata = ColorMetadata {
        mastering_display: mp4_color.mastering_display,
        content_light_level: mp4_color.content_light_level,
        ..Default::default()
    };

    let info = StreamInfo {
        codec: codec.clone(),
        width,
        height,
        frame_rate,
        duration,
        pixel_format: PixelFormat::Yuv420p,
        color_space: ColorSpace::Bt709,
        total_frames: sample_count as u64,
        bitrate,
        // SDR defaults for primaries/transfer/matrix at demux layer —
        // those flow from the decoder's sequence_callback (NVDEC) or
        // SPS VUI parser (HEVC CPU). Mastering display + content
        // light level live in MP4 sample-entry boxes (extracted above)
        // so they CAN come from the demuxer directly.
        color_metadata: initial_color_metadata,
    };

    let cursor = Cursor::new(data);
    let mut reader = Mp4Reader::read_header(cursor, size).context("re-reading MP4 for samples")?;

    let mut samples = Vec::with_capacity(sample_count as usize);

    let needs_annexb = matches!(codec.as_str(), "h264" | "h265");
    // length_size defaults to 4 (the ISOBMFF near-universal pick); when
    // we can reach the avcC/hvcC box we override with the recorded value.
    // A length_size of 2 or even 1 is legal and has been observed in
    // streaming-profile MP4s.
    let (sps_pps, length_size) = if needs_annexb {
        if codec == "h264" {
            match extract_avc_config(data) {
                Some(cfg) => (cfg.parameter_sets, cfg.length_size),
                // mp4 0.14 successfully parsed the avcC high-level but we
                // couldn't recover length_size from the box bytes — fall
                // back to the crate's parsed SPS/PPS and assume 4-byte.
                None => (extract_sps_pps(&reader, track_id), 4u8),
            }
        } else {
            // h265: parse hvcC straight from the box bytes (mp4 0.14
            // doesn't surface either length_size or the hvcC arrays).
            match extract_hevc_config(data) {
                Some(cfg) => (cfg.parameter_sets, cfg.length_size),
                None => (Vec::new(), 4u8),
            }
        }
    } else {
        (Vec::new(), 4u8)
    };

    // Per-stream parameter-set emission tracker (#67/#68). Replaces the
    // older `prepend on sample_idx==1` heuristic, which mishandled
    // ExoPlayer open-GOP MP4s where sample 0 is `SPS + non-IDR slice`
    // and the first IRAP arrives later carrying only a slice NAL.
    // The tracker scans inline NAL types per sample and prepends only
    // the parameter-set kinds that are still missing on the first IRAP.
    let mut avc_tracker = if needs_annexb {
        Some(ParamSetTracker::new(if codec == "h264" {
            NaluCodec::Avc
        } else {
            NaluCodec::Hevc
        }))
    } else {
        None
    };

    for sample_idx in 1..=sample_count {
        let sample = reader
            .read_sample(track_id, sample_idx)
            .context("reading sample")?;

        if let Some(sample) = sample {
            let sample_data = sample.bytes.to_vec();

            if let Some(tracker) = avc_tracker.as_mut() {
                let annexb =
                    length_prefixed_to_annexb_tracked(&sample_data, length_size, tracker, &sps_pps);
                samples.push(annexb);
            } else {
                samples.push(sample_data);
            }
        }
    }

    // Replace the hard-coded yuv420p with a real sniff from the first
    // sample's sequence header. detect() is safe on short/malformed
    // data — falls back to Yuv420p.
    let detected_pf = codec::pixel_format::detect(&codec, &samples);
    let info = StreamInfo {
        pixel_format: detected_pf,
        ..info
    };

    let audio = super::audio::extract_mp4_audio(data);

    Ok(DemuxResult {
        codec,
        info,
        samples,
        audio,
    })
}

// ---------------------------------------------------------------------------
// Streaming demuxer
// ---------------------------------------------------------------------------

/// MP4 / MOV streaming demuxer. Owns the input bytes (so its
/// `Mp4Reader<Cursor<Vec<u8>>>` cursor is self-contained) and walks
/// `read_sample(track_id, idx)` one sample at a time. Per-sample
/// AVCC→Annex-B + parameter-set tracking (Squad-14) is preserved.
/// Per-sample location record built when the input is a fragmented
/// MP4. The `mp4` crate (v0.14) returns garbage (typically the bytes
/// of an adjacent `moof` box) from `read_sample` on fragmented inputs
/// — affects BOTH video and audio tracks. Side-stepping `read_sample`
/// for fragmented input by pre-computing sample
/// (file_offset, size, pts, duration) from the moof->traf->trun chain
/// produces correct bytes regardless of track kind. The track filter
/// is `track_id` (parameter on the walker chain) — generic across
/// video/audio/anything else with a track_id.
///
/// Bug history: the audio-extraction path WAS originally claimed to
/// "walk boxes itself" (per a prior comment here) but in fact it
/// called `reader.read_sample(audio_track_id, idx)` — the same buggy
/// path video uses. Burned 2026-05-09: malformed audio segments
/// (8-byte first AU containing the source's `moof` header bytes
/// `00 00 NN NN 6d 6f 6f 66`, every following AU mid-box-tree)
/// passed dedup hash unchanged because they're size-deterministic
/// per source, MSE rejected them with `Number of bands exceeds limit`
/// → SourceBuffer error → MediaSource readyState ended → all video
/// appendBuffer calls failed.
#[derive(Debug, Clone, Copy)]
pub(super) struct FragSample {
    pub(super) offset: u64,
    pub(super) size: u32,
    pub(super) pts_ticks: i64,
    pub(super) duration_ticks: u32,
}

pub struct Mp4StreamingDemuxer {
    // Owned for the box-tree slice walkers (extract_*); the reader's
    // cursor consumes a clone.
    data: Vec<u8>,
    reader: Mp4Reader<Cursor<Vec<u8>>>,
    header: DemuxHeader,
    audio: Option<AudioTrack>,
    track_id: u32,
    sample_count: u32,
    next_idx: u32,
    // For AVC/HEVC: codec-specific config. Empty for the rest.
    sps_pps: Vec<Vec<u8>>,
    length_size: u8,
    tracker: Option<ParamSetTracker>,
    /// `Some` when the input is fragmented MP4. Each entry is a
    /// (file_offset, size, pts, duration) tuple resolved from
    /// moof/traf/trun. `next_video_sample` reads bytes directly from
    /// `self.data` at these offsets instead of going through the mp4
    /// crate's `read_sample`.
    fragmented_samples: Option<Vec<FragSample>>,
}

pub(crate) fn demux_mp4_streaming_init(data: &[u8]) -> Result<Mp4StreamingDemuxer> {
    // Same lenient pre-pass as `demux_mp4` — see comment there for
    // the iPhone / QuickTime `wave` atom rationale.
    let owned = sanitize_isobmff_box_sizes(data);
    let size = owned.len() as u64;
    // Build a probe reader against an immutable borrow first — same as
    // legacy `demux_mp4`. This pulls track / codec metadata before we
    // commit the owned buffer to the cursor that backs the streaming
    // reader.
    let probe = Mp4Reader::read_header(Cursor::new(owned.as_slice()), size)
        .context("reading MP4 header")?;

    let video_track = probe
        .tracks()
        .values()
        .find(|t| t.track_type().ok() == Some(mp4::TrackType::Video))
        .context("no video track in MP4")?;

    let track_id = video_track.track_id();
    let codec_from_mp4 = format_codec(video_track);
    let codec = if codec_from_mp4 == "unknown" && has_av01_sample_entry(&owned) {
        "av1".to_string()
    } else if codec_from_mp4 == "unknown" && hevc_sample_entry_fourcc(&owned).is_some() {
        "h265".to_string()
    } else if codec_from_mp4 == "unknown" && prores_sample_entry_fourcc(&owned).is_some() {
        "prores".to_string()
    } else {
        codec_from_mp4
    };
    let width = video_track.width() as u32;
    let height = video_track.height() as u32;
    let sample_count = video_track.sample_count();
    let duration = video_track.duration().as_secs_f64();
    let video_track_timescale = video_track.timescale();
    let frame_rate = mp4_frame_rate(video_track, duration);
    let bitrate = video_track.bitrate() as u64;

    let mp4_color = super::hdr::extract_mp4_visual_color_metadata(&owned);
    let initial_color_metadata = ColorMetadata {
        mastering_display: mp4_color.mastering_display,
        content_light_level: mp4_color.content_light_level,
        ..Default::default()
    };

    let mut info = StreamInfo {
        codec: codec.clone(),
        width,
        height,
        frame_rate,
        duration,
        pixel_format: PixelFormat::Yuv420p,
        color_space: ColorSpace::Bt709,
        total_frames: sample_count as u64,
        bitrate,
        color_metadata: initial_color_metadata,
    };

    let needs_annexb = matches!(codec.as_str(), "h264" | "h265");
    let (sps_pps, length_size) = if needs_annexb {
        if codec == "h264" {
            match extract_avc_config(&owned) {
                Some(cfg) => (cfg.parameter_sets, cfg.length_size),
                None => (extract_sps_pps(&probe, track_id), 4u8),
            }
        } else {
            match extract_hevc_config(&owned) {
                Some(cfg) => (cfg.parameter_sets, cfg.length_size),
                None => (Vec::new(), 4u8),
            }
        }
    } else {
        (Vec::new(), 4u8)
    };

    // Pixel-format detection needs the SPS / sequence header. For hvc1 / avc1
    // the parameter sets live in the sample entry (`sps_pps`), NOT the first
    // VCL sample — detecting on the raw sample alone silently reports 8-bit for
    // a 10-bit Main 10 / Hi10P source, which then mis-sizes the encoder. Detect
    // on the parameter sets (Annex-B) when present; fall back to the first
    // sample for hev1 / avc3 (in-band) and AV1 / VP9 (sequence header in band).
    if sample_count > 0 {
        let detect_input: Vec<u8> = if !sps_pps.is_empty() {
            let mut buf = Vec::new();
            for ps in &sps_pps {
                buf.extend_from_slice(&[0, 0, 0, 1]);
                buf.extend_from_slice(ps);
            }
            buf
        } else {
            let mut probe_for_pf = Mp4Reader::read_header(Cursor::new(owned.as_slice()), size)
                .context("re-reading MP4 for pixel-format probe")?;
            match probe_for_pf.read_sample(track_id, 1) {
                Ok(Some(s)) => s.bytes.to_vec(),
                _ => Vec::new(),
            }
        };
        if !detect_input.is_empty() {
            info.pixel_format = codec::pixel_format::detect(&codec, &[detect_input]);
        }
    }

    drop(probe);

    let audio = super::audio::extract_mp4_audio(&owned);

    // Build the streaming reader against an owned cursor.
    let reader_cursor = Cursor::new(owned.clone());
    let reader =
        Mp4Reader::read_header(reader_cursor, size).context("opening MP4 streaming reader")?;

    let tracker = if needs_annexb {
        Some(ParamSetTracker::new(if codec == "h264" {
            NaluCodec::Avc
        } else {
            NaluCodec::Hevc
        }))
    } else {
        None
    };

    let _ = needs_annexb; // tracker presence reflects this

    // Detect fragmented MP4 + build a sample table from moof/traf/trun
    // when applicable. The mp4 crate's `read_sample` returns garbage
    // (typically the bytes of an adjacent moof box header) for any
    // fragmented track regardless of kind, so for fragmented input
    // we bypass `read_sample` entirely and read sample bytes directly
    // from `owned` at the offsets in this table. `extract_mp4_audio`
    // does the same against its own `data` slice.
    let fragmented_samples = build_fragmented_sample_table(&owned, track_id, 0, 0).map(|table| {
        tracing::info!(
            track_id,
            sample_count = table.len(),
            "fragmented MP4 detected; built sample table from moof/traf/trun"
        );
        table
    });
    let final_sample_count = match &fragmented_samples {
        Some(table) => table.len() as u32,
        None => sample_count,
    };

    // Recompute frame_rate + duration from fragmented sample timestamps
    // when (a) we built a fragmented sample table AND (b) the static
    // moov sample table was empty or had a zero duration. Pure
    // fragmented MP4 — common from web recorders, screen capture
    // tools, and modern phone exports — leaves moov with no static
    // samples + tkhd.duration=0; the previous fallback was the 30.0
    // sentinel, which silently encoded a 24-fps VFR source as 30-fps
    // CFR and produced ~20% short output. The fragmented sample
    // table's actual duration_ticks (from moof.traf.trun per-sample
    // duration entries) carries the truth. Trust the static table
    // when it's populated — that path was correct already.
    if let Some(table) = fragmented_samples.as_ref() {
        if !table.is_empty() && (sample_count == 0 || duration <= 0.0) && video_track_timescale > 0
        {
            let total_ticks: u64 = table.iter().map(|s| s.duration_ticks as u64).sum();
            if total_ticks > 0 {
                let total_seconds = total_ticks as f64 / video_track_timescale as f64;
                if total_seconds > 0.0 {
                    let avg_fps = table.len() as f64 / total_seconds;
                    info.frame_rate = avg_fps.clamp(1.0, 240.0);
                    info.duration = total_seconds;
                    info.total_frames = table.len() as u64;
                    tracing::info!(
                        track_id,
                        avg_fps,
                        total_seconds,
                        sample_count = table.len(),
                        timescale = video_track_timescale,
                        "fragmented MP4: recomputed frame_rate + duration from \
                         moof/traf/trun timestamps (static moov sample table \
                         was empty)"
                    );
                }
            }
        }
    }
    Ok(Mp4StreamingDemuxer {
        data: owned,
        reader,
        header: DemuxHeader { codec, info },
        audio,
        track_id,
        sample_count: final_sample_count,
        next_idx: 1,
        sps_pps,
        length_size,
        tracker,
        fragmented_samples,
    })
}

impl StreamingDemuxer for Mp4StreamingDemuxer {
    fn header(&self) -> &DemuxHeader {
        &self.header
    }

    fn next_video_sample(&mut self) -> Result<Option<Sample>> {
        // Fragmented MP4 path: pull bytes directly from the input buffer
        // at the offsets we resolved at init time.
        if let Some(table) = self.fragmented_samples.as_ref() {
            let idx_zero_based = (self.next_idx - 1) as usize;
            if idx_zero_based >= table.len() {
                return Ok(None);
            }
            self.next_idx += 1;
            let entry = table[idx_zero_based];
            let off = entry.offset as usize;
            let end = off.saturating_add(entry.size as usize);
            if end > self.data.len() {
                tracing::warn!(
                    idx = idx_zero_based + 1,
                    offset = entry.offset,
                    size = entry.size,
                    data_len = self.data.len(),
                    "fragmented sample reaches past EOF; stopping at the previous frame"
                );
                return Ok(None);
            }
            let raw = self.data[off..end].to_vec();
            let data = if let Some(tracker) = self.tracker.as_mut() {
                length_prefixed_to_annexb_tracked(&raw, self.length_size, tracker, &self.sps_pps)
            } else {
                raw
            };
            return Ok(Some(Sample {
                data,
                pts_ticks: entry.pts_ticks,
                duration_ticks: entry.duration_ticks,
            }));
        }
        loop {
            if self.next_idx > self.sample_count {
                return Ok(None);
            }
            let idx = self.next_idx;
            self.next_idx += 1;
            // Mirror the audio-track tolerance in `extract_mp4_audio`:
            // when a mid-track read_sample fails on a fragmented MP4
            // with a truncated `traf.trun` index — the typical iPhone /
            // Android broken-recording shape — surface a warn and
            // signal soft EOF to the encode loop. The frames that DID
            // demux upstream still flow through, the encoder produces
            // an AV1 sequence header from the first one, and the CMAF
            // muxer's `finalize` writes a valid (truncated) init
            // segment. Without this, a single missing trun entry
            // halfway through a clip would propagate as `TranscodeFailure`
            // for the whole job — the symptom we hit 2026-05-08.
            let s = match self.reader.read_sample(self.track_id, idx) {
                Ok(s) => s,
                Err(e) => {
                    tracing::warn!(
                        track_id = self.track_id,
                        idx,
                        emitted = idx.saturating_sub(1),
                        sample_count = self.sample_count,
                        error = %e,
                        "video stream: read_sample error mid-track; \
                         stopping at sample {} of {} (truncated source — \
                         iPhone fragmented MP4 with a missing trun entry \
                         is the typical cause)",
                        idx.saturating_sub(1),
                        self.sample_count,
                    );
                    return Ok(None);
                }
            };
            let Some(sample) = s else { continue };
            let pts_ticks = sample.start_time as i64;
            let duration_ticks = sample.duration;
            let raw = sample.bytes.to_vec();
            let data = if let Some(tracker) = self.tracker.as_mut() {
                length_prefixed_to_annexb_tracked(&raw, self.length_size, tracker, &self.sps_pps)
            } else {
                raw
            };
            return Ok(Some(Sample {
                data,
                pts_ticks,
                duration_ticks,
            }));
        }
    }

    fn audio(&self) -> Option<&AudioTrack> {
        self.audio.as_ref()
    }
}

impl Mp4StreamingDemuxer {
    /// For tests + the legacy `demux()` adapter: reach back at the
    /// owned input bytes (e.g. for an opt-in re-probe).
    #[allow(dead_code)]
    pub(crate) fn raw_bytes(&self) -> &[u8] {
        &self.data
    }
}

// ---------------------------------------------------------------------------
// Fragmented MP4 sample table builder
// ---------------------------------------------------------------------------

/// Walk top-level `moof` boxes in `data`, gather per-sample
/// (file_offset, size, pts, duration) tuples for the track id matching
/// `track_id` (works for video, audio, or any other track kind).
/// Returns `Some(table)` when the input is fragmented (at least one
/// top-level moof exists), `None` otherwise. An empty `Some(vec![])`
/// means "fragmented, but this track id had no samples in any moof"
/// — that's distinct from non-fragmented (None) and the caller
/// shouldn't fall back to `read_sample` in that case (it'd return
/// the same garbage bytes that prompted the fragmented path in the
/// first place).
///
/// Best-effort: silently skips moofs / trafs / truns that don't parse,
/// or that reference unknown tracks. Each successfully-walked trun
/// contributes its samples in order so the resulting Vec is decode-
/// order across the file.
pub(super) fn build_fragmented_sample_table(
    data: &[u8],
    track_id: u32,
    default_sample_duration_from_trex: u32,
    default_sample_size_from_trex: u32,
) -> Option<Vec<FragSample>> {
    let mut samples: Vec<FragSample> = Vec::new();
    let mut pos: usize = 0;
    let mut accumulated_pts: i64 = 0;
    let mut found_any_moof = false;

    while pos + 8 <= data.len() {
        let box_size_field = u32::from_be_bytes(data[pos..pos + 4].try_into().ok()?);
        let box_type = &data[pos + 4..pos + 8];
        let (box_size, header_size): (usize, usize) = if box_size_field == 1 {
            // 64-bit largesize form.
            if pos + 16 > data.len() {
                break;
            }
            let big = u64::from_be_bytes(data[pos + 8..pos + 16].try_into().ok()?);
            (big as usize, 16)
        } else if box_size_field == 0 {
            // box extends to EOF — stop walking after this one.
            (data.len() - pos, 8)
        } else {
            (box_size_field as usize, 8)
        };
        if box_size < header_size || pos + box_size > data.len() {
            break;
        }

        if box_type == b"moof" {
            found_any_moof = true;
            let moof_start = pos;
            let moof_end = pos + box_size;
            walk_moof(
                data,
                moof_start + header_size,
                moof_end,
                moof_start as u64,
                track_id,
                default_sample_duration_from_trex,
                default_sample_size_from_trex,
                &mut accumulated_pts,
                &mut samples,
            );
        }
        pos = pos
            .checked_add(box_size)
            .filter(|&n| n <= data.len())
            .unwrap_or(data.len());
    }

    if found_any_moof { Some(samples) } else { None }
}

#[allow(clippy::too_many_arguments)]
fn walk_moof(
    data: &[u8],
    children_start: usize,
    moof_end: usize,
    moof_offset: u64,
    track_id: u32,
    default_sample_duration_from_trex: u32,
    default_sample_size_from_trex: u32,
    accumulated_pts: &mut i64,
    samples: &mut Vec<FragSample>,
) {
    let mut pos = children_start;
    while pos + 8 <= moof_end {
        let size = u32::from_be_bytes(match data[pos..pos + 4].try_into() {
            Ok(b) => b,
            Err(_) => break,
        });
        let typ = &data[pos + 4..pos + 8];
        if size == 0 || size as usize + pos > moof_end {
            break;
        }
        if typ == b"traf" {
            walk_traf(
                data,
                pos + 8,
                pos + size as usize,
                moof_offset,
                track_id,
                default_sample_duration_from_trex,
                default_sample_size_from_trex,
                accumulated_pts,
                samples,
            );
        }
        pos += size as usize;
    }
}

#[allow(clippy::too_many_arguments)]
fn walk_traf(
    data: &[u8],
    children_start: usize,
    traf_end: usize,
    moof_offset: u64,
    track_id: u32,
    default_sample_duration_from_trex: u32,
    default_sample_size_from_trex: u32,
    accumulated_pts: &mut i64,
    samples: &mut Vec<FragSample>,
) {
    // First pass: find tfhd (always first child of traf per spec) +
    // collect tfhd-derived defaults + base_data_offset semantics.
    let mut this_track: Option<u32> = None;
    let mut tfhd_default_sample_duration: u32 = default_sample_duration_from_trex;
    let mut tfhd_default_sample_size: u32 = default_sample_size_from_trex;
    let mut base_data_offset: u64 = moof_offset; // default-base-is-moof
    let mut base_data_offset_explicit = false;
    let mut tfdt_base_pts: Option<i64> = None;

    let mut pos = children_start;
    while pos + 8 <= traf_end {
        let size = u32::from_be_bytes(match data[pos..pos + 4].try_into() {
            Ok(b) => b,
            Err(_) => break,
        });
        let typ = &data[pos + 4..pos + 8];
        if size == 0 || size as usize + pos > traf_end {
            break;
        }
        if typ == b"tfhd" {
            // tfhd: u8 version + u24 flags + u32 track_id + optional fields per flag bits
            if pos + 16 > traf_end {
                pos += size as usize;
                continue;
            }
            let flags = u32::from_be_bytes(match data[pos + 8..pos + 12].try_into() {
                Ok(b) => b,
                Err(_) => break,
            }) & 0x00ff_ffff;
            let tk = u32::from_be_bytes(match data[pos + 12..pos + 16].try_into() {
                Ok(b) => b,
                Err(_) => break,
            });
            this_track = Some(tk);
            let mut p = pos + 16;
            // base_data_offset_present
            if flags & 0x01 != 0 {
                if p + 8 > traf_end {
                    break;
                }
                base_data_offset = u64::from_be_bytes(match data[p..p + 8].try_into() {
                    Ok(b) => b,
                    Err(_) => break,
                });
                base_data_offset_explicit = true;
                p += 8;
            }
            // sample_description_index_present
            if flags & 0x02 != 0 {
                p += 4;
            }
            // default_sample_duration_present
            if flags & 0x08 != 0 {
                if p + 4 > traf_end {
                    break;
                }
                tfhd_default_sample_duration =
                    u32::from_be_bytes(match data[p..p + 4].try_into() {
                        Ok(b) => b,
                        Err(_) => break,
                    });
                p += 4;
            }
            // default_sample_size_present
            if flags & 0x10 != 0 {
                if p + 4 > traf_end {
                    break;
                }
                tfhd_default_sample_size = u32::from_be_bytes(match data[p..p + 4].try_into() {
                    Ok(b) => b,
                    Err(_) => break,
                });
                p += 4;
            }
            // default_sample_flags_present (skip 4 bytes)
            if flags & 0x20 != 0 {
                p += 4;
            }
            // default-base-is-moof flag: when set AND base_data_offset
            // not present, base is the moof start (which is our default).
            let _ = p;
        } else if typ == b"tfdt" {
            // tfdt: version u8 + flags u24 + base_media_decode_time (u32 v0 / u64 v1)
            if pos + 12 > traf_end {
                pos += size as usize;
                continue;
            }
            let version = data[pos + 8];
            if version == 1 {
                if pos + 20 > traf_end {
                    pos += size as usize;
                    continue;
                }
                let bmdt =
                    u64::from_be_bytes(data[pos + 12..pos + 20].try_into().unwrap_or([0; 8]));
                tfdt_base_pts = Some(bmdt as i64);
            } else {
                let bmdt =
                    u32::from_be_bytes(data[pos + 12..pos + 16].try_into().unwrap_or([0; 4]));
                tfdt_base_pts = Some(bmdt as i64);
            }
        }
        pos += size as usize;
    }

    let Some(tk) = this_track else {
        return;
    };
    if tk != track_id {
        return;
    }

    if let Some(bp) = tfdt_base_pts {
        *accumulated_pts = bp;
    }

    // Second pass: walk trun boxes in declaration order.
    let mut pos = children_start;
    while pos + 8 <= traf_end {
        let size = u32::from_be_bytes(match data[pos..pos + 4].try_into() {
            Ok(b) => b,
            Err(_) => break,
        });
        let typ = &data[pos + 4..pos + 8];
        if size == 0 || size as usize + pos > traf_end {
            break;
        }
        if typ == b"trun" {
            walk_trun(
                data,
                pos + 8,
                pos + size as usize,
                if base_data_offset_explicit {
                    base_data_offset
                } else {
                    moof_offset
                },
                tfhd_default_sample_duration,
                tfhd_default_sample_size,
                accumulated_pts,
                samples,
            );
        }
        pos += size as usize;
    }
    let _ = base_data_offset_explicit;
}

#[allow(clippy::too_many_arguments)]
fn walk_trun(
    data: &[u8],
    children_start: usize,
    trun_end: usize,
    base_offset: u64,
    default_sample_duration: u32,
    default_sample_size: u32,
    accumulated_pts: &mut i64,
    samples: &mut Vec<FragSample>,
) {
    if children_start + 8 > trun_end {
        return;
    }
    let version = data[children_start];
    let flags = u32::from_be_bytes(match data[children_start..children_start + 4].try_into() {
        Ok(b) => b,
        Err(_) => return,
    }) & 0x00ff_ffff;
    let sample_count = u32::from_be_bytes(
        match data[children_start + 4..children_start + 8].try_into() {
            Ok(b) => b,
            Err(_) => return,
        },
    );
    let mut p = children_start + 8;
    let mut data_offset_in_trun: i32 = 0;
    if flags & 0x000_001 != 0 {
        if p + 4 > trun_end {
            return;
        }
        data_offset_in_trun = i32::from_be_bytes(match data[p..p + 4].try_into() {
            Ok(b) => b,
            Err(_) => return,
        });
        p += 4;
    }
    if flags & 0x000_004 != 0 {
        // first-sample-flags-present: skip 4 bytes
        p += 4;
    }

    let sample_duration_present = flags & 0x000_100 != 0;
    let sample_size_present = flags & 0x000_200 != 0;
    let sample_flags_present = flags & 0x000_400 != 0;
    let sample_cto_present = flags & 0x000_800 != 0;

    let mut current_offset = base_offset.wrapping_add(data_offset_in_trun as u64);
    for _ in 0..sample_count {
        let dur = if sample_duration_present {
            if p + 4 > trun_end {
                return;
            }
            let d = u32::from_be_bytes(match data[p..p + 4].try_into() {
                Ok(b) => b,
                Err(_) => return,
            });
            p += 4;
            d
        } else {
            default_sample_duration
        };
        let sz = if sample_size_present {
            if p + 4 > trun_end {
                return;
            }
            let s = u32::from_be_bytes(match data[p..p + 4].try_into() {
                Ok(b) => b,
                Err(_) => return,
            });
            p += 4;
            s
        } else {
            default_sample_size
        };
        if sample_flags_present {
            p += 4;
        }
        let cto: i32 = if sample_cto_present {
            if p + 4 > trun_end {
                return;
            }
            let c = if version == 0 {
                u32::from_be_bytes(match data[p..p + 4].try_into() {
                    Ok(b) => b,
                    Err(_) => return,
                }) as i32
            } else {
                i32::from_be_bytes(match data[p..p + 4].try_into() {
                    Ok(b) => b,
                    Err(_) => return,
                })
            };
            p += 4;
            c
        } else {
            0
        };

        if sz > 0 {
            samples.push(FragSample {
                offset: current_offset,
                size: sz,
                pts_ticks: accumulated_pts.saturating_add(cto as i64),
                duration_ticks: dur,
            });
        }
        current_offset = current_offset.saturating_add(sz as u64);
        *accumulated_pts = accumulated_pts.saturating_add(dur as i64);
    }
}

// ---------------------------------------------------------------------------
// Sample-entry detection helpers
// ---------------------------------------------------------------------------

/// Walk the ISOBMFF box tree looking for an `av01` sample entry inside
/// `moov/trak/mdia/minf/stbl/stsd`. Returns true if found at the expected
/// nesting level. Doing a full tree walk (vs naive byte-search for "av01")
/// avoids false positives from sample data in mdat that happens to contain
/// those bytes.
pub(super) fn has_av01_sample_entry(data: &[u8]) -> bool {
    let path: &[&[u8; 4]] = &[b"moov", b"trak", b"mdia", b"minf", b"stbl", b"stsd"];
    let Some(stsd_body) = super::find_box_body(data, path) else {
        return false;
    };
    if stsd_body.len() < 16 {
        return false;
    }
    let mut pos = 8; // skip version/flags/entry_count
    while pos + 8 <= stsd_body.len() {
        let entry_size = u32::from_be_bytes([
            stsd_body[pos],
            stsd_body[pos + 1],
            stsd_body[pos + 2],
            stsd_body[pos + 3],
        ]) as usize;
        if entry_size == 0 {
            break;
        }
        if pos + 4 < stsd_body.len() && &stsd_body[pos + 4..pos + 8] == b"av01" {
            return true;
        }
        pos = pos.saturating_add(entry_size);
    }
    false
}

/// Find the HEVC sample-entry fourcc (`hvc1`, `hev1`, `hvc2`, `hev2`,
/// `dvh1`, `dvhe`) in the video track's stsd box. Returns the 4-byte
/// fourcc or None. Used as the mp4 0.14 crate detection fallback —
/// its `media_type()` only returns H265 for `hev1`, so `hvc1` (the
/// Jellyfin corpus's HEVC flavor) needs this path.
fn hevc_sample_entry_fourcc(data: &[u8]) -> Option<[u8; 4]> {
    let path: &[&[u8; 4]] = &[b"moov", b"trak", b"mdia", b"minf", b"stbl", b"stsd"];
    let stsd_body = super::find_box_body(data, path)?;
    if stsd_body.len() < 16 {
        return None;
    }
    let mut pos = 8; // skip version/flags/entry_count
    while pos + 8 <= stsd_body.len() {
        let entry_size = u32::from_be_bytes([
            stsd_body[pos],
            stsd_body[pos + 1],
            stsd_body[pos + 2],
            stsd_body[pos + 3],
        ]) as usize;
        let entry_type: [u8; 4] = stsd_body[pos + 4..pos + 8].try_into().ok()?;
        match &entry_type {
            b"hvc1" | b"hev1" | b"hvc2" | b"hev2" | b"dvh1" | b"dvhe" => {
                return Some(entry_type);
            }
            _ => {}
        }
        if entry_size == 0 {
            break;
        }
        pos = pos.saturating_add(entry_size);
    }
    None
}

/// Look for an Apple ProRes sample entry in the video track's stsd box.
/// Six fourccs cover the product family:
///   apcn = ProRes 422 Standard    apch = ProRes 422 HQ
///   apcs = ProRes 422 LT          apco = ProRes 422 Proxy
///   ap4h = ProRes 4444            ap4x = ProRes 4444 XQ
/// All share the same container layout (self-contained frame samples, no
/// length-prefix wrapping), so from demux's perspective they are
/// interchangeable — we return the first one we see so callers can log
/// which specific profile the input used. Decode dispatch uses the
/// unified `"prores"` codec label produced by `demux_mp4`.
pub(super) fn prores_sample_entry_fourcc(data: &[u8]) -> Option<[u8; 4]> {
    let path: &[&[u8; 4]] = &[b"moov", b"trak", b"mdia", b"minf", b"stbl", b"stsd"];
    let stsd_body = super::find_box_body(data, path)?;
    if stsd_body.len() < 16 {
        return None;
    }
    let mut pos = 8;
    while pos + 8 <= stsd_body.len() {
        let entry_size = u32::from_be_bytes([
            stsd_body[pos],
            stsd_body[pos + 1],
            stsd_body[pos + 2],
            stsd_body[pos + 3],
        ]) as usize;
        let entry_type: [u8; 4] = stsd_body[pos + 4..pos + 8].try_into().ok()?;
        match &entry_type {
            b"apcn" | b"apch" | b"apcs" | b"apco" | b"ap4h" | b"ap4x" => {
                return Some(entry_type);
            }
            _ => {}
        }
        if entry_size == 0 {
            break;
        }
        pos = pos.saturating_add(entry_size);
    }
    None
}

/// Find the AVC sample entry in MP4 and return its parsed avcC config
/// (length_size + SPS/PPS NAL units). Returns None when no `avc1`/`avc3`
/// sample entry is present or the avcC box is malformed.
fn extract_avc_config(data: &[u8]) -> Option<AvcConfig> {
    let path: &[&[u8; 4]] = &[b"moov", b"trak", b"mdia", b"minf", b"stbl", b"stsd"];
    let stsd_body = super::find_box_body(data, path)?;
    if stsd_body.len() < 16 {
        return None;
    }

    let mut pos = 8;
    while pos + 8 <= stsd_body.len() {
        let entry_size = u32::from_be_bytes([
            stsd_body[pos],
            stsd_body[pos + 1],
            stsd_body[pos + 2],
            stsd_body[pos + 3],
        ]) as usize;
        let entry_type = &stsd_body[pos + 4..pos + 8];
        let is_avc = matches!(entry_type, b"avc1" | b"avc3");
        if !is_avc {
            if entry_size == 0 {
                break;
            }
            pos = pos.saturating_add(entry_size);
            continue;
        }
        let end = pos.saturating_add(entry_size);
        if end > stsd_body.len() {
            return None;
        }
        let child_start = pos + 8 + 78; // VisualSampleEntry fixed header
        if child_start >= end {
            return None;
        }
        let avcc = super::find_direct_child(&stsd_body[child_start..end], b"avcC")?;
        return parse_avcc(avcc);
    }
    None
}

fn extract_hevc_config(data: &[u8]) -> Option<HevcConfig> {
    let path: &[&[u8; 4]] = &[b"moov", b"trak", b"mdia", b"minf", b"stbl", b"stsd"];
    let stsd_body = super::find_box_body(data, path)?;
    if stsd_body.len() < 16 {
        return None;
    }
    let mut pos = 8;
    while pos + 8 <= stsd_body.len() {
        let entry_size = u32::from_be_bytes([
            stsd_body[pos],
            stsd_body[pos + 1],
            stsd_body[pos + 2],
            stsd_body[pos + 3],
        ]) as usize;
        let entry_type = &stsd_body[pos + 4..pos + 8];
        let is_hevc = matches!(
            entry_type,
            b"hvc1" | b"hev1" | b"hvc2" | b"hev2" | b"dvh1" | b"dvhe"
        );
        if !is_hevc {
            if entry_size == 0 {
                break;
            }
            pos = pos.saturating_add(entry_size);
            continue;
        }
        let end = pos.saturating_add(entry_size);
        if end > stsd_body.len() {
            return None;
        }
        let child_start = pos + 8 + 78; // VisualSampleEntry fixed header
        if child_start >= end {
            return None;
        }
        let hvcc = super::find_direct_child(&stsd_body[child_start..end], b"hvcC")?;
        return parse_hvcc(hvcc);
    }
    None
}

#[allow(dead_code)]
fn extract_hevc_parameter_sets(data: &[u8]) -> Vec<Vec<u8>> {
    extract_hevc_config(data)
        .map(|cfg| cfg.parameter_sets)
        .unwrap_or_default()
}

/// Parse the SPS/PPS parameter sets out of an avcC box (as a `Vec<Vec<u8>>`
/// of raw NAL units without start codes). Used by tests and as the fallback
/// when `extract_avc_config` is unavailable. Returns an empty Vec on any
/// parse failure — callers must tolerate that.
#[allow(dead_code)]
pub(super) fn parse_avcc_param_sets(avcc: &[u8]) -> Vec<Vec<u8>> {
    parse_avcc(avcc)
        .map(|cfg| cfg.parameter_sets)
        .unwrap_or_default()
}

#[allow(dead_code)]
fn parse_hvcc_param_sets(hvcc: &[u8]) -> Vec<Vec<u8>> {
    parse_hvcc(hvcc)
        .map(|cfg| cfg.parameter_sets)
        .unwrap_or_default()
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

fn format_codec(track: &mp4::Mp4Track) -> String {
    match track.media_type() {
        Ok(mp4::MediaType::H264) => "h264".into(),
        Ok(mp4::MediaType::H265) => "h265".into(),
        Ok(mp4::MediaType::VP9) => "vp9".into(),
        _ => "unknown".into(),
    }
}

fn mp4_frame_rate(track: &mp4::Mp4Track, duration: f64) -> f64 {
    let stts = &track.trak.mdia.minf.stbl.stts;
    if stts.entries.len() == 1 && stts.entries[0].sample_delta > 0 {
        return track.timescale() as f64 / stts.entries[0].sample_delta as f64;
    }
    if duration > 0.0 {
        track.sample_count() as f64 / duration
    } else {
        30.0
    }
}

fn extract_sps_pps(reader: &Mp4Reader<Cursor<&[u8]>>, track_id: u32) -> Vec<Vec<u8>> {
    let mut nalus = Vec::new();
    if let Some(track) = reader.tracks().get(&track_id)
        && let Some(ref avc1) = track.trak.mdia.minf.stbl.stsd.avc1
    {
        for sps in &avc1.avcc.sequence_parameter_sets {
            nalus.push(sps.bytes.to_vec());
        }
        for pps in &avc1.avcc.picture_parameter_sets {
            nalus.push(pps.bytes.to_vec());
        }
    }
    nalus
}
