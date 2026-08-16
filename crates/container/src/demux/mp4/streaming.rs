/// MP4 / MOV streaming demuxer, fragmented-MP4 sample-table builder, and
/// related walk helpers.
///
/// `demux_mp4_streaming_init` returns an `Mp4StreamingDemuxer` that implements
/// `StreamingDemuxer` — the standard one-sample-at-a-time interface used by
/// the pipeline's streaming migration path (Squad-55 P1).
///
/// `build_fragmented_sample_table` is also called by `demux/audio.rs` (via
/// `super::mp4::build_fragmented_sample_table`) to bypass the mp4 crate's
/// broken `read_sample` on fragmented tracks.
use anyhow::{Context, Result};
use frame::{ColorMetadata, ColorSpace, PixelFormat, StreamInfo};
use mp4::Mp4Reader;
use std::io::Cursor;

use crate::annexb::{NaluCodec, ParamSetTracker, length_prefixed_to_annexb_tracked};
use crate::mp4_sanitize::sanitize_isobmff_box_sizes;
use crate::streaming::{DemuxHeader, Sample, StreamingDemuxer};

use super::super::AudioTrack;
use super::sample_entry::{
    extract_avc_config, extract_hevc_config, has_av01_sample_entry, hevc_sample_entry_fourcc,
    prores_sample_entry_fourcc,
};

// ---------------------------------------------------------------------------
// FragSample — per-sample offset record for fragmented MP4
// ---------------------------------------------------------------------------

/// Per-sample (file_offset, size, pts, duration) record resolved from the
/// moof → traf → trun chain. Fields are `pub(crate)` so `demux/audio.rs`
/// can iterate the returned `Vec<FragSample>` and read the fields after
/// calling `build_fragmented_sample_table` via `super::mp4::…`.
/// (Original visibility was `pub(super)` relative to `demux/mp4.rs`;
/// `pub(crate)` is the minimal widening required when the struct lives in
/// a deeper submodule — external visibility is unchanged since the type is
/// re-exported from `mp4/mod.rs` with `pub(super)`.)
pub(crate) struct FragSample {
    pub(crate) offset: u64,
    pub(crate) size: u32,
    pub(crate) pts_ticks: i64,
    pub(crate) duration_ticks: u32,
}

// ---------------------------------------------------------------------------
// Mp4StreamingDemuxer
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
pub struct Mp4StreamingDemuxer {
    // Owned for the box-tree slice walkers (extract_*); the reader's
    // cursor consumes a clone.
    data: bytes::Bytes,
    reader: Mp4Reader<Cursor<bytes::Bytes>>,
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

pub(crate) fn demux_mp4_streaming_init(data: bytes::Bytes) -> Result<Mp4StreamingDemuxer> {
    // Same lenient pre-pass as `demux_mp4` — see comment there for
    // the iPhone / QuickTime `wave` atom rationale. This one rewrites box
    // sizes, so unlike the other containers MP4 can't share the caller's
    // buffer outright; wrapping the result in `Bytes` at least keeps the
    // probe/reader split below from copying it a second time.
    let owned = bytes::Bytes::from(sanitize_isobmff_box_sizes(&data));
    let size = owned.len() as u64;
    // Build a probe reader against an immutable borrow first — same as
    // legacy `demux_mp4`. This pulls track / codec metadata before we
    // commit the owned buffer to the cursor that backs the streaming
    // reader.
    let probe = Mp4Reader::read_header(Cursor::new(&owned[..]), size)
        .context("reading MP4 header")?;

    let video_track = probe
        .tracks()
        .values()
        .find(|t| t.track_type().ok() == Some(mp4::TrackType::Video))
        .context("no video track in MP4")?;

    let track_id = video_track.track_id();
    let codec_from_mp4 = super::format_codec(video_track);
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
    let frame_rate = super::mp4_frame_rate(video_track, duration);
    let bitrate = video_track.bitrate() as u64;

    let mp4_color = super::super::hdr::extract_mp4_visual_color_metadata(&owned);
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
                None => (super::extract_sps_pps(&probe, track_id), 4u8),
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
            let mut probe_for_pf = Mp4Reader::read_header(Cursor::new(&owned[..]), size)
                .context("re-reading MP4 for pixel-format probe")?;
            match probe_for_pf.read_sample(track_id, 1) {
                Ok(Some(s)) => s.bytes.to_vec(),
                _ => Vec::new(),
            }
        };
        if !detect_input.is_empty() {
            info.pixel_format = frame::pixel_format::detect(&codec, &[detect_input]);
        }
    }

    drop(probe);

    let audio = super::super::audio::extract_mp4_audio(&owned);

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
        header: DemuxHeader {
            codec,
            info,
            timescale: video_track_timescale,
        },
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
            let entry = &table[idx_zero_based];
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
pub(crate) fn build_fragmented_sample_table(
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
