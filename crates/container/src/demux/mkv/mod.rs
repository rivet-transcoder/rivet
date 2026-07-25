/// Matroska / WebM demux, Colour element parsing, EBML raw scanner, and the
/// `MkvStreamingDemuxer` implementation (Squad streaming-migration-55 P1).

mod colour;
mod ebml;

use anyhow::{Context, Result, bail};
use codec::frame::{ColorMetadata, ColorSpace, ContentLightLevel, PixelFormat, StreamInfo};
use matroska_demuxer::{Frame as MkvFrame, MatroskaFile, TrackType as MkvTrackType};
use std::io::Cursor;

use crate::annexb::{
    NaluCodec, ParamSetTracker, length_prefixed_to_annexb_tracked, parse_avcc, parse_hvcc,
};
use crate::streaming::{DemuxHeader, Sample, StreamingDemuxer};
use crate::MkvColorInfo;

use super::subtitle::{extract_mkv_subtitles, SubtitleTrack};
use super::{AudioTrack, DemuxResult};

use colour::{bitrate_from_tags, colour_to_pipeline};
use ebml::scan_mkv_colour_raw;

// Re-export the two VInt readers that `demux/tests.rs` pulls directly as
// `super::mkv::{read_id_vint, read_size_vint}`.
#[allow(unused_imports)] // used only by demux/tests.rs under #[cfg(test)]
pub(crate) use ebml::{read_id_vint, read_size_vint};

// ---------------------------------------------------------------------------
// Public demux entry point
// ---------------------------------------------------------------------------

pub fn demux_mkv(data: &[u8]) -> Result<DemuxResult> {
    let cursor = Cursor::new(data);
    let mut mkv =
        MatroskaFile::open(cursor).map_err(|e| anyhow::anyhow!("reading MKV header: {e}"))?;

    // AVC/HEVC in MKV: CodecPrivate holds the avcC / hvcC configuration record
    // verbatim. Length-prefixed Block samples need the same Annex-B conversion
    // we do for MP4, plus VPS/SPS/PPS prepended to the first sample of the
    // track. VP8/VP9/AV1 are self-contained and skip this dance.
    //
    // Snapshot every field we need off TrackEntry before `next_frame` starts
    // mutating `mkv` below — TrackEntry borrows from `mkv` and hold times
    // conflict with the &mut self on `next_frame`.
    let (
        track_number,
        track_uid,
        codec_id,
        width,
        height,
        annexb_prepend,
        length_size,
        color_space,
        mut color_metadata,
        mut color_info,
        track_default_duration_ns,
    ) = {
        let track_info = mkv
            .tracks()
            .iter()
            .find(|t| t.track_type() == MkvTrackType::Video)
            .context("no video track in MKV")?;

        let track_number = track_info.track_number().get();
        let track_uid = track_info.track_uid().get();
        let codec_id = track_info.codec_id().to_string();
        // Per-track DefaultDuration (`0x23E383`, ns per frame) — Matroska's
        // canonical frame-rate hint. Used as the frame_rate fallback when the
        // segment's `Duration` element is absent (live-recorded MKVs and some
        // streaming WebMs ship without one). Squad-32: this fallback was
        // previously missing — frame_rate would silently default to 30.0
        // even when DefaultDuration cleanly described e.g. 23.976 / 60 fps.
        let default_duration_ns = track_info.default_duration().map(|d| d.get());

        // Parse avcC/hvcC CodecPrivate once to recover both the parameter
        // sets and the recorded length_size_minus_one — 4-byte prefixes
        // are the common case, but the spec allows 1 or 2 bytes.
        let (annexb_prepend, length_size): (Vec<Vec<u8>>, u8) = if codec_id == "V_MPEG4/ISO/AVC" {
            let priv_bytes = track_info
                .codec_private()
                .context("V_MPEG4/ISO/AVC CodecPrivate missing")?;
            let cfg = parse_avcc(priv_bytes).context("V_MPEG4/ISO/AVC CodecPrivate malformed")?;
            (cfg.parameter_sets, cfg.length_size)
        } else if codec_id == "V_MPEGH/ISO/HEVC" {
            let priv_bytes = track_info
                .codec_private()
                .context("V_MPEGH/ISO/HEVC CodecPrivate missing")?;
            let cfg = parse_hvcc(priv_bytes).context("V_MPEGH/ISO/HEVC CodecPrivate malformed")?;
            (cfg.parameter_sets, cfg.length_size)
        } else {
            (Vec::new(), 4)
        };

        if mkv_codec_needs_annexb(&codec_id) && annexb_prepend.is_empty() {
            bail!("AVC/HEVC MKV CodecPrivate missing or empty — no parameter sets to prepend");
        }

        let video = track_info
            .video()
            .context("video track missing Video element")?;
        let w = video.pixel_width().get() as u32;
        let h = video.pixel_height().get() as u32;

        // Parse the Colour element into a ColorMetadata + ColorSpace +
        // extended MkvColorInfo. Legacy MKVs without Colour produce the
        // SDR BT.709 default.
        let (color_space, color_metadata, color_info) = match video.colour() {
            Some(colour) => colour_to_pipeline(colour),
            None => (
                ColorSpace::Bt709,
                ColorMetadata::default(),
                MkvColorInfo::default(),
            ),
        };

        (
            track_number,
            track_uid,
            codec_id,
            w,
            h,
            annexb_prepend,
            length_size,
            color_space,
            color_metadata,
            color_info,
            default_duration_ns,
        )
    };

    // Squad-21: matroska-demuxer 0.7's `Colour::new` reads MaxCLL/MaxFALL from
    // the wrong ElementId offset (it actually reads MatrixCoefficients), and
    // `MasteringMetadata::new` reads each `_chromaticity_y` from the matching
    // `_chromaticity_x` ElementId — so all three primaries' y values come back
    // holding the corresponding x value. Re-scan the raw EBML bytes to recover
    // the canonical values; the same workaround already lives in
    // `probe_mkv_color_info`. We MUST also clear the unified
    // `ColorMetadata.content_light_level` and the mastering display y-fields
    // we synthesized from the poisoned typed accessors so a scan miss doesn't
    // leave the wrong value in place.
    color_info.max_cll = None;
    color_info.max_fall = None;
    color_metadata.content_light_level = None;
    if let Some(md) = color_metadata.mastering_display.as_mut() {
        // The y values are poisoned with the matching x values — clear them
        // in case the raw scan can't recover (defensive: leave 0 vs garbage).
        md.primaries_r_y = 0;
        md.primaries_g_y = 0;
        md.primaries_b_y = 0;
    }
    if let Some(local) = color_info.mastering.as_mut() {
        local.primary_r_chromaticity_y = None;
        local.primary_g_chromaticity_y = None;
        local.primary_b_chromaticity_y = None;
    }
    if let Some(fix) = scan_mkv_colour_raw(data) {
        color_info.max_cll = fix.max_cll;
        color_info.max_fall = fix.max_fall;
        if fix.max_cll.is_some() || fix.max_fall.is_some() {
            color_metadata.content_light_level = Some(ContentLightLevel {
                max_cll: fix.max_cll.unwrap_or(0).min(u16::MAX as u32) as u16,
                max_fall: fix.max_fall.unwrap_or(0).min(u16::MAX as u32) as u16,
            });
        }
        // Re-fold the recovered y-chromaticities (HEVC SEI D.2.28 wire
        // domain: 0.00002 increments → multiply by 50_000, saturate to u16).
        let chrom = |v: f64| (v * 50_000.0).round().clamp(0.0, u16::MAX as f64) as u16;
        if let Some(md) = color_metadata.mastering_display.as_mut() {
            if let Some(y) = fix.primary_r_chromaticity_y {
                md.primaries_r_y = chrom(y);
            }
            if let Some(y) = fix.primary_g_chromaticity_y {
                md.primaries_g_y = chrom(y);
            }
            if let Some(y) = fix.primary_b_chromaticity_y {
                md.primaries_b_y = chrom(y);
            }
        }
        if let Some(local) = color_info.mastering.as_mut() {
            if fix.primary_r_chromaticity_y.is_some() {
                local.primary_r_chromaticity_y = fix.primary_r_chromaticity_y;
            }
            if fix.primary_g_chromaticity_y.is_some() {
                local.primary_g_chromaticity_y = fix.primary_g_chromaticity_y;
            }
            if fix.primary_b_chromaticity_y.is_some() {
                local.primary_b_chromaticity_y = fix.primary_b_chromaticity_y;
            }
        }
    }

    let needs_annexb = mkv_codec_needs_annexb(&codec_id);
    let codec = match codec_id.as_str() {
        "V_VP9" => "vp9".to_string(),
        "V_VP8" => "vp8".to_string(),
        "V_AV1" => "av1".to_string(),
        "V_MPEG4/ISO/AVC" => "h264".to_string(),
        "V_MPEGH/ISO/HEVC" => "h265".to_string(),
        other => other.to_lowercase(),
    };

    let timestamp_scale = mkv.info().timestamp_scale().get();
    let duration_ticks = mkv.info().duration().unwrap_or(0.0);
    // timestamp_scale is in ns; duration is in ticks (float)
    let duration = duration_ticks * (timestamp_scale as f64) / 1_000_000_000.0;

    // Tag-based bitrate: preferred over the computed fallback when a
    // muxer wrote a `BIT_RATE` Matroska Tag scoped to our track UID.
    // See `bitrate_from_tags` for scope-resolution details.
    let tag_bitrate = mkv
        .tags()
        .and_then(|tags| bitrate_from_tags(tags, track_uid));
    // Emit the extended metadata we can't (yet) carry on `StreamInfo`
    // on a structured log line — downstream work-items #HDR10 and mux
    // SEI passthrough will read them via `probe_mkv_color_info`.
    if color_info != MkvColorInfo::default() {
        tracing::info!(
            bits_per_channel = ?color_info.bits_per_channel,
            max_cll = ?color_info.max_cll,
            max_fall = ?color_info.max_fall,
            mastering = ?color_info.mastering,
            "MKV Colour: parsed HDR-adjacent metadata"
        );
    }

    let mut samples: Vec<Vec<u8>> = Vec::new();
    let mut frame = MkvFrame::default();
    let mut total_video_bytes: u64 = 0;
    // Same per-stream tracker as the MP4 path. MKV's CodecPrivate carries
    // the avcC / hvcC bytes verbatim, so the same first-IRAP-prepend
    // heuristic applies (and is more robust than the old
    // `is_first_video_sample` flag, which assumed sample 0 was always IRAP).
    let mut mkv_tracker = if needs_annexb {
        Some(ParamSetTracker::new(if codec_id == "V_MPEG4/ISO/AVC" {
            NaluCodec::Avc
        } else {
            NaluCodec::Hevc
        }))
    } else {
        None
    };
    loop {
        match mkv.next_frame(&mut frame) {
            Ok(true) => {
                if frame.track == track_number {
                    let raw = std::mem::take(&mut frame.data);
                    total_video_bytes += raw.len() as u64;
                    if let Some(tracker) = mkv_tracker.as_mut() {
                        let annexb = length_prefixed_to_annexb_tracked(
                            &raw,
                            length_size,
                            tracker,
                            &annexb_prepend,
                        );
                        samples.push(annexb);
                    } else {
                        samples.push(raw);
                    }
                }
            }
            Ok(false) => break,
            Err(e) => bail!("MKV frame read error: {e}"),
        }
    }

    let total_frames = samples.len() as u64;
    // Frame rate fallback chain (Squad-32):
    //   1. samples / segment_duration  (most accurate when both are known)
    //   2. 1 / DefaultDuration          (Matroska's canonical per-frame ns)
    //   3. 30.0                         (last-resort sentinel)
    let frame_rate = if duration > 0.0 {
        total_frames as f64 / duration
    } else if let Some(dd_ns) = track_default_duration_ns.filter(|n| *n > 0) {
        1_000_000_000.0 / dd_ns as f64
    } else {
        30.0
    };

    let detected_pf = codec::pixel_format::detect(&codec, &samples);

    // Bitrate priority: Tag `BIT_RATE` if present → summed sample bytes
    // over the segment duration. Never 0 unless the file has no samples
    // AND no tag (in which case bitrate is genuinely unknowable and we
    // keep the historical 0 sentinel).
    let bitrate = match tag_bitrate {
        Some(b) if b > 0 => b,
        _ => {
            if duration > 0.0 && total_video_bytes > 0 {
                ((total_video_bytes as f64 * 8.0) / duration) as u64
            } else {
                0
            }
        }
    };

    let info = StreamInfo {
        codec: codec.clone(),
        width,
        height,
        frame_rate,
        duration,
        pixel_format: detected_pf,
        color_space,
        total_frames,
        bitrate,
        color_metadata,
    };

    // Audio passthrough uses its own MatroskaFile handle (re-opened) since
    // next_frame above already consumed the stream.
    let audio = super::audio::extract_mkv_audio(data);

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

/// MKV / WebM streaming demuxer. Wraps `MatroskaFile` whose `next_frame`
/// API is already pull-shaped, so the streaming impl is a thin wrapper:
/// pull next frame, filter to the video track, AVCC→Annex-B convert if
/// AVC/HEVC, surface as a `Sample`.
pub struct MkvStreamingDemuxer {
    mkv: MatroskaFile<Cursor<Vec<u8>>>,
    header: DemuxHeader,
    audio: Option<AudioTrack>,
    /// Text subtitle track, buffered at construction like `audio`.
    subtitles: Option<SubtitleTrack>,
    track_number: u64,
    timestamp_scale: u64,
    annexb_prepend: Vec<Vec<u8>>,
    length_size: u8,
    tracker: Option<ParamSetTracker>,
    /// Default-duration in ns from the track header — used as the
    /// fallback per-sample duration when the Block doesn't carry one.
    default_duration_ns: Option<u64>,
    /// Lazily set on the first `next_video_sample()` call by running
    /// `pixel_format::detect` against the first emitted sample.
    /// `header.info.pixel_format` is then patched in place. Subsequent
    /// calls skip the probe (codec sequence headers don't change
    /// mid-stream for the codecs we support).
    pixel_format_detected: bool,
}

pub(crate) fn demux_mkv_streaming_init(data: &[u8]) -> Result<MkvStreamingDemuxer> {
    let owned = data.to_vec();
    // First pass: open with a borrow to harvest header metadata without
    // consuming the buffer that backs the streaming reader.
    let cursor = Cursor::new(owned.as_slice());
    let probe =
        MatroskaFile::open(cursor).map_err(|e| anyhow::anyhow!("reading MKV header: {e}"))?;

    let (
        track_number,
        track_uid,
        codec_id,
        width,
        height,
        annexb_prepend,
        length_size,
        color_space,
        mut color_metadata,
        mut color_info,
        track_default_duration_ns,
    ) = {
        let track_info = probe
            .tracks()
            .iter()
            .find(|t| t.track_type() == MkvTrackType::Video)
            .context("no video track in MKV")?;

        let track_number = track_info.track_number().get();
        let track_uid = track_info.track_uid().get();
        let codec_id = track_info.codec_id().to_string();
        let default_duration_ns = track_info.default_duration().map(|d| d.get());

        let (annexb_prepend, length_size): (Vec<Vec<u8>>, u8) = if codec_id == "V_MPEG4/ISO/AVC" {
            let priv_bytes = track_info
                .codec_private()
                .context("V_MPEG4/ISO/AVC CodecPrivate missing")?;
            let cfg = parse_avcc(priv_bytes).context("V_MPEG4/ISO/AVC CodecPrivate malformed")?;
            (cfg.parameter_sets, cfg.length_size)
        } else if codec_id == "V_MPEGH/ISO/HEVC" {
            let priv_bytes = track_info
                .codec_private()
                .context("V_MPEGH/ISO/HEVC CodecPrivate missing")?;
            let cfg = parse_hvcc(priv_bytes).context("V_MPEGH/ISO/HEVC CodecPrivate malformed")?;
            (cfg.parameter_sets, cfg.length_size)
        } else {
            (Vec::new(), 4)
        };

        if mkv_codec_needs_annexb(&codec_id) && annexb_prepend.is_empty() {
            bail!("AVC/HEVC MKV CodecPrivate missing or empty — no parameter sets to prepend");
        }

        let video = track_info
            .video()
            .context("video track missing Video element")?;
        let w = video.pixel_width().get() as u32;
        let h = video.pixel_height().get() as u32;

        let (color_space, color_metadata, color_info) = match video.colour() {
            Some(colour) => colour_to_pipeline(colour),
            None => (
                ColorSpace::Bt709,
                ColorMetadata::default(),
                MkvColorInfo::default(),
            ),
        };

        (
            track_number,
            track_uid,
            codec_id,
            w,
            h,
            annexb_prepend,
            length_size,
            color_space,
            color_metadata,
            color_info,
            default_duration_ns,
        )
    };

    // Apply the matroska-demuxer 0.7 raw-scan workarounds — same as the
    // legacy demux_mkv path.
    color_info.max_cll = None;
    color_info.max_fall = None;
    color_metadata.content_light_level = None;
    if let Some(md) = color_metadata.mastering_display.as_mut() {
        md.primaries_r_y = 0;
        md.primaries_g_y = 0;
        md.primaries_b_y = 0;
    }
    if let Some(local) = color_info.mastering.as_mut() {
        local.primary_r_chromaticity_y = None;
        local.primary_g_chromaticity_y = None;
        local.primary_b_chromaticity_y = None;
    }
    if let Some(fix) = scan_mkv_colour_raw(&owned) {
        color_info.max_cll = fix.max_cll;
        color_info.max_fall = fix.max_fall;
        if fix.max_cll.is_some() || fix.max_fall.is_some() {
            color_metadata.content_light_level = Some(ContentLightLevel {
                max_cll: fix.max_cll.unwrap_or(0).min(u16::MAX as u32) as u16,
                max_fall: fix.max_fall.unwrap_or(0).min(u16::MAX as u32) as u16,
            });
        }
        let chrom = |v: f64| (v * 50_000.0).round().clamp(0.0, u16::MAX as f64) as u16;
        if let Some(md) = color_metadata.mastering_display.as_mut() {
            if let Some(y) = fix.primary_r_chromaticity_y {
                md.primaries_r_y = chrom(y);
            }
            if let Some(y) = fix.primary_g_chromaticity_y {
                md.primaries_g_y = chrom(y);
            }
            if let Some(y) = fix.primary_b_chromaticity_y {
                md.primaries_b_y = chrom(y);
            }
        }
        if let Some(local) = color_info.mastering.as_mut() {
            if fix.primary_r_chromaticity_y.is_some() {
                local.primary_r_chromaticity_y = fix.primary_r_chromaticity_y;
            }
            if fix.primary_g_chromaticity_y.is_some() {
                local.primary_g_chromaticity_y = fix.primary_g_chromaticity_y;
            }
            if fix.primary_b_chromaticity_y.is_some() {
                local.primary_b_chromaticity_y = fix.primary_b_chromaticity_y;
            }
        }
    }

    let needs_annexb = mkv_codec_needs_annexb(&codec_id);
    let codec = match codec_id.as_str() {
        "V_VP9" => "vp9".to_string(),
        "V_VP8" => "vp8".to_string(),
        "V_AV1" => "av1".to_string(),
        "V_MPEG4/ISO/AVC" => "h264".to_string(),
        "V_MPEGH/ISO/HEVC" => "h265".to_string(),
        other => other.to_lowercase(),
    };

    let timestamp_scale = probe.info().timestamp_scale().get();
    let duration_ticks = probe.info().duration().unwrap_or(0.0);
    let duration = duration_ticks * (timestamp_scale as f64) / 1_000_000_000.0;
    let tag_bitrate = probe
        .tags()
        .and_then(|tags| bitrate_from_tags(tags, track_uid));
    if color_info != MkvColorInfo::default() {
        tracing::info!(
            bits_per_channel = ?color_info.bits_per_channel,
            max_cll = ?color_info.max_cll,
            max_fall = ?color_info.max_fall,
            mastering = ?color_info.mastering,
            "MKV Colour: parsed HDR-adjacent metadata"
        );
    }

    drop(probe);

    // Audio: extract from the owned bytes via a separate MatroskaFile
    // open (same as legacy demux_mkv). The video reader below needs its
    // own clean cursor.
    let audio = super::audio::extract_mkv_audio(&owned);
    // Same for text subtitles — one more pass over the owned bytes, so the
    // video reader below still gets a clean cursor.
    let subtitles = extract_mkv_subtitles(&owned);

    // Build the streaming MKV reader against the owned buffer.
    let mkv = MatroskaFile::open(Cursor::new(owned.clone()))
        .map_err(|e| anyhow::anyhow!("opening MKV streaming reader: {e}"))?;

    // Bitrate / frame_rate / pixel_format are best-effort at construction
    // time. Bitrate falls back to 0 (unknown) if no tag exists; the
    // legacy path computes it by summing sample bytes which is fine for
    // Vec-materialized output but blows the streaming budget. We surface
    // the tag bitrate when present and 0 otherwise — pipeline already
    // tolerates 0 (matches the AVI / TS behaviour).
    let bitrate = tag_bitrate.unwrap_or(0);

    // For frame_rate we apply the Squad-32 fallback chain as far as it
    // goes without the materialized sample count. samples/duration is
    // unknowable in streaming, so use DefaultDuration first then 30.0.
    let frame_rate = if let Some(dd_ns) = track_default_duration_ns.filter(|n| *n > 0) {
        1_000_000_000.0 / dd_ns as f64
    } else if duration > 0.0 {
        // duration-only fallback: assume 30 fps × duration as the floor.
        // This matches what the legacy path produced when sample count
        // was tiny; for normal media DefaultDuration is virtually always
        // present.
        30.0
    } else {
        30.0
    };

    // Pixel format detection requires a sample. For the streaming
    // demuxer's StreamInfo we keep the codec-defaulted Yuv420p — the
    // actual decoded format is whatever the decoder produces.
    // (The legacy `demux_mkv()` adapter re-runs `pixel_format::detect`
    // on the materialized samples after the drain.)
    let pixel_format = PixelFormat::Yuv420p;

    let info = StreamInfo {
        codec: codec.clone(),
        width,
        height,
        frame_rate,
        duration,
        pixel_format,
        color_space,
        total_frames: 0, // unknown until drained
        bitrate,
        color_metadata,
    };

    let tracker = if needs_annexb {
        Some(ParamSetTracker::new(if codec_id == "V_MPEG4/ISO/AVC" {
            NaluCodec::Avc
        } else {
            NaluCodec::Hevc
        }))
    } else {
        None
    };

    let _ = needs_annexb; // tracker presence reflects this
    Ok(MkvStreamingDemuxer {
        mkv,
        header: DemuxHeader { codec, info },
        audio,
        subtitles,
        track_number,
        timestamp_scale,
        annexb_prepend,
        length_size,
        tracker,
        default_duration_ns: track_default_duration_ns,
        pixel_format_detected: false,
    })
}

impl StreamingDemuxer for MkvStreamingDemuxer {
    fn header(&self) -> &DemuxHeader {
        &self.header
    }

    fn next_video_sample(&mut self) -> Result<Option<Sample>> {
        let mut frame = MkvFrame::default();
        loop {
            match self.mkv.next_frame(&mut frame) {
                Ok(true) => {
                    if frame.track != self.track_number {
                        continue;
                    }
                    let raw = std::mem::take(&mut frame.data);
                    let data = if let Some(tracker) = self.tracker.as_mut() {
                        length_prefixed_to_annexb_tracked(
                            &raw,
                            self.length_size,
                            tracker,
                            &self.annexb_prepend,
                        )
                    } else {
                        raw
                    };
                    // Lazy pixel-format detection on the first sample.
                    // `pixel_format::detect` only ever reads `samples[0]`,
                    // so a one-shot probe against the first emitted sample
                    // matches the legacy `demux_mkv()` behaviour without
                    // requiring the full Vec to be materialised first.
                    if !self.pixel_format_detected {
                        let detected = codec::pixel_format::detect(
                            &self.header.codec,
                            std::slice::from_ref(&data),
                        );
                        self.header.info.pixel_format = detected;
                        self.pixel_format_detected = true;
                    }
                    let pts_ticks = frame.timestamp.saturating_mul(self.timestamp_scale) as i64;
                    let duration_ticks = frame
                        .duration
                        .or(self.default_duration_ns)
                        .map(|ns| ns.min(u32::MAX as u64) as u32)
                        .unwrap_or(0);
                    return Ok(Some(Sample {
                        data,
                        pts_ticks,
                        duration_ticks,
                    }));
                }
                Ok(false) => return Ok(None),
                Err(e) => bail!("MKV frame read error: {e}"),
            }
        }
    }

    fn audio(&self) -> Option<&AudioTrack> {
        self.audio.as_ref()
    }

    fn subtitles(&self) -> Option<&SubtitleTrack> {
        self.subtitles.as_ref()
    }
}

// ---------------------------------------------------------------------------
// Public probe helper
// ---------------------------------------------------------------------------

/// Re-open an MKV container solely to extract the extended Colour
/// sub-elements that don't fit on `StreamInfo.color_metadata`
/// (MaxCLL / MaxFALL / SMPTE-2086 mastering primaries / bits_per_channel /
/// chroma siting). Intended for downstream paths that need HDR10 side
/// data for muxing; returns `None` when the file has no video track,
/// no `Colour` element, or isn't a well-formed MKV.
pub fn probe_mkv_color_info(data: &[u8]) -> Option<MkvColorInfo> {
    let cursor = Cursor::new(data);
    let mkv = MatroskaFile::open(cursor).ok()?;
    let track = mkv
        .tracks()
        .iter()
        .find(|t| t.track_type() == MkvTrackType::Video)?;
    let colour = track.video()?.colour()?;
    let (_, _, mut info) = colour_to_pipeline(colour);

    // matroska-demuxer 0.7 has two known bugs we work around with a raw
    // EBML scan (see `scan_mkv_colour_raw` doc):
    //   * `Colour::new` misreads MaxCLL/MaxFALL at the MatrixCoefficients
    //     ElementId offset (so both come back holding the matrix value).
    //   * `MasteringMetadata::new` misreads each `_chromaticity_y` at the
    //     matching `_chromaticity_x` ElementId (so all three primaries' y
    //     values come back holding the corresponding x value).
    // Clear the poisoned fields before the raw scan overrides them so a
    // scan miss doesn't leave the wrong value in place.
    info.max_cll = None;
    info.max_fall = None;
    if let Some(local) = info.mastering.as_mut() {
        local.primary_r_chromaticity_y = None;
        local.primary_g_chromaticity_y = None;
        local.primary_b_chromaticity_y = None;
    }
    if let Some(fix) = scan_mkv_colour_raw(data) {
        info.max_cll = fix.max_cll;
        info.max_fall = fix.max_fall;
        if let Some(local) = info.mastering.as_mut() {
            if fix.primary_r_chromaticity_y.is_some() {
                local.primary_r_chromaticity_y = fix.primary_r_chromaticity_y;
            }
            if fix.primary_g_chromaticity_y.is_some() {
                local.primary_g_chromaticity_y = fix.primary_g_chromaticity_y;
            }
            if fix.primary_b_chromaticity_y.is_some() {
                local.primary_b_chromaticity_y = fix.primary_b_chromaticity_y;
            }
        }
    }
    Some(info)
}

// ---------------------------------------------------------------------------
// Codec-ID helpers
// ---------------------------------------------------------------------------

/// True for MKV CodecIDs whose samples are length-prefixed (AVCC/HVCC) and
/// require SPS/PPS pulled from the track's CodecPrivate to feed a decoder
/// that expects Annex-B. demux_mkv bails on these until the Annex-B path is
/// wired — currently only VP8/VP9/AV1 are safe through MKV.
pub(super) fn mkv_codec_needs_annexb(codec_id: &str) -> bool {
    matches!(codec_id, "V_MPEG4/ISO/AVC" | "V_MPEGH/ISO/HEVC")
}

