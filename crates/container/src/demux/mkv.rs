/// Matroska / WebM demux, Colour element parsing, EBML raw scanner, and the
/// `MkvStreamingDemuxer` implementation (Squad streaming-migration-55 P1).
use anyhow::{Context, Result, bail};
use codec::frame::{
    ColorMetadata, ColorSpace, ContentLightLevel, MasteringDisplay, PixelFormat, StreamInfo,
    TransferFn,
};
use matroska_demuxer::{
    Colour as MkvColour, Frame as MkvFrame, MasteringMetadata as MkvMastering, MatrixCoefficients,
    MatroskaFile, Primaries, Range as MkvRange, TrackType as MkvTrackType, TransferCharacteristics,
};
use std::io::Cursor;

use crate::annexb::{
    NaluCodec, ParamSetTracker, length_prefixed_to_annexb_tracked, parse_avcc, parse_hvcc,
};
use crate::streaming::{DemuxHeader, Sample, StreamingDemuxer};
use crate::{MkvColorInfo, MkvMasteringMetadata};

use super::{AudioTrack, DemuxResult};

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

// ---------------------------------------------------------------------------
// Colour element → pipeline types mapping
// ---------------------------------------------------------------------------

/// Map a Matroska `Colour` element into our pipeline's color-space,
/// per-H.273 `ColorMetadata`, and extended `MkvColorInfo`. Unspecified
/// sub-elements default to the SDR BT.709 baseline so decoders that
/// never read a Colour element keep behaving exactly as before.
fn colour_to_pipeline(colour: &MkvColour) -> (ColorSpace, ColorMetadata, MkvColorInfo) {
    let matrix_u8 = colour
        .matrix_coefficients()
        .map(matrix_coefficients_to_h273);
    let primaries_u8 = colour.primaries().map(primaries_to_h273);
    let transfer_u8 = colour.transfer_characteristics().map(transfer_to_h273);
    let range = colour.range();

    let color_space = match colour.matrix_coefficients() {
        Some(MatrixCoefficients::Bt709) => ColorSpace::Bt709,
        Some(MatrixCoefficients::Bt470bg) | Some(MatrixCoefficients::Smpte170) => ColorSpace::Bt601,
        Some(MatrixCoefficients::Bt2020Ncl)
        | Some(MatrixCoefficients::Bt2020Cl)
        | Some(MatrixCoefficients::Bt2100) => ColorSpace::Bt2020,
        _ => ColorSpace::Bt709,
    };

    let mastering = colour.mastering_metadata().map(mkv_mastering_to_local);
    let mkv_max_cll = colour.max_cll().and_then(|v| u32::try_from(v).ok());
    let mkv_max_fall = colour.max_fall().and_then(|v| u32::try_from(v).ok());

    // Squad-21: also synthesize the unified ColorMetadata HDR fields from
    // the MKV `MasteringMetadata` + `MaxCLL` / `MaxFALL` so the muxer
    // (Squad-20) can write `mdcv`/`clli` without re-reading the
    // MKV-specific MkvColorInfo struct. matroska-demuxer 0.7's MaxCLL/
    // MaxFALL bug (see `probe_mkv_color_info`) means the values here
    // come from the typed accessor — for the canonical scan we re-read
    // raw bytes in `probe_mkv_color_info`. The two paths agree on
    // well-formed MKVs and disagree only on malformed ones (where the
    // raw scan wins). Pipeline plumbs the raw-scan path for MKV.
    let unified_mastering = mastering.as_ref().and_then(mkv_mastering_to_unified);
    let unified_cll = match (mkv_max_cll, mkv_max_fall) {
        (None, None) => None,
        (cll, fall) => Some(ContentLightLevel {
            max_cll: cll.unwrap_or(0).min(u16::MAX as u32) as u16,
            max_fall: fall.unwrap_or(0).min(u16::MAX as u32) as u16,
        }),
    };

    let color_metadata = ColorMetadata {
        transfer: transfer_u8.map(TransferFn::from_h273).unwrap_or_default(),
        matrix_coefficients: matrix_u8.unwrap_or(1),
        colour_primaries: primaries_u8.unwrap_or(1),
        // H.273 full_range_flag: Matroska Range=2 (Full) sets it; any
        // other value (Broadcast, Defined, Unknown) keeps the studio
        // 16..235 default.
        full_range: matches!(range, Some(MkvRange::Full)),
        // Squad-21 wires MKV float chromaticities + max_cll/fall into
        // the H.265-spec u16 encoding via `mkv_mastering_to_unified` and
        // the f64 → cd/m² conversion above (also recovers around two
        // matroska-demuxer 0.7 bugs that misread MaxCLL/MaxFALL and y
        // chromaticities at the wrong ElementIds).
        mastering_display: unified_mastering,
        content_light_level: unified_cll,
    };

    let extra = MkvColorInfo {
        bits_per_channel: colour.bits_per_channel().and_then(|v| u8::try_from(v).ok()),
        chroma_subsampling_horz: colour
            .chroma_subsampling_horz()
            .and_then(|v| u8::try_from(v).ok()),
        chroma_subsampling_vert: colour
            .chroma_subsampling_vert()
            .and_then(|v| u8::try_from(v).ok()),
        chroma_siting_horz: colour.chroma_sitting_horz().map(chroma_siting_horz_to_u8),
        chroma_siting_vert: colour.chroma_sitting_vert().map(chroma_siting_vert_to_u8),
        max_cll: mkv_max_cll,
        max_fall: mkv_max_fall,
        mastering,
    };

    (color_space, color_metadata, extra)
}

/// Convert the Matroska f64 chromaticities (range 0..=1) and luminance
/// (cd/m²) into the integer encoding the unified `MasteringDisplay`
/// uses (HEVC SEI D.2.28 wire format). Returns `None` when no
/// sub-element of the MasteringMetadata was populated.
fn mkv_mastering_to_unified(m: &MkvMasteringMetadata) -> Option<MasteringDisplay> {
    if m.primary_r_chromaticity_x.is_none()
        && m.primary_g_chromaticity_x.is_none()
        && m.primary_b_chromaticity_x.is_none()
        && m.white_point_chromaticity_x.is_none()
        && m.luminance_max.is_none()
        && m.luminance_min.is_none()
    {
        return None;
    }
    let chrom = |v: Option<f64>| -> u16 {
        // 0.00002 increments per HEVC SEI D.2.28 — map [0.0, ~1.31)
        // into a u16 with saturation.
        let scaled = (v.unwrap_or(0.0) * 50_000.0).round();
        scaled.clamp(0.0, u16::MAX as f64) as u16
    };
    let max_lum = (m.luminance_max.unwrap_or(0.0) * 10_000.0).round();
    let min_lum = (m.luminance_min.unwrap_or(0.0) * 10_000.0).round();
    Some(MasteringDisplay {
        primaries_r_x: chrom(m.primary_r_chromaticity_x),
        primaries_r_y: chrom(m.primary_r_chromaticity_y),
        primaries_g_x: chrom(m.primary_g_chromaticity_x),
        primaries_g_y: chrom(m.primary_g_chromaticity_y),
        primaries_b_x: chrom(m.primary_b_chromaticity_x),
        primaries_b_y: chrom(m.primary_b_chromaticity_y),
        white_point_x: chrom(m.white_point_chromaticity_x),
        white_point_y: chrom(m.white_point_chromaticity_y),
        max_luminance: max_lum.clamp(0.0, u32::MAX as f64) as u32,
        min_luminance: min_lum.clamp(0.0, u32::MAX as f64) as u32,
    })
}

fn mkv_mastering_to_local(m: &MkvMastering) -> MkvMasteringMetadata {
    MkvMasteringMetadata {
        primary_r_chromaticity_x: m.primary_r_chromaticity_x(),
        primary_r_chromaticity_y: m.primary_r_chromaticity_y(),
        primary_g_chromaticity_x: m.primary_g_chromaticity_x(),
        primary_g_chromaticity_y: m.primary_g_chromaticity_y(),
        primary_b_chromaticity_x: m.primary_b_chromaticity_x(),
        primary_b_chromaticity_y: m.primary_b_chromaticity_y(),
        white_point_chromaticity_x: m.white_point_chromaticity_x(),
        white_point_chromaticity_y: m.white_point_chromaticity_y(),
        luminance_max: m.luminance_max(),
        luminance_min: m.luminance_min(),
    }
}

/// MatroskaElement MatrixCoefficients (0x55B1) uses the H.273 numbering
/// 1:1, but the `matroska-demuxer` enum hides the raw u8. Reverse the
/// mapping so downstream (mux `colr nclx`, nvenc encode params) can
/// write the original numeric value back out without re-deriving it.
fn matrix_coefficients_to_h273(m: MatrixCoefficients) -> u8 {
    match m {
        MatrixCoefficients::Identity => 0,
        MatrixCoefficients::Bt709 => 1,
        MatrixCoefficients::Fcc73682 => 4,
        MatrixCoefficients::Bt470bg => 5,
        MatrixCoefficients::Smpte170 => 6,
        MatrixCoefficients::Smpte240 => 7,
        MatrixCoefficients::YCoCg => 8,
        MatrixCoefficients::Bt2020Ncl => 9,
        MatrixCoefficients::Bt2020Cl => 10,
        MatrixCoefficients::SmpteSt2085 => 11,
        MatrixCoefficients::ChromaDerivedNcl => 12,
        MatrixCoefficients::ChromaDerivedCl => 13,
        MatrixCoefficients::Bt2100 => 14,
        MatrixCoefficients::Unknown => 2, // H.273 "unspecified"
    }
}

fn transfer_to_h273(t: TransferCharacteristics) -> u8 {
    match t {
        TransferCharacteristics::Bt709 => 1,
        TransferCharacteristics::Bt407m => 4,
        TransferCharacteristics::Bt407bg => 5,
        TransferCharacteristics::Smpte170 => 6,
        TransferCharacteristics::Smpte240 => 7,
        TransferCharacteristics::Linear => 8,
        TransferCharacteristics::Log => 9,
        TransferCharacteristics::LogSqrt => 10,
        TransferCharacteristics::Iec61966_2_4 => 11,
        TransferCharacteristics::Bt1361 => 12,
        TransferCharacteristics::Iec61966_2_1 => 13,
        TransferCharacteristics::Bt220_10 => 14,
        TransferCharacteristics::Bt220_12 => 15,
        TransferCharacteristics::Bt2100 => 16,
        TransferCharacteristics::SmpteSt428_1 => 17,
        TransferCharacteristics::Hlg => 18,
        TransferCharacteristics::Unknown => 2,
    }
}

fn primaries_to_h273(p: Primaries) -> u8 {
    match p {
        Primaries::Bt709 => 1,
        Primaries::Bt470m => 4,
        Primaries::Bt601 => 5,
        Primaries::Smpte170 => 6,
        Primaries::Smpte240 => 7,
        Primaries::Film => 8,
        Primaries::Bt2020 => 9,
        Primaries::SmpteSt428_1 => 10,
        Primaries::SmpteRp432_2 => 11,
        Primaries::SmpteEg432_2 => 12,
        Primaries::JedecP22 => 22,
        Primaries::Unknown => 2,
    }
}

fn chroma_siting_horz_to_u8(s: matroska_demuxer::ChromaSitingHorz) -> u8 {
    match s {
        matroska_demuxer::ChromaSitingHorz::LeftCollated => 1,
        matroska_demuxer::ChromaSitingHorz::Half => 2,
        matroska_demuxer::ChromaSitingHorz::Unknown => 0,
    }
}

fn chroma_siting_vert_to_u8(s: matroska_demuxer::ChromaSitingVert) -> u8 {
    match s {
        matroska_demuxer::ChromaSitingVert::LeftCollated => 1,
        matroska_demuxer::ChromaSitingVert::Half => 2,
        matroska_demuxer::ChromaSitingVert::Unknown => 0,
    }
}

/// Resolve a track-scoped `BIT_RATE` Matroska Tag to a bits-per-second
/// value. Matroska's tag-scoping rules (spec §"Tagging") say: a Tag
/// applies to the track whose `TagTrackUID` matches, or to every track
/// in the segment if `TagTrackUID` is absent or 0. We prefer an exact
/// UID match, fall back to a segment-wide tag when no per-track value
/// exists.
///
/// `BIT_RATE` is the canonical Matroska target tag name (FFmpeg writes
/// it; the MKVToolNix matrix documents it). Some encoders emit
/// `BPS` / `BPS-eng` instead — we accept both for robustness. Values
/// are strings of base-10 digits in bits per second.
fn bitrate_from_tags(tags: &[matroska_demuxer::Tag], track_uid: u64) -> Option<u64> {
    let matches_track = |tag: &matroska_demuxer::Tag| -> bool {
        match tag.targets() {
            None => true, // Segment-wide — applies to all tracks.
            Some(t) => match t.tag_track_uid() {
                None | Some(0) => true,
                Some(uid) => uid == track_uid,
            },
        }
    };
    let mut segment_wide: Option<u64> = None;
    let mut track_scoped: Option<u64> = None;
    for tag in tags {
        if !matches_track(tag) {
            continue;
        }
        for st in tag.simple_tags() {
            let name = st.name();
            let is_bitrate = name.eq_ignore_ascii_case("BIT_RATE")
                || name.eq_ignore_ascii_case("BPS")
                || name.to_ascii_uppercase().starts_with("BPS-");
            if !is_bitrate {
                continue;
            }
            let Some(val) = st.string() else {
                continue;
            };
            let Ok(parsed) = val.trim().parse::<u64>() else {
                continue;
            };
            let is_track_scoped = tag
                .targets()
                .and_then(|t| t.tag_track_uid())
                .map(|uid| uid == track_uid)
                .unwrap_or(false);
            if is_track_scoped {
                track_scoped = Some(parsed);
            } else if segment_wide.is_none() {
                segment_wide = Some(parsed);
            }
        }
    }
    track_scoped.or(segment_wide)
}

// ---------------------------------------------------------------------------
// Raw EBML scanner for matroska-demuxer 0.7 bug workarounds
// ---------------------------------------------------------------------------

/// Raw-bytes EBML walk for the Colour element's MaxCLL (0x55BC),
/// MaxFALL (0x55BD), and the mastering display chromaticity_y fields
/// (0x55D2 / 0x55D4 / 0x55D6). Used exclusively as a workaround for
/// matroska-demuxer 0.7 bugs:
///   * `Colour::new` reads MaxCLL / MaxFALL from MatrixCoefficients
///     (lib.rs:725..728 in matroska-demuxer-0.7.0/src/lib.rs).
///   * `MasteringMetadata::new` reads `primary_{r,g,b}_chromaticity_y`
///     from the matching X ElementId (lib.rs:846/848/850), so all three
///     y values come back holding the corresponding x value.
/// Returns `None` when the file is not well-formed enough to reach the
/// Colour element, or when neither bug-recovery field is present.
#[derive(Default)]
struct RawColourFix {
    max_cll: Option<u32>,
    max_fall: Option<u32>,
    /// Mastering display y-chromaticity recoveries — Squad-21.
    primary_r_chromaticity_y: Option<f64>,
    primary_g_chromaticity_y: Option<f64>,
    primary_b_chromaticity_y: Option<f64>,
}

fn scan_mkv_colour_raw(data: &[u8]) -> Option<RawColourFix> {
    // Top-level: EBML header (0x1A45DFA3) then Segment (0x18538067).
    // We walk linearly until we find the Segment element and grab its
    // payload bytes — all subsequent work is inside that slice.
    let mut cursor = 0;
    let seg_body: &[u8] = loop {
        let (el, after) = next_ebml_element(data, cursor)?;
        if el.id == 0x18538067 {
            break &data[el.body_start..el.body_start + el.body_len];
        }
        cursor = after;
    };

    // Segment → Tracks (0x1654AE6B). Segment may carry many top-level
    // elements in any order — walk them until we find Tracks.
    let tracks = find_ebml_child(seg_body, 0x1654AE6B)?;
    // Tracks → TrackEntry* (0xAE). Look for the first TrackEntry whose
    // Video sub-element has a Colour; that's the path we care about.
    let mut cur = 0;
    while cur < tracks.len() {
        let (el, after) = next_ebml_element(tracks, cur)?;
        cur = after;
        if el.id != 0xAE {
            continue;
        }
        let entry = &tracks[el.body_start..el.body_start + el.body_len];
        let Some(video) = find_ebml_child(entry, 0xE0) else {
            continue;
        };
        let Some(colour) = find_ebml_child(video, 0x55B0) else {
            continue;
        };

        let mut fix = RawColourFix::default();
        let mut c = 0;
        while c < colour.len() {
            let (ce, after_ce) = match next_ebml_element(colour, c) {
                Some(v) => v,
                None => break,
            };
            c = after_ce;
            let value_bytes = &colour[ce.body_start..ce.body_start + ce.body_len];
            match ce.id {
                0x55BC => {
                    fix.max_cll = read_unsigned(value_bytes).and_then(|v| u32::try_from(v).ok());
                }
                0x55BD => {
                    fix.max_fall = read_unsigned(value_bytes).and_then(|v| u32::try_from(v).ok());
                }
                // MasteringMetadata sub-element (0x55D0). Walk its children
                // and pull the three buggy y-chromaticities so callers can
                // override the typed-accessor reads.
                0x55D0 => {
                    let md = value_bytes;
                    let mut mc = 0;
                    while mc < md.len() {
                        let (mce, after_mce) = match next_ebml_element(md, mc) {
                            Some(v) => v,
                            None => break,
                        };
                        mc = after_mce;
                        let mv = &md[mce.body_start..mce.body_start + mce.body_len];
                        match mce.id {
                            0x55D2 => fix.primary_r_chromaticity_y = read_float(mv),
                            0x55D4 => fix.primary_g_chromaticity_y = read_float(mv),
                            0x55D6 => fix.primary_b_chromaticity_y = read_float(mv),
                            _ => {}
                        }
                    }
                }
                _ => {}
            }
        }
        if fix.max_cll.is_some()
            || fix.max_fall.is_some()
            || fix.primary_r_chromaticity_y.is_some()
            || fix.primary_g_chromaticity_y.is_some()
            || fix.primary_b_chromaticity_y.is_some()
        {
            return Some(fix);
        }
    }
    None
}

/// Walk the direct children of `buf` (assumed to be an EBML master
/// element body, NOT starting with the master's own header) and
/// return the payload slice of the first element with id `want`.
fn find_ebml_child(buf: &[u8], want: u32) -> Option<&[u8]> {
    let mut cur = 0;
    while cur < buf.len() {
        let (el, after) = next_ebml_element(buf, cur)?;
        cur = after;
        if el.id == want {
            return Some(&buf[el.body_start..el.body_start + el.body_len]);
        }
    }
    None
}

#[derive(Debug)]
struct RawEbmlElement {
    id: u32,
    body_start: usize,
    body_len: usize,
}

/// Read a single EBML element at `off` within `buf`. Returns the
/// element descriptor plus the byte offset immediately after the
/// element (header + body). Only handles up to 4-byte IDs (all
/// Matroska elements fit) and size VInts up to 8 bytes.
fn next_ebml_element(buf: &[u8], off: usize) -> Option<(RawEbmlElement, usize)> {
    if off >= buf.len() {
        return None;
    }
    let (id, id_len) = read_id_vint(&buf[off..])?;
    let body_off = off + id_len;
    if body_off >= buf.len() {
        return None;
    }
    let (size, size_len) = read_size_vint(&buf[body_off..])?;
    let body_start = body_off + size_len;
    if body_start + size as usize > buf.len() {
        return None;
    }
    let elem = RawEbmlElement {
        id,
        body_start,
        body_len: size as usize,
    };
    Some((elem, body_start + size as usize))
}

/// Read an EBML Class A/B/C/D ID (top-bit marker determines width,
/// 1..=4 bytes). Returns (raw id with marker bits preserved, byte-count).
pub(super) fn read_id_vint(buf: &[u8]) -> Option<(u32, usize)> {
    if buf.is_empty() {
        return None;
    }
    let first = buf[0];
    let len = if first & 0x80 != 0 {
        1
    } else if first & 0x40 != 0 {
        2
    } else if first & 0x20 != 0 {
        3
    } else if first & 0x10 != 0 {
        4
    } else {
        return None;
    };
    if buf.len() < len {
        return None;
    }
    let mut id: u32 = 0;
    for b in &buf[..len] {
        id = (id << 8) | (*b as u32);
    }
    Some((id, len))
}

/// Read an EBML size VInt (1..=8 bytes). Strips the marker bit and
/// returns the numeric value plus byte-count.
pub(super) fn read_size_vint(buf: &[u8]) -> Option<(u64, usize)> {
    if buf.is_empty() {
        return None;
    }
    let first = buf[0];
    if first == 0 {
        return None;
    }
    let len = first.leading_zeros() as usize + 1;
    if len > 8 || buf.len() < len {
        return None;
    }
    // Mask off the leading marker bit. `len == 8` (first byte 0x01) has
    // *no* value bits in the first byte — all 56 value bits live in
    // bytes 1..8. `u8 >> 8` is UB, so branch explicitly.
    let mask: u8 = if len == 8 { 0 } else { 0xFFu8 >> len };
    let mut v: u64 = (first & mask) as u64;
    for b in &buf[1..len] {
        v = (v << 8) | (*b as u64);
    }
    Some((v, len))
}

/// Read a big-endian unsigned integer (1..=8 bytes) from a Matroska
/// value payload. Zero-length payloads encode 0.
fn read_unsigned(buf: &[u8]) -> Option<u64> {
    if buf.len() > 8 {
        return None;
    }
    let mut v: u64 = 0;
    for b in buf {
        v = (v << 8) | (*b as u64);
    }
    Some(v)
}

/// Read a big-endian Matroska float payload — 4 bytes encode an f32,
/// 8 bytes encode an f64. Anything else is malformed.
fn read_float(buf: &[u8]) -> Option<f64> {
    match buf.len() {
        4 => {
            let arr: [u8; 4] = buf.try_into().ok()?;
            Some(f32::from_be_bytes(arr) as f64)
        }
        8 => {
            let arr: [u8; 8] = buf.try_into().ok()?;
            Some(f64::from_be_bytes(arr))
        }
        _ => None,
    }
}
