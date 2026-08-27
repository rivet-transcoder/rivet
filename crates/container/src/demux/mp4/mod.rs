/// MP4 / MOV box-tree demux, codec detection, AVC/HEVC config extraction,
/// fragmented-MP4 sample-table builder, and the `Mp4StreamingDemuxer`
/// implementation (Squad streaming-migration-55 P1).
///
/// This is the module root. Concerns are split across three files:
///   - `mod.rs`        — `demux_mp4` entry point, format/frame-rate helpers,
///                       re-exports, module declarations
///   - `streaming.rs`  — `Mp4StreamingDemuxer` + `demux_mp4_streaming_init`
///                       + `FragSample` + `build_fragmented_sample_table`
///   - `sample_entry.rs` — sample-entry detection + AVC/HEVC config extraction
use anyhow::{Context, Result};
use frame::{ColorMetadata, ColorSpace, PixelFormat, StreamInfo};
use mp4::Mp4Reader;
use std::io::Cursor;

use crate::annexb::{NaluCodec, ParamSetTracker, length_prefixed_to_annexb_tracked};
use crate::mp4_sanitize::sanitize_isobmff_box_sizes;

use super::DemuxResult;

mod sample_entry;
mod streaming;
mod subtitle;

// ---------------------------------------------------------------------------
// Re-exports — public surface consumed by `demux/mod.rs`
// ---------------------------------------------------------------------------

// `demux/mod.rs` does:
//   pub use mp4::{demux_mp4, Mp4StreamingDemuxer};
//   pub(crate) use mp4::demux_mp4_streaming_init;
// `Mp4StreamingDemuxer` is re-exported here as pub(super) so that
// `demux/mod.rs` (the parent) can in turn re-export it as `pub`.
// `FragSample` stays pub(super) — matching the original visibility in
// the flat mp4.rs where it was `pub(super)` (visible only to `demux`).
pub use streaming::Mp4StreamingDemuxer;
#[allow(unused_imports)] // used only by demux/tests.rs under #[cfg(test)]
pub(super) use streaming::FragSample;
pub(crate) use streaming::demux_mp4_streaming_init;

// Internal re-exports: `demux` siblings (`audio.rs`, `tests.rs`) reach these
// helpers via `super::mp4::<item>` without the items appearing in the
// crate's public API.
pub(crate) use sample_entry::{has_av01_sample_entry, prores_sample_entry_fourcc};
#[allow(unused_imports)] // used only by demux/tests.rs under #[cfg(test)]
pub(crate) use sample_entry::parse_avcc_param_sets;
pub(crate) use streaming::build_fragmented_sample_table;
pub(crate) use subtitle::extract_mp4_subtitle_tracks;

// Private imports from submodules needed directly inside `demux_mp4` below.
use sample_entry::{extract_avc_config, extract_hevc_config, hevc_sample_entry_fourcc};

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
    let detected_pf = frame::pixel_format::detect(&codec, &samples);
    let mut info = StreamInfo {
        pixel_format: detected_pf,
        ..info
    };
    // Colour: the `colr` box when the file has one, else the SPS VUI. Without
    // this an HDR MP4 kept the SDR default transfer and was never tonemapped.
    let vui = if needs_annexb { super::hdr::colour_from_parameter_sets(&codec, &sps_pps) } else { None };
    super::hdr::apply_colour_description(&mut info, mp4_color.nclx, vui);

    let audio = super::audio::extract_mp4_audio(data);

    Ok(DemuxResult {
        codec,
        info,
        samples,
        audio,
    })
}

// ---------------------------------------------------------------------------
// Format / frame-rate helpers (private; also accessible from streaming.rs
// via `super::` since streaming is a child module of this mod)
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
