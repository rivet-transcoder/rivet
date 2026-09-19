use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use bytes::Bytes;

use container::demux::subtitle::SubtitleTrack;
use container::hls::{AudioVariantSpec, SubtitleVariantSpec, VideoVariantSpec, write_hls_package};
use container::streaming::DemuxHeader;

use crate::cmaf_util::{self, keyframe_interval_for_segment};
use crate::decode_pump::ClipSource;
use crate::multigpu::{self, MultiGpuParams, RungManifest};
use crate::progress::ProgressSink;
use crate::spec::OutputSpec;
use crate::validate::needs_chroma_downsample;

use super::{RungArtifact, RungOutput, report_failed};
use super::audio::{PreparedAudio, build_audio_rendition};
use super::subtitles::build_subtitle_renditions;
use super::splice::trim_frame;

// ---------------------------------------------------------------------------
// Decode-pump config builder
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
/// Decode-pump config for one clip: codec/info/color from the header, tonemap +
/// filters from the spec. `gpu` is a placeholder for the splice plan — the
/// multi-GPU `clip_sources_for` overrides it per pump.
pub(super) fn pump_cfg_for(
    header: &DemuxHeader,
    spec: &OutputSpec,
    filters: Arc<codec::filter::FilterChain>,
    gpu: Option<u32>,
) -> crate::decode_pump::DecodePumpConfig {
    crate::decode_pump::DecodePumpConfig::for_source(header, spec, filters, gpu)
}

// ---------------------------------------------------------------------------
// HLS orchestration
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
pub(super) async fn run_hls(
    input: Bytes,
    spec: &OutputSpec,
    segment_seconds: f32,
    header: &DemuxHeader,
    frame_rate: f64,
    audio: Option<&PreparedAudio>,
    // Already trimmed / re-based text tracks; one WebVTT rendition each.
    subtitles: &[SubtitleTrack],
    filter_chain: Arc<codec::filter::FilterChain>,
    output_dir: Option<&Path>,
    sink: Arc<dyn ProgressSink>,
    // Splice plan: explicit clips (concat). Empty ⇒ single `input`, trimmed to
    // the spec's `[trim_start, trim_end)` window if set, else un-spliced.
    spliced_clips: Vec<ClipSource>,
    // Pre-summed trimmed/concat frame total; `None` ⇒ derive from the source.
    effective_total: Option<u64>,
    // The source's late video start `(ticks, timescale)`; `(0, 1)` for none.
    video_delay: (u64, u32),
) -> Result<(Vec<RungOutput>, Option<PathBuf>, Option<PathBuf>)> {
    let root = match output_dir {
        Some(d) => d.to_path_buf(),
        None => tempfile::Builder::new()
            .prefix("rivet-hls-")
            .tempdir()
            .context("creating HLS temp dir")?
            .keep(),
    };

    let timescale = (frame_rate * 1000.0).round().max(1.0) as u32;
    let per_frame_ticks = (timescale as f64 / frame_rate.max(1.0)).round().max(1.0) as u32;
    let keyframe_interval = keyframe_interval_for_segment(segment_seconds as f64, frame_rate);
    let segment_target_ticks = (keyframe_interval as u64) * (per_frame_ticks as u64);

    // Resolve the decode plan. Concat clips win; otherwise a single input honors
    // the spec trim window (empty plan ⇒ the multi-GPU pump's input fallback).
    let start_frame = trim_frame(spec.trim_start, frame_rate).unwrap_or(0);
    let end_frame = trim_frame(spec.trim_end, frame_rate);
    let spliced_clips = if !spliced_clips.is_empty() {
        spliced_clips
    } else if start_frame == 0 && end_frame.is_none() {
        Vec::new()
    } else {
        vec![ClipSource {
            cfg: pump_cfg_for(header, spec, Arc::clone(&filter_chain), None),
            input: input.clone(),
            start_frame,
            end_frame,
        }]
    };

    let source_total = if header.info.total_frames > 0 {
        header.info.total_frames
    } else {
        (header.info.duration * frame_rate).round().max(0.0) as u64
    };
    let total_input_frames = effective_total.unwrap_or_else(|| match end_frame {
        Some(end) => end.saturating_sub(start_frame),
        None => source_total.saturating_sub(start_frame),
    });

    let (output_color_metadata, output_pixel_format) =
        spec.resolve_output(header.info.color_metadata, header.info.pixel_format);
    // A policy that leaves nothing to encode the output on is refused here,
    // by name, before a frame is decoded — see `gpu_pool_for_policy`. Every
    // worker builds its encoder for the output's format on the card it
    // leased, so the pool holds only cards that take that format.
    let gpu_pool =
        multigpu::gpu_pool_for_policy(spec.encode_policy, spec.video_codec.codec(), output_pixel_format)?;
    // Bitrate rungs are coded by the software encoder only; the ladder's
    // workers lease from this pool and never read the pin.
    multigpu::check_rate_pool(spec, &gpu_pool, output_pixel_format, None)?;
    let params = MultiGpuParams {
        input,
        spliced_clips,
        codec: spec.video_codec.codec(),
        rungs: &spec.rungs,
        header: header.clone(),
        source_color_metadata: header.info.color_metadata,
        source_pixel_format: header.info.pixel_format,
        tonemap_to_sdr: spec.tonemaps(),
        output_color_metadata,
        output_pixel_format,
        needs_downsample: needs_chroma_downsample(header.info.pixel_format),
        chroma_downsample: spec.chroma_downsample,
        filters: Arc::clone(&filter_chain),
        frame_rate,
        gpu_pool,
        host: multigpu::HostCards::Detected,
        gpu_indices: multigpu::policy_gpu_indices(spec.encode_policy),
        decode: spec.decode_policy,
        encode: spec.encode_policy,
        output_root: root.clone(),
        timescale,
        per_frame_ticks,
        keyframe_interval,
        segment_target_ticks,
        total_input_frames,
        // HLS segments are independent files — no stitched seams to flatten.
        constant_qp: false,
        cancel: None,
        video_delay_ticks: container::edit::rescale_round(video_delay.0, timescale, video_delay.1),
    };
    let manifests = multigpu::run_multigpu_hls(params, Arc::clone(&sink)).await?;

    let mut rung_outputs = Vec::new();
    let mut video_specs = Vec::new();
    for (idx, m) in manifests.into_iter().enumerate() {
        match m {
            Some(rm) => {
                let dir = root.join(&rm.relative_dir);
                let bytes = dir_size(&dir);
                video_specs.push(build_video_variant_spec(&rm, frame_rate, bytes));
                rung_outputs.push(RungOutput {
                    label: rm.label.clone(),
                    width: rm.width,
                    height: rm.height,
                    frames: total_input_frames,
                    bytes,
                    artifact: RungArtifact::HlsRendition {
                        dir,
                        relative_dir: rm.relative_dir,
                    },
                });
            }
            None => {
                if let Some(rung) = spec.rungs.get(idx) {
                    report_failed(sink.as_ref(), idx, rung, "rung produced no segments");
                }
            }
        }
    }
    if rung_outputs.is_empty() {
        bail!("all {} rung(s) failed", spec.rungs.len());
    }

    let audio_spec = match audio {
        Some(a) => build_audio_rendition(&root, a, segment_seconds).context("building HLS audio rendition")?,
        None => None,
    };
    // Subtitles segment on the first rendition's grid. Every rendition shares
    // that grid (segments open on the same keyframes), so the WebVTT segment
    // boundaries agree with every variant a player might be on.
    let subtitle_specs = build_subtitle_renditions(&root, subtitles, &video_specs)
        .context("building HLS subtitle renditions")?;
    add_rendition_rates(&mut video_specs, audio_spec.as_ref(), &subtitle_specs);
    let target_duration = segment_seconds.ceil() as u32;
    let paths = write_hls_package(
        &root,
        &video_specs,
        audio_spec.as_ref(),
        &subtitle_specs,
        target_duration,
    )
    .context("writing HLS package")?;

    Ok((rung_outputs, Some(root), Some(paths.master_path)))
}

fn build_video_variant_spec(rm: &RungManifest, frame_rate: f64, bytes: u64) -> VideoVariantSpec {
    let codec_string = cmaf_util::codec_string_from_init(&rm.manifest.init_path)
        .unwrap_or_else(|_| "av01.0.08M.08.0.110.01.01.01.0".to_string());
    // RFC 8216 §4.3.4.2: BANDWIDTH is the peak segment bit rate and
    // AVERAGE-BANDWIDTH the average one, both measured from the segments the
    // rung wrote. A manifest with no timed segment falls back to the
    // directory's bytes over its duration for both.
    let (average, peak) = cmaf_util::measure_bandwidth(&rm.manifest);
    let (average, bandwidth) = if peak > 0 {
        (average, peak)
    } else {
        let dur = rm.manifest.duration_seconds().max(0.001);
        let rate = ((bytes as f64 * 8.0) / dur) as u32;
        (rate, rate)
    };
    VideoVariantSpec {
        width: rm.width,
        height: rm.height,
        frame_rate,
        average_bandwidth_bps: average,
        bandwidth_bps: bandwidth,
        codec_string,
        supplemental_codecs: None,
        video_range: None,
        relative_dir: rm.relative_dir.clone(),
        manifest: rm.manifest.clone(),
    }
}

/// Grow every variant's BANDWIDTH and AVERAGE-BANDWIDTH by the renditions it
/// plays with. RFC 8216 §4.3.4.2: a variant's BANDWIDTH is "the largest sum
/// of peak segment bit rates that is produced by any playable combination of
/// Renditions", and AVERAGE-BANDWIDTH the same sum of average rates. Every
/// variant here plays with the one audio rendition, when there is one, and
/// with any one subtitle rendition, so each adds the audio's rates and the
/// largest subtitle rendition's. The video rates alone let a player pick a
/// variant its link could not carry once the audio was added.
fn add_rendition_rates(
    video: &mut [VideoVariantSpec],
    audio: Option<&AudioVariantSpec>,
    subtitles: &[SubtitleVariantSpec],
) {
    let (audio_avg, audio_peak) = audio.map_or((0, 0), |a| cmaf_util::measure_bandwidth(&a.manifest));
    let (subs_avg, subs_peak) = subtitles
        .iter()
        .map(|s| cmaf_util::measure_segments(&s.manifest.segments, s.manifest.timescale))
        .fold((0, 0), |(avg, peak), (a, p)| (avg.max(a), peak.max(p)));
    for v in video {
        v.average_bandwidth_bps = v.average_bandwidth_bps.saturating_add(audio_avg).saturating_add(subs_avg);
        v.bandwidth_bps = v.bandwidth_bps.saturating_add(audio_peak).saturating_add(subs_peak);
    }
}

fn dir_size(dir: &Path) -> u64 {
    let mut total = 0;
    if let Ok(entries) = std::fs::read_dir(dir) {
        for e in entries.flatten() {
            if let Ok(meta) = e.metadata()
                && meta.is_file()
            {
                total += meta.len();
            }
        }
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;
    use container::cmaf::{CmafTrackManifest, SegmentInfo};

    /// A rung of `(bytes, seconds)` segments at a 90 kHz timescale.
    fn rung(segments: &[(u64, u64)]) -> RungManifest {
        RungManifest {
            rung_index: 0,
            width: 64,
            height: 64,
            label: "64p".into(),
            relative_dir: "video/64p".into(),
            manifest: CmafTrackManifest {
                init_path: PathBuf::from("no-init-here.mp4"),
                segments: segments
                    .iter()
                    .enumerate()
                    .map(|(i, &(bytes, secs))| SegmentInfo {
                        sequence_number: i as u32 + 1,
                        path: PathBuf::new(),
                        byte_size: bytes,
                        duration_ticks: secs * 90_000,
                    })
                    .collect(),
                timescale: 90_000,
            },
        }
    }

    /// BANDWIDTH is the largest segment's rate and AVERAGE-BANDWIDTH the
    /// rung's average, not the peak twice.
    #[test]
    fn average_bandwidth_is_the_average_and_bandwidth_the_peak() {
        let v = build_video_variant_spec(&rung(&[(100_000, 1), (50_000, 1), (60_000, 2)]), 30.0, 210_000);
        assert_eq!(v.bandwidth_bps, 800_000, "peak: 100 000 bytes in one second");
        assert_eq!(v.average_bandwidth_bps, 420_000, "average: 210 000 bytes in four seconds");
    }

    /// Every variant's rates grow by the audio rendition's and by the
    /// largest subtitle rendition's — peak by peak, average by average —
    /// and a package with neither is left as it was.
    #[test]
    fn variant_rates_include_the_audio_and_the_largest_subtitle_rendition() {
        let track = |segs: &[(u64, u64)]| rung(segs).manifest;
        let audio = AudioVariantSpec {
            codec_string: "mp4a.40.2".into(),
            channels: 2,
            sample_rate: 48_000,
            relative_dir: "audio".into(),
            language: "und".into(),
            name: "Audio".into(),
            // Peak 16 000 bytes in one second, average 24 000 in two.
            manifest: track(&[(16_000, 1), (8_000, 1)]),
        };
        let subs = |segs: &[(u64, u64)]| SubtitleVariantSpec {
            language: "en".into(),
            name: "English".into(),
            relative_dir: "subs/en".into(),
            default: false,
            manifest: container::webvtt::WebVttManifest { segments: track(segs).segments, timescale: 90_000 },
        };
        let video = || vec![build_video_variant_spec(&rung(&[(100_000, 1), (50_000, 1)]), 30.0, 150_000)];
        let mut v = video();
        add_rendition_rates(&mut v, Some(&audio), &[subs(&[(100, 1), (100, 1)]), subs(&[(300, 1), (100, 1)])]);
        assert_eq!(v[0].bandwidth_bps, 800_000 + 128_000 + 2_400, "video peak + audio peak + largest subtitle peak");
        assert_eq!(v[0].average_bandwidth_bps, 600_000 + 96_000 + 1_600, "the same sum of averages");
        let mut bare = video();
        add_rendition_rates(&mut bare, None, &[]);
        assert_eq!((bare[0].bandwidth_bps, bare[0].average_bandwidth_bps), (800_000, 600_000));
    }
}
