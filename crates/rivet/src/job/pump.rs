use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use bytes::Bytes;

use container::hls::{VideoVariantSpec, write_hls_package};
use container::streaming::DemuxHeader;

use crate::cmaf_util::{self, keyframe_interval_for_segment};
use crate::decode_pump::ClipSource;
use crate::multigpu::{self, MultiGpuParams, RungManifest};
use crate::progress::ProgressSink;
use crate::spec::OutputSpec;
use crate::validate::needs_chroma_downsample;

use super::{RungArtifact, RungOutput, report_failed};
use super::audio::{PreparedAudio, build_audio_rendition};
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
    crate::decode_pump::DecodePumpConfig {
        codec_name: header.codec.clone(),
        info_for_decoder: header.info.clone(),
        source_color_metadata: header.info.color_metadata,
        source_pixel_format: header.info.pixel_format,
        needs_downsample: needs_chroma_downsample(header.info.pixel_format),
        tonemap_to_sdr: spec.tonemaps(),
        gpu_index: gpu,
        sample_range: None,
        rotation_degrees: header.rotation_degrees,
        filters,
    }
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
    filter_chain: Arc<codec::filter::FilterChain>,
    output_dir: Option<&Path>,
    sink: Arc<dyn ProgressSink>,
    // Splice plan: explicit clips (concat). Empty ⇒ single `input`, trimmed to
    // the spec's `[trim_start, trim_end)` window if set, else un-spliced.
    spliced_clips: Vec<ClipSource>,
    // Pre-summed trimmed/concat frame total; `None` ⇒ derive from the source.
    effective_total: Option<u64>,
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

    let gpu_pool = multigpu::gpu_pool_for_policy(spec.encode_policy, spec.video_codec.codec());
    let (output_color_metadata, output_pixel_format) =
        spec.resolve_output(header.info.color_metadata, header.info.pixel_format);
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
        filters: Arc::clone(&filter_chain),
        frame_rate,
        gpu_pool,
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
    let target_duration = segment_seconds.ceil() as u32;
    let paths = write_hls_package(&root, &video_specs, audio_spec.as_ref(), target_duration)
        .context("writing HLS package")?;

    Ok((rung_outputs, Some(root), Some(paths.master_path)))
}

fn build_video_variant_spec(rm: &RungManifest, frame_rate: f64, bytes: u64) -> VideoVariantSpec {
    let codec_string = cmaf_util::codec_string_from_init(&rm.manifest.init_path)
        .unwrap_or_else(|_| "av01.0.08M.08.0.110.01.01.01.0".to_string());
    let (_avg, peak) = cmaf_util::measure_bandwidth(&rm.manifest);
    let bandwidth = if peak > 0 {
        peak
    } else {
        let dur = rm.manifest.duration_seconds().max(0.001);
        ((bytes as f64 * 8.0) / dur) as u32
    };
    VideoVariantSpec {
        width: rm.width,
        height: rm.height,
        frame_rate,
        average_bandwidth_bps: bandwidth,
        bandwidth_bps: bandwidth,
        codec_string,
        supplemental_codecs: None,
        video_range: None,
        relative_dir: rm.relative_dir.clone(),
        manifest: rm.manifest.clone(),
    }
}

fn dir_size(dir: &Path) -> u64 {
    let mut total = 0;
    if let Ok(entries) = std::fs::read_dir(dir) {
        for e in entries.flatten() {
            if let Ok(meta) = e.metadata() {
                if meta.is_file() {
                    total += meta.len();
                }
            }
        }
    }
    total
}
