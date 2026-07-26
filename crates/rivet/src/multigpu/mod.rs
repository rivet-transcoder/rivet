//! Multi-GPU reactive variant phase — **the rung benefit**.
//!
//! Decode the source **once** and dynamically schedule every rung's CMAF
//! segments across all available GPUs using a fair lease pool with mid-flight
//! helper dispatch:
//!
//! ```text
//!   decode pump (decode once)
//!        │  fan out normalized frames
//!        ▼
//!   per-rung scaler ──► SegmentChunkQueue ──► encoder worker (holds a GpuLease)
//!                                        ──► helper worker (claims a freed lease)
//! ```
//!
//! - One encoder per GPU at a time ([`GpuPool`] enforces it — concurrent
//!   NVENC sessions on one context deadlock).
//! - A fast rung releases its lease early; the **helper dispatcher** grabs the
//!   freed lease and attaches an extra worker to a still-busy rung, so a slow
//!   rung finishes sooner. Segment work is the unit of parallelism.
//! - Helpers may land on a different GPU **vendor** than the rung's first
//!   worker; the per-rung AV1 **codec invariant** ([`RungCodecInvariant`])
//!   guarantees every contributed segment shares the `av1C` contract, so a
//!   cross-vendor (NVENC + QSV) rendition still decodes cleanly. A mismatched
//!   helper requeues its chunk and exits — the run never aborts on it.
//!
//! Storage/transport specifics stay out of the engine: progress is reported
//! through the generic [`ProgressSink`], so a consumer can layer an uploader
//! (object storage, a status queue, …) on top by watching `RungStatus::Completed`.

mod gpu_policy;
mod hls;
mod single_file;

pub use gpu_policy::{detect_gpu_pool, gpu_pool_for_policy, policy_gpu_indices, serial_gpu_for_policy};
pub use hls::run_multigpu_hls;
pub use single_file::{RungPackets, run_multigpu_single_file};

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use bytes::Bytes;
use codec::frame::{ColorMetadata, PixelFormat, VideoCodec};
use container::cmaf::CmafTrackManifest;
use container::streaming::DemuxHeader;

use crate::decode_pump::{ClipSource, DecodePumpConfig};
use crate::gpu_pool::GpuPool;
use crate::progress::{ProgressSink, RungProgress, RungStatus};
use crate::spec::Rung;

pub(super) const QUEUE_CAPACITY: usize = 2;
pub(super) const FANOUT_CHANNEL_CAPACITY: usize = 4;
pub(super) const HELPER_POLL_INTERVAL: std::time::Duration = std::time::Duration::from_millis(200);
pub(super) const PROGRESS_TICK: std::time::Duration = std::time::Duration::from_millis(500);

/// One rung's finalized CMAF manifest.
#[derive(Debug, Clone)]
pub struct RungManifest {
    pub rung_index: usize,
    pub width: u32,
    pub height: u32,
    pub label: String,
    /// Directory relative to the asset root, e.g. `"video/720p"`.
    pub relative_dir: String,
    pub manifest: CmafTrackManifest,
}

/// Inputs to [`run_multigpu_hls`].
pub struct MultiGpuParams<'a> {
    pub input: Bytes,
    /// Output video codec — drives the per-worker encoder dispatch, the codec
    /// invariant parse, and the stitch muxer's sample-entry choice.
    pub codec: VideoCodec,
    pub rungs: &'a [Rung],
    pub header: DemuxHeader,
    pub source_color_metadata: ColorMetadata,
    pub source_pixel_format: PixelFormat,
    /// Whether the decode pump tonemaps HDR→SDR (from the spec's `ColorPolicy`).
    pub tonemap_to_sdr: bool,
    /// Resolved **output** color metadata + pixel format the encoders target
    /// (from `OutputSpec::resolve_output`).
    pub output_color_metadata: ColorMetadata,
    pub output_pixel_format: PixelFormat,
    pub needs_downsample: bool,
    /// Prepared per-frame video filter chain applied in the decode pump (before
    /// scaling). Overlay images are loaded once at prepare time.
    pub filters: Arc<codec::filter::FilterChain>,
    pub frame_rate: f64,
    pub gpu_pool: Arc<GpuPool>,
    /// GPU indices the encode policy selected, in detection order. The decode
    /// pump pins to these (round-robin for per-rung pumps) so decode honors the
    /// same `Family` / `SingleGpu` / `AllGpus` constraint as encode. Empty ⇒
    /// the decoder dispatch auto-selects (legacy behavior).
    pub gpu_indices: Vec<u32>,
    /// Explicit decode-pump GPU override. `Some(i)` forces every decode pump
    /// onto GPU `i` regardless of `gpu_indices`; `None` follows the policy.
    pub decode_gpu: Option<u32>,
    pub output_root: PathBuf,
    pub timescale: u32,
    pub per_frame_ticks: u32,
    pub keyframe_interval: u32,
    pub segment_target_ticks: u64,
    pub total_input_frames: u64,
    /// Force constant-QP chunk encoding (single-file `ChunkSeamMode::ParallelConstQp`)
    /// so stitched chunk seams are quality-flat. `false` for HLS (segments are
    /// independent) and the default `Parallel` single-file mode.
    pub constant_qp: bool,
    /// Decode plan for the HLS pump: one entry per spliced clip, each carrying
    /// its own decoder config + `[start_frame, end_frame)` trim range. A single
    /// whole-input clip is the un-spliced case — behaviourally identical to the
    /// old single-input pump (`run_shared_*` is a one-whole-clip wrapper). The
    /// per-clip `cfg.gpu_index` is a placeholder; `clip_sources_for` overrides
    /// it with each pump's GPU. Unused by the single-file multi-GPU path (which
    /// decodes from `input`).
    pub spliced_clips: Vec<ClipSource>,
}

impl MultiGpuParams<'_> {
    /// Resolve the decode-pump GPU for the `i`-th per-rung pump (or the shared
    /// pump when `i == 0`): the explicit `decode_gpu` override wins, else the
    /// policy's GPU indices round-robin, else `None` (decoder auto-select).
    pub(super) fn decode_gpu_for(&self, i: usize) -> Option<u32> {
        if self.decode_gpu.is_some() {
            return self.decode_gpu;
        }
        if self.gpu_indices.is_empty() {
            return None;
        }
        Some(self.gpu_indices[i % self.gpu_indices.len()])
    }

    /// Per-clip decode sources for a pump pinned to `gpu`. When `spliced_clips`
    /// is empty (the un-spliced case) this is one whole clip built from `input`
    /// + the header — behaviourally identical to the old single-input pump.
    /// Otherwise it clones the splice plan, overriding each clip's `gpu_index`
    /// so every pump honours its assigned GPU while keeping the per-clip
    /// codec / color / trim.
    pub(super) fn clip_sources_for(&self, gpu: Option<u32>) -> Vec<ClipSource> {
        if self.spliced_clips.is_empty() {
            return vec![ClipSource {
                cfg: DecodePumpConfig {
                    codec_name: self.header.codec.clone(),
                    info_for_decoder: self.header.info.clone(),
                    source_color_metadata: self.source_color_metadata,
                    source_pixel_format: self.source_pixel_format,
                    needs_downsample: self.needs_downsample,
                    tonemap_to_sdr: self.tonemap_to_sdr,
                    gpu_index: gpu,
                    filters: self.filters.clone(),
                },
                input: self.input.clone(),
                start_frame: 0,
                end_frame: None,
            }];
        }
        self.spliced_clips
            .iter()
            .map(|c| ClipSource {
                cfg: DecodePumpConfig { gpu_index: gpu, ..c.cfg.clone() },
                input: c.input.clone(),
                start_frame: c.start_frame,
                end_frame: c.end_frame,
            })
            .collect()
    }
}

/// Per-job constants shared by every encoder worker.
#[derive(Clone)]
pub(super) struct WorkerCtx {
    pub(super) codec: VideoCodec,
    pub(super) frame_rate: f64,
    pub(super) output_color_metadata: ColorMetadata,
    pub(super) output_pixel_format: PixelFormat,
    pub(super) timescale: u32,
    pub(super) per_frame_ticks: u32,
    pub(super) keyframe_interval: u32,
    pub(super) segment_target_ticks: u64,
    pub(super) output_root: PathBuf,
    pub(super) constant_qp: bool,
}

/// Periodic per-rung progress reporter. Reads the shared frame counters and
/// emits `Running` updates until stopped; skips rungs already finalized.
pub(super) fn spawn_progress_reporter(
    rungs: Vec<Rung>,
    frames_encoded: Vec<Arc<AtomicU64>>,
    bytes_encoded: Vec<Arc<AtomicU64>>,
    finalized: Arc<Vec<AtomicBool>>,
    total_input_frames: u64,
    sink: Arc<dyn ProgressSink>,
    stop: Arc<AtomicBool>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        loop {
            if stop.load(Ordering::Acquire) {
                break;
            }
            tokio::time::sleep(PROGRESS_TICK).await;
            for (idx, rung) in rungs.iter().enumerate() {
                if finalized[idx].load(Ordering::Acquire) {
                    continue;
                }
                let done = frames_encoded[idx].load(Ordering::Relaxed);
                let bytes = bytes_encoded[idx].load(Ordering::Relaxed);
                report(
                    sink.as_ref(),
                    idx,
                    rung,
                    RungStatus::Running,
                    done,
                    Some(total_input_frames),
                    0,
                    bytes,
                    None,
                );
            }
        }
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn report(
    sink: &dyn ProgressSink,
    rung_index: usize,
    rung: &Rung,
    status: RungStatus,
    frames_done: u64,
    frames_total: Option<u64>,
    segments: u32,
    bytes_out: u64,
    message: Option<String>,
) {
    let percent = match status {
        RungStatus::Completed => 100.0,
        RungStatus::Pending => 0.0,
        _ => match frames_total {
            Some(t) if t > 0 => ((frames_done as f32 / t as f32) * 100.0).min(99.0),
            _ => 1.0,
        },
    };
    sink.on_rung(RungProgress {
        rung_index,
        label: rung.label.clone(),
        width: rung.width,
        height: rung.height,
        status,
        percent,
        frames_done,
        frames_total,
        segments_written: segments,
        bytes_out,
        message,
    });
}
