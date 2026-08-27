//! Worker configuration + output types.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::RwLock;
use codec::frame::{ColorMetadata, PixelFormat, VideoCodec};
use container::cmaf::SegmentInfo;
use super::RungCodecInvariant;

#[derive(Clone)]
pub struct EncoderWorkerConfig {
    pub rung_idx: usize,
    /// Output video codec (AV1 / H.264 / H.265) — drives the encoder dispatch
    /// and the per-rung codec invariant parse.
    pub codec: VideoCodec,
    pub width: u32,
    pub height: u32,
    pub frame_rate: f64,
    /// Legacy CRF escape hatch (`u8::MAX` = derive from `target`).
    pub quality: u8,
    /// Speed preset escape hatch (`u8::MAX` = derive from `tier`).
    pub speed_preset: u8,
    /// Perceptual quality target (used when `quality` is the sentinel).
    pub target: codec::encode::tuning::QualityTarget,
    /// Speed tier (used when `speed_preset` is the sentinel).
    pub tier: codec::encode::tuning::SpeedTier,
    /// Per-rung knob set, already resolved by the caller's `EncodePolicy`.
    ///
    /// The worker does not resolve policy itself, and deliberately: a rung's
    /// position in the ladder is the *caller's* fact, and a worker that
    /// re-derived it would have to be told the ladder anyway. It is handed the
    /// answer. `Default` is inert — see `codec::encode::tuning::EncodeOverrides`.
    pub overrides: codec::encode::tuning::EncodeOverrides,
    pub threads: usize,
    pub gpu_index: Option<u32>,
    pub gpu_vendor: Option<codec::gpu::GpuVendor>,
    /// Ask `select_encoder` for this backend by name instead of running its
    /// dispatch chain. `None` (a card) runs the chain, which the lease's
    /// vendor pin steers. A software lease sets the codec's software backend
    /// here so every chunk skips the hardware probes the pool already knows
    /// will decline — one `detect_gpus` and two "init failed, falling back"
    /// warnings per chunk otherwise, on a ladder of thousands of chunks.
    pub backend: Option<codec::encode::EncoderBackend>,
    /// Resolved **output** color metadata + pixel format (the encoder's input
    /// format and bitstream signaling). The engine computes these from the
    /// `OutputSpec`'s `ColorPolicy` / `BitDepth` via `resolve_output`, so the
    /// worker no longer folds HDR→SDR itself — it just encodes to this format.
    pub output_color_metadata: ColorMetadata,
    pub output_pixel_format: PixelFormat,
    /// Prefer constant-QP rate control (seam-flat chunked single-file under
    /// `ChunkSeamMode::ParallelConstQp`). Forwarded to `EncoderConfig.constant_qp`.
    pub constant_qp: bool,
    pub timescale: u32,
    pub per_frame_ticks: u32,
    pub keyframe_interval: u32,
    pub segment_target_ticks: u64,
    pub output_dir: PathBuf,
    /// Shared per-rung codec invariant slot. First worker on the rung
    /// SETS it; helpers (any vendor) COMPARE on their first packet.
    /// On mismatch the helper requeues its chunk and exits cleanly so
    /// the run continues without it — never aborts mission-critical
    /// jobs. See `validate_or_set_rung_invariant` + the requeue path
    /// in `run_encoder_worker_blocking`.
    pub rung_invariant: Arc<RwLock<Option<RungCodecInvariant>>>,
}

#[derive(Debug, Clone)]
pub struct WorkerOutput {
    pub gpu_index: Option<u32>,
    pub segments: Vec<SegmentInfo>,
}
