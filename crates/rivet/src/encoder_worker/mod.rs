//! Per-segment encoder worker: pop a chunk → encode K frames →
//! emit one CMAF segment file → repeat.
//!
//! v2 multi-GPU model (2026-05-11): each worker owns one GPU lease
//! and one encoder for its lifetime, but builds a fresh
//! `CmafVideoMuxer` per claimed segment. The muxer is configured
//! with the segment's index + base decode time so the on-disk
//! filename + tfdt match what a single-encoder pipeline would
//! produce. Helpers attaching mid-flight just start popping from
//! the queue's current head; no decode-and-discard.
//!
//! Workers exit when `queue.pop()` returns `None` (pump closed +
//! queue drained). The returned `WorkerOutput` lists every segment
//! the worker wrote so the orchestrator can merge contributions
//! into the per-rung manifest.

mod invariant;
mod config;
mod cmaf_worker;
mod chunk_worker;
#[cfg(test)]
mod tests;

pub use invariant::{
    Av1Invariant, H26xInvariant, InvariantCheck, RungCodecInvariant,
    validate_or_set_rung_invariant,
};
pub use config::{EncoderWorkerConfig, WorkerOutput};
pub use cmaf_worker::{UnitOutcome, encode_segment_unit, run_encoder_worker_blocking};
pub use chunk_worker::{ChunkPackets, run_chunk_encoder_worker_blocking};

use codec::encode::EncoderConfig;

/// Build the per-rung `EncoderConfig` from the resolved output format + quality
/// knobs. Shared by the CMAF and packet workers.
fn build_enc_config(cfg: &EncoderWorkerConfig) -> EncoderConfig {
    EncoderConfig {
        codec: cfg.codec,
        width: cfg.width,
        height: cfg.height,
        frame_rate: cfg.frame_rate,
        quality: cfg.quality,
        speed_preset: cfg.speed_preset,
        keyframe_interval: cfg.keyframe_interval,
        threads: cfg.threads,
        pixel_format: cfg.output_pixel_format,
        color_metadata: cfg.output_color_metadata,
        gpu_index: cfg.gpu_index,
        gpu_vendor: cfg.gpu_vendor,
        target: cfg.target,
        tier: cfg.tier,
        constant_qp: cfg.constant_qp,
        ..EncoderConfig::default()
    }
}
