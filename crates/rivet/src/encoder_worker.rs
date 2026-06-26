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

use anyhow::{Context, Result, anyhow};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::RwLock;

use codec::encode::{self, EncoderConfig};
use codec::frame::{ColorMetadata, PixelFormat};
use codec::pixel_format::{Av1SequenceHeader, parse_av1_sequence_header};
use container::cmaf::{CmafVideoMuxer, CmafVideoMuxerOptions, SegmentInfo};
use tokio::sync::mpsc;

use crate::cmaf_util::add_packet_with_segment_flush;
use crate::frame_queue::{SegmentChunk, SegmentChunkQueue};

/// Mandatory AV1 sequence-header fields that every encoder
/// contributing segments to a single rendition MUST agree on.
///
/// Why these specific fields: each is part of the codec-init contract
/// that the player sets up once from `av1C` and expects to hold for
/// every segment. The decoder re-parses the inline OBU sequence
/// header in each segment's IDR; if its parsed values disagree with
/// the av1C from `init.mp4` on any of these fields, strict decoders
/// (dav1d in conformance mode, Safari AVFoundation, hls.js+libdav1d)
/// will reject the segment. Optional fields not listed here (timing
/// info presence, decoder model presence, film grain `present` flag,
/// operating-point details) are tolerated by every major player; we
/// deliberately don't check them so that NVENC + QSV + AMF + rav1e
/// can co-exist on one rendition without cosmetic byte differences
/// triggering false rejections.
///
/// First worker on a rung SETS the invariant. Subsequent workers
/// (helpers from any vendor) COMPARE; mismatch fails the run loudly
/// instead of silently corrupting output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RungCodecInvariant {
    pub seq_profile: u8,
    pub seq_level_idx_0: u8,
    pub seq_tier_0: u8,
    pub bit_depth: u8,
    pub monochrome: bool,
    pub chroma_subsampling_x: bool,
    pub chroma_subsampling_y: bool,
    pub color_primaries: u8,
    pub transfer_characteristics: u8,
    pub matrix_coefficients: u8,
    pub color_range: bool,
    pub max_frame_width_minus1: u32,
    pub max_frame_height_minus1: u32,
    pub still_picture: bool,
}

impl RungCodecInvariant {
    pub fn from_sequence_header(sh: &Av1SequenceHeader) -> Self {
        Self {
            seq_profile: sh.seq_profile,
            seq_level_idx_0: sh.seq_level_idx_0,
            seq_tier_0: sh.seq_tier_0,
            bit_depth: sh.bit_depth,
            monochrome: sh.monochrome,
            chroma_subsampling_x: sh.chroma_subsampling_x,
            chroma_subsampling_y: sh.chroma_subsampling_y,
            color_primaries: sh.color_primaries,
            transfer_characteristics: sh.transfer_characteristics,
            matrix_coefficients: sh.matrix_coefficients,
            color_range: sh.color_range,
            max_frame_width_minus1: sh.max_frame_width_minus1,
            max_frame_height_minus1: sh.max_frame_height_minus1,
            still_picture: sh.still_picture,
        }
    }

    /// Human-readable diff for error messages.
    fn describe_diff(&self, other: &Self) -> String {
        let mut diffs = Vec::new();
        macro_rules! diff_field {
            ($field:ident) => {
                if self.$field != other.$field {
                    diffs.push(format!(
                        "{}: rung={:?}, this worker={:?}",
                        stringify!($field),
                        self.$field,
                        other.$field
                    ));
                }
            };
        }
        diff_field!(seq_profile);
        diff_field!(seq_level_idx_0);
        diff_field!(seq_tier_0);
        diff_field!(bit_depth);
        diff_field!(monochrome);
        diff_field!(chroma_subsampling_x);
        diff_field!(chroma_subsampling_y);
        diff_field!(color_primaries);
        diff_field!(transfer_characteristics);
        diff_field!(matrix_coefficients);
        diff_field!(color_range);
        diff_field!(max_frame_width_minus1);
        diff_field!(max_frame_height_minus1);
        diff_field!(still_picture);
        diffs.join("; ")
    }
}

/// Outcome of comparing a worker's first packet against the rung's
/// codec invariant. The caller — `run_encoder_worker_blocking` —
/// branches on this to decide whether to keep encoding, soft-fail
/// (requeue the chunk for another worker), or hard-fail (parse error
/// from a malformed bitstream).
#[derive(Debug)]
pub enum InvariantCheck {
    /// First worker on the rung. Invariant has been recorded.
    SetByThisWorker,
    /// Matches the rung's invariant. Proceed to publish.
    Matched,
    /// Mandatory fields mismatch. Worker should requeue its chunk and
    /// exit cleanly; the rung continues with workers whose vendors
    /// agree with the invariant the first worker set. **Mission-
    /// critical jobs DO NOT abort on this** — only this one helper's
    /// contribution is lost, and another worker picks up the chunk.
    Mismatched { diff: String },
}

/// Parse a worker's first packet, derive the codec invariant, and
/// compare-or-set it against the per-rung slot. Returns
/// [`InvariantCheck`] on a successful parse; an `Err` only on
/// malformed bitstream (the encoder failed to emit an
/// `OBU_SEQUENCE_HEADER` at all, which is a configuration bug that
/// nothing downstream can recover from).
pub fn validate_or_set_rung_invariant(
    rung_idx: usize,
    gpu_vendor: Option<codec::gpu::GpuVendor>,
    slot: &RwLock<Option<RungCodecInvariant>>,
    first_packet: &[u8],
) -> Result<InvariantCheck> {
    let parsed = parse_av1_sequence_header(first_packet).ok_or_else(|| {
        anyhow!(
            "rung {} (vendor {:?}): could not parse AV1 sequence header from first encoded packet; \
             encoder did not emit OBU_SEQUENCE_HEADER as required for CMAF segment alignment",
            rung_idx,
            gpu_vendor,
        )
    })?;
    let observed = RungCodecInvariant::from_sequence_header(&parsed);

    // Fast path: read lock, check if set + matches.
    if let Some(existing) = &*slot.read().unwrap() {
        if existing == &observed {
            return Ok(InvariantCheck::Matched);
        }
        return Ok(InvariantCheck::Mismatched {
            diff: existing.describe_diff(&observed),
        });
    }
    // First worker — write under write-lock with double-check (race
    // against another worker setting the slot between read and write).
    let mut w = slot.write().unwrap();
    match &*w {
        Some(existing) if existing != &observed => Ok(InvariantCheck::Mismatched {
            diff: existing.describe_diff(&observed),
        }),
        Some(_) => Ok(InvariantCheck::Matched),
        None => {
            tracing::info!(
                rung_idx,
                gpu_vendor = ?gpu_vendor,
                seq_profile = observed.seq_profile,
                seq_level_idx_0 = observed.seq_level_idx_0,
                bit_depth = observed.bit_depth,
                "rung codec invariant captured from first worker"
            );
            *w = Some(observed);
            Ok(InvariantCheck::SetByThisWorker)
        }
    }
}

#[derive(Clone)]
pub struct EncoderWorkerConfig {
    pub rung_idx: usize,
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
    pub threads: usize,
    pub gpu_index: Option<u32>,
    pub gpu_vendor: Option<codec::gpu::GpuVendor>,
    /// Resolved **output** color metadata + pixel format (the encoder's input
    /// format and bitstream signaling). The engine computes these from the
    /// `OutputSpec`'s `ColorPolicy` / `PixelDepth` via `resolve_output`, so the
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

/// Run the encoder loop until the chunk queue is closed and drained.
/// Designed to be wrapped in `tokio::task::spawn_blocking`.
///
/// `progress_tx` receives the shared cumulative `frames_encoded_total`
/// after every encoded frame; the caller's drain task fires wire
/// events from this stream. Multiple workers bump the same counter,
/// so the progress reading stays monotonic across worker handoffs.
#[allow(clippy::too_many_arguments)]
pub fn run_encoder_worker_blocking(
    cfg: EncoderWorkerConfig,
    queue: Arc<SegmentChunkQueue>,
    rt: tokio::runtime::Handle,
    shared_frames_encoded: Arc<std::sync::atomic::AtomicU64>,
    progress_tx: mpsc::Sender<u64>,
) -> Result<WorkerOutput> {
    let enc_config = build_enc_config(&cfg);
    let encoder_color_metadata = cfg.output_color_metadata;

    let mut segments_written: Vec<SegmentInfo> = Vec::new();
    let mut init_segment_written = false;

    tracing::debug!(rung_idx = cfg.rung_idx, gpu_index = ?cfg.gpu_index, "encoder worker started; awaiting first chunk");
    loop {
        let chunk = match rt.block_on(queue.pop()) {
            Some(c) => c,
            None => break,
        };
        tracing::debug!(rung_idx = cfg.rung_idx, segment = chunk.segment_idx, frames = chunk.frames.len(), "encoder worker popped chunk");
        match encode_one_segment(
            &cfg,
            &enc_config,
            encoder_color_metadata,
            chunk,
            &mut init_segment_written,
            &shared_frames_encoded,
            &progress_tx,
        )? {
            SegmentOutcome::Wrote {
                info,
                segment_idx,
                frames,
            } => {
                let role = if segment_idx == 0 {
                    "primary"
                } else {
                    "worker"
                };
                tracing::info!(
                    rung_idx = cfg.rung_idx,
                    gpu_index = ?cfg.gpu_index,
                    role,
                    segment = segment_idx,
                    frames_encoded = frames,
                    "rung segment flushed",
                );
                segments_written.push(info);
            }
            SegmentOutcome::RequeuedOnMismatch {
                chunk: rejected,
                diff,
            } => {
                // Helper from a vendor whose AV1 sequence header diverges
                // from the rung's invariant on mandatory fields. Put the
                // chunk back at the head of the queue so a matching-vendor
                // worker (always at least the initial worker) picks it up.
                // Exit clean — the run completes without this helper.
                tracing::warn!(
                    rung_idx = cfg.rung_idx,
                    gpu_index = ?cfg.gpu_index,
                    gpu_vendor = ?cfg.gpu_vendor,
                    rejected_segment = rejected.segment_idx,
                    diff = %diff,
                    "encoder worker: codec invariant mismatch on first packet — \
                     requeuing chunk for a matching-vendor worker and exiting",
                );
                let _ = queue.push_front(rejected);
                break;
            }
        }
    }

    Ok(WorkerOutput {
        gpu_index: cfg.gpu_index,
        segments: segments_written,
    })
}

/// Outcome of an `encode_one_segment` call. `Wrote` is the happy
/// path; `RequeuedOnMismatch` returns the chunk verbatim so the outer
/// loop can put it back at the head of the queue for another worker.
enum SegmentOutcome {
    Wrote {
        info: SegmentInfo,
        segment_idx: usize,
        frames: usize,
    },
    RequeuedOnMismatch {
        chunk: SegmentChunk,
        diff: String,
    },
}

fn encode_one_segment(
    cfg: &EncoderWorkerConfig,
    enc_config: &EncoderConfig,
    encoder_color_metadata: ColorMetadata,
    chunk: SegmentChunk,
    init_segment_written: &mut bool,
    shared_frames_encoded: &std::sync::atomic::AtomicU64,
    progress_tx: &mpsc::Sender<u64>,
) -> Result<SegmentOutcome> {
    let write_init = chunk.segment_idx == 0 && !*init_segment_written;
    let muxer_options = CmafVideoMuxerOptions {
        first_segment_index: (chunk.segment_idx as u32) + 1,
        first_segment_base_decode_time: chunk.segment_idx as u64 * cfg.segment_target_ticks,
        write_init_segment: write_init,
    };
    let mut muxer = CmafVideoMuxer::new_with_options(
        &cfg.output_dir,
        cfg.width,
        cfg.height,
        cfg.timescale,
        encoder_color_metadata,
        muxer_options,
    )
    .with_context(|| {
        format!(
            "creating CmafVideoMuxer for segment {} in {}",
            chunk.segment_idx,
            cfg.output_dir.display()
        )
    })?;

    let mut encoder =
        encode::select_encoder(enc_config.clone(), None).context("creating encoder for segment")?;

    // Buffered packets emitted from the encoder, awaiting either
    // commit-to-muxer (after invariant validation passes) or discard
    // (on mismatch). The first packet's bytes are the AV1 sequence
    // header OBU that we feed to the invariant validator.
    let mut pending_packets: Vec<codec::encode::EncodedPacket> = Vec::new();
    let mut first_packet_decision: Option<bool> = None; // None=undecided, Some(true)=commit, Some(false)=reject

    let segment_idx = chunk.segment_idx;
    let frame_count = chunk.frames.len();

    for frame in &chunk.frames {
        encoder
            .send_frame(frame)
            .context("encoder.send_frame in worker")?;
        while let Some(packet) = encoder
            .receive_packet()
            .context("encoder.receive_packet in worker")?
        {
            if first_packet_decision.is_none() {
                match validate_or_set_rung_invariant(
                    cfg.rung_idx,
                    cfg.gpu_vendor,
                    &cfg.rung_invariant,
                    &packet.data,
                )? {
                    InvariantCheck::Matched | InvariantCheck::SetByThisWorker => {
                        first_packet_decision = Some(true);
                    }
                    InvariantCheck::Mismatched { diff } => {
                        // Discard everything in flight. The muxer hasn't
                        // flushed any segment yet (first packet of a
                        // chunk is far below the segment-duration target),
                        // and init.mp4 is only written by finalize() —
                        // which we don't call. Drop muxer + encoder
                        // implicitly when we return.
                        return Ok(SegmentOutcome::RequeuedOnMismatch { chunk, diff });
                    }
                }
                pending_packets.push(packet);
                continue;
            }
            // first_packet_decision == Some(true): commit
            // First drain any buffered packets we held back during
            // validation.
            if !pending_packets.is_empty() {
                for held in pending_packets.drain(..) {
                    add_packet_with_segment_flush(
                        &mut muxer,
                        &held,
                        cfg.per_frame_ticks,
                        cfg.segment_target_ticks,
                    )
                    .context("CMAF segment-flush add (held)")?;
                }
            }
            add_packet_with_segment_flush(
                &mut muxer,
                &packet,
                cfg.per_frame_ticks,
                cfg.segment_target_ticks,
            )
            .context("CMAF segment-flush add (worker)")?;
        }
        let n = shared_frames_encoded.fetch_add(1, std::sync::atomic::Ordering::AcqRel) + 1;
        let _ = progress_tx.try_send(n);
    }

    // Drain remaining held packets (e.g. if the only packets emitted
    // were buffered during the single validation step).
    if first_packet_decision == Some(true) && !pending_packets.is_empty() {
        for held in pending_packets.drain(..) {
            add_packet_with_segment_flush(
                &mut muxer,
                &held,
                cfg.per_frame_ticks,
                cfg.segment_target_ticks,
            )
            .context("CMAF segment-flush add (final-held)")?;
        }
    }

    encoder.flush().context("encoder.flush in worker")?;
    while let Some(packet) = encoder
        .receive_packet()
        .context("encoder.receive_packet after flush")?
    {
        add_packet_with_segment_flush(
            &mut muxer,
            &packet,
            cfg.per_frame_ticks,
            cfg.segment_target_ticks,
        )
        .context("CMAF segment-flush add post-flush (worker)")?;
    }

    let manifest = muxer
        .finalize()
        .context("finalize CmafVideoMuxer (per-segment worker)")?;

    if write_init {
        *init_segment_written = true;
    }

    let info = manifest
        .segments
        .last()
        .ok_or_else(|| {
            anyhow::anyhow!(
                "encoder worker produced no segment for chunk idx {} (rung {}, gpu {:?}); \
                 frames in chunk = {}",
                segment_idx,
                cfg.rung_idx,
                cfg.gpu_index,
                frame_count,
            )
        })?
        .clone();
    Ok(SegmentOutcome::Wrote {
        info,
        segment_idx,
        frames: frame_count,
    })
}

// ---------------------------------------------------------------------------
// Single-file chunked encode: workers collect packets (instead of writing CMAF
// segments) so the orchestrator can stitch them, in segment order, into one MP4.
// ---------------------------------------------------------------------------

/// One chunk's encoded packets, in encode (= display, no B-frames) order.
#[derive(Debug)]
pub struct ChunkPackets {
    pub segment_idx: usize,
    pub packets: Vec<encode::EncodedPacket>,
}

/// Build the per-rung `EncoderConfig` from the resolved output format + quality
/// knobs. Shared by the CMAF and packet workers.
fn build_enc_config(cfg: &EncoderWorkerConfig) -> EncoderConfig {
    EncoderConfig {
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

/// Encoder worker that COLLECTS packets per chunk (single-file path). Each
/// chunk is encoded by a fresh encoder (first frame an IDR); the cross-vendor
/// codec invariant is enforced on the first packet (mismatch → requeue + exit,
/// exactly like the CMAF worker). Ordered `ChunkPackets` are pushed to `out`.
#[allow(clippy::too_many_arguments)]
pub fn run_chunk_encoder_worker_blocking(
    cfg: EncoderWorkerConfig,
    queue: Arc<SegmentChunkQueue>,
    rt: tokio::runtime::Handle,
    shared_frames_encoded: Arc<std::sync::atomic::AtomicU64>,
    progress_tx: mpsc::Sender<u64>,
    out: Arc<std::sync::Mutex<Vec<ChunkPackets>>>,
) -> Result<()> {
    let enc_config = build_enc_config(&cfg);
    loop {
        let chunk = match rt.block_on(queue.pop()) {
            Some(c) => c,
            None => break,
        };
        match encode_chunk_to_packets(&cfg, &enc_config, chunk, &shared_frames_encoded, &progress_tx)?
        {
            ChunkOutcome::Encoded(c) => out.lock().unwrap().push(c),
            ChunkOutcome::RequeuedOnMismatch { chunk, diff } => {
                tracing::warn!(
                    rung_idx = cfg.rung_idx,
                    gpu_vendor = ?cfg.gpu_vendor,
                    diff = %diff,
                    "chunk worker: codec invariant mismatch — requeuing chunk and exiting"
                );
                let _ = queue.push_front(chunk);
                break;
            }
        }
    }
    Ok(())
}

enum ChunkOutcome {
    Encoded(ChunkPackets),
    RequeuedOnMismatch { chunk: SegmentChunk, diff: String },
}

fn encode_chunk_to_packets(
    cfg: &EncoderWorkerConfig,
    enc_config: &EncoderConfig,
    chunk: SegmentChunk,
    shared_frames_encoded: &std::sync::atomic::AtomicU64,
    progress_tx: &mpsc::Sender<u64>,
) -> Result<ChunkOutcome> {
    let mut encoder =
        encode::select_encoder(enc_config.clone(), None).context("creating encoder for chunk")?;
    let segment_idx = chunk.segment_idx;
    let mut packets: Vec<encode::EncodedPacket> = Vec::new();
    let mut pending: Vec<encode::EncodedPacket> = Vec::new();
    let mut decided = false;

    for frame in &chunk.frames {
        encoder.send_frame(frame).context("send_frame in chunk worker")?;
        while let Some(packet) = encoder.receive_packet().context("receive_packet in chunk worker")? {
            if !decided {
                match validate_or_set_rung_invariant(
                    cfg.rung_idx,
                    cfg.gpu_vendor,
                    &cfg.rung_invariant,
                    &packet.data,
                )? {
                    InvariantCheck::Matched | InvariantCheck::SetByThisWorker => decided = true,
                    InvariantCheck::Mismatched { diff } => {
                        return Ok(ChunkOutcome::RequeuedOnMismatch { chunk, diff });
                    }
                }
                pending.push(packet);
                continue;
            }
            packets.append(&mut pending);
            packets.push(packet);
        }
        let n = shared_frames_encoded.fetch_add(1, std::sync::atomic::Ordering::AcqRel) + 1;
        let _ = progress_tx.try_send(n);
    }
    if decided {
        packets.append(&mut pending);
    }
    encoder.flush().context("flush in chunk worker")?;
    while let Some(packet) = encoder
        .receive_packet()
        .context("receive_packet after flush in chunk worker")?
    {
        packets.push(packet);
    }
    Ok(ChunkOutcome::Encoded(ChunkPackets { segment_idx, packets }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_clone_preserves_fields() {
        let cfg = EncoderWorkerConfig {
            rung_idx: 2,
            width: 1280,
            height: 720,
            frame_rate: 30.0,
            quality: 32,
            speed_preset: u8::MAX,
            target: codec::encode::tuning::QualityTarget::Standard,
            tier: codec::encode::tuning::SpeedTier::Standard,
            threads: 4,
            gpu_index: Some(1),
            gpu_vendor: None,
            output_color_metadata: ColorMetadata::default(),
            output_pixel_format: PixelFormat::Yuv420p,
            constant_qp: false,
            timescale: 30000,
            per_frame_ticks: 1000,
            keyframe_interval: 60,
            segment_target_ticks: 60_000,
            output_dir: PathBuf::from("/tmp/x"),
            rung_invariant: Arc::new(RwLock::new(None)),
        };
        let copy = cfg.clone();
        assert_eq!(copy.rung_idx, 2);
        assert_eq!(copy.keyframe_interval, 60);
    }

    #[test]
    fn invariant_matches_itself() {
        let a = RungCodecInvariant {
            seq_profile: 0,
            seq_level_idx_0: 8,
            seq_tier_0: 0,
            bit_depth: 8,
            monochrome: false,
            chroma_subsampling_x: true,
            chroma_subsampling_y: true,
            color_primaries: 1,
            transfer_characteristics: 1,
            matrix_coefficients: 1,
            color_range: false,
            max_frame_width_minus1: 1919,
            max_frame_height_minus1: 1079,
            still_picture: false,
        };
        assert_eq!(a.clone(), a);
        assert_eq!(a.describe_diff(&a), "");
    }

    #[test]
    fn invariant_diff_lists_changed_fields() {
        let a = RungCodecInvariant {
            seq_profile: 0,
            seq_level_idx_0: 8,
            seq_tier_0: 0,
            bit_depth: 8,
            monochrome: false,
            chroma_subsampling_x: true,
            chroma_subsampling_y: true,
            color_primaries: 1,
            transfer_characteristics: 1,
            matrix_coefficients: 1,
            color_range: false,
            max_frame_width_minus1: 1919,
            max_frame_height_minus1: 1079,
            still_picture: false,
        };
        let mut b = a.clone();
        b.bit_depth = 10;
        b.color_primaries = 9;
        let diff = a.describe_diff(&b);
        assert!(diff.contains("bit_depth"));
        assert!(diff.contains("color_primaries"));
        assert!(!diff.contains("seq_profile"));
    }

    #[test]
    fn validator_parse_error_returns_err_not_mismatch() {
        // Junk bytes — no recognisable AV1 sequence header OBU.
        // Distinct from a mismatch: this is a malformed-bitstream
        // condition that nothing downstream can recover from. The
        // worker propagates this Err and fails the run, unlike the
        // soft-fail Mismatched case.
        let slot: RwLock<Option<RungCodecInvariant>> = RwLock::new(None);
        let junk = vec![0u8; 8];
        let err =
            validate_or_set_rung_invariant(0, Some(codec::gpu::GpuVendor::Intel), &slot, &junk)
                .unwrap_err();
        assert!(
            err.to_string()
                .contains("could not parse AV1 sequence header")
        );
        assert!(slot.read().unwrap().is_none());
    }

    #[test]
    fn mismatched_diff_includes_changed_field() {
        let existing = RungCodecInvariant {
            seq_profile: 0,
            seq_level_idx_0: 8,
            seq_tier_0: 0,
            bit_depth: 8,
            monochrome: false,
            chroma_subsampling_x: true,
            chroma_subsampling_y: true,
            color_primaries: 1,
            transfer_characteristics: 1,
            matrix_coefficients: 1,
            color_range: false,
            max_frame_width_minus1: 1919,
            max_frame_height_minus1: 1079,
            still_picture: false,
        };
        let mut other = existing.clone();
        other.bit_depth = 10;
        let diff = existing.describe_diff(&other);
        assert!(
            diff.contains("bit_depth"),
            "diff should mention bit_depth; got {diff}"
        );
    }
}
