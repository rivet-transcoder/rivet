//! Single-file (chunk-and-stitch) on the ladder core: [`run_multigpu_single_file`].
//!
//! The scheduling — range-split decode, per-rung scalers, ladder workers,
//! the finished rule — is [`super::ladder`]. What is single-file here: a
//! chunk is several GOPs with a one-GOP lead-in margin, a worker encodes it to
//! packets in memory (a fresh encoder per chunk → its first kept frame is an
//! IDR, so chunks encoded out of order on different cards concatenate), and a
//! rung's finalizer stitches its chunks, in order, into one packet stream the
//! caller muxes into an MP4 — no disk round-trip.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Result, anyhow, bail};
use tokio::sync::mpsc;

use codec::encode::EncodedPacket;
use codec::frame::VideoCodec;

use crate::cmaf_util::total_segments_for_rung;
use crate::encoder_worker::{ChunkPackets, ChunkUnitOutcome, encode_chunk_unit};
use crate::progress::ProgressSink;

use super::ladder::{self, EncodeUnit, Ladder, LadderShape, Running, UnitOutcome};
use super::{MultiGpuParams, WorkerCtx, report, spawn_progress_reporter};

/// Check that the collected chunk indices cover `expected` chunks contiguously
/// from zero, returning the operator-facing message when they don't.
///
/// `expected` must be the number of chunks the scalers actually **pushed**, not
/// `ceil(total_input_frames / frames_per_chunk)`. The latter is derived from a
/// frame count that is only an estimate (`duration * fps`) for any container
/// without an explicit one — Matroska never has one — and an estimate landing
/// one frame into the next bucket used to fail the job *after* a complete
/// encode with "expected 1324 contiguous chunks, got 1323".
///
/// `indices` must already be sorted and deduplicated.
fn coverage_error(label: &str, expected: usize, indices: &[usize]) -> Option<String> {
    let got = indices.len();
    let contiguous = indices.iter().enumerate().all(|(i, idx)| *idx == i);
    if got == expected && contiguous {
        return None;
    }
    Some(format!(
        "rung {label} chunk coverage incomplete: the scaler pushed {expected} chunks, \
         {got} came back{}",
        if contiguous { "" } else { " (and they aren't contiguous from 0)" }
    ))
}

/// How many GOPs make up one scheduling chunk on the single-file path.
///
/// Must be >= 1. Raising it makes chunk seams rarer (a chunk seam is more
/// visible than the ordinary IDRs at GOP boundaries) at the cost of coarser
/// load balancing; at 1 every GOP boundary is a seam, which is where this
/// started.
///
/// Chunk length is NOT the GOP length. They were the same number, which meant
/// every GOP boundary was also a chunk boundary — and a chunk boundary is far
/// more visible than an IDR: measured on 1080p content, a chunk seam shows
/// 2.27x the inter-frame discontinuity of the source where a plain IDR (single
/// encoder, or ffmpeg at the same GOP) shows 1.21x. At a 2 s GOP that put a
/// visible stutter every 2 seconds for the whole film. The GOP is a
/// decode/seek property and stays where it is; chunk length, for single-file
/// output, is only a load-balancing parameter — MP4 has no segmentation
/// requirement — so a chunk is a whole number of GOPs and the seam artifact
/// happens once per chunk. 10 GOPs ≈ 20 s; on a 44-minute source that is still
/// ~130 chunks to spread across the GPUs.
const GOPS_PER_CHUNK: u32 = 10;

/// How the chunk lead-in margin is made safe.
///
/// A margin is only correct if the first *kept* frame is a random-access point,
/// or the chunk can't stand alone and the stitch produces a stream whose
/// references point at frames that were discarded.
///
/// Asking for one per frame does not work here: on iHD's VDENC path
/// `mfxEncodeCtrl.FrameType = I|IDR|REF` is ignored. So the margin is sized to
/// **exactly one GOP** instead. The encoder places IDRs every `GopPicSize`
/// frames from its own frame 0, so a one-GOP margin puts the first kept frame
/// on a GOP boundary, where it gets an IDR by the encoder's own cadence — and
/// every later IDR in the chunk lands on the global cadence too. No per-frame
/// control needed, and it holds on any backend.
///
/// `Encoder::force_keyframe_next` is still called at the boundary. It's
/// belt-and-braces for backends that do honour it; correctness doesn't depend
/// on it.
///
/// Changes here must be checked against three gates, all of which caught a
/// broken attempt at this: container sample count == decoded frame count,
/// zero decoder errors, and an unbroken IDR cadence. Quality metrics do not
/// work — when the promotion silently failed, mean PSNR went *up*, because
/// ffmpeg conceals the missing references.
fn lead_in_margin(keyframe_interval: u32) -> usize {
    keyframe_interval as usize
}

/// One rung's full ordered packet stream, stitched from chunks encoded across
/// GPUs. The caller muxes these into a single MP4 (+ audio).
#[derive(Debug)]
pub struct RungPackets {
    pub rung_index: usize,
    pub codec: VideoCodec,
    pub width: u32,
    pub height: u32,
    pub label: String,
    pub packets: Vec<EncodedPacket>,
}

/// Single-file counterpart to [`run_multigpu_hls`](super::run_multigpu_hls):
/// decode once (split across the cards where the source allows), fan to
/// per-rung scalers, and let every GPU take the next GOP-sized chunk of
/// whichever rung is furthest behind. Each worker encodes its chunk to packets
/// (a fresh encoder per chunk → first kept frame is an IDR); the finalizer
/// concatenates them in chunk order into one ordered packet stream per rung.
pub async fn run_multigpu_single_file(
    params: MultiGpuParams<'_>,
    sink: Arc<dyn ProgressSink>,
) -> Result<Vec<Option<RungPackets>>> {
    let rungs = params.rungs;
    let n = rungs.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    let shape = LadderShape {
        frames_per_chunk: params.keyframe_interval.saturating_mul(GOPS_PER_CHUNK).max(1),
        overlap: lead_in_margin(params.keyframe_interval),
    };
    let total_segments = total_segments_for_rung(params.total_input_frames, shape.frames_per_chunk);
    if total_segments == 0 {
        bail!(
            "multigpu single-file: total_segments == 0 (frames={}, frames_per_chunk={})",
            params.total_input_frames,
            shape.frames_per_chunk
        );
    }

    // Pre-flight encoder probe (same fail-fast as the HLS path).
    {
        let probe = codec::encode::EncoderConfig {
            width: rungs[0].width,
            height: rungs[0].height,
            frame_rate: params.frame_rate,
            gpu_index: None,
            codec: params.codec,
            ..Default::default()
        };
        codec::encode::select_encoder(probe, None).map_err(|e| {
            anyhow!(
                "no {:?} encoder available on this host ({e}); need NVENC / AMF / QSV, or build \
                 with `rav1e-fallback` for software AV1",
                params.codec
            )
        })?;
    }

    let capacity = params.gpu_pool.capacity().max(1);
    tracing::info!(
        rungs = n,
        total_segments,
        gpu_pool_capacity = params.gpu_pool.capacity(),
        decode = ?params.decode,
        encode = ?params.encode,
        "multi-GPU single-file phase starting"
    );

    let ladder: Arc<Ladder<ChunkPackets>> = Arc::new(Ladder::new(rungs, shape.frames_per_chunk));

    let progress_stop = Arc::new(AtomicBool::new(false));
    let progress_handle = spawn_progress_reporter(
        rungs.to_vec(),
        ladder.frames_encoded.clone(),
        ladder.bytes_encoded.clone(),
        Arc::clone(&ladder.finalized),
        params.total_input_frames,
        Arc::clone(&sink),
        Arc::clone(&progress_stop),
    );

    // Finalizers: stitch each rung's chunks (sorted, deduped) into one stream.
    let total_input_frames = params.total_input_frames;
    let codec = params.codec;
    let (finalizer_tx, finalizer_rx) = mpsc::channel::<(usize, Result<Option<RungPackets>>)>(n.max(1));
    let mut finalizer_handles = Vec::with_capacity(n);
    for idx in 0..n {
        let ladder_h = Arc::clone(&ladder);
        let tx = finalizer_tx.clone();
        let rung = rungs[idx].clone();
        let sink = Arc::clone(&sink);
        finalizer_handles.push(tokio::spawn(async move {
            ladder_h.wait_rung_finished(idx).await;
            let mut chunks: Vec<ChunkPackets> = ladder_h.take_contributions(idx);
            if chunks.is_empty() {
                ladder_h.finalized[idx].store(true, Ordering::Release);
                let _ = tx.send((idx, Ok(None))).await;
                return;
            }
            chunks.sort_by_key(|c| c.segment_idx);
            chunks.dedup_by_key(|c| c.segment_idx);
            // Coverage: contiguous 0..pushed. Every scaler on this rung has
            // finished by now (the wait above), so `pushed_segments` is final
            // and exact — it still catches a chunk lost to a dead worker,
            // without inheriting the frame-count estimate's error.
            let expected = ladder_h.queues[idx].pushed_segments();
            let indices: Vec<usize> = chunks.iter().map(|c| c.segment_idx).collect();
            // Before the terminal report, not after: the reporter tests this
            // flag and then reports, so finalizing in between let a `Running`
            // tick print after `Completed`.
            ladder_h.finalized[idx].store(true, Ordering::Release);
            let result = if let Some(err) = coverage_error(&rung.label, expected, &indices) {
                Err(anyhow!(err))
            } else {
                let mut packets: Vec<EncodedPacket> = Vec::new();
                for c in chunks {
                    packets.extend(c.packets);
                }
                let bytes: u64 = packets.iter().map(|p| p.data.len() as u64).sum();
                report(
                    sink.as_ref(),
                    idx,
                    &rung,
                    crate::progress::RungStatus::Completed,
                    total_input_frames,
                    Some(total_input_frames),
                    indices.len() as u32,
                    bytes,
                    None,
                );
                Ok(Some(RungPackets {
                    rung_index: idx,
                    codec,
                    width: rung.width,
                    height: rung.height,
                    label: rung.label.clone(),
                    packets,
                }))
            };
            let _ = tx.send((idx, result)).await;
        }));
    }
    drop(finalizer_tx);

    // Decode, scale, encode ------------------------------------------------
    let ranges = ladder::plan_ranges(&params, shape, capacity);
    let (pumps, receivers) = ladder::spawn_pumps(&params, &ranges, n);
    let scalers = ladder::spawn_scalers(rungs, &ranges, shape, receivers, &ladder);

    let ctx = WorkerCtx {
        codec: params.codec,
        frame_rate: params.frame_rate,
        output_color_metadata: params.output_color_metadata,
        output_pixel_format: params.output_pixel_format,
        timescale: params.timescale,
        per_frame_ticks: params.per_frame_ticks,
        keyframe_interval: params.keyframe_interval,
        segment_target_ticks: params.segment_target_ticks,
        output_root: params.output_root.clone(),
        constant_qp: params.constant_qp,
    };
    // A unit of single-file work: one chunk → its packets, in memory.
    let encode: Arc<dyn EncodeUnit<ChunkPackets>> = Arc::new(
        |cfg: &crate::encoder_worker::EncoderWorkerConfig,
         chunk,
         _init_written: &mut bool,
         frames: &std::sync::atomic::AtomicU64,
         bytes: &std::sync::atomic::AtomicU64,
         tx: &mpsc::Sender<u64>| {
            Ok(match encode_chunk_unit(cfg, chunk, frames, bytes, tx)? {
                ChunkUnitOutcome::Encoded(packets) => UnitOutcome::Done(packets),
                ChunkUnitOutcome::Rejected { chunk, diff } => UnitOutcome::Rejected { chunk, diff },
            })
        },
    );
    let (workers, _) = match ladder::spawn_workers(&params, &ctx, rungs, &ladder, encode).await {
        Ok(w) => w,
        Err(e) => {
            progress_stop.store(true, Ordering::Release);
            let _ = progress_handle.await;
            return Err(e);
        }
    };
    ladder.release_setup_guard();

    let result = ladder::drain(Running { pumps, scalers, workers, finalizer_rx, finalizers_remaining: n }).await;

    progress_stop.store(true, Ordering::Release);
    let _ = progress_handle.await;
    let completed = result?;
    for h in finalizer_handles {
        let _ = h.await;
    }
    Ok(completed)
}

#[cfg(test)]
mod tests {
    use super::coverage_error;

    #[test]
    fn full_contiguous_coverage_is_accepted() {
        assert_eq!(coverage_error("1080p", 4, &[0, 1, 2, 3]), None);
        // The degenerate single-chunk and empty cases are still coverage.
        assert_eq!(coverage_error("1080p", 1, &[0]), None);
        assert_eq!(coverage_error("1080p", 0, &[]), None);
    }

    #[test]
    fn a_missing_tail_chunk_is_caught() {
        // The scaler pushed 4, only 3 came back — a worker died holding one.
        let err = coverage_error("1080p", 4, &[0, 1, 2]).expect("should fail");
        assert!(err.contains("pushed 4"), "{err}");
        assert!(err.contains("3 came back"), "{err}");
    }

    #[test]
    fn a_hole_in_the_middle_is_caught_and_named() {
        let err = coverage_error("1080p", 4, &[0, 1, 3]).expect("should fail");
        assert!(err.contains("contiguous"), "a gap should say so: {err}");
    }

    #[test]
    fn coverage_is_judged_against_what_was_pushed_not_an_estimate() {
        // The regression: an estimated frame count one bucket high asked for
        // 1324 chunks when the scaler only ever produced 1323. Judged against
        // the pushed count, a complete encode passes.
        let indices: Vec<usize> = (0..1323).collect();
        assert_eq!(coverage_error("1080p", 1323, &indices), None);
        // And a genuinely lost chunk still fails, at the same scale.
        let short: Vec<usize> = (0..1322).collect();
        assert!(coverage_error("1080p", 1323, &short).is_some());
    }
}
