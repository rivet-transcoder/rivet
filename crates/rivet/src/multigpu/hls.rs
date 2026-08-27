//! HLS on the ladder core: [`run_multigpu_hls`].
//!
//! The scheduling — range-split decode, per-rung scalers, ladder workers,
//! the finished rule — is [`super::ladder`]. What is HLS here: a chunk is one
//! CMAF segment (`keyframe_interval` frames, no lead-in margin, because every
//! segment is a real file a player fetches on its own), a worker turns it
//! into a segment file on disk, and a rung's finalizer merges its workers'
//! segment lists into one manifest and checks coverage.

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Result, anyhow, bail};
use container::cmaf::CmafTrackManifest;
use tokio::sync::mpsc;

use crate::cmaf_util::{RungContribution, merge_rung_contributions, total_segments_for_rung};
use crate::encoder_worker::{UnitOutcome as SegmentOutcome, WorkerOutput, encode_segment_unit};
use crate::progress::ProgressSink;

use super::ladder::{self, EncodeUnit, Ladder, LadderShape, Running, UnitOutcome};
use super::{MultiGpuParams, RungManifest, WorkerCtx, report, spawn_progress_reporter};

/// Run the multi-GPU HLS ladder. Returns one `Option<RungManifest>` per rung
/// (in rung order); `None` means the rung produced no segments.
pub async fn run_multigpu_hls(
    params: MultiGpuParams<'_>,
    sink: Arc<dyn ProgressSink>,
) -> Result<Vec<Option<RungManifest>>> {
    let rungs = params.rungs;
    let n = rungs.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    let total_segments = total_segments_for_rung(params.total_input_frames, params.keyframe_interval);
    if total_segments == 0 {
        bail!(
            "multigpu: total_segments == 0 (total_input_frames={}, keyframe_interval={})",
            params.total_input_frames,
            params.keyframe_interval
        );
    }

    // Pre-flight: verify this host can actually construct an encoder for the
    // requested codec before spawning the orchestration. Fail fast with a clear
    // error instead of dispatching workers that fail at encoder construction —
    // and, on drivers that re-init a failed NVENC session badly (e.g. Ampere
    // with no AV1-encode silicon), would otherwise hang an uncancellable task.
    ladder::preflight_encoder(&params, rungs[0].width, rungs[0].height)?;

    let capacity = params.gpu_pool.capacity().max(1);
    tracing::info!(
        rungs = n,
        total_segments,
        gpu_pool_capacity = params.gpu_pool.capacity(),
        software_pool = params.gpu_pool.is_software(),
        threads_per_slot = ?params.gpu_pool.software_threads(),
        decode = ?params.decode,
        encode = ?params.encode,
        "multi-GPU ladder starting"
    );

    // HLS segments are real files that must each stand alone; there's no
    // stitch to hide a margin in, so no overlap here.
    let shape = LadderShape { frames_per_chunk: params.keyframe_interval, overlap: 0 };
    let ladder: Arc<Ladder<WorkerOutput>> = Arc::new(Ladder::new(rungs, shape.frames_per_chunk));

    // Periodic progress reporter.
    let progress_stop = Arc::new(AtomicBool::new(false));
    let progress_handle = spawn_progress_reporter(
        rungs.to_vec(),
        ladder.frames_encoded.clone(),
        // HLS writes CMAF segments straight to disk, so there's no in-memory
        // packet tally to report mid-run; size lands at finalize. Zero means
        // "unknown" to the CLI, which omits the field rather than printing 0 B.
        ladder.bytes_encoded.clone(),
        Arc::clone(&ladder.finalized),
        params.total_input_frames,
        Arc::clone(&sink),
        Arc::clone(&progress_stop),
    );

    // Finalizers: one per rung, merges contributions → RungManifest ---------
    let total_input_frames = params.total_input_frames;
    let (finalizer_tx, finalizer_rx) = mpsc::channel::<(usize, Result<Option<RungManifest>>)>(n.max(1));
    let mut finalizer_handles = Vec::with_capacity(n);
    for idx in 0..n {
        let ladder_h = Arc::clone(&ladder);
        let tx = finalizer_tx.clone();
        let rung = rungs[idx].clone();
        let rel_dir = format!("video/{}", rung.label);
        let output_root = params.output_root.clone();
        let timescale = params.timescale;
        let sink = Arc::clone(&sink);
        finalizer_handles.push(tokio::spawn(async move {
            ladder_h.wait_rung_finished(idx).await;
            if ladder_h.is_aborted() {
                // The run was stopped under us; whatever this rung has is not a
                // rung, and nobody is reading the channel any more.
                ladder_h.finalized[idx].store(true, Ordering::Release);
                let _ = tx.send((idx, Err(anyhow!("run aborted")))).await;
                return;
            }
            let outputs: Vec<WorkerOutput> = ladder_h.take_contributions(idx);
            if outputs.is_empty() {
                ladder_h.finalized[idx].store(true, Ordering::Release);
                let _ = tx.send((idx, Ok(None))).await;
                return;
            }
            let init_path = output_root.join(&rel_dir).join("init.mp4");
            let contribs: Vec<RungContribution> = outputs
                .into_iter()
                .map(|wo| RungContribution {
                    width: rung.width,
                    height: rung.height,
                    relative_dir: rel_dir.clone(),
                    manifest: CmafTrackManifest {
                        init_path: init_path.clone(),
                        segments: wo.segments,
                        timescale,
                    },
                })
                .collect();
            let result = match merge_rung_contributions(contribs) {
                Ok(merged) => {
                    let got = merged.manifest.segments.len();
                    if got != total_segments as usize {
                        let present: HashSet<u32> =
                            merged.manifest.segments.iter().map(|s| s.sequence_number).collect();
                        let missing: Vec<u32> =
                            (1..=total_segments).filter(|s| !present.contains(s)).collect();
                        Err(anyhow!(
                            "rung {} coverage incomplete: expected {} segments, got {} \
                             (first 10 missing: {:?})",
                            rung.label,
                            total_segments,
                            got,
                            missing.iter().take(10).collect::<Vec<_>>(),
                        ))
                    } else {
                        let bytes: u64 = merged.manifest.segments.iter().map(|s| s.byte_size).sum();
                        let rung_manifest = RungManifest {
                            rung_index: idx,
                            width: rung.width,
                            height: rung.height,
                            label: rung.label.clone(),
                            relative_dir: rel_dir.clone(),
                            manifest: merged.manifest,
                        };
                        // The manifest first, then the status: a consumer that
                        // ships a rung on `on_rung_complete` and announces it on
                        // `Completed` sees them in the order it would want.
                        sink.on_rung_complete(&rung_manifest);
                        report(
                            sink.as_ref(),
                            idx,
                            &rung,
                            crate::progress::RungStatus::Completed,
                            total_input_frames,
                            Some(total_input_frames),
                            got as u32,
                            bytes,
                            None,
                        );
                        Ok(Some(rung_manifest))
                    }
                }
                Err(e) => Err(anyhow!("merging contributions for rung {}: {e}", rung.label)),
            };
            ladder_h.finalized[idx].store(true, Ordering::Release);
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
    // A unit of HLS work: one chunk → one CMAF segment file.
    let encode: Arc<dyn EncodeUnit<WorkerOutput>> = Arc::new(
        |cfg: &crate::encoder_worker::EncoderWorkerConfig,
         chunk,
         init_written: &mut bool,
         frames: &std::sync::atomic::AtomicU64,
         _bytes: &std::sync::atomic::AtomicU64,
         tx: &mpsc::Sender<u64>| {
            Ok(match encode_segment_unit(cfg, chunk, init_written, frames, tx)? {
                SegmentOutcome::Wrote(info) => {
                    UnitOutcome::Done(WorkerOutput { gpu_index: cfg.gpu_index, segments: vec![info] })
                }
                SegmentOutcome::Rejected { chunk, diff } => UnitOutcome::Rejected { chunk, diff },
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

    let result = ladder::drain(Running {
        pumps,
        scalers,
        workers,
        finalizer_rx,
        finalizers_remaining: n,
        abort: Arc::clone(&ladder.abort),
        cancel: params.cancel.clone(),
    })
    .await;

    progress_stop.store(true, Ordering::Release);
    let _ = progress_handle.await;
    let completed = result?;
    for h in finalizer_handles {
        let _ = h.await;
    }
    Ok(completed)
}
