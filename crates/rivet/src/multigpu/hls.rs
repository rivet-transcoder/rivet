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
            // Coverage is judged against the segments the scalers pushed — every
            // scaler on this rung has finished (the wait above), so the count is
            // final and exact — not against `total_segments`, which is only as
            // good as the source's frame count: an estimate (`duration * fps`)
            // for Matroska, and for a transport stream a count of its PES
            // packets. Judged against the estimate, a complete encode failed
            // when it ran over ("expected 2 segments, got 3" on an MKV whose
            // Duration understated it) and when it fell short.
            let pushed = ladder_h.queues[idx].pushed_segments();
            let result = match merge_rung_contributions(contribs) {
                Ok(merged) => {
                    let got = merged.manifest.segments.len();
                    let numbers: Vec<u32> =
                        merged.manifest.segments.iter().map(|s| s.sequence_number).collect();
                    if let Some(err) = segment_coverage_error(&rung.label, pushed, &numbers) {
                        Err(anyhow!(err))
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
                        // The frames the rung encoded, not the estimate it
                        // was planned from.
                        let frames = ladder_h.frames_encoded[idx].load(Ordering::Relaxed);
                        report(
                            sink.as_ref(),
                            idx,
                            &rung,
                            crate::progress::RungStatus::Completed,
                            frames,
                            Some(frames),
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
        video_delay_ticks: params.video_delay_ticks,
    };
    // A unit of HLS work: one chunk → one CMAF segment file.
    let encode: Arc<dyn EncodeUnit<WorkerOutput>> = Arc::new(
        |cfg: &crate::encoder_worker::EncoderWorkerConfig,
         chunk,
         init_written: &mut bool,
         _sessions: &mut crate::encoder_worker::EncoderSessionPool,
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
            // The pumps and scalers are already running in blocking threads.
            // Returning without stopping them left a scaler parked on a full
            // queue nobody would drain, the pump behind it, and a runtime that
            // could not shut down — the run sat at `0/N frames` forever. The
            // abort closes the queues, which unwinds both.
            ladder.abort.abort();
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

/// Check that a rung's segments are exactly the ones its scalers pushed: HLS
/// sequence numbers `1..=pushed`, each once. `None` when they are.
fn segment_coverage_error(label: &str, pushed: usize, numbers: &[u32]) -> Option<String> {
    let present: HashSet<u32> = numbers.iter().copied().collect();
    let expected = 1..=u32::try_from(pushed).unwrap_or(u32::MAX);
    if numbers.len() == pushed && expected.clone().all(|n| present.contains(&n)) {
        return None;
    }
    let missing: Vec<u32> = expected.filter(|n| !present.contains(n)).take(10).collect();
    Some(format!(
        "rung {label} coverage incomplete: the scalers pushed {pushed} segments, {} came back          (first 10 missing: {missing:?})",
        numbers.len()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_pool::GpuPool;
    use crate::progress::NullSink;
    use crate::spec::{EncodePolicy, GpuFamily, Rung};
    use codec::frame::VideoCodec;
    use std::time::Duration;

    /// The bug, on the HLS path: `--encode family:intel` on a host with no
    /// Intel card handed the ladder an empty pool, and the ladder started
    /// decoding before it found out. The input here is not a container, so
    /// had a pump started the error would say "decode"; the refusal has to
    /// come first, name the pin, and come back at once — the control build
    /// sat at `0/120 frames` until it was killed.
    #[test]
    fn an_empty_pool_is_refused_by_name_before_any_decode() {
        let verdict = super::super::test_support::within(
            Duration::from_secs(20),
            "the HLS ladder on an empty pool waited for a lease instead of refusing",
            || async {
                let rungs = vec![Rung::new(64, 64)];
                let params = super::super::test_support::params_with_pool(
                    &rungs,
                    Arc::new(GpuPool::new(&[])),
                    EncodePolicy::Family(GpuFamily::Intel),
                    VideoCodec::H264,
                );
                run_multigpu_hls(params, Arc::new(NullSink)).await.map(|_| ()).map_err(|e| format!("{e:#}"))
            },
        );
        let msg = verdict.expect_err("nothing to encode on");
        assert!(msg.contains("no encoder matches `--encode family:intel` for H.264 on this host"), "{msg}");
        assert!(msg.contains("Present: synth-0 (gpu 0, NVIDIA, encodes H.264)"), "the params' host, not this machine: {msg}");
        assert!(!msg.contains("decode"), "refused only after a decode had started: {msg}");
    }

    #[test]
    fn coverage_is_judged_against_what_the_scalers_pushed_not_the_estimate() {
        // An MKV whose Duration understates it plans 2 segments and pushes 3; a
        // frame count that overstates plans more than come. Either way the
        // segments that came back are the whole rung.
        assert_eq!(segment_coverage_error("360p", 3, &[1, 2, 3]), None);
        assert_eq!(segment_coverage_error("360p", 1, &[1]), None);
        assert_eq!(segment_coverage_error("360p", 0, &[]), None);
        // A segment lost to a dead worker is still caught, by number.
        let err = segment_coverage_error("360p", 3, &[1, 3]).expect("one missing");
        assert!(err.contains("pushed 3 segments, 2 came back"), "{err}");
        assert!(err.contains("[2]"), "{err}");
        // A duplicate does not stand in for a missing one.
        assert!(segment_coverage_error("360p", 3, &[1, 1, 3]).is_some());
        assert!(segment_coverage_error("360p", 2, &[1, 2, 3]).is_some());
    }
}
