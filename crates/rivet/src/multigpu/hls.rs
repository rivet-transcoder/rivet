//! HLS multi-GPU orchestration: [`run_multigpu_hls`] + per-rung encoder worker
//! spawn helper.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

use anyhow::{Result, anyhow, bail};
use container::cmaf::CmafTrackManifest;
use tokio::sync::{Notify, mpsc};
use tokio::task::JoinSet;

use crate::cmaf_util::{RungContribution, merge_rung_contributions, total_segments_for_rung};
use crate::encoder_worker::{
    EncoderWorkerConfig, RungCodecInvariant, WorkerOutput, run_encoder_worker_blocking,
};
use crate::frame_queue::SegmentChunkQueue;
use crate::gpu_pool::GpuLease;
use crate::progress::ProgressSink;
use crate::spec::Rung;

use super::{
    FANOUT_CHANNEL_CAPACITY, HELPER_POLL_INTERVAL, QUEUE_CAPACITY, MultiGpuParams, RungManifest,
    WorkerCtx, report, spawn_progress_reporter,
};

/// Run the reactive multi-GPU variant phase. Returns one `Option<RungManifest>`
/// per rung (in rung order); `None` means the rung produced no segments.
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
                 with `rav1e-fallback` for a software AV1 encoder",
                params.codec
            )
        })?;
    }

    tracing::info!(
        rungs = n,
        total_segments,
        gpu_pool_capacity = params.gpu_pool.capacity(),
        "multi-GPU variant phase starting"
    );

    // Per-rung shared state.
    let queues: Vec<Arc<SegmentChunkQueue>> =
        (0..n).map(|_| Arc::new(SegmentChunkQueue::new(QUEUE_CAPACITY))).collect();
    let frames_encoded: Vec<Arc<AtomicU64>> = (0..n).map(|_| Arc::new(AtomicU64::new(0))).collect();
    let scaler_active: Vec<Arc<AtomicBool>> =
        (0..n).map(|_| Arc::new(AtomicBool::new(false))).collect();
    let rung_invariants: Vec<Arc<std::sync::RwLock<Option<RungCodecInvariant>>>> =
        (0..n).map(|_| Arc::new(std::sync::RwLock::new(None))).collect();
    let contributions: Arc<Vec<std::sync::Mutex<Vec<WorkerOutput>>>> =
        Arc::new((0..n).map(|_| std::sync::Mutex::new(Vec::new())).collect());
    let active_workers: Arc<Vec<AtomicUsize>> =
        Arc::new((0..n).map(|_| AtomicUsize::new(0)).collect());
    let rung_done: Arc<Vec<Notify>> = Arc::new((0..n).map(|_| Notify::new()).collect());
    let finalized: Arc<Vec<AtomicBool>> =
        Arc::new((0..n).map(|_| AtomicBool::new(false)).collect());

    // Periodic progress reporter.
    let progress_stop = Arc::new(AtomicBool::new(false));
    let progress_handle = spawn_progress_reporter(
        rungs.to_vec(),
        frames_encoded.clone(),
        // HLS writes CMAF segments straight to disk, so there's no in-memory
        // packet tally to report mid-run; size lands at finalize. Zero means
        // "unknown" to the CLI, which omits the field rather than printing 0 B.
        (0..n).map(|_| Arc::new(AtomicU64::new(0))).collect(),
        finalized.clone(),
        params.total_input_frames,
        Arc::clone(&sink),
        Arc::clone(&progress_stop),
    );

    // Finalizers: one per rung, merges contributions → RungManifest.
    let total_input_frames = params.total_input_frames;
    let (finalizer_tx, mut finalizer_rx) =
        mpsc::channel::<(usize, Result<Option<RungManifest>>)>(n.max(1));
    let mut finalizer_handles = Vec::with_capacity(n);
    for idx in 0..n {
        let contributions_h = Arc::clone(&contributions);
        let active_h = Arc::clone(&active_workers);
        let rung_done_h = Arc::clone(&rung_done);
        let finalized_h = Arc::clone(&finalized);
        let tx = finalizer_tx.clone();
        let rung = rungs[idx].clone();
        let rel_dir = format!("video/{}", rung.label);
        let output_root = params.output_root.clone();
        let timescale = params.timescale;
        let total_segments = total_segments;
        let sink = Arc::clone(&sink);
        finalizer_handles.push(tokio::spawn(async move {
            // Wait for all of this rung's workers + the scaler to finish.
            loop {
                let notified = rung_done_h[idx].notified();
                if active_h[idx].load(Ordering::Acquire) == 0 {
                    break;
                }
                notified.await;
            }
            let outputs: Vec<WorkerOutput> = std::mem::take(&mut *contributions_h[idx].lock().unwrap());
            if outputs.is_empty() {
                finalized_h[idx].store(true, Ordering::Release);
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
                        Err(anyhow!(
                            "rung {} coverage incomplete: expected {} segments, got {}",
                            rung.label,
                            total_segments,
                            got
                        ))
                    } else {
                        let bytes: u64 = merged.manifest.segments.iter().map(|s| s.byte_size).sum();
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
                        Ok(Some(RungManifest {
                            rung_index: idx,
                            width: rung.width,
                            height: rung.height,
                            label: rung.label.clone(),
                            relative_dir: rel_dir.clone(),
                            manifest: merged.manifest,
                        }))
                    }
                }
                Err(e) => Err(anyhow!("merging contributions for rung {}: {e}", rung.label)),
            };
            finalized_h[idx].store(true, Ordering::Release);
            let _ = tx.send((idx, result)).await;
        }));
    }
    drop(finalizer_tx);

    // Smallest-first claim order for initial workers.
    let mut indexed: Vec<(usize, Rung)> = rungs.iter().cloned().enumerate().collect();
    indexed.sort_by_key(|(_, r)| r.short_side());

    // Decode pump(s) + fan-out channels.
    let mut frame_senders = Vec::with_capacity(n);
    let mut frame_receivers: Vec<Option<tokio::sync::mpsc::Receiver<codec::frame::VideoFrame>>> =
        Vec::with_capacity(n);
    for _ in 0..n {
        let (tx, rx) = tokio::sync::mpsc::channel(FANOUT_CHANNEL_CAPACITY);
        frame_senders.push(tx);
        frame_receivers.push(Some(rx));
    }

    let use_shared_pump = n <= params.gpu_pool.capacity();
    let mut pump_tasks: JoinSet<Result<u64>> = JoinSet::new();
    if use_shared_pump {
        let clips = params.clip_sources_for(params.decode_gpu_for(0));
        let senders = frame_senders;
        let rt = tokio::runtime::Handle::current();
        pump_tasks.spawn(async move {
            tokio::task::spawn_blocking(move || {
                crate::decode_pump::run_spliced_decode_pump_blocking(clips, senders, rt)
            })
            .await
            .map_err(|e| anyhow!("shared pump join error: {e}"))
            .and_then(|r| r)
        });
    } else {
        for (idx, sender) in frame_senders.into_iter().enumerate() {
            let clips = params.clip_sources_for(params.decode_gpu_for(idx));
            let rt = tokio::runtime::Handle::current();
            pump_tasks.spawn(async move {
                tokio::task::spawn_blocking(move || {
                    crate::decode_pump::run_spliced_decode_pump_blocking(clips, vec![sender], rt)
                })
                .await
                .map_err(|e| anyhow!("per-rung pump {idx} join error: {e}"))
                .and_then(|r| r)
            });
        }
    }

    // Per-rung scalers.
    let mut scaler_tasks: JoinSet<(usize, Result<usize>)> = JoinSet::new();
    for (idx, rung) in rungs.iter().cloned().enumerate() {
        let rx = frame_receivers[idx].take().expect("scaler rx slot");
        let cfg = crate::rung_scaler::RungScalerConfig {
            rung_idx: idx,
            target_width: rung.width,
            target_height: rung.height,
            frames_per_chunk: params.keyframe_interval,
            // HLS segments are real files that must each stand alone; there's
            // no stitch to hide a margin in, so no overlap here.
            overlap: 0,
            // Whole-source pump: numbering starts at 0 and this scaler
            // owns the end of the stream.
            first_segment_idx: 0,
            is_final_range: true,
        };
        let queue = Arc::clone(&queues[idx]);
        let rt = tokio::runtime::Handle::current();
        let scaler_flag = Arc::clone(&scaler_active[idx]);
        let active_h = Arc::clone(&active_workers);
        let rung_done_h = Arc::clone(&rung_done);
        scaler_flag.store(true, Ordering::Release);
        active_h[idx].fetch_add(1, Ordering::AcqRel);
        scaler_tasks.spawn(async move {
            let result = tokio::task::spawn_blocking(move || {
                crate::rung_scaler::run_rung_scaler_blocking(cfg, rx, queue, rt)
            })
            .await
            .map_err(|e| anyhow!("scaler join error: {e}"))
            .and_then(|r| r);
            scaler_flag.store(false, Ordering::Release);
            let prev = active_h[idx].fetch_sub(1, Ordering::AcqRel);
            if prev == 1 {
                rung_done_h[idx].notify_one();
            }
            (idx, result)
        });
    }

    // Initial encoder workers (one per rung, smallest first).
    let mut worker_tasks: JoinSet<(usize, Result<()>)> = JoinSet::new();
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
    for (idx, rung) in indexed.iter().cloned() {
        let lease = match Arc::clone(&params.gpu_pool).claim().await {
            Some(l) => l,
            None => {
                progress_stop.store(true, Ordering::Release);
                let _ = progress_handle.await;
                bail!("multigpu: GPU pool returned no lease on a CPU-only host; at least one GPU is required");
            }
        };
        spawn_encoder_worker(
            &ctx,
            idx,
            &rung,
            Arc::clone(&queues[idx]),
            Arc::clone(&frames_encoded[idx]),
            lease,
            Arc::clone(&contributions),
            Arc::clone(&active_workers),
            Arc::clone(&rung_done),
            Arc::clone(&rung_invariants[idx]),
            Some(&mut worker_tasks),
        );
    }

    // Helper dispatcher.
    let helper_cancel = Arc::new(AtomicBool::new(false));
    let helper_handle = {
        let cancel = Arc::clone(&helper_cancel);
        let pool = Arc::clone(&params.gpu_pool);
        let queues = queues.clone();
        let scaler_active = scaler_active.clone();
        let frames_encoded = frames_encoded.clone();
        let contributions = Arc::clone(&contributions);
        let active_workers = Arc::clone(&active_workers);
        let rung_done = Arc::clone(&rung_done);
        let rung_invariants = rung_invariants.clone();
        let rungs_owned: Vec<Rung> = rungs.to_vec();
        let ctx = ctx.clone();
        tokio::spawn(async move {
            loop {
                if cancel.load(Ordering::Acquire) {
                    break;
                }
                tokio::time::sleep(HELPER_POLL_INTERVAL).await;
                if pool.pending_claimers() > 0 {
                    continue;
                }
                let mut target = None;
                for (idx, q) in queues.iter().enumerate() {
                    let scaler_alive = scaler_active[idx].load(Ordering::Acquire);
                    let has_pending = q.pushed_segments() > q.popped_segments();
                    if scaler_alive || has_pending {
                        target = Some(idx);
                        break;
                    }
                }
                let Some(rung_idx) = target else { break };
                let lease = match pool.try_claim() {
                    Some(l) => l,
                    None => continue,
                };
                tracing::info!(rung_idx, gpu_index = lease.gpu_index, "multigpu helper dispatch");
                spawn_encoder_worker(
                    &ctx,
                    rung_idx,
                    &rungs_owned[rung_idx],
                    Arc::clone(&queues[rung_idx]),
                    Arc::clone(&frames_encoded[rung_idx]),
                    lease,
                    Arc::clone(&contributions),
                    Arc::clone(&active_workers),
                    Arc::clone(&rung_done),
                    Arc::clone(&rung_invariants[rung_idx]),
                    None,
                );
            }
        })
    };

    // Drain everything.
    let mut completed: Vec<Option<RungManifest>> = (0..n).map(|_| None).collect();
    let mut pumps_remaining = pump_tasks.len();
    let mut scalers_remaining = n;
    let mut workers_remaining = n;
    let mut finalizers_remaining = n;

    macro_rules! teardown_err {
        ($e:expr) => {{
            helper_cancel.store(true, Ordering::Release);
            let _ = helper_handle.await;
            progress_stop.store(true, Ordering::Release);
            let _ = progress_handle.await;
            return Err($e);
        }};
    }

    while pumps_remaining > 0 || scalers_remaining > 0 || workers_remaining > 0 || finalizers_remaining > 0 {
        tokio::select! {
            biased;
            p = pump_tasks.join_next(), if pumps_remaining > 0 => match p {
                Some(Ok(Ok(n))) => { tracing::info!(frames = n, "decode pump finished"); pumps_remaining -= 1; }
                Some(Ok(Err(e))) => teardown_err!(anyhow!("decode pump failed: {e}")),
                Some(Err(je)) => teardown_err!(anyhow!("pump join error: {je}")),
                None => pumps_remaining = 0,
            },
            s = scaler_tasks.join_next(), if scalers_remaining > 0 => match s {
                Some(Ok((idx, Ok(n)))) => { tracing::info!(idx, chunks = n, "scaler finished"); scalers_remaining -= 1; }
                Some(Ok((idx, Err(e)))) => teardown_err!(anyhow!("scaler {idx} failed: {e}")),
                Some(Err(je)) => teardown_err!(anyhow!("scaler join error: {je}")),
                None => scalers_remaining = 0,
            },
            w = worker_tasks.join_next(), if workers_remaining > 0 => match w {
                Some(Ok((idx, Ok(())))) => { tracing::info!(idx, "initial worker finished"); workers_remaining -= 1; }
                Some(Ok((idx, Err(e)))) => teardown_err!(anyhow!("worker for rung {idx} failed: {e}")),
                Some(Err(je)) => teardown_err!(anyhow!("worker join error: {je}")),
                None => workers_remaining = 0,
            },
            f = finalizer_rx.recv(), if finalizers_remaining > 0 => match f {
                Some((idx, Ok(opt))) => { completed[idx] = opt; finalizers_remaining -= 1; }
                Some((idx, Err(e))) => teardown_err!(anyhow!("finalizer for rung {idx} failed: {e}")),
                None => finalizers_remaining = 0,
            },
        }
    }

    helper_cancel.store(true, Ordering::Release);
    let _ = helper_handle.await;
    progress_stop.store(true, Ordering::Release);
    let _ = progress_handle.await;
    for h in finalizer_handles {
        let _ = h.await;
    }

    Ok(completed)
}

#[allow(clippy::too_many_arguments)]
fn spawn_encoder_worker(
    ctx: &WorkerCtx,
    rung_idx: usize,
    rung: &Rung,
    queue: Arc<SegmentChunkQueue>,
    frames_encoded: Arc<AtomicU64>,
    lease: GpuLease,
    contributions: Arc<Vec<std::sync::Mutex<Vec<WorkerOutput>>>>,
    active_workers: Arc<Vec<AtomicUsize>>,
    rung_done: Arc<Vec<Notify>>,
    rung_invariant: Arc<std::sync::RwLock<Option<RungCodecInvariant>>>,
    worker_tasks: Option<&mut JoinSet<(usize, Result<()>)>>,
) {
    let rel_dir = format!("video/{}", rung.label);
    let output_dir = ctx.output_root.join(&rel_dir);
    let gpu_index = lease.gpu_index;
    let gpu_vendor = lease.vendor;

    let cfg = EncoderWorkerConfig {
        overrides: Default::default(),
        rung_idx,
        codec: ctx.codec,
        width: rung.width,
        height: rung.height,
        frame_rate: ctx.frame_rate,
        quality: rung.quality.crf.unwrap_or(codec::encode::AUTO_FROM_TARGET),
        speed_preset: rung.quality.speed_preset.unwrap_or(codec::encode::AUTO_FROM_TARGET),
        target: rung.quality.target,
        tier: rung.quality.tier,
        threads: 0,
        gpu_index: Some(gpu_index),
        gpu_vendor: Some(gpu_vendor),
        output_color_metadata: ctx.output_color_metadata,
        output_pixel_format: ctx.output_pixel_format,
        constant_qp: ctx.constant_qp,
        timescale: ctx.timescale,
        per_frame_ticks: ctx.per_frame_ticks,
        keyframe_interval: ctx.keyframe_interval,
        segment_target_ticks: ctx.segment_target_ticks,
        output_dir,
        rung_invariant,
    };

    active_workers[rung_idx].fetch_add(1, Ordering::AcqRel);
    let body = async move {
        let (progress_tx, mut progress_rx) = mpsc::channel::<u64>(32);
        let cfg_for_worker = cfg.clone();
        let queue_for_worker = Arc::clone(&queue);
        let rt = tokio::runtime::Handle::current();
        let counter = Arc::clone(&frames_encoded);
        let blocking = tokio::task::spawn_blocking(move || {
            run_encoder_worker_blocking(cfg_for_worker, queue_for_worker, rt, counter, progress_tx)
        });
        // Drain the per-frame progress channel (the shared AtomicU64 counter is
        // the source of truth the reporter task reads; here we just keep the
        // channel from backpressuring the worker).
        let drain = async move { while progress_rx.recv().await.is_some() {} };
        let (_, br) = tokio::join!(drain, blocking);

        let task_status: Result<()> = match br {
            Ok(Ok(out)) => {
                contributions[rung_idx].lock().unwrap().push(out);
                Ok(())
            }
            Ok(Err(e)) => Err(e),
            Err(e) => Err(anyhow!("worker join error: {e}")),
        };
        drop(lease);
        let prev = active_workers[rung_idx].fetch_sub(1, Ordering::AcqRel);
        if prev == 1 {
            rung_done[rung_idx].notify_one();
        }
        (rung_idx, task_status)
    };

    match worker_tasks {
        Some(set) => {
            set.spawn(body);
        }
        None => {
            tokio::spawn(async move {
                let _ = body.await;
            });
        }
    }
}
