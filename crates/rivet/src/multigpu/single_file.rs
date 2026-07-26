//! Single-file multi-GPU orchestration: [`run_multigpu_single_file`] + chunk
//! worker spawn helper.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

use anyhow::{Result, anyhow, bail};
use tokio::sync::{Notify, mpsc};
use tokio::task::JoinSet;

use codec::encode::EncodedPacket;
use codec::frame::VideoCodec;

use crate::encoder_worker::{
    ChunkPackets, EncoderWorkerConfig, RungCodecInvariant, run_chunk_encoder_worker_blocking,
};
use crate::frame_queue::SegmentChunkQueue;
use crate::gpu_pool::GpuLease;
use crate::progress::ProgressSink;
use crate::spec::Rung;

use crate::cmaf_util::total_segments_for_rung;

use super::{
    FANOUT_CHANNEL_CAPACITY, HELPER_POLL_INTERVAL, QUEUE_CAPACITY, MultiGpuParams, WorkerCtx,
    report, spawn_progress_reporter,
};

/// One rung's full ordered AV1 packet stream, stitched from chunks encoded
/// across GPUs. The caller muxes these into a single MP4 (+ audio).
#[derive(Debug)]
pub struct RungPackets {
    pub rung_index: usize,
    pub codec: VideoCodec,
    pub width: u32,
    pub height: u32,
    pub label: String,
    pub packets: Vec<EncodedPacket>,
}

/// Single-file counterpart to [`run_multigpu_hls`]: decode once, fan to per-rung
/// scalers, and dynamically schedule each rung's GOP-sized chunks across all
/// GPUs (fair lease pool + mid-flight helper dispatch + cross-vendor codec
/// invariant). Each worker encodes its chunk to packets (a fresh encoder per
/// chunk → first frame is an IDR); the finalizer concatenates them in segment
/// order into one ordered packet stream per rung — no disk round-trip.
pub async fn run_multigpu_single_file(
    params: MultiGpuParams<'_>,
    sink: Arc<dyn ProgressSink>,
) -> Result<Vec<Option<RungPackets>>> {
    let rungs = params.rungs;
    let n = rungs.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    let total_segments = total_segments_for_rung(params.total_input_frames, params.keyframe_interval);
    if total_segments == 0 {
        bail!(
            "multigpu single-file: total_segments == 0 (frames={}, keyframe_interval={})",
            params.total_input_frames,
            params.keyframe_interval
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
                 with the `ffmpeg` feature",
                params.codec
            )
        })?;
    }

    tracing::info!(
        rungs = n,
        total_segments,
        gpu_pool_capacity = params.gpu_pool.capacity(),
        "multi-GPU single-file phase starting"
    );

    let queues: Vec<Arc<SegmentChunkQueue>> =
        (0..n).map(|_| Arc::new(SegmentChunkQueue::new(QUEUE_CAPACITY))).collect();
    let frames_encoded: Vec<Arc<AtomicU64>> = (0..n).map(|_| Arc::new(AtomicU64::new(0))).collect();
    let bytes_encoded: Vec<Arc<AtomicU64>> = (0..n).map(|_| Arc::new(AtomicU64::new(0))).collect();
    let scaler_active: Vec<Arc<AtomicBool>> =
        (0..n).map(|_| Arc::new(AtomicBool::new(false))).collect();
    let rung_invariants: Vec<Arc<std::sync::RwLock<Option<RungCodecInvariant>>>> =
        (0..n).map(|_| Arc::new(std::sync::RwLock::new(None))).collect();
    // Per-rung packet collectors (each its own Arc so chunk workers can push).
    let contributions: Vec<Arc<std::sync::Mutex<Vec<ChunkPackets>>>> =
        (0..n).map(|_| Arc::new(std::sync::Mutex::new(Vec::new()))).collect();
    let active_workers: Arc<Vec<AtomicUsize>> =
        Arc::new((0..n).map(|_| AtomicUsize::new(0)).collect());
    let rung_done: Arc<Vec<Notify>> = Arc::new((0..n).map(|_| Notify::new()).collect());
    let finalized: Arc<Vec<AtomicBool>> = Arc::new((0..n).map(|_| AtomicBool::new(false)).collect());

    let progress_stop = Arc::new(AtomicBool::new(false));
    let progress_handle = spawn_progress_reporter(
        rungs.to_vec(),
        frames_encoded.clone(),
        bytes_encoded.clone(),
        finalized.clone(),
        params.total_input_frames,
        Arc::clone(&sink),
        Arc::clone(&progress_stop),
    );

    // Finalizers: stitch each rung's chunks (sorted, deduped) into one stream.
    let total_input_frames = params.total_input_frames;
    let codec = params.codec; // Copy; captured by each finalizer closure
    let (finalizer_tx, mut finalizer_rx) =
        mpsc::channel::<(usize, Result<Option<RungPackets>>)>(n.max(1));
    let mut finalizer_handles = Vec::with_capacity(n);
    for idx in 0..n {
        let collector = Arc::clone(&contributions[idx]);
        let active_h = Arc::clone(&active_workers);
        let rung_done_h = Arc::clone(&rung_done);
        let finalized_h = Arc::clone(&finalized);
        let tx = finalizer_tx.clone();
        let rung = rungs[idx].clone();
        let total_segments = total_segments;
        let sink = Arc::clone(&sink);
        finalizer_handles.push(tokio::spawn(async move {
            loop {
                let notified = rung_done_h[idx].notified();
                if active_h[idx].load(Ordering::Acquire) == 0 {
                    break;
                }
                notified.await;
            }
            let mut chunks: Vec<ChunkPackets> = std::mem::take(&mut *collector.lock().unwrap());
            if chunks.is_empty() {
                finalized_h[idx].store(true, Ordering::Release);
                let _ = tx.send((idx, Ok(None))).await;
                return;
            }
            chunks.sort_by_key(|c| c.segment_idx);
            chunks.dedup_by_key(|c| c.segment_idx);
            // Coverage: contiguous 0..total_segments.
            let got = chunks.len();
            let contiguous = chunks
                .iter()
                .enumerate()
                .all(|(i, c)| c.segment_idx == i);
            let result = if got != total_segments as usize || !contiguous {
                Err(anyhow!(
                    "rung {} chunk coverage incomplete: expected {} contiguous chunks, got {}",
                    rung.label,
                    total_segments,
                    got
                ))
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
                    got as u32,
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
            finalized_h[idx].store(true, Ordering::Release);
            let _ = tx.send((idx, result)).await;
        }));
    }
    drop(finalizer_tx);

    let mut indexed: Vec<(usize, Rung)> = rungs.iter().cloned().enumerate().collect();
    indexed.sort_by_key(|(_, r)| r.short_side());

    // Decode pump(s) + fan-out.
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

    // Initial chunk workers.
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
                bail!("multigpu single-file: GPU pool returned no lease; at least one GPU required");
            }
        };
        spawn_chunk_worker(
            &ctx,
            idx,
            &rung,
            Arc::clone(&queues[idx]),
            Arc::clone(&frames_encoded[idx]),
            Arc::clone(&bytes_encoded[idx]),
            lease,
            Arc::clone(&contributions[idx]),
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
        let bytes_encoded = bytes_encoded.clone();
        let contributions = contributions.clone();
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
                tracing::info!(rung_idx, gpu_index = lease.gpu_index, "single-file helper dispatch");
                spawn_chunk_worker(
                    &ctx,
                    rung_idx,
                    &rungs_owned[rung_idx],
                    Arc::clone(&queues[rung_idx]),
                    Arc::clone(&frames_encoded[rung_idx]),
                    Arc::clone(&bytes_encoded[rung_idx]),
                    lease,
                    Arc::clone(&contributions[rung_idx]),
                    Arc::clone(&active_workers),
                    Arc::clone(&rung_done),
                    Arc::clone(&rung_invariants[rung_idx]),
                    None,
                );
            }
        })
    };

    // Drain.
    let mut completed: Vec<Option<RungPackets>> = (0..n).map(|_| None).collect();
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
                Some(Ok(Ok(_))) => pumps_remaining -= 1,
                Some(Ok(Err(e))) => teardown_err!(anyhow!("decode pump failed: {e}")),
                Some(Err(je)) => teardown_err!(anyhow!("pump join error: {je}")),
                None => pumps_remaining = 0,
            },
            s = scaler_tasks.join_next(), if scalers_remaining > 0 => match s {
                Some(Ok((_, Ok(_)))) => scalers_remaining -= 1,
                Some(Ok((idx, Err(e)))) => teardown_err!(anyhow!("scaler {idx} failed: {e}")),
                Some(Err(je)) => teardown_err!(anyhow!("scaler join error: {je}")),
                None => scalers_remaining = 0,
            },
            w = worker_tasks.join_next(), if workers_remaining > 0 => match w {
                Some(Ok((_, Ok(())))) => workers_remaining -= 1,
                Some(Ok((idx, Err(e)))) => teardown_err!(anyhow!("chunk worker for rung {idx} failed: {e}")),
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
fn spawn_chunk_worker(
    ctx: &WorkerCtx,
    rung_idx: usize,
    rung: &Rung,
    queue: Arc<SegmentChunkQueue>,
    frames_encoded: Arc<AtomicU64>,
    bytes_encoded: Arc<AtomicU64>,
    lease: GpuLease,
    collector: Arc<std::sync::Mutex<Vec<ChunkPackets>>>,
    active_workers: Arc<Vec<AtomicUsize>>,
    rung_done: Arc<Vec<Notify>>,
    rung_invariant: Arc<std::sync::RwLock<Option<RungCodecInvariant>>>,
    worker_tasks: Option<&mut JoinSet<(usize, Result<()>)>>,
) {
    let gpu_index = lease.gpu_index;
    let gpu_vendor = lease.vendor;
    let cfg = EncoderWorkerConfig {
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
        output_dir: ctx.output_root.clone(),
        rung_invariant,
    };
    active_workers[rung_idx].fetch_add(1, Ordering::AcqRel);
    let body = async move {
        let (progress_tx, mut progress_rx) = mpsc::channel::<u64>(32);
        let cfg_for_worker = cfg.clone();
        let queue_for_worker = Arc::clone(&queue);
        let rt = tokio::runtime::Handle::current();
        let counter = Arc::clone(&frames_encoded);
        let byte_counter = Arc::clone(&bytes_encoded);
        let out = Arc::clone(&collector);
        let blocking = tokio::task::spawn_blocking(move || {
            run_chunk_encoder_worker_blocking(
                cfg_for_worker,
                queue_for_worker,
                rt,
                counter,
                byte_counter,
                progress_tx,
                out,
            )
        });
        let drain = async move { while progress_rx.recv().await.is_some() {} };
        let (_, br) = tokio::join!(drain, blocking);
        let task_status: Result<()> = match br {
            Ok(Ok(())) => Ok(()),
            Ok(Err(e)) => Err(e),
            Err(e) => Err(anyhow!("chunk worker join error: {e}")),
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
