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
//! Ported from the transcoder microservice's `shared_decoder_phase`, with the
//! SQS/S3 specifics (StatusReporter, incremental publish) replaced by the
//! generic [`ProgressSink`]. The microservice layers its S3 upload back on by
//! watching `RungStatus::Completed` from the sink.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

use anyhow::{Result, anyhow, bail};
use bytes::Bytes;
use codec::encode::AUTO_FROM_TARGET;
use codec::frame::{ColorMetadata, PixelFormat};
use container::cmaf::CmafTrackManifest;
use container::streaming::DemuxHeader;
use tokio::sync::{Notify, mpsc};
use tokio::task::JoinSet;

use codec::encode::EncodedPacket;

use crate::cmaf_util::{RungContribution, merge_rung_contributions, total_segments_for_rung};
use crate::decode_pump::DecodePumpConfig;
use crate::encoder_worker::{
    ChunkPackets, EncoderWorkerConfig, RungCodecInvariant, WorkerOutput,
    run_chunk_encoder_worker_blocking, run_encoder_worker_blocking,
};
use crate::frame_queue::SegmentChunkQueue;
use crate::gpu_pool::{GpuLease, GpuPool};
use crate::progress::{ProgressSink, RungProgress, RungStatus};
use crate::spec::{EncodePolicy, GpuFamily, Rung};

const QUEUE_CAPACITY: usize = 2;
const FANOUT_CHANNEL_CAPACITY: usize = 4;
const HELPER_POLL_INTERVAL: std::time::Duration = std::time::Duration::from_millis(200);
const PROGRESS_TICK: std::time::Duration = std::time::Duration::from_millis(500);

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
    pub rungs: &'a [Rung],
    pub header: DemuxHeader,
    pub source_color_metadata: ColorMetadata,
    pub source_pixel_format: PixelFormat,
    pub needs_downsample: bool,
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
}

impl MultiGpuParams<'_> {
    /// Resolve the decode-pump GPU for the `i`-th per-rung pump (or the shared
    /// pump when `i == 0`): the explicit `decode_gpu` override wins, else the
    /// policy's GPU indices round-robin, else `None` (decoder auto-select).
    fn decode_gpu_for(&self, i: usize) -> Option<u32> {
        if self.decode_gpu.is_some() {
            return self.decode_gpu;
        }
        if self.gpu_indices.is_empty() {
            return None;
        }
        Some(self.gpu_indices[i % self.gpu_indices.len()])
    }
}

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

    // Pre-flight: verify this host can actually construct an AV1 encoder
    // before spawning the orchestration. Fail fast with a clear error instead
    // of dispatching workers that fail at encoder construction — and, on
    // drivers that re-init a failed NVENC session badly (e.g. Ampere with no
    // AV1-encode silicon), would otherwise hang an uncancellable blocking task.
    {
        let probe = codec::encode::EncoderConfig {
            width: rungs[0].width,
            height: rungs[0].height,
            frame_rate: params.frame_rate,
            gpu_index: None,
            ..Default::default()
        };
        codec::encode::select_encoder(probe, None).map_err(|e| {
            anyhow!(
                "no AV1 encoder available on this host ({e}); need NVENC (Ada+) / AMF \
                 (RDNA3+) / QSV (Arc+), or build with the `ffmpeg` feature for a software encoder"
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
                            RungStatus::Completed,
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
    let make_pump_cfg = |gpu_index: Option<u32>| DecodePumpConfig {
        codec_name: params.header.codec.clone(),
        info_for_decoder: params.header.info.clone(),
        source_color_metadata: params.source_color_metadata,
        source_pixel_format: params.source_pixel_format,
        needs_downsample: params.needs_downsample,
        gpu_index,
    };
    if use_shared_pump {
        let cfg = make_pump_cfg(params.decode_gpu_for(0));
        let senders = frame_senders;
        let input = params.input.clone();
        let rt = tokio::runtime::Handle::current();
        pump_tasks.spawn(async move {
            tokio::task::spawn_blocking(move || {
                crate::decode_pump::run_shared_decode_pump_blocking(cfg, input, senders, rt)
            })
            .await
            .map_err(|e| anyhow!("shared pump join error: {e}"))
            .and_then(|r| r)
        });
    } else {
        for (idx, sender) in frame_senders.into_iter().enumerate() {
            let cfg = make_pump_cfg(params.decode_gpu_for(idx));
            let input = params.input.clone();
            let rt = tokio::runtime::Handle::current();
            pump_tasks.spawn(async move {
                tokio::task::spawn_blocking(move || {
                    crate::decode_pump::run_shared_decode_pump_blocking(cfg, input, vec![sender], rt)
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

    // Initial encoder workers (one per rung, smallest first).
    let mut worker_tasks: JoinSet<(usize, Result<()>)> = JoinSet::new();
    let ctx = WorkerCtx {
        frame_rate: params.frame_rate,
        source_color_metadata: params.source_color_metadata,
        source_pixel_format: params.source_pixel_format,
        timescale: params.timescale,
        per_frame_ticks: params.per_frame_ticks,
        keyframe_interval: params.keyframe_interval,
        segment_target_ticks: params.segment_target_ticks,
        output_root: params.output_root.clone(),
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

/// Per-job constants shared by every encoder worker.
#[derive(Clone)]
struct WorkerCtx {
    frame_rate: f64,
    source_color_metadata: ColorMetadata,
    source_pixel_format: PixelFormat,
    timescale: u32,
    per_frame_ticks: u32,
    keyframe_interval: u32,
    segment_target_ticks: u64,
    output_root: PathBuf,
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
        rung_idx,
        width: rung.width,
        height: rung.height,
        frame_rate: ctx.frame_rate,
        quality: rung.quality.crf.unwrap_or(AUTO_FROM_TARGET),
        speed_preset: rung.quality.speed_preset.unwrap_or(AUTO_FROM_TARGET),
        target: rung.quality.target,
        tier: rung.quality.tier,
        threads: 0,
        gpu_index: Some(gpu_index),
        gpu_vendor: Some(gpu_vendor),
        source_color_metadata: ctx.source_color_metadata,
        source_pixel_format: ctx.source_pixel_format,
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

/// Periodic per-rung progress reporter. Reads the shared frame counters and
/// emits `Running` updates until stopped; skips rungs already finalized.
fn spawn_progress_reporter(
    rungs: Vec<Rung>,
    frames_encoded: Vec<Arc<AtomicU64>>,
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
                report(
                    sink.as_ref(),
                    idx,
                    rung,
                    RungStatus::Running,
                    done,
                    Some(total_input_frames),
                    0,
                    0,
                    None,
                );
            }
        }
    })
}

#[allow(clippy::too_many_arguments)]
fn report(
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

/// Build a [`GpuPool`] from the host's detected GPU inventory.
pub fn detect_gpu_pool() -> Arc<GpuPool> {
    Arc::new(GpuPool::new(&codec::gpu::detect_gpus()))
}

fn policy_vendor(fam: GpuFamily) -> codec::gpu::GpuVendor {
    match fam {
        GpuFamily::Nvidia => codec::gpu::GpuVendor::Nvidia,
        GpuFamily::Amd => codec::gpu::GpuVendor::Amd,
        GpuFamily::Intel => codec::gpu::GpuVendor::Intel,
    }
}

/// The host GPUs selected by an [`EncodePolicy`]: all of them for `AllGpus`,
/// the first / pinned index for `SingleGpu`, every device of one vendor for
/// `Family`.
fn select_gpus_for_policy(policy: EncodePolicy) -> Vec<codec::gpu::GpuDevice> {
    let gpus = codec::gpu::detect_gpus();
    match policy {
        EncodePolicy::AllGpus => gpus,
        EncodePolicy::SingleGpu(None) => gpus.into_iter().take(1).collect(),
        EncodePolicy::SingleGpu(Some(idx)) => gpus.into_iter().filter(|g| g.index == idx).collect(),
        EncodePolicy::Family(fam) => {
            let v = policy_vendor(fam);
            gpus.into_iter().filter(|g| g.vendor == v).collect()
        }
    }
}

/// Build a [`GpuPool`] constrained to the given [`EncodePolicy`]. An empty pool
/// (e.g. a pinned index or vendor family that isn't present) yields capacity 0,
/// so the orchestrator's pre-flight probe / lease claim surfaces a clear error.
pub fn gpu_pool_for_policy(policy: EncodePolicy) -> Arc<GpuPool> {
    Arc::new(GpuPool::new(&select_gpus_for_policy(policy)))
}

/// The GPU indices an [`EncodePolicy`] selects, in detection order. Used to pin
/// the decode pump to a device consistent with the policy (so decode honors a
/// `Family` / `SingleGpu` constraint, not just encode).
pub fn policy_gpu_indices(policy: EncodePolicy) -> Vec<u32> {
    select_gpus_for_policy(policy).into_iter().map(|g| g.index).collect()
}

/// The GPU index to pin a *serial* (single-GPU) encode/decode to under a
/// policy: `None` (auto/first-available) for `AllGpus`, the pinned index for
/// `SingleGpu`, the first device of the vendor for `Family`.
pub fn serial_gpu_for_policy(policy: EncodePolicy) -> Option<u32> {
    match policy {
        EncodePolicy::AllGpus => None,
        EncodePolicy::SingleGpu(idx) => idx,
        EncodePolicy::Family(_) => select_gpus_for_policy(policy).first().map(|g| g.index),
    }
}

// ---------------------------------------------------------------------------
// Single-file multi-GPU: chunk each rung across GPUs, stitch packets into MP4.
// ---------------------------------------------------------------------------

/// One rung's full ordered AV1 packet stream, stitched from chunks encoded
/// across GPUs. The caller muxes these into a single MP4 (+ audio).
#[derive(Debug)]
pub struct RungPackets {
    pub rung_index: usize,
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
            ..Default::default()
        };
        codec::encode::select_encoder(probe, None).map_err(|e| {
            anyhow!(
                "no AV1 encoder available on this host ({e}); need NVENC (Ada+) / AMF \
                 (RDNA3+) / QSV (Arc+), or build with the `ffmpeg` feature"
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
        finalized.clone(),
        params.total_input_frames,
        Arc::clone(&sink),
        Arc::clone(&progress_stop),
    );

    // Finalizers: stitch each rung's chunks (sorted, deduped) into one stream.
    let total_input_frames = params.total_input_frames;
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
                    RungStatus::Completed,
                    total_input_frames,
                    Some(total_input_frames),
                    got as u32,
                    bytes,
                    None,
                );
                Ok(Some(RungPackets {
                    rung_index: idx,
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
    let make_pump_cfg = |gpu_index: Option<u32>| DecodePumpConfig {
        codec_name: params.header.codec.clone(),
        info_for_decoder: params.header.info.clone(),
        source_color_metadata: params.source_color_metadata,
        source_pixel_format: params.source_pixel_format,
        needs_downsample: params.needs_downsample,
        gpu_index,
    };
    if use_shared_pump {
        let cfg = make_pump_cfg(params.decode_gpu_for(0));
        let senders = frame_senders;
        let input = params.input.clone();
        let rt = tokio::runtime::Handle::current();
        pump_tasks.spawn(async move {
            tokio::task::spawn_blocking(move || {
                crate::decode_pump::run_shared_decode_pump_blocking(cfg, input, senders, rt)
            })
            .await
            .map_err(|e| anyhow!("shared pump join error: {e}"))
            .and_then(|r| r)
        });
    } else {
        for (idx, sender) in frame_senders.into_iter().enumerate() {
            let cfg = make_pump_cfg(params.decode_gpu_for(idx));
            let input = params.input.clone();
            let rt = tokio::runtime::Handle::current();
            pump_tasks.spawn(async move {
                tokio::task::spawn_blocking(move || {
                    crate::decode_pump::run_shared_decode_pump_blocking(cfg, input, vec![sender], rt)
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
        frame_rate: params.frame_rate,
        source_color_metadata: params.source_color_metadata,
        source_pixel_format: params.source_pixel_format,
        timescale: params.timescale,
        per_frame_ticks: params.per_frame_ticks,
        keyframe_interval: params.keyframe_interval,
        segment_target_ticks: params.segment_target_ticks,
        output_root: params.output_root.clone(),
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
        width: rung.width,
        height: rung.height,
        frame_rate: ctx.frame_rate,
        quality: rung.quality.crf.unwrap_or(AUTO_FROM_TARGET),
        speed_preset: rung.quality.speed_preset.unwrap_or(AUTO_FROM_TARGET),
        target: rung.quality.target,
        tier: rung.quality.tier,
        threads: 0,
        gpu_index: Some(gpu_index),
        gpu_vendor: Some(gpu_vendor),
        source_color_metadata: ctx.source_color_metadata,
        source_pixel_format: ctx.source_pixel_format,
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
        let out = Arc::clone(&collector);
        let blocking = tokio::task::spawn_blocking(move || {
            run_chunk_encoder_worker_blocking(cfg_for_worker, queue_for_worker, rt, counter, progress_tx, out)
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
