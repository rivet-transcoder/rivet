//! HLS multi-GPU orchestration: [`run_multigpu_hls`].
//!
//! # The shape
//!
//! ```text
//!   decode pump per range ──► per-rung scaler ──► SegmentChunkQueue (per rung)
//!   (one card each)            (one per range × rung)        │
//!                                                            ▼
//!                                       ladder worker (one per GPU, serves EVERY rung)
//! ```
//!
//! Two things distinguish this from a worker-per-rung ladder, and both are
//! about never letting a card sit idle while work exists:
//!
//! **Workers serve the whole ladder.** Each holds one GPU lease for the life
//! of the job and repeatedly takes the next chunk from whichever rung is
//! furthest behind. A per-rung worker idled the moment its rung was blocked
//! even with another rung's chunks sitting ready; it also capped the rungs in
//! flight at the GPU count, so a longer ladder fell back to decoding the source
//! once per rung — and decode is the dominant cost of a transcode. Because no
//! rung can now be left without a consumer, the pump is always shared and the
//! ladder costs exactly one decode however many rungs it has.
//!
//! **The decode is split across the cards.** One decoder for the whole ladder
//! is one decoder, and the giveaway that it is the limiter is rungs of very
//! different encode cost advancing in lockstep on the same segment number.
//! [`plan_decode_ranges`](crate::decode_pump::plan_decode_ranges) cuts the
//! source at keyframes that fall on segment boundaries; one pump per range,
//! pinned to its own card, feeds every rung's scaler, and the segment numbering
//! stays continuous across the join. A source that cannot be split safely
//! (an unfamiliar codec, too few keyframes, none on a boundary) is decoded
//! whole, which is exactly the behaviour before ranges existed.
//!
//! One encoder per GPU is still exactly true: `capacity` workers, each holding
//! its lease for its lifetime, each running one encode at a time. That
//! invariant is load-bearing — concurrent sessions on one device deadlocked at
//! init — and nothing here widens it.

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

use anyhow::{Result, anyhow, bail};
use container::cmaf::CmafTrackManifest;
use tokio::sync::{Notify, mpsc};
use tokio::task::JoinSet;

use crate::cmaf_util::{RungContribution, merge_rung_contributions, total_segments_for_rung};
use crate::decode_pump::DecodeRange;
use crate::encoder_worker::{
    EncoderWorkerConfig, RungCodecInvariant, UnitOutcome, WorkerOutput, encode_segment_unit,
};
use crate::frame_queue::SegmentChunkQueue;
use crate::gpu_pool::GpuLease;
use crate::progress::ProgressSink;
use crate::spec::Rung;

use super::{
    FANOUT_CHANNEL_CAPACITY, MultiGpuParams, RungManifest, WorkerCtx, queue_capacity_for,
    report, spawn_progress_reporter,
};

/// How long a ladder worker waits when every queue is empty but the job is
/// not over — the normal state of a rung whose scaler is mid-chunk. Short,
/// because the wait is on the encode critical path; not zero, because a
/// worker that never yields spins a core against the decoders.
const IDLE_POLL: std::time::Duration = std::time::Duration::from_millis(5);

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

    let capacity = params.gpu_pool.capacity().max(1);
    tracing::info!(
        rungs = n,
        total_segments,
        gpu_pool_capacity = params.gpu_pool.capacity(),
        "multi-GPU ladder starting"
    );

    // Per-rung shared state ------------------------------------------------
    //
    // Queue depth comes from a byte budget rather than a fixed count: the
    // figure scales with rung count *and* frame area, and a fixed depth that
    // was comfortable for one ladder is the OOM killer on a 4K six-rung one.
    let queues: Vec<Arc<SegmentChunkQueue>> = rungs
        .iter()
        .map(|r| {
            let depth = queue_capacity_for(r.width, r.height, params.keyframe_interval, n);
            Arc::new(SegmentChunkQueue::new(depth))
        })
        .collect();
    let frames_encoded: Vec<Arc<AtomicU64>> = (0..n).map(|_| Arc::new(AtomicU64::new(0))).collect();
    let rung_invariants: Vec<Arc<std::sync::RwLock<Option<RungCodecInvariant>>>> =
        (0..n).map(|_| Arc::new(std::sync::RwLock::new(None))).collect();
    let contributions: Arc<Vec<std::sync::Mutex<Vec<WorkerOutput>>>> =
        Arc::new((0..n).map(|_| std::sync::Mutex::new(Vec::new())).collect());
    // Who is working on each rung right now: its scalers, plus a ladder worker
    // for as long as it holds one of the rung's chunks.
    //
    // **Seeded at 1, not 0** — a setup guard released once every scaler has
    // been spawned. The finalizers are spawned before the scalers, and a
    // finalizer's first act is to break out of its wait if the count is
    // already zero; with a 0 seed the runtime only had to schedule a finalizer
    // before its scaler's `fetch_add` for that rung to conclude "nobody is
    // working on me" and return empty. Load-dependent, so it hid on a
    // two-rung three-second clip and showed up on a five-rung four-minute one.
    let active_workers: Arc<Vec<AtomicUsize>> =
        Arc::new((0..n).map(|_| AtomicUsize::new(1)).collect());
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

    // Finalizers: one per rung, merges contributions → RungManifest ---------
    let total_input_frames = params.total_input_frames;
    let (finalizer_tx, mut finalizer_rx) =
        mpsc::channel::<(usize, Result<Option<RungManifest>>)>(n.max(1));
    let mut finalizer_handles = Vec::with_capacity(n);
    for idx in 0..n {
        let contributions_h = Arc::clone(&contributions);
        let queues_h = queues.clone();
        let active_h = Arc::clone(&active_workers);
        let rung_done_h = Arc::clone(&rung_done);
        let finalized_h = Arc::clone(&finalized);
        let tx = finalizer_tx.clone();
        let rung = rungs[idx].clone();
        let rel_dir = format!("video/{}", rung.label);
        let output_root = params.output_root.clone();
        let timescale = params.timescale;
        let sink = Arc::clone(&sink);
        finalizer_handles.push(tokio::spawn(async move {
            // The rung is finished only when nothing is working on it *and*
            // nothing can be handed out: queue closed, queue empty.
            //
            // A count of zero alone used to mean "finished", which was true
            // when a rung had one worker for its whole life. A ladder worker
            // takes one chunk at a time from whichever rung is furthest
            // behind, so this rung's count legitimately returns to zero between
            // chunks — every time the last card working on it moves to another
            // rung. Finalising there takes whatever segments exist so far and
            // calls the rung done, which the coverage check then rejects.
            loop {
                let notified = rung_done_h[idx].notified();
                let queue_drained = queues_h[idx].is_closed() && queues_h[idx].depth() == 0;
                if active_h[idx].load(Ordering::Acquire) == 0 && queue_drained {
                    break;
                }
                notified.await;
            }
            let outputs: Vec<WorkerOutput> =
                std::mem::take(&mut *contributions_h[idx].lock().unwrap_or_else(|p| p.into_inner()));
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
            finalized_h[idx].store(true, Ordering::Release);
            let _ = tx.send((idx, result)).await;
        }));
    }
    drop(finalizer_tx);

    // Decode ranges --------------------------------------------------------
    //
    // Split only the un-spliced, untrimmed single input: a range is addressed
    // by demuxed sample index and its segment numbering assumes the source
    // starts at segment 0, neither of which survives a trim window or a
    // concat. Those decode whole, as they always did.
    let want_ranges = params.decode_ranges.unwrap_or(capacity).max(1);
    let ranges: Vec<DecodeRange> = if params.spliced_clips.is_empty() {
        crate::decode_pump::plan_decode_ranges(
            &params.input,
            &params.header.codec,
            params.keyframe_interval,
            want_ranges,
        )
        .unwrap_or_else(|| vec![DecodeRange::whole_source()])
    } else {
        vec![DecodeRange::whole_source()]
    };
    let multi_range = ranges.len() > 1;

    // `frame_channels[range][rung]`: one pump per range, fanning out to one
    // scaler per rung, so a rung's queue is fed by every range.
    let mut frame_senders: Vec<Vec<mpsc::Sender<codec::frame::VideoFrame>>> =
        Vec::with_capacity(ranges.len());
    let mut frame_receivers: Vec<Vec<Option<mpsc::Receiver<codec::frame::VideoFrame>>>> =
        Vec::with_capacity(ranges.len());
    for _ in 0..ranges.len() {
        let mut txs = Vec::with_capacity(n);
        let mut rxs = Vec::with_capacity(n);
        for _ in 0..n {
            let (tx, rx) = mpsc::channel(FANOUT_CHANNEL_CAPACITY);
            txs.push(tx);
            rxs.push(Some(rx));
        }
        frame_senders.push(txs);
        frame_receivers.push(rxs);
    }

    // Pumps: one per range, each on its own card ---------------------------
    //
    // With a single range the pump follows the decode policy (an explicit
    // pin, else the first decode-capable policy GPU): it feeds rungs whose
    // encoders sit on different cards, so there is no "right" one, and decoded
    // frames land in system memory anyway — a cross-adapter handoff is a
    // memcpy. With several ranges the choice does matter, because the point is
    // to have the cards decoding different stretches of the source at the same
    // time — and it has to be a card that can decode this codec, which the
    // policy's list does not promise (see `decode_capable_gpus`).
    let decode_gpus = params.decode_capable_gpus();
    let mut pump_tasks: JoinSet<Result<u64>> = JoinSet::new();
    for (range_idx, (range, senders)) in ranges.iter().zip(frame_senders.into_iter()).enumerate() {
        let mut clips =
            params.clip_sources_for(params.range_decode_gpu_for(range_idx, &decode_gpus));
        if multi_range {
            for clip in clips.iter_mut() {
                clip.cfg.sample_range = range.sample_range();
            }
        }
        let rt = tokio::runtime::Handle::current();
        pump_tasks.spawn(async move {
            tokio::task::spawn_blocking(move || {
                crate::decode_pump::run_spliced_decode_pump_blocking(clips, senders, rt)
            })
            .await
            .map_err(|e| anyhow!("decode pump (range {range_idx}) join error: {e}"))
            .and_then(|r| r)
        });
    }
    if multi_range {
        tracing::info!(
            rungs = n,
            ranges = ranges.len(),
            gpu_pool_capacity = capacity,
            boundaries = ?ranges.iter().map(|r| r.start_sample).collect::<Vec<_>>(),
            "range-parallel decode engaged — every card decodes its own stretch of the source",
        );
    } else {
        tracing::info!(rungs = n, "shared decode pump engaged (one decode for the whole ladder)");
    }

    // Scalers: one per (range, rung) ---------------------------------------
    //
    // A rung's queue is fed by every range's scaler and closed by whichever
    // finishes last — closing on the first exit would drain the workers while
    // other ranges were still feeding, losing every segment after the first
    // range's end.
    let mut scaler_tasks: JoinSet<(usize, Result<usize>)> = JoinSet::new();
    let rung_producers: Vec<Arc<AtomicUsize>> =
        (0..n).map(|_| Arc::new(AtomicUsize::new(ranges.len()))).collect();
    let last_range_idx = ranges.len() - 1;
    for (range_idx, range) in ranges.iter().enumerate() {
        // Segment numbering picks up where the previous range left off.
        // `plan_decode_ranges` guarantees the boundary is a multiple of the
        // keyframe interval, so this division is exact and the rung's
        // segments stay contiguous however the source was split.
        let first_segment_idx = (range.start_frame / u64::from(params.keyframe_interval)) as usize;
        for (idx, rung) in rungs.iter().cloned().enumerate() {
            let rx = frame_receivers[range_idx][idx].take().expect("scaler rx slot");
            let cfg = crate::rung_scaler::RungScalerConfig {
                rung_idx: idx,
                target_width: rung.width,
                target_height: rung.height,
                frames_per_chunk: params.keyframe_interval,
                // HLS segments are real files that must each stand alone;
                // there's no stitch to hide a margin in, so no overlap here.
                overlap: 0,
                first_segment_idx,
                // Only the last range owns the end of the source. A middle
                // range also finishes on a short chunk — its boundary — and
                // marking that final would end the stream mid-video.
                is_final_range: range_idx == last_range_idx,
            };
            let queue = Arc::clone(&queues[idx]);
            let rt = tokio::runtime::Handle::current();
            let active_h = Arc::clone(&active_workers);
            let rung_done_h = Arc::clone(&rung_done);
            let producers = Arc::clone(&rung_producers[idx]);
            active_h[idx].fetch_add(1, Ordering::AcqRel);
            scaler_tasks.spawn(async move {
                let result = tokio::task::spawn_blocking(move || {
                    crate::rung_scaler::run_rung_scaler_blocking_shared(cfg, rx, queue, rt, producers)
                })
                .await
                .map_err(|e| anyhow!("scaler join error: {e}"))
                .and_then(|r| r);
                if active_h[idx].fetch_sub(1, Ordering::AcqRel) == 1 {
                    rung_done_h[idx].notify_one();
                }
                (idx, result)
            });
        }
    }

    // Ladder workers: one per GPU, each serving every rung ------------------
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
    let mut worker_tasks: JoinSet<(usize, Result<()>)> = JoinSet::new();
    let mut ladder_workers = 0usize;
    for slot in 0..capacity {
        let lease = match Arc::clone(&params.gpu_pool).claim().await {
            Some(l) => l,
            None if slot == 0 => {
                progress_stop.store(true, Ordering::Release);
                let _ = progress_handle.await;
                bail!(
                    "multigpu: GPU pool returned no lease on a CPU-only host; at least one GPU is \
                     required"
                );
            }
            None => break,
        };
        spawn_ladder_worker(
            &ctx,
            slot,
            rungs,
            lease,
            queues.clone(),
            frames_encoded.clone(),
            Arc::clone(&contributions),
            Arc::clone(&active_workers),
            Arc::clone(&rung_done),
            rung_invariants.clone(),
            &mut worker_tasks,
        );
        ladder_workers += 1;
    }
    tracing::info!(
        ladder_workers,
        rungs = n,
        "ladder workers started — each serves every rung, so a card idles only when the job is done",
    );

    // Release the setup guard seeded above, now that every scaler for every
    // rung has bumped its own count. From here a zero means what the
    // finalizer thinks it means. A rung whose scalers all finished during
    // setup is why the notify is here too — without it that rung's finalizer
    // would wait forever on a wake-up that already happened.
    for idx in 0..n {
        if active_workers[idx].fetch_sub(1, Ordering::AcqRel) == 1 {
            rung_done[idx].notify_one();
        }
    }

    // Drain everything -----------------------------------------------------
    let mut completed: Vec<Option<RungManifest>> = (0..n).map(|_| None).collect();
    let mut pumps_remaining = pump_tasks.len();
    let mut scalers_remaining = scaler_tasks.len();
    let mut workers_remaining = ladder_workers;
    let mut finalizers_remaining = n;

    macro_rules! teardown_err {
        ($e:expr) => {{
            progress_stop.store(true, Ordering::Release);
            let _ = progress_handle.await;
            return Err($e);
        }};
    }

    while pumps_remaining > 0 || scalers_remaining > 0 || workers_remaining > 0 || finalizers_remaining > 0 {
        tokio::select! {
            biased;
            p = pump_tasks.join_next(), if pumps_remaining > 0 => match p {
                Some(Ok(Ok(frames))) => { tracing::info!(frames, pumps_remaining = pumps_remaining - 1, "decode pump finished"); pumps_remaining -= 1; }
                Some(Ok(Err(e))) => teardown_err!(anyhow!("decode pump failed: {e:#}")),
                Some(Err(je)) => teardown_err!(anyhow!("pump join error: {je}")),
                None => pumps_remaining = 0,
            },
            s = scaler_tasks.join_next(), if scalers_remaining > 0 => match s {
                Some(Ok((idx, Ok(chunks)))) => { tracing::debug!(idx, chunks, "scaler finished"); scalers_remaining -= 1; }
                Some(Ok((idx, Err(e)))) => teardown_err!(anyhow!("scaler {idx} failed: {e:#}")),
                Some(Err(je)) => teardown_err!(anyhow!("scaler join error: {je}")),
                None => scalers_remaining = 0,
            },
            w = worker_tasks.join_next(), if workers_remaining > 0 => match w {
                Some(Ok((slot, Ok(())))) => { tracing::debug!(slot, "ladder worker finished"); workers_remaining -= 1; }
                Some(Ok((slot, Err(e)))) => teardown_err!(anyhow!("ladder worker {slot} failed: {e:#}")),
                Some(Err(je)) => teardown_err!(anyhow!("worker join error: {je}")),
                None => workers_remaining = 0,
            },
            f = finalizer_rx.recv(), if finalizers_remaining > 0 => match f {
                Some((idx, Ok(opt))) => { completed[idx] = opt; finalizers_remaining -= 1; }
                Some((idx, Err(e))) => teardown_err!(anyhow!("finalizer for rung {idx} failed: {e:#}")),
                None => finalizers_remaining = 0,
            },
        }
    }

    progress_stop.store(true, Ordering::Release);
    let _ = progress_handle.await;
    for h in finalizer_handles {
        let _ = h.await;
    }

    Ok(completed)
}

/// The per-rung worker config for one card: the rung's own knobs, the job's
/// output format, and this worker's lease.
fn rung_worker_config(
    ctx: &WorkerCtx,
    rung_idx: usize,
    rung: &Rung,
    lease: &GpuLease,
    rung_invariant: Arc<std::sync::RwLock<Option<RungCodecInvariant>>>,
) -> EncoderWorkerConfig {
    EncoderWorkerConfig {
        overrides: rung.quality.overrides,
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
        gpu_index: Some(lease.gpu_index),
        gpu_vendor: Some(lease.vendor),
        output_color_metadata: ctx.output_color_metadata,
        output_pixel_format: ctx.output_pixel_format,
        constant_qp: ctx.constant_qp,
        timescale: ctx.timescale,
        per_frame_ticks: ctx.per_frame_ticks,
        keyframe_interval: ctx.keyframe_interval,
        segment_target_ticks: ctx.segment_target_ticks,
        output_dir: ctx.output_root.join(format!("video/{}", rung.label)),
        rung_invariant,
    }
}

/// One worker, every rung.
///
/// Holds a single GPU lease for its lifetime — so the one-encoder-per-GPU
/// invariant is untouched — and repeatedly takes the next chunk from whichever
/// rung is furthest behind.
///
/// # Why "furthest behind" rather than "smallest rung first"
///
/// The shared decode pump stalls when *any* rung queue is full. Serving the
/// fullest queue is what keeps the decode moving: it attacks the rung closest
/// to blocking everyone. Preferring the cheapest rung would publish early
/// quality sooner and then wedge the pump behind the rung nobody was serving.
///
/// # When it stops
///
/// Only when every queue is closed *and* empty. A worker that finds nothing
/// waits a beat and asks again rather than exiting, because "this rung has
/// nothing right now" is the normal state of a rung whose scaler is mid-chunk;
/// exiting on it would retire a card with work still coming.
///
/// # A rung this card cannot serve
///
/// The first packet a rung sees fixes its codec invariant, and a card of a
/// vendor whose sequence header disagrees on a mandatory field can never
/// contribute to that rung — the disagreement is a property of the silicon,
/// not of the chunk. Such a chunk goes back to the head of the queue for
/// another card, and the rung is struck off this worker's list, so it does not
/// spin re-building encoders against a rung it will be refused by every time.
#[allow(clippy::too_many_arguments)]
fn spawn_ladder_worker(
    ctx: &WorkerCtx,
    slot: usize,
    rungs: &[Rung],
    lease: GpuLease,
    queues: Vec<Arc<SegmentChunkQueue>>,
    frames_encoded: Vec<Arc<AtomicU64>>,
    contributions: Arc<Vec<std::sync::Mutex<Vec<WorkerOutput>>>>,
    active_workers: Arc<Vec<AtomicUsize>>,
    rung_done: Arc<Vec<Notify>>,
    rung_invariants: Vec<Arc<std::sync::RwLock<Option<RungCodecInvariant>>>>,
    worker_tasks: &mut JoinSet<(usize, Result<()>)>,
) {
    let gpu_index = lease.gpu_index;
    let gpu_vendor = lease.vendor;
    let configs: Vec<EncoderWorkerConfig> = rungs
        .iter()
        .enumerate()
        .map(|(idx, rung)| rung_worker_config(ctx, idx, rung, &lease, Arc::clone(&rung_invariants[idx])))
        .collect();

    let body = async move {
        // The per-frame progress channel is a formality here: the shared
        // counters are what the reporter reads. It exists so the worker is
        // never backpressured by nobody listening.
        let (progress_tx, mut progress_rx) = mpsc::channel::<u64>(32);
        let drain = tokio::spawn(async move { while progress_rx.recv().await.is_some() {} });

        let blocking = tokio::task::spawn_blocking(move || -> Result<()> {
            let mut init_written: Vec<bool> = vec![false; configs.len()];
            let mut refused: HashSet<usize> = HashSet::new();
            loop {
                // Pick the rung closest to blocking the pump.
                let mut best: Option<(usize, usize)> = None;
                for (idx, q) in queues.iter().enumerate() {
                    if refused.contains(&idx) {
                        continue;
                    }
                    let depth = q.depth();
                    if depth == 0 {
                        continue;
                    }
                    if best.is_none_or(|(_, d)| depth > d) {
                        best = Some((idx, depth));
                    }
                }

                let Some((rung_idx, _)) = best else {
                    // Nothing anywhere. Finished only if nothing can arrive.
                    if queues.iter().all(|q| q.is_closed() && q.depth() == 0) {
                        break;
                    }
                    std::thread::sleep(IDLE_POLL);
                    continue;
                };

                let Some(chunk) = queues[rung_idx].try_pop() else {
                    // Another worker took it between the look and the grab.
                    continue;
                };

                // Held across the encode so this rung's finalizer cannot decide
                // the rung is finished while a chunk of it is still in a card.
                active_workers[rung_idx].fetch_add(1, Ordering::AcqRel);
                let outcome = encode_segment_unit(
                    &configs[rung_idx],
                    chunk,
                    &mut init_written[rung_idx],
                    &frames_encoded[rung_idx],
                    &progress_tx,
                );
                let released = || {
                    if active_workers[rung_idx].fetch_sub(1, Ordering::AcqRel) == 1 {
                        rung_done[rung_idx].notify_one();
                    }
                };
                match outcome {
                    Ok(UnitOutcome::Wrote(info)) => {
                        // Which card did which segment of which rung — the line
                        // that answers "what is actually happening" on a fleet.
                        tracing::info!(
                            rung_idx,
                            gpu_index,
                            segment = info.sequence_number,
                            "rung segment flushed",
                        );
                        // Recorded before the count drops, so the finalizer
                        // woken by that drop sees this segment.
                        contributions[rung_idx]
                            .lock()
                            .unwrap_or_else(|p| p.into_inner())
                            .push(WorkerOutput { gpu_index: Some(gpu_index), segments: vec![info] });
                        released();
                    }
                    Ok(UnitOutcome::Rejected { chunk, diff }) => {
                        tracing::warn!(
                            rung_idx,
                            gpu_index,
                            gpu_vendor = ?gpu_vendor,
                            rejected_segment = chunk.segment_idx,
                            diff = %diff,
                            "codec invariant mismatch — returning the chunk for another card \
                             and leaving this rung to them",
                        );
                        queues[rung_idx].push_front(chunk);
                        refused.insert(rung_idx);
                        released();
                    }
                    Err(e) => {
                        released();
                        return Err(e);
                    }
                }
            }
            Ok(())
        });

        let status: Result<()> = match blocking.await {
            Ok(Ok(())) => {
                tracing::info!(slot, gpu_index, "ladder worker exited cleanly");
                Ok(())
            }
            Ok(Err(e)) => Err(e),
            Err(e) => Err(anyhow!("ladder worker join error: {e}")),
        };
        // `progress_tx` moved into the blocking task and dropped with it, which
        // is what ends the drain.
        let _ = drain.await;
        drop(lease);
        (slot, status)
    };
    worker_tasks.spawn(body);
}
