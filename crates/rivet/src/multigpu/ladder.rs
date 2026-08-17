//! The ladder core shared by the HLS and single-file paths.
//!
//! ```text
//!   decode pump per range ──► per-rung scaler ──► SegmentChunkQueue (per rung)
//!   (one card each)            (one per range × rung)        │
//!                                                            ▼
//!                                       ladder worker (one per GPU, serves EVERY rung)
//! ```
//!
//! Everything about *how* the ladder is scheduled lives here, once: the
//! range-split decode, the scalers and their continuous segment numbering,
//! the byte-budgeted queues, the workers, the setup guard on the active count
//! and the "finished" rule the finalizers wait on. What differs between the
//! two output paths — what a worker does with a chunk (write a CMAF segment
//! file, or collect its packets) and what a finalizer does with a rung's
//! contributions (merge manifests, or stitch packets) — is passed in.
//!
//! # Two things distinguish this from a worker-per-rung ladder
//!
//! Both are about never letting a card sit idle while work exists.
//!
//! **Workers serve the whole ladder.** Each holds one GPU lease for the life
//! of the job and repeatedly takes the next chunk from whichever rung is
//! furthest behind ([`RungSchedule::Ladder`]). A per-rung worker idled the
//! moment its rung was blocked even with another rung's chunks sitting ready;
//! it also capped the rungs in flight at the GPU count, so a longer ladder fell
//! back to decoding the source once per rung — and decode is the dominant cost
//! of a transcode. Because no rung can now be left without a consumer, the pump
//! is always shared and the ladder costs exactly one decode however many rungs
//! it has. [`RungSchedule::PerRung`] keeps the pinned shape available for
//! comparison and for hosts where placement matters more than throughput.
//!
//! **The decode is split across the cards.** One decoder for the whole ladder
//! is one decoder, and the giveaway that it is the limiter is rungs of very
//! different encode cost advancing in lockstep on the same segment number.
//! [`plan_decode_ranges`](crate::decode_pump::plan_decode_ranges) cuts the
//! source at keyframes that fall on chunk boundaries; one pump per range,
//! pinned to its own card, feeds every rung's scaler, and the numbering stays
//! continuous across the join ([`DecodeSplit`](crate::spec::DecodeSplit)). A source that cannot be split
//! safely is decoded whole, which is exactly the behaviour before ranges
//! existed.
//!
//! One encoder per GPU is still exactly true: `capacity` workers, each holding
//! its lease for its lifetime, each running one encode at a time. That
//! invariant is load-bearing — concurrent sessions on one device deadlocked at
//! init — and nothing here widens it.

use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use anyhow::{Result, anyhow, bail};
use tokio::sync::{Notify, mpsc};
use tokio::task::JoinSet;

use codec::frame::VideoFrame;

use crate::decode_pump::DecodeRange;
use crate::encoder_worker::{EncoderWorkerConfig, RungCodecInvariant};
use crate::frame_queue::{SegmentChunk, SegmentChunkQueue};
use crate::gpu_pool::GpuLease;
use crate::spec::{Rung, RungSchedule};

use super::{FANOUT_CHANNEL_CAPACITY, MultiGpuParams, WorkerCtx, queue_capacity_for};

/// How long a ladder worker waits when every queue it serves is empty but the
/// job is not over — the normal state of a rung whose scaler is mid-chunk.
/// Short, because the wait is on the encode critical path; not zero, because a
/// worker that never yields spins a core against the decoders.
const IDLE_POLL: std::time::Duration = std::time::Duration::from_millis(5);

/// The chunking a path asks for.
#[derive(Debug, Clone, Copy)]
pub(super) struct LadderShape {
    /// Frames per unit of work — a CMAF segment for HLS, several GOPs for
    /// single-file. Also the grid decode ranges must land on.
    pub frames_per_chunk: u32,
    /// Lead-in margin frames replayed from the previous chunk's tail (encoded
    /// to warm the encoder, then discarded). `0` for HLS, whose segments must
    /// each stand alone; one GOP for single-file chunk-and-stitch.
    ///
    /// At a decode-range boundary the first chunk of the new range has no
    /// tail to replay — that range's scaler never saw those frames — so it
    /// starts cold, exactly as the file's first chunk always does. Its first
    /// kept frame is still an IDR by the encoder's own cadence, so the stitch
    /// is correct; the seam is merely one warm-up less flat, once per range.
    pub overlap: usize,
}

/// What one unit of work produced, whatever the unit is.
pub(super) enum UnitOutcome<T> {
    /// The unit was encoded and its result recorded.
    Done(T),
    /// This worker's vendor disagrees with the rung's codec invariant on a
    /// mandatory field. The chunk comes back untouched for another card.
    Rejected { chunk: SegmentChunk, diff: String },
}

/// The per-run state every part of the ladder shares. `T` is one worker's
/// contribution to a rung — a segment's [`SegmentInfo`](container::cmaf::SegmentInfo)
/// wrapped as a `WorkerOutput`, or a chunk's packets.
pub(super) struct Ladder<T> {
    pub queues: Vec<Arc<SegmentChunkQueue>>,
    pub frames_encoded: Vec<Arc<AtomicU64>>,
    pub bytes_encoded: Vec<Arc<AtomicU64>>,
    pub rung_invariants: Vec<Arc<RwLock<Option<RungCodecInvariant>>>>,
    /// Outputs from every worker on a rung, accumulated until the rung's
    /// finalizer drains it.
    pub contributions: Arc<Vec<Mutex<Vec<T>>>>,
    /// Who is working on each rung right now: its scalers, plus a worker for
    /// as long as it holds one of the rung's chunks.
    ///
    /// **Seeded at 1, not 0** — a setup guard released by
    /// [`Self::release_setup_guard`] once every scaler has been spawned. The
    /// finalizers are spawned before the scalers, and a finalizer's first act
    /// is to break out of its wait if the count is already zero; with a 0 seed
    /// the runtime only had to schedule a finalizer before its scaler's
    /// `fetch_add` for that rung to conclude "nobody is working on me" and
    /// return empty. Load-dependent, so it hid on a two-rung three-second clip
    /// and showed up on a five-rung four-minute one.
    pub active_workers: Arc<Vec<AtomicUsize>>,
    pub rung_done: Arc<Vec<Notify>>,
    /// Set by each finalizer before its terminal report, so the periodic
    /// progress reporter stops printing `Running` for a rung that is done.
    pub finalized: Arc<Vec<AtomicBool>>,
}

impl<T: Send + 'static> Ladder<T> {
    /// Per-rung state for `rungs`, with each queue's depth derived from the
    /// byte budget rather than a fixed count (see `queue_capacity_for`).
    pub fn new(rungs: &[Rung], frames_per_chunk: u32) -> Self {
        let n = rungs.len();
        Self {
            queues: rungs
                .iter()
                .map(|r| {
                    let depth = queue_capacity_for(r.width, r.height, frames_per_chunk, n);
                    Arc::new(SegmentChunkQueue::new(depth))
                })
                .collect(),
            frames_encoded: (0..n).map(|_| Arc::new(AtomicU64::new(0))).collect(),
            bytes_encoded: (0..n).map(|_| Arc::new(AtomicU64::new(0))).collect(),
            rung_invariants: (0..n).map(|_| Arc::new(RwLock::new(None))).collect(),
            contributions: Arc::new((0..n).map(|_| Mutex::new(Vec::new())).collect()),
            active_workers: Arc::new((0..n).map(|_| AtomicUsize::new(1)).collect()),
            rung_done: Arc::new((0..n).map(|_| Notify::new()).collect()),
            finalized: Arc::new((0..n).map(|_| AtomicBool::new(false)).collect()),
        }
    }

    /// Wait until rung `idx` is finished: nothing is working on it *and*
    /// nothing can be handed out — queue closed, queue empty.
    ///
    /// A count of zero alone used to mean "finished", which was true when a
    /// rung had one worker for its whole life. A ladder worker takes one chunk
    /// at a time from whichever rung is furthest behind, so this rung's count
    /// legitimately returns to zero between chunks — every time the last card
    /// working on it moves to another rung. Finalising there takes whatever
    /// segments exist so far and calls the rung done, which the coverage check
    /// then rejects.
    pub async fn wait_rung_finished(&self, idx: usize) {
        loop {
            let notified = self.rung_done[idx].notified();
            let queue_drained = self.queues[idx].is_closed() && self.queues[idx].depth() == 0;
            if self.active_workers[idx].load(Ordering::Acquire) == 0 && queue_drained {
                return;
            }
            notified.await;
        }
    }

    /// Everything the workers recorded for rung `idx`, leaving it empty.
    pub fn take_contributions(&self, idx: usize) -> Vec<T> {
        std::mem::take(&mut *self.contributions[idx].lock().unwrap_or_else(|p| p.into_inner()))
    }

    /// Release the setup guard seeded into `active_workers`, once every scaler
    /// for every rung has bumped its own count. From here a zero means what
    /// the finalizer thinks it means. A rung whose scalers all finished during
    /// setup is why the notify is here too — without it that rung's finalizer
    /// would wait forever on a wake-up that already happened.
    pub fn release_setup_guard(&self) {
        for (idx, active) in self.active_workers.iter().enumerate() {
            if active.fetch_sub(1, Ordering::AcqRel) == 1 {
                self.rung_done[idx].notify_one();
            }
        }
    }

    fn worker_done_with(&self, rung_idx: usize) {
        if self.active_workers[rung_idx].fetch_sub(1, Ordering::AcqRel) == 1 {
            self.rung_done[rung_idx].notify_one();
        }
    }
}

/// The decode ranges for this job under the spec's [`DecodeSplit`].
///
/// Only an un-spliced, untrimmed single input is split: a range is addressed
/// by demuxed sample index and its numbering assumes the source starts at
/// chunk 0, neither of which survives a trim window or a concat. Those decode
/// whole, as they always did — and so does anything `plan_decode_ranges`
/// cannot cut safely.
pub(super) fn plan_ranges(params: &MultiGpuParams<'_>, shape: LadderShape, capacity: usize) -> Vec<DecodeRange> {
    let want = params.decode_split.ranges_for(capacity);
    if want > 1 && params.spliced_clips.is_empty() {
        if let Some(ranges) = crate::decode_pump::plan_decode_ranges(
            &params.input,
            &params.header.codec,
            shape.frames_per_chunk,
            want,
        ) {
            return ranges;
        }
    }
    vec![DecodeRange::whole_source()]
}

/// One pump per range, each on its own decode-capable card, each fanning out
/// to one channel per rung. Returns the pump tasks and
/// `receivers[range][rung]`.
///
/// With a single range the pump follows the decode policy (an explicit pin,
/// else the first decode-capable policy GPU): it feeds rungs whose encoders
/// sit on different cards, so there is no "right" one, and decoded frames land
/// in system memory anyway — a cross-adapter handoff is a memcpy. With several
/// ranges the choice does matter, because the point is to have the cards
/// decoding different stretches of the source at the same time — and it has
/// to be a card that can decode this codec, which the policy's list does not
/// promise (see `decode_capable_gpus`).
pub(super) fn spawn_pumps(
    params: &MultiGpuParams<'_>,
    ranges: &[DecodeRange],
    n_rungs: usize,
) -> (JoinSet<Result<u64>>, Vec<Vec<Option<mpsc::Receiver<VideoFrame>>>>) {
    let multi_range = ranges.len() > 1;
    let decode_gpus = params.decode_capable_gpus();
    let mut pump_tasks: JoinSet<Result<u64>> = JoinSet::new();
    let mut receivers: Vec<Vec<Option<mpsc::Receiver<VideoFrame>>>> = Vec::with_capacity(ranges.len());

    for (range_idx, range) in ranges.iter().enumerate() {
        let mut senders = Vec::with_capacity(n_rungs);
        let mut rxs = Vec::with_capacity(n_rungs);
        for _ in 0..n_rungs {
            let (tx, rx) = mpsc::channel(FANOUT_CHANNEL_CAPACITY);
            senders.push(tx);
            rxs.push(Some(rx));
        }
        receivers.push(rxs);

        let mut clips = params.clip_sources_for(params.range_decode_gpu_for(range_idx, &decode_gpus));
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
            rungs = n_rungs,
            ranges = ranges.len(),
            boundaries = ?ranges.iter().map(|r| r.start_sample).collect::<Vec<_>>(),
            "range-parallel decode engaged — every card decodes its own stretch of the source",
        );
    } else {
        tracing::info!(rungs = n_rungs, "shared decode pump engaged (one decode for the whole ladder)");
    }
    (pump_tasks, receivers)
}

/// One scaler per (range × rung), numbering chunks from the range's first
/// frame so a rung's chunks stay contiguous however the source was split.
///
/// A rung's queue is fed by every range's scaler and closed by whichever
/// finishes last — closing on the first exit would drain the workers while
/// other ranges were still feeding, losing every chunk after the first
/// range's end. Only the last range's scaler may mark a chunk final: a middle
/// range also finishes on a short chunk — its boundary — and marking that
/// final would end the stream mid-video.
pub(super) fn spawn_scalers<T: Send + 'static>(
    rungs: &[Rung],
    ranges: &[DecodeRange],
    shape: LadderShape,
    mut receivers: Vec<Vec<Option<mpsc::Receiver<VideoFrame>>>>,
    ladder: &Ladder<T>,
) -> JoinSet<(usize, Result<usize>)> {
    let mut scaler_tasks: JoinSet<(usize, Result<usize>)> = JoinSet::new();
    let rung_producers: Vec<Arc<AtomicUsize>> =
        (0..rungs.len()).map(|_| Arc::new(AtomicUsize::new(ranges.len()))).collect();
    let last_range_idx = ranges.len() - 1;
    for (range_idx, range) in ranges.iter().enumerate() {
        // `plan_decode_ranges` guarantees the boundary is a multiple of
        // `frames_per_chunk`, so this division is exact.
        let first_segment_idx = (range.start_frame / u64::from(shape.frames_per_chunk)) as usize;
        for (idx, rung) in rungs.iter().enumerate() {
            let rx = receivers[range_idx][idx].take().expect("scaler rx slot");
            let cfg = crate::rung_scaler::RungScalerConfig {
                rung_idx: idx,
                target_width: rung.width,
                target_height: rung.height,
                frames_per_chunk: shape.frames_per_chunk,
                overlap: shape.overlap,
                first_segment_idx,
                is_final_range: range_idx == last_range_idx,
            };
            let queue = Arc::clone(&ladder.queues[idx]);
            let rt = tokio::runtime::Handle::current();
            let active_h = Arc::clone(&ladder.active_workers);
            let rung_done_h = Arc::clone(&ladder.rung_done);
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
    scaler_tasks
}

/// The per-rung worker config for one card: the rung's own knobs, the job's
/// output format, and this worker's lease. `output_dir` is the rung's
/// directory under the output root, keyed by label; the single-file path never
/// writes there.
fn rung_worker_config(
    ctx: &WorkerCtx,
    rung_idx: usize,
    rung: &Rung,
    lease: &GpuLease,
    rung_invariant: Arc<RwLock<Option<RungCodecInvariant>>>,
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

/// What a worker does with one chunk of one rung.
///
/// Called with that rung's config, the chunk, the worker's `init_written`
/// flag for the rung (only the CMAF path reads it), the rung's shared frame
/// and byte counters, and the progress channel. `Done(T)` is recorded against
/// the rung; `Rejected` puts the chunk back and strikes the rung off this
/// worker's list.
pub(super) trait EncodeUnit<T>: Send + Sync + 'static {
    fn encode(
        &self,
        cfg: &EncoderWorkerConfig,
        chunk: SegmentChunk,
        init_written: &mut bool,
        frames_encoded: &AtomicU64,
        bytes_encoded: &AtomicU64,
        progress_tx: &mpsc::Sender<u64>,
    ) -> Result<UnitOutcome<T>>;
}

impl<T, F> EncodeUnit<T> for F
where
    F: Fn(&EncoderWorkerConfig, SegmentChunk, &mut bool, &AtomicU64, &AtomicU64, &mpsc::Sender<u64>) -> Result<UnitOutcome<T>>
        + Send
        + Sync
        + 'static,
{
    fn encode(
        &self,
        cfg: &EncoderWorkerConfig,
        chunk: SegmentChunk,
        init_written: &mut bool,
        frames_encoded: &AtomicU64,
        bytes_encoded: &AtomicU64,
        progress_tx: &mpsc::Sender<u64>,
    ) -> Result<UnitOutcome<T>> {
        self(cfg, chunk, init_written, frames_encoded, bytes_encoded, progress_tx)
    }
}

/// Claim a lease per GPU and start the workers. Returns the worker tasks and
/// how many started. Fails only when the pool hands out nothing at all.
pub(super) async fn spawn_workers<T: Send + 'static>(
    params: &MultiGpuParams<'_>,
    ctx: &WorkerCtx,
    rungs: &[Rung],
    ladder: &Arc<Ladder<T>>,
    encode: Arc<dyn EncodeUnit<T>>,
) -> Result<(JoinSet<(usize, Result<()>)>, usize)> {
    let capacity = params.gpu_pool.capacity().max(1);
    let mut worker_tasks: JoinSet<(usize, Result<()>)> = JoinSet::new();
    let mut leases = Vec::with_capacity(capacity);
    for slot in 0..capacity {
        match Arc::clone(&params.gpu_pool).claim().await {
            Some(l) => leases.push(l),
            None if slot == 0 => {
                bail!("multigpu: GPU pool returned no lease on a CPU-only host; at least one GPU is required");
            }
            None => break,
        }
    }
    let workers = leases.len();
    for (slot, lease) in leases.into_iter().enumerate() {
        // Which rungs this worker may take from.
        let serves: Vec<usize> = match params.schedule {
            RungSchedule::Ladder => (0..rungs.len()).collect(),
            RungSchedule::PerRung => (0..rungs.len()).filter(|idx| idx % workers == slot).collect(),
        };
        spawn_ladder_worker(ctx, slot, rungs, serves, lease, Arc::clone(ladder), Arc::clone(&encode), &mut worker_tasks);
    }
    tracing::info!(
        ladder_workers = workers,
        rungs = rungs.len(),
        schedule = ?params.schedule,
        "ladder workers started — each serves every rung it is scheduled for, so a card idles only when that work is done",
    );
    Ok((worker_tasks, workers))
}

/// One worker, every rung it serves.
///
/// Holds a single GPU lease for its lifetime — so the one-encoder-per-GPU
/// invariant is untouched — and repeatedly takes the next chunk from
/// whichever of its rungs is furthest behind.
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
/// Only when every queue it serves is closed *and* empty. A worker that finds
/// nothing waits a beat and asks again rather than exiting, because "this rung
/// has nothing right now" is the normal state of a rung whose scaler is
/// mid-chunk; exiting on it would retire a card with work still coming.
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
fn spawn_ladder_worker<T: Send + 'static>(
    ctx: &WorkerCtx,
    slot: usize,
    rungs: &[Rung],
    serves: Vec<usize>,
    lease: GpuLease,
    ladder: Arc<Ladder<T>>,
    encode: Arc<dyn EncodeUnit<T>>,
    worker_tasks: &mut JoinSet<(usize, Result<()>)>,
) {
    let gpu_index = lease.gpu_index;
    let gpu_vendor = lease.vendor;
    let configs: Vec<EncoderWorkerConfig> = rungs
        .iter()
        .enumerate()
        .map(|(idx, rung)| rung_worker_config(ctx, idx, rung, &lease, Arc::clone(&ladder.rung_invariants[idx])))
        .collect();

    let body = async move {
        // The per-frame progress channel is a formality here: the shared
        // counters are what the reporter reads. It exists so the worker is
        // never backpressured by nobody listening.
        let (progress_tx, mut progress_rx) = mpsc::channel::<u64>(32);
        let drain = tokio::spawn(async move { while progress_rx.recv().await.is_some() {} });

        let ladder_for_worker = Arc::clone(&ladder);
        let blocking = tokio::task::spawn_blocking(move || -> Result<()> {
            let ladder = ladder_for_worker;
            let mut init_written: Vec<bool> = vec![false; configs.len()];
            let mut refused: HashSet<usize> = HashSet::new();
            loop {
                // Pick the rung closest to blocking the pump.
                let mut best: Option<(usize, usize)> = None;
                for &idx in &serves {
                    if refused.contains(&idx) {
                        continue;
                    }
                    let depth = ladder.queues[idx].depth();
                    if depth == 0 {
                        continue;
                    }
                    if best.is_none_or(|(_, d)| depth > d) {
                        best = Some((idx, depth));
                    }
                }

                let Some((rung_idx, _)) = best else {
                    // Nothing anywhere. Finished only if nothing can arrive.
                    if serves.iter().all(|&idx| ladder.queues[idx].is_closed() && ladder.queues[idx].depth() == 0) {
                        break;
                    }
                    std::thread::sleep(IDLE_POLL);
                    continue;
                };

                let Some(chunk) = ladder.queues[rung_idx].try_pop() else {
                    // Another worker took it between the look and the grab.
                    continue;
                };

                // Held across the encode so this rung's finalizer cannot decide
                // the rung is finished while a chunk of it is still in a card.
                ladder.active_workers[rung_idx].fetch_add(1, Ordering::AcqRel);
                let segment_idx = chunk.segment_idx;
                let outcome = encode.encode(
                    &configs[rung_idx],
                    chunk,
                    &mut init_written[rung_idx],
                    &ladder.frames_encoded[rung_idx],
                    &ladder.bytes_encoded[rung_idx],
                    &progress_tx,
                );
                match outcome {
                    Ok(UnitOutcome::Done(contribution)) => {
                        // Which card did which chunk of which rung — the line
                        // that answers "what is actually happening" on a fleet.
                        tracing::info!(rung_idx, gpu_index, chunk = segment_idx, "rung chunk done");
                        // Recorded before the count drops, so the finalizer
                        // woken by that drop sees this contribution.
                        ladder.contributions[rung_idx]
                            .lock()
                            .unwrap_or_else(|p| p.into_inner())
                            .push(contribution);
                        ladder.worker_done_with(rung_idx);
                    }
                    Ok(UnitOutcome::Rejected { chunk, diff }) => {
                        tracing::warn!(
                            rung_idx,
                            gpu_index,
                            gpu_vendor = ?gpu_vendor,
                            rejected_chunk = chunk.segment_idx,
                            diff = %diff,
                            "codec invariant mismatch — returning the chunk for another card \
                             and leaving this rung to them",
                        );
                        ladder.queues[rung_idx].push_front(chunk);
                        refused.insert(rung_idx);
                        ladder.worker_done_with(rung_idx);
                    }
                    Err(e) => {
                        ladder.worker_done_with(rung_idx);
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

/// The tasks a run waits on, and how many are still running.
pub(super) struct Running<R> {
    pub pumps: JoinSet<Result<u64>>,
    pub scalers: JoinSet<(usize, Result<usize>)>,
    pub workers: JoinSet<(usize, Result<()>)>,
    pub finalizer_rx: mpsc::Receiver<(usize, Result<Option<R>>)>,
    pub finalizers_remaining: usize,
}

/// Wait for every pump, scaler, worker and finalizer; the first error wins.
/// The caller stops its progress reporter (and awaits the finalizer handles on
/// success) around this.
pub(super) async fn drain<R>(mut run: Running<R>) -> Result<Vec<Option<R>>> {
    let mut completed: Vec<Option<R>> = (0..run.finalizers_remaining).map(|_| None).collect();
    let mut pumps_remaining = run.pumps.len();
    let mut scalers_remaining = run.scalers.len();
    let mut workers_remaining = run.workers.len();
    let mut finalizers_remaining = run.finalizers_remaining;

    while pumps_remaining > 0 || scalers_remaining > 0 || workers_remaining > 0 || finalizers_remaining > 0 {
        tokio::select! {
            biased;
            p = run.pumps.join_next(), if pumps_remaining > 0 => match p {
                Some(Ok(Ok(frames))) => { pumps_remaining -= 1; tracing::info!(frames, pumps_remaining, "decode pump finished"); }
                Some(Ok(Err(e))) => return Err(anyhow!("decode pump failed: {e:#}")),
                Some(Err(je)) => return Err(anyhow!("pump join error: {je}")),
                None => pumps_remaining = 0,
            },
            s = run.scalers.join_next(), if scalers_remaining > 0 => match s {
                Some(Ok((idx, Ok(chunks)))) => { tracing::debug!(idx, chunks, "scaler finished"); scalers_remaining -= 1; }
                Some(Ok((idx, Err(e)))) => return Err(anyhow!("scaler {idx} failed: {e:#}")),
                Some(Err(je)) => return Err(anyhow!("scaler join error: {je}")),
                None => scalers_remaining = 0,
            },
            w = run.workers.join_next(), if workers_remaining > 0 => match w {
                Some(Ok((slot, Ok(())))) => { tracing::debug!(slot, "ladder worker finished"); workers_remaining -= 1; }
                Some(Ok((slot, Err(e)))) => return Err(anyhow!("ladder worker {slot} failed: {e:#}")),
                Some(Err(je)) => return Err(anyhow!("worker join error: {je}")),
                None => workers_remaining = 0,
            },
            f = run.finalizer_rx.recv(), if finalizers_remaining > 0 => match f {
                Some((idx, Ok(opt))) => { completed[idx] = opt; finalizers_remaining -= 1; }
                Some((idx, Err(e))) => return Err(anyhow!("finalizer for rung {idx} failed: {e:#}")),
                None => finalizers_remaining = 0,
            },
        }
    }
    Ok(completed)
}
