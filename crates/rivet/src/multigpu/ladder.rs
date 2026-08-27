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
//! furthest behind ([`EncodePolicy::AllGpus`](crate::spec::EncodePolicy::AllGpus)). A per-rung worker idled the
//! moment its rung was blocked even with another rung's chunks sitting ready;
//! it also capped the rungs in flight at the GPU count, so a longer ladder fell
//! back to decoding the source once per rung — and decode is the dominant cost
//! of a transcode. Because no rung can now be left without a consumer, the pump
//! is always shared and the ladder costs exactly one decode however many rungs
//! it has. [`EncodePolicy::PerRung`](crate::spec::EncodePolicy::PerRung) keeps the pinned shape available for
//! comparison and for hosts where placement matters more than throughput.
//!
//! **The decode is split across the cards.** One decoder for the whole ladder
//! is one decoder, and the giveaway that it is the limiter is rungs of very
//! different encode cost advancing in lockstep on the same segment number.
//! [`plan_decode_ranges`](crate::decode_pump::plan_decode_ranges) cuts the
//! source at keyframes that fall on chunk boundaries; one pump per range,
//! pinned to its own card, feeds every rung's scaler, and the numbering stays
//! continuous across the join ([`DecodePolicy`](crate::spec::DecodePolicy)). A source that cannot be split
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
use tokio::sync::{Notify, mpsc, watch};
use tokio::task::JoinSet;

use codec::frame::VideoFrame;

use crate::decode_pump::DecodeRange;
use crate::encoder_worker::{EncoderSessionPool, EncoderWorkerConfig, RungCodecInvariant};
use crate::frame_queue::{SegmentChunk, SegmentChunkQueue};
use crate::gpu_pool::GpuLease;
use crate::spec::Rung;

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
    /// The stop signal for everything on this ladder — see [`AbortSignal`].
    pub abort: Arc<AbortSignal>,
}

/// How a run is stopped before it is finished — by a caller's cancel, or by
/// the first failure — without leaving anything behind.
///
/// Every part of the ladder that can block does so on one of two things: a
/// queue (scalers push, workers pop) or the `rung_done` notify (finalizers).
/// So stopping is: raise the flag, close and empty every queue, wake every
/// finalizer. A scaler mid-push gets `false` and returns, which drops its
/// frame receiver, which is what ends its pump. A worker sees the flag at the
/// top of its loop and returns its lease. A finalizer wakes, sees the flag and
/// returns without merging. Nothing waits on a wake-up that will not come, and
/// the queued frames — up to the whole byte budget — go with the run instead
/// of living on in a task nobody is joining.
///
/// This is not generic over the contribution type on purpose: the thing that
/// waits on the run ([`drain`]) knows the finalizer's output type but not the
/// worker's, and it is the one that has to be able to pull the plug.
pub(super) struct AbortSignal {
    flag: AtomicBool,
    queues: Vec<Arc<SegmentChunkQueue>>,
    rung_done: Arc<Vec<Notify>>,
}

impl AbortSignal {
    /// Whether the run has been told to stop.
    pub fn is_aborted(&self) -> bool {
        self.flag.load(Ordering::Acquire)
    }

    /// Stop the run: flag, close and empty every queue, wake every finalizer.
    /// Idempotent.
    pub fn abort(&self) {
        self.flag.store(true, Ordering::Release);
        for q in &self.queues {
            q.close();
            while q.try_pop().is_some() {}
        }
        for n in self.rung_done.iter() {
            n.notify_waiters();
            n.notify_one();
        }
    }
}

impl<T: Send + 'static> Ladder<T> {
    /// Per-rung state for `rungs`, with each queue's depth derived from the
    /// byte budget rather than a fixed count (see `queue_capacity_for`).
    pub fn new(rungs: &[Rung], frames_per_chunk: u32) -> Self {
        let n = rungs.len();
        let queues: Vec<Arc<SegmentChunkQueue>> = rungs
            .iter()
            .map(|r| {
                let depth = queue_capacity_for(r.width, r.height, frames_per_chunk, n);
                Arc::new(SegmentChunkQueue::new(depth))
            })
            .collect();
        let rung_done: Arc<Vec<Notify>> = Arc::new((0..n).map(|_| Notify::new()).collect());
        Self {
            abort: Arc::new(AbortSignal {
                flag: AtomicBool::new(false),
                queues: queues.clone(),
                rung_done: Arc::clone(&rung_done),
            }),
            queues,
            frames_encoded: (0..n).map(|_| Arc::new(AtomicU64::new(0))).collect(),
            bytes_encoded: (0..n).map(|_| Arc::new(AtomicU64::new(0))).collect(),
            rung_invariants: (0..n).map(|_| Arc::new(RwLock::new(None))).collect(),
            contributions: Arc::new((0..n).map(|_| Mutex::new(Vec::new())).collect()),
            active_workers: Arc::new((0..n).map(|_| AtomicUsize::new(1)).collect()),
            rung_done,
            finalized: Arc::new((0..n).map(|_| AtomicBool::new(false)).collect()),
        }
    }

    /// Whether the run has been stopped early (cancelled, or failed elsewhere).
    /// A finalizer that wakes to this returns without merging.
    pub fn is_aborted(&self) -> bool {
        self.abort.is_aborted()
    }

    /// Wait until rung `idx` is finished: nothing is working on it *and*
    /// nothing can be handed out — queue closed, queue empty. Also returns,
    /// early, when the run is aborted; check [`Self::is_aborted`] after.
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
            if self.is_aborted() {
                return;
            }
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

/// Pre-flight: can this host construct an encoder for the job's codec at all?
///
/// A pool of cards answers by building one (unpinned, so the chain runs as a
/// worker's would): fail fast with a clear error rather than after the
/// orchestration is up — and, on drivers that re-init a failed NVENC session
/// badly, rather than by hanging an uncancellable task. A software pool
/// already *is* the answer: the pool's builder handed out software slots
/// because the build has a software encoder for this codec, and constructing
/// one just to ask spins up a worker pool sized to the whole machine (32
/// threads on this box) to encode nothing. So software is checked from the
/// feature flags, and the encoder is built once per unit of work, as it
/// would be anyway.
pub(super) fn preflight_encoder(params: &MultiGpuParams<'_>, width: u32, height: u32) -> Result<()> {
    if params.gpu_pool.is_software() {
        if !codec::encode::software_encode_available(params.codec) {
            bail!(
                "the encode pool is software but this build has no software {:?} encoder \
                 (rebuild with `--features {}`)",
                params.codec,
                codec::encode::software_feature_for(params.codec)
            );
        }
        return Ok(());
    }
    let probe = codec::encode::EncoderConfig {
        width,
        height,
        frame_rate: params.frame_rate,
        gpu_index: None,
        codec: params.codec,
        ..Default::default()
    };
    codec::encode::select_encoder(probe, None).map_err(|e| {
        anyhow!(
            "no {:?} encoder available on this host ({e}); need NVENC / AMF / QSV, or build \
             with `rav1e-fallback` (software AV1) / `h26x-fallback` (software H.264 / H.265)",
            params.codec
        )
    })?;
    Ok(())
}

/// The decode ranges for this job under the spec's [`DecodePolicy`](crate::spec::DecodePolicy).
///
/// Only an un-spliced, untrimmed single input is split: a range is addressed
/// by demuxed sample index and its numbering assumes the source starts at
/// chunk 0, neither of which survives a trim window or a concat. Those decode
/// whole, as they always did — and so does anything `plan_decode_ranges`
/// cannot cut safely.
pub(super) fn plan_ranges(params: &MultiGpuParams<'_>, shape: LadderShape, capacity: usize) -> Vec<DecodeRange> {
    // `Auto` means one range per card. Software slots are not cards: they
    // share the cores a split decode would also run on, and the software
    // encoders are far slower than the software decoder, so splitting the
    // decode buys nothing and costs a decoder instance per range. One range,
    // unless the policy names a count (`ranges:N`) outright.
    let cards = if params.gpu_pool.is_software() { 1 } else { capacity };
    let want = params.decode.ranges_for(cards);
    // A temporal filter (hqdn3d) makes each frame depend on the ones before
    // it, and a range starts with no history: split, the frames at every
    // range start would differ from a whole decode. One stream, one pump.
    if want > 1 && params.filters.is_stateful() {
        tracing::info!(
            "decode ranges: the filter chain is temporal (frame history); decoding whole rather than in {want} ranges"
        );
        return vec![DecodeRange::whole_source()];
    }
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
        // A software lease is a share of the CPU: its thread budget goes to
        // the encoder, and the encoder is asked for by name so a chunk never
        // re-runs the hardware probes the pool already ran. A card leaves
        // `threads` at 0 (the encoder does not run on host threads) and
        // `backend` unset (the vendor pin steers the chain).
        threads: lease.threads(),
        gpu_index: lease.gpu_index(),
        gpu_vendor: lease.vendor(),
        backend: if lease.is_software() { codec::encode::software_backend_for(ctx.codec) } else { None },
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
/// flag for the rung (only the CMAF path reads it), the worker's encoder
/// session pool (only the single-file path uses it — one pool per worker,
/// kept across every chunk and every rung it serves), the rung's shared
/// frame and byte counters, and the progress channel. `Done(T)` is recorded
/// against the rung; `Rejected` puts the chunk back and strikes the rung off
/// this worker's list.
pub(super) trait EncodeUnit<T>: Send + Sync + 'static {
    #[allow(clippy::too_many_arguments)]
    fn encode(
        &self,
        cfg: &EncoderWorkerConfig,
        chunk: SegmentChunk,
        init_written: &mut bool,
        sessions: &mut EncoderSessionPool,
        frames_encoded: &AtomicU64,
        bytes_encoded: &AtomicU64,
        progress_tx: &mpsc::Sender<u64>,
    ) -> Result<UnitOutcome<T>>;
}

impl<T, F> EncodeUnit<T> for F
where
    F: Fn(
            &EncoderWorkerConfig,
            SegmentChunk,
            &mut bool,
            &mut EncoderSessionPool,
            &AtomicU64,
            &AtomicU64,
            &mpsc::Sender<u64>,
        ) -> Result<UnitOutcome<T>>
        + Send
        + Sync
        + 'static,
{
    fn encode(
        &self,
        cfg: &EncoderWorkerConfig,
        chunk: SegmentChunk,
        init_written: &mut bool,
        sessions: &mut EncoderSessionPool,
        frames_encoded: &AtomicU64,
        bytes_encoded: &AtomicU64,
        progress_tx: &mpsc::Sender<u64>,
    ) -> Result<UnitOutcome<T>> {
        self(cfg, chunk, init_written, sessions, frames_encoded, bytes_encoded, progress_tx)
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
                // The pool is empty, and the pool's builder already decided
                // that software was not an answer here — say why, by name.
                bail!(
                    "multigpu: the encode pool has nothing to lease: {}",
                    super::gpu_policy::empty_pool_reason(
                        params.encode,
                        ctx.codec,
                        codec::encode::software_encode_available(ctx.codec),
                    )
                );
            }
            None => break,
        }
    }
    let workers = leases.len();
    let software = leases.iter().filter(|l| l.is_software()).count();
    for (slot, lease) in leases.into_iter().enumerate() {
        // Which rungs this worker may take from.
        let serves: Vec<usize> = if params.encode.pins_rungs() {
            (0..rungs.len()).filter(|idx| idx % workers == slot).collect()
        } else {
            (0..rungs.len()).collect()
        };
        spawn_ladder_worker(ctx, slot, rungs, serves, lease, Arc::clone(ladder), Arc::clone(&encode), &mut worker_tasks);
    }
    if software > 0 {
        tracing::info!(
            ladder_workers = workers,
            software_leases = software,
            threads_per_lease = ?params.gpu_pool.software_threads(),
            rungs = rungs.len(),
            encode = ?params.encode,
            backend = ?codec::encode::software_backend_for(ctx.codec),
            "ladder workers started on SOFTWARE leases — no GPU can encode this codec in this build; \
             each worker runs one software encoder at a time on its thread share",
        );
    } else {
        tracing::info!(
            ladder_workers = workers,
            rungs = rungs.len(),
            encode = ?params.encode,
            "ladder workers started — each serves every rung it is scheduled for, so a card idles only when that work is done",
        );
    }
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
    let gpu_index = lease.gpu_index();
    let gpu_vendor = lease.vendor();
    // "gpu 0 (Nvidia)" or "software slot 3 (4 threads)": the log lines below
    // are the answer to "what is actually running this chunk", and on a
    // CPU-only host `gpu_index=None` alone would not say.
    let lease_label = lease.kind().to_string();
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
        let lease_label_after = lease_label.clone();
        let blocking = tokio::task::spawn_blocking(move || -> Result<()> {
            let ladder = ladder_for_worker;
            let mut init_written: Vec<bool> = vec![false; configs.len()];
            let mut refused: HashSet<usize> = HashSet::new();
            // This worker's encoder session, kept between chunks and reset
            // rather than rebuilt while consecutive chunks share a rung. One
            // per worker, so one live session per lease — the
            // one-encoder-per-GPU invariant is exactly as true as before.
            let mut sessions = EncoderSessionPool::new();
            loop {
                // Stopped from outside — cancelled, or another part of the run
                // failed. Return the lease now rather than after draining what
                // is left in the queues.
                if ladder.is_aborted() {
                    tracing::info!(slot, gpu_index = ?gpu_index, lease = %lease_label, "ladder worker stopping: run aborted");
                    break;
                }
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
                    &mut sessions,
                    &ladder.frames_encoded[rung_idx],
                    &ladder.bytes_encoded[rung_idx],
                    &progress_tx,
                );
                match outcome {
                    Ok(UnitOutcome::Done(contribution)) => {
                        // Which card did which chunk of which rung — the line
                        // that answers "what is actually happening" on a fleet.
                        tracing::info!(rung_idx, gpu_index = ?gpu_index, lease = %lease_label, chunk = segment_idx, "rung chunk done");
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
                            gpu_index = ?gpu_index,
                            lease = %lease_label,
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
            // The evidence for the session pool: on a run that used to build
            // one encoder per chunk, `built + reused` is the chunk count and
            // `reused` is the saving.
            let stats = sessions.stats();
            tracing::info!(
                slot,
                gpu_index,
                built = stats.built,
                reused = stats.reused,
                evicted = stats.evicted,
                reset_unsupported = stats.reset_unsupported,
                reset_failed = stats.reset_failed,
                "ladder worker encoder sessions: built vs reused"
            );
            Ok(())
        });

        let status: Result<()> = match blocking.await {
            Ok(Ok(())) => {
                tracing::info!(slot, gpu_index = ?gpu_index, lease = %lease_label_after, "ladder worker exited cleanly");
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
    /// The ladder's stop signal, pulled on cancel and on the first failure.
    pub abort: Arc<AbortSignal>,
    /// The caller's cancel signal, if it has one: `true` means stop.
    pub cancel: Option<watch::Receiver<bool>>,
}

/// Wait for every pump, scaler, worker and finalizer; the first error wins.
/// The caller stops its progress reporter (and awaits the finalizer handles on
/// success) around this.
///
/// # Stopping early
///
/// On the first failure, or when the caller's `cancel` signal turns true, the
/// run is [aborted](AbortSignal::abort) and this waits for the **workers** to
/// return before returning the error — they hold the GPU leases, and the next
/// job's `claim()` must not find them still held by a run that is over. That
/// wait is bounded by one unit of work: a worker checks the flag between
/// units, not inside one. Pumps and scalers stop on their own once the queues
/// are closed and are not waited for; they hold no leases and drop what they
/// were carrying as they go.
///
/// A cancel comes back as [`Cancelled`](super::Cancelled), so a caller can
/// tell "asked to stop" from "failed" without reading the message.
pub(super) async fn drain<R>(mut run: Running<R>) -> Result<Vec<Option<R>>> {
    let mut completed: Vec<Option<R>> = (0..run.finalizers_remaining).map(|_| None).collect();
    let mut pumps_remaining = run.pumps.len();
    let mut scalers_remaining = run.scalers.len();
    let mut workers_remaining = run.workers.len();
    let mut finalizers_remaining = run.finalizers_remaining;

    // A cancel signal that is already raised, or absent, is handled here so the
    // select below only has to watch for a change.
    let mut cancel = run.cancel.take();
    if cancel.as_ref().is_some_and(|c| *c.borrow()) {
        return Err(stop(&mut run, super::Cancelled.into()).await);
    }

    while pumps_remaining > 0 || scalers_remaining > 0 || workers_remaining > 0 || finalizers_remaining > 0 {
        let outcome: Result<()> = tokio::select! {
            biased;
            changed = watch_cancel(&mut cancel) => match changed {
                // The sender is gone: nobody can cancel us any more.
                Err(()) => { cancel = None; Ok(()) }
                Ok(()) => Err(super::Cancelled.into()),
            },
            p = run.pumps.join_next(), if pumps_remaining > 0 => match p {
                Some(Ok(Ok(frames))) => { pumps_remaining -= 1; tracing::info!(frames, pumps_remaining, "decode pump finished"); Ok(()) }
                Some(Ok(Err(e))) => Err(anyhow!("decode pump failed: {e:#}")),
                Some(Err(je)) => Err(anyhow!("pump join error: {je}")),
                None => { pumps_remaining = 0; Ok(()) }
            },
            s = run.scalers.join_next(), if scalers_remaining > 0 => match s {
                Some(Ok((idx, Ok(chunks)))) => { tracing::debug!(idx, chunks, "scaler finished"); scalers_remaining -= 1; Ok(()) }
                Some(Ok((idx, Err(e)))) => Err(anyhow!("scaler {idx} failed: {e:#}")),
                Some(Err(je)) => Err(anyhow!("scaler join error: {je}")),
                None => { scalers_remaining = 0; Ok(()) }
            },
            w = run.workers.join_next(), if workers_remaining > 0 => match w {
                Some(Ok((slot, Ok(())))) => { tracing::debug!(slot, "ladder worker finished"); workers_remaining -= 1; Ok(()) }
                Some(Ok((slot, Err(e)))) => Err(anyhow!("ladder worker {slot} failed: {e:#}")),
                Some(Err(je)) => Err(anyhow!("worker join error: {je}")),
                None => { workers_remaining = 0; Ok(()) }
            },
            f = run.finalizer_rx.recv(), if finalizers_remaining > 0 => match f {
                Some((idx, Ok(opt))) => { completed[idx] = opt; finalizers_remaining -= 1; Ok(()) }
                Some((idx, Err(e))) => Err(anyhow!("finalizer for rung {idx} failed: {e:#}")),
                None => { finalizers_remaining = 0; Ok(()) }
            },
        };
        if let Err(e) = outcome {
            return Err(stop(&mut run, e).await);
        }
    }
    Ok(completed)
}

/// Resolve when the cancel signal turns true; `Err(())` when its sender has
/// gone. Pending forever while there is no signal, so the select arm is
/// simply never taken.
async fn watch_cancel(cancel: &mut Option<watch::Receiver<bool>>) -> std::result::Result<(), ()> {
    match cancel {
        None => std::future::pending().await,
        Some(rx) => loop {
            if *rx.borrow_and_update() {
                return Ok(());
            }
            if rx.changed().await.is_err() {
                return Err(());
            }
        },
    }
}

/// Abort the run and wait for the workers — the lease holders — to return,
/// then hand back the error that ended it. The finalizer channel is left
/// alone: a finalizer that wakes to the abort sends nothing anyone reads.
async fn stop<R>(run: &mut Running<R>, why: anyhow::Error) -> anyhow::Error {
    if why.is::<super::Cancelled>() {
        tracing::info!("ladder run cancelled; stopping the workers");
    } else {
        tracing::warn!(error = %format!("{why:#}"), "ladder run failed; stopping the workers");
    }
    run.abort.abort();
    while let Some(joined) = run.workers.join_next().await {
        match joined {
            Ok((slot, Ok(()))) => tracing::debug!(slot, "ladder worker returned its lease"),
            Ok((slot, Err(e))) => {
                tracing::debug!(slot, error = %format!("{e:#}"), "ladder worker ended with an error while stopping")
            }
            Err(je) => tracing::debug!(%je, "ladder worker join error while stopping"),
        }
    }
    why
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;
    use codec::frame::{ColorSpace, PixelFormat};
    use std::time::Duration;

    fn frame(idx: u64) -> VideoFrame {
        let mut data = vec![idx as u8; 16 * 16];
        data.extend(vec![128u8; 8 * 8]);
        data.extend(vec![128u8; 8 * 8]);
        VideoFrame::new(Bytes::from(data), 16, 16, PixelFormat::Yuv420p, ColorSpace::Bt709, idx)
    }

    fn chunk(idx: usize) -> SegmentChunk {
        SegmentChunk { segment_idx: idx, frames: vec![frame(0), frame(1)], lead_in: 0, keep: 2, is_final: false }
    }

    fn two_rungs() -> Vec<Rung> {
        vec![Rung::new(64, 64), Rung::new(32, 32)]
    }

    /// Aborting closes and empties every queue and wakes every finalizer —
    /// nothing is left holding frames or waiting on a notify that will not
    /// come.
    #[tokio::test]
    async fn abort_closes_empties_and_wakes() {
        let ladder: Ladder<()> = Ladder::new(&two_rungs(), 2);
        assert!(ladder.queues[0].push(chunk(0)).await);
        assert!(ladder.queues[0].push(chunk(1)).await);
        assert!(ladder.queues[1].push(chunk(0)).await);
        assert_eq!(ladder.queues[0].depth(), 2);
        assert!(!ladder.is_aborted());

        // A finalizer parked on a rung nobody has finished (the setup guard is
        // still held, so `active_workers` is 1 and the queue is open).
        let ladder = Arc::new(ladder);
        let waiter = {
            let l = Arc::clone(&ladder);
            tokio::spawn(async move { l.wait_rung_finished(0).await })
        };
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert!(!waiter.is_finished(), "finalizer must wait while the rung is open");

        ladder.abort.abort();

        assert!(ladder.is_aborted());
        for q in &ladder.queues {
            assert!(q.is_closed());
            assert_eq!(q.depth(), 0, "abort must drop what was queued");
        }
        // A push after the abort is refused (the scaler's exit condition).
        assert!(!ladder.queues[0].push(chunk(2)).await);
        tokio::time::timeout(Duration::from_secs(2), waiter)
            .await
            .expect("finalizer must be woken by the abort")
            .unwrap();
    }

    /// A cancel signal that is already raised stops the run before it waits
    /// on anything, and the error's root cause is `Cancelled`.
    #[tokio::test]
    async fn drain_returns_cancelled_when_signal_is_already_raised() {
        let ladder: Ladder<()> = Ladder::new(&two_rungs(), 2);
        let (_tx, mut rx) = watch::channel(false);
        // Raise it through a sender we keep alive so the receiver sees `true`.
        let tx = _tx;
        tx.send(true).unwrap();
        rx.mark_unchanged();
        let (_ftx, finalizer_rx) = mpsc::channel::<(usize, Result<Option<()>>)>(2);
        let run = Running {
            pumps: JoinSet::new(),
            scalers: JoinSet::new(),
            workers: JoinSet::new(),
            finalizer_rx,
            finalizers_remaining: 2,
            abort: Arc::clone(&ladder.abort),
            cancel: Some(rx),
        };
        let err = drain(run).await.expect_err("must not complete");
        assert!(err.is::<super::super::Cancelled>(), "root cause must be Cancelled, got {err:#}");
        assert!(ladder.is_aborted(), "cancel must abort the ladder");
    }

    /// A cancel raised while the run is waiting stops it, and the workers
    /// still in flight are joined before the error comes back.
    #[tokio::test]
    async fn drain_stops_on_cancel_and_joins_workers() {
        let ladder: Ladder<()> = Ladder::new(&two_rungs(), 2);
        let (tx, rx) = watch::channel(false);
        let (_ftx, finalizer_rx) = mpsc::channel::<(usize, Result<Option<()>>)>(2);
        // A "worker" that only returns once the run has been aborted — the
        // shape of a real worker checking the flag at the top of its loop.
        let mut workers: JoinSet<(usize, Result<()>)> = JoinSet::new();
        let abort = Arc::clone(&ladder.abort);
        let worker_saw_abort = Arc::new(AtomicBool::new(false));
        let saw = Arc::clone(&worker_saw_abort);
        workers.spawn(async move {
            while !abort.is_aborted() {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
            saw.store(true, Ordering::Release);
            (0, Ok(()))
        });
        let run = Running {
            pumps: JoinSet::new(),
            scalers: JoinSet::new(),
            workers,
            finalizer_rx,
            finalizers_remaining: 2,
            abort: Arc::clone(&ladder.abort),
            cancel: Some(rx),
        };
        let handle = tokio::spawn(drain(run));
        tokio::time::sleep(Duration::from_millis(30)).await;
        assert!(!handle.is_finished(), "must be waiting on the finalizers");
        tx.send(true).unwrap();
        let err = tokio::time::timeout(Duration::from_secs(2), handle)
            .await
            .expect("cancel must end the run")
            .unwrap()
            .expect_err("must not complete");
        assert!(err.is::<super::super::Cancelled>(), "got {err:#}");
        assert!(worker_saw_abort.load(Ordering::Acquire), "the worker must have been joined after the abort");
    }

    /// The first failure aborts the ladder too, so the other parts of the run
    /// stop instead of blocking on queues nobody will drain.
    #[tokio::test]
    async fn drain_aborts_ladder_on_failure() {
        let ladder: Ladder<()> = Ladder::new(&two_rungs(), 2);
        assert!(ladder.queues[0].push(chunk(0)).await);
        let (_ftx, finalizer_rx) = mpsc::channel::<(usize, Result<Option<()>>)>(2);
        let mut scalers: JoinSet<(usize, Result<usize>)> = JoinSet::new();
        scalers.spawn(async { (0, Err(anyhow!("scaler exploded"))) });
        let run = Running {
            pumps: JoinSet::new(),
            scalers,
            workers: JoinSet::new(),
            finalizer_rx,
            finalizers_remaining: 2,
            abort: Arc::clone(&ladder.abort),
            cancel: None,
        };
        let err = drain(run).await.expect_err("must fail");
        assert!(!err.is::<super::super::Cancelled>());
        assert!(format!("{err:#}").contains("scaler exploded"));
        assert!(ladder.is_aborted());
        assert!(ladder.queues[0].is_closed());
        assert_eq!(ladder.queues[0].depth(), 0);
    }
}
