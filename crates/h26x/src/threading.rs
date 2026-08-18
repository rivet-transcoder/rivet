//! Frame-level threading: a worker pool that decodes whole pictures in
//! parallel, and the row-progress handshake that lets a picture read from a
//! reference that is still being decoded — libavcodec's frame-threading
//! model. Every picture is decoded top to bottom; it publishes how many
//! luma rows are final (deblocked, SAO'd, edge-extended) as it goes, and a
//! later picture that needs samples from it waits for exactly the rows its
//! motion vector reaches. Dependencies only ever point at pictures earlier
//! in decode order, which started earlier, so nothing can wait in a cycle.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

/// Row progress of a picture being decoded.
///
/// `decoded` — luma rows whose blocks are reconstructed (samples before the
/// loop filters, motion vectors final); `done` — luma rows that are final
/// and edge-extended, safe to motion-compensate from. Both are prefix
/// counts (everything above them is ready), never decreasing, and reach the
/// picture height (or [`Progress::COMPLETE`]) when the picture is finished
/// or abandoned.
pub struct Progress {
    decoded: AtomicI32,
    done: AtomicI32,
    lock: Mutex<()>,
    cv: Condvar,
    /// Set when decoding hit an error; the samples are whatever was written.
    pub error: AtomicBool,
}

impl Progress {
    /// A value larger than any picture height.
    pub const COMPLETE: i32 = i32::MAX / 2;

    /// Nothing done yet.
    pub fn new() -> Self {
        Progress { decoded: AtomicI32::new(0), done: AtomicI32::new(0), lock: Mutex::new(()), cv: Condvar::new(), error: AtomicBool::new(false) }
    }

    /// Already complete (a generated or synchronously decoded picture).
    pub fn complete() -> Self {
        let p = Self::new();
        p.decoded.store(Self::COMPLETE, Ordering::Release);
        p.done.store(Self::COMPLETE, Ordering::Release);
        p
    }

    /// Rows `< y` are reconstructed.
    pub fn set_decoded(&self, y: i32) {
        if self.decoded.load(Ordering::Relaxed) >= y {
            return;
        }
        let _g = self.lock.lock().unwrap();
        // Monotonic: two publishers may race (a decoding task and a filter
        // task announcing neighbouring rows); the larger value stays.
        self.decoded.fetch_max(y, Ordering::AcqRel);
        self.cv.notify_all();
    }

    /// Rows `< y` are final.
    pub fn set_done(&self, y: i32) {
        if self.done.load(Ordering::Relaxed) >= y {
            return;
        }
        let _g = self.lock.lock().unwrap();
        self.done.fetch_max(y, Ordering::AcqRel);
        self.cv.notify_all();
    }

    /// Everything is final (or abandoned after an error).
    pub fn finish(&self) {
        let _g = self.lock.lock().unwrap();
        self.decoded.store(Self::COMPLETE, Ordering::Release);
        self.done.store(Self::COMPLETE, Ordering::Release);
        self.cv.notify_all();
    }

    /// Current `done` count.
    pub fn done_rows(&self) -> i32 {
        self.done.load(Ordering::Acquire)
    }

    /// Whether the picture is finished.
    pub fn is_complete(&self) -> bool {
        self.done.load(Ordering::Acquire) >= Self::COMPLETE
    }

    /// Block until rows `< y` are final.
    pub fn wait_done(&self, y: i32) {
        if self.done.load(Ordering::Acquire) >= y {
            return;
        }
        let t = std::time::Instant::now();
        let mut g = self.lock.lock().unwrap();
        while self.done.load(Ordering::Acquire) < y {
            g = self.cv.wait(g).unwrap();
        }
        if prof::enabled() {
            prof::add(&prof::WAIT_REF, t);
        }
    }

    /// Block until rows `< y` are reconstructed.
    pub fn wait_decoded(&self, y: i32) {
        if self.decoded.load(Ordering::Acquire) >= y {
            return;
        }
        let t = std::time::Instant::now();
        let mut g = self.lock.lock().unwrap();
        while self.decoded.load(Ordering::Acquire) < y {
            g = self.cv.wait(g).unwrap();
        }
        if prof::enabled() {
            prof::add(&prof::WAIT_REF, t);
        }
    }

    /// Block until finished.
    pub fn wait_complete(&self) {
        self.wait_done(Self::COMPLETE);
    }
}

impl Default for Progress {
    fn default() -> Self {
        Self::new()
    }
}

type Job = Box<dyn FnOnce() + Send + 'static>;

struct PoolState {
    queue: VecDeque<Job>,
    /// Jobs queued or running.
    active: usize,
    shutdown: bool,
}

/// A FIFO pool of worker threads for the decoders' tasks.
///
/// Tasks block on one another (a CTB row waits for the row above, a picture
/// for rows of its references). Deadlock freedom comes from ordering, not
/// from the pool: every task is queued after every task it can wait on, and
/// the queue is strictly FIFO, so whatever a running task waits on is
/// already running or done. Blocked tasks do hold a worker, which is why
/// the decoders run more workers than hardware threads.
///
/// [`Pool::submit`] blocks while `capacity` jobs are outstanding (the
/// caller-side back-pressure); [`Pool::spawn`] never blocks (for workers).
pub struct Pool {
    state: Arc<(Mutex<PoolState>, Condvar, Condvar)>,
    workers: Vec<JoinHandle<()>>,
    capacity: usize,
}

impl Pool {
    /// `threads` workers; `capacity` jobs may be outstanding (`usize::MAX`
    /// for unbounded).
    pub fn new(threads: usize, capacity: usize) -> Arc<Self> {
        let state = Arc::new((Mutex::new(PoolState { queue: VecDeque::new(), active: 0, shutdown: false }), Condvar::new(), Condvar::new()));
        let mut workers = Vec::with_capacity(threads);
        for i in 0..threads.max(1) {
            let st = state.clone();
            let h = std::thread::Builder::new()
                .name(format!("h26x-worker-{i}"))
                .spawn(move || {
                    let (m, job_cv, done_cv) = &*st;
                    loop {
                        let job = {
                            let mut g = m.lock().unwrap();
                            loop {
                                if let Some(j) = g.queue.pop_front() {
                                    break Some(j);
                                }
                                if g.shutdown {
                                    break None;
                                }
                                g = job_cv.wait(g).unwrap();
                            }
                        };
                        let Some(job) = job else { return };
                        // A panicking job must not take the pool down.
                        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(job));
                        let mut g = m.lock().unwrap();
                        g.active -= 1;
                        done_cv.notify_all();
                    }
                })
                .expect("spawn h26x worker");
            workers.push(h);
        }
        Arc::new(Pool { state, workers, capacity: capacity.max(1) })
    }

    /// Number of worker threads.
    pub fn threads(&self) -> usize {
        self.workers.len()
    }

    /// Queue `job`, waiting while the pool is at capacity. Never call this
    /// from a worker of the same pool (it could wait for itself); workers use
    /// [`Pool::spawn`].
    pub fn submit(&self, job: Job) {
        let (m, job_cv, done_cv) = &*self.state;
        let mut g = m.lock().unwrap();
        while g.active >= self.capacity {
            g = done_cv.wait(g).unwrap();
        }
        g.active += 1;
        g.queue.push_back(job);
        job_cv.notify_one();
    }

    /// Queue `job` without waiting (from workers).
    pub fn spawn(&self, job: Job) {
        let (m, job_cv, _) = &*self.state;
        let mut g = m.lock().unwrap();
        g.active += 1;
        g.queue.push_back(job);
        job_cv.notify_one();
    }

    /// Wait until no job is queued or running.
    pub fn wait_idle(&self) {
        let (m, _, done_cv) = &*self.state;
        let mut g = m.lock().unwrap();
        while g.active > 0 {
            g = done_cv.wait(g).unwrap();
        }
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        {
            let (m, job_cv, _) = &*self.state;
            let mut g = m.lock().unwrap();
            g.shutdown = true;
            job_cv.notify_all();
        }
        for h in self.workers.drain(..) {
            let _ = h.join();
        }
    }
}

/// The default worker count: the machine's parallelism, capped. Workers
/// spend part of their time blocked on dependencies, so the decoders run
/// more of them than there are hardware threads.
pub fn default_threads() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1).clamp(1, 32)
}

/// Coarse per-process profiling counters (nanoseconds), printed on request
/// (`H26X_PROF=1`) when a decoder is dropped. Cheap enough to leave in.
pub mod prof {
    use std::sync::atomic::{AtomicU64, Ordering};
    /// Time decoding CTUs / macroblocks (parse + reconstruction).
    pub static DECODE: AtomicU64 = AtomicU64::new(0);
    /// Time in the in-loop filters.
    pub static FILTER: AtomicU64 = AtomicU64::new(0);
    /// Time waiting for neighbouring blocks of the same picture.
    pub static WAIT_NEIGHBOUR: AtomicU64 = AtomicU64::new(0);
    /// Time waiting for reference-picture progress.
    pub static WAIT_REF: AtomicU64 = AtomicU64::new(0);
    /// Time waiting for tasks (finisher).
    pub static WAIT_TASKS: AtomicU64 = AtomicU64::new(0);
    /// Time on the caller's thread inside push_nal.
    pub static MAIN: AtomicU64 = AtomicU64::new(0);
    /// Whether profiling is on (read once).
    pub fn enabled() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var_os("H26X_PROF").is_some())
    }
    /// Add elapsed nanoseconds since `t` to `c`.
    #[inline]
    pub fn add(c: &AtomicU64, t: std::time::Instant) {
        c.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
    /// Print the counters.
    pub fn report() {
        if !enabled() {
            return;
        }
        let ms = |c: &AtomicU64| c.load(Ordering::Relaxed) as f64 / 1e6;
        eprintln!(
            "h26x prof: decode {:.1} ms, filter {:.1} ms, wait-neighbour {:.1} ms, wait-ref {:.1} ms, wait-tasks {:.1} ms, main {:.1} ms",
            ms(&DECODE),
            ms(&FILTER),
            ms(&WAIT_NEIGHBOUR),
            ms(&WAIT_REF),
            ms(&WAIT_TASKS),
            ms(&MAIN)
        );
    }
}
