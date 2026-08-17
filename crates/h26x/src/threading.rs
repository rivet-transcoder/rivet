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
        self.decoded.store(y, Ordering::Release);
        self.cv.notify_all();
    }

    /// Rows `< y` are final.
    pub fn set_done(&self, y: i32) {
        if self.done.load(Ordering::Relaxed) >= y {
            return;
        }
        let _g = self.lock.lock().unwrap();
        self.done.store(y, Ordering::Release);
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
        let mut g = self.lock.lock().unwrap();
        while self.done.load(Ordering::Acquire) < y {
            g = self.cv.wait(g).unwrap();
        }
    }

    /// Block until rows `< y` are reconstructed.
    pub fn wait_decoded(&self, y: i32) {
        if self.decoded.load(Ordering::Acquire) >= y {
            return;
        }
        let mut g = self.lock.lock().unwrap();
        while self.decoded.load(Ordering::Acquire) < y {
            g = self.cv.wait(g).unwrap();
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
    active: usize,
    shutdown: bool,
}

/// A fixed set of worker threads with a bounded number of jobs in flight
/// (queued or running): [`Pool::submit`] blocks while `capacity` jobs are
/// outstanding, which is the back-pressure that keeps a decoder from
/// running arbitrarily far ahead of its consumer.
pub struct Pool {
    state: Arc<(Mutex<PoolState>, Condvar, Condvar)>,
    workers: Vec<JoinHandle<()>>,
    capacity: usize,
}

impl Pool {
    /// `threads` workers; `capacity` jobs may be outstanding.
    pub fn new(threads: usize, capacity: usize) -> Self {
        let state = Arc::new((Mutex::new(PoolState { queue: VecDeque::new(), active: 0, shutdown: false }), Condvar::new(), Condvar::new()));
        let mut workers = Vec::with_capacity(threads);
        for i in 0..threads {
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
        Pool { state, workers, capacity: capacity.max(1) }
    }

    /// Number of worker threads.
    pub fn threads(&self) -> usize {
        self.workers.len()
    }

    /// Queue `job`, waiting while the pool is at capacity.
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

/// The default worker count: the machine's parallelism, capped.
pub fn default_threads() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1).clamp(1, 16)
}
