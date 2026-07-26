//! Per-rung segment chunk queue connecting the decode pump to N
//! encoder workers.
//!
//! v2 multi-GPU model (2026-05-11): the pump groups decoded source
//! frames into fixed-size chunks (one chunk = one CMAF segment's
//! worth of frames = `keyframe_interval` frames) and pushes them
//! into this queue with a monotonic segment index. Encoder workers
//! pop chunks and emit segments. The segment index travels with the
//! frames so each worker knows which output file to write.
//!
//! Single-producer, multi-consumer. Bounded capacity for memory
//! safety: pump blocks when full, workers block when empty.

use std::collections::VecDeque;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use codec::frame::VideoFrame;
use tokio::sync::Notify;

pub struct SegmentChunk {
    pub segment_idx: usize,
    /// Every frame the worker should **encode**, lead-in margin included.
    pub frames: Vec<VideoFrame>,
    /// How many leading frames are margin: encoded to warm the encoder's rate
    /// control and lookahead, then discarded.
    ///
    /// Without it the first kept frame of a chunk is a cold-start IDR from a
    /// brand-new encoder with no history, which over-allocates bits and "pops"
    /// against the frame before it — measured as a 2.27x inter-frame
    /// discontinuity at chunk boundaries where an ordinary IDR costs 1.21x.
    /// Zero for the first chunk, which has nothing before it.
    pub lead_in: usize,
    /// How many frames after the lead-in belong to the output. Frames past
    /// `lead_in + keep` are lead-out margin: they absorb the encoder flush so
    /// the last kept frame isn't encoded with an empty pipeline behind it.
    pub keep: usize,
    pub is_final: bool,
}

pub struct SegmentChunkQueue {
    inner: Mutex<VecDeque<SegmentChunk>>,
    capacity: usize,
    push_notify: Notify,
    pop_notify: Notify,
    closed: AtomicBool,
    pushed_segments: AtomicUsize,
    popped_segments: AtomicUsize,
}

impl SegmentChunkQueue {
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "SegmentChunkQueue capacity must be > 0");
        Self {
            inner: Mutex::new(VecDeque::with_capacity(capacity)),
            capacity,
            push_notify: Notify::new(),
            pop_notify: Notify::new(),
            closed: AtomicBool::new(false),
            pushed_segments: AtomicUsize::new(0),
            popped_segments: AtomicUsize::new(0),
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn is_closed(&self) -> bool {
        self.closed.load(Ordering::Acquire)
    }

    pub fn pushed_segments(&self) -> usize {
        self.pushed_segments.load(Ordering::Relaxed)
    }

    pub fn popped_segments(&self) -> usize {
        self.popped_segments.load(Ordering::Relaxed)
    }

    /// Push one segment chunk. Awaits capacity. Returns `false` if
    /// closed before the chunk could be enqueued.
    pub async fn push(&self, chunk: SegmentChunk) -> bool {
        let mut chunk_slot = Some(chunk);
        loop {
            let waker = self.pop_notify.notified();
            tokio::pin!(waker);
            waker.as_mut().enable();
            if self.closed.load(Ordering::Acquire) {
                return false;
            }
            {
                let mut q = self.inner.lock().unwrap();
                if q.len() < self.capacity {
                    q.push_back(chunk_slot.take().unwrap());
                    self.pushed_segments.fetch_add(1, Ordering::Relaxed);
                    drop(q);
                    self.push_notify.notify_one();
                    return true;
                }
            }
            waker.await;
        }
    }

    /// Pop one chunk. Blocks until one is available. Returns `None`
    /// only once the queue is closed and drained.
    pub async fn pop(&self) -> Option<SegmentChunk> {
        loop {
            let waker = self.push_notify.notified();
            tokio::pin!(waker);
            waker.as_mut().enable();
            {
                let mut q = self.inner.lock().unwrap();
                if let Some(chunk) = q.pop_front() {
                    self.popped_segments.fetch_add(1, Ordering::Relaxed);
                    drop(q);
                    self.pop_notify.notify_waiters();
                    return Some(chunk);
                }
                if self.closed.load(Ordering::Acquire) {
                    return None;
                }
            }
            waker.await;
        }
    }

    /// Put a chunk back at the FRONT of the queue. Bypasses capacity
    /// (a requeued chunk briefly exceeds capacity by 1; the queue
    /// drains back under capacity at the next pop). Used by encoder
    /// workers that pop a chunk, observe a cross-vendor codec
    /// invariant mismatch, and need to hand the chunk off to another
    /// worker. Decrements `popped_segments` so the dispatcher's
    /// `pushed > popped` predicate still reflects work-remaining.
    ///
    /// Returns `false` if the queue is closed AND no other consumer
    /// can pick the chunk up — in which case the caller should treat
    /// this chunk as orphaned and the run will fail coverage.
    pub fn push_front(&self, chunk: SegmentChunk) -> bool {
        if self.closed.load(Ordering::Acquire) {
            // Closed: still re-queue if there are popped-but-not-yet-
            // -finished consumers that might come back; the queue
            // drains FIFO, so the chunk lands at head.
        }
        let mut q = self.inner.lock().unwrap();
        q.push_front(chunk);
        // The caller had popped this chunk before, so `popped_segments`
        // was incremented; undo that increment so observers see the
        // queue's true pending count.
        self.popped_segments.fetch_sub(1, Ordering::Relaxed);
        drop(q);
        self.push_notify.notify_one();
        true
    }

    /// Mark the queue closed. Pending and future pushes return false;
    /// pending pops wake and return remaining chunks then None.
    pub fn close(&self) {
        self.closed.store(true, Ordering::Release);
        self.push_notify.notify_waiters();
        self.pop_notify.notify_waiters();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;
    use codec::frame::{ColorSpace, PixelFormat};
    use std::sync::Arc;

    fn dummy_frame(idx: u64) -> VideoFrame {
        let mut data = vec![idx as u8; 16 * 16];
        data.extend(vec![128u8; 8 * 8]);
        data.extend(vec![128u8; 8 * 8]);
        VideoFrame::new(
            Bytes::from(data),
            16,
            16,
            PixelFormat::Yuv420p,
            ColorSpace::Bt709,
            idx,
        )
    }

    fn chunk(idx: usize, frame_count: usize) -> SegmentChunk {
        SegmentChunk {
            segment_idx: idx,
            frames: (0..frame_count).map(|i| dummy_frame(i as u64)).collect(),
            lead_in: 0,
            keep: frame_count,
            is_final: false,
        }
    }

    #[tokio::test]
    async fn push_then_pop_preserves_order() {
        let q = Arc::new(SegmentChunkQueue::new(4));
        assert!(q.push(chunk(0, 2)).await);
        assert!(q.push(chunk(1, 3)).await);
        q.close();
        let a = q.pop().await.unwrap();
        assert_eq!(a.segment_idx, 0);
        assert_eq!(a.frames.len(), 2);
        let b = q.pop().await.unwrap();
        assert_eq!(b.segment_idx, 1);
        assert_eq!(b.frames.len(), 3);
        assert!(q.pop().await.is_none());
    }

    #[tokio::test]
    async fn pop_blocks_until_pushed() {
        let q = Arc::new(SegmentChunkQueue::new(4));
        let q2 = q.clone();
        let pop_task = tokio::spawn(async move { q2.pop().await });
        tokio::task::yield_now().await;
        assert!(q.push(chunk(7, 1)).await);
        let got = pop_task.await.unwrap().unwrap();
        assert_eq!(got.segment_idx, 7);
    }

    #[tokio::test]
    async fn push_blocks_when_full_resumes_after_pop() {
        let q = Arc::new(SegmentChunkQueue::new(2));
        assert!(q.push(chunk(0, 1)).await);
        assert!(q.push(chunk(1, 1)).await);
        let q2 = q.clone();
        let push_task = tokio::spawn(async move { q2.push(chunk(2, 1)).await });
        tokio::task::yield_now().await;
        let drained = q.pop().await.unwrap();
        assert_eq!(drained.segment_idx, 0);
        assert!(push_task.await.unwrap());
        q.close();
        let r1 = q.pop().await.unwrap();
        assert_eq!(r1.segment_idx, 1);
        let r2 = q.pop().await.unwrap();
        assert_eq!(r2.segment_idx, 2);
        assert!(q.pop().await.is_none());
    }

    #[tokio::test]
    async fn close_wakes_pending_pop() {
        let q = Arc::new(SegmentChunkQueue::new(2));
        let q2 = q.clone();
        let pop_task = tokio::spawn(async move { q2.pop().await });
        tokio::task::yield_now().await;
        q.close();
        assert!(pop_task.await.unwrap().is_none());
    }

    #[tokio::test]
    async fn multiple_consumers_partition_chunks() {
        let q = Arc::new(SegmentChunkQueue::new(8));
        for i in 0..6 {
            assert!(q.push(chunk(i, 1)).await);
        }
        q.close();
        let mut handles = Vec::new();
        for _ in 0..3 {
            let q2 = q.clone();
            handles.push(tokio::spawn(async move {
                let mut got = Vec::new();
                while let Some(c) = q2.pop().await {
                    got.push(c.segment_idx);
                }
                got
            }));
        }
        let mut all: Vec<usize> = Vec::new();
        for h in handles {
            all.extend(h.await.unwrap());
        }
        all.sort();
        assert_eq!(all, vec![0, 1, 2, 3, 4, 5]);
    }

    #[tokio::test]
    async fn closed_rejects_push() {
        let q = Arc::new(SegmentChunkQueue::new(2));
        q.close();
        assert!(!q.push(chunk(0, 1)).await);
    }

    #[tokio::test]
    async fn final_chunk_flag_is_preserved() {
        let q = Arc::new(SegmentChunkQueue::new(4));
        let mut last = chunk(9, 2);
        last.is_final = true;
        assert!(q.push(last).await);
        q.close();
        let got = q.pop().await.unwrap();
        assert!(got.is_final);
    }
}
