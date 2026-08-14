//! Per-rung scaler task: consume raw normalized source frames from
//! the shared decode pump, scale to rung dims, group K frames into a
//! `SegmentChunk`, push to the rung's `SegmentChunkQueue`.
//!
//! v3 multi-GPU model (2026-05-12): one scaler per rung sits between
//! the shared pump and that rung's encoder workers. The pump fans
//! frames out to every scaler's input channel; each scaler does its
//! own bilinear scale (CPU work) and chunks the result so workers
//! see one chunk per CMAF segment.
//!
//! Scalers exit when their input channel returns `None` (pump closed
//! all senders). On exit, the scaler flushes any in-progress chunk
//! (final partial segment) and closes the queue so encoder workers
//! drain and exit cleanly.

use anyhow::{Context, Result};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use codec::colorspace;
use codec::frame::VideoFrame;

use crate::frame_queue::{SegmentChunk, SegmentChunkQueue};

#[derive(Clone)]
pub struct RungScalerConfig {
    pub rung_idx: usize,
    pub target_width: u32,
    pub target_height: u32,
    /// Frames per segment chunk — the *kept* count, excluding overlap margin.
    pub frames_per_chunk: u32,
    /// Lead-in margin: frames replayed from the previous chunk's tail, encoded
    /// to warm the encoder and then discarded. `0` disables overlap.
    pub overlap: usize,
    /// Segment index this scaler's first chunk gets.
    ///
    /// Zero for a scaler fed by a pump that decodes the whole source. When the
    /// source is split into decode ranges across GPUs, each range's scaler
    /// starts where the previous range ended, so the segment numbering of the
    /// finished rung is continuous no matter which card produced which part.
    pub first_segment_idx: usize,
    /// Whether this scaler owns the end of the source.
    ///
    /// Only the last decode range may mark a chunk `is_final`. A middle range
    /// also ends with a short chunk — its range boundary — and flagging that
    /// as final would tell the muxer the stream ended in the middle of the
    /// video.
    pub is_final_range: bool,
}

/// Blocking scaler loop. Designed for `tokio::task::spawn_blocking`.
/// Returns the total number of segment chunks pushed.
pub fn run_rung_scaler_blocking(
    cfg: RungScalerConfig,
    frame_rx: tokio::sync::mpsc::Receiver<VideoFrame>,
    queue: Arc<SegmentChunkQueue>,
    rt: tokio::runtime::Handle,
) -> Result<usize> {
    // One producer: this scaler is the only thing feeding the queue, so its
    // exit is the queue's end.
    run_rung_scaler_blocking_shared(cfg, frame_rx, queue, rt, Arc::new(AtomicUsize::new(1)))
}

/// As [`run_rung_scaler_blocking`], for a queue fed by more than one scaler.
///
/// When the source is split into decode ranges, each range has its own scaler
/// pushing into the *same* rung queue. `producers` counts them, and the queue
/// is closed by whichever finishes last.
///
/// Closing on the first exit — which is what a per-scaler `queue.close()` does —
/// would wake the encoder workers and drain them while the other ranges were
/// still producing, losing every segment after the first range's end.
pub fn run_rung_scaler_blocking_shared(
    cfg: RungScalerConfig,
    mut frame_rx: tokio::sync::mpsc::Receiver<VideoFrame>,
    queue: Arc<SegmentChunkQueue>,
    rt: tokio::runtime::Handle,
    producers: Arc<AtomicUsize>,
) -> Result<usize> {
    let outcome = scaler_loop(&cfg, &mut frame_rx, &queue, &rt);
    if producers.fetch_sub(1, Ordering::AcqRel) == 1 {
        queue.close();
    }
    outcome
}

fn scaler_loop(
    cfg: &RungScalerConfig,
    frame_rx: &mut tokio::sync::mpsc::Receiver<VideoFrame>,
    queue: &Arc<SegmentChunkQueue>,
    rt: &tokio::runtime::Handle,
) -> Result<usize> {
    let chunk_size = cfg.frames_per_chunk as usize;
    assert!(chunk_size > 0, "frames_per_chunk must be > 0");

    let mut current_chunk: Vec<VideoFrame> = Vec::with_capacity(chunk_size);
    let mut next_segment_idx: usize = cfg.first_segment_idx;
    let mut pushed_segments: usize = 0;
    let mut producer_aborted = false;
    // Trailing frames of the previous chunk, replayed as this chunk's lead-in
    // so its encoder starts warm. Held as a ring of at most `OVERLAP_FRAMES`.
    let mut carry: Vec<VideoFrame> = Vec::with_capacity(cfg.overlap);

    let emit = |lead: &[VideoFrame],
                chunk_frames: Vec<VideoFrame>,
                idx: usize,
                is_final: bool|
     -> Result<bool> {
        let keep = chunk_frames.len();
        let mut frames = Vec::with_capacity(lead.len() + keep);
        frames.extend_from_slice(lead);
        frames.extend(chunk_frames);
        let chunk = SegmentChunk {
            segment_idx: idx,
            frames,
            lead_in: lead.len(),
            keep,
            is_final,
        };
        let q = Arc::clone(queue);
        let accepted = rt.block_on(async move { q.push(chunk).await });
        Ok(accepted)
    };

    loop {
        let frame = match rt.block_on(frame_rx.recv()) {
            Some(f) => f,
            None => break,
        };
        let scaled = colorspace::scale_frame(&frame, cfg.target_width, cfg.target_height)
            .with_context(|| {
                format!(
                    "rung {} scaler: scale_frame to {}×{}",
                    cfg.rung_idx, cfg.target_width, cfg.target_height
                )
            })?;
        current_chunk.push(scaled);
        if current_chunk.len() >= chunk_size {
            let full = std::mem::replace(&mut current_chunk, Vec::with_capacity(chunk_size));
            let idx = next_segment_idx;
            next_segment_idx += 1;
            // Keep this chunk's tail as the next one's lead-in before handing
            // it off. Cloning a handful of frames per chunk is cheap next to
            // encoding them, and `VideoFrame`'s payload is a refcounted
            // `Bytes`, so this copies headers rather than pixels.
            let next_carry: Vec<VideoFrame> = if cfg.overlap == 0 {
                Vec::new()
            } else {
                full.iter().rev().take(cfg.overlap).rev().cloned().collect()
            };
            if !emit(&carry, full, idx, false)? {
                producer_aborted = true;
                break;
            }
            carry = next_carry;
            pushed_segments += 1;
        }
    }

    if !producer_aborted && !current_chunk.is_empty() {
        let idx = next_segment_idx;
        // `is_final` only if this scaler owns the end of the source. A middle
        // decode range ends with a short chunk too, and calling that final
        // would end the stream partway through the video.
        if emit(&carry, current_chunk, idx, cfg.is_final_range)? {
            pushed_segments += 1;
        }
    }

    Ok(pushed_segments)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_clone_preserves_fields() {
        let cfg = RungScalerConfig {
            rung_idx: 1,
            target_width: 1280,
            target_height: 720,
            frames_per_chunk: 60,
            overlap: 16,
            first_segment_idx: 0,
            is_final_range: true,
        };
        let copy = cfg.clone();
        assert_eq!(copy.rung_idx, 1);
        assert_eq!(copy.frames_per_chunk, 60);
    }

    #[test]
    fn the_queue_closes_only_when_the_last_range_finishes() {
        // Two decode ranges feeding one rung. The first to exit must leave the
        // queue open, or the encoder workers drain and every segment belonging
        // to the other range is lost.
        let producers = Arc::new(AtomicUsize::new(2));
        let queue = Arc::new(SegmentChunkQueue::new(4));

        assert_eq!(producers.fetch_sub(1, Ordering::AcqRel), 2, "first exit is not the last");
        assert!(!queue.is_closed(), "queue must stay open while a range is still producing");

        assert_eq!(producers.fetch_sub(1, Ordering::AcqRel), 1, "second exit is the last");
        queue.close();
        assert!(queue.is_closed());
    }
}
