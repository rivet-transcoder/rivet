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
}

/// Blocking scaler loop. Designed for `tokio::task::spawn_blocking`.
/// Returns the total number of segment chunks pushed.
pub fn run_rung_scaler_blocking(
    cfg: RungScalerConfig,
    mut frame_rx: tokio::sync::mpsc::Receiver<VideoFrame>,
    queue: Arc<SegmentChunkQueue>,
    rt: tokio::runtime::Handle,
) -> Result<usize> {
    let outcome = scaler_loop(&cfg, &mut frame_rx, &queue, &rt);
    // Always close the queue on exit so encoder workers wake + exit.
    queue.close();
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
    let mut next_segment_idx: usize = 0;
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
        if emit(&carry, current_chunk, idx, true)? {
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
        };
        let copy = cfg.clone();
        assert_eq!(copy.rung_idx, 1);
        assert_eq!(copy.frames_per_chunk, 60);
    }
}
