//! Single-file chunked encode: workers collect packets (instead of writing CMAF
//! segments) so the orchestrator can stitch them, in segment order, into one MP4.

use anyhow::{Context, Result};
use std::sync::Arc;
use tokio::sync::mpsc;
use codec::encode::{self, EncoderConfig};
use crate::frame_queue::{SegmentChunk, SegmentChunkQueue};
use super::{EncoderWorkerConfig, InvariantCheck, validate_or_set_rung_invariant};

/// One chunk's encoded packets, in encode (= display, no B-frames) order.
#[derive(Debug)]
pub struct ChunkPackets {
    pub segment_idx: usize,
    pub packets: Vec<encode::EncodedPacket>,
}

/// Encoder worker that COLLECTS packets per chunk (single-file path). Each
/// chunk is encoded by a fresh encoder (first frame an IDR); the cross-vendor
/// codec invariant is enforced on the first packet (mismatch → requeue + exit,
/// exactly like the CMAF worker). Ordered `ChunkPackets` are pushed to `out`.
#[allow(clippy::too_many_arguments)]
pub fn run_chunk_encoder_worker_blocking(
    cfg: EncoderWorkerConfig,
    queue: Arc<SegmentChunkQueue>,
    rt: tokio::runtime::Handle,
    shared_frames_encoded: Arc<std::sync::atomic::AtomicU64>,
    // Encoded payload bytes so far for this rung — lets the CLI show size to
    // date and project a finished size. `bytes_out` used to be reported as a
    // flat zero for the whole run.
    shared_bytes_encoded: Arc<std::sync::atomic::AtomicU64>,
    progress_tx: mpsc::Sender<u64>,
    out: Arc<std::sync::Mutex<Vec<ChunkPackets>>>,
) -> Result<()> {
    let enc_config = super::build_enc_config(&cfg);
    loop {
        let chunk = match rt.block_on(queue.pop()) {
            Some(c) => c,
            None => break,
        };
        match encode_chunk_to_packets(
            &cfg,
            &enc_config,
            chunk,
            &shared_frames_encoded,
            &shared_bytes_encoded,
            &progress_tx,
        )?
        {
            ChunkOutcome::Encoded(c) => out.lock().unwrap().push(c),
            ChunkOutcome::RequeuedOnMismatch { chunk, diff } => {
                tracing::warn!(
                    rung_idx = cfg.rung_idx,
                    gpu_vendor = ?cfg.gpu_vendor,
                    diff = %diff,
                    "chunk worker: codec invariant mismatch — requeuing chunk and exiting"
                );
                let _ = queue.push_front(chunk);
                break;
            }
        }
    }
    Ok(())
}

/// What one unit of chunk-encode work produced — the single-file counterpart
/// of [`super::UnitOutcome`], for callers that own their own scheduling.
pub enum ChunkUnitOutcome {
    /// The chunk was encoded; its packets, in order.
    Encoded(ChunkPackets),
    /// This worker's vendor disagrees with the rung's codec invariant on a
    /// mandatory field. The chunk comes back untouched for another worker;
    /// nothing was recorded.
    Rejected { chunk: SegmentChunk, diff: String },
}

/// Encode exactly one chunk to packets.
///
/// `run_chunk_encoder_worker_blocking` is this in a loop over one rung's
/// queue; this entry point exists for the ladder-wide shape, where a worker
/// takes whatever unit is next and `cfg` changes between calls because the
/// next unit belongs to a different rung. The encoder is created per chunk
/// either way, so hopping rungs costs nothing extra.
pub fn encode_chunk_unit(
    cfg: &EncoderWorkerConfig,
    chunk: SegmentChunk,
    shared_frames_encoded: &std::sync::atomic::AtomicU64,
    shared_bytes_encoded: &std::sync::atomic::AtomicU64,
    progress_tx: &mpsc::Sender<u64>,
) -> Result<ChunkUnitOutcome> {
    let enc_config = super::build_enc_config(cfg);
    match encode_chunk_to_packets(
        cfg,
        &enc_config,
        chunk,
        shared_frames_encoded,
        shared_bytes_encoded,
        progress_tx,
    )? {
        ChunkOutcome::Encoded(c) => Ok(ChunkUnitOutcome::Encoded(c)),
        ChunkOutcome::RequeuedOnMismatch { chunk, diff } => {
            Ok(ChunkUnitOutcome::Rejected { chunk, diff })
        }
    }
}

/// Which of a chunk's frames reach the output.
///
/// A chunk is encoded with a lead-in margin ahead of its first kept frame, and
/// that margin is discarded afterwards. There are two ways it gets dropped:
/// encoded and then sliced off (`lead_in > 0`), or never submitted at all
/// (`skip > 0`, the fallback when the backend can't force a keyframe). Exactly
/// one of the two is non-zero, and either way the surviving frames start at
/// `lead_in + skip`.
///
/// Returned as one range so the progress counter and the packet slice are
/// driven from the same arithmetic — they disagreed before, and only the
/// progress line showed it.
fn kept_range(
    lead_in: usize,
    skip: usize,
    keep: usize,
    frames: usize,
) -> std::ops::Range<usize> {
    let start = (lead_in + skip).min(frames);
    start..(start + keep).min(frames)
}

enum ChunkOutcome {
    Encoded(ChunkPackets),
    RequeuedOnMismatch { chunk: SegmentChunk, diff: String },
}

fn encode_chunk_to_packets(
    cfg: &EncoderWorkerConfig,
    enc_config: &EncoderConfig,
    chunk: SegmentChunk,
    shared_frames_encoded: &std::sync::atomic::AtomicU64,
    shared_bytes_encoded: &std::sync::atomic::AtomicU64,
    progress_tx: &mpsc::Sender<u64>,
) -> Result<ChunkOutcome> {
    // One encoder per chunk. Each chunk has to be an independently decodable
    // IDR-led GOP so the stitcher can concatenate chunks encoded out of order
    // on different GPUs, and a fresh session is the simple way to guarantee
    // that. It isn't free — on Intel every session opens a VA display and
    // re-runs Query/Init, ~1300 times for a feature-length file — but the
    // pipeline is decode-bound well before that shows up. Pooling sessions and
    // using `MFXVideoENCODE_Reset` per chunk is the optimisation; see TODO.md.
    let mut encoder =
        encode::select_encoder(enc_config.clone(), None).context("creating encoder for chunk")?;
    let segment_idx = chunk.segment_idx;
    let mut packets: Vec<encode::EncodedPacket> = Vec::new();
    let mut pending: Vec<encode::EncodedPacket> = Vec::new();
    let mut decided = false;

    // The chunk's first *kept* frame has to be an IDR so chunks concatenate.
    // With a lead-in margin ahead of it, it is no longer the encoder's frame 0,
    // so the encoder would put its IDR on the first margin frame instead — a
    // frame we're about to throw away. Promote the right one explicitly.
    //
    // If the backend can't force a keyframe, fall back to no margin: encode the
    // kept range only, which is exactly the previous behaviour.
    let mut lead_in = chunk.lead_in;
    if lead_in > 0 && encoder.force_keyframe_next().is_err() {
        tracing::debug!(
            rung_idx = cfg.rung_idx,
            "encoder cannot force a keyframe; encoding this chunk without a lead-in margin"
        );
        lead_in = 0;
    }

    // With the margin dropped, skip the frames it would have covered.
    let skip = if lead_in == 0 { chunk.lead_in } else { 0 };

    // Which submitted frames survive into the output. Both the progress counter
    // and the packet slice below are driven from this one range so they cannot
    // drift apart.
    let kept = kept_range(lead_in, skip, chunk.keep, chunk.frames.len());

    for (i, frame) in chunk.frames.iter().enumerate().skip(skip) {
        if lead_in > 0 && i == lead_in {
            encoder
                .force_keyframe_next()
                .context("forcing the chunk's opening IDR")?;
        }
        encoder.send_frame(frame).context("send_frame in chunk worker")?;
        while let Some(packet) = encoder.receive_packet().context("receive_packet in chunk worker")? {
            if !decided {
                match validate_or_set_rung_invariant(
                    cfg.rung_idx,
                    cfg.gpu_vendor,
                    &cfg.rung_invariant,
                    &packet.data,
                    cfg.codec,
                )? {
                    InvariantCheck::Matched | InvariantCheck::SetByThisWorker => decided = true,
                    InvariantCheck::Mismatched { diff } => {
                        return Ok(ChunkOutcome::RequeuedOnMismatch { chunk, diff });
                    }
                }
                pending.push(packet);
                continue;
            }
            packets.append(&mut pending);
            packets.push(packet);
        }
        // Only frames that reach the output. The margin is encoded and thrown
        // away, so counting it walked `frames_done` past the input's real
        // frame count — 69827 against 63544 on a feature-length file, a 9.9%
        // overshoot that the percentage's `.min(99.0)` then capped, so the
        // line read "99.0%  69827/63544".
        if kept.contains(&i) {
            let n = shared_frames_encoded.fetch_add(1, std::sync::atomic::Ordering::AcqRel) + 1;
            let _ = progress_tx.try_send(n);
        }
    }
    if decided {
        packets.append(&mut pending);
    }
    encoder.flush().context("flush in chunk worker")?;
    while let Some(packet) = encoder
        .receive_packet()
        .context("receive_packet after flush in chunk worker")?
    {
        packets.push(packet);
    }
    // Drop the margin. Packets are 1:1 with submitted frames here — no
    // B-frames (`GopRefDist = 1`), so encode order is display order — and the
    // stitch depends on that, so verify rather than assume: slicing a
    // mismatched vector would silently shift a chunk against its neighbours.
    let submitted = chunk.frames.len() - skip;
    if packets.len() == submitted {
        // Packets are 1:1 with the frames submitted from `skip` onwards, so
        // rebase the kept frame range onto them.
        let (start, end) = (kept.start - skip, (kept.end - skip).min(packets.len()));
        if start > 0 || end < packets.len() {
            packets = packets[start..end].to_vec();
        }
    } else {
        // Unconditional: a packet/frame mismatch means the stitch would be
        // wrong whether or not there's a margin to locate. This used to be
        // checked only when a margin was present, so a short chunk on the
        // no-margin path slipped through and silently shortened the output.
        anyhow::bail!(
            "chunk {segment_idx}: encoder returned {} packets for {submitted} frames — the stitch assumes one packet per frame (no B-frames), so this would shift the chunk against its neighbours",
            packets.len()
        );
    }

    // Counted once from the finished vector rather than per packet: that way
    // the tally includes both the packets held back pending the invariant check
    // and everything drained after flush.
    let chunk_bytes: u64 = packets.iter().map(|p| p.data.len() as u64).sum();
    shared_bytes_encoded.fetch_add(chunk_bytes, std::sync::atomic::Ordering::Relaxed);
    Ok(ChunkOutcome::Encoded(ChunkPackets { segment_idx, packets }))
}

#[cfg(test)]
mod tests {
    use super::kept_range;

    /// The bug this range exists to prevent: a chunk encodes `lead_in + keep`
    /// frames and keeps only `keep` of them, so counting submitted frames
    /// overshoots by the whole margin. At the shipped 10-GOP chunk with a
    /// one-GOP margin that is 10%, which is what put "99.0%  69827/63544" on
    /// the progress line of a 63544-frame file.
    #[test]
    fn the_margin_is_not_counted() {
        // 48-frame margin ahead of 480 kept frames.
        let r = kept_range(48, 0, 480, 528);
        assert_eq!(r, 48..528);
        assert_eq!(r.len(), 480, "only the kept frames count");

        // Summed over a whole file, the margin is what overshot.
        let chunks = 63544usize.div_ceil(480);
        let submitted: usize = (0..chunks).map(|i| if i == 0 { 480 } else { 528 }).sum();
        assert!(submitted > 63544, "counting submissions overshoots");
        let kept: usize = (0..chunks)
            .map(|i| {
                let lead = if i == 0 { 0 } else { 48 };
                kept_range(lead, 0, 480, lead + 480).len()
            })
            .sum();
        assert_eq!(kept, chunks * 480, "counting kept frames does not");
    }

    #[test]
    fn the_no_margin_fallback_skips_instead_of_slicing() {
        // `lead_in` forced to 0 because the backend can't force a keyframe, so
        // the margin is never submitted. Same surviving frames either way.
        assert_eq!(kept_range(0, 48, 480, 528), 48..528);
        assert_eq!(kept_range(48, 0, 480, 528).len(), kept_range(0, 48, 480, 528).len());
    }

    #[test]
    fn the_first_chunk_has_no_margin() {
        assert_eq!(kept_range(0, 0, 480, 480), 0..480);
    }

    #[test]
    fn a_short_final_chunk_clamps_to_what_it_has() {
        // The tail of the file: fewer frames than `keep` asks for.
        assert_eq!(kept_range(48, 0, 480, 200), 48..200);
        // And degenerately, a chunk shorter than its own margin.
        assert_eq!(kept_range(48, 0, 480, 20), 20..20);
        assert!(kept_range(48, 0, 480, 20).is_empty());
    }

    #[test]
    fn the_range_rebases_onto_packets_the_way_the_slice_does() {
        // Packets are 1:1 with frames submitted from `skip` onward, so the
        // rebased range must stay inside a packet vector of that length.
        for &(lead_in, skip, keep, frames) in &[
            (48usize, 0usize, 480usize, 528usize),
            (0, 48, 480, 528),
            (0, 0, 480, 480),
            (48, 0, 480, 200),
        ] {
            let kept = kept_range(lead_in, skip, keep, frames);
            let submitted = frames - skip;
            let (start, end) = (kept.start - skip, (kept.end - skip).min(submitted));
            assert!(start <= end && end <= submitted, "{lead_in}/{skip}/{keep}/{frames}");
            assert_eq!(end - start, kept.len(), "slice and counter must agree");
        }
    }
}
