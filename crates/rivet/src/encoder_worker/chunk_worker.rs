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
        let n = shared_frames_encoded.fetch_add(1, std::sync::atomic::Ordering::AcqRel) + 1;
        let _ = progress_tx.try_send(n);
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
        let start = lead_in;
        let end = (start + chunk.keep).min(packets.len());
        if start > 0 || end < packets.len() {
            packets = packets[start..end].to_vec();
        }
    } else if lead_in > 0 || skip > 0 {
        anyhow::bail!(
            "chunk {segment_idx}: encoder returned {} packets for {submitted} frames, so the              lead-in margin can't be located; refusing to guess where the chunk starts",
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
