//! Single-file chunked encode: workers collect packets (instead of writing CMAF
//! segments) so the orchestrator can stitch them, in segment order, into one MP4.

use anyhow::{Context, Result};
use std::sync::Arc;
use tokio::sync::mpsc;
use codec::encode::{self, EncoderConfig};
use crate::frame_queue::{SegmentChunk, SegmentChunkQueue};
use super::{EncoderSessionPool, EncoderWorkerConfig, InvariantCheck, validate_or_set_rung_invariant};

/// One chunk's encoded packets, in encode (= display, no B-frames) order.
#[derive(Debug)]
pub struct ChunkPackets {
    pub segment_idx: usize,
    pub packets: Vec<encode::EncodedPacket>,
}

/// Encoder worker that COLLECTS packets per chunk (single-file path). Each
/// chunk is a fresh stream (first frame an IDR) on a session the worker keeps
/// between chunks and resets — see [`EncoderSessionPool`]; the cross-vendor
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
    let mut sessions = EncoderSessionPool::new();
    loop {
        let chunk = match rt.block_on(queue.pop()) {
            Some(c) => c,
            None => break,
        };
        match encode_chunk_to_packets(
            &cfg,
            &enc_config,
            chunk,
            &mut sessions,
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
    let stats = sessions.stats();
    tracing::info!(
        rung_idx = cfg.rung_idx,
        gpu_index = ?cfg.gpu_index,
        built = stats.built,
        reused = stats.reused,
        evicted = stats.evicted,
        reset_unsupported = stats.reset_unsupported,
        reset_failed = stats.reset_failed,
        "chunk worker done; encoder sessions built vs reused"
    );
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
/// next unit belongs to a different rung. `sessions` is that worker's pool:
/// a chunk of the same rung as the last one reuses the session after a
/// reset, a rung hop evicts it and builds — so hopping costs one
/// construction, which is what every chunk used to cost.
pub fn encode_chunk_unit(
    cfg: &EncoderWorkerConfig,
    chunk: SegmentChunk,
    sessions: &mut EncoderSessionPool,
    shared_frames_encoded: &std::sync::atomic::AtomicU64,
    shared_bytes_encoded: &std::sync::atomic::AtomicU64,
    progress_tx: &mpsc::Sender<u64>,
) -> Result<ChunkUnitOutcome> {
    let enc_config = super::build_enc_config(cfg);
    match encode_chunk_to_packets(
        cfg,
        &enc_config,
        chunk,
        sessions,
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
    sessions: &mut EncoderSessionPool,
    shared_frames_encoded: &std::sync::atomic::AtomicU64,
    shared_bytes_encoded: &std::sync::atomic::AtomicU64,
    progress_tx: &mpsc::Sender<u64>,
) -> Result<ChunkOutcome> {
    // One *stream* per chunk. Each chunk has to be an independently decodable
    // IDR-led GOP so the stitcher can concatenate chunks encoded out of order
    // on different GPUs. That used to mean one encoder per chunk — a fresh
    // session is the simple way to guarantee it, at ~1300 constructions on a
    // feature-length file. The pool gives the same guarantee through
    // `Encoder::reset`, and builds only when the backend cannot reset or the
    // configuration changed. Everything below is unchanged by that: the
    // margin logic, the IDR promotion and the packet-count check all reason
    // about this chunk's stream alone, which a reset session is.
    let mut encoder = sessions.acquire(enc_config, cfg.backend)?;
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
                        // The session is dropped with the rejection: its
                        // stream was never finished, and this worker is about
                        // to be struck off the rung anyway.
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
    // Flushed and drained: the stream is over and the session may be reset
    // for the next chunk. Handed back before the packet-count check so a
    // chunk that fails it still returns its session — the failure is in what
    // the stream produced, not in the session.
    sessions.release(enc_config, encoder);
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
    use super::*;
    use super::super::PoolStats;
    use codec::encode::{Encoder, EncoderBackend, EncoderConfig};
    use codec::frame::{ColorMetadata, ColorSpace, PixelFormat, VideoCodec, VideoFrame};
    use std::sync::{Arc, RwLock};

    // ── The pooled session and the chunk's IDR ─────────────────────────
    //
    // Real bitstreams from the native H.264 encoder (always compiled, asked
    // for by name), because the rung invariant parses the first packet's
    // SPS and a fake could not satisfy it. Tiny frames keep it fast.

    fn frame(width: u32, height: u32, pts: u64, seed: u8) -> VideoFrame {
        let luma = (width * height) as usize;
        let chroma = ((width / 2) * (height / 2)) as usize;
        let mut data = Vec::with_capacity(luma + 2 * chroma);
        for i in 0..luma {
            data.push(seed.wrapping_add((i % 7) as u8 * 9));
        }
        data.extend(std::iter::repeat_n(128u8, 2 * chroma));
        VideoFrame::new(bytes::Bytes::from(data), width, height, PixelFormat::Yuv420p, ColorSpace::Bt709, pts)
    }

    fn worker_config(width: u32, height: u32, keyframe_interval: u32) -> EncoderWorkerConfig {
        EncoderWorkerConfig {
            overrides: Default::default(),
            backend: None,
            rung_idx: 0,
            codec: VideoCodec::H264,
            width,
            height,
            frame_rate: 30.0,
            quality: 30,
            speed_preset: u8::MAX,
            target: codec::encode::tuning::QualityTarget::Standard,
            tier: codec::encode::tuning::SpeedTier::Standard,
            threads: 1,
            gpu_index: None,
            gpu_vendor: None,
            output_color_metadata: ColorMetadata::default(),
            output_pixel_format: PixelFormat::Yuv420p,
            constant_qp: false,
            timescale: 30000,
            per_frame_ticks: 1000,
            keyframe_interval,
            segment_target_ticks: 60_000,
            output_dir: std::path::PathBuf::from("unused"),
            rung_invariant: Arc::new(RwLock::new(None)),
        }
    }

    /// A chunk of `lead_in + keep` frames starting at `first_pts`.
    fn chunk(cfg: &EncoderWorkerConfig, segment_idx: usize, first_pts: u64, lead_in: usize, keep: usize) -> SegmentChunk {
        let frames = (0..lead_in + keep)
            .map(|i| frame(cfg.width, cfg.height, first_pts + i as u64, (first_pts as u8).wrapping_add(i as u8 * 3)))
            .collect();
        SegmentChunk { segment_idx, frames, lead_in, keep, is_final: false }
    }

    /// A pool over the native H.264 encoder, optionally wrapped so that
    /// `reset` succeeds without doing anything — the mutation the IDR check
    /// exists to catch.
    fn h264_pool(honour_reset: bool) -> EncoderSessionPool {
        EncoderSessionPool::with_builder(Box::new(move |config: &EncoderConfig, _backend| {
            let inner = encode::select_encoder(config.clone(), Some(EncoderBackend::H26x))?;
            Ok(if honour_reset { inner } else { Box::new(NoOpReset(inner)) })
        }))
    }

    /// Forwards everything and claims a reset it never performs.
    struct NoOpReset(Box<dyn Encoder>);
    impl Encoder for NoOpReset {
        fn send_frame(&mut self, f: &VideoFrame) -> Result<()> {
            self.0.send_frame(f)
        }
        fn flush(&mut self) -> Result<()> {
            self.0.flush()
        }
        fn receive_packet(&mut self) -> Result<Option<encode::EncodedPacket>> {
            self.0.receive_packet()
        }
        fn force_keyframe_next(&mut self) -> Result<()> {
            self.0.force_keyframe_next()
        }
        fn reset(&mut self) -> Result<()> {
            Ok(())
        }
    }

    fn encode_two_chunks(honour_reset: bool) -> (Vec<ChunkPackets>, PoolStats) {
        let cfg = worker_config(32, 16, 1000);
        let enc_config = super::super::build_enc_config(&cfg);
        let mut pool = h264_pool(honour_reset);
        let frames = std::sync::atomic::AtomicU64::new(0);
        let bytes = std::sync::atomic::AtomicU64::new(0);
        let (tx, _rx) = mpsc::channel(64);
        let mut out = Vec::new();
        // Chunk 0 has no margin; chunk 1 carries a 2-frame lead-in.
        for (idx, first_pts, lead_in, keep) in [(0usize, 0u64, 0usize, 4usize), (1, 4, 2, 4)] {
            let c = chunk(&cfg, idx, first_pts.saturating_sub(lead_in as u64), lead_in, keep);
            match encode_chunk_to_packets(&cfg, &enc_config, c, &mut pool, &frames, &bytes, &tx).unwrap() {
                ChunkOutcome::Encoded(p) => out.push(p),
                ChunkOutcome::RequeuedOnMismatch { diff, .. } => panic!("unexpected mismatch: {diff}"),
            }
        }
        assert_eq!(frames.load(std::sync::atomic::Ordering::SeqCst), 8, "only kept frames are counted");
        (out, pool.stats())
    }

    /// The session is built once and reused; each chunk still opens with an
    /// IDR and carries exactly its kept frames.
    #[test]
    fn a_reused_session_still_opens_every_chunk_with_an_idr() {
        let (chunks, stats) = encode_two_chunks(true);
        assert_eq!(stats, PoolStats { built: 1, reused: 1, ..Default::default() });
        for c in &chunks {
            assert_eq!(c.packets.len(), 4, "chunk {} keeps 4 frames", c.segment_idx);
            assert!(c.packets[0].is_keyframe, "chunk {} must open with an IDR", c.segment_idx);
            assert!(c.packets[1..].iter().all(|p| !p.is_keyframe), "one IDR per chunk at this GOP");
        }
        // And the margin frames are the ones dropped: chunk 1's first kept
        // pts is 4, not 2.
        assert_eq!(chunks[1].packets[0].pts, 4);
    }

    /// The mutation: a `reset` that reports success and does nothing. The
    /// lead-in margin's forced IDR still lands (that is `force_keyframe_next`,
    /// not `reset`), so this test drives the margin-less path — the one every
    /// backend without `force_keyframe_next` takes, NVENC included — where the
    /// chunk's opening IDR comes from the reset alone. It must go red.
    #[test]
    fn a_reset_that_does_nothing_loses_the_chunks_idr() {
        let cfg = worker_config(32, 16, 1000);
        let enc_config = super::super::build_enc_config(&cfg);
        let frames = std::sync::atomic::AtomicU64::new(0);
        let bytes = std::sync::atomic::AtomicU64::new(0);
        let (tx, _rx) = mpsc::channel(64);
        let run = |honour_reset: bool| -> Vec<ChunkPackets> {
            let mut pool = h264_pool(honour_reset);
            let mut out = Vec::new();
            for (idx, first_pts) in [(0usize, 0u64), (1, 4)] {
                let c = chunk(&cfg, idx, first_pts, 0, 4);
                match encode_chunk_to_packets(&cfg, &enc_config, c, &mut pool, &frames, &bytes, &tx).unwrap() {
                    ChunkOutcome::Encoded(p) => out.push(p),
                    ChunkOutcome::RequeuedOnMismatch { diff, .. } => panic!("unexpected mismatch: {diff}"),
                }
            }
            assert_eq!(pool.stats().reused, 1, "both variants reuse the session");
            out
        };
        let honest = run(true);
        assert!(honest[1].packets[0].is_keyframe, "a real reset opens chunk 1 with an IDR");
        let mutated = run(false);
        assert!(
            !mutated[1].packets[0].is_keyframe,
            "with reset stubbed to a no-op, chunk 1 predicts from chunk 0 — the IDR check catches it"
        );
    }

    /// A rung hop evicts; the packets are still right.
    #[test]
    fn a_rung_hop_evicts_and_rebuilds() {
        let a = worker_config(32, 16, 1000);
        let b = worker_config(16, 16, 1000);
        let mut pool = h264_pool(true);
        let frames = std::sync::atomic::AtomicU64::new(0);
        let bytes = std::sync::atomic::AtomicU64::new(0);
        let (tx, _rx) = mpsc::channel(64);
        for (cfg, idx) in [(&a, 0usize), (&b, 0), (&a, 1)] {
            let enc_config = super::super::build_enc_config(cfg);
            let c = chunk(cfg, idx, idx as u64 * 4, 0, 4);
            let ChunkOutcome::Encoded(p) =
                encode_chunk_to_packets(cfg, &enc_config, c, &mut pool, &frames, &bytes, &tx).unwrap()
            else {
                panic!("mismatch")
            };
            assert!(p.packets[0].is_keyframe);
            assert_eq!(p.packets.len(), 4);
        }
        assert_eq!(pool.stats(), PoolStats { built: 3, evicted: 2, ..Default::default() });
    }

    // ── The kept range ──────────────────────────────────────────────────

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
