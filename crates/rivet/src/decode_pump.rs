//! Shared source decode pump.
//!
//! One pump per job (not per rung): demux + decode the source **once**, run
//! the rung-agnostic per-frame work (4:4:4 → 4:2:0 downsample + HDR tonemap),
//! and fan the normalized frame out to N per-rung mpsc channels via cheap
//! `VideoFrame::clone()` (the inner `Bytes` is `Arc`-backed).
//!
//! Per-rung scaling + encoding consume from those channels. Eliminating the
//! redundant per-rung decode is the whole point — a 5-rung ladder decodes the
//! source once, not five times. The cost: the slowest rung backpressures the
//! pump (usually the largest rung, whose encoder is slowest).

use std::time::Instant;

use anyhow::{Context, Result};
use bytes::Bytes;

use codec::frame::{ColorMetadata, PixelFormat, VideoFrame};
use codec::{colorspace, decode};
use container::streaming;

/// Configuration for one decode pump.
#[derive(Clone)]
pub struct DecodePumpConfig {
    /// Source video codec label (e.g. `"h264"`).
    pub codec_name: String,
    /// Stream info handed to the decoder.
    pub info_for_decoder: codec::frame::StreamInfo,
    /// Source color metadata (drives HDR-aware tonemap vs SDR passthrough).
    pub source_color_metadata: ColorMetadata,
    /// Source pixel format.
    pub source_pixel_format: PixelFormat,
    /// Whether to run the 4:4:4 → 4:2:0 downsample per frame.
    pub needs_downsample: bool,
    /// Tonemap policy (from the [`OutputSpec`](crate::spec::OutputSpec)): when
    /// `true`, HDR (PQ/HLG) sources are mapped down to 8-bit SDR BT.709; when
    /// `false`, the source color/transfer/bit-depth passes through unchanged.
    /// The pump does not decide this on its own — the caller sets it from the
    /// spec's [`ColorPolicy`](crate::spec::ColorPolicy).
    pub tonemap_to_sdr: bool,
    /// Pin the decoder to this physical GPU; `None` = first matching adapter.
    pub gpu_index: Option<u32>,
    /// Prepared per-frame video filter chain (crop/pad/flip/rotate/grayscale/
    /// overlay/colour), applied after colorspace normalize and before the frame
    /// is fanned out to the per-rung scalers. Overlay images are loaded once at
    /// prepare time. `Arc` so the per-GPU pump configs clone it cheaply.
    pub filters: std::sync::Arc<codec::filter::FilterChain>,
}

/// One clip of a splice: a decode config, its source bytes, and the **source
/// frame range** to keep. The first `start_frame` decoded frames are dropped
/// (the trim in-point); decoding stops once the source index reaches
/// `end_frame` (exclusive — the trim out-point). `end_frame = None` keeps the
/// clip to its end. A single full-range clip (`start_frame = 0`,
/// `end_frame = None`) is a plain, un-spliced transcode.
#[derive(Clone)]
pub struct ClipSource {
    pub cfg: DecodePumpConfig,
    pub input: Bytes,
    pub start_frame: u64,
    pub end_frame: Option<u64>,
}

impl ClipSource {
    /// A whole clip, no trim.
    pub fn whole(cfg: DecodePumpConfig, input: Bytes) -> Self {
        Self { cfg, input, start_frame: 0, end_frame: None }
    }
}

/// Single-input decode pump (no trim, no concat) — the common case. A thin
/// wrapper over [`run_spliced_decode_pump_blocking`] with one whole clip.
pub fn run_shared_decode_pump_blocking(
    cfg: DecodePumpConfig,
    input_data: Bytes,
    senders: Vec<tokio::sync::mpsc::Sender<VideoFrame>>,
    rt: tokio::runtime::Handle,
) -> Result<u64> {
    run_spliced_decode_pump_blocking(vec![ClipSource::whole(cfg, input_data)], senders, rt)
}

/// Spliced decode pump, designed for `tokio::task::spawn_blocking`. Decodes
/// each clip in order, **drops** frames outside the clip's `[start_frame,
/// end_frame)` source range (trim), and fans the kept frames out to all
/// `senders` **continuously across clips** (concat). Because the muxer numbers
/// output frames by count — not by source PTS — the join is automatically
/// gap-free and the timeline is zero-based, with no PTS rewriting.
///
/// If a sender's channel is closed (its rung gave up) the pump keeps going with
/// the rest; it stops only when *every* sender is closed. `rt` bridges into the
/// async `send().await`. Returns the total number of frames emitted.
pub fn run_spliced_decode_pump_blocking(
    clips: Vec<ClipSource>,
    senders: Vec<tokio::sync::mpsc::Sender<VideoFrame>>,
    rt: tokio::runtime::Handle,
) -> Result<u64> {
    let mut total: u64 = 0;
    let result = (|| {
        for (clip_idx, clip) in clips.iter().enumerate() {
            match decode_clip(clip, &senders, &rt, &mut total)
                .with_context(|| format!("decoding splice clip {clip_idx}"))?
            {
                Flow::Continue => {}
                Flow::AllReceiversClosed => break,
            }
        }
        Ok(total)
    })();
    // Drop senders so receivers wake and exit.
    drop(senders);
    result
}

enum Flow {
    Continue,
    AllReceiversClosed,
}

/// Decode one clip, applying its trim range, fanning kept frames to `senders`
/// and advancing the shared output counter `total`.
fn decode_clip(
    clip: &ClipSource,
    senders: &[tokio::sync::mpsc::Sender<VideoFrame>],
    rt: &tokio::runtime::Handle,
    total: &mut u64,
) -> Result<Flow> {
    let cfg = &clip.cfg;
    let mut demuxer =
        streaming::demux_streaming(&clip.input).context("demuxing clip for decode pump")?;
    let mut decoder =
        decode::create_decoder_on(&cfg.codec_name, cfg.info_for_decoder.clone(), cfg.gpu_index)
            .context("creating decoder for decode pump")?;

    // Source-frame index within THIS clip — drives the trim decision.
    let mut src_idx: u64 = 0;
    loop {
        match demuxer
            .next_video_sample()
            .context("demuxing next video sample in decode pump")?
        {
            Some(sample) => {
                decoder
                    .push_sample(&sample.data)
                    .context("pushing sample to decode pump decoder")?;
                while let Some(frame) =
                    decoder.decode_next().context("decoding frame in decode pump")?
                {
                    match handle_frame(clip, cfg, frame, senders, rt, &mut src_idx, total)? {
                        FrameAction::Continue => {}
                        FrameAction::ClipDone => return Ok(Flow::Continue),
                        FrameAction::StopAll => return Ok(Flow::AllReceiversClosed),
                    }
                }
            }
            None => {
                decoder.finish().context("decoder finish in decode pump")?;
                while let Some(frame) = decoder
                    .decode_next()
                    .context("decoding frame after finish in decode pump")?
                {
                    match handle_frame(clip, cfg, frame, senders, rt, &mut src_idx, total)? {
                        FrameAction::Continue => {}
                        FrameAction::ClipDone => return Ok(Flow::Continue),
                        FrameAction::StopAll => return Ok(Flow::AllReceiversClosed),
                    }
                }
                break;
            }
        }
    }
    Ok(Flow::Continue)
}

enum FrameAction {
    Continue,
    ClipDone,
    StopAll,
}

/// Apply the clip's trim range to one decoded frame: drop frames before the
/// in-point, signal `ClipDone` at the out-point, otherwise normalize + fan out.
fn handle_frame(
    clip: &ClipSource,
    cfg: &DecodePumpConfig,
    frame: VideoFrame,
    senders: &[tokio::sync::mpsc::Sender<VideoFrame>],
    rt: &tokio::runtime::Handle,
    src_idx: &mut u64,
    total: &mut u64,
) -> Result<FrameAction> {
    if clip.end_frame.is_some_and(|end| *src_idx >= end) {
        return Ok(FrameAction::ClipDone); // reached the out-point
    }
    if *src_idx >= clip.start_frame {
        let normalized = normalize_frame(cfg, frame)?;
        if !fan_out(senders, normalized, rt)? {
            return Ok(FrameAction::StopAll);
        }
        *total += 1;
    }
    *src_idx += 1;
    Ok(FrameAction::Continue)
}

/// Rung-agnostic per-frame work: 4:4:4 → 4:2:0 downsample (if needed) then,
/// when the spec's color policy asks for it (`tonemap_to_sdr`), an HDR-aware
/// colorspace convert (tonemap PQ/HLG → SDR BT.709, identity for SDR). When the
/// policy is passthrough/HDR, the downsampled source is forwarded unchanged.
/// Per-rung scaling is NOT done here.
fn normalize_frame(cfg: &DecodePumpConfig, frame: VideoFrame) -> Result<VideoFrame> {
    let downsampled = if cfg.needs_downsample {
        colorspace::downsample_444_to_420_frame(&frame)
            .context("shared decode pump 4:4:4 → 4:2:0 downsample")?
    } else {
        frame
    };
    let normalized = if !cfg.tonemap_to_sdr {
        // Passthrough / HDR output: preserve the source color + bit depth.
        downsampled
    } else {
        colorspace::convert_to_sdr_bt709(&downsampled, &cfg.source_color_metadata)
            .context("shared decode pump colorspace convert (HDR-aware)")?
    };
    // Video filters (crop/pad/flip/rotate/grayscale/overlay/colour) run on the
    // normalized 4:2:0 frame, before the per-rung scalers see it.
    if cfg.filters.is_empty() {
        Ok(normalized)
    } else {
        cfg.filters.apply(normalized).context("shared decode pump video filters")
    }
}

/// Number of frames timed per candidate when benchmarking decoders. Chosen so
/// the measurement amortises driver init yet stays well under a second per
/// candidate even on a modest GPU.
pub const DECODE_BENCH_FRAMES: usize = 120;

/// Benchmark each candidate GPU by decoding a short prefix of `input` on it and
/// return the fastest `gpu_index` (what `--decode-with-fastest` pins the pump
/// to). Construction + first-frame latency is excluded — the clock starts after
/// a small warmup — so the number reflects steady-state decode throughput, not
/// driver init. Candidates that fail to construct or decode are skipped;
/// returns `None` if no candidate produced frames or fewer than two candidates
/// were given (nothing to choose).
pub fn fastest_decode_gpu(
    codec_name: &str,
    info: &codec::frame::StreamInfo,
    input: &Bytes,
    candidates: &[u32],
    measure_frames: usize,
) -> Option<u32> {
    if candidates.len() < 2 {
        return candidates.first().copied();
    }
    let mut best: Option<(u32, f64)> = None;
    for &gpu in candidates {
        match bench_decode_gpu(codec_name, info, input, gpu, measure_frames) {
            Ok(Some(fps)) => {
                tracing::info!(
                    gpu_index = gpu,
                    fps = format!("{fps:.1}"),
                    "decode-with-fastest: benchmarked candidate"
                );
                if best.is_none_or(|(_, b)| fps > b) {
                    best = Some((gpu, fps));
                }
            }
            Ok(None) => {
                tracing::warn!(gpu_index = gpu, "decode-with-fastest: no frames; skipping candidate")
            }
            Err(e) => tracing::warn!(
                gpu_index = gpu,
                error = %e,
                "decode-with-fastest: bench failed; skipping candidate"
            ),
        }
    }
    if let Some((gpu, fps)) = best {
        tracing::info!(
            gpu_index = gpu,
            fps = format!("{fps:.1}"),
            "decode-with-fastest: selected fastest decode GPU"
        );
    }
    best.map(|(g, _)| g)
}

/// Decode up to `measure_frames` frames (after an 8-frame warmup) from `input`
/// on `gpu`, returning the measured fps — or `None` if it produced no frames.
fn bench_decode_gpu(
    codec_name: &str,
    info: &codec::frame::StreamInfo,
    input: &Bytes,
    gpu: u32,
    measure_frames: usize,
) -> Result<Option<f64>> {
    const WARMUP: usize = 8;
    let target = WARMUP + measure_frames;
    let mut demuxer = streaming::demux_streaming(input).context("demux for decode bench")?;
    let mut decoder = decode::create_decoder_on(codec_name, info.clone(), Some(gpu))
        .context("create decoder for bench")?;
    let mut decoded = 0usize;
    let mut clock: Option<Instant> = None;
    'outer: loop {
        match demuxer.next_video_sample().context("bench next sample")? {
            Some(s) => {
                decoder.push_sample(&s.data).context("bench push")?;
                while decoder.decode_next().context("bench decode")?.is_some() {
                    decoded += 1;
                    if decoded == WARMUP {
                        clock = Some(Instant::now());
                    }
                    if decoded >= target {
                        break 'outer;
                    }
                }
            }
            None => {
                decoder.finish().context("bench finish")?;
                while decoder.decode_next().context("bench drain")?.is_some() {
                    decoded += 1;
                    if decoded == WARMUP {
                        clock = Some(Instant::now());
                    }
                    if decoded >= target {
                        break 'outer;
                    }
                }
                break;
            }
        }
    }
    let measured = decoded.saturating_sub(WARMUP);
    Ok(match clock {
        Some(t) if measured > 0 => {
            let secs = t.elapsed().as_secs_f64();
            (secs > 0.0).then_some(measured as f64 / secs)
        }
        // Tiny clip (< WARMUP+1 frames): every candidate decodes the same few
        // frames, so return the count — equal across candidates, first wins.
        _ => (decoded > 0).then_some(decoded as f64),
    })
}

/// Fan one frame out to every sender. Cloning `VideoFrame` is cheap (inner
/// `Bytes` is `Arc`-backed). Returns `false` only if EVERY sender is closed.
fn fan_out(
    senders: &[tokio::sync::mpsc::Sender<VideoFrame>],
    frame: VideoFrame,
    rt: &tokio::runtime::Handle,
) -> Result<bool> {
    let mut any_alive = false;
    for (idx, sender) in senders.iter().enumerate() {
        let frame_clone = frame.clone();
        let sender = sender.clone();
        let accepted = rt.block_on(async move { sender.send(frame_clone).await });
        match accepted {
            Ok(()) => any_alive = true,
            Err(_) => {
                tracing::warn!(rung_idx = idx, "shared decode pump: rung dropped its receiver");
            }
        }
    }
    Ok(any_alive)
}
