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
//!
//! Ported from the transcoder microservice's `pipeline::decode_pump`.

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
}

/// Blocking decode loop, designed for `tokio::task::spawn_blocking`. Fans
/// every normalized frame out to all `senders`. If a sender's channel is
/// closed (its rung gave up) the pump continues with the rest; it stops only
/// when *every* sender is closed. `rt` bridges into the async `send().await`.
///
/// Returns the number of frames pushed.
pub fn run_shared_decode_pump_blocking(
    cfg: DecodePumpConfig,
    input_data: Bytes,
    senders: Vec<tokio::sync::mpsc::Sender<VideoFrame>>,
    rt: tokio::runtime::Handle,
) -> Result<u64> {
    let outcome = decode_loop(&cfg, input_data, &senders, &rt);
    // Drop senders so receivers wake and exit.
    drop(senders);
    outcome
}

fn decode_loop(
    cfg: &DecodePumpConfig,
    input_data: Bytes,
    senders: &[tokio::sync::mpsc::Sender<VideoFrame>],
    rt: &tokio::runtime::Handle,
) -> Result<u64> {
    let mut demuxer =
        streaming::demux_streaming(&input_data).context("demuxing input for shared decode pump")?;
    let mut decoder =
        decode::create_decoder_on(&cfg.codec_name, cfg.info_for_decoder.clone(), cfg.gpu_index)
            .context("creating decoder for shared decode pump")?;

    let mut frames_pushed: u64 = 0;
    'outer: loop {
        match demuxer
            .next_video_sample()
            .context("demuxing next video sample in shared decode pump")?
        {
            Some(sample) => {
                decoder
                    .push_sample(&sample.data)
                    .context("pushing sample to shared decode pump decoder")?;
                while let Some(frame) = decoder
                    .decode_next()
                    .context("decoding frame in shared decode pump")?
                {
                    let normalized = normalize_frame(cfg, frame)?;
                    if !fan_out(senders, normalized, rt)? {
                        break 'outer;
                    }
                    frames_pushed += 1;
                }
            }
            None => {
                decoder
                    .finish()
                    .context("decoder finish in shared decode pump")?;
                while let Some(frame) = decoder
                    .decode_next()
                    .context("decoding frame after finish in shared decode pump")?
                {
                    let normalized = normalize_frame(cfg, frame)?;
                    if !fan_out(senders, normalized, rt)? {
                        break;
                    }
                    frames_pushed += 1;
                }
                break;
            }
        }
    }

    Ok(frames_pushed)
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
    if !cfg.tonemap_to_sdr {
        // Passthrough / HDR output: preserve the source color + bit depth.
        return Ok(downsampled);
    }
    let normalized = colorspace::convert_to_sdr_bt709(&downsampled, &cfg.source_color_metadata)
        .context("shared decode pump colorspace convert (HDR-aware)")?;
    Ok(normalized)
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
