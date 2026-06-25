//! GPU-only decode dispatch.
//!
//! Per the 2026-05-08 directive: every CPU decoder (openh264, libde265,
//! libvpx, rav1d, libmpeg2, libxvidcore, pure-Rust ProRes) was deleted
//! along with the legacy `FallbackDecoder` GPU→CPU fallover. The
//! production binary supports exactly two backends:
//!
//!   - NVDEC (NVIDIA, via libnvcuvid)
//!   - QSV   (Intel,  via libvpl + iHD)
//!
//! Hosts without one of those (no NVIDIA, no Intel Arc / Meteor Lake,
//! or a codec the local GPU can't decode) hard-fail at
//! [`create_decoder`]. There is no CPU decode path of any shape.

#[cfg(feature = "ffmpeg")]
pub mod ffmpeg;
pub mod nvdec;
pub mod qsv;

use crate::frame::{StreamInfo, VideoFrame};
use crate::gpu;
use anyhow::{Result, bail};

pub trait Decoder: Send {
    fn stream_info(&self) -> &StreamInfo;

    /// Feed one Annex-B (or codec-native — AV1 OBU, VP9 superframe) sample
    /// into the decoder. Implementations may buffer internally until
    /// `finish` is called or may decode eagerly and buffer produced
    /// frames. Pull frames via `decode_next` at any point.
    fn push_sample(&mut self, data: &[u8]) -> Result<()>;

    /// Signal end-of-stream. After this, no more `push_sample` calls;
    /// `decode_next` drains remaining frames.
    fn finish(&mut self) -> Result<()>;

    fn decode_next(&mut self) -> Result<Option<VideoFrame>>;
}

/// Truthy-string parse for env-var opt-outs. `1` / `true` / `yes` / `on`
/// / `y` / `t` (case-insensitive) all resolve true; anything else is
/// false. Mirrors the encode-side helper for symmetry.
fn env_flag_truthy(name: &str) -> bool {
    match std::env::var(name) {
        Ok(v) => {
            let v = v.to_ascii_lowercase();
            matches!(v.as_str(), "1" | "true" | "yes" | "on" | "y" | "t")
        }
        Err(_) => false,
    }
}

/// Per-codec NVDEC opt-out check. Mirrors the previous-stack
/// `DISABLE_NVDEC_<CODEC>` granular knob: `DISABLE_NVDEC=1` blocks every
/// codec, `DISABLE_NVDEC_H264=1` blocks just one. Used as a debugging
/// escape hatch when a specific codec/driver combo is misbehaving on
/// the active host (e.g. Blackwell + 4K H.264 silent-stall).
fn nvdec_disabled_for(codec_lower: &str) -> bool {
    if env_flag_truthy("DISABLE_NVDEC") {
        return true;
    }
    let codec_canonical = match codec_lower {
        "h264" | "avc1" | "avc" => "H264",
        "h265" | "hevc" | "hvc1" | "hev1" | "hvc2" | "hev2" => "HEVC",
        "vp8" => "VP8",
        "vp9" | "vp09" => "VP9",
        "av1" | "av01" => "AV1",
        "mpeg2" | "mpeg2video" => "MPEG2",
        "mpeg4" | "mp4v" => "MPEG4",
        _ => return false,
    };
    env_flag_truthy(&format!("DISABLE_NVDEC_{codec_canonical}"))
}

/// Codecs the NVDEC streaming dispatch supports.
fn nvdec_supports(codec_lower: &str) -> bool {
    matches!(
        codec_lower,
        "h264"
            | "avc1"
            | "avc"
            | "h265"
            | "hevc"
            | "hvc1"
            | "hev1"
            | "hvc2"
            | "hev2"
            | "vp8"
            | "vp9"
            | "vp09"
            | "av1"
            | "av01"
            | "mpeg2"
            | "mpeg2video"
            | "mpeg4"
            | "mp4v"
    )
}

/// Codecs the QSV (libvpl/iHD) decoder supports.
fn qsv_supports(codec_lower: &str) -> bool {
    matches!(
        codec_lower,
        "h264"
            | "avc1"
            | "avc"
            | "h265"
            | "hevc"
            | "hvc1"
            | "hev1"
            | "hvc2"
            | "hev2"
            | "vp9"
            | "vp09"
            | "av1"
            | "av01"
    )
}

/// Construct a hardware decoder for `codec`. NVIDIA GPUs win on tie
/// when both vendors are present (NVDEC is generally lower-latency on
/// the standard codec set + is what the production fleet has been
/// tuned against). When NVDEC is disabled per env-var or doesn't
/// support the codec, fall through to QSV. If neither fits, hard-fail
/// — there is no CPU fallback.
pub fn create_decoder(codec: &str, info: StreamInfo) -> Result<Box<dyn Decoder>> {
    create_decoder_on(codec, info, None)
}

/// Construct a decoder pinned to a specific `gpu_index` when one is
/// supplied. `None` preserves the legacy "pick the first matching
/// adapter" behaviour for one-shot callers (thumbnails, tests, benches)
/// that don't care about distributing work across physical GPUs.
///
/// The pipeline's per-rung decode pumps should ALWAYS pass `Some(idx)`
/// so each rung's decode session lands on a distinct adapter — without
/// this, every QSV session piles onto the first physical Intel card
/// regardless of what the GPU pool's lease said. See the project memo
/// on QSV multi-adapter session pinning.
pub fn create_decoder_on(
    codec: &str,
    info: StreamInfo,
    gpu_index: Option<u32>,
) -> Result<Box<dyn Decoder>> {
    let codec_lower = codec.to_ascii_lowercase();
    let gpus = gpu::detect_gpus();

    // Pick the device. If the caller specified gpu_index, honour it
    // (matching against `g.index`). Otherwise fall back to the first
    // of each vendor — the legacy behaviour for callers that don't
    // care about pinning.
    let nvidia = match gpu_index {
        Some(idx) => gpus
            .iter()
            .find(|g| matches!(g.vendor, gpu::GpuVendor::Nvidia) && g.index == idx),
        None => gpus
            .iter()
            .find(|g| matches!(g.vendor, gpu::GpuVendor::Nvidia)),
    };
    let intel = match gpu_index {
        Some(idx) => gpus
            .iter()
            .find(|g| matches!(g.vendor, gpu::GpuVendor::Intel) && g.index == idx),
        None => gpus
            .iter()
            .find(|g| matches!(g.vendor, gpu::GpuVendor::Intel)),
    };

    // NVIDIA / NVDEC first.
    if let Some(dev) = nvidia
        && nvdec_supports(&codec_lower)
        && !nvdec_disabled_for(&codec_lower)
    {
        tracing::info!(
            backend = "nvdec",
            codec = %codec_lower,
            width = info.width,
            height = info.height,
            gpu_index = dev.index,
            gpu_name = %dev.name,
            "NVDEC decoder engaged (GPU-only — no CPU fallback)"
        );
        eprintln!(
            "[decode] nvdec constructed for codec={} gpu_index={}",
            codec_lower, dev.index
        );
        return Ok(nvdec::NvdecDecoder::new(info, dev.index));
    }

    // Intel / QSV next.
    if let Some(dev) = intel
        && qsv_supports(&codec_lower)
    {
        tracing::info!(
            backend = "qsv",
            codec = %codec_lower,
            width = info.width,
            height = info.height,
            gpu_index = dev.index,
            gpu_name = %dev.name,
            "QSV (oneVPL/iHD) decoder engaged (GPU-only — no CPU fallback)"
        );
        eprintln!(
            "[decode] qsv constructed for codec={} gpu_index={}",
            codec_lower, dev.index
        );
        return Ok(Box::new(qsv::QsvDecoder::new(info, dev.index)?));
    }

    bail!(
        "no GPU decoder available for codec '{}' on this host \
         (NVIDIA GPUs cover h264/h265/vp8/vp9/av1/mpeg2/mpeg4; \
          Intel Arc/Meteor Lake+ covers h264/h265/vp9/av1). \
         CPU decoders were removed per the GPU-only directive.",
        codec_lower
    )
}
