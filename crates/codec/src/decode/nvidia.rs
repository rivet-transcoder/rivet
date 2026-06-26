//! Unified NVIDIA NVDEC decode façade — one entry point over both backends.
//!
//! NVIDIA hosts decode through two implementations, picked per `(codec, bit
//! depth)` so callers don't have to branch:
//!
//! - **`shiguredo_nvcodec`** (the maintained wrapper, `nvidia` feature) for
//!   **8-bit** H.264 / HEVC / AV1 / VP8 / VP9.
//! - the **built-in NVDEC** (always compiled) for what the wrapper doesn't
//!   expose — **MPEG-2**, **MPEG-4 Part 2**, and **10-bit** (P016), including
//!   10-bit HEVC Main10 / HDR.
//!
//! The bit-depth split is load-bearing: the shiguredo wrapper is **NV12-only**,
//! so a 10-bit source routed there would be silently truncated to 8-bit.
//! [`create`] keeps 10-bit sources on the built-in P016 path so HDR survives
//! decode (e.g. for `ColorPolicy::Hdr10`/`Hlg`/`Passthrough`).

use anyhow::Result;

use super::Decoder;
use crate::frame::{PixelFormat, StreamInfo};

/// Construct the best NVDEC decoder for `info`. 8-bit modern codecs use the
/// shiguredo wrapper when the `nvidia` feature is built; 10-bit sources and
/// MPEG-2 / MPEG-4 Part 2 use the built-in NVDEC (which decodes P016 rather than
/// truncating to NV12). The caller has already confirmed an NVIDIA device is
/// present, the codec is NVDEC-decodable, and NVDEC isn't disabled for it.
pub fn create(codec_lower: &str, info: StreamInfo, gpu_index: u32) -> Result<Box<dyn Decoder>> {
    let ten_bit = matches!(info.pixel_format, PixelFormat::Yuv420p10le);

    // 8-bit modern codecs → the maintained shiguredo wrapper.
    #[cfg(feature = "nvidia")]
    if !ten_bit && super::nvcodec_dec::supports(codec_lower) {
        tracing::info!(
            backend = "nvcodec",
            codec = %codec_lower,
            gpu_index,
            "NVDEC (shiguredo_nvcodec) decoder engaged"
        );
        return Ok(Box::new(super::nvcodec_dec::NvcodecDecoder::new(
            info, gpu_index,
        )?));
    }

    // Everything the wrapper can't do — MPEG-2 / MPEG-4 Part 2, and any 10-bit
    // source (P016) — stays on the built-in NVDEC.
    tracing::info!(
        backend = "nvdec-builtin",
        codec = %codec_lower,
        ten_bit,
        gpu_index,
        "built-in NVDEC decoder engaged (MPEG-2/4 + 10-bit P016 path)"
    );
    Ok(super::nvdec::NvdecDecoder::new(info, gpu_index))
}
