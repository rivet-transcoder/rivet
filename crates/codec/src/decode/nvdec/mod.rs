//! **Legacy fallback** — retained for the default-feature build and
//! as a failover target when the `codec/ffmpeg` feature is enabled.
//! New dispatch prefers `super::ffmpeg::FfmpegDecoder` which wires
//! `hwaccel=cuda` onto libavcodec to drive the same NVDEC silicon
//! with a battle-tested frame pipeline (see the 2026-04-19 migration
//! in `mod.rs::create_decoder`). This custom libnvcuvid wrapper remains
//! engaged for the default-feature build (no FFmpeg dep) and as a
//! failover when the FFmpeg path errors.
//!
//! NVDEC hardware video decoder via NVIDIA CUDA Video Decoder API.
//!
//! Loads libcuda and libnvcuvid at runtime via dlopen. No compile-time
//! CUDA SDK needed — the vendored headers in `vendor/nvidia/` are the
//! authoritative reference for the struct layouts and function
//! signatures used here.
//!
//! Flow:
//! 1. cuInit + cuCtxCreate                      (driver init)
//! 2. cuvidCreateVideoParser                    (stateless parser)
//! 3. per sample: cuCtxPushCurrent + cuvidParseVideoData + cuCtxPopCurrent
//!    - pfn_sequence_callback: cuvidCreateDecoder (first time)
//!    - pfn_decode_picture:    cuvidDecodePicture
//!    - pfn_display_picture:   cuvidMapVideoFrame + cuMemcpy2D then push
//!      NV12 bytes into FrameCollector
//! 4. cuvidDestroyVideoParser + cuvidDestroyDecoder + cuCtxDestroy
//!
//! Library-lifetime note: CUDA + CUVID libraries are stored as fields on
//! NvdecDecoder and declared LAST so they drop after every resource that
//! references them (Rust drops struct fields in source order — Reference
//! §10.8). All FFI fn pointers captured into CallbackState are borrowed
//! from libraries whose Library handles outlive the callback dispatch.
//!
//! Thread-safety note: cuCtxCreate makes the context current only on the
//! calling thread. Every cuvid* call happens under cuCtxPushCurrent /
//! cuCtxPopCurrent so a tokio worker that migrates between threads still
//! has the right context bound before touching the decoder.
//!
//! ## Module layout
//!
//! | File             | Contents                                      |
//! |------------------|-----------------------------------------------|
//! | `mod.rs`         | Public error type + module declarations + re-exports |
//! | `ffi.rs`         | FFI structs, type aliases, constants, size witnesses |
//! | `state.rs`       | `DecodedFrame`, `FrameCollector`, `CallbackState`, `CtxScope` |
//! | `convert.rs`     | Codec-string → cuvid id, format validate, NV12/P016 deinterleave |
//! | `callbacks.rs`   | `sequence_callback`, `decode_callback`, `display_callback`, `get_operating_point_callback` |
//! | `eager.rs`       | `NvdecDecoder` (eager / post-decode cursor)   |
//! | `push.rs`        | `NvdecPushDecoder` (buffer-until-finish wrapper)|
//! | `streaming.rs`   | `NvdecStreamingDecoder` (Squad-36 incremental parse) + `NvdecInitErrorDecoder` |

mod callbacks;
mod convert;
mod eager;
mod ffi;
mod push;
mod state;
mod streaming;

use std::os::raw::c_int;

// ─── Typed errors surfaced to the caller ──────────────────────────
//
// `anyhow::Error::downcast_ref::<NvdecError>()` lets callers (and tests)
// pattern-match on specific NVDEC reject reasons without string-matching
// the display message. The decode_next / new paths wrap these in
// anyhow::Error so the Decoder trait signature stays unchanged.
//
// Reviewer note (codec-review-2 HIGH-1, HIGH-2): previously any of
// these rejects surfaced as an opaque "NVDEC produced no frames: <string>"
// anyhow and the pipeline couldn't tell "4:2:2 unsupported" from
// "driver OOM". A typed variant keeps the CPU-fallback decision in
// decode/mod.rs explainable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NvdecError {
    /// Chroma subsampling reported by the CUVID parser is not one of
    /// the formats this backend supports. Currently only 4:2:0 (value 1)
    /// passes; monochrome (0), 4:2:2 (2), 4:4:4 (3) all produce this.
    UnsupportedChroma {
        chroma_format: c_int,
        label: &'static str,
        width: u32,
        height: u32,
    },
    /// Bit depth outside the 8/10/12-bit 4:2:0 envelope. HEVC Rext
    /// 14-bit / 16-bit content lands here; the existing NV12/P016 copy
    /// math does not generalize.
    UnsupportedPixelFormat { bit_depth: u8 },
    /// The GPU's NVDEC reported (via `cuvidGetDecoderCaps`) that it can't decode
    /// this codec/chroma/bit-depth combination, or the frame exceeds the
    /// hardware's max decode dimensions. Distinct from the format rejects above:
    /// those are formats we never support; this is a per-GPU hardware limit.
    UnsupportedByHardware { reason: String },
}

impl std::fmt::Display for NvdecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedChroma {
                chroma_format,
                label,
                width,
                height,
            } => write!(
                f,
                "NVDEC reject: chroma_format={} ({}) at {}x{} — only 4:2:0 supported",
                chroma_format, label, width, height
            ),
            Self::UnsupportedPixelFormat { bit_depth } => write!(
                f,
                "NVDEC reject: {}-bit content — only 8/10/12-bit 4:2:0 supported",
                bit_depth
            ),
            Self::UnsupportedByHardware { reason } => {
                write!(f, "NVDEC reject: GPU capability — {reason}")
            }
        }
    }
}

impl std::error::Error for NvdecError {}

// ─── Public re-exports ────────────────────────────────────────────
//
// Everything exported from `nvdec.rs` before the split is re-exported
// here so all call sites (`use crate::decode::nvdec::NvdecDecoder`,
// `use codec::decode::nvdec::validate_format`, etc.) continue to
// resolve unchanged — the directory split is transparent to consumers.
pub use convert::{
    OutputGeometry, deinterleave_p016_to_yuv420p10le, output_geometry, validate_format,
};
pub use eager::NvdecDecoder;
pub use push::NvdecPushDecoder;
pub use streaming::NvdecStreamingDecoder;
