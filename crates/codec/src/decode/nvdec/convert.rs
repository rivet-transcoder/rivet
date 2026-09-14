//! Pure-Rust conversion and validation helpers:
//! codec-string → CUVID codec-ID mapping, format validation,
//! NV12/P016 deinterleave, and decoded-frame → VideoFrame conversion.

use bytes::Bytes;
use std::os::raw::c_int;

use crate::frame::{PixelFormat, VideoFrame};
use super::NvdecError;
use super::ffi::{
    CUVID_AV1, CUVID_CHROMA_420, CUVID_H264, CUVID_HEVC, CUVID_MPEG2, CUVID_MPEG4, CUVID_VP8,
    CUVID_VP9,
};
use super::state::DecodedFrame;

pub fn codec_to_cuvid(codec: &str) -> Option<c_int> {
    match codec {
        "h264" | "avc1" | "avc" => Some(CUVID_H264),
        "h265" | "hevc" | "hvc1" | "hev1" => Some(CUVID_HEVC),
        "vp8" => Some(CUVID_VP8),
        "vp9" | "vp09" => Some(CUVID_VP9),
        "av1" | "av01" => Some(CUVID_AV1),
        "mpeg2" | "mpeg2video" => Some(CUVID_MPEG2),
        "mpeg4" | "mp4v" => Some(CUVID_MPEG4),
        _ => None,
    }
}

/// Pure-Rust validator for the subset of CUVIDEOFORMAT fields this
/// backend cares about. Extracted out of `sequence_callback` so the
/// chroma / bit-depth reject matrix can be unit-tested without
/// spinning up a GPU context. Returns `None` when the format is
/// acceptable for NVDEC decoding on this backend.
///
/// Contract (codec-review-2 HIGH-1 + HIGH-2):
///   chroma_format values → action
///     0 (Monochrome) → Err UnsupportedChroma
///     1 (4:2:0)      → accept (subject to bit depth check)
///     2 (4:2:2)      → Err UnsupportedChroma
///     3 (4:4:4)      → Err UnsupportedChroma
///   bit_depth_luma_minus8 values → action
///     0 (8-bit)      → accept (NV12 surface)
///     2 (10-bit)     → accept (P016 surface)
///     4 (12-bit)     → accept (P016 surface, shares wire format)
///     >4             → Err UnsupportedPixelFormat
pub fn validate_format(
    chroma_format: c_int,
    bit_depth_luma_minus8: u8,
    coded_width: u32,
    coded_height: u32,
) -> Option<NvdecError> {
    if chroma_format != CUVID_CHROMA_420 {
        let label: &'static str = match chroma_format {
            0 => "Monochrome",
            2 => "4:2:2",
            3 => "4:4:4",
            _ => "unknown",
        };
        return Some(NvdecError::UnsupportedChroma {
            chroma_format,
            label,
            width: coded_width,
            height: coded_height,
        });
    }
    let bit_depth = bit_depth_luma_minus8 + 8;
    if bit_depth > 12 {
        return Some(NvdecError::UnsupportedPixelFormat { bit_depth });
    }
    None
}

/// The picture NVDEC hands back, worked out from the parser's
/// `CUVIDEOFORMAT`: the stream's display area — H.264 frame cropping, the
/// HEVC conformance window, the AV1 / VP9 frame size — and never the padded
/// coded surface.
///
/// The coded surface is the picture rounded up to the codec's block size
/// (16 rows for H.264 macroblocks, the CTB for HEVC): a 640x360 stream is
/// coded 640x368, 1080p is coded 1088. The driver reports both. The frame
/// the pipeline sees has to be the display rectangle, because everything
/// downstream sizes itself from the container (640x360) and a 368-row frame
/// gets resampled to fit — a vertical squash by 8/368 that cost 20 dB on
/// every job decoded through NVDEC.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OutputGeometry {
    /// The coded (padded) surface the decoder is created with.
    pub coded_width: u32,
    /// See `coded_width`.
    pub coded_height: u32,
    /// The display rectangle inside the coded surface, in luma samples,
    /// right and bottom exclusive. Handed to the decoder as its
    /// `display_area`, so its post-processor crops on the way out.
    pub display_left: u16,
    /// See `display_left`.
    pub display_top: u16,
    /// See `display_left`.
    pub display_right: u16,
    /// See `display_left`.
    pub display_bottom: u16,
    /// The output picture: the display rectangle's size, rounded up to even
    /// because the decoder's post-processor wants an even target (ffmpeg's
    /// cuviddec does the same). A 4:2:0 H.264 / HEVC crop is even already;
    /// only an odd AV1 frame rounds.
    pub width: u32,
    /// See `width`.
    pub height: u32,
    /// The driver reported no usable display area (empty, inverted, or
    /// outside the coded surface) and the coded size was taken instead. The
    /// caller logs it: a padded picture must never be silent.
    pub coded_fallback: bool,
}

/// Work out [`OutputGeometry`] from the six `CUVIDEOFORMAT` fields.
pub fn output_geometry(
    coded_width: u32,
    coded_height: u32,
    display_left: i32,
    display_top: i32,
    display_right: i32,
    display_bottom: i32,
) -> OutputGeometry {
    let usable = display_left >= 0
        && display_top >= 0
        && display_right > display_left
        && display_bottom > display_top
        && i64::from(display_right) <= i64::from(coded_width)
        && i64::from(display_bottom) <= i64::from(coded_height)
        // The create-info rectangle fields are i16.
        && display_right <= i32::from(i16::MAX)
        && display_bottom <= i32::from(i16::MAX);
    let (left, top, right, bottom) = if usable {
        (
            display_left as u32,
            display_top as u32,
            display_right as u32,
            display_bottom as u32,
        )
    } else {
        (0, 0, coded_width, coded_height)
    };
    let even = |v: u32| (v + 1) & !1;
    OutputGeometry {
        coded_width,
        coded_height,
        display_left: left as u16,
        display_top: top as u16,
        display_right: right as u16,
        display_bottom: bottom as u16,
        width: even(right - left),
        height: even(bottom - top),
        coded_fallback: !usable,
    }
}

/// Pure-Rust P016 → Yuv420p10le deinterleave + 10-bit normalization.
/// Extracted out of `decode_next` so the right-shift, UV interleave,
/// and odd-dimension handling can be unit-tested without a GPU.
///
/// Input layout (`p016_bytes`):
///   Y plane: `w * h` samples × 2 bytes LE, 10-bit value in the HIGH
///            bits of each u16 (low 6 bits zero per SDK).
///   UV plane: `ceil(w/2) * ceil(h/2)` interleaved UV pairs, each pair
///             is 4 bytes (U u16 LE + V u16 LE), same high-bit layout.
///
/// Output layout (`Vec<u8>`, little-endian u16 packed):
///   Y plane: `w * h * 2` bytes, 10-bit value in the LOW bits.
///   U plane: `ceil(w/2) * ceil(h/2) * 2` bytes, 10-bit low bits.
///   V plane: `ceil(w/2) * ceil(h/2) * 2` bytes, 10-bit low bits.
///
/// 12-bit content also uses this path; the >>6 shift clips to 10-bit
/// range which is what the downstream 10-bit pipeline expects.
pub fn deinterleave_p016_to_yuv420p10le(p016_bytes: &[u8], w: usize, h: usize) -> Vec<u8> {
    let cw = w.div_ceil(2);
    let ch = h.div_ceil(2);
    let uv_pairs = cw * ch;
    let y_bytes = w * h * 2;
    let mut out = Vec::with_capacity(y_bytes + uv_pairs * 4);

    // Y plane: u16 LE samples, right-shift by 6 and re-emit LE.
    let y_src = &p016_bytes[..y_bytes.min(p016_bytes.len())];
    for chunk in y_src.chunks_exact(2) {
        let sample = u16::from_le_bytes([chunk[0], chunk[1]]);
        out.extend_from_slice(&(sample >> 6).to_le_bytes());
    }
    if out.len() < y_bytes {
        out.resize(y_bytes, 0);
    }

    // UV interleave: pair stride = 4 bytes (U u16 LE, V u16 LE).
    if p016_bytes.len() > y_bytes {
        let uv = &p016_bytes[y_bytes..];
        let mut u = Vec::with_capacity(uv_pairs * 2);
        let mut v = Vec::with_capacity(uv_pairs * 2);
        for i in 0..uv_pairs {
            let base = i * 4;
            if base + 3 < uv.len() {
                let us = u16::from_le_bytes([uv[base], uv[base + 1]]) >> 6;
                let vs = u16::from_le_bytes([uv[base + 2], uv[base + 3]]) >> 6;
                u.extend_from_slice(&us.to_le_bytes());
                v.extend_from_slice(&vs.to_le_bytes());
            }
        }
        out.extend_from_slice(&u);
        out.extend_from_slice(&v);
    }
    out
}

/// Convert one `DecodedFrame` (NV12 or P016 bytes) into a `VideoFrame`.
/// Shared between the eager `NvdecDecoder` (Vec drain) and the
/// streaming `NvdecStreamingDecoder` (VecDeque drain) paths so the
/// deinterleave / planar conversion has a single source of truth.
pub fn decoded_frame_to_video_frame(frame: &DecodedFrame) -> VideoFrame {
    let w = frame.width as usize;
    let h = frame.height as usize;
    // Round up to keep odd-sized chroma planes intact (M-A10). For
    // subsampled 4:2:0, chroma dimensions are ceil(w/2) × ceil(h/2).
    let cw = w.div_ceil(2);
    let ch = h.div_ceil(2);
    let uv_pairs = cw * ch;

    let (yuv, pixel_format) = if frame.bit_depth_minus8 > 0 {
        // P016 → Yuv420p10le — routed through the pure-Rust
        // helper so the deinterleave + 10-bit normalize path has
        // unit coverage (codec-review-2 HIGH-2). The helper
        // right-shifts each u16 sample by 6 so the 10-bit value
        // lands in the LOW bits of the emitted LE u16, matching
        // what the encoder / colorspace consumer expects.
        let _ = uv_pairs; // silence unused warn on this branch
        let out = deinterleave_p016_to_yuv420p10le(&frame.nv12, w, h);
        (out, PixelFormat::Yuv420p10le)
    } else {
        // NV12 → Yuv420p. 1 byte per sample, interleaved UV pair
        // stride is 2 bytes.
        let y_size = w * h;
        let mut out = Vec::with_capacity(y_size + uv_pairs * 2);
        out.extend_from_slice(&frame.nv12[..y_size.min(frame.nv12.len())]);
        if frame.nv12.len() > y_size {
            let uv = &frame.nv12[y_size..];
            let mut u = Vec::with_capacity(uv_pairs);
            let mut v = Vec::with_capacity(uv_pairs);
            for i in 0..uv_pairs {
                if i * 2 + 1 < uv.len() {
                    u.push(uv[i * 2]);
                    v.push(uv[i * 2 + 1]);
                }
            }
            out.extend_from_slice(&u);
            out.extend_from_slice(&v);
        }
        (out, PixelFormat::Yuv420p)
    };

    VideoFrame::new(
        Bytes::from(yuv),
        frame.width,
        frame.height,
        pixel_format,
        frame.color_space,
        frame.timestamp,
    )
}
