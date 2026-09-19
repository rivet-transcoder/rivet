//! Pixel-format detection from codec sequence headers.
//!
//! Given raw bitstream samples (the same `Vec<Vec<u8>>` our decoders
//! consume), parse just enough of the first sequence header to
//! extract chroma subsampling + luma bit depth, then map to our
//! PixelFormat enum.
//!
//! Why not use the full decoder: our CPU decoders (H.264 openh264,
//! HEVC Rust, VP9 Rust, rav1d AV1) each have their own parser
//! entry points, but none of them expose a "just probe the format"
//! API. NVDEC's sequence_callback tells us, but only after decode
//! starts. This module gives the pipeline a fast, codec-agnostic
//! probe path that runs before decoder construction.

use crate::PixelFormat;

mod av1;
mod bitreader;
mod h264;
mod hevc;
mod mpeg2;

#[cfg(test)]
mod tests;

pub use av1::*;
pub use h264::*;
pub use hevc::*;
pub use mpeg2::*;

/// Detect pixel format from the first sequence header in `samples`.
/// Falls back to Yuv420p on any parse failure — that matches the
/// previous hard-coded behavior so a bad probe doesn't block the
/// transcode, just the probe payload accuracy.
pub fn detect(codec: &str, samples: &[Vec<u8>]) -> PixelFormat {
    if samples.is_empty() {
        return PixelFormat::Yuv420p;
    }

    let result = match codec.to_lowercase().as_str() {
        "h264" | "avc1" | "avc" => h264::detect_h264(&samples[0]),
        "h265" | "hevc" | "hvc1" | "hev1" => hevc::detect_hevc(&samples[0]),
        "vp9" | "vp09" => detect_vp9(&samples[0]),
        "av1" | "av01" => av1::detect_av1(&samples[0]),
        _ => None,
    };

    result.unwrap_or(PixelFormat::Yuv420p)
}

/// Public entry point — dispatch by codec and return `Some((width,
/// height))` if the sequence header in `samples[0]` is parseable,
/// `None` otherwise.
///
/// Callers should treat `None` as "keep the existing width/height" —
/// it's load-bearing for MPEG-TS where `StreamInfo` would otherwise
/// carry `0×0`, but a parse failure on MP4/MKV (which already have
/// width/height in the sample-entry / track-header) is a no-op.
pub fn detect_dims(codec: &str, samples: &[Vec<u8>]) -> Option<(u32, u32)> {
    if samples.is_empty() {
        return None;
    }
    let sample = &samples[0];
    match codec.to_lowercase().as_str() {
        "h264" | "avc1" | "avc" | "avc3" => {
            let info = parse_h264_sps(sample)?;
            Some((info.width?, info.height?))
        }
        "h265" | "hevc" | "hvc1" | "hev1" | "hvc2" | "hev2" => {
            let info = parse_hevc_sps(sample)?;
            Some((info.width?, info.height?))
        }
        "mpeg2" | "mpeg2video" | "mp2v" => {
            let info = parse_mpeg2_sequence_header(sample)?;
            Some((info.width, info.height))
        }
        _ => None,
    }
}

// ─── VP9 uncompressed-header pixel-format detection ────────────────
// Private — called only by detect() above. Parses just enough of the
// VP9 frame header to derive chroma subsampling + bit depth.

fn detect_vp9(sample: &[u8]) -> Option<PixelFormat> {
    if sample.len() < 2 {
        return None;
    }
    let mut br = bitreader::BitReader::new(sample);
    let frame_marker = br.read_bits(2)?;
    if frame_marker != 2 {
        return None;
    }
    let profile_low = br.read_bits(1)?;
    let profile_high = br.read_bits(1)?;
    let profile = (profile_high << 1) | profile_low;
    if profile == 3 {
        let _reserved_zero = br.read_bits(1)?;
    }
    let show_existing_frame = br.read_bits(1)?;
    if show_existing_frame == 1 {
        return None;
    }
    let frame_type = br.read_bits(1)?;
    let _show_frame = br.read_bits(1)?;
    let _error_resilient = br.read_bits(1)?;

    // color_config only appears on keyframes.
    if frame_type != 0 {
        return None;
    }

    // Keyframe sync code: 3 bytes {0x49, 0x83, 0x42}. 24 bits.
    let sync = br.read_bits(24)?;
    if sync != 0x498342 {
        return None;
    }

    let bit_depth = if profile >= 2 {
        if br.read_bits(1)? == 0 { 10 } else { 12 }
    } else {
        8
    };
    let _color_space = br.read_bits(3)?;
    // color_range + subsampling — layout depends on color_space
    // For simplicity: for Profile 0/2 the subsampling is 4:2:0. Profile
    // 1/3 read subsampling_x/y fields to distinguish 4:2:2 vs 4:4:4.
    let (sx, sy) = if profile == 1 || profile == 3 {
        let _color_range = br.read_bits(1)?;
        let sx = br.read_bits(1)?;
        let sy = br.read_bits(1)?;
        (sx, sy)
    } else {
        (1, 1) // 4:2:0
    };

    let chroma_idc = match (sx, sy) {
        (1, 1) => 1, // 4:2:0
        (1, 0) => 2, // 4:2:2
        (0, 0) => 3, // 4:4:4
        _ => 1,
    };

    Some(PixelFormat::from_chroma_and_depth(chroma_idc, bit_depth))
}

/// The colour a VP9 keyframe's `color_config()` states (VP9 bitstream
/// specification §6.2.2): its `color_space` (0 `CS_UNKNOWN`, 1 `CS_BT_601`,
/// 2 `CS_BT_709`, 3 `CS_SMPTE_170`, 4 `CS_SMPTE_240`, 5 `CS_BT_2020`,
/// 7 `CS_RGB`) and whether `color_range` is full. `CS_RGB` is full range by
/// definition. `None` for anything but a keyframe's uncompressed header, the
/// only frames that carry `color_config()` in every profile.
pub fn parse_vp9_colour(sample: &[u8]) -> Option<Vp9Colour> {
    let mut br = bitreader::BitReader::new(sample);
    if br.read_bits(2)? != 2 {
        return None;
    }
    let profile_low = br.read_bits(1)?;
    let profile = (br.read_bits(1)? << 1) | profile_low;
    if profile == 3 {
        let _reserved_zero = br.read_bits(1)?;
    }
    if br.read_bits(1)? == 1 {
        return None; // show_existing_frame
    }
    if br.read_bits(1)? != 0 {
        return None; // not a keyframe
    }
    let _show_frame = br.read_bits(1)?;
    let _error_resilient = br.read_bits(1)?;
    if br.read_bits(24)? != 0x498342 {
        return None;
    }
    if profile >= 2 {
        let _ten_or_twelve_bit = br.read_bits(1)?;
    }
    let color_space = br.read_bits(3)? as u8;
    let full_range = if color_space == 7 {
        true
    } else {
        br.read_bits(1)? == 1
    };
    Some(Vp9Colour {
        color_space,
        full_range,
    })
}

/// What [`parse_vp9_colour`] read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Vp9Colour {
    /// VP9 `color_space` (3 bits).
    pub color_space: u8,
    /// `color_range` 1 (or `CS_RGB`).
    pub full_range: bool,
}

impl Vp9Colour {
    /// The H.273 `matrix_coefficients` the colour space names, as libavcodec
    /// maps it (`vp9.c`): `CS_BT_601` → 5, `CS_BT_709` → 1, `CS_SMPTE_170` → 6,
    /// `CS_SMPTE_240` → 7, `CS_BT_2020` → 9, `CS_RGB` → 0; `CS_UNKNOWN` and the
    /// reserved 6 say nothing (2). VP9 states no primaries or transfer.
    pub fn matrix_coefficients(&self) -> u8 {
        match self.color_space {
            1 => 5,
            2 => 1,
            3 => 6,
            4 => 7,
            5 => 9,
            7 => 0,
            _ => 2,
        }
    }
}
