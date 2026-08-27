//! AMF config building helpers shared by the AV1 and H.26x paths:
//! per-pixel-format dispatch, H.273 code mapping, the colour-profile mapping
//! and the frame-rate rational. The wide-string and `SetProperty` helpers
//! live in `crate::amf_runtime`.

use anyhow::{Result, bail};

use crate::frame::{PixelFormat, TransferFn};

use crate::amf_ffi::{
    AMF_COLOR_BIT_DEPTH_8, AMF_COLOR_BIT_DEPTH_10, AMF_COLOR_PROFILE_709, AMF_COLOR_PROFILE_2020,
    AMF_COLOR_PROFILE_FULL_709, AMF_COLOR_PROFILE_FULL_2020, AMF_SURFACE_NV12, AMF_SURFACE_P010,
};

// ─── Per-pixel-format dispatch ────────────────────────────────────
//
// Every AMF encoder here takes NV12 (8-bit) or P010 (10-bit) host-memory
// surfaces; both are interleaved-chroma YUV 4:2:0. Selecting the wrong
// surface format for the input depth produces silent garbage (the 8-bit path
// on a wide-word surface reads two adjacent samples per byte → noise + halved
// width), so the dispatch is one function, tested, and the session captures
// its answer once.

pub(super) fn amf_surface_format_for(fmt: PixelFormat) -> Result<i32> {
    match fmt {
        PixelFormat::Yuv420p => Ok(AMF_SURFACE_NV12),
        PixelFormat::Yuv420p10le => Ok(AMF_SURFACE_P010),
        other => bail!("AMF expects Yuv420p or Yuv420p10le, got {other:?}"),
    }
}

/// `AMF_COLOR_BIT_DEPTH_ENUM` value for a pixel format
/// (`components/ColorSpace.h:106-107`: the enum values are the literal
/// depths, 8 and 10).
pub(super) const fn amf_color_bit_depth_for(fmt: PixelFormat) -> i64 {
    match fmt {
        PixelFormat::Yuv420p10le => AMF_COLOR_BIT_DEPTH_10,
        _ => AMF_COLOR_BIT_DEPTH_8,
    }
}

const _: () = assert!(amf_color_bit_depth_for(PixelFormat::Yuv420p10le) == 10);
const _: () = assert!(amf_color_bit_depth_for(PixelFormat::Yuv420p) == 8);

/// Translate `TransferFn` → ITU-T H.273 numeric code, which is also what
/// `AMF_COLOR_TRANSFER_CHARACTERISTIC_ENUM` uses ("as in VUI
/// transfer_characteristic AVC and HEVC", `components/ColorSpace.h:80`). Same
/// table as `nvenc.rs::transfer_to_h273`, `qsv/config.rs::transfer_to_h273`
/// and the mux's — keeping them in lockstep means HDR signalling matches
/// across the container `colr nclx` and every encoder's bitstream.
pub(super) fn transfer_to_h273(tf: TransferFn) -> i64 {
    match tf {
        TransferFn::Bt709 => 1,
        TransferFn::Bt470Bg => 4,
        TransferFn::Linear => 8,
        TransferFn::St2084 => 16,
        TransferFn::AribStdB67 => 18,
        TransferFn::Unspecified => 1,
    }
}

/// `AMF_VIDEO_CONVERTER_COLOR_PROFILE_ENUM` for a matrix + range
/// (`components/ColorSpace.h:46-57`). AMF has no direct
/// `matrix_coefficients` knob on its encoders; the colour profile is how the
/// matrix (and, for AVC, the range) reaches the VUI. H.273 matrix 9 / 10 is
/// BT.2020 (NCL / CL); everything else this pipeline produces is BT.709.
pub(super) fn amf_color_profile_for(matrix_coefficients: u8, full_range: bool) -> i64 {
    let bt2020 = matches!(matrix_coefficients, 9 | 10);
    match (bt2020, full_range) {
        (true, true) => AMF_COLOR_PROFILE_FULL_2020,
        (true, false) => AMF_COLOR_PROFILE_2020,
        (false, true) => AMF_COLOR_PROFILE_FULL_709,
        (false, false) => AMF_COLOR_PROFILE_709,
    }
}

/// The frame rate as the `AMFRate` the `…FrameRate` properties take. Integer
/// rates pass through; a fractional rate becomes a `/1000` rational
/// (29.97 → 29970/1000), which is what the level tables and the rate
/// controller need — an exact 30000/1001 is not distinguishable at the
/// precision the config carries.
pub(super) fn frame_rate_rational(fps: f64) -> (u32, u32) {
    let fps = if fps.is_finite() && fps > 0.0 { fps } else { 30.0 };
    if (fps - fps.round()).abs() < 1e-6 {
        (fps.round() as u32, 1)
    } else {
        ((fps * 1000.0).round() as u32, 1000)
    }
}
