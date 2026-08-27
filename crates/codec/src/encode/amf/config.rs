//! AMF property/config building helpers shared by the AV1 and H.26x paths.
//!
//! Wide-string encoding, per-pixel-format dispatch, H.273 code mapping, the
//! colour-profile mapping, and the `set_*_property` helpers that drive every
//! `SetProperty` call on a component or a surface.

use anyhow::{Result, bail};
use std::ffi::c_void;

use crate::frame::{PixelFormat, TransferFn};

// Items from ffi.rs accessed via the parent (amf) module's private-use
// re-export (`use self::ffi::*;` in mod.rs).
use super::{
    AMF_COLOR_BIT_DEPTH_8, AMF_COLOR_BIT_DEPTH_10, AMF_COLOR_PROFILE_709, AMF_COLOR_PROFILE_2020,
    AMF_COLOR_PROFILE_FULL_709, AMF_COLOR_PROFILE_FULL_2020, AMF_OK, AMF_SURFACE_NV12,
    AMF_SURFACE_P010, AmfObj, AmfVariant, AmfWchar, result_name,
};

// ─── Wide-string helpers ──────────────────────────────────────────

/// Encode a null-terminated `wchar_t` string the way the SDK's property
/// names are declared (`const wchar_t*`): UTF-16 on Windows, UTF-32 on
/// Linux — see [`AmfWchar`]. Every property name in the headers is ASCII,
/// so both encodings are the code points themselves.
pub(super) fn wide(s: &str) -> Vec<AmfWchar> {
    let mut out: Vec<AmfWchar> = s.chars().map(|c| c as u32 as AmfWchar).collect();
    out.push(0);
    out
}

/// Decode a null-terminated `wchar_t` string back to UTF-8 (tests and logs).
#[cfg(test)]
pub(super) unsafe fn from_wide(p: *const AmfWchar) -> String {
    unsafe {
        let mut len = 0usize;
        while *p.add(len) != 0 {
            len += 1;
        }
        std::slice::from_raw_parts(p, len)
            .iter()
            .map(|&c| char::from_u32(c as u32).unwrap_or('\u{fffd}'))
            .collect()
    }
}

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

// ─── Property setters ─────────────────────────────────────────────

/// Set one property on any AMF object (component, surface, buffer, context)
/// through its `AMFPropertyStorage` prefix. Returns the `AMF_RESULT` as a
/// Rust `Result` so the call site can bail cleanly when the driver rejects a
/// knob value.
pub(super) unsafe fn set_property(obj: *mut c_void, name: &str, value: AmfVariant) -> Result<()> {
    unsafe {
        let vt = &*(*(obj as *mut AmfObj)).vtbl;
        let wname = wide(name);
        let rc = (vt.set_property)(obj, wname.as_ptr(), value);
        if rc != AMF_OK {
            bail!(
                "AMF SetProperty({name}) failed: {rc} ({})",
                result_name(rc)
            );
        }
        Ok(())
    }
}

/// `SetProperty(name, amf_int64)`.
pub(super) unsafe fn set_int_property(obj: *mut c_void, name: &str, value: i64) -> Result<()> {
    unsafe {
        set_property(obj, name, AmfVariant::int64(value))
            .map_err(|e| anyhow::anyhow!("{e} (value {value})"))
    }
}

/// `SetProperty(name, amf_bool)`.
pub(super) unsafe fn set_bool_property(obj: *mut c_void, name: &str, value: bool) -> Result<()> {
    unsafe {
        set_property(obj, name, AmfVariant::bool_(value))
            .map_err(|e| anyhow::anyhow!("{e} (value {value})"))
    }
}

/// `SetProperty(name, AMFRate { num, den })`.
pub(super) unsafe fn set_rate_property(
    obj: *mut c_void,
    name: &str,
    num: u32,
    den: u32,
) -> Result<()> {
    unsafe {
        set_property(obj, name, AmfVariant::rate(num, den))
            .map_err(|e| anyhow::anyhow!("{e} (value {num}/{den})"))
    }
}

/// `GetProperty(name)` as an `amf_int64`, or `None` when the object does not
/// carry the property or it is not int-typed.
pub(super) unsafe fn get_int_property(obj: *mut c_void, name: &str) -> Option<i64> {
    unsafe {
        let vt = &*(*(obj as *mut AmfObj)).vtbl;
        let wname = wide(name);
        let mut var = AmfVariant::empty();
        if (vt.get_property)(obj, wname.as_ptr(), &mut var) != AMF_OK {
            return None;
        }
        var.as_int64()
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
