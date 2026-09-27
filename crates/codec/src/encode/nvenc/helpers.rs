//! Small pixel-format and frame-rate mapping helpers.

use anyhow::{bail, Result};
use std::os::raw::c_uint;

use crate::frame::{PixelFormat, TransferFn};

use super::constants::{NV_ENC_BUFFER_FORMAT_IYUV, NV_ENC_BUFFER_FORMAT_YUV420_10BIT};

// ─── Pixel-format dispatch helpers ────────────────────────────────
//
// Mirrors `crates/codec/src/encode/rav1e_enc.rs`'s pixel-format dispatch.
// Centralises (a) the input pixel format → NVENC buffer format mapping,
// (b) the per-format bytes/sample, and (c) the AV1 OBU `BitDepth` value.
// Keeping these in three small functions side-by-side makes the
// 8-bit / 10-bit branches obvious at the call site without scattering
// `match frame.format { … }` blocks throughout `upload_frame` and
// `encode_pending`.

/// Map a `PixelFormat` to its NVENC `NV_ENC_BUFFER_FORMAT` constant.
/// Returns the format only; the per-pixel bit depth lives in
/// `pixel_bit_depth_for_format`. Bails on unsupported chroma
/// (NVENC AV1 in this service is 4:2:0 only — H.264/HEVC have other
/// 4:2:2 / 4:4:4 paths but this encoder is AV1).
pub(super) fn nvenc_buffer_format_for(fmt: PixelFormat) -> Result<c_uint> {
    match fmt {
        PixelFormat::Yuv420p => Ok(NV_ENC_BUFFER_FORMAT_IYUV),
        PixelFormat::Yuv420p10le => Ok(NV_ENC_BUFFER_FORMAT_YUV420_10BIT),
        other => bail!(
            "NVENC AV1 expects Yuv420p or Yuv420p10le, got {other:?} \
             (4:2:2 / 4:4:4 / RGB / alpha not supported on this backend)"
        ),
    }
}

/// Returns the `pixel_bit_depth_minus_8` value for the AV1 codec config.
/// 0 = 8-bit, 2 = 10-bit. Drives the AV1 sequence header `BitDepth`
/// signalling so a decoder knows the sample width up front.
pub(super) const fn pixel_bit_depth_minus8_for(fmt: PixelFormat) -> u32 {
    match fmt {
        PixelFormat::Yuv420p10le => 2,
        _ => 0,
    }
}

/// Translate `TransferFn` → ITU-T H.273 numeric code for the AV1 OBU
/// sequence header `transfer_characteristics` field.
///
/// The mux side (`crates/container/src/mux.rs::transfer_to_h273`)
/// uses the same mapping — keeping a sibling helper here lets the
/// in-bitstream code match what gets written into `colr nclx` so a
/// downstream player sees consistent metadata between container and
/// elementary stream. Unspecified collapses to canonical Bt709 (1)
/// because the AV1 spec has no "unspecified" sentinel for transfer.
pub(super) fn transfer_to_h273(tf: TransferFn) -> u32 {
    match tf {
        TransferFn::Bt709 => 1,
        TransferFn::Bt470Bg => 4,
        TransferFn::Linear => 8,
        TransferFn::St2084 => 16,
        TransferFn::AribStdB67 => 18,
        TransferFn::Unspecified => 1,
    }
}

// ─── Frame-rate rational mapping ──────────────────────────────────
//
// NVENC init params carry `frameRateNum` / `frameRateDen` as separate
// u32s. Pass the canonical rational for broadcast rates so 1001-family
// NTSC rates (23.976, 29.97, 59.94) encode with exact sync instead of
// the lossy `(fps*1000)/1000` shortcut (review task #3 MEDIUM-4).

/// Map a float fps to its canonical (num, den) pair. Common broadcast
/// rates are returned exactly; any other value falls back to
/// `(round(fps*1000), 1000)`.
///
/// 1001-family detector: if `fps ≈ k/1001` for integer `k`, treat as
/// `(k, 1001)` — keeps exact sync for precise 1001-family inputs.
pub(super) fn fps_to_rational(fps: f64) -> (u32, u32) {
    // Exact hits first — avoids float rounding games for values that
    // can be represented cleanly. Tolerance ≤ 1e-3 covers both
    // 29.97 → 30000/1001 and 23.976 → 24000/1001 inputs from
    // user-facing config files.
    const EXACT: &[(f64, u32, u32)] = &[
        (23.976, 24_000, 1001),
        (24.0, 24, 1),
        (25.0, 25, 1),
        (29.97, 30_000, 1001),
        (30.0, 30, 1),
        (48.0, 48, 1),
        (50.0, 50, 1),
        (59.94, 60_000, 1001),
        (60.0, 60, 1),
    ];
    for &(f, n, d) in EXACT {
        if (fps - f).abs() < 1e-3 {
            return (n, d);
        }
    }

    // Integer fps shortcut — if `fps` rounds to itself (whole number),
    // prefer `(n, 1)` over any `k/1001` representation. Otherwise
    // 100.0 would hit the 1001-family detector below as (100100, 1001)
    // since 100*1001/1001 = 100 exactly.
    if (fps - fps.round()).abs() < 1e-6 && fps > 0.0 {
        return (fps.round() as u32, 1);
    }

    // 1001-family detector for more precise inputs like 23.9760239760…
    // If rounding `fps*1001` to an integer and dividing back lands
    // within 1e-4 of the original fps, treat it as a k/1001 rate.
    let k = (fps * 1001.0).round();
    if (k / 1001.0 - fps).abs() < 1e-4 && k > 0.0 {
        let k_u = k as u32;
        return (k_u, 1001);
    }

    // Generic fallback — (round(fps*1000), 1000). Round instead of
    // truncation so odd rates like 47.97 land at 47970/1000 exactly.
    let num = (fps * 1000.0).round().max(1.0) as u32;
    (num, 1000)
}

// Compile-time pin: bit-depth dispatch must stay in sync with the
// NVENC buffer format and AV1 codec-config writers.
const _: () = assert!(pixel_bit_depth_minus8_for(PixelFormat::Yuv420p10le) == 2);
const _: () = assert!(pixel_bit_depth_minus8_for(PixelFormat::Yuv420p) == 0);

// ─── Constant bitrate (CBR) ───────────────────────────────────────

/// Lay a constant-rate rung (`rate=cbr`) into `NV_ENC_RC_PARAMS`, replacing
/// whatever the quality path chose: `NV_ENC_PARAMS_RC_CBR`, with
/// `averageBitRate` = `maxBitRate` = the rate, `vbvBufferSize` the rung's
/// HRD buffer and `vbvInitialDelay` its initial fullness, both in bits.
/// `targetQuality` is cleared (it is a VBR knob; the quality target is not
/// consulted at a constant rate). The same for every codec this backend
/// encodes — these are rate-control fields, not codec-union ones.
pub(super) fn apply_constant_rate(rc: &mut super::ffi::NvEncRcParams, rate: crate::encode::tuning::ConstantRate) {
    let bits = |v: u64| u32::try_from(v).unwrap_or(u32::MAX);
    rc.rate_control_mode = super::constants::NV_ENC_PARAMS_RC_CBR;
    rc.average_bitrate = rate.bps;
    rc.max_bitrate = rate.bps;
    rc.vbv_buffer_size = bits(rate.buffer_bits());
    rc.vbv_initial_delay = bits(rate.initial_fullness_bits());
    rc.target_quality = 0;
    rc.target_quality_lsb = 0;
}
