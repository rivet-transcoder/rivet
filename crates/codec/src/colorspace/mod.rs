use anyhow::{Result, bail};
use bytes::BytesMut;

use crate::frame::{ColorMetadata, ColorSpace, PixelFormat, TransferFn, VideoFrame};
use crate::tonemap::tonemap_yuv420p10le_bt2020_to_yuv420p_bt709;

mod bt601_to_709;
mod bt601_to_709_10bit;
mod chroma_convert;
mod downsample_444;
mod scale;

#[cfg(test)]
mod tests;

// ── public re-exports ─────────────────────────────────────────────────────────

pub use bt601_to_709::{bt601_to_bt709_planes, bt601_to_bt709_planes_scalar};
pub use bt601_to_709_10bit::{
    bt601_to_bt709_planes_10bit, bt601_to_bt709_planes_10bit_scalar,
};
pub use downsample_444::{
    downsample_444_to_420_frame, downsample_chroma_444_to_420,
    downsample_chroma_444_to_420_10bit,
};
pub use scale::{
    bilinear_scale_plane, bilinear_scale_plane_scalar, bilinear_scale_plane_u16,
    bilinear_scale_plane_u16_scalar, scale_frame,
};

// =============================================================================
// Shared BT.601 → BT.709 matrix constants
// =============================================================================
//
// Derived from BT.601 M_YUV→RGB composed with BT.709 M_RGB→YUV, both in
// 8-bit studio-range form (Y in [16,235], Cb/Cr in [16,240]).
//
// Result (matrix applied to deltas):
//   ΔY709  = 1.00000·ΔY - 0.11555·ΔCb - 0.20794·ΔCr
//   ΔCb709 = 0·ΔY + 1.01864·ΔCb + 0.11462·ΔCr
//   ΔCr709 = 0·ΔY + 0.07505·ΔCb + 1.02533·ΔCr
//
// Kept here (parent module) so both bt601_to_709 and bt601_to_709_10bit child
// modules can reference them as `super::Q15`, `super::M_Y_CB`, etc. without
// duplication. Child modules may access private parent items.

const Q15: i32 = 15;
const Q15_ROUND: i32 = 1 << (Q15 - 1);

// Row 0 (Y): Y709 = Y601·1.0 + M_Y_CB·ΔCb + M_Y_CR·ΔCr. The 1.0 coefficient
// is applied as a direct copy (no fixed-point multiply).
#[allow(dead_code)] // documented identity; not emitted into the hot path
const M_Y_Y: i32 = 32768;
const M_Y_CB: i32 = (-0.11554975_f64 * 32768.0) as i32; // -3786
const M_Y_CR: i32 = (-0.20793764_f64 * 32768.0) as i32; // -6814
// Row 1 (Cb): no luma coupling
const M_CB_CB: i32 = (1.01863972_f64 * 32768.0).round() as i32; // 33379
const M_CB_CR: i32 = (0.11461795_f64 * 32768.0).round() as i32; //  3756
// Row 2 (Cr): no luma coupling
const M_CR_CB: i32 = (0.07504945_f64 * 32768.0).round() as i32; //  2459
const M_CR_CR: i32 = (1.02532707_f64 * 32768.0).round() as i32; // 33598

// =============================================================================
// Shared byte-order helpers
// =============================================================================
//
// Used by scale, downsample_444, and chroma_convert child modules.
// Private visibility is sufficient — child modules can always access a private
// parent item as `super::read_u16le`.

fn read_u16le(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

fn write_u16le(out: &mut BytesMut, samples: &[u16]) {
    for s in samples {
        out.extend_from_slice(&s.to_le_bytes());
    }
}

// =============================================================================
// Top-level dispatch entry points
// =============================================================================

/// Normalize a decoder frame for the encoder.
///
/// **8-bit path** (target `Yuv420p` / `Bt709`): every supported 8-bit
/// pixel format is converted to packed 4:2:0 BT.709 limited-range. The
/// dispatcher does (a) chroma layout normalisation — NV12/NV21
/// deinterleave, 4:2:2 vertical-2:1 average, 4:4:4 box average — then
/// (b) RGB → YUV matrix when the source is RGB, then (c) BT.601 → BT.709
/// matrix correction for tagged-BT.601 YUV sources.
///
/// **10-bit path** (target `Yuv420p10le`, HDR-aware): 10-bit and
/// alpha-bearing 10-bit formats are downsampled to `Yuv420p10le` and
/// returned as-is on the matrix axis. The pipeline preserves the source's
/// `ColorMetadata` (primaries / transfer / matrix) so the muxer's
/// `colr nclx` box and the AV1 sequence header carry the HDR / wide-gamut
/// signaling unchanged. Squad-19, roadmap #5.
///
/// **Format coverage** (input → output):
/// - `Yuv420p` (BT.709) → passthrough
/// - `Yuv420p` (BT.601 / BT.2020) → matrix correction to BT.709 (BT.2020
///   8-bit is rare; treated as BT.601 for the matrix, downstream `colr
///   nclx` keeps the truth)
/// - `Yuv422p` / `Yuv422p10le` → vertical 2:1 chroma average
/// - `Yuv444p` / `Yuv444p10le` / `Yuva444p10le` → 2×2 box average
///   (alpha dropped for `Yuva444p10le`)
/// - `Nv12` → UV deinterleave
/// - `Nv21` → VU deinterleave (same as NV12 with planes swapped)
/// - `Rgb24` / `Rgba32` → BT.709 RGB→YUV matrix (alpha discarded for
///   `Rgba32`)
/// - `Yuv420p10le` → passthrough
/// - `Yuv420p12le` → not yet wired; `bail!` (no decoder in tree emits
///   12-bit today)
/// HDR-aware variant. When the source `ColorMetadata` indicates a PQ /
/// HLG transfer function, the 10-bit input is tonemapped to 8-bit BT.709
/// limited via the Hable filmic curve (`crate::tonemap`). For SDR
/// sources (transfer Bt709 / Bt470Bg / Linear / Unspecified), behaviour
/// is identical to `convert_to_yuv420p_bt709` — including the existing
/// 10-bit BT.709 passthrough.
///
/// This is the dispatch the pipeline should call when it has access to
/// the source's `ColorMetadata`. Existing 8-bit-only callers that only
/// have a frame in scope can continue to use `convert_to_yuv420p_bt709`
/// directly; SDR semantics there are unchanged.
pub fn convert_to_sdr_bt709(
    frame: &VideoFrame,
    color_metadata: &ColorMetadata,
) -> Result<VideoFrame> {
    let is_hdr_transfer = matches!(
        color_metadata.transfer,
        TransferFn::St2084 | TransferFn::AribStdB67
    );
    if is_hdr_transfer && matches!(frame.format, PixelFormat::Yuv420p10le) {
        let max_white_nits = color_metadata
            .mastering_display
            .as_ref()
            // mastering_display.max_luminance is in 0.0001 cd/m² ticks
            // per H.265 SEI 137 / ST 2086. Divide to get nits.
            .map(|m| (m.max_luminance as f32) / 10_000.0)
            .filter(|n| *n > 0.0);
        return tonemap_yuv420p10le_bt2020_to_yuv420p_bt709(
            frame,
            color_metadata.transfer,
            max_white_nits,
        );
    }
    // SDR path — also handles Yuv422p10le / Yuv444p10le HDR by first
    // funnelling through the existing 10-bit passthrough chain. Those
    // chroma formats are rarely HDR in practice; if they show up the
    // mux's colr nclx still tags them PQ / HLG and downstream playback
    // honours the transfer. Future work: extend the tonemap to accept
    // those chroma layouts directly.
    convert_to_yuv420p_bt709(frame)
}

pub fn convert_to_yuv420p_bt709(frame: &VideoFrame) -> Result<VideoFrame> {
    use PixelFormat::*;

    // ── 10-bit / wide-gamut path ──────────────────────────────────────
    // HDR / wide-gamut passthrough on the matrix axis. Chroma layout
    // gets normalised to 4:2:0 if needed, but matrix coefficients are
    // preserved on the frame's `color_space` field — the encoder
    // signals it through the AV1 sequence header and the mux writes
    // `colr nclx` so a player/browser can reverse the matrix.
    match frame.format {
        Yuv420p10le => return Ok(frame.clone()),
        Yuv422p10le => return chroma_convert::yuv422p10le_to_yuv420p10le(frame),
        Yuv444p10le | Yuva444p10le => {
            return downsample_444::downsample_444_to_420_frame(frame)
        }
        Yuv420p12le => bail!(
            "Yuv420p12le not yet supported in convert_to_yuv420p_bt709 \
             (no decoder in tree emits 12-bit; add a 12→10-bit dither \
             when a decoder lands that does)"
        ),
        _ => {}
    }

    // ── 8-bit path: RGB sources go straight to Yuv420p/Bt709 ─────────
    match frame.format {
        Rgb24 => return chroma_convert::rgb_to_yuv420p_bt709(frame, /*has_alpha=*/ false),
        Rgba32 => return chroma_convert::rgb_to_yuv420p_bt709(frame, /*has_alpha=*/ true),
        _ => {}
    }

    // ── 8-bit path: YUV chroma-layout normalize → Yuv420p ────────────
    let yuv420p = match frame.format {
        Yuv420p => frame.clone(),
        Nv12 => chroma_convert::nv12_to_yuv420p(frame)?,
        Nv21 => chroma_convert::nv21_to_yuv420p(frame)?,
        Yuv422p => chroma_convert::yuv422p_to_yuv420p(frame)?,
        Yuv444p => downsample_444::downsample_444_to_420_frame(frame)?,
        other => bail!(
            "unsupported conversion: {:?}/{:?} → Yuv420p/Bt709",
            other,
            frame.color_space
        ),
    };

    // ── 8-bit path: matrix correction → Bt709 ────────────────────────
    if yuv420p.color_space == ColorSpace::Bt709 {
        Ok(yuv420p)
    } else {
        // BT.601 and BT.2020 (rare in 8-bit SDR) both route through the
        // BT.601 → BT.709 matrix. BT.2020-via-BT.601 produces a slight
        // hue shift but the alternative — bailing — would block every
        // BT.2020-tagged 8-bit input from transcoding. The mux's
        // `colr nclx` carries the post-conversion BT.709 tag so a
        // downstream player applies the right inverse.
        bt601_to_709::recolor_yuv420p_bt601_to_bt709(&yuv420p)
    }
}
