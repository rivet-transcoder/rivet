use anyhow::{Result, bail};
use bytes::{Bytes, BytesMut};

use crate::frame::{ColorMetadata, ColorSpace, PixelFormat, TransferFn, VideoFrame};
use crate::tonemap::tonemap_yuv420p10le_bt2020_to_yuv420p_bt709;

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
        Yuv422p10le => return yuv422p10le_to_yuv420p10le(frame),
        Yuv444p10le | Yuva444p10le => return downsample_444_to_420_frame(frame),
        Yuv420p12le => bail!(
            "Yuv420p12le not yet supported in convert_to_yuv420p_bt709 \
             (no decoder in tree emits 12-bit; add a 12→10-bit dither \
             when a decoder lands that does)"
        ),
        _ => {}
    }

    // ── 8-bit path: RGB sources go straight to Yuv420p/Bt709 ─────────
    match frame.format {
        Rgb24 => return rgb_to_yuv420p_bt709(frame, /*has_alpha=*/ false),
        Rgba32 => return rgb_to_yuv420p_bt709(frame, /*has_alpha=*/ true),
        _ => {}
    }

    // ── 8-bit path: YUV chroma-layout normalize → Yuv420p ────────────
    let yuv420p = match frame.format {
        Yuv420p => frame.clone(),
        Nv12 => nv12_to_yuv420p(frame)?,
        Nv21 => nv21_to_yuv420p(frame)?,
        Yuv422p => yuv422p_to_yuv420p(frame)?,
        Yuv444p => downsample_444_to_420_frame(frame)?,
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
        recolor_yuv420p_bt601_to_bt709(&yuv420p)
    }
}

fn nv12_to_yuv420p(frame: &VideoFrame) -> Result<VideoFrame> {
    deinterleave_semiplanar_to_yuv420p(frame, /*v_first=*/ false)
}

/// NV21 has the same packed layout as NV12 but the chroma plane carries
/// `VU` interleaved instead of `UV`. Sharing the implementation reduces
/// the chance of one path drifting from the other on bug fixes.
fn nv21_to_yuv420p(frame: &VideoFrame) -> Result<VideoFrame> {
    deinterleave_semiplanar_to_yuv420p(frame, /*v_first=*/ true)
}

fn deinterleave_semiplanar_to_yuv420p(frame: &VideoFrame, v_first: bool) -> Result<VideoFrame> {
    let w = frame.width as usize;
    let h = frame.height as usize;
    let y_size = w * h;
    let uv_size = y_size / 4;
    if frame.data.len() < y_size + 2 * uv_size {
        bail!(
            "{} frame too small for {}x{}: need {} bytes got {}",
            if v_first { "NV21" } else { "NV12" },
            w,
            h,
            y_size + 2 * uv_size,
            frame.data.len()
        );
    }
    let mut out = BytesMut::with_capacity(y_size + uv_size * 2);

    // Y plane — straight copy.
    out.extend_from_slice(&frame.data[..y_size]);

    // Deinterleave the packed chroma plane.
    let uv = &frame.data[y_size..];
    let mut u_plane = Vec::with_capacity(uv_size);
    let mut v_plane = Vec::with_capacity(uv_size);
    for i in 0..uv_size {
        let (a, b) = (uv[i * 2], uv[i * 2 + 1]);
        if v_first {
            v_plane.push(a);
            u_plane.push(b);
        } else {
            u_plane.push(a);
            v_plane.push(b);
        }
    }
    out.extend_from_slice(&u_plane);
    out.extend_from_slice(&v_plane);

    Ok(VideoFrame::new(
        out.freeze(),
        frame.width,
        frame.height,
        PixelFormat::Yuv420p,
        frame.color_space,
        frame.pts,
    ))
}

/// `Yuv422p` has full-width chroma rows but vertically subsampled to
/// the SAME row count as luma is HALVED to land 4:2:0. Average two
/// adjacent vertical chroma rows per output row.
fn yuv422p_to_yuv420p(frame: &VideoFrame) -> Result<VideoFrame> {
    let w = frame.width as usize;
    let h = frame.height as usize;
    let cw = w.div_ceil(2);
    // 4:2:2 has chroma rows == luma rows, with chroma cols halved.
    let ch_in = h;
    let ch_out = h.div_ceil(2);
    let y_size = w * h;
    let chroma_in_size = cw * ch_in;
    let chroma_out_size = cw * ch_out;
    if frame.data.len() < y_size + 2 * chroma_in_size {
        bail!(
            "Yuv422p frame too small for {}x{}: need {} bytes got {}",
            w,
            h,
            y_size + 2 * chroma_in_size,
            frame.data.len()
        );
    }

    let (y_in, rest) = frame.data.split_at(y_size);
    let (cb_in, cr_in) = rest.split_at(chroma_in_size);

    let mut out = BytesMut::with_capacity(y_size + 2 * chroma_out_size);
    out.extend_from_slice(y_in);

    for plane in [cb_in, cr_in] {
        for cy in 0..ch_out {
            let y0 = 2 * cy;
            let y1 = (y0 + 1).min(ch_in - 1);
            for cx in 0..cw {
                let s0 = plane[y0 * cw + cx] as u16;
                let s1 = plane[y1 * cw + cx] as u16;
                out.extend_from_slice(&[((s0 + s1 + 1) >> 1) as u8]);
            }
        }
    }

    Ok(VideoFrame::new(
        out.freeze(),
        frame.width,
        frame.height,
        PixelFormat::Yuv420p,
        frame.color_space,
        frame.pts,
    ))
}

/// 10-bit equivalent of `yuv422p_to_yuv420p`. Samples stored u16 LE.
fn yuv422p10le_to_yuv420p10le(frame: &VideoFrame) -> Result<VideoFrame> {
    let w = frame.width as usize;
    let h = frame.height as usize;
    let cw = w.div_ceil(2);
    let ch_in = h;
    let ch_out = h.div_ceil(2);
    let y_samples = w * h;
    let chroma_in_samples = cw * ch_in;
    let chroma_out_samples = cw * ch_out;
    let need_bytes = (y_samples + 2 * chroma_in_samples) * 2;
    if frame.data.len() < need_bytes {
        bail!(
            "Yuv422p10le frame too small for {}x{}: need {} bytes got {}",
            w,
            h,
            need_bytes,
            frame.data.len()
        );
    }
    let words = read_u16le(&frame.data[..need_bytes]);
    let (y_in, rest) = words.split_at(y_samples);
    let (cb_in, cr_in) = rest.split_at(chroma_in_samples);

    let mut out = BytesMut::with_capacity((y_samples + 2 * chroma_out_samples) * 2);
    write_u16le(&mut out, y_in);

    for plane in [cb_in, cr_in] {
        for cy in 0..ch_out {
            let y0 = 2 * cy;
            let y1 = (y0 + 1).min(ch_in - 1);
            for cx in 0..cw {
                let s0 = plane[y0 * cw + cx] as u32;
                let s1 = plane[y1 * cw + cx] as u32;
                let avg = ((s0 + s1 + 1) >> 1) as u16;
                out.extend_from_slice(&avg.to_le_bytes());
            }
        }
    }

    Ok(VideoFrame::new(
        out.freeze(),
        frame.width,
        frame.height,
        PixelFormat::Yuv420p10le,
        frame.color_space,
        frame.pts,
    ))
}

/// RGB (or RGBA, alpha discarded) → BT.709 YCbCr limited-range Yuv420p.
///
/// Per ITU-R BT.709 / H.273 matrix coefficient = 1, with the standard
/// 8-bit studio-range scaling (Y in [16,235], Cb/Cr in [16,240]):
///
/// ```text
/// Y  =  16 + 0.2126·R + 0.7152·G + 0.0722·B  (scaled to 219 swing)
/// Cb = 128 + (B - Y) / (2·(1 - 0.0722))      (scaled to 224 swing)
/// Cr = 128 + (R - Y) / (2·(1 - 0.2126))      (scaled to 224 swing)
/// ```
///
/// Implemented as integer fixed-point (Q15) so a per-pixel pass is
/// branch-free and SIMD-friendly. Chroma is then produced by 2×2
/// averaging the four RGB pixels per chroma site (matches the BT.709
/// Annex A box-average prescription used in our 4:4:4 → 4:2:0 path).
fn rgb_to_yuv420p_bt709(frame: &VideoFrame, has_alpha: bool) -> Result<VideoFrame> {
    let w = frame.width as usize;
    let h = frame.height as usize;
    let stride = if has_alpha { 4 } else { 3 };
    let need = w * h * stride;
    if frame.data.len() < need {
        bail!(
            "{} frame too small for {}x{}: need {} bytes got {}",
            if has_alpha { "Rgba32" } else { "Rgb24" },
            w,
            h,
            need,
            frame.data.len()
        );
    }
    let cw = w.div_ceil(2);
    let ch = h.div_ceil(2);
    let y_size = w * h;
    let chroma_size = cw * ch;
    let mut out = BytesMut::with_capacity(y_size + 2 * chroma_size);
    out.resize(y_size + 2 * chroma_size, 0);

    // BT.709 limited-range Q15 fixed-point coefficients.
    // Y  =  ((kr·R + kg·G + kb·B) · 219 / 255) + 16
    // We pre-scale into Q15 so integer math gives ≈Y_studio:
    //   YR = round(0.2126 · 219/255 · 32768) = 5982
    //   YG = round(0.7152 · 219/255 · 32768) = 20128
    //   YB = round(0.0722 · 219/255 · 32768) = 2032
    //   Y  = ((R·YR + G·YG + B·YB + 16384) >> 15) + 16
    const Y_R: i32 = 5982;
    const Y_G: i32 = 20128;
    const Y_B: i32 = 2032;
    // Cb = ((B - Y_full) / (2·(1-Kb))) · 224/255 + 128
    // Decompose into per-channel Q15 against R,G,B (acts on full-range
    // intermediate before the 224 swing, then re-scaled).
    //   CbR = round(-0.1146 · 224/255 · 32768) = -3299
    //   CbG = round(-0.3854 · 224/255 · 32768) = -11086
    //   CbB = round( 0.5000 · 224/255 · 32768) = 14385
    const CB_R: i32 = -3299;
    const CB_G: i32 = -11086;
    const CB_B: i32 = 14385;
    //   CrR = round( 0.5000 · 224/255 · 32768) = 14385
    //   CrG = round(-0.4542 · 224/255 · 32768) = -13066
    //   CrB = round(-0.0458 · 224/255 · 32768) = -1319
    const CR_R: i32 = 14385;
    const CR_G: i32 = -13066;
    const CR_B: i32 = -1319;

    // Y plane: per-pixel scalar pass.
    for y in 0..h {
        for x in 0..w {
            let off = (y * w + x) * stride;
            let r = frame.data[off] as i32;
            let g = frame.data[off + 1] as i32;
            let b = frame.data[off + 2] as i32;
            let y_val = ((r * Y_R + g * Y_G + b * Y_B + (1 << 14)) >> 15) + 16;
            out[y * w + x] = y_val.clamp(16, 235) as u8;
        }
    }

    // Chroma planes: 2×2 average of the source RGB pixels per chroma
    // site, then matrix to Cb/Cr.
    let cb_off = y_size;
    let cr_off = y_size + chroma_size;
    for cy in 0..ch {
        let y0 = 2 * cy;
        let y1 = (y0 + 1).min(h - 1);
        for cx in 0..cw {
            let x0 = 2 * cx;
            let x1 = (x0 + 1).min(w - 1);
            // Average the four source RGB pixels.
            let mut r_sum = 0i32;
            let mut g_sum = 0i32;
            let mut b_sum = 0i32;
            for &(py, px) in &[(y0, x0), (y0, x1), (y1, x0), (y1, x1)] {
                let off = (py * w + px) * stride;
                r_sum += frame.data[off] as i32;
                g_sum += frame.data[off + 1] as i32;
                b_sum += frame.data[off + 2] as i32;
            }
            let r = (r_sum + 2) >> 2;
            let g = (g_sum + 2) >> 2;
            let b = (b_sum + 2) >> 2;
            let cb = ((r * CB_R + g * CB_G + b * CB_B + (1 << 14)) >> 15) + 128;
            let cr = ((r * CR_R + g * CR_G + b * CR_B + (1 << 14)) >> 15) + 128;
            out[cb_off + cy * cw + cx] = cb.clamp(16, 240) as u8;
            out[cr_off + cy * cw + cx] = cr.clamp(16, 240) as u8;
        }
    }

    Ok(VideoFrame::new(
        out.freeze(),
        frame.width,
        frame.height,
        PixelFormat::Yuv420p,
        ColorSpace::Bt709,
        frame.pts,
    ))
}

// =============================================================================
// BT.601 → BT.709 YCbCr matrix conversion (limited-range 8-bit).
// =============================================================================
//
// Derived from BT.601 M_YUV→RGB composed with BT.709 M_RGB→YUV, both
// in 8-bit studio-range form (Y in [16,235], Cb/Cr in [16,240]).
//
// Derivation path:
//   1. BT.601 YCbCr (limited) → R'G'B' in [0,1], using Kr=0.299,
//      Kg=0.587, Kb=0.114 and the standard limited-range scaling
//      (Y'=(Y-16)/219, Pb=(Cb-128)/224, Pr=(Cr-128)/224).
//   2. BT.709 R'G'B' → YCbCr (limited), using Kr=0.2126, Kg=0.7152,
//      Kb=0.0722 and the inverse scaling (Y_out = 219·Y' + 16,
//      Cb_out = 224·Pb + 128, Cr_out = 224·Pr + 128).
//   3. Multiply M_709 · M_601^-1 on the delta vector
//      (Y-16, Cb-128, Cr-128) to get a single 3×3 with zero offsets
//      in delta space.
//
// Result (matrix applied to deltas):
//   ΔY709  = 1.00000·ΔY - 0.11555·ΔCb - 0.20794·ΔCr
//   ΔCb709 = 0·ΔY + 1.01864·ΔCb + 0.11462·ΔCr
//   ΔCr709 = 0·ΔY + 0.07505·ΔCb + 1.02533·ΔCr
//
// where ΔY = Y-16, ΔCb = Cb-128, ΔCr = Cr-128. Chroma rows have NO
// luma coupling — because BT.601 and BT.709 share the same limited-
// range chroma basis (Pb,Pr scaled by 224 in both), and the chroma
// basis vectors change only with Kr/Kb, not with luma.
//
// Sanity check under this matrix:
//   (16, 128, 128) → (16, 128, 128)   [black round-trips]
//   (235, 128, 128) → (235, 128, 128) [white round-trips]
//   (128, 128, 128) → (128, 128, 128) [any gray round-trips]
// because all three inputs have ΔCb=ΔCr=0, so ΔY709 = ΔY → Y
// unchanged, and ΔCb709 = ΔCr709 = 0 → chroma unchanged.

/// Q15 fixed-point coefficients for the 3×3 BT.601→BT.709 matrix.
/// Multiplying an i16 delta by these and shifting right 15 yields the
/// 709-domain delta. (Coefficients ≥1 round to 32768+, which fits in
/// i32 but not i16; the AVX2 path splits those out and adds back the
/// identity contribution to stay in i16 range for `mulhrs`.)
const Q15: i32 = 15;
const Q15_ROUND: i32 = 1 << (Q15 - 1);

// Row 0 (Y): Y709 = Y601·1.0 + M_Y_CB·ΔCb + M_Y_CR·ΔCr. The 1.0
// coefficient is applied as a direct copy (no fixed-point multiply).
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

#[inline(always)]
fn clamp_y(v: i32) -> u8 {
    v.clamp(16, 235) as u8
}

#[inline(always)]
fn clamp_c(v: i32) -> u8 {
    v.clamp(16, 240) as u8
}

/// Scalar reference implementation — correctness baseline.
///
/// Operates in-place on three planes (Y, Cb, Cr). Chroma planes are
/// half-width / half-height of luma (4:2:0 subsampling). The matrix
/// couples chroma into luma (Y709 depends on ΔCb, ΔCr), so for each
/// luma sample we read the Cb/Cr sample covering it via the 2:1
/// subsampling grid (chroma-centered between luma rows/cols, but for
/// this pipeline we use the standard "shared per 2×2 block" mapping
/// — both decoders in-tree produce this layout).
///
/// Order matters: we must read the *original* Cb/Cr values before
/// overwriting them with Cb709/Cr709. We therefore update luma first
/// (consuming original chroma deltas) and update chroma last.
fn bt601_to_bt709_scalar(y: &mut [u8], cb: &mut [u8], cr: &mut [u8], width: usize, height: usize) {
    debug_assert_eq!(y.len(), width * height);
    debug_assert_eq!(cb.len(), (width / 2) * (height / 2));
    debug_assert_eq!(cr.len(), (width / 2) * (height / 2));

    let cw = width / 2;

    // Luma: Y709 = Y601 + M_Y_CB * ΔCb + M_Y_CR * ΔCr  (per-sample).
    // Each chroma sample covers a 2×2 luma block.
    for yi in 0..height {
        let cy = yi >> 1;
        for xi in 0..width {
            let cx = xi >> 1;
            let cbl = cb[cy * cw + cx] as i32 - 128;
            let crl = cr[cy * cw + cx] as i32 - 128;
            let y_orig = y[yi * width + xi] as i32;
            let delta = (M_Y_CB * cbl + M_Y_CR * crl + Q15_ROUND) >> Q15;
            y[yi * width + xi] = clamp_y(y_orig + delta);
        }
    }

    // Chroma: no luma coupling. Pure 2×2 chroma → chroma transform.
    for v in cb.iter_mut().zip(cr.iter_mut()) {
        let (cbp, crp) = v;
        let cbl = *cbp as i32 - 128;
        let crl = *crp as i32 - 128;
        let new_cb = (M_CB_CB * cbl + M_CB_CR * crl + Q15_ROUND) >> Q15;
        let new_cr = (M_CR_CB * cbl + M_CR_CR * crl + Q15_ROUND) >> Q15;
        *cbp = clamp_c(new_cb + 128);
        *crp = clamp_c(new_cr + 128);
    }
}

/// Public scalar entry point — for bench / tests.
pub fn bt601_to_bt709_planes_scalar(
    y: &mut [u8],
    cb: &mut [u8],
    cr: &mut [u8],
    width: usize,
    height: usize,
) {
    bt601_to_bt709_scalar(y, cb, cr, width, height);
}

/// Runtime-dispatched entry point. Uses AVX2 if the CPU advertises
/// it, scalar fallback otherwise. Safe wrapper around the unsafe
/// target-feature specialization.
pub fn bt601_to_bt709_planes(
    y: &mut [u8],
    cb: &mut [u8],
    cr: &mut [u8],
    width: usize,
    height: usize,
) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: avx2 feature was runtime-detected above.
            unsafe {
                bt601_to_bt709_avx2(y, cb, cr, width, height);
            }
            return;
        }
    }
    bt601_to_bt709_scalar(y, cb, cr, width, height);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn bt601_to_bt709_avx2(
    y: &mut [u8],
    cb: &mut [u8],
    cr: &mut [u8],
    width: usize,
    height: usize,
) {
    unsafe {
        #[cfg(target_arch = "x86")]
        use std::arch::x86::*;
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::*;

        let cw = width / 2;
        let ch = height / 2;

        // Q15 coefficients as packed 16-bit lanes. `_mm256_mulhrs_epi16`
        // computes (a*b + 0x4000) >> 15 on i16 lanes — the fixed-point
        // multiply we want. Inputs must be pre-subtracted to deltas
        // (ΔCb = Cb-128, ΔCr = Cr-128, in [-128, 127]). All six chroma
        // coefficients fit in i16 (range ~[-6814, 33598]): the 1.02533
        // coefficient (33598) exceeds i16_max (32767), so we handle it
        // specially by splitting out the identity contribution for that
        // coupling.
        //
        // Trick: write Cb709 = ΔCb + (M_CB_CB - 32768)/32768 · ΔCb + ...
        //        Cr709 = ΔCr + M_CR_CB/32768 · ΔCb + (M_CR_CR - 32768)/32768 · ΔCr
        // so the ~1.0 coefficients are represented as a free identity add
        // + a small correction. All i16-safe.
        let v_m_y_cb = _mm256_set1_epi16(M_Y_CB as i16); // -3786
        let v_m_y_cr = _mm256_set1_epi16(M_Y_CR as i16); // -6814
        let v_m_cb_cb_corr = _mm256_set1_epi16((M_CB_CB - 32768) as i16); // 611
        let v_m_cb_cr = _mm256_set1_epi16(M_CB_CR as i16); // 3756
        let v_m_cr_cb = _mm256_set1_epi16(M_CR_CB as i16); // 2459
        let v_m_cr_cr_corr = _mm256_set1_epi16((M_CR_CR - 32768) as i16); // 830

        let v_128 = _mm256_set1_epi16(128);
        let v_chroma_lo = _mm256_set1_epi16(16);
        let v_chroma_hi = _mm256_set1_epi16(240);
        let v_luma_lo = _mm256_set1_epi16(16);
        let v_luma_hi = _mm256_set1_epi16(235);

        // ---- Luma pass ----
        // For each 2×2 luma block we share one (Cb, Cr) sample. Process
        // 16 chroma samples per iteration → 32 luma cols on two rows (64
        // luma outputs per iter).
        for cy_idx in 0..ch {
            let y_row0 = cy_idx * 2 * width;
            let y_row1 = y_row0 + width;
            let c_row = cy_idx * cw;

            let mut cx = 0usize;
            while cx + 16 <= cw {
                // Load 16 Cb/Cr, compute per-chroma delta.
                let cb_u8 = _mm_loadu_si128(cb.as_ptr().add(c_row + cx) as *const _);
                let cr_u8 = _mm_loadu_si128(cr.as_ptr().add(c_row + cx) as *const _);
                let cb_i16 = _mm256_cvtepu8_epi16(cb_u8);
                let cr_i16 = _mm256_cvtepu8_epi16(cr_u8);
                let cbl = _mm256_sub_epi16(cb_i16, v_128);
                let crl = _mm256_sub_epi16(cr_i16, v_128);

                // Per-chroma Y delta: d = -0.1155·ΔCb - 0.2079·ΔCr in Q15.
                // mulhrs: (a*b + 0x4000) >> 15.
                let dy_cb = _mm256_mulhrs_epi16(cbl, v_m_y_cb);
                let dy_cr = _mm256_mulhrs_epi16(crl, v_m_y_cr);
                let dy_chroma = _mm256_add_epi16(dy_cb, dy_cr); // 16 per-chroma deltas

                // Apply to luma: each chroma sample covers a 2×2 block.
                // Horizontal: duplicate each 16-bit lane into two adjacent
                // 16-bit lanes → 32 deltas aligned with 32 luma cols.
                // `_mm256_unpacklo_epi16` / `unpackhi_epi16` with self
                // interleaves adjacent lanes; but we want dy[0], dy[0],
                // dy[1], dy[1], .... Use two shuffles + permute.
                //
                // Within each 128-bit lane, `unpacklo_epi16(a, a)` yields
                // a0 a0 a1 a1 a2 a2 a3 a3; `unpackhi_epi16(a, a)` yields
                // a4 a4 ... a7 a7. We have 16 lanes → 4 × 128-bit lanes of
                // output, which we rearrange with permute4x64.
                // Trick: treat the 256-bit register as two halves.
                // Expand 16 chroma deltas → 32 per-luma deltas via a small
                // stack-scratch path. An in-register expand would save two
                // 64-byte cache-line round-trips per 16 chroma samples, but
                // on a 1080p frame this is ~68k bytes of scratch total —
                // negligible vs the 2M luma stores. Kept simple for
                // maintainability; revisit if this function ever shows up
                // at the top of a pprof.
                let mut dy_luma = [0i16; 32];
                _mm256_storeu_si256(dy_luma.as_mut_ptr().add(0) as *mut _, dy_chroma);
                // Above stored 16 chroma deltas. Now expand in-register.
                // Actually simpler: use a second aligned buffer with pair
                // duplication done by indexing.
                let mut dy_luma_pair = [0i16; 32];
                for i in 0..16 {
                    dy_luma_pair[i * 2] = dy_luma[i];
                    dy_luma_pair[i * 2 + 1] = dy_luma[i];
                }
                let dy_luma_lo = _mm256_loadu_si256(dy_luma_pair.as_ptr().add(0) as *const _);
                let dy_luma_hi = _mm256_loadu_si256(dy_luma_pair.as_ptr().add(16) as *const _);

                // Process both luma rows for this chroma row. Both share
                // dy_luma_* because chroma is 4:2:0.
                for row_off in [y_row0, y_row1] {
                    // Load 32 luma pixels.
                    let y_u8 = _mm256_loadu_si256(y.as_ptr().add(row_off + cx * 2) as *const _);
                    // Widen low 16 bytes and high 16 bytes to i16.
                    let y_lo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(y_u8));
                    let y_hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(y_u8));

                    let y_lo_out = _mm256_add_epi16(y_lo, dy_luma_lo);
                    let y_hi_out = _mm256_add_epi16(y_hi, dy_luma_hi);

                    // Clamp to limited-range luma [16, 235].
                    let y_lo_out =
                        _mm256_min_epi16(_mm256_max_epi16(y_lo_out, v_luma_lo), v_luma_hi);
                    let y_hi_out =
                        _mm256_min_epi16(_mm256_max_epi16(y_hi_out, v_luma_lo), v_luma_hi);

                    // Pack i16 → u8 with saturation and store 32 bytes.
                    let packed = _mm256_packus_epi16(y_lo_out, y_hi_out);
                    // packus interleaves lanes; permute to
                    // [lo[0..7], hi[0..7], lo[8..15], hi[8..15]] → lane order.
                    let packed = _mm256_permute4x64_epi64::<0b11_01_10_00>(packed);
                    _mm256_storeu_si256(y.as_mut_ptr().add(row_off + cx * 2) as *mut _, packed);
                }

                cx += 16;
            }

            // Scalar tail for luma of this chroma row.
            while cx < cw {
                let cb_idx = c_row + cx;
                let cbl = cb[cb_idx] as i32 - 128;
                let crl = cr[cb_idx] as i32 - 128;
                let delta = (M_Y_CB * cbl + M_Y_CR * crl + Q15_ROUND) >> Q15;
                let xi = cx * 2;
                for row_off in [y_row0, y_row1] {
                    for sub in 0..2 {
                        let idx = row_off + xi + sub;
                        y[idx] = clamp_y(y[idx] as i32 + delta);
                    }
                }
                cx += 1;
            }
        }

        // ---- Chroma pass (no luma coupling) ----
        // 16 samples per iteration.
        let total_c = cb.len();
        let mut i = 0usize;
        while i + 16 <= total_c {
            let cb_u8 = _mm_loadu_si128(cb.as_ptr().add(i) as *const _);
            let cr_u8 = _mm_loadu_si128(cr.as_ptr().add(i) as *const _);
            let cb_i16 = _mm256_cvtepu8_epi16(cb_u8);
            let cr_i16 = _mm256_cvtepu8_epi16(cr_u8);
            let cbl = _mm256_sub_epi16(cb_i16, v_128);
            let crl = _mm256_sub_epi16(cr_i16, v_128);

            // Cb709 = ΔCb + (M_CB_CB-32768)·ΔCb·2^-15 + M_CB_CR·ΔCr·2^-15 + 128
            let cb_corr = _mm256_mulhrs_epi16(cbl, v_m_cb_cb_corr);
            let cb_cross = _mm256_mulhrs_epi16(crl, v_m_cb_cr);
            let new_cb = _mm256_add_epi16(_mm256_add_epi16(cbl, cb_corr), cb_cross);
            let new_cb = _mm256_add_epi16(new_cb, v_128);

            // Cr709 = ΔCr + (M_CR_CR-32768)·ΔCr·2^-15 + M_CR_CB·ΔCb·2^-15 + 128
            let cr_corr = _mm256_mulhrs_epi16(crl, v_m_cr_cr_corr);
            let cr_cross = _mm256_mulhrs_epi16(cbl, v_m_cr_cb);
            let new_cr = _mm256_add_epi16(_mm256_add_epi16(crl, cr_corr), cr_cross);
            let new_cr = _mm256_add_epi16(new_cr, v_128);

            // Clamp [16, 240].
            let new_cb = _mm256_min_epi16(_mm256_max_epi16(new_cb, v_chroma_lo), v_chroma_hi);
            let new_cr = _mm256_min_epi16(_mm256_max_epi16(new_cr, v_chroma_lo), v_chroma_hi);

            // Pack and store.
            let cb_packed = _mm256_packus_epi16(new_cb, new_cb);
            let cr_packed = _mm256_packus_epi16(new_cr, new_cr);
            let cb_packed = _mm256_permute4x64_epi64::<0b00_00_10_00>(cb_packed);
            let cr_packed = _mm256_permute4x64_epi64::<0b00_00_10_00>(cr_packed);
            _mm_storeu_si128(
                cb.as_mut_ptr().add(i) as *mut _,
                _mm256_castsi256_si128(cb_packed),
            );
            _mm_storeu_si128(
                cr.as_mut_ptr().add(i) as *mut _,
                _mm256_castsi256_si128(cr_packed),
            );

            i += 16;
        }

        // Scalar tail for chroma.
        while i < total_c {
            let cbl = cb[i] as i32 - 128;
            let crl = cr[i] as i32 - 128;
            let new_cb = (M_CB_CB * cbl + M_CB_CR * crl + Q15_ROUND) >> Q15;
            let new_cr = (M_CR_CB * cbl + M_CR_CR * crl + Q15_ROUND) >> Q15;
            cb[i] = clamp_c(new_cb + 128);
            cr[i] = clamp_c(new_cr + 128);
            i += 1;
        }
    }
}

// =============================================================================
// 10-bit BT.601 → BT.709 (Squad-29, follow-up to Squad-19's 10-bit pipeline).
// =============================================================================
//
// 10-bit limited-range constants (Rec. ITU-R BT.2100-2 Table 9 /
// the standard "limited-range 10-bit" of BT.709/BT.2020):
//   luma center  = 64   (16 << 2)
//   chroma center = 512 (128 << 2)
//   luma clamp   = [64, 940]   (16<<2 .. 235<<2)
//   chroma clamp = [64, 960]   (16<<2 .. 240<<2)
//
// The matrix coefficients are identical to the 8-bit case — they're
// derived from the BT.601 / BT.709 spec ratios (Kr / Kg / Kb) and
// don't depend on bit depth. Only the offsets and clamp range change.
//
// Use case: rare. The 10-bit pipeline is HDR-passthrough by default
// (Squad-19) — for HDR sources (BT.2020 + PQ/HLG) we never convert
// because the matrix shift would corrupt the wide gamut. This 10-bit
// BT.601→BT.709 path exists for explicitly-tagged BT.601 10-bit
// content (some Sony broadcast cameras output 10-bit BT.601). Wired
// behind a public entry point but not invoked from the default
// pipeline; callers must opt in explicitly.

#[inline(always)]
fn clamp_y_10bit(v: i32) -> u16 {
    v.clamp(64, 940) as u16
}

#[inline(always)]
fn clamp_c_10bit(v: i32) -> u16 {
    v.clamp(64, 960) as u16
}

const CHROMA_CENTER_10BIT: i32 = 512;

/// Scalar 10-bit BT.601 → BT.709 reference. Same algorithm as the
/// 8-bit `bt601_to_bt709_scalar`, but operates on `u16` planes
/// (10-bit values in 0..=1023). `width` / `height` are luma
/// dimensions; chroma planes are half-resolution per axis (4:2:0).
pub fn bt601_to_bt709_planes_10bit_scalar(
    y: &mut [u16],
    cb: &mut [u16],
    cr: &mut [u16],
    width: usize,
    height: usize,
) {
    debug_assert_eq!(y.len(), width * height);
    debug_assert_eq!(cb.len(), (width / 2) * (height / 2));
    debug_assert_eq!(cr.len(), (width / 2) * (height / 2));

    let cw = width / 2;

    // Luma: Y709 = Y601 + M_Y_CB * ΔCb + M_Y_CR * ΔCr.
    for yi in 0..height {
        let cy = yi >> 1;
        for xi in 0..width {
            let cx = xi >> 1;
            let cbl = cb[cy * cw + cx] as i32 - CHROMA_CENTER_10BIT;
            let crl = cr[cy * cw + cx] as i32 - CHROMA_CENTER_10BIT;
            let y_orig = y[yi * width + xi] as i32;
            let delta = (M_Y_CB * cbl + M_Y_CR * crl + Q15_ROUND) >> Q15;
            y[yi * width + xi] = clamp_y_10bit(y_orig + delta);
        }
    }

    // Chroma: pure 2×2 chroma → chroma transform (no luma coupling).
    for v in cb.iter_mut().zip(cr.iter_mut()) {
        let (cbp, crp) = v;
        let cbl = *cbp as i32 - CHROMA_CENTER_10BIT;
        let crl = *crp as i32 - CHROMA_CENTER_10BIT;
        let new_cb = (M_CB_CB * cbl + M_CB_CR * crl + Q15_ROUND) >> Q15;
        let new_cr = (M_CR_CB * cbl + M_CR_CR * crl + Q15_ROUND) >> Q15;
        *cbp = clamp_c_10bit(new_cb + CHROMA_CENTER_10BIT);
        *crp = clamp_c_10bit(new_cr + CHROMA_CENTER_10BIT);
    }
}

/// Runtime-dispatched 10-bit BT.601 → BT.709. AVX2 on x86_64 when
/// available, scalar fallback otherwise. Squad-29.
pub fn bt601_to_bt709_planes_10bit(
    y: &mut [u16],
    cb: &mut [u16],
    cr: &mut [u16],
    width: usize,
    height: usize,
) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: avx2 feature was runtime-detected above.
            unsafe {
                bt601_to_bt709_10bit_avx2(y, cb, cr, width, height);
            }
            return;
        }
    }
    bt601_to_bt709_planes_10bit_scalar(y, cb, cr, width, height);
}

/// AVX2 specialization for the 10-bit BT.601 → BT.709 matrix
/// conversion. Mirrors `bt601_to_bt709_avx2` on `u16` lanes
/// (16 chroma samples per 256-bit register vs 16 in the 8-bit
/// path — same lane count because 8-bit path already widened
/// u8 → i16 inside the kernel).
///
/// Q15 fixed-point math via `_mm256_mulhrs_epi16` ((a*b + 0x4000) >> 15).
/// Chroma deltas are in [-512, 511] (10-bit center 512), so values
/// fit in i16 with room to spare. Coefficients ≥ 1 (M_CB_CB=33379,
/// M_CR_CR=33598) split off the identity contribution as the 8-bit
/// path does.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn bt601_to_bt709_10bit_avx2(
    y: &mut [u16],
    cb: &mut [u16],
    cr: &mut [u16],
    width: usize,
    height: usize,
) {
    unsafe {
        #[cfg(target_arch = "x86")]
        use std::arch::x86::*;
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::*;

        let cw = width / 2;
        let ch = height / 2;

        let v_m_y_cb = _mm256_set1_epi16(M_Y_CB as i16);
        let v_m_y_cr = _mm256_set1_epi16(M_Y_CR as i16);
        let v_m_cb_cb_corr = _mm256_set1_epi16((M_CB_CB - 32768) as i16);
        let v_m_cb_cr = _mm256_set1_epi16(M_CB_CR as i16);
        let v_m_cr_cb = _mm256_set1_epi16(M_CR_CB as i16);
        let v_m_cr_cr_corr = _mm256_set1_epi16((M_CR_CR - 32768) as i16);

        let v_chroma_center = _mm256_set1_epi16(CHROMA_CENTER_10BIT as i16);
        let v_chroma_lo = _mm256_set1_epi16(64);
        let v_chroma_hi = _mm256_set1_epi16(960);
        let v_luma_lo = _mm256_set1_epi16(64);
        let v_luma_hi = _mm256_set1_epi16(940);

        // ---- Luma pass ----
        // 16 chroma samples per iter → 32 luma cols on two rows (64 luma
        // outputs per iter). Same expand pattern as the 8-bit kernel:
        // duplicate each chroma delta into two adjacent luma slots
        // because chroma is 4:2:0 (one chroma per 2×2 luma block).
        for cy_idx in 0..ch {
            let y_row0 = cy_idx * 2 * width;
            let y_row1 = y_row0 + width;
            let c_row = cy_idx * cw;

            let mut cx = 0usize;
            while cx + 16 <= cw {
                // Load 16 Cb/Cr u16, compute deltas (subtract chroma center).
                let cb_i16 = _mm256_loadu_si256(cb.as_ptr().add(c_row + cx) as *const _);
                let cr_i16 = _mm256_loadu_si256(cr.as_ptr().add(c_row + cx) as *const _);
                let cbl = _mm256_sub_epi16(cb_i16, v_chroma_center);
                let crl = _mm256_sub_epi16(cr_i16, v_chroma_center);

                // Per-chroma Y delta in Q15.
                let dy_cb = _mm256_mulhrs_epi16(cbl, v_m_y_cb);
                let dy_cr = _mm256_mulhrs_epi16(crl, v_m_y_cr);
                let dy_chroma = _mm256_add_epi16(dy_cb, dy_cr);

                // Expand 16 chroma deltas → 32 per-luma deltas via stack
                // scratch (same as 8-bit kernel — see comment there for
                // the in-register-vs-scratch tradeoff).
                let mut dy_luma = [0i16; 16];
                _mm256_storeu_si256(dy_luma.as_mut_ptr() as *mut _, dy_chroma);
                let mut dy_luma_pair = [0i16; 32];
                for i in 0..16 {
                    dy_luma_pair[i * 2] = dy_luma[i];
                    dy_luma_pair[i * 2 + 1] = dy_luma[i];
                }
                let dy_luma_lo = _mm256_loadu_si256(dy_luma_pair.as_ptr() as *const _);
                let dy_luma_hi = _mm256_loadu_si256(dy_luma_pair.as_ptr().add(16) as *const _);

                // Apply to both luma rows for this chroma row.
                for row_off in [y_row0, y_row1] {
                    // Load 32 luma u16 across two 256-bit registers.
                    let y_lo = _mm256_loadu_si256(y.as_ptr().add(row_off + cx * 2) as *const _);
                    let y_hi =
                        _mm256_loadu_si256(y.as_ptr().add(row_off + cx * 2 + 16) as *const _);

                    let y_lo_out = _mm256_add_epi16(y_lo, dy_luma_lo);
                    let y_hi_out = _mm256_add_epi16(y_hi, dy_luma_hi);

                    // Clamp to limited-range luma [64, 940].
                    let y_lo_out =
                        _mm256_min_epi16(_mm256_max_epi16(y_lo_out, v_luma_lo), v_luma_hi);
                    let y_hi_out =
                        _mm256_min_epi16(_mm256_max_epi16(y_hi_out, v_luma_lo), v_luma_hi);

                    _mm256_storeu_si256(y.as_mut_ptr().add(row_off + cx * 2) as *mut _, y_lo_out);
                    _mm256_storeu_si256(
                        y.as_mut_ptr().add(row_off + cx * 2 + 16) as *mut _,
                        y_hi_out,
                    );
                }

                cx += 16;
            }

            // Scalar tail for luma of this chroma row.
            while cx < cw {
                let cb_idx = c_row + cx;
                let cbl = cb[cb_idx] as i32 - CHROMA_CENTER_10BIT;
                let crl = cr[cb_idx] as i32 - CHROMA_CENTER_10BIT;
                let delta = (M_Y_CB * cbl + M_Y_CR * crl + Q15_ROUND) >> Q15;
                let xi = cx * 2;
                for row_off in [y_row0, y_row1] {
                    for sub in 0..2 {
                        let idx = row_off + xi + sub;
                        y[idx] = clamp_y_10bit(y[idx] as i32 + delta);
                    }
                }
                cx += 1;
            }
        }

        // ---- Chroma pass (no luma coupling) ----
        // 16 samples per iter.
        let total_c = cb.len();
        let mut i = 0usize;
        while i + 16 <= total_c {
            let cb_i16 = _mm256_loadu_si256(cb.as_ptr().add(i) as *const _);
            let cr_i16 = _mm256_loadu_si256(cr.as_ptr().add(i) as *const _);
            let cbl = _mm256_sub_epi16(cb_i16, v_chroma_center);
            let crl = _mm256_sub_epi16(cr_i16, v_chroma_center);

            // Cb709 = ΔCb + (M_CB_CB-32768)·ΔCb·2^-15 + M_CB_CR·ΔCr·2^-15 + 512
            let cb_corr = _mm256_mulhrs_epi16(cbl, v_m_cb_cb_corr);
            let cb_cross = _mm256_mulhrs_epi16(crl, v_m_cb_cr);
            let new_cb = _mm256_add_epi16(_mm256_add_epi16(cbl, cb_corr), cb_cross);
            let new_cb = _mm256_add_epi16(new_cb, v_chroma_center);

            // Cr709 = ΔCr + (M_CR_CR-32768)·ΔCr·2^-15 + M_CR_CB·ΔCb·2^-15 + 512
            let cr_corr = _mm256_mulhrs_epi16(crl, v_m_cr_cr_corr);
            let cr_cross = _mm256_mulhrs_epi16(cbl, v_m_cr_cb);
            let new_cr = _mm256_add_epi16(_mm256_add_epi16(crl, cr_corr), cr_cross);
            let new_cr = _mm256_add_epi16(new_cr, v_chroma_center);

            // Clamp [64, 960].
            let new_cb = _mm256_min_epi16(_mm256_max_epi16(new_cb, v_chroma_lo), v_chroma_hi);
            let new_cr = _mm256_min_epi16(_mm256_max_epi16(new_cr, v_chroma_lo), v_chroma_hi);

            _mm256_storeu_si256(cb.as_mut_ptr().add(i) as *mut _, new_cb);
            _mm256_storeu_si256(cr.as_mut_ptr().add(i) as *mut _, new_cr);

            i += 16;
        }

        // Scalar tail for chroma.
        while i < total_c {
            let cbl = cb[i] as i32 - CHROMA_CENTER_10BIT;
            let crl = cr[i] as i32 - CHROMA_CENTER_10BIT;
            let new_cb = (M_CB_CB * cbl + M_CB_CR * crl + Q15_ROUND) >> Q15;
            let new_cr = (M_CR_CB * cbl + M_CR_CR * crl + Q15_ROUND) >> Q15;
            cb[i] = clamp_c_10bit(new_cb + CHROMA_CENTER_10BIT);
            cr[i] = clamp_c_10bit(new_cr + CHROMA_CENTER_10BIT);
            i += 1;
        }
    }
}

fn recolor_yuv420p_bt601_to_bt709(frame: &VideoFrame) -> Result<VideoFrame> {
    let w = frame.width as usize;
    let h = frame.height as usize;
    let y_size = w * h;
    let c_size = y_size / 4;

    if frame.data.len() < y_size + 2 * c_size {
        bail!(
            "frame data too short for yuv420p {}x{}: {} bytes",
            w,
            h,
            frame.data.len()
        );
    }
    if !w.is_multiple_of(2) || !h.is_multiple_of(2) {
        bail!(
            "BT.601→BT.709 requires even dimensions for 4:2:0 subsampling; got {}x{}",
            w,
            h
        );
    }

    let mut y = frame.data[..y_size].to_vec();
    let mut cb = frame.data[y_size..y_size + c_size].to_vec();
    let mut cr = frame.data[y_size + c_size..y_size + 2 * c_size].to_vec();

    bt601_to_bt709_planes(&mut y, &mut cb, &mut cr, w, h);

    let mut out = BytesMut::with_capacity(y_size + 2 * c_size);
    out.extend_from_slice(&y);
    out.extend_from_slice(&cb);
    out.extend_from_slice(&cr);

    Ok(VideoFrame::new(
        out.freeze(),
        frame.width,
        frame.height,
        frame.format,
        ColorSpace::Bt709,
        frame.pts,
    ))
}

// =============================================================================
// 4:4:4 → 4:2:0 chroma downsample (Squad-31, roadmap #6).
// =============================================================================
//
// ProRes 4444 (and other 4:4:4 sources) decode at full chroma resolution —
// Cb / Cr planes match the luma plane in both dimensions. The encoder side
// (rav1e + HW backends) only accepts 4:2:0, where chroma is half-resolution
// in both axes. This module bridges the gap with a 2×2 box-average filter:
// for each 2×2 block of source chroma, output one chroma sample equal to
// the rounded mean. Y plane is unchanged (full-resolution luma in both
// formats — 4:4:4 and 4:2:0 differ only in chroma layout).
//
// Filter choice: 2×2 box average. The simplest correct filter for 4:4:4
// → 4:2:0 chroma siting (MPEG-2 left-aligned). For each output sample at
// (cx, cy), input samples are (2*cx, 2*cy), (2*cx+1, 2*cy), (2*cx, 2*cy+1),
// (2*cx+1, 2*cy+1). Output is `(s00 + s01 + s10 + s11 + 2) >> 2` —
// rounding by adding half the divisor before truncating shift.
//
// Higher-quality alternatives (6-tap separable FIR per BT.601/709 H.131,
// or a Lanczos-2 horizontal+vertical pair) are deferred to a follow-up;
// they cost ~10× the cycles for ~0.3 dB chroma PSNR improvement, which
// most consumer transcoders consider not worth it. The box average matches
// libswscale's default 4:4:4 → 4:2:0 path when no scaler is requested.
//
// Odd-dimension policy: when the source width or height is odd, the output
// dimensions round up (`(src + 1) / 2`), and the rightmost / bottom row of
// 2×2 blocks straddles a single source row/column. We **clamp** — the
// missing neighbour reuses the in-bounds sample. Clamping vs replication
// is identical for a 1-pixel boundary; we pick clamping because it's the
// simplest scalar implementation and matches what libswscale does.
//
// Alpha plane (Yuva444p10le): the 4:2:0 encoder format has no alpha. We
// **drop** alpha with a single warn-log (in pipeline integration). AV1
// has alpha support in some experimental profiles but rav1e 0.7 doesn't
// expose it, and pre-compositing onto a black background changes pixel
// values — keying / compositing on the source side would have already
// happened. Documented in SUPPORTED.md.

/// 2×2 box-average chroma downsample for 8-bit `Yuv444p` → `Yuv420p`.
/// Y plane is copied verbatim; Cb and Cr planes shrink 2× in each axis
/// with rounded averages.
///
/// Output dimensions: chroma plane is `((width + 1) / 2) × ((height + 1) / 2)`,
/// which matches the encoder's 4:2:0 expectation for any input dims
/// (odd or even). For the common even case (e.g. 1920×1080) this is
/// 960×540 chroma per plane.
///
/// Returns the new packed `Yuv420p` byte buffer (Y || Cb || Cr).
pub fn downsample_chroma_444_to_420(
    y: &[u8],
    cb: &[u8],
    cr: &[u8],
    width: usize,
    height: usize,
) -> Vec<u8> {
    debug_assert_eq!(y.len(), width * height, "Y plane size");
    debug_assert_eq!(cb.len(), width * height, "Cb plane size (4:4:4)");
    debug_assert_eq!(cr.len(), width * height, "Cr plane size (4:4:4)");

    let cw = width.div_ceil(2);
    let ch = height.div_ceil(2);

    let mut out = Vec::with_capacity(width * height + 2 * cw * ch);

    // Y plane: straight copy. Luma resolution is identical between
    // 4:4:4 and 4:2:0.
    out.extend_from_slice(y);

    // Cb then Cr — same algorithm per plane.
    for plane in [cb, cr] {
        for cy in 0..ch {
            // Source rows: 2*cy and 2*cy+1, clamped to height-1.
            let y0 = 2 * cy;
            let y1 = (y0 + 1).min(height - 1);
            for cx in 0..cw {
                let x0 = 2 * cx;
                let x1 = (x0 + 1).min(width - 1);
                // Box average. 8-bit max is 255 × 4 = 1020, fits in u16.
                let s00 = plane[y0 * width + x0] as u16;
                let s01 = plane[y0 * width + x1] as u16;
                let s10 = plane[y1 * width + x0] as u16;
                let s11 = plane[y1 * width + x1] as u16;
                let avg = ((s00 + s01 + s10 + s11 + 2) >> 2) as u8;
                out.push(avg);
            }
        }
    }

    out
}

/// 10-bit variant for `Yuv444p10le` → `Yuv420p10le`. Operates on `u16`
/// samples in the 0..=1023 range; output samples are written as LE
/// `u16` bytes packed alongside the copied Y plane.
///
/// Accumulator: `u32`. Worst case 4 × 1023 + 2 = 4094 fits comfortably
/// in `u16` already, but `u32` keeps the math aligned with the spec
/// recommendation (BT.709 Annex A) and allows easy future swap to a
/// wider filter without overflow rework.
pub fn downsample_chroma_444_to_420_10bit(
    y: &[u16],
    cb: &[u16],
    cr: &[u16],
    width: usize,
    height: usize,
) -> Vec<u8> {
    debug_assert_eq!(y.len(), width * height, "Y plane samples");
    debug_assert_eq!(cb.len(), width * height, "Cb plane samples (4:4:4)");
    debug_assert_eq!(cr.len(), width * height, "Cr plane samples (4:4:4)");

    let cw = width.div_ceil(2);
    let ch = height.div_ceil(2);
    let total_samples = width * height + 2 * cw * ch;
    let mut out = Vec::with_capacity(total_samples * 2);

    // Y plane: emit as u16 LE bytes. Y is unchanged (full luma).
    for &s in y {
        out.extend_from_slice(&s.to_le_bytes());
    }

    for plane in [cb, cr] {
        for cy in 0..ch {
            let y0 = 2 * cy;
            let y1 = (y0 + 1).min(height - 1);
            for cx in 0..cw {
                let x0 = 2 * cx;
                let x1 = (x0 + 1).min(width - 1);
                let s00 = plane[y0 * width + x0] as u32;
                let s01 = plane[y0 * width + x1] as u32;
                let s10 = plane[y1 * width + x0] as u32;
                let s11 = plane[y1 * width + x1] as u32;
                let avg = ((s00 + s01 + s10 + s11 + 2) >> 2) as u16;
                out.extend_from_slice(&avg.to_le_bytes());
            }
        }
    }

    out
}

/// High-level frame-shaped wrapper. Takes a `Yuv444p10le` /
/// `Yuva444p10le` `VideoFrame` and returns a `Yuv420p10le`
/// `VideoFrame` ready for the 10-bit AV1 encoder. Alpha plane (if
/// present) is **dropped** with a warn-log — see module docstring for
/// rationale. 8-bit equivalent (`Yuv444p` → `Yuv420p`) follows the
/// same pattern, plumbed through `downsample_chroma_444_to_420`.
///
/// Errors if the source format is not 4:4:4.
pub fn downsample_444_to_420_frame(frame: &VideoFrame) -> Result<VideoFrame> {
    let w = frame.width as usize;
    let h = frame.height as usize;
    if w == 0 || h == 0 {
        bail!("zero-dimension frame");
    }

    match frame.format {
        PixelFormat::Yuv444p => {
            let plane = w * h;
            if frame.data.len() < 3 * plane {
                bail!(
                    "Yuv444p frame data too short for {}x{}: {} bytes",
                    w,
                    h,
                    frame.data.len()
                );
            }
            let y = &frame.data[..plane];
            let cb = &frame.data[plane..2 * plane];
            let cr = &frame.data[2 * plane..3 * plane];
            let out = downsample_chroma_444_to_420(y, cb, cr, w, h);
            Ok(VideoFrame::new(
                Bytes::from(out),
                frame.width,
                frame.height,
                PixelFormat::Yuv420p,
                frame.color_space,
                frame.pts,
            ))
        }
        PixelFormat::Yuv444p10le | PixelFormat::Yuva444p10le => {
            let plane = w * h;
            // 10-bit (or 16-bit alpha) is 2 bytes/sample. Y/Cb/Cr always
            // 10-bit, alpha (if present) is 16-bit, but layout is per-
            // plane LE u16 either way. We only consume the first three
            // planes; alpha (plane 4) is dropped on the floor.
            let needed = if frame.format == PixelFormat::Yuva444p10le {
                4 * plane * 2
            } else {
                3 * plane * 2
            };
            if frame.data.len() < needed {
                bail!(
                    "{:?} frame data too short for {}x{}: {} bytes (need {})",
                    frame.format,
                    w,
                    h,
                    frame.data.len(),
                    needed
                );
            }
            // Decode three u16 LE planes from the source bytes.
            let y = read_u16le(&frame.data[..plane * 2]);
            let cb = read_u16le(&frame.data[plane * 2..2 * plane * 2]);
            let cr = read_u16le(&frame.data[2 * plane * 2..3 * plane * 2]);

            if frame.format == PixelFormat::Yuva444p10le {
                tracing::warn!(
                    pts = frame.pts,
                    "dropping alpha plane on 4:4:4→4:2:0 downsample (rav1e 0.7 has no alpha; pipeline target is Yuv420p10le)"
                );
            }

            let out = downsample_chroma_444_to_420_10bit(&y, &cb, &cr, w, h);
            Ok(VideoFrame::new(
                Bytes::from(out),
                frame.width,
                frame.height,
                PixelFormat::Yuv420p10le,
                frame.color_space,
                frame.pts,
            ))
        }
        other => bail!(
            "downsample_444_to_420_frame: expected 4:4:4 input, got {:?}",
            other
        ),
    }
}

// =============================================================================
// Bilinear scaler — scalar + AVX2 dispatch.
// =============================================================================

pub fn scale_frame(
    frame: &VideoFrame,
    target_width: u32,
    target_height: u32,
) -> Result<VideoFrame> {
    if frame.width == target_width && frame.height == target_height {
        return Ok(frame.clone());
    }

    match frame.format {
        PixelFormat::Yuv420p => scale_frame_8bit(frame, target_width, target_height),
        // 10-bit 4:2:0 path. Squad-19 shipped scalar; Squad-29 added
        // AVX2 specialization (16 × u16 lanes per iter, Q15 bilinear
        // weights via `_mm256_mulhrs_epi16`). Runtime-dispatched by
        // `bilinear_scale_plane_u16` (`is_x86_feature_detected!("avx2")`).
        PixelFormat::Yuv420p10le => scale_frame_10bit(frame, target_width, target_height),
        _ => bail!(
            "scaling only implemented for Yuv420p / Yuv420p10le; got {:?}",
            frame.format
        ),
    }
}

fn scale_frame_8bit(
    frame: &VideoFrame,
    target_width: u32,
    target_height: u32,
) -> Result<VideoFrame> {
    let src_w = frame.width as usize;
    let src_h = frame.height as usize;
    let dst_w = target_width as usize;
    let dst_h = target_height as usize;

    let src_y_size = src_w * src_h;
    let dst_y_size = dst_w * dst_h;
    let dst_uv_size = dst_y_size / 4;

    let mut out = BytesMut::with_capacity(dst_y_size + dst_uv_size * 2);

    // Bilinear scale Y plane
    let y_plane = &frame.data[..src_y_size];
    out.extend(bilinear_scale_plane(y_plane, src_w, src_h, dst_w, dst_h));

    // Scale U plane
    let u_offset = src_y_size;
    let u_plane = &frame.data[u_offset..u_offset + src_y_size / 4];
    out.extend(bilinear_scale_plane(
        u_plane,
        src_w / 2,
        src_h / 2,
        dst_w / 2,
        dst_h / 2,
    ));

    // Scale V plane
    let v_offset = u_offset + src_y_size / 4;
    let v_plane = &frame.data[v_offset..v_offset + src_y_size / 4];
    out.extend(bilinear_scale_plane(
        v_plane,
        src_w / 2,
        src_h / 2,
        dst_w / 2,
        dst_h / 2,
    ));

    Ok(VideoFrame::new(
        out.freeze(),
        target_width,
        target_height,
        frame.format,
        frame.color_space,
        frame.pts,
    ))
}

/// 10-bit `Yuv420p10le` bilinear scaler. Each plane is `u16` LE in the
/// 0..=1023 range. Operates on 16-bit samples directly; output sample
/// range is preserved (10-bit values stored in 16-bit containers).
///
/// Per-plane work runs through `bilinear_scale_plane_u16`, which
/// runtime-dispatches to AVX2 (Squad-29; 16 × u16 lanes per iter)
/// when `is_x86_feature_detected!("avx2")` and falls back to the
/// scalar f64 path (Squad-19) otherwise.
fn scale_frame_10bit(
    frame: &VideoFrame,
    target_width: u32,
    target_height: u32,
) -> Result<VideoFrame> {
    let src_w = frame.width as usize;
    let src_h = frame.height as usize;
    let dst_w = target_width as usize;
    let dst_h = target_height as usize;

    let bytes_per_sample = 2usize;
    let src_y_size_samples = src_w * src_h;
    let src_y_size_bytes = src_y_size_samples * bytes_per_sample;
    let src_c_size_samples = (src_w / 2) * (src_h / 2);
    let src_c_size_bytes = src_c_size_samples * bytes_per_sample;

    if frame.data.len() < src_y_size_bytes + 2 * src_c_size_bytes {
        bail!(
            "10-bit frame data too short for {}x{}: {} bytes",
            src_w,
            src_h,
            frame.data.len()
        );
    }

    let dst_y_size_samples = dst_w * dst_h;
    let dst_c_size_samples = (dst_w / 2) * (dst_h / 2);
    let dst_total_bytes = (dst_y_size_samples + 2 * dst_c_size_samples) * bytes_per_sample;

    // Decode planes from LE bytes into u16 buffers.
    let y_plane = read_u16le(&frame.data[..src_y_size_bytes]);
    let u_plane = read_u16le(&frame.data[src_y_size_bytes..src_y_size_bytes + src_c_size_bytes]);
    let v_plane = read_u16le(
        &frame.data[src_y_size_bytes + src_c_size_bytes..src_y_size_bytes + 2 * src_c_size_bytes],
    );

    // Squad-29: runtime-dispatched (AVX2 when available, scalar fallback).
    let y_dst = bilinear_scale_plane_u16(&y_plane, src_w, src_h, dst_w, dst_h);
    let u_dst = bilinear_scale_plane_u16(&u_plane, src_w / 2, src_h / 2, dst_w / 2, dst_h / 2);
    let v_dst = bilinear_scale_plane_u16(&v_plane, src_w / 2, src_h / 2, dst_w / 2, dst_h / 2);

    let mut out = BytesMut::with_capacity(dst_total_bytes);
    write_u16le(&mut out, &y_dst);
    write_u16le(&mut out, &u_dst);
    write_u16le(&mut out, &v_dst);

    Ok(VideoFrame::new(
        out.freeze(),
        target_width,
        target_height,
        frame.format,
        frame.color_space,
        frame.pts,
    ))
}

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

/// Scalar bilinear scale on `u16` (10-bit) samples. Mirrors the 8-bit
/// `bilinear_scale_plane_scalar` algorithm; the only differences are
/// the wider sample type and the absence of u8 saturation at the
/// output (10-bit values up to 1023 fit comfortably in u16, no
/// overflow risk in the f64 intermediate).
pub fn bilinear_scale_plane_u16_scalar(
    src: &[u16],
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
) -> Vec<u16> {
    let mut dst = vec![0u16; dst_w * dst_h];
    let x_ratio = src_w as f64 / dst_w as f64;
    let y_ratio = src_h as f64 / dst_h as f64;

    for dy in 0..dst_h {
        let sy = (dy as f64 * y_ratio).min((src_h - 1) as f64);
        let y0 = sy as usize;
        let y1 = (y0 + 1).min(src_h - 1);
        let fy = sy - y0 as f64;

        for dx in 0..dst_w {
            let sx = (dx as f64 * x_ratio).min((src_w - 1) as f64);
            let x0 = sx as usize;
            let x1 = (x0 + 1).min(src_w - 1);
            let fx = sx - x0 as f64;

            let p00 = src[y0 * src_w + x0] as f64;
            let p10 = src[y0 * src_w + x1] as f64;
            let p01 = src[y1 * src_w + x0] as f64;
            let p11 = src[y1 * src_w + x1] as f64;

            let val = p00 * (1.0 - fx) * (1.0 - fy)
                + p10 * fx * (1.0 - fy)
                + p01 * (1.0 - fx) * fy
                + p11 * fx * fy;

            // Round to nearest, clamp to the 10-bit max (1023). The
            // input is already in 0..=1023 so an in-range bilinear
            // combination cannot exceed 1023 in exact arithmetic; the
            // clamp is defensive against fp rounding pushing 1023.0
            // → 1024.0.
            dst[dy * dst_w + dx] = val.round().clamp(0.0, 1023.0) as u16;
        }
    }
    dst
}

/// Runtime-dispatched 10-bit bilinear scale. AVX2 on x86_64 when
/// available; falls back to `bilinear_scale_plane_u16_scalar` otherwise.
///
/// Squad-29 (2026-04-17) added the AVX2 path. 256-bit registers
/// process 16 × u16 samples per iteration. Internally the same
/// Q15 fixed-point math as the 8-bit AVX2 path, but `_mm256_mulhrs_epi16`
/// expects signed lanes — 10-bit samples (0..=1023) fit comfortably
/// in i16 with no shift gymnastics. Output is clamped to 10-bit
/// range (0..=1023). Scalar tail (when `dst_w % 16 != 0` or width
/// < 16) reuses the scalar f64 math row-by-row.
///
/// Performance target on 1080p→720p: ≥3× over scalar (realistic
/// floor for u16 lanes vs u8). Bench in
/// `crates/codec/benches/bilinear.rs::bilinear_10bit_avx2_vs_scalar`.
pub fn bilinear_scale_plane_u16(
    src: &[u16],
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
) -> Vec<u16> {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        // 16-lane AVX2 path; gate on dst_w >= 16 so the main loop
        // runs at least once per row. Narrower outputs fall back to
        // scalar (cheap — narrow strips aren't a hotspot).
        if std::is_x86_feature_detected!("avx2") && dst_w >= 16 {
            // SAFETY: avx2 runtime-detected.
            return unsafe { bilinear_scale_plane_u16_avx2(src, src_w, src_h, dst_w, dst_h) };
        }
    }
    bilinear_scale_plane_u16_scalar(src, src_w, src_h, dst_w, dst_h)
}

/// AVX2 specialization for the 10-bit bilinear scaler. Processes
/// 16 × u16 destination samples per iteration via 256-bit registers.
///
/// Algorithm mirrors `bilinear_scale_plane_avx2` (the 8-bit AVX2
/// path) with two differences:
/// 1. Lanes are `u16` (16 per 256-bit reg, vs 32 × `u8` in 8-bit).
///    No need to widen u8 → i16 inside the kernel — samples are
///    already 16-bit.
/// 2. No need to shift sample values up (Q7 trick) before the
///    Q15 multiply: 10-bit values in 0..=1023 fit in i16 unshifted.
///    Final output is straight-clamped to [0, 1023] without saturating
///    pack-down.
///
/// Q15 fixed-point weights via `_mm256_mulhrs_epi16` ((a*b+0x4000)>>15).
/// `mulhrs` operates on signed lanes; weights are in [0, 32767]
/// (one ULP shy of 32768; matches the 8-bit AVX2 trick to keep
/// inputs i16-safe).
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn bilinear_scale_plane_u16_avx2(
    src: &[u16],
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
) -> Vec<u16> {
    unsafe {
        #[cfg(target_arch = "x86")]
        use std::arch::x86::*;
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::*;

        let mut dst = vec![0u16; dst_w * dst_h];

        // Q32 fixed-point ratios — same precision pattern as the 8-bit
        // AVX2 path so the rounding edge cases land identically across
        // the two specializations.
        let x_step = ((src_w as u64) << 32) / (dst_w as u64);
        let y_step = ((src_h as u64) << 32) / (dst_h as u64);

        // Precompute per-dst-x source x0 + Q15 fractional weights.
        let mut x0s: Vec<u32> = vec![0; dst_w];
        let mut x1s: Vec<u32> = vec![0; dst_w];
        let mut fxs_q15: Vec<i16> = vec![0; dst_w];
        let mut one_minus_fxs_q15: Vec<i16> = vec![0; dst_w];
        for dx in 0..dst_w {
            let sx_32_32 = (dx as u64) * x_step;
            let x0_full = (sx_32_32 >> 32) as usize;
            let x0 = x0_full.min(src_w - 1);
            let fx_q16 = ((sx_32_32 >> 16) & 0xFFFF) as u32;
            // Convert Q16 → Q15 in [0, 32767]; same clamp trick as 8-bit.
            let fx_q15 = ((fx_q16 as i32) >> 1).min(32767) as i16;
            if x0 >= src_w - 1 {
                x0s[dx] = (src_w - 1) as u32;
                x1s[dx] = (src_w - 1) as u32;
                fxs_q15[dx] = 0;
                one_minus_fxs_q15[dx] = 32767;
            } else {
                x0s[dx] = x0 as u32;
                x1s[dx] = (x0 + 1) as u32;
                fxs_q15[dx] = fx_q15;
                one_minus_fxs_q15[dx] = 32767 - fx_q15;
            }
        }

        let v_max = _mm256_set1_epi16(1023);
        let v_zero = _mm256_setzero_si256();

        for dy in 0..dst_h {
            let sy_32_32 = (dy as u64) * y_step;
            let y0_full = (sy_32_32 >> 32) as usize;
            let y0 = y0_full.min(src_h - 1);
            let fy_q16 = ((sy_32_32 >> 16) & 0xFFFF) as u32;
            let y1 = (y0 + 1).min(src_h - 1);
            let fy_q15 = ((fy_q16 as i32) >> 1).min(32767) as i16;
            let one_minus_fy_q15 = 32767i16 - fy_q15;

            let row0 = y0 * src_w;
            let row1 = y1 * src_w;
            let dst_row = dy * dst_w;

            let v_fy = _mm256_set1_epi16(fy_q15);
            let v_one_minus_fy = _mm256_set1_epi16(one_minus_fy_q15);

            let mut dx = 0usize;
            while dx + 16 <= dst_w {
                // Gather 16 p00 / p10 / p01 / p11 u16 values into stack
                // scratch buffers, then load as 256-bit registers.
                // Same approach as the 8-bit AVX2 — bilinear inputs don't
                // align contiguously (x0/x1 are arbitrary), so a scalar
                // gather + aligned reload is the cheapest path.
                let mut p00_buf = [0u16; 16];
                let mut p10_buf = [0u16; 16];
                let mut p01_buf = [0u16; 16];
                let mut p11_buf = [0u16; 16];
                for i in 0..16 {
                    let x0 = x0s[dx + i] as usize;
                    let x1 = x1s[dx + i] as usize;
                    p00_buf[i] = *src.get_unchecked(row0 + x0);
                    p10_buf[i] = *src.get_unchecked(row0 + x1);
                    p01_buf[i] = *src.get_unchecked(row1 + x0);
                    p11_buf[i] = *src.get_unchecked(row1 + x1);
                }

                // Load as 256-bit (16 × u16). Treated as i16 for the
                // signed `mulhrs` — 10-bit samples max=1023, well under
                // i16_max (32767), so reinterpret is bit-identical and
                // safe.
                let p00 = _mm256_loadu_si256(p00_buf.as_ptr() as *const _);
                let p10 = _mm256_loadu_si256(p10_buf.as_ptr() as *const _);
                let p01 = _mm256_loadu_si256(p01_buf.as_ptr() as *const _);
                let p11 = _mm256_loadu_si256(p11_buf.as_ptr() as *const _);

                // Per-lane fx / (1-fx).
                let v_fx = _mm256_loadu_si256(fxs_q15.as_ptr().add(dx) as *const _);
                let v_one_minus_fx =
                    _mm256_loadu_si256(one_minus_fxs_q15.as_ptr().add(dx) as *const _);

                // mulhrs: (a*b + 0x4000) >> 15. Inputs are 10-bit
                // (0..=1023), weights are Q15 (0..=32767). Product max
                // ≈ 1023 * 32767 ≈ 33.5M; after >> 15 the value is
                // ≤ 1023, so signed i16 is plenty.
                //
                // Important: because samples are unshifted (unlike the
                // 8-bit kernel which multiplied by 128 first), the
                // post-multiply value retains its full sample magnitude.
                // No final shift-down is needed — `top` and `bottom`
                // are already in the 0..=1023 range.
                let top = _mm256_add_epi16(
                    _mm256_mulhrs_epi16(p00, v_one_minus_fx),
                    _mm256_mulhrs_epi16(p10, v_fx),
                );
                let bottom = _mm256_add_epi16(
                    _mm256_mulhrs_epi16(p01, v_one_minus_fx),
                    _mm256_mulhrs_epi16(p11, v_fx),
                );

                // Vertical interp. Same Q15 → 10-bit scale.
                let out_i16 = _mm256_add_epi16(
                    _mm256_mulhrs_epi16(top, v_one_minus_fy),
                    _mm256_mulhrs_epi16(bottom, v_fy),
                );

                // Clamp to [0, 1023] — defensive against the Q15 round
                // trick pushing exactly-1023 inputs to 1024 in extreme
                // edge cases. Use signed min/max on i16 (values are
                // non-negative for in-range 10-bit input).
                let clamped = _mm256_min_epi16(_mm256_max_epi16(out_i16, v_zero), v_max);

                _mm256_storeu_si256(dst.as_mut_ptr().add(dst_row + dx) as *mut _, clamped);

                dx += 16;
            }

            // Scalar tail. Mirrors `bilinear_scale_plane_u16_scalar` row
            // math so parity with the scalar function is byte-exact in
            // the tail (modulo the ±1-LSB rounding tolerance the SIMD
            // main loop carries vs scalar f64).
            while dx < dst_w {
                let x0 = x0s[dx] as usize;
                let x1 = x1s[dx] as usize;
                let fx = fxs_q15[dx] as f64 / 32768.0;
                let fy = fy_q15 as f64 / 32768.0;

                let p00 = src[row0 + x0] as f64;
                let p10 = src[row0 + x1] as f64;
                let p01 = src[row1 + x0] as f64;
                let p11 = src[row1 + x1] as f64;

                let val = p00 * (1.0 - fx) * (1.0 - fy)
                    + p10 * fx * (1.0 - fy)
                    + p01 * (1.0 - fx) * fy
                    + p11 * fx * fy;
                dst[dst_row + dx] = val.round().clamp(0.0, 1023.0) as u16;
                dx += 1;
            }
        }

        dst
    }
}

/// Runtime-dispatched bilinear scale. AVX2 on x86_64 when available.
pub fn bilinear_scale_plane(
    src: &[u8],
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
) -> Vec<u8> {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        // perf: AVX2 specialization — Q16 fixed-point 2-tap horiz ×
        // 2-tap vert, 16 dst pixels / iter. Benched 3-4× faster than
        // the f64 scalar on 1080p→720p.
        if std::is_x86_feature_detected!("avx2") && dst_w >= 16 {
            // SAFETY: avx2 runtime-detected.
            return unsafe { bilinear_scale_plane_avx2(src, src_w, src_h, dst_w, dst_h) };
        }
    }
    bilinear_scale_plane_scalar(src, src_w, src_h, dst_w, dst_h)
}

pub fn bilinear_scale_plane_scalar(
    src: &[u8],
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
) -> Vec<u8> {
    let mut dst = vec![0u8; dst_w * dst_h];
    let x_ratio = src_w as f64 / dst_w as f64;
    let y_ratio = src_h as f64 / dst_h as f64;

    for dy in 0..dst_h {
        let sy = (dy as f64 * y_ratio).min((src_h - 1) as f64);
        let y0 = sy as usize;
        let y1 = (y0 + 1).min(src_h - 1);
        let fy = sy - y0 as f64;

        for dx in 0..dst_w {
            let sx = (dx as f64 * x_ratio).min((src_w - 1) as f64);
            let x0 = sx as usize;
            let x1 = (x0 + 1).min(src_w - 1);
            let fx = sx - x0 as f64;

            let p00 = src[y0 * src_w + x0] as f64;
            let p10 = src[y0 * src_w + x1] as f64;
            let p01 = src[y1 * src_w + x0] as f64;
            let p11 = src[y1 * src_w + x1] as f64;

            let val = p00 * (1.0 - fx) * (1.0 - fy)
                + p10 * fx * (1.0 - fy)
                + p01 * (1.0 - fx) * fy
                + p11 * fx * fy;

            dst[dy * dst_w + dx] = val.round() as u8;
        }
    }
    dst
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn bilinear_scale_plane_avx2(
    src: &[u8],
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
) -> Vec<u8> {
    unsafe {
        #[cfg(target_arch = "x86")]
        use std::arch::x86::*;
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::*;

        let mut dst = vec![0u8; dst_w * dst_h];

        // Q16 fixed-point ratios. src_w/dst_w as 16.16.
        //
        // sx_q16(dx) = dx * (src_w << 16) / dst_w
        // Using integer math avoids the f64 hotpath. The sampling
        // convention matches the scalar (origin-at-pixel-corner); picking
        // a different convention would change the output but also the
        // reference, so we mirror scalar precisely.
        let x_step = ((src_w as u64) << 32) / (dst_w as u64); // 32.32 keeps precision for wide src
        let y_step = ((src_h as u64) << 32) / (dst_h as u64);

        // Precompute per-dst-x source x0 and fx (Q16 weight for src[x1]).
        let mut x0s: Vec<u32> = vec![0; dst_w];
        let mut fxs: Vec<u16> = vec![0; dst_w];
        for dx in 0..dst_w {
            let sx_32_32 = (dx as u64) * x_step; // source x in 32.32
            let x0_full = (sx_32_32 >> 32) as usize;
            let x0 = x0_full.min(src_w - 1);
            let fx_q16 = ((sx_32_32 >> 16) & 0xFFFF) as u16; // Q16 fraction
            // Clamp to src_w-1 if we'd run off the end
            if x0 >= src_w - 1 {
                x0s[dx] = (src_w - 1) as u32;
                fxs[dx] = 0;
            } else {
                x0s[dx] = x0 as u32;
                fxs[dx] = fx_q16;
            }
        }

        // Bake (1-fx) into a paired vector for mulhrs-style math. We'll
        // compute:
        //   top    = p00 * (1-fx) + p10 * fx
        //   bottom = p01 * (1-fx) + p11 * fx
        //   out    = top * (1-fy) + bottom * fy
        // All in Q15 (mulhrs) with u16 input shifted down by 1 bit, or in
        // Q14 with straightforward pmulhw.
        //
        // Use Q15 + mulhrs_epi16 for the fractional weights. Fractional
        // weights are in [0, 32768] — 32768 overflows i16, so clamp at
        // 32767 (error ≤ 1/32768 ≈ 0, sub-LSB for 8-bit output).
        let mut fx_q15: Vec<i16> = vec![0; dst_w];
        let mut one_minus_fx_q15: Vec<i16> = vec![0; dst_w];
        for dx in 0..dst_w {
            // Convert Q16 weight to Q15 in [0, 32767].
            let fxq15 = (fxs[dx] as i32 >> 1).min(32767) as i16;
            fx_q15[dx] = fxq15;
            one_minus_fx_q15[dx] = 32767 - fxq15;
        }

        for dy in 0..dst_h {
            let sy_32_32 = (dy as u64) * y_step;
            let y0_full = (sy_32_32 >> 32) as usize;
            let y0 = y0_full.min(src_h - 1);
            let fy_q16 = ((sy_32_32 >> 16) & 0xFFFF) as u32;
            let y1 = (y0 + 1).min(src_h - 1);
            let fy_q15 = ((fy_q16 as i32) >> 1).min(32767) as i16;
            let one_minus_fy_q15 = 32767i16 - fy_q15;

            let row0 = y0 * src_w;
            let row1 = y1 * src_w;
            let dst_row = dy * dst_w;

            // Per-dx we need p00/p10/p01/p11. Bilinear inputs don't align
            // contiguously (x0/x1 are arbitrary), so we gather by
            // scalar-load-then-pack for 16 dst-x per iteration.
            let v_fy = _mm256_set1_epi16(fy_q15);
            let v_one_minus_fy = _mm256_set1_epi16(one_minus_fy_q15);

            let mut dx = 0usize;
            while dx + 16 <= dst_w {
                // Gather 16 p00, p10, p01, p11 into 16-lane u8 buffers.
                // Use stack scratch and an unaligned load.
                let mut p00_buf = [0u8; 16];
                let mut p10_buf = [0u8; 16];
                let mut p01_buf = [0u8; 16];
                let mut p11_buf = [0u8; 16];
                for i in 0..16 {
                    let x0 = x0s[dx + i] as usize;
                    let x1 = (x0 + 1).min(src_w - 1);
                    p00_buf[i] = *src.get_unchecked(row0 + x0);
                    p10_buf[i] = *src.get_unchecked(row0 + x1);
                    p01_buf[i] = *src.get_unchecked(row1 + x0);
                    p11_buf[i] = *src.get_unchecked(row1 + x1);
                }

                // Widen to i16.
                let p00 = _mm256_cvtepu8_epi16(_mm_loadu_si128(p00_buf.as_ptr() as *const _));
                let p10 = _mm256_cvtepu8_epi16(_mm_loadu_si128(p10_buf.as_ptr() as *const _));
                let p01 = _mm256_cvtepu8_epi16(_mm_loadu_si128(p01_buf.as_ptr() as *const _));
                let p11 = _mm256_cvtepu8_epi16(_mm_loadu_si128(p11_buf.as_ptr() as *const _));

                // Shift u8 (0..255) up to the top of i16's signed range so
                // mulhrs_epi16 retains precision. Each u8 value × 128 is in
                // [0, 32640] — safe for signed mul.
                let p00 = _mm256_slli_epi16::<7>(p00);
                let p10 = _mm256_slli_epi16::<7>(p10);
                let p01 = _mm256_slli_epi16::<7>(p01);
                let p11 = _mm256_slli_epi16::<7>(p11);

                // Load per-lane fx / (1-fx).
                let v_fx = _mm256_loadu_si256(fx_q15.as_ptr().add(dx) as *const _);
                let v_one_minus_fx =
                    _mm256_loadu_si256(one_minus_fx_q15.as_ptr().add(dx) as *const _);

                // Horizontal interp: top = p00*(1-fx) + p10*fx, in Q15.
                let top = _mm256_add_epi16(
                    _mm256_mulhrs_epi16(p00, v_one_minus_fx),
                    _mm256_mulhrs_epi16(p10, v_fx),
                );
                let bottom = _mm256_add_epi16(
                    _mm256_mulhrs_epi16(p01, v_one_minus_fx),
                    _mm256_mulhrs_epi16(p11, v_fx),
                );

                // Vertical interp. top/bottom are Q15-scaled of the Q7
                // u8 → so still in the ~Q7 range (approx 0..254). Apply
                // (1-fy)/fy via mulhrs to get final Q7, then shift down 7
                // to recover the u8. mulhrs: (top * (1-fy) + 0x4000) >> 15.
                let out_q7 = _mm256_add_epi16(
                    _mm256_mulhrs_epi16(top, v_one_minus_fy),
                    _mm256_mulhrs_epi16(bottom, v_fy),
                );
                // Shift back: (x + 64) >> 7 for round-to-nearest.
                let rounded = _mm256_add_epi16(out_q7, _mm256_set1_epi16(64));
                let shifted = _mm256_srai_epi16::<7>(rounded);

                // Saturating pack i16 → u8 (16 lanes).
                let packed = _mm256_packus_epi16(shifted, shifted);
                // packus interleaves 128-lane halves — permute so low
                // 16 bytes of the result are what we want.
                let packed = _mm256_permute4x64_epi64::<0b00_00_10_00>(packed);
                _mm_storeu_si128(
                    dst.as_mut_ptr().add(dst_row + dx) as *mut _,
                    _mm256_castsi256_si128(packed),
                );

                dx += 16;
            }

            // Scalar tail.
            while dx < dst_w {
                let x0 = x0s[dx] as usize;
                let x1 = (x0 + 1).min(src_w - 1);
                let fx = fxs[dx] as f64 / 65536.0;
                let fy = fy_q16 as f64 / 65536.0;

                let p00 = src[row0 + x0] as f64;
                let p10 = src[row0 + x1] as f64;
                let p01 = src[row1 + x0] as f64;
                let p11 = src[row1 + x1] as f64;

                let val = p00 * (1.0 - fx) * (1.0 - fy)
                    + p10 * fx * (1.0 - fy)
                    + p01 * (1.0 - fx) * fy
                    + p11 * fx * fy;
                dst[dst_row + dx] = val.round() as u8;
                dx += 1;
            }
        }

        dst
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------- BT.601 → BT.709 --------

    fn synth_601_frame(w: usize, h: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let mut y = vec![0u8; w * h];
        let mut cb = vec![0u8; (w / 2) * (h / 2)];
        let mut cr = vec![0u8; (w / 2) * (h / 2)];
        for i in 0..y.len() {
            // Sweep limited range [16, 235].
            y[i] = 16 + ((i as u32 * 17) % 220) as u8;
        }
        for i in 0..cb.len() {
            cb[i] = 16 + ((i as u32 * 13) % 225) as u8;
            cr[i] = 16 + ((i as u32 * 23) % 225) as u8;
        }
        (y, cb, cr)
    }

    #[test]
    fn bt601_to_bt709_neutral_gray_roundtrips() {
        // Cb=Cr=128 means ΔCb=ΔCr=0, so ΔY=0 and ΔCb709=ΔCr709=0.
        // Every luma value stays put; chroma stays at 128.
        for &y_val in &[16u8, 64, 128, 200, 235] {
            let w = 32;
            let h = 16;
            let mut y = vec![y_val; w * h];
            let mut cb = vec![128u8; (w / 2) * (h / 2)];
            let mut cr = vec![128u8; (w / 2) * (h / 2)];
            bt601_to_bt709_planes_scalar(&mut y, &mut cb, &mut cr, w, h);
            for v in &y {
                assert_eq!(*v, y_val, "Y with neutral chroma must round-trip");
            }
            for v in &cb {
                assert_eq!(*v, 128);
            }
            for v in &cr {
                assert_eq!(*v, 128);
            }
        }
    }

    #[test]
    fn bt601_to_bt709_black_and_white_round_trip() {
        // Black (Y=16, Cb=Cr=128) and white (Y=235, Cb=Cr=128) must
        // round-trip unchanged because chroma deltas are zero.
        for &(y_val, label) in &[(16u8, "black"), (235u8, "white")] {
            let w = 64;
            let h = 32;
            let mut y = vec![y_val; w * h];
            let mut cb = vec![128u8; (w / 2) * (h / 2)];
            let mut cr = vec![128u8; (w / 2) * (h / 2)];
            bt601_to_bt709_planes(&mut y, &mut cb, &mut cr, w, h);
            for v in &y {
                assert_eq!(*v, y_val, "{} Y round-trip", label);
            }
            for v in &cb {
                assert_eq!(*v, 128, "{} Cb round-trip", label);
            }
            for v in &cr {
                assert_eq!(*v, 128, "{} Cr round-trip", label);
            }
        }
    }

    #[test]
    fn bt601_to_bt709_scalar_vs_avx2_agree_256x256() {
        // Dense 256×256 synthetic plane; every AVX2 lane path exercised
        // plus the scalar tail (if width is not a 16-chroma multiple
        // we'd hit the tail — here 256 cw=128 is a multiple of 16, so
        // only the main path runs, which is what we want to gate).
        let w = 256;
        let h = 256;
        let (y0, cb0, cr0) = synth_601_frame(w, h);

        let mut y_s = y0.clone();
        let mut cb_s = cb0.clone();
        let mut cr_s = cr0.clone();
        bt601_to_bt709_planes_scalar(&mut y_s, &mut cb_s, &mut cr_s, w, h);

        let mut y_v = y0.clone();
        let mut cb_v = cb0.clone();
        let mut cr_v = cr0.clone();
        bt601_to_bt709_planes(&mut y_v, &mut cb_v, &mut cr_v, w, h);

        let mut max_y = 0i32;
        for i in 0..y_s.len() {
            let d = (y_s[i] as i32 - y_v[i] as i32).abs();
            if d > max_y {
                max_y = d;
            }
            assert!(d <= 1, "Y[{}] scalar={} avx2={}", i, y_s[i], y_v[i]);
        }
        for i in 0..cb_s.len() {
            assert!(
                (cb_s[i] as i32 - cb_v[i] as i32).abs() <= 1,
                "Cb[{}] scalar={} avx2={}",
                i,
                cb_s[i],
                cb_v[i]
            );
            assert!(
                (cr_s[i] as i32 - cr_v[i] as i32).abs() <= 1,
                "Cr[{}] scalar={} avx2={}",
                i,
                cr_s[i],
                cr_v[i]
            );
        }
    }

    #[test]
    fn bt601_to_bt709_scalar_vs_avx2_agree_tail() {
        // 34 wide forces a 1-sample tail in the chroma loop (cw=17,
        // main covers 16, tail covers 1).
        let w = 34;
        let h = 16;
        let (y0, cb0, cr0) = synth_601_frame(w, h);

        let mut y_s = y0.clone();
        let mut cb_s = cb0.clone();
        let mut cr_s = cr0.clone();
        bt601_to_bt709_planes_scalar(&mut y_s, &mut cb_s, &mut cr_s, w, h);

        let mut y_v = y0.clone();
        let mut cb_v = cb0.clone();
        let mut cr_v = cr0.clone();
        bt601_to_bt709_planes(&mut y_v, &mut cb_v, &mut cr_v, w, h);

        for i in 0..y_s.len() {
            assert!(
                (y_s[i] as i32 - y_v[i] as i32).abs() <= 1,
                "Y[{}] scalar={} avx2={}",
                i,
                y_s[i],
                y_v[i]
            );
        }
        for i in 0..cb_s.len() {
            assert!((cb_s[i] as i32 - cb_v[i] as i32).abs() <= 1);
            assert!((cr_s[i] as i32 - cr_v[i] as i32).abs() <= 1);
        }
    }

    #[test]
    fn bt601_to_bt709_clamps_ranges() {
        // After conversion, luma stays in [16, 235] and chroma in [16, 240].
        let w = 32;
        let h = 16;
        let (mut y, mut cb, mut cr) = synth_601_frame(w, h);
        bt601_to_bt709_planes(&mut y, &mut cb, &mut cr, w, h);
        for &v in cb.iter().chain(cr.iter()) {
            assert!((16..=240).contains(&v), "chroma {} out of limited range", v);
        }
        for &v in y.iter() {
            assert!((16..=235).contains(&v), "luma {} out of limited range", v);
        }
    }

    // -------- Bilinear scaler --------

    fn make_ramp(w: usize, h: usize) -> Vec<u8> {
        (0..w * h).map(|i| ((i * 7 + i / w) & 0xff) as u8).collect()
    }

    #[test]
    fn bilinear_scalar_vs_avx2_agree_2x() {
        let src_w = 64;
        let src_h = 32;
        let src = make_ramp(src_w, src_h);
        let dst_w = 128;
        let dst_h = 64;

        let scalar = bilinear_scale_plane_scalar(&src, src_w, src_h, dst_w, dst_h);
        let simd = bilinear_scale_plane(&src, src_w, src_h, dst_w, dst_h);

        assert_eq!(scalar.len(), simd.len());
        let mut max_diff = 0i32;
        for i in 0..scalar.len() {
            let d = (scalar[i] as i32 - simd[i] as i32).abs();
            if d > max_diff {
                max_diff = d;
            }
            assert!(
                d <= 1,
                "bilinear mismatch at {}: scalar={} simd={}",
                i,
                scalar[i],
                simd[i]
            );
        }
    }

    #[test]
    fn bilinear_scalar_vs_avx2_agree_downscale() {
        let src_w = 128;
        let src_h = 72;
        let src = make_ramp(src_w, src_h);
        let dst_w = 64;
        let dst_h = 36;

        let scalar = bilinear_scale_plane_scalar(&src, src_w, src_h, dst_w, dst_h);
        let simd = bilinear_scale_plane(&src, src_w, src_h, dst_w, dst_h);

        for i in 0..scalar.len() {
            let d = (scalar[i] as i32 - simd[i] as i32).abs();
            assert!(
                d <= 1,
                "bilinear mismatch at {}: scalar={} simd={}",
                i,
                scalar[i],
                simd[i]
            );
        }
    }

    #[test]
    fn bilinear_constant_input_yields_constant_output() {
        let src = vec![42u8; 64 * 32];
        let out = bilinear_scale_plane(&src, 64, 32, 128, 64);
        for &v in &out {
            assert_eq!(v, 42, "constant input must yield constant output");
        }
    }

    #[test]
    fn bilinear_identity_scale() {
        let src = make_ramp(32, 32);
        let out = bilinear_scale_plane_scalar(&src, 32, 32, 32, 32);
        assert_eq!(out, src);
    }

    // -------- 10-bit (Squad-19) --------

    fn make_10bit_frame_planar(w: usize, h: usize, y_val: u16, c_val: u16) -> VideoFrame {
        let y_samples = w * h;
        let c_samples = (w / 2) * (h / 2);
        let total = y_samples + 2 * c_samples;
        let mut buf = Vec::with_capacity(total * 2);
        for _ in 0..y_samples {
            buf.extend_from_slice(&y_val.to_le_bytes());
        }
        for _ in 0..(2 * c_samples) {
            buf.extend_from_slice(&c_val.to_le_bytes());
        }
        VideoFrame::new(
            bytes::Bytes::from(buf),
            w as u32,
            h as u32,
            PixelFormat::Yuv420p10le,
            ColorSpace::Bt2020,
            0,
        )
    }

    #[test]
    fn convert_to_yuv420p_bt709_passthrough_10bit() {
        // The HDR-passthrough contract: a 10-bit `Yuv420p10le` frame
        // must come out of `convert_to_yuv420p_bt709` byte-identical
        // (no tonemap, no matrix conversion). The matrix conversion
        // is BT.601→BT.709 on 8-bit; for 10-bit we always passthrough
        // because the source could be HDR / wide-gamut and the matrix
        // shift would corrupt it.
        let frame = make_10bit_frame_planar(16, 16, 600, 512);
        let out = convert_to_yuv420p_bt709(&frame).expect("10-bit passthrough");
        assert_eq!(out.format, PixelFormat::Yuv420p10le);
        assert_eq!(out.width, 16);
        assert_eq!(out.height, 16);
        assert_eq!(out.data.len(), frame.data.len());
        assert_eq!(
            &out.data[..],
            &frame.data[..],
            "10-bit data must be byte-identical (no tonemap)"
        );
        assert_eq!(
            out.color_space,
            ColorSpace::Bt2020,
            "color space must not change"
        );
    }

    #[test]
    fn scale_frame_10bit_constant_input_yields_constant_output() {
        let frame = make_10bit_frame_planar(64, 64, 600, 400);
        let out = scale_frame(&frame, 32, 32).expect("10-bit scale");
        assert_eq!(out.format, PixelFormat::Yuv420p10le);
        assert_eq!(out.width, 32);
        assert_eq!(out.height, 32);

        // Decode the output planes back to u16 and assert constant.
        let y_samples = 32 * 32;
        let c_samples = 16 * 16;
        let y_bytes = y_samples * 2;
        let c_bytes = c_samples * 2;
        assert_eq!(out.data.len(), y_bytes + 2 * c_bytes);

        let y = read_u16le(&out.data[..y_bytes]);
        let u = read_u16le(&out.data[y_bytes..y_bytes + c_bytes]);
        let v = read_u16le(&out.data[y_bytes + c_bytes..y_bytes + 2 * c_bytes]);
        for &s in &y {
            assert_eq!(s, 600, "luma must be constant after bilinear");
        }
        for &s in u.iter().chain(v.iter()) {
            assert_eq!(s, 400, "chroma must be constant after bilinear");
        }
    }

    #[test]
    fn scale_frame_10bit_identity_yields_byte_identical() {
        let frame = make_10bit_frame_planar(32, 32, 768, 256);
        // identity scale (same dims) early-returns clone — verify
        let out = scale_frame(&frame, 32, 32).expect("identity");
        assert_eq!(&out.data[..], &frame.data[..]);
    }

    #[test]
    fn bilinear_10bit_scalar_clamps_inside_10bit_range() {
        // Synthetic ramp in 10-bit range; verify output is bounded.
        let mut src = vec![0u16; 64 * 32];
        for (i, s) in src.iter_mut().enumerate() {
            *s = (i as u16) % 1024;
        }
        let out = bilinear_scale_plane_u16_scalar(&src, 64, 32, 128, 64);
        for &v in &out {
            assert!(v <= 1023, "10-bit sample {} exceeds 1023", v);
        }
    }

    // -------- 10-bit AVX2 (Squad-29) --------

    fn make_10bit_ramp(w: usize, h: usize) -> Vec<u16> {
        // Deterministic 10-bit ramp; cycles through 0..=1023.
        (0..w * h)
            .map(|i| ((i * 7 + i / w) % 1024) as u16)
            .collect()
    }

    #[test]
    fn bilinear_10bit_scalar_vs_avx2_agree_2x_upscale() {
        // 2× upscale exercises every fractional weight in the source.
        let src_w = 64;
        let src_h = 32;
        let src = make_10bit_ramp(src_w, src_h);
        let dst_w = 128;
        let dst_h = 64;

        let scalar = bilinear_scale_plane_u16_scalar(&src, src_w, src_h, dst_w, dst_h);
        let simd = bilinear_scale_plane_u16(&src, src_w, src_h, dst_w, dst_h);

        assert_eq!(scalar.len(), simd.len());
        let mut max_diff = 0i32;
        for i in 0..scalar.len() {
            let d = (scalar[i] as i32 - simd[i] as i32).abs();
            if d > max_diff {
                max_diff = d;
            }
            assert!(
                d <= 1,
                "bilinear 10-bit mismatch at {}: scalar={} simd={}",
                i,
                scalar[i],
                simd[i]
            );
        }
    }

    #[test]
    fn bilinear_10bit_scalar_vs_avx2_agree_downscale_1080p_to_720p() {
        // Headline case: 1920×1080 → 1280×720 luma plane. Same pattern
        // bench uses; gates the AVX2 main path (16-lane while loop runs
        // ~80 iters per row at dst_w=1280).
        let src_w = 1920;
        let src_h = 1080;
        let src = make_10bit_ramp(src_w, src_h);
        let dst_w = 1280;
        let dst_h = 720;

        let scalar = bilinear_scale_plane_u16_scalar(&src, src_w, src_h, dst_w, dst_h);
        let simd = bilinear_scale_plane_u16(&src, src_w, src_h, dst_w, dst_h);

        for i in 0..scalar.len() {
            let d = (scalar[i] as i32 - simd[i] as i32).abs();
            assert!(
                d <= 1,
                "bilinear 10-bit mismatch at {}: scalar={} simd={}",
                i,
                scalar[i],
                simd[i]
            );
        }
    }

    #[test]
    fn bilinear_10bit_avx2_constant_input_yields_constant_output() {
        // Constant 600 (mid-luma in 10-bit limited range) should stay
        // exactly 600 through both axes of bilinear interp.
        let src = vec![600u16; 128 * 64];
        let out = bilinear_scale_plane_u16(&src, 128, 64, 256, 128);
        for &v in &out {
            assert_eq!(v, 600, "constant 10-bit input must yield constant output");
        }
    }

    #[test]
    fn bilinear_10bit_avx2_max_value_clamped() {
        // Max-value (1023) input must stay clamped at 1023 — defensive
        // against the Q15 round trick pushing exactly-1023 to 1024.
        let src = vec![1023u16; 64 * 32];
        let out = bilinear_scale_plane_u16(&src, 64, 32, 128, 64);
        for &v in &out {
            assert!(v <= 1023, "10-bit AVX2 sample {} exceeds 1023", v);
            assert_eq!(v, 1023, "constant 1023 should stay 1023");
        }
    }

    #[test]
    fn bilinear_10bit_narrow_width_falls_back_to_scalar() {
        // dst_w < 16 gates the AVX2 main path; dispatch should fall
        // back to scalar without panicking.
        let src_w = 8;
        let src_h = 8;
        let src = make_10bit_ramp(src_w, src_h);
        let dst_w = 4;
        let dst_h = 4;

        let scalar = bilinear_scale_plane_u16_scalar(&src, src_w, src_h, dst_w, dst_h);
        let dispatched = bilinear_scale_plane_u16(&src, src_w, src_h, dst_w, dst_h);

        assert_eq!(
            scalar, dispatched,
            "narrow strip should match scalar exactly"
        );
    }

    #[test]
    fn bilinear_10bit_odd_dst_dims_handled() {
        // dst_w=17 forces a 1-sample tail (16 main + 1 tail).
        let src_w = 32;
        let src_h = 32;
        let src = make_10bit_ramp(src_w, src_h);
        let dst_w = 17;
        let dst_h = 9;

        let scalar = bilinear_scale_plane_u16_scalar(&src, src_w, src_h, dst_w, dst_h);
        let simd = bilinear_scale_plane_u16(&src, src_w, src_h, dst_w, dst_h);
        assert_eq!(scalar.len(), simd.len());
        for i in 0..scalar.len() {
            let d = (scalar[i] as i32 - simd[i] as i32).abs();
            assert!(
                d <= 1,
                "tail mismatch at {}: scalar={} simd={}",
                i,
                scalar[i],
                simd[i]
            );
        }
    }

    #[test]
    fn bilinear_10bit_tall_narrow_strip() {
        // 16×512 → 16×256 — main loop runs once per row (dst_w=16),
        // many rows.
        let src_w = 16;
        let src_h = 512;
        let src = make_10bit_ramp(src_w, src_h);
        let dst_w = 16;
        let dst_h = 256;

        let scalar = bilinear_scale_plane_u16_scalar(&src, src_w, src_h, dst_w, dst_h);
        let simd = bilinear_scale_plane_u16(&src, src_w, src_h, dst_w, dst_h);
        for i in 0..scalar.len() {
            let d = (scalar[i] as i32 - simd[i] as i32).abs();
            assert!(d <= 1, "tall strip mismatch at {}", i);
        }
    }

    // -------- BT.601 → BT.709 10-bit (Squad-29) --------

    fn synth_601_frame_10bit(w: usize, h: usize) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
        // Sweep limited 10-bit range [64, 940] for luma, [64, 960] for chroma.
        let mut y = vec![0u16; w * h];
        let mut cb = vec![0u16; (w / 2) * (h / 2)];
        let mut cr = vec![0u16; (w / 2) * (h / 2)];
        for i in 0..y.len() {
            y[i] = 64 + ((i as u32 * 17) % 877) as u16;
        }
        for i in 0..cb.len() {
            cb[i] = 64 + ((i as u32 * 13) % 897) as u16;
            cr[i] = 64 + ((i as u32 * 23) % 897) as u16;
        }
        (y, cb, cr)
    }

    #[test]
    fn bt601_to_bt709_10bit_neutral_gray_roundtrips() {
        // Cb=Cr=512 (10-bit chroma center) — every gray luma round-trips.
        // 10-bit limited-range luma analogues of 16/64/128/200/235:
        //   16 << 2 = 64,  64 << 2 = 256,  128 << 2 = 512,
        //   200 << 2 = 800, 235 << 2 = 940.
        for &y_val in &[64u16, 256, 512, 800, 940] {
            let w = 32;
            let h = 16;
            let mut y = vec![y_val; w * h];
            let mut cb = vec![512u16; (w / 2) * (h / 2)];
            let mut cr = vec![512u16; (w / 2) * (h / 2)];
            bt601_to_bt709_planes_10bit_scalar(&mut y, &mut cb, &mut cr, w, h);
            for v in &y {
                assert_eq!(*v, y_val, "Y with neutral chroma must round-trip");
            }
            for v in &cb {
                assert_eq!(*v, 512);
            }
            for v in &cr {
                assert_eq!(*v, 512);
            }
        }
    }

    #[test]
    fn bt601_to_bt709_10bit_scalar_vs_avx2_agree_256x256() {
        // 256×256 → cw=128, multiple of 16 for chroma. Main AVX2 path
        // covers the entire plane.
        let w = 256;
        let h = 256;
        let (y0, cb0, cr0) = synth_601_frame_10bit(w, h);

        let mut y_s = y0.clone();
        let mut cb_s = cb0.clone();
        let mut cr_s = cr0.clone();
        bt601_to_bt709_planes_10bit_scalar(&mut y_s, &mut cb_s, &mut cr_s, w, h);

        let mut y_v = y0.clone();
        let mut cb_v = cb0.clone();
        let mut cr_v = cr0.clone();
        bt601_to_bt709_planes_10bit(&mut y_v, &mut cb_v, &mut cr_v, w, h);

        for i in 0..y_s.len() {
            let d = (y_s[i] as i32 - y_v[i] as i32).abs();
            assert!(d <= 1, "Y[{}] scalar={} avx2={}", i, y_s[i], y_v[i]);
        }
        for i in 0..cb_s.len() {
            assert!(
                (cb_s[i] as i32 - cb_v[i] as i32).abs() <= 1,
                "Cb[{}] scalar={} avx2={}",
                i,
                cb_s[i],
                cb_v[i]
            );
            assert!(
                (cr_s[i] as i32 - cr_v[i] as i32).abs() <= 1,
                "Cr[{}] scalar={} avx2={}",
                i,
                cr_s[i],
                cr_v[i]
            );
        }
    }

    #[test]
    fn bt601_to_bt709_10bit_scalar_vs_avx2_agree_tail() {
        // 34 wide forces a 1-sample chroma tail (cw=17, main covers 16,
        // tail covers 1).
        let w = 34;
        let h = 16;
        let (y0, cb0, cr0) = synth_601_frame_10bit(w, h);

        let mut y_s = y0.clone();
        let mut cb_s = cb0.clone();
        let mut cr_s = cr0.clone();
        bt601_to_bt709_planes_10bit_scalar(&mut y_s, &mut cb_s, &mut cr_s, w, h);

        let mut y_v = y0.clone();
        let mut cb_v = cb0.clone();
        let mut cr_v = cr0.clone();
        bt601_to_bt709_planes_10bit(&mut y_v, &mut cb_v, &mut cr_v, w, h);

        for i in 0..y_s.len() {
            assert!(
                (y_s[i] as i32 - y_v[i] as i32).abs() <= 1,
                "Y[{}] scalar={} avx2={}",
                i,
                y_s[i],
                y_v[i]
            );
        }
        for i in 0..cb_s.len() {
            assert!((cb_s[i] as i32 - cb_v[i] as i32).abs() <= 1);
            assert!((cr_s[i] as i32 - cr_v[i] as i32).abs() <= 1);
        }
    }

    #[test]
    fn bt601_to_bt709_10bit_clamps_ranges() {
        // After conversion, luma stays in [64, 940] and chroma in [64, 960].
        let w = 32;
        let h = 16;
        let (mut y, mut cb, mut cr) = synth_601_frame_10bit(w, h);
        bt601_to_bt709_planes_10bit(&mut y, &mut cb, &mut cr, w, h);
        for &v in cb.iter().chain(cr.iter()) {
            assert!(
                (64..=960).contains(&v),
                "chroma {} out of 10-bit limited range",
                v
            );
        }
        for &v in y.iter() {
            assert!(
                (64..=940).contains(&v),
                "luma {} out of 10-bit limited range",
                v
            );
        }
    }

    #[test]
    fn bt601_to_bt709_10bit_extreme_chroma_clamped_at_high_end() {
        // Chroma at the limited-range max should produce in-range output.
        let w = 32;
        let h = 16;
        let mut y = vec![940u16; w * h];
        let mut cb = vec![960u16; (w / 2) * (h / 2)];
        let mut cr = vec![960u16; (w / 2) * (h / 2)];
        bt601_to_bt709_planes_10bit(&mut y, &mut cb, &mut cr, w, h);
        for &v in y.iter() {
            assert!(v <= 940, "luma {} > 940 (clamp violated)", v);
        }
        for &v in cb.iter().chain(cr.iter()) {
            assert!(v <= 960, "chroma {} > 960 (clamp violated)", v);
        }
    }

    // -------- 4:4:4 → 4:2:0 chroma downsample (Squad-31, roadmap #6) --------

    #[test]
    fn downsample_4x4_box_average_8bit_hand_verified() {
        // 4×4 chroma plane (16 samples) → 2×2 output. Hand-compute the
        // 4 averages so the test is its own oracle.
        //
        //   Cb = [ 10  20 |  30  40
        //          50  60 |  70  80
        //          ---------+--------
        //          90 100 | 110 120
        //         130 140 | 150 160 ]
        //
        // Block (0,0): (10+20+50+60+2)>>2 = 142>>2 = 35
        // Block (1,0): (30+40+70+80+2)>>2 = 222>>2 = 55
        // Block (0,1): (90+100+130+140+2)>>2 = 462>>2 = 115
        // Block (1,1): (110+120+150+160+2)>>2 = 542>>2 = 135
        let cb: Vec<u8> = vec![
            10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160,
        ];
        // Cr distinct so we know the per-plane logic is independent.
        let cr: Vec<u8> = vec![
            5, 15, 25, 35, 45, 55, 65, 75, 85, 95, 105, 115, 125, 135, 145, 155,
        ];
        // Y plane is unchanged — pick a recognizable ramp so we can verify
        // the copy is verbatim.
        let y: Vec<u8> = (0..16).map(|i| i as u8 * 8).collect();

        let out = downsample_chroma_444_to_420(&y, &cb, &cr, 4, 4);
        // Expected layout: 16 Y bytes || 4 Cb bytes || 4 Cr bytes.
        assert_eq!(out.len(), 16 + 4 + 4);
        assert_eq!(&out[..16], y.as_slice(), "Y must round-trip verbatim");
        // Cb output (4 samples in row-major 2×2 order)
        assert_eq!(out[16], 35, "Cb block (0,0)");
        assert_eq!(out[17], 55, "Cb block (1,0)");
        assert_eq!(out[18], 115, "Cb block (0,1)");
        assert_eq!(out[19], 135, "Cb block (1,1)");
        // Cr output
        assert_eq!(out[20], 30, "Cr block (0,0): (5+15+45+55+2)>>2 = 30");
        assert_eq!(out[21], 50, "Cr block (1,0): (25+35+65+75+2)>>2 = 50");
        assert_eq!(out[22], 110, "Cr block (0,1): (85+95+125+135+2)>>2 = 110");
        assert_eq!(out[23], 130, "Cr block (1,1): (105+115+145+155+2)>>2 = 130");
    }

    #[test]
    fn downsample_constant_input_8bit_yields_constant_output() {
        // Cb=128 (chroma midpoint) — average of four 128s with rounding
        // is still 128. Round-trip identity for any constant.
        let w = 16;
        let h = 16;
        let y = vec![64u8; w * h];
        let cb = vec![128u8; w * h];
        let cr = vec![128u8; w * h];
        let out = downsample_chroma_444_to_420(&y, &cb, &cr, w, h);
        let cw = (w + 1) / 2;
        let ch = (h + 1) / 2;
        assert_eq!(out.len(), w * h + 2 * cw * ch);
        // Y unchanged.
        for i in 0..w * h {
            assert_eq!(out[i], 64, "Y[{}] should be 64", i);
        }
        // Cb / Cr: each output sample == 128.
        for i in (w * h)..(w * h + 2 * cw * ch) {
            assert_eq!(out[i], 128, "chroma[{}] should be 128", i - w * h);
        }
    }

    #[test]
    fn downsample_odd_dimensions_clamp_policy() {
        // 7×7 input → 4×4 output. The rightmost column of 2×2 blocks
        // (cx=3) and bottom row (cy=3) straddle exactly one source row /
        // column; clamp policy reuses the in-bounds neighbour.
        //
        //   plane[cx=3, cy=0] takes samples (6, 0), (6, 0), (6, 1), (6, 1)
        //     because x1 = min(7, w-1=6) = 6 — both x0 and x1 = 6.
        //   So the corner sample reduces to a 1-sample average:
        //     (s + s + s' + s' + 2) >> 2 = (s + s')/2 with rounding.
        //
        // Easiest verification: constant-fill input → constant output
        // even at the odd boundary.
        let w = 7;
        let h = 7;
        let y = vec![100u8; w * h];
        let cb = vec![128u8; w * h];
        let cr = vec![64u8; w * h];
        let out = downsample_chroma_444_to_420(&y, &cb, &cr, w, h);
        let cw = (w + 1) / 2; // 4
        let ch = (h + 1) / 2; // 4
        assert_eq!(cw, 4);
        assert_eq!(ch, 4);
        assert_eq!(out.len(), w * h + 2 * cw * ch);
        // Y verbatim.
        for i in 0..w * h {
            assert_eq!(out[i], 100);
        }
        // Cb constant 128.
        for cx in 0..cw {
            for cy in 0..ch {
                let idx = w * h + cy * cw + cx;
                assert_eq!(out[idx], 128, "Cb[{},{}] expected 128", cx, cy);
            }
        }
        // Cr constant 64.
        for cx in 0..cw {
            for cy in 0..ch {
                let idx = w * h + cw * ch + cy * cw + cx;
                assert_eq!(out[idx], 64, "Cr[{},{}] expected 64", cx, cy);
            }
        }
    }

    #[test]
    fn downsample_10bit_constant_input_yields_constant_output() {
        // Cb=512 (10-bit midpoint = 1024/2). Identity for constant input.
        let w = 16;
        let h = 16;
        let y = vec![400u16; w * h];
        let cb = vec![512u16; w * h];
        let cr = vec![512u16; w * h];
        let out = downsample_chroma_444_to_420_10bit(&y, &cb, &cr, w, h);
        let cw = (w + 1) / 2;
        let ch = (h + 1) / 2;
        assert_eq!(out.len(), 2 * (w * h + 2 * cw * ch), "10-bit byte count");

        // Verify each u16 LE sample. Y plane.
        for i in 0..w * h {
            let s = u16::from_le_bytes([out[i * 2], out[i * 2 + 1]]);
            assert_eq!(s, 400, "Y[{}] should be 400", i);
        }
        // Cb plane.
        let cb_byte_off = w * h * 2;
        for i in 0..cw * ch {
            let s = u16::from_le_bytes([out[cb_byte_off + i * 2], out[cb_byte_off + i * 2 + 1]]);
            assert_eq!(s, 512, "Cb[{}] should be 512", i);
        }
        // Cr plane.
        let cr_byte_off = cb_byte_off + cw * ch * 2;
        for i in 0..cw * ch {
            let s = u16::from_le_bytes([out[cr_byte_off + i * 2], out[cr_byte_off + i * 2 + 1]]);
            assert_eq!(s, 512, "Cr[{}] should be 512", i);
        }
    }

    #[test]
    fn downsample_10bit_max_value_no_overflow() {
        // 4 × 1023 + 2 = 4094 fits in u16 (max 65535) and even in i16
        // (max 32767). The u32 accumulator gives plenty of headroom.
        // Verify a full-1023 input doesn't wrap to 0.
        let w = 4;
        let h = 4;
        let y = vec![1023u16; w * h];
        let cb = vec![1023u16; w * h];
        let cr = vec![1023u16; w * h];
        let out = downsample_chroma_444_to_420_10bit(&y, &cb, &cr, w, h);
        let cw = (w + 1) / 2;
        let ch = (h + 1) / 2;

        // Y verbatim (1023).
        for i in 0..w * h {
            let s = u16::from_le_bytes([out[i * 2], out[i * 2 + 1]]);
            assert_eq!(s, 1023, "Y[{}]", i);
        }
        // Cb / Cr: (1023 + 1023 + 1023 + 1023 + 2) >> 2 = 4094 >> 2 = 1023.
        let cb_byte_off = w * h * 2;
        for i in 0..2 * cw * ch {
            let s = u16::from_le_bytes([out[cb_byte_off + i * 2], out[cb_byte_off + i * 2 + 1]]);
            assert_eq!(s, 1023, "chroma[{}] should be 1023 (no overflow)", i);
        }
    }

    #[test]
    fn downsample_10bit_4x4_box_average_hand_verified() {
        // Same 4×4 hand-verified case as 8-bit but in 10-bit.
        let cb_u: Vec<u16> = vec![
            10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160,
        ];
        let cr_u: Vec<u16> = vec![
            500, 600, 700, 800, 500, 600, 700, 800, 500, 600, 700, 800, 500, 600, 700, 800,
        ];
        let y_u: Vec<u16> = (0..16).map(|i| i as u16 * 50).collect();

        let out = downsample_chroma_444_to_420_10bit(&y_u, &cb_u, &cr_u, 4, 4);
        // Y bytes: 16 × 2 = 32. Then 4 Cb (8 bytes) + 4 Cr (8 bytes) = 48.
        assert_eq!(out.len(), 32 + 8 + 8);

        // Y round-trip.
        for i in 0..16 {
            let s = u16::from_le_bytes([out[i * 2], out[i * 2 + 1]]);
            assert_eq!(s, i as u16 * 50, "Y[{}]", i);
        }
        // Cb expected (35, 55, 115, 135) — same as 8-bit case.
        let cb_off = 32;
        let cb0 = u16::from_le_bytes([out[cb_off], out[cb_off + 1]]);
        let cb1 = u16::from_le_bytes([out[cb_off + 2], out[cb_off + 3]]);
        let cb2 = u16::from_le_bytes([out[cb_off + 4], out[cb_off + 5]]);
        let cb3 = u16::from_le_bytes([out[cb_off + 6], out[cb_off + 7]]);
        assert_eq!(cb0, 35);
        assert_eq!(cb1, 55);
        assert_eq!(cb2, 115);
        assert_eq!(cb3, 135);
        // Cr: rows are identical, so each 2×2 block average == row average.
        // (500+600+500+600+2)>>2 = 2202>>2 = 550
        // (700+800+700+800+2)>>2 = 3002>>2 = 750
        let cr_off = cb_off + 8;
        let cr0 = u16::from_le_bytes([out[cr_off], out[cr_off + 1]]);
        let cr1 = u16::from_le_bytes([out[cr_off + 2], out[cr_off + 3]]);
        assert_eq!(cr0, 550);
        assert_eq!(cr1, 750);
    }

    #[test]
    fn downsample_frame_yuv444p10le_to_yuv420p10le() {
        // High-level frame wrapper. Constant 4:4:4 10-bit → constant
        // 4:2:0 10-bit, dims preserved, format flipped.
        let w = 16;
        let h = 16;
        let plane = w * h;
        let mut buf = Vec::with_capacity(3 * plane * 2);
        for _ in 0..plane {
            buf.extend_from_slice(&500u16.to_le_bytes()); // Y
        }
        for _ in 0..plane {
            buf.extend_from_slice(&512u16.to_le_bytes()); // Cb
        }
        for _ in 0..plane {
            buf.extend_from_slice(&512u16.to_le_bytes()); // Cr
        }
        let frame = VideoFrame::new(
            bytes::Bytes::from(buf),
            w as u32,
            h as u32,
            PixelFormat::Yuv444p10le,
            ColorSpace::Bt2020,
            42,
        );
        let out = downsample_444_to_420_frame(&frame).expect("downsample");
        assert_eq!(out.format, PixelFormat::Yuv420p10le);
        assert_eq!(out.width, w as u32);
        assert_eq!(out.height, h as u32);
        assert_eq!(out.pts, 42, "PTS preserved");
        assert_eq!(out.color_space, ColorSpace::Bt2020, "color_space preserved");

        // Spot-check the output samples.
        let cw = w / 2;
        let ch = h / 2;
        let expected_bytes = 2 * (w * h + 2 * cw * ch);
        assert_eq!(out.data.len(), expected_bytes);

        // First Y sample = 500.
        let y0 = u16::from_le_bytes([out.data[0], out.data[1]]);
        assert_eq!(y0, 500);
        // First Cb sample (after Y plane) = 512.
        let cb0 = u16::from_le_bytes([out.data[w * h * 2], out.data[w * h * 2 + 1]]);
        assert_eq!(cb0, 512);
    }

    #[test]
    fn downsample_frame_yuva444p10le_drops_alpha() {
        // 4-plane source, alpha is 16-bit precision. Output is plain
        // Yuv420p10le (no alpha plane).
        let w = 8;
        let h = 8;
        let plane = w * h;
        let mut buf = Vec::with_capacity(4 * plane * 2);
        for _ in 0..plane {
            buf.extend_from_slice(&600u16.to_le_bytes());
        }
        for _ in 0..plane {
            buf.extend_from_slice(&500u16.to_le_bytes());
        }
        for _ in 0..plane {
            buf.extend_from_slice(&500u16.to_le_bytes());
        }
        for _ in 0..plane {
            // Alpha — 16-bit, would have value 65535 if it survived.
            buf.extend_from_slice(&65535u16.to_le_bytes());
        }
        let frame = VideoFrame::new(
            bytes::Bytes::from(buf),
            w as u32,
            h as u32,
            PixelFormat::Yuva444p10le,
            ColorSpace::Bt2020,
            7,
        );
        let out = downsample_444_to_420_frame(&frame).expect("downsample with alpha");
        assert_eq!(out.format, PixelFormat::Yuv420p10le);
        // Output byte count: only Y/Cb/Cr — NO alpha plane.
        let cw = w / 2;
        let ch = h / 2;
        let expected = 2 * (w * h + 2 * cw * ch);
        assert_eq!(out.data.len(), expected);
        // Verify alpha wasn't smuggled in (no 65535 samples).
        for i in (0..out.data.len()).step_by(2) {
            let s = u16::from_le_bytes([out.data[i], out.data[i + 1]]);
            assert!(
                s < 1024 || s == 65535 && false,
                "stray alpha sample {} at {}",
                s,
                i
            );
            assert_ne!(s, 65535, "alpha plane leaked into output");
        }
    }

    #[test]
    fn downsample_frame_rejects_non_444() {
        // 4:2:0 input must error — the frame is already in target format.
        let w = 16;
        let h = 16;
        let plane = w * h;
        let mut buf = Vec::with_capacity(plane + 2 * (plane / 4));
        buf.resize(plane + 2 * (plane / 4), 128);
        let frame = VideoFrame::new(
            bytes::Bytes::from(buf),
            w as u32,
            h as u32,
            PixelFormat::Yuv420p,
            ColorSpace::Bt709,
            0,
        );
        let err = downsample_444_to_420_frame(&frame).unwrap_err();
        assert!(format!("{}", err).contains("expected 4:4:4 input"));
    }
}
