// Chroma-layout and RGB→YUV conversion helpers.
//
// Provides the per-format conversion paths called by the top-level
// `convert_to_yuv420p_bt709` dispatcher in the parent module.

use anyhow::{Result, bail};
use bytes::BytesMut;

use crate::frame::{ColorSpace, PixelFormat, VideoFrame};

pub(super) fn nv12_to_yuv420p(frame: &VideoFrame) -> Result<VideoFrame> {
    deinterleave_semiplanar_to_yuv420p(frame, /*v_first=*/ false)
}

/// NV21 has the same packed layout as NV12 but the chroma plane carries
/// `VU` interleaved instead of `UV`. Sharing the implementation reduces
/// the chance of one path drifting from the other on bug fixes.
pub(super) fn nv21_to_yuv420p(frame: &VideoFrame) -> Result<VideoFrame> {
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
pub(super) fn yuv422p_to_yuv420p(frame: &VideoFrame) -> Result<VideoFrame> {
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
pub(super) fn yuv422p10le_to_yuv420p10le(frame: &VideoFrame) -> Result<VideoFrame> {
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
    let words = super::read_u16le(&frame.data[..need_bytes]);
    let (y_in, rest) = words.split_at(y_samples);
    let (cb_in, cr_in) = rest.split_at(chroma_in_samples);

    let mut out = BytesMut::with_capacity((y_samples + 2 * chroma_out_samples) * 2);
    super::write_u16le(&mut out, y_in);

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
pub(super) fn rgb_to_yuv420p_bt709(frame: &VideoFrame, has_alpha: bool) -> Result<VideoFrame> {
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
