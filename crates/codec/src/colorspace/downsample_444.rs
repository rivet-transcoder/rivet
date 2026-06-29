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

use anyhow::{Result, bail};
use bytes::Bytes;

use crate::frame::{PixelFormat, VideoFrame};

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
            let y = super::read_u16le(&frame.data[..plane * 2]);
            let cb = super::read_u16le(&frame.data[plane * 2..2 * plane * 2]);
            let cr = super::read_u16le(&frame.data[2 * plane * 2..3 * plane * 2]);

            if frame.format == PixelFormat::Yuva444p10le {
                tracing::warn!(
                    pts = frame.pts,
                    "dropping alpha plane on 4:4:4→4:2:0 downsample \
                     (rav1e 0.7 has no alpha; pipeline target is Yuv420p10le)"
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
