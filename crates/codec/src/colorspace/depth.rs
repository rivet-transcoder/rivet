// =============================================================================
// Bit-depth narrowing (12 → 10, 12 → 8, 10 → 8) and widening (8 → 10).
// =============================================================================
//
// No encoder in the tree takes 12-bit input (NVENC / QSV / AMF / rav1e top
// out at 10; the native h26x encoders at 8), so a 12-bit picture — which the
// native HEVC decoder now produces for Main 12 and the RExt 4:2:2 / 4:4:4
// 12-bit profiles — has to be narrowed before it reaches the encoder. The
// same kernel narrows a 10-bit SDR source to 8 when the output is 8-bit
// (`--bit-depth 8bit`, or the software H.264 / H.265 encoders, which are
// 8-bit only).
//
// Narrowing is a rounded right shift: `(v + 2^(s-1)) >> s`, clamped to the
// target range. Round-to-nearest, no dither — the sources this serves are
// smooth 12-bit masters and the encoder's own quantiser dominates any
// banding a shift could introduce. The same shift is the JCT-VC "conversion
// to lower bit depth" of HM (`TComPicYuv::convertToLowerBitDepth`, HM-16
// videoIO: `(x + (1 << (shift-1))) >> shift`, clipped), so a 12-bit
// conformance clip narrowed here matches HM's 10-bit rendering of it.
//
// Widening 8 → 10 is `v << 2` (HM's `convertToHigherBitDepth`), exact and
// reversible by the narrowing shift.
//
// Layout-agnostic: every planar YUV format is Y then Cb then Cr with the
// same sample width, so the kernel runs over the whole buffer and only the
// format tag changes. `Yuva444p10le` is refused — its alpha plane is 16-bit
// and would be narrowed as if it were 10-bit; the 4:4:4 downsample drops
// alpha first, and the pipeline never narrows before it.
//
// Scalar reference + AVX2 (16 × u16 lanes / iteration) behind the runtime
// dispatch every other colorspace kernel uses; the two agree bit for bit,
// which `tests.rs` checks over the full 12-bit range.

use anyhow::{Result, bail};
use bytes::Bytes;

use crate::frame::{PixelFormat, VideoFrame};

/// Bit depth of a planar YUV `PixelFormat`'s samples, or `None` for formats
/// this module does not narrow (semi-planar, RGB, alpha-bearing).
pub fn planar_bit_depth(format: PixelFormat) -> Option<u8> {
    use PixelFormat::*;
    Some(match format {
        Yuv420p | Yuv422p | Yuv444p => 8,
        Yuv420p10le | Yuv422p10le | Yuv444p10le => 10,
        Yuv420p12le | Yuv422p12le | Yuv444p12le => 12,
        Yuva444p10le | Nv12 | Nv21 | Rgb24 | Rgba32 => return None,
    })
}

/// The same chroma layout as `format` at `bits` per sample, or `None` when
/// the pipeline has no such format.
pub fn with_bit_depth(format: PixelFormat, bits: u8) -> Option<PixelFormat> {
    use PixelFormat::*;
    let layout = match format {
        Yuv420p | Yuv420p10le | Yuv420p12le => 1,
        Yuv422p | Yuv422p10le | Yuv422p12le => 2,
        Yuv444p | Yuv444p10le | Yuv444p12le => 3,
        _ => return None,
    };
    Some(match (layout, bits) {
        (1, 8) => Yuv420p,
        (1, 10) => Yuv420p10le,
        (1, 12) => Yuv420p12le,
        (2, 8) => Yuv422p,
        (2, 10) => Yuv422p10le,
        (2, 12) => Yuv422p12le,
        (3, 8) => Yuv444p,
        (3, 10) => Yuv444p10le,
        (3, 12) => Yuv444p12le,
        _ => return None,
    })
}

/// Number of samples in a planar frame of `format` at `w`×`h`.
fn planar_sample_count(format: PixelFormat, w: usize, h: usize) -> usize {
    use PixelFormat::*;
    let (cw, ch) = match format {
        Yuv420p | Yuv420p10le | Yuv420p12le => (w.div_ceil(2), h.div_ceil(2)),
        Yuv422p | Yuv422p10le | Yuv422p12le => (w.div_ceil(2), h),
        _ => (w, h),
    };
    w * h + 2 * cw * ch
}

// ── scalar reference kernels ─────────────────────────────────────────────────

/// Narrow LE u16 samples by `shift` bits with rounding, into u16 LE, clamped
/// to `max_out`. `src` is the raw byte buffer (pairs of bytes per sample).
pub fn narrow_u16_to_u16_scalar(src: &[u8], shift: u32, max_out: u16, out: &mut Vec<u8>) {
    let half = 1u32 << (shift - 1);
    for pair in src.chunks_exact(2) {
        let v = u16::from_le_bytes([pair[0], pair[1]]) as u32;
        let n = ((v + half) >> shift).min(max_out as u32) as u16;
        out.extend_from_slice(&n.to_le_bytes());
    }
}

/// Narrow LE u16 samples by `shift` bits with rounding, into u8, clamped to
/// 255.
pub fn narrow_u16_to_u8_scalar(src: &[u8], shift: u32, out: &mut Vec<u8>) {
    let half = 1u32 << (shift - 1);
    for pair in src.chunks_exact(2) {
        let v = u16::from_le_bytes([pair[0], pair[1]]) as u32;
        out.push(((v + half) >> shift).min(255) as u8);
    }
}

/// Widen u8 samples by `shift` bits (exact) into u16 LE.
pub fn widen_u8_to_u16_scalar(src: &[u8], shift: u32, out: &mut Vec<u8>) {
    for &v in src {
        out.extend_from_slice(&((v as u16) << shift).to_le_bytes());
    }
}

// ── AVX2 kernels ─────────────────────────────────────────────────────────────

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn narrow_u16_to_u16_avx2(src: &[u8], shift: u32, max_out: u16, out: &mut Vec<u8>) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let n = src.len() / 2;
    let start = out.len();
    out.resize(start + n * 2, 0);
    let dst = &mut out[start..];
    unsafe {
        let v_half = _mm256_set1_epi16((1u16 << (shift - 1)) as i16);
        let v_max = _mm256_set1_epi16(max_out as i16);
        let sh = _mm_cvtsi32_si128(shift as i32);
        let mut i = 0usize;
        while i + 16 <= n {
            let v = _mm256_loadu_si256(src.as_ptr().add(i * 2) as *const _);
            // Saturating add keeps 0xFFFF + half from wrapping (a sample above
            // the nominal range still lands on the clamp, not on zero).
            let r = _mm256_srl_epi16(_mm256_adds_epu16(v, v_half), sh);
            let r = _mm256_min_epu16(r, v_max);
            _mm256_storeu_si256(dst.as_mut_ptr().add(i * 2) as *mut _, r);
            i += 16;
        }
        let half = 1u32 << (shift - 1);
        while i < n {
            let v = u16::from_le_bytes([src[i * 2], src[i * 2 + 1]]) as u32;
            let r = ((v + half) >> shift).min(max_out as u32) as u16;
            dst[i * 2..i * 2 + 2].copy_from_slice(&r.to_le_bytes());
            i += 1;
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn narrow_u16_to_u8_avx2(src: &[u8], shift: u32, out: &mut Vec<u8>) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let n = src.len() / 2;
    let start = out.len();
    out.resize(start + n, 0);
    let dst = &mut out[start..];
    unsafe {
        let v_half = _mm256_set1_epi16((1u16 << (shift - 1)) as i16);
        let sh = _mm_cvtsi32_si128(shift as i32);
        let mut i = 0usize;
        while i + 32 <= n {
            let a = _mm256_loadu_si256(src.as_ptr().add(i * 2) as *const _);
            let b = _mm256_loadu_si256(src.as_ptr().add(i * 2 + 32) as *const _);
            let a = _mm256_srl_epi16(_mm256_adds_epu16(a, v_half), sh);
            let b = _mm256_srl_epi16(_mm256_adds_epu16(b, v_half), sh);
            // packus saturates to 0..=255 (the clamp) but interleaves the two
            // 128-bit halves; the permute restores sample order.
            let p = _mm256_packus_epi16(a, b);
            let p = _mm256_permute4x64_epi64(p, 0b11_01_10_00);
            _mm256_storeu_si256(dst.as_mut_ptr().add(i) as *mut _, p);
            i += 32;
        }
        let half = 1u32 << (shift - 1);
        while i < n {
            let v = u16::from_le_bytes([src[i * 2], src[i * 2 + 1]]) as u32;
            dst[i] = ((v + half) >> shift).min(255) as u8;
            i += 1;
        }
    }
}

// ── runtime dispatch ─────────────────────────────────────────────────────────

/// Narrow LE u16 samples by `shift` bits into u16 LE (AVX2 when available).
pub fn narrow_u16_to_u16(src: &[u8], shift: u32, max_out: u16, out: &mut Vec<u8>) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: avx2 runtime-detected.
            unsafe { narrow_u16_to_u16_avx2(src, shift, max_out, out) };
            return;
        }
    }
    narrow_u16_to_u16_scalar(src, shift, max_out, out)
}

/// Narrow LE u16 samples by `shift` bits into u8 (AVX2 when available).
pub fn narrow_u16_to_u8(src: &[u8], shift: u32, out: &mut Vec<u8>) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: avx2 runtime-detected.
            unsafe { narrow_u16_to_u8_avx2(src, shift, out) };
            return;
        }
    }
    narrow_u16_to_u8_scalar(src, shift, out)
}

// ── frame entry ──────────────────────────────────────────────────────────────

/// Convert a planar YUV frame to `target_bits` (8, 10 or 12) per sample,
/// keeping its chroma layout: 12 → 10, 12 → 8 and 10 → 8 narrow with
/// rounding; 8 → 10 and 8 → 12 widen exactly; 10 → 12 widens exactly. A
/// frame already at `target_bits` is returned as a cheap clone.
///
/// Errors for formats without a planar bit depth (`Nv12`, RGB,
/// `Yuva444p10le`) and for a `target_bits` the layout has no format for.
pub fn convert_bit_depth_frame(frame: &VideoFrame, target_bits: u8) -> Result<VideoFrame> {
    let Some(src_bits) = planar_bit_depth(frame.format) else {
        bail!(
            "bit-depth conversion needs a planar YUV frame, got {:?}",
            frame.format
        );
    };
    if src_bits == target_bits {
        return Ok(frame.clone());
    }
    let Some(target) = with_bit_depth(frame.format, target_bits) else {
        bail!(
            "no pixel format for {:?}'s chroma layout at {} bits",
            frame.format,
            target_bits
        );
    };
    let w = frame.width as usize;
    let h = frame.height as usize;
    let samples = planar_sample_count(frame.format, w, h);
    let src_bytes = samples * if src_bits > 8 { 2 } else { 1 };
    if frame.data.len() < src_bytes {
        bail!(
            "{:?} frame too small for {}x{}: need {} bytes got {}",
            frame.format,
            w,
            h,
            src_bytes,
            frame.data.len()
        );
    }
    let src = &frame.data[..src_bytes];
    let out_bytes = samples * if target_bits > 8 { 2 } else { 1 };
    let mut out = Vec::with_capacity(out_bytes);
    match (src_bits, target_bits) {
        (s, t) if s > 8 && t > 8 && s > t => {
            narrow_u16_to_u16(src, (s - t) as u32, (1u16 << t) - 1, &mut out)
        }
        (s, t) if s > 8 && t == 8 => narrow_u16_to_u8(src, (s - 8) as u32, &mut out),
        (8, t) => widen_u8_to_u16_scalar(src, (t - 8) as u32, &mut out),
        (s, t) => {
            // 10 → 12: exact left shift on u16.
            let sh = (t - s) as u32;
            for pair in src.chunks_exact(2) {
                let v = u16::from_le_bytes([pair[0], pair[1]]) << sh;
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
    }
    Ok(VideoFrame::new(
        Bytes::from(out),
        frame.width,
        frame.height,
        target,
        frame.color_space,
        frame.pts,
    ))
}
