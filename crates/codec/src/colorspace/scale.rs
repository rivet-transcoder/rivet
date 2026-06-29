// Bilinear scaler — scalar + AVX2 dispatch for 8-bit (Yuv420p) and
// 10-bit (Yuv420p10le) frames.

use anyhow::{Result, bail};
use bytes::BytesMut;

use crate::frame::{PixelFormat, VideoFrame};

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
    let y_plane = super::read_u16le(&frame.data[..src_y_size_bytes]);
    let u_plane =
        super::read_u16le(&frame.data[src_y_size_bytes..src_y_size_bytes + src_c_size_bytes]);
    let v_plane = super::read_u16le(
        &frame.data
            [src_y_size_bytes + src_c_size_bytes..src_y_size_bytes + 2 * src_c_size_bytes],
    );

    // Squad-29: runtime-dispatched (AVX2 when available, scalar fallback).
    let y_dst = bilinear_scale_plane_u16(&y_plane, src_w, src_h, dst_w, dst_h);
    let u_dst =
        bilinear_scale_plane_u16(&u_plane, src_w / 2, src_h / 2, dst_w / 2, dst_h / 2);
    let v_dst =
        bilinear_scale_plane_u16(&v_plane, src_w / 2, src_h / 2, dst_w / 2, dst_h / 2);

    let mut out = BytesMut::with_capacity(dst_total_bytes);
    super::write_u16le(&mut out, &y_dst);
    super::write_u16le(&mut out, &u_dst);
    super::write_u16le(&mut out, &v_dst);

    Ok(VideoFrame::new(
        out.freeze(),
        target_width,
        target_height,
        frame.format,
        frame.color_space,
        frame.pts,
    ))
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
            return unsafe {
                bilinear_scale_plane_u16_avx2(src, src_w, src_h, dst_w, dst_h)
            };
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
                let clamped =
                    _mm256_min_epi16(_mm256_max_epi16(out_i16, v_zero), v_max);

                _mm256_storeu_si256(
                    dst.as_mut_ptr().add(dst_row + dx) as *mut _,
                    clamped,
                );

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
                let p00 = _mm256_cvtepu8_epi16(
                    _mm_loadu_si128(p00_buf.as_ptr() as *const _),
                );
                let p10 = _mm256_cvtepu8_epi16(
                    _mm_loadu_si128(p10_buf.as_ptr() as *const _),
                );
                let p01 = _mm256_cvtepu8_epi16(
                    _mm_loadu_si128(p01_buf.as_ptr() as *const _),
                );
                let p11 = _mm256_cvtepu8_epi16(
                    _mm_loadu_si128(p11_buf.as_ptr() as *const _),
                );

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
