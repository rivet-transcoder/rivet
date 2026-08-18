//! AVX2 versions of the H.265 kernels for 8-bit sample planes (x86-64).
//!
//! Thirty-two 8-bit lanes per vector. Interpolation uses `pmaddubsw` on
//! interleaved neighbour pairs (samples `x, x+1` for horizontal taps, rows
//! `k, k+1` for vertical): each unsigned sample times its signed tap sums in
//! 16 bits, and the HEVC filters cannot overflow that for 8-bit input
//! (|sum| ≤ 255 · 112 = 28560), so the whole first stage runs in 16-bit
//! lanes — twice the width of the 16-bit-sample kernels. Combination and the
//! loop filters narrow to bytes with `packus`, whose saturation is exactly
//! the clip the standard asks for. The second (vertical over 16-bit) stage,
//! the inverse transform and the deblocking arithmetic are sample-size
//! independent and shared with [`super::hevc_avx2`]. Every kernel is checked
//! bit-exact against the scalar reference in the tests below.

#![cfg(target_arch = "x86_64")]

use std::arch::x86_64::*;

use super::hevc::HevcDsp;
use super::hevc_avx2 as w16;
use crate::hevc::tables::{EPEL_FILTERS, QPEL_FILTERS};

/// Replace the scalar entries of `d` with the AVX2 kernels.
pub fn install(d: &mut HevcDsp<u8>) {
    d.idct = [w16::idct_avx2::<4>, w16::idct_avx2::<8>, w16::idct_avx2::<16>, w16::idct_avx2::<32>];
    d.add_residual = add_residual_avx2;
    d.qpel_copy = copy_avx2;
    d.qpel_h = qpel_h_avx2;
    d.qpel_v = qpel_v_avx2;
    d.qpel_v2 = w16::qpel_v2_avx2;
    d.epel_copy = copy_avx2;
    d.epel_h = epel_h_avx2;
    d.epel_v = epel_v_avx2;
    d.epel_v2 = w16::epel_v2_avx2;
    d.uni = uni_avx2;
    d.bi = bi_avx2;
    d.weighted_uni = weighted_uni_avx2;
    d.weighted_bi = weighted_bi_avx2;
    d.sao_band = sao_band_avx2;
    d.sao_edge = sao_edge_avx2;
    d.deblock_luma_v = deblock_luma_v_avx2;
    d.deblock_luma_h = deblock_luma_h_avx2;
    d.deblock_chroma_v = deblock_chroma_v_avx2;
    d.deblock_chroma_h = deblock_chroma_h_avx2;
}

// ----------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------

/// A pair of taps `(a, b)` as one 16-bit lane `a | b << 8` (the low byte
/// multiplies the even sample of an interleaved pair).
#[inline(always)]
fn pair8(a: i8, b: i8) -> i16 {
    (a as u8 as i16) | ((b as i16) << 8)
}

/// Store the first `n` (≤ 8) i16 lanes of `v`.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn store_i16_128(dst: *mut i16, v: __m128i, n: usize) {
    unsafe {
        match n {
            8 => _mm_storeu_si128(dst as *mut __m128i, v),
            4 => _mm_storel_epi64(dst as *mut __m128i, v),
            2 => std::ptr::write_unaligned(dst as *mut u32, _mm_cvtsi128_si32(v) as u32),
            _ => {
                let mut t = [0i16; 8];
                _mm_storeu_si128(t.as_mut_ptr() as *mut __m128i, v);
                std::ptr::copy_nonoverlapping(t.as_ptr(), dst, n);
            }
        }
    }
}

/// Store the first `n` (≤ 16) bytes of `v`.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn store_bytes(dst: *mut u8, v: __m128i, n: usize) {
    unsafe {
        match n {
            16 => _mm_storeu_si128(dst as *mut __m128i, v),
            8 => _mm_storel_epi64(dst as *mut __m128i, v),
            4 => std::ptr::write_unaligned(dst as *mut u32, _mm_cvtsi128_si32(v) as u32),
            2 => std::ptr::write_unaligned(dst as *mut u16, _mm_cvtsi128_si32(v) as u16),
            _ => {
                let mut t = [0u8; 16];
                _mm_storeu_si128(t.as_mut_ptr() as *mut __m128i, v);
                std::ptr::copy_nonoverlapping(t.as_ptr(), dst, n);
            }
        }
    }
}

/// Store the first `n` (≤ 32) bytes of `v`.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn store_bytes32(dst: *mut u8, v: __m256i, n: usize) {
    unsafe {
        if n == 32 {
            _mm256_storeu_si256(dst as *mut __m256i, v);
        } else if n > 16 {
            _mm_storeu_si128(dst as *mut __m128i, _mm256_castsi256_si128(v));
            store_bytes(dst.add(16), _mm256_extracti128_si256(v, 1), n - 16);
        } else {
            store_bytes(dst, _mm256_castsi256_si128(v), n);
        }
    }
}

/// Load 32 bytes, or the first `avail` zero-padded.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn load_bytes32(src: *const u8, avail: usize) -> __m256i {
    unsafe {
        if avail >= 32 {
            _mm256_loadu_si256(src as *const __m256i)
        } else if avail == 16 {
            _mm256_zextsi128_si256(_mm_loadu_si128(src as *const __m128i))
        } else {
            let mut t = [0u8; 32];
            std::ptr::copy_nonoverlapping(src, t.as_mut_ptr(), avail);
            _mm256_loadu_si256(t.as_ptr() as *const __m256i)
        }
    }
}

/// 16 i16 lanes to 16 bytes, saturating to `0..=255`.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn pack16(v: __m256i) -> __m128i {
    unsafe { _mm_packus_epi16(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1)) }
}

/// Whether reading `w` samples starting `x` into a row of `stride`, for
/// `rows` rows, plus `extra` samples along, stays inside `len` for the
/// vector width the kernels use at that block width.
#[inline(always)]
fn fits(len: usize, stride: usize, rows: usize, w: usize, extra: usize) -> bool {
    let (vec, last_x) = if w <= 8 {
        (8, 0)
    } else if w <= 16 {
        (16, 0)
    } else {
        (32, (w - 1) / 32 * 32)
    };
    (rows - 1) * stride + last_x + extra + vec <= len
}

// ----------------------------------------------------------------------
// Interpolation
// ----------------------------------------------------------------------

fn copy_avx2(dst: &mut [i16], src: &[u8], src_stride: usize, w: usize, h: usize, shift: i32) {
    // 16-byte loads at every 16-sample step of each row.
    if (h - 1) * src_stride + (w - 1) / 16 * 16 + 16 > src.len() {
        return (HevcDsp::<u8>::SCALAR.qpel_copy)(dst, src, src_stride, w, h, shift);
    }
    unsafe { copy_impl(dst, src, src_stride, w, h, shift) }
}

#[target_feature(enable = "avx2")]
unsafe fn copy_impl(dst: &mut [i16], src: &[u8], src_stride: usize, w: usize, h: usize, shift: i32) {
    unsafe {
        let sh = _mm_cvtsi32_si128(shift);
        for y in 0..h {
            let s = src.as_ptr().add(y * src_stride);
            let d = dst.as_mut_ptr().add(y * w);
            let mut x = 0;
            while x < w {
                let v = _mm256_cvtepu8_epi16(_mm_loadu_si128(s.add(x) as *const __m128i));
                w16::store_n(d.add(x), _mm256_sll_epi16(v, sh), (w - x).min(16));
                x += 16;
            }
        }
    }
}

/// Horizontal FIR with `TAPS` taps over bytes.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn fir_h<const TAPS: usize>(dst: *mut i16, src: *const u8, src_stride: usize, w: usize, h: usize, taps: &[i8], shift: i32) {
    unsafe {
        let mut c = [_mm256_setzero_si256(); 4];
        for k in 0..TAPS / 2 {
            c[k] = _mm256_set1_epi16(pair8(taps[2 * k], taps[2 * k + 1]));
        }
        let c8: [__m128i; 4] = [_mm256_castsi256_si128(c[0]), _mm256_castsi256_si128(c[1]), _mm256_castsi256_si128(c[2]), _mm256_castsi256_si128(c[3])];
        let sh = _mm_cvtsi32_si128(shift);
        if w <= 8 {
            // Narrow blocks: two rows per vector, one 128-bit lane each.
            let mut y = 0;
            while y + 1 < h {
                let s0 = src.add(y * src_stride);
                let s1 = s0.add(src_stride);
                let mut acc = _mm256_setzero_si256();
                for k in 0..TAPS / 2 {
                    let a = _mm256_setr_m128i(_mm_loadl_epi64(s0.add(2 * k) as *const __m128i), _mm_loadl_epi64(s1.add(2 * k) as *const __m128i));
                    let b = _mm256_setr_m128i(_mm_loadl_epi64(s0.add(2 * k + 1) as *const __m128i), _mm_loadl_epi64(s1.add(2 * k + 1) as *const __m128i));
                    acc = _mm256_add_epi16(acc, _mm256_maddubs_epi16(_mm256_unpacklo_epi8(a, b), c[k]));
                }
                let r = _mm256_sra_epi16(acc, sh);
                store_i16_128(dst.add(y * w), _mm256_castsi256_si128(r), w);
                store_i16_128(dst.add((y + 1) * w), _mm256_extracti128_si256(r, 1), w);
                y += 2;
            }
            if y < h {
                let s0 = src.add(y * src_stride);
                let mut acc = _mm_setzero_si128();
                for k in 0..TAPS / 2 {
                    let a = _mm_loadl_epi64(s0.add(2 * k) as *const __m128i);
                    let b = _mm_loadl_epi64(s0.add(2 * k + 1) as *const __m128i);
                    acc = _mm_add_epi16(acc, _mm_maddubs_epi16(_mm_unpacklo_epi8(a, b), c8[k]));
                }
                store_i16_128(dst.add(y * w), _mm_sra_epi16(acc, sh), w);
            }
            return;
        }
        if w <= 16 {
            for y in 0..h {
                let s = src.add(y * src_stride);
                let mut lo = _mm_setzero_si128();
                let mut hi = _mm_setzero_si128();
                for k in 0..TAPS / 2 {
                    let a = _mm_loadu_si128(s.add(2 * k) as *const __m128i);
                    let b = _mm_loadu_si128(s.add(2 * k + 1) as *const __m128i);
                    lo = _mm_add_epi16(lo, _mm_maddubs_epi16(_mm_unpacklo_epi8(a, b), c8[k]));
                    hi = _mm_add_epi16(hi, _mm_maddubs_epi16(_mm_unpackhi_epi8(a, b), c8[k]));
                }
                let r = _mm256_sra_epi16(_mm256_setr_m128i(lo, hi), sh);
                w16::store_n(dst.add(y * w), r, w);
            }
            return;
        }
        for y in 0..h {
            let s = src.add(y * src_stride);
            let d = dst.add(y * w);
            let mut x = 0;
            while x < w {
                let mut lo = _mm256_setzero_si256();
                let mut hi = _mm256_setzero_si256();
                for k in 0..TAPS / 2 {
                    let a = _mm256_loadu_si256(s.add(x + 2 * k) as *const __m256i);
                    let b = _mm256_loadu_si256(s.add(x + 2 * k + 1) as *const __m256i);
                    lo = _mm256_add_epi16(lo, _mm256_maddubs_epi16(_mm256_unpacklo_epi8(a, b), c[k]));
                    hi = _mm256_add_epi16(hi, _mm256_maddubs_epi16(_mm256_unpackhi_epi8(a, b), c[k]));
                }
                let lo = _mm256_sra_epi16(lo, sh);
                let hi = _mm256_sra_epi16(hi, sh);
                // lo = outputs 0..8 | 16..24, hi = 8..16 | 24..32.
                let n = w - x;
                w16::store_n(d.add(x), _mm256_permute2x128_si256(lo, hi, 0x20), n.min(16));
                if n > 16 {
                    w16::store_n(d.add(x + 16), _mm256_permute2x128_si256(lo, hi, 0x31), (n - 16).min(16));
                }
                x += 32;
            }
        }
    }
}

/// Vertical FIR with `TAPS` taps over byte rows.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn fir_v<const TAPS: usize>(dst: *mut i16, src: *const u8, src_stride: usize, w: usize, h: usize, taps: &[i8], shift: i32) {
    unsafe {
        let mut c = [_mm256_setzero_si256(); 4];
        for k in 0..TAPS / 2 {
            c[k] = _mm256_set1_epi16(pair8(taps[2 * k], taps[2 * k + 1]));
        }
        let c8: [__m128i; 4] = [_mm256_castsi256_si128(c[0]), _mm256_castsi256_si128(c[1]), _mm256_castsi256_si128(c[2]), _mm256_castsi256_si128(c[3])];
        let sh = _mm_cvtsi32_si128(shift);
        let row = |r: usize| src.add(r * src_stride);
        if w <= 8 {
            let mut y = 0;
            while y + 1 < h {
                let mut acc = _mm256_setzero_si256();
                for k in 0..TAPS / 2 {
                    let r0 = _mm_loadl_epi64(row(y + 2 * k) as *const __m128i);
                    let r1 = _mm_loadl_epi64(row(y + 2 * k + 1) as *const __m128i);
                    let r2 = _mm_loadl_epi64(row(y + 2 * k + 2) as *const __m128i);
                    let a = _mm256_setr_m128i(r0, r1);
                    let b = _mm256_setr_m128i(r1, r2);
                    acc = _mm256_add_epi16(acc, _mm256_maddubs_epi16(_mm256_unpacklo_epi8(a, b), c[k]));
                }
                let r = _mm256_sra_epi16(acc, sh);
                store_i16_128(dst.add(y * w), _mm256_castsi256_si128(r), w);
                store_i16_128(dst.add((y + 1) * w), _mm256_extracti128_si256(r, 1), w);
                y += 2;
            }
            if y < h {
                let mut acc = _mm_setzero_si128();
                for k in 0..TAPS / 2 {
                    let a = _mm_loadl_epi64(row(y + 2 * k) as *const __m128i);
                    let b = _mm_loadl_epi64(row(y + 2 * k + 1) as *const __m128i);
                    acc = _mm_add_epi16(acc, _mm_maddubs_epi16(_mm_unpacklo_epi8(a, b), c8[k]));
                }
                store_i16_128(dst.add(y * w), _mm_sra_epi16(acc, sh), w);
            }
            return;
        }
        if w <= 16 {
            for y in 0..h {
                let mut lo = _mm_setzero_si128();
                let mut hi = _mm_setzero_si128();
                for k in 0..TAPS / 2 {
                    let a = _mm_loadu_si128(row(y + 2 * k) as *const __m128i);
                    let b = _mm_loadu_si128(row(y + 2 * k + 1) as *const __m128i);
                    lo = _mm_add_epi16(lo, _mm_maddubs_epi16(_mm_unpacklo_epi8(a, b), c8[k]));
                    hi = _mm_add_epi16(hi, _mm_maddubs_epi16(_mm_unpackhi_epi8(a, b), c8[k]));
                }
                let r = _mm256_sra_epi16(_mm256_setr_m128i(lo, hi), sh);
                w16::store_n(dst.add(y * w), r, w);
            }
            return;
        }
        for y in 0..h {
            let d = dst.add(y * w);
            let mut x = 0;
            while x < w {
                let mut lo = _mm256_setzero_si256();
                let mut hi = _mm256_setzero_si256();
                for k in 0..TAPS / 2 {
                    let a = _mm256_loadu_si256(row(y + 2 * k).add(x) as *const __m256i);
                    let b = _mm256_loadu_si256(row(y + 2 * k + 1).add(x) as *const __m256i);
                    lo = _mm256_add_epi16(lo, _mm256_maddubs_epi16(_mm256_unpacklo_epi8(a, b), c[k]));
                    hi = _mm256_add_epi16(hi, _mm256_maddubs_epi16(_mm256_unpackhi_epi8(a, b), c[k]));
                }
                let lo = _mm256_sra_epi16(lo, sh);
                let hi = _mm256_sra_epi16(hi, sh);
                let n = w - x;
                w16::store_n(d.add(x), _mm256_permute2x128_si256(lo, hi, 0x20), n.min(16));
                if n > 16 {
                    w16::store_n(d.add(x + 16), _mm256_permute2x128_si256(lo, hi, 0x31), (n - 16).min(16));
                }
                x += 32;
            }
        }
    }
}

fn qpel_h_avx2(dst: &mut [i16], src: &[u8], src_stride: usize, w: usize, h: usize, frac: usize, shift: i32) {
    if !fits(src.len(), src_stride, h, w, 7) {
        return (HevcDsp::<u8>::SCALAR.qpel_h)(dst, src, src_stride, w, h, frac, shift);
    }
    unsafe { fir_h::<8>(dst.as_mut_ptr(), src.as_ptr(), src_stride, w, h, &QPEL_FILTERS[frac][..8], shift) }
}

fn qpel_v_avx2(dst: &mut [i16], src: &[u8], src_stride: usize, w: usize, h: usize, frac: usize, shift: i32) {
    if !fits(src.len(), src_stride, h + 7, w, 0) {
        return (HevcDsp::<u8>::SCALAR.qpel_v)(dst, src, src_stride, w, h, frac, shift);
    }
    unsafe { fir_v::<8>(dst.as_mut_ptr(), src.as_ptr(), src_stride, w, h, &QPEL_FILTERS[frac][..8], shift) }
}

fn epel_h_avx2(dst: &mut [i16], src: &[u8], src_stride: usize, w: usize, h: usize, frac: usize, shift: i32) {
    if !fits(src.len(), src_stride, h, w, 3) {
        return (HevcDsp::<u8>::SCALAR.epel_h)(dst, src, src_stride, w, h, frac, shift);
    }
    unsafe { fir_h::<4>(dst.as_mut_ptr(), src.as_ptr(), src_stride, w, h, &EPEL_FILTERS[frac], shift) }
}

fn epel_v_avx2(dst: &mut [i16], src: &[u8], src_stride: usize, w: usize, h: usize, frac: usize, shift: i32) {
    if !fits(src.len(), src_stride, h + 3, w, 0) {
        return (HevcDsp::<u8>::SCALAR.epel_v)(dst, src, src_stride, w, h, frac, shift);
    }
    unsafe { fir_v::<4>(dst.as_mut_ptr(), src.as_ptr(), src_stride, w, h, &EPEL_FILTERS[frac], shift) }
}

// ----------------------------------------------------------------------
// Combination / weighting
// ----------------------------------------------------------------------

fn uni_avx2(dst: &mut [u8], stride: usize, src: &[i16], w: usize, h: usize, shift: i32, max: i32) {
    debug_assert_eq!(max, 255);
    unsafe { uni_impl(dst, stride, src, w, h, shift) }
}

#[target_feature(enable = "avx2")]
unsafe fn uni_impl(dst: &mut [u8], stride: usize, src: &[i16], w: usize, h: usize, shift: i32) {
    unsafe {
        let round = _mm256_set1_epi16(if shift > 0 { 1 << (shift - 1) } else { 0 });
        let sh = _mm_cvtsi32_si128(shift);
        for y in 0..h {
            let mut x = 0;
            while x < w {
                let n = (w - x).min(16);
                let s = w16::load_n(src.as_ptr().add(y * w + x), w - x);
                // 14-bit + round fits i16 (< 16384 + 8192).
                let v = _mm256_sra_epi16(_mm256_adds_epi16(s, round), sh);
                store_bytes(dst.as_mut_ptr().add(y * stride + x), pack16(v), n);
                x += 16;
            }
        }
    }
}

fn bi_avx2(dst: &mut [u8], stride: usize, a: &[i16], b: &[i16], w: usize, h: usize, shift: i32, max: i32) {
    debug_assert_eq!(max, 255);
    unsafe { bi_impl(dst, stride, a, b, w, h, shift) }
}

#[target_feature(enable = "avx2")]
unsafe fn bi_impl(dst: &mut [u8], stride: usize, a: &[i16], b: &[i16], w: usize, h: usize, shift: i32) {
    unsafe {
        let round = _mm256_set1_epi16(1 << (shift - 1));
        let sh = _mm_cvtsi32_si128(shift);
        for y in 0..h {
            let mut x = 0;
            while x < w {
                let n = (w - x).min(16);
                let va = w16::load_n(a.as_ptr().add(y * w + x), w - x);
                let vb = w16::load_n(b.as_ptr().add(y * w + x), w - x);
                // Saturating sums: a + b can exceed i16 only when both are
                // far above the 8-bit range, and then the clip to 255 gives
                // the same answer as the exact 32-bit sum would.
                let v = _mm256_sra_epi16(_mm256_adds_epi16(_mm256_adds_epi16(va, vb), round), sh);
                store_bytes(dst.as_mut_ptr().add(y * stride + x), pack16(v), n);
                x += 16;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn weighted_uni_avx2(dst: &mut [u8], stride: usize, src: &[i16], w: usize, h: usize, log2_wd: i32, wt: i32, o: i32, max: i32) {
    debug_assert_eq!(max, 255);
    unsafe { weighted_uni_impl(dst, stride, src, w, h, log2_wd, wt, o) }
}

#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn weighted_uni_impl(dst: &mut [u8], stride: usize, src: &[i16], w: usize, h: usize, log2_wd: i32, wt: i32, o: i32) {
    unsafe {
        let round = _mm256_set1_epi32(if log2_wd >= 1 { 1 << (log2_wd - 1) } else { 0 });
        let sh = _mm_cvtsi32_si128(log2_wd.max(0));
        let wv = _mm256_set1_epi32(wt);
        let ov = _mm256_set1_epi32(o);
        for y in 0..h {
            let mut x = 0;
            while x < w {
                let n = (w - x).min(16);
                let s = w16::load_n(src.as_ptr().add(y * w + x), w - x);
                let lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(s));
                let hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(s, 1));
                let lo = _mm256_add_epi32(_mm256_sra_epi32(_mm256_add_epi32(_mm256_mullo_epi32(lo, wv), round), sh), ov);
                let hi = _mm256_add_epi32(_mm256_sra_epi32(_mm256_add_epi32(_mm256_mullo_epi32(hi, wv), round), sh), ov);
                let p = _mm256_permute4x64_epi64(_mm256_packs_epi32(lo, hi), 0b11_01_10_00);
                store_bytes(dst.as_mut_ptr().add(y * stride + x), pack16(p), n);
                x += 16;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn weighted_bi_avx2(dst: &mut [u8], stride: usize, a: &[i16], b: &[i16], w: usize, h: usize, log2_wd: i32, w0: i32, w1: i32, o0: i32, o1: i32, max: i32) {
    debug_assert_eq!(max, 255);
    unsafe { weighted_bi_impl(dst, stride, a, b, w, h, log2_wd, w0, w1, o0, o1) }
}

#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn weighted_bi_impl(dst: &mut [u8], stride: usize, a: &[i16], b: &[i16], w: usize, h: usize, log2_wd: i32, w0: i32, w1: i32, o0: i32, o1: i32) {
    unsafe {
        let round = _mm256_set1_epi32((o0 + o1 + 1) << log2_wd);
        let sh = _mm_cvtsi32_si128(log2_wd + 1);
        let w0v = _mm256_set1_epi32(w0);
        let w1v = _mm256_set1_epi32(w1);
        for y in 0..h {
            let mut x = 0;
            while x < w {
                let n = (w - x).min(16);
                let va = w16::load_n(a.as_ptr().add(y * w + x), w - x);
                let vb = w16::load_n(b.as_ptr().add(y * w + x), w - x);
                let alo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(va));
                let ahi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(va, 1));
                let blo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vb));
                let bhi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(vb, 1));
                let lo = _mm256_sra_epi32(_mm256_add_epi32(_mm256_add_epi32(_mm256_mullo_epi32(alo, w0v), _mm256_mullo_epi32(blo, w1v)), round), sh);
                let hi = _mm256_sra_epi32(_mm256_add_epi32(_mm256_add_epi32(_mm256_mullo_epi32(ahi, w0v), _mm256_mullo_epi32(bhi, w1v)), round), sh);
                let p = _mm256_permute4x64_epi64(_mm256_packs_epi32(lo, hi), 0b11_01_10_00);
                store_bytes(dst.as_mut_ptr().add(y * stride + x), pack16(p), n);
                x += 16;
            }
        }
    }
}

// ----------------------------------------------------------------------
// Residual add
// ----------------------------------------------------------------------

fn add_residual_avx2(dst: &mut [u8], stride: usize, res: &[i16], n: usize, max: i32) {
    debug_assert_eq!(max, 255);
    unsafe { add_residual_impl(dst, stride, res, n) }
}

#[target_feature(enable = "avx2")]
unsafe fn add_residual_impl(dst: &mut [u8], stride: usize, res: &[i16], n: usize) {
    unsafe {
        match n {
            n if n >= 32 => {
                for y in 0..n {
                    let mut x = 0;
                    while x < n {
                        let d = dst.as_mut_ptr().add(y * stride + x);
                        let p = _mm256_loadu_si256(d as *const __m256i);
                        let r0 = _mm256_loadu_si256(res.as_ptr().add(y * n + x) as *const __m256i);
                        let r1 = _mm256_loadu_si256(res.as_ptr().add(y * n + x + 16) as *const __m256i);
                        let lo = _mm256_add_epi16(_mm256_cvtepu8_epi16(_mm256_castsi256_si128(p)), r0);
                        let hi = _mm256_add_epi16(_mm256_cvtepu8_epi16(_mm256_extracti128_si256(p, 1)), r1);
                        let v = _mm256_permute4x64_epi64(_mm256_packus_epi16(lo, hi), 0b11_01_10_00);
                        _mm256_storeu_si256(d as *mut __m256i, v);
                        x += 32;
                    }
                }
            }
            16 => {
                for y in 0..16 {
                    let d = dst.as_mut_ptr().add(y * stride);
                    let p = _mm256_cvtepu8_epi16(_mm_loadu_si128(d as *const __m128i));
                    let r = _mm256_loadu_si256(res.as_ptr().add(y * 16) as *const __m256i);
                    _mm_storeu_si128(d as *mut __m128i, pack16(_mm256_add_epi16(p, r)));
                }
            }
            8 => {
                // Two rows per vector.
                for y in (0..8).step_by(2) {
                    let d0 = dst.as_mut_ptr().add(y * stride);
                    let d1 = d0.add(stride);
                    let p = _mm256_cvtepu8_epi16(_mm_unpacklo_epi64(_mm_loadl_epi64(d0 as *const __m128i), _mm_loadl_epi64(d1 as *const __m128i)));
                    let r = _mm256_loadu_si256(res.as_ptr().add(y * 8) as *const __m256i);
                    let v = pack16(_mm256_add_epi16(p, r));
                    _mm_storel_epi64(d0 as *mut __m128i, v);
                    _mm_storel_epi64(d1 as *mut __m128i, _mm_unpackhi_epi64(v, v));
                }
            }
            _ => {
                // 4x4: all four rows in one vector.
                let d = dst.as_mut_ptr();
                let rd = |k: usize| std::ptr::read_unaligned(d.add(k * stride) as *const u32) as i32;
                let p = _mm256_cvtepu8_epi16(_mm_setr_epi32(rd(0), rd(1), rd(2), rd(3)));
                let r = _mm256_loadu_si256(res.as_ptr() as *const __m256i);
                let v = pack16(_mm256_add_epi16(p, r));
                let mut t = [0u32; 4];
                _mm_storeu_si128(t.as_mut_ptr() as *mut __m128i, v);
                for k in 0..4 {
                    std::ptr::write_unaligned(d.add(k * stride) as *mut u32, t[k]);
                }
            }
        }
    }
}

// ----------------------------------------------------------------------
// SAO
// ----------------------------------------------------------------------

/// `v + off` on bytes, clipped to `0..=255`, with `off` in `-128..=127`.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn add_offset_u8(v: __m256i, off: __m256i) -> __m256i {
    unsafe {
        let zero = _mm256_setzero_si256();
        let pos = _mm256_max_epi8(off, zero);
        let neg = _mm256_max_epi8(_mm256_sub_epi8(zero, off), zero);
        _mm256_subs_epu8(_mm256_adds_epu8(v, pos), neg)
    }
}

#[allow(clippy::too_many_arguments)]
fn sao_band_avx2(dst: &mut [u8], dst_stride: usize, src: &[u8], src_stride: usize, w: usize, h: usize, table: &[i16; 32], shift: i32, max: i32) {
    if shift != 3 || table.iter().any(|&o| !(-128..=127).contains(&o)) {
        return (HevcDsp::<u8>::SCALAR.sao_band)(dst, dst_stride, src, src_stride, w, h, table, shift, max);
    }
    debug_assert_eq!(max, 255);
    unsafe { sao_band_impl(dst, dst_stride, src, src_stride, w, h, table, shift) }
}

#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn sao_band_impl(dst: &mut [u8], dst_stride: usize, src: &[u8], src_stride: usize, w: usize, h: usize, table: &[i16; 32], shift: i32) {
    unsafe {
        // The four consecutive bands (mod 32) with nonzero offsets.
        let mut bands = [0u8; 4];
        let mut offs = [0i8; 4];
        let mut k = 0;
        for b in 0..32 {
            if table[b] != 0 && k < 4 {
                bands[k] = (b as u8) << shift;
                offs[k] = table[b] as i8;
                k += 1;
            }
        }
        let mask = _mm256_set1_epi8((0xFFu32 << shift) as u8 as i8);
        let bv: [__m256i; 4] = std::array::from_fn(|i| _mm256_set1_epi8(bands[i] as i8));
        let ov: [__m256i; 4] = std::array::from_fn(|i| _mm256_set1_epi8(offs[i]));
        for y in 0..h {
            let mut x = 0;
            while x < w {
                let n = (w - x).min(32);
                let v = load_bytes32(src.as_ptr().add(y * src_stride + x), n);
                let band = _mm256_and_si256(v, mask);
                let mut off = _mm256_setzero_si256();
                for i in 0..k {
                    off = _mm256_blendv_epi8(off, ov[i], _mm256_cmpeq_epi8(band, bv[i]));
                }
                store_bytes32(dst.as_mut_ptr().add(y * dst_stride + x), add_offset_u8(v, off), n);
                x += 32;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn sao_edge_avx2(dst: &mut [u8], src: &[u8], origin: usize, stride: usize, w: usize, h: usize, na: isize, nb: isize, off: &[i16; 5], max: i32) {
    if off.iter().any(|&o| !(-128..=127).contains(&o)) {
        return (HevcDsp::<u8>::SCALAR.sao_edge)(dst, src, origin, stride, w, h, na, nb, off, max);
    }
    debug_assert_eq!(max, 255);
    unsafe { sao_edge_impl(dst, src, origin, stride, w, h, na, nb, off) }
}

#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn sao_edge_impl(dst: &mut [u8], src: &[u8], origin: usize, stride: usize, w: usize, h: usize, na: isize, nb: isize, off: &[i16; 5]) {
    unsafe {
        // edgeIdx = 2 + sign(v-a) + sign(v-b) in 0..=4 indexes the offsets
        // through a byte shuffle.
        let o = |i: usize| off[i] as i8;
        let tab = _mm256_setr_epi8(
            o(0), o(1), o(2), o(3), o(4), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            o(0), o(1), o(2), o(3), o(4), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        );
        let two = _mm256_set1_epi8(2);
        let lo_reach = na.min(nb).min(0);
        let hi_reach = na.max(nb).max(0);
        for y in 0..h {
            let mut x = 0;
            while x < w {
                let n = (w - x).min(32);
                let i = origin + y * stride + x;
                if (i as isize + lo_reach) < 0 || (i as isize + hi_reach) as usize + 32 > src.len() || i + 32 > dst.len() {
                    // Tail near the buffer end: scalar.
                    for xx in x..w {
                        let ii = origin + y * stride + xx;
                        let v = src[ii] as i32;
                        let a = src[(ii as isize + na) as usize] as i32;
                        let b = src[(ii as isize + nb) as usize] as i32;
                        let e = (2 + (v - a).signum() + (v - b).signum()) as usize;
                        dst[ii] = (v + off[e] as i32).clamp(0, 255) as u8;
                    }
                    break;
                }
                let v = _mm256_loadu_si256(src.as_ptr().add(i) as *const __m256i);
                let a = _mm256_loadu_si256(src.as_ptr().offset(i as isize + na) as *const __m256i);
                let b = _mm256_loadu_si256(src.as_ptr().offset(i as isize + nb) as *const __m256i);
                // Unsigned compares: ge = (max(v, a) == v), gt = ge & !eq, lt = !ge.
                let ge_a = _mm256_cmpeq_epi8(_mm256_max_epu8(v, a), v);
                let gt_a = _mm256_andnot_si256(_mm256_cmpeq_epi8(v, a), ge_a);
                let ge_b = _mm256_cmpeq_epi8(_mm256_max_epu8(v, b), v);
                let gt_b = _mm256_andnot_si256(_mm256_cmpeq_epi8(v, b), ge_b);
                // e = 2 + gt_a - lt_a + gt_b - lt_b with masks of -1: 2 - gt + lt.
                let ones = _mm256_cmpeq_epi8(v, v);
                let lt_a = _mm256_xor_si256(ge_a, ones);
                let lt_b = _mm256_xor_si256(ge_b, ones);
                let e = _mm256_add_epi8(_mm256_sub_epi8(_mm256_sub_epi8(two, gt_a), gt_b), _mm256_add_epi8(lt_a, lt_b));
                let o = _mm256_shuffle_epi8(tab, e);
                store_bytes32(dst.as_mut_ptr().add(i), add_offset_u8(v, o), n);
                x += 32;
            }
        }
    }
}

// ----------------------------------------------------------------------
// Deblocking — the shared i32-lane filters with byte loads and stores.
// ----------------------------------------------------------------------

/// Eight consecutive bytes as 8 x i32.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn ld8_u8(p: *const u8) -> __m256i {
    unsafe { _mm256_cvtepu8_epi32(_mm_loadl_epi64(p as *const __m128i)) }
}

/// 8 x i32 (each within a byte) to eight bytes in the low half.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn pack8_u8(v: __m256i) -> __m128i {
    unsafe {
        let p = w16::pack8_u16(v);
        _mm_packus_epi16(p, p)
    }
}

#[allow(clippy::too_many_arguments)]
fn deblock_luma_v_avx2(data: &mut [u8], off: usize, stride: usize, beta: [i32; 2], tc: [i32; 2], no_p: [bool; 2], no_q: [bool; 2], max: i32) {
    if (beta[0] == 0 && tc[0] == 0) && (beta[1] == 0 && tc[1] == 0) {
        return;
    }
    assert!(off >= 4 && off + 7 * stride + 4 <= data.len());
    unsafe { deblock_luma_v_impl(data.as_mut_ptr().add(off), stride, beta, tc, no_p, no_q, max) }
}

#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn deblock_luma_v_impl(data: *mut u8, stride: usize, beta: [i32; 2], tc: [i32; 2], no_p: [bool; 2], no_q: [bool; 2], max: i32) {
    unsafe {
        let mut r = [_mm_setzero_si128(); 8];
        for i in 0..8 {
            r[i] = _mm_cvtepu8_epi16(_mm_loadl_epi64(data.add(i * stride).sub(4) as *const __m128i));
        }
        w16::transpose8_u16(&mut r);
        let mut v: w16::Lines8 = [_mm256_setzero_si256(); 8];
        for k in 0..8 {
            v[k] = _mm256_cvtepu16_epi32(r[k]);
        }
        w16::luma_filter8(&mut v, beta, tc, no_p, no_q, max);
        for k in 0..8 {
            r[k] = w16::pack8_u16(v[k]);
        }
        w16::transpose8_u16(&mut r);
        for i in 0..8 {
            _mm_storel_epi64(data.add(i * stride).sub(4) as *mut __m128i, _mm_packus_epi16(r[i], r[i]));
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn deblock_luma_h_avx2(data: &mut [u8], off: usize, stride: usize, beta: [i32; 2], tc: [i32; 2], no_p: [bool; 2], no_q: [bool; 2], max: i32) {
    if (beta[0] == 0 && tc[0] == 0) && (beta[1] == 0 && tc[1] == 0) {
        return;
    }
    assert!(off >= 4 * stride && off + 3 * stride + 8 <= data.len());
    unsafe { deblock_luma_h_impl(data.as_mut_ptr().add(off), stride, beta, tc, no_p, no_q, max) }
}

#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn deblock_luma_h_impl(data: *mut u8, stride: usize, beta: [i32; 2], tc: [i32; 2], no_p: [bool; 2], no_q: [bool; 2], max: i32) {
    unsafe {
        let mut v: w16::Lines8 = [_mm256_setzero_si256(); 8];
        for k in 0..8 {
            v[k] = ld8_u8(data.offset((k as isize - 4) * stride as isize));
        }
        w16::luma_filter8(&mut v, beta, tc, no_p, no_q, max);
        for k in 1..7 {
            _mm_storel_epi64(data.offset((k as isize - 4) * stride as isize) as *mut __m128i, pack8_u8(v[k]));
        }
    }
}

fn deblock_chroma_v_avx2(data: &mut [u8], off: usize, stride: usize, tc: [i32; 4], no_p: [bool; 4], no_q: [bool; 4], max: i32) {
    if tc.iter().all(|&t| t == 0) {
        return;
    }
    assert!(off >= 2 && off + 7 * stride + 2 <= data.len());
    unsafe { deblock_chroma_v_impl(data.as_mut_ptr().add(off), stride, tc, no_p, no_q, max) }
}

#[target_feature(enable = "avx2")]
unsafe fn deblock_chroma_v_impl(data: *mut u8, stride: usize, tc: [i32; 4], no_p: [bool; 4], no_q: [bool; 4], max: i32) {
    unsafe {
        let mut r = [_mm_setzero_si128(); 8];
        for i in 0..8 {
            let q = std::ptr::read_unaligned(data.add(i * stride).sub(2) as *const u32);
            r[i] = _mm_cvtepu8_epi16(_mm_cvtsi32_si128(q as i32));
        }
        let a0 = _mm_unpacklo_epi16(r[0], r[1]);
        let a1 = _mm_unpacklo_epi16(r[2], r[3]);
        let a2 = _mm_unpacklo_epi16(r[4], r[5]);
        let a3 = _mm_unpacklo_epi16(r[6], r[7]);
        let b0 = _mm_unpacklo_epi32(a0, a1); // p1 r0..3 | p0 r0..3
        let b1 = _mm_unpackhi_epi32(a0, a1); // q0 r0..3 | q1 r0..3
        let b2 = _mm_unpacklo_epi32(a2, a3);
        let b3 = _mm_unpackhi_epi32(a2, a3);
        let mut v = [
            _mm256_cvtepu16_epi32(_mm_unpacklo_epi64(b0, b2)),
            _mm256_cvtepu16_epi32(_mm_unpackhi_epi64(b0, b2)),
            _mm256_cvtepu16_epi32(_mm_unpacklo_epi64(b1, b3)),
            _mm256_cvtepu16_epi32(_mm_unpackhi_epi64(b1, b3)),
        ];
        w16::chroma_filter8(&mut v, tc, no_p, no_q, max);
        // (p0, q0) byte pairs per row.
        let p0 = w16::pack8_u16(v[1]);
        let q0 = w16::pack8_u16(v[2]);
        let pairs = _mm_packus_epi16(_mm_unpacklo_epi16(p0, q0), _mm_unpackhi_epi16(p0, q0));
        let mut t = [0u16; 8];
        _mm_storeu_si128(t.as_mut_ptr() as *mut __m128i, pairs);
        for i in 0..8 {
            std::ptr::write_unaligned(data.add(i * stride).sub(1) as *mut u16, t[i]);
        }
    }
}

fn deblock_chroma_h_avx2(data: &mut [u8], off: usize, stride: usize, tc: [i32; 4], no_p: [bool; 4], no_q: [bool; 4], max: i32) {
    if tc.iter().all(|&t| t == 0) {
        return;
    }
    assert!(off >= 2 * stride && off + stride + 8 <= data.len());
    unsafe { deblock_chroma_h_impl(data.as_mut_ptr().add(off), stride, tc, no_p, no_q, max) }
}

#[target_feature(enable = "avx2")]
unsafe fn deblock_chroma_h_impl(data: *mut u8, stride: usize, tc: [i32; 4], no_p: [bool; 4], no_q: [bool; 4], max: i32) {
    unsafe {
        let mut v = [ld8_u8(data.sub(2 * stride)), ld8_u8(data.sub(stride)), ld8_u8(data), ld8_u8(data.add(stride))];
        w16::chroma_filter8(&mut v, tc, no_p, no_q, max);
        _mm_storel_epi64(data.sub(stride) as *mut __m128i, pack8_u8(v[1]));
        _mm_storel_epi64(data as *mut __m128i, pack8_u8(v[2]));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::hevc::HevcDsp;

    fn lcg(seed: &mut u64) -> u32 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (*seed >> 33) as u32
    }

    fn avx2() -> Option<HevcDsp<u8>> {
        if !std::is_x86_feature_detected!("avx2") {
            return None;
        }
        let mut d = HevcDsp::<u8>::SCALAR;
        install(&mut d);
        Some(d)
    }

    #[test]
    fn interp_matches_scalar_u8() {
        let Some(d) = avx2() else { return };
        let s = HevcDsp::<u8>::SCALAR;
        let mut seed = 1u64;
        let stride = 96;
        for trial in 0..3 {
            let src: Vec<u8> = (0..stride * 96)
                .map(|_| match trial {
                    0 => lcg(&mut seed) as u8,
                    1 => [0u8, 255][(lcg(&mut seed) % 2) as usize],
                    _ => (lcg(&mut seed) % 4) as u8 * 85,
                })
                .collect();
            for &(w, h) in &[(2usize, 4usize), (2, 8), (4, 4), (4, 8), (4, 3), (6, 8), (8, 4), (8, 8), (8, 5), (12, 16), (16, 16), (24, 32), (32, 8), (48, 64), (64, 64)] {
                for frac in 1..8 {
                    let mut a = vec![0i16; w * h];
                    let mut b = vec![0i16; w * h];
                    if frac < 4 {
                        (s.qpel_h)(&mut a, &src, stride, w, h, frac, 0);
                        (d.qpel_h)(&mut b, &src, stride, w, h, frac, 0);
                        assert_eq!(a, b, "qpel_h {w}x{h} frac={frac} trial={trial}");
                        (s.qpel_v)(&mut a, &src, stride, w, h, frac, 0);
                        (d.qpel_v)(&mut b, &src, stride, w, h, frac, 0);
                        assert_eq!(a, b, "qpel_v {w}x{h} frac={frac} trial={trial}");
                        let mid: Vec<i16> = (0..stride * 96).map(|_| (lcg(&mut seed) % 30000) as i16 - 15000).collect();
                        (s.qpel_v2)(&mut a, &mid, stride, w, h, frac);
                        (d.qpel_v2)(&mut b, &mid, stride, w, h, frac);
                        assert_eq!(a, b, "qpel_v2 {w}x{h} frac={frac}");
                    }
                    (s.epel_h)(&mut a, &src, stride, w, h, frac, 0);
                    (d.epel_h)(&mut b, &src, stride, w, h, frac, 0);
                    assert_eq!(a, b, "epel_h {w}x{h} frac={frac} trial={trial}");
                    (s.epel_v)(&mut a, &src, stride, w, h, frac, 0);
                    (d.epel_v)(&mut b, &src, stride, w, h, frac, 0);
                    assert_eq!(a, b, "epel_v {w}x{h} frac={frac} trial={trial}");
                    let mid: Vec<i16> = (0..stride * 96).map(|_| (lcg(&mut seed) % 30000) as i16 - 15000).collect();
                    (s.epel_v2)(&mut a, &mid, stride, w, h, frac);
                    (d.epel_v2)(&mut b, &mid, stride, w, h, frac);
                    assert_eq!(a, b, "epel_v2 {w}x{h} frac={frac}");
                }
                let mut a = vec![0i16; w * h];
                let mut b = vec![0i16; w * h];
                (s.qpel_copy)(&mut a, &src, stride, w, h, 6);
                (d.qpel_copy)(&mut b, &src, stride, w, h, 6);
                assert_eq!(a, b, "copy {w}x{h}");
            }
        }
    }

    #[test]
    fn combine_matches_scalar_u8() {
        let Some(d) = avx2() else { return };
        let s = HevcDsp::<u8>::SCALAR;
        let mut seed = 3u64;
        let max = 255;
        for &(w, h) in &[(2usize, 4usize), (4, 4), (6, 8), (8, 8), (12, 16), (16, 8), (24, 4), (32, 32), (64, 64)] {
            for range in [16000i32, 22500] {
                let a: Vec<i16> = (0..w * h).map(|_| (lcg(&mut seed) % (2 * range as u32)) as i16 - range as i16).collect();
                let b: Vec<i16> = (0..w * h).map(|_| (lcg(&mut seed) % (2 * range as u32)) as i16 - range as i16).collect();
                let stride = w + 5;
                let mut d1 = vec![0u8; stride * h];
                let mut d2 = vec![0u8; stride * h];
                (s.uni)(&mut d1, stride, &a, w, h, 6, max);
                (d.uni)(&mut d2, stride, &a, w, h, 6, max);
                assert_eq!(d1, d2, "uni {w}x{h}");
                (s.bi)(&mut d1, stride, &a, &b, w, h, 7, max);
                (d.bi)(&mut d2, stride, &a, &b, w, h, 7, max);
                assert_eq!(d1, d2, "bi {w}x{h} range={range}");
                for &(log2_wd, wt, o) in &[(6 + 6, 128, 0), (6, 1, 5), (7 + 6, -20, -3), (3 + 6, 255, 127)] {
                    (s.weighted_uni)(&mut d1, stride, &a, w, h, log2_wd, wt, o, max);
                    (d.weighted_uni)(&mut d2, stride, &a, w, h, log2_wd, wt, o, max);
                    assert_eq!(d1, d2, "wuni {w}x{h} {log2_wd} {wt} {o}");
                    (s.weighted_bi)(&mut d1, stride, &a, &b, w, h, log2_wd, wt, 3 - wt, o, -o, max);
                    (d.weighted_bi)(&mut d2, stride, &a, &b, w, h, log2_wd, wt, 3 - wt, o, -o, max);
                    assert_eq!(d1, d2, "wbi {w}x{h}");
                }
            }
            let res: Vec<i16> = (0..w * w).map(|_| (lcg(&mut seed) % 700) as i16 - 350).collect();
            if w == h && w >= 4 && w.is_power_of_two() {
                let stride = w + 5;
                let base: Vec<u8> = (0..stride * h).map(|_| lcg(&mut seed) as u8).collect();
                let mut d1 = base.clone();
                let mut d2 = base.clone();
                (s.add_residual)(&mut d1, stride, &res, w, max);
                (d.add_residual)(&mut d2, stride, &res, w, max);
                assert_eq!(d1, d2, "add_residual {w}");
            }
        }
    }

    #[test]
    fn sao_matches_scalar_u8() {
        let Some(d) = avx2() else { return };
        let s = HevcDsp::<u8>::SCALAR;
        let mut seed = 11u64;
        let stride = 80;
        let max = 255;
        for trial in 0..3 {
            let src: Vec<u8> = (0..stride * 80)
                .map(|_| match trial {
                    0 => lcg(&mut seed) as u8,
                    1 => (lcg(&mut seed) % 3) as u8 + 100,
                    _ => [0u8, 255, 254, 1][(lcg(&mut seed) % 4) as usize],
                })
                .collect();
            for &(w, h) in &[(3usize, 5usize), (8, 8), (16, 16), (31, 17), (33, 9), (64, 64), (72, 3)] {
                let mut table = [0i16; 32];
                let pos = (lcg(&mut seed) % 32) as usize;
                for k in 0..4 {
                    table[(pos + k) & 31] = (lcg(&mut seed) % 15) as i16 - 7;
                }
                let mut d1 = src.clone();
                let mut d2 = src.clone();
                let off = 8 * stride + 8;
                (s.sao_band)(&mut d1[off..], stride, &src[off..], stride, w, h, &table, 3, max);
                (d.sao_band)(&mut d2[off..], stride, &src[off..], stride, w, h, &table, 3, max);
                assert_eq!(d1, d2, "band {w}x{h} trial={trial}");
                let offs: [i16; 5] = [(lcg(&mut seed) % 8) as i16, (lcg(&mut seed) % 8) as i16, 0, -((lcg(&mut seed) % 8) as i16), -((lcg(&mut seed) % 8) as i16)];
                for &(na, nb) in &[(-1isize, 1isize), (-(stride as isize), stride as isize), (-(stride as isize) - 1, stride as isize + 1), (-(stride as isize) + 1, stride as isize - 1)] {
                    let mut d1 = src.clone();
                    let mut d2 = src.clone();
                    (s.sao_edge)(&mut d1, &src, off, stride, w, h, na, nb, &offs, max);
                    (d.sao_edge)(&mut d2, &src, off, stride, w, h, na, nb, &offs, max);
                    assert_eq!(d1, d2, "edge {w}x{h} {na} {nb} trial={trial}");
                }
            }
        }
    }

    #[test]
    fn deblocking_matches_scalar_u8() {
        let Some(d) = avx2() else { return };
        let s = HevcDsp::<u8>::SCALAR;
        let mut seed = 23u64;
        let stride = 40;
        let max = 255;
        for trial in 0..600 {
            let base = lcg(&mut seed) % 256;
            let spread = 1 + lcg(&mut seed) % 16;
            let plane: Vec<u8> = (0..stride * 32).map(|_| (base + lcg(&mut seed) % spread).min(255) as u8).collect();
            let rnd = |seed: &mut u64, n: u32| lcg(seed) % n;
            let v = |seed: &mut u64, n: u32| rnd(seed, n) as i32;
            let beta = [rnd(&mut seed, 3).min(1) as i32 * v(&mut seed, 64), rnd(&mut seed, 3).min(1) as i32 * v(&mut seed, 64)];
            let tc = [rnd(&mut seed, 3).min(1) as i32 * v(&mut seed, 25), rnd(&mut seed, 3).min(1) as i32 * v(&mut seed, 25)];
            let np = [rnd(&mut seed, 5) == 0, rnd(&mut seed, 5) == 0];
            let nq = [rnd(&mut seed, 5) == 0, rnd(&mut seed, 5) == 0];
            let tc4 = [v(&mut seed, 25) * (rnd(&mut seed, 2) as i32), v(&mut seed, 25), 0, v(&mut seed, 25)];
            let np4 = [rnd(&mut seed, 5) == 0, rnd(&mut seed, 5) == 0, false, rnd(&mut seed, 5) == 0];
            let nq4 = [rnd(&mut seed, 5) == 0, false, rnd(&mut seed, 5) == 0, rnd(&mut seed, 5) == 0];
            let off = 8 * stride + 8;
            let mut a = plane.clone();
            let mut b = plane.clone();
            match trial % 4 {
                0 => {
                    (s.deblock_luma_v)(&mut a, off, stride, beta, tc, np, nq, max);
                    (d.deblock_luma_v)(&mut b, off, stride, beta, tc, np, nq, max);
                }
                1 => {
                    (s.deblock_luma_h)(&mut a, off, stride, beta, tc, np, nq, max);
                    (d.deblock_luma_h)(&mut b, off, stride, beta, tc, np, nq, max);
                }
                2 => {
                    (s.deblock_chroma_v)(&mut a, off, stride, tc4, np4, nq4, max);
                    (d.deblock_chroma_v)(&mut b, off, stride, tc4, np4, nq4, max);
                }
                _ => {
                    (s.deblock_chroma_h)(&mut a, off, stride, tc4, np4, nq4, max);
                    (d.deblock_chroma_h)(&mut b, off, stride, tc4, np4, nq4, max);
                }
            }
            assert_eq!(a, b, "hevc u8 deblock kind {} trial {trial} beta {beta:?} tc {tc:?}", trial % 4);
        }
    }
}
