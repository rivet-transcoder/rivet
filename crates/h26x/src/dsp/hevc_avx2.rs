//! AVX2 versions of the H.265 kernels (x86-64).
//!
//! Sixteen 16-bit lanes per vector. Interpolation uses `pmaddwd` on
//! interleaved neighbour pairs (samples `x, x+1` for horizontal taps, rows
//! `k, k+1` for vertical), which yields 32-bit sums for lanes 0..3 / 8..11
//! (unpacklo) and 4..7 / 12..15 (unpackhi) that `packs_epi32` puts back in
//! order. The inverse DCT is the matrix product vectorised across columns
//! (first stage) and along rows (second stage) with the same pair trick,
//! restricted to the nonzero region the parser reported. Every kernel is
//! checked bit-exact against the scalar reference in the tests below.

#![cfg(target_arch = "x86_64")]

use std::arch::x86_64::*;

use super::hevc::HevcDsp;
use crate::hevc::tables::{EPEL_FILTERS, QPEL_FILTERS, TRANSFORM32};

/// Replace the scalar entries of `d` with the AVX2 kernels.
pub fn install(d: &mut HevcDsp) {
    d.idct = [idct_avx2::<4>, idct_avx2::<8>, idct_avx2::<16>, idct_avx2::<32>];
    d.add_residual = add_residual_avx2;
    d.qpel_copy = copy_avx2;
    d.qpel_h = qpel_h_avx2;
    d.qpel_v = qpel_v_avx2;
    d.qpel_v2 = qpel_v2_avx2;
    d.epel_copy = copy_avx2;
    d.epel_h = epel_h_avx2;
    d.epel_v = epel_v_avx2;
    d.epel_v2 = epel_v2_avx2;
    d.uni = uni_avx2;
    d.bi = bi_avx2;
    d.weighted_uni = weighted_uni_avx2;
    d.weighted_bi = weighted_bi_avx2;
    d.sao_band = sao_band_avx2;
    d.sao_edge = sao_edge_avx2;
}

// ----------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------

/// A pair of taps `(a, b)` broadcast as 32-bit lanes `a | b << 16`.
#[inline(always)]
fn pair(a: i8, b: i8) -> i32 {
    (a as i16 as u16 as i32) | ((b as i16 as u16 as i32) << 16)
}

/// Store the first `n` (≤ 16) lanes of `v` to `dst`.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn store_n(dst: *mut i16, v: __m256i, n: usize) {
    unsafe {
        if n == 16 {
            _mm256_storeu_si256(dst as *mut __m256i, v);
        } else {
            let mut t = [0i16; 16];
            _mm256_storeu_si256(t.as_mut_ptr() as *mut __m256i, v);
            std::ptr::copy_nonoverlapping(t.as_ptr(), dst, n);
        }
    }
}

/// Store the first `n` (≤ 16) lanes of `v` (u16 samples) to `dst`.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn store_n_u16(dst: *mut u16, v: __m256i, n: usize) {
    unsafe {
        if n == 16 {
            _mm256_storeu_si256(dst as *mut __m256i, v);
        } else {
            let mut t = [0u16; 16];
            _mm256_storeu_si256(t.as_mut_ptr() as *mut __m256i, v);
            std::ptr::copy_nonoverlapping(t.as_ptr(), dst, n);
        }
    }
}

/// Load 16 lanes from `src`, or the first `n` zero-padded when the slice
/// ends sooner (`avail` = lanes that may be read).
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn load_n(src: *const i16, avail: usize) -> __m256i {
    unsafe {
        if avail >= 16 {
            _mm256_loadu_si256(src as *const __m256i)
        } else {
            let mut t = [0i16; 16];
            std::ptr::copy_nonoverlapping(src, t.as_mut_ptr(), avail);
            _mm256_loadu_si256(t.as_ptr() as *const __m256i)
        }
    }
}

/// Whether reading `w` lanes starting `x` into a row of `stride`, for `rows`
/// rows, plus `extra` samples along, stays inside `len` when the load reaches
/// 16 lanes.
#[inline(always)]
fn fits(len: usize, stride: usize, rows: usize, w: usize, extra: usize) -> bool {
    // Last row start + last vector start (rounded up to 16) + extra + 16.
    let last_x = if w == 0 { 0 } else { (w - 1) / 16 * 16 };
    (rows - 1) * stride + last_x + extra + 16 <= len
}

// ----------------------------------------------------------------------
// Interpolation
// ----------------------------------------------------------------------

fn copy_avx2(dst: &mut [i16], src: &[u16], src_stride: usize, w: usize, h: usize, shift: i32) {
    if !fits(src.len(), src_stride, h, w, 0) {
        return (HevcDsp::SCALAR.qpel_copy)(dst, src, src_stride, w, h, shift);
    }
    unsafe { copy_impl(dst, src, src_stride, w, h, shift) }
}

#[target_feature(enable = "avx2")]
unsafe fn copy_impl(dst: &mut [i16], src: &[u16], src_stride: usize, w: usize, h: usize, shift: i32) {
    unsafe {
        let sh = _mm_cvtsi32_si128(shift);
        for y in 0..h {
            let s = src.as_ptr().add(y * src_stride);
            let d = dst.as_mut_ptr().add(y * w);
            let mut x = 0;
            while x < w {
                let v = _mm256_loadu_si256(s.add(x) as *const __m256i);
                store_n(d.add(x), _mm256_sll_epi16(v, sh), (w - x).min(16));
                x += 16;
            }
        }
    }
}

/// Horizontal FIR with `TAPS` taps over u16 samples.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn fir_h<const TAPS: usize>(dst: *mut i16, src: *const u16, src_stride: usize, w: usize, h: usize, taps: &[i8], shift: i32) {
    unsafe {
        let mut c = [_mm256_setzero_si256(); 4];
        for k in 0..TAPS / 2 {
            c[k] = _mm256_set1_epi32(pair(taps[2 * k], taps[2 * k + 1]));
        }
        let sh = _mm_cvtsi32_si128(shift);
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
                    lo = _mm256_add_epi32(lo, _mm256_madd_epi16(_mm256_unpacklo_epi16(a, b), c[k]));
                    hi = _mm256_add_epi32(hi, _mm256_madd_epi16(_mm256_unpackhi_epi16(a, b), c[k]));
                }
                let r = _mm256_packs_epi32(_mm256_sra_epi32(lo, sh), _mm256_sra_epi32(hi, sh));
                store_n(d.add(x), r, (w - x).min(16));
                x += 16;
            }
        }
    }
}

/// Vertical FIR with `TAPS` taps over u16 or i16 rows (`T` = 2-byte lanes).
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn fir_v<const TAPS: usize, T>(dst: *mut i16, src: *const T, src_stride: usize, w: usize, h: usize, taps: &[i8], shift: i32) {
    unsafe {
        let mut c = [_mm256_setzero_si256(); 4];
        for k in 0..TAPS / 2 {
            c[k] = _mm256_set1_epi32(pair(taps[2 * k], taps[2 * k + 1]));
        }
        let sh = _mm_cvtsi32_si128(shift);
        for y in 0..h {
            let d = dst.add(y * w);
            let mut x = 0;
            while x < w {
                let mut lo = _mm256_setzero_si256();
                let mut hi = _mm256_setzero_si256();
                for k in 0..TAPS / 2 {
                    let a = _mm256_loadu_si256(src.add((y + 2 * k) * src_stride + x) as *const __m256i);
                    let b = _mm256_loadu_si256(src.add((y + 2 * k + 1) * src_stride + x) as *const __m256i);
                    lo = _mm256_add_epi32(lo, _mm256_madd_epi16(_mm256_unpacklo_epi16(a, b), c[k]));
                    hi = _mm256_add_epi32(hi, _mm256_madd_epi16(_mm256_unpackhi_epi16(a, b), c[k]));
                }
                let r = _mm256_packs_epi32(_mm256_sra_epi32(lo, sh), _mm256_sra_epi32(hi, sh));
                store_n(d.add(x), r, (w - x).min(16));
                x += 16;
            }
        }
    }
}

fn qpel_h_avx2(dst: &mut [i16], src: &[u16], src_stride: usize, w: usize, h: usize, frac: usize, shift: i32) {
    if !fits(src.len(), src_stride, h, w, 8) {
        return (HevcDsp::SCALAR.qpel_h)(dst, src, src_stride, w, h, frac, shift);
    }
    unsafe { fir_h::<8>(dst.as_mut_ptr(), src.as_ptr(), src_stride, w, h, &QPEL_FILTERS[frac][..8], shift) }
}

fn qpel_v_avx2(dst: &mut [i16], src: &[u16], src_stride: usize, w: usize, h: usize, frac: usize, shift: i32) {
    if !fits(src.len(), src_stride, h + 7, w, 0) {
        return (HevcDsp::SCALAR.qpel_v)(dst, src, src_stride, w, h, frac, shift);
    }
    unsafe { fir_v::<8, u16>(dst.as_mut_ptr(), src.as_ptr(), src_stride, w, h, &QPEL_FILTERS[frac][..8], shift) }
}

fn qpel_v2_avx2(dst: &mut [i16], src: &[i16], src_stride: usize, w: usize, h: usize, frac: usize) {
    if !fits(src.len(), src_stride, h + 7, w, 0) {
        return (HevcDsp::SCALAR.qpel_v2)(dst, src, src_stride, w, h, frac);
    }
    unsafe { fir_v::<8, i16>(dst.as_mut_ptr(), src.as_ptr(), src_stride, w, h, &QPEL_FILTERS[frac][..8], 6) }
}

fn epel_h_avx2(dst: &mut [i16], src: &[u16], src_stride: usize, w: usize, h: usize, frac: usize, shift: i32) {
    if !fits(src.len(), src_stride, h, w, 4) {
        return (HevcDsp::SCALAR.epel_h)(dst, src, src_stride, w, h, frac, shift);
    }
    unsafe { fir_h::<4>(dst.as_mut_ptr(), src.as_ptr(), src_stride, w, h, &EPEL_FILTERS[frac], shift) }
}

fn epel_v_avx2(dst: &mut [i16], src: &[u16], src_stride: usize, w: usize, h: usize, frac: usize, shift: i32) {
    if !fits(src.len(), src_stride, h + 3, w, 0) {
        return (HevcDsp::SCALAR.epel_v)(dst, src, src_stride, w, h, frac, shift);
    }
    unsafe { fir_v::<4, u16>(dst.as_mut_ptr(), src.as_ptr(), src_stride, w, h, &EPEL_FILTERS[frac], shift) }
}

fn epel_v2_avx2(dst: &mut [i16], src: &[i16], src_stride: usize, w: usize, h: usize, frac: usize) {
    if !fits(src.len(), src_stride, h + 3, w, 0) {
        return (HevcDsp::SCALAR.epel_v2)(dst, src, src_stride, w, h, frac);
    }
    unsafe { fir_v::<4, i16>(dst.as_mut_ptr(), src.as_ptr(), src_stride, w, h, &EPEL_FILTERS[frac], 6) }
}

// ----------------------------------------------------------------------
// Combination / weighting
// ----------------------------------------------------------------------

/// Clip 16 lanes of i16 to `0..=max` (max < 32768) as u16.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn clip_u16(v: __m256i, maxv: __m256i) -> __m256i {
    unsafe { _mm256_min_epi16(_mm256_max_epi16(v, _mm256_setzero_si256()), maxv) }
}

fn uni_avx2(dst: &mut [u16], stride: usize, src: &[i16], w: usize, h: usize, shift: i32, max: i32) {
    unsafe { uni_impl(dst, stride, src, w, h, shift, max) }
}

#[target_feature(enable = "avx2")]
unsafe fn uni_impl(dst: &mut [u16], stride: usize, src: &[i16], w: usize, h: usize, shift: i32, max: i32) {
    unsafe {
        let round = _mm256_set1_epi16(if shift > 0 { 1 << (shift - 1) } else { 0 });
        let sh = _mm_cvtsi32_si128(shift);
        let maxv = _mm256_set1_epi16(max as i16);
        for y in 0..h {
            let mut x = 0;
            while x < w {
                let n = (w - x).min(16);
                let s = load_n(src.as_ptr().add(y * w + x), w - x);
                // 14-bit + round fits i16 (< 16384 + 8192).
                let v = _mm256_sra_epi16(_mm256_adds_epi16(s, round), sh);
                store_n_u16(dst.as_mut_ptr().add(y * stride + x), clip_u16(v, maxv), n);
                x += 16;
            }
        }
    }
}

fn bi_avx2(dst: &mut [u16], stride: usize, a: &[i16], b: &[i16], w: usize, h: usize, shift: i32, max: i32) {
    unsafe { bi_impl(dst, stride, a, b, w, h, shift, max) }
}

#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn bi_impl(dst: &mut [u16], stride: usize, a: &[i16], b: &[i16], w: usize, h: usize, shift: i32, max: i32) {
    unsafe {
        let round = _mm256_set1_epi32(1 << (shift - 1));
        let sh = _mm_cvtsi32_si128(shift);
        let maxv = _mm256_set1_epi16(max as i16);
        for y in 0..h {
            let mut x = 0;
            while x < w {
                let n = (w - x).min(16);
                let va = load_n(a.as_ptr().add(y * w + x), w - x);
                let vb = load_n(b.as_ptr().add(y * w + x), w - x);
                // Sum in 32 bits (a + b can exceed i16), shift, pack.
                let lo = _mm256_add_epi32(_mm256_add_epi32(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(va)), _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vb))), round);
                let hi = _mm256_add_epi32(_mm256_add_epi32(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(va, 1)), _mm256_cvtepi16_epi32(_mm256_extracti128_si256(vb, 1))), round);
                let lo = _mm256_sra_epi32(lo, sh);
                let hi = _mm256_sra_epi32(hi, sh);
                // packs per 128-bit lane: [lo0..3, hi0..3, lo4..7, hi4..7] -> fix with permute.
                let p = _mm256_packs_epi32(lo, hi);
                let p = _mm256_permute4x64_epi64(p, 0b11_01_10_00);
                store_n_u16(dst.as_mut_ptr().add(y * stride + x), clip_u16(p, maxv), n);
                x += 16;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn weighted_uni_avx2(dst: &mut [u16], stride: usize, src: &[i16], w: usize, h: usize, log2_wd: i32, wt: i32, o: i32, max: i32) {
    unsafe { weighted_uni_impl(dst, stride, src, w, h, log2_wd, wt, o, max) }
}

#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn weighted_uni_impl(dst: &mut [u16], stride: usize, src: &[i16], w: usize, h: usize, log2_wd: i32, wt: i32, o: i32, max: i32) {
    unsafe {
        let round = _mm256_set1_epi32(if log2_wd >= 1 { 1 << (log2_wd - 1) } else { 0 });
        let sh = _mm_cvtsi32_si128(log2_wd.max(0));
        let wv = _mm256_set1_epi32(wt);
        let ov = _mm256_set1_epi32(o);
        let maxv = _mm256_set1_epi16(max as i16);
        for y in 0..h {
            let mut x = 0;
            while x < w {
                let n = (w - x).min(16);
                let s = load_n(src.as_ptr().add(y * w + x), w - x);
                let lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(s));
                let hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(s, 1));
                let lo = _mm256_add_epi32(_mm256_sra_epi32(_mm256_add_epi32(_mm256_mullo_epi32(lo, wv), round), sh), ov);
                let hi = _mm256_add_epi32(_mm256_sra_epi32(_mm256_add_epi32(_mm256_mullo_epi32(hi, wv), round), sh), ov);
                let p = _mm256_permute4x64_epi64(_mm256_packs_epi32(lo, hi), 0b11_01_10_00);
                store_n_u16(dst.as_mut_ptr().add(y * stride + x), clip_u16(p, maxv), n);
                x += 16;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn weighted_bi_avx2(dst: &mut [u16], stride: usize, a: &[i16], b: &[i16], w: usize, h: usize, log2_wd: i32, w0: i32, w1: i32, o0: i32, o1: i32, max: i32) {
    unsafe { weighted_bi_impl(dst, stride, a, b, w, h, log2_wd, w0, w1, o0, o1, max) }
}

#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn weighted_bi_impl(dst: &mut [u16], stride: usize, a: &[i16], b: &[i16], w: usize, h: usize, log2_wd: i32, w0: i32, w1: i32, o0: i32, o1: i32, max: i32) {
    unsafe {
        let round = _mm256_set1_epi32((o0 + o1 + 1) << log2_wd);
        let sh = _mm_cvtsi32_si128(log2_wd + 1);
        let w0v = _mm256_set1_epi32(w0);
        let w1v = _mm256_set1_epi32(w1);
        let maxv = _mm256_set1_epi16(max as i16);
        for y in 0..h {
            let mut x = 0;
            while x < w {
                let n = (w - x).min(16);
                let va = load_n(a.as_ptr().add(y * w + x), w - x);
                let vb = load_n(b.as_ptr().add(y * w + x), w - x);
                let alo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(va));
                let ahi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(va, 1));
                let blo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vb));
                let bhi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(vb, 1));
                let lo = _mm256_sra_epi32(_mm256_add_epi32(_mm256_add_epi32(_mm256_mullo_epi32(alo, w0v), _mm256_mullo_epi32(blo, w1v)), round), sh);
                let hi = _mm256_sra_epi32(_mm256_add_epi32(_mm256_add_epi32(_mm256_mullo_epi32(ahi, w0v), _mm256_mullo_epi32(bhi, w1v)), round), sh);
                let p = _mm256_permute4x64_epi64(_mm256_packs_epi32(lo, hi), 0b11_01_10_00);
                store_n_u16(dst.as_mut_ptr().add(y * stride + x), clip_u16(p, maxv), n);
                x += 16;
            }
        }
    }
}

// ----------------------------------------------------------------------
// Residual add
// ----------------------------------------------------------------------

fn add_residual_avx2(dst: &mut [u16], stride: usize, res: &[i16], n: usize, max: i32) {
    unsafe { add_residual_impl(dst, stride, res, n, max) }
}

#[target_feature(enable = "avx2")]
unsafe fn add_residual_impl(dst: &mut [u16], stride: usize, res: &[i16], n: usize, max: i32) {
    unsafe {
        let maxv = _mm256_set1_epi16(max as i16);
        if n >= 16 {
            for y in 0..n {
                let mut x = 0;
                while x < n {
                    let d = dst.as_mut_ptr().add(y * stride + x);
                    let p = _mm256_loadu_si256(d as *const __m256i);
                    let r = _mm256_loadu_si256(res.as_ptr().add(y * n + x) as *const __m256i);
                    // Samples < 4096 and residuals fit: adds saturate correctly.
                    let v = clip_u16(_mm256_adds_epi16(p, r), maxv);
                    _mm256_storeu_si256(d as *mut __m256i, v);
                    x += 16;
                }
            }
        } else if n == 8 {
            let maxv = _mm256_castsi256_si128(maxv);
            for y in 0..n {
                let d = dst.as_mut_ptr().add(y * stride);
                let p = _mm_loadu_si128(d as *const __m128i);
                let r = _mm_loadu_si128(res.as_ptr().add(y * 8) as *const __m128i);
                let v = _mm_min_epi16(_mm_max_epi16(_mm_adds_epi16(p, r), _mm_setzero_si128()), maxv);
                _mm_storeu_si128(d as *mut __m128i, v);
            }
        } else {
            // 4x4: two rows per 128-bit vector.
            let maxv = _mm256_castsi256_si128(maxv);
            for y in (0..4).step_by(2) {
                let d0 = dst.as_mut_ptr().add(y * stride);
                let d1 = dst.as_mut_ptr().add((y + 1) * stride);
                let p = _mm_unpacklo_epi64(_mm_loadl_epi64(d0 as *const __m128i), _mm_loadl_epi64(d1 as *const __m128i));
                let r = _mm_loadu_si128(res.as_ptr().add(y * 4) as *const __m128i);
                let v = _mm_min_epi16(_mm_max_epi16(_mm_adds_epi16(p, r), _mm_setzero_si128()), maxv);
                _mm_storel_epi64(d0 as *mut __m128i, v);
                _mm_storel_epi64(d1 as *mut __m128i, _mm_unpackhi_epi64(v, v));
            }
        }
    }
}

// ----------------------------------------------------------------------
// Inverse DCT
// ----------------------------------------------------------------------

/// The transform matrix rows for size `n` as interleaved pairs of rows
/// `(j, j+1)`: `[c[j][0], c[j+1][0], c[j][1], c[j+1][1], ...]` (n lanes × 2).
struct PairRows {
    rows32: [[i16; 64]; 16],
    rows16: [[i16; 32]; 8],
    rows8: [[i16; 16]; 4],
    rows4: [[i16; 8]; 2],
}

const fn build_pairs() -> PairRows {
    let mut p = PairRows { rows32: [[0; 64]; 16], rows16: [[0; 32]; 8], rows8: [[0; 16]; 4], rows4: [[0; 8]; 2] };
    let mut j = 0;
    while j < 16 {
        let mut k = 0;
        while k < 32 {
            p.rows32[j][2 * k] = TRANSFORM32[2 * j][k] as i16;
            p.rows32[j][2 * k + 1] = TRANSFORM32[2 * j + 1][k] as i16;
            k += 1;
        }
        j += 1;
    }
    let mut j = 0;
    while j < 8 {
        let mut k = 0;
        while k < 16 {
            p.rows16[j][2 * k] = TRANSFORM32[4 * j][k] as i16;
            p.rows16[j][2 * k + 1] = TRANSFORM32[4 * j + 2][k] as i16;
            k += 1;
        }
        j += 1;
    }
    let mut j = 0;
    while j < 4 {
        let mut k = 0;
        while k < 8 {
            p.rows8[j][2 * k] = TRANSFORM32[8 * j][k] as i16;
            p.rows8[j][2 * k + 1] = TRANSFORM32[8 * j + 4][k] as i16;
            k += 1;
        }
        j += 1;
    }
    let mut j = 0;
    while j < 2 {
        let mut k = 0;
        while k < 4 {
            p.rows4[j][2 * k] = TRANSFORM32[16 * j][k] as i16;
            p.rows4[j][2 * k + 1] = TRANSFORM32[16 * j + 8][k] as i16;
            k += 1;
        }
        j += 1;
    }
    p
}

static PAIRS: PairRows = build_pairs();

#[inline(always)]
fn pair_row(n: usize, j: usize) -> &'static [i16] {
    match n {
        32 => &PAIRS.rows32[j],
        16 => &PAIRS.rows16[j],
        8 => &PAIRS.rows8[j],
        _ => &PAIRS.rows4[j],
    }
}

fn idct_avx2<const N: usize>(coeffs: &mut [i16], bd_shift: i32, max_x: usize, max_y: usize) {
    if max_x == 0 && max_y == 0 {
        // DC only.
        let round2 = 1i32 << (bd_shift - 1);
        let v = ((coeffs[0] as i32 * 64 + 64) >> 7).clamp(-32768, 32767);
        let r = ((v * 64 + round2) >> bd_shift).clamp(-32768, 32767) as i16;
        coeffs[..N * N].fill(r);
        return;
    }
    if N == 4 {
        // Not worth a vector: the scalar butterfly is 4 lines.
        return (HevcDsp::SCALAR.idct[0])(coeffs, bd_shift, max_x, max_y);
    }
    unsafe { idct_impl::<N>(coeffs, bd_shift, max_x, max_y) }
}

#[target_feature(enable = "avx2")]
unsafe fn idct_impl<const N: usize>(coeffs: &mut [i16], bd_shift: i32, max_x: usize, max_y: usize) {
    unsafe {
        let mut tmp = [0i16; 32 * 32];
        // Stage 1 (columns): tmp[y][x] = clip((sum_j c[j][y] * coef[j][x] + 64) >> 7),
        // vectorised across x for each y; pairs of input rows (j, j+1).
        let nzy = max_y + 1;
        let npairs = nzy.div_ceil(2);
        let round1 = _mm256_set1_epi32(64);
        let step = 32 / N;
        for y in 0..N {
            let mut x = 0;
            while x <= max_x {
                let mut lo = round1;
                let mut hi = round1;
                for p in 0..npairs {
                    let j = 2 * p;
                    let a = load_n(coeffs.as_ptr().add(j * N + x), N - x);
                    let b = if j + 1 < nzy { load_n(coeffs.as_ptr().add((j + 1) * N + x), N - x) } else { _mm256_setzero_si256() };
                    let c = _mm256_set1_epi32(pair(TRANSFORM32[j * step][y], TRANSFORM32[(j + 1) * step][y]));
                    lo = _mm256_add_epi32(lo, _mm256_madd_epi16(_mm256_unpacklo_epi16(a, b), c));
                    hi = _mm256_add_epi32(hi, _mm256_madd_epi16(_mm256_unpackhi_epi16(a, b), c));
                }
                let r = _mm256_packs_epi32(_mm256_srai_epi32(lo, 7), _mm256_srai_epi32(hi, 7));
                store_n(tmp.as_mut_ptr().add(y * N + x), r, (N - x).min(16));
                x += 16;
            }
        }
        // Stage 2 (rows): out[y][x] = clip((sum_j c[j][x] * tmp[y][j] + round) >> shift),
        // vectorised across x with the interleaved pair rows of the matrix.
        let nzx = max_x + 1;
        let npairs = nzx.div_ceil(2);
        let round2 = _mm256_set1_epi32(1 << (bd_shift - 1));
        let sh = _mm_cvtsi32_si128(bd_shift);
        for y in 0..N {
            let row = tmp.as_ptr().add(y * N);
            let mut x = 0;
            while x < N {
                let mut lo = round2;
                let mut hi = round2;
                for p in 0..npairs {
                    let j = 2 * p;
                    let t0 = *row.add(j) as i32;
                    let t1 = if j + 1 < nzx { *row.add(j + 1) as i32 } else { 0 };
                    let tv = _mm256_set1_epi32((t0 as u16 as i32) | ((t1 as u16 as i32) << 16));
                    let pr = pair_row(N, p);
                    let cl = _mm256_loadu_si256(pr.as_ptr().add(2 * x) as *const __m256i); // pairs for x..x+8
                    let ch = if N - x > 8 { _mm256_loadu_si256(pr.as_ptr().add(2 * x + 16) as *const __m256i) } else { _mm256_setzero_si256() };
                    lo = _mm256_add_epi32(lo, _mm256_madd_epi16(cl, tv));
                    hi = _mm256_add_epi32(hi, _mm256_madd_epi16(ch, tv));
                }
                // lo = outputs x..x+8 (in order across the two 128-bit lanes: 0..3 | 4..7),
                // hi = x+8..x+16. packs per lane would interleave; permute to fix.
                let r = _mm256_packs_epi32(_mm256_sra_epi32(lo, sh), _mm256_sra_epi32(hi, sh));
                let r = _mm256_permute4x64_epi64(r, 0b11_01_10_00);
                store_n(coeffs.as_mut_ptr().add(y * N + x), r, (N - x).min(16));
                x += 16;
            }
        }
    }
}

// ----------------------------------------------------------------------
// SAO
// ----------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn sao_band_avx2(dst: &mut [u16], dst_stride: usize, src: &[u16], src_stride: usize, w: usize, h: usize, table: &[i16; 32], shift: i32, max: i32) {
    unsafe { sao_band_impl(dst, dst_stride, src, src_stride, w, h, table, shift, max) }
}

#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn sao_band_impl(dst: &mut [u16], dst_stride: usize, src: &[u16], src_stride: usize, w: usize, h: usize, table: &[i16; 32], shift: i32, max: i32) {
    unsafe {
        // The four consecutive bands (mod 32) with nonzero offsets.
        let mut bands = [0i16; 4];
        let mut offs = [0i16; 4];
        let mut k = 0;
        for b in 0..32 {
            if table[b] != 0 && k < 4 {
                bands[k] = b as i16;
                offs[k] = table[b];
                k += 1;
            }
        }
        let sh = _mm_cvtsi32_si128(shift);
        let maxv = _mm256_set1_epi16(max as i16);
        let bv: [__m256i; 4] = std::array::from_fn(|i| _mm256_set1_epi16(bands[i]));
        let ov: [__m256i; 4] = std::array::from_fn(|i| _mm256_set1_epi16(offs[i]));
        for y in 0..h {
            let mut x = 0;
            while x < w {
                let n = (w - x).min(16);
                let s = src.as_ptr().add(y * src_stride + x);
                let v = if n == 16 { _mm256_loadu_si256(s as *const __m256i) } else { load_n(s as *const i16, n) };
                let band = _mm256_srl_epi16(v, sh);
                let mut off = _mm256_setzero_si256();
                for i in 0..k {
                    off = _mm256_blendv_epi8(off, ov[i], _mm256_cmpeq_epi16(band, bv[i]));
                }
                let r = clip_u16(_mm256_add_epi16(v, off), maxv);
                store_n_u16(dst.as_mut_ptr().add(y * dst_stride + x), r, n);
                x += 16;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn sao_edge_avx2(dst: &mut [u16], src: &[u16], origin: usize, stride: usize, w: usize, h: usize, na: isize, nb: isize, off: &[i16; 5], max: i32) {
    unsafe { sao_edge_impl(dst, src, origin, stride, w, h, na, nb, off, max) }
}

#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn sao_edge_impl(dst: &mut [u16], src: &[u16], origin: usize, stride: usize, w: usize, h: usize, na: isize, nb: isize, off: &[i16; 5], max: i32) {
    unsafe {
        let maxv = _mm256_set1_epi16(max as i16);
        let one = _mm256_set1_epi16(1);
        // edgeIdx = 2 + sign(v-a) + sign(v-b) in 0..=4 → offsets via compares.
        let o0 = _mm256_set1_epi16(off[0]);
        let o1 = _mm256_set1_epi16(off[1]);
        let o3 = _mm256_set1_epi16(off[3]);
        let o4 = _mm256_set1_epi16(off[4]);
        let two = _mm256_set1_epi16(2);
        let three = _mm256_set1_epi16(3);
        let four = _mm256_set1_epi16(4);
        for y in 0..h {
            let mut x = 0;
            while x < w {
                let n = (w - x).min(16);
                let i = origin + y * stride + x;
                let last = i + n - 1;
                // All three loads must stay inside the slice.
                let need = last as isize + 16 - n as isize;
                if (last as isize + na.max(nb)) as usize + (16 - n) >= src.len() || need as usize >= src.len() {
                    // Tail near the buffer end: scalar.
                    for xx in x..w {
                        let ii = origin + y * stride + xx;
                        let v = src[ii] as i32;
                        let a = src[(ii as isize + na) as usize] as i32;
                        let b = src[(ii as isize + nb) as usize] as i32;
                        let e = (2 + (v - a).signum() + (v - b).signum()) as usize;
                        dst[ii] = (v + off[e] as i32).clamp(0, max) as u16;
                    }
                    break;
                }
                let v = _mm256_loadu_si256(src.as_ptr().add(i) as *const __m256i);
                let a = _mm256_loadu_si256(src.as_ptr().offset(i as isize + na) as *const __m256i);
                let b = _mm256_loadu_si256(src.as_ptr().offset(i as isize + nb) as *const __m256i);
                // sign(v - a) = (v > a) - (v < a); samples < 32768 so signed compares are exact.
                let sa = _mm256_sub_epi16(_mm256_and_si256(_mm256_cmpgt_epi16(v, a), one), _mm256_and_si256(_mm256_cmpgt_epi16(a, v), one));
                let sb = _mm256_sub_epi16(_mm256_and_si256(_mm256_cmpgt_epi16(v, b), one), _mm256_and_si256(_mm256_cmpgt_epi16(b, v), one));
                let e = _mm256_add_epi16(_mm256_add_epi16(sa, sb), two);
                let mut o = _mm256_setzero_si256();
                o = _mm256_blendv_epi8(o, o0, _mm256_cmpeq_epi16(e, _mm256_setzero_si256()));
                o = _mm256_blendv_epi8(o, o1, _mm256_cmpeq_epi16(e, one));
                o = _mm256_blendv_epi8(o, o3, _mm256_cmpeq_epi16(e, three));
                o = _mm256_blendv_epi8(o, o4, _mm256_cmpeq_epi16(e, four));
                let r = clip_u16(_mm256_add_epi16(v, o), maxv);
                store_n_u16(dst.as_mut_ptr().add(i), r, n);
                x += 16;
            }
        }
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

    fn avx2() -> Option<HevcDsp> {
        if !std::is_x86_feature_detected!("avx2") {
            return None;
        }
        let mut d = HevcDsp::SCALAR;
        install(&mut d);
        Some(d)
    }

    #[test]
    fn interp_matches_scalar() {
        let Some(d) = avx2() else { return };
        let s = HevcDsp::SCALAR;
        let mut seed = 1u64;
        let stride = 96;
        for &bd in &[8u32, 10, 12] {
            let maxv = (1u32 << bd) - 1;
            let src: Vec<u16> = (0..stride * 96).map(|_| (lcg(&mut seed) % (maxv + 1)) as u16).collect();
            let shift1 = bd.min(12) as i32 - 8;
            for &(w, h) in &[(2usize, 4usize), (4, 4), (4, 8), (6, 8), (8, 4), (12, 16), (16, 16), (24, 32), (32, 8), (48, 64), (64, 64)] {
                for frac in 1..8 {
                    let mut a = vec![0i16; w * h];
                    let mut b = vec![0i16; w * h];
                    if frac < 4 {
                        (s.qpel_h)(&mut a, &src, stride, w, h, frac, shift1);
                        (d.qpel_h)(&mut b, &src, stride, w, h, frac, shift1);
                        assert_eq!(a, b, "qpel_h bd={bd} {w}x{h} frac={frac}");
                        (s.qpel_v)(&mut a, &src, stride, w, h, frac, shift1);
                        (d.qpel_v)(&mut b, &src, stride, w, h, frac, shift1);
                        assert_eq!(a, b, "qpel_v bd={bd} {w}x{h} frac={frac}");
                        let mid: Vec<i16> = (0..stride * 96).map(|_| (lcg(&mut seed) % 30000) as i16 - 15000).collect();
                        (s.qpel_v2)(&mut a, &mid, stride, w, h, frac);
                        (d.qpel_v2)(&mut b, &mid, stride, w, h, frac);
                        assert_eq!(a, b, "qpel_v2 {w}x{h} frac={frac}");
                    }
                    (s.epel_h)(&mut a, &src, stride, w, h, frac, shift1);
                    (d.epel_h)(&mut b, &src, stride, w, h, frac, shift1);
                    assert_eq!(a, b, "epel_h bd={bd} {w}x{h} frac={frac}");
                    (s.epel_v)(&mut a, &src, stride, w, h, frac, shift1);
                    (d.epel_v)(&mut b, &src, stride, w, h, frac, shift1);
                    assert_eq!(a, b, "epel_v bd={bd} {w}x{h} frac={frac}");
                    let mid: Vec<i16> = (0..stride * 96).map(|_| (lcg(&mut seed) % 30000) as i16 - 15000).collect();
                    (s.epel_v2)(&mut a, &mid, stride, w, h, frac);
                    (d.epel_v2)(&mut b, &mid, stride, w, h, frac);
                    assert_eq!(a, b, "epel_v2 {w}x{h} frac={frac}");
                }
                let mut a = vec![0i16; w * h];
                let mut b = vec![0i16; w * h];
                (s.qpel_copy)(&mut a, &src, stride, w, h, 14 - bd as i32);
                (d.qpel_copy)(&mut b, &src, stride, w, h, 14 - bd as i32);
                assert_eq!(a, b, "copy {w}x{h}");
            }
        }
    }

    #[test]
    fn combine_matches_scalar() {
        let Some(d) = avx2() else { return };
        let s = HevcDsp::SCALAR;
        let mut seed = 3u64;
        for &bd in &[8u32, 10, 12] {
            let max = (1i32 << bd) - 1;
            for &(w, h) in &[(2usize, 4usize), (4, 4), (6, 8), (8, 8), (12, 16), (16, 8), (24, 4), (32, 32), (64, 64)] {
                let a: Vec<i16> = (0..w * h).map(|_| (lcg(&mut seed) % 32000) as i16 - 16000).collect();
                let b: Vec<i16> = (0..w * h).map(|_| (lcg(&mut seed) % 32000) as i16 - 16000).collect();
                let stride = w + 5;
                let mut d1 = vec![0u16; stride * h];
                let mut d2 = vec![0u16; stride * h];
                (s.uni)(&mut d1, stride, &a, w, h, 14 - bd as i32, max);
                (d.uni)(&mut d2, stride, &a, w, h, 14 - bd as i32, max);
                assert_eq!(d1, d2, "uni {w}x{h} bd={bd}");
                (s.bi)(&mut d1, stride, &a, &b, w, h, 15 - bd as i32, max);
                (d.bi)(&mut d2, stride, &a, &b, w, h, 15 - bd as i32, max);
                assert_eq!(d1, d2, "bi {w}x{h} bd={bd}");
                for &(log2_wd, wt, o) in &[(6 + 14 - bd as i32, 128, 0), (0 + 14 - bd as i32, 1, 5), (7 + 14 - bd as i32, -20, -3), (3 + 14 - bd as i32, 255, 127)] {
                    (s.weighted_uni)(&mut d1, stride, &a, w, h, log2_wd, wt, o, max);
                    (d.weighted_uni)(&mut d2, stride, &a, w, h, log2_wd, wt, o, max);
                    assert_eq!(d1, d2, "wuni {w}x{h} bd={bd} {log2_wd} {wt} {o}");
                    (s.weighted_bi)(&mut d1, stride, &a, &b, w, h, log2_wd, wt, 3 - wt, o, -o, max);
                    (d.weighted_bi)(&mut d2, stride, &a, &b, w, h, log2_wd, wt, 3 - wt, o, -o, max);
                    assert_eq!(d1, d2, "wbi {w}x{h} bd={bd}");
                }
                let res: Vec<i16> = (0..w * w).map(|_| (lcg(&mut seed) % 2000) as i16 - 1000).collect();
                if w == h && w >= 4 && w.is_power_of_two() {
                    let base: Vec<u16> = (0..stride * h).map(|_| (lcg(&mut seed) % (max as u32 + 1)) as u16).collect();
                    let mut d1 = base.clone();
                    let mut d2 = base.clone();
                    (s.add_residual)(&mut d1, stride, &res, w, max);
                    (d.add_residual)(&mut d2, stride, &res, w, max);
                    assert_eq!(d1, d2, "add_residual {w}");
                }
            }
        }
    }

    #[test]
    fn idct_matches_scalar() {
        let Some(d) = avx2() else { return };
        let s = HevcDsp::SCALAR;
        let mut seed = 9u64;
        for &(n, log2) in &[(4usize, 2u32), (8, 3), (16, 4), (32, 5)] {
            for trial in 0..300 {
                let mut c = vec![0i16; n * n];
                let (mx, my) = if trial % 4 == 0 { (n - 1, n - 1) } else { ((lcg(&mut seed) as usize) % n, (lcg(&mut seed) as usize) % n) };
                for y in 0..=my {
                    for x in 0..=mx {
                        if lcg(&mut seed) % 2 == 0 {
                            c[y * n + x] = (lcg(&mut seed) as i32 % 65536 - 32768) as i16;
                        }
                    }
                }
                let bd_shift = 20 - 8 - (trial % 3) as i32 * 2;
                let mut a = c.clone();
                let mut b = c.clone();
                (s.idct[(log2 - 2) as usize])(&mut a, bd_shift, mx, my);
                (d.idct[(log2 - 2) as usize])(&mut b, bd_shift, mx, my);
                assert_eq!(a, b, "idct n={n} trial={trial} mx={mx} my={my}");
            }
        }
    }

    #[test]
    fn sao_matches_scalar() {
        let Some(d) = avx2() else { return };
        let s = HevcDsp::SCALAR;
        let mut seed = 11u64;
        let stride = 80;
        for &bd in &[8u32, 10] {
            let max = (1i32 << bd) - 1;
            let src: Vec<u16> = (0..stride * 80).map(|_| (lcg(&mut seed) % (max as u32 + 1)) as u16).collect();
            for &(w, h) in &[(3usize, 5usize), (8, 8), (16, 16), (31, 17), (64, 64)] {
                let mut table = [0i16; 32];
                let pos = (lcg(&mut seed) % 32) as usize;
                for k in 0..4 {
                    table[(pos + k) & 31] = (lcg(&mut seed) % 15) as i16 - 7;
                }
                let mut d1 = src.clone();
                let mut d2 = src.clone();
                let off = 8 * stride + 8;
                (s.sao_band)(&mut d1[off..], stride, &src[off..], stride, w, h, &table, bd as i32 - 5, max);
                (d.sao_band)(&mut d2[off..], stride, &src[off..], stride, w, h, &table, bd as i32 - 5, max);
                assert_eq!(d1, d2, "band {w}x{h}");
                let offs: [i16; 5] = [(lcg(&mut seed) % 7) as i16, (lcg(&mut seed) % 7) as i16, 0, -((lcg(&mut seed) % 7) as i16), -((lcg(&mut seed) % 7) as i16)];
                for &(na, nb) in &[(-1isize, 1isize), (-(stride as isize), stride as isize), (-(stride as isize) - 1, stride as isize + 1), (-(stride as isize) + 1, stride as isize - 1)] {
                    let mut d1 = src.clone();
                    let mut d2 = src.clone();
                    (s.sao_edge)(&mut d1, &src, off, stride, w, h, na, nb, &offs, max);
                    (d.sao_edge)(&mut d2, &src, off, stride, w, h, na, nb, &offs, max);
                    assert_eq!(d1, d2, "edge {w}x{h} {na} {nb}");
                }
            }
        }
    }
}
