//! AVX2 versions of the H.264 kernels (x86-64).
//!
//! Sixteen 16-bit lanes per vector; a block row of up to 16 samples is one
//! vector. The six-tap sums (`a - 5b + 20c + 20d - 5e + f`) fit 16 bits for
//! 8-bit input, so the horizontal and vertical half-sample filters run in
//! `i16` and pack with saturation; the centre position filters the 16-bit
//! horizontal intermediates vertically with 32-bit `pmaddwd` pairs. Quarter
//! positions are `pavgb` of two 8-bit results.

#![cfg(target_arch = "x86_64")]

use std::arch::x86_64::*;

use super::h264::{H264Dsp, NO_DC, PRED_STRIDE};

/// Replace the scalar entries of `d` with the AVX2 kernels.
pub fn install(d: &mut H264Dsp) {
    d.qpel = [
        qpel_avx2::<0, 0>,
        qpel_avx2::<1, 0>,
        qpel_avx2::<2, 0>,
        qpel_avx2::<3, 0>,
        qpel_avx2::<0, 1>,
        qpel_avx2::<1, 1>,
        qpel_avx2::<2, 1>,
        qpel_avx2::<3, 1>,
        qpel_avx2::<0, 2>,
        qpel_avx2::<1, 2>,
        qpel_avx2::<2, 2>,
        qpel_avx2::<3, 2>,
        qpel_avx2::<0, 3>,
        qpel_avx2::<1, 3>,
        qpel_avx2::<2, 3>,
        qpel_avx2::<3, 3>,
    ];
    d.chroma = chroma_avx2;
    d.copy = copy_avx2;
    d.avg = avg_avx2;
    d.weighted_uni = weighted_uni_avx2;
    d.weighted_bi = weighted_bi_avx2;
    d.deblock_luma_v = deblock_luma_v_avx2;
    d.deblock_luma_h = deblock_luma_h_avx2;
    d.deblock_luma_v_intra = deblock_luma_v_intra_avx2;
    d.deblock_luma_h_intra = deblock_luma_h_intra_avx2;
    d.deblock_chroma_v = deblock_chroma_v_avx2;
    d.deblock_chroma_h = deblock_chroma_h_avx2;
    d.deblock_chroma_v_intra = deblock_chroma_v_intra_avx2;
    d.deblock_chroma_h_intra = deblock_chroma_h_intra_avx2;
    d.idct4_add = idct4_add_avx2;
    d.idct8_add = idct8_add_avx2;
    d.idct4_dc_add = idct4_dc_add_avx2;
    d.idct8_dc_add = idct8_dc_add_avx2;
    d.residual4 = residual4_avx2;
    d.residual8 = residual8_avx2;
}

/// Store the first `n` (≤ 16) bytes of the low 128 bits of `v`.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn store_u8_n(dst: *mut u8, v: __m128i, n: usize) {
    unsafe {
        if n == 16 {
            _mm_storeu_si128(dst as *mut __m128i, v);
        } else if n == 8 {
            _mm_storel_epi64(dst as *mut __m128i, v);
        } else if n == 4 {
            *(dst as *mut i32) = _mm_cvtsi128_si32(v);
        } else {
            let mut t = [0u8; 16];
            _mm_storeu_si128(t.as_mut_ptr() as *mut __m128i, v);
            std::ptr::copy_nonoverlapping(t.as_ptr(), dst, n);
        }
    }
}

/// Load 16 bytes as 16 × i16.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn load16(p: *const u8) -> __m256i {
    unsafe { _mm256_cvtepu8_epi16(_mm_loadu_si128(p as *const __m128i)) }
}

/// Six-tap sum over six i16 vectors: `a - 5b + 20c + 20d - 5e + f`.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn tap6_16(a: __m256i, b: __m256i, c: __m256i, d: __m256i, e: __m256i, f: __m256i) -> __m256i {
    unsafe {
        let t = _mm256_add_epi16(c, d);
        let u = _mm256_add_epi16(b, e);
        let v = _mm256_add_epi16(a, f);
        // v + 20t - 5u = v + (t << 4) + (t << 2) - (u << 2) - u
        let t20 = _mm256_add_epi16(_mm256_slli_epi16(t, 4), _mm256_slli_epi16(t, 2));
        let u5 = _mm256_add_epi16(_mm256_slli_epi16(u, 2), u);
        _mm256_sub_epi16(_mm256_add_epi16(v, t20), u5)
    }
}

/// `clip((v + 16) >> 5)` of 16 i16 lanes packed to 16 u8 (low 128 bits).
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn round5_pack(v: __m256i) -> __m128i {
    unsafe {
        let r = _mm256_srai_epi16(_mm256_add_epi16(v, _mm256_set1_epi16(16)), 5);
        let p = _mm256_packus_epi16(r, r); // per lane: [lo8 lo8 | hi8 hi8]
        let p = _mm256_permute4x64_epi64(p, 0b11_01_10_00);
        _mm256_castsi256_si128(p)
    }
}

/// Horizontal half-sample intermediate (i16) for window row `row`.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn b1_row(src: *const u8, stride: usize, row: usize) -> __m256i {
    unsafe {
        let p = src.add(row * stride);
        tap6_16(load16(p), load16(p.add(1)), load16(p.add(2)), load16(p.add(3)), load16(p.add(4)), load16(p.add(5)))
    }
}

/// Vertical half-sample intermediate (i16) at window column `col`, block row `y`.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn h1_row(src: *const u8, stride: usize, col: usize, y: usize) -> __m256i {
    unsafe {
        let p = src.add(y * stride + col);
        tap6_16(load16(p), load16(p.add(stride)), load16(p.add(2 * stride)), load16(p.add(3 * stride)), load16(p.add(4 * stride)), load16(p.add(5 * stride)))
    }
}

/// Centre position row `y`: vertical six-tap over b1 rows y..y+5 with 32-bit
/// accumulation, `clip((v + 512) >> 10)`.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn j_row(src: *const u8, stride: usize, y: usize) -> __m128i {
    unsafe {
        let r0 = b1_row(src, stride, y);
        let r1 = b1_row(src, stride, y + 1);
        let r2 = b1_row(src, stride, y + 2);
        let r3 = b1_row(src, stride, y + 3);
        let r4 = b1_row(src, stride, y + 4);
        let r5 = b1_row(src, stride, y + 5);
        let c01 = _mm256_set1_epi32(pair(1, -5));
        let c23 = _mm256_set1_epi32(pair(20, 20));
        let c45 = _mm256_set1_epi32(pair(-5, 1));
        let round = _mm256_set1_epi32(512);
        let lo = _mm256_add_epi32(
            _mm256_add_epi32(_mm256_madd_epi16(_mm256_unpacklo_epi16(r0, r1), c01), _mm256_madd_epi16(_mm256_unpacklo_epi16(r2, r3), c23)),
            _mm256_add_epi32(_mm256_madd_epi16(_mm256_unpacklo_epi16(r4, r5), c45), round),
        );
        let hi = _mm256_add_epi32(
            _mm256_add_epi32(_mm256_madd_epi16(_mm256_unpackhi_epi16(r0, r1), c01), _mm256_madd_epi16(_mm256_unpackhi_epi16(r2, r3), c23)),
            _mm256_add_epi32(_mm256_madd_epi16(_mm256_unpackhi_epi16(r4, r5), c45), round),
        );
        // packs per lane keeps order (lo = lanes 0..3 | 8..11, hi = 4..7 | 12..15).
        let v = _mm256_packs_epi32(_mm256_srai_epi32(lo, 10), _mm256_srai_epi32(hi, 10));
        let p = _mm256_packus_epi16(v, v);
        let p = _mm256_permute4x64_epi64(p, 0b11_01_10_00);
        _mm256_castsi256_si128(p)
    }
}

#[inline(always)]
fn pair(a: i16, b: i16) -> i32 {
    (a as u16 as i32) | ((b as u16 as i32) << 16)
}

/// Full samples of block row `y` (window offset 2, 2) as u8x16.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn g_row(src: *const u8, stride: usize, y: usize, dx: usize) -> __m128i {
    unsafe { _mm_loadu_si128(src.add((y + 2) * stride + 2 + dx) as *const __m128i) }
}

fn qpel_avx2<const XF: usize, const YF: usize>(dst: &mut [u8], src: &[u8], stride: usize, w: usize, h: usize) {
    // The window is (w + 5) x (h + 5); a 16-lane load from column x reads
    // x + 15 (+5 for taps): needs w + 5 + 16 - 1 <= slice extent per row.
    let need = (h + 5 - 1) * stride + 21;
    if src.len() < need {
        return (H264Dsp::SCALAR.qpel[YF * 4 + XF])(dst, src, stride, w, h);
    }
    unsafe { qpel_impl::<XF, YF>(dst, src, stride, w, h) }
}

#[target_feature(enable = "avx2")]
unsafe fn qpel_impl<const XF: usize, const YF: usize>(dst: &mut [u8], src: &[u8], stride: usize, w: usize, h: usize) {
    unsafe {
        let s = src.as_ptr();
        for y in 0..h {
            let d = dst.as_mut_ptr().add(y * PRED_STRIDE);
            let b = || round5_pack(b1_row(s, stride, y + 2));
            let b_below = || round5_pack(b1_row(s, stride, y + 3));
            let hh = || round5_pack(h1_row(s, stride, 2, y));
            let hh_right = || round5_pack(h1_row(s, stride, 3, y));
            let v: __m128i = match (XF, YF) {
                (0, 0) => g_row(s, stride, y, 0),
                (1, 0) => _mm_avg_epu8(g_row(s, stride, y, 0), b()),
                (2, 0) => b(),
                (3, 0) => _mm_avg_epu8(g_row(s, stride, y, 1), b()),
                (0, 1) => _mm_avg_epu8(g_row(s, stride, y, 0), hh()),
                (0, 2) => hh(),
                (0, 3) => _mm_avg_epu8(_mm_loadu_si128(s.add((y + 3) * stride + 2) as *const __m128i), hh()),
                (2, 2) => j_row(s, stride, y),
                (1, 1) => _mm_avg_epu8(b(), hh()),
                (3, 1) => _mm_avg_epu8(b(), hh_right()),
                (1, 3) => _mm_avg_epu8(hh(), b_below()),
                (3, 3) => _mm_avg_epu8(hh_right(), b_below()),
                (2, 1) => _mm_avg_epu8(b(), j_row(s, stride, y)),
                (2, 3) => _mm_avg_epu8(j_row(s, stride, y), b_below()),
                (1, 2) => _mm_avg_epu8(hh(), j_row(s, stride, y)),
                (3, 2) => _mm_avg_epu8(j_row(s, stride, y), hh_right()),
                _ => unreachable!(),
            };
            // The scratch row is 16 wide whatever `w` is.
            _mm_storeu_si128(d as *mut __m128i, v);
        }
    }
}

fn chroma_avx2(dst: &mut [u8], src: &[u8], stride: usize, w: usize, h: usize, xf: i32, yf: i32) {
    if src.len() < h * stride + 9 {
        return (H264Dsp::SCALAR.chroma)(dst, src, stride, w, h, xf, yf);
    }
    unsafe { chroma_impl(dst, src, stride, w, h, xf, yf) }
}

#[target_feature(enable = "avx2")]
unsafe fn chroma_impl(dst: &mut [u8], src: &[u8], stride: usize, w: usize, h: usize, xf: i32, yf: i32) {
    unsafe {
        // Chroma blocks are at most 8 wide: eight i16 lanes.
        let a = _mm_set1_epi16(((8 - xf) * (8 - yf)) as i16);
        let b = _mm_set1_epi16((xf * (8 - yf)) as i16);
        let c = _mm_set1_epi16(((8 - xf) * yf) as i16);
        let d = _mm_set1_epi16((xf * yf) as i16);
        let round = _mm_set1_epi16(32);
        let s = src.as_ptr();
        let ld = |p: *const u8| _mm_cvtepu8_epi16(_mm_loadl_epi64(p as *const __m128i));
        let _ = w;
        let mut r0v = ld(s);
        let mut r0v1 = ld(s.add(1));
        for y in 0..h {
            let r1 = s.add((y + 1) * stride);
            let r1v = ld(r1);
            let r1v1 = ld(r1.add(1));
            let v = _mm_add_epi16(
                _mm_add_epi16(_mm_mullo_epi16(r0v, a), _mm_mullo_epi16(r0v1, b)),
                _mm_add_epi16(_mm_mullo_epi16(r1v, c), _mm_mullo_epi16(r1v1, d)),
            );
            let v = _mm_srli_epi16(_mm_add_epi16(v, round), 6);
            _mm_storel_epi64(dst.as_mut_ptr().add(y * PRED_STRIDE) as *mut __m128i, _mm_packus_epi16(v, v));
            r0v = r1v;
            r0v1 = r1v1;
        }
    }
}

fn copy_avx2(dst: &mut [u8], stride: usize, src: &[u8], w: usize, h: usize) {
    assert!((h - 1) * stride + w <= dst.len() && h * PRED_STRIDE <= src.len());
    unsafe { copy_impl(dst.as_mut_ptr(), stride, src.as_ptr(), w, h) }
}

#[target_feature(enable = "avx2")]
unsafe fn copy_impl(dst: *mut u8, stride: usize, src: *const u8, w: usize, h: usize) {
    unsafe {
        for y in 0..h {
            let s = src.add(y * PRED_STRIDE);
            let d = dst.add(y * stride);
            match w {
                16 => _mm_storeu_si128(d as *mut __m128i, _mm_loadu_si128(s as *const __m128i)),
                8 => std::ptr::write_unaligned(d as *mut u64, std::ptr::read_unaligned(s as *const u64)),
                4 => std::ptr::write_unaligned(d as *mut u32, std::ptr::read_unaligned(s as *const u32)),
                _ => std::ptr::copy_nonoverlapping(s, d, w),
            }
        }
    }
}

fn avg_avx2(dst: &mut [u8], stride: usize, a: &[u8], b: &[u8], w: usize, h: usize) {
    unsafe { avg_impl(dst, stride, a, b, w, h) }
}

#[target_feature(enable = "avx2")]
unsafe fn avg_impl(dst: &mut [u8], stride: usize, a: &[u8], b: &[u8], w: usize, h: usize) {
    unsafe {
        // The scratch rows are 16 wide, so a full load is always in bounds;
        // only the store into the plane is sized.
        for y in 0..h {
            let va = _mm_loadu_si128(a.as_ptr().add(y * PRED_STRIDE) as *const __m128i);
            let vb = _mm_loadu_si128(b.as_ptr().add(y * PRED_STRIDE) as *const __m128i);
            store_u8_n(dst.as_mut_ptr().add(y * stride), _mm_avg_epu8(va, vb), w);
        }
    }
}

fn weighted_uni_avx2(dst: &mut [u8], stride: usize, src: &[u8], w: usize, h: usize, log_wd: i32, wt: i32, o: i32) {
    unsafe { weighted_uni_impl(dst, stride, src, w, h, log_wd, wt, o) }
}

#[target_feature(enable = "avx2")]
unsafe fn weighted_uni_impl(dst: &mut [u8], stride: usize, src: &[u8], w: usize, h: usize, log_wd: i32, wt: i32, o: i32) {
    unsafe {
        // src * wt + round fits i16 only for |wt| <= 128 (spec: -128..127) ✓.
        let wv = _mm256_set1_epi16(wt as i16);
        let ov = _mm256_set1_epi16(o as i16);
        let round = _mm256_set1_epi16(if log_wd >= 1 { 1 << (log_wd - 1) } else { 0 });
        let sh = _mm_cvtsi32_si128(log_wd.max(0));
        for y in 0..h {
            let s = load16(src.as_ptr().add(y * PRED_STRIDE));
            let v = _mm256_add_epi16(_mm256_sra_epi16(_mm256_add_epi16(_mm256_mullo_epi16(s, wv), round), sh), ov);
            let p = _mm256_packus_epi16(v, v);
            let p = _mm256_permute4x64_epi64(p, 0b11_01_10_00);
            store_u8_n(dst.as_mut_ptr().add(y * stride), _mm256_castsi256_si128(p), w);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn weighted_bi_avx2(dst: &mut [u8], stride: usize, a: &[u8], b: &[u8], w: usize, h: usize, log_wd: i32, w0: i32, w1: i32, o0: i32, o1: i32) {
    unsafe { weighted_bi_impl(dst, stride, a, b, w, h, log_wd, w0, w1, o0, o1) }
}

#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn weighted_bi_impl(dst: &mut [u8], stride: usize, a: &[u8], b: &[u8], w: usize, h: usize, log_wd: i32, w0: i32, w1: i32, o0: i32, o1: i32) {
    unsafe {
        // a * w0 + b * w1 can reach 2 * 255 * 128 = 65280: use 32-bit lanes.
        let w0v = _mm256_set1_epi32(w0);
        let w1v = _mm256_set1_epi32(w1);
        let round = _mm256_set1_epi32(1 << log_wd);
        let off = _mm256_set1_epi32((o0 + o1 + 1) >> 1);
        let sh = _mm_cvtsi32_si128(log_wd + 1);
        for y in 0..h {
            let va = _mm_loadu_si128(a.as_ptr().add(y * PRED_STRIDE) as *const __m128i);
            let vb = _mm_loadu_si128(b.as_ptr().add(y * PRED_STRIDE) as *const __m128i);
            let alo = _mm256_cvtepu8_epi32(va);
            let ahi = _mm256_cvtepu8_epi32(_mm_srli_si128(va, 8));
            let blo = _mm256_cvtepu8_epi32(vb);
            let bhi = _mm256_cvtepu8_epi32(_mm_srli_si128(vb, 8));
            let lo = _mm256_add_epi32(_mm256_sra_epi32(_mm256_add_epi32(_mm256_add_epi32(_mm256_mullo_epi32(alo, w0v), _mm256_mullo_epi32(blo, w1v)), round), sh), off);
            let hi = _mm256_add_epi32(_mm256_sra_epi32(_mm256_add_epi32(_mm256_add_epi32(_mm256_mullo_epi32(ahi, w0v), _mm256_mullo_epi32(bhi, w1v)), round), sh), off);
            let v16 = _mm256_permute4x64_epi64(_mm256_packs_epi32(lo, hi), 0b11_01_10_00);
            let p = _mm256_packus_epi16(v16, v16);
            let p = _mm256_permute4x64_epi64(p, 0b11_01_10_00);
            store_u8_n(dst.as_mut_ptr().add(y * stride), _mm256_castsi256_si128(p), w);
        }
    }
}

// ----------------------------------------------------------------------
// Deblocking
// ----------------------------------------------------------------------
//
// Sixteen lines of a luma edge are sixteen i16 lanes of one 256-bit vector
// per sample position (p2..q2, and p3/q3 for bS 4); eight lines of a chroma
// edge are eight lanes of a 128-bit one. A horizontal edge loads a sample
// position as one row; a vertical edge transposes 16 rows x 8 bytes into 8
// column vectors, filters, and transposes back. Lines the standard leaves
// alone (bS 0, or the alpha/beta test failing) keep their old values by a
// blend on the filter mask.

/// `|a - b| < t` per i16 lane, as a mask.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn diff_lt(a: __m256i, b: __m256i, t: __m256i) -> __m256i {
    unsafe { _mm256_cmpgt_epi16(t, _mm256_abs_epi16(_mm256_sub_epi16(a, b))) }
}

#[target_feature(enable = "avx2")]
#[inline]
unsafe fn diff_lt128(a: __m128i, b: __m128i, t: __m128i) -> __m128i {
    unsafe { _mm_cmpgt_epi16(t, _mm_abs_epi16(_mm_sub_epi16(a, b))) }
}

/// The eight positions of sixteen luma lines: `[p3, p2, p1, p0, q0, q1, q2, q3]`
/// as 16 x i16 each.
type LumaLines = [__m256i; 8];

/// bS < 4 luma filter on sixteen lines (8.7.2.3), in place on the vectors.
/// `tc0v` holds the line's tC0 (−1 = bS 0).
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn luma_filter_normal(v: &mut LumaLines, alpha: i32, beta: i32, tc0v: __m256i) {
    unsafe {
        let [_, p2, p1, p0, q0, q1, q2, _] = *v;
        let alpha = _mm256_set1_epi16(alpha as i16);
        let beta = _mm256_set1_epi16(beta as i16);
        let zero = _mm256_setzero_si256();
        let bs_on = _mm256_cmpgt_epi16(tc0v, _mm256_set1_epi16(-1));
        let mask = _mm256_and_si256(
            _mm256_and_si256(diff_lt(p0, q0, alpha), diff_lt(p1, p0, beta)),
            _mm256_and_si256(diff_lt(q1, q0, beta), bs_on),
        );
        let ap = diff_lt(p2, p0, beta);
        let aq = diff_lt(q2, q0, beta);
        // tc = tc0 + (ap < beta) + (aq < beta); masks are -1.
        let tc = _mm256_sub_epi16(_mm256_sub_epi16(tc0v, ap), aq);
        // delta = clip3(-tc, tc, ((q0 - p0) * 4 + (p1 - q1) + 4) >> 3)
        let d = _mm256_srai_epi16(
            _mm256_add_epi16(
                _mm256_add_epi16(_mm256_slli_epi16(_mm256_sub_epi16(q0, p0), 2), _mm256_sub_epi16(p1, q1)),
                _mm256_set1_epi16(4),
            ),
            3,
        );
        let d = _mm256_min_epi16(_mm256_max_epi16(d, _mm256_sub_epi16(zero, tc)), tc);
        let np0 = _mm256_add_epi16(p0, d);
        let nq0 = _mm256_sub_epi16(q0, d);
        // p1' = p1 + clip3(-tc0, tc0, (p2 + ((p0 + q0 + 1) >> 1) - 2 p1) >> 1), when ap
        let avg = _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(p0, q0), _mm256_set1_epi16(1)), 1);
        let ntc0 = _mm256_sub_epi16(zero, tc0v);
        let dp1 = _mm256_srai_epi16(_mm256_sub_epi16(_mm256_add_epi16(p2, avg), _mm256_slli_epi16(p1, 1)), 1);
        let dp1 = _mm256_min_epi16(_mm256_max_epi16(dp1, ntc0), tc0v);
        let np1 = _mm256_add_epi16(p1, _mm256_and_si256(dp1, ap));
        let dq1 = _mm256_srai_epi16(_mm256_sub_epi16(_mm256_add_epi16(q2, avg), _mm256_slli_epi16(q1, 1)), 1);
        let dq1 = _mm256_min_epi16(_mm256_max_epi16(dq1, ntc0), tc0v);
        let nq1 = _mm256_add_epi16(q1, _mm256_and_si256(dq1, aq));
        // Clip to 8 bits (p1'/q1' cannot leave the range; p0'/q0' can).
        let clip = |x: __m256i| _mm256_min_epi16(_mm256_max_epi16(x, zero), _mm256_set1_epi16(255));
        v[2] = _mm256_blendv_epi8(p1, np1, mask);
        v[3] = _mm256_blendv_epi8(p0, clip(np0), mask);
        v[4] = _mm256_blendv_epi8(q0, clip(nq0), mask);
        v[5] = _mm256_blendv_epi8(q1, nq1, mask);
    }
}

/// bS 4 luma filter on sixteen lines (8.7.2.4).
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn luma_filter_intra(v: &mut LumaLines, alpha: i32, beta: i32) {
    unsafe {
        let [p3, p2, p1, p0, q0, q1, q2, q3] = *v;
        let alphav = _mm256_set1_epi16(alpha as i16);
        let beta = _mm256_set1_epi16(beta as i16);
        let mask = _mm256_and_si256(_mm256_and_si256(diff_lt(p0, q0, alphav), diff_lt(p1, p0, beta)), diff_lt(q1, q0, beta));
        let strong = diff_lt(p0, q0, _mm256_set1_epi16(((alpha >> 2) + 2) as i16));
        let ap = _mm256_and_si256(diff_lt(p2, p0, beta), strong);
        let aq = _mm256_and_si256(diff_lt(q2, q0, beta), strong);
        let two = _mm256_set1_epi16(2);
        let four = _mm256_set1_epi16(4);
        let add = |a, b| _mm256_add_epi16(a, b);
        let dbl = |a| _mm256_slli_epi16(a, 1);
        // Weak: p0' = (2 p1 + p0 + q1 + 2) >> 2, q0' = (2 q1 + q0 + p1 + 2) >> 2.
        let wp0 = _mm256_srai_epi16(add(add(dbl(p1), p0), add(q1, two)), 2);
        let wq0 = _mm256_srai_epi16(add(add(dbl(q1), q0), add(p1, two)), 2);
        // Strong p side.
        let p0q0 = add(p0, q0);
        let sp0 = _mm256_srai_epi16(add(add(p2, dbl(add(p1, p0q0))), add(q1, four)), 3);
        let sp1 = _mm256_srai_epi16(add(add(p2, p1), add(p0q0, two)), 2);
        let sp2 = _mm256_srai_epi16(add(add(dbl(p3), add(p2, dbl(p2))), add(add(p1, p0q0), four)), 3);
        // Strong q side.
        let sq0 = _mm256_srai_epi16(add(add(p1, dbl(add(p0q0, q1))), add(q2, four)), 3);
        let sq1 = _mm256_srai_epi16(add(add(p0q0, q1), add(q2, two)), 2);
        let sq2 = _mm256_srai_epi16(add(add(dbl(q3), add(q2, dbl(q2))), add(add(q1, p0q0), four)), 3);
        let np0 = _mm256_blendv_epi8(wp0, sp0, ap);
        let np1 = _mm256_blendv_epi8(p1, sp1, ap);
        let np2 = _mm256_blendv_epi8(p2, sp2, ap);
        let nq0 = _mm256_blendv_epi8(wq0, sq0, aq);
        let nq1 = _mm256_blendv_epi8(q1, sq1, aq);
        let nq2 = _mm256_blendv_epi8(q2, sq2, aq);
        v[1] = _mm256_blendv_epi8(p2, np2, mask);
        v[2] = _mm256_blendv_epi8(p1, np1, mask);
        v[3] = _mm256_blendv_epi8(p0, np0, mask);
        v[4] = _mm256_blendv_epi8(q0, nq0, mask);
        v[5] = _mm256_blendv_epi8(q1, nq1, mask);
        v[6] = _mm256_blendv_epi8(q2, nq2, mask);
    }
}

/// tC0 per lane for sixteen luma lines (four per segment).
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn tc0_luma(tc0: &[i8; 4]) -> __m256i {
    unsafe {
        let t = |k: usize| tc0[k] as i16;
        _mm256_setr_epi16(t(0), t(0), t(0), t(0), t(1), t(1), t(1), t(1), t(2), t(2), t(2), t(2), t(3), t(3), t(3), t(3))
    }
}

/// Pack sixteen i16 lanes to sixteen bytes.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn pack16(v: __m256i) -> __m128i {
    unsafe {
        let p = _mm256_packus_epi16(v, v);
        _mm256_castsi256_si128(_mm256_permute4x64_epi64(p, 0b11_01_10_00))
    }
}

/// Load the sixteen rows x 8 bytes around a vertical edge (`q0` at `data`)
/// as eight column vectors p3..q3.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn load_transposed_16x8(data: *const u8, stride: usize) -> LumaLines {
    unsafe {
        let mut r = [_mm_setzero_si128(); 16];
        for i in 0..16 {
            r[i] = _mm_loadl_epi64(data.add(i * stride).sub(4) as *const __m128i);
        }
        // Bytes: pairs of rows.
        let mut a = [_mm_setzero_si128(); 8];
        for j in 0..8 {
            a[j] = _mm_unpacklo_epi8(r[2 * j], r[2 * j + 1]);
        }
        // Words: quads of rows; lo = columns 0..3, hi = columns 4..7.
        let mut b = [_mm_setzero_si128(); 8];
        for j in 0..4 {
            b[2 * j] = _mm_unpacklo_epi16(a[2 * j], a[2 * j + 1]);
            b[2 * j + 1] = _mm_unpackhi_epi16(a[2 * j], a[2 * j + 1]);
        }
        // Dwords: octets of rows. c[c2][half]: columns pair.
        // b[0]: rows 0-3 cols 0..3; b[2]: rows 4-7 cols 0..3; b[4]: rows 8-11; b[6]: rows 12-15.
        // b[1],b[3],b[5],b[7]: the same for cols 4..7.
        let c01_lo = _mm_unpacklo_epi32(b[0], b[2]); // col0 rows0-7, col1 rows0-7
        let c23_lo = _mm_unpackhi_epi32(b[0], b[2]); // col2, col3 rows 0-7
        let c01_hi = _mm_unpacklo_epi32(b[4], b[6]); // col0, col1 rows 8-15
        let c23_hi = _mm_unpackhi_epi32(b[4], b[6]);
        let c45_lo = _mm_unpacklo_epi32(b[1], b[3]);
        let c67_lo = _mm_unpackhi_epi32(b[1], b[3]);
        let c45_hi = _mm_unpacklo_epi32(b[5], b[7]);
        let c67_hi = _mm_unpackhi_epi32(b[5], b[7]);
        let col = |lo: __m128i, hi: __m128i, second: bool| -> __m256i {
            let bytes = if second { _mm_unpackhi_epi64(lo, hi) } else { _mm_unpacklo_epi64(lo, hi) };
            _mm256_cvtepu8_epi16(bytes)
        };
        [
            col(c01_lo, c01_hi, false),
            col(c01_lo, c01_hi, true),
            col(c23_lo, c23_hi, false),
            col(c23_lo, c23_hi, true),
            col(c45_lo, c45_hi, false),
            col(c45_lo, c45_hi, true),
            col(c67_lo, c67_hi, false),
            col(c67_lo, c67_hi, true),
        ]
    }
}

/// Store eight column vectors back as sixteen rows x 8 bytes.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn store_transposed_16x8(data: *mut u8, stride: usize, v: &LumaLines) {
    unsafe {
        let c: [__m128i; 8] = [pack16(v[0]), pack16(v[1]), pack16(v[2]), pack16(v[3]), pack16(v[4]), pack16(v[5]), pack16(v[6]), pack16(v[7])];
        // Bytes: column pairs -> rows interleaved.
        let a01_lo = _mm_unpacklo_epi8(c[0], c[1]); // rows 0-7: c0 c1 per row
        let a01_hi = _mm_unpackhi_epi8(c[0], c[1]); // rows 8-15
        let a23_lo = _mm_unpacklo_epi8(c[2], c[3]);
        let a23_hi = _mm_unpackhi_epi8(c[2], c[3]);
        let a45_lo = _mm_unpacklo_epi8(c[4], c[5]);
        let a45_hi = _mm_unpackhi_epi8(c[4], c[5]);
        let a67_lo = _mm_unpacklo_epi8(c[6], c[7]);
        let a67_hi = _mm_unpackhi_epi8(c[6], c[7]);
        // Words: rows with c0..c3 / c4..c7.
        let b0123_r0 = _mm_unpacklo_epi16(a01_lo, a23_lo); // rows 0-3: c0..c3
        let b0123_r4 = _mm_unpackhi_epi16(a01_lo, a23_lo); // rows 4-7
        let b0123_r8 = _mm_unpacklo_epi16(a01_hi, a23_hi);
        let b0123_r12 = _mm_unpackhi_epi16(a01_hi, a23_hi);
        let b4567_r0 = _mm_unpacklo_epi16(a45_lo, a67_lo);
        let b4567_r4 = _mm_unpackhi_epi16(a45_lo, a67_lo);
        let b4567_r8 = _mm_unpacklo_epi16(a45_hi, a67_hi);
        let b4567_r12 = _mm_unpackhi_epi16(a45_hi, a67_hi);
        // Dwords: whole rows (8 bytes), two per vector.
        let rows = [
            _mm_unpacklo_epi32(b0123_r0, b4567_r0),  // rows 0,1
            _mm_unpackhi_epi32(b0123_r0, b4567_r0),  // rows 2,3
            _mm_unpacklo_epi32(b0123_r4, b4567_r4),  // rows 4,5
            _mm_unpackhi_epi32(b0123_r4, b4567_r4),  // rows 6,7
            _mm_unpacklo_epi32(b0123_r8, b4567_r8),  // rows 8,9
            _mm_unpackhi_epi32(b0123_r8, b4567_r8),  // rows 10,11
            _mm_unpacklo_epi32(b0123_r12, b4567_r12), // rows 12,13
            _mm_unpackhi_epi32(b0123_r12, b4567_r12), // rows 14,15
        ];
        for (k, pair) in rows.iter().enumerate() {
            let d0 = data.add(2 * k * stride).sub(4);
            let d1 = data.add((2 * k + 1) * stride).sub(4);
            _mm_storel_epi64(d0 as *mut __m128i, *pair);
            _mm_storel_epi64(d1 as *mut __m128i, _mm_srli_si128(*pair, 8));
        }
    }
}

fn deblock_luma_v_avx2(data: &mut [u8], off: usize, stride: usize, alpha: i32, beta: i32, tc0: &[i8; 4]) {
    if tc0.iter().all(|&t| t < 0) {
        return;
    }
    assert!(off >= 4 && off + 15 * stride + 4 <= data.len());
    unsafe { deblock_luma_v_impl(data.as_mut_ptr().add(off), stride, alpha, beta, tc0) }
}

#[target_feature(enable = "avx2")]
unsafe fn deblock_luma_v_impl(data: *mut u8, stride: usize, alpha: i32, beta: i32, tc0: &[i8; 4]) {
    unsafe {
        let mut v = load_transposed_16x8(data, stride);
        luma_filter_normal(&mut v, alpha, beta, tc0_luma(tc0));
        store_transposed_16x8(data, stride, &v);
    }
}

fn deblock_luma_v_intra_avx2(data: &mut [u8], off: usize, stride: usize, alpha: i32, beta: i32) {
    assert!(off >= 4 && off + 15 * stride + 4 <= data.len());
    unsafe { deblock_luma_v_intra_impl(data.as_mut_ptr().add(off), stride, alpha, beta) }
}

#[target_feature(enable = "avx2")]
unsafe fn deblock_luma_v_intra_impl(data: *mut u8, stride: usize, alpha: i32, beta: i32) {
    unsafe {
        let mut v = load_transposed_16x8(data, stride);
        luma_filter_intra(&mut v, alpha, beta);
        store_transposed_16x8(data, stride, &v);
    }
}

fn deblock_luma_h_avx2(data: &mut [u8], off: usize, stride: usize, alpha: i32, beta: i32, tc0: &[i8; 4]) {
    if tc0.iter().all(|&t| t < 0) {
        return;
    }
    assert!(off >= 3 * stride && off + 2 * stride + 16 <= data.len());
    unsafe { deblock_luma_h_impl(data.as_mut_ptr().add(off), stride, alpha, beta, tc0) }
}

#[target_feature(enable = "avx2")]
unsafe fn deblock_luma_h_impl(data: *mut u8, stride: usize, alpha: i32, beta: i32, tc0: &[i8; 4]) {
    unsafe {
        let ld = |k: isize| load16(data.offset(k * stride as isize));
        let mut v: LumaLines = [_mm256_setzero_si256(), ld(-3), ld(-2), ld(-1), ld(0), ld(1), ld(2), _mm256_setzero_si256()];
        luma_filter_normal(&mut v, alpha, beta, tc0_luma(tc0));
        _mm_storeu_si128(data.offset(-2 * stride as isize) as *mut __m128i, pack16(v[2]));
        _mm_storeu_si128(data.offset(-(stride as isize)) as *mut __m128i, pack16(v[3]));
        _mm_storeu_si128(data as *mut __m128i, pack16(v[4]));
        _mm_storeu_si128(data.add(stride) as *mut __m128i, pack16(v[5]));
    }
}

fn deblock_luma_h_intra_avx2(data: &mut [u8], off: usize, stride: usize, alpha: i32, beta: i32) {
    assert!(off >= 4 * stride && off + 3 * stride + 16 <= data.len());
    unsafe { deblock_luma_h_intra_impl(data.as_mut_ptr().add(off), stride, alpha, beta) }
}

#[target_feature(enable = "avx2")]
unsafe fn deblock_luma_h_intra_impl(data: *mut u8, stride: usize, alpha: i32, beta: i32) {
    unsafe {
        let ld = |k: isize| load16(data.offset(k * stride as isize));
        let mut v: LumaLines = [ld(-4), ld(-3), ld(-2), ld(-1), ld(0), ld(1), ld(2), ld(3)];
        luma_filter_intra(&mut v, alpha, beta);
        for k in 1..7 {
            _mm_storeu_si128(data.offset((k as isize - 4) * stride as isize) as *mut __m128i, pack16(v[k]));
        }
    }
}

// Chroma: eight lines, positions [p1, p0, q0, q1] as 8 x i16 in 128-bit vectors.
type ChromaLines = [__m128i; 4];

#[target_feature(enable = "avx2")]
#[inline]
unsafe fn chroma_filter_normal(v: &mut ChromaLines, alpha: i32, beta: i32, tc0v: __m128i) {
    unsafe {
        let [p1, p0, q0, q1] = *v;
        let alpha = _mm_set1_epi16(alpha as i16);
        let beta = _mm_set1_epi16(beta as i16);
        let zero = _mm_setzero_si128();
        let bs_on = _mm_cmpgt_epi16(tc0v, _mm_set1_epi16(-1));
        let mask = _mm_and_si128(_mm_and_si128(diff_lt128(p0, q0, alpha), diff_lt128(p1, p0, beta)), _mm_and_si128(diff_lt128(q1, q0, beta), bs_on));
        let tc = _mm_add_epi16(tc0v, _mm_set1_epi16(1));
        let d = _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(_mm_slli_epi16(_mm_sub_epi16(q0, p0), 2), _mm_sub_epi16(p1, q1)), _mm_set1_epi16(4)), 3);
        let d = _mm_min_epi16(_mm_max_epi16(d, _mm_sub_epi16(zero, tc)), tc);
        let clip = |x: __m128i| _mm_min_epi16(_mm_max_epi16(x, zero), _mm_set1_epi16(255));
        v[1] = _mm_blendv_epi8(p0, clip(_mm_add_epi16(p0, d)), mask);
        v[2] = _mm_blendv_epi8(q0, clip(_mm_sub_epi16(q0, d)), mask);
    }
}

#[target_feature(enable = "avx2")]
#[inline]
unsafe fn chroma_filter_intra(v: &mut ChromaLines, alpha: i32, beta: i32) {
    unsafe {
        let [p1, p0, q0, q1] = *v;
        let alpha = _mm_set1_epi16(alpha as i16);
        let beta = _mm_set1_epi16(beta as i16);
        let mask = _mm_and_si128(_mm_and_si128(diff_lt128(p0, q0, alpha), diff_lt128(p1, p0, beta)), diff_lt128(q1, q0, beta));
        let two = _mm_set1_epi16(2);
        let np0 = _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(_mm_slli_epi16(p1, 1), p0), _mm_add_epi16(q1, two)), 2);
        let nq0 = _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(_mm_slli_epi16(q1, 1), q0), _mm_add_epi16(p1, two)), 2);
        v[1] = _mm_blendv_epi8(p0, np0, mask);
        v[2] = _mm_blendv_epi8(q0, nq0, mask);
    }
}

/// tC0 per lane for eight chroma lines (two per segment).
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn tc0_chroma(tc0: &[i8; 4]) -> __m128i {
    unsafe {
        let t = |k: usize| tc0[k] as i16;
        _mm_setr_epi16(t(0), t(0), t(1), t(1), t(2), t(2), t(3), t(3))
    }
}

/// Eight bytes -> eight i16 lanes.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn load8(p: *const u8) -> __m128i {
    unsafe { _mm_cvtepu8_epi16(_mm_loadl_epi64(p as *const __m128i)) }
}

/// Eight i16 lanes -> eight bytes.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn store8(p: *mut u8, v: __m128i) {
    unsafe { _mm_storel_epi64(p as *mut __m128i, _mm_packus_epi16(v, v)) }
}

/// Load 8 rows x 4 bytes (p1 p0 q0 q1) around a vertical chroma edge as four
/// column vectors.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn load_transposed_8x4(data: *const u8, stride: usize) -> ChromaLines {
    unsafe {
        let mut r = [_mm_setzero_si128(); 8];
        for i in 0..8 {
            r[i] = _mm_cvtsi32_si128(std::ptr::read_unaligned(data.add(i * stride).sub(2) as *const i32));
        }
        let a0 = _mm_unpacklo_epi8(r[0], r[1]); // p1r0 p1r1 p0r0 p0r1 q0r0 q0r1 q1r0 q1r1
        let a1 = _mm_unpacklo_epi8(r[2], r[3]);
        let a2 = _mm_unpacklo_epi8(r[4], r[5]);
        let a3 = _mm_unpacklo_epi8(r[6], r[7]);
        let b0 = _mm_unpacklo_epi16(a0, a1); // p1 r0..3, p0 r0..3, q0 r0..3, q1 r0..3
        let b1 = _mm_unpacklo_epi16(a2, a3); // rows 4..7
        let c0 = _mm_unpacklo_epi32(b0, b1); // p1 r0..7 | p0 r0..7
        let c1 = _mm_unpackhi_epi32(b0, b1); // q0 r0..7 | q1 r0..7
        [
            _mm_cvtepu8_epi16(c0),
            _mm_cvtepu8_epi16(_mm_srli_si128(c0, 8)),
            _mm_cvtepu8_epi16(c1),
            _mm_cvtepu8_epi16(_mm_srli_si128(c1, 8)),
        ]
    }
}

/// Store the p0 / q0 columns of eight rows back (p1, q1 are unchanged).
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn store_transposed_8x4(data: *mut u8, stride: usize, v: &ChromaLines) {
    unsafe {
        // Interleave p0 and q0 bytes per row and store two bytes at x-1, x.
        let p0 = _mm_packus_epi16(v[1], v[1]);
        let q0 = _mm_packus_epi16(v[2], v[2]);
        let pq = _mm_unpacklo_epi8(p0, q0); // p0r0 q0r0 p0r1 q0r1 ...
        let mut t = [0u8; 16];
        _mm_storeu_si128(t.as_mut_ptr() as *mut __m128i, pq);
        for i in 0..8 {
            let d = data.add(i * stride).sub(1);
            *d = t[2 * i];
            *d.add(1) = t[2 * i + 1];
        }
    }
}

fn deblock_chroma_v_avx2(data: &mut [u8], off: usize, stride: usize, alpha: i32, beta: i32, tc0: &[i8; 4]) {
    if tc0.iter().all(|&t| t < 0) {
        return;
    }
    assert!(off >= 2 && off + 7 * stride + 2 <= data.len());
    unsafe { deblock_chroma_v_impl(data.as_mut_ptr().add(off), stride, alpha, beta, tc0) }
}

#[target_feature(enable = "avx2")]
unsafe fn deblock_chroma_v_impl(data: *mut u8, stride: usize, alpha: i32, beta: i32, tc0: &[i8; 4]) {
    unsafe {
        let mut v = load_transposed_8x4(data, stride);
        chroma_filter_normal(&mut v, alpha, beta, tc0_chroma(tc0));
        store_transposed_8x4(data, stride, &v);
    }
}

fn deblock_chroma_v_intra_avx2(data: &mut [u8], off: usize, stride: usize, alpha: i32, beta: i32) {
    assert!(off >= 2 && off + 7 * stride + 2 <= data.len());
    unsafe { deblock_chroma_v_intra_impl(data.as_mut_ptr().add(off), stride, alpha, beta) }
}

#[target_feature(enable = "avx2")]
unsafe fn deblock_chroma_v_intra_impl(data: *mut u8, stride: usize, alpha: i32, beta: i32) {
    unsafe {
        let mut v = load_transposed_8x4(data, stride);
        chroma_filter_intra(&mut v, alpha, beta);
        store_transposed_8x4(data, stride, &v);
    }
}

fn deblock_chroma_h_avx2(data: &mut [u8], off: usize, stride: usize, alpha: i32, beta: i32, tc0: &[i8; 4]) {
    if tc0.iter().all(|&t| t < 0) {
        return;
    }
    assert!(off >= 2 * stride && off + stride + 8 <= data.len());
    unsafe { deblock_chroma_h_impl(data.as_mut_ptr().add(off), stride, alpha, beta, tc0) }
}

#[target_feature(enable = "avx2")]
unsafe fn deblock_chroma_h_impl(data: *mut u8, stride: usize, alpha: i32, beta: i32, tc0: &[i8; 4]) {
    unsafe {
        let mut v: ChromaLines = [load8(data.sub(2 * stride)), load8(data.sub(stride)), load8(data), load8(data.add(stride))];
        chroma_filter_normal(&mut v, alpha, beta, tc0_chroma(tc0));
        store8(data.sub(stride), v[1]);
        store8(data, v[2]);
    }
}

fn deblock_chroma_h_intra_avx2(data: &mut [u8], off: usize, stride: usize, alpha: i32, beta: i32) {
    assert!(off >= 2 * stride && off + stride + 8 <= data.len());
    unsafe { deblock_chroma_h_intra_impl(data.as_mut_ptr().add(off), stride, alpha, beta) }
}

#[target_feature(enable = "avx2")]
unsafe fn deblock_chroma_h_intra_impl(data: *mut u8, stride: usize, alpha: i32, beta: i32) {
    unsafe {
        let mut v: ChromaLines = [load8(data.sub(2 * stride)), load8(data.sub(stride)), load8(data), load8(data.add(stride))];
        chroma_filter_intra(&mut v, alpha, beta);
        store8(data.sub(stride), v[1]);
        store8(data, v[2]);
    }
}

// ----------------------------------------------------------------------
// Inverse transforms
// ----------------------------------------------------------------------
//
// Rows first, then columns, as the standard orders them (the `>> 1` inside
// each pass makes the order matter). A row register holds one row's samples,
// so the row pass runs on the transposed block and the column pass on the
// block transposed back — two small transposes around the arithmetic.

/// Add `(v + 32) >> 6` rows to `dst`, clipping, `n` = 4 or 8 samples per row.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn add_row(dst: *mut u8, v: __m128i, n: usize) {
    unsafe {
        let r = _mm_srai_epi16(_mm_add_epi16(v, _mm_set1_epi16(32)), 6);
        if n == 4 {
            let p = _mm_cvtepu8_epi16(_mm_cvtsi32_si128(std::ptr::read_unaligned(dst as *const i32)));
            let s = _mm_packus_epi16(_mm_add_epi16(p, r), _mm_setzero_si128());
            std::ptr::write_unaligned(dst as *mut i32, _mm_cvtsi128_si32(s));
        } else {
            let p = _mm_cvtepu8_epi16(_mm_loadl_epi64(dst as *const __m128i));
            let s = _mm_packus_epi16(_mm_add_epi16(p, r), _mm_setzero_si128());
            _mm_storel_epi64(dst as *mut __m128i, s);
        }
    }
}

fn idct4_add_avx2(dst: &mut [u8], stride: usize, coeffs: &[i16; 16]) {
    assert!(3 * stride + 4 <= dst.len());
    unsafe { idct4_add_impl(dst.as_mut_ptr(), stride, coeffs) }
}

#[target_feature(enable = "avx2")]
unsafe fn idct4_add_impl(dst: *mut u8, stride: usize, c: &[i16; 16]) {
    unsafe {
        let r0 = _mm_loadl_epi64(c.as_ptr() as *const __m128i);
        let r1 = _mm_loadl_epi64(c.as_ptr().add(4) as *const __m128i);
        let r2 = _mm_loadl_epi64(c.as_ptr().add(8) as *const __m128i);
        let r3 = _mm_loadl_epi64(c.as_ptr().add(12) as *const __m128i);
        // Columns of the block, four lanes each.
        let t0 = _mm_unpacklo_epi16(r0, r1);
        let t1 = _mm_unpacklo_epi16(r2, r3);
        let c01 = _mm_unpacklo_epi32(t0, t1);
        let c23 = _mm_unpackhi_epi32(t0, t1);
        let (c0, c1, c2, c3) = (c01, _mm_srli_si128(c01, 8), c23, _mm_srli_si128(c23, 8));
        // Row pass.
        let e0 = _mm_add_epi16(c0, c2);
        let e1 = _mm_sub_epi16(c0, c2);
        let e2 = _mm_sub_epi16(_mm_srai_epi16(c1, 1), c3);
        let e3 = _mm_add_epi16(c1, _mm_srai_epi16(c3, 1));
        let f0 = _mm_add_epi16(e0, e3);
        let f1 = _mm_add_epi16(e1, e2);
        let f2 = _mm_sub_epi16(e1, e2);
        let f3 = _mm_sub_epi16(e0, e3);
        // Back to rows.
        let u0 = _mm_unpacklo_epi16(f0, f1);
        let u1 = _mm_unpacklo_epi16(f2, f3);
        let r01 = _mm_unpacklo_epi32(u0, u1);
        let r23 = _mm_unpackhi_epi32(u0, u1);
        let (row0, row1, row2, row3) = (r01, _mm_srli_si128(r01, 8), r23, _mm_srli_si128(r23, 8));
        // Column pass.
        let g0 = _mm_add_epi16(row0, row2);
        let g1 = _mm_sub_epi16(row0, row2);
        let g2 = _mm_sub_epi16(_mm_srai_epi16(row1, 1), row3);
        let g3 = _mm_add_epi16(row1, _mm_srai_epi16(row3, 1));
        add_row(dst, _mm_add_epi16(g0, g3), 4);
        add_row(dst.add(stride), _mm_add_epi16(g1, g2), 4);
        add_row(dst.add(2 * stride), _mm_sub_epi16(g1, g2), 4);
        add_row(dst.add(3 * stride), _mm_sub_epi16(g0, g3), 4);
    }
}

/// Transpose eight 8-lane i16 rows.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn transpose8(r: &mut [__m128i; 8]) {
    unsafe {
        let a0 = _mm_unpacklo_epi16(r[0], r[1]);
        let a1 = _mm_unpackhi_epi16(r[0], r[1]);
        let a2 = _mm_unpacklo_epi16(r[2], r[3]);
        let a3 = _mm_unpackhi_epi16(r[2], r[3]);
        let a4 = _mm_unpacklo_epi16(r[4], r[5]);
        let a5 = _mm_unpackhi_epi16(r[4], r[5]);
        let a6 = _mm_unpacklo_epi16(r[6], r[7]);
        let a7 = _mm_unpackhi_epi16(r[6], r[7]);
        let b0 = _mm_unpacklo_epi32(a0, a2);
        let b1 = _mm_unpackhi_epi32(a0, a2);
        let b2 = _mm_unpacklo_epi32(a1, a3);
        let b3 = _mm_unpackhi_epi32(a1, a3);
        let b4 = _mm_unpacklo_epi32(a4, a6);
        let b5 = _mm_unpackhi_epi32(a4, a6);
        let b6 = _mm_unpacklo_epi32(a5, a7);
        let b7 = _mm_unpackhi_epi32(a5, a7);
        r[0] = _mm_unpacklo_epi64(b0, b4);
        r[1] = _mm_unpackhi_epi64(b0, b4);
        r[2] = _mm_unpacklo_epi64(b1, b5);
        r[3] = _mm_unpackhi_epi64(b1, b5);
        r[4] = _mm_unpacklo_epi64(b2, b6);
        r[5] = _mm_unpackhi_epi64(b2, b6);
        r[6] = _mm_unpacklo_epi64(b3, b7);
        r[7] = _mm_unpackhi_epi64(b3, b7);
    }
}

/// One 8-point pass (8.5.13.2) across eight registers.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn idct8_pass(d: &[__m128i; 8]) -> [__m128i; 8] {
    unsafe {
        let add = |a, b| _mm_add_epi16(a, b);
        let sub = |a, b| _mm_sub_epi16(a, b);
        let sh1 = |a| _mm_srai_epi16(a, 1);
        let sh2 = |a| _mm_srai_epi16(a, 2);
        let a0 = add(d[0], d[4]);
        let a4 = sub(d[0], d[4]);
        let a2 = sub(sh1(d[2]), d[6]);
        let a6 = add(d[2], sh1(d[6]));
        let b0 = add(a0, a6);
        let b2 = add(a4, a2);
        let b4 = sub(a4, a2);
        let b6 = sub(a0, a6);
        // a1 = -d3 + d5 - d7 - (d7 >> 1)
        let a1 = sub(sub(sub(d[5], d[3]), d[7]), sh1(d[7]));
        // a3 = d1 + d7 - d3 - (d3 >> 1)
        let a3 = sub(sub(add(d[1], d[7]), d[3]), sh1(d[3]));
        // a5 = -d1 + d7 + d5 + (d5 >> 1)
        let a5 = add(add(sub(d[7], d[1]), d[5]), sh1(d[5]));
        // a7 = d3 + d5 + d1 + (d1 >> 1)
        let a7 = add(add(add(d[3], d[5]), d[1]), sh1(d[1]));
        let b1 = add(a1, sh2(a7));
        let b7 = sub(a7, sh2(a1));
        let b3 = add(a3, sh2(a5));
        let b5 = sub(sh2(a3), a5);
        [add(b0, b7), add(b2, b5), add(b4, b3), add(b6, b1), sub(b6, b1), sub(b4, b3), sub(b2, b5), sub(b0, b7)]
    }
}

fn idct8_add_avx2(dst: &mut [u8], stride: usize, coeffs: &[i16; 64]) {
    assert!(7 * stride + 8 <= dst.len());
    unsafe { idct8_add_impl(dst.as_mut_ptr(), stride, coeffs) }
}

#[target_feature(enable = "avx2")]
unsafe fn idct8_add_impl(dst: *mut u8, stride: usize, c: &[i16; 64]) {
    unsafe {
        let mut r = [_mm_setzero_si128(); 8];
        for i in 0..8 {
            r[i] = _mm_loadu_si128(c.as_ptr().add(i * 8) as *const __m128i);
        }
        transpose8(&mut r);
        let mut f = idct8_pass(&r);
        transpose8(&mut f);
        let h = idct8_pass(&f);
        for i in 0..8 {
            add_row(dst.add(i * stride), h[i], 8);
        }
    }
}

fn idct4_dc_add_avx2(dst: &mut [u8], stride: usize, dc: i32) {
    assert!(3 * stride + 4 <= dst.len());
    unsafe { dc_add_impl(dst.as_mut_ptr(), stride, dc, 4) }
}

fn idct8_dc_add_avx2(dst: &mut [u8], stride: usize, dc: i32) {
    assert!(7 * stride + 8 <= dst.len());
    unsafe { dc_add_impl(dst.as_mut_ptr(), stride, dc, 8) }
}

#[target_feature(enable = "avx2")]
unsafe fn dc_add_impl(dst: *mut u8, stride: usize, dc: i32, n: usize) {
    unsafe {
        let v = _mm_set1_epi16(dc as i16);
        for i in 0..n {
            add_row(dst.add(i * stride), v, n);
        }
    }
}

/// Dequantise sixteen levels (two vectors of eight i32) to one vector of
/// sixteen i16, `qp`-dependent shift with or without rounding.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn dequant16(levels: *const i32, scale: *const i32, up: bool, sh: i32, round: i32) -> __m256i {
    unsafe {
        let l0 = _mm256_loadu_si256(levels as *const __m256i);
        let l1 = _mm256_loadu_si256(levels.add(8) as *const __m256i);
        let s0 = _mm256_loadu_si256(scale as *const __m256i);
        let s1 = _mm256_loadu_si256(scale.add(8) as *const __m256i);
        let mut v0 = _mm256_mullo_epi32(l0, s0);
        let mut v1 = _mm256_mullo_epi32(l1, s1);
        let cnt = _mm_cvtsi32_si128(sh);
        if up {
            v0 = _mm256_sll_epi32(v0, cnt);
            v1 = _mm256_sll_epi32(v1, cnt);
        } else {
            let r = _mm256_set1_epi32(round);
            v0 = _mm256_sra_epi32(_mm256_add_epi32(v0, r), cnt);
            v1 = _mm256_sra_epi32(_mm256_add_epi32(v1, r), cnt);
        }
        // packs keeps lane order per 128-bit half: [v0.lo v1.lo | v0.hi v1.hi]
        // -> permute to [v0.lo v0.hi v1.lo v1.hi].
        _mm256_permute4x64_epi64(_mm256_packs_epi32(v0, v1), 0b11_01_10_00)
    }
}

fn residual4_avx2(dst: &mut [u8], stride: usize, levels: &[i32; 16], scale: &[i32; 16], qp: i32, dc: i32) {
    assert!(3 * stride + 4 <= dst.len());
    unsafe { residual4_impl(dst.as_mut_ptr(), stride, levels, scale, qp, dc) }
}

#[target_feature(enable = "avx2")]
unsafe fn residual4_impl(dst: *mut u8, stride: usize, levels: &[i32; 16], scale: &[i32; 16], qp: i32, dc: i32) {
    unsafe {
        let q6 = qp / 6;
        let mut c = if qp >= 24 {
            dequant16(levels.as_ptr(), scale.as_ptr(), true, q6 - 4, 0)
        } else {
            dequant16(levels.as_ptr(), scale.as_ptr(), false, 4 - q6, 1 << (3 - q6))
        };
        if dc != NO_DC {
            c = _mm256_insert_epi16(c, dc as i16, 0);
        }
        // Any AC nonzero? Zero lane 0, compare the rest with zero.
        let ac = _mm256_andnot_si256(_mm256_setr_epi16(-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), c);
        if _mm256_testz_si256(ac, ac) != 0 {
            let d = _mm256_extract_epi16(c, 0) as i16 as i32;
            if d != 0 {
                dc_add_impl(dst, stride, d, 4);
            }
            return;
        }
        let mut coeffs = [0i16; 16];
        _mm256_storeu_si256(coeffs.as_mut_ptr() as *mut __m256i, c);
        idct4_add_impl(dst, stride, &coeffs);
    }
}

fn residual8_avx2(dst: &mut [u8], stride: usize, levels: &[i32; 64], scale: &[i32; 64], qp: i32) {
    assert!(7 * stride + 8 <= dst.len());
    unsafe { residual8_impl(dst.as_mut_ptr(), stride, levels, scale, qp) }
}

#[target_feature(enable = "avx2")]
unsafe fn residual8_impl(dst: *mut u8, stride: usize, levels: &[i32; 64], scale: &[i32; 64], qp: i32) {
    unsafe {
        let q6 = qp / 6;
        let (up, sh, round) = if qp >= 36 { (true, q6 - 6, 0) } else { (false, 6 - q6, 1 << (5 - q6)) };
        let mut coeffs = [0i16; 64];
        let mut ac = _mm256_setzero_si256();
        for k in 0..4 {
            let c = dequant16(levels.as_ptr().add(16 * k), scale.as_ptr().add(16 * k), up, sh, round);
            _mm256_storeu_si256(coeffs.as_mut_ptr().add(16 * k) as *mut __m256i, c);
            let masked = if k == 0 { _mm256_andnot_si256(_mm256_setr_epi16(-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), c) } else { c };
            ac = _mm256_or_si256(ac, masked);
        }
        if _mm256_testz_si256(ac, ac) != 0 {
            let d = coeffs[0] as i32;
            if d != 0 {
                dc_add_impl(dst, stride, d, 8);
            }
            return;
        }
        idct8_add_impl(dst, stride, &coeffs);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: &mut u64) -> u32 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (*seed >> 33) as u32
    }

    fn avx2() -> Option<H264Dsp> {
        if !std::is_x86_feature_detected!("avx2") {
            return None;
        }
        let mut d = H264Dsp::SCALAR;
        install(&mut d);
        Some(d)
    }

    #[test]
    fn qpel_matches_scalar() {
        let Some(d) = avx2() else { return };
        let s = H264Dsp::SCALAR;
        let mut seed = 5u64;
        let stride = 64;
        let src: Vec<u8> = (0..stride * 64).map(|_| lcg(&mut seed) as u8).collect();
        for &(w, h) in &[(4usize, 4usize), (4, 8), (8, 4), (8, 8), (8, 16), (16, 8), (16, 16)] {
            // Only the w x h block of the stride-16 scratch is compared: the
            // SIMD kernels may write the rest of each row.
            let block = |v: &[u8]| -> Vec<u8> { (0..h).flat_map(|y| v[y * PRED_STRIDE..y * PRED_STRIDE + w].to_vec()).collect() };
            for pos in 0..16 {
                let mut a = vec![0u8; 16 * PRED_STRIDE];
                let mut b = vec![0u8; 16 * PRED_STRIDE];
                (s.qpel[pos])(&mut a, &src[stride * 3 + 3..], stride, w, h);
                (d.qpel[pos])(&mut b, &src[stride * 3 + 3..], stride, w, h);
                assert_eq!(block(&a), block(&b), "qpel pos={pos} {w}x{h}");
            }
            for xf in 0..8 {
                for yf in 0..8 {
                    let (cw, ch) = (w / 2, h / 2);
                    let mut a = vec![0u8; 16 * PRED_STRIDE];
                    let mut b = vec![0u8; 16 * PRED_STRIDE];
                    (s.chroma)(&mut a, &src[stride * 5 + 5..], stride, cw, ch, xf, yf);
                    (d.chroma)(&mut b, &src[stride * 5 + 5..], stride, cw, ch, xf, yf);
                    let cb = |v: &[u8]| -> Vec<u8> { (0..ch).flat_map(|y| v[y * PRED_STRIDE..y * PRED_STRIDE + cw].to_vec()).collect() };
                    assert_eq!(cb(&a), cb(&b), "chroma {xf},{yf} {cw}x{ch}");
                }
            }
            let a: Vec<u8> = (0..16 * PRED_STRIDE).map(|_| lcg(&mut seed) as u8).collect();
            let b: Vec<u8> = (0..16 * PRED_STRIDE).map(|_| lcg(&mut seed) as u8).collect();
            let ds = w + 3;
            let mut d1 = vec![0u8; ds * h];
            let mut d2 = vec![0u8; ds * h];
            (s.avg)(&mut d1, ds, &a, &b, w, h);
            (d.avg)(&mut d2, ds, &a, &b, w, h);
            assert_eq!(d1, d2, "avg {w}x{h}");
            for &(lwd, wt, o) in &[(6, 64, 0), (0, 1, 3), (5, -20, -7), (7, 127, 127), (2, 33, -128)] {
                (s.weighted_uni)(&mut d1, ds, &a, w, h, lwd, wt, o);
                (d.weighted_uni)(&mut d2, ds, &a, w, h, lwd, wt, o);
                assert_eq!(d1, d2, "wuni {w}x{h} {lwd} {wt} {o}");
                (s.weighted_bi)(&mut d1, ds, &a, &b, w, h, lwd, wt, 64 - wt, o, -o);
                (d.weighted_bi)(&mut d2, ds, &a, &b, w, h, lwd, wt, 64 - wt, o, -o);
                assert_eq!(d1, d2, "wbi {w}x{h} {lwd} {wt} {o}");
            }
        }
    }

    #[test]
    fn deblocking_matches_scalar() {
        let Some(d) = avx2() else { return };
        let s = H264Dsp::SCALAR;
        let mut seed = 11u64;
        let stride = 48;
        for trial in 0..400 {
            // Smooth-ish content so the alpha/beta tests pass often.
            let base = lcg(&mut seed) % 256;
            let spread = 1 + lcg(&mut seed) % 64;
            let plane: Vec<u8> = (0..stride * 40).map(|_| (base + lcg(&mut seed) % spread).min(255) as u8).collect();
            let alpha = (lcg(&mut seed) % 256) as i32;
            let beta = (lcg(&mut seed) % 20) as i32;
            let mut tc0 = [0i8; 4];
            for t in tc0.iter_mut() {
                *t = (lcg(&mut seed) % 6) as i8 - 1;
            }
            let off = 8 * stride + 8;
            let mut a = plane.clone();
            let mut b = plane.clone();
            match trial % 8 {
                0 => {
                    (s.deblock_luma_v)(&mut a, off, stride, alpha, beta, &tc0);
                    (d.deblock_luma_v)(&mut b, off, stride, alpha, beta, &tc0);
                }
                1 => {
                    (s.deblock_luma_h)(&mut a, off, stride, alpha, beta, &tc0);
                    (d.deblock_luma_h)(&mut b, off, stride, alpha, beta, &tc0);
                }
                2 => {
                    (s.deblock_luma_v_intra)(&mut a, off, stride, alpha, beta);
                    (d.deblock_luma_v_intra)(&mut b, off, stride, alpha, beta);
                }
                3 => {
                    (s.deblock_luma_h_intra)(&mut a, off, stride, alpha, beta);
                    (d.deblock_luma_h_intra)(&mut b, off, stride, alpha, beta);
                }
                4 => {
                    (s.deblock_chroma_v)(&mut a, off, stride, alpha, beta, &tc0);
                    (d.deblock_chroma_v)(&mut b, off, stride, alpha, beta, &tc0);
                }
                5 => {
                    (s.deblock_chroma_h)(&mut a, off, stride, alpha, beta, &tc0);
                    (d.deblock_chroma_h)(&mut b, off, stride, alpha, beta, &tc0);
                }
                6 => {
                    (s.deblock_chroma_v_intra)(&mut a, off, stride, alpha, beta);
                    (d.deblock_chroma_v_intra)(&mut b, off, stride, alpha, beta);
                }
                _ => {
                    (s.deblock_chroma_h_intra)(&mut a, off, stride, alpha, beta);
                    (d.deblock_chroma_h_intra)(&mut b, off, stride, alpha, beta);
                }
            }
            assert_eq!(a, b, "deblock kind {} trial {trial} alpha {alpha} beta {beta} tc0 {tc0:?}", trial % 8);
        }
    }

    #[test]
    fn transforms_match_scalar() {
        let Some(d) = avx2() else { return };
        let s = H264Dsp::SCALAR;
        let mut seed = 17u64;
        let stride = 24;
        for trial in 0..500 {
            let base: Vec<u8> = (0..stride * 8).map(|_| lcg(&mut seed) as u8).collect();
            // Within the standard's 16-bit intermediate range (a conforming
            // stream never exceeds it; the SIMD kernels work in i16 like the
            // scalar reference's callers rely on).
            let range = if trial % 3 == 0 { 2000 } else { 300 };
            let range8 = if trial % 3 == 0 { 500 } else { 100 };
            let mut c4 = [0i16; 16];
            let mut c8 = [0i16; 64];
            for v in c4.iter_mut() {
                *v = (lcg(&mut seed) % (2 * range) as u32) as i16 - range;
            }
            for v in c8.iter_mut() {
                *v = (lcg(&mut seed) % (2 * range8) as u32) as i16 - range8;
            }
            let dc = (lcg(&mut seed) % 8000) as i32 - 4000;
            let mut a = base.clone();
            let mut b = base.clone();
            (s.idct4_add)(&mut a, stride, &c4);
            (d.idct4_add)(&mut b, stride, &c4);
            assert_eq!(a, b, "idct4 trial {trial}");
            let mut a = base.clone();
            let mut b = base.clone();
            (s.idct8_add)(&mut a, stride, &c8);
            (d.idct8_add)(&mut b, stride, &c8);
            assert_eq!(a, b, "idct8 trial {trial}");
            let mut a = base.clone();
            let mut b = base.clone();
            (s.idct4_dc_add)(&mut a, stride, dc);
            (d.idct4_dc_add)(&mut b, stride, dc);
            assert_eq!(a, b, "dc4 trial {trial}");
            let mut a = base.clone();
            let mut b = base.clone();
            (s.idct8_dc_add)(&mut a, stride, dc);
            (d.idct8_dc_add)(&mut b, stride, dc);
            assert_eq!(a, b, "dc8 trial {trial}");
            // Fused dequantisation: levels, a scale table, a QP.
            let qp = (lcg(&mut seed) % 52) as i32;
            let mut lv4 = [0i32; 16];
            let mut lv8 = [0i32; 64];
            let mut sc4 = [0i32; 16];
            let mut sc8 = [0i32; 64];
            let dc_only = trial % 4 == 1;
            // Levels sized so the dequantised values stay in the range a
            // conforming stream keeps them in (an encoder quantises harder at
            // high QP): |level * scale << shift| well inside 16 bits.
            let lmax = ((2000i32 >> (qp / 6 - 4).max(0)) / 480).max(1) as u32;
            let lmax8 = ((800i32 >> (qp / 6 - 6).max(0)) / 480).max(1) as u32;
            for i in 0..16 {
                lv4[i] = if dc_only && i != 0 { 0 } else { (lcg(&mut seed) % (2 * lmax + 1)) as i32 - lmax as i32 };
                sc4[i] = 16 * (10 + (lcg(&mut seed) % 20) as i32);
            }
            for i in 0..64 {
                lv8[i] = if dc_only && i != 0 { 0 } else { (lcg(&mut seed) % (2 * lmax8 + 1)) as i32 - lmax8 as i32 };
                sc8[i] = 16 * (10 + (lcg(&mut seed) % 20) as i32);
            }
            let dcv = if trial % 2 == 0 { NO_DC } else { (lcg(&mut seed) % 4001) as i32 - 2000 };
            let mut a = base.clone();
            let mut b = base.clone();
            (s.residual4)(&mut a, stride, &lv4, &sc4, qp, dcv);
            (d.residual4)(&mut b, stride, &lv4, &sc4, qp, dcv);
            assert_eq!(a, b, "residual4 trial {trial} qp {qp}");
            let mut a = base.clone();
            let mut b = base.clone();
            (s.residual8)(&mut a, stride, &lv8, &sc8, qp);
            (d.residual8)(&mut b, stride, &lv8, &sc8, qp);
            assert_eq!(a, b, "residual8 trial {trial} qp {qp}");
        }
    }
}
