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

use super::h264::H264Dsp;

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
            let d = dst.as_mut_ptr().add(y * w);
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
            store_u8_n(d, v, w);
        }
    }
}

fn chroma_avx2(dst: &mut [u8], src: &[u8], stride: usize, w: usize, h: usize, xf: i32, yf: i32) {
    if src.len() < h * stride + 17 {
        return (H264Dsp::SCALAR.chroma)(dst, src, stride, w, h, xf, yf);
    }
    unsafe { chroma_impl(dst, src, stride, w, h, xf, yf) }
}

#[target_feature(enable = "avx2")]
unsafe fn chroma_impl(dst: &mut [u8], src: &[u8], stride: usize, w: usize, h: usize, xf: i32, yf: i32) {
    unsafe {
        let a = _mm256_set1_epi16(((8 - xf) * (8 - yf)) as i16);
        let b = _mm256_set1_epi16((xf * (8 - yf)) as i16);
        let c = _mm256_set1_epi16(((8 - xf) * yf) as i16);
        let d = _mm256_set1_epi16((xf * yf) as i16);
        let round = _mm256_set1_epi16(32);
        let s = src.as_ptr();
        for y in 0..h {
            let r0 = s.add(y * stride);
            let r1 = s.add((y + 1) * stride);
            let v = _mm256_add_epi16(
                _mm256_add_epi16(_mm256_mullo_epi16(load16(r0), a), _mm256_mullo_epi16(load16(r0.add(1)), b)),
                _mm256_add_epi16(_mm256_mullo_epi16(load16(r1), c), _mm256_mullo_epi16(load16(r1.add(1)), d)),
            );
            let v = _mm256_srli_epi16(_mm256_add_epi16(v, round), 6);
            let p = _mm256_packus_epi16(v, v);
            let p = _mm256_permute4x64_epi64(p, 0b11_01_10_00);
            store_u8_n(dst.as_mut_ptr().add(y * w), _mm256_castsi256_si128(p), w);
        }
    }
}

fn copy_avx2(dst: &mut [u8], stride: usize, src: &[u8], w: usize, h: usize) {
    for y in 0..h {
        dst[y * stride..y * stride + w].copy_from_slice(&src[y * w..y * w + w]);
    }
}

fn avg_avx2(dst: &mut [u8], stride: usize, a: &[u8], b: &[u8], w: usize, h: usize) {
    unsafe { avg_impl(dst, stride, a, b, w, h) }
}

#[target_feature(enable = "avx2")]
unsafe fn avg_impl(dst: &mut [u8], stride: usize, a: &[u8], b: &[u8], w: usize, h: usize) {
    unsafe {
        // Rows are packed contiguously (w * h): treat as one run when w >= 16
        // rows would still need per-row stores into the strided destination.
        for y in 0..h {
            let mut t = [0u8; 16];
            std::ptr::copy_nonoverlapping(a.as_ptr().add(y * w), t.as_mut_ptr(), w);
            let va = _mm_loadu_si128(t.as_ptr() as *const __m128i);
            std::ptr::copy_nonoverlapping(b.as_ptr().add(y * w), t.as_mut_ptr(), w);
            let vb = _mm_loadu_si128(t.as_ptr() as *const __m128i);
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
            let mut t = [0u8; 16];
            std::ptr::copy_nonoverlapping(src.as_ptr().add(y * w), t.as_mut_ptr(), w);
            let s = load16(t.as_ptr());
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
            let mut ta = [0u8; 16];
            let mut tb = [0u8; 16];
            std::ptr::copy_nonoverlapping(a.as_ptr().add(y * w), ta.as_mut_ptr(), w);
            std::ptr::copy_nonoverlapping(b.as_ptr().add(y * w), tb.as_mut_ptr(), w);
            let va = _mm_loadu_si128(ta.as_ptr() as *const __m128i);
            let vb = _mm_loadu_si128(tb.as_ptr() as *const __m128i);
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
            for pos in 0..16 {
                let mut a = vec![0u8; w * h];
                let mut b = vec![0u8; w * h];
                (s.qpel[pos])(&mut a, &src[stride * 3 + 3..], stride, w, h);
                (d.qpel[pos])(&mut b, &src[stride * 3 + 3..], stride, w, h);
                assert_eq!(a, b, "qpel pos={pos} {w}x{h}");
            }
            for xf in 0..8 {
                for yf in 0..8 {
                    let (cw, ch) = (w / 2, h / 2);
                    let mut a = vec![0u8; cw * ch];
                    let mut b = vec![0u8; cw * ch];
                    (s.chroma)(&mut a, &src[stride * 5 + 5..], stride, cw, ch, xf, yf);
                    (d.chroma)(&mut b, &src[stride * 5 + 5..], stride, cw, ch, xf, yf);
                    assert_eq!(a, b, "chroma {xf},{yf} {cw}x{ch}");
                }
            }
            let a: Vec<u8> = (0..w * h).map(|_| lcg(&mut seed) as u8).collect();
            let b: Vec<u8> = (0..w * h).map(|_| lcg(&mut seed) as u8).collect();
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
}
