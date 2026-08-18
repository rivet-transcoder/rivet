//! NEON versions of the H.265 kernels (AArch64).
//!
//! Eight 16-bit lanes per vector. Interpolation multiplies each tap into
//! 32-bit accumulators (`vmlal`), narrows with saturation; the inverse DCT is
//! the matrix product across columns then along rows with `vmlal_n` per
//! coefficient, restricted to the nonzero region. Every kernel is checked
//! bit-exact against the scalar reference by the tests (run on AArch64 in CI).

#![cfg(target_arch = "aarch64")]

use std::arch::aarch64::*;

use super::hevc::HevcDsp;
use crate::hevc::tables::{EPEL_FILTERS, QPEL_FILTERS, TRANSFORM32};

/// Replace the scalar entries of `d` with the NEON kernels.
pub fn install(d: &mut HevcDsp) {
    d.idct = [idct_neon::<4>, idct_neon::<8>, idct_neon::<16>, idct_neon::<32>];
    d.add_residual = add_residual_neon;
    d.qpel_copy = copy_neon;
    d.qpel_h = qpel_h_neon;
    d.qpel_v = qpel_v_neon;
    d.qpel_v2 = qpel_v2_neon;
    d.epel_copy = copy_neon;
    d.epel_h = epel_h_neon;
    d.epel_v = epel_v_neon;
    d.epel_v2 = epel_v2_neon;
    d.uni = uni_neon;
    d.bi = bi_neon;
    d.weighted_uni = weighted_uni_neon;
    d.weighted_bi = weighted_bi_neon;
    d.sao_band = sao_band_neon;
    d.sao_edge = sao_edge_neon;
    d.deblock_luma_v = deblock_luma_v_neon;
    d.deblock_luma_h = deblock_luma_h_neon;
    d.deblock_chroma_v = deblock_chroma_v_neon;
    d.deblock_chroma_h = deblock_chroma_h_neon;
}

// ----------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------

/// Store the first `n` (≤ 8) lanes of `v`.
#[inline(always)]
unsafe fn store_n(dst: *mut i16, v: int16x8_t, n: usize) {
    unsafe {
        if n >= 8 {
            vst1q_s16(dst, v);
        } else {
            let mut t = [0i16; 8];
            vst1q_s16(t.as_mut_ptr(), v);
            std::ptr::copy_nonoverlapping(t.as_ptr(), dst, n);
        }
    }
}

#[inline(always)]
unsafe fn store_n_u16(dst: *mut u16, v: uint16x8_t, n: usize) {
    unsafe {
        if n >= 8 {
            vst1q_u16(dst, v);
        } else {
            let mut t = [0u16; 8];
            vst1q_u16(t.as_mut_ptr(), v);
            std::ptr::copy_nonoverlapping(t.as_ptr(), dst, n);
        }
    }
}

/// Load 8 lanes, or `avail` zero-padded.
#[inline(always)]
unsafe fn load_n(src: *const i16, avail: usize) -> int16x8_t {
    unsafe {
        if avail >= 8 {
            vld1q_s16(src)
        } else {
            let mut t = [0i16; 8];
            std::ptr::copy_nonoverlapping(src, t.as_mut_ptr(), avail);
            vld1q_s16(t.as_ptr())
        }
    }
}

#[inline(always)]
unsafe fn load_n_u16(src: *const u16, avail: usize) -> uint16x8_t {
    unsafe {
        if avail >= 8 {
            vld1q_u16(src)
        } else {
            let mut t = [0u16; 8];
            std::ptr::copy_nonoverlapping(src, t.as_mut_ptr(), avail);
            vld1q_u16(t.as_ptr())
        }
    }
}

/// Whether 8-lane loads for `w` columns over `rows` rows (+`extra` along)
/// stay inside `len`.
#[inline(always)]
fn fits(len: usize, stride: usize, rows: usize, w: usize, extra: usize) -> bool {
    let last_x = if w == 0 { 0 } else { (w - 1) / 8 * 8 };
    (rows - 1) * stride + last_x + extra + 8 <= len
}

/// `(acc >> shift)` for two int32x4 halves, narrowed with saturation to 8 × i16.
#[inline(always)]
unsafe fn narrow_shift(lo: int32x4_t, hi: int32x4_t, shift: i32) -> int16x8_t {
    unsafe {
        let s = vdupq_n_s32(-shift);
        vcombine_s16(vqmovn_s32(vshlq_s32(lo, s)), vqmovn_s32(vshlq_s32(hi, s)))
    }
}

/// Clip 8 × i16 to `0..=max` as u16.
#[inline(always)]
unsafe fn clip_u16(v: int16x8_t, maxv: int16x8_t) -> uint16x8_t {
    unsafe { vreinterpretq_u16_s16(vminq_s16(vmaxq_s16(v, vdupq_n_s16(0)), maxv)) }
}

// ----------------------------------------------------------------------
// Interpolation
// ----------------------------------------------------------------------

fn copy_neon(dst: &mut [i16], src: &[u16], src_stride: usize, w: usize, h: usize, shift: i32) {
    if !fits(src.len(), src_stride, h, w, 0) {
        return (HevcDsp::SCALAR.qpel_copy)(dst, src, src_stride, w, h, shift);
    }
    unsafe {
        let sh = vdupq_n_s16(shift as i16);
        for y in 0..h {
            let mut x = 0;
            while x < w {
                let v = vreinterpretq_s16_u16(vld1q_u16(src.as_ptr().add(y * src_stride + x)));
                store_n(dst.as_mut_ptr().add(y * w + x), vshlq_s16(v, sh), (w - x).min(8));
                x += 8;
            }
        }
    }
}

/// Horizontal FIR over u16 samples: `src` at the first tap.
#[inline(always)]
unsafe fn fir_h<const TAPS: usize>(dst: *mut i16, src: *const u16, src_stride: usize, w: usize, h: usize, taps: &[i8], shift: i32) {
    unsafe {
        let mut c = [0i16; 8];
        for k in 0..TAPS {
            c[k] = taps[k] as i16;
        }
        for y in 0..h {
            let s = src.add(y * src_stride);
            let d = dst.add(y * w);
            let mut x = 0;
            while x < w {
                let mut lo = vdupq_n_s32(0);
                let mut hi = vdupq_n_s32(0);
                for k in 0..TAPS {
                    let v = vreinterpretq_s16_u16(vld1q_u16(s.add(x + k)));
                    lo = vmlal_n_s16(lo, vget_low_s16(v), c[k]);
                    hi = vmlal_high_n_s16(hi, v, c[k]);
                }
                store_n(d.add(x), narrow_shift(lo, hi, shift), (w - x).min(8));
                x += 8;
            }
        }
    }
}

/// Vertical FIR over u16 (`T = u16`) or i16 (`T = i16`) rows.
#[inline(always)]
unsafe fn fir_v<const TAPS: usize>(dst: *mut i16, src: *const i16, src_stride: usize, w: usize, h: usize, taps: &[i8], shift: i32) {
    unsafe {
        let mut c = [0i16; 8];
        for k in 0..TAPS {
            c[k] = taps[k] as i16;
        }
        for y in 0..h {
            let d = dst.add(y * w);
            let mut x = 0;
            while x < w {
                let mut lo = vdupq_n_s32(0);
                let mut hi = vdupq_n_s32(0);
                for k in 0..TAPS {
                    let v = vld1q_s16(src.add((y + k) * src_stride + x));
                    lo = vmlal_n_s16(lo, vget_low_s16(v), c[k]);
                    hi = vmlal_high_n_s16(hi, v, c[k]);
                }
                store_n(d.add(x), narrow_shift(lo, hi, shift), (w - x).min(8));
                x += 8;
            }
        }
    }
}

fn qpel_h_neon(dst: &mut [i16], src: &[u16], src_stride: usize, w: usize, h: usize, frac: usize, shift: i32) {
    if !fits(src.len(), src_stride, h, w, 8) {
        return (HevcDsp::SCALAR.qpel_h)(dst, src, src_stride, w, h, frac, shift);
    }
    unsafe { fir_h::<8>(dst.as_mut_ptr(), src.as_ptr(), src_stride, w, h, &QPEL_FILTERS[frac][..8], shift) }
}

fn qpel_v_neon(dst: &mut [i16], src: &[u16], src_stride: usize, w: usize, h: usize, frac: usize, shift: i32) {
    if !fits(src.len(), src_stride, h + 7, w, 0) {
        return (HevcDsp::SCALAR.qpel_v)(dst, src, src_stride, w, h, frac, shift);
    }
    // Samples < 2^15: reinterpreting the u16 plane as i16 is exact.
    unsafe { fir_v::<8>(dst.as_mut_ptr(), src.as_ptr() as *const i16, src_stride, w, h, &QPEL_FILTERS[frac][..8], shift) }
}

fn qpel_v2_neon(dst: &mut [i16], src: &[i16], src_stride: usize, w: usize, h: usize, frac: usize) {
    if !fits(src.len(), src_stride, h + 7, w, 0) {
        return (HevcDsp::SCALAR.qpel_v2)(dst, src, src_stride, w, h, frac);
    }
    unsafe { fir_v::<8>(dst.as_mut_ptr(), src.as_ptr(), src_stride, w, h, &QPEL_FILTERS[frac][..8], 6) }
}

fn epel_h_neon(dst: &mut [i16], src: &[u16], src_stride: usize, w: usize, h: usize, frac: usize, shift: i32) {
    if !fits(src.len(), src_stride, h, w, 4) {
        return (HevcDsp::SCALAR.epel_h)(dst, src, src_stride, w, h, frac, shift);
    }
    unsafe { fir_h::<4>(dst.as_mut_ptr(), src.as_ptr(), src_stride, w, h, &EPEL_FILTERS[frac], shift) }
}

fn epel_v_neon(dst: &mut [i16], src: &[u16], src_stride: usize, w: usize, h: usize, frac: usize, shift: i32) {
    if !fits(src.len(), src_stride, h + 3, w, 0) {
        return (HevcDsp::SCALAR.epel_v)(dst, src, src_stride, w, h, frac, shift);
    }
    unsafe { fir_v::<4>(dst.as_mut_ptr(), src.as_ptr() as *const i16, src_stride, w, h, &EPEL_FILTERS[frac], shift) }
}

fn epel_v2_neon(dst: &mut [i16], src: &[i16], src_stride: usize, w: usize, h: usize, frac: usize) {
    if !fits(src.len(), src_stride, h + 3, w, 0) {
        return (HevcDsp::SCALAR.epel_v2)(dst, src, src_stride, w, h, frac);
    }
    unsafe { fir_v::<4>(dst.as_mut_ptr(), src.as_ptr(), src_stride, w, h, &EPEL_FILTERS[frac], 6) }
}

// ----------------------------------------------------------------------
// Combination / weighting
// ----------------------------------------------------------------------

fn uni_neon(dst: &mut [u16], stride: usize, src: &[i16], w: usize, h: usize, shift: i32, max: i32) {
    unsafe {
        let round = vdupq_n_s16(if shift > 0 { 1 << (shift - 1) } else { 0 });
        let sh = vdupq_n_s16(-(shift as i16));
        let maxv = vdupq_n_s16(max as i16);
        for y in 0..h {
            let mut x = 0;
            while x < w {
                let n = (w - x).min(8);
                let s = load_n(src.as_ptr().add(y * w + x), w - x);
                let v = vshlq_s16(vqaddq_s16(s, round), sh);
                store_n_u16(dst.as_mut_ptr().add(y * stride + x), clip_u16(v, maxv), n);
                x += 8;
            }
        }
    }
}

fn bi_neon(dst: &mut [u16], stride: usize, a: &[i16], b: &[i16], w: usize, h: usize, shift: i32, max: i32) {
    unsafe {
        let round = vdupq_n_s32(1 << (shift - 1));
        let maxv = vdupq_n_s16(max as i16);
        for y in 0..h {
            let mut x = 0;
            while x < w {
                let n = (w - x).min(8);
                let va = load_n(a.as_ptr().add(y * w + x), w - x);
                let vb = load_n(b.as_ptr().add(y * w + x), w - x);
                let lo = vaddq_s32(vaddl_s16(vget_low_s16(va), vget_low_s16(vb)), round);
                let hi = vaddq_s32(vaddl_high_s16(va, vb), round);
                let v = narrow_shift(lo, hi, shift);
                store_n_u16(dst.as_mut_ptr().add(y * stride + x), clip_u16(v, maxv), n);
                x += 8;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn weighted_uni_neon(dst: &mut [u16], stride: usize, src: &[i16], w: usize, h: usize, log2_wd: i32, wt: i32, o: i32, max: i32) {
    unsafe {
        let round = vdupq_n_s32(if log2_wd >= 1 { 1 << (log2_wd - 1) } else { 0 });
        let ov = vdupq_n_s32(o);
        let wv = vdupq_n_s32(wt);
        let sh = vdupq_n_s32(-log2_wd.max(0));
        let maxv = vdupq_n_s16(max as i16);
        for y in 0..h {
            let mut x = 0;
            while x < w {
                let n = (w - x).min(8);
                let s = load_n(src.as_ptr().add(y * w + x), w - x);
                let lo = vaddq_s32(vshlq_s32(vaddq_s32(vmulq_s32(vmovl_s16(vget_low_s16(s)), wv), round), sh), ov);
                let hi = vaddq_s32(vshlq_s32(vaddq_s32(vmulq_s32(vmovl_high_s16(s), wv), round), sh), ov);
                let v = vcombine_s16(vqmovn_s32(lo), vqmovn_s32(hi));
                store_n_u16(dst.as_mut_ptr().add(y * stride + x), clip_u16(v, maxv), n);
                x += 8;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn weighted_bi_neon(dst: &mut [u16], stride: usize, a: &[i16], b: &[i16], w: usize, h: usize, log2_wd: i32, w0: i32, w1: i32, o0: i32, o1: i32, max: i32) {
    unsafe {
        let round = vdupq_n_s32((o0 + o1 + 1) << log2_wd);
        let w0v = vdupq_n_s32(w0);
        let w1v = vdupq_n_s32(w1);
        let sh = vdupq_n_s32(-(log2_wd + 1));
        let maxv = vdupq_n_s16(max as i16);
        for y in 0..h {
            let mut x = 0;
            while x < w {
                let n = (w - x).min(8);
                let va = load_n(a.as_ptr().add(y * w + x), w - x);
                let vb = load_n(b.as_ptr().add(y * w + x), w - x);
                let lo = vshlq_s32(vaddq_s32(vaddq_s32(vmulq_s32(vmovl_s16(vget_low_s16(va)), w0v), vmulq_s32(vmovl_s16(vget_low_s16(vb)), w1v)), round), sh);
                let hi = vshlq_s32(vaddq_s32(vaddq_s32(vmulq_s32(vmovl_high_s16(va), w0v), vmulq_s32(vmovl_high_s16(vb), w1v)), round), sh);
                let v = vcombine_s16(vqmovn_s32(lo), vqmovn_s32(hi));
                store_n_u16(dst.as_mut_ptr().add(y * stride + x), clip_u16(v, maxv), n);
                x += 8;
            }
        }
    }
}

// ----------------------------------------------------------------------
// Residual add
// ----------------------------------------------------------------------

fn add_residual_neon(dst: &mut [u16], stride: usize, res: &[i16], n: usize, max: i32) {
    unsafe {
        let maxv = vdupq_n_s16(max as i16);
        for y in 0..n {
            let mut x = 0;
            while x < n {
                let cnt = (n - x).min(8);
                let d = dst.as_mut_ptr().add(y * stride + x);
                let p = vreinterpretq_s16_u16(load_n_u16(d, cnt));
                let r = load_n(res.as_ptr().add(y * n + x), cnt);
                let v = clip_u16(vqaddq_s16(p, r), maxv);
                store_n_u16(d, v, cnt);
                x += 8;
            }
        }
    }
}

// ----------------------------------------------------------------------
// Inverse DCT
// ----------------------------------------------------------------------

/// The 32x32 transform matrix as i16 rows.
const fn build_t16() -> [[i16; 32]; 32] {
    let mut t = [[0i16; 32]; 32];
    let mut j = 0;
    while j < 32 {
        let mut k = 0;
        while k < 32 {
            t[j][k] = TRANSFORM32[j][k] as i16;
            k += 1;
        }
        j += 1;
    }
    t
}

static T16: [[i16; 32]; 32] = build_t16();

fn idct_neon<const N: usize>(coeffs: &mut [i16], bd_shift: i32, max_x: usize, max_y: usize) {
    if max_x == 0 && max_y == 0 {
        let round2 = 1i32 << (bd_shift - 1);
        let v = ((coeffs[0] as i32 * 64 + 64) >> 7).clamp(-32768, 32767);
        let r = ((v * 64 + round2) >> bd_shift).clamp(-32768, 32767) as i16;
        coeffs[..N * N].fill(r);
        return;
    }
    if N == 4 {
        return (HevcDsp::SCALAR.idct[0])(coeffs, bd_shift, max_x, max_y);
    }
    unsafe {
        let mut tmp = [0i16; 32 * 32];
        let step = 32 / N;
        // Stage 1: columns 0..=max_x, all N output rows.
        let nzy = max_y + 1;
        for y in 0..N {
            let mut x = 0;
            while x <= max_x {
                let mut lo = vdupq_n_s32(64);
                let mut hi = vdupq_n_s32(64);
                for j in 0..nzy {
                    let c = T16[j * step][y];
                    let v = load_n(coeffs.as_ptr().add(j * N + x), N - x);
                    lo = vmlal_n_s16(lo, vget_low_s16(v), c);
                    hi = vmlal_high_n_s16(hi, v, c);
                }
                store_n(tmp.as_mut_ptr().add(y * N + x), narrow_shift(lo, hi, 7), (N - x).min(8));
                x += 8;
            }
        }
        // Stage 2: rows; inputs tmp[y][0..=max_x].
        let nzx = max_x + 1;
        let round2 = 1i32 << (bd_shift - 1);
        for y in 0..N {
            let mut x = 0;
            while x < N {
                let mut lo = vdupq_n_s32(round2);
                let mut hi = vdupq_n_s32(round2);
                for j in 0..nzx {
                    let t = tmp[y * N + j];
                    let row = vld1q_s16(T16[j * step].as_ptr().add(x));
                    lo = vmlal_n_s16(lo, vget_low_s16(row), t);
                    hi = vmlal_high_n_s16(hi, row, t);
                }
                store_n(coeffs.as_mut_ptr().add(y * N + x), narrow_shift(lo, hi, bd_shift), (N - x).min(8));
                x += 8;
            }
        }
    }
}

// ----------------------------------------------------------------------
// SAO
// ----------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn sao_band_neon(dst: &mut [u16], dst_stride: usize, src: &[u16], src_stride: usize, w: usize, h: usize, table: &[i16; 32], shift: i32, max: i32) {
    unsafe {
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
        let sh = vdupq_n_s16(-(shift as i16));
        let maxv = vdupq_n_s16(max as i16);
        for y in 0..h {
            let mut x = 0;
            while x < w {
                let n = (w - x).min(8);
                let v = vreinterpretq_s16_u16(load_n_u16(src.as_ptr().add(y * src_stride + x), n));
                let band = vshlq_s16(v, sh);
                let mut off = vdupq_n_s16(0);
                for i in 0..k {
                    let m = vceqq_s16(band, vdupq_n_s16(bands[i]));
                    off = vbslq_s16(m, vdupq_n_s16(offs[i]), off);
                }
                let r = clip_u16(vaddq_s16(v, off), maxv);
                store_n_u16(dst.as_mut_ptr().add(y * dst_stride + x), r, n);
                x += 8;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn sao_edge_neon(dst: &mut [u16], src: &[u16], origin: usize, stride: usize, w: usize, h: usize, na: isize, nb: isize, off: &[i16; 5], max: i32) {
    unsafe {
        let maxv = vdupq_n_s16(max as i16);
        let one = vdupq_n_s16(1);
        let ov: [int16x8_t; 5] = [vdupq_n_s16(off[0]), vdupq_n_s16(off[1]), vdupq_n_s16(0), vdupq_n_s16(off[3]), vdupq_n_s16(off[4])];
        for y in 0..h {
            let mut x = 0;
            while x < w {
                let n = (w - x).min(8);
                let i = origin + y * stride + x;
                let last = i + n - 1;
                if (last as isize + na.max(nb)) as usize + (8 - n) >= src.len() || last + (8 - n) >= src.len() {
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
                let v = vreinterpretq_s16_u16(vld1q_u16(src.as_ptr().add(i)));
                let a = vreinterpretq_s16_u16(vld1q_u16(src.as_ptr().offset(i as isize + na)));
                let b = vreinterpretq_s16_u16(vld1q_u16(src.as_ptr().offset(i as isize + nb)));
                // sign(v - a) = (v > a) - (v < a)
                let sa = vsubq_s16(vandq_s16(vreinterpretq_s16_u16(vcgtq_s16(v, a)), one), vandq_s16(vreinterpretq_s16_u16(vcgtq_s16(a, v)), one));
                let sb = vsubq_s16(vandq_s16(vreinterpretq_s16_u16(vcgtq_s16(v, b)), one), vandq_s16(vreinterpretq_s16_u16(vcgtq_s16(b, v)), one));
                let e = vaddq_s16(vaddq_s16(sa, sb), vdupq_n_s16(2));
                let mut o = vdupq_n_s16(0);
                for k in [0usize, 1, 3, 4] {
                    let m = vceqq_s16(e, vdupq_n_s16(k as i16));
                    o = vbslq_s16(m, ov[k], o);
                }
                let r = clip_u16(vaddq_s16(v, o), maxv);
                store_n_u16(dst.as_mut_ptr().add(i), r, n);
                x += 8;
            }
        }
    }
}

// ----------------------------------------------------------------------
// Deblocking
// ----------------------------------------------------------------------
//
// One 4-line segment is one vector of four i32 lanes per sample position,
// so the per-segment decisions (8.7.2.5.3) are per vector; a call covers
// two luma segments (or four 2-line chroma segments as two vectors).

/// `[p3, p2, p1, p0, q0, q1, q2, q3]` of one 4-line segment.
type Seg = [int32x4_t; 8];

/// Filter one segment in place. Returns without touching it when its
/// parameters are zero or the segment does not pass the beta test.
#[inline(always)]
unsafe fn luma_segment(v: &mut Seg, beta: i32, tc: i32, no_p: bool, no_q: bool, max: i32) {
    unsafe {
        if beta == 0 && tc == 0 {
            return;
        }
        let [p3, p2, p1, p0, q0, q1, q2, q3] = *v;
        let add = |a, b| vaddq_s32(a, b);
        let sub = |a, b| vsubq_s32(a, b);
        let dbl = |a| vshlq_n_s32::<1>(a);
        let dpv = vabsq_s32(add(sub(p2, dbl(p1)), p0));
        let dqv = vabsq_s32(add(sub(q2, dbl(q1)), q0));
        let ev = add(vabdq_s32(p3, p0), vabdq_s32(q0, q3));
        let fv = vabdq_s32(p0, q0);
        let dp = [vgetq_lane_s32::<0>(dpv), vgetq_lane_s32::<3>(dpv)];
        let dq = [vgetq_lane_s32::<0>(dqv), vgetq_lane_s32::<3>(dqv)];
        let e = [vgetq_lane_s32::<0>(ev), vgetq_lane_s32::<3>(ev)];
        let f = [vgetq_lane_s32::<0>(fv), vgetq_lane_s32::<3>(fv)];
        let dpq0 = dp[0] + dq[0];
        let dpq3 = dp[1] + dq[1];
        if dpq0 + dpq3 >= beta {
            return;
        }
        let dsam = |l: usize, dpq: i32| dpq < (beta >> 2) && e[l] < (beta >> 3) && f[l] < ((5 * tc + 1) >> 1);
        let strong = dsam(0, 2 * dpq0) && dsam(1, 2 * dpq3);
        let side = (beta + (beta >> 1)) >> 3;
        let dep = dp[0] + dp[1] < side;
        let deq = dq[0] + dq[1] < side;
        let tcv = vdupq_n_s32(tc);
        let tc2 = vdupq_n_s32(2 * tc);
        let tch = vdupq_n_s32(tc >> 1);
        let zero = vdupq_n_s32(0);
        let maxv = vdupq_n_s32(max);
        let clamp = |x, lo, hi| vminq_s32(vmaxq_s32(x, lo), hi);
        let (np0, np1, np2, nq0, nq1, nq2);
        if strong {
            let two = vdupq_n_s32(2);
            let four = vdupq_n_s32(4);
            let p0q0 = add(p0, q0);
            np0 = clamp(vshrq_n_s32::<3>(add(add(p2, dbl(add(p1, p0q0))), add(q1, four))), sub(p0, tc2), add(p0, tc2));
            np1 = clamp(vshrq_n_s32::<2>(add(add(p2, p1), add(p0q0, two))), sub(p1, tc2), add(p1, tc2));
            np2 = clamp(vshrq_n_s32::<3>(add(add(dbl(p3), add(p2, dbl(p2))), add(add(p1, p0q0), four))), sub(p2, tc2), add(p2, tc2));
            nq0 = clamp(vshrq_n_s32::<3>(add(add(p1, dbl(add(p0q0, q1))), add(q2, four))), sub(q0, tc2), add(q0, tc2));
            nq1 = clamp(vshrq_n_s32::<2>(add(add(p0q0, q1), add(q2, two))), sub(q1, tc2), add(q1, tc2));
            nq2 = clamp(vshrq_n_s32::<3>(add(add(p0q0, q1), add(add(q2, dbl(q2)), add(dbl(q3), four)))), sub(q2, tc2), add(q2, tc2));
        } else {
            let delta = vshrq_n_s32::<4>(add(sub(vmulq_n_s32(sub(q0, p0), 9), vmulq_n_s32(sub(q1, p1), 3)), vdupq_n_s32(8)));
            let w_m = vcltq_s32(vabsq_s32(delta), vdupq_n_s32(10 * tc));
            let delta = clamp(delta, vnegq_s32(tcv), tcv);
            let wp0 = clamp(add(p0, delta), zero, maxv);
            let wq0 = clamp(sub(q0, delta), zero, maxv);
            let one = vdupq_n_s32(1);
            let dpv2 = clamp(vshrq_n_s32::<1>(add(sub(vshrq_n_s32::<1>(add(add(p2, p0), one)), p1), delta)), vnegq_s32(tch), tch);
            let dqv2 = clamp(vshrq_n_s32::<1>(sub(sub(vshrq_n_s32::<1>(add(add(q2, q0), one)), q1), delta)), vnegq_s32(tch), tch);
            let wp1 = clamp(add(p1, dpv2), zero, maxv);
            let wq1 = clamp(add(q1, dqv2), zero, maxv);
            np0 = vbslq_s32(w_m, wp0, p0);
            nq0 = vbslq_s32(w_m, wq0, q0);
            np1 = if dep { vbslq_s32(w_m, wp1, p1) } else { p1 };
            nq1 = if deq { vbslq_s32(w_m, wq1, q1) } else { q1 };
            np2 = p2;
            nq2 = q2;
        }
        if !no_p {
            v[1] = np2;
            v[2] = np1;
            v[3] = np0;
        }
        if !no_q {
            v[4] = nq0;
            v[5] = nq1;
            v[6] = nq2;
        }
    }
}

/// Four consecutive u16 as 4 x i32.
#[inline(always)]
unsafe fn ld4_u16(p: *const u16) -> int32x4_t {
    unsafe { vreinterpretq_s32_u32(vmovl_u16(vld1_u16(p))) }
}

/// 4 x i32 (within u16) to four u16.
#[inline(always)]
unsafe fn pack4_u16(v: int32x4_t) -> uint16x4_t {
    unsafe { vqmovun_s32(v) }
}

#[allow(clippy::too_many_arguments)]
fn deblock_luma_h_neon(data: &mut [u16], off: usize, stride: usize, beta: [i32; 2], tc: [i32; 2], no_p: [bool; 2], no_q: [bool; 2], max: i32) {
    if (beta[0] == 0 && tc[0] == 0) && (beta[1] == 0 && tc[1] == 0) {
        return;
    }
    assert!(off >= 4 * stride && off + 3 * stride + 8 <= data.len());
    unsafe {
        let p = data.as_mut_ptr().add(off);
        for seg in 0..2 {
            if beta[seg] == 0 && tc[seg] == 0 {
                continue;
            }
            let base = p.add(4 * seg);
            let mut v: Seg = [vdupq_n_s32(0); 8];
            for k in 0..8 {
                v[k] = ld4_u16(base.offset((k as isize - 4) * stride as isize));
            }
            luma_segment(&mut v, beta[seg], tc[seg], no_p[seg], no_q[seg], max);
            for k in 1..7 {
                vst1_u16(base.offset((k as isize - 4) * stride as isize), pack4_u16(v[k]));
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn deblock_luma_v_neon(data: &mut [u16], off: usize, stride: usize, beta: [i32; 2], tc: [i32; 2], no_p: [bool; 2], no_q: [bool; 2], max: i32) {
    if (beta[0] == 0 && tc[0] == 0) && (beta[1] == 0 && tc[1] == 0) {
        return;
    }
    assert!(off >= 4 && off + 7 * stride + 4 <= data.len());
    unsafe {
        let p = data.as_mut_ptr().add(off);
        for seg in 0..2 {
            if beta[seg] == 0 && tc[seg] == 0 {
                continue;
            }
            // Four rows x 8 samples (p3..q3): transpose to eight columns of 4.
            let base = p.add(4 * seg * stride);
            let r0 = vld1q_u16(base.sub(4));
            let r1 = vld1q_u16(base.add(stride).sub(4));
            let r2 = vld1q_u16(base.add(2 * stride).sub(4));
            let r3 = vld1q_u16(base.add(3 * stride).sub(4));
            let a = vtrnq_u16(r0, r1);
            let b = vtrnq_u16(r2, r3);
            let c = vtrnq_u32(vreinterpretq_u32_u16(a.0), vreinterpretq_u32_u16(b.0));
            let d = vtrnq_u32(vreinterpretq_u32_u16(a.1), vreinterpretq_u32_u16(b.1));
            // Columns: c.0 low = col0, d.0 low = col1, c.1 low = col2, d.1 low = col3,
            // c.0 high = col4, d.0 high = col5, c.1 high = col6, d.1 high = col7.
            let cols = [
                vget_low_u16(vreinterpretq_u16_u32(c.0)),
                vget_low_u16(vreinterpretq_u16_u32(d.0)),
                vget_low_u16(vreinterpretq_u16_u32(c.1)),
                vget_low_u16(vreinterpretq_u16_u32(d.1)),
                vget_high_u16(vreinterpretq_u16_u32(c.0)),
                vget_high_u16(vreinterpretq_u16_u32(d.0)),
                vget_high_u16(vreinterpretq_u16_u32(c.1)),
                vget_high_u16(vreinterpretq_u16_u32(d.1)),
            ];
            let mut v: Seg = [vdupq_n_s32(0); 8];
            for k in 0..8 {
                v[k] = vreinterpretq_s32_u32(vmovl_u16(cols[k]));
            }
            luma_segment(&mut v, beta[seg], tc[seg], no_p[seg], no_q[seg], max);
            // Back to rows: transpose the eight 4-lane columns.
            let mut cc = [vdup_n_u16(0); 8];
            for k in 0..8 {
                cc[k] = pack4_u16(v[k]);
            }
            let l0 = vcombine_u16(cc[0], cc[4]); // col0 | col4
            let l1 = vcombine_u16(cc[1], cc[5]);
            let l2 = vcombine_u16(cc[2], cc[6]);
            let l3 = vcombine_u16(cc[3], cc[7]);
            let a = vtrnq_u16(l0, l1);
            let b = vtrnq_u16(l2, l3);
            let c = vtrnq_u32(vreinterpretq_u32_u16(a.0), vreinterpretq_u32_u16(b.0));
            let d = vtrnq_u32(vreinterpretq_u32_u16(a.1), vreinterpretq_u32_u16(b.1));
            // Row i: [col0..3 at row i | col4..7 at row i] — the same
            // network gives rows back in the same layout as the columns came in.
            let rows = [c.0, d.0, c.1, d.1];
            for i in 0..4 {
                vst1q_u16(base.add(i * stride).sub(4), vreinterpretq_u16_u32(rows[i]));
            }
        }
    }
}

/// Chroma filter on one vector of four lines with per-pair `tc`.
#[inline(always)]
unsafe fn chroma_lines4(v: &mut [int32x4_t; 4], tc: [i32; 2], no_p: [bool; 2], no_q: [bool; 2], max: i32) {
    unsafe {
        let [p1, p0, q0, q1] = *v;
        let t = [tc[0], tc[0], tc[1], tc[1]];
        let tcv = vld1q_s32(t.as_ptr());
        let m = |a: [bool; 2]| {
            let x = [-(a[0] as i32), -(a[0] as i32), -(a[1] as i32), -(a[1] as i32)];
            vreinterpretq_u32_s32(vld1q_s32(x.as_ptr()))
        };
        let on = vcgtq_s32(tcv, vdupq_n_s32(0));
        let wp = vbicq_u32(on, m(no_p));
        let wq = vbicq_u32(on, m(no_q));
        let zero = vdupq_n_s32(0);
        let maxv = vdupq_n_s32(max);
        let d = vshrq_n_s32::<3>(vaddq_s32(vaddq_s32(vshlq_n_s32::<2>(vsubq_s32(q0, p0)), vsubq_s32(p1, q1)), vdupq_n_s32(4)));
        let d = vminq_s32(vmaxq_s32(d, vnegq_s32(tcv)), tcv);
        let np0 = vminq_s32(vmaxq_s32(vaddq_s32(p0, d), zero), maxv);
        let nq0 = vminq_s32(vmaxq_s32(vsubq_s32(q0, d), zero), maxv);
        v[1] = vbslq_s32(wp, np0, p0);
        v[2] = vbslq_s32(wq, nq0, q0);
    }
}

fn deblock_chroma_h_neon(data: &mut [u16], off: usize, stride: usize, tc: [i32; 4], no_p: [bool; 4], no_q: [bool; 4], max: i32) {
    if tc.iter().all(|&t| t == 0) {
        return;
    }
    assert!(off >= 2 * stride && off + stride + 8 <= data.len());
    unsafe {
        let p = data.as_mut_ptr().add(off);
        for half in 0..2 {
            let base = p.add(4 * half);
            let mut v = [ld4_u16(base.sub(2 * stride)), ld4_u16(base.sub(stride)), ld4_u16(base), ld4_u16(base.add(stride))];
            chroma_lines4(&mut v, [tc[2 * half], tc[2 * half + 1]], [no_p[2 * half], no_p[2 * half + 1]], [no_q[2 * half], no_q[2 * half + 1]], max);
            vst1_u16(base.sub(stride), pack4_u16(v[1]));
            vst1_u16(base, pack4_u16(v[2]));
        }
    }
}

fn deblock_chroma_v_neon(data: &mut [u16], off: usize, stride: usize, tc: [i32; 4], no_p: [bool; 4], no_q: [bool; 4], max: i32) {
    if tc.iter().all(|&t| t == 0) {
        return;
    }
    assert!(off >= 2 && off + 7 * stride + 2 <= data.len());
    unsafe {
        let p = data.as_mut_ptr().add(off);
        for half in 0..2 {
            let base = p.add(4 * half * stride);
            // Four rows x 4 samples (p1 p0 q0 q1) -> four columns of 4.
            let r0 = vld1_u16(base.sub(2));
            let r1 = vld1_u16(base.add(stride).sub(2));
            let r2 = vld1_u16(base.add(2 * stride).sub(2));
            let r3 = vld1_u16(base.add(3 * stride).sub(2));
            let a = vtrn_u16(r0, r1);
            let b = vtrn_u16(r2, r3);
            let c = vtrn_u32(vreinterpret_u32_u16(a.0), vreinterpret_u32_u16(b.0));
            let d = vtrn_u32(vreinterpret_u32_u16(a.1), vreinterpret_u32_u16(b.1));
            let cols = [vreinterpret_u16_u32(c.0), vreinterpret_u16_u32(d.0), vreinterpret_u16_u32(c.1), vreinterpret_u16_u32(d.1)];
            let mut v = [
                vreinterpretq_s32_u32(vmovl_u16(cols[0])),
                vreinterpretq_s32_u32(vmovl_u16(cols[1])),
                vreinterpretq_s32_u32(vmovl_u16(cols[2])),
                vreinterpretq_s32_u32(vmovl_u16(cols[3])),
            ];
            chroma_lines4(&mut v, [tc[2 * half], tc[2 * half + 1]], [no_p[2 * half], no_p[2 * half + 1]], [no_q[2 * half], no_q[2 * half + 1]], max);
            // (p0, q0) per row.
            let p0 = pack4_u16(v[1]);
            let q0 = pack4_u16(v[2]);
            let pq = vzip_u16(p0, q0); // p0r0 q0r0 p0r1 q0r1 | p0r2 q0r2 p0r3 q0r3
            let mut t = [0u16; 8];
            vst1_u16(t.as_mut_ptr(), pq.0);
            vst1_u16(t.as_mut_ptr().add(4), pq.1);
            for i in 0..4 {
                let dst = base.add(i * stride).sub(1);
                *dst = t[2 * i];
                *dst.add(1) = t[2 * i + 1];
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

    fn neon() -> HevcDsp {
        let mut d = HevcDsp::SCALAR;
        install(&mut d);
        d
    }

    #[test]
    fn interp_matches_scalar() {
        let d = neon();
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
    fn combine_and_idct_and_sao_match_scalar() {
        let d = neon();
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
                for &(log2_wd, wt, o) in &[(6 + 14 - bd as i32, 128, 0), (14 - bd as i32, 1, 5), (7 + 14 - bd as i32, -20, -3), (3 + 14 - bd as i32, 255, 127)] {
                    (s.weighted_uni)(&mut d1, stride, &a, w, h, log2_wd, wt, o, max);
                    (d.weighted_uni)(&mut d2, stride, &a, w, h, log2_wd, wt, o, max);
                    assert_eq!(d1, d2, "wuni {w}x{h} bd={bd}");
                    (s.weighted_bi)(&mut d1, stride, &a, &b, w, h, log2_wd, wt, 3 - wt, o, -o, max);
                    (d.weighted_bi)(&mut d2, stride, &a, &b, w, h, log2_wd, wt, 3 - wt, o, -o, max);
                    assert_eq!(d1, d2, "wbi {w}x{h} bd={bd}");
                }
                if w == h && w >= 4 && w.is_power_of_two() {
                    let res: Vec<i16> = (0..w * w).map(|_| (lcg(&mut seed) % 2000) as i16 - 1000).collect();
                    let base: Vec<u16> = (0..stride * h).map(|_| (lcg(&mut seed) % (max as u32 + 1)) as u16).collect();
                    let mut d1 = base.clone();
                    let mut d2 = base.clone();
                    (s.add_residual)(&mut d1, stride, &res, w, max);
                    (d.add_residual)(&mut d2, stride, &res, w, max);
                    assert_eq!(d1, d2, "add_residual {w}");
                }
            }
        }
        for &(n, log2) in &[(4usize, 2u32), (8, 3), (16, 4), (32, 5)] {
            for trial in 0..200 {
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
                assert_eq!(a, b, "idct n={n} trial={trial}");
            }
        }
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

    #[test]
    fn deblocking_matches_scalar() {
        let d = neon();
        let s = HevcDsp::SCALAR;
        let mut seed = 23u64;
        let stride = 40;
        for trial in 0..600 {
            let bd = [8u32, 10, 12][trial % 3];
            let max = (1i32 << bd) - 1;
            let base = lcg(&mut seed) % (max as u32 + 1);
            let spread = 1 + lcg(&mut seed) % (1 << (bd - 4));
            let plane: Vec<u16> = (0..stride * 32).map(|_| (base + lcg(&mut seed) % spread).min(max as u32) as u16).collect();
            let rnd = |seed: &mut u64, n: u32| lcg(seed) % n;
            let sh = bd - 8;
            let v = |seed: &mut u64, n: u32| (rnd(seed, n) as i32) << sh;
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
            assert_eq!(a, b, "hevc deblock kind {} trial {trial} beta {beta:?} tc {tc:?}", trial % 4);
        }
    }
}
