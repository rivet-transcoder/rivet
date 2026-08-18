//! NEON versions of the H.264 kernels (AArch64).
//!
//! Eight 16-bit lanes per vector, so a 16-wide block row is two vectors.
//! The six-tap sums fit 16 bits for 8-bit input; the centre position
//! filters the 16-bit horizontal intermediates vertically with 32-bit
//! multiply-accumulates. Quarter positions are `vrhadd` of two 8-bit
//! results. Checked bit-exact against the scalar reference by the tests
//! (run on AArch64 in CI).

#![cfg(target_arch = "aarch64")]

use std::arch::aarch64::*;

use super::h264::{H264Dsp, NO_DC, PRED_STRIDE};

/// Replace the scalar entries of `d` with the NEON kernels.
pub fn install(d: &mut H264Dsp<u8>) {
    d.qpel = [
        qpel_neon::<0, 0>,
        qpel_neon::<1, 0>,
        qpel_neon::<2, 0>,
        qpel_neon::<3, 0>,
        qpel_neon::<0, 1>,
        qpel_neon::<1, 1>,
        qpel_neon::<2, 1>,
        qpel_neon::<3, 1>,
        qpel_neon::<0, 2>,
        qpel_neon::<1, 2>,
        qpel_neon::<2, 2>,
        qpel_neon::<3, 2>,
        qpel_neon::<0, 3>,
        qpel_neon::<1, 3>,
        qpel_neon::<2, 3>,
        qpel_neon::<3, 3>,
    ];
    d.chroma = chroma_neon;
    d.avg = avg_neon;
    d.weighted_uni = weighted_uni_neon;
    d.weighted_bi = weighted_bi_neon;
    d.deblock_luma_v = deblock_luma_v_neon;
    d.deblock_luma_h = deblock_luma_h_neon;
    d.deblock_luma_v_intra = deblock_luma_v_intra_neon;
    d.deblock_luma_h_intra = deblock_luma_h_intra_neon;
    d.deblock_chroma_v = deblock_chroma_v_neon;
    d.deblock_chroma_h = deblock_chroma_h_neon;
    d.deblock_chroma_v_intra = deblock_chroma_v_intra_neon;
    d.deblock_chroma_h_intra = deblock_chroma_h_intra_neon;
    d.idct4_add = idct4_add_neon;
    d.idct8_add = idct8_add_neon;
    d.idct4_dc_add = idct4_dc_add_neon;
    d.idct8_dc_add = idct8_dc_add_neon;
    d.residual4 = residual4_neon;
    d.residual8 = residual8_neon;
}

/// Store the first `n` (≤ 8) bytes of `v`.
#[inline(always)]
unsafe fn store8_n(dst: *mut u8, v: uint8x8_t, n: usize) {
    unsafe {
        if n >= 8 {
            vst1_u8(dst, v);
        } else {
            let mut t = [0u8; 8];
            vst1_u8(t.as_mut_ptr(), v);
            std::ptr::copy_nonoverlapping(t.as_ptr(), dst, n);
        }
    }
}

/// Load 8 bytes as 8 × i16.
#[inline(always)]
unsafe fn load8(p: *const u8) -> int16x8_t {
    unsafe { vreinterpretq_s16_u16(vmovl_u8(vld1_u8(p))) }
}

/// `a - 5b + 20c + 20d - 5e + f` in i16.
#[inline(always)]
unsafe fn tap6(a: int16x8_t, b: int16x8_t, c: int16x8_t, d: int16x8_t, e: int16x8_t, f: int16x8_t) -> int16x8_t {
    unsafe {
        let t = vaddq_s16(c, d);
        let u = vaddq_s16(b, e);
        let v = vaddq_s16(a, f);
        let t20 = vaddq_s16(vshlq_n_s16::<4>(t), vshlq_n_s16::<2>(t));
        let u5 = vaddq_s16(vshlq_n_s16::<2>(u), u);
        vsubq_s16(vaddq_s16(v, t20), u5)
    }
}

/// `clip((v + 16) >> 5)` packed to 8 × u8.
#[inline(always)]
unsafe fn round5(v: int16x8_t) -> uint8x8_t {
    unsafe { vqmovun_s16(vshrq_n_s16::<5>(vaddq_s16(v, vdupq_n_s16(16)))) }
}

/// Horizontal intermediate for window row `row`, columns `x0..x0+8`.
#[inline(always)]
unsafe fn b1_row(src: *const u8, stride: usize, row: usize, x0: usize) -> int16x8_t {
    unsafe {
        let p = src.add(row * stride + x0);
        tap6(load8(p), load8(p.add(1)), load8(p.add(2)), load8(p.add(3)), load8(p.add(4)), load8(p.add(5)))
    }
}

/// Vertical intermediate at window column `col`, block row `y`, 8 lanes.
#[inline(always)]
unsafe fn h1_row(src: *const u8, stride: usize, col: usize, y: usize) -> int16x8_t {
    unsafe {
        let p = src.add(y * stride + col);
        tap6(load8(p), load8(p.add(stride)), load8(p.add(2 * stride)), load8(p.add(3 * stride)), load8(p.add(4 * stride)), load8(p.add(5 * stride)))
    }
}

/// Centre position: vertical six-tap over b1 rows y..y+5, `clip((v + 512) >> 10)`.
#[inline(always)]
unsafe fn j_row(src: *const u8, stride: usize, y: usize, x0: usize) -> uint8x8_t {
    unsafe {
        let r: [int16x8_t; 6] = [
            b1_row(src, stride, y, x0),
            b1_row(src, stride, y + 1, x0),
            b1_row(src, stride, y + 2, x0),
            b1_row(src, stride, y + 3, x0),
            b1_row(src, stride, y + 4, x0),
            b1_row(src, stride, y + 5, x0),
        ];
        let taps: [i16; 6] = [1, -5, 20, 20, -5, 1];
        let mut lo = vdupq_n_s32(512);
        let mut hi = vdupq_n_s32(512);
        for k in 0..6 {
            lo = vmlal_n_s16(lo, vget_low_s16(r[k]), taps[k]);
            hi = vmlal_high_n_s16(hi, r[k], taps[k]);
        }
        let v = vcombine_s16(vqmovn_s32(vshrq_n_s32::<10>(lo)), vqmovn_s32(vshrq_n_s32::<10>(hi)));
        vqmovun_s16(v)
    }
}

fn qpel_neon<const XF: usize, const YF: usize>(dst: &mut [u8], src: &[u8], stride: usize, w: usize, h: usize, _max: i32) {
    // 8-lane loads up to column x0 + 5 + 7 of the last row used.
    let need = (h + 5 - 1) * stride + w.div_ceil(8) * 8 + 5 + 8;
    if src.len() < need {
        return (H264Dsp::<u8>::SCALAR.qpel[YF * 4 + XF])(dst, src, stride, w, h, 255);
    }
    unsafe {
        let s = src.as_ptr();
        for y in 0..h {
            let mut x0 = 0;
            while x0 < w {
                let n = (w - x0).min(8);
                let g = |dx: usize, dy: usize| vld1_u8(s.add((y + 2 + dy) * stride + 2 + x0 + dx));
                let b = || round5(b1_row(s, stride, y + 2, x0));
                let b_below = || round5(b1_row(s, stride, y + 3, x0));
                let hh = || round5(h1_row(s, stride, 2 + x0, y));
                let hh_right = || round5(h1_row(s, stride, 3 + x0, y));
                let v: uint8x8_t = match (XF, YF) {
                    (0, 0) => g(0, 0),
                    (1, 0) => vrhadd_u8(g(0, 0), b()),
                    (2, 0) => b(),
                    (3, 0) => vrhadd_u8(g(1, 0), b()),
                    (0, 1) => vrhadd_u8(g(0, 0), hh()),
                    (0, 2) => hh(),
                    (0, 3) => vrhadd_u8(g(0, 1), hh()),
                    (2, 2) => j_row(s, stride, y, x0),
                    (1, 1) => vrhadd_u8(b(), hh()),
                    (3, 1) => vrhadd_u8(b(), hh_right()),
                    (1, 3) => vrhadd_u8(hh(), b_below()),
                    (3, 3) => vrhadd_u8(hh_right(), b_below()),
                    (2, 1) => vrhadd_u8(b(), j_row(s, stride, y, x0)),
                    (2, 3) => vrhadd_u8(j_row(s, stride, y, x0), b_below()),
                    (1, 2) => vrhadd_u8(hh(), j_row(s, stride, y, x0)),
                    (3, 2) => vrhadd_u8(j_row(s, stride, y, x0), hh_right()),
                    _ => unreachable!(),
                };
                // The scratch row is 16 wide whatever `w` is.
                vst1_u8(dst.as_mut_ptr().add(y * PRED_STRIDE + x0), v);
                x0 += 8;
            }
        }
    }
}

fn chroma_neon(dst: &mut [u8], src: &[u8], stride: usize, w: usize, h: usize, xf: i32, yf: i32) {
    if src.len() < h * stride + w.div_ceil(8) * 8 + 9 {
        return (H264Dsp::<u8>::SCALAR.chroma)(dst, src, stride, w, h, xf, yf);
    }
    unsafe {
        let a = ((8 - xf) * (8 - yf)) as i16;
        let b = (xf * (8 - yf)) as i16;
        let c = ((8 - xf) * yf) as i16;
        let d = (xf * yf) as i16;
        let s = src.as_ptr();
        for y in 0..h {
            let mut x0 = 0;
            while x0 < w {
                let n = (w - x0).min(8);
                let r0 = s.add(y * stride + x0);
                let r1 = s.add((y + 1) * stride + x0);
                let mut v = vdupq_n_s16(32);
                v = vmlaq_n_s16(v, load8(r0), a);
                v = vmlaq_n_s16(v, load8(r0.add(1)), b);
                v = vmlaq_n_s16(v, load8(r1), c);
                v = vmlaq_n_s16(v, load8(r1.add(1)), d);
                let p = vqmovun_s16(vshrq_n_s16::<6>(v));
                vst1_u8(dst.as_mut_ptr().add(y * PRED_STRIDE + x0), p);
                x0 += 8;
            }
        }
    }
}

fn avg_neon(dst: &mut [u8], stride: usize, a: &[u8], b: &[u8], w: usize, h: usize) {
    unsafe {
        for y in 0..h {
            let mut x0 = 0;
            while x0 < w {
                let n = (w - x0).min(8);
                let v = vrhadd_u8(vld1_u8(a.as_ptr().add(y * PRED_STRIDE + x0)), vld1_u8(b.as_ptr().add(y * PRED_STRIDE + x0)));
                store8_n(dst.as_mut_ptr().add(y * stride + x0), v, n);
                x0 += 8;
            }
        }
    }
}

fn weighted_uni_neon(dst: &mut [u8], stride: usize, src: &[u8], w: usize, h: usize, log_wd: i32, wt: i32, o: i32, _max: i32) {
    unsafe {
        let round = vdupq_n_s16(if log_wd >= 1 { 1 << (log_wd - 1) } else { 0 });
        let sh = vdupq_n_s16(-(log_wd.max(0) as i16));
        let ov = vdupq_n_s16(o as i16);
        for y in 0..h {
            let mut x0 = 0;
            while x0 < w {
                let n = (w - x0).min(8);
                let s = load8(src.as_ptr().add(y * PRED_STRIDE + x0));
                let v = vaddq_s16(vshlq_s16(vaddq_s16(vmulq_n_s16(s, wt as i16), round), sh), ov);
                store8_n(dst.as_mut_ptr().add(y * stride + x0), vqmovun_s16(v), n);
                x0 += 8;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn weighted_bi_neon(dst: &mut [u8], stride: usize, a: &[u8], b: &[u8], w: usize, h: usize, log_wd: i32, w0: i32, w1: i32, o0: i32, o1: i32, _max: i32) {
    unsafe {
        let round = vdupq_n_s32(1 << log_wd);
        let off = vdupq_n_s32((o0 + o1 + 1) >> 1);
        let sh = vdupq_n_s32(-(log_wd + 1));
        for y in 0..h {
            let mut x0 = 0;
            while x0 < w {
                let n = (w - x0).min(8);
                let va = load8(a.as_ptr().add(y * PRED_STRIDE + x0));
                let vb = load8(b.as_ptr().add(y * PRED_STRIDE + x0));
                let mut lo = round;
                let mut hi = round;
                lo = vmlal_n_s16(lo, vget_low_s16(va), w0 as i16);
                hi = vmlal_high_n_s16(hi, va, w0 as i16);
                lo = vmlal_n_s16(lo, vget_low_s16(vb), w1 as i16);
                hi = vmlal_high_n_s16(hi, vb, w1 as i16);
                let lo = vaddq_s32(vshlq_s32(lo, sh), off);
                let hi = vaddq_s32(vshlq_s32(hi, sh), off);
                let v = vcombine_s16(vqmovn_s32(lo), vqmovn_s32(hi));
                store8_n(dst.as_mut_ptr().add(y * stride + x0), vqmovun_s16(v), n);
                x0 += 8;
            }
        }
    }
}

// ----------------------------------------------------------------------
// Deblocking
// ----------------------------------------------------------------------
//
// Sixteen luma lines are two vectors of eight i16 lanes per sample
// position; eight chroma lines one vector. A horizontal edge loads a sample
// position as one row; a vertical edge transposes 16 rows x 8 bytes into
// column vectors (two 8x8 byte transposes) and back.

/// The eight positions of eight lines: `[p3, p2, p1, p0, q0, q1, q2, q3]`.
type Lines8 = [int16x8_t; 8];

/// `|a - b| < t` per lane, as a mask.
#[inline(always)]
unsafe fn diff_lt(a: int16x8_t, b: int16x8_t, t: int16x8_t) -> uint16x8_t {
    unsafe { vcltq_s16(vabdq_s16(a, b), t) }
}

/// bS < 4 luma filter on eight lines (8.7.2.3); `tc0v` = tC0 per lane (−1 = bS 0).
#[inline(always)]
unsafe fn luma_filter_normal(v: &mut Lines8, alpha: i32, beta: i32, tc0v: int16x8_t) {
    unsafe {
        let [_, p2, p1, p0, q0, q1, q2, _] = *v;
        let alpha = vdupq_n_s16(alpha as i16);
        let beta = vdupq_n_s16(beta as i16);
        let zero = vdupq_n_s16(0);
        let bs_on = vcgtq_s16(tc0v, vdupq_n_s16(-1));
        let mask = vandq_u16(vandq_u16(diff_lt(p0, q0, alpha), diff_lt(p1, p0, beta)), vandq_u16(diff_lt(q1, q0, beta), bs_on));
        let ap = diff_lt(p2, p0, beta);
        let aq = diff_lt(q2, q0, beta);
        // tc = tc0 + (ap < beta) + (aq < beta): masks are all-ones (-1).
        let tc = vsubq_s16(vsubq_s16(tc0v, vreinterpretq_s16_u16(ap)), vreinterpretq_s16_u16(aq));
        let d = vshrq_n_s16::<3>(vaddq_s16(vaddq_s16(vshlq_n_s16::<2>(vsubq_s16(q0, p0)), vsubq_s16(p1, q1)), vdupq_n_s16(4)));
        let d = vminq_s16(vmaxq_s16(d, vnegq_s16(tc)), tc);
        let np0 = vaddq_s16(p0, d);
        let nq0 = vsubq_s16(q0, d);
        let avg = vshrq_n_s16::<1>(vaddq_s16(vaddq_s16(p0, q0), vdupq_n_s16(1)));
        let ntc0 = vnegq_s16(tc0v);
        let dp1 = vshrq_n_s16::<1>(vsubq_s16(vaddq_s16(p2, avg), vshlq_n_s16::<1>(p1)));
        let dp1 = vminq_s16(vmaxq_s16(dp1, ntc0), tc0v);
        let np1 = vaddq_s16(p1, vandq_s16(dp1, vreinterpretq_s16_u16(ap)));
        let dq1 = vshrq_n_s16::<1>(vsubq_s16(vaddq_s16(q2, avg), vshlq_n_s16::<1>(q1)));
        let dq1 = vminq_s16(vmaxq_s16(dq1, ntc0), tc0v);
        let nq1 = vaddq_s16(q1, vandq_s16(dq1, vreinterpretq_s16_u16(aq)));
        let clip = |x: int16x8_t| vminq_s16(vmaxq_s16(x, zero), vdupq_n_s16(255));
        v[2] = vbslq_s16(mask, np1, p1);
        v[3] = vbslq_s16(mask, clip(np0), p0);
        v[4] = vbslq_s16(mask, clip(nq0), q0);
        v[5] = vbslq_s16(mask, nq1, q1);
    }
}

/// bS 4 luma filter on eight lines (8.7.2.4).
#[inline(always)]
unsafe fn luma_filter_intra(v: &mut Lines8, alpha: i32, beta: i32) {
    unsafe {
        let [p3, p2, p1, p0, q0, q1, q2, q3] = *v;
        let alphav = vdupq_n_s16(alpha as i16);
        let beta = vdupq_n_s16(beta as i16);
        let mask = vandq_u16(vandq_u16(diff_lt(p0, q0, alphav), diff_lt(p1, p0, beta)), diff_lt(q1, q0, beta));
        let strong = diff_lt(p0, q0, vdupq_n_s16(((alpha >> 2) + 2) as i16));
        let ap = vandq_u16(diff_lt(p2, p0, beta), strong);
        let aq = vandq_u16(diff_lt(q2, q0, beta), strong);
        let two = vdupq_n_s16(2);
        let four = vdupq_n_s16(4);
        let add = |a, b| vaddq_s16(a, b);
        let dbl = |a| vshlq_n_s16::<1>(a);
        let wp0 = vshrq_n_s16::<2>(add(add(dbl(p1), p0), add(q1, two)));
        let wq0 = vshrq_n_s16::<2>(add(add(dbl(q1), q0), add(p1, two)));
        let p0q0 = add(p0, q0);
        let sp0 = vshrq_n_s16::<3>(add(add(p2, dbl(add(p1, p0q0))), add(q1, four)));
        let sp1 = vshrq_n_s16::<2>(add(add(p2, p1), add(p0q0, two)));
        let sp2 = vshrq_n_s16::<3>(add(add(dbl(p3), add(p2, dbl(p2))), add(add(p1, p0q0), four)));
        let sq0 = vshrq_n_s16::<3>(add(add(p1, dbl(add(p0q0, q1))), add(q2, four)));
        let sq1 = vshrq_n_s16::<2>(add(add(p0q0, q1), add(q2, two)));
        let sq2 = vshrq_n_s16::<3>(add(add(dbl(q3), add(q2, dbl(q2))), add(add(q1, p0q0), four)));
        let np0 = vbslq_s16(ap, sp0, wp0);
        let np1 = vbslq_s16(ap, sp1, p1);
        let np2 = vbslq_s16(ap, sp2, p2);
        let nq0 = vbslq_s16(aq, sq0, wq0);
        let nq1 = vbslq_s16(aq, sq1, q1);
        let nq2 = vbslq_s16(aq, sq2, q2);
        v[1] = vbslq_s16(mask, np2, p2);
        v[2] = vbslq_s16(mask, np1, p1);
        v[3] = vbslq_s16(mask, np0, p0);
        v[4] = vbslq_s16(mask, nq0, q0);
        v[5] = vbslq_s16(mask, nq1, q1);
        v[6] = vbslq_s16(mask, nq2, q2);
    }
}

/// tC0 per lane for lines `8 * half .. 8 * half + 8` (four per segment).
#[inline(always)]
unsafe fn tc0_luma(tc0: &[i16; 4], half: usize) -> int16x8_t {
    unsafe {
        let a = tc0[2 * half] as i16;
        let b = tc0[2 * half + 1] as i16;
        let t = [a, a, a, a, b, b, b, b];
        vld1q_s16(t.as_ptr())
    }
}

/// Transpose an 8x8 block of bytes held as eight row vectors.
#[inline(always)]
unsafe fn transpose8x8_u8(r: &mut [uint8x8_t; 8]) {
    unsafe {
        let t0 = vtrn_u8(r[0], r[1]);
        let t1 = vtrn_u8(r[2], r[3]);
        let t2 = vtrn_u8(r[4], r[5]);
        let t3 = vtrn_u8(r[6], r[7]);
        let u0 = vtrn_u16(vreinterpret_u16_u8(t0.0), vreinterpret_u16_u8(t1.0));
        let u1 = vtrn_u16(vreinterpret_u16_u8(t0.1), vreinterpret_u16_u8(t1.1));
        let u2 = vtrn_u16(vreinterpret_u16_u8(t2.0), vreinterpret_u16_u8(t3.0));
        let u3 = vtrn_u16(vreinterpret_u16_u8(t2.1), vreinterpret_u16_u8(t3.1));
        let w0 = vtrn_u32(vreinterpret_u32_u16(u0.0), vreinterpret_u32_u16(u2.0));
        let w1 = vtrn_u32(vreinterpret_u32_u16(u1.0), vreinterpret_u32_u16(u3.0));
        let w2 = vtrn_u32(vreinterpret_u32_u16(u0.1), vreinterpret_u32_u16(u2.1));
        let w3 = vtrn_u32(vreinterpret_u32_u16(u1.1), vreinterpret_u32_u16(u3.1));
        r[0] = vreinterpret_u8_u32(w0.0);
        r[1] = vreinterpret_u8_u32(w1.0);
        r[2] = vreinterpret_u8_u32(w2.0);
        r[3] = vreinterpret_u8_u32(w3.0);
        r[4] = vreinterpret_u8_u32(w0.1);
        r[5] = vreinterpret_u8_u32(w1.1);
        r[6] = vreinterpret_u8_u32(w2.1);
        r[7] = vreinterpret_u8_u32(w3.1);
    }
}

/// Load 16 rows x 8 bytes around a vertical edge (`q0` at `data`) as eight
/// column vectors of sixteen bytes, split into two i16 halves each.
#[inline(always)]
unsafe fn load_transposed_16x8(data: *const u8, stride: usize) -> [Lines8; 2] {
    unsafe {
        let mut a = [vdup_n_u8(0); 8];
        let mut b = [vdup_n_u8(0); 8];
        for i in 0..8 {
            a[i] = vld1_u8(data.add(i * stride).sub(4));
            b[i] = vld1_u8(data.add((i + 8) * stride).sub(4));
        }
        transpose8x8_u8(&mut a);
        transpose8x8_u8(&mut b);
        let mut lo: Lines8 = [vdupq_n_s16(0); 8];
        let mut hi: Lines8 = [vdupq_n_s16(0); 8];
        for c in 0..8 {
            lo[c] = vreinterpretq_s16_u16(vmovl_u8(a[c]));
            hi[c] = vreinterpretq_s16_u16(vmovl_u8(b[c]));
        }
        [lo, hi]
    }
}

/// Store eight column vectors (two i16 halves each) back as 16 rows x 8 bytes.
#[inline(always)]
unsafe fn store_transposed_16x8(data: *mut u8, stride: usize, v: &[Lines8; 2]) {
    unsafe {
        let mut a = [vdup_n_u8(0); 8];
        let mut b = [vdup_n_u8(0); 8];
        for c in 0..8 {
            a[c] = vqmovun_s16(v[0][c]);
            b[c] = vqmovun_s16(v[1][c]);
        }
        transpose8x8_u8(&mut a);
        transpose8x8_u8(&mut b);
        for i in 0..8 {
            vst1_u8(data.add(i * stride).sub(4), a[i]);
            vst1_u8(data.add((i + 8) * stride).sub(4), b[i]);
        }
    }
}

fn deblock_luma_v_neon(data: &mut [u8], off: usize, stride: usize, alpha: i32, beta: i32, tc0: &[i16; 4], _max: i32) {
    if tc0.iter().all(|&t| t < 0) {
        return;
    }
    assert!(off >= 4 && off + 15 * stride + 4 <= data.len());
    unsafe {
        let p = data.as_mut_ptr().add(off);
        let mut v = load_transposed_16x8(p, stride);
        luma_filter_normal(&mut v[0], alpha, beta, tc0_luma(tc0, 0));
        luma_filter_normal(&mut v[1], alpha, beta, tc0_luma(tc0, 1));
        store_transposed_16x8(p, stride, &v);
    }
}

fn deblock_luma_v_intra_neon(data: &mut [u8], off: usize, stride: usize, alpha: i32, beta: i32, _max: i32) {
    assert!(off >= 4 && off + 15 * stride + 4 <= data.len());
    unsafe {
        let p = data.as_mut_ptr().add(off);
        let mut v = load_transposed_16x8(p, stride);
        luma_filter_intra(&mut v[0], alpha, beta);
        luma_filter_intra(&mut v[1], alpha, beta);
        store_transposed_16x8(p, stride, &v);
    }
}

/// Load a 16-byte row as two i16 halves.
#[inline(always)]
unsafe fn ld16(p: *const u8) -> (int16x8_t, int16x8_t) {
    unsafe {
        let v = vld1q_u8(p);
        (vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(v))), vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(v))))
    }
}

#[inline(always)]
unsafe fn st16(p: *mut u8, lo: int16x8_t, hi: int16x8_t) {
    unsafe { vst1q_u8(p, vcombine_u8(vqmovun_s16(lo), vqmovun_s16(hi))) }
}

fn deblock_luma_h_neon(data: &mut [u8], off: usize, stride: usize, alpha: i32, beta: i32, tc0: &[i16; 4], _max: i32) {
    if tc0.iter().all(|&t| t < 0) {
        return;
    }
    assert!(off >= 3 * stride && off + 2 * stride + 16 <= data.len());
    unsafe {
        let p = data.as_mut_ptr().add(off);
        let mut lo: Lines8 = [vdupq_n_s16(0); 8];
        let mut hi: Lines8 = [vdupq_n_s16(0); 8];
        for k in 1..7 {
            let (l, h) = ld16(p.offset((k as isize - 4) * stride as isize));
            lo[k] = l;
            hi[k] = h;
        }
        luma_filter_normal(&mut lo, alpha, beta, tc0_luma(tc0, 0));
        luma_filter_normal(&mut hi, alpha, beta, tc0_luma(tc0, 1));
        for k in 2..6 {
            st16(p.offset((k as isize - 4) * stride as isize), lo[k], hi[k]);
        }
    }
}

fn deblock_luma_h_intra_neon(data: &mut [u8], off: usize, stride: usize, alpha: i32, beta: i32, _max: i32) {
    assert!(off >= 4 * stride && off + 3 * stride + 16 <= data.len());
    unsafe {
        let p = data.as_mut_ptr().add(off);
        let mut lo: Lines8 = [vdupq_n_s16(0); 8];
        let mut hi: Lines8 = [vdupq_n_s16(0); 8];
        for k in 0..8 {
            let (l, h) = ld16(p.offset((k as isize - 4) * stride as isize));
            lo[k] = l;
            hi[k] = h;
        }
        luma_filter_intra(&mut lo, alpha, beta);
        luma_filter_intra(&mut hi, alpha, beta);
        for k in 1..7 {
            st16(p.offset((k as isize - 4) * stride as isize), lo[k], hi[k]);
        }
    }
}

// Chroma: eight lines, `[p1, p0, q0, q1]`.
type ChromaLines = [int16x8_t; 4];

#[inline(always)]
unsafe fn chroma_filter_normal(v: &mut ChromaLines, alpha: i32, beta: i32, tc0v: int16x8_t) {
    unsafe {
        let [p1, p0, q0, q1] = *v;
        let alpha = vdupq_n_s16(alpha as i16);
        let beta = vdupq_n_s16(beta as i16);
        let zero = vdupq_n_s16(0);
        let bs_on = vcgtq_s16(tc0v, vdupq_n_s16(-1));
        let mask = vandq_u16(vandq_u16(diff_lt(p0, q0, alpha), diff_lt(p1, p0, beta)), vandq_u16(diff_lt(q1, q0, beta), bs_on));
        let tc = vaddq_s16(tc0v, vdupq_n_s16(1));
        let d = vshrq_n_s16::<3>(vaddq_s16(vaddq_s16(vshlq_n_s16::<2>(vsubq_s16(q0, p0)), vsubq_s16(p1, q1)), vdupq_n_s16(4)));
        let d = vminq_s16(vmaxq_s16(d, vnegq_s16(tc)), tc);
        let clip = |x: int16x8_t| vminq_s16(vmaxq_s16(x, zero), vdupq_n_s16(255));
        v[1] = vbslq_s16(mask, clip(vaddq_s16(p0, d)), p0);
        v[2] = vbslq_s16(mask, clip(vsubq_s16(q0, d)), q0);
    }
}

#[inline(always)]
unsafe fn chroma_filter_intra(v: &mut ChromaLines, alpha: i32, beta: i32) {
    unsafe {
        let [p1, p0, q0, q1] = *v;
        let alpha = vdupq_n_s16(alpha as i16);
        let beta = vdupq_n_s16(beta as i16);
        let mask = vandq_u16(vandq_u16(diff_lt(p0, q0, alpha), diff_lt(p1, p0, beta)), diff_lt(q1, q0, beta));
        let two = vdupq_n_s16(2);
        let np0 = vshrq_n_s16::<2>(vaddq_s16(vaddq_s16(vshlq_n_s16::<1>(p1), p0), vaddq_s16(q1, two)));
        let nq0 = vshrq_n_s16::<2>(vaddq_s16(vaddq_s16(vshlq_n_s16::<1>(q1), q0), vaddq_s16(p1, two)));
        v[1] = vbslq_s16(mask, np0, p0);
        v[2] = vbslq_s16(mask, nq0, q0);
    }
}

#[inline(always)]
unsafe fn tc0_chroma(tc0: &[i16; 4]) -> int16x8_t {
    unsafe {
        let t = [tc0[0] as i16, tc0[0] as i16, tc0[1] as i16, tc0[1] as i16, tc0[2] as i16, tc0[2] as i16, tc0[3] as i16, tc0[3] as i16];
        vld1q_s16(t.as_ptr())
    }
}

/// Load 8 rows x 4 bytes (p1 p0 q0 q1) around a vertical chroma edge as
/// four column vectors.
#[inline(always)]
unsafe fn load_transposed_8x4(data: *const u8, stride: usize) -> ChromaLines {
    unsafe {
        let mut rows = [0u8; 32];
        for i in 0..8 {
            std::ptr::copy_nonoverlapping(data.add(i * stride).sub(2), rows.as_mut_ptr().add(4 * i), 4);
        }
        // vld4 de-interleaves the four positions across the eight rows.
        let cols = vld4_u8(rows.as_ptr());
        [
            vreinterpretq_s16_u16(vmovl_u8(cols.0)),
            vreinterpretq_s16_u16(vmovl_u8(cols.1)),
            vreinterpretq_s16_u16(vmovl_u8(cols.2)),
            vreinterpretq_s16_u16(vmovl_u8(cols.3)),
        ]
    }
}

#[inline(always)]
unsafe fn store_transposed_8x4(data: *mut u8, stride: usize, v: &ChromaLines) {
    unsafe {
        let p0 = vqmovun_s16(v[1]);
        let q0 = vqmovun_s16(v[2]);
        let pq = vzip_u8(p0, q0); // p0r0 q0r0 p0r1 q0r1 ... (two vectors of 8)
        let mut t = [0u8; 16];
        vst1_u8(t.as_mut_ptr(), pq.0);
        vst1_u8(t.as_mut_ptr().add(8), pq.1);
        for i in 0..8 {
            let d = data.add(i * stride).sub(1);
            *d = t[2 * i];
            *d.add(1) = t[2 * i + 1];
        }
    }
}

fn deblock_chroma_v_neon(data: &mut [u8], off: usize, stride: usize, alpha: i32, beta: i32, tc0: &[i16; 4], _max: i32) {
    if tc0.iter().all(|&t| t < 0) {
        return;
    }
    assert!(off >= 2 && off + 7 * stride + 2 <= data.len());
    unsafe {
        let p = data.as_mut_ptr().add(off);
        let mut v = load_transposed_8x4(p, stride);
        chroma_filter_normal(&mut v, alpha, beta, tc0_chroma(tc0));
        store_transposed_8x4(p, stride, &v);
    }
}

fn deblock_chroma_v_intra_neon(data: &mut [u8], off: usize, stride: usize, alpha: i32, beta: i32, _max: i32) {
    assert!(off >= 2 && off + 7 * stride + 2 <= data.len());
    unsafe {
        let p = data.as_mut_ptr().add(off);
        let mut v = load_transposed_8x4(p, stride);
        chroma_filter_intra(&mut v, alpha, beta);
        store_transposed_8x4(p, stride, &v);
    }
}

fn deblock_chroma_h_neon(data: &mut [u8], off: usize, stride: usize, alpha: i32, beta: i32, tc0: &[i16; 4], _max: i32) {
    if tc0.iter().all(|&t| t < 0) {
        return;
    }
    assert!(off >= 2 * stride && off + stride + 8 <= data.len());
    unsafe {
        let p = data.as_mut_ptr().add(off);
        let mut v: ChromaLines = [load8(p.sub(2 * stride)), load8(p.sub(stride)), load8(p), load8(p.add(stride))];
        chroma_filter_normal(&mut v, alpha, beta, tc0_chroma(tc0));
        vst1_u8(p.sub(stride), vqmovun_s16(v[1]));
        vst1_u8(p, vqmovun_s16(v[2]));
    }
}

fn deblock_chroma_h_intra_neon(data: &mut [u8], off: usize, stride: usize, alpha: i32, beta: i32, _max: i32) {
    assert!(off >= 2 * stride && off + stride + 8 <= data.len());
    unsafe {
        let p = data.as_mut_ptr().add(off);
        let mut v: ChromaLines = [load8(p.sub(2 * stride)), load8(p.sub(stride)), load8(p), load8(p.add(stride))];
        chroma_filter_intra(&mut v, alpha, beta);
        vst1_u8(p.sub(stride), vqmovun_s16(v[1]));
        vst1_u8(p, vqmovun_s16(v[2]));
    }
}

// ----------------------------------------------------------------------
// Inverse transforms
// ----------------------------------------------------------------------

/// Add `(v + 32) >> 6` to `n` (4 or 8) samples of a row, clipping.
#[inline(always)]
unsafe fn add_row(dst: *mut u8, v: int16x8_t, n: usize) {
    unsafe {
        let r = vshrq_n_s16::<6>(vaddq_s16(v, vdupq_n_s16(32)));
        if n == 4 {
            let mut t = [0u8; 8];
            std::ptr::copy_nonoverlapping(dst, t.as_mut_ptr(), 4);
            let p = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(t.as_ptr())));
            vst1_u8(t.as_mut_ptr(), vqmovun_s16(vaddq_s16(p, r)));
            std::ptr::copy_nonoverlapping(t.as_ptr(), dst, 4);
        } else {
            let p = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(dst)));
            vst1_u8(dst, vqmovun_s16(vaddq_s16(p, r)));
        }
    }
}

/// Transpose a 4x4 of i16 held in the low halves of four vectors.
#[inline(always)]
unsafe fn transpose4(r: &mut [int16x8_t; 4]) {
    unsafe {
        let a = vtrnq_s16(r[0], r[1]);
        let b = vtrnq_s16(r[2], r[3]);
        let c = vtrnq_s32(vreinterpretq_s32_s16(a.0), vreinterpretq_s32_s16(b.0));
        let d = vtrnq_s32(vreinterpretq_s32_s16(a.1), vreinterpretq_s32_s16(b.1));
        r[0] = vreinterpretq_s16_s32(c.0);
        r[1] = vreinterpretq_s16_s32(d.0);
        r[2] = vreinterpretq_s16_s32(c.1);
        r[3] = vreinterpretq_s16_s32(d.1);
    }
}

fn idct4_add_neon(dst: &mut [u8], stride: usize, coeffs: &[i16; 16], _max: i32) {
    assert!(3 * stride + 4 <= dst.len());
    unsafe { idct4_add_impl(dst.as_mut_ptr(), stride, coeffs) }
}

#[inline(always)]
unsafe fn idct4_add_impl(dst: *mut u8, stride: usize, c: &[i16; 16]) {
    unsafe {
        let ld = |i: usize| vcombine_s16(vld1_s16(c.as_ptr().add(4 * i)), vdup_n_s16(0));
        let mut r = [ld(0), ld(1), ld(2), ld(3)];
        // Row pass on the transposed block (columns as vectors).
        transpose4(&mut r);
        let [c0, c1, c2, c3] = r;
        let e0 = vaddq_s16(c0, c2);
        let e1 = vsubq_s16(c0, c2);
        let e2 = vsubq_s16(vshrq_n_s16::<1>(c1), c3);
        let e3 = vaddq_s16(c1, vshrq_n_s16::<1>(c3));
        let mut f = [vaddq_s16(e0, e3), vaddq_s16(e1, e2), vsubq_s16(e1, e2), vsubq_s16(e0, e3)];
        transpose4(&mut f);
        let [r0, r1, r2, r3] = f;
        let g0 = vaddq_s16(r0, r2);
        let g1 = vsubq_s16(r0, r2);
        let g2 = vsubq_s16(vshrq_n_s16::<1>(r1), r3);
        let g3 = vaddq_s16(r1, vshrq_n_s16::<1>(r3));
        add_row(dst, vaddq_s16(g0, g3), 4);
        add_row(dst.add(stride), vaddq_s16(g1, g2), 4);
        add_row(dst.add(2 * stride), vsubq_s16(g1, g2), 4);
        add_row(dst.add(3 * stride), vsubq_s16(g0, g3), 4);
    }
}

/// Transpose eight 8-lane i16 rows.
#[inline(always)]
unsafe fn transpose8(r: &mut [int16x8_t; 8]) {
    unsafe {
        let a0 = vtrnq_s16(r[0], r[1]);
        let a1 = vtrnq_s16(r[2], r[3]);
        let a2 = vtrnq_s16(r[4], r[5]);
        let a3 = vtrnq_s16(r[6], r[7]);
        let b0 = vtrnq_s32(vreinterpretq_s32_s16(a0.0), vreinterpretq_s32_s16(a1.0));
        let b1 = vtrnq_s32(vreinterpretq_s32_s16(a0.1), vreinterpretq_s32_s16(a1.1));
        let b2 = vtrnq_s32(vreinterpretq_s32_s16(a2.0), vreinterpretq_s32_s16(a3.0));
        let b3 = vtrnq_s32(vreinterpretq_s32_s16(a2.1), vreinterpretq_s32_s16(a3.1));
        // 64-bit halves: combine low/high halves across the b pairs.
        let lo = |x: int32x4_t| vget_low_s32(x);
        let hi = |x: int32x4_t| vget_high_s32(x);
        r[0] = vreinterpretq_s16_s32(vcombine_s32(lo(b0.0), lo(b2.0)));
        r[1] = vreinterpretq_s16_s32(vcombine_s32(lo(b1.0), lo(b3.0)));
        r[2] = vreinterpretq_s16_s32(vcombine_s32(lo(b0.1), lo(b2.1)));
        r[3] = vreinterpretq_s16_s32(vcombine_s32(lo(b1.1), lo(b3.1)));
        r[4] = vreinterpretq_s16_s32(vcombine_s32(hi(b0.0), hi(b2.0)));
        r[5] = vreinterpretq_s16_s32(vcombine_s32(hi(b1.0), hi(b3.0)));
        r[6] = vreinterpretq_s16_s32(vcombine_s32(hi(b0.1), hi(b2.1)));
        r[7] = vreinterpretq_s16_s32(vcombine_s32(hi(b1.1), hi(b3.1)));
    }
}

/// One 8-point pass (8.5.13.2) across eight vectors.
#[inline(always)]
unsafe fn idct8_pass(d: &[int16x8_t; 8]) -> [int16x8_t; 8] {
    unsafe {
        let add = |a, b| vaddq_s16(a, b);
        let sub = |a, b| vsubq_s16(a, b);
        let sh1 = |a| vshrq_n_s16::<1>(a);
        let sh2 = |a| vshrq_n_s16::<2>(a);
        let a0 = add(d[0], d[4]);
        let a4 = sub(d[0], d[4]);
        let a2 = sub(sh1(d[2]), d[6]);
        let a6 = add(d[2], sh1(d[6]));
        let b0 = add(a0, a6);
        let b2 = add(a4, a2);
        let b4 = sub(a4, a2);
        let b6 = sub(a0, a6);
        let a1 = sub(sub(sub(d[5], d[3]), d[7]), sh1(d[7]));
        let a3 = sub(sub(add(d[1], d[7]), d[3]), sh1(d[3]));
        let a5 = add(add(sub(d[7], d[1]), d[5]), sh1(d[5]));
        let a7 = add(add(add(d[3], d[5]), d[1]), sh1(d[1]));
        let b1 = add(a1, sh2(a7));
        let b7 = sub(a7, sh2(a1));
        let b3 = add(a3, sh2(a5));
        let b5 = sub(sh2(a3), a5);
        [add(b0, b7), add(b2, b5), add(b4, b3), add(b6, b1), sub(b6, b1), sub(b4, b3), sub(b2, b5), sub(b0, b7)]
    }
}

fn idct8_add_neon(dst: &mut [u8], stride: usize, coeffs: &[i16; 64], _max: i32) {
    assert!(7 * stride + 8 <= dst.len());
    unsafe { idct8_add_impl(dst.as_mut_ptr(), stride, coeffs) }
}

#[inline(always)]
unsafe fn idct8_add_impl(dst: *mut u8, stride: usize, c: &[i16; 64]) {
    unsafe {
        let mut r = [vdupq_n_s16(0); 8];
        for i in 0..8 {
            r[i] = vld1q_s16(c.as_ptr().add(8 * i));
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

fn idct4_dc_add_neon(dst: &mut [u8], stride: usize, dc: i32, _max: i32) {
    assert!(3 * stride + 4 <= dst.len());
    unsafe {
        let v = vdupq_n_s16(dc as i16);
        for i in 0..4 {
            add_row(dst.as_mut_ptr().add(i * stride), v, 4);
        }
    }
}

fn idct8_dc_add_neon(dst: &mut [u8], stride: usize, dc: i32, _max: i32) {
    assert!(7 * stride + 8 <= dst.len());
    unsafe {
        let v = vdupq_n_s16(dc as i16);
        for i in 0..8 {
            add_row(dst.as_mut_ptr().add(i * stride), v, 8);
        }
    }
}

/// Dequantise sixteen levels to sixteen i16 (two vectors), saturating.
#[inline(always)]
unsafe fn dequant16(levels: *const i32, scale: *const i32, up: bool, sh: i32, round: i32) -> (int16x8_t, int16x8_t) {
    unsafe {
        let mut out = [vdup_n_s16(0); 4];
        for k in 0..4 {
            let l = vld1q_s32(levels.add(4 * k));
            let s = vld1q_s32(scale.add(4 * k));
            let v = vmulq_s32(l, s);
            let v = if up { vshlq_s32(v, vdupq_n_s32(sh)) } else { vshlq_s32(vaddq_s32(v, vdupq_n_s32(round)), vdupq_n_s32(-sh)) };
            out[k] = vqmovn_s32(v);
        }
        (vcombine_s16(out[0], out[1]), vcombine_s16(out[2], out[3]))
    }
}

fn residual4_neon(dst: &mut [u8], stride: usize, levels: &[i32; 16], scale: &[i32; 16], qp: i32, dc: i32, _max: i32) {
    assert!(3 * stride + 4 <= dst.len());
    unsafe {
        let q6 = qp / 6;
        let (mut lo, hi) = if qp >= 24 {
            dequant16(levels.as_ptr(), scale.as_ptr(), true, q6 - 4, 0)
        } else {
            dequant16(levels.as_ptr(), scale.as_ptr(), false, 4 - q6, 1 << (3 - q6))
        };
        if dc != NO_DC {
            lo = vsetq_lane_s16::<0>(dc as i16, lo);
        }
        let ac = vorrq_s16(vsetq_lane_s16::<0>(0, lo), hi);
        if vmaxvq_u16(vreinterpretq_u16_s16(ac)) == 0 {
            let d = vgetq_lane_s16::<0>(lo) as i32;
            if d != 0 {
                idct4_dc_add_neon(dst, stride, d, 255);
            }
            return;
        }
        let mut coeffs = [0i16; 16];
        vst1q_s16(coeffs.as_mut_ptr(), lo);
        vst1q_s16(coeffs.as_mut_ptr().add(8), hi);
        idct4_add_impl(dst.as_mut_ptr(), stride, &coeffs);
    }
}

fn residual8_neon(dst: &mut [u8], stride: usize, levels: &[i32; 64], scale: &[i32; 64], qp: i32, _max: i32) {
    assert!(7 * stride + 8 <= dst.len());
    unsafe {
        let q6 = qp / 6;
        let (up, sh, round) = if qp >= 36 { (true, q6 - 6, 0) } else { (false, 6 - q6, 1 << (5 - q6)) };
        let mut coeffs = [0i16; 64];
        let mut ac = vdupq_n_s16(0);
        for k in 0..4 {
            let (lo, hi) = dequant16(levels.as_ptr().add(16 * k), scale.as_ptr().add(16 * k), up, sh, round);
            vst1q_s16(coeffs.as_mut_ptr().add(16 * k), lo);
            vst1q_s16(coeffs.as_mut_ptr().add(16 * k + 8), hi);
            let lo_ac = if k == 0 { vsetq_lane_s16::<0>(0, lo) } else { lo };
            ac = vorrq_s16(ac, vorrq_s16(lo_ac, hi));
        }
        if vmaxvq_u16(vreinterpretq_u16_s16(ac)) == 0 {
            let d = coeffs[0] as i32;
            if d != 0 {
                idct8_dc_add_neon(dst, stride, d, 255);
            }
            return;
        }
        idct8_add_impl(dst.as_mut_ptr(), stride, &coeffs);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: &mut u64) -> u32 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (*seed >> 33) as u32
    }

    #[test]
    fn kernels_match_scalar() {
        let mut d = H264Dsp::<u8>::SCALAR;
        install(&mut d);
        let s = H264Dsp::<u8>::SCALAR;
        let mut seed = 5u64;
        let stride = 64;
        let src: Vec<u8> = (0..stride * 64).map(|_| lcg(&mut seed) as u8).collect();
        for &(w, h) in &[(4usize, 4usize), (4, 8), (8, 4), (8, 8), (8, 16), (16, 8), (16, 16)] {
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
        let mut d = H264Dsp::<u8>::SCALAR;
        install(&mut d);
        let s = H264Dsp::<u8>::SCALAR;
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
        let mut d = H264Dsp::<u8>::SCALAR;
        install(&mut d);
        let s = H264Dsp::<u8>::SCALAR;
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
