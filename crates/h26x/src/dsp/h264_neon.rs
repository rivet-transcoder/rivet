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

use super::h264::H264Dsp;

/// Replace the scalar entries of `d` with the NEON kernels.
pub fn install(d: &mut H264Dsp) {
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

fn qpel_neon<const XF: usize, const YF: usize>(dst: &mut [u8], src: &[u8], stride: usize, w: usize, h: usize) {
    // 8-lane loads up to column x0 + 5 + 7 of the last row used.
    let need = (h + 5 - 1) * stride + w.div_ceil(8) * 8 + 5 + 8;
    if src.len() < need {
        return (H264Dsp::SCALAR.qpel[YF * 4 + XF])(dst, src, stride, w, h);
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
                store8_n(dst.as_mut_ptr().add(y * w + x0), v, n);
                x0 += 8;
            }
        }
    }
}

fn chroma_neon(dst: &mut [u8], src: &[u8], stride: usize, w: usize, h: usize, xf: i32, yf: i32) {
    if src.len() < h * stride + w.div_ceil(8) * 8 + 9 {
        return (H264Dsp::SCALAR.chroma)(dst, src, stride, w, h, xf, yf);
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
                store8_n(dst.as_mut_ptr().add(y * w + x0), p, n);
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
                let mut ta = [0u8; 8];
                let mut tb = [0u8; 8];
                std::ptr::copy_nonoverlapping(a.as_ptr().add(y * w + x0), ta.as_mut_ptr(), n);
                std::ptr::copy_nonoverlapping(b.as_ptr().add(y * w + x0), tb.as_mut_ptr(), n);
                let v = vrhadd_u8(vld1_u8(ta.as_ptr()), vld1_u8(tb.as_ptr()));
                store8_n(dst.as_mut_ptr().add(y * stride + x0), v, n);
                x0 += 8;
            }
        }
    }
}

fn weighted_uni_neon(dst: &mut [u8], stride: usize, src: &[u8], w: usize, h: usize, log_wd: i32, wt: i32, o: i32) {
    unsafe {
        let round = vdupq_n_s16(if log_wd >= 1 { 1 << (log_wd - 1) } else { 0 });
        let sh = vdupq_n_s16(-(log_wd.max(0) as i16));
        let ov = vdupq_n_s16(o as i16);
        for y in 0..h {
            let mut x0 = 0;
            while x0 < w {
                let n = (w - x0).min(8);
                let mut t = [0u8; 8];
                std::ptr::copy_nonoverlapping(src.as_ptr().add(y * w + x0), t.as_mut_ptr(), n);
                let s = load8(t.as_ptr());
                let v = vaddq_s16(vshlq_s16(vaddq_s16(vmulq_n_s16(s, wt as i16), round), sh), ov);
                store8_n(dst.as_mut_ptr().add(y * stride + x0), vqmovun_s16(v), n);
                x0 += 8;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn weighted_bi_neon(dst: &mut [u8], stride: usize, a: &[u8], b: &[u8], w: usize, h: usize, log_wd: i32, w0: i32, w1: i32, o0: i32, o1: i32) {
    unsafe {
        let round = vdupq_n_s32(1 << log_wd);
        let off = vdupq_n_s32((o0 + o1 + 1) >> 1);
        let sh = vdupq_n_s32(-(log_wd + 1));
        for y in 0..h {
            let mut x0 = 0;
            while x0 < w {
                let n = (w - x0).min(8);
                let mut ta = [0u8; 8];
                let mut tb = [0u8; 8];
                std::ptr::copy_nonoverlapping(a.as_ptr().add(y * w + x0), ta.as_mut_ptr(), n);
                std::ptr::copy_nonoverlapping(b.as_ptr().add(y * w + x0), tb.as_mut_ptr(), n);
                let va = load8(ta.as_ptr());
                let vb = load8(tb.as_ptr());
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

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: &mut u64) -> u32 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (*seed >> 33) as u32
    }

    #[test]
    fn kernels_match_scalar() {
        let mut d = H264Dsp::SCALAR;
        install(&mut d);
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
