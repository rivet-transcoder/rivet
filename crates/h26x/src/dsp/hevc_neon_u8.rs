//! NEON versions of the H.265 kernels for 8-bit sample planes (AArch64).
//!
//! Sixteen 8-bit lanes per vector. Interpolation multiplies the bytes by
//! each tap's magnitude straight into 16-bit accumulators (`vmlal_u8` for
//! positive taps, `vmlsl_u8` for negative ones): the HEVC filters cannot
//! overflow 16 bits for 8-bit input, and two's-complement wrap makes the
//! mixed-sign sum exact — eight outputs per multiply, twice the width of
//! the 16-bit-sample kernels. Narrowing to bytes uses the saturating
//! rounding shifts, which are exactly the standard's `(v + round) >> shift`
//! and clip. The second (16-bit) stage of hv, the inverse transform and the
//! deblocking arithmetic are sample-size independent and shared with
//! [`super::hevc_neon`]. Every kernel is checked bit-exact against the
//! scalar reference by the tests (run on AArch64 in CI).

#![cfg(target_arch = "aarch64")]

use std::arch::aarch64::*;

use super::hevc::HevcDsp;
use super::hevc_neon as w16;
use crate::hevc::tables::{EPEL_FILTERS, QPEL_FILTERS};

/// Replace the scalar entries of `d` with the NEON kernels.
pub fn install(d: &mut HevcDsp<u8>) {
    d.idct = [w16::idct_neon::<4>, w16::idct_neon::<8>, w16::idct_neon::<16>, w16::idct_neon::<32>];
    d.add_residual = add_residual_neon;
    d.qpel_copy = copy_neon;
    d.qpel_h = qpel_h_neon;
    d.qpel_v = qpel_v_neon;
    d.qpel_v2 = w16::qpel_v2_neon;
    d.epel_copy = copy_neon;
    d.epel_h = epel_h_neon;
    d.epel_v = epel_v_neon;
    d.epel_v2 = w16::epel_v2_neon;
    d.uni = uni_neon;
    d.bi = bi_neon;
    d.weighted_uni = weighted_uni_neon;
    d.weighted_bi = weighted_bi_neon;
    d.qpel_uni = qpel_uni_neon;
    d.epel_uni = epel_uni_neon;
    d.qpel_bi = qpel_bi_neon;
    d.epel_bi = epel_bi_neon;
    d.fused_mc = true;
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

/// Store the first `n` (≤ 8) bytes of `v`.
#[inline(always)]
unsafe fn store_bytes(dst: *mut u8, v: uint8x8_t, n: usize) {
    unsafe {
        match n {
            8 => vst1_u8(dst, v),
            4 => std::ptr::write_unaligned(dst as *mut u32, vget_lane_u32::<0>(vreinterpret_u32_u8(v))),
            2 => std::ptr::write_unaligned(dst as *mut u16, vget_lane_u16::<0>(vreinterpret_u16_u8(v))),
            _ => {
                let mut t = [0u8; 8];
                vst1_u8(t.as_mut_ptr(), v);
                std::ptr::copy_nonoverlapping(t.as_ptr(), dst, n);
            }
        }
    }
}

/// Store the first `n` (≤ 16) bytes of `v`.
#[inline(always)]
unsafe fn store_bytes16(dst: *mut u8, v: uint8x16_t, n: usize) {
    unsafe {
        if n == 16 {
            vst1q_u8(dst, v);
        } else if n > 8 {
            vst1_u8(dst, vget_low_u8(v));
            store_bytes(dst.add(8), vget_high_u8(v), n - 8);
        } else {
            store_bytes(dst, vget_low_u8(v), n);
        }
    }
}

/// Load 16 bytes, or the first `avail` zero-padded.
#[inline(always)]
unsafe fn load_bytes16(src: *const u8, avail: usize) -> uint8x16_t {
    unsafe {
        if avail >= 16 {
            vld1q_u8(src)
        } else {
            let mut t = [0u8; 16];
            std::ptr::copy_nonoverlapping(src, t.as_mut_ptr(), avail);
            vld1q_u8(t.as_ptr())
        }
    }
}

/// Whether reading `w` samples starting `x` into a row of `stride`, for
/// `rows` rows, plus `extra` samples along, stays inside `len` for the
/// vector width the kernels use at that block width.
#[inline(always)]
fn fits(len: usize, stride: usize, rows: usize, w: usize, extra: usize) -> bool {
    let vec = if w >= 16 { 16 } else { 8 };
    let last_x = if w == 0 { 0 } else { (w - 1) / vec * vec };
    (rows - 1) * stride + last_x + extra + vec <= len
}

/// Whether the second stage's `w`-stride 14-bit rows can be read 8 lanes at
/// a time for `rows` rows within `len`.
#[inline(always)]
fn fits_i16(len: usize, w: usize, rows: usize) -> bool {
    let last_x = if w == 0 { 0 } else { (w - 1) / 8 * 8 };
    (rows - 1) * w + last_x + 8 <= len
}

/// The taps of a filter split by sign: magnitudes of the positive and of
/// the negative ones (0 where the tap has the other sign or is zero).
struct Taps {
    pos: [u8; 8],
    neg: [u8; 8],
    /// Whether tap `k` is positive / negative (skips idle multiplies).
    is_pos: [bool; 8],
    is_neg: [bool; 8],
}

impl Taps {
    #[inline(always)]
    fn of(taps: &[i8]) -> Self {
        let mut t = Taps { pos: [0; 8], neg: [0; 8], is_pos: [false; 8], is_neg: [false; 8] };
        for (k, &c) in taps.iter().enumerate() {
            if c > 0 {
                t.pos[k] = c as u8;
                t.is_pos[k] = true;
            } else if c < 0 {
                t.neg[k] = (-(c as i16)) as u8;
                t.is_neg[k] = true;
            }
        }
        t
    }
}

// ----------------------------------------------------------------------
// Interpolation
// ----------------------------------------------------------------------

/// What a FIR stage produces, per output kind (`MODE_*`).
#[derive(Clone, Copy)]
struct Out {
    /// `MODE_I16`: 14-bit predictions, stride `w`.
    i16: *mut i16,
    /// `MODE_UNI` / `MODE_BI`: samples, stride `stride`.
    u8: *mut u8,
    /// Sample stride.
    stride: usize,
    /// `MODE_BI`: the other list's 14-bit prediction, stride `w`.
    other: *const i16,
    /// Block width (the stride of `i16` and `other`).
    w: usize,
}

/// 14-bit predictions (the two-pass path and the first stage of hv).
const MODE_I16: u8 = 0;
/// Default-weighted uni-prediction samples: `(v + 32) >> 6`.
const MODE_UNI: u8 = 1;
/// Default-weighted bi-prediction samples: `(v + other + 64) >> 7`.
const MODE_BI: u8 = 2;

/// Emit 8 lanes of a stage's output (`v`, 14-bit) at (`row`, `x`), the
/// first `n` lanes.
#[inline(always)]
unsafe fn emit<const MODE: u8>(out: &Out, row: usize, x: usize, v: int16x8_t, n: usize) {
    unsafe {
        match MODE {
            MODE_I16 => w16::store_n(out.i16.add(row * out.w + x), v, n),
            MODE_UNI => store_bytes(out.u8.add(row * out.stride + x), vqrshrun_n_s16::<6>(v), n),
            _ => {
                // Saturating sum: exact after the clip (see `bi_neon`).
                let o = w16::load_n(out.other.add(row * out.w + x), n);
                store_bytes(out.u8.add(row * out.stride + x), vqrshrun_n_s16::<7>(vqaddq_s16(v, o)), n);
            }
        }
    }
}

/// Eight taps' worth of one row (or column) of bytes into 8 x i16.
#[inline(always)]
unsafe fn tap8(acc: uint16x8_t, s: uint8x8_t, t: &Taps, k: usize) -> uint16x8_t {
    unsafe {
        let mut acc = acc;
        if t.is_pos[k] {
            acc = vmlal_u8(acc, s, vdup_n_u8(t.pos[k]));
        }
        if t.is_neg[k] {
            acc = vmlsl_u8(acc, s, vdup_n_u8(t.neg[k]));
        }
        acc
    }
}

/// Horizontal FIR with `TAPS` taps over bytes.
#[inline(always)]
unsafe fn fir_h<const TAPS: usize, const MODE: u8>(out: &Out, src: *const u8, src_stride: usize, w: usize, h: usize, taps: &[i8], shift: i32) {
    unsafe {
        let t = Taps::of(taps);
        let sh = vdupq_n_s16(-(shift as i16));
        for y in 0..h {
            let s = src.add(y * src_stride);
            let mut x = 0;
            while x < w {
                let n = w - x;
                if n >= 16 {
                    let mut lo = vdupq_n_u16(0);
                    let mut hi = vdupq_n_u16(0);
                    for k in 0..TAPS {
                        let v = vld1q_u8(s.add(x + k));
                        lo = tap8(lo, vget_low_u8(v), &t, k);
                        hi = tap8(hi, vget_high_u8(v), &t, k);
                    }
                    emit::<MODE>(out, y, x, vshlq_s16(vreinterpretq_s16_u16(lo), sh), 8);
                    emit::<MODE>(out, y, x + 8, vshlq_s16(vreinterpretq_s16_u16(hi), sh), 8);
                    x += 16;
                } else {
                    let mut acc = vdupq_n_u16(0);
                    for k in 0..TAPS {
                        acc = tap8(acc, vld1_u8(s.add(x + k)), &t, k);
                    }
                    emit::<MODE>(out, y, x, vshlq_s16(vreinterpretq_s16_u16(acc), sh), n.min(8));
                    x += 8;
                }
            }
        }
    }
}

/// Vertical FIR with `TAPS` taps over byte rows.
#[inline(always)]
unsafe fn fir_v<const TAPS: usize, const MODE: u8>(out: &Out, src: *const u8, src_stride: usize, w: usize, h: usize, taps: &[i8], shift: i32) {
    unsafe {
        let t = Taps::of(taps);
        let sh = vdupq_n_s16(-(shift as i16));
        for y in 0..h {
            let mut x = 0;
            while x < w {
                let n = w - x;
                if n >= 16 {
                    let mut lo = vdupq_n_u16(0);
                    let mut hi = vdupq_n_u16(0);
                    for k in 0..TAPS {
                        let v = vld1q_u8(src.add((y + k) * src_stride + x));
                        lo = tap8(lo, vget_low_u8(v), &t, k);
                        hi = tap8(hi, vget_high_u8(v), &t, k);
                    }
                    emit::<MODE>(out, y, x, vshlq_s16(vreinterpretq_s16_u16(lo), sh), 8);
                    emit::<MODE>(out, y, x + 8, vshlq_s16(vreinterpretq_s16_u16(hi), sh), 8);
                    x += 16;
                } else {
                    let mut acc = vdupq_n_u16(0);
                    for k in 0..TAPS {
                        acc = tap8(acc, vld1_u8(src.add((y + k) * src_stride + x)), &t, k);
                    }
                    emit::<MODE>(out, y, x, vshlq_s16(vreinterpretq_s16_u16(acc), sh), n.min(8));
                    x += 8;
                }
            }
        }
    }
}

/// Vertical FIR with `TAPS` taps over 14-bit rows (the second stage of hv):
/// 32-bit sums, `>> 6`.
#[inline(always)]
unsafe fn fir_v2<const TAPS: usize, const MODE: u8>(out: &Out, src: *const i16, src_stride: usize, w: usize, h: usize, taps: &[i8]) {
    unsafe {
        let mut c = [0i16; 8];
        for k in 0..TAPS {
            c[k] = taps[k] as i16;
        }
        for y in 0..h {
            let mut x = 0;
            while x < w {
                let mut lo = vdupq_n_s32(0);
                let mut hi = vdupq_n_s32(0);
                for k in 0..TAPS {
                    let v = vld1q_s16(src.add((y + k) * src_stride + x));
                    lo = vmlal_n_s16(lo, vget_low_s16(v), c[k]);
                    hi = vmlal_high_n_s16(hi, v, c[k]);
                }
                emit::<MODE>(out, y, x, w16::narrow_shift(lo, hi, 6), (w - x).min(8));
                x += 8;
            }
        }
    }
}

fn copy_neon(dst: &mut [i16], src: &[u8], src_stride: usize, w: usize, h: usize, shift: i32) {
    if (h - 1) * src_stride + (w - 1) / 8 * 8 + 8 > src.len() || dst.len() < w * h {
        return (HevcDsp::<u8>::SCALAR.qpel_copy)(dst, src, src_stride, w, h, shift);
    }
    unsafe {
        let sh = vdupq_n_s16(shift as i16);
        for y in 0..h {
            let mut x = 0;
            while x < w {
                let v = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(src.as_ptr().add(y * src_stride + x))));
                w16::store_n(dst.as_mut_ptr().add(y * w + x), vshlq_s16(v, sh), (w - x).min(8));
                x += 8;
            }
        }
    }
}

fn qpel_h_neon(dst: &mut [i16], src: &[u8], src_stride: usize, w: usize, h: usize, frac: usize, shift: i32) {
    if !fits(src.len(), src_stride, h, w, 7) || dst.len() < w * h {
        return (HevcDsp::<u8>::SCALAR.qpel_h)(dst, src, src_stride, w, h, frac, shift);
    }
    let out = Out { i16: dst.as_mut_ptr(), u8: std::ptr::null_mut(), stride: 0, other: std::ptr::null(), w };
    unsafe { fir_h::<8, MODE_I16>(&out, src.as_ptr(), src_stride, w, h, &QPEL_FILTERS[frac][..8], shift) }
}

fn qpel_v_neon(dst: &mut [i16], src: &[u8], src_stride: usize, w: usize, h: usize, frac: usize, shift: i32) {
    if !fits(src.len(), src_stride, h + 7, w, 0) || dst.len() < w * h {
        return (HevcDsp::<u8>::SCALAR.qpel_v)(dst, src, src_stride, w, h, frac, shift);
    }
    let out = Out { i16: dst.as_mut_ptr(), u8: std::ptr::null_mut(), stride: 0, other: std::ptr::null(), w };
    unsafe { fir_v::<8, MODE_I16>(&out, src.as_ptr(), src_stride, w, h, &QPEL_FILTERS[frac][..8], shift) }
}

fn epel_h_neon(dst: &mut [i16], src: &[u8], src_stride: usize, w: usize, h: usize, frac: usize, shift: i32) {
    if !fits(src.len(), src_stride, h, w, 3) || dst.len() < w * h {
        return (HevcDsp::<u8>::SCALAR.epel_h)(dst, src, src_stride, w, h, frac, shift);
    }
    let out = Out { i16: dst.as_mut_ptr(), u8: std::ptr::null_mut(), stride: 0, other: std::ptr::null(), w };
    unsafe { fir_h::<4, MODE_I16>(&out, src.as_ptr(), src_stride, w, h, &EPEL_FILTERS[frac], shift) }
}

fn epel_v_neon(dst: &mut [i16], src: &[u8], src_stride: usize, w: usize, h: usize, frac: usize, shift: i32) {
    if !fits(src.len(), src_stride, h + 3, w, 0) || dst.len() < w * h {
        return (HevcDsp::<u8>::SCALAR.epel_v)(dst, src, src_stride, w, h, frac, shift);
    }
    let out = Out { i16: dst.as_mut_ptr(), u8: std::ptr::null_mut(), stride: 0, other: std::ptr::null(), w };
    unsafe { fir_v::<4, MODE_I16>(&out, src.as_ptr(), src_stride, w, h, &EPEL_FILTERS[frac], shift) }
}

// ----------------------------------------------------------------------
// Fused interpolation + prediction
// ----------------------------------------------------------------------

/// Copy a `w x h` byte block (whole-sample uni-prediction).
#[inline(always)]
unsafe fn copy_rows_u8(dst: *mut u8, dst_stride: usize, src: *const u8, src_stride: usize, w: usize, h: usize) {
    unsafe {
        for y in 0..h {
            let s = src.add(y * src_stride);
            let d = dst.add(y * dst_stride);
            let mut x = 0;
            while x < w {
                let n = w - x;
                if n >= 16 {
                    vst1q_u8(d.add(x), vld1q_u8(s.add(x)));
                    x += 16;
                } else if n >= 8 {
                    vst1_u8(d.add(x), vld1_u8(s.add(x)));
                    x += 8;
                } else if n >= 4 {
                    std::ptr::write_unaligned(d.add(x) as *mut u32, std::ptr::read_unaligned(s.add(x) as *const u32));
                    x += 4;
                } else {
                    std::ptr::write_unaligned(d.add(x) as *mut u16, std::ptr::read_unaligned(s.add(x) as *const u16));
                    x += 2;
                }
            }
        }
    }
}

/// The fused kernels: `TAPS` (8 luma / 4 chroma), `MODE_UNI` or `MODE_BI`.
#[allow(clippy::too_many_arguments)]
fn fused<const TAPS: usize, const MODE: u8>(dst: &mut [u8], dst_stride: usize, src: &[u8], src_stride: usize, w: usize, h: usize, fx: usize, fy: usize, tmp: &mut [i16], other: &[i16]) {
    let reach = TAPS / 2 - 1;
    let at_block = reach * src_stride + reach;
    let hh = h + TAPS - 1;
    let ok = w >= 2
        && h >= 1
        && (h - 1) * dst_stride + w <= dst.len()
        && (MODE != MODE_BI || other.len() >= w * h)
        && tmp.len() >= super::hevc::MC_TMP_LEN
        && match (fx, fy) {
            (0, 0) => (h - 1) * src_stride + w + at_block <= src.len(),
            (_, 0) => src.len() > reach * src_stride && fits(src.len() - reach * src_stride, src_stride, h, w, TAPS - 1),
            (0, _) => src.len() > reach && fits(src.len() - reach, src_stride, hh, w, 0),
            _ => fits(src.len(), src_stride, hh, w, TAPS - 1) && fits_i16(super::hevc::MC_TMP_LEN, w, hh),
        };
    if !ok {
        let s = HevcDsp::<u8>::SCALAR;
        return match (TAPS, MODE) {
            (8, MODE_UNI) => (s.qpel_uni)(dst, dst_stride, src, src_stride, w, h, fx, fy, tmp, 8),
            (8, _) => (s.qpel_bi)(dst, dst_stride, src, src_stride, w, h, fx, fy, tmp, other, 8),
            (_, MODE_UNI) => (s.epel_uni)(dst, dst_stride, src, src_stride, w, h, fx, fy, tmp, 8),
            _ => (s.epel_bi)(dst, dst_stride, src, src_stride, w, h, fx, fy, tmp, other, 8),
        };
    }
    let (tx, ty): (&[i8], &[i8]) = if TAPS == 8 { (&QPEL_FILTERS[fx][..8], &QPEL_FILTERS[fy][..8]) } else { (&EPEL_FILTERS[fx], &EPEL_FILTERS[fy]) };
    let out = Out { i16: std::ptr::null_mut(), u8: dst.as_mut_ptr(), stride: dst_stride, other: other.as_ptr(), w };
    unsafe {
        match (fx, fy) {
            (0, 0) => {
                if MODE == MODE_UNI {
                    copy_rows_u8(dst.as_mut_ptr(), dst_stride, src.as_ptr().add(at_block), src_stride, w, h);
                } else {
                    // Whole-sample bi: widen, then the usual average.
                    let (pred, _) = tmp.split_at_mut(w * h);
                    copy_neon(pred, &src[at_block..], src_stride, w, h, 6);
                    bi_neon(dst, dst_stride, other, pred, w, h, 7, 255);
                }
            }
            (_, 0) => fir_h::<TAPS, MODE>(&out, src.as_ptr().add(reach * src_stride), src_stride, w, h, tx, 0),
            (0, _) => fir_v::<TAPS, MODE>(&out, src.as_ptr().add(reach), src_stride, w, h, ty, 0),
            _ => {
                let mid = Out { i16: tmp.as_mut_ptr(), u8: std::ptr::null_mut(), stride: 0, other: std::ptr::null(), w };
                fir_h::<TAPS, MODE_I16>(&mid, src.as_ptr(), src_stride, w, hh, tx, 0);
                fir_v2::<TAPS, MODE>(&out, tmp.as_ptr(), w, w, h, ty);
            }
        }
    }
}

fn qpel_uni_neon(dst: &mut [u8], dst_stride: usize, src: &[u8], src_stride: usize, w: usize, h: usize, fx: usize, fy: usize, tmp: &mut [i16], bit_depth: u32) {
    debug_assert_eq!(bit_depth, 8);
    fused::<8, MODE_UNI>(dst, dst_stride, src, src_stride, w, h, fx, fy, tmp, &[])
}

fn epel_uni_neon(dst: &mut [u8], dst_stride: usize, src: &[u8], src_stride: usize, w: usize, h: usize, fx: usize, fy: usize, tmp: &mut [i16], bit_depth: u32) {
    debug_assert_eq!(bit_depth, 8);
    fused::<4, MODE_UNI>(dst, dst_stride, src, src_stride, w, h, fx, fy, tmp, &[])
}

#[allow(clippy::too_many_arguments)]
fn qpel_bi_neon(dst: &mut [u8], dst_stride: usize, src: &[u8], src_stride: usize, w: usize, h: usize, fx: usize, fy: usize, tmp: &mut [i16], other: &[i16], bit_depth: u32) {
    debug_assert_eq!(bit_depth, 8);
    fused::<8, MODE_BI>(dst, dst_stride, src, src_stride, w, h, fx, fy, tmp, other)
}

#[allow(clippy::too_many_arguments)]
fn epel_bi_neon(dst: &mut [u8], dst_stride: usize, src: &[u8], src_stride: usize, w: usize, h: usize, fx: usize, fy: usize, tmp: &mut [i16], other: &[i16], bit_depth: u32) {
    debug_assert_eq!(bit_depth, 8);
    fused::<4, MODE_BI>(dst, dst_stride, src, src_stride, w, h, fx, fy, tmp, other)
}

// ----------------------------------------------------------------------
// Combination / weighting
// ----------------------------------------------------------------------

fn uni_neon(dst: &mut [u8], stride: usize, src: &[i16], w: usize, h: usize, shift: i32, max: i32) {
    if shift != 6 || max != 255 {
        return (HevcDsp::<u8>::SCALAR.uni)(dst, stride, src, w, h, shift, max);
    }
    unsafe {
        for y in 0..h {
            let mut x = 0;
            while x < w {
                let n = (w - x).min(8);
                let s = w16::load_n(src.as_ptr().add(y * w + x), w - x);
                store_bytes(dst.as_mut_ptr().add(y * stride + x), vqrshrun_n_s16::<6>(s), n);
                x += 8;
            }
        }
    }
}

fn bi_neon(dst: &mut [u8], stride: usize, a: &[i16], b: &[i16], w: usize, h: usize, shift: i32, max: i32) {
    if shift != 7 || max != 255 {
        return (HevcDsp::<u8>::SCALAR.bi)(dst, stride, a, b, w, h, shift, max);
    }
    unsafe {
        for y in 0..h {
            let mut x = 0;
            while x < w {
                let n = (w - x).min(8);
                let va = w16::load_n(a.as_ptr().add(y * w + x), w - x);
                let vb = w16::load_n(b.as_ptr().add(y * w + x), w - x);
                // Saturating sum: a + b can exceed i16 only when both are
                // far above the 8-bit range, and then the clip to 255 gives
                // the same answer as the exact 32-bit sum would.
                store_bytes(dst.as_mut_ptr().add(y * stride + x), vqrshrun_n_s16::<7>(vqaddq_s16(va, vb)), n);
                x += 8;
            }
        }
    }
}

/// 8 x i32 (two vectors) to eight bytes, saturating.
#[inline(always)]
unsafe fn narrow_u8(lo: int32x4_t, hi: int32x4_t) -> uint8x8_t {
    unsafe { vqmovn_u16(vcombine_u16(vqmovun_s32(lo), vqmovun_s32(hi))) }
}

#[allow(clippy::too_many_arguments)]
fn weighted_uni_neon(dst: &mut [u8], stride: usize, src: &[i16], w: usize, h: usize, log2_wd: i32, wt: i32, o: i32, max: i32) {
    debug_assert_eq!(max, 255);
    unsafe {
        let round = vdupq_n_s32(if log2_wd >= 1 { 1 << (log2_wd - 1) } else { 0 });
        let ov = vdupq_n_s32(o);
        let wv = vdupq_n_s32(wt);
        let sh = vdupq_n_s32(-log2_wd.max(0));
        for y in 0..h {
            let mut x = 0;
            while x < w {
                let n = (w - x).min(8);
                let s = w16::load_n(src.as_ptr().add(y * w + x), w - x);
                let lo = vaddq_s32(vshlq_s32(vaddq_s32(vmulq_s32(vmovl_s16(vget_low_s16(s)), wv), round), sh), ov);
                let hi = vaddq_s32(vshlq_s32(vaddq_s32(vmulq_s32(vmovl_high_s16(s), wv), round), sh), ov);
                store_bytes(dst.as_mut_ptr().add(y * stride + x), narrow_u8(lo, hi), n);
                x += 8;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn weighted_bi_neon(dst: &mut [u8], stride: usize, a: &[i16], b: &[i16], w: usize, h: usize, log2_wd: i32, w0: i32, w1: i32, o0: i32, o1: i32, max: i32) {
    debug_assert_eq!(max, 255);
    unsafe {
        let round = vdupq_n_s32((o0 + o1 + 1) << log2_wd);
        let w0v = vdupq_n_s32(w0);
        let w1v = vdupq_n_s32(w1);
        let sh = vdupq_n_s32(-(log2_wd + 1));
        for y in 0..h {
            let mut x = 0;
            while x < w {
                let n = (w - x).min(8);
                let va = w16::load_n(a.as_ptr().add(y * w + x), w - x);
                let vb = w16::load_n(b.as_ptr().add(y * w + x), w - x);
                let lo = vshlq_s32(vaddq_s32(vaddq_s32(vmulq_s32(vmovl_s16(vget_low_s16(va)), w0v), vmulq_s32(vmovl_s16(vget_low_s16(vb)), w1v)), round), sh);
                let hi = vshlq_s32(vaddq_s32(vaddq_s32(vmulq_s32(vmovl_high_s16(va), w0v), vmulq_s32(vmovl_high_s16(vb), w1v)), round), sh);
                store_bytes(dst.as_mut_ptr().add(y * stride + x), narrow_u8(lo, hi), n);
                x += 8;
            }
        }
    }
}

// ----------------------------------------------------------------------
// Residual add
// ----------------------------------------------------------------------

fn add_residual_neon(dst: &mut [u8], stride: usize, res: &[i16], n: usize, max: i32) {
    debug_assert_eq!(max, 255);
    unsafe {
        if n == 4 {
            for y in 0..4 {
                let d = dst.as_mut_ptr().add(y * stride);
                let p = vreinterpretq_s16_u16(vmovl_u8(vreinterpret_u8_u32(vdup_n_u32(std::ptr::read_unaligned(d as *const u32)))));
                let r = vcombine_s16(vld1_s16(res.as_ptr().add(y * 4)), vdup_n_s16(0));
                std::ptr::write_unaligned(d as *mut u32, vget_lane_u32::<0>(vreinterpret_u32_u8(vqmovun_s16(vaddq_s16(p, r)))));
            }
            return;
        }
        for y in 0..n {
            let mut x = 0;
            while x < n {
                let d = dst.as_mut_ptr().add(y * stride + x);
                let p = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(d)));
                let r = vld1q_s16(res.as_ptr().add(y * n + x));
                vst1_u8(d, vqmovun_s16(vaddq_s16(p, r)));
                x += 8;
            }
        }
    }
}

// ----------------------------------------------------------------------
// SAO
// ----------------------------------------------------------------------

/// `v + off` on bytes, clipped to `0..=255`, with `off` in `-128..=127`.
#[inline(always)]
unsafe fn add_offset_u8(v: uint8x16_t, off: int8x16_t) -> uint8x16_t {
    unsafe {
        let zero = vdupq_n_s8(0);
        let pos = vreinterpretq_u8_s8(vmaxq_s8(off, zero));
        let neg = vreinterpretq_u8_s8(vmaxq_s8(vnegq_s8(off), zero));
        vqsubq_u8(vqaddq_u8(v, pos), neg)
    }
}

#[allow(clippy::too_many_arguments)]
fn sao_band_neon(dst: &mut [u8], dst_stride: usize, src: &[u8], src_stride: usize, w: usize, h: usize, table: &[i16; 32], shift: i32, max: i32) {
    if shift != 3 || table.iter().any(|&o| !(-128..=127).contains(&o)) {
        return (HevcDsp::<u8>::SCALAR.sao_band)(dst, dst_stride, src, src_stride, w, h, table, shift, max);
    }
    unsafe {
        // The four consecutive bands (mod 32) with nonzero offsets.
        let mut bands = [0u8; 4];
        let mut offs = [0i8; 4];
        let mut k = 0;
        for b in 0..32 {
            if table[b] != 0 && k < 4 {
                bands[k] = (b as u8) << 3;
                offs[k] = table[b] as i8;
                k += 1;
            }
        }
        let mask = vdupq_n_u8(0xF8);
        for y in 0..h {
            let mut x = 0;
            while x < w {
                let n = (w - x).min(16);
                let v = load_bytes16(src.as_ptr().add(y * src_stride + x), n);
                let band = vandq_u8(v, mask);
                let mut off = vdupq_n_s8(0);
                for i in 0..k {
                    off = vbslq_s8(vceqq_u8(band, vdupq_n_u8(bands[i])), vdupq_n_s8(offs[i]), off);
                }
                store_bytes16(dst.as_mut_ptr().add(y * dst_stride + x), add_offset_u8(v, off), n);
                x += 16;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn sao_edge_neon(dst: &mut [u8], src: &[u8], origin: usize, stride: usize, w: usize, h: usize, na: isize, nb: isize, off: &[i16; 5], max: i32) {
    if off.iter().any(|&o| !(-128..=127).contains(&o)) {
        return (HevcDsp::<u8>::SCALAR.sao_edge)(dst, src, origin, stride, w, h, na, nb, off, max);
    }
    unsafe {
        // edgeIdx = 2 + sign(v-a) + sign(v-b) in 0..=4 indexes the offsets
        // through a byte table lookup.
        let mut tab = [0i8; 16];
        for i in 0..5 {
            tab[i] = off[i] as i8;
        }
        let tab = vld1q_s8(tab.as_ptr());
        let two = vdupq_n_s8(2);
        let lo_reach = na.min(nb).min(0);
        let hi_reach = na.max(nb).max(0);
        for y in 0..h {
            let mut x = 0;
            while x < w {
                let n = (w - x).min(16);
                let i = origin + y * stride + x;
                if (i as isize + lo_reach) < 0 || (i as isize + hi_reach) as usize + 16 > src.len() || i + 16 > dst.len() {
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
                let v = vld1q_u8(src.as_ptr().add(i));
                let a = vld1q_u8(src.as_ptr().offset(i as isize + na));
                let b = vld1q_u8(src.as_ptr().offset(i as isize + nb));
                // sign(v - a) as (v < a ? -1 : 0) - (v > a ? -1 : 0).
                let sa = vsubq_s8(vreinterpretq_s8_u8(vcltq_u8(v, a)), vreinterpretq_s8_u8(vcgtq_u8(v, a)));
                let sb = vsubq_s8(vreinterpretq_s8_u8(vcltq_u8(v, b)), vreinterpretq_s8_u8(vcgtq_u8(v, b)));
                let e = vaddq_s8(vaddq_s8(sa, sb), two);
                let o = vqtbl1q_s8(tab, vreinterpretq_u8_s8(e));
                store_bytes16(dst.as_mut_ptr().add(i), add_offset_u8(v, o), n);
                x += 16;
            }
        }
    }
}

// ----------------------------------------------------------------------
// Deblocking — the shared i32-lane filters with byte loads and stores.
// ----------------------------------------------------------------------

/// Four consecutive bytes as 4 x i32.
#[inline(always)]
unsafe fn ld4_u8(p: *const u8) -> int32x4_t {
    unsafe {
        let b = vreinterpret_u8_u32(vdup_n_u32(std::ptr::read_unaligned(p as *const u32)));
        vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(vmovl_u8(b))))
    }
}

/// 4 x i32 (each within a byte) to four bytes, stored at `p`.
#[inline(always)]
unsafe fn st4_u8(p: *mut u8, v: int32x4_t) {
    unsafe {
        let h = vqmovun_s32(v);
        let b = vqmovn_u16(vcombine_u16(h, h));
        std::ptr::write_unaligned(p as *mut u32, vget_lane_u32::<0>(vreinterpret_u32_u8(b)));
    }
}

#[allow(clippy::too_many_arguments)]
fn deblock_luma_h_neon(data: &mut [u8], off: usize, stride: usize, beta: [i32; 2], tc: [i32; 2], no_p: [bool; 2], no_q: [bool; 2], max: i32) {
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
            let mut v: w16::Seg = [vdupq_n_s32(0); 8];
            for k in 0..8 {
                v[k] = ld4_u8(base.offset((k as isize - 4) * stride as isize));
            }
            w16::luma_segment(&mut v, beta[seg], tc[seg], no_p[seg], no_q[seg], max);
            for k in 1..7 {
                st4_u8(base.offset((k as isize - 4) * stride as isize), v[k]);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn deblock_luma_v_neon(data: &mut [u8], off: usize, stride: usize, beta: [i32; 2], tc: [i32; 2], no_p: [bool; 2], no_q: [bool; 2], max: i32) {
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
            let r0 = vmovl_u8(vld1_u8(base.sub(4)));
            let r1 = vmovl_u8(vld1_u8(base.add(stride).sub(4)));
            let r2 = vmovl_u8(vld1_u8(base.add(2 * stride).sub(4)));
            let r3 = vmovl_u8(vld1_u8(base.add(3 * stride).sub(4)));
            let a = vtrnq_u16(r0, r1);
            let b = vtrnq_u16(r2, r3);
            let c = vtrnq_u32(vreinterpretq_u32_u16(a.0), vreinterpretq_u32_u16(b.0));
            let d = vtrnq_u32(vreinterpretq_u32_u16(a.1), vreinterpretq_u32_u16(b.1));
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
            let mut v: w16::Seg = [vdupq_n_s32(0); 8];
            for k in 0..8 {
                v[k] = vreinterpretq_s32_u32(vmovl_u16(cols[k]));
            }
            w16::luma_segment(&mut v, beta[seg], tc[seg], no_p[seg], no_q[seg], max);
            // Back to rows: transpose the eight 4-lane columns.
            let mut cc = [vdup_n_u16(0); 8];
            for k in 0..8 {
                cc[k] = vqmovun_s32(v[k]);
            }
            let l0 = vcombine_u16(cc[0], cc[4]);
            let l1 = vcombine_u16(cc[1], cc[5]);
            let l2 = vcombine_u16(cc[2], cc[6]);
            let l3 = vcombine_u16(cc[3], cc[7]);
            let a = vtrnq_u16(l0, l1);
            let b = vtrnq_u16(l2, l3);
            let c = vtrnq_u32(vreinterpretq_u32_u16(a.0), vreinterpretq_u32_u16(b.0));
            let d = vtrnq_u32(vreinterpretq_u32_u16(a.1), vreinterpretq_u32_u16(b.1));
            let rows = [c.0, d.0, c.1, d.1];
            for i in 0..4 {
                vst1_u8(base.add(i * stride).sub(4), vqmovn_u16(vreinterpretq_u16_u32(rows[i])));
            }
        }
    }
}

fn deblock_chroma_h_neon(data: &mut [u8], off: usize, stride: usize, tc: [i32; 4], no_p: [bool; 4], no_q: [bool; 4], max: i32) {
    if tc.iter().all(|&t| t == 0) {
        return;
    }
    assert!(off >= 2 * stride && off + stride + 8 <= data.len());
    unsafe {
        let p = data.as_mut_ptr().add(off);
        for half in 0..2 {
            let base = p.add(4 * half);
            let mut v = [ld4_u8(base.sub(2 * stride)), ld4_u8(base.sub(stride)), ld4_u8(base), ld4_u8(base.add(stride))];
            w16::chroma_lines4(&mut v, [tc[2 * half], tc[2 * half + 1]], [no_p[2 * half], no_p[2 * half + 1]], [no_q[2 * half], no_q[2 * half + 1]], max);
            st4_u8(base.sub(stride), v[1]);
            st4_u8(base, v[2]);
        }
    }
}

fn deblock_chroma_v_neon(data: &mut [u8], off: usize, stride: usize, tc: [i32; 4], no_p: [bool; 4], no_q: [bool; 4], max: i32) {
    if tc.iter().all(|&t| t == 0) {
        return;
    }
    assert!(off >= 2 && off + 7 * stride + 2 <= data.len());
    unsafe {
        let p = data.as_mut_ptr().add(off);
        for half in 0..2 {
            let base = p.add(4 * half * stride);
            // Four rows x 4 samples (p1 p0 q0 q1) -> four columns of 4.
            let ld = |q: *const u8| vget_low_u16(vmovl_u8(vreinterpret_u8_u32(vdup_n_u32(std::ptr::read_unaligned(q as *const u32)))));
            let r0 = ld(base.sub(2));
            let r1 = ld(base.add(stride).sub(2));
            let r2 = ld(base.add(2 * stride).sub(2));
            let r3 = ld(base.add(3 * stride).sub(2));
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
            w16::chroma_lines4(&mut v, [tc[2 * half], tc[2 * half + 1]], [no_p[2 * half], no_p[2 * half + 1]], [no_q[2 * half], no_q[2 * half + 1]], max);
            // (p0, q0) per row.
            let p0 = vqmovun_s32(v[1]);
            let q0 = vqmovun_s32(v[2]);
            let pq = vzip_u16(p0, q0); // p0r0 q0r0 p0r1 q0r1 | p0r2 q0r2 p0r3 q0r3
            let bytes = vqmovn_u16(vcombine_u16(pq.0, pq.1));
            let mut t = [0u8; 8];
            vst1_u8(t.as_mut_ptr(), bytes);
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

    fn neon() -> HevcDsp<u8> {
        let mut d = HevcDsp::<u8>::SCALAR;
        install(&mut d);
        d
    }

    #[test]
    fn interp_matches_scalar_u8() {
        let d = neon();
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
                    }
                    (s.epel_h)(&mut a, &src, stride, w, h, frac, 0);
                    (d.epel_h)(&mut b, &src, stride, w, h, frac, 0);
                    assert_eq!(a, b, "epel_h {w}x{h} frac={frac} trial={trial}");
                    (s.epel_v)(&mut a, &src, stride, w, h, frac, 0);
                    (d.epel_v)(&mut b, &src, stride, w, h, frac, 0);
                    assert_eq!(a, b, "epel_v {w}x{h} frac={frac} trial={trial}");
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
    fn fused_matches_scalar_u8() {
        let d = neon();
        let s = HevcDsp::<u8>::SCALAR;
        let mut seed = 5u64;
        let stride = 96;
        let mut tmp1 = vec![0i16; crate::dsp::hevc::MC_TMP_LEN];
        let mut tmp2 = vec![0i16; crate::dsp::hevc::MC_TMP_LEN];
        let mut checked = 0;
        for trial in 0..2 {
            let src: Vec<u8> = (0..stride * 96).map(|_| if trial == 0 { lcg(&mut seed) as u8 } else { [0u8, 255][(lcg(&mut seed) % 2) as usize] }).collect();
            for &(w, h) in &[(2usize, 4usize), (2, 8), (4, 4), (4, 8), (4, 3), (6, 8), (8, 4), (8, 8), (8, 5), (12, 16), (16, 16), (24, 32), (32, 8), (48, 64), (64, 64)] {
                let other: Vec<i16> = (0..w * h).map(|_| (lcg(&mut seed) % 30000) as i16 - 6000).collect();
                for fx in 0..8 {
                    for fy in 0..8 {
                        for luma in [true, false] {
                            if luma && (fx >= 4 || fy >= 4) {
                                continue;
                            }
                            let dstride = w + 3;
                            let mut d1 = vec![7u8; dstride * h + 8];
                            let mut d2 = d1.clone();
                            let off = 8 * stride + 8;
                            if luma {
                                (s.qpel_uni)(&mut d1, dstride, &src[off..], stride, w, h, fx, fy, &mut tmp1, 8);
                                (d.qpel_uni)(&mut d2, dstride, &src[off..], stride, w, h, fx, fy, &mut tmp2, 8);
                            } else {
                                (s.epel_uni)(&mut d1, dstride, &src[off..], stride, w, h, fx, fy, &mut tmp1, 8);
                                (d.epel_uni)(&mut d2, dstride, &src[off..], stride, w, h, fx, fy, &mut tmp2, 8);
                            }
                            assert_eq!(d1, d2, "uni luma={luma} {w}x{h} fx={fx} fy={fy} trial={trial}");
                            if luma {
                                (s.qpel_bi)(&mut d1, dstride, &src[off..], stride, w, h, fx, fy, &mut tmp1, &other, 8);
                                (d.qpel_bi)(&mut d2, dstride, &src[off..], stride, w, h, fx, fy, &mut tmp2, &other, 8);
                            } else {
                                (s.epel_bi)(&mut d1, dstride, &src[off..], stride, w, h, fx, fy, &mut tmp1, &other, 8);
                                (d.epel_bi)(&mut d2, dstride, &src[off..], stride, w, h, fx, fy, &mut tmp2, &other, 8);
                            }
                            assert_eq!(d1, d2, "bi luma={luma} {w}x{h} fx={fx} fy={fy} trial={trial}");
                            checked += 1;
                        }
                    }
                }
            }
        }
        assert_eq!(checked, 2 * 15 * (16 + 64));
    }

    #[test]
    fn combine_matches_scalar_u8() {
        let d = neon();
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
            if w == h && w >= 4 && w.is_power_of_two() && w <= 32 {
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
        let d = neon();
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
        let d = neon();
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
