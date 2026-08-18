//! H.264 kernels: quarter-sample luma interpolation (8.4.2.2.1), chroma
//! bilinear interpolation (8.4.2.2.2), sample combination and weighting
//! (8.4.2.3), generic over the sample type (`u8` for 8-bit streams, `u16`
//! above). Scalar reference here; SIMD versions for 8-bit planes are
//! installed by [`super::h264_avx2`] / [`super::h264_neon`].
//!
//! Prediction blocks are samples at every stage in H.264 (each interpolation
//! position rounds and clips to a sample), so kernels produce sample blocks
//! and the combiners average / weight those. A prediction block lives in the
//! top-left of a 16x16 scratch buffer with row stride [`PRED_STRIDE`], whatever
//! its size: a kernel may read and write the whole 16-sample row, so the SIMD
//! versions need no tail handling for 4- and 8-wide blocks. Kernels that clip
//! take the sample maximum (`(1 << BitDepth) - 1`); the 8-bit SIMD kernels
//! ignore it.

use super::Cpu;
use crate::sample::Sample;

/// Row stride of a prediction scratch block (bytes). Every prediction buffer
/// is at least `16 * PRED_STRIDE` bytes.
pub const PRED_STRIDE: usize = 16;

/// Interpolate a `w x h` luma block into a [`PRED_STRIDE`]-strided scratch
/// block. `src` points at the top-left of the six-tap window: two samples left
/// of and two rows above the block's full-sample position; `src_stride` is the
/// plane stride.
pub type QpelFn<S> = fn(dst: &mut [S], src: &[S], src_stride: usize, w: usize, h: usize, max: i32);
/// Chroma bilinear into a [`PRED_STRIDE`]-strided scratch block: `src` at the
/// block's integer chroma position, fractions `xf`/`yf` in eighths.
pub type ChromaFn<S> = fn(dst: &mut [S], src: &[S], src_stride: usize, w: usize, h: usize, xf: i32, yf: i32);
/// `dst = src` (a [`PRED_STRIDE`]-strided scratch block into a strided plane).
pub type CopyFn<S> = fn(dst: &mut [S], stride: usize, src: &[S], w: usize, h: usize);
/// `dst = (a + b + 1) >> 1` (both [`PRED_STRIDE`]-strided scratch blocks).
pub type AvgFn<S> = fn(dst: &mut [S], stride: usize, a: &[S], b: &[S], w: usize, h: usize);
/// `dst = clip(((src * w + round) >> log_wd) + o)` (8-278 / 8-279).
pub type WeightedUniFn<S> = fn(dst: &mut [S], stride: usize, src: &[S], w: usize, h: usize, log_wd: i32, wt: i32, o: i32, max: i32);
/// `dst = clip(((a * w0 + b * w1 + 2^log_wd) >> (log_wd + 1)) + ((o0 + o1 + 1) >> 1))` (8-280).
pub type WeightedBiFn<S> = fn(dst: &mut [S], stride: usize, a: &[S], b: &[S], w: usize, h: usize, log_wd: i32, w0: i32, w1: i32, o0: i32, o1: i32, max: i32);

/// Deblock sixteen lines of a luma edge with bS < 4 (8.7.2.3). `off` is the
/// offset of q0 on the first line; a *vertical* edge (`_v`) has its lines
/// `stride` apart and p/q samples 1 apart, a *horizontal* edge (`_h`) the
/// other way round. `tc0[i / 4]` is the segment's tC0 (already scaled to
/// the bit depth, as are alpha and beta), or −1 for bS 0 (leave the line
/// alone).
pub type LumaDeblockFn<S> = fn(data: &mut [S], off: usize, stride: usize, alpha: i32, beta: i32, tc0: &[i16; 4], max: i32);
/// Deblock sixteen lines of a luma edge with bS 4 (8.7.2.4).
pub type LumaDeblockIntraFn<S> = fn(data: &mut [S], off: usize, stride: usize, alpha: i32, beta: i32, max: i32);
/// Deblock eight lines of a 4:2:0 chroma edge with bS < 4; `tc0[i / 2]`.
pub type ChromaDeblockFn<S> = fn(data: &mut [S], off: usize, stride: usize, alpha: i32, beta: i32, tc0: &[i16; 4], max: i32);
/// Deblock eight lines of a 4:2:0 chroma edge with bS 4.
pub type ChromaDeblockIntraFn<S> = fn(data: &mut [S], off: usize, stride: usize, alpha: i32, beta: i32, max: i32);

/// Inverse 4x4 transform (8.5.12.2) of dequantised coefficients in raster
/// order, added to the prediction in `dst` with clipping.
pub type Idct4AddFn<S> = fn(dst: &mut [S], stride: usize, coeffs: &[i16; 16], max: i32);
/// Inverse 8x8 transform (8.5.13.2), added with clipping.
pub type Idct8AddFn<S> = fn(dst: &mut [S], stride: usize, coeffs: &[i16; 64], max: i32);
/// A DC-only block: `(dc + 32) >> 6` added to every sample of the 4x4 / 8x8.
pub type DcAddFn<S> = fn(dst: &mut [S], stride: usize, dc: i32, max: i32);
/// The whole residual path of one 4x4 block: dequantise `levels` (raster)
/// with `scale` at `qp` (8.5.12.1), replace position 0 by `dc` when
/// `dc != NO_DC` (an Intra_16x16 / chroma DC already scaled), inverse
/// transform and add to `dst`. Blocks with no AC take the DC-only path.
pub type Residual4Fn<S> = fn(dst: &mut [S], stride: usize, levels: &[i32; 16], scale: &[i32; 16], qp: i32, dc: i32, max: i32);
/// The same for an 8x8 block (8.5.13.1); no separate DC.
pub type Residual8Fn<S> = fn(dst: &mut [S], stride: usize, levels: &[i32; 64], scale: &[i32; 64], qp: i32, max: i32);
/// `dc` value of [`Residual4Fn`] meaning "position 0 is a level like the rest".
pub const NO_DC: i32 = i32::MIN;

/// The kernel table.
#[derive(Clone, Copy)]
pub struct H264Dsp<S: Sample = u8> {
    /// Which CPU features the table was built for.
    pub cpu: Cpu,
    /// Luma interpolation by position `yf * 4 + xf`.
    pub qpel: [QpelFn<S>; 16],
    /// Chroma bilinear.
    pub chroma: ChromaFn<S>,
    /// Copy a packed block into the plane.
    pub copy: CopyFn<S>,
    /// Rounded average of two packed blocks into the plane.
    pub avg: AvgFn<S>,
    /// Explicit/implicit weighting, one list.
    pub weighted_uni: WeightedUniFn<S>,
    /// Explicit/implicit weighting, both lists.
    pub weighted_bi: WeightedBiFn<S>,
    /// Deblocking: luma vertical edge (bS < 4).
    pub deblock_luma_v: LumaDeblockFn<S>,
    /// Deblocking: luma horizontal edge (bS < 4).
    pub deblock_luma_h: LumaDeblockFn<S>,
    /// Deblocking: luma vertical edge (bS 4).
    pub deblock_luma_v_intra: LumaDeblockIntraFn<S>,
    /// Deblocking: luma horizontal edge (bS 4).
    pub deblock_luma_h_intra: LumaDeblockIntraFn<S>,
    /// Deblocking: chroma vertical edge (bS < 4).
    pub deblock_chroma_v: ChromaDeblockFn<S>,
    /// Deblocking: chroma horizontal edge (bS < 4).
    pub deblock_chroma_h: ChromaDeblockFn<S>,
    /// Deblocking: chroma vertical edge (bS 4).
    pub deblock_chroma_v_intra: ChromaDeblockIntraFn<S>,
    /// Deblocking: chroma horizontal edge (bS 4).
    pub deblock_chroma_h_intra: ChromaDeblockIntraFn<S>,
    /// Inverse 4x4 transform + add.
    pub idct4_add: Idct4AddFn<S>,
    /// Inverse 8x8 transform + add.
    pub idct8_add: Idct8AddFn<S>,
    /// DC-only 4x4 add.
    pub idct4_dc_add: DcAddFn<S>,
    /// DC-only 8x8 add.
    pub idct8_dc_add: DcAddFn<S>,
    /// Dequantise + inverse 4x4 + add.
    pub residual4: Residual4Fn<S>,
    /// Dequantise + inverse 8x8 + add.
    pub residual8: Residual8Fn<S>,
}

macro_rules! scalar_table {
    ($S:ty) => {
        H264Dsp {
            cpu: Cpu::SCALAR,
            qpel: [
                qpel_scalar::<$S, 0, 0>,
                qpel_scalar::<$S, 1, 0>,
                qpel_scalar::<$S, 2, 0>,
                qpel_scalar::<$S, 3, 0>,
                qpel_scalar::<$S, 0, 1>,
                qpel_scalar::<$S, 1, 1>,
                qpel_scalar::<$S, 2, 1>,
                qpel_scalar::<$S, 3, 1>,
                qpel_scalar::<$S, 0, 2>,
                qpel_scalar::<$S, 1, 2>,
                qpel_scalar::<$S, 2, 2>,
                qpel_scalar::<$S, 3, 2>,
                qpel_scalar::<$S, 0, 3>,
                qpel_scalar::<$S, 1, 3>,
                qpel_scalar::<$S, 2, 3>,
                qpel_scalar::<$S, 3, 3>,
            ],
            chroma: chroma_scalar::<$S>,
            copy: copy_scalar::<$S>,
            avg: avg_scalar::<$S>,
            weighted_uni: weighted_uni_scalar::<$S>,
            weighted_bi: weighted_bi_scalar::<$S>,
            deblock_luma_v: |d, off, stride, a, b, tc0, max| deblock_luma_scalar(d, off, stride, 1, a, b, Some(tc0), max),
            deblock_luma_h: |d, off, stride, a, b, tc0, max| deblock_luma_scalar(d, off, 1, stride, a, b, Some(tc0), max),
            deblock_luma_v_intra: |d, off, stride, a, b, max| deblock_luma_scalar(d, off, stride, 1, a, b, None, max),
            deblock_luma_h_intra: |d, off, stride, a, b, max| deblock_luma_scalar(d, off, 1, stride, a, b, None, max),
            deblock_chroma_v: |d, off, stride, a, b, tc0, max| deblock_chroma_scalar(d, off, stride, 1, a, b, Some(tc0), max),
            deblock_chroma_h: |d, off, stride, a, b, tc0, max| deblock_chroma_scalar(d, off, 1, stride, a, b, Some(tc0), max),
            deblock_chroma_v_intra: |d, off, stride, a, b, max| deblock_chroma_scalar(d, off, stride, 1, a, b, None, max),
            deblock_chroma_h_intra: |d, off, stride, a, b, max| deblock_chroma_scalar(d, off, 1, stride, a, b, None, max),
            idct4_add: idct4_add_scalar::<$S>,
            idct8_add: idct8_add_scalar::<$S>,
            idct4_dc_add: |d, s, dc, max| dc_add_scalar(d, s, dc, 4, max),
            idct8_dc_add: |d, s, dc, max| dc_add_scalar(d, s, dc, 8, max),
            residual4: residual4_scalar::<$S>,
            residual8: residual8_scalar::<$S>,
        }
    };
}

impl H264Dsp<u8> {
    /// The scalar reference table (8-bit samples).
    pub const SCALAR: H264Dsp<u8> = scalar_table!(u8);
}

impl H264Dsp<u16> {
    /// The scalar reference table (16-bit samples).
    pub const SCALAR: H264Dsp<u16> = scalar_table!(u16);
}

impl<S: Sample> H264Dsp<S> {
    /// The scalar reference table.
    pub fn scalar() -> Self {
        scalar_table!(S)
    }

    /// The best table for `cpu`.
    pub fn new(cpu: Cpu) -> Self {
        let mut d = Self::scalar();
        d.cpu = cpu;
        S::install_h264_simd(&mut d, cpu);
        d
    }
}

/// SIMD kernels for 8-bit sample planes.
#[allow(unused_variables)]
pub fn install_simd_u8(d: &mut H264Dsp<u8>, cpu: Cpu) {
    #[cfg(target_arch = "x86_64")]
    if cpu.avx2 {
        super::h264_avx2::install(d);
    }
    #[cfg(target_arch = "aarch64")]
    if cpu.neon {
        super::h264_neon::install(d);
    }
}

// ----------------------------------------------------------------------
// Luma interpolation (scalar)
// ----------------------------------------------------------------------

#[inline(always)]
fn tap6(a: i32, b: i32, c: i32, d: i32, e: i32, f: i32) -> i32 {
    a - 5 * b + 20 * c + 20 * d - 5 * e + f
}

/// The block's full samples `G` at `(x, y)` in window coordinates.
#[inline(always)]
fn g<S: Sample>(src: &[S], stride: usize, x: usize, y: usize) -> i32 {
    src[(y + 2) * stride + x + 2].to_i32()
}

/// Horizontal six-tap intermediate `b1` at window row `yy` (window
/// coordinates: row 0 = two above the block), block column `x`.
#[inline(always)]
fn b1<S: Sample>(src: &[S], stride: usize, x: usize, yy: usize) -> i32 {
    let r = &src[yy * stride + x..];
    tap6(r[0].to_i32(), r[1].to_i32(), r[2].to_i32(), r[3].to_i32(), r[4].to_i32(), r[5].to_i32())
}

/// Vertical six-tap intermediate `h1` at window column `xx`, block row `y`.
#[inline(always)]
fn h1<S: Sample>(src: &[S], stride: usize, xx: usize, y: usize) -> i32 {
    let c = &src[y * stride + xx..];
    tap6(c[0].to_i32(), c[stride].to_i32(), c[2 * stride].to_i32(), c[3 * stride].to_i32(), c[4 * stride].to_i32(), c[5 * stride].to_i32())
}

/// Scalar interpolation for position `(XF, YF)`.
fn qpel_scalar<S: Sample, const XF: usize, const YF: usize>(dst: &mut [S], src: &[S], stride: usize, w: usize, h: usize, max: i32) {
    // b: half-sample horizontally at block (x, y): b1 at window row y + 2.
    let b = |x: usize, y: usize| ((b1(src, stride, x, y + 2) + 16) >> 5).clamp(0, max);
    // hh: half-sample vertically at block (x, y): h1 at window column x + 2.
    let hh = |x: usize, y: usize| ((h1(src, stride, x + 2, y) + 16) >> 5).clamp(0, max);
    // j: centre, vertical six-tap over b1 rows y..y+5 (window rows).
    let j = |x: usize, y: usize| {
        let j1 = tap6(b1(src, stride, x, y), b1(src, stride, x, y + 1), b1(src, stride, x, y + 2), b1(src, stride, x, y + 3), b1(src, stride, x, y + 4), b1(src, stride, x, y + 5));
        ((j1 + 512) >> 10).clamp(0, max)
    };
    for y in 0..h {
        for x in 0..w {
            let v = match (XF, YF) {
                (0, 0) => g(src, stride, x, y),
                (1, 0) => (g(src, stride, x, y) + b(x, y) + 1) >> 1,
                (2, 0) => b(x, y),
                (3, 0) => (g(src, stride, x + 1, y) + b(x, y) + 1) >> 1,
                (0, 1) => (g(src, stride, x, y) + hh(x, y) + 1) >> 1,
                (0, 2) => hh(x, y),
                (0, 3) => (g(src, stride, x, y + 1) + hh(x, y) + 1) >> 1,
                (2, 2) => j(x, y),
                (1, 1) => (b(x, y) + hh(x, y) + 1) >> 1,
                (3, 1) => (b(x, y) + hh(x + 1, y) + 1) >> 1,
                (1, 3) => (hh(x, y) + b(x, y + 1) + 1) >> 1,
                (3, 3) => (hh(x + 1, y) + b(x, y + 1) + 1) >> 1,
                (2, 1) => (b(x, y) + j(x, y) + 1) >> 1,
                (2, 3) => (j(x, y) + b(x, y + 1) + 1) >> 1,
                (1, 2) => (hh(x, y) + j(x, y) + 1) >> 1,
                (3, 2) => (j(x, y) + hh(x + 1, y) + 1) >> 1,
                _ => unreachable!(),
            };
            dst[y * PRED_STRIDE + x] = S::from_i32(v);
        }
    }
}

// ----------------------------------------------------------------------
// Chroma / combination (scalar)
// ----------------------------------------------------------------------

fn chroma_scalar<S: Sample>(dst: &mut [S], src: &[S], stride: usize, w: usize, h: usize, xf: i32, yf: i32) {
    let (a, b, c, d) = ((8 - xf) * (8 - yf), xf * (8 - yf), (8 - xf) * yf, xf * yf);
    for y in 0..h {
        let r0 = &src[y * stride..];
        let r1 = &src[(y + 1) * stride..];
        for x in 0..w {
            let v = a * r0[x].to_i32() + b * r0[x + 1].to_i32() + c * r1[x].to_i32() + d * r1[x + 1].to_i32();
            dst[y * PRED_STRIDE + x] = S::from_i32((v + 32) >> 6);
        }
    }
}

fn copy_scalar<S: Sample>(dst: &mut [S], stride: usize, src: &[S], w: usize, h: usize) {
    for y in 0..h {
        dst[y * stride..y * stride + w].copy_from_slice(&src[y * PRED_STRIDE..y * PRED_STRIDE + w]);
    }
}

fn avg_scalar<S: Sample>(dst: &mut [S], stride: usize, a: &[S], b: &[S], w: usize, h: usize) {
    for y in 0..h {
        for x in 0..w {
            dst[y * stride + x] = S::from_i32((a[y * PRED_STRIDE + x].to_i32() + b[y * PRED_STRIDE + x].to_i32() + 1) >> 1);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn weighted_uni_scalar<S: Sample>(dst: &mut [S], stride: usize, src: &[S], w: usize, h: usize, log_wd: i32, wt: i32, o: i32, max: i32) {
    if log_wd >= 1 {
        let round = 1 << (log_wd - 1);
        for y in 0..h {
            for x in 0..w {
                dst[y * stride + x] = S::from_i32((((src[y * PRED_STRIDE + x].to_i32() * wt + round) >> log_wd) + o).clamp(0, max));
            }
        }
    } else {
        for y in 0..h {
            for x in 0..w {
                dst[y * stride + x] = S::from_i32((src[y * PRED_STRIDE + x].to_i32() * wt + o).clamp(0, max));
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn weighted_bi_scalar<S: Sample>(dst: &mut [S], stride: usize, a: &[S], b: &[S], w: usize, h: usize, log_wd: i32, w0: i32, w1: i32, o0: i32, o1: i32, max: i32) {
    let off = (o0 + o1 + 1) >> 1;
    let round = 1 << log_wd;
    for y in 0..h {
        for x in 0..w {
            let v = ((a[y * PRED_STRIDE + x].to_i32() * w0 + b[y * PRED_STRIDE + x].to_i32() * w1 + round) >> (log_wd + 1)) + off;
            dst[y * stride + x] = S::from_i32(v.clamp(0, max));
        }
    }
}

// ----------------------------------------------------------------------
// Deblocking (scalar)
// ----------------------------------------------------------------------

/// Filter one line of samples across an edge (8.7.2.3 / 8.7.2.4).
/// `p[0..4]` are p0..p3 (p0 nearest the edge), `q[0..4]` are q0..q3.
/// `tc0 == None` is bS 4. `max` is the sample maximum.
#[inline]
pub(crate) fn deblock_line(p: &mut [i32; 4], q: &mut [i32; 4], tc0: Option<i32>, alpha: i32, beta: i32, chroma: bool, max: i32) {
    let (p0, p1, p2, p3) = (p[0], p[1], p[2], p[3]);
    let (q0, q1, q2, q3) = (q[0], q[1], q[2], q[3]);
    if !((p0 - q0).abs() < alpha && (p1 - p0).abs() < beta && (q1 - q0).abs() < beta) {
        return;
    }
    let ap = (p2 - p0).abs();
    let aq = (q2 - q0).abs();
    if let Some(tc0) = tc0 {
        let tc = if chroma { tc0 + 1 } else { tc0 + (ap < beta) as i32 + (aq < beta) as i32 };
        let delta = ((((q0 - p0) << 2) + (p1 - q1) + 4) >> 3).clamp(-tc, tc);
        p[0] = (p0 + delta).clamp(0, max);
        q[0] = (q0 - delta).clamp(0, max);
        if !chroma {
            if ap < beta {
                p[1] = p1 + ((p2 + ((p0 + q0 + 1) >> 1) - (p1 << 1)) >> 1).clamp(-tc0, tc0);
            }
            if aq < beta {
                q[1] = q1 + ((q2 + ((p0 + q0 + 1) >> 1) - (q1 << 1)) >> 1).clamp(-tc0, tc0);
            }
        }
    } else {
        let strong = (p0 - q0).abs() < ((alpha >> 2) + 2);
        if !chroma && ap < beta && strong {
            p[0] = (p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) >> 3;
            p[1] = (p2 + p1 + p0 + q0 + 2) >> 2;
            p[2] = (2 * p3 + 3 * p2 + p1 + p0 + q0 + 4) >> 3;
        } else {
            p[0] = (2 * p1 + p0 + q1 + 2) >> 2;
        }
        if !chroma && aq < beta && strong {
            q[0] = (p1 + 2 * p0 + 2 * q0 + 2 * q1 + q2 + 4) >> 3;
            q[1] = (p0 + q0 + q1 + q2 + 2) >> 2;
            q[2] = (2 * q3 + 3 * q2 + q1 + q0 + p0 + 4) >> 3;
        } else {
            q[0] = (2 * q1 + q0 + p1 + 2) >> 2;
        }
    }
}

/// Sixteen luma lines: `step` moves along the edge, `across` across it.
#[allow(clippy::too_many_arguments)]
fn deblock_luma_scalar<S: Sample>(data: &mut [S], off: usize, step: usize, across: usize, alpha: i32, beta: i32, tc0: Option<&[i16; 4]>, max: i32) {
    for i in 0..16 {
        let t = match tc0 {
            Some(t) => {
                if t[i / 4] < 0 {
                    continue;
                }
                Some(t[i / 4] as i32)
            }
            None => None,
        };
        let base = off + i * step;
        let mut p = [0i32; 4];
        let mut q = [0i32; 4];
        for k in 0..4 {
            p[k] = data[base - (k + 1) * across].to_i32();
            q[k] = data[base + k * across].to_i32();
        }
        deblock_line(&mut p, &mut q, t, alpha, beta, false, max);
        for k in 0..3 {
            data[base - (k + 1) * across] = S::from_i32(p[k]);
            data[base + k * across] = S::from_i32(q[k]);
        }
    }
}

/// Eight chroma lines (4:2:0): line `i` uses `tc0[i / 2]`.
#[allow(clippy::too_many_arguments)]
fn deblock_chroma_scalar<S: Sample>(data: &mut [S], off: usize, step: usize, across: usize, alpha: i32, beta: i32, tc0: Option<&[i16; 4]>, max: i32) {
    for i in 0..8 {
        let t = match tc0 {
            Some(t) => {
                if t[i / 2] < 0 {
                    continue;
                }
                Some(t[i / 2] as i32)
            }
            None => None,
        };
        let base = off + i * step;
        let mut p = [0i32; 4];
        let mut q = [0i32; 4];
        for k in 0..2 {
            p[k] = data[base - (k + 1) * across].to_i32();
            q[k] = data[base + k * across].to_i32();
        }
        deblock_line(&mut p, &mut q, t, alpha, beta, true, max);
        data[base - across] = S::from_i32(p[0]);
        data[base] = S::from_i32(q[0]);
    }
}

// ----------------------------------------------------------------------
// Inverse transforms (scalar)
// ----------------------------------------------------------------------

/// The 4x4 inverse transform. For 8-bit samples in wrapping i16 arithmetic
/// — the SIMD kernels work in i16 lanes, and a conforming 8-bit stream never
/// leaves 16 bits, so the scalar reference matches them bit for bit on every
/// input; deeper samples take the i32 path (their transform values may need
/// up to `7 + BitDepth` bits).
fn idct4_add_scalar<S: Sample>(dst: &mut [S], stride: usize, coeffs: &[i16; 16], max: i32) {
    if S::BYTES != 1 {
        return idct4_add_wide(dst, stride, coeffs, max);
    }
    let mut d = *coeffs;
    let mut tmp = [0i16; 16];
    for i in 0..4 {
        let (d0, d1, d2, d3) = (d[i * 4], d[i * 4 + 1], d[i * 4 + 2], d[i * 4 + 3]);
        let e0 = d0.wrapping_add(d2);
        let e1 = d0.wrapping_sub(d2);
        let e2 = (d1 >> 1).wrapping_sub(d3);
        let e3 = d1.wrapping_add(d3 >> 1);
        tmp[i * 4] = e0.wrapping_add(e3);
        tmp[i * 4 + 1] = e1.wrapping_add(e2);
        tmp[i * 4 + 2] = e1.wrapping_sub(e2);
        tmp[i * 4 + 3] = e0.wrapping_sub(e3);
    }
    for j in 0..4 {
        let (f0, f1, f2, f3) = (tmp[j], tmp[4 + j], tmp[8 + j], tmp[12 + j]);
        let g0 = f0.wrapping_add(f2);
        let g1 = f0.wrapping_sub(f2);
        let g2 = (f1 >> 1).wrapping_sub(f3);
        let g3 = f1.wrapping_add(f3 >> 1);
        d[j] = g0.wrapping_add(g3);
        d[4 + j] = g1.wrapping_add(g2);
        d[8 + j] = g1.wrapping_sub(g2);
        d[12 + j] = g0.wrapping_sub(g3);
    }
    add_rows_scalar(dst, stride, &d, 4, max);
}

/// `(v + 32) >> 6` added to the samples, in the same wrapping i16 steps as
/// the SIMD kernels.
#[inline]
fn add_rows_scalar<S: Sample>(dst: &mut [S], stride: usize, v: &[i16], n: usize, max: i32) {
    for y in 0..n {
        for x in 0..n {
            let r = v[y * n + x].wrapping_add(32) >> 6;
            let p = &mut dst[y * stride + x];
            *p = S::from_i32(((p.to_i32() as i16).wrapping_add(r) as i32).clamp(0, max));
        }
    }
}

/// The 4x4 inverse transform in i32 (samples deeper than 8 bits).
fn idct4_add_wide<S: Sample>(dst: &mut [S], stride: usize, coeffs: &[i16; 16], max: i32) {
    let c: [i32; 16] = std::array::from_fn(|i| coeffs[i] as i32);
    idct4_add_i32(dst, stride, &c, max);
}

/// The 4x4 inverse transform of i32 coefficients (deeper samples: their
/// scaled coefficients reach `7 + BitDepth` bits).
fn idct4_add_i32<S: Sample>(dst: &mut [S], stride: usize, coeffs: &[i32; 16], max: i32) {
    let mut tmp = [0i32; 16];
    for i in 0..4 {
        let (d0, d1, d2, d3) = (coeffs[i * 4], coeffs[i * 4 + 1], coeffs[i * 4 + 2], coeffs[i * 4 + 3]);
        let e0 = d0 + d2;
        let e1 = d0 - d2;
        let e2 = (d1 >> 1) - d3;
        let e3 = d1 + (d3 >> 1);
        tmp[i * 4] = e0 + e3;
        tmp[i * 4 + 1] = e1 + e2;
        tmp[i * 4 + 2] = e1 - e2;
        tmp[i * 4 + 3] = e0 - e3;
    }
    for j in 0..4 {
        let (f0, f1, f2, f3) = (tmp[j], tmp[4 + j], tmp[8 + j], tmp[12 + j]);
        let g0 = f0 + f2;
        let g1 = f0 - f2;
        let g2 = (f1 >> 1) - f3;
        let g3 = f1 + (f3 >> 1);
        let out = [g0 + g3, g1 + g2, g1 - g2, g0 - g3];
        for (i, v) in out.into_iter().enumerate() {
            let p = &mut dst[i * stride + j];
            *p = S::from_i32((p.to_i32() + ((v + 32) >> 6)).clamp(0, max));
        }
    }
}

/// One dimension of the 8x8 inverse transform in i32.
#[inline(always)]
fn idct8_1d_i32(d: &[i32; 8]) -> [i32; 8] {
    let a0 = d[0] + d[4];
    let a4 = d[0] - d[4];
    let a2 = (d[2] >> 1) - d[6];
    let a6 = d[2] + (d[6] >> 1);
    let b0 = a0 + a6;
    let b2 = a4 + a2;
    let b4 = a4 - a2;
    let b6 = a0 - a6;
    let a1 = d[5] - d[3] - d[7] - (d[7] >> 1);
    let a3 = d[1] + d[7] - d[3] - (d[3] >> 1);
    let a5 = d[7] - d[1] + d[5] + (d[5] >> 1);
    let a7 = d[3] + d[5] + d[1] + (d[1] >> 1);
    let b1 = a1 + (a7 >> 2);
    let b7 = a7 - (a1 >> 2);
    let b3 = a3 + (a5 >> 2);
    let b5 = (a3 >> 2) - a5;
    [b0 + b7, b2 + b5, b4 + b3, b6 + b1, b6 - b1, b4 - b3, b2 - b5, b0 - b7]
}

/// The 8x8 inverse transform in i32 (samples deeper than 8 bits).
fn idct8_add_wide<S: Sample>(dst: &mut [S], stride: usize, coeffs: &[i16; 64], max: i32) {
    let c: [i32; 64] = std::array::from_fn(|i| coeffs[i] as i32);
    idct8_add_i32(dst, stride, &c, max);
}

/// The 8x8 inverse transform of i32 coefficients.
fn idct8_add_i32<S: Sample>(dst: &mut [S], stride: usize, coeffs: &[i32; 64], max: i32) {
    let mut tmp = [0i32; 64];
    for i in 0..8 {
        let row: [i32; 8] = std::array::from_fn(|k| coeffs[i * 8 + k]);
        tmp[i * 8..i * 8 + 8].copy_from_slice(&idct8_1d_i32(&row));
    }
    for j in 0..8 {
        let col: [i32; 8] = std::array::from_fn(|k| tmp[k * 8 + j]);
        let o = idct8_1d_i32(&col);
        for i in 0..8 {
            let p = &mut dst[i * stride + j];
            *p = S::from_i32((p.to_i32() + ((o[i] + 32) >> 6)).clamp(0, max));
        }
    }
}

#[inline(always)]
fn idct8_1d_i16(d: &[i16; 8]) -> [i16; 8] {
    let a0 = d[0].wrapping_add(d[4]);
    let a4 = d[0].wrapping_sub(d[4]);
    let a2 = (d[2] >> 1).wrapping_sub(d[6]);
    let a6 = d[2].wrapping_add(d[6] >> 1);
    let b0 = a0.wrapping_add(a6);
    let b2 = a4.wrapping_add(a2);
    let b4 = a4.wrapping_sub(a2);
    let b6 = a0.wrapping_sub(a6);
    let a1 = d[5].wrapping_sub(d[3]).wrapping_sub(d[7]).wrapping_sub(d[7] >> 1);
    let a3 = d[1].wrapping_add(d[7]).wrapping_sub(d[3]).wrapping_sub(d[3] >> 1);
    let a5 = d[7].wrapping_sub(d[1]).wrapping_add(d[5]).wrapping_add(d[5] >> 1);
    let a7 = d[3].wrapping_add(d[5]).wrapping_add(d[1]).wrapping_add(d[1] >> 1);
    let b1 = a1.wrapping_add(a7 >> 2);
    let b7 = a7.wrapping_sub(a1 >> 2);
    let b3 = a3.wrapping_add(a5 >> 2);
    let b5 = (a3 >> 2).wrapping_sub(a5);
    [
        b0.wrapping_add(b7),
        b2.wrapping_add(b5),
        b4.wrapping_add(b3),
        b6.wrapping_add(b1),
        b6.wrapping_sub(b1),
        b4.wrapping_sub(b3),
        b2.wrapping_sub(b5),
        b0.wrapping_sub(b7),
    ]
}

/// The 8x8 inverse transform in wrapping i16 arithmetic (see `idct4_add_scalar`).
fn idct8_add_scalar<S: Sample>(dst: &mut [S], stride: usize, coeffs: &[i16; 64], max: i32) {
    if S::BYTES != 1 {
        return idct8_add_wide(dst, stride, coeffs, max);
    }
    let mut tmp = [0i16; 64];
    for i in 0..8 {
        let row: [i16; 8] = coeffs[i * 8..i * 8 + 8].try_into().unwrap();
        tmp[i * 8..i * 8 + 8].copy_from_slice(&idct8_1d_i16(&row));
    }
    let mut out = [0i16; 64];
    for j in 0..8 {
        let col = [tmp[j], tmp[8 + j], tmp[16 + j], tmp[24 + j], tmp[32 + j], tmp[40 + j], tmp[48 + j], tmp[56 + j]];
        let o = idct8_1d_i16(&col);
        for i in 0..8 {
            out[i * 8 + j] = o[i];
        }
    }
    add_rows_scalar(dst, stride, &out, 8, max);
}

fn dc_add_scalar<S: Sample>(dst: &mut [S], stride: usize, dc: i32, n: usize, max: i32) {
    let v = if S::BYTES == 1 { ((dc as i16).wrapping_add(32) >> 6) as i32 } else { (dc + 32) >> 6 };
    for y in 0..n {
        for x in 0..n {
            let p = &mut dst[y * stride + x];
            *p = S::from_i32((p.to_i32() + v).clamp(0, max));
        }
    }
}

/// Scalar dequantisation of a 4x4 block into i16 (saturating: a conforming
/// stream stays within 16 bits, and the SIMD versions saturate too); returns
/// whether any AC coefficient is nonzero.
#[inline]
pub(crate) fn dequant4_scalar(levels: &[i32; 16], scale: &[i32; 16], qp: i32, dc: i32, out: &mut [i16; 16]) -> bool {
    let q6 = qp / 6;
    let start = if dc != NO_DC { 1 } else { 0 };
    let mut ac = 0i32;
    if qp >= 24 {
        let sh = q6 - 4;
        for i in start..16 {
            let v = (levels[i] * scale[i]) << sh;
            out[i] = v.clamp(-32768, 32767) as i16;
            ac |= v & (-((i != 0) as i32));
        }
    } else {
        let sh = 4 - q6;
        let round = 1 << (3 - q6);
        for i in start..16 {
            let v = (levels[i] * scale[i] + round) >> sh;
            out[i] = v.clamp(-32768, 32767) as i16;
            ac |= v & (-((i != 0) as i32));
        }
    }
    if dc != NO_DC {
        out[0] = dc as i16;
    }
    ac != 0
}

/// Scalar dequantisation of an 8x8 block into i16; returns whether any AC
/// coefficient is nonzero.
#[inline]
pub(crate) fn dequant8_scalar(levels: &[i32; 64], scale: &[i32; 64], qp: i32, out: &mut [i16; 64]) -> bool {
    let q6 = qp / 6;
    let mut ac = 0i32;
    if qp >= 36 {
        let sh = q6 - 6;
        for i in 0..64 {
            let v = (levels[i] * scale[i]) << sh;
            out[i] = v.clamp(-32768, 32767) as i16;
            ac |= v & (-((i != 0) as i32));
        }
    } else {
        let sh = 6 - q6;
        let round = 1 << (5 - q6);
        for i in 0..64 {
            let v = (levels[i] * scale[i] + round) >> sh;
            out[i] = v.clamp(-32768, 32767) as i16;
            ac |= v & (-((i != 0) as i32));
        }
    }
    ac != 0
}

fn residual4_scalar<S: Sample>(dst: &mut [S], stride: usize, levels: &[i32; 16], scale: &[i32; 16], qp: i32, dc: i32, max: i32) {
    if S::BYTES != 1 {
        // Deeper samples: scaled coefficients do not fit i16 — the whole
        // path in i32.
        let mut c = [0i32; 16];
        if dequant4_i32(levels, scale, qp, dc, &mut c) {
            idct4_add_i32(dst, stride, &c, max);
        } else if c[0] != 0 {
            dc_add_scalar(dst, stride, c[0], 4, max);
        }
        return;
    }
    let mut c = [0i16; 16];
    if dequant4_scalar(levels, scale, qp, dc, &mut c) {
        idct4_add_scalar(dst, stride, &c, max);
    } else if c[0] != 0 {
        dc_add_scalar(dst, stride, c[0] as i32, 4, max);
    }
}

fn residual8_scalar<S: Sample>(dst: &mut [S], stride: usize, levels: &[i32; 64], scale: &[i32; 64], qp: i32, max: i32) {
    if S::BYTES != 1 {
        let mut c = [0i32; 64];
        if dequant8_i32(levels, scale, qp, &mut c) {
            idct8_add_i32(dst, stride, &c, max);
        } else if c[0] != 0 {
            dc_add_scalar(dst, stride, c[0], 8, max);
        }
        return;
    }
    let mut c = [0i16; 64];
    if dequant8_scalar(levels, scale, qp, &mut c) {
        idct8_add_scalar(dst, stride, &c, max);
    } else if c[0] != 0 {
        dc_add_scalar(dst, stride, c[0] as i32, 8, max);
    }
}

/// Dequantisation of a 4x4 block into i32 (deeper samples); returns whether
/// any AC coefficient is nonzero.
#[inline]
fn dequant4_i32(levels: &[i32; 16], scale: &[i32; 16], qp: i32, dc: i32, out: &mut [i32; 16]) -> bool {
    let q6 = qp / 6;
    let start = if dc != NO_DC { 1 } else { 0 };
    let mut ac = 0i32;
    if qp >= 24 {
        let sh = q6 - 4;
        for i in start..16 {
            let v = (levels[i] * scale[i]) << sh;
            out[i] = v;
            ac |= v & (-((i != 0) as i32));
        }
    } else {
        let sh = 4 - q6;
        let round = 1 << (3 - q6);
        for i in start..16 {
            let v = (levels[i] * scale[i] + round) >> sh;
            out[i] = v;
            ac |= v & (-((i != 0) as i32));
        }
    }
    if dc != NO_DC {
        out[0] = dc;
    }
    ac != 0
}

/// Dequantisation of an 8x8 block into i32 (deeper samples).
#[inline]
fn dequant8_i32(levels: &[i32; 64], scale: &[i32; 64], qp: i32, out: &mut [i32; 64]) -> bool {
    let q6 = qp / 6;
    let mut ac = 0i32;
    if qp >= 36 {
        let sh = q6 - 6;
        for i in 0..64 {
            let v = (levels[i] * scale[i]) << sh;
            out[i] = v;
            ac |= v & (-((i != 0) as i32));
        }
    } else {
        let sh = 6 - q6;
        let round = 1 << (5 - q6);
        for i in 0..64 {
            let v = (levels[i] * scale[i] + round) >> sh;
            out[i] = v;
            ac |= v & (-((i != 0) as i32));
        }
    }
    ac != 0
}
