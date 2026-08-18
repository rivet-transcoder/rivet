//! H.264 kernels: quarter-sample luma interpolation (8.4.2.2.1), chroma
//! bilinear interpolation (8.4.2.2.2), sample combination and weighting
//! (8.4.2.3), on 8-bit planes. Scalar reference here; SIMD versions are
//! installed by [`super::h264_avx2`] / [`super::h264_neon`].
//!
//! Prediction blocks are 8-bit at every stage in H.264 (each interpolation
//! position rounds and clips to a sample), so kernels produce `u8` blocks and
//! the combiners average / weight those. A prediction block lives in the
//! top-left of a 16x16 scratch buffer with row stride [`PRED_STRIDE`], whatever
//! its size: a kernel may read and write the whole 16-sample row, so the SIMD
//! versions need no tail handling for 4- and 8-wide blocks.

use super::Cpu;

/// Row stride of a prediction scratch block (bytes). Every prediction buffer
/// is at least `16 * PRED_STRIDE` bytes.
pub const PRED_STRIDE: usize = 16;

/// Interpolate a `w x h` luma block into a [`PRED_STRIDE`]-strided scratch
/// block. `src` points at the top-left of the six-tap window: two samples left
/// of and two rows above the block's full-sample position; `src_stride` is the
/// plane stride.
pub type QpelFn = fn(dst: &mut [u8], src: &[u8], src_stride: usize, w: usize, h: usize);
/// Chroma bilinear into a [`PRED_STRIDE`]-strided scratch block: `src` at the
/// block's integer chroma position, fractions `xf`/`yf` in eighths.
pub type ChromaFn = fn(dst: &mut [u8], src: &[u8], src_stride: usize, w: usize, h: usize, xf: i32, yf: i32);
/// `dst = src` (a [`PRED_STRIDE`]-strided scratch block into a strided plane).
pub type CopyFn = fn(dst: &mut [u8], stride: usize, src: &[u8], w: usize, h: usize);
/// `dst = (a + b + 1) >> 1` (both [`PRED_STRIDE`]-strided scratch blocks).
pub type AvgFn = fn(dst: &mut [u8], stride: usize, a: &[u8], b: &[u8], w: usize, h: usize);
/// `dst = clip(((src * w + round) >> log_wd) + o)` (8-278 / 8-279).
pub type WeightedUniFn = fn(dst: &mut [u8], stride: usize, src: &[u8], w: usize, h: usize, log_wd: i32, wt: i32, o: i32);
/// `dst = clip(((a * w0 + b * w1 + 2^log_wd) >> (log_wd + 1)) + ((o0 + o1 + 1) >> 1))` (8-280).
pub type WeightedBiFn = fn(dst: &mut [u8], stride: usize, a: &[u8], b: &[u8], w: usize, h: usize, log_wd: i32, w0: i32, w1: i32, o0: i32, o1: i32);

/// Deblock sixteen lines of a luma edge with bS < 4 (8.7.2.3). `off` is the
/// offset of q0 on the first line; a *vertical* edge (`_v`) has its lines
/// `stride` apart and p/q samples 1 apart, a *horizontal* edge (`_h`) the
/// other way round. `tc0[i / 4]` is the segment's tC0, or −1 for bS 0
/// (leave the line alone).
pub type LumaDeblockFn = fn(data: &mut [u8], off: usize, stride: usize, alpha: i32, beta: i32, tc0: &[i8; 4]);
/// Deblock sixteen lines of a luma edge with bS 4 (8.7.2.4).
pub type LumaDeblockIntraFn = fn(data: &mut [u8], off: usize, stride: usize, alpha: i32, beta: i32);
/// Deblock eight lines of a 4:2:0 chroma edge with bS < 4; `tc0[i / 2]`.
pub type ChromaDeblockFn = fn(data: &mut [u8], off: usize, stride: usize, alpha: i32, beta: i32, tc0: &[i8; 4]);
/// Deblock eight lines of a 4:2:0 chroma edge with bS 4.
pub type ChromaDeblockIntraFn = fn(data: &mut [u8], off: usize, stride: usize, alpha: i32, beta: i32);

/// Inverse 4x4 transform (8.5.12.2) of dequantised coefficients in raster
/// order, added to the prediction in `dst` with clipping.
pub type Idct4AddFn = fn(dst: &mut [u8], stride: usize, coeffs: &[i16; 16]);
/// Inverse 8x8 transform (8.5.13.2), added with clipping.
pub type Idct8AddFn = fn(dst: &mut [u8], stride: usize, coeffs: &[i16; 64]);
/// A DC-only block: `(dc + 32) >> 6` added to every sample of the 4x4 / 8x8.
pub type DcAddFn = fn(dst: &mut [u8], stride: usize, dc: i32);

/// The kernel table.
#[derive(Clone, Copy)]
pub struct H264Dsp {
    /// Which CPU features the table was built for.
    pub cpu: Cpu,
    /// Luma interpolation by position `yf * 4 + xf`.
    pub qpel: [QpelFn; 16],
    /// Chroma bilinear.
    pub chroma: ChromaFn,
    /// Copy a packed block into the plane.
    pub copy: CopyFn,
    /// Rounded average of two packed blocks into the plane.
    pub avg: AvgFn,
    /// Explicit/implicit weighting, one list.
    pub weighted_uni: WeightedUniFn,
    /// Explicit/implicit weighting, both lists.
    pub weighted_bi: WeightedBiFn,
    /// Deblocking: luma vertical edge (bS < 4).
    pub deblock_luma_v: LumaDeblockFn,
    /// Deblocking: luma horizontal edge (bS < 4).
    pub deblock_luma_h: LumaDeblockFn,
    /// Deblocking: luma vertical edge (bS 4).
    pub deblock_luma_v_intra: LumaDeblockIntraFn,
    /// Deblocking: luma horizontal edge (bS 4).
    pub deblock_luma_h_intra: LumaDeblockIntraFn,
    /// Deblocking: chroma vertical edge (bS < 4).
    pub deblock_chroma_v: ChromaDeblockFn,
    /// Deblocking: chroma horizontal edge (bS < 4).
    pub deblock_chroma_h: ChromaDeblockFn,
    /// Deblocking: chroma vertical edge (bS 4).
    pub deblock_chroma_v_intra: ChromaDeblockIntraFn,
    /// Deblocking: chroma horizontal edge (bS 4).
    pub deblock_chroma_h_intra: ChromaDeblockIntraFn,
    /// Inverse 4x4 transform + add.
    pub idct4_add: Idct4AddFn,
    /// Inverse 8x8 transform + add.
    pub idct8_add: Idct8AddFn,
    /// DC-only 4x4 add.
    pub idct4_dc_add: DcAddFn,
    /// DC-only 8x8 add.
    pub idct8_dc_add: DcAddFn,
}

impl H264Dsp {
    /// The scalar reference table.
    pub const SCALAR: H264Dsp = H264Dsp {
        cpu: Cpu::SCALAR,
        qpel: [
            qpel_scalar::<0, 0>,
            qpel_scalar::<1, 0>,
            qpel_scalar::<2, 0>,
            qpel_scalar::<3, 0>,
            qpel_scalar::<0, 1>,
            qpel_scalar::<1, 1>,
            qpel_scalar::<2, 1>,
            qpel_scalar::<3, 1>,
            qpel_scalar::<0, 2>,
            qpel_scalar::<1, 2>,
            qpel_scalar::<2, 2>,
            qpel_scalar::<3, 2>,
            qpel_scalar::<0, 3>,
            qpel_scalar::<1, 3>,
            qpel_scalar::<2, 3>,
            qpel_scalar::<3, 3>,
        ],
        chroma: chroma_scalar,
        copy: copy_scalar,
        avg: avg_scalar,
        weighted_uni: weighted_uni_scalar,
        weighted_bi: weighted_bi_scalar,
        deblock_luma_v: |d, off, stride, a, b, tc0| deblock_luma_scalar(d, off, stride, 1, a, b, Some(tc0)),
        deblock_luma_h: |d, off, stride, a, b, tc0| deblock_luma_scalar(d, off, 1, stride, a, b, Some(tc0)),
        deblock_luma_v_intra: |d, off, stride, a, b| deblock_luma_scalar(d, off, stride, 1, a, b, None),
        deblock_luma_h_intra: |d, off, stride, a, b| deblock_luma_scalar(d, off, 1, stride, a, b, None),
        deblock_chroma_v: |d, off, stride, a, b, tc0| deblock_chroma_scalar(d, off, stride, 1, a, b, Some(tc0)),
        deblock_chroma_h: |d, off, stride, a, b, tc0| deblock_chroma_scalar(d, off, 1, stride, a, b, Some(tc0)),
        deblock_chroma_v_intra: |d, off, stride, a, b| deblock_chroma_scalar(d, off, stride, 1, a, b, None),
        deblock_chroma_h_intra: |d, off, stride, a, b| deblock_chroma_scalar(d, off, 1, stride, a, b, None),
        idct4_add: idct4_add_scalar,
        idct8_add: idct8_add_scalar,
        idct4_dc_add: |d, s, dc| dc_add_scalar(d, s, dc, 4),
        idct8_dc_add: |d, s, dc| dc_add_scalar(d, s, dc, 8),
    };

    /// The best table for `cpu`.
    pub fn new(cpu: Cpu) -> H264Dsp {
        let mut d = H264Dsp::SCALAR;
        d.cpu = cpu;
        #[cfg(target_arch = "x86_64")]
        if cpu.avx2 {
            super::h264_avx2::install(&mut d);
        }
        #[cfg(target_arch = "aarch64")]
        if cpu.neon {
            super::h264_neon::install(&mut d);
        }
        d
    }
}

// ----------------------------------------------------------------------
// Luma interpolation (scalar)
// ----------------------------------------------------------------------

#[inline(always)]
fn tap6(a: i32, b: i32, c: i32, d: i32, e: i32, f: i32) -> i32 {
    a - 5 * b + 20 * c + 20 * d - 5 * e + f
}

#[inline(always)]
fn clip8(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

/// The block's full samples `G` at `(x, y)` in window coordinates.
#[inline(always)]
fn g(src: &[u8], stride: usize, x: usize, y: usize) -> i32 {
    src[(y + 2) * stride + x + 2] as i32
}

/// Horizontal six-tap intermediate `b1` at window row `yy` (window
/// coordinates: row 0 = two above the block), block column `x`.
#[inline(always)]
fn b1(src: &[u8], stride: usize, x: usize, yy: usize) -> i32 {
    let r = &src[yy * stride + x..];
    tap6(r[0] as i32, r[1] as i32, r[2] as i32, r[3] as i32, r[4] as i32, r[5] as i32)
}

/// Vertical six-tap intermediate `h1` at window column `xx`, block row `y`.
#[inline(always)]
fn h1(src: &[u8], stride: usize, xx: usize, y: usize) -> i32 {
    let c = &src[y * stride + xx..];
    tap6(c[0] as i32, c[stride] as i32, c[2 * stride] as i32, c[3 * stride] as i32, c[4 * stride] as i32, c[5 * stride] as i32)
}

/// Scalar interpolation for position `(XF, YF)`.
fn qpel_scalar<const XF: usize, const YF: usize>(dst: &mut [u8], src: &[u8], stride: usize, w: usize, h: usize) {
    // b: half-sample horizontally at block (x, y): b1 at window row y + 2.
    let b = |x: usize, y: usize| clip8((b1(src, stride, x, y + 2) + 16) >> 5) as i32;
    // hh: half-sample vertically at block (x, y): h1 at window column x + 2.
    let hh = |x: usize, y: usize| clip8((h1(src, stride, x + 2, y) + 16) >> 5) as i32;
    // j: centre, vertical six-tap over b1 rows y..y+5 (window rows).
    let j = |x: usize, y: usize| {
        let j1 = tap6(b1(src, stride, x, y), b1(src, stride, x, y + 1), b1(src, stride, x, y + 2), b1(src, stride, x, y + 3), b1(src, stride, x, y + 4), b1(src, stride, x, y + 5));
        clip8((j1 + 512) >> 10) as i32
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
            dst[y * PRED_STRIDE + x] = v as u8;
        }
    }
}

// ----------------------------------------------------------------------
// Chroma / combination (scalar)
// ----------------------------------------------------------------------

fn chroma_scalar(dst: &mut [u8], src: &[u8], stride: usize, w: usize, h: usize, xf: i32, yf: i32) {
    let (a, b, c, d) = ((8 - xf) * (8 - yf), xf * (8 - yf), (8 - xf) * yf, xf * yf);
    for y in 0..h {
        let r0 = &src[y * stride..];
        let r1 = &src[(y + 1) * stride..];
        for x in 0..w {
            let v = a * r0[x] as i32 + b * r0[x + 1] as i32 + c * r1[x] as i32 + d * r1[x + 1] as i32;
            dst[y * PRED_STRIDE + x] = ((v + 32) >> 6) as u8;
        }
    }
}

fn copy_scalar(dst: &mut [u8], stride: usize, src: &[u8], w: usize, h: usize) {
    for y in 0..h {
        dst[y * stride..y * stride + w].copy_from_slice(&src[y * PRED_STRIDE..y * PRED_STRIDE + w]);
    }
}

fn avg_scalar(dst: &mut [u8], stride: usize, a: &[u8], b: &[u8], w: usize, h: usize) {
    for y in 0..h {
        for x in 0..w {
            dst[y * stride + x] = ((a[y * PRED_STRIDE + x] as u16 + b[y * PRED_STRIDE + x] as u16 + 1) >> 1) as u8;
        }
    }
}

fn weighted_uni_scalar(dst: &mut [u8], stride: usize, src: &[u8], w: usize, h: usize, log_wd: i32, wt: i32, o: i32) {
    if log_wd >= 1 {
        let round = 1 << (log_wd - 1);
        for y in 0..h {
            for x in 0..w {
                dst[y * stride + x] = clip8(((src[y * PRED_STRIDE + x] as i32 * wt + round) >> log_wd) + o);
            }
        }
    } else {
        for y in 0..h {
            for x in 0..w {
                dst[y * stride + x] = clip8(src[y * PRED_STRIDE + x] as i32 * wt + o);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn weighted_bi_scalar(dst: &mut [u8], stride: usize, a: &[u8], b: &[u8], w: usize, h: usize, log_wd: i32, w0: i32, w1: i32, o0: i32, o1: i32) {
    let off = (o0 + o1 + 1) >> 1;
    let round = 1 << log_wd;
    for y in 0..h {
        for x in 0..w {
            let v = ((a[y * PRED_STRIDE + x] as i32 * w0 + b[y * PRED_STRIDE + x] as i32 * w1 + round) >> (log_wd + 1)) + off;
            dst[y * stride + x] = clip8(v);
        }
    }
}

// ----------------------------------------------------------------------
// Deblocking (scalar)
// ----------------------------------------------------------------------

/// Filter one line of samples across an edge (8.7.2.3 / 8.7.2.4).
/// `p[0..4]` are p0..p3 (p0 nearest the edge), `q[0..4]` are q0..q3.
/// `tc0 == None` is bS 4.
#[inline]
pub(crate) fn deblock_line(p: &mut [i32; 4], q: &mut [i32; 4], tc0: Option<i32>, alpha: i32, beta: i32, chroma: bool) {
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
        p[0] = (p0 + delta).clamp(0, 255);
        q[0] = (q0 - delta).clamp(0, 255);
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
fn deblock_luma_scalar(data: &mut [u8], off: usize, step: usize, across: usize, alpha: i32, beta: i32, tc0: Option<&[i8; 4]>) {
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
            p[k] = data[base - (k + 1) * across] as i32;
            q[k] = data[base + k * across] as i32;
        }
        deblock_line(&mut p, &mut q, t, alpha, beta, false);
        for k in 0..3 {
            data[base - (k + 1) * across] = p[k] as u8;
            data[base + k * across] = q[k] as u8;
        }
    }
}

/// Eight chroma lines (4:2:0): line `i` uses `tc0[i / 2]`.
fn deblock_chroma_scalar(data: &mut [u8], off: usize, step: usize, across: usize, alpha: i32, beta: i32, tc0: Option<&[i8; 4]>) {
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
            p[k] = data[base - (k + 1) * across] as i32;
            q[k] = data[base + k * across] as i32;
        }
        deblock_line(&mut p, &mut q, t, alpha, beta, true);
        data[base - across] = p[0] as u8;
        data[base] = q[0] as u8;
    }
}

// ----------------------------------------------------------------------
// Inverse transforms (scalar)
// ----------------------------------------------------------------------

fn idct4_add_scalar(dst: &mut [u8], stride: usize, coeffs: &[i16; 16]) {
    let mut c = [0i32; 16];
    for i in 0..16 {
        c[i] = coeffs[i] as i32;
    }
    crate::h264::transform::idct4x4(&mut c);
    crate::h264::transform::add_residual(dst, stride, &c, 4);
}

fn idct8_add_scalar(dst: &mut [u8], stride: usize, coeffs: &[i16; 64]) {
    let mut c = [0i32; 64];
    for i in 0..64 {
        c[i] = coeffs[i] as i32;
    }
    crate::h264::transform::idct8x8(&mut c);
    crate::h264::transform::add_residual(dst, stride, &c, 8);
}

fn dc_add_scalar(dst: &mut [u8], stride: usize, dc: i32, n: usize) {
    let v = (dc + 32) >> 6;
    for y in 0..n {
        for x in 0..n {
            let p = &mut dst[y * stride + x];
            *p = (*p as i32 + v).clamp(0, 255) as u8;
        }
    }
}
