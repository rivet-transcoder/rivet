//! H.265 kernels: inverse transforms, residual add, interpolation filters,
//! sample combination / weighting, SAO. Every kernel has a scalar reference
//! here; [`HevcDsp::new`] swaps in the SIMD versions from
//! [`super::hevc_avx2`] / [`super::hevc_neon`] when the CPU has them.
//!
//! Sample planes are `u16` at any bit depth; interpolation intermediates are
//! `i16` at 14-bit precision (8.5.3.3.3), coefficients and residuals `i16`
//! (the standard clips them to 16 bits, 8.6.2 / 8.6.4.2).

use super::Cpu;
use crate::hevc::frame::Sample;
use crate::hevc::tables::{EPEL_FILTERS, QPEL_FILTERS, TRANSFORM32};

/// Inverse DCT of an `n x n` block in place: `coeffs` (raster, scaled
/// coefficients) becomes residual samples. `bd_shift` = 20 − BitDepth;
/// `max_x` / `max_y` bound the nonzero coefficients (inclusive).
pub type IdctFn = fn(coeffs: &mut [i16], bd_shift: i32, max_x: usize, max_y: usize);
/// Add a residual block to `dst` with clipping to `0..=max`.
pub type AddResidualFn<S> = fn(dst: &mut [S], stride: usize, res: &[i16], n: usize, max: i32);
/// Interpolate a `w x h` block. `src` points at the first tap of the first
/// output sample (3 samples / rows before the block for luma, 1 for chroma),
/// `frac` is the sub-sample phase (1..=3 luma, 1..=7 chroma), `shift` the
/// normalisation shift (shift1 = Min(4, BitDepth − 8)).
pub type InterpFn<S> = fn(dst: &mut [i16], src: &[S], src_stride: usize, w: usize, h: usize, frac: usize, shift: i32);
/// Copy with `<< shift` into the 14-bit domain.
pub type CopyFn<S> = fn(dst: &mut [i16], src: &[S], src_stride: usize, w: usize, h: usize, shift: i32);
/// Uni-prediction: `dst = clip((src + round) >> shift)`.
pub type UniFn<S> = fn(dst: &mut [S], stride: usize, src: &[i16], w: usize, h: usize, shift: i32, max: i32);
/// Bi-prediction: `dst = clip((a + b + round) >> shift)`.
pub type BiFn<S> = fn(dst: &mut [S], stride: usize, a: &[i16], b: &[i16], w: usize, h: usize, shift: i32, max: i32);
/// Weighted uni-prediction: `dst = clip(((src * w + round) >> log2wd) + o)`.
pub type WeightedUniFn<S> = fn(dst: &mut [S], stride: usize, src: &[i16], w: usize, h: usize, log2_wd: i32, wt: i32, o: i32, max: i32);
/// Weighted bi-prediction: `dst = clip((a * w0 + b * w1 + ((o0 + o1 + 1) << log2wd)) >> (log2wd + 1))`.
pub type WeightedBiFn<S> =
    fn(dst: &mut [S], stride: usize, a: &[i16], b: &[i16], w: usize, h: usize, log2_wd: i32, w0: i32, w1: i32, o0: i32, o1: i32, max: i32);
/// SAO band offset over a `w x h` region: `dst[i] = clip(src[i] + table[src[i] >> shift])`.
pub type SaoBandFn<S> = fn(dst: &mut [S], dst_stride: usize, src: &[S], src_stride: usize, w: usize, h: usize, table: &[i16; 32], shift: i32, max: i32);
/// SAO edge offset over a `w x h` region at index `origin` of `dst` and `src`
/// (same geometry, `stride`), whose two neighbours at offsets `na` / `nb`
/// (in samples, relative to the sample) are all inside the picture and
/// usable: `dst = clip(src + off[2 + sign(src - a) + sign(src - b)])` with
/// `off` indexed by the raw edgeIdx (0..=4, index 2 = 0).
pub type SaoEdgeFn<S> = fn(dst: &mut [S], src: &[S], origin: usize, stride: usize, w: usize, h: usize, na: isize, nb: isize, off: &[i16; 5], max: i32);

/// The kernel table.
#[derive(Clone, Copy)]
pub struct HevcDsp<S: Sample = u16> {
    /// Which CPU features the table was built for.
    pub cpu: Cpu,
    /// Inverse DCT by `log2 - 2`.
    pub idct: [IdctFn; 4],
    /// Inverse DST 4x4 (intra luma 4x4).
    pub idst4: IdctFn,
    /// Residual add.
    pub add_residual: AddResidualFn<S>,
    /// Luma copy / horizontal / vertical 8-tap.
    pub qpel_copy: CopyFn<S>,
    /// See `qpel_copy`.
    pub qpel_h: InterpFn<S>,
    /// See `qpel_copy`.
    pub qpel_v: InterpFn<S>,
    /// Vertical 8-tap over 14-bit intermediates (second stage of hv).
    pub qpel_v2: fn(dst: &mut [i16], src: &[i16], src_stride: usize, w: usize, h: usize, frac: usize),
    /// Chroma copy / horizontal / vertical 4-tap.
    pub epel_copy: CopyFn<S>,
    /// See `epel_copy`.
    pub epel_h: InterpFn<S>,
    /// See `epel_copy`.
    pub epel_v: InterpFn<S>,
    /// Vertical 4-tap over 14-bit intermediates.
    pub epel_v2: fn(dst: &mut [i16], src: &[i16], src_stride: usize, w: usize, h: usize, frac: usize),
    /// Sample combination.
    pub uni: UniFn<S>,
    /// See `uni`.
    pub bi: BiFn<S>,
    /// See `uni`.
    pub weighted_uni: WeightedUniFn<S>,
    /// See `uni`.
    pub weighted_bi: WeightedBiFn<S>,
    /// SAO.
    pub sao_band: SaoBandFn<S>,
    /// See `sao_band`.
    pub sao_edge: SaoEdgeFn<S>,
    /// Deblocking: two 4-line luma segments of a vertical edge.
    pub deblock_luma_v: LumaDeblockFn<S>,
    /// Deblocking: two 4-line luma segments of a horizontal edge.
    pub deblock_luma_h: LumaDeblockFn<S>,
    /// Deblocking: four 2-line chroma segments of a vertical edge.
    pub deblock_chroma_v: ChromaDeblockFn<S>,
    /// Deblocking: four 2-line chroma segments of a horizontal edge.
    pub deblock_chroma_h: ChromaDeblockFn<S>,
}

/// Deblock eight lines of a luma edge — two 4-line segments with their own
/// `beta`, `tc` and p/q exemptions (8.7.2.5.3–7); a segment with `tc == 0`
/// and `beta == 0` is left alone. `off` is the offset of q0 on the first
/// line; a vertical edge (`_v`) has its lines `stride` apart and p/q
/// samples 1 apart, a horizontal edge (`_h`) the other way round.
pub type LumaDeblockFn<S> = fn(data: &mut [S], off: usize, stride: usize, beta: [i32; 2], tc: [i32; 2], no_p: [bool; 2], no_q: [bool; 2], max: i32);
/// Deblock eight lines of a 4:2:0 chroma edge — four 2-line segments with
/// their own `tc` (0 = leave alone) and exemptions (8.7.2.5.5).
pub type ChromaDeblockFn<S> = fn(data: &mut [S], off: usize, stride: usize, tc: [i32; 4], no_p: [bool; 4], no_q: [bool; 4], max: i32);

impl<S: Sample> HevcDsp<S> {
    /// The scalar reference table.
    pub fn scalar() -> Self {
        HevcDsp {
            cpu: Cpu::SCALAR,
            idct: [idct_scalar::<4>, idct_scalar::<8>, idct_scalar::<16>, idct_scalar::<32>],
            idst4: idst4_scalar,
            add_residual: add_residual_scalar::<S>,
            qpel_copy: copy_scalar::<S>,
            qpel_h: qpel_h_scalar::<S>,
            qpel_v: qpel_v_scalar::<S>,
            qpel_v2: qpel_v2_scalar,
            epel_copy: copy_scalar::<S>,
            epel_h: epel_h_scalar::<S>,
            epel_v: epel_v_scalar::<S>,
            epel_v2: epel_v2_scalar,
            uni: uni_scalar::<S>,
            bi: bi_scalar::<S>,
            weighted_uni: weighted_uni_scalar::<S>,
            weighted_bi: weighted_bi_scalar::<S>,
            sao_band: sao_band_scalar::<S>,
            sao_edge: sao_edge_scalar::<S>,
            deblock_luma_v: |d, off, stride, beta, tc, np, nq, max| deblock_luma_scalar(d, off, 1, stride, beta, tc, np, nq, max),
            deblock_luma_h: |d, off, stride, beta, tc, np, nq, max| deblock_luma_scalar(d, off, stride, 1, beta, tc, np, nq, max),
            deblock_chroma_v: |d, off, stride, tc, np, nq, max| deblock_chroma_scalar(d, off, 1, stride, tc, np, nq, max),
            deblock_chroma_h: |d, off, stride, tc, np, nq, max| deblock_chroma_scalar(d, off, stride, 1, tc, np, nq, max),
        }
    }

    /// The best table for `cpu`.
    pub fn new(cpu: Cpu) -> Self {
        let mut d = Self::scalar();
        d.cpu = cpu;
        S::install_simd(&mut d, cpu);
        d
    }
}

impl HevcDsp<u16> {
    /// The scalar reference table (16-bit samples).
    pub const SCALAR: HevcDsp<u16> = HevcDsp {
        cpu: Cpu::SCALAR,
        idct: [idct_scalar::<4>, idct_scalar::<8>, idct_scalar::<16>, idct_scalar::<32>],
        idst4: idst4_scalar,
        add_residual: add_residual_scalar::<u16>,
        qpel_copy: copy_scalar::<u16>,
        qpel_h: qpel_h_scalar::<u16>,
        qpel_v: qpel_v_scalar::<u16>,
        qpel_v2: qpel_v2_scalar,
        epel_copy: copy_scalar::<u16>,
        epel_h: epel_h_scalar::<u16>,
        epel_v: epel_v_scalar::<u16>,
        epel_v2: epel_v2_scalar,
        uni: uni_scalar::<u16>,
        bi: bi_scalar::<u16>,
        weighted_uni: weighted_uni_scalar::<u16>,
        weighted_bi: weighted_bi_scalar::<u16>,
        sao_band: sao_band_scalar::<u16>,
        sao_edge: sao_edge_scalar::<u16>,
        deblock_luma_v: |d, off, stride, beta, tc, np, nq, max| deblock_luma_scalar(d, off, 1, stride, beta, tc, np, nq, max),
        deblock_luma_h: |d, off, stride, beta, tc, np, nq, max| deblock_luma_scalar(d, off, stride, 1, beta, tc, np, nq, max),
        deblock_chroma_v: |d, off, stride, tc, np, nq, max| deblock_chroma_scalar(d, off, 1, stride, tc, np, nq, max),
        deblock_chroma_h: |d, off, stride, tc, np, nq, max| deblock_chroma_scalar(d, off, stride, 1, tc, np, nq, max),
    };
}

impl HevcDsp<u8> {
    /// The scalar reference table (8-bit samples).
    pub const SCALAR: HevcDsp<u8> = HevcDsp {
        cpu: Cpu::SCALAR,
        idct: [idct_scalar::<4>, idct_scalar::<8>, idct_scalar::<16>, idct_scalar::<32>],
        idst4: idst4_scalar,
        add_residual: add_residual_scalar::<u8>,
        qpel_copy: copy_scalar::<u8>,
        qpel_h: qpel_h_scalar::<u8>,
        qpel_v: qpel_v_scalar::<u8>,
        qpel_v2: qpel_v2_scalar,
        epel_copy: copy_scalar::<u8>,
        epel_h: epel_h_scalar::<u8>,
        epel_v: epel_v_scalar::<u8>,
        epel_v2: epel_v2_scalar,
        uni: uni_scalar::<u8>,
        bi: bi_scalar::<u8>,
        weighted_uni: weighted_uni_scalar::<u8>,
        weighted_bi: weighted_bi_scalar::<u8>,
        sao_band: sao_band_scalar::<u8>,
        sao_edge: sao_edge_scalar::<u8>,
        deblock_luma_v: |d, off, stride, beta, tc, np, nq, max| deblock_luma_scalar(d, off, 1, stride, beta, tc, np, nq, max),
        deblock_luma_h: |d, off, stride, beta, tc, np, nq, max| deblock_luma_scalar(d, off, stride, 1, beta, tc, np, nq, max),
        deblock_chroma_v: |d, off, stride, tc, np, nq, max| deblock_chroma_scalar(d, off, 1, stride, tc, np, nq, max),
        deblock_chroma_h: |d, off, stride, tc, np, nq, max| deblock_chroma_scalar(d, off, stride, 1, tc, np, nq, max),
    };
}

/// SIMD kernels for 16-bit sample planes.
#[allow(unused_variables)]
pub fn install_simd_u16(d: &mut HevcDsp<u16>, cpu: Cpu) {
    #[cfg(target_arch = "x86_64")]
    if cpu.avx2 {
        super::hevc_avx2::install(d);
    }
    #[cfg(target_arch = "aarch64")]
    if cpu.neon {
        super::hevc_neon::install(d);
    }
}

/// SIMD kernels for 8-bit sample planes.
#[allow(unused_variables)]
pub fn install_simd_u8(d: &mut HevcDsp<u8>, cpu: Cpu) {
    #[cfg(target_arch = "x86_64")]
    if cpu.avx2 {
        super::hevc_avx2_u8::install(d);
    }
    #[cfg(target_arch = "aarch64")]
    if cpu.neon {
        super::hevc_neon_u8::install(d);
    }
}

// ----------------------------------------------------------------------
// Inverse transforms
// ----------------------------------------------------------------------

/// 1-D inverse DCT of `n` coefficients (`src[k * stride]`, only the first
/// `nz` possibly nonzero) into `out[0..n]`, no shift — the even/odd partial
/// butterfly, which is the matrix product regrouped (exact).
#[inline]
fn idct1(src: &[i16], stride: usize, n: usize, nz: usize, out: &mut [i32]) {
    if n == 1 {
        out[0] = 64 * src[0] as i32;
        return;
    }
    let half = n / 2;
    let mut e = [0i32; 16];
    idct1(src, stride * 2, half, nz.div_ceil(2), &mut e);
    let step = 32 / n;
    for k in 0..half {
        let mut o = 0i32;
        let mut j = 1;
        while j < nz {
            o += TRANSFORM32[j * step][k] as i32 * src[j * stride] as i32;
            j += 2;
        }
        out[k] = e[k] + o;
        out[n - 1 - k] = e[k] - o;
    }
}

/// Scalar inverse DCT (8.6.4.2), columns then rows, with the 16-bit clip
/// after the first stage.
fn idct_scalar<const N: usize>(coeffs: &mut [i16], bd_shift: i32, max_x: usize, max_y: usize) {
    let round2 = 1i32 << (bd_shift - 1);
    if max_x == 0 && max_y == 0 {
        // DC only: every output sample is the same.
        let v = ((coeffs[0] as i32 * 64 + 64) >> 7).clamp(-32768, 32767);
        let r = ((v * 64 + round2) >> bd_shift).clamp(-32768, 32767) as i16;
        coeffs[..N * N].fill(r);
        return;
    }
    let mut tmp = [0i16; 32 * 32];
    let mut out = [0i32; 32];
    // Columns 0..=max_x; the rest stay zero.
    for x in 0..=max_x {
        idct1(&coeffs[x..], N, N, max_y + 1, &mut out);
        for y in 0..N {
            tmp[y * N + x] = ((out[y] + 64) >> 7).clamp(-32768, 32767) as i16;
        }
    }
    // Rows.
    for y in 0..N {
        idct1(&tmp[y * N..], 1, N, max_x + 1, &mut out);
        for x in 0..N {
            coeffs[y * N + x] = ((out[x] + round2) >> bd_shift).clamp(-32768, 32767) as i16;
        }
    }
}

/// Inverse DST 4x4 (8.6.4.2, `trType == 1`).
fn idst4_scalar(coeffs: &mut [i16], bd_shift: i32, _max_x: usize, _max_y: usize) {
    const M: [[i32; 4]; 4] = [[29, 55, 74, 84], [74, 74, 0, -74], [84, -29, -74, 55], [55, -84, 74, -29]];
    let round2 = 1i32 << (bd_shift - 1);
    let mut tmp = [0i16; 16];
    for x in 0..4 {
        for i in 0..4 {
            let mut s = 0i32;
            for j in 0..4 {
                s += M[j][i] * coeffs[j * 4 + x] as i32;
            }
            tmp[i * 4 + x] = ((s + 64) >> 7).clamp(-32768, 32767) as i16;
        }
    }
    for y in 0..4 {
        for i in 0..4 {
            let mut s = 0i32;
            for j in 0..4 {
                s += M[j][i] * tmp[y * 4 + j] as i32;
            }
            coeffs[y * 4 + i] = ((s + round2) >> bd_shift).clamp(-32768, 32767) as i16;
        }
    }
}

fn add_residual_scalar<S: Sample>(dst: &mut [S], stride: usize, res: &[i16], n: usize, max: i32) {
    for y in 0..n {
        let row = &mut dst[y * stride..y * stride + n];
        let r = &res[y * n..y * n + n];
        for x in 0..n {
            row[x] = S::from_i32((row[x].to_i32() + r[x] as i32).clamp(0, max));
        }
    }
}

// ----------------------------------------------------------------------
// Interpolation
// ----------------------------------------------------------------------

fn copy_scalar<S: Sample>(dst: &mut [i16], src: &[S], src_stride: usize, w: usize, h: usize, shift: i32) {
    for y in 0..h {
        let s = &src[y * src_stride..y * src_stride + w];
        let d = &mut dst[y * w..y * w + w];
        for x in 0..w {
            d[x] = (s[x].to_i32() << shift) as i16;
        }
    }
}

fn qpel_h_scalar<S: Sample>(dst: &mut [i16], src: &[S], src_stride: usize, w: usize, h: usize, frac: usize, shift: i32) {
    let f = &QPEL_FILTERS[frac];
    for y in 0..h {
        let base = y * src_stride;
        for x in 0..w {
            let mut acc = 0i32;
            for k in 0..8 {
                acc += f[k] as i32 * src[base + x + k].to_i32();
            }
            dst[y * w + x] = (acc >> shift) as i16;
        }
    }
}

fn qpel_v_scalar<S: Sample>(dst: &mut [i16], src: &[S], src_stride: usize, w: usize, h: usize, frac: usize, shift: i32) {
    let f = &QPEL_FILTERS[frac];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0i32;
            for k in 0..8 {
                acc += f[k] as i32 * src[(y + k) * src_stride + x].to_i32();
            }
            dst[y * w + x] = (acc >> shift) as i16;
        }
    }
}

fn qpel_v2_scalar(dst: &mut [i16], src: &[i16], src_stride: usize, w: usize, h: usize, frac: usize) {
    let f = &QPEL_FILTERS[frac];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0i32;
            for k in 0..8 {
                acc += f[k] as i32 * src[(y + k) * src_stride + x] as i32;
            }
            dst[y * w + x] = (acc >> 6) as i16;
        }
    }
}

fn epel_h_scalar<S: Sample>(dst: &mut [i16], src: &[S], src_stride: usize, w: usize, h: usize, frac: usize, shift: i32) {
    let f = &EPEL_FILTERS[frac];
    for y in 0..h {
        let base = y * src_stride;
        for x in 0..w {
            let mut acc = 0i32;
            for k in 0..4 {
                acc += f[k] as i32 * src[base + x + k].to_i32();
            }
            dst[y * w + x] = (acc >> shift) as i16;
        }
    }
}

fn epel_v_scalar<S: Sample>(dst: &mut [i16], src: &[S], src_stride: usize, w: usize, h: usize, frac: usize, shift: i32) {
    let f = &EPEL_FILTERS[frac];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0i32;
            for k in 0..4 {
                acc += f[k] as i32 * src[(y + k) * src_stride + x].to_i32();
            }
            dst[y * w + x] = (acc >> shift) as i16;
        }
    }
}

fn epel_v2_scalar(dst: &mut [i16], src: &[i16], src_stride: usize, w: usize, h: usize, frac: usize) {
    let f = &EPEL_FILTERS[frac];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0i32;
            for k in 0..4 {
                acc += f[k] as i32 * src[(y + k) * src_stride + x] as i32;
            }
            dst[y * w + x] = (acc >> 6) as i16;
        }
    }
}

// ----------------------------------------------------------------------
// Combination / weighting
// ----------------------------------------------------------------------

fn uni_scalar<S: Sample>(dst: &mut [S], stride: usize, src: &[i16], w: usize, h: usize, shift: i32, max: i32) {
    let round = if shift > 0 { 1 << (shift - 1) } else { 0 };
    for y in 0..h {
        for x in 0..w {
            dst[y * stride + x] = S::from_i32(((src[y * w + x] as i32 + round) >> shift).clamp(0, max));
        }
    }
}

fn bi_scalar<S: Sample>(dst: &mut [S], stride: usize, a: &[i16], b: &[i16], w: usize, h: usize, shift: i32, max: i32) {
    let round = 1 << (shift - 1);
    for y in 0..h {
        for x in 0..w {
            dst[y * stride + x] = S::from_i32(((a[y * w + x] as i32 + b[y * w + x] as i32 + round) >> shift).clamp(0, max));
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn weighted_uni_scalar<S: Sample>(dst: &mut [S], stride: usize, src: &[i16], w: usize, h: usize, log2_wd: i32, wt: i32, o: i32, max: i32) {
    if log2_wd >= 1 {
        let round = 1 << (log2_wd - 1);
        for y in 0..h {
            for x in 0..w {
                dst[y * stride + x] = S::from_i32((((src[y * w + x] as i32 * wt + round) >> log2_wd) + o).clamp(0, max));
            }
        }
    } else {
        for y in 0..h {
            for x in 0..w {
                dst[y * stride + x] = S::from_i32((src[y * w + x] as i32 * wt + o).clamp(0, max));
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn weighted_bi_scalar<S: Sample>(dst: &mut [S], stride: usize, a: &[i16], b: &[i16], w: usize, h: usize, log2_wd: i32, w0: i32, w1: i32, o0: i32, o1: i32, max: i32) {
    let round = (o0 + o1 + 1) << log2_wd;
    for y in 0..h {
        for x in 0..w {
            let v = (a[y * w + x] as i32 * w0 + b[y * w + x] as i32 * w1 + round) >> (log2_wd + 1);
            dst[y * stride + x] = S::from_i32(v.clamp(0, max));
        }
    }
}

// ----------------------------------------------------------------------
// SAO
// ----------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn sao_band_scalar<S: Sample>(dst: &mut [S], dst_stride: usize, src: &[S], src_stride: usize, w: usize, h: usize, table: &[i16; 32], shift: i32, max: i32) {
    for y in 0..h {
        for x in 0..w {
            let v = src[y * src_stride + x].to_i32();
            dst[y * dst_stride + x] = S::from_i32((v + table[(v >> shift) as usize] as i32).clamp(0, max));
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn sao_edge_scalar<S: Sample>(dst: &mut [S], src: &[S], origin: usize, stride: usize, w: usize, h: usize, na: isize, nb: isize, off: &[i16; 5], max: i32) {
    for y in 0..h {
        for x in 0..w {
            let i = origin + y * stride + x;
            let v = src[i].to_i32();
            let a = src[(i as isize + na) as usize].to_i32();
            let b = src[(i as isize + nb) as usize].to_i32();
            let e = (2 + (v - a).signum() + (v - b).signum()) as usize;
            dst[i] = S::from_i32((v + off[e] as i32).clamp(0, max));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The plain matrix product the butterfly must equal.
    fn idct_matrix(n: usize, coeffs: &[i16], bd_shift: i32) -> Vec<i16> {
        let step = 32 / n;
        let mut tmp = vec![0i32; n * n];
        for x in 0..n {
            for i in 0..n {
                let mut s = 0i64;
                for j in 0..n {
                    s += TRANSFORM32[j * step][i] as i64 * coeffs[j * n + x] as i64;
                }
                tmp[i * n + x] = ((s + 64) >> 7).clamp(-32768, 32767) as i32;
            }
        }
        let mut out = vec![0i16; n * n];
        let round = 1i64 << (bd_shift - 1);
        for y in 0..n {
            for i in 0..n {
                let mut s = 0i64;
                for j in 0..n {
                    s += TRANSFORM32[j * step][i] as i64 * tmp[y * n + j] as i64;
                }
                out[y * n + i] = ((s + round) >> bd_shift).clamp(-32768, 32767) as i16;
            }
        }
        out
    }

    fn lcg(seed: &mut u64) -> u32 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (*seed >> 33) as u32
    }

    #[test]
    fn butterfly_equals_matrix() {
        let mut seed = 7u64;
        for &(n, log2) in &[(4usize, 2u32), (8, 3), (16, 4), (32, 5)] {
            for trial in 0..200 {
                let mut c = vec![0i16; n * n];
                // Sparse blocks with a bounding box, plus dense ones.
                let (mx, my) = if trial % 3 == 0 { (n - 1, n - 1) } else { ((lcg(&mut seed) as usize) % n, (lcg(&mut seed) as usize) % n) };
                for y in 0..=my {
                    for x in 0..=mx {
                        if lcg(&mut seed) % 3 == 0 {
                            c[y * n + x] = (lcg(&mut seed) as i32 % 65536 - 32768) as i16;
                        }
                    }
                }
                let bd_shift = 20 - 8 - (trial % 3) as i32 * 2;
                let want = idct_matrix(n, &c, bd_shift);
                let mut got = c.clone();
                (HevcDsp::<u16>::SCALAR.idct[(log2 - 2) as usize])(&mut got, bd_shift, mx, my);
                assert_eq!(got, want, "n={n} trial={trial}");
            }
        }
    }
}

// ----------------------------------------------------------------------
// Deblocking (scalar)
// ----------------------------------------------------------------------

/// Filter one 4-line luma edge segment. `pos` is the offset of q0 of the
/// first line, `step` the distance across the edge, `along` the distance
/// between lines.
#[allow(clippy::too_many_arguments)]
pub(crate) fn luma_edge_scalar<S: Sample>(d: &mut [S], pos: usize, step: usize, along: usize, beta: i32, tc: i32, no_p: bool, no_q: bool, max: i32) {
    let s = |d: &[S], line: usize, k: isize| -> i32 { d[(pos as isize + (line * along) as isize + k * step as isize) as usize].to_i32() };
    let dp0 = (s(d, 0, -3) - 2 * s(d, 0, -2) + s(d, 0, -1)).abs();
    let dp3 = (s(d, 3, -3) - 2 * s(d, 3, -2) + s(d, 3, -1)).abs();
    let dq0 = (s(d, 0, 2) - 2 * s(d, 0, 1) + s(d, 0, 0)).abs();
    let dq3 = (s(d, 3, 2) - 2 * s(d, 3, 1) + s(d, 3, 0)).abs();
    let dpq0 = dp0 + dq0;
    let dpq3 = dp3 + dq3;
    let dp = dp0 + dp3;
    let dq = dq0 + dq3;
    let dd = dpq0 + dpq3;
    if dd >= beta {
        return;
    }
    let dsam = |d: &[S], line: usize, dpq: i32| -> bool {
        dpq < (beta >> 2)
            && (s(d, line, -4) - s(d, line, -1)).abs() + (s(d, line, 0) - s(d, line, 3)).abs() < (beta >> 3)
            && (s(d, line, -1) - s(d, line, 0)).abs() < ((5 * tc + 1) >> 1)
    };
    let strong = dsam(d, 0, 2 * dpq0) && dsam(d, 3, 2 * dpq3);
    let dep = dp < ((beta + (beta >> 1)) >> 3);
    let deq = dq < ((beta + (beta >> 1)) >> 3);
    for line in 0..4 {
        let base = pos + line * along;
        let at = |k: isize| -> usize { (base as isize + k * step as isize) as usize };
        let p0 = d[at(-1)].to_i32();
        let p1 = d[at(-2)].to_i32();
        let p2 = d[at(-3)].to_i32();
        let p3 = d[at(-4)].to_i32();
        let q0 = d[at(0)].to_i32();
        let q1 = d[at(1)].to_i32();
        let q2 = d[at(2)].to_i32();
        let q3 = d[at(3)].to_i32();
        if strong {
            if !no_p {
                d[at(-1)] = S::from_i32(((p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) >> 3).clamp(p0 - 2 * tc, p0 + 2 * tc));
                d[at(-2)] = S::from_i32(((p2 + p1 + p0 + q0 + 2) >> 2).clamp(p1 - 2 * tc, p1 + 2 * tc));
                d[at(-3)] = S::from_i32(((2 * p3 + 3 * p2 + p1 + p0 + q0 + 4) >> 3).clamp(p2 - 2 * tc, p2 + 2 * tc));
            }
            if !no_q {
                d[at(0)] = S::from_i32(((p1 + 2 * p0 + 2 * q0 + 2 * q1 + q2 + 4) >> 3).clamp(q0 - 2 * tc, q0 + 2 * tc));
                d[at(1)] = S::from_i32(((p0 + q0 + q1 + q2 + 2) >> 2).clamp(q1 - 2 * tc, q1 + 2 * tc));
                d[at(2)] = S::from_i32(((p0 + q0 + q1 + 3 * q2 + 2 * q3 + 4) >> 3).clamp(q2 - 2 * tc, q2 + 2 * tc));
            }
        } else {
            let mut delta = (9 * (q0 - p0) - 3 * (q1 - p1) + 8) >> 4;
            if delta.abs() < tc * 10 {
                delta = delta.clamp(-tc, tc);
                if !no_p {
                    d[at(-1)] = S::from_i32((p0 + delta).clamp(0, max));
                }
                if !no_q {
                    d[at(0)] = S::from_i32((q0 - delta).clamp(0, max));
                }
                if dep && !no_p {
                    let dp = ((((p2 + p0 + 1) >> 1) - p1 + delta) >> 1).clamp(-(tc >> 1), tc >> 1);
                    d[at(-2)] = S::from_i32((p1 + dp).clamp(0, max));
                }
                if deq && !no_q {
                    let dq = ((((q2 + q0 + 1) >> 1) - q1 - delta) >> 1).clamp(-(tc >> 1), tc >> 1);
                    d[at(1)] = S::from_i32((q1 + dq).clamp(0, max));
                }
            }
        }
    }
}

/// Filter `n` lines of a chroma edge.
#[allow(clippy::too_many_arguments)]
pub(crate) fn chroma_edge_scalar<S: Sample>(d: &mut [S], pos: usize, step: usize, along: usize, n: usize, tc: i32, no_p: bool, no_q: bool, max: i32) {
    for line in 0..n {
        let base = pos + line * along;
        let at = |k: isize| -> usize { (base as isize + k * step as isize) as usize };
        let p0 = d[at(-1)].to_i32();
        let p1 = d[at(-2)].to_i32();
        let q0 = d[at(0)].to_i32();
        let q1 = d[at(1)].to_i32();
        let delta = ((((q0 - p0) << 2) + p1 - q1 + 4) >> 3).clamp(-tc, tc);
        if !no_p {
            d[at(-1)] = S::from_i32((p0 + delta).clamp(0, max));
        }
        if !no_q {
            d[at(0)] = S::from_i32((q0 - delta).clamp(0, max));
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn deblock_luma_scalar<S: Sample>(d: &mut [S], off: usize, step: usize, along: usize, beta: [i32; 2], tc: [i32; 2], no_p: [bool; 2], no_q: [bool; 2], max: i32) {
    for seg in 0..2 {
        if beta[seg] == 0 && tc[seg] == 0 {
            continue;
        }
        luma_edge_scalar(d, off + 4 * seg * along, step, along, beta[seg], tc[seg], no_p[seg], no_q[seg], max);
    }
}

#[allow(clippy::too_many_arguments)]
fn deblock_chroma_scalar<S: Sample>(d: &mut [S], off: usize, step: usize, along: usize, tc: [i32; 4], no_p: [bool; 4], no_q: [bool; 4], max: i32) {
    for seg in 0..4 {
        if tc[seg] == 0 {
            continue;
        }
        chroma_edge_scalar(d, off + 2 * seg * along, step, along, 2, tc[seg], no_p[seg], no_q[seg], max);
    }
}
