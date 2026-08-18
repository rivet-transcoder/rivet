//! Scaling and inverse transforms (H.264 clause 8.5): the 4x4 and 8x8
//! integer inverse transforms, the Intra_16x16 luma DC and chroma DC
//! Hadamard transforms, and the dequantisation tables built from the
//! scaling lists.

use super::sps::ScalingLists;
use super::tables::{DEQUANT4_INIT, DEQUANT8_INIT, DEQUANT8_INIT_SCAN};

/// `LevelScale4x4[list][qP % 6][raster]` and `LevelScale8x8[list][qP % 6][raster]`
/// (8.5.9), for the six 4x4 lists (Y/Cb/Cr intra, Y/Cb/Cr inter) and the
/// six 8x8 lists.
pub struct Dequant {
    /// 4x4: `[list][qp % 6][pos]`.
    pub scale4: [[[i32; 16]; 6]; 6],
    /// 8x8: `[list][qp % 6][pos]`.
    pub scale8: [[[i32; 64]; 6]; 6],
}

impl Dequant {
    /// Build from the effective scaling lists.
    pub fn new(lists: &ScalingLists) -> Self {
        let mut scale4 = [[[0i32; 16]; 6]; 6];
        let mut scale8 = [[[0i32; 64]; 6]; 6];
        for list in 0..6 {
            for m in 0..6 {
                for pos in 0..16 {
                    let (i, j) = (pos / 4, pos % 4);
                    // normAdjust4x4(m, i, j): v_m0 for (i%2==0 && j%2==0),
                    // v_m1 for (i%2==1 && j%2==1), v_m2 otherwise. The
                    // generated table is ordered [even-even, mixed, odd-odd],
                    // i.e. by the number of odd indices.
                    let class = (i % 2) + (j % 2);
                    scale4[list][m][pos] = lists.list4x4[list][pos] as i32 * DEQUANT4_INIT[m][class] as i32;
                }
                for pos in 0..64 {
                    let (i, j) = (pos / 8, pos % 8);
                    // normAdjust8x8(m, i, j) by (i%4, j%4) class table.
                    let class = DEQUANT8_INIT_SCAN[(i % 4) * 4 + (j % 4)] as usize;
                    scale8[list][m][pos] = lists.list8x8[list][pos] as i32 * DEQUANT8_INIT[m][class] as i32;
                }
            }
        }
        Dequant { scale4, scale8 }
    }
}

/// Dequantise a 4x4 block of coefficients in place (8.5.12.1), all
/// positions (the caller zeroes position 0 or handles DC separately for
/// Intra16x16/chroma). `qp` is qP for the block (luma QP_Y or chroma QP_C).
#[inline]
pub fn dequant4x4(coeffs: &mut [i32; 16], scale: &[i32; 16], qp: i32, skip_dc: bool) {
    let q6 = qp / 6;
    let start = if skip_dc { 1 } else { 0 };
    if qp >= 24 {
        let sh = q6 - 4;
        for i in start..16 {
            coeffs[i] = (coeffs[i] * scale[i]) << sh;
        }
    } else {
        let sh = 4 - q6;
        let round = 1 << (3 - q6);
        for i in start..16 {
            coeffs[i] = (coeffs[i] * scale[i] + round) >> sh;
        }
    }
}

/// Dequantise an 8x8 block in place (8.5.13.1).
#[inline]
pub fn dequant8x8(coeffs: &mut [i32; 64], scale: &[i32; 64], qp: i32) {
    let q6 = qp / 6;
    if qp >= 36 {
        let sh = q6 - 6;
        for i in 0..64 {
            coeffs[i] = (coeffs[i] * scale[i]) << sh;
        }
    } else {
        let sh = 6 - q6;
        let round = 1 << (5 - q6);
        for i in 0..64 {
            coeffs[i] = (coeffs[i] * scale[i] + round) >> sh;
        }
    }
}

/// Inverse 4x4 transform (8.5.12.2) producing the residual `r[i][j]`
/// (already `(x + 32) >> 6`), in place.
#[inline]
pub fn idct4x4(d: &mut [i32; 16]) {
    let mut tmp = [0i32; 16];
    // Rows.
    for i in 0..4 {
        let (d0, d1, d2, d3) = (d[i * 4], d[i * 4 + 1], d[i * 4 + 2], d[i * 4 + 3]);
        let e0 = d0 + d2;
        let e1 = d0 - d2;
        let e2 = (d1 >> 1) - d3;
        let e3 = d1 + (d3 >> 1);
        tmp[i * 4] = e0 + e3;
        tmp[i * 4 + 1] = e1 + e2;
        tmp[i * 4 + 2] = e1 - e2;
        tmp[i * 4 + 3] = e0 - e3;
    }
    // Columns.
    for j in 0..4 {
        let (f0, f1, f2, f3) = (tmp[j], tmp[4 + j], tmp[8 + j], tmp[12 + j]);
        let g0 = f0 + f2;
        let g1 = f0 - f2;
        let g2 = (f1 >> 1) - f3;
        let g3 = f1 + (f3 >> 1);
        d[j] = (g0 + g3 + 32) >> 6;
        d[4 + j] = (g1 + g2 + 32) >> 6;
        d[8 + j] = (g1 - g2 + 32) >> 6;
        d[12 + j] = (g0 - g3 + 32) >> 6;
    }
}

/// Inverse 8x8 transform (8.5.13.2), in place, `(x + 32) >> 6` applied.
pub fn idct8x8(d: &mut [i32; 64]) {
    let mut tmp = [0i32; 64];
    for i in 0..8 {
        let r = &d[i * 8..i * 8 + 8];
        let mut out = [0i32; 8];
        idct8_1d(r, &mut out);
        tmp[i * 8..i * 8 + 8].copy_from_slice(&out);
    }
    for j in 0..8 {
        let col = [tmp[j], tmp[8 + j], tmp[16 + j], tmp[24 + j], tmp[32 + j], tmp[40 + j], tmp[48 + j], tmp[56 + j]];
        let mut out = [0i32; 8];
        idct8_1d(&col, &mut out);
        for i in 0..8 {
            d[i * 8 + j] = (out[i] + 32) >> 6;
        }
    }
}

#[inline(always)]
fn idct8_1d(d: &[i32], out: &mut [i32; 8]) {
    let a0 = d[0] + d[4];
    let a4 = d[0] - d[4];
    let a2 = (d[2] >> 1) - d[6];
    let a6 = d[2] + (d[6] >> 1);
    let b0 = a0 + a6;
    let b2 = a4 + a2;
    let b4 = a4 - a2;
    let b6 = a0 - a6;
    let a1 = -d[3] + d[5] - d[7] - (d[7] >> 1);
    let a3 = d[1] + d[7] - d[3] - (d[3] >> 1);
    let a5 = -d[1] + d[7] + d[5] + (d[5] >> 1);
    let a7 = d[3] + d[5] + d[1] + (d[1] >> 1);
    let b1 = a1 + (a7 >> 2);
    let b7 = a7 - (a1 >> 2);
    let b3 = a3 + (a5 >> 2);
    let b5 = (a3 >> 2) - a5;
    out[0] = b0 + b7;
    out[1] = b2 + b5;
    out[2] = b4 + b3;
    out[3] = b6 + b1;
    out[4] = b6 - b1;
    out[5] = b4 - b3;
    out[6] = b2 - b5;
    out[7] = b0 - b7;
}

/// Intra_16x16 luma DC: inverse Hadamard (8.5.10) then scaling; `dc` is
/// the 4x4 of DC coefficients in raster order, replaced by the dequantised
/// DC values `dcY[i][j]`.
pub fn luma_dc_transform(dc: &mut [i32; 16], scale00: i32, qp: i32) {
    let mut tmp = [0i32; 16];
    for i in 0..4 {
        let (c0, c1, c2, c3) = (dc[i * 4], dc[i * 4 + 1], dc[i * 4 + 2], dc[i * 4 + 3]);
        // Rows of the Hadamard matrix: (1,1,1,1), (1,1,-1,-1), (1,-1,-1,1), (1,-1,1,-1).
        let e0 = c0 + c1;
        let e1 = c0 - c1;
        let e2 = c2 - c3;
        let e3 = c2 + c3;
        tmp[i * 4] = e0 + e3;
        tmp[i * 4 + 1] = e0 - e3;
        tmp[i * 4 + 2] = e1 - e2;
        tmp[i * 4 + 3] = e1 + e2;
    }
    let mut f = [0i32; 16];
    for j in 0..4 {
        let (c0, c1, c2, c3) = (tmp[j], tmp[4 + j], tmp[8 + j], tmp[12 + j]);
        let e0 = c0 + c1;
        let e1 = c0 - c1;
        let e2 = c2 - c3;
        let e3 = c2 + c3;
        f[j] = e0 + e3;
        f[4 + j] = e0 - e3;
        f[8 + j] = e1 - e2;
        f[12 + j] = e1 + e2;
    }
    let q6 = qp / 6;
    for i in 0..16 {
        dc[i] = if qp >= 36 {
            (f[i] * scale00) << (q6 - 6)
        } else {
            (f[i] * scale00 + (1 << (5 - q6))) >> (6 - q6)
        };
    }
}

/// Chroma DC for 4:2:0 (8.5.11): 2x2 Hadamard then scaling.
pub fn chroma_dc_transform_420(dc: &mut [i32; 4], scale00: i32, qp: i32) {
    let (c0, c1, c2, c3) = (dc[0], dc[1], dc[2], dc[3]);
    let f0 = c0 + c1 + c2 + c3;
    let f1 = c0 - c1 + c2 - c3;
    let f2 = c0 + c1 - c2 - c3;
    let f3 = c0 - c1 - c2 + c3;
    let q6 = qp / 6;
    let f = [f0, f1, f2, f3];
    for i in 0..4 {
        dc[i] = ((f[i] * scale00) << q6) >> 5;
    }
}

/// Chroma DC for 4:2:2 (8.5.11.1 / 8.5.11.2): the 4x2 array `c` (raster,
/// rows of two) transformed as `f = A c B` with the 4-point and 2-point
/// Hadamard matrices, then scaled at `QP'c,DC = QP'c + 3` with `>> 6`.
/// The result stays in raster order, which is the chroma block order.
pub fn chroma_dc_transform_422(dc: &mut [i32; 8], scale00: i32, qp: i32) {
    // Rows of c: r0..r3, each (col0, col1).
    let c = *dc;
    // A rows: [1 1 1 1], [1 1 -1 -1], [1 -1 -1 1], [1 -1 1 -1] applied to
    // the column vectors (over rows), then B = [[1 1], [1 -1]] over columns.
    let mut f = [0i32; 8];
    for col in 0..2 {
        let (r0, r1, r2, r3) = (c[col], c[2 + col], c[4 + col], c[6 + col]);
        f[col] = r0 + r1 + r2 + r3;
        f[2 + col] = r0 + r1 - r2 - r3;
        f[4 + col] = r0 - r1 - r2 + r3;
        f[6 + col] = r0 - r1 + r2 - r3;
    }
    for row in 0..4 {
        let (a, b) = (f[row * 2], f[row * 2 + 1]);
        f[row * 2] = a + b;
        f[row * 2 + 1] = a - b;
    }
    let qp_dc = qp + 3;
    let q6 = qp_dc / 6;
    for i in 0..8 {
        dc[i] = ((f[i] * scale00) << q6) >> 6;
    }
}

/// Add a residual block to the prediction in place with clipping (8-bit).
#[inline(always)]
pub fn add_residual(dst: &mut [u8], stride: usize, res: &[i32], size: usize) {
    for y in 0..size {
        let row = &mut dst[y * stride..y * stride + size];
        for x in 0..size {
            let v = row[x] as i32 + res[y * size + x];
            row[x] = v.clamp(0, 255) as u8;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn idct4_of_dc_only_is_flat() {
        let mut d = [0i32; 16];
        d[0] = 64;
        idct4x4(&mut d);
        assert!(d.iter().all(|&v| v == 1));
    }

    #[test]
    fn idct8_of_dc_only_is_flat() {
        let mut d = [0i32; 64];
        d[0] = 64;
        idct8x8(&mut d);
        assert!(d.iter().all(|&v| v == 1));
    }

    #[test]
    fn dequant_matches_the_two_formulas() {
        let flat = ScalingLists::flat();
        let dq = Dequant::new(&flat);
        // qP 28: LevelScale = 16 * 10 = 160 for class 0 at qp%6 = 4 -> v = 16 (25? no: DEQUANT4_INIT[4][0] = 16)
        let mut c = [0i32; 16];
        c[0] = 3;
        dequant4x4(&mut c, &dq.scale4[0][28 % 6], 28, false);
        // 3 * (16*16) << (4-4) = 768
        assert_eq!(c[0], 3 * 16 * DEQUANT4_INIT[4][0] as i32);
        let mut c = [0i32; 16];
        c[0] = 3;
        dequant4x4(&mut c, &dq.scale4[0][10 % 6], 10, false);
        // qP 10: q6 = 1, (3*16*v + 4) >> 3
        assert_eq!(c[0], (3 * 16 * DEQUANT4_INIT[4][0] as i32 + 4) >> 3);
    }
}
