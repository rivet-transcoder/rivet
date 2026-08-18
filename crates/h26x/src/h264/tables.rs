//! H.264 constant tables. The large numeric ones (VLC codes, CABAC init
//! values, dequant constants, deblocking thresholds) are in the generated
//! `tables_gen`; the small structural ones live here.

pub use super::tables_gen::*;

/// 4x4 zig-zag scan (frame), 8.5.6: `ZIGZAG4X4[scan_pos] = raster_pos`.
pub static ZIGZAG4X4: [u8; 16] = [0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15];
/// 4x4 field scan (Table 8-13), same convention.
pub static FIELD_SCAN4X4: [u8; 16] = [0, 4, 1, 8, 12, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15];
/// 8x8 field scan (Table 8-14): `FIELD_SCAN8X8[scan_pos] = raster_pos`.
#[rustfmt::skip]
pub static FIELD_SCAN8X8: [u8; 64] = [
     0,  8, 16,  1,  9, 24, 32, 17,  2, 25, 40, 48, 56, 33, 10,  3,
    18, 41, 49, 57, 26, 11,  4, 19, 34, 42, 50, 58, 27, 12,  5, 20,
    35, 43, 51, 59, 28, 13,  6, 21, 36, 44, 52, 60, 29, 14, 22, 37,
    45, 53, 61, 30,  7, 15, 38, 46, 54, 62, 23, 31, 39, 47, 55, 63,
];

/// 4:2:2 chroma DC (8.5.11.1): the eight parsed coefficients `c0..c7`
/// placed in the 4x2 (rows x columns) array `{{c0, c2}, {c1, c5}, {c3, c6},
/// {c4, c7}}` — `SCAN_CHROMA_DC_422[i]` is the raster position
/// (`row * 2 + col`) of coefficient `i`.
pub static SCAN_CHROMA_DC_422: [u8; 8] = [0, 2, 1, 4, 6, 3, 5, 7];

/// `coeff_token` for ChromaDCLevel with ChromaArrayType 2 (Table 9-5, the
/// `nC == -2` column): `[TotalCoeff][TrailingOnes]` -> code length / bits.
pub static CHROMA422_DC_COEFF_TOKEN_LEN: [[u8; 4]; 9] = [
    [1, 0, 0, 0],
    [7, 2, 0, 0],
    [7, 7, 3, 0],
    [9, 7, 7, 5],
    [9, 9, 7, 6],
    [10, 10, 9, 7],
    [11, 11, 10, 7],
    [12, 12, 11, 10],
    [13, 12, 12, 11],
];
/// See [`CHROMA422_DC_COEFF_TOKEN_LEN`].
pub static CHROMA422_DC_COEFF_TOKEN_BITS: [[u16; 4]; 9] = [
    [1, 0, 0, 0],
    [15, 1, 0, 0],
    [14, 13, 1, 0],
    [7, 12, 11, 1],
    [6, 5, 10, 1],
    [7, 6, 4, 9],
    [7, 6, 5, 8],
    [7, 6, 5, 4],
    [7, 5, 4, 4],
];
/// `total_zeros` for 4:2:2 chroma DC (Table 9-9 b): `[tzVlcIndex - 1][total_zeros]`.
pub static CHROMA422_DC_TOTAL_ZEROS_LEN: [[u8; 8]; 7] = [
    [1, 3, 3, 4, 4, 4, 5, 5],
    [3, 2, 3, 3, 3, 3, 3, 0],
    [3, 3, 2, 2, 3, 3, 0, 0],
    [3, 2, 2, 2, 3, 0, 0, 0],
    [2, 2, 2, 2, 0, 0, 0, 0],
    [2, 2, 1, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 0, 0, 0],
];
/// See [`CHROMA422_DC_TOTAL_ZEROS_LEN`].
pub static CHROMA422_DC_TOTAL_ZEROS_BITS: [[u8; 8]; 7] = [
    [1, 2, 3, 2, 3, 1, 1, 0],
    [0, 1, 1, 4, 5, 6, 7, 0],
    [0, 1, 1, 2, 6, 7, 0, 0],
    [6, 0, 1, 2, 7, 0, 0, 0],
    [0, 1, 2, 3, 0, 0, 0, 0],
    [0, 1, 1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
];

/// 8x8 zig-zag scan (frame), 8.5.7.
#[rustfmt::skip]
pub static ZIGZAG8X8: [u8; 64] = [
     0,  1,  8, 16,  9,  2,  3, 10, 17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

/// `QPc` as a function of `qPI` (Table 8-15) for `qPI` in `0..=51`.
#[rustfmt::skip]
pub static CHROMA_QP: [u8; 52] = [
     0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 29, 30,
    31, 32, 32, 33, 34, 34, 35, 35, 36, 36, 37, 37, 37, 38, 38, 38,
    39, 39, 39, 39,
];

/// The luma 4x4 block index (0..16, in the standard's decoding order) of
/// each raster 4x4 position `(bx, by)` in a macroblock:
/// `BLK_IDX_FROM_RASTER[by * 4 + bx]`.
#[rustfmt::skip]
pub static BLK4X4_FROM_RASTER: [u8; 16] = [
     0,  1,  4,  5,
     2,  3,  6,  7,
     8,  9, 12, 13,
    10, 11, 14, 15,
];

/// The raster position `(x, y)` in 4x4-block units of luma 4x4 block
/// `blkIdx` (inverse of the above): `BLK4X4_X[blkIdx]`, `BLK4X4_Y[blkIdx]`.
pub static BLK4X4_X: [u8; 16] = [0, 1, 0, 1, 2, 3, 2, 3, 0, 1, 0, 1, 2, 3, 2, 3];
/// See [`BLK4X4_X`].
pub static BLK4X4_Y: [u8; 16] = [0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 2, 2, 3, 3];

/// Chroma DC 2x2 scan (4:2:0): raster order.
pub static CHROMA_DC_SCAN: [u8; 4] = [0, 1, 2, 3];
