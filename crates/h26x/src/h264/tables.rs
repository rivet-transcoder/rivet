//! H.264 constant tables. The large numeric ones (VLC codes, CABAC init
//! values, dequant constants, deblocking thresholds) are in the generated
//! `tables_gen`; the small structural ones live here.

pub use super::tables_gen::*;

/// 4x4 zig-zag scan (frame), 8.5.6: `ZIGZAG4X4[scan_pos] = raster_pos`.
pub static ZIGZAG4X4: [u8; 16] = [0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15];

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
