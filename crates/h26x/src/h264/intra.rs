//! Intra prediction (H.264 clause 8.3): Intra_4x4, Intra_8x8 (with the
//! reference sample filter), Intra_16x16 and chroma. Each predictor
//! writes its prediction straight into the plane at the block position;
//! the residual is added afterwards.

use super::frame::PaddedPlane;
use crate::sample::Sample;
use crate::{Error, Result};

/// Which neighbouring samples of a block are available for prediction.
#[derive(Debug, Clone, Copy, Default)]
pub struct IntraAvail {
    /// `p[x, -1]` for the block's own width.
    pub top: bool,
    /// `p[-1, y]`.
    pub left: bool,
    /// `p[-1, -1]`.
    pub top_left: bool,
    /// `p[x, -1]` for x beyond the block's width (top-right).
    pub top_right: bool,
}

/// Intra_4x4 prediction (8.3.1.2) of the block at plane offset `off`
/// (top-left sample), mode 0..=8.
pub fn predict_4x4<S: Sample>(
    p: &mut PaddedPlane<S>,
    off: usize,
    stride: usize,
    mode: u8,
    av: IntraAvail,
    bit_depth: u32,
) -> Result<()> {
    // Gather neighbours: top[0..8] = p[x,-1] x=0..7, left[0..4], corner.
    let mut top = [0i32; 8];
    let mut left = [0i32; 4];
    let corner: i32;
    if av.top {
        for x in 0..4 {
            top[x] = p.data[off - stride + x].to_i32();
        }
        if av.top_right {
            for x in 4..8 {
                top[x] = p.data[off - stride + x].to_i32();
            }
        } else {
            for x in 4..8 {
                top[x] = top[3];
            }
        }
    }
    if av.left {
        for y in 0..4 {
            left[y] = p.data[off + y * stride - 1].to_i32();
        }
    }
    corner = if av.top_left {
        p.data[off - stride - 1].to_i32()
    } else {
        0
    };
    let mut pred = [S::default(); 16];
    match mode {
        0 => {
            if !av.top {
                return Err(Error::bitstream("Intra_4x4 vertical without top samples"));
            }
            for y in 0..4 {
                for x in 0..4 {
                    pred[y * 4 + x] = S::from_i32(top[x]);
                }
            }
        }
        1 => {
            if !av.left {
                return Err(Error::bitstream(
                    "Intra_4x4 horizontal without left samples",
                ));
            }
            for y in 0..4 {
                for x in 0..4 {
                    pred[y * 4 + x] = S::from_i32(left[y]);
                }
            }
        }
        2 => {
            let v = match (av.top, av.left) {
                (true, true) => {
                    (top[0] + top[1] + top[2] + top[3] + left[0] + left[1] + left[2] + left[3] + 4)
                        >> 3
                }
                (true, false) => (top[0] + top[1] + top[2] + top[3] + 2) >> 2,
                (false, true) => (left[0] + left[1] + left[2] + left[3] + 2) >> 2,
                (false, false) => 1 << (bit_depth - 1),
            };
            pred.fill(S::from_i32(v));
        }
        3 => {
            // Diagonal down left.
            if !av.top {
                return Err(Error::bitstream(
                    "Intra_4x4 diagonal-down-left without top samples",
                ));
            }
            for y in 0..4 {
                for x in 0..4 {
                    let v = if x == 3 && y == 3 {
                        (top[6] + 3 * top[7] + 2) >> 2
                    } else {
                        (top[x + y] + 2 * top[x + y + 1] + top[x + y + 2] + 2) >> 2
                    };
                    pred[y * 4 + x] = S::from_i32(v);
                }
            }
        }
        4 => {
            // Diagonal down right.
            if !(av.top && av.left && av.top_left) {
                return Err(Error::bitstream(
                    "Intra_4x4 diagonal-down-right without neighbours",
                ));
            }
            // p[-1..3, -1] and p[-1, -1..3]: build the edge e[i] for i in -4..=3
            // as e(x) with x = column index -1 -> corner.
            let at = |x: i32, y: i32| -> i32 {
                if y == -1 {
                    if x == -1 { corner } else { top[x as usize] }
                } else {
                    left[y as usize]
                }
            };
            for y in 0..4i32 {
                for x in 0..4i32 {
                    let v = if x > y {
                        (at(x - y - 2, -1) + 2 * at(x - y - 1, -1) + at(x - y, -1) + 2) >> 2
                    } else if x < y {
                        (at(-1, y - x - 2) + 2 * at(-1, y - x - 1) + at(-1, y - x) + 2) >> 2
                    } else {
                        (at(0, -1) + 2 * at(-1, -1) + at(-1, 0) + 2) >> 2
                    };
                    pred[(y * 4 + x) as usize] = S::from_i32(v);
                }
            }
        }
        5 => {
            // Vertical right.
            if !(av.top && av.left && av.top_left) {
                return Err(Error::bitstream(
                    "Intra_4x4 vertical-right without neighbours",
                ));
            }
            let at = |x: i32, y: i32| -> i32 {
                if y == -1 {
                    if x == -1 { corner } else { top[x as usize] }
                } else {
                    left[y as usize]
                }
            };
            for y in 0..4i32 {
                for x in 0..4i32 {
                    let z = 2 * x - y;
                    let v = if z >= 0 && z % 2 == 0 {
                        (at(x - (y >> 1) - 1, -1) + at(x - (y >> 1), -1) + 1) >> 1
                    } else if z >= 0 {
                        (at(x - (y >> 1) - 2, -1)
                            + 2 * at(x - (y >> 1) - 1, -1)
                            + at(x - (y >> 1), -1)
                            + 2)
                            >> 2
                    } else if z == -1 {
                        (at(-1, 0) + 2 * at(-1, -1) + at(0, -1) + 2) >> 2
                    } else {
                        (at(-1, y - 1) + 2 * at(-1, y - 2) + at(-1, y - 3) + 2) >> 2
                    };
                    pred[(y * 4 + x) as usize] = S::from_i32(v);
                }
            }
        }
        6 => {
            // Horizontal down.
            if !(av.top && av.left && av.top_left) {
                return Err(Error::bitstream(
                    "Intra_4x4 horizontal-down without neighbours",
                ));
            }
            let at = |x: i32, y: i32| -> i32 {
                if y == -1 {
                    if x == -1 { corner } else { top[x as usize] }
                } else {
                    left[y as usize]
                }
            };
            for y in 0..4i32 {
                for x in 0..4i32 {
                    let z = 2 * y - x;
                    let v = if z >= 0 && z % 2 == 0 {
                        (at(-1, y - (x >> 1) - 1) + at(-1, y - (x >> 1)) + 1) >> 1
                    } else if z >= 0 {
                        (at(-1, y - (x >> 1) - 2)
                            + 2 * at(-1, y - (x >> 1) - 1)
                            + at(-1, y - (x >> 1))
                            + 2)
                            >> 2
                    } else if z == -1 {
                        (at(-1, 0) + 2 * at(-1, -1) + at(0, -1) + 2) >> 2
                    } else {
                        (at(x - 1, -1) + 2 * at(x - 2, -1) + at(x - 3, -1) + 2) >> 2
                    };
                    pred[(y * 4 + x) as usize] = S::from_i32(v);
                }
            }
        }
        7 => {
            // Vertical left.
            if !av.top {
                return Err(Error::bitstream(
                    "Intra_4x4 vertical-left without top samples",
                ));
            }
            for y in 0..4usize {
                for x in 0..4usize {
                    let v = if y % 2 == 0 {
                        (top[x + (y >> 1)] + top[x + (y >> 1) + 1] + 1) >> 1
                    } else {
                        (top[x + (y >> 1)] + 2 * top[x + (y >> 1) + 1] + top[x + (y >> 1) + 2] + 2)
                            >> 2
                    };
                    pred[y * 4 + x] = S::from_i32(v);
                }
            }
        }
        8 => {
            // Horizontal up.
            if !av.left {
                return Err(Error::bitstream(
                    "Intra_4x4 horizontal-up without left samples",
                ));
            }
            for y in 0..4i32 {
                for x in 0..4i32 {
                    let z = x + 2 * y;
                    let v = if z > 5 {
                        left[3]
                    } else if z == 5 {
                        (left[2] + 3 * left[3] + 2) >> 2
                    } else if z % 2 == 0 {
                        (left[(y + (x >> 1)) as usize] + left[(y + (x >> 1) + 1) as usize] + 1) >> 1
                    } else {
                        (left[(y + (x >> 1)) as usize]
                            + 2 * left[(y + (x >> 1) + 1) as usize]
                            + left[(y + (x >> 1) + 2) as usize]
                            + 2)
                            >> 2
                    };
                    pred[(y * 4 + x) as usize] = S::from_i32(v);
                }
            }
        }
        _ => return Err(Error::bitstream("Intra_4x4 prediction mode out of range")),
    }
    for y in 0..4 {
        p.data[off + y * stride..off + y * stride + 4].copy_from_slice(&pred[y * 4..y * 4 + 4]);
    }
    Ok(())
}

/// Intra_8x8 prediction (8.3.2.2) with reference sample filtering.
pub fn predict_8x8<S: Sample>(
    p: &mut PaddedPlane<S>,
    off: usize,
    stride: usize,
    mode: u8,
    av: IntraAvail,
    bit_depth: u32,
) -> Result<()> {
    // Unfiltered neighbours: top[0..16], left[0..8], corner.
    let mut top = [0i32; 16];
    let mut left = [0i32; 8];
    let mut corner = 0i32;
    if av.top {
        for x in 0..8 {
            top[x] = p.data[off - stride + x].to_i32();
        }
        if av.top_right {
            for x in 8..16 {
                top[x] = p.data[off - stride + x].to_i32();
            }
        } else {
            for x in 8..16 {
                top[x] = top[7];
            }
        }
    }
    if av.left {
        for y in 0..8 {
            left[y] = p.data[off + y * stride - 1].to_i32();
        }
    }
    if av.top_left {
        corner = p.data[off - stride - 1].to_i32();
    }
    // Filtering (8.3.2.2.1).
    let mut ftop = [0i32; 16];
    let mut fleft = [0i32; 8];
    let mut fcorner = 0i32;
    if av.top {
        // p'[0,-1]
        ftop[0] = if av.top_left {
            (corner + 2 * top[0] + top[1] + 2) >> 2
        } else {
            (3 * top[0] + top[1] + 2) >> 2
        };
        for x in 1..15 {
            ftop[x] = (top[x - 1] + 2 * top[x] + top[x + 1] + 2) >> 2;
        }
        ftop[15] = (top[14] + 3 * top[15] + 2) >> 2;
    }
    if av.top_left {
        fcorner = match (av.top, av.left) {
            (true, true) => (top[0] + 2 * corner + left[0] + 2) >> 2,
            (true, false) => (3 * corner + top[0] + 2) >> 2,
            (false, true) => (3 * corner + left[0] + 2) >> 2,
            (false, false) => corner,
        };
    }
    if av.left {
        fleft[0] = if av.top_left {
            (corner + 2 * left[0] + left[1] + 2) >> 2
        } else {
            (3 * left[0] + left[1] + 2) >> 2
        };
        for y in 1..7 {
            fleft[y] = (left[y - 1] + 2 * left[y] + left[y + 1] + 2) >> 2;
        }
        fleft[7] = (left[6] + 3 * left[7] + 2) >> 2;
    }
    let (top, left, corner) = (ftop, fleft, fcorner);
    let at = |x: i32, y: i32| -> i32 {
        if y == -1 {
            if x == -1 { corner } else { top[x as usize] }
        } else {
            left[y as usize]
        }
    };
    let mut pred = [S::default(); 64];
    match mode {
        0 => {
            if !av.top {
                return Err(Error::bitstream("Intra_8x8 vertical without top samples"));
            }
            for y in 0..8 {
                for x in 0..8 {
                    pred[y * 8 + x] = S::from_i32(top[x]);
                }
            }
        }
        1 => {
            if !av.left {
                return Err(Error::bitstream(
                    "Intra_8x8 horizontal without left samples",
                ));
            }
            for y in 0..8 {
                for x in 0..8 {
                    pred[y * 8 + x] = S::from_i32(left[y]);
                }
            }
        }
        2 => {
            let st: i32 = top[..8].iter().sum();
            let sl: i32 = left.iter().sum();
            let v = match (av.top, av.left) {
                (true, true) => (st + sl + 8) >> 4,
                (true, false) => (st + 4) >> 3,
                (false, true) => (sl + 4) >> 3,
                (false, false) => 1 << (bit_depth - 1),
            };
            pred.fill(S::from_i32(v));
        }
        3 => {
            if !av.top {
                return Err(Error::bitstream(
                    "Intra_8x8 diagonal-down-left without top samples",
                ));
            }
            for y in 0..8usize {
                for x in 0..8usize {
                    let v = if x == 7 && y == 7 {
                        (top[14] + 3 * top[15] + 2) >> 2
                    } else {
                        (top[x + y] + 2 * top[x + y + 1] + top[x + y + 2] + 2) >> 2
                    };
                    pred[y * 8 + x] = S::from_i32(v);
                }
            }
        }
        4 => {
            if !(av.top && av.left && av.top_left) {
                return Err(Error::bitstream(
                    "Intra_8x8 diagonal-down-right without neighbours",
                ));
            }
            for y in 0..8i32 {
                for x in 0..8i32 {
                    let v = if x > y {
                        (at(x - y - 2, -1) + 2 * at(x - y - 1, -1) + at(x - y, -1) + 2) >> 2
                    } else if x < y {
                        (at(-1, y - x - 2) + 2 * at(-1, y - x - 1) + at(-1, y - x) + 2) >> 2
                    } else {
                        (at(0, -1) + 2 * at(-1, -1) + at(-1, 0) + 2) >> 2
                    };
                    pred[(y * 8 + x) as usize] = S::from_i32(v);
                }
            }
        }
        5 => {
            if !(av.top && av.left && av.top_left) {
                return Err(Error::bitstream(
                    "Intra_8x8 vertical-right without neighbours",
                ));
            }
            for y in 0..8i32 {
                for x in 0..8i32 {
                    let z = 2 * x - y;
                    let v = if z >= 0 && z % 2 == 0 {
                        (at(x - (y >> 1) - 1, -1) + at(x - (y >> 1), -1) + 1) >> 1
                    } else if z >= 0 {
                        (at(x - (y >> 1) - 2, -1)
                            + 2 * at(x - (y >> 1) - 1, -1)
                            + at(x - (y >> 1), -1)
                            + 2)
                            >> 2
                    } else if z == -1 {
                        (at(-1, 0) + 2 * at(-1, -1) + at(0, -1) + 2) >> 2
                    } else {
                        (at(-1, y - 2 * x - 1)
                            + 2 * at(-1, y - 2 * x - 2)
                            + at(-1, y - 2 * x - 3)
                            + 2)
                            >> 2
                    };
                    pred[(y * 8 + x) as usize] = S::from_i32(v);
                }
            }
        }
        6 => {
            if !(av.top && av.left && av.top_left) {
                return Err(Error::bitstream(
                    "Intra_8x8 horizontal-down without neighbours",
                ));
            }
            for y in 0..8i32 {
                for x in 0..8i32 {
                    let z = 2 * y - x;
                    let v = if z >= 0 && z % 2 == 0 {
                        (at(-1, y - (x >> 1) - 1) + at(-1, y - (x >> 1)) + 1) >> 1
                    } else if z >= 0 {
                        (at(-1, y - (x >> 1) - 2)
                            + 2 * at(-1, y - (x >> 1) - 1)
                            + at(-1, y - (x >> 1))
                            + 2)
                            >> 2
                    } else if z == -1 {
                        (at(-1, 0) + 2 * at(-1, -1) + at(0, -1) + 2) >> 2
                    } else {
                        (at(x - 2 * y - 1, -1)
                            + 2 * at(x - 2 * y - 2, -1)
                            + at(x - 2 * y - 3, -1)
                            + 2)
                            >> 2
                    };
                    pred[(y * 8 + x) as usize] = S::from_i32(v);
                }
            }
        }
        7 => {
            if !av.top {
                return Err(Error::bitstream(
                    "Intra_8x8 vertical-left without top samples",
                ));
            }
            for y in 0..8usize {
                for x in 0..8usize {
                    let v = if y % 2 == 0 {
                        (top[x + (y >> 1)] + top[x + (y >> 1) + 1] + 1) >> 1
                    } else {
                        (top[x + (y >> 1)] + 2 * top[x + (y >> 1) + 1] + top[x + (y >> 1) + 2] + 2)
                            >> 2
                    };
                    pred[y * 8 + x] = S::from_i32(v);
                }
            }
        }
        8 => {
            if !av.left {
                return Err(Error::bitstream(
                    "Intra_8x8 horizontal-up without left samples",
                ));
            }
            for y in 0..8i32 {
                for x in 0..8i32 {
                    let z = x + 2 * y;
                    let v = if z > 13 {
                        left[7]
                    } else if z == 13 {
                        (left[6] + 3 * left[7] + 2) >> 2
                    } else if z % 2 == 0 {
                        (left[(y + (x >> 1)) as usize] + left[(y + (x >> 1) + 1) as usize] + 1) >> 1
                    } else {
                        (left[(y + (x >> 1)) as usize]
                            + 2 * left[(y + (x >> 1) + 1) as usize]
                            + left[(y + (x >> 1) + 2) as usize]
                            + 2)
                            >> 2
                    };
                    pred[(y * 8 + x) as usize] = S::from_i32(v);
                }
            }
        }
        _ => return Err(Error::bitstream("Intra_8x8 prediction mode out of range")),
    }
    for y in 0..8 {
        p.data[off + y * stride..off + y * stride + 8].copy_from_slice(&pred[y * 8..y * 8 + 8]);
    }
    Ok(())
}

/// Intra_16x16 prediction (8.3.3): 0 vertical, 1 horizontal, 2 DC, 3 plane.
pub fn predict_16x16<S: Sample>(
    p: &mut PaddedPlane<S>,
    off: usize,
    stride: usize,
    mode: u8,
    av: IntraAvail,
    bit_depth: u32,
) -> Result<()> {
    predict_planar_block(
        p,
        off,
        stride,
        16,
        mode,
        av,
        5,
        5,
        16,
        "Intra_16x16",
        bit_depth,
    )
}

/// Chroma prediction (8.3.4) for the 8-wide, `h`-tall (8 for 4:2:0, 16 for
/// 4:2:2) chroma block: 0 DC, 1 horizontal, 2 vertical, 3 plane — note the
/// different mode numbering from luma.
pub fn predict_chroma<S: Sample>(
    p: &mut PaddedPlane<S>,
    off: usize,
    stride: usize,
    mode: u8,
    av: IntraAvail,
    left_rows: [bool; 4],
    bit_depth: u32,
    h: usize,
) -> Result<()> {
    let w = 8usize;
    let max = (1i32 << bit_depth) - 1;
    match mode {
        0 => {
            // DC per 4x4 chroma block (8.3.4.1..3): the top-left block and the
            // blocks with both offsets nonzero average both neighbours; the
            // rest of the top row prefers the top samples, the rest of the
            // left column the left ones. The left samples of each block row
            // have their own availability (`left_rows`: in an MBAFF frame
            // the left column may span two macroblocks).
            let mut top = [0i32; 8];
            let mut left = [0i32; 16];
            if av.top {
                for x in 0..w {
                    top[x] = p.data[off - stride + x].to_i32();
                }
            }
            for y in 0..h {
                if left_rows[y / 4] {
                    left[y] = p.data[off + y * stride - 1].to_i32();
                }
            }
            for by in 0..h / 4 {
                let left_ok = left_rows[by];
                for bx in 0..w / 4 {
                    let st: i32 = top[bx * 4..bx * 4 + 4].iter().sum();
                    let sl: i32 = left[by * 4..by * 4 + 4].iter().sum();
                    let v = if (bx == 0 && by == 0) || (bx > 0 && by > 0) {
                        match (av.top, left_ok) {
                            (true, true) => (st + sl + 4) >> 3,
                            (true, false) => (st + 2) >> 2,
                            (false, true) => (sl + 2) >> 2,
                            (false, false) => 1 << (bit_depth - 1),
                        }
                    } else if bx > 0 {
                        // Top row, right of the first block: prefers top.
                        match (av.top, left_ok) {
                            (true, _) => (st + 2) >> 2,
                            (false, true) => (sl + 2) >> 2,
                            (false, false) => 1 << (bit_depth - 1),
                        }
                    } else {
                        // Left column below the first block: prefers left.
                        match (left_ok, av.top) {
                            (true, _) => (sl + 2) >> 2,
                            (false, true) => (st + 2) >> 2,
                            (false, false) => 1 << (bit_depth - 1),
                        }
                    };
                    for y in 0..4 {
                        let row = off + (by * 4 + y) * stride + bx * 4;
                        p.data[row..row + 4].fill(S::from_i32(v));
                    }
                }
            }
            Ok(())
        }
        1 => {
            if !av.left {
                return Err(Error::bitstream(
                    "chroma horizontal prediction without left samples",
                ));
            }
            for y in 0..h {
                let v = p.data[off + y * stride - 1];
                p.data[off + y * stride..off + y * stride + w].fill(v);
            }
            Ok(())
        }
        2 => {
            if !av.top {
                return Err(Error::bitstream(
                    "chroma vertical prediction without top samples",
                ));
            }
            let mut top = [S::default(); 8];
            top.copy_from_slice(&p.data[off - stride..off - stride + w]);
            for y in 0..h {
                p.data[off + y * stride..off + y * stride + w].copy_from_slice(&top);
            }
            Ok(())
        }
        3 => {
            if !(av.top && av.left && av.top_left) {
                return Err(Error::bitstream(
                    "chroma plane prediction without neighbours",
                ));
            }
            // 8.3.4.4 with xCF = 0 and yCF = 4 for 4:2:2.
            let ycf = (h / 8 - 1) as i32 * 4; // 0 or 4
            let at_top = |x: i32| -> i32 {
                if x < 0 {
                    p.data[off - stride - 1].to_i32()
                } else {
                    p.data[off - stride + x as usize].to_i32()
                }
            };
            let at_left = |y: i32| -> i32 {
                if y < 0 {
                    p.data[off - stride - 1].to_i32()
                } else {
                    p.data[off + y as usize * stride - 1].to_i32()
                }
            };
            let mut hh = 0i32;
            let mut vv = 0i32;
            for i in 0..4 {
                hh += (i + 1) * (at_top(4 + i) - at_top(2 - i));
            }
            for i in 0..4 + ycf {
                vv += (i + 1) * (at_left(4 + ycf + i) - at_left(2 + ycf - i));
            }
            let b = (34 * hh + 32) >> 6;
            let cmul = if h == 16 { 5 } else { 34 };
            let c = (cmul * vv + 32) >> 6;
            let a = 16 * (at_left(h as i32 - 1) + at_top(w as i32 - 1));
            for y in 0..h as i32 {
                for x in 0..w as i32 {
                    let val = (a + b * (x - 3) + c * (y - 3 - ycf) + 16) >> 5;
                    p.data[off + y as usize * stride + x as usize] = S::from_i32(val.clamp(0, max));
                }
            }
            Ok(())
        }
        _ => Err(Error::bitstream("intra_chroma_pred_mode out of range")),
    }
}

/// Vertical / horizontal / DC / plane for a square block of `size` 8 or 16
/// (the luma 16x16 and chroma 8x8 predictors share this shape). Modes use
/// the *luma* numbering (0 V, 1 H, 2 DC, 3 plane); chroma passes 3 only.
#[allow(clippy::too_many_arguments)]
fn predict_planar_block<S: Sample>(
    p: &mut PaddedPlane<S>,
    off: usize,
    stride: usize,
    size: usize,
    mode: u8,
    av: IntraAvail,
    bmul: i32,
    cmul: i32,
    _n: usize,
    what: &str,
    bit_depth: u32,
) -> Result<()> {
    let max = (1i32 << bit_depth) - 1;
    match mode {
        0 => {
            if !av.top {
                return Err(Error::bitstream(format!(
                    "{what} vertical prediction without top samples"
                )));
            }
            let mut top = [S::default(); 16];
            top[..size].copy_from_slice(&p.data[off - stride..off - stride + size]);
            for y in 0..size {
                p.data[off + y * stride..off + y * stride + size].copy_from_slice(&top[..size]);
            }
        }
        1 => {
            if !av.left {
                return Err(Error::bitstream(format!(
                    "{what} horizontal prediction without left samples"
                )));
            }
            for y in 0..size {
                let v = p.data[off + y * stride - 1];
                p.data[off + y * stride..off + y * stride + size].fill(v);
            }
        }
        2 => {
            let mut st = 0i32;
            let mut sl = 0i32;
            if av.top {
                for x in 0..size {
                    st += p.data[off - stride + x].to_i32();
                }
            }
            if av.left {
                for y in 0..size {
                    sl += p.data[off + y * stride - 1].to_i32();
                }
            }
            let shift = if size == 16 { 4 } else { 3 };
            let v = match (av.top, av.left) {
                (true, true) => (st + sl + size as i32) >> (shift + 1),
                (true, false) => (st + (size as i32 >> 1)) >> shift,
                (false, true) => (sl + (size as i32 >> 1)) >> shift,
                (false, false) => 1 << (bit_depth - 1),
            };
            for y in 0..size {
                p.data[off + y * stride..off + y * stride + size].fill(S::from_i32(v));
            }
        }
        3 => {
            if !(av.top && av.left && av.top_left) {
                return Err(Error::bitstream(format!(
                    "{what} plane prediction without neighbours"
                )));
            }
            let half = (size / 2) as i32;
            let mut h = 0i32;
            let mut v = 0i32;
            let at_top = |x: i32| -> i32 {
                if x < 0 {
                    p.data[off - stride - 1].to_i32()
                } else {
                    p.data[off - stride + x as usize].to_i32()
                }
            };
            let at_left = |y: i32| -> i32 {
                if y < 0 {
                    p.data[off - stride - 1].to_i32()
                } else {
                    p.data[off + y as usize * stride - 1].to_i32()
                }
            };
            for i in 0..half {
                h += (i + 1) * (at_top(half + i) - at_top(half - 2 - i));
                v += (i + 1) * (at_left(half + i) - at_left(half - 2 - i));
            }
            let (b, c) = if size == 16 {
                ((5 * h + 32) >> 6, (5 * v + 32) >> 6)
            } else {
                ((34 * h + 32) >> 6, (34 * v + 32) >> 6)
            };
            let _ = (bmul, cmul);
            let a = 16 * (at_left(size as i32 - 1) + at_top(size as i32 - 1));
            for y in 0..size as i32 {
                for x in 0..size as i32 {
                    let val = (a + b * (x - half + 1) + c * (y - half + 1) + 16) >> 5;
                    p.data[off + y as usize * stride + x as usize] = S::from_i32(val.clamp(0, max));
                }
            }
        }
        _ => {
            return Err(Error::bitstream(format!(
                "{what} prediction mode out of range"
            )));
        }
    }
    Ok(())
}
