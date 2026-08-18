//! CABAC-coded macroblock layer (H.264 clause 9.3): context selection
//! (9.3.3.1), binarisations (9.3.2) and the residual block (9.3.3.1.3).

use crate::cabac::{Cabac, Ctx, init_ctx_h264};
use crate::{Error, Result};

use super::cavlc::{
    b_sub_mb_type, mb_partitions, p_sub_mb_type, part_index_of, predicted_intra_mode, sub_partition_rect,
};
use super::frame::Mv;
use super::mb::{
    MbKind, MbLayer, MbNeighbours, PicInfo, SliceCtx, SubMbShape, PRED_BI, PRED_L0, PRED_L1,
};
use super::slice::SliceType;
use super::tables::*;

/// Number of context variables (ctxIdx 0..=1023).
pub const NUM_CTX: usize = 1024;

/// Initialise all 1024 contexts for a slice (9.3.1.1).
pub fn init_contexts(ctxs: &mut [Ctx; NUM_CTX], slice_type: SliceType, cabac_init_idc: u32, qp: i32) {
    let table: &[[i8; 2]; 1024] =
        if slice_type.is_intra() { &CABAC_INIT_I } else { &CABAC_INIT_PB[cabac_init_idc as usize] };
    for (i, c) in ctxs.iter_mut().enumerate() {
        let (m, n) = (table[i][0] as i32, table[i][1] as i32);
        *c = init_ctx_h264(m, n, qp);
    }
}

// ctxIdx offsets (Table 9-34).
const CTX_MB_TYPE_SI_PREFIX: usize = 0;
const CTX_MB_TYPE_I: usize = 3;
const CTX_MB_SKIP_P: usize = 11;
const CTX_MB_TYPE_P_PREFIX: usize = 14;
const CTX_MB_TYPE_P_SUFFIX: usize = 17;
const CTX_SUB_MB_TYPE_P: usize = 21;
const CTX_MB_SKIP_B: usize = 24;
const CTX_MB_TYPE_B_PREFIX: usize = 27;
const CTX_MB_TYPE_B_SUFFIX: usize = 32;
const CTX_SUB_MB_TYPE_B: usize = 36;
const CTX_MVD_X: usize = 40;
const CTX_MVD_Y: usize = 47;
const CTX_REF_IDX: usize = 54;
const CTX_MB_QP_DELTA: usize = 60;
const CTX_INTRA_CHROMA_PRED_MODE: usize = 64;
const CTX_PREV_INTRA_PRED_MODE_FLAG: usize = 68;
const CTX_REM_INTRA_PRED_MODE: usize = 69;
const CTX_CBP_LUMA: usize = 73;
const CTX_CBP_CHROMA: usize = 77;
const CTX_CODED_BLOCK_FLAG: usize = 85;
const CTX_SIG_COEFF: usize = 105;
const CTX_LAST_COEFF: usize = 166;
const CTX_COEFF_ABS: usize = 227;
const CTX_TRANSFORM_8X8: usize = 399;
const CTX_SIG_COEFF_8X8: usize = 402;
const CTX_LAST_COEFF_8X8: usize = 417;
const CTX_COEFF_ABS_8X8: usize = 426;

/// Per-`ctxBlockCat` context bases (Table 9-34's ctxIdxOffset plus Table
/// 9-40's ctxIdxBlockCatOffset) for the fourteen block categories: luma
/// DC / AC / 4x4 / (chroma DC / AC) / luma 8x8, then the same five luma-style
/// categories for Cb (6..=9) and Cr (10..=13) in 4:4:4.
const CBF_CTX_BASE: [usize; 14] = [85, 89, 93, 97, 101, 1012, 460, 464, 468, 1016, 472, 476, 480, 1020];
const SIG_CTX_BASE: [usize; 14] = [105, 120, 134, 149, 152, 402, 484, 499, 513, 660, 528, 543, 557, 718];
const LAST_CTX_BASE: [usize; 14] = [166, 181, 195, 210, 213, 417, 572, 587, 601, 690, 616, 631, 645, 748];
const ABS_CTX_BASE: [usize; 14] = [227, 237, 247, 257, 266, 426, 952, 962, 972, 708, 982, 992, 1002, 766];

/// `ctxBlockCat` values.
const CAT_LUMA_DC: usize = 0;
const CAT_LUMA_AC: usize = 1;
const CAT_LUMA_4X4: usize = 2;
const CAT_CHROMA_DC: usize = 3;
const CAT_CHROMA_AC: usize = 4;
const CAT_LUMA_8X8: usize = 5;
/// The luma-style categories `[DC, AC, 4x4, 8x8]` of colour plane `p`
/// (luma, and Cb / Cr in 4:4:4).
const PLANE_CATS: [[usize; 4]; 3] = [[0, 1, 2, 5], [6, 7, 8, 9], [10, 11, 12, 13]];

/// The colour plane a luma-style category belongs to (chroma DC / AC: 0).
#[inline]
fn cat_plane(cat: usize) -> usize {
    match cat {
        6..=9 => 1,
        10..=13 => 2,
        _ => 0,
    }
}

/// The CABAC slice decoding state carried across macroblocks.
pub struct CabacState {
    /// The 1024 context variables.
    pub ctx: [Ctx; NUM_CTX],
    /// `mb_qp_delta != 0` for the previous macroblock in decoding order
    /// (false when that macroblock was skipped, PCM, or had no residual).
    pub prev_qp_delta_nonzero: bool,
}

impl CabacState {
    /// Fresh state for a slice.
    pub fn new(slice_type: SliceType, cabac_init_idc: u32, qp: i32) -> Self {
        let mut ctx = [0u8; NUM_CTX];
        init_contexts(&mut ctx, slice_type, cabac_init_idc, qp);
        Self { ctx, prev_qp_delta_nonzero: false }
    }
}

#[inline]
fn bin(c: &mut Cabac, st: &mut CabacState, ctx: usize) -> u32 {
    c.decision(&mut st.ctx[ctx])
}

/// `mb_skip_flag` (9.3.3.1.1.1).
pub fn decode_mb_skip(c: &mut Cabac, st: &mut CabacState, info: &PicInfo, nb: &MbNeighbours, is_b: bool) -> bool {
    let cond = |a: Option<usize>| -> usize {
        match a {
            Some(addr) if !info.mbs[addr].kind.is_skip() => 1,
            _ => 0,
        }
    };
    let inc = cond(nb.a) + cond(nb.b);
    let base = if is_b { CTX_MB_SKIP_B } else { CTX_MB_SKIP_P };
    bin(c, st, base + inc) != 0
}

/// `end_of_slice_flag`.
pub fn decode_end_of_slice(c: &mut Cabac) -> bool {
    c.terminate() != 0
}

/// Intra `mb_type` binarisation shared by I slices (`base` = 3, with the
/// neighbour-dependent first bin) and the P/B suffixes (`base` = 17 / 32).
fn decode_intra_mb_type(
    c: &mut Cabac,
    st: &mut CabacState,
    base: usize,
    intra_slice: bool,
    info: &PicInfo,
    nb: &MbNeighbours,
) -> u32 {
    if intra_slice {
        let cond = |a: Option<usize>| -> usize {
            match a {
                Some(addr) => match info.mbs[addr].kind {
                    // "mb_type != I_NxN" for available neighbours (SI is not supported).
                    MbKind::I4x4 | MbKind::I8x8 => 0,
                    _ => 1,
                },
                None => 0,
            }
        };
        let inc = cond(nb.a) + cond(nb.b);
        if bin(c, st, base + inc) == 0 {
            return 0; // I_NxN
        }
    } else if bin(c, st, base) == 0 {
        return 0;
    }
    if c.terminate() != 0 {
        return 25; // I_PCM
    }
    // I_16x16: bins for cbp luma, cbp chroma, prediction mode.
    let (b_luma, b_chroma_nz, b_chroma_two, b_pred0, b_pred1) = if intra_slice {
        (base + 3, base + 4, base + 5, base + 6, base + 7)
    } else {
        (base + 1, base + 2, base + 2, base + 3, base + 3)
    };
    let mut t = 1;
    t += 12 * bin(c, st, b_luma);
    if bin(c, st, b_chroma_nz) != 0 {
        t += 4 + 4 * bin(c, st, b_chroma_two);
    }
    t += 2 * bin(c, st, b_pred0);
    t += bin(c, st, b_pred1);
    t
}

/// `mb_type` for the slice type: returns the value in the same numbering the
/// CAVLC parser uses (P: 0..=4 inter, 5+ intra; B: 0..=22 inter, 23+ intra;
/// I: intra numbering).
fn decode_mb_type(
    c: &mut Cabac,
    st: &mut CabacState,
    ctx: &SliceCtx,
    info: &PicInfo,
    nb: &MbNeighbours,
) -> Result<u32> {
    match ctx.slice_type {
        SliceType::I => Ok(decode_intra_mb_type(c, st, CTX_MB_TYPE_I, true, info, nb)),
        SliceType::Si => {
            // SI prefix then the I binarisation. SI slices are refused
            // upstream; keep the parse coherent anyway.
            let _ = CTX_MB_TYPE_SI_PREFIX;
            Err(Error::unsupported("SI slices"))
        }
        SliceType::P | SliceType::Sp => {
            if bin(c, st, CTX_MB_TYPE_P_PREFIX) != 0 {
                return Ok(5 + decode_intra_mb_type(c, st, CTX_MB_TYPE_P_SUFFIX, false, info, nb));
            }
            if bin(c, st, CTX_MB_TYPE_P_PREFIX + 1) == 0 {
                // b1 = 0: b2 (ctx 16): 0 -> P_L0_16x16, 1 -> P_8x8
                Ok(3 * bin(c, st, CTX_MB_TYPE_P_PREFIX + 2))
            } else {
                // b1 = 1: b2 (ctx 17): 1 -> 16x8, 0 -> 8x16
                Ok(2 - bin(c, st, CTX_MB_TYPE_P_PREFIX + 3))
            }
        }
        SliceType::B => {
            let cond = |a: Option<usize>| -> usize {
                match a {
                    Some(addr) => match info.mbs[addr].kind {
                        MbKind::BSkip | MbKind::BDirect16x16 => 0,
                        _ => 1,
                    },
                    None => 0,
                }
            };
            let inc = cond(nb.a) + cond(nb.b);
            if bin(c, st, CTX_MB_TYPE_B_PREFIX + inc) == 0 {
                return Ok(0); // B_Direct_16x16
            }
            if bin(c, st, CTX_MB_TYPE_B_PREFIX + 3) == 0 {
                return Ok(1 + bin(c, st, CTX_MB_TYPE_B_PREFIX + 5));
            }
            let mut bits = bin(c, st, CTX_MB_TYPE_B_PREFIX + 4) << 3;
            bits |= bin(c, st, CTX_MB_TYPE_B_PREFIX + 5) << 2;
            bits |= bin(c, st, CTX_MB_TYPE_B_PREFIX + 5) << 1;
            bits |= bin(c, st, CTX_MB_TYPE_B_PREFIX + 5);
            if bits < 8 {
                return Ok(bits + 3);
            }
            match bits {
                13 => Ok(23 + decode_intra_mb_type(c, st, CTX_MB_TYPE_B_SUFFIX, false, info, nb)),
                14 => Ok(11),
                15 => Ok(22),
                _ => {
                    let bits = (bits << 1) | bin(c, st, CTX_MB_TYPE_B_PREFIX + 5);
                    Ok(bits - 4)
                }
            }
        }
    }
}

fn decode_sub_mb_type_p(c: &mut Cabac, st: &mut CabacState) -> u32 {
    if bin(c, st, CTX_SUB_MB_TYPE_P) != 0 {
        return 0;
    }
    if bin(c, st, CTX_SUB_MB_TYPE_P + 1) == 0 {
        return 1;
    }
    if bin(c, st, CTX_SUB_MB_TYPE_P + 2) != 0 { 2 } else { 3 }
}

fn decode_sub_mb_type_b(c: &mut Cabac, st: &mut CabacState) -> u32 {
    if bin(c, st, CTX_SUB_MB_TYPE_B) == 0 {
        return 0;
    }
    if bin(c, st, CTX_SUB_MB_TYPE_B + 1) == 0 {
        return 1 + bin(c, st, CTX_SUB_MB_TYPE_B + 3);
    }
    let mut t = 3;
    if bin(c, st, CTX_SUB_MB_TYPE_B + 2) != 0 {
        if bin(c, st, CTX_SUB_MB_TYPE_B + 3) != 0 {
            return 11 + bin(c, st, CTX_SUB_MB_TYPE_B + 3);
        }
        t += 4;
    }
    t += 2 * bin(c, st, CTX_SUB_MB_TYPE_B + 3);
    t += bin(c, st, CTX_SUB_MB_TYPE_B + 3);
    t
}

/// Whether the 4x4 block `(mb, blk)` belongs to a direct-predicted
/// partition (B_Skip / B_Direct_16x16 / B_Direct_8x8), for the ref_idx and
/// mvd contexts (`predModeEqualFlag` = 0).
fn is_direct_block(info: &PicInfo, layer: &MbLayer, cur: usize, addr: usize, blk: usize) -> bool {
    if addr == cur {
        return layer.kind.is_direct16x16() || layer.sub_shape[(blk / 8) * 2 + (blk % 4) / 2] == SubMbShape::Direct;
    }
    let m = &info.mbs[addr];
    m.kind.is_direct16x16() || (m.kind == MbKind::Inter8x8 && (m.sub_direct >> ((blk / 8) * 2 + (blk % 4) / 2)) & 1 != 0)
}

/// `ref_idx_lX` (9.3.3.1.1.6): unary, contexts 54..=59.
fn decode_ref_idx(
    c: &mut Cabac,
    st: &mut CabacState,
    info: &PicInfo,
    layer: &MbLayer,
    nb: &MbNeighbours,
    frame_motion: &[Vec<super::frame::BlockMotion>; 2],
    list: usize,
    bx: i32,
    by: i32,
) -> Result<i8> {
    let cond = |dx: i32, dy: i32| -> usize {
        let Some((addr, blk)) = nb.block(bx + dx, by + dy) else { return 0 };
        if addr == nb.addr {
            // Inside the current macroblock: the partition's own already-set
            // reference index (partitions are decoded in order, so a left/up
            // neighbour inside the MB is always earlier).
            if is_direct_block(info, layer, nb.addr, addr, blk) {
                return 0;
            }
            let part = (blk / 8) * 2 + (blk % 4) / 2;
            if layer.pred_dir[part] & (1 << list) == 0 {
                return 0;
            }
            return (layer.ref_idx[list][part] > 0) as usize;
        }
        let m = &info.mbs[addr];
        if m.kind.is_skip() || m.kind.is_intra() || is_direct_block(info, layer, nb.addr, addr, blk) {
            return 0;
        }
        (frame_motion[list][addr * 16 + blk].ref_idx > 0) as usize
    };
    let inc = cond(-1, 0) + 2 * cond(0, -1);
    let mut v: i8 = 0;
    let mut ctx = CTX_REF_IDX + inc;
    while bin(c, st, ctx) != 0 {
        v += 1;
        ctx = CTX_REF_IDX + if v == 1 { 4 } else { 5 };
        if v > 31 {
            return Err(Error::bitstream("ref_idx runaway"));
        }
    }
    Ok(v)
}

/// `mvd_lX` component (9.3.3.1.1.7): TU prefix (cMax 9) + UEG3 suffix + sign.
fn decode_mvd_component(
    c: &mut Cabac,
    st: &mut CabacState,
    info: &PicInfo,
    layer: &MbLayer,
    nb: &MbNeighbours,
    list: usize,
    comp: usize,
    bx: i32,
    by: i32,
) -> Result<i16> {
    let abs_of = |dx: i32, dy: i32| -> i32 {
        let Some((addr, blk)) = nb.block(bx + dx, by + dy) else { return 0 };
        if addr == nb.addr {
            if is_direct_block(info, layer, nb.addr, addr, blk) {
                return 0;
            }
            let m = layer.mvd[blk].mvd[list];
            return if comp == 0 { m.x.abs() as i32 } else { m.y.abs() as i32 };
        }
        let mi = &info.mbs[addr];
        if mi.kind.is_skip() || mi.kind.is_intra() || is_direct_block(info, layer, nb.addr, addr, blk) {
            return 0;
        }
        let m = info.mvd[list][addr * 16 + blk];
        if comp == 0 { m.x.abs() as i32 } else { m.y.abs() as i32 }
    };
    let sum = abs_of(-1, 0) + abs_of(0, -1);
    let base = if comp == 0 { CTX_MVD_X } else { CTX_MVD_Y };
    let inc = if sum < 3 {
        0
    } else if sum <= 32 {
        1
    } else {
        2
    };
    // Prefix: TU with cMax = 9; bin 0 uses inc, bins 1.. use 3,4,5,6,6,6,6,6.
    let mut prefix = 0u32;
    if bin(c, st, base + inc) != 0 {
        prefix = 1;
        let incs = [3usize, 4, 5, 6, 6, 6, 6, 6];
        while prefix < 9 && bin(c, st, base + incs[(prefix - 1) as usize]) != 0 {
            prefix += 1;
        }
    }
    let mut abs = prefix as i32;
    if prefix >= 9 {
        // UEG3 suffix (uCoff = 9).
        let mut k = 3u32;
        loop {
            if c.bypass() != 0 {
                abs += 1 << k;
                k += 1;
                if k > 24 {
                    return Err(Error::bitstream("mvd suffix runaway"));
                }
            } else {
                break;
            }
        }
        while k > 0 {
            k -= 1;
            abs += (c.bypass() as i32) << k;
        }
    }
    if abs == 0 {
        return Ok(0);
    }
    let sign = c.bypass();
    let v = if sign != 0 { -abs } else { abs };
    if !(-32768..=32767).contains(&v) {
        return Err(Error::bitstream("mvd out of range"));
    }
    Ok(v as i16)
}

/// `intra_chroma_pred_mode` (9.3.3.1.1.8).
fn decode_chroma_pred_mode(c: &mut Cabac, st: &mut CabacState, info: &PicInfo, nb: &MbNeighbours) -> u8 {
    let cond = |a: Option<usize>| -> usize {
        match a {
            Some(addr) => {
                let m = &info.mbs[addr];
                if !m.kind.is_intra() || m.kind == MbKind::IPcm || m.chroma_mode == 0 { 0 } else { 1 }
            }
            None => 0,
        }
    };
    let inc = cond(nb.a) + cond(nb.b);
    if bin(c, st, CTX_INTRA_CHROMA_PRED_MODE + inc) == 0 {
        return 0;
    }
    if bin(c, st, CTX_INTRA_CHROMA_PRED_MODE + 3) == 0 {
        return 1;
    }
    if bin(c, st, CTX_INTRA_CHROMA_PRED_MODE + 3) == 0 { 2 } else { 3 }
}

/// `prev_intra4x4_pred_mode_flag` / `rem_intra4x4_pred_mode` → the mode.
fn decode_intra_pred_mode(c: &mut Cabac, st: &mut CabacState, pred: u8) -> u8 {
    if bin(c, st, CTX_PREV_INTRA_PRED_MODE_FLAG) != 0 {
        return pred;
    }
    let mut rem = bin(c, st, CTX_REM_INTRA_PRED_MODE);
    rem |= bin(c, st, CTX_REM_INTRA_PRED_MODE) << 1;
    rem |= bin(c, st, CTX_REM_INTRA_PRED_MODE) << 2;
    let rem = rem as u8;
    if rem < pred { rem } else { rem + 1 }
}

/// `coded_block_pattern` (9.3.3.1.1.4).
fn decode_cbp(c: &mut Cabac, st: &mut CabacState, info: &PicInfo, nb: &MbNeighbours, chroma: bool) -> u8 {
    let mut cbp: u32 = 0;
    // Luma: four bins, 8x8 blocks 0..3; each context from the left/above 8x8
    // block (in this MB from earlier bins).
    for b8 in 0..4u32 {
        let (bx8, by8) = (b8 & 1, b8 >> 1);
        let cond = |dx: i32, dy: i32| -> u32 {
            let nx = bx8 as i32 + dx;
            let ny = by8 as i32 + dy;
            if nx >= 0 && ny >= 0 {
                // Inside the current MB: prior bin.
                let nb8 = (ny as u32) * 2 + nx as u32;
                return if (cbp >> nb8) & 1 != 0 { 0 } else { 1 };
            }
            let (addr, other_b8) = if nx < 0 {
                match nb.a {
                    Some(a) => (a, by8 * 2 + 1),
                    None => return 0,
                }
            } else {
                match nb.b {
                    Some(a) => (a, 2 + bx8),
                    None => return 0,
                }
            };
            let m = &info.mbs[addr];
            if m.kind == MbKind::IPcm {
                return 0;
            }
            if m.kind.is_skip() {
                return 1;
            }
            if (m.cbp >> other_b8) & 1 != 0 { 0 } else { 1 }
        };
        let inc = cond(-1, 0) + 2 * cond(0, -1);
        cbp |= bin(c, st, CTX_CBP_LUMA + inc as usize) << b8;
    }
    if chroma {
        let cond = |a: Option<usize>, want_two: bool| -> usize {
            match a {
                Some(addr) => {
                    let m = &info.mbs[addr];
                    if m.kind == MbKind::IPcm {
                        return 1;
                    }
                    if m.kind.is_skip() {
                        return 0;
                    }
                    let ch = m.cbp >> 4;
                    if want_two { (ch == 2) as usize } else { (ch != 0) as usize }
                }
                None => 0,
            }
        };
        let inc = cond(nb.a, false) + 2 * cond(nb.b, false);
        if bin(c, st, CTX_CBP_CHROMA + inc) != 0 {
            let inc2 = cond(nb.a, true) + 2 * cond(nb.b, true) + 4;
            let two = bin(c, st, CTX_CBP_CHROMA + inc2);
            cbp |= (1 + two) << 4;
        }
    }
    cbp as u8
}

/// `mb_qp_delta` (9.3.3.1.1.5): mapped unary.
fn decode_qp_delta(c: &mut Cabac, st: &mut CabacState) -> Result<i32> {
    let inc = st.prev_qp_delta_nonzero as usize;
    if bin(c, st, CTX_MB_QP_DELTA + inc) == 0 {
        return Ok(0);
    }
    let mut k = 1u32;
    let mut ctx = CTX_MB_QP_DELTA + 2;
    while bin(c, st, ctx) != 0 {
        k += 1;
        ctx = CTX_MB_QP_DELTA + 3;
        if k > 52 {
            return Err(Error::bitstream("mb_qp_delta runaway"));
        }
    }
    let v = if k & 1 == 1 { ((k + 1) / 2) as i32 } else { -((k / 2) as i32) };
    Ok(v)
}

/// `transform_size_8x8_flag` (9.3.3.1.1.10).
fn decode_transform_8x8(c: &mut Cabac, st: &mut CabacState, info: &PicInfo, nb: &MbNeighbours) -> bool {
    let cond = |a: Option<usize>| -> usize {
        match a {
            Some(addr) => info.mbs[addr].transform_8x8 as usize,
            None => 0,
        }
    };
    let inc = cond(nb.a) + cond(nb.b);
    bin(c, st, CTX_TRANSFORM_8X8 + inc) != 0
}

// ---------------------------------------------------------------------------
// Residual blocks (9.3.3.1.1.9, 9.3.3.1.3)
// ---------------------------------------------------------------------------

/// The `coded_block_flag` context increment for a block of category `cat`
/// at luma 4x4 raster `(bx, by)` (cat 1/2), chroma block `(comp, blk)`
/// (cat 4), or the DC blocks (cat 0/3).
#[allow(clippy::too_many_arguments)]
fn cbf_ctx_inc(
    info: &PicInfo,
    layer: &MbLayer,
    nb: &MbNeighbours,
    x264_old_444: bool,
    cat: usize,
    bx: usize,
    by: usize,
    comp: usize,
    blk: usize,
    chroma_rows: usize,
) -> usize {
    let cur_intra = layer.kind.is_intra();
    let p = cat_plane(cat);
    let cat_8x8 = matches!(cat, 5 | 9 | 13);
    // condTermFlagN for the luma-style 4x4 (or 8x8) block of plane `p` at
    // 4x4 offset `(dx, dy)` from block `(bx, by)`. The block's flag is its
    // nonzero count: an 8x8 block without a coded_block_flag (non-4:4:4,
    // inferred 1) always carries a coefficient, and one with the flag
    // (4:4:4) carries one iff the flag was 1 — so the count stands in for
    // both, and for a neighbouring 8x8-transformed macroblock it is the
    // 8x8 block's flag on all four of its 4x4s. An 8x8 block's own flag
    // (categories 5 / 9 / 13, 4:4:4 only) looks at neighbouring 8x8
    // *transform* blocks: a neighbour transformed 4x4 contributes 0.
    let cond_luma = |dx: i32, dy: i32| -> usize {
        let Some((addr, nblk)) = nb.block(bx as i32 + dx, by as i32 + dy) else {
            return cur_intra as usize;
        };
        if addr == nb.addr {
            // Current MB: block available iff its 8x8 has cbp set (it has,
            // or we would not be decoding it) — the flag is the count so far.
            let b8 = (nblk / 8) * 2 + (nblk % 4) / 2;
            if layer.cbp & (1 << b8) == 0 {
                return 0;
            }
            return (layer.nz[p][nblk] != 0) as usize;
        }
        let m = &info.mbs[addr];
        if cat_8x8 && !m.transform_8x8 && x264_old_444 {
            // Old x264 (before build 151) coded these as if the neighbour
            // were unavailable.
            return cur_intra as usize;
        }
        if m.kind == MbKind::IPcm {
            return 1;
        }
        if m.kind.is_skip() || (cat_8x8 && !m.transform_8x8) {
            return 0;
        }
        let b8 = (nblk / 8) * 2 + (nblk % 4) / 2;
        if m.cbp & (1 << b8) == 0 {
            return 0;
        }
        (info.plane_nz(p, addr, nblk) != 0) as usize
    };
    let cond_dc = |a: Option<usize>, bit: u8, needs_i16: bool| -> usize {
        match a {
            None => cur_intra as usize,
            Some(addr) => {
                let m = &info.mbs[addr];
                if m.kind == MbKind::IPcm {
                    return 1;
                }
                if needs_i16 {
                    if m.kind != MbKind::I16x16 {
                        return 0;
                    }
                } else if m.kind.is_skip() || m.cbp & 0x30 == 0 {
                    return 0;
                }
                ((m.dc_cbf >> bit) & 1) as usize
            }
        }
    };
    let cond_chroma_ac = |dx: i32, dy: i32| -> usize {
        // Chroma 2-column grid neighbours (2 rows for 4:2:0, 4 for 4:2:2).
        let (cx, cy) = ((blk % 2) as i32 + dx, (blk / 2) as i32 + dy);
        let (addr, nblk) = if cx >= 0 && cy >= 0 {
            (nb.addr, (cy * 2 + cx) as usize)
        } else if cx < 0 {
            match nb.a {
                Some(a) => (a, (cy * 2 + 1) as usize),
                None => return cur_intra as usize,
            }
        } else {
            match nb.b {
                Some(a) => (a, ((chroma_rows as i32 - 1) * 2 + cx) as usize),
                None => return cur_intra as usize,
            }
        };
        if addr == nb.addr {
            return (layer.chroma_nz[comp][nblk] != 0) as usize;
        }
        let m = &info.mbs[addr];
        if m.kind == MbKind::IPcm {
            return 1;
        }
        if m.kind.is_skip() || (m.cbp >> 4) != 2 {
            return 0;
        }
        (info.chroma_nz[addr * 32 + comp * 16 + nblk] != 0) as usize
    };
    match cat {
        // Luma-style DC (Intra_16x16 of the plane): bit p of `dc_cbf`.
        0 | 6 | 10 => cond_dc(nb.a, p as u8, true) + 2 * cond_dc(nb.b, p as u8, true),
        // Luma-style AC / 4x4 / 8x8 (the 8x8's top-left 4x4 is `(bx, by)`).
        1 | 2 | 5 | 7 | 8 | 9 | 11 | 12 | 13 => cond_luma(-1, 0) + 2 * cond_luma(0, -1),
        CAT_CHROMA_DC => cond_dc(nb.a, 1 + comp as u8, false) + 2 * cond_dc(nb.b, 1 + comp as u8, false),
        CAT_CHROMA_AC => cond_chroma_ac(-1, 0) + 2 * cond_chroma_ac(0, -1),
        _ => 0,
    }
}

/// Decode one residual block's coefficients into `levels` (scan order,
/// `max_coeff` entries; the caller maps to raster). Returns the number of
/// nonzero coefficients. `cbf_inc` is `None` when coded_block_flag is not
/// present (8x8 luma in non-4:4:4), else its ctxIdxInc.
fn residual_block_cabac(
    c: &mut Cabac,
    st: &mut CabacState,
    cat: usize,
    cbf_inc: Option<usize>,
    levels: &mut [i32],
    max_coeff: usize,
) -> Result<usize> {
    // Chroma DC significance contexts: `Min(i / NumC8x8, 2)`, NumC8x8 = 1 for
    // 4:2:0 (4 coefficients) and 2 for 4:2:2 (8).
    let dc_div = if cat == CAT_CHROMA_DC && max_coeff == 8 { 2 } else { 1 };
    if let Some(inc) = cbf_inc {
        if bin(c, st, CBF_CTX_BASE[cat] + inc) == 0 {
            return Ok(0);
        }
    }
    let (sig_base, last_base, abs_base) = (SIG_CTX_BASE[cat], LAST_CTX_BASE[cat], ABS_CTX_BASE[cat]);
    let is_8x8 = max_coeff == 64;
    // Significance map.
    let mut sig = [false; 64];
    let mut n_sig = 0usize;
    let mut last_pos = max_coeff - 1;
    let mut i = 0usize;
    while i < max_coeff - 1 {
        let (sig_inc, last_inc) = if is_8x8 {
            (SIG_COEFF_8X8_CTX[0][i] as usize, LAST_COEFF_8X8_CTX[i] as usize)
        } else if cat == CAT_CHROMA_DC {
            ((i / dc_div).min(2), (i / dc_div).min(2))
        } else {
            (i, i)
        };
        if bin(c, st, sig_base + sig_inc) != 0 {
            sig[i] = true;
            n_sig += 1;
            if bin(c, st, last_base + last_inc) != 0 {
                last_pos = i;
                break;
            }
        }
        i += 1;
    }
    if i == max_coeff - 1 {
        // Reached the end without a "last": the final coefficient is significant.
        sig[max_coeff - 1] = true;
        n_sig += 1;
        last_pos = max_coeff - 1;
    }
    // Levels, in reverse scan order.
    let mut num_gt1 = 0usize;
    let mut num_eq1 = 0usize;
    for pos in (0..=last_pos).rev() {
        if !sig[pos] {
            continue;
        }
        let inc0 = if num_gt1 != 0 { 0 } else { (1 + num_eq1).min(4) };
        let mut abs_m1: i32;
        if bin(c, st, abs_base + inc0) == 0 {
            abs_m1 = 0;
        } else {
            let inc1 = 5 + (if cat == CAT_CHROMA_DC { 3 } else { 4 }).min(num_gt1);
            let mut prefix = 1;
            while prefix < 14 && bin(c, st, abs_base + inc1) != 0 {
                prefix += 1;
            }
            abs_m1 = prefix;
            if prefix >= 14 {
                // UEG0 suffix.
                let mut k = 0u32;
                loop {
                    if c.bypass() != 0 {
                        abs_m1 += 1 << k;
                        k += 1;
                        if k > 24 {
                            return Err(Error::bitstream("coeff_abs_level suffix runaway"));
                        }
                    } else {
                        break;
                    }
                }
                while k > 0 {
                    k -= 1;
                    abs_m1 += (c.bypass() as i32) << k;
                }
            }
        }
        let abs = abs_m1 + 1;
        if abs == 1 {
            num_eq1 += 1;
        } else {
            num_gt1 += 1;
        }
        let sign = c.bypass();
        levels[pos] = if sign != 0 { -abs } else { abs };
    }
    if c.overrun() {
        return Err(Error::bitstream("CABAC: slice data truncated"));
    }
    Ok(n_sig)
}

/// `residual_luma()` (7.3.5.3.1) for CABAC, for colour plane `p`: the luma
/// plane, or Cb / Cr in 4:4:4 (coded like luma with their own categories,
/// and with a coded_block_flag on the 8x8 blocks too).
fn parse_residual_luma_like_cabac(
    c: &mut Cabac,
    st: &mut CabacState,
    ctx: &SliceCtx,
    info: &PicInfo,
    nb: &MbNeighbours,
    layer: &mut MbLayer,
    p: usize,
) -> Result<()> {
    let [cat_dc, cat_ac, cat_4x4, cat_8x8] = PLANE_CATS[p];
    let mut levels = [0i32; 64];
    if layer.kind == MbKind::I16x16 {
        levels[..16].fill(0);
        let inc = cbf_ctx_inc(info, layer, nb, ctx.x264_old_444, cat_dc, 0, 0, 0, 0, 2);
        let n = residual_block_cabac(c, st, cat_dc, Some(inc), &mut levels[..16], 16)?;
        if n > 0 {
            layer.dc_cbf |= 1 << p;
            for i in 0..16 {
                layer.dc[p][ZIGZAG4X4[i] as usize] = levels[i];
            }
        }
    }
    for blk8 in 0..4 {
        let (bx8, by8) = ((blk8 & 1) * 2, (blk8 >> 1) * 2);
        if layer.cbp & (1 << blk8) == 0 {
            continue;
        }
        if layer.transform_8x8 {
            levels.fill(0);
            // The 8x8 block's coded_block_flag is only coded in 4:4:4;
            // otherwise it is inferred 1.
            let inc = if ctx.chroma_format_idc == 3 { Some(cbf_ctx_inc(info, layer, nb, ctx.x264_old_444, cat_8x8, bx8, by8, 0, 0, 2)) } else { None };
            let n = residual_block_cabac(c, st, cat_8x8, inc, &mut levels, 64)?;
            for sub in 0..4 {
                let (bx, by) = (bx8 + (sub & 1), by8 + (sub >> 1));
                layer.nz[p][by * 4 + bx] = n as u8;
            }
            if n > 0 {
                let base = blk8 * 64;
                for i in 0..64 {
                    layer.coef[p][base + ZIGZAG8X8[i] as usize] = levels[i];
                }
            }
        } else {
            for sub in 0..4 {
                let (bx, by) = (bx8 + (sub & 1), by8 + (sub >> 1));
                let raster = by * 4 + bx;
                levels[..16].fill(0);
                let n = if layer.kind == MbKind::I16x16 {
                    let inc = cbf_ctx_inc(info, layer, nb, ctx.x264_old_444, cat_ac, bx, by, 0, 0, 2);
                    // AC: 15 coefficients at scan positions 1..15.
                    residual_block_cabac(c, st, cat_ac, Some(inc), &mut levels[1..16], 15)?
                } else {
                    let inc = cbf_ctx_inc(info, layer, nb, ctx.x264_old_444, cat_4x4, bx, by, 0, 0, 2);
                    residual_block_cabac(c, st, cat_4x4, Some(inc), &mut levels[..16], 16)?
                };
                layer.nz[p][raster] = n as u8;
                if n > 0 {
                    let base = raster * 16;
                    for i in 0..16 {
                        layer.coef[p][base + ZIGZAG4X4[i] as usize] = levels[i];
                    }
                }
            }
        }
    }
    Ok(())
}

/// `residual()` for CABAC, filling the layer's coefficient arrays.
fn parse_residual_cabac(
    c: &mut Cabac,
    st: &mut CabacState,
    ctx: &SliceCtx,
    info: &PicInfo,
    nb: &MbNeighbours,
    layer: &mut MbLayer,
) -> Result<()> {
    let mut levels = [0i32; 64];
    parse_residual_luma_like_cabac(c, st, ctx, info, nb, layer, 0)?;
    if ctx.chroma_format_idc == 3 {
        parse_residual_luma_like_cabac(c, st, ctx, info, nb, layer, 1)?;
        parse_residual_luma_like_cabac(c, st, ctx, info, nb, layer, 2)?;
    }
    if (ctx.chroma_format_idc == 1 || ctx.chroma_format_idc == 2) && layer.cbp & 0x30 != 0 {
        let c422 = ctx.chroma_format_idc == 2;
        let (n_dc, rows) = if c422 { (8usize, 4usize) } else { (4, 2) };
        for comp in 0..2 {
            levels[..n_dc].fill(0);
            let inc = cbf_ctx_inc(info, layer, nb, false, CAT_CHROMA_DC, 0, 0, comp, 0, rows);
            let n = residual_block_cabac(c, st, CAT_CHROMA_DC, Some(inc), &mut levels[..n_dc], n_dc)?;
            if n > 0 {
                layer.dc_cbf |= 2 << comp;
                if c422 {
                    for i in 0..8 {
                        layer.chroma_dc[comp][SCAN_CHROMA_DC_422[i] as usize] = levels[i];
                    }
                } else {
                    layer.chroma_dc[comp][..4].copy_from_slice(&levels[..4]);
                }
            }
        }
        if layer.cbp & 0x20 != 0 {
            for comp in 0..2 {
                for blk in 0..2 * rows {
                    levels[..16].fill(0);
                    let inc = cbf_ctx_inc(info, layer, nb, false, CAT_CHROMA_AC, 0, 0, comp, blk, rows);
                    let n = residual_block_cabac(c, st, CAT_CHROMA_AC, Some(inc), &mut levels[1..16], 15)?;
                    layer.chroma_nz[comp][blk] = n as u8;
                    if n > 0 {
                        for i in 1..16 {
                            layer.chroma_ac[comp][blk][ZIGZAG4X4[i] as usize] = levels[i];
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

/// Parse one CABAC macroblock (after `mb_skip_flag` was 0), including the
/// residual. `frame_motion` is the current picture's motion so far (for the
/// ref_idx contexts of neighbours in other macroblocks).
pub fn parse_mb_cabac(
    c: &mut Cabac,
    st: &mut CabacState,
    ctx: &SliceCtx,
    info: &PicInfo,
    nb: &MbNeighbours,
    frame_motion: &[Vec<super::frame::BlockMotion>; 2],    layer: &mut MbLayer,
) -> Result<()> {
    layer.reset(MbKind::I4x4, true);
    let t = decode_mb_type(c, st, ctx, info, nb)?;
    match ctx.slice_type {
        SliceType::I | SliceType::Si => super::cavlc::intra_mb_type(t, layer)?,
        SliceType::P | SliceType::Sp => {
            if t < 5 {
                super::cavlc::p_mb_type(t, layer)?;
            } else {
                super::cavlc::intra_mb_type(t - 5, layer)?;
            }
        }
        SliceType::B => {
            if t < 23 {
                super::cavlc::b_mb_type(t, layer)?;
            } else {
                super::cavlc::intra_mb_type(t - 23, layer)?;
            }
        }
    }

    if layer.kind == MbKind::IPcm {
        // The arithmetic decoder stopped after the terminate bin; raw
        // samples follow at the next byte boundary, then it re-initialises.
        let r = c.reader();
        r.align();
        let n = 256 + match ctx.chroma_format_idc {
            0 => 0,
            1 => 128,
            2 => 256,
            _ => 512,
        };
        layer.pcm = (0..n).map(|_| r.bits(ctx.bit_depth) as u16).collect();
        if r.overrun() {
            return Err(Error::bitstream("I_PCM samples truncated"));
        }
        c.reinit();
        st.prev_qp_delta_nonzero = false;
        return Ok(());
    }

    let mut no_sub_mb_part_less_than_8x8 = true;
    if layer.kind == MbKind::Inter8x8 {
        for part in 0..4 {
            let t = if ctx.slice_type.is_b() { decode_sub_mb_type_b(c, st) } else { decode_sub_mb_type_p(c, st) };
            let (shape, dir) = if ctx.slice_type.is_b() { b_sub_mb_type(t)? } else { p_sub_mb_type(t)? };
            layer.sub_shape[part] = shape;
            layer.pred_dir[part] = dir;
            if shape == SubMbShape::Direct {
                if !ctx.direct_8x8_inference {
                    no_sub_mb_part_less_than_8x8 = false;
                }
            } else if shape.count() > 1 {
                no_sub_mb_part_less_than_8x8 = false;
            }
        }
        for list in 0..2 {
            for part in 0..4 {
                if layer.sub_shape[part] == SubMbShape::Direct || layer.pred_dir[part] & (1 << list) == 0 {
                    continue;
                }
                let n = ctx.num_ref_idx[list];
                let (bx, by) = (((part & 1) * 2) as i32, ((part >> 1) * 2) as i32);
                let ri = if n <= 1 { 0 } else { decode_ref_idx(c, st, info, &layer, nb, frame_motion, list, bx, by)? };
                if ri as u32 >= n.max(1) {
                    return Err(Error::bitstream("ref_idx out of range"));
                }
                layer.ref_idx[list][part] = ri;
            }
        }
        for list in 0..2 {
            for part in 0..4 {
                let shape = layer.sub_shape[part];
                if shape == SubMbShape::Direct || layer.pred_dir[part] & (1 << list) == 0 {
                    continue;
                }
                for sub in 0..shape.count() {
                    let (x, y, _, _) = sub_partition_rect(part, shape, sub);
                    let (bx, by) = ((x / 4) as i32, (y / 4) as i32);
                    let mx = decode_mvd_component(c, st, info, &layer, nb, list, 0, bx, by)?;
                    let my = decode_mvd_component(c, st, info, &layer, nb, list, 1, bx, by)?;
                    // The mvd applies to every 4x4 of the sub-partition (for
                    // later neighbours' contexts).
                    let (_, _, w, h) = sub_partition_rect(part, shape, sub);
                    for yy in y / 4..(y + h) / 4 {
                        for xx in x / 4..(x + w) / 4 {
                            layer.mvd[yy * 4 + xx].mvd[list] = Mv::new(mx, my);
                        }
                    }
                }
            }
        }
    } else {
        if ctx.transform_8x8_mode && layer.kind == MbKind::I4x4 {
            layer.transform_8x8 = decode_transform_8x8(c, st, info, nb);
            if layer.transform_8x8 {
                layer.kind = MbKind::I8x8;
            }
        }
        match layer.kind {
            MbKind::I4x4 => {
                for blk in 0..16 {
                    let raster = super::mb::raster_of_blk(blk);
                    let (bx, by) = (raster % 4, raster / 4);
                    let pred = predicted_intra_mode(info, &layer, nb, ctx, bx, by, false);
                    layer.intra_modes[raster] = decode_intra_pred_mode(c, st, pred);
                }
                if ctx.chroma_format_idc == 1 || ctx.chroma_format_idc == 2 {
                    layer.chroma_mode = decode_chroma_pred_mode(c, st, info, nb);
                }
            }
            MbKind::I8x8 => {
                for blk8 in 0..4 {
                    let (bx, by) = ((blk8 & 1) * 2, (blk8 >> 1) * 2);
                    let pred = predicted_intra_mode(info, &layer, nb, ctx, bx, by, true);
                    let mode = decode_intra_pred_mode(c, st, pred);
                    for dy in 0..2 {
                        for dx in 0..2 {
                            layer.intra_modes[(by + dy) * 4 + bx + dx] = mode;
                        }
                    }
                }
                if ctx.chroma_format_idc == 1 || ctx.chroma_format_idc == 2 {
                    layer.chroma_mode = decode_chroma_pred_mode(c, st, info, nb);
                }
            }
            MbKind::I16x16 => {
                if ctx.chroma_format_idc == 1 || ctx.chroma_format_idc == 2 {
                    layer.chroma_mode = decode_chroma_pred_mode(c, st, info, nb);
                }
            }
            MbKind::BDirect16x16 => {}
            _ => {
                let parts = mb_partitions(layer.kind);
                for list in 0..2 {
                    for &(x, y, w, h) in parts {
                        let part = part_index_of(x, y);
                        if layer.pred_dir[part] & (1 << list) == 0 {
                            continue;
                        }
                        let n = ctx.num_ref_idx[list];
                        let ri = if n <= 1 {
                            0
                        } else {
                            decode_ref_idx(c, st, info, &layer, nb, frame_motion, list, (x / 4) as i32, (y / 4) as i32)?
                        };
                        if ri as u32 >= n.max(1) {
                            return Err(Error::bitstream("ref_idx out of range"));
                        }
                        for by in y / 8..(y + h) / 8 {
                            for bx in x / 8..(x + w) / 8 {
                                layer.ref_idx[list][by * 2 + bx] = ri;
                            }
                        }
                    }
                }
                for list in 0..2 {
                    for &(x, y, w, h) in parts {
                        let part = part_index_of(x, y);
                        if layer.pred_dir[part] & (1 << list) == 0 {
                            continue;
                        }
                        let (bx, by) = ((x / 4) as i32, (y / 4) as i32);
                        let mx = decode_mvd_component(c, st, info, &layer, nb, list, 0, bx, by)?;
                        let my = decode_mvd_component(c, st, info, &layer, nb, list, 1, bx, by)?;
                        for yy in y / 4..(y + h) / 4 {
                            for xx in x / 4..(x + w) / 4 {
                                layer.mvd[yy * 4 + xx].mvd[list] = Mv::new(mx, my);
                            }
                        }
                    }
                }
            }
        }
    }

    if layer.kind != MbKind::I16x16 {
        layer.cbp = decode_cbp(c, st, info, nb, ctx.chroma_format_idc == 1 || ctx.chroma_format_idc == 2);
        if layer.cbp & 15 != 0
            && ctx.transform_8x8_mode
            && !layer.kind.is_intra()
            && no_sub_mb_part_less_than_8x8
            && (layer.kind != MbKind::BDirect16x16 || ctx.direct_8x8_inference)
        {
            layer.transform_8x8 = decode_transform_8x8(c, st, info, nb);
        }
    }

    if layer.has_residual() {
        layer.qp_delta = decode_qp_delta(c, st)?;
        if !(-26..=25).contains(&layer.qp_delta) {
            return Err(Error::bitstream("mb_qp_delta out of range"));
        }
        st.prev_qp_delta_nonzero = layer.qp_delta != 0;
        parse_residual_cabac(c, st, ctx, info, nb, layer)?;
    } else {
        st.prev_qp_delta_nonzero = false;
    }
    if c.overrun() {
        return Err(Error::bitstream("slice data truncated in macroblock"));
    }
    Ok(())
}

/// The value the CAVLC and CABAC parsers agree `PRED_*` on (re-exported so
/// callers have one import).
pub const _PRED_CHECK: (u8, u8, u8) = (PRED_L0, PRED_L1, PRED_BI);
