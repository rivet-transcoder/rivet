//! CAVLC (H.264 clause 9.2) and the CAVLC-coded macroblock layer (7.3.5).

use std::sync::OnceLock;

use crate::bitreader::BitReader;
use crate::{Error, Result};

use super::mb::{
    MbKind, MbLayer, MbNeighbours, PicInfo, SliceCtx, SubMbShape, PRED_BI, PRED_L0, PRED_L1,
};
use super::frame::Mv;
use super::slice::SliceType;
use super::tables::*;

/// A decoded VLC entry: `(code length, value)`; length 0 = invalid code.
#[derive(Clone, Copy, Default)]
struct VlcEntry {
    len: u8,
    a: u8,
    b: u8,
}

/// Lookup tables built once: `coeff_token` for the five nC classes as two
/// levels of 8 bits (a first-level entry with `len == ESCAPE` names a
/// second-level table for the next 8 bits — the tables stay a few KiB and in
/// L1, where one flat 16-bit table per class was a megabyte of cache misses),
/// `total_zeros` (peek 9 bits) and `run_before` (peek 3 bits, or arithmetic).
struct VlcTables {
    coeff_token: Vec<[VlcEntry; 256]>,
    coeff_token2: Vec<[VlcEntry; 256]>,
    total_zeros: [[VlcEntry; 512]; 15],
    chroma_dc_total_zeros: [[VlcEntry; 512]; 3],
    /// 4:2:2 chroma DC (eight coefficients): tzVlcIndex 1..=7.
    chroma422_dc_total_zeros: [[VlcEntry; 512]; 7],
    /// `run_before` for zerosLeft 1..=6 (codes are at most 3 bits); zerosLeft
    /// > 6 is decoded arithmetically (see `read_run_before`).
    run_before: [[VlcEntry; 8]; 6],
}

/// `len` of a first-level `coeff_token` entry that continues in
/// `coeff_token2[a]`.
const ESCAPE: u8 = 0xff;

/// Build the two-level table for one class from `(len, code) -> (tc, t1)`.
fn build_coeff_token(first: &mut [VlcEntry; 256], second: &mut Vec<[VlcEntry; 256]>, codes: &[(u8, u32, u8, u8)]) {
    for &(len, code, tc, t1) in codes {
        if len == 0 {
            continue;
        }
        if len <= 8 {
            build_table(first, 8, len, code, tc, t1);
        } else {
            let prefix = (code >> (len - 8)) as usize;
            let sub_id = if first[prefix].len == ESCAPE {
                first[prefix].a as usize
            } else {
                second.push([VlcEntry::default(); 256]);
                first[prefix] = VlcEntry { len: ESCAPE, a: (second.len() - 1) as u8, b: 0 };
                second.len() - 1
            };
            let rest_len = len - 8;
            let rest_code = code & ((1u32 << rest_len) - 1);
            build_table(&mut second[sub_id], 8, rest_len, rest_code, tc, t1);
        }
    }
}

fn build_table(entries: &mut [VlcEntry], bits: u32, len: u8, code: u32, a: u8, b: u8) {
    if len == 0 {
        return;
    }
    let shift = bits - len as u32;
    let base = (code as usize) << shift;
    for i in 0..(1usize << shift) {
        entries[base + i] = VlcEntry { len, a, b };
    }
}

fn tables() -> &'static VlcTables {
    static T: OnceLock<Box<VlcTables>> = OnceLock::new();
    T.get_or_init(|| {
        let mut coeff_token: Vec<[VlcEntry; 256]> = Vec::with_capacity(6);
        let mut coeff_token2: Vec<[VlcEntry; 256]> = Vec::new();
        for cls in 0..4 {
            let mut tab = [VlcEntry::default(); 256];
            let mut codes = Vec::new();
            for tc in 0..17 {
                for t1 in 0..4 {
                    codes.push((COEFF_TOKEN_LEN[cls][tc][t1], COEFF_TOKEN_BITS[cls][tc][t1] as u32, tc as u8, t1 as u8));
                }
            }
            build_coeff_token(&mut tab, &mut coeff_token2, &codes);
            coeff_token.push(tab);
        }
        {
            let mut tab = [VlcEntry::default(); 256];
            let mut codes = Vec::new();
            for tc in 0..5 {
                for t1 in 0..4 {
                    codes.push((CHROMA_DC_COEFF_TOKEN_LEN[tc][t1], CHROMA_DC_COEFF_TOKEN_BITS[tc][t1] as u32, tc as u8, t1 as u8));
                }
            }
            build_coeff_token(&mut tab, &mut coeff_token2, &codes);
            coeff_token.push(tab);
        }
        {
            // Class 5: 4:2:2 chroma DC (nC == -2).
            let mut tab = [VlcEntry::default(); 256];
            let mut codes = Vec::new();
            for tc in 0..9 {
                for t1 in 0..4 {
                    codes.push((CHROMA422_DC_COEFF_TOKEN_LEN[tc][t1], CHROMA422_DC_COEFF_TOKEN_BITS[tc][t1] as u32, tc as u8, t1 as u8));
                }
            }
            build_coeff_token(&mut tab, &mut coeff_token2, &codes);
            coeff_token.push(tab);
        }
        let mut total_zeros = [[VlcEntry::default(); 512]; 15];
        for (tc1, tab) in total_zeros.iter_mut().enumerate() {
            for tz in 0..16 {
                build_table(tab, 9, TOTAL_ZEROS_LEN[tc1][tz], TOTAL_ZEROS_BITS[tc1][tz] as u32, tz as u8, 0);
            }
        }
        let mut chroma_dc_total_zeros = [[VlcEntry::default(); 512]; 3];
        for (tc1, tab) in chroma_dc_total_zeros.iter_mut().enumerate() {
            for tz in 0..4 {
                build_table(
                    tab,
                    9,
                    CHROMA_DC_TOTAL_ZEROS_LEN[tc1][tz],
                    CHROMA_DC_TOTAL_ZEROS_BITS[tc1][tz] as u32,
                    tz as u8,
                    0,
                );
            }
        }
        let mut chroma422_dc_total_zeros = [[VlcEntry::default(); 512]; 7];
        for (tc1, tab) in chroma422_dc_total_zeros.iter_mut().enumerate() {
            for tz in 0..8 {
                build_table(tab, 9, CHROMA422_DC_TOTAL_ZEROS_LEN[tc1][tz], CHROMA422_DC_TOTAL_ZEROS_BITS[tc1][tz] as u32, tz as u8, 0);
            }
        }
        let mut run_before = [[VlcEntry::default(); 8]; 6];
        for (zl1, tab) in run_before.iter_mut().enumerate() {
            for rb in 0..16 {
                build_table(tab, 3, RUN_BEFORE_LEN[zl1][rb], RUN_BEFORE_BITS[zl1][rb] as u32, rb as u8, 0);
            }
        }
        Box::new(VlcTables { coeff_token, coeff_token2, total_zeros, chroma_dc_total_zeros, chroma422_dc_total_zeros, run_before })
    })
}

/// Decode `coeff_token` for the given `nC` (−1 for 4:2:0 chroma DC, −2 for
/// 4:2:2 chroma DC). Returns `(TotalCoeff, TrailingOnes)`.
#[inline(always)]
fn read_coeff_token(r: &mut BitReader, nc: i32) -> Result<(usize, usize)> {
    let t = tables();
    let cls = match nc {
        -2 => 5,
        -1 => 4,
        0..=1 => 0,
        2..=3 => 1,
        4..=7 => 2,
        _ => 3,
    };
    let bits = r.peek(16) as usize;
    let mut e = t.coeff_token[cls][bits >> 8];
    let mut len = e.len as u32;
    if e.len == ESCAPE {
        e = t.coeff_token2[e.a as usize][bits & 0xff];
        len = 8 + e.len as u32;
    }
    if e.len == 0 {
        return Err(Error::bitstream("CAVLC: invalid coeff_token"));
    }
    r.skip(len);
    Ok((e.a as usize, e.b as usize))
}

/// `residual_block_cavlc` (7.3.5.3.2 / 9.2). Writes the nonzero levels into
/// `out[scan[i]]` for scan positions `i` in `start_idx..=end_idx` — `out` is
/// the block in raster order and must be zero where nothing is written (the
/// macroblock layer's reset guarantees it). Returns `TotalCoeff`.
pub fn residual_block(
    r: &mut BitReader,
    nc: i32,
    out: &mut [i32],
    scan: &[u8],
    start_idx: usize,
    end_idx: usize,
    max_num_coeff: usize,
) -> Result<usize> {
    // Work on a local copy of the reader so its fields live in registers
    // through the many small reads below, then hand the position back.
    let mut rd = r.clone();
    let res = residual_block_inner(&mut rd, nc, out, scan, start_idx, end_idx, max_num_coeff);
    *r = rd;
    res
}

#[inline(always)]
fn residual_block_inner(
    r: &mut BitReader,
    nc: i32,
    out: &mut [i32],
    scan: &[u8],
    start_idx: usize,
    end_idx: usize,
    max_num_coeff: usize,
) -> Result<usize> {
    let (total_coeff, trailing_ones) = read_coeff_token(r, nc)?;
    if total_coeff > end_idx - start_idx + 1 {
        return Err(Error::bitstream("CAVLC: TotalCoeff larger than the block"));
    }
    if total_coeff == 0 {
        return Ok(0);
    }
    let mut level_val = [0i32; 16];
    let mut suffix_length: u32 = if total_coeff > 10 && trailing_ones < 3 { 1 } else { 0 };
    if trailing_ones > 0 {
        let signs = r.bits(trailing_ones as u32);
        for i in 0..trailing_ones {
            level_val[i] = 1 - 2 * ((signs >> (trailing_ones - 1 - i)) & 1) as i32;
        }
    }
    for i in trailing_ones..total_coeff {
        {
            // level_prefix: leading zeros before a 1.
            let level_prefix = r.peek(32).leading_zeros();
            if level_prefix >= 32 {
                return Err(Error::bitstream("CAVLC: level_prefix runaway"));
            }
            r.skip(level_prefix + 1);
            let mut level_code: i32 = ((level_prefix.min(15)) << suffix_length) as i32;
            if suffix_length > 0 || level_prefix >= 14 {
                let level_suffix_size = if level_prefix == 14 && suffix_length == 0 {
                    4
                } else if level_prefix >= 15 {
                    level_prefix - 3
                } else {
                    suffix_length
                };
                if level_suffix_size > 0 {
                    level_code += r.bits(level_suffix_size) as i32;
                }
            }
            if level_prefix >= 15 && suffix_length == 0 {
                level_code += 15;
            }
            if level_prefix >= 16 {
                level_code += (1i32 << (level_prefix - 3)) - 4096;
            }
            if i == trailing_ones && trailing_ones < 3 {
                level_code += 2;
            }
            level_val[i] = if level_code % 2 == 0 { (level_code + 2) >> 1 } else { (-level_code - 1) >> 1 };
            if suffix_length == 0 {
                suffix_length = 1;
            }
            if level_val[i].unsigned_abs() > (3u32 << (suffix_length - 1)) && suffix_length < 6 {
                suffix_length += 1;
            }
        }
    }

    let mut zeros_left: usize = 0;
    if total_coeff < end_idx - start_idx + 1 {
        let t = tables();
        let e = if max_num_coeff == 4 {
            t.chroma_dc_total_zeros[total_coeff - 1][r.peek(9) as usize]
        } else if max_num_coeff == 8 {
            t.chroma422_dc_total_zeros[total_coeff - 1][r.peek(9) as usize]
        } else {
            t.total_zeros[total_coeff - 1][r.peek(9) as usize]
        };
        if e.len == 0 {
            return Err(Error::bitstream("CAVLC: invalid total_zeros"));
        }
        r.skip(e.len as u32);
        zeros_left = e.a as usize;
    }
    let mut run_val = [0usize; 16];
    for run in run_val.iter_mut().take(total_coeff.saturating_sub(1)) {
        if zeros_left > 0 {
            if zeros_left > 6 {
                // Table 9-10, zerosLeft > 6: three-bit codes 111..001 for
                // runs 0..6, then a run of `k` zeros and a one for run 4 + k.
                let bits = r.peek(11);
                let top3 = bits >> 8;
                if top3 != 0 {
                    r.skip(3);
                    *run = (7 - top3) as usize;
                } else {
                    let lz = (bits << 21).leading_zeros(); // within the 11 peeked bits
                    if lz >= 11 {
                        return Err(Error::bitstream("CAVLC: invalid run_before"));
                    }
                    r.skip(lz + 1);
                    *run = lz as usize + 4;
                }
            } else {
                let t = tables();
                let e = t.run_before[zeros_left - 1][r.peek(3) as usize];
                if e.len == 0 {
                    return Err(Error::bitstream("CAVLC: invalid run_before"));
                }
                r.skip(e.len as u32);
                *run = e.a as usize;
            }
            if *run > zeros_left {
                return Err(Error::bitstream("CAVLC: run_before exceeds zerosLeft"));
            }
            zeros_left -= *run;
        } else {
            *run = 0;
        }
    }
    run_val[total_coeff - 1] = zeros_left;
    let mut coeff_num: isize = -1;
    for i in (0..total_coeff).rev() {
        coeff_num += run_val[i] as isize + 1;
        let idx = start_idx as isize + coeff_num;
        if idx as usize > end_idx {
            return Err(Error::bitstream("CAVLC: coefficient position past the block"));
        }
        out[scan[idx as usize] as usize] = level_val[i];
    }
    if r.overrun() {
        return Err(Error::bitstream("CAVLC: slice data truncated"));
    }
    Ok(total_coeff)
}

// ---------------------------------------------------------------------------
// nC derivation (9.2.1)
// ---------------------------------------------------------------------------

/// The `TotalCoeff` a neighbouring 4x4 block of colour plane `p` (luma, or
/// a 4:4:4 chroma plane) contributes to nC: 0 for a skipped MB, 16 for
/// I_PCM, else the stored count.
fn plane_nb_count(info: &PicInfo, layer: &MbLayer, cur: usize, addr: usize, p: usize, blk: usize) -> u8 {
    if addr == cur {
        return layer.nz[p][blk];
    }
    let m = &info.mbs[addr];
    match m.kind {
        MbKind::PSkip | MbKind::BSkip => 0,
        MbKind::IPcm => 16,
        _ => info.plane_nz(p, addr, blk),
    }
}

fn chroma_nb_count(info: &PicInfo, layer: &MbLayer, cur: usize, addr: usize, comp: usize, blk: usize) -> u8 {
    if addr == cur {
        return layer.chroma_nz[comp][blk];
    }
    let m = &info.mbs[addr];
    match m.kind {
        MbKind::PSkip | MbKind::BSkip => 0,
        MbKind::IPcm => 16,
        _ => info.chroma_nz[addr * 32 + comp * 16 + blk],
    }
}

/// nC for the 4x4 block at raster `(bx, by)` of colour plane `p` (luma, or
/// Cb / Cr in 4:4:4, whose neighbours are the same plane's blocks).
pub fn plane_nc(info: &PicInfo, layer: &MbLayer, nb: &MbNeighbours, p: usize, bx: usize, by: usize) -> i32 {
    let a = nb.block(bx as i32 - 1, by as i32).map(|(addr, blk)| plane_nb_count(info, layer, nb.addr, addr, p, blk));
    let b = nb.block(bx as i32, by as i32 - 1).map(|(addr, blk)| plane_nb_count(info, layer, nb.addr, addr, p, blk));
    match (a, b) {
        (Some(a), Some(b)) => (a as i32 + b as i32 + 1) >> 1,
        (Some(a), None) => a as i32,
        (None, Some(b)) => b as i32,
        (None, None) => 0,
    }
}

/// nC for chroma AC block `(bx, by)` of component `comp`: a 2-column grid of
/// `rows` rows (2 for 4:2:0, 4 for 4:2:2).
pub fn chroma_nc(info: &PicInfo, layer: &MbLayer, nb: &MbNeighbours, comp: usize, bx: usize, by: usize, rows: usize) -> i32 {
    // Chroma block neighbours: left -> MB A's block (by*2 + 1); above -> MB
    // B's bottom-row block ((rows - 1) * 2 + bx); inside the MB otherwise.
    let a = if bx > 0 {
        Some((nb.addr, by * 2 + bx - 1))
    } else {
        nb.a.map(|addr| (addr, by * 2 + 1))
    };
    let b = if by > 0 { Some((nb.addr, (by - 1) * 2 + bx)) } else { nb.b.map(|addr| (addr, (rows - 1) * 2 + bx)) };
    let a = a.map(|(addr, blk)| chroma_nb_count(info, layer, nb.addr, addr, comp, blk));
    let b = b.map(|(addr, blk)| chroma_nb_count(info, layer, nb.addr, addr, comp, blk));
    match (a, b) {
        (Some(a), Some(b)) => (a as i32 + b as i32 + 1) >> 1,
        (Some(a), None) => a as i32,
        (None, Some(b)) => b as i32,
        (None, None) => 0,
    }
}

// ---------------------------------------------------------------------------
// Intra prediction mode derivation (8.3.1.1 / 8.3.2.1), shared with CABAC
// ---------------------------------------------------------------------------

/// The predicted Intra4x4/8x8 mode for the 4x4 block at raster `(bx, by)`
/// (for 8x8 blocks, its top-left 4x4), from neighbours A and B.
pub fn predicted_intra_mode(
    info: &PicInfo,
    layer: &MbLayer,
    nb: &MbNeighbours,
    ctx: &SliceCtx,
    bx: usize,
    by: usize,
    is_8x8: bool,
) -> u8 {
    let mode_of = |addr: usize, blk: usize, is_a: bool| -> Option<u8> {
        if addr == nb.addr {
            return Some(layer.intra_modes[blk]);
        }
        let m = &info.mbs[addr];
        if !m.kind.is_intra() && ctx.constrained_intra_pred {
            // dcPredModePredictedFlag: inter neighbour under constrained
            // intra prediction -> DC.
            return None;
        }
        match m.kind {
            MbKind::I4x4 => {
                if is_8x8 {
                    // 8.3.2.1: an I4x4 neighbour of an 8x8 block contributes
                    // the mode of the sub-block adjacent to the current
                    // block: n = 1 (A) or 2 (B) within the neighbouring 8x8.
                    let bx8 = (blk % 4) / 2 * 2;
                    let by8 = (blk / 4) / 2 * 2;
                    let (sx, sy) = if is_a { (bx8 + 1, by8) } else { (bx8, by8 + 1) };
                    Some(info.intra_modes[addr * 16 + sy * 4 + sx])
                } else {
                    Some(info.intra_modes[addr * 16 + blk])
                }
            }
            MbKind::I8x8 => Some(info.intra_modes[addr * 16 + blk]),
            _ => Some(2),
        }
    };
    let a = nb.block(bx as i32 - 1, by as i32);
    let b = nb.block(bx as i32, by as i32 - 1);
    let (Some((aa, ab)), Some((ba, bb))) = (a, b) else { return 2 };
    let ma = mode_of(aa, ab, true);
    let mb = mode_of(ba, bb, false);
    match (ma, mb) {
        (Some(x), Some(y)) => x.min(y),
        _ => 2,
    }
}

// ---------------------------------------------------------------------------
// The CAVLC macroblock layer
// ---------------------------------------------------------------------------

/// Map `mb_type` (ue) of an I slice / the intra part of P and B slices.
pub fn intra_mb_type(t: u32, layer: &mut MbLayer) -> Result<()> {
    match t {
        0 => layer.kind = MbKind::I4x4,
        1..=24 => {
            layer.kind = MbKind::I16x16;
            let t = t - 1;
            layer.intra16_mode = (t % 4) as u8;
            let chroma = ((t / 4) % 3) as u8;
            let luma = if t >= 12 { 15 } else { 0 };
            layer.cbp = luma | (chroma << 4);
        }
        25 => layer.kind = MbKind::IPcm,
        _ => return Err(Error::bitstream(format!("mb_type {t} out of range for an intra macroblock"))),
    }
    Ok(())
}

/// Map a P-slice `mb_type` (0..=4) to kind and directions.
pub fn p_mb_type(t: u32, layer: &mut MbLayer) -> Result<bool> {
    // Returns whether the mb is P_8x8ref0.
    match t {
        0 => {
            layer.kind = MbKind::Inter16x16;
            layer.pred_dir = [PRED_L0; 4];
        }
        1 => {
            layer.kind = MbKind::Inter16x8;
            layer.pred_dir = [PRED_L0; 4];
        }
        2 => {
            layer.kind = MbKind::Inter8x16;
            layer.pred_dir = [PRED_L0; 4];
        }
        3 | 4 => {
            layer.kind = MbKind::Inter8x8;
            layer.pred_dir = [PRED_L0; 4];
            return Ok(t == 4);
        }
        _ => return Err(Error::bitstream("P mb_type out of range")),
    }
    Ok(false)
}

/// B-slice `mb_type` 0..=22 → kind and per-partition directions (Table 7-14).
pub fn b_mb_type(t: u32, layer: &mut MbLayer) -> Result<()> {
    const B16X16: [u8; 3] = [PRED_L0, PRED_L1, PRED_BI];
    // Table rows 4..=21: (part0 dir, part1 dir, is16x8) for t-4.
    const B2: [(u8, u8); 9] = [
        (PRED_L0, PRED_L0),
        (PRED_L1, PRED_L1),
        (PRED_L0, PRED_L1),
        (PRED_L1, PRED_L0),
        (PRED_L0, PRED_BI),
        (PRED_L1, PRED_BI),
        (PRED_BI, PRED_L0),
        (PRED_BI, PRED_L1),
        (PRED_BI, PRED_BI),
    ];
    match t {
        0 => layer.kind = MbKind::BDirect16x16,
        1..=3 => {
            layer.kind = MbKind::Inter16x16;
            layer.pred_dir = [B16X16[(t - 1) as usize]; 4];
        }
        4..=21 => {
            let i = ((t - 4) / 2) as usize;
            let (d0, d1) = B2[i];
            if (t - 4) % 2 == 0 {
                layer.kind = MbKind::Inter16x8;
                layer.pred_dir = [d0, d0, d1, d1];
            } else {
                layer.kind = MbKind::Inter8x16;
                layer.pred_dir = [d0, d1, d0, d1];
            }
        }
        22 => layer.kind = MbKind::Inter8x8,
        _ => return Err(Error::bitstream("B mb_type out of range")),
    }
    Ok(())
}

/// P `sub_mb_type` → shape (all L0).
pub fn p_sub_mb_type(t: u32) -> Result<(SubMbShape, u8)> {
    Ok(match t {
        0 => (SubMbShape::S8x8, PRED_L0),
        1 => (SubMbShape::S8x4, PRED_L0),
        2 => (SubMbShape::S4x8, PRED_L0),
        3 => (SubMbShape::S4x4, PRED_L0),
        _ => return Err(Error::bitstream("P sub_mb_type out of range")),
    })
}

/// B `sub_mb_type` → shape and direction (Table 7-18).
pub fn b_sub_mb_type(t: u32) -> Result<(SubMbShape, u8)> {
    Ok(match t {
        0 => (SubMbShape::Direct, PRED_BI),
        1 => (SubMbShape::S8x8, PRED_L0),
        2 => (SubMbShape::S8x8, PRED_L1),
        3 => (SubMbShape::S8x8, PRED_BI),
        4 => (SubMbShape::S8x4, PRED_L0),
        5 => (SubMbShape::S4x8, PRED_L0),
        6 => (SubMbShape::S8x4, PRED_L1),
        7 => (SubMbShape::S4x8, PRED_L1),
        8 => (SubMbShape::S8x4, PRED_BI),
        9 => (SubMbShape::S4x8, PRED_BI),
        10 => (SubMbShape::S4x4, PRED_L0),
        11 => (SubMbShape::S4x4, PRED_L1),
        12 => (SubMbShape::S4x4, PRED_BI),
        _ => return Err(Error::bitstream("B sub_mb_type out of range")),
    })
}

/// The 4x4 blocks (raster) covered by 8x8 partition `part` /
/// sub-partition `sub` of `shape`, as `(x, y, w, h)` in samples.
pub fn sub_partition_rect(part: usize, shape: SubMbShape, sub: usize) -> (usize, usize, usize, usize) {
    let px = (part & 1) * 8;
    let py = (part >> 1) * 8;
    match shape {
        SubMbShape::S8x8 | SubMbShape::Direct => (px, py, 8, 8),
        SubMbShape::S8x4 => (px, py + sub * 4, 8, 4),
        SubMbShape::S4x8 => (px + sub * 4, py, 4, 8),
        SubMbShape::S4x4 => (px + (sub & 1) * 4, py + (sub >> 1) * 4, 4, 4),
    }
}

/// The partitions of a macroblock kind as `(x, y, w, h)` rectangles.
pub fn mb_partitions(kind: MbKind) -> &'static [(usize, usize, usize, usize)] {
    match kind {
        MbKind::Inter16x8 => &[(0, 0, 16, 8), (0, 8, 16, 8)],
        MbKind::Inter8x16 => &[(0, 0, 8, 16), (8, 0, 8, 16)],
        MbKind::Inter8x8 => &[(0, 0, 8, 8), (8, 0, 8, 8), (0, 8, 8, 8), (8, 8, 8, 8)],
        _ => &[(0, 0, 16, 16)],
    }
}

/// Which 8x8 partition a partition rectangle belongs to (its top-left).
#[inline]
pub fn part_index_of(x: usize, y: usize) -> usize {
    (y / 8) * 2 + x / 8
}

/// Read `mb_type` and the prediction syntax; then `coded_block_pattern`,
/// `transform_size_8x8_flag`, `mb_qp_delta` and the residual. `mb_type_raw`
/// is the already-read `mb_type` (the caller reads it because in P/B slices
/// it follows the skip run).
pub fn parse_mb_cavlc(
    r: &mut BitReader,
    ctx: &SliceCtx,
    info: &PicInfo,
    nb: &MbNeighbours,
    mb_type_raw: u32,    layer: &mut MbLayer,
) -> Result<()> {
    layer.reset(MbKind::I4x4, false);
    let mut p8x8ref0 = false;
    match ctx.slice_type {
        SliceType::I | SliceType::Si => intra_mb_type(mb_type_raw, layer)?,
        SliceType::P | SliceType::Sp => {
            if mb_type_raw < 5 {
                p8x8ref0 = p_mb_type(mb_type_raw, layer)?;
            } else {
                intra_mb_type(mb_type_raw - 5, layer)?;
            }
        }
        SliceType::B => {
            if mb_type_raw < 23 {
                b_mb_type(mb_type_raw, layer)?;
            } else {
                intra_mb_type(mb_type_raw - 23, layer)?;
            }
        }
    }

    if layer.kind == MbKind::IPcm {
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
        return Ok(());
    }

    let mut no_sub_mb_part_less_than_8x8 = true;
    if layer.kind == MbKind::Inter8x8 {
        // sub_mb_pred()
        for part in 0..4 {
            let t = r.ue();
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
                layer.ref_idx[list][part] = if p8x8ref0 || n <= 1 { 0 } else { read_ref_idx(r, n)? };
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
                    let mvd = Mv::new(r.se() as i16, r.se() as i16);
                    layer.mvd[(y / 4) * 4 + x / 4].mvd[list] = mvd;
                }
            }
        }
    } else {
        if ctx.transform_8x8_mode && layer.kind == MbKind::I4x4 {
            layer.transform_8x8 = r.flag();
            if layer.transform_8x8 {
                layer.kind = MbKind::I8x8;
            }
        }
        // mb_pred()
        match layer.kind {
            MbKind::I4x4 => {
                for blk in 0..16 {
                    let raster = super::mb::raster_of_blk(blk);
                    let (bx, by) = (raster % 4, raster / 4);
                    let pred = predicted_intra_mode(info, &layer, nb, ctx, bx, by, false);
                    let mode = if r.flag() {
                        pred
                    } else {
                        let rem = r.bits(3) as u8;
                        if rem < pred { rem } else { rem + 1 }
                    };
                    layer.intra_modes[raster] = mode;
                }
                if ctx.chroma_format_idc == 1 || ctx.chroma_format_idc == 2 {
                    layer.chroma_mode = read_chroma_mode(r)?;
                }
            }
            MbKind::I8x8 => {
                for blk8 in 0..4 {
                    let (bx, by) = ((blk8 & 1) * 2, (blk8 >> 1) * 2);
                    let pred = predicted_intra_mode(info, &layer, nb, ctx, bx, by, true);
                    let mode = if r.flag() {
                        pred
                    } else {
                        let rem = r.bits(3) as u8;
                        if rem < pred { rem } else { rem + 1 }
                    };
                    for dy in 0..2 {
                        for dx in 0..2 {
                            layer.intra_modes[(by + dy) * 4 + bx + dx] = mode;
                        }
                    }
                }
                if ctx.chroma_format_idc == 1 || ctx.chroma_format_idc == 2 {
                    layer.chroma_mode = read_chroma_mode(r)?;
                }
            }
            MbKind::I16x16 => {
                if ctx.chroma_format_idc == 1 || ctx.chroma_format_idc == 2 {
                    layer.chroma_mode = read_chroma_mode(r)?;
                }
            }
            MbKind::BDirect16x16 => {}
            _ => {
                let parts = mb_partitions(layer.kind);
                for list in 0..2 {
                    for &(x, y, _, _) in parts {
                        let part = part_index_of(x, y);
                        if layer.pred_dir[part] & (1 << list) == 0 {
                            continue;
                        }
                        let n = ctx.num_ref_idx[list];
                        let ri = if n <= 1 { 0 } else { read_ref_idx(r, n)? };
                        // The reference index applies to the whole partition.
                        for &(px, py, pw, ph) in parts.iter().filter(|p| p.0 == x && p.1 == y) {
                            for by in py / 8..(py + ph) / 8 {
                                for bx in px / 8..(px + pw) / 8 {
                                    layer.ref_idx[list][by * 2 + bx] = ri;
                                }
                            }
                        }
                    }
                }
                for list in 0..2 {
                    for &(x, y, _, _) in parts {
                        let part = part_index_of(x, y);
                        if layer.pred_dir[part] & (1 << list) == 0 {
                            continue;
                        }
                        let mvd = Mv::new(r.se() as i16, r.se() as i16);
                        layer.mvd[(y / 4) * 4 + x / 4].mvd[list] = mvd;
                    }
                }
            }
        }
    }

    if layer.kind != MbKind::I16x16 {
        let code = r.ue();
        if code > 47 {
            return Err(Error::bitstream("coded_block_pattern out of range"));
        }
        let intra = layer.kind.is_intra();
        layer.cbp = if ctx.chroma_format_idc == 1 || ctx.chroma_format_idc == 2 {
            if intra { GOLOMB_TO_INTRA4X4_CBP[code as usize] } else { GOLOMB_TO_INTER_CBP[code as usize] }
        } else {
            if code > 15 {
                return Err(Error::bitstream("coded_block_pattern out of range"));
            }
            if intra { GOLOMB_TO_INTRA4X4_CBP_GRAY[code as usize] } else { GOLOMB_TO_INTER_CBP_GRAY[code as usize] }
        };
        if layer.cbp & 15 != 0
            && ctx.transform_8x8_mode
            && !layer.kind.is_intra()
            && no_sub_mb_part_less_than_8x8
            && (layer.kind != MbKind::BDirect16x16 || ctx.direct_8x8_inference)
        {
            layer.transform_8x8 = r.flag();
        }
    }

    if layer.has_residual() {
        layer.qp_delta = r.se();
        if !(-26..=25).contains(&layer.qp_delta) {
            return Err(Error::bitstream("mb_qp_delta out of range"));
        }
        parse_residual_cavlc(r, ctx, info, nb, layer)?;
    }
    if r.overrun() {
        return Err(Error::bitstream("slice data truncated in macroblock"));
    }
    Ok(())
}

fn read_chroma_mode(r: &mut BitReader) -> Result<u8> {
    let m = r.ue();
    if m > 3 {
        return Err(Error::bitstream("intra_chroma_pred_mode out of range"));
    }
    Ok(m as u8)
}

fn read_ref_idx(r: &mut BitReader, num_ref_idx: u32) -> Result<i8> {
    let v = r.te(num_ref_idx - 1);
    if v >= num_ref_idx {
        return Err(Error::bitstream("ref_idx out of range"));
    }
    Ok(v as i8)
}

/// `residual()` for CAVLC (7.3.5.3), filling the layer's coefficient
/// arrays (raster order) and nonzero counts.
/// For CAVLC 8x8 blocks (four interleaved 4x4 scans): where scan position
/// `i` of sub-block `sub` lands in the 8x8 raster.
static SCAN8_SUB: [[u8; 16]; 4] = {
    let mut t = [[0u8; 16]; 4];
    let mut sub = 0;
    while sub < 4 {
        let mut i = 0;
        while i < 16 {
            t[sub][i] = ZIGZAG8X8[4 * i + sub];
            i += 1;
        }
        sub += 1;
    }
    t
};

/// Chroma DC scan (identity over the 2x2).
static SCAN_CHROMA_DC: [u8; 4] = [0, 1, 2, 3];

/// `residual_luma()` (7.3.5.3.1) for colour plane `p`: the luma plane, or
/// Cb / Cr in 4:4:4, which are coded exactly like it.
fn parse_residual_luma_like(r: &mut BitReader, info: &PicInfo, nb: &MbNeighbours, layer: &mut MbLayer, p: usize) -> Result<()> {
    if layer.kind == MbKind::I16x16 {
        let nc = plane_nc(info, layer, nb, p, 0, 0);
        residual_block(r, nc, &mut layer.dc[p], &ZIGZAG4X4, 0, 15, 16)?;
    }
    for blk8 in 0..4 {
        let (bx8, by8) = ((blk8 & 1) * 2, (blk8 >> 1) * 2);
        if layer.cbp & (1 << blk8) == 0 {
            continue;
        }
        if !layer.transform_8x8 {
            for sub in 0..4 {
                let (bx, by) = (bx8 + (sub & 1), by8 + (sub >> 1));
                let raster = by * 4 + bx;
                let nc = plane_nc(info, layer, nb, p, bx, by);
                let base = raster * 16;
                let n = if layer.kind == MbKind::I16x16 {
                    residual_block(r, nc, &mut layer.coef[p][base..base + 16], &ZIGZAG4X4, 1, 15, 15)?
                } else {
                    residual_block(r, nc, &mut layer.coef[p][base..base + 16], &ZIGZAG4X4, 0, 15, 16)?
                };
                layer.nz[p][raster] = n as u8;
            }
        } else {
            // 8x8 transform with CAVLC: four interleaved 4x4 blocks.
            let base = blk8 * 64;
            for sub in 0..4 {
                let (bx, by) = (bx8 + (sub & 1), by8 + (sub >> 1));
                let raster = by * 4 + bx;
                let nc = plane_nc(info, layer, nb, p, bx, by);
                let n = residual_block(r, nc, &mut layer.coef[p][base..base + 64], &SCAN8_SUB[sub], 0, 15, 16)?;
                layer.nz[p][raster] = n as u8;
            }
        }
    }
    Ok(())
}

fn parse_residual_cavlc(
    r: &mut BitReader,
    ctx: &SliceCtx,
    info: &PicInfo,
    nb: &MbNeighbours,
    layer: &mut MbLayer,
) -> Result<()> {
    // Luma, then (4:4:4) Cb and Cr coded the same way.
    parse_residual_luma_like(r, info, nb, layer, 0)?;
    if ctx.chroma_format_idc == 3 {
        parse_residual_luma_like(r, info, nb, layer, 1)?;
        parse_residual_luma_like(r, info, nb, layer, 2)?;
    }
    // Chroma (4:2:0: 2x2 blocks per component and 4 DC coefficients; 4:2:2:
    // 2x4 blocks and 8 DC coefficients).
    if (ctx.chroma_format_idc == 1 || ctx.chroma_format_idc == 2) && layer.cbp & 0x30 != 0 {
        let c422 = ctx.chroma_format_idc == 2;
        let (n_dc, rows) = if c422 { (8usize, 4usize) } else { (4, 2) };
        for comp in 0..2 {
            if c422 {
                residual_block(r, -2, &mut layer.chroma_dc[comp], &SCAN_CHROMA_DC_422, 0, 7, 8)?;
            } else {
                residual_block(r, -1, &mut layer.chroma_dc[comp][..4], &SCAN_CHROMA_DC, 0, 3, 4)?;
            }
        }
        let _ = n_dc;
        if layer.cbp & 0x20 != 0 {
            for comp in 0..2 {
                for blk in 0..2 * rows {
                    let (bx, by) = (blk & 1, blk >> 1);
                    let nc = chroma_nc(info, layer, nb, comp, bx, by, rows);
                    let n = residual_block(r, nc, &mut layer.chroma_ac[comp][blk], &ZIGZAG4X4, 1, 15, 15)?;
                    layer.chroma_nz[comp][blk] = n as u8;
                }
            }
        }
    }
    Ok(())
}
