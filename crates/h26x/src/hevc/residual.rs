//! Residual coding (H.265 7.3.8.11 / 9.3.4.2.3–7): parsing the transform
//! coefficient levels of one transform block, then scaling (8.6.3); the
//! inverse transforms live in [`crate::dsp::hevc`].

use crate::cabac::Cabac;
use crate::{Error, Result};

use super::ctx::*;
use super::tables::{DIAG_SCAN4X4_X, DIAG_SCAN4X4_Y, DIAG_SCAN8X8_X, DIAG_SCAN8X8_Y};

/// The scan of positions inside a 4x4 sub-block, and of sub-blocks inside
/// the transform block, for `scan_idx` (0 diagonal, 1 horizontal, 2 vertical).
/// Returns `(x, y)` for scan position `i` in a `size x size` grid.
#[inline]
fn scan_pos(scan_idx: u32, log2_size: u32, i: usize) -> (usize, usize) {
    match (scan_idx, log2_size) {
        (0, 0) => (0, 0),
        (0, 1) => {
            const X: [usize; 4] = [0, 0, 1, 1];
            const Y: [usize; 4] = [0, 1, 0, 1];
            (X[i], Y[i])
        }
        (0, 2) => (DIAG_SCAN4X4_X[i] as usize, DIAG_SCAN4X4_Y[i] as usize),
        (0, 3) => (DIAG_SCAN8X8_X[i] as usize, DIAG_SCAN8X8_Y[i] as usize),
        (1, l) => {
            let n = 1usize << l;
            (i % n, i / n)
        }
        (2, l) => {
            let n = 1usize << l;
            (i / n, i % n)
        }
        _ => (0, 0),
    }
}

/// `ctxIdxMap` for 4x4 significance (Table 9-50).
const CTX_IDX_MAP_4X4: [u8; 15] = [0, 1, 4, 5, 2, 3, 4, 5, 6, 6, 8, 8, 7, 7, 8];

/// Per scan: the `(xP, yP)` of the sixteen positions of a 4x4 sub-block, and
/// the position-dependent part of `sigCtx` (9.3.4.2.5) for each `prevCsbf`.
struct SubBlockTables {
    /// `[scan_idx][n] = (xP, yP)`.
    pos: [[(u8, u8); 16]; 3],
    /// `[scan_idx][prev_csbf][n]` = 0, 1 or 2.
    sig: [[[u8; 16]; 4]; 3],
    /// Inverse 4x4 scan: `[scan_idx][yP * 4 + xP]` = scan position.
    inv4: [[u8; 16]; 3],
    /// Inverse sub-block scan: `[scan_idx][log2_sb][yS * w + xS]` = index.
    inv_sb: [[[u8; 64]; 4]; 3],
}

fn sub_block_tables() -> &'static SubBlockTables {
    static T: std::sync::OnceLock<SubBlockTables> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        let mut t = SubBlockTables { pos: [[(0, 0); 16]; 3], sig: [[[0; 16]; 4]; 3], inv4: [[0; 16]; 3], inv_sb: [[[0; 64]; 4]; 3] };
        for scan in 0..3 {
            for log2_sb in 0..4 {
                let w = 1usize << log2_sb;
                for i in 0..w * w {
                    let (x, y) = scan_pos(scan as u32, log2_sb as u32, i);
                    t.inv_sb[scan][log2_sb][y * w + x] = i as u8;
                }
            }
            for n in 0..16 {
                let (xp, yp) = scan_pos(scan as u32, 2, n);
                t.inv4[scan][yp * 4 + xp] = n as u8;
            }
            for n in 0..16 {
                let (xp, yp) = scan_pos(scan as u32, 2, n);
                t.pos[scan][n] = (xp as u8, yp as u8);
                for prev in 0..4 {
                    t.sig[scan][prev][n] = match prev {
                        0 => {
                            if xp + yp == 0 {
                                2
                            } else if xp + yp < 3 {
                                1
                            } else {
                                0
                            }
                        }
                        1 => {
                            if yp == 0 {
                                2
                            } else if yp == 1 {
                                1
                            } else {
                                0
                            }
                        }
                        2 => {
                            if xp == 0 {
                                2
                            } else if xp == 1 {
                                1
                            } else {
                                0
                            }
                        }
                        _ => 2,
                    };
                }
            }
        }
        t
    })
}

/// What the transform block needs from its surroundings.
pub struct ResidualParams {
    /// `log2TrafoSize`.
    pub log2_size: u32,
    /// Colour component 0..=2.
    pub c_idx: usize,
    /// `scanIdx` (7.4.9.11).
    pub scan_idx: u32,
    /// `cu_transquant_bypass_flag`.
    pub bypass: bool,
    /// `transform_skip_enabled_flag` (and the size allows it: log2 <= 2).
    pub transform_skip_allowed: bool,
    /// `sign_data_hiding_enabled_flag`.
    pub sign_hiding: bool,
    /// The CU is intra (`CuPredMode == MODE_INTRA`).
    pub intra: bool,
    /// The intra prediction mode of the block's component (for implicit RDPCM).
    pub pred_mode_intra: u32,
    /// `transform_skip_context_enabled_flag`.
    pub ts_context: bool,
    /// `implicit_rdpcm_enabled_flag`.
    pub implicit_rdpcm: bool,
    /// `explicit_rdpcm_enabled_flag`.
    pub explicit_rdpcm: bool,
    /// `persistent_rice_adaptation_enabled_flag`.
    pub persistent_rice: bool,
    /// Print the parse (debugging).
    pub trace: bool,
}

/// What [`parse_residual`] found.
#[derive(Debug, Clone, Copy)]
pub struct ResidualInfo {
    /// `transform_skip_flag`.
    pub transform_skip: bool,
    /// Largest column with a nonzero coefficient.
    pub max_x: usize,
    /// Largest row with a nonzero coefficient.
    pub max_y: usize,
    /// Residual DPCM to apply (range extension): `Some(vertical)`.
    pub rdpcm: Option<bool>,
}

/// Parse `residual_coding()` for one transform block into `coeffs`
/// (raster order, `1 << (2 * log2_size)` entries used, all set — zeros
/// included).
pub fn parse_residual(cabac: &mut Cabac, cx: &mut Contexts, p: &ResidualParams, coeffs: &mut [i16]) -> Result<ResidualInfo> {
    let log2 = p.log2_size;
    let n = 1usize << log2;
    coeffs[..n * n].fill(0);
    let c_idx = p.c_idx;
    let mut max_x = 0usize;
    let mut max_y = 0usize;

    let mut transform_skip = false;
    if p.transform_skip_allowed && !p.bypass {
        transform_skip = cabac.decision(&mut cx.c[TRANSFORM_SKIP_FLAG_OFFSET + (c_idx > 0) as usize]) != 0;
    }
    let ts_or_bypass = transform_skip || p.bypass;
    // Residual DPCM: explicit for inter blocks, implicit for intra blocks
    // predicted purely horizontally / vertically (range extension).
    let mut rdpcm = None;
    if !p.intra && p.explicit_rdpcm && ts_or_bypass {
        let cinc = (c_idx > 0) as usize;
        if cabac.decision(&mut cx.c[EXPLICIT_RDPCM_FLAG_OFFSET + cinc]) != 0 {
            rdpcm = Some(cabac.decision(&mut cx.c[EXPLICIT_RDPCM_DIR_FLAG_OFFSET + cinc]) != 0);
        }
    }
    let implicit = p.intra && p.implicit_rdpcm && ts_or_bypass && (p.pred_mode_intra == 10 || p.pred_mode_intra == 26);
    if implicit {
        rdpcm = Some(p.pred_mode_intra == 26);
    }
    // One significance context for transform-skipped / bypassed blocks.
    let ts_sig_ctx: Option<usize> = if p.ts_context && ts_or_bypass { Some(if c_idx == 0 { 42 } else { 27 + 16 }) } else { None };
    // Persistent Rice adaptation: the sub-block type's running statistic.
    let sb_type = (c_idx == 0) as usize * 2 + ts_or_bypass as usize;

    // last_sig_coeff_{x,y}_prefix / suffix.
    let (ctx_offset, ctx_shift) = if c_idx == 0 {
        (3 * (log2 - 2) + ((log2 - 1) >> 2), (log2 + 1) >> 2)
    } else {
        (15, log2 - 2)
    };
    let c_max = (log2 << 1) - 1;
    let mut last_x_prefix = 0u32;
    while last_x_prefix < c_max
        && cabac.decision(&mut cx.c[LAST_SIGNIFICANT_COEFF_X_PREFIX_OFFSET + (ctx_offset + (last_x_prefix >> ctx_shift)) as usize]) != 0
    {
        last_x_prefix += 1;
    }
    let mut last_y_prefix = 0u32;
    while last_y_prefix < c_max
        && cabac.decision(&mut cx.c[LAST_SIGNIFICANT_COEFF_Y_PREFIX_OFFSET + (ctx_offset + (last_y_prefix >> ctx_shift)) as usize]) != 0
    {
        last_y_prefix += 1;
    }
    let mut last_x = last_x_prefix;
    if last_x_prefix > 3 {
        let nb = (last_x_prefix >> 1) - 1;
        let suffix = cabac.bypass_bits(nb);
        last_x = (1 << nb) * (2 + (last_x_prefix & 1)) + suffix;
    }
    let mut last_y = last_y_prefix;
    if last_y_prefix > 3 {
        let nb = (last_y_prefix >> 1) - 1;
        let suffix = cabac.bypass_bits(nb);
        last_y = (1 << nb) * (2 + (last_y_prefix & 1)) + suffix;
    }
    if p.scan_idx == 2 {
        std::mem::swap(&mut last_x, &mut last_y);
    }
    if p.trace {
        eprintln!("  residual: log2={log2} c={c_idx} scan={} ts={transform_skip} last=({last_x},{last_y}) cabac_pos={}", p.scan_idx, cabac.position());
    }
    if last_x as usize >= n || last_y as usize >= n {
        return Err(Error::bitstream("last significant coefficient outside the block"));
    }

    // Locate the last sub-block and position within it: the inverse scans.
    let log2_sb = log2 - 2; // sub-block grid is (1 << log2_sb) squared
    let tabs = sub_block_tables();
    let last_sub_block = tabs.inv_sb[p.scan_idx as usize][log2_sb as usize][((last_y as usize >> 2) << log2_sb) + (last_x as usize >> 2)] as usize;
    let last_scan_pos = tabs.inv4[p.scan_idx as usize][((last_y as usize & 3) << 2) + (last_x as usize & 3)] as usize;

    // coded_sub_block_flag storage: (1 << log2_sb) squared, raster.
    let sb_w = 1usize << log2_sb;
    let mut csbf = [0u8; 64];
    let mut greater1_ctx_state: u32 = 1; // greater1Ctx carried across sub-blocks
    let mut first_sb_processed = true;

    for i in (0..=last_sub_block).rev() {
        let (xs, ys) = scan_pos(p.scan_idx, log2_sb, i);
        let mut infer_sb_dc_sig = false;
        let coded: bool;
        if i < last_sub_block && i > 0 {
            let mut csbf_ctx = 0u32;
            if xs < sb_w - 1 {
                csbf_ctx += csbf[ys * sb_w + xs + 1] as u32;
            }
            if ys < sb_w - 1 {
                csbf_ctx += csbf[(ys + 1) * sb_w + xs] as u32;
            }
            let inc = if c_idx == 0 { csbf_ctx.min(1) } else { 2 + csbf_ctx.min(1) };
            coded = cabac.decision(&mut cx.c[SIGNIFICANT_COEFF_GROUP_FLAG_OFFSET + inc as usize]) != 0;
            infer_sb_dc_sig = true;
        } else {
            coded = true; // the first and last sub-blocks are inferred coded
        }
        csbf[ys * sb_w + xs] = coded as u8;

        // Significance flags of this sub-block, in reverse scan; the
        // significant positions are kept as a list (reverse scan order).
        let mut sig_pos = [0u8; 16];
        let mut n_sig = 0usize;
        let start_n = if i == last_sub_block { last_scan_pos as i32 - 1 } else { 15 };
        if i == last_sub_block {
            sig_pos[0] = last_scan_pos as u8;
            n_sig = 1;
        }
        // prevCsbf for sigCtx.
        let mut prev_csbf = 0usize;
        if xs < sb_w - 1 {
            prev_csbf += csbf[ys * sb_w + xs + 1] as usize;
        }
        if ys < sb_w - 1 {
            prev_csbf += (csbf[(ys + 1) * sb_w + xs] as usize) << 1;
        }
        let pos_tab = &tabs.pos[p.scan_idx as usize];
        let sig_tab = &tabs.sig[p.scan_idx as usize][prev_csbf];
        // The sub-block-level part of sigCtx (everything but the position).
        let sig_base: u32 = if log2 == 2 {
            0
        } else if c_idx == 0 {
            (if xs + ys > 0 { 3 } else { 0 }) + if log2 == 3 { if p.scan_idx == 0 { 9 } else { 15 } } else { 21 }
        } else if log2 == 3 {
            9
        } else {
            12
        };
        let sig_ctx_off = SIGNIFICANT_COEFF_FLAG_OFFSET + if c_idx == 0 { 0 } else { 27 };
        if coded {
            let mut nn = start_n;
            while nn >= 0 {
                let npos = nn as usize;
                if npos > 0 || !infer_sb_dc_sig {
                    let sig_ctx: u32 = if let Some(c) = ts_sig_ctx {
                        c as u32 - if c_idx == 0 { 0 } else { 27 }
                    } else if log2 == 2 {
                        let (xp, yp) = pos_tab[npos];
                        CTX_IDX_MAP_4X4[((yp as usize) << 2) + xp as usize] as u32
                    } else if xs + ys == 0 && npos == 0 {
                        // (xC, yC) == (0, 0): the DC of the block.
                        0
                    } else {
                        sig_base + sig_tab[npos] as u32
                    };
                    let f = cabac.decision(&mut cx.c[sig_ctx_off + sig_ctx as usize]) != 0;
                    if f {
                        sig_pos[n_sig] = npos as u8;
                        n_sig += 1;
                        infer_sb_dc_sig = false;
                    }
                } else {
                    // DC of a coded sub-block with no other significant coefficient: inferred 1.
                    sig_pos[n_sig] = 0;
                    n_sig += 1;
                }
                nn -= 1;
            }
        }
        if p.trace {
            eprintln!("  sb {i} ({xs},{ys}) coded={coded} sig={:?} n_sig={n_sig}", &sig_pos[..n_sig]);
        }
        if n_sig == 0 {
            continue;
        }
        let sig_pos = &sig_pos[..n_sig];

        // Levels: greater1 (up to 8), greater2 (one), signs, remaining.
        let mut ctx_set: u32 = if i == 0 || c_idx > 0 { 0 } else { 2 };
        // 9.3.4.2.6: lastGreater1Ctx is the previous sub-block's greater1Ctx
        // after its final update (a flag of 1 zeroes it, else it grows); a
        // zero bumps the context set.
        if !first_sb_processed && greater1_ctx_state == 0 {
            ctx_set += 1;
        }
        first_sb_processed = false;
        let mut greater1_ctx: u32 = 1;
        let mut abs_level = [1i32; 16]; // indexed like sig_pos
        let mut last_greater1_idx: i32 = -1;
        let g1_ctx_base = COEFF_ABS_LEVEL_GREATER1_FLAG_OFFSET + (ctx_set * 4) as usize + if c_idx > 0 { 16 } else { 0 };
        for k in 0..n_sig.min(8) {
            let g1 = cabac.decision(&mut cx.c[g1_ctx_base + greater1_ctx.min(3) as usize]) != 0;
            if greater1_ctx > 0 {
                greater1_ctx = if g1 { 0 } else { greater1_ctx + 1 };
            }
            if g1 {
                abs_level[k] = 2;
                if last_greater1_idx == -1 {
                    last_greater1_idx = k as i32;
                }
            }
        }
        greater1_ctx_state = greater1_ctx;
        let first_sig_scan_pos = sig_pos[n_sig - 1] as i32;
        let last_sig_scan_pos = sig_pos[0] as i32;
        let sign_hidden = !p.bypass && p.sign_hiding && rdpcm.is_none() && (last_sig_scan_pos - first_sig_scan_pos > 3);
        if last_greater1_idx != -1 {
            let inc = ctx_set as usize + if c_idx > 0 { 4 } else { 0 };
            let g2 = cabac.decision(&mut cx.c[COEFF_ABS_LEVEL_GREATER2_FLAG_OFFSET + inc]) != 0;
            if g2 {
                abs_level[last_greater1_idx as usize] = 3;
            }
        }
        // Signs: one bypass read for all of them (MSB = first in reverse
        // scan); the hidden sign, if any, is the last position's.
        let n_signs = n_sig - sign_hidden as usize;
        let signs = cabac.bypass_bits(n_signs as u32);
        // Remaining levels.
        let mut sum_abs = 0i32;
        let mut c_last_abs: i32 = 0;
        let mut c_last_rice: u32 = 0;
        let mut first_remaining = true;
        let rice_init: u32 = if p.persistent_rice { (cx.stat_coeff[sb_type] / 4) as u32 } else { 0 };
        for k in 0..n_sig {
            let base_level = abs_level[k];
            let threshold = if k < 8 {
                if k as i32 == last_greater1_idx {
                    3
                } else {
                    2
                }
            } else {
                1
            };
            let mut level = base_level;
            if base_level == threshold {
                let rice = if first_remaining {
                    rice_init
                } else {
                    let up = c_last_rice + (c_last_abs > 3 * (1 << c_last_rice)) as u32;
                    if p.persistent_rice { up } else { up.min(4) }
                };
                let rem = decode_abs_level_remaining(cabac, rice)?;
                if first_remaining && p.persistent_rice {
                    // StatCoeff update on the sub-block's first remaining level.
                    let st = &mut cx.stat_coeff[sb_type];
                    if rem >= (3 << (*st / 4)) {
                        *st += 1;
                    } else if 2 * rem < (1 << (*st / 4)) && *st > 0 {
                        *st -= 1;
                    }
                }
                first_remaining = false;
                level = base_level + rem;
                c_last_abs = level;
                c_last_rice = rice;
            }
            let neg = k < n_signs && (signs >> (n_signs - 1 - k)) & 1 != 0;
            let mut v = if neg { -level } else { level };
            if sign_hidden {
                sum_abs += level;
                if k == n_sig - 1 && sum_abs % 2 == 1 {
                    v = -v;
                }
            }
            let (xp, yp) = pos_tab[sig_pos[k] as usize];
            let xc = (xs << 2) + xp as usize;
            let yc = (ys << 2) + yp as usize;
            coeffs[yc * n + xc] = v.clamp(-32768, 32767) as i16;
            max_x = max_x.max(xc);
            max_y = max_y.max(yc);
            if p.trace {
                eprintln!("    n={} ({xc},{yc}) base={base_level} level={level} v={v} sign_hidden={sign_hidden}", sig_pos[k]);
            }
        }
    }
    if cabac.overrun() {
        return Err(Error::bitstream("slice data exhausted in residual coding"));
    }
    Ok(ResidualInfo { transform_skip, max_x, max_y, rdpcm })
}

/// Rotate a 4x4 residual by 180 degrees (`transform_skip_rotation_enabled_flag`).
pub fn rotate_residual4(coeffs: &mut [i16]) {
    coeffs[..16].reverse();
}

/// Residual DPCM (8.6.6 / 8.6.8): accumulate the residual along rows
/// (horizontal) or down columns (`vertical`).
pub fn rdpcm_residual(coeffs: &mut [i16], log2: u32, vertical: bool) {
    let n = 1usize << log2;
    if vertical {
        for y in 1..n {
            for x in 0..n {
                coeffs[y * n + x] = coeffs[y * n + x].wrapping_add(coeffs[(y - 1) * n + x]);
            }
        }
    } else {
        for y in 0..n {
            for x in 1..n {
                coeffs[y * n + x] = coeffs[y * n + x].wrapping_add(coeffs[y * n + x - 1]);
            }
        }
    }
}

/// `coeff_abs_level_remaining` (9.3.3.11 without extended precision).
#[inline]
fn decode_abs_level_remaining(cabac: &mut Cabac, rice: u32) -> Result<i32> {
    let mut prefix = 0u32;
    while prefix < 4 && cabac.bypass() != 0 {
        prefix += 1;
    }
    if prefix < 4 {
        let suffix = cabac.bypass_bits(rice);
        return Ok(((prefix << rice) + suffix) as i32);
    }
    // EGk with k = rice + 1 for (value - (4 << rice)).
    let mut k = rice + 1;
    let mut v: i32 = 0;
    loop {
        if cabac.bypass() != 0 {
            v += 1 << k;
            k += 1;
            if k > 30 {
                return Err(Error::bitstream("coeff_abs_level_remaining runaway"));
            }
        } else {
            break;
        }
    }
    v += cabac.bypass_bits(k) as i32;
    Ok((4 << rice) as i32 + v)
}

/// The scaling factor `m[x][y]` source for a transform block.
pub enum ScalingSource<'a> {
    /// Flat 16.
    Flat,
    /// A scaling list at the block's size: 16 (4x4) or 64 (8x8 grid, replicated
    /// for 16x16 / 32x32) raster values, plus the DC for 16/32.
    List(&'a [u8], u8),
}

/// Scale (dequantise) the coefficients of one transform block in place
/// (8.6.3): `d = Clip3(coeffMin, coeffMax, ((c * m * levelScale[qP % 6] << (qP / 6)) + (1 << (bdShift - 1))) >> bdShift)`
/// with `bdShift = BitDepth + Log2(nTbS) - 5`, `m = 16` when scaling lists are off
/// (or the block is transform-skipped and larger than 4x4), else the list.
/// Only the `0..=max_x` × `0..=max_y` region can hold nonzero coefficients.
#[allow(clippy::too_many_arguments)]
pub fn scale_coefficients(
    coeffs: &mut [i16],
    log2: u32,
    qp: i32,
    bit_depth: u32,
    scaling: ScalingSource,
    transform_skip: bool,
    max_x: usize,
    max_y: usize,
) {
    const LEVEL_SCALE: [i32; 6] = [40, 45, 51, 57, 64, 72];
    let n = 1usize << log2;
    let bd_shift = bit_depth as i32 + log2 as i32 - 5;
    let round = 1i64 << (bd_shift - 1);
    let ls = LEVEL_SCALE[(qp % 6) as usize] as i64;
    let q6 = qp / 6;
    let flat = matches!(scaling, ScalingSource::Flat) || (transform_skip && n > 4);
    // Flat scaling with a factor that keeps `c * factor` in i32: a branchless
    // row loop the compiler vectorises (zeros scale to zero: round < 2^bdShift).
    let factor = (16 * ls) << q6;
    if flat && factor < 65536 {
        let factor = factor as i32;
        let round = round as i32;
        for y in 0..=max_y {
            let row = &mut coeffs[y * n..y * n + max_x + 1];
            for c in row.iter_mut() {
                *c = ((*c as i32 * factor + round) >> bd_shift).clamp(-32768, 32767) as i16;
            }
        }
        return;
    }
    for y in 0..=max_y {
        for x in 0..=max_x {
            let c = coeffs[y * n + x];
            if c == 0 {
                continue;
            }
            let m: i64 = if flat {
                16
            } else {
                match &scaling {
                    ScalingSource::List(list, dc) => {
                        if n == 4 {
                            list[y * 4 + x] as i64
                        } else if n == 8 {
                            list[y * 8 + x] as i64
                        } else if x == 0 && y == 0 {
                            *dc as i64
                        } else {
                            let r = n / 8;
                            list[(y / r) * 8 + x / r] as i64
                        }
                    }
                    ScalingSource::Flat => 16,
                }
            };
            let v = ((c as i64 * m * ls) << q6) + round;
            coeffs[y * n + x] = (v >> bd_shift).clamp(-32768, 32767) as i16;
        }
    }
}

/// Transform-skip residual (8.6.4.2 with `transform_skip_flag`): `r = (d << tsShift + round) >> bdShift`.
pub fn transform_skip_residual(coeffs: &mut [i16], log2: u32, bit_depth: u32) {
    let n = 1usize << log2;
    let ts_shift = 5 + log2 as i32;
    let bd_shift = 20 - bit_depth as i32;
    let round = 1i32 << (bd_shift - 1);
    for v in coeffs.iter_mut().take(n * n) {
        *v = ((((*v as i32) << ts_shift) + round) >> bd_shift).clamp(-32768, 32767) as i16;
    }
}
