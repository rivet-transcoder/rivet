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
    // (explicit rdpcm: range extension only.)

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

    // Locate the last sub-block and position within it.
    let log2_sb = log2 - 2; // sub-block grid is (1 << log2_sb) squared
    let num_sb = 1usize << (2 * log2_sb);
    let mut last_sub_block = num_sb - 1;
    let mut last_scan_pos = 16usize;
    loop {
        if last_scan_pos == 0 {
            last_scan_pos = 16;
            if last_sub_block == 0 {
                return Err(Error::bitstream("last significant coefficient not found in scan"));
            }
            last_sub_block -= 1;
        }
        last_scan_pos -= 1;
        let (xs, ys) = scan_pos(p.scan_idx, log2_sb, last_sub_block);
        let (xp, yp) = scan_pos(p.scan_idx, 2, last_scan_pos);
        let xc = (xs << 2) + xp;
        let yc = (ys << 2) + yp;
        if xc == last_x as usize && yc == last_y as usize {
            break;
        }
    }

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

        // Significance flags of this sub-block, in reverse scan.
        let mut sig = [false; 16];
        let mut n_sig = 0usize;
        let start_n = if i == last_sub_block { last_scan_pos as i32 - 1 } else { 15 };
        if i == last_sub_block {
            sig[last_scan_pos] = true;
            n_sig = 1;
        }
        // prevCsbf for sigCtx.
        let mut prev_csbf = 0u32;
        if xs < sb_w - 1 {
            prev_csbf += csbf[ys * sb_w + xs + 1] as u32;
        }
        if ys < sb_w - 1 {
            prev_csbf += (csbf[(ys + 1) * sb_w + xs] as u32) << 1;
        }
        let mut nn = start_n;
        while nn >= 0 {
            let npos = nn as usize;
            let (xp, yp) = scan_pos(p.scan_idx, 2, npos);
            let xc = (xs << 2) + xp;
            let yc = (ys << 2) + yp;
            if coded && (npos > 0 || !infer_sb_dc_sig) {
                // sigCtx (9.3.4.2.5).
                let sig_ctx: u32 = if log2 == 2 {
                    CTX_IDX_MAP_4X4[(yc << 2) + xc] as u32
                } else if xc + yc == 0 {
                    0
                } else {
                    let mut s = match prev_csbf {
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
                    if c_idx == 0 {
                        if xs + ys > 0 {
                            s += 3;
                        }
                        if log2 == 3 {
                            s += if p.scan_idx == 0 { 9 } else { 15 };
                        } else {
                            s += 21;
                        }
                    } else if log2 == 3 {
                        s += 9;
                    } else {
                        s += 12;
                    }
                    s
                };
                let inc = if c_idx == 0 { sig_ctx } else { 27 + sig_ctx };
                let f = cabac.decision(&mut cx.c[SIGNIFICANT_COEFF_FLAG_OFFSET + inc as usize]) != 0;
                if f {
                    sig[npos] = true;
                    n_sig += 1;
                    infer_sb_dc_sig = false;
                }
            } else if coded && npos == 0 && infer_sb_dc_sig {
                // DC of a coded sub-block with no other significant coefficient: inferred 1.
                sig[0] = true;
                n_sig += 1;
            }
            nn -= 1;
        }
        if p.trace {
            eprintln!("  sb {i} ({xs},{ys}) coded={coded} sig={:?} n_sig={n_sig}", (0..16).filter(|&k| sig[k]).collect::<Vec<_>>());
        }
        if n_sig == 0 {
            continue;
        }

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
        let mut abs_level = [0i32; 16];
        let mut greater1_flags = [false; 16];
        let mut num_greater1 = 0usize;
        let mut last_greater1_scan_pos: i32 = -1;
        let mut first_sig_scan_pos = 16usize;
        let mut last_sig_scan_pos: i32 = -1;
        let mut last_g1_flag_in_sb = false;
        let mut escape_data_present = false;
        for npos in (0..16).rev() {
            if !sig[npos] {
                continue;
            }
            abs_level[npos] = 1;
            if num_greater1 < 8 {
                let inc = (ctx_set * 4 + greater1_ctx.min(3)) as usize + if c_idx > 0 { 16 } else { 0 };
                let g1 = cabac.decision(&mut cx.c[COEFF_ABS_LEVEL_GREATER1_FLAG_OFFSET + inc]) != 0;
                num_greater1 += 1;
                greater1_flags[npos] = g1;
                last_g1_flag_in_sb = g1;
                if greater1_ctx > 0 {
                    greater1_ctx = if g1 { 0 } else { greater1_ctx + 1 };
                }
                if g1 {
                    abs_level[npos] = 2;
                    if last_greater1_scan_pos == -1 {
                        last_greater1_scan_pos = npos as i32;
                    } else {
                        escape_data_present = true;
                    }
                }
            } else {
                escape_data_present = true;
            }
            if last_sig_scan_pos == -1 {
                last_sig_scan_pos = npos as i32;
            }
            first_sig_scan_pos = npos;
        }
        let _ = (escape_data_present, last_g1_flag_in_sb);
        greater1_ctx_state = greater1_ctx;

        let sign_hidden = !p.bypass && p.sign_hiding && (last_sig_scan_pos - first_sig_scan_pos as i32 > 3);
        if last_greater1_scan_pos != -1 {
            let inc = ctx_set as usize + if c_idx > 0 { 4 } else { 0 };
            let g2 = cabac.decision(&mut cx.c[COEFF_ABS_LEVEL_GREATER2_FLAG_OFFSET + inc]) != 0;
            if g2 {
                abs_level[last_greater1_scan_pos as usize] = 3;
            }
        }
        // Signs.
        let mut signs = [false; 16];
        for npos in (0..16).rev() {
            if sig[npos] && (!sign_hidden || npos != first_sig_scan_pos) {
                signs[npos] = cabac.bypass() != 0;
            }
        }
        // Remaining levels.
        let mut num_sig_coeff = 0usize;
        let mut sum_abs = 0i32;
        let mut c_last_abs: i32 = 0;
        let mut c_last_rice: u32 = 0;
        let mut first_remaining = true;
        for npos in (0..16).rev() {
            if !sig[npos] {
                continue;
            }
            let base_level = abs_level[npos];
            let threshold = if num_sig_coeff < 8 {
                if npos as i32 == last_greater1_scan_pos {
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
                    0
                } else {
                    (c_last_rice + (c_last_abs > 3 * (1 << c_last_rice)) as u32).min(4)
                };
                first_remaining = false;
                let rem = decode_abs_level_remaining(cabac, rice)?;
                level = base_level + rem;
                c_last_abs = level;
                c_last_rice = rice;
            }
            let mut v = if signs[npos] { -level } else { level };
            if sign_hidden {
                sum_abs += level;
                if npos == first_sig_scan_pos && sum_abs % 2 == 1 {
                    v = -v;
                }
            }
            let (xp, yp) = scan_pos(p.scan_idx, 2, npos);
            let xc = (xs << 2) + xp;
            let yc = (ys << 2) + yp;
            coeffs[yc * n + xc] = v.clamp(-32768, 32767) as i16;
            max_x = max_x.max(xc);
            max_y = max_y.max(yc);
            num_sig_coeff += 1;
            if p.trace {
                eprintln!("    n={npos} ({xc},{yc}) base={base_level} level={level} v={v} sign_hidden={sign_hidden}");
            }
        }
    }
    if cabac.overrun() {
        return Err(Error::bitstream("slice data exhausted in residual coding"));
    }
    Ok(ResidualInfo { transform_skip, max_x, max_y })
}

/// `coeff_abs_level_remaining` (9.3.3.11 without the range extension).
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
