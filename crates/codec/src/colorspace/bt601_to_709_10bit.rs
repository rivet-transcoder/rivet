// =============================================================================
// 10-bit BT.601 → BT.709 (Squad-29, follow-up to Squad-19's 10-bit pipeline).
// =============================================================================
//
// 10-bit limited-range constants (Rec. ITU-R BT.2100-2 Table 9 /
// the standard "limited-range 10-bit" of BT.709/BT.2020):
//   luma center  = 64   (16 << 2)
//   chroma center = 512 (128 << 2)
//   luma clamp   = [64, 940]   (16<<2 .. 235<<2)
//   chroma clamp = [64, 960]   (16<<2 .. 240<<2)
//
// The matrix coefficients are identical to the 8-bit case — they're
// derived from the BT.601 / BT.709 spec ratios (Kr / Kg / Kb) and
// don't depend on bit depth. Only the offsets and clamp range change.
// Constants live in the parent module (super::Q15, super::M_Y_CB, etc.)
// and are shared with the 8-bit variant.
//
// Use case: rare. The 10-bit pipeline is HDR-passthrough by default
// (Squad-19) — for HDR sources (BT.2020 + PQ/HLG) we never convert
// because the matrix shift would corrupt the wide gamut. This 10-bit
// BT.601→BT.709 path exists for explicitly-tagged BT.601 10-bit
// content (some Sony broadcast cameras output 10-bit BT.601). Wired
// behind a public entry point but not invoked from the default
// pipeline; callers must opt in explicitly.

#[inline(always)]
fn clamp_y_10bit(v: i32) -> u16 {
    v.clamp(64, 940) as u16
}

#[inline(always)]
fn clamp_c_10bit(v: i32) -> u16 {
    v.clamp(64, 960) as u16
}

const CHROMA_CENTER_10BIT: i32 = 512;

/// Scalar 10-bit BT.601 → BT.709 reference. Same algorithm as the
/// 8-bit `bt601_to_bt709_scalar`, but operates on `u16` planes
/// (10-bit values in 0..=1023). `width` / `height` are luma
/// dimensions; chroma planes are half-resolution per axis (4:2:0).
pub fn bt601_to_bt709_planes_10bit_scalar(
    y: &mut [u16],
    cb: &mut [u16],
    cr: &mut [u16],
    width: usize,
    height: usize,
) {
    debug_assert_eq!(y.len(), width * height);
    debug_assert_eq!(cb.len(), (width / 2) * (height / 2));
    debug_assert_eq!(cr.len(), (width / 2) * (height / 2));

    let cw = width / 2;

    // Luma: Y709 = Y601 + M_Y_CB * ΔCb + M_Y_CR * ΔCr.
    for yi in 0..height {
        let cy = yi >> 1;
        for xi in 0..width {
            let cx = xi >> 1;
            let cbl = cb[cy * cw + cx] as i32 - CHROMA_CENTER_10BIT;
            let crl = cr[cy * cw + cx] as i32 - CHROMA_CENTER_10BIT;
            let y_orig = y[yi * width + xi] as i32;
            let delta =
                (super::M_Y_CB * cbl + super::M_Y_CR * crl + super::Q15_ROUND) >> super::Q15;
            y[yi * width + xi] = clamp_y_10bit(y_orig + delta);
        }
    }

    // Chroma: pure 2×2 chroma → chroma transform (no luma coupling).
    for v in cb.iter_mut().zip(cr.iter_mut()) {
        let (cbp, crp) = v;
        let cbl = *cbp as i32 - CHROMA_CENTER_10BIT;
        let crl = *crp as i32 - CHROMA_CENTER_10BIT;
        let new_cb =
            (super::M_CB_CB * cbl + super::M_CB_CR * crl + super::Q15_ROUND) >> super::Q15;
        let new_cr =
            (super::M_CR_CB * cbl + super::M_CR_CR * crl + super::Q15_ROUND) >> super::Q15;
        *cbp = clamp_c_10bit(new_cb + CHROMA_CENTER_10BIT);
        *crp = clamp_c_10bit(new_cr + CHROMA_CENTER_10BIT);
    }
}

/// Runtime-dispatched 10-bit BT.601 → BT.709. AVX2 on x86_64 when
/// available, scalar fallback otherwise. Squad-29.
pub fn bt601_to_bt709_planes_10bit(
    y: &mut [u16],
    cb: &mut [u16],
    cr: &mut [u16],
    width: usize,
    height: usize,
) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: avx2 feature was runtime-detected above.
            unsafe {
                bt601_to_bt709_10bit_avx2(y, cb, cr, width, height);
            }
            return;
        }
    }
    bt601_to_bt709_planes_10bit_scalar(y, cb, cr, width, height);
}

/// AVX2 specialization for the 10-bit BT.601 → BT.709 matrix
/// conversion. Mirrors `bt601_to_bt709_avx2` on `u16` lanes
/// (16 chroma samples per 256-bit register vs 16 in the 8-bit
/// path — same lane count because 8-bit path already widened
/// u8 → i16 inside the kernel).
///
/// Q15 fixed-point math via `_mm256_mulhrs_epi16` ((a*b + 0x4000) >> 15).
/// Chroma deltas are in [-512, 511] (10-bit center 512), so values
/// fit in i16 with room to spare. Coefficients ≥ 1 (M_CB_CB=33379,
/// M_CR_CR=33598) split off the identity contribution as the 8-bit
/// path does.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn bt601_to_bt709_10bit_avx2(
    y: &mut [u16],
    cb: &mut [u16],
    cr: &mut [u16],
    width: usize,
    height: usize,
) {
    unsafe {
        #[cfg(target_arch = "x86")]
        use std::arch::x86::*;
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::*;

        let cw = width / 2;
        let ch = height / 2;

        let v_m_y_cb = _mm256_set1_epi16(super::M_Y_CB as i16);
        let v_m_y_cr = _mm256_set1_epi16(super::M_Y_CR as i16);
        let v_m_cb_cb_corr = _mm256_set1_epi16((super::M_CB_CB - 32768) as i16);
        let v_m_cb_cr = _mm256_set1_epi16(super::M_CB_CR as i16);
        let v_m_cr_cb = _mm256_set1_epi16(super::M_CR_CB as i16);
        let v_m_cr_cr_corr = _mm256_set1_epi16((super::M_CR_CR - 32768) as i16);

        let v_chroma_center = _mm256_set1_epi16(CHROMA_CENTER_10BIT as i16);
        let v_chroma_lo = _mm256_set1_epi16(64);
        let v_chroma_hi = _mm256_set1_epi16(960);
        let v_luma_lo = _mm256_set1_epi16(64);
        let v_luma_hi = _mm256_set1_epi16(940);

        // ---- Luma pass ----
        // 16 chroma samples per iter → 32 luma cols on two rows (64 luma
        // outputs per iter). Same expand pattern as the 8-bit kernel:
        // duplicate each chroma delta into two adjacent luma slots
        // because chroma is 4:2:0 (one chroma per 2×2 luma block).
        for cy_idx in 0..ch {
            let y_row0 = cy_idx * 2 * width;
            let y_row1 = y_row0 + width;
            let c_row = cy_idx * cw;

            let mut cx = 0usize;
            while cx + 16 <= cw {
                // Load 16 Cb/Cr u16, compute deltas (subtract chroma center).
                let cb_i16 =
                    _mm256_loadu_si256(cb.as_ptr().add(c_row + cx) as *const _);
                let cr_i16 =
                    _mm256_loadu_si256(cr.as_ptr().add(c_row + cx) as *const _);
                let cbl = _mm256_sub_epi16(cb_i16, v_chroma_center);
                let crl = _mm256_sub_epi16(cr_i16, v_chroma_center);

                // Per-chroma Y delta in Q15.
                let dy_cb = _mm256_mulhrs_epi16(cbl, v_m_y_cb);
                let dy_cr = _mm256_mulhrs_epi16(crl, v_m_y_cr);
                let dy_chroma = _mm256_add_epi16(dy_cb, dy_cr);

                // Expand 16 chroma deltas → 32 per-luma deltas via stack
                // scratch (same as 8-bit kernel — see comment there for
                // the in-register-vs-scratch tradeoff).
                let mut dy_luma = [0i16; 16];
                _mm256_storeu_si256(dy_luma.as_mut_ptr() as *mut _, dy_chroma);
                let mut dy_luma_pair = [0i16; 32];
                for i in 0..16 {
                    dy_luma_pair[i * 2] = dy_luma[i];
                    dy_luma_pair[i * 2 + 1] = dy_luma[i];
                }
                let dy_luma_lo =
                    _mm256_loadu_si256(dy_luma_pair.as_ptr() as *const _);
                let dy_luma_hi =
                    _mm256_loadu_si256(dy_luma_pair.as_ptr().add(16) as *const _);

                // Apply to both luma rows for this chroma row.
                for row_off in [y_row0, y_row1] {
                    // Load 32 luma u16 across two 256-bit registers.
                    let y_lo =
                        _mm256_loadu_si256(y.as_ptr().add(row_off + cx * 2) as *const _);
                    let y_hi = _mm256_loadu_si256(
                        y.as_ptr().add(row_off + cx * 2 + 16) as *const _,
                    );

                    let y_lo_out = _mm256_add_epi16(y_lo, dy_luma_lo);
                    let y_hi_out = _mm256_add_epi16(y_hi, dy_luma_hi);

                    // Clamp to limited-range luma [64, 940].
                    let y_lo_out = _mm256_min_epi16(
                        _mm256_max_epi16(y_lo_out, v_luma_lo),
                        v_luma_hi,
                    );
                    let y_hi_out = _mm256_min_epi16(
                        _mm256_max_epi16(y_hi_out, v_luma_lo),
                        v_luma_hi,
                    );

                    _mm256_storeu_si256(
                        y.as_mut_ptr().add(row_off + cx * 2) as *mut _,
                        y_lo_out,
                    );
                    _mm256_storeu_si256(
                        y.as_mut_ptr().add(row_off + cx * 2 + 16) as *mut _,
                        y_hi_out,
                    );
                }

                cx += 16;
            }

            // Scalar tail for luma of this chroma row.
            while cx < cw {
                let cb_idx = c_row + cx;
                let cbl = cb[cb_idx] as i32 - CHROMA_CENTER_10BIT;
                let crl = cr[cb_idx] as i32 - CHROMA_CENTER_10BIT;
                let delta = (super::M_Y_CB * cbl + super::M_Y_CR * crl + super::Q15_ROUND)
                    >> super::Q15;
                let xi = cx * 2;
                for row_off in [y_row0, y_row1] {
                    for sub in 0..2 {
                        let idx = row_off + xi + sub;
                        y[idx] = clamp_y_10bit(y[idx] as i32 + delta);
                    }
                }
                cx += 1;
            }
        }

        // ---- Chroma pass (no luma coupling) ----
        // 16 samples per iter.
        let total_c = cb.len();
        let mut i = 0usize;
        while i + 16 <= total_c {
            let cb_i16 = _mm256_loadu_si256(cb.as_ptr().add(i) as *const _);
            let cr_i16 = _mm256_loadu_si256(cr.as_ptr().add(i) as *const _);
            let cbl = _mm256_sub_epi16(cb_i16, v_chroma_center);
            let crl = _mm256_sub_epi16(cr_i16, v_chroma_center);

            // Cb709 = ΔCb + (M_CB_CB-32768)·ΔCb·2^-15 + M_CB_CR·ΔCr·2^-15 + 512
            let cb_corr = _mm256_mulhrs_epi16(cbl, v_m_cb_cb_corr);
            let cb_cross = _mm256_mulhrs_epi16(crl, v_m_cb_cr);
            let new_cb = _mm256_add_epi16(_mm256_add_epi16(cbl, cb_corr), cb_cross);
            let new_cb = _mm256_add_epi16(new_cb, v_chroma_center);

            // Cr709 = ΔCr + (M_CR_CR-32768)·ΔCr·2^-15 + M_CR_CB·ΔCb·2^-15 + 512
            let cr_corr = _mm256_mulhrs_epi16(crl, v_m_cr_cr_corr);
            let cr_cross = _mm256_mulhrs_epi16(cbl, v_m_cr_cb);
            let new_cr = _mm256_add_epi16(_mm256_add_epi16(crl, cr_corr), cr_cross);
            let new_cr = _mm256_add_epi16(new_cr, v_chroma_center);

            // Clamp [64, 960].
            let new_cb =
                _mm256_min_epi16(_mm256_max_epi16(new_cb, v_chroma_lo), v_chroma_hi);
            let new_cr =
                _mm256_min_epi16(_mm256_max_epi16(new_cr, v_chroma_lo), v_chroma_hi);

            _mm256_storeu_si256(cb.as_mut_ptr().add(i) as *mut _, new_cb);
            _mm256_storeu_si256(cr.as_mut_ptr().add(i) as *mut _, new_cr);

            i += 16;
        }

        // Scalar tail for chroma.
        while i < total_c {
            let cbl = cb[i] as i32 - CHROMA_CENTER_10BIT;
            let crl = cr[i] as i32 - CHROMA_CENTER_10BIT;
            let new_cb = (super::M_CB_CB * cbl + super::M_CB_CR * crl + super::Q15_ROUND)
                >> super::Q15;
            let new_cr = (super::M_CR_CB * cbl + super::M_CR_CR * crl + super::Q15_ROUND)
                >> super::Q15;
            cb[i] = clamp_c_10bit(new_cb + CHROMA_CENTER_10BIT);
            cr[i] = clamp_c_10bit(new_cr + CHROMA_CENTER_10BIT);
            i += 1;
        }
    }
}
