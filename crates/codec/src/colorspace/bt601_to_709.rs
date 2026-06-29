// =============================================================================
// BT.601 → BT.709 YCbCr matrix conversion (limited-range 8-bit).
// =============================================================================
//
// Derived from BT.601 M_YUV→RGB composed with BT.709 M_RGB→YUV, both
// in 8-bit studio-range form (Y in [16,235], Cb/Cr in [16,240]).
//
// Derivation path:
//   1. BT.601 YCbCr (limited) → R'G'B' in [0,1], using Kr=0.299,
//      Kg=0.587, Kb=0.114 and the standard limited-range scaling
//      (Y'=(Y-16)/219, Pb=(Cb-128)/224, Pr=(Cr-128)/224).
//   2. BT.709 R'G'B' → YCbCr (limited), using Kr=0.2126, Kg=0.7152,
//      Kb=0.0722 and the inverse scaling (Y_out = 219·Y' + 16,
//      Cb_out = 224·Pb + 128, Cr_out = 224·Pr + 128).
//   3. Multiply M_709 · M_601^-1 on the delta vector
//      (Y-16, Cb-128, Cr-128) to get a single 3×3 with zero offsets
//      in delta space.
//
// Result (matrix applied to deltas):
//   ΔY709  = 1.00000·ΔY - 0.11555·ΔCb - 0.20794·ΔCr
//   ΔCb709 = 0·ΔY + 1.01864·ΔCb + 0.11462·ΔCr
//   ΔCr709 = 0·ΔY + 0.07505·ΔCb + 1.02533·ΔCr
//
// where ΔY = Y-16, ΔCb = Cb-128, ΔCr = Cr-128. Chroma rows have NO
// luma coupling — because BT.601 and BT.709 share the same limited-
// range chroma basis (Pb,Pr scaled by 224 in both), and the chroma
// basis vectors change only with Kr/Kb, not with luma.
//
// Sanity check under this matrix:
//   (16, 128, 128) → (16, 128, 128)   [black round-trips]
//   (235, 128, 128) → (235, 128, 128) [white round-trips]
//   (128, 128, 128) → (128, 128, 128) [any gray round-trips]
// because all three inputs have ΔCb=ΔCr=0, so ΔY709 = ΔY → Y
// unchanged, and ΔCb709 = ΔCr709 = 0 → chroma unchanged.
//
// Matrix constants live in the parent module (super::Q15, super::M_Y_CB, etc.)
// so they can be shared with the 10-bit variant without duplication.

use anyhow::{Result, bail};
use bytes::BytesMut;

use crate::frame::{ColorSpace, VideoFrame};

#[inline(always)]
fn clamp_y(v: i32) -> u8 {
    v.clamp(16, 235) as u8
}

#[inline(always)]
fn clamp_c(v: i32) -> u8 {
    v.clamp(16, 240) as u8
}

/// Scalar reference implementation — correctness baseline.
///
/// Operates in-place on three planes (Y, Cb, Cr). Chroma planes are
/// half-width / half-height of luma (4:2:0 subsampling). The matrix
/// couples chroma into luma (Y709 depends on ΔCb, ΔCr), so for each
/// luma sample we read the Cb/Cr sample covering it via the 2:1
/// subsampling grid (chroma-centered between luma rows/cols, but for
/// this pipeline we use the standard "shared per 2×2 block" mapping
/// — both decoders in-tree produce this layout).
///
/// Order matters: we must read the *original* Cb/Cr values before
/// overwriting them with Cb709/Cr709. We therefore update luma first
/// (consuming original chroma deltas) and update chroma last.
fn bt601_to_bt709_scalar(y: &mut [u8], cb: &mut [u8], cr: &mut [u8], width: usize, height: usize) {
    debug_assert_eq!(y.len(), width * height);
    debug_assert_eq!(cb.len(), (width / 2) * (height / 2));
    debug_assert_eq!(cr.len(), (width / 2) * (height / 2));

    let cw = width / 2;

    // Luma: Y709 = Y601 + M_Y_CB * ΔCb + M_Y_CR * ΔCr  (per-sample).
    // Each chroma sample covers a 2×2 luma block.
    for yi in 0..height {
        let cy = yi >> 1;
        for xi in 0..width {
            let cx = xi >> 1;
            let cbl = cb[cy * cw + cx] as i32 - 128;
            let crl = cr[cy * cw + cx] as i32 - 128;
            let y_orig = y[yi * width + xi] as i32;
            let delta =
                (super::M_Y_CB * cbl + super::M_Y_CR * crl + super::Q15_ROUND) >> super::Q15;
            y[yi * width + xi] = clamp_y(y_orig + delta);
        }
    }

    // Chroma: no luma coupling. Pure 2×2 chroma → chroma transform.
    for v in cb.iter_mut().zip(cr.iter_mut()) {
        let (cbp, crp) = v;
        let cbl = *cbp as i32 - 128;
        let crl = *crp as i32 - 128;
        let new_cb =
            (super::M_CB_CB * cbl + super::M_CB_CR * crl + super::Q15_ROUND) >> super::Q15;
        let new_cr =
            (super::M_CR_CB * cbl + super::M_CR_CR * crl + super::Q15_ROUND) >> super::Q15;
        *cbp = clamp_c(new_cb + 128);
        *crp = clamp_c(new_cr + 128);
    }
}

/// Public scalar entry point — for bench / tests.
pub fn bt601_to_bt709_planes_scalar(
    y: &mut [u8],
    cb: &mut [u8],
    cr: &mut [u8],
    width: usize,
    height: usize,
) {
    bt601_to_bt709_scalar(y, cb, cr, width, height);
}

/// Runtime-dispatched entry point. Uses AVX2 if the CPU advertises
/// it, scalar fallback otherwise. Safe wrapper around the unsafe
/// target-feature specialization.
pub fn bt601_to_bt709_planes(
    y: &mut [u8],
    cb: &mut [u8],
    cr: &mut [u8],
    width: usize,
    height: usize,
) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: avx2 feature was runtime-detected above.
            unsafe {
                bt601_to_bt709_avx2(y, cb, cr, width, height);
            }
            return;
        }
    }
    bt601_to_bt709_scalar(y, cb, cr, width, height);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn bt601_to_bt709_avx2(
    y: &mut [u8],
    cb: &mut [u8],
    cr: &mut [u8],
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

        // Q15 coefficients as packed 16-bit lanes. `_mm256_mulhrs_epi16`
        // computes (a*b + 0x4000) >> 15 on i16 lanes — the fixed-point
        // multiply we want. Inputs must be pre-subtracted to deltas
        // (ΔCb = Cb-128, ΔCr = Cr-128, in [-128, 127]). All six chroma
        // coefficients fit in i16 (range ~[-6814, 33598]): the 1.02533
        // coefficient (33598) exceeds i16_max (32767), so we handle it
        // specially by splitting out the identity contribution for that
        // coupling.
        //
        // Trick: write Cb709 = ΔCb + (M_CB_CB - 32768)/32768 · ΔCb + ...
        //        Cr709 = ΔCr + M_CR_CB/32768 · ΔCb + (M_CR_CR - 32768)/32768 · ΔCr
        // so the ~1.0 coefficients are represented as a free identity add
        // + a small correction. All i16-safe.
        let v_m_y_cb = _mm256_set1_epi16(super::M_Y_CB as i16); // -3786
        let v_m_y_cr = _mm256_set1_epi16(super::M_Y_CR as i16); // -6814
        let v_m_cb_cb_corr = _mm256_set1_epi16((super::M_CB_CB - 32768) as i16); // 611
        let v_m_cb_cr = _mm256_set1_epi16(super::M_CB_CR as i16); // 3756
        let v_m_cr_cb = _mm256_set1_epi16(super::M_CR_CB as i16); // 2459
        let v_m_cr_cr_corr = _mm256_set1_epi16((super::M_CR_CR - 32768) as i16); // 830

        let v_128 = _mm256_set1_epi16(128);
        let v_chroma_lo = _mm256_set1_epi16(16);
        let v_chroma_hi = _mm256_set1_epi16(240);
        let v_luma_lo = _mm256_set1_epi16(16);
        let v_luma_hi = _mm256_set1_epi16(235);

        // ---- Luma pass ----
        // For each 2×2 luma block we share one (Cb, Cr) sample. Process
        // 16 chroma samples per iteration → 32 luma cols on two rows (64
        // luma outputs per iter).
        for cy_idx in 0..ch {
            let y_row0 = cy_idx * 2 * width;
            let y_row1 = y_row0 + width;
            let c_row = cy_idx * cw;

            let mut cx = 0usize;
            while cx + 16 <= cw {
                // Load 16 Cb/Cr, compute per-chroma delta.
                let cb_u8 = _mm_loadu_si128(cb.as_ptr().add(c_row + cx) as *const _);
                let cr_u8 = _mm_loadu_si128(cr.as_ptr().add(c_row + cx) as *const _);
                let cb_i16 = _mm256_cvtepu8_epi16(cb_u8);
                let cr_i16 = _mm256_cvtepu8_epi16(cr_u8);
                let cbl = _mm256_sub_epi16(cb_i16, v_128);
                let crl = _mm256_sub_epi16(cr_i16, v_128);

                // Per-chroma Y delta: d = -0.1155·ΔCb - 0.2079·ΔCr in Q15.
                // mulhrs: (a*b + 0x4000) >> 15.
                let dy_cb = _mm256_mulhrs_epi16(cbl, v_m_y_cb);
                let dy_cr = _mm256_mulhrs_epi16(crl, v_m_y_cr);
                let dy_chroma = _mm256_add_epi16(dy_cb, dy_cr); // 16 per-chroma deltas

                // Apply to luma: each chroma sample covers a 2×2 block.
                // Horizontal: duplicate each 16-bit lane into two adjacent
                // 16-bit lanes → 32 deltas aligned with 32 luma cols.
                // `_mm256_unpacklo_epi16` / `unpackhi_epi16` with self
                // interleaves adjacent lanes; but we want dy[0], dy[0],
                // dy[1], dy[1], .... Use two shuffles + permute.
                //
                // Within each 128-bit lane, `unpacklo_epi16(a, a)` yields
                // a0 a0 a1 a1 a2 a2 a3 a3; `unpackhi_epi16(a, a)` yields
                // a4 a4 ... a7 a7. We have 16 lanes → 4 × 128-bit lanes of
                // output, which we rearrange with permute4x64.
                // Trick: treat the 256-bit register as two halves.
                // Expand 16 chroma deltas → 32 per-luma deltas via a small
                // stack-scratch path. An in-register expand would save two
                // 64-byte cache-line round-trips per 16 chroma samples, but
                // on a 1080p frame this is ~68k bytes of scratch total —
                // negligible vs the 2M luma stores. Kept simple for
                // maintainability; revisit if this function ever shows up
                // at the top of a pprof.
                let mut dy_luma = [0i16; 32];
                _mm256_storeu_si256(dy_luma.as_mut_ptr().add(0) as *mut _, dy_chroma);
                // Above stored 16 chroma deltas. Now expand in-register.
                // Actually simpler: use a second aligned buffer with pair
                // duplication done by indexing.
                let mut dy_luma_pair = [0i16; 32];
                for i in 0..16 {
                    dy_luma_pair[i * 2] = dy_luma[i];
                    dy_luma_pair[i * 2 + 1] = dy_luma[i];
                }
                let dy_luma_lo =
                    _mm256_loadu_si256(dy_luma_pair.as_ptr().add(0) as *const _);
                let dy_luma_hi =
                    _mm256_loadu_si256(dy_luma_pair.as_ptr().add(16) as *const _);

                // Process both luma rows for this chroma row. Both share
                // dy_luma_* because chroma is 4:2:0.
                for row_off in [y_row0, y_row1] {
                    // Load 32 luma pixels.
                    let y_u8 =
                        _mm256_loadu_si256(y.as_ptr().add(row_off + cx * 2) as *const _);
                    // Widen low 16 bytes and high 16 bytes to i16.
                    let y_lo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(y_u8));
                    let y_hi =
                        _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(y_u8));

                    let y_lo_out = _mm256_add_epi16(y_lo, dy_luma_lo);
                    let y_hi_out = _mm256_add_epi16(y_hi, dy_luma_hi);

                    // Clamp to limited-range luma [16, 235].
                    let y_lo_out = _mm256_min_epi16(
                        _mm256_max_epi16(y_lo_out, v_luma_lo),
                        v_luma_hi,
                    );
                    let y_hi_out = _mm256_min_epi16(
                        _mm256_max_epi16(y_hi_out, v_luma_lo),
                        v_luma_hi,
                    );

                    // Pack i16 → u8 with saturation and store 32 bytes.
                    let packed = _mm256_packus_epi16(y_lo_out, y_hi_out);
                    // packus interleaves lanes; permute to
                    // [lo[0..7], hi[0..7], lo[8..15], hi[8..15]] → lane order.
                    let packed = _mm256_permute4x64_epi64::<0b11_01_10_00>(packed);
                    _mm256_storeu_si256(
                        y.as_mut_ptr().add(row_off + cx * 2) as *mut _,
                        packed,
                    );
                }

                cx += 16;
            }

            // Scalar tail for luma of this chroma row.
            while cx < cw {
                let cb_idx = c_row + cx;
                let cbl = cb[cb_idx] as i32 - 128;
                let crl = cr[cb_idx] as i32 - 128;
                let delta = (super::M_Y_CB * cbl + super::M_Y_CR * crl + super::Q15_ROUND)
                    >> super::Q15;
                let xi = cx * 2;
                for row_off in [y_row0, y_row1] {
                    for sub in 0..2 {
                        let idx = row_off + xi + sub;
                        y[idx] = clamp_y(y[idx] as i32 + delta);
                    }
                }
                cx += 1;
            }
        }

        // ---- Chroma pass (no luma coupling) ----
        // 16 samples per iteration.
        let total_c = cb.len();
        let mut i = 0usize;
        while i + 16 <= total_c {
            let cb_u8 = _mm_loadu_si128(cb.as_ptr().add(i) as *const _);
            let cr_u8 = _mm_loadu_si128(cr.as_ptr().add(i) as *const _);
            let cb_i16 = _mm256_cvtepu8_epi16(cb_u8);
            let cr_i16 = _mm256_cvtepu8_epi16(cr_u8);
            let cbl = _mm256_sub_epi16(cb_i16, v_128);
            let crl = _mm256_sub_epi16(cr_i16, v_128);

            // Cb709 = ΔCb + (M_CB_CB-32768)·ΔCb·2^-15 + M_CB_CR·ΔCr·2^-15 + 128
            let cb_corr = _mm256_mulhrs_epi16(cbl, v_m_cb_cb_corr);
            let cb_cross = _mm256_mulhrs_epi16(crl, v_m_cb_cr);
            let new_cb = _mm256_add_epi16(_mm256_add_epi16(cbl, cb_corr), cb_cross);
            let new_cb = _mm256_add_epi16(new_cb, v_128);

            // Cr709 = ΔCr + (M_CR_CR-32768)·ΔCr·2^-15 + M_CR_CB·ΔCb·2^-15 + 128
            let cr_corr = _mm256_mulhrs_epi16(crl, v_m_cr_cr_corr);
            let cr_cross = _mm256_mulhrs_epi16(cbl, v_m_cr_cb);
            let new_cr = _mm256_add_epi16(_mm256_add_epi16(crl, cr_corr), cr_cross);
            let new_cr = _mm256_add_epi16(new_cr, v_128);

            // Clamp [16, 240].
            let new_cb =
                _mm256_min_epi16(_mm256_max_epi16(new_cb, v_chroma_lo), v_chroma_hi);
            let new_cr =
                _mm256_min_epi16(_mm256_max_epi16(new_cr, v_chroma_lo), v_chroma_hi);

            // Pack and store.
            let cb_packed = _mm256_packus_epi16(new_cb, new_cb);
            let cr_packed = _mm256_packus_epi16(new_cr, new_cr);
            let cb_packed = _mm256_permute4x64_epi64::<0b00_00_10_00>(cb_packed);
            let cr_packed = _mm256_permute4x64_epi64::<0b00_00_10_00>(cr_packed);
            _mm_storeu_si128(
                cb.as_mut_ptr().add(i) as *mut _,
                _mm256_castsi256_si128(cb_packed),
            );
            _mm_storeu_si128(
                cr.as_mut_ptr().add(i) as *mut _,
                _mm256_castsi256_si128(cr_packed),
            );

            i += 16;
        }

        // Scalar tail for chroma.
        while i < total_c {
            let cbl = cb[i] as i32 - 128;
            let crl = cr[i] as i32 - 128;
            let new_cb =
                (super::M_CB_CB * cbl + super::M_CB_CR * crl + super::Q15_ROUND) >> super::Q15;
            let new_cr =
                (super::M_CR_CB * cbl + super::M_CR_CR * crl + super::Q15_ROUND) >> super::Q15;
            cb[i] = clamp_c(new_cb + 128);
            cr[i] = clamp_c(new_cr + 128);
            i += 1;
        }
    }
}

pub(super) fn recolor_yuv420p_bt601_to_bt709(frame: &VideoFrame) -> Result<VideoFrame> {
    let w = frame.width as usize;
    let h = frame.height as usize;
    let y_size = w * h;
    let c_size = y_size / 4;

    if frame.data.len() < y_size + 2 * c_size {
        bail!(
            "frame data too short for yuv420p {}x{}: {} bytes",
            w,
            h,
            frame.data.len()
        );
    }
    if !w.is_multiple_of(2) || !h.is_multiple_of(2) {
        bail!(
            "BT.601→BT.709 requires even dimensions for 4:2:0 subsampling; got {}x{}",
            w,
            h
        );
    }

    let mut y = frame.data[..y_size].to_vec();
    let mut cb = frame.data[y_size..y_size + c_size].to_vec();
    let mut cr = frame.data[y_size + c_size..y_size + 2 * c_size].to_vec();

    bt601_to_bt709_planes(&mut y, &mut cb, &mut cr, w, h);

    let mut out = BytesMut::with_capacity(y_size + 2 * c_size);
    out.extend_from_slice(&y);
    out.extend_from_slice(&cb);
    out.extend_from_slice(&cr);

    Ok(VideoFrame::new(
        out.freeze(),
        frame.width,
        frame.height,
        frame.format,
        ColorSpace::Bt709,
        frame.pts,
    ))
}
