// =============================================================================
// 4:4:4 → 4:2:0 chroma downsample, separable Lanczos-2 (the `lanczos` option).
// =============================================================================
//
// The default 2×2 box average (`downsample_444.rs`) is sited at the centre of
// the 2×2 block — half a sample right of where every H.264 / HEVC / AV1
// decoder assumes 4:2:0 chroma sits by default (`chroma_sample_loc_type` 0,
// MPEG-2 siting: horizontally co-sited with the left luma of the pair,
// vertically midway between the rows). The box also has a wide, slow
// roll-off that lets chroma alias through. This filter fixes both:
//
//   horizontal — centred on the even column 2·cx (co-sited), Lanczos-2
//                windowed sinc for factor-2 decimation sampled at integer
//                offsets −3..3, quantised to Q6 and renormalised:
//                  offset: −3   −2   −1    0   +1   +2   +3
//                  tap   : −2    0   18   32   18    0   −2      (/64)
//   vertical   — centred midway between rows 2·cy and 2·cy+1, the same
//                kernel sampled at half-integer offsets, 6 taps:
//                  rows  : 2cy−2 2cy−1 2cy 2cy+1 2cy+2 2cy+3
//                  tap   :  −3     7    28   28    7    −3      (/64)
//
// Lanczos-2, L(x) = sinc(x/2)·sinc(x/4) on |x| < 4 in source-sample units:
// horizontal L(0)=1, L(±1)=.573, L(±2)=0, L(±3)=−.064 → [−.032 0 .284 .495 …];
// vertical L(±.5)=.877, L(±1.5)=.235, L(±2.5)=−.086 → [−.042 .115 .427 …]
// (the ±3.5 taps, −.018 each, are dropped). Both pass at Q6 so the whole
// filter is one `(Σ + 2048) >> 12` with an i32 accumulator; 12-bit input
// stays exact (|Σ| ≤ 4095 · 68 · 70 ≈ 1.9e7).
//
// Two passes, horizontal first into an i32 intermediate at Q6 (no rounding
// between passes), then vertical. Edges replicate (indices clamped), which
// is what the box filter and libswscale do.
//
// Scalar reference + AVX2 (8 × i32 lanes; the row de-interleave into even /
// odd columns is 16 × u16 per load). Bit-exact with each other — `tests.rs`
// checks random 8/10/12-bit planes at odd sizes.

/// Horizontal taps at offsets −3..=3 around the even column, Q6.
const H_TAPS: [i32; 7] = [-2, 0, 18, 32, 18, 0, -2];
/// Vertical taps for rows 2cy−2 ..= 2cy+3, Q6.
const V_TAPS: [i32; 6] = [-3, 7, 28, 28, 7, -3];
const Q12_ROUND: i32 = 1 << 11;

/// Horizontal pass, scalar: one source row of `width` samples → `cw`
/// Q6 values.
fn h_pass_row_scalar(row: &[u16], width: usize, out: &mut [i32]) {
    let cw = width.div_ceil(2);
    let last = width - 1;
    for cx in 0..cw {
        let c = 2 * cx;
        let s = |off: isize| -> i32 {
            let i = (c as isize + off).clamp(0, last as isize) as usize;
            row[i] as i32
        };
        // Offsets ±2 carry zero weight.
        out[cx] = H_TAPS[0] * (s(-3) + s(3)) + H_TAPS[2] * (s(-1) + s(1)) + H_TAPS[3] * s(0);
    }
}

/// Vertical pass, scalar: six Q6 rows → one output chroma row, clamped
/// to `0..=max`.
fn v_pass_row_scalar(rows: [&[i32]; 6], cw: usize, max: i32, out: &mut [u16]) {
    for cx in 0..cw {
        let mut acc = Q12_ROUND;
        for (r, tap) in rows.iter().zip(V_TAPS) {
            acc += tap * r[cx];
        }
        out[cx] = (acc >> 12).clamp(0, max) as u16;
    }
}

/// The six source-row indices feeding output chroma row `cy`, clamped.
fn v_rows(cy: usize, height: usize) -> [usize; 6] {
    let base = 2 * cy as isize - 2;
    std::array::from_fn(|k| (base + k as isize).clamp(0, height as isize - 1) as usize)
}

/// Scalar reference. One full-resolution `u16` chroma plane of
/// `width`×`height` → its 4:2:0 plane (`⌈w/2⌉`×`⌈h/2⌉`), samples in
/// `0..=max`.
pub fn downsample_plane_lanczos_scalar(
    plane: &[u16],
    width: usize,
    height: usize,
    max: u16,
) -> Vec<u16> {
    assert!(width > 0 && height > 0);
    debug_assert_eq!(plane.len(), width * height);
    let cw = width.div_ceil(2);
    let ch = height.div_ceil(2);
    let mut h = vec![0i32; cw * height];
    for r in 0..height {
        h_pass_row_scalar(
            &plane[r * width..(r + 1) * width],
            width,
            &mut h[r * cw..(r + 1) * cw],
        );
    }
    let mut out = vec![0u16; cw * ch];
    for cy in 0..ch {
        let rows = v_rows(cy, height).map(|r| &h[r * cw..(r + 1) * cw]);
        v_pass_row_scalar(rows, cw, max as i32, &mut out[cy * cw..(cy + 1) * cw]);
    }
    out
}

/// Runtime-dispatched (AVX2 on x86 when available) version of
/// [`downsample_plane_lanczos_scalar`].
pub fn downsample_plane_lanczos(plane: &[u16], width: usize, height: usize, max: u16) -> Vec<u16> {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx2") && width >= 32 {
            // SAFETY: avx2 runtime-detected.
            return unsafe { avx2::downsample_plane_lanczos_avx2(plane, width, height, max) };
        }
    }
    downsample_plane_lanczos_scalar(plane, width, height, max)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod avx2 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    use super::{H_TAPS, Q12_ROUND, V_TAPS, h_pass_row_scalar, v_pass_row_scalar, v_rows};

    /// Horizontal pass for one row. `even`/`odd` are scratch (i32,
    /// `cw + 4` long): `odd[k]` holds the odd column `2(k−2)+1`, clamped,
    /// so the four odd taps around `cx` are `odd[cx..cx+4]`.
    #[target_feature(enable = "avx2")]
    unsafe fn h_pass_row(
        row: &[u16],
        width: usize,
        even: &mut [i32],
        odd: &mut [i32],
        out: &mut [i32],
    ) {
        unsafe {
            let cw = width.div_ceil(2);
            let last = width - 1;
            // De-interleave: 16 u16 → 8 even + 8 odd i32 lanes per load.
            let lo_mask = _mm256_set1_epi32(0xFFFF);
            let mut cx = 0usize;
            while cx + 8 <= cw && 2 * cx + 16 <= width {
                let v = _mm256_loadu_si256(row.as_ptr().add(2 * cx) as *const _);
                _mm256_storeu_si256(
                    even.as_mut_ptr().add(cx) as *mut _,
                    _mm256_and_si256(v, lo_mask),
                );
                _mm256_storeu_si256(
                    odd.as_mut_ptr().add(cx + 2) as *mut _,
                    _mm256_srli_epi32(v, 16),
                );
                cx += 8;
            }
            while cx < cw {
                even[cx] = row[2 * cx] as i32;
                odd[cx + 2] = row[(2 * cx + 1).min(last)] as i32;
                cx += 1;
            }
            // Replicated edges: odd[−2], odd[−1] → column 0; odd[cw] → last.
            odd[0] = row[0] as i32;
            odd[1] = row[0] as i32;
            odd[cw + 2] = row[last] as i32;
            odd[cw + 3] = row[last] as i32;

            let t_m3 = _mm256_set1_epi32(H_TAPS[0]);
            let t_m1 = _mm256_set1_epi32(H_TAPS[2]);
            let t_0 = _mm256_set1_epi32(H_TAPS[3]);
            let mut cx = 0usize;
            while cx + 8 <= cw {
                let e0 = _mm256_loadu_si256(even.as_ptr().add(cx) as *const _);
                let o_m3 = _mm256_loadu_si256(odd.as_ptr().add(cx) as *const _);
                let o_m1 = _mm256_loadu_si256(odd.as_ptr().add(cx + 1) as *const _);
                let o_p1 = _mm256_loadu_si256(odd.as_ptr().add(cx + 2) as *const _);
                let o_p3 = _mm256_loadu_si256(odd.as_ptr().add(cx + 3) as *const _);
                let acc = _mm256_mullo_epi32(e0, t_0);
                let acc =
                    _mm256_add_epi32(acc, _mm256_mullo_epi32(_mm256_add_epi32(o_m1, o_p1), t_m1));
                let acc =
                    _mm256_add_epi32(acc, _mm256_mullo_epi32(_mm256_add_epi32(o_m3, o_p3), t_m3));
                _mm256_storeu_si256(out.as_mut_ptr().add(cx) as *mut _, acc);
                cx += 8;
            }
            if cx < cw {
                // Scalar tail from the original row (same clamping rule).
                let mut tail = vec![0i32; cw];
                h_pass_row_scalar(row, width, &mut tail);
                out[cx..cw].copy_from_slice(&tail[cx..cw]);
            }
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn v_pass_row(rows: [&[i32]; 6], cw: usize, max: i32, out: &mut [u16]) {
        unsafe {
            let taps = V_TAPS.map(|t| _mm256_set1_epi32(t));
            let round = _mm256_set1_epi32(Q12_ROUND);
            let zero = _mm256_setzero_si256();
            let vmax = _mm256_set1_epi32(max);
            let mut cx = 0usize;
            while cx + 8 <= cw {
                let mut acc = round;
                for k in 0..6 {
                    let r = _mm256_loadu_si256(rows[k].as_ptr().add(cx) as *const _);
                    acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(r, taps[k]));
                }
                let v = _mm256_srai_epi32(acc, 12);
                let v = _mm256_min_epi32(_mm256_max_epi32(v, zero), vmax);
                // 8 × i32 → 8 × u16: packus within lanes, then fix the order.
                let p = _mm256_packus_epi32(v, v);
                let p = _mm256_permute4x64_epi64(p, 0b11_01_10_00);
                _mm_storeu_si128(
                    out.as_mut_ptr().add(cx) as *mut _,
                    _mm256_castsi256_si128(p),
                );
                cx += 8;
            }
            if cx < cw {
                let mut tail = vec![0u16; cw];
                v_pass_row_scalar(rows, cw, max, &mut tail);
                out[cx..cw].copy_from_slice(&tail[cx..cw]);
            }
        }
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn downsample_plane_lanczos_avx2(
        plane: &[u16],
        width: usize,
        height: usize,
        max: u16,
    ) -> Vec<u16> {
        unsafe {
            assert!(width > 0 && height > 0);
            let cw = width.div_ceil(2);
            let ch = height.div_ceil(2);
            let mut even = vec![0i32; cw + 8];
            let mut odd = vec![0i32; cw + 12];
            let mut h = vec![0i32; cw * height];
            for r in 0..height {
                h_pass_row(
                    &plane[r * width..(r + 1) * width],
                    width,
                    &mut even,
                    &mut odd,
                    &mut h[r * cw..(r + 1) * cw],
                );
            }
            let mut out = vec![0u16; cw * ch];
            for cy in 0..ch {
                let rows = v_rows(cy, height).map(|r| &h[r * cw..(r + 1) * cw]);
                v_pass_row(rows, cw, max as i32, &mut out[cy * cw..(cy + 1) * cw]);
            }
            out
        }
    }
}
