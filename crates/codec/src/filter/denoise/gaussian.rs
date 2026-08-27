//! Gaussian (separable low-pass) denoise.

use super::simd::{Simd, Tier, round_clamp_u8, tiered};
use super::{clamp_idx, for_row_bands};

const K: [f32; 5] = [1.0, 4.0, 6.0, 4.0, 1.0];
const KSUM: f32 = 16.0;
const R: isize = 2;
const RU: usize = R as usize;

/// Separable 5-tap Gaussian blur (σ≈1.0, kernel `[1,4,6,4,1]/16`) — a plain
/// low-pass that smooths noise and detail alike. Border uses edge-replicate.
pub(super) fn plane(src: &[u8], w: usize, h: usize) -> Vec<u8> {
    plane_tiered(src, w, h, Tier::detect())
}

/// [`plane`] at an explicit tier, so tests can hold every tier to the scalar one.
fn plane_tiered(src: &[u8], w: usize, h: usize, tier: Tier) -> Vec<u8> {
    // Horizontal pass → f32 scratch.
    let mut tmp = vec![0f32; w * h];
    for_row_bands(&mut tmp, w, 512, |y0, rows| {
        for (i, row) in rows.chunks_mut(w).enumerate() {
            hpass_row(tier, &src[(y0 + i) * w..][..w], row);
        }
    });
    // Vertical pass → u8.
    let mut out = vec![0u8; w * h];
    for_row_bands(&mut out, w, 512, |y0, rows| {
        for (i, row) in rows.chunks_mut(w).enumerate() {
            vpass_row(tier, &tmp, w, h, y0 + i, row);
        }
    });
    out
}

fn hpass_span(src: &[u8], tmp: &mut [f32], range: std::ops::Range<usize>) {
    let w = src.len();
    for x in range {
        let mut acc = 0f32;
        for (k, &kw) in K.iter().enumerate() {
            let xx = clamp_idx(x as isize + k as isize - R, w);
            acc += kw * src[xx] as f32;
        }
        tmp[x] = acc / KSUM;
    }
}

fn hpass_scalar(src: &[u8], tmp: &mut [f32]) {
    hpass_span(src, tmp, 0..src.len());
}

#[cfg_attr(
    not(any(target_arch = "x86", target_arch = "x86_64")),
    allow(dead_code)
)]
#[inline(always)]
unsafe fn hpass_body<S: Simd>(src: &[u8], tmp: &mut [f32]) {
    unsafe {
        let w = src.len();
        let lo = RU.min(w);
        let hi = w.saturating_sub(RU);
        hpass_span(src, tmp, 0..lo);
        let ksum = S::set1_f32(KSUM);
        let mut x = lo;
        while x + S::LANES <= hi {
            let mut acc = S::set1_f32(0.0);
            for (k, &kw) in K.iter().enumerate() {
                let v = S::load_u8_f32(src.as_ptr().add(x + k - RU));
                acc = S::add_f32(acc, S::mul_f32(S::set1_f32(kw), v));
            }
            S::store_f32(tmp.as_mut_ptr().add(x), S::div_f32(acc, ksum));
            x += S::LANES;
        }
        hpass_span(src, tmp, x..w);
    }
}

tiered!(fn hpass_row(src: &[u8], tmp: &mut [f32]) => hpass_body, scalar hpass_scalar);

fn vpass_span(
    tmp: &[f32],
    w: usize,
    h: usize,
    y: usize,
    out: &mut [u8],
    range: std::ops::Range<usize>,
) {
    for x in range {
        let mut acc = 0f32;
        for (k, &kw) in K.iter().enumerate() {
            let yy = clamp_idx(y as isize + k as isize - R, h);
            acc += kw * tmp[yy * w + x];
        }
        out[x] = (acc / KSUM).round().clamp(0.0, 255.0) as u8;
    }
}

fn vpass_scalar(tmp: &[f32], w: usize, h: usize, y: usize, out: &mut [u8]) {
    vpass_span(tmp, w, h, y, out, 0..w);
}

/// The vertical pass clamps rows, not columns, so every column vectorises.
#[cfg_attr(
    not(any(target_arch = "x86", target_arch = "x86_64")),
    allow(dead_code)
)]
#[inline(always)]
unsafe fn vpass_body<S: Simd>(tmp: &[f32], w: usize, h: usize, y: usize, out: &mut [u8]) {
    unsafe {
        let rows: [*const f32; 5] = std::array::from_fn(|k| {
            tmp.as_ptr()
                .add(clamp_idx(y as isize + k as isize - R, h) * w)
        });
        let ksum = S::set1_f32(KSUM);
        let mut x = 0;
        while x + S::LANES <= w {
            let mut acc = S::set1_f32(0.0);
            for (k, &kw) in K.iter().enumerate() {
                acc = S::add_f32(
                    acc,
                    S::mul_f32(S::set1_f32(kw), S::load_f32(rows[k].add(x))),
                );
            }
            S::store_f32_u8(
                out.as_mut_ptr().add(x),
                round_clamp_u8::<S>(S::div_f32(acc, ksum)),
            );
            x += S::LANES;
        }
        vpass_span(tmp, w, h, y, out, x..w);
    }
}

tiered!(fn vpass_row(tmp: &[f32], w: usize, h: usize, y: usize, out: &mut [u8]) => vpass_body, scalar vpass_scalar);

#[cfg(test)]
mod tests {
    use super::super::test_support::planes;
    use super::*;

    #[test]
    fn every_tier_matches_the_scalar_reference_bit_for_bit() {
        for (w, h, src) in planes() {
            let want = plane_tiered(&src, w, h, Tier::Scalar);
            for tier in Tier::available() {
                assert_eq!(
                    plane_tiered(&src, w, h, tier),
                    want,
                    "{tier:?} diverged at {w}x{h}"
                );
            }
        }
    }
}
