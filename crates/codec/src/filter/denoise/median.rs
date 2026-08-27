//! Median denoise (impulse / salt-and-pepper).

use super::simd::{Simd, Tier, tiered};
use super::{clamp_idx, for_row_bands};

/// 3×3 median filter — replaces each sample with the median of its 3×3
/// neighbourhood, which removes isolated impulse (salt-and-pepper) samples
/// outright while leaving edges intact. Border uses edge-replicate.
pub(super) fn plane(src: &[u8], w: usize, h: usize) -> Vec<u8> {
    plane_tiered(src, w, h, Tier::detect())
}

/// [`plane`] at an explicit tier, so tests can hold every tier to the scalar one.
fn plane_tiered(src: &[u8], w: usize, h: usize, tier: Tier) -> Vec<u8> {
    let mut out = vec![0u8; w * h];
    for_row_bands(&mut out, w, |y0, rows| {
        for (i, row) in rows.chunks_mut(w).enumerate() {
            median_row(tier, src, w, h, y0 + i, row);
        }
    });
    out
}

fn median_span(src: &[u8], w: usize, h: usize, y: usize, out: &mut [u8], range: std::ops::Range<usize>) {
    let mut window = [0u8; 9];
    for x in range {
        let mut n = 0;
        for dy in -1isize..=1 {
            for dx in -1isize..=1 {
                let yy = clamp_idx(y as isize + dy, h);
                let xx = clamp_idx(x as isize + dx, w);
                window[n] = src[yy * w + xx];
                n += 1;
            }
        }
        window.sort_unstable();
        out[x] = window[4]; // median of 9
    }
}

fn median_scalar(src: &[u8], w: usize, h: usize, y: usize, out: &mut [u8]) {
    median_span(src, w, h, y, out, 0..w);
}

/// The median of nine through a 19-comparator network of `min`/`max` pairs
/// (Devillard's `opt_med9`). The median is a function of the multiset alone,
/// so any correct network gives exactly what the sort does.
#[cfg_attr(not(any(target_arch = "x86", target_arch = "x86_64")), allow(dead_code))]
#[inline(always)]
unsafe fn median_body<S: Simd>(src: &[u8], w: usize, h: usize, y: usize, out: &mut [u8]) {
    unsafe {
        let lo = 1.min(w);
        let hi = w.saturating_sub(1);
        median_span(src, w, h, y, out, 0..lo);
        let rows: [*const u8; 3] =
            std::array::from_fn(|k| src.as_ptr().add(clamp_idx(y as isize + k as isize - 1, h) * w));
        let mut x = lo;
        while x + S::LANES8 <= hi {
            let mut p: [S::B; 9] = std::array::from_fn(|k| S::load_u8(rows[k / 3].add(x + k % 3 - 1)));
            macro_rules! sort {
                ($a:expr, $b:expr) => {{
                    let lo = S::min_u8(p[$a], p[$b]);
                    let hi = S::max_u8(p[$a], p[$b]);
                    p[$a] = lo;
                    p[$b] = hi;
                }};
            }
            sort!(1, 2);
            sort!(4, 5);
            sort!(7, 8);
            sort!(0, 1);
            sort!(3, 4);
            sort!(6, 7);
            sort!(1, 2);
            sort!(4, 5);
            sort!(7, 8);
            sort!(0, 3);
            sort!(5, 8);
            sort!(4, 7);
            sort!(3, 6);
            sort!(1, 4);
            sort!(2, 5);
            sort!(4, 7);
            sort!(4, 2);
            sort!(6, 4);
            sort!(4, 2);
            S::store_u8(out.as_mut_ptr().add(x), p[4]);
            x += S::LANES8;
        }
        median_span(src, w, h, y, out, x..w);
    }
}

tiered!(fn median_row(src: &[u8], w: usize, h: usize, y: usize, out: &mut [u8]) => median_body, scalar median_scalar);

#[cfg(test)]
mod tests {
    use super::super::test_support::planes;
    use super::*;

    #[test]
    fn every_tier_matches_the_scalar_reference_bit_for_bit() {
        for (w, h, src) in planes() {
            let want = plane_tiered(&src, w, h, Tier::Scalar);
            for tier in Tier::available() {
                assert_eq!(plane_tiered(&src, w, h, tier), want, "{tier:?} diverged at {w}x{h}");
            }
        }
    }
}
