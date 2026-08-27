//! Bilateral denoise (edge-preserving).

use super::simd::{Simd, Tier, round_clamp_u8, tiered};
use super::for_row_bands;

/// Window radius (5×5).
const R: isize = 2;
const RU: usize = R as usize;
const SPATIAL_SIGMA: f32 = 2.0;
const RANGE_SIGMA: f32 = 20.0;

/// The two weight tables. Both paths read the same bits — the vector path
/// gathers from `range` rather than recomputing it — which is part of why
/// they agree exactly.
struct Tables {
    spatial: [[f32; 5]; 5],
    range: [f32; 256],
}

fn tables() -> Tables {
    let mut spatial = [[0f32; 5]; 5];
    for dy in -R..=R {
        for dx in -R..=R {
            let d2 = (dx * dx + dy * dy) as f32;
            spatial[(dy + R) as usize][(dx + R) as usize] =
                (-d2 / (2.0 * SPATIAL_SIGMA * SPATIAL_SIGMA)).exp();
        }
    }
    let mut range = [0f32; 256];
    for (d, wt) in range.iter_mut().enumerate() {
        *wt = (-((d * d) as f32) / (2.0 * RANGE_SIGMA * RANGE_SIGMA)).exp();
    }
    Tables { spatial, range }
}

/// Edge-preserving bilateral filter over a 5×5 window. Each output sample is a
/// weighted average of its neighbourhood where the weight is `spatial(distance)
/// × range(|intensity − centre|)` — so samples across a strong intensity step
/// (an edge) barely contribute and edges stay sharp while flat noise averages
/// out. Border samples shrink the window (out-of-range neighbours are skipped).
pub(super) fn plane(src: &[u8], w: usize, h: usize) -> Vec<u8> {
    plane_tiered(src, w, h, Tier::detect())
}

/// [`plane`] at an explicit tier, so tests can hold every tier to the scalar one.
fn plane_tiered(src: &[u8], w: usize, h: usize, tier: Tier) -> Vec<u8> {
    let t = tables();
    let mut out = vec![0u8; w * h];
    for_row_bands(&mut out, w, 64, |y0, rows| {
        for (i, out_row) in rows.chunks_mut(w).enumerate() {
            row(tier, src, w, h, y0 + i, &t, out_row);
        }
    });
    out
}

/// One output row, columns `range`, scalar — the reference.
fn row_span(src: &[u8], w: usize, h: usize, y: usize, t: &Tables, out: &mut [u8], range: std::ops::Range<usize>) {
    for x in range {
        let centre = src[y * w + x] as i32;
        let mut sum = 0f32;
        let mut wsum = 0f32;
        for dy in -R..=R {
            let yy = y as isize + dy;
            if yy < 0 || yy >= h as isize {
                continue;
            }
            for dx in -R..=R {
                let xx = x as isize + dx;
                if xx < 0 || xx >= w as isize {
                    continue;
                }
                let s = src[yy as usize * w + xx as usize] as i32;
                let wt = t.spatial[(dy + R) as usize][(dx + R) as usize]
                    * t.range[(s - centre).unsigned_abs() as usize];
                sum += wt * s as f32;
                wsum += wt;
            }
        }
        out[x] = (sum / wsum).round().clamp(0.0, 255.0) as u8;
    }
}

fn row_scalar(src: &[u8], w: usize, h: usize, y: usize, t: &Tables, out: &mut [u8]) {
    row_span(src, w, h, y, t, out, 0..w);
}

/// The vector row: the columns whose whole 5-wide window is inside the row
/// take `LANES` at a time — the same 25 taps, in the same order, each a
/// gather from the same range table — and the two borders stay scalar.
#[cfg_attr(not(any(target_arch = "x86", target_arch = "x86_64")), allow(dead_code))]
#[inline(always)]
unsafe fn row_body<S: Simd>(src: &[u8], w: usize, h: usize, y: usize, t: &Tables, out: &mut [u8]) {
    unsafe {
        let lo = RU.min(w);
        let hi = w.saturating_sub(RU);
        row_span(src, w, h, y, t, out, 0..lo);
        let zero = S::set1_f32(0.0);
        let mut x = lo;
        while x + S::LANES <= hi {
            let centre = S::load_u8_i32(src.as_ptr().add(y * w + x));
            let mut sum = zero;
            let mut wsum = zero;
            for dy in -R..=R {
                let yy = y as isize + dy;
                if yy < 0 || yy >= h as isize {
                    continue;
                }
                let row = src.as_ptr().add(yy as usize * w);
                for dx in -R..=R {
                    let s_i = S::load_u8_i32(row.add((x as isize + dx) as usize));
                    let d = S::abs_i32(S::sub_i32(s_i, centre));
                    let wt = S::mul_f32(
                        S::set1_f32(t.spatial[(dy + R) as usize][(dx + R) as usize]),
                        S::gather_f32(&t.range, d),
                    );
                    sum = S::add_f32(sum, S::mul_f32(wt, S::i32_to_f32(s_i)));
                    wsum = S::add_f32(wsum, wt);
                }
            }
            S::store_f32_u8(out.as_mut_ptr().add(x), round_clamp_u8::<S>(S::div_f32(sum, wsum)));
            x += S::LANES;
        }
        row_span(src, w, h, y, t, out, x..w);
    }
}

tiered!(fn row(src: &[u8], w: usize, h: usize, y: usize, t: &Tables, out: &mut [u8]) => row_body, scalar row_scalar);

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
