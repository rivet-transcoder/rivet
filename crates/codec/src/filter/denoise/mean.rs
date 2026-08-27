//! Mean (box) denoise — the cheapest smoother.

use super::simd::{Simd, Tier, tiered};
use super::{clamp_idx, for_row_bands};

/// `⌈2¹⁶ / 9⌉`. `(n · 7282) >> 16 == n / 9` exactly for every `n < 32768`, and
/// the vertical sum is at most `3 · 3 · 255 + 4`.
const DIV9_MUL: u16 = 7282;

/// Plain 3×3 **mean** (box) blur, separable. Cheapest smoother; blurs noise and
/// detail alike. Border uses edge-replicate.
pub(super) fn plane(src: &[u8], w: usize, h: usize) -> Vec<u8> {
    plane_tiered(src, w, h, Tier::detect())
}

/// [`plane`] at an explicit tier, so tests can hold every tier to the scalar one.
fn plane_tiered(src: &[u8], w: usize, h: usize, tier: Tier) -> Vec<u8> {
    // Horizontal 3-sum into u16 scratch, then vertical 3-sum / 9.
    let mut tmp = vec![0u16; w * h];
    for_row_bands(&mut tmp, w, 512, |y0, rows| {
        for (i, row) in rows.chunks_mut(w).enumerate() {
            hsum_row(tier, &src[(y0 + i) * w..][..w], row);
        }
    });
    let mut out = vec![0u8; w * h];
    for_row_bands(&mut out, w, 512, |y0, rows| {
        for (i, row) in rows.chunks_mut(w).enumerate() {
            vsum_row(tier, &tmp, w, h, y0 + i, row);
        }
    });
    out
}

fn hsum_span(src: &[u8], tmp: &mut [u16], range: std::ops::Range<usize>) {
    let w = src.len();
    for x in range {
        let l = clamp_idx(x as isize - 1, w);
        let r = clamp_idx(x as isize + 1, w);
        tmp[x] = src[l] as u16 + src[x] as u16 + src[r] as u16;
    }
}

fn hsum_scalar(src: &[u8], tmp: &mut [u16]) {
    hsum_span(src, tmp, 0..src.len());
}

#[cfg_attr(not(any(target_arch = "x86", target_arch = "x86_64")), allow(dead_code))]
#[inline(always)]
unsafe fn hsum_body<S: Simd>(src: &[u8], tmp: &mut [u16]) {
    unsafe {
        let w = src.len();
        let lo = 1.min(w);
        let hi = w.saturating_sub(1);
        hsum_span(src, tmp, 0..lo);
        let mut x = lo;
        while x + S::LANES16 <= hi {
            let p = src.as_ptr().add(x);
            let s = S::add_u16(
                S::add_u16(S::load_u8_u16(p.sub(1)), S::load_u8_u16(p)),
                S::load_u8_u16(p.add(1)),
            );
            S::store_u16(tmp.as_mut_ptr().add(x), s);
            x += S::LANES16;
        }
        hsum_span(src, tmp, x..w);
    }
}

tiered!(fn hsum_row(src: &[u8], tmp: &mut [u16]) => hsum_body, scalar hsum_scalar);

fn vsum_span(tmp: &[u16], w: usize, h: usize, y: usize, out: &mut [u8], range: std::ops::Range<usize>) {
    let u = clamp_idx(y as isize - 1, h);
    let d = clamp_idx(y as isize + 1, h);
    for x in range {
        out[x] = ((tmp[u * w + x] + tmp[y * w + x] + tmp[d * w + x] + 4) / 9) as u8;
    }
}

fn vsum_scalar(tmp: &[u16], w: usize, h: usize, y: usize, out: &mut [u8]) {
    vsum_span(tmp, w, h, y, out, 0..w);
}

#[cfg_attr(not(any(target_arch = "x86", target_arch = "x86_64")), allow(dead_code))]
#[inline(always)]
unsafe fn vsum_body<S: Simd>(tmp: &[u16], w: usize, h: usize, y: usize, out: &mut [u8]) {
    unsafe {
        let up = tmp.as_ptr().add(clamp_idx(y as isize - 1, h) * w);
        let mid = tmp.as_ptr().add(y * w);
        let down = tmp.as_ptr().add(clamp_idx(y as isize + 1, h) * w);
        let four = S::set1_u16(4);
        let m = S::set1_u16(DIV9_MUL);
        let mut x = 0;
        while x + S::LANES16 <= w {
            let n = S::add_u16(
                S::add_u16(S::add_u16(S::load_u16(up.add(x)), S::load_u16(mid.add(x))), S::load_u16(down.add(x))),
                four,
            );
            S::store_u16_u8(out.as_mut_ptr().add(x), S::mulhi_u16(n, m));
            x += S::LANES16;
        }
        vsum_span(tmp, w, h, y, out, x..w);
    }
}

tiered!(fn vsum_row(tmp: &[u16], w: usize, h: usize, y: usize, out: &mut [u8]) => vsum_body, scalar vsum_scalar);

#[cfg(test)]
mod tests {
    use super::super::test_support::planes;
    use super::*;

    #[test]
    fn the_multiply_high_divides_by_nine_exactly_over_the_whole_range() {
        for n in 0..=(3 * 3 * 255 + 4) as u32 {
            assert_eq!((n * DIV9_MUL as u32) >> 16, n / 9, "n = {n}");
        }
    }

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
