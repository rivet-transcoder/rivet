//! Mean (box) denoise — the cheapest smoother.

use super::clamp_idx;

/// Plain 3×3 **mean** (box) blur, separable. Cheapest smoother; blurs noise and
/// detail alike. Border uses edge-replicate.
pub(super) fn plane(src: &[u8], w: usize, h: usize) -> Vec<u8> {
    // Horizontal 3-sum into u16 scratch, then vertical 3-sum / 9.
    let mut tmp = vec![0u16; w * h];
    for y in 0..h {
        for x in 0..w {
            let l = clamp_idx(x as isize - 1, w);
            let r = clamp_idx(x as isize + 1, w);
            tmp[y * w + x] = src[y * w + l] as u16 + src[y * w + x] as u16 + src[y * w + r] as u16;
        }
    }
    let mut out = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            let u = clamp_idx(y as isize - 1, h);
            let d = clamp_idx(y as isize + 1, h);
            out[y * w + x] = ((tmp[u * w + x] + tmp[y * w + x] + tmp[d * w + x] + 4) / 9) as u8;
        }
    }
    out
}
