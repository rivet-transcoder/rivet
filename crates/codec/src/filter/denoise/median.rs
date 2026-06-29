//! Median denoise (impulse / salt-and-pepper).

use super::clamp_idx;

/// 3×3 median filter — replaces each sample with the median of its 3×3
/// neighbourhood, which removes isolated impulse (salt-and-pepper) samples
/// outright while leaving edges intact. Border uses edge-replicate.
pub(super) fn plane(src: &[u8], w: usize, h: usize) -> Vec<u8> {
    let mut out = vec![0u8; w * h];
    let mut window = [0u8; 9];
    for y in 0..h {
        for x in 0..w {
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
            out[y * w + x] = window[4]; // median of 9
        }
    }
    out
}
