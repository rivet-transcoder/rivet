//! Gaussian (separable low-pass) denoise.

use super::clamp_idx;

/// Separable 5-tap Gaussian blur (σ≈1.0, kernel `[1,4,6,4,1]/16`) — a plain
/// low-pass that smooths noise and detail alike. Border uses edge-replicate.
pub(super) fn plane(src: &[u8], w: usize, h: usize) -> Vec<u8> {
    const K: [f32; 5] = [1.0, 4.0, 6.0, 4.0, 1.0];
    const KSUM: f32 = 16.0;
    const R: isize = 2;
    // Horizontal pass → f32 scratch.
    let mut tmp = vec![0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0f32;
            for (k, &kw) in K.iter().enumerate() {
                let xx = clamp_idx(x as isize + k as isize - R, w);
                acc += kw * src[y * w + xx] as f32;
            }
            tmp[y * w + x] = acc / KSUM;
        }
    }
    // Vertical pass → u8.
    let mut out = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0f32;
            for (k, &kw) in K.iter().enumerate() {
                let yy = clamp_idx(y as isize + k as isize - R, h);
                acc += kw * tmp[yy * w + x];
            }
            out[y * w + x] = (acc / KSUM).round().clamp(0.0, 255.0) as u8;
        }
    }
    out
}
