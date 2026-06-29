//! Anisotropic diffusion (Perona–Malik) denoise — edge-preserving.

use super::clamp_idx;

/// **Anisotropic diffusion** (Perona–Malik): iterate `u += λ·Σ g(∇)·∇` over the
/// 4-neighbour gradients, where the conduction `g(∇) = exp(−(∇/κ)²)` falls to
/// ~0 at strong gradients — so the image diffuses (smooths) inside flat regions
/// but the flow stops at edges. 8 iterations, `λ = 0.20` (≤ ¼ for 4-neighbour
/// stability), `κ = 20`. Border uses edge-replicate.
pub(super) fn plane(src: &[u8], w: usize, h: usize) -> Vec<u8> {
    const ITERS: usize = 8;
    let kappa = 20.0f32;
    let lambda = 0.20f32;
    let g = |grad: f32| {
        let q = grad / kappa;
        (-(q * q)).exp()
    };
    let mut img: Vec<f32> = src.iter().map(|&v| v as f32).collect();
    let mut next = img.clone();
    for _ in 0..ITERS {
        for y in 0..h {
            for x in 0..w {
                let c = img[y * w + x];
                let n = img[clamp_idx(y as isize - 1, h) * w + x] - c;
                let s = img[clamp_idx(y as isize + 1, h) * w + x] - c;
                let e = img[y * w + clamp_idx(x as isize + 1, w)] - c;
                let we = img[y * w + clamp_idx(x as isize - 1, w)] - c;
                next[y * w + x] = c + lambda * (g(n) * n + g(s) * s + g(e) * e + g(we) * we);
            }
        }
        std::mem::swap(&mut img, &mut next);
    }
    img.iter().map(|&v| v.round().clamp(0.0, 255.0) as u8).collect()
}
