//! Non-local means denoise — highest classical quality, slowest.

use super::clamp_idx;

/// **Non-local means**: each output sample is an average of the samples in a 7×7
/// search window, weighted by the SSD between the 3×3 patch around the centre and
/// the 3×3 patch around each candidate — so samples whose *surroundings* look
/// like the centre's contribute most. Denoises repeating texture without
/// blurring it, at the cost of being the slowest method here (~`49 × 9` ops per
/// output sample). Border uses edge-replicate.
pub(super) fn plane(src: &[u8], w: usize, h: usize) -> Vec<u8> {
    const SR: isize = 3; // 7×7 search window
    const PR: isize = 1; // 3×3 patch
    const PN: f32 = ((2 * PR + 1) * (2 * PR + 1)) as f32;
    let h_param = 10.0f32; // filter strength (decay of the patch-distance weight)
    let h2 = h_param * h_param;
    let at = |xx: isize, yy: isize| src[clamp_idx(yy, h) * w + clamp_idx(xx, w)] as i32;
    let patch_ssd = |x1: isize, y1: isize, x2: isize, y2: isize| -> f32 {
        let mut s = 0i32;
        for py in -PR..=PR {
            for px in -PR..=PR {
                let d = at(x1 + px, y1 + py) - at(x2 + px, y2 + py);
                s += d * d;
            }
        }
        s as f32 / PN
    };
    let mut out = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            let (xi, yi) = (x as isize, y as isize);
            let mut sum = 0f32;
            let mut wsum = 0f32;
            for dy in -SR..=SR {
                for dx in -SR..=SR {
                    let dist = patch_ssd(xi, yi, xi + dx, yi + dy);
                    let wt = (-dist / h2).exp();
                    sum += wt * at(xi + dx, yi + dy) as f32;
                    wsum += wt;
                }
            }
            out[y * w + x] = (sum / wsum).round().clamp(0.0, 255.0) as u8;
        }
    }
    out
}
