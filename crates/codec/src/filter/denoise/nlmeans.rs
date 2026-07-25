//! Non-local means denoise — highest classical quality, slowest.
//!
//! Two entry points:
//!
//! - [`plane`] — the fixed-parameter kernel behind `denoise=nlmeans:STRENGTH`,
//!   where every method runs at a moderate internal setting and `strength` only
//!   blends the result back with the source.
//! - [`plane_params`] — the **parameterized** kernel behind the ffmpeg-compatible
//!   `nlmeans=s=..:p=..:r=..` filter, where the patch size, research-window size,
//!   and σ-style strength are the caller's to choose.

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

/// Number of entries in the patch-distance → weight lookup table. Mirrors
/// ffmpeg's `WEIGHT_LUT_NB`; the table spans `0..max_meaningful_diff`, past
/// which the weight has decayed below `1/255` and the candidate is skipped.
const WEIGHT_LUT_NB: usize = 1 << 9;

/// ffmpeg forces the patch / research **sizes** odd (`size |= 1`), so a 0 or an
/// even value becomes the next odd one up. The kernel works in radii.
fn odd_radius(size: u32) -> isize {
    ((size | 1) / 2) as isize
}

/// **Parameterized non-local means**, matching `ffmpeg -vf nlmeans` semantics.
///
/// For every offset in the `research`×`research` window, each sample is weighted
/// by `exp(-SSD / (s·10)²)`, where `SSD` is the *sum* of squared differences over
/// the `patch`×`patch` neighbourhood (not the mean — this is what makes ffmpeg's
/// `s` range `1.0..=30.0` behave the way it does: `s=1` is a light touch because
/// the sum over a 7×7 patch decays the exponent fast).
///
/// `patch` and `research` are **sizes in samples**, as on the ffmpeg command
/// line; they're forced odd and halved into radii here. Border addressing is
/// edge-replicate.
///
/// The per-offset SSD is evaluated through a **summed-area table** of the squared
/// difference plane, so the cost is `O(research² · w · h)` rather than
/// `O(research² · patch² · w · h)` — the patch size is free. Still an offline
/// filter: the default `r=15` is 225 offsets, i.e. 450 passes over the plane.
pub(super) fn plane_params(
    src: &[u8],
    w: usize,
    h: usize,
    patch: u32,
    research: u32,
    sigma: f32,
) -> Vec<u8> {
    if w == 0 || h == 0 {
        return src.to_vec();
    }
    let pr = odd_radius(patch);
    let rr = odd_radius(research);
    // A 1×1 research window only ever sees the centre (SSD 0, weight 1), so the
    // output is the input — skip the work.
    if rr == 0 {
        return src.to_vec();
    }

    // ffmpeg: `h = sigma * 10`, `pdiff_scale = 1/h²`, and a candidate whose patch
    // distance exceeds `ln(255)·h²` contributes less than one 8-bit step.
    let h_param = (sigma.max(0.01)) * 10.0;
    let pdiff_scale = 1.0 / (h_param * h_param);
    let max_meaningful_diff = (255.0f32).ln() / pdiff_scale;
    let pdiff_lut_scale = WEIGHT_LUT_NB as f32 / max_meaningful_diff;
    let weight_lut: Vec<f32> = (0..WEIGHT_LUT_NB)
        .map(|i| (-(i as f32) / pdiff_lut_scale * pdiff_scale).exp())
        .collect();

    let n = w * h;
    let mut sum = vec![0f32; n];
    let mut wsum = vec![0f32; n];
    // Summed-area table of the squared-difference plane for the current offset:
    // `(w+1)×(h+1)` with a zero first row/column so the four-corner lookup needs
    // no bounds special-casing. Allocated once and overwritten per offset.
    let iw = w + 1;
    let mut sat = vec![0u64; iw * (h + 1)];

    for dy in -rr..=rr {
        for dx in -rr..=rr {
            for y in 0..h {
                let sy = clamp_idx(y as isize + dy, h);
                let mut row_acc = 0u64;
                for x in 0..w {
                    let sx = clamp_idx(x as isize + dx, w);
                    let d = src[y * w + x] as i32 - src[sy * w + sx] as i32;
                    row_acc += (d * d) as u64;
                    sat[(y + 1) * iw + (x + 1)] = sat[y * iw + (x + 1)] + row_acc;
                }
            }
            for y in 0..h {
                let y0 = (y as isize - pr).clamp(0, h as isize) as usize;
                let y1 = (y as isize + pr + 1).clamp(0, h as isize) as usize;
                let sy = clamp_idx(y as isize + dy, h);
                for x in 0..w {
                    let x0 = (x as isize - pr).clamp(0, w as isize) as usize;
                    let x1 = (x as isize + pr + 1).clamp(0, w as isize) as usize;
                    let ssd = (sat[y1 * iw + x1] + sat[y0 * iw + x0]
                        - sat[y0 * iw + x1]
                        - sat[y1 * iw + x0]) as f32;
                    if ssd >= max_meaningful_diff {
                        continue;
                    }
                    let wt = weight_lut[((ssd * pdiff_lut_scale) as usize).min(WEIGHT_LUT_NB - 1)];
                    let sx = clamp_idx(x as isize + dx, w);
                    sum[y * w + x] += wt * src[sy * w + sx] as f32;
                    wsum[y * w + x] += wt;
                }
            }
        }
    }

    (0..n)
        .map(|i| {
            if wsum[i] > 0.0 {
                (sum[i] / wsum[i]).round().clamp(0.0, 255.0) as u8
            } else {
                src[i]
            }
        })
        .collect()
}
