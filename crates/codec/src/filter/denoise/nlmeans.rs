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
    plane_params_threaded(src, w, h, patch, research, sigma, band_count(h))
}

/// How many row bands to split the plane into — one per available core, capped
/// so a band is never thinner than the vertical halo each one has to
/// re-materialise (`MIN_BAND_ROWS`), which is where the split stops paying.
fn band_count(h: usize) -> usize {
    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    const MIN_BAND_ROWS: usize = 32;
    cores.min(h.div_ceil(MIN_BAND_ROWS)).max(1)
}

/// [`plane_params`] with an explicit band count, so tests can prove the split
/// doesn't change the result.
fn plane_params_threaded(
    src: &[u8],
    w: usize,
    h: usize,
    patch: u32,
    research: u32,
    sigma: f32,
    bands: usize,
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

    let params = BandParams {
        w,
        h,
        pr,
        rr,
        max_meaningful_diff,
        pdiff_lut_scale,
    };

    let mut out = vec![0u8; w * h];
    let bands = bands.max(1);
    let rows_per_band = h.div_ceil(bands);

    // Bands are independent: each writes only its own output rows and rebuilds
    // the halo rows it needs, so there is nothing to synchronise and nothing to
    // reduce afterwards. Output is identical to the single-band result.
    std::thread::scope(|scope| {
        for (band_idx, out_rows) in out.chunks_mut(rows_per_band * w).enumerate() {
            let y0 = band_idx * rows_per_band;
            let lut = &weight_lut;
            scope.spawn(move || denoise_band(src, &params, y0, out_rows, lut));
        }
    });

    out
}

/// The per-band constants, bundled so the worker closure captures one thing.
#[derive(Clone, Copy)]
struct BandParams {
    w: usize,
    h: usize,
    pr: isize,
    rr: isize,
    max_meaningful_diff: f32,
    pdiff_lut_scale: f32,
}

/// Denoise output rows `y0 .. y0 + out_rows.len()/w` into `out_rows`.
///
/// Builds its summed-area table over only the rows it needs — the band plus a
/// `pr` halo on each side — rather than the whole plane. That is what makes the
/// split possible, and it's also faster on its own: a full-plane SAT for 1080p
/// is 16 MiB per offset and streams through cache badly, where a band's is a
/// few MiB and stays resident.
fn denoise_band(src: &[u8], p: &BandParams, y0: usize, out_rows: &mut [u8], weight_lut: &[f32]) {
    let (w, h, pr, rr) = (p.w, p.h, p.pr, p.rr);
    let band_h = out_rows.len() / w;
    if band_h == 0 {
        return;
    }
    let y_end = y0 + band_h;

    // Source rows this band's patch windows can reach.
    let lo = (y0 as isize - pr).max(0) as usize;
    let hi = ((y_end as isize + pr) as usize).min(h);
    let sat_rows = hi - lo;

    let iw = w + 1;
    // `u32`, not `u64`, and deliberately allowed to wrap.
    //
    // The table itself overflows a u32 easily — a 1080p band accumulates ~2e10 —
    // but nothing ever reads a table entry. Only the four-corner *difference* is
    // read, and that is exact modulo 2^32, so it is exact full stop as long as
    // the true window sum fits: the largest possible is 255² x 99 x 99 = 6.4e8
    // for the biggest patch ffmpeg allows, well inside u32. Halving the table
    // halves the memory traffic of the inner loop, which is what this kernel is
    // bound by.
    let mut sat = vec![0u32; iw * (sat_rows + 1)];
    let mut sum = vec![0f32; band_h * w];
    let mut wsum = vec![0f32; band_h * w];

    for dy in -rr..=rr {
        for dx in -rr..=rr {
            // The centre offset compares every sample with itself: SSD is zero
            // everywhere, so the weight is exactly `weight_lut[0]` and there is
            // no table to build. At the minimum useful window (r=3) that is one
            // offset in nine.
            if dy == 0 && dx == 0 {
                let w0 = weight_lut[0];
                for y in y0..y_end {
                    let orow = (y - y0) * w;
                    for x in 0..w {
                        sum[orow + x] += w0 * src[y * w + x] as f32;
                        wsum[orow + x] += w0;
                    }
                }
                continue;
            }

            // Squared-difference SAT for this offset, over the band's rows only.
            for r in 0..sat_rows {
                let y = lo + r;
                let sy = clamp_idx(y as isize + dy, h);
                let mut row_acc = 0u32;
                for x in 0..w {
                    let sx = clamp_idx(x as isize + dx, w);
                    let d = src[y * w + x] as i32 - src[sy * w + sx] as i32;
                    row_acc = row_acc.wrapping_add((d * d) as u32);
                    sat[(r + 1) * iw + (x + 1)] =
                        sat[r * iw + (x + 1)].wrapping_add(row_acc);
                }
            }

            for y in y0..y_end {
                // Patch rows, clamped to the plane then rebased onto the SAT.
                let ya = (y as isize - pr).clamp(0, h as isize) as usize - lo;
                let yb = (y as isize + pr + 1).clamp(0, h as isize) as usize - lo;
                let sy = clamp_idx(y as isize + dy, h);
                let orow = (y - y0) * w;
                for x in 0..w {
                    let xa = (x as isize - pr).clamp(0, w as isize) as usize;
                    let xb = (x as isize + pr + 1).clamp(0, w as isize) as usize;
                    // Wrapping throughout — see the note on `sat`.
                    let ssd = sat[yb * iw + xb]
                        .wrapping_add(sat[ya * iw + xa])
                        .wrapping_sub(sat[ya * iw + xb])
                        .wrapping_sub(sat[yb * iw + xa]) as f32;
                    if ssd >= p.max_meaningful_diff {
                        continue;
                    }
                    let wt =
                        weight_lut[((ssd * p.pdiff_lut_scale) as usize).min(WEIGHT_LUT_NB - 1)];
                    let sx = clamp_idx(x as isize + dx, w);
                    sum[orow + x] += wt * src[sy * w + sx] as f32;
                    wsum[orow + x] += wt;
                }
            }
        }
    }

    for i in 0..band_h * w {
        out_rows[i] = if wsum[i] > 0.0 {
            (sum[i] / wsum[i]).round().clamp(0.0, 255.0) as u8
        } else {
            src[y0 * w + i]
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-noise so the comparison has real content to chew on.
    fn noisy(n: usize) -> Vec<u8> {
        let mut s = 0x2545_F491u32;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((s >> 16) & 0xFF) as u8
            })
            .collect()
    }

    #[test]
    fn band_split_does_not_change_the_result() {
        // The whole point of splitting by rows: each band rebuilds the halo it
        // needs, so the output must be bit-identical however many bands there
        // are. A halo off by one row would show up here and nowhere else.
        let (w, h) = (61, 47); // deliberately not a multiple of any band count
        let src = noisy(w * h);
        let single = plane_params_threaded(&src, w, h, 7, 5, 3.0, 1);
        for bands in [2usize, 3, 4, 7, 16, 64] {
            let split = plane_params_threaded(&src, w, h, 7, 5, 3.0, bands);
            assert_eq!(single, split, "{bands} bands changed the output");
        }
    }

    #[test]
    fn band_split_holds_for_a_large_patch_and_window() {
        // A patch wider than a band's own rows forces the halo to span several
        // bands, which is the case most likely to be got wrong.
        let (w, h) = (40, 33);
        let src = noisy(w * h);
        let single = plane_params_threaded(&src, w, h, 15, 9, 6.0, 1);
        for bands in [2usize, 5, 33] {
            assert_eq!(
                single,
                plane_params_threaded(&src, w, h, 15, 9, 6.0, bands),
                "{bands} bands changed the output at patch 15 / window 9"
            );
        }
    }

    #[test]
    fn a_single_row_plane_still_works() {
        let src = noisy(20);
        let a = plane_params_threaded(&src, 20, 1, 7, 5, 3.0, 1);
        let b = plane_params_threaded(&src, 20, 1, 7, 5, 3.0, 8);
        assert_eq!(a, b);
        assert_eq!(a.len(), 20);
    }

    #[test]
    fn band_count_is_bounded_by_the_plane_height() {
        // No point spawning a thread per two rows.
        assert_eq!(band_count(1), 1);
        assert_eq!(band_count(16), 1);
        assert!(band_count(1080) >= 1);
    }
}
