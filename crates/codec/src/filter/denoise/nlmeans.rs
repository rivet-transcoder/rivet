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

use std::sync::OnceLock;

use super::simd::{Simd, Tier, tiered};
use super::{clamp_idx, max_threads};

// ── the fixed-parameter kernel (`denoise=nlmeans`) ───────────────────────────

/// Search-window radius (7×7).
const SR: isize = 3;
/// Patch radius (3×3).
const PR: isize = 1;
const PRU: usize = PR as usize;
/// Samples in a patch.
const PN: f32 = ((2 * PR + 1) * (2 * PR + 1)) as f32;
/// Filter strength (decay of the patch-distance weight), squared.
const H2: f32 = 10.0 * 10.0;
/// The largest patch SSD: every sample differs by 255.
const SSD_MAX: usize = ((2 * PR + 1) * (2 * PR + 1) * 255 * 255) as usize;

/// The weight for every integer patch SSD, evaluated by the reference
/// expression `exp(−(ssd / 9) / h²)` on that integer — so a lookup gives the
/// same bits the reference computes. The table stops where the weight reaches
/// zero and carries one trailing zero, so `lut[min(ssd, len − 1)]` is the
/// weight for any SSD. Built once per process (~0.4 MB).
fn weight_lut() -> &'static [f32] {
    static LUT: OnceLock<Vec<f32>> = OnceLock::new();
    LUT.get_or_init(|| {
        let full: Vec<f32> = (0..=SSD_MAX)
            .map(|ssd| {
                let dist = ssd as f32 / PN;
                (-dist / H2).exp()
            })
            .collect();
        let last_nonzero = full.iter().rposition(|&w| w != 0.0).map_or(0, |i| i + 1);
        let mut lut = full[..last_nonzero].to_vec();
        lut.push(0.0);
        lut
    })
}

/// **Non-local means**: each output sample is an average of the samples in a 7×7
/// search window, weighted by the SSD between the 3×3 patch around the centre and
/// the 3×3 patch around each candidate — so samples whose *surroundings* look
/// like the centre's contribute most. Denoises repeating texture without
/// blurring it. Border uses edge-replicate.
///
/// Evaluated offset by offset through a summed-area table of the squared
/// differences, on row bands across the cores, with SIMD row kernels — the
/// same shape as [`plane_params`] — but weighting exactly as the direct
/// per-sample loop it replaced does: the SSD is an integer, the weight is that
/// integer's entry in [`weight_lut`], and the 49 contributions are summed in
/// the same order, so the output is bit-identical to the reference
/// (`plane_reference`, kept under test).
pub(super) fn plane(src: &[u8], w: usize, h: usize) -> Vec<u8> {
    plane_fixed(src, w, h, Tier::detect(), band_count(h))
}

/// [`plane`] at an explicit tier and band count.
fn plane_fixed(src: &[u8], w: usize, h: usize, tier: Tier, bands: usize) -> Vec<u8> {
    if w == 0 || h == 0 {
        return src.to_vec();
    }
    let lut = weight_lut();
    // The plane with a `PR`-wide edge-replicated border on every side. Every
    // patch — centre and candidate — is then a rectangle inside it, exactly
    // the reference's clamped addressing, and nothing in the kernels clamps.
    let (pw, ph) = (w + 2 * PRU, h + 2 * PRU);
    let mut pad = vec![0u8; pw * ph];
    for py in 0..ph {
        let row = &src[clamp_idx(py as isize - PR, h) * w..][..w];
        let prow = &mut pad[py * pw..][..pw];
        prow[PRU..PRU + w].copy_from_slice(row);
        prow[..PRU].fill(row[0]);
        prow[PRU + w..].fill(row[w - 1]);
    }

    let mut out = vec![0u8; w * h];
    let rows_per_band = h.div_ceil(bands.max(1));
    std::thread::scope(|scope| {
        for (band_idx, out_rows) in out.chunks_mut(rows_per_band * w).enumerate() {
            let pad = &pad;
            scope.spawn(move || fixed_band(pad, w, h, band_idx * rows_per_band, out_rows, lut, tier));
        }
    });
    out
}

/// Denoise output rows `y0 .. y0 + out_rows.len() / w` of the padded plane.
fn fixed_band(pad: &[u8], w: usize, h: usize, y0: usize, out_rows: &mut [u8], lut: &[f32], tier: Tier) {
    let pw = w + 2 * PRU;
    let band_h = out_rows.len() / w;
    if band_h == 0 {
        return;
    }
    let y_end = y0 + band_h;
    // Padded rows this band's patch windows reach: output row `y` is padded
    // row `y + PR`, and its patch spans `y ..= y + 2·PR`.
    let lo = y0;
    let hi = y_end + 2 * PRU;
    let sat_rows = hi - lo;
    let iw = pw + 1;
    let mut sat = vec![0u32; iw * (sat_rows + 1)];
    // For each padded row, the candidate row at this offset: what the patch
    // difference is taken against, and what gets averaged in.
    let mut offrows = vec![0u8; sat_rows * pw];
    let mut sum = vec![0f32; band_h * w];
    let mut wsum = vec![0f32; band_h * w];

    for dy in -SR..=SR {
        for dx in -SR..=SR {
            for r in 0..sat_rows {
                let py = clamp_idx((lo + r) as isize + dy - PR, h) + PRU;
                shift_row(&pad[py * pw..][..pw], dx, w, &mut offrows[r * pw..][..pw]);
            }
            for r in 0..sat_rows {
                let (done, rest) = sat.split_at_mut((r + 1) * iw);
                sat_row_fixed(
                    tier,
                    &pad[(lo + r) * pw..][..pw],
                    &offrows[r * pw..][..pw],
                    &done[r * iw..][..iw],
                    &mut rest[..iw],
                );
            }
            for y in y0..y_end {
                let ya = y - lo;
                let yb = ya + 2 * PRU + 1;
                let orow = (y - y0) * w;
                accumulate_fixed(
                    tier,
                    lut,
                    &sat[ya * iw..][..iw],
                    &sat[yb * iw..][..iw],
                    &offrows[(y + PRU - lo) * pw..][..pw],
                    &mut sum[orow..][..w],
                    &mut wsum[orow..][..w],
                );
            }
        }
    }

    for i in 0..band_h * w {
        out_rows[i] = (sum[i] / wsum[i]).round().clamp(0.0, 255.0) as u8;
    }
}

/// `out[X] = row[clampP(X + dx)]` over a padded row, where `clampP` clamps
/// the *unpadded* column and re-pads — the reference's edge-replicate for the
/// candidate, expressed as a shifted copy so the kernels read it contiguously.
fn shift_row(row: &[u8], dx: isize, w: usize, out: &mut [u8]) {
    for (x, o) in out.iter_mut().enumerate() {
        *o = row[clamp_idx(x as isize + dx - PR, w) + PRU];
    }
}

fn sat_row_fixed_scalar(cur: &[u8], off: &[u8], prev: &[u32], next: &mut [u32]) {
    sat_span(cur, off, 0, prev, next, 0..cur.len(), &mut 0);
}

/// One row of the squared-difference summed-area table, `LANES` columns at a
/// time: square the differences, prefix-sum the lanes, add the running total
/// and the row above.
#[cfg_attr(not(any(target_arch = "x86", target_arch = "x86_64")), allow(dead_code))]
#[inline(always)]
unsafe fn sat_row_fixed_body<S: Simd>(cur: &[u8], off: &[u8], prev: &[u32], next: &mut [u32]) {
    unsafe {
        let w = cur.len();
        let mut acc = 0u32;
        let mut x = 0;
        while x + S::LANES <= w {
            let d = S::sub_i32(S::load_u8_i32(cur.as_ptr().add(x)), S::load_u8_i32(off.as_ptr().add(x)));
            let s = S::prefix_sum_i32(S::mullo_i32(d, d));
            let s = S::add_i32(s, S::set1_i32(acc as i32));
            acc = S::extract_last_i32(s) as u32;
            S::store_u32(next.as_mut_ptr().add(x + 1), S::add_i32(S::load_u32(prev.as_ptr().add(x + 1)), s));
            x += S::LANES;
        }
        sat_span(cur, off, 0, prev, next, x..w, &mut acc);
    }
}

tiered!(fn sat_row_fixed(cur: &[u8], off: &[u8], prev: &[u32], next: &mut [u32]) => sat_row_fixed_body, scalar sat_row_fixed_scalar);

/// The accumulate for one output row at one offset, scalar: four-corner SSD,
/// table weight, running sums.
fn accumulate_fixed_span(
    lut: &[f32],
    sat_a: &[u32],
    sat_b: &[u32],
    cand: &[u8],
    sum: &mut [f32],
    wsum: &mut [f32],
    range: std::ops::Range<usize>,
) {
    let cut = lut.len() - 1;
    for x in range {
        let xb = x + 2 * PRU + 1;
        let ssd = sat_b[xb].wrapping_add(sat_a[x]).wrapping_sub(sat_a[xb]).wrapping_sub(sat_b[x]) as usize;
        let wt = lut[ssd.min(cut)];
        sum[x] += wt * cand[x + PRU] as f32;
        wsum[x] += wt;
    }
}

fn accumulate_fixed_scalar(lut: &[f32], sat_a: &[u32], sat_b: &[u32], cand: &[u8], sum: &mut [f32], wsum: &mut [f32]) {
    accumulate_fixed_span(lut, sat_a, sat_b, cand, sum, wsum, 0..sum.len());
}

#[cfg_attr(not(any(target_arch = "x86", target_arch = "x86_64")), allow(dead_code))]
#[inline(always)]
unsafe fn accumulate_fixed_body<S: Simd>(
    lut: &[f32],
    sat_a: &[u32],
    sat_b: &[u32],
    cand: &[u8],
    sum: &mut [f32],
    wsum: &mut [f32],
) {
    unsafe {
        let w = sum.len();
        let cut = S::set1_i32((lut.len() - 1) as i32);
        let span = 2 * PRU + 1;
        let mut x = 0;
        while x + S::LANES <= w {
            let a_lo = S::load_u32(sat_a.as_ptr().add(x));
            let a_hi = S::load_u32(sat_a.as_ptr().add(x + span));
            let b_lo = S::load_u32(sat_b.as_ptr().add(x));
            let b_hi = S::load_u32(sat_b.as_ptr().add(x + span));
            // Exact: the window sum is at most `SSD_MAX`, far inside i32.
            let ssd = S::sub_i32(S::sub_i32(S::add_i32(b_hi, a_lo), a_hi), b_lo);
            let wt = S::gather_f32(lut, S::min_i32(ssd, cut));
            let v = S::load_u8_f32(cand.as_ptr().add(x + PRU));
            // Separate multiply and add, deliberately not an FMA.
            S::store_f32(sum.as_mut_ptr().add(x), S::add_f32(S::load_f32(sum.as_ptr().add(x)), S::mul_f32(wt, v)));
            S::store_f32(wsum.as_mut_ptr().add(x), S::add_f32(S::load_f32(wsum.as_ptr().add(x)), wt));
            x += S::LANES;
        }
        accumulate_fixed_span(lut, sat_a, sat_b, cand, sum, wsum, x..w);
    }
}

tiered!(fn accumulate_fixed(lut: &[f32], sat_a: &[u32], sat_b: &[u32], cand: &[u8], sum: &mut [f32], wsum: &mut [f32]) => accumulate_fixed_body, scalar accumulate_fixed_scalar);

/// The direct per-sample loop the fixed-parameter kernel replaced — the
/// specification [`plane`] is held to, bit for bit.
#[cfg(test)]
fn plane_reference(src: &[u8], w: usize, h: usize) -> Vec<u8> {
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
/// `O(research² · patch² · w · h)` — the patch size is free. On top of that the
/// plane is split into row bands across the available cores, and the two inner
/// row loops have AVX2 kernels. Even so, the default `r=15` is 225 offsets and
/// stays offline-tier; `r=3` is the setting that runs at a useful rate.
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
    const MIN_BAND_ROWS: usize = 32;
    max_threads().min(h.div_ceil(MIN_BAND_ROWS)).max(1)
}

/// Whether the AVX2 kernels run: the host has them and `RIVET_DENOISE_MAX_SIMD`
/// has not capped the family below them (this kernel has no 128-bit form, so
/// a cap to `sse41` runs it scalar). Resolved once per plane and carried in
/// [`BandParams`], not re-detected per row.
fn have_avx2() -> bool {
    Tier::detect() >= Tier::Avx2
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
    plane_params_with(src, w, h, patch, research, sigma, bands, have_avx2())
}

/// [`plane_params_threaded`] with the SIMD path forced on or off, so tests can
/// prove the two agree sample for sample.
#[allow(clippy::too_many_arguments)]
fn plane_params_with(
    src: &[u8],
    w: usize,
    h: usize,
    patch: u32,
    research: u32,
    sigma: f32,
    bands: usize,
    simd: bool,
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
        simd,
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
    /// Take the AVX2 kernels. See [`have_avx2`].
    simd: bool,
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
                // Row `r` is read while row `r + 1` is written, so split the
                // table between them rather than indexing it twice.
                let (done, rest) = sat.split_at_mut((r + 1) * iw);
                sat_row(
                    &src[y * w..][..w],
                    &src[sy * w..][..w],
                    dx,
                    &done[r * iw..][..iw],
                    &mut rest[..iw],
                    p.simd,
                );
            }

            for y in y0..y_end {
                // Patch rows, clamped to the plane then rebased onto the SAT.
                let ya = (y as isize - pr).clamp(0, h as isize) as usize - lo;
                let yb = (y as isize + pr + 1).clamp(0, h as isize) as usize - lo;
                let sy = clamp_idx(y as isize + dy, h);
                let orow = (y - y0) * w;
                accumulate_row(
                    p,
                    dx,
                    weight_lut,
                    &sat[ya * iw..][..iw],
                    &sat[yb * iw..][..iw],
                    &src[sy * w..][..w],
                    &mut sum[orow..][..w],
                    &mut wsum[orow..][..w],
                );
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

// ---------------------------------------------------------------------------
// Row kernels
//
// The two loops below are where essentially all the time goes, so each has an
// AVX2 form alongside the scalar one. The AVX2 forms are **bit-identical** to
// the scalar ones, not merely equivalent: same LUT, same truncation, and a
// separate multiply and add rather than an FMA, because the scalar `sum[x] +=
// wt * v` rounds twice and a fused multiply-add rounds once. Anything else and
// a file's checksum would depend on which machine encoded it. The test
// `the_avx2_path_matches_the_scalar_one_bit_for_bit` holds this.
//
// Both kernels vectorise only the columns where no index expression clamps —
// the source offset `x + dx` and, for the accumulate, the patch edges `x ± pr`.
// Outside that span the reads aren't contiguous, so the borders stay scalar.
// ---------------------------------------------------------------------------

/// The scalar SAT recurrence over `range`, carrying the running row sum in
/// `acc`: `next[x + 1] = prev[x + 1] + Σ (cur[i] − off[i + dx])²`.
#[inline]
fn sat_span(
    cur: &[u8],
    off: &[u8],
    dx: isize,
    prev: &[u32],
    next: &mut [u32],
    range: std::ops::Range<usize>,
    acc: &mut u32,
) {
    let w = cur.len();
    for x in range {
        let sx = clamp_idx(x as isize + dx, w);
        let d = cur[x] as i32 - off[sx] as i32;
        *acc = acc.wrapping_add((d * d) as u32);
        next[x + 1] = prev[x + 1].wrapping_add(*acc);
    }
}

/// One row of the squared-difference summed-area table.
fn sat_row(cur: &[u8], off: &[u8], dx: isize, prev: &[u32], next: &mut [u32], simd: bool) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if simd {
        // SAFETY: `simd` is `have_avx2()`, detected at runtime in
        // `plane_params_with`.
        unsafe { sat_row_avx2(cur, off, dx, prev, next) };
        return;
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    let _ = simd;

    sat_span(cur, off, dx, prev, next, 0..cur.len(), &mut 0);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn sat_row_avx2(cur: &[u8], off: &[u8], dx: isize, prev: &[u32], next: &mut [u32]) {
    unsafe {
        #[cfg(target_arch = "x86")]
        use std::arch::x86::*;
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::*;

        let w = cur.len();
        let wi = w as isize;
        // Columns where `x + dx` lands inside the row, so `off` can be read as a
        // contiguous vector.
        let lo = (-dx).max(0).min(wi) as usize;
        let hi = (wi - dx).clamp(0, wi) as usize;

        let mut acc = 0u32;
        sat_span(cur, off, dx, prev, next, 0..lo, &mut acc);

        let mut x = lo;
        while x + 8 <= hi {
            let a = _mm256_cvtepu8_epi32(_mm_loadl_epi64(cur.as_ptr().add(x) as *const __m128i));
            let b = _mm256_cvtepu8_epi32(_mm_loadl_epi64(
                off.as_ptr().add((x as isize + dx) as usize) as *const __m128i,
            ));
            let d = _mm256_sub_epi32(a, b);
            // d is in -255..=255, so d² can't overflow the low 32 bits.
            let mut s = _mm256_mullo_epi32(d, d);

            // Prefix-sum the eight lanes. `slli_si256` shifts within each 128-bit
            // half, so two shift-adds give a prefix per half; then broadcast the
            // low half's total (its lane 3) into the high half and add.
            s = _mm256_add_epi32(s, _mm256_slli_si256::<4>(s));
            s = _mm256_add_epi32(s, _mm256_slli_si256::<8>(s));
            let low_total =
                _mm256_shuffle_epi32::<0xFF>(_mm256_permute2x128_si256::<0x08>(s, s));
            s = _mm256_add_epi32(s, low_total);

            // Fold in the running total and hand the new one to the next block.
            s = _mm256_add_epi32(s, _mm256_set1_epi32(acc as i32));
            acc = _mm256_extract_epi32::<7>(s) as u32;

            // Vertical step: this row's prefix on top of the row above. Wrapping
            // throughout, as in the scalar path — see the note on `sat`.
            let above = _mm256_loadu_si256(prev.as_ptr().add(x + 1) as *const __m256i);
            _mm256_storeu_si256(
                next.as_mut_ptr().add(x + 1) as *mut __m256i,
                _mm256_add_epi32(above, s),
            );
            x += 8;
        }

        sat_span(cur, off, dx, prev, next, x..w, &mut acc);
    }
}

/// The scalar accumulate over `range`: four-corner SSD, weight, running sums.
#[allow(clippy::too_many_arguments)]
#[inline]
fn accumulate_span(
    p: &BandParams,
    dx: isize,
    lut: &[f32],
    sat_a: &[u32],
    sat_b: &[u32],
    off: &[u8],
    sum: &mut [f32],
    wsum: &mut [f32],
    range: std::ops::Range<usize>,
) {
    let (w, pr) = (p.w, p.pr);
    for x in range {
        let xa = (x as isize - pr).clamp(0, w as isize) as usize;
        let xb = (x as isize + pr + 1).clamp(0, w as isize) as usize;
        // Wrapping throughout — see the note on `sat`.
        let ssd = sat_b[xb]
            .wrapping_add(sat_a[xa])
            .wrapping_sub(sat_a[xb])
            .wrapping_sub(sat_b[xa]) as f32;
        if ssd >= p.max_meaningful_diff {
            continue;
        }
        let wt = lut[((ssd * p.pdiff_lut_scale) as usize).min(WEIGHT_LUT_NB - 1)];
        let sx = clamp_idx(x as isize + dx, w);
        sum[x] += wt * off[sx] as f32;
        wsum[x] += wt;
    }
}

/// One output row's worth of weighted accumulation for a single offset.
#[allow(clippy::too_many_arguments)]
fn accumulate_row(
    p: &BandParams,
    dx: isize,
    lut: &[f32],
    sat_a: &[u32],
    sat_b: &[u32],
    off: &[u8],
    sum: &mut [f32],
    wsum: &mut [f32],
) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if p.simd {
        // SAFETY: `p.simd` is `have_avx2()`, detected at runtime in
        // `plane_params_with`.
        unsafe { accumulate_row_avx2(p, dx, lut, sat_a, sat_b, off, sum, wsum) };
        return;
    }

    accumulate_span(p, dx, lut, sat_a, sat_b, off, sum, wsum, 0..p.w);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn accumulate_row_avx2(
    p: &BandParams,
    dx: isize,
    lut: &[f32],
    sat_a: &[u32],
    sat_b: &[u32],
    off: &[u8],
    sum: &mut [f32],
    wsum: &mut [f32],
) {
    unsafe {
        #[cfg(target_arch = "x86")]
        use std::arch::x86::*;
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::*;

        let (w, pr) = (p.w, p.pr);
        let wi = w as isize;
        // Columns where none of the three index expressions clamps: the patch
        // edges `x - pr` and `x + pr + 1`, and the source offset `x + dx`.
        let lo = pr.max(-dx).max(0).min(wi);
        let hi = (wi - pr).min(wi - dx).clamp(lo, wi);
        let (lo, hi, pru) = (lo as usize, hi as usize, pr as usize);

        accumulate_span(p, dx, lut, sat_a, sat_b, off, sum, wsum, 0..lo);

        let max_v = _mm256_set1_ps(p.max_meaningful_diff);
        let scale_v = _mm256_set1_ps(p.pdiff_lut_scale);
        let idx_hi = _mm256_set1_epi32(WEIGHT_LUT_NB as i32 - 1);
        let zero = _mm256_setzero_si256();

        let mut x = lo;
        while x + 8 <= hi {
            // The four SAT corners are four contiguous streams, because both
            // patch edges advance with x.
            let a_lo = _mm256_loadu_si256(sat_a.as_ptr().add(x - pru) as *const __m256i);
            let a_hi = _mm256_loadu_si256(sat_a.as_ptr().add(x + pru + 1) as *const __m256i);
            let b_lo = _mm256_loadu_si256(sat_b.as_ptr().add(x - pru) as *const __m256i);
            let b_hi = _mm256_loadu_si256(sat_b.as_ptr().add(x + pru + 1) as *const __m256i);
            let ssd_i =
                _mm256_sub_epi32(_mm256_sub_epi32(_mm256_add_epi32(b_hi, a_lo), a_hi), b_lo);
            // The window sum is exact and at most 255² · 99² = 6.4e8, so the
            // u32 and i32 readings agree and this matches the scalar `as f32`.
            let ssd = _mm256_cvtepi32_ps(ssd_i);

            // `cvttps` truncates toward zero, as `as usize` does. The clamp to
            // 0 is what `as usize`'s saturation does for a negative, and it is
            // also what keeps the gather provably inside the table.
            let idx = _mm256_cvttps_epi32(_mm256_mul_ps(ssd, scale_v));
            let idx = _mm256_min_epi32(_mm256_max_epi32(idx, zero), idx_hi);

            // Masking the weight to zero is exactly the scalar `continue`: it
            // adds nothing to either running sum.
            let keep = _mm256_cmp_ps::<_CMP_LT_OQ>(ssd, max_v);
            let wt = _mm256_and_ps(_mm256_i32gather_ps::<4>(lut.as_ptr(), idx), keep);

            let v = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64(
                off.as_ptr().add((x as isize + dx) as usize) as *const __m128i,
            )));
            // Separate multiply and add, deliberately not an FMA — see the note
            // above the kernels.
            _mm256_storeu_ps(
                sum.as_mut_ptr().add(x),
                _mm256_add_ps(_mm256_loadu_ps(sum.as_ptr().add(x)), _mm256_mul_ps(wt, v)),
            );
            _mm256_storeu_ps(
                wsum.as_mut_ptr().add(x),
                _mm256_add_ps(_mm256_loadu_ps(wsum.as_ptr().add(x)), wt),
            );
            x += 8;
        }

        accumulate_span(p, dx, lut, sat_a, sat_b, off, sum, wsum, x..w);
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
    fn the_avx2_path_matches_the_scalar_one_bit_for_bit() {
        if !have_avx2() {
            return;
        }
        // Widths on and off a multiple of 8, so the scalar tail is exercised;
        // patches wider than the vector block; and a window large enough that
        // `dx` pushes the clamped border past the first vector block.
        let cases: [(usize, usize, u32, u32, f32); 5] = [
            (61, 47, 7, 5, 3.0),
            (64, 40, 7, 3, 1.0),
            (40, 33, 15, 9, 6.0),
            (20, 1, 7, 5, 3.0),
            (9, 9, 3, 3, 1.0),
        ];
        for (w, h, patch, research, sigma) in cases {
            let src = noisy(w * h);
            let scalar = plane_params_with(&src, w, h, patch, research, sigma, 1, false);
            let avx2 = plane_params_with(&src, w, h, patch, research, sigma, 3, true);
            assert_eq!(scalar, avx2, "avx2 diverged at {w}x{h} p={patch} r={research}");
        }
    }

    #[test]
    fn the_fixed_kernel_matches_the_direct_reference_bit_for_bit() {
        // The SAT + weight-table formulation against the per-sample loop it
        // replaced, at every tier and with the plane split into bands — the
        // padded border, the table cut-off and the summation order all have
        // to be exactly right for this to hold on random content and at the
        // extremes.
        for (w, h, src) in super::super::test_support::planes() {
            let want = plane_reference(&src, w, h);
            for tier in Tier::available() {
                for bands in [1usize, 3, 16] {
                    assert_eq!(
                        plane_fixed(&src, w, h, tier, bands),
                        want,
                        "{tier:?} x {bands} bands diverged at {w}x{h}"
                    );
                }
            }
        }
    }

    #[test]
    fn the_weight_table_is_the_reference_expression_and_ends_in_zero() {
        let lut = weight_lut();
        assert_eq!(lut[0], 1.0);
        assert_eq!(*lut.last().unwrap(), 0.0);
        assert!(lut.len() > 1000 && lut.len() <= SSD_MAX + 2, "{}", lut.len());
        for (ssd, &w) in lut.iter().enumerate().take(lut.len() - 1) {
            assert_eq!(w, (-(ssd as f32 / PN) / H2).exp(), "ssd {ssd}");
        }
        // Everything past the cut really is zero in the reference too.
        for ssd in (lut.len() - 1..=SSD_MAX).step_by(997) {
            assert_eq!((-(ssd as f32 / PN) / H2).exp(), 0.0, "ssd {ssd}");
        }
    }

    #[test]
    fn band_count_is_bounded_by_the_plane_height() {
        // No point spawning a thread per two rows.
        assert_eq!(band_count(1), 1);
        assert_eq!(band_count(16), 1);
        assert!(band_count(1080) >= 1);
    }
}
