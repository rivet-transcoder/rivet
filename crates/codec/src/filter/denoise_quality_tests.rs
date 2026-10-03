//! Denoising *quality* of the two parameterized denoisers, `nlmeans` and
//! `hqdn3d`, measured on synthetic pictures with known clean content: PSNR
//! gain against additive Gaussian noise at several levels and strengths, edge
//! sharpness after filtering, and (for the temporal filter) convergence and
//! flicker on a static scene. Everything runs through the public filter API.
//!
//! `cargo test -p rivet-codec --lib denoise_quality -- --ignored --nocapture`
//! prints the full figure table (`print_figures`).

use std::sync::Arc;

use bytes::Bytes;

use super::*;
use crate::frame::ColorSpace;

const W: usize = 128;
const H: usize = 96;

/// A clean test picture: a smooth diagonal gradient, a bright rectangle, a dark
/// disk and a band of soft sinusoidal texture — flats, gentle slopes, hard
/// edges and fine detail.
fn clean_picture() -> Vec<u8> {
    let mut p = vec![0u8; W * H];
    for y in 0..H {
        for x in 0..W {
            let mut v = 60.0 + 80.0 * (x + y) as f32 / (W + H) as f32;
            if (20..60).contains(&x) && (12..44).contains(&y) {
                v = 200.0;
            }
            let (dx, dy) = (x as f32 - 92.0, y as f32 - 30.0);
            if dx * dx + dy * dy < 18.0 * 18.0 {
                v = 30.0;
            }
            if (64..88).contains(&y) {
                v += 14.0 * ((x as f32) * 0.35).sin() * ((y as f32) * 0.5).cos();
            }
            p[y * W + x] = v.round().clamp(0.0, 255.0) as u8;
        }
    }
    p
}

/// Deterministic N(0, σ²) noise (LCG + Box–Muller), added and clamped.
fn add_noise(clean: &[u8], sigma: f32, seed: u32) -> Vec<u8> {
    let mut s = seed.wrapping_mul(0x9E37_79B9) ^ 0x5EED_1234;
    let mut uni = || {
        s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        ((s >> 8) as f32 + 0.5) / (1u32 << 24) as f32
    };
    clean
        .iter()
        .map(|&c| {
            let (u1, u2) = (uni(), uni());
            let g = (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos();
            (c as f32 + sigma * g).round().clamp(0.0, 255.0) as u8
        })
        .collect()
}

fn psnr(a: &[u8], b: &[u8]) -> f64 {
    let mse = a.iter().zip(b).map(|(&x, &y)| ((x as f64) - (y as f64)).powi(2)).sum::<f64>() / a.len() as f64;
    if mse == 0.0 {
        return 99.0;
    }
    10.0 * (255.0f64 * 255.0 / mse).log10()
}

fn frame_of(luma: &[u8], w: usize, h: usize) -> VideoFrame {
    let mut data = luma.to_vec();
    data.extend(std::iter::repeat_n(128u8, 2 * (w / 2) * (h / 2)));
    VideoFrame::new(Bytes::from(data), w as u32, h as u32, PixelFormat::Yuv420p, ColorSpace::Bt709, 0)
}

fn luma_of(f: &VideoFrame) -> Vec<u8> {
    f.data[..(f.width * f.height) as usize].to_vec()
}

fn nlmeans(noisy: &[u8], w: usize, h: usize, s: f32, p: u32, r: u32) -> Vec<u8> {
    luma_of(&apply(&frame_of(noisy, w, h), &VideoFilter::Nlmeans { s, p, pc: 0, r, rc: 0 }).unwrap())
}

fn hqdn3d_instance(spec: &str) -> FilterInstance {
    Arc::new(FilterChain::prepare(&parse_chain(spec).unwrap()).unwrap()).instantiate()
}

/// One frame through a fresh hqdn3d instance (spatial stage only).
fn hqdn3d_once(noisy: &[u8], w: usize, h: usize, spec: &str) -> Vec<u8> {
    luma_of(&hqdn3d_instance(spec).apply(frame_of(noisy, w, h)).unwrap())
}

/// A vertical step edge (60 → 180 at x = 32) under noise. The rows are
/// averaged into one profile first (so the noise, independent per row,
/// averages out and only the blur the filter put on the edge is left); returns
/// the profile's 10 %–90 % rise distance in samples (sub-sample by linear
/// interpolation; an ideal step measures 0.8) and the retained contrast (mean
/// of x ∈ 36..44 minus mean of x ∈ 20..28, over the step's 120).
fn edge_metrics(out: &[u8], w: usize, h: usize) -> (f64, f64) {
    let (lo, hi) = (60.0, 180.0);
    let rows = 2..h - 2;
    let n = rows.len() as f64;
    let profile: Vec<f64> =
        (0..w).map(|x| rows.clone().map(|y| out[y * w + x] as f64).sum::<f64>() / n).collect();
    let cross = |t: f64| -> f64 {
        (24..40)
            .find(|&x| profile[x] < t && profile[x + 1] >= t)
            .map(|x| x as f64 + (t - profile[x]) / (profile[x + 1] - profile[x]))
            .unwrap_or(f64::NAN)
    };
    let width = cross(lo + 0.9 * (hi - lo)) - cross(lo + 0.1 * (hi - lo));
    let mean = |r: std::ops::Range<usize>| profile[r.clone()].iter().sum::<f64>() / r.len() as f64;
    (width, (mean(36..44) - mean(20..28)) / (hi - lo))
}

fn step_picture(w: usize, h: usize) -> Vec<u8> {
    (0..w * h).map(|i| if i % w < 32 { 60 } else { 180 }).collect()
}

// ── nlmeans ─────────────────────────────────────────────────────────────────

#[test]
fn nlmeans_gains_psnr_at_every_noise_level_when_s_matches_it() {
    let clean = clean_picture();
    for (sigma, min_gain) in [(5.0f32, 3.0), (10.0, 5.0), (20.0, 6.0)] {
        let noisy = add_noise(&clean, sigma, sigma as u32);
        let before = psnr(&noisy, &clean);
        let after = psnr(&nlmeans(&noisy, W, H, sigma, 7, 15), &clean);
        assert!(
            after - before >= min_gain,
            "σ={sigma}: PSNR {before:.2} → {after:.2} dB (want ≥ +{min_gain})"
        );
    }
}

#[test]
fn nlmeans_strength_orders_the_smoothing() {
    // On a noisy picture, raising s up to the noise level must keep improving
    // the result; s far below the noise does little.
    let clean = clean_picture();
    let noisy = add_noise(&clean, 10.0, 7);
    let at = |s| psnr(&nlmeans(&noisy, W, H, s, 5, 11), &clean);
    let (p1, p4, p10) = (at(1.0), at(4.0), at(10.0));
    let p0 = psnr(&noisy, &clean);
    assert!(p1 - p0 < 1.0, "s=1 should barely touch σ=10 noise ({p0:.2} → {p1:.2})");
    assert!(p4 > p1 && p10 > p4, "PSNR must rise with s toward the noise level: {p1:.2} {p4:.2} {p10:.2}");
}

#[test]
fn nlmeans_keeps_edges_sharp() {
    let (w, h) = (64, 48);
    let noisy = add_noise(&step_picture(w, h), 10.0, 3);
    let (width, contrast) = edge_metrics(&nlmeans(&noisy, w, h, 10.0, 7, 15), w, h);
    // An ideal step measures 0.8 samples; a 3×3 box blur ~2.4.
    assert!(width < 1.0, "edge rise widened to {width:.2} samples");
    assert!(contrast > 0.98, "edge contrast fell to {contrast:.3}");
    // The metric does see blur: a 3×3 box on the same input widens the rise.
    let (box_width, _) = edge_metrics(&super::denoise::test_box3(&noisy, w, h), w, h);
    assert!(box_width > 2.0, "a 3×3 box blur measured only {box_width:.2}");
}

#[test]
fn nlmeans_is_deterministic_and_handles_odd_sizes() {
    for (w, h) in [(1, 1), (1, 9), (9, 1), (3, 5), (17, 13), (33, 7)] {
        let src = add_noise(&vec![128u8; w * h], 12.0, (w * 31 + h) as u32);
        // 4:2:0 frames need even sizes for chroma; test the luma kernel
        // through the frame API where possible, else the plane directly.
        let a = super::denoise::test_nlmeans_plane(&src, w, h, 10.0, 5, 9);
        let b = super::denoise::test_nlmeans_plane(&src, w, h, 10.0, 5, 9);
        assert_eq!(a, b, "{w}x{h}");
        assert_eq!(a.len(), w * h);
        if w * h > 1 {
            let dev = |p: &[u8]| p.iter().map(|&v| (v as f64 - 128.0).abs()).sum::<f64>();
            assert!(dev(&a) <= dev(&src), "{w}x{h}: denoising added energy");
        }
    }
}

// ── hqdn3d ──────────────────────────────────────────────────────────────────

#[test]
fn hqdn3d_spatial_stage_gains_psnr_and_strength_orders_it() {
    // One frame (no history yet): the spatial stage alone. A strength around
    // 1.5× the noise σ is the sweet spot; half the noise σ does little but
    // must not hurt.
    let clean = clean_picture();
    for sigma in [3.0f32, 6.0, 10.0] {
        let noisy = add_noise(&clean, sigma, 11 + sigma as u32);
        let before = psnr(&noisy, &clean);
        let at = |ls: f32| psnr(&hqdn3d_once(&noisy, W, H, &format!("hqdn3d={ls}")), &clean);
        let (weak, matched) = (at(0.5 * sigma), at(1.5 * sigma));
        assert!(weak >= before - 0.05, "σ={sigma}: a weak setting made it worse ({before:.2} → {weak:.2})");
        assert!(matched > weak + 2.0, "σ={sigma}: a stronger setting must gain more ({weak:.2} vs {matched:.2})");
        assert!(matched > before + 4.0, "σ={sigma}: PSNR {before:.2} → {matched:.2}");
    }
}

#[test]
fn hqdn3d_keeps_edges_sharp() {
    let (w, h) = (64, 48);
    let noisy = add_noise(&step_picture(w, h), 4.0, 5);
    let (width, contrast) = edge_metrics(&hqdn3d_once(&noisy, w, h, "hqdn3d=4:3:6:4.5"), w, h);
    assert!(width < 1.0, "edge rise widened to {width:.2} samples");
    assert!(contrast > 0.98, "edge contrast fell to {contrast:.3}");
}

#[test]
fn hqdn3d_static_scene_converges_and_stops_flickering() {
    // The same clean picture under fresh noise every frame: the temporal stage
    // must drive the error well below a single frame's, and the frame-to-frame
    // change (flicker) far below the input's.
    let clean = clean_picture();
    let mut inst = hqdn3d_instance("hqdn3d=4:3:6:4.5");
    let mut psnrs = Vec::new();
    let mut flicker_in = 0.0;
    let mut flicker_out = 0.0;
    let (mut prev_in, mut prev_out): (Option<Vec<u8>>, Option<Vec<u8>>) = (None, None);
    for k in 0..20u32 {
        let noisy = add_noise(&clean, 4.0, 100 + k);
        let out = luma_of(&inst.apply(frame_of(&noisy, W, H)).unwrap());
        psnrs.push(psnr(&out, &clean));
        if k >= 10 {
            let mad = |a: &[u8], b: &[u8]| a.iter().zip(b).map(|(&x, &y)| (x as f64 - y as f64).abs()).sum::<f64>() / a.len() as f64;
            flicker_in += mad(&noisy, prev_in.as_ref().unwrap());
            flicker_out += mad(&out, prev_out.as_ref().unwrap());
        }
        prev_in = Some(noisy);
        prev_out = Some(out);
    }
    assert!(psnrs[19] > psnrs[0] + 2.0, "no temporal gain: {psnrs:?}");
    assert!(flicker_out < 0.35 * flicker_in, "flicker {flicker_in:.1} → {flicker_out:.1}");
}

#[test]
fn hqdn3d_lets_motion_through() {
    // A bright square jumps 16 samples between frames: the new position must
    // show at once, not as a ghost blended with the old one.
    let (w, h) = (64, 32);
    let pic = |x0: usize| -> Vec<u8> {
        (0..w * h).map(|i| if (x0..x0 + 12).contains(&(i % w)) && (10..22).contains(&(i / w)) { 200 } else { 50 }).collect()
    };
    let mut inst = hqdn3d_instance("hqdn3d=4:3:6:4.5");
    for _ in 0..5 {
        inst.apply(frame_of(&pic(8), w, h)).unwrap();
    }
    let out = luma_of(&inst.apply(frame_of(&pic(40), w, h)).unwrap());
    let want = pic(40);
    let worst = out.iter().zip(&want).map(|(&a, &b)| (a as i32 - b as i32).abs()).max().unwrap();
    assert!(worst <= 2, "motion ghosted (max error {worst})");
}

#[test]
fn hqdn3d_is_deterministic_and_handles_odd_sizes() {
    for (w, h) in [(2, 2), (3, 3), (17, 9), (31, 5), (5, 31)] {
        let run = || -> Vec<Vec<u8>> {
            let mut inst = hqdn3d_instance("hqdn3d");
            (0..3)
                .map(|k| {
                    let mut data = add_noise(&vec![120u8; w * h], 5.0, k);
                    data.extend(add_noise(&vec![90u8; 2 * (w / 2) * (h / 2)], 5.0, k + 50));
                    let f = VideoFrame::new(Bytes::from(data), w as u32, h as u32, PixelFormat::Yuv420p, ColorSpace::Bt709, 0);
                    inst.apply(f).unwrap().data.to_vec()
                })
                .collect()
        };
        let (a, b) = (run(), run());
        assert_eq!(a, b, "{w}x{h}");
        assert_eq!(a[0].len(), w * h + 2 * (w / 2) * (h / 2));
    }
}

// ── figures ─────────────────────────────────────────────────────────────────

#[test]
#[ignore = "prints the figure table; run with --ignored --nocapture"]
fn print_figures() {
    let clean = clean_picture();
    println!("nlmeans (p=7 r=15), {W}x{H} test picture, PSNR dB");
    for sigma in [5.0f32, 10.0, 20.0] {
        let noisy = add_noise(&clean, sigma, sigma as u32);
        let mut line = format!("  noise σ={sigma:>4}: input {:.2}", psnr(&noisy, &clean));
        for s in [1.0f32, 3.0, 5.0, 10.0, 15.0, 20.0, 30.0] {
            line += &format!(" | s={s}: {:.2}", psnr(&nlmeans(&noisy, W, H, s, 7, 15), &clean));
        }
        println!("{line}");
    }
    println!("hqdn3d single frame (spatial), PSNR dB");
    for sigma in [3.0f32, 6.0, 10.0] {
        let noisy = add_noise(&clean, sigma, 11 + sigma as u32);
        let mut line = format!("  noise σ={sigma:>4}: input {:.2}", psnr(&noisy, &clean));
        for ls in [1.0f32, 2.0, 4.0, 8.0, 12.0, 16.0] {
            line += &format!(" | ls={ls}: {:.2}", psnr(&hqdn3d_once(&noisy, W, H, &format!("hqdn3d={ls}")), &clean));
        }
        println!("{line}");
    }
    println!("hqdn3d static scene, 20 frames, PSNR dB at frames 1/5/20");
    for sigma in [3.0f32, 6.0, 10.0] {
        let mut line = format!("  noise σ={sigma:>4}:");
        for spec in ["hqdn3d=2", "hqdn3d", "hqdn3d=8", "hqdn3d=12"] {
            let mut inst = hqdn3d_instance(spec);
            let mut ps = Vec::new();
            let mut inp = 0.0;
            for k in 0..20u32 {
                let noisy = add_noise(&clean, sigma, 100 + k);
                inp = psnr(&noisy, &clean);
                ps.push(psnr(&luma_of(&inst.apply(frame_of(&noisy, W, H)).unwrap()), &clean));
            }
            line += &format!(" | {spec} (in {inp:.1}): {:.2}/{:.2}/{:.2}", ps[0], ps[4], ps[19]);
        }
        println!("{line}");
    }
    let (w, h) = (64, 48);
    for sigma in [4.0f32, 10.0] {
        let noisy = add_noise(&step_picture(w, h), sigma, 3);
        let (iw, ic) = edge_metrics(&step_picture(w, h), w, h);
        let (nw, nc) = edge_metrics(&nlmeans(&noisy, w, h, sigma, 7, 15), w, h);
        let (hw, hc) = edge_metrics(&hqdn3d_once(&noisy, w, h, "hqdn3d"), w, h);
        let blur = super::denoise::test_box3(&noisy, w, h);
        let (bw, bc) = edge_metrics(&blur, w, h);
        println!(
            "edge (σ={sigma}): rise width / contrast — clean {iw:.2}/{ic:.3}, nlmeans s={sigma} {nw:.2}/{nc:.3}, hqdn3d default {hw:.2}/{hc:.3}, 3x3 box {bw:.2}/{bc:.3}"
        );
    }
}
