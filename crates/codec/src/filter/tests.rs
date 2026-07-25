//! Tests for the whole filter module — they drive the public [`apply`] /
//! [`parse_chain`] surface (plus the internal `overlay::PreparedOverlay`), so
//! the per-filter implementation files stay focused on the algorithm.

use super::overlay::PreparedOverlay;
use super::*;
use crate::frame::ColorSpace;
use bytes::Bytes;

/// A `w×h` frame whose luma ramps `0,1,2,…` and chroma is flat 100/200.
fn frame(w: u32, h: u32) -> VideoFrame {
    let (wu, hu) = (w as usize, h as usize);
    let mut data = Vec::new();
    for r in 0..hu {
        for c in 0..wu {
            data.push((r * wu + c) as u8);
        }
    }
    data.extend(std::iter::repeat(100).take((wu / 2) * (hu / 2)));
    data.extend(std::iter::repeat(200).take((wu / 2) * (hu / 2)));
    VideoFrame::new(Bytes::from(data), w, h, PixelFormat::Yuv420p, ColorSpace::Bt709, 0)
}

/// A flat `w×h` frame with the given luma + chroma values.
fn flat(w: u32, h: u32, yv: u8, uv: u8, vv: u8) -> VideoFrame {
    let (wu, hu) = (w as usize, h as usize);
    let mut data = vec![yv; wu * hu];
    data.extend(std::iter::repeat(uv).take((wu / 2) * (hu / 2)));
    data.extend(std::iter::repeat(vv).take((wu / 2) * (hu / 2)));
    VideoFrame::new(Bytes::from(data), w, h, PixelFormat::Yuv420p, ColorSpace::Bt709, 0)
}

fn luma(f: &VideoFrame) -> &[u8] {
    &f.data[..(f.width * f.height) as usize]
}

#[test]
fn parse_and_display_round_trip() {
    let c = parse_chain("crop=1280:720,hflip,overlay=logo.png:24:24,brightness=10,saturation=1.5,invert").unwrap();
    assert_eq!(c[0], VideoFilter::Crop { w: 1280, h: 720, x: None, y: None });
    assert_eq!(c[2], VideoFilter::Overlay { image: "logo.png".into(), x: 24, y: 24 });
    assert_eq!(c[3], VideoFilter::Brightness(10));
    assert_eq!(c[4], VideoFilter::Saturation(1.5));
    assert_eq!(c[5], VideoFilter::Invert);
    assert_eq!(chain_to_string(&c), "crop=1280:720,hflip,overlay=logo.png:24:24,brightness=10,saturation=1.5,invert");
    assert_eq!(parse_chain("overlay=a.png").unwrap()[0], VideoFilter::Overlay { image: "a.png".into(), x: 0, y: 0 });
    assert_eq!(parse_chain("negate").unwrap()[0], VideoFilter::Invert);
    assert_eq!(parse_chain("contrast=1.2").unwrap()[0], VideoFilter::Contrast(1.2));
    assert!(parse_chain("brightness=x").is_err());
    assert!(parse_chain("rotate=45").is_err());
}

#[cfg(feature = "serde")]
#[test]
fn structured_json_round_trips() {
    let json = r#"[{"crop":{"w":1280,"h":720}},"hflip",{"overlay":{"image":"logo.png","x":24,"y":24}},{"brightness":10},"invert"]"#;
    let from_list: FilterSpec = serde_json::from_str(json).unwrap();
    let expect = vec![
        VideoFilter::Crop { w: 1280, h: 720, x: None, y: None },
        VideoFilter::HFlip,
        VideoFilter::Overlay { image: "logo.png".into(), x: 24, y: 24 },
        VideoFilter::Brightness(10),
        VideoFilter::Invert,
    ];
    assert_eq!(from_list.resolve().unwrap(), expect);
    assert_eq!(parse_chain(&chain_to_string(&expect)).unwrap(), expect);
}

#[test]
fn hflip_reverses_rows() {
    let out = apply(&frame(4, 2), &VideoFilter::HFlip).unwrap();
    assert_eq!(&luma(&out)[..4], &[3, 2, 1, 0]);
}

#[test]
fn rotate_dims_and_roundtrip() {
    let f = frame(4, 2);
    let r90 = apply(&f, &VideoFilter::Rotate(90)).unwrap();
    assert_eq!((r90.width, r90.height), (2, 4));
    let back = apply(&r90, &VideoFilter::Rotate(270)).unwrap();
    assert_eq!(luma(&back), luma(&f));
    assert!(apply(&f, &VideoFilter::Rotate(45)).is_err());
}

#[test]
fn color_filters() {
    // brightness: +20 on a flat-100 luma → 120
    let b = apply(&flat(4, 4, 100, 128, 128), &VideoFilter::Brightness(20)).unwrap();
    assert!(luma(&b).iter().all(|&p| p == 120));
    // invert: 100 → 155, chroma 128 → 127
    let inv = apply(&flat(2, 2, 100, 128, 128), &VideoFilter::Invert).unwrap();
    assert_eq!(luma(&inv)[0], 155);
    assert_eq!(inv.data[4], 127);
    // saturation 0 → chroma collapses to 128 (grayscale)
    let s0 = apply(&flat(4, 4, 100, 200, 60), &VideoFilter::Saturation(0.0)).unwrap();
    assert!(s0.data[16..].iter().all(|&p| p == 128));
    // brightness on a 10-bit frame is rejected
    let ten = VideoFrame::new(Bytes::from(vec![0u8; 2 * (4 * 4 + 2 * 4)]), 4, 4, PixelFormat::Yuv420p10le, ColorSpace::Bt709, 0);
    assert!(apply(&ten, &VideoFilter::Brightness(10)).is_err());
}

#[test]
fn overlay_composites_with_alpha() {
    // 2×2 RGBA overlay: top row opaque red, bottom row fully transparent.
    let red = [255u8, 0, 0, 255];
    let clear = [0u8, 0, 0, 0];
    let mut rgba = Vec::new();
    rgba.extend_from_slice(&red);
    rgba.extend_from_slice(&red);
    rgba.extend_from_slice(&clear);
    rgba.extend_from_slice(&clear);
    let ov = PreparedOverlay::from_rgba(&rgba, 2, 2, 0, 0).unwrap();
    // composite onto a 4×4 flat grey frame
    let base = flat(4, 4, 100, 128, 128);
    let out = ov.composite(&base).unwrap();
    let y = luma(&out);
    // opaque red top-left → red's luma (≈ 16 + 0.183*255 ≈ 63), NOT 100
    assert!(y[0] > 50 && y[0] < 90, "opaque red luma was {}", y[0]);
    // transparent bottom row → unchanged grey 100
    assert_eq!(y[2 * 4], 100);
    // out-of-overlay region (col ≥ 2) unchanged
    assert_eq!(y[2], 100);
}

#[test]
fn overlay_via_apply_errors_without_prepare() {
    let r = apply(&flat(4, 4, 100, 128, 128), &VideoFilter::Overlay { image: "x.png".into(), x: 0, y: 0 });
    assert!(r.is_err());
}

#[test]
fn filter_chain_prepare_missing_image_errors() {
    let r = FilterChain::prepare(&[VideoFilter::Overlay { image: "/nope/missing.png".into(), x: 0, y: 0 }]);
    assert!(r.is_err());
}

#[test]
fn filter_chain_applies_stateless() {
    let chain = FilterChain::prepare(&[VideoFilter::HFlip, VideoFilter::Brightness(10)]).unwrap();
    assert!(!chain.is_empty());
    let out = chain.apply(frame(4, 2)).unwrap();
    assert_eq!((out.width, out.height), (4, 2));
}

#[test]
fn ten_bit_geometric_still_works() {
    let mut data: Vec<u8> = Vec::new();
    for s in [0u16, 1, 2, 3] {
        data.extend_from_slice(&s.to_le_bytes());
    }
    data.extend_from_slice(&(512u16).to_le_bytes());
    data.extend_from_slice(&(512u16).to_le_bytes());
    let f = VideoFrame::new(Bytes::from(data), 2, 2, PixelFormat::Yuv420p10le, ColorSpace::Bt709, 0);
    let out = apply(&f, &VideoFilter::HFlip).unwrap();
    assert_eq!(&out.data[0..2], &1u16.to_le_bytes());
}

// ── denoise family ──────────────────────────────────────────────────────────

const DENOISE_METHODS: [DenoiseMethod; 6] = [
    DenoiseMethod::Bilateral,
    DenoiseMethod::Gaussian,
    DenoiseMethod::Median,
    DenoiseMethod::Mean,
    DenoiseMethod::Nlmeans,
    DenoiseMethod::Anisotropic,
];

/// Build a `w×h` Yuv420p frame with the given luma + flat neutral chroma.
fn frame_with_luma(luma: Vec<u8>, w: u32, h: u32) -> VideoFrame {
    let (wu, hu) = (w as usize, h as usize);
    assert_eq!(luma.len(), wu * hu);
    let mut data = luma;
    data.extend(std::iter::repeat(128).take(2 * (wu / 2) * (hu / 2)));
    VideoFrame::new(Bytes::from(data), w, h, PixelFormat::Yuv420p, ColorSpace::Bt709, 0)
}

/// Denoise a luma pattern, return the output luma plane.
fn denoise_luma(plane: Vec<u8>, w: u32, h: u32, method: DenoiseMethod, strength: f32) -> Vec<u8> {
    let f = frame_with_luma(plane, w, h);
    let out = apply(&f, &VideoFilter::Denoise { method, strength }).unwrap();
    luma(&out).to_vec()
}

#[test]
fn denoise_parse_and_display() {
    let bil = |s| VideoFilter::Denoise { method: DenoiseMethod::Bilateral, strength: s };
    assert_eq!(parse_chain("denoise").unwrap()[0], bil(0.5));
    assert_eq!(parse_chain("denoise=0.7").unwrap()[0], bil(0.7));
    assert_eq!(parse_chain("denoise=median").unwrap()[0], VideoFilter::Denoise { method: DenoiseMethod::Median, strength: 0.5 });
    assert_eq!(parse_chain("denoise=nlmeans:0.3").unwrap()[0], VideoFilter::Denoise { method: DenoiseMethod::Nlmeans, strength: 0.3 });
    assert_eq!(parse_chain("denoise=0.3:gaussian").unwrap()[0], VideoFilter::Denoise { method: DenoiseMethod::Gaussian, strength: 0.3 });
    assert_eq!(parse_chain("nr=pm").unwrap()[0], VideoFilter::Denoise { method: DenoiseMethod::Anisotropic, strength: 0.5 });
    assert_eq!(chain_to_string(&parse_chain("denoise=median:0.8").unwrap()), "denoise=median:0.8");
    assert!(parse_chain("denoise=2.0").is_err());
    assert!(parse_chain("denoise=foo").is_err());
}

#[test]
fn denoise_flat_is_unchanged() {
    for m in DENOISE_METHODS {
        let out = denoise_luma(vec![100u8; 64], 8, 8, m, 1.0);
        assert!(out.iter().all(|&p| (p as i32 - 100).abs() <= 1), "{m:?} altered a flat plane");
    }
}

#[test]
fn denoise_strength_zero_is_identity() {
    let luma: Vec<u8> = (0..64).map(|i| (i * 3) as u8).collect();
    for m in DENOISE_METHODS {
        assert_eq!(denoise_luma(luma.clone(), 8, 8, m, 0.0), luma, "{m:?} @ strength 0 must be identity");
    }
}

#[test]
fn denoise_smooths_checkerboard() {
    let luma: Vec<u8> = (0..64).map(|i| if (i / 8 + i % 8) % 2 == 0 { 122 } else { 134 }).collect();
    for m in [
        DenoiseMethod::Bilateral,
        DenoiseMethod::Gaussian,
        DenoiseMethod::Mean,
        DenoiseMethod::Nlmeans,
        DenoiseMethod::Anisotropic,
    ] {
        let out = denoise_luma(luma.clone(), 8, 8, m, 1.0);
        let maxdev = out.iter().map(|&p| (p as i32 - 128).abs()).max().unwrap();
        assert!(maxdev < 6, "{m:?} didn't smooth the checkerboard (maxdev {maxdev})");
    }
}

#[test]
fn denoise_median_removes_impulse() {
    let mut luma = vec![100u8; 64];
    luma[3 * 8 + 3] = 250;
    let out = denoise_luma(luma, 8, 8, DenoiseMethod::Median, 1.0);
    assert_eq!(out[3 * 8 + 3], 100, "median should remove the impulse");
}

#[test]
fn denoise_bilateral_preserves_edge() {
    let luma: Vec<u8> = (0..64).map(|i| if (i % 8) < 4 { 50 } else { 200 }).collect();
    let out = denoise_luma(luma, 8, 8, DenoiseMethod::Bilateral, 1.0);
    for r in 0..8 {
        assert!(out[r * 8 + 1] < 80, "left edge blurred: {}", out[r * 8 + 1]);
        assert!(out[r * 8 + 6] > 170, "right edge blurred: {}", out[r * 8 + 6]);
    }
}

#[test]
fn denoise_rejects_10bit() {
    let ten = VideoFrame::new(Bytes::from(vec![0u8; 2 * (4 * 4 + 2 * 4)]), 4, 4, PixelFormat::Yuv420p10le, ColorSpace::Bt709, 0);
    assert!(apply(&ten, &VideoFilter::Denoise { method: DenoiseMethod::Gaussian, strength: 0.5 }).is_err());
}

// ── nlmeans (parameterized, ffmpeg-compatible) ──────────────────────────────

/// Deterministic ±12 uniform-ish noise around `centre`, so the smoothing tests
/// don't depend on an RNG crate or a seed that drifts between runs.
fn noisy_plane(n: usize, centre: u8) -> Vec<u8> {
    let mut state = 0x1234_5678u32;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (centre as i32 + ((state >> 16) % 25) as i32 - 12) as u8
        })
        .collect()
}

/// Apply the parameterized nlmeans to a luma pattern; return the output plane.
fn nlmeans_luma(plane: Vec<u8>, w: u32, h: u32, s: f32, p: u32, r: u32) -> Vec<u8> {
    let f = frame_with_luma(plane, w, h);
    let out = apply(&f, &VideoFilter::Nlmeans { s, p, pc: 0, r, rc: 0 }).unwrap();
    luma(&out).to_vec()
}

/// Mean absolute deviation of a plane from a constant — the "how much noise is
/// left" measure for the smoothing tests.
fn mean_abs_dev(plane: &[u8], centre: u8) -> f32 {
    plane.iter().map(|&v| (v as i32 - centre as i32).unsigned_abs() as f32).sum::<f32>()
        / plane.len() as f32
}

#[test]
fn nlmeans_parse_and_display() {
    // The exact spelling from an ffmpeg command line must parse as-is.
    assert_eq!(
        parse_chain("nlmeans=s=1:p=7:pc=5:r=3:rc=3").unwrap()[0],
        VideoFilter::Nlmeans { s: 1.0, p: 7, pc: 5, r: 3, rc: 3 }
    );
    // Bare `nlmeans` is ffmpeg's defaults; pc/rc stay 0 = "same as luma".
    assert_eq!(
        parse_chain("nlmeans").unwrap()[0],
        VideoFilter::Nlmeans { s: 1.0, p: 7, pc: 0, r: 15, rc: 0 }
    );
    // Positional values fill s:p:pc:r:rc in declaration order.
    assert_eq!(
        parse_chain("nlmeans=2:5").unwrap()[0],
        VideoFilter::Nlmeans { s: 2.0, p: 5, pc: 0, r: 15, rc: 0 }
    );
    // Display emits the full key=value form and round-trips.
    let c = parse_chain("nlmeans=s=3:p=5:r=7").unwrap();
    assert_eq!(chain_to_string(&c), "nlmeans=s=3:p=5:pc=0:r=7:rc=0");
    assert_eq!(parse_chain(&chain_to_string(&c)).unwrap(), c);
    // Range + spelling errors surface at parse time, not at apply time.
    assert!(parse_chain("nlmeans=s=0.5").is_err(), "below ffmpeg's 1.0 sigma floor");
    assert!(parse_chain("nlmeans=s=31").is_err(), "above the 30.0 sigma ceiling");
    assert!(parse_chain("nlmeans=p=101").is_err());
    assert!(parse_chain("nlmeans=q=3").is_err());
}

#[test]
fn nlmeans_flat_is_unchanged() {
    let out = nlmeans_luma(vec![100u8; 256], 16, 16, 10.0, 3, 5);
    assert!(out.iter().all(|&v| v == 100), "nlmeans altered a flat plane");
}

#[test]
fn nlmeans_degenerate_research_window_is_identity() {
    // A 1×1 (or 0) research window can only ever see the centre sample, so there
    // is nothing to average — short-circuit rather than burn a pass over the plane.
    let src: Vec<u8> = (0..256).map(|i| (i * 7 % 251) as u8).collect();
    assert_eq!(nlmeans_luma(src.clone(), 16, 16, 10.0, 3, 1), src);
    assert_eq!(nlmeans_luma(src.clone(), 16, 16, 10.0, 3, 0), src);
}

#[test]
fn nlmeans_strong_sigma_smooths_noise() {
    let noisy = noisy_plane(32 * 32, 128);
    let before = mean_abs_dev(&noisy, 128);
    let after = mean_abs_dev(&nlmeans_luma(noisy, 32, 32, 30.0, 3, 7), 128);
    assert!(
        after < before / 2.0,
        "strong-sigma nlmeans barely denoised (mean |dev| {before:.2} → {after:.2})"
    );
}

#[test]
fn nlmeans_sigma_is_monotonic() {
    // `s` has to actually mean "strength": more sigma, less residual noise.
    let noisy = noisy_plane(32 * 32, 128);
    let weak = mean_abs_dev(&nlmeans_luma(noisy.clone(), 32, 32, 2.0, 3, 5), 128);
    let strong = mean_abs_dev(&nlmeans_luma(noisy, 32, 32, 20.0, 3, 5), 128);
    assert!(strong < weak, "raising s didn't denoise harder ({weak:.2} → {strong:.2})");
}

#[test]
fn nlmeans_preserves_repeating_texture_at_default_sigma() {
    // The whole point of non-local means: a repeating pattern is *signal*. At
    // ffmpeg's default s=1 only near-identical patches carry weight, so the
    // checkerboard survives where a plain blur would flatten it (compare
    // `denoise_smooths_checkerboard`, which drives every method to ~flat).
    let luma: Vec<u8> =
        (0..1024).map(|i| if (i / 32 + i % 32) % 2 == 0 { 100 } else { 160 }).collect();
    let out = nlmeans_luma(luma.clone(), 32, 32, 1.0, 3, 5);
    let maxdev =
        out.iter().zip(&luma).map(|(&a, &b)| (a as i32 - b as i32).abs()).max().unwrap();
    assert!(maxdev <= 4, "default-sigma nlmeans flattened a repeating texture (maxdev {maxdev})");
}

#[test]
fn nlmeans_chroma_params_are_independent_of_luma() {
    // `pc`/`rc` must drive the chroma planes on their own — and 0 must mean
    // "reuse the luma value", per ffmpeg.
    let (w, h) = (16u32, 16u32);
    let (cw, ch) = ((w / 2) as usize, (h / 2) as usize);
    let mut data = noisy_plane((w * h) as usize, 128);
    let chroma = noisy_plane(cw * ch, 110);
    data.extend_from_slice(&chroma);
    data.extend_from_slice(&chroma);
    let f = VideoFrame::new(Bytes::from(data), w, h, PixelFormat::Yuv420p, ColorSpace::Bt709, 0);
    let chroma_of = |fr: &VideoFrame| fr.data[(w * h) as usize..].to_vec();

    let implicit = apply(&f, &VideoFilter::Nlmeans { s: 20.0, p: 3, pc: 0, r: 5, rc: 0 }).unwrap();
    let explicit = apply(&f, &VideoFilter::Nlmeans { s: 20.0, p: 3, pc: 3, r: 5, rc: 5 }).unwrap();
    assert_eq!(chroma_of(&implicit), chroma_of(&explicit), "pc/rc=0 must mean 'same as p/r'");

    // rc=1 collapses the chroma research window to the centre sample, so chroma
    // passes through untouched while luma is still denoised.
    let chroma_off = apply(&f, &VideoFilter::Nlmeans { s: 20.0, p: 3, pc: 0, r: 5, rc: 1 }).unwrap();
    assert_eq!(chroma_of(&chroma_off), [chroma.clone(), chroma].concat(), "rc=1 must leave chroma alone");
    assert_ne!(luma(&chroma_off), luma(&f), "luma should still be denoised");
}

#[test]
fn nlmeans_rejects_10bit() {
    let ten = VideoFrame::new(Bytes::from(vec![0u8; 2 * (4 * 4 + 2 * 4)]), 4, 4, PixelFormat::Yuv420p10le, ColorSpace::Bt709, 0);
    assert!(apply(&ten, &VideoFilter::Nlmeans { s: 1.0, p: 7, pc: 0, r: 15, rc: 0 }).is_err());
}
