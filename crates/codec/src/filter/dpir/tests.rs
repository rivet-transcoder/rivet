//! Tests for `denoise=dpir`. The pure parts (grammar, tiling, levels, colour,
//! model-path resolution) run in every build; the network itself runs on the
//! reduced-width fixtures under `tests/fixtures/dpir_tiny_*.pth` with the
//! `dpir` feature; the `#[ignore]`d tests need the 130 MB release model.

use super::*;
use crate::filter::{FilterChain, apply, chain_to_string, parse_chain};
use crate::frame::{PixelFormat, VideoFrame};
use bytes::Bytes;
use std::path::PathBuf;

// ── grammar ──────────────────────────────────────────────────────────────────

#[test]
fn parse_and_display() {
    assert_eq!(parse_chain("denoise=dpir").unwrap()[0], VideoFilter::Dpir { sigma: DEFAULT_SIGMA, color: false });
    assert_eq!(parse_chain("denoise=dpir:25").unwrap()[0], VideoFilter::Dpir { sigma: 25.0, color: false });
    assert_eq!(parse_chain("denoise=25:dpir").unwrap()[0], VideoFilter::Dpir { sigma: 25.0, color: false });
    assert_eq!(parse_chain("nr=DPIR:7.5:color").unwrap()[0], VideoFilter::Dpir { sigma: 7.5, color: true });
    assert_eq!(parse_chain("denoise=dpir:rgb:10").unwrap()[0], VideoFilter::Dpir { sigma: 10.0, color: true });
    assert_eq!(parse_chain("denoise=dpir:gray").unwrap()[0], VideoFilter::Dpir { sigma: DEFAULT_SIGMA, color: false });
    let c = parse_chain("crop=64:32,denoise=dpir:25:color,hflip").unwrap();
    assert_eq!(chain_to_string(&c), "crop=64:32,denoise=dpir:25:color,hflip");
    assert_eq!(parse_chain(&chain_to_string(&c)).unwrap(), c);
    assert_eq!(VideoFilter::Dpir { sigma: 15.0, color: false }.to_string(), "denoise=dpir:15");
}

#[test]
fn parse_rejects_bad_sigma_and_options() {
    let e = parse_chain("denoise=dpir:80").unwrap_err().to_string();
    assert!(e.contains("0..=50"), "{e}");
    assert!(parse_chain("denoise=dpir:-1").is_err());
    let e = parse_chain("denoise=dpir:strong").unwrap_err().to_string();
    assert!(e.contains("unknown dpir option 'strong'"), "{e}");
    // The classical grammar is untouched: a bare number is still a 0..=1 blend.
    assert!(parse_chain("denoise=25").is_err());
    assert!(parse_chain("denoise=bilateral:0.7").is_ok());
}

#[test]
fn stateless_apply_refuses_dpir() {
    let f = frame8(16, 16, |_, _| 128);
    let e = apply(&f, &VideoFilter::Dpir { sigma: 10.0, color: false }).unwrap_err().to_string();
    assert!(e.contains("resource filter"), "{e}");
}

#[cfg(not(feature = "dpir"))]
#[test]
fn prepare_without_feature_names_it() {
    let e = FilterChain::prepare(&[VideoFilter::Dpir { sigma: 10.0, color: false }]).err().expect("no feature → error");
    let msg = format!("{e:#}");
    assert!(msg.contains("`dpir` feature"), "{msg}");
    assert!(msg.contains("--features dpir"), "{msg}");
}

// ── model file resolution ────────────────────────────────────────────────────

#[test]
fn model_path_resolution() {
    let cache = PathBuf::from("/cache");
    assert_eq!(
        expected_model_path(DpirModel::Gray, None, Some(&cache)).unwrap(),
        PathBuf::from("/cache").join("drunet_gray.pth")
    );
    assert_eq!(
        expected_model_path(DpirModel::Color, Some(&PathBuf::from("/models")), Some(&cache)).unwrap(),
        PathBuf::from("/models").join("drunet_color.pth")
    );
    assert_eq!(
        expected_model_path(DpirModel::Color, Some(&PathBuf::from("/x/my.pth")), Some(&cache)).unwrap(),
        PathBuf::from("/x/my.pth")
    );
    assert_eq!(
        expected_model_path(DpirModel::Gray, Some(&PathBuf::from("/x/w.SafeTensors")), None).unwrap(),
        PathBuf::from("/x/w.SafeTensors")
    );
    let e = expected_model_path(DpirModel::Gray, None, None).unwrap_err().to_string();
    assert!(e.contains(ENV_MODEL), "{e}");
}

#[test]
fn missing_model_error_names_url_and_path() {
    let p = PathBuf::from("/cache/drunet_color.pth");
    let e = missing_model_error(DpirModel::Color, &p).to_string();
    assert!(e.contains("https://github.com/cszn/KAIR/releases/download/v1.0/drunet_color.pth"), "{e}");
    assert!(e.contains("drunet_color.pth"), "{e}");
    assert!(e.contains("curl -L"), "{e}");
    assert!(e.contains(ENV_MODEL), "{e}");
}

#[test]
fn model_url_is_the_kair_release() {
    assert_eq!(DpirModel::Gray.url(), "https://github.com/cszn/KAIR/releases/download/v1.0/drunet_gray.pth");
    assert_eq!(DpirModel::for_color(true), DpirModel::Color);
    assert_eq!(DpirModel::Gray.image_channels() + 1, 2);
    assert_eq!(DpirModel::Color.image_channels() + 1, 4);
}

// ── tiling ───────────────────────────────────────────────────────────────────

/// Every pixel is kept by exactly one tile; every fed region lies inside the
/// frame, contains its keep region, and carries `overlap` context except at
/// the frame edge.
fn check_tiling(w: usize, h: usize, tile: usize, overlap: usize) -> Vec<Tile> {
    let ts = tiles(w, h, tile, overlap);
    let mut covered = vec![0u8; w * h];
    for t in &ts {
        assert!(t.x + t.w <= w && t.y + t.h <= h, "{t:?} outside {w}x{h}");
        assert!(t.keep_x >= t.x && t.keep_y >= t.y);
        assert!(t.keep_x + t.keep_w <= t.x + t.w && t.keep_y + t.keep_h <= t.y + t.h);
        assert_eq!(t.keep_x - t.x, t.keep_x.min(overlap), "left context {t:?}");
        assert_eq!(t.keep_y - t.y, t.keep_y.min(overlap), "top context {t:?}");
        assert_eq!((t.x + t.w) - (t.keep_x + t.keep_w), (w - t.keep_x - t.keep_w).min(overlap), "right context {t:?}");
        assert_eq!((t.y + t.h) - (t.keep_y + t.keep_h), (h - t.keep_y - t.keep_h).min(overlap), "bottom context {t:?}");
        for y in t.keep_y..t.keep_y + t.keep_h {
            for x in t.keep_x..t.keep_x + t.keep_w {
                covered[y * w + x] += 1;
            }
        }
    }
    assert!(covered.iter().all(|&c| c == 1), "{w}x{h} tile {tile} ov {overlap}: not a partition");
    ts
}

#[test]
fn tiles_partition_the_frame() {
    assert_eq!(check_tiling(1920, 1080, 512, 32).len(), 4 * 3);
    assert_eq!(check_tiling(1280, 720, 512, 32).len(), 3 * 2);
    assert_eq!(check_tiling(100, 60, 0, 32).len(), 1); // 0 = whole frame
    assert_eq!(check_tiling(100, 60, 1000, 32).len(), 1);
    check_tiling(33, 17, 8, 3); // odd sizes, partial tiles
    check_tiling(8, 8, 8, 8);
    let t = tiles(1920, 1080, 512, 32)[5]; // second row, second column
    assert_eq!(t, Tile { x: 480, y: 480, w: 576, h: 576, keep_x: 512, keep_y: 512, keep_w: 512, keep_h: 512 });
    let last = *tiles(1920, 1080, 512, 32).last().unwrap();
    assert_eq!(last, Tile { x: 1504, y: 992, w: 416, h: 88, keep_x: 1536, keep_y: 1024, keep_w: 384, keep_h: 56 });
}

#[test]
fn align_up_rounds_to_multiples() {
    assert_eq!(align_up(0, 8), 0);
    assert_eq!(align_up(1, 8), 8);
    assert_eq!(align_up(8, 8), 8);
    assert_eq!(align_up(1080, 8), 1080);
    assert_eq!(align_up(1081, 8), 1088);
    assert_eq!(align_up(576, 8), 576);
}

// ── samples, levels, colour ──────────────────────────────────────────────────

#[test]
fn plane_round_trips_at_both_depths() {
    let v8: Vec<u8> = (0..=255).collect();
    let f = plane_to_f32(&v8, 1);
    assert_eq!(f[17], 17.0);
    assert_eq!(f32_to_plane(&f, 1, 255.0), v8);
    let samples: Vec<u16> = (0..1024).collect();
    let bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
    let f = plane_to_f32(&bytes, 2);
    assert_eq!(f[1000], 1000.0);
    assert_eq!(f32_to_plane(&f, 2, 1023.0), bytes);
    // rounding and clamping
    assert_eq!(f32_to_plane(&[-3.0, 0.49, 0.5, 254.5, 300.0], 1, 255.0), vec![0, 0, 1, 255, 255]);
    assert_eq!(f32_to_plane(&[1023.4, 1024.0], 2, 1023.0), vec![0xff, 0x03, 0xff, 0x03]);
}

#[test]
fn levels_scale_with_depth() {
    let l8 = Levels::for_bps(1);
    assert_eq!((l8.max, l8.black, l8.y_range, l8.c_mid, l8.c_range), (255.0, 16.0, 219.0, 128.0, 224.0));
    let l10 = Levels::for_bps(2);
    assert_eq!((l10.max, l10.black, l10.y_range, l10.c_mid, l10.c_range), (1023.0, 64.0, 876.0, 512.0, 896.0));
}

#[test]
fn sigma_channel_mapping() {
    assert_eq!(sigma_channel(25.0, DpirModel::Gray), 25.0 / 255.0);
    assert_eq!(sigma_channel(25.0, DpirModel::Color), 25.0 / 219.0);
    assert_eq!(sigma_channel(0.0, DpirModel::Gray), 0.0);
}

#[test]
fn kr_kb_per_matrix() {
    assert_eq!(kr_kb(ColorSpace::Bt601), (0.299, 0.114));
    assert_eq!(kr_kb(ColorSpace::Bt709), (0.2126, 0.0722));
    assert_eq!(kr_kb(ColorSpace::Bt2020), (0.2627, 0.0593));
}

/// A `w×h` 4:2:0 frame in code values, from per-pixel (Y, U, V) at chroma
/// resolution (so it is exactly representable after 2×2 replication).
fn yuv_planes(w: usize, h: usize, f: impl Fn(usize, usize) -> (f32, f32, f32)) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let (cw, ch) = (w / 2, h / 2);
    let mut y = vec![0f32; w * h];
    let mut u = vec![0f32; cw * ch];
    let mut v = vec![0f32; cw * ch];
    for r in 0..h {
        for c in 0..w {
            y[r * w + c] = f(c, r).0;
        }
    }
    for r in 0..ch {
        for c in 0..cw {
            let (_, uu, vv) = f(c * 2, r * 2);
            u[r * cw + c] = uu;
            v[r * cw + c] = vv;
        }
    }
    (y, u, v)
}

#[test]
fn yuv_rgb_round_trip_is_exact_in_gamut() {
    for (cs, bps) in [(ColorSpace::Bt709, 1), (ColorSpace::Bt601, 1), (ColorSpace::Bt2020, 2)] {
        let lv = Levels::for_bps(bps);
        let k = if bps == 2 { 4.0 } else { 1.0 };
        // mid-grey-ish luma ramp with mild chroma: stays inside the RGB cube
        let (y, u, v) = yuv_planes(24, 12, |x, yy| (
            (60.0 + (x + yy) as f32 * 4.0) * k,
            (128.0 + (x as f32 - 10.0)) * k,
            (128.0 - (yy as f32 * 2.0 - 6.0)) * k,
        ));
        let rgb = yuv420_to_rgb(&y, &u, &v, 24, 12, cs, lv);
        assert!(rgb.iter().flatten().all(|c| (0.0..=1.0).contains(c)));
        let (y2, u2, v2) = rgb_to_yuv420(&rgb, 24, 12, cs, lv);
        let err = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(p, q)| (p - q).abs()).fold(0f32, f32::max);
        assert!(err(&y, &y2) < 0.01 * k, "{cs:?} luma err {}", err(&y, &y2));
        assert!(err(&u, &u2) < 0.01 * k, "{cs:?} cb err {}", err(&u, &u2));
        assert!(err(&v, &v2) < 0.01 * k, "{cs:?} cr err {}", err(&v, &v2));
    }
}

#[test]
fn yuv_rgb_round_trip_is_lossless_outside_the_cube_too() {
    // A BT.601 clip tagged BT.709 (what the pipeline does with untagged
    // sources) puts most pixels outside the RGB cube; the conversion must not
    // clamp, or the round trip loses them. Saturated bars at full chroma swing.
    let lv = Levels::for_bps(1);
    let (y, u, v) = yuv_planes(16, 8, |x, yy| (30.0 + x as f32 * 12.0, if x % 2 == 0 { 16.0 } else { 240.0 }, if yy % 4 < 2 { 240.0 } else { 16.0 }));
    let rgb = yuv420_to_rgb(&y, &u, &v, 16, 8, ColorSpace::Bt709, lv);
    assert!(rgb.iter().flatten().any(|c| *c < 0.0 || *c > 1.0), "the sample must leave the cube");
    let (y2, u2, v2) = rgb_to_yuv420(&rgb, 16, 8, ColorSpace::Bt709, lv);
    let err = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(p, q)| (p - q).abs()).fold(0f32, f32::max);
    assert!(err(&y, &y2) < 0.01 && err(&u, &u2) < 0.01 && err(&v, &v2) < 0.01, "{} {} {}", err(&y, &y2), err(&u, &u2), err(&v, &v2));
}

#[test]
fn yuv_rgb_known_values() {
    // BT.709 8-bit: reference white (235,128,128) → (1,1,1); black (16,128,128) → 0;
    // pure red-ish (Y=63, U=102, V=240) → (1, 0, 0) within a couple of code values.
    let lv = Levels::for_bps(1);
    let (y, u, v) = yuv_planes(2, 2, |_, _| (235.0, 128.0, 128.0));
    let rgb = yuv420_to_rgb(&y, &u, &v, 2, 2, ColorSpace::Bt709, lv);
    assert!(rgb.iter().flatten().all(|c| (c - 1.0).abs() < 1e-5));
    let (y, u, v) = yuv_planes(2, 2, |_, _| (16.0, 128.0, 128.0));
    let rgb = yuv420_to_rgb(&y, &u, &v, 2, 2, ColorSpace::Bt709, lv);
    assert!(rgb.iter().flatten().all(|c| c.abs() < 1e-5));
    let (y, u, v) = yuv_planes(2, 2, |_, _| (63.0, 102.0, 240.0));
    let rgb = yuv420_to_rgb(&y, &u, &v, 2, 2, ColorSpace::Bt709, lv);
    assert!((rgb[0][0] - 1.0).abs() < 0.01 && rgb[1][0] < 0.01 && rgb[2][0] < 0.01, "{:?}", [rgb[0][0], rgb[1][0], rgb[2][0]]);
    // and back
    let (y2, u2, v2) = rgb_to_yuv420(&rgb, 2, 2, ColorSpace::Bt709, lv);
    assert!((y2[0] - 63.0).abs() < 1.5 && (u2[0] - 102.0).abs() < 1.5 && (v2[0] - 240.0).abs() < 1.5, "{} {} {}", y2[0], u2[0], v2[0]);
}

#[test]
fn rgb_to_yuv_chroma_is_the_2x2_mean() {
    // The round-trip tests feed 2×2-replicated chroma, where "take one sample"
    // and "average the four" agree; here the four pixels of each 2×2 block
    // carry different chroma, so only the mean is right. Grey pixels with
    // R' = G' = B' + δ give Cb' ∝ −δ and Cr' ∝ δ per pixel.
    let lv = Levels::for_bps(1);
    let (kr, kb) = kr_kb(ColorSpace::Bt709);
    let (w, h) = (4, 2);
    let deltas = [0.10f32, -0.02, 0.06, 0.02, 0.0, 0.04, -0.08, 0.0];
    let mut rgb = [vec![0f32; w * h], vec![0f32; w * h], vec![0f32; w * h]];
    for i in 0..w * h {
        rgb[0][i] = 0.5 + deltas[i];
        rgb[1][i] = 0.5 + deltas[i];
        rgb[2][i] = 0.5;
    }
    let (_, u, v) = rgb_to_yuv420(&rgb, w, h, ColorSpace::Bt709, lv);
    assert_eq!((u.len(), v.len()), (2, 2));
    // block 0 = pixels (0,0),(1,0),(0,1),(1,1) = deltas 0,1,4,5; block 1 = 2,3,6,7
    for (c, idx) in [[0usize, 1, 4, 5], [2, 3, 6, 7]].iter().enumerate() {
        let mean_d = idx.iter().map(|&i| deltas[i]).sum::<f32>() / 4.0;
        // per pixel: Y' = kr·R' + kg·G' + kb·B' = 0.5 + (1 − kb)·δ, so
        // Cb' = (B' − Y')/(2(1 − kb)) = −δ/2 and Cr' = (R' − Y')/(2(1 − kr)) = δ(1 − (1 − kb))/(2(1 − kr)) = δ·kb/(2(1 − kr))
        let want_u = lv.c_mid - mean_d / 2.0 * lv.c_range;
        let want_v = lv.c_mid + mean_d * kb / (2.0 * (1.0 - kr)) * lv.c_range;
        assert!((u[c] - want_u).abs() < 1e-3, "block {c}: cb {} want {want_u}", u[c]);
        assert!((v[c] - want_v).abs() < 1e-3, "block {c}: cr {} want {want_v}", v[c]);
    }
}

#[test]
fn odd_sizes_reuse_the_edge_chroma() {
    let lv = Levels::for_bps(1);
    let w = 5;
    let h = 3;
    let y = vec![128.0; w * h];
    let u = vec![100.0; 2];
    let v = vec![160.0; 2];
    let rgb = yuv420_to_rgb(&y, &u, &v, w, h, ColorSpace::Bt709, lv);
    // every pixel sees the same chroma, so the RGB is flat
    for c in &rgb {
        assert!(c.iter().all(|&p| (p - c[0]).abs() < 1e-6));
    }
    let (y2, u2, v2) = rgb_to_yuv420(&rgb, w, h, ColorSpace::Bt709, lv);
    assert_eq!((y2.len(), u2.len(), v2.len()), (15, 2, 2));
}

// ── frames ───────────────────────────────────────────────────────────────────

/// An 8-bit 4:2:0 frame with luma from `f(x, y)` and neutral chroma.
fn frame8(w: usize, h: usize, f: impl Fn(usize, usize) -> u8) -> VideoFrame {
    let mut data = Vec::with_capacity(w * h * 3 / 2);
    for y in 0..h {
        for x in 0..w {
            data.push(f(x, y));
        }
    }
    data.extend(std::iter::repeat_n(128u8, (w / 2) * (h / 2) * 2));
    VideoFrame::new(Bytes::from(data), w as u32, h as u32, PixelFormat::Yuv420p, ColorSpace::Bt709, 0)
}

/// The same content as [`frame8`] at 10 bits (`v << 2`).
#[cfg(feature = "dpir")]
fn frame10(w: usize, h: usize, f: impl Fn(usize, usize) -> u8) -> VideoFrame {
    let f8 = frame8(w, h, f);
    let data: Vec<u8> = f8.data.iter().flat_map(|&v| ((v as u16) << 2).to_le_bytes()).collect();
    VideoFrame::new(Bytes::from(data), w as u32, h as u32, PixelFormat::Yuv420p10le, ColorSpace::Bt709, 0)
}

/// Deterministic noise in `-amp..=amp` (LCG), for synthetic noisy content.
#[cfg(feature = "dpir")]
fn noise(seed: u64, amp: i32) -> impl FnMut() -> i32 {
    let mut s = seed;
    move || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((s >> 33) % (2 * amp as u64 + 1)) as i32 - amp
    }
}

// ── the network, on the reduced fixtures ─────────────────────────────────────

#[cfg(feature = "dpir")]
mod with_candle {
    use super::*;
    use crate::filter::dpir::pth::{PthTensor, read_legacy_pth};
    use candle_core::{Device, Tensor};
    use std::collections::HashMap;

    const TINY_GRAY: &[u8] = include_bytes!("../../../../codec/tests/fixtures/dpir_tiny_gray.pth");
    const TINY_COLOR: &[u8] = include_bytes!("../../../../codec/tests/fixtures/dpir_tiny_color.pth");

    fn tensors(pth: &[u8]) -> HashMap<String, Tensor> {
        read_legacy_pth(pth)
            .unwrap()
            .into_iter()
            .map(|t| (t.name, Tensor::from_vec(t.data, t.shape, &Device::Cpu).unwrap()))
            .collect()
    }

    fn prepared(pth: &[u8], model: DpirModel, sigma: f32, tile: usize, overlap: usize) -> PreparedDpir {
        PreparedDpir::from_tensors(tensors(pth), model, sigma, Device::Cpu, tile, overlap).unwrap()
    }

    /// The fixture generator's LCG, to check values and not just shapes.
    fn lcg_uniform(s: &mut u64) -> f64 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (*s >> 11) as f64 / (1u64 << 53) as f64
    }

    #[test]
    fn legacy_pth_reader_matches_the_fixture() {
        let ts = read_legacy_pth(TINY_GRAY).unwrap();
        assert_eq!(ts.len(), 22, "UNetRes(nb=1) has 22 conv weights");
        let by_name = |n: &str| ts.iter().find(|t| t.name == n).unwrap_or_else(|| panic!("no {n}"));
        assert_eq!(by_name("m_head.weight").shape, vec![2, 2, 3, 3]);
        assert_eq!(by_name("m_tail.weight").shape, vec![1, 2, 3, 3]);
        assert_eq!(by_name("m_up3.0.weight").shape, vec![16, 8, 2, 2]);
        assert_eq!(by_name("m_down3.1.weight").shape, vec![16, 8, 2, 2]);
        assert!(ts.iter().all(|t| t.data.len() == t.shape.iter().product::<usize>()));
        assert!(ts.iter().all(|t| !t.name.starts_with('_')), "_metadata is not a tensor");
        // Values: the generator draws Kaiming-uniform from an LCG seeded with 1;
        // m_head is the first tensor drawn and has fan_in 2·3·3 = 18.
        let mut s = 1u64;
        let a = (6.0f64 / 18.0).sqrt();
        for (i, &v) in by_name("m_head.weight").data.iter().enumerate() {
            let want = ((lcg_uniform(&mut s) * 2.0 - 1.0) * a) as f32;
            assert!((v - want).abs() < 1e-7, "m_head[{i}] = {v}, want {want}");
        }
        // Storage keys are written in lexicographic order ('0','1','10',…), so
        // the 11th storage (key '10', m_body.0.res.0) sits after '1' in the
        // file, not after '9'; it must still come out as its own values —
        // the next 288 draws after the first ten tensors' 36+36+36+32+144+144+128+576+576+512.
        let mut s = 1u64;
        for _ in 0..(36 + 36 + 36 + 32 + 144 + 144 + 128 + 576 + 576 + 512) {
            lcg_uniform(&mut s);
        }
        let a = (6.0f64 / (16 * 9) as f64).sqrt();
        let body = by_name("m_body.0.res.0.weight");
        assert_eq!(body.shape, vec![16, 16, 3, 3]);
        for (i, &v) in body.data.iter().enumerate() {
            let want = ((lcg_uniform(&mut s) * 2.0 - 1.0) * a) as f32;
            assert!((v - want).abs() < 1e-7, "m_body.0.res.0[{i}] = {v}, want {want}");
        }
        let color = read_legacy_pth(TINY_COLOR).unwrap();
        assert_eq!(color.iter().find(|t| t.name == "m_head.weight").unwrap().shape, vec![2, 4, 3, 3]);
    }

    #[test]
    fn legacy_pth_reader_rejects_other_files() {
        let e = read_legacy_pth(b"PK\x03\x04not really a zip").unwrap_err().to_string();
        assert!(e.contains("zip-format"), "{e}");
        let e = read_legacy_pth(b"\x80\x02}q\x00.").unwrap_err().to_string();
        assert!(e.contains("magic"), "{e}");
        let truncated = &TINY_GRAY[..TINY_GRAY.len() - 100];
        let e = read_legacy_pth(truncated).unwrap_err().to_string();
        assert!(e.contains("truncated") || e.contains("reading"), "{e}");
        let mut extra = TINY_GRAY.to_vec();
        extra.extend_from_slice(&[0; 3]);
        let e = read_legacy_pth(&extra).unwrap_err().to_string();
        assert!(e.contains("trailing"), "{e}");
    }

    #[test]
    fn arch_is_inferred_from_the_state_dict() {
        let ts = tensors(TINY_GRAY);
        let shapes = ts.iter().map(|(k, t)| (k.clone(), t.dims().to_vec())).collect();
        let arch = super::super::net::Arch::infer(&shapes).unwrap();
        assert_eq!(arch, super::super::net::Arch { in_nc: 2, out_nc: 1, nc: [2, 4, 8, 16], nb: 1 });
        let mut partial: HashMap<String, Vec<usize>> = shapes.clone();
        partial.remove("m_down2.1.weight");
        assert!(super::super::net::Arch::infer(&partial).unwrap_err().to_string().contains("m_down2.1.weight"));
        let _ = PthTensor { name: String::new(), shape: vec![], data: vec![] };
    }

    #[test]
    fn model_variant_must_match_the_weights() {
        let e = PreparedDpir::from_tensors(tensors(TINY_GRAY), DpirModel::Color, 10.0, Device::Cpu, 0, 0).err().unwrap().to_string();
        assert!(e.contains("2 channels in") && e.contains("drunet_color.pth"), "{e}");
        let e = PreparedDpir::from_tensors(tensors(TINY_COLOR), DpirModel::Gray, 10.0, Device::Cpu, 0, 0).err().unwrap().to_string();
        assert!(e.contains("4 channels in") && e.contains("drunet_gray.pth"), "{e}");
        assert!(PreparedDpir::from_tensors(tensors(TINY_GRAY), DpirModel::Gray, 60.0, Device::Cpu, 0, 0).is_err());
    }

    fn luma(f: &VideoFrame) -> &[u8] {
        &f.data[..(f.width * f.height) as usize]
    }

    #[test]
    fn gray_path_touches_luma_only_and_keeps_geometry() {
        let f = frame8(40, 24, |x, y| ((x * 5 + y * 3) % 200) as u8 + 20);
        let d = prepared(TINY_GRAY, DpirModel::Gray, 20.0, 0, 0);
        let out = d.apply(&f).unwrap();
        assert_eq!((out.width, out.height, out.format), (40, 24, PixelFormat::Yuv420p));
        assert_eq!(out.data.len(), f.data.len());
        assert_eq!(&out.data[40 * 24..], &f.data[40 * 24..], "chroma is copied through");
        assert_ne!(luma(&out), luma(&f), "the network changed the luma");
        // A different σ gives a different answer: the noise-level channel is wired.
        let out2 = prepared(TINY_GRAY, DpirModel::Gray, 40.0, 0, 0).apply(&f).unwrap();
        assert_ne!(luma(&out), luma(&out2));
    }

    #[test]
    fn full_overlap_tiling_equals_the_whole_frame_exactly() {
        // With an overlap wider than the frame every tile is fed the whole
        // frame, so the tiled output must equal the whole-frame output bit for
        // bit — a check of the crop / copy bookkeeping, independent of the
        // network's receptive field.
        let f = frame8(40, 24, |x, y| ((x * 7 + y * 11) % 180) as u8 + 30);
        let whole = prepared(TINY_GRAY, DpirModel::Gray, 20.0, 0, 0).apply(&f).unwrap();
        let tiled = prepared(TINY_GRAY, DpirModel::Gray, 20.0, 16, 64).apply(&f).unwrap();
        assert_eq!(luma(&whole), luma(&tiled));
        // Real overlap: close but not identical; no overlap: worse.
        let diff = |a: &VideoFrame, b: &VideoFrame| luma(a).iter().zip(luma(b)).map(|(p, q)| (*p as i32 - *q as i32).abs()).max().unwrap();
        let ov8 = prepared(TINY_GRAY, DpirModel::Gray, 20.0, 16, 8).apply(&f).unwrap();
        let ov0 = prepared(TINY_GRAY, DpirModel::Gray, 20.0, 16, 0).apply(&f).unwrap();
        assert!(diff(&whole, &ov8) <= diff(&whole, &ov0), "overlap 8: {} vs none: {}", diff(&whole, &ov8), diff(&whole, &ov0));
        assert!(diff(&whole, &ov0) > 0, "with no overlap the seams must show on random weights");
    }

    #[test]
    fn ten_bit_matches_eight_bit_content() {
        let pat = |x: usize, y: usize| ((x * 3 + y * 5) % 160) as u8 + 40;
        let d = prepared(TINY_GRAY, DpirModel::Gray, 20.0, 0, 0);
        let o8 = d.apply(&frame8(32, 16, pat)).unwrap();
        let o10 = d.apply(&frame10(32, 16, pat)).unwrap();
        assert_eq!(o10.format, PixelFormat::Yuv420p10le);
        assert_eq!(o10.data.len(), 32 * 16 * 3 / 2 * 2);
        let y10: Vec<f32> = plane_to_f32(&o10.data[..32 * 16 * 2], 2);
        for (i, (&v8, v10)) in luma(&o8).iter().zip(&y10).enumerate() {
            assert!((v8 as f32 - v10 / 4.0).abs() <= 1.0, "sample {i}: 8-bit {v8} vs 10-bit {v10}");
        }
        // 10-bit chroma is copied through untouched
        assert_eq!(&o10.data[32 * 16 * 2..], &frame10(32, 16, pat).data[32 * 16 * 2..]);
    }

    #[test]
    fn color_path_runs_on_all_planes() {
        let f = frame8(32, 16, |x, y| ((x * 9 + y * 4) % 150) as u8 + 50);
        let d = prepared(TINY_COLOR, DpirModel::Color, 20.0, 16, 8);
        let out = d.apply(&f).unwrap();
        assert_eq!(out.data.len(), f.data.len());
        assert_ne!(luma(&out), luma(&f));
        assert_ne!(&out.data[32 * 16..], &f.data[32 * 16..], "chroma went through the RGB model");
        let out10 = d.apply(&frame10(32, 16, |x, y| ((x * 9 + y * 4) % 150) as u8 + 50)).unwrap();
        assert_eq!(out10.format, PixelFormat::Yuv420p10le);
        assert_eq!(out10.data.len(), f.data.len() * 2);
    }

    #[test]
    fn rejects_non_420_frames() {
        let d = prepared(TINY_GRAY, DpirModel::Gray, 20.0, 0, 0);
        let f = VideoFrame::new(Bytes::from(vec![0u8; 64]), 4, 4, PixelFormat::Nv12, ColorSpace::Bt709, 0);
        assert!(d.apply(&f).is_err());
        assert_eq!(d.device_name(), "cpu");
    }

    #[test]
    fn filter_chain_prepare_reports_a_missing_model() {
        // Point the override at a directory with no model in it.
        let dir = std::env::temp_dir().join("rivet-dpir-no-model-here");
        std::fs::create_dir_all(&dir).unwrap();
        // SAFETY: tests in this module that read the env run in this process;
        // the variable is restored below.
        let prev = std::env::var_os(ENV_MODEL);
        unsafe { std::env::set_var(ENV_MODEL, &dir) };
        let r = FilterChain::prepare(&[VideoFilter::Dpir { sigma: 10.0, color: false }]);
        match prev {
            Some(p) => unsafe { std::env::set_var(ENV_MODEL, p) },
            None => unsafe { std::env::remove_var(ENV_MODEL) },
        }
        let msg = format!("{:#}", r.err().expect("no model → error"));
        assert!(msg.contains("drunet_gray.pth") && msg.contains("curl -L"), "{msg}");
    }

    // ── the release model (needs the 130 MB file; `cargo test -- --ignored`) ──

    /// FNV-1a over the luma plane.
    fn fnv(bytes: &[u8]) -> u64 {
        bytes.iter().fold(0xcbf29ce484222325u64, |h, &b| (h ^ b as u64).wrapping_mul(0x100000001b3))
    }

    /// A 96×64 synthetic frame: gradient + stripes + uniform noise of ±25.
    fn noisy_frame() -> VideoFrame {
        let mut n = noise(7, 25);
        let mut data = Vec::with_capacity(96 * 64 * 3 / 2);
        for y in 0..64 {
            for x in 0..96 {
                let clean = 40 + (x * 3 / 2) as i32 + if (x / 12 + y / 12) % 2 == 0 { 30 } else { 0 };
                data.push((clean + n()).clamp(0, 255) as u8);
            }
        }
        data.extend(std::iter::repeat_n(128u8, 48 * 32 * 2));
        VideoFrame::new(Bytes::from(data), 96, 64, PixelFormat::Yuv420p, ColorSpace::Bt709, 0)
    }

    /// `tile` as [`PreparedDpir::from_tensors`] takes it: `0` = whole frame.
    fn release_model(model: DpirModel, device: Device, tile: usize) -> PreparedDpir {
        let path = model_path(model).expect("release model present (see the error for the download command)");
        let tensors = super::super::run::load_state_dict(&path, &device).unwrap();
        PreparedDpir::from_tensors(tensors, model, 25.0, device, tile, TILE_OVERLAP).unwrap()
    }

    /// The CPU path's golden hash: the release `drunet_gray.pth` on the frame
    /// above at σ=25 on the CPU. Pinned after measuring the same value at
    /// 1, 8 and 32 threads (`RAYON_NUM_THREADS`); a change here means the
    /// numerics changed — a candle upgrade, or a bug.
    #[test]
    #[ignore = "needs the 130 MB release model (RIVET_DPIR_MODEL / cache dir)"]
    fn release_gray_cpu_golden_hash() {
        let d = release_model(DpirModel::Gray, Device::Cpu, 0);
        let out = d.apply(&noisy_frame()).unwrap();
        let hash = fnv(luma(&out));
        eprintln!("golden hash: {hash:#018x}");
        // The denoiser must pull the noisy frame toward the clean one.
        let f = noisy_frame();
        let clean = |x: usize, y: usize| 40 + (x * 3 / 2) as i32 + if (x / 12 + y / 12) % 2 == 0 { 30 } else { 0 };
        let mse = |p: &[u8]| p.iter().enumerate().map(|(i, &v)| (v as i32 - clean(i % 96, i / 96)).pow(2) as f64).sum::<f64>() / p.len() as f64;
        let (before, after) = (mse(luma(&f)), mse(luma(&out)));
        eprintln!("mse before {before:.1} after {after:.1}");
        assert!(after < before / 4.0, "expected ≥ 6 dB gain, mse {before:.1} → {after:.1}");
        assert_eq!(hash, GOLDEN_GRAY_CPU, "golden hash moved: {hash:#018x}");
    }

    /// See [`release_gray_cpu_golden_hash`].
    const GOLDEN_GRAY_CPU: u64 = 0x210e7cc2e15489ab;

    /// CPU vs CUDA on the release model: six whole-frame 160×96 synthetic
    /// frames plus one 640×360 frame through real tiling (256-px tiles, the
    /// production overlap), so the seams are in the sample. The two devices
    /// reduce in different orders (and cuDNN picks its own convolution
    /// algorithms), so bit-exactness is off the table; `TOLERANCE` is the
    /// measured ceiling with headroom — the run prints what it measured, and
    /// `docs/filters/denoise.md` records it per build.
    #[cfg(feature = "dpir-cuda")]
    #[test]
    #[ignore = "needs the 130 MB release model and a CUDA device"]
    fn release_cpu_vs_cuda_within_tolerance() {
        const TOLERANCE: i32 = 2; // 8-bit code values
        let noisy = |w: usize, h: usize, seed: u64| {
            let mut n = noise(seed + 1, 20);
            let f = frame8(w, h, |x, y| ((x * 2 + y + seed as usize * 9) % 200) as i32 as u8);
            let mut data = f.data.to_vec();
            for v in &mut data[..w * h] {
                *v = (*v as i32 + n()).clamp(0, 255) as u8;
            }
            VideoFrame::new(Bytes::from(data), w as u32, h as u32, PixelFormat::Yuv420p, ColorSpace::Bt709, 0)
        };
        let mut worst = 0;
        let mut hist = [0usize; 8]; // diff 0..=6, and 7+
        let mut total = 0usize;
        let mut compare = |cpu: &PreparedDpir, cuda: &PreparedDpir, f: &VideoFrame| {
            let (a, b) = (cpu.apply(f).unwrap(), cuda.apply(f).unwrap());
            for (p, q) in luma(&a).iter().zip(luma(&b)) {
                let d = (*p as i32 - *q as i32).abs();
                worst = worst.max(d);
                hist[(d as usize).min(7)] += 1;
                total += 1;
            }
        };
        let cuda_dev = Device::new_cuda(0).unwrap();
        {
            let cpu = release_model(DpirModel::Gray, Device::Cpu, 0);
            let cuda = release_model(DpirModel::Gray, cuda_dev.clone(), 0);
            assert_eq!(cuda.device_name(), "cuda:0");
            for seed in 0..6u64 {
                compare(&cpu, &cuda, &noisy(160, 96, seed));
            }
        }
        {
            let cpu = release_model(DpirModel::Gray, Device::Cpu, 256);
            let cuda = release_model(DpirModel::Gray, cuda_dev, 256);
            compare(&cpu, &cuda, &noisy(640, 360, 11));
        }
        let differing = total - hist[0];
        eprintln!(
            "cpu vs cuda: max abs diff {worst}, {differing}/{total} samples differ ({:.3} %); by |diff| 0..=6,7+: {hist:?}",
            differing as f64 * 100.0 / total as f64
        );
        assert!(worst <= TOLERANCE, "max abs diff {worst} > {TOLERANCE}");
    }
}
