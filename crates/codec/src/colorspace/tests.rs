use bytes::Bytes;

use crate::frame::{ColorSpace, PixelFormat, VideoFrame};

// Re-import all pub items from the colorspace module tree.
use super::{
    bilinear_scale_plane, bilinear_scale_plane_scalar, bilinear_scale_plane_u16,
    bilinear_scale_plane_u16_scalar, bt601_to_bt709_planes, bt601_to_bt709_planes_10bit,
    bt601_to_bt709_planes_10bit_scalar, bt601_to_bt709_planes_scalar,
    convert_to_yuv420p_bt709, downsample_444_to_420_frame, downsample_chroma_444_to_420,
    downsample_chroma_444_to_420_10bit, scale_frame,
};

// -------- BT.601 → BT.709 --------

fn synth_601_frame(w: usize, h: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut y = vec![0u8; w * h];
    let mut cb = vec![0u8; (w / 2) * (h / 2)];
    let mut cr = vec![0u8; (w / 2) * (h / 2)];
    for i in 0..y.len() {
        // Sweep limited range [16, 235].
        y[i] = 16 + ((i as u32 * 17) % 220) as u8;
    }
    for i in 0..cb.len() {
        cb[i] = 16 + ((i as u32 * 13) % 225) as u8;
        cr[i] = 16 + ((i as u32 * 23) % 225) as u8;
    }
    (y, cb, cr)
}

#[test]
fn bt601_to_bt709_neutral_gray_roundtrips() {
    // Cb=Cr=128 means ΔCb=ΔCr=0, so ΔY=0 and ΔCb709=ΔCr709=0.
    // Every luma value stays put; chroma stays at 128.
    for &y_val in &[16u8, 64, 128, 200, 235] {
        let w = 32;
        let h = 16;
        let mut y = vec![y_val; w * h];
        let mut cb = vec![128u8; (w / 2) * (h / 2)];
        let mut cr = vec![128u8; (w / 2) * (h / 2)];
        bt601_to_bt709_planes_scalar(&mut y, &mut cb, &mut cr, w, h);
        for v in &y {
            assert_eq!(*v, y_val, "Y with neutral chroma must round-trip");
        }
        for v in &cb {
            assert_eq!(*v, 128);
        }
        for v in &cr {
            assert_eq!(*v, 128);
        }
    }
}

#[test]
fn bt601_to_bt709_black_and_white_round_trip() {
    // Black (Y=16, Cb=Cr=128) and white (Y=235, Cb=Cr=128) must
    // round-trip unchanged because chroma deltas are zero.
    for &(y_val, label) in &[(16u8, "black"), (235u8, "white")] {
        let w = 64;
        let h = 32;
        let mut y = vec![y_val; w * h];
        let mut cb = vec![128u8; (w / 2) * (h / 2)];
        let mut cr = vec![128u8; (w / 2) * (h / 2)];
        bt601_to_bt709_planes(&mut y, &mut cb, &mut cr, w, h);
        for v in &y {
            assert_eq!(*v, y_val, "{} Y round-trip", label);
        }
        for v in &cb {
            assert_eq!(*v, 128, "{} Cb round-trip", label);
        }
        for v in &cr {
            assert_eq!(*v, 128, "{} Cr round-trip", label);
        }
    }
}

#[test]
fn bt601_to_bt709_scalar_vs_avx2_agree_256x256() {
    // Dense 256×256 synthetic plane; every AVX2 lane path exercised
    // plus the scalar tail (if width is not a 16-chroma multiple
    // we'd hit the tail — here 256 cw=128 is a multiple of 16, so
    // only the main path runs, which is what we want to gate).
    let w = 256;
    let h = 256;
    let (y0, cb0, cr0) = synth_601_frame(w, h);

    let mut y_s = y0.clone();
    let mut cb_s = cb0.clone();
    let mut cr_s = cr0.clone();
    bt601_to_bt709_planes_scalar(&mut y_s, &mut cb_s, &mut cr_s, w, h);

    let mut y_v = y0.clone();
    let mut cb_v = cb0.clone();
    let mut cr_v = cr0.clone();
    bt601_to_bt709_planes(&mut y_v, &mut cb_v, &mut cr_v, w, h);

    let mut max_y = 0i32;
    for i in 0..y_s.len() {
        let d = (y_s[i] as i32 - y_v[i] as i32).abs();
        if d > max_y {
            max_y = d;
        }
        assert!(d <= 1, "Y[{}] scalar={} avx2={}", i, y_s[i], y_v[i]);
    }
    for i in 0..cb_s.len() {
        assert!(
            (cb_s[i] as i32 - cb_v[i] as i32).abs() <= 1,
            "Cb[{}] scalar={} avx2={}",
            i,
            cb_s[i],
            cb_v[i]
        );
        assert!(
            (cr_s[i] as i32 - cr_v[i] as i32).abs() <= 1,
            "Cr[{}] scalar={} avx2={}",
            i,
            cr_s[i],
            cr_v[i]
        );
    }
}

#[test]
fn bt601_to_bt709_scalar_vs_avx2_agree_tail() {
    // 34 wide forces a 1-sample tail in the chroma loop (cw=17,
    // main covers 16, tail covers 1).
    let w = 34;
    let h = 16;
    let (y0, cb0, cr0) = synth_601_frame(w, h);

    let mut y_s = y0.clone();
    let mut cb_s = cb0.clone();
    let mut cr_s = cr0.clone();
    bt601_to_bt709_planes_scalar(&mut y_s, &mut cb_s, &mut cr_s, w, h);

    let mut y_v = y0.clone();
    let mut cb_v = cb0.clone();
    let mut cr_v = cr0.clone();
    bt601_to_bt709_planes(&mut y_v, &mut cb_v, &mut cr_v, w, h);

    for i in 0..y_s.len() {
        assert!(
            (y_s[i] as i32 - y_v[i] as i32).abs() <= 1,
            "Y[{}] scalar={} avx2={}",
            i,
            y_s[i],
            y_v[i]
        );
    }
    for i in 0..cb_s.len() {
        assert!((cb_s[i] as i32 - cb_v[i] as i32).abs() <= 1);
        assert!((cr_s[i] as i32 - cr_v[i] as i32).abs() <= 1);
    }
}

#[test]
fn bt601_to_bt709_clamps_ranges() {
    // After conversion, luma stays in [16, 235] and chroma in [16, 240].
    let w = 32;
    let h = 16;
    let (mut y, mut cb, mut cr) = synth_601_frame(w, h);
    bt601_to_bt709_planes(&mut y, &mut cb, &mut cr, w, h);
    for &v in cb.iter().chain(cr.iter()) {
        assert!((16..=240).contains(&v), "chroma {} out of limited range", v);
    }
    for &v in y.iter() {
        assert!((16..=235).contains(&v), "luma {} out of limited range", v);
    }
}

// -------- Bilinear scaler --------

fn make_ramp(w: usize, h: usize) -> Vec<u8> {
    (0..w * h).map(|i| ((i * 7 + i / w) & 0xff) as u8).collect()
}

#[test]
fn bilinear_scalar_vs_avx2_agree_2x() {
    let src_w = 64;
    let src_h = 32;
    let src = make_ramp(src_w, src_h);
    let dst_w = 128;
    let dst_h = 64;

    let scalar = bilinear_scale_plane_scalar(&src, src_w, src_h, dst_w, dst_h);
    let simd = bilinear_scale_plane(&src, src_w, src_h, dst_w, dst_h);

    assert_eq!(scalar.len(), simd.len());
    let mut max_diff = 0i32;
    for i in 0..scalar.len() {
        let d = (scalar[i] as i32 - simd[i] as i32).abs();
        if d > max_diff {
            max_diff = d;
        }
        assert!(
            d <= 1,
            "bilinear mismatch at {}: scalar={} simd={}",
            i,
            scalar[i],
            simd[i]
        );
    }
}

#[test]
fn bilinear_scalar_vs_avx2_agree_downscale() {
    let src_w = 128;
    let src_h = 72;
    let src = make_ramp(src_w, src_h);
    let dst_w = 64;
    let dst_h = 36;

    let scalar = bilinear_scale_plane_scalar(&src, src_w, src_h, dst_w, dst_h);
    let simd = bilinear_scale_plane(&src, src_w, src_h, dst_w, dst_h);

    for i in 0..scalar.len() {
        let d = (scalar[i] as i32 - simd[i] as i32).abs();
        assert!(
            d <= 1,
            "bilinear mismatch at {}: scalar={} simd={}",
            i,
            scalar[i],
            simd[i]
        );
    }
}

#[test]
fn bilinear_constant_input_yields_constant_output() {
    let src = vec![42u8; 64 * 32];
    let out = bilinear_scale_plane(&src, 64, 32, 128, 64);
    for &v in &out {
        assert_eq!(v, 42, "constant input must yield constant output");
    }
}

#[test]
fn bilinear_identity_scale() {
    let src = make_ramp(32, 32);
    let out = bilinear_scale_plane_scalar(&src, 32, 32, 32, 32);
    assert_eq!(out, src);
}

// -------- 10-bit (Squad-19) --------

fn make_10bit_frame_planar(w: usize, h: usize, y_val: u16, c_val: u16) -> VideoFrame {
    let y_samples = w * h;
    let c_samples = (w / 2) * (h / 2);
    let total = y_samples + 2 * c_samples;
    let mut buf = Vec::with_capacity(total * 2);
    for _ in 0..y_samples {
        buf.extend_from_slice(&y_val.to_le_bytes());
    }
    for _ in 0..(2 * c_samples) {
        buf.extend_from_slice(&c_val.to_le_bytes());
    }
    VideoFrame::new(
        Bytes::from(buf),
        w as u32,
        h as u32,
        PixelFormat::Yuv420p10le,
        ColorSpace::Bt2020,
        0,
    )
}

#[test]
fn convert_to_yuv420p_bt709_passthrough_10bit() {
    // The HDR-passthrough contract: a 10-bit `Yuv420p10le` frame
    // must come out of `convert_to_yuv420p_bt709` byte-identical
    // (no tonemap, no matrix conversion). The matrix conversion
    // is BT.601→BT.709 on 8-bit; for 10-bit we always passthrough
    // because the source could be HDR / wide-gamut and the matrix
    // shift would corrupt it.
    let frame = make_10bit_frame_planar(16, 16, 600, 512);
    let out = convert_to_yuv420p_bt709(&frame).expect("10-bit passthrough");
    assert_eq!(out.format, PixelFormat::Yuv420p10le);
    assert_eq!(out.width, 16);
    assert_eq!(out.height, 16);
    assert_eq!(out.data.len(), frame.data.len());
    assert_eq!(
        &out.data[..],
        &frame.data[..],
        "10-bit data must be byte-identical (no tonemap)"
    );
    assert_eq!(
        out.color_space,
        ColorSpace::Bt2020,
        "color space must not change"
    );
}

#[test]
fn scale_frame_10bit_constant_input_yields_constant_output() {
    let frame = make_10bit_frame_planar(64, 64, 600, 400);
    let out = scale_frame(&frame, 32, 32).expect("10-bit scale");
    assert_eq!(out.format, PixelFormat::Yuv420p10le);
    assert_eq!(out.width, 32);
    assert_eq!(out.height, 32);

    // Decode the output planes back to u16 and assert constant.
    let y_samples = 32 * 32;
    let c_samples = 16 * 16;
    let y_bytes = y_samples * 2;
    let c_bytes = c_samples * 2;
    assert_eq!(out.data.len(), y_bytes + 2 * c_bytes);

    // `read_u16le` is private in mod.rs but accessible to this child
    // module via `super::`.
    let y = super::read_u16le(&out.data[..y_bytes]);
    let u = super::read_u16le(&out.data[y_bytes..y_bytes + c_bytes]);
    let v = super::read_u16le(&out.data[y_bytes + c_bytes..y_bytes + 2 * c_bytes]);
    for &s in &y {
        assert_eq!(s, 600, "luma must be constant after bilinear");
    }
    for &s in u.iter().chain(v.iter()) {
        assert_eq!(s, 400, "chroma must be constant after bilinear");
    }
}

#[test]
fn scale_frame_10bit_identity_yields_byte_identical() {
    let frame = make_10bit_frame_planar(32, 32, 768, 256);
    // identity scale (same dims) early-returns clone — verify
    let out = scale_frame(&frame, 32, 32).expect("identity");
    assert_eq!(&out.data[..], &frame.data[..]);
}

#[test]
fn bilinear_10bit_scalar_clamps_inside_10bit_range() {
    // Synthetic ramp in 10-bit range; verify output is bounded.
    let mut src = vec![0u16; 64 * 32];
    for (i, s) in src.iter_mut().enumerate() {
        *s = (i as u16) % 1024;
    }
    let out = bilinear_scale_plane_u16_scalar(&src, 64, 32, 128, 64);
    for &v in &out {
        assert!(v <= 1023, "10-bit sample {} exceeds 1023", v);
    }
}

// -------- 10-bit AVX2 (Squad-29) --------

fn make_10bit_ramp(w: usize, h: usize) -> Vec<u16> {
    // Deterministic 10-bit ramp; cycles through 0..=1023.
    (0..w * h)
        .map(|i| ((i * 7 + i / w) % 1024) as u16)
        .collect()
}

#[test]
fn bilinear_10bit_scalar_vs_avx2_agree_2x_upscale() {
    // 2× upscale exercises every fractional weight in the source.
    let src_w = 64;
    let src_h = 32;
    let src = make_10bit_ramp(src_w, src_h);
    let dst_w = 128;
    let dst_h = 64;

    let scalar = bilinear_scale_plane_u16_scalar(&src, src_w, src_h, dst_w, dst_h);
    let simd = bilinear_scale_plane_u16(&src, src_w, src_h, dst_w, dst_h);

    assert_eq!(scalar.len(), simd.len());
    let mut max_diff = 0i32;
    for i in 0..scalar.len() {
        let d = (scalar[i] as i32 - simd[i] as i32).abs();
        if d > max_diff {
            max_diff = d;
        }
        assert!(
            d <= 1,
            "bilinear 10-bit mismatch at {}: scalar={} simd={}",
            i,
            scalar[i],
            simd[i]
        );
    }
}

#[test]
fn bilinear_10bit_scalar_vs_avx2_agree_downscale_1080p_to_720p() {
    // Headline case: 1920×1080 → 1280×720 luma plane. Same pattern
    // bench uses; gates the AVX2 main path (16-lane while loop runs
    // ~80 iters per row at dst_w=1280).
    let src_w = 1920;
    let src_h = 1080;
    let src = make_10bit_ramp(src_w, src_h);
    let dst_w = 1280;
    let dst_h = 720;

    let scalar = bilinear_scale_plane_u16_scalar(&src, src_w, src_h, dst_w, dst_h);
    let simd = bilinear_scale_plane_u16(&src, src_w, src_h, dst_w, dst_h);

    for i in 0..scalar.len() {
        let d = (scalar[i] as i32 - simd[i] as i32).abs();
        assert!(
            d <= 1,
            "bilinear 10-bit mismatch at {}: scalar={} simd={}",
            i,
            scalar[i],
            simd[i]
        );
    }
}

#[test]
fn bilinear_10bit_avx2_constant_input_yields_constant_output() {
    // Constant 600 (mid-luma in 10-bit limited range) should stay
    // exactly 600 through both axes of bilinear interp.
    let src = vec![600u16; 128 * 64];
    let out = bilinear_scale_plane_u16(&src, 128, 64, 256, 128);
    for &v in &out {
        assert_eq!(v, 600, "constant 10-bit input must yield constant output");
    }
}

#[test]
fn bilinear_10bit_avx2_max_value_clamped() {
    // Max-value (1023) input must stay clamped at 1023 — defensive
    // against the Q15 round trick pushing exactly-1023 to 1024.
    let src = vec![1023u16; 64 * 32];
    let out = bilinear_scale_plane_u16(&src, 64, 32, 128, 64);
    for &v in &out {
        assert!(v <= 1023, "10-bit AVX2 sample {} exceeds 1023", v);
        assert_eq!(v, 1023, "constant 1023 should stay 1023");
    }
}

#[test]
fn bilinear_10bit_narrow_width_falls_back_to_scalar() {
    // dst_w < 16 gates the AVX2 main path; dispatch should fall
    // back to scalar without panicking.
    let src_w = 8;
    let src_h = 8;
    let src = make_10bit_ramp(src_w, src_h);
    let dst_w = 4;
    let dst_h = 4;

    let scalar = bilinear_scale_plane_u16_scalar(&src, src_w, src_h, dst_w, dst_h);
    let dispatched = bilinear_scale_plane_u16(&src, src_w, src_h, dst_w, dst_h);

    assert_eq!(
        scalar, dispatched,
        "narrow strip should match scalar exactly"
    );
}

#[test]
fn bilinear_10bit_odd_dst_dims_handled() {
    // dst_w=17 forces a 1-sample tail (16 main + 1 tail).
    let src_w = 32;
    let src_h = 32;
    let src = make_10bit_ramp(src_w, src_h);
    let dst_w = 17;
    let dst_h = 9;

    let scalar = bilinear_scale_plane_u16_scalar(&src, src_w, src_h, dst_w, dst_h);
    let simd = bilinear_scale_plane_u16(&src, src_w, src_h, dst_w, dst_h);
    assert_eq!(scalar.len(), simd.len());
    for i in 0..scalar.len() {
        let d = (scalar[i] as i32 - simd[i] as i32).abs();
        assert!(
            d <= 1,
            "tail mismatch at {}: scalar={} simd={}",
            i,
            scalar[i],
            simd[i]
        );
    }
}

#[test]
fn bilinear_10bit_tall_narrow_strip() {
    // 16×512 → 16×256 — main loop runs once per row (dst_w=16),
    // many rows.
    let src_w = 16;
    let src_h = 512;
    let src = make_10bit_ramp(src_w, src_h);
    let dst_w = 16;
    let dst_h = 256;

    let scalar = bilinear_scale_plane_u16_scalar(&src, src_w, src_h, dst_w, dst_h);
    let simd = bilinear_scale_plane_u16(&src, src_w, src_h, dst_w, dst_h);
    for i in 0..scalar.len() {
        let d = (scalar[i] as i32 - simd[i] as i32).abs();
        assert!(d <= 1, "tall strip mismatch at {}", i);
    }
}

// -------- BT.601 → BT.709 10-bit (Squad-29) --------

fn synth_601_frame_10bit(w: usize, h: usize) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
    // Sweep limited 10-bit range [64, 940] for luma, [64, 960] for chroma.
    let mut y = vec![0u16; w * h];
    let mut cb = vec![0u16; (w / 2) * (h / 2)];
    let mut cr = vec![0u16; (w / 2) * (h / 2)];
    for i in 0..y.len() {
        y[i] = 64 + ((i as u32 * 17) % 877) as u16;
    }
    for i in 0..cb.len() {
        cb[i] = 64 + ((i as u32 * 13) % 897) as u16;
        cr[i] = 64 + ((i as u32 * 23) % 897) as u16;
    }
    (y, cb, cr)
}

#[test]
fn bt601_to_bt709_10bit_neutral_gray_roundtrips() {
    // Cb=Cr=512 (10-bit chroma center) — every gray luma round-trips.
    // 10-bit limited-range luma analogues of 16/64/128/200/235:
    //   16 << 2 = 64,  64 << 2 = 256,  128 << 2 = 512,
    //   200 << 2 = 800, 235 << 2 = 940.
    for &y_val in &[64u16, 256, 512, 800, 940] {
        let w = 32;
        let h = 16;
        let mut y = vec![y_val; w * h];
        let mut cb = vec![512u16; (w / 2) * (h / 2)];
        let mut cr = vec![512u16; (w / 2) * (h / 2)];
        bt601_to_bt709_planes_10bit_scalar(&mut y, &mut cb, &mut cr, w, h);
        for v in &y {
            assert_eq!(*v, y_val, "Y with neutral chroma must round-trip");
        }
        for v in &cb {
            assert_eq!(*v, 512);
        }
        for v in &cr {
            assert_eq!(*v, 512);
        }
    }
}

#[test]
fn bt601_to_bt709_10bit_scalar_vs_avx2_agree_256x256() {
    // 256×256 → cw=128, multiple of 16 for chroma. Main AVX2 path
    // covers the entire plane.
    let w = 256;
    let h = 256;
    let (y0, cb0, cr0) = synth_601_frame_10bit(w, h);

    let mut y_s = y0.clone();
    let mut cb_s = cb0.clone();
    let mut cr_s = cr0.clone();
    bt601_to_bt709_planes_10bit_scalar(&mut y_s, &mut cb_s, &mut cr_s, w, h);

    let mut y_v = y0.clone();
    let mut cb_v = cb0.clone();
    let mut cr_v = cr0.clone();
    bt601_to_bt709_planes_10bit(&mut y_v, &mut cb_v, &mut cr_v, w, h);

    for i in 0..y_s.len() {
        let d = (y_s[i] as i32 - y_v[i] as i32).abs();
        assert!(d <= 1, "Y[{}] scalar={} avx2={}", i, y_s[i], y_v[i]);
    }
    for i in 0..cb_s.len() {
        assert!(
            (cb_s[i] as i32 - cb_v[i] as i32).abs() <= 1,
            "Cb[{}] scalar={} avx2={}",
            i,
            cb_s[i],
            cb_v[i]
        );
        assert!(
            (cr_s[i] as i32 - cr_v[i] as i32).abs() <= 1,
            "Cr[{}] scalar={} avx2={}",
            i,
            cr_s[i],
            cr_v[i]
        );
    }
}

#[test]
fn bt601_to_bt709_10bit_scalar_vs_avx2_agree_tail() {
    // 34 wide forces a 1-sample chroma tail (cw=17, main covers 16,
    // tail covers 1).
    let w = 34;
    let h = 16;
    let (y0, cb0, cr0) = synth_601_frame_10bit(w, h);

    let mut y_s = y0.clone();
    let mut cb_s = cb0.clone();
    let mut cr_s = cr0.clone();
    bt601_to_bt709_planes_10bit_scalar(&mut y_s, &mut cb_s, &mut cr_s, w, h);

    let mut y_v = y0.clone();
    let mut cb_v = cb0.clone();
    let mut cr_v = cr0.clone();
    bt601_to_bt709_planes_10bit(&mut y_v, &mut cb_v, &mut cr_v, w, h);

    for i in 0..y_s.len() {
        assert!(
            (y_s[i] as i32 - y_v[i] as i32).abs() <= 1,
            "Y[{}] scalar={} avx2={}",
            i,
            y_s[i],
            y_v[i]
        );
    }
    for i in 0..cb_s.len() {
        assert!((cb_s[i] as i32 - cb_v[i] as i32).abs() <= 1);
        assert!((cr_s[i] as i32 - cr_v[i] as i32).abs() <= 1);
    }
}

#[test]
fn bt601_to_bt709_10bit_clamps_ranges() {
    // After conversion, luma stays in [64, 940] and chroma in [64, 960].
    let w = 32;
    let h = 16;
    let (mut y, mut cb, mut cr) = synth_601_frame_10bit(w, h);
    bt601_to_bt709_planes_10bit(&mut y, &mut cb, &mut cr, w, h);
    for &v in cb.iter().chain(cr.iter()) {
        assert!(
            (64..=960).contains(&v),
            "chroma {} out of 10-bit limited range",
            v
        );
    }
    for &v in y.iter() {
        assert!(
            (64..=940).contains(&v),
            "luma {} out of 10-bit limited range",
            v
        );
    }
}

#[test]
fn bt601_to_bt709_10bit_extreme_chroma_clamped_at_high_end() {
    // Chroma at the limited-range max should produce in-range output.
    let w = 32;
    let h = 16;
    let mut y = vec![940u16; w * h];
    let mut cb = vec![960u16; (w / 2) * (h / 2)];
    let mut cr = vec![960u16; (w / 2) * (h / 2)];
    bt601_to_bt709_planes_10bit(&mut y, &mut cb, &mut cr, w, h);
    for &v in y.iter() {
        assert!(v <= 940, "luma {} > 940 (clamp violated)", v);
    }
    for &v in cb.iter().chain(cr.iter()) {
        assert!(v <= 960, "chroma {} > 960 (clamp violated)", v);
    }
}

// -------- 4:4:4 → 4:2:0 chroma downsample (Squad-31, roadmap #6) --------

#[test]
fn downsample_4x4_box_average_8bit_hand_verified() {
    // 4×4 chroma plane (16 samples) → 2×2 output. Hand-compute the
    // 4 averages so the test is its own oracle.
    //
    //   Cb = [ 10  20 |  30  40
    //          50  60 |  70  80
    //          ---------+--------
    //          90 100 | 110 120
    //         130 140 | 150 160 ]
    //
    // Block (0,0): (10+20+50+60+2)>>2 = 142>>2 = 35
    // Block (1,0): (30+40+70+80+2)>>2 = 222>>2 = 55
    // Block (0,1): (90+100+130+140+2)>>2 = 462>>2 = 115
    // Block (1,1): (110+120+150+160+2)>>2 = 542>>2 = 135
    let cb: Vec<u8> = vec![
        10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160,
    ];
    // Cr distinct so we know the per-plane logic is independent.
    let cr: Vec<u8> = vec![
        5, 15, 25, 35, 45, 55, 65, 75, 85, 95, 105, 115, 125, 135, 145, 155,
    ];
    // Y plane is unchanged — pick a recognizable ramp so we can verify
    // the copy is verbatim.
    let y: Vec<u8> = (0..16).map(|i| i as u8 * 8).collect();

    let out = downsample_chroma_444_to_420(&y, &cb, &cr, 4, 4);
    // Expected layout: 16 Y bytes || 4 Cb bytes || 4 Cr bytes.
    assert_eq!(out.len(), 16 + 4 + 4);
    assert_eq!(&out[..16], y.as_slice(), "Y must round-trip verbatim");
    // Cb output (4 samples in row-major 2×2 order)
    assert_eq!(out[16], 35, "Cb block (0,0)");
    assert_eq!(out[17], 55, "Cb block (1,0)");
    assert_eq!(out[18], 115, "Cb block (0,1)");
    assert_eq!(out[19], 135, "Cb block (1,1)");
    // Cr output
    assert_eq!(out[20], 30, "Cr block (0,0): (5+15+45+55+2)>>2 = 30");
    assert_eq!(out[21], 50, "Cr block (1,0): (25+35+65+75+2)>>2 = 50");
    assert_eq!(out[22], 110, "Cr block (0,1): (85+95+125+135+2)>>2 = 110");
    assert_eq!(out[23], 130, "Cr block (1,1): (105+115+145+155+2)>>2 = 130");
}

#[test]
fn downsample_constant_input_8bit_yields_constant_output() {
    // Cb=128 (chroma midpoint) — average of four 128s with rounding
    // is still 128. Round-trip identity for any constant.
    let w = 16;
    let h = 16;
    let y = vec![64u8; w * h];
    let cb = vec![128u8; w * h];
    let cr = vec![128u8; w * h];
    let out = downsample_chroma_444_to_420(&y, &cb, &cr, w, h);
    let cw = (w + 1) / 2;
    let ch = (h + 1) / 2;
    assert_eq!(out.len(), w * h + 2 * cw * ch);
    // Y unchanged.
    for i in 0..w * h {
        assert_eq!(out[i], 64, "Y[{}] should be 64", i);
    }
    // Cb / Cr: each output sample == 128.
    for i in (w * h)..(w * h + 2 * cw * ch) {
        assert_eq!(out[i], 128, "chroma[{}] should be 128", i - w * h);
    }
}

#[test]
fn downsample_odd_dimensions_clamp_policy() {
    // 7×7 input → 4×4 output. The rightmost column of 2×2 blocks
    // (cx=3) and bottom row (cy=3) straddle exactly one source row /
    // column; clamp policy reuses the in-bounds neighbour.
    //
    //   plane[cx=3, cy=0] takes samples (6, 0), (6, 0), (6, 1), (6, 1)
    //     because x1 = min(7, w-1=6) = 6 — both x0 and x1 = 6.
    //   So the corner sample reduces to a 1-sample average:
    //     (s + s + s' + s' + 2) >> 2 = (s + s')/2 with rounding.
    //
    // Easiest verification: constant-fill input → constant output
    // even at the odd boundary.
    let w = 7;
    let h = 7;
    let y = vec![100u8; w * h];
    let cb = vec![128u8; w * h];
    let cr = vec![64u8; w * h];
    let out = downsample_chroma_444_to_420(&y, &cb, &cr, w, h);
    let cw = (w + 1) / 2; // 4
    let ch = (h + 1) / 2; // 4
    assert_eq!(cw, 4);
    assert_eq!(ch, 4);
    assert_eq!(out.len(), w * h + 2 * cw * ch);
    // Y verbatim.
    for i in 0..w * h {
        assert_eq!(out[i], 100);
    }
    // Cb constant 128.
    for cx in 0..cw {
        for cy in 0..ch {
            let idx = w * h + cy * cw + cx;
            assert_eq!(out[idx], 128, "Cb[{},{}] expected 128", cx, cy);
        }
    }
    // Cr constant 64.
    for cx in 0..cw {
        for cy in 0..ch {
            let idx = w * h + cw * ch + cy * cw + cx;
            assert_eq!(out[idx], 64, "Cr[{},{}] expected 64", cx, cy);
        }
    }
}

#[test]
fn downsample_10bit_constant_input_yields_constant_output() {
    // Cb=512 (10-bit midpoint = 1024/2). Identity for constant input.
    let w = 16;
    let h = 16;
    let y = vec![400u16; w * h];
    let cb = vec![512u16; w * h];
    let cr = vec![512u16; w * h];
    let out = downsample_chroma_444_to_420_10bit(&y, &cb, &cr, w, h);
    let cw = (w + 1) / 2;
    let ch = (h + 1) / 2;
    assert_eq!(out.len(), 2 * (w * h + 2 * cw * ch), "10-bit byte count");

    // Verify each u16 LE sample. Y plane.
    for i in 0..w * h {
        let s = u16::from_le_bytes([out[i * 2], out[i * 2 + 1]]);
        assert_eq!(s, 400, "Y[{}] should be 400", i);
    }
    // Cb plane.
    let cb_byte_off = w * h * 2;
    for i in 0..cw * ch {
        let s = u16::from_le_bytes([out[cb_byte_off + i * 2], out[cb_byte_off + i * 2 + 1]]);
        assert_eq!(s, 512, "Cb[{}] should be 512", i);
    }
    // Cr plane.
    let cr_byte_off = cb_byte_off + cw * ch * 2;
    for i in 0..cw * ch {
        let s = u16::from_le_bytes([out[cr_byte_off + i * 2], out[cr_byte_off + i * 2 + 1]]);
        assert_eq!(s, 512, "Cr[{}] should be 512", i);
    }
}

#[test]
fn downsample_10bit_max_value_no_overflow() {
    // 4 × 1023 + 2 = 4094 fits in u16 (max 65535) and even in i16
    // (max 32767). The u32 accumulator gives plenty of headroom.
    // Verify a full-1023 input doesn't wrap to 0.
    let w = 4;
    let h = 4;
    let y = vec![1023u16; w * h];
    let cb = vec![1023u16; w * h];
    let cr = vec![1023u16; w * h];
    let out = downsample_chroma_444_to_420_10bit(&y, &cb, &cr, w, h);
    let cw = (w + 1) / 2;
    let ch = (h + 1) / 2;

    // Y verbatim (1023).
    for i in 0..w * h {
        let s = u16::from_le_bytes([out[i * 2], out[i * 2 + 1]]);
        assert_eq!(s, 1023, "Y[{}]", i);
    }
    // Cb / Cr: (1023 + 1023 + 1023 + 1023 + 2) >> 2 = 4094 >> 2 = 1023.
    let cb_byte_off = w * h * 2;
    for i in 0..2 * cw * ch {
        let s = u16::from_le_bytes([out[cb_byte_off + i * 2], out[cb_byte_off + i * 2 + 1]]);
        assert_eq!(s, 1023, "chroma[{}] should be 1023 (no overflow)", i);
    }
}

#[test]
fn downsample_10bit_4x4_box_average_hand_verified() {
    // Same 4×4 hand-verified case as 8-bit but in 10-bit.
    let cb_u: Vec<u16> = vec![
        10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160,
    ];
    let cr_u: Vec<u16> = vec![
        500, 600, 700, 800, 500, 600, 700, 800, 500, 600, 700, 800, 500, 600, 700, 800,
    ];
    let y_u: Vec<u16> = (0..16).map(|i| i as u16 * 50).collect();

    let out = downsample_chroma_444_to_420_10bit(&y_u, &cb_u, &cr_u, 4, 4);
    // Y bytes: 16 × 2 = 32. Then 4 Cb (8 bytes) + 4 Cr (8 bytes) = 48.
    assert_eq!(out.len(), 32 + 8 + 8);

    // Y round-trip.
    for i in 0..16 {
        let s = u16::from_le_bytes([out[i * 2], out[i * 2 + 1]]);
        assert_eq!(s, i as u16 * 50, "Y[{}]", i);
    }
    // Cb expected (35, 55, 115, 135) — same as 8-bit case.
    let cb_off = 32;
    let cb0 = u16::from_le_bytes([out[cb_off], out[cb_off + 1]]);
    let cb1 = u16::from_le_bytes([out[cb_off + 2], out[cb_off + 3]]);
    let cb2 = u16::from_le_bytes([out[cb_off + 4], out[cb_off + 5]]);
    let cb3 = u16::from_le_bytes([out[cb_off + 6], out[cb_off + 7]]);
    assert_eq!(cb0, 35);
    assert_eq!(cb1, 55);
    assert_eq!(cb2, 115);
    assert_eq!(cb3, 135);
    // Cr: rows are identical, so each 2×2 block average == row average.
    // (500+600+500+600+2)>>2 = 2202>>2 = 550
    // (700+800+700+800+2)>>2 = 3002>>2 = 750
    let cr_off = cb_off + 8;
    let cr0 = u16::from_le_bytes([out[cr_off], out[cr_off + 1]]);
    let cr1 = u16::from_le_bytes([out[cr_off + 2], out[cr_off + 3]]);
    assert_eq!(cr0, 550);
    assert_eq!(cr1, 750);
}

#[test]
fn downsample_frame_yuv444p10le_to_yuv420p10le() {
    // High-level frame wrapper. Constant 4:4:4 10-bit → constant
    // 4:2:0 10-bit, dims preserved, format flipped.
    let w = 16;
    let h = 16;
    let plane = w * h;
    let mut buf = Vec::with_capacity(3 * plane * 2);
    for _ in 0..plane {
        buf.extend_from_slice(&500u16.to_le_bytes()); // Y
    }
    for _ in 0..plane {
        buf.extend_from_slice(&512u16.to_le_bytes()); // Cb
    }
    for _ in 0..plane {
        buf.extend_from_slice(&512u16.to_le_bytes()); // Cr
    }
    let frame = VideoFrame::new(
        Bytes::from(buf),
        w as u32,
        h as u32,
        PixelFormat::Yuv444p10le,
        ColorSpace::Bt2020,
        42,
    );
    let out = downsample_444_to_420_frame(&frame).expect("downsample");
    assert_eq!(out.format, PixelFormat::Yuv420p10le);
    assert_eq!(out.width, w as u32);
    assert_eq!(out.height, h as u32);
    assert_eq!(out.pts, 42, "PTS preserved");
    assert_eq!(out.color_space, ColorSpace::Bt2020, "color_space preserved");

    // Spot-check the output samples.
    let cw = w / 2;
    let ch = h / 2;
    let expected_bytes = 2 * (w * h + 2 * cw * ch);
    assert_eq!(out.data.len(), expected_bytes);

    // First Y sample = 500.
    let y0 = u16::from_le_bytes([out.data[0], out.data[1]]);
    assert_eq!(y0, 500);
    // First Cb sample (after Y plane) = 512.
    let cb0 = u16::from_le_bytes([out.data[w * h * 2], out.data[w * h * 2 + 1]]);
    assert_eq!(cb0, 512);
}

#[test]
fn downsample_frame_yuva444p10le_drops_alpha() {
    // 4-plane source, alpha is 16-bit precision. Output is plain
    // Yuv420p10le (no alpha plane).
    let w = 8;
    let h = 8;
    let plane = w * h;
    let mut buf = Vec::with_capacity(4 * plane * 2);
    for _ in 0..plane {
        buf.extend_from_slice(&600u16.to_le_bytes());
    }
    for _ in 0..plane {
        buf.extend_from_slice(&500u16.to_le_bytes());
    }
    for _ in 0..plane {
        buf.extend_from_slice(&500u16.to_le_bytes());
    }
    for _ in 0..plane {
        // Alpha — 16-bit, would have value 65535 if it survived.
        buf.extend_from_slice(&65535u16.to_le_bytes());
    }
    let frame = VideoFrame::new(
        Bytes::from(buf),
        w as u32,
        h as u32,
        PixelFormat::Yuva444p10le,
        ColorSpace::Bt2020,
        7,
    );
    let out = downsample_444_to_420_frame(&frame).expect("downsample with alpha");
    assert_eq!(out.format, PixelFormat::Yuv420p10le);
    // Output byte count: only Y/Cb/Cr — NO alpha plane.
    let cw = w / 2;
    let ch = h / 2;
    let expected = 2 * (w * h + 2 * cw * ch);
    assert_eq!(out.data.len(), expected);
    // Verify alpha wasn't smuggled in (no 65535 samples).
    for i in (0..out.data.len()).step_by(2) {
        let s = u16::from_le_bytes([out.data[i], out.data[i + 1]]);
        assert!(
            s < 1024 || s == 65535 && false,
            "stray alpha sample {} at {}",
            s,
            i
        );
        assert_ne!(s, 65535, "alpha plane leaked into output");
    }
}

#[test]
fn downsample_frame_rejects_non_444() {
    // 4:2:0 input must error — the frame is already in target format.
    let w = 16;
    let h = 16;
    let plane = w * h;
    let mut buf = Vec::with_capacity(plane + 2 * (plane / 4));
    buf.resize(plane + 2 * (plane / 4), 128);
    let frame = VideoFrame::new(
        Bytes::from(buf),
        w as u32,
        h as u32,
        PixelFormat::Yuv420p,
        ColorSpace::Bt709,
        0,
    );
    let err = downsample_444_to_420_frame(&frame).unwrap_err();
    assert!(format!("{}", err).contains("expected 4:4:4 input"));
}
