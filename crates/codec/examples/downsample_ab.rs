//! 4:4:4 → 4:2:0 chroma downsample: write both filters' output for
//! measurement, and time box vs Lanczos and Lanczos scalar vs AVX2.
//!
//! ```text
//! downsample_ab <raw yuv444p> <width> <height> <out prefix> [frames] [reps]
//! ```
//!
//! Writes `<prefix>_box.yuv` and `<prefix>_lanczos.yuv` (raw yuv420p, same
//! frame count) so a script can compare them against libswscale's output and
//! against the 4:4:4 source after a round trip. Timing is paired and
//! alternating with a same-path control, as `tonemap_ab` does.

use std::time::Instant;

use codec::colorspace::{
    downsample_chroma_444_to_420, downsample_plane_lanczos, downsample_plane_lanczos_scalar,
};

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 { v[n / 2] } else { 0.5 * (v[n / 2 - 1] + v[n / 2]) }
}

fn spread(v: &[f64]) -> (f64, f64) {
    (
        v.iter().cloned().fold(f64::INFINITY, f64::min),
        v.iter().cloned().fold(0.0, f64::max),
    )
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 {
        eprintln!("usage: downsample_ab <raw yuv444p> <width> <height> <out prefix> [frames] [reps]");
        std::process::exit(2);
    }
    let w: usize = args[2].parse().expect("width");
    let h: usize = args[3].parse().expect("height");
    let prefix = &args[4];
    let max_frames: usize = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(15);
    let reps: usize = args.get(6).and_then(|s| s.parse().ok()).unwrap_or(9);
    let bytes = std::fs::read(&args[1]).expect("read raw");
    let plane = w * h;
    let frames: Vec<&[u8]> = bytes.chunks_exact(3 * plane).take(max_frames).collect();
    assert!(!frames.is_empty());
    println!("{} frames of {w}x{h} yuv444p; avx2: {}", frames.len(), std::is_x86_feature_detected!("avx2"));

    let wide = |p: &[u8]| -> Vec<u16> { p.iter().map(|&v| v as u16).collect() };
    let mut out_box = Vec::new();
    let mut out_lz = Vec::new();
    for f in &frames {
        let (y, cb, cr) = (&f[..plane], &f[plane..2 * plane], &f[2 * plane..]);
        out_box.extend_from_slice(&downsample_chroma_444_to_420(y, cb, cr, w, h));
        out_lz.extend_from_slice(y);
        for p in [cb, cr] {
            let ds = downsample_plane_lanczos(&wide(p), w, h, 255);
            let ds_s = downsample_plane_lanczos_scalar(&wide(p), w, h, 255);
            assert_eq!(ds, ds_s, "AVX2 and scalar Lanczos must agree");
            out_lz.extend(ds.iter().map(|&v| v as u8));
        }
    }
    std::fs::write(format!("{prefix}_box.yuv"), &out_box).unwrap();
    std::fs::write(format!("{prefix}_lanczos.yuv"), &out_lz).unwrap();
    println!("wrote {prefix}_box.yuv and {prefix}_lanczos.yuv ({} bytes each)", out_box.len());

    // Timing over the Cb planes of every frame.
    let cbs: Vec<Vec<u16>> = frames.iter().map(|f| wide(&f[plane..2 * plane])).collect();
    let cbs8: Vec<&[u8]> = frames.iter().map(|f| &f[plane..2 * plane]).collect();
    let time_box = || {
        let t = Instant::now();
        for (f, cb) in frames.iter().zip(&cbs8) {
            std::hint::black_box(downsample_chroma_444_to_420(&f[..plane], cb, cb, w, h));
        }
        t.elapsed().as_secs_f64() * 1000.0 / frames.len() as f64
    };
    let time_lz = |avx2: bool| {
        let t = Instant::now();
        for cb in &cbs {
            let o = if avx2 {
                downsample_plane_lanczos(cb, w, h, 255)
            } else {
                downsample_plane_lanczos_scalar(cb, w, h, 255)
            };
            std::hint::black_box(o);
        }
        // two planes' worth, to match the box call (which does Cb + Cr)
        2.0 * t.elapsed().as_secs_f64() * 1000.0 / frames.len() as f64
    };
    let _ = (time_box(), time_lz(false), time_lz(true));
    let (mut ctrl, mut r_sv, mut r_bl) = (vec![], vec![], vec![]);
    let (mut ms_s, mut ms_v, mut ms_b) = (vec![], vec![], vec![]);
    for r in 0..reps {
        let (a, b) = (time_lz(false), time_lz(false));
        ctrl.push(if r % 2 == 0 { a / b } else { b / a });
        let (s, v) = if r % 2 == 0 { (time_lz(false), time_lz(true)) } else { let v = time_lz(true); (time_lz(false), v) };
        let bx = time_box();
        ms_s.push(s);
        ms_v.push(v);
        ms_b.push(bx);
        r_sv.push(s / v);
        r_bl.push(v / bx);
    }
    let (c0, c1) = spread(&ctrl);
    println!("control (scalar/scalar) median {:.3} spread {c0:.3}..{c1:.3} over {reps} reps", median(&mut ctrl.clone()));
    println!(
        "per frame (Cb+Cr): box {:.2} ms, lanczos scalar {:.2} ms, lanczos avx2 {:.2} ms",
        median(&mut ms_b.clone()),
        median(&mut ms_s.clone()),
        median(&mut ms_v.clone())
    );
    let (a0, a1) = spread(&r_sv);
    let (b0, b1) = spread(&r_bl);
    println!("lanczos scalar/avx2 median {:.3}x spread {a0:.3}..{a1:.3}", median(&mut r_sv.clone()));
    println!("lanczos-avx2 / box cost median {:.3}x spread {b0:.3}..{b1:.3}", median(&mut r_bl.clone()));
}
