//! Scalar vs AVX2 tonemap: agreement on a real clip, and paired timing.
//!
//! ```text
//! tonemap_ab <raw yuv420p10le> <width> <height> [pq|hlg] [frames] [reps]
//! ```
//!
//! Reads up to `frames` frames of raw `yuv420p10le` (an `ffmpeg -pix_fmt
//! yuv420p10le -f rawvideo` dump of an HDR clip), runs both paths on every
//! frame and reports the largest per-sample difference and how many samples
//! differ at all. Then times the two: for each rep both paths run over the
//! same frames, alternating which goes first; the report is the median of the
//! per-rep ratios with the spread, after a same-path control (scalar against
//! scalar) that shows the smallest ratio this machine can resolve. One
//! binary, both paths, no environment variable involved.

use std::time::Instant;

use bytes::Bytes;
use codec::frame::{ColorSpace, PixelFormat, TransferFn, VideoFrame};
use codec::tonemap::{
    tonemap_yuv420p10le_bt2020_to_yuv420p_bt709_avx2,
    tonemap_yuv420p10le_bt2020_to_yuv420p_bt709_scalar,
};

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 { v[n / 2] } else { 0.5 * (v[n / 2 - 1] + v[n / 2]) }
}

fn time_frames(frames: &[VideoFrame], transfer: TransferFn, avx2: bool) -> f64 {
    let t = Instant::now();
    for f in frames {
        let out = if avx2 {
            tonemap_yuv420p10le_bt2020_to_yuv420p_bt709_avx2(f, transfer, None)
        } else {
            tonemap_yuv420p10le_bt2020_to_yuv420p_bt709_scalar(f, transfer, None)
        }
        .expect("tonemap");
        std::hint::black_box(out);
    }
    t.elapsed().as_secs_f64() * 1000.0 / frames.len() as f64
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("usage: tonemap_ab <raw yuv420p10le> <width> <height> [pq|hlg] [frames] [reps]");
        std::process::exit(2);
    }
    let path = &args[1];
    let w: u32 = args[2].parse().expect("width");
    let h: u32 = args[3].parse().expect("height");
    let transfer = match args.get(4).map(String::as_str).unwrap_or("pq") {
        "hlg" => TransferFn::AribStdB67,
        _ => TransferFn::St2084,
    };
    let max_frames: usize = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(10);
    let reps: usize = args.get(6).and_then(|s| s.parse().ok()).unwrap_or(9);

    let bytes = std::fs::read(path).expect("read raw");
    let frame_bytes = PixelFormat::Yuv420p10le.bytes_per_frame(w, h);
    let frames: Vec<VideoFrame> = bytes
        .chunks_exact(frame_bytes)
        .take(max_frames)
        .map(|c| VideoFrame::new(Bytes::copy_from_slice(c), w, h, PixelFormat::Yuv420p10le, ColorSpace::Bt2020, 0))
        .collect();
    assert!(!frames.is_empty(), "no whole frames in {path} at {w}x{h}");
    let avx2 = std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma");
    println!("{} frames of {w}x{h} {transfer:?}; avx2+fma available: {avx2}", frames.len());

    // ── agreement ──
    let mut max_diff = 0u8;
    let mut differing = 0usize;
    let mut total = 0usize;
    let mut hist = [0usize; 3];
    for f in &frames {
        let s = tonemap_yuv420p10le_bt2020_to_yuv420p_bt709_scalar(f, transfer, None).unwrap();
        let v = tonemap_yuv420p10le_bt2020_to_yuv420p_bt709_avx2(f, transfer, None).unwrap();
        for (a, b) in s.data.iter().zip(v.data.iter()) {
            let d = a.abs_diff(*b);
            max_diff = max_diff.max(d);
            differing += (d > 0) as usize;
            hist[(d as usize).min(2)] += 1;
        }
        total += s.data.len();
    }
    println!(
        "agreement: max |scalar - avx2| = {max_diff} LSB; {differing}/{total} samples differ ({:.4}%); |d|=1: {}, |d|>=2: {}",
        100.0 * differing as f64 / total as f64,
        hist[1],
        hist[2]
    );

    // ── timing ──
    let warm = time_frames(&frames, transfer, false);
    let _ = time_frames(&frames, transfer, true);
    println!("warm-up scalar {warm:.2} ms/frame");
    let mut control = Vec::with_capacity(reps);
    let mut ratios = Vec::with_capacity(reps);
    let mut scalar_ms = Vec::with_capacity(reps);
    let mut avx2_ms = Vec::with_capacity(reps);
    for r in 0..reps {
        // Control: the same path twice, alternating order.
        let (a, b) = if r % 2 == 0 {
            (time_frames(&frames, transfer, false), time_frames(&frames, transfer, false))
        } else {
            let b = time_frames(&frames, transfer, false);
            (time_frames(&frames, transfer, false), b)
        };
        control.push(a / b);
        // Paired: scalar vs avx2, alternating order.
        let (s, v) = if r % 2 == 0 {
            let s = time_frames(&frames, transfer, false);
            let v = time_frames(&frames, transfer, true);
            (s, v)
        } else {
            let v = time_frames(&frames, transfer, true);
            let s = time_frames(&frames, transfer, false);
            (s, v)
        };
        scalar_ms.push(s);
        avx2_ms.push(v);
        ratios.push(s / v);
    }
    let cmin = control.iter().cloned().fold(f64::INFINITY, f64::min);
    let cmax = control.iter().cloned().fold(0.0, f64::max);
    let rmin = ratios.iter().cloned().fold(f64::INFINITY, f64::min);
    let rmax = ratios.iter().cloned().fold(0.0, f64::max);
    println!(
        "control (scalar/scalar) median {:.3} spread {:.3}..{:.3} over {reps} reps",
        median(&mut control.clone()),
        cmin,
        cmax
    );
    println!(
        "scalar {:.2} ms/frame (min {:.2}), avx2 {:.2} ms/frame (min {:.2})",
        median(&mut scalar_ms.clone()),
        scalar_ms.iter().cloned().fold(f64::INFINITY, f64::min),
        median(&mut avx2_ms.clone()),
        avx2_ms.iter().cloned().fold(f64::INFINITY, f64::min)
    );
    println!(
        "speedup (scalar/avx2) median {:.3}x spread {:.3}..{:.3} over {reps} paired reps",
        median(&mut ratios.clone()),
        rmin,
        rmax
    );
}
