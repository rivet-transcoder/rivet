//! Criterion bench for `codec::colorspace::convert_to_yuv420p_bt709`
//! on the NV12 input path. See `perf/benchmarks/README.md` for wiring.

use bytes::Bytes;
use codec::colorspace;
use codec::frame::{ColorSpace, PixelFormat, VideoFrame};
use criterion::{Criterion, black_box, criterion_group, criterion_main};

fn make_nv12_frame(width: u32, height: u32) -> VideoFrame {
    let y_size = (width as usize) * (height as usize);
    let uv_size = y_size / 2;
    let mut data = Vec::with_capacity(y_size + uv_size);
    // Deterministic pattern so the optimizer can't constant-fold.
    for i in 0..(y_size + uv_size) {
        data.push((i as u8).wrapping_mul(31));
    }
    VideoFrame::new(
        Bytes::from(data),
        width,
        height,
        PixelFormat::Nv12,
        ColorSpace::Bt709,
        0,
    )
}

fn bench_nv12_to_yuv420p(c: &mut Criterion) {
    let mut group = c.benchmark_group("nv12_to_yuv420p");

    for (w, h, label) in [(1280, 720, "720p"), (1920, 1080, "1080p")] {
        let frame = make_nv12_frame(w, h);
        group.bench_function(label, |b| {
            b.iter(|| {
                let out = colorspace::convert_to_yuv420p_bt709(black_box(&frame)).expect("convert");
                black_box(out);
            })
        });
    }

    group.finish();
}

criterion_group!(benches, bench_nv12_to_yuv420p);
criterion_main!(benches);
