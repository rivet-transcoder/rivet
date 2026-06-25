//! Criterion bench for `codec::colorspace::scale_frame` bilinear.
//! Covers the common shrink cases in the reference transcode ladder.
//! See `perf/benchmarks/README.md` for wiring.

use bytes::Bytes;
use codec::colorspace;
use codec::frame::{ColorSpace, PixelFormat, VideoFrame};
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};

fn make_yuv420p_frame(width: u32, height: u32) -> VideoFrame {
    let y_size = (width as usize) * (height as usize);
    let uv_size = y_size / 4;
    let total = y_size + uv_size * 2;
    let mut data = Vec::with_capacity(total);
    for i in 0..total {
        data.push((i as u8).wrapping_mul(17));
    }
    VideoFrame::new(
        Bytes::from(data),
        width,
        height,
        PixelFormat::Yuv420p,
        ColorSpace::Bt709,
        0,
    )
}

fn bench_scale_bilinear(c: &mut Criterion) {
    let mut group = c.benchmark_group("scale_bilinear");

    let src = make_yuv420p_frame(1920, 1080);
    for (dw, dh, label) in [
        (1280, 720, "1080p_to_720p"),
        (960, 540, "1080p_to_540p"),
        (640, 360, "1080p_to_360p"),
        (480, 270, "1080p_to_270p"),
    ] {
        group.bench_with_input(
            BenchmarkId::from_parameter(label),
            &(dw, dh),
            |b, &(dw, dh)| {
                b.iter(|| {
                    let out = colorspace::scale_frame(black_box(&src), dw, dh).expect("scale");
                    black_box(out);
                })
            },
        );
    }

    // Also exercise the 720p source since 720p60 is the reference input.
    let src720 = make_yuv420p_frame(1280, 720);
    for (dw, dh, label) in [
        (960, 540, "720p_to_540p"),
        (640, 360, "720p_to_360p"),
        (320, 180, "720p_to_180p"),
    ] {
        group.bench_with_input(
            BenchmarkId::from_parameter(label),
            &(dw, dh),
            |b, &(dw, dh)| {
                b.iter(|| {
                    let out = colorspace::scale_frame(black_box(&src720), dw, dh).expect("scale");
                    black_box(out);
                })
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_scale_bilinear);
criterion_main!(benches);
