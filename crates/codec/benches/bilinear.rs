//! Criterion bench for the bilinear scaler — scalar reference vs
//! the AVX2 dispatch on a 1920×1080 → 1280×720 Y plane (the most
//! common downscale in the reference ladder). User memory rule: if
//! AVX2 doesn't show ≥1.5× gain on this bench, the specialization
//! is reverted.

use codec::colorspace::{
    bilinear_scale_plane, bilinear_scale_plane_scalar, bilinear_scale_plane_u16,
    bilinear_scale_plane_u16_scalar,
};
use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};

fn make_plane(w: usize, h: usize) -> Vec<u8> {
    (0..w * h)
        .map(|i| (i.wrapping_mul(31) as u8).wrapping_add((i / w) as u8))
        .collect()
}

fn make_plane_u16(w: usize, h: usize) -> Vec<u16> {
    // 10-bit ramp; cycles through 0..=1023.
    (0..w * h)
        .map(|i| ((i.wrapping_mul(31) + i / w) % 1024) as u16)
        .collect()
}

fn bench_bilinear(c: &mut Criterion) {
    // 1080p Y plane → 720p Y plane is the R3 hotspot.
    let src_w = 1920usize;
    let src_h = 1080usize;
    let dst_w = 1280usize;
    let dst_h = 720usize;
    let src = make_plane(src_w, src_h);

    let mut group = c.benchmark_group("bilinear");
    group.throughput(Throughput::Elements((dst_w * dst_h) as u64));

    group.bench_function("1080p_to_720p_scalar", |b| {
        b.iter(|| {
            let dst = bilinear_scale_plane_scalar(black_box(&src), src_w, src_h, dst_w, dst_h);
            black_box(dst);
        })
    });

    group.bench_function("1080p_to_720p_dispatch", |b| {
        b.iter(|| {
            let dst = bilinear_scale_plane(black_box(&src), src_w, src_h, dst_w, dst_h);
            black_box(dst);
        })
    });

    group.finish();
}

/// Squad-29: 10-bit bilinear scalar vs AVX2 dispatch.
/// Same shape as the 8-bit bench: 1920×1080 → 1280×720 Y plane.
/// Target speedup ≥3× (realistic floor for u16 lanes vs u8).
fn bench_bilinear_10bit_avx2_vs_scalar(c: &mut Criterion) {
    let src_w = 1920usize;
    let src_h = 1080usize;
    let dst_w = 1280usize;
    let dst_h = 720usize;
    let src = make_plane_u16(src_w, src_h);

    let mut group = c.benchmark_group("bilinear_10bit_avx2_vs_scalar");
    group.throughput(Throughput::Elements((dst_w * dst_h) as u64));

    group.bench_function("1080p_to_720p_scalar", |b| {
        b.iter(|| {
            let dst = bilinear_scale_plane_u16_scalar(black_box(&src), src_w, src_h, dst_w, dst_h);
            black_box(dst);
        })
    });

    group.bench_function("1080p_to_720p_avx2", |b| {
        b.iter(|| {
            let dst = bilinear_scale_plane_u16(black_box(&src), src_w, src_h, dst_w, dst_h);
            black_box(dst);
        })
    });

    group.finish();
}

criterion_group!(benches, bench_bilinear, bench_bilinear_10bit_avx2_vs_scalar);
criterion_main!(benches);
