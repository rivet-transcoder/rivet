//! Criterion bench for the BT.601 → BT.709 matrix conversion —
//! compares the scalar reference against the AVX2 specialization on
//! a 1920×1080 YCbCr 4:2:0 frame. This bench is the one the perf
//! memory rule is evaluated against: AVX2 must show ≥1.5× gain over
//! scalar on the hotpath or the specialization is reverted.

use codec::colorspace::{
    bt601_to_bt709_planes, bt601_to_bt709_planes_10bit, bt601_to_bt709_planes_10bit_scalar,
    bt601_to_bt709_planes_scalar,
};
use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};

fn make_planes(w: usize, h: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut y = vec![0u8; w * h];
    let mut cb = vec![0u8; (w / 2) * (h / 2)];
    let mut cr = vec![0u8; (w / 2) * (h / 2)];
    for i in 0..y.len() {
        y[i] = 16 + ((i as u32 * 17) % 220) as u8;
    }
    for i in 0..cb.len() {
        cb[i] = 16 + ((i as u32 * 13) % 225) as u8;
        cr[i] = 16 + ((i as u32 * 23) % 225) as u8;
    }
    (y, cb, cr)
}

fn make_planes_10bit(w: usize, h: usize) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
    // Limited 10-bit: luma [64, 940], chroma [64, 960]. Sweep both.
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

fn bench_601_to_709(c: &mut Criterion) {
    let w = 1920usize;
    let h = 1080usize;
    let (y0, cb0, cr0) = make_planes(w, h);

    let mut group = c.benchmark_group("colorspace_601_to_709");
    // Throughput in px per iter = luma pixel count. Criterion will
    // print per-pixel latency for easy scalar/avx2 comparison.
    group.throughput(Throughput::Elements((w * h) as u64));

    group.bench_function("1080p_scalar", |b| {
        b.iter_batched(
            || (y0.clone(), cb0.clone(), cr0.clone()),
            |(mut y, mut cb, mut cr)| {
                bt601_to_bt709_planes_scalar(
                    black_box(&mut y),
                    black_box(&mut cb),
                    black_box(&mut cr),
                    w,
                    h,
                );
                (y, cb, cr)
            },
            criterion::BatchSize::LargeInput,
        )
    });

    group.bench_function("1080p_dispatch", |b| {
        b.iter_batched(
            || (y0.clone(), cb0.clone(), cr0.clone()),
            |(mut y, mut cb, mut cr)| {
                bt601_to_bt709_planes(
                    black_box(&mut y),
                    black_box(&mut cb),
                    black_box(&mut cr),
                    w,
                    h,
                );
                (y, cb, cr)
            },
            criterion::BatchSize::LargeInput,
        )
    });

    group.finish();
}

/// Squad-29: 10-bit BT.601→BT.709 scalar vs AVX2 dispatch.
/// 1920×1080 4:2:0 frame; same shape as the 8-bit bench. Target
/// speedup ≥3× (realistic floor for u16 lanes vs u8 — Squad-4's
/// 8-bit kernel hit 18.78× because it processed 32 × u8 per iter
/// vs 16 × u16 here).
fn bench_601_to_709_10bit_avx2_vs_scalar(c: &mut Criterion) {
    let w = 1920usize;
    let h = 1080usize;
    let (y0, cb0, cr0) = make_planes_10bit(w, h);

    let mut group = c.benchmark_group("colorspace_601_to_709_10bit_avx2_vs_scalar");
    group.throughput(Throughput::Elements((w * h) as u64));

    group.bench_function("1080p_scalar", |b| {
        b.iter_batched(
            || (y0.clone(), cb0.clone(), cr0.clone()),
            |(mut y, mut cb, mut cr)| {
                bt601_to_bt709_planes_10bit_scalar(
                    black_box(&mut y),
                    black_box(&mut cb),
                    black_box(&mut cr),
                    w,
                    h,
                );
                (y, cb, cr)
            },
            criterion::BatchSize::LargeInput,
        )
    });

    group.bench_function("1080p_avx2", |b| {
        b.iter_batched(
            || (y0.clone(), cb0.clone(), cr0.clone()),
            |(mut y, mut cb, mut cr)| {
                bt601_to_bt709_planes_10bit(
                    black_box(&mut y),
                    black_box(&mut cb),
                    black_box(&mut cr),
                    w,
                    h,
                );
                (y, cb, cr)
            },
            criterion::BatchSize::LargeInput,
        )
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_601_to_709,
    bench_601_to_709_10bit_avx2_vs_scalar
);
criterion_main!(benches);
