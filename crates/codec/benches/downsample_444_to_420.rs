//! Microbench for the 4:4:4 → 4:2:0 chroma downsample (Squad-31,
//! roadmap #6). Mostly here so the next squad sizing the AVX2
//! follow-up has a baseline to compare against. 1080p case on the
//! dev box is the headline number quoted in the squad summary.

use codec::colorspace::{downsample_chroma_444_to_420, downsample_chroma_444_to_420_10bit};
use criterion::{Criterion, black_box, criterion_group, criterion_main};

fn bench_8bit_1080p(c: &mut Criterion) {
    let w = 1920usize;
    let h = 1080usize;
    let plane = w * h;
    let y = vec![128u8; plane];
    let cb = vec![128u8; plane];
    let cr = vec![128u8; plane];
    c.bench_function("downsample_444_to_420_8bit_1080p", |b| {
        b.iter(|| {
            let out =
                downsample_chroma_444_to_420(black_box(&y), black_box(&cb), black_box(&cr), w, h);
            black_box(out);
        });
    });
}

fn bench_10bit_1080p(c: &mut Criterion) {
    let w = 1920usize;
    let h = 1080usize;
    let plane = w * h;
    let y = vec![512u16; plane];
    let cb = vec![512u16; plane];
    let cr = vec![512u16; plane];
    c.bench_function("downsample_444_to_420_10bit_1080p", |b| {
        b.iter(|| {
            let out = downsample_chroma_444_to_420_10bit(
                black_box(&y),
                black_box(&cb),
                black_box(&cr),
                w,
                h,
            );
            black_box(out);
        });
    });
}

criterion_group!(benches, bench_8bit_1080p, bench_10bit_1080p);
criterion_main!(benches);
