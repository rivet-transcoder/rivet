//! NVDEC end-to-end decode throughput on real test_media/ samples.
//!
//! Measures `decode::create_decoder` + `decode_next()` in a tight loop
//! until EOS, so each sample covers demux-output feed-in through the
//! synchronous cuMemcpy2D chain documented at H5 in
//! `perf/hotpath-analysis.md`. Output is seconds per full-file decode;
//! divide by `demuxed.info.total_frames` for fps.
//!
//! This bench is GPU-only — it silently skips samples when no NVIDIA
//! GPU is present (`codec::gpu::has_nvidia() == false`). Runs on the
//! RTX 3090 dev box; numbers calibrate the published A10G reference
//! throughput used in the original hotpath estimates.

use std::path::PathBuf;

use codec::decode;
use codec::gpu;
use container::demux;
use criterion::{Criterion, black_box, criterion_group, criterion_main};

struct Sample {
    name: &'static str,
    path: PathBuf,
}

fn load_candidates() -> Vec<Sample> {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let media = manifest
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("test_media");

    // Cover each decode codec path with a short-ish clip so criterion's
    // 100-sample default doesn't run for an hour. `fetch.sh` populates
    // these; CI skips the ones that aren't present.
    let wanted = [
        (
            "h264_720p60",
            "bigbuck_bunny_8bit_750kbps_720p_60.0fps_h264.mp4",
        ),
        (
            "hevc_720p60",
            "bigbuck_bunny_8bit_750kbps_720p_60.0fps_hevc.mp4",
        ),
        (
            "vp9_720p60_mkv",
            "bigbuck_bunny_8bit_750kbps_720p_60.0fps_vp9.mkv",
        ),
        ("av1_1080p24", "jellyfin_av1_main_1080p_24fps.mp4"),
        (
            "h264_high_1080p24",
            "jellyfin_h264_high_l40_1080p_24fps.mp4",
        ),
    ];
    wanted
        .iter()
        .filter_map(|(label, f)| {
            let p = media.join(f);
            p.exists().then_some(Sample {
                name: label,
                path: p,
            })
        })
        .collect()
}

fn bench_nvdec_decode(c: &mut Criterion) {
    if !gpu::has_nvidia() {
        eprintln!("no NVIDIA GPU detected — NVDEC bench skipped");
        return;
    }

    let samples = load_candidates();
    if samples.is_empty() {
        eprintln!("no test_media samples present — NVDEC bench skipped");
        return;
    }

    let mut group = c.benchmark_group("nvdec_decode");
    // Each iteration decodes the entire file, which for a 10 s clip at
    // 60 fps = 600 frames and is non-trivial work. Hold the sample count
    // low to keep wallclock in check.
    group.sample_size(10);

    for sample in &samples {
        // Pre-read + pre-demux once so the bench body is purely the
        // decoder factory + decode loop. Criterion will iterate on that.
        let bytes = match std::fs::read(&sample.path) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("skip {}: read {}: {e}", sample.name, sample.path.display());
                continue;
            }
        };
        let demuxed = match demux::demux(&bytes) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skip {}: demux: {e}", sample.name);
                continue;
            }
        };
        let codec_name = demuxed.codec.clone();
        let info = demuxed.info.clone();
        let total_frames = demuxed.info.total_frames;
        let samples_vec = demuxed.samples;

        eprintln!(
            "{}: codec={} {}x{} frames={} duration={:.2}s",
            sample.name, codec_name, info.width, info.height, total_frames, info.duration
        );

        group.bench_function(sample.name, |b| {
            b.iter_custom(|iters| {
                let start = std::time::Instant::now();
                for _ in 0..iters {
                    // create_decoder tries NVDEC first for supported codecs,
                    // falling back to CPU if dlopen/init fails. On the 3090
                    // we expect NVDEC wins every time.
                    let mut dec = match decode::create_decoder(&codec_name, info.clone()) {
                        Ok(d) => d,
                        Err(e) => {
                            eprintln!("create_decoder failed: {e:#}");
                            return start.elapsed();
                        }
                    };
                    for s in &samples_vec {
                        if let Err(e) = dec.push_sample(s) {
                            eprintln!("push_sample failed: {e:#}");
                            return start.elapsed();
                        }
                    }
                    if let Err(e) = dec.finish() {
                        eprintln!("decoder finish failed: {e:#}");
                        return start.elapsed();
                    }
                    let mut n = 0u64;
                    loop {
                        match dec.decode_next() {
                            Ok(Some(frame)) => {
                                black_box(frame);
                                n += 1;
                            }
                            Ok(None) => break,
                            Err(e) => {
                                eprintln!("decode_next error after {n} frames: {e:#}");
                                break;
                            }
                        }
                    }
                    black_box(n);
                }
                start.elapsed()
            })
        });
    }

    group.finish();
}

criterion_group!(benches, bench_nvdec_decode);
criterion_main!(benches);
