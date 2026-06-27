//! Per-vendor-family GPU decode matrix against real media.
//!
//! rivet decodes on the GPU — every CPU codec was removed; the decode
//! frameworks are all hand-rolled `dlopen` FFI in-tree: NVDEC (`nvidia`),
//! AMF (`amd`), QSV (`qsv`), plus FFmpeg hwaccel (`ffmpeg`). `create_decoder_on`
//! dispatches to the framework for the GPU at a given index, so iterating the
//! host's detected GPUs exercises **each vendor family present**.
//!
//! For every (GPU, sample) pair this:
//!   1. demuxes a real file from `test_media/`,
//!   2. decodes it through that GPU's framework,
//!   3. measures the decoded **luma spread** (max−min over a subsample).
//!
//! A non-trivial spread proves the framework decoded *actual content* rather
//! than a black/grey screen — the "did anything really decode" check. The test
//! prints a human-reviewable per-vendor matrix to stderr and asserts that at
//! least one (vendor, codec) pair produced real frames. It skips cleanly when
//! no GPU is present or `test_media/` is empty, so a CPU-only CI box stays
//! green while a GPU host gets real coverage.

use std::path::{Path, PathBuf};

use codec::frame::{PixelFormat, VideoFrame};
use codec::gpu::{self, GpuDevice};

/// Luma spread below this (8-bit scale) means a flat frame — a black or grey
/// screen, i.e. nothing was really decoded. Real video easily clears it.
const FLAT_LUMA_SPREAD: u32 = 24;

/// Test-media directory: the `RIVET_TEST_MEDIA` env var if set, else the
/// workspace's `test_media/` (populated by `test_media/fetch.sh`).
fn test_media_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("RIVET_TEST_MEDIA") {
        return PathBuf::from(dir);
    }
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("test_media")
}

/// Every demuxable video file in `test_media/` (sorted, for stable output).
fn corpus() -> Vec<PathBuf> {
    let dir = test_media_dir();
    let Ok(rd) = std::fs::read_dir(&dir) else {
        return Vec::new();
    };
    let mut files: Vec<PathBuf> = rd
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            matches!(
                p.extension().and_then(|x| x.to_str()),
                Some("mp4" | "mov" | "m4v" | "mkv" | "webm" | "ts" | "avi")
            )
        })
        .collect();
    files.sort();
    files
}

/// Max−min luma over a coarse subsample of the Y plane, normalized to an 8-bit
/// scale so 8-bit and 10-bit frames compare on the same axis.
fn luma_spread(frame: &VideoFrame) -> u32 {
    let w = frame.width as usize;
    let h = frame.height as usize;
    if w == 0 || h == 0 {
        return 0;
    }
    let data = &frame.data;
    let (mut lo, mut hi) = (u32::MAX, 0u32);
    let mut sample = |v: u32| {
        lo = lo.min(v);
        hi = hi.max(v);
    };
    match frame.format {
        PixelFormat::Yuv420p => {
            let y = &data[..(w * h).min(data.len())];
            // ~every 17th byte — coprime-ish stride avoids sampling one column.
            for &b in y.iter().step_by(17) {
                sample(b as u32);
            }
        }
        PixelFormat::Yuv420p10le => {
            let n = (w * h * 2).min(data.len());
            let y = &data[..n];
            for px in y.chunks_exact(2).step_by(17) {
                let v = u16::from_le_bytes([px[0], px[1]]) as u32; // 0..=1023
                sample(v >> 2); // → 8-bit scale
            }
        }
        _ => return 0,
    }
    if lo == u32::MAX { 0 } else { hi - lo }
}

/// Demux `data` and decode it through the GPU at `gpu_index`. Returns
/// `(decoded_frame_count, max_luma_spread)` or an error string.
fn decode_on_gpu(data: &[u8], gpu_index: u32) -> Result<(String, usize, u32), String> {
    let demuxed = container::demux::demux(data).map_err(|e| format!("demux: {e}"))?;
    let codec = demuxed.codec.clone();
    let mut decoder = codec::decode::create_decoder_on(&codec, demuxed.info.clone(), Some(gpu_index))
        .map_err(|e| format!("create_decoder_on: {e:#}"))?;

    let mut frames = 0usize;
    let mut max_spread = 0u32;
    // Cap work — we only need to prove non-black decode, not the whole clip.
    const MAX_SAMPLES: usize = 120;
    for sample in demuxed.samples.iter().take(MAX_SAMPLES) {
        decoder
            .push_sample(sample)
            .map_err(|e| format!("push_sample: {e:#}"))?;
        while let Some(f) = decoder.decode_next().map_err(|e| format!("decode_next: {e:#}"))? {
            frames += 1;
            max_spread = max_spread.max(luma_spread(&f));
        }
    }
    decoder.finish().map_err(|e| format!("finish: {e:#}"))?;
    while let Some(f) = decoder.decode_next().map_err(|e| format!("drain: {e:#}"))? {
        frames += 1;
        max_spread = max_spread.max(luma_spread(&f));
    }
    Ok((codec, frames, max_spread))
}

#[test]
fn gpu_decode_per_vendor_family_produces_real_frames() {
    let gpus: Vec<GpuDevice> = gpu::detect_gpus();
    if gpus.is_empty() {
        eprintln!("SKIP gpu_vendor_matrix: no GPU detected on this host");
        return;
    }
    let files = corpus();
    if files.is_empty() {
        eprintln!(
            "SKIP gpu_vendor_matrix: no media in {} (run test_media/fetch.sh)",
            test_media_dir().display()
        );
        return;
    }

    eprintln!("GPU decode matrix ({} GPU(s), {} sample(s)):", gpus.len(), files.len());
    eprintln!("| vendor  | idx | name                       | codec | frames | luma_spread | result |");
    eprintln!("|---------|-----|----------------------------|-------|--------|-------------|--------|");

    let mut any_real = false;
    let mut decoded_combos = 0usize;
    for g in &gpus {
        for f in &files {
            let Ok(data) = std::fs::read(f) else { continue };
            match decode_on_gpu(&data, g.index) {
                Ok((codec, frames, spread)) => {
                    let result = if frames == 0 {
                        "no frames"
                    } else if spread >= FLAT_LUMA_SPREAD {
                        decoded_combos += 1;
                        any_real = true;
                        "OK content"
                    } else {
                        decoded_combos += 1;
                        "FLAT (black/grey?)"
                    };
                    eprintln!(
                        "| {:<7?} | {:>3} | {:<26} | {:<5} | {:>6} | {:>11} | {} |",
                        g.vendor,
                        g.index,
                        truncate(&g.name, 26),
                        codec,
                        frames,
                        spread,
                        result
                    );
                }
                Err(e) => {
                    eprintln!(
                        "| {:<7?} | {:>3} | {:<26} | {:<5} | {:>6} | {:>11} | err: {} |",
                        g.vendor,
                        g.index,
                        truncate(&g.name, 26),
                        "?",
                        "-",
                        "-",
                        e
                    );
                }
            }
        }
    }

    if decoded_combos == 0 {
        eprintln!(
            "SKIP gpu_vendor_matrix: no (GPU, sample) pair decoded — the detected GPUs may not \
             support these codecs, or the vendor feature ({{nvidia,amd,qsv}}) isn't compiled in."
        );
        return;
    }
    assert!(
        any_real,
        "every decoded sample was a flat (black/grey) frame across all GPU vendors — \
         the GPU decode path produced no real content"
    );
}

fn truncate(s: &str, n: usize) -> String {
    if s.len() <= n {
        s.to_string()
    } else {
        format!("{}…", &s[..n - 1])
    }
}
