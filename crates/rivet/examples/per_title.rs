//! Ask a clip how many bits it needs.
//!
//! Runs a per-title pass over one file: samples a segment's worth of frames
//! from the middle, encodes it at every candidate delta across the GPU pool,
//! and prints the curve and the shift the floor selects.
//!
//! ```sh
//! cargo run --release --features qsv --example per_title -- input.mp4 [floor=0.9985] [WxH]
//! ```
//!
//! `WxH` is the top rung the sample is measured at (default: the source
//! size); a per-title pass should measure the rung a viewer sees, not the
//! encoder — see `codec::bench`.

use std::sync::Arc;

use anyhow::{Context, Result};
use bytes::Bytes;

use rivet::per_title::{DEFAULT_CANDIDATES, SampleSpec, Selection, sample_frames, select_shift, sweep_on_pool};

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt().with_env_filter("info").init();

    let args: Vec<String> = std::env::args().collect();
    let Some(input_path) = args.get(1) else {
        eprintln!("usage: per_title <input> [floor=0.9985] [WxH]");
        std::process::exit(2);
    };
    let floor: f64 = args.get(2).map(|s| s.parse()).transpose().context("floor")?.unwrap_or(0.9985);

    let input = Bytes::from(std::fs::read(input_path).context("reading input")?);
    let header = rivet::container::streaming::demux_streaming(&input)?.header().clone();
    let (width, height) = match args.get(3) {
        Some(dims) => {
            let (w, h) = dims.split_once('x').context("WxH")?;
            (w.parse()?, h.parse()?)
        }
        None => header.upright_dims(),
    };

    let frames = sample_frames(&input, &header, &SampleSpec::default())?;
    println!("sampled {} frames from {}x{} {}", frames.len(), header.info.width, header.info.height, header.codec);

    let base = rivet::codec::encode::EncoderConfig {
        width,
        height,
        frame_rate: header.info.frame_rate,
        keyframe_interval: (header.info.frame_rate * 4.0).round() as u32,
        pixel_format: header.info.pixel_format,
        color_metadata: header.info.color_metadata,
        target: rivet::codec::encode::tuning::QualityTarget::High,
        tier: rivet::codec::encode::tuning::SpeedTier::Archive,
        ..Default::default()
    };
    let gpu_pool = Arc::new(rivet::detect_gpu_pool());
    let deltas = DEFAULT_CANDIDATES.to_vec();

    let sweep = sweep_on_pool(&base, frames, &deltas, &gpu_pool, |done, total| {
        println!("  candidate {done}/{total} landed");
    })
    .await?;

    println!("{:>6} {:>10} {:>9} {:>8} {:>7}", "delta", "bytes", "ssim", "dB", "psnr");
    for s in &sweep.samples {
        println!(
            "{:>6} {:>10} {:>9.6} {:>8.2} {:>7.2}",
            s.quality_delta, s.trimmed_bytes, s.ssim, s.ssim_db(), s.psnr
        );
    }
    match select_shift(&sweep, floor, &deltas) {
        Selection::Chosen { sample, capped } => println!(
            "floor {floor}: shift the ladder by {:+} (ssim {:.6}){}",
            sample.quality_delta,
            sample.ssim,
            if capped { " — capped by the candidate range; widen it to find where the content stops" } else { "" }
        ),
        Selection::KeptBase => println!("floor {floor}: nothing reached it; the clip keeps its base quality"),
    }
    Ok(())
}
