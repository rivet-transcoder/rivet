//! One source through the entry points that decode it outside the job engine,
//! written out so their pictures can be compared across decoders and against
//! the ladder: the per-title sample (its frames, raw, to `<prefix>.sample.yuv`)
//! and, with the `thumbnail` feature, the poster (`<prefix>.thumb.avif`).
//! `transcode_bytes` is reached from the CLI: `rivet pipe` with no settings.
//!
//! ```sh
//! cargo run --example entry_points --features thumbnail -- input.mp4 out/prefix
//! ```
//!
//! Decoder choice follows the environment, as everywhere (`DISABLE_NVDEC=1`,
//! the build's features).

use anyhow::{Context, Result};
use bytes::Bytes;

fn main() -> Result<()> {
    tracing_subscriber::fmt().with_env_filter("info").init();
    let args: Vec<String> = std::env::args().collect();
    let (Some(input), Some(prefix)) = (args.get(1), args.get(2)) else {
        eprintln!("usage: entry_points <input> <out-prefix>");
        std::process::exit(2);
    };
    let data = Bytes::from(std::fs::read(input).context("reading input")?);
    let header = rivet::container::streaming::demux_streaming(&data)?
        .header()
        .clone();
    let (width, height) = header.upright_dims();
    let output = rivet::OutputSpec::single_file(vec![rivet::Rung::new(width, height)]);

    let spec = rivet::per_title::SampleSpec {
        frames: 8,
        ..Default::default()
    };
    // Each entry point is measured on its own: a sample that fails still
    // leaves the thumbnail to look at.
    match rivet::per_title::sample_frames(&data, &header, &spec, &output) {
        Ok(frames) if !frames.is_empty() => {
            let first = &frames[0];
            let raw: Vec<u8> = frames.iter().flat_map(|f| f.data.iter().copied()).collect();
            std::fs::write(format!("{prefix}.sample.yuv"), &raw).context("writing the sample")?;
            println!(
                "sample: {} frames, {:?} {:?} {}x{}",
                frames.len(),
                first.format,
                first.color_space,
                first.width,
                first.height
            );
        }
        Ok(_) => eprintln!("sample: empty"),
        Err(e) => eprintln!("sample: {e:#}"),
    }

    #[cfg(feature = "thumbnail")]
    {
        let thumb = rivet::thumbnail::generate_thumbnail(
            &data,
            0.5,
            rivet::thumbnail::DEFAULT_THUMBNAIL_QUALITY,
            rivet::thumbnail::DEFAULT_THUMBNAIL_SPEED,
        )?;
        std::fs::write(format!("{prefix}.thumb.avif"), &thumb.bytes)
            .context("writing the thumbnail")?;
        println!(
            "thumbnail: {}x{}, {} bytes",
            thumb.width,
            thumb.height,
            thumb.bytes.len()
        );
    }
    Ok(())
}
