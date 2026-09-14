//! Decode one file through rivet's decoder dispatch and write the raw planar
//! frames, so the picture a decoder hands back can be compared byte for byte
//! with a reference decode of the same file:
//!
//! ```text
//! cargo run -p rivet-codec --features nvidia --example nvdec_dump -- in.mp4 out.yuv
//! ffmpeg -i in.mp4 -f rawvideo -pix_fmt yuv420p ref.yuv && cmp out.yuv ref.yuv
//! ```
//!
//! On an NVIDIA host the dispatch engages NVDEC (`DISABLE_NVDEC=1` routes the
//! same run through the software tier for a control). Prints the frame count
//! and every distinct (width, height, pixel format) the decoder produced —
//! a padded coded surface shows up here as a second shape, or as the wrong
//! one.

use std::io::Write;

use codec::frame::{PixelFormat, VideoFrame};

fn main() -> anyhow::Result<()> {
    let _ = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_writer(std::io::stderr)
        .try_init();
    let mut args = std::env::args().skip(1);
    let (Some(input), Some(output)) = (args.next(), args.next()) else {
        anyhow::bail!("usage: nvdec_dump <in.mp4|mkv|webm> <out.yuv>");
    };
    let data = std::fs::read(&input)?;
    let demuxed = container::demux::demux(&data)?;
    eprintln!(
        "demux: codec={} {}x{} {} samples",
        demuxed.codec,
        demuxed.info.width,
        demuxed.info.height,
        demuxed.samples.len()
    );

    let mut decoder = codec::decode::create_decoder(&demuxed.codec, demuxed.info.clone())?;
    let mut out = std::io::BufWriter::new(std::fs::File::create(&output)?);
    let mut frames = 0usize;
    let mut shapes: Vec<(u32, u32, PixelFormat)> = Vec::new();
    let mut emit = |f: VideoFrame, out: &mut std::io::BufWriter<std::fs::File>| -> anyhow::Result<()> {
        let shape = (f.width, f.height, f.format);
        if !shapes.contains(&shape) {
            shapes.push(shape);
        }
        out.write_all(&f.data)?;
        frames += 1;
        Ok(())
    };
    for sample in &demuxed.samples {
        decoder.push_sample(sample)?;
        while let Some(f) = decoder.decode_next()? {
            emit(f, &mut out)?;
        }
    }
    decoder.finish()?;
    while let Some(f) = decoder.decode_next()? {
        emit(f, &mut out)?;
    }
    out.flush()?;
    println!("frames={frames} shapes={shapes:?}");
    Ok(())
}
