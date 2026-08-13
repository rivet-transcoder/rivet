//! How many pictures does a decoder actually produce for a file?
//!
//! Written to reproduce a production failure where a 221-second upload
//! decoded to eleven frames: every rendition came out ~0.4s long while the
//! audio track ran to full length. The worker reported 5,533 samples pushed
//! and 11 frames seen, so the demuxer was fine and the decoder was not — but
//! the only fixtures in the suite are short enough that eleven frames *is*
//! the whole clip, which is exactly why it survived.
//!
//! Usage: `decode_count <file> [expected_fps]`

use std::{env, fs};

fn main() -> anyhow::Result<()> {
    let path = env::args().nth(1).expect("usage: decode_count <file> [fps]");
    let data = fs::read(&path)?;
    println!("file: {} ({} bytes)", path, data.len());

    let mut demuxer = container::streaming::demux_streaming(&data)?;
    let codec_name = demuxer.header().codec.clone();
    let info = demuxer.header().info.clone();

    println!("codec: {codec_name}  {}x{}", info.width, info.height);

    let mut decoder = codec::decode::create_decoder(&codec_name, info)?;

    let mut samples = 0u64;
    let mut frames = 0u64;
    let mut first_sample_bytes = 0usize;

    while let Some(sample) = demuxer.next_video_sample()? {
        samples += 1;
        if samples == 1 {
            first_sample_bytes = sample.data.len();
            // The first bytes say which framing the decoder is being handed:
            // `00 00 00 01` is Annex-B, anything else is probably a length.
            let head: Vec<String> =
                sample.data.iter().take(8).map(|b| format!("{b:02x}")).collect();
            println!("first sample: {} bytes, head {}", first_sample_bytes, head.join(" "));
        }

        decoder.push_sample(&sample.data)?;
        while decoder.decode_next()?.is_some() {
            frames += 1;
        }
    }

    decoder.finish()?;
    while decoder.decode_next()?.is_some() {
        frames += 1;
    }

    println!("samples={samples} frames={frames}");

    if samples > 0 {
        let ratio = frames as f64 / samples as f64;
        println!("frames/sample = {ratio:.4}");
        // One picture per access unit is the expectation for the progressive
        // material this pipeline accepts.
        if ratio < 0.9 {
            println!("FAIL: the decoder dropped {} of them", samples - frames);
            std::process::exit(1);
        }
    }

    println!("OK");
    Ok(())
}
