//! Implementation of `rivet pipe` (stdin → stdout streaming transcode).

use anyhow::{bail, Context, Result};
use rivet::TranscodeSettings;

use crate::{AudioArg, ColorArg, PixelArg};

/// Raw CLI arguments for the `pipe` subcommand (one-to-one with the flags).
pub(crate) struct PipeArgs {
    pub crf: Option<u8>,
    pub audio: Option<AudioArg>,
    pub audio_bitrate: Option<String>,
    pub audio_filter: Option<String>,
    pub color: Option<ColorArg>,
    pub bit_depth: Option<PixelArg>,
    pub max_fps: Option<f64>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub gpu: Option<u32>,
    pub filter: Option<String>,
}

pub(crate) fn run(args: PipeArgs) -> Result<()> {
    use std::io::{Read, Write};

    let settings = TranscodeSettings {
        crf: args.crf,
        audio: args.audio.map(Into::into),
        audio_bitrate: args
            .audio_bitrate
            .as_deref()
            .map(rivet::settings::parse_bitrate)
            .transpose()
            .context("parsing --audio-bitrate")?,
        audio_filters: match args.audio_filter {
            Some(s) => codec::audio::filter::parse_chain(&s).context("parsing --audio-filter")?,
            None => Vec::new(),
        },
        color: args.color.map(Into::into),
        bit_depth: args.bit_depth.map(Into::into),
        max_fps: args.max_fps,
        width: args.width,
        height: args.height,
        gpu: args.gpu,
        filters: match args.filter {
            Some(s) => codec::filter::parse_chain(&s).context("parsing --filter")?,
            None => Vec::new(),
        },
        ..Default::default()
    };

    let mut input = Vec::new();
    std::io::stdin()
        .lock()
        .read_to_end(&mut input)
        .context("reading media from stdin")?;
    if input.is_empty() {
        bail!("empty stdin — pipe media in, e.g. `cat in.mkv | rivet pipe > out.mp4`");
    }
    eprintln!("rivet pipe: {} bytes in, transcoding…", input.len());
    let (bytes, frames, audio) = super::stream_transcode(&input, &settings)?;
    let mut stdout = std::io::stdout().lock();
    stdout.write_all(&bytes).context("writing AV1/MP4 to stdout")?;
    stdout.flush().ok();
    eprintln!("rivet pipe: {frames} frames → {} bytes out ({audio})", bytes.len());
    Ok(())
}
