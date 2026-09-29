//! Implementation of `rivet pipe` (stdin → stdout streaming transcode).

use anyhow::{bail, Context, Result};
use rivet::TranscodeSettings;

use crate::{AudioArg, ColorArg, PixelArg, value_name};

/// Raw CLI arguments for the `pipe` subcommand (one-to-one with the flags).
pub(crate) struct PipeArgs {
    pub crf: Option<u8>,
    pub target: Option<rivet::codec::encode::tuning::QualityTarget>,
    pub gop: Option<String>,
    pub video_bitrate: Option<String>,
    pub video_buffer: Option<String>,
    pub rate_mode: Option<String>,
    pub audio: Option<AudioArg>,
    pub audio_bitrate: Option<String>,
    pub audio_channels: Option<String>,
    pub audio_filter: Option<String>,
    pub color: Option<ColorArg>,
    pub chroma_downsample: Option<crate::ChromaArg>,
    pub bit_depth: Option<PixelArg>,
    pub max_fps: Option<String>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub gpu: Option<u32>,
    pub decode: rivet::DecodePolicy,
    pub encode: Option<rivet::EncodePolicy>,
    pub filter: Option<String>,
    pub fitting: super::FitArgs,
}

pub(crate) fn run(args: PipeArgs) -> Result<()> {
    use std::io::{Read, Write};

    let mut settings = TranscodeSettings {
        crf: args.crf,
        target: args.target,
        audio_filters: match args.audio_filter {
            Some(s) => codec::audio::filter::parse_chain(&s).context("parsing --audio-filter")?,
            None => Vec::new(),
        },
        width: args.width,
        height: args.height,
        gpu: args.gpu,
        decode_policy: args.decode,
        encode: args.encode,
        filters: match args.filter {
            Some(s) => codec::filter::parse_chain(&s).context("parsing --filter")?,
            None => Vec::new(),
        },
        ..Default::default()
    };
    // Worded values go through the settings vocabulary, like every surface.
    for (key, value) in [("gop", &args.gop), ("audio-bitrate", &args.audio_bitrate), ("max-fps", &args.max_fps)] {
        if let Some(v) = value {
            settings.apply_kv(key, v).with_context(|| format!("parsing --{key}"))?;
        }
    }
    args.fitting.apply(&mut settings)?;
    if let Some(a) = args.audio {
        settings.apply_kv("audio", &value_name(a))?;
    }
    if let Some(c) = &args.audio_channels {
        settings.apply_kv("audio-channels", c).context("parsing --audio-channels")?;
    }
    if let Some(c) = args.color {
        settings.apply_kv("color", &value_name(c))?;
    }
    if let Some(f) = args.chroma_downsample {
        settings.apply_kv("chroma-downsample", &value_name(f))?;
    }
    if let Some(b) = args.bit_depth {
        settings.apply_kv("bit-depth", &value_name(b))?;
    }
    if let Some(v) = &args.video_bitrate {
        settings.apply_kv("video-bitrate", v).context("parsing --video-bitrate")?;
    }
    if let Some(v) = &args.video_buffer {
        settings.apply_kv("video-buffer", v).context("parsing --video-buffer")?;
    }
    if let Some(v) = &args.rate_mode {
        settings.apply_kv("rate-mode", v).context("parsing --rate-mode")?;
    }

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
