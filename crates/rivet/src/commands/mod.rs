//! Subcommand implementations for the `rivet` CLI.

pub mod capabilities;
pub mod devices;
pub mod pipe;
pub mod probe;
pub mod progress;
pub mod splice;
pub mod transcode;

#[cfg(feature = "batch")]
pub mod batch;
#[cfg(feature = "ipc")]
pub mod ipc;
#[cfg(feature = "server")]
pub mod serve;

use std::sync::Arc;

use anyhow::{bail, Context, Result};
use rivet::progress::RungProgress;
use rivet::{RungArtifact, TranscodeSettings};

use crate::{ChromaArg, ColorArg, PixelArg, value_name};

/// How each rung's `WxH` box is filled — `--fit`, `--orientation`,
/// `--upscale` — for the commands that take a size (`transcode`, `pipe`).
/// Placed in the settings under the keys every surface uses.
#[derive(clap::Args, Debug, Default, Clone)]
pub(crate) struct FitArgs {
    /// How the source meets each rung's box, which is a maximum, not the
    /// output size: `contain` (default: inside the box, keeping the source's
    /// shape), `cover` (fill the box, centre-cropping the overflow), `pad`
    /// (contain, then black bars to exactly the box) or `stretch` (exactly
    /// the box, distorting the picture). A rung's own `:FIT` wins
    /// (`--rung 1080x1920:cover:fixed`).
    #[arg(long, value_name = "FIT")]
    pub fit: Option<String>,
    /// `auto` (default): a box turns to the source's orientation, so a
    /// 1920x1080 rung on a portrait source is 1080x1920. `fixed`: boxes are
    /// used as written.
    #[arg(long, value_name = "auto|fixed")]
    pub orientation: Option<String>,
    /// Let a rung be larger than the source. Off by default: a source smaller
    /// than a box comes out at its own size, and rungs that collapse onto the
    /// same size are merged.
    #[arg(long)]
    pub upscale: bool,
}

impl FitArgs {
    pub(crate) fn apply(&self, settings: &mut TranscodeSettings) -> Result<()> {
        if let Some(f) = &self.fit {
            settings.apply_kv("fit", f).context("parsing --fit")?;
        }
        if let Some(o) = &self.orientation {
            settings.apply_kv("orientation", o).context("parsing --orientation")?;
        }
        settings.upscale = self.upscale;
        Ok(())
    }
}

/// The output-shaping flags `rivet transcode` and `rivet splice` share, as
/// clap parsed them: what the output looks like (colour, depth, chroma
/// filter, video filters, quality, GOP) and how transcoded audio is made.
/// [`OutputShaping::apply`] places them in [`TranscodeSettings`] the way every
/// surface does — typed values directly, worded ones through `apply_kv` under
/// the keys the IPC socket, the HTTP API and the batch manifest use — so both
/// commands reach the same spec builder and the same validation.
#[derive(clap::Args)]
pub(crate) struct OutputShaping {
    /// Perceptual quality target: `visually_lossless`, `high`, `standard`
    /// (default), `low`, or `vmaf=N` — see `rivet transcode --help`.
    #[arg(long, value_parser = rivet::settings::parse_quality_target)]
    pub target: Option<rivet::codec::encode::tuning::QualityTarget>,
    /// GOP length in frames (default: two seconds at the output frame rate).
    #[arg(long, visible_alias = "keyframe-interval")]
    pub gop: Option<u32>,
    /// Video bitrate, e.g. `3M`: code every rung without its own
    /// (`--rung WxH@RATE`) to a rate rather than to `--target`. An average
    /// rate (the default `--rate-mode`) is coded by the native software
    /// H.264 / H.265 encoder, and a job whose encode pool is GPUs is refused
    /// before a frame is decoded; a constant one (`--rate-mode cbr`) is coded
    /// by the GPU encoders and the software H.264 / H.265 encoder.
    #[arg(long = "video-bitrate", value_name = "BPS")]
    pub video_bitrate: Option<String>,
    /// Coded picture buffer for every bitrate rung, e.g. `500ms` (`0` for
    /// none; one second when not given): the stream declares it and keeps to
    /// it, which is what bounds its peaks (and an HLS rendition's BANDWIDTH).
    #[arg(long = "video-buffer", value_name = "DURATION")]
    pub video_buffer: Option<String>,
    /// Rate mode for every bitrate rung: `average` (default; `abr`) or `cbr`
    /// (`constant`) — a constant rate, the rate also the maximum within the
    /// declared buffer (`--video-buffer`, one second by default), coded by
    /// the GPU encoders (QSV, NVENC, AMF; AV1 included) and the software
    /// H.264 / H.265 encoder (not rav1e). A `cbr` rung with
    /// no rate of its own takes `--video-bitrate`, else a default by codec,
    /// size and frame rate (H.264 1080p30 5 Mb/s, 720p 3M, 480p 1.2M, 360p
    /// 0.8M, 2160p 16M; H.265 0.65x, AV1 0.5x; more above 30 fps).
    #[arg(long = "rate-mode", value_name = "MODE")]
    pub rate_mode: Option<String>,
    /// Target bitrate for transcoded audio, e.g. `240k` (Opus; MP3 takes
    /// 32k..320k on the MPEG-1 ladder). Ignored for passthrough tracks.
    #[arg(long = "audio-bitrate", value_name = "BPS")]
    pub audio_bitrate: Option<String>,
    /// Output channel layout: `source` (default), `mono`, `stereo`, `5.1`,
    /// `7.1` — see `rivet transcode --help`.
    #[arg(long = "audio-channels", value_name = "LAYOUT")]
    pub audio_channels: Option<String>,
    /// Audio filter chain applied before the Opus encoder, e.g.
    /// `channelmap=FL-FL|FR-FR:stereo` — see `rivet transcode --help`.
    #[arg(long = "audio-filter", value_name = "CHAIN")]
    pub audio_filter: Option<String>,
    /// Output color / tonemap policy. The output follows the first clip:
    /// `passthrough` keeps its colour and depth, and later clips are
    /// mapped into it.
    #[arg(long, value_enum, default_value = "sdr")]
    pub color: ColorArg,
    /// 4:4:4 → 4:2:0 chroma filter for 4:4:4 clips (`box` default).
    #[arg(long = "chroma-downsample", value_enum, default_value = "box")]
    pub chroma_downsample: ChromaArg,
    /// Output luma bit depth: `auto` follows the color policy and the first
    /// clip; `8bit` encodes a 10-bit clip at 8 bits.
    #[arg(long, value_enum, default_value = "auto")]
    pub pixel_format: PixelArg,
    /// Video filter chain applied to every clip before scaling, e.g.
    /// `crop=1280:720,hflip` — see `rivet transcode --help`.
    #[arg(long)]
    pub filter: Option<String>,
}

impl OutputShaping {
    pub(crate) fn apply(&self, settings: &mut TranscodeSettings) -> Result<()> {
        settings.filters = match self.filter.as_deref() {
            Some(s) => codec::filter::parse_chain(s).context("parsing --filter")?,
            None => Vec::new(),
        };
        settings.audio_filters = match self.audio_filter.as_deref() {
            Some(s) => codec::audio::filter::parse_chain(s).context("parsing --audio-filter")?,
            None => Vec::new(),
        };
        settings.audio_bitrate = self
            .audio_bitrate
            .as_deref()
            .map(rivet::settings::parse_bitrate)
            .transpose()
            .context("parsing --audio-bitrate")?;
        if let Some(c) = &self.audio_channels {
            settings.apply_kv("audio-channels", c).context("parsing --audio-channels")?;
        }
        settings.target = self.target;
        settings.gop = self.gop;
        if let Some(v) = &self.video_bitrate {
            settings.apply_kv("video-bitrate", v).context("parsing --video-bitrate")?;
        }
        if let Some(v) = &self.video_buffer {
            settings.apply_kv("video-buffer", v).context("parsing --video-buffer")?;
        }
        if let Some(v) = &self.rate_mode {
            settings.apply_kv("rate-mode", v).context("parsing --rate-mode")?;
        }
        settings.apply_kv("color", &value_name(self.color))?;
        settings.apply_kv("chroma-downsample", &value_name(self.chroma_downsample))?;
        settings.apply_kv("bit-depth", &value_name(self.pixel_format))?;
        Ok(())
    }
}

/// Convert a [`rivet::progress::RungStatus`] to a short display label.
pub(crate) fn status_str(s: rivet::progress::RungStatus) -> &'static str {
    match s {
        rivet::progress::RungStatus::Pending => "pend",
        rivet::progress::RungStatus::Running => "run",
        rivet::progress::RungStatus::Finalizing => "final",
        rivet::progress::RungStatus::Completed => "done",
        rivet::progress::RungStatus::Failed => "FAIL",
    }
}

/// JSON-escape a bare string value (no surrounding quotes).
pub(crate) fn esc(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

/// Transcode `input` honouring `settings`; returns `(mp4_bytes, frame_count, audio_label)`.
///
/// All-default settings take the fast [`rivet::transcode_bytes`] path; any set field
/// routes through [`rivet::TranscodeSettings::into_spec`] + the full `run_job` engine.
pub(crate) fn stream_transcode(
    input: &[u8],
    settings: &TranscodeSettings,
) -> Result<(Vec<u8>, u64, String)> {
    if settings.is_empty() {
        let out = rivet::transcode_bytes(input).context("transcoding")?;
        return Ok((
            out.output_bytes,
            out.frames_processed,
            out.audio_handling.label(),
        ));
    }
    let probed = rivet::probe_bytes(input).context("probing input")?;
    let spec = settings
        .clone()
        .into_spec_for(&probed)
        .context("invalid settings")?;
    if matches!(spec.mode, rivet::OutputMode::Hls { .. }) {
        bail!(
            "HLS/segmented output isn't supported over pipe/ipc (a single stream) — \
             use `rivet transcode -o <dir>` or the HTTP API"
        );
    }
    let sink = Arc::new(rivet::fn_sink(|_p: RungProgress| {}));
    let out = rivet::run_job_blocking(input, &spec, None, sink).context("transcoding")?;
    let audio = out.audio_handling.clone();
    for r in out.rungs {
        let frames = r.frames;
        if let RungArtifact::File(bytes) = r.artifact {
            return Ok((bytes, frames, audio));
        }
    }
    bail!("no single-file output produced")
}
