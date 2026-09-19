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
    /// (`--rung WxH@RATE`) to a rate rather than to `--target`. The native
    /// software H.264 / H.265 encoder codes to a rate; a job whose encode pool
    /// is GPUs is refused before a frame is decoded.
    #[arg(long = "video-bitrate", value_name = "BPS")]
    pub video_bitrate: Option<String>,
    /// Coded picture buffer for every bitrate rung, e.g. `500ms` (`0` for
    /// none; one second when not given): the stream declares it and keeps to
    /// it, which is what bounds its peaks (and an HLS rendition's BANDWIDTH).
    #[arg(long = "video-buffer", value_name = "DURATION")]
    pub video_buffer: Option<String>,
    /// Target Opus bitrate for transcoded audio, e.g. `240k`. Ignored for
    /// passthrough tracks.
    #[arg(long = "audio-bitrate", value_name = "BPS")]
    pub audio_bitrate: Option<String>,
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
        settings.target = self.target;
        settings.gop = self.gop;
        if let Some(v) = &self.video_bitrate {
            settings.apply_kv("video-bitrate", v).context("parsing --video-bitrate")?;
        }
        if let Some(v) = &self.video_buffer {
            settings.apply_kv("video-buffer", v).context("parsing --video-buffer")?;
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
        .into_spec(probed.width, probed.height)
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
