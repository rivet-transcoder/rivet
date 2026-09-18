//! Implementation of `rivet splice`.

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use rivet::{RungArtifact, TranscodeSettings};

use crate::{AudioArg, ModeArg, value_name};

/// The `splice` subcommand's arguments, as clap parses them.
#[derive(clap::Args)]
pub(crate) struct SpliceArgs {
    /// Output: an MP4 file (`--mode single`) or a directory (`--mode hls`).
    #[arg(short, long)]
    pub output: PathBuf,
    /// Input clips in order: `PATH` or `PATH@START-END` (seconds).
    #[arg(required = true)]
    pub clips: Vec<String>,
    /// Output shape: `single` (one MP4) or `hls` (a CMAF/HLS package).
    #[arg(long, value_enum, default_value = "single")]
    pub mode: ModeArg,
    /// HLS target segment length (seconds); only used with `--mode hls`.
    #[arg(long, default_value_t = 4.0)]
    pub segment_seconds: f32,
    /// Output video codec: `av1` (default), `h264`, or `h265`.
    #[arg(long)]
    pub codec: Option<String>,
    /// Constant rate factor (quality; lower = better).
    #[arg(long)]
    pub crf: Option<u8>,
    /// Audio handling: `auto` (default), `opus`, `drop`.
    #[arg(long, value_enum, default_value = "auto")]
    pub audio: AudioArg,
    /// Subtitle tracks to carry: `all` (default), `none`, or a language
    /// list such as `eng,deu`. Each clip's cues are re-based onto the
    /// joined timeline and merged by language.
    #[arg(long, default_value = "all", value_name = "SELECTION")]
    pub subtitles: String,
    /// The decode plan: `auto` (default), `whole`, `fastest`, `gpu:N` or
    /// `ranges:N` — see `rivet transcode --help`. `--decode-gpu N` still
    /// works and means `gpu:N`.
    #[arg(long, visible_alias = "decode-gpu", default_value = "auto", value_parser = rivet::settings::parse_decode_plan)]
    pub decode: rivet::DecodePolicy,
    /// The encode plan: `all` (default), `per-rung`, `single`, `gpu:N` or
    /// `family:VENDOR` — see `rivet transcode --help`. A splice always takes
    /// the serial encode path, so here this chooses the card (`gpu:N`).
    #[arg(long, value_parser = rivet::settings::parse_encode_plan)]
    pub encode: Option<rivet::EncodePolicy>,
    /// `--color`, `--pixel-format` and the rest of the output-shaping flags
    /// transcode has, placed in the settings the way transcode places them.
    #[command(flatten)]
    pub shaping: super::OutputShaping,
}

impl SpliceArgs {
    /// The settings this splice runs with — everything but the output
    /// resolution, which comes from probing the first clip.
    pub(crate) fn settings(&self) -> Result<TranscodeSettings> {
        let video_codec = self
            .codec
            .as_deref()
            .map(rivet::settings::parse_video_codec)
            .transpose()
            .context("parsing --codec")?;
        let mut settings = TranscodeSettings {
            segment_seconds: Some(self.segment_seconds),
            crf: self.crf,
            video_codec,
            decode_policy: self.decode,
            encode: self.encode,
            ..Default::default()
        };
        self.shaping.apply(&mut settings)?;
        // Worded values go through the settings vocabulary, like every surface.
        settings.apply_kv("mode", &value_name(self.mode))?;
        settings.apply_kv("audio", &value_name(self.audio))?;
        settings.apply_kv("subtitles", &self.subtitles)?;
        Ok(settings)
    }
}

pub(crate) fn run(args: SpliceArgs) -> Result<()> {
    let parsed = args
        .clips
        .iter()
        .map(|s| parse_clip_spec(s))
        .collect::<Result<Vec<_>>>()?;
    let mut clip_bytes = Vec::with_capacity(parsed.len());
    for (path, _, _) in &parsed {
        clip_bytes
            .push(std::fs::read(path).with_context(|| format!("reading clip {}", path.display()))?);
    }
    // Probe the first clip to resolve the output resolution.
    let probed = rivet::probe_bytes(&clip_bytes[0]).context("probing first clip")?;
    let is_hls = matches!(args.mode, ModeArg::Hls);
    let output = args.output.clone();
    let spec = args
        .settings()?
        .into_spec(probed.width, probed.height)
        .context("building output spec")?;

    let splice_clips: Vec<rivet::Clip> = parsed
        .iter()
        .zip(clip_bytes)
        .map(|((_, start, end), bytes)| rivet::Clip::trimmed(bytes, *start, *end))
        .collect();

    let sink = Arc::new(super::progress::ProgressPrinter::new(spec.rungs.len()));

    // HLS writes a package into the output directory; single-file returns the
    // MP4 bytes in memory (one rung at source resolution). The directory is
    // made before the job runs, so an unusable path fails before any work; a
    // job that ends with nothing in it (refused by the encode pool's
    // preflight, say) takes back what this run made when `made_dir` drops, and
    // a directory that already existed is never removed.
    let made_dir = if is_hls {
        Some(
            rivet::output_dir::CreatedDir::create(&output)
                .with_context(|| format!("creating output dir {}", output.display()))?,
        )
    } else {
        None
    };
    let out_dir = is_hls.then(|| output.clone());
    let out = rivet::run_splice_job_blocking(splice_clips, &spec, out_dir.as_deref(), sink)
        .context("splicing clips")?;
    if let Some(made) = made_dir {
        made.keep();
    }

    if !is_hls
        && let Some(r) = out.rungs.first()
        && let RungArtifact::File(bytes) = &r.artifact
    {
        std::fs::write(&output, bytes).with_context(|| format!("writing {}", output.display()))?;
    }
    eprintln!(
        "  spliced {} clip(s) → {} ({:.2} MiB) in {:.2}s",
        parsed.len(),
        output.display(),
        out.rungs.iter().map(|r| r.bytes as f64).sum::<f64>() / (1024.0 * 1024.0),
        out.elapsed.as_secs_f64(),
    );
    Ok(())
}

/// Parse a splice clip spec: `PATH` or `PATH@START-END` (seconds, either side optional).
/// The `@` separator avoids the `:` in Windows drive paths.
fn parse_clip_spec(s: &str) -> Result<(PathBuf, Option<f64>, Option<f64>)> {
    match s.rfind('@') {
        Some(at) => {
            let path = &s[..at];
            let range = &s[at + 1..];
            let (start_s, end_s) = range
                .split_once('-')
                .with_context(|| format!("clip trim must be START-END, got '@{range}'"))?;
            let parse = |x: &str, what: &str| -> Result<Option<f64>> {
                if x.is_empty() {
                    Ok(None)
                } else {
                    Ok(Some(x.parse::<f64>().with_context(|| format!("bad {what} time '{x}'"))?))
                }
            };
            Ok((PathBuf::from(path), parse(start_s, "start")?, parse(end_s, "end")?))
        }
        None => Ok((PathBuf::from(s), None, None)),
    }
}
