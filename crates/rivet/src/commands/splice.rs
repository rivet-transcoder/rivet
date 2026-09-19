//! Implementation of `rivet splice`.

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use rivet::{RungArtifact, TranscodeSettings};

use crate::{AudioArg, ModeArg, value_name};

/// Collected CLI arguments for the `splice` subcommand.
pub(crate) struct SpliceArgs {
    pub output: PathBuf,
    pub clips: Vec<String>,
    pub mode: ModeArg,
    pub segment_seconds: f32,
    pub codec: Option<String>,
    pub crf: Option<u8>,
    pub video_bitrate: Option<String>,
    pub video_buffer: Option<String>,
    pub audio: AudioArg,
    pub subtitles: String,
    pub decode: rivet::DecodePolicy,
    pub encode: Option<rivet::EncodePolicy>,
}

pub(crate) fn run(args: SpliceArgs) -> Result<()> {
    let SpliceArgs {
        output,
        clips,
        mode,
        segment_seconds,
        codec,
        crf,
        video_bitrate,
        video_buffer,
        audio,
        subtitles,
        decode,
        encode,
    } = args;
    let parsed = clips
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
    let video_codec = codec
        .as_deref()
        .map(rivet::settings::parse_video_codec)
        .transpose()
        .context("parsing --codec")?;
    let is_hls = matches!(mode, ModeArg::Hls);
    let mut settings = TranscodeSettings {
        segment_seconds: Some(segment_seconds),
        crf,
        video_codec,
        decode_policy: decode,
        encode,
        ..Default::default()
    };
    // Worded values go through the settings vocabulary, like every surface.
    settings.apply_kv("mode", &value_name(mode))?;
    settings.apply_kv("audio", &value_name(audio))?;
    settings.apply_kv("subtitles", &subtitles)?;
    if let Some(v) = &video_bitrate {
        settings.apply_kv("video-bitrate", v).context("parsing --video-bitrate")?;
    }
    if let Some(v) = &video_buffer {
        settings.apply_kv("video-buffer", v).context("parsing --video-buffer")?;
    }
    let spec = settings
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
