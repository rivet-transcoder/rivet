//! Implementation of `rivet splice`.

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use rivet::{RungArtifact, TranscodeSettings};

use crate::{AudioArg, ModeArg, value_name};

#[allow(clippy::too_many_arguments)]
pub(crate) fn run(
    output: PathBuf,
    clips: Vec<String>,
    mode: ModeArg,
    segment_seconds: f32,
    codec: Option<String>,
    crf: Option<u8>,
    audio: AudioArg,
    subtitles: String,
    decode: rivet::DecodePolicy,
    encode: Option<rivet::EncodePolicy>,
) -> Result<()> {
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
    // MP4 bytes in memory (one rung at source resolution).
    let out_dir = if is_hls {
        std::fs::create_dir_all(&output)
            .with_context(|| format!("creating output dir {}", output.display()))?;
        Some(output.clone())
    } else {
        None
    };
    let out = rivet::run_splice_job_blocking(splice_clips, &spec, out_dir.as_deref(), sink)
        .context("splicing clips")?;

    if !is_hls {
        if let Some(r) = out.rungs.first() {
            if let RungArtifact::File(bytes) = &r.artifact {
                std::fs::write(&output, bytes)
                    .with_context(|| format!("writing {}", output.display()))?;
            }
        }
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
