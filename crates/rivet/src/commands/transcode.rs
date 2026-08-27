//! Implementation of `rivet transcode`.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use rivet::{JobOutput, RungArtifact, TranscodeSettings};

use crate::{AudioArg, ColorArg, GpuFamilyArg, ModeArg, PixelArg, SeamArg, value_name};

/// Collected CLI arguments for the `transcode` subcommand.
pub(crate) struct TranscodeArgs {
    pub input: PathBuf,
    pub output: Option<PathBuf>,
    pub mode: ModeArg,
    pub rungs: Vec<String>,
    pub ladder: bool,
    pub max_short_side: Option<u32>,
    pub segment_seconds: f32,
    pub crf: Option<u8>,
    pub target: Option<rivet::codec::encode::tuning::QualityTarget>,
    pub gop: Option<u32>,
    pub audio: AudioArg,
    pub audio_bitrate: Option<String>,
    pub audio_filter: Option<String>,
    pub subtitles: String,
    pub max_fps: Option<f64>,
    pub gpu: Option<u32>,
    pub single_gpu: bool,
    pub gpu_family: Option<GpuFamilyArg>,
    pub decode: rivet::DecodePolicy,
    pub encode: Option<rivet::EncodePolicy>,
    pub encode_policy: Option<String>,
    pub color: ColorArg,
    pub chroma_downsample: crate::ChromaArg,
    pub pixel_format: PixelArg,
    pub seam_mode: SeamArg,
    pub filter: Option<String>,
    pub codec: Option<String>,
    pub trim_start: Option<f64>,
    pub trim_end: Option<f64>,
}

pub(crate) fn run(args: TranscodeArgs) -> Result<()> {
    // One allocation for the whole file, shared by the probe, the job engine,
    // and every demuxer underneath. A 9 GB remux copied per consumer is how
    // this used to reach 34 GB RSS and get OOM-killed.
    let bytes = bytes::Bytes::from(
        std::fs::read(&args.input)
            .with_context(|| format!("reading input {}", args.input.display()))?,
    );

    // Probe to resolve the ladder when not given explicitly.
    let probed = rivet::probe_bytes_shared(bytes.clone()).context("probing input")?;

    // Build the canonical `TranscodeSettings` (the same knob set the HTTP API
    // and pipe/ipc fill), then the one shared spec builder.
    let rungs = args
        .rungs
        .iter()
        .map(|s| parse_wxh(s))
        .collect::<Result<Vec<_>>>()?;
    let filters = match args.filter.as_deref() {
        Some(s) => codec::filter::parse_chain(s).context("parsing --filter")?,
        None => Vec::new(),
    };
    let audio_filters = match args.audio_filter.as_deref() {
        Some(s) => codec::audio::filter::parse_chain(s).context("parsing --audio-filter")?,
        None => Vec::new(),
    };
    let audio_bitrate = args
        .audio_bitrate
        .as_deref()
        .map(rivet::settings::parse_bitrate)
        .transpose()
        .context("parsing --audio-bitrate")?;
    let video_codec = args
        .codec
        .as_deref()
        .map(rivet::settings::parse_video_codec)
        .transpose()
        .context("parsing --codec")?;
    // Typed values are placed directly; every *worded* value goes through
    // `apply_kv` under the same key the IPC socket, the HTTP API and the batch
    // manifest use, so the CLI interprets nothing on its own — the clap enums
    // only validate spelling for `--help`.
    let mut settings = TranscodeSettings {
        rungs,
        ladder: args.ladder,
        max_short_side: args.max_short_side,
        segment_seconds: Some(args.segment_seconds),
        crf: args.crf,
        target: args.target,
        gop: args.gop,
        audio_bitrate,
        audio_filters,
        max_fps: args.max_fps,
        gpu: args.gpu,
        single_gpu: args.single_gpu,
        decode_policy: args.decode,
        encode: args.encode,
        encode_policy: args
            .encode_policy
            .as_deref()
            .map(rivet::settings::parse_encode_policy)
            .transpose()
            .context("parsing --encode-policy")?,
        filters,
        video_codec,
        trim_start: args.trim_start,
        trim_end: args.trim_end,
        ..Default::default()
    };
    settings.apply_kv("mode", &value_name(args.mode))?;
    settings.apply_kv("audio", &value_name(args.audio))?;
    settings.apply_kv("subtitles", &args.subtitles)?;
    settings.apply_kv("color", &value_name(args.color))?;
    settings.apply_kv("chroma-downsample", &value_name(args.chroma_downsample))?;
    settings.apply_kv("bit-depth", &value_name(args.pixel_format))?;
    settings.apply_kv("seam", &value_name(args.seam_mode))?;
    if let Some(family) = args.gpu_family {
        settings.apply_kv("gpu-family", &value_name(family))?;
    }
    let spec = settings
        .into_spec(probed.width, probed.height)
        .context("building output spec")?;

    // Progress: throttled, with rate / elapsed / ETA / projected size.
    let sink = Arc::new(super::progress::ProgressPrinter::new(spec.rungs.len()));

    // Determine output target.
    let (output_dir, single_file_target) = plan_output(&args)?;

    let out = rivet::run_job_blocking_owned(
        bytes.clone(),
        &spec,
        output_dir.as_deref(),
        sink,
    )
    .with_context(|| format!("transcoding {}", args.input.display()))?;

    write_outputs(&args, &out, output_dir.as_deref(), single_file_target.as_deref())?;
    print_summary(&args.input, &out);
    Ok(())
}

/// Decide where outputs go.
/// Returns `(output_dir, single_file_target)`.
fn plan_output(args: &TranscodeArgs) -> Result<(Option<PathBuf>, Option<PathBuf>)> {
    match args.mode {
        ModeArg::Hls => {
            let dir = args
                .output
                .clone()
                .unwrap_or_else(|| default_dir(&args.input, "hls"));
            std::fs::create_dir_all(&dir)
                .with_context(|| format!("creating output dir {}", dir.display()))?;
            Ok((Some(dir), None))
        }
        ModeArg::Single => {
            // Multi-rung → directory; single-rung → file.
            let multi = args.rungs.len() > 1 || args.ladder;
            if multi {
                let dir = args
                    .output
                    .clone()
                    .unwrap_or_else(|| default_dir(&args.input, "av1"));
                std::fs::create_dir_all(&dir)
                    .with_context(|| format!("creating output dir {}", dir.display()))?;
                // SingleFile bytes are returned in memory; write_outputs places
                // each rung at `<dir>/<label>.mp4`.
                Ok((Some(dir), None))
            } else {
                let file = args
                    .output
                    .clone()
                    .unwrap_or_else(|| default_file(&args.input));
                Ok((None, Some(file)))
            }
        }
    }
}

fn write_outputs(
    args: &TranscodeArgs,
    out: &JobOutput,
    output_dir: Option<&Path>,
    single_file_target: Option<&Path>,
) -> Result<()> {
    match args.mode {
        ModeArg::Hls => {
            // HLS package already written under output_dir by the engine.
        }
        ModeArg::Single => {
            if let Some(file) = single_file_target {
                // Exactly one rung.
                if let Some(r) = out.rungs.first() {
                    if let RungArtifact::File(bytes) = &r.artifact {
                        std::fs::write(file, bytes)
                            .with_context(|| format!("writing {}", file.display()))?;
                    }
                }
            } else if let Some(dir) = output_dir {
                for r in &out.rungs {
                    if let RungArtifact::File(bytes) = &r.artifact {
                        let path = dir.join(format!("{}.mp4", r.label));
                        std::fs::write(&path, bytes)
                            .with_context(|| format!("writing {}", path.display()))?;
                    }
                }
            }
        }
    }
    Ok(())
}

fn print_summary(input: &Path, out: &JobOutput) {
    println!(
        "{} ({}x{} @ {:.3} fps {})",
        input.display(),
        out.source_dims.0,
        out.source_dims.1,
        out.source_frame_rate,
        out.source_codec,
    );
    println!("  audio: {}", out.audio_handling);
    for r in &out.rungs {
        let where_ = match &r.artifact {
            RungArtifact::File(_) => "mp4".to_string(),
            RungArtifact::HlsRendition { relative_dir, .. } => relative_dir.clone(),
        };
        println!(
            "  {:<6} {}x{}  {} frames  {:.2} MiB  [{}]",
            r.label,
            r.width,
            r.height,
            r.frames,
            r.bytes as f64 / (1024.0 * 1024.0),
            where_,
        );
    }
    if let Some(master) = &out.master_playlist {
        println!("  master playlist: {}", master.display());
    }
    println!("  done in {:.2}s", out.elapsed.as_secs_f64());
}

fn parse_wxh(s: &str) -> Result<(u32, u32)> {
    let (w, h) = s
        .split_once(['x', 'X'])
        .ok_or_else(|| anyhow::anyhow!("rung '{s}' is not WxH (e.g. 1280x720)"))?;
    let w: u32 = w.trim().parse().with_context(|| format!("bad width in '{s}'"))?;
    let h: u32 = h.trim().parse().with_context(|| format!("bad height in '{s}'"))?;
    if w == 0 || h == 0 {
        bail!("rung '{s}' has a zero dimension");
    }
    Ok((w & !1, h & !1))
}

fn default_file(input: &Path) -> PathBuf {
    let stem = input
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "output".to_string());
    let mut out = input.to_path_buf();
    out.set_file_name(format!("{stem}.av1.mp4"));
    out
}

fn default_dir(input: &Path, suffix: &str) -> PathBuf {
    let stem = input
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "output".to_string());
    let mut out = input.to_path_buf();
    out.set_file_name(format!("{stem}.{suffix}"));
    out
}
