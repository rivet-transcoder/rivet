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
    pub max_short_side: Option<String>,
    pub segment_seconds: f32,
    pub crf: Option<u8>,
    pub target: Option<rivet::codec::encode::tuning::QualityTarget>,
    pub gop: Option<String>,
    pub video_bitrate: Option<String>,
    pub video_buffer: Option<String>,
    pub rate_mode: Option<String>,
    pub video_speed: Option<String>,
    pub audio: AudioArg,
    pub audio_bitrate: Option<String>,
    pub audio_quality: Option<String>,
    pub audio_channels: Option<String>,
    pub audio_stereo_fallback: bool,
    pub audio_bit_depth: Option<String>,
    pub he_aac: Option<String>,
    pub audio_decode_deny: Option<String>,
    pub metadata_keep: Option<String>,
    pub flac_compression: Option<String>,
    pub audio_container: Option<String>,
    pub audio_filter: Option<String>,
    pub subtitles: String,
    pub max_fps: Option<String>,
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
    pub fitting: super::FitArgs,
    pub file: super::FileArgs,
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
        segment_seconds: Some(args.segment_seconds),
        crf: args.crf,
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
        video_codec,
        trim_start: args.trim_start,
        trim_end: args.trim_end,
        ..Default::default()
    };
    super::OutputShaping {
        target: args.target,
        gop: args.gop.clone(),
        video_bitrate: args.video_bitrate.clone(),
        video_buffer: args.video_buffer.clone(),
        rate_mode: args.rate_mode.clone(),
        video_speed: args.video_speed.clone(),
        audio_bitrate: args.audio_bitrate.clone(),
        audio_channels: args.audio_channels.clone(),
        audio_filter: args.audio_filter.clone(),
        color: args.color,
        chroma_downsample: args.chroma_downsample,
        pixel_format: args.pixel_format,
        filter: args.filter.clone(),
    }
    .apply(&mut settings)?;
    if let Some(v) = &args.max_short_side {
        settings.apply_kv("max-short-side", v).context("parsing --max-short-side")?;
    }
    if let Some(v) = &args.max_fps {
        settings.apply_kv("max-fps", v).context("parsing --max-fps")?;
    }
    args.fitting.apply(&mut settings)?;
    args.file.apply(&mut settings)?;
    settings.apply_kv("mode", &value_name(args.mode))?;
    settings.apply_kv("audio", &value_name(args.audio))?;
    settings.audio_stereo_fallback = args.audio_stereo_fallback;
    for (key, value) in [
        ("audio-bit-depth", &args.audio_bit_depth),
        ("he-aac", &args.he_aac),
        ("audio-decode-deny", &args.audio_decode_deny),
        ("metadata-keep", &args.metadata_keep),
        ("flac-compression", &args.flac_compression),
        ("audio-container", &args.audio_container),
        ("audio-quality", &args.audio_quality),
    ] {
        if let Some(v) = value {
            settings.apply_kv(key, v).with_context(|| format!("parsing --{key}"))?;
        }
    }
    settings.apply_kv("subtitles", &args.subtitles)?;
    settings.apply_kv("seam", &value_name(args.seam_mode))?;
    if let Some(family) = args.gpu_family {
        settings.apply_kv("gpu-family", &value_name(family))?;
    }
    let spec = settings
        .into_spec_for(&probed)
        .context("building output spec")?;

    // Progress: throttled, with rate / elapsed / ETA / projected size.
    let sink = Arc::new(super::progress::ProgressPrinter::new(spec.rungs.len()));

    // Determine output target — by the spec's shape, since an input with no
    // video makes a single-file job an audio-only one.
    let (output_dir, single_file_target) = plan_output(&args, &spec);
    // Made before the job runs, so an unusable path fails before any work.
    // A job that ends with nothing in it (refused by the encode pool's
    // preflight, say) takes back what this run made when `made_dir` drops;
    // a directory that already existed is never removed.
    let made_dir = match output_dir.as_deref() {
        Some(dir) => Some(
            rivet::output_dir::CreatedDir::create(dir)
                .with_context(|| format!("creating output dir {}", dir.display()))?,
        ),
        None => None,
    };

    let out = rivet::run_job_blocking_owned(
        bytes.clone(),
        &spec,
        output_dir.as_deref(),
        sink,
    )
    .with_context(|| format!("transcoding {}", args.input.display()))?;

    write_outputs(&args, &out, output_dir.as_deref(), single_file_target.as_deref(), spec.file_extension())?;
    if let Some(made) = made_dir {
        made.keep();
    }
    print_summary(&args.input, &out, spec.file_extension());
    Ok(())
}

/// Decide where outputs go.
/// Returns `(output_dir, single_file_target)`. Makes nothing: the caller
/// creates the directory for the run.
fn plan_output(args: &TranscodeArgs, spec: &rivet::OutputSpec) -> (Option<PathBuf>, Option<PathBuf>) {
    if spec.mode == rivet::OutputMode::AudioOnly {
        let file = args.output.clone().unwrap_or_else(|| default_file_ext(&args.input, spec.file_extension()));
        return (None, Some(file));
    }
    match args.mode {
        ModeArg::Audio => unreachable!("an audio-mode spec is AudioOnly"),
        ModeArg::Hls => {
            let dir = args
                .output
                .clone()
                .unwrap_or_else(|| default_dir(&args.input, "hls"));
            (Some(dir), None)
        }
        ModeArg::Single => {
            // Multi-rung → directory; single-rung → file.
            let multi = args.rungs.len() > 1 || args.ladder;
            if multi {
                let dir = args
                    .output
                    .clone()
                    .unwrap_or_else(|| default_dir(&args.input, "av1"));
                // SingleFile bytes are returned in memory; write_outputs places
                // each rung at `<dir>/<label>.mp4`.
                (Some(dir), None)
            } else {
                let file = args
                    .output
                    .clone()
                    .unwrap_or_else(|| default_file(&args.input, spec));
                (None, Some(file))
            }
        }
    }
}

fn write_outputs(
    args: &TranscodeArgs,
    out: &JobOutput,
    output_dir: Option<&Path>,
    single_file_target: Option<&Path>,
    ext: &str,
) -> Result<()> {
    match args.mode {
        ModeArg::Hls => {
            // HLS package already written under output_dir by the engine.
        }
        ModeArg::Single | ModeArg::Audio => {
            if let Some(file) = single_file_target {
                // Exactly one rung.
                if let Some(r) = out.rungs.first()
                    && let RungArtifact::File(bytes) = &r.artifact
                {
                    std::fs::write(file, bytes)
                        .with_context(|| format!("writing {}", file.display()))?;
                }
            } else if let Some(dir) = output_dir {
                for r in &out.rungs {
                    if let RungArtifact::File(bytes) = &r.artifact {
                        let path = dir.join(format!("{}.{ext}", r.label));
                        std::fs::write(&path, bytes)
                            .with_context(|| format!("writing {}", path.display()))?;
                    }
                }
            }
        }
    }
    Ok(())
}

fn print_summary(input: &Path, out: &JobOutput, ext: &str) {
    println!(
        "{} ({}x{} @ {:.3} fps {})",
        input.display(),
        out.source_dims.0,
        out.source_dims.1,
        out.source_frame_rate,
        out.source_codec,
    );
    match &out.audio_codecs {
        Some(codecs) => println!("  audio: {} [codecs={codecs}]", out.audio_handling),
        None => println!("  audio: {}", out.audio_handling),
    }
    for r in &out.rungs {
        let where_ = match &r.artifact {
            RungArtifact::File(_) => ext.to_string(),
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
    for r in out.renditions.iter().filter(|r| r.duplicate_of.is_some()) {
        println!(
            "  {}x{} not written: the source fits it at {}x{}, the same as {} (--upscale to enlarge)",
            r.requested.0, r.requested.1, r.output.0, r.output.1, r.label,
        );
    }
    if let Some(master) = &out.master_playlist {
        println!("  master playlist: {}", master.display());
    }
    println!("  done in {:.2}s", out.elapsed.as_secs_f64());
}

fn parse_wxh(s: &str) -> Result<rivet::settings::RungArg> {
    let rung = rivet::settings::parse_rung(s)?;
    if rung.width == 0 || rung.height == 0 {
        bail!("rung '{s}' has a zero dimension");
    }
    Ok(rivet::settings::RungArg { width: rung.width & !1, height: rung.height & !1, ..rung })
}

/// `<stem>.<codec>.<ext>` beside the input: `clip.av1.mp4`, `clip.prores.mov`,
/// `clip.vp9.webm`.
fn default_file(input: &Path, spec: &rivet::OutputSpec) -> PathBuf {
    let stem = input
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "output".to_string());
    let codec = spec.video_codec.codec().label();
    let mut out = input.to_path_buf();
    out.set_file_name(format!("{stem}.{codec}.{}", spec.file_extension()));
    out
}

fn default_file_ext(input: &Path, ext: &str) -> PathBuf {
    let stem = input
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "output".to_string());
    let mut out = input.to_path_buf();
    out.set_file_name(format!("{stem}.{ext}"));
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
