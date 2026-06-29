//! `rivet` — command-line video transcoder.
//!
//! ```text
//! # Single MP4 (source resolution)
//! rivet transcode input.mkv -o output.mp4
//!
//! # Multi-rung ABR ladder of MP4s into a directory
//! rivet transcode input.mkv -o out_dir/ --rung 1920x1080 --rung 1280x720 --rung 640x360
//!
//! # Standard ladder, auto-derived from the source
//! rivet transcode input.mkv -o out_dir/ --ladder
//!
//! # CMAF/HLS package with 4-second segments
//! rivet transcode input.mkv -o hls_dir/ --mode hls --ladder --segment-seconds 4
//!
//! # Quality / audio knobs
//! rivet transcode input.mkv -o out.mp4 --crf 28 --speed 6 --audio opus
//!
//! rivet probe input.mkv [--json]
//! ```
//!
//! Logging verbosity is controlled by `RUST_LOG` (e.g. `RUST_LOG=debug`).

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand, ValueEnum};
use tracing_subscriber::EnvFilter;

use rivet::progress::{RungProgress, RungStatus};
use rivet::spec::{AudioCodecPolicy, BitDepth, ChunkSeamMode, ColorPolicy, GpuFamily};
use rivet::{JobOutput, RungArtifact, TranscodeSettings};

#[derive(Parser)]
#[command(
    name = "rivet",
    version,
    about = "Modular GPU-accelerated video transcoder (AV1 + Opus).",
    long_about = None
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Clone, Copy, ValueEnum)]
enum ModeArg {
    /// One self-contained MP4 per rung.
    Single,
    /// Segmented CMAF + HLS package.
    Hls,
}

#[derive(Clone, Copy, ValueEnum)]
enum AudioArg {
    /// Passthrough when possible, else transcode to Opus, else drop.
    Auto,
    /// Produce Opus audio.
    Opus,
    /// Drop audio (video only).
    Drop,
}

impl From<AudioArg> for AudioCodecPolicy {
    fn from(a: AudioArg) -> Self {
        match a {
            AudioArg::Auto => AudioCodecPolicy::Auto,
            AudioArg::Opus => AudioCodecPolicy::ForceOpus,
            AudioArg::Drop => AudioCodecPolicy::Drop,
        }
    }
}

#[derive(Clone, Copy, ValueEnum)]
enum GpuFamilyArg {
    Nvidia,
    Amd,
    Intel,
}

impl From<GpuFamilyArg> for GpuFamily {
    fn from(a: GpuFamilyArg) -> Self {
        match a {
            GpuFamilyArg::Nvidia => GpuFamily::Nvidia,
            GpuFamilyArg::Amd => GpuFamily::Amd,
            GpuFamilyArg::Intel => GpuFamily::Intel,
        }
    }
}

#[derive(Clone, Copy, ValueEnum)]
enum ColorArg {
    /// Tonemap HDR sources to SDR BT.709 (default).
    Sdr,
    /// HDR10: BT.2020 + PQ, 10-bit (needs a 10-bit encoder: nvidia/amd/qsv/ffmpeg).
    Hdr10,
    /// HLG: BT.2020 + ARIB STD-B67, 10-bit (needs a 10-bit encoder: nvidia/amd/qsv/ffmpeg).
    Hlg,
    /// Preserve the source color/transfer/bit-depth verbatim.
    Passthrough,
}

impl From<ColorArg> for ColorPolicy {
    fn from(a: ColorArg) -> Self {
        match a {
            ColorArg::Sdr => ColorPolicy::TonemapToSdr,
            ColorArg::Hdr10 => ColorPolicy::Hdr10,
            ColorArg::Hlg => ColorPolicy::Hlg,
            ColorArg::Passthrough => ColorPolicy::Passthrough,
        }
    }
}

#[derive(Clone, Copy, ValueEnum)]
enum PixelArg {
    /// Follow the color policy (default).
    Auto,
    #[value(name = "8bit")]
    Eight,
    #[value(name = "10bit")]
    Ten,
}

impl From<PixelArg> for BitDepth {
    fn from(a: PixelArg) -> Self {
        match a {
            PixelArg::Auto => BitDepth::Auto,
            PixelArg::Eight => BitDepth::EightBit,
            PixelArg::Ten => BitDepth::TenBit,
        }
    }
}

#[derive(Clone, Copy, ValueEnum)]
enum SeamArg {
    /// Chunk a single file across all GPUs for speed (default). NVENC chunks run
    /// VBR — possible mild quality steps at the ~2 s seams.
    Parallel,
    /// Chunk across GPUs but force constant-QP so seams are quality-flat. The QP
    /// is derived from the quality target, so quality still tracks it.
    Constqp,
    /// One encoder for the whole file: seam-free + quality-target-accurate, no
    /// multi-GPU single-file speedup.
    Serial,
}

impl From<SeamArg> for ChunkSeamMode {
    fn from(a: SeamArg) -> Self {
        match a {
            SeamArg::Parallel => ChunkSeamMode::Parallel,
            SeamArg::Constqp => ChunkSeamMode::ParallelConstQp,
            SeamArg::Serial => ChunkSeamMode::Serial,
        }
    }
}

#[derive(Subcommand)]
enum Command {
    /// Transcode an input file to AV1.
    Transcode {
        /// Input media file (any supported container/codec).
        input: PathBuf,
        /// Output path: a file (single mode, one rung) or a directory
        /// (single mode multi-rung, or HLS). Defaults to `<input>.av1.mp4`
        /// for the simple single-rung case.
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Output mode.
        #[arg(long, value_enum, default_value = "single")]
        mode: ModeArg,
        /// A ladder rung as `WxH` (repeatable). If omitted, a single rung at
        /// the source resolution is used (unless `--ladder` is set).
        #[arg(long = "rung", value_name = "WxH")]
        rungs: Vec<String>,
        /// Auto-derive a standard ABR ladder from the source resolution.
        #[arg(long)]
        ladder: bool,
        /// Ladder cap on the short side (with `--ladder`). Default 1080.
        #[arg(long)]
        max_short_side: Option<u32>,
        /// Target segment length in seconds (HLS mode).
        #[arg(long, default_value_t = 4.0)]
        segment_seconds: f32,
        /// Constant rate factor (encoder-native, lower = better quality).
        #[arg(long)]
        crf: Option<u8>,
        /// Encoder speed preset (encoder-native).
        #[arg(long)]
        speed: Option<u8>,
        /// Audio handling.
        #[arg(long, value_enum, default_value = "auto")]
        audio: AudioArg,
        /// Cap the output frame rate.
        #[arg(long)]
        max_fps: Option<f64>,
        /// Pin hardware encode/decode to this GPU index (implies single-GPU).
        #[arg(long)]
        gpu: Option<u32>,
        /// Encode serially on a single GPU instead of chunk-encoding across all
        /// GPUs. Without `--gpu N` this picks the first GPU. Default: all GPUs.
        #[arg(long)]
        single_gpu: bool,
        /// Constrain encode to one GPU vendor family (e.g. all NVIDIA cards,
        /// ignoring an integrated AMD/Intel GPU).
        #[arg(long, value_enum)]
        gpu_family: Option<GpuFamilyArg>,
        /// Pin the decode pump to this GPU index (default: follows the encode
        /// policy). E.g. decode on an iGPU while the dGPUs encode.
        #[arg(long)]
        decode_gpu: Option<u32>,
        /// Output color / tonemap policy.
        #[arg(long, value_enum, default_value = "sdr")]
        color: ColorArg,
        /// Output luma bit depth.
        #[arg(long, value_enum, default_value = "auto")]
        pixel_format: PixelArg,
        /// Multi-GPU single-file chunk seam handling: `parallel` (fastest),
        /// `constqp` (seam-flat constant-QP, quality still tracks the target), or
        /// `serial` (one encoder, seam-free, no multi-GPU single-file speedup).
        #[arg(long = "seam-mode", value_enum, default_value = "parallel")]
        seam_mode: SeamArg,
        /// Video filter chain (ffmpeg-`-vf`-style), applied before scaling, e.g.
        /// `crop=1280:720,hflip` or `pad=1920:1080` / `rotate=90` / `grayscale`.
        #[arg(long)]
        filter: Option<String>,
        /// Output video codec: `av1` (default, royalty-clean), `h264`, or `h265`.
        /// All three work for single-file MP4 and CMAF/HLS.
        #[arg(long)]
        codec: Option<String>,
        /// Splice: trim the input, keeping from this time (seconds). The output
        /// is re-based to zero. Trimmed jobs use the serial encode path.
        #[arg(long)]
        trim_start: Option<f64>,
        /// Splice: trim the input, keeping until this time (seconds).
        #[arg(long)]
        trim_end: Option<f64>,
    },
    /// Splice: concatenate (and per-clip trim) several inputs into one MP4.
    ///
    /// Clips are joined in order and re-encoded to a uniform output, so they may
    /// differ in codec / resolution / color. Trim a clip with `PATH@START-END`
    /// (seconds, either side optional), e.g.
    /// `rivet splice -o out.mp4 a.mp4@0-5 b.mp4@10-20 c.mp4`.
    Splice {
        /// Output MP4 file.
        #[arg(short, long)]
        output: PathBuf,
        /// Input clips in order: `PATH` or `PATH@START-END` (seconds).
        #[arg(required = true)]
        clips: Vec<String>,
        /// Output video codec: `av1` (default), `h264`, or `h265`.
        #[arg(long)]
        codec: Option<String>,
        /// Constant rate factor (quality; lower = better).
        #[arg(long)]
        crf: Option<u8>,
        /// Audio handling: `auto` (default), `opus`, `drop`.
        #[arg(long, value_enum, default_value = "auto")]
        audio: AudioArg,
    },
    /// Inspect an input file without transcoding it.
    Probe {
        /// Input media file.
        input: PathBuf,
        /// Emit machine-readable JSON instead of a human summary.
        #[arg(long)]
        json: bool,
    },
    /// List detected GPU devices (vendor, name, VRAM, AV1-encode, live load).
    Devices {
        /// Emit machine-readable JSON instead of a human table.
        #[arg(long)]
        json: bool,
    },
    /// Report what this build + host can do: enabled backends, encode/decode
    /// codec support, and the detected devices.
    #[command(visible_alias = "caps")]
    Capabilities {
        /// Emit machine-readable JSON instead of a human summary.
        #[arg(long)]
        json: bool,
    },
    /// Stream a transcode: read media from **stdin**, write the AV1/MP4 to
    /// **stdout**. With no options it's the source-resolution single-file
    /// default; the flags override quality/size/color/audio. E.g.
    /// `cat in.mkv | rivet pipe --crf 28 --color hdr10 > out.mp4`.
    Pipe {
        /// Constant rate factor (lower = higher quality).
        #[arg(long)]
        crf: Option<u8>,
        /// Encoder speed preset.
        #[arg(long)]
        speed: Option<u8>,
        /// Audio policy.
        #[arg(long, value_enum)]
        audio: Option<AudioArg>,
        /// Output color / tonemap policy.
        #[arg(long, value_enum)]
        color: Option<ColorArg>,
        /// Output bit depth.
        #[arg(long = "bit-depth", visible_alias = "pixel-format", value_enum)]
        bit_depth: Option<PixelArg>,
        /// Cap the output frame rate.
        #[arg(long = "max-fps")]
        max_fps: Option<f64>,
        /// Output width (scales; defaults to source).
        #[arg(long)]
        width: Option<u32>,
        /// Output height (scales; defaults to source).
        #[arg(long)]
        height: Option<u32>,
        /// Pin encode to this GPU index.
        #[arg(long)]
        gpu: Option<u32>,
        /// Video filter chain (e.g. `crop=1280:720,hflip`).
        #[arg(long)]
        filter: Option<String>,
    },
    /// Run a **Unix-domain-socket** IPC server (needs the `ipc` feature; Unix
    /// only at runtime). Each connection: the client writes media, half-closes
    /// its write side, then reads the transcoded AV1/MP4 back. Per-job settings
    /// can prefix the stream as a `#rivet key=value …\n` header line. Lets an
    /// app stream data in and out without HTTP or temp files.
    #[cfg(feature = "ipc")]
    Ipc {
        /// Socket path to bind, e.g. `/tmp/rivet.sock`.
        #[arg(long)]
        socket: PathBuf,
    },
    /// Convert many files from a YAML/JSON **manifest** in one run (needs the
    /// `batch` feature). See `docs/batch.md` for the DSL.
    #[cfg(feature = "batch")]
    Batch {
        /// Manifest path (.yaml / .yml / .json).
        manifest: PathBuf,
        /// Parse + validate + list the planned jobs without converting anything.
        #[arg(long)]
        dry_run: bool,
        /// Abort on the first failed job (overrides the manifest's `on_error`).
        #[arg(long)]
        stop_on_error: bool,
    },
    /// Run the HTTP transcode API server so another app can signal transcodes
    /// over the network (needs the `server` feature).
    #[cfg(feature = "server")]
    Serve {
        /// Address to bind, e.g. `0.0.0.0:8080`.
        #[arg(long, default_value = "127.0.0.1:8080")]
        addr: String,
    },
}

fn main() -> ExitCode {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")))
        .with_writer(std::io::stderr)
        .init();

    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Transcode {
            input,
            output,
            mode,
            rungs,
            ladder,
            max_short_side,
            segment_seconds,
            crf,
            speed,
            audio,
            max_fps,
            gpu,
            single_gpu,
            gpu_family,
            decode_gpu,
            color,
            pixel_format,
            seam_mode,
            filter,
            codec,
            trim_start,
            trim_end,
        } => transcode_cmd(TranscodeArgs {
            input,
            output,
            mode,
            rungs,
            ladder,
            max_short_side,
            segment_seconds,
            crf,
            speed,
            audio,
            max_fps,
            gpu,
            single_gpu,
            gpu_family,
            decode_gpu,
            color,
            pixel_format,
            seam_mode,
            filter,
            codec,
            trim_start,
            trim_end,
        }),
        Command::Splice {
            output,
            clips,
            codec,
            crf,
            audio,
        } => splice_cmd(output, clips, codec, crf, audio),
        Command::Probe { input, json } => {
            let info = rivet::probe_file(&input)
                .with_context(|| format!("probing {}", input.display()))?;
            if json {
                println!("{}", probe_json(&info));
            } else {
                print_probe(&input, &info);
            }
            Ok(())
        }
        Command::Devices { json } => {
            devices_cmd(json);
            Ok(())
        }
        Command::Capabilities { json } => {
            capabilities_cmd(json);
            Ok(())
        }
        Command::Pipe {
            crf,
            speed,
            audio,
            color,
            bit_depth,
            max_fps,
            width,
            height,
            gpu,
            filter,
        } => pipe_cmd(TranscodeSettings {
            crf,
            speed,
            audio: audio.map(Into::into),
            color: color.map(Into::into),
            bit_depth: bit_depth.map(Into::into),
            max_fps,
            width,
            height,
            gpu,
            filters: match filter {
                Some(s) => codec::filter::parse_chain(&s).context("parsing --filter")?,
                None => Vec::new(),
            },
            ..Default::default()
        }),
        #[cfg(feature = "ipc")]
        Command::Ipc { socket } => ipc_cmd(&socket),
        #[cfg(feature = "batch")]
        Command::Batch {
            manifest,
            dry_run,
            stop_on_error,
        } => batch_cmd(&manifest, dry_run, stop_on_error),
        #[cfg(feature = "server")]
        Command::Serve { addr } => {
            let addr: std::net::SocketAddr = addr.parse().context("parsing --addr")?;
            let rt = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .context("building tokio runtime")?;
            eprintln!("rivet transcode API on http://{addr} (POST media to /v1/transcode)");
            rt.block_on(rivet::server::serve(addr))
        }
    }
}

struct TranscodeArgs {
    input: PathBuf,
    output: Option<PathBuf>,
    mode: ModeArg,
    rungs: Vec<String>,
    ladder: bool,
    max_short_side: Option<u32>,
    segment_seconds: f32,
    crf: Option<u8>,
    speed: Option<u8>,
    audio: AudioArg,
    max_fps: Option<f64>,
    gpu: Option<u32>,
    single_gpu: bool,
    gpu_family: Option<GpuFamilyArg>,
    decode_gpu: Option<u32>,
    color: ColorArg,
    pixel_format: PixelArg,
    seam_mode: SeamArg,
    filter: Option<String>,
    codec: Option<String>,
    trim_start: Option<f64>,
    trim_end: Option<f64>,
}

fn transcode_cmd(args: TranscodeArgs) -> Result<()> {
    let bytes = std::fs::read(&args.input)
        .with_context(|| format!("reading input {}", args.input.display()))?;

    // Probe to resolve the ladder when not given explicitly.
    let probed = rivet::probe_bytes(&bytes).context("probing input")?;

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
    let video_codec = args
        .codec
        .as_deref()
        .map(rivet::settings::parse_video_codec)
        .transpose()
        .context("parsing --codec")?;
    let settings = TranscodeSettings {
        mode: Some(match args.mode {
            ModeArg::Single => rivet::Mode::Single,
            ModeArg::Hls => rivet::Mode::Hls,
        }),
        rungs,
        ladder: args.ladder,
        max_short_side: args.max_short_side,
        segment_seconds: Some(args.segment_seconds),
        crf: args.crf,
        speed: args.speed,
        audio: Some(args.audio.into()),
        color: Some(args.color.into()),
        bit_depth: Some(args.pixel_format.into()),
        seam: Some(args.seam_mode.into()),
        max_fps: args.max_fps,
        gpu: args.gpu,
        gpu_family: args.gpu_family.map(Into::into),
        single_gpu: args.single_gpu,
        decode_gpu: args.decode_gpu,
        width: None,
        height: None,
        filters,
        video_codec,
        trim_start: args.trim_start,
        trim_end: args.trim_end,
    };
    let spec = settings
        .into_spec(probed.width, probed.height)
        .context("building output spec")?;

    // Progress: one carriage-return line per rung update.
    let sink = Arc::new(rivet::fn_sink(|p: RungProgress| {
        eprintln!(
            "  [{:>6}] {:<6} {:>5.1}%  {} frames{}",
            p.label,
            status_str(p.status),
            p.percent,
            p.frames_done,
            p.message.as_deref().map(|m| format!("  ({m})")).unwrap_or_default(),
        );
    }));

    // Determine output target.
    let (output_dir, single_file_target) = plan_output(&args)?;

    let out = rivet::run_job_blocking(
        &bytes,
        &spec,
        output_dir.as_deref(),
        sink,
    )
    .with_context(|| format!("transcoding {}", args.input.display()))?;

    write_outputs(&args, &out, output_dir.as_deref(), single_file_target.as_deref())?;
    print_summary(&args.input, &out);
    Ok(())
}

/// Parse a splice clip spec: `PATH` or `PATH@START-END` (seconds, either side
/// optional). The `@` separator avoids the `:` in Windows drive paths.
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

fn splice_cmd(
    output: PathBuf,
    clip_specs: Vec<String>,
    codec: Option<String>,
    crf: Option<u8>,
    audio: AudioArg,
) -> Result<()> {
    let parsed = clip_specs
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
    let settings = TranscodeSettings {
        mode: Some(rivet::Mode::Single),
        crf,
        audio: Some(audio.into()),
        video_codec,
        ..Default::default()
    };
    let spec = settings
        .into_spec(probed.width, probed.height)
        .context("building output spec")?;

    let clips: Vec<rivet::Clip> = parsed
        .iter()
        .zip(clip_bytes)
        .map(|((_, start, end), bytes)| rivet::Clip::trimmed(bytes, *start, *end))
        .collect();

    let sink = Arc::new(rivet::fn_sink(|p: RungProgress| {
        eprintln!(
            "  [{:>6}] {:<6} {:>5.1}%  {} frames{}",
            p.label,
            status_str(p.status),
            p.percent,
            p.frames_done,
            p.message.as_deref().map(|m| format!("  ({m})")).unwrap_or_default(),
        );
    }));

    let out = rivet::run_splice_job_blocking(clips, &spec, None, sink).context("splicing clips")?;

    if let Some(r) = out.rungs.first() {
        if let rivet::RungArtifact::File(bytes) = &r.artifact {
            std::fs::write(&output, bytes)
                .with_context(|| format!("writing {}", output.display()))?;
        }
    }
    eprintln!(
        "  spliced {} clip(s) → {} ({:.2} MiB) in {:.2}s",
        parsed.len(),
        output.display(),
        out.rungs.first().map(|r| r.bytes as f64 / (1024.0 * 1024.0)).unwrap_or(0.0),
        out.elapsed.as_secs_f64(),
    );
    Ok(())
}

/// Build the rung list from `--rung` / `--ladder` / default-source.
/// Decide where outputs go. Returns (hls/multi output dir, single-file target).
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

fn status_str(s: RungStatus) -> &'static str {
    match s {
        RungStatus::Pending => "pend",
        RungStatus::Running => "run",
        RungStatus::Finalizing => "final",
        RungStatus::Completed => "done",
        RungStatus::Failed => "FAIL",
    }
}

fn print_probe(input: &Path, info: &rivet::MediaInfo) {
    println!("{}", input.display());
    println!("  container : {}", info.container);
    println!("  video     : {}", info.video_codec);
    println!("  dimensions: {}x{}", info.width, info.height);
    println!("  frame rate: {:.3} fps", info.frame_rate);
    if info.duration > 0.0 {
        println!("  duration  : {:.3} s", info.duration);
    }
    println!("  pixel fmt : {}", info.pixel_format);
    match &info.audio {
        Some(a) => println!("  audio     : {} {} Hz {} ch", a.codec, a.sample_rate, a.channels),
        None => println!("  audio     : (none)"),
    }
}

fn probe_json(info: &rivet::MediaInfo) -> String {
    let audio = match &info.audio {
        Some(a) => format!(
            "{{\"codec\":\"{}\",\"sample_rate\":{},\"channels\":{}}}",
            esc(&a.codec),
            a.sample_rate,
            a.channels
        ),
        None => "null".to_string(),
    };
    format!(
        "{{\"container\":\"{}\",\"video_codec\":\"{}\",\"width\":{},\"height\":{},\"frame_rate\":{},\"duration\":{},\"pixel_format\":\"{}\",\"audio\":{}}}",
        esc(&info.container),
        esc(&info.video_codec),
        info.width,
        info.height,
        info.frame_rate,
        info.duration,
        esc(&info.pixel_format),
        audio,
    )
}

fn esc(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

// ── `rivet devices` ────────────────────────────────────────────────

fn devices_cmd(json: bool) {
    let devices = codec::gpu::detect_gpus();
    if json {
        println!("{}", devices_json(&devices));
        return;
    }
    if devices.is_empty() {
        println!(
            "No GPUs detected (CPU-only host). GPU transcode needs a `nvidia` / `amd` / `qsv` \
             feature build with the matching hardware; the `ffmpeg` feature provides software."
        );
        return;
    }
    let util = codec::gpu::GpuUtilizationReader::new();
    println!("{} GPU(s) detected:\n", devices.len());
    for d in &devices {
        println!(
            "  [{}] {} {}",
            d.index,
            codec::gpu::manufacturer_label(d.vendor),
            d.name
        );
        println!("      generation : {}", d.generation);
        if d.vram_mib > 0 {
            println!("      VRAM       : {} MiB", d.vram_mib);
        }
        println!("      PCI        : {}", d.host_pci_address);
        println!(
            "      AV1 encode : {}",
            if codec::encode::av1_encode_capable(d) { "yes" } else { "no" }
        );
        // Live load is read via NVML — meaningful on NVIDIA only.
        if matches!(d.vendor, codec::gpu::GpuVendor::Nvidia) {
            let u = util.read(d);
            print!(
                "      load       : gpu {}% · enc {}% · dec {}% · mem {}/{} MiB",
                u.util_percent, u.encoder_percent, u.decoder_percent, u.mem_used_mib, u.mem_total_mib
            );
            if let Some(t) = u.temperature_c {
                print!(" · {t}°C");
            }
            println!();
        }
        println!();
    }
    println!("Run `rivet capabilities` for what this build can encode/decode.");
}

fn devices_json(devices: &[codec::gpu::GpuDevice]) -> String {
    let util = codec::gpu::GpuUtilizationReader::new();
    let items: Vec<String> = devices
        .iter()
        .map(|d| {
            let load = if matches!(d.vendor, codec::gpu::GpuVendor::Nvidia) {
                let u = util.read(d);
                let temp = u
                    .temperature_c
                    .map(|t| t.to_string())
                    .unwrap_or_else(|| "null".into());
                format!(
                    ",\"load\":{{\"gpu_percent\":{},\"encoder_percent\":{},\"decoder_percent\":{},\"mem_used_mib\":{},\"mem_total_mib\":{},\"temperature_c\":{}}}",
                    u.util_percent, u.encoder_percent, u.decoder_percent, u.mem_used_mib, u.mem_total_mib, temp
                )
            } else {
                String::new()
            };
            format!(
                "{{\"index\":{},\"vendor\":\"{}\",\"name\":\"{}\",\"generation\":\"{}\",\"vram_mib\":{},\"pci\":\"{}\",\"av1_encode\":{}{}}}",
                d.index,
                codec::gpu::manufacturer_label(d.vendor),
                esc(&d.name),
                esc(&d.generation),
                d.vram_mib,
                esc(&d.host_pci_address),
                codec::encode::av1_encode_capable(d),
                load
            )
        })
        .collect();
    format!("{{\"gpus\":[{}]}}", items.join(","))
}

// ── `rivet capabilities` ───────────────────────────────────────────

fn capabilities_cmd(json: bool) {
    let enc = codec::encode::encode_backends();
    let dec_backends = codec::decode::decode_backends();
    let caps = codec::encode::build_output_caps();
    let dec = codec::decode::decode_capabilities();
    let devices = codec::gpu::detect_gpus();

    if json {
        let enc_b = enc
            .iter()
            .map(|b| format!("\"{b}\""))
            .collect::<Vec<_>>()
            .join(",");
        let dec_b = dec_backends
            .iter()
            .map(|b| format!("\"{b}\""))
            .collect::<Vec<_>>()
            .join(",");
        let codecs = dec
            .iter()
            .map(|d| {
                let bs = d
                    .backends
                    .iter()
                    .map(|b| format!("\"{b}\""))
                    .collect::<Vec<_>>()
                    .join(",");
                format!("{{\"codec\":\"{}\",\"backends\":[{}]}}", d.codec, bs)
            })
            .collect::<Vec<_>>()
            .join(",");
        println!(
            "{{\"encode\":{{\"codec\":\"av1\",\"backends\":[{}],\"max_bit_depth\":{},\"hdr\":{}}},\
             \"decode\":{{\"backends\":[{}],\"codecs\":[{}]}},\"devices\":{}}}",
            enc_b,
            caps.max_bit_depth,
            caps.hdr,
            dec_b,
            codecs,
            devices_json(&devices)
        );
        return;
    }

    println!("rivet capabilities\n");
    println!("Encode — AV1 (4:2:0):");
    if enc.is_empty() {
        println!("  (none) build with a `nvidia` / `amd` / `qsv` / `ffmpeg` feature");
    } else {
        println!("  backends   : {}", enc.join(", "));
        println!("  max depth  : {}-bit", caps.max_bit_depth);
        println!(
            "  HDR        : {}",
            if caps.hdr {
                "yes (PQ / HLG, BT.2020, 10-bit)"
            } else {
                "no"
            }
        );
    }

    println!("\nDecode — codec → backends:");
    if dec_backends.is_empty() {
        println!("  (none) build with a `nvidia` / `amd` / `qsv` / `ffmpeg` feature");
    } else {
        for d in &dec {
            let b = if d.backends.is_empty() {
                "—".to_string()
            } else {
                d.backends.join(", ")
            };
            println!("  {:<8} {}", d.codec, b);
        }
    }

    println!("\nDevices — {} detected:", devices.len());
    if devices.is_empty() {
        println!("  (none) CPU-only host — only the `ffmpeg` software path can run here");
    } else {
        for dv in &devices {
            print!(
                "  [{}] {} {}",
                dv.index,
                codec::gpu::manufacturer_label(dv.vendor),
                dv.name
            );
            if dv.vram_mib > 0 {
                print!(" ({} MiB)", dv.vram_mib);
            }
            // Authoritative AV1-encode verdict (the same probe the encode pool
            // uses to drop incapable cards) — so a pre-Ada NVIDIA shows "no".
            let av1 = if codec::encode::av1_encode_capable(dv) { "yes" } else { "no" };
            println!(" · AV1 encode: {av1}");
        }
    }
}

// ── streaming transcode (shared by `pipe` flags / `ipc` header) ──
//
// Both surfaces fill a `rivet::TranscodeSettings` — the one canonical knob set,
// the same type the CLI `transcode` and the HTTP API build — and run it here.
// `pipe` fills it from CLI flags; `ipc` from a `#rivet key=value …` header via
// `TranscodeSettings::parse_kv_line`.

/// Transcode `input` honoring `settings`: all-default settings take the fast
/// `transcode_bytes` path; any set field routes through the shared
/// `TranscodeSettings::into_spec` + the full `run_job` single-file engine.
/// Returns `(mp4_bytes, frames, audio_label)`.
fn stream_transcode(input: &[u8], settings: &TranscodeSettings) -> Result<(Vec<u8>, u64, String)> {
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
        if let rivet::RungArtifact::File(bytes) = r.artifact {
            return Ok((bytes, frames, audio));
        }
    }
    bail!("no single-file output produced")
}

// ── `rivet pipe` — stdin → stdout streaming ────────────────────────

fn pipe_cmd(settings: TranscodeSettings) -> Result<()> {
    use std::io::{Read, Write};
    let mut input = Vec::new();
    std::io::stdin()
        .lock()
        .read_to_end(&mut input)
        .context("reading media from stdin")?;
    if input.is_empty() {
        bail!("empty stdin — pipe media in, e.g. `cat in.mkv | rivet pipe > out.mp4`");
    }
    eprintln!("rivet pipe: {} bytes in, transcoding…", input.len());
    let (bytes, frames, audio) = stream_transcode(&input, &settings)?;
    let mut stdout = std::io::stdout().lock();
    stdout.write_all(&bytes).context("writing AV1/MP4 to stdout")?;
    stdout.flush().ok();
    eprintln!("rivet pipe: {frames} frames → {} bytes out ({audio})", bytes.len());
    Ok(())
}

// ── `rivet ipc` — Unix-domain-socket streaming server ──────────────

/// Split an optional `#rivet key=value …\n` settings header off the front of
/// the stream. Real container magic bytes never start with `#rivet`, so this is
/// unambiguous. Returns the parsed settings and the remaining media slice.
#[cfg(all(feature = "ipc", unix))]
fn split_ipc_settings(input: &[u8]) -> (Result<TranscodeSettings>, &[u8]) {
    const MAGIC: &[u8] = b"#rivet";
    if input.starts_with(MAGIC) {
        let nl = input.iter().position(|&b| b == b'\n').unwrap_or(input.len());
        let media_start = (nl + 1).min(input.len());
        let line = std::str::from_utf8(&input[MAGIC.len()..nl])
            .map(str::trim)
            .unwrap_or("");
        (TranscodeSettings::parse_kv_line(line), &input[media_start..])
    } else {
        (Ok(TranscodeSettings::default()), input)
    }
}

#[cfg(all(feature = "ipc", unix))]
fn ipc_cmd(socket: &Path) -> Result<()> {
    use std::io::{Read, Write};
    use std::os::unix::net::{UnixListener, UnixStream};

    // Drop a stale socket from a previous run (ignore "not found").
    let _ = std::fs::remove_file(socket);
    let listener = UnixListener::bind(socket)
        .with_context(|| format!("binding Unix socket {}", socket.display()))?;
    eprintln!(
        "rivet ipc: listening on {}\n           per connection: [optional `#rivet k=v …\\n` header] media → half-close → read AV1/MP4 back\n           e.g.  socat - UNIX-CONNECT:{} < in.mkv > out.mp4",
        socket.display(),
        socket.display(),
    );

    fn handle(mut stream: UnixStream) {
        let mut input = Vec::new();
        if let Err(e) = stream.read_to_end(&mut input) {
            eprintln!("rivet ipc: read error: {e}");
            return;
        }
        if input.is_empty() {
            return; // probe/keepalive connection
        }
        let (settings, media) = split_ipc_settings(&input);
        let settings = match settings {
            Ok(s) => s,
            Err(e) => {
                eprintln!("rivet ipc: bad settings header: {e:#}");
                return;
            }
        };
        eprintln!("rivet ipc: {} media bytes in", media.len());
        match stream_transcode(media, &settings) {
            Ok((bytes, frames, audio)) => {
                if let Err(e) = stream.write_all(&bytes) {
                    eprintln!("rivet ipc: write error: {e}");
                    return;
                }
                stream.flush().ok();
                let _ = stream.shutdown(std::net::Shutdown::Write);
                eprintln!("rivet ipc: {frames} frames → {} bytes out ({audio})", bytes.len());
            }
            Err(e) => eprintln!("rivet ipc: transcode error: {e:#}"),
        }
    }

    for stream in listener.incoming() {
        match stream {
            // One thread per connection; the process-wide GPU pool serializes
            // the actual GPU work, so concurrent clients just queue on it.
            Ok(s) => {
                std::thread::spawn(move || handle(s));
            }
            Err(e) => eprintln!("rivet ipc: accept error: {e}"),
        }
    }
    Ok(())
}

#[cfg(all(feature = "ipc", not(unix)))]
fn ipc_cmd(_socket: &Path) -> Result<()> {
    bail!(
        "`rivet ipc` (Unix-domain socket) is Unix-only. On Windows, use \
         `rivet pipe` (stdin/stdout) or `rivet serve` (HTTP)."
    )
}

// ── `rivet batch` — YAML/JSON manifest of conversions ──────────────

#[cfg(feature = "batch")]
fn batch_cmd(manifest_path: &Path, dry_run: bool, stop_on_error: bool) -> Result<()> {
    use rivet::manifest;

    let text = std::fs::read_to_string(manifest_path)
        .with_context(|| format!("reading manifest {}", manifest_path.display()))?;
    let mut m = manifest::parse_manifest(&text, manifest::Format::from_path(manifest_path))?;
    if stop_on_error {
        m.on_error = Some("stop".into());
    }
    let base = manifest_path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."))
        .to_path_buf();

    if dry_run {
        let planned = manifest::plan_manifest(&m, &base)?;
        eprintln!("batch dry-run: {} job(s) planned\n", planned.len());
        for (i, job) in planned.iter().enumerate() {
            let s = &job.spec;
            let mode = s.mode.as_deref().unwrap_or("single");
            let mut bits = vec![format!("mode={mode}")];
            if s.ladder == Some(true) {
                bits.push("ladder".into());
            }
            if let Some(r) = &s.rungs {
                bits.push(format!("rungs={}", r.join(",")));
            }
            if let Some(c) = s.crf {
                bits.push(format!("crf={c}"));
            }
            if let Some(c) = &s.color {
                bits.push(format!("color={c}"));
            }
            if let Some(o) = &s.output {
                bits.push(format!("output={o}"));
            }
            eprintln!("  [{}] {}  ({})", i + 1, job.input.display(), bits.join(" "));
        }
        eprintln!("\n(dry run — nothing converted)");
        return Ok(());
    }

    let report = manifest::run_manifest(&m, &base)?;

    println!(
        "\nbatch: {} ok, {} failed (of {})",
        report.ok_count(),
        report.failed_count(),
        report.outcomes.len()
    );
    for o in &report.outcomes {
        match &o.status {
            manifest::JobStatus::Ok => println!(
                "  ok    {} -> {}",
                o.input.display(),
                o.output.as_ref().map(|p| p.display().to_string()).unwrap_or_default()
            ),
            manifest::JobStatus::Failed(e) => {
                println!("  FAIL  {}: {}", o.input.display(), e)
            }
        }
    }
    if !report.all_ok() {
        bail!("{} job(s) failed", report.failed_count());
    }
    Ok(())
}
