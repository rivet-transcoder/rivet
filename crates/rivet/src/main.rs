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
//! rivet transcode input.mkv -o out.mp4 --crf 28 --audio opus --audio-bitrate 240k
//!
//! rivet probe input.mkv [--json]
//! ```
//!
//! Logging verbosity is controlled by `RUST_LOG` (e.g. `RUST_LOG=debug`).

use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};
use tracing_subscriber::EnvFilter;

mod commands;

// ── CLI value enums ────────────────────────────────────────────────
//
// These exist for clap: they give `--help` its value lists and completion its
// candidates, and they reject a misspelling before anything runs. They do
// **not** decide what a value means. Meaning lives in one place —
// `rivet::settings` — and every subcommand hands the enum's *name* to
// `TranscodeSettings::apply_kv` under the same key the IPC socket, the HTTP
// API and the batch manifest use, so `--audio opus`, `audio=opus` on the
// socket, `?audio=opus` on the API and `audio: opus` in a manifest are one
// code path. `settings_vocabulary_covers_every_cli_value` pins that every
// variant here parses there.

/// The name clap prints for a value-enum variant — the word the settings
/// vocabulary understands.
pub(crate) fn value_name<T: ValueEnum>(v: T) -> String {
    v.to_possible_value()
        .expect("every CLI value enum variant is a possible value")
        .get_name()
        .to_owned()
}

#[derive(Clone, Copy, ValueEnum)]
pub(crate) enum ModeArg {
    /// One self-contained MP4 per rung.
    Single,
    /// Segmented CMAF + HLS package.
    Hls,
}

#[derive(Clone, Copy, ValueEnum)]
pub(crate) enum AudioArg {
    /// Passthrough when possible, else transcode to Opus, else drop.
    Auto,
    /// Produce Opus audio.
    Opus,
    /// Drop audio (video only).
    Drop,
}

#[derive(Clone, Copy, ValueEnum)]
pub(crate) enum SubtitleArg {
    /// Carry text subtitles into the output MP4 as a tx3g track (default).
    Copy,
    /// Emit no subtitle track.
    Drop,
}

#[derive(Clone, Copy, ValueEnum)]
pub(crate) enum GpuFamilyArg {
    Nvidia,
    Amd,
    Intel,
}

#[derive(Clone, Copy, ValueEnum)]
pub(crate) enum ColorArg {
    /// Tonemap HDR sources to SDR BT.709 (default).
    Sdr,
    /// HDR10: BT.2020 + PQ, 10-bit (needs a 10-bit encoder: nvidia/amd/qsv/ffmpeg).
    Hdr10,
    /// HLG: BT.2020 + ARIB STD-B67, 10-bit (needs a 10-bit encoder: nvidia/amd/qsv/ffmpeg).
    Hlg,
    /// Preserve the source color/transfer/bit-depth verbatim.
    Passthrough,
}

#[derive(Clone, Copy, ValueEnum)]
pub(crate) enum PixelArg {
    /// Follow the color policy (default).
    Auto,
    #[value(name = "8bit")]
    Eight,
    #[value(name = "10bit")]
    Ten,
}

#[derive(Clone, Copy, ValueEnum)]
pub(crate) enum SeamArg {
    /// Chunk a single file across all GPUs for speed (default). NVENC chunks run
    /// VBR — possible mild quality steps at the chunk seams.
    Parallel,
    /// Chunk across GPUs but force constant-QP so seams are quality-flat. The QP
    /// is derived from the quality target, so quality still tracks it.
    Constqp,
    /// Legacy alias for `--encode single`: no seams at all is an encode plan
    /// (one encoder per rung), not a seam mode.
    Serial,
}

// ── CLI structs ────────────────────────────────────────────────────

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
        /// Constant rate factor (encoder-native, lower = better quality). Names
        /// the quantiser directly; when set, `--target` is not consulted.
        #[arg(long)]
        crf: Option<u8>,
        /// Perceptual quality target for every rung: `visually_lossless`, `high`,
        /// `standard` (default), `low`, or `vmaf=N` — a VMAF score to aim for,
        /// mapped to each backend's quantiser through the calibrated tables.
        #[arg(long, value_parser = rivet::settings::parse_quality_target)]
        target: Option<rivet::codec::encode::tuning::QualityTarget>,
        /// GOP length in frames for every rung (default: two seconds at the
        /// output frame rate). Single file: the keyframe cadence and, across
        /// GPUs, the chunk grid. HLS: the segment grid stays `--segment-seconds`;
        /// a shorter GOP adds keyframes inside each segment.
        #[arg(long, visible_alias = "keyframe-interval")]
        gop: Option<u32>,
        /// Audio handling.
        #[arg(long, value_enum, default_value = "auto")]
        audio: AudioArg,
        /// Target Opus bitrate for transcoded audio, e.g. `240k`. Omit to let
        /// the encoder derive it from the channel layout (64k mono, 96k stereo,
        /// 320k for 5.1). Ignored for passthrough tracks.
        #[arg(long = "audio-bitrate", value_name = "BPS")]
        audio_bitrate: Option<String>,
        /// Audio filter chain (ffmpeg-`-filter:a`-style), applied to decoded PCM
        /// before the Opus encoder, e.g.
        /// `channelmap=FL-FL|FR-FR|FC-FC|LFE-LFE|SL-BL|SR-BR:5.1`.
        #[arg(long = "audio-filter", value_name = "CHAIN")]
        audio_filter: Option<String>,
        /// Subtitle handling: `copy` (default) carries the source's text
        /// subtitles into the MP4 as a tx3g track; `drop` emits none. Bitmap
        /// subtitles (PGS / VobSub) are always dropped — tx3g can't hold them.
        #[arg(long, value_enum, default_value = "copy")]
        subtitles: SubtitleArg,
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
        /// The decode plan: `auto` (default — split the source into one range
        /// per capable card where the bitstream allows, each card decoding its
        /// own stretch), `whole` (one decoder for the whole source), `fastest`
        /// (benchmark the cards, one decoder on the quickest), `gpu:N` (one
        /// decoder pinned to card N — e.g. an iGPU while the dGPUs encode) or
        /// `ranges:N`. The source only splits where it safely can — an
        /// un-spliced H.264/H.265 input with keyframes on chunk boundaries;
        /// anything else decodes whole. Output is byte-identical either way.
        /// `--decode-gpu N` still works and means `gpu:N`.
        #[arg(long, visible_alias = "decode-gpu", default_value = "auto", value_parser = rivet::settings::parse_decode_plan)]
        decode: rivet::DecodePolicy,
        /// The encode plan: `all` (default — every capable card, each worker
        /// serving every rung and taking the next chunk of whichever is
        /// furthest behind), `per-rung` (every card, each pinned to its own
        /// rungs — one rung, one GPU when the ladder fits the pool), `single`
        /// (one card, one encoder per rung, serial — seam-free single-file),
        /// `gpu:N` (single, pinned to card N) or `family:nvidia|amd|intel`.
        /// `--gpu`, `--single-gpu` and `--gpu-family` are older spellings of the
        /// same choices and still work; this flag wins when both are given.
        #[arg(long, value_parser = rivet::settings::parse_encode_plan)]
        encode: Option<rivet::EncodePolicy>,
        /// Per-rung encoder knobs by ladder position: `recommended` (softer
        /// going down, one tile below 4K, three reference frames — the measured
        /// ladder policy), `off`, or the rule grammar, e.g.
        /// `qstep=2;top:q=-2;short<=2159:tiles=1x1;any:refs=3`. Default: none.
        #[arg(long)]
        encode_policy: Option<String>,
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
        /// `crop=1280:720,hflip` or `pad=1920:1080` / `rotate=90` / `grayscale`,
        /// `denoise=bilateral:0.5`, `nlmeans=s=1:p=7:r=3`, `hqdn3d=4:3:6:4.5`.
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
        /// Output: an MP4 file (`--mode single`) or a directory (`--mode hls`).
        #[arg(short, long)]
        output: PathBuf,
        /// Input clips in order: `PATH` or `PATH@START-END` (seconds).
        #[arg(required = true)]
        clips: Vec<String>,
        /// Output shape: `single` (one MP4) or `hls` (a CMAF/HLS package).
        #[arg(long, value_enum, default_value = "single")]
        mode: ModeArg,
        /// HLS target segment length (seconds); only used with `--mode hls`.
        #[arg(long, default_value_t = 4.0)]
        segment_seconds: f32,
        /// Output video codec: `av1` (default), `h264`, or `h265`.
        #[arg(long)]
        codec: Option<String>,
        /// Constant rate factor (quality; lower = better).
        #[arg(long)]
        crf: Option<u8>,
        /// Audio handling: `auto` (default), `opus`, `drop`.
        #[arg(long, value_enum, default_value = "auto")]
        audio: AudioArg,
        /// The decode plan: `auto` (default), `whole`, `fastest`, `gpu:N` or
        /// `ranges:N` — see `rivet transcode --help`. `--decode-gpu N` still
        /// works and means `gpu:N`.
        #[arg(long, visible_alias = "decode-gpu", default_value = "auto", value_parser = rivet::settings::parse_decode_plan)]
        decode: rivet::DecodePolicy,
        /// The encode plan: `all` (default), `per-rung`, `single`, `gpu:N` or
        /// `family:VENDOR` — see `rivet transcode --help`. A splice always takes
        /// the serial encode path, so here this chooses the card (`gpu:N`).
        #[arg(long, value_parser = rivet::settings::parse_encode_plan)]
        encode: Option<rivet::EncodePolicy>,
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
        /// Perceptual quality target: `visually_lossless`, `high`, `standard`,
        /// `low`, or `vmaf=N` — see `rivet transcode --help`.
        #[arg(long, value_parser = rivet::settings::parse_quality_target)]
        target: Option<rivet::codec::encode::tuning::QualityTarget>,
        /// GOP length in frames (default: two seconds).
        #[arg(long, visible_alias = "keyframe-interval")]
        gop: Option<u32>,
        /// Audio policy.
        #[arg(long, value_enum)]
        audio: Option<AudioArg>,
        /// Target Opus bitrate for transcoded audio, e.g. `240k`.
        #[arg(long = "audio-bitrate", value_name = "BPS")]
        audio_bitrate: Option<String>,
        /// Audio filter chain, e.g. `channelmap=FL-FL|FR-FR:stereo`.
        #[arg(long = "audio-filter", value_name = "CHAIN")]
        audio_filter: Option<String>,
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
        /// The decode plan: `auto` (default), `whole`, `fastest`, `gpu:N` or
        /// `ranges:N` — see `rivet transcode --help`.
        #[arg(long, visible_alias = "decode-gpu", default_value = "auto", value_parser = rivet::settings::parse_decode_plan)]
        decode: rivet::DecodePolicy,
        /// The encode plan: `all` (default), `per-rung`, `single`, `gpu:N` or
        /// `family:VENDOR` — see `rivet transcode --help`. Wins over `--gpu`.
        #[arg(long, value_parser = rivet::settings::parse_encode_plan)]
        encode: Option<rivet::EncodePolicy>,
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

// ── entry points ───────────────────────────────────────────────────

fn main() -> ExitCode {
    quiet_libva();
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

/// Stop libva narrating every driver open onto our stderr.
///
/// On an Intel host each encoder or decoder session opens a VA display, and
/// libva prints four `libva info:` lines every time it does — driver path, init
/// symbol, version, result. That's the driver's own logging, not ours, and it
/// interleaves with the progress lines badly enough to bury real warnings.
///
/// `LIBVA_MESSAGING_LEVEL=0` leaves errors visible and silences the chatter.
/// An explicit setting from the caller always wins, so `LIBVA_MESSAGING_LEVEL=2
/// rivet …` still gets the verbose form when debugging a driver problem.
fn quiet_libva() {
    if std::env::var_os("LIBVA_MESSAGING_LEVEL").is_none() {
        // SAFETY: single-threaded here — this runs as the first statement of
        // `main`, before the tracing subscriber or any runtime spawns a thread.
        unsafe { std::env::set_var("LIBVA_MESSAGING_LEVEL", "0") };
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
            target,
            gop,
            audio,
            audio_bitrate,
            audio_filter,
            subtitles,
            max_fps,
            gpu,
            single_gpu,
            gpu_family,
            decode,
            encode,
            encode_policy,
            color,
            pixel_format,
            seam_mode,
            filter,
            codec,
            trim_start,
            trim_end,
        } => commands::transcode::run(commands::transcode::TranscodeArgs {
            input,
            output,
            mode,
            rungs,
            ladder,
            max_short_side,
            segment_seconds,
            crf,
            target,
            gop,
            audio,
            audio_bitrate,
            audio_filter,
            subtitles,
            max_fps,
            gpu,
            single_gpu,
            gpu_family,
            decode,
            encode,
            encode_policy,
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
            mode,
            segment_seconds,
            codec,
            crf,
            audio,
            decode,
            encode,
        } => commands::splice::run(output, clips, mode, segment_seconds, codec, crf, audio, decode, encode),
        Command::Probe { input, json } => commands::probe::run(input, json),
        Command::Devices { json } => {
            commands::devices::run(json);
            Ok(())
        }
        Command::Capabilities { json } => {
            commands::capabilities::run(json);
            Ok(())
        }
        Command::Pipe {
            crf,
            target,
            gop,
            audio,
            audio_bitrate,
            audio_filter,
            color,
            bit_depth,
            max_fps,
            width,
            height,
            gpu,
            decode,
            encode,
            filter,
        } => commands::pipe::run(commands::pipe::PipeArgs {
            crf,
            target,
            gop,
            audio,
            audio_bitrate,
            audio_filter,
            color,
            bit_depth,
            max_fps,
            width,
            height,
            gpu,
            decode,
            encode,
            filter,
        }),
        #[cfg(feature = "ipc")]
        Command::Ipc { socket } => commands::ipc::run(&socket),
        #[cfg(feature = "batch")]
        Command::Batch {
            manifest,
            dry_run,
            stop_on_error,
        } => commands::batch::run(&manifest, dry_run, stop_on_error),
        #[cfg(feature = "server")]
        Command::Serve { addr } => commands::serve::run(addr),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rivet::TranscodeSettings;

    /// The clap enums list the words; `rivet::settings` decides what they
    /// mean. Every variant of every enum must therefore be a word the settings
    /// vocabulary accepts under the key the subcommands hand it to — otherwise
    /// a flag that clap happily accepts would fail (or, worse, mean something
    /// else) at the single point of interpretation.
    #[test]
    fn settings_vocabulary_covers_every_cli_value() {
        fn check<T: ValueEnum + Copy>(key: &str) {
            for v in T::value_variants() {
                let name = value_name(*v);
                let mut s = TranscodeSettings::default();
                s.apply_kv(key, &name)
                    .unwrap_or_else(|e| panic!("`--{key} {name}` is not in the settings vocabulary: {e:#}"));
            }
        }
        check::<ModeArg>("mode");
        check::<AudioArg>("audio");
        check::<SubtitleArg>("subtitles");
        check::<GpuFamilyArg>("gpu-family");
        check::<ColorArg>("color");
        check::<PixelArg>("bit-depth");
        check::<SeamArg>("seam");
    }

    #[test]
    fn the_legacy_serial_seam_is_the_single_encode_plan_whatever_the_order() {
        // `seam=serial` and `encode=...` may arrive in either order on any
        // surface; the settings layer resolves them the same way regardless.
        let mut a = TranscodeSettings::default();
        a.apply_kv("seam", "serial").unwrap();
        a.apply_kv("encode", "all").unwrap();
        let mut b = TranscodeSettings::default();
        b.apply_kv("encode", "all").unwrap();
        b.apply_kv("seam", "serial").unwrap();
        let sa = a.into_spec(1280, 720).unwrap();
        let sb = b.into_spec(1280, 720).unwrap();
        assert_eq!(sa.encode_policy, sb.encode_policy);
        // An explicit encode plan wins over the legacy spelling…
        assert_eq!(sa.encode_policy, rivet::EncodePolicy::AllGpus);
        // …and alone, the legacy spelling is `single`.
        let mut c = TranscodeSettings::default();
        c.apply_kv("seam", "serial").unwrap();
        assert_eq!(c.into_spec(1280, 720).unwrap().encode_policy, rivet::EncodePolicy::SingleGpu(None));
    }
}
