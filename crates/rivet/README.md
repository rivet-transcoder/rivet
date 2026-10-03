# rivet

[![crates.io](https://img.shields.io/crates/v/rivet-transcoder.svg?logo=rust)](https://crates.io/crates/rivet-transcoder)
[![Downloads](https://img.shields.io/crates/d/rivet-transcoder.svg)](https://crates.io/crates/rivet-transcoder)
[![docs.rs](https://img.shields.io/docsrs/rivet-transcoder.svg?logo=docsdotrs)](https://docs.rs/rivet-transcoder)
[![License](https://img.shields.io/badge/license-source--available-orange.svg)](LICENSE.md)

A modular, GPU-accelerated video transcoding **library** and **command-line
tool**, written in Rust. Install the CLI with `cargo install rivet-transcoder`
(the command is `rivet`), or add the library with `cargo add rivet-transcoder`.

`rivet` takes an arbitrary input file and transcodes it to **AV1, H.264, or
H.265** — as a single MP4, a multi-rendition ABR ladder, or a segmented
**CMAF/HLS** package. It also writes the audio alone (`.mp3`, `.flac`, `.m4a`,
`.ogg`)
and, with the `image` feature, still images (AVIF / WebP / JPEG / PNG, from
a picture or from a video).
The output is fully configurable: you choose the **output
mode**, the **codec**, the **quality**, the **container/muxer**, and the exact
**rungs**, and you get an **asynchronous progress callback** with a uniform
per-rung status struct. AV1 is the default (royalty-clean AV1 + Opus in MP4);
H.264/H.265 are there for legacy-player compatibility — see [Choosing the output
codec](#choosing-the-output-codec).

It is built from clean-room demuxers, muxers, and hardware-codec dispatch.
There is **no FFmpeg** in any build: no `ffmpeg-next`, no libav* linkage, no
FFmpeg libraries on the host, and no feature that adds them. Software AV1 is
this workspace's own [`av1`](https://github.com/rivet-transcoder/rivet/tree/HEAD/crates/av1) crate — the decoder always in, the
encoder as a fallback with `av1-sw-fallback` — and so are software H.264 /
H.265: this workspace's own [`h26x`](https://github.com/rivet-transcoder/rivet/tree/HEAD/crates/h26x) decoders (always in) and
encoders (`h26x-fallback`). ProRes, VP8, VP9, MPEG-1 / MPEG-2 and MPEG-4 Part 2
sources decode on any host too, through decoders this workspace wrote
clean-room from each format's specification (always in), and the still-image
codecs (PNG, JPEG, GIF, BMP, TIFF; AVIF through the AV1 crate) are the
workspace's own as well. See [No FFmpeg](#no-ffmpeg).

📖 **Detailed docs** live in [`docs/`](https://github.com/rivet-transcoder/rivet/tree/HEAD/docs). Start with
[Architecture](https://github.com/rivet-transcoder/rivet/blob/HEAD/docs/architecture.md) (the codebase map) and
[Design decisions](https://github.com/rivet-transcoder/rivet/blob/HEAD/docs/decisions.md) (the *why*); then
[Pipeline](https://github.com/rivet-transcoder/rivet/blob/HEAD/docs/pipeline.md) (data flow), the per-crate references
([codec decode](https://github.com/rivet-transcoder/rivet/blob/HEAD/docs/codec-decode.md) · [codec encode](https://github.com/rivet-transcoder/rivet/blob/HEAD/docs/codec-encode.md) ·
[container](https://github.com/rivet-transcoder/rivet/blob/HEAD/docs/container.md) · [engine](https://github.com/rivet-transcoder/rivet/blob/HEAD/docs/engine.md)), and the usage guides
([OutputSpec](https://github.com/rivet-transcoder/rivet/blob/HEAD/docs/output-spec.md) · [Batch manifest](https://github.com/rivet-transcoder/rivet/blob/HEAD/docs/batch.md) ·
[CLI](https://github.com/rivet-transcoder/rivet/blob/HEAD/docs/cli.md) · [HTTP API](https://github.com/rivet-transcoder/rivet/blob/HEAD/docs/api.md) · [Hooks](https://github.com/rivet-transcoder/rivet/blob/HEAD/docs/hooks.md) ·
[Lossless audio](https://github.com/rivet-transcoder/rivet/blob/HEAD/docs/lossless-audio.md)). The full index is
[docs/README.md](https://github.com/rivet-transcoder/rivet/blob/HEAD/docs/README.md). This README is the quick tour.

## Why "rivet"

**rivet is the transcoding *service* layer that FFmpeg leaves to you.** Calling
an encoder is the easy part; the rest — a job model, structured per-rendition
progress, cross-vendor GPU dispatch that fails fast instead of degrading
silently, a decode-once ABR ladder that scales across GPUs, and royalty-clean
defaults that actually play in a browser — is real engineering you would
otherwise rebuild for every project. rivet packages exactly that, three ways: a
**library** you embed, a **CLI** you run, and an **HTTP service** you call. The
name fits — a rivet fastens that orchestration into one reusable component.

**Why teams pick rivet — at a glance:**

- **A service, not a CLI to wrap.** A configurable job model, a uniform async
  per-rendition progress callback, and an optional HTTP API (`rivet serve`) — the
  orchestration you'd otherwise build around shell-outs and stderr scraping.
- **Royalty-clean by default.** AV1 + Opus in MP4 carries no patent-licensing
  obligations. H.264 / H.265 are first-class but **opt-in**, for legacy players —
  so the codecs that carry MPEG-LA / HEVC-pool royalties are a deliberate choice
  you make, not the default you stumble into.
- **A commercial-friendly license.** Source-available and **royalty-free for every
  use** — internal tooling, commercial products, and hosted "transcoder-as-a-service"
  deployments alike — **not GPL/LGPL**. No copyleft to reason about when you embed it
  (attribution is required for commercial use; see [License](#license)).
- **No FFmpeg, no toolchain hell.** Clean-room demuxers/muxers + hand-rolled
  `dlopen` GPU FFI mean no build pulls in **FFmpeg or LLVM**, builds
  on **Windows MSVC *and* Linux** identically, links the C runtime statically on
  Windows, and keeps your dependency + licensing story simple.
- **Cross-vendor GPU that fails loud.** Detects the GPUs and dispatches per vendor
  (NVENC / AMF / QSV); a host that can't encode the chosen codec **errors at
  startup** instead of silently dropping to a slow software path the way an
  `-hwaccel` misconfig does.
- **Near-linear ladder throughput.** Decode the source **once** — split across
  the cards at segment-aligned keyframes — fan frames out to every rung, and keep
  **every** GPU on whichever rung is furthest behind. A 5-rung ABR ladder decodes
  once (not five times), no card idles while any rung has work, and throughput
  scales close to linearly with GPU count.
- **Web-correct, automatically.** AV1 + Opus, faststart MP4 or segment-aligned
  CMAF/HLS, and HDR tonemapped down to 8-bit SDR BT.709 by policy — the per-source
  decisions that usually need a video engineer, shipped as defaults you can override.
- **Bounded memory at any size.** A streaming demuxer holds the input in a small,
  fixed working set regardless of file length, so transcoding a multi-hour source
  doesn't balloon RSS into gigabytes.
- **Your code inside the job.** [Hooks](https://github.com/rivet-transcoder/rivet/blob/HEAD/docs/hooks.md) run caller-supplied code
  at fixed points — the source bytes, the probe, decoded frames, encoder
  frames, stills, each output, the end — and can reject the job. A digest and a
  perceptual-fingerprint hook are built in; [`examples/yolo`](https://github.com/rivet-transcoder/rivet/tree/HEAD/examples/yolo)
  runs a YOLO detector on them.

The detail behind each, in narrative:

FFmpeg is the usual answer to "just transcode this", and a superb codec toolbox —
but it's a CLI and a C library, **not a service**. There's no job model, no
structured per-rendition progress, no HTTP surface: you shell out, scrape stderr,
and wire up the orchestration yourself. rivet ships that part — a configurable
job engine, a uniform async progress callback, and an optional HTTP API
(`rivet serve`) so another application can signal a transcode over the network
and poll it. (And nothing is hidden: the component crates — `codec`, `container`
— are re-exported, so you can drop down to a single muxer or encoder when the
engine's defaults aren't enough.)

**Hardware selection is the other half.** Getting GPU encode/decode right across
vendors with FFmpeg means hand-picking `-hwaccel` flags, per-vendor encoder
names, pixel/surface formats, and init options — and it quietly falls back to a
slow software path when any of that is wrong. rivet detects the GPUs, dispatches
to the right framework per vendor (NVDEC/NVENC, AMF, QSV, with a software
tier), leases them fairly across the ABR ladder, and **fails fast** instead of
degrading silently.

**And it's built to be fast at the ladder.** The source is decoded **once** and
the frames are fanned out to every rendition — a 5-rung ABR ladder decodes the
input one time, not five (the naïve one-process-per-rung approach decodes it N
times) — and on a multi-GPU host the decode itself is **split across the cards**
at keyframes that fall on segment boundaries, so no rung waits on a single
decoder. Encode work is segment-sized and served by **one worker per GPU that
takes the next chunk of whichever rung is furthest behind**: a card idles only
when the whole job is out of work, never because "its" rung is blocked while
another rung's chunks wait, and throughput scales close to linearly with GPU
count. Single-file output uses the same workers — chunk-encode the one
rendition across the GPUs and stitch the segments back together losslessly. A
per-rung codec invariant keeps cross-vendor chunks bit-compatible, so an NVENC +
QSV mix on the same rendition still decodes cleanly. Stitched chunks always play (each is an independent IDR-led GOP), and
`ChunkSeamMode` (CLI `--seam-mode`, API `seam`) controls quality across the
seams: `Parallel` (default, fastest) or `ParallelConstQp` (constant-QP,
seam-flat); no seams at all is an encode plan — `EncodePolicy::SingleGpu`, one
encoder per rung — see the [CLI reference](https://github.com/rivet-transcoder/rivet/blob/HEAD/docs/cli.md#chunk-seams---seam-mode).

> The full data flow — demux → decode-once pump → per-rung scale → multi-GPU
> lease engine → mux — is documented in
> **[docs/pipeline.md](https://github.com/rivet-transcoder/rivet/blob/HEAD/docs/pipeline.md)** (with a diagram and a code map).

**"Optimized for web" is a pile of decisions FFmpeg leaves to you.** rivet bakes
in defaults that just play in a browser (and lets you override them): AV1 (the
royalty-clean codec target) + Opus audio, faststart MP4 or segment-aligned
CMAF/HLS for ABR, and correct color — HDR tonemapped down to 8-bit SDR BT.709 by
policy, so a clip doesn't land eye-searingly bright or washed-out on a viewer's
screen. Picking those knobs correctly per source is exactly the expertise rivet
encodes so you don't have to.

## Usage

How to drive rivet — the quick start, the library API, the CLI, the HTTP server,
and how to pick the output codec. Each surface configures the same `OutputSpec`.

### Quick start

Library — one file in, one file out:

```rust
let outcome = rivet::transcode_file("input.mkv", "output.mp4")?;
println!("{} frames out", outcome.frames_processed);
```

CLI — same thing:

```sh
rivet transcode input.mkv -o output.mp4
```

The deeper knobs (ladders, HLS, progress, GPU selection) are in
[Library usage](#library-usage) and [CLI usage](#cli-usage) below.

### What you configure

A job is described by an [`OutputSpec`](https://github.com/rivet-transcoder/rivet/blob/HEAD/crates/rivet/src/spec/mod.rs):

| Dimension       | Type                         | Choices |
|-----------------|------------------------------|---------|
| **Output mode** | `OutputMode`                 | `SingleFile`, `Hls { segment_seconds }`, `AudioOnly` (the audio alone as an `.mp3`, a native `.flac`, an `.m4a` or an `.ogg`). Still images are a separate spec, [`rivet::image::ImageSpec`](https://github.com/rivet-transcoder/rivet/blob/HEAD/crates/rivet/src/image/mod.rs) |
| **Video codec** | `VideoCodecPolicy`           | `Av1` (default), `H264`, `H265`, `Vp9`, `Vp8`, `Mpeg2`, `Mpeg4`, or `ProRes(profile)` — see [Choosing the output codec](#choosing-the-output-codec) |
| **Audio**       | `AudioCodecPolicy`           | `Auto` (passthrough/transcode), `ForceOpus`, `ForceMp3`, `ForceAac`, `ForceHeAac`, `ForceHeAacV2`, `ForceVorbis`, `ForceAc3`, `ForceEac3`, `ForceDts`, `Flac`, `Alac` (lossless), `Drop` |
| **Channels**    | `AudioChannels`              | `Source` (default), `Mono`, `Stereo`, `Surround51`, `Surround71` — downmix, never upmix |
| **Container**   | `Container`                  | `Mp4`, `Mov`, `WebM`, `Cmaf`, `Mp3`, `Flac`, `M4a`, `Ogg` |
| **Muxer**       | `Muxer`                      | `Mp4File`, `WebmFile`, `CmafHls`, `Mp3File`, `FlacFile`, `M4aFile`, `OggFile` |
| **Rungs**       | `Vec<Rung>`                  | each `Rung` = a `width × height` **box** the source is fitted into + per-rung `Quality` (crf / speed / target / tier / keyframe interval) |
| **Fit**         | `Fit` / `Orientation` / `upscale` | `Contain` (default: keep the source's shape inside the box), `Cover` (fill and centre-crop), `Pad` (black bars to exactly the box), `Stretch`; boxes turn to a portrait source; no upscaling unless asked; a source with non-square pixels is fitted by its display shape — see [fitting](https://github.com/rivet-transcoder/rivet/blob/HEAD/docs/output-spec.md#fitting-the-source-into-a-rung) |
| **GPU policy**  | `EncodePolicy` / `DecodePolicy` | all GPUs / per-rung / single / pinned / vendor-family, and the decode plan (split across cards / whole / one card / fastest) — see [GPU scheduling](#gpu-scheduling-the-rung-benefit) |
| **Metadata**    | `container::metadata::Keep`  | none by default; `metadata_keep` names what identifying source metadata (location, capture time, device, descriptive) to carry into single-file, audio-only or image output |
| **Hooks**       | `rivet::hooks::Hooks`        | caller code at fixed points of the job (`with_hooks`) — see [Hooks](#hooks) |

Progress is reported through a [`ProgressSink`](https://github.com/rivet-transcoder/rivet/blob/HEAD/crates/rivet/src/progress.rs) as
a uniform [`RungProgress`](https://github.com/rivet-transcoder/rivet/blob/HEAD/crates/rivet/src/progress.rs) (status, percent,
frames, segments, bytes) per rung — wire it to a closure, a Tokio mpsc channel,
or your own implementation.

> **Measuring, not guessing:** [`bench/`](https://github.com/rivet-transcoder/rivet/blob/HEAD/bench/README.md) scores a ladder
> against its source with VMAF/SSIM (a reproducible corpus, a scorer that
> upscales each rung to source and scores past any fade, and one command from a
> clip plus any flags to a scored ladder). Every number in these docs came from
> it. `--target vmaf=93` aims a job at a VMAF score; the bench says whether it
> got there.

> **Complete reference: [Configuring a transcode — the `OutputSpec`
> guide](https://github.com/rivet-transcoder/rivet/blob/HEAD/docs/output-spec.md)** documents every builder method, enum, and field
> (rungs/quality, audio, color/bit-depth, [video filters](https://github.com/rivet-transcoder/rivet/blob/HEAD/docs/filters/README.md), GPU
> policy, chunk seams) with examples and how to run a job. The sections below are
> a tour of the highlights.

### Library usage

```toml
[dependencies]
# Published as `rivet-transcoder` (the crate name `rivet` was taken); the lib is
# `rivet`, so the rename keeps `use rivet::…` working as below.
rivet = { package = "rivet-transcoder", version = "0.2" }
```

(Or `cargo add rivet-transcoder` and `use rivet_transcoder as rivet;`.)

#### One file in, one file out

```rust
let outcome = rivet::transcode_file("input.mkv", "output.mp4")?;
println!("{} frames out", outcome.frames_processed);

let info = rivet::probe_file("input.mkv")?;
println!("{}x{} {}", info.width, info.height, info.video_codec);
```

#### A configurable job with progress

```rust
use std::sync::Arc;
use rivet::{OutputSpec, Rung, AudioCodecPolicy, run_job_blocking, fn_sink};
use rivet::progress::RungProgress;

let bytes = std::fs::read("input.mkv")?;

// A 3-rung HLS ladder, 4-second segments, audio auto-handled.
let spec = OutputSpec::hls(
    vec![Rung::new(1920, 1080), Rung::new(1280, 720), Rung::new(640, 360)],
    4.0,
)
.with_audio(AudioCodecPolicy::Auto);

// Uniform progress callback (status + percent + counters per rung).
let sink = Arc::new(fn_sink(|p: RungProgress| {
    println!("{:<6} {:?} {:>5.1}%  {} frames", p.label, p.status, p.percent, p.frames_done);
}));

// `output_dir` is the HLS asset root; `None` uses a temp dir.
let out = run_job_blocking(&bytes, &spec, Some("hls_out".as_ref()), sink)?;
println!("master playlist: {:?}", out.master_playlist);
```

For an **async** progress stream, use `channel_sink(tx)` with a
`tokio::sync::mpsc::Sender<RungProgress>` and `run_job(...).await` from inside a
runtime. Derive a sensible ladder from the source with
`rivet::standard_ladder(width, height, max_short_side)`.

#### Color, bit depth & frame rate

A fully-specified single-file job, picking the codec quality, frame-rate cap,
color/tonemap policy, and output bit depth per [the table below](#output-color--bit-depth):

```rust
use rivet::{OutputSpec, Rung, Quality, AudioCodecPolicy};
use rivet::spec::PerceptualTarget;

let spec = OutputSpec::single_file(vec![
    Rung::new(1920, 1080).with_quality(Quality::crf(28)),
    Rung::new(1280, 720).with_quality(Quality::target(PerceptualTarget::Standard)),
])
.with_audio(AudioCodecPolicy::Auto)
.with_max_frame_rate(30.0)   // cap output cadence at 30 fps
.web_sdr();                  // BT.709 8-bit SDR, tonemapping any HDR source down (default)

spec.validate()?; // rejects e.g. an HDR request on a build with no 10-bit encoder
```

The `.web_sdr()` line is a **color preset** — one call in place of
`.with_color(ColorPolicy::TonemapToSdr).with_bit_depth(BitDepth::EightBit)`.
There are exactly two color/depth knobs: `with_color` (the `ColorPolicy` bundles
the *gamut* and *transfer* — see [Output color & bit
depth](#output-color--bit-depth)) and `with_bit_depth`. To keep HDR instead of
tonemapping (needs a 10-bit AV1 encoder — `nvidia`, `amd`, or `qsv`):

```rust
let spec = OutputSpec::single_file(rungs).hdr10();   // BT.2020 + PQ, 10-bit — one call
// also: .hlg() · .passthrough() · or the low-level .with_color(..).with_bit_depth(..)
```

> **Jargon, briefly.** *Gamut* = which colors are representable: **BT.709** is
> the standard HD/SDR gamut (what most video uses), **BT.2020** is the wider one
> HDR uses. *Transfer* = the SDR-vs-HDR brightness curve: **PQ** (HDR10) and
> **HLG** (broadcast HDR). *Bit depth* is separate and the on-disk pixel format
> follows from it — **8-bit → `yuv420p`**, **10-bit → `yuv420p10le`** (always
> 4:2:0). HDR presets imply 10-bit, so you never set both. See
> [Output color & bit depth](#output-color--bit-depth).

#### Choosing GPUs

`encode_policy` controls how encode spreads across GPUs; `decode_policy` sets
the decode plan. See [GPU scheduling](#gpu-scheduling-the-rung-benefit)
for what each policy does.

```rust
use rivet::{OutputSpec, EncodePolicy, DecodePolicy, GpuFamily};

// All NVIDIA cards (ignore an integrated AMD/Intel GPU), but decode on GPU 0.
let spec = OutputSpec::single_file(rungs)
    .encode_policy(EncodePolicy::Family(GpuFamily::Nvidia))
    .decode_policy(DecodePolicy::SpecificGpu(0));

// Or pin everything to one GPU:
let spec = OutputSpec::single_file(rungs)
    .encode_policy(EncodePolicy::SingleGpu(Some(1)));
```

#### Escape hatch

Need finer control than the engine offers? Reach through the re-exported
component crates:

```rust
use rivet::codec::encode::{select_encoder, EncoderConfig};
use rivet::container::cmaf::CmafVideoMuxer;
```

### CLI usage

> **Full reference: [docs/cli.md](https://github.com/rivet-transcoder/rivet/blob/HEAD/docs/cli.md)** — every subcommand, flag, and
> environment variable. A taste:

```sh
# Single MP4 at the source resolution (output defaults to <input>.av1.mp4)
rivet transcode input.mkv -o output.mp4

# Explicit rungs → a directory of MP4s. Each size is a maximum: the source
# keeps its shape (a 4:3 or portrait video is not stretched) and is not upscaled.
rivet transcode input.mkv -o out_dir/ --rung 1920x1080 --rung 1280x720 --rung 640x360

# A vertical rung that centre-crops a landscape source to 9:16
rivet transcode input.mkv -o out_dir/ --rung 1920x1080 --rung 1080x1920:cover:fixed

# Auto-derived standard ABR ladder
rivet transcode input.mkv -o out_dir/ --ladder --max-short-side 1080

# CMAF/HLS package with 4-second segments
rivet transcode input.mkv -o hls_dir/ --mode hls --ladder --segment-seconds 4

# Quality + audio knobs
rivet transcode input.mkv -o out.mp4 --crf 28 --audio opus --audio-bitrate 240k

# 5.1 downmixed to stereo; MP3, AC-3, HE-AAC audio; the audio alone as an .mp3
rivet transcode input.mkv -o out.mp4 --audio-channels stereo
rivet transcode input.mkv -o out.mp4 --audio mp3
rivet transcode input.mkv -o out.mp4 --audio ac3 --audio-bitrate 448k
rivet transcode input.mkv -o out.mp4 --audio he-aac --audio-bitrate 48k
rivet transcode input.mkv -o out.mp3 --mode audio

# Vorbis in a WebM, Ogg Opus / Ogg Vorbis alone
rivet transcode input.mkv -o out.webm --codec vp9 --audio vorbis --audio-quality 6
rivet transcode input.mkv -o out.opus --mode audio --audio opus

# Lossless audio: FLAC beside the video, or the audio alone as a native .flac
rivet transcode input.mkv -o out.mp4 --audio flac
rivet transcode album.flac -o album.m4a --mode audio --audio alac

# Carry named source metadata (none is written by default); refuse to decode a codec
rivet transcode clip.mov -o out.mp4 --metadata-keep location:approximate,capture_time:date
rivet transcode input.mkv -o out.mp4 --audio-decode-deny aac,mp3

# Still images (feature `image`): sizes and formats of a photo, or stills from a video
rivet image photo.heic -o out --format avif,jpeg,png --rung 1920x1920 --rung 640x640
rivet image talk.mp4 -o stills --format jpeg --frames-count 12 --rung 320x320

# Splice — trim one input, or concatenate (with per-clip trims) several
rivet transcode input.mkv -o cut.mp4 --trim-start 2 --trim-end 7
rivet splice -o out.mp4 a.mp4@0-5 b.mp4@10-20 c.mp4

# Inspect without transcoding
rivet probe input.mkv [--json]

# Inspect the host + build
rivet devices [--json]        # detected GPUs: vendor, VRAM, live load, PCI BAR / Resizable BAR (Linux)
rivet capabilities [--json]   # what this build can encode/decode (alias: caps)

# Stream media in and out (no temp files)
cat input.mkv | rivet pipe > output.mp4                       # stdin → stdout (cross-platform)
cat input.mkv | rivet pipe --crf 28 --width 1280 --height 720 > out.mp4  # with settings
rivet ipc --socket /tmp/rivet.sock           # Unix-socket server; clients prefix a `#rivet k=v` header

# Convert many files from a YAML/JSON manifest (feature `batch`) — see docs/batch.md
rivet batch jobs.yaml --dry-run     # preview the plan
rivet batch jobs.yaml               # run it
```

GPU selection — the encode plan and the decode plan, one value each (they
mirror `EncodePolicy` / `DecodePolicy`, and the same words work as `encode=` /
`decode=` on the IPC socket, the HTTP API and the batch manifest):

```sh
rivet transcode in.mkv -o out.mp4 --encode all               # every card, ladder-scheduled (default)
rivet transcode in.mkv -o out.mp4 --encode per-rung          # every card, each pinned to its own rungs
rivet transcode in.mkv -o out.mp4 --encode single            # one card, one encoder per rung (seam-free MP4)
rivet transcode in.mkv -o out.mp4 --encode gpu:1             # …pinned to GPU 1   (`--gpu 1` still works)
rivet transcode in.mkv -o out.mp4 --encode family:nvidia     # all NVIDIA cards   (`--gpu-family nvidia` still works)
rivet transcode in.mkv -o out.mp4 --decode auto              # split the decode across the cards (default)
rivet transcode in.mkv -o out.mp4 --decode whole             # one decoder for the whole source
rivet transcode in.mkv -o out.mp4 --decode gpu:0             # one decoder on GPU 0 (`--decode-gpu 0` still works)
rivet transcode in.mkv -o out.mp4 --decode fastest           # benchmark, one decoder on the quickest card
```

Every setting left out has a word that states its default (`--gop 2s`,
`--max-fps source`, `--target standard`, `--video-speed standard`,
`--audio-bitrate standard`, …), so a
caller can name every setting and get the same job — see
[Stating the defaults](https://github.com/rivet-transcoder/rivet/blob/HEAD/docs/output-spec.md#stating-the-defaults).

Set `RUST_LOG=debug` for verbose logging. Force an encoder backend with
`TRANSCODE_ENCODER_BACKEND=nvenc|amf|qsv|h26x|av1` (`rav1e` is still accepted
for `av1`).

### HTTP API (`server` feature)

> **Full reference: [docs/api.md](https://github.com/rivet-transcoder/rivet/blob/HEAD/docs/api.md)** — endpoints, the output-spec
> query params, the job lifecycle, and the OpenAPI/Swagger/Redoc docs.

For a service deployment — where another application **signals** rivet to
transcode something — build with the `server` feature and run `rivet serve`. It
exposes the same engine over HTTP:

```sh
cargo build --release --features server,nvidia   # the API + an AV1 encoder
rivet serve --addr 0.0.0.0:8080
```

`POST /v1/transcode` takes either a **structured JSON body** — point at a
server-side input/output **file path** (or inline base64), with a structured
`spec` — or a **streamed binary body** with the spec in query params (so
streaming the media is optional):

```sh
curl -X POST http://localhost:8080/v1/transcode -H 'Content-Type: application/json' \
  -d '{"input":{"path":"/data/in.mkv"},"output":{"path":"/data/out.mp4"},
       "spec":{"rungs":["1280x720"],"crf":28},"sync":true}'
```

Interactive docs ship with it: **`/swagger`** (Swagger UI), **`/redoc`** (Redoc),
and the raw **`/openapi.json`** (OpenAPI 3.0); `/` links to all three.

A server started with hooks (`rivet::server::serve_with_hooks`) lists them at
`GET /v1/hooks`; a request opts into optional ones with `?hooks=a,b` or
`"hooks": [...]`, the job's hook report is in `GET /v1/jobs/{id}`, and a job a
hook rejects ends with `status: "rejected"` (`422` for `?sync=true`).

### Hooks

> **Full reference: [docs/hooks.md](https://github.com/rivet-transcoder/rivet/blob/HEAD/docs/hooks.md)**, with a
> [cookbook](https://github.com/rivet-transcoder/rivet/blob/HEAD/docs/hooks-cookbook.md) of sixteen recipes and a
> [YOLO object-detection guide](https://github.com/rivet-transcoder/rivet/blob/HEAD/docs/hooks-yolo.md).

Hooks are code you supply that rivet runs at fixed points of every job. Each
kind has its own trait and gets only what exists at its point: the source
bytes (`SourceHook`), the probe (`ProbeHook`), decoded frames
(`DecodedFrameHook`), the frames the encoders receive (`EncoderFrameHook`),
stills in an image job (`StillHook`), each output (`ArtifactHook`), and the
end of the job (`CompletedHook` / `FailedHook`). A hook returns a verdict —
carry on, or reject the job — and values to record in the job's report. It
can block the job or run in the background, and fail open or closed.

```rust
use rivet::hooks::*;

let hooks = Hooks::new()
    .source("source-digest", SourceDigest::new(&[DigestAlgorithm::Sha256]))
    .decoded_frames(
        "fingerprint",
        PerceptualFingerprint::new(&[PerceptualAlgorithm::PHash]).sampling(FrameSampling::every_seconds(1.0)),
    );
let spec = OutputSpec::single_file(rungs).with_hooks(hooks);
```

[`rivet::hooks::frame`](https://github.com/rivet-transcoder/rivet/blob/HEAD/crates/rivet/src/hooks/frame.rs) has the pixel
helpers a model needs (RGB, resized, letterboxed, planar `f32`).
[`examples/yolo`](https://github.com/rivet-transcoder/rivet/tree/HEAD/examples/yolo) is a separate crate that runs a YOLO detector
as a decoded-frame and still hook through ONNX Runtime, on the CPU or with
CUDA, DirectML or OpenVINO; ONNX Runtime never becomes a dependency of rivet.

### Choosing the output codec

The output codec is a first-class, selectable dimension. In Rust you pick it with
a [`VideoCodecPolicy`](https://github.com/rivet-transcoder/rivet/blob/HEAD/crates/rivet/src/spec/policy.rs) — the video analogue of
[`AudioCodecPolicy`](https://github.com/rivet-transcoder/rivet/blob/HEAD/crates/rivet/src/spec/policy.rs) — which is `Av1` (default), `H264`,
or `H265` — or `Vp9`, `Vp8`, `Mpeg2`, `Mpeg4`, `ProRes(profile)`. **AV1** is the recommended target (AV1 + Opus in MP4 = zero royalty
exposure); **H.264 / H.265** are there for legacy-player compatibility and carry
the patent-licensing obligations AV1 was chosen to avoid. The encode tier is
GPU-accelerated (NVENC / AMF / QSV). All three work for single-file MP4 **and**
CMAF/HLS (the muxer emits `av01`/`avc1`/`hvc1` sample entries — `avc3`/`hev1`
only where the parameter sets change mid-stream — and the right `CODECS=`
strings); AV1 stays the cross-vendor default.

**Every codec rivet decodes, it can encode.** VP9, VP8, MPEG-2, MPEG-4 Part 2
and ProRes are written by this workspace's own clean-room encoders, in
software, in every build (no feature, no GPU), each into the files that carry
it:

| Codec | Single file (default first) | HLS | Encode |
|---|---|---|---|
| VP9 | WebM, MP4 (`vp09`) | yes (`vp09`) | profile 0, 8-bit 4:2:0, fixed quantiser |
| VP8 | WebM, MP4 (`vp08`) | no | 8-bit 4:2:0, fixed quantiser |
| MPEG-2 | MP4 (`mp4v`), QuickTime | no | Main Profile, 8-bit 4:2:0, I/P/B; quantiser or average bitrate |
| MPEG-4 Part 2 | MP4 (`mp4v`), QuickTime | no | Simple (Advanced Simple with B-VOPs), 8-bit 4:2:0; quantiser or average bitrate |
| ProRes | QuickTime (`.mov`) only | no | Proxy / LT / 422 / HQ / 4444 / 4444 XQ, intra-only, 8- or 10-bit, HDR-tagged |

`--codec vp9|vp8|mpeg2|mpeg4|prores[-proxy|-lt|-422|-hq|-4444|-4444xq]`,
`--container mp4|mov|webm`, `--prores-profile`; the same keys everywhere.
What each can't do (10-bit VP9, ProRes in HLS, a bitrate for VP9, …) is
refused by name before anything is decoded. Every output is verified by
reading it back with rivet's own demuxers and decoders (frame count,
timestamps, PSNR against the source).

You pick the codec the same way in every surface — codecs are the strings `av1`
/ `h264` / `h265` / `vp9` / `vp8` / `mpeg2` / `mpeg4` / `prores` (aliases `avc`/`hevc`/`x264`/`x265`/`av01`/`vp09`/`xvid`/`apch`/… accepted). Omit it
and you get AV1.

```rust
// Rust — the VideoCodecPolicy, alongside the AudioCodecPolicy
use rivet::{OutputSpec, Rung, VideoCodecPolicy, AudioCodecPolicy};

let spec = OutputSpec::single_file(vec![Rung::new(1280, 720)])
    .with_video_codec(VideoCodecPolicy::H265)   // av1 (default) · h264 · h265
    .with_audio(AudioCodecPolicy::Auto);        // passthrough / transcode-to-Opus / drop
```

```sh
# CLI
rivet transcode in.mp4 -o out.mp4 --codec h265
rivet transcode in.mp4 --codec prores-hq          # -> in.prores.mov
rivet transcode in.mp4 --codec vp9 -o out.webm

# Batch manifest (YAML) — `rivet batch jobs.yaml`
#   defaults: { codec: h264 }
#   jobs: [ { input: a.mkv, codec: h265 }, { input: b.mp4 } ]   # b → av1

# HTTP API — query param or JSON body
curl --data-binary @in.mp4 "http://localhost:8080/v1/transcode?mode=hls&codec=h265"
curl -X POST -H 'content-type: application/json' \
     -d '{"input":{"path":"in.mp4"},"spec":{"mode":"hls","codec":"h265"}}' \
     http://localhost:8080/v1/transcode

# Settings DSL / IPC header (the `#rivet k=v …` line) — key=value
#   #rivet codec=h265 mode=hls
```

See [OutputSpec](https://github.com/rivet-transcoder/rivet/blob/HEAD/docs/output-spec.md), [CLI](https://github.com/rivet-transcoder/rivet/blob/HEAD/docs/cli.md),
[Batch](https://github.com/rivet-transcoder/rivet/blob/HEAD/docs/batch.md), and [HTTP API](https://github.com/rivet-transcoder/rivet/blob/HEAD/docs/api.md) for the full field set.

## Features

What rivet does and what it supports — the multi-GPU scheduler, and the
compatibility matrix of codecs, colors, containers, and output modes.

### GPU scheduling (the rung benefit)

Both HLS and single-file jobs run on the multi-GPU orchestrator
([`multigpu`](https://github.com/rivet-transcoder/rivet/tree/HEAD/crates/rivet/src/multigpu)) that makes the ladder cheap:

- **Decode once, split across the cards.** The whole ladder is fed by one
  decode — a 5-rung ladder decodes the source one time, not five — and on a
  multi-GPU host the source is cut into ranges at keyframes that fall on
  segment boundaries, one decode pump per card, so the cards decode different
  stretches of the source at the same time. Segment numbering stays continuous
  across the join. Sources that cannot be split safely decode whole.
- **Lease pool.** A process-wide [`GpuPool`](https://github.com/rivet-transcoder/rivet/blob/HEAD/crates/rivet/src/gpu_pool.rs)
  hands out one encoder lease per GPU (concurrent NVENC sessions on one context
  deadlock — this is the load-bearing invariant), so work runs in parallel
  *across* GPUs.
- **Ladder workers.** One worker per GPU holds its lease for the whole
  job and takes the next segment-sized chunk of whichever rung is furthest
  behind. A card idles only when the job is out of work — never because its
  rung is blocked while another rung's chunks wait — and a ladder longer than
  the GPU count still costs one decode. Single-file jobs run on the same
  ladder core: a chunk is several GOPs, encoded in memory, and each rung's
  chunks are stitched in order (chunk-and-stitch). `EncodePolicy::PerRung`
  pins each card to its own rungs instead.
- **Cross-vendor safety.** Cards of different vendors (NVENC + QSV) serve the
  same rendition; a per-rung codec invariant guarantees every segment shares
  the `av1C` / `avcC` / `hvcC` contract, and a card that mismatches a rung hands
  the chunk back and leaves that rung to the others without aborting the job.
- **Capability-aware pool.** Cards that can't encode AV1 (e.g. a pre-Ada NVIDIA
  that decodes via NVDEC but has no AV1 encode silicon) are dropped from the
  *encode* pool but kept for the *decode* pump. So a heterogeneous host —
  say a pre-Ada NVIDIA + an Arc — decodes on the NVIDIA and encodes on the Arc
  automatically, instead of aborting when a chunk lands on the card that can't
  encode.

For **single-file** output, each rung is chunked at GOP boundaries and the
chunks are encoded across the GPUs, then stitched — in segment order, in memory,
no disk round-trip — into one MP4 per rung. Because the encoder runs
constant-quality (CQP/CRF), independent chunks have no rate-control
discontinuity at the seams; each chunk just starts with an IDR. On a single-GPU
host (or when the frame count is unknown, or the job is trimmed) it uses the
serial decode-once path instead, with no chunk overhead. Either way, a host
with no encoder for the chosen codec fails fast with a clear error.

#### Encode policy

`OutputSpec::encode_policy(..)` selects how encode work spreads across GPUs (set
it from the library or the CLI — see above):

| Policy | Single-file | HLS |
|--------|-------------|-----|
| `EncodePolicy::AllGpus` *(default)* | chunk across all GPUs, stitch | ladder across all GPUs |
| `EncodePolicy::PerRung` | every GPU, each pinned to its own rungs | every GPU, each pinned to its own rungs |
| `EncodePolicy::SingleGpu(None)` | runs on the first GPU | runs on the first GPU |
| `EncodePolicy::SingleGpu(Some(i))` | runs on GPU `i` | runs on GPU `i` |
| `EncodePolicy::Family(GpuFamily::Nvidia)` | chunk across that vendor's GPUs | ladder across that vendor's GPUs |

For `SingleGpu` both modes run the same way — sequentially on one GPU — they just
reach it differently: single-file takes a lean serial path (no GOP chunking,
nothing to parallelize on one GPU), while HLS always runs the lease-pool
orchestrator (one lease) because its output is inherently segmented. For
`AllGpus` / `Family` they genuinely differ: single-file chunks-and-stitches,
HLS ladders-and-segments across the selected GPUs.

The **decode pump follows the policy**: it is pinned to a GPU from the policy's
selected set (round-robin over those indices for per-rung pumps), so a `Family`
/ `SingleGpu` constraint governs *decode* too, not just encode. Override it
independently with `OutputSpec::decode_policy(DecodePolicy::SpecificGpu(i))` —
e.g. decode on an integrated GPU while the discrete GPUs encode. The other
decode plans are `Auto` (default: split the source into ranges across the
cards where it can), `Whole`, `FastestGpu` and `Ranges(n)`.

### Compatibility matrix

#### Input — video decode

GPU decode is feature-gated — each vendor's tier is an opt-in cargo feature.
Software decode is always in, for every codec in the table: H.264 / HEVC
(this workspace's `h26x`) and AV1, VP8, VP9, MPEG-1 / MPEG-2, MPEG-4 Part 2 and
ProRes (this workspace's `av1`, `vp8`, `vp9`, `mpeg2`, `mpeg4` and `prores`,
one decoder per format, each written clean-room from its specification). All
decoders plug into the shared decode pump (`create_decoder` → `push_sample` →
`decode_next`), tried in the order NVDEC → AMF → QSV → rivet's own software
decoders (`h26x`, `av1`, `vp8`, `vp9`, `mpeg2`, `mpeg4`, `prores`; each takes
only its own codec) → openh264.

`openh264-fallback` adds openh264 for H.264 as a last resort, behind `h26x`.
Every codec in the table decodes on a host with no GPU. See [No
FFmpeg](#no-ffmpeg).

| Codec          | NVDEC `nvidia` | AMF `amd` † | QSV `qsv` | rivet's own (always) | openh264 `openh264-fallback` |
|----------------|:--------------:|:----------:|:----------:|:--------------------:|:----------------------------:|
| H.264 / AVC    | ✅             | ✅         | ✅         | ✅ `h26x`            | ✅ |
| HEVC / H.265   | ✅             | ✅         | ✅         | ✅ `h26x`            | —  |
| VP8            | ✅             | —          | —          | ✅ `vp8`             | —  |
| VP9            | ✅             | ✅         | ✅         | ✅ `vp9`             | —  |
| AV1            | ✅             | ✅         | ✅         | ✅ `av1`             | —  |
| MPEG-2         | ✅             | —          | —          | ✅ `mpeg2`           | —  |
| MPEG-1         | —              | —          | —          | ✅ `mpeg2`           | —  |
| MPEG-4 Part 2  | ✅             | —          | —          | ✅ `mpeg4`           | —  |
| ProRes         | —              | —          | —          | ✅ `prores`          | —  |
- **NVDEC `nvidia`** — a single, in-repo **hand-rolled CUVID FFI** decoder
  (`decode/nvdec.rs`, dlopen, no external crate). One path for everything NVDEC
  does: H.264/HEVC/AV1/VP8/VP9, MPEG-2, MPEG-4 Part 2, and **10-bit P016**.
  Builds on **both Windows MSVC and Linux**.
- **QSV `qsv`** (`decode/qsv_dec.rs`) — hand-rolled oneVPL FFI (our own SDK-mirror
  code, no external crate). **Hardware-verified on 3× Intel Arc** (H.264 / HEVC /
  AV1 / VP9, including 10-bit P010 via the oneVPL 2.x internal-allocation +
  `FrameInterface::Map` path). Builds on Windows + Linux.
- **AMF `amd`** (`decode/amf_dec.rs`) — hand-rolled AMF decode FFI. † **Verified-
  by-review only** — no AMD card on the dev box yet; tracked in
  [TODO.md](https://github.com/rivet-transcoder/rivet/blob/HEAD/TODO.md).
- **rivet's own** (`decode/{h26x,av1,prores,vp8,vp9,mpeg2,mpeg4}_sw.rs`,
  always compiled, no feature) — adapters onto this workspace's codec
  submodules (see [Crates](#crates)). `h26x` gives 4:2:0 / 4:2:2 / 4:4:4 up to
  12 bits; AV1 8 / 10 / 12-bit 4:2:0 / 4:2:2 / 4:4:4 (monochrome as 4:2:0 with
  neutral chroma, film grain applied) — single-threaded, about 6 megapixels a
  second on streams that use the whole toolbox (some 7 fps at 720p, 3 fps at
  1080p; rivet's own encoder's output decodes at about 23), on a worker
  thread that runs a few frames ahead of the rest of the pipeline; ProRes
  4:2:2 at 10 bits and 4:4:4 at 12 (an alpha plane is dropped); VP8 8-bit
  4:2:0; VP9 8 / 10 / 12-bit 4:2:0 / 4:2:2 / 4:4:4 (4:4:0 and RGB-coded streams
  are refused); MPEG-1 / MPEG-2 8-bit 4:2:0 / 4:2:2; MPEG-4 Part 2 8-bit 4:2:0
  (Simple and Advanced Simple Profile and the H.263 short header; reversible
  VLCs are refused). The same crates' encoders are rivet's VP8, VP9, MPEG-2,
  MPEG-4 and ProRes output and its software AV1 encoder (below).

What happens to a 10-bit / HDR source is the **`ColorPolicy`'s** call, not a
fixed rule (the decode pump never tonemaps on its own): the default
`TonemapToSdr` maps HDR → 8-bit SDR BT.709 for maximum web compatibility, while
`Hdr10` / `Hlg` / `Passthrough` keep it **10-bit HDR** through to a 10-bit
encoder (NVENC / AMF / QSV) — see [Output color & bit
depth](#output-color--bit-depth). Decoding 10-bit needs a 10-bit-preserving
decoder: **NVIDIA** NVDEC decodes 10-bit **P016** natively and **Intel** QSV
decodes 10-bit **P010** (both carry 10-bit HEVC Main10 / HDR through). The
software tiers keep depth too: `h26x` decodes HEVC Main 10 / Main 12, `vp9`
decodes VP9 profiles 2 and 3 at 10 and 12 bits, ProRes comes out at 10 or 12
bits, and `av1` decodes AV1 at 8, 10 and 12 bits (4:2:0, 4:2:2, 4:4:4).

#### Output — video encode (by vendor)

rivet encodes **AV1** (default, royalty-clean), **H.264**, or **H.265**, 4:2:0 —
pick the codec per [Choosing the output codec](#choosing-the-output-codec) —
and **VP9, VP8, MPEG-2, MPEG-4 Part 2 and ProRes** in software with this
workspace's own encoders, in every build (the last table below). One
table per vendor: rows are the output codecs, columns are the output pixel
format. ✅ = hardware-validated · ⏳ = follow-up (the backend rejects the codec
with a clear error rather than silently emitting AV1). AV1 carries 10-bit (pair
with a HDR `ColorPolicy` for HDR10/HLG; on its own, higher-precision SDR).
**H.265 also encodes 10-bit (Main 10)** on NVENC, AMF and QSV; **H.264 is
8-bit only in hardware** — there is no Hi10P profile on NVENC, AMF or QSV, so a
10-bit H.264 request is capability-rejected there rather than down-converted
(the software `h26x` encoder does 10-bit H.264).

**NVENC — NVIDIA (`nvidia`)**

| Codec | 8-bit 4:2:0 | 10-bit 4:2:0 |
|-------|:-----------:|:------------:|
| AV1   | ✅ (Ada+)   | ✅ (`Yuv420_10bit`, Ada+) |
| H.264 | ✅ (Kepler+, RTX 3090-validated) | ❌ (no NVENC Hi10P silicon) |
| H.265 | ✅ (Maxwell+, RTX 3090-validated) | ✅ (Main 10, RTX 3090-validated) |

**AMF — AMD (`amd`)**

| Codec | 8-bit 4:2:0 | 10-bit 4:2:0 |
|-------|:-----------:|:------------:|
| AV1   | ⚠ by-review (RDNA3+) | ⚠ by-review (`P010`, RDNA3+) |
| H.264 | ✅ (`VCE_AVC`, Ryzen 9 9950X iGPU-validated) | ❌ (no AMF Hi10P profile) |
| H.265 | ✅ (`HW_HEVC`, iGPU-validated) | ✅ (Main 10, iGPU-validated) |

**QSV — Intel Arc / Meteor Lake+ (`qsv`)**

| Codec | 8-bit 4:2:0 | 10-bit 4:2:0 |
|-------|:-----------:|:------------:|
| AV1   | ✅          | ✅ (P010) |
| H.264 | ✅ (Arc-validated) | ❌ (no `AVC High 10` in oneVPL) |
| H.265 | ✅ (Arc-validated) | ✅ (Main 10, Arc-validated) |

**Software (`av1-sw-fallback` for AV1, `h26x-fallback` for H.264 / H.265)**

| Codec | 8-bit 4:2:0 | 10-bit 4:2:0 |
|-------|:-----------:|:------------:|
| AV1   | ✅ (`av1`, in-tree; profile 0, up to 4096 wide) | ✅ (SDR only — the encoder writes no colour description, so HDR10 / HLG AV1 needs a GPU) |
| H.264 | ✅ (h26x, in-tree; SELF + cross-checked against the JM reference decoder) | ✅ (High 10, h26x — the only 10-bit H.264 encoder here) |
| H.265 | ✅ (h26x, in-tree; SELF + cross-checked against the HM reference decoder) | ✅ (Main 10 / 12-bit, h26x; cross-checked at 10 and 12 bits; HDR10 / HLG signalled in the SPS VUI plus the HDR10 static-metadata SEIs, read back by MediaInfo and HM) |

**rivet's own (every build, no feature) — VP9, VP8, MPEG-2, MPEG-4 Part 2, ProRes**

| Codec | 8-bit 4:2:0 | 10-bit | Files |
|-------|:-----------:|:------:|-------|
| VP9   | ✅ (profile 0) | ✅ (profile 2, SDR) | WebM, MP4, HLS |
| VP8   | ✅ | — | WebM, MP4 |
| MPEG-2 | ✅ (Main Profile, I/P/B) | — | MP4, QuickTime |
| MPEG-4 Part 2 | ✅ (SP / ASP) | — | MP4, QuickTime |
| ProRes | ✅ (upsampled to 4:2:2 / 4:4:4) | ✅ (HDR-tagged) | QuickTime |

VP9 and software AV1 take an average-bitrate rung as well as a quality target
(one-pass rate control; `rate=cbr` and a coded picture buffer are refused by
name), and `--video-speed draft|standard|archive` trades their speed for
compression.

GPU-first — a host with no encode silicon for the chosen codec and no software
fallback fails fast at encoder construction (the five codecs above have no
GPU path here, so their own encoder is the encoder in every build). 4:2:2 / 4:4:4 and 12-bit are not
produced. All hardware encoders are hand-rolled `dlopen` FFI in-tree (NVENC, AMF
`P010`, QSV oneVPL) and build on Windows + Linux. H.264/H.265 emit **Annex-B**,
which the muxer repackages to length-prefixed `avc1`/`hvc1` samples
(single-file MP4 **and** CMAF/HLS) — see [codec encode](https://github.com/rivet-transcoder/rivet/blob/HEAD/docs/codec-encode.md).

#### Output color & bit depth

Two orthogonal axes: **color** (`with_color(ColorPolicy)` — gamut + SDR/HDR
transfer) and **bit depth** (`with_bit_depth(BitDepth)` — bits per sample). Most
callers don't touch them directly — the **presets** bundle both:
`.web_sdr()` (default), `.hdr10()`, `.hlg()`, `.passthrough()`. The decode pump
tonemaps **only** when the policy says so (it never decides on its own).
`validate()` rejects any combination this build can't actually produce:

| `ColorPolicy`  | Tonemap | Output signaling          | Bit depth | Needs |
|----------------|:-------:|---------------------------|:---------:|-------|
| `TonemapToSdr` *(default)* | HDR→SDR | BT.709 SDR             | 8-bit     | any encoder |
| `Passthrough`  | no      | source color verbatim     | source    | 10-bit encoder if source is 10-bit |
| `Hdr10`        | no      | BT.2020 + PQ (ST 2084)    | 10-bit    | a 10-bit encoder (below) |
| `Hlg`          | no      | BT.2020 + ARIB STD-B67    | 10-bit    | a 10-bit encoder (below) |

`BitDepth` is `Auto` (follow the color policy — the usual choice), `EightBit`
(`yuv420p`), or `TenBit` (`yuv420p10le`). 10-bit / HDR output needs a 10-bit
encoder **for the output codec**: AV1 on `nvidia`, `amd`, or `qsv` (per the
per-vendor tables above; the software AV1 tier, `av1-sw-fallback`, is 10-bit
SDR only, never HDR), H.265 on those or `h26x-fallback`, H.264 on
`h26x-fallback` only; VP9 is 10-bit SDR (profile 2) in every build. 10-bit AV1 is the
web-safe **Main** profile (4:2:0), HDR-tagged in the container via the
`colr`/`mdcv`/`clli` atoms, which browsers decode and tonemap. A spec this build
cannot encode for its codec fails `validate()` with an error naming the feature
that would serve it; the per-codec capability is queryable at runtime via
`rivet::spec::CodecOutputCaps::of_this_build(codec)` (or `rivet capabilities`).

For **web compatibility** keep the default — `.web_sdr()` (i.e. `TonemapToSdr` +
`Auto`) yields 8-bit SDR BT.709 AV1, which every browser and device that
supports AV1 plays.

#### Containers

| Container             | Demux (in) | Mux (out) |
|-----------------------|:----------:|:---------:|
| MP4 / MOV             | ✅         | ✅ (single-file + CMAF) |
| MKV / WebM            | ✅         | ✅ (WebM: VP8 / VP9 + Opus or Vorbis) |
| MPEG-TS               | ✅         | — |
| AVI (+OpenDML >1 GiB) | ✅         | — |
| CMAF / HLS            | —          | ✅ (segments + master/media playlists) |
| MP3 (`.mp3` / `.mp2`) | ✅ (audio only) | ✅ (`.mp3`, audio-only output) |
| FLAC (`.flac`)        | ✅ (audio only) | ✅ (audio-only output) |
| M4A                   | ✅ (as MP4) | ✅ (audio-only output) |
| Ogg (`.ogg` / `.opus`) | ✅ (Opus, Vorbis; audio only) | ✅ (Opus, Vorbis; audio-only output) |

Still images (JPEG, PNG, WebP, AVIF, GIF, TIFF, BMP, HEIC in; AVIF, WebP,
JPEG, PNG out) are the `image` feature's, every codec the workspace's own — see
[output-spec.md §11](https://github.com/rivet-transcoder/rivet/blob/HEAD/docs/output-spec.md#11-still-images--modeimage).

#### Audio

Every audio codec, both ways, is this workspace's own (pure Rust, written
from the standards, each in its own repository as a submodule): no libopus,
no LAME, no minimp3, no lewton, no FFmpeg.

| Codec  | Passthrough | Decoded | Encoded (`--audio …`) |
|--------|:-----------:|:-------:|:--------:|
| AAC-LC | ✅          | ✅ (`crates/aac`) | ✅ `aac` |
| HE-AAC / HE-AAC v2 | ✅ | ✅ (full rate: SBR, PS) | ✅ `he-aac`, `he-aacv2` |
| Opus   | ✅          | ✅ (`crates/opus`, stereo and surround) | ✅ `opus` (1–8 ch) |
| AC-3   | ✅          | ✅ (`crates/ac3`, A/52) | ✅ `ac3` (up to 5.1) |
| E-AC-3 | ✅          | ✅ (independent substream; 7.1 decodes as its 5.1 core) | ✅ `eac3` (up to 5.1) |
| DTS    | ✅          | ✅ (core; `crates/dts`) | ✅ `dts` (core, up to 5.1) |
| MP3    | ✅ (single-file MP4, `.mp3`) | ✅ (`crates/mp3`; MP2 and MP1 too) | ✅ `mp3` (CBR, stereo) |
| Vorbis | ✅ (WebM, `.ogg`) | ✅ (`crates/vorbis`) | ✅ `vorbis` (WebM, `.ogg`; 1–8 ch) |
| PCM    | — | ✅ | — |
| FLAC   | ✅ (`--audio flac`) | ✅ (`crates/lossless`) | ✅ `flac` |
| ALAC   | ✅ (`--audio alac`) | ✅ (`crates/lossless`) | ✅ `alac` |

`AudioCodecPolicy::Auto` passes through AAC/Opus/AC-3/E-AC-3/DTS, and MP3 into a
single-file MP4 (Opus and Vorbis into a WebM); transcodes the rest to Opus, and
drops what cannot be decoded. Every passthrough codec is also decoded when a
job needs its PCM — a downmix, an audio filter, another codec. The codecs live
in their own repositories: [rivet-opus](https://github.com/rivet-transcoder/rivet-opus),
[rivet-mp3](https://github.com/rivet-transcoder/rivet-mp3),
[rivet-vorbis](https://github.com/rivet-transcoder/rivet-vorbis),
[rivet-aac](https://github.com/rivet-transcoder/rivet-aac),
[rivet-ac3](https://github.com/rivet-transcoder/rivet-ac3),
[rivet-dts](https://github.com/rivet-transcoder/rivet-dts) and
[rivet-lossless](https://github.com/rivet-transcoder/rivet-lossless) (the
`crates/…` submodules).

Each `Force…` policy keeps a source already in its codec and encodes the rest:

- `ForceOpus` (`--audio opus`): 1–8 channels (family 0 for mono/stereo, family 1
  multistream for 3–8, RFC 7845 §5.1.1.2), into MP4 / MOV / WebM, HLS, or alone
  as an Ogg Opus file (`.opus`).
- `ForceMp3` (`--audio mp3`): CBR MP3 — into a single-file MP4 (`mp4a`, object
  type 0x6B, `codecs="mp3"`) or, with `--mode audio`, a bare `.mp3` behind the
  encoder's own gapless `Info` frame (encoder `rivetmp3`); HLS refuses it.
- `ForceAac` (`--audio aac`): AAC-LC (`mp4a.40.2`), mono to 7.1, for players
  that cannot take Opus (iOS / Safari before 17); 128k stereo, 64k mono, 384k
  5.1, 512k 7.1 by default. `ForceHeAac` (`he-aac`, `mp4a.40.5`) and
  `ForceHeAacV2` (`he-aacv2`, `mp4a.40.29`, stereo) add spectral band
  replication and parametric stereo for low rates (48k / 32k stereo by
  default), at 32, 44.1 or 48 kHz. Into MP4 / MOV, HLS, or an `.m4a`.
- `ForceVorbis` (`--audio vorbis`, `--audio-quality -1..10`, 5 by default):
  into a WebM, or alone as an Ogg Vorbis file. MP4 and CMAF have no Vorbis.
- `ForceAc3` / `ForceEac3` (`--audio ac3|eac3`): Dolby Digital (32–640 kb/s,
  448k for 5.1 by default) and Dolby Digital Plus (32–6144 kb/s), up to 5.1
  (`dac3` / `dec3`); `ForceDts` (`--audio dts`): the DTS core, up to 5.1, at
  ETSI TS 102 114's rates (1536 kb/s by default; `ddts`). Into MP4 / MOV, HLS,
  or an `.m4a`.

AAC, AC-3, E-AC-3 and DTS may be subject to patent licensing in some
jurisdictions; rivet grants no patent rights and makes no claim about whether
anyone needs a licence. An HE-AAC source decodes in full (SBR at the full
rate, parametric stereo to two channels); `--he-aac core` decodes only its
AAC-LC core and `--he-aac passthrough` never decodes it. `Drop` yields
video-only output.
`--audio-channels source|mono|stereo|5.1|7.1` sets the output layout: a
downmix by ITU-R BS.775 (LFE dropped, normalised so nothing clips), never an
upmix — asking for more channels than the source has is an error. HLS can add
a stereo downmix rendition beside a surround one (`--audio-stereo-fallback`).

Lossless output: `--audio flac` / `--audio alac` encode FLAC or ALAC with
rivet's own clean-room encoders (a source already in that codec is copied),
beside the video in MP4 or HLS (`CODECS="fLaC"` / `"alac"`), or alone with
`--mode audio` as a native `.flac` or an `.m4a`. FLAC in MP4 plays in Chrome,
Edge, Firefox and Safari; ALAC on Apple platforms and in Safari. The FLAC and
ALAC encoders and decoders live in their own repository,
[rivet-lossless](https://github.com/rivet-transcoder/rivet-lossless) (the
`crates/lossless` submodule). See
[docs/lossless-audio.md](https://github.com/rivet-transcoder/rivet/blob/HEAD/docs/lossless-audio.md).
`--audio-filter channelmap=…` remaps decoded PCM first
([docs/audio-filters.md](https://github.com/rivet-transcoder/rivet/blob/HEAD/docs/audio-filters.md)); 5.1 AAC is decoded to
downmix or re-encode it like any other surround source; see
[docs/output-spec.md](https://github.com/rivet-transcoder/rivet/blob/HEAD/docs/output-spec.md#3-audio--with_audioaudiocodecpolicy).
`--audio-decode-deny aac,mp3,…` names source codecs that may not be decoded: a
denied track is passed through where the output can carry it, and a job that
would have to decode it is refused
([details](https://github.com/rivet-transcoder/rivet/blob/HEAD/docs/output-spec.md#restricting-decoders--audio_decode_deny)).

#### Metadata

Identifying source metadata — location, capture time, device (make, model,
software, lens; serials and owner only with `device:all`) and descriptive
tags — is read from MP4 / MOV, Matroska, FLAC, MP3 and still images, and is
**not written** to any output unless named: `--metadata-keep` (settings key
`metadata-keep`) carries the named categories, at a level
(`location:approximate`, `capture_time:date`), into single-file, audio-only
and image output; HLS takes none. A copied FLAC stream keeps its STREAMINFO
block only, and with the device not kept a copied AAC or MP3 stream has the
source encoder's name cleared without its audio changing.

#### Output modes

| Mode     | Result |
|----------|--------|
| `single` | One self-contained file per rung: a faststart MP4 (AV1 + audio by default), a QuickTime movie (ProRes; or `--container mov`), or a WebM (VP8 / VP9, Opus or Vorbis audio). |
| `audio`  | The audio alone as one `.mp3`, a native `.flac`, an `.m4a` (ALAC, AAC / HE-AAC, AC-3, E-AC-3, DTS, or FLAC / Opus / MP3 with `--audio-container mp4`) or an Ogg file (Opus, Vorbis) — the file follows the codec unless `--audio-container` names one; also what `single` becomes for an input with no video. |
| `hls`    | A CMAF package: per-rung `init.mp4` + `seg-*.m4s`, a shared audio rendition, a media playlist per rung, and a `master.m3u8`. |
| `image`  | *(the `image` feature; `rivet image` or `rivet::image::run_image_job`)* Still images in AVIF / WebP / JPEG / PNG at one or more sizes, of a still image or of frames picked from a video. Upright, sRGB, and without EXIF / XMP / GPS unless `metadata-keep` names a category. |

## Crates

| Crate       | Responsibility |
|-------------|----------------|
| `h26x`      | **Native H.264 / HEVC decoders**, pure Rust, written from the ITU-T specs: bit-exact against the JVT and JCT-VC conformance suites, frame + wavefront threaded, AVX2 / NEON kernels at run time. rivet's software decode tier for the two codecs. A **git submodule** of [rivet-transcoder/rivet-h26x-codecs](https://github.com/rivet-transcoder/rivet-h26x-codecs) (published as [`rivet-h26x`](https://crates.io/crates/rivet-h26x)): clone with `--recurse-submodules` (or `git submodule update --init`), and change it there — commit and push inside `crates/h26x`, then commit the new pointer here. Its own [README](https://github.com/rivet-transcoder/rivet-h26x-codecs/blob/develop/README.md). |
| `aac`       | **AAC-LC, HE-AAC and HE-AAC v2 encoder and decoder**, pure Rust, written from the ISO/IEC standards. A **git submodule** of [rivet-transcoder/rivet-aac](https://github.com/rivet-transcoder/rivet-aac) (published as `rivet-aac`); changed there the same way as `h26x`. Its own [README](https://github.com/rivet-transcoder/rivet-aac/blob/develop/README.md). |
| `ac3`       | **AC-3 / E-AC-3 decoder and encoder**, pure Rust, written from ATSC A/52:2018: decodes AC-3 in full and E-AC-3 independent substream 0 (7.1 decodes as its 5.1 core); encodes AC-3 at 32–640 kb/s and E-AC-3 at 32–6144 kb/s. A **git submodule** of [rivet-transcoder/rivet-ac3](https://github.com/rivet-transcoder/rivet-ac3) (published as `rivet-ac3`); changed there the same way as `h26x`. Its own [README](https://github.com/rivet-transcoder/rivet-ac3/blob/develop/README.md). |
| `dts`       | **DTS Coherent Acoustics decoder and core encoder**, pure Rust, written from ETSI TS 102 114; a DTS-HD track decodes as its core (rivet asks for the core alone). A **git submodule** of [rivet-transcoder/rivet-dts](https://github.com/rivet-transcoder/rivet-dts) (published as `rivet-dts`); changed there the same way as `h26x`. Its own [README](https://github.com/rivet-transcoder/rivet-dts/blob/develop/README.md). |
| `opus`      | **Opus encoder and decoder**, pure Rust, written from RFC 6716 / 8251 / 7845: SILK, CELT and hybrid, every frame size, multistream (mono to 7.1); the decoder matches the reference's final range on all twelve official test vectors. A **git submodule** of [rivet-transcoder/rivet-opus](https://github.com/rivet-transcoder/rivet-opus) (published as `rivet-opus`); changed there the same way as `h26x`. Its own [README](https://github.com/rivet-transcoder/rivet-opus/blob/develop/README.md). |
| `mp3`       | **MPEG audio decoder (Layers I, II, III) and MP3 encoder**, pure Rust, written from ISO/IEC 11172-3 and 13818-3: the decoder meets ISO's full-accuracy criterion on all 64 conformance sequences; the encoder writes CBR / VBR Layer III with a gapless `Info` tag. A **git submodule** of [rivet-transcoder/rivet-mp3](https://github.com/rivet-transcoder/rivet-mp3) (published as `rivet-mp3`); changed there the same way as `h26x`. Its own [README](https://github.com/rivet-transcoder/rivet-mp3/blob/develop/README.md). |
| `vorbis`    | **Vorbis I decoder and encoder** with an Ogg reader and writer, pure Rust, written from the Vorbis I specification and RFC 3533. A **git submodule** of [rivet-transcoder/rivet-vorbis](https://github.com/rivet-transcoder/rivet-vorbis) (published as `rivet-vorbis`); changed there the same way as `h26x`. Its own [README](https://github.com/rivet-transcoder/rivet-vorbis/blob/develop/README.md). |
| `lossless`  | **FLAC and ALAC encoders and decoders** and the core they share, pure Rust, written from RFC 9639 and the published ALAC format description. A **git submodule** of [rivet-transcoder/rivet-lossless](https://github.com/rivet-transcoder/rivet-lossless) (published as `rivet-lossless`); changed there the same way as `h26x`. Its own [README](https://github.com/rivet-transcoder/rivet-lossless/blob/develop/README.md). |
| `prores`    | **Apple ProRes decoder and encoder**, pure Rust, written from SMPTE RDD 36: all six profiles, 4:2:2 and 4:4:4, interlaced, alpha. rivet's ProRes decode tier, the only one in the chain (alpha is dropped); the encoder is rivet's output encoder for the codec. A **git submodule** of [rivet-transcoder/rivet-prores](https://github.com/rivet-transcoder/rivet-prores) (published as `rivet-prores`); changed there the same way as `h26x`. Its own [README](https://github.com/rivet-transcoder/rivet-prores/blob/develop/README.md). |
| `vp8`       | **VP8 decoder and encoder**, pure Rust, written from RFC 6386: the decoder is bit-exact on all 18 comprehensive test vectors. rivet's software VP8 decode tier, behind NVDEC; the encoder is rivet's output encoder for the codec. A **git submodule** of [rivet-transcoder/rivet-vp8](https://github.com/rivet-transcoder/rivet-vp8) (published as `rivet-vp8`); changed there the same way as `h26x`. Its own [README](https://github.com/rivet-transcoder/rivet-vp8/blob/develop/README.md). |
| `vp9`       | **VP9 decoder and encoder**, pure Rust, written from the VP9 bitstream specification: the decoder takes profiles 0–3 and is bit-exact on 352 of the 353 public test vectors; the encoder writes profiles 0–3 (8 / 10 / 12-bit, 4:2:0 to 4:4:4) with rate-distortion partition and transform search and one- or two-pass rate control. rivet's software VP9 decode tier, behind NVDEC / AMF / QSV; the encoder is rivet's output encoder for the codec (4:2:0 at 8 or 10 bits). A **git submodule** of [rivet-transcoder/rivet-vp9](https://github.com/rivet-transcoder/rivet-vp9) (published as `rivet-vp9`); changed there the same way as `h26x`. Its own [README](https://github.com/rivet-transcoder/rivet-vp9/blob/develop/README.md). |
| `av1`       | **AV1 decoder and encoder**, pure Rust, written from the AV1 Bitstream & Decoding Process Specification: the decoder takes the whole specification and is bit-exact on all 244 AOM test vectors and all 3,015 Argon conformance streams (single-threaded, about 6 megapixels a second); the encoder writes profile 0, 8- or 10-bit 4:2:0, one tile, key and inter frames, at a fixed quantiser or under simple rate control. rivet's software AV1 decode tier, behind NVDEC / AMF / QSV, its software AV1 encoder (`av1-sw-fallback`) and its AVIF encoder. A **git submodule** of [rivet-transcoder/rivet-av1](https://github.com/rivet-transcoder/rivet-av1); changed there the same way as `h26x`. Its own [README](https://github.com/rivet-transcoder/rivet-av1/blob/develop/README.md). |
| `png`       | **PNG and APNG decoder and encoder**, with its own DEFLATE / zlib, pure Rust, written from the W3C PNG specification (third edition) and RFCs 1950 / 1951. rivet's PNG input and output (feature `image`) and the `overlay` filter's PNG reader. A **git submodule** of [rivet-transcoder/rivet-png](https://github.com/rivet-transcoder/rivet-png) (library `rpng`); changed there the same way as `h26x`. Its own [README](https://github.com/rivet-transcoder/rivet-png/blob/develop/README.md). |
| `jpeg`      | **JPEG decoder and encoder**, pure Rust, written from ITU-T T.81 and T.871: baseline, extended, progressive and lossless, Huffman and arithmetic coding, CMYK / YCCK. rivet's JPEG input and output (feature `image`). A **git submodule** of [rivet-transcoder/rivet-jpeg](https://github.com/rivet-transcoder/rivet-jpeg); changed there the same way as `h26x`. Its own [README](https://github.com/rivet-transcoder/rivet-jpeg/blob/develop/README.md). |
| `webp`      | **WebP decoder and encoder**, pure Rust, written from RFC 9649: lossy (VP8, through `vp8`), lossless (VP8L), alpha (`ALPH`), animation, ICC / EXIF / XMP. rivet's WebP input and output (feature `image`). A **git submodule** of [rivet-transcoder/rivet-webp](https://github.com/rivet-transcoder/rivet-webp) (library `webp`; its git dependency on rivet-vp8 is patched to `crates/vp8`); changed there the same way as `h26x`. Its own [README](https://github.com/rivet-transcoder/rivet-webp/blob/develop/README.md). |
| `imagecodecs` | **GIF, BMP and TIFF decoders and encoders** (`rivet-gif`, `rivet-bmp`, `rivet-tiff`: BigTIFF, LZW / Deflate / PackBits / CCITT), pure Rust, written from GIF89a, Microsoft's BMP documentation and TIFF 6.0. rivet's GIF, BMP and TIFF input (feature `image`). A separate cargo workspace, not a member of rivet's (its crates are path dependencies); a **git submodule** of [rivet-transcoder/rivet-imagecodecs](https://github.com/rivet-transcoder/rivet-imagecodecs); changed there the same way as `h26x`. Its own [README](https://github.com/rivet-transcoder/rivet-imagecodecs/blob/develop/README.md). |
| `mpeg2`     | **MPEG-2 Video (H.262) and MPEG-1 video decoder, Main Profile encoder**, pure Rust, written from ITU-T H.262: the decoder takes every main- and 4:2:2-profile stream of the ISO/IEC 13818-4 conformance suite. rivet's software MPEG-1 / MPEG-2 decode tier, behind NVDEC; the encoder is rivet's output encoder for the codec. A **git submodule** of [rivet-transcoder/rivet-mpeg2](https://github.com/rivet-transcoder/rivet-mpeg2) (published as `rivet-mpeg2`); changed there the same way as `h26x`. Its own [README](https://github.com/rivet-transcoder/rivet-mpeg2/blob/develop/README.md). |
| `mpeg4`     | **MPEG-4 Part 2 Visual decoder and encoder**, pure Rust, written from ISO/IEC 14496-2: Simple and Advanced Simple Profile and the H.263 short header (reversible VLCs refused). rivet's software MPEG-4 Part 2 decode tier, behind NVDEC; the encoder is rivet's output encoder for the codec. A **git submodule** of [rivet-transcoder/rivet-mpeg4](https://github.com/rivet-transcoder/rivet-mpeg4) (published as `rivet-mpeg4`); changed there the same way as `h26x`. Its own [README](https://github.com/rivet-transcoder/rivet-mpeg4/blob/develop/README.md). |
| `frame`     | The value types the codec and container layers share (`StreamInfo`, `VideoFrame`, `PixelFormat`, colour metadata, `EncodedPacket`) and the bitstream pixel-format probe, so `container` needs nothing from `codec`. |
| `codec`     | GPU detection (with PCI BAR / Resizable BAR reporting), decode (NVDEC / AMF / QSV / native H.264+HEVC, AV1, ProRes, VP8, VP9, MPEG-1/2, MPEG-4 Part 2), **AV1 / H.264 / H.265** encode (NVENC / AMF / QSV / software) and **VP9 / VP8 / MPEG-2 / MPEG-4 / ProRes** encode (the submodules' encoders, every build), colorspace + HDR→SDR tonemap, video and audio filters, audio decode/encode (Opus, AAC / HE-AAC, MP3, Vorbis, AC-3, E-AC-3, DTS, FLAC, ALAC, and decode of MP2 / PCM), probe. The H.264 / HEVC, AV1, ProRes, VP8, VP9, MPEG-2, MPEG-4, Opus, MPEG audio, Vorbis, AAC, AC-3, DTS, FLAC and ALAC codecs themselves are the submodules above, behind adapters here. Re-exports `frame`'s types at their old paths. |
| `container` | Demuxers (MP4/MOV/MKV/WebM/TS/MPEG-PS/AVI, bare MP3, FLAC and Ogg), MP4 / QuickTime muxer (AV1/H.264/H.265/VP9/VP8/MPEG-2/MPEG-4/ProRes) with audio and subtitles, a WebM muxer (VP8/VP9 + Opus or Vorbis), fragmented-MP4 (CMAF) writers, HLS playlist generation, `.mp3` / `.flac` / `.m4a` / `.ogg` writers, identifying-metadata read and write, bounded-RSS streaming demuxer. |
| `rivet`     | The configurable job engine (`run_job`), the output `spec`, the `progress` sink, the multi-GPU engine, the ABR `ladder` helper, rung `fit`ting, the shared `decode_pump`, `hooks`, still `image` jobs (feature `image`) with its own AVIF (HEIF) writer, plus simple `transcode`/`probe` helpers, the `rivet` CLI and the HTTP server. Re-exports `codec` + `container`. |

[`examples/yolo`](https://github.com/rivet-transcoder/rivet/tree/HEAD/examples/yolo) is a workspace member too, but not part of
rivet: an example program (unpublished) running YOLO detection on the hooks
through ONNX Runtime — see [docs/hooks-yolo.md](https://github.com/rivet-transcoder/rivet/blob/HEAD/docs/hooks-yolo.md).

## Building

The default build is Rust throughout — every audio and video codec in it is a
workspace crate — so it needs:

- **Rust 1.99** or newer: the workspace's `rust-version` (edition 2024), held
  by CI's MSRV job; every submodule crate declares the same.
- No CMake and no codec library. The GPU features need nothing at build time
  either; their runtimes are loaded with `dlopen`. No codec needs a C
  compiler, the still-image ones included.
- **nasm** — only for `openh264-fallback` (openh264's assembly).

On Windows the project links the static MSVC CRT (see `.cargo/config.toml`).

```sh
cargo build --release
cargo build --release --features qsv
cargo build --release --features av1-sw-fallback
```

### Optional features

| Feature     | Adds |
|-------------|------|
| `nvidia`    | NVENC hardware **encoder** (H.264, H.265; AV1 on Ada+) + NVDEC **decoder**, hand-rolled `dlopen` FFI (nvEncodeAPI / CUVID). |
| `amd`       | AMF hardware **encoder** (H.264 / H.265 on any AMF-capable AMD GPU, hardware-validated; AV1 on RDNA3+, by-review) and **decoder**, hand-rolled `dlopen` FFI mirrored from the AMF SDK v1.4.36 headers. |
| `qsv`       | Intel QSV hardware **encoder** (AV1, H.264, H.265) and **decoder**, hand-rolled `dlopen` oneVPL FFI (8-bit + 10-bit). Intel Arc / Meteor Lake+. |
| `av1-sw-fallback` | Lets the encoder chain fall back to **software AV1 encode** — this workspace's own [`av1`](https://github.com/rivet-transcoder/rivet/tree/HEAD/crates/av1) crate (pure Rust, profile 0, 8- and 10-bit 4:2:0, SDR) — when no hardware backend takes the job. No system libraries. The AV1 **decoder** needs no feature: it is always in the decode chain. `rav1e-fallback` is kept as an alias of this feature and `rav1d-fallback` as a no-op, for existing build scripts. |
| `h26x-fallback` | Lets the encoder chain fall back to **software H.264 / H.265 encode** — this workspace's own [`h26x`](https://github.com/rivet-transcoder/rivet/tree/HEAD/crates/h26x) crate (pure Rust, 4:2:0 at 8 and 10 bits, HDR10 / HLG signalled in the SPS VUI and the HDR10 static-metadata SEIs; SSE2→AVX-512 + NEON kernels). The matching **decoders** need no feature: they are always in the decode chain. |
| `openh264-fallback` | openh264 as the last-resort software H.264 **decoder**, below the native `h26x` decoder. Needs **NASM** on the build host. |
| `dpir` / `dpir-cuda` / `dpir-cudnn` | `--filter denoise=dpir[:SIGMA]` — deep denoise with DPIR's DRUNet on [candle](https://crates.io/crates/candle-core) (CPU; `dpir-cuda` needs nvcc at build time, `dpir-cudnn` adds cuDNN). A 130 MB model is downloaded once. See [docs/filters/denoise.md](https://github.com/rivet-transcoder/rivet/blob/HEAD/docs/filters/denoise.md#dpir--deep-denoise). |
| `thumbnail` | `rivet::thumbnail::generate_thumbnail` — capture a frame and encode an AVIF still (pulls the `av1` crate; rivet writes the AVIF container itself). |
| `image` | Still images (`rivet image`, `rivet::image::run_image_job`, `mode=image` in settings): JPEG / PNG / WebP / AVIF / GIF / TIFF / BMP / HEIC in, AVIF / WebP / JPEG / PNG out at several sizes, and stills from a video. Implies `thumbnail`; adds the workspace's `png`, `jpeg`, `webp`, GIF, BMP and TIFF crates and `moxcms` (ICC colour management). See [output-spec.md](https://github.com/rivet-transcoder/rivet/blob/HEAD/docs/output-spec.md#11-still-images--modeimage). |
| `batch`     | `rivet batch` — a YAML/JSON **manifest DSL** to convert many files in one run (pulls serde + a YAML/JSON parser + glob). See [docs/batch.md](https://github.com/rivet-transcoder/rivet/blob/HEAD/docs/batch.md). |
| `server`    | HTTP transcode API (`rivet serve`) — an axum webserver so another app can signal transcodes over the network. See [HTTP API](#http-api-server-feature). |
| `ipc`       | `rivet ipc` — a Unix-domain-socket server for streaming media in/out (Unix only at runtime). `rivet pipe` needs no feature. See [CLI](https://github.com/rivet-transcoder/rivet/blob/HEAD/docs/cli.md#rivet-ipc). |

Hooks need no feature. The YOLO example's own features (`cuda`, `directml`,
`openvino`, `image-jobs`) are in [`examples/yolo/Cargo.toml`](https://github.com/rivet-transcoder/rivet/blob/HEAD/examples/yolo/Cargo.toml).

### No FFmpeg

rivet does not depend on FFmpeg, in any build: no `ffmpeg-next`, no
libav\* linkage, nothing to install, and no feature that brings it in.

FFmpeg was removed entirely on 2026-08-12, restored on 2026-08-14 as an opt-in
software decode tier (the `ffmpeg` feature: libavcodec below the hardware and
`h26x` tiers, nothing else), and removed for good on 2026-10-02. The project
takes no dependency on FFmpeg of any kind, opt-in or not. What it cost was
never the code — it was the build: FFmpeg ≥ 7.0 development libraries on the
host, LLVM and libclang for bindgen, matching shared objects on the runtime
image, and an LGPL surface beside this project's own licence. A host without
all of that silently lost its software codec path, and the version window was narrow enough that a newer
FFmpeg broke the bindings outright.

What it did is covered in-tree, with no external toolchain:

| Was | Is |
|---|---|
| libavcodec software AV1 encode (`libsvtav1` / `libaom` / `librav1e`) | `av1-sw-fallback` — this workspace's own [`av1`](https://github.com/rivet-transcoder/rivet/tree/HEAD/crates/av1) encoder, pure Rust |
| libavcodec software AV1 decode | [`av1`](https://github.com/rivet-transcoder/rivet/tree/HEAD/crates/av1) — this workspace's own decoder, pure Rust, bit-exact on the AOM test vectors and the Argon conformance streams, always in the chain |
| libavcodec software H.264 / HEVC decode | [`h26x`](https://github.com/rivet-transcoder/rivet/tree/HEAD/crates/h26x) — this workspace's own decoders, pure Rust, bit-exact against the JVT / JCT-VC conformance suites, always in the chain |
| libavcodec software H.264 / HEVC encode (`libx264` / `libx265`) | `h26x-fallback` — the same crate's encoders, held to a SELF + JM / HM cross-check gate |
| libavcodec software ProRes, VP8, VP9, MPEG-1 / MPEG-2 and MPEG-4 Part 2 decode | [`prores`](https://github.com/rivet-transcoder/rivet/tree/HEAD/crates/prores), [`vp8`](https://github.com/rivet-transcoder/rivet/tree/HEAD/crates/vp8), [`vp9`](https://github.com/rivet-transcoder/rivet/tree/HEAD/crates/vp9), [`mpeg2`](https://github.com/rivet-transcoder/rivet/tree/HEAD/crates/mpeg2), [`mpeg4`](https://github.com/rivet-transcoder/rivet/tree/HEAD/crates/mpeg4) — this workspace's own decoders, pure Rust, each written clean-room from its format's specification (no other implementation's code read), always in the chain |
| libavcodec hwaccel decode | NVDEC / AMF / QSV, hand-rolled `dlopen` FFI, no SDK at build time |
| libavformat demux | this workspace's own MP4 / MKV / AVI / TS readers |

What rivet does not have, stated plainly: **HDR AV1 without a GPU** — the
software AV1 encoder writes 10-bit SDR at most, with no colour description in
its sequence header; **ProRes alpha**, which
is decoded and dropped (the pipeline has no alpha plane); and what each
decoder refuses by name (VP9 4:4:0 and RGB-coded streams, MPEG-4 Part 2
reversible VLCs and the tools its README lists, MPEG-2's scalable
extensions). (Before the first removal, the FFmpeg decoder was never
constructed by `create_decoder`, so the capability report claimed codecs it
never served; `rivet capabilities` lists only backends `create_decoder` can
build.) H.264 and HEVC came back in-tree as [`h26x`](https://github.com/rivet-transcoder/rivet/tree/HEAD/crates/h26x) (2026-08-18:
decode; 2026-08-27: encode), and ProRes, VP8, VP9, MPEG-1 / MPEG-2 and MPEG-4
Part 2 decode on 2026-10-02, and AV1 decode and encode on 2026-10-03, in
place of rav1d and rav1e; a GPU-less host decodes all of them and encodes AV1,
H.264 and HEVC with the fallback features on.

### Software codecs, and what the fallback features actually gate

The `av1` and `h26x` encoders are **always compiled** — they are pure Rust,
need no SDK, no bindgen and no system library, so there is nothing to gate a
build on. They are always testable, and a caller can always ask for one by
name (`TRANSCODE_ENCODER_BACKEND=h26x|av1`).

`av1-sw-fallback` / `h26x-fallback` gate something narrower: whether the
dispatch chain **falls back** to software on its own when every hardware
backend has declined or failed to initialise (`rav1e-fallback` is kept as an
alias of `av1-sw-fallback`, and `rav1d-fallback` as a no-op). (The `h26x` and
`av1` *decoders*, and the ProRes, VP8, VP9, MPEG-1 / MPEG-2 and MPEG-4 Part 2
ones, are not gated at all — they sit in the decode chain below the hardware tiers
unconditionally, since a decoder that refuses hands the stream on and costs
nothing when silicon takes it first.)

That is a policy decision rather than a capability one, which is why it is a
build-time switch and why it is off by default:

- A **throughput fleet** wants it off. Software AV1 is one to two orders of
  magnitude slower than a fixed-function encoder, so a node quietly degrading
  into it looks like a capacity problem rather than the missing driver it
  actually is. Off, the host fails loudly and gets fixed.
- A **workstation, CI runner, or GPU-less container** wants it on, because a
  slow file beats a diagnostic.

Either way software is tried **last**, and when it engages it says so at `warn`
with the reason.

The software AV1 encoder has no assembly and no tile or frame threading;
`--video-speed draft|standard|archive` (settings key `video-speed`) is the
dial between its speed and its compression, and VP9's.

```sh
# a laptop or CI box with no encode silicon: software AV1, H.264 and H.265
cargo build --release --features av1-sw-fallback,h26x-fallback
```

The hardware **encoders** are opt-in. All three are **hand-rolled `dlopen` FFI
in-tree** — no external wrapper crates, no bindgen, no build-time SDK link — so
they **build on both Windows MSVC and Linux** (`cargo build --features nvidia`
etc. works on either). A default build has no hardware encoder; enable `nvidia`
/ `amd` / `qsv` for your target silicon, or `av1-sw-fallback` / `h26x-fallback`
for software AV1 / H.264 / H.265. **Decode** is in-tree for all three vendors
too — NVDEC (`nvidia`), AMF (`amd`), and QSV (`qsv`), the same hand-rolled-FFI
approach — with `h26x` (H.264 / HEVC), `av1`, `prores`, `vp8`, `vp9`, `mpeg2`
and `mpeg4` (all always in) as the vendor-independent software paths.

## Contributing

rivet is **web-first and deliberately focused** — the web codecs (AV1 / H.264 /
H.265) and containers (MP4 / CMAF·HLS) are in scope; niche/legacy formats and
"everything FFmpeg does" are explicit **non-goals**. See
**[CONTRIBUTING.md](https://github.com/rivet-transcoder/rivet/blob/HEAD/CONTRIBUTING.md)** for the scope (in vs out), the dev setup,
and what a good PR looks like. The filter question for any feature: *does this
make video play better on the web, for real users?*

## License

**Open Encoding Attribution License v1.0** — a *source-available* license (not
OSI "open source"). It is **royalty-free for every use**. Personal, hobby,
nonprofit/academic/research, government, and purely-internal for-profit use are
free with no further obligation beyond keeping the existing notices. Shipping it
in a **commercial product** or running it as a **commercial service** (the
"hosted transcoder" case) is also permitted, but must **display attribution**
per §5. All distribution must keep existing notices and carry the
[NOTICE](https://github.com/rivet-transcoder/rivet/blob/HEAD/NOTICE) file (§4). Includes a patent grant with defensive termination
(§3). Not GPL-compatible. See [LICENSE.md](https://github.com/rivet-transcoder/rivet/blob/HEAD/LICENSE.md) for the full terms and the
use-case gist table.

All GPU codec FFI is hand-rolled in-tree (mirroring the vendor SDK headers);
no third-party GPU codec wrapper crates are used. (NVIDIA load and VRAM
readings go through the `nvml-wrapper` crate, which loads NVML at run time.)
