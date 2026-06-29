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
**CMAF/HLS** package. The output is fully configurable: you choose the **output
mode**, the **codec**, the **quality**, the **container/muxer**, and the exact
**rungs**, and you get an **asynchronous progress callback** with a uniform
per-rung status struct. AV1 is the default (royalty-clean AV1 + Opus in MP4);
H.264/H.265 are there for legacy-player compatibility — see [Choosing the output
codec](#choosing-the-output-codec).

It is built from clean-room demuxers, muxers, and hardware-codec dispatch —
**no FFmpeg required** by default (FFmpeg is available as an optional decode
backend behind a feature flag).

📖 **Detailed docs** live in [`docs/`](docs/). Start with
[Architecture](docs/architecture.md) (the codebase map) and
[Design decisions](docs/decisions.md) (the *why*); then
[Pipeline](docs/pipeline.md) (data flow), the per-crate references
([codec decode](docs/codec-decode.md) · [codec encode](docs/codec-encode.md) ·
[container](docs/container.md) · [engine](docs/engine.md)), and the usage guides
([OutputSpec](docs/output-spec.md) · [Batch manifest](docs/batch.md) ·
[CLI](docs/cli.md) · [HTTP API](docs/api.md)). This README is the quick tour.

## Why "rivet"

It fastens generic transcoding logic into a single, reusable component — a
library you can embed, a CLI you can run, and an HTTP service you can call.

The usual answer to "just transcode this" is FFmpeg — but FFmpeg is a CLI and a
C library, **not a service**. There's no job model, no structured per-rendition
progress, no HTTP surface: you shell out, scrape stderr, and build all the
orchestration yourself. rivet ships that part — a configurable job engine, a
uniform async progress callback, and an optional HTTP API (`rivet serve`) so
another application can signal a transcode over the network and poll it.

**Hardware selection is the other half.** Getting GPU encode/decode right across
vendors with FFmpeg means hand-picking `-hwaccel` flags, per-vendor encoder
names, pixel/surface formats, and init options — and it quietly falls back to a
slow software path when any of that is wrong. rivet detects the GPUs, dispatches
to the right framework per vendor (NVDEC/NVENC, AMF, QSV, with an optional FFmpeg
tier), leases them fairly across the ABR ladder, and **fails fast** instead of
degrading silently.

**And it's built to be fast at the ladder.** The source is decoded **once** and
the frames are fanned out to every rendition — a 5-rung ABR ladder decodes the
input one time, not five (the naïve `ffmpeg`-per-rung approach decodes it N
times). Encode work is then chunked and **leased across all available GPUs**
with mid-flight helper dispatch: when a fast rung frees its GPU, the freed lease
picks up another rung's chunks, so a slow rung finishes sooner and throughput
scales close to linearly with GPU count. Single-file output uses the same engine
— chunk-encode the one rendition across the GPUs and stitch the segments back
together losslessly. A per-rung codec invariant keeps cross-vendor chunks
bit-compatible, so an NVENC + QSV mix on the same rendition still decodes
cleanly. Stitched chunks always play (each is an independent IDR-led GOP), and
`ChunkSeamMode` (CLI `--seam-mode`, API `seam`) controls quality across the
seams: `Parallel` (default, fastest), `ParallelConstQp` (constant-QP, seam-flat),
or `Serial` (one encoder, seam-free) — see the [CLI reference](docs/cli.md#chunk-seams---seam-mode).

> The full data flow — demux → decode-once pump → per-rung scale → multi-GPU
> lease engine → mux — is documented in
> **[docs/pipeline.md](docs/pipeline.md)** (with a diagram and a code map).

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

A job is described by an [`OutputSpec`](crates/rivet/src/spec.rs):

| Dimension       | Type                         | Choices |
|-----------------|------------------------------|---------|
| **Output mode** | `OutputMode`                 | `SingleFile`, `Hls { segment_seconds }` |
| **Video codec** | `VideoCodecPolicy`           | `Av1` (default), `H264`, or `H265` — see [Choosing the output codec](#choosing-the-output-codec) |
| **Audio**       | `AudioCodecPolicy`           | `Auto` (passthrough/transcode), `ForceOpus`, `Drop` |
| **Container**   | `Container`                  | `Mp4`, `Cmaf` |
| **Muxer**       | `Muxer`                      | `Mp4File`, `CmafHls` |
| **Rungs**       | `Vec<Rung>`                  | each `Rung` = `width × height` + per-rung `Quality` (crf / speed / target / tier / keyframe interval) |
| **GPU policy**  | `EncodePolicy` / `decode_gpu`| all GPUs / single / pinned / vendor-family, plus a decode-pump GPU override — see [GPU scheduling](#gpu-scheduling-the-rung-benefit) |

Progress is reported through a [`ProgressSink`](crates/rivet/src/progress.rs) as
a uniform [`RungProgress`](crates/rivet/src/progress.rs) (status, percent,
frames, segments, bytes) per rung — wire it to a closure, a Tokio mpsc channel,
or your own implementation.

> **Complete reference: [Configuring a transcode — the `OutputSpec`
> guide](docs/output-spec.md)** documents every builder method, enum, and field
> (rungs/quality, audio, color/bit-depth, [video filters](docs/filters.md), GPU
> policy, chunk seams) with examples and how to run a job. The sections below are
> a tour of the highlights.

### Library usage

```toml
[dependencies]
# Published as `rivet-transcoder` (the crate name `rivet` was taken); the lib is
# `rivet`, so the rename keeps `use rivet::…` working as below.
rivet = { package = "rivet-transcoder", version = "0.1" }
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
use rivet::{OutputSpec, Rung, Quality, AudioCodecPolicy, PerceptualTarget};

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
tonemapping (needs a 10-bit AV1 encoder — `nvidia`, `amd`, `qsv`, or `ffmpeg`):

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

`encode_policy` controls how encode spreads across GPUs; `decode_gpu` overrides
the decode-pump device. See [GPU scheduling](#gpu-scheduling-the-rung-benefit)
for what each policy does.

```rust
use rivet::{OutputSpec, EncodePolicy, GpuFamily};

// All NVIDIA cards (ignore an integrated AMD/Intel GPU), but decode on GPU 0.
let spec = OutputSpec::single_file(rungs)
    .encode_policy(EncodePolicy::Family(GpuFamily::Nvidia))
    .decode_gpu(Some(0));

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

> **Full reference: [docs/cli.md](docs/cli.md)** — every subcommand, flag, and
> environment variable. A taste:

```sh
# Single MP4 at the source resolution (output defaults to <input>.av1.mp4)
rivet transcode input.mkv -o output.mp4

# Explicit rungs → a directory of MP4s
rivet transcode input.mkv -o out_dir/ --rung 1920x1080 --rung 1280x720 --rung 640x360

# Auto-derived standard ABR ladder
rivet transcode input.mkv -o out_dir/ --ladder --max-short-side 1080

# CMAF/HLS package with 4-second segments
rivet transcode input.mkv -o hls_dir/ --mode hls --ladder --segment-seconds 4

# Quality + audio knobs
rivet transcode input.mkv -o out.mp4 --crf 28 --speed 6 --audio opus

# Inspect without transcoding
rivet probe input.mkv [--json]

# Inspect the host + build
rivet devices [--json]        # detected GPUs: vendor, VRAM, live load
rivet capabilities [--json]   # what this build can encode/decode (alias: caps)

# Stream media in and out (no temp files)
cat input.mkv | rivet pipe > output.mp4                       # stdin → stdout (cross-platform)
cat input.mkv | rivet pipe --crf 28 --width 1280 --height 720 > out.mp4  # with settings
rivet ipc --socket /tmp/rivet.sock           # Unix-socket server; clients prefix a `#rivet k=v` header

# Convert many files from a YAML/JSON manifest (feature `batch`) — see docs/batch.md
rivet batch jobs.yaml --dry-run     # preview the plan
rivet batch jobs.yaml               # run it
```

GPU selection (mirrors `EncodePolicy` / `decode_gpu`):

```sh
rivet transcode in.mkv -o out.mp4 --gpu 1            # pin to GPU 1
rivet transcode in.mkv -o out.mp4 --single-gpu       # first GPU, serial
rivet transcode in.mkv -o out.mp4 --gpu-family nvidia # all NVIDIA cards
rivet transcode in.mkv -o out.mp4 --decode-gpu 0     # decode on GPU 0 (encode follows policy)
```

Set `RUST_LOG=debug` for verbose logging. Force an encoder backend with
`TRANSCODE_ENCODER_BACKEND=nvenc|amf|qsv`.

### HTTP API (`server` feature)

> **Full reference: [docs/api.md](docs/api.md)** — endpoints, the output-spec
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

### Choosing the output codec

The output codec is a first-class, selectable dimension. In Rust you pick it with
a [`VideoCodecPolicy`](crates/rivet/src/spec.rs) — the video analogue of
[`AudioCodecPolicy`](crates/rivet/src/spec.rs) — which is `Av1` (default), `H264`,
or `H265`. **AV1** is the recommended target (AV1 + Opus in MP4 = zero royalty
exposure); **H.264 / H.265** are there for legacy-player compatibility and carry
the patent-licensing obligations AV1 was chosen to avoid. The encode tier is
GPU-accelerated (NVENC / AMF / QSV). All three work for single-file MP4 **and**
CMAF/HLS (the muxer emits `av01`/`avc1`/`avc3`/`hvc1`/`hev1` sample entries and
the right `CODECS=` strings); AV1 stays the cross-vendor default.

You pick the codec the same way in every surface — codecs are the strings `av1`
/ `h264` / `h265` (aliases `avc`/`hevc`/`x264`/`x265`/`av01`/… accepted). Omit it
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

See [OutputSpec](docs/output-spec.md), [CLI](docs/cli.md),
[Batch](docs/batch.md), and [HTTP API](docs/api.md) for the full field set.

## Features

What rivet does and what it supports — the multi-GPU scheduler, and the
compatibility matrix of codecs, colors, containers, and output modes.

### GPU scheduling (the rung benefit)

Both HLS and single-file jobs run on a reactive multi-GPU orchestrator
([`multigpu`](crates/rivet/src/multigpu.rs)) that makes the ladder cheap:

- **Decode once.** A single decode pump feeds every rung — a 5-rung ladder
  decodes the source one time, not five.
- **Lease pool.** A process-wide [`GpuPool`](crates/rivet/src/gpu_pool.rs)
  hands out one encoder lease per GPU (concurrent NVENC sessions on one context
  deadlock — this is the load-bearing invariant), so work runs in parallel
  *across* GPUs.
- **Helpers.** When a fast unit of work releases its lease, the helper
  dispatcher grabs the freed lease and attaches an extra worker to a still-busy
  rung — segments/chunks are the unit of work, so a slow rung finishes sooner.
- **Cross-vendor safety.** A helper may land on a different GPU vendor (NVENC +
  QSV on the same rendition); a per-rung AV1 codec invariant guarantees every
  segment shares the `av1C` contract, and a mismatched helper requeues its
  chunk and exits without aborting the job.
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
host (or when the frame count is unknown) it uses the serial decode-once path
instead, with no chunk overhead. Either way, a host without AV1-encode silicon
fails fast with a clear error.

#### Encode policy

`OutputSpec::encode_policy(..)` selects how encode work spreads across GPUs (set
it from the library or the CLI — see above):

| Policy | Single-file | HLS |
|--------|-------------|-----|
| `EncodePolicy::AllGpus` *(default)* | chunk across all GPUs, stitch | ladder across all GPUs |
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
independently with `OutputSpec::decode_gpu(Some(i))` — e.g. decode on an
integrated GPU while the discrete GPUs encode.

### Compatibility matrix

#### Input — video decode

GPU decode is feature-gated — each vendor's tier is an opt-in cargo feature, and
`ffmpeg` adds the software catalogue (incl. ProRes). All decoders plug into the
shared decode pump (`create_decoder` → `push_sample` → `decode_next`).

| Codec          | NVDEC `nvidia` | AMF `amd` † | QSV `qsv` | FFmpeg `ffmpeg` |
|----------------|:--------------:|:----------:|:----------:|:---------------:|
| H.264 / AVC    | ✅             | ✅         | ✅         | ✅ |
| HEVC / H.265   | ✅             | ✅         | ✅         | ✅ |
| VP8            | ✅             | —          | —          | ✅ |
| VP9            | ✅             | ✅         | ✅         | ✅ |
| AV1            | ✅             | ✅         | ✅         | ✅ |
| MPEG-2         | ✅             | —          | —          | ✅ |
| MPEG-4 Part 2  | ✅             | —          | —          | ✅ |
| ProRes         | —              | —          | —          | ✅ |

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
  [TODO.md](TODO.md). `ffmpeg` is the fallback if the path proves unreliable.

What happens to a 10-bit / HDR source is the **`ColorPolicy`'s** call, not a
fixed rule (the decode pump never tonemaps on its own): the default
`TonemapToSdr` maps HDR → 8-bit SDR BT.709 for maximum web compatibility, while
`Hdr10` / `Hlg` / `Passthrough` keep it **10-bit HDR** through to a 10-bit
encoder (NVENC / AMF / QSV / `ffmpeg`) — see [Output color & bit
depth](#output-color--bit-depth). Decoding 10-bit needs a 10-bit-preserving
decoder: **NVIDIA** NVDEC decodes 10-bit **P016** natively and **Intel** QSV
decodes 10-bit **P010** (both carry 10-bit HEVC Main10 / HDR through), and
`ffmpeg` decodes 10-bit too.

#### Output — video encode (by vendor)

rivet encodes **AV1** (default, royalty-clean), **H.264**, or **H.265**, 4:2:0 —
pick the codec per [Choosing the output codec](#choosing-the-output-codec). One
table per vendor: rows are the output codecs, columns are the output pixel
format. ✅ = hardware-validated · ⏳ = follow-up (the backend rejects the codec
with a clear error rather than silently emitting AV1). AV1 carries 10-bit (pair
with a HDR `ColorPolicy` for HDR10/HLG; on its own, higher-precision SDR).
**H.265 also encodes 10-bit (Main 10)** on NVENC + QSV; **H.264 is 8-bit only** —
there is no hardware Hi10P profile on NVENC or QSV, so a 10-bit H.264 request is
capability-rejected rather than down-converted.

**NVENC — NVIDIA (`nvidia`)**

| Codec | 8-bit 4:2:0 | 10-bit 4:2:0 |
|-------|:-----------:|:------------:|
| AV1   | ✅ (Ada+)   | ✅ (`Yuv420_10bit`, Ada+) |
| H.264 | ✅ (Kepler+, RTX 3090-validated) | ❌ (no NVENC Hi10P silicon) |
| H.265 | ✅ (Maxwell+, RTX 3090-validated) | ✅ (Main 10, RTX 3090-validated) |

**AMF — AMD RDNA3+ (`amd`)**

| Codec | 8-bit 4:2:0 | 10-bit 4:2:0 |
|-------|:-----------:|:------------:|
| AV1   | ✅          | ✅ (`P010`) |
| H.264 | ⏳ (`VCE_AVC` follow-up) | — |
| H.265 | ⏳ (HEVC follow-up) | — |

**QSV — Intel Arc / Meteor Lake+ (`qsv`)**

| Codec | 8-bit 4:2:0 | 10-bit 4:2:0 |
|-------|:-----------:|:------------:|
| AV1   | ✅          | ✅ (P010) |
| H.264 | ✅ (Arc-validated) | ❌ (no `AVC High 10` in oneVPL) |
| H.265 | ✅ (Arc-validated) | ✅ (Main 10, Arc-validated) |

**FFmpeg (`ffmpeg`, software + hwaccel)**

| Codec | 8-bit 4:2:0 | 10-bit 4:2:0 |
|-------|:-----------:|:------------:|
| AV1   | ✅          | ✅ |
| H.264 | ⏳ (`h264_*` dispatch follow-up) | — |
| H.265 | ⏳ (`hevc_*` dispatch follow-up) | — |

GPU-only by default — a host with no encode silicon for the chosen codec (and no
`ffmpeg`) fails fast at encoder construction. 4:2:2 / 4:4:4 and 12-bit are not
produced. All hardware encoders are hand-rolled `dlopen` FFI in-tree (NVENC, AMF
`P010`, QSV oneVPL) and build on Windows + Linux. H.264/H.265 emit **Annex-B**,
which the muxer repackages to length-prefixed `avc1`/`avc3`/`hvc1`/`hev1` samples
(single-file MP4 **and** CMAF/HLS) — see [codec encode](docs/codec-encode.md).

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
(`yuv420p`), or `TenBit` (`yuv420p10le`). 10-bit / HDR output works on
**hardware** — `nvidia`, `amd`, or `qsv` — **no `ffmpeg` needed** — or in
software with `ffmpeg` (per the per-vendor tables above). The 10-bit output is
web-safe AV1 **Main** profile (4:2:0), HDR-tagged in the container via the
`colr`/`mdcv`/`clli` atoms, which browsers decode and tonemap. On a build with
no 10-bit encoder, `validate()` returns a clear error; the capability is
queryable at runtime via `codec::encode::build_output_caps()`.

For **web compatibility** keep the default — `.web_sdr()` (i.e. `TonemapToSdr` +
`Auto`) yields 8-bit SDR BT.709 AV1, which every browser and device that
supports AV1 plays.

#### Containers

| Container             | Demux (in) | Mux (out) |
|-----------------------|:----------:|:---------:|
| MP4 / MOV             | ✅         | ✅ (single-file + CMAF) |
| MKV / WebM            | ✅         | — |
| MPEG-TS               | ✅         | — |
| AVI (+OpenDML >1 GiB) | ✅         | — |
| CMAF / HLS            | —          | ✅ (segments + master/media playlists) |

#### Audio

| Codec  | Passthrough | Transcode → Opus |
|--------|:-----------:|:----------------:|
| AAC-LC | ✅          | — |
| Opus   | ✅          | (kept as-is)     |
| AC-3   | ✅          | — |
| E-AC-3 | ✅          | — |
| MP3    | —           | ✅ |
| Vorbis | —           | ✅ |

`AudioCodecPolicy::Auto` passes through AAC/Opus/AC-3/E-AC-3, transcodes MP3/Vorbis to
Opus, and drops the rest. `ForceOpus` produces Opus from any decodable source;
`Drop` yields video-only output. (Multichannel ≥3ch transcode is not yet
supported and is dropped with a warning.)

#### Output modes

| Mode     | Result |
|----------|--------|
| `single` | One self-contained MP4 per rung (faststart, AV1 + audio). |
| `hls`    | A CMAF package: per-rung `init.mp4` + `seg-*.m4s`, a shared audio rendition, a media playlist per rung, and a `master.m3u8`. |

## Crates

| Crate       | Responsibility |
|-------------|----------------|
| `codec`     | Frame types, pixel formats, GPU detection, decode (NVDEC / QSV / optional FFmpeg), **AV1** encode (NVENC / AMF / QSV), colorspace + HDR→SDR tonemap, audio decode/encode, probe. |
| `container` | Demuxers (MP4/MOV/MKV/WebM/TS/AVI), MP4 muxer (AV1/H.264/H.265) with audio, fragmented-MP4 (CMAF) writers, HLS playlist generation, bounded-RSS streaming demuxer. |
| `rivet`     | The configurable job engine (`run_job`), the output `spec`, the `progress` sink, the multi-GPU engine, the ABR `ladder` helper, the shared `decode_pump`, plus simple `transcode`/`probe` helpers and the `rivet` CLI. Re-exports `codec` + `container`. |

## Building

The default build links native libraries, so it needs a C toolchain plus:

- **nasm** — x86 assembly for the codec stack.
- **CMake** + a C/C++ compiler — builds libopus (Opus audio encode). Also builds
  Intel oneVPL when the `qsv` feature is enabled.

On Windows the project links the static MSVC CRT (see `.cargo/config.toml`). With
a modern CMake (4.x) you may need `CMAKE_POLICY_VERSION_MINIMUM=3.5` so libopus's
older `CMakeLists.txt` configures.

```sh
cargo build --release
cargo build --release --features qsv
cargo build --release --features ffmpeg
```

### Optional features

| Feature     | Adds |
|-------------|------|
| `nvidia`    | NVENC AV1 hardware **encoder** + NVDEC **decoder**, hand-rolled `dlopen` FFI (nvEncodeAPI / CUVID). NVIDIA Ada+ for AV1 encode. |
| `amd`       | AMF AV1 hardware **encoder**, hand-rolled `dlopen` FFI. AMD RDNA3+. (AMD decode → `ffmpeg`.) |
| `qsv`       | Intel QSV AV1 hardware **encoder**, hand-rolled `dlopen` oneVPL FFI (8-bit + 10-bit). Intel Arc / Meteor Lake+. (Intel decode → `ffmpeg`.) |
| `ffmpeg`    | libavcodec as the primary decode path (full software catalogue + Vulkan/NVDEC/D3D11/VAAPI hwaccel + AV1 software encode). Needs FFmpeg ≥7.0 dev libs + LLVM/libclang. |
| `thumbnail` | `rivet::thumbnail::generate_thumbnail` — capture a frame and encode an AVIF still (pulls `ravif`/rav1e). |
| `batch`     | `rivet batch` — a YAML/JSON **manifest DSL** to convert many files in one run (pulls serde + a YAML/JSON parser + glob). See [docs/batch.md](docs/batch.md). |
| `server`    | HTTP transcode API (`rivet serve`) — an axum webserver so another app can signal transcodes over the network. See [HTTP API](#http-api-server-feature). |
| `ipc`       | `rivet ipc` — a Unix-domain-socket server for streaming media in/out (Unix only at runtime). `rivet pipe` needs no feature. See [CLI](docs/cli.md#rivet-ipc). |

The hardware **encoders** are opt-in. All three are **hand-rolled `dlopen` FFI
in-tree** — no external wrapper crates, no bindgen, no build-time SDK link — so
they **build on both Windows MSVC and Linux** (`cargo build --features nvidia`
etc. works on either). A default build has no hardware encoder; enable `nvidia`
/ `amd` / `qsv` (or `ffmpeg`) for your target silicon. **Decode** is in-tree for
all three vendors too — NVDEC (`nvidia`), AMF (`amd`), and QSV (`qsv`), the same
hand-rolled-FFI approach — with `ffmpeg` as the cross-vendor fallback.

## License

**Open Encoding Attribution License v1.0** — a *source-available* license (not
OSI "open source"). It is **royalty-free for every use**. Personal, hobby,
nonprofit/academic/research, government, and purely-internal for-profit use are
free with no further obligation beyond keeping the existing notices. Shipping it
in a **commercial product** or running it as a **commercial service** (the
"hosted transcoder" case) is also permitted, but must **display attribution**
per §5. All distribution must keep existing notices and carry the
[NOTICE](NOTICE) file (§4). Includes a patent grant with defensive termination
(§3). Not GPL-compatible. See [LICENSE.md](LICENSE.md) for the full terms and the
use-case gist table.

All GPU hardware FFI is hand-rolled in-tree (mirroring the vendor SDK headers);
no third-party GPU wrapper crates are used.
