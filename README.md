# rivet

A modular, GPU-accelerated video transcoding **library** and **command-line
tool**, written in Rust.

`rivet` takes an arbitrary input file and transcodes it to **AV1** — as a
single MP4, a multi-rendition ABR ladder, or a segmented **CMAF/HLS** package.
The output is fully configurable: you choose the **output mode**, the **codec**,
the **quality**, the **container/muxer**, and the exact **rungs**, and you get
an **asynchronous progress callback** with a uniform per-rung status struct.

It is built from clean-room demuxers, muxers, and hardware-codec dispatch —
**no FFmpeg required** by default (FFmpeg is available as an optional decode
backend behind a feature flag).

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

**"Optimized for web" is a pile of decisions FFmpeg leaves to you.** rivet bakes
in defaults that just play in a browser (and lets you override them): AV1 (the
royalty-clean codec target) + Opus audio, faststart MP4 or segment-aligned
CMAF/HLS for ABR, and correct color — HDR tonemapped down to 8-bit SDR BT.709 by
policy, so a clip doesn't land eye-searingly bright or washed-out on a viewer's
screen. Picking those knobs correctly per source is exactly the expertise rivet
encodes so you don't have to.

## Quick start

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

## What you configure

A job is described by an [`OutputSpec`](crates/rivet/src/spec.rs):

| Dimension       | Type                         | Choices |
|-----------------|------------------------------|---------|
| **Output mode** | `OutputMode`                 | `SingleFile`, `Hls { segment_seconds }` |
| **Video codec** | `VideoCodec`                 | `Av1` (the only implemented codec — see [note](#a-note-on-the-output-codec)) |
| **Audio**       | `AudioPolicy`                | `Auto` (passthrough/transcode), `ForceOpus`, `Drop` |
| **Container**   | `Container`                  | `Mp4`, `Cmaf` |
| **Muxer**       | `Muxer`                      | `Mp4File`, `CmafHls` |
| **Rungs**       | `Vec<Rung>`                  | each `Rung` = `width × height` + per-rung `Quality` (crf / speed / target / tier / keyframe interval) |
| **GPU policy**  | `EncodePolicy` / `decode_gpu`| all GPUs / single / pinned / vendor-family, plus a decode-pump GPU override — see [GPU scheduling](#gpu-scheduling-the-rung-benefit) |

Progress is reported through a [`ProgressSink`](crates/rivet/src/progress.rs) as
a uniform [`RungProgress`](crates/rivet/src/progress.rs) (status, percent,
frames, segments, bytes) per rung — wire it to a closure, a Tokio mpsc channel,
or your own implementation.

## Library usage

```toml
[dependencies]
rivet = { git = "https://github.com/elyerinfox/rivet" }
```

### One file in, one file out

```rust
let outcome = rivet::transcode_file("input.mkv", "output.mp4")?;
println!("{} frames out", outcome.frames_processed);

let info = rivet::probe_file("input.mkv")?;
println!("{}x{} {}", info.width, info.height, info.video_codec);
```

### A configurable job with progress

```rust
use std::sync::Arc;
use rivet::{OutputSpec, Rung, AudioPolicy, run_job_blocking, fn_sink};
use rivet::progress::RungProgress;

let bytes = std::fs::read("input.mkv")?;

// A 3-rung HLS ladder, 4-second segments, audio auto-handled.
let spec = OutputSpec::hls(
    vec![Rung::new(1920, 1080), Rung::new(1280, 720), Rung::new(640, 360)],
    4.0,
)
.with_audio(AudioPolicy::Auto);

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

### Color, bit depth & frame rate

A fully-specified single-file job, picking the codec quality, frame-rate cap,
color/tonemap policy, and output bit depth per [the table below](#output-color--bit-depth):

```rust
use rivet::{OutputSpec, Rung, Quality, AudioPolicy, ColorPolicy, PixelDepth, PerceptualTarget};

let spec = OutputSpec::single_file(vec![
    Rung::new(1920, 1080).with_quality(Quality::crf(28)),
    Rung::new(1280, 720).with_quality(Quality::target(PerceptualTarget::Standard)),
])
.with_audio(AudioPolicy::Auto)
.with_max_frame_rate(30.0)                  // cap output cadence at 30 fps
.with_color(ColorPolicy::TonemapToSdr)      // tonemap HDR sources → SDR BT.709 (default)
.with_pixel_format(PixelDepth::Eight);      // 8-bit 4:2:0 — universal web compatibility

spec.validate()?; // rejects e.g. HDR without 10-bit, or 10-bit on an 8-bit-only build
```

To keep HDR instead of tonemapping (needs a 10-bit-capable encoder — the
`ffmpeg` feature):

```rust
let spec = OutputSpec::single_file(rungs)
    .with_color(ColorPolicy::Hdr10)         // BT.2020 + PQ, no tonemap
    .with_pixel_format(PixelDepth::Ten);    // 10-bit
```

### Choosing GPUs

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

### Escape hatch

Need finer control than the engine offers? Reach through the re-exported
component crates:

```rust
use rivet::codec::encode::{select_encoder, EncoderConfig};
use rivet::container::cmaf::CmafVideoMuxer;
```

## CLI usage

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

## HTTP API (`server` feature)

For a service deployment — where another application **signals** rivet to
transcode something — build with the `server` feature and run `rivet serve`. It
exposes the same engine over HTTP:

```sh
cargo build --release --features server,nvidia   # the API + an AV1 encoder
rivet serve --addr 0.0.0.0:8080
```

Interactive docs ship with it: **`/swagger`** (Swagger UI), **`/redoc`** (Redoc),
and the raw **`/openapi.json`** (OpenAPI 3.0); `/` links to all three.

| Endpoint | Purpose |
|----------|---------|
| `GET  /v1/health` | liveness + detected GPUs + this build's output capabilities |
| `POST /v1/probe` | body = media bytes → JSON media info |
| `POST /v1/transcode` | body = media bytes, spec from query params → `202 {job_id}` (async) |
| `GET  /v1/jobs/{id}` | job status + per-rung progress + output list |
| `GET  /v1/jobs/{id}/artifacts/{label}` | download a single-file rung's MP4 |
| `GET  /v1/jobs/{id}/files/{*path}` | fetch an HLS file (`master.m3u8`, `…/seg-00001.m4s`) |
| `GET  /openapi.json`, `/swagger`, `/redoc` | OpenAPI 3.0 document + Swagger UI / Redoc |

`/v1/transcode` takes the output spec as query params — `mode` (single|hls),
`rungs=1280x720,640x360` (or `ladder=true&max_short_side=1080`), `crf`, `speed`,
`audio`, `color`, `pixel_format`, `max_fps`, `gpu`, `segment_seconds` — mirroring
the CLI. Pass `?sync=true` to block and get a single-file MP4 back directly.

```sh
# Signal a transcode: POST the media, poll the job.
curl -s http://localhost:8080/v1/health
job=$(curl -s --data-binary @input.mkv \
      "http://localhost:8080/v1/transcode?mode=single&crf=28" | jq -r .job_id)
curl -s "http://localhost:8080/v1/jobs/$job"               # status + progress
curl -so out.mp4 "http://localhost:8080/v1/jobs/$job/artifacts/720p"   # download

# Or synchronously (single-file, single rung):
curl -s --data-binary @input.mkv "http://localhost:8080/v1/transcode?sync=true" -o out.mp4
```

The job registry is in-memory and completed single-file artifacts are held in
RAM, so this is a sidecar/worker, not a public CDN — a production deployment
would offload outputs to object storage from a `ProgressSink` watching
`RungStatus::Completed`.

## GPU scheduling (the rung benefit)

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

For **single-file** output, each rung is chunked at GOP boundaries and the
chunks are encoded across the GPUs, then stitched — in segment order, in memory,
no disk round-trip — into one MP4 per rung. Because the encoder runs
constant-quality (CQP/CRF), independent chunks have no rate-control
discontinuity at the seams; each chunk just starts with an IDR. On a single-GPU
host (or when the frame count is unknown) it uses the serial decode-once path
instead, with no chunk overhead. Either way, a host without AV1-encode silicon
fails fast with a clear error.

### Encode policy

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

### A note on the output codec

AV1 is the only implemented video codec — it is the project's locked,
royalty-clean target (AV1 + Opus). `VideoCodec` is an enum so the dimension is
selectable and future codecs can be added without an API break. The encode tier
is GPU-accelerated (NVENC / AMF / QSV).

## Compatibility matrix

### Input — video decode

Default builds decode on the GPU via the built-in NVDEC. The `nvidia` / `amd` /
`qsv` features add decoders via the shiguredo wrapper crates, and `ffmpeg` adds
the software catalogue (incl. ProRes). All decoders plug into the shared decode
pump (`create_decoder` → `push_sample` → `decode_next`).

| Codec          | NVDEC built-in | NVDEC `nvidia` | AMF `amd` | QSV `qsv` | FFmpeg `ffmpeg` |
|----------------|:--------------:|:--------------:|:---------:|:---------:|:---------------:|
| H.264 / AVC    | ✅             | ✅             | ✅        | ✅        | ✅ |
| HEVC / H.265   | ✅             | ✅             | ✅        | ✅        | ✅ |
| VP8            | ✅             | ✅             | —         | —         | ✅ |
| VP9            | ✅             | ✅             | —         | ✅        | ✅ |
| AV1            | ✅             | ✅             | ✅        | ✅        | ✅ |
| MPEG-2         | ✅             | —              | —         | —         | ✅ |
| MPEG-4 Part 2  | ✅             | —              | —         | —         | ✅ |
| ProRes         | —              | —              | —         | —         | ✅ |

- **NVDEC built-in** — the hand-rolled NVDEC, always compiled (no feature).
- **NVDEC `nvidia`** — `shiguredo_nvcodec`; preferred over built-in for the
  codecs it covers when the feature is on (MPEG-2/4 fall back to built-in).
- **AMF `amd`** — `shiguredo_amf`, a new AMD decode tier.
- The `nvidia` / `amd` / `qsv` features are the same Apache-2.0 shiguredo crates
  as the encoders — they build on Linux but not on a Windows MSVC host (see the
  [platform note](#optional-features)).

10-bit / HDR sources decode and are tonemapped to 8-bit SDR BT.709 before encode
(single-output policy). The shiguredo decoder wrappers output 8-bit NV12; the
built-in NVDEC handles 10-bit/P016.

### Output — video encode

| Codec | NVENC (NVIDIA Ada+) | AMF (AMD RDNA3+) | QSV (Intel Arc+, `qsv`) | FFmpeg (`ffmpeg`) |
|-------|:-------------------:|:----------------:|:-----------------------:|:-----------------:|
| AV1   | ✅                  | ✅               | ✅                      | ✅ (av1_nvenc / amf / qsv / vaapi / svt / aom) |

GPU-only by default — a host with no AV1-encode silicon fails at encoder
construction (use the `ffmpeg` feature for a software fallback).

### Output color & bit depth

`OutputSpec::with_color(ColorPolicy)` + `with_pixel_format(PixelDepth)` choose
the output color and bit depth; the decode pump tonemaps **only** when the
policy says so (it never decides on its own). `validate()` rejects any
combination this build can't actually produce:

| `ColorPolicy`  | Tonemap | Output signaling          | Bit depth | Needs |
|----------------|:-------:|---------------------------|:---------:|-------|
| `TonemapToSdr` *(default)* | HDR→SDR | BT.709 SDR             | 8-bit     | any encoder |
| `Passthrough`  | no      | source color verbatim     | source    | 10-bit encoder if source is 10-bit |
| `Hdr10`        | no      | BT.2020 + PQ (ST 2084)    | 10-bit    | `ffmpeg` feature |
| `Hlg`          | no      | BT.2020 + ARIB STD-B67    | 10-bit    | `ffmpeg` feature |

`PixelDepth` is `Auto` (follow the policy), `Eight`, or `Ten`. What each
encoder path can produce:

| Encoder path             | Max bit depth | HDR signaling |
|--------------------------|:-------------:|:-------------:|
| NVENC / AMF / QSV (shiguredo wrappers) | 8-bit | — |
| FFmpeg (`ffmpeg` feature) | 10-bit       | ✅            |

So 10-bit / HDR output requires the `ffmpeg` feature today; on a default
(hardware-only) build, `validate()` returns a clear error pointing at it. The
matrix is queryable at runtime via `codec::encode::build_output_caps()`.

For **web compatibility** keep the defaults — `TonemapToSdr` + `Auto` yields
8-bit SDR BT.709 AV1, which every browser and device that supports AV1 plays.

### Containers

| Container             | Demux (in) | Mux (out) |
|-----------------------|:----------:|:---------:|
| MP4 / MOV             | ✅         | ✅ (single-file + CMAF) |
| MKV / WebM            | ✅         | — |
| MPEG-TS               | ✅         | — |
| AVI (+OpenDML >1 GiB) | ✅         | — |
| CMAF / HLS            | —          | ✅ (segments + master/media playlists) |

### Audio

| Codec  | Passthrough | Transcode → Opus |
|--------|:-----------:|:----------------:|
| AAC-LC | ✅          | — |
| Opus   | ✅          | (kept as-is)     |
| AC-3   | ✅          | — |
| E-AC-3 | ✅          | — |
| MP3    | —           | ✅ |
| Vorbis | —           | ✅ |

`AudioPolicy::Auto` passes through AAC/Opus/AC-3/E-AC-3, transcodes MP3/Vorbis to
Opus, and drops the rest. `ForceOpus` produces Opus from any decodable source;
`Drop` yields video-only output. (Multichannel ≥3ch transcode is not yet
supported and is dropped with a warning.)

### Output modes

| Mode     | Result |
|----------|--------|
| `single` | One self-contained MP4 per rung (faststart, AV1 + audio). |
| `hls`    | A CMAF package: per-rung `init.mp4` + `seg-*.m4s`, a shared audio rendition, a media playlist per rung, and a `master.m3u8`. |

## Crates

| Crate       | Responsibility |
|-------------|----------------|
| `codec`     | Frame types, pixel formats, GPU detection, decode (NVDEC / QSV / optional FFmpeg), **AV1** encode (NVENC / AMF / QSV), colorspace + HDR→SDR tonemap, audio decode/encode, probe. |
| `container` | Demuxers (MP4/MOV/MKV/WebM/TS/AVI), AV1 MP4 muxer with audio, fragmented-MP4 (CMAF) writers, HLS playlist generation, bounded-RSS streaming demuxer. |
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
| `nvidia`    | NVENC AV1 hardware **encoder** + NVDEC **decoder** via [`shiguredo_nvcodec`](https://github.com/shiguredo/nvcodec-rs) (Apache-2.0). Needs libclang at build time; dlopens CUDA at runtime. NVIDIA Ada+. |
| `amd`       | AMF AV1 hardware **encoder** + **decoder** via [`shiguredo_amf`](https://github.com/shiguredo/amf-rs) (Apache-2.0). Needs libclang at build time; dlopens the AMF runtime. AMD RDNA3+. |
| `qsv`       | Intel QuickSync / oneVPL hardware decode + encode via [`shiguredo_vpl`](https://github.com/shiguredo/vpl-rs) (Apache-2.0; needs CMake + libvpl). Intel Arc / Meteor Lake+. |
| `ffmpeg`    | libavcodec as the primary decode path (full software catalogue + Vulkan/NVDEC/D3D11/VAAPI hwaccel + AV1 software encode). Needs FFmpeg ≥7.0 dev libs + LLVM/libclang. |
| `thumbnail` | `rivet::thumbnail::generate_thumbnail` — capture a frame and encode an AVIF still (pulls `ravif`/rav1e). |
| `server` | HTTP transcode API (`rivet serve`) — an axum webserver so another app can signal transcodes over the network. See [HTTP API](#http-api-server-feature). |

The hardware **encoders/decoders** are opt-in: the NVENC, AMF, and QSV backends
are wrappers over the Apache-2.0 `shiguredo_{nvcodec,amf,vpl}` crates (the
hand-rolled FFI mirrors were retired). A default build has no hardware encoder —
enable `nvidia` / `amd` / `qsv` (or `ffmpeg`) for your target silicon. NVIDIA
**decode** (NVDEC) remains built-in.

> ⚠️ **Platform note.** The three `shiguredo_*` crates bindgen the vendor SDK
> headers and compile on **Linux** (the production / Docker target) but **not on
> a Windows MSVC host**. Under the MSVC ABI a non-negative C enum is signed
> (`int` → `i32`); under the Linux ABI it is `unsigned int` (→ `u32`), which is
> what the crates expect. So build the `nvidia` / `amd` / `qsv` features on Linux
> (or in the Docker image); on a Windows dev box use the `ffmpeg` feature or
> leave them off. Each feature needs `libclang` at build time (`LIBCLANG_PATH`).

## License

Apache-2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE). The NOTICE file credits
the Apache-2.0 third-party components used for platform-specific GPU codec access
(`shiguredo_nvcodec` / `shiguredo_amf` / `shiguredo_vpl`).
