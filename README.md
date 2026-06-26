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

It fastens the generic transcoding logic that grew up inside a video-processing
microservice into a standalone, reusable component — a library you can embed
and a CLI you can run.

## What you configure

A job is described by an [`OutputSpec`](crates/rivet/src/spec.rs):

| Dimension      | Type          | Choices |
|----------------|---------------|---------|
| **Output mode**| `OutputMode`  | `SingleFile`, `Hls { segment_seconds }` |
| **Video codec**| `VideoCodec`  | `Av1` (the only implemented codec — see below) |
| **Audio**      | `AudioPolicy` | `Auto` (passthrough/transcode), `ForceOpus`, `Drop` |
| **Container**  | `Container`   | `Mp4`, `Cmaf` |
| **Muxer**      | `Muxer`       | `Mp4File`, `CmafHls` |
| **Rungs**      | `Vec<Rung>`   | each `Rung` = `width × height` + per-rung `Quality` (crf / speed / target / tier / keyframe interval) |

Progress is reported through a [`ProgressSink`](crates/rivet/src/progress.rs)
as a uniform [`RungProgress`] (status, percent, frames, segments, bytes) per
rung — wire it to a closure, a Tokio mpsc channel, or your own implementation.

## Multi-GPU engine (the rung benefit)

HLS jobs run on a reactive multi-GPU orchestrator
([`multigpu`](crates/rivet/src/multigpu.rs)) that makes the ABR ladder cheap:

- **Decode once.** A single decode pump feeds every rung — a 5-rung ladder
  decodes the source one time, not five.
- **Lease pool.** A process-wide [`GpuPool`](crates/rivet/src/gpu_pool.rs)
  hands out one encoder lease per GPU (concurrent NVENC sessions on one context
  deadlock — this is the load-bearing invariant), so rungs encode in parallel
  *across* GPUs.
- **Helpers.** When a fast rung releases its lease, the helper dispatcher grabs
  the freed lease and attaches an extra worker to a still-busy rung — segments
  are the unit of work, so a slow rung finishes sooner.
- **Cross-vendor safety.** A helper may land on a different GPU vendor (NVENC +
  QSV on the same rendition); a per-rung AV1 codec invariant guarantees every
  segment shares the `av1C` contract, and a mismatched helper requeues its
  chunk and exits without aborting the job.

Single-file jobs use the same decode-once pump with one MP4 muxer per rung. On
a host without AV1-encode silicon the job fails fast with a clear error.

> **Output codec.** AV1 is the only implemented video codec — it is the
> project's locked, royalty-clean target (AV1 + Opus). `VideoCodec` is an enum
> so the dimension is selectable and future codecs can be added without an API
> break. The encode tier is GPU-accelerated (NVENC / AMF / QSV).

## Crates

| Crate       | Responsibility |
|-------------|----------------|
| `codec`     | Frame types, pixel formats, GPU detection, decode (NVDEC / QSV / optional FFmpeg), **AV1** encode (NVENC / AMF / QSV), colorspace + HDR→SDR tonemap, audio decode/encode, probe. |
| `container` | Demuxers (MP4/MOV/MKV/WebM/TS/AVI), AV1 MP4 muxer with audio, fragmented-MP4 (CMAF) writers, HLS playlist generation, bounded-RSS streaming demuxer. |
| `rivet`     | The configurable job engine (`run_job`), the output `spec`, the `progress` sink, the ABR `ladder` helper, the shared `decode_pump`, plus simple `transcode`/`probe` helpers and the `rivet` CLI. Re-exports `codec` + `container`. |

## Library usage

```toml
[dependencies]
rivet = { git = "https://github.com/elyerinfox/rivet" }
```

### Simple: one file in, one file out

```rust
let outcome = rivet::transcode_file("input.mkv", "output.mp4")?;
println!("{} frames out", outcome.frames_processed);

let info = rivet::probe_file("input.mkv")?;
println!("{}x{} {}", info.width, info.height, info.video_codec);
```

### Configurable: output modes, rungs, and progress

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

Set `RUST_LOG=debug` for verbose logging. Force an encoder backend with
`TRANSCODE_ENCODER_BACKEND=nvenc|amf|qsv`.

## Compatibility matrix

### Input — video decode

Default builds decode on the GPU via the built-in NVDEC. The `nvidia` / `amd`
/ `qsv` features add decoders via the shiguredo wrapper crates, and `ffmpeg`
adds the software catalogue (incl. ProRes). All decoders plug into the shared
decode pump (`create_decoder` → `push_sample` → `decode_next`).

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
- The `nvidia` / `amd` / `qsv` features are the same Apache-2.0 shiguredo
  crates as the encoders, so they build on Linux but not on a Windows MSVC
  host (see the features note below).

10-bit / HDR sources decode and are tonemapped to 8-bit SDR BT.709 before
encode (single-output policy). The shiguredo decoder wrappers output 8-bit
NV12; the built-in NVDEC handles 10-bit/P016.

### Output — video encode

| Codec | NVENC (NVIDIA Ada+) | AMF (AMD RDNA3+) | QSV (Intel Arc+, `qsv`) | FFmpeg (`ffmpeg`) |
|-------|:-------------------:|:----------------:|:-----------------------:|:-----------------:|
| AV1   | ✅                  | ✅               | ✅                      | ✅ (av1_nvenc / amf / qsv / vaapi / svt / aom) |

GPU-only by default — a host with no AV1-encode silicon fails at encoder
construction (use the `ffmpeg` feature for a software fallback).

### Containers

| Container   | Demux (in) | Mux (out) |
|-------------|:----------:|:---------:|
| MP4 / MOV   | ✅         | ✅ (single-file + CMAF) |
| MKV / WebM  | ✅         | — |
| MPEG-TS     | ✅         | — |
| AVI (+OpenDML >1 GiB) | ✅ | — |
| CMAF / HLS  | —          | ✅ (segments + master/media playlists) |

### Audio

| Codec        | Passthrough | Transcode → Opus |
|--------------|:-----------:|:----------------:|
| AAC-LC       | ✅          | — |
| Opus         | ✅          | (kept as-is)     |
| AC-3         | ✅          | — |
| E-AC-3       | ✅          | — |
| MP3          | —           | ✅ |
| Vorbis       | —           | ✅ |

`AudioPolicy::Auto` passes through AAC/Opus/AC-3/E-AC-3, transcodes MP3/Vorbis
to Opus, and drops the rest. `ForceOpus` produces Opus from any decodable
source; `Drop` yields video-only output. (Multichannel ≥3ch transcode is not
yet supported and is dropped with a warning.)

### Output modes

| Mode     | Result |
|----------|--------|
| `single` | One self-contained MP4 per rung (faststart, AV1 + audio). |
| `hls`    | A CMAF package: per-rung `init.mp4` + `seg-*.m4s`, a shared audio rendition, a media playlist per rung, and a `master.m3u8`. |

## Building

The default build links native libraries, so it needs a C toolchain plus:

- **nasm** — x86 assembly for the codec stack.
- **CMake** + a C/C++ compiler — builds libopus (Opus audio encode). Also
  builds Intel oneVPL when the `qsv` feature is enabled.

On Windows the project links the static MSVC CRT (see `.cargo/config.toml`).
With a modern CMake (4.x) you may need `CMAKE_POLICY_VERSION_MINIMUM=3.5` so
libopus's older `CMakeLists.txt` configures.

```sh
cargo build --release
cargo run --release -- transcode input.mkv -o output.mp4
```

### Optional features

| Feature     | Adds |
|-------------|------|
| `nvidia`    | NVENC AV1 hardware **encoder** via [`shiguredo_nvcodec`](https://github.com/shiguredo/nvcodec-rs) (Apache-2.0). Needs libclang at build time; dlopens CUDA at runtime. NVIDIA Ada+. |
| `amd`       | AMF AV1 hardware **encoder** via [`shiguredo_amf`](https://github.com/shiguredo/amf-rs) (Apache-2.0). Needs libclang at build time; dlopens the AMF runtime. AMD RDNA3+. |
| `qsv`       | Intel QuickSync / oneVPL hardware decode + encode via [`shiguredo_vpl`] (Apache-2.0; needs CMake + libvpl). Intel Arc / Meteor Lake+. |
| `ffmpeg`    | libavcodec as the primary decode path (full software catalogue + Vulkan/NVDEC/D3D11/VAAPI hwaccel + AV1 software encode). Needs FFmpeg ≥7.0 dev libs + LLVM/libclang. |
| `thumbnail` | `rivet::thumbnail::generate_thumbnail` — capture a frame and encode an AVIF still (pulls `ravif`/rav1e). |

> The hardware **encoders** are opt-in features: the NVENC, AMF, and QSV
> backends are wrappers over the Apache-2.0 `shiguredo_{nvcodec,amf,vpl}`
> crates (the hand-rolled FFI mirrors were retired). A default build has no
> hardware encoder — enable `nvidia` / `amd` / `qsv` (or `ffmpeg`) for your
> target silicon. NVIDIA **decode** (NVDEC) remains built-in.
>
> ⚠️ Platform note: the three `shiguredo_*` crates bindgen the vendor SDK
> headers and compile on **Linux** (the production / Docker target) but **not
> on a Windows MSVC host**. Under the MSVC ABI a non-negative C enum is signed
> (`int` → `i32`); under the Linux ABI it is `unsigned int` (→ `u32`), which is
> what the crates expect. So build the `nvidia` / `amd` / `qsv` features on
> Linux (or in the Docker image); on a Windows dev box use the `ffmpeg` feature
> or leave them off. (Each feature needs `libclang` at build time —
> `LIBCLANG_PATH`.)

```sh
cargo build --release --features qsv
cargo build --release --features ffmpeg
```

## License

Apache-2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE). The NOTICE file
credits the Apache-2.0 third-party components used for platform-specific GPU
codec access (`shiguredo_nvcodec` / `shiguredo_amf` / `shiguredo_vpl`).
