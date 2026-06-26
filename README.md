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

Default builds decode on the GPU. Software decode (and ProRes) requires the
optional `ffmpeg` feature.

| Codec          | NVDEC (NVIDIA) | QSV (Intel, `qsv` feature) | FFmpeg (`ffmpeg` feature) |
|----------------|:--------------:|:--------------------------:|:-------------------------:|
| H.264 / AVC    | ✅             | ✅                         | ✅ |
| HEVC / H.265   | ✅             | ✅                         | ✅ |
| VP8            | ✅             | —                          | ✅ |
| VP9            | ✅             | ✅                         | ✅ |
| AV1            | ✅             | ✅                         | ✅ |
| MPEG-2         | ✅             | —                          | ✅ |
| MPEG-4 Part 2  | ✅             | —                          | ✅ |
| ProRes         | —              | —                          | ✅ |

10-bit / HDR sources decode and are tonemapped to 8-bit SDR BT.709 before
encode (single-output policy).

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

| Feature  | Adds |
|----------|------|
| `qsv`    | Intel QuickSync / oneVPL hardware decode + encode (off by default; needs CMake + libvpl, useful on Intel Arc / Meteor Lake+). |
| `ffmpeg` | libavcodec as the primary decode path (full software catalogue + Vulkan/NVDEC/D3D11/VAAPI hwaccel + AV1 software encode). Needs FFmpeg ≥7.0 dev libs + LLVM/libclang. |

```sh
cargo build --release --features qsv
cargo build --release --features ffmpeg
```

## License

MIT — see [LICENSE](LICENSE).
