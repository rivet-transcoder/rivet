# rivet

A modular, GPU-accelerated video transcoding **library** and **command-line
tool**, written in Rust.

`rivet` takes an arbitrary input file and produces a single **AV1** video +
**Opus / AAC** audio stream muxed into **MP4** (or a segmented **CMAF-HLS**
package via the lower-level crates). It is built from clean-room demuxers,
muxers, and hardware-codec dispatch — **no FFmpeg required** by default
(FFmpeg is available as an optional decode backend behind a feature flag).

## Why "rivet"

It fastens generic transcoding logic that grew up inside a video-processing
microservice into a standalone, reusable component — a library you can embed
and a CLI you can run.

## Output policy

The output target is intentionally fixed and royalty-clean:

| Stream | Codec                                            | Container |
|--------|--------------------------------------------------|-----------|
| Video  | AV1                                              | MP4       |
| Audio  | Opus (transcoded) or AAC/Opus/AC-3/E-AC-3 (passthrough) | MP4 |

Input may be any container/codec the crates can demux + decode:

- **Containers:** MP4/MOV, MKV/WebM, MPEG-TS, AVI (incl. OpenDML >1 GiB).
- **Video:** H.264, HEVC, VP8/VP9, AV1, MPEG-2, MPEG-4 Part 2, ProRes.
- **Audio:** AAC, Opus, AC-3, E-AC-3 (passthrough); MP3, Vorbis (→ Opus).

## Crates

| Crate       | Responsibility |
|-------------|----------------|
| `codec`     | Frame types, pixel formats, GPU detection, decode (FFmpeg / NVDEC / QSV + legacy fallbacks), AV1 encode (rav1e / NVENC / AMF / QSV), colorspace + HDR→SDR tonemap, audio decode/encode, probe. |
| `container` | Demuxers (MP4/MOV/MKV/WebM/TS/AVI), AV1 MP4 muxer with audio, fragmented-MP4 (CMAF) writers, HLS playlist generation, bounded-RSS streaming demuxer. |
| `rivet`     | Ergonomic facade: single-file `transcode` + `probe`, plus the `rivet` CLI binary. Re-exports `codec` and `container` for lower-level access. |

## Library usage

```toml
[dependencies]
rivet = { git = "https://github.com/elyerinfox/rivet" }
```

```rust
// Transcode a file to AV1/Opus MP4.
let outcome = rivet::transcode_file("input.mkv", "output.mp4")?;
println!("{} frames out", outcome.frames_processed);

// Probe without transcoding.
let info = rivet::probe_file("input.mkv")?;
println!("{}x{} {}", info.width, info.height, info.video_codec);
```

For finer control (custom encoder configs, CMAF segments, per-frame access),
reach through the re-exported component crates:

```rust
use rivet::codec::encode::{select_encoder, EncoderConfig};
use rivet::container::mux::Av1Mp4Muxer;
```

## CLI usage

```sh
# Transcode (output defaults to <input>.av1.mp4)
rivet transcode input.mkv -o output.mp4

# Probe
rivet probe input.mkv
rivet probe input.mkv --json
```

Set `RUST_LOG=debug` for verbose logging. Force an encoder backend with
`TRANSCODE_ENCODER_BACKEND=nvenc|amf|qsv`.

## Building

Hardware-codec backends (NVENC/NVDEC, QSV via Intel oneVPL) link native
libraries, so the default build needs a C toolchain plus:

- **nasm** — x86 assembly for the codec stack.
- **CMake** + a C/C++ compiler — builds Intel oneVPL (`shiguredo_vpl`).

On Windows the project links the static MSVC CRT (see `.cargo/config.toml`).

```sh
cargo build --release
cargo run --release -- transcode input.mkv -o output.mp4
```

### Optional: FFmpeg decode backend

Enabling the `ffmpeg` feature routes decode through libavcodec (the full
codec catalogue + Vulkan/NVDEC/D3D11/VAAPI hwaccel) as the primary path,
with the native backends as fallback. Requires FFmpeg ≥7.0 dev libraries and
LLVM/libclang (bindgen). See `crates/codec/Cargo.toml` for the full setup.

```sh
cargo build --release --features ffmpeg
```

## License

MIT — see [LICENSE](LICENSE).
