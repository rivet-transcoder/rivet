# rivet CLI reference

The `rivet` binary has three subcommands: [`transcode`](#rivet-transcode),
[`probe`](#rivet-probe), and [`serve`](#rivet-serve). Build it with:

```sh
cargo build --release                     # CPU/GPU decode + GPU encode tiers
cargo build --release --features ffmpeg   # + libavcodec software/hwaccel fallback
cargo build --release --features nvidia   # + NVENC AV1 encoder (Linux; see Building)
```

The binary is at `target/release/rivet`. Run `rivet --help` or
`rivet <command> --help` for generated usage at any time.

> rivet encodes **AV1** only (the locked, royalty-clean target). The output
> container is MP4 (single file) or CMAF/HLS (segmented). See
> [the compatibility matrix](../README.md#compatibility-matrix) for codecs in.

---

## `rivet transcode`

```
rivet transcode <INPUT> [OPTIONS]
```

Transcodes `<INPUT>` (any supported container/codec) to AV1.

### Arguments

| Argument | Description |
|----------|-------------|
| `<INPUT>` | Input media file. Container/codec is auto-detected. |

### Options

| Flag | Values / default | Description |
|------|------------------|-------------|
| `-o`, `--output <PATH>` | default `<input>.av1.mp4` | Output file (single mode, one rung) or **directory** (multi-rung single mode, or HLS). |
| `--mode <MODE>` | `single` *(default)*, `hls` | Output shape: one self-contained MP4 per rung, or a CMAF/HLS package. |
| `--rung <WxH>` | repeatable | A ladder rung, e.g. `--rung 1920x1080 --rung 1280x720`. Omit for a single rung at the source resolution. |
| `--ladder` | flag | Auto-derive a standard ABR ladder from the source resolution (instead of `--rung`). |
| `--max-short-side <N>` | default `1080` | With `--ladder`, cap the tallest rung's short side. |
| `--segment-seconds <S>` | default `4.0` | HLS target segment length (segments still break on keyframes). |
| `--crf <N>` | encoder-native | Constant rate factor (lower = better quality). Omit to derive from the quality target. |
| `--speed <N>` | encoder-native | Encoder speed preset. |
| `--audio <POLICY>` | `auto` *(default)*, `opus`, `drop` | `auto`: passthrough AAC/Opus/AC-3/E-AC-3, transcode MP3/Vorbis to Opus, drop the rest. `opus`: force Opus. `drop`: video only. |
| `--max-fps <F>` | — | Cap the output frame rate (source cadence otherwise preserved). |
| `--color <POLICY>` | `sdr` *(default)*, `hdr10`, `hlg`, `passthrough` | Output color / tonemap policy — see [Color & bit depth](#color--bit-depth). |
| `--pixel-format <FMT>` | `auto` *(default)*, `8bit`, `10bit` | Output luma bit depth. |

### GPU selection

| Flag | Description |
|------|-------------|
| `--gpu <N>` | Pin encode/decode to GPU index `N` (implies single-GPU). |
| `--single-gpu` | Encode serially on one GPU instead of chunking across all GPUs. Without `--gpu`, picks the first GPU. |
| `--gpu-family <VENDOR>` | `nvidia` \| `amd` \| `intel` — use only that vendor's GPUs (e.g. ignore an integrated GPU). |
| `--decode-gpu <N>` | Pin the **decode pump** to GPU `N`, independent of the encode policy (e.g. decode on an iGPU while the dGPUs encode). Default: follows the encode policy. |

See [GPU scheduling](../README.md#gpu-scheduling-the-rung-benefit) for how
`AllGpus` / `SingleGpu` / `Family` actually distribute work.

### Color & bit depth

The decode pump tonemaps only when the policy says so — it never decides on its
own:

| `--color` | Output | Bit depth | Needs |
|-----------|--------|-----------|-------|
| `sdr` *(default)* | tonemap HDR → SDR BT.709 | 8-bit | any encoder |
| `passthrough` | source color verbatim | source | 10-bit encoder if source is 10-bit |
| `hdr10` | BT.2020 + PQ | 10-bit | a 10-bit encoder (below) |
| `hlg` | BT.2020 + HLG | 10-bit | a 10-bit encoder (below) |

10-bit / HDR output works on **hardware** with the `nvidia` (NVENC) or `amd`
(AMF) feature — no `ffmpeg` required — or in software with `ffmpeg`. QSV stays
8-bit (`shiguredo_vpl` has no P010). It's web-safe AV1 Main-profile 4:2:0 10-bit,
HDR-tagged in the container (`colr`/`mdcv`/`clli`). The transcode fails fast with
a clear message if you request something the build can't produce.

### Output layout

- **single** — one MP4 per rung. One rung → the `-o` file (faststart AV1 + audio).
  Multiple rungs → `-o` must be a directory; files are named per rung.
- **hls** — `-o` is the asset root: `master.m3u8`, an `audio/` rendition group,
  and `video/<height>p/{init.mp4, seg-*.m4s, playlist.m3u8}` per rung,
  segment-aligned across the ladder for clean ABR.

### Examples

```sh
# Single MP4 at the source resolution
rivet transcode input.mkv -o output.mp4

# Explicit 3-rung ladder → a directory of MP4s
rivet transcode input.mkv -o out_dir/ --rung 1920x1080 --rung 1280x720 --rung 640x360

# Auto ABR ladder capped at 1080p short side
rivet transcode input.mkv -o out_dir/ --ladder --max-short-side 1080

# CMAF/HLS package, 4 s segments
rivet transcode input.mkv -o hls_dir/ --mode hls --ladder --segment-seconds 4

# Quality + audio + frame-rate knobs
rivet transcode input.mkv -o out.mp4 --crf 28 --speed 6 --audio opus --max-fps 30

# Pin to one GPU / one vendor / decode elsewhere
rivet transcode input.mkv -o out.mp4 --gpu 1
rivet transcode input.mkv -o out.mp4 --gpu-family nvidia --decode-gpu 0

# HDR10 passthrough (needs a nvidia/amd hardware or ffmpeg build)
rivet transcode input.mkv -o out.mp4 --color hdr10 --pixel-format 10bit
```

---

## `rivet probe`

```
rivet probe <INPUT> [--json]
```

Inspect a file without transcoding. `--json` emits a machine-readable object
(`video_codec`, `width`, `height`, `frame_rate`, `duration`); otherwise a human
summary is printed.

```sh
rivet probe input.mkv
rivet probe input.mkv --json
```

---

## `rivet serve`

```
rivet serve [--addr <ADDR>]
```

Runs the HTTP transcode API (requires a `--features server` build). `--addr`
defaults to `127.0.0.1:8080`. See the [HTTP API reference](api.md) for endpoints.

```sh
cargo build --release --features server,nvidia
rivet serve --addr 0.0.0.0:8080
```

---

## Environment variables

| Variable | Effect |
|----------|--------|
| `RUST_LOG` | Log filter, e.g. `RUST_LOG=debug` or `RUST_LOG=rivet=info`. |
| `TRANSCODE_ENCODER_BACKEND` | Force an encoder backend: `nvenc` \| `amf` \| `qsv`. |
| `DISABLE_NVDEC` | Skip NVDEC for every codec (fall through to the next decode tier). |
| `DISABLE_NVDEC_<CODEC>` | Skip NVDEC for one family, e.g. `DISABLE_NVDEC_AV1=1`. |
| `DISABLE_VULKAN_VIDEO` | Disable the Vulkan Video decode tier (where present). |
| `RIVET_TEST_MEDIA` | Integration tests: directory of real media to run against. |
