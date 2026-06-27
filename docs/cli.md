# rivet CLI reference

> What happens under the hood for any of these commands — demux → decode-once
> pump → multi-GPU encode → mux — is in [pipeline & architecture](pipeline.md).

The `rivet` binary has seven subcommands: [`transcode`](#rivet-transcode),
[`probe`](#rivet-probe), [`devices`](#rivet-devices),
[`capabilities`](#rivet-capabilities), [`pipe`](#rivet-pipe),
[`ipc`](#rivet-ipc), and [`serve`](#rivet-serve). Build it with:

```sh
cargo build --release                     # CPU/GPU decode + GPU encode tiers
cargo build --release --features ffmpeg   # + libavcodec software/hwaccel fallback
cargo build --release --features nvidia   # + NVENC AV1 encoder (Windows or Linux)
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
| `--seam-mode <MODE>` | `parallel` *(default)* \| `constqp` \| `serial` — how the multi-GPU **single-file** path keeps quality flat across the chunk seams it stitches. |

See [GPU scheduling](../README.md#gpu-scheduling-the-rung-benefit) for how
`AllGpus` / `SingleGpu` / `Family` actually distribute work.

#### Chunk seams (`--seam-mode`)

When more than one GPU encodes a **single file**, each rung is chunked at GOP
boundaries, encoded in parallel, and the AV1 packets are stitched into one MP4.
Each chunk is an independent IDR-led GOP, so the result always plays — but each
chunk's rate control is independent, so quality can step at the ~2 s seams. AMD
(AMF) and Intel (QSV) chunks are constant-QP and already seam-flat; this knob
chiefly governs **NVENC** (which otherwise runs VBR per chunk):

| Mode | Seams | Speed | Notes |
|------|-------|-------|-------|
| `parallel` *(default)* | possible mild NVENC steps | fastest (all GPUs) | each chunk uses its encoder's normal rate control |
| `constqp` | flat | fast (all GPUs) | forces constant-QP; the QP is derived from the quality target, so quality still tracks it |
| `serial` | none | slower (one GPU) | one encoder for the whole file — seam-free and quality-accurate; HLS still uses every GPU |

(Single-GPU hosts, `--single-gpu`/`--gpu`, and HLS jobs are unaffected — HLS
segments are independent files by design.)

### Color & bit depth

The decode pump tonemaps only when the policy says so — it never decides on its
own:

| `--color` | Output | Bit depth | Needs |
|-----------|--------|-----------|-------|
| `sdr` *(default)* | tonemap HDR → SDR BT.709 | 8-bit | any encoder |
| `passthrough` | source color verbatim | source | 10-bit encoder if source is 10-bit |
| `hdr10` | BT.2020 + PQ | 10-bit | a 10-bit encoder (below) |
| `hlg` | BT.2020 + HLG | 10-bit | a 10-bit encoder (below) |

10-bit / HDR output works on **hardware** with the `nvidia` (NVENC), `amd` (AMF),
or `qsv` (oneVPL P010) feature — no `ffmpeg` required — or in software with
`ffmpeg`. It's web-safe AV1 Main-profile 4:2:0 10-bit, HDR-tagged in the
container (`colr`/`mdcv`/`clli`). The transcode fails fast with a clear message
if you request something the build can't produce.

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

## `rivet devices`

```
rivet devices [--json]
```

List the GPUs rivet detects on this host — vendor, name, generation, VRAM, PCI
address, and (NVIDIA only, via NVML) a live load snapshot (GPU / encoder /
decoder utilization, memory, temperature). `--json` emits
`{ "gpus": [ { index, vendor, name, generation, vram_mib, pci, load? } ] }`.

```sh
rivet devices
rivet devices --json
```

This is **hardware inventory** — what's plugged in. What this *build* can actually
do with it is [`rivet capabilities`](#rivet-capabilities) (it depends on which
GPU feature the binary was compiled with).

## `rivet capabilities` (alias `caps`)

```
rivet capabilities [--json]
rivet caps [--json]
```

Report what this **build + host** can do:

- **Encode** — AV1 4:2:0 (the only output codec): the compiled backends
  (`nvenc` / `amf` / `qsv` / `ffmpeg`), the max bit depth (8 or 10), and whether
  HDR (PQ/HLG, BT.2020) is producible. Driven by `codec::encode::build_output_caps()`
  — the same query `OutputSpec::validate()` consults.
- **Decode** — a codec → backends table (which of `nvdec` / `amf` / `qsv` /
  `ffmpeg` decode `h264` / `hevc` / `vp8` / `vp9` / `av1` / `mpeg2` / `mpeg4` /
  `prores`).
- **Devices** — a one-line summary of the detected GPUs.

A backend only appears if its **feature was compiled in** (`--features nvidia`
etc.); the actual silicon (e.g. NVENC AV1 needs Ada+) is verified at encode time.

```sh
cargo build --release --features qsv
rivet capabilities            # Encode: qsv 10-bit HDR · Decode: h264/hevc/av1/vp9 → qsv
rivet caps --json
```

---

## `rivet pipe`

```
rivet pipe [--crf N] [--speed N] [--audio auto|opus|drop]
           [--color sdr|hdr10|hlg|passthrough] [--bit-depth auto|8bit|10bit]
           [--max-fps F] [--width W] [--height H] [--gpu I]
```

Stream a transcode through standard I/O: read media from **stdin**, write the
AV1/MP4 to **stdout** (progress goes to stderr so stdout stays clean). With no
flags it's the single-file default (source resolution, AV1 + AAC/Opus
passthrough, 8-bit SDR). The flags override per job — `--width/--height` scale,
`--color/--bit-depth` set HDR/depth, `--crf/--speed` set quality:

```sh
cat input.mkv | rivet pipe > output.mp4                       # defaults
cat input.mkv | rivet pipe --crf 28 --width 1280 --height 720 > out.mp4
ffmpeg -i src.mov -f matroska - | rivet pipe --color hdr10 | ./my-uploader
```

Single MP4 only — for an HLS package or a multi-rung ladder use
[`transcode`](#rivet-transcode) with a directory output, or the
[HTTP API](api.md).

## `rivet ipc`

```
rivet ipc --socket <PATH>
```

Run a **Unix-domain-socket** server (Unix only) so a long-running application can
stream jobs in and out without spawning a process per file or going through HTTP.
Bind a socket, then for **each connection**: the client optionally writes a
**settings header line**, then the input media, **half-closes** its write side
(signals end-of-input), and reads the transcoded AV1/MP4 back until EOF. One
thread per connection; the process-wide GPU pool serializes the actual GPU work,
so concurrent clients simply queue.

**Settings header** (optional): if the stream begins with `#rivet`, the first
line is parsed as space-separated `key=value` settings and stripped before
decode; the keys mirror the `pipe` flags
(`crf` `speed` `audio` `color` `bit-depth` `max-fps` `width` `height` `gpu`).
Real container magic bytes never start with `#rivet`, so a raw media stream
without a header just gets the defaults.

```
#rivet crf=28 color=hdr10 width=1280 height=720\n
<media bytes…>
```

```sh
rivet ipc --socket /tmp/rivet.sock &
# any client that does write → shutdown(WR) → read works, e.g. socat (no header):
socat - UNIX-CONNECT:/tmp/rivet.sock < input.mkv > output.mp4
```

A minimal client with settings (Python):

```python
import socket
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
s.connect("/tmp/rivet.sock")
s.sendall(b"#rivet crf=28 width=1280 height=720\n")   # optional settings header
s.sendall(open("input.mkv", "rb").read())
s.shutdown(socket.SHUT_WR)                             # end-of-input
out = b"".join(iter(lambda: s.recv(65536), b""))
open("output.mp4", "wb").write(out)                    # AV1/MP4
```

Single MP4 per connection. On **Windows** `rivet ipc` is unavailable — use
[`rivet pipe`](#rivet-pipe) (stdin/stdout) or [`rivet serve`](#rivet-serve)
(HTTP).

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
| `DISABLE_FFMPEG` | Skip the `ffmpeg` decode tier (only relevant in an `ffmpeg` build). |
| `FFMPEG_HWACCEL` | Override the `ffmpeg` hwaccel preference (e.g. `cuda`, `vaapi`). |
| `RIVET_TEST_MEDIA` | Integration tests: directory of real media to run against. |
