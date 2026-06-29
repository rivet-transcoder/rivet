# rivet CLI reference

> What happens under the hood for any of these commands — demux → decode-once
> pump → multi-GPU encode → mux — is in [pipeline & architecture](pipeline.md).

The `rivet` binary has these subcommands: [`transcode`](#rivet-transcode),
[`probe`](#rivet-probe), [`devices`](#rivet-devices),
[`capabilities`](#rivet-capabilities), [`pipe`](#rivet-pipe),
[`batch`](#rivet-batch) (feature `batch`), [`ipc`](#rivet-ipc) (feature `ipc`),
and [`serve`](#rivet-serve) (feature `server`). Build it with:

```sh
cargo build --release                     # CPU/GPU decode + GPU encode tiers
cargo build --release --features ffmpeg   # + libavcodec software/hwaccel fallback
cargo build --release --features nvidia   # + NVENC AV1 encoder (Windows or Linux)
```

The binary is at `target/release/rivet`. Run `rivet --help` or
`rivet <command> --help` for generated usage at any time.

> rivet encodes **AV1** (default, royalty-clean), **H.264**, or **H.265** —
> select with `--codec av1|h264|h265`. The output container is MP4 (single file)
> or CMAF/HLS (segmented); all three codecs work in both. See
> [the compatibility matrix](../README.md#compatibility-matrix) for codecs in.

---

## `rivet transcode`

```
rivet transcode <INPUT> [OPTIONS]
```

Transcodes `<INPUT>` (any supported container/codec) to AV1 (default), H.264, or
H.265 — pick with `--codec`.

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
| `--filter <CHAIN>` | e.g. `crop=1280:720,hflip` | Video filter chain applied before scaling — see [Video filters](filters.md). |
| `--trim-start <S>` | seconds | **Splice/trim:** keep from this time. The output is re-based to zero. Trimmed jobs take the serial encode path. |
| `--trim-end <S>` | seconds | **Splice/trim:** keep until this time. The kept range is `[start, end)`, exact at any frame rate. To *join* clips, use [`rivet splice`](#rivet-splice). |
| `--codec <CODEC>` | `av1` *(default)*, `h264`, `h265` | Output video codec. `av1` is royalty-clean (the project default); `h264`/`h265` are for legacy-player compatibility (patent-licensing caveats). All three work for **single-file MP4 and CMAF/HLS**. H.264/H.265 are encoded on **NVENC** (validated on RTX 3090) + **QSV** (validated on Intel Arc); AMF and the `ffmpeg`-wrapper H.264/H.265 paths are a follow-up. |

### GPU selection

| Flag | Description |
|------|-------------|
| `--gpu <N>` | Pin encode/decode to GPU index `N` (implies single-GPU). |
| `--single-gpu` | Encode serially on one GPU instead of chunking across all GPUs. Without `--gpu`, picks the first GPU. |
| `--gpu-family <VENDOR>` | `nvidia` \| `amd` \| `intel` — use only that vendor's GPUs (e.g. ignore an integrated GPU). |
| `--decode-gpu <auto\|fastest\|N>` | Decode-pump GPU policy (default `auto`): `auto` follows the encode policy; a GPU index `N` pins decode to that card (e.g. decode on an iGPU while the dGPUs encode); `fastest` benchmarks every decode-capable GPU on a prefix of the input and pins to the quickest (a no-op on single-GPU hosts). Also available on `rivet splice`. |
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
`ffmpeg`. It's web-safe 4:2:0 10-bit, HDR-tagged in the container
(`colr`/`mdcv`/`clli`). 10-bit applies to **`--codec av1`** (AV1 Main profile) and
**`--codec h265`** (HEVC Main 10, on NVENC + QSV); **`--codec h264` is 8-bit
only** (no hardware Hi10P), so a 10-bit + `h264` combination is rejected. The
transcode fails fast with a clear message if you request something the build
can't produce.

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

# Benchmark decoders up front and decode on the fastest GPU (multi-GPU hosts)
rivet transcode input.mkv -o out.mp4 --decode-gpu fastest

# HDR10 passthrough (needs a nvidia/amd hardware or ffmpeg build)
rivet transcode input.mkv -o out.mp4 --color hdr10 --pixel-format 10bit

# Splice/trim: cut a single input to [2s, 7s)
rivet transcode input.mkv -o cut.mp4 --trim-start 2 --trim-end 7
```

---

## `rivet splice`

**Concatenate** (and per-clip **trim**) several inputs into one output. Clips are
joined in order; each is decoded with its own decoder, trimmed to its window,
and the kept frames are re-encoded into one continuous, zero-based timeline (the
muxer numbers frames by count, so the join is gap-free with no PTS rewriting).
Because everything is re-encoded to a uniform output, the inputs **may differ**
in codec, resolution, or color — output config follows the **first** clip. Audio
is trimmed per clip and concatenated to match. Outputs a single MP4
(`--mode single`, the default) or a CMAF/HLS package (`--mode hls`) — for HLS the
spliced frame stream feeds the same multi-GPU engine as a normal ladder, so
segments stay keyframe-aligned across the join.

```
rivet splice -o <OUTPUT> [OPTIONS] <CLIP>...
```

`<OUTPUT>` is a file for `--mode single`, or a directory for `--mode hls`. Each
`<CLIP>` is a path, or `PATH@START-END` to trim it (seconds, either side
optional). `@` is the separator so a Windows drive `C:\…` is unambiguous:

| Clip spec | Meaning |
|-----------|---------|
| `a.mp4` | the whole clip |
| `a.mp4@2-7` | seconds `[2, 7)` |
| `a.mp4@2-` | from 2 s to the end |
| `a.mp4@-7` | from the start to 7 s |

| Flag | Values / default | Description |
|------|------------------|-------------|
| `-o`, `--output <PATH>` | required | Output MP4 file (`single`) or directory (`hls`). |
| `--mode <MODE>` | `single` *(default)*, `hls` | Output shape: one MP4, or a CMAF/HLS package. |
| `--segment-seconds <S>` | default `4.0` | HLS target segment length (`--mode hls` only). |
| `--codec <CODEC>` | `av1` *(default)*, `h264`, `h265` | Output video codec (as for `transcode`). |
| `--crf <N>` | encoder-native | Constant rate factor. |
| `--audio <POLICY>` | `auto` *(default)*, `opus`, `drop` | Audio handling. |

### Examples

```sh
# Join three clips end-to-end
rivet splice -o out.mp4 intro.mp4 body.mkv outro.mov

# Join with per-clip trims (first 5 s of A, then 10–20 s of B, then all of C)
rivet splice -o out.mp4 a.mp4@0-5 b.mp4@10-20 c.mp4 --codec h265

# A single trimmed clip is just a trim (same as transcode --trim-*)
rivet splice -o cut.mp4 a.mp4@2-7

# Concatenate straight into an HLS package
rivet splice -o out_hls/ --mode hls a.mp4 b.mp4 c.mp4 --codec h265
```

> The library equivalents are `rivet::run_splice_job(Vec<Clip>, &spec, …)` and
> `OutputSpec::with_trim(start, end)` for the single-input case.

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
           [--max-fps F] [--width W] [--height H] [--gpu I] [--filter CHAIN]
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

## `rivet batch`

```
cargo build --release --features batch   # opt-in
rivet batch <MANIFEST> [--dry-run] [--stop-on-error]
```

Convert **many files in one run** from a YAML or JSON **manifest** — you list the
files (and how), rivet does them. Each job is an input (file or glob), an output,
and any transcode setting, on top of optional shared `defaults`. `--dry-run`
parses + expands globs + lists the planned jobs without converting; `--stop-on-error`
aborts on the first failure (default keeps going and exits non-zero if any failed).

```sh
rivet batch jobs.yaml --dry-run
rivet batch jobs.yaml
```

```yaml
output_dir: out
defaults: { crf: 28, color: sdr }
jobs:
  - input: in/a.mkv
    output: out/a.mp4
    crf: 24
  - input: "clips/*.mp4"   # glob -> one job per file -> out/<name>.mp4
    output: out/
```

**Full DSL reference: [batch.md](batch.md)** — every key, the output-path rules,
glob inputs, defaults merge, and JSON examples. A ready-to-edit manifest is in
[`examples/batch.yaml`](../examples/batch.yaml) / [`.json`](../examples/batch.json).

## `rivet ipc`

```
cargo build --release --features ipc   # opt-in; the subcommand only exists in an ipc build
rivet ipc --socket <PATH>
```

Run a **Unix-domain-socket** server (opt-in `ipc` feature; Unix only at runtime)
so a long-running application can stream jobs in and out without spawning a
process per file or going through HTTP. `rivet pipe` (stdin/stdout streaming) is
always available and needs no feature.
Bind a socket, then for **each connection**: the client optionally writes a
**settings header line**, then the input media, **half-closes** its write side
(signals end-of-input), and reads the transcoded AV1/MP4 back until EOF. One
thread per connection; the process-wide GPU pool serializes the actual GPU work,
so concurrent clients simply queue.

**Settings header** (optional): if the stream begins with `#rivet`, the first
line is parsed as space-separated `key=value` settings and stripped before
decode. The keys are the shared `TranscodeSettings` vocabulary — the same names
as the CLI flags (`crf` `speed` `audio` `color` `bit-depth` `max-fps` `width`
`height` `gpu` `gpu-family` `single-gpu` `decode-gpu` `seam` `filter`). Real container
magic bytes never start with `#rivet`, so a raw media stream without a header
just gets the defaults. (A single socket connection produces one MP4, so
`mode=hls`/multi-rung isn't supported here — use the HTTP API for that.)

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
