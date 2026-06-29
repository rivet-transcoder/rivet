# Configuring a transcode — the complete `OutputSpec` guide

Everything you can configure for a rivet job lives on one struct,
[`OutputSpec`](../crates/rivet/src/spec.rs). You **build** it (constructor +
chained `with_*` setters), optionally **validate** it, then **run** it. This page
documents every knob; for the internals see [pipeline & architecture](pipeline.md),
and for the CLI equivalents see the [CLI reference](cli.md).

```rust
use rivet::{OutputSpec, Rung, Quality, AudioPolicy, EncodePolicy,
            ChunkSeamMode, PerceptualTarget, run_job_blocking, fn_sink};
use rivet::progress::RungProgress;
use std::sync::Arc;

// A 3-rung single-file ladder, fully specified.
let spec = OutputSpec::single_file(vec![
    Rung::new(1920, 1080).with_quality(Quality::crf(28)),
    Rung::new(1280, 720).with_quality(Quality::target(PerceptualTarget::Standard)),
    Rung::new(854, 480),                              // default quality
])
.with_audio(AudioPolicy::Auto)                        // passthrough / transcode / drop
.with_max_frame_rate(30.0)                            // cap output cadence
.web_sdr()                                            // color preset: BT.709 8-bit SDR
.encode_policy(EncodePolicy::AllGpus)                 // chunk-encode across every GPU
.chunk_seam_mode(ChunkSeamMode::ParallelConstQp);     // keep seams quality-flat

spec.validate()?;                                     // fail fast on incoherent specs

let bytes = std::fs::read("input.mkv")?;
let sink = Arc::new(fn_sink(|p: RungProgress| {
    println!("{:<6} {:?} {:>5.1}%  {} frames", p.label, p.status, p.percent, p.frames_done);
}));
let out = run_job_blocking(&bytes, &spec, Some("out_dir".as_ref()), sink)?;
```

> **Just want one file in, one file out?** Skip the spec entirely:
> `rivet::transcode_file("in.mkv", "out.mp4")?` uses sensible defaults
> (source-resolution single rung, AAC/Opus passthrough, 8-bit SDR, all GPUs).

---

## 1. Construct — the output shape

| Constructor | Output |
|-------------|--------|
| `OutputSpec::single_file(rungs)` | One self-contained faststart **MP4** per rung (video + audio; AV1 by default — set `with_video_codec` for H.264/H.265). |
| `OutputSpec::hls(rungs, segment_seconds)` | A segmented **CMAF/HLS** package: `master.m3u8` + an audio rendition group + `video/<h>p/{init.mp4, seg-*.m4s, playlist.m3u8}` per rung, segment-aligned for clean ABR. |

`rungs` is a `Vec<Rung>` (next section). `segment_seconds` is the HLS target
segment length (segments still break on keyframes). The constructor wires the
matching `Container` + `Muxer` + `OutputMode` for you.

---

## 2. The ladder — rungs & quality

A [`Rung`](../crates/rivet/src/spec.rs) is one rendition: a target size + a
per-rung [`Quality`](../crates/rivet/src/spec.rs).

```rust
Rung::new(1280, 720)                       // auto label "720p", default quality
    .with_quality(Quality::crf(28))        // or .with_quality(Quality::target(..))
    .with_label("hd")                      // override the auto label
```

| `Rung` method | Effect |
|---------------|--------|
| `Rung::new(width, height)` | A rung at `width × height`; label auto-set to `"<short-side>p"`, default quality. |
| `.with_quality(Quality)` | Set the per-rung encoder quality. |
| `.with_label(impl Into<String>)` | Override the auto label. |
| `.short_side()` | The "p" number (`min(width, height)`). |

Public fields: `width`, `height`, `label`, `quality`.

### Quality

```rust
Quality::crf(28)                           // constant rate factor (lower = better)
Quality::target(PerceptualTarget::High)    // perceptual target instead of a CRF
```

| `Quality` field | Type | Meaning |
|-----------------|------|---------|
| `crf` | `Option<u8>` | Constant rate factor, encoder-native (rav1e/NVENC 0..=255). `None` → derive from `target`. |
| `speed_preset` | `Option<u8>` | Encoder-native speed preset. `None` → derive from `tier`. |
| `target` | `QualityTarget` | Perceptual target (used when `crf` is `None`). |
| `tier` | `SpeedTier` | Speed/efficiency tier (used when `speed_preset` is `None`). |
| `keyframe_interval` | `Option<u32>` | GOP length in frames. `None` → `2 × fps` (a 2-second GOP). |

`Quality::crf` / `Quality::target` are the two constructors; set the rest with
struct-update syntax, e.g. `Quality { tier: Speed::Archive, keyframe_interval:
Some(120), ..Quality::crf(30) }`.

- **`QualityTarget`** (re-exported as `PerceptualTarget`): `VisuallyLossless`,
  `High`, `Standard`, `Low`, `Vmaf(u8)` (target a specific VMAF score).
- **`SpeedTier`** (re-exported as `Speed`): `Draft` (fastest), `Standard`,
  `Archive` (slowest/most efficient).

### Auto ladder

Don't want to hand-write rungs? Derive a standard ABR ladder from the source:

```rust
let rungs = rivet::standard_ladder(source_w, source_h, /* max_short_side */ 1080);
let spec = OutputSpec::single_file(rungs);
```

It snaps to standard short sides (2160/1440/1080/720/480/360/240), preserves
aspect ratio, even-aligns dims, and caps the top rung.

---

## 3. Audio — `with_audio(AudioPolicy)`

| `AudioPolicy` | Behavior |
|---------------|----------|
| `Auto` *(default)* | Passthrough AAC / Opus / AC-3 / E-AC-3 verbatim; transcode MP3 / Vorbis → Opus; drop anything else. |
| `ForceOpus` | Always produce Opus (passthrough Opus, transcode everything else). |
| `Drop` | Video-only output. |

```rust
spec.with_audio(AudioPolicy::ForceOpus)
```

---

## 4. Color & bit depth

Two orthogonal axes. Most callers use a **preset**; the low-level setters are
there when you need them.

```rust
spec.web_sdr()       // BT.709 8-bit SDR, tonemap any HDR source down (the default)
spec.hdr10()         // BT.2020 + PQ, 10-bit, no tonemap
spec.hlg()           // BT.2020 + HLG, 10-bit, no tonemap
spec.passthrough()   // keep the source's color + bit depth verbatim
```

Under the presets are exactly **two** methods:

| Method | Sets | Values |
|--------|------|--------|
| `with_color(ColorPolicy)` | **gamut + transfer + tonemap decision** | `TonemapToSdr` *(default)* · `Passthrough` · `Hdr10` · `Hlg` |
| `with_bit_depth(BitDepth)` | **bits per sample** | `Auto` *(default — follow the color policy)* · `EightBit` · `TenBit` |

There is intentionally **no** `with_gamut` / `with_transfer` / `with_color_space`
— `ColorPolicy` bundles them because only a few combinations are web-safe:

- **Gamut** = the color *primaries* (which colors are representable): **BT.709**
  (standard SDR) or **BT.2020** (wide, for HDR).
- **Transfer** = the *transfer function* / EOTF (the curve mapping stored values
  ↔ light, i.e. the brightness response): SDR **gamma** (~2.2/2.4), **PQ** (SMPTE
  ST 2084, absolute brightness — HDR10), or **HLG** (ARIB STD-B67, relative —
  broadcast HDR).

| `ColorPolicy` | Gamut | Transfer | Bit depth (with `Auto`) | Tonemap |
|---------------|-------|----------|:-----------------------:|:-------:|
| `TonemapToSdr` | BT.709 | gamma | 8-bit | HDR → SDR |
| `Passthrough`  | source | source | source | no |
| `Hdr10`        | BT.2020 | PQ | 10-bit | no |
| `Hlg`          | BT.2020 | HLG | 10-bit | no |

The on-disk pixel format follows from bit depth: 8-bit → `yuv420p`, 10-bit →
`yuv420p10le` (4:2:0). HDR needs a 10-bit encoder (`nvidia`, `amd`,
`qsv`, or `ffmpeg`); `validate()` rejects an HDR request a build can't produce.
HDR is tagged in the container via `colr`/`mdcv`/`clli` atoms.

## 5b. Output codec — `with_video_codec(...)`

The output video codec is `VideoCodec::Av1` (default), `H264`, or `H265`:

```rust
let spec = OutputSpec::single_file(rungs).with_video_codec(VideoCodec::H264);
```

**AV1** is the royalty-clean default (AV1 + Opus in MP4 = zero royalty exposure);
**H.264 / H.265** are for legacy-player compatibility and carry the
patent-licensing obligations AV1 was chosen to avoid. All three work for
single-file MP4 **and** CMAF/HLS — the muxer emits `av01`/`avc1`/`avc3`/`hvc1`/
`hev1` sample entries with the matching config box and `CODECS=` string.
H.264/H.265 are **8-bit 4:2:0** (no HDR); AV1 supports 10-bit HDR. The encoder
backend is chosen per GPU vendor: NVENC + QSV encode H.264/H.265; AMF and the
ffmpeg wrapper currently reject them (a follow-up). The same string vocabulary
(`av1`/`h264`/`h265`) drives the CLI `--codec`, the `codec=` settings key, the
batch manifest `codec:`, and the HTTP `codec` field.

---

## 5. Frame rate — `with_max_frame_rate(fps)`

Cap the output cadence; the source cadence is otherwise preserved.

```rust
spec.with_max_frame_rate(30.0)   // never exceed 30 fps
```

---

## 6. Video filters — `with_filters(...)`

Per-frame transforms — geometry (crop, pad, flip, rotate, grayscale), an image
**overlay** (PNG logo/watermark with alpha), and colour (invert, brightness,
contrast, saturation) — applied to the decoded source **once**, before per-rung
scaling, so a filter applies to every rendition. `spec.filters` is a list of
`codec::filter::VideoFilter`:

```rust
spec.with_filters(vec![
    VideoFilter::Crop { w: 1920, h: 1080, x: None, y: None },
    VideoFilter::Overlay { image: "logo.png".into(), x: 24, y: 24 },
]);
// or parse the equivalent ffmpeg-style string form:
spec.with_filters(codec::filter::parse_chain("crop=1920:1080,overlay=logo.png:24:24")?);
```

See **[Video filters](filters.md)** for the full filter set, the string +
structured-object forms, and per-surface usage.

---

## 7. GPU selection

How encode work spreads across the host's GPUs.

| Method | Effect |
|--------|--------|
| `encode_policy(EncodePolicy)` | The spread policy (below). |
| `with_gpu_index(u32)` | Pin to one GPU index — shorthand for `encode_policy(SingleGpu(Some(idx)))`. |
| `decode_gpu(Option<u32>)` | Pin the **decode pump** to a GPU, independent of encode. `None` follows the encode policy. E.g. decode on an iGPU while the dGPUs encode. |

`EncodePolicy`:

| Variant | Meaning |
|---------|---------|
| `AllGpus` *(default)* | Chunk-encode across every detected GPU and stitch (the multi-GPU engine — see [pipeline §4](pipeline.md#4-the-multi-gpu-lease-engine--the-rung-benefit)). |
| `SingleGpu(Option<u32>)` | One GPU — pinned to `Some(i)`, or the first GPU with `None`. Serial, no chunk overhead. |
| `Family(GpuFamily)` | Restrict to one vendor: `GpuFamily::{Nvidia, Amd, Intel}` (e.g. ignore an integrated GPU). |

```rust
spec.encode_policy(EncodePolicy::Family(rivet::GpuFamily::Nvidia))
    .decode_gpu(Some(0));   // decode on GPU 0, encode on the NVIDIA cards
```

---

## 8. Chunk seams — `chunk_seam_mode(ChunkSeamMode)`

Only relevant when **multiple GPUs** encode a **single file**: each rung is
chunked at GOP boundaries, encoded in parallel, and stitched. Each chunk is an
independent IDR-led GOP so it always plays, but per-chunk rate control can step
quality at the ~2 s seams. This knob governs that (chiefly for NVENC, which
otherwise runs VBR per chunk; AMD/QSV chunks are already constant-QP):

| `ChunkSeamMode` | Seams | Speed |
|-----------------|-------|-------|
| `Parallel` *(default)* | possible mild NVENC steps | fastest (all GPUs) |
| `ParallelConstQp` | flat (forced constant-QP, quality still tracks the target) | fast (all GPUs) |
| `Serial` | none (one encoder for the whole file) | slower; HLS still uses every GPU |

Single-GPU hosts, `--gpu`/`SingleGpu`, and HLS jobs are unaffected (HLS segments
are independent by design).

---

## 9. Validate — `validate()`

```rust
spec.validate()?;
```

Rejects incoherent specs before any work starts: no rungs, zero/odd dimensions,
container/muxer/mode mismatch, HDR with forced 8-bit, or 10-bit/HDR on a build
with no 10-bit encoder (queryable at runtime via
`codec::encode::build_output_caps()`).

---

## 10. Run it

| Function | Use |
|----------|-----|
| `rivet::transcode_file(input, output)` | One file → one file, default spec. Returns a `TranscodeOutcome`. |
| `rivet::transcode_bytes(&bytes, ..)` | The in-memory variant. |
| `rivet::run_job_blocking(&bytes, &spec, out_dir, sink)` | Run a full `OutputSpec` synchronously. `out_dir: Option<&Path>` (the HLS/multi-rung asset root; `None` = temp dir). Returns `JobOutput`. |
| `rivet::run_job(&bytes, &spec, out_dir, sink).await` | The async variant (drive from a Tokio runtime). |
| `rivet::probe_file(path)` / `probe_bytes(&bytes)` | Inspect without transcoding → `MediaInfo`. |

### Progress

Both `run_job*` take a `ProgressSink` that streams a uniform
[`RungProgress`](../crates/rivet/src/progress.rs) per rung — `label`, `status`
(`RungStatus`: `Pending` → `Running` → `Completed`/`Failed`), `percent`,
`frames_done`, segment + byte counters. Wire it however you like:

```rust
use std::sync::Arc;

// a closure
let sink = Arc::new(rivet::fn_sink(|p| println!("{} {:.0}%", p.label, p.percent)));

// or a Tokio channel (async)
let (tx, mut rx) = tokio::sync::mpsc::channel(64);
let sink = Arc::new(rivet::channel_sink(tx));
```

---

## Full method reference

| `OutputSpec` | Signature | Section |
|--------------|-----------|---------|
| `single_file` | `(Vec<Rung>) -> Self` | [1](#1-construct--the-output-shape) |
| `hls` | `(Vec<Rung>, f32) -> Self` | [1](#1-construct--the-output-shape) |
| `with_audio` | `(AudioPolicy) -> Self` | [3](#3-audio--with_audioaudiopolicy) |
| `with_max_frame_rate` | `(f64) -> Self` | [5](#5-frame-rate--with_max_frame_ratefps) |
| `with_color` | `(ColorPolicy) -> Self` | [4](#4-color--bit-depth) |
| `with_bit_depth` | `(BitDepth) -> Self` | [4](#4-color--bit-depth) |
| `web_sdr` / `hdr10` / `hlg` / `passthrough` | `() -> Self` | [4](#4-color--bit-depth) |
| `with_gpu_index` | `(u32) -> Self` | [6](#6-gpu-selection) |
| `encode_policy` | `(EncodePolicy) -> Self` | [6](#6-gpu-selection) |
| `decode_gpu` | `(Option<u32>) -> Self` | [6](#6-gpu-selection) |
| `chunk_seam_mode` | `(ChunkSeamMode) -> Self` | [7](#7-chunk-seams--chunk_seam_modechunkseammode) |
| `validate` | `(&self) -> Result<()>` | [8](#8-validate--validate) |
| `tonemaps` | `(&self) -> bool` | (does this spec tonemap?) |
| `resolve_output` | `(ColorMetadata, PixelFormat) -> (ColorMetadata, PixelFormat)` | (resolve color/depth vs a source) |

All `OutputSpec` fields are `pub`, so anything above can also be set directly
(`spec.color = ColorPolicy::Hdr10;`): `mode`, `video_codec`, `audio`, `container`,
`muxer`, `rungs`, `max_frame_rate`, `gpu_index`, `encode_policy`, `decode_gpu`,
`color`, `bit_depth`, `chunk_seam_mode`. The builders are the recommended path
(they keep linked fields — e.g. `gpu_index` and `encode_policy` — in sync).
