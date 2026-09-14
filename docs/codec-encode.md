# codec: encode, colorspace & audio

The **output side** of the `codec` crate — everything that turns a normalized
decoder frame into the bytes that get muxed. This is the companion to
[pipeline.md](pipeline.md), which covers the end-to-end job flow (demux →
decode-once pump → per-rung scale → multi-GPU lease engine → mux). Read that
first for *where* these pieces sit; this doc is the *what + why* of the encode
half itself.

Three load-bearing decisions shape this whole side, and they recur below:

1. **AV1 is the default output codec; H.264 and H.265 are also supported.** AV1
   is the recommended, royalty-clean target (AV1 video + Opus audio + MP4
   container = zero royalty exposure — see the
   [README's "note on the output codec"](../README.md#a-note-on-the-output-codec)).
   **H.264 / H.265** are available for legacy-player compatibility — they carry
   the patent-licensing obligations AV1 was chosen to avoid. The codec is
   selected per job (`OutputSpec::with_video_codec(VideoCodecPolicy::H264)` /
   `--codec h264` / `codec=h264`; values `av1|h264|h265`). See
   [Output codecs](#output-codecs-av1--h264--h265) below.
2. **Hardware encoders are layered, not consolidated.** Each vendor gets a
   hand-rolled, in-tree `dlopen` FFI encoder (NVENC / AMF / QSV). They *stack*;
   software — rav1e for AV1 (`rav1e-fallback`), the workspace's own `h26x`
   encoders for H.264 / H.265 (`h26x-fallback`) — is the last resort, and it is
   opt-in. New
   tiers add to the chain, they don't replace it. (The Vulkan Video encode tier
   was removed 2026-05-08, and the FFmpeg tier 2026-08-12 — see
   [select_encoder](#the-encode-dispatch--capability-query) and [No
   FFmpeg](../README.md#no-ffmpeg).)
3. **HDR is tonemapped to SDR by policy.** The default single-output policy maps
   every HDR source down to 8-bit BT.709 at transcode time so a clip never lands
   eye-searingly bright on a viewer's screen. HDR-passthrough is a latent,
   policy-gated path, not the default. See [Tonemapping](#tonemapping--the-single-output-policy).

---

## Module map

| File | Purpose |
|------|---------|
| [`encode/mod.rs`](../crates/codec/src/encode/mod.rs) | The `Encoder` trait, `EncoderConfig`, `select_encoder` dispatch, `OutputCaps` runtime capability query, the `TRANSCODE_ENCODER_BACKEND` override. |
| [`encode/tuning/`](../crates/codec/src/encode/tuning/) | The calibration layer. `QualityTarget` / `SpeedTier` → per-encoder knobs (CQ, q-index, ICQ, presets, tile grid) in `adapters.rs`, and the per-rung override vocabulary (`EncodeOverrides`, `RungPolicy`) in `overrides.rs`. |
| [`encode/nvenc.rs`](../crates/codec/src/encode/nvenc.rs) + [`nvenc_stub.rs`](../crates/codec/src/encode/nvenc_stub.rs) | NVENC AV1 encoder (NVIDIA Ada+), hand-rolled `nvEncodeAPI` FFI. Stub when `nvidia` is off. |
| [`encode/amf/`](../crates/codec/src/encode/amf/) + [`amf_stub.rs`](../crates/codec/src/encode/amf_stub.rs) | AMF encoders: H.264 (`VCE_AVC`) and H.265 (`HW_HEVC`, Main / Main 10) on every AMF-capable AMD GPU, AV1 (`HW_AV1`) on RDNA3+. Hand-rolled AMF runtime FFI mirrored slot-for-slot from the SDK v1.4.36 C headers (`ffi.rs`), one session flow (`mod.rs`) and a property sequence per codec (`av1.rs`, `h26x.rs`). Stub when `amd` is off. |
| [`encode/qsv.rs`](../crates/codec/src/encode/qsv.rs) + [`qsv_stub.rs`](../crates/codec/src/encode/qsv_stub.rs) | QSV AV1 encoder (Intel Arc / Meteor Lake+), hand-rolled oneVPL FFI. Stub when `qsv` is off. |
| [`encode/rav1e_sw.rs`](../crates/codec/src/encode/rav1e_sw.rs) | Software AV1 encoder via [rav1e](https://crates.io/crates/rav1e) — pure Rust, 8-bit 4:2:0. Gated on `rav1e-fallback`. |
| [`encode/h26x_sw.rs`](../crates/codec/src/encode/h26x_sw.rs) | Software H.264 / H.265 encoders via the workspace's own [`h26x`](../crates/h26x) crate — pure Rust, 4:2:0 at 8 bits (H.265 also 10-bit Main 10), CABAC, constant QP on the shared H.26x anchor table, `force_keyframe_next` honoured. The output colour (`ColorMetadata`) goes into the SPS VUI and the HDR10 static metadata into SEIs 137 / 144, so HDR10 / HLG output validates on a build with no GPU. Fallback gated on `h26x-fallback`; always constructible by name. |
| [`colorspace.rs`](../crates/codec/src/colorspace.rs) | Frame normalization: chroma-layout convert, BT.601→709 matrix, 4:4:4→4:2:0 downsample, bilinear scaling — scalar + AVX2 runtime dispatch. |
| [`tonemap.rs`](../crates/codec/src/tonemap.rs) | HDR→SDR tonemap: PQ/HLG inverse EOTF → BT.2020→709 gamut → Hable filmic curve → 8-bit BT.709. |
| [`audio/mod.rs`](../crates/codec/src/audio/mod.rs) | Audio decode→Opus transcode framework: traits, wire types, `create_decoder` / `create_encoder`. |
| [`audio/decode/mp3.rs`](../crates/codec/src/audio/decode/mp3.rs), [`vorbis.rs`](../crates/codec/src/audio/decode/vorbis.rs) | MP3 (minimp3) and Vorbis (lewton) decoders → interleaved f32 PCM. |
| [`audio/encode/opus.rs`](../crates/codec/src/audio/encode/opus.rs) | Opus encoder (libopus), mono/stereo + multistream surround; emits the `dOps` config + `pre_skip`. |
| [`audio/resample.rs`](../crates/codec/src/audio/resample.rs) | Sample-rate conversion (rubato sinc) — e.g. 44.1 kHz MP3 → 48 kHz Opus. |

---

## Output codecs (AV1 + H.264 / H.265)

AV1 is the default, royalty-clean output. `EncoderConfig.codec`
([`VideoCodec`](../crates/codec/src/frame.rs)) also selects **H.264** or
**H.265** for legacy-player compatibility — all three work for single-file MP4,
CMAF/HLS, and the multi-GPU chunk-stitch path. Per-backend status:

| Backend | AV1 | H.264 / H.265 |
|---------|-----|---------------|
| **QSV** (Intel Arc+) | ✅ | ✅ **validated** — `codec_id` = AVC/HEVC, AV1 tile ext buffer skipped; emits Annex-B NAL |
| **NVENC** (NVIDIA) | ✅ (Ada+) | ✅ **validated** — codec GUID dispatch (H.264 Kepler+, H.265 Maxwell+); preset-seeded config + 1-in-1-out drain |
| **AMF** (AMD) | ⚠ by-review (RDNA3+ only; the dev box's iGPU has no AV1 block) | ✅ **validated on a Ryzen 9 9950X iGPU** — `AMFVideoEncoderVCE_AVC` / `AMFVideoEncoderHW_HEVC`, Annex-B frame output with in-band SPS/PPS(/VPS) on every IDR, H.265 Main 10 via P010; 1080p H.264 41 dB, 720p H.265 43 dB, Main 10 53 dB luma PSNR vs source, HLS segments decode |
| rav1e (software) | ✅ 8-bit | ❌ rejected — rav1e is an AV1 encoder |
| **h26x** (software, in-tree) | ❌ rejected — H.264 / H.265 only | ✅ 8- and 10-bit — H.265 Main / Main 10, H.264 High / High 10 (the only 10-bit H.264 here); the crate's own encoders, every stream gated SELF (our decoder reproduces the encoder's reconstruction) + CROSS (libavcodec agrees) |

H.264/H.265 encoders emit **Annex-B** NAL; the muxer's
[`nal_mux`](../crates/container/src/nal_mux.rs) splits each packet into per-frame
access units (HW encoders pack several frames per buffer), captures SPS/PPS(/VPS)
for the `avcC`/`hvcC` config box, and repackages slices as length-prefixed
samples (`avc1`/`hvc1`). rav1e rejects H.264/H.265 rather than silently emit
AV1; the `h26x` software tier is the mirror image and rejects AV1.

### Bit depth (H.265 8/10-bit everywhere, H.264 10-bit in software only)

**H.265 encodes 8- or 10-bit** (Main / Main 10, 4:2:0) on NVENC, QSV and AMF,
all hardware-validated — `with_bit_depth(TenBit)` or a HDR `ColorPolicy`
produces a genuine Main 10 stream:
- **NVENC** (RTX 3090): selects the `HEVC_PROFILE_MAIN10` GUID and sets
  `NV_ENC_CONFIG_HEVC.output/inputBitDepth = 10` (a typed view onto the codec-
  config union — without it `NvEncCreateInputBuffer` rejects the P010 surface).
  The input is the **semi-planar** `YUV420_10BIT` surface (interleaved UV, P010-
  style, `sample << 6`). Verified: `profile=Main 10`, `pix_fmt=yuv420p10le`,
  PSNR Y 46 / U 43 / V 43 dB vs the 10-bit source, single-file **and** HLS
  (`CODECS="hev1.2.4…"`).
- **QSV** (Intel Arc): selects `MFX_PROFILE_HEVC_MAIN10` + P010 surfaces with
  `Shift=1` and `BitDepthLuma/Chroma=10`. Verified `profile=Main 10` /
  `yuv420p10le`.
- **AMF** (Ryzen 9 9950X iGPU): `HevcProfile = MAIN_10` + `HevcColorBitDepth = 10`
  with P010 host surfaces (`sample << 6`). Verified `profile=Main 10` /
  `yuv420p10le`, level 3.1 at 720p30, 53 dB luma PSNR vs the 10-bit source.
  Main 10 runs **constant QP** on AMF: the driver ignored the QVBR quality
  level at 10 bits (levels 1 / 26 / 32 / 38 all produced the identical
  17.3 Mbit/s stream) while CQP tracks the QP as it should.

The muxer's `build_hvcc` parses the bit depth from the SPS, so the `hvcC` carries
`bitDepthLumaMinus8 = 2` for Main 10. `build_avcc` does the same for H.264: every
profile but Baseline / Main / Extended gets the record's high-profile extension
(ISO/IEC 14496-15 §5.3.3.1.2 — `chroma_format`, `bit_depth_luma_minus8`,
`bit_depth_chroma_minus8`, zero SPS extensions), byte for byte what ffmpeg's
writer emits (`fd f8 f8 00` for 8-bit High, `fd fa fa 00` for High 10). Before
2026-09-13 no `avcC` rivet wrote carried it, 8-bit High included.

**H.264 at 10 bits is the software tier's alone.** Neither NVENC (no `High 10`
profile GUID), QSV (no `AVC High 10` in oneVPL) nor AMF (no 10-bit `Profile`
value) exposes a hardware Hi10P encoder, so a 10-bit H.264 request is
**refused** on each of them with a clear error ("does not support 10-bit H264
encode") rather than silently down-converted to 8-bit. The native `h26x` tier
encodes it: `yuv420p10le` becomes an SPS with `profile_idc` 110 and
`bit_depth_luma_minus8 = 2` (**High 10**), the VUI colour description written as
for every other stream here. Verified end to end on `rivet transcode --codec
h264` of a 10-bit HEVC source: ffprobe `profile=High 10`, `pix_fmt=yuv420p10le`,
`ffmpeg -v error` decodes with no output, single-file and HLS.
`backend_output_caps_for(backend, VideoCodec::H264)` says so per backend
(10-bit HDR for `H26x`, 8-bit SDR for the three hardware backends).

What an HDR policy does with `--codec h264`: `OutputSpec::validate` checks
the colour and depth against the caps **for the spec's codec** —
`backend_output_caps_for` over `compiled_encode_backends()`, whose union is
`build_output_caps_for(codec)` (rivet's `spec::CodecOutputCaps`). On a
hardware-only build (no `h26x-fallback`) `--color hdr10|hlg` or a forced
10-bit depth with `--codec h264` is refused before the job starts ("h264 at 10
bits … needs the software tier (build with `h26x-fallback`)"). Until
2026-09-13 it validated against the codec-agnostic `build_output_caps` and
failed at the encoder's refusal after the job had started. On an
`h26x-fallback` build it produces High 10 BT.2020 PQ / HLG. The same check
refuses `--codec av1` at 10 bits on a build whose only encoders are software
(h26x has no AV1; rav1e is 8-bit). A backend pinned by name
(`TRANSCODE_ENCODER_BACKEND`) is added to the compiled set, since
`create_backend` builds `h26x` / `rav1e` by name with no feature check. It
checks the build, not the silicon: an
AV1 request that the build's NVENC could serve still fails at encoder
construction on a card without AV1 encode.

**NVENC H.264/H.265** uses the codec's GUID for capability validation, preset
selection, and session init; the preset (`GetEncodePresetConfigEx`) seeds the
codec config union so the H264/HEVC layout doesn't have to be mirrored. H.264 is
pinned to High profile, H.265 to Main. Without a `bframes` override the encoder
is **strictly 1-in-1-out** for H.264/H.265 (clear `enableLookahead`, set
`zeroReorderDelay`, every non-IDR picture forced P) and the ring-of-4 sync drain
emits one packet per `EncodePicture`. With `bframes = N` (`--encode-policy
…:bframes=N`, non-pyramid) `zeroReorderDelay` is cleared, the picture type is
left to the driver so it may choose B, and the drain walks the in-flight FIFO
from the *oldest* surface (first lock blocking — `SUCCESS` guarantees it — the
rest non-blocking), so packets come out in **decode order** each carrying the
presentation timestamp of the picture it codes; `flush_eos` drains the same FIFO
in submission order. The muxer derives the composition offsets from those
timestamps ([container.md](container.md#composition-offsets-ctts-for-b-pictures)).
When every frame is drained during encode, the EOS flush is skipped (sending it
busy-waits on the SDK 13 driver).

### Multi-GPU + capability dropout (all codecs)

The cross-cutting engine features apply to H.264/H.265 too, not just AV1:
- **Decode pump, video filters, and the multi-rung ABR ladder** were always
  codec-agnostic (upstream of the encoder).
- **Capability dropout** is codec-aware: `encode_capable(dev, codec)` (cached per
  `(gpu_index, codec)`) probes the actual encoder, so `gpu_pool_for_policy` drops
  GPUs that can't encode the *requested* codec — e.g. an NVIDIA Ampere card is
  dropped from an AV1 pool but kept for an H.264/H.265 pool.
- **Multi-GPU chunk-and-stitch** covers AV1, H.264, and H.265. The cross-vendor
  codec invariant is an enum — `Av1Invariant` (sequence-header fields) +
  `H26xInvariant` (profile / level / chroma / bit-depth / dims from the SPS, via
  `parse_h264_sps` / `parse_hevc_sps`). Each chunk is a closed GOP (first frame an
  IDR), so stitched H.264/H.265 reset references cleanly at chunk boundaries. HLS
  output covers all three codecs too — the CMAF muxer emits `av01`/`avc1`/`avc3`/
  `hvc1`/`hev1` init segments and `codec_string_from_init` reads the matching
  `av1C`/`avcC`/`hvcC` config box for the `CODECS=` attribute.
- **Inline parameter sets** make the stitch robust across vendors. Chunks come
  from independent encoders whose SPS/PPS may agree on the invariant yet differ
  cosmetically (VUI) or in PPS (entropy mode). Mirroring AV1's inline OBU sequence
  headers, the stitch muxer (`new_with_codec_inline`) keeps SPS/PPS(/VPS) inline
  in each access unit and emits the `avc3`/`hev1` sample entry (in-band parameter
  sets) instead of `avc1`/`hvc1`, so every chunk decodes with its own parameter
  sets. The serial single-file path keeps `avc1`/`hvc1` (one encoder, params
  out-of-band).

Validation:
- **NVENC on RTX 3090** (this repo's dev box): H.264 + H.265 each decode 96/96
  frames, 0 errors, BT.709, ~0.9 s (full NVDEC→NVENC round-trip).
- **QSV single-GPU on the Arc box**: H.264/H.265/AV1 each decode 96/96 frames, 0
  errors, identical PSNR-vs-source, consistent BT.709.
- **QSV multi-GPU on the 3× Arc box**: H.264 + H.265 chunk-and-stitch across all
  three Arcs (A310/A380/A750), 5 segments dispatched over the lease pool, H26x
  invariant captured + matched with 0 mismatches. Output is `avc3`/`hev1` with
  inline parameter sets and decodes 300/300 frames, 0 errors, BT.709.

## The encode dispatch & capability query

> Source: [`crates/codec/src/encode/mod.rs`](../crates/codec/src/encode/mod.rs)

### What

Every encoder backend implements one trait
([`Encoder`](../crates/codec/src/encode/mod.rs#L69)):

```rust
pub trait Encoder: Send {
    fn send_frame(&mut self, frame: &VideoFrame) -> Result<()>;
    fn flush(&mut self) -> Result<()>;
    fn receive_packet(&mut self) -> Result<Option<EncodedPacket>>;
}
```

`send_frame` pushes one normalized frame; `receive_packet` drains
[`EncodedPacket`](../crates/codec/src/encode/mod.rs#L75)s (raw AV1 OBU bytes +
PTS + keyframe flag); `flush` signals end-of-stream so the encoder drains its
lookahead/B-frame queue. This is the same push/drain shape the decode side uses,
so the pipeline treats every vendor identically.

[`EncoderConfig`](../crates/codec/src/encode/mod.rs#L89) carries everything a
backend needs: dimensions, frame rate, keyframe interval, the perceptual
`target` + `tier` (see [tuning](#quality-tuning-perceptual-target--encoder-knobs)),
the input `pixel_format` (8-bit `Yuv420p` vs 10-bit `Yuv420p10le`), source
`color_metadata`, and three multi-GPU dispatch hints — `gpu_index`,
`gpu_vendor`, and `constant_qp`.

[`select_encoder`](../crates/codec/src/encode/mod.rs#L282) is the factory. It
detects GPUs at runtime and tries backends **in tier order**:

1. **Vendor-pin shortcut** — if `config.gpu_vendor` is set (the CMAF
   orchestrator does this via the `GpuPool` lease), dispatch *directly* to that
   vendor's backend, skipping the preference chain
   ([mod.rs:333-378](../crates/codec/src/encode/mod.rs#L333)).
2. **Auto-select chain** — NVENC (Ada+) → AMF (RDNA3+) → QSV (Arc / Meteor
   Lake+) ([mod.rs:388-460](../crates/codec/src/encode/mod.rs#L388)).
3. **Software** (opt-in) — the last tier, so a build with it on never quietly
   prefers CPU over silicon that was merely busy: rav1e for AV1
   (`rav1e-fallback`), the workspace's own `h26x` encoders for H.264 / H.265
   (`h26x-fallback`). Each is tried only for its own codec.
4. **Hard fail** — no encode silicon for the codec, no software fallback
   compiled in; the error names the feature that would have caught it.

`TRANSCODE_ENCODER_BACKEND=nvenc|amf|qsv|h26x|rav1e` (the README CLI note) maps
to the `preferred: Option<EncoderBackend>` argument, which routes through
[`create_backend`](../crates/codec/src/encode/mod.rs#L471) and bypasses the
chain entirely — including the fallback features, which gate only the *unasked*
route: `h26x` and `rav1e` by name always construct.

### Why

- **GPU-only by default, fail-fast.** With no software feature enabled the auto
  chain ends in an `Err`, not a CPU fallback.

  rav1e and Vulkan encode were both deleted on 2026-05-08 — "rav1e on Archive
  preset doesn't keep up with real-time throughput at 4K and the Vulkan-encode
  binding never made it past scaffolding". **rav1e came back**, as
  `encode/rav1e_sw.rs` behind the off-by-default `rav1e-fallback` feature, and
  it is the last tier before the hard-fail rather than a peer of the vendor
  backends. Vulkan encode did not come back. Read the 2026-05-08 note as the
  reason software encode is not *preferred*, not as a claim that it is absent.
  Degrading silently to a 20× slower CPU encode is worse than telling the
  operator to reprovision — so a host with no AV1-encode silicon errors at
  encoder construction with a clear message. The H.264 / H.265 software tier
  (`encode/h26x_sw.rs`, 2026-08-27) follows the same rule under its own
  `h26x-fallback` feature.
- **The vendor-pin shortcut exists because the preference chain is greedy.**
  Without it, a host with both an NVIDIA and an Intel GPU routed *every* variant
  to NVENC (the chain hits `pick_vendor_device(Nvidia, …)` first), leaving the
  Arc idle even when NVENC sessions were saturated. The CMAF orchestrator leases
  a specific GPU and pins the vendor so work actually spreads
  ([mod.rs:324-332](../crates/codec/src/encode/mod.rs#L324)).
- **A capability gap is not an error.** An NVIDIA GPU whose NVENC predates AV1
  (consumer 30-series and older) logs an INFO and *falls through* to the next
  vendor rather than failing — it can still decode, just not AV1-encode
  ([mod.rs:404-413](../crates/codec/src/encode/mod.rs#L404)).
- **`pick_vendor_device` honours an explicit index but falls through on a
  vendor mismatch** ([mod.rs:44-53](../crates/codec/src/encode/mod.rs#L44)) so
  that `gpu_index = Some(2)` pinned to an NVIDIA slot, when GPU 2 is actually
  AMD, returns `None` from the NVIDIA tier and gets matched by the AMD tier's
  own `find()` pass. This keeps multi-GPU variant→device pinning correct without
  the caller knowing each device's vendor.

### Capability query — `OutputCaps`

`select_encoder` answers "encode this frame *now*." Before a job starts, the
engine needs "can this build encode *that format at all?*" — that is
[`build_output_caps()`](../crates/codec/src/encode/mod.rs#L234), the runtime
query [`OutputSpec::validate`](pipeline.md#6-color--bit-depth) consults to reject
e.g. an HDR (10-bit) request on a build with no 10-bit encoder.

| Function | Returns |
|----------|---------|
| [`backend_output_caps(backend)`](../crates/codec/src/encode/mod.rs#L221) | Per-backend caps. All three HW backends report `{max_bit_depth: 10, hdr: true}` — NVENC via `Yuv420_10bit`, AMF via `P010`, QSV via in-repo oneVPL P010. The software `h26x` tier reports the same: H.265 Main 10 with the colour description in the SPS VUI (`h26x_sw::colour_description`). `rav1e` is `{8, false}`. |
| [`build_output_caps()`](../crates/codec/src/encode/mod.rs#L234) | The **union over compiled paths**. 10-bit+HDR if any of `nvidia`/`amd`/`qsv`/`h26x-fallback` is on; `rav1e-fallback` alone is 8-bit. |
| [`backend_output_caps_for(backend, codec)`](../crates/codec/src/encode/mod.rs) | Per backend **and codec**. Differs from the per-backend answer for H.264: 8-bit SDR on NVENC / AMF / QSV (no High 10 encoder), 10-bit HDR on `h26x`. A codec the backend does not serve reports the 8-bit floor. |
| [`build_output_caps_for(codec)`](../crates/codec/src/encode/mod.rs) | The union of the above over compiled paths: H.264 is 10-bit only with `h26x-fallback`. What `OutputSpec::validate` checks a spec's codec against (via rivet's `spec::CodecOutputCaps`), and what `rivet capabilities` prints per codec. |
| [`compiled_encode_backends()`](../crates/codec/src/encode/mod.rs) | The compiled backends as `EncoderBackend`s, in dispatch order — the set the union is taken over, for a caller that needs the per-backend answers behind it (rivet's refusal message names them). |
| [`encode_backends()`](../crates/codec/src/encode/mod.rs#L249) | The compiled backends in dispatch order — `["nvenc", "amf", "qsv", "rav1e", "h26x"]` filtered by feature flags. Drives `rivet capabilities`. |

Why a runtime union and not a compile-time constant: features are additive and
the answer the validator wants ("can this *binary* produce 10-bit AV1?") is a
property of the whole build, queryable without constructing an encoder.

---

## Hardware encoder backends (and why stubs exist)

All three HW encoders share a shape: a hand-rolled `dlopen` FFI binding (no
external wrapper crate, no bindgen, no build-time SDK link, so they **build on
both Windows MSVC and Linux** even without the hardware present), a `RING_SIZE`
input-surface ring, a per-frame YUV→vendor-surface upload, and a flush/drain at
EOS. Each is **spec-conformant-by-review** — the dev box is NVIDIA Ampere (RTX
3090, no AV1-encode silicon), so none is E2E-verified on its own target. Each
file carries a battery of `const_assert!` size checks that fire at compile time
if a vendored struct layout drifts ([nvenc.rs:30-37](../crates/codec/src/encode/nvenc.rs#L30)).

### The stub pattern (`*_stub.rs`)

> Source: [`nvenc_stub.rs`](../crates/codec/src/encode/nvenc_stub.rs),
> [`amf_stub.rs`](../crates/codec/src/encode/amf_stub.rs),
> [`qsv_stub.rs`](../crates/codec/src/encode/qsv_stub.rs)

`encode/mod.rs` uses `#[path = "…_stub.rs"]` to swap a stub in when a vendor
feature is off ([mod.rs:1-17](../crates/codec/src/encode/mod.rs#L1)). The stub
keeps `nvenc::NvencEncoder` (etc.) a **real type with the same `new()`
signature** so the dispatcher in `select_encoder` compiles unchanged — but
`new()` always `bail!`s with a "rebuild with the `nvidia` feature" message. The
trait methods are `unreachable!()` because the encoder is never constructed.

**Why:** it lets the dispatch logic reference all three backends without
`#[cfg]` noise at every call site. Auto-select simply sees the stub's
construction error and skips that tier; an explicit `EncoderBackend::Qsv`
request surfaces the helpful "not compiled in" error instead of a cryptic
missing-symbol link failure.

### NVENC (`nvenc.rs`)

> NVIDIA Ada+ (RTX 4000+, Ampere datacenter A10/A10G/L4/L40).

Drives the NVENC API through the `NV_ENCODE_API_FUNCTION_LIST` function-pointer
table (`NvEncodeAPICreateInstance`) rather than dlsym-ing each symbol — matching
how OBS/FFmpeg drive it ([nvenc.rs:6-11](../crates/codec/src/encode/nvenc.rs#L6)).
Session flow is documented in the module header
([nvenc.rs:13-28](../crates/codec/src/encode/nvenc.rs#L13)): open session →
preset config → init → input/bitstream ring buffers → per-frame
lock/copy/encode/extract → EOS flush → teardown in reverse alloc order.

10-bit uses `NV_ENC_BUFFER_FORMAT_YUV420_10BIT`; the pipeline stores 10-bit in
the *lower* 10 bits of each `u16`, so `upload_frame_10bit` performs the `<<6`
shift on copy to satisfy NVENC's P010-style *upper-10-bits* convention
([nvenc.rs:69-77](../crates/codec/src/encode/nvenc.rs#L69)).

### AMF (`amf/`)

> Any AMD GPU the AMF runtime drives for H.264 (`AMFVideoEncoderVCE_AVC`) and
> H.265 (`AMFVideoEncoderHW_HEVC`); RDNA3+ (Radeon RX 7000+) for AV1
> (`AMFVideoEncoderHW_AV1`). **H.264 / H.265 are hardware-validated** on the
> dev box's Ryzen 9 9950X iGPU (2026-08-27); AV1 is by-review (that iGPU answers
> `AMF_CODEC_NOT_SUPPORTED` for the AV1 component, as it should).

Property-driven: every knob is an `AMFComponent::SetProperty(name, value)` call
with wide-string names copied — with the header line cited beside each — from
the AMF SDK v1.4.36 `components/VideoEncoderVCE.h` / `VideoEncoderHEVC.h` /
`VideoEncoderAV1.h` ([h26x.rs](../crates/codec/src/encode/amf/h26x.rs),
[av1.rs](../crates/codec/src/encode/amf/av1.rs)). The vtables in
[ffi.rs](../crates/codec/src/encode/amf/ffi.rs) list every slot of every C
`…Vtbl` in header order, and a `const` block pins each called slot's byte
offset and each vtable's size to the header — the AMF C ABI has traps the
earlier AV1-only binding had fallen into (`AMFInterface` is
**Acquire/Release/QueryInterface**, ten `AMFPropertyStorage` slots precede
every interface's own methods, `InitVulkan` lives on `AMFContext1`,
`AMFVariantStruct` is 24 bytes, `AMF_RESULT` is sequential so `AMF_EOF` is 23
and `AMF_INPUT_FULL` 25). A test against the *installed* runtime
(`test_amf_runtime_property_storage_abi`) round-trips int / bool / rate
variants through a real context and `QueryInterface`s it to `AMFContext1`.

Session flow ([mod.rs](../crates/codec/src/encode/amf/mod.rs)): `AMFInit` →
`CreateContext` → on Windows a D3D11 device on the chosen AMD adapter
(`amf_device`, so a mixed host binds the AMD card and not DXGI adapter 0) to
`InitDX11(dev, AMF_DX11_1)`, elsewhere `QueryInterface(AMFContext1)` →
`InitVulkan(null)` → `CreateComponent` → the codec's property sequence →
`Init(NV12 | P010, w, h)` → per-frame `AllocSurface` / copy / `SubmitInput` /
`QueryOutput` → `Drain` → teardown in reverse.

What the H.26x sequence sets: `Usage = TRANSCODING`, `Profile = High` /
`HevcProfile = Main | Main 10`, a level from the H.264 / H.265 level tables
(frame size × rate, plus the level's bitrate ceiling), the quality preset by
tier (each codec numbers its preset enum differently), `FrameRate`, no B
pictures (`BPicturesPattern = 0`), `IDRPeriod` / `HevcGOPSize` +
`HevcGOPSPerIDR = 1` from the keyframe interval, frame-level `OutputMode`,
`CABACEnable`, colour profile / transfer / primaries / range in and out,
`ColorBitDepth`. Rate control is `QUALITY_VBR` with `QvbrQualityLevel`,
resolution-scaled `TargetBitrate` / `PeakBitrate` / `VBVBufferSize` (capped by
the level) and `EnforceHRD`; `VisuallyLossless`, `ParallelConstQp` and Main 10
use `CONSTANT_QP` with `QPI` / `QPP` (/ `QPB`). Every IDR — frame 0, each GOP
boundary, and `force_keyframe_next` — is forced from our side
(`ForcePictureType = IDR` / `HevcForcePictureType = IDR` on the surface) with
`InsertSPS` + `InsertPPS` / `HevcInsertHeader`, so parameter sets are in band
on every random-access point and a segment cut there is self-describing; the
output buffer's `OutputDataType` / `HevcOutputDataType == IDR` tags the packet.

Three things measured on hardware that the headers do not say:

- **`QvbrQualityLevel` runs higher = better.** H.264 1080p: level 1 → 35.9 dB
  at 1.3 Mbit/s, 26 → 41.1 dB at 4.1 Mbit/s, 51 → 47.1 dB at 8.2 Mbit/s
  (H.265 720p the same shape). The adapter therefore hands over `52 − QP`
  (`tuning::qvbr_level_for_qp`), which keeps the Standard target's QP 26 at
  level 26. The AV1 sequence applies the same inversion by inference — its
  header words the property identically — and is unverified.
- **Main 10 ignores the QVBR level** on this driver (see Bit depth above), so
  it is constant QP.
- **After `Drain`, `QueryOutput` must be polled until `AMF_EOF`.** The frames
  already submitted are still in flight; stopping at the first `AMF_REPEAT`
  lost the tail (114 of 120 frames). The flush now polls (bounded at 10 s).

The **`AMF_INPUT_FULL` retry policy** still applies: it is a *transient*
status, not a failure. **Don't** release the surface (releasing it makes the
retry a use-after-free); drain output via `QueryOutput` to free an input slot,
then retry `SubmitInput` with the *same* surface pointer. Only after the
eventual `AMF_OK` does the encoder take its own ref and we release ours.

### QSV (`qsv.rs`)

> Intel Arc (DG2/BMG) + Meteor/Lunar Lake iGPUs. oneVPL `libvpl`.

Struct-driven (everything lives in `mfxVideoParam` fields, no property bag). The
flow runs a `Query` pass first so the runtime can adjust params, then `Init`,
then a 4-deep surface ring ([qsv.rs:9-36](../crates/codec/src/encode/qsv.rs#L9)).
Shared `mfx` struct layouts live in `crate::qsv_ffi` so encode and decode can't
drift apart ([qsv.rs:63-68](../crates/codec/src/encode/qsv.rs#L63)).

Three QSV decisions are worth calling out:

- **LowPower / VDENC is ON, not OFF.** AV1 QSV encode is **VDENC (low-power)
  only** — it's the only AV1 encode entry point the iHD driver exposes — so
  `LowPower` must be `MFX_CODINGOPTION_ON` or `Query` rejects with
  `MFX_ERR_UNSUPPORTED` ([qsv.rs:518-519](../crates/codec/src/encode/qsv.rs#L518),
  asserted by the test at [qsv.rs:985-987](../crates/codec/src/encode/qsv.rs#L985)).
  *Note:* the `QsvAv1Params.low_power` field doc in `tuning/` still reads
  "Always `MFX_CODINGOPTION_OFF`"
  ([tuning/](../crates/codec/src/encode/tuning/)(../crates/codec/src/encode/tuning/)) — that comment
  is **stale**; the actual emitted value is ON.
- **ICQ is rate-control mode 9, not 8.** `MFX_RATECONTROL_ICQ = 9`; **8 is
  `MFX_RATECONTROL_LA`** (lookahead). The original code used 8 and AV1/Arc
  rejected `Query` with `MFX_ERR_UNSUPPORTED`
  ([qsv.rs:98-102](../crates/codec/src/encode/qsv.rs#L98)). ICQ (Intelligent
  Constant Quality) is the QSV equivalent of CRF and the right match for a
  perceptual target; lookahead-bitrate is not used. The numeric value in
  `tuning/`'s `QsvRateControl` enum is documentary — `qsv.rs` holds the
  authoritative wire constant and only consumes the tuning enum to pick the
  CQP-vs-ICQ *branch* ([qsv.rs:691-701](../crates/codec/src/encode/qsv.rs#L691)).
- **16-multiple coded dims + neutral-black NV12 fill (the "green bars" fix).**
  AV1 requires coded dimensions that are a multiple of 16, so e.g. 572×240
  encodes at 576×240 and 1080 at 1088. The surface is allocated at the aligned
  size (`width` → `align_up(.., 16)`, with `crop_w`/`crop_h` set to the real
  dims; pitch aligned to 64 bytes for Arc DMA — [qsv.rs:723-728](../crates/codec/src/encode/qsv.rs#L723),
  [qsv.rs:960-962](../crates/codec/src/encode/qsv.rs#L960)). The per-frame upload
  only touches the real pixels, so the padding rows/cols would otherwise be
  **zero**, which a browser decodes through BT.709 as the distinctive **green
  bars**. The fix: pre-fill each ring surface with *neutral black* — `Y=16,
  Cb/Cr=128` for 8-bit BT.709 limited (and `<<6` for P010 10-bit) — so the
  untouched padding decodes as black ([qsv.rs:970-993](../crates/codec/src/encode/qsv.rs#L970)).

---

## Quality tuning: perceptual target → encoder knobs

> Source: [`crates/codec/src/encode/tuning/`](../crates/codec/src/encode/tuning/)

### What

The user picks two backend-agnostic things; the adapter translates them into
each encoder's native parameters so identical inputs yield visually consistent
output across vendors.

- [`QualityTarget`](../crates/codec/src/encode/tuning/) — a **perceptual
  goal** expressed in VMAF/SSIMULACRA2 bands, *not* an encoder CRF:
  `VisuallyLossless` (~VMAF 98) · `High` (~95) · `Standard` (~90, default) ·
  `Low` (~85) · `Vmaf(u8)` (explicit escape hatch).
- [`SpeedTier`](../crates/codec/src/encode/tuning/) — how much wall-clock
  to spend: `Draft` · `Standard` (default) · `Archive`. Maps to native speed
  presets (NVENC P5/P6/P7, etc.).

The `*_av1_params(target, tier, width, height)` functions
([nvenc_av1_params](../crates/codec/src/encode/tuning/),
[amf_av1_params](../crates/codec/src/encode/tuning/),
[qsv_av1_params](../crates/codec/src/encode/tuning/)) each return a
concrete params struct the matching encoder splats into its SDK structs; the
H.26x adapters (`qsv_h26x_params`, `amf_h26x_params`, `h26x_sw_params`) share
one 0..51 QP anchor table so a job keeps its QP whichever backend runs it.
Resolution is an input because tile grid and lookahead sizing depend on frame
size.

These connect to `EncoderConfig` via the `AUTO_FROM_TARGET = u8::MAX` sentinel
([mod.rs:168](../crates/codec/src/encode/mod.rs#L168)): when `quality` /
`speed_preset` are left at the sentinel, the encoder derives the quantizer/preset
from `target`/`tier`; a non-sentinel value is a legacy per-encoder override
(e.g. a literal CQP q-index, used by `ParallelConstQp` chunk seams).

### Why

- **libaom is the cross-encoder reference.** Every backend is equalized *to*
  libaom's VMAF at each quality band
  ([libaom_cq_for_target](../crates/codec/src/encode/tuning/)), then a
  per-encoder calibration shift compensates for that encoder's
  compression-efficiency gap (NVENC ~3-4 CQ lower, AMF ~8 q-index lower in
  0..255 space). The `Vmaf(u8)` escape hatch interpolates between calibrated
  anchor tables ([piecewise_cq](../crates/codec/src/encode/tuning/)).
  Source tables: `docs/av1-tuning-research.md`.
- **The QP scales genuinely differ per vendor**, and the doc comments encode the
  traps: NVENC AV1 CQ is **0..63** (not the 0..51 H.264/HEVC range); AMF q-index
  is the full AV1 **0..255**; QSV ICQ is **1..51** (an oneVPL idiosyncrasy that
  scales AV1's 0..63 into 0..51 for API parity). A value sent on the wrong scale
  is silently mis-quantized or rejected.
- **Fewer tiles = better compression on HW encoders.** Tile boundaries break
  loop-filter continuity and AV1 tiles are entropy-coded independently, so the
  shared HW tile grid ([tile_grid_hw](../crates/codec/src/encode/tuning/))
  caps at 2×2 even at 4K — the HW encoders have enough internal parallelism that
  they don't need rav1e's aggressive 4×4 grid for throughput. A regression test
  pins every grid inside AV1 Level 5.1 tile limits
  ([tuning/](../crates/codec/src/encode/tuning/)(../crates/codec/src/encode/tuning/)).
- **No low-latency presets.** This is a batch transcode service, so NVENC
  P1–P4, AMF `Speed`, and the streaming/CBR rate-control modes are deliberately
  never selected — `VisuallyLossless`/`Archive` uses constant-QP for reproducible
  bitstreams, everything else uses a quality-targeting VBR.

---

## Per-rung tuning: when one quality for the whole ladder is wrong

Everything above derives its numbers from two enums — a `QualityTarget` and a
`SpeedTier` — and nothing else. That is a good default and a poor ceiling: it
produces **one quality value for every rung of an ABR ladder**, and a constant
quantizer is not a constant perceptual quality. The same quantizer at a quarter
of the resolution is a far finer quantizer relative to what an eye can resolve,
so the small rungs come out expensive. Measured on a real 1920×960 ladder before
this existed:

| rung | pixels | bytes/pixel |
|---|---|---|
| 1920×960 | 1,843,200 | 56 |
| 1440×720 | 1,036,800 | 73 |
| 960×480 | 460,800 | 117 |
| 720×360 | 259,200 | 145 |
| 480×240 | 115,200 | **228** |

The 240p rung — the one that exists so a phone on a train can play *something* —
spent four times the bits per pixel of the rung most people watch, and the four
rungs below the top were 65% of the storage for the upload.

### The shape

The fix is not a special case for "ICQ +2 per rung" inside an adapter. It is
that the **caller** — which knows what a rung is *for*, and which rung is the
top one — gets to say so, and [`tuning::overrides`](../crates/codec/src/encode/tuning/overrides.rs)
is the vocabulary it says it in.

- [`EncodeOverrides`] is the knob set. Every field is optional; `None` means
  "leave whatever the target and tier chose". An empty override is exactly the
  behaviour above, and that is a *tested property* across all four backends ×
  every target × every tier × six resolutions — the mechanism is only safe if
  the empty case is provably inert.
- [`RungPolicy`] resolves overrides for one rung: a global set, then any number
  of [`RungRule`]s whose [`RungSelector`] matches, in declaration order, later
  wins.

```rust
use codec::encode::tuning::{EncodeOverrides, RungPolicy, RungSelector, TileGrid};

// Softer going down the ladder, sharper at the top, one tile below 4K.
let policy = RungPolicy::new()
    .with_quality_step_per_rung(2)                    // +2 per position below the top, compounding
    .with_rule(RungSelector::Top,
               EncodeOverrides { quality_delta: -2, ..Default::default() })
    .with_rule(RungSelector::ShortSideAtMost(2159),
               EncodeOverrides { tiles: Some(TileGrid::SINGLE), ..Default::default() });
```

The caller then resolves per rung and hands the result to `EncoderConfig`:

```rust
let overrides = policy.resolve(&RungContext {
    width: rung.width, height: rung.height, index, rung_count,
});
let config = EncoderConfig { overrides, ..base };
```

### Quality is denominated in libaom-CQ-equivalent steps

`quality_delta` is the one field that needs explaining, and the reason is that
the backends disagree about units *and* direction:

| Backend | Native scale | Direction |
|---|---|---|
| QSV (oneVPL) | ICQ 1..51 | up = worse |
| NVENC, constant-QP | CQ 0..63 | up = worse |
| NVENC, VBR | `targetQuality` 0..100 | up = **better** |
| AMF AV1, constant-QP | `q_index` = libaom × 4 − 8 | up = worse |
| AMF H.264 / H.265 | QP 0..51 | up = worse |
| AMF, QVBR (all codecs) | `QvbrQualityLevel` 1..51 = 52 − QP | up = **better** (measured) |
| rav1e | quantizer 0..255, ≈ 4× libaom | up = worse |

A raw "native units" delta would mean five different things. libaom CQ is the
currency this module already converts through, so it is the currency here:
**positive is always smaller-and-worse, on every backend**, each adapter
converts into its own scale and applies whatever sign that scale needs. A caller
says "+2 softer" once and it means the same change on an Arc and on a 5060.

### Two traps worth knowing

**The CRF escape hatch bypasses all of it — unless you fold it.**
`EncoderConfig::quality` is documented as "a CRF, or `AUTO_FROM_TARGET` to
derive one from `target`", and every backend honours that by skipping the whole
`tuning` path when a real CRF is present — which is where the delta would be
applied. A caller setting both a CRF and a policy therefore got the CRF and
silently none of the policy. `select_encoder` now folds the delta into the CRF
as well, in one place, and the two paths are mutually exclusive by construction:
a real CRF means the adapters were never consulted. If you are wondering why a
policy appears to do nothing, this is the first thing to check — and note the
corollary, that `target` and `tier` are equally inert for such a caller.

**Buffering knobs are requests, not instructions.** `lookahead_frames`,
`bframes` and `reference_frames` all make the encoder *hold input surfaces*. An
encoder whose surface pool assumes one-in-one-out will hand the next frame the
same memory, which is a silently corrupted picture rather than an error — it
was exactly that bug that had lookahead and multi-pass disabled across every
backend for a while. Only set them on a backend whose pool selects by "the
runtime has released this". Hardware may also simply refuse: oneVPL's
`MFXVideoENCODE_Query` adjusts the parameters and the code uses the adjusted
struct, so ask, then read back what you got.

### What is plumbed and ignored

`film_grain`. AV1 film-grain synthesis is real and would be a genuine
perceived-quality win at negative bitrate cost, but NVENC exposes
`enableFilmGrainParams` and **will not analyse grain for you** — the caller must
hand it a populated `NV_ENC_FILM_GRAIN_PARAMS_AV1`, i.e. write a grain
estimator — and oneVPL's vendored headers expose no equivalent at all. The knob
exists so the plumbing does and the gap is visible in the type rather than in
somebody's memory; the adapters currently ignore it rather than pretending.

### Opt-in tools in the software tier: `aq` and `wp` (measured, off by default)

The native H.264 and H.265 encoders have two tools the software tier uses only
when asked: **adaptive quantisation** (`aq=<strength>`, 0.0–4.0 — a quantiser
offset per H.265 CTB or H.264 macroblock from luma variance, flat blocks finer,
textured coarser, zero-mean over the picture) and **weighted prediction**
(`wp=on` — a weight and offset per P picture, fitted against the reference and
used where it lowers the residual). They are [`EncodeOverrides`] fields
(`aq_strength_tenths`, `weighted_pred`), spelled in the policy grammar:
`--encode-policy "any:wp=on"`, `"short>=720:aq=1.0"`. Off at every quality
target for both codecs; off, the stream is byte-identical to the tier before
they existed — the control arm `any:aq=0,wp=off` was `cmp`-equal to no policy in
every cell of both measurements below. The hardware backends ignore them. The
H.264 encoder gained both in h26x `d1471ce`; before that bump this tier logged
and dropped them for H.264. **Lookahead is not one of them:** it informs a rate
controller, and this tier is constant-QP, so there is none to inform — the
encoders refuse a lookahead without a bitrate target (H.264 refuses one outright,
uncalibrated), and the tier logs and ignores `lookahead=` rather than inventing
a target.

#### H.265

**How it was measured.** rivet's HLS ladder path on the software pool, one
640x360 rung, 1 s segments, `--codec h265 --target high|standard|low` (QP 22 /
26 / 32), one binary (`397eb96`), knob on vs off. Four 4 s, 30 fps lavfi clips:
`fade` (testsrc2 fading in over 1.5 s and out over 1.5 s), `flat` (a slow
`gradients` field), `busy` (testsrc2 + temporal noise), `flatbusy` (gradient
left half, noisy texture right half). Every output decoded by ffmpeg with no
error output; luma PSNR by frame **index** against ffmpeg's decode of the source.
Size is every init + segment byte. *ΔY at equal size* is the knob-on PSNR minus
the knob-off arm's PSNR interpolated at the same size along its three-QP curve
(linear in log bytes; `*` extrapolated) — what separates "better" from "smaller".
On `flat` the knob-off curve is not monotonic (QP 32 is both smaller and 3 dB
worse than QP 26), so that column means nothing there.

Adaptive quantisation, Δ against knob-off at the same QP (strength 0.5 / 1.0):

| clip | target (QP) | size | ΔY PSNR | ΔY at equal size | Δ flat half / busy half (1.0) |
|---|---|---:|---:|---:|---:|
| fade | high (22) | −7.1% / −15.0% | −1.00 / −2.25 | −0.26 / −0.62 | — |
| fade | standard (26) | −6.8% / −13.8% | −1.00 / −2.17 | −0.27 / −0.65 | — |
| fade | low (32) | −6.0% / −10.1% | −0.83 / −1.72 | −0.20* / −0.63* | — |
| busy | high (22) | −1.4% / −2.4% | −0.21 / −0.43 | −0.09 / −0.23 | — |
| busy | standard (26) | −2.0% / −3.3% | −0.18 / −0.34 | −0.11 / −0.23 | — |
| busy | low (32) | −3.6% / −5.0% | −0.07 / −0.15 | +0.05* / +0.01* | — |
| flatbusy | high (22) | −8.5% / −18.2% | −0.61 / −1.33 | −0.28 / −0.57 | +0.67 / −1.35 |
| flatbusy | standard (26) | −11.2% / −23.7% | −0.39 / −0.85 | −0.15 / −0.31 | +0.45 / −0.86 |
| flatbusy | low (32) | −11.2% / −17.9% | −0.47 / −0.98 | −0.23* / −0.59* | +1.00 / −1.00 |
| flat | all three | +0.6% to +0.7% | 0.00 | — | 0.00 / 0.00 |

Per-frame spread on `fade` (standard deviation of per-frame luma PSNR, off →
0.5 / 1.0): 4.80 → 4.99 / 5.25 at QP 22, 5.25 → 5.46 / 5.74 at 26, 5.78 → 5.95 /
6.21 at 32; the worst frame drops 0.88–1.11 dB at 0.5 and 2.18–2.65 dB at 1.0. On
`busy` and `flatbusy` the across-frame spread moves by 0.03 dB or less. The mean
within-picture spread of 16x16-block PSNR *rose* with AQ on every clip that has
texture (+0.1 to +1.6 dB), but flat blocks that decode exactly score 99 dB and
dominate that statistic; the flat / busy halves are the honest view of the
redistribution.

Weighted prediction, Δ against knob-off at the same QP:

| clip | target (QP) | size | ΔY PSNR | ΔY worst frame | ΔY at equal size |
|---|---|---:|---:|---:|---:|
| fade | high (22) | −4.7% | −0.09 | +0.00 | +0.39 |
| fade | standard (26) | −5.1% | −0.03 | +0.00 | +0.51 |
| fade | low (32) | −5.0% | +0.04 | +0.00 | +0.57* |
| busy, flatbusy | all three | +116 bytes (+0.0%) | 0.00 | 0.00 | — |
| flat | high / standard | +108 / +77 bytes (+0.1%) | 0.00 | 0.00 | — |
| flat | low (32) | −0.5% | −0.02 | 0.00 | — |

Encode time, paired, one binary, five reps in alternating order, whole
`rivet transcode` wall clock at `standard`: on `fade` (knob-off median 1500 ms)
the median paired ratio is 0.992 for `wp` and 0.990 for `aq=1.0`. On `busy` every
wall time lands near 2.0 s or near 2.5 s — a step in the pipeline, not the
encoder — and the paired ratios straddle it (`wp` 0.80–1.26, `aq` 0.79–1.03), so
that clip measured neither tool's cost.

**Decision: both stay off at every target.**

- **`aq` stays off.** At every target it buys size with PSNR and loses at
  equal size, 0.1–0.65 dB on `fade`, `busy` and `flatbusy`; on a flat picture it
  changes no pixel and costs the `cu_qp_delta` syntax (+0.6–0.7%). What it does
  deliver is the redistribution it exists for — on `flatbusy` the flat half gains
  0.45–1.0 dB while the busy half pays 0.4–1.35 dB, at 8–24% fewer bytes. That is a
  perceptual trade (banding and blocking in flat areas against invisible loss in
  texture) which PSNR cannot credit and nothing here measures, so it is a knob
  for a caller who wants that trade, not a default.
- **`wp` stays off, and is the one worth turning on for content with fades.**
  On the fade it is about 5% smaller at the same PSNR at every target (+0.4 to
  +0.6 dB at equal size); without a fade it costs about 116 bytes per 4 s (the
  per-P-slice table) with PSNR unchanged. It is not a default because the
  evidence is one synthetic fade and one inconclusive timing clip, and a default
  changes every software H.265 stream's bytes; `any:wp=on` is one word away.

#### H.264

**How it was measured** (2026-09-14, h26x `1092d4c`). The serial single-file
path, `TRANSCODE_ENCODER_BACKEND=h26x`, `--codec h264 --target
high|standard|low` (QP 22 / 26 / 32), one binary, knob on vs off. Four 640x360,
30 fps, 4 s clips: `fade` (testsrc2 fading in over 1.5 s and out over 1.5 s),
`testsrc2`, `zoom` (a mandelbrot zoom with temporal grain and a slight blur)
and `pan` (a blurred mandelbrot field panned at 300 px/s, with grain). Every
output 120 / 120 frames, ffmpeg's full decode printed nothing, and every rep's
md5 matched the first. Luma PSNR by frame **index** against ffmpeg's decode of the
source. Size is the video stream's packet bytes. *ΔY at equal size* is as
above (`*` extrapolated).

Adaptive quantisation, Δ against knob-off at the same QP (strength 0.5 / 1.0):

| clip | target (QP) | size | ΔY PSNR | ΔY at equal size |
|---|---|---:|---:|---:|
| fade | high (22) | −5.9% / −12.3% | −1.16 / −2.10 | −0.42 / −0.51 |
| fade | standard (26) | −5.1% / −13.7% | −0.96 / −2.46 | −0.38 / −0.86 |
| fade | low (32) | −5.1% / −12.8% | −0.88 / −1.97 | −0.31* / −0.48* |
| testsrc2 | high (22) | −7.7% / −15.1% | −1.04 / −2.77 | +0.06 / −0.53 |
| testsrc2 | standard (26) | −6.6% / −15.6% | −0.93 / −2.40 | −0.04 / −0.19 |
| testsrc2 | low (32) | −10.5% / −22.1% | −1.41 / −3.10 | +0.02* / +0.14* |
| zoom | high (22) | −8.7% / −16.6% | −0.82 / −1.67 | −0.17 / −0.39 |
| zoom | standard (26) | −8.9% / −20.9% | −0.79 / −1.86 | −0.18 / −0.33 |
| zoom | low (32) | −12.4% / −24.5% | −0.73 / −1.67 | +0.13* / +0.16* |
| pan | high (22) | −1.8% / +5.3% | −0.22 / −0.36 | −0.19 / −0.44* |
| pan | standard (26) | −1.4% / −1.4% | −0.27 / −0.65 | −0.20 / −0.58 |
| pan | low (32) | −1.0% / −2.9% | −0.47 / −1.03 | −0.42* / −0.89* |

Weighted prediction, Δ against knob-off at the same QP:

| clip | target (QP) | size | ΔY PSNR | ΔY worst frame | ΔY at equal size |
|---|---|---:|---:|---:|---:|
| fade | high (22) | −9.8% | +0.02 | −0.19 | +1.27 |
| fade | standard (26) | −10.3% | +0.11 | −0.12 | +1.29 |
| fade | low (32) | −12.3% | +0.33 | −0.02 | +1.76* |
| testsrc2, zoom, pan | all three | −39 to +795 bytes (−0.02% to +0.07%) | 0.00 | 0.00 | — |

Encode time is not resolved: whole-`rivet transcode` wall and CPU, three paired
reps, give median ratios from 0.76 to 1.36 for the knobs with no direction, and
the explicit-off control, which codes the same bytes, spans 0.64 to 1.09 by
itself.

**Decision: both stay off at every target, as for H.265.**

- **`aq` stays off.** It buys size with PSNR at every target and loses at equal
  size on `fade`, `zoom` and `pan` (−0.2 to −0.9 dB at strength 1.0). The few
  non-negative cells are extrapolated or within 0.06 dB. The perceptual case is
  the one made for H.265 above, and nothing here measures it.
- **`wp` stays off.** On the fade it is 10–12% smaller at the same or better
  PSNR (+1.3 to +1.8 dB at equal size), more than it bought H.265. Without a fade it changes the
  size by a table per P slice and no pixel. It is not a default for the H.265
  reason — one synthetic fade, and a default changes every software H.264
  stream's bytes. `any:wp=on` is the knob for content with fades.

[`EncodeOverrides`]: ../crates/codec/src/encode/tuning/overrides.rs
[`RungPolicy`]: ../crates/codec/src/encode/tuning/overrides.rs
[`RungRule`]: ../crates/codec/src/encode/tuning/overrides.rs
[`RungSelector`]: ../crates/codec/src/encode/tuning/overrides.rs

---

## Colorspace: normalizing decoder frames for the encoder

> Source: [`crates/codec/src/colorspace.rs`](../crates/codec/src/colorspace.rs)

### What

AV1 encoders accept 4:2:0 only (8-bit BT.709 limited, or 10-bit for HDR
passthrough). Decoders emit a zoo of layouts — NV12/NV21, 4:2:2, 4:4:4, RGB,
8-bit, 10-bit, BT.601/709/2020. This module is the funnel. Two public entry
points:

- [`convert_to_yuv420p_bt709`](../crates/codec/src/colorspace.rs#L80) — the
  8-bit-aware normalizer. Dispatches by format: 10-bit/wide-gamut passes through
  on the matrix axis (chroma layout still normalized to 4:2:0); RGB goes through
  a BT.709 RGB→YUV matrix; YUV chroma layouts are deinterleaved/averaged to
  4:2:0; then a BT.601→709 matrix correction runs for any non-709-tagged YUV
  source. The full input→output coverage table is in the function's doc comment
  ([colorspace.rs:23-37](../crates/codec/src/colorspace.rs#L23)).
- [`convert_to_sdr_bt709`](../crates/codec/src/colorspace.rs#L49) — the
  **HDR-aware** dispatch the pipeline calls when it has the source
  `ColorMetadata`. PQ/HLG + `Yuv420p10le` → tonemap to 8-bit BT.709 (see next
  section); everything else falls through to `convert_to_yuv420p_bt709` with SDR
  semantics unchanged.

Plus the scaler: [`scale_frame`](../crates/codec/src/colorspace.rs#L1302)
bilinear-scales `Yuv420p` / `Yuv420p10le` to the rung's dimensions (an identity
fast-path returns a cheap clone when dims already match).

### Why & the AVX2 runtime-dispatch pattern

The hot kernels — BT.601→709 matrix, 4:4:4→4:2:0 downsample, bilinear scale —
each ship as a **scalar reference** plus an `#[target_feature(enable = "avx2")]`
SIMD specialization, behind a safe public dispatcher that runtime-detects AVX2
(`is_x86_feature_detected!("avx2")`) and falls back to scalar otherwise
([bt601_to_bt709_planes](../crates/codec/src/colorspace.rs#L536),
[bilinear_scale_plane_u16](../crates/codec/src/colorspace.rs#L1517)). The CPUID
check is the safety boundary for the `unsafe` SIMD fn. This is the project-wide
AVX dispatch convention (`feedback_avx_runtime_dispatch.md`): runtime-detect,
keep a scalar fallback, only specialize loops that actually bench hot. The scalar
path stays `pub` so benches and non-x86 builds can target it directly.

Notable decisions:

- **BT.601→709 is a delta-space matrix with no luma-into-chroma coupling.** The
  3×3 is derived by composing BT.601 YUV→RGB with BT.709 RGB→YUV in limited-range
  form; the derivation and a black/white/gray round-trip sanity check are written
  out in the source ([colorspace.rs:410-464](../crates/codec/src/colorspace.rs#L410)).
  The AVX2 kernel uses `_mm256_mulhrs_epi16` for Q15 fixed-point multiplies and
  splits off the identity contribution for the ~1.0 coefficients that overflow
  i16 ([colorspace.rs:583-592](../crates/codec/src/colorspace.rs#L583)).
- **10-bit BT.601→709 exists but is off the default path.** The 10-bit pipeline
  is HDR-passthrough/tonemap, never matrix-converted (a BT.601 matrix would
  corrupt a wide gamut). The 10-bit converter is wired behind a public entry for
  explicitly-tagged BT.601 10-bit content (some Sony broadcast cameras) but
  callers must opt in ([colorspace.rs:776-782](../crates/codec/src/colorspace.rs#L776)).
- **4:4:4 → 4:2:0 is a 2×2 box average by default, with a Lanczos-2 option.**
  The box is sited at the centre of the 2×2 block (JPEG / MPEG-1 siting) and
  keeps every output byte-identical to earlier releases. `chroma-downsample=lanczos`
  (`--chroma-downsample lanczos` on the CLI, the same key on the API / manifest /
  IPC) runs a separable Lanczos-2 (`downsample_fir.rs`: Q6 taps
  `[-2 0 18 32 18 0 -2]/64` horizontally, co-sited with the even luma column;
  `[-3 7 28 28 7 -3]/64` vertically, midway between the rows — the
  `chroma_sample_loc_type 0` siting H.264 / HEVC infer when nothing is
  signalled), scalar + AVX2 (bit-exact; 720p Cb+Cr: box 0.81 ms, Lanczos
  scalar 2.12 ms, Lanczos AVX2 0.93 ms — 1.16× the box).
  Measured against libswscale on a 1280×720 4:4:4 `testsrc2` (15 frames, Cb/Cr
  PSNR): our Lanczos matches swscale's bicubic told to site left at 61.2 / 58.6 dB
  and its lanczos-left at 54.2 / 50.7; the box matches swscale's centre-sited
  bicubic at 50.9 / 47.1. Round-tripping 4:2:0 → 4:4:4 through swscale's bicubic
  upsampler *at the candidate's own siting* against the original 4:4:4 chroma:
  box 47.6 / 43.7, Lanczos 42.7 / 39.0, swscale bicubic-left 42.8 / 39.1,
  swscale lanczos-left 43.6 / 39.8, swscale lanczos-centre 46.0 / 42.2 — on that
  metric no filter beats the box, including a centre-sited Lanczos (44.9 / 41.1)
  and swscale's own; what dominates is siting: a box output read as left-sited,
  or a Lanczos output read as centre-sited, drops to 39.0 / 35.6. On a
  `mandelbrot` source everything lands within 0.1 dB (aliasing-bound). So the
  earlier "~0.3 dB chroma PSNR for a FIR" note did not survive measurement; the
  option's value is the siting (for consumers that follow the spec default) and
  alias suppression, not a round-trip PSNR gain. Alpha (from `Yuva444p10le`,
  i.e. ProRes 4444) is **dropped** — the 4:2:0 encoder format has no alpha and
  rav1e/HW don't expose AV1's experimental alpha
  ([colorspace.rs:1069-1105](../crates/codec/src/colorspace.rs#L1069)).
- **Matrix is preserved on passthrough, not silently rewritten.** 10-bit/wide-gamut
  frames keep their `color_space`; the encoder signals it in the AV1 sequence
  header and the mux writes `colr nclx`, so a player can reverse the matrix. The
  one exception is 8-bit BT.2020 (rare), which routes through the BT.601 matrix
  with a documented slight hue shift rather than bailing
  ([colorspace.rs:122-134](../crates/codec/src/colorspace.rs#L122)).

---

## Tonemapping & the single-output policy

> Source: [`crates/codec/src/tonemap.rs`](../crates/codec/src/tonemap.rs)

### What

[`tonemap_yuv420p10le_bt2020_to_yuv420p_bt709`](../crates/codec/src/tonemap.rs#L238)
maps a 10-bit BT.2020 PQ/HLG frame down to an 8-bit BT.709 limited-range frame.
The pipeline (per pixel) is: 10-bit Y'CbCr → R'G'B' (BT.2020 NCL matrix) →
scene-linear RGB (PQ or HLG inverse EOTF) → BT.709 gamut → **Hable filmic curve**
→ BT.709 OETF → 8-bit BT.709 limited Y'CbCr
([tonemap.rs:1-9](../crates/codec/src/tonemap.rs#L1)). Chroma is downsampled by
averaging the four per-pixel post-tonemap chroma values per 2×2 block (rather
than tonemapping once per chroma site), which avoids hue shifts at high
luminance ([tonemap.rs:233-237](../crates/codec/src/tonemap.rs#L233)).

`convert_to_sdr_bt709` (above) is the caller; the scene-linear white point comes
from the source's mastering-display `max_luminance` when present, else a
1000-nit HDR10 default ([tonemap.rs:221](../crates/codec/src/tonemap.rs#L221)).

### Why the single-output tonemap-to-SDR policy

Stated in the module header
([tonemap.rs:8-12](../crates/codec/src/tonemap.rs#L8)) and the
[README's web-defaults pitch](../README.md): every HDR upload is tonemapped to
SDR at transcode time and the encoded ABR ladder is 8-bit BT.709, so **every
viewer sees a correctly-mapped image regardless of display capability**. Shipping
native HDR without the upstream UI/processing work (YouTube/Instagram have given
whole talks on it) lands badly-converted, eye-searing or washed-out clips on
viewers. HDR-fidelity-for-HDR-viewers is a future dual-rendition path that reuses
these same primitives for the SDR rungs; the latent passthrough paths (10-bit
encode, `mdcv`/`clli` mux atoms, sequence-header HDR signaling) all stay in tree
and re-engage if a creator-opt-in HDR mode ships.

Two implementation "why"s worth flagging:

- **The HLG path applies an OOTF (γ=1.2), not just the inverse OETF.** HLG
  signals are *scene*-referred; without the scene→display OOTF, midtones land in
  the wrong place — this is exactly why iPhone HLG clips famously read ~1 stop
  too bright on naive pipelines (the camera assumes Apple's downstream tonemapper
  applies it) ([tonemap.rs:56-104](../crates/codec/src/tonemap.rs#L56)).
- **Hable's coefficients + exposure bias 2.0 are the published values
  verbatim** ([tonemap.rs:121-146](../crates/codec/src/tonemap.rs#L121)),
  cross-checked against `libavfilter`'s `tonemap_hable` numbers — reference
  comparison only, no FFmpeg link-time dependency.
- **Scalar reference + AVX2/FMA kernel, runtime-dispatched.** The scalar f32
  path is the reference; the AVX2 path does the same arithmetic eight pixels at
  a time with Cephes `exp`/`log` polynomials for the transcendentals, and agrees
  with the reference to **≤ 1 LSB per 8-bit sample** (checked over every 10-bit
  luma code against a chroma grid for PQ and HLG in the unit tests, and on real
  clips by `cargo run --release --example tonemap_ab`: 361 of 31.1 M samples
  differ by one code on a 1080p PQ clip, none by more). Measured on the dev box
  (release, paired, alternating order, after a scalar-vs-scalar control whose
  spread was 0.79..1.10): 1080p PQ 223 → 39 ms/frame, 1080p HLG 220 → 36,
  4K PQ 1183 → 190 — median speedup 5.5–5.8×. The old claim that scalar fit a
  1080p60 budget was not true (≈4.5 fps single-threaded); AVX2 lands ≈25 fps at
  1080p per thread. `RIVET_TONEMAP_SCALAR=1` forces the reference, and the
  dispatcher logs `HDR → SDR tonemap kernel selected path=…` once.
- **Two reference fixes came out of matching the paths.** The PQ/HLG signal is
  clamped to its [0, 1] domain before the inverse EOTF (matrix overshoot just
  above 1 used to blow the Hable curve up to NaN, which `as u8` turned into
  Y = 0), and an out-of-gamut negative BT.709 channel is clipped *before* the
  Hable curve, whose rational form has a pole at x ≈ −0.062 — clipping only at
  the OETF, after the curve, sent such a channel to white.

---

## The audio pipeline: decode → Opus transcode

> Source: [`crates/codec/src/audio/`](../crates/codec/src/audio/mod.rs)

### What

The audio side is a small decode→encode framework. The
[pipeline routing](pipeline.md#7-audio) decides per source codec:

| Source | Action | Output |
|--------|--------|--------|
| AAC, Opus, AC-3, E-AC-3 | **Passthrough** (no decode) | carried verbatim into the container |
| MP3, Vorbis | **Decode → re-encode to Opus** | Opus + `dOps` |
| AC-3, E-AC-3 with `--audio opus` or an audio filter | **Decode → re-encode to Opus** ([in-tree decoder](codec-decode.md#ac-3--e-ac-3-decoder)) | Opus + `dOps` |
| everything else | **Drop** (video-only, warn) | — |

This crate owns the middle row. The wire model
([audio/mod.rs](../crates/codec/src/audio/mod.rs)):

- [`AudioFrame`](../crates/codec/src/audio/mod.rs#L58) — interleaved f32 PCM in
  [-1.0, 1.0] (`LRLR…`) + rate/channels + µs PTS. The canonical exchange type.
- [`AudioDecoder`](../crates/codec/src/audio/mod.rs#L99) /
  [`AudioEncoder`](../crates/codec/src/audio/mod.rs#L109) — object-safe traits;
  `create_decoder("mp3"|"vorbis", …)` and `create_encoder(AudioCodec::Opus)` are
  the routing entry points ([audio/mod.rs:141-168](../crates/codec/src/audio/mod.rs#L141)).
  `AudioCodec` has exactly one variant: `Opus`.

Decoders:

- [`Mp3Decoder`](../crates/codec/src/audio/decode/mp3.rs) wraps `minimp3`
  (MIT C lib via FFI). It adapts minimp3's `io::Read` model to a packet-in
  trait with an internal compacting byte cursor, tolerates ID3 prefixes / sync
  errors, and derives PTS from the per-frame sample count (1152 for MPEG-1, 576
  for MPEG-2).
- [`VorbisDecoder`](../crates/codec/src/audio/decode/vorbis.rs) wraps `lewton`
  (pure-Rust). It takes MKV's `CodecPrivate` (the three Xiph-laced setup headers)
  as `extra_data`, parses the Xiph lacing
  ([vorbis.rs:169](../crates/codec/src/audio/decode/vorbis.rs#L169)), and uses
  lewton's per-packet API.

Encoder + resampler:

- [`OpusEncoder`](../crates/codec/src/audio/encode/opus.rs) wraps `audiopus`
  (libopus FFI). It always runs libopus **internally at 48 kHz** (resampling the
  input via [`AudioResampler`](../crates/codec/src/audio/resample.rs) when the
  source rate differs), uses **20 ms / 960-sample** frames, and emits the `dOps`
  config body ([build_dops](../crates/codec/src/audio/encode/opus.rs#L595)) +
  `pre_skip` (48 kHz lookahead ticks) the mux side needs per RFC 7845. Mono/stereo
  use the regular libopus encoder; 3–8 channels (5.1/7.1) use the libopus
  **Multistream** API with RFC 7845 §5.1.1.2 channel-mapping family 1; >8 channels
  is `Unsupported`. Family 1 orders its channels as Vorbis does (5.1 = FL FC FR RL
  RR LFE) while the pipeline carries ffmpeg's native order (FL FR FC LFE BL BR),
  so each 20 ms frame is permuted in place before `opus_multistream_encode_float`
  (`audio::rfc7845_family1_order`); the round-trip test decodes through the
  multistream decoder and checks each RFC channel against the native slot it
  must carry, so a dropped permutation fails it.
- [`AudioResampler`](../crates/codec/src/audio/resample.rs) wraps rubato's
  `SincFixedIn` (band-limited windowed sinc), deinterleaving in / re-interleaving
  out since rubato wants planar.

### Why

- **Why Opus, and why it's royalty-clean.** The audio-expansion decision (per
  `audio/mod.rs:1-9`) picked Opus over AAC because **libopus is BSD and audiopus
  is ISC** — no Fraunhofer license, unlike `fdk-aac` — and modern browsers all
  play Opus-in-MP4. This is the audio half of the project's royalty posture: AV1
  video + Opus audio + MP4 container = zero royalty exposure on output. AAC
  passthrough stays royalty-clean precisely *because* it's a pure byte transmux —
  we never decode or encode AAC, so no codec license is engaged. Force-Opus
  (dropping AAC passthrough) was rejected because it would require an AAC
  *decoder* dependency, reintroducing the Fraunhofer problem.
- **Why 48 kHz internal + own resampler.** Keeping libopus at a fixed 48 kHz
  makes `pre_skip` semantics uniform (always reported in 48 kHz ticks per the
  RFC) and lets the `dOps` `InputSampleRate` field cleanly carry the *original*
  source rate ([opus.rs:6-15](../crates/codec/src/audio/encode/opus.rs#L6)).
- **Why `Application::Audio` and VBR.** Tuned for fidelity over latency (vs Voip
  / LowDelay) — this is offline transcode, so the ~26 ms one-way latency from a
  20 ms frame + libopus lookahead is irrelevant
  ([opus.rs:29-31](../crates/codec/src/audio/encode/opus.rs#L29)).
- **Why the PTS/pre_skip plumbing matters.** Resampling and the libopus encoder
  both add lookahead; the design collapses all of it into the single `pre_skip`
  count written into `dOps`, so a conformant decoder discards the right amount of
  front padding and downstream callers see no PTS drift
  ([resample.rs:21-25](../crates/codec/src/audio/resample.rs#L21)).

---

## Key decisions on the encode side (recap)

- **AV1-default output (H.264 / H.265 also selectable), GPU-only encode.** No CPU
  encode tier — `select_encoder` hard-fails on a host without NVENC/AMF/QSV
  encode silicon rather than degrading to a 20× slower software path — unless
  the build opted into `rav1e-fallback` (AV1) / `h26x-fallback` (H.264 /
  H.265), which sit *below* the vendor chain so they are a floor, never a
  preference.
- **Layered vendor encoders, stubbed when off.** Each is hand-rolled in-tree FFI
  that builds cross-platform; a stub type keeps the dispatcher `#[cfg]`-free and
  turns "feature not compiled" into a clear error instead of a link failure.
- **Perceptual targets, not raw CRF.** `QualityTarget`/`SpeedTier` map to native
  knobs via libaom-referenced, per-vendor-calibrated tables, so the same job
  looks the same across NVENC/AMF/QSV. HW tile grids cap at 2×2; no
  low-latency presets.
- **Per-vendor gotchas are load-bearing.** QSV AV1 is VDENC-only (`LowPower` ON);
  QSV ICQ is mode 9 (8 is lookahead); QSV pads to 16-multiple coded dims and
  pre-fills surfaces with neutral black to avoid green bars; AMF treats
  `AMF_INPUT_FULL` as a transient retry, not a failure.
- **AVX2 with scalar fallback, runtime-dispatched.** Every hot colorspace/scale
  kernel keeps a scalar reference and a CPUID-gated AVX2 specialization behind a
  safe wrapper.
- **HDR tonemapped to SDR by default.** Single-output policy: one correctly-mapped
  8-bit BT.709 ladder for every viewer; the HLG OOTF and Hable curve are the
  reason iPhone HLG doesn't come out a stop too bright. Passthrough paths stay
  latent.
- **Royalty-clean audio.** Opus (BSD/ISC libs) for transcode + AAC/Opus/AC-3/E-AC-3
  passthrough; no `fdk-aac`, no Fraunhofer exposure.
