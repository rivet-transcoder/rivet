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
   hand-rolled, in-tree `dlopen` FFI encoder (NVENC / AMF / QSV). They *stack*
   with an optional FFmpeg tier on top; CPU is the last resort. New tiers add to
   the chain, they don't replace it. (As of 2026-05-08 the rav1e CPU and Vulkan
   encode tiers were **removed** — the build is GPU-encode-only; see
   [select_encoder](#the-encode-dispatch--capability-query).)
3. **HDR is tonemapped to SDR by policy.** The default single-output policy maps
   every HDR source down to 8-bit BT.709 at transcode time so a clip never lands
   eye-searingly bright on a viewer's screen. HDR-passthrough is a latent,
   policy-gated path, not the default. See [Tonemapping](#tonemapping--the-single-output-policy).

---

## Module map

| File | Purpose |
|------|---------|
| [`encode/mod.rs`](../crates/codec/src/encode/mod.rs) | The `Encoder` trait, `EncoderConfig`, `select_encoder` dispatch, `OutputCaps` runtime capability query, the `TRANSCODE_ENCODER_BACKEND` override. |
| [`encode/tuning.rs`](../crates/codec/src/encode/tuning.rs) | Backend-agnostic `QualityTarget` / `SpeedTier` → per-encoder knobs (CQ, q-index, ICQ, presets, tile grid). The calibration layer. |
| [`encode/nvenc.rs`](../crates/codec/src/encode/nvenc.rs) + [`nvenc_stub.rs`](../crates/codec/src/encode/nvenc_stub.rs) | NVENC AV1 encoder (NVIDIA Ada+), hand-rolled `nvEncodeAPI` FFI. Stub when `nvidia` is off. |
| [`encode/amf.rs`](../crates/codec/src/encode/amf.rs) + [`amf_stub.rs`](../crates/codec/src/encode/amf_stub.rs) | AMF AV1 encoder (AMD RDNA3+), hand-rolled AMF runtime FFI. Stub when `amd` is off. |
| [`encode/qsv.rs`](../crates/codec/src/encode/qsv.rs) + [`qsv_stub.rs`](../crates/codec/src/encode/qsv_stub.rs) | QSV AV1 encoder (Intel Arc / Meteor Lake+), hand-rolled oneVPL FFI. Stub when `qsv` is off. |
| [`encode/ffmpeg_enc.rs`](../crates/codec/src/encode/ffmpeg_enc.rs) | libavcodec AV1 encoder catalogue (HW + SW) behind one interface. Gated on `ffmpeg`. |
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
| AMF (AMD RDNA3+) | ✅ | ❌ rejected — native `VCE_AVC` / HEVC component is a follow-up |
| ffmpeg | ✅ | ❌ rejected — `h264_*`/`hevc_*` dispatch is a follow-up |

H.264/H.265 encoders emit **Annex-B** NAL; the muxer's
[`nal_mux`](../crates/container/src/nal_mux.rs) splits each packet into per-frame
access units (HW encoders pack several frames per buffer), captures SPS/PPS(/VPS)
for the `avcC`/`hvcC` config box, and repackages slices as length-prefixed
samples (`avc1`/`hvc1`). AMF/ffmpeg reject H.264/H.265 rather than silently emit
AV1.

### Bit depth (H.265 8/10-bit, H.264 8-bit only)

**H.265 encodes 8- or 10-bit** (Main / Main 10, 4:2:0) on NVENC and QSV, both
hardware-validated — `with_bit_depth(TenBit)` or a HDR `ColorPolicy` produces a
genuine Main 10 stream:
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

The muxer's `build_hvcc` parses the bit depth from the SPS, so the `hvcC` carries
`bitDepthLumaMinus8 = 2` for Main 10.

**H.264 is 8-bit only.** Neither NVENC (no `High 10` profile GUID) nor QSV (no
`AVC High 10` in oneVPL) exposes a hardware Hi10P encoder, so a 10-bit H.264
request is **capability-rejected** with a clear error ("does not support 10-bit
H264 encode") rather than silently down-converted to 8-bit.

**NVENC H.264/H.265** uses the codec's GUID for capability validation, preset
selection, and session init; the preset (`GetEncodePresetConfigEx`) seeds the
codec config union so the H264/HEVC layout doesn't have to be mirrored. H.264 is
pinned to High profile, H.265 to Main. The encoder is forced **strictly
1-in-1-out** for H.264/H.265 (clear `enableLookahead`, set `zeroReorderDelay`,
no B-frames) because the ring-of-4 sync drain emits one packet per
`EncodePicture`; lookahead/reorder buffering would otherwise strand the tail
frames or deadlock the EOS flush. When every frame is drained during encode, the
EOS flush is skipped (sending it busy-waits on the SDK 13 driver).

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

1. **FFmpeg** (`ffmpeg` feature, unless `DISABLE_FFMPEG`) — tier 0, one
   interface over `av1_nvenc` / `av1_amf` / `av1_qsv` / `av1_vaapi` / `libsvtav1`
   / `libaom-av1` / `librav1e` ([mod.rs:300-322](../crates/codec/src/encode/mod.rs#L300)).
2. **Vendor-pin shortcut** — if `config.gpu_vendor` is set (the CMAF
   orchestrator does this via the `GpuPool` lease), dispatch *directly* to that
   vendor's backend, skipping the preference chain
   ([mod.rs:333-378](../crates/codec/src/encode/mod.rs#L333)).
3. **Auto-select chain** — NVENC (Ada+) → AMF (RDNA3+) → QSV (Arc / Meteor
   Lake+) ([mod.rs:388-460](../crates/codec/src/encode/mod.rs#L388)).
4. **Hard fail** — no AV1 silicon, no fallback
   ([mod.rs:465-468](../crates/codec/src/encode/mod.rs#L465)).

`TRANSCODE_ENCODER_BACKEND=nvenc|amf|qsv` (the README CLI note) maps to the
`preferred: Option<EncoderBackend>` argument, which routes through
[`create_backend`](../crates/codec/src/encode/mod.rs#L471) and bypasses the
chain entirely.

### Why

- **GPU-only, fail-fast.** The auto chain ends in an `Err`, not a CPU fallback
  ([mod.rs:462-468](../crates/codec/src/encode/mod.rs#L462)). rav1e and Vulkan
  encode were deleted 2026-05-08: "rav1e on Archive preset doesn't keep up with
  real-time throughput at 4K and the Vulkan-encode binding never made it past
  scaffolding" ([mod.rs:278-281](../crates/codec/src/encode/mod.rs#L278)).
  Degrading silently to a 20× slower CPU encode is worse than telling the
  operator to reprovision — so a host with no AV1-encode silicon errors at
  encoder construction with a clear message.
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
| [`backend_output_caps(backend)`](../crates/codec/src/encode/mod.rs#L221) | Per-backend caps. All three HW backends report `{max_bit_depth: 10, hdr: true}` — NVENC via `Yuv420_10bit`, AMF via `P010`, QSV via in-repo oneVPL P010. |
| [`build_output_caps()`](../crates/codec/src/encode/mod.rs#L234) | The **union over compiled paths**. 10-bit+HDR if any of `ffmpeg`/`nvidia`/`amd`/`qsv` is on; otherwise 8-bit only. |
| [`encode_backends()`](../crates/codec/src/encode/mod.rs#L249) | The compiled backends in dispatch order — `["nvenc", "amf", "qsv", "ffmpeg"]` filtered by feature flags. Drives `rivet capabilities`. |

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

### AMF (`amf.rs`)

> AMD RDNA3+ (Radeon RX 7000+). AV1 component is `AMFVideoEncoderVCN_AV1`.

Property-driven: every knob is an `AMFComponent::SetProperty(name, value)` call
with wide-string names from `vendor/amd/VideoEncoderAV1.h`. Session flow:
`AMFInit` → context (DX11 on Windows / Vulkan on Linux) → `CreateComponent` →
set properties → `Init(NV12, w, h)` → per-frame alloc-surface/copy/submit/query
→ `Drain` → teardown ([amf.rs:9-31](../crates/codec/src/encode/amf.rs#L9)).

The notable gotcha is the **`AMF_INPUT_FULL` retry policy**
([amf.rs:33-54](../crates/codec/src/encode/amf.rs#L33)): `AMF_INPUT_FULL` is a
*transient* status, not a failure. The correct sequence is: **don't** release
the surface (releasing it makes the retry a use-after-free), drain one output
packet via `QueryOutput` to free an input slot, then retry `SubmitInput` with
the *same* surface pointer. Only after the eventual `AMF_OK` does the encoder
take its own ref and we release ours.

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
  *Note:* the `QsvAv1Params.low_power` field doc in `tuning.rs` still reads
  "Always `MFX_CODINGOPTION_OFF`"
  ([tuning.rs:227-231](../crates/codec/src/encode/tuning.rs#L227)) — that comment
  is **stale**; the actual emitted value is ON.
- **ICQ is rate-control mode 9, not 8.** `MFX_RATECONTROL_ICQ = 9`; **8 is
  `MFX_RATECONTROL_LA`** (lookahead). The original code used 8 and AV1/Arc
  rejected `Query` with `MFX_ERR_UNSUPPORTED`
  ([qsv.rs:98-102](../crates/codec/src/encode/qsv.rs#L98)). ICQ (Intelligent
  Constant Quality) is the QSV equivalent of CRF and the right match for a
  perceptual target; lookahead-bitrate is not used. The numeric value in
  `tuning.rs`'s `QsvRateControl` enum is documentary — `qsv.rs` holds the
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

> Source: [`crates/codec/src/encode/tuning.rs`](../crates/codec/src/encode/tuning.rs)

### What

The user picks two backend-agnostic things; the adapter translates them into
each encoder's native parameters so identical inputs yield visually consistent
output across vendors.

- [`QualityTarget`](../crates/codec/src/encode/tuning.rs#L37) — a **perceptual
  goal** expressed in VMAF/SSIMULACRA2 bands, *not* an encoder CRF:
  `VisuallyLossless` (~VMAF 98) · `High` (~95) · `Standard` (~90, default) ·
  `Low` (~85) · `Vmaf(u8)` (explicit escape hatch).
- [`SpeedTier`](../crates/codec/src/encode/tuning.rs#L54) — how much wall-clock
  to spend: `Draft` · `Standard` (default) · `Archive`. Maps to native speed
  presets (NVENC P5/P6/P7, etc.).

The `*_av1_params(target, tier, width, height)` functions
([nvenc_av1_params](../crates/codec/src/encode/tuning.rs#L332),
[amf_av1_params](../crates/codec/src/encode/tuning.rs#L396),
[qsv_av1_params](../crates/codec/src/encode/tuning.rs#L465)) each return a
concrete params struct the matching encoder splats into its SDK structs.
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
  ([libaom_cq_for_target](../crates/codec/src/encode/tuning.rs#L532)), then a
  per-encoder calibration shift compensates for that encoder's
  compression-efficiency gap (NVENC ~3-4 CQ lower, AMF ~8 q-index lower in
  0..255 space). The `Vmaf(u8)` escape hatch interpolates between calibrated
  anchor tables ([piecewise_cq](../crates/codec/src/encode/tuning.rs#L573)).
  Source tables: `docs/av1-tuning-research.md`.
- **The QP scales genuinely differ per vendor**, and the doc comments encode the
  traps: NVENC AV1 CQ is **0..63** (not the 0..51 H.264/HEVC range); AMF q-index
  is the full AV1 **0..255**; QSV ICQ is **1..51** (an oneVPL idiosyncrasy that
  scales AV1's 0..63 into 0..51 for API parity). A value sent on the wrong scale
  is silently mis-quantized or rejected.
- **Fewer tiles = better compression on HW encoders.** Tile boundaries break
  loop-filter continuity and AV1 tiles are entropy-coded independently, so the
  shared HW tile grid ([tile_grid_hw](../crates/codec/src/encode/tuning.rs#L643))
  caps at 2×2 even at 4K — the HW encoders have enough internal parallelism that
  they don't need rav1e's aggressive 4×4 grid for throughput. A regression test
  pins every grid inside AV1 Level 5.1 tile limits
  ([tuning.rs:1109](../crates/codec/src/encode/tuning.rs#L1109)).
- **No low-latency presets.** This is a batch transcode service, so NVENC
  P1–P4, AMF `Speed`, and the streaming/CBR rate-control modes are deliberately
  never selected — `VisuallyLossless`/`Archive` uses constant-QP for reproducible
  bitstreams, everything else uses a quality-targeting VBR.

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
- **4:4:4 → 4:2:0 is a 2×2 box average.** The simplest correct filter for
  MPEG-2-sited chroma; it matches libswscale's default and trades ~0.3 dB chroma
  PSNR for ~10× fewer cycles than a separable FIR. Alpha (from `Yuva444p10le`,
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
  comparison only, no FFmpeg link-time dependency. The implementation is scalar
  f32; it's hot but single-threaded with per-frame fan-out, so the per-thread
  budget lands inside a 1080p60 window even without AVX2 (a noted follow-up).

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
  is `Unsupported`.
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
  encode silicon rather than degrading to a 20× slower software path. FFmpeg (if compiled) sits as tier 0 over all
  vendors; the native NVENC/AMF/QSV chain is the failover.
- **Layered vendor encoders, stubbed when off.** Each is hand-rolled in-tree FFI
  that builds cross-platform; a stub type keeps the dispatcher `#[cfg]`-free and
  turns "feature not compiled" into a clear error instead of a link failure.
- **Perceptual targets, not raw CRF.** `QualityTarget`/`SpeedTier` map to native
  knobs via libaom-referenced, per-vendor-calibrated tables, so the same job
  looks the same across NVENC/AMF/QSV/FFmpeg. HW tile grids cap at 2×2; no
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
