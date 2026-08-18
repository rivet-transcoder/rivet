# codec — decode & media inspection

The `codec` crate has two halves: **encode** (AV1 out) and **decode + media
inspection** (everything in). This document is the reference for the *in* half —
how a demuxed bitstream sample becomes a normalized `VideoFrame`, how the crate
detects GPUs and picks a hardware decoder, and the pure-Rust bitstream parsers
that answer "what is this stream?" without a full decode.

For where decode sits in the larger flow (demux → **decode-once pump** → per-rung
scale → multi-GPU encode → mux), read **[docs/pipeline.md](pipeline.md)** first —
this document does not re-explain the end-to-end data flow, it drills into the
decode-side modules the pump calls.

## Why this side of the crate exists the way it does

rivet is "no FFmpeg, in any capacity." That means decode cannot lean on an
FFmpeg wrapper crate for the hardware paths — so every GPU vendor's decoder is
**hand-rolled `dlopen` FFI in-tree**: NVIDIA via libcuda + libnvcuvid (CUVID),
Intel via libvpl (oneVPL), AMD via the AMF runtime. No external wrapper crate, no
`bindgen`, no build-time SDK link. The payoff is that `cargo build --features
nvidia|amd|qsv` compiles on **both Windows MSVC and Linux** with nothing but a C
toolchain — the same binary the production fleet ships. The cost is that we own
the vendor ABI: the FFI struct layouts are mirrored by hand from the vendor
headers and pinned with compile-time size assertions (see
[the qsv_ffi ABI layer](#the-qsv_ffi-abi-layer) and the NVDEC ABI witnesses), so
a driver/SDK layout drift fails the build instead of corrupting memory at
runtime.

The output codec defaults to **AV1** (royalty-clean: AV1 + Opus in MP4), with
**H.264 / H.265** also selectable for legacy-player compatibility; the *input*
side accepts an even wider codec set
(H.264, HEVC, VP8/VP9, AV1, MPEG-2, MPEG-4 Part 2) because the job is to
transcode whatever a user uploads — on a host whose GPU decodes it, or, for
H.264 and HEVC, on any host at all: rivet's own pure-Rust decoders
([`crates/h26x`](../crates/h26x/README.md)) are the software tier for those two,
always compiled and always in the chain below the hardware. GPU decode is
feature-gated per vendor; `ffmpeg` (libavcodec) is the broad optional software
tier behind the native one, `rav1d-fallback` adds a software path for AV1. Every backend
implements one trait — [`Decoder`](#the-decoder-trait) — so the pump drives them
all identically (`push_sample` → `decode_next`), and every backend emits frames
in one normalized layout (`Yuv420p` / `Yuv420p10le`) so the rest of the pipeline
never branches on which GPU produced the pixels.

---

## Module map

| File | Purpose |
|------|---------|
| [`src/lib.rs`](../crates/codec/src/lib.rs) | Crate root: module declarations + the `PixelFormat` / `VideoFrame` / `GpuDevice` re-exports. |
| [`src/frame.rs`](../crates/codec/src/frame.rs) | Core data types: `VideoFrame`, `PixelFormat`, `ColorSpace`/`TransferFn`, `StreamInfo`, and the HDR metadata structs (`ColorMetadata`, `MasteringDisplay`, `ContentLightLevel`). |
| [`src/decode/mod.rs`](../crates/codec/src/decode/mod.rs) | The `Decoder` trait + `create_decoder` GPU dispatch, the `DISABLE_*` env knobs, shared NV12/P010 deinterleave helpers, and the `capabilities` introspection. |
| [`src/decode/nvdec.rs`](../crates/codec/src/decode/nvdec.rs) | NVIDIA NVDEC/CUVID streaming decode — hand-rolled libnvcuvid FFI with ABI-pinned structs. |
| [`src/decode/qsv_dec.rs`](../crates/codec/src/decode/qsv_dec.rs) | Intel QSV/oneVPL decode — hand-rolled libvpl FFI, internal-allocation + `FrameInterface::Map`. |
| [`src/decode/amf_dec.rs`](../crates/codec/src/decode/amf_dec.rs) | AMD AMF decode — hand-rolled AMF COM-style vtable FFI. Init + Windows adapter routing exercised; per-frame decode verified-by-review (no AMF-capable card on hand). |
| [`src/amf_device.rs`](../crates/codec/src/amf_device.rs) | Windows-only: hand-rolled DXGI/D3D11 dlopen FFI that makes a D3D11 device on a *specific* AMD adapter, so AMF's `InitDX11` binds to the right GPU on a mixed host (gated `windows` + `amd`). |
| [`src/decode/h26x_sw.rs`](../crates/codec/src/decode/h26x_sw.rs) | **Native H.264 / HEVC decode** on this workspace's own [`h26x`](../crates/h26x/README.md) crate — pure Rust, always compiled, frame + wavefront threaded, AVX2/NEON kernels, bit-exact against the JVT / JCT-VC conformance suites. The software tier for the two codecs it serves; refuses (`Unsupported`) up front what it does not do, so the guard rebuilds the next tier. |
| [`src/decode/ffmpeg.rs`](../crates/codec/src/decode/ffmpeg.rs) | libavcodec software decode (optional `ffmpeg` feature; the one backend needing host libraries at build time). Behind the native tier: catches the H.264 the native tier refuses (slice groups, SP/SI, data partitioning), the odd profile, and the other codecs. |
| [`src/decode/openh264_sw.rs`](../crates/codec/src/decode/openh264_sw.rs) | Software H.264 via openh264 (optional `openh264-fallback`), the narrow last resort below libavcodec. |
| [`src/decode/rav1d_sw.rs`](../crates/codec/src/decode/rav1d_sw.rs) | Software AV1 decode via [rav1d](https://crates.io/crates/rav1d) (optional `rav1d-fallback` feature) — hand-rolled `extern "C"` over the dav1d ABI, no system library. |
| [`src/gpu.rs`](../crates/codec/src/gpu.rs) | GPU detection (`detect_gpus`), `GpuDevice`/`GpuVendor`, NVML + sysfs (Linux) / WMI (Windows) enrichment, global vs vendor-local indices, live-utilisation reader, `supports_av1_encode`. |
| [`src/cuda_lock.rs`](../crates/codec/src/cuda_lock.rs) | Process-wide CUDA-init mutex shared by NVENC + NVDEC (`nvidia` feature only). |
| [`src/probe.rs`](../crates/codec/src/probe.rs) | Media probing without a full decode (MP4 header walk + container sniff + HDR box extraction). |
| [`src/pixel_format.rs`](../crates/codec/src/pixel_format.rs) | Pure-Rust bitstream parsers: SPS/PPS/sequence-header walkers for H.264 / HEVC / AV1 / MPEG-2 (pixel format + dimensions). |
| [`src/hevc_sei.rs`](../crates/codec/src/hevc_sei.rs) | HEVC SEI 137/144 scanner → HDR10 mastering-display + content-light-level metadata. |
| [`src/codec_strings.rs`](../crates/codec/src/codec_strings.rs) | HLS/DASH `CODECS="…"` string formatters (AV1 + AAC-LC) parsed from the bitstream. |
| [`src/qsv_ffi.rs`](../crates/codec/src/qsv_ffi.rs) | Shared oneVPL `mfx*` struct mirrors used by both the QSV encoder and decoder, pinned with `offsetof`-verified size asserts. |

---

## Frame types

**What.** [`frame.rs`](../crates/codec/src/frame.rs) defines the values every
decoder produces and every consumer reads.

- [`VideoFrame`](../crates/codec/src/frame.rs#L137) — the unit of decoded output:
  `data: Bytes` (the planar pixels), `width`/`height`, a `PixelFormat`, a
  `ColorSpace`, and a `pts`. The pixel buffer is `bytes::Bytes`, which is
  `Arc`-backed — that is load-bearing for the decode-once pump: fanning one
  decoded frame out to N rungs is a `clone()` that bumps a refcount, not a pixel
  copy (see [pipeline.md §2](pipeline.md#2-decode-once--the-shared-pump)).
- [`PixelFormat`](../crates/codec/src/frame.rs#L4) — the planar layouts the crate
  can carry (4:2:0/4:2:2/4:4:4 at 8/10/12-bit, NV12/NV21, RGB). In practice the
  GPU decoders normalize to just `Yuv420p` and `Yuv420p10le`; the wider set
  exists because the probe/parsers can *report* formats the encoder won't
  ultimately produce (AV1 out is always 4:2:0). [`bytes_per_frame`](../crates/codec/src/frame.rs#L30)
  gives the packed size, and [`from_chroma_and_depth`](../crates/codec/src/frame.rs#L67)
  maps a `(chroma_idc, bit_depth)` pair (what the bitstream parsers extract) to
  the enum, with a defensive `Yuv420p` default.
- [`ColorSpace`](../crates/codec/src/frame.rs#L81) (BT.601/709/2020) and
  [`TransferFn`](../crates/codec/src/frame.rs#L103) (BT.709 gamma, PQ/ST2084, HLG,
  …). These are **deliberately separate**: every decoder already emits
  `VideoFrame { color_space, .. }` and the converters/encoder dispatch on it, so
  keeping the transfer function on a side channel let HDR support land without
  touching every call site. [`TransferFn::from_h273`](../crates/codec/src/frame.rs#L124)
  maps the raw ITU-T H.273 `transfer_characteristics` byte down to the subset the
  pipeline knows.
- [`StreamInfo`](../crates/codec/src/frame.rs#L166) — the demuxer's header
  (codec string, dims, frame rate, duration, source pixel format, color
  metadata). It's the input to `create_decoder` and the output of `probe`.
- [`ColorMetadata`](../crates/codec/src/frame.rs#L191) bundles all HDR-relevant
  signaling: transfer function, raw H.273 `matrix_coefficients` / `colour_primaries`
  / `full_range_flag`, plus optional [`MasteringDisplay`](../crates/codec/src/frame.rs#L259)
  (SMPTE ST 2086 / HEVC SEI 137) and [`ContentLightLevel`](../crates/codec/src/frame.rs#L286)
  (CTA-861.3 / HEVC SEI 144).

**Why.** The metadata sub-struct exists so the source's HDR signaling survives
the trip from the bitstream all the way to the MP4 mux's `colr`/`mdcv`/`clli`
atoms. **Crucially it `Default`s to an SDR BT.709 baseline** (matrix=1,
primaries=1, transfer=Bt709, studio range) — so every existing `StreamInfo { … }`
literal compiles unchanged via `..Default::default()`, and only HDR-aware
producers (the NVDEC sequence callback, the HEVC SEI scanner, the MP4 probe)
populate non-default values. The struct field names are a documented load-bearing
contract: the probe and SEI parsers write them directly and the mux reads them
verbatim, so renaming them silently breaks HDR round-trip.

**Notes / gotchas.**
- The unit conventions on `MasteringDisplay` are exact wire-domain integers
  (chromaticities in 0.00002 steps, luminance in 0.0001 cd/m²) — they are *not*
  scaled, because they pass straight into the MP4 box bytes. The doc comment on
  the struct is the authority.
- `lib.rs` re-exports `ColorSpace`, `PixelFormat`, `VideoFrame`, `GpuDevice`,
  `GpuVendor` at the crate root for convenience; everything else is reached via
  its module.

---

## The decode dispatch & tiers

**What.** [`decode/mod.rs`](../crates/codec/src/decode/mod.rs) defines the
[`Decoder`](#the-decoder-trait) trait and the
[`create_decoder`](../crates/codec/src/decode/mod.rs#L254) /
[`create_decoder_on`](../crates/codec/src/decode/mod.rs#L268) factory that picks a
hardware decoder for a `(codec, StreamInfo)` and an optional GPU index.

### The `Decoder` trait

[`Decoder`](../crates/codec/src/decode/mod.rs#L108) is three methods plus
`stream_info()`:

```rust
fn push_sample(&mut self, data: &[u8]) -> Result<()>;  // feed one Annex-B / OBU sample
fn finish(&mut self) -> Result<()>;                     // end-of-stream
fn decode_next(&mut self) -> Result<Option<VideoFrame>>;// pull a decoded frame
```

This streaming push/pull shape is what keeps peak RSS bounded — the pump pushes
one demuxed sample, drains whatever frames are ready, and never materialises the
whole stream. Implementations may buffer internally (QSV accumulates until it has
a full header) or decode eagerly; the contract only says frames come out of
`decode_next` in display order.

### Dispatch order and the actual tiers

[`create_decoder_on`](../crates/codec/src/decode/mod.rs#L268) calls
[`gpu::detect_gpus()`](#gpu-detection) once, then tries, in order:

1. **NVDEC** (`nvidia` feature) — if an NVIDIA device is present (or matches the
   requested `gpu_index`), the codec is in
   [`nvdec_supports`](../crates/codec/src/decode/mod.rs#L163), and it isn't
   disabled by env-var, return an `NvdecDecoder`. NVIDIA wins ties because NVDEC
   is generally lower-latency on the standard codec set and is what the fleet is
   tuned against (comment at `create_decoder`).
2. **AMF** (`amd` feature) — first AMD device + [`amf_dec::supports`](../crates/codec/src/decode/amf_dec.rs#L228).
3. **QSV** (`qsv` feature) — first Intel device + [`qsv_dec::supports`](../crates/codec/src/decode/qsv_dec.rs#L94).
4. **Native H.264 / HEVC** ([`h26x_sw`](../crates/codec/src/decode/h26x_sw.rs),
   always compiled) — rivet's own decoders, first among the software tiers for
   those two codecs. Wrapped in the same guard as the hardware tiers: a stream
   they refuse (an `Unsupported` on the parameter set — H.264 FMO, SP/SI,
   data partitioning; HEVC beyond what the RExt profiles libavcodec also has) is
   replayed into the next tier with nothing lost. `RIVET_DISABLE_H26X=1` skips
   it. See [Native H.264 / HEVC](#native-h264--hevc--decodeh26x_swrs).
5. **libavcodec** (`ffmpeg` feature) — the broad software tier, behind the
   native one; removed 2026-08-12, restored 2026-08-14 once `create_decoder`
   actually constructed it.
6. **openh264** (`openh264-fallback`), H.264 only — the narrow last resort.
7. **Software AV1** (`rav1d-fallback` feature, AV1 only) — off by default, and
   below every vendor path so it is a floor rather than a preference.
8. Otherwise **hard-fail** with a message naming what each tier covers.

   The module header still records the 2026-05-08 directive that deleted every
   CPU decoder (openh264, libde265, libvpx, rav1d, …) along with the legacy
   `FallbackDecoder` GPU→CPU fallover. What came back came back deliberately:
   rav1d for AV1 (the format rivet itself produces), libavcodec as a gated broad
   tier, and — the reason the software story is now different in kind — the
   native `h26x` decoders, which need nothing from the host, are threaded across
   the machine, and are checked bit-exact against the conformance suites.

**Why hardware first, and loud about software.** The README's whole pitch is
that getting GPU decode right per vendor is the hard part a generic toolbox
leaves to you, and it "quietly falls back to a slow software path when any of
that is wrong." rivet keeps the hardware tiers first and every software
engagement says so at `info`/`warn` — but for H.264 and HEVC a host with no
decode silicon now decodes them natively rather than failing the job.

### The `DISABLE_*` env knobs

[`nvdec_disabled_for`](../crates/codec/src/decode/mod.rs#L144) is a debugging
escape hatch: `DISABLE_NVDEC=1` skips NVDEC for every codec, and
`DISABLE_NVDEC_<CODEC>=1` (e.g. `DISABLE_NVDEC_H264`, `DISABLE_NVDEC_AV1`) skips
one codec family. The point is operational — when a specific codec/driver combo
misbehaves on a host (the comment cites a "Blackwell + 4K H.264 silent-stall"),
you can disable just that path and fall through to QSV without a rebuild.
[`env_flag_truthy`](../crates/codec/src/decode/mod.rs#L128) parses `1`/`true`/`yes`/`on`/`y`/`t`.

### Shared deinterleave helpers

Two `pub(crate)` helpers convert vendor surface layouts into the crate's packed
planar convention, shared so the three GPU backends can't drift:
- [`nv12_planes_to_yuv420p`](../crates/codec/src/decode/mod.rs#L32) — NV12
  (Y plane + interleaved UV, each with its own stride) → packed `[Y | U | V]`.
- [`p010_planes_to_yuv420p10le`](../crates/codec/src/decode/mod.rs#L68) — host
  P010 (10-bit in the **high** bits of each u16) → `Yuv420p10le` (10-bit in the
  **low** bits via `>> 6`).

### Capability introspection

[`decode_backends()`](../crates/codec/src/decode/mod.rs#L188) and
[`decode_capabilities()`](../crates/codec/src/decode/mod.rs#L216) report which
compiled backends can decode each codec — these back the `rivet capabilities`
CLI command, not the runtime dispatch. The QSV row is not a static guess: it
comes from a **runtime hardware probe**
([`qsv_dec::probe_decode_caps`](../crates/codec/src/decode/qsv_dec.rs)) that opens
a oneVPL HW session and `MFXVideoDECODE_Query`s each codec, so the report shows
QSV decode only on a host where the Intel runtime + adapter actually initialise.

**Notes / gotchas (drift to be aware of).**
- **The `gpu_index` argument is load-bearing for multi-GPU.** `create_decoder`
  (no index) keeps the legacy "first matching adapter" behaviour for one-shot
  callers; the pipeline's per-rung pumps should pass `Some(idx)` so each rung's
  decode session lands on a distinct physical adapter (the doc comment on
  `create_decoder_on` flags that, without it, every QSV session piles onto the
  first Intel card).
- **`FallbackDecoder` does not exist**, but late fallback does:
  `HardwareThenSoftware` (in `decode/mod.rs`) wraps a hardware tier — and the
  native `h26x` tier — so a decoder that accepts construction and refuses the
  first real sample is replaced by the next tier, with everything fed so far
  replayed. Once a decoder has produced a frame the guard is dropped: a failure
  on sample nine thousand is a stream error, not a capability question.
  `create_decoder_on` wires NVDEC → AMF → QSV → h26x → libavcodec → openh264 →
  rav1d → hard-fail.
  *(Historical note: an FFmpeg tier used to be listed here as "present but not
  wired" — capability-listed and never constructed — which is why it was removed
  outright on 2026-08-12 and restored, constructed, on 2026-08-14.)*
- The module header says "exactly two backends (NVDEC + QSV)", but `amf_dec` is a
  real third tier behind the `amd` feature — the prose predates the AMF decode
  landing.

---

## GPU decode backends

All three hardware backends share the same shape: `dlopen` the vendor runtime,
stand up a context/session, drive it through the `Decoder` trait, and copy decoded
surfaces back to host memory in `Yuv420p` / `Yuv420p10le`. They differ in the
vendor API and how much the dev box could verify.

### NVDEC (NVIDIA) — `decode/nvdec.rs`

**What.** [`nvdec.rs`](../crates/codec/src/decode/nvdec.rs) loads `libcuda` +
`libnvcuvid` at runtime and drives the CUVID stateless-parser API:
`cuInit` → `cuCtxCreate` → `cuvidCreateVideoParser`, then per sample
`cuCtxPushCurrent` + `cuvidParseVideoData` + `cuCtxPopCurrent`. Three parser
callbacks do the work: the sequence callback creates the decoder + validates the
format, the decode callback runs `cuvidDecodePicture`, and the display callback
maps the frame (`cuvidMapVideoFrame` + `cuMemcpy2D`) and pushes NV12/P016 bytes
into a shared collector. The production path is the **streaming**
`NvdecStreamingDecoder` ([engaged from `NvdecDecoder::new`](../crates/codec/src/decode/nvdec.rs#L1006)):
each `push_sample` parses just-pushed bytes, and the display callback enqueues
into a bounded `VecDeque<DecodedFrame>` that `decode_next` drains one at a time.

**Why streaming, not eager.** The eager `new_with_pts` constructor (retained as
library/test code) buffers the entire decoded run in RAM before draining — fine
for smoke tests, catastrophic for a 15-minute clip. The streaming decoder bounds
peak heap to roughly one bitstream sample plus a reorder-window-sized queue
(≤ B-pyramid depth ≈ 16 frames); CUVID's DPB lives GPU-side, not in RSS.

**Why the obsessive ABI pinning.** This file is where the project's "we own the
vendor ABI" tax is paid in full. The FFI structs mirror NVIDIA Video Codec SDK
12.2, and a *too-small* Rust struct lets the driver write past our allocation into
adjacent state — surfacing as a `STATUS_ACCESS_VIOLATION` on long streams or,
worse, silent wrong frames. So:
- Compile-time size assertions pin the exact layout: `CUVIDPARSERPARAMS` at
  [136 bytes](../crates/codec/src/decode/nvdec.rs#L556), `CUVIDPICPARAMS` at
  [4280](../crates/codec/src/decode/nvdec.rs#L584) (its `codec_specific` region is
  the SDK's 4096-byte `CodecReserved[1024]` envelope), `CUVIDPARSERDISPINFO`,
  `CUVIDDECODECAPS`, and `CUVIDSOURCEDATAPACKET` *per platform* (the `c_ulong`
  width differs Windows vs Linux — [24 vs 32 bytes](../crates/codec/src/decode/nvdec.rs#L601)).
  The comment block lists the real bugs this caught: the parser-params 80→136 fix
  and the pic-params 2048→4280 fix, both of which had produced the segfault-hunt
  class of corruption.
- **Per-codec "shape witness" structs** ([H264/HEVC/AV1/VP9/VP8/MPEG2/MPEG4](../crates/codec/src/decode/nvdec.rs#L338-L488))
  are dead-code mirrors that exist *only* so a `const_assert!` proves each codec's
  pic-params variant fits in the 4096-byte envelope. They're never used at
  runtime; they're a tripwire so a future SDK that grows one variant fails
  compilation instead of silently overflowing the parser state and reproducing the
  original segfault on a different code path.

**Typed rejects.** [`validate_format`](../crates/codec/src/decode/nvdec.rs#L729)
is a pure function (unit-testable without a GPU) that turns the CUVID-reported
chroma/bit-depth into a typed [`NvdecError`](../crates/codec/src/decode/nvdec.rs#L63):
`UnsupportedChroma` (only 4:2:0 passes; monochrome/4:2:2/4:4:4 reject),
`UnsupportedPixelFormat` (>12-bit), `UnsupportedByHardware` (per-GPU caps). Why
typed: a reviewer note records that these used to surface as an opaque "NVDEC
produced no frames: <string>", and the pipeline couldn't tell "4:2:2 unsupported"
(a format we'll never decode) from "driver OOM" (transient) — the typed variant
keeps a fallback/abort decision explainable via `downcast_ref::<NvdecError>()`.

**Notes / gotchas.**
- 10-bit comes back as **P016** (10 bits in the high bits of each u16);
  [`deinterleave_p016_to_yuv420p10le`](../crates/codec/src/decode/nvdec.rs#L773)
  does the `>> 6` normalize + UV split and handles odd dimensions. 12-bit shares
  the path (the shift clips to 10-bit range, which is what downstream expects).
- `CUVID_CREATE_PREFER_CUVID` forces the CUVID software-parser backend over
  DXVA on Windows — the SDK default DXVA path produced different surface layouts
  and was the suspected root cause of an H.264 segfault.
- The library handles are stored **last** on the struct so Rust's source-order
  drop tears down decoder/parser/context before unloading the `.so`/`.dll` whose
  fn pointers they reference.

### QSV (Intel) — `decode/qsv_dec.rs`

**What.** [`qsv_dec.rs`](../crates/codec/src/decode/qsv_dec.rs) `dlopen`s
`libvpl` and drives oneVPL: `MFXInit(HW)` →
[`MFXVideoDECODE_DecodeHeader`](../crates/codec/src/decode/qsv_dec.rs#L152) on the
first buffered samples → `MFXVideoDECODE_Init` → per sample
[`DecodeFrameAsync` + `SyncOperation`](../crates/codec/src/decode/qsv_dec.rs#L220),
then [read the surface](../crates/codec/src/decode/qsv_dec.rs#L280) into a
`VideoFrame`. Decodes H.264/HEVC/AV1/VP9 (8-bit NV12 and 10-bit P010). It reuses
the exact `mfx*` struct layouts from [`qsv_ffi`](#the-qsv_ffi-abi-layer), shared
with the QSV encoder.

**Why the internal-allocation path.** `DecodeFrameAsync` runs with
`surface_work = NULL`, which engages oneVPL 2.x **internal surface allocation**;
the decoder then reads the returned surface through its `mfxFrameSurfaceInterface`
vtable (`Map` to access planes, `Release` when done). The comments record that the
*external* work-surface pool "never produced frames on the iHD 2.x runtime" — this
is the path that actually works (and the one shiguredo_vpl uses).

**Why trust DecodeHeader's format.** `try_init` deliberately does **not** force
fourcc/bit-depth/shift — it lets the iHD driver report NV12 for 8-bit and P010 for
10-bit (Main10) and derives `ten_bit` from the returned fourcc. The comment notes
that forcing those fields ourselves made HEVC Main10 `Init` fail.

**Notes / gotchas.**
- [`read_surface`](../crates/codec/src/decode/qsv_dec.rs#L280) emits the **crop
  (display)** dims, not the coded dims — 1080p codes as 1088 (16-aligned), and
  feeding 1088-tall frames into a 1080-configured encoder fails
  `EncodeFrameAsync`.
- Plane pointers are valid only between `Map`/`Unmap`; the read copies out inside
  that window.
- **Hardware-verified on a 3× Intel Arc box** (A310 / A380 / A750, oneVPL 2.16 /
  iHD): H.264, HEVC, VP9, and AV1 each decode end-to-end (transcode to AV1 on a
  qsv-only build, with the QSV decoder engaged).

**Capability probe.** [`probe_decode_caps`](../crates/codec/src/decode/qsv_dec.rs)
is the decode side of `rivet capabilities`: it opens one HW `MFXInit` session and
`MFXVideoDECODE_Query`s each codec. iHD's Query is *advisory* — on the Arc box it
returns an error for every codec it nonetheless decodes — so a successful
`MFXInit(HW)` is the load-bearing signal: when Query yields nothing, the probe
reports the build's codec list (the runtime is usable) rather than claim no
decode; a non-empty Query result is trusted as-is (to drop, say, AV1 on a pre-Arc
iGPU). It returns empty on a non-Intel host, so the report shows QSV decode only
where it actually runs. This is wired into
[`decode_capabilities`](../crates/codec/src/decode/mod.rs#L216).

### AMF (AMD) — `decode/amf_dec.rs`

**What.** [`amf_dec.rs`](../crates/codec/src/decode/amf_dec.rs) `dlopen`s the AMF
runtime and drives its COM-style vtable API: `AMFInit` (factory) → `CreateContext`
+ `InitDX11`/`InitVulkan` → `CreateComponent(<decoder id>)`, then per sample wrap
the bytes in an `AMFBuffer`, `SubmitInput`, loop `QueryOutput`, downcast the
`AMFData` to an `AMFSurface`, and read NV12/P010 planes. Decodes
H.264/HEVC/AV1/VP9. The whole AMF object model is reproduced as `#[repr(C)]` vtable
structs ([factory/context/component/surface/plane/buffer](../crates/codec/src/decode/amf_dec.rs#L61-L208)),
shared in spirit with the AMF encoder.

**Multi-adapter routing (Windows).** AMF's `InitDX11(null)` lets the runtime
create its device on **DXGI adapter 0** — which on a mixed host (an NVIDIA card in
slot 0 + an AMD GPU elsewhere) is the wrong, non-AMD adapter, and init fails. So
on Windows we build the D3D11 device ourselves on the chosen AMD adapter:
[`amf_device::create_amd_d3d11_device`](../crates/codec/src/amf_device.rs)
enumerates adapters via DXGI, finds the `vendor_index`-th `0x1002` one, and
`D3D11CreateDevice`s it with `D3D11_CREATE_DEVICE_VIDEO_SUPPORT` — then hands that
device to `InitDX11` (the decoder keeps it alive for the context's lifetime). On
Linux the `InitVulkan(null)` path is kept (AMF picks the first AMD GPU). This is
the `gpu_index` plumbing actually doing something — see [GPU detection](#gpu-detection)
for how a global index maps to a vendor-local adapter.

**Why it's the most caveated backend.** The only AMD silicon on the dev box is a
Ryzen desktop iGPU whose VCN the AMF runtime does **not** support — `InitDX11`
returns `AMF_NOT_FOUND` for it (the AMF *encoder* probe fails on it too). So
**detection, the D3D11 adapter routing, and the init/teardown path are exercised,
but the per-frame `SubmitInput` → `QueryOutput` → surface-readback loop has not run
end-to-end** — that needs a discrete Radeon (RDNA) or a supported APU. The most
fragile decode-loop guess is the
[`AMF_IID_SURFACE` GUID](../crates/codec/src/decode/amf_dec.rs#L66) used to
`QueryInterface` the output `AMFData` into an `AMFSurface` — a wrong IID fails every
output. That, the host-memory read-back, and the `Convert` slot are flagged
`// VERIFY:`; for AV1, `rav1d-fallback` is the documented AMD fallback — other
codecs have no software path.

**Notes / gotchas.**
- `gpu_index` now selects the AMD adapter on Windows (via the D3D11 routing
  above), **not** "adapter 0 unconditionally" as older comments claimed.
- A **failed `InitDX11` no longer segfaults.** Tearing down an AMF context whose
  `InitDX11(external device)` failed corrupts the runtime's state (even
  `Release`/`Terminate` crash), so on that cold error path the half-initialised
  context is leaked and a clear error returned. This is what lets `--decode-gpu
  fastest` benchmark an AMD GPU, catch an init failure, and **skip it** instead of
  taking the process down; an AMF-incapable GPU surfaces as
  `AMFContext::InitDX11 … AMF_NOT_FOUND=11`.

### Native H.264 / HEVC — `decode/h26x_sw.rs`

**What.** [`h26x_sw.rs`](../crates/codec/src/decode/h26x_sw.rs) drives
[`crates/h26x`](../crates/h26x/README.md): two decoders written from the ITU-T
specifications (H.264 frames, field pictures and MBAFF, 4:0:0 / 4:2:0 /
4:2:2 / 4:4:4 at 8–14-bit — Baseline through High 4:4:4 Predictive, separate
colour planes and lossless included — with every entropy coder and profile tool that
implies:
CAVLC/CABAC, B-frames, temporal/spatial direct, weighted prediction, 8x8
transform, scaling matrices, MMCO/long-term, PCM; HEVC
Main / Main 10 / Main 12 and the format range extensions — 4:0:0 / 4:2:0 /
4:2:2 / 4:4:4 up to 12-bit with the RExt coding tools libavcodec also has —
with WPP, tiles, dependent slices, SAO, PCM, transform skip, scaling lists,
TMVP, weighted prediction). Both are bit-exact against the conformance suites
(JVT AVCv1 + FRExt: 199/199 supported streams, professional profiles: 35/35
accepted; JCT-VC HEVC_v1: 146/147, RExt: 31/31 accepted) and both are threaded — pictures in flight
concurrently with reference-progress waits, plus wavefront rows / tiles inside a
picture for HEVC — with AVX2 (x86-64) and NEON (AArch64) kernels selected at run
time. `H26X_THREADS`, `H26X_NO_SIMD`, `H26X_INFLIGHT` tune it; the crate README
lists the rest.

**Where.** First among the software tiers for the two codecs, below every
hardware tier. It is always compiled — pure Rust, no toolchain — so unlike
`ffmpeg` it is present in every build, and unlike `openh264-fallback` it handles
the profiles that actually arrive. libavcodec, when built, sits behind it and
takes what it refuses.

**Output.** 8-bit as `Yuv420p`, 10/12-bit as `Yuv420p10le` / `Yuv420p12le`
(9-bit widened to 10), 4:2:2 / 4:4:4 as `Yuv422p` / `Yuv444p` and their
10-bit forms (12-bit 4:2:2 / 4:4:4 and the 11 / 13 / 14-bit depths H.264's
professional profiles allow have no pipeline pixel format and are reported as
such); monochrome travels as 4:2:0 with grey chroma. Frames come out in output (POC)
order, numbered from zero, like the AV1 tier.

**Refusals.** An `h26x::Error::Unsupported` is returned from `push_sample`
before any picture exists — on the parameter set for H.264, at the first slice
where the SPS is checked — and the tier guard hands the stream on. Bitstream
errors after the first picture are logged and skipped, matching the other
software tiers (a stream that starts mid-GOP is not a failed job).

### Software AV1 — `decode/rav1d_sw.rs`

**What.** [`rav1d_sw.rs`](../crates/codec/src/decode/rav1d_sw.rs) (gated on
`rav1d-fallback`) drives [rav1d](https://crates.io/crates/rav1d) — a Rust port
of dav1d — behind the `Decoder` trait. It is the only CPU decoder in the tree,
and it decodes **AV1 8-bit 4:2:0 only**; anything else errors at `convert`
rather than guessing.

rav1d exposes the dav1d **C ABI**, so this module declares that ABI in a local
`unsafe extern "C"` block (`dav1d_open` / `dav1d_send_data` /
`dav1d_get_picture` / `dav1d_picture_unref` / `dav1d_close`) rather than pulling
a `-sys` crate. Same reasoning as the GPU FFI: no bindgen, no build-time link
against anything the host has to supply.

**Two traps it is written around**, both of which look like decoder bugs:

- **Plane copies are row-wise.** dav1d hands back planes with their own stride,
  and a flat `copy_from_slice` shears the picture progressively down the frame.
- **`EAGAIN` at end-of-stream is not "wait for more input".** During the drain
  it means *done*. Treating it as a retry hangs; treating a mid-stream `EAGAIN`
  as done truncates. `drain_inner(at_eos)` distinguishes the two, and `finish()`
  deliberately does **not** call `dav1d_flush`, which would discard the very
  pictures the drain is trying to collect.

**Why it exists.** A CPU-only host could not decode anything at all — the
factory ran NVDEC → AMF → QSV → hard-fail. AV1 is the format rivet itself
produces, so being able to read one back without a GPU is what makes a
round-trip test runnable in CI, and it costs no system dependency to have.

**`rav1d-asm`** turns on rav1d's hand-written assembly. It is off by default
because it needs **NASM** on the build host; the pure-Rust path is slower and
builds anywhere.

---

## GPU detection

**What.** [`gpu.rs`](../crates/codec/src/gpu.rs) enumerates the host's GPUs and
exposes live utilisation. [`detect_gpus()`](../crates/codec/src/gpu.rs#L65)
concatenates per-vendor scans into `Vec<`[`GpuDevice`](../crates/codec/src/gpu.rs#L13)`>`
(vendor, name, index, generation, PCI id, VRAM, serial, bus address).

**Two indices, because a host can be mixed.** Each per-vendor scan numbers its own
devices from 0, so on an NVIDIA + AMD box both would be index 0. `GpuDevice`
therefore carries **two**: `vendor_index` (the device's position *within its
vendor*, what the per-vendor SDK enumerates — the CUDA ordinal, the QSV/AMF
adapter) and a globally-unique `index` that `detect_gpus` reassigns across the
concatenated list. The global `index` is what the user addresses (`--decode-gpu N`,
the GPU policy) and what `create_decoder_on` matches; it then constructs the
chosen backend with that device's **`vendor_index`** so the hardware selects the
right physical adapter. On a single-vendor host the two coincide.

- **NVIDIA** via [libcuda dlopen](../crates/codec/src/gpu.rs#L99) (`cuInit` +
  `cuDeviceGetCount` + `cuDeviceGetName`), enriched by NVML for VRAM/PCI/serial.
  **Why dlopen, not `nvidia-smi`:** minimal container images often lack the
  `nvidia-smi` binary, but the NVIDIA Container Toolkit bind-mounts the driver's
  user-mode libraries — so probing the library directly works where shelling out
  wouldn't. (NVML init even retries the SONAME-versioned `libnvidia-ml.so.1`
  because the toolkit mounts only that, not the unsuffixed alias.)
- **AMD / Intel** via, on **Linux**, a [sysfs PCI scan](../crates/codec/src/gpu.rs#L370)
  (`/sys/bus/pci/devices`, matching vendor `0x1002` / `0x8086` + a display class);
  on **Windows**, a WMI query (`Get-CimInstance Win32_VideoController`, cached for
  the process) parsing the `PNPDeviceID` `VEN_`/`DEV_` fields. Both feed the same
  device-id → generation/label tables. This is what makes an AMD GPU visible to
  the AMF decode path on a Windows host (without it the sysfs-only scan returned
  empty there, so only the NVML-detected NVIDIA card showed up).

The generation/label tables ([`nvidia_generation_from_name`](../crates/codec/src/gpu.rs#L300),
[`intel_label_from_device_id`](../crates/codec/src/gpu.rs#L680), …) are partly
cosmetic (the admin inventory page) but partly functional: the Intel labeller was
added because, without it, every Intel device was tagged "Integrated GPU" and the
AV1 dispatch's `contains("arc")` substring check missed discrete Arc cards.

[`GpuUtilizationReader`](../crates/codec/src/gpu.rs#L752) holds an NVML handle
across reads and returns a per-device [`GpuUtilization`](../crates/codec/src/gpu.rs#L731)
snapshot (compute/encoder/decoder busy %, VRAM, temperature) on each load tick.
NVIDIA reads come from NVML; Intel is a coarse sysfs freq-ratio + DRM-fdinfo VRAM
proxy; AMD is a no-op stand-in (radeontop/amdsmi deferred).

**Why `supports_av1_encode` admits everything.**
[`supports_av1_encode`](../crates/codec/src/gpu.rs#L1052) returns `true` for every
vendor on purpose. It used to carry a brittle board-name substring list, and a
missed SKU (the RTX 5060, once) would *hard-fail* a job since there's no CPU
fallback. The decision was to defer to the **real driver capability query** in the
encoder constructor (`nvEncGetEncodeCaps` / AMF `CreateComponent` / oneVPL
`MFXVideoENCODE_Query`), which authoritatively bails if AV1 silicon is absent — so
detection admits the GPU and lets the real query be the gate.

**Notes / gotchas.** `manufacturer_label` must stay in lockstep with the
worker's `capabilities.rs` `vendor_label` so registration and the hello frame
agree on spelling. Fields that a platform can't read (consumer-GeForce serials,
older-kernel VRAM) come back empty/`None`/`0` rather than synthesised, and the
literal `"0"` serial is treated as `None` per NVML's documented sentinel.

---

## The CUDA process-wide lock

**What.** [`cuda_lock.rs`](../crates/codec/src/cuda_lock.rs) is a single
process-wide `Mutex<()>` ([`CUDA_INIT_LOCK`](../crates/codec/src/cuda_lock.rs#L39))
with a poison-tolerant accessor
([`lock_for_cuda_init`](../crates/codec/src/cuda_lock.rs#L43)), compiled only under
the `nvidia` feature.

**Why.** This is a scar from a production segfault. When several NVENC encoder
constructions ran in parallel, the NVIDIA driver segfaulted inside
`NvEncOpenEncodeSessionEx`. Serialising NVENC alone wasn't enough — the *first*
encoder still crashed, because an `NvdecStreamingDecoder` was being constructed on
a sibling thread doing its **own** `cuInit` + `cuCtxCreate` + parser-create at the
same time. The driver's session table can't handle simultaneous CUDA context
creation from different code paths on the same GPU, even when each path is
internally single-threaded. The mutex serialises just the brief
CUDA-init / first-FFI-call window across **both** NVENC and NVDEC; once each
backend has its context + handle it releases the lock and per-frame work runs
concurrently as before. Cost is ~50–200 ms cold-start per run; frame throughput is
unchanged. Poisoning is treated as recoverable because the lock protects no
in-memory invariant — only "no two CUDA inits at once."

---

## Bitstream parsers

**What.** [`pixel_format.rs`](../crates/codec/src/pixel_format.rs) is a pile of
pure-Rust, no-allocation-heavy bitstream walkers built on a tiny
[`BitReader`](../crates/codec/src/pixel_format.rs#L38) (Exp-Golomb `ue`/`se`, AV1
`su`/`uvlc`, byte-align). Two layers:

1. **Pixel-format detection** — [`detect`](../crates/codec/src/pixel_format.rs#L21)
   dispatches by codec to `detect_h264` / `detect_hevc` / `detect_vp9` /
   `detect_av1`, which parse *just enough* of the first sequence header to pull
   `chroma_format_idc` + luma bit depth and map them via
   `PixelFormat::from_chroma_and_depth`. Any parse failure falls back to `Yuv420p`
   (matching the previous hard-coded behaviour — a bad probe degrades payload
   accuracy, it doesn't block the transcode).
2. **Dimension + deep parse** — [`detect_dims`](../crates/codec/src/pixel_format.rs#L2365)
   dispatches to full SPS/sequence-header walkers
   ([`parse_h264_sps`](../crates/codec/src/pixel_format.rs#L2393),
   [`parse_hevc_sps`](../crates/codec/src/pixel_format.rs#L2663),
   [`parse_mpeg2_sequence_header`](../crates/codec/src/pixel_format.rs#L3288)) that
   go all the way through scaling lists, `pic_order_cnt_type` branches, and frame
   cropping / conformance windows to compute the **displayable** width/height. It
   returns `None` to mean "keep existing dims."

**Why.** Two distinct needs. First, the pipeline wants a fast, codec-agnostic
*format* probe **before** decoder construction — none of the underlying decoders
expose a "just probe the format" API (NVDEC tells us, but only after decode
starts), so this module fills that gap. Second, **MPEG-TS carries no
container-level dimensions** (no sample-entry atom, no track header — the SPS is
the only source), so `container::ts` calls `detect_dims` during demux to populate
`StreamInfo.width`/`.height`, which would otherwise be `0×0`. That's why
`detect_dims` returning `None` is load-bearing for TS but a harmless no-op for
MP4/MKV (which already have dims).

**The deeper parsers** (`H264SpsInfo` / `HevcSpsInfo` / `H265VpsInfo` /
`H265PpsInfo` / `H265SliceHeader` / [`Av1SequenceHeader`](../crates/codec/src/pixel_format.rs#L730)
/ [`Av1FrameHeader`](../crates/codec/src/pixel_format.rs#L792), and
[`parse_h264_pps`](../crates/codec/src/pixel_format.rs#L3401) /
`parse_h264_slice_header` / `parse_h265_*` / [`parse_av1_frame_header`](../crates/codec/src/pixel_format.rs#L1245))
expose far more fields than dimension detection needs — constraint flags, POC
branch predicates, tile info, quantization, loop-filter/CDEF/segmentation. The
struct doc comments say why: these were built to construct the Vulkan Video `Std*`
parameter/picture-info structs for a hand-rolled Vulkan decoder, and
[`Av1SequenceHeader`](../crates/codec/src/pixel_format.rs#L730) doubles as the input
to the [HLS AV1 codec-string formatter](#hlsdash-codecs-strings). They remain
useful standalone utilities (e.g. the H.264 decoder's chroma-reject sniff reads
profile + chroma in one pass via `parse_h264_sps`).

**Notes / gotchas.**
- `detect_av1` deliberately **bails to `Yuv420p`** after the timing-info block
  rather than walking the full operating-points loop — the comment judges the
  full parse not worth the maintenance cost since almost all VOD AV1 is 4:2:0. The
  *separate* `parse_av1_sequence_header` does the full color-config walk when a
  caller actually needs it.
- The AV1 sequence-header parser carries an in-code fix note:
  `initial_display_delay_present_flag` lives **outside** the timing-info branch;
  nesting it (an earlier bug) desynced every following field.
- `remove_h264_rbsp_stuffing` strips emulation-prevention bytes before any SPS
  walk — shared by the H.264 and HEVC paths.

---

## HDR SEI extraction

**What.** [`hevc_sei.rs`](../crates/codec/src/hevc_sei.rs) scans a raw Annex-B
HEVC buffer for prefix/suffix SEI NAL units (types 39/40) and extracts two HDR10
payloads: **mastering display colour volume** (payload type 137, H.265 D.2.28) and
**content light level** (type 144, D.2.35).
[`parse_annexb`](../crates/codec/src/hevc_sei.rs#L67) returns a
[`HevcHdrSei`](../crates/codec/src/hevc_sei.rs#L39) that callers fold into
`ColorMetadata` (it [`merge`](../crates/codec/src/hevc_sei.rs#L49)s newest-wins
since HDR tooling repeats the SEI on every IRAP).

**Why a hand-rolled parser.** libde265 (and the GPU decoders) don't surface SEI
messages through their public API — the `sei_message` type is internal C++ — so
the only way to recover HDR10 static metadata is to scan the bitstream ourselves.
This runs **once at demux/decoder-construction time** and never touches the decode
path; it just caches the two structs so the MP4 mux can round-trip `mdcv`/`clli`
(without which Apple devices fall back to BT.709 limited even when `colr nclx`
signals BT.2020).

**Notes / gotchas.** The parser strips HEVC emulation-prevention bytes per-NAL
before reading fields, decodes the SEI `payload_type`/`payload_size` 0xFF-run
encoding, and **remaps the spec's GBR wire order to the struct's RGB field order**
(`parse_mastering_display`). `probe.rs` re-exports this as `parse_hevc_hdr_sei` for
callers that want the scan without constructing a decoder.

---

## Probe

**What.** [`probe.rs`](../crates/codec/src/probe.rs) inspects a media file
**without a full decode**. [`probe_mp4`](../crates/codec/src/probe.rs#L25) reads an
MP4/MOV header via the `mp4` crate into a
[`ProbeResult`](../crates/codec/src/probe.rs#L15) (codec, dims, frame rate from
`sample_count / duration`, bitrate, audio track, file size) and additionally walks
the ISOBMFF box tree by hand
([`probe_mp4_visual_color_metadata`](../crates/codec/src/probe.rs#L108)) to pull
the `mdcv`/`clli` HDR atoms out of the visual sample entry.
[`detect_container`](../crates/codec/src/probe.rs#L234) sniffs MP4/MKV/AVI from
magic bytes.

**Why.** Two reasons it hand-walks boxes instead of trusting the `mp4` crate. The
HDR `mdcv`/`clli` atoms are the canonical container-side HDR10 carriers — without
surfacing them at probe time the muxer can't write them on output. And the helper
deliberately **duplicates** `container::demux::find_box_body` rather than depend on
the `container` crate, to keep `codec` free of a `container` dependency.

**Notes / gotchas.** `probe_mp4` is MP4/MOV-only; non-MP4 inputs are demuxed by the
`container` crate's streaming demuxers (which produce the `StreamInfo` the decoder
consumes — see [pipeline.md §1](pipeline.md#1-demux)). Audio sample-rate/channels
are not yet filled in here (`None`).

---

## HLS/DASH `CODECS=` strings

**What.** [`codec_strings.rs`](../crates/codec/src/codec_strings.rs) formats the
exact `CODECS="…"` attribute bytes for an HLS master playlist:
[`av1_codec_string`](../crates/codec/src/codec_strings.rs#L62) from a parsed
`Av1SequenceHeader` (`av01.P.LLT.DD.M.CCC.TTT.MMM.F`), the constant
[`AAC_LC_CODEC_STRING`](../crates/codec/src/codec_strings.rs#L103) (`mp4a.40.2`),
and [`hls_codecs_attribute`](../crates/codec/src/codec_strings.rs#L108) to join
`<video>,<audio>`.

**Why parsed from the bitstream, never composed from config.** These strings are
what hls.js / Safari native HLS / DASH players use to decide playability **before
downloading any media** — a wrong string silently drops the variant. So they must
reflect the actual encoded bitstream. The AV1 formatter encodes a hard-won
playback fix: it emits the **short form** (`av01.0.08M.08`) at SDR BT.709 defaults
and the **long** 9-component form only for HDR/wide-gamut/monochrome/full-range,
because some hls.js/Chrome/Edge versions reject the long form via
`MediaSource.isTypeSupported` even when the underlying `av1C` is byte-identical to
what the same browser plays via direct rendition load. The doc comment on the
function tells that story in full.

---

## The qsv_ffi ABI layer

**What.** [`qsv_ffi.rs`](../crates/codec/src/qsv_ffi.rs) holds the oneVPL `mfx*`
struct mirrors (`MfxFrameInfo`, `MfxInfoMfx`, `MfxVideoParam`, `MfxFrameData`,
`MfxFrameSurface1`, `MfxBitstream`, …) and shared codec/status constants used by
**both** the QSV decoder and the QSV encoder.

**Why it's its own module, and why `offsetof`-verified.** These structs were
previously duplicated in `encode/qsv.rs` and `decode/qsv_dec.rs` — "which is how
the same struct layout bug shipped in two places" (module header). Defining them
once removes that drift. And because we mirror the C ABI by hand, each struct ends
with a [compile-time size assertion](../crates/codec/src/qsv_ffi.rs#L170)
(`MfxFrameInfo == 68`, `MfxInfoMfx == 136`, `MfxVideoParam == 208`,
`MfxFrameSurface1 == 184`, `MfxBitstream == 72`, …). The sizes were verified by
`offsetof` against the installed **oneVPL 2.16** headers on a real Intel Arc box —
explicitly *not* against the dev box's vendored `mfxstructs.h`, which the comment
notes was "a wrong hand-simplified copy." Same rationale as the NVDEC ABI
witnesses: a wrong field offset means the driver reads/writes the wrong bytes
(silent garbage frames or a crash), so the layout is a build-time invariant, not a
runtime hope.

**Notes / gotchas.** The asserts are platform/ABI-sensitive (e.g. `mfxFrameId` is
`[u16; 4]`, 2-aligned, *not* a `u64`; `MfxFrameData` Y/U/V plane pointers land at
offsets 48/56/64). Touching any field without re-checking `offsetof` will trip a
`const` assertion at compile time — which is the point.

---

## Key decisions on the decode side

- **Hand-rolled `dlopen` FFI per vendor, no wrapper crate.** Buys a Windows-MSVC +
  Linux build with just a C toolchain; costs us ownership of the vendor ABI, paid
  back by compile-time size assertions + per-codec shape witnesses (NVDEC) and
  `offsetof`-verified size guards (qsv_ffi).
- **Hardware first; software says so.** No *silent* degradation — every
  software engagement is logged. `create_decoder` dispatches NVDEC → AMF → QSV →
  native h26x (H.264/HEVC, always present) → libavcodec (`ffmpeg`) → openh264
  (`openh264-fallback`) → rav1d (`rav1d-fallback`) → hard-fail.
- **The workspace owns its H.264/HEVC decoders.** `crates/h26x` is written from
  the specs, conformance-tested, threaded and SIMD'd — so the two codecs that
  make up nearly every upload decode on any host without a system library, and
  a licence question does not sit in the software tier.
- **One trait, one normalized output.** Every backend is a `Decoder`
  (`push_sample`/`decode_next`) emitting `Yuv420p`/`Yuv420p10le`, so the
  decode-once pump and everything downstream never branch on which GPU decoded.
- **Streaming over eager.** NVDEC (and QSV/AMF) drain per-sample into bounded
  queues to keep peak RSS flat on long inputs.
- **Typed rejects, not opaque strings.** `NvdecError` distinguishes "format we'll
  never support" from "transient hardware limit" so callers can steer policy.
- **One process-wide CUDA-init lock** across NVENC + NVDEC, because the driver
  segfaults on concurrent context creation from different code paths.
- **Probe + bitstream parsers are pure-Rust and decode-free**, so the pipeline can
  answer "what is this?" (format, dims, HDR metadata, codec strings) before
  constructing a decoder — and recover the data (TS dimensions, HEVC HDR SEI) that
  no container layer carries.

> **Drift flagged for maintainers:** `create_decoder` wires no `FallbackDecoder`
> despite comments implying a GPU → CPU fallover chain. The dispatch order is
> stated above and is the authority; the `FallbackDecoder` comments are the
> stale half.
