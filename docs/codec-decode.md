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

rivet's default build has no FFmpeg (an opt-in `ffmpeg` feature adds
libavcodec as a software decode tier only; see
[No FFmpeg](../README.md#no-ffmpeg)). That means decode cannot lean on an
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
tier behind the native one, and `rav1d-fallback` lets the chain fall back to
rav1d for AV1 (rav1d itself is always compiled). Every backend
implements one trait — [`Decoder`](#the-decoder-trait) — so the pump drives them
all identically (`push_sample` → `decode_next`). The hardware backends emit
`Yuv420p` / `Yuv420p10le`; the software tiers can also emit 12-bit and
4:2:2 / 4:4:4 planar frames, which the colorspace layer narrows before any
encoder sees them, so the rest of the pipeline never branches on which GPU
produced the pixels.

---

## Module map

| File | Purpose |
|------|---------|
| [`src/lib.rs`](../crates/codec/src/lib.rs) | Crate root: module declarations + the `ColorSpace` / `PixelFormat` / `VideoFrame` / `GpuDevice` / `GpuVendor` re-exports, and `pixel_format` re-exported from `rivet-frame`. |
| [`src/frame.rs`](../crates/codec/src/frame.rs) | A re-export of the [`rivet-frame`](../crates/frame/README.md) crate ([`crates/frame/src/lib.rs`](../crates/frame/src/lib.rs)), which holds the core data types: `VideoFrame`, `PixelFormat`, `ColorSpace`/`TransferFn`, `StreamInfo`, `VideoCodec`, `EncodedPacket`, and the HDR metadata structs (`ColorMetadata`, `MasteringDisplay`, `ContentLightLevel`). They live there so `rivet-container` can use them without this crate's GPU/audio dependencies. |
| [`src/decode/mod.rs`](../crates/codec/src/decode/mod.rs) | The `Decoder` trait + `create_decoder` dispatch, the `HardwareThenSoftware` late-fallback guard, `RotatingDecoder`, the `DISABLE_*` / `RIVET_DISABLE_H26X` env knobs, shared NV12/P010 deinterleave helpers, and the capability introspection (`decode_backends`, `decode_capabilities`, `decode_capable_gpu_indices`). |
| [`src/decode/nvdec/`](../crates/codec/src/decode/nvdec/mod.rs) | NVIDIA NVDEC/CUVID streaming decode — hand-rolled libnvcuvid FFI with ABI-pinned structs. Split into `ffi.rs` (structs + size witnesses), `state.rs`, `convert.rs` (format validation, display geometry, P016 deinterleave), `callbacks.rs`, `streaming.rs` (the production decoder), `eager.rs` / `push.rs` (retained library/test paths). |
| [`src/decode/qsv_dec.rs`](../crates/codec/src/decode/qsv_dec.rs) | Intel QSV/oneVPL decode — hand-rolled libvpl FFI, internal-allocation + `FrameInterface::Map`. |
| [`src/decode/amf_dec.rs`](../crates/codec/src/decode/amf_dec.rs) | AMD AMF decode over the shared AMF FFI. H.264 and HEVC (8-bit and Main 10) verified on hardware; what the host's GPU decodes is probed from the runtime. |
| [`src/amf_ffi.rs`](../crates/codec/src/amf_ffi.rs), [`src/amf_runtime.rs`](../crates/codec/src/amf_runtime.rs) | The AMF SDK v1.4.36 vtable mirrors (slot offsets pinned by `const` asserts) and the runtime / context lifecycle (dlopen, `AMFInit`, context bound to the chosen AMD GPU, property helpers), shared by the AMF decoder and encoder (`amd` feature). |
| [`src/amf_device.rs`](../crates/codec/src/amf_device.rs) | Windows-only: hand-rolled DXGI/D3D11 dlopen FFI that makes a D3D11 device on a *specific* AMD adapter, so AMF's `InitDX11` binds to the right GPU on a mixed host (gated `windows` + `amd`). |
| [`src/decode/h26x_sw.rs`](../crates/codec/src/decode/h26x_sw.rs) | **Native H.264 / HEVC decode** on this workspace's own [`h26x`](../crates/h26x/README.md) crate — pure Rust, always compiled, frame + wavefront threaded, SIMD kernels chosen at run time, bit-exact against the JVT / JCT-VC conformance suites. The software tier for the two codecs it serves; refuses (`Unsupported`) up front what it does not do, so the guard rebuilds the next tier. |
| [`src/decode/ffmpeg.rs`](../crates/codec/src/decode/ffmpeg.rs) | libavcodec software decode (optional `ffmpeg` feature; the one backend needing host libraries at build time). Behind the native tier: catches what the native tier refuses (H.264 data partitioning, unequal luma / chroma depths; HEVC SCC, multi-layer), and the other codecs (VP8/VP9/AV1/MPEG-2/MPEG-4/ProRes). |
| [`src/decode/openh264_sw.rs`](../crates/codec/src/decode/openh264_sw.rs) | Software H.264 via openh264 (optional `openh264-fallback`), the narrow last resort below libavcodec. |
| [`src/decode/rav1d_sw.rs`](../crates/codec/src/decode/rav1d_sw.rs) | Software AV1 decode via [rav1d](https://crates.io/crates/rav1d) — always compiled; the `rav1d-fallback` feature decides whether the dispatch chain falls back to it. Hand-rolled `extern "C"` over the dav1d ABI, no system library. |
| [`src/audio/decode/`](../crates/codec/src/audio/decode/mod.rs) | Audio decoders behind `audio::create_decoder`: AAC (via the `crates/aac` submodule), AC-3 / E-AC-3, DTS core, FLAC, ALAC, MP1/MP2/MP3 (minimp3), Opus (libopus), Vorbis (lewton), linear PCM. See [AC-3 / E-AC-3](#ac-3--e-ac-3-decoder), [AAC](#aac-decoder), [FLAC and ALAC](#flac-and-alac-decoders) and [Other audio decoders](#other-audio-decoders) below. |
| [`src/audio/decode/ac3/`](../crates/codec/src/audio/decode/ac3/mod.rs) | **In-tree AC-3 / E-AC-3 decoder**, pure Rust, written from ATSC A/52:2018 — tables (with per-table tests against the spec pages), bit reader, parametric bit allocation, IMDCT, syncframe decoder, `AudioDecoder` adapter. Cross-checked against libavcodec; see [AC-3 / E-AC-3 decoder](#ac-3--e-ac-3-decoder). |
| [`src/gpu/`](../crates/codec/src/gpu/mod.rs) | GPU detection (`detect_gpus`, `detect_gpus_cached`, `vendor_index_of`), `GpuDevice`/`GpuVendor`, per-vendor scans (`nvidia.rs`, `amd.rs`, `intel.rs`) with NVML + sysfs (Linux) / WMI (Windows) enrichment, render-node filtering, the PCI BAR report (`bar.rs`), the live-utilisation reader, `supports_av1_encode`. |
| [`src/cuda_lock.rs`](../crates/codec/src/cuda_lock.rs) | Process-wide CUDA-init mutex shared by NVENC + NVDEC (`nvidia` feature only). |
| [`src/probe.rs`](../crates/codec/src/probe.rs) | Media probing without a full decode (MP4 header walk + container sniff + HDR box extraction). |
| [`crates/frame/src/pixel_format/`](../crates/frame/src/pixel_format/mod.rs) | Pure-Rust bitstream parsers: SPS/PPS/sequence-header walkers for H.264 / HEVC / AV1 / MPEG-2 (pixel format + dimensions), VP9 colour, keyframe detection. Lives in `rivet-frame`; reached here as `codec::pixel_format`. |
| [`src/hevc_sei.rs`](../crates/codec/src/hevc_sei.rs) | Re-export of `frame::hdr_sei`: the H.264 / HEVC SEI 137/144 scanner → HDR10 mastering-display + content-light-level metadata. |
| [`src/codec_strings.rs`](../crates/codec/src/codec_strings.rs) | HLS/DASH `CODECS="…"` string formatters (AV1, H.264, H.265, AAC-LC) parsed from the bitstream. |
| [`src/qsv_ffi.rs`](../crates/codec/src/qsv_ffi.rs) | Shared oneVPL `mfx*` struct mirrors used by both the QSV encoder and decoder, pinned with `offsetof`-verified size asserts. |

---

## Frame types

**What.** [`rivet-frame`](../crates/frame/src/lib.rs) (re-exported unchanged as
`codec::frame`) defines the values every decoder produces and every consumer
reads.

- [`VideoFrame`](../crates/frame/src/lib.rs#L194) — the unit of decoded output:
  `data: Bytes` (the planar pixels), `width`/`height`, a `PixelFormat`, a
  `ColorSpace`, and a `pts`. The pixel buffer is `bytes::Bytes`, which is
  `Arc`-backed — that is load-bearing for the decode-once pump: fanning one
  decoded frame out to N rungs is a `clone()` that bumps a refcount, not a pixel
  copy (see [pipeline.md §2](pipeline.md#2-decode-once--the-shared-pump)).
- [`PixelFormat`](../crates/frame/src/lib.rs#L51) — the planar layouts the crate
  can carry (4:2:0/4:2:2/4:4:4 at 8/10/12-bit, 10-bit 4:4:4 with alpha,
  NV12/NV21, RGB). The GPU decoders normalize to just `Yuv420p` and
  `Yuv420p10le`; the native H.264/HEVC and rav1d tiers also emit the 12-bit and
  4:2:2 / 4:4:4 formats, and the probe/parsers can *report* formats the encoder
  won't ultimately produce (every encoder takes 4:2:0). [`bytes_per_frame`](../crates/frame/src/lib.rs#L83)
  gives the packed size, and [`from_chroma_and_depth`](../crates/frame/src/lib.rs#L122)
  maps a `(chroma_idc, bit_depth)` pair (what the bitstream parsers extract) to
  the enum, with a defensive `Yuv420p` default.
- [`ColorSpace`](../crates/frame/src/lib.rs#L139) (BT.601/709/2020) and
  [`TransferFn`](../crates/frame/src/lib.rs#L161) (BT.709 gamma, PQ/ST2084, HLG,
  …). These are **deliberately separate**: every decoder already emits
  `VideoFrame { color_space, .. }` and the converters/encoder dispatch on it, so
  keeping the transfer function on a side channel let HDR support land without
  touching every call site. [`TransferFn::from_h273`](../crates/frame/src/lib.rs#L181)
  maps the raw ITU-T H.273 `transfer_characteristics` byte down to the subset the
  pipeline knows.
- [`StreamInfo`](../crates/frame/src/lib.rs#L224) — the demuxer's header
  (codec string, dims, frame rate, duration, source pixel format, color
  metadata). It's the input to `create_decoder` and the output of `probe`.
- [`ColorMetadata`](../crates/frame/src/lib.rs#L248) bundles all HDR-relevant
  signaling: transfer function, raw H.273 `matrix_coefficients` / `colour_primaries`
  / `full_range_flag`, plus optional [`MasteringDisplay`](../crates/frame/src/lib.rs#L317)
  (SMPTE ST 2086 / HEVC SEI 137) and [`ContentLightLevel`](../crates/frame/src/lib.rs#L344)
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
[`create_decoder`](../crates/codec/src/decode/mod.rs#L366) /
[`create_decoder_on`](../crates/codec/src/decode/mod.rs#L380) factory that picks a
decoder for a `(codec, StreamInfo)` and an optional GPU index.

### The `Decoder` trait

[`Decoder`](../crates/codec/src/decode/mod.rs#L187) is three methods plus
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

[`create_decoder_on`](../crates/codec/src/decode/mod.rs#L380) calls
[`gpu::detect_gpus()`](#gpu-detection) once (on a build with a GPU feature), then
tries, in order:

1. **NVDEC** (`nvidia` feature) — if an NVIDIA device is present (or matches the
   requested `gpu_index`), the codec is in
   [`nvdec_supports`](../crates/codec/src/decode/mod.rs#L242), and it isn't
   disabled by env-var, return an `NvdecDecoder`. A CUDA/parser init failure
   does not fail construction: `NvdecDecoder::new` returns a decoder that
   reports the error on the first push, which the guard turns into a
   fallback. NVIDIA wins ties because NVDEC is generally lower-latency on the
   standard codec set and is what the fleet is tuned against (comment at
   `create_decoder`).
2. **AMF** (`amd` feature) — first AMD device (or the requested one) +
   [`amf_dec::host_supports`](../crates/codec/src/decode/amf_dec.rs#L142), a
   per-process probe of which decoder components this host's AMD GPU actually
   has.
3. **QSV** (`qsv` feature) — first Intel device (or the requested one) +
   [`qsv_dec::supports`](../crates/codec/src/decode/qsv_dec.rs#L118).

   An AMF or QSV decoder whose construction fails (no runtime, a context or
   session that will not open) logs a warning and the chain moves on to the
   next tier rather than failing the job. A refusal that comes later — QSV's
   `MFXVideoDECODE_Init` returning `MFX_ERR_UNSUPPORTED` once the header has
   been parsed, for a profile or size the fixed-function block will not take —
   is caught by the late-fallback guard described below.
4. **Native H.264 / HEVC** ([`h26x_sw`](../crates/codec/src/decode/h26x_sw.rs),
   always compiled) — rivet's own decoders, first among the software tiers for
   those two codecs. Wrapped in the same guard as the hardware tiers: a stream
   they refuse (an `Unsupported` on the parameter set — H.264 data
   partitioning, unequal luma / chroma depths, SP/SI outside the Extended
   profile's shape; HEVC SCC, multi-layer, separate colour planes), or a
   first picture with no pipeline pixel format, is replayed into the next
   tier with nothing lost. `RIVET_DISABLE_H26X=1` skips
   it. See [Native H.264 / HEVC](#native-h264--hevc--decodeh26x_swrs).
5. **libavcodec** (`ffmpeg` feature) — the broad software tier, behind the
   native one; removed 2026-08-12, restored 2026-08-14 once `create_decoder`
   actually constructed it.
6. **openh264** (`openh264-fallback`), H.264 only — the narrow last resort.
7. **Software AV1** (rav1d, AV1 only; reached only on a `rav1d-fallback`
   build) — off by default, and below every vendor path so it is a floor
   rather than a preference.
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

### Rotation and stills

[`RotatingDecoder`](../crates/codec/src/decode/mod.rs#L133) wraps any decoder
so its frames come out already turned by the container's rotation (90, 180 or
270 clockwise, through `filter::VideoFilter::Rotate`), with `stream_info()`
reporting the swapped dimensions for 90 / 270. Every other angle, 0 included,
returns the inner decoder unchanged. The decode pump, the thumbnail path and
per-title sampling wrap their decoders with it, so no consumer sees an
unrotated frame.

Still images use the same dispatch: an AVIF's coded items go to the AV1
chain and a HEIC's to the HEVC chain (rivet's `image` feature; see
[output-spec.md §11](output-spec.md#11-still-images--modeimage)). rav1d
decodes every AV1 layout and depth for that reason — 4:4:4 is what most AVIF
encoders write.

### The `DISABLE_*` env knobs

[`nvdec_disabled_for`](../crates/codec/src/decode/mod.rs#L223) (`nvidia` builds) is a debugging
escape hatch: `DISABLE_NVDEC=1` skips NVDEC for every codec, and
`DISABLE_NVDEC_<CODEC>=1` (e.g. `DISABLE_NVDEC_H264`, `DISABLE_NVDEC_AV1`) skips
one codec family. The point is operational — when a specific codec/driver combo
misbehaves on a host (the comment cites a "Blackwell + 4K H.264 silent-stall"),
you can disable just that path and fall through to the next tier without a
rebuild. [`env_flag_truthy`](../crates/codec/src/decode/mod.rs#L207) parses
`1`/`true`/`yes`/`on`/`y`/`t` (case-insensitive). `RIVET_DISABLE_H26X` takes
the same values and removes the native H.264 / HEVC tier, for comparing
against the tiers below it.

### Shared deinterleave helpers

Two `pub(crate)` helpers convert vendor surface layouts into the crate's packed
planar convention, shared so the GPU backends can't drift (NVDEC's 10-bit P016
path has its own, below):
- [`nv12_planes_to_yuv420p`](../crates/codec/src/decode/mod.rs#L44) — NV12
  (Y plane + interleaved UV, each with its own stride) → packed `[Y | U | V]`.
- [`p010_planes_to_yuv420p10le`](../crates/codec/src/decode/mod.rs#L80) — host
  P010 (10-bit in the **high** bits of each u16) → `Yuv420p10le` (10-bit in the
  **low** bits via `>> 6`).

### Capability introspection

[`decode_backends()`](../crates/codec/src/decode/mod.rs#L267) (the compiled
backends in dispatch order: `nvdec`, `amf`, `qsv`, `h26x`, `ffmpeg`,
`openh264`, `rav1d`, filtered by feature) and
[`decode_capabilities()`](../crates/codec/src/decode/mod.rs#L302) (which of them
decodes each of H.264, HEVC, VP8, VP9, AV1, MPEG-2, MPEG-4 and ProRes) back the
`rivet capabilities` CLI command, not the runtime dispatch. The software rows
are listed only for the codecs each tier serves. The AMF and QSV rows are not
static guesses: they come from **runtime hardware probes** —
[`amf_dec::probe_decode_caps`](../crates/codec/src/decode/amf_dec.rs#L110)
(one `CreateComponent` per decoder id) and
[`qsv_dec::probe_decode_caps`](../crates/codec/src/decode/qsv_dec.rs#L152) (a
oneVPL HW session and `MFXVideoDECODE_Query` per codec) — so the report shows
AMF or QSV decode only on a host where that runtime + adapter actually
initialise.

[`decode_capable_gpu_indices(codec)`](../crates/codec/src/decode/mod.rs#L821)
lists the global GPU indices whose vendor decoder takes `codec` in this build
(honouring the `DISABLE_NVDEC*` knobs) — the candidates `--decode fastest`
benchmarks before pinning the pump to the quickest.

**Notes / gotchas (drift to be aware of).**
- **The `gpu_index` argument is load-bearing for multi-GPU.** `create_decoder`
  (no index) keeps the legacy "first matching adapter" behaviour for one-shot
  callers; the pipeline's per-rung pumps should pass `Some(idx)` so each rung's
  decode session lands on a distinct physical adapter (the doc comment on
  `create_decoder_on` flags that, without it, every QSV session piles onto the
  first Intel card). NVDEC and AMF open the chosen device's `vendor_index`;
  `QsvDecoder::new` takes it but does not use it yet — its
  `MFXInit(MFX_IMPL_HARDWARE_ANY)` lets the runtime pick the adapter.
- **`FallbackDecoder` does not exist**, but late fallback does:
  `HardwareThenSoftware` (in `decode/mod.rs`) wraps a hardware tier — and the
  native `h26x` tier — so a decoder that accepts construction and then refuses
  the stream is replaced by the next tier, with everything fed so far
  replayed. The guard holds until the decoder has produced a **frame**, not
  until it has accepted a sample: NVDEC's parser reads a sample's last NAL unit
  when the next one arrives, so it refuses an unsupported sequence (10-bit or
  4:4:4 H.264, 4:2:2 HEVC on an RTX 3090) on the second push, and until
  2026-09-18 the guard had already dropped the fallback after the first and the
  job failed. A refusal from a push, from `finish`, from `decode_next`, or a
  stream taken to the end with no frame out, all degrade; the replay is capped
  at 64 MiB. NVDEC surfaces what its callbacks record (`cuvidCreateDecoder`
  failing, a picture that will not decode or map) while it has produced
  nothing, instead of accepting samples and yielding nothing. With no software
  tier for the codec the error names both ("the hardware decoder refused this
  stream (…), and no software decoder can take it: …"). Once a decoder has
  produced a frame the guard is dropped: a failure on sample nine thousand is a
  stream error, not a capability question.
  `create_decoder_on` wires NVDEC → AMF → QSV → h26x → libavcodec → openh264 →
  rav1d → hard-fail.
  *(Historical note: an FFmpeg tier used to be listed here as "present but not
  wired" — capability-listed and never constructed — which is why it was removed
  outright on 2026-08-12 and restored, constructed, on 2026-08-14.)*
- The module header still says "exactly two backends (NVDEC + QSV)" and "no
  CPU decode path of any shape"; both predate the AMF decoder and the software
  tiers listed above. The dispatch order above is the authority.

---

## GPU decode backends

All three hardware backends share the same shape: `dlopen` the vendor runtime,
stand up a context/session, drive it through the `Decoder` trait, and copy decoded
surfaces back to host memory in `Yuv420p` / `Yuv420p10le`. They differ in the
vendor API and how much the dev box could verify.

### NVDEC (NVIDIA) — `decode/nvdec/`

**What.** [`nvdec/`](../crates/codec/src/decode/nvdec/mod.rs) loads `libcuda` +
`libnvcuvid` at runtime and drives the CUVID stateless-parser API:
`cuInit` → `cuCtxCreate` → `cuvidCreateVideoParser`, then per sample
`cuCtxPushCurrent` + `cuvidParseVideoData` + `cuCtxPopCurrent`. Three parser
callbacks do the work: the sequence callback creates the decoder + validates the
format, the decode callback runs `cuvidDecodePicture`, and the display callback
maps the frame (`cuvidMapVideoFrame` + `cuMemcpy2D`) and pushes NV12/P016 bytes
into a shared collector. The production path is the **streaming**
[`NvdecStreamingDecoder`](../crates/codec/src/decode/nvdec/streaming.rs#L113) ([engaged from `NvdecDecoder::new`](../crates/codec/src/decode/nvdec/eager.rs#L75)):
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
  [136 bytes](../crates/codec/src/decode/nvdec/ffi.rs#L457), `CUVIDPICPARAMS` at
  [4280](../crates/codec/src/decode/nvdec/ffi.rs#L485) (its `codec_specific` region is
  the SDK's 4096-byte `CodecReserved[1024]` envelope), `CUVIDPARSERDISPINFO`,
  `CUVIDDECODECAPS`, and `CUVIDSOURCEDATAPACKET` *per platform* (the `c_ulong`
  width differs Windows vs Linux — [24 vs 32 bytes](../crates/codec/src/decode/nvdec/ffi.rs#L502)).
  The comment block lists the real bugs this caught: the parser-params 80→136 fix
  and the pic-params 2048→4280 fix, both of which had produced the segfault-hunt
  class of corruption.
- **Per-codec "shape witness" structs** ([H264/HEVC/AV1/VP9/VP8/MPEG2/MPEG4](../crates/codec/src/decode/nvdec/ffi.rs#L260-L389))
  are dead-code mirrors that exist *only* so a `const_assert!` proves each codec's
  pic-params variant fits in the 4096-byte envelope. They're never used at
  runtime; they're a tripwire so a future SDK that grows one variant fails
  compilation instead of silently overflowing the parser state and reproducing the
  original segfault on a different code path.

**Typed rejects.** [`validate_format`](../crates/codec/src/decode/nvdec/convert.rs#L46)
is a pure function (unit-testable without a GPU) that turns the CUVID-reported
chroma/bit-depth into a typed [`NvdecError`](../crates/codec/src/decode/nvdec/mod.rs#L74):
`UnsupportedChroma` (only 4:2:0 passes; monochrome/4:2:2/4:4:4 reject),
`UnsupportedPixelFormat` (>12-bit), `UnsupportedByHardware` (per-GPU caps). Why
typed: a reviewer note records that these used to surface as an opaque "NVDEC
produced no frames: <string>", and the pipeline couldn't tell "4:2:2 unsupported"
(a format we'll never decode) from "driver OOM" (transient) — the typed variant
keeps a fallback/abort decision explainable via `downcast_ref::<NvdecError>()`.

**The frame is the display area, not the coded surface.**
[`output_geometry`](../crates/codec/src/decode/nvdec/convert.rs#L115) works out,
from the parser's `CUVIDEOFORMAT`, the coded (padded) surface and the display
rectangle inside it — H.264 frame cropping, the HEVC conformance window, the
AV1 / VP9 frame size — and hands the rectangle to `cuvidCreateDecoder` as its
`display_area`, so frames come out at the stream's display size. A 640x360
stream is coded 640x368 and 1080p is coded 1088; a 368-row frame resampled
into a 360-row rung was a vertical squash that cost 20 dB on every job decoded
through NVDEC. If the driver reports no usable display area, the padded
surface is used and a warning logged.

**Notes / gotchas.**
- 10-bit comes back as **P016** (10 bits in the high bits of each u16);
  [`deinterleave_p016_to_yuv420p10le`](../crates/codec/src/decode/nvdec/convert.rs#L173)
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
[`MFXVideoDECODE_DecodeHeader`](../crates/codec/src/decode/qsv_dec.rs#L269) on the
first buffered samples → `MFXVideoDECODE_Init` → per sample
[`DecodeFrameAsync` + `SyncOperation`](../crates/codec/src/decode/qsv_dec.rs#L384),
then [read the surface](../crates/codec/src/decode/qsv_dec.rs#L444) into a
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
- [`read_surface`](../crates/codec/src/decode/qsv_dec.rs#L444) emits the **crop
  (display)** dims, not the coded dims — 1080p codes as 1088 (16-aligned), and
  feeding 1088-tall frames into a 1080-configured encoder fails
  `EncodeFrameAsync`. `try_init` fills a zero crop from the coded size and
  aligns the surface to 16 (32 rows for interlaced content).
- **The adapter index is not used yet.** `QsvDecoder::new(info, _gpu_index)`
  initialises with `MFX_IMPL_HARDWARE_ANY`, so on a multi-Intel host the
  runtime picks the adapter, whatever `gpu_index` the caller pinned.
- Plane pointers are valid only between `Map`/`Unmap`; the read copies out inside
  that window.
- **Hardware-verified on a 3× Intel Arc box** (A310 / A380 / A750, oneVPL 2.16 /
  iHD): H.264, HEVC, VP9, and AV1 each decode end-to-end (transcode to AV1 on a
  qsv-only build, with the QSV decoder engaged).

**Capability probe.** [`probe_decode_caps`](../crates/codec/src/decode/qsv_dec.rs#L152)
is the decode side of `rivet capabilities`: it opens one HW `MFXInit` session and
`MFXVideoDECODE_Query`s each codec. iHD's Query is *advisory* — on the Arc box it
returns an error for every codec it nonetheless decodes — so a successful
`MFXInit(HW)` is the load-bearing signal: when Query yields nothing, the probe
reports the build's codec list (the runtime is usable) rather than claim no
decode; a non-empty Query result is trusted as-is (to drop, say, AV1 on a pre-Arc
iGPU). It returns empty on a non-Intel host, so the report shows QSV decode only
where it actually runs. This is wired into
[`decode_capabilities`](../crates/codec/src/decode/mod.rs#L302).

### AMF (AMD) — `decode/amf_dec.rs`

**What.** [`amf_dec.rs`](../crates/codec/src/decode/amf_dec.rs) drives the AMF
runtime through the COM-style vtables mirrored in
[`amf_ffi.rs`](../crates/codec/src/amf_ffi.rs) and the runtime / context
lifecycle in [`amf_runtime.rs`](../crates/codec/src/amf_runtime.rs), both shared
with the AMF encoder: `AmfRuntime::open` (dlopen → `AMFInit` → `CreateContext` →
a context bound to the chosen AMD GPU) → `CreateComponent(<decoder id>)` →
`Init(NV12 | P010, w, h)`, then per sample `AllocBuffer(HOST)` + copy the
Annex-B access unit → `SetPts` → `SubmitInput` (with the `AMF_INPUT_FULL`
drain-and-retry) → loop `QueryOutput` → `QueryInterface(IID_AMFSurface)` →
`Convert(HOST)` → read the NV12 / P010 planes into `Yuv420p` / `Yuv420p10le`.
`finish` is `Drain`, then `QueryOutput` polled to `AMF_EOF` (bounded at 10 s).
Decodes H.264 / HEVC / VP9 / AV1, as far as the GPU has the block.

**What this host decodes is asked, not assumed.**
[`probe_decode_caps`](../crates/codec/src/decode/amf_dec.rs#L110) opens one
context on the first AMD GPU and tries `CreateComponent` for each decoder id
once per process; [`host_supports`](../crates/codec/src/decode/amf_dec.rs#L142)
answers from that, for the dispatch and for `rivet capabilities`. A VCN
without an AV1 block is not offered AV1.

**Multi-adapter routing (Windows).** AMF's `InitDX11(null)` lets the runtime
create its device on **DXGI adapter 0** — which on a mixed host (an NVIDIA card in
slot 0 + an AMD GPU elsewhere) is the wrong, non-AMD adapter, and init fails. So
on Windows we build the D3D11 device ourselves on the chosen AMD adapter:
[`amf_device::create_amd_d3d11_device`](../crates/codec/src/amf_device.rs)
enumerates adapters via DXGI, finds the `vendor_index`-th `0x1002` one, and
`D3D11CreateDevice`s it with `D3D11_CREATE_DEVICE_VIDEO_SUPPORT` — then hands that
device to `InitDX11(dev, AMF_DX11_1)` (`AmfRuntime` keeps it alive for the
context's lifetime). On Linux the context is `QueryInterface(AMFContext1)` →
`InitVulkan(null)`, which picks the first AMD GPU (a non-zero `vendor_index` is
logged as a warning). See [GPU detection](#gpu-detection) for how a global
index maps to a vendor-local adapter.

**`QueryOutput` is read by its data pointer, not its result code.** After
`Drain`, the UVD decoder on this driver hands back the frames still in flight
tagged `AMF_REPEAT` *with a non-null buffer*, and only the last as `AMF_OK`;
treating `AMF_REPEAT` as "nothing yet" lost the tail of every stream (58 of 60
frames measured). The drain takes a frame whenever the buffer is non-null with
`AMF_OK` or `AMF_REPEAT`, as libavcodec's `amf_receive_frame` does.

**Verified on hardware** (2026-09-13, Ryzen 9 9950X iGPU): H.264 and HEVC
8-bit and HEVC Main 10 decode byte-for-byte equal to ffmpeg and to the native
`h26x` decoders
([`tests/amf_decode_pixels.rs`](../crates/codec/tests/amf_decode_pixels.rs)).
The test also carries an AV1 clip, checked against ffmpeg only; VP9 has no
on-hardware test.

**Notes / gotchas.**
- `gpu_index` selects the AMD adapter on Windows (via the D3D11 routing
  above), **not** "adapter 0 unconditionally" as older comments claimed.
- A failed `InitDX11` / `InitVulkan` releases the context and returns an error
  naming the adapter ("… this GPU is not one the AMF runtime drives"); the
  dispatch logs it and tries the next tier, and `--decode fastest` skips that
  GPU instead of taking the process down.

### Native H.264 / HEVC — `decode/h26x_sw.rs`

**What.** [`h26x_sw.rs`](../crates/codec/src/decode/h26x_sw.rs) drives
[`crates/h26x`](../crates/h26x/README.md): two decoders written from the ITU-T
specifications. H.264: Baseline through High 4:4:4 Predictive and the Intra
profiles — frames, field pictures and MBAFF, 4:0:0 / 4:2:0 / 4:2:2 / 4:4:4 at
8–14-bit, separate colour planes, lossless, CAVLC/CABAC, B-frames,
temporal/spatial direct, weighted prediction, 8x8 transform, scaling matrices,
MMCO/long-term, PCM, slice groups (FMO), ASO, and the Extended profile's SP /
SI slices. HEVC: Main / Main 10 / Main 12 and the format range extensions
(4:0:0 / 4:2:0 / 4:2:2 / 4:4:4, 8–16-bit, unequal luma / chroma depths) with
WPP, tiles, dependent slices, SAO, PCM, transform skip, scaling lists, TMVP,
weighted prediction. The crate README has the full supported / refused table.
Both are bit-exact against the conformance suites (JVT AVCv1 + FRExt: all
204; JVT professional profiles: all 38; JCT-VC HEVC_v1: 147/147; RExt: all
49) and both are threaded — pictures in flight concurrently with
reference-progress waits, plus wavefront rows / tiles inside a picture for
HEVC — with SIMD kernels picked at run time (SSE2 through AVX2, plus AVX-512
for some HEVC kernels, on x86-64; NEON, plus dot product where present, on
AArch64). `H26X_THREADS`, `H26X_INFLIGHT`, `H26X_NO_SIMD`, `H26X_MAX_SIMD`
tune it; the crate README lists the rest.

**Where.** First among the software tiers for the two codecs, below every
hardware tier. It is always compiled — pure Rust, no toolchain — so unlike
`ffmpeg` it is present in every build, and unlike `openh264-fallback` it handles
the profiles that actually arrive. libavcodec, when built, sits behind it and
takes what it refuses. `RIVET_DISABLE_H26X=1` takes it out of the chain.

**Output.** 4:2:0 as `Yuv420p` / `Yuv420p10le` / `Yuv420p12le`, 4:2:2 and
4:4:4 as `Yuv422p*` / `Yuv444p*` at the same three depths; 9-bit is widened
to 10 and 11-bit to 12. Monochrome travels as 4:2:0 with grey chroma. Depths
above 12 (H.264's 13 / 14-bit professional profiles, HEVC RExt 16-bit) and
pictures whose luma and chroma depths differ have no pipeline pixel format and
are refused by name. Frames come out in output (POC) order, numbered from
zero, like the AV1 tier.

**Refusals.** An `h26x::Error::Unsupported` is returned from `push_sample`
before any picture exists — on the parameter set for H.264, at the first slice
where the SPS is checked — and the tier guard hands the stream on. A bitstream
error before the first picture is likewise a refusal, so the next tier gets a
go; after it, errors are logged and skipped, matching the other software tiers
(a stream with a damaged NAL is not a failed job).

### Software AV1 — `decode/rav1d_sw.rs`

**What.** [`rav1d_sw.rs`](../crates/codec/src/decode/rav1d_sw.rs) drives
[rav1d](https://crates.io/crates/rav1d) — a Rust port of dav1d — behind the
`Decoder` trait. It is always compiled; the `rav1d-fallback` feature decides
whether the dispatch chain falls back to it (engagement is logged at `warn`).
It is the only software AV1 decoder in the tree apart from libavcodec, and it
decodes every layout and depth AV1 defines: 4:2:0, 4:2:2 and 4:4:4 at
8, 10 and 12 bits, as the same planar formats the native HEVC decoder emits,
with monochrome (an AVIF's alpha plane) as 4:2:0 with neutral chroma. Until
2026-09-29 it took 8-bit 4:2:0 only, which refused most AVIF stills.

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

**Why it exists.** When it came back, a CPU-only host could not decode
anything at all — the factory ran NVDEC → AMF → QSV → hard-fail. It still
matters on GPU hosts: NVDEC gained AV1 in Ampere while NVENC only got it in
Ada, so a host can encode AV1 in hardware and have no way to decode it. AV1 is
the format rivet itself
produces, so being able to read one back without a GPU is what makes a
round-trip test runnable in CI, and it costs no system dependency to have.

**`rav1d-asm`** turns on rav1d's hand-written assembly. It is off by default
because it needs **NASM** on the build host; the pure-Rust path is slower and
builds anywhere.

---

## AC-3 / E-AC-3 decoder

**What.** [`audio/decode/ac3/`](../crates/codec/src/audio/decode/ac3/mod.rs) is
an in-tree, pure-Rust Dolby Digital / Digital Plus decoder written from ATSC
A/52:2018, reached through `audio::create_decoder("ac3" | "eac3")` and the
job layer's decodable list (so 5.1 AC-3 / E-AC-3 → Opus 5.1 goes through the
normal decode → `channelmap` → Opus path). Files:

| File | Purpose |
|------|---------|
| `tables.rs` | Every normative table — Tables 5.18, 7.6–7.16, 7.18–7.23, 7.33, E2.10–E2.12, E3.1/E3.2/E3.6/E3.13/E3.14 and the VQ codebooks E4.1–E4.7 — transcribed from the spec PDF (`pdftotext -layout`, `pdftoppm` for the pages that matter), the PDF page on each doc comment, and a test per table that re-reads spot values off the rendered page plus a checksum. The KBD window is also derived analytically and matched to Table 7.33 to five decimals. |
| `bits.rs` | MSB-first reader. A read past the frame is an error, not zero padding: it means the side information was mis-parsed, and zeros would turn that into plausible garbage. |
| `bitalloc.rs` | §7.2.2 in the spec's fixed-point integer arithmetic (fbw / LFE / coupling initialisations, delta bit allocation) plus the E-AC-3 `hebap` lookup. Must be bit-exact with the encoder or the mantissa field widths diverge. |
| `imdct.rs` | §7.9.4 as the spec writes it — pre-twiddle, N/4- or N/8-point complex IFFT, post-twiddle, window, de-interleave, overlap-add — with a radix-2 FFT standing in for the O(N²) sum (a test checks the two agree). |
| `decoder.rs` | `syncinfo` / `bsi` (AC-3 Table 5.2, E-AC-3 Table E1.2), `audfrm`, `audblk`, exponents (§7.1.3), mantissas incl. the grouped 3/5/11-level quantisers (§7.3), coupling with phase flags (§7.4), spectral extension (Annex E §3.6), rematrixing (§7.5), `dynrng` (§7.7.1), AHT (VQ + GAQ, §3.4). |
| `mod.rs` | The `AudioDecoder` adapter: resynchronises on 0x0B77, buffers partial syncframes, derives pts from the sample count; `Ac3Options::drc_scale`. |

Coverage: **AC-3 (bsid ≤ 8) complete** — block switching, dither, coupling
with phase flags, rematrixing, delta bit allocation, `dynrng` (applied by
default, scalable). **E-AC-3 (bsid 16) independent substream 0** — every
`numblkscod`, reduced sample rates, frame exponent strategies, the three
SNR-offset strategies, standard coupling, spectral extension with
attenuation, AHT. Refused by name: enhanced coupling (`ecplinu = 1`), bsid
9/10. Skipped by name (Annex E §3.8.1): dependent substreams and independent
substreams other than 0, so 7.1 decodes as its 5.1 core. `dialnorm` / `compr`
are parsed, not applied — libavcodec's default. Output is interleaved f32 in
ffmpeg's native order for the layout (5.1: FL FR FC LFE SL SR), which is what
`channelmap` and the Opus encoder assume for a channel count.

**Licence.** Every table came from the spec on disk; nothing from libavcodec
or any other implementation. libavcodec is used only as a black-box oracle
through the real ffmpeg binary.

**Verification** ([`tests/ac3_decode_vectors.rs`](../crates/codec/tests/ac3_decode_vectors.rs);
vectors from [`tests/data/ac3_make_vectors.sh`](../crates/codec/tests/data/ac3_make_vectors.sh),
Dolby-encoded streams from ffmpeg's FATE suite, `RIVET_AC3_VECTORS=<dir>`).
A/52 defines the bit allocation in exact integers but leaves the transform
and dequantisation to floating point and lets dither and the SPX noise be
"any reasonably random sequence", so two conformant decoders agree to float
rounding where the stream is deterministic and differ by their independent
noise where it is not. The gate is therefore relative to a *measured* noise
floor (a second decode with the noise fill off; libavcodec's noise is
independent, so the expected difference is √2 × ours): per channel RMS ≤
max(1 LSB16, 1.5 × √2 × floor), peak ≤ max(8 LSB16, 2.5 × floor peak). Numbers
from 2026-09-13, in 16-bit LSBs (1 LSB16 = 1/32768):

- **30 ffmpeg-made streams** (mono → 5.1, 32 / 44.1 / 48 kHz, AC-3 64–448 kbit/s,
  E-AC-3 48–256 kbit/s, tones / pink / white / brown noise / clicks, plus
  copies with `blksw` forced by `ac3_make_blksw_vector.py`), each with and
  without `dynrng`: the dither-stripped copies (`examples/ac3_strip_dither.rs`
  clears every `dithflag` and re-solves crc1/crc2) agree with libavcodec to
  **≤ 0.03 RMS / ≤ 0.32 peak** — float rounding; the dithered originals sit at
  the floor (RMS / expected 0.9–1.1).
- **Dolby-encoded FATE streams.** `monsters_inc_5.1_448` (AC-3, coupling,
  `dynrng` every block): RMS 0.19–0.29 against a floor of 0.12–0.20 (ratio
  1.04–1.09; dither-stripped 0.03). `matrix2_commentary1_stereo_192` (E-AC-3,
  coupling, `dynrng` in 766/780 blocks): ratio 0.99–1.01.
  `serenity_english_5.1_1536` (E-AC-3, one block per frame): ≤ 0.11 RMS
  absolute. `millers_crossing_4.0` (3/1) and `monsters_inc_2.0_192`: identical
  to libavcodec (dither-stripped 0.01–0.07 RMS) except one block each, below.
- **`csi_miami_5.1_256_spx` / `csi_miami_stereo_128_spx`** (E-AC-3 with spectral
  extension **and** AHT — 206 / 592 AHT channel-frames, VQ and GAQ incl. the
  large-mantissa path — plus `dynrng`): the fbw channels sit at 1.2–1.8 × the
  dither-only expectation. The excess is level-proportional and uncorrelated
  with AHT (AHT channel-frames err 2.2 % of level, non-AHT 1.5 %), so it is the
  SPX noise blend's random sequence (Annex E §3.6.4.2 fixes no distribution),
  not the VQ / GAQ arithmetic. Their LFE: libavcodec noise-fills zero-bit AHT
  bins on the LFE (a frame with every `hebap` 0 at exponent 15 comes out at
  ≈ 0.9 LSB16 = 0.707·2⁻¹⁵ in libavcodec, silent here; §7.3.4 dither is per
  fbw channel). The gate widens for exactly these two cases (SPX streams
  2.5× / 3.5×; the LFE of AHT streams a 2 / 16 LSB16 floor) and names them in
  the report line.
- **Where libavcodec is wrong.** In `millers_crossing_4.0` frame 38 block 5 (C
  block-switched and taken out of coupling) and `monsters_inc_2.0` frame 122
  block 4 (R switched), libavcodec's output for the block fits, to 0.01–0.03
  LSB16 residual, "own head + the *switched* channel's previous tail" on a
  neighbouring channel and "own head + nothing" on the switched one: a
  cross-channel overlap-add that jumps at the block boundary (−2718 → +224
  where ours continues −2718 → −2177) and dies out by the next block. §7.9.4
  step 6 overlap-adds every channel with its own tail, and the following
  block agrees again. Forcing `blksw` on one channel in block 0 of an
  ffmpeg-made stream does not trigger it in libavcodec (both decoders then
  agree to 0.00 and the other channel is unchanged in both), so it is tied
  to the coupling change a real transient brings. `FrameDecoder::mixed_transform_blocks()`
  lists such blocks; the harness masks them and reports the count (one
  block per stream here).
- **Mutation** (run 2026-09-13). `hth[0][47]` 0x0800 → 0x0700 (Table 7.15,
  PDF p.75): `tables::hth_table_7_15` fails (row checksum 50048 ≠ 50304) and
  the fixture cross-check fails in frame 0 with "absolute exponent 25 out of
  0..=24" — the bit allocation hands out different mantissa widths and the
  parse runs into the next channel's exponents.
- **End to end.** `rivet transcode <5.1 AC-3 | E-AC-3 in MP4 / MKV / TS> --audio opus`:
  ffprobe `codec_name=opus channels=6 channel_layout=5.1`, full ffmpeg
  decode error-free, and — because a probe cannot see a permutation —
  [`tests/data/opus_channel_identity.py`](../crates/codec/tests/data/opus_channel_identity.py)
  decodes output and source with ffmpeg and prints the 6×6 correlation
  matrix (the sources carry a distinct tone per channel). That check is
  what found the Opus encoder feeding libopus's family-1 mapping in the
  native order rather than RFC 7845's ([codec-encode.md](codec-encode.md)):
  every 5.1 source but Vorbis had come out with FC/FR swapped and LFE/SL/SR
  rotated while ffprobe reported a perfect 5.1 track. With the fix, all
  seven sources (AC-3 and E-AC-3 in MP4 / MKV / TS, 5.1 Vorbis in MKV) pass:
  every output channel correlates ≥ 0.95 with its own source channel and
  ≤ 0.01 with any other. TS needed the PES
  `private_stream_1` (0xBD) id (ATSC A/53 Part 3 §6.5), which the audio PES
  parser had refused.

Tools: [`examples/ac3_decode.rs`](../crates/codec/examples/ac3_decode.rs) (the
counterpart of `ffmpeg -i x.ac3 -f f32le`; `RUST_LOG=trace` for the syntax
trace, `AC3_DECODE_FRAMES=1` for per-frame tool usage),
[`examples/ac3_strip_dither.rs`](../crates/codec/examples/ac3_strip_dither.rs),
[`tests/data/ac3_make_blksw_vector.py`](../crates/codec/tests/data/ac3_make_blksw_vector.py).

**Why.**
- **Spot tests from the rendered page, not only checksums.** A checksum pins
  a transcription; a spot value re-read from the page pins it to the spec.
  A wrong `hth` entry does not degrade audio, it desynchronises the bit
  allocation and the mantissas parse as garbage from that bin on.
- **A relative gate.** An absolute tolerance either fails conformant
  dithered decodes or lets real bugs through; measuring our own noise
  contribution gives a bound that is tight (float rounding) where the stream
  is deterministic and honest where it is not.
- **The spec wins over the oracle, with the fit as evidence.** Where
  libavcodec differs, the disagreement is localised, reproduced from our own
  internals by least squares, and checked for physical sense (continuity at
  the block boundary) before being masked — and the mask is named in every
  report line so it cannot hide.

---

## AAC decoder

[`AacDecoder`](../crates/codec/src/audio/decode/aac.rs) adapts the decoder of
the `crates/aac` submodule (the rivet-aac repository; provenance in
[decisions.md §26](decisions.md#26-aac-lc-is-encoded-and-decoded-here-from-the-standards)).
Pure Rust, written from ISO/IEC 13818-7 and 14496-3; no other decoder's
source was consulted.

- **Input.** Raw access units under the AudioSpecificConfig the demuxer keeps
  in `AudioTrack::asc` (MP4 `esds`, Matroska CodecPrivate, or the one the TS
  demuxer synthesises from the first ADTS header), or ADTS bytes when there is
  no configuration.
- **What it decodes.** AAC-LC: channel configurations 1–7 and
  program_config_element layouts, long / start / short / stop windows with
  sine and KBD shapes, M/S, intensity stereo, PNS, TNS and pulse data. Output
  is in the native order for the layout (5.1: FL FR FC LFE BL BR;
  configuration 7: FL FR FC LFE BL BR SL SR), which `layout()` names; a PCE
  whose elements do not fit its own position rules comes out in its element
  order with the layout left to the channel count.
- **HE-AAC.** Spectral band replication and parametric stereo are not
  implemented, on purpose (§26): an HE-AAC or HE-AAC v2 stream decodes as its
  AAC-LC core, at half the stream's rate. `decode::aac::probe` reads the
  first access unit to say so (explicit signalling in the configuration, or
  SBR data in the access unit), which the job uses to keep such a track
  undecoded unless it needs its PCM (`he-aac`, [output-spec.md](output-spec.md#3-audio--with_audioaudiocodecpolicy)).
- **Refused by name.** AAC Main, SSR, LTP and the other object types, 960-sample
  frames and coupling channel elements: `AudioError::Unsupported`.
- **Verified** against ffmpeg's decoder as a black box, on its own streams
  and on fdk-aac's, to float rounding (figures in the submodule's README).

## FLAC and ALAC decoders

**What.** [`audio/decode/flac.rs`](../crates/codec/src/audio/decode/flac.rs)
and [`audio/decode/alac.rs`](../crates/codec/src/audio/decode/alac.rs) are
clean-room, pure-Rust lossless decoders, reached through
`audio::create_decoder("flac" | "alac")`. FLAC: every subframe type, wasted
bits, the three stereo modes, 4–32 bits, 1–8 channels, fixed and variable
block sizes, both CRCs, and the STREAMINFO MD5 (a mismatch at the end of the
stream is a warning). ALAC: SCE / CPE / LFE elements, the adaptive predictor
and Rice coder, pair mixing, shifted and escaped elements, 16/20/24/32 bits,
1–8 channels, reordered from ALAC's centre-first layouts to the native order
(four channels reported as 4.0 through `AudioDecoder::layout`). Both have an
exact integer interface (`decode_int`) beside the f32 one. Inputs: MP4
(`fLaC` / `alac`), Matroska (`A_FLAC` / `A_ALAC`) and native `.flac`; ALAC in
CAF is not read. See [lossless-audio.md](lossless-audio.md).

## Other audio decoders

The rest of what `audio::create_decoder` routes to
([`audio/decode/`](../crates/codec/src/audio/decode/mod.rs)); the Opus, MP3
and Vorbis adapters are described with the encoders in
[codec-encode.md](codec-encode.md#the-audio-pipeline-decode--opus--aac--mp3--flac--alac).

- **DTS core** ([`audio/decode/dts/`](../crates/codec/src/audio/decode/dts/mod.rs);
  `"dts" | "dca" | "dtsc"`) — written from ETSI TS 102 114: the core
  substream, up to 5.1 at ≤ 48 kHz. XCh / XXCh / X96 and DTS-HD extension
  substreams are skipped, so a DTS-HD track yields its lossy core. Two
  codebooks ETSI does not print decide the rest: a subband coded with ADPCM
  prediction is refused by name (`DtsError::Unsupported`; most disc-sourced
  DTS predicts somewhere), and high-frequency VQ subbands decode as silence,
  which §5.4.3 allows, with one warning per decoder. Output is in the
  pipeline's native order; the decoder reports its `AMODE` layout. Checked
  against libavcodec on ffmpeg-made vectors to ~1e-6 relative RMS
  ([`tests/dts_core.rs`](../crates/codec/tests/dts_core.rs)).
- **MPEG audio** ([`audio/decode/mp3.rs`](../crates/codec/src/audio/decode/mp3.rs);
  `"mp3" | "mp2" | "mp1" | …`) — minimp3, which reads Layers I and II as well
  as III.
- **Opus** ([`audio/decode/opus.rs`](../crates/codec/src/audio/decode/opus.rs))
  — libopus's multistream decoder for channel-mapping families 0 and 1
  (family 255 is refused by name), always 48 kHz, pre-skip kept for the
  container's edit to remove.
- **Vorbis** ([`audio/decode/vorbis.rs`](../crates/codec/src/audio/decode/vorbis.rs)) — lewton.
- **Linear PCM** ([`audio/decode/pcm.rs`](../crates/codec/src/audio/decode/pcm.rs);
  `pcm_u8`, `pcm_s16le`, `pcm_s24le`, `pcm_s32le`, `pcm_f32le`, `pcm_f64le`) —
  AVI's WAVE formats converted to f32; a partial sample frame at a packet's
  end waits for the next packet.

Which of these a job may run is the caller's choice: rivet's
`audio-decode-deny` names source codecs that must not be decoded (a denied
track is passed through or the job refused, never silently decoded) — see
[output-spec.md](output-spec.md#restricting-decoders--audio_decode_deny).

---

## GPU detection

**What.** [`gpu/`](../crates/codec/src/gpu/mod.rs) enumerates the host's GPUs and
exposes live utilisation. [`detect_gpus()`](../crates/codec/src/gpu/mod.rs#L21)
concatenates per-vendor scans (NVIDIA, then AMD, then Intel) into
`Vec<`[`GpuDevice`](../crates/codec/src/gpu/types.rs#L4)`>` (vendor, name,
index, vendor index, generation, PCI id, VRAM, serial, bus address).
[`detect_gpus_cached()`](../crates/codec/src/gpu/mod.rs#L87) probes once per
process for callers that would otherwise repeat the NVML / WMI / sysfs walk
per encoder, and [`vendor_index_of(global)`](../crates/codec/src/gpu/mod.rs#L101)
translates a global index to the vendor-local one.

**Two indices, because a host can be mixed.** Each per-vendor scan numbers its own
devices from 0, so on an NVIDIA + AMD box both would be index 0. `GpuDevice`
therefore carries **two**: `vendor_index` (the device's position *within its
vendor*, what the per-vendor SDK enumerates — the CUDA ordinal, the QSV/AMF
adapter) and a globally-unique `index` that `detect_gpus` reassigns across the
concatenated list. The global `index` is what the user addresses (`--decode gpu:N`,
the GPU policy) and what `create_decoder_on` matches; it then constructs the
chosen backend with that device's **`vendor_index`** so the hardware selects the
right physical adapter. On a single-vendor host the two coincide.

**Only the AMD / Intel cards this process can open (Linux).** The sysfs walk
sees every card in the machine, but a process reaches a card only through its
DRM render node, and a container is usually handed just some of them (a
Kubernetes device plugin passes through the allocated cards'
`/dev/dri/renderD*`). oneVPL and AMF enumerate only the cards whose render node
they can open, in render-node order. So
[`usable_by_this_process`](../crates/codec/src/gpu/mod.rs#L50) keeps an AMD or
Intel card only if its `renderD` node opens read-write, sorts the survivors by
render node and renumbers `vendor_index` to match; a card whose render node
sysfs does not reveal is kept, in its original order. Before this, the extra
cards gave `vendor_index` values naming the wrong adapter and every session on
them failed (`MFXCreateSession -9`). NVIDIA devices are numbered by CUDA and
are not filtered this way; on Windows nothing is filtered.

- **NVIDIA** via [libcuda dlopen](../crates/codec/src/gpu/nvidia.rs#L17) (`cuInit` +
  `cuDeviceGetCount` + `cuDeviceGetName`), enriched by NVML for VRAM/PCI/serial.
  **Why dlopen, not `nvidia-smi`:** minimal container images often lack the
  `nvidia-smi` binary, but the NVIDIA Container Toolkit bind-mounts the driver's
  user-mode libraries — so probing the library directly works where shelling out
  wouldn't. (NVML init even retries the SONAME-versioned `libnvidia-ml.so.1`
  because the toolkit mounts only that, not the unsuffixed alias.)
- **AMD / Intel** via, on **Linux**, a sysfs PCI scan
  ([`amd.rs`](../crates/codec/src/gpu/amd.rs#L5),
  [`intel.rs`](../crates/codec/src/gpu/intel.rs#L5): `/sys/bus/pci/devices`,
  matching vendor `0x1002` / `0x8086` + a display class);
  on **Windows**, a WMI query (`Get-CimInstance Win32_VideoController`, cached for
  the process) parsing the `PNPDeviceID` `VEN_`/`DEV_` fields. Both feed the same
  device-id → generation/label tables. This is what makes an AMD GPU visible to
  the AMF decode path on a Windows host (without it the sysfs-only scan returned
  empty there, so only the NVML-detected NVIDIA card showed up). WMI reports
  no usable VRAM (`AdapterRAM` is u32-capped), so `vram_mib` is 0 there.

The generation/label tables ([`nvidia_generation_from_name`](../crates/codec/src/gpu/nvidia.rs#L220),
[`intel_label_from_device_id`](../crates/codec/src/gpu/intel.rs#L163), …) are mostly
cosmetic (the admin inventory page). The Intel labeller was added because,
without it, every Intel device was tagged "Integrated GPU" and the AV1
dispatch's old `contains("arc")` name check missed discrete Arc cards; that
check is gone (see `supports_av1_encode` below).

[`GpuUtilizationReader`](../crates/codec/src/gpu/utilization.rs#L9) holds an NVML handle
across reads and returns a per-device [`GpuUtilization`](../crates/codec/src/gpu/types.rs#L73)
snapshot (compute/encoder/decoder busy %, VRAM, temperature) on each load tick.
NVIDIA reads come from NVML; Intel is a coarse sysfs freq-ratio + DRM-fdinfo VRAM
proxy; AMD is a no-op stand-in (radeontop/amdsmi deferred).

### PCI BAR report

[`bar_report(&GpuDevice)`](../crates/codec/src/gpu/bar.rs#L102) says whether a
discrete card's VRAM window — its largest memory BAR — covers its VRAM, i.e.
whether Resizable BAR is in effect. Linux only, read from sysfs without
privileges: the BAR's current size from `/sys/bus/pci/devices/<bdf>/resource`,
whether the card can resize it and to what from `resourceN_resize` (Linux 6.1+;
`resizable: None` on older kernels), and whether the host is a VM (the CPU's
`hypervisor` flag). It returns `None` on other platforms, for an integrated GPU,
or when sysfs does not say. [`BarReport`](../crates/codec/src/gpu/bar.rs#L22)
carries the fields; `verdict()` gives a
[`BarVerdict`](../crates/codec/src/gpu/bar.rs#L42) (`Full` when the window is at
least half the VRAM, `Small`, or `Unknown` with no VRAM figure); `describe()` is
the one-line text with the cause and the firmware fix (Above 4G Decoding +
Resizable BAR, and passing the full BAR through on a VM); `consequence()` says
what a small window costs on an Intel card.

**Why.** Encode and decode do not care, but Intel's compute runtime (OpenCL,
Level Zero, OpenVINO's GPU plugin) will not expose an Arc card behind a small
BAR on the upstream `i915` driver and says only `WARNING: Small BAR detected`,
while QSV carries on — so the cause is otherwise hard to find. `rivet devices`
prints it as the `PCI BAR` line and `--json` carries it as `pci_bar` (see
[cli.md](cli.md#rivet-devices)).

### `supports_av1_encode` admits everything

[`supports_av1_encode`](../crates/codec/src/gpu/mod.rs#L125) returns `true` for every
vendor on purpose. It used to carry a brittle board-name substring list, and a
missed SKU (the RTX 5060, once) would *hard-fail* a job when no software
fallback is built in. The decision was to defer to the **real driver capability query** in the
encoder constructor (`nvEncGetEncodeCaps` / AMF `CreateComponent` / oneVPL
`MFXVideoENCODE_Query`), which authoritatively bails if AV1 silicon is absent — so
detection admits the GPU and lets the real query be the gate.

**Notes / gotchas.** [`manufacturer_label`](../crates/codec/src/gpu/mod.rs#L113)
is the one vendor spelling (`NVIDIA` / `AMD` / `Intel`) that `rivet devices`,
`rivet capabilities` and the GPU policy's messages print; its doc comment
still asks that it stay in lockstep with a `vendor_label` in
`transcoder/src/capabilities.rs`, a file not in this repository. Fields that a platform can't
read (consumer-GeForce serials, older-kernel VRAM) come back empty/`None`/`0`
rather than synthesised, and the literal `"0"` serial is treated as `None` per
NVML's documented sentinel.

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

**What.** [`pixel_format/`](../crates/frame/src/pixel_format/mod.rs) — in the
`rivet-frame` crate, re-exported as `codec::pixel_format` — is a set of pure-Rust
bitstream walkers built on a small
[`BitReader`](../crates/frame/src/pixel_format/bitreader.rs#L5) (Exp-Golomb
`ue`/`se`, AV1 `su`, byte-align; AV1 `uvlc` lives beside the sequence-header
parser). It sits in `rivet-frame` because the demuxers use it and
`rivet-container` does not depend on this crate. Two layers:

1. **Pixel-format detection** — [`detect`](../crates/frame/src/pixel_format/mod.rs#L35)
   dispatches by codec to `detect_h264` / `detect_hevc` / `detect_vp9` /
   `detect_av1`, which parse the first sequence header for
   `chroma_format_idc` + luma bit depth and map them via
   `PixelFormat::from_chroma_and_depth`. Any parse failure falls back to `Yuv420p`
   (matching the previous hard-coded behaviour — a bad probe degrades payload
   accuracy, it doesn't block the transcode).
2. **Dimension + deep parse** — [`detect_dims`](../crates/frame/src/pixel_format/mod.rs#L59)
   dispatches to full SPS/sequence-header walkers
   ([`parse_h264_sps`](../crates/frame/src/pixel_format/h264.rs#L143),
   [`parse_hevc_sps`](../crates/frame/src/pixel_format/hevc.rs#L273),
   [`parse_mpeg2_sequence_header`](../crates/frame/src/pixel_format/mpeg2.rs#L29)) that
   go all the way through scaling lists, `pic_order_cnt_type` branches, and frame
   cropping / conformance windows to compute the **displayable** width/height. It
   returns `None` to mean "keep existing dims."

**Why.** Two distinct needs. First, the pipeline wants a fast, codec-agnostic
*format* probe **before** decoder construction — the decoders do not expose a
"just probe the format" API (NVDEC tells us, but only after decode starts), so
this module fills that gap. Second, **MPEG-TS carries no
container-level dimensions** (no sample-entry atom, no track header — the SPS is
the only source), so `container::ts` calls `detect_dims` during demux to populate
`StreamInfo.width`/`.height`, which would otherwise be `0×0`. That's why
`detect_dims` returning `None` is load-bearing for TS but a harmless no-op for
MP4/MKV (which already have dims).

**The deeper parsers** (`H264SpsInfo` / `H264PpsInfo` / `HevcSpsInfo` /
`H265VpsInfo` / `H265PpsInfo` / `H265SliceHeader` /
[`Av1SequenceHeader`](../crates/frame/src/pixel_format/av1/sequence.rs#L13)
/ [`Av1FrameHeader`](../crates/frame/src/pixel_format/av1/frame.rs#L15), and
[`parse_h264_pps`](../crates/frame/src/pixel_format/h264.rs#L487) /
`parse_h264_slice_header` / `parse_h265_*` / [`parse_av1_frame_header`](../crates/frame/src/pixel_format/av1/frame.rs#L225))
expose far more fields than dimension detection needs — constraint flags, POC
branch predicates, tile info, quantization, loop-filter/CDEF/segmentation. The
struct doc comments say why: these were built to construct the Vulkan Video `Std*`
parameter/picture-info structs for a hand-rolled Vulkan decoder (since removed).
They remain in use: [`Av1SequenceHeader`](../crates/frame/src/pixel_format/av1/sequence.rs#L13)
is the input to the [AV1 codec-string formatter](#hlsdash-codecs-strings),
`H264SpsInfo` / `HevcSpsInfo` feed the H.264 / H.265 ones, and
`parse_h264_sps` / `parse_hevc_sps` give the multi-GPU chunk-stitch its
`H26xInvariant` and the muxer its `avcC` / `hvcC` fields. The module also holds
the VP9 colour reader (`parse_vp9_colour`), the MPEG-2 display-aspect and
colour-description readers, and keyframe / first-slice offset helpers
(`av1_packet_is_keyframe`, `h264_first_slice_nal_offset`, …).

**Notes / gotchas.**
- `detect_av1` walks the full sequence header (through the operating points
  to `color_config`) via `parse_av1_sequence_header`. It used to stop at the
  timing info and answer 8-bit 4:2:0 for every stream, so a 10-bit AV1 source —
  every HDR one — read as 8-bit.
- The AV1 sequence-header parser carries an in-code fix note:
  `initial_display_delay_present_flag` lives **outside** the timing-info branch;
  nesting it (an earlier bug) desynced every following field.
- `remove_h264_rbsp_stuffing` strips emulation-prevention bytes before any SPS
  walk — shared by the H.264 and HEVC paths.

---

## HDR SEI extraction

**What.** [`hdr_sei`](../crates/frame/src/hdr_sei.rs) (in `rivet-frame`;
[`codec::hevc_sei`](../crates/codec/src/hevc_sei.rs) re-exports it at its old
path) scans an Annex-B buffer for SEI NAL units — HEVC prefix/suffix types 39/40,
H.264 type 6 — and extracts two HDR10 payloads, which have the same syntax in
both codecs: **mastering display colour volume** (payload type 137, H.265
D.2.28 / H.264 D.1.29) and **content light level** (type 144, D.2.35 /
D.1.31). [`parse_annexb`](../crates/frame/src/hdr_sei.rs#L56) (HEVC),
[`parse_h264_annexb`](../crates/frame/src/hdr_sei.rs#L61) and
[`parse_annexb_for(codec, …)`](../crates/frame/src/hdr_sei.rs#L68) return an
[`HdrSei`](../crates/frame/src/hdr_sei.rs#L27) (`HevcHdrSei` is its old name, kept
as an alias) that callers fold into `ColorMetadata` (it
[`merge`](../crates/frame/src/hdr_sei.rs#L40)s newest-wins since HDR tooling
repeats the SEI on every IRAP). `frame::hdr_sei::parse_av1_obus` reads the same
two from AV1 metadata OBUs; it is not part of the `codec::hevc_sei` re-export.

**Why a hand-rolled parser.** The GPU decoders don't surface SEI messages
through their APIs, and a container may carry no HDR static metadata at all —
no MP4 `mdcv` / `clli`, no Matroska `MasteringMetadata`, and MPEG-TS or AVI
have nowhere to put it — while the stream carries both messages in its first
IRAP access unit (x265 `hdr10=1` does exactly this). The demuxers
(`container::demux::hdr`) read them from there, once, so the values reach the
encoders' SEIs and the output's `mdcv`/`clli` boxes (without which Apple
devices fall back to BT.709 limited even when `colr nclx` signals BT.2020). It
never touches the decode path.

**Notes / gotchas.** The parser strips emulation-prevention bytes per-NAL
before reading fields, decodes the SEI `payload_type`/`payload_size` 0xFF-run
encoding, and **remaps the spec's GBR wire order to the struct's RGB field order**
(`parse_mastering_display`). `probe.rs` re-exports `parse_annexb` as
`parse_hevc_hdr_sei` for callers that want the scan without constructing a
decoder.

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
[`av1_codec_string`](../crates/codec/src/codec_strings.rs#L60) from a parsed
`Av1SequenceHeader` (`av01.P.LLT.DD.M.CCC.TTT.MMM.F`),
[`avc_codec_string`](../crates/codec/src/codec_strings.rs#L92) (`avc1.PPCCLL`
from the SPS's profile, constraint flags and level) and
[`hevc_codec_string`](../crates/codec/src/codec_strings.rs#L112), each taking the
sample-entry fourcc (`avc1` / `avc3`, `hvc1` / `hev1`), the constant
[`AAC_LC_CODEC_STRING`](../crates/codec/src/codec_strings.rs#L152) (`mp4a.40.2`),
and [`hls_codecs_attribute`](../crates/codec/src/codec_strings.rs#L157) to join
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
  (`push_sample`/`decode_next`) emitting packed planar YUV — `Yuv420p` /
  `Yuv420p10le` from the GPUs, plus 12-bit and 4:2:2 / 4:4:4 from the software
  tiers, which the colorspace layer narrows — so the decode-once pump and
  everything downstream never branch on which GPU decoded.
- **Streaming over eager.** NVDEC (and QSV/AMF) drain per-sample into bounded
  queues to keep peak RSS flat on long inputs.
- **Typed rejects, not opaque strings.** `NvdecError` distinguishes "format we'll
  never support" from "transient hardware limit" so callers can steer policy.
- **One process-wide CUDA-init lock** across NVENC + NVDEC, because the driver
  segfaults on concurrent context creation from different code paths.
- **Probe + bitstream parsers are pure-Rust and decode-free**, so the pipeline can
  answer "what is this?" (format, dims, HDR metadata, codec strings) before
  constructing a decoder — and recover the data (TS dimensions, H.264 / HEVC
  HDR SEI) that no container layer carries. The parsers and SEI scanner live in
  `rivet-frame` so the demuxers can use them.

> **Drift flagged for maintainers:** `create_decoder` wires no `FallbackDecoder`
> despite comments implying a GPU → CPU fallover chain (the fallback that exists
> is `HardwareThenSoftware`). The `decode/mod.rs` and `ffmpeg.rs` module headers,
> and `nvdec/mod.rs`'s "legacy fallback … `FfmpegDecoder` with `hwaccel=cuda`"
> header, describe dispatch orders that no longer hold. The dispatch order is
> stated above and is the authority; those comments are the stale half.
