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
| [`encode/h26x_sw.rs`](../crates/codec/src/encode/h26x_sw.rs) | Software H.264 / H.265 encoders via the workspace's own [`h26x`](../crates/h26x) crate — pure Rust, 4:2:0 at 8 bits (H.265 also 10-bit Main 10), CABAC, constant QP on the shared H.26x anchor table — or, for a rung that names a bitrate, the encoder's own rate controller with an optional coded picture buffer (see [bitrate rungs](#bitrate-rungs-in-the-software-tier-measured)) — `force_keyframe_next` honoured. The output colour (`ColorMetadata`) goes into the SPS VUI and the HDR10 static metadata into SEIs 137 / 144, so HDR10 / HLG output validates on a build with no GPU. Fallback gated on `h26x-fallback`; always constructible by name. |
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
samples (`avc1`/`hvc1`). The box keeps one set per id, in id order: a stream
that codes some pictures with a second PPS (id 1) and re-sends it in their
access units gets both in the box and neither in the samples. A set re-sent
under its id with different contents is warned about by kind and id, and the
first is kept, because the box holds one set per id (inline `avc3`/`hev1`
output keeps every set in-band instead, where a re-sent set legitimately
replaces the old one). rav1e rejects H.264/H.265 rather than silently emit
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
  bitstreams, everything else uses a quality-targeting VBR. The one exception is
  a rung that names a bitrate, which only the software H.264 / H.265 tier codes
  ([bitrate rungs](#bitrate-rungs-in-the-software-tier-measured)); the hardware
  backends refuse one by name.

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

### Opt-in tools in the software tier: `aq` and `wp` (measured; `aq` off, `wp` on by default)

The native H.264 and H.265 encoders have two tools the tuning table decides:
**adaptive quantisation** (`aq=<strength>`, 0.0–4.0 — a quantiser offset per
H.265 CTB or H.264 macroblock from luma variance, flat blocks finer, textured
coarser, zero-mean over the picture) and **weighted prediction** (`wp=on|off` — a
weight and offset per P picture, fitted against the reference and used where it
lowers the residual). They are [`EncodeOverrides`] fields (`aq_strength_tenths`,
`weighted_pred`), spelled in the policy grammar: `--encode-policy "any:wp=off"`,
`"short>=720:aq=1.0"`. `aq` is off at every quality target for both codecs.
`wp` is **on** at every target for both codecs, measured in
[Weighted prediction by default](#weighted-prediction-by-default-measured-both-codecs)
below. The two measurements that follow were taken while both were off by
default. In both, the control arm `any:aq=0,wp=off` was `cmp`-equal to no policy
in every cell. The hardware backends ignore both knobs. The
H.264 encoder gained both in h26x `d1471ce`; before that bump this tier logged
and dropped them for H.264. **Lookahead is not one of them:** it informs a rate
controller, and a constant-QP rung has none to inform — the encoders refuse a
lookahead without a bitrate target (H.264 refuses one outright, uncalibrated),
and the tier logs and ignores `lookahead=` on such a rung rather than inventing a
target. On an H.265 rung that names a bitrate it reaches the encoder; see
[bitrate rungs](#bitrate-rungs-in-the-software-tier-measured).

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
- **`wp` stayed off here; it is now on by default.** On the fade it was about
  5% smaller at the same PSNR at every target (+0.4 to +0.6 dB at equal size).
  Without a fade it cost about 116 bytes per 4 s (the per-P-slice table) with
  PSNR unchanged. It was not made a default then because the evidence was one
  synthetic fade and one inconclusive timing clip. Superseded by
  [Weighted prediction by default](#weighted-prediction-by-default-measured-both-codecs),
  measured on the current encoder over five clips and five quantisers.

#### H.264

These H.264 numbers predate h26x `d88e24a` (h264drift), which quantises the
I_16x16 luma DC one shift coarser and so changes H.264 bytes wherever I_16x16
is chosen. On these four clips at QP 22 to 45, and 10-bit `testsrc2`, that fix
moved encodes with no knob by −7.7% to +1.4% bytes and −0.25 to +0.18 dB at the
same QP. The tables have not been re-measured on it. The decision is kept
because the differences it rests on are larger than that shift: `aq` losing 0.2 to
0.9 dB at equal size, and `wp` saving 10 to 12% on a fade.

**How it was measured** (2026-09-14, h26x `1092d4c`). `rivet transcode` to one
MP4 with `TRANSCODE_ENCODER_BACKEND=h26x` on a build with no GPU encoder, which
takes the single-file path on software leases (`rivet::multigpu::single_file`):
one segment, one chunk, one h26x encoder at 4 threads. `--codec h264 --target
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

**Decision at the time: both off at every target, as for H.265.**

- **`aq` stays off.** It buys size with PSNR at every target and loses at equal
  size on `fade`, `zoom` and `pan` (−0.2 to −0.9 dB at strength 1.0). The few
  non-negative cells are extrapolated or within 0.06 dB. The perceptual case is
  the one made for H.265 above, and nothing here measures it.
- **`wp` stayed off; it is now on by default.** On the fade it was 10–12%
  smaller at the same or better PSNR (+1.3 to +1.8 dB at equal size). Without a
  fade it changed the size by a table per P slice and no pixel. Superseded by
  [Weighted prediction by default](#weighted-prediction-by-default-measured-both-codecs),
  after h264drift made the fade case much stronger.

### Weighted prediction by default (measured, both codecs)

The tuning table turns weighted prediction **on** for both codecs at every
target and tier (`H26xSwParams::weighted_pred`). `--encode-policy "any:wp=off"`
restores the previous default, and the stream is then byte-identical to the
table before this change.

**Why now.** h26x `d88e24a` (h264drift) quantises the H.264 I_16x16 luma DC at
the right shift. On a fade that fix costs 2 to 4 dB on the near-black P frames
after a flat IDR, and no quantiser change won them back (h264tools, measured on
the h26x corpus). Weighted prediction does.

**How it was measured** (2026-09-14, rivet `dabfde8`, h26x `54bdc3a`). The same
path as the tables above: single-file, software leases, one chunk, one h26x
encoder at 4 threads, `TRANSCODE_ENCODER_BACKEND=h26x`. One binary, whose table
still had weighted prediction off, with `wp=on` added through the policy. The
changed table's default was then checked md5-identical to that arm. Clips:
`fade`, `pan`, `testsrc2` and `zoom` at 640x360 as above, plus `testsrc2` at 10
bits. Quantisers: QP 22 / 26 / 32 (`high` / `standard` / `low`), and `--crf 40` and
`--crf 45`. H.264 at `draft`, `standard` and `archive`; H.265 at `draft` and
`standard` (`archive` codes `standard`'s H.265 stream). 330 runs, every output
decoded by ffmpeg with no error output and every frame present. Luma PSNR by
frame index. *At equal size* interpolates along the wp-off arm's five-QP curve
(`*` extrapolated). Time: `standard` tier, QP 26 and 40, three reps, paired.

Weighted prediction on against off, over every tier and quantiser measured:

| codec | clip | size at the same QP | ΔY at the same QP | ΔY at equal size |
|---|---|---:|---:|---:|
| H.264 | fade | −9.8% to −15.7% | −0.08 to +1.36 | +1.19 to +3.79* |
| H.264 | testsrc2, 8- and 10-bit | +236 bytes (+0.03% to +0.13%) | 0.000 on every frame | −0.004 to −0.019 |
| H.264 | zoom | +236 to +2,355 bytes (+0.01% to +0.80%) | −0.02 to 0.00 | −0.001 to −0.054 |
| H.264 | pan | −114 to +1,433 bytes (−0.06% to +0.91%) | 0.00 to +0.09 | +0.011 to −0.233 |
| H.265 | fade | −5.0% to −7.0% | −0.05 to +0.28 | +0.53 to +1.06 |
| H.265 | testsrc2, 8-bit | +118 bytes at QP 22–32 (identical frames); −117 to +366 at 40–45 | −0.01 to 0.00 | −0.002 to −0.023 |
| H.265 | testsrc2, 10-bit | +118 bytes at QP 22–32; −320 to +819 at 40–45 | 0.00 to +0.03 | −0.038 to +0.040* |
| H.265 | zoom | −163 to +582 bytes (±0.05%) | −0.01 to +0.01 | −0.010 to +0.008* |
| H.265 | pan | −253 to +516 bytes (±0.22%) | 0.00 to +0.04 | −0.015 to +0.026* |

On `testsrc2` the H.264 encoder never chooses a weight, so the +236 bytes are the
`pred_weight_table` alone and every frame decodes to the same pixels. On `zoom`
and `pan` at QP 32 and above it does choose weights on some pictures that do not
fade: single frames move by up to −0.28 dB (`zoom`) and +1.23 dB (`pan`), and the
cost at equal size reaches 0.05 dB on `zoom` and 0.23 dB on `pan` at `archive`,
QP 45. That is the encoder's per-picture decision, not the table. The H.265
non-fade rows stay within ±0.04 dB at equal size.

The fade's near-black ends, H.264 at `standard`. Each frame reads PSNR before the
I_16x16 DC fix (h26x `1092d4c`), then after it with weighted prediction off, then
after it with weighted prediction on, then the share of the fix's loss recovered:

| QP | fade-in P frames 1 / 2 / 3 / 4 / 5 | fade-out frames 115 / 116 / 117 / 118 / 119 | summed over frames that lost |
|---|---|---|---|
| 32 | 55.48/51.90/51.90 (0%), 51.90/52.12/53.08, 50.58/50.09/50.83 (151%), 49.52/49.78/49.24, 48.62/48.42/48.62 (98%) | 47.67/48.25/49.61, 49.23/48.60/50.39 (284%), 50.55/51.00/52.24, 52.12/51.34/54.76 (437%), 55.89/52.24/59.11 (188%) | loss 9.33 dB, recovered 13.02 dB (140%); clip −11.17% bytes, +0.34 dB |
| 40 | 50.73/48.51/49.80 (58%), 50.80/46.63/46.95 (8%), 47.76/46.60/46.63 (3%), 46.66/44.85/45.91 (59%), 44.78/45.68/46.36 | 45.30/44.33/47.24 (298%), 45.80/46.37/49.84, 47.90/47.69/49.34, 50.63/50.57/53.30, 53.87/51.40/59.75 (337%) | loss 13.10 dB, recovered 18.37 dB (140%); clip −13.47% bytes, +0.70 dB |
| 45 | 46.31/46.36/46.36, 44.23/43.23/45.76 (254%), 46.29/42.93/42.66 (−8%), 45.52/42.05/44.24 (63%), 43.58/40.94/44.52 (135%) | 43.21/43.76/47.10, 44.51/43.43/47.63 (388%), 44.95/44.30/49.58 (807%), 47.36/45.29/52.32 (339%), 48.94/51.33/56.36 | loss 14.30 dB, recovered 24.56 dB (172%); clip −15.74% bytes, +1.09 dB |

The fade-out end recovers well past the pre-fix encoder. The first fade-in P
frames recover least (QP 40 frames 2 and 3: 8% and 3%; QP 45 frame 3: −8%), as
h264tools found on the h26x corpus. There its recovery read 95 / 94 / 133% at
−11 / −14 / −16% bytes, counted over its own frame selection.

Encode time is not resolved. Median paired CPU ratios, on over off, run from
0.91 to 1.22 (H.264) and 0.84 to 1.14 (H.265) across clips, with no direction
and single reps spanning ±20%.

**Decision: on at every target and tier, for both codecs.**

- **H.264.** On a fade it is 10–16% smaller at the same QP and 1.2–3.8 dB better
  at equal size, and it recovers the near-black frames the I_16x16 DC fix
  exposed. Where no weight is chosen it costs a 236-byte table per 4 s. The
  measured loss is on `pan` and `zoom` at QP 40–45, up to 0.23 dB at equal size,
  where the encoder picks weights it should not. That is an order of magnitude
  below the fade gain, and it is an encoder decision to improve, not a table
  setting.
- **H.265.** On a fade it is 5–7% smaller at the same QP and 0.5–1.1 dB better at
  equal size. Everywhere else it is within ±0.04 dB at equal size, for a
  118-byte table where no weight is chosen.
- `any:wp=off` turns it off for a caller who wants the old streams.

### H.265 coding quadtree depth in the software tier (measured, per speed tier)

The native H.265 encoder can split each coding tree block into smaller coding
units, deciding every split by rate and distortion (`h26x::encode::Config::max_cu_depth`;
the crate's own measurement is in `crates/h26x/src/encode/h265.rs`). The tier
always passes a number from the tuning table (`H26xSwParams::max_cu_depth`), never
`None`. `None` would be the crate's default, so a submodule bump that moved it
would silently change every software H.265 stream. H.264 codes 16x16 macroblocks,
has no quadtree, and its row is 0.

**The CTB size caps the depth.** The encoder chooses a 32x32 or a 16x16 CTB,
whichever pads the coded picture less, 32 on a tie (`Geometry::new` in
`crates/h26x/src/encode/h265_syntax.rs`), and never splits below the 8x8
minimum coding block. So a 16x16 CTB reaches depth 1 at most, and **at a 16x16-CTB
size depth 2 codes a stream identical to depth 1**: at 640x360 the two were
byte-identical in all 21 cells below. The SPS of real streams from this tier
(`ffmpeg -bsf:v trace_headers`; CTB = 8 << `log2_diff_max_min_luma_coding_block_size`,
`log2_min_luma_coding_block_size_minus3` = 0 in every one):

| picture | coded size | `log2_diff_max_min` | CTB | conformance window |
|---|---|---:|---:|---|
| 640x360 | 640x368 | 1 | 16x16 | bottom 4 (chroma units) |
| 854x480 | 864x480 | 2 | 32x32 | right 5 |
| 1000x562 | 1008x576 | 1 | 16x16 | right 4, bottom 7 |
| 1280x720 | 1280x720 | 1 | 16x16 | none |
| 1920x1080 | 1920x1088 | 2 | 32x32 | bottom 4 |
| 3840x2160 | 3840x2160 | 1 | 16x16 | none |

This is the encoder's CTB policy, not the tier's, and a later h26x change to it
(a 32x32 CTB everywhere) will change the streams at the 16x16 sizes.

**How it was measured** (2026-09-14, h26x `1092d4c`). The path is as for the
H.264 tools above: single-file on software leases, one chunk, one h26x encoder
at 4 threads, `TRANSCODE_ENCODER_BACKEND=h26x`. `--codec h265`, one binary with
the table's depth overridden per run (a measurement-only environment variable, not
committed). Arms per cell: depth 0, 1, 2 and a second depth 0 as the control.
The order rotates every rep. The tier is set with `--encode-policy
any:speed=draft|standard`. `archive` codes the same H.265 stream as `standard`,
md5-equal in 6 / 6 checks, because the tier moves only SAO (off at `draft`)
for H.265. Clips as in the H.264 table above (`testsrc2`, `zoom`, `pan`). Every
output decoded by ffmpeg with no error output and all frames present, and every
rep's md5 equal to the first. Size is video packet bytes, luma PSNR by frame
index. Time is the whole `rivet transcode`, as the median of paired ratios over
three reps against depth 0; the control lands at 0.92–1.06 (CPU) and 0.98–1.02
(wall).

640x360 (16x16 CTB), depth 1 (= depth 2) against depth 0 at the same target and
tier:

| clip | tier | target (QP) | size | ΔY PSNR | CPU | wall |
|---|---|---|---:|---:|---:|---:|
| testsrc2 | standard | visually_lossless (18) | −30.3% | +1.67 | 1.85x | 1.61x |
| testsrc2 | standard | high (22) | −28.8% | +1.74 | 1.70x | 1.63x |
| testsrc2 | standard | standard (26) | −25.0% | +1.74 | 1.96x | 1.61x |
| testsrc2 | standard | low (32) | −16.9% | +1.23 | 1.97x | 1.83x |
| testsrc2 | draft | high / standard / low | −29.7% / −25.4% / −19.2% | +2.35 / +2.13 / +1.55 | 2.06–2.34x | 1.53–2.00x |
| zoom | standard | visually_lossless (18) | −9.5% | +0.37 | 3.00x | 2.89x |
| zoom | standard | high (22) | −9.0% | +0.45 | 2.92x | 2.50x |
| zoom | standard | standard (26) | −5.2% | +0.46 | 2.58x | 2.36x |
| zoom | standard | low (32) | −0.6% | +0.31 | 2.62x | 2.20x |
| zoom | draft | high / standard / low | −8.8% / −5.2% / −0.4% | +0.55 / +0.55 / +0.40 | 3.18–3.46x | 2.48–3.09x |
| pan | standard | visually_lossless (18) | −1.8% | +0.21 | 3.30x | 3.27x |
| pan | standard | high (22) | −11.2% | +0.05 | 3.12x | 2.87x |
| pan | standard | standard (26) | −14.5% | +0.06 | 2.90x | 2.67x |
| pan | standard | low (32) | −12.0% | +0.13 | 2.65x | 2.12x |
| pan | draft | high / standard / low | −12.2% / −14.7% / −10.8% | +0.10 / +0.11 / +0.18 | 2.75–3.71x | 2.43–3.06x |

Over the nine `draft` cells: −14.0% bytes, +0.88 dB, CPU 3.18x (2.06–3.71). Over
the twelve `standard` cells: −13.7% bytes, +0.70 dB, CPU 2.63x (1.70–3.30).
Smaller and better at the same QP in every cell.

1920x1080 (32x32 CTB) and 1280x720 (16x16 CTB): `testsrc2` and `zoom` at 60
frames, the `standard` and `draft` tiers, the `high` and `standard` targets,
three reps. Depth 1 and depth 2 against depth 0:

| clip | tier | target (QP) | depth 1: size, ΔY, CPU, wall | depth 2: size, ΔY, CPU, wall |
|---|---|---|---|---|
| testsrc2 1080p | standard | high (22) | −20.9%, +1.30, 1.34x, 1.50x | −33.4%, +2.68, 1.95x, 2.02x |
| testsrc2 1080p | standard | standard (26) | −18.5%, +1.27, 1.49x, 1.49x | −27.2%, +2.64, 2.07x, 1.98x |
| zoom 1080p | standard | high (22) | −15.8%, +0.52, 1.92x, 2.04x | −22.5%, +0.94, 4.07x, 4.26x |
| zoom 1080p | standard | standard (26) | −13.7%, +0.62, 1.95x, 1.99x | −17.2%, +1.04, 3.69x, 3.86x |
| testsrc2 1080p | draft | high (22) | −20.7%, +1.51, 1.90x, 1.90x | −33.4%, +3.24, 2.84x, 2.76x |
| testsrc2 1080p | draft | standard (26) | −17.5%, +1.58, 1.66x, 1.58x | −27.7%, +3.12, 2.51x, 2.32x |
| zoom 1080p | draft | high (22) | −16.2%, +0.72, 2.69x, 2.76x | −23.2%, +1.18, 6.33x, 6.59x |
| zoom 1080p | draft | standard (26) | −14.5%, +0.76, 2.46x, 2.39x | −18.1%, +1.24, 5.55x, 5.57x |
| testsrc2 720p | standard | high / standard | −19.8% / −14.4%, +1.47 / +1.42, 1.64x / 1.87x | identical to depth 1 |
| zoom 720p | standard | high / standard | −8.3% / −3.9%, +0.45 / +0.44, 2.40x / 2.43x | identical to depth 1 |
| testsrc2 720p | draft | high / standard | −20.1% / −16.3%, +1.91 / +1.67, 2.28x / 2.29x | identical to depth 1 |
| zoom 720p | draft | high / standard | −8.1% / −3.8%, +0.54 / +0.54, 3.31x / 2.71x | identical to depth 1 |

At 1080p, over the four cells of each tier: depth 1 is −17.2% bytes and +0.93 dB at
CPU 1.70x (`standard`), −17.3% and +1.14 dB at 2.18x (`draft`); depth 2 is
−25.1% and +1.82 dB at 2.88x (1.95–4.07, `standard`), −25.6% and +2.20 dB at
4.19x (2.51–6.33, `draft`). The depth-0 control arm lands at 0.96–1.07 CPU.
Depth 2 is smaller **and** better than depth 1 at the same QP in all eight 1080p
cells.

**Decision: `max_cu_depth` 2 at `standard` and `archive`, 1 at `draft`, the
same at every target; H.264 0.**

- **`standard` / `archive`: 2.** Where the CTB is 32x32, depth 2 buys a quarter
  of the bytes and nearly 2 dB over one unit per CTB, and dominates depth 1 at
  the same QP. The step from depth 1 costs about 1.7x depth 1's CPU. This is
  the software tier, already the slowest path in the dispatch order. A caller
  here has chosen quality per byte over time, and a quarter fewer bytes at a
  better PSNR is more than any other tool in the table buys. At a 16x16-CTB size
  it costs and codes exactly what depth 1 does.
- **`draft`: 1.** `draft` is the tier that pays least for search (SAO is off
  there). Depth 2 costs 4.2x the CPU of depth 0 at 1080p, and 6.3x on `zoom`.
  Depth 1 costs 2.2x and keeps two thirds of the byte saving (−17.3% of
  −25.6%) and half the PSNR. It is not 0: even at `draft`, depth 1 is smaller
  and better at the same QP in every cell measured at all three sizes.
- **Per tier, not per target.** The gain holds at every target measured: at
  1080p at `high` and `standard` alike, and at 640x360 from `visually_lossless`
  to `low`, smallest at `low` on `zoom` (−0.4% to −0.6%, still +0.3 to +0.4 dB).
  No target has depth 0 winning at the same QP, so nothing separates the targets.
- **Not tuned around the CTB cap.** When the encoder codes 32x32 CTBs at every
  size, `standard` and `archive` get depth 2 at 640x360, 1280x720 and 3840x2160
  too, and their cost there should be re-measured.

**Per rung, by name.** The policy grammar's `cu_depth=` key (`0`, `1` or `2`)
replaces the tier's depth for the rungs it selects. For example,
`--encode-policy "any:speed=draft;short>=1080:cu_depth=2"` encodes every rung at
`draft` and gives the rungs with a short side of 1080 or more depth 2, and
`"any:cu_depth=0"` restores one unit per CTB. `3` and up do not parse. An H.264
rung that names a depth above 0 is refused by name ("cu_depth=1 names an H.265
coding quadtree depth; the native H.264 encoder … has no quadtree"), and
`cu_depth=0` on H.264 is its own row. The hardware backends ignore it.

Against the tier before this table, which coded one unit per CTB, the bytes of
every software H.265 stream change, since no row keeps depth 0. The H.264 rows
do not change.

The measurement is on h26x `1092d4c`. At `675f8b2` the H.265 streams with
weighted prediction off are byte-identical to it: 10 of 10 md5s over `testsrc2`,
`zoom`, `pan` and `fade` at `standard` and `draft`, `fade` with `bframes=2`, and
10-bit. So the tables above still describe the tier. Weighted bi-prediction
(`wp=on` with B pictures) does move: `fade` at `bframes=2` is −0.95% bytes and
−0.04 dB.

### Bitrate rungs in the software tier (measured)

A rung can be coded to a **rate** instead of a quality target:
`EncodeOverrides::bitrate` and `buffer_ms`. On the surfaces these are
`--rung 1280x720@3M`, `--video-bitrate`, `--video-buffer`, `bitrate=` /
`buffer=` in `--encode-policy`, and the same keys in the API, the manifest and
the IPC header. [output-spec.md](output-spec.md) has the knobs and their
precedence.

The software H.264 / H.265 tier (`h26x_sw`) then builds the encoder with:
- `RateControl::Bitrate`: the h26x rate controller picks a quantiser per
  picture to spend the rate, and the target and `q=` delta are not consulted;
- the rung's buffer as `cpb_ms`: the stream carries the HRD (VUI HRD
  parameters and a buffering period per keyframe), and the controller keeps
  every picture inside it;
- for H.265, the rung's `lookahead=`.

A rung without a rate is the constant-QP encode it always was, byte for byte.

**Defaults:** a one-second buffer and no lookahead. Both are measured below.

Every encoder is a stream of its own, and its controller starts from
nothing: the whole file on the serial path, one chunk after its lead-in on
the chunked path, one segment on the HLS ladder.

The rate the encoder is handed is scaled by `fps / frame_rate`. h26x's
`Config::fps` is a whole number and its controller budgets `bps / fps` per
picture, so without the scaling a 29.97 fps rung would spend 0.1 % under its
target and a 12.5 fps rung 4 % under. This is a workaround until `Config`
takes a rational frame rate.

**Refused, by name, before a frame is decoded:**
- a rate beside a CRF;
- a rate under `--seam-mode constqp` (single file);
- a buffer without a rate;
- a rate on AV1;
- a bitrate job whose encode pool is GPUs: only this tier codes to a rate.
  NVENC, AMF, QSV and rav1e each refuse a rate at construction too.

#### How it was measured

- **Binary:** release-fast `h26x-fallback` build of the branch, software
  pool. The tables below were taken at h26x `54bdc3a`. The same cells were
  run again at `cb3ef0c` (rivet 2b5c4ee), which derives each stream's level
  and changed H.265 coding; see [after the h26x bump](#after-the-h26x-bump-to-cb3ef0c).
- **Clips:**
  - `trailer`: a 48 s, 24 fps cinema trailer at 1280x720. It opens on black,
    fades a logo in, and cuts between scenes of very different complexity.
  - `stock`: 25 s of 29.97 fps natural footage at 1280x720.
  - `testsrc2`: 30 s, 30 fps.
  - `grain`: 20 s of `testsrc2` under heavy temporal noise.
- **Ladders and files:**
  - HLS: two rungs, 1280x720 and 640x360, 4 s segments.
  - Serial single file: 1280x720, `--encode single`.
  - Chunked single file: the software pool's eight slots.
- **Targets:** 0.5x, 1x and 2x (H.265: 0.5x and 1x) of each clip's own rate
  at the `standard` constant QP (26) on the same rung. The CQP curve is QP
  20..32 on the same ladder.
- **Rate:** video payload bytes over duration, per rung and per segment.
- **Buffer:** h26x's `h26xhrd`, which reads everything from the stream, on
  every buffered segment on its own (init + segment as Annex B) and on every
  buffered file.
- **Quality:**
  - Global Y PSNR (the mean MSE over the clip, not a mean of per-frame dB:
    the trailer's black frames score 100 dB).
  - Measured against the source at 720p, and against rivet's own scaled
    picture at 360p. That is a QP 0 encode of the rung, because ffmpeg's
    scaler is not rivet's and scoring against it gives a flat ~30 dB.
  - "ΔY at equal rate" is against the CQP curve at the achieved rate, linear
    in log rate; `*` marks an extrapolation.

#### Results

**Rate and buffer, HLS, 1x, one-second buffer:**

| clip | codec | rung | achieved / target | 4 s segments | HRD | peak segment | ΔY at equal rate |
|---|---|---|---:|---:|---:|---:|---:|
| stock | H.264 | 720p / 360p | 1.002 / 1.001 | 0.999–1.004 | 14/14 | 1.004 / 1.006 | +0.13 / +0.19 |
| stock | H.265 | 720p / 360p | 1.008 / 1.005 | 1.000–1.012 | 14/14 | 1.013 / 1.053 | +0.55 / +0.37 |
| testsrc2 | H.264 | 720p / 360p | 0.997 / 0.999 | 0.997–1.000 | 16/16 | 1.000 / 1.000 | −0.11 / −0.20 |
| testsrc2 | H.265 | 720p / 360p | 0.998 / 0.998 | 0.997–0.999 | 16/16 | 1.001 / 1.001 | −0.06 / −0.07 |
| grain | H.264 | 720p / 360p | 0.999 / 1.000 | 0.998–1.002 | 10/10 | 1.000 / 1.002 | −0.03 / +0.00 |
| trailer | H.264 | 720p / 360p | 0.920 / 0.923 | 0.046–1.011 | 24/24 | 1.003 / 1.011 | −0.63 / −0.38 |
| trailer | H.265 | 720p / 360p | 0.923 / 0.927 | 0.056–1.021 | 24/24 | 1.009 / 1.021 | −0.35 / −0.12 |
| trailer 10-bit | H.264 | 720p / 360p | 0.920 / 0.924 | 0.048–1.010 | 24/24 | 1.004 / 1.010 | — |
| trailer 10-bit | H.265 | 720p / 360p | 0.925 / 0.928 | 0.071–1.014 | 24/24 | 1.012 / 1.014 | — |

**Every buffered output kept to its buffer.** That is 578 of 578 HLS
segments (every clip, target, codec, depth, buffer, segment length and
lookahead above) and 8 of 8 single files. The master playlist's BANDWIDTH
is the measured peak segment rate plus the audio rendition's. On the uniform
clips the peak 4 s segment is within 1.4 % of the target. A short final
segment may spend up to the rate plus its buffer, and on stock H.265 360p the
one-second last segment set the peak: 5 % over at 1x, 34 % at 0.5x.

**Targets away from the constant-QP rate.** At 0.5x and 2x the 720p
segments still land within 0.996–1.014 of target on stock, testsrc2 and
grain. At the
same target the 1 s buffer and no buffer differ by a median 0.01 dB across
all HLS cells, 0.15 dB at the worst: H.264 trailer at 0.5x, 360p.

**What a segment cannot do is borrow.** The trailer's HLS rungs spend
0.90–0.93 of their target:
- Its opening segments are black and a logo fade, which even quantiser 0
  codes in a few percent of the rate (the lowest segment is 0.02–0.11 of
  target).
- An encoder per segment cannot carry that surplus into the next one.
- No segment goes over, so the rung comes in under.

The price, against the constant-QP curve at the rate actually spent:
- −0.1 to −0.6 dB on the trailer;
- −0.8 to +1.0 dB on the uniform clips (grain at 2x the worst, stock H.265
  at 0.5x the best).

A rate controller spends evenly; a constant quantiser spends where the
picture needs it.

**Single files: the buffer is what bounds the peak** (720p, 1x):

| clip | codec | file | buffer | achieved | peak 4 s window | ΔY at equal rate |
|---|---|---|---|---:|---:|---:|
| trailer | H.264 | serial | none | 0.997 | 2.054 | −3.00 |
| trailer | H.264 | serial | 1 s | 0.923 | 1.221 | −2.03 |
| trailer | H.264 | chunked | 1 s | 0.929 | 1.227 | −1.16 |
| trailer | H.265 | serial | none | 1.000 | 2.157 | −2.55 |
| trailer | H.265 | serial | 1 s | 0.938 | 1.238 | −1.82 |
| trailer | H.265 | chunked | 1 s | 0.940 | 1.238 | −1.09 |
| stock | H.264 / H.265 | serial | none | 0.997 / 1.000 | 1.006 / 1.021 | −0.80 / −0.52 |
| stock | H.264 / H.265 | serial | 1 s | 0.995 / 1.000 | 1.005 / 1.020 | −0.81 / −0.50 |

**Without a buffer, one controller over a whole uneven file bursts and then
starves.**
- It cannot spend the rate on the black opening, then spends the surplus on
  the scenes after it: the 8–12 s window runs at 1.7–2.1x the rate.
- It then under-spends the complex scenes at 14–20 s, where Y PSNR drops to
  33–38 dB against 44 dB for the constant QP.
- The buffer caps the burst, at the cost of the opening's unspent bits
  (0.92–0.94 of target). It lifts quality 0.34 dB (H.265) and 0.55 dB
  (H.264), and holds the peak at 1.22–1.24x.
- On uniform content the buffer changes nothing (±0.02 dB).

**Why a one-second buffer by default:**
- A rate with no buffer promises nothing about peaks, and a peak is what an
  HLS BANDWIDTH declares.
- The promise held on every segment and file measured.
- It costs a median 0.01 dB on the ladder.
- It halves a single file's worst window.
- A 500 ms buffer measured the same as 1 s on the ladder (trailer 1x: H.264
  0.915 of target, −0.61; H.265 identical to 1 s).
- `--video-buffer 0` declares none.

**Keyframes, not the cold start, are where a bitrate rung loses.** Y PSNR
of the first second after every IDR against the rest (720p, 1x, 1 s buffer):

| clip | codec | constant QP 26 | HLS (IDR at each 4 s segment, fresh encoder) | serial (IDR every 2 s, warm encoder) |
|---|---|---:|---:|---:|
| stock | H.264 | +0.38 | −1.19 | −2.17 |
| stock | H.265 | +0.16 | −0.82 | −2.12 |
| trailer | H.264 | +0.47 | −0.44 | −2.49 |
| trailer | H.265 | +0.15 | −0.55 | −2.15 |
| testsrc2 | H.264 | +0.22 | +0.39 | −0.41 |

- **The controller under-spends keyframes.** A warm IDR gets 1.2–4.5x a P
  picture's bits, against 5–12x at a constant QP on the natural clips. The whole GOP then
  predicts from a soft reference.
- **A fresh encoder's first IDR dips less than a warm one's.** Every HLS
  segment's encoder starts cold, so a lead-in to warm it (as the chunked
  path has) would make the segment start worse, not better, and is not
  built.
- **The fix belongs to the controller's keyframe allocation in h26x**,
  which is being worked on there.

**H.265 lookahead is off by default** because at this h26x it makes the
keyframe starvation worse:

| clip | lookahead | IDR / P bits | first second after IDR vs rest | ΔY at equal rate |
|---|---:|---:|---:|---:|
| stock | 0 | 9.2 | −0.82 | +0.55 |
| stock | 8 / 16 | 1.3 / 1.3 | −4.69 / −4.71 | −1.11 / −1.11 |
| trailer | 0 | 5.5 | −0.55 | −0.35 |
| trailer | 8 / 16 | 0.6 / 0.6 | −2.29 / −2.29 | −0.79 / −0.79 |

A named `lookahead=` still reaches an H.265 bitrate rung. H.264 has no
calibrated lookahead and logs and ignores one.

**CPU.** Serial single file at 720p, median of three interleaved runs, in
CPU seconds; other load on the host spreads single runs by about ±10 %:

| | constant QP 26 | rate, no buffer | rate, 1 s buffer | 1 s buffer + lookahead 8 / 16 |
|---|---:|---:|---:|---:|
| H.264 stock, 2.5 Mbit/s | 26.9 | 27.6 | 26.2 | — |
| H.264 trailer, 1.6 Mbit/s | 43.3 | 45.2 | 57.0 | — |
| H.265 stock, 2.0 Mbit/s | 129.6 | 122.0 | 134.2 | 117.1 / 117.5 |

- Rate control itself costs nothing measurable.
- A buffer costs where pictures overflow it: the encoder codes such a
  picture again, up to three attempts. That is +32 % on the trailer's
  H.264, and within the noise on uniform content.

**Not in the software tier's hands:**
- A rate controller that spends evenly will lose to a constant quantiser on
  uneven content at equal bytes. A capped-quality mode (a constant quantiser
  held under a peak rate) is what an uneven HLS ladder would want, and h26x
  has none.
- The keyframe allocation above.
- The whole-number frame rate (hence the scaling).

#### After the h26x bump to `cb3ef0c`

The same cells at the same targets, each scored against its own
constant-QP curve. The changes at `cb3ef0c`:
- every stream now claims the level (and, for H.265, the tier) its rate and
  buffer need;
- H.265 codes 32x32 CTBs with partial edge CTBs;
- the H.265 reference-set fix.

H.264 came out identical in every cell (rate, segments, PSNR).

H.265 at 720p, 1x, before → after:

| clip | file | buffer | achieved | ΔY at equal rate | global Y PSNR |
|---|---|---|---:|---:|---:|
| stock | HLS | 1 s | 1.008 → 1.005 | +0.55 → +0.44 | 40.80 → 41.03 |
| stock | serial | 1 s | 1.000 → 1.000 | −0.50 → −0.50 | 39.71 → 40.07 |
| trailer | HLS | 1 s | 0.923 → 0.923 | −0.35 → −0.26 | 41.61 → 42.58 |
| trailer | serial | none | 1.000 → 1.000 | −2.55 → −1.43 | 39.91 → 41.72 |
| trailer | serial | 1 s | 0.938 → 0.937 | −1.82 → −1.84 | 40.24 → 41.06 |
| trailer | chunked | 1 s | 0.940 → 0.940 | −1.09 → −1.05 | 40.99 → 41.86 |
| stock, lookahead 8 | HLS | 1 s | 1.011 → 1.010 | −1.11 → −0.98 | 39.15 → 39.63 |
| trailer, lookahead 8 | HLS | 1 s | 0.922 → 0.921 | −0.79 → −0.73 | 41.17 → 42.10 |

**What moved and what did not:**
- H.265 is 0.2–1.8 dB better at the same rate in the 1x cells, and 3.4 dB
  better on the trailer at 0.5x. That is about as much as its constant-QP
  curve moved, so the gap to constant QP mostly holds. The exception is the
  unbuffered serial trailer, where the gap narrows from −2.55 to −1.43 dB.
- Every buffered output still conforms: 596 of 596 segments and files. That
  now includes a 19 Mbit/s H.265 rung on the default buffer, which the old
  fixed Level 4.0 made the tier refuse.
- The keyframe dips barely moved. The first second after an IDR against the
  rest:
  - HLS: stock −0.82 → −0.65, trailer −0.55 → −0.58;
  - serial: stock −2.12 → −1.90, trailer −2.15 → −2.54.
- Lookahead 8 still starves the H.265 keyframe (0.5–1.2x a P picture's
  bits) and still loses 0.3–1.4 dB against none, so it stays off.

**The pending keyframe-seed work in h26x (rcfix4) would move:**
- the first-second-after-IDR dip on every HLS segment, whose opening IDR
  is planned from the seed;
- the lookahead rows, whose keyframes are the ones it under-spends;
- the short final segment's peak, which one IDR dominates;
- the lookahead default, which should be measured again once it lands.

The rate, the segment range and the HRD rows should not move: the buffer
and the per-segment budget bound them.

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
