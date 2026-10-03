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
   [README's "Choosing the output codec"](../README.md#choosing-the-output-codec)).
   **H.264 / H.265** are available for legacy-player compatibility — they carry
   the patent-licensing obligations AV1 was chosen to avoid. The codec is
   selected per job (`OutputSpec::with_video_codec(VideoCodecPolicy::H264)` /
   `--codec h264` / `codec=h264`; values `av1|h264|h265`). See
   [Output codecs](#output-codecs-av1--h264--h265) below. Every other codec
   rivet decodes can be written too — VP9, VP8, MPEG-2, MPEG-4 Part 2 and
   ProRes, by the workspace's own clean-room encoders, in software, in every
   build: see [The other output codecs](#the-other-output-codecs-vp9-vp8-mpeg-2-mpeg-4-part-2-prores).
2. **Hardware encoders are layered, not consolidated.** Each vendor gets a
   hand-rolled, in-tree `dlopen` FFI encoder (NVENC / AMF / QSV). They *stack*;
   software — the workspace's own `av1` encoder for AV1 (`av1-sw-fallback`),
   its own `h26x` encoders for H.264 / H.265 (`h26x-fallback`) — is the last
   resort, and it is
   opt-in. New
   tiers add to the chain, they don't replace it. (The Vulkan Video encode tier
   was removed 2026-05-08, and the FFmpeg tier 2026-08-12 — see
   [select_encoder](#the-encode-dispatch--capability-query) and [No
   FFmpeg](../README.md#no-ffmpeg).)
3. **HDR is tonemapped to SDR by policy.** The default single-output policy maps
   every HDR source down to 8-bit BT.709 at transcode time so a clip never lands
   eye-searingly bright on a viewer's screen. HDR output (`--color
   passthrough|hdr10|hlg`) is opt-in, not the default. See
   [Tonemapping](#tonemapping--the-single-output-policy).

---

## Module map

| File | Purpose |
|------|---------|
| [`encode/mod.rs`](../crates/codec/src/encode/mod.rs) | The `Encoder` trait, `EncoderConfig`, `select_encoder` dispatch, `OutputCaps` runtime capability query, the rate-request checks (`refuse_rate`, `constant_rate_request`), the `TRANSCODE_ENCODER_BACKEND` override's `create_backend`. |
| [`encode/tuning/`](../crates/codec/src/encode/tuning/) | The calibration layer. `QualityTarget` / `SpeedTier` and the libaom anchors in `mod.rs`; per-encoder knobs (CQ, q-index, ICQ, presets, tile grid) in `adapters.rs` / `params.rs`; the per-rung override vocabulary (`EncodeOverrides`, `RungPolicy`) in `overrides.rs` and its text grammar in `policy_grammar.rs`; average vs constant rate (`RateMode`, `default_cbr_bitrate`) in `rate.rs`. |
| [`encode/nvenc/`](../crates/codec/src/encode/nvenc/mod.rs) + [`nvenc_stub.rs`](../crates/codec/src/encode/nvenc_stub.rs) | NVENC encoder — AV1 (Ada+), H.264, H.265 — hand-rolled `nvEncodeAPI` FFI (`ffi.rs`, `buffers.rs`, `constants.rs`, `session.rs`, `upload.rs`). Stub when `nvidia` is off. |
| [`encode/amf/`](../crates/codec/src/encode/amf/mod.rs) + [`amf_stub.rs`](../crates/codec/src/encode/amf_stub.rs) | AMF encoders: H.264 (`VCE_AVC`) and H.265 (`HW_HEVC`, Main / Main 10) on every AMF-capable AMD GPU, AV1 (`HW_AV1`) on RDNA3+. One session flow (`mod.rs`) and a property sequence per codec (`av1.rs`, `h26x.rs`); the vtables mirrored slot-for-slot from the SDK v1.4.36 C headers are [`amf_ffi.rs`](../crates/codec/src/amf_ffi.rs) and the runtime / context lifecycle [`amf_runtime.rs`](../crates/codec/src/amf_runtime.rs), both shared with the AMF decoder. Stub when `amd` is off. |
| [`encode/qsv/`](../crates/codec/src/encode/qsv/mod.rs) + [`qsv_stub.rs`](../crates/codec/src/encode/qsv_stub.rs) | QSV encoder — AV1 (Intel Arc / Meteor Lake+), H.264, H.265 — hand-rolled oneVPL FFI (`ffi.rs`, `config.rs`, `session.rs`, `surface.rs`; the shared `mfx*` structs in `crate::qsv_ffi`). Stub when `qsv` is off. |
| [`encode/av1_sw.rs`](../crates/codec/src/encode/av1_sw.rs) | Software AV1 encoder on the workspace's own [`av1`](../crates/av1/README.md) crate (backend `av1`, `EncoderBackend::Av1`) — pure Rust, profile 0, 8- and 10-bit 4:2:0, one tile (up to 4096 wide); a quality target (4 × libaom's cq-level, `tuning::av1_sw_params`) or an average bitrate; no colour description in the sequence header, so SDR only. Always compiled; `av1-sw-fallback` gates only whether the chain falls back to it. See [Software AV1](#software-av1--encodeav1_swrs). |
| [`encode/h26x_sw.rs`](../crates/codec/src/encode/h26x_sw.rs) | Software H.264 / H.265 encoders via the workspace's own [`h26x`](../crates/h26x) crate — pure Rust, 4:2:0 at 8 and 10 bits (H.264 High / High 10, H.265 Main / Main 10), CABAC, constant QP on the shared H.26x anchor table — or, for a rung that names a bitrate, the encoder's own rate controller with an optional coded picture buffer (see [bitrate rungs](#bitrate-rungs-in-the-software-tier-measured)), or a constant rate with `cbr_flag` and filler data — `force_keyframe_next` honoured. The output colour (`ColorMetadata`) goes into the SPS VUI and the HDR10 static metadata into SEIs 137 / 144, so HDR10 / HLG output validates on a build with no GPU. Always compiled; fallback gated on `h26x-fallback`; always constructible by name. |
| [`encode/vp9_sw.rs`](../crates/codec/src/encode/vp9_sw.rs), [`vp8_sw.rs`](../crates/codec/src/encode/vp8_sw.rs), [`mpeg2_sw.rs`](../crates/codec/src/encode/mpeg2_sw.rs), [`mpeg4_sw.rs`](../crates/codec/src/encode/mpeg4_sw.rs), [`prores_sw.rs`](../crates/codec/src/encode/prores_sw.rs) + [`native.rs`](../crates/codec/src/encode/native.rs) | The workspace's own VP9 / VP8 / MPEG-2 / MPEG-4 Part 2 / ProRes encoders (`crates/{vp9,vp8,mpeg2,mpeg4,prores}`) behind `Encoder`, and what they share (frame checks, the frame rate as a ratio, the quantiser a rung asks for, reference-first picture timestamps). The only encoders of their codecs: built directly, in every build. See [The other output codecs](#the-other-output-codecs-vp9-vp8-mpeg-2-mpeg-4-part-2-prores). |
| [`colorspace/`](../crates/codec/src/colorspace/mod.rs) | Frame normalization: chroma-layout convert (`chroma_convert.rs`), BT.601→709 matrix (`bt601_to_709*.rs`), 4:4:4→4:2:0 downsample (`downsample_444.rs`, `downsample_fir.rs`), bit-depth narrowing / widening (`depth.rs`), bilinear scaling and `scale_region` crop / resize / pad (`scale.rs`), SDR placed in an HDR signal (`sdr_in_hdr.rs`) — scalar + AVX2 runtime dispatch. |
| [`tonemap.rs`](../crates/codec/src/tonemap.rs) | HDR→SDR tonemap: PQ/HLG inverse EOTF → BT.2020→709 gamut → Hable filmic curve → 8-bit BT.709. |
| [`audio/mod.rs`](../crates/codec/src/audio/mod.rs) | Audio framework: traits, wire types, `create_decoder` / `create_encoder`, the MP3 output parameters. |
| [`audio/decode/`](../crates/codec/src/audio/decode/mod.rs) | Decoders → interleaved f32 PCM, adapters onto the workspace's codec crates: MP3 / MP2 / MP1 (`crates/mp3`), Vorbis (`crates/vorbis`), Opus (`crates/opus`), AC-3 / E-AC-3 and DTS (`crates/ac3`, `crates/dts`), AAC / HE-AAC (`crates/aac`), FLAC and ALAC (`crates/lossless`), linear PCM. [codec-decode.md](codec-decode.md) describes the AC-3 / E-AC-3, AAC, FLAC and ALAC decoders. |
| [`audio/encode/`](../crates/codec/src/audio/encode/mod.rs) | Encoders, adapters onto the same crates: Opus (`opus.rs`), MP3 (`mp3.rs`), Vorbis (`vorbis.rs`), AAC-LC / HE-AAC / HE-AAC v2 (`aac.rs`), AC-3 / E-AC-3 (`ac3.rs`), DTS (`dts.rs`), FLAC (`flac.rs`) and ALAC (`alac.rs`). |
| [`audio/remix.rs`](../crates/codec/src/audio/remix.rs) | Layout-to-layout downmix matrices (ITU-R BS.775) and the layout each codec carries a source in. |
| [`audio/resample.rs`](../crates/codec/src/audio/resample.rs) | Sample-rate conversion (rubato sinc), and `AlignedResampler`, which an encoder puts in front of itself: delay trimmed, length exact — e.g. 44.1 kHz → 48 kHz Opus. |

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
| **av1** (software, in-tree) | ✅ 8- and 10-bit SDR (profile 0) | ❌ rejected — an AV1 encoder |
| **h26x** (software, in-tree) | ❌ rejected — H.264 / H.265 only | ✅ 8- and 10-bit — H.265 Main / Main 10, H.264 High / High 10 (the only 10-bit H.264 here); the crate's own encoders, every stream gated SELF (our decoder reproduces the encoder's reconstruction) + CROSS (libavcodec agrees) |

H.264/H.265 encoders emit **Annex-B** NAL; the muxer's
[`nal_mux`](../crates/container/src/nal_mux.rs) splits each packet into per-frame
access units (HW encoders pack several frames per buffer), captures SPS/PPS(/VPS)
for the `avcC`/`hvcC` config box, and repackages slices as length-prefixed
samples (`avc1`/`hvc1`). The box keeps one set per id, in id order: a stream
that codes some pictures with a second PPS (id 1) and re-sends it in their
access units gets both in the box and neither in the samples. A set re-sent
under its id with different contents is warned about by kind and id; the box
keeps the first (it holds one set per id), and from that access unit on every
set travels in band under the `avc3`/`hev1` entry, where a re-sent set
legitimately replaces the old one. The software `av1` encoder rejects H.264/H.265 rather than
silently emit AV1; the `h26x` software tier is the mirror image and rejects
AV1.

### Sample entries: `avc1` / `hvc1`, and `avc3` / `hev1` only where the sets change

Every H.264 / H.265 output — single-file MP4 and CMAF/HLS, from any backend
(software `h26x`, QSV, NVENC, AMF) — is written with the `avc1` / `hvc1` sample
entry: every parameter set in a complete `avcC` / `hvcC` (`hvcC` arrays with
`array_completeness = 1`), and the `CODECS=` string named after it
(`avc1.64001E`, `hvc1.1.6.L93.B0`). That is the entry every player takes:
Safari's plain `<video>` element on iOS refuses `avc3`
(`canPlayType('video/mp4; codecs="avc3.64001E"')` is `""`, and the file fails
with `MEDIA_ERR_SRC_NOT_SUPPORTED`) where the same stream under `avc1` plays,
and Apple's HLS authoring specification asks for `hvc1`.

`avc3` / `hev1` — parameter sets in band — is written only where the sets
really change, which one encoder's stream never does:

- **Single-file, one encoder** (the serial path): the sets are stripped from
  the samples into the config box. Should a stream change a set under its id
  after all, the writer keeps every set in band from that access unit on and
  writes `avc3` / `hev1`, logging the kind and id.
- **Single-file, chunk-and-stitch**: chunks come from independent encoder
  sessions. Before muxing, the stitch checks every chunk's sets
  (`nal_mux::parameter_sets_fixed`): sessions of one encoder with one
  configuration write them byte for byte alike, so the rung is written `avc1` /
  `hvc1` out of band. Chunks whose sets differ under the same id (another
  vendor, another configuration) keep them in band in every access unit under
  `avc3` / `hev1`, so each chunk decodes with its own.
- **CMAF/HLS**: `init.mp4` is written `avc1` / `hvc1` from the first segment's
  sets, and the segments keep their sets in band as well (each segment still
  self-describes). Once a rendition's segments are all written — the multi-GPU
  helpers' included — `cmaf::settle_video_sample_entry` reads every segment's
  in-band sets against the config box: all the box's, and the entry stays
  `avc1` / `hvc1`; any other (a helper encoder that wrote different sets), and
  the entry is rewritten `avc3` / `hev1` in place (the fourcc and `hvcC`'s
  completeness bits; the file keeps its size). The master playlist's `CODECS=`
  is read from the settled init segment, so the two always agree.

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
  (`CODECS="hev1.2.4…"` then; `hvc1.2.4…` since the sample entry is `hvc1`).
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
refuses `--codec av1` with an HDR policy on a build whose only encoders are
software (h26x has no AV1; the software `av1` encoder is 10-bit SDR, so
`--pixel-format 10bit` passes and `--color hdr10` does not). The refusal
treats an HDR policy as needing 10 bits *and* HDR, so it does not offer a
10-bit SDR tier as the fix. A backend pinned by name
(`TRANSCODE_ENCODER_BACKEND`) is added to the compiled set, since
`create_backend` builds `h26x` / `av1` by name with no feature check. It
checks the build, not the silicon: an
AV1 request that the build's NVENC could serve still fails at encoder
construction on a card without AV1 encode.

**NVENC H.264/H.265** uses the codec's GUID for capability validation, preset
selection, and session init; the preset (`GetEncodePresetConfigEx`) seeds the
codec config union so the H264/HEVC layout doesn't have to be mirrored. H.264 is
pinned to High profile, H.265 to Main. Without a `bframes` override the encoder
is **strictly 1-in-1-out** for H.264/H.265 (clear `enableLookahead`, set
`zeroReorderDelay`, every non-IDR picture forced P) and the ring's sync drain
emits one packet per `EncodePicture` (the ring is `RING_SIZE` = 16 surfaces deep, slots chosen by "not outstanding"). With `bframes = N` (`--encode-policy
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
  output covers all three codecs too — the CMAF muxer emits `av01`/`avc1`/
  `hvc1` init segments (`avc3`/`hev1` where the segments' sets differ) and
  `codec_string_from_init` reads the matching `av1C`/`avcC`/`hvcC` config box
  for the `CODECS=` attribute.
- **Inline parameter sets where chunks disagree.** Chunks come from independent
  encoders whose SPS/PPS may agree on the invariant yet differ cosmetically
  (VUI) or in PPS (entropy mode). When they do, the stitch muxer
  (`new_with_codec_inline`) keeps SPS/PPS(/VPS) inline in each access unit and
  emits the `avc3`/`hev1` sample entry, so every chunk decodes with its own
  parameter sets. When every chunk's sets are byte-identical (one encoder, one
  configuration — the software pool on a host with no GPU), the stitch is
  written `avc1`/`hvc1` like the serial path. See
  [Sample entries](#sample-entries-avc1--hvc1-and-avc3--hev1-only-where-the-sets-change).

Validation:
- **NVENC on RTX 3090** (this repo's dev box): H.264 + H.265 each decode 96/96
  frames, 0 errors, BT.709, ~0.9 s (full NVDEC→NVENC round-trip).
- **QSV single-GPU on the Arc box**: H.264/H.265/AV1 each decode 96/96 frames, 0
  errors, identical PSNR-vs-source, consistent BT.709.
- **QSV multi-GPU on the 3× Arc box**: H.264 + H.265 chunk-and-stitch across all
  three Arcs (A310/A380/A750), 5 segments dispatched over the lease pool, H26x
  invariant captured + matched with 0 mismatches. Output was `avc3`/`hev1` with
  inline parameter sets (every stitch was, then) and decodes 300/300 frames, 0
  errors, BT.709.

## Software AV1 — `encode/av1_sw.rs`

> Source: [`encode/av1_sw.rs`](../crates/codec/src/encode/av1_sw.rs),
> [`tuning::av1_sw_params` / `av1_sw_params_with`](../crates/codec/src/encode/tuning/adapters.rs);
> the codec is the submodule `crates/av1` (rivet-av1).

The software AV1 tier is the workspace's own encoder, written clean-room from
the AV1 specification; it replaced rav1e on 2026-10-03
([decisions.md §39](decisions.md#39-av1-and-every-still-image-codec-are-the-workspaces-own-rav1e-rav1d-and-the-image-crate-are-gone)).
Backend `av1` (`EncoderBackend::Av1`) in the capabilities report,
`/v1/health`, the OpenAPI enum and `TRANSCODE_ENCODER_BACKEND` (where `rav1e`
is still accepted). Always compiled; `av1-sw-fallback` (alias
`rav1e-fallback`) decides whether `select_encoder` falls back to it unasked,
and it says so at `warn` when it does.

- **Writes.** Profile 0, 8- or 10-bit 4:2:0 (`yuv420p`, `yuv420p10le`), one
  tile, so at most 4096 pixels wide; one temporal unit per frame, every frame
  shown, in display order; key frames at the interval, inter frames from the
  previous frame.
- **Quality.** `base_q_idx` 1-255: a CRF on AV1's 0-63 scale times four, else
  four times libaom's `cq-level` for the quality target — the same table the
  hardware tiers are equalised against.
- **Rate.** A bitrate rung is coded to its average rate by the crate's rate
  controller (bits per frame = bitrate / frame rate). `rate=cbr` and a coded
  picture buffer are refused by name.
- **Speed.** The tier is the motion search range: ±8 (`draft`), ±16
  (`standard`), ±32 (`archive`). Single-threaded scalar: about 10 frames/s at
  352x288 and 2 frames/s (1.9 megapixels/s) at 1280x720, so about a second a
  frame at 1080p. A fallback, not a production encoder.
- **Colour.** The crate writes no colour description into the sequence
  header; the MP4 `colr` box carries the colour. The tier's caps are
  therefore 10-bit SDR (`{10, false}`): 10-bit SDR AV1 works on a CPU-only
  build, HDR10 / HLG AV1 still needs a GPU.
- **Sessions.** `force_keyframe_next` starts a fresh encoder for the next
  frame (the crate has no forced-key call, and a key frame resets every
  reference anyway), carrying the rate controller's quantiser over. `reset`
  is supported, so the session pool reuses the encoder across chunks.

## The other output codecs: VP9, VP8, MPEG-2, MPEG-4 Part 2, ProRes

> Source: [`encode/{vp9,vp8,mpeg2,mpeg4,prores}_sw.rs`](../crates/codec/src/encode/),
> [`encode/native.rs`](../crates/codec/src/encode/native.rs); the codecs are
> the submodules `crates/{vp9,vp8,mpeg2,mpeg4,prores}` ([decisions §34, §35](decisions.md#35-every-codec-rivet-decodes-it-can-encode--in-software-in-every-build)).

Every codec rivet decodes it can also write. The five beyond the web set are
encoded by the clean-room crates' own encoders, through adapters that mirror
`h26x_sw`: each takes the pipeline's normalized frame, hands its crate the
planes, and gives back one `EncodedPacket` per picture with the frame's own
timestamp and a keyframe flag read from the bitstream.

| Codec (`EncoderBackend`) | Writes | Takes | Rate | Keyframes / order |
|---|---|---|---|---|
| **VP9** (`Vp9`) | profile 0 or 2 (the crate writes 0–3; the pipeline hands it 4:2:0) | 8- or 10-bit 4:2:0 (and 4:2:2 / 4:4:4 at 8 / 10 / 12 bits from a caller's own frames) | fixed `base_q_idx` 1-255 (crf × 4; never 0, which is lossless), or an average bitrate (one-pass rate control, frames recoded when they miss their budget by more than 12 %) | interval + `force_keyframe_next`; inter frames from LAST or GOLDEN (every 8 frames, coded finer); in order |
| **VP8** (`Vp8`) | RFC 6386 | 8-bit 4:2:0 | fixed `q_index` 0-127 (crf × 2) | interval + `force_keyframe_next`; in order |
| **MPEG-2** (`Mpeg2`) | Main Profile, progressive | 8-bit 4:2:0 | constant `quantiser_scale_code` 1-31, or an average bitrate (Test Model 5 allocation, no VBV) | GOPs of the keyframe interval (closed first, open after), I/P/B with `overrides.bframes` (default 2); coded reference-first |
| **MPEG-4 Part 2** (`Mpeg4`) | Simple Profile; Advanced Simple with B-VOPs | 8-bit 4:2:0 | constant `vop_quant` 1-31, or an average bitrate | I-VOP every keyframe interval; B-VOPs only when `overrides.bframes` asks (1-8); coded reference-first |
| **ProRes** (`ProRes`) | the profile in `VideoCodec::ProRes(profile)`: Proxy, LT, 422, HQ (4:2:2), 4444, 4444 XQ (4:4:4) | 8- or 10-bit 4:2:0, upsampled to the profile's chroma | the profile's target frame size (Apple's published rates, area-scaled) | every frame a key frame |

**Dispatch.** No hardware backend here encodes these codecs, so the software
encoder is not a fallback below silicon; it is the encoder.
[`select_encoder`](../crates/codec/src/encode/mod.rs) builds it directly
(`native_backend_for`), with no feature and whatever vendor a lease named;
`encode_capable` answers no for every card, so the encode pool is software;
NVENC, AMF and QSV refuse the codecs by name (`refuse_non_hardware_codec`) if
asked by name. `software_backend_for` returns their backend in every build.

**Quality.** A rung's quality target becomes each codec's quantiser through
the H.26x QP table every software path shares
([`tuning::native_sw_quantizer`](../crates/codec/src/encode/tuning/adapters.rs)):
VP8 `2 × QP`, VP9 `3.8 × QP`, MPEG-2 / MPEG-4 the H.264 step
`0.625 · 2^(QP/6)` over 2.6 (QP 26 → 5). It is a first mapping, not a VMAF
calibration. A `crf` is the codec's own scale: VP8 / VP9 the libvpx 0-63
`cq-level`, MPEG-2 / MPEG-4 their 1-31 codes; ProRes has none and refuses one.
The speed tier (`video-speed`, or an `encode-policy` `speed=` word) sets the
motion search range (and MPEG-4's four-vector macroblocks at `Archive`). For
VP9 it also picks the crate's speed and partitioning, because its
rate-distortion search is what costs:

| tier | crate speed | partition | motion search | single-threaded at 352x288 |
|---|---|---|---|---|
| `draft` | 2 | fixed 32x32 | ±8 | faster than `standard` |
| `standard` (default) | 2 | fixed 16x16 | ±16 | about 10 frames/s |
| `archive` | 1 | searched, NONE / SPLIT, two transform sizes | ±32 | about 1.7 frames/s |

GOLDEN is on at every tier: it is nearly free and worth 7-12 % at the same
quality.

**Refused, by name, before a frame is decoded** (`OutputSpec::validate`) and
again by the adapters: a bitrate for VP8 / ProRes; a constant rate
(`rate=cbr`) or a coded picture buffer for VP9 / MPEG-2 / MPEG-4; B frames for VP8 /
VP9 / ProRes; more than 7 (MPEG-2) or 8 (MPEG-4) B pictures; a crf for ProRes;
HDR for VP9 and 10 bits or HDR for VP8 / MPEG-2 / MPEG-4
(`backend_output_caps_for`: VP9 10-bit SDR, the others 8-bit SDR; ProRes is
10-bit with HDR); sizes past MPEG-2's 4095 × 2800,
MPEG-4's 8191 × 8191 or VP8's 16383 × 16383. An MPEG-2 frame rate H.262
cannot signal (Table 6-4 and its `frame_rate_extension` reach) is refused when
the encoder is built.

**Colour.** ProRes writes the H.273 primaries / transfer / matrix into its
frame header, so an HDR ProRes is HDR in the bitstream as well as in `colr`.
VP9 writes `color_space` (from the matrix) and `color_range`. MPEG-4 writes
`video_signal_type()` — `video_full_range_flag` and the H.273 colour
description (primaries, transfer, matrix) from the source's colour metadata —
into the VOL. VP8 and MPEG-2 carry no colour in the bitstream from these
encoders; the container's `colr` / `Colour` does. MPEG-4's
`force_keyframe_next` makes the next frame an I-VOP.

**Order.** MPEG-2 and MPEG-4 hold B pictures back and code each reference
picture before the B pictures that precede it in display order. Their output
is split into one access unit per picture (the sequence / GOP / VOL headers
ride with the picture they precede), and `ReferenceFirst` stamps each with its
own frame's timestamp — the newest frame held is the reference, then the
others, oldest first — so the muxers' composition offsets
([`crate::reorder`](../crates/container/src/reorder.rs)) come out right.

**Sessions.** `reset` rebuilds the inner encoder (the next frame opens a new
stream with a key frame); `force_keyframe_next` is honoured by VP8, VP9,
MPEG-4 (the next frame an I-VOP) and ProRes (trivially) and not by MPEG-2,
whose crate has no such call.
They never chunk: the multi-GPU single-file engine runs the web set only.

**Where they go.** [container.md](container.md#the-other-codecs-sample-entries-quicktime-and-webm)
has the sample entries and the WebM muxer; [output-spec.md](output-spec.md#5b-output-codec--with_video_codec)
which file each codec goes in.

## The encode dispatch & capability query

> Source: [`crates/codec/src/encode/mod.rs`](../crates/codec/src/encode/mod.rs)

### What

Every encoder backend implements one trait
([`Encoder`](../crates/codec/src/encode/mod.rs#L60)):

```rust
pub trait Encoder: Send {
    fn send_frame(&mut self, frame: &VideoFrame) -> Result<()>;
    fn flush(&mut self) -> Result<()>;
    fn receive_packet(&mut self) -> Result<Option<EncodedPacket>>;
    fn force_keyframe_next(&mut self) -> Result<()> { /* default: unsupported */ }
    fn reset(&mut self) -> Result<()> { /* default: Err(ResetUnsupported) */ }
}
```

`send_frame` pushes one normalized frame; `receive_packet` drains
[`EncodedPacket`](../crates/frame/src/lib.rs#L378)s (the coded bytes — AV1
OBUs or Annex-B NAL units — plus PTS and a keyframe flag; the type lives in the
`rivet-frame` crate and is re-exported here); `flush` signals end-of-stream so
the encoder drains its lookahead/B-frame queue. This is the same push/drain
shape the decode side uses, so the pipeline treats every vendor identically.
The two defaulted methods serve the chunked multi-GPU path:
`force_keyframe_next` promotes the next frame to an IDR (the first kept frame
after a discarded lead-in), and `reset` restarts a session as a new closed-GOP
stream while keeping the device context and surface rings, so one session can
encode many chunks. A backend without them says so (`reset` by the
[`ResetUnsupported`](../crates/codec/src/encode/mod.rs#L114) type), and the
caller rebuilds instead.

[`EncoderConfig`](../crates/codec/src/encode/mod.rs#L138) carries everything a
backend needs: dimensions, frame rate, keyframe interval, the output `codec`,
the perceptual `target` + `tier` (see [tuning](#quality-tuning-perceptual-target--encoder-knobs))
and the legacy `quality` / `speed_preset` escape hatches, the per-rung
`overrides`, a `threads` budget, the input `pixel_format` (8-bit `Yuv420p` vs
10-bit `Yuv420p10le`), source `color_metadata`, and three multi-GPU dispatch
hints — `gpu_index`, `gpu_vendor`, and `constant_qp`.

[`select_encoder`](../crates/codec/src/encode/mod.rs#L607) is the factory. It
detects GPUs at runtime and tries backends **in tier order**:

0. **rivet's own encoders** — for VP9, VP8, MPEG-2, MPEG-4 Part 2 and
   ProRes, the workspace's encoder, directly, before any GPU is looked at
   ([above](#the-other-output-codecs-vp9-vp8-mpeg-2-mpeg-4-part-2-prores)).
1. **Vendor-pin shortcut** — if `config.gpu_vendor` is set (the CMAF
   orchestrator does this via the `GpuPool` lease), dispatch *directly* to that
   vendor's backend, skipping the preference chain
   ([mod.rs:635-755](../crates/codec/src/encode/mod.rs#L635)). The leased card
   is tried first, then its siblings of the same vendor (a host's Arc A310 can
   advertise AV1 encode and then refuse the session while an A380 and A750
   beside it can encode); for Intel, when every card refuses, one unpinned
   legacy `MFXInit` session is tried, with a warning that the job will not
   spread. If all of that fails the error names each card's refusal. There is
   still no CPU fallback on this path.
2. **Auto-select chain** — NVENC (Ada+) → AMF (RDNA3+) → QSV (Arc / Meteor
   Lake+) ([mod.rs:757-841](../crates/codec/src/encode/mod.rs#L757)).
3. **Software** (opt-in) — the last tier, so a build with it on never quietly
   prefers CPU over silicon that was merely busy: the workspace's own `av1`
   encoder for AV1 (`av1-sw-fallback`), its own `h26x` encoders for H.264 / H.265
   (`h26x-fallback`). Each is tried only for its own codec.
4. **Hard fail** — no encode silicon for the codec, no software fallback
   compiled in; the error names the feature that would have caught it.

`TRANSCODE_ENCODER_BACKEND=nvenc|amf|qsv|h26x|av1|prores|vp8|vp9|mpeg2|mpeg4` (the README CLI note; `rav1e` is still accepted for `av1`) maps
to the `preferred: Option<EncoderBackend>` argument, which routes through
[`create_backend`](../crates/codec/src/encode/mod.rs#L955) and bypasses the
chain entirely — including the fallback features, which gate only the *unasked*
route: `h26x` and `av1` by name always construct.

### Why

- **GPU-only by default, fail-fast.** With no software feature enabled the auto
  chain ends in an `Err`, not a CPU fallback.

  rav1e and Vulkan encode were both deleted on 2026-05-08 — "rav1e on Archive
  preset doesn't keep up with real-time throughput at 4K and the Vulkan-encode
  binding never made it past scaffolding". **rav1e came back**, as
  `encode/rav1e_sw.rs`, always compiled; the off-by-default `rav1e-fallback`
  feature decided only whether the chain fell back to it unasked, and then it
  was the last tier before the hard-fail rather than a peer of the vendor
  backends. On 2026-10-03 the workspace's own AV1 encoder replaced it
  (`encode/av1_sw.rs`, `av1-sw-fallback`, with `rav1e-fallback` kept as an
  alias) in the same place, on the same terms
  ([decisions.md §39](decisions.md#39-av1-and-every-still-image-codec-are-the-workspaces-own-rav1e-rav1d-and-the-image-crate-are-gone)).
  Vulkan encode did not come back. Read the 2026-05-08 note as the
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
  ([mod.rs:635-643](../crates/codec/src/encode/mod.rs#L635)).
- **A capability gap is not an error.** An NVIDIA GPU whose NVENC predates AV1
  (consumer 30-series and older) logs an INFO and *falls through* to the next
  vendor rather than failing — it can still decode, just not AV1-encode
  ([mod.rs:782-791](../crates/codec/src/encode/mod.rs#L782)).
- **`pick_vendor_device` honours an explicit index but falls through on a
  vendor mismatch** ([mod.rs:36-58](../crates/codec/src/encode/mod.rs#L36)) so
  that `gpu_index = Some(2)` pinned to an NVIDIA slot, when GPU 2 is actually
  AMD, returns `None` from the NVIDIA tier and gets matched by the AMD tier's
  own `find()` pass. This keeps multi-GPU variant→device pinning correct without
  the caller knowing each device's vendor.

### Capability query — `OutputCaps`

`select_encoder` answers "encode this frame *now*." Before a job starts, the
engine needs "can this build encode *that format at all?*" — that is
[`build_output_caps()`](../crates/codec/src/encode/mod.rs#L394), the runtime
query [`OutputSpec::validate`](pipeline.md#6-color--bit-depth) consults to reject
e.g. an HDR (10-bit) request on a build with no 10-bit encoder.

| Function | Returns |
|----------|---------|
| [`backend_output_caps(backend)`](../crates/codec/src/encode/mod.rs#L366) | Per-backend caps. All three HW backends report `{max_bit_depth: 10, hdr: true}` — NVENC via `Yuv420_10bit`, AMF via `P010`, QSV via in-repo oneVPL P010. The software `h26x` tier reports the same: H.265 Main 10 and H.264 High 10 with the colour description in the SPS VUI (`h26x_sw::colour_description`). The software `av1` encoder is `{10, false}`: 10-bit, but no colour description in its sequence header. |
| [`build_output_caps()`](../crates/codec/src/encode/mod.rs#L394) | The **union over compiled paths**. 10-bit+HDR if any of `nvidia`/`amd`/`qsv`/`h26x-fallback` is on; `av1-sw-fallback` alone is 10-bit SDR. |
| [`backend_output_caps_for(backend, codec)`](../crates/codec/src/encode/mod.rs#L418) | Per backend **and codec**. Differs from the per-backend answer for H.264: 8-bit SDR on NVENC / AMF / QSV (no High 10 encoder), 10-bit HDR on `h26x`. A codec the backend does not serve reports the 8-bit floor. |
| [`build_output_caps_for(codec)`](../crates/codec/src/encode/mod.rs#L432) | The union of the above over compiled paths: H.264 is 10-bit only with `h26x-fallback`. What `OutputSpec::validate` checks a spec's codec against (via rivet's `spec::CodecOutputCaps`), and what `rivet capabilities` prints per codec. |
| [`compiled_encode_backends()`](../crates/codec/src/encode/mod.rs#L449) | The compiled backends as `EncoderBackend`s, in dispatch order — the set the union is taken over, for a caller that needs the per-backend answers behind it (rivet's refusal message names them). |
| [`encode_backends()`](../crates/codec/src/encode/mod.rs#L478) | The compiled backends in dispatch order — `["nvenc", "amf", "qsv", "av1", "h26x"]` filtered by feature flags. Drives `rivet capabilities`. |
| [`software_backend_for(codec)`](../crates/codec/src/encode/mod.rs#L509) / `software_encode_available` / `software_feature_for` | The software backend the chain would fall back to for a codec in this build (`av1` for AV1 under `av1-sw-fallback`, `h26x` for H.264 / H.265 under `h26x-fallback`), answered from the feature flags without building an encoder, and the feature to name in an error. |
| [`backend_codes_constant_rate(backend)`](../crates/codec/src/encode/mod.rs#L286) | Whether a backend codes a [constant-rate rung](#constant-rate-cbr-rungs): QSV, NVENC, AMF and `h26x` yes, the software `av1` and the other native software encoders no. |

Why a runtime union and not a compile-time constant: features are additive and
the answer the validator wants ("can this *binary* produce 10-bit AV1?") is a
property of the whole build, queryable without constructing an encoder.

---

## Hardware encoder backends (and why stubs exist)

All three HW encoders share a shape: a hand-rolled `dlopen` FFI binding (no
external wrapper crate, no bindgen, no build-time SDK link, so they **build on
both Windows MSVC and Linux** even without the hardware present), an
input-surface pool (NVENC's `RING_SIZE` and QSV's `POOL_SIZE` are both 16), a
per-frame YUV→vendor-surface upload, and a flush/drain at EOS. The FFI layouts
are **spec-conformant-by-review**; what has run on hardware is listed per
backend above (H.264 / H.265 on NVENC, QSV and AMF; AV1 on QSV). AV1 on NVENC
and AMF has not: the dev box's RTX 3090 (Ampere) and Ryzen iGPU have no AV1
encode block. The FFI layers carry `const` size assertions that fire at compile
time if a vendored struct layout drifts
([nvenc/buffers.rs](../crates/codec/src/encode/nvenc/buffers.rs#L298),
[nvenc/ffi.rs](../crates/codec/src/encode/nvenc/ffi.rs),
[qsv/ffi.rs](../crates/codec/src/encode/qsv/ffi.rs#L257),
[amf_ffi.rs](../crates/codec/src/amf_ffi.rs)).

### The stub pattern (`*_stub.rs`)

> Source: [`nvenc_stub.rs`](../crates/codec/src/encode/nvenc_stub.rs),
> [`amf_stub.rs`](../crates/codec/src/encode/amf_stub.rs),
> [`qsv_stub.rs`](../crates/codec/src/encode/qsv_stub.rs)

`encode/mod.rs` uses `#[path = "…_stub.rs"]` to swap a stub in when a vendor
feature is off ([mod.rs:1-15](../crates/codec/src/encode/mod.rs#L1)). The stub
keeps `nvenc::NvencEncoder` (etc.) a **real type with the same `new()`
signature** so the dispatcher in `select_encoder` compiles unchanged — but
`new()` always `bail!`s with a "rebuild with the `nvidia` feature" message. The
trait methods are `unreachable!()` because the encoder is never constructed.

**Why:** it lets the dispatch logic reference all three backends without
`#[cfg]` noise at every call site. Auto-select simply sees the stub's
construction error and skips that tier; an explicit `EncoderBackend::Qsv`
request surfaces the helpful "not compiled in" error instead of a cryptic
missing-symbol link failure.

### NVENC (`nvenc/`)

> AV1: NVIDIA Ada+ (RTX 4000+, Ampere datacenter A10/A10G/L4/L40). H.264 / H.265
> on the older generations too (see the codec table above).

Drives the NVENC API through the `NV_ENCODE_API_FUNCTION_LIST` function-pointer
table (`NvEncodeAPICreateInstance`) rather than dlsym-ing each symbol — the
entry point the NVENC Programming Guide documents ([nvenc/mod.rs:6-11](../crates/codec/src/encode/nvenc/mod.rs#L6)).
Session flow is documented in the module header
([nvenc/mod.rs:13-28](../crates/codec/src/encode/nvenc/mod.rs#L13)): open session →
preset config → init → input/bitstream ring buffers → per-frame
lock/copy/encode/extract → EOS flush → teardown in reverse alloc order.

10-bit uses `NV_ENC_BUFFER_FORMAT_YUV420_10BIT`; the pipeline stores 10-bit in
the *lower* 10 bits of each `u16`, so `upload_frame_10bit` performs the `<<6`
shift on copy to satisfy NVENC's P010-style *upper-10-bits* convention
([nvenc/upload.rs:111](../crates/codec/src/encode/nvenc/upload.rs#L111)).

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
[amf_ffi.rs](../crates/codec/src/amf_ffi.rs) (shared with the AMF decoder) list every slot of every C
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
use `CONSTANT_QP` with `QPI` / `QPP` (/ `QPB`); a
[constant-rate rung](#constant-rate-cbr-rungs) uses `CBR`. Every IDR — frame 0, each GOP
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

### QSV (`qsv/`)

> Intel Arc (DG2/BMG) + Meteor/Lunar Lake iGPUs. oneVPL `libvpl`.

Struct-driven (everything lives in `mfxVideoParam` fields, no property bag). The
flow runs a `Query` pass first so the runtime can adjust params, then `Init`,
then a 16-surface pool (`POOL_SIZE`, [qsv/surface.rs](../crates/codec/src/encode/qsv/surface.rs#L35);
session flow in [qsv/mod.rs:9-36](../crates/codec/src/encode/qsv/mod.rs#L9)).
Shared `mfx` struct layouts live in `crate::qsv_ffi` so encode and decode can't
drift apart ([qsv/mod.rs:62-64](../crates/codec/src/encode/qsv/mod.rs#L62));
`qsv/ffi.rs` holds only the encode-side constants and ext buffers.

Three QSV decisions are worth calling out:

- **LowPower / VDENC is ON, not OFF.** AV1 QSV encode is **VDENC (low-power)
  only** — it's the only AV1 encode entry point the iHD driver exposes — so
  `LowPower` must be `MFX_CODINGOPTION_ON` or `Query` rejects with
  `MFX_ERR_UNSUPPORTED` ([qsv/mod.rs:530-534](../crates/codec/src/encode/qsv/mod.rs#L530),
  set by the adapters at [tuning/adapters.rs:351, 501](../crates/codec/src/encode/tuning/adapters.rs#L351)
  and asserted in [tuning/tests.rs:265](../crates/codec/src/encode/tuning/tests.rs#L265)).
  *Note:* the `QsvAv1Params.low_power` field doc still reads
  "Always `MFX_CODINGOPTION_OFF`"
  ([tuning/params.rs:257](../crates/codec/src/encode/tuning/params.rs#L257)) — that comment
  is **stale**; the actual emitted value is ON.
- **ICQ is rate-control mode 9, not 8.** `MFX_RATECONTROL_ICQ = 9`; **8 is
  `MFX_RATECONTROL_LA`** (lookahead). The original code used 8 and AV1/Arc
  rejected `Query` with `MFX_ERR_UNSUPPORTED`
  ([qsv/ffi.rs:66-72](../crates/codec/src/encode/qsv/ffi.rs#L66)). ICQ (Intelligent
  Constant Quality) is the QSV equivalent of CRF and the right match for a
  perceptual target; lookahead-bitrate is not used. The numeric value in
  `tuning/`'s `QsvRateControl` enum is documentary — `qsv/ffi.rs` holds the
  authoritative wire constant and `qsv/mod.rs` only consumes the tuning enum to pick the
  CQP-vs-ICQ *branch* ([qsv/mod.rs:421-440](../crates/codec/src/encode/qsv/mod.rs#L421)).
  A [constant-rate rung](#constant-rate-cbr-rungs) bypasses both:
  `MFX_RATECONTROL_CBR` (1).
- **16-multiple coded dims + neutral-black NV12 fill (the "green bars" fix).**
  AV1 requires coded dimensions that are a multiple of 16, so e.g. 572×240
  encodes at 576×240 and 1080 at 1088. The surface is allocated at the aligned
  size (`width` → `align_up(.., 16)`, with `crop_w`/`crop_h` set to the real
  dims; pitch aligned to 64 bytes for Arc DMA — [qsv/mod.rs:487-491](../crates/codec/src/encode/qsv/mod.rs#L487),
  [qsv/mod.rs:750-751](../crates/codec/src/encode/qsv/mod.rs#L750)). The per-frame upload
  only touches the real pixels, so the padding rows/cols would otherwise be
  **zero**, which a browser decodes through BT.709 as the distinctive **green
  bars**. The fix: pre-fill each pool surface with *neutral black* — `Y=16,
  Cb/Cr=128` for 8-bit BT.709 limited (and `<<6` for P010 10-bit) — so the
  untouched padding decodes as black ([qsv/mod.rs:759-780](../crates/codec/src/encode/qsv/mod.rs#L759)).

---

## Quality tuning: perceptual target → encoder knobs

> Source: [`crates/codec/src/encode/tuning/`](../crates/codec/src/encode/tuning/)

### What

The user picks two backend-agnostic things; the adapter translates them into
each encoder's native parameters so identical inputs yield visually consistent
output across vendors.

- [`QualityTarget`](../crates/codec/src/encode/tuning/mod.rs#L79) — a **perceptual
  goal** expressed in VMAF/SSIMULACRA2 bands, *not* an encoder CRF:
  `VisuallyLossless` (~VMAF 98) · `High` (~95) · `Standard` (~90, default) ·
  `Low` (~85) · `Vmaf(u8)` (explicit escape hatch).
- [`SpeedTier`](../crates/codec/src/encode/tuning/mod.rs#L96) — how much wall-clock
  to spend: `Draft` · `Standard` (default) · `Archive`. Maps to native speed
  presets (NVENC P5/P6/P7, etc.; the software AV1 encoder's motion search
  range ±8 / ±16 / ±32; VP9's speed and partitioning, above). Set for every
  rung by `video-speed` (CLI `--video-speed`, API / manifest `video_speed`);
  an `encode-policy` `speed=` word for one rung wins.

The `*_av1_params(target, tier, width, height)` functions
([nvenc_av1_params](../crates/codec/src/encode/tuning/adapters.rs#L61),
[amf_av1_params](../crates/codec/src/encode/tuning/adapters.rs#L127),
[qsv_av1_params](../crates/codec/src/encode/tuning/adapters.rs#L447)) each return a
concrete params struct the matching encoder splats into its SDK structs; the
H.26x adapters (`qsv_h26x_params`, `amf_h26x_params`, `h26x_sw_params`) share
one 0..51 QP anchor table so a job keeps its QP whichever backend runs it.
Resolution is an input because tile grid and lookahead sizing depend on frame
size.

These connect to `EncoderConfig` via the `AUTO_FROM_TARGET = u8::MAX` sentinel
([mod.rs:228](../crates/codec/src/encode/mod.rs#L228)): when `quality` /
`speed_preset` are left at the sentinel, the encoder derives the quantizer/preset
from `target`/`tier`; a non-sentinel value is a legacy per-encoder override
(e.g. a literal CQP q-index, used by `ParallelConstQp` chunk seams).

### Why

- **libaom is the cross-encoder reference.** Every backend is equalized *to*
  libaom's VMAF at each quality band
  ([libaom_cq_for_target](../crates/codec/src/encode/tuning/mod.rs#L151)), then a
  per-encoder calibration shift compensates for that encoder's
  compression-efficiency gap (NVENC ~3-4 CQ lower, AMF ~8 q-index lower in
  0..255 space). The `Vmaf(u8)` escape hatch interpolates between calibrated
  anchor tables ([piecewise_cq](../crates/codec/src/encode/tuning/mod.rs#L193)).
  The tuning source comments cite `docs/av1-tuning-research.md` for the
  source tables; that file is not in this repository.
- **The QP scales genuinely differ per vendor**, and the doc comments encode the
  traps: NVENC AV1 CQ is **0..63** (not the 0..51 H.264/HEVC range); AMF q-index
  is the full AV1 **0..255**; QSV ICQ is **1..51** (an oneVPL idiosyncrasy that
  scales AV1's 0..63 into 0..51 for API parity). A value sent on the wrong scale
  is silently mis-quantized or rejected.
- **Fewer tiles = better compression on HW encoders.** Tile boundaries break
  loop-filter continuity and AV1 tiles are entropy-coded independently, so the
  shared HW tile grid ([tile_grid_hw](../crates/codec/src/encode/tuning/mod.rs#L291))
  caps at 2×2 even at 4K — the HW encoders have enough internal parallelism that
  they don't need a software encoder's aggressive 4×4 grid for throughput. A regression test
  pins every grid inside AV1 Level 5.1 tile limits
  ([tuning/tests.rs:388](../crates/codec/src/encode/tuning/tests.rs#L388)).
- **No low-latency presets.** This is a batch transcode service, so NVENC
  P1–P4 and AMF `Speed` are deliberately never selected, and a rung with no
  rate of its own never gets a bitrate mode — `VisuallyLossless`/`Archive` uses
  constant-QP for reproducible bitstreams, everything else uses a
  quality-targeting VBR. Rate modes are opt-in per rung: an average bitrate,
  which only the software H.264 / H.265 tier codes
  ([bitrate rungs](#bitrate-rungs-in-the-software-tier-measured); the hardware
  backends refuse one by name), and a constant rate (`rate=cbr`), which QSV,
  NVENC, AMF and the software tier code ([constant-rate rungs](#constant-rate-cbr-rungs)).

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
| av1 (software) | `base_q_idx` 1..255 = 4 × libaom | up = worse |

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
- a bitrate job whose encode pool is GPUs: only the software tiers code to an
  average rate. NVENC, AMF and QSV each refuse one at construction
  too (`encode::refuse_rate`). A constant rate is a different request; see
  [constant-rate rungs](#constant-rate-cbr-rungs).

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

### Constant-rate (CBR) rungs

A bitrate rung spends its rate on average unless it asks for a constant one:
`EncodeOverrides::rate_mode = Some(RateMode::Constant)`
([`tuning/rate.rs`](../crates/codec/src/encode/tuning/rate.rs)), spelled
`rate=cbr` (or `constant`; `average` / `abr` is the default) in the policy
grammar and `rate-mode=cbr` in the settings ([output-spec.md](output-spec.md)).
The rate is then also the maximum, an HRD buffer is always declared (one
second, `CBR_DEFAULT_BUFFER_MS`, unless the rung names another), and the
encoder holds the rate. The decoder's buffer starts 48/64 full
(`CBR_INITIAL_FULLNESS_64THS`).

Who codes it (`backend_codes_constant_rate`):
- **QSV** (AV1, H.264, H.265): `MFX_RATECONTROL_CBR` with `TargetKbps` =
  `MaxKbps`, `InitialDelayInKB` and `BufferSizeInKB`, scaled by
  `BRCParamMultiplier` past a `u16`.
- **NVENC** (every codec): `NV_ENC_PARAMS_RC_CBR`, average = max = the rate,
  `vbvBufferSize` / `vbvInitialDelay` in bits.
- **AMF** (AV1, H.264, H.265): the CBR rate-control method, target = peak,
  VBV size and initial fullness, the HRD enforced and filler data on; an
  H.264 / H.265 level is raised until it admits the rate.
- **The software H.264 / H.265 tier**: a bitrate rung with h26x's `cbr` set,
  so the NAL HRD declares `cbr_flag` 1 and filler data (H.264 NAL type 12,
  H.265 `FD_NUT`) keeps the coded picture buffer exact.
- **The software AV1 and VP9 encoders** refuse it by name: they code an
  average rate, not a constant one.

Each hardware backend calls `encode::constant_rate_request` before it touches
a driver: a constant-rate rung it can code becomes a `ConstantRate`, a rung
with no rate keeps its quality target, and an average rate is refused as
above. Refused by name in the knob's own words
(`tuning::constant_rate_refusal`): a constant rate beside a CRF, under
`--seam-mode constqp`, with `buffer=0`, or with no bitrate at the encoder.
A constant-rate rung that names no rate is given one where the frame rate is
known, the engine's job setup: `tuning::default_cbr_bitrate`, a streaming
table by short side (H.264 at up to 30 fps: 0.2 Mb/s at 144 to 16 Mb/s at
2160), scaled up above 30 fps by half the extra frame rate, and 0.65x for
H.265, 0.5x for AV1. Those anchors are common streaming-ladder rates, not a
measurement of these encoders.

[`EncodeOverrides`]: ../crates/codec/src/encode/tuning/overrides.rs
[`RungPolicy`]: ../crates/codec/src/encode/tuning/overrides.rs
[`RungRule`]: ../crates/codec/src/encode/tuning/overrides.rs
[`RungSelector`]: ../crates/codec/src/encode/tuning/overrides.rs

---

## Colorspace: normalizing decoder frames for the encoder

> Source: [`crates/codec/src/colorspace/`](../crates/codec/src/colorspace/mod.rs)

### What

The encoders accept 4:2:0 only (8-bit BT.709 limited, or 10-bit for HDR
output). Decoders emit a zoo of layouts — NV12/NV21, 4:2:2, 4:4:4, RGB,
8-, 10- and 12-bit, BT.601/709/2020, studio or full range. This module is the
funnel. Public entry points, all in [`colorspace/mod.rs`](../crates/codec/src/colorspace/mod.rs):

- [`convert_to_yuv420p_bt709`](../crates/codec/src/colorspace/mod.rs#L273) /
  [`convert_to_yuv420p_bt709_in_range`](../crates/codec/src/colorspace/mod.rs#L282)
  — the 8-bit-aware normalizer. Dispatches by format: 10- and 12-bit /
  wide-gamut passes through on the matrix axis (chroma layout still normalized
  to 4:2:0, 12-bit narrowed to 10); RGB goes through a BT.709 RGB→YUV matrix;
  YUV chroma layouts are deinterleaved/averaged to 4:2:0; then a BT.601→709
  matrix correction runs for any non-709-tagged YUV source, with the
  full-range coefficients when the source declares full range. The full
  input→output coverage table is in the doc comment above
  `convert_to_sdr_bt709` ([colorspace/mod.rs:109-142](../crates/codec/src/colorspace/mod.rs#L109)).
- [`convert_to_sdr_bt709`](../crates/codec/src/colorspace/mod.rs#L154) — the
  **HDR-aware** dispatch the pipeline calls when it has the source
  `ColorMetadata`. PQ/HLG in any layout above 8 bits → normalized to
  `Yuv420p10le` (a full-range source re-coded to studio range first) and
  tonemapped to 8-bit BT.709 (see next section); everything else falls through
  to `convert_to_yuv420p_bt709_in_range` with SDR semantics unchanged.
- [`normalize_layout_to_420`](../crates/codec/src/colorspace/mod.rs#L247) —
  layout and depth only, no matrix and no tonemap: what the passthrough / HDR
  output policies use, so the encoder gets a layout it takes and the mux keeps
  the source's colour tags.
- [`SdrToHdr`](../crates/codec/src/colorspace/sdr_in_hdr.rs#L177) — the
  opposite direction, for an SDR source under `--color hdr10|hlg`: the picture
  placed in a PQ or HLG signal per ITU-R BT.2408 (SDR reference white at
  203 cd/m², BT.1886 EOTF, source primaries → BT.2020, 10-bit limited
  BT.2020 NCL out), rather than re-tagged. A source it cannot map (already
  HDR, linear light, an unmapped matrix or primaries code) is refused by name
  before any frame.
- [`depth`](../crates/codec/src/colorspace/depth.rs) — rounded right-shift
  narrowing (12 → 10, 12 → 8, 10 → 8, HM's `convertToLowerBitDepth`) and
  `<< 2` widening, scalar + AVX2, bit-exact with each other.

Plus the scalers, in [`scale.rs`](../crates/codec/src/colorspace/scale.rs):
[`scale_frame`](../crates/codec/src/colorspace/scale.rs#L9) bilinear-scales
`Yuv420p` / `Yuv420p10le` to the rung's dimensions (an identity fast-path
returns a cheap clone when dims already match), and
[`scale_region`](../crates/codec/src/colorspace/scale.rs#L687) does a crop, a
resize and a pad in one pass: the crop window of the source (snapped to even
offsets and sizes), scaled, placed at an offset on a canvas of limited-range
black (luma 16 / 64, chroma 128 / 512). That is what a fitted rung needs —
cutting a wider picture to the output's shape, letterboxing a narrower one —
and rivet's fit placement calls it. Unlike `scale_frame` it reads an
odd-sized frame with the `ceil(w/2) x ceil(h/2)` chroma planes the decoders
write; `scale_frame` takes them as `w/2`, which put the V plane of an 853x480
picture inside U. A whole even frame resized to an equal canvas takes
`scale_frame`'s path unchanged.

### Why & the AVX2 runtime-dispatch pattern

The hot kernels — BT.601→709 matrix, 4:4:4→4:2:0 downsample (box and
Lanczos), bit-depth narrowing, bilinear scale — each ship as a **scalar reference** plus an `#[target_feature(enable = "avx2")]`
SIMD specialization, behind a safe public dispatcher that runtime-detects AVX2
(`is_x86_feature_detected!("avx2")`) and falls back to scalar otherwise
([bt601_to_bt709_planes](../crates/codec/src/colorspace/bt601_to_709.rs#L170),
[bilinear_scale_plane_u16](../crates/codec/src/colorspace/scale.rs#L215)). The CPUID
check is the safety boundary for the `unsafe` SIMD fn. This is the project-wide
AVX dispatch convention (`feedback_avx_runtime_dispatch.md`): runtime-detect,
keep a scalar fallback, only specialize loops that actually bench hot. The scalar
path stays `pub` so benches and non-x86 builds can target it directly.

Notable decisions:

- **BT.601→709 is a delta-space matrix with no luma-into-chroma coupling.** The
  3×3 is derived by composing BT.601 YUV→RGB with BT.709 RGB→YUV in limited-range
  form; the derivation and a black/white/gray round-trip sanity check are written
  out in the source ([bt601_to_709.rs:1-36](../crates/codec/src/colorspace/bt601_to_709.rs#L1)).
  A full-range source takes the same matrix with the luma row's chroma terms
  scaled by 224/219 (`bt601_to_bt709_planes_full_range`).
  The AVX2 kernel uses `_mm256_mulhrs_epi16` for Q15 fixed-point multiplies and
  splits off the identity contribution for the ~1.0 coefficients that overflow
  i16 ([bt601_to_709.rs:208-220](../crates/codec/src/colorspace/bt601_to_709.rs#L208)).
- **10-bit BT.601→709 exists but is off the default path.** The 10-bit pipeline
  is HDR-passthrough/tonemap, never matrix-converted (a BT.601 matrix would
  corrupt a wide gamut). The 10-bit converter is wired behind a public entry for
  explicitly-tagged BT.601 10-bit content (some Sony broadcast cameras) but
  callers must opt in ([bt601_to_709_10bit.rs:17-23](../crates/codec/src/colorspace/bt601_to_709_10bit.rs#L17)).
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
  neither the software AV1 encoder nor the hardware ones expose AV1's experimental alpha
  ([downsample_444.rs:34-40](../crates/codec/src/colorspace/downsample_444.rs#L34)).
- **Matrix is preserved on passthrough, not silently rewritten.** 10-bit/wide-gamut
  frames keep their `color_space`; the encoder signals it in the AV1 sequence
  header and the mux writes `colr nclx`, so a player can reverse the matrix. The
  one exception is 8-bit BT.2020 (rare), which routes through the BT.601 matrix
  with a documented slight hue shift rather than bailing
  ([colorspace/mod.rs:316-325](../crates/codec/src/colorspace/mod.rs#L316)).

---

## Tonemapping & the single-output policy

> Source: [`crates/codec/src/tonemap.rs`](../crates/codec/src/tonemap.rs)

### What

[`tonemap_yuv420p10le_bt2020_to_yuv420p_bt709`](../crates/codec/src/tonemap.rs#L396)
maps a 10-bit BT.2020 PQ/HLG frame down to an 8-bit BT.709 limited-range frame.
The pipeline (per pixel) is: 10-bit Y'CbCr → R'G'B' (BT.2020 NCL matrix) →
scene-linear RGB (PQ or HLG inverse EOTF) → BT.709 gamut → **Hable filmic curve**
→ BT.709 OETF → 8-bit BT.709 limited Y'CbCr
([tonemap.rs:1-5](../crates/codec/src/tonemap.rs#L1)). Chroma is downsampled by
averaging the four per-pixel post-tonemap chroma values per 2×2 block (rather
than tonemapping once per chroma site), which avoids hue shifts at high
luminance ([tonemap.rs:384-395](../crates/codec/src/tonemap.rs#L384)).

`convert_to_sdr_bt709` (above) is the caller; the scene-linear white point comes
from the source's mastering-display `max_luminance` when present, else a
1000-nit HDR10 default ([tonemap.rs:292](../crates/codec/src/tonemap.rs#L292)).

### Why the single-output tonemap-to-SDR policy

Stated in the module header
([tonemap.rs:7-11](../crates/codec/src/tonemap.rs#L7)) and the
[README's web-defaults pitch](../README.md): every HDR upload is tonemapped to
SDR at transcode time and the encoded ABR ladder is 8-bit BT.709, so **every
viewer sees a correctly-mapped image regardless of display capability**. Shipping
native HDR without the upstream UI/processing work (YouTube/Instagram have given
whole talks on it) lands badly-converted, eye-searing or washed-out clips on
viewers. That is the default (`ColorPolicy::TonemapToSdr`); HDR output is
opt-in per job: `--color passthrough` keeps the source's colour and depth, and
`--color hdr10|hlg` forces BT.2020 PQ / HLG at 10 bits, mapping an SDR source
into the HDR signal with `colorspace::SdrToHdr` rather than re-tagging it
(see [pipeline.md §6](pipeline.md#6-color--bit-depth) and
[output-spec.md §4](output-spec.md#4-color--bit-depth)). Those paths use the
10-bit encode, the `mdcv`/`clli` mux atoms and sequence-header / VUI HDR
signalling. A dual-rendition ladder (HDR for HDR viewers beside SDR rungs from
these primitives) does not exist.

Two implementation "why"s worth flagging:

- **The HLG path applies an OOTF (γ=1.2), not just the inverse OETF.** HLG
  signals are *scene*-referred; without the scene→display OOTF, midtones land in
  the wrong place — this is exactly why iPhone HLG clips famously read ~1 stop
  too bright on naive pipelines (the camera assumes Apple's downstream tonemapper
  applies it) ([tonemap.rs:78-123](../crates/codec/src/tonemap.rs#L78)).
- **Hable's coefficients + exposure bias 2.0 are the published values
  verbatim** ([tonemap.rs:139-150](../crates/codec/src/tonemap.rs#L139)):
  A–F = 0.15 / 0.50 / 0.10 / 0.20 / 0.02 / 0.30, `ExposureBias = 2.0` and the
  `Uncharted2Tonemap` rational, from John Hable's "Filmic Tonemapping
  Operators" (filmicworlds.com, 2010), the published form of his GDC 2010
  "Uncharted 2: HDR Lighting" talk.
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

## The audio pipeline: decode → Opus / AAC / HE-AAC / MP3 / Vorbis / AC-3 / E-AC-3 / DTS / FLAC / ALAC

> Source: [`crates/codec/src/audio/`](../crates/codec/src/audio/mod.rs)

### What

The audio side is a small decode→encode framework over the workspace's own
codec crates — every codec, both ways, is a clean-room crate of its own in its
own repository, brought in as a submodule: `crates/opus`
([rivet-opus](https://github.com/rivet-transcoder/rivet-opus)), `crates/mp3`
([rivet-mp3](https://github.com/rivet-transcoder/rivet-mp3)), `crates/vorbis`
([rivet-vorbis](https://github.com/rivet-transcoder/rivet-vorbis)), `crates/aac`,
`crates/ac3`, `crates/dts` and `crates/lossless`. No C library, no build script,
no third-party codec crate. The [pipeline routing](pipeline.md#7-audio) decides
per source codec:

| Source | Action | Output |
|--------|--------|--------|
| AAC, Opus, AC-3, E-AC-3, DTS (and MP3 into an MP4; Opus and Vorbis into a WebM) | **Passthrough** (no decode) | carried verbatim into the container |
| Any other decodable source — Vorbis, MP2, PCM, FLAC, ALAC (MP3 for HLS) | **Decode → re-encode to Opus** | Opus + `dOps` |
| Any decodable source with `--audio opus`, a filter, or a layout change | **Decode → remix → re-encode** | Opus + `dOps` |
| Any decodable source with `--audio aac` / `he-aac` / `he-aacv2` (an AAC source of that kind is copied) | **Decode → remix → encode AAC-LC / HE-AAC / HE-AAC v2** | AAC + `esds` (`mp4a.40.2` / `.5` / `.29`) |
| Any decodable source with `--audio ac3` / `eac3` / `dts` | **Decode → remix (≤ 5.1) → encode** | `ac-3` + `dac3` / `ec-3` + `dec3` / `dtsc` + `ddts` |
| Any decodable source with `--audio vorbis` | **Decode → remix → encode Vorbis** | WebM `A_VORBIS`, or an Ogg file |
| Any decodable source with `--audio mp3` or `--mode audio` | **Decode → remix (≤ 2 ch) → encode MP3** | MP3 frames |
| Any decodable source with `--audio flac` / `alac` ([lossless audio](lossless-audio.md)); a source already in that codec is copied | **Decode → (remix only if asked) → encode losslessly** | FLAC + `dfLa` / ALAC + cookie |
| everything else | **Drop** (video-only, warn) | — |

"Decodable" is the job's list (`mp3`, `mp2`, `vorbis`, `opus`, `ac3`, `eac3`,
`dts`, `flac`, `alac`, linear PCM, and AAC whose first access unit the AAC
probe accepts), minus any codec `audio-decode-deny` names. An HE-AAC source
decodes in full (SBR at the full rate, parametric stereo to two channels); the
`he-aac` setting can keep it undecoded or decode only its core
([output-spec.md §3](output-spec.md#3-audio--with_audioaudiocodecpolicy)).

This crate owns the decode, remix and encode steps; the routing itself is
rivet's (`job/audio.rs`). The wire model
([audio/mod.rs](../crates/codec/src/audio/mod.rs)):

- [`AudioFrame`](../crates/codec/src/audio/mod.rs) — interleaved f32 PCM in
  [-1.0, 1.0] (`LRLR…`) + rate/channels + µs PTS. The canonical exchange type.
- [`AudioDecoder`](../crates/codec/src/audio/mod.rs) /
  [`AudioEncoder`](../crates/codec/src/audio/mod.rs) — object-safe traits;
  `create_decoder("mp3"|"mp2"|"vorbis"|"opus"|"ac3"|"eac3"|"dts"|"aac"|"flac"|"alac"|"pcm_s16le"|…, …)` and
  `create_encoder(AudioEncoderConfig { codec: AudioCodec::Opus | Mp3 | Aac | HeAac | HeAacV2 | Vorbis | Ac3 | Eac3 | Dts | Flac { .. } | Alac { .. }, .. })`
  are the routing entry points. `AudioEncoderConfig` carries the input rate and
  channels, the bit rate (0: the codec's default for the layout), Vorbis's
  quality, and the input's speakers (`layout`), which AC-3 and DTS need: four
  channels are 4.0, quad(side) or 3.1 to them. A decoder that knows its
  stream's speakers reports them (`AudioDecoder::layout`: AC-3's `acmod`, DTS's
  `AMODE`, AAC's channel configuration); an encoder reports the rate it codes
  at (`AudioEncoder::sample_rate`, the timescale of its packet durations), its
  delay (`pre_skip`), its configuration (`extra_data`: the `OpusHead`, the
  AudioSpecificConfig, the Vorbis headers in Xiph lacing; empty for MP3, AC-3
  and DTS, whose frames describe themselves) and, for MP3, the bare file's tag
  frame (`file_header`).
- [`AlignedResampler`](../crates/codec/src/audio/resample.rs) puts an encoder's
  input at a rate it codes (rubato's band-limited sinc), the filter's measured
  delay trimmed and the output cut to the input's length at the new rate, so an
  encoder's `pre_skip` is its own codec delay alone and the output keeps the
  input's timing to within half a sample.
- The lossless encoders, clean-room and pure Rust, in the `lossless` crate
  ([`crates/lossless`](../crates/lossless/README.md), a git submodule: the
  [rivet-lossless](https://github.com/rivet-transcoder/rivet-lossless)
  repository; details, verification and compression figures in
  [lossless-audio.md](lossless-audio.md)):
  [`FlacEncoder`](../crates/lossless/src/flac/encode.rs) (`lossless::flac::Encoder`,
  behind `FlacAudioEncoder`, its `AudioEncoder` adapter in
  [`encode/flac.rs`](../crates/codec/src/audio/encode/flac.rs); ALAC likewise) — 4096-sample
  frames; per subframe the cheapest of constant, verbatim, fixed orders 0–4
  and LPC (Tukey-windowed autocorrelation → Levinson-Durbin → quantised with
  error feedback), with a Rice partition search and raw-bits escapes; stereo
  tries independent, left/side, side/right and mid/side; wasted bits;
  STREAMINFO with the MD5; three efforts (`FlacLevel`).
  [`AlacEncoder`](../crates/lossless/src/alac/encode.rs) (`lossless::alac::Encoder`,
  behind `AlacAudioEncoder` in
  [`encode/alac.rs`](../crates/codec/src/audio/encode/alac.rs)) — ALAC's
  sign-LMS adaptive predictor seeded per frame from LPC (orders 4 and 8), the
  adaptive Rice coder, weighted pair mixing, low-byte splitting for 20/24-bit
  (the Rice parameter caps at 14 bits) and an escape to raw samples when
  smaller, keeping the predictor's arithmetic within the 32-bit range a
  reference decoder uses.
- [`remix`](../crates/codec/src/audio/remix.rs) builds the matrix between two
  layouts (ITU-R BS.775 downmix, LFE dropped, side/back surrounds relabelled
  or folded, a back centre split into the surround pair, mono as the folded
  stereo downmix, normalised so nothing clips) and says which layout each
  codec carries a source in: Opus and Vorbis (`opus_layout`, `vorbis_layout`:
  the eight Vorbis-order arrangements), AAC and HE-AAC (`aac_layout`: quad as
  5.0, 2.1 as 5.1, 6.1 as 7.1), AC-3, E-AC-3 and DTS (`surround_core_layout`:
  A/52's `acmod` 1/0 to 3/2 with or without the LFE, 5.1 as 5.1(side), 7.1
  downmixed to 5.1(side)), MP3 and HE-AAC v2 (stereo at most). It never
  upmixes: an output speaker the input has nothing for is silent, and the job
  refuses a request for more channels than the source has.

Decoders:

- [`Mp3Decoder`](../crates/codec/src/audio/decode/mp3.rs) adapts the
  `crates/mp3` decoder (Layers I, II and III; MPEG-1, MPEG-2 LSF, MPEG-2.5) to
  packets: byte runs in any chunking, sync confirmed against the next header,
  ID3 tags and garbage skipped, a Xing / Info / VBRI frame recognised and not
  played. Its own gapless trimming is off — the container's edit (or the `.mp3`
  tag, which `container::mp3::read_file` turns into one) is applied by the job —
  so the output keeps the encoder's delay plus the decoder's 529 samples. MP2
  and MP1 tracks (`"mp2"`, `"mp1"`) go to the same decoder.
- [`VorbisDecoder`](../crates/codec/src/audio/decode/vorbis.rs) adapts the
  `crates/vorbis` decoder. It takes Matroska's `CodecPrivate` (the three
  Xiph-laced headers; an Ogg file's demuxer laces them the same way) as
  `extra_data`, decodes each packet to the overlapping halves, and permutes
  Vorbis order (5.1 = FL FC FR RL RR LFE) into the native one.
- [`OpusDecoder`](../crates/codec/src/audio/decode/opus.rs) adapts the
  `crates/opus` multistream decoder for every layout (family 0 as one stream,
  family 1 permuted from the RFC 7845 order back into the native one, the head's
  output gain applied); it keeps the pre-skip, which the container's edit (or
  the `OpusHead`, when the container states none) hides.
- [`PcmDecoder`](../crates/codec/src/audio/decode/pcm.rs) converts AVI's WAVE
  linear PCM (`pcm_u8`, `pcm_s16le`, `pcm_s24le`, `pcm_s32le`, `pcm_f32le`,
  `pcm_f64le`) to f32.
- AC-3 / E-AC-3, DTS, AAC, FLAC and ALAC decode through adapters onto the
  `crates/ac3`, `crates/dts`, `crates/aac` and `crates/lossless` submodules;
  see [codec-decode.md](codec-decode.md).

Encoders:

- [`OpusEncoder`](../crates/codec/src/audio/encode/opus.rs) adapts the
  `crates/opus` encoder (written from RFC 6716 / 8251 / 7845): one
  `MultistreamEncoder` for every layout — family 0 for mono and stereo, family
  1 for 3.0 to 7.1, whose input is in Vorbis order, so each 20 ms frame is
  permuted from the native order on the way in (`audio::rfc7845_family1_order`;
  the round-trip test decodes through rivet's decoder and checks each channel
  keeps its tone). CELT (`Application::Audio`), VBR, 20 ms (960-sample)
  packets at 48 kHz, other input rates through `AlignedResampler`. `pre_skip`
  is the encoder's lookahead (312 samples at 48 kHz), and the stream is padded
  with silent packets until the decoded output covers the pre-skip and every
  input sample, so the edit list (or an Ogg granule position) ends it exactly.
  The bit rate is the total for all streams, 6–510 kb/s per stream; 0 is 64k
  per mono and 96k per coupled stream (96k stereo, 320k 5.1, 416k 7.1).
  `extra_data` is the `OpusHead` body (the `dOps` box's fields).
- [`Mp3Encoder`](../crates/codec/src/audio/encode/mp3.rs) adapts the
  `crates/mp3` Layer III encoder (written from ISO/IEC 11172-3 / 13818-3):
  CBR on the MPEG-1 ladder (128k stereo / 64k mono by default), joint stereo
  chosen frame by frame, at 32 / 44.1 / 48 kHz (other rates resampled: the
  11.025 kHz family to 44.1, the rest to 48). One packet per frame. `pre_skip`
  is the encoder's delay (528) plus the decoder's (529), which the MP4 edit
  list hides; for a bare `.mp3`, `file_header` is the encoder's own `Info`
  frame with its LAME-style extension (encoder string `rivetmp3`, delay,
  padding, music CRC, tag CRC), so a gapless player — rivet's own reader among
  them — presents exactly the input.
- [`VorbisEncoder`](../crates/codec/src/audio/encode/vorbis.rs) adapts the
  `crates/vorbis` encoder: quality −1 to 10 (5 by default; no bit rate:
  Vorbis is variable-rate by design), 8 to 192 kHz as the input has it, mono
  to 7.1 permuted into Vorbis order. Packets are timed by their granule
  positions — the first lasts zero samples, the last ends at the input's
  length — so the durations add up to the input exactly; no priming.
  `extra_data` is the three headers in Xiph lacing (WebM's `CodecPrivate`).
- [`AacEncoder`](../crates/codec/src/audio/encode/aac.rs) adapts the encoder of
  the `crates/aac` submodule (the rivet-aac repository, written from ISO/IEC
  13818-7 / 14496-3; provenance in
  [decisions.md §26](decisions.md#26-aac-lc-is-encoded-and-decoded-here-from-the-standards)),
  in three profiles. **AAC-LC** codes at 8 / 11.025 / 12 / 16 / 22.05 / 24 /
  32 / 44.1 / 48 kHz: a source at one of them keeps its rate (8–16 kHz
  speech is no longer resampled up to 22.05 / 24 kHz); `lc_rate` picks the
  target otherwise (the lowest coded rate at or above the input's, in its
  family, or above it when an explicit bit rate is over the 6144-bit-a-channel
  decoder buffer of the input's rate: 48 kb/s a channel at 8 kHz), channel
  configurations 1–7 (mono, stereo, 3.0, 4.0, 5.0, 5.1 and 7.1, with the SCE /
  CPE / LFE element order of Table 42), one raw access unit per 1024 samples
  plus the 2-byte AudioSpecificConfig; `adts_frame` wraps one for TS. Inside: a
  sine-window MDCT with long / short block switching, a psychoacoustic model on
  the MDCT spectrum, per-band M/S, a rate loop that finds one noise-to-mask
  offset per frame by bisection against a bit-reservoir budget, and exact
  sectioning by dynamic programme. The priming is 1024 samples (`pre_skip`). A
  bitrate of 0 takes `default_bitrate`: 64k mono, 128k stereo, 384k 5.1, 512k
  7.1. **HE-AAC** (mono to 7.1) and **HE-AAC v2** (stereo) code at 32, 44.1 or
  48 kHz (`he_aac_rate`): an AAC-LC core at half the rate plus SBR data from a
  64-band QMF analysis (and, for v2, a parametric-stereo downmix), 2048 output
  samples per access unit, 3586 samples of priming (`HE_AAC_DELAY`), 48k / 32k
  stereo by default. Their AudioSpecificConfig signals SBR / PS explicitly and
  hierarchically (object type 5 or 29 first), the form MP4 and Apple's players
  read as `mp4a.40.5` / `mp4a.40.29`.
- [`Ac3Encoder`](../crates/codec/src/audio/encode/ac3.rs) adapts the
  `crates/ac3` encoder (written from ATSC A/52:2018, Annex E for E-AC-3):
  **AC-3** at Table 5.18's rates (32–640 kb/s; 96k mono, 192k stereo, 384k for
  three or four channels, 448k for 5.1 by default) and **E-AC-3** at 32–6144
  kb/s in whole kb/s (96k, 192k, 256k, 384k), at 48 / 44.1 / 32 kHz, any A/52
  arrangement from 1/0 to 3/2 with or without the LFE, the pipeline's speakers
  reordered into the encoder's. Whole syncframes, 1536 samples each; 256
  samples of transform delay (`pre_skip`). The muxer derives `dac3` / `dec3`
  from the first syncframe (`AudioInfo::from_ac3_frame`).
- [`DtsEncoder`](../crates/codec/src/audio/encode/dts.rs) adapts the
  `crates/dts` core encoder (ETSI TS 102 114): the core arrangements (mono,
  stereo, 3.0, 3.0(back), 4.0, quad(side), 5.0(side), each with or without the
  LFE) at 48 / 44.1 / 32 kHz and Table 5-7's rates (1536 / 1411.2 / 1024 kb/s,
  the full rate, by default); 512-sample frames of constant size, no ADPCM
  prediction (so any core decoder decodes them), 512 samples of filterbank
  delay. `ddts` comes from the first frame (`AudioInfo::from_dts_frame`).

### Why

- **Why our own codecs.** Every audio codec rivet writes or reads is a
  clean-room crate of this project's: no C to build (the libopus build needed
  CMake), nothing loaded at run time (MP3 encode used to dlopen LAME behind a
  feature), and the same pure-Rust build on every host. Each crate is verified
  on its own — the Opus decoder against all twelve RFC 8251 test vectors (final
  range and `opus_compare`), the MPEG audio decoder against ISO's 64 conformance
  sequences at full accuracy, the Vorbis decoder against Xiph's vectors, the
  AAC decoder against ISO/IEC 14496-26 — and rivet's tests encode, mux, demux
  and decode every output with rivet's own code (`crates/rivet/tests/audio_codecs_e2e.rs`).
- **Why Opus by default, and why it's royalty-clean.** Opus carries
  royalty-free licensing commitments to the IETF and modern browsers all play
  Opus-in-MP4: AV1 video + Opus audio + MP4 container is the project's
  royalty-clean output. AAC (LC and HE), AC-3, E-AC-3 and DTS are there for the
  players and pipelines that want them, asked for by name; `auto` passes them
  through verbatim where it can.
- **Why 48 kHz and an aligned resampler.** Opus always codes at 48 kHz, so
  `pre_skip` is uniformly in 48 kHz ticks per the RFC and the `OpusHead`
  `InputSampleRate` carries the *original* source rate. Every encoder that
  resamples trims the filter's delay itself, so the only lead-in an output's
  edit list (or `OpusHead`, or MP3 tag) has to state is the codec's own.
- **Why the PTS/pre_skip plumbing matters.** The design collapses an encoder's
  lead-in into the single `pre_skip` count written into `dOps` / the edit list,
  and pads each stream so its decoded length covers that and every input
  sample: a conformant decoder discards exactly the front padding and presents
  exactly the input's length.

---

## Key decisions on the encode side (recap)

- **AV1-default output (H.264 / H.265 also selectable), GPU-only encode.** No CPU
  encode tier — `select_encoder` hard-fails on a host without NVENC/AMF/QSV
  encode silicon rather than degrading to a 20× slower software path — unless
  the build opted into `av1-sw-fallback` (AV1) / `h26x-fallback` (H.264 /
  H.265), which sit *below* the vendor chain so they are a floor, never a
  preference.
- **VP9, VP8, MPEG-2, MPEG-4 Part 2 and ProRes are software, always.** rivet's
  own clean-room encoders are the only encoders of those codecs, so they are
  built directly in every build — no fallback feature, because there is no
  faster tier to fall back from.
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
- **Royalty-clean audio by default, every codec our own.** Opus for transcode +
  AAC/Opus/AC-3/E-AC-3/DTS passthrough. AAC-LC, HE-AAC, HE-AAC v2, MP3, Vorbis,
  AC-3, E-AC-3, DTS, FLAC and ALAC are opt-in outputs — all from the
  workspace's own clean-room codec crates, no third-party codec library.
