# rivet — GPU backend status & hardware-verification backlog

Every GPU backend is hand-rolled `dlopen` FFI in-tree (no external wrapper crate;
builds on Windows MSVC + Linux). This tracks what's been **run on real silicon**
vs. what's only been **reviewed** and still needs a card. AV1 is the default
output codec (4:2:0, Main profile, 8- or 10-bit); H.264 / H.265 are selectable.

| Vendor | Feature | Decode | Encode (AV1) |
|--------|---------|--------|--------------|
| Intel  | `qsv`   | ✅ verified | ✅ verified |
| NVIDIA | `nvidia`| ✅ verified | ⚠ by-review |
| AMD    | `amd`   | ⚠ by-review (its FFI still has the pre-2026-08-27 vtable layout — see below) | ⚠ by-review (AV1); **✅ verified H.264 / H.265** on the Ryzen 9 9950X iGPU |
| Software | `rav1d-fallback` / `rav1e-fallback` | ✅ AV1 | ✅ AV1 8-bit |
| Software | `h26x` (always) / `h26x-fallback` | ✅ H.264 + HEVC, conformance bit-exact | ✅ H.264 + H.265 8-bit, SELF + libavcodec cross-checked |

---

## Intel — `qsv` ✅ COMPLETE

Hardware-verified end-to-end on a **3× Intel Arc** box (A310/A380/A750, Ubuntu
26.04, iHD 26.1.2), 2026-06-27.

- **Decode** (oneVPL, `decode/qsv_dec.rs`): H.264, HEVC, AV1, VP9; 8-bit **and**
  10-bit P010 (HEVC Main10 → AV1 `yuv420p10le` verified). Uses the oneVPL 2.x
  internal-allocation + `FrameInterface::Map` path.
- **Encode** (oneVPL AV1, `encode/qsv.rs`): AV1 8-bit + 10-bit P010, verified.
- Also verified: **multi-GPU** chunk-and-stitch across all 3 cards, the **HLS ABR
  ladder**, and **non-16-multiple rungs** (572×240, neutral-black padding so no
  green bars).

Remaining: only a nice-to-have.
- [ ] By-eye browser QA of an odd-width source — confirm no green bars when a
      player decodes the coded frame and ignores the crop.

---

## NVIDIA — `nvidia`

- **Decode** (NVDEC / CUVID, `decode/nvdec.rs`): H.264, HEVC, AV1, VP8, VP9,
  MPEG-2, MPEG-4 Part 2; 10-bit **P016**. ✅ **Verified on RTX 3090**
  (`nvdec_smoke` 17/17).
- **Encode** (NVENC AV1, `encode/nvenc.rs`): AV1 8-bit + 10-bit. ⚠ **By-review
  only** — the dev box is Ampere (RTX 3090), which has no AV1-encode silicon. The
  capability query *is* hardware-proven on the 3090 (it correctly reports "2
  codecs, none AV1" and rejects).

- [ ] **NVENC AV1 encode** end-to-end on **Ada+** (RTX 4000+ / L4 / A10G):
      correct pixels, valid `av1C`, and the 10-bit (`YUV420_10BIT`) path.

---

## AMD — `amd`

Hand-rolled AMF FFI mirroring the AMD AMF SDK headers (`decode/amf_dec.rs`,
`encode/amf.rs`), plus `amf_device.rs` (Windows DXGI/D3D11 adapter routing).

**Done (2026-06-29, on the RTX 3090 + Ryzen 9 9950X box):**
- **Windows AMD/Intel GPU detection** (WMI `Win32_VideoController`) — AMD GPUs are
  enumerated on Windows, not just via Linux sysfs.
- **Heterogeneous index space** — `GpuDevice::vendor_index` (vendor-local, for the
  hardware adapter) + a globally-unique `index` (what the user addresses), so an
  NVIDIA + AMD host no longer collides on index 0.
- **AMF multi-adapter routing** — a D3D11 device made on the chosen AMD adapter
  (`D3D11_CREATE_DEVICE_VIDEO_SUPPORT`) is handed to `InitDX11`, so AMF binds to
  the right GPU on a mixed host instead of DXGI adapter 0 (the NVIDIA card). The
  iGPU is detected as global index 1 and AMF reaches it (D3D11 create/drop test
  passes).
- **Graceful failure** — a failed AMF init no longer segfaults (the
  external-device failure path corrupts the context, so it's leaked on that cold
  path); `--decode-gpu fastest` skips an AMF-incapable GPU and an explicit pin
  errors cleanly.

**Done (2026-08-27, `agent/amf-h26x`): native H.264 / H.265 encode, and the AMF
FFI re-mirrored from the SDK v1.4.36 C headers.** The encoder's vtables had not
matched the headers (`AMFInterface` is Acquire/Release/QueryInterface, ten
`AMFPropertyStorage` slots precede every interface's methods, `InitVulkan` is on
`AMFContext1`, `AMFVariantStruct` is 24 bytes, `AMF_RESULT` is sequential —
`AMF_EOF` 23 / `AMF_INPUT_FULL` 25 / `AMF_NEED_MORE_INPUT` 44 — and the AV1
component id, several property names and enum values were off). `encode/amf/ffi.rs`
now lists every slot in header order with compile-time offset assertions, and
`test_amf_runtime_property_storage_abi` exercises the property-storage ABI on the
installed `amfrt64.dll`.

> **The Ryzen 9 9950X iGPU (`AMD Radeon(TM) Graphics`, driver 32.0.21045.5002) *is*
> AMF-capable for H.264 / H.265** (the old note called it a 9700X; `Win32_Processor`
> says 9950X). The earlier
> "`InitDX11` returns `AMF_NOT_FOUND`" was the mis-slotted vtable calling
> `GetProperty` (slot 4 in the header) where `InitDX11` was assumed, and
> `AMF_NOT_FOUND` (11) is what `GetProperty` returns for a missing name. With the
> corrected layout `AmfEncoder::new` succeeds for H.264 and H.265 on it, and AV1
> fails with `AMF_CODEC_NOT_SUPPORTED` (30), which is right for a VCN 3.1 iGPU.
> Verified there: 360p / 720p / 1080p H.264 and H.265, H.265 Main 10 (P010), HLS,
> `force_keyframe_next` (IDR + in-band SPS/PPS/VPS), one packet per frame after
> the flush fix, luma PSNR 41-53 dB vs source, ffmpeg full decode clean.

> `decode/amf_dec.rs` still carries the **old vtable layout** (same slot errors,
> plus its `AMF_IID_SURFACE` guess and the 2020-series result codes), so its
> "InitDX11 failed (rc=11)" is the same GetProperty misread; it falls through to
> the software decoders. Port it onto `encode/amf/ffi.rs` (the shared
> `AmfPropertyStorageVtbl` / `AmfDataVtbl` / `AmfSurfaceVtbl` / `AmfContextVtbl`)
> and it can be verified on this box for H.264 / HEVC (VP9 / AV1 decode too).

> Expect the same class of struct-layout / init-flow surprises QSV had on first
> real hardware. QSV needed: every mfx struct offsetof-verified, the MFXLoad
> dispatcher (not legacy init), an advisory Query (proceed to Init on the
> driver's spurious `-3`), LowPower=ON, and a frame-sized output buffer. Budget
> for an equivalent debugging pass on AMF.

Verify:
- [ ] **AMF decode** — port `decode/amf_dec.rs` onto `encode/amf/ffi.rs`'s
      vtables (`IID_AMFSurface` is `{0x3075dbe3, 0x8718, 0x4cfa, {0x86, 0xfb,
      0x21, 0x14, 0xc0, 0xa5, 0xa4, 0x51}}`, `core/Surface.h:222`; `Convert` is
      `AMFData` slot 15), then verify H.264 / HEVC / VP9 pixels on the 9950X iGPU
      against the software decoders, AV1 on a discrete RDNA card.
- [ ] **AMF AV1 encode** (RDNA3+, RX 7000+) — the AV1 property sequence in
      `encode/amf/av1.rs` is by-review: names/values from `VideoEncoderAV1.h`,
      the same session flow as the validated H.26x components, the QVBR
      inversion inferred from the H.26x measurement. Needs 8-bit + P010 end to
      end, correct pixels, and a check that `Av1QvbrQualityLevel` really runs
      higher = better.
- [ ] **AMF H.264 / H.265 on a discrete Radeon** — validated on the 9950X iGPU
      (Adrenalin, 2026-08) only. A discrete RDNA2/3 card, and Linux via
      `AMFContext1::InitVulkan`, are still owed; so is a VMAF sweep of the shared
      H.26x QP anchors and of `52 - QP` as the QVBR level.
- [ ] **AMF Main 10 rate control** — constant QP because this driver ignores the
      QVBR level at 10 bits (levels 1 / 26 / 32 / 38 gave the identical
      17.3 Mbit/s stream). Re-test on a newer driver / discrete card; if QVBR
      works there, gate the CQP rule on the driver instead of on bit depth.
- [ ] **AMF QVBR bitrate ceiling** — `TargetBitrate` / `PeakBitrate` /
      `VBVBufferSize` + `EnforceHRD` are set; no encode here came near its
      ceiling, so enforcement is unmeasured.

---

## Software AV1 — `rav1e-fallback` / `rav1d-fallback` (optional)

rav1e (encode) and rav1d (decode), both pure Rust and both **AV1 8-bit 4:2:0
only**. No system libraries, no bindgen, no LLVM — which is the point: they are
the safety net for a host with no usable encode/decode silicon, without making
the build environment part of the deployment story.

No hardware verification is owed (there is no hardware), but the round-trip is
covered by `crates/codec/tests/software_av1_roundtrip.rs`, which encodes and
decodes a synthetic frame and checks a hard vertical edge on **every row** —
a stride or plane-origin bug shears the picture progressively down the frame
and a spot-check misses it.

Not covered, and deliberately: software decode of VP8 / VP9 / MPEG-2 / MPEG-4 /
ProRes. See [No FFmpeg](README.md#no-ffmpeg).

---

## Software H.264 / H.265 — `crates/h26x` (`h26x-fallback` for encode)

The workspace's own codec pair, a git submodule of
[rivet-h26x-codecs](https://github.com/rivet-transcoder/rivet-h26x-codecs).
**Decode** (2026-08-18): H.264 and HEVC, bit-exact against the JVT / JCT-VC
conformance suites (199/199, 35/35, 146/147, 32/32 accepted), frame- and
wavefront-threaded, SSE2→AVX-512 + NEON; always in the decode chain below the
hardware tiers. **Encode** (2026-08-27): both codecs, 8-bit 4:2:0, wired in as
`encode/h26x_sw.rs` behind `h26x-fallback` (same policy switch as rav1e) and
always constructible by name (`TRANSCODE_ENCODER_BACKEND=h26x`). The encoder
gate (`crates/h26x/tools/verify_encode.sh`, 280 cells over 9 clips × 31
configs) holds four properties per cell: SELF (our decoder reproduces the
encoder's own reconstruction byte for byte), CROSS (libavcodec agrees with our
decoder), PSNR reported, and rate / CPB objectives hit where set. Round-trip
through rivet's adapters: `crates/codec/tests/software_h26x_roundtrip.rs`.

What the encoders have: H.264 CAVLC + CABAC, I/P/B with real motion search and
spatial direct, 16x8 / 8x16 / 8x8 partitions, 8x8 transform, all four chroma
formats, lossless, ABR rate control; H.265 intra/P/B, TU splits, deblocking,
SAO, lossless, ABR + VBV/HRD with panic-mode re-code, RDOQ (intra). rivet uses
CABAC, no B pictures (the muxer carries no composition offsets), constant QP
from the shared H.26x anchor table, tools chosen by `SpeedTier`.

Open, in order of value to a transcoder:
- [x] **10-bit H.265 encode** (2026-08-27, h26x `632478a`): the H.265 encoder is
      generic over the sample type (Main 10 and 12-bit; 4:2:0/4:2:2/4:4:4), 35
      deep gate cells SELF + CROSS (libavcodec at 10/12-bit) + BOX green, 8-bit
      output byte-identical; rivet's software tier takes `yuv420p10le` for H.265.
- [ ] **H.264 High 10 encode** — every H.264 decision module is concretely `u8`
      (~190 sites across 7 files); a track-sized job. No hardware backend here
      does H.264 10-bit either, so the pipeline refuses it by name.
- [ ] **VUI colour description in the h26x encoders** — neither writes
      `video_signal_type_present_flag` / colour primaries / transfer / matrix, so
      HDR metadata travels only in the container `colr` box and
      `backend_output_caps` reports the tier as 10-bit without HDR; rivet's
      validator therefore refuses HDR10/HLG on a software-only build. Small
      writer change (H.264 already has a VUI for HRD) + `ColorMetadata` plumbing.
- [ ] **Speed as a gate axis.** RDOQ is 51% of all-intra encode time for 1.81%
      BD-rate (measured 2026-08-20, 64x64, 96 frames); it shipped with quality
      reported five ways and no cost number. Report encode time beside size and
      PSNR, then decide whether RDOQ defaults on. An early-out (a block whose
      last coefficient is large can never want trimming) probably keeps most of
      the gain cheaply.
- [ ] **H.264 counted shape rate** (`agent/subparts` branch, `904eabf`,
      held): pricing P-partition shapes with real bins instead of constants
      made the encoder split *less* and lost 0.03% overall — the old undercharge
      had been standing in for the missing inter residual rate. Land it together
      with a counted residual term, not before (lead's ruling, 2026-08-20).
- [ ] **H.264 B partition shapes** (`B_16x8` / `B_8x16` / `B_8x8`) are refused
      by name; B pictures code as `B_16x16` / direct.
- [ ] **H.264 has no CPB model** — `encode::hrd` is codec-agnostic; only the
      SPS/VUI writer is H.265-specific.
- [ ] **B pictures in rivet** — the encoders do them (non-pyramid); the muxer
      would need `ctts` / composition offsets to carry the reorder.
- [ ] **crates.io publish of `rivet-h26x`** — irreversible, needs an explicit
      go-ahead; the next `rivet-codec` publish depends on it (path dep 0.2.0).
- [ ] The multi-GPU ladder (HLS, chunked single-file) needs a GPU lease and
      bails on a CPU-only host; serial single-file is the software path today.
      Same limit as `rav1e-fallback`.

---

## Filters — denoise

The spatial denoise family is implemented (`codec::filter`, `denoise=METHOD:STRENGTH`):
**bilateral, gaussian, median, mean, nlmeans, anisotropic** — selectable, 8-bit,
unit-tested + verified end-to-end (720p, 30 fps): mean/gaussian ≈ baseline,
median/bilateral fast, anisotropic ~0.09 s/frame, nlmeans ~0.84 s/frame
(offline-only). See [docs/filters/denoise.md](docs/filters/denoise.md).

Non-local means is additionally exposed with its own ffmpeg-compatible
parameters (`nlmeans=s=..:p=..:pc=..:r=..:rc=..` — patch size, research window,
separate chroma values, σ strength), evaluated through a summed-area table so
the patch size is free and only the research window drives the cost. See
[docs/filters/nlmeans.md](docs/filters/nlmeans.md).

Follow-ups:
- [ ] **Deep denoise — DPIR** ([cszn/DPIR](https://github.com/cszn/DPIR), DRUNet):
      a `denoise=dpir` method running the DRUNet CNN via ONNX (`tract` pure-Rust
      CPU, or `ort` for CUDA/DirectML GPU). Export the model to ONNX once + vendor
      it (~32 MB, takes a σ noise-level channel ← STRENGTH); load it in
      `FilterChain::prepare` (resource-filter pattern, like `overlay`); luma-only
      `drunet_gray` first, full YUV→RGB→DRUNet→YUV colour as a refinement.
      GPU-bound, opt-in, offline. A self-contained sprint (ML dep + model asset).
- [x] **Temporal denoise** — `hqdn3d` (ffmpeg's `ls:cs:lt:ct`, tables and
      16-bit arithmetic mirrored). The stateless `Arc<FilterChain>` stays the
      shared, immutable part; `FilterChain::instantiate()` gives each decode
      stream (each clip of each pump) a `FilterInstance` holding its own
      history, so rungs / ranges / splice clips never share one. Range-parallel
      decode falls back to whole for a temporal chain. See
      [docs/filters/hqdn3d.md](docs/filters/hqdn3d.md).
- [ ] **NLM-temporal** — a non-local-means variant whose research window spans
      the previous frame(s) as well; the per-stream state now exists to hold
      the frames.
- [x] **AVX2 (+ SSE4.1) denoise kernels** — bilateral, gaussian, mean, median
      and the fixed `denoise=nlmeans` (now SAT-based) run on 128- or 256-bit
      lanes, bit-identical to the scalar reference (per-kernel tests over
      random + edge planes at every tier; `RIVET_DENOISE_MAX_SIMD=none|sse41`
      caps the tier, `RIVET_DENOISE_THREADS=1` the row bands). Anisotropic
      stays scalar: its conduction is `exp` of a non-integer, which no lane
      kernel can reproduce bit-exactly against the host libm. Numbers in
      [docs/filters/denoise.md](docs/filters/denoise.md#cost).

---

## Chunk seams — fixed

Chunk boundaries used to be far more visible than the IDRs at GOP boundaries,
which read as an evenly spaced stutter. Measured with inter-frame motion
(`tblend=difference,signalstats`) against the source on 1080p content — *not*
PSNR, which misses this entirely because each frame is individually fine and it
is the join between them that jumps:

| | excess motion at chunk boundaries |
|---|---|
| originally (chunk length == GOP length) | 2.27x |
| 5-GOP chunks, no margin | 1.86x |
| **10-GOP chunks + 1-GOP margin** | **1.19x** |
| single encoder (`--seam-mode serial`) | 1.21x |
| ffmpeg `hevc_qsv -g 48` | 1.21x |

At or below the single-encoder reference, with seams every ~20 s rather than
every 2 s, and full multi-GPU parallelism retained. Three things got it there:

1. **Chunk length decoupled from GOP length.** They were the same variable, so
   every GOP boundary was a chunk boundary.
2. **One output bitstream per ring slot.** The ring shared one, so under
   sustained pressure two frames landed in it between syncs and were emitted as
   a single packet — the packet count then didn't match the frame count, which
   is what the MP4 sample table is built from. This is what made long chunks
   unusable and blocked everything else.
3. **A one-GOP lead-in margin**, encoded to warm rate control and lookahead and
   then discarded, so the chunk's first kept frame is neither a cold-start IDR
   nor preceded by a flushed tail.

Regressions to watch for, each of which caught a broken attempt: container
sample count vs decoded frame count, decoder errors, and IDR cadence. A quality
metric will not catch any of them — a stream whose chunks opened on P-frames
predicting from discarded margin frames scored *higher* mean PSNR than the
correct one, because ffmpeg conceals the missing references.

---

## Encoder session reuse (chunked multi-GPU)

`chunk_worker::encode_chunk_to_packets` builds a fresh encoder for every chunk.
That's what makes each chunk an independently decodable IDR-led GOP, which the
stitcher relies on — chunks are encoded out of order across GPUs and
concatenated — but it means ~1300 session constructions on a feature-length
file. Measured on the 3x Arc box: 89 constructions in 70 s of wall clock.

- [ ] Pool sessions per (GPU, encoder config) and use `MFXVideoENCODE_Reset`
      between chunks instead of tearing the session down. Reset restarts the
      GOP, so the IDR-led guarantee survives. Needs a `reset()` on the
      `Encoder` trait, defaulting to "unsupported" so NVENC/AMF keep rebuilding
      until they grow an equivalent.
- [ ] Not urgent: the single-file pipeline is **decode-bound** long before
      session setup matters. Measured 109 fps for 1080p H.264 -> HEVC with all
      three Arcs available, one decode pump feeding them; the helpers spend
      most of their time waiting on frames, not on session init. Fix the
      decode side first if throughput is the goal.

---

## Encode tuning — H.264 / H.265 calibration

`tuning::qsv_params` now branches per codec: AV1 keeps its 0..255 q-index and
H.264/HEVC get an ordinary 0..51 QP, which is what stopped an HEVC job being
handed `libaom_cq * 4` (up to 152) as its QPI.

The two branches don't have equal provenance, and shouldn't be read as if they
do. The **AV1** anchors are measured against libaom as the cross-encoder
reference ([docs/av1-tuning-research.md](docs/av1-tuning-research.md)). The
**H.264 / HEVC** anchors are the conventional x264 / x265 CRF values per tier
(18 / 22 / 26 / 32) — a sound starting point, but convention, not measurement.

- [ ] Run the same offline VMAF sweep for QSV HEVC and H.264 that §2.6 defines
      for AV1, and replace the anchors in `qsv_h26x_params` with the measured
      values. Until then a given `QualityTarget` is *not* guaranteed to land in
      the same VMAF band across codecs the way it does across AV1 backends.
- [ ] Same gap on the NVENC side: `nvenc_av1_params` is the only NVENC table,
      so H.264/H.265 on NVENC inherit AV1 calibration too.

---

## Audio — multichannel decode

The **encode** side of surround is done and wired: `channelmap`
([docs/audio-filters.md](docs/audio-filters.md)) remaps channels on decoded PCM,
and the Opus encoder carries 1–8 channels (family 0 for mono/stereo, family 1
multistream for 3–8, RFC 7845 §5.1.1.2). The job layer no longer drops >2ch.

What's binding is the **decode** side: rivet decodes **MP3 and Vorbis** only. So
5.1 Vorbis → Opus 5.1 works today, and 5.1 AC-3 / E-AC-3 / AAC can only be
passed through untouched — which is the common case for real files.

- [ ] **In-tree AC-3 / E-AC-3 decoder** (`codec/src/audio/decode/ac3.rs`). The
      *header* half already exists and is solid — `container/src/ac3_sync.rs`
      parses both AC-3 and E-AC-3 syncinfo/bsi including `acmod` + `lfeon`, so
      the channel layout is already known. What's missing is decode-to-PCM:
      exponent ungrouping (D15/D25/D45), the A/52 §7.2 bit-allocation routine,
      mantissa dequantization, coupling, rematrixing, and the 256/512-point
      IMDCT with overlap-add.

      **Blocked on the ETSI TS 102 366 normative tables, which must be
      transcribed from the spec, not reconstructed.** Specifically:
      `hth[3][50]` (hearing threshold — 150 arbitrary values, not derivable),
      `latab[256]`, `bndtab`/`bndsz`, `slowdec`/`fastdec`/`slowgain`/`floortab`/
      `fastgain`, and the grouped-mantissa quantizer levels. Getting `hth` wrong
      doesn't degrade the audio — it desynchronises bit allocation, so the
      mantissa field widths are wrong and the bitstream reads as garbage from
      that point on. Do this with the spec open; don't estimate.

      The KBD window and `frmsizetab` *are* derivable (Kaiser-Bessel α=5 and the
      bitrate ladder respectively), so those don't need transcription.

- [ ] **AAC-LC decoder** — the other common multichannel source. Comparable
      scope to AC-3 but with Huffman codebooks; `container/src/aac_asc.rs`
      already parses the AudioSpecificConfig, so channel config is known.

      **BLOCKED on a lawful table source (checked 2026-08-27).** The 12 Huffman
      codebooks (~1362 codewords), the `swb_offset` tables (12 rates × long/short)
      and `TNS_MAX_BANDS` exist only in ISO/IEC 14496-3 / 13818-7, which are
      paywalled; every free document that might carry them defers to ISO by
      reference — 3GPP TS 26.402 / 26.403 (ARIB republication), ITU-R BS.1196-8,
      ARIB STD-B32; ISO's public-standards site is closed; the 1998 MPEG-4 FCD
      PDFs are dead links with no archive capture; 3GPP TS 26.410/26.411 are C
      reference code (excluded by the licence rule, as are libavcodec/faad2
      tables and the unauthorised spec copies floating around). Nothing past the
      ICS header parses without the codebooks, so there is no partial deliverable.
      Not blocked: KBD/sine windows, the 1024/128 IMDCT, ADTS/ASC parsing.
      Decision needed: (a) buy ISO/IEC 13818-7:2006 (MPEG-2 AAC; the LC tables
      are identical) and transcribe from it; (b) depend on `symphonia-codec-aac`
      (MPL-2.0, file-level copyleft, tables trace to NihAV/MIT) — not in-tree;
      (c) platform decoders (Media Foundation / AudioToolbox) — not portable.

---

## Subtitles — ✅ done

Text subtitle passthrough (`-c:s copy`): every text track (Matroska SRT / ASS /
WebVTT; MP4 `tx3g` / `wvtt`) is demuxed and markup-stripped, `--subtitles
all|none|<lang,lang>` selects by language on every surface (CLI, settings
header, batch manifest, HTTP API), single-file MP4 gets a gap-filled `tx3g`
`trak` per language, an HLS package gets a segmented-WebVTT rendition per
language on the video's segment grid (`EXT-X-MEDIA:TYPE=SUBTITLES`,
`X-TIMESTAMP-MAP` per segment), and trims / `splice` re-base each clip's cues
onto the output timeline and merge tracks by language. See
[docs/cli.md#subtitles](docs/cli.md#subtitles).

Deliberately not done:
- **Bitmap subtitles** (PGS / VobSub / DVB) are dropped with a warning and
  will stay dropped — neither `tx3g` nor WebVTT has a bitmap form. Carrying
  them would mean a different output container.
- **Styling** is stripped, not translated: `tx3g` styles by byte range in
  side boxes and WebVTT by inline tags, and a faithful mapping from ASS
  overrides is a project of its own.

---

## Codebase modularization (one-thing-per-file) — ✅ done

Every large source file across all three crates was split into a directory of
small, single-purpose files (a thin `mod.rs` re-exporting the public API +
per-concern submodules + a `tests.rs`), the paradigm set by `codec::filter`.
Pure mechanical splits, no behaviour change — each verified by build + tests
before commit. The 2k–4.6k-line monoliths are gone (largest remaining is a
cohesive parser / encoder core or a test file).

- [x] **codec**: `filter` (per-filter + `denoise/` per-algorithm), `colorspace`,
      `gpu`, `encode/tuning`, `pixel_format` (bitreader/h264/hevc/av1/mpeg2),
      `encode/{nvenc,amf,qsv}`, `decode/nvdec`, `audio/encode/opus`.
- [x] **container**: `mux`, `demux`, `ts`, `cmaf`, `avi`.
- [x] **rivet**: `job`, `multigpu`, `server`, `spec` (policy/rung), `encoder_worker`,
      and `main.rs` (kept as the binary entry; subcommands extracted to `commands/`).
- [x] **second tier** (nested sub-dirs): `pixel_format/av1` (obu/sequence/frame),
      `demux/{mp4,mkv,audio}`, and the two largest files in the tree —
      `mux/tests` + `ts/tests` (split by concern into `tests/` directories).

**No file exceeds ~1300 lines.** The only files still over 1000 are deliberately
left whole — each is a single cohesive function that can't be split by pure code
movement (splitting would mean restructuring the function, i.e. a behaviour-risky
refactor): `encode/nvenc/mod.rs` + `encode/qsv/mod.rs` (the FFI encoder `new()`
/encode), `pixel_format/av1/frame.rs` (the AV1 uncompressed-header parser),
`mux/mod.rs` (the muxer `finalize`). The `nvdec_smoke.rs` integration test is
also left (a test *binary*, awkward to split without changing the binary layout).

Verification: 668 lib+integration tests pass across the three crates; per-file
`#[test]` counts + active assertion counts are byte-for-byte unchanged from before
the work (no test was weakened). One pre-existing failure remains —
`create_decoder_accepts_prores_codec_label` — unrelated to this work (it predates
it; `decode/mod.rs` is unchanged): a stale test expecting a ProRes CPU decoder
that the GPU-only directive removed.
