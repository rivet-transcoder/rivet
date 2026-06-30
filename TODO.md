# rivet — GPU backend status & hardware-verification backlog

Every GPU backend is hand-rolled `dlopen` FFI in-tree (no external wrapper crate;
builds on Windows MSVC + Linux). This tracks what's been **run on real silicon**
vs. what's only been **reviewed** and still needs a card. AV1 is the only output
codec (4:2:0, Main profile, 8- or 10-bit).

| Vendor | Feature | Decode | Encode (AV1) |
|--------|---------|--------|--------------|
| Intel  | `qsv`   | ✅ verified | ✅ verified |
| NVIDIA | `nvidia`| ✅ verified | ⚠ by-review |
| AMD    | `amd`   | ⚠ by-review | ⚠ by-review |
| FFmpeg | `ffmpeg`| ✅ (reference) | ✅ software |

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

**Done (2026-06-29, on the RTX 3090 + Ryzen 9700X box):**
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

> The only AMD silicon on hand is the **Ryzen 9700X desktop iGPU, which is not
> AMF-capable** — `InitDX11` returns `AMF_NOT_FOUND` for it (the encode probe fails
> too). So the per-frame decode loop still can't be run here; it needs a discrete
> Radeon (RDNA) or a supported APU.

> Expect the same class of struct-layout / init-flow surprises QSV had on first
> real hardware. QSV needed: every mfx struct offsetof-verified, the MFXLoad
> dispatcher (not legacy init), an advisory Query (proceed to Init on the
> driver's spurious `-3`), LowPower=ON, and a frame-sized output buffer. Budget
> for an equivalent debugging pass on AMF.

Verify on RDNA-class silicon (RX 7000+ for AV1 encode):
- [ ] **AMF decode pixels** — H.264 / HEVC / AV1 produce correct frames. The
      `SubmitInput`→`QueryOutput`→readback loop, the `AMF_IID_SURFACE` GUID, and
      the host-memory `Convert` slot are still best-guess; compare a frame hash
      against `ffmpeg`. (Detection + adapter routing + init/teardown are done.)
- [ ] **AMF encode** — AV1 8-bit and 10-bit (P010) end-to-end, correct pixels.

---

## FFmpeg — `ffmpeg` (optional, cross-vendor fallback)

libavcodec as the decode catalogue (incl. ProRes) + software/hwaccel + AV1
software encode. Needs FFmpeg ≥7.0 dev libs + LLVM/libclang. It's the reference
implementation, so no hardware verification is owed — it's the safety net when a
vendor's hand-rolled path isn't available or proves unreliable.

---

## Filters — denoise

The spatial denoise family is implemented (`codec::filter`, `denoise=METHOD:STRENGTH`):
**bilateral, gaussian, median, mean, nlmeans, anisotropic** — selectable, 8-bit,
unit-tested + verified end-to-end (720p, 30 fps): mean/gaussian ≈ baseline,
median/bilateral fast, anisotropic ~0.09 s/frame, nlmeans ~0.84 s/frame
(offline-only). See [docs/filters/denoise.md](docs/filters/denoise.md).

Follow-ups:
- [ ] **Deep denoise — DPIR** ([cszn/DPIR](https://github.com/cszn/DPIR), DRUNet):
      a `denoise=dpir` method running the DRUNet CNN via ONNX (`tract` pure-Rust
      CPU, or `ort` for CUDA/DirectML GPU). Export the model to ONNX once + vendor
      it (~32 MB, takes a σ noise-level channel ← STRENGTH); load it in
      `FilterChain::prepare` (resource-filter pattern, like `overlay`); luma-only
      `drunet_gray` first, full YUV→RGB→DRUNet→YUV colour as a refinement.
      GPU-bound, opt-in, offline. A self-contained sprint (ML dep + model asset).
- [ ] **Temporal denoise** (hqdn3d / NLM-temporal) — needs per-stream frame
      history, which the stateless `Arc<FilterChain>` doesn't carry today.
- [ ] **AVX2 denoise kernels** — the bilateral / nlmeans inner loops are the
      perf-sensitive ones; mirror the existing AVX2 colorspace/scale dispatch.

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
