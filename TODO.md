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
`encode/amf.rs`). **Both decode and encode are by-review only — no AMD card on
either box yet.**

> Expect the same class of struct-layout / init-flow surprises QSV had on first
> real hardware. QSV needed: every mfx struct offsetof-verified, the MFXLoad
> dispatcher (not legacy init), an advisory Query (proceed to Init on the
> driver's spurious `-3`), LowPower=ON, and a frame-sized output buffer. Budget
> for an equivalent debugging pass on AMF.

Verify on RDNA-class silicon (RX 7000+ for AV1 encode):
- [ ] **AMF decode** — H.264 / HEVC / AV1 produce correct pixels. The
      `AMF_IID_SURFACE` GUID and the host-memory surface read-back are best-guess;
      compare a frame hash against `ffmpeg`.
- [ ] **AMF encode** — AV1 8-bit and 10-bit (P010) end-to-end, correct pixels.

---

## FFmpeg — `ffmpeg` (optional, cross-vendor fallback)

libavcodec as the decode catalogue (incl. ProRes) + software/hwaccel + AV1
software encode. Needs FFmpeg ≥7.0 dev libs + LLVM/libclang. It's the reference
implementation, so no hardware verification is owed — it's the safety net when a
vendor's hand-rolled path isn't available or proves unreliable.
