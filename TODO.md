# rivet — TODO / hardware-verification backlog

## AMD (AMF) + Intel (QSV) hardware **decode** — verify on real silicon

Status: **implemented as hand-rolled FFI, verified-by-review only.** Neither was
testable on the dev box (RTX 3090 + Ryzen iGPU — no AMD RDNA3+ discrete, no Intel
Arc). Both are our own FFI mirrors of the vendor SDK headers (no shiguredo code,
no attribution owed), modeled on the in-tree encoders (`encode/amf.rs`,
`encode/qsv.rs`) + the AMD AMF / Intel oneVPL decode APIs.

When the **Intel Arc** and **AMD RDNA-class** cards arrive, verify:

### AMD — `decode/amf_dec.rs` (AMF decode)
- [ ] H.264 / HEVC / AV1 decode produces correct pixels (luma spread non-flat;
      compare a frame hash against ffmpeg).
- [ ] **`AMF_IID_SURFACE` GUID** — `QueryOutput` returns `AMFData`; we downcast
      to `AMFSurface` via `QueryInterface(AMF_IID_SURFACE)`. The GUID bytes are a
      **best guess** from the AMF SDK `core/Surface.h` — confirm against the
      installed SDK header; a wrong IID makes every `QueryOutput` fail.
- [ ] **Extradata / SPS-PPS**: H.264/HEVC AMF decoders may need
      `AMF_VIDEO_DECODER_EXTRADATA` set before `Init`. We currently rely on
      in-band parameter sets (Annex-B). Confirm whether MP4-sourced streams
      (which carry SPS/PPS out-of-band) need the extradata property set.
- [ ] **P010 / 10-bit** output path (HEVC Main10) → `Yuv420p10le` deinterleave.
- [ ] Drain (`Drain` + `QueryOutput` until `AMF_EOF`) flushes all frames.
- [ ] Multi-AMD adapter routing (AMF init picks adapter 0 unconditionally).

### Intel — `decode/qsv_dec.rs` (oneVPL decode)
- [ ] H.264 / HEVC / AV1 / VP9 decode produces correct pixels.
- [ ] **`MFXVideoDECODE_DecodeHeader`** correctly parses the bitstream header
      into `mfxVideoParam` (we feed the first sample(s) and retry on
      `MFX_ERR_MORE_DATA`).
- [ ] Work-surface pool sizing from `MFXVideoDECODE_QueryIOSurf`
      (`Suggested` count) — we currently allocate a fixed pool; confirm it's
      enough for the stream's DPB depth.
- [ ] **P010 / 10-bit** output (HEVC Main10 / VP9 Profile 2) with the `Shift`
      handling on read-back.
- [ ] Drain (null bitstream `DecodeFrameAsync` until `MFX_ERR_MORE_DATA`).
- [ ] DRM render-node selection on multi-Intel hosts (the hand-rolled QSV
      encoder picks the implementation via the dispatcher; decode should match).

### Both
- [ ] Wire into `create_decoder` dispatch (done — AMD/Intel branches restored,
      gated behind `amd` / `qsv`).
- [ ] Confirm `cargo build --features amd` / `--features qsv` on a Linux host
      with the vendor runtime present, then end-to-end decode→AV1-encode.
- [ ] If a path proves unreliable, the `ffmpeg` decode feature remains the
      fallback for that vendor.

## Other

- [ ] NVENC AV1 encode end-to-end on Ada+ silicon (RTX 4000+) — the dev box is
      Ampere (no AV1 encode); the NVENC path + the 10-bit P010 path are
      verified-by-review. The capability query correctly rejects AV1 on the 3090.
- [ ] AMF / QSV AV1 **encode** end-to-end on RDNA3+ / Arc — verified-by-review.
- [ ] NVENC resolution / 10-bit caps rejection branches (only the AV1-support
      gate is hardware-proven on the 3090; the max-dim / 10-bit branches aren't).
