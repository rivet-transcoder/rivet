# rivet-h26x

Native **H.264/AVC** and **H.265/HEVC** decoders for the
**[rivet](https://crates.io/crates/rivet-transcoder)** transcoder: pure Rust,
no C, no system libraries. They are the software decode tier for the two codecs
every camera, phone and broadcast chain emits, sitting under the GPU decoders
(NVDEC / AMF / QSV) so a machine without a usable GPU still decodes.

Published as `rivet-h26x`; **imported as `h26x`** (`use h26x::…`). This is an
internal crate of the rivet project — see the
**[rivet-transcoder](https://crates.io/crates/rivet-transcoder)** crate and the
[repository](https://github.com/rivet-transcoder/rivet) for the full architecture.

## What it decodes

| | supported | refused with `Error::Unsupported` |
|---|---|---|
| **H.264** | Baseline / Main / High: progressive frames, 8-bit 4:2:0, CAVLC + CABAC, I/P/B, spatial + temporal direct, explicit + implicit weighting, 8x8 transform, PCM, MMCO, multi-slice, frame-num gaps, all three POC types, deblocking, VUI reorder hints | interlaced (field / MBAFF), 4:2:2 / 4:4:4, > 8-bit, FMO / ASO, data partitioning, SP / SI |
| **H.265** | Main / Main 10 / Main 12 and the format range extensions (4:0:0 / 4:2:0 / 4:2:2 / 4:4:4, 8–12-bit): CTB 16–64, AMP, transform skip (any size, rotation, single-context), scaling lists, sign hiding, PCM, `cu_transquant_bypass` (lossless), cu_qp_delta, chroma QP offset lists, cross-component prediction, implicit / explicit RDPCM, persistent Rice adaptation, high-precision weighted-prediction offsets, intra smoothing disabling, tiles, WPP (and both together), dependent slice segments, merge / AMVP / TMVP, explicit weighting, deblocking, SAO, long-term references, CRA / BLA / RASL handling, `pic_output_flag`, `no_output_of_prior_pics`, decoded-picture-hash SEI verification (`H26X_VERIFY_HASH=1`) | unequal luma/chroma bit depth, > 12-bit, extended precision processing, CABAC bypass alignment, separate colour planes, SCC, multi-layer |

Both decoders are **bit-exact**. H.264 passes **101 of the 101** JVT
conformance bitstreams (AVCv1 + FRExt) it does not refuse, against the suite's
reconstructed YUV — the other 102 are refused up front (interlaced 77, 4:2:2 17,
4:0:0 2, 10-bit 2, FMO 3, SP/SI 1) — and matches libavcodec on the workspace
fixtures (851 frames across CAVLC/CABAC, B-pyramids, weighting, 8x8, slices,
CQM). H.265 passes **146 of the 147** JCT-VC HEVC_v1 conformance bitstreams
against the suite's own MD5s (the one exception is the unequal-bit-depth
stream, which is refused) and **31 of the 31** RExt bitstreams it accepts (the
other 18 are refused up front: 16-bit, extended precision, CABAC bypass
alignment, unequal bit depths — the same set libavcodec declines), plus
fifteen x265 feature fixtures; every HM stream's decoded-picture-hash SEI is
checked as well.

`Unsupported` is a *classification*, not a failure: rivet's decode tier list
falls through to the next backend (libavcodec when built with the `ffmpeg`
feature, openh264 for H.264).

## Threading and SIMD

Both decoders are threaded two ways, on one FIFO worker pool sized to the
machine (`H26X_THREADS`, default every core, up to 32):

- **Frame threading.** Pictures decode concurrently; a picture that references
  another waits only for the rows it needs (motion compensation waits on the
  reference's *filtered* row progress, temporal motion prediction on its
  *decoded* row progress), and the deblocking / SAO of a picture trail its
  decode row by row so the next picture can start before this one ends. Up to
  `H26X_INFLIGHT` pictures are in flight (default `threads.clamp(2, 16)`).
- **Intra-picture parallelism (H.265).** Wavefront rows (WPP), tiles and slice
  segments each run as their own task; a row waits on the CTB above-right, and
  the next row is spawned as soon as this one is two CTBs in. Dependent slice
  segments resume their predecessor's CABAC state.

The pixel kernels — interpolation, weighting, inverse transforms, residual add,
SAO for H.265; the sixteen quarter-sample positions, chroma bilinear, averaging
and weighting for H.264 — sit behind a function table filled at run time from
what the CPU has: **AVX2** on x86-64, **NEON** on AArch64, scalar otherwise
(`H26X_NO_SIMD=1` forces scalar). Every SIMD kernel is checked bit-exact against
the scalar reference by the crate's tests, on both architectures in CI.
`H26X_PROF=1` prints where the time went.

**Speed** (2026-08-18, one Ryzen 5 5600X 6C/12T box, 720p Big Buck Bunny
clips, whole-file decode into output pictures; libavcodec 8.0 writing raw
video to /dev/null in brackets): H.264 High/CAVLC 241 frames — 1 thread
0.70 s [0.37], 12 threads 0.17 s [0.14]. HEVC Main with WPP, 120 frames —
0.28 s [0.22], 0.10 s [0.13]. HEVC Main without WPP, 241 frames — 0.72 s
[0.56], 0.31 s [0.28]. Single-threaded it is within 1.3–1.9x of libavcodec;
with threads it is at parity, and ahead where wavefront rows give it
intra-picture parallelism. 8-bit HEVC streams decode into 8-bit planes with
their own AVX2 / NEON kernels (byte-wide taps, one fused pass from reference
bytes to picture bytes for default-weighted prediction); the remaining
single-thread gap is per-block bookkeeping and entropy decoding, where the
decoder executes about half again as many instructions as libavcodec.

## Provenance and licensing

This is not a translation of libavcodec's C, which is LGPL and could not be
carried under this crate's license. The decoders were written from the ITU-T
Recommendations (H.264 08/2024, H.265 V11 01/2026) — the decoding process is
fully specified there — using libavcodec's *architecture* as the model:
parameter-set tables, a slice-driven decode loop over a decoded-picture buffer,
entropy decoding into per-block syntax, then prediction, inverse transform,
reconstruction and in-loop filters, with the pixel kernels behind a
runtime-dispatched DSP layer. Numeric tables (VLC codes, CABAC initialisation
values, scan orders, filter taps) are the standard's tables.

**Patents.** The AVC and HEVC patent pools (Via LA, Access Advance, Velos)
license *products and services* — encoders, decoders, and content shipped to
end users — not the act of writing or publishing an implementation; several
pools additionally waive royalties for application software distributed at no
charge. Nothing here is a licence to any patent, and anyone shipping a product
built on this crate is responsible for their own patent position, exactly as
with x264, x265, OpenH264 or FFmpeg. Source availability was never the
licensed act; distribution of a decoding product is.

## Using it

```rust
let mut dec = h26x::hevc::HevcDecoder::new();   // or h26x::h264::H264Decoder
dec.push_annexb(&bytes)?;                        // whole NAL units, Annex-B framed
while let Some(pic) = dec.next_picture() { /* h26x::Picture, output order */ }
dec.flush()?;                                    // drain the reorder buffer
```

`examples/h26xdec.rs` decodes a raw `.264` / `.265` file and prints one line per
output frame with the MD5 of the packed planar picture — the same layout
`ffmpeg -f framemd5` hashes — or writes raw YUV. Debug environment variables:
`H26X_NO_DEBLOCK`, `H26X_NO_SAO`, `H26X_TRACE=<mbaddr>` / `H26X_TRACE_DPB`
(H.264), `H26X_TRACE_PS` (parsed SPS/PPS), `H26X_TRACE_CU`, `H26X_TRACE_PU=x,y`,
`H26X_TRACE_TB=c,x,y`, `H26X_VERIFY_HASH=1` (check every decoded-picture-hash
SEI, count a warning per mismatch) (H.265).

## License

Open Encoding Attribution License v1.0 — a source-available (not OSI open-source)
license, royalty-free, with a commercial-attribution requirement. See
[LICENSE.md](LICENSE.md) and [NOTICE](NOTICE).
