# Design decisions — the *why*

This catalogs the load-bearing decisions in rivet: **what** was decided, **why**
it's needed, and **where** it lives. Most of the codebase's shape follows from
these — if a piece of code looks odd, the reason is usually here. For the
structure see [architecture.md](architecture.md); for the flow see
[pipeline.md](pipeline.md).

---

## Output policy

### 1. AV1 is the default output video codec (H.264/H.265 opt-in)
**Decision.** Jobs output **AV1** video by default + **Opus/AAC** audio in
**MP4**. **H.264 and H.265** are also supported output codecs (opt-in) for
legacy-player compatibility. The `VideoCodec` enum has variants for AV1
(default), H.264, and H.265.

**Why.** Royalty position. AV1 + Opus + MP4 carries **zero codec-royalty
exposure** on the output: AV1 and Opus are royalty-free, the MP4 (ISO-BMFF)
container is itself royalty-free, and AAC *passthrough* is not a licensed
activity (we transmit bytes, we don't encode/decode AAC). AV1 was the original
**locked** target precisely for this reason, and it remains the **royalty-clean
default** — any job that doesn't explicitly request otherwise gets AV1.

**Subsequently added.** H.264 and H.265 were added as opt-in output codecs for
legacy-player compatibility. They knowingly carry the patent-licensing
obligations AV1 was chosen to avoid, so they are an explicit per-job opt-in, not
the default. The framing is **AV1-first / AV1-default**, not AV1-only —
suggesting H.264/H.265 as a *replacement* for the AV1 default is still wrong by
construction.

**Caveat (tracked).** AV1's "royalty-free" claim should be revisited "when we
have 100,000 users" (Dolby AV1 suit + Sysvel pool claims are open industry
issues); SVT-AV1 is a noted future encoder candidate. Not actionable now.

**Where.** `VideoCodec` in [`spec.rs`](../crates/rivet/src/spec.rs); the
audio routing (passthrough vs Opus) in [`transcode.rs`](../crates/rivet/src/transcode.rs)
and [`codec/audio/`](../crates/codec/src/audio/).

### 2. Audio: passthrough what's clean, transcode the rest to Opus, drop the unplayable
**Decision.** AAC / Opus / AC-3 / E-AC-3 pass through verbatim; MP3 / Vorbis are
transcoded to Opus; anything else is dropped (video-only) with a warning.

**Why.** Passthrough avoids re-encoding (quality + royalty cleanliness). Opus is
the royalty-free transcode target and plays in MP4 on modern Apple + browsers.
Adding an AAC *encoder* (e.g. `fdk-aac`) was rejected — it reintroduces a
Fraunhofer license, and silently dropping AAC sources would be worse than
passthrough. See [decisions.md §1].

---

## No FFmpeg by default; clean-room + hand-rolled FFI

### 3. The demuxers and muxers are hand-written clean-room parsers
**Decision.** MP4/MOV/MKV/WebM/TS/AVI demux and MP4 / CMAF / HLS mux are
all hand-written in the [`container`](../crates/container/) crate. No FFmpeg, no
container library.

**Why.** Licensing independence (FFmpeg is LGPL/GPL), full control over the exact
bytes we emit (faststart, Apple brand sets, HDR atoms, segment alignment), and a
build that has **no FFmpeg prerequisite**. The cost — reimplementing parsers — is
paid once and bought back in deployment simplicity and output correctness.

**Where.** [container.md](container.md); the box writers in
[`mux.rs`](../crates/container/src/mux.rs) / [`cmaf.rs`](../crates/container/src/cmaf.rs).

### 4. GPU codec backends are hand-rolled `dlopen` FFI mirroring the vendor SDK headers
**Decision.** NVENC/NVDEC, AMF, and QSV (oneVPL) are reached through our own FFI
that mirrors the vendor C structs, loaded at runtime with `libloading`. No
external wrapper crate.

**Why.** (a) Cross-platform: this builds on **Windows MSVC + Linux**, where the
obvious wrapper (shiguredo_vpl) does not. (b) Runtime `dlopen` means **one binary
runs whether or not the GPU libraries are present** on the host — it engages the
GPU when the driver is there, with no link-time dependency on it. (c) We control
the exact ABI. (Note: the *codec* paths are GPU-only as built — see §5; the
`dlopen` boundary is about not link-depending on driver libs, not a CPU fallback.)

**The ABI hazard, and the guard.** Mirroring C structs by hand is fragile: a
wrong offset silently corrupts a neighbouring field. So the FFI structs carry
`const_assert!` **size/offset witnesses** verified against the real installed
headers (e.g. [`qsv_ffi.rs`](../crates/codec/src/qsv_ffi.rs) — every mfx
struct is offsetof-checked; the per-codec NVDEC pic-params have shape witnesses).
A future SDK that changes a layout fails the build instead of producing garbage
at runtime.

**Why the `*_stub.rs` files.** Each HW backend has a stub sibling
(`nvenc_stub.rs`, `amf_stub.rs`, `qsv_stub.rs`) compiled when that vendor's
Cargo feature is off, so the dispatch code always type-checks and a
default/cross-vendor build still compiles. See [codec-decode.md](codec-decode.md)
and [codec-encode.md](codec-encode.md).

### 5. Codecs are GPU-only as built; FFmpeg is an optional *encode* tier
**Decision.** The default build is **hardware-only**:
- **Decode** ([`decode/mod.rs`](../crates/codec/src/decode/mod.rs)
  `create_decoder`) tries **NVDEC → AMF → QSV** for the detected GPU and
  **hard-fails** if none matches. There is no CPU decoder, and although a
  `FfmpegDecoder` type exists in `decode/ffmpeg.rs`, it is **not wired into the
  factory** (only its own tests construct it). The in-code comment is explicit:
  "CPU decoders were removed per the GPU-only directive."
- **Encode** ([`encode/mod.rs`](../crates/codec/src/encode/mod.rs)
  `select_encoder`) tries a **FFmpeg AV1 encoder first** *only when* the `ffmpeg`
  feature is built and `DISABLE_FFMPEG` is unset (libavcodec's
  av1_nvenc/av1_qsv/libsvtav1/libaom probe chain — this is the **only software
  encode path**), then the hand-rolled **NVENC → AMF → QSV** backends. There is
  **no native rav1e CPU fallback** (removed per the 2026-05-08 GPU-only
  directive); a pinned-vendor init failure is a hard error.

**Why GPU-only.** The production target is GPU hosts; a silent CPU fallback would
mask a misconfigured GPU as a slow-but-working job. Failing fast surfaces the real
driver error on the job's failed event instead.

**Why FFmpeg is optional, not default.** It's the reference implementation and a
useful safety net (software encode, ProRes, exotic inputs) — but making it
mandatory would impose its build prerequisites (FFmpeg ≥7 dev libs + LLVM/libclang)
and license posture on every build, so it stays behind `--features ffmpeg`.

> **Doc-vs-code drift.** Several module-header comments and the README still
> describe FFmpeg as the *primary decode* path and rav1e as a CPU encode
> fallback. Neither is wired in the current factory — the description above is the
> as-built behavior. See the maintainer notes in
> [codec-decode.md](codec-decode.md) and [codec-encode.md](codec-encode.md).

---

## GPU scheduling — the rung benefit

### 6. Decode the source once and fan out to every rendition
**Decision.** A job has **one** decode pump; decoded frames are cloned (cheap,
`Arc`-backed) to every rung's scaler.

**Why.** The naïve `ffmpeg`-per-rung approach decodes the input N times for an
N-rung ladder. Decoding once and fanning out turns that into a single decode —
the dominant saving on a ladder. See
[`decode_pump.rs`](../crates/rivet/src/decode_pump.rs) and
[pipeline.md](pipeline.md).

### 7. One encoder per GPU, enforced by a lease pool
**Decision.** A process-wide [`GpuPool`](../crates/rivet/src/gpu_pool.rs) hands
out one `GpuLease` per GPU; an encoder worker holds it for its lifetime. Encoders
run in parallel *across* GPUs, never two on one GPU.

**Why.** Empirically (2026-05-02), concurrent NVENC sessions on the same CUDA
context **deadlocked at ~session 5/5 init** — the GPU went idle and no frames
encoded. One-encoder-per-GPU is the invariant that avoids it; the pool's job is
to enforce it while still parallelizing across devices. On CPU-only hosts
`claim()` returns `None` and callers fall back to CPU without queuing.

### 8. Mid-flight helper dispatch + a cross-vendor codec invariant
**Decision.** When a fast rung releases its lease early, a **helper dispatcher**
grabs the freed lease and attaches an extra encoder worker to a still-busy rung.
Helpers may land on a different GPU **vendor**; a per-rung `RungCodecInvariant`
guarantees every contributed segment shares the same codec-config contract
(`av1C` for AV1, `avcC`/`hvcC` for H.264/H.265).

**Why.** Without helper dispatch, a slow rung leaves fast GPUs idle and throughput
is bounded by the slowest rung. With it, freed GPUs pick up the slow rung's
chunks and throughput scales close to linearly with GPU count. The codec
invariant is what makes a mixed NVENC+QSV contribution to one rendition still
decode cleanly. See [`multigpu.rs`](../crates/rivet/src/multigpu.rs).

### 9. Single-file output on multiple GPUs is chunk-and-stitch
**Decision.** A single MP4 on multiple GPUs is encoded as independent IDR-led GOP
chunks across the GPUs, then stitched. `ChunkSeamMode` (`Parallel` /
`ParallelConstQp` / `Serial`) trades seam quality for speed.

**Why.** It lets the same reactive engine accelerate a single-file job, not just
a ladder. Each chunk is an independent GOP so the result always plays; the seam
mode exists because per-chunk VBR (NVENC) can step quality at the ~2 s seams —
`ParallelConstQp` flattens that, `Serial` removes seams entirely (one encoder).

---

## Streaming & memory

### 10. Demux streams one sample at a time
**Decision.** `container::streaming::demux_streaming` yields one video sample per
call instead of materializing the whole file; the pipeline pulls → decodes →
fans out → frees.

**Why.** A 15-minute 1080p60 source would otherwise materialize gigabytes in RAM.
Streaming keeps **peak RSS low** (the migration measured roughly a 500× reduction
vs. the materialize-everything projection). The bounded
[`SegmentChunkQueue`](../crates/rivet/src/frame_queue.rs) is the back-pressure
point: the pump blocks when the queue is full, the slowest rung throttles the
rest. See [container.md](container.md) and [engine.md](engine.md).

### 11. NVDEC decode is incremental, not buffer-everything
**Decision.** The NVDEC path drives `cuvidParseVideoData` once per pushed sample
and pops one frame per call, rather than accumulating all decoded surfaces.

**Why.** Buffering every decoded NV12/P016 surface for a long source projected
hundreds of GiB. Incremental parse keeps NVDEC inside the same streaming RSS
budget as the CPU paths. See [codec-decode.md](codec-decode.md).

---

## Color & HDR

### 12. HDR is tonemapped to SDR by policy (single output)
**Decision.** Every HDR source is tonemapped to 8-bit BT.709 SDR at transcode
time; the output ladder is single-flavor SDR. No HDR output, no parallel HDR
rendition — by default.

**Why.** Most UGC "HDR" is captured accidentally: iOS records HLG ~1 stop bright
expecting Apple's tonemapper to bring it down by viewing conditions (which fails
the moment the file leaves Apple); Samsung writes HLG with a viewing-condition
variable nearly every conversion drops. YouTube/Meta have given talks on the
policy + UI work needed to make HDR feeds tolerable — work inappropriate for our
scale. Shipping native HDR without it lands eye-searing / washed-out clips on
viewers. Tonemapping at upload normalizes this on our side. The tonemap is in
[`tonemap.rs`](../crates/codec/src/tonemap.rs); the dispatch in
[`colorspace.rs`](../crates/codec/src/colorspace.rs).

**Escape hatch retained.** The 10-bit pipeline, the HDR mux atoms
(`mdcv`/`clli`), HW 10-bit encode, and HDR metadata extraction all remain in tree
as latent paths — a future creator-opt-in HDR-output mode re-engages them by
routing on `creator_opted_in && is_hdr` instead of `is_hdr` alone.

### 13. AV1 needs 16-multiple coded dimensions; pad with neutral black, not zeros
**Decision.** Coded frame dimensions are rounded up to a multiple of 16 (e.g.
572×240 encodes at 576×240) and the scratch NV12/P010 buffer is pre-filled
**neutral black** (Y=16, Cb/Cr=128; 10-bit `<<6`) before the content copy.

**Why.** AV1's quantization works on 16-aligned blocks, so odd aspect ratios need
padding. Most implementations (and ffmpeg) **zero-fill** the scratch buffer —
and a browser decoding NV12 zeros as BT.709 limited-range renders the padding as
distinctive **green bars**. A neutral-black fill makes the padding black instead.
See [codec-encode.md](codec-encode.md).

---

## Web-ready output

### 14. Defaults that "just play" in a browser
**Decision.** Faststart MP4 (moov before mdat), segment-aligned CMAF/HLS for ABR,
`colr nclx` color tagging, AV1 **Main** profile 4:2:0, AAC/Opus audio, and an
Apple-friendly `ftyp` brand set (`av01`/`iso6`/`mp42`).

**Why.** "Optimized for web" is a pile of choices FFmpeg leaves to the caller.
Faststart lets a clip start playing before it's fully downloaded; segment
alignment across the ladder lets hls.js switch renditions cleanly; `colr` stops
QuickTime/iOS Safari silently applying BT.709-limited fallback (which breaks
non-709 sources); the brand set is what iOS Safari needs to accept the
largesize/co64 path. See [container.md](container.md).

### 15. `co64` / `mdat` largesize auto-upgrade for >4 GiB outputs
**Decision.** The MP4 muxer auto-upgrades `stco`→`co64` and the `mdat` short
header → 64-bit largesize when the payload would exceed `u32::MAX`.

**Why.** A large/long output exceeds 32-bit box offsets; without the upgrade the
chunk offsets wrap and the file is corrupt. Both fire together past 4 GiB.

---

## One definition for every front-end

### 16. CLI, HTTP, and IPC share one `TranscodeSettings`
**Decision.** The CLI flags, the HTTP JSON/query spec, and the IPC `#rivet`
header are thin adapters over one canonical
[`TranscodeSettings`](../crates/rivet/src/settings.rs) with a single
`into_spec()` builder and one set of `parse_*` string parsers.

**Why.** Before this, the spec-building logic existed **three times** (the server's
`build_spec`, the CLI's `resolve_rungs`, the IPC's `JobSettings`) and a new option
meant editing all three. Now an option is a one-place change and the three
surfaces map 1:1. See [engine.md](engine.md#front-ends) and
[output-spec.md](output-spec.md).

### 17. The IPC socket is opt-in; stdin/stdout piping is always on
**Decision.** `rivet ipc` (Unix-domain socket server) is behind the `ipc` Cargo
feature; `rivet pipe` (stdin→stdout streaming) needs no feature.

**Why.** The socket server is a specialized deployment surface (Unix-only at
runtime), so it shouldn't be in every build; piping is the universal,
cross-platform streaming path and stays available everywhere.

### 18. File-path I/O on the HTTP API is sandboxable
**Decision.** The JSON API can read an input and write an output by **server file
path** (no upload/download); `RIVET_FILE_ROOT`, when set, confines those paths to
a directory.

**Why.** Pointing at a shared filesystem avoids streaming large media over HTTP.
Reading/writing arbitrary server paths is a real LFI/arbitrary-write risk, so the
sandbox env var exists; the server also binds localhost by default (trusted-local
posture). See [api.md](api.md) and [engine.md](engine.md).

---

## Conventions

### 19. Deleted scaffolds, not "kept for reference"
When a vendored library replaces a hand-rolled scaffold, the scaffold is
**deleted**. Dead code that mimics a real path (e.g. a stub returning grey
pixels) is a misleading diagnostic surface, so it's removed rather than retained.

### 20. No forking external crates — wrap in-repo
A missing capability in a dependency is solved by wrapping its raw FFI **in this
repo**, not by forking/patching the upstream crate. (This is why the GPU FFI is
hand-rolled rather than a patched wrapper — see §4.)
