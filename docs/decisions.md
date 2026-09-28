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
**Decision.** AAC / Opus / AC-3 / E-AC-3 / DTS pass through verbatim, and so does
MP3 into a single-file MP4; Vorbis, MP2, PCM (and MP3 for HLS) are transcoded to
Opus; anything else is dropped (video-only) with a warning. MP3 is also an
output (`audio=mp3`, §21), and the output channel layout is a knob of its own
(§22).

**Why.** Passthrough avoids re-encoding (quality + royalty cleanliness). Opus is
the royalty-free transcode target and plays in MP4 on modern Apple + browsers.
Adding an AAC *encoder* (e.g. `fdk-aac`) was rejected — it reintroduces a
Fraunhofer license, and silently dropping AAC sources would be worse than
passthrough. There is no AAC *decoder* yet (TODO.md: blocked on a lawful source
for its tables), which is why an AAC track can be passed through but not
downmixed. MP3 joined the passthrough set
in 2026-09: every browser plays MP3 in an MP4, and re-encoding a lossy track to
another lossy codec only loses quality. CMAF has no MP3 profile, so an HLS
package still transcodes it. See [decisions.md §1].

---

## No FFmpeg, in any capacity; clean-room + hand-rolled FFI

### 3. The demuxers and muxers are hand-written clean-room parsers
**Decision.** MP4/MOV/MKV/WebM/TS/AVI demux and MP4 / CMAF / HLS mux are
all hand-written in the [`container`](../crates/container/) crate. No FFmpeg, no
container library — and, since 2026-08-12, no FFmpeg anywhere else in the
workspace either (see [No FFmpeg](../README.md#no-ffmpeg)).

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

### 5. Codecs are GPU-first as built; software AV1 is an opt-in floor
**Decision.** The default build is **hardware-only**, and the software tier is
opt-in and last:
- **Decode** ([`decode/mod.rs`](../crates/codec/src/decode/mod.rs)
  `create_decoder`) tries **NVDEC → AMF → QSV** for the detected GPU, then
  **software AV1** when built with `rav1d-fallback`, and **hard-fails** if none
  matches. The software decoder handles **AV1 8-bit 4:2:0 and nothing else**.
- **Encode** ([`encode/mod.rs`](../crates/codec/src/encode/mod.rs)
  `select_encoder`) tries the hand-rolled **NVENC → AMF → QSV** backends, then
  **software AV1** via rav1e when built with `rav1e-fallback` (8-bit 4:2:0). A
  *pinned*-vendor init failure stays a hard error — a lease that named a GPU
  means the caller wanted that GPU, and quietly serving it from the CPU would
  make a broken driver look like a slow one.

**Why GPU-first.** The production target is GPU hosts; a silent CPU fallback
would mask a misconfigured GPU as a slow-but-working job. That is why the
software tier is opt-in *and* sits below the vendor chain rather than above it:
a build that has it still prefers silicon, and a host that lacks the feature
still fails fast with the real driver error on the job's failed event.

**Why software AV1 is pure Rust.** rav1e and rav1d are ordinary cargo
dependencies — no system libraries, no bindgen, no LLVM, nothing the deployment
image has to ship. That is the whole reason they could be made a default-off
feature instead of a build-environment decision; see [No
FFmpeg](../README.md#no-ffmpeg) for the tier they replaced.

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

### 8. Ladder workers serve every rung, and the decode is split across the cards
**Decision.** For an HLS ladder, one worker per GPU holds its lease for the whole
job and takes the next chunk of whichever rung is furthest behind. The source is
decoded once, and — for an un-spliced H.264/H.265 source — cut into ranges at
segment-aligned keyframes with one decode pump per card. Cards of different
**vendors** serve the same rung; a per-rung `RungCodecInvariant` guarantees every
contributed segment shares the same codec-config contract (`av1C` for AV1,
`avcC`/`hvcC` for H.264/H.265).

**Why.** The previous shape — one worker per rung plus a helper dispatcher
attaching extra workers to a busy rung whenever a lease freed — still let a card
idle while work existed: its rung was blocked and another rung's queued chunks
were not its to take. It also capped the rungs in flight at the GPU count, so a
longer ladder fell back to decoding the source once per rung, and decode is the
dominant cost (a 1080p rung costs ~15% more than a 240p one despite twenty times
the pixels). Serving the whole ladder from every card means a card idles only
when the job is out of work and the ladder costs one decode at any depth;
splitting that one decode across the cards removes the last single-card ceiling.
Furthest-behind rather than cheapest-first because the shared pump stalls when
any queue fills. Measured faster in the service this engine was extracted from;
that is the whole reason it replaced the helper shape rather than joining it.
Single-file (chunk-and-stitch of one rendition) keeps the helper dispatcher. See
[`multigpu/hls.rs`](../crates/rivet/src/multigpu/hls.rs).

### 9. Single-file output on multiple GPUs is chunk-and-stitch
**Decision.** A single MP4 on multiple GPUs is encoded as independent IDR-led GOP
chunks across the GPUs, then stitched. `ChunkSeamMode` (`Parallel` /
`ParallelConstQp`) trades seam quality for speed; no seams at all is
`EncodePolicy::SingleGpu` (one encoder per rung), not a seam mode.

**Why.** It lets the same ladder engine accelerate a single-file job, not just
a ladder. Each chunk is an independent GOP so the result always plays; the seam
mode exists because per-chunk VBR (NVENC) can step quality at the chunk seams —
`ParallelConstQp` flattens that. There used to be a `Serial` seam mode that
quietly turned a multi-GPU job serial; that was two knobs for one question, and
the same conflict existed between `DecodePolicy` (which card) and a separate
decode-split knob. Each question now has exactly one enum: `DecodePolicy` is
the whole decode plan (split / whole / pinned / fastest / N ranges) and
`EncodePolicy` is the whole encode plan (all cards ladder-scheduled / all cards
pinned per rung / a family / one card serial), so no two settings can
contradict each other.

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

---

## Audio output

### 21. MP3 output: LAME, loaded at run time, behind the `lame` feature
**Decision.** `audio=mp3` encodes constant-bitrate MPEG-1 Layer III with
**LAME**, found at run time with `dlopen` (`libmp3lame.so.0` /
`libmp3lame.dylib` / `libmp3lame.dll`, or `RIVET_LAME_LIBRARY`) and compiled in
only with the **`lame`** cargo feature, off by default. Without the feature,
`audio=mp3` is refused by `validate()`; with it, a host without the library
fails the first MP3 encode saying what to install. MP3 *decode* (minimp3, MIT),
MP3 *passthrough* and the MP3 muxing need no feature.

**Why this encoder.** The order of preference was a permissively licensed
encoder, then LAME loaded dynamically, then a statically linked LGPL crate. The
permissive candidates were measured before being passed over:

- **oxideav-mp3 0.1.3** (MIT, pure Rust, on crates.io). On a 20 s stereo test
  signal (tones, a vibrato, decaying partials over pink noise) at 128 kbit/s it
  scored **17.7 dB SNR against LAME's 25.3 dB**, and at 256 kbit/s **18.4 dB
  against 46.4 dB** — it did not improve with bitrate — and kept **1–10 % of
  the energy above 11 kHz** where LAME keeps 57–83 %: audibly muffled. Its
  quality presets changed neither number. It encoded at **about 2× real time**
  (9–13 s for 20 s of audio) where LAME took 0.19 s, and it buffers the whole
  stream until `finish`. An hour of audio would take half an hour to encode,
  and sound worse.
- **encoRust** (MIT/Apache-2.0) is research-stage, not on crates.io, and
  defers its bit reservoir and VBR.

LAME is the reference-quality MP3 encoder and ships in every distribution
(`libmp3lame0`). Loading it with `dlopen` keeps it out of the binary: rivet
neither links nor redistributes LAME, and its LGPL obligations fall on
whoever installs the library, exactly as with the GPU runtimes (§4). The
feature is off by default because it reaches for an LGPL library at all; a
build without it has no LGPL code path. MP3's patents have expired (the last
in 2017), so unlike AAC there is no licence to encode it.

**What rivet adds around LAME.** The output rate (32 / 44.1 / 48 kHz pass
through; the 11.025 kHz family is resampled to 44.1, the rest to 48 — LAME is
told its output rate outright, since left alone it drops to the MPEG-2 half
rates at low bitrates), the downmix to two channels (§22), cutting the byte
stream into one packet per frame, and the encoder delay (LAME's 576 + the
decoder's 529 + the resampler's), which an MP4 edit list or a bare `.mp3`'s
LAME tag hides. LAME's own tag frame is switched off: in an MP4 it would be a
sample that decodes to a frame of silence. **Where:**
[`codec::audio::encode::mp3`](../crates/codec/src/audio/encode/mp3/mod.rs).

### 22. Channel layouts: downmix by BS.775, never upmix
**Decision.** `audio-channels=source|mono|stereo|5.1|7.1`. `source` keeps the
source's layout where the output codec carries it; the others downmix with
ITU-R BS.775's coefficients (centre and surrounds at −3 dB into the fronts), the
**LFE dropped**, side and back surrounds relabelled or folded, and the matrix
**normalised** so no output clips. A request for more channels than the source
has is **an error**, everywhere — never a silent upmix, and never a narrower
file that claims otherwise.

**Why.** An upmix fabricates channels a mix never had; a stereo file under a
5.1 label misleads the player. Refusing is the one behaviour that is honest in
both directions, and it is consistent across the CLI, the API and the batch
manifest because it is decided in one place (`prepare_audio`). The LFE is
dropped because BS.775 and A/52's own downmix drop it: bass management is the
playback system's, and folding a channel mixed +10 dB in band into full-range
speakers makes the downmix boom. Normalising costs 7.7 dB on the fronts of a
5.1 → stereo downmix, which is what `ffmpeg -ac 2` gives too; clipping a loud
centre-panned passage costs more.

**Layouts Opus has no mapping for** (2.1, 3.1, 4.0, 4.1, AC-3's 2/1) go out in
the narrowest Opus channel-mapping family 1 layout that has a place for every
speaker, the missing ones silent (2.1 → 5.1, 4.0 → 5.0 with the back centre in
both surrounds): no content is made up, and an LFE is never folded away. The
decoders report the layout they decode to (AC-3's `acmod`, DTS's `AMODE`),
because a channel count does not say it: four AC-3 channels are 4.0,
quad(side) or 3.1. **Where:** [`codec::audio::remix`](../crates/codec/src/audio/remix.rs),
[`rivet::job::audio`](../crates/rivet/src/job/audio.rs).

### 23. MP3 in MP4 is `mp4a` / 0x6B, and its `codecs` value is `mp3`
**Decision.** MP3 in an MP4 is an `mp4a` sample entry whose `esds` names object
type 0x6B (0x69 at the MPEG-2 half rates) with no DecoderSpecificInfo — what
ffmpeg and Apple write. The RFC 6381 `codecs` value rivet reports for it
(`JobOutput::audio_codecs`) is **`mp3`**.

**Why.** The spelling RFC 6381 derives from that `esds` is `mp4a.6B`. Chromium
accepts it (and `mp4a.69`, and `mp3`: `media/base/mime_util_internal.cc`);
Gecko's MP4 reader recognises MP3 only as `mp3` and rejects `mp4a.6B`
(`dom/media/mp4/MP4Decoder.cpp`). `mp4a.40.34` (MPEG-4 audio object type 34)
describes a different `esds` and neither engine accepts it. `mp3` is the one
string both engines play from. Safari was not checked.

### 24. No MP3 in HLS
**Decision.** `audio=mp3` with HLS output is a validation error.

**Why.** RFC 8216 carries MP3 in MPEG-2 TS segments or as packed audio; rivet's
HLS is CMAF (fMP4), for which ISO/IEC 23000-19 defines no MP3 media profile and
Apple's HLS authoring spec lists no MP3. A rendition built anyway is one a
player is free to skip, which is worse than a clear refusal. `auto` transcodes
an MP3 source to Opus for HLS, as it always did.

### 25. Audio-only output is a bare `.mp3`
**Decision.** `mode=audio` (`OutputMode::AudioOnly`) writes the audio alone as
one `.mp3` file: the frames behind an `Info` frame (frame and byte counts, a
seek table) whose LAME extension carries the encoder delay and end padding, so
a gapless player presents exactly the source's samples. A single-file job
whose input has no video becomes one by itself. `audio=auto` means MP3 there;
`audio=opus` is refused.

**Why.** MP3 is the audio-only deliverable that plays everywhere — podcast
feeds, previews, devices — and a bare `.mp3` is what those consumers take. An
audio-only MP4 (`.m4a`) would serve Opus and AAC too; it is future work, since
the MP4 muxer is built around a video track.
