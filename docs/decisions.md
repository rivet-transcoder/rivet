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
(default), H.264, and H.265 — and, since §35, VP9, VP8, MPEG-2, MPEG-4 Part 2
and ProRes, each an explicit opt-in too.

**Why.** Royalty position. AV1 + Opus + MP4 carries **zero codec-royalty
exposure** on the output: AV1 and Opus are royalty-free, the MP4 (ISO-BMFF)
container is itself royalty-free, and AAC *passthrough* transmits the source's
bytes without decoding or encoding them (rivet decodes AAC only when a job
needs its PCM, and encodes it only when a job asks for AAC: §26). AV1 was the original
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

**Where.** `VideoCodecPolicy` (defaulting to `Av1`) in
[`spec/policy.rs`](../crates/rivet/src/spec/policy.rs); the audio routing
(passthrough vs transcode) in `prepare_audio`
([`job/audio.rs`](../crates/rivet/src/job/audio.rs)), in
[`transcode.rs`](../crates/rivet/src/transcode.rs) for the one-call path,
and [`codec/audio/`](../crates/codec/src/audio/).

### 2. Audio: passthrough what's clean, transcode the rest to Opus, drop the unplayable
**Decision.** AAC / Opus / AC-3 / E-AC-3 / DTS pass through verbatim, and so does
MP3 into a single-file MP4 (and Opus and Vorbis into a WebM); Vorbis, MP2, PCM,
FLAC and ALAC (and MP3 for HLS) are transcoded to Opus; anything else is
dropped (video-only) with a warning. Every other codec rivet reads is an
output too, asked for by name — MP3 (`audio=mp3`, §21), AAC-LC, HE-AAC and
HE-AAC v2 (`audio=aac|he-aac|he-aacv2`, §26), Vorbis, AC-3, E-AC-3 and DTS
(§37), FLAC / ALAC (§27) — and the output channel layout is a knob of its own
(§22).
`audio-decode-deny` names source codecs that may not be decoded at all: such a
track is passed through where the output can carry it and the job refused
where it needs the PCM, never silently skipped.

**Why.** Passthrough avoids re-encoding (quality + royalty cleanliness). Opus is
the royalty-free transcode target and plays in MP4 on modern Apple + browsers.
Adding an AAC *encoder library* (e.g. `fdk-aac`) was rejected — it
reintroduces a Fraunhofer license, and silently dropping AAC sources would be
worse than passthrough. AAC output now comes from rivet's own encoder
(§26), and only when a job asks for it: what `auto` does is unchanged. AAC
sources are decoded by rivet's own decoder (§26) when a job needs their PCM
— a downmix, a filter, another codec asked for — and still passed through
when nothing does; HE-AAC decodes in full (`he-aac` can keep it undecoded or
decode only its core). MP3 joined the passthrough set
in 2026-09: every browser plays MP3 in an MP4, and re-encoding a lossy track to
another lossy codec only loses quality. CMAF has no MP3 profile, so an HLS
package still transcodes it. See §1.

---

## No FFmpeg; clean-room + hand-rolled FFI

### 3. The demuxers and muxers are hand-written clean-room parsers
**Decision.** MP4/MOV/MKV/WebM/TS/AVI demux and MP4 / CMAF / HLS mux are
all hand-written in the [`container`](../crates/container/) crate. No FFmpeg, no
container library. FFmpeg was removed from the whole workspace on 2026-08-12, and is
absent from every build (see [No FFmpeg](../README.md#no-ffmpeg)).

**Why.** Licensing independence (FFmpeg is LGPL/GPL), full control over the exact
bytes we emit (faststart, Apple brand sets, HDR atoms, segment alignment), and a
build that has **no FFmpeg prerequisite**. The cost — reimplementing parsers — is
paid once and bought back in deployment simplicity and output correctness.

**History: the opt-in libavcodec tier (2026-08-14 to 2026-10-02).** On
2026-08-14 an `ffmpeg` cargo feature (off by default) brought libavcodec back
for **video decode only**, as a software tier below the hardware decoders and
below the workspace's own H.264 / HEVC decoders (`h26x`), catching what they
refused and the codecs nothing else here decodes in software (VP8, VP9,
MPEG-2, MPEG-4, ProRes). It was restored because a software H.264 path that
only had openh264 decoded eleven of a High-profile upload's 5,533 frames in
production — a gap the native `h26x` decoders (2026-08-18) have since closed.
On 2026-10-02 the feature, the `ffmpeg-next` dependency and the tier were
removed for good: rivet takes no dependency on FFmpeg of any kind, opt-in or
not. What only libavcodec had decoded in software is covered the same day by
the workspace's own clean-room decoders, each in a repository of its own and
carried here as a submodule (§34): ProRes (`crates/prores`, from SMPTE RDD
36), VP8 (`crates/vp8`, from RFC 6386), VP9 (`crates/vp9`, from the VP9
bitstream specification), MPEG-2 and MPEG-1 video (`crates/mpeg2`, from ITU-T
H.262) and MPEG-4 Part 2 (`crates/mpeg4`, from ISO/IEC 14496-2). See
[codec-decode.md](codec-decode.md).

**Where.** [container.md](container.md); the box writers in
[`mux/`](../crates/container/src/mux/mod.rs) / [`cmaf/`](../crates/container/src/cmaf/mod.rs).

### 4. GPU codec backends are hand-rolled `dlopen` FFI mirroring the vendor SDK headers
**Decision.** NVENC/NVDEC, AMF, and QSV (oneVPL) are reached through our own FFI
that mirrors the vendor C structs, loaded at runtime with `libloading`. No
external wrapper crate.

**Why.** (a) Cross-platform: this builds on **Windows MSVC + Linux**, where the
obvious wrapper (shiguredo_vpl) does not. (b) Runtime `dlopen` means **one binary
runs whether or not the GPU libraries are present** on the host — it engages the
GPU when the driver is there, with no link-time dependency on it. (c) We control
the exact ABI. (Note: the `dlopen` boundary is about not link-depending on
driver libs, not a CPU fallback; the software tiers are separate and sit
below the GPU backends — see §5.)

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

### 5. Codecs are GPU-first as built; the software tiers sit below the silicon
**Decision.** Hardware first, software below it, and every software encoder
opt-in:
- **Decode** ([`decode/mod.rs`](../crates/codec/src/decode/mod.rs)
  `create_decoder`) tries **NVDEC → AMF → QSV** for the detected GPU, then the
  software tiers: the workspace's own decoders (pure Rust, always in the
  chain, one per codec: `h26x` for H.264 / HEVC, `av1` for AV1 (since §38;
  before it, rav1d behind `rav1d-fallback`), and `prores`, `vp8`, `vp9`,
  `mpeg2`, `mpeg4` for ProRes, VP8, VP9, MPEG-1 / MPEG-2 and MPEG-4 Part 2),
  then openh264 (`openh264-fallback`), only when built, and **hard-fails** if
  none matches. A hardware decoder that cannot start a stream declines and
  the next tier is tried; one that refuses its first sample falls back with
  what it was fed replayed.
- **Encode** ([`encode/mod.rs`](../crates/codec/src/encode/mod.rs)
  `select_encoder`) tries the hand-rolled **NVENC → AMF → QSV** backends, then
  **software AV1** via the workspace's own `av1` encoder when built with
  `av1-sw-fallback` (8- and 10-bit 4:2:0, SDR; rav1e, 8-bit, until §38) and
  **software H.264 / H.265** via the `h26x` encoders when built with
  `h26x-fallback` (8- and 10-bit 4:2:0). A default build has no software
  encoder. A *pinned*-vendor init failure stays a hard error — a lease that named a GPU
  means the caller wanted that GPU, and quietly serving it from the CPU would
  make a broken driver look like a slow one.

**Why GPU-first.** The production target is GPU hosts; a silent CPU fallback
would mask a misconfigured GPU as a slow-but-working job. That is why the
software tier is opt-in *and* sits below the vendor chain rather than above it:
a build that has it still prefers silicon, and a host that lacks the feature
still fails fast with the real driver error on the job's failed event.

**Why software AV1 is pure Rust.** rav1e and rav1d were ordinary cargo
dependencies — no system libraries, no bindgen, no LLVM, nothing the deployment
image has to ship. That is the whole reason they could be made a default-off
feature instead of a build-environment decision; see [No
FFmpeg](../README.md#no-ffmpeg) for the tier they replaced. (Superseded by
§38: AV1 is now the workspace's own `crates/av1`, pure Rust for the same
reason, and its decoder is no longer gated.) The same holds for
the `h26x` crate and the ProRes, VP8, VP9, MPEG-2 and MPEG-4 crates, which is
why their decoders can be in every build.

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
**Decision.** For a ladder (HLS, or multi-GPU single-file), one worker per GPU holds its lease for the whole
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
Single-file (chunk-and-stitch, §9) runs on the same ladder core — range-split
decode, per-rung scalers, ladder workers — with a chunk of several GOPs as
its unit. See [`multigpu/ladder.rs`](../crates/rivet/src/multigpu/ladder.rs),
[`multigpu/hls.rs`](../crates/rivet/src/multigpu/hls.rs) and
[`multigpu/single_file.rs`](../crates/rivet/src/multigpu/single_file.rs).

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
**Decision.** By default (`ColorPolicy::TonemapToSdr`, `color=sdr`) every HDR
source is tonemapped to 8-bit BT.709 SDR at transcode time; the output ladder
is single-flavor SDR. No HDR output unless a job asks for it, and never a
parallel HDR rendition beside the SDR one (a spec has one colour policy for
every rung).

**Why.** Most UGC "HDR" is captured accidentally: iOS records HLG ~1 stop bright
expecting Apple's tonemapper to bring it down by viewing conditions (which fails
the moment the file leaves Apple); Samsung writes HLG with a viewing-condition
variable nearly every conversion drops. YouTube/Meta have given talks on the
policy + UI work needed to make HDR feeds tolerable — work inappropriate for our
scale. Shipping native HDR without it lands eye-searing / washed-out clips on
viewers. Tonemapping at upload normalizes this on our side. The tonemap is in
[`tonemap.rs`](../crates/codec/src/tonemap.rs); the dispatch in
[`colorspace/`](../crates/codec/src/colorspace/mod.rs).

**Escape hatch, now a per-job opt-in.** The 10-bit pipeline, the HDR mux atoms
(`mdcv`/`clli`), 10-bit encode and HDR metadata extraction are reachable per
job through the colour policy: `passthrough` keeps the source's colour and
depth, `hdr10` / `hlg` output BT.2020 PQ / HLG at 10 bits (an SDR source
mapped in by ITU-R BT.2408, never only re-tagged), and `validate` refuses
what the build cannot encode for the job's codec. The pump never tonemaps on
its own; the policy decides. Nothing converts between PQ and HLG: one on the
other is refused by name. See [output-spec.md §4](output-spec.md#4-color--bit-depth)
and `ColorPolicy` in [`spec/policy.rs`](../crates/rivet/src/spec/policy.rs).

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
`colr nclx` color tagging, AV1 **Main** profile 4:2:0, AAC/Opus audio, an
Apple-friendly `ftyp` brand set (`iso6` major; `iso6`/`iso2`, the codec's brand
— `av01`, `avc1` or `hvc1` — and `mp41`/`mp42` compatible), and H.264 / H.265
under the `avc1` / `hvc1` sample entries, parameter sets out of band, with
`avc3` / `hev1` only where a stream really changes a parameter set under its id.

**Why.** "Optimized for web" is a pile of choices FFmpeg leaves to the caller.
Faststart lets a clip start playing before it's fully downloaded; segment
alignment across the ladder lets hls.js switch renditions cleanly; `colr` stops
QuickTime/iOS Safari silently applying BT.709-limited fallback (which breaks
non-709 sources); the brand set is what iOS Safari needs to accept the
largesize/co64 path. `avc1` / `hvc1` because some players refuse the in-band
entries outright — Safari's `<video>` on iOS rejects an `avc3` file with
`MEDIA_ERR_SRC_NOT_SUPPORTED` where the same stream under `avc1` plays — and
Apple's HLS authoring specification asks for `hvc1`. Stitched chunks and HLS
renditions get them too: the stitch writes the sets out of band when the
chunks' sets are byte-identical, and each HLS rendition's entry is settled
from its segments before its `CODECS` string is read. See
[container.md](container.md).

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
`into_spec()` builder (`into_image_spec()` for `mode=image`) and one set of
`parse_*` string parsers. Every key a caller can leave out has a word that
states its default and builds the same job (`gop=2s`, `max-fps=source`,
`audio-bitrate=standard`, ...), so a caller can name every setting
([output-spec.md](output-spec.md#stating-the-defaults)).

**Why.** Before this, the spec-building logic existed **three times** (the server's
`build_spec`, the CLI's `resolve_rungs`, the IPC's `JobSettings`) and a new option
meant editing all three. Now an option is a one-place change and the three
surfaces map 1:1. See [engine.md](engine.md#the-front-ends-and-the-shared-transcodesettings) and
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

### 21. MP3 output: rivet's own encoder (LAME, and the `lame` feature, are gone)
**Decision.** `audio=mp3` encodes constant-bitrate MPEG-1 Layer III with the
workspace's own MP3 encoder: the `rivet-mp3` crate (imported as `mp3`), kept
in its own repository,
[rivet-transcoder/rivet-mp3](https://github.com/rivet-transcoder/rivet-mp3),
and carried here as the `crates/mp3` submodule (§34). The same crate decodes
MPEG audio (Layers I, II and III; MPEG-1, MPEG-2 LSF, MPEG-2.5) in place of
minimp3. Every build encodes MP3: there is no feature, nothing is loaded at
run time, and `validate()` no longer refuses `audio=mp3` for want of one.

**History.** From 2026-09 to 2026-10-03 MP3 was encoded by LAME, loaded at run
time with `dlopen` behind an off-by-default `lame` feature, because no encoder
of the project's own existed: a first clean-room attempt (2026-09-27) stopped
when no copy of ISO/IEC 11172-3 with its normative annexes (the Huffman tables,
the scalefactor bands, the analysis window) was to hand, and the permissively
licensed Rust encoders measured then (oxideav-mp3: 17.7 dB SNR against LAME's
25.3 dB at 128 kbit/s, muffled above 11 kHz, at 2× real time; encoRust:
research-stage) were not good enough. rivet-mp3 was then written from
ISO/IEC 11172-3 and 13818-3 (its `docs/PROVENANCE.md` records the sources);
no MPEG audio implementation's source was read or run. Its decoder meets ISO's
full-accuracy criterion on all 64 conformance sequences it was checked on, and
its encoder's own suite decodes every configuration strictly (its README has
the figures; mean noise-to-mask +1.4 dB at 128 kbit/s, −4.0 dB at 192). The
`lame` feature was removed rather than kept as an alias: it would be a switch
that switches nothing. MP3's patents have expired (the last in 2017).

**What rivet does around the encoder.** The output rate (32 / 44.1 / 48 kHz
pass through; the 11.025 kHz family is resampled to 44.1, the rest to 48 —
MPEG-1 rates only, which every player and MP4 reader takes, though the encoder
codes the MPEG-2 and 2.5 rates too), the downmix to two channels (§22), one
packet per frame, and the delay: the encoder's 528 samples
(`Encoder::delay`) plus the decoder's 529 (`mp3::xing::DECODER_DELAY`), which
an MP4 edit list hides. A bare `.mp3` opens with the encoder's own `Info` tag
frame (`Encoder::tag_frame`: frame and byte counts, seek table, and the
LAME-style extension with the delay, the padding and the CRCs, under the
encoder string `rivetmp3`), so a gapless player presents exactly the input;
inside an MP4 that frame would be a sample decoding to silence, so it goes
only into the `.mp3`. rivet's own `.mp3` reader trusts the extension of any
encoder whose tag CRC checks out (and LAME's and the `Lavf` / `Lavc` muxers'
by name, as before). **Where:**
[`codec::audio::encode::mp3`](../crates/codec/src/audio/encode/mp3.rs),
[`decode::mp3`](../crates/codec/src/audio/decode/mp3.rs),
[`job/audio_only.rs`](../crates/rivet/src/job/audio_only.rs).

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
whose input has no video becomes one by itself. `audio=auto` means MP3 there.

**Why.** MP3 is the audio-only deliverable that plays everywhere — podcast
feeds, previews, devices — and a bare `.mp3` is what those consumers take.

**Since.** Lossless audio (§27) adds two audio-only files: a native `.flac`
for `audio=flac` and an audio-only MP4 (`.m4a`, written by its own small
faststart writer rather than the video muxer) for `audio=alac`;
`audio-container=mp4` puts any codec the MP4 muxer takes, Opus and AAC included, in
an `.m4a`. `audio-container` names the file; left out, it follows the codec.
Since 2026-10-03 the codec's own file is chosen for each (§37): an Ogg file
(`.opus` / `.ogg`) for Opus and Vorbis, an `.m4a` for AAC, HE-AAC, AC-3,
E-AC-3 and DTS (`audio=opus` used to be refused, for want of a file).

### 26. AAC-LC is encoded and decoded here, from the standards
**Decision.** rivet encodes and decodes AAC-LC with its own codec, the
`rivet-aac` crate (imported as `aac`), kept in its own repository,
[rivet-transcoder/rivet-aac](https://github.com/rivet-transcoder/rivet-aac),
and carried here as the `crates/aac` submodule, as the H.264 / H.265 decoders
are (`crates/h26x`). Pure Rust, no library, written from the ISO/IEC
standards and the published literature; not a wrapper around, or a port of,
any existing AAC encoder or decoder. The codec crate adapts it:
[`encode::aac`](../crates/codec/src/audio/encode/aac.rs) (resampling to a
coded rate, packet timing) and [`decode::aac`](../crates/codec/src/audio/decode/aac.rs).
The encoder was written in this repository first (2026-09-27) and moved to
the new one with its history (`git filter-repo`) when the decoder was added
(2026-09-28); the two share one set of tables.

**Why.** Opus in MP4 plays in Safari and on iOS only from version 17; AAC-LC
plays on every browser and device that plays video. The library route was
already closed (§2: `fdk-aac` brings Fraunhofer's licence), and the other
encoders are either copyleft or tied to a platform. An in-tree encoder has
no licence dependency and needs nothing outside this repository, as §3 asks.
The decoder closes the other half: before it, an AAC track could only be
passed through, so a 5.1 AAC source could not be downmixed, and a job asking
Opus, MP3, FLAC or ALAC of an AAC source passed the AAC through instead (or
was refused, for a bare `.mp3` / `.flac`). AAC may be subject to patent
licensing in some jurisdictions; this project makes no claim either way, and
AAC is decoded or encoded only when a job needs it, never by `auto` on a
source it can pass through.

**HE-AAC and HE-AAC v2 are implemented, both ways, at the owner's request
(2026-10-02), reversing the 2026-09-28 decision to leave them out.** That
decision rested on licence exposure (patents on SBR and parametric stereo
still in force by the owner's reckoning, parametric stereo's in the US until
2028-12-09); whether their use needs a licence is the user's to determine, and
nothing here is a licence to any patent. xHE-AAC (USAC) remains absent. The
decoder decodes spectral band replication at the full rate and parametric
stereo to two channels (held to ISO/IEC 14496-26's conformance streams; the
crate's README has the figures). rivet decodes an HE-AAC source in full, as
any AAC track: the `he-aac` setting
([output-spec.md](output-spec.md#3-audio--with_audioaudiocodecpolicy)) is
`auto` (the default: passed through where the output carries it and nothing
asks for a change, decoded in full otherwise), `passthrough` (never decoded,
refusing what would need it) or `core` (decoded as its AAC-LC core only — half
the rate, a quarter of the bandwidth, HE-AAC v2's mono core: the cheaper,
older decode; its handling still reads `he-aac (lc core) → …`, the wording
job-output consumers count core-only decodes by). A full decode reads
`he-aac → …` / `he-aacv2 → …`. `audio=he-aac` (mono to 7.1, 32 / 44.1 / 48
kHz) and `audio=he-aacv2` (stereo) encode them, with the AudioSpecificConfig
signalling SBR / PS explicitly and hierarchically (object type 5 / 29 first,
the core's sampling frequency, then the SBR rate as the extension sampling
frequency: `mp4a.40.5` / `mp4a.40.29`); `audio=aac` keeps any AAC source,
`he-aac` an HE-AAC one, `he-aacv2` an HE-AAC v2 one. The container's own ASC
parser read the hierarchical form's leading sampling frequency as the SBR
rate until this change (it is the core's, ISO/IEC 14496-3 1.6.2.1), and did
not see the backward-compatible form's sync extension; both are fixed, so an
HE-AAC track's rate and timescale are right in a passthrough too.

**Provenance.** The full record is in the rivet-aac repository's
[`docs/PROVENANCE.md`](https://github.com/rivet-transcoder/rivet-aac/blob/develop/docs/PROVENANCE.md).
Written from these sources only:
- ISO/IEC 13818-7:2004 (MPEG-2 AAC): clause 6 (ADTS, raw_data_block and
  element syntax, the program_config_element), clause 7.1.6 (TNS_MAX_ORDER,
  TNS_MAX_BANDS), clause 8 (element semantics, window sequences, the
  scalefactor band tables 45–57, grouping and the order of spectral data in
  8.3.4–8.3.5, the LFE restrictions of 8.4, the implicit channel mapping of
  Table 42, the sampling-frequency mapping of Table 38, the extension types
  of Table 40, the decoder buffer and bit reservoir of 8.2.2), clause 9
  (noiseless coding: codeword indices, sign bits, escape sequences, the pulse
  tool, de-interleaving), clauses 10–11 (quantization, scalefactors), 12.1
  (M/S), 12.2 (intensity stereo), 14 (TNS), 15 (filterbank, window shapes,
  block switching) and Annex A (the Huffman codebooks). From the informative
  Annex C, for the encoder: the structure of the psychoacoustic model and its
  spreading function (C.1), the MDCT definition (C.3), M/S (C.6.1), the
  quantizer and its rounding constant, the bit reservoir control (C.7), and
  sectioning (C.8).
- **Where the tables came from — an owner exception.** The normative
  tables were transcribed, by a script reading the PDF's text positions, from
  a copy of ISO/IEC 13818-7:2004 retrieved on 2026-09-27 from
  `https://ossrs.net/lts/zh-cn/assets/files/ISO_IEC_13818-7-AAC-2004-67b015c6ddfc9a4af83665738477124a.pdf`.
  Its footer identifies it as a licensee's copy ("Reproduced by IHS under
  license with ISO … IHS Licensee=etri") re-hosted without authorisation:
  not a purchased copy, and the kind of source the AAC-decoder entry in
  TODO.md had ruled out. The owner reviewed this and explicitly approved
  using it for the normative tables on 2026-09-28. For the encoder
  (2026-09-27): the twelve Huffman codebooks (Tables A.1–A.12) with their
  parameters (Table 59), the scalefactor band offsets for 22.05–48 kHz
  (Tables 45–47, 52, 53) and the sampling-frequency indices (Table 35). For
  the decoder (2026-09-28), from the same copy: the scalefactor band offsets
  for 8–16 kHz and 64–96 kHz (Tables 48–51, 54–57), TNS_MAX_BANDS (Table
  33) and the explicit-rate mapping (Table 38). Nothing else came from it by
  transcription: the windows are computed from their formulas, and every
  algorithm is the crate's own. The tables are verified: every codebook is a
  complete prefix code (Kraft sum exactly 1) and every codeword decodes to
  its own index; every band table rises in multiples of four to 1024 or 128;
  ffmpeg decodes the encoder's output of every rate × bit rate × layout with
  no error; and the decoder's PCM agrees with ffmpeg's at every sampling rate
  (below). Buying ISO/IEC 13818-7:2006 (whose LC tables are the same) to
  re-verify them remains an option.
- ISO/IEC 14496-3 (MPEG-4 Audio), from the published syntax of these clauses:
  AudioSpecificConfig (1.6.2.1) and GASpecificConfig (4.4.1) for the MP4
  `esds` and the decoder's configuration; the MPEG-4 form of the ADTS
  header; SBR / PS signalling (1.6.5), which the decoder only recognises;
  perceptual noise substitution (4.6.13) for the decoder.
- Literature: Johnston, "Transform coding of audio signals using perceptual
  noise criteria", IEEE JSAC 6(2), 1988 (tonality from spectral flatness;
  14.5 + Bark dB for tones, 5.5 dB for noise); Zwicker & Terhardt, JASA
  68(5), 1980 (Bark); Terhardt, Hearing Research 1, 1979 (threshold in
  quiet); Johnston & Ferreira, "Sum-difference stereo transform coding",
  ICASSP 1992 (M/S); Princen & Bradley, IEEE TASSP 34(5), 1986 (TDAC);
  Malvar, *Signal Processing with Lapped Transforms*, 1992, and Britanak,
  Yip & Rao, *Discrete Cosine and Sine Transforms*, 2007 (the MDCT and IMDCT
  through a quarter-length FFT); Herre & Johnston, AES 101st Convention, 1996
  (TNS).
- **No AAC implementation's source was consulted**, for either half: not
  FDK-AAC, FAAC, FFmpeg's AAC encoder or decoder, faad2, symphonia, NihAV,
  Nero, VisualOn, Apple's, the 3GPP reference code or any other; no table
  was derived by probing a decoder. `ffmpeg` / `ffprobe` served only as
  black boxes: to make test streams (its own encoder, and fdk-aac through
  it, as a command-line tool), to decode them, and to compare the PCM.

**The decoder, and what was measured.** AAC-LC: ADTS (any chunking, resynced
on the syncword) and raw access units with the AudioSpecificConfig;
channel configurations 1–7 and program_config_element layouts; long,
start, short and stop windows with sine and KBD shapes; M/S, intensity
stereo, PNS, TNS, pulse data. Output follows the native layouts (5.1: FL FR
FC LFE BL BR; configuration 7's outside-front pair as the side pair, so 7.1
is FL FR FC LFE BL BR SL SR, as the encoder sends it). A PCE whose elements
do not fit its own position rules (ffmpeg's encoder writes some) comes out in
its element order, the layout left to the channel count. AAC Main, SSR, LTP,
960-sample frames and coupling channel elements are refused by name. Against
ffmpeg's decoder, on ffmpeg's own streams (mono to 7.1, PCE layouts,
22.05–48 kHz, 32–320 kb/s, CBR and VBR, ADTS and MP4, with M/S, intensity
stereo and TNS) and on fdk-aac's (8–96 kHz, mono to 7.1), the PCM agrees
to float rounding (see the rivet-aac README for the figures); streams with
PNS, whose noise is random by definition, agree in energy per block. A
property test feeds arbitrary and mutated input to every entry point:
errors, never a panic.

**The encoder's shape, and what was measured.**
- *Rate control* is one noise-to-mask offset for every band of every channel
  of a frame, found by bisection against the frame's bit budget (Annex
  C.7.4's "constant NMR"); the threshold in quiet stays an absolute floor, so
  spare bits go to audible bands instead of inaudible ones. The budget
  follows the frame's perceptual entropy against a running geometric mean,
  and the reservoir obeys 8.2.2 (fill elements when it would overflow), so
  the stream is constant-rate at the decoder-buffer level; totals land within
  one buffer of the target.
- *Block switching*: an energy-ratio detector on 128-sample sub-blocks that
  coincide with the short windows. On a castanet-like click train (64 kb/s
  mono) the error energy 21.3 to 2.7 ms ahead of the onsets is 22 dB below a
  long-windows-only encode; in the last 2.7 ms (inside the short window that
  holds the onset, within backward masking) 2 dB.
- *TNS* was implemented and measured, and is left out. In the short window
  that holds an attack the MDCT's time-domain aliasing folds the onset back
  onto the samples before it, so TNS's temporal shaping put 1–3 dB *more*
  error there; on long windows, compensating the synthesis filter's noise
  gain cost 1.5–3.5 dB of SNR on music, and not compensating it only moved
  noise. Worth revisiting with listening tests, not with these metrics.
- *Quality* (steady-state SNR through the test decoder):
  | Signal | Rate | SNR / segmental SNR |
  |---|---|---|
  | 997 / 1499 Hz sines, stereo, 22.05–48 kHz | 128 kb/s | 66–73 dB |
  | harmonic "music", stereo 48 kHz | 64 / 128 / 192 / 320 kb/s | 8 / 27 / 43 / 46 dB SNR; 19 / 41 / 48 / 50 dB seg. |
  | the same over a noise bed | 64 / 128 / 192 / 320 kb/s | 4 / 12 / 20 / 40 dB SNR |
  | one sine per channel, 5.1 and 7.1 | defaults | ≥ 63 dB mains, 46 dB LFE |
  A noise bed is coded to its masked threshold (about 5.5 dB SNR at the
  margin), which is what drags SNR there; SNR is not what a perceptual coder
  optimises, and none of this replaces listening.
- Left out, all optional for an encoder: intensity stereo, PNS, the pulse
  tool, KBD windows (every window half is a sine half).

**Where.** The codec: `crates/aac` (the rivet-aac submodule), the provenance
of each part in its module's docs and `docs/PROVENANCE.md`. The adapters:
[`encode/aac.rs`](../crates/codec/src/audio/encode/aac.rs),
[`decode/aac.rs`](../crates/codec/src/audio/decode/aac.rs). `audio=aac` wires
the encoder into jobs ([`job/audio.rs`](../crates/rivet/src/job/audio.rs)): a
single-file MP4 / MOV, HLS or an `.m4a`, `mp4a.40.2` (`.5` / `.29` for
`he-aac` / `he-aacv2`), the channel configuration from the layout
([`remix::aac_layout`](../crates/codec/src/audio/remix.rs)), the priming
hidden by the edit list. The decoder is wired into the same place: an AAC
track is probed on its first access unit, decoded when the job needs its
PCM, and `he-aac` decides for an HE-AAC one.

**Proposal, not decided: should `auto` fall back to AAC instead of Opus?**
Today `auto` transcodes what it cannot pass through to Opus. AAC would reach
players Opus does not — Safari and iOS before 17 play no Opus in MP4, and AAC
plays wherever H.264 does — at some cost: AAC-LC needs a higher bit rate than
Opus for the same quality (Opus is transparent around 96–128k stereo, this
encoder's defaults are 128k stereo / 384k 5.1), and AAC may be subject to
patent licensing where Opus is designed not to be, which cuts against §1's
royalty-clean default. A middle way is to key the fallback to the video: AAC
when the job's video is H.264 (a job that has already chosen legacy reach),
Opus beside AV1. Left for a product-level decision; nothing here changes
`auto`.

### 27. Lossless audio is clean-room FLAC and ALAC
**Decision.** FLAC and ALAC are decoded and encoded by rivet's own
pure-Rust implementations ([lossless-audio.md](lossless-audio.md)) — the
`rivet-lossless` crate (imported as `lossless`), kept in its own repository,
[rivet-transcoder/rivet-lossless](https://github.com/rivet-transcoder/rivet-lossless),
and carried here as the `crates/lossless` submodule, as the AAC codec is
(§26) — selected
per job with `audio=flac|alac`, beside video in MP4 and HLS or alone (§25) as
a native `.flac` or an `.m4a`. `audio=auto` is unchanged in spirit: FLAC and
ALAC sources, now decodable, are transcoded to Opus like any other source
that is not passed through.

**Why these two, in a web-first engine.** They are the lossless formats the
web plays: FLAC in MP4 in Chrome, Edge, Firefox and Safari, ALAC in MP4 across
Apple platforms and Safari, and both in fMP4 HLS per Apple's authoring
specification. Both are royalty-free — FLAC is an open format (RFC 9639), and
Apple published ALAC under Apache 2.0 — so, unlike MP3's encoder (§21),
they need nothing outside this repository and keep the output's royalty
position. They serve masters, archive copies and lossless music delivery,
which lossy audio cannot. Nothing else (WavPack, Monkey's Audio, TTA, …) plays
in a browser, so nothing else is in scope.

**Why clean-room.** Licensing independence, as for the containers (§3), and
the same reason there is no FFmpeg (§3): the codec is small enough to own.

**Provenance.** Written from:
- the FLAC format specification, IETF RFC 9639 (and the xiph.org format
  documentation it standardises), including "Encapsulation of FLAC in ISO
  Base Media File Format" (xiph.org) for `fLaC` / `dfLa`;
- the published description of the Apple Lossless format: the
  `ALACSpecificConfig` magic cookie, its channel layouts and `chan` box, the
  frame element syntax, and the adaptive Golomb-Rice and adaptive-predictor
  coding scheme;
- published literature on linear prediction (autocorrelation method,
  Levinson-Durbin recursion, coefficient quantisation) and on Rice / Golomb
  coding;
- Apple's HLS authoring specification (codec strings `fLaC`, `alac`) and MDN's
  audio codec guide (browser support).

**No implementation's source was consulted** — not libFLAC, not FFmpeg's FLAC
or ALAC codecs, not Apple's ALAC reference code, not claxon, symphonia or any
other decoder or encoder. The `flac` command-line tool and ffmpeg were used
only as black boxes: to produce test inputs, and to decode rivet's outputs so
they could be compared with the source PCM
([`lossless_oracle.rs`](../crates/codec/tests/lossless_oracle.rs) here, and
the crate's own [`tests/oracle.rs`](../crates/lossless/tests/oracle.rs)). The
code was written in this repository first and moved to its own, with its
history, on 2026-10-02.

**Where.** The codecs: [`crates/lossless`](../crates/lossless/README.md) (the
rivet-lossless submodule; `codec::audio::lossless` re-exports it). Their
adapters: `codec/src/audio/decode/{flac,alac}.rs` and
`codec/src/audio/encode/{flac,alac}.rs`. Then
[`container/src/demux/audio/lossless.rs`](../crates/container/src/demux/audio/lossless.rs),
[`container/src/mux/lossless.rs`](../crates/container/src/mux/lossless.rs), and
`prepare_audio` in [`rivet/src/job/audio.rs`](../crates/rivet/src/job/audio.rs)
and the audio-only writer in
[`rivet/src/job/audio_only.rs`](../crates/rivet/src/job/audio_only.rs).

---

## Still images

### 28. Still images are web media, and get the web's formats

**Decision.** With the `image` feature, rivet makes still images: of an
uploaded picture (JPEG, PNG, WebP, AVIF, GIF — its first frame — TIFF, BMP,
HEIC/HEIF), or stills taken from a video. Output is the web's four picture
formats — **AVIF, WebP, JPEG, PNG** — at any number of sizes, each fitted
exactly as a video rung is (§ [fitting](output-spec.md#fitting-the-source-into-a-rung)),
but to the pixel rather than to video's even grid. `mode=image` in the
settings, `rivet image` on the CLI, `rivet::image::run_image_job` in the
library.

**Why.** The north star is media that plays well on the web, and a page is
mostly pictures: a poster for every video, a `srcset` for every photo. The
same asymmetry holds as for video — ingest what people actually upload (an
iPhone takes HEIC; cameras write JPEG and TIFF), emit only what every browser
decodes. AVIF is the default for the reason AV1 is (§1): the smallest output
at a given quality, royalty-free, and coded by the AV1 encoder rivet already
has (rav1e, through ravif, at the time; rivet's own `av1` crate and HEIF
writer since §38). WebP and JPEG are there for reach, PNG for
lossless. Nothing else is: no JPEG XL (Safari alone decodes it), no GIF or
animated output, no ICO.

**What every output gets, unasked.**
- **Upright.** EXIF orientation (JPEG, and whatever else carries it) and
  HEIF's `irot` / `imir` are applied to the pixels.
- **No metadata.** Outputs are encoded from pixels, so EXIF, XMP, GPS,
  serial numbers and embedded thumbnails never reach them — a privacy
  property, not an optimisation. The colour profile can be kept
  (`image-keep-icc`), and since §29 a caller can name identifying metadata
  to keep (`metadata-keep`), which is then written as a fresh EXIF block
  holding only that.
- **sRGB.** A source tagged otherwise (an ICC profile, or a HEIF `nclx`) is
  converted with moxcms, because a browser shows an untagged picture as sRGB.
  AVIF output is always converted: the AVIF writer writes no ICC (ravif did
  not either; rivet's own writer, §38, does not yet).

**HEIC is HEVC.** A HEIC is an HEVC picture in a HEIF box structure, so
decoding one is decoding HEVC, with HEVC's patent position. rivet does not
add a decoder for it: the HEIF items go through the same decode dispatch as
an HEVC video — the GPU's decoder, else rivet's own software HEVC decoder
(`h26x`) — and AVIF through the AV1 dispatch (NVDEC / QSV, else, at the time,
rav1d with `rav1d-fallback`, which then learnt to decode every AV1 layout and
depth rather than 8-bit 4:2:0 alone, since 4:4:4 is what most AVIF encoders
write; since §38 rivet's own `av1` decoder, in every build). A
deployment that does not decode HEVC says `image-decode-deny=heic`, and a
HEIC job fails up front with the setting's name in the error — as
`audio-decode-deny` does for audio (§2), never a silent skip. The probe
reports a HEIC's codec as `hevc` and an AVIF's as `av1` for the same reason.

**The encoders**, all permissively licensed: ravif/rav1e (AVIF), libwebp
through the `webp` crate (WebP; BSD, compiled from vendored C with `cc` — the
only lossy WebP encoder there is), jpeg-encoder (progressive, 4:2:0,
optimised Huffman tables), and the `image` crate's PNG encoder. Decoding the
raster formats is the `image` crate's, which is pure Rust. (Superseded by
§38: every one of these is now the workspace's own; WebP is rivet-webp.)

**Limits.** A source over 100 megapixels is refused from its header, before
it is decoded. Outputs are at most 16384 pixels a side (WebP: 16383).
Derived HEIF items other than `grid` (`iden`, `iovl`) and HEIF items coded
with anything but AV1 or HEVC are refused by name. HDR stills (PQ, HLG, gain
maps) are not tone-mapped; their SDR base is what comes out.

**Where.** [`rivet/src/image/`](../crates/rivet/src/image/mod.rs) (`heif.rs`
for AVIF/HEIC, `colour.rs`, `scale.rs`, `encode.rs`),
[`fit.rs`](../crates/rivet/src/fit.rs) `place_aligned`, the multi-frame
capture in [`thumbnail.rs`](../crates/rivet/src/thumbnail.rs), and
[`codec/src/decode/av1_sw.rs`](../crates/codec/src/decode/av1_sw.rs) (once
`rav1d_sw.rs`).

---

## Privacy

### 29. Outputs carry none of the source's identifying metadata unless asked
**Decision.** No output carries the source's location, device, capture time
or descriptive tags by default: the muxers write none of them, stills are
encoded from pixels (§28), and a copied AAC or MP3 stream has the source
encoder's name cleared (an AAC frame's leading fill element, MP3's ancillary
bytes and its LAME tag's version) without a bit of its audio changing.
`metadata-keep` (`OutputSpec::metadata_keep`, `ImageSpec::metadata_keep`, a
`container::metadata::Keep`) names what to carry, per category and level:
`location` or `location:approximate` (two decimal places, about a
kilometre; no altitude or place name), `capture_time` or `capture_time:date`,
`device` (make, model, software, lens) or `device:all` (serial numbers and
owner too), `descriptive`, `all`, `none`. What is named is written into each
single-file MP4, `.m4a`, `.flac` or `.mp3`, and into stills as a fresh EXIF
block. HLS output and splices refuse it.

**Why.** An upload's metadata says where someone was, with what, and when;
publishing that should be a decision, not a side effect, so the default is
what every output already carried — nothing — and keeping is named per
category. The levels exist because "roughly where" and "which day" are
often what a caller wants, and a serial number is rarely needed when the
model is. HLS refuses because a player reads no file-level metadata from
segments, so anything written there would be carried and never used; a
splice refuses because its clips can each say something different.
`Metadata::violations` checks an output against a policy, refusing what it
cannot classify, so the tests can show an output carries nothing beyond it.

**Where.** [`container/src/metadata/`](../crates/container/src/metadata/mod.rs)
(`read`, `Keep`, `write`, `scrub`), `keep_metadata` in
[`job/mod.rs`](../crates/rivet/src/job/mod.rs), the encoder-name clearing
(`metadata::scrub`, asked for from [`job/audio.rs`](../crates/rivet/src/job/audio.rs)),
and `metadata::exif` for stills, called from [`image/`](../crates/rivet/src/image/mod.rs).

---

## Video shape

### 30. A rung is a box the source is fitted into, not a size to stretch to
**Decision.** A rung's `WxH` is a maximum box. Once the source is probed
(upright, through the size-changing filters, at its display shape from the
sample aspect ratio) each rung's size and label become the output's:
`fit=contain` (default) keeps the source's shape inside the box, `cover`
centre-crops to fill it, `pad` letterboxes to exactly it, and `stretch` is
the old resize, by name. `orientation=auto` (default) turns a box to a
portrait source; `upscale` is off by default, so a smaller source comes out
at its own size and rungs that collapse onto one output are merged.
Outputs have square pixels.

**Why.** An explicit rung used to be a straight resize: a 640x480 source
through a 1280x720 rung came out stretched sideways and upscaled, and a
portrait phone video through a landscape rung was squashed into landscape.
Derived ladders were right and explicit ones were not. Fitting before
anything is encoded means encoders, muxers, playlists and progress all see
the real size, and `JobOutput::renditions` reports the box asked for beside
the size produced.

**Where.** [`fit.rs`](../crates/rivet/src/fit.rs) (`place`, `fit_rungs`),
`OutputSpec::with_rungs_fitted`, and
[output-spec.md](output-spec.md#fitting-the-source-into-a-rung).

### 31. A frame-rate cap drops frames; it never retimes them
**Decision.** When `max-fps` is below the source's rate the decode pump
decimates: each output frame period gets the source frame showing at its
start, counted from the trim in-point with absolute indexes, so every pump
and clip agrees. Frame totals (progress, HLS segment counts, chunk plans)
are in output frames, and the decode is not split into ranges under a cap.

**Why.** The cap used to lower only the rate the frames were timed at while
every source frame was still encoded, so a capped output played in slow
motion against its audio: 60 fps capped at 5 ran twelve times longer than
the source. Dropping frames keeps the duration and coarsens the motion,
which is what a cap means. Range-split decode is skipped because a sample
index no longer counts the output frames before it.

**Where.** `decimation` and `DecodePumpConfig::decimate` in
[`decode_pump.rs`](../crates/rivet/src/decode_pump.rs).

---

## Extension

### 32. Hooks are a designed extension point, typed per point of the job
**Decision.** A job runs caller-supplied code at fixed points through
[`rivet::hooks`](../crates/rivet/src/hooks/mod.rs), carried on
`OutputSpec::hooks`: source, probe, decoded frame, encoder frame, still,
artifact, completed and failed. Each is a kind of its own with its own
trait, handed only what exists at that point. A hook answers with a verdict
(carry on, or reject the job) and annotations, collected into a per-job
`HookReport`; a policy says whether it blocks or runs on the session's
background worker, fails open or closed, and is required or opt-in. The
HTTP server takes hooks with `serve_with_hooks`. The built-ins only compute
and record (`SourceDigest`, `PerceptualFingerprint`, `ArtifactDigest`).

**Why.** Integrations — fingerprinting, content review, model inference —
need to see the job at specific points without forking the pipeline. A
trait per point keeps each hook honest about what it can see (a source
hash never gets frames; a decoded-frame hook gets the source's pixels
before any colour work, an encoder-frame hook what the encoder receives),
and the engine has no opinion on what a hook is for. An empty set costs a
job nothing: each point checks for hooks of its kind first, and a frame no
hook selects is never cloned. The background worker's queue is bounded so
a slow hook slows the pipeline rather than piling frames up in memory.

**Where.** [`hooks/`](../crates/rivet/src/hooks/mod.rs);
[hooks.md](hooks.md), [hooks-cookbook.md](hooks-cookbook.md).

### 33. Model inference lives in its own crate, and ONNX Runtime is loaded at run time
**Decision.** The worked vision-model integration — a YOLO detector on the
decoded-frame and still hooks — is a workspace crate of its own,
[`examples/yolo`](../examples/yolo/Cargo.toml) (`rivet-yolo-example`,
unpublished), not one of rivet's `examples/`. It reaches ONNX Runtime
through the `ort` crate's `load-dynamic`: the `onnxruntime` shared library
is loaded at run time from `--ort` or `ORT_DYLIB_PATH`, nothing is linked or
downloaded at build time, and CUDA, DirectML and OpenVINO execution
providers are features of that crate.

**Why.** ONNX Runtime must never become a dependency of rivet: hooks are
the extension point (§32), and an integration brings its own runtime. A
crate of its own keeps `ort` out of rivet's dependency graph entirely,
where a rivet example would put it in rivet's dev-dependencies. Loading at
run time is also the only way it fits this workspace's build: the MSVC
target links the C runtime statically (`+crt-static` in
[`.cargo/config.toml`](../.cargo/config.toml), so the binary needs no
`vcruntime140.dll`), and ORT's prebuilt static library needs the dynamic
MSVC runtime.

**Where.** [`examples/yolo/`](../examples/yolo/Cargo.toml);
[hooks-yolo.md](hooks-yolo.md).

---

## Codecs

### 34. Codecs we don't have, we write clean-room, each in its own repository
**Decision.** When rivet needs a codec that no pure-Rust crate with a clean
licence provides, it is written here, from the format's specification, and
kept in a repository of its own under
[rivet-transcoder](https://github.com/rivet-transcoder), carried in this
workspace as a git submodule under `crates/` and adapted by the `codec`
crate. That is how H.264 / HEVC (`crates/h26x`), AAC (§26), AC-3 / E-AC-3,
DTS and FLAC / ALAC (§27) came in; on 2026-10-02, the five video decoders that
replaced libavcodec's software decode (§3); and on 2026-10-03 Opus, MPEG audio
and Vorbis, which replaced the last third-party audio codecs (§37):

| Crate | Repository | Written from | rivet uses |
|---|---|---|---|
| `crates/prores` | [rivet-prores](https://github.com/rivet-transcoder/rivet-prores) | SMPTE RDD 36:2022 | the decoder (the only ProRes decoder in the chain) |
| `crates/vp8` | [rivet-vp8](https://github.com/rivet-transcoder/rivet-vp8) | RFC 6386, its prose and tables (not the source in its section 20) | the decoder, behind NVDEC |
| `crates/vp9` | [rivet-vp9](https://github.com/rivet-transcoder/rivet-vp9) | the VP9 Bitstream & Decoding Process Specification v0.6 / v0.7 | the decoder, behind NVDEC / AMF / QSV |
| `crates/mpeg2` | [rivet-mpeg2](https://github.com/rivet-transcoder/rivet-mpeg2) | ITU-T H.262 (and ISO/IEC 11172-2 for MPEG-1) | the decoder, behind NVDEC |
| `crates/mpeg4` | [rivet-mpeg4](https://github.com/rivet-transcoder/rivet-mpeg4) | ISO/IEC 14496-2 (and ITU-T H.263 for the short header) | the decoder, behind NVDEC |
| `crates/opus` | [rivet-opus](https://github.com/rivet-transcoder/rivet-opus) | RFC 6716 as updated by RFC 8251, RFC 7845 | the encoder and the decoder (replacing libopus) |
| `crates/mp3` | [rivet-mp3](https://github.com/rivet-transcoder/rivet-mp3) | ISO/IEC 11172-3, 13818-3 | the encoder and the decoder (replacing LAME and minimp3) |
| `crates/vorbis` | [rivet-vorbis](https://github.com/rivet-transcoder/rivet-vorbis) | the Vorbis I specification, RFC 3533 | the encoder and the decoder (replacing lewton) |
| `crates/av1` | [rivet-av1](https://github.com/rivet-transcoder/rivet-av1) | the AV1 Bitstream & Decoding Process Specification | the decoder, behind NVDEC / AMF / QSV, the software encoder and the AVIF encoder (replacing rav1d, rav1e and ravif; §38) |
| `crates/png` | [rivet-png](https://github.com/rivet-transcoder/rivet-png) | the W3C PNG specification (third edition), RFC 1950 / 1951 | PNG in and out (§38) |
| `crates/jpeg` | [rivet-jpeg](https://github.com/rivet-transcoder/rivet-jpeg) | ITU-T T.81, T.871, the EXIF / ICC / Adobe APP14 conventions | JPEG in and out (§38) |
| `crates/webp` | [rivet-webp](https://github.com/rivet-transcoder/rivet-webp) | RFC 9649, ITU-R BT.601 (lossy frames through rivet-vp8, RFC 6386) | WebP in and out (§38) |
| `crates/imagecodecs` | [rivet-imagecodecs](https://github.com/rivet-transcoder/rivet-imagecodecs) | GIF89a, Microsoft's BMP documentation, TIFF 6.0 | GIF, BMP and TIFF in (§38) |

Each crate's encoder is rivet's encoder for its codec too (§35).

"Clean-room" means no other implementation's source was read — not
libavcodec, not libvpx, not the reference software — and none was run to
make or check anything: each crate is checked against the published
conformance material for its format (test vectors with MD5s, the ISO/IEC
13818-4 suite and its traces), against frames assembled by hand from the
specification, against round trips through its own encoder, and against
other encoders' streams used as data. Each crate's README says what it
decodes and refuses and how it was checked; its NOTICE records the
provenance and the patent and trademark position.

Each crate has an encoder as well. On 2026-10-02 the scope question was
settled (the owner asked for an encoder for every codec rivet reads), and all
five are rivet output codecs: see §35.

**Why.** The alternative for these formats was libavcodec, and §3 is why
that is gone: the build, not the code, was the cost — FFmpeg development
libraries, LLVM and libclang for bindgen, matching shared objects at run
time, an LGPL surface — and a host without all of it silently lost its
software decode. A codec written from its specification costs none of that,
and owning it settles the licence question for the code (the patent
question each format carries is the format's, not the code's, and each
NOTICE says so). The formats are small enough to own: each decoder is one
specification. It matters most for inputs: ProRes masters, VP8 / VP9 from
WebM, MPEG-2 from broadcast transport streams and MPEG-4 Part 2 from old
DivX / Xvid files arrive on hosts whose GPU cannot decode them, or with no
GPU at all.

**Why a repository each.** A codec is useful outside rivet and changes on
its own schedule; its tests (conformance suites, sample fetches) are its
own; and its provenance and licence notes have to travel with it. A
submodule keeps the code in the build with nothing to install, and each
codec's history, CI and issues in one place. The cost is the two-step
change — commit and push inside the submodule, then commit the new pointer
here — which [CONTRIBUTING.md](../CONTRIBUTING.md) spells out.

**Where.** `crates/{h26x,av1,aac,ac3,dts,opus,mp3,vorbis,lossless,prores,vp8,vp9,mpeg2,mpeg4,png,jpeg,imagecodecs}`
([`.gitmodules`](../.gitmodules)); the adapters in
[`decode/`](../crates/codec/src/decode/mod.rs) and
[`audio/`](../crates/codec/src/audio/mod.rs);
[codec-decode.md](codec-decode.md); the root [NOTICE](../NOTICE).

### 35. Every codec rivet decodes, it can encode — in software, in every build
**Decision.** VP9, VP8, MPEG-2, MPEG-4 Part 2 and ProRes are output codecs,
encoded by the clean-room crates' own encoders (§34) behind rivet's `Encoder`
trait — `encode/{vp9,vp8,mpeg2,mpeg4,prores}_sw.rs`, mirroring `h26x_sw` —
and written into the files that carry them:

| Codec | Single file | HLS / CMAF | What it is |
|---|---|---|---|
| VP9 | WebM (`V_VP9`, default), MP4 (`vp09` + `vpcC`) | yes (`vp09` init segment, `CODECS="vp09.…"`) | profile 0, 8-bit 4:2:0 (profile 2, 10-bit, since §38) |
| VP8 | WebM (`V_VP8`, default), MP4 (`vp08` + `vpcC`) | refused: no CMAF binding | 8-bit 4:2:0 |
| MPEG-2 | MP4 (`mp4v`, `esds` object type 0x61, default), QuickTime | refused | Main Profile, 8-bit 4:2:0, I/P/B |
| MPEG-4 Part 2 | MP4 (`mp4v`, `esds` 0x20 with the VOL, default), QuickTime | refused | Simple, or Advanced Simple with B-VOPs, 8-bit 4:2:0 |
| ProRes (6 profiles) | QuickTime only (`apco` … `ap4x`, `ftyp qt  `) | refused | intra-only, 4:2:2 / 4:4:4, 8- or 10-bit, HDR-tagged |

The file defaults to the codec's own (`VideoCodecPolicy::default_container`);
`container=mp4|mov|webm` picks another, and `validate()` refuses a codec in a
file that does not carry it, by name. WebM is written by rivet's own Matroska
muxer (`container::webm`), with Opus audio; a QuickTime movie is the MP4
muxer's box tree under `ftyp qt  `.

**Why software, and why with no feature.** No hardware backend here is wired
for these codecs, so their software encoder is not a fallback below silicon
— it is the encoder. §5's rule ("a missing GPU must not silently become a slow
CPU path") guards a choice that does not exist for them, so `select_encoder`
builds them directly, in every build, and `encode_capable` says no card takes
them (the encode pool is software, the hardware backends refuse them by name).
The `-fallback` features still mean what they meant for AV1, H.264 and H.265.

**What follows from them being software.** They take the serial single-file
path, one encoder per rung: the multi-GPU chunk-and-stitch engine (§9) runs
the web set only (`VideoCodecPolicy::chunkable`) — chunks would buy nothing on
cards that cannot encode the codec, and MPEG-2's open GOPs would not stand
alone. Their rate control is their crates': a fixed quantiser for VP9 / VP8
(no bitrate rungs; VP9 codes an average-bitrate rung since §38), the profile's frame size for ProRes (no crf, no bitrate),
an average rate for MPEG-2 / MPEG-4 (no CBR, no buffer); each limit is refused
by name before a frame is decoded. Their quality targets map onto their
quantisers through the H.26x QP table (`tuning::native_sw_quantizer`) — a first
mapping, not a VMAF calibration.

**What the pipeline decides for them.** Every encoder is handed 4:2:0 at 8 or
10 bits (§12's normalisation), so ProRes 4:2:2 / 4:4:4 is upsampled from it in
the adapter, and a 4:2:2 ProRes source round-trips with its chroma halved
vertically on the way. VP9, VP8, MPEG-2 and MPEG-4 are 8-bit SDR here (VP9's
encoder wrote profile 0 only; since §38 VP9 is 8- or 10-bit SDR); ProRes is 10-bit with HDR. MPEG-2 and MPEG-4
code B pictures reference-first; the adapters stamp each picture with its own
frame's timestamp and the muxers write the composition offsets.

**How it is verified.** Without any other implementation: every output is read
back with rivet's own demuxers and decoded with rivet's own decoders, and
checked for codec, size, frame count, timestamps and luma PSNR against the
source (`crates/codec/tests/native_codec_containers.rs`,
`crates/rivet/tests/new_codecs_e2e.rs`; [testing.md](testing.md)).

**Where.** [`encode/`](../crates/codec/src/encode/mod.rs) (`native_backend_for`,
the `*_sw.rs` adapters, `native.rs`);
[`spec/policy.rs`](../crates/rivet/src/spec/policy.rs) (`VideoCodecPolicy`,
`Container`); [`container/src/webm.rs`](../crates/container/src/webm.rs),
[`mux/video_track.rs`](../crates/container/src/mux/video_track.rs),
[`vpx.rs`](../crates/container/src/vpx.rs),
[`mpeg_es.rs`](../crates/container/src/mpeg_es.rs);
[codec-encode.md](codec-encode.md), [container.md](container.md),
[output-spec.md](output-spec.md).

## Filters

### 36. `hqdn3d` and `nlmeans` are clean-room rewrites
**Decision.** The temporal `hqdn3d` and the non-local-means kernel (both the
`nlmeans` filter and `denoise=nlmeans`) were replaced on 2026-10-03 by
implementations written from scratch. Their option syntax — names, positional
order, defaults, ranges, the derivation of omitted `hqdn3d` strengths — is
unchanged and stays compatible with the familiar command-line spelling; their
output is not meant to match any other implementation.

**Why.** The previous files carried comments naming internal functions and
constants of another project's filters of the same names, one of which is
GPL-licensed, which suggested they had been ported from that source. That is
incompatible with the clean-room rule this workspace follows everywhere else
(§3, §27, §34): no reading other implementations' source, no derived code. The
old bodies were deleted without being consulted. `nlmeans` was rebuilt from
the published papers (Buades, Coll & Morel, CVPR 2005 and IPOL 2011; the
offset-major integral-image speed-up of Wang et al. 2006 and Darbon et al.
2008). `hqdn3d` has no paper: it is our own design — an edge-preserving
recursive low-pass, swept both ways along rows and columns and then blended
into the previous output frame, with a retention that falls with the sample
difference — built from the public, user-facing description of such a filter
and its public option documentation. Their old hand-written SIMD paths went
with them; the new kernels are scalar, multi-threaded and deterministic.

**How it is verified.** Without any other implementation: a quality suite on
synthetic pictures with known clean content measures PSNR gain against
Gaussian noise across noise levels and strengths, edge sharpness (10–90 %
rise of a step), temporal convergence and flicker on a static scene, motion
pass-through, odd sizes and determinism; the fast `nlmeans` kernel is held bit
for bit to a direct evaluation of its formula
(`crates/codec/src/filter/denoise_quality_tests.rs`,
`crates/codec/src/filter/denoise/{nlmeans,hqdn3d}.rs`).

**Where.** [filters/nlmeans.md](filters/nlmeans.md),
[filters/hqdn3d.md](filters/hqdn3d.md) (each with a Provenance section and
the measured figures).

## Audio codecs

### 37. Every audio codec is the workspace's own, and every one is an output
**Decision.** No third-party audio codec remains in any build: Opus is
`crates/opus` (rivet-opus, in place of libopus through `audiopus`), MPEG audio
`crates/mp3` (rivet-mp3, in place of minimp3 and LAME, §21), Vorbis
`crates/vorbis` (rivet-vorbis, in place of lewton), beside the AAC (§26),
AC-3 / E-AC-3, DTS and FLAC / ALAC (§27) crates. Each is clean-room (written
from its specification; no implementation's source read), in its own
repository (§34), and the codec crate adapts it. And every codec rivet decodes
is an output, asked for by `audio=`:

| `audio=` | Codec | Files | Layouts, rates | Default rate |
|---|---|---|---|---|
| `opus` | Opus (CELT, VBR, 20 ms) | MP4 / MOV (`Opus` + `dOps`), WebM (`A_OPUS`), HLS, `.opus` (Ogg), `.m4a` | 1–8 ch (family 0 / 1), 48 kHz | 64k per mono, 96k per coupled stream |
| `vorbis` | Vorbis I (VBR by quality) | WebM (`A_VORBIS`, Xiph-laced `CodecPrivate`), `.ogg` | 1–8 ch, 8–192 kHz | `audio-quality` 5 |
| `ac3` | AC-3 | MP4 / MOV (`ac-3` + `dac3`), HLS (`ac-3`), `.m4a` | `acmod` 1/0–3/2 ± LFE, 48 / 44.1 / 32 kHz | 192k stereo, 448k 5.1 (Table 5.18's rates) |
| `eac3` | E-AC-3 | MP4 / MOV (`ec-3` + `dec3`), HLS (`ec-3`), `.m4a` | as AC-3 | 192k stereo, 384k 5.1 (32k–6144k) |
| `dts` | DTS core | MP4 / MOV (`dtsc` + `ddts`), HLS (`dtsc`), `.m4a` | `AMODE` 0, 2, 5–9 ± LFE, 48 / 44.1 / 32 kHz | the full rate (1536k at 48 kHz; Table 5-7's rates) |
| `he-aac`, `he-aacv2` | HE-AAC (v2) | as AAC | §26 | 48k / 32k stereo |

**Why.** The codecs were the last C and the last run-time library in the
build: libopus needed CMake (and `CMAKE_POLICY_VERSION_MINIMUM` under CMake 4),
minimp3 a C compiler, and LAME a library on the host behind a feature. With
them gone a build is Rust only (the `image` feature's libwebp aside, until
§38 removed it), MP3
encoding needs no feature, and every audio path is verified the same way the
video ones are (§35): the output read back with rivet's demuxers and decoded
with rivet's decoders. AC-3, E-AC-3 and DTS output exist because the
crates now encode them, and broadcast, disc and home-theatre pipelines want
them; Vorbis because WebM takes it and the crate encodes it.

**What follows.**
- **Files.** Vorbis has no MP4 or CMAF mapping (ISO/IEC 14496-12 and 23000-19
  define none), so `audio=vorbis` is WebM or Ogg only. Rivet now writes an Ogg
  file (`container::ogg`: the pages are rivet-vorbis's RFC 3533 writer, the
  Opus and Vorbis mappings rivet's) and reads one: audio-only output takes
  `audio-container=ogg`, and an `.ogg` / `.opus` is an input. Ogg's granule
  positions carry the Opus pre-skip and the end of the stream, so the length
  is exact, as in an MP4 edit list; WebM has no end trim, so a WebM's last
  Opus or Vorbis packet plays whole.
- **Layouts.** AC-3, E-AC-3 and DTS code A/52's arrangements and the DTS
  core's (the same set: 1/0 to 3/2 with or without the LFE); 5.1 goes out as
  5.1(side), a 6.1 source's back centre is split into the side pair, and 7.1 is
  downmixed (`remix::surround_core_layout`) — except to E-AC-3, which writes
  7.1 as a 3/2 + LFE independent substream and a 2/0 dependent one on the
  back surrounds (`remix::eac3_layout`), with a `dec3` naming both and a
  decoder that puts them back together (from 2026-10-03). The encoder is told the speakers, not only a count
  (`AudioEncoderConfig::layout`), since four channels are 4.0, quad(side) or
  3.1 to these codecs.
- **Configuration from the stream.** `dac3`, `dec3` and `ddts` are built from
  the encoder's first frame (`AudioInfo::from_ac3_frame` /
  `from_dts_frame`), exactly as a demuxer builds them for a Matroska or
  transport-stream source.
- **Delay and length.** Each encoder that resamples does it through
  `AlignedResampler` (delay trimmed, length exact), so every output's edit list
  states the codec's own priming only — Opus 312 samples, MP3 1057, AAC 1024,
  HE-AAC 3586, AC-3 256, DTS 512, Vorbis none — and every one presents exactly
  the input's length, which `crates/rivet/tests/audio_codecs_e2e.rs` checks
  for each codec in each file.
- **Settings.** `audio=he-aac|he-aacv2|vorbis|ac3|eac3|dts` on every surface
  (CLI, HTTP API and its OpenAPI document, batch manifest, IPC);
  `audio-quality` (−1 to 10) for Vorbis, which refuses `audio-bitrate`;
  `audio-container=ogg`; each codec's bit rates checked by `validate()`, and a
  layout a codec cannot carry (`audio-channels=7.1` with AC-3) refused by name.

**Where.** [`codec::audio`](../crates/codec/src/audio/mod.rs) (`encode/*.rs`,
`decode/*.rs`, `remix.rs`, `resample.rs`),
[`container::ogg`](../crates/container/src/ogg.rs),
[`container::webm`](../crates/container/src/webm.rs),
[`rivet::job::audio`](../crates/rivet/src/job/audio.rs),
[`spec/policy.rs`](../crates/rivet/src/spec/policy.rs); [codec-encode.md](codec-encode.md#the-audio-pipeline-decode--opus--aac--he-aac--mp3--vorbis--ac-3--e-ac-3--dts--flac--alac),
[output-spec.md](output-spec.md#3-audio--with_audioaudiocodecpolicy).

## Video and image codecs

### 38. AV1 and every still-image codec are the workspace's own; rav1e, rav1d and the `image` crate are gone
**Decision.** On 2026-10-03 the last third-party codecs left the build. AV1 is
`crates/av1` (rivet-av1): its decoder replaced rav1d, its encoder rav1e, and
with rivet's own HEIF writer (`crates/rivet/src/avif.rs`) it replaced ravif
for AVIF. The still-image codecs are `crates/png` (rivet-png, with its own
DEFLATE), `crates/jpeg` (rivet-jpeg), `crates/webp` (rivet-webp, lossy
through rivet-vp8; it landed in the same change) and `crates/imagecodecs`
(rivet-gif, rivet-bmp, rivet-tiff), in place of the `image` crate (and with it png,
jpeg-decoder, zune-\*, gif, tiff and image-webp), jpeg-encoder, and the
`webp` crate with Google's libwebp. Each is clean-room and in its own
repository, as §34 asks; the third-party crates left on media paths are not
codecs — moxcms (ICC colour management), rubato (resampling), `mp4` and
`matroska-demuxer` (container parsing), candle (`dpir`) — plus openh264 behind
`openh264-fallback`.

**Why.** §34's rule — a codec we need and cannot take with a clean licence and
no build cost, we write — had two exceptions left, and both cost something:
libwebp was the last C in the build (a C compiler for the `image` feature),
and rav1d and rav1e brought NASM for their assembly features, a decoder with a
known hang that needed a dev-profile workaround (`debug-assertions = false`
for rav1d), and a software AV1 encoder that was 8-bit only. Owning AV1 also
lets rivet write AVIF itself, so the still-image path has one AV1 encoder,
not two.

**What replaced what, and the features.**
- **Decode.** The `av1` decoder (`decode/av1_sw.rs`) is always in the decode
  chain behind NVDEC / AMF / QSV, ungated, like the `h26x`, VP8, VP9, MPEG-1 /
  MPEG-2, MPEG-4 and ProRes decoders (§5): a decoder that is not asked costs
  nothing. It takes the whole specification, bit-exact on all 244 AOM test
  vectors and all 3,015 Argon conformance streams, and gives 8 / 10 / 12-bit
  4:2:0 / 4:2:2 / 4:4:4 (monochrome as 4:2:0 with neutral chroma, film grain
  applied). It is single-threaded scalar — about 6 megapixels a second on one
  core on streams that use the whole toolbox, some 7 fps at 720p and 3 fps at
  1080p (rivet's own encoder's simpler output decodes at about 23) — and the
  crate's API offers no
  tile or frame parallelism, so the adapter runs it on its own worker thread
  three temporal units ahead of the caller, to overlap the decode with
  conversion, scaling and encoding (`RIVET_AV1_DECODE_THREAD=0` decodes on the
  caller's thread). That bounds throughput at the decoder's own rate; making
  it faster is the crate's work, not the adapter's. AVIF input now decodes in
  every `image` build, with no feature.
- **Encode.** `encode/av1_sw.rs`, backend `av1` (`EncoderBackend::Av1`, the
  name in the capabilities report, `/v1/health`, the OpenAPI enum and
  `TRANSCODE_ENCODER_BACKEND`, where `rav1e` is still accepted). Profile 0,
  8- **and** 10-bit 4:2:0, one tile (at most 4096 wide); the quality target
  becomes a quantiser of 4 × the libaom cq-level (`tuning::av1_sw_params`;
  a CRF is multiplied by 4); an average-bitrate rung is coded by the crate's
  rate control; `rate=cbr` and a coded picture buffer are refused by name. The
  crate writes no colour description into the sequence header (the MP4 `colr`
  box carries the colour), so its capability is 10-bit SDR, not HDR; and it
  has no forced-keyframe call, so `force_keyframe_next` starts a fresh encoder.
  `reset` is supported, for the session pool.
- **Features.** `av1-sw-fallback` is the policy switch §5 describes for the
  encoder (the encoder is always compiled and can always be asked for by
  name). `rav1e-fallback` stays as its alias and `rav1d-fallback` as a no-op,
  so existing build scripts still build; `rav1e-asm` / `rav1d-asm` are gone,
  and NASM now matters only for openh264. `thumbnail` pulls the `av1` crate
  alone; `image` adds the still-image crates and moxcms.
- **Speed.** A new setting, `video-speed` = `draft` | `standard` (default) |
  `archive`, sets the speed tier of every rung beneath the encode policy (an
  `encode-policy` `speed=` word wins); the old `speed=` / `preset=` keys stay
  refused and their message names `video-speed`. For AV1 the tier is the
  motion search range (±8 / ±16 / ±32). For VP9, whose encoder moved on at the
  same time (profiles 0–3, rate-distortion partition and transform search,
  GOLDEN references, one- and two-pass rate control), `standard` is the
  crate's speed 2 with fixed 16x16 partitions and ±16 search, about 10 fps at
  352x288 on one core, because speed 1 — the RD search — runs at about 1.7 fps
  there; that is `archive` (±32), and `draft` is speed 2 with fixed 32x32
  partitions and ±8. VP9 now also takes 10-bit (profile 2, SDR) and an
  average-bitrate rung.
- **Still images.** AVIF is written by rivet: `ftyp` avif / mif1 / miaf, a
  `meta` with `pict` handler, `pitm`, `iloc`, `iinf`, `iref` and the
  `ispe` / `pixi` / `av1C` / `colr` (nclx: BT.709 primaries, sRGB transfer,
  BT.601 matrix, full range) / `auxC` properties; alpha as an auxiliary item
  at three quarters of the colour quantiser; image quality 1–100 mapped to
  `base_q_idx` through anchors (1→255, 20→205, 40→162, 60→120, 80→72, 90→44,
  100→1); 8-bit 4:2:0, no ICC (so AVIF output is converted to sRGB, as
  before). A picture over 2048x2048, or wider than the encoder's 4096, is
  written as a `grid` of equal tiles of at most 2048 pixels, encoded in
  parallel — which is also the only parallelism the single-tile encoder gets.
  JPEG output is progressive with optimised Huffman tables, 4:2:0, ICC kept
  with `image-keep-icc`. PNG's compression is now a dial: `image-speed` (1–10)
  maps to a DEFLATE level (1→9, 2→8, 3→7, 4–6→6, 7→5, 8→4, 9→3, 10→1; the
  default 6 is level 6), because level 9 in rivet-png takes up to 8.5 s on a
  2048x2048 picture, too slow for a default, and buys little: on a
  photo-like one, levels 6 and 9 gave the same 6.11 MB in 1.12 s and 1.59 s
  (level 1: 6.44 MB in 0.25 s). It used to be AVIF's effort, and
  AVIF now ignores it. Resampling is rivet's own Lanczos-3.
- **WebP.** rivet-webp (`crates/webp`, a submodule and workspace member,
  library `webp`; a `[patch]` makes its git dependency on rivet-vp8 the
  `crates/vp8` submodule) landed in the same change, so WebP is read and
  written by it, and libwebp and image-webp are gone. In: a still, or an
  animation's first frame as composited, with its ICC profile. Out: lossy at
  the job's quality (default 80), or lossless with `image-lossless`; ICC kept
  with `image-keep-icc`, EXIF through `metadata-keep` as before.
  `image-speed` also sets WebP's effort (rivet-webp's 0–6: 1–2→6, 3–4→5,
  5–6→4, 7–8→2, 9–10→0; the default 6 gives 4, the codec's own default). On
  a 160x120 synthetic picture at quality 90 it gives 37.29 dB RGB PSNR in
  520 bytes (AVIF 44.70 dB in 1,032, JPEG 37.98 dB in 1,907); lossless WebP
  round-trips exactly.

**Consequences.**
- No codec in a default or `image` build is third-party, and no feature
  needs a C compiler.
- Regressions, stated plainly: software AV1 encode is slower and codes worse than rav1e did at the same
  quantiser (about 10 fps at 352x288 and 2 fps at 1280x720, single tile, no
  assembly); software AV1 decode is bounded at about 6 megapixels a second on
  full-toolbox streams; HDR10 / HLG AV1 still
  needs a GPU (10-bit SDR AV1 now works on a CPU-only build with
  `av1-sw-fallback`, which rav1e could not do).
- Verification is rivet's own throughout: the AV1, VP9 and image outputs are
  read back with rivet's demuxers and decoded with rivet's decoders
  (`crates/rivet/tests/new_codecs_e2e.rs`, the image tests;
  [testing.md](testing.md)).

**Where.** [`decode/av1_sw.rs`](../crates/codec/src/decode/av1_sw.rs),
[`encode/av1_sw.rs`](../crates/codec/src/encode/av1_sw.rs),
[`encode/vp9_sw.rs`](../crates/codec/src/encode/vp9_sw.rs),
[`encode/tuning/`](../crates/codec/src/encode/tuning/mod.rs);
[`rivet/src/avif.rs`](../crates/rivet/src/avif.rs),
[`rivet/src/image/`](../crates/rivet/src/image/mod.rs) (`webp.rs` for
WebP, `raster.rs` for the resampler); `crates/{av1,png,jpeg,webp,imagecodecs}`;
the root [NOTICE](../NOTICE); [output-spec.md](output-spec.md),
[codec-encode.md](codec-encode.md), [codec-decode.md](codec-decode.md).
