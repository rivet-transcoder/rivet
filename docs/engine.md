# rivet engine internals

A "what + why" reference for the orchestration layer of the **`rivet` crate** —
the configurable job engine, the reactive multi-GPU scheduler, and the
CLI / HTTP / IPC front-ends that drive them. Where this doc explains *how the
pieces fit and why each exists*, its siblings cover the bookends:

- The **end-to-end data flow** (demux → decode-once pump → per-rung scale →
  multi-GPU lease engine → mux, with a diagram, the hook points, the
  audio-only and still-image paths) lives in [pipeline.md](pipeline.md). This
  doc does not re-narrate it.
- The **config surface** — `OutputSpec`, `Rung`, `Quality`, `ColorPolicy`,
  `BitDepth`, presets, and `validate()` — lives in
  [output-spec.md](output-spec.md). This doc references those types but does not
  document the builder API.
- **Hooks** — every kind, its event, policies and reports — are in
  [hooks.md](hooks.md). The [Hooks](#hooks-hooks) section below covers where
  the engine emits them.
- The user-facing **CLI flags** are in [cli.md](cli.md); the **HTTP wire API**
  is in [api.md](api.md). The "Front-ends" section below describes the *internals*
  and links out for the surface detail.

## What the `rivet` crate is, and why

`rivet` is the top crate of the workspace. The two lower crates do the heavy
lifting — `codec` (decode/encode dispatch, colorspace, probe, GPU detection) and
`container` (demux, mux, CMAF, HLS) — and `rivet` is the *orchestration* that
turns them into a usable product: a job model, a uniform progress callback, a
fair multi-GPU scheduler, and three front-ends (a CLI, an HTTP service, a Unix
socket). The README frames the motivation directly: FFmpeg is a CLI and a C
library, **not a service** — no job model, no structured per-rendition progress,
no HTTP surface, and getting GPU encode/decode right across vendors means
hand-picking `-hwaccel` flags that silently fall back to software when wrong.
rivet ships that missing orchestration layer.

The crate also exposes a small **facade** (`transcode_file` / `transcode_bytes`)
for the trivial "one file in, one file out" case, a configurable **job
engine** (`run_job`, and `run_splice_job` for several clips joined into one
output) for everything else, and — behind the `image` feature — a **still-image
job** (`image::run_image_job`). The output defaults to a deliberate,
royalty-clean target: **AV1 video + Opus/AAC-passthrough audio in MP4** (or
CMAF/HLS). **H.264 / H.265** are also selectable output codecs for legacy-player
compatibility — they carry the patent-licensing obligations AV1 was chosen to
avoid. AV1 is the recommended default, asserted as such at the
facade (see [`lib.rs`](../crates/rivet/src/lib.rs) module docs and
[output-spec.md](output-spec.md#5b-output-codec--with_video_codec)).

---

## Module map

| File | Purpose |
|------|---------|
| [`lib.rs`](../crates/rivet/src/lib.rs) | The facade + crate re-exports; declares the AV1+Opus+MP4 output policy. |
| [`job/`](../crates/rivet/src/job/) | The job engine: `run_job` / `run_job_blocking` / `run_splice_job` and `JobOutput` / `RungOutput` / `RungArtifact` (`mod.rs`); single-file orchestration (`run.rs`); HLS orchestration (`pump.rs`); audio prep (`audio.rs`); audio-only output (`audio_only.rs`); splice clips and trims (`splice.rs`); subtitle tracks (`subtitles.rs`). |
| [`transcode.rs`](../crates/rivet/src/transcode.rs) | The one-shot single-file path (demux→decode→normalize→encode→mux); `TranscodeOutcome`, `AudioHandling`. |
| [`decode_pump.rs`](../crates/rivet/src/decode_pump.rs) | The shared decode-once pump + fan-out — *the rung benefit*; `FrameNormalizer`, frame-rate-cap decimation, decode-range planning, the decoded-frame / encoder-frame hook calls. |
| [`fit.rs`](../crates/rivet/src/fit.rs) | Fitting the source into each rung's box (`Fit`, `Orientation`, `Placement`, `fit_rungs`). |
| [`multigpu/`](../crates/rivet/src/multigpu/) | The multi-GPU orchestrator: the ladder core (`ladder.rs`: range-split decode, scalers, ladder workers serving every rung, the drain/abort loop), the HLS and single-file units (`hls.rs`, `single_file.rs`), and the policy → device-set and pool helpers (`gpu_policy.rs`). |
| [`gpu_pool.rs`](../crates/rivet/src/gpu_pool.rs) | `GpuPool` / `GpuLease` — one-encoder-per-GPU reservation (the NVENC deadlock fix), or software slots on a host with no usable encode silicon. |
| [`frame_queue.rs`](../crates/rivet/src/frame_queue.rs) | `SegmentChunkQueue` — bounded single-producer / multi-consumer chunk queue. |
| [`rung_scaler.rs`](../crates/rivet/src/rung_scaler.rs) | Per-rung scaler task: fit + scale → group K frames into a `SegmentChunk`. |
| [`encoder_worker/`](../crates/rivet/src/encoder_worker/) | Per-segment (HLS, `cmaf_worker.rs`) and per-chunk (single-file, `chunk_worker.rs`) encode, the `RungCodecInvariant` check (`invariant.rs`), and the per-worker encoder session pool (`session_pool.rs`). |
| [`cmaf_util.rs`](../crates/rivet/src/cmaf_util.rs) | CMAF/HLS orchestration helpers: segment flushing, contribution merge, bandwidth, codec strings. |
| [`cmaf_validate.rs`](../crates/rivet/src/cmaf_validate.rs) | A programmatic CMAF segment-alignment validator, for callers. |
| [`ladder.rs`](../crates/rivet/src/ladder.rs) | `standard_ladder` — derive an ABR rung set from a source resolution. |
| [`per_title.rs`](../crates/rivet/src/per_title.rs) | Per-title quality: measure how many bits the content needs. |
| [`progress.rs`](../crates/rivet/src/progress.rs) | `ProgressSink` / `RungProgress` / `RungStatus` / `JobEvent` + `fn_sink` / `channel_sink`. |
| [`hooks/`](../crates/rivet/src/hooks/) | Hooks: the kinds and their events (`kinds.rs`), sessions, policies and reports (`mod.rs`), the built-in digest / fingerprint hooks, and the `frame` / `phash` building blocks. See [hooks.md](hooks.md). |
| [`image/`](../crates/rivet/src/image/) | Still images in and out, and stills from a video (`image` feature). |
| [`probe.rs`](../crates/rivet/src/probe.rs) | `MediaInfo` facade — inspect an input without transcoding. |
| [`validate.rs`](../crates/rivet/src/validate.rs) | Advisory input-policy gates + `needs_chroma_downsample`. |
| [`thumbnail.rs`](../crates/rivet/src/thumbnail.rs) | Opt-in single-frame AVIF thumbnail capture; the frame capture the image job takes stills from a video with. |
| [`settings.rs`](../crates/rivet/src/settings.rs) | `TranscodeSettings` — the single surface-agnostic knob set + `into_spec` / `into_spec_for` / `into_image_spec`. |
| [`spec/`](../crates/rivet/src/spec/) | The `OutputSpec` config surface — see [output-spec.md](output-spec.md). |
| [`output_dir.rs`](../crates/rivet/src/output_dir.rs) | An output directory made for one run, removed again if the run puts nothing in it. |
| [`manifest.rs`](../crates/rivet/src/manifest.rs) | The batch manifest DSL (`batch` feature) — see [batch.md](batch.md). |
| [`main.rs`](../crates/rivet/src/main.rs) + [`commands/`](../crates/rivet/src/commands/) | The CLI: argument parsing in `main.rs`, one module per subcommand in `commands/`. |
| [`server/`](../crates/rivet/src/server/) | The HTTP transcode API (`rivet serve`, behind the `server` feature). |

---

## The facade & entry points

**What.** [`lib.rs`](../crates/rivet/src/lib.rs) declares every engine module,
re-exports the `codec` and `container` crates wholesale, and *flattens* the most
common types to the crate root so a caller can `use rivet::{run_job, OutputSpec,
RungProgress, …}` without knowing which module they live in
([`lib.rs:83`](../crates/rivet/src/lib.rs)). Hooks and the image job are not
flattened: they are reached as `rivet::hooks::…` and `rivet::image::…`. Three
tiers of entry point exist:

- **Facade**: `transcode_file(in, out)` / `transcode_bytes(&[u8])` (re-exported
  from [`transcode.rs`](../crates/rivet/src/transcode.rs)) and `probe_file` /
  `probe_bytes`. The trivial case in one call.
- **Job engine**: `run_job(...).await` / `run_job_blocking(...)` /
  `run_job_blocking_owned(...)`, and `run_splice_job` for a list of `Clip`s
  (from [`job/`](../crates/rivet/src/job/mod.rs)) — the configurable path that
  takes an `OutputSpec` and a `ProgressSink`.
- **Image job** (`image` feature): `image::run_image_job` /
  `run_image_job_with_hooks` (from [`image/`](../crates/rivet/src/image/mod.rs))
  — takes an `ImageSpec`; synchronous, for a blocking context.

**Why.** The two component crates were extracted so the generic transcoding
logic is reusable; the facade re-exports them (`pub use codec; pub use container;`,
[`lib.rs:80`](../crates/rivet/src/lib.rs)) so downstream code depends on a single
`rivet` crate yet can still reach the full low-level API (custom `EncoderConfig`,
segment-level `CmafVideoMuxer`, etc.). The flattening at the root is purely
ergonomic — the README's quick-start examples assume it.

**Output policy.** The default output codec is **AV1 (video) + Opus / AAC
passthrough (audio) muxed into MP4** — "a deliberate, royalty-clean target."
**H.264 and H.265** are also selectable for legacy-player compatibility (they
carry the patent-licensing obligations AV1 avoids). Input may be anything
`container` + `codec` can demux and decode. `VideoCodecPolicy` is accordingly
an enum — `Av1` (the default) plus `H264` / `H265` — a *selectable* output
dimension (`OutputSpec::with_video_codec` / `--codec` / `codec=`, values
`av1|h264|h265`; see [`spec/policy.rs`](../crates/rivet/src/spec/policy.rs)),
resolved to the encoder's `VideoCodec`, rather than a hard-coded constant.

---

## The job engine (`run_job`)

**What.** [`run_job`](../crates/rivet/src/job/mod.rs#L125) is the configurable
entry point: it takes input `Bytes`, an `&OutputSpec`, an optional output
directory, and an `Arc<dyn ProgressSink>`, and drives the whole pipeline to a
`JobOutput`. `run_job_blocking` ([`job/mod.rs:551`](../crates/rivet/src/job/mod.rs))
is the sync wrapper that builds a multi-threaded Tokio runtime and `block_on`s
it; `run_job_blocking_owned` is the same for a caller that hands over its
`Bytes` (what `rivet transcode` uses, so a large input is not copied).

The flow inside `run_job`:

1. When the spec has hooks, start a session (unless the caller started one)
   and run the **source** hooks on the input bytes, off the runtime, before
   anything parses them ([`job/mod.rs:131`](../crates/rivet/src/job/mod.rs)).
   The rest runs in `run_job_inner`, under `Hooks::run` (see
   [Hooks](#hooks-hooks)).
2. `spec.validate()` (fail fast on an impossible request — e.g. HDR on a build
   with no 10-bit encoder). `OutputMode::AudioOnly` goes to
   [`audio_only::run`](../crates/rivet/src/job/audio_only.rs) here.
3. Demux just the header + audio track + subtitle tracks once
   ([`job/mod.rs:234`](../crates/rivet/src/job/mod.rs)). An input with no
   video under a single-file spec becomes its audio-only form instead of an
   error.
4. **Fit the rungs** to the source (`fit_to` → `OutputSpec::with_rungs_fitted`,
   [`crate::fit`](../crates/rivet/src/fit.rs)): each rung's `WxH` is a box the
   upright source — sample aspect ratio applied, crop/pad filters accounted for
   — is fitted into; rungs that collapse onto the same output are merged and the
   progress sink is remapped so a caller still sees the rung indices it asked
   for. Then resolve the per-rung policy, and refuse a colour policy or a source
   depth the build cannot honour before a frame is decoded.
5. Run the **probe** hooks on the demuxed header.
6. Resolve `DecodePolicy::FastestGpu` by benchmarking each decode-capable GPU
   on a prefix of the input; emit `JobEvent::Started` / `Probed` on the sink.
7. Resolve the effective frame rate (source rate, clamped by
   `spec.max_frame_rate` — the pump drops frames to meet a cap), default the
   rate of any constant-rate rung, and take `frames_total` from the container
   header when known.
8. `prepare_audio` once ([`job/audio.rs:279`](../crates/rivet/src/job/audio.rs))
   — the audio is shared across every rung, not re-decoded per rung — plus a
   stereo downmix for an HLS stereo fallback when asked.
9. Prepare the video filter chain once (loading any overlay images).
10. Branch on `spec.mode`: `OutputMode::SingleFile` → `run_single_file` (then
    write any `metadata_keep` categories into each file);
    `OutputMode::Hls { segment_seconds }` → `run_hls`.
11. Emit `JobEvent::Finished` and assemble `JobOutput`; `Hooks::run` then
    runs the **artifact** hooks on each output and the **completed** hooks, or
    the **failed** hooks on any error.

**Key types.** `RungArtifact` ([`job/mod.rs:63`](../crates/rivet/src/job/mod.rs))
is either `File(Vec<u8>)` (single-file MP4 — or audio-only file — bytes held in
memory) or `HlsRendition { dir, relative_dir }` (a directory of CMAF segments +
a media playlist). `RungOutput` ([`job/mod.rs:75`](../crates/rivet/src/job/mod.rs))
wraps one rung's artifact with its label/dims/frames/bytes. `JobOutput`
([`job/mod.rs:86`](../crates/rivet/src/job/mod.rs)) collects the completed
rungs plus HLS-only fields (`hls_root`, `master_playlist`), source/audio
metadata, the fitted `renditions`, and the hooks' `HookReport`. A *failed* rung
is **not** in `JobOutput.rungs` — it is reported through the sink as
`RungStatus::Failed`, and the job only hard-errors if *every* rung failed
([`job/run.rs:269`](../crates/rivet/src/job/run.rs),
[`job/pump.rs:197`](../crates/rivet/src/job/pump.rs)).

`run_splice_job` ([`job/mod.rs:601`](../crates/rivet/src/job/mod.rs)) is the
same engine over several `Clip`s (each an input with an optional trim): every
clip is demuxed and probed (its own source and probe hook events, carrying its
position), the first clip drives the output configuration, the clips' audio and
subtitles are joined, and one spliced decode pump feeds either the serial
single-file path or the HLS engine. Audio-only output and `metadata_keep` are
refused for a splice.

### SingleFile orchestration

`run_single_file` ([`job/run.rs:27`](../crates/rivet/src/job/run.rs)) has **two
strategies** and picks between them:

- **Multi-GPU chunk-and-stitch** when the encode policy spreads (anything but
  `SingleGpu`), the frame count is known (from the header, else duration ×
  rate), the encode pool has capacity > 1, *and* there is no trim
  ([`job/run.rs:77`](../crates/rivet/src/job/run.rs); `RIVET_FORCE_CHUNKED=1`
  forces it on a one-card host, to exercise the path). It calls
  `run_single_file_multigpu` → `multigpu::run_multigpu_single_file`, which
  chunks each rung at GOP boundaries, encodes the chunks across all GPUs, and
  returns ordered packet streams. `mux_rung_packets_to_mp4`
  ([`job/run.rs:372`](../crates/rivet/src/job/run.rs)) then stitches each
  rung's packets (plus shared audio and subtitles) into one MP4 — in memory, no
  disk round-trip. For H.264 / H.265 the parameter sets go out of band
  (`avc1` / `hvc1`) when every chunk wrote the same ones, else in band (`avc3` /
  `hev1`).
- **Serial decode-once fan-out** otherwise, and for every splice:
  `run_serial_single_file` ([`job/run.rs:207`](../crates/rivet/src/job/run.rs))
  runs one (spliced) decode pump that fans frames to one
  `encode_rung_single_file` worker per rung over a bounded mpsc channel
  (`FRAME_CHANNEL_CAPACITY = 8`,
  [`job/mod.rs:59`](../crates/rivet/src/job/mod.rs)), each scaling + encoding +
  muxing its own MP4 with a single encoder on the policy's GPU. A software
  encoder's threads are divided between the rungs rather than each taking the
  whole machine.

**Why two strategies.** The README's GPU-scheduling note explains it: chunking a
single file only pays off when there are multiple GPUs to spread the chunks
across; on a single-GPU host (or when the frame count is unknown so chunks can't
be planned) the lean serial path runs with no GOP-chunking overhead.
`EncodePolicy::SingleGpu` takes it on purpose even on a multi-GPU host, for a
seam-free, quality-target-accurate stream. What makes the chunked path safe to
stitch is that every chunk is an independently decodable IDR-led GOP and the
cross-vendor codec invariant keeps every chunk's decoder-init contract
identical ([`job/run.rs:274`](../crates/rivet/src/job/run.rs) doc comment);
`ChunkSeamMode::ParallelConstQp` additionally forces constant QP so the
quality is flat across the seams.

### HLS orchestration

`run_hls` ([`job/pump.rs:45`](../crates/rivet/src/job/pump.rs)) is thinner: it
computes the segment grid (timescale, per-frame ticks, `keyframe_interval`,
`segment_target_ticks`), turns a trim into a one-clip decode plan, hands
everything to `multigpu::run_multigpu_hls`, then **assembles the package**: it
settles each H.264 / H.265 rendition's sample entry (`avc1`/`hvc1`, or
`avc3`/`hev1` when a worker's encoder wrote other parameter sets), builds a
`VideoVariantSpec` per rung (measuring bandwidth and reading the `CODECS`
string — `av01.…`, `avc1.…` or `hvc1.…` — from each rung's `init.mp4` via
[`cmaf_util`](../crates/rivet/src/cmaf_util.rs)), muxes the shared audio into a
CMAF audio rendition (`build_audio_rendition`,
[`job/audio.rs:926`](../crates/rivet/src/job/audio.rs); with a stereo
fallback, a stereo rendition first and the surround beside it), builds one
WebVTT rendition per subtitle track, and calls
`container::hls::write_hls_package` to emit `master.m3u8` + per-variant
playlists. The orchestrator produces the segments; `job/` produces the
playlists around them.

### Audio prep

`prepare_audio` ([`job/audio.rs:279`](../crates/rivet/src/job/audio.rs)) runs
once and yields a `PreparedAudio` (stream info, packets, a `handling` label,
the output edit, and an MP3 encoder name to keep) shared by every rung. What
it does is decided by an `AudioRequest` built from the spec: the
`AudioCodecPolicy`, bitrate, filters, channel layout, lossless depth / FLAC
level, `he-aac`, `audio-decode-deny`, and where the track is going
(`AudioOutput`: single-file MP4, CMAF, a bare `.mp3`, a native `.flac`).

- **Pass through** when the output carries the codec and nothing asks for PCM
  (no filter, the source's own layout): under `Auto`, AAC / Opus / AC-3 / E-AC-3
  / DTS, and MP3 into a single-file MP4. The source's edit list is applied by
  whole packets. With the device metadata category not kept, the source
  encoder's name is cleared from a copied AAC or MP3 stream.
- **Decode and encode** otherwise — to Opus under `Auto`, or the codec a
  policy forces (`ForceOpus`, `ForceMp3`, `ForceAac`, `ForceHeAac`,
  `ForceHeAacV2`, `ForceVorbis`, `ForceAc3`, `ForceEac3`, `ForceDts`, `Flac`,
  `Alac`; MP3 for an `.mp3` file), every encoder the workspace's own. Decoding
  covers MP3, MP2, Vorbis, DTS, AC-3, E-AC-3, Opus, FLAC, ALAC, linear PCM,
  and AAC (HE-AAC and HE-AAC v2 in full; as their AAC-LC core under
  `he-aac=core`). rivet does not upmix.
- **Forced but unreachable** — a forced codec the source cannot be decoded for
  (or an HE-AAC track under `he-aac=passthrough`) passes through instead,
  where the output holds the source.
- **Refuse or drop** — a codec `audio-decode-deny` names is never decoded: the
  job is refused if the output needs its PCM. A track that can be neither
  carried nor decoded is refused when a filter, a layout, or a bare-file output
  needs it, and otherwise **dropped** with a warn.

`AudioCodecPolicy::Drop` removes audio. A `with_audio` rejection at mux time
degrades to **video-only with a warn** rather than losing the customer's video
([`job/run.rs:409`](../crates/rivet/src/job/run.rs)).

### Audio-only output

`audio_only::run` ([`job/audio_only.rs:61`](../crates/rivet/src/job/audio_only.rs))
serves `OutputMode::AudioOnly` and a single-file job whose input has no video.
It reads the track with `container::streaming::demux_audio` (no video demuxer,
no decode pump, no rungs), runs the probe hooks, runs the same `prepare_audio`
toward the file it writes — an `.mp3` (a gapless `Info` frame carrying the
encoder delay and padding), a native `.flac`, or an `.m4a` — and reports it as
one rung, so artifact hooks and `metadata_keep` apply as for a single file. A
trim is refused.

> **Gotcha:** the job-engine audio path (`prepare_audio`) and the one-shot path's
> `wire_audio` ([`transcode.rs:332`](../crates/rivet/src/transcode.rs)) are
> independent implementations of the same routing policy. They must stay in sync.

---

## The single-shot path (`transcode.rs`)

**What.** [`transcode_bytes`](../crates/rivet/src/transcode.rs#L107) is the
primary library entry for "one buffer in, one AV1/MP4 buffer out." It is a
straight-line, single-threaded loop — demux → `create_decoder` (wrapped in
`RotatingDecoder`) → per-sample `push_sample`/`decode_next` →
`FrameNormalizer::normalize` → `send_frame` → `receive_packet` → `add_packet` →
drain → `finalize` ([`transcode.rs:194`](../crates/rivet/src/transcode.rs)).
`transcode_file` wraps it with file read/write. The result is a
`TranscodeOutcome` ([`transcode.rs:41`](../crates/rivet/src/transcode.rs))
carrying input/output metadata, frame/packet counts, an `AudioHandling`
([`transcode.rs:66`](../crates/rivet/src/transcode.rs)) tag, and elapsed time.

**Why it exists separately from the job engine.** It is the no-ceremony path: no
rungs, no GPU pool, no scaling, no progress sink, no hooks, no async runtime.
The module doc is explicit that for segmented CMAF-HLS or an ABR ladder you
should drive the job engine (or the `container`/`codec` crates) instead. It
targets the source resolution (upright), caps the frame rate at 60 (by
retiming, not by dropping frames), and uses a fixed 2-second keyframe interval
([`transcode.rs:134`](../crates/rivet/src/transcode.rs)). It exists so the
trivial case stays trivial and so there's a reference implementation of the
decode→encode→mux loop without the orchestration noise.

**Notes / decisions.**
- It normalizes frames with the job engine's own `FrameNormalizer`, built
  (`transcode_plan`, [`transcode.rs:292`](../crates/rivet/src/transcode.rs))
  from `DecodePumpConfig::for_source` under the default policy (`--color sdr`,
  bit depth auto), so it makes the same picture of a source as `run_job`: the
  source's resolved colour rather than the decoder's tag, a BT.601 source
  re-matrixed and tagged BT.709, a PQ / HLG source tonemapped. (It used to call
  `convert_to_yuv420p_bt709` on the decoder's tag and write the source's
  colour metadata.) A 10-bit SDR source keeps its depth, so a build whose AV1
  encoders are 8-bit refuses it up front, naming `--pixel-format 8bit`.
- It honours the source's presentation edit (hidden frames, a late start) and
  a source's held frames, as the job engine does.
- Audio is wired inline by `wire_audio`
  ([`transcode.rs:332`](../crates/rivet/src/transcode.rs)) with the same
  passthrough/transcode/drop routing as the job engine's `Auto`.
- Both `transcode_bytes` and the job engine honor
  `TRANSCODE_ENCODER_BACKEND=nvenc|amf|qsv|h26x|av1` (`rav1e` still accepted) as a backend override
  ([`transcode.rs:151`](../crates/rivet/src/transcode.rs),
  [`job/run.rs:526`](../crates/rivet/src/job/run.rs)).

---

## The multi-GPU engine — the heart

This is the core orchestration: decode the source **once** — split across the
cards where the bitstream allows — fan frames out to N per-rung scalers, and
keep **every** GPU on whichever rung is furthest behind. The README calls it
"the rung benefit"; [pipeline.md](pipeline.md#4-the-multi-gpu-lease-engine--the-rung-benefit)
has the diagram. Below is the *why* of each component.

### Which GPUs do what

Worth stating plainly, because "multi-GPU engine" and "decode once" sit next to
each other everywhere in this document and it is easy to read them as the same
claim:

| Phase | GPUs used | Chosen how |
|---|---|---|
| Demux | none | CPU |
| **Decode** | **one per range** — every card, when the source splits; exactly one otherwise | the `DecodePolicy`'s pinned card wins (`gpu:N`, or the benchmarked winner under `--decode fastest`); else ranges round-robin over the decode-capable cards among the policy's GPUs, and a single pump takes the first of them |
| Normalize (downsample / tonemap / filters) | none | CPU, once per frame, before fanout |
| Scale | none | CPU, per rung |
| **Encode** | **all of them** | a fair lease pool (software slots on a host with no usable encode silicon); one ladder worker per lease serving every rung — a CMAF segment per unit for HLS, a chunk of several GOPs for single-file |

**Why encode spreads.** Encoding is embarrassingly parallel along two axes at
once — rungs are independent of each other, and within a rung, segments are
independent because the encoder is forced to an IDR at every boundary. So any
GPU can take any unit of work and the results still stitch. The `GpuPool` lease
is what keeps it one-encoder-per-card; the ladder worker is what keeps every card
busy: it holds its lease for the whole job and takes the next chunk of whichever
rung has the deepest queue, so a card idles only when the whole job is out of
work — never because "its" rung is blocked while another rung's chunks wait.

**Why decode is split, and only sometimes.** A decoder is a serial dependency
chain — frame *n* generally cannot be produced without frame *n-1* — so a second
card can only help once the stream is cut into independently decodable pieces at
keyframe boundaries. [`plan_decode_ranges`](../crates/rivet/src/decode_pump.rs)
does exactly that: an index pass (demux only, no decode) finds the keyframes,
keeps the ones that fall on a **segment boundary** (a multiple of
`keyframe_interval`, so every range's first segment index is
`start_frame / keyframe_interval` exactly and the rung's numbering stays
continuous through the join), and cuts as evenly as those candidates allow —
one range per GPU. Each range's pump demuxes past the samples before it (the
demuxer has no seek; parsing is cheap, decoding is what is skipped), replays the
parameter sets in force at the split ahead of the range's first IDR (mp4 keeps
SPS/PPS in `avcC` and the demuxer emits them once, at the top), and flushes the
decoder at the range's end.

It only happens for the codecs whose keyframes and parameter sets can be read out
of a sample — H.264 and H.265 — and only when the source is un-spliced and
untrimmed (a range is addressed by demuxed sample and assumes segment 0 is the
start), the output frame rate is not capped below the source's (dropped frames
break the sample-to-segment arithmetic), the filter chain has no temporal
filter (a range would start with no frame history), and the leases are cards
rather than software slots
([`plan_ranges`](../crates/rivet/src/multigpu/ladder.rs#L340)). Anything else
decodes **once for the whole ladder**, which is still the larger win at these
ladder depths: a five-rung ladder does one decode, not five.

**The consequence when it does not split.** Once the ladder is wide enough, a
single decoder is the ceiling — every encoder can be idle waiting on it, and
adding GPUs does nothing. The giveaway is rungs of very different encode cost
sitting on the identical segment number. If you are looking at a host where the
cards are all busy but throughput is flat, that is the shape to check first, and
`--decode fastest` exists precisely because *which* card the single pump
lands on turns out to matter more than it should.

**Not the same thing as `EncodePolicy`.** [`spec::EncodePolicy`](../crates/rivet/src/spec/policy.rs)
says *which* GPUs the job may use (`AllGpus`, `PerRung`, `SingleGpu`, `Family`). Both the
encode pool and the decode ranges draw from that set — `SingleGpu` means one
range and one worker.

**And the shape is the caller's to choose — one enum per question.**
[`spec::DecodePolicy`](../crates/rivet/src/spec/policy.rs) is the whole decode
plan: `Auto` (split, one range per capable card), `Whole`, `SpecificGpu(i)`,
`FastestGpu`, `Ranges(n)` — which card(s) and whether the decode is split, in
one value, so "pin to card 2" and "split across every card" are not both
sayable. [`spec::EncodePolicy`](../crates/rivet/src/spec/policy.rs) is the
whole encode plan: `AllGpus` (every card, ladder-scheduled), `PerRung` (every
card, each pinned to its own rungs — "one rung, one GPU"), `Family(vendor)`,
`SingleGpu(idx?)` (one encoder per rung, serial). The defaults are the
measured-fastest shape; the alternatives are the control arms. Both apply to
HLS and to multi-GPU single-file, which run on the same core
([`multigpu/ladder.rs`](../crates/rivet/src/multigpu/ladder.rs)) with a
different unit of work.

### `decode_pump.rs` — decode once, fan out

**What.** [`run_spliced_decode_pump_blocking`](../crates/rivet/src/decode_pump.rs#L374)
demuxes + decodes the source one time (each clip in turn, for a splice; only
its `sample_range`, for a split decode) and fans every normalized frame out to a
`Vec` of per-rung mpsc senders; `run_shared_decode_pump_blocking` is the
one-whole-clip wrapper. Per decoded frame (`handle_frame`,
[`decode_pump.rs:643`](../crates/rivet/src/decode_pump.rs)) it places the frame
on the source's presentation edit and the clip's trim window, drops it when a
frame-rate cap leaves no output period starting on it (`DecodePumpConfig::decimate`,
from `decimation(source_fps, max_frame_rate)`), runs the job's
**decoded-frame** hooks, normalizes it, runs the **encoder-frame** hooks, and
fans it out.

Normalization is `FrameNormalizer` → `normalize_frame`
([`decode_pump.rs:868`](../crates/rivet/src/decode_pump.rs)): the frame is
tagged with the source's resolved colour space (not the decoder's — decoders
disagree), 4:4:4 → 4:2:0 downsampled (when `needs_downsample`), then — *only
when the spec's color policy says so* (`tonemap_to_sdr`) — HDR-aware
colorspace converted (`convert_to_sdr_bt709`, PQ/HLG → SDR BT.709); under a
passthrough/HDR policy only the layout changes, and an SDR source bound for an
HDR output is mapped into it (`sdr_to_hdr`, BT.2408). The bit depth is then
matched to the encoder's configured format, and last the spec's
[video filters](filters/README.md) run — a `codec::filter::FilterChain`
prepared once in `run_job` (loading any overlay images) and instantiated per
clip. The pump never decides to tonemap on its own; the caller sets the flag
from the `OutputSpec`'s `ColorPolicy`. `transcode_bytes` and the per-title
sample build the same `FrameNormalizer` (`DecodePumpConfig::for_source`), so
every entry point makes the same picture of a source.

**Why.** This is the entire performance argument for the crate: a 5-rung ABR
ladder decodes the input **once, not five times** (the naïve ffmpeg-per-rung
approach decodes N times). Fanout is cheap because `VideoFrame::clone()` is a
refcount bump — the pixel `Bytes` is `Arc`-backed
([`decode_pump.rs:1042`](../crates/rivet/src/decode_pump.rs)). Normalization is
done once *before* fanout precisely because it's identical for every rung; only
per-rung fitting and scaling differ, and that's pushed down to the scalers.

**Notes / gotchas.**
- The cost is **backpressure**: the slowest rung (usually the largest, whose
  encoder is slowest) throttles the pump (module doc,
  [`decode_pump.rs:10`](../crates/rivet/src/decode_pump.rs)).
- `fan_out` returns `false` *only when every sender is closed*
  ([`decode_pump.rs:1050`](../crates/rivet/src/decode_pump.rs)); a single rung
  giving up doesn't stop the pump.
- The loop is blocking (built for `spawn_blocking`); it bridges into the async
  `send().await` via a passed-in `tokio::runtime::Handle`
  ([`decode_pump.rs:1062`](../crates/rivet/src/decode_pump.rs)).
- A blocking hook's rejection (or its error, under `OnError::Reject`) surfaces
  as the pump's error and stops the run; a frame no hook wants costs only the
  sampling check.

### `gpu_pool.rs` — one encoder per GPU

**What.** [`GpuPool`](../crates/rivet/src/gpu_pool.rs#L61) is a process-wide
reservation pool: each detected GPU is a slot, callers `claim()` an available
slot and hold the returned `GpuLease` for the lifetime of their work, and the
lease's `Drop` releases the slot ([`gpu_pool.rs:190`](../crates/rivet/src/gpu_pool.rs)).
With N GPUs and M waiters, the first N get leases immediately; the rest park on a
Tokio `Semaphore` until a lease drops.

**Why — the load-bearing invariant.** The module doc is emphatic and dated: this
is the deliberate design decision from 2026-05-02 because **concurrent NVENC
sessions on the same CUDA context deadlocked at ~session 5/5 init** — the GPU
went idle and no frames encoded ([`gpu_pool.rs:10`](../crates/rivet/src/gpu_pool.rs)).
One-encoder-per-GPU is the invariant; the pool's job is to enforce it *while
still running encoders in parallel across GPUs*.

**Key mechanics.**
- **Vendor on the lease is load-bearing.** Each slot records its `GpuVendor`
  ([`gpu_pool.rs:70`](../crates/rivet/src/gpu_pool.rs)), and a lease says what
  it is through `LeaseKind::Gpu { index, vendor }`. Without it, a
  multi-vendor host (NVIDIA + Intel Arc, both exposing index 0) *always* picked
  NVENC because the encoder factory tries NVIDIA first; the Arc sat idle. The
  lease tells the factory which backend to use (test
  `lease_carries_vendor_for_dispatch`,
  [`gpu_pool.rs:515`](../crates/rivet/src/gpu_pool.rs)).
- **The encode pool drops cards that can't encode the requested output.**
  `gpu_pool_for_policy(policy, codec, output_pixel_format)`
  ([`multigpu/gpu_policy.rs:615`](../crates/rivet/src/multigpu/gpu_policy.rs))
  filters a multi-GPU selection through
  [`codec::encode::encode_capable_at(dev, codec, ten_bit)`](../crates/codec/src/encode/mod.rs)
  — the authoritative probe that runs the same `select_encoder` dispatch a worker
  uses, cached per `(index, codec, ten_bit)` (a card may encode H.264/H.265 but
  not AV1, or H.264 only at 8 bits). A card that can't encode the chosen codec
  at the output's depth (e.g. a **pre-Ada NVIDIA** that decodes via NVDEC and
  encodes H.264/H.265 but has no AV1 encode silicon — when the job asks for
  AV1) is dropped from the *encode* pool, so no worker leases it and hard-fails
  the run; the capable cards (the Arc) encode. It stays in `policy_gpu_indices`
  (intentionally **not** filtered), so the decode pump can still use it — a
  pre-Ada NVIDIA + Arc decodes on the NVIDIA (NVDEC) and encodes on the Arc
  (QSV) with no flags. When nothing is left, the call is an error naming why
  (pinning `--gpu` to an incapable card surfaces it up front instead of
  aborting mid-run) — unless the next point applies.
- **Software slots.** On a host with no usable encode silicon for the codec,
  whose build has a software encoder for it (`av1-sw-fallback`,
  `h26x-fallback`) and whose policy does not pin silicon, the pool is
  `GpuPool::software(slots, threads)`: the same lease discipline, but each
  lease (`LeaseKind::Software`) is a share of the CPU carrying the thread
  budget its encoder runs on, so `slots × threads` covers the machine once.
  `RIVET_SOFTWARE_SLOTS` overrides the slot count.
- **Sparse indices** are preserved (slot stores `GpuDevice.index`, not vec
  position) to handle `CUDA_VISIBLE_DEVICES=[0,2,5]`
  ([`gpu_pool.rs:64`](../crates/rivet/src/gpu_pool.rs), test `sparse_indices_preserved`).
- **`claim()` vs `try_claim()`.** `claim().await` is the blocking path the
  ladder workers take, and increments `pending_claimers` via a
  `PendingClaimGuard` RAII bracket ([`gpu_pool.rs:110`](../crates/rivet/src/gpu_pool.rs))
  — the guard decrements even if the await is cancelled. `try_claim()` is a
  non-blocking claim for spare capacity; it does *not* touch
  `pending_claimers`, because Tokio's `Semaphore` is FIFO and a permit freed
  while a real worker is parked is reserved for that worker — so `try_claim`
  can't steal it (test `try_claim_does_not_steal_from_blocked_claimer`,
  [`gpu_pool.rs:702`](../crates/rivet/src/gpu_pool.rs)). `pending_claimers()`
  ([`gpu_pool.rs:259`](../crates/rivet/src/gpu_pool.rs)) is the fairness
  signal for such a caller. Nothing in the engine calls `try_claim` today: the
  single-file helper dispatcher it was built for was replaced by the ladder
  workers.
- **CPU-only host:** an empty inventory makes `claim()`/`try_claim()` return
  `None` immediately so call sites need no special-casing
  ([`gpu_pool.rs:305`](../crates/rivet/src/gpu_pool.rs)).
- The free-slot scan is a lock-free CAS loop guarded by the semaphore count, so a
  successful acquire always finds a free slot — a `None` there is treated as an
  invariant violation (`unreachable!` / `expect`,
  [`gpu_pool.rs:326`](../crates/rivet/src/gpu_pool.rs)).

### `frame_queue.rs` — the bounded chunk queue

**What.** [`SegmentChunkQueue`](../crates/rivet/src/frame_queue.rs#L41) connects
one producer (a rung's scaler — or one per decode range) to N consumers (the
ladder workers). The unit of transfer is a `SegmentChunk`
([`frame_queue.rs:21`](../crates/rivet/src/frame_queue.rs)) tagged with a
monotonic `segment_idx` so each worker knows which output segment or chunk
it's producing. For HLS a chunk is one CMAF segment's worth of frames
(`keyframe_interval` frames); for multi-GPU single-file it is several GOPs
plus margin: `lead_in` frames replayed from the previous chunk to warm the
encoder and then discarded, and `keep` frames that reach the output.

**Why.** Single-producer / multi-consumer with **bounded** capacity for memory
safety: the pump/scaler blocks when the queue is full, workers block when it's
empty ([`frame_queue.rs:11`](../crates/rivet/src/frame_queue.rs)). The segment
index travels with the frames so the work is self-describing — a worker arriving
at a rung mid-flight just takes the queue head, no decode-and-discard.

**Notes.**
- `push_front` ([`frame_queue.rs:173`](../crates/rivet/src/frame_queue.rs)) is
  the **requeue** path: a worker that pops a chunk and then detects a cross-vendor
  codec-invariant mismatch puts the chunk back at the head (briefly exceeding
  capacity by 1) so a compatible worker picks it up. It decrements
  `popped_segments` so the `pushed > popped` "work remaining" predicate stays
  accurate.
- `depth()` / `try_pop()` are the ladder worker's interface: it reads every
  rung's depth, takes from the deepest, and never blocks on one queue.
  `pushed_segments()` ([`frame_queue.rs:73`](../crates/rivet/src/frame_queue.rs))
  is what the finalizers check coverage against once the scalers are done.
- The depth is per rung and derived from a **byte budget**
  (`QUEUE_BYTE_BUDGET`, [`multigpu/mod.rs`](../crates/rivet/src/multigpu/mod.rs)):
  a fixed count that was comfortable for one ladder is the OOM killer on a 4K
  six-rung one, so a rung keeps at least one chunk and never more than
  `QUEUE_CAPACITY`, trimmed where the budget says so.
- When the source is decoded in ranges, a rung's queue is fed by one scaler per
  range (`run_rung_scaler_blocking_shared`) and closed by the last of them —
  closing on the first exit would drain the workers while other ranges were
  still feeding.

### `rung_scaler.rs` — per-rung scale → chunk

**What.** [`run_rung_scaler_blocking`](../crates/rivet/src/rung_scaler.rs#L57):
one scaler per rung consumes normalized frames from the pump's fanout channel,
fits each to the rung — the rung's [`Placement`](../crates/rivet/src/fit.rs)
crops, bilinear-scales and pads in one pass (`colorspace::scale_region`; CPU
work, AVX2 where it pays), or a plain `scale_frame` resize when no placement
was planned — and groups `frames_per_chunk` frames (plus any `overlap` lead-in
replayed from the previous chunk) into a `SegmentChunk` with a monotonic
index, pushing into the rung's `SegmentChunkQueue`. A frame of another size
than planned (the next clip of a splice, a resolution change mid-stream) is
fitted into the same canvas afresh, since the encoder's size is fixed.

**Why.** Scaling is the one per-frame step that *is* per-rung, so it's pushed out
of the shared pump to here (each scaler runs on its own thread). On exit (the
input channel returns `None` because the pump closed all senders), the scaler
flushes the final partial chunk and — if it is the last producer on that queue
— **closes the queue** so encoder workers drain and exit cleanly
([`rung_scaler.rs:86`](../crates/rivet/src/rung_scaler.rs)).

### `encoder_worker/` — per-unit encode + the codec invariant

**What.** Two worker bodies share a config (`config.rs`), the invariant check
(`invariant.rs`), and `build_enc_config`:
- [`encode_segment_unit`](../crates/rivet/src/encoder_worker/cmaf_worker.rs#L129)
  (HLS path): encode one chunk → write one CMAF segment file. The ladder worker
  calls it in a loop with whichever rung's config the next chunk belongs to; the
  encoder is created per segment either way, so hopping rungs costs nothing
  extra. Each segment gets a **fresh `CmafVideoMuxer`**, configured with the
  segment's index + base decode time so the on-disk filename and `tfdt` match
  what a single-encoder pipeline would produce.
  `run_encoder_worker_blocking` is the same body draining one rung's queue.
- [`encode_chunk_unit`](../crates/rivet/src/encoder_worker/chunk_worker.rs#L100)
  (single-file path): the same shape, but *collects* the chunk's kept packets
  into a `ChunkPackets` ([`chunk_worker.rs:13`](../crates/rivet/src/encoder_worker/chunk_worker.rs))
  instead of writing a segment, so the orchestrator can stitch them into one
  MP4. `run_chunk_encoder_worker_blocking` is the one-rung loop around it.
  Its encoder comes from the worker's
  [`EncoderSessionPool`](../crates/rivet/src/encoder_worker/session_pool.rs):
  a chunk of the same rung as the last one reuses the session after
  `Encoder::reset` (the next frame is an IDR opening a closed GOP, nothing of
  the previous chunk survives); a rung hop, or a backend that cannot reset,
  builds a new one. `RIVET_ENCODER_POOL=off` rebuilds for every chunk, the
  behaviour before the pool, for comparison.

**Why the codec invariant.** [`RungCodecInvariant`](../crates/rivet/src/encoder_worker/invariant.rs#L35)
captures the decode-init fields every encoder contributing to one rendition
**must** agree on: for AV1 the mandatory sequence-header fields (`seq_profile`,
level/tier, bit depth, chroma subsampling, the four color fields, max frame
dims, …); for H.264 / H.265 the SPS profile, level, chroma format, bit depths
and dimensions (the `avcC` / `hvcC` contract). The reason is spelled out in the
type doc ([`invariant.rs:11`](../crates/rivet/src/encoder_worker/invariant.rs)):
another card may be of a different GPU *vendor* than the rung's first worker
(NVENC + QSV + AMF + the software AV1 encoder can all touch one rendition), and the player sets up
its decoder once from `init.mp4`'s `av1C`; if a later segment's inline OBU
sequence header disagrees on a mandatory field, strict decoders (dav1d in
conformance mode, Safari AVFoundation, hls.js+libdav1d) reject the segment. The
first worker on a rung **sets** the invariant; subsequent workers **compare**
on their first packet (`validate_or_set_rung_invariant`,
[`invariant.rs:201`](../crates/rivet/src/encoder_worker/invariant.rs)).

The check deliberately **ignores** cosmetic optional fields (timing info /
decoder-model presence, film-grain present flag, operating-point detail) so
cross-vendor encoders co-exist without byte-difference false rejections.

**The three outcomes** (`InvariantCheck`,
[`invariant.rs:182`](../crates/rivet/src/encoder_worker/invariant.rs)):
- `SetByThisWorker` / `Matched` → proceed to publish.
- `Mismatched` → the worker **requeues its chunk** (`push_front`) and leaves
  the rung to the others (a ladder worker strikes it off its list) — nothing of that worker's is lost, another (matching-vendor)
  worker picks the chunk up, and the run never aborts. This is the
  "mission-critical jobs do not abort" rule.
- An `Err` (parse failure: the encoder emitted no `OBU_SEQUENCE_HEADER`, or no
  SPS, at all) is a hard configuration bug that *does* fail the run — distinct from a soft
  mismatch.

**Notes.** Packets are buffered until the first-packet decision is made
([`cmaf_worker.rs:208`](../crates/rivet/src/encoder_worker/cmaf_worker.rs)): nothing is
committed to the muxer (and `init.mp4` is only written by `finalize`, which a
rejecting worker never calls) until validation passes, so a mismatched worker
discards everything in flight with no on-disk side effects.

### `multigpu/` — the orchestrator

**What.** Two near-mirror functions —
[`run_multigpu_hls`](../crates/rivet/src/multigpu/hls.rs) (returns one
`RungManifest` per rung) and
[`run_multigpu_single_file`](../crates/rivet/src/multigpu/single_file.rs) (returns one
`RungPackets` per rung) — wire the pieces together on one core,
[`multigpu/ladder.rs`](../crates/rivet/src/multigpu/ladder.rs). Both take a
`MultiGpuParams` ([`multigpu/mod.rs:142`](../crates/rivet/src/multigpu/mod.rs)) carrying
the input (or a splice plan), rungs, source/output color + pixel format, the
filter chain, the job's hooks (handed to every pump), the segment grid, the
`GpuPool`, the policy's GPU indices, the decode and encode plans, and an
optional cancel signal.

The orchestration, step by step:

1. **Pre-flight encoder probe** (`preflight_encoder`,
   [`ladder.rs:294`](../crates/rivet/src/multigpu/ladder.rs)):
   construct a throwaway encoder for the output codec and format, pinned to the
   pool's first card, to verify this host can produce it *before* spawning any
   workers. This fails fast with a clear error and — importantly — avoids
   dispatching workers that would fail at encoder construction, which on some
   drivers (Ampere with no AV1-encode silicon) would hang an *uncancellable*
   blocking task. A software pool is checked from the feature flags instead of
   by building an encoder.
2. **Per-rung shared state** (`Ladder`,
   [`ladder.rs:102`](../crates/rivet/src/multigpu/ladder.rs)):
   one `SegmentChunkQueue`, encoded-frame and encoded-byte `AtomicU64`
   counters, a `RwLock<Option<RungCodecInvariant>>` slot, a contributions
   `Mutex`, a `serving_workers` count, an `active_workers` count, a `rung_done`
   `Notify`, and a `finalized` flag — plus one abort signal for the ladder.
3. **Finalizers** (one task per rung): wait until nothing is working on the
   rung *and* nothing can be handed out — `active_workers == 0` **and** the
   rung's queue is closed and empty (a ladder worker's count legitimately
   returns to zero between chunks, so the count alone is not "finished") — then
   merge the contributions into a `RungManifest` (HLS) or stitch the packets
   into a `RungPackets` (single-file), checking **coverage** — exactly
   `total_segments` contiguous segments, no gaps or dupes.
4. **Decode pump(s)** (`plan_ranges` / `spawn_pumps`,
   [`ladder.rs:340`](../crates/rivet/src/multigpu/ladder.rs)):
   `plan_decode_ranges` cuts an un-spliced H.264/H.265 source into as many
   ranges as the `DecodePolicy` asks (by default one per card) at
   segment-aligned keyframes; one pump per range, each on its own card via
   `range_decode_gpu_for` (a pinned card wins, else the decode-capable cards
   among the policy's round-robin), each fanning out to every rung. A source
   that cannot be split — or a job with a frame-rate cap, a temporal filter, or
   software slots — decodes whole through one pump. HLS and single-file plan
   ranges the same way.
5. **Scalers**, one per (range × rung), numbering segments from the range's
   `first_segment_idx`; only the last range's scalers may mark a chunk final.
6. **Ladder workers**: one per GPU, claimed up front and held for the whole
   job. Each loops: pick the rung with the deepest queue, `try_pop`, encode the
   unit with that rung's config (a CMAF segment for HLS, a chunk of GOPs to
   packets for single-file), record the result, repeat; sleep 5 ms when every
   queue is empty; exit when every queue is closed and empty. A card whose
   vendor mismatches a rung's invariant hands the chunk back and strikes that
   rung off its own list. Both output paths run these same workers
   ([`ladder.rs`](../crates/rivet/src/multigpu/ladder.rs) is the one
   implementation; `hls.rs` and `single_file.rs` supply the unit).
7. **Drain loop**: a `biased` `tokio::select!` over the pump/scaler/worker
   `JoinSet`s, the finalizer channel, and the caller's **cancel signal**
   (`MultiGpuParams::cancel`, an optional `watch::Receiver<bool>`). The first
   error, or the signal turning true, **aborts** the ladder — every queue is
   closed *and emptied*, every finalizer is woken and returns without merging —
   and the loop then waits for the workers (the lease holders) to come back
   before returning the error, so the next job's `claim()` never finds a card
   still held by a run that is over. That wait is bounded by one unit of work.
   A cancel comes back with `multigpu::Cancelled` as the root cause
   (`err.is::<Cancelled>()`), so a consumer can tell "asked to stop" from
   "failed". Pumps and scalers stop on their own once the queues are closed
   (a scaler's push is refused, it exits, its receiver drops, its pump ends).

**Why furthest-behind rather than cheapest-first.** The shared pump stalls when
*any* rung queue is full. Serving the fullest queue attacks the rung closest to
blocking everyone and keeps decode moving; preferring the cheapest rung would
publish early quality sooner and then wedge the pump behind the rung nobody was
serving. And because a ladder worker serves every rung, no rung can be left
without a consumer however many there are — which is the condition the shared
pump needs, so the ladder costs one decode at any depth. (Before this, a ladder
longer than the GPU count fell back to one pump *per rung*.)

**Policy helpers.** `gpu_pool_for_policy` / `gpu_pool_for_serial` /
`policy_gpu_indices` / `serial_gpu_for_policy` / `serial_target`
([`multigpu/gpu_policy.rs`](../crates/rivet/src/multigpu/gpu_policy.rs))
translate an `EncodePolicy` (`AllGpus` / `PerRung` / `SingleGpu(idx)` /
`Family(vendor)`) into a concrete device set — so a `Family`/`SingleGpu`
constraint governs both encode *and* decode (the decode pump pins to the same
selected set). A selection with nothing capable left becomes software slots
when the build and policy allow, and otherwise a refusal naming why, before a
frame is decoded. `detect_gpu_pool` builds an unconstrained pool from the host
inventory. The policy helpers read
the host detected once per process, and an empty-pool refusal's list of cards
(which of them encode the codec at the output's depth) is probed once per codec
and depth: the first refusal on a loaded machine took seconds, and the
installed cards do not change during a run. The ladder's own refusals name
`MultiGpuParams::host` — `HostCards::Detected` for a run, a fixed inventory in
unit tests, whose time bounds are about the refusal and not the machine.

**Gotchas.**
- The `active_workers` count per rung is **seeded at 1** — a setup guard
  released once every scaler has been spawned. The finalizers are spawned first,
  and a finalizer's first act is to break out of its wait if the count is
  already zero; with a 0 seed the runtime only had to schedule a finalizer
  before its scaler's `fetch_add` to conclude "nobody is working on me" and
  return empty. Load-dependent: it hid on a two-rung three-second clip and
  showed up on a five-rung four-minute one.
- `QUEUE_CAPACITY = 2` is a ceiling; the depth a rung actually gets comes from
  `QUEUE_BYTE_BUDGET` (2 GiB across the ladder). `FANOUT_CHANNEL_CAPACITY = 4`
  is the pump → scaler slack.
- Stopping is the abort above, and only that. Dropping the future without it
  leaves blocking threads (scalers parked on a full queue, workers holding
  leases) and the queued frames — up to the whole byte budget — alive in tasks
  nobody joins. A long-lived service passes its shutdown watch as `cancel`;
  the CLI passes `None` and lets the process end.

### `cmaf_util.rs` — the CMAF/HLS glue

**What.** Shared helpers used by both the job engine and the orchestrator
([`cmaf_util.rs:1`](../crates/rivet/src/cmaf_util.rs)):
- `keyframe_interval_for_segment` / `total_segments_for_rung` (ceil-division
  segment count) — the segment-grid math.
- `add_packet_with_segment_flush` ([`cmaf_util.rs:39`](../crates/rivet/src/cmaf_util.rs)):
  flush the prior segment when the next packet is a keyframe *and* the buffered
  duration has reached the segment target — so **each segment opens on an IDR**,
  which is what keeps the ladder segment-aligned for clean ABR. It returns the
  segment it closed, so a caller can ship it before the rung is done. The audio
  counterpart flushes on the same time grid.
- `merge_rung_contributions` ([`cmaf_util.rs:83`](../crates/rivet/src/cmaf_util.rs)):
  combine several workers' segment lists for one rung into one ordered manifest,
  erroring on disagreeing dims/timescale, **duplicate** segment numbers, or
  internal **gaps** — the coverage guarantee the finalizer relies on.
- `measure_bandwidth` (avg/peak bits/sec for the HLS variant `BANDWIDTH`) and
  `codec_string_from_init` ([`cmaf_util.rs:184`](../crates/rivet/src/cmaf_util.rs)),
  which finds the visual sample entry in an `init.mp4` and recovers the exact
  `CODECS` string for the playlist from its config box: `av01.…` from `av1C`,
  `avc1.…` from `avcC`, `hvc1.…` / `hev1.…` from `hvcC`.

**Why.** These are the bits of CMAF bookkeeping that are identical whether the
single-encoder or multi-worker path produced the segments; centralizing them
keeps the segment-on-IDR rule and the merge/coverage logic in one place rather
than duplicated across `job/`, `encoder_worker/`, and `multigpu/`.

---

## ABR ladder (`ladder.rs`)

**What.** [`standard_ladder(src_w, src_h, max_short_side)`](../crates/rivet/src/ladder.rs#L59)
derives a sensible `Vec<Rung>` from a source resolution. It snaps to the standard
short-side quantizations (2160/1440/1080/720/480/360/240), preserves the source
aspect ratio, even-aligns every dimension (AV1 4:2:0 needs even dims), and caps
the top rung at `max_short_side` (default 1080). It also drops the standard
rung just under a source that is barely above it (within
`SOURCE_SNAP_TOLERANCE`, 15%): a 1920×818 source ships one 818 rung rather than
818 *and* 720. `standard_ladder_with_quality` stamps a `Quality` on every rung.

**Why.** It's the convenience path so callers don't hand-build a ladder for the
common case; the README and CLI `--ladder` flow through here. Callers who want
full control build `Rung`s by hand and skip the module entirely
([`ladder.rs:4`](../crates/rivet/src/ladder.rs)). The "p" number always refers to
the **short** side regardless of orientation, so portrait sources get correctly
labelled rungs (`1080p` for a 1080×1920 source — test
`ladder_portrait_short_side_labels`). The 1080 default cap is the web-safe ceiling;
lifting it to 1440/2160 unlocks QHD/4K rungs. `MIN_DIMENSION = 200`
([`ladder.rs:22`](../crates/rivet/src/ladder.rs)) drops rungs too small to be
worth a separate rendition.

---

## Progress reporting (`progress.rs`)

**What.** Every job streams progress through a
[`ProgressSink`](../crates/rivet/src/progress.rs#L103) — a tiny trait with
`on_rung(RungProgress)` (called repeatedly as each rung advances), an
optional `on_event(JobEvent)` for coarse lifecycle (`Started`, `Probed`,
`Finished`), and an optional `on_rung_complete(&RungManifest)` — HLS only,
called once per rung the moment its segments are all on disk, the seam for an
uploader that ships each rendition as it completes. [`RungProgress`](../crates/rivet/src/progress.rs#L35)
is the **uniform** per-rung struct (index, label, dims, `RungStatus`, percent,
frames done/total, segments, bytes, optional message) that a consumer can render
into a progress bar *without knowing the output mode*. `RungStatus`
([`progress.rs:18`](../crates/rivet/src/progress.rs)) is the lifecycle:
`Pending → Running → Finalizing → Completed`/`Failed`.

**Why a sink, not a return value.** Progress is emitted *as the job runs*, so it
needs a push channel. The trait is deliberately small and synchronous; to bridge
into async you wrap a Tokio mpsc with [`channel_sink`](../crates/rivet/src/progress.rs#L164)
(turning the callback into a `.recv().await` stream) or a closure with
[`fn_sink`](../crates/rivet/src/progress.rs#L139). `NullSink` drops everything.

**Notes.** `ChannelSink` uses `try_send` and **drops** updates when the channel
is full or closed ([`progress.rs:159`](../crates/rivet/src/progress.rs)) —
progress is advisory, never load-bearing, so it must never block or fail the job.
The same events back the CLI's progress lines, the HTTP API's job-status polling
(via `RegistrySink`), and the README's library examples.

---

## Inspection helpers

### `probe.rs` — inspect without transcoding

[`probe_bytes`](../crates/rivet/src/probe.rs#L106) (and `probe_bytes_shared`,
which takes `Bytes` without a copy) demuxes only the container header +
audio/subtitle-track metadata (no decode) and reports a
[`MediaInfo`](../crates/rivet/src/probe.rs#L16): container label, video codec,
upright and stored dims, rotation, sample aspect ratio, frame rate, duration,
pixel format, audio stream shape, and subtitle tracks. An input with no video
reports `video_codec: "none"` and its audio; with the `image` feature a still
image reports its format and size. It's the data source for `rivet probe`, for
the CLI/HTTP code resolving `--ladder` rungs and the source size
(`TranscodeSettings::into_spec_for`), and for `POST /v1/probe`. The container
label comes from `container::sniff_container`
([`probe.rs:119`](../crates/rivet/src/probe.rs)), the same magic-byte sniff the
demuxer dispatch uses, so the reported label matches the demuxer actually used.

### `validate.rs` — advisory input gates

[`validate_stream`](../crates/rivet/src/validate.rs#L52) checks a demuxed stream
against the reference resolution / frame-rate / duration / pixel-format policy
(`MIN_RESOLUTION`, `MIN_FRAME_RATE`, `MAX_DURATION_SECS`). **The job engine does
not call it** — the module doc is explicit that these are *advisory* so rivet
transcodes whatever it's given; they exist for policy-bearing callers (a hosted
service) to gate uploads with the same limits the reference transcoder uses
([`validate.rs:1`](../crates/rivet/src/validate.rs)). The one function the engine
*does* use is `needs_chroma_downsample`
([`validate.rs:141`](../crates/rivet/src/validate.rs)), which tells the pump
whether to run the 4:4:4 → 4:2:0 step.

### `thumbnail.rs` — opt-in AVIF still

[`generate_thumbnail`](../crates/rivet/src/thumbnail.rs#L65) (behind the
`thumbnail` feature) decodes the source up to a target frame (default 10% in, so
it's past intros/fades), turns it upright, converts it to 8-bit RGB with the
matrix and range the source declared (whatever pixel format the decoder
produced), and encodes a still **AVIF** with rivet's own AV1 encoder
(`crates/av1`) in rivet's own HEIF writer ([`avif.rs`](../crates/rivet/src/avif.rs);
until 2026-10-03, `ravif`). `DEFAULT_THUMBNAIL_SPEED` is kept for API
compatibility and no longer read. Two rationale notes from the module doc: a *separate* decode pass
(rather than tapping the variant decoders) gives an isolated failure mode — a
thumbnail miss never blocks the variant pipeline — and is cheap because it only
decodes up to the capture frame. The cost is that what the pump does to a frame
has to be mirrored here (rotation, non-4:2:0 formats and the source's colour
each once went missing). AVIF is chosen so the still reuses the same AV1
client-codec story as the video (every browser that plays the video plays the
thumbnail) without adding a JPEG/WebP encoder to the dep graph
([`thumbnail.rs:29`](../crates/rivet/src/thumbnail.rs)). Its frame capture
(`capture_frames`) is also how the image job takes stills from a video.

---

## Hooks (`hooks/`)

**What.** A spec carries a [`Hooks`](../crates/rivet/src/hooks/mod.rs) set
(`OutputSpec::with_hooks`): caller-supplied code of eight kinds, each a trait
written for one point of a job and handed exactly what exists there — source,
probe, decoded frame, encoder frame, still, artifact, completed, failed. Each
answers with a verdict (proceed or reject) and annotations, collected into a
`HookReport` on `JobOutput::hooks` (`ImageJobOutput::hooks` for an image job).
The kinds, their events, sampling, policies and built-ins are in
[hooks.md](hooks.md); this section is where the engine emits them.

**Where.**
- `run_job` / `run_splice_job` ([`job/mod.rs:125`](../crates/rivet/src/job/mod.rs),
  [`job/mod.rs:601`](../crates/rivet/src/job/mod.rs)) start a session when
  the spec has hooks (`Hooks::ensure_session`, unless the caller started one
  with `Hooks::session`), run the **source** hooks on each input's bytes off
  the async runtime, and wrap the rest in `Hooks::run`
  ([`hooks/mod.rs:1226`](../crates/rivet/src/hooks/mod.rs)), which — if the job
  returned and no hook rejected — builds the **artifact** events (each
  single-file output's bytes, each HLS rendition directory, the master
  playlist) when a hook wants them, then runs the **completed** hooks; on any
  error it runs the **failed** hooks and returns the rejection, if a hook made
  one, as the job's error (`hooks::rejection_of` finds it).
- **probe** hooks run in `run_job_inner` once the header is demuxed and the
  rungs fitted, in `run_splice_job` per clip, and in `audio_only::run` once
  the track is read.
- **decoded-frame** and **encoder-frame** hooks run in the decode pump
  (`handle_frame`, [`decode_pump.rs:643`](../crates/rivet/src/decode_pump.rs)),
  on either side of `FrameNormalizer` — so they see every pump: the serial
  single-file pump, each range's pump of a split decode, each clip of a
  splice. `DecodePumpConfig::hooks` and `MultiGpuParams::hooks` carry them
  there. Neither runs in an audio-only or image job, nor in `transcode_bytes`,
  which has no spec.
- **still** hooks run only in the image job (below).

**Blocking and background.** A **blocking** hook runs on the thread that
reached its point and its verdict takes effect at once (a rejected source never
reaches the demuxer; a rejected frame stops the pump). A **background** hook is
queued to the session's one worker thread (`rivet-hooks`, a bounded queue of 64
tasks, so a slow hook holds the pipeline up at the queue instead of piling
frames up in memory); its rejection stops the job at the next point it reaches,
and the job waits for the queue to drain before it finishes. An erroring hook
is recorded and then ignored or treated as a rejection, per its `OnError`.

**Building blocks.** `hooks::frame` gives a hook the frame in forms a model
takes: 8-bit luma or RGB, PGM/PPM, and `rgb8_resized` / `rgb8_letterboxed` /
`planar_f32_letterboxed`, which read only the source samples the output needs
straight from the decoder's planes (YUV 4:2:0 / 4:2:2 / 4:4:4 at 8 to 12 bits,
NV12 / NV21, RGB(A)), so their cost follows the model's input size rather than
the frame's. `hooks::phash` and `DigestAlgorithm` are the hashing the built-in
`SourceDigest`, `PerceptualFingerprint` and `ArtifactDigest` use.

## Still images (`image/`)

**What.** Behind the `image` feature,
[`run_image_job`](../crates/rivet/src/image/mod.rs#L594) takes input bytes and
an `ImageSpec` (built by `TranscodeSettings::into_image_spec` for
`mode=image`) and returns every rendition in every requested format. It is
synchronous and CPU-bound — call it from a blocking context — and it does not
go through `run_job`: there are no rungs, decode pump or GPU pool.
`run_image_job_with_hooks` runs a `Hooks` set around it (source → probe → one
**still** event per picture, as upright 8-bit RGBA → artifact per encoded file
→ completed / failed).

**How.** It sniffs the input. A still image (JPEG, PNG, WebP — an animation's
first frame — AVIF, GIF's first frame, TIFF, BMP, HEIC/HEIF) is decoded by
`image::decode` on the workspace's own codecs (`crates/jpeg`, `crates/png`,
`crates/webp` through `image/webp.rs`, `crates/imagecodecs`); HEIC and AVIF go
through `image::heif` and the same HEVC / AV1 decoder dispatch as video.
`ImageDecodeDeny` (`image-decode-deny`) refuses a format by name. A video gives
stills per `FrameSelection` (a poster 10% in, N evenly spaced, or at given
times) through `thumbnail::capture_frames`. Each picture is turned upright,
converted to sRGB unless its ICC profile is kept, planned per rendition with
the same [`crate::fit`](../crates/rivet/src/fit.rs) rules as a video rung but
on a one-pixel grid (renditions that collapse onto one size are made once),
resampled with rivet's own Lanczos-3 (`image/raster.rs`), and encoded to each
format — AVIF (rivet's own AV1 encoder and HEIF writer, `avif.rs`; a picture
over 2048x2048 or wider than 4096 as a `grid` of tiles encoded in parallel),
WebP (rivet-webp), JPEG (rivet-jpeg), PNG (rivet-png). Outputs are encoded
from pixels, so no source metadata reaches them unless `metadata_keep` names a
category, which is then written as a fresh EXIF block.

**Why separate.** A picture is a different job from a ladder: one frame, many
sizes and formats, sRGB rather than a video colour policy, alpha. Sharing the
fitting rules and the decoders keeps a still and a video rung of the same
source the same shape and the same colours. See
[output-spec.md](output-spec.md#11-still-images--modeimage).

---

## The front-ends, and the shared `TranscodeSettings`

Every front-end — the CLI, the HTTP API, the IPC header, and the batch
manifest — parses its own syntax into **one** canonical knob set,
[`TranscodeSettings`](../crates/rivet/src/settings.rs#L65), then calls
[`TranscodeSettings::into_spec`](../crates/rivet/src/settings.rs#L235) — the
**single** `OutputSpec`-building implementation (`into_spec_for` wraps it
against a probed source, turning a video-less input into its audio-only form;
`into_image_spec` builds the `ImageSpec` for `mode=image`). The module doc
states the design goal directly: add a new option *once* here (a field + a line
in `into_spec` + a `parse_*` function + an `apply_kv` arm) and every surface
picks it up, instead of maintaining a copy of the spec-building logic per
surface ([`settings.rs:1`](../crates/rivet/src/settings.rs)). It is also the
single point of *interpretation*: the CLI's clap value enums only list the
words, and a test pins them to the `parse_*` vocabulary. `into_spec` also
encodes the encode-plan precedence — an explicit `encode` plan > the legacy
`seam=serial` > pinned index > vendor family > single-gpu > all-gpus
([`settings.rs:330`](../crates/rivet/src/settings.rs)) — and calls
`spec.validate()` so an impossible request is rejected at the surface.

### CLI (`main.rs` + `commands/`)

[`main.rs`](../crates/rivet/src/main.rs) is a `clap` app; each subcommand's
body is a module of [`commands/`](../crates/rivet/src/commands/):

| Subcommand | What it does |
|------------|--------------|
| `transcode` | The main path: fills `TranscodeSettings` from flags → `into_spec` → `run_job_blocking_owned`, with a per-rung progress line and on-disk output placement ([`commands/transcode.rs:56`](../crates/rivet/src/commands/transcode.rs)). |
| `splice` | Concatenate (and per-clip trim, `PATH@START-END`) several inputs into one output through `run_splice_job_blocking` ([`commands/splice.rs:83`](../crates/rivet/src/commands/splice.rs)). |
| `image` | Still images (`image` feature): `into_image_spec` → `image::run_image_job`, one file per rendition × format (× still, for a video) in the output directory ([`commands/image.rs:64`](../crates/rivet/src/commands/image.rs)). |
| `probe` | `probe_file` → human table or `--json`. |
| `devices` | List detected GPUs (vendor, name, generation, VRAM, PCI address, per-codec encode verdicts, live NVML load on NVIDIA); on Linux also the PCI BAR line from `codec::gpu::bar_report` — whether Resizable BAR lets the CPU reach all of a discrete card's VRAM, and what a small BAR costs; `--json` available, with the BAR as `pci_bar` ([`commands/devices.rs:5`](../crates/rivet/src/commands/devices.rs)). |
| `capabilities` (alias `caps`) | What this *build + host* can encode/decode — enabled backends, max bit depth and HDR per output codec, per-codec decode backends, devices ([`commands/capabilities.rs:10`](../crates/rivet/src/commands/capabilities.rs)). |
| `pipe` | Stream stdin → stdout, no temp files; flags override quality/size/color/audio ([`commands/pipe.rs:33`](../crates/rivet/src/commands/pipe.rs)). |
| `ipc` | A Unix-domain-socket server (`ipc` feature, Unix only): per connection the client writes media, half-closes, and reads the transcoded MP4 back; an optional `#rivet k=v …\n` header line carries settings ([`commands/ipc.rs:36`](../crates/rivet/src/commands/ipc.rs)). |
| `batch` | Convert many files from a YAML/JSON manifest (`batch` feature) — see [batch.md](batch.md). |
| `serve` | The HTTP API (`server` feature) — delegates to `rivet::server::serve`. |

`pipe` and `ipc` share `stream_transcode` ([`commands/mod.rs:196`](../crates/rivet/src/commands/mod.rs)):
all-default settings take the fast `transcode_bytes` path; any set field routes
through `into_spec_for` + the single-file `run_job`. Both reject HLS output (a
single stream can't carry a segmented package). The `#rivet` header is split off by
`split_ipc_settings` and parsed via `TranscodeSettings::parse_kv_line`
([`settings.rs:652`](../crates/rivet/src/settings.rs)). `devices` /
`capabilities` reach straight into `codec::gpu` / `codec::encode` /
`codec::decode` for host/build introspection. The CLI runs no hooks; hooks are
a library and HTTP-server facility. **For every flag's meaning and defaults,
see [cli.md](cli.md).**

### HTTP server (`server/`)

[`server/`](../crates/rivet/src/server/mod.rs) (the `server` feature) is a small
axum app so another application can *signal* a transcode over the network:
routes and state in `mod.rs`, handlers in `handlers.rs`, the request types and
path handling in `spec.rs`, the OpenAPI document in `docs.rs`. The internals
worth knowing:

- **In-memory job registry.** `AppState` holds an `RwLock<HashMap<Uuid,
  Arc<JobHandle>>>` and the server's hooks ([`server/mod.rs:66`](../crates/rivet/src/server/mod.rs)).
  Each `JobHandle` tracks phase (`queued` / `running` / `completed` / `failed`
  / `rejected`), per-rung progress, artifacts, error, HLS output dir, the
  fitted renditions, and the job's hook session. The module doc is candid that
  completed single-file artifacts are held in RAM until process exit — fine
  for a sidecar/worker, not a public CDN; a production deployment would offload
  from a `ProgressSink` watching `RungStatus::Completed`
  ([`server/mod.rs:27`](../crates/rivet/src/server/mod.rs)).
- **`RegistrySink`** ([`server/mod.rs:241`](../crates/rivet/src/server/mod.rs)) is the
  `ProgressSink` impl that mirrors per-rung updates into the `JobHandle` so
  `GET /v1/jobs/{id}` can report them — the same progress plumbing the CLI uses,
  pointed at a registry slot instead of stdout.
- **Two submission shapes.** `POST /v1/transcode` branches on `Content-Type`
  ([`server/handlers.rs:98`](../crates/rivet/src/server/handlers.rs)): `application/json` →
  a structured `TranscodeRequest` (input from a server **file path** or inline
  **base64**, an optional server `output.path`, a structured `spec`, and
  `hooks`); any other content type → a streamed binary body with the spec (and
  `hooks=a,b`) in **query params**. Both forms collapse onto
  `TranscodeParams::to_settings` → `TranscodeSettings::into_spec_for` (against
  a probe of the media), reusing the shared `settings::parse_*` vocabulary so
  the API carries no copy of the spec logic
  ([`server/spec.rs:129`](../crates/rivet/src/server/spec.rs)). The API runs
  video and audio-only jobs; `mode=image` is refused there.
- **Hooks.** A server built with `build_router_with_hooks` / `serve_with_hooks`
  runs its required hooks on every job and the optional ones a request names;
  a request chooses among configured hooks only and cannot define one. Each
  job gets a hook session keyed by its job id, its status carries the hook
  report, and a job a hook rejected ends `rejected` (a `?sync=true` request
  gets `422`). `GET /v1/hooks` lists the configured hooks
  (`Hooks::describe`). `rivet serve` itself starts the server with no hooks;
  see [hooks.md](hooks.md#http-api).
- **File-path I/O + sandbox.** `resolve_path` ([`server/spec.rs:432`](../crates/rivet/src/server/spec.rs))
  canonicalizes request-supplied paths; when `RIVET_FILE_ROOT` is set, a path
  must resolve *under* that root or it's rejected ("path escapes
  RIVET_FILE_ROOT sandbox"). With no root set, the server (bound to
  `127.0.0.1:8080` by default) treats paths as trusted-local. The HLS
  `/files/{*path}` route has its own `..`/empty-component traversal guard
  ([`server/handlers.rs:379`](../crates/rivet/src/server/handlers.rs)).
- **Sync vs async.** Default is fire-and-forget: `202 { job_id }` and the job
  runs in a spawned task; poll `GET /v1/jobs/{id}`. `?sync=true` (or `"sync":
  true`) runs the job inline and returns the artifact directly when the job
  made exactly one single-file artifact held in RAM (MP4, or the audio file of
  an audio-only job), or the status JSON otherwise — several rungs, HLS, or
  artifacts written to `output.path`
  ([`server/handlers.rs:312`](../crates/rivet/src/server/handlers.rs)).
- Ships hand-authored OpenAPI 3.0 + Swagger UI + Redoc at `/openapi.json` /
  `/swagger` / `/redoc`.

**For the endpoint/wire reference, see [api.md](api.md).**

---

## Key decisions in the engine — recap

- **Decode once, fan out.** The pump decodes the source a single time and clones
  `Arc`-backed frames to every rung — the ladder is cheap. Normalization
  (4:4:4→4:2:0, policy-driven HDR→SDR, bit depth, filters) happens once,
  pre-fanout, because it's rung-agnostic.
  ([`decode_pump.rs`](../crates/rivet/src/decode_pump.rs))
- **One encoder per GPU.** `GpuPool` enforces it because concurrent NVENC sessions
  on one CUDA context deadlocked at init (2026-05-02). Work still runs in parallel
  *across* GPUs, and the lease carries the GPU **vendor** so multi-vendor hosts
  dispatch correctly; a host with no usable silicon gets software slots under
  the same discipline. ([`gpu_pool.rs`](../crates/rivet/src/gpu_pool.rs))
- **Ladder workers, deepest rung first.** One worker per GPU serves every rung
  for the whole job, so a card idles only when the job is out of work and a
  ladder of any depth costs one decode; the decode itself is split across the
  cards at segment-aligned keyframes when the bitstream allows.
  ([`multigpu/ladder.rs`](../crates/rivet/src/multigpu/ladder.rs))
- **Single-file on the same core.** Multi-GPU single-file runs the same ladder
  workers with a different unit — a chunk of several GOPs, encoded to packets
  on a session reset between chunks, stitched per rung in order.
  ([`multigpu/single_file.rs`](../crates/rivet/src/multigpu/single_file.rs))
- **Cross-vendor codec invariant.** A per-rung decoder-init contract (the AV1
  sequence header, or the H.264/H.265 SPS) lets NVENC + QSV + AMF + the software AV1 encoder
  contribute to one rendition safely; a card that mismatches hands the chunk
  back and leaves that rung to the others without aborting the job.
  ([`encoder_worker/invariant.rs`](../crates/rivet/src/encoder_worker/invariant.rs))
- **Fail fast, don't degrade.** A pre-flight encoder probe (for the requested
  codec and format) rejects a host with no matching encode silicon up front (and
  dodges an uncancellable hang on some drivers); `spec.validate()` and the
  source checks reject impossible color/depth combos before a frame is decoded.
- **Single-file chooses its engine.** When it helps (a spreading encode policy,
  more than one capable card, a known frame count, no trim), single-file
  chunk-encodes across GPUs and stitches in memory; otherwise it takes a lean
  serial decode-once path. Independent IDR-led chunks and the codec invariant
  make the stitch safe; `ParallelConstQp` flattens quality across the seams.
  ([`job/run.rs`](../crates/rivet/src/job/run.rs))
- **Fit, don't stretch.** A rung's size is a box the source is fitted into,
  keeping its display shape, never enlarged unless asked.
  ([`fit.rs`](../crates/rivet/src/fit.rs))
- **Hooks at fixed points.** Caller code runs where the pipeline already has
  what it needs — the bytes, the header, a decoded or encoder-bound frame, a
  still, an output — and can stop the job; a job without hooks pays nothing.
  ([`hooks/`](../crates/rivet/src/hooks/mod.rs))
- **One knob set, every surface.** CLI, HTTP, IPC and the batch manifest all
  parse into `TranscodeSettings` → `into_spec`, so a new option is added once.
  ([`settings.rs`](../crates/rivet/src/settings.rs))
- **One picture of a source.** The one-shot `transcode_bytes` normalizes with
  the job engine's own `FrameNormalizer` under the default policy, so it makes
  the same picture `run_job` does; the job engine's pump follows the spec's
  `ColorPolicy` (`tonemap_to_sdr`, SDR → HDR). See
  [output-spec.md](output-spec.md#4-color--bit-depth) for the policy surface.
