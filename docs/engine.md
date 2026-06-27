# rivet engine internals

A "what + why" reference for the orchestration layer of the **`rivet` crate** —
the configurable job engine, the reactive multi-GPU scheduler, and the
CLI / HTTP / IPC front-ends that drive them. Where this doc explains *how the
pieces fit and why each exists*, two siblings cover the bookends:

- The **end-to-end data flow** (demux → decode-once pump → per-rung scale →
  multi-GPU lease engine → mux, with a diagram) lives in
  [pipeline.md](pipeline.md). This doc does not re-narrate it.
- The **config surface** — `OutputSpec`, `Rung`, `Quality`, `ColorPolicy`,
  `BitDepth`, presets, and `validate()` — lives in
  [output-spec.md](output-spec.md). This doc references those types but does not
  document the builder API.
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
for the trivial "one file in, one file out" case, and a configurable **job
engine** (`run_job`) for everything else. The output is a deliberate,
royalty-clean target: **AV1 video + Opus/AAC-passthrough audio in MP4** (or
CMAF/HLS). That policy is load-bearing for the project and is asserted at the
facade (see [`lib.rs`](../crates/rivet/src/lib.rs) module docs and
[output-spec.md](output-spec.md#a-note-on-the-output-codec)).

---

## Module map

| File | Purpose |
|------|---------|
| [`lib.rs`](../crates/rivet/src/lib.rs) | The facade + crate re-exports; declares the AV1+Opus+MP4 output policy. |
| [`job.rs`](../crates/rivet/src/job.rs) | The job engine: `run_job` / `run_job_blocking`; `JobOutput` / `RungOutput` / `RungArtifact`; SingleFile vs HLS orchestration + audio prep. |
| [`transcode.rs`](../crates/rivet/src/transcode.rs) | The one-shot single-file path (demux→decode→colorspace→encode→mux); `TranscodeOutcome`, `AudioHandling`. |
| [`decode_pump.rs`](../crates/rivet/src/decode_pump.rs) | The shared decode-once pump + fan-out — *the rung benefit*. |
| [`multigpu.rs`](../crates/rivet/src/multigpu.rs) | The reactive multi-GPU orchestrator: lease pool + mid-flight helper dispatch + cross-vendor codec invariant. |
| [`gpu_pool.rs`](../crates/rivet/src/gpu_pool.rs) | `GpuPool` / `GpuLease` — one-encoder-per-GPU reservation (the NVENC deadlock fix). |
| [`frame_queue.rs`](../crates/rivet/src/frame_queue.rs) | `SegmentChunkQueue` — bounded single-producer / multi-consumer chunk queue. |
| [`rung_scaler.rs`](../crates/rivet/src/rung_scaler.rs) | Per-rung scaler task: scale → group K frames into a `SegmentChunk`. |
| [`encoder_worker.rs`](../crates/rivet/src/encoder_worker.rs) | Per-segment encoder worker + the `RungCodecInvariant` check. |
| [`cmaf_util.rs`](../crates/rivet/src/cmaf_util.rs) | CMAF/HLS orchestration helpers: segment flushing, contribution merge, bandwidth, codec strings. |
| [`ladder.rs`](../crates/rivet/src/ladder.rs) | `standard_ladder` — derive an ABR rung set from a source resolution. |
| [`progress.rs`](../crates/rivet/src/progress.rs) | `ProgressSink` / `RungProgress` / `RungStatus` / `JobEvent` + `fn_sink` / `channel_sink`. |
| [`probe.rs`](../crates/rivet/src/probe.rs) | `MediaInfo` facade — inspect an input without transcoding. |
| [`validate.rs`](../crates/rivet/src/validate.rs) | Advisory input-policy gates + `needs_chroma_downsample`. |
| [`thumbnail.rs`](../crates/rivet/src/thumbnail.rs) | Opt-in single-frame AVIF thumbnail capture. |
| [`settings.rs`](../crates/rivet/src/settings.rs) | `TranscodeSettings` — the single surface-agnostic knob set + `into_spec`. |
| [`spec.rs`](../crates/rivet/src/spec.rs) | The `OutputSpec` config surface — see [output-spec.md](output-spec.md). |
| [`main.rs`](../crates/rivet/src/main.rs) | The CLI (`transcode`/`probe`/`devices`/`capabilities`/`pipe`/`ipc`/`serve`). |
| [`server.rs`](../crates/rivet/src/server.rs) | The HTTP transcode API (`rivet serve`, behind the `server` feature). |

---

## The facade & entry points

**What.** [`lib.rs`](../crates/rivet/src/lib.rs) declares every engine module,
re-exports the `codec` and `container` crates wholesale, and *flattens* the most
common types to the crate root so a caller can `use rivet::{run_job, OutputSpec,
RungProgress, …}` without knowing which module they live in
([`lib.rs:65`](../crates/rivet/src/lib.rs)). Two tiers of entry point exist:

- **Facade**: `transcode_file(in, out)` / `transcode_bytes(&[u8])` (re-exported
  from [`transcode.rs`](../crates/rivet/src/transcode.rs)) and `probe_file` /
  `probe_bytes`. The trivial case in one call.
- **Job engine**: `run_job(...).await` / `run_job_blocking(...)` (from
  [`job.rs`](../crates/rivet/src/job.rs)) — the configurable path that takes an
  `OutputSpec` and a `ProgressSink`.

**Why.** The two component crates were extracted so the generic transcoding
logic is reusable; the facade re-exports them (`pub use codec; pub use container;`,
[`lib.rs:65`](../crates/rivet/src/lib.rs)) so downstream code depends on a single
`rivet` crate yet can still reach the full low-level API (custom `EncoderConfig`,
segment-level `CmafVideoMuxer`, etc.). The flattening at the root is purely
ergonomic — the README's quick-start examples assume it.

**Output policy.** The `lib.rs` module doc states it outright: the output codec
is **AV1 (video) + Opus / AAC passthrough (audio) muxed into MP4** — "a
deliberate, royalty-clean target." Input may be anything `container` + `codec`
can demux and decode. This is why `VideoCodec` is an enum with a single `Av1`
variant (a *selectable dimension* for the future, see
[`spec.rs`](../crates/rivet/src/spec.rs)) rather than a hard-coded constant.

---

## The job engine (`run_job`)

**What.** [`run_job`](../crates/rivet/src/job.rs#L93) is the configurable entry
point: it takes input `Bytes`, an `&OutputSpec`, an optional output directory,
and an `Arc<dyn ProgressSink>`, and drives the whole pipeline to a `JobOutput`.
`run_job_blocking` ([`job.rs:186`](../crates/rivet/src/job.rs)) is the sync
wrapper that builds a multi-threaded Tokio runtime and `block_on`s it.

The flow inside `run_job`:

1. `spec.validate()` (fail fast on an impossible request — e.g. HDR on a build
   with no 10-bit encoder).
2. Demux just the header + audio track once
   ([`job.rs:102`](../crates/rivet/src/job.rs)); emit `JobEvent::Started` /
   `Probed` on the sink.
3. Resolve the effective frame rate (source rate, clamped by
   `spec.max_frame_rate`) and `frames_total` (from the container header when
   known).
4. `prepare_audio` once ([`job.rs:132`](../crates/rivet/src/job.rs)) — the audio
   is shared across every rung, not re-decoded per rung.
5. Branch on `spec.mode`: `OutputMode::SingleFile` → `run_single_file`;
   `OutputMode::Hls { segment_seconds }` → `run_hls`.
6. Emit `JobEvent::Finished` and assemble `JobOutput`.

**Key types.** `RungArtifact` ([`job.rs:48`](../crates/rivet/src/job.rs)) is
either `File(Vec<u8>)` (single-file MP4 bytes held in memory) or
`HlsRendition { dir, relative_dir }` (a directory of CMAF segments + a media
playlist). `RungOutput` ([`job.rs:60`](../crates/rivet/src/job.rs)) wraps one
rung's artifact with its label/dims/frames/bytes. `JobOutput`
([`job.rs:71`](../crates/rivet/src/job.rs)) collects the completed rungs plus
HLS-only fields (`hls_root`, `master_playlist`) and source/audio metadata. A
*failed* rung is **not** in `JobOutput.rungs` — it is reported through the sink
as `RungStatus::Failed`, and the job only hard-errors if *every* rung failed
([`job.rs:311`](../crates/rivet/src/job.rs)).

### SingleFile orchestration

`run_single_file` ([`job.rs:204`](../crates/rivet/src/job.rs)) has **two
strategies** and picks between them:

- **Multi-GPU chunk-and-stitch** when the encode policy is `AllGpus`/`Family`,
  the frame count is known, the pool has capacity > 1, *and* the seam mode isn't
  `Serial` ([`job.rs:225`](../crates/rivet/src/job.rs)). It calls
  `run_single_file_multigpu` → `multigpu::run_multigpu_single_file`, which
  chunks each rung at GOP boundaries, encodes the chunks across all GPUs, and
  returns ordered AV1 packet streams. `mux_rung_packets_to_mp4`
  ([`job.rs:382`](../crates/rivet/src/job.rs)) then stitches each rung's packets
  (plus shared audio) into one MP4 — in memory, no disk round-trip.
- **Serial decode-once fan-out** otherwise
  ([`job.rs:250`](../crates/rivet/src/job.rs)): one shared decode pump fans
  frames to one `encode_rung_single_file` worker per rung over a bounded mpsc
  channel (`FRAME_CHANNEL_CAPACITY = 8`,
  [`job.rs:44`](../crates/rivet/src/job.rs)), each scaling + encoding + muxing
  its own MP4 with a single encoder.

**Why two strategies.** The README's GPU-scheduling note explains it: chunking a
single file only pays off when there are multiple GPUs to spread the chunks
across; on a single-GPU host (or when the frame count is unknown so chunks can't
be planned) the lean serial path runs with no GOP-chunking overhead.
`ChunkSeamMode::Serial` forces one encoder even on a multi-GPU host because its
whole point is a seam-free, quality-target-accurate stream. Constant-quality
encoding (CQP/CRF) is what makes the chunked path safe to stitch: independent
IDR-led GOPs have no rate-control discontinuity at the ~2 s seams
([`job.rs:317`](../crates/rivet/src/job.rs) doc comment).

### HLS orchestration

`run_hls` ([`job.rs:485`](../crates/rivet/src/job.rs)) is thinner: it computes
the segment grid (timescale, per-frame ticks, `keyframe_interval`,
`segment_target_ticks`), hands everything to
`multigpu::run_multigpu_hls`, then **assembles the package**: it builds a
`VideoVariantSpec` per rung (measuring bandwidth and extracting the `av01.…`
codec string from each rung's `init.mp4` via
[`cmaf_util`](../crates/rivet/src/cmaf_util.rs)), muxes the shared audio into a
CMAF audio rendition (`build_audio_rendition`,
[`job.rs:720`](../crates/rivet/src/job.rs)), and calls
`container::hls::write_hls_package` to emit `master.m3u8` + per-variant
playlists. The orchestrator produces the segments; `job.rs` produces the
playlists around them.

### Audio prep

`prepare_audio` ([`job.rs:625`](../crates/rivet/src/job.rs)) runs once and
yields a `PreparedAudio { info, samples, handling }` shared by every rung. The
routing mirrors the README's audio table: passthrough for AAC/Opus/AC-3/E-AC-3;
decode→Opus for MP3/Vorbis; drop (with a warn) for the rest, including
multichannel ≥3 ch. `AudioPolicy::ForceOpus` forces a transcode of anything not
already Opus; `Drop` removes audio. A `with_audio` rejection at mux time degrades
to **video-only with a warn** rather than losing the customer's video
([`job.rs:392`](../crates/rivet/src/job.rs)).

> **Gotcha:** the job-engine audio path (`prepare_audio`) and the one-shot path's
> `wire_audio` ([`transcode.rs:228`](../crates/rivet/src/transcode.rs)) are
> independent implementations of the same routing policy. They must stay in sync.

---

## The single-shot path (`transcode.rs`)

**What.** [`transcode_bytes`](../crates/rivet/src/transcode.rs#L106) is the
primary library entry for "one buffer in, one AV1/MP4 buffer out." It is a
straight-line, single-threaded loop — demux → `create_decoder` → per-sample
`push_sample`/`decode_next` → `convert_to_yuv420p_bt709` → `send_frame` →
`receive_packet` → `add_packet` → drain → `finalize`
([`transcode.rs:164`](../crates/rivet/src/transcode.rs)). `transcode_file`
wraps it with file read/write. The result is a `TranscodeOutcome`
([`transcode.rs:40`](../crates/rivet/src/transcode.rs)) carrying input/output
metadata, frame/packet counts, an `AudioHandling`
([`transcode.rs:65`](../crates/rivet/src/transcode.rs)) tag, and elapsed time.

**Why it exists separately from the job engine.** It is the no-ceremony path: no
rungs, no GPU pool, no scaling, no progress sink, no async runtime. The module
doc is explicit that for segmented CMAF-HLS or an ABR ladder you should drive the
job engine (or the `container`/`codec` crates) instead. It targets the source
resolution, caps frame rate at 60, and uses a fixed 2-second keyframe interval
([`transcode.rs:129`](../crates/rivet/src/transcode.rs)). It exists so the
trivial case stays trivial and so there's a reference implementation of the
decode→encode→mux loop without the orchestration noise.

**Notes / decisions.**
- It uses `colorspace::convert_to_yuv420p_bt709`
  ([`transcode.rs:217`](../crates/rivet/src/transcode.rs)), the fixed SDR
  conversion — *not* the policy-driven `convert_to_sdr_bt709` the pump uses. The
  one-shot path has no `OutputSpec`, so there's no `ColorPolicy` to consult; it
  always produces SDR BT.709.
- Audio is wired inline by `wire_audio`
  ([`transcode.rs:228`](../crates/rivet/src/transcode.rs)) with the same
  passthrough/transcode/drop routing as the job engine.
- Both `transcode_bytes` and the job engine honor
  `TRANSCODE_ENCODER_BACKEND=nvenc|amf|qsv` as a backend override
  ([`transcode.rs:142`](../crates/rivet/src/transcode.rs),
  [`job.rs:756`](../crates/rivet/src/job.rs)).

---

## The multi-GPU reactive engine — the heart

This is the core orchestration: decode the source **once**, fan frames out to N
per-rung scalers, and dynamically schedule every rung's segments/chunks across
**all** available GPUs with a fair lease pool and mid-flight helper dispatch. The
README calls it "the rung benefit"; [pipeline.md](pipeline.md#4-the-multi-gpu-lease-engine--the-rung-benefit)
has the diagram. Below is the *why* of each component.

### `decode_pump.rs` — decode once, fan out

**What.** [`run_shared_decode_pump_blocking`](../crates/rivet/src/decode_pump.rs#L49)
demuxes + decodes the source one time and fans every normalized frame out to a
`Vec` of per-rung mpsc senders. It runs the rung-*agnostic* per-frame work in
`normalize_frame` ([`decode_pump.rs:121`](../crates/rivet/src/decode_pump.rs)):
4:4:4 → 4:2:0 downsample (when `needs_downsample`), then — *only when the spec's
color policy says so* (`tonemap_to_sdr`) — an HDR-aware colorspace convert
(`convert_to_sdr_bt709`, PQ/HLG → SDR BT.709), then the spec's
[video filters](filters.md) — a `codec::filter::FilterChain` prepared once in
`run_job` (loading any overlay images) and applied per frame. The pump never
decides to tonemap on its own; the caller sets the flag from the `OutputSpec`'s
`ColorPolicy`.

**Why.** This is the entire performance argument for the crate: a 5-rung ABR
ladder decodes the input **once, not five times** (the naïve ffmpeg-per-rung
approach decodes N times). Fanout is cheap because `VideoFrame::clone()` is a
refcount bump — the pixel `Bytes` is `Arc`-backed
([`decode_pump.rs:137`](../crates/rivet/src/decode_pump.rs)). Normalization is
done once *before* fanout precisely because it's identical for every rung; only
per-rung scaling differs, and that's pushed down to the scalers.

**Notes / gotchas.**
- The cost is **backpressure**: the slowest rung (usually the largest, whose
  encoder is slowest) throttles the pump (module doc,
  [`decode_pump.rs:11`](../crates/rivet/src/decode_pump.rs)).
- `fan_out` returns `false` *only when every sender is closed*
  ([`decode_pump.rs:139`](../crates/rivet/src/decode_pump.rs)); a single rung
  giving up doesn't stop the pump.
- The loop is blocking (built for `spawn_blocking`); it bridges into the async
  `send().await` via a passed-in `tokio::runtime::Handle`
  ([`decode_pump.rs:148`](../crates/rivet/src/decode_pump.rs)).

### `gpu_pool.rs` — one encoder per GPU

**What.** [`GpuPool`](../crates/rivet/src/gpu_pool.rs#L26) is a process-wide
reservation pool: each detected GPU is a slot, callers `claim()` an available
slot and hold the returned `GpuLease` for the lifetime of their work, and the
lease's `Drop` releases the slot ([`gpu_pool.rs:110`](../crates/rivet/src/gpu_pool.rs)).
With N GPUs and M waiters, the first N get leases immediately; the rest park on a
Tokio `Semaphore` until a lease drops.

**Why — the load-bearing invariant.** The module doc is emphatic and dated: this
is the deliberate design decision from 2026-05-02 because **concurrent NVENC
sessions on the same CUDA context deadlocked at ~session 5/5 init** — the GPU
went idle and no frames encoded ([`gpu_pool.rs:8`](../crates/rivet/src/gpu_pool.rs)).
One-encoder-per-GPU is the invariant; the pool's job is to enforce it *while
still running encoders in parallel across GPUs*.

**Key mechanics.**
- **Vendor on the lease is load-bearing.** Each slot records its `GpuVendor`
  ([`gpu_pool.rs:35`](../crates/rivet/src/gpu_pool.rs)). Without it, a
  multi-vendor host (NVIDIA + Intel Arc, both exposing index 0) *always* picked
  NVENC because the encoder factory tries NVIDIA first; the Arc sat idle. The
  lease tells the factory which backend to use (test
  `lease_carries_vendor_for_dispatch`,
  [`gpu_pool.rs:390`](../crates/rivet/src/gpu_pool.rs)).
- **The encode pool drops AV1-incapable cards.** `gpu_pool_for_policy` filters a
  multi-GPU selection through
  [`codec::encode::av1_encode_capable`](../crates/codec/src/encode/mod.rs) — the
  authoritative probe that runs the same `select_encoder` dispatch a worker uses,
  cached per index. A card that can't encode AV1 (e.g. a **pre-Ada NVIDIA** that
  decodes via NVDEC but has no AV1 encode silicon) is dropped from the *encode*
  pool, so no worker leases it and hard-fails the run; the capable cards (the
  Arc) encode. It stays in `policy_gpu_indices` (intentionally **not** filtered),
  so the decode pump can still use it — a pre-Ada NVIDIA + Arc decodes on the
  NVIDIA (NVDEC) and encodes on the Arc (QSV) with no flags. (Pinning `--gpu` to
  an incapable card now surfaces an empty-pool error up front instead of
  aborting mid-run.)
- **Sparse indices** are preserved (slot stores `GpuDevice.index`, not vec
  position) to handle `CUDA_VISIBLE_DEVICES=[0,2,5]`
  ([`gpu_pool.rs:28`](../crates/rivet/src/gpu_pool.rs), test `sparse_indices_preserved`).
- **`claim()` vs `try_claim()`.** `claim().await` is the blocking path for
  initial workers and increments `pending_claimers` via a `PendingClaimGuard`
  RAII bracket ([`gpu_pool.rs:70`](../crates/rivet/src/gpu_pool.rs)) — the guard
  decrements even if the await is cancelled. `try_claim()` is the non-blocking
  path the **helper dispatcher** uses; it does *not* touch `pending_claimers`,
  because Tokio's `Semaphore` is FIFO and a permit freed while a real worker is
  parked is reserved for that worker — so `try_claim` can't steal it (test
  `try_claim_does_not_steal_from_blocked_claimer`,
  [`gpu_pool.rs:577`](../crates/rivet/src/gpu_pool.rs)). `pending_claimers()`
  ([`gpu_pool.rs:146`](../crates/rivet/src/gpu_pool.rs)) is the fairness signal
  the dispatcher reads.
- **CPU-only host:** an empty inventory makes `claim()`/`try_claim()` return
  `None` immediately so call sites need no special-casing
  ([`gpu_pool.rs:188`](../crates/rivet/src/gpu_pool.rs)).
- The free-slot scan is a lock-free CAS loop guarded by the semaphore count, so a
  successful acquire always finds a free slot — a `None` there is treated as an
  invariant violation (`unreachable!` / `expect`,
  [`gpu_pool.rs:208`](../crates/rivet/src/gpu_pool.rs)).

### `frame_queue.rs` — the bounded chunk queue

**What.** [`SegmentChunkQueue`](../crates/rivet/src/frame_queue.rs#L27) connects
one producer (a rung's scaler) to N consumers (that rung's encoder workers). The
unit of transfer is a `SegmentChunk` ([`frame_queue.rs:21`](../crates/rivet/src/frame_queue.rs))
— one CMAF segment's worth of frames (`keyframe_interval` frames) tagged with a
monotonic `segment_idx` so each worker knows which output file/segment it's
producing.

**Why.** Single-producer / multi-consumer with **bounded** capacity for memory
safety: the pump/scaler blocks when the queue is full, workers block when it's
empty ([`frame_queue.rs:11`](../crates/rivet/src/frame_queue.rs)). The segment
index travels with the frames so the work is self-describing — a helper attaching
mid-flight just starts popping from the queue head, no decode-and-discard.

**Notes.**
- `push_front` ([`frame_queue.rs:126`](../crates/rivet/src/frame_queue.rs)) is
  the **requeue** path: a worker that pops a chunk and then detects a cross-vendor
  codec-invariant mismatch puts the chunk back at the head (briefly exceeding
  capacity by 1) and exits, so a compatible worker picks it up. It decrements
  `popped_segments` so the dispatcher's `pushed > popped` "work remaining"
  predicate stays accurate.
- `pushed_segments()` / `popped_segments()`
  ([`frame_queue.rs:59`](../crates/rivet/src/frame_queue.rs)) are exactly the
  counters the helper dispatcher polls to decide which rung still has pending
  work.

### `rung_scaler.rs` — per-rung scale → chunk

**What.** [`run_rung_scaler_blocking`](../crates/rivet/src/rung_scaler.rs#L35):
one scaler per rung consumes normalized frames from the pump's fanout channel,
bilinear-scales each to the rung's dimensions (CPU work, AVX2 where it pays), and
groups `frames_per_chunk` (= `keyframe_interval`) frames into a `SegmentChunk`
with a monotonic index, pushing into the rung's `SegmentChunkQueue`.

**Why.** Scaling is the one per-frame step that *is* per-rung, so it's pushed out
of the shared pump to here (each scaler runs on its own thread). On exit (the
input channel returns `None` because the pump closed all senders), the scaler
flushes the final partial chunk and **closes the queue** so encoder workers drain
and exit cleanly ([`rung_scaler.rs:43`](../crates/rivet/src/rung_scaler.rs)).

### `encoder_worker.rs` — per-segment encode + the codec invariant

**What.** Two worker bodies share a config and the invariant check:
- [`run_encoder_worker_blocking`](../crates/rivet/src/encoder_worker.rs#L248)
  (HLS path): pop a chunk → encode K frames → write one CMAF segment file →
  repeat. Each worker owns one GPU lease and **one encoder** for its lifetime but
  builds a **fresh `CmafVideoMuxer` per segment**, configured with the segment's
  index + base decode time so the on-disk filename and `tfdt` match what a
  single-encoder pipeline would produce
  ([`encoder_worker.rs:342`](../crates/rivet/src/encoder_worker.rs)).
- [`run_chunk_encoder_worker_blocking`](../crates/rivet/src/encoder_worker.rs#L539)
  (single-file path): identical shape, but *collects* the chunk's packets into a
  `ChunkPackets` ([`encoder_worker.rs:507`](../crates/rivet/src/encoder_worker.rs))
  instead of writing a segment, so the orchestrator can stitch them into one MP4.

**Why the codec invariant.** [`RungCodecInvariant`](../crates/rivet/src/encoder_worker.rs#L51)
captures the mandatory AV1 sequence-header fields (`seq_profile`, level/tier,
bit depth, chroma subsampling, the four color fields, max frame dims, …) that
every encoder contributing to one rendition **must** agree on. The reason is
spelled out in the type doc ([`encoder_worker.rs:31`](../crates/rivet/src/encoder_worker.rs)):
a helper may land on a different GPU *vendor* than the rung's first worker
(NVENC + QSV + AMF + rav1e can all touch one rendition), and the player sets up
its decoder once from `init.mp4`'s `av1C`; if a later segment's inline OBU
sequence header disagrees on a mandatory field, strict decoders (dav1d in
conformance mode, Safari AVFoundation, hls.js+libdav1d) reject the segment. The
first worker on a rung **sets** the invariant; subsequent workers **compare**
on their first packet (`validate_or_set_rung_invariant`,
[`encoder_worker.rs:146`](../crates/rivet/src/encoder_worker.rs)).

The check deliberately **ignores** cosmetic optional fields (timing info /
decoder-model presence, film-grain present flag, operating-point detail) so
cross-vendor encoders co-exist without byte-difference false rejections.

**The three outcomes** (`InvariantCheck`,
[`encoder_worker.rs:127`](../crates/rivet/src/encoder_worker.rs)):
- `SetByThisWorker` / `Matched` → proceed to publish.
- `Mismatched` → the worker **requeues its chunk** (`push_front`) and exits
  *cleanly* — only that one helper's contribution is lost; another (matching-vendor)
  worker picks the chunk up, and the run never aborts. This is the
  "mission-critical jobs do not abort" rule.
- An `Err` (parse failure: the encoder emitted no `OBU_SEQUENCE_HEADER` at all)
  is a hard configuration bug that *does* fail the run — distinct from a soft
  mismatch.

**Notes.** Packets are buffered until the first-packet decision is made
([`encoder_worker.rs:380`](../crates/rivet/src/encoder_worker.rs)): nothing is
committed to the muxer (and `init.mp4` is only written by `finalize`, which a
rejecting worker never calls) until validation passes, so a mismatched worker
discards everything in flight with no on-disk side effects.

### `multigpu.rs` — the orchestrator

**What.** Two near-mirror functions —
[`run_multigpu_hls`](../crates/rivet/src/multigpu.rs#L126) (returns one
`RungManifest` per rung) and
[`run_multigpu_single_file`](../crates/rivet/src/multigpu.rs#L769) (returns one
`RungPackets` per rung) — wire the pieces together. Both take a
`MultiGpuParams` ([`multigpu.rs:74`](../crates/rivet/src/multigpu.rs)) carrying
the input, rungs, source/output color + pixel format, the segment grid, the
`GpuPool`, and the policy's GPU indices.

The orchestration, step by step:

1. **Pre-flight encoder probe** ([`multigpu.rs:149`](../crates/rivet/src/multigpu.rs)):
   construct a throwaway encoder to verify this host can produce AV1 *before*
   spawning any workers. This fails fast with a clear "no AV1 encoder available"
   message and — importantly — avoids dispatching workers that would fail at
   encoder construction, which on some drivers (Ampere with no AV1-encode
   silicon) would hang an *uncancellable* blocking task.
2. **Per-rung shared state** ([`multigpu.rs:173`](../crates/rivet/src/multigpu.rs)):
   one `SegmentChunkQueue`, an encoded-frame `AtomicU64` counter, a `scaler_active`
   flag, a `RwLock<Option<RungCodecInvariant>>` slot, a contributions `Mutex`, an
   `active_workers` count, a `rung_done` `Notify`, and a `finalized` flag.
3. **Finalizers** (one task per rung): wait until that rung's scaler + all its
   workers are done (`active_workers == 0`, woken by `rung_done`), then merge the
   contributions into a `RungManifest` (HLS) or stitch the packets into a
   `RungPackets` (single-file), checking **coverage** — exactly
   `total_segments` contiguous segments, no gaps or dupes
   ([`multigpu.rs:245`](../crates/rivet/src/multigpu.rs),
   [`multigpu.rs:866`](../crates/rivet/src/multigpu.rs)).
4. **Decode pump(s)** ([`multigpu.rs:300`](../crates/rivet/src/multigpu.rs)):
   one *shared* pump when `n <= gpu_pool.capacity()`, else one pump *per rung*.
   Each pump is pinned to a policy GPU via `decode_gpu_for(i)`
   ([`multigpu.rs:113`](../crates/rivet/src/multigpu.rs)) — the explicit
   `decode_gpu` override wins, else the policy's indices round-robin.
5. **Per-rung scalers** ([`multigpu.rs:341`](../crates/rivet/src/multigpu.rs)).
6. **Initial encoder workers**, one per rung, claimed **smallest-first**
   ([`multigpu.rs:287`](../crates/rivet/src/multigpu.rs)) so the cheap rungs grab
   leases first and free them sooner for helper dispatch.
7. **Helper dispatcher** ([`multigpu.rs:411`](../crates/rivet/src/multigpu.rs)):
   a loop that, every `HELPER_POLL_INTERVAL` (200 ms), checks fairness
   (`pending_claimers() > 0` → back off so a parked real worker claims first),
   finds the first rung with a live scaler or pending segments, `try_claim()`s a
   freed lease, and attaches an extra worker to that rung. When no rung has work
   left, the loop exits.
8. **Drain loop** ([`multigpu.rs:482`](../crates/rivet/src/multigpu.rs)): a
   `biased` `tokio::select!` over the pump/scaler/worker `JoinSet`s and the
   finalizer channel; any error triggers the `teardown_err!` macro
   ([`multigpu.rs:472`](../crates/rivet/src/multigpu.rs)) which cancels the
   helper + progress tasks before returning.

**Why mid-flight helpers.** Segment/chunk work is the unit of parallelism. When a
fast rung finishes and releases its lease, the freed GPU shouldn't sit idle — the
helper dispatcher hands it an *extra* worker on a still-busy rung, so a slow rung
finishes sooner and throughput scales close to linearly with GPU count (README
"GPU scheduling"). The cross-vendor codec invariant (above) is what makes it safe
for that helper to land on a different vendor.

**Policy helpers.** `gpu_pool_for_policy` / `policy_gpu_indices` /
`serial_gpu_for_policy` / `select_gpus_for_policy`
([`multigpu.rs:699`](../crates/rivet/src/multigpu.rs)) translate an
`EncodePolicy` (`AllGpus` / `SingleGpu(idx)` / `Family(vendor)`) into a concrete
device set — so a `Family`/`SingleGpu` constraint governs both encode *and*
decode (the decode pump pins to the same selected set). An empty selection yields
a capacity-0 pool, and the pre-flight probe / lease claim then surfaces a clear
error. `detect_gpu_pool` ([`multigpu.rs:695`](../crates/rivet/src/multigpu.rs))
builds an unconstrained pool from the host inventory.

**Gotchas.**
- `use_shared_pump = n <= capacity` ([`multigpu.rs:300`](../crates/rivet/src/multigpu.rs)):
  with at most one rung per GPU, one shared pump suffices; with more rungs than
  GPUs the engine spins up per-rung pumps (each pinned round-robin to a policy
  GPU). *(Rationale inferred from the code — the choice of shared vs per-rung
  pump is not annotated with a comment; the effect is that decode parallelism
  matches the available GPU count.)*
- `QUEUE_CAPACITY = 2` and `FANOUT_CHANNEL_CAPACITY = 4`
  ([`multigpu.rs:56`](../crates/rivet/src/multigpu.rs)) are the backpressure
  tuning knobs between pump → scaler → worker. *(The specific values are not
  justified in a comment — inferred as small bounded buffers to keep peak RSS
  low while leaving a little slack.)*

### `cmaf_util.rs` — the CMAF/HLS glue

**What.** Shared helpers used by both the job engine and the orchestrator
([`cmaf_util.rs:1`](../crates/rivet/src/cmaf_util.rs)):
- `keyframe_interval_for_segment` / `total_segments_for_rung` (ceil-division
  segment count) — the segment-grid math.
- `add_packet_with_segment_flush` ([`cmaf_util.rs:32`](../crates/rivet/src/cmaf_util.rs)):
  flush the prior segment when the next packet is a keyframe *and* the buffered
  duration has reached the segment target — so **each segment opens on an IDR**,
  which is what keeps the ladder segment-aligned for clean ABR. The audio
  counterpart flushes on the same time grid.
- `merge_rung_contributions` ([`cmaf_util.rs:74`](../crates/rivet/src/cmaf_util.rs)):
  combine several workers' segment lists for one rung into one ordered manifest,
  erroring on disagreeing dims/timescale, **duplicate** segment numbers, or
  internal **gaps** — the coverage guarantee the finalizer relies on.
- `measure_bandwidth` (avg/peak bits/sec for the HLS variant `BANDWIDTH`) and
  `av1_codec_string_from_init` ([`cmaf_util.rs:167`](../crates/rivet/src/cmaf_util.rs)),
  which walks the `moov…av01.av1C` box tree of an `init.mp4` to recover the exact
  `av01.…` codec string for the playlist.

**Why.** These are the bits of CMAF bookkeeping that are identical whether the
single-encoder or multi-worker path produced the segments; centralizing them
keeps the segment-on-IDR rule and the merge/coverage logic in one place rather
than duplicated across `job.rs`, `encoder_worker.rs`, and `multigpu.rs`.

---

## ABR ladder (`ladder.rs`)

**What.** [`standard_ladder(src_w, src_h, max_short_side)`](../crates/rivet/src/ladder.rs#L26)
derives a sensible `Vec<Rung>` from a source resolution. It snaps to the standard
short-side quantizations (2160/1440/1080/720/480/360/240), preserves the source
aspect ratio, even-aligns every dimension (AV1 4:2:0 needs even dims), and caps
the top rung at `max_short_side` (default 1080). `standard_ladder_with_quality`
stamps a `Quality` on every rung.

**Why.** It's the convenience path so callers don't hand-build a ladder for the
common case; the README and CLI `--ladder` flow through here. Callers who want
full control build `Rung`s by hand and skip the module entirely
([`ladder.rs:4`](../crates/rivet/src/ladder.rs)). The "p" number always refers to
the **short** side regardless of orientation, so portrait sources get correctly
labelled rungs (`1080p` for a 1080×1920 source — test
`ladder_portrait_short_side_labels`). The 1080 default cap is the web-safe ceiling;
lifting it to 1440/2160 unlocks QHD/4K rungs. `MIN_DIMENSION = 200`
([`ladder.rs:20`](../crates/rivet/src/ladder.rs)) drops rungs too small to be
worth a separate rendition.

---

## Progress reporting (`progress.rs`)

**What.** Every job streams progress through a
[`ProgressSink`](../crates/rivet/src/progress.rs#L103) — a tiny trait with
`on_rung(RungProgress)` (called repeatedly as each rung advances) and an
optional `on_event(JobEvent)` for coarse lifecycle. [`RungProgress`](../crates/rivet/src/progress.rs#L35)
is the **uniform** per-rung struct (index, label, dims, `RungStatus`, percent,
frames done/total, segments, bytes, optional message) that a consumer can render
into a progress bar *without knowing the output mode*. `RungStatus`
([`progress.rs:18`](../crates/rivet/src/progress.rs)) is the lifecycle:
`Pending → Running → Finalizing → Completed`/`Failed`.

**Why a sink, not a return value.** Progress is emitted *as the job runs*, so it
needs a push channel. The trait is deliberately small and synchronous; to bridge
into async you wrap a Tokio mpsc with [`channel_sink`](../crates/rivet/src/progress.rs#L152)
(turning the callback into a `.recv().await` stream) or a closure with
[`fn_sink`](../crates/rivet/src/progress.rs#L127). `NullSink` drops everything.

**Notes.** `ChannelSink` uses `try_send` and **drops** updates when the channel
is full or closed ([`progress.rs:146`](../crates/rivet/src/progress.rs)) —
progress is advisory, never load-bearing, so it must never block or fail the job.
The same events back the CLI's progress lines, the HTTP API's job-status polling
(via `RegistrySink`), and the README's library examples.

---

## Inspection helpers

### `probe.rs` — inspect without transcoding

[`probe_bytes`](../crates/rivet/src/probe.rs#L55) demuxes only the container
header + audio-track metadata (no decode) and reports a
[`MediaInfo`](../crates/rivet/src/probe.rs#L16): container label, video codec,
dims, frame rate, duration, pixel format, and audio stream shape. It's the data
source for `rivet probe`, for the CLI/HTTP code resolving `--ladder` rungs, and
for the HTTP `validate`/source-resolution paths. `detect_container`
([`probe.rs:80`](../crates/rivet/src/probe.rs)) mirrors the magic-byte dispatch
in `container::streaming::demux_streaming` so the reported label matches the
demuxer actually used.

### `validate.rs` — advisory input gates

[`validate_stream`](../crates/rivet/src/validate.rs#L45) checks a demuxed stream
against the reference resolution / frame-rate / duration / pixel-format policy
(`MIN_RESOLUTION`, `MIN_FRAME_RATE`, `MAX_DURATION_SECS`). **The job engine does
not call it** — the module doc is explicit that these are *advisory* so rivet
transcodes whatever it's given; they exist for policy-bearing callers (a hosted
service) to gate uploads with the same limits the reference transcoder uses
([`validate.rs:1`](../crates/rivet/src/validate.rs)). The one function the engine
*does* use is `needs_chroma_downsample`
([`validate.rs:93`](../crates/rivet/src/validate.rs)), which tells the pump
whether to run the 4:4:4 → 4:2:0 step.

### `thumbnail.rs` — opt-in AVIF still

[`generate_thumbnail`](../crates/rivet/src/thumbnail.rs#L52) (behind the
`thumbnail` feature) decodes the source up to a target frame (default 10% in, so
it's past intros/fades), converts BT.709-limited YUV → RGB, and encodes a still
**AVIF** via `ravif` (rav1e + a HEIF box writer). Two rationale notes from the
module doc: a *separate* decode pass (rather than tapping the variant decoders)
gives an isolated failure mode — a thumbnail miss never blocks the variant
pipeline — and is cheap because it only decodes up to the capture frame. AVIF is
chosen so the still reuses the same AV1 client-codec story as the video (every
browser that plays the video plays the thumbnail) without adding a JPEG/WebP
encoder to the dep graph ([`thumbnail.rs:16`](../crates/rivet/src/thumbnail.rs)).

---

## The front-ends, and the shared `TranscodeSettings`

All three front-ends parse their own syntax into **one** canonical knob set,
[`TranscodeSettings`](../crates/rivet/src/settings.rs#L26), then call
[`TranscodeSettings::into_spec`](../crates/rivet/src/settings.rs#L58) — the
**single** `OutputSpec`-building implementation. The module doc states the design
goal directly: add a new option *once* here (a field + a line in `into_spec` +
a `parse_*` arm) and every surface picks it up, instead of maintaining three
copies of the spec-building logic ([`settings.rs:1`](../crates/rivet/src/settings.rs)).
`into_spec` also encodes the GPU-policy precedence — pinned index > vendor family
> single-gpu > all-gpus ([`settings.rs:109`](../crates/rivet/src/settings.rs)) —
and calls `spec.validate()` so an impossible request is rejected at the surface.

### CLI (`main.rs`)

[`main.rs`](../crates/rivet/src/main.rs) is a `clap` app with these subcommands:

| Subcommand | What it does |
|------------|--------------|
| `transcode` | The main path: fills `TranscodeSettings` from flags → `into_spec` → `run_job_blocking`, with a per-rung progress line and on-disk output placement ([`main.rs:434`](../crates/rivet/src/main.rs)). |
| `probe` | `probe_file` → human table or `--json`. |
| `devices` | List detected GPUs (vendor, name, VRAM, PCI, live NVML load on NVIDIA); `--json` available ([`main.rs:691`](../crates/rivet/src/main.rs)). |
| `capabilities` (alias `caps`) | What this *build + host* can encode/decode — enabled backends, max bit depth, HDR, per-codec decode backends, devices ([`main.rs:770`](../crates/rivet/src/main.rs)). |
| `pipe` | Stream stdin → stdout, no temp files; flags override quality/size/color/audio ([`main.rs:909`](../crates/rivet/src/main.rs)). |
| `ipc` | A Unix-domain-socket server (`ipc` feature, Unix only): per connection the client writes media, half-closes, and reads the AV1/MP4 back; an optional `#rivet k=v …\n` header line carries settings ([`main.rs:948`](../crates/rivet/src/main.rs)). |
| `serve` | The HTTP API (`server` feature) — delegates to `rivet::server::serve`. |

`pipe` and `ipc` share `stream_transcode` ([`main.rs:875`](../crates/rivet/src/main.rs)):
all-default settings take the fast `transcode_bytes` path; any set field routes
through `into_spec` + the single-file `run_job`. Both reject HLS output (a single
stream can't carry a segmented package). The `#rivet` header is split off by
`split_ipc_settings` and parsed via `TranscodeSettings::parse_kv_line`
([`settings.rs:158`](../crates/rivet/src/settings.rs)). `devices` /
`capabilities` reach straight into `codec::gpu` / `codec::encode` /
`codec::decode` for host/build introspection. **For every flag's meaning and
defaults, see [cli.md](cli.md).**

### HTTP server (`server.rs`)

[`server.rs`](../crates/rivet/src/server.rs) (the `server` feature) is a small
axum app so another application can *signal* a transcode over the network. The
internals worth knowing:

- **In-memory job registry.** `AppState` holds an `RwLock<HashMap<Uuid,
  Arc<JobHandle>>>` ([`server.rs:52`](../crates/rivet/src/server.rs)). Each
  `JobHandle` tracks phase, per-rung progress, artifacts, error, and HLS output
  dir. The module doc is candid that completed single-file artifacts are held in
  RAM until process exit — fine for a sidecar/worker, not a public CDN; a
  production deployment would offload from a `ProgressSink` watching
  `RungStatus::Completed` ([`server.rs:19`](../crates/rivet/src/server.rs)).
- **`RegistrySink`** ([`server.rs:197`](../crates/rivet/src/server.rs)) is the
  `ProgressSink` impl that mirrors per-rung updates into the `JobHandle` so
  `GET /v1/jobs/{id}` can report them — the same progress plumbing the CLI uses,
  pointed at a registry slot instead of stdout.
- **Two submission shapes.** `POST /v1/transcode` branches on `Content-Type`
  ([`server.rs:534`](../crates/rivet/src/server.rs)): `application/json` →
  a structured `TranscodeRequest` (input from a server **file path** or inline
  **base64**, an optional server `output.path`, and a structured `spec`); any
  other content type → a streamed binary body with the spec in **query params**.
  Both forms collapse onto `TranscodeParams::into_settings` →
  `TranscodeSettings::into_spec`, reusing the shared `settings::parse_*`
  vocabulary so the API carries no copy of the spec logic
  ([`server.rs:428`](../crates/rivet/src/server.rs)).
- **File-path I/O + sandbox.** `resolve_path` ([`server.rs:360`](../crates/rivet/src/server.rs))
  canonicalizes request-supplied paths; when `RIVET_FILE_ROOT` is set, a path
  must resolve *under* that root or it's rejected ("path escapes sandbox"). With
  no root set, the server (localhost-bound by default) treats paths as
  trusted-local. The HLS `/files/{*path}` route has its own `..`/empty-component
  traversal guard ([`server.rs:761`](../crates/rivet/src/server.rs)).
- **Sync vs async.** Default is fire-and-forget: `202 { job_id }` and the job
  runs in a spawned task; poll `GET /v1/jobs/{id}`. `?sync=true` (or `"sync":
  true`) runs the job inline and returns the MP4 directly for a single-file
  single-rung job, or a status JSON otherwise ([`server.rs:575`](../crates/rivet/src/server.rs)).
- Ships hand-authored OpenAPI 3.0 + Swagger UI + Redoc at `/openapi.json` /
  `/swagger` / `/redoc`.

**For the endpoint/wire reference, see [api.md](api.md).**

---

## Key decisions in the engine — recap

- **Decode once, fan out.** The pump decodes the source a single time and clones
  `Arc`-backed frames to every rung — the ladder is cheap. Normalization
  (4:4:4→4:2:0, policy-driven HDR→SDR) happens once, pre-fanout, because it's
  rung-agnostic. ([`decode_pump.rs`](../crates/rivet/src/decode_pump.rs))
- **One encoder per GPU.** `GpuPool` enforces it because concurrent NVENC sessions
  on one CUDA context deadlocked at init (2026-05-02). Work still runs in parallel
  *across* GPUs, and the lease carries the GPU **vendor** so multi-vendor hosts
  dispatch correctly. ([`gpu_pool.rs`](../crates/rivet/src/gpu_pool.rs))
- **Mid-flight helper dispatch.** Segment/chunk work is the unit of parallelism;
  a freed lease is reassigned to a still-busy rung so throughput scales with GPU
  count. Fairness via `pending_claimers` keeps `try_claim` from stealing a parked
  worker's permit. ([`multigpu.rs`](../crates/rivet/src/multigpu.rs))
- **Cross-vendor codec invariant.** A per-rung AV1 sequence-header contract lets
  NVENC + QSV + AMF + rav1e contribute to one rendition safely; a mismatched
  helper requeues its chunk and exits without aborting the job.
  ([`encoder_worker.rs`](../crates/rivet/src/encoder_worker.rs))
- **Fail fast, don't degrade.** A pre-flight encoder probe rejects a host with no
  AV1-encode silicon up front (and dodges an uncancellable hang on some drivers);
  `spec.validate()` rejects impossible color/depth combos at the surface.
- **Single-file uses the same engine.** When it helps (multi-GPU, known frame
  count, non-`Serial` seams), single-file chunk-encodes across GPUs and stitches
  in memory; otherwise it takes a lean serial decode-once path. Constant-quality
  encoding makes the stitched seams safe. ([`job.rs`](../crates/rivet/src/job.rs))
- **One knob set, three surfaces.** CLI, HTTP, and IPC all parse into
  `TranscodeSettings` → `into_spec`, so a new option is added once.
  ([`settings.rs`](../crates/rivet/src/settings.rs))
- **Two color paths, by design.** The one-shot `transcode_bytes` always produces
  SDR BT.709 (no spec to consult); the job engine's pump is policy-driven
  (`tonemap_to_sdr` from `ColorPolicy`). See
  [output-spec.md](output-spec.md#4-color--bit-depth) for the policy surface.
