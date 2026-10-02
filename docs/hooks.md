# Hooks

Hooks are code you supply that rivet runs at specific points of every job.
Each kind of hook has its own trait, written for one point in the pipeline,
and it gets only what exists at that point. A hook that hashes the source
bytes is a different kind from one that sees decoded frames, and both are
different from one that sees the frames the encoder receives.

For each event, a hook returns a verdict (carry on, or reject the job) and
any values it wants recorded. Everything the hooks return is collected into a
per-job report.

The module is [`rivet::hooks`](../crates/rivet/src/hooks/mod.rs).
[`examples/hooks.rs`](../crates/rivet/examples/hooks.rs) is a complete,
runnable example, and the **[hook cookbook](hooks-cookbook.md)** has sixteen
worked recipes, one per common job. To run a machine-learning model on a
job's pictures, **[inference.md](inference.md)** is the guide (where to hook,
which frames, getting pixels in, runtimes, GPUs, deployment), and
**[hooks-yolo.md](hooks-yolo.md)** a complete worked example: a YOLO detector
on CPU, CUDA, DirectML and OpenVINO.

## The kinds

| Kind | Trait | Register with | Where it hooks in | Gets |
|------|-------|---------------|-------------------|------|
| source | `SourceHook` | `Hooks::source` | The source bytes as they arrived, before anything parses them. Once per source; each clip of a splice is its own. | `SourceEvent`: `bytes`, `sniffed`, `clip` |
| probe | `ProbeHook` | `Hooks::probe` | The source once demuxed (or an image's header read), before anything is decoded. | `ProbeEvent`: a `MediaSummary` (container, codecs, upright size, rate, duration, frame count, `still`) |
| decoded frame | `DecodedFrameHook` | `Hooks::decoded_frames` | Video frames as the decoder produced them, turned upright. This is before tonemapping, chroma or bit-depth conversion, the spec's filters, or any scaling, so it's what the source holds whatever output is asked for. | `FrameEvent`: the `VideoFrame` in the decoder's pixel format, `index`, `seconds`, `clip` |
| encoder frame | `EncoderFrameHook` | `Hooks::encoder_frames` | Video frames exactly as the encoders receive them. That's after the colour pipeline and the filters, and before each rung's scaling. | `FrameEvent`: always `yuv420p` / `yuv420p10le` |
| still | `StillHook` | `Hooks::stills` | Still pictures in an image job: the decoded image, or each still taken from a video. Every one (no sampling). | `StillEvent`: upright 8-bit RGBA, `index`, `seconds`, `from_video` |
| artifact | `ArtifactHook` | `Hooks::artifacts` | Each output of the `ArtifactKind`s it accepts (`video`, `audio`, `image`, `rendition`, `playlist`), before the job returns it. | `ArtifactEvent`: `kind`, `label`, `media_type`, size, `data` (`Bytes`, an HLS `Directory`, or a `File`) |
| completed | `CompletedHook` | `Hooks::completed` | The job produced everything it was asked for. This is the last gate: a rejection here still fails the job. | `CompletedEvent`: artifact count, elapsed time |
| failed | `FailedHook` | `Hooks::failed` | The job failed, including when a hook rejected it. Runs once. Its verdict is ignored. | `FailedEvent`: the error chain, and the `HookRejection` if a hook rejected the job |

Each registration method has a `_with` form that takes a `HookPolicy`, for
example `decoded_frames_with(name, hook, HookPolicy::background())`.

Names are unique within a `Hooks` set: registering a name that's already
taken **replaces** the earlier hook. To register one value at two points (a
model on both decoded frames and stills, say), give each registration its own
name. At each point, hooks run in the order they were registered.

Where each kind runs in each kind of job:

| Job | source | probe | decoded frame | encoder frame | still | artifact | completed / failed |
|-----|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| `run_job` (video out) | ✓ | ✓ | ✓ | ✓ | | ✓ video / rendition / playlist | ✓ |
| `run_job` (audio-only out) | ✓ | ✓ | | | | ✓ audio | ✓ |
| `run_splice_job` | ✓ per clip | ✓ per clip | ✓ | ✓ | | ✓ | ✓ |
| `image::run_image_job_with_hooks` | ✓ | ✓ | | | ✓ | ✓ image | ✓ |

### Source-material hashing

Hashing the source material means hooking two specific points: the source
bytes, and the decoded source frames.

```rust
use rivet::hooks::*;

let hooks = Hooks::new()
    .source("source-digest", SourceDigest::new(&[DigestAlgorithm::Sha256, DigestAlgorithm::Md5]))
    .decoded_frames(
        "source-fingerprint",
        PerceptualFingerprint::new(&[PerceptualAlgorithm::PHash]).sampling(FrameSampling::every_seconds(1.0)),
    )
    .stills("still-fingerprint", PerceptualFingerprint::new(&[PerceptualAlgorithm::PHash]));
let spec = spec.with_hooks(hooks);
```

`SourceDigest` records `sha256` / `md5` / `sha1` of the bytes.
`PerceptualFingerprint` records `phash` / `dhash` / `ahash` (16 hex digits)
of each frame it gets. It implements `DecodedFrameHook`, `EncoderFrameHook`
and `StillHook`, so the method you register it with decides which pictures it
fingerprints. An integration that sends these values elsewhere implements the
same traits and computes the same values with the public building blocks
(below).

## Frame sampling

The two frame kinds pick their frames with `sampling() -> FrameSampling`:

| Field | Meaning |
|-------|---------|
| `every_seconds` | The first frame of each interval of this many seconds. The default is 1.0. |
| `every_frames` | Every Nth frame. Combined with `every_seconds` as OR. |
| `max_frames` | Stop after this many frames have been handed over. Counted per hook, per job. |

The first frame is always selected. `FrameSampling::all()` selects every
frame. Indices are presentation order in the source, before any trim or
frame-rate cap. When a source is decoded in ranges on several GPUs, frames
reach hooks from several threads at once and out of order, so hooks must be
`Send + Sync`.

`rivet::hooks::frame` turns any pixel format into something easy to work
with. `luma8` gives 8-bit luma, `rgb8` gives 8-bit RGB, and
`encode(frame, FrameFormat::Ppm | Pgm | Raw)` gives an image file. For a
model's input there are `rgb8_resized` (stretched to a size),
`rgb8_letterboxed` (fitted with the aspect kept, plus the `Letterbox` that
maps the model's coordinates back), `rgb8_to_planar_f32` (the NCHW tensor
layout), and `planar_f32_letterboxed`, which goes straight from the frame to
that tensor. The resizing ones read only the source pixels their output needs,
so their cost follows the output's size, not the frame's.
[hooks-yolo.md](hooks-yolo.md) uses them.

## Verdicts, policies, and failure

Every kind's method returns `Result<HookOutcome>`:

```rust
HookOutcome::proceed().annotate("key", value)   // carry on, record key = value (any JSON)
HookOutcome::reject("reason")                   // stop the job
```

`HookPolicy` (default: blocking, fail open, required):

| Field | Values | Meaning |
|-------|--------|---------|
| `mode` | `Blocking` / `Background` (`HookPolicy::background()`) | `Blocking` runs on the thread that reached the point, so a rejection takes effect at once. `Background` runs on the session's worker thread. A background rejection stops the job at its next point, and at the latest before the job returns, because the job waits for background hooks first. The worker's queue is bounded (64), so a hook that can't keep up slows the pipeline instead of piling frames up in memory. |
| `on_error` | `Continue` / `Reject` (`.fail_closed()`) | What a hook returning `Err` means: record it and carry on (fail open), or treat it as a rejection (fail closed). |
| `required` | `true` / `false` (`.optional()`) | Runs on every job, or only on jobs that name it (`Hooks::select`, the API's `hooks=`). |

Once a hook rejects, every point on every thread stops at its next check. The
job returns an error whose chain contains a `HookRejection` (the hook, its
kind, stage, subject and reason). Find it with `rejection_of(&err)`. The first
rejection wins: later ones are recorded but don't replace it. A failed hook
can't reject, since the job has already failed.

There's no timeout: a hook that never returns holds its point (or, in the
background, the worker) until it does. A hook that returns `Err` is handled by
its policy, but a **panic** isn't. The workspace's release profile builds with
`panic = "abort"`, so a panicking hook ends the process. Return an error
instead.

## Sessions and reports

`Hooks` is the set of hooks, and a session is one job's run of them. To read
the report of a job that failed, start the session yourself and keep a clone:

```rust
let session = hooks.session(my_job_id, rivet::hooks::JobKind::Transcode);
let result = rivet::run_job(input, &spec.with_hooks(session.clone()), None, sink).await;
let report = session.report();          // whatever `result` is
```

On success, the report is also in `JobOutput::hooks` / `ImageJobOutput::hooks`.
A `HookReport` holds one `HookRecord` per hook per event: the hook, its
`kind`, stage, subject, verdict, annotations, error, elapsed time, and
whether it ran in the background. Plus the rejection, if there was one. The
helpers are `by_kind`, `by_hook`, `annotations(key)`, `errors()`, and
`to_json()`.

## Building blocks

| Item | What it does |
|------|--------------|
| `SourceDigest` (source) | SHA-256 / SHA-1 / MD5 of the source bytes. |
| `PerceptualFingerprint` (decoded frame, encoder frame, still) | aHash / dHash / pHash of each picture it gets. |
| `ArtifactDigest` (artifact) | Digests of each output of the kinds it accepts. A directory records an object mapping each file to its digest. |
| `DigestAlgorithm` | `.digest(bytes)`, `.hex(bytes)`. |
| `phash` | `PerceptualAlgorithm::{AHash, DHash, PHash}` with `.hash_frame(&frame)` / `.hash_luma(..)`, plus `hamming`, `to_hex`, `from_hex`, `shrink`. Bit layout matches the common `imagehash` implementations. |
| `frame` | `luma8`, `rgb8`, `encode` (PPM / PGM / raw); `rgb8_resized`, `rgb8_letterboxed` + `Letterbox`, `rgb8_to_planar_f32`, `planar_f32_letterboxed` for model input ([inference.md](inference.md#4-getting-pixels-into-a-model)). |

The built-ins compute and record. Comparing, storing or forwarding what they
compute is up to the integration.

An `Arc` of any hook kind is that kind. One value can be registered at
several points to share state (see cookbook recipe 7), or kept by the caller
to read back after the job.

### The general trait

`Hook` (one `call` for any set of `Stage`s, registered with `Hooks::with`) is
what the kinds adapt to. Use it only for a hook that genuinely spans the job,
such as `LogHook`, which logs every event it gets. It gets a `HookEvent`,
whose `stage()`, `subject()` and `metadata_json()` describe any event.

`fn_hook` wraps a closure as a general hook, which suits tests and quick
experiments:

```rust
let hooks = Hooks::new().with(
    "count-frames",
    fn_hook(StageSet::of(&[Stage::DecodedFrame]), |_ctx, event| {
        Ok(HookOutcome::proceed().annotate("subject", event.subject().to_json()))
    })
    .sampling(FrameSampling::every_seconds(5.0)),
);
```

`StageSet::parse("source,decoded-frame")` reads a comma-separated list of stage
names, which is handy for configuration.

### Working with a set

| Method | What it does |
|--------|--------------|
| `names()`, `len()`, `is_empty()` | What's registered, in order. |
| `describe()` | Each hook's name, kind, description, stages, mode, `on_error`, `required`, sampling and artifact kinds, as JSON: what `GET /v1/hooks` serves. |
| `select(&names)` | The set a job runs: every required hook, plus the optional ones named. Naming one that isn't registered is an error, so a typo isn't a silently skipped hook. |
| `merged(&other)` | Every hook of `other` added, a name in both taking `other`'s. |
| `wants(stage)` | Whether any hook runs at `stage`. |
| `session(job_id, kind)` | The set with a fresh session (see above). `job_id()`, `report()` and `rejection()` read it. |
| `frames_exhausted()` | Whether every frame hook has had its `max_frames`. |

## HTTP API

```rust
let hooks = Hooks::new()
    .source("source-digest", SourceDigest::new(&[DigestAlgorithm::Sha256]))
    .decoded_frames_with("review", MyFrameIntegration::new(), HookPolicy::background().optional());
rivet::server::serve_with_hooks(addr, hooks).await?;     // or build_router_with_hooks(hooks)
```

- `GET /v1/hooks` lists them: name, `kind`, description, stages, mode,
  `on_error`, `required`, `frames` (sampling, for the frame kinds), and
  `artifact_kinds`.
- Required hooks run on every job. A request opts into optional ones with
  `?hooks=a,b` or `"hooks": [...]`. Naming one that isn't configured is a
  `400`. Requests can only choose among configured hooks.
- `GET /v1/jobs/{id}` includes `hooks`, the job's report, with each record's
  `kind`. It is readable while the job runs and after it ends.
- A rejected job ends with `status: "rejected"`. A `?sync=true` request gets
  `422`.

## Cost

An empty `Hooks` does nothing. Each point first checks whether any hook of
its kind is registered, and a frame no hook selects is never cloned. A
selected frame is a reference-count bump, not a pixel copy. Artifact events
copy a single-file output's bytes, and only when an artifact hook is
registered.
