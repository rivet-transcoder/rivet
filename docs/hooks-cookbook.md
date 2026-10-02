# Hook cookbook

Worked examples of the [hook engine](hooks.md). Each recipe is one small hook
of one specific kind. All the code is in
[`examples/hook_cookbook.rs`](../crates/rivet/examples/hook_cookbook.rs),
which builds with the crate's examples, so the recipes can't drift from the
API. Run it on a video or an image to see every recipe's report:

```sh
cargo run --release --example hook_cookbook --features rav1e-fallback -- input.mp4
cargo run --release --example hook_cookbook --features image -- photo.png
```

(`rav1e-fallback` gives a host without an AV1-encoding GPU a software
encoder, so the job gets as far as frames and artifacts.)

For a full vision-model integration, a YOLO object detector on the decoded
frames and stills, see [YOLO object detection with hooks](hooks-yolo.md).

| # | Recipe | Kind |
|---|--------|------|
| 1 | [Refuse sources over a size](#1-refuse-sources-over-a-size) | source |
| 2 | [Accept only some containers](#2-accept-only-some-containers) | source |
| 3 | [Source-material hashing](#3-source-material-hashing) | source + decoded frame + still |
| 4 | [Limits on what the source is](#4-limits-on-what-the-source-is) | probe |
| 5 | [Flag blank frames](#5-flag-blank-frames) | decoded frame |
| 6 | [Your own hashing library](#6-your-own-hashing-library) | decoded frame + still |
| 7 | [Did the filters change the picture?](#7-did-the-filters-change-the-picture) | decoded frame + encoder frame, shared state |
| 8 | [Forward to your own system without blocking](#8-forward-to-your-own-system-without-blocking) | decoded frame, background |
| 9 | [Cap output size](#9-cap-output-size) | artifact |
| 10 | [Write a sidecar manifest](#10-write-a-sidecar-manifest) | artifact |
| 11 | [Metrics on how jobs end](#11-metrics-on-how-jobs-end) | completed + failed |
| 12 | [Read the report of a rejected job](#12-read-the-report-of-a-rejected-job) | sessions |
| 13 | [Fail open, fail closed, background](#13-fail-open-fail-closed-background) | policies |
| 14 | [Hooks on the HTTP API](#14-hooks-on-the-http-api) | server |
| 15 | [Unit-test a hook without running a job](#15-unit-test-a-hook-without-running-a-job) | testing |
| 16 | [See every event while developing](#16-see-every-event-while-developing) | general |

All the snippets assume:

```rust
use rivet::hooks::*;
use anyhow::Result;
```

---

## 1. Refuse sources over a size

A source hook runs before anything parses the input. A rejection here means
the bytes are never demuxed.

```rust
pub struct MaxSourceSize(pub usize);

impl SourceHook for MaxSourceSize {
    fn on_source(&self, _ctx: &HookContext, source: &SourceEvent) -> Result<HookOutcome> {
        Ok(if source.bytes.len() > self.0 {
            HookOutcome::reject(format!("the source is {} bytes; the limit is {}", source.bytes.len(), self.0))
        } else {
            HookOutcome::proceed()
        })
    }
    fn describe(&self) -> String {
        format!("max source size {} bytes", self.0)
    }
}

let hooks = Hooks::new().source("max-source-size", MaxSourceSize(4 << 30));
```

`describe` is what `GET /v1/hooks` shows. It is optional, and each kind has a
default.

## 2. Accept only some containers

`SourceEvent::sniffed` is what the first bytes look like (`mp4`, `matroska`,
`mpegts`, `jpeg`, `png`, ... or `unknown`). It's a sniff, not a parse, so it's
cheap enough to gate on.

```rust
pub struct AllowedContainers(pub &'static [&'static str]);

impl SourceHook for AllowedContainers {
    fn on_source(&self, _ctx: &HookContext, source: &SourceEvent) -> Result<HookOutcome> {
        Ok(if self.0.contains(&source.sniffed.as_str()) {
            HookOutcome::proceed().annotate("container", source.sniffed.clone())
        } else {
            HookOutcome::reject(format!("`{}` sources are not accepted", source.sniffed))
        })
    }
}
```

## 3. Source-material hashing

Two specific points: the source bytes, and the source's decoded pictures. For
video that's the decoded frames; for an image job it's the stills. The
built-ins cover all three:

```rust
let hooks = Hooks::new()
    .source("source-digest", SourceDigest::new(&[DigestAlgorithm::Sha256, DigestAlgorithm::Md5]))
    .decoded_frames(
        "source-fingerprint",
        PerceptualFingerprint::new(&[PerceptualAlgorithm::PHash]).sampling(FrameSampling::every_seconds(1.0)),
    )
    .stills("still-fingerprint", PerceptualFingerprint::new(&[PerceptualAlgorithm::PHash]));
```

What a 3-second clip records (from the cookbook run):

```text
source-digest       source         { bytes: 488807, md5: "e095cf…", sha256: "642859…" }
source-fingerprint  decoded-frame  { phash: "8449a2b646ceab9f" }   frame 0
source-fingerprint  decoded-frame  { phash: "860f07b35e0f13ec" }   frame 24
source-fingerprint  decoded-frame  { phash: "a60cfd13cc21ff48" }   frame 48
```

Decoded frames are hashed before tonemapping, filters and scaling, so the
fingerprint describes the source whatever output was asked for. A hook
registered with `.decoded_frames` never fires in an image job, and one
registered with `.stills` never fires in a video job. Register both if
ingestion takes both.

## 4. Limits on what the source is

A probe hook gets the demuxed description before a single frame is decoded.
It's the cheap place to refuse sources that are too large, too long, or
missing video.

```rust
pub struct SourceLimits {
    pub max_width: u32,
    pub max_height: u32,
    pub max_seconds: f64,
    pub require_video: bool,
}

impl ProbeHook for SourceLimits {
    fn on_probe(&self, _ctx: &HookContext, probe: &ProbeEvent) -> Result<HookOutcome> {
        let m = &probe.media;
        if self.require_video && m.video_codec.is_none() {
            return Ok(HookOutcome::reject("the source has no video"));
        }
        if m.width > self.max_width || m.height > self.max_height {
            return Ok(HookOutcome::reject(format!("{}x{} is over {}x{}", m.width, m.height, self.max_width, self.max_height)));
        }
        if m.duration > self.max_seconds {
            return Ok(HookOutcome::reject(format!("{:.0}s is over {:.0}s", m.duration, self.max_seconds)));
        }
        Ok(HookOutcome::proceed()
            .annotate("codec", m.video_codec.clone().unwrap_or_default())
            .annotate("duration", m.duration))
    }
}
```

`MediaSummary` is `width`/`height` (upright, after the container's rotation),
`frame_rate`, `duration`, `frames`, `container`, `video_codec`,
`audio_codec`, `still`.

## 5. Flag blank frames

A decoded-frame hook with its own sampling (every half second). It uses
`frame::luma8` to get 8-bit luma from whatever the decoder produced.

```rust
pub struct BlankFrames {
    pub threshold: u8,
    pub blank: std::sync::atomic::AtomicU64,
}

impl DecodedFrameHook for BlankFrames {
    fn sampling(&self) -> FrameSampling {
        FrameSampling::every_seconds(0.5)
    }
    fn on_decoded_frame(&self, _ctx: &HookContext, f: &FrameEvent) -> Result<HookOutcome> {
        let luma = rivet::hooks::frame::luma8(&f.frame)?;
        let mean = luma.iter().map(|&v| u64::from(v)).sum::<u64>() / luma.len() as u64;
        let mut outcome = HookOutcome::proceed().annotate("mean_luma", mean);
        if mean <= u64::from(self.threshold) {
            self.blank.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            outcome = outcome.annotate("blank", true);
        }
        Ok(outcome)
    }
}
```

Hooks are called from the decode threads, possibly several at once, so keep
state in atomics or a `Mutex`.

## 6. Your own hashing library

Wrap the library in a type and implement the kinds for the points you want it
at. The same type can be registered as a decoded-frame hook and a still hook:

```rust
pub struct MyHash;

impl MyHash {
    fn hash(&self, frame: &rivet::codec::frame::VideoFrame) -> Result<String> {
        let rgb = rivet::hooks::frame::rgb8(frame)?;                      // 8-bit RGB, row-major
        Ok(format!("{:016x}", my_hasher::compute(&rgb, frame.width, frame.height)))
    }
}

impl DecodedFrameHook for MyHash {
    fn sampling(&self) -> FrameSampling {
        FrameSampling::every_seconds(1.0).max_frames(300)
    }
    fn on_decoded_frame(&self, _ctx: &HookContext, f: &FrameEvent) -> Result<HookOutcome> {
        Ok(HookOutcome::proceed().annotate("my_hash", self.hash(&f.frame)?))
    }
}

impl StillHook for MyHash {
    fn on_still(&self, _ctx: &HookContext, s: &StillEvent) -> Result<HookOutcome> {
        Ok(HookOutcome::proceed().annotate("my_hash", self.hash(&s.frame)?))
    }
}

let hooks = Hooks::new()
    .decoded_frames_with("my-hash", MyHash, HookPolicy::default().fail_closed())
    .stills_with("my-hash-stills", MyHash, HookPolicy::default().fail_closed());
```

`fail_closed()` makes a hashing error fail the job instead of letting the
job through unhashed. `max_frames(300)` caps the work on a long source. If
your library wants another layout (BGR, luma only, a fixed size), convert
from `rgb8` / `luma8`, or use `phash::shrink` to downscale luma.

## 7. Did the filters change the picture?

Decoded-frame and encoder-frame hooks see the same frame (same `clip` and
`index`) before and after the colour pipeline and the spec's filters. One
value registered at both points can compare them. An `Arc` of any hook kind
is that kind, so share state like this:

```rust
#[derive(Default)]
pub struct FilterDrift {
    decoded: std::sync::Mutex<std::collections::HashMap<(usize, u64), u64>>,
}

impl DecodedFrameHook for FilterDrift {
    fn sampling(&self) -> FrameSampling { FrameSampling::every_seconds(5.0) }
    fn on_decoded_frame(&self, _ctx: &HookContext, f: &FrameEvent) -> Result<HookOutcome> {
        let h = PerceptualAlgorithm::PHash.hash_frame(&f.frame)?;
        self.decoded.lock().unwrap().insert((f.clip, f.index), h);
        Ok(HookOutcome::proceed())
    }
}

impl EncoderFrameHook for FilterDrift {
    fn sampling(&self) -> FrameSampling { FrameSampling::every_seconds(5.0) }
    fn on_encoder_frame(&self, _ctx: &HookContext, f: &FrameEvent) -> Result<HookOutcome> {
        let after = PerceptualAlgorithm::PHash.hash_frame(&f.frame)?;
        let before = self.decoded.lock().unwrap().get(&(f.clip, f.index)).copied();
        Ok(match before {
            Some(b) => HookOutcome::proceed().annotate("filter_drift_bits", phash::hamming(b, after)),
            None => HookOutcome::proceed(),
        })
    }
}

let drift = std::sync::Arc::new(FilterDrift::default());
let hooks = Hooks::new()
    .decoded_frames("filter-drift-before", std::sync::Arc::clone(&drift))
    .encoder_frames("filter-drift-after", drift);
```

The decoded-frame hook for a frame always runs before that frame's
encoder-frame hook. Expect a few bits of drift even with no filters: the
colour pipeline (re-matrixing to BT.709, bit-depth and chroma conversion)
changes the pixels a little. The cookbook's unfiltered clip shows 8 bits.

## 8. Forward to your own system without blocking

To send events to a queue, a database or a service, hand them to a consumer
of your own and register the hook in the background. A slow consumer then
never stalls decoding. The job still waits for background hooks before it
returns, and a background rejection still fails the job.

```rust
#[derive(Debug)]
pub struct Forwarded { pub job_id: String, pub clip: usize, pub index: u64, pub phash: String }

pub struct Forward(pub std::sync::Mutex<std::sync::mpsc::Sender<Forwarded>>);

impl DecodedFrameHook for Forward {
    fn on_decoded_frame(&self, ctx: &HookContext, f: &FrameEvent) -> Result<HookOutcome> {
        let phash = phash::to_hex(PerceptualAlgorithm::PHash.hash_frame(&f.frame)?);
        self.0.lock().unwrap()
            .send(Forwarded { job_id: ctx.job_id.clone(), clip: f.clip, index: f.index, phash })
            .map_err(|_| anyhow::anyhow!("the consumer is gone"))?;
        Ok(HookOutcome::proceed())
    }
}

let (tx, rx) = std::sync::mpsc::channel();
std::thread::spawn(move || for item in rx { /* publish `item` */ });
let hooks = Hooks::new().decoded_frames_with("forward", Forward(std::sync::Mutex::new(tx)), HookPolicy::background());
```

For an async consumer, use a `tokio::sync::mpsc::Sender` and `blocking_send`
(hooks run on blocking threads, never on the async runtime). `ctx.job_id` is
the key to correlate with: it's the HTTP API's job id, or the id you gave
`Hooks::session`.

## 9. Cap output size

An artifact hook names the output kinds it wants (`Video`, `Audio`, `Image`,
`Rendition`, `Playlist`) and gets only those.

```rust
pub struct MaxOutputSize(pub usize);

impl ArtifactHook for MaxOutputSize {
    fn kinds(&self) -> Vec<ArtifactKind> {
        vec![ArtifactKind::Video, ArtifactKind::Audio, ArtifactKind::Image]
    }
    fn on_artifact(&self, _ctx: &HookContext, a: &ArtifactEvent) -> Result<HookOutcome> {
        let ArtifactData::Bytes(bytes) = &a.data else { return Ok(HookOutcome::proceed()) };
        Ok(if bytes.len() > self.0 {
            HookOutcome::reject(format!("`{}` came out {} bytes; the limit is {}", a.label, bytes.len(), self.0))
        } else {
            HookOutcome::proceed()
        })
    }
}
```

An HLS rendition arrives as `ArtifactData::Directory { path, files }` (already
on disk) and the master playlist as `ArtifactData::File`. Rejecting one fails
the job. The HTTP server then removes an output directory it created.

## 10. Write a sidecar manifest

```rust
pub struct Manifest(pub std::path::PathBuf);

impl ArtifactHook for Manifest {
    fn on_artifact(&self, ctx: &HookContext, a: &ArtifactEvent) -> Result<HookOutcome> {
        use std::io::Write as _;
        let (bytes, sha256) = match &a.data {
            ArtifactData::Bytes(b) => (Some(b.len()), Some(DigestAlgorithm::Sha256.hex(b))),
            _ => (None, None),
        };
        let line = serde_json::json!({
            "job": ctx.job_id, "label": a.label, "kind": a.kind.as_str(),
            "media_type": a.media_type, "bytes": bytes, "sha256": sha256,
        });
        let mut f = std::fs::OpenOptions::new().create(true).append(true).open(&self.0)?;
        writeln!(f, "{line}")?;
        Ok(HookOutcome::proceed())
    }
}
```

When all you need is the digests in the report, the built-in
`ArtifactDigest::new(&[DigestAlgorithm::Sha256]).kinds(&[ArtifactKind::Video])`
does it.

## 11. Metrics on how jobs end

Completed and failed are separate kinds. A completed hook is the job's last
gate (it can still reject). A failed hook runs once per failed job, and
`FailedEvent::rejection` says whether a hook stopped the job.

```rust
#[derive(Default)]
pub struct Metrics { pub completed: AtomicU64, pub failed: AtomicU64, pub rejected: AtomicU64 }

pub struct CountCompleted(pub Arc<Metrics>);
impl CompletedHook for CountCompleted {
    fn on_completed(&self, _ctx: &HookContext, done: &CompletedEvent) -> Result<HookOutcome> {
        self.0.completed.fetch_add(1, Ordering::Relaxed);
        Ok(HookOutcome::proceed().annotate("elapsed_ms", done.elapsed.as_millis() as u64))
    }
}

pub struct CountFailed(pub Arc<Metrics>);
impl FailedHook for CountFailed {
    fn on_failed(&self, _ctx: &HookContext, failed: &FailedEvent) -> Result<HookOutcome> {
        match &failed.rejection {
            Some(r) => { self.0.rejected.fetch_add(1, Ordering::Relaxed); eprintln!("rejected by {} hook `{}`: {}", r.kind, r.hook, r.reason); }
            None => { self.0.failed.fetch_add(1, Ordering::Relaxed); }
        }
        Ok(HookOutcome::proceed())
    }
}
```

Because the `Hooks` value is shared, one `Arc<Metrics>` counts across every
job that uses it.

## 12. Read the report of a rejected job

On success the report comes back in `JobOutput::hooks`. A rejected job
returns an error instead, so start the session yourself and keep a clone:

```rust
let session = hooks.session("upload-1234", JobKind::Transcode);
let spec = rivet::OutputSpec::single_file(vec![rivet::Rung::new(1280, 720)]).with_hooks(session.clone());
match rivet::run_job(input, &spec, None, sink).await {
    Ok(out) => println!("{} records", out.hooks.records.len()),
    Err(e) => match rejection_of(&e) {
        Some(r) => println!("rejected by {} hook `{}` at {}: {}", r.kind, r.hook, r.subject, r.reason),
        None => println!("failed: {e:#}"),
    },
}
let report = session.report();                       // the same, whatever happened
for (record, sha) in report.annotations("sha256") { println!("{} {}", record.hook, sha); }
for record in report.by_kind(HookKind::DecodedFrame) { /* every decoded-frame record */ }
println!("{}", report.to_json());
```

A record in `to_json()`:

```json
{ "hook": "source-fingerprint", "kind": "decoded-frame", "stage": "decoded-frame",
  "subject": { "type": "frame", "clip": 0, "index": 24, "seconds": 1.0 },
  "verdict": "continue", "reason": null, "annotations": { "phash": "860f07b35e0f13ec" },
  "error": null, "elapsed_ms": 0.41, "background": false }
```

For an image job, pass the sessioned hooks to
`image::run_image_job_with_hooks(&input, &spec, &session)`.

## 13. Fail open, fail closed, background

```rust
// Blocking, fail open, required: the default.
.source("a", A)
// A hook error or timeout fails the job.
.source_with("b", B, HookPolicy::default().fail_closed())
// Off the pipeline's thread; the job waits for it before returning.
.decoded_frames_with("c", C, HookPolicy::background())
// Both.
.decoded_frames_with("d", D, HookPolicy::background().fail_closed())
// Runs only on jobs that ask for it by name (the HTTP API's `hooks=`).
.probe_with("e", E, HookPolicy::default().optional())
```

| Policy | When the hook rejects | When the hook errors |
|--------|-----------------------|----------------------|
| blocking, fail open | the job stops now | recorded; the job carries on |
| blocking, fail closed | the job stops now | the job stops now |
| background, fail open | the job stops at its next point, at the latest before it returns | recorded; the job carries on |
| background, fail closed | as above | as a rejection |

The background queue is bounded, so a hook that can't keep up slows the job
down rather than buffering frames without limit.

## 14. Hooks on the HTTP API

Build the server with your hooks:

```rust
let hooks = Hooks::new()
    .source("source-digest", SourceDigest::new(&[DigestAlgorithm::Sha256]))
    .decoded_frames_with("my-hash", MyHash, HookPolicy::background().fail_closed())
    .probe_with("strict-limits", SourceLimits { max_width: 1920, max_height: 1080, max_seconds: 600.0, require_video: true },
                HookPolicy::default().optional());
rivet::server::serve_with_hooks("0.0.0.0:8080".parse()?, hooks).await?;
```

```sh
# What the server runs, with each hook's kind:
curl -s localhost:8080/v1/hooks | jq '.hooks[] | {name, kind, required}'

# Required hooks run on every job; opt into optional ones by name:
curl -s -X POST --data-binary @upload.mp4 "localhost:8080/v1/transcode?hooks=strict-limits"

# The job's hook report, live and after it ends:
curl -s localhost:8080/v1/jobs/$JOB | jq '.status, .hooks.rejection, [.hooks.records[] | {hook, kind, annotations}]'
```

A rejected job ends `"status": "rejected"`, and `?sync=true` gets `422`.
Requests can only choose among configured hooks. Naming an unknown one is a
`400`, and nothing in a request can define a hook.

## 15. Unit-test a hook without running a job

A session's `emit_*` methods are the calls the engine makes. Drive them
directly to test a hook in isolation:

```rust
#[test]
fn refuses_large_sources() {
    let hooks = Hooks::new().source("max", MaxSourceSize(4)).session("test", JobKind::Transcode);
    let err = hooks.emit_source(0, &bytes::Bytes::from_static(b"too big")).unwrap_err();
    assert_eq!(rejection_of(&err).unwrap().hook, "max");
}

#[test]
fn fingerprints_a_frame() {
    let hooks = Hooks::new()
        .decoded_frames("fp", PerceptualFingerprint::new(&[PerceptualAlgorithm::PHash]))
        .session("test", JobKind::Transcode);
    let frame = rivet::codec::frame::VideoFrame::new(
        bytes::Bytes::from(vec![128u8; 64 * 64 * 3 / 2]), 64, 64,
        rivet::codec::frame::PixelFormat::Yuv420p, rivet::codec::frame::ColorSpace::Bt709, 0,
    );
    hooks.emit_decoded_frame(0, 0, 30.0, &frame).unwrap();
    assert!(hooks.report().annotations("phash").next().is_some());
}
```

Also available: `emit_probe`, `emit_encoder_frame`, `emit_still`,
`emit_artifact`, `emit_completed`, and `emit_failed`.

## 16. See every event while developing

The general `Hook` trait sees whichever stages it lists. `LogHook` logs every
event it's handed through `tracing` (run with `RUST_LOG=info`):

```rust
let hooks = Hooks::new().with("log", LogHook { stages: StageSet::ALL, sampling: FrameSampling::every_seconds(10.0) });
```

Use a specific kind for real hooks. The general trait is for cross-cutting
tools like this one.
