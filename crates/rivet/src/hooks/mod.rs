//! Hooks: caller-supplied code run at specific points of every job.
//!
//! Each kind of hook has its own trait, written for one place in the pipeline
//! and handed exactly what exists there:
//!
//! | Kind | Trait | Register with | Where it hooks in | Handed |
//! |------|-------|---------------|-------------------|--------|
//! | source | [`SourceHook`] | [`Hooks::source`] | the source bytes, before anything parses them; once per input (each clip of a splice) | [`SourceEvent`] |
//! | probe | [`ProbeHook`] | [`Hooks::probe`] | the source once demuxed (an image once its header is read), before anything is decoded | [`ProbeEvent`] |
//! | decoded frame | [`DecodedFrameHook`] | [`Hooks::decoded_frames`] | video frames as the decoder produced them, turned upright — before tonemapping, chroma or bit-depth conversion, the spec's filters or any scaling | [`FrameEvent`] |
//! | encoder frame | [`EncoderFrameHook`] | [`Hooks::encoder_frames`] | video frames exactly as the encoders receive them — after the colour pipeline and the filters, before per-rung scaling | [`FrameEvent`] |
//! | still | [`StillHook`] | [`Hooks::stills`] | still pictures in an image job: the decoded image, or each still taken from a video | [`StillEvent`] |
//! | artifact | [`ArtifactHook`] | [`Hooks::artifacts`] | each output of the [`ArtifactKind`]s it accepts, before the job returns it | [`ArtifactEvent`] |
//! | completed | [`CompletedHook`] | [`Hooks::completed`] | the job made everything it was asked for (its last gate: it can still reject) | [`CompletedEvent`] |
//! | failed | [`FailedHook`] | [`Hooks::failed`] | the job failed, a rejection by a hook included | [`FailedEvent`] |
//!
//! Each answers with a [`HookOutcome`]: a [`Verdict`] (carry on, or reject the
//! job) and any [`Annotation`]s it wants recorded. Everything a job's hooks
//! said is collected into a [`HookReport`], returned with the output
//! ([`JobOutput::hooks`](crate::job::JobOutput)) and, for a job that was
//! stopped, readable from the session the caller started.
//!
//! The frame hooks choose which frames they are handed ([`FrameSampling`]): a
//! decode produces every frame, and a hook rarely needs them all.
//!
//! The general [`Hook`] trait — any set of [`Stage`]s, one `call` for all of
//! them — is what the kinds adapt to, and is there for the rare hook that
//! genuinely spans a job ([`Hooks::with`]).
//!
//! The engine has no opinion on what a hook is for. The built-in ones compute
//! and record: [`SourceDigest`] (content digests of the source bytes),
//! [`PerceptualFingerprint`] (perceptual hashes of decoded frames, encoder
//! frames or stills) and [`ArtifactDigest`] (digests of outputs). What is done
//! with a digest or a hash is the integration's: the building blocks are
//! public ([`phash`], [`DigestAlgorithm`], [`frame`]), so a custom hook
//! computes exactly what a built-in one does and takes it from there.
//!
//! ## Blocking, background, and failure
//!
//! Each registered hook carries a [`HookPolicy`]. A **blocking** hook runs on
//! the thread that reached its stage and its verdict takes effect at once — a
//! rejected source never reaches the demuxer, a rejected frame stops every
//! decode pump. A **background** hook runs on the session's worker thread so a
//! slow one does not hold up the pipeline; a rejection from it stops the job
//! at the next stage it reaches, and at the latest before the job returns its
//! output, because the job waits for its background hooks before it finishes.
//!
//! A hook that errors is recorded, and then either ignored
//! ([`OnError::Continue`], the default) or treated as a rejection
//! ([`OnError::Reject`]) — fail open or fail closed, as the deployment needs.
//!
//! ## Sessions
//!
//! A [`Hooks`] value is the set of hooks; a *session* is one job's run of
//! them: its id, its records, its rejection. The engine starts one for every
//! job that has hooks, unless the caller started one already
//! ([`Hooks::session`]) — which is how a caller gets at the report of a job
//! that failed or was rejected: keep a clone of the sessioned `Hooks` and call
//! [`Hooks::report`] whatever the job returned.
//!
//! ```
//! use rivet::hooks::{
//!     DigestAlgorithm, FrameSampling, HookContext, HookOutcome, Hooks, PerceptualAlgorithm,
//!     PerceptualFingerprint, SourceDigest, SourceEvent, SourceHook,
//! };
//!
//! /// Refuses sources over a size.
//! struct SizeLimit(usize);
//!
//! impl SourceHook for SizeLimit {
//!     fn on_source(&self, _ctx: &HookContext, source: &SourceEvent) -> anyhow::Result<HookOutcome> {
//!         Ok(if source.bytes.len() > self.0 {
//!             HookOutcome::reject("the source is over the size limit")
//!         } else {
//!             HookOutcome::proceed()
//!         })
//!     }
//! }
//!
//! let hooks = Hooks::new()
//!     .source("size-limit", SizeLimit(1 << 30))
//!     .source("source-digest", SourceDigest::new(&[DigestAlgorithm::Sha256]))
//!     .decoded_frames(
//!         "source-fingerprint",
//!         PerceptualFingerprint::new(&[PerceptualAlgorithm::PHash]).sampling(FrameSampling::every_seconds(1.0)),
//!     );
//! let spec = rivet::OutputSpec::default().with_hooks(hooks);
//! # let _ = spec;
//! ```

use std::fmt;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Result, bail};
use bytes::Bytes;
use serde_json::{Value, json};

use codec::frame::VideoFrame;

mod builtin;
mod digest;
pub mod frame;
mod kinds;
pub mod phash;
#[cfg(test)]
mod tests;

pub use builtin::{ArtifactDigest, PerceptualFingerprint, SourceDigest};
pub use digest::DigestAlgorithm;
pub use frame::FrameFormat;
pub use kinds::{
    ArtifactHook, ArtifactKind, CompletedHook, DecodedFrameHook, EncoderFrameHook, FailedHook, HookKind, ProbeHook,
    SourceHook, StillHook,
};
pub use phash::PerceptualAlgorithm;

// ---------------------------------------------------------------------------
// Stages
// ---------------------------------------------------------------------------

/// A point in a job where hooks run. One per hook kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Stage {
    /// The source bytes, before anything parses them.
    Source,
    /// The source, described: container, codecs, size, rate, duration.
    Probe,
    /// A video frame as decoded, before the colour pipeline and filters.
    DecodedFrame,
    /// A video frame as the encoders receive it.
    EncoderFrame,
    /// A still picture in an image job.
    Still,
    /// One output the job made, before it is handed back.
    Artifact,
    /// The job made everything it was asked for.
    Completed,
    /// The job failed, a rejection by a hook included.
    Failed,
}

impl Stage {
    /// Every stage, in the order a job reaches them.
    pub const ALL: [Stage; 8] = [
        Stage::Source,
        Stage::Probe,
        Stage::DecodedFrame,
        Stage::EncoderFrame,
        Stage::Still,
        Stage::Artifact,
        Stage::Completed,
        Stage::Failed,
    ];

    /// The stage's name in listings and reports.
    pub fn as_str(self) -> &'static str {
        match self {
            Stage::Source => "source",
            Stage::Probe => "probe",
            Stage::DecodedFrame => "decoded-frame",
            Stage::EncoderFrame => "encoder-frame",
            Stage::Still => "still",
            Stage::Artifact => "artifact",
            Stage::Completed => "completed",
            Stage::Failed => "failed",
        }
    }

    /// The frame stages: handed frames from a running decode, sampled.
    pub fn is_sampled(self) -> bool {
        matches!(self, Stage::DecodedFrame | Stage::EncoderFrame)
    }

    fn bit(self) -> u8 {
        1 << (self as u8)
    }
}

impl fmt::Display for Stage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for Stage {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        Ok(match s.trim().to_ascii_lowercase().replace('_', "-").as_str() {
            "source" => Stage::Source,
            "probe" => Stage::Probe,
            "decoded-frame" | "decoded-frames" => Stage::DecodedFrame,
            "encoder-frame" | "encoder-frames" => Stage::EncoderFrame,
            "still" | "stills" => Stage::Still,
            "artifact" | "artifacts" => Stage::Artifact,
            "completed" => Stage::Completed,
            "failed" => Stage::Failed,
            other => bail!(
                "unknown hook stage `{other}` (source, probe, decoded-frame, encoder-frame, still, artifact, completed, failed)"
            ),
        })
    }
}

/// A set of [`Stage`]s.
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub struct StageSet(u8);

impl StageSet {
    pub const NONE: StageSet = StageSet(0);
    pub const ALL: StageSet = StageSet(u8::MAX);

    pub fn of(stages: &[Stage]) -> Self {
        Self(stages.iter().fold(0, |acc, s| acc | s.bit()))
    }

    pub fn contains(self, stage: Stage) -> bool {
        self.0 & stage.bit() != 0
    }

    pub fn with(self, stage: Stage) -> Self {
        Self(self.0 | stage.bit())
    }

    pub fn union(self, other: StageSet) -> Self {
        Self(self.0 | other.0)
    }

    pub fn intersect(self, other: StageSet) -> Self {
        Self(self.0 & other.0)
    }

    pub fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Whether any stage in the set is a sampled frame stage.
    pub fn has_sampled(self) -> bool {
        self.iter().any(Stage::is_sampled)
    }

    pub fn iter(self) -> impl Iterator<Item = Stage> {
        Stage::ALL.into_iter().filter(move |s| self.contains(*s))
    }

    /// `source,decoded-frame` → the set. `all` is every stage.
    pub fn parse(list: &str) -> Result<Self> {
        let mut set = StageSet::NONE;
        for part in list.split([',', '+', ' ']).filter(|p| !p.trim().is_empty()) {
            if part.trim().eq_ignore_ascii_case("all") {
                return Ok(StageSet::ALL);
            }
            set = set.with(part.parse()?);
        }
        if set.is_empty() {
            bail!("a hook needs at least one stage");
        }
        Ok(set)
    }
}

impl fmt::Debug for StageSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_set().entries(self.iter().map(Stage::as_str)).finish()
    }
}

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------

/// What kind of job a session belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JobKind {
    /// [`run_job`](crate::job::run_job): one input, video out.
    Transcode,
    /// [`run_splice_job`](crate::job::run_splice_job): several inputs joined.
    Splice,
    /// An audio-only output.
    AudioOnly,
    /// A still-image job (the `image` feature).
    Image,
}

impl JobKind {
    pub fn as_str(self) -> &'static str {
        match self {
            JobKind::Transcode => "transcode",
            JobKind::Splice => "splice",
            JobKind::AudioOnly => "audio",
            JobKind::Image => "image",
        }
    }
}

/// Who an event is about, for the report.
#[derive(Debug, Clone, PartialEq)]
pub enum Subject {
    /// The job as a whole.
    Job,
    /// One source (the clip's position in a splice; `0` otherwise).
    Source { clip: usize },
    /// One video frame: its source, its index in that source's frames
    /// (presentation order, before any trim), and its time in seconds.
    Frame { clip: usize, index: u64, seconds: f64 },
    /// One still: its position in the job's stills, and for a still taken
    /// from a video, its time.
    Still { clip: usize, index: u64, seconds: f64 },
    /// One output, by label.
    Artifact { label: String },
}

impl Subject {
    pub fn to_json(&self) -> Value {
        match self {
            Subject::Job => json!({ "type": "job" }),
            Subject::Source { clip } => json!({ "type": "source", "clip": clip }),
            Subject::Frame { clip, index, seconds } => {
                json!({ "type": "frame", "clip": clip, "index": index, "seconds": seconds })
            }
            Subject::Still { clip, index, seconds } => {
                json!({ "type": "still", "clip": clip, "index": index, "seconds": seconds })
            }
            Subject::Artifact { label } => json!({ "type": "artifact", "label": label }),
        }
    }
}

impl fmt::Display for Subject {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Subject::Job => f.write_str("the job"),
            Subject::Source { clip } => write!(f, "source {clip}"),
            Subject::Frame { clip, index, seconds } => write!(f, "frame {index} of source {clip} ({seconds:.3}s)"),
            Subject::Still { clip, index, .. } => write!(f, "still {index} of source {clip}"),
            Subject::Artifact { label } => write!(f, "artifact `{label}`"),
        }
    }
}

/// The source bytes, before anything parses them.
#[derive(Debug, Clone)]
pub struct SourceEvent {
    /// The source's position in a splice; `0` otherwise.
    pub clip: usize,
    pub bytes: Bytes,
    /// What the first bytes look like (`mp4`, `matroska`, `jpeg`, ...), or
    /// `unknown`. A sniff, not a parse.
    pub sniffed: String,
}

/// A source, described.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MediaSummary {
    /// Container (`mp4`, `matroska`, ...) or image format (`jpeg`, ...).
    pub container: String,
    /// Video (or coded image) codec; `None` for an audio-only source.
    pub video_codec: Option<String>,
    /// Upright size, in pixels.
    pub width: u32,
    pub height: u32,
    pub frame_rate: f64,
    pub duration: f64,
    /// Frame count, when the container states it.
    pub frames: Option<u64>,
    pub audio_codec: Option<String>,
    /// A still image rather than a video.
    pub still: bool,
}

impl MediaSummary {
    /// From what [`crate::probe`] reports.
    pub fn of_media_info(info: &crate::probe::MediaInfo, still: bool) -> Self {
        Self {
            container: info.container.clone(),
            video_codec: (info.video_codec != "none").then(|| info.video_codec.clone()),
            width: info.width,
            height: info.height,
            frame_rate: info.frame_rate,
            duration: info.duration,
            frames: None,
            audio_codec: info.audio.as_ref().map(|a| a.codec.clone()),
            still,
        }
    }

    pub fn to_json(&self) -> Value {
        json!({
            "container": self.container,
            "video_codec": self.video_codec,
            "width": self.width,
            "height": self.height,
            "frame_rate": self.frame_rate,
            "duration": self.duration,
            "frames": self.frames,
            "audio_codec": self.audio_codec,
            "still": self.still,
        })
    }

    pub(crate) fn of_header(container: &str, header: &container::streaming::DemuxHeader, audio_codec: Option<String>) -> Self {
        let (width, height) = header.upright_dims();
        Self {
            container: container.to_string(),
            video_codec: Some(header.codec.to_ascii_lowercase()),
            width,
            height,
            frame_rate: header.info.frame_rate,
            duration: header.info.duration,
            frames: (header.info.total_frames > 0).then_some(header.info.total_frames),
            audio_codec,
            still: false,
        }
    }
}

/// The source described, once demuxed.
#[derive(Debug, Clone)]
pub struct ProbeEvent {
    pub clip: usize,
    pub media: MediaSummary,
}

/// One video frame from a running decode — handed to decoded-frame and
/// encoder-frame hooks, each at its own point.
#[derive(Debug, Clone)]
pub struct FrameEvent {
    pub clip: usize,
    /// The frame's index among its source's frames, in presentation order,
    /// before any trim or frame-rate cap.
    pub index: u64,
    /// Its time in seconds from the start of its source.
    pub seconds: f64,
    /// The pixels. Cheap to clone (the plane data is shared). See [`frame`]
    /// for 8-bit luma / RGB views and PGM / PPM encodings.
    pub frame: VideoFrame,
}

/// One still picture in an image job.
#[derive(Debug, Clone)]
pub struct StillEvent {
    pub clip: usize,
    /// Its position among the job's stills (`0` for a still image).
    pub index: u64,
    /// For a still taken from a video, its time in seconds; `0` otherwise.
    pub seconds: f64,
    /// Upright 8-bit RGBA.
    pub frame: VideoFrame,
    /// Taken from a video rather than decoded from an image.
    pub from_video: bool,
}

/// What an artifact holds.
#[derive(Debug, Clone)]
pub enum ArtifactData {
    /// The artifact's bytes (a single-file rung, an image, an audio file).
    Bytes(Bytes),
    /// A directory already written (an HLS rendition), and the files the
    /// rendition is made of, relative to it.
    Directory { path: PathBuf, files: Vec<String> },
    /// A file already written (an HLS master playlist).
    File(PathBuf),
}

/// One output the job made.
#[derive(Debug, Clone)]
pub struct ArtifactEvent {
    pub kind: ArtifactKind,
    pub label: String,
    /// Media type (`video/mp4`, `image/avif`, `application/vnd.apple.mpegurl`, ...).
    pub media_type: String,
    pub width: u32,
    pub height: u32,
    pub data: ArtifactData,
}

/// The job made everything it was asked for.
#[derive(Debug, Clone)]
pub struct CompletedEvent {
    pub artifacts: usize,
    pub elapsed: Duration,
}

/// The job failed.
#[derive(Debug, Clone)]
pub struct FailedEvent {
    /// The whole error chain.
    pub error: String,
    /// When a hook stopped the job, which and why.
    pub rejection: Option<HookRejection>,
}

/// Everything a hook can be handed: one variant per [`Stage`].
#[derive(Debug, Clone)]
pub enum HookEvent {
    Source(SourceEvent),
    Probe(ProbeEvent),
    DecodedFrame(FrameEvent),
    EncoderFrame(FrameEvent),
    Still(StillEvent),
    Artifact(ArtifactEvent),
    Completed(CompletedEvent),
    Failed(FailedEvent),
}

impl HookEvent {
    pub fn stage(&self) -> Stage {
        match self {
            HookEvent::Source(_) => Stage::Source,
            HookEvent::Probe(_) => Stage::Probe,
            HookEvent::DecodedFrame(_) => Stage::DecodedFrame,
            HookEvent::EncoderFrame(_) => Stage::EncoderFrame,
            HookEvent::Still(_) => Stage::Still,
            HookEvent::Artifact(_) => Stage::Artifact,
            HookEvent::Completed(_) => Stage::Completed,
            HookEvent::Failed(_) => Stage::Failed,
        }
    }

    pub fn subject(&self) -> Subject {
        match self {
            HookEvent::Source(e) => Subject::Source { clip: e.clip },
            HookEvent::Probe(e) => Subject::Source { clip: e.clip },
            HookEvent::DecodedFrame(e) | HookEvent::EncoderFrame(e) => {
                Subject::Frame { clip: e.clip, index: e.index, seconds: e.seconds }
            }
            HookEvent::Still(e) => Subject::Still { clip: e.clip, index: e.index, seconds: e.seconds },
            HookEvent::Artifact(e) => Subject::Artifact { label: e.label.clone() },
            HookEvent::Completed(_) | HookEvent::Failed(_) => Subject::Job,
        }
    }

    /// The event's metadata as JSON — everything but the pixel and byte
    /// payloads. For logging, and for integrations that forward events.
    pub fn metadata_json(&self) -> Value {
        let frame_json = |clip: usize, index: u64, seconds: f64, f: &VideoFrame| {
            json!({
                "clip": clip,
                "index": index,
                "seconds": seconds,
                "width": f.width,
                "height": f.height,
                "pixel_format": f.format.as_ffmpeg_str(),
            })
        };
        match self {
            HookEvent::Source(e) => json!({ "clip": e.clip, "bytes": e.bytes.len(), "sniffed": e.sniffed }),
            HookEvent::Probe(e) => json!({ "clip": e.clip, "media": e.media.to_json() }),
            HookEvent::DecodedFrame(e) | HookEvent::EncoderFrame(e) => frame_json(e.clip, e.index, e.seconds, &e.frame),
            HookEvent::Still(e) => {
                let mut v = frame_json(e.clip, e.index, e.seconds, &e.frame);
                v["from_video"] = e.from_video.into();
                v
            }
            HookEvent::Artifact(e) => {
                let (data, bytes, path, files) = match &e.data {
                    ArtifactData::Bytes(b) => ("bytes", Some(b.len()), None, None),
                    ArtifactData::Directory { path, files } => {
                        ("directory", None, Some(path.display().to_string()), Some(files.clone()))
                    }
                    ArtifactData::File(p) => ("file", None, Some(p.display().to_string()), None),
                };
                json!({
                    "kind": e.kind.as_str(),
                    "label": e.label,
                    "media_type": e.media_type,
                    "width": e.width,
                    "height": e.height,
                    "data": data,
                    "bytes": bytes,
                    "path": path,
                    "files": files,
                })
            }
            HookEvent::Completed(e) => json!({ "artifacts": e.artifacts, "elapsed_ms": e.elapsed.as_millis() as u64 }),
            HookEvent::Failed(e) => json!({
                "error": e.error,
                "rejection": e.rejection.as_ref().map(HookRejection::to_json),
            }),
        }
    }
}

/// What a hook is told about the job besides the event.
#[derive(Debug, Clone)]
pub struct HookContext {
    pub job_id: String,
    pub job_kind: JobKind,
    /// The name the hook was registered under.
    pub hook: String,
}

// ---------------------------------------------------------------------------
// Outcomes
// ---------------------------------------------------------------------------

/// A hook's answer about whether the job goes on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    Continue,
    Reject { reason: String },
}

impl Verdict {
    pub fn as_str(&self) -> &'static str {
        match self {
            Verdict::Continue => "continue",
            Verdict::Reject { .. } => "reject",
        }
    }
}

/// One named value a hook wants recorded.
#[derive(Debug, Clone, PartialEq)]
pub struct Annotation {
    pub key: String,
    pub value: Value,
}

/// A hook's answer to one event.
#[derive(Debug, Clone, PartialEq)]
pub struct HookOutcome {
    pub verdict: Verdict,
    pub annotations: Vec<Annotation>,
}

impl HookOutcome {
    /// Carry on, nothing to record.
    pub fn proceed() -> Self {
        Self { verdict: Verdict::Continue, annotations: Vec::new() }
    }

    /// Stop the job.
    pub fn reject(reason: impl Into<String>) -> Self {
        Self { verdict: Verdict::Reject { reason: reason.into() }, annotations: Vec::new() }
    }

    /// Record `key = value` with the outcome.
    pub fn annotate(mut self, key: impl Into<String>, value: impl Into<Value>) -> Self {
        self.annotations.push(Annotation { key: key.into(), value: value.into() });
        self
    }

    /// The same outcome rejecting, keeping its annotations.
    pub fn rejecting(mut self, reason: impl Into<String>) -> Self {
        self.verdict = Verdict::Reject { reason: reason.into() };
        self
    }
}

/// A hook stopped the job. The error a job returns when that happens; find it
/// in an `anyhow::Error` with [`rejection_of`].
#[derive(Debug, Clone, PartialEq)]
pub struct HookRejection {
    pub hook: String,
    pub kind: HookKind,
    pub stage: Stage,
    pub subject: Subject,
    pub reason: String,
}

impl HookRejection {
    pub fn to_json(&self) -> Value {
        json!({
            "hook": self.hook,
            "kind": self.kind.as_str(),
            "stage": self.stage.as_str(),
            "subject": self.subject.to_json(),
            "reason": self.reason,
        })
    }
}

impl fmt::Display for HookRejection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "rejected by {} hook `{}` ({}): {}", self.kind, self.hook, self.subject, self.reason)
    }
}

impl std::error::Error for HookRejection {}

/// The [`HookRejection`] anywhere in `err`'s chain: `Some` when a hook stopped
/// the job, `None` when it failed for any other reason.
pub fn rejection_of(err: &anyhow::Error) -> Option<&HookRejection> {
    err.chain().find_map(|e| e.downcast_ref::<HookRejection>())
}

// ---------------------------------------------------------------------------
// Frame sampling
// ---------------------------------------------------------------------------

/// Which frames a decoded-frame or encoder-frame hook is handed.
///
/// A decode runs every frame through the pipeline; a hook rarely needs them
/// all. A frame is selected when it is the first, or falls on `every_frames`,
/// or starts a new `every_seconds` interval — and only until `max_frames`
/// have been handed over. With neither stride set, every frame is selected.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameSampling {
    pub every_frames: Option<u64>,
    pub every_seconds: Option<f64>,
    pub max_frames: Option<u64>,
}

impl Default for FrameSampling {
    /// One frame a second, no limit.
    fn default() -> Self {
        Self { every_frames: None, every_seconds: Some(1.0), max_frames: None }
    }
}

impl FrameSampling {
    /// Every frame.
    pub fn all() -> Self {
        Self { every_frames: None, every_seconds: None, max_frames: None }
    }

    pub fn every_frames(n: u64) -> Self {
        Self { every_frames: Some(n.max(1)), every_seconds: None, max_frames: None }
    }

    pub fn every_seconds(s: f64) -> Self {
        Self { every_frames: None, every_seconds: Some(s), max_frames: None }
    }

    pub fn max_frames(self, n: u64) -> Self {
        Self { max_frames: Some(n), ..self }
    }

    /// Whether frame `index` of a stream at `fps` is selected (before
    /// `max_frames`, which the session counts).
    pub fn selects(&self, index: u64, fps: f64) -> bool {
        if index == 0 {
            return true;
        }
        let by_frames = self.every_frames.map(|n| index.is_multiple_of(n.max(1)));
        let by_seconds = self.every_seconds.filter(|s| *s > 0.0).map(|interval| {
            let fps = if fps > 0.0 { fps } else { 30.0 };
            let bucket = |i: u64| ((i as f64 / fps) / interval + 1e-9).floor() as u64;
            bucket(index) != bucket(index - 1)
        });
        match (by_frames, by_seconds) {
            (None, None) => true,
            (a, b) => a.unwrap_or(false) || b.unwrap_or(false),
        }
    }

    pub fn to_json(&self) -> Value {
        json!({
            "every_frames": self.every_frames,
            "every_seconds": self.every_seconds,
            "max_frames": self.max_frames,
        })
    }
}

// ---------------------------------------------------------------------------
// The general hook trait
// ---------------------------------------------------------------------------

/// A hook at whichever [`Stage`]s it lists, one `call` for all of them. The
/// specific kinds ([`SourceHook`], [`DecodedFrameHook`], ...) adapt to this;
/// implement it directly only for a hook that genuinely spans the job.
///
/// `call` may run on any thread — a decode pump's, a Tokio worker, the
/// session's background thread — and for frames, on several at once (a source
/// decoded in ranges on several GPUs). Keep it `Send + Sync` and do not
/// assume events arrive in order.
pub trait Hook: Send + Sync {
    /// The stages it runs at.
    fn stages(&self) -> StageSet;

    /// Which frames it is handed at the sampled frame stages.
    fn sampling(&self) -> FrameSampling {
        FrameSampling::default()
    }

    /// Answer one event.
    fn call(&self, ctx: &HookContext, event: &HookEvent) -> Result<HookOutcome>;

    /// A one-line description for listings (`GET /v1/hooks`).
    fn describe(&self) -> String {
        String::from("custom")
    }

    /// Which kind it is. [`HookKind::Event`] for a general hook.
    fn kind(&self) -> HookKind {
        HookKind::Event
    }

    /// The outputs it is handed at [`Stage::Artifact`]. Default: every kind.
    fn artifact_kinds(&self) -> Vec<ArtifactKind> {
        ArtifactKind::ALL.to_vec()
    }
}

/// A general hook from a closure; see [`fn_hook`].
pub struct FnHook<F> {
    stages: StageSet,
    sampling: FrameSampling,
    f: F,
}

impl<F> FnHook<F> {
    pub fn sampling(mut self, sampling: FrameSampling) -> Self {
        self.sampling = sampling;
        self
    }
}

impl<F> Hook for FnHook<F>
where
    F: Fn(&HookContext, &HookEvent) -> Result<HookOutcome> + Send + Sync,
{
    fn stages(&self) -> StageSet {
        self.stages
    }
    fn sampling(&self) -> FrameSampling {
        self.sampling
    }
    fn call(&self, ctx: &HookContext, event: &HookEvent) -> Result<HookOutcome> {
        (self.f)(ctx, event)
    }
    fn describe(&self) -> String {
        format!("closure at {:?}", self.stages)
    }
}

/// Wrap a closure as a general [`Hook`] run at `stages`.
pub fn fn_hook<F>(stages: StageSet, f: F) -> FnHook<F>
where
    F: Fn(&HookContext, &HookEvent) -> Result<HookOutcome> + Send + Sync,
{
    FnHook { stages, sampling: FrameSampling::default(), f }
}

/// A general hook that logs every event it is handed through `tracing`, and
/// records nothing. For seeing what a job's hooks will be handed.
pub struct LogHook {
    pub stages: StageSet,
    pub sampling: FrameSampling,
}

impl Hook for LogHook {
    fn stages(&self) -> StageSet {
        self.stages
    }
    fn sampling(&self) -> FrameSampling {
        self.sampling
    }
    fn call(&self, ctx: &HookContext, event: &HookEvent) -> Result<HookOutcome> {
        tracing::info!(
            hook = %ctx.hook,
            job = %ctx.job_id,
            stage = %event.stage(),
            event = %event.metadata_json(),
            "hook event"
        );
        Ok(HookOutcome::proceed())
    }
    fn describe(&self) -> String {
        format!("log at {:?}", self.stages)
    }
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// How a hook runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HookMode {
    /// On the thread that reached the stage; the verdict takes effect at once.
    #[default]
    Blocking,
    /// On the session's worker thread; a rejection stops the job at its next
    /// stage, and the job waits for these before it returns.
    Background,
}

/// What a hook that errors means for the job.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OnError {
    /// Record the error and carry on (fail open).
    #[default]
    Continue,
    /// Treat it as a rejection (fail closed).
    Reject,
}

/// How one registered hook runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HookPolicy {
    pub mode: HookMode,
    pub on_error: OnError,
    /// Runs on every job. A hook that is not runs only on a job that names it
    /// ([`Hooks::select`]) — the HTTP API's `hooks=` parameter.
    pub required: bool,
}

impl Default for HookPolicy {
    fn default() -> Self {
        Self { mode: HookMode::Blocking, on_error: OnError::Continue, required: true }
    }
}

impl HookPolicy {
    pub fn background() -> Self {
        Self { mode: HookMode::Background, ..Self::default() }
    }

    pub fn fail_closed(self) -> Self {
        Self { on_error: OnError::Reject, ..self }
    }

    pub fn optional(self) -> Self {
        Self { required: false, ..self }
    }
}

#[derive(Clone)]
struct Entry {
    name: String,
    hook: Arc<dyn Hook>,
    policy: HookPolicy,
    kind: HookKind,
    stages: StageSet,
    sampling: FrameSampling,
    artifact_kinds: Vec<ArtifactKind>,
}

/// A set of hooks, and — once a job starts — that job's session of them.
///
/// Cheap to clone; carried on [`OutputSpec::hooks`](crate::spec::OutputSpec).
/// Empty by default, and an empty set costs a job nothing.
#[derive(Clone, Default)]
pub struct Hooks {
    entries: Arc<Vec<Entry>>,
    session: Option<Arc<Session>>,
}

impl fmt::Debug for Hooks {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut d = f.debug_struct("Hooks");
        d.field("hooks", &self.entries.iter().map(|e| format!("{} ({})", e.name, e.kind)).collect::<Vec<_>>());
        if let Some(s) = &self.session {
            d.field("job_id", &s.job_id);
        }
        d.finish()
    }
}

impl Hooks {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a general `hook` under `name`, with the default [`HookPolicy`].
    /// Prefer the method for the hook's kind ([`source`](Self::source),
    /// [`decoded_frames`](Self::decoded_frames), ...).
    pub fn with(self, name: impl Into<String>, hook: impl Hook + 'static) -> Self {
        self.with_policy(name, Arc::new(hook), HookPolicy::default())
    }

    /// Add `hook` under `name` with `policy`. A name already taken is
    /// replaced.
    pub fn with_policy(mut self, name: impl Into<String>, hook: Arc<dyn Hook>, policy: HookPolicy) -> Self {
        let name = name.into();
        let entry = Entry {
            stages: hook.stages(),
            sampling: hook.sampling(),
            kind: hook.kind(),
            artifact_kinds: hook.artifact_kinds(),
            name: name.clone(),
            hook,
            policy,
        };
        let entries = Arc::make_mut(&mut self.entries);
        entries.retain(|e| e.name != name);
        entries.push(entry);
        self
    }

    /// Every hook in `other` added to this set (a name in both takes
    /// `other`'s).
    pub fn merged(mut self, other: &Hooks) -> Self {
        for e in other.entries.iter() {
            self = self.with_policy(e.name.clone(), Arc::clone(&e.hook), e.policy);
        }
        self
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// The registered names, in the order they run.
    pub fn names(&self) -> Vec<String> {
        self.entries.iter().map(|e| e.name.clone()).collect()
    }

    /// Each hook's name, kind, stages, policy and description — for listings.
    pub fn describe(&self) -> Vec<Value> {
        self.entries
            .iter()
            .map(|e| {
                json!({
                    "name": e.name,
                    "kind": e.kind.as_str(),
                    "description": e.hook.describe(),
                    "stages": e.stages.iter().map(Stage::as_str).collect::<Vec<_>>(),
                    "mode": match e.policy.mode { HookMode::Blocking => "blocking", HookMode::Background => "background" },
                    "on_error": match e.policy.on_error { OnError::Continue => "continue", OnError::Reject => "reject" },
                    "required": e.policy.required,
                    "frames": e.stages.has_sampled().then(|| e.sampling.to_json()),
                    "artifact_kinds": e.stages.contains(Stage::Artifact)
                        .then(|| e.artifact_kinds.iter().map(|k| k.as_str()).collect::<Vec<_>>()),
                })
            })
            .collect()
    }

    /// The hooks a job runs: every required one, and the optional ones it
    /// names. Naming a hook that is not registered is an error, so a typo is
    /// not a silently skipped hook.
    pub fn select(&self, names: &[String]) -> Result<Hooks> {
        for n in names {
            if !self.entries.iter().any(|e| &e.name == n) {
                bail!("no hook named `{n}` is configured (configured: {})", self.names().join(", "));
            }
        }
        let entries: Vec<Entry> =
            self.entries.iter().filter(|e| e.policy.required || names.contains(&e.name)).cloned().collect();
        Ok(Hooks { entries: Arc::new(entries), session: None })
    }

    /// Whether any hook runs at `stage`.
    pub fn wants(&self, stage: Stage) -> bool {
        self.entries.iter().any(|e| e.stages.contains(stage))
    }

    // -- sessions ----------------------------------------------------------

    /// This set with a fresh session for job `job_id`. Keep a clone to read
    /// the [`report`](Self::report) whatever the job returns.
    pub fn session(&self, job_id: impl Into<String>, kind: JobKind) -> Hooks {
        Hooks {
            entries: Arc::clone(&self.entries),
            session: Some(Arc::new(Session::new(job_id.into(), kind, self.entries.len()))),
        }
    }

    /// This set with its session, starting one (with a generated id) when it
    /// has none. What the engine calls at the top of a job.
    pub fn ensure_session(&self, kind: JobKind) -> Hooks {
        match &self.session {
            Some(_) => self.clone(),
            None => self.session(generate_job_id(), kind),
        }
    }

    /// The session's job id, when there is a session.
    pub fn job_id(&self) -> Option<&str> {
        self.session.as_deref().map(|s| s.job_id.as_str())
    }

    /// Everything the session's hooks said so far. Empty with no session.
    pub fn report(&self) -> HookReport {
        match &self.session {
            Some(s) => HookReport {
                job_id: Some(s.job_id.clone()),
                records: s.records.lock().unwrap().clone(),
                rejection: s.rejection.lock().unwrap().clone(),
            },
            None => HookReport::default(),
        }
    }

    /// The session's rejection so far, when a hook stopped the job.
    pub fn rejection(&self) -> Option<HookRejection> {
        self.session.as_deref().and_then(|s| s.rejection.lock().unwrap().clone())
    }

    /// Whether every frame hook has been handed its `max_frames`.
    pub fn frames_exhausted(&self) -> bool {
        let Some(s) = &self.session else { return true };
        self.entries.iter().enumerate().filter(|(_, e)| e.stages.has_sampled()).all(|(i, e)| {
            e.sampling.max_frames.is_some_and(|max| s.frame_counts[i].load(Ordering::Relaxed) >= max)
        })
    }

    // -- dispatch: what the engine calls at each stage -----------------------

    /// `Err` with the session's rejection when there is one — what every stage
    /// checks first, so one rejection stops every thread at its next stage.
    pub fn check(&self) -> Result<()> {
        match self.rejection() {
            Some(r) => Err(anyhow::Error::new(r)),
            None => Ok(()),
        }
    }

    /// The source bytes of source `clip`, to the source hooks.
    pub fn emit_source(&self, clip: usize, bytes: &Bytes) -> Result<()> {
        if !self.wants(Stage::Source) {
            return self.check();
        }
        let sniffed = sniff_label(bytes);
        self.dispatch(HookEvent::Source(SourceEvent { clip, bytes: bytes.clone(), sniffed }), |_, _| true)
    }

    /// The description of source `clip`, to the probe hooks.
    pub fn emit_probe(&self, clip: usize, media: MediaSummary) -> Result<()> {
        if !self.wants(Stage::Probe) {
            return self.check();
        }
        self.dispatch(HookEvent::Probe(ProbeEvent { clip, media }), |_, _| true)
    }

    /// A video frame as decoded, to the decoded-frame hooks whose sampling
    /// selects it. `index` is its place in its source (presentation order);
    /// `fps` the source's rate, which turns the index into seconds and drives
    /// interval sampling.
    pub fn emit_decoded_frame(&self, clip: usize, index: u64, fps: f64, frame: &VideoFrame) -> Result<()> {
        self.emit_sampled(Stage::DecodedFrame, clip, index, fps, frame)
    }

    /// A video frame as the encoders receive it, to the encoder-frame hooks
    /// whose sampling selects it.
    pub fn emit_encoder_frame(&self, clip: usize, index: u64, fps: f64, frame: &VideoFrame) -> Result<()> {
        self.emit_sampled(Stage::EncoderFrame, clip, index, fps, frame)
    }

    /// A still picture, to every still hook.
    pub fn emit_still(&self, clip: usize, index: u64, seconds: f64, frame: &VideoFrame, from_video: bool) -> Result<()> {
        if !self.wants(Stage::Still) {
            return self.check();
        }
        let event = StillEvent { clip, index, seconds, frame: frame.clone(), from_video };
        self.dispatch(HookEvent::Still(event), |_, _| true)
    }

    /// One output, to the artifact hooks that accept its kind.
    pub fn emit_artifact(&self, event: ArtifactEvent) -> Result<()> {
        if !self.wants(Stage::Artifact) {
            return self.check();
        }
        let kind = event.kind;
        self.dispatch(HookEvent::Artifact(event), |_, e| e.artifact_kinds.contains(&kind))
    }

    /// The job made everything: waits for the background hooks, then runs the
    /// completed hooks. `Err` when a hook rejected the job along the way
    /// (a background one included) or rejects it here.
    pub fn emit_completed(&self, artifacts: usize) -> Result<()> {
        let Some(session) = self.session.clone() else { return Ok(()) };
        session.background.wait_idle();
        self.check()?;
        if self.wants(Stage::Completed) {
            let elapsed = session.started.elapsed();
            self.dispatch(HookEvent::Completed(CompletedEvent { artifacts, elapsed }), |_, _| true)?;
            session.background.wait_idle();
            self.check()?;
        }
        Ok(())
    }

    /// The job failed with `error`: waits for the background hooks, then runs
    /// the failed hooks (whose verdicts change nothing: the job has failed).
    /// Runs once per session however many paths report the failure.
    pub fn emit_failed(&self, error: &anyhow::Error) {
        let Some(session) = self.session.clone() else { return };
        if session.failed_reported.swap(true, Ordering::SeqCst) {
            return;
        }
        session.background.wait_idle();
        if !self.wants(Stage::Failed) {
            return;
        }
        let rejection = rejection_of(error).cloned().or_else(|| self.rejection());
        let event = HookEvent::Failed(FailedEvent { error: format!("{error:#}"), rejection });
        self.dispatch_unchecked(event, |_, _| true);
        session.background.wait_idle();
    }

    fn emit_sampled(&self, stage: Stage, clip: usize, index: u64, fps: f64, frame: &VideoFrame) -> Result<()> {
        if !self.wants(stage) {
            return self.check();
        }
        let Some(session) = self.session.clone() else { return Ok(()) };
        // Who takes this frame, decided (and counted against `max_frames`)
        // before the event is built, so a frame nobody wants costs nothing.
        let mut takers = vec![false; self.entries.len()];
        let mut any = false;
        for (i, e) in self.entries.iter().enumerate() {
            if !e.stages.contains(stage) || !e.sampling.selects(index, fps) {
                continue;
            }
            if let Some(max) = e.sampling.max_frames {
                if session.frame_counts[i].fetch_add(1, Ordering::Relaxed) >= max {
                    continue;
                }
            }
            takers[i] = true;
            any = true;
        }
        if !any {
            return self.check();
        }
        let seconds = index as f64 / if fps > 0.0 { fps } else { 30.0 };
        let event = FrameEvent { clip, index, seconds, frame: frame.clone() };
        let event = if stage == Stage::DecodedFrame { HookEvent::DecodedFrame(event) } else { HookEvent::EncoderFrame(event) };
        self.dispatch(event, |i, _| takers[i])
    }

    /// Run `job` under this session: its source already emitted by the
    /// caller, its artifacts and completed / failed hooks run here at the end,
    /// and any rejection — however the pipeline reported it — returned as the
    /// job's error.
    pub(crate) async fn run<T, F>(&self, artifacts_of: impl FnOnce(&T) -> Vec<ArtifactEvent>, job: F) -> Result<T>
    where
        F: std::future::Future<Output = Result<T>>,
    {
        let result = match job.await {
            // A pipeline can swallow a pump's error into a failed rung and
            // still return; the rejection is the job's answer regardless.
            Ok(out) => match self.check() {
                Ok(()) => {
                    // Built only when a hook wants them: an event holds the
                    // artifact's bytes.
                    let events = if self.wants(Stage::Artifact) { artifacts_of(&out) } else { Vec::new() };
                    let count = events.len();
                    let artifacts = self
                        .offload(move |h| events.into_iter().try_for_each(|e| h.emit_artifact(e)))
                        .await;
                    match artifacts {
                        Ok(()) => self.offload(move |h| h.emit_completed(count)).await.map(|_| out),
                        Err(e) => Err(e),
                    }
                }
                Err(e) => Err(e),
            },
            Err(e) => Err(e),
        };
        match result {
            Ok(out) => Ok(out),
            Err(e) => {
                let e = self.as_rejection(e);
                let hooks = self.clone();
                match tokio::task::spawn_blocking(move || {
                    hooks.emit_failed(&e);
                    e
                })
                .await
                {
                    Ok(e) => Err(e),
                    Err(join) => Err(anyhow::anyhow!("hook task: {join}")),
                }
            }
        }
    }

    /// [`run`](Self::run) for a synchronous job (the image job, which runs on
    /// a blocking thread already).
    #[cfg_attr(not(feature = "image"), allow(dead_code))]
    pub(crate) fn run_blocking<T>(
        &self,
        artifacts_of: impl FnOnce(&T) -> Vec<ArtifactEvent>,
        job: impl FnOnce() -> Result<T>,
    ) -> Result<T> {
        let result = job().and_then(|out| {
            self.check()?;
            let events = if self.wants(Stage::Artifact) { artifacts_of(&out) } else { Vec::new() };
            let count = events.len();
            events.into_iter().try_for_each(|e| self.emit_artifact(e))?;
            self.emit_completed(count)?;
            Ok(out)
        });
        result.map_err(|e| {
            let e = self.as_rejection(e);
            self.emit_failed(&e);
            e
        })
    }

    /// The rejection, rather than whatever the pipeline wrapped it in on its
    /// way out, when a hook stopped the job.
    fn as_rejection(&self, e: anyhow::Error) -> anyhow::Error {
        match (rejection_of(&e), self.rejection()) {
            (None, Some(r)) => anyhow::Error::new(r).context(format!("{e:#}")),
            _ => e,
        }
    }

    /// Run `f` on these hooks off the async runtime (hooks may block: a
    /// digest of a large source, an integration's network call).
    pub(crate) async fn offload(&self, f: impl FnOnce(&Hooks) -> Result<()> + Send + 'static) -> Result<()> {
        if self.is_empty() || self.session.is_none() {
            return Ok(());
        }
        let hooks = self.clone();
        tokio::task::spawn_blocking(move || f(&hooks)).await.map_err(|e| anyhow::anyhow!("hook task: {e}"))?
    }

    fn dispatch(&self, event: HookEvent, take: impl Fn(usize, &Entry) -> bool) -> Result<()> {
        self.check()?;
        self.dispatch_unchecked(event, take);
        self.check()
    }

    fn dispatch_unchecked(&self, event: HookEvent, take: impl Fn(usize, &Entry) -> bool) {
        let Some(session) = self.session.clone() else { return };
        let stage = event.stage();
        let event = Arc::new(event);
        for (i, entry) in self.entries.iter().enumerate() {
            if !entry.stages.contains(stage) || !take(i, entry) {
                continue;
            }
            match entry.policy.mode {
                HookMode::Blocking => {
                    run_one(&session, entry, &event, false);
                    // A blocking rejection stops the rest of this stage's
                    // hooks too: the job is over.
                    if stage != Stage::Failed && session.rejection.lock().unwrap().is_some() {
                        break;
                    }
                }
                HookMode::Background => {
                    let (session2, entry2, event2) = (Arc::clone(&session), entry.clone(), Arc::clone(&event));
                    session.background.submit(Box::new(move || run_one(&session2, &entry2, &event2, true)));
                }
            }
        }
    }
}

/// Run one hook on one event and record what it said.
fn run_one(session: &Session, entry: &Entry, event: &HookEvent, background: bool) {
    let ctx = HookContext { job_id: session.job_id.clone(), job_kind: session.kind, hook: entry.name.clone() };
    let stage = event.stage();
    let subject = event.subject();
    let started = Instant::now();
    let result = entry.hook.call(&ctx, event);
    let elapsed = started.elapsed();
    let (verdict, annotations, error) = match result {
        Ok(outcome) => (Some(outcome.verdict), outcome.annotations, None),
        Err(e) => {
            let message = format!("{e:#}");
            tracing::warn!(hook = %entry.name, kind = %entry.kind, error = %message, "hook failed");
            let verdict = match entry.policy.on_error {
                OnError::Continue => None,
                OnError::Reject => Some(Verdict::Reject { reason: format!("the hook failed: {message}") }),
            };
            (verdict, Vec::new(), Some(message))
        }
    };
    if let Some(Verdict::Reject { reason }) = &verdict {
        // Failed hooks cannot reject: the job is over either way.
        if stage != Stage::Failed {
            let mut slot = session.rejection.lock().unwrap();
            if slot.is_none() {
                tracing::warn!(hook = %entry.name, kind = %entry.kind, %subject, %reason, "hook rejected the job");
                *slot = Some(HookRejection {
                    hook: entry.name.clone(),
                    kind: entry.kind,
                    stage,
                    subject: subject.clone(),
                    reason: reason.clone(),
                });
            }
        }
    }
    session.records.lock().unwrap().push(HookRecord {
        hook: entry.name.clone(),
        kind: entry.kind,
        stage,
        subject,
        verdict,
        annotations,
        error,
        elapsed,
        background,
    });
}

// ---------------------------------------------------------------------------
// Sessions
// ---------------------------------------------------------------------------

struct Session {
    job_id: String,
    kind: JobKind,
    started: Instant,
    records: Mutex<Vec<HookRecord>>,
    rejection: Mutex<Option<HookRejection>>,
    /// Frames handed to each entry so far, for `max_frames`.
    frame_counts: Vec<AtomicU64>,
    failed_reported: AtomicBool,
    background: Background,
}

impl Session {
    fn new(job_id: String, kind: JobKind, entries: usize) -> Self {
        Self {
            job_id,
            kind,
            started: Instant::now(),
            records: Mutex::new(Vec::new()),
            rejection: Mutex::new(None),
            frame_counts: (0..entries).map(|_| AtomicU64::new(0)).collect(),
            failed_reported: AtomicBool::new(false),
            background: Background::default(),
        }
    }
}

type Task = Box<dyn FnOnce() + Send>;

/// The session's background worker: one thread, started on the first
/// background task, fed through a bounded queue — a hook slower than the
/// pipeline holds the pipeline up at the queue rather than piling frames up in
/// memory.
#[derive(Default)]
struct Background {
    tx: Mutex<Option<std::sync::mpsc::SyncSender<Task>>>,
    pending: Arc<(Mutex<usize>, Condvar)>,
}

/// Tasks queued before a background submit blocks.
const BACKGROUND_QUEUE: usize = 64;

impl Background {
    fn submit(&self, task: Task) {
        {
            let (count, _) = &*self.pending;
            *count.lock().unwrap() += 1;
        }
        let tx = {
            let mut slot = self.tx.lock().unwrap();
            slot.get_or_insert_with(|| {
                let (tx, rx) = std::sync::mpsc::sync_channel::<Task>(BACKGROUND_QUEUE);
                let pending = Arc::clone(&self.pending);
                std::thread::Builder::new()
                    .name("rivet-hooks".into())
                    .spawn(move || {
                        for task in rx {
                            // A panicking hook is that hook's failure, not the
                            // worker's: the rest of the queue still runs.
                            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(task));
                            let (count, cv) = &*pending;
                            let mut n = count.lock().unwrap();
                            *n = n.saturating_sub(1);
                            cv.notify_all();
                        }
                    })
                    .expect("spawning the hooks worker thread");
                tx
            })
            .clone()
        };
        if tx.send(task).is_err() {
            let (count, cv) = &*self.pending;
            let mut n = count.lock().unwrap();
            *n = n.saturating_sub(1);
            cv.notify_all();
        }
    }

    fn wait_idle(&self) {
        let (count, cv) = &*self.pending;
        let mut n = count.lock().unwrap();
        while *n > 0 {
            n = cv.wait(n).unwrap();
        }
    }
}

impl Drop for Background {
    fn drop(&mut self) {
        // Closing the queue ends the worker once it has drained.
        self.tx.lock().unwrap().take();
    }
}

fn generate_job_id() -> String {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let n = NEXT.fetch_add(1, Ordering::Relaxed);
    format!("{:016x}-{:04x}-{:x}", nanos, std::process::id() & 0xffff, n)
}

/// What the first bytes look like: an image format, a container, or `unknown`.
pub(crate) fn sniff_label(bytes: &[u8]) -> String {
    #[cfg(feature = "image")]
    if let Some(f) = crate::image::sniff(bytes) {
        return f.label().to_string();
    }
    let kind = container::sniff_container(bytes);
    if kind.is_known() { kind.label().to_string() } else { "unknown".to_string() }
}

// ---------------------------------------------------------------------------
// Reports
// ---------------------------------------------------------------------------

/// What one hook said about one event.
#[derive(Debug, Clone, PartialEq)]
pub struct HookRecord {
    pub hook: String,
    pub kind: HookKind,
    pub stage: Stage,
    pub subject: Subject,
    /// `None` when the hook errored and its policy is to carry on.
    pub verdict: Option<Verdict>,
    pub annotations: Vec<Annotation>,
    pub error: Option<String>,
    pub elapsed: Duration,
    pub background: bool,
}

impl HookRecord {
    pub fn to_json(&self) -> Value {
        let annotations: serde_json::Map<String, Value> =
            self.annotations.iter().map(|a| (a.key.clone(), a.value.clone())).collect();
        json!({
            "hook": self.hook,
            "kind": self.kind.as_str(),
            "stage": self.stage.as_str(),
            "subject": self.subject.to_json(),
            "verdict": self.verdict.as_ref().map(Verdict::as_str),
            "reason": match &self.verdict { Some(Verdict::Reject { reason }) => Some(reason.as_str()), _ => None },
            "annotations": annotations,
            "error": self.error,
            "elapsed_ms": self.elapsed.as_secs_f64() * 1000.0,
            "background": self.background,
        })
    }

    /// The annotation under `key`, if this record has one.
    pub fn annotation(&self, key: &str) -> Option<&Value> {
        self.annotations.iter().find(|a| a.key == key).map(|a| &a.value)
    }
}

/// Everything a job's hooks said.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct HookReport {
    pub job_id: Option<String>,
    /// In the order the hooks finished (not necessarily event order: frames
    /// decoded in ranges, and background hooks, finish when they finish).
    pub records: Vec<HookRecord>,
    /// When a hook stopped the job, which and why.
    pub rejection: Option<HookRejection>,
}

impl HookReport {
    pub fn is_empty(&self) -> bool {
        self.records.is_empty() && self.rejection.is_none()
    }

    pub fn is_rejected(&self) -> bool {
        self.rejection.is_some()
    }

    /// Every record made by the hook named `hook`.
    pub fn by_hook<'a>(&'a self, hook: &'a str) -> impl Iterator<Item = &'a HookRecord> + 'a {
        self.records.iter().filter(move |r| r.hook == hook)
    }

    /// Every record made by hooks of `kind`.
    pub fn by_kind(&self, kind: HookKind) -> impl Iterator<Item = &HookRecord> {
        self.records.iter().filter(move |r| r.kind == kind)
    }

    /// Every `(record, value)` with an annotation under `key`.
    pub fn annotations<'a>(&'a self, key: &'a str) -> impl Iterator<Item = (&'a HookRecord, &'a Value)> + 'a {
        self.records.iter().filter_map(move |r| r.annotation(key).map(|v| (r, v)))
    }

    /// Records with an error.
    pub fn errors(&self) -> impl Iterator<Item = &HookRecord> {
        self.records.iter().filter(|r| r.error.is_some())
    }

    pub fn to_json(&self) -> Value {
        json!({
            "job_id": self.job_id,
            "rejected": self.rejection.is_some(),
            "rejection": self.rejection.as_ref().map(HookRejection::to_json),
            "records": self.records.iter().map(HookRecord::to_json).collect::<Vec<_>>(),
        })
    }
}
