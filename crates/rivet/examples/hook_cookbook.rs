//! The hook cookbook: one small, working hook per recipe in
//! `docs/hooks-cookbook.md`. Compiled with the crate's examples so the recipes
//! cannot drift from the API.
//!
//! ```text
//! cargo run --example hook_cookbook -- input.mp4      # a video job
//! cargo run --example hook_cookbook --features image -- photo.jpg   # an image job
//! ```

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, mpsc};

use anyhow::{Context, Result};
use rivet::hooks::{
    ArtifactData, ArtifactDigest, ArtifactEvent, ArtifactHook, ArtifactKind, CompletedEvent, CompletedHook,
    DecodedFrameHook, DigestAlgorithm, EncoderFrameHook, FailedEvent, FailedHook, FrameEvent, FrameSampling,
    HookContext, HookOutcome, HookPolicy, Hooks, LogHook, PerceptualAlgorithm, PerceptualFingerprint, ProbeEvent,
    ProbeHook, SourceDigest, SourceEvent, SourceHook, StageSet, StillEvent, StillHook, phash,
};

// ---------------------------------------------------------------------------
// 1. Source: refuse sources over a size
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// 2. Source: accept only some containers
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// 3. Probe: limits on what the source is
// ---------------------------------------------------------------------------

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
            return Ok(HookOutcome::reject(format!(
                "{}x{} is over {}x{}",
                m.width, m.height, self.max_width, self.max_height
            )));
        }
        if m.duration > self.max_seconds {
            return Ok(HookOutcome::reject(format!("{:.0}s is over {:.0}s", m.duration, self.max_seconds)));
        }
        Ok(HookOutcome::proceed()
            .annotate("codec", m.video_codec.clone().unwrap_or_default())
            .annotate("duration", m.duration))
    }
}

// ---------------------------------------------------------------------------
// 4. Decoded frames: flag blank (near-black) frames in the source
// ---------------------------------------------------------------------------

pub struct BlankFrames {
    /// Mean 8-bit luma at or under which a frame counts as blank.
    pub threshold: u8,
    pub blank: AtomicU64,
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
            self.blank.fetch_add(1, Ordering::Relaxed);
            outcome = outcome.annotate("blank", true);
        }
        Ok(outcome)
    }
}

// ---------------------------------------------------------------------------
// 5. Decoded frames: your own hashing library
// ---------------------------------------------------------------------------

/// Stands in for a hashing library your project already has. Here: a 64-bit
/// FNV-1a of an 8×8 luma thumbnail.
pub mod my_hasher {
    pub fn compute(rgb: &[u8], width: u32, height: u32) -> u64 {
        let luma: Vec<u8> = rgb.chunks_exact(3).map(|p| ((p[0] as u32 + p[1] as u32 + p[2] as u32) / 3) as u8).collect();
        let small = rivet::hooks::phash::shrink(&luma, width as usize, height as usize, 8, 8);
        small.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &v| (h ^ v as u64).wrapping_mul(0x100_0000_01b3))
    }
}

pub struct MyHash;

impl MyHash {
    fn hash(&self, frame: &rivet::codec::frame::VideoFrame) -> Result<String> {
        let rgb = rivet::hooks::frame::rgb8(frame)?;
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
    fn describe(&self) -> String {
        "my hash of decoded frames".into()
    }
}

// The same type, at a different point: an image job's pictures.
impl StillHook for MyHash {
    fn on_still(&self, _ctx: &HookContext, s: &StillEvent) -> Result<HookOutcome> {
        Ok(HookOutcome::proceed().annotate("my_hash", self.hash(&s.frame)?))
    }
    fn describe(&self) -> String {
        "my hash of stills".into()
    }
}

// ---------------------------------------------------------------------------
// 6. Encoder frames: did the filters change the picture?
// ---------------------------------------------------------------------------

/// Keeps the pHash of each decoded frame and compares the encoder's frame to
/// it: the Hamming distance says how far the spec's crop / overlay / colour
/// filters moved the picture. One value, registered at both points through an
/// `Arc` (an `Arc` of any hook kind is that kind).
#[derive(Default)]
pub struct FilterDrift {
    decoded: Mutex<std::collections::HashMap<(usize, u64), u64>>,
}

impl DecodedFrameHook for FilterDrift {
    fn sampling(&self) -> FrameSampling {
        FrameSampling::every_seconds(5.0)
    }
    fn on_decoded_frame(&self, _ctx: &HookContext, f: &FrameEvent) -> Result<HookOutcome> {
        let h = PerceptualAlgorithm::PHash.hash_frame(&f.frame)?;
        self.decoded.lock().unwrap().insert((f.clip, f.index), h);
        Ok(HookOutcome::proceed())
    }
}

impl EncoderFrameHook for FilterDrift {
    fn sampling(&self) -> FrameSampling {
        FrameSampling::every_seconds(5.0)
    }
    fn on_encoder_frame(&self, _ctx: &HookContext, f: &FrameEvent) -> Result<HookOutcome> {
        let after = PerceptualAlgorithm::PHash.hash_frame(&f.frame)?;
        let before = self.decoded.lock().unwrap().get(&(f.clip, f.index)).copied();
        Ok(match before {
            Some(b) => HookOutcome::proceed().annotate("filter_drift_bits", phash::hamming(b, after)),
            None => HookOutcome::proceed(),
        })
    }
}

// ---------------------------------------------------------------------------
// 7. Any kind: forward events to your own system without blocking the job
// ---------------------------------------------------------------------------

/// What gets forwarded: job id, where, and the hash.
#[derive(Debug)]
pub struct Forwarded {
    pub job_id: String,
    pub clip: usize,
    pub index: u64,
    pub phash: String,
}

/// Sends each sampled source frame's pHash down a channel to a consumer of
/// your own (a queue publisher, a database writer). Registered in the
/// background so a slow consumer never stalls decoding.
pub struct Forward(pub Mutex<mpsc::Sender<Forwarded>>);

impl DecodedFrameHook for Forward {
    fn on_decoded_frame(&self, ctx: &HookContext, f: &FrameEvent) -> Result<HookOutcome> {
        let phash = phash::to_hex(PerceptualAlgorithm::PHash.hash_frame(&f.frame)?);
        self.0
            .lock()
            .unwrap()
            .send(Forwarded { job_id: ctx.job_id.clone(), clip: f.clip, index: f.index, phash })
            .context("the consumer is gone")?;
        Ok(HookOutcome::proceed())
    }
}

// ---------------------------------------------------------------------------
// 8. Artifacts: cap output size, and write a sidecar manifest
// ---------------------------------------------------------------------------

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

/// Appends one JSON line per output — label, type, size, SHA-256 — to a file.
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

// ---------------------------------------------------------------------------
// 9. Completed / failed: metrics
// ---------------------------------------------------------------------------

#[derive(Default)]
pub struct Metrics {
    pub completed: AtomicU64,
    pub failed: AtomicU64,
    pub rejected: AtomicU64,
}

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
            Some(r) => {
                self.0.rejected.fetch_add(1, Ordering::Relaxed);
                eprintln!("rejected by {} hook `{}`: {}", r.kind, r.hook, r.reason);
            }
            None => {
                self.0.failed.fetch_add(1, Ordering::Relaxed);
            }
        }
        Ok(HookOutcome::proceed())
    }
}

// ---------------------------------------------------------------------------
// Putting it together
// ---------------------------------------------------------------------------

/// Every recipe, registered at its point.
pub fn cookbook_hooks(metrics: Arc<Metrics>, forward: mpsc::Sender<Forwarded>, manifest: std::path::PathBuf) -> Hooks {
    let drift = Arc::new(FilterDrift::default());
    Hooks::new()
        // Source bytes
        .source("max-source-size", MaxSourceSize(4 << 30))
        .source("containers", AllowedContainers(&["mp4", "matroska", "mpegts", "jpeg", "png", "webp", "heic", "avif"]))
        .source("source-digest", SourceDigest::new(&[DigestAlgorithm::Sha256, DigestAlgorithm::Md5]))
        // Source description
        .probe(
            "source-limits",
            SourceLimits { max_width: 7680, max_height: 4320, max_seconds: 4.0 * 3600.0, require_video: false },
        )
        // Decoded source frames
        .decoded_frames(
            "source-fingerprint",
            PerceptualFingerprint::new(&[PerceptualAlgorithm::PHash]).sampling(FrameSampling::every_seconds(1.0)),
        )
        .decoded_frames("blank-frames", BlankFrames { threshold: 20, blank: AtomicU64::new(0) })
        .decoded_frames_with("my-hash", MyHash, HookPolicy::default().fail_closed())
        .decoded_frames_with("forward", Forward(Mutex::new(forward)), HookPolicy::background())
        .decoded_frames("filter-drift-before", Arc::clone(&drift))
        // Encoder frames
        .encoder_frames("filter-drift-after", drift)
        // Stills (image jobs)
        .stills("still-fingerprint", PerceptualFingerprint::new(&[PerceptualAlgorithm::PHash]))
        .stills_with("my-hash-stills", MyHash, HookPolicy::default().fail_closed())
        // Outputs
        .artifacts("max-output-size", MaxOutputSize(2 << 30))
        .artifacts("manifest", Manifest(manifest))
        .artifacts("output-digest", ArtifactDigest::new(&[DigestAlgorithm::Sha256]))
        // The end
        .completed("count-completed", CountCompleted(Arc::clone(&metrics)))
        .failed("count-failed", CountFailed(metrics))
}

/// A general hook for seeing every event while developing.
pub fn debug_hooks() -> Hooks {
    Hooks::new().with("log", LogHook { stages: StageSet::ALL, sampling: FrameSampling::every_seconds(10.0) })
}

fn main() -> Result<()> {
    let path = std::env::args().nth(1).context("usage: hook_cookbook <input>")?;
    let input = bytes::Bytes::from(std::fs::read(&path).with_context(|| format!("reading {path}"))?);

    let metrics = Arc::new(Metrics::default());
    let (tx, rx) = mpsc::channel();
    let consumer = std::thread::spawn(move || rx.into_iter().count());
    let manifest = std::env::temp_dir().join("rivet-hook-manifest.jsonl");
    let hooks = cookbook_hooks(Arc::clone(&metrics), tx, manifest.clone());

    #[cfg(feature = "image")]
    if rivet::image::sniff(&input).is_some() {
        let spec = rivet::image::ImageSpec { formats: vec![rivet::image::ImageFormat::Avif], ..Default::default() };
        let session = hooks.session("cookbook-image", rivet::hooks::JobKind::Image);
        let result = rivet::image::run_image_job_with_hooks(&input, &spec, &session);
        println!("{}", serde_json::to_string_pretty(&session.report().to_json())?);
        drop(session);
        drop(hooks);
        println!("forwarded {} frames; manifest at {}", consumer.join().unwrap(), manifest.display());
        return result.map(|_| ());
    }

    let info = rivet::probe_bytes(&input)?;
    let session = hooks.session("cookbook-video", rivet::hooks::JobKind::Transcode);
    let spec = rivet::OutputSpec::single_file(vec![rivet::Rung::new(info.width, info.height)]).with_hooks(session.clone());
    let result = rivet::run_job_blocking_owned(input, &spec, None, Arc::new(rivet::progress::NullSink));
    println!("{}", serde_json::to_string_pretty(&session.report().to_json())?);
    drop((spec, session, hooks));
    println!(
        "completed {} failed {} rejected {}; forwarded {} frames; manifest at {}",
        metrics.completed.load(Ordering::Relaxed),
        metrics.failed.load(Ordering::Relaxed),
        metrics.rejected.load(Ordering::Relaxed),
        consumer.join().unwrap(),
        manifest.display()
    );
    result.map(|_| ())
}

// ---------------------------------------------------------------------------
// 15. Unit-testing hooks without a job (`cargo test --example hook_cookbook`)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rivet::codec::frame::{ColorSpace, PixelFormat, VideoFrame};
    use rivet::hooks::{JobKind, rejection_of};

    fn grey(w: u32, h: u32, luma: u8) -> VideoFrame {
        let mut data = vec![luma; (w * h) as usize];
        data.resize((w * h * 3 / 2) as usize, 128);
        VideoFrame::new(bytes::Bytes::from(data), w, h, PixelFormat::Yuv420p, ColorSpace::Bt709, 0)
    }

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
        hooks.emit_decoded_frame(0, 0, 30.0, &grey(64, 64, 128)).unwrap();
        assert!(hooks.report().annotations("phash").next().is_some());
    }

    #[test]
    fn flags_blank_frames() {
        let blank = Arc::new(BlankFrames { threshold: 20, blank: AtomicU64::new(0) });
        let hooks = Hooks::new().decoded_frames("blank", Arc::clone(&blank)).session("test", JobKind::Transcode);
        hooks.emit_decoded_frame(0, 0, 2.0, &grey(32, 32, 16)).unwrap();
        hooks.emit_decoded_frame(0, 1, 2.0, &grey(32, 32, 200)).unwrap();
        assert_eq!(blank.blank.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn filter_drift_is_shared_across_two_points() {
        let drift = Arc::new(FilterDrift::default());
        let hooks = Hooks::new()
            .decoded_frames("before", Arc::clone(&drift))
            .encoder_frames("after", drift)
            .session("test", JobKind::Transcode);
        let f = grey(64, 64, 90);
        hooks.emit_decoded_frame(0, 0, 30.0, &f).unwrap();
        hooks.emit_encoder_frame(0, 0, 30.0, &f).unwrap();
        let report = hooks.report();
        let (_, bits) = report.annotations("filter_drift_bits").next().unwrap();
        assert_eq!(bits, 0, "the same picture at both points");
    }

    #[test]
    fn limits_and_metrics() {
        let metrics = Arc::new(Metrics::default());
        let hooks = Hooks::new()
            .probe("limits", SourceLimits { max_width: 1920, max_height: 1080, max_seconds: 60.0, require_video: true })
            .failed("count", CountFailed(Arc::clone(&metrics)))
            .session("test", JobKind::Transcode);
        let media = rivet::hooks::MediaSummary {
            video_codec: Some("h264".into()),
            width: 3840,
            height: 2160,
            ..Default::default()
        };
        let err = hooks.emit_probe(0, media).unwrap_err();
        hooks.emit_failed(&err);
        assert_eq!(metrics.rejected.load(Ordering::Relaxed), 1);
    }
}
