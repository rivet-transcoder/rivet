use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::Result;
use bytes::Bytes;
use codec::frame::{ColorSpace, PixelFormat, VideoFrame};

use super::*;

/// A source hook from a closure.
struct Src<F>(F);
impl<F: Fn(&SourceEvent) -> HookOutcome + Send + Sync> SourceHook for Src<F> {
    fn on_source(&self, _: &HookContext, e: &SourceEvent) -> Result<HookOutcome> {
        Ok((self.0)(e))
    }
}

/// Counts what it is handed, at whichever kind it is registered as.
#[derive(Clone, Default)]
struct Counter {
    seen: Arc<Mutex<Vec<(Stage, u64)>>>,
    sampling: Option<FrameSampling>,
}

impl Counter {
    fn with_sampling(sampling: FrameSampling) -> Self {
        Self { sampling: Some(sampling), ..Self::default() }
    }
    fn seen(&self) -> Vec<(Stage, u64)> {
        self.seen.lock().unwrap().clone()
    }
    fn push(&self, stage: Stage, index: u64) -> Result<HookOutcome> {
        self.seen.lock().unwrap().push((stage, index));
        Ok(HookOutcome::proceed())
    }
}

impl DecodedFrameHook for Counter {
    fn sampling(&self) -> FrameSampling {
        self.sampling.unwrap_or_default()
    }
    fn on_decoded_frame(&self, _: &HookContext, f: &FrameEvent) -> Result<HookOutcome> {
        self.push(Stage::DecodedFrame, f.index)
    }
}
impl EncoderFrameHook for Counter {
    fn sampling(&self) -> FrameSampling {
        self.sampling.unwrap_or_default()
    }
    fn on_encoder_frame(&self, _: &HookContext, f: &FrameEvent) -> Result<HookOutcome> {
        self.push(Stage::EncoderFrame, f.index)
    }
}
impl StillHook for Counter {
    fn on_still(&self, _: &HookContext, s: &StillEvent) -> Result<HookOutcome> {
        self.push(Stage::Still, s.index)
    }
}
impl SourceHook for Counter {
    fn on_source(&self, _: &HookContext, _: &SourceEvent) -> Result<HookOutcome> {
        self.push(Stage::Source, 0)
    }
}
impl ProbeHook for Counter {
    fn on_probe(&self, _: &HookContext, _: &ProbeEvent) -> Result<HookOutcome> {
        self.push(Stage::Probe, 0)
    }
}
impl ArtifactHook for Counter {
    fn on_artifact(&self, _: &HookContext, _: &ArtifactEvent) -> Result<HookOutcome> {
        self.push(Stage::Artifact, 0)
    }
}
impl CompletedHook for Counter {
    fn on_completed(&self, _: &HookContext, _: &CompletedEvent) -> Result<HookOutcome> {
        self.push(Stage::Completed, 0)
    }
}
impl FailedHook for Counter {
    fn on_failed(&self, _: &HookContext, _: &FailedEvent) -> Result<HookOutcome> {
        self.push(Stage::Failed, 0)
    }
}

/// A `w × h` 8-bit 4:2:0 frame whose luma is `f(x, y)`, chroma neutral.
fn yuv_frame(w: u32, h: u32, f: impl Fn(u32, u32) -> u8) -> VideoFrame {
    let mut data = Vec::with_capacity((w * h * 3 / 2) as usize);
    for y in 0..h {
        for x in 0..w {
            data.push(f(x, y));
        }
    }
    data.resize((w * h + 2 * (w / 2) * (h / 2)) as usize, 128);
    VideoFrame::new(Bytes::from(data), w, h, PixelFormat::Yuv420p, ColorSpace::Bt709, 0)
}

fn artifact(kind: ArtifactKind, label: &str) -> ArtifactEvent {
    ArtifactEvent {
        kind,
        label: label.into(),
        media_type: "application/octet-stream".into(),
        width: 0,
        height: 0,
        data: ArtifactData::Bytes(Bytes::from_static(b"out")),
    }
}

fn read_test_media(name: &str) -> Option<Bytes> {
    let dir = match std::env::var_os("RIVET_TEST_MEDIA") {
        Some(dir) => std::path::PathBuf::from(dir),
        None => std::path::Path::new(env!("CARGO_MANIFEST_DIR")).parent()?.parent()?.join("test_media"),
    };
    std::fs::read(dir.join(name)).ok().map(Bytes::from)
}

// -- stages, sampling --------------------------------------------------------

#[test]
fn stage_lists_parse() {
    let set = StageSet::parse("source, decoded-frame+encoder_frame").unwrap();
    assert!(set.contains(Stage::Source) && set.contains(Stage::DecodedFrame) && set.contains(Stage::EncoderFrame));
    assert!(!set.contains(Stage::Still));
    assert!(set.has_sampled());
    assert_eq!(StageSet::parse("all").unwrap(), StageSet::ALL);
    assert!(StageSet::ALL.iter().count() == Stage::ALL.len());
    assert!(StageSet::parse("frame").is_err(), "frames are named by where they are hooked");
    assert!(StageSet::parse("").is_err());
}

#[test]
fn interval_sampling_takes_the_first_frame_of_each_interval() {
    let s = FrameSampling::every_seconds(1.0);
    let picked: Vec<u64> = (0..100).filter(|&i| s.selects(i, 25.0)).collect();
    assert_eq!(picked, vec![0, 25, 50, 75]);
    // 300 frames at 29.97 are 10.01 s, the last starting at 9.98 s.
    assert_eq!((0..300).filter(|&i| s.selects(i, 29.97)).count(), 10);
    assert!(s.selects(300, 29.97));
}

#[test]
fn frame_stride_and_all() {
    let s = FrameSampling::every_frames(10);
    assert_eq!((0..35).filter(|&i| s.selects(i, 30.0)).collect::<Vec<_>>(), vec![0, 10, 20, 30]);
    assert!((0..10).all(|i| FrameSampling::all().selects(i, 30.0)));
}

// -- digests, perceptual hashes, frame views ---------------------------------

#[test]
fn digests_match_the_standard_vectors() {
    assert_eq!(
        DigestAlgorithm::Sha256.hex(b"abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    assert_eq!(DigestAlgorithm::Sha1.hex(b"abc"), "a9993e364706816aba3e25717850c26c9cd0d89d");
    assert_eq!(DigestAlgorithm::Md5.hex(b"abc"), "900150983cd24fb0d6963f7d28e17f72");
}

#[test]
fn perceptual_hashes_survive_scaling_and_tell_pictures_apart() {
    let picture = |w: u32, h: u32| {
        yuv_frame(w, h, move |x, y| {
            let (fx, fy) = (x as f32 / w as f32, y as f32 / h as f32);
            if fx > 0.6 && fy < 0.3 { 235 } else { (16.0 + 200.0 * (fx + fy) / 2.0) as u8 }
        })
    };
    let mirrored = |w: u32, h: u32| {
        let p = picture(w, h);
        let luma = frame::luma8(&p).unwrap();
        yuv_frame(w, h, move |x, y| luma[(y * w + (w - 1 - x)) as usize])
    };
    for algo in PerceptualAlgorithm::ALL {
        let big = algo.hash_frame(&picture(640, 360)).unwrap();
        let small = algo.hash_frame(&picture(160, 90)).unwrap();
        let other = algo.hash_frame(&mirrored(640, 360)).unwrap();
        assert!(phash::hamming(big, small) <= 6, "{algo}: {big:016x} vs scaled {small:016x}");
        assert!(phash::hamming(big, other) >= 12, "{algo}: {big:016x} vs mirrored {other:016x}");
    }
}

#[test]
fn hash_hex_round_trips() {
    let h = 0xd1c4_b2a3_9f8e_7d60u64;
    assert_eq!(phash::to_hex(h), "d1c4b2a39f8e7d60");
    assert_eq!(phash::from_hex("0xd1c4b2a39f8e7d60").unwrap(), h);
    assert!(phash::from_hex("not-hex").is_err());
}

#[test]
fn luma_and_rgb_views_of_each_layout() {
    let f = yuv_frame(4, 2, |x, _| if x < 2 { 16 } else { 235 });
    assert_eq!(frame::luma8(&f).unwrap(), vec![16, 16, 235, 235, 16, 16, 235, 235]);
    let rgb = frame::rgb8(&f).unwrap();
    assert_eq!(&rgb[..3], &[0, 0, 0]);
    assert_eq!(&rgb[6..9], &[255, 255, 255]);

    let mut data = Vec::new();
    for v in [64u16, 940, 64, 940, 512, 512] {
        data.extend_from_slice(&v.to_le_bytes());
    }
    let f10 = VideoFrame::new(Bytes::from(data), 2, 2, PixelFormat::Yuv420p10le, ColorSpace::Bt709, 0);
    assert_eq!(frame::luma8(&f10).unwrap(), vec![16, 235, 16, 235]);

    let rgba = VideoFrame::new(
        Bytes::from(vec![255, 0, 0, 255, 0, 255, 0, 255]),
        2,
        1,
        PixelFormat::Rgba32,
        ColorSpace::Bt709,
        0,
    );
    assert_eq!(frame::rgb8(&rgba).unwrap(), vec![255, 0, 0, 0, 255, 0]);
    assert_eq!(frame::luma8(&rgba).unwrap(), vec![77, 149]);

    let ppm = frame::encode(&f, FrameFormat::Ppm).unwrap();
    assert!(ppm.starts_with(b"P6\n4 2\n255\n"));
    assert_eq!(ppm.len(), b"P6\n4 2\n255\n".len() + 4 * 2 * 3);
    assert!(frame::encode(&f, FrameFormat::Pgm).unwrap().starts_with(b"P5\n4 2\n255\n"));
    assert!(frame::luma8(&VideoFrame::new(Bytes::new(), 4, 4, PixelFormat::Yuv420p, ColorSpace::Bt709, 0)).is_err());
}

// -- kinds: each hooks in at its own point only ------------------------------

#[test]
fn each_kind_is_handed_only_its_own_events() {
    let [src, probe, still, art, done, fail] = std::array::from_fn(|_| Counter::default());
    let dec = Counter::with_sampling(FrameSampling::all());
    let enc = Counter::with_sampling(FrameSampling::all());
    let hooks = Hooks::new()
        .source("src", src.clone())
        .probe("probe", probe.clone())
        .decoded_frames("dec", dec.clone())
        .encoder_frames("enc", enc.clone())
        .stills("still", still.clone())
        .artifacts("art", art.clone())
        .completed("done", done.clone())
        .failed("fail", fail.clone())
        .session("j", JobKind::Transcode);
    let f = yuv_frame(16, 16, |_, _| 90);
    hooks.emit_source(0, &Bytes::from_static(b"x")).unwrap();
    hooks.emit_probe(0, MediaSummary::default()).unwrap();
    hooks.emit_decoded_frame(0, 3, 30.0, &f).unwrap();
    hooks.emit_encoder_frame(0, 4, 30.0, &f).unwrap();
    hooks.emit_still(0, 5, 0.0, &f, false).unwrap();
    hooks.emit_artifact(artifact(ArtifactKind::Video, "720p")).unwrap();
    hooks.emit_completed(1).unwrap();

    assert_eq!(src.seen(), vec![(Stage::Source, 0)]);
    assert_eq!(probe.seen(), vec![(Stage::Probe, 0)]);
    assert_eq!(dec.seen(), vec![(Stage::DecodedFrame, 3)]);
    assert_eq!(enc.seen(), vec![(Stage::EncoderFrame, 4)]);
    assert_eq!(still.seen(), vec![(Stage::Still, 5)]);
    assert_eq!(art.seen(), vec![(Stage::Artifact, 0)]);
    assert_eq!(done.seen(), vec![(Stage::Completed, 0)]);
    assert!(fail.seen().is_empty());

    // The report and the listing say which kind each hook is.
    let report = hooks.report();
    assert_eq!(report.by_kind(HookKind::DecodedFrame).count(), 1);
    assert_eq!(report.by_hook("enc").next().unwrap().kind, HookKind::EncoderFrame);
    let listed = hooks.describe();
    let kinds: Vec<&str> = listed.iter().map(|h| h["kind"].as_str().unwrap()).collect();
    assert_eq!(
        kinds,
        ["source", "probe", "decoded-frame", "encoder-frame", "still", "artifact", "completed", "failed"]
    );
    assert_eq!(listed[0]["stages"], serde_json::json!(["source"]));
    assert!(listed[0]["frames"].is_null());
    assert_eq!(listed[2]["frames"]["every_seconds"], serde_json::Value::Null);
}

#[test]
fn artifact_hooks_get_only_the_kinds_they_accept() {
    let hooks = Hooks::new()
        .artifacts("video-only", ArtifactDigest::new(&[DigestAlgorithm::Md5]).kinds(&[ArtifactKind::Video]))
        .artifacts("everything", ArtifactDigest::new(&[DigestAlgorithm::Md5]))
        .session("j", JobKind::Transcode);
    hooks.emit_artifact(artifact(ArtifactKind::Video, "720p")).unwrap();
    hooks.emit_artifact(artifact(ArtifactKind::Playlist, "master")).unwrap();
    let r = hooks.report();
    assert_eq!(r.by_hook("video-only").count(), 1);
    assert_eq!(r.by_hook("everything").count(), 2);
    assert_eq!(hooks.describe()[0]["artifact_kinds"], serde_json::json!(["video"]));
}

// -- sessions, verdicts, policies --------------------------------------------

#[test]
fn a_rejection_is_the_error_and_is_reported() {
    let hooks = Hooks::new()
        .source("annotate", Src(|e: &SourceEvent| HookOutcome::proceed().annotate("bytes", e.bytes.len())))
        .source("gate", Src(|_: &SourceEvent| HookOutcome::reject("not today")))
        .source("after", Src(|_: &SourceEvent| -> HookOutcome { panic!("a blocking rejection stops the stage") }))
        .session("job-1", JobKind::Transcode);
    let err = hooks.emit_source(0, &Bytes::from_static(b"hello")).unwrap_err();
    let rejection = rejection_of(&err).expect("the error is the rejection");
    assert_eq!((rejection.hook.as_str(), rejection.kind, rejection.stage), ("gate", HookKind::Source, Stage::Source));
    assert_eq!(rejection.reason, "not today");
    assert!(err.to_string().contains("rejected by source hook `gate`"));

    let report = hooks.report();
    assert_eq!(report.job_id.as_deref(), Some("job-1"));
    assert!(report.is_rejected());
    assert_eq!(report.records.len(), 2);
    assert_eq!(report.annotations("bytes").next().unwrap().1, &serde_json::json!(5));
    // Every later stage stops on the same rejection.
    assert!(rejection_of(&hooks.emit_probe(0, MediaSummary::default()).unwrap_err()).is_some());
    assert_eq!(report.to_json()["rejection"]["kind"], "source");
}

#[test]
fn a_failing_hook_fails_open_or_closed_as_configured() {
    struct Broken;
    impl SourceHook for Broken {
        fn on_source(&self, _: &HookContext, _: &SourceEvent) -> Result<HookOutcome> {
            anyhow::bail!("unreachable service")
        }
    }
    let open = Hooks::new().source("broken", Broken).session("j", JobKind::Transcode);
    open.emit_source(0, &Bytes::from_static(b"x")).unwrap();
    assert_eq!(open.report().errors().count(), 1);
    assert!(!open.report().is_rejected());

    let closed = Hooks::new().source_with("broken", Broken, HookPolicy::default().fail_closed()).session("j", JobKind::Transcode);
    let err = closed.emit_source(0, &Bytes::from_static(b"x")).unwrap_err();
    assert!(rejection_of(&err).unwrap().reason.contains("unreachable service"));
}

#[test]
fn a_background_rejection_stops_the_job_by_completion() {
    let slow = Src(|_: &SourceEvent| {
        std::thread::sleep(std::time::Duration::from_millis(50));
        HookOutcome::reject("found later")
    });
    let hooks = Hooks::new().source_with("slow", slow, HookPolicy::background()).session("j", JobKind::Transcode);
    // The source stage does not wait for it ...
    hooks.emit_source(0, &Bytes::from_static(b"x")).unwrap();
    // ... completion does, and fails on it.
    let err = hooks.emit_completed(0).unwrap_err();
    assert_eq!(rejection_of(&err).unwrap().reason, "found later");
    assert!(hooks.report().records[0].background);
}

#[test]
fn failed_hooks_run_once_with_the_rejection() {
    struct OnFail(Arc<AtomicUsize>);
    impl FailedHook for OnFail {
        fn on_failed(&self, _: &HookContext, e: &FailedEvent) -> Result<HookOutcome> {
            assert_eq!(e.rejection.as_ref().unwrap().hook, "gate");
            self.0.fetch_add(1, Ordering::SeqCst);
            Ok(HookOutcome::proceed())
        }
    }
    let seen = Arc::new(AtomicUsize::new(0));
    let hooks = Hooks::new()
        .source("gate", Src(|_: &SourceEvent| HookOutcome::reject("no")))
        .failed("on-failure", OnFail(Arc::clone(&seen)))
        .session("j", JobKind::Transcode);
    let err = hooks.emit_source(0, &Bytes::from_static(b"x")).unwrap_err();
    hooks.emit_failed(&err);
    hooks.emit_failed(&err);
    assert_eq!(seen.load(Ordering::SeqCst), 1);
}

#[test]
fn frame_kinds_sample_and_cap_independently() {
    let dec = Counter::with_sampling(FrameSampling::all().max_frames(3));
    let enc = Counter::with_sampling(FrameSampling::every_frames(5));
    let hooks = Hooks::new()
        .decoded_frames("dec", dec.clone())
        .encoder_frames("enc", enc.clone())
        .session("j", JobKind::Transcode);
    let f = yuv_frame(16, 16, |_, _| 100);
    for i in 0..20 {
        hooks.emit_decoded_frame(0, i, 30.0, &f).unwrap();
        hooks.emit_encoder_frame(0, i, 30.0, &f).unwrap();
    }
    assert_eq!(dec.seen().iter().map(|s| s.1).collect::<Vec<_>>(), vec![0, 1, 2]);
    assert_eq!(enc.seen().iter().map(|s| s.1).collect::<Vec<_>>(), vec![0, 5, 10, 15]);
    assert!(!hooks.frames_exhausted(), "the encoder-frame hook has no cap");
}

#[test]
fn select_runs_required_hooks_and_named_optional_ones() {
    let all = Hooks::new()
        .source("always", Src(|_: &SourceEvent| HookOutcome::proceed()))
        .source_with("extra", Src(|_: &SourceEvent| HookOutcome::proceed()), HookPolicy::default().optional());
    assert_eq!(all.select(&[]).unwrap().names(), vec!["always"]);
    assert_eq!(all.select(&["extra".into()]).unwrap().names(), vec!["always", "extra"]);
    assert!(all.select(&["typo".into()]).is_err());
    assert_eq!(all.describe()[1]["required"], false);
}

#[test]
fn no_hooks_and_no_session_cost_nothing() {
    let none = Hooks::default();
    none.emit_source(0, &Bytes::from_static(b"x")).unwrap();
    none.emit_completed(0).unwrap();
    assert!(none.report().is_empty());
    let unsessioned = Hooks::new().source("gate", Src(|_: &SourceEvent| HookOutcome::reject("no")));
    unsessioned.emit_source(0, &Bytes::from_static(b"x")).unwrap();
}

#[test]
fn builtin_hooks_record_digests_and_fingerprints_where_registered() {
    let fp = || PerceptualFingerprint::new(&PerceptualAlgorithm::ALL).sampling(FrameSampling::all());
    let hooks = Hooks::new()
        .source("source-digest", SourceDigest::new(&DigestAlgorithm::ALL))
        .decoded_frames("decoded-fp", fp())
        .stills("still-fp", fp())
        .session("j", JobKind::Transcode);
    let f = yuv_frame(64, 64, |x, y| ((x * 3 + y) % 256) as u8);
    hooks.emit_source(0, &Bytes::from_static(b"abc")).unwrap();
    hooks.emit_decoded_frame(0, 0, 30.0, &f).unwrap();
    hooks.emit_encoder_frame(0, 0, 30.0, &f).unwrap();
    hooks.emit_still(0, 0, 0.0, &f, false).unwrap();
    let r = hooks.report();
    assert_eq!(
        r.annotations("sha256").next().unwrap().1,
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    let stages: Vec<Stage> = r.annotations("phash").map(|(rec, _)| rec.stage).collect();
    assert_eq!(stages, vec![Stage::DecodedFrame, Stage::Still], "no encoder-frame fingerprint was registered");
    let (_, v) = r.annotations("dhash").next().unwrap();
    assert_eq!(v.as_str().unwrap().len(), 16);
}

// -- the pipeline ------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn a_source_rejection_stops_run_job_before_anything_is_parsed() {
    let hooks = Hooks::new()
        .source("gate", Src(|_: &SourceEvent| HookOutcome::reject("refused")))
        .session("j", JobKind::Transcode);
    let spec = crate::OutputSpec::default().with_hooks(hooks.clone());
    let err = crate::job::run_job(Bytes::from_static(b"not media"), &spec, None, Arc::new(crate::progress::NullSink))
        .await
        .unwrap_err();
    assert_eq!(rejection_of(&err).unwrap().reason, "refused");
    assert!(hooks.report().is_rejected());
}

#[tokio::test(flavor = "multi_thread")]
async fn probe_hooks_see_the_demuxed_source() {
    let Some(input) = read_test_media("bbb_h264_360p_short.mp4") else {
        eprintln!("test media missing; skipped");
        return;
    };
    struct Gate;
    impl ProbeHook for Gate {
        fn on_probe(&self, _: &HookContext, p: &ProbeEvent) -> Result<HookOutcome> {
            let m = &p.media;
            Ok(HookOutcome::reject(format!("{} {}x{}", m.video_codec.as_deref().unwrap_or("?"), m.width, m.height)))
        }
    }
    let hooks = Hooks::new().probe("probe-gate", Gate).session("j", JobKind::Transcode);
    let spec = crate::OutputSpec::single_file(vec![crate::Rung::new(640, 360)]).with_hooks(hooks.clone());
    let err = crate::job::run_job(input, &spec, None, Arc::new(crate::progress::NullSink)).await.unwrap_err();
    let r = rejection_of(&err).unwrap_or_else(|| panic!("not a rejection: {err:#}"));
    assert_eq!(r.kind, HookKind::Probe);
    assert!(r.reason.starts_with("h264 "), "{}", r.reason);
}

#[test]
fn the_decode_pump_feeds_decoded_and_encoder_frame_hooks_separately() {
    let Some(input) = read_test_media("bbb_h264_360p_short.mp4") else {
        eprintln!("test media missing; skipped");
        return;
    };
    struct EncoderFormat;
    impl EncoderFrameHook for EncoderFormat {
        fn sampling(&self) -> FrameSampling {
            FrameSampling::all().max_frames(5)
        }
        fn on_encoder_frame(&self, _: &HookContext, f: &FrameEvent) -> Result<HookOutcome> {
            assert_eq!(f.frame.format, PixelFormat::Yuv420p);
            Ok(HookOutcome::proceed())
        }
    }
    let hooks = Hooks::new()
        .decoded_frames("source-fp", PerceptualFingerprint::new(&[PerceptualAlgorithm::PHash]))
        .encoder_frames("encoder-format", EncoderFormat)
        .session("j", JobKind::Transcode);
    let demuxer = container::streaming::demux_streaming_shared(input.clone()).unwrap();
    let header = demuxer.header().clone();
    drop(demuxer);
    let spec = crate::OutputSpec::default().with_hooks(hooks.clone());
    let filters = Arc::new(codec::filter::FilterChain::prepare(&[]).unwrap());
    let cfg = crate::decode_pump::DecodePumpConfig::for_source(&header, &spec, filters, None);
    let rt = tokio::runtime::Runtime::new().unwrap();
    let (tx, mut rx) = tokio::sync::mpsc::channel(8);
    let drain = rt.spawn(async move { while rx.recv().await.is_some() {} });
    let frames = crate::decode_pump::run_shared_decode_pump_blocking(cfg, input, vec![tx], rt.handle().clone()).unwrap();
    rt.block_on(drain).unwrap();

    let report = hooks.report();
    let fps = header.info.frame_rate;
    let expected = (0..frames).filter(|&i| FrameSampling::default().selects(i, fps)).count();
    let hashed: Vec<&HookRecord> = report.by_hook("source-fp").collect();
    assert!(!hashed.is_empty());
    assert_eq!(hashed.len(), expected, "one hash a second of {frames} frames at {fps}");
    assert!(hashed.iter().all(|r| r.stage == Stage::DecodedFrame && r.annotation("phash").is_some()));
    assert_eq!(report.by_hook("encoder-format").count(), 5);
    assert!(report.errors().next().is_none());
}

#[cfg(feature = "image")]
#[test]
fn the_image_job_runs_its_kinds() {
    let mut png = Vec::new();
    let img = ::image::RgbaImage::from_fn(48, 32, |x, y| ::image::Rgba([(x * 5) as u8, (y * 7) as u8, 90, 255]));
    ::image::DynamicImage::ImageRgba8(img)
        .write_to(&mut std::io::Cursor::new(&mut png), ::image::ImageFormat::Png)
        .unwrap();
    let all = Counter::default();
    let hooks = Hooks::new()
        .source("src", all.clone())
        .probe("probe", all.clone())
        .decoded_frames("dec", all.clone())
        .stills("still", all.clone())
        .artifacts("art", all.clone())
        .completed("done", all.clone())
        .source("source-digest", SourceDigest::new(&[DigestAlgorithm::Sha256]))
        .artifacts("artifact-digest", ArtifactDigest::new(&[DigestAlgorithm::Sha256]).kinds(&[ArtifactKind::Image]));
    let spec = crate::image::ImageSpec { formats: vec![crate::image::ImageFormat::Png], ..Default::default() };
    let out = crate::image::run_image_job_with_hooks(&Bytes::from(png), &spec, &hooks).unwrap();
    let stages: Vec<Stage> = all.seen().into_iter().map(|s| s.0).collect();
    // An image is a still, not a decoded video frame.
    assert_eq!(stages, vec![Stage::Source, Stage::Probe, Stage::Still, Stage::Artifact, Stage::Completed]);
    assert_eq!(out.hooks.annotations("sha256").count(), 2, "the source and the one image");
}
