//! Constant-rate (`rate=cbr`) encodes on an Intel GPU (QSV), read back from
//! the files the job wrote: for AV1, H.264 and H.265, the average rate is
//! within ±10% of the target, and no one-second window of the stream spends
//! more than the rate plus the buffer it declares (with a small margin);
//! and an HLS rendition coded at a constant rate declares that rate, plus
//! the audio's, as its BANDWIDTH.
//!
//! ffmpeg makes the input and is not a dependency of rivet: the test skips
//! with a message when it is not on PATH, and when this build or host has
//! no Intel encoder (build with `--features qsv`). The Intel GPU CI job sets
//! `RIVET_REQUIRE_QSV=1`, which turns every skip into a failure, and
//! `TRANSCODE_ENCODER_BACKEND=qsv`, which pins the serial single-file
//! encoder to QSV.

use std::path::Path;
use std::process::Command;
use std::sync::{Arc, Mutex};

use rivet::codec::encode::tuning::{EncodeOverrides, RateMode};
use rivet::{EncodePolicy, GpuFamily, OutputSpec, Quality, Rung, RungArtifact, RungStatus, VideoCodecPolicy, fn_sink, run_job_blocking};

const FPS: u32 = 30;
const TARGET: u32 = 2_000_000;
const BUFFER_MS: u32 = 1000;

fn required() -> bool {
    std::env::var("RIVET_REQUIRE_QSV").is_ok_and(|v| v == "1")
}

/// Skip, or fail when the Intel GPU job requires the test to run.
fn skip(why: &str) {
    assert!(!required(), "cbr_rates: RIVET_REQUIRE_QSV=1 but: {why}");
    eprintln!("cbr_rates: SKIP, {why}");
}

fn ffmpeg_available() -> bool {
    Command::new("ffmpeg").arg("-version").output().map(|o| o.status.success()).unwrap_or(false)
}

/// Six seconds of noisy 1280x720 at 30 fps — content that wants more than
/// the target, so the rate controller is the thing holding the rate — with
/// a stereo AAC track.
fn make_input(dir: &Path) -> Vec<u8> {
    let path = dir.join("in.mp4");
    let out = Command::new("ffmpeg")
        .args(["-hide_banner", "-loglevel", "error", "-y"])
        .args(["-f", "lavfi", "-i", &format!("testsrc2=size=1280x720:rate={FPS}:duration=6,noise=alls=10:allf=t+u")])
        .args(["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=6"])
        .args(["-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-pix_fmt", "yuv420p"])
        .args(["-c:a", "aac", "-b:a", "128k", "-ac", "2"])
        .arg(&path)
        .output()
        .expect("spawn ffmpeg");
    assert!(out.status.success(), "ffmpeg: {}", String::from_utf8_lossy(&out.stderr));
    std::fs::read(&path).unwrap()
}

fn cbr() -> Quality {
    Quality::default().with_overrides(EncodeOverrides {
        rate_mode: Some(RateMode::Constant),
        bitrate: Some(TARGET),
        buffer_ms: Some(BUFFER_MS),
        ..Default::default()
    })
}

/// Run `spec`; `None` (after a skip) when this host or build cannot encode
/// it on the card.
fn run(input: &[u8], spec: &OutputSpec, out: Option<&Path>) -> Option<rivet::JobOutput> {
    let failures: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let sink = {
        let failures = Arc::clone(&failures);
        fn_sink(move |p| {
            if p.status == RungStatus::Failed {
                failures.lock().unwrap().push(p.message.unwrap_or_default());
            }
        })
    };
    match run_job_blocking(input, spec, out, Arc::new(sink)) {
        Ok(o) if !o.rungs.is_empty() => Some(o),
        Ok(_) => {
            skip(&format!("no rung was produced: {}", failures.lock().unwrap().join(" | ")));
            None
        }
        Err(e) => {
            skip(&format!("no Intel encoder for this job: {e:#}; rungs: {}", failures.lock().unwrap().join(" | ")));
            None
        }
    }
}

/// Every video sample's size, in bytes, in decode order.
fn sample_sizes(mp4: &[u8]) -> Vec<usize> {
    let mut demux = rivet::container::streaming::demux_streaming_shared(bytes::Bytes::copy_from_slice(mp4)).expect("demux the output");
    let mut sizes = Vec::new();
    while let Some(s) = demux.next_video_sample().expect("sample") {
        sizes.push(s.data.len());
    }
    sizes
}

#[test]
fn a_constant_rate_holds_its_rate_on_an_intel_gpu() {
    if !ffmpeg_available() {
        skip("ffmpeg not on PATH");
        return;
    }
    let work = tempfile::tempdir().expect("temp dir");
    let input = make_input(work.path());
    let buffer_bits = f64::from(TARGET) * f64::from(BUFFER_MS) / 1000.0;
    for (policy, name) in [(VideoCodecPolicy::Av1, "AV1"), (VideoCodecPolicy::H264, "H.264"), (VideoCodecPolicy::H265, "H.265")] {
        // One encoder for the whole file, so the rate is held across it.
        let spec = OutputSpec::single_file(vec![Rung::new(1280, 720).with_quality(cbr())])
            .with_video_codec(policy)
            .encode_policy(EncodePolicy::SingleGpu(None));
        let Some(out) = run(&input, &spec, None) else { return };
        let RungArtifact::File(mp4) = &out.rungs[0].artifact else { panic!("{name}: a single file") };
        let sizes = sample_sizes(mp4);
        assert!(sizes.len() >= (FPS * 5) as usize, "{name}: {} frames", sizes.len());

        let seconds = sizes.len() as f64 / f64::from(FPS);
        let average = sizes.iter().sum::<usize>() as f64 * 8.0 / seconds;
        let achieved = average / f64::from(TARGET);
        let windows: Vec<f64> =
            sizes.windows(FPS as usize).map(|w| w.iter().sum::<usize>() as f64 * 8.0).collect();
        let peak = windows.iter().copied().fold(0.0, f64::max);
        let bound = f64::from(TARGET) + buffer_bits;
        eprintln!(
            "cbr_rates: {name}: {} frames, average {average:.0} bit/s ({achieved:.3} of {TARGET}), peak one-second \
             window {peak:.0} bits (bound {bound:.0})",
            sizes.len()
        );
        assert!((0.90..=1.10).contains(&achieved), "{name}: average {average:.0} bit/s against {TARGET} ({achieved:.3})");
        assert!(peak <= bound * 1.05, "{name}: a one-second window spent {peak:.0} bits, over the rate plus the buffer ({bound:.0})");
    }
}

/// The average bit rate of `segments`.
fn rates_avg(segments: &[(f64, u64)]) -> f64 {
    let (secs, bytes) = segments.iter().fold((0.0, 0u64), |(s, b), &(ss, bb)| (s + ss, b + bb));
    bytes as f64 * 8.0 / secs
}

/// `(seconds, bytes)` of every segment the media playlist `playlist` lists.
fn segments(playlist: &Path) -> Vec<(f64, u64)> {
    let dir = playlist.parent().unwrap();
    let playlist = std::fs::read_to_string(playlist).expect("media playlist");
    let lines: Vec<&str> = playlist.lines().collect();
    lines
        .iter()
        .enumerate()
        .filter_map(|(i, l)| {
            let secs: f64 = l.strip_prefix("#EXTINF:")?.trim_end_matches(',').parse().ok()?;
            let bytes = std::fs::metadata(dir.join(lines[i + 1].trim())).expect("segment file").len();
            Some((secs, bytes))
        })
        .collect()
}

#[test]
fn an_hls_constant_rate_rendition_declares_its_rate_plus_the_audio() {
    if !ffmpeg_available() {
        skip("ffmpeg not on PATH");
        return;
    }
    let work = tempfile::tempdir().expect("temp dir");
    let input = make_input(work.path());
    let spec = OutputSpec::hls(vec![Rung::new(1280, 720).with_quality(cbr())], 2.0)
        .with_video_codec(VideoCodecPolicy::H264)
        .encode_policy(EncodePolicy::Family(GpuFamily::Intel));
    let root = work.path().join("hls");
    if run(&input, &spec, Some(&root)).is_none() {
        return;
    }
    let master = std::fs::read_to_string(root.join("master.m3u8")).expect("master playlist");
    let audio = segments(&root.join("audio").join("audio.m3u8"));
    let audio_peak = audio.iter().map(|&(s, b)| b as f64 * 8.0 / s).fold(0.0, f64::max);
    let inf = master
        .lines()
        .take_while(|l| l.trim() != "video/720p/playlist.m3u8")
        .last()
        .expect("the rendition's STREAM-INF");
    let attr = |name: &str| -> f64 {
        inf.split([',', ':'])
            .find_map(|a| a.strip_prefix(&format!("{name}=")))
            .unwrap_or_else(|| panic!("{name} in {inf}"))
            .parse()
            .unwrap()
    };
    let (bandwidth, average) = (attr("BANDWIDTH"), attr("AVERAGE-BANDWIDTH"));
    let video_avg = rates_avg(&segments(&root.join("video").join("720p").join("playlist.m3u8")));
    let audio_avg = rates_avg(&audio);
    eprintln!("cbr_rates: HLS BANDWIDTH {bandwidth} AVERAGE-BANDWIDTH {average}; target {TARGET}, video average {video_avg:.0}, audio peak {audio_peak:.0}");
    assert!(average <= bandwidth, "AVERAGE-BANDWIDTH {average} over BANDWIDTH {bandwidth}");
    assert!(
        ((average - (video_avg + audio_avg)) / average).abs() <= 0.01,
        "AVERAGE-BANDWIDTH {average} is not the measured {video_avg:.0} + {audio_avg:.0}"
    );
    // BANDWIDTH is the declared rate plus the audio's peak — or, when the
    // card ran over its rate on average, that average: never below it.
    let floor = f64::from(TARGET) + audio_peak;
    let ceiling = f64::from(TARGET).max(video_avg) * 1.01 + audio_peak;
    assert!(
        bandwidth >= floor - 3.0 && bandwidth <= ceiling,
        "BANDWIDTH {bandwidth} is not the declared {TARGET} (or the {video_avg:.0} average) plus the audio's {audio_peak:.0}: {inf}"
    );
}
