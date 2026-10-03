//! An HLS bitrate ladder through the job, read back from the files it wrote:
//! every rung spends close to its target, every segment keeps to the coded
//! picture buffer it declares, and the master playlist's BANDWIDTH and
//! AVERAGE-BANDWIDTH are the video rendition's peak and average segment rates
//! plus the audio rendition's (RFC 8216 §4.3.4.2).
//!
//! The input (H.264 + AAC) is made here by this workspace's own encoders
//! (`common::synth`). The test skips when the build has no software H.264
//! encoder to code a bitrate with, or when this host's encode pool is cards,
//! which a bitrate job refuses by name — so it runs under `h26x-fallback` on
//! a host (or build) without a card.

mod common;

use std::path::Path;
use std::sync::{Arc, Mutex};

use rivet::codec::encode::tuning::{EncodeOverrides, RungPolicy};
use rivet::{OutputSpec, Quality, Rung, RungStatus, VideoCodecPolicy, fn_sink, run_job_blocking};

/// Eight seconds of 320x180 H.264 at 24 fps with a stereo AAC track: the
/// test pattern with noise over it, coded at a fine fixed quantiser, so
/// every rung has more detail than its rate buys and the rate controller is
/// what holds it.
fn make_input() -> Vec<u8> {
    common::synth::clip(320, 180, 24, 8.0, 8, 0, true)
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

/// `(average, peak)` bit rate of `segments`, the way the master is measured.
fn rates(segments: &[(f64, u64)]) -> (f64, f64) {
    let (secs, bytes): (f64, u64) = segments.iter().fold((0.0, 0), |(s, b), &(ss, bb)| (s + ss, b + bb));
    let peak = segments.iter().map(|&(s, b)| b as f64 * 8.0 / s).fold(0.0, f64::max);
    (bytes as f64 * 8.0 / secs, peak)
}

/// The `mdat` payload of one CMAF segment as Annex B: its samples are
/// four-byte length-prefixed NAL units, parameter sets in band too.
fn segment_annexb(segment: &[u8]) -> Vec<u8> {
    let mut at = 0;
    while at + 8 <= segment.len() {
        let size = u32::from_be_bytes(segment[at..at + 4].try_into().unwrap()) as usize;
        if &segment[at + 4..at + 8] == b"mdat" {
            let mut body = &segment[at + 8..at + size];
            let mut out = Vec::new();
            while body.len() >= 4 {
                let n = u32::from_be_bytes(body[..4].try_into().unwrap()) as usize;
                out.extend_from_slice(&[0, 0, 0, 1]);
                out.extend_from_slice(&body[4..4 + n]);
                body = &body[4 + n..];
            }
            return out;
        }
        at += size.max(8);
    }
    panic!("no mdat in the segment");
}

/// `(BANDWIDTH, AVERAGE-BANDWIDTH)` the master declares for `uri`.
fn declared(master: &str, uri: &str) -> (f64, f64) {
    let lines: Vec<&str> = master.lines().collect();
    let at = lines.iter().position(|l| l.trim() == uri).expect("variant in the master");
    let inf = lines[at - 1];
    let attr = |name: &str| -> f64 {
        let from = inf.find(&format!("{name}=")).expect("attribute") + name.len() + 1;
        inf[from..].split(',').next().unwrap().parse().unwrap()
    };
    (attr("BANDWIDTH"), attr("AVERAGE-BANDWIDTH"))
}

#[test]
fn an_hls_bitrate_ladder_spends_its_targets_and_declares_its_renditions() {
    let work = tempfile::tempdir().expect("temp dir");
    let input = make_input();
    let rate = |bps: u32| Quality::default().with_overrides(EncodeOverrides { bitrate: Some(bps), ..Default::default() });
    let targets = [(320, 180, 400_000u32), (160, 90, 120_000)];
    let rungs = targets.iter().map(|&(w, h, bps)| Rung::new(w, h).with_quality(rate(bps))).collect();
    let one_second = EncodeOverrides { buffer_ms: Some(1000), ..Default::default() };
    let spec = OutputSpec::hls(rungs, 2.0)
        .with_video_codec(VideoCodecPolicy::H264)
        .with_rung_policy(RungPolicy::new().with_global(one_second));
    let root = work.path().join("hls");
    let failures: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let sink = {
        let failures = Arc::clone(&failures);
        fn_sink(move |p| {
            if p.status == RungStatus::Failed {
                failures.lock().unwrap().push(p.message.unwrap_or_default());
            }
        })
    };
    if let Err(e) = run_job_blocking(&input, &spec, Some(&root), Arc::new(sink)) {
        let msg = format!("{e:#}; rungs: {}", failures.lock().unwrap().join(" | "));
        // No software H.264 encoder in this build, or a pool of cards that
        // a bitrate job is refused on: nothing here codes a rate.
        assert!(
            msg.contains("no encoder matches") || msg.contains("is coded to a bitrate"),
            "the bitrate ladder failed: {msg}"
        );
        eprintln!("hls_rates: SKIP, no software H.264 pool on this host/build: {msg}");
        return;
    }

    let master = std::fs::read_to_string(root.join("master.m3u8")).expect("master playlist");
    let audio = segments(&root.join("audio").join("audio.m3u8"));
    let (audio_avg, audio_peak) = rates(&audio);
    assert!(audio_avg > 0.0, "the source's audio has a rendition");
    for (w, h, target) in targets {
        let label = format!("{}p", w.min(h));
        let dir = root.join("video").join(&label);
        let video = segments(&dir.join("playlist.m3u8"));
        let (avg, peak) = rates(&video);
        let achieved = avg / f64::from(target);
        assert!((0.85..=1.15).contains(&achieved), "{label}: {avg:.0} bit/s against {target} ({achieved:.3})");

        // Every segment is a stream of its own and keeps to the buffer it
        // declares, at the rate it was asked for (snapped down to the
        // syntax's 64 bit/s unit).
        let names: Vec<String> = std::fs::read_to_string(dir.join("playlist.m3u8"))
            .unwrap()
            .lines()
            .filter(|l| l.ends_with(".m4s"))
            .map(str::to_string)
            .collect();
        assert_eq!(names.len(), video.len());
        for name in &names {
            let annexb = segment_annexb(&std::fs::read(dir.join(name)).unwrap());
            let report = h26x::encode::hrd::verify(&annexb).unwrap_or_else(|e| panic!("{label}/{name}: {e}"));
            assert_eq!(report.bit_rate, u64::from(target) / 64 * 64, "{label}/{name}: declared rate");
            assert!(report.conforms(), "{label}/{name}: {report:?}");
        }

        // The master: the video's own rates plus the audio rendition's,
        // peak with peak and average with average.
        let (bandwidth, average) = declared(&master, &format!("video/{label}/playlist.m3u8"));
        let close = |got: f64, want: f64| (got - want).abs() <= (want * 0.001).max(3.0);
        assert!(close(bandwidth, peak + audio_peak), "{label}: BANDWIDTH {bandwidth} vs {peak:.0} + {audio_peak:.0}");
        assert!(close(average, avg + audio_avg), "{label}: AVERAGE-BANDWIDTH {average} vs {avg:.0} + {audio_avg:.0}");
        assert!(average < bandwidth, "{label}: the average is not the peak");
    }
}
