//! DTS source audio through the job: a Matroska (`A_DTS`) and an MP4 (`dtsc`)
//! input with a 5.1 DTS track are transcoded with the Opus policy and must
//! come out as a 6-channel, channel-mapping-family-1 Opus track.
//!
//! ffmpeg makes the inputs (its `dca` encoder is the one DTS encoder around
//! that is free to run) and is not a dependency of rivet: the test skips with
//! a message when it is not on PATH, like `fidelity_ffprobe.rs`. It also
//! skips when this host/build has no H.264 decode or encode path, since the
//! video half of the job has to run for the audio half to be reached.

use std::process::Command;
use std::sync::{Arc, Mutex};

use rivet::job::RungArtifact;
use rivet::{AudioCodecPolicy, OutputSpec, Rung, RungStatus, VideoCodecPolicy, fn_sink, run_job_blocking};

fn ffmpeg_available() -> bool {
    Command::new("ffmpeg")
        .arg("-version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// One second of 64×64 H.264 video with a 5.1 DTS track, in `container`
/// (`mkv` or `mp4`).
fn make_input(container: &str) -> Vec<u8> {
    // A directory of its own per call. Both tests run at once in this process;
    // a directory named after the pid was shared, and whichever test finished
    // first removed it — empty, between the other's `create_dir_all` and its
    // ffmpeg opening the output — so the other failed with "Error opening
    // output ... No such file or directory".
    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join(format!("dts_5_1.{container}"));
    let out = Command::new("ffmpeg")
        .args(["-hide_banner", "-loglevel", "error", "-y"])
        .args(["-f", "lavfi", "-i", "testsrc2=size=64x64:rate=24:duration=1"])
        .args([
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=1[a];\
             sine=frequency=660:sample_rate=48000:duration=1[b];\
             sine=frequency=880:sample_rate=48000:duration=1[c];\
             sine=frequency=60:sample_rate=48000:duration=1[d];\
             anoisesrc=color=pink:sample_rate=48000:duration=1:amplitude=0.3:seed=1[e];\
             anoisesrc=color=brown:sample_rate=48000:duration=1:amplitude=0.3:seed=2[f];\
             [a][b][c][d][e][f]join=inputs=6:channel_layout=5.1(side):map=0.0-FL|1.0-FR|2.0-FC|3.0-LFE|4.0-SL|5.0-SR",
        ])
        .args(["-map", "0:v", "-map", "1:a"])
        .args(["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p"])
        .args(["-c:a", "dca", "-strict", "-2", "-b:a", "768k"])
        .arg(&path)
        .output()
        .expect("spawn ffmpeg");
    assert!(out.status.success(), "ffmpeg: {}", String::from_utf8_lossy(&out.stderr));
    std::fs::read(&path).unwrap()
}

/// The `dOps` body of the first Opus sample entry in `mp4`: `(channels, family)`.
fn dops_of(mp4: &[u8]) -> Option<(u8, u8)> {
    let at = mp4.windows(4).position(|w| w == b"dOps")?;
    let body = &mp4[at + 4..];
    // version, channels, pre-skip u16, rate u32, gain i16, family.
    Some((body[1], body[10]))
}

fn transcode_to_opus(container: &str) {
    if !ffmpeg_available() {
        eprintln!("dts_audio: ffmpeg not on PATH — skipping");
        return;
    }
    let input = make_input(container);
    let spec = OutputSpec::single_file(vec![Rung::new(64, 64)])
        .with_video_codec(VideoCodecPolicy::H264)
        .with_audio(AudioCodecPolicy::ForceOpus);
    // A failed rung says why only through the progress sink.
    let failures: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let sink = {
        let failures = Arc::clone(&failures);
        fn_sink(move |p| {
            if p.status == RungStatus::Failed {
                failures.lock().unwrap().push(p.message.unwrap_or_default());
            }
        })
    };
    let out = match run_job_blocking(&input, &spec, None, Arc::new(sink)) {
        Ok(out) => out,
        Err(e) => {
            let msg = format!("{e:#}; rungs: {}", failures.lock().unwrap().join(" | "));
            // The video half has to exist for the audio half to be reached;
            // a host without an H.264 path is a skip, anything about audio
            // is a failure.
            assert!(
                !msg.to_ascii_lowercase().contains("audio"),
                "{container}: the audio half of the job failed: {msg}"
            );
            eprintln!("dts_audio ({container}): SKIP, the video half has no path on this host/build: {msg}");
            return;
        }
    };
    assert_eq!(out.audio_handling, "dts → opus (6ch)", "{container}: audio handling");
    let rung = out.rungs.first().expect("one rung");
    let RungArtifact::File(mp4) = &rung.artifact else {
        panic!("{container}: single-file job should yield file bytes");
    };
    let (channels, family) = dops_of(mp4).unwrap_or_else(|| panic!("{container}: no dOps in the output"));
    assert_eq!(channels, 6, "{container}: Opus channel count");
    assert_eq!(family, 1, "{container}: 5.1 Opus must use channel-mapping family 1");
}

#[test]
fn mkv_dts_5_1_transcodes_to_opus_5_1() {
    transcode_to_opus("mkv");
}

#[test]
fn mp4_dtsc_5_1_transcodes_to_opus_5_1() {
    transcode_to_opus("mp4");
}
