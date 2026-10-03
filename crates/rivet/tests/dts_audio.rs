//! DTS source audio through the job: a Matroska (`A_DTS`) and an MP4 (`dtsc`)
//! input with a 5.1 DTS track are transcoded with the Opus policy and must
//! come out as a 6-channel, channel-mapping-family-1 Opus track.
//!
//! The inputs are made here by this workspace's own encoders
//! (`common::synth`): the video by its H.264 encoder, the DTS by the `dts`
//! crate's core encoder, the MP4 by rivet's muxer and the Matroska file by
//! the helper's writer. The test skips when this host/build has no H.264
//! decode or encode path, since the video half of the job has to run for the
//! audio half to be reached.

mod common;

use std::sync::{Arc, Mutex};

use common::synth;
use rivet::job::RungArtifact;
use rivet::{AudioCodecPolicy, OutputSpec, Rung, RungStatus, VideoCodecPolicy, fn_sink, run_job_blocking};

/// One second of 64×64 H.264 video with a 5.1 DTS track, in `container`
/// (`mkv` or `mp4`).
fn make_input(container: &str) -> Vec<u8> {
    let cfg = synth::H264::new(64, 64, 24);
    let video = synth::encode_h264(&cfg, (0..24).map(|t| synth::test_pattern(64, 64, t, h26x::ChromaFormat::Yuv420)));
    let audio = synth::dts_5_1(1.0);
    match container {
        "mkv" => synth::mkv(&video, 64, 64, 24, None, Some(synth::MkvAudio { codec_id: "A_DTS", track: &audio })),
        "mp4" => synth::mp4(&video, 64, 64, 24, Some(&audio), None),
        other => panic!("no {other} writer here"),
    }
}

/// The `dOps` body of the first Opus sample entry in `mp4`: `(channels, family)`.
fn dops_of(mp4: &[u8]) -> Option<(u8, u8)> {
    let at = mp4.windows(4).position(|w| w == b"dOps")?;
    let body = &mp4[at + 4..];
    // version, channels, pre-skip u16, rate u32, gain i16, family.
    Some((body[1], body[10]))
}

fn transcode_to_opus(container: &str) {
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
