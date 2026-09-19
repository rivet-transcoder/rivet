use container::AudioInfo;

use super::audio::PreparedAudio;
use super::splice::{trim_audio, trim_frame};

#[test]
fn trim_frame_is_half_open_exact() {
    // `[start, end)` must be exact even at a non-integer detected fps: a frame
    // whose time is < end_sec is kept (ceil), regardless of rounding.
    // 29.9 fps: 7 s = frame 209.3, so the exclusive end is 210 → frame 209
    // (at 6.99 s) IS kept.
    assert_eq!(trim_frame(Some(7.0), 29.9), Some(210));
    assert_eq!(trim_frame(Some(2.0), 29.9), Some(60)); // ceil(59.8)
    // 30 fps exact boundaries.
    assert_eq!(trim_frame(Some(2.0), 30.0), Some(60));
    assert_eq!(trim_frame(Some(5.0), 30.0), Some(150));
    // Open bound and zero.
    assert_eq!(trim_frame(None, 30.0), None);
    assert_eq!(trim_frame(Some(0.0), 30.0), Some(0));
    // Negative time clamps to 0.
    assert_eq!(trim_frame(Some(-3.0), 30.0), Some(0));
}

#[test]
fn trim_audio_keeps_window_and_concat_appends() {
    // 8 packets, 1000 ticks each, timescale 1000 → one packet per second.
    let info = AudioInfo {
        codec: "opus".into(),
        sample_rate: 48000,
        channels: 2,
        timescale: 1000,
        asc_bytes: Vec::new(),
        codec_private: Vec::new(),
    };
    let mk = |n: usize| PreparedAudio {
        info: info.clone(),
        samples: (0..n).map(|i| (vec![i as u8], 1000u32)).collect(),
        handling: "passthrough".into(),
        edit: Default::default(),
    };
    let a = mk(8);
    // Trim [2s, 5s) keeps packets starting at t=2,3,4 → indices 2,3,4.
    let t = trim_audio(Some(&a), Some(2.0), Some(5.0)).unwrap();
    assert_eq!(t.samples.len(), 3);
    assert_eq!(t.samples[0].0, vec![2u8]);
    assert_eq!(t.samples[2].0, vec![4u8]);
    // Open start keeps from 0; open end keeps to the end.
    assert_eq!(trim_audio(Some(&a), None, Some(3.0)).unwrap().samples.len(), 3);
    assert_eq!(trim_audio(Some(&a), Some(6.0), None).unwrap().samples.len(), 2);
    // No bounds → unchanged.
    assert_eq!(trim_audio(Some(&a), None, None).unwrap().samples.len(), 8);
    // Concat appends.
    let mut joined = mk(3);
    joined.extend(&mk(2));
    assert_eq!(joined.samples.len(), 5);
}

#[test]
fn a_trim_on_edited_audio_cuts_the_presentation_exactly() {
    use container::edit::TrackEdit;
    // AAC-shaped: 10 packets of 1024 ticks at 48 kHz, the first 1024 ticks
    // (priming) hidden by the edit carried from the source.
    let info = AudioInfo {
        codec: "aac".into(),
        sample_rate: 48000,
        channels: 2,
        timescale: 48000,
        asc_bytes: vec![0x11, 0x90],
        codec_private: Vec::new(),
    };
    let edited = PreparedAudio {
        info,
        samples: (0..10).map(|i| (vec![i as u8], 1024u32)).collect(),
        handling: "aac passthrough".into(),
        edit: TrackEdit { delay: 0, media_time: 1024, duration: None },
    };
    // From 0.1 s of presentation = media 1024 + 4800 = 5824, inside packet 5
    // (5120..6144); packet 4 is its preroll and the edit hides 1728 of it.
    let t = trim_audio(Some(&edited), Some(0.1), None).unwrap();
    assert_eq!(t.samples.first().unwrap().0, vec![4u8]);
    assert_eq!(t.samples.len(), 6);
    assert_eq!(t.edit, TrackEdit { delay: 0, media_time: 1728, duration: None });
    // The same trim without an edit is the packet-granular trim it always was.
    let plain = PreparedAudio { edit: TrackEdit::default(), ..edited };
    let t = trim_audio(Some(&plain), Some(0.1), None).unwrap();
    assert_eq!(t.samples.first().unwrap().0, vec![5u8]);
    assert!(t.edit.is_identity());
}

#[test]
fn a_later_clip_joins_its_audio_where_its_video_starts() {
    use container::edit::TrackEdit;
    let info = AudioInfo {
        codec: "aac".into(),
        sample_rate: 48000,
        channels: 2,
        timescale: 1000,
        asc_bytes: vec![0x11, 0x90],
        codec_private: Vec::new(),
    };
    let clip = |n: u8, edit: TrackEdit| PreparedAudio {
        info: info.clone(),
        samples: (0..6).map(|i| (vec![n, i], 1000u32)).collect(),
        handling: "aac passthrough".into(),
        edit,
    };
    let order = |a: &PreparedAudio| {
        a.samples
            .iter()
            .map(|(p, _)| (p[0], p[1]))
            .collect::<Vec<_>>()
    };
    // The second clip's video starts 2 s into its audio (a transport stream
    // cut mid-GOP). Its video joins with no late start, so its audio joins
    // from 2 s: its first two packets go (the second is only the preroll, and
    // the join drops it at the boundary).
    let late_video = super::splice::trim_audio_to_video(
        Some(&clip(1, TrackEdit::default())),
        (180_000, 90_000),
        None,
        None,
    )
    .unwrap();
    let mut joined = clip(0, TrackEdit::default());
    joined.extend(&late_video);
    let mut want: Vec<(u8, u8)> = (0..6).map(|i| (0, i)).collect();
    want.extend((2..6).map(|i| (1, i)));
    assert_eq!(order(&joined), want);
    assert_eq!(
        joined.edit.duration,
        Some(10_000),
        "6 s and the 4 s after the video's start"
    );
    // A trim composes: from 1 s of the video is 3 s of the audio.
    let trimmed = super::splice::trim_audio_to_video(
        Some(&clip(1, TrackEdit::default())),
        (2, 1),
        Some(1.0),
        None,
    )
    .unwrap();
    let mut joined = clip(0, TrackEdit::default());
    joined.extend(&trimmed);
    assert_eq!(order(&joined)[6..], [(1, 3), (1, 4), (1, 5)]);
    // Audio that starts later than its video by more than the video's own
    // delay keeps the difference as a late start, which a join cannot write:
    // it joins gap-free (and warns), as before.
    let later_audio = super::splice::trim_audio_to_video(
        Some(&clip(
            1,
            TrackEdit {
                delay: 3000,
                media_time: 0,
                duration: None,
            },
        )),
        (2, 1),
        None,
        None,
    )
    .unwrap();
    assert_eq!(later_audio.edit.delay, 1000);
    assert_eq!(later_audio.samples.len(), 6);
    // The first clip (no video delay passed) is the plain trim.
    let first = super::splice::trim_audio_to_video(
        Some(&clip(0, TrackEdit::default())),
        (0, 1),
        None,
        None,
    );
    assert_eq!(first.unwrap().samples.len(), 6);
}

#[test]
fn concat_applies_an_edit_inside_the_join_to_whole_packets() {
    use container::edit::TrackEdit;
    let info = AudioInfo {
        codec: "opus".into(),
        sample_rate: 48000,
        channels: 2,
        timescale: 1000,
        asc_bytes: Vec::new(),
        codec_private: Vec::new(),
    };
    let mk = |edit: TrackEdit| PreparedAudio {
        info: info.clone(),
        samples: (0..4).map(|i| (vec![i as u8], 1000u32)).collect(),
        handling: "passthrough".into(),
        edit,
    };
    // The first clip presents 2.5 s of its 4; the next hides its first 1.5 s.
    // The first cut lands at 2 s (the nearer boundary; a tie keeps fewer), half
    // a packet short, so the next clip's start moves back by that half: 1 s.
    let mut joined = mk(TrackEdit { delay: 0, media_time: 0, duration: Some(2500) });
    joined.extend(&mk(TrackEdit { delay: 0, media_time: 1500, duration: None }));
    let order: Vec<u8> = joined.samples.iter().map(|(p, _)| p[0]).collect();
    assert_eq!(order, vec![0, 1, 1, 2, 3]);
    // The edit presents both clips' lengths, exactly: 2.5 s + 2.5 s.
    assert_eq!(joined.edit.duration, Some(5000));
}

#[test]
fn joins_across_edits_do_not_accumulate_error() {
    use container::edit::TrackEdit;
    let info = AudioInfo {
        codec: "aac".into(),
        sample_rate: 48000,
        channels: 2,
        timescale: 1000,
        asc_bytes: vec![0x11, 0x90],
        codec_private: Vec::new(),
    };
    // Ten clips of three 1000-tick packets, each presenting 2600 ticks: every
    // tail cut lands 400 ticks past where it should. Uncompensated, the tenth
    // clip would start 3600 ticks late.
    let clip = |n: u8| PreparedAudio {
        info: info.clone(),
        samples: (0..3).map(|i| (vec![n, i], 1000u32)).collect(),
        handling: "aac passthrough".into(),
        edit: TrackEdit { delay: 0, media_time: 0, duration: Some(2600) },
    };
    let mut joined = clip(0);
    for n in 1..10 {
        joined.extend(&clip(n));
    }
    assert_eq!(joined.edit.duration, Some(26_000));
    // Every clip's first kept packet starts within half a packet of where that
    // clip's presentation starts on the joined timeline.
    let mut at = 0i64;
    let mut seen = std::collections::HashSet::new();
    for (payload, d) in &joined.samples {
        let (n, i) = (payload[0], payload[1]);
        if seen.insert(n) {
            let intended = i64::from(n) * 2600 + i64::from(i) * 1000;
            assert!((at - intended).abs() <= 500, "clip {n} starts {at}, intended {intended}");
        }
        at += i64::from(*d);
    }
    assert_eq!(seen.len(), 10);
}

#[test]
fn a_joined_clips_short_last_packet_counts_at_its_decoded_length() {
    use container::edit::TrackEdit;
    let info = AudioInfo {
        codec: "aac".into(),
        sample_rate: 48000,
        channels: 2,
        timescale: 1000,
        asc_bytes: vec![0x11, 0x90],
        codec_private: Vec::new(),
    };
    // Four clips as ffmpeg writes AAC: a priming frame the clip's edit hides
    // (media_time 1000), whole 1000-tick frames, and a last frame stamped 750
    // (end padding) though a decoder turns it into a full 1000. Each clip
    // presents 2750 ticks, so its packet `i` is presented at `i * 1000 - 1000`
    // into the clip and decoded at `i * 1000` on the joined media timeline
    // where the clip is placed right.
    let clip = |n: u8| PreparedAudio {
        info: info.clone(),
        samples: vec![(vec![n, 0], 1000u32), (vec![n, 1], 1000), (vec![n, 2], 1000), (vec![n, 3], 750)],
        handling: "aac passthrough".into(),
        edit: TrackEdit { delay: 0, media_time: 1000, duration: None },
    };
    let mut joined = clip(0);
    for n in 1..4 {
        joined.extend(&clip(n));
    }
    assert_eq!(joined.edit.duration, Some(11_000));
    // On the decoded timeline (every packet 1000 ticks) each clip starts
    // within half a frame of where its presentation starts. Counting the
    // padding at 750 instead puts the fourth clip 750 ticks late.
    let mut decoded_at = 0i64;
    let mut seen = std::collections::HashSet::new();
    for (payload, _) in &joined.samples {
        let (n, i) = (payload[0], payload[1]);
        if seen.insert(n) {
            let intended = i64::from(n) * 2750 + i64::from(i) * 1000;
            assert!((decoded_at - intended).abs() <= 500, "clip {n} decodes from {decoded_at}, intended {intended}");
        }
        decoded_at += 1000;
    }
    assert_eq!(seen.len(), 4);
    // Every packet inside the joined track carries its decoded length; only
    // the very end keeps the short stamp, where the output edit cuts it.
    let durations: Vec<u32> = joined.samples.iter().map(|(_, d)| *d).collect();
    assert!(durations[..durations.len() - 1].iter().all(|&d| d == 1000), "{durations:?}");
}

#[test]
fn a_pcm_window_cuts_decoded_samples_to_the_edit() {
    use super::audio::PcmWindow;
    use codec::audio::AudioFrame;
    // Stereo frames of 1024 samples, each sample's value its position.
    let frame = |start: usize| AudioFrame {
        samples: (start..start + 1024).flat_map(|i| [i as f32, i as f32]).collect(),
        sample_rate: 48_000,
        channels: 2,
        pts: 0,
    };
    // Present media 1500..2500 of a track timed at its sample rate.
    let edit = container::edit::AudioEdit { delay: 0, media_start: 1500, media_end: Some(2500) };
    let mut w = PcmWindow::new(&edit, 48_000, 48_000);
    assert!(w.take(&frame(0)).is_none(), "0..1024 is all hidden");
    let f = w.take(&frame(1024)).expect("1024..2048 is partly presented");
    assert_eq!((f.samples.len(), f.samples[0]), ((2048 - 1500) * 2, 1500.0));
    let f = w.take(&frame(2048)).expect("2048..3072 is partly presented");
    assert_eq!((f.samples.len(), *f.samples.last().unwrap()), ((2500 - 2048) * 2, 2499.0));
    assert!(w.take(&frame(3072)).is_none(), "past the end");
    // A track timed in milliseconds: 1500..2500 ms at 48 kHz.
    let mut ms = PcmWindow::new(
        &container::edit::AudioEdit { delay: 0, media_start: 1500, media_end: None },
        1000,
        48_000,
    );
    let samples: usize = (0..80).filter_map(|i| ms.take(&frame(i * 1024))).map(|f| f.samples.len() / 2).sum();
    assert_eq!(samples, 80 * 1024 - 72_000);
}

#[test]
fn a_hole_in_a_decoded_track_is_filled_with_its_length_of_silence() {
    use container::edit::AudioGap;
    use crate::spec::AudioCodecPolicy;
    // 0.4 s of AC-3 at 48 kHz from a transport stream, decoded to Opus: a hole
    // the reader reports after its fourth frame (a transport stream's audio
    // PES lost there) plays as that much silence, so the output presents that
    // much more, and what follows it plays where its timestamps put it.
    let ts = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../container/tests/fixtures/timing/ac3_video_first.ts"
    ));
    let demuxer = container::streaming::demux_streaming(ts).expect("demux");
    let track = demuxer.audio().expect("the AC-3 track").clone();
    assert_eq!((track.codec.as_str(), track.sample_rate, track.timescale), ("ac3", 48_000, 48_000));
    let opus = |gaps: &[AudioGap]| {
        super::audio::prepare_audio(Some(&track), None, gaps, AudioCodecPolicy::ForceOpus, None, &[])
            .expect("prepare")
            .expect("an audio track")
    };
    let whole = opus(&[]);
    let holed = opus(&[AudioGap { after_packet: 3, ticks: 4800 }]);
    assert_eq!(whole.handling, "ac3 → opus (1ch)");
    assert_eq!(holed.edit.duration, whole.edit.duration.map(|d| d + 4800));
}

#[test]
fn a_hole_inside_a_joined_clip_does_not_lengthen_its_last_packet() {
    use container::edit::TrackEdit;
    let info = AudioInfo {
        codec: "ac3".into(),
        sample_rate: 48000,
        channels: 2,
        timescale: 48000,
        asc_bytes: Vec::new(),
        codec_private: Vec::new(),
    };
    // AC-3 frames of 1536 ticks; the second carries a 4800-tick hole after it
    // (its duration holds the hole, as a transport stream's reader writes
    // it), and the last is stamped short, its padding hidden by the edit. At
    // the join the last packet decodes to one frame, not to the hole's
    // length.
    let clip = |n: u8| PreparedAudio {
        info: info.clone(),
        samples: vec![(vec![n, 0], 1536u32), (vec![n, 1], 1536 + 4800), (vec![n, 2], 1536), (vec![n, 3], 1000)],
        handling: "ac3 passthrough".into(),
        edit: TrackEdit { delay: 0, media_time: 0, duration: Some(3 * 1536 + 4800 + 1000) },
    };
    let mut joined = clip(0);
    joined.extend(&clip(1));
    assert_eq!(joined.samples[3], (vec![0, 3], 1536));
    // Durations that only round a frame (Matroska's millisecond timestamps:
    // 21.33 ms frames as 21, 21, 22) still take the longest of them.
    let info = AudioInfo { timescale: 1000, ..info };
    let clip = |n: u8| PreparedAudio {
        info: info.clone(),
        samples: vec![(vec![n, 0], 21u32), (vec![n, 1], 21), (vec![n, 2], 22), (vec![n, 3], 21), (vec![n, 4], 20)],
        handling: "ac3 passthrough".into(),
        edit: TrackEdit { delay: 0, media_time: 0, duration: Some(105) },
    };
    let mut joined = clip(0);
    joined.extend(&clip(1));
    assert_eq!(joined.samples[4], (vec![0, 4], 22));
}

// ---- a silicon pin the host cannot serve, at the job's front door ----

mod refusal {
    use std::sync::Arc;
    use std::time::Duration;

    use bytes::Bytes;
    use codec::encode::EncodedPacket;
    use codec::frame::VideoCodec;
    use codec::gpu::GpuVendor;
    use container::mux::Av1Mp4Muxer;

    use crate::multigpu::test_support::within;
    use crate::progress::NullSink;
    use crate::spec::{EncodePolicy, GpuFamily, OutputSpec, Rung, VideoCodecPolicy};

    /// An MP4 the demuxer accepts and no decoder can use: the crate's own
    /// muxer around an AV1 track of filler samples (the subtitle round-trip
    /// fixture). A job that gets as far as decoding it fails saying so, and
    /// one that falls through to an encoder writes something — so a refusal
    /// that comes back naming the pin came first.
    fn undecodable_mp4() -> Bytes {
        let mut muxer = Av1Mp4Muxer::new(64, 64, 30.0).unwrap();
        let header: u8 = (1 << 3) | (1 << 1);
        let mut first = vec![header, 5];
        first.extend_from_slice(&[0u8; 5]);
        muxer.add_packet(EncodedPacket { data: Bytes::from(first), pts: 0, is_keyframe: true }).unwrap();
        for i in 1..30u64 {
            muxer.add_packet(EncodedPacket { data: Bytes::from(vec![0xAA; 64]), pts: i, is_keyframe: false }).unwrap();
        }
        muxer.finalize().unwrap()
    }

    /// A family with no card on this host, and its `--encode` spelling.
    /// `None` on a host with every vendor (the tests say so and skip).
    ///
    /// Asks the question the refusal will ask — which cards encode 8-bit
    /// H.264 — so the host is detected and probed here, before the tests'
    /// time bound starts: the job reads the same per-process answer, and the
    /// bound measures the refusal, not how fast a loaded machine probes its
    /// cards.
    fn a_family_this_host_lacks() -> Option<(GpuFamily, &'static str)> {
        let present: Vec<GpuVendor> =
            crate::multigpu::host_verdicts(VideoCodec::H264, false).iter().map(|c| c.device.vendor).collect();
        [
            (GpuFamily::Intel, GpuVendor::Intel, "intel"),
            (GpuFamily::Amd, GpuVendor::Amd, "amd"),
            (GpuFamily::Nvidia, GpuVendor::Nvidia, "nvidia"),
        ]
        .into_iter()
        .find(|(_, vendor, _)| !present.contains(vendor))
        .map(|(fam, _, flag)| (fam, flag))
    }

    fn pinned(mut spec: OutputSpec, fam: GpuFamily) -> OutputSpec {
        spec.video_codec = VideoCodecPolicy::H264;
        spec.encode_policy = EncodePolicy::Family(fam);
        spec
    }

    /// The serial path (one card's worth of pool, no chunking): the control
    /// build encoded `--encode family:intel` on NVENC and exited 0. It has
    /// to refuse, by name, before the filler is decoded.
    #[test]
    fn a_single_file_job_pinned_to_an_absent_family_is_refused_by_name() {
        let Some((fam, flag)) = a_family_this_host_lacks() else {
            eprintln!("every vendor is present on this host; nothing to refuse");
            return;
        };
        let msg = within(
            Duration::from_secs(30),
            "a single-file job pinned to an absent family waited instead of refusing",
            move || async move {
                let spec = pinned(OutputSpec::single_file(vec![Rung::new(64, 64)]), fam);
                super::super::run_job(undecodable_mp4(), &spec, None, Arc::new(NullSink))
                    .await
                    .map(|_| ())
                    .map_err(|e| format!("{e:#}"))
            },
        )
        .expect_err("nothing this job pinned can encode");
        assert!(msg.contains(&format!("no encoder matches `--encode family:{flag}` for H.264 on this host: no ")), "{msg}");
    }

    /// The HLS job: the control build sat at `0/120 frames` on the same
    /// pin. Refused by name, nothing written under the output root.
    #[test]
    fn an_hls_job_pinned_to_an_absent_family_is_refused_by_name() {
        let Some((fam, flag)) = a_family_this_host_lacks() else {
            eprintln!("every vendor is present on this host; nothing to refuse");
            return;
        };
        let root = tempfile::tempdir().unwrap();
        let dir = root.path().to_path_buf();
        let msg = within(
            Duration::from_secs(30),
            "an HLS job pinned to an absent family waited for a lease instead of refusing",
            move || async move {
                let spec = pinned(OutputSpec::hls(vec![Rung::new(64, 64)], 1.0), fam);
                super::super::run_job(undecodable_mp4(), &spec, Some(dir.as_path()), Arc::new(NullSink))
                    .await
                    .map(|_| ())
                    .map_err(|e| format!("{e:#}"))
            },
        )
        .expect_err("nothing this job pinned can encode");
        assert!(msg.contains(&format!("no encoder matches `--encode family:{flag}` for H.264 on this host: no ")), "{msg}");
        let written: Vec<_> = std::fs::read_dir(root.path()).unwrap().collect();
        assert!(written.is_empty(), "a refused job wrote {} entries", written.len());
    }

    /// A splice encodes serially too; the same pin is refused the same way,
    /// before a clip is decoded.
    #[test]
    fn a_splice_job_pinned_to_an_absent_family_is_refused_by_name() {
        let Some((fam, flag)) = a_family_this_host_lacks() else {
            eprintln!("every vendor is present on this host; nothing to refuse");
            return;
        };
        let msg = within(
            Duration::from_secs(30),
            "a splice job pinned to an absent family waited instead of refusing",
            move || async move {
                let spec = pinned(OutputSpec::single_file(vec![Rung::new(64, 64)]), fam);
                let clips = vec![super::super::Clip::new(undecodable_mp4())];
                super::super::run_splice_job(clips, &spec, None, Arc::new(NullSink))
                    .await
                    .map(|_| ())
                    .map_err(|e| format!("{e:#}"))
            },
        )
        .expect_err("nothing this job pinned can encode");
        assert!(msg.contains(&format!("no encoder matches `--encode family:{flag}` for H.264 on this host: no ")), "{msg}");
    }
}

/// A rung that fails reports its whole error chain — in the log and in the
/// progress message the CLI and `/v1/jobs` show — not just the outermost
/// context: "finalize" alone hid the duplicated timestamps behind every
/// failed two-clip splice.
#[test]
fn a_failed_rung_reports_its_whole_error_chain() {
    use std::sync::Mutex;

    use crate::progress::{ProgressSink, RungProgress, RungStatus};
    use crate::spec::Rung;

    struct Recorder(Mutex<Vec<RungProgress>>);
    impl ProgressSink for Recorder {
        fn on_rung(&self, update: RungProgress) {
            self.0.lock().unwrap().push(update);
        }
    }
    let sink = Recorder(Mutex::new(Vec::new()));
    let error = anyhow::anyhow!("presentation timestamp 30 appears on two samples")
        .context("placing video samples by presentation order")
        .context("finalize");
    super::report_rung_error(&sink, 2, &Rung::new(640, 360), &error);
    let got = sink.0.lock().unwrap();
    assert_eq!(got.len(), 1);
    assert_eq!((got[0].rung_index, got[0].status), (2, RungStatus::Failed));
    assert_eq!(
        got[0].message.as_deref(),
        Some("finalize: placing video samples by presentation order: presentation timestamp 30 appears on two samples")
    );
}
