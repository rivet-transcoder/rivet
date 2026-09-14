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

// ---- a silicon pin the host cannot serve, at the job's front door ----

mod refusal {
    use std::sync::Arc;
    use std::time::Duration;

    use bytes::Bytes;
    use codec::encode::EncodedPacket;
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
    fn a_family_this_host_lacks() -> Option<(GpuFamily, &'static str)> {
        let present: Vec<GpuVendor> = codec::gpu::detect_gpus().iter().map(|g| g.vendor).collect();
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
