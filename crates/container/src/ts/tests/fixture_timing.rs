//! The program clock on real muxes: transport streams from GStreamer's
//! mpegtsmux (`tests/fixtures/timing/make_fixtures.sh`), read through both
//! readers. The expected starts are the first PTSes in the PES headers, as
//! `ts_times.py` reads them straight from the packets (in the script's output),
//! so the late start each reader gives a stream is the gap the headers show.

use crate::edit::AudioEdit;

macro_rules! fixture {
    ($name:literal) => {
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/timing/",
            $name
        ))
    };
}

/// The video's late start (90 kHz) and the audio's edit, from the streaming
/// reader; the whole-file reader must agree.
fn starts(name: &str, ts: &[u8]) -> (u64, Option<AudioEdit>) {
    let demuxer = crate::streaming::demux_streaming(ts).expect("streaming demux");
    let video = demuxer.video_presentation().map_or(0, |p| {
        assert_eq!((p.delay_timescale, p.hidden.len()), (90_000, 0), "{name}");
        p.delay_ticks
    });
    let audio = demuxer.audio_edit();
    let whole = crate::demux::demux(ts).expect("whole-file demux");
    assert_eq!(
        (
            whole.video_presentation.map_or(0, |p| p.delay_ticks),
            whole.audio_edit
        ),
        (video, audio),
        "{name}: both readers"
    );
    (video, audio)
}

fn late(delay: u64) -> Option<AudioEdit> {
    Some(AudioEdit {
        delay,
        media_start: 0,
        media_end: None,
    })
}

#[test]
fn aac_after_the_video_starts_late_by_the_gap_the_pes_headers_show() {
    // PES headers: video 324000000 (the IDR, presented first), audio
    // 324005400: 5400 ticks (the 60 ms offset), 2880 samples at 48 kHz.
    assert_eq!(
        starts("video_first.ts", fixture!("video_first.ts")),
        (0, late(2880))
    );
}

#[test]
fn video_after_the_aac_starts_late_by_the_gap_the_pes_headers_show() {
    // PES headers: audio 324000000, video 324009000 (the 100 ms offset).
    assert_eq!(
        starts("audio_first.ts", fixture!("audio_first.ts")),
        (9_000, None)
    );
}

#[test]
fn ac3_after_the_video_starts_late_by_the_gap_the_pes_headers_show() {
    // PES headers: video 324000000, AC-3 324007200: 7200 ticks, 3840 samples.
    let ts = fixture!("ac3_video_first.ts");
    assert_eq!(starts("ac3_video_first.ts", ts), (0, late(3840)));
    let demuxer = crate::streaming::demux_streaming(ts).expect("streaming demux");
    assert_eq!(demuxer.audio().map(|a| a.codec.as_str()), Some("ac3"));
}

#[test]
fn a_mid_gop_start_keeps_its_idr_where_it_was_against_the_audio() {
    // PES headers: three access units before the IDR at 324028800 (PTS
    // 324025200, 324018000, 324021600 — a P and two B pictures), audio at
    // 324011520. The base is the audio's first frame, and the IDR is 17280
    // ticks past it.
    let ts = fixture!("midgop.ts");
    assert_eq!(starts("midgop.ts", ts), (17_280, None));
    let mut demuxer = crate::streaming::demux_streaming(ts).expect("streaming demux");
    let first = demuxer.next_video_sample().expect("sample").expect("one");
    assert_eq!(first.pts_ticks, 324_028_800, "the first sample is the IDR");
    assert!(crate::nal_mux::sample_is_keyframe(
        &first.data,
        crate::nal_mux::NalMuxCodec::H264
    ));
}

#[test]
fn a_mux_across_the_pts_wrap_keeps_the_gap_the_frame_rate_and_the_times() {
    // PES headers: video 2^33 - 19592, audio 2^33 - 14192 — 5400 ticks apart
    // across 2^33, as in video_first.ts.
    let ts = fixture!("wrap.ts");
    assert_eq!(starts("wrap.ts", ts), (0, late(2880)));
    let mut demuxer = crate::streaming::demux_streaming(ts).expect("streaming demux");
    assert_eq!(demuxer.header().info.frame_rate, 25.0);
    let mut times = Vec::new();
    while let Some(s) = demuxer.next_video_sample().expect("sample") {
        times.push(s.pts_ticks);
    }
    let first = (1i64 << 33) - 19_592;
    assert_eq!(times.first(), Some(&first));
    let mut sorted = times.clone();
    sorted.sort_unstable();
    let expected: Vec<i64> = (0..10).map(|i| first + i * 3600).collect();
    assert_eq!(
        sorted, expected,
        "ten frames 40 ms apart, on through the wrap"
    );
    let whole = crate::demux::demux(ts).expect("whole-file demux");
    assert_eq!(whole.info.frame_rate, 25.0);
    assert!(whole.info.duration > 0.3, "{}", whole.info.duration);
}

/// The frame count, frame rate and duration the streaming reader gives each
/// fixture: the frames a decoder makes of it (the PES packets `ts_times.py`
/// counts, less the pictures a decoder starting there cannot output).
#[test]
fn every_fixture_counts_the_frames_a_decoder_makes() {
    let cases: [(&str, &[u8], u64, f64); 8] = [
        ("video_first.ts", fixture!("video_first.ts"), 10, 25.0),
        ("audio_first.ts", fixture!("audio_first.ts"), 10, 25.0),
        (
            "ac3_video_first.ts",
            fixture!("ac3_video_first.ts"),
            10,
            25.0,
        ),
        ("wrap.ts", fixture!("wrap.ts"), 10, 25.0),
        // Eleven PES packets, three before the IDR, which the reader drops.
        ("midgop.ts", fixture!("midgop.ts"), 8, 25.0),
        // Sixteen field pictures, one PES each, 1800 ticks apart: eight frames
        // at 25 fps, not sixteen at 50.
        ("paff_fields.ts", fixture!("paff_fields.ts"), 8, 25.0),
        // The same fields, a pair to a PES.
        ("paff_pairs.ts", fixture!("paff_pairs.ts"), 8, 25.0),
        // Eleven PES packets from a CRA, two of them its RASL pictures.
        ("rasl_cut.ts", fixture!("rasl_cut.ts"), 9, 25.0),
    ];
    for (name, ts, frames, rate) in cases {
        let demuxer = crate::streaming::demux_streaming(ts).expect("streaming demux");
        let info = &demuxer.header().info;
        assert_eq!(
            (info.total_frames, info.frame_rate),
            (frames, rate),
            "{name}"
        );
        assert!(
            (info.duration - frames as f64 / rate).abs() < 1e-9,
            "{name}: {}",
            info.duration
        );
        let whole = crate::demux::demux(ts).expect("whole-file demux");
        assert_eq!(
            whole.info.frame_rate, rate,
            "{name}: the whole-file reader's rate"
        );
    }
}
