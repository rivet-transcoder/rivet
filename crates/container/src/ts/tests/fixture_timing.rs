//! The program clock on real muxes: ffmpeg transport streams from
//! `tests/fixtures/timing/make_fixtures.sh`, read through both readers. The
//! expected starts are ffprobe's first packet PTSes (in the script's output),
//! so the late start each reader gives a stream is the gap ffprobe shows.

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
fn aac_after_the_video_starts_late_by_the_gap_ffprobe_shows() {
    // ffprobe: video 133200 (the IDR, presented first), audio 136680: 3480
    // ticks, 1856 samples at 48 kHz. (-itsoffset 0.06 less the AAC priming
    // ffmpeg takes off: 5400 - 1920.)
    assert_eq!(
        starts("video_first.ts", fixture!("video_first.ts")),
        (0, late(1856))
    );
}

#[test]
fn video_after_the_aac_starts_late_by_the_gap_ffprobe_shows() {
    // ffprobe: audio 126000, video 138720.
    assert_eq!(
        starts("audio_first.ts", fixture!("audio_first.ts")),
        (12_720, None)
    );
}

#[test]
fn ac3_after_the_video_starts_late_by_the_gap_ffprobe_shows() {
    // ffprobe: video 133200, AC-3 139920: 6720 ticks, 3584 samples.
    let ts = fixture!("ac3_video_first.ts");
    assert_eq!(starts("ac3_video_first.ts", ts), (0, late(3584)));
    let demuxer = crate::streaming::demux_streaming(ts).expect("streaming demux");
    assert_eq!(demuxer.audio().map(|a| a.codec.as_str()), Some("ac3"));
}

#[test]
fn a_mid_gop_start_keeps_its_idr_where_it_was_against_the_audio() {
    // ffprobe: three access units before the IDR at 145920 (PTS 142320,
    // 135120, 138720 — a P and two B pictures), audio at 127680. The base is
    // the audio's first frame, and the IDR is 18240 ticks past it.
    let ts = fixture!("midgop.ts");
    assert_eq!(starts("midgop.ts", ts), (18_240, None));
    let mut demuxer = crate::streaming::demux_streaming(ts).expect("streaming demux");
    let first = demuxer.next_video_sample().expect("sample").expect("one");
    assert_eq!(first.pts_ticks, 145_920, "the first sample is the IDR");
    assert!(crate::nal_mux::sample_is_keyframe(
        &first.data,
        crate::nal_mux::NalMuxCodec::H264
    ));
}

#[test]
fn a_mux_across_the_pts_wrap_keeps_the_gap_the_frame_rate_and_the_times() {
    // ffprobe (which unwraps): video -19592, audio -16112 — 3480 ticks apart
    // across 2^33, as in video_first.ts.
    let ts = fixture!("wrap.ts");
    assert_eq!(starts("wrap.ts", ts), (0, late(1856)));
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
