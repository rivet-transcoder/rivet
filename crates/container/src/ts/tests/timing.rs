//! The program clock: where the video and the audio start against each other,
//! across the 33-bit wrap, when the stream opens mid-GOP, and on both readers.

use super::super::clock::{PTS_MODULUS, PtsUnwrapper, unwrap_pts};
use super::super::{
    STREAM_TYPE_AAC_ADTS, STREAM_TYPE_H264, STREAM_TYPE_HEVC, demux_ts_streaming_init,
};
use super::build_adts_header_7;
use crate::edit::AudioEdit;
use crate::streaming::StreamingDemuxer;

const VIDEO_PID: u16 = 0x100;
const AUDIO_PID: u16 = 0x101;
/// 25 fps on the 90 kHz clock.
const FRAME: u64 = 3600;
/// One AAC frame, 1024 samples at 48 kHz, on the 90 kHz clock.
const AAC_FRAME: u64 = 1920;

/// A PTS as the five bytes of a PES header ('0010' prefix, marker bits).
fn pts_bytes(pts: u64) -> [u8; 5] {
    [
        0x21 | ((pts >> 29) & 0x0E) as u8,
        (pts >> 22) as u8,
        ((pts >> 14) as u8 & 0xFE) | 1,
        (pts >> 7) as u8,
        ((pts << 1) as u8 & 0xFE) | 1,
    ]
}

/// A PES packet: header with the PTS (or none), then `es`.
fn pes(stream_id: u8, pts: Option<u64>, es: &[u8]) -> Vec<u8> {
    let mut out = vec![0, 0, 1, stream_id, 0, 0, 0x80];
    match pts {
        Some(pts) => {
            out.extend_from_slice(&[0x80, 5]);
            out.extend_from_slice(&pts_bytes(pts));
        }
        None => out.extend_from_slice(&[0x00, 0]),
    }
    out.extend_from_slice(es);
    out
}

/// `pes` as TS packets on `pid`: the last one's short payload is padded with
/// adaptation-field stuffing, as a muxer does, so no filler reaches the
/// elementary stream.
fn packetize(pid: u16, pes: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    for (i, chunk) in pes.chunks(184).enumerate() {
        let mut p = [0xFFu8; 188];
        p[0] = 0x47;
        p[1] = if i == 0 { 0x40 } else { 0 } | (pid >> 8) as u8;
        p[2] = pid as u8;
        if chunk.len() == 184 {
            p[3] = 0x10;
            p[4..].copy_from_slice(chunk);
        } else {
            p[3] = 0x30;
            let stuffing = 183 - chunk.len();
            p[4] = stuffing as u8;
            if stuffing > 0 {
                p[5] = 0;
            }
            p[188 - chunk.len()..].copy_from_slice(chunk);
        }
        out.extend_from_slice(&p);
    }
    out
}

/// PAT (program 1 on PMT 0x1000) and a PMT with the video stream (H.264 or
/// HEVC) and AAC-ADTS audio.
fn psi(video_stream_type: u8) -> Vec<u8> {
    let mut pat = vec![0x00, 0xB0, 13, 0x00, 0x01, 0xC1, 0x00, 0x00];
    pat.extend_from_slice(&[0x00, 0x01, 0xF0, 0x00, 0, 0, 0, 0]);
    let mut pmt = vec![0x02, 0xB0, 23, 0x00, 0x01, 0xC1, 0x00, 0x00];
    pmt.extend_from_slice(&[0xE1, 0x00, 0xF0, 0x00]);
    pmt.extend_from_slice(&[video_stream_type, 0xE1, 0x00, 0xF0, 0x00]);
    pmt.extend_from_slice(&[STREAM_TYPE_AAC_ADTS, 0xE1, 0x01, 0xF0, 0x00]);
    pmt.extend_from_slice(&[0, 0, 0, 0]);
    let mut out = packetize(0, &[&[0u8][..], &pat].concat());
    out.extend(packetize(0x1000, &[&[0u8][..], &pmt].concat()));
    out
}

/// An H.264 access unit: one slice NAL of `nal_type` (5 = IDR, 1 = non-IDR).
fn h264_au(nal_type: u8) -> Vec<u8> {
    vec![0, 0, 0, 1, 0x60 | nal_type, 0x88, 0x84, 0x21]
}

/// An HEVC access unit: one slice NAL of `nal_type` (19 = IDR_W_RADL,
/// 21 = CRA, 6 = RADL_N, 8 = RASL_N, 1 = TRAIL_R).
fn hevc_au(nal_type: u8) -> Vec<u8> {
    vec![0, 0, 0, 1, nal_type << 1, 0x01, 0xAF, 0x09]
}

/// One AAC-LC 48 kHz stereo ADTS frame with a silent payload.
fn adts_frame() -> Vec<u8> {
    let mut f = build_adts_header_7(1, 3, 2, 7 + 20).to_vec();
    f.extend_from_slice(&[0u8; 20]);
    f
}

/// A transport stream: the video access units (with their PTSes, in decoding
/// order), then the audio PES packets, each holding `frames` ADTS frames.
fn stream(
    video_stream_type: u8,
    video: &[(Vec<u8>, Option<u64>)],
    audio: &[(Option<u64>, Vec<u8>)],
) -> Vec<u8> {
    let mut ts = psi(video_stream_type);
    for (au, pts) in video {
        ts.extend(packetize(VIDEO_PID, &pes(0xE0, *pts, au)));
    }
    for (pts, es) in audio {
        ts.extend(packetize(AUDIO_PID, &pes(0xC0, *pts, es)));
    }
    ts
}

/// Eight H.264 access units at 25 fps from `first`: an IDR, then non-IDR.
fn gop(first: u64) -> Vec<(Vec<u8>, Option<u64>)> {
    (0..8)
        .map(|i| {
            let nal = if i == 0 { 5 } else { 1 };
            (h264_au(nal), Some((first + i * FRAME) % PTS_MODULUS as u64))
        })
        .collect()
}

/// Ten AAC PES packets of one frame each, from `first`.
fn aac(first: u64) -> Vec<(Option<u64>, Vec<u8>)> {
    (0..10)
        .map(|i| {
            (
                Some((first + i * AAC_FRAME) % PTS_MODULUS as u64),
                adts_frame(),
            )
        })
        .collect()
}

/// What both readers say about where the streams start: the video's delay in
/// 90 kHz ticks and the audio's edit. They must agree.
fn starts(ts: &[u8]) -> (u64, Option<AudioEdit>) {
    let demuxer = demux_ts_streaming_init(bytes::Bytes::from(ts.to_vec())).expect("streaming");
    let video = demuxer.video_presentation().map_or(0, |p| {
        assert_eq!(p.delay_timescale, 90_000);
        assert!(p.hidden.is_empty(), "a late start hides nothing");
        p.delay_ticks
    });
    let audio = demuxer.audio_edit();
    assert_eq!(demuxer.audio().map(|a| a.timescale), Some(48_000));
    let whole = crate::demux::demux(ts).expect("whole-file");
    assert_eq!(
        whole.video_presentation.map_or(0, |p| p.delay_ticks),
        video,
        "the whole-file reader agrees on the video"
    );
    assert_eq!(
        whole.audio_edit, audio,
        "the whole-file reader agrees on the audio"
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
fn unwrapping_keeps_order_and_distance_across_the_wrap() {
    let top = PTS_MODULUS - 900;
    assert_eq!(
        unwrap_pts(900, top),
        PTS_MODULUS + 900,
        "just after the wrap"
    );
    assert_eq!(unwrap_pts(top as u64, 900), -900, "just before it");
    assert_eq!(unwrap_pts(5000, 3000), 5000, "no wrap between");
    assert_eq!(unwrap_pts(1000, 3000), 1000, "an earlier one stays earlier");
    // A stream running across the wrap: 900 ticks before it, then on in
    // steps of 3600.
    let mut u = PtsUnwrapper::default();
    let run: Vec<i64> = [top as u64, 2700, 6300, 9900].map(|p| u.unwrap(p)).to_vec();
    assert_eq!(run, vec![top, top + 3600, top + 7200, top + 10800]);
}

#[test]
fn audio_starting_after_the_video_starts_late_by_the_gap() {
    // Video at 10 s, audio 20 ms (1800 ticks) later: 960 samples at 48 kHz.
    let v = 900_000;
    let ts = stream(STREAM_TYPE_H264, &gop(v), &aac(v + 1800));
    assert_eq!(starts(&ts), (0, late(960)));
}

#[test]
fn video_starting_after_the_audio_starts_late_by_the_gap() {
    // The ffmpeg shape: audio at the mux delay, the first picture one AAC
    // frame (21.3 ms) later.
    let v = 132_000;
    let ts = stream(STREAM_TYPE_H264, &gop(v), &aac(v - AAC_FRAME));
    assert_eq!(starts(&ts), (AAC_FRAME, None));
}

#[test]
fn streams_starting_together_need_nothing() {
    let v = 126_000;
    let ts = stream(STREAM_TYPE_H264, &gop(v), &aac(v));
    assert_eq!(starts(&ts), (0, None));
}

#[test]
fn a_start_either_side_of_the_wrap_keeps_the_gap() {
    // The video 900 ticks before the wrap, the audio 1800 ticks after it
    // (900 past zero).
    let top = PTS_MODULUS as u64 - 900;
    let ts = stream(STREAM_TYPE_H264, &gop(top), &aac(900));
    assert_eq!(starts(&ts), (0, late(960)), "audio after, across the wrap");
    // And the other way round: the audio before the wrap, the video after.
    let ts = stream(STREAM_TYPE_H264, &gop(900), &aac(top));
    assert_eq!(starts(&ts), (1800, None), "video after, across the wrap");
}

#[test]
fn a_wrap_inside_the_stream_keeps_the_frame_rate_duration_and_sample_times() {
    // Three frames before the wrap, five after.
    let v = PTS_MODULUS as u64 - 3 * FRAME;
    let ts = stream(STREAM_TYPE_H264, &gop(v), &aac(v));
    let mut demuxer = demux_ts_streaming_init(bytes::Bytes::from(ts.clone())).expect("streaming");
    assert_eq!(demuxer.header().info.frame_rate, 25.0);
    let mut times = Vec::new();
    while let Some(s) = demuxer.next_video_sample().expect("sample") {
        times.push(s.pts_ticks);
    }
    let expected: Vec<i64> = (0..8).map(|i| v as i64 + i * FRAME as i64).collect();
    assert_eq!(times, expected, "sample times run on through the wrap");
    let whole = crate::demux::demux(&ts).expect("whole-file");
    assert_eq!(whole.info.frame_rate, 25.0);
    assert!(
        (whole.info.duration - 7.0 * 0.04).abs() < 1e-9,
        "{}",
        whole.info.duration
    );
    assert_eq!(starts(&ts), (0, None));
}

/// Two access units before the IDR, 25 fps from `first`.
fn mid_gop(first: u64) -> Vec<(Vec<u8>, Option<u64>)> {
    let mut units = vec![(h264_au(1), Some(first)), (h264_au(1), Some(first + FRAME))];
    units.extend(gop(first + 2 * FRAME));
    units
}

#[test]
fn a_mid_gop_start_drops_its_leading_units_and_keeps_them_on_the_clock() {
    let v = 900_000;
    // Audio before everything: the base is the audio's first frame, and the
    // IDR — the first picture presented — is 1800 + 2 frames past it.
    let ts = stream(STREAM_TYPE_H264, &mid_gop(v), &aac(v - 1800));
    assert_eq!(starts(&ts), (1800 + 2 * FRAME, None));
    let mut demuxer = demux_ts_streaming_init(bytes::Bytes::from(ts)).expect("streaming");
    let first = demuxer.next_video_sample().expect("sample").expect("one");
    assert_eq!(
        first.pts_ticks,
        (v + 2 * FRAME) as i64,
        "the first sample is the IDR"
    );
    // Audio between the dropped units and the IDR: the base is the first
    // dropped unit (as without audio), the audio a frame past it.
    let ts = stream(STREAM_TYPE_H264, &mid_gop(v), &aac(v + FRAME));
    assert_eq!(starts(&ts), (2 * FRAME, late(1920)));
    // No audio: the late start the dropped units leave, as before.
    let video_only = {
        let mut ts = psi(STREAM_TYPE_H264);
        for (au, pts) in mid_gop(v) {
            ts.extend(packetize(VIDEO_PID, &pes(0xE0, pts, &au)));
        }
        ts
    };
    let demuxer = demux_ts_streaming_init(bytes::Bytes::from(video_only)).expect("streaming");
    assert_eq!(
        demuxer.video_presentation().map(|p| p.delay_ticks),
        Some(2 * FRAME)
    );
}

#[test]
fn hevc_decodable_leading_pictures_start_the_video_and_skipped_ones_do_not() {
    let v = 900_000;
    // IDR_W_RADL presented third: its two RADL pictures come after it in
    // decoding order and before it on screen, so the video starts at the
    // first of them, with the audio.
    let radl = vec![
        (hevc_au(19), Some(v + 2 * FRAME)),
        (hevc_au(6), Some(v)),
        (hevc_au(6), Some(v + FRAME)),
        (hevc_au(1), Some(v + 3 * FRAME)),
    ];
    let ts = stream(STREAM_TYPE_HEVC, &radl, &aac(v));
    assert_eq!(starts(&ts), (0, None));
    // A CRA with RASL pictures in the same places: a decoder starting at the
    // CRA does not output them, so the video starts at the CRA.
    let rasl = vec![
        (hevc_au(21), Some(v + 2 * FRAME)),
        (hevc_au(8), Some(v)),
        (hevc_au(8), Some(v + FRAME)),
        (hevc_au(1), Some(v + 3 * FRAME)),
    ];
    let ts = stream(STREAM_TYPE_HEVC, &rasl, &aac(v));
    assert_eq!(starts(&ts), (2 * FRAME, None));
}

#[test]
fn the_first_audio_frame_is_placed_by_the_first_pes_that_times_one() {
    let v = 900_000;
    // The first audio PES carries no PTS: the second one's frame is placed
    // by its PTS and the first frame one frame before it — 20 ms after the
    // video.
    let mut audio = aac(v + 1800);
    audio[0].0 = None;
    let ts = stream(STREAM_TYPE_H264, &gop(v), &audio);
    assert_eq!(starts(&ts), (0, late(960)));
    // The first PES opens on the tail of a frame cut off before the stream
    // began: its PTS is the first whole frame's, which is where the track
    // starts.
    let mut audio = aac(v + 1800);
    audio[0].1 = [vec![0u8; 11], adts_frame()].concat();
    let ts = stream(STREAM_TYPE_H264, &gop(v), &audio);
    assert_eq!(starts(&ts), (0, late(960)));
    // Two frames in one PES: the PTS is the first one's, and the next
    // PES's frames follow.
    let paired: Vec<(Option<u64>, Vec<u8>)> = (0..5)
        .map(|i| {
            (
                Some(v + 3600 + i * 2 * AAC_FRAME),
                [adts_frame(), adts_frame()].concat(),
            )
        })
        .collect();
    let ts = stream(STREAM_TYPE_H264, &gop(v), &paired);
    assert_eq!(starts(&ts), (0, late(1920)));
}
