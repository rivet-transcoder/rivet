//! A hole, a dropout, a splice: the audio of `tests/fixtures/timing`'s
//! `robust_*.ts` (`robust_ts.py`) placed by its timestamps against the video's
//! pictures ([`crate::ts::retime`]), and the time-base breaks found on the PCR
//! PID ([`crate::ts::discontinuity`]).
//!
//! robust_src.ts: 40 frames of H.264 at 25 fps from PTS 133200 (3600 apart),
//! 76 AAC frames at 48 kHz from PTS 136680 (1920 apart, 1024 ticks each), the
//! PCR on the video PID. Each case checks where every audio packet lands on
//! the track's timeline (the running sum of the durations before it) against
//! where the source's own timestamps, measured against the pictures the output
//! presents, put it.

use std::collections::HashMap;

use crate::demux::AudioTrack;
use crate::edit::AudioGap;
use crate::ts::audio::TsAudio;
use crate::ts::clock::{PTS_MODULUS, ProgramClock};
use crate::ts::discontinuity::{Segmenter, Stretch, pcr_pid, time_base_breaks};
use crate::ts::pictures::VideoSegment;
use crate::ts::retime::place_program_audio;
use crate::ts::{TS_PACKET, TS_SYNC};

macro_rules! fixture {
    ($name:literal) => {
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/timing/",
            $name
        ))
    };
}

/// robust_src.ts's audio frames: index by payload.
fn source_frames() -> HashMap<Vec<u8>, i64> {
    let demuxer = crate::streaming::demux_streaming(fixture!("robust_src.ts")).expect("demux");
    let track = demuxer.audio().expect("audio");
    let frames: HashMap<Vec<u8>, i64> = track
        .samples
        .iter()
        .enumerate()
        .map(|(i, s)| (s.clone(), i as i64))
        .collect();
    assert_eq!(
        frames.len(),
        76,
        "every AAC frame of the source tells itself apart"
    );
    frames
}

/// Each audio packet of `ts` (both readers agreeing): which copy of the
/// source it comes from (a new copy starts where a frame index goes back),
/// its source frame index, and where it starts on the track's timeline
/// (ticks). With the holes the streaming reader reports.
fn placed(name: &str, ts: &[u8]) -> (Vec<(usize, i64, i64)>, Vec<AudioGap>) {
    let frames = source_frames();
    let demuxer = crate::streaming::demux_streaming(ts).expect("streaming demux");
    let track = demuxer.audio().expect("audio").clone();
    let gaps = demuxer.audio_gaps().to_vec();
    let whole = crate::demux::demux(ts).expect("whole-file demux");
    let whole_track = whole.audio.expect("whole-file audio");
    assert_eq!(
        (&whole_track.samples, &whole_track.durations),
        (&track.samples, &track.durations),
        "{name}: both readers place the audio alike"
    );
    let (mut copy, mut last, mut at) = (0usize, -1i64, 0i64);
    let mut out = Vec::new();
    for (sample, &duration) in track.samples.iter().zip(&track.durations) {
        let m = frames[sample];
        if m < last {
            copy += 1;
        }
        last = m;
        out.push((copy, m, at));
        at += i64::from(duration);
    }
    // The holes are the durations past a frame's own.
    let extra: Vec<(usize, u64)> = track
        .durations
        .iter()
        .enumerate()
        .filter(|&(_, &d)| d != 1024)
        .map(|(i, &d)| (i, u64::from(d - 1024)))
        .collect();
    assert_eq!(
        extra,
        gaps.iter()
            .map(|g| (g.after_packet, g.ticks))
            .collect::<Vec<_>>(),
        "{name}: every lengthened packet is a hole the reader reports"
    );
    (out, gaps)
}

/// The time-base breaks of `ts` (packet indices).
fn breaks(ts: &[u8]) -> Vec<usize> {
    let layout = crate::ts::detect_packet_layout(ts).expect("layout");
    time_base_breaks(ts, layout, pcr_pid(ts, layout, 0x1000))
}

#[test]
fn the_pcr_pid_is_the_pmts() {
    let ts = fixture!("robust_src.ts");
    let layout = crate::ts::detect_packet_layout(ts).expect("layout");
    assert_eq!(pcr_pid(ts, layout, 0x1000), Some(0x100));
}

#[test]
fn a_whole_stream_is_left_exactly_as_it_was() {
    assert!(breaks(fixture!("robust_src.ts")).is_empty());
    let (packets, gaps) = placed("robust_src.ts", fixture!("robust_src.ts"));
    assert!(gaps.is_empty());
    assert_eq!(packets.len(), 76);
    for (copy, m, at) in packets {
        assert_eq!((copy, at), (0, m * 1024), "frame {m}");
    }
}

#[test]
fn a_hole_in_the_audio_alone_is_kept_as_time() {
    // Frames 27..=40 are gone (PTS 186600 to 215400); the video has every
    // picture across them, so frame 41 still plays 41 frames in.
    assert!(breaks(fixture!("robust_hole.ts")).is_empty());
    let (packets, gaps) = placed("robust_hole.ts", fixture!("robust_hole.ts"));
    assert_eq!(
        gaps,
        [AudioGap {
            after_packet: 26,
            ticks: 14 * 1024
        }]
    );
    assert_eq!(packets.len(), 62);
    for (_, m, at) in packets {
        assert_eq!(at, m * 1024, "frame {m}");
    }
}

#[test]
fn a_dropout_of_both_streams_closes_up_in_both() {
    // The video's GOP from PTS 205200 to 241200 is gone (ten pictures, 0.4 s:
    // 19200 ticks of audio) and the audio beside it. The output presents the
    // pictures either side of the gap one after the other, so the audio after
    // it plays 19200 ticks before its own timestamps say: within half a frame
    // (512 ticks) of where its pictures play, with no hole.
    let (packets, gaps) = placed("robust_dropout.ts", fixture!("robust_dropout.ts"));
    assert!(gaps.is_empty(), "{gaps:?}");
    for (_, m, at) in packets {
        let want = if m <= 36 { m * 1024 } else { m * 1024 - 19200 };
        assert!((at - want).abs() <= 512, "frame {m} at {at}, wanted {want}");
    }
}

#[test]
fn an_audio_pes_sent_twice_plays_once() {
    // The source's second audio PES (TS packet 24, one AAC frame in one
    // packet) sent again right after itself: the copy overlaps the frame it
    // repeats and goes; everything after plays where it did.
    let src = fixture!("robust_src.ts");
    let mut ts = src[..25 * TS_PACKET].to_vec();
    ts.extend_from_slice(&src[24 * TS_PACKET..25 * TS_PACKET]);
    ts.extend_from_slice(&src[25 * TS_PACKET..]);
    let (packets, gaps) = placed("repeated PES", &ts);
    assert!(gaps.is_empty());
    assert_eq!(packets.len(), 76);
    for (copy, m, at) in packets {
        assert_eq!((copy, at), (0, m * 1024), "frame {m}");
    }
}

/// robust_src.ts's second copy, after a join: its pictures play from 40
/// frames in (144000 at 90 kHz, 76800 ticks of audio), and its audio frame
/// `m` 1856 ticks after its first picture plus `m` frames — 76800 + 1024 m
/// from where the first copy's audio starts, within half a frame (the first
/// copy's audio runs 1024 ticks past its last picture, so a frame of the
/// second goes).
fn second_copy_plays_against_its_pictures(name: &str, ts: &[u8]) {
    let (packets, gaps) = placed(name, ts);
    assert!(gaps.is_empty(), "{name}: {gaps:?}");
    assert!(packets.iter().any(|p| p.0 == 1), "{name}: a second copy");
    for (copy, m, at) in packets {
        let want = if copy == 0 {
            m * 1024
        } else {
            76800 + m * 1024
        };
        assert!(
            (at - want).abs() <= 512,
            "{name}: copy {copy} frame {m} at {at}, wanted {want}"
        );
    }
}

#[test]
fn audio_after_the_clock_jumps_back_plays_against_the_pictures_after_it() {
    // `cat a.ts a.ts`: the PCR jumps back at the join, unmarked.
    let src = fixture!("robust_src.ts");
    let concat = [src.as_slice(), src.as_slice()].concat();
    assert_eq!(breaks(&concat).len(), 1);
    second_copy_plays_against_its_pictures("concat", &concat);
}

#[test]
fn a_marked_discontinuity_is_a_break_where_the_clock_only_steps_on() {
    // The second copy's clock resumes 2 s after the first's ends: no jump
    // test sees it, the discontinuity_indicator does.
    let ts = fixture!("robust_splice.ts");
    let found = breaks(ts);
    assert_eq!(found.len(), 1);
    // The break is the second copy's first PCR packet.
    let layout = crate::ts::detect_packet_layout(ts).expect("layout");
    let packets_per_copy = layout.0 / 2;
    assert_eq!(found[0], packets_per_copy + 3);
    second_copy_plays_against_its_pictures("robust_splice.ts", ts);
}

/// robust_splice.ts with its discontinuity_indicator cleared: one timeline
/// with a 2 s step in both streams.
fn unmarked_splice() -> Vec<u8> {
    let mut ts = fixture!("robust_splice.ts").to_vec();
    for p in ts.chunks_exact_mut(TS_PACKET) {
        if p[3] & 0x20 != 0 && p[4] > 0 {
            p[5] &= !0x80;
        }
    }
    ts
}

/// The PTS of every audio PES (PID 0x101) from TS packet `from` on moved by
/// `by` ticks.
fn move_audio_pts(ts: &mut [u8], from: usize, by: i64) {
    for p in ts.chunks_exact_mut(TS_PACKET).skip(from) {
        let pid = (u16::from(p[1] & 0x1F) << 8) | u16::from(p[2]);
        if pid != 0x101 || p[1] & 0x40 == 0 {
            continue;
        }
        let o = 4 + if p[3] & 0x20 != 0 {
            1 + p[4] as usize
        } else {
            0
        };
        let b = &mut p[o..];
        if b[..3] != [0, 0, 1] || b[7] & 0x80 == 0 {
            continue;
        }
        let pts = (i64::from(b[9] >> 1) & 7) << 30
            | ((i64::from(b[10]) << 7 | i64::from(b[11]) >> 1) & 0x7FFF) << 15
            | ((i64::from(b[12]) << 7 | i64::from(b[13]) >> 1) & 0x7FFF);
        let v = (pts + by).rem_euclid(PTS_MODULUS);
        b[9] = (b[9] & 0xF0) | ((v >> 29) & 0x0E) as u8 | 1;
        b[10] = (v >> 22) as u8;
        b[11] = ((v >> 14) & 0xFE) as u8 | 1;
        b[12] = (v >> 7) as u8;
        b[13] = ((v << 1) & 0xFE) as u8 | 1;
    }
}

#[test]
fn an_unmarked_step_closes_up_against_the_pictures_all_the_same() {
    // No break: the video's pictures either side of the step play one after
    // the other, and so does the audio.
    let ts = unmarked_splice();
    assert!(breaks(&ts).is_empty());
    second_copy_plays_against_its_pictures("unmarked splice", &ts);
}

#[test]
fn audio_leading_the_picture_after_a_step_goes_with_that_picture() {
    // The second copy's audio 0.1 s earlier against its pictures: its first
    // frames fall in the 2 s the video has no picture for, just before the
    // picture after the step. They play against that picture, 4800 ticks
    // sooner than the second copy's audio did above (and what overlaps the
    // first copy's audio goes) — not 2 s on from the picture before the step.
    let mut ts = unmarked_splice();
    let half = ts.len() / TS_PACKET / 2;
    move_audio_pts(&mut ts, half, -9000);
    let (packets, gaps) = placed("unmarked splice, audio leading", &ts);
    assert!(gaps.is_empty(), "{gaps:?}");
    assert!(packets.iter().any(|p| p.0 == 1));
    for (copy, m, at) in packets {
        let want = if copy == 0 {
            m * 1024
        } else {
            76800 - 4800 + m * 1024
        };
        assert!(
            (at - want).abs() <= 512,
            "copy {copy} frame {m} at {at}, wanted {want}"
        );
    }
}

/// A packet on `pid` whose adaptation field carries `pcr` (90 kHz base), and
/// the discontinuity_indicator when `flagged`.
fn pcr_packet(pid: u16, pcr: Option<u64>, flagged: bool) -> [u8; TS_PACKET] {
    let mut p = [0xFFu8; TS_PACKET];
    p[0] = TS_SYNC;
    p[1] = ((pid >> 8) & 0x1F) as u8;
    p[2] = (pid & 0xFF) as u8;
    p[3] = 0x20; // adaptation field only
    p[4] = 183;
    p[5] = if flagged { 0x80 } else { 0 };
    if let Some(base) = pcr {
        p[5] |= 0x10;
        p[6] = (base >> 25) as u8;
        p[7] = (base >> 17) as u8;
        p[8] = (base >> 9) as u8;
        p[9] = (base >> 1) as u8;
        p[10] = ((base & 1) << 7) as u8 | 0x7E;
        p[11] = 0;
    }
    p
}

#[test]
fn a_break_is_a_marked_pcr_packet_or_a_pcr_that_jumps() {
    let second = 90_000u64;
    let packets = [
        // Marked before any PCR (a stream cut from a longer one): one time
        // base, not two.
        pcr_packet(0x100, Some(10 * second), true),
        pcr_packet(0x100, Some(11 * second), false),
        // Marked on another PID: not the program's clock.
        pcr_packet(0x101, None, true),
        // Nine seconds on: a step, not a jump.
        pcr_packet(0x100, Some(20 * second), false),
        // Back.
        pcr_packet(0x100, Some(5 * second), false),
        pcr_packet(0x100, Some(6 * second), false),
        // Marked, the clock stepping on as if nothing happened.
        pcr_packet(0x100, Some(7 * second), true),
        // Eleven seconds on.
        pcr_packet(0x100, Some(18 * second), false),
        // Across the 33-bit wrap, continuous: no break.
        pcr_packet(0x100, Some(PTS_MODULUS as u64 - second / 2), false),
        pcr_packet(0x100, Some(second / 2), false),
    ];
    let data: Vec<u8> = packets.concat();
    let layout = (packets.len(), TS_PACKET, 0);
    // The jump to just before the wrap is itself a break (it is more than
    // ten seconds on); the step across the wrap is not.
    assert_eq!(time_base_breaks(&data, layout, Some(0x100)), [4, 6, 7, 8]);
    assert!(time_base_breaks(&data, layout, None).is_empty());
}

#[test]
fn a_stream_is_cut_at_the_breaks_and_where_its_own_timestamps_jump() {
    let second = 90_000u64;
    let mut s = Segmenter::new(&[10]);
    let at = |breaks, jumps| Stretch { breaks, jumps };
    assert_eq!(
        s.place(0, Some(PTS_MODULUS as u64 - second)),
        (at(0, 0), Some(PTS_MODULUS - 90_000))
    );
    // Across the wrap: unwrapped, same stretch.
    assert_eq!(
        s.place(1, Some(second)),
        (at(0, 0), Some(PTS_MODULUS + 90_000))
    );
    // No PTS: same stretch, none placed.
    assert_eq!(s.place(2, None), (at(0, 0), None));
    // Eleven seconds on: its own jump.
    assert_eq!(s.place(3, Some(12 * second)), (at(0, 1), Some(12 * 90_000)));
    // Back half a second (a B picture's worth, and more): not a jump.
    assert_eq!(
        s.place(4, Some(12 * second - second / 2)),
        (at(0, 1), Some(12 * 90_000 - 45_000))
    );
    // Back two seconds: a jump.
    assert_eq!(s.place(5, Some(10 * second)), (at(0, 2), Some(10 * 90_000)));
    // Past the break at packet 10: a new time base, whatever its PTS.
    assert_eq!(
        s.place(11, Some(10 * second + 1)),
        (at(1, 0), Some(10 * 90_000 + 1))
    );
}

#[test]
fn a_stretch_after_a_break_starts_after_the_frames_the_output_presents() {
    // An HEVC stream opening on a CRA with three RASL pictures, which the
    // reader drops: eight pictures counted before the break, five presented.
    // Twenty AAC frames, one to a PES: ten before the break, ten after it on a
    // new clock, the first of them at the second stretch's first picture. That
    // picture is the output's sixth (5 x 3600 at 90 kHz: 9600 ticks), where
    // the audio after the break goes; the first stretch's audio runs to 10240,
    // so the first frame after the break overlaps it and goes.
    let pts = |i: i64| {
        if i < 10 {
            100_000 + i * 1920
        } else {
            500_000 + (i - 10) * 1920
        }
    };
    let audio = TsAudio {
        track: AudioTrack {
            codec: "aac".into(),
            samples: (0..20u8).map(|i| vec![i]).collect(),
            sample_rate: 48_000,
            channels: 1,
            asc: Vec::new(),
            codec_private: Vec::new(),
            timescale: 48_000,
            durations: vec![1024; 20],
        },
        first_pts: Some(100_000),
        pes: (0..20)
            .map(|i| (i * 10, Some(pts(i as i64) as u64), i * 2))
            .collect(),
        frame_starts: (0..20).map(|i| i * 10).collect(),
    };
    let stretch = |breaks| crate::ts::discontinuity::Stretch { breaks, jumps: 0 };
    let segments = [
        VideoSegment {
            stretch: stretch(0),
            frames_before: 0,
            ptses: (0..8).map(|k| 89_200 + k * 3600).collect(),
        },
        VideoSegment {
            stretch: stretch(1),
            frames_before: 8,
            ptses: (0..8).map(|k| 500_000 + k * 3600).collect(),
        },
    ];
    let clock = ProgramClock {
        video_delay: 0,
        audio_delay: 0,
    };
    let (track, gaps) = place_program_audio(Some(audio), &[19], &segments, 3, clock, 25.0);
    let track = track.expect("audio");
    assert!(gaps.is_empty(), "{gaps:?}");
    let mut at = 0i64;
    for (sample, &duration) in track.samples.iter().zip(&track.durations) {
        let m = i64::from(sample[0]);
        let want = if m < 10 {
            m * 1024
        } else {
            9600 + (m - 10) * 1024
        };
        assert!((at - want).abs() <= 512, "frame {m} at {at}, wanted {want}");
        at += i64::from(duration);
    }
    assert_eq!(track.samples.len(), 19);
}
