//! A transport stream's audio placed by its own timestamps.
//!
//! The reader hands the audio on as packets and durations, and every output
//! times it by the running sum of those durations. That is right while the
//! stream is whole. Two things in a real transport stream break it:
//!
//! - **A hole**: audio PES packets lost in reception, or never sent. The
//!   packets after it carry the PTSes they should play at; summed from the
//!   start, they played the length of the hole early, for the rest of the
//!   stream. The hole is kept as time — the packet before it lasts longer (the
//!   duration ffmpeg's MP4 muxer writes for it, and what every output's timing
//!   follows), and a decoded track is given that much silence ([`AudioGap`]) —
//!   but only as much of it as the video still has pictures for: the output
//!   presents the video's frames one after another, so a dropout that took
//!   both streams closes up in the video, and must in the audio too. Audio
//!   that overlaps what came before by more than half a frame (a PES sent
//!   twice) is dropped.
//! - **A time-base discontinuity** (a splice, two recordings joined; see
//!   [`super::discontinuity`]): the timestamps after it start again, so a hole
//!   or an overlap measured across it means nothing. The audio after one is
//!   placed against the video after it instead, at the picture its PTS names;
//!   what overlaps the audio before it is dropped, and a gap is kept, as above.
//!
//! Both measure the audio against the video's pictures ([`VideoTimeline`]): a
//! PTS plays where the output presents the picture nearest it, and the time
//! from that picture. A stream with neither — every packet within
//! half a frame of where the one before it ends — is left exactly as it was,
//! whatever its video's timing.

use std::collections::HashSet;

use super::audio::{AudioPes, TsAudio};
use super::clock::{PTS_HZ, PTS_MODULUS, ProgramClock, unwrap_pts};
use super::discontinuity::{Segmenter, Stretch};
use super::pictures::VideoSegment;
use crate::demux::AudioTrack;
use crate::edit::{AudioGap, rescale_round};

/// The video's pictures on the output's timeline, which the audio is measured
/// against.
#[derive(Debug, Clone, Copy)]
struct VideoTimeline<'a> {
    /// The video's stretches ([`super::pictures::count_frames`]), counting
    /// the frames the output presents.
    segments: &'a [VideoSegment],
    /// Where the video's first presented picture plays on the program's
    /// timeline (90 kHz).
    video_delay: u64,
    /// The output's frame rate: the output presents frame `n` at
    /// `video_delay + n / frame_rate`.
    frame_rate: f64,
}

impl<'a> VideoTimeline<'a> {
    /// The video's stretch `stretch`, when it has one with a timestamp.
    fn segment(&self, stretch: Stretch) -> Option<&'a VideoSegment> {
        let segments: &'a [VideoSegment] = self.segments;
        segments
            .iter()
            .find(|s| s.stretch == stretch && !s.ptses.is_empty())
            .filter(|_| self.frame_rate > 0.0)
    }

    /// Where `pts` (on `segment`'s timeline) plays on the program's timeline,
    /// 90 kHz: where the output presents the picture nearest it, and the time
    /// from that picture. The nearest, because across a gap in the video the
    /// output presents the pictures either side of it one after the other:
    /// audio just before the picture after a gap goes with that picture.
    fn place(&self, segment: &VideoSegment, pts: i64) -> f64 {
        let ptses = &segment.ptses;
        let pts = unwrap_pts(pts.rem_euclid(PTS_MODULUS) as u64, ptses[0]);
        let after = ptses.partition_point(|&v| v <= pts);
        let k = match (after.checked_sub(1), ptses.get(after)) {
            (Some(before), Some(&next)) if next - pts < pts - ptses[before] => after,
            (Some(before), _) => before,
            (None, _) => 0,
        };
        let frame = (segment.frames_before + k as u64) as f64;
        self.video_delay as f64
            + frame * f64::from(PTS_HZ) / self.frame_rate
            + (pts - ptses[k]) as f64
    }
}

/// The audio track placed on the program's timeline.
#[derive(Debug)]
struct PlacedAudio {
    /// The track: the packets kept, their durations carrying the holes.
    track: AudioTrack,
    /// The holes, for a decoded track to fill with silence.
    gaps: Vec<AudioGap>,
    /// Packets dropped as overlapping the audio before them.
    dropped: usize,
    /// Stretches after a time-base break placed against their video.
    rebased: usize,
}

/// A program's audio placed on its timeline: holes the video has pictures
/// for kept as time, the audio after a time-base break placed against the
/// video after it. `segments` are the video's stretches as
/// [`super::pictures::count_frames`] counts them, `rasl` the RASL pictures of
/// its first IRAP it counts and the output does not present, `clock` where the
/// two streams start, and `frame_rate` the output's. The track, and its holes
/// for a decoded track to fill.
pub(super) fn place_program_audio(
    audio: Option<TsAudio>,
    breaks: &[usize],
    segments: &[VideoSegment],
    rasl: u64,
    clock: ProgramClock,
    frame_rate: f64,
) -> (Option<AudioTrack>, Vec<AudioGap>) {
    let Some(audio) = audio else {
        return (None, Vec::new());
    };
    // The RASL pictures are the first stretch's, and the output presents none
    // of them: a stretch after it starts that many frames sooner. (They lead
    // the first stretch in presentation order, which the audio's first frame
    // is placed on by the program clock, not by its pictures.)
    let segments: Vec<VideoSegment> = segments
        .iter()
        .map(|s| VideoSegment {
            stretch: s.stretch,
            frames_before: s.frames_before.saturating_sub(rasl),
            ptses: s.ptses.clone(),
        })
        .collect();
    let timeline = VideoTimeline {
        segments: &segments,
        video_delay: clock.video_delay,
        frame_rate,
    };
    let placed = place_audio(audio, breaks, &timeline, clock.audio_delay);
    if !breaks.is_empty() || !placed.gaps.is_empty() || placed.dropped > 0 {
        let rate = f64::from(placed.track.timescale.max(1));
        tracing::info!(
            time_base_breaks = breaks.len(),
            video_stretches = segments.len(),
            audio_stretches_placed_on_video = placed.rebased,
            holes = placed.gaps.len(),
            hole_seconds = placed.gaps.iter().map(|g| g.ticks as f64).sum::<f64>() / rate,
            overlapping_packets_dropped = placed.dropped,
            "TS: audio placed by its timestamps against the video's pictures: holes kept as \
             time (silence when decoded), audio after a time-base discontinuity placed against \
             the video after it"
        );
    }
    (Some(placed.track), placed.gaps)
}

/// Place `audio` on the program's timeline: the program's time-base `breaks`,
/// the `video` it is measured against, and where its first frame plays
/// (`audio_delay`, 90 kHz).
fn place_audio(
    audio: TsAudio,
    breaks: &[usize],
    video: &VideoTimeline,
    audio_delay: u64,
) -> PlacedAudio {
    let TsAudio {
        track,
        pes,
        frame_starts,
        ..
    } = audio;
    let n = track.samples.len();
    let stamps = frame_stamps(
        &pes,
        &frame_starts,
        &track.durations,
        track.timescale,
        breaks,
    );
    if n == 0 || stamps.len() != n {
        return PlacedAudio {
            track,
            gaps: Vec::new(),
            dropped: 0,
            rebased: 0,
        };
    }
    let rate = track.timescale;
    // 90 kHz -> the track's ticks, and a packet's duration the other way.
    let to_ticks = |t: f64| (t * f64::from(rate) / f64::from(PTS_HZ)).round() as i64;
    let length = |d: u32| rescale_round(u64::from(d), PTS_HZ, rate) as i64;

    let mut samples = Vec::with_capacity(n);
    let mut durations: Vec<u32> = Vec::with_capacity(n);
    let mut gaps = Vec::new();
    let (mut dropped, mut rebased) = (0usize, 0usize);
    // The audio stretch being walked, and the video stretch it is measured
    // against.
    let mut current: Option<Stretch> = None;
    let mut against: Option<&VideoSegment> = None;
    // Added to a stamp to put it on the timeline the stretch is walked on
    // (non-zero after a jump in the audio's own timestamps, which the walk
    // carries on across), and where on it the next frame is due, 90 kHz.
    let mut shift = 0i64;
    let mut expect = 0i64;
    // Where the next packet is due on the program's timeline, track ticks.
    let mut due = to_ticks(audio_delay as f64);
    // A stretch opened after a break, placed against its video: its packets
    // go where the video says until one is kept.
    let mut placing = false;
    for (i, (packet, &duration)) in track.samples.into_iter().zip(&track.durations).enumerate() {
        let (stretch, stamp) = stamps[i];
        let nominal = i64::from(duration);
        if current != Some(stretch) {
            match (current.replace(stretch), video.segment(stretch)) {
                // The first frame plays where the program clock put it.
                (None, segment) => {
                    against = segment;
                    shift = 0;
                    expect = stamp;
                }
                (Some(_), Some(segment)) => {
                    against = Some(segment);
                    shift = 0;
                    placing = true;
                    rebased += 1;
                }
                // A jump in the audio's own timestamps, its video running on:
                // the walk carries on, on the timeline before the jump.
                (Some(before), None) if before.breaks == stretch.breaks => {
                    shift = unwrap_pts(stamp.rem_euclid(PTS_MODULUS) as u64, expect) - stamp;
                }
                // A break with no video to place it against: gap-free.
                (Some(_), None) => {
                    against = None;
                    shift = 0;
                    expect = stamp;
                }
            }
        }
        let t = stamp + shift;
        // How far after `due` the packet plays, track ticks.
        let off = match against.filter(|_| placing) {
            Some(segment) => to_ticks(video.place(segment, t)) - due,
            None => {
                let own = t - expect;
                if own < -length(duration) / 2 {
                    dropped += 1;
                    continue;
                }
                if own > length(duration) / 2 {
                    // A hole in the audio: as long as the video's pictures
                    // across it last.
                    to_ticks(
                        against.map_or(own as f64, |s| video.place(s, t) - video.place(s, expect)),
                    )
                } else {
                    0
                }
            }
        };
        if off < -nominal / 2 {
            dropped += 1;
            continue;
        }
        if off > nominal / 2
            && let Some(last) = durations.last_mut()
        {
            let hole = u32::try_from(off).unwrap_or(u32::MAX);
            *last = last.saturating_add(hole);
            gaps.push(AudioGap {
                after_packet: samples.len() - 1,
                ticks: u64::from(hole),
            });
            due += i64::from(hole);
        }
        placing = false;
        samples.push(packet);
        durations.push(duration);
        due += nominal;
        expect = t + length(duration);
    }
    PlacedAudio {
        track: AudioTrack {
            samples,
            durations,
            ..track
        },
        gaps,
        dropped,
        rebased,
    }
}

/// Each frame's stretch and PTS (90 kHz, on its stretch's unwrapped
/// timeline). A PES packet's PTS is that of the first frame commencing in it
/// (ISO/IEC 13818-1 §2.4.3.7); a frame after it is stamped where the frame
/// before it ends, and one at the head of a stretch where the frame after it
/// starts, less its own length. A stretch in which no frame is stamped is
/// taken as part of the one before it. Empty when no frame is stamped.
fn frame_stamps(
    pes: &[AudioPes],
    frame_starts: &[usize],
    durations: &[u32],
    rate: u32,
    breaks: &[usize],
) -> Vec<(Stretch, i64)> {
    let mut segmenter = Segmenter::new(breaks);
    let placed: Vec<(Stretch, Option<i64>)> = pes
        .iter()
        .map(|&(_, pts, packet)| segmenter.place(packet, pts))
        .collect();
    let mut frames: Vec<(Stretch, Option<i64>)> = frame_starts
        .iter()
        .enumerate()
        .map(|(i, &start)| {
            let Some(p) = pes
                .partition_point(|&(offset, _, _)| offset <= start)
                .checked_sub(1)
            else {
                return (Stretch::default(), None);
            };
            let first_in_pes = i == 0 || frame_starts[i - 1] < pes[p].0;
            (placed[p].0, placed[p].1.filter(|_| first_in_pes))
        })
        .collect();
    let stamped: HashSet<Stretch> = frames
        .iter()
        .filter_map(|&(s, pts)| pts.map(|_| s))
        .collect();
    let Some(first) = frames.iter().position(|f| f.1.is_some()) else {
        return Vec::new();
    };
    for i in 0..frames.len() {
        if !stamped.contains(&frames[i].0) {
            frames[i].0 = if i == 0 {
                frames[first].0
            } else {
                frames[i - 1].0
            };
        }
    }
    let length = |i: usize| rescale_round(u64::from(durations[i]), PTS_HZ, rate) as i64;
    for i in 1..frames.len() {
        if frames[i].1.is_none()
            && frames[i].0 == frames[i - 1].0
            && let Some(before) = frames[i - 1].1
        {
            frames[i].1 = Some(before + length(i - 1));
        }
    }
    for i in (0..frames.len().saturating_sub(1)).rev() {
        if frames[i].1.is_none()
            && frames[i].0 == frames[i + 1].0
            && let Some(after) = frames[i + 1].1
        {
            frames[i].1 = Some(after - length(i));
        }
    }
    frames
        .into_iter()
        .map(|(s, pts)| (s, pts.unwrap_or_default()))
        .collect()
}
