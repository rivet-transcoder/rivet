//! The program clock: where each selected stream starts on the program's one
//! time base.
//!
//! Every PES timestamp in a program counts the same 90 kHz clock (ISO/IEC
//! 13818-1 §2.4.2), so the gap between the video's first picture and the
//! audio's first frame is a fact of the source — 21 ms on an ffmpeg-muxed
//! H.264 + AAC transport stream, a second on one cut mid-GOP. A reader that
//! starts each stream at its own first timestamp loses it, and the output
//! plays the audio that much early or late.
//!
//! The base is the earliest first timestamp among the selected streams. Each
//! stream starts late by how far its own first timestamp is past the base:
//! the video by its first presented picture's, the audio by its first frame's.
//! The outputs already write both (an MP4 empty edit, the first CMAF `tfdt`).
//!
//! The timestamps are 33-bit and wrap every 2^33 ticks (26.5 hours), which a
//! long broadcast capture crosses. Each is unwrapped against a reference —
//! the video's first timestamp, or the one before it in the same stream — so
//! a start either side of the wrap, and a wrap anywhere inside the stream,
//! keeps the order and the distance.

use crate::edit::{AudioEdit, VideoPresentation, rescale_round};

/// PES timestamps count a 90 kHz clock modulo 2^33 (ISO/IEC 13818-1
/// §2.4.3.7).
pub(super) const PTS_MODULUS: i64 = 1 << 33;

/// The PES clock's ticks per second.
pub(super) const PTS_HZ: u32 = 90_000;

/// `pts` on a timeline that runs on through the wrap: the value congruent to
/// it modulo 2^33 that is nearest `reference` (itself already on that
/// timeline). Two timestamps less than 2^32 ticks (13.25 hours) apart keep
/// their order and their distance whichever side of a wrap each falls.
pub(super) fn unwrap_pts(pts: u64, reference: i64) -> i64 {
    let ahead = (pts as i64 - reference).rem_euclid(PTS_MODULUS);
    if ahead >= PTS_MODULUS / 2 {
        reference + ahead - PTS_MODULUS
    } else {
        reference + ahead
    }
}

/// A stream's timestamps in stream order, each unwrapped against the one
/// before it: a wrap anywhere in the stream is crossed, and the first
/// timestamp is taken as it is.
#[derive(Debug, Default, Clone, Copy)]
pub(super) struct PtsUnwrapper {
    last: Option<i64>,
}

impl PtsUnwrapper {
    /// The next timestamp, unwrapped.
    pub(super) fn unwrap(&mut self, pts: u64) -> i64 {
        let out = match self.last {
            Some(last) => unwrap_pts(pts, last),
            None => pts as i64,
        };
        self.last = Some(out);
        out
    }
}

/// Where the video starts, from the scan over the head of the stream, on the
/// timeline of the first timestamp the scan read (`reference`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct VideoStart {
    /// The first video timestamp the scan read, as the stream has it — what
    /// the others, and the audio's, are unwrapped against.
    pub(super) reference: u64,
    /// The earliest timestamp the stream opens with: the first presented
    /// picture's, or earlier when the stream opens mid-GOP with access units
    /// that are dropped (they are part of the program's start all the same —
    /// the audio beside them plays).
    pub(super) earliest: i64,
    /// The first picture a decoder presents.
    pub(super) first_presented: i64,
}

/// Where the selected streams start against the program's base, 90 kHz.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(super) struct ProgramClock {
    /// The video's first presented picture past the base.
    pub(super) video_delay: u64,
    /// The audio's first frame past the base.
    pub(super) audio_delay: u64,
}

impl ProgramClock {
    /// The streams placed against each other: the base is the earlier of the
    /// video's earliest timestamp and the audio's first frame. With no video
    /// timestamp there is nothing to place the audio against, and both start
    /// at zero, as they did before the program clock was read; so does the
    /// audio when its first frame's timestamp is unknown.
    pub(super) fn new(video: Option<&VideoStart>, audio_first: Option<u64>) -> Self {
        let Some(video) = video else {
            return Self::default();
        };
        let audio = audio_first.map(|pts| unwrap_pts(pts, video.reference as i64));
        let base = audio.map_or(video.earliest, |a| a.min(video.earliest));
        Self {
            video_delay: (video.first_presented - base).max(0) as u64,
            audio_delay: audio.map_or(0, |a| (a - base) as u64),
        }
    }

    /// The video's late start as the presentation every output writes (an
    /// MP4 empty edit, the first CMAF `tfdt`); `None` for none. It hides
    /// nothing and bounds nothing (`presented` / `samples` are unknown until
    /// the stream is drained, so `u64::MAX`).
    pub(super) fn video_presentation(&self) -> Option<VideoPresentation> {
        (self.video_delay > 0).then_some(VideoPresentation {
            hidden: Vec::new(),
            presented: u64::MAX,
            samples: u64::MAX,
            delay_ticks: self.video_delay,
            delay_timescale: PTS_HZ,
        })
    }

    /// The audio's late start as an edit on its own timescale (`timescale`
    /// ticks a second), rounded to the nearest tick; `None` for none.
    pub(super) fn audio_edit(&self, timescale: u32) -> Option<AudioEdit> {
        let delay = rescale_round(self.audio_delay, timescale, PTS_HZ);
        (delay > 0).then_some(AudioEdit {
            delay,
            media_start: 0,
            media_end: None,
        })
    }
}
