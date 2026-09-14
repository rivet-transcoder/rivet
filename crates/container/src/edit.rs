//! Presentation edits: which of a track's samples a player shows, and when.
//!
//! A container can say that a track's samples are not all meant to be seen, or
//! not from time zero. MP4 and MOV say it with an edit list (`edts`/`elst`,
//! ISO/IEC 14496-12 §8.6.6), and three shapes of it are everywhere:
//!
//! - **a hidden lead-in**: `ffmpeg -ss T -i in.mp4 -c copy out.mp4` cannot cut
//!   inside a GOP, so it keeps the whole GOP before `T` and writes an edit whose
//!   `media_time` starts presentation at `T`. The frames before it are decoded
//!   (the frames after them reference them) and never shown.
//! - **encoder priming**: an AAC encoder's first 1024 (or 2048) output samples
//!   are silence the decoder needs to warm up; the audio edit starts
//!   presentation after them.
//! - **a late start**: an empty edit (`media_time = -1`) puts time before a
//!   track's first sample — audio recorded after the video started, or
//!   `-itsoffset`.
//!
//! A transcode that ignores the edit shows the hidden frames, plays the priming
//! and loses the delay: extra frames at the start and an audio/video offset.
//!
//! The MP4 demuxer reduces an edit list to the types here (anything it cannot
//! represent is refused by name there); the pipeline honours them; the muxers
//! write the part that survives back out as a [`TrackEdit`].

/// What a source's video track presents, counted in decoded frames.
///
/// Decoded frames come out of a decoder in display order and are counted from
/// zero; an edit hides some of them at the start and may stop presenting before
/// the last. The hidden frames are named by their decoded index rather than
/// counted, because they need not be the first few: an edit hides the pictures
/// of the *samples* before it, and when a container's timestamps do not say how
/// the stream reorders its pictures (a raw elementary stream remuxed without
/// composition offsets) those pictures are spread through the first GOPs in
/// display order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VideoPresentation {
    /// Decoded-frame indices the edit hides, ascending, every one of them
    /// before the last frame it presents.
    pub hidden: Vec<u64>,
    /// Frames the edit presents: the decoded frames that are not hidden, up
    /// to this many.
    pub presented: u64,
    /// Every sample in the track: hidden, presented, and any after the edit ends.
    pub samples: u64,
    /// Empty time before the first presented frame, in ticks of
    /// [`delay_timescale`](Self::delay_timescale).
    pub delay_ticks: u64,
    /// Ticks per second of [`delay_ticks`](Self::delay_ticks).
    pub delay_timescale: u32,
}

impl VideoPresentation {
    /// A presentation hiding the first `hidden` decoded frames and presenting
    /// the `presented` after them, from time zero.
    pub fn leading(hidden: u64, presented: u64, samples: u64) -> Self {
        Self { hidden: (0..hidden).collect(), presented, samples, delay_ticks: 0, delay_timescale: 1 }
    }

    /// True when the edit changes nothing: every frame presented, from time zero.
    pub fn is_identity(&self) -> bool {
        self.hidden.is_empty() && self.presented == self.samples && self.delay_ticks == 0
    }

    /// The delay in ticks of `timescale`, rounded to the nearest tick.
    pub fn delay_in(&self, timescale: u32) -> u64 {
        rescale_round(self.delay_ticks, timescale, self.delay_timescale)
    }

    /// The decoded frame with absolute index `decoded` (0 = the first frame
    /// the track decodes to), placed on the presentation.
    pub fn place(&self, decoded: u64) -> FramePlace {
        let before = self.hidden.partition_point(|&h| h < decoded);
        if self.hidden.get(before) == Some(&decoded) {
            return FramePlace::Hidden;
        }
        let presented = decoded - before as u64;
        if presented >= self.presented { FramePlace::PastEnd } else { FramePlace::Presented(presented) }
    }

    /// The presented index of the decoded frame `decoded` when no hidden frame
    /// comes at or after it — the condition for a decode that starts at that
    /// frame to need nothing from before it to place its frames. `None`
    /// otherwise, or when the frame is past the end.
    pub fn presented_index_after_hidden(&self, decoded: u64) -> Option<u64> {
        if self.hidden.last().is_some_and(|&h| h >= decoded) {
            return None;
        }
        match self.place(decoded) {
            FramePlace::Presented(p) => Some(p),
            _ => None,
        }
    }
}

/// Where one decoded frame falls on a [`VideoPresentation`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FramePlace {
    /// Hidden by the edit: decoded for its references, not shown.
    Hidden,
    /// Shown, at this presented index.
    Presented(u64),
    /// After the edit ends: nothing from here on is shown.
    PastEnd,
}

/// What a source's audio track presents, in ticks of the track's own timescale
/// ([`AudioTrack::timescale`](crate::demux::AudioTrack::timescale)), on the
/// timeline where the first sample starts at zero and every sample follows the
/// last (the running sum of the track's durations).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AudioEdit {
    /// Empty time before the first presented sample.
    pub delay: u64,
    /// First presented media time.
    pub media_start: u64,
    /// End of the presented media, exclusive; `None` runs to the end of the track.
    pub media_end: Option<u64>,
}

impl AudioEdit {
    /// True when the edit changes nothing for a track `total` ticks long.
    pub fn is_identity(&self, total: u64) -> bool {
        self.delay == 0 && self.media_start == 0 && self.media_end.is_none_or(|e| e >= total)
    }
}

/// An edit to write on an output track, in ticks of that track's timescale.
///
/// The muxers write nothing for the identity (`Default`), so an output with no
/// edit is byte-for-byte what it was before edits existed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TrackEdit {
    /// Empty time before the first presented sample.
    pub delay: u64,
    /// Media time of the first presented sample: how far into the written
    /// samples presentation starts (decoder preroll, priming, a partial packet).
    pub media_time: u64,
    /// How long the track presents from `media_time`; `None` = to the end of
    /// the written samples.
    pub duration: Option<u64>,
}

impl TrackEdit {
    /// True when there is nothing to write.
    pub fn is_identity(&self) -> bool {
        self.delay == 0 && self.media_time == 0 && self.duration.is_none()
    }

    /// This edit over written samples `total` ticks long, narrowed to the
    /// presentation window `[start, end)` (ticks from presentation time zero,
    /// `end = None` = open) — as an [`AudioEdit`] over the same samples, ready
    /// for [`cut_audio_packets`].
    pub fn window(&self, total: u64, start: u64, end: Option<u64>) -> AudioEdit {
        // Presentation p shows media m = media_time + (p - delay), for p >= delay.
        let own_end = self.duration.map_or(total, |d| self.media_time.saturating_add(d)).min(total);
        let media_at = |p: u64| self.media_time.saturating_add(p.saturating_sub(self.delay)).min(own_end);
        let media_start = media_at(start);
        let media_end = end.map_or(own_end, |e| media_at(e).max(media_start));
        AudioEdit { delay: self.delay.saturating_sub(start), media_start, media_end: Some(media_end) }
    }
}

/// How much of an audio track ahead of the first presented sample a decoder
/// needs to have seen for that sample to come out right.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AudioPreroll {
    /// At least this many whole packets.
    pub packets: usize,
    /// And at least this many ticks.
    pub ticks: u64,
}

impl AudioPreroll {
    /// The preroll for a codec label, at the track's `timescale`.
    ///
    /// AAC, AC-3 and E-AC-3 are transform codecs whose frames overlap their
    /// neighbours by half a window: a frame decoded without the one before it
    /// is wrong for its first half. One packet ahead fixes that (the MP4
    /// sample group for AAC says the same: `roll_distance = -1`). DTS's
    /// subband filter bank has a delay shorter than a frame, so one packet
    /// covers it too. Opus asks for 80 ms (RFC 7845 §4.2), which is several
    /// 20 ms packets.
    pub fn for_codec(codec: &str, timescale: u32) -> Self {
        if codec.eq_ignore_ascii_case("opus") {
            Self { packets: 0, ticks: rescale_round(80, timescale, 1000) }
        } else {
            Self { packets: 1, ticks: 0 }
        }
    }
}

/// The packets of an audio track to keep for an edit, and the output edit that
/// presents exactly the edit's samples from them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AudioCut {
    /// Indices of the packets to keep.
    pub packets: std::ops::Range<usize>,
    /// The edit to write on the output track holding exactly those packets.
    pub edit: TrackEdit,
}

/// Cut an audio track (packet `durations`, in ticks) to an [`AudioEdit`].
///
/// Whole packets that end before the edit starts — beyond `preroll` — and
/// packets that start at or after it ends are dropped; the samples inside the
/// kept packets that the edit does not present are left for the output edit to
/// hide, so the cut is exact to the sample without re-encoding.
pub fn cut_audio_packets(durations: &[u32], edit: &AudioEdit, preroll: AudioPreroll) -> AudioCut {
    let mut starts = Vec::with_capacity(durations.len() + 1);
    let mut at = 0u64;
    for &d in durations {
        starts.push(at);
        at += u64::from(d);
    }
    let total = at;
    starts.push(total);
    let end = edit.media_end.map_or(total, |e| e.min(total));
    let start = edit.media_start.min(end);
    if start >= end {
        return AudioCut { packets: 0..0, edit: TrackEdit { delay: edit.delay, ..TrackEdit::default() } };
    }

    // The packet holding the first presented sample, then back over the preroll.
    let first = (0..durations.len()).find(|&i| starts[i + 1] > start).unwrap_or(durations.len());
    let mut keep_from = first;
    let mut backed = 0usize;
    while keep_from > 0 && (backed < preroll.packets || start - starts[keep_from] < preroll.ticks) {
        keep_from -= 1;
        backed += 1;
    }
    // The first packet starting at or after the end is the first one dropped.
    let keep_to = (0..durations.len()).find(|&i| starts[i] >= end).unwrap_or(durations.len());

    let media_time = start - starts[keep_from];
    let presented = end - start;
    let kept_ticks = starts[keep_to] - starts[keep_from];
    let duration = if media_time + presented >= kept_ticks { None } else { Some(presented) };
    AudioCut { packets: keep_from..keep_to, edit: TrackEdit { delay: edit.delay, media_time, duration } }
}

/// `value * to / from`, rounded to the nearest integer (ties away from zero) —
/// how ffmpeg (`av_rescale`) moves an edit between timescales, so the frames an
/// edit keeps here are the frames it keeps there. A zero `from` gives zero.
pub fn rescale_round(value: u64, to: u32, from: u32) -> u64 {
    if from == 0 {
        return 0;
    }
    let from = u128::from(from);
    ((u128::from(value) * u128::from(to) + from / 2) / from) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rescale_rounds_to_nearest_like_ffmpeg() {
        // elst in a 1000-tick movie clock, media at 48 kHz: 21.333 ms of priming.
        assert_eq!(rescale_round(1024, 1000, 48_000), 21);
        assert_eq!(rescale_round(1800, 1_200_000, 1000), 2_160_000);
        assert_eq!(rescale_round(3, 1, 2), 2, "ties go away from zero");
        assert_eq!(rescale_round(5, 7, 0), 0);
    }

    #[test]
    fn a_hidden_lead_in_places_frames_after_it() {
        // A trim: three frames before the edit, 45 presented.
        let p = VideoPresentation::leading(3, 45, 48);
        assert!(!p.is_identity());
        assert_eq!(p.place(0), FramePlace::Hidden);
        assert_eq!(p.place(2), FramePlace::Hidden);
        assert_eq!(p.place(3), FramePlace::Presented(0));
        assert_eq!(p.place(47), FramePlace::Presented(44));
        assert_eq!(p.place(48), FramePlace::PastEnd);
    }

    #[test]
    fn hidden_pictures_spread_through_display_order_are_skipped_where_they_are() {
        // WPP_C remuxed with no composition offsets: the pictures of samples
        // 0..3 come out of the decoder at display positions 0, 4 and 8.
        let p = VideoPresentation { hidden: vec![0, 4, 8], ..VideoPresentation::leading(0, 45, 48) };
        let placed: Vec<FramePlace> = (0..49).map(|d| p.place(d)).collect();
        assert_eq!(placed[0], FramePlace::Hidden);
        assert_eq!(placed[1], FramePlace::Presented(0));
        assert_eq!(placed[3], FramePlace::Presented(2));
        assert_eq!(placed[4], FramePlace::Hidden);
        assert_eq!(placed[5], FramePlace::Presented(3));
        assert_eq!(placed[8], FramePlace::Hidden);
        assert_eq!(placed[9], FramePlace::Presented(6));
        assert_eq!(placed[47], FramePlace::Presented(44));
        assert_eq!(placed[48], FramePlace::PastEnd);
        assert_eq!(placed.iter().filter(|f| matches!(f, FramePlace::Presented(_))).count(), 45);
        // A decode starting at frame 9 has every hidden frame behind it; one
        // starting at 6 does not.
        assert_eq!(p.presented_index_after_hidden(9), Some(6));
        assert_eq!(p.presented_index_after_hidden(6), None);
    }

    #[test]
    fn an_edit_that_ends_early_stops_presenting() {
        let p = VideoPresentation::leading(0, 10, 30);
        assert_eq!(p.place(9), FramePlace::Presented(9));
        assert_eq!(p.place(10), FramePlace::PastEnd);
        assert!(!p.is_identity());
        let whole = VideoPresentation::leading(0, 30, 30);
        assert!(whole.is_identity());
        assert!(!VideoPresentation { delay_ticks: 1, ..whole.clone() }.is_identity());
        assert_eq!(VideoPresentation { delay_ticks: 500, delay_timescale: 1000, ..whole }.delay_in(90_000), 45_000);
    }

    #[test]
    fn aac_priming_keeps_the_priming_packet_as_preroll_and_hides_it() {
        // 1024 ticks of priming at the start of 1024-tick packets: packet 0 is
        // all priming but stays (it is packet 1's preroll); the output edit
        // starts presentation 1024 ticks in — ffmpeg's own `-c copy`.
        let durations = [1024u32; 10];
        let edit = AudioEdit { delay: 0, media_start: 1024, media_end: None };
        let cut = cut_audio_packets(&durations, &edit, AudioPreroll::for_codec("aac", 48_000));
        assert_eq!(cut.packets, 0..10);
        assert_eq!(cut.edit, TrackEdit { delay: 0, media_time: 1024, duration: None });
    }

    #[test]
    fn a_trimmed_audio_start_drops_whole_packets_before_the_preroll() {
        // Presentation starts 5000 ticks in: packet 4 (4096..5120) holds it,
        // packet 3 is its preroll, packets 0..3 go. 5000 - 3072 = 1928 ticks
        // of the kept packets are hidden by the output edit.
        let durations = [1024u32; 10];
        let edit = AudioEdit { delay: 0, media_start: 5000, media_end: None };
        let cut = cut_audio_packets(&durations, &edit, AudioPreroll::for_codec("aac", 48_000));
        assert_eq!(cut.packets, 3..10);
        assert_eq!(cut.edit.media_time, 1928);
        assert_eq!(cut.edit.duration, None);
        // Without preroll the cut starts at the packet itself.
        let bare = cut_audio_packets(&durations, &edit, AudioPreroll { packets: 0, ticks: 0 });
        assert_eq!(bare.packets, 4..10);
        assert_eq!(bare.edit.media_time, 904);
    }

    #[test]
    fn an_audio_end_inside_a_packet_is_kept_and_cut_by_the_edit_duration() {
        let durations = [1024u32; 10];
        let edit = AudioEdit { delay: 0, media_start: 1024, media_end: Some(6000) };
        let cut = cut_audio_packets(&durations, &edit, AudioPreroll::for_codec("aac", 48_000));
        // 6000 is inside packet 5 (5120..6144): packets 0..6 kept.
        assert_eq!(cut.packets, 0..6);
        assert_eq!(cut.edit, TrackEdit { delay: 0, media_time: 1024, duration: Some(4976) });
        // An end on a packet boundary needs no duration.
        let on_boundary = AudioEdit { media_end: Some(6144), ..edit };
        let cut = cut_audio_packets(&durations, &on_boundary, AudioPreroll::for_codec("aac", 48_000));
        assert_eq!(cut.packets, 0..6);
        assert_eq!(cut.edit.duration, None);
    }

    #[test]
    fn opus_preroll_is_eighty_milliseconds_of_packets() {
        let durations = [960u32; 20];
        let edit = AudioEdit { delay: 0, media_start: 9600, media_end: None };
        let cut = cut_audio_packets(&durations, &edit, AudioPreroll::for_codec("opus", 48_000));
        // 3840 ticks back from 9600 is 5760 = packet 6.
        assert_eq!(cut.packets, 6..20);
        assert_eq!(cut.edit.media_time, 3840);
    }

    #[test]
    fn a_delay_carries_through_and_the_identity_changes_nothing() {
        let durations = [1024u32; 4];
        let delayed = AudioEdit { delay: 24_000, media_start: 0, media_end: None };
        let cut = cut_audio_packets(&durations, &delayed, AudioPreroll::for_codec("aac", 48_000));
        assert_eq!(cut.packets, 0..4);
        assert_eq!(cut.edit, TrackEdit { delay: 24_000, media_time: 0, duration: None });
        let identity = AudioEdit { delay: 0, media_start: 0, media_end: Some(4096) };
        assert!(identity.is_identity(4096));
        let cut = cut_audio_packets(&durations, &identity, AudioPreroll::for_codec("aac", 48_000));
        assert_eq!(cut.packets, 0..4);
        assert!(cut.edit.is_identity());
        assert!(!delayed.is_identity(4096));
    }

    #[test]
    fn an_edit_presenting_nothing_keeps_no_packets() {
        let edit = AudioEdit { delay: 7, media_start: 9000, media_end: Some(9000) };
        let cut = cut_audio_packets(&[1024; 4], &edit, AudioPreroll::for_codec("aac", 48_000));
        assert!(cut.packets.is_empty());
        assert_eq!(cut.edit.delay, 7);
    }

    #[test]
    fn a_user_trim_composes_with_an_output_edit() {
        // Written: priming 1024 hidden, 0.5 s (24000 ticks) delay.
        let e = TrackEdit { delay: 24_000, media_time: 1024, duration: None };
        // Trim from 1 s: past the delay by 24000, so 24000 further into the media.
        assert_eq!(e.window(96_000, 48_000, None), AudioEdit { delay: 0, media_start: 25_024, media_end: Some(96_000) });
        // Trim from 0.25 s to 1 s: 0.25 s of the delay is left, media ends 24000 in.
        assert_eq!(
            e.window(96_000, 12_000, Some(48_000)),
            AudioEdit { delay: 12_000, media_start: 1024, media_end: Some(25_024) }
        );
        // A trim ending inside the delay presents no media.
        let w = e.window(96_000, 0, Some(10_000));
        assert_eq!(w.media_start, 1024);
        assert_eq!(w.media_end, Some(1024));
        // The identity edit windows to the plain trim.
        assert_eq!(
            TrackEdit::default().window(96_000, 4_800, Some(9_600)),
            AudioEdit { delay: 0, media_start: 4_800, media_end: Some(9_600) }
        );
    }
}
