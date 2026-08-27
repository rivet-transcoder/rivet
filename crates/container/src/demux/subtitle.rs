//! Text subtitle extraction, for `-c:s copy`-style passthrough.
//!
//! Only **text** subtitles are carried. MP4's native subtitle format is `tx3g`
//! (3GPP timed text, what ffmpeg calls `mov_text`), which holds UTF-8 strings
//! with optional styling — so SRT, ASS/SSA, and WebVTT all convert into it, but
//! *bitmap* formats (PGS, VobSub, DVB) have no representation and are dropped
//! with a warning rather than silently mangled.
//!
//! Cue timing is carried as `(start, duration)` in the track's timescale. Gaps
//! between cues are the muxer's problem: `tx3g` requires a continuous timeline,
//! so it fills the holes with empty samples ([`crate::mux::subtitle_track`]).
//!
//! Every text track the source carries comes out, in source order, so the
//! caller can pick by language. The timeline operations on
//! [`SubtitleTrack`] — [`window`](SubtitleTrack::window),
//! [`shifted`](SubtitleTrack::shifted), [`append`](SubtitleTrack::append) —
//! are what a trim or a splice needs to re-base cues the way the audio
//! samples are re-based: clip the cues to the kept range, move them onto the
//! joined timeline, and concatenate.

use std::io::Cursor;

use matroska_demuxer::{Frame as MkvFrame, MatroskaFile, TrackType as MkvTrackType};

/// One text subtitle cue.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubtitleCue {
    /// Start time in the track's `timescale` ticks.
    pub start: u64,
    /// On-screen duration in `timescale` ticks. Always ≥ 1.
    pub duration: u32,
    /// The cue text, already stripped of any markup the source format wrapped
    /// it in (see [`strip_markup`]). UTF-8, may contain newlines.
    pub text: String,
}

impl SubtitleCue {
    /// Exclusive end time in the track's `timescale` ticks.
    pub fn end(&self) -> u64 {
        self.start + self.duration as u64
    }
}

/// A text subtitle track extracted for passthrough.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubtitleTrack {
    /// Source format label, for reporting: `subrip` / `ass` / `webvtt` / `tx3g`.
    pub codec: String,
    /// Cues in presentation order.
    pub cues: Vec<SubtitleCue>,
    /// Ticks per second for `start` / `duration`.
    pub timescale: u32,
    /// ISO-639-2 language, or `"und"`.
    pub language: String,
}

/// The timescale every re-based track is normalised to: milliseconds, which is
/// what Matroska hands us and more than a subtitle needs.
pub const REBASE_TIMESCALE: u32 = 1_000;

impl SubtitleTrack {
    /// End time of the last cue, in `timescale` ticks — the track's duration.
    pub fn end_time(&self) -> u64 {
        self.cues.iter().map(SubtitleCue::end).max().unwrap_or(0)
    }

    /// Ticks for `seconds` on this track's timescale, rounded to nearest.
    pub fn ticks(&self, seconds: f64) -> u64 {
        (seconds.max(0.0) * self.timescale.max(1) as f64).round() as u64
    }

    /// The cues that overlap `[start, end)` (ticks), clipped to the window and
    /// moved so that `start` becomes 0 — the subtitle half of a trim. `None`
    /// for `end` keeps everything after `start`. A cue that straddles a bound
    /// is cut at the bound rather than dropped: the viewer still sees the
    /// words that were on screen when the kept range began.
    pub fn window(&self, start: u64, end: Option<u64>) -> SubtitleTrack {
        let cues = self
            .cues
            .iter()
            .filter_map(|c| {
                let s = c.start.max(start);
                let e = match end {
                    Some(end) => c.end().min(end),
                    None => c.end(),
                };
                (e > s).then(|| SubtitleCue {
                    start: s - start,
                    duration: (e - s).min(u32::MAX as u64) as u32,
                    text: c.text.clone(),
                })
            })
            .collect();
        SubtitleTrack { cues, ..self.clone() }
    }

    /// Every cue moved later by `offset` ticks — placing a clip's cues on the
    /// joined timeline of a splice, where this clip starts at `offset`.
    pub fn shifted(&self, offset: u64) -> SubtitleTrack {
        let cues = self
            .cues
            .iter()
            .map(|c| SubtitleCue { start: c.start + offset, ..c.clone() })
            .collect();
        SubtitleTrack { cues, ..self.clone() }
    }

    /// The same cues expressed on another timescale, rounded to the nearest
    /// tick. A no-op when the timescales already agree.
    pub fn rescaled(&self, timescale: u32) -> SubtitleTrack {
        let timescale = timescale.max(1);
        if timescale == self.timescale {
            return self.clone();
        }
        let from = self.timescale.max(1) as u128;
        let to = timescale as u128;
        let conv = |t: u64| ((t as u128 * to + from / 2) / from) as u64;
        let cues = self
            .cues
            .iter()
            .map(|c| {
                let start = conv(c.start);
                let end = conv(c.end());
                SubtitleCue {
                    start,
                    duration: end.saturating_sub(start).clamp(1, u32::MAX as u64) as u32,
                    text: c.text.clone(),
                }
            })
            .collect();
        SubtitleTrack { cues, timescale, ..self.clone() }
    }

    /// Append `other`'s cues after this track's, converting them onto this
    /// track's timescale first. The caller has already shifted `other` onto
    /// the joined timeline; this only concatenates and re-sorts.
    pub fn append(&mut self, other: &SubtitleTrack) {
        let other = other.rescaled(self.timescale);
        self.cues.extend(other.cues);
        self.cues.sort_by_key(|c| c.start);
    }
}

/// The MKV `CodecID`s that carry text we can convert to `tx3g`.
///
/// Deliberately not exhaustive over Matroska's subtitle codecs: `S_HDMV/PGS`,
/// `S_VOBSUB`, and `S_DVBSUB` are bitmap formats with no `tx3g` equivalent.
fn mkv_text_codec(codec_id: &str) -> Option<&'static str> {
    match codec_id {
        "S_TEXT/UTF8" => Some("subrip"),
        "S_TEXT/ASS" | "S_TEXT/SSA" => Some("ass"),
        "S_TEXT/WEBVTT" => Some("webvtt"),
        _ => None,
    }
}

/// Extract every text subtitle track from a Matroska/WebM file, in track
/// order.
///
/// Bitmap tracks are skipped with a warning; a text track with no usable cues
/// is left out. An empty result means the file has nothing `tx3g` can carry.
pub fn extract_mkv_subtitle_tracks(data: &[u8]) -> Vec<SubtitleTrack> {
    let cursor = Cursor::new(data);
    let Ok(mut mkv) = MatroskaFile::open(cursor) else { return Vec::new() };

    // Matroska block timestamps are in units of `TimestampScale` nanoseconds;
    // the default is 1 ms. Cue timing doesn't need sample accuracy, so a
    // millisecond timescale is both sufficient and exactly what the source
    // gives us.
    let timestamp_scale = mkv.info().timestamp_scale().get();

    // (track number, codec label, language, cues) per text track, in the
    // order the header lists them.
    let mut tracks: Vec<(u64, &'static str, String, Vec<SubtitleCue>)> = Vec::new();
    for t in mkv.tracks().iter().filter(|t| t.track_type() == MkvTrackType::Subtitle) {
        match mkv_text_codec(t.codec_id()) {
            Some(c) => tracks.push((
                t.track_number().get(),
                c,
                t.language().unwrap_or("und").to_string(),
                Vec::new(),
            )),
            None => tracing::warn!(
                codec = t.codec_id(),
                "subtitle track skipped: bitmap subtitles have no tx3g representation"
            ),
        }
    }
    if tracks.is_empty() {
        return Vec::new();
    }

    let mut frame = MkvFrame::default();
    loop {
        match mkv.next_frame(&mut frame) {
            Ok(true) => {
                let Some(slot) = tracks.iter_mut().find(|t| t.0 == frame.track) else {
                    continue;
                };
                let Ok(raw) = std::str::from_utf8(&frame.data) else {
                    tracing::warn!("subtitle cue skipped: not valid UTF-8");
                    continue;
                };
                let text = strip_markup(raw, slot.1);
                if text.is_empty() {
                    continue;
                }
                // Block timestamps and durations are in TimestampScale units;
                // convert both to milliseconds.
                let to_ms = |v: u64| (v as u128 * timestamp_scale as u128 / 1_000_000) as u64;
                let start = to_ms(frame.timestamp);
                // A cue with no duration is a bug in the source; give it a
                // readable two seconds rather than a zero-length flash.
                let duration = frame.duration.map(to_ms).unwrap_or(2_000).max(1) as u32;
                slot.3.push(SubtitleCue { start, duration, text });
            }
            Ok(false) => break,
            Err(_) => break,
        }
    }

    tracks
        .into_iter()
        .filter_map(|(_, codec, language, cues)| finish(codec, cues, REBASE_TIMESCALE, language))
        .collect()
}

/// Shared tail: sort, de-overlap, and reject an empty result.
pub(crate) fn finish(
    codec: &str,
    mut cues: Vec<SubtitleCue>,
    timescale: u32,
    language: String,
) -> Option<SubtitleTrack> {
    if cues.is_empty() {
        return None;
    }
    // Matroska doesn't guarantee subtitle blocks arrive in presentation order.
    cues.sort_by_key(|c| c.start);
    // tx3g samples are laid end to end on one timeline, so two cues can't be
    // on screen at once. Where the source overlaps them, truncate the earlier
    // one — losing an overlap is better than desynchronising everything after
    // it, which is what a negative gap would do to the muxer's sample table.
    for i in 0..cues.len().saturating_sub(1) {
        let end = cues[i].start + cues[i].duration as u64;
        let next = cues[i + 1].start;
        if end > next {
            cues[i].duration = (next.saturating_sub(cues[i].start)).max(1) as u32;
        }
    }
    cues.retain(|c| c.duration > 0);
    if cues.is_empty() {
        return None;
    }
    Some(SubtitleTrack { codec: codec.to_string(), cues, timescale, language })
}

/// Reduce a source cue to the plain UTF-8 text `tx3g` carries.
///
/// `tx3g` styling lives in separate boxes keyed by byte range, not inline, so
/// the markup has to come out of the string either way. Dropping it is lossy
/// and deliberate: the alternative is showing the viewer raw `{\an8}` tags.
pub fn strip_markup(raw: &str, codec: &str) -> String {
    let text = match codec {
        // Matroska carries an ASS/SSA event *without* the `Dialogue:` keyword
        // and without Start/End (the block's own timestamps replace them), so
        // the block is nine comma-separated fields:
        //   ReadOrder, Layer, Style, Name, MarginL, MarginR, MarginV, Effect, Text
        // Text is the last, and may itself contain commas — hence `splitn`.
        // A line that still carries `Dialogue:` came from a raw .ass file and
        // has the two extra time fields, so it splits ten ways instead.
        "ass" => {
            let (fields, body) = match raw.strip_prefix("Dialogue:") {
                Some(rest) => (10, rest.trim_start()),
                None => (9, raw),
            };
            body.splitn(fields, ',').nth(fields - 1).unwrap_or(body)
        }
        _ => raw,
    };
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            // ASS override blocks: `{\an8}`, `{\i1}`, …
            '{' if codec == "ass" => {
                for skipped in chars.by_ref() {
                    if skipped == '}' {
                        break;
                    }
                }
            }
            // ASS hard line break (`\N`) and soft break (`\n`).
            '\\' if codec == "ass" && matches!(chars.peek(), Some('N') | Some('n')) => {
                chars.next();
                out.push('\n');
            }
            // SRT / WebVTT inline tags: `<i>`, `<b>`, `<font …>`, `<c.class>`.
            '<' if codec != "ass" => {
                for skipped in chars.by_ref() {
                    if skipped == '>' {
                        break;
                    }
                }
            }
            _ => out.push(c),
        }
    }
    // WebVTT payload text is HTML-escaped (`&amp;`, `&lt;`, …); `tx3g` and
    // the other sources want the literal characters.
    if codec == "webvtt" {
        out = decode_vtt_entities(&out);
    }
    out.trim().to_string()
}

/// The character references WebVTT allows in cue text (WebVTT §3.4 "cue text
/// span" — `&amp;`, `&lt;`, `&gt;`, `&lrm;`, `&rlm;`, `&nbsp;`). Anything else
/// is left as written.
fn decode_vtt_entities(s: &str) -> String {
    if !s.contains('&') {
        return s.to_string();
    }
    s.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&nbsp;", "\u{a0}")
        .replace("&lrm;", "\u{200e}")
        .replace("&rlm;", "\u{200f}")
        .replace("&amp;", "&")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cue(start: u64, duration: u32, text: &str) -> SubtitleCue {
        SubtitleCue { start, duration, text: text.into() }
    }

    fn track(cues: Vec<SubtitleCue>) -> SubtitleTrack {
        SubtitleTrack { codec: "subrip".into(), cues, timescale: 1_000, language: "eng".into() }
    }

    #[test]
    fn ass_events_lose_their_field_prefix_and_overrides() {
        // The Matroska form: nine fields, no `Dialogue:`, no Start/End.
        let raw = "0,0,Default,,0,0,0,,{\\an8}Hello\\Nthere";
        assert_eq!(strip_markup(raw, "ass"), "Hello\nthere");
        // Commas inside the text survive the field split.
        assert_eq!(strip_markup("0,0,D,,0,0,0,,Yes, really", "ass"), "Yes, really");
        // The raw-.ass form carries `Dialogue:` plus two time fields.
        let dialogue = "Dialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,Hi";
        assert_eq!(strip_markup(dialogue, "ass"), "Hi");
    }

    #[test]
    fn srt_and_vtt_lose_their_inline_tags() {
        assert_eq!(strip_markup("<i>Hello</i> there", "subrip"), "Hello there");
        assert_eq!(strip_markup("<c.yellow>Hi</c>", "webvtt"), "Hi");
        // A bare `<` that never closes shouldn't swallow the rest silently —
        // it does get dropped, but the surrounding text survives.
        assert_eq!(strip_markup("a<b", "subrip"), "a");
    }

    #[test]
    fn vtt_character_references_become_characters() {
        assert_eq!(strip_markup("Tom &amp; Jerry &lt;3", "webvtt"), "Tom & Jerry <3");
        // Only WebVTT is HTML-escaped; SRT text is literal.
        assert_eq!(strip_markup("Tom &amp; Jerry", "subrip"), "Tom &amp; Jerry");
    }

    #[test]
    fn plain_text_is_untouched_apart_from_trimming() {
        assert_eq!(strip_markup("  Hello there \n", "subrip"), "Hello there");
        // ASS braces are not special in SRT.
        assert_eq!(strip_markup("{not a tag}", "subrip"), "{not a tag}");
    }

    #[test]
    fn overlapping_cues_are_truncated_not_reordered() {
        let cues = vec![cue(0, 5_000, "first"), cue(2_000, 1_000, "second")];
        let t = finish("subrip", cues, 1_000, "und".into()).unwrap();
        assert_eq!(t.cues[0].duration, 2_000, "first cue truncated to the second's start");
        assert_eq!(t.cues[1].start, 2_000);
        assert_eq!(t.end_time(), 3_000);
    }

    #[test]
    fn cues_are_sorted_by_start() {
        let cues = vec![cue(4_000, 1_000, "b"), cue(1_000, 1_000, "a")];
        let t = finish("subrip", cues, 1_000, "und".into()).unwrap();
        assert_eq!(t.cues.iter().map(|c| c.text.as_str()).collect::<Vec<_>>(), ["a", "b"]);
    }

    #[test]
    fn an_empty_track_is_no_track() {
        assert!(finish("subrip", Vec::new(), 1_000, "und".into()).is_none());
    }

    #[test]
    fn bitmap_codecs_are_not_text() {
        assert_eq!(mkv_text_codec("S_TEXT/UTF8"), Some("subrip"));
        assert_eq!(mkv_text_codec("S_TEXT/ASS"), Some("ass"));
        assert_eq!(mkv_text_codec("S_TEXT/WEBVTT"), Some("webvtt"));
        assert_eq!(mkv_text_codec("S_HDMV/PGS"), None);
        assert_eq!(mkv_text_codec("S_VOBSUB"), None);
    }

    // ── timeline operations: the trim / splice half ────────────────────────

    #[test]
    fn window_keeps_overlapping_cues_clipped_and_rebased() {
        // Cues at [1,2), [3,5), [6,7) seconds; keep [2.5, 6.5).
        let t = track(vec![cue(1_000, 1_000, "a"), cue(3_000, 2_000, "b"), cue(6_000, 1_000, "c")]);
        let w = t.window(2_500, Some(6_500));
        // "a" ends before the window; "b" is inside and moves left by 2.5 s;
        // "c" straddles the end and is cut there.
        assert_eq!(w.cues, vec![cue(500, 2_000, "b"), cue(3_500, 500, "c")]);
        // A cue straddling the *start* keeps only its tail, from 0.
        let w = t.window(4_000, None);
        assert_eq!(w.cues, vec![cue(0, 1_000, "b"), cue(2_000, 1_000, "c")]);
        // An open window is the identity.
        assert_eq!(t.window(0, None), t);
    }

    #[test]
    fn shifted_moves_every_cue_and_nothing_else() {
        let t = track(vec![cue(0, 1_000, "a"), cue(5_000, 1_000, "b")]);
        let s = t.shifted(8_000);
        assert_eq!(s.cues, vec![cue(8_000, 1_000, "a"), cue(13_000, 1_000, "b")]);
        assert_eq!((s.codec.as_str(), s.timescale, s.language.as_str()), ("subrip", 1_000, "eng"));
    }

    #[test]
    fn rescaled_rounds_to_the_nearest_tick_and_never_zeroes_a_cue() {
        // 600 Hz → 1000 Hz: tick 601 = 1001.67 ms → 1002.
        let t = SubtitleTrack { timescale: 600, ..track(vec![cue(601, 300, "a"), cue(1_200, 1, "b")]) };
        let r = t.rescaled(1_000);
        assert_eq!(r.timescale, 1_000);
        assert_eq!(r.cues[0].start, 1_002);
        assert_eq!(r.cues[0].duration, 500);
        // A 1-tick cue at 600 Hz is 1.67 ms; it stays at least one tick.
        assert!(r.cues[1].duration >= 1);
        // Same timescale is the identity.
        assert_eq!(t.rescaled(600), t);
    }

    #[test]
    fn append_concatenates_on_one_timescale_in_time_order() {
        let mut a = track(vec![cue(0, 1_000, "a")]);
        // A second clip's track on a different timescale, already shifted.
        let b = SubtitleTrack { timescale: 90_000, ..track(vec![cue(180_000, 90_000, "b")]) };
        a.append(&b);
        assert_eq!(a.cues, vec![cue(0, 1_000, "a"), cue(2_000, 1_000, "b")]);
        assert_eq!(a.timescale, 1_000);
    }

    #[test]
    fn ticks_rounds_seconds_on_the_track_timescale() {
        let t = track(Vec::new());
        assert_eq!(t.ticks(2.0), 2_000);
        assert_eq!(t.ticks(0.0005), 1);
        assert_eq!(t.ticks(-3.0), 0);
    }
}
