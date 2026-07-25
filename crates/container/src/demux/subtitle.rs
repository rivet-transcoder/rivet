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

/// A text subtitle track extracted for passthrough.
#[derive(Debug, Clone)]
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

impl SubtitleTrack {
    /// End time of the last cue, in `timescale` ticks — the track's duration.
    pub fn end_time(&self) -> u64 {
        self.cues.iter().map(|c| c.start + c.duration as u64).max().unwrap_or(0)
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

/// Extract the first text subtitle track from a Matroska/WebM file.
///
/// Returns `None` when there's no subtitle track, when the only ones present
/// are bitmap formats, or when the track carries no cues.
pub fn extract_mkv_subtitles(data: &[u8]) -> Option<SubtitleTrack> {
    let cursor = Cursor::new(data);
    let mut mkv = MatroskaFile::open(cursor).ok()?;

    // Matroska block timestamps are in units of `TimestampScale` nanoseconds;
    // the default is 1 ms. Cue timing doesn't need sample accuracy, so a
    // millisecond timescale is both sufficient and exactly what the source
    // gives us.
    const TIMESCALE: u32 = 1_000;
    let timestamp_scale = mkv.info().timestamp_scale().get();

    let (track_number, codec, language) = {
        let mut chosen = None;
        for t in mkv.tracks().iter().filter(|t| t.track_type() == MkvTrackType::Subtitle) {
            match mkv_text_codec(t.codec_id()) {
                Some(c) => {
                    chosen = Some((
                        t.track_number().get(),
                        c,
                        t.language().unwrap_or("und").to_string(),
                    ));
                    break;
                }
                None => tracing::warn!(
                    codec = t.codec_id(),
                    "subtitle track skipped: bitmap subtitles have no tx3g representation"
                ),
            }
        }
        chosen?
    };

    let mut cues: Vec<SubtitleCue> = Vec::new();
    let mut frame = MkvFrame::default();
    loop {
        match mkv.next_frame(&mut frame) {
            Ok(true) => {
                if frame.track != track_number {
                    continue;
                }
                let Ok(raw) = std::str::from_utf8(&frame.data) else {
                    tracing::warn!("subtitle cue skipped: not valid UTF-8");
                    continue;
                };
                let text = strip_markup(raw, codec);
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
                cues.push(SubtitleCue { start, duration, text });
            }
            Ok(false) => break,
            Err(_) => break,
        }
    }

    finish(codec, cues, TIMESCALE, language)
}

/// Shared tail: sort, de-overlap, and reject an empty result.
fn finish(
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
    out.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn plain_text_is_untouched_apart_from_trimming() {
        assert_eq!(strip_markup("  Hello there \n", "subrip"), "Hello there");
        // ASS braces are not special in SRT.
        assert_eq!(strip_markup("{not a tag}", "subrip"), "{not a tag}");
    }

    #[test]
    fn overlapping_cues_are_truncated_not_reordered() {
        let cues = vec![
            SubtitleCue { start: 0, duration: 5_000, text: "first".into() },
            SubtitleCue { start: 2_000, duration: 1_000, text: "second".into() },
        ];
        let t = finish("subrip", cues, 1_000, "und".into()).unwrap();
        assert_eq!(t.cues[0].duration, 2_000, "first cue truncated to the second's start");
        assert_eq!(t.cues[1].start, 2_000);
        assert_eq!(t.end_time(), 3_000);
    }

    #[test]
    fn cues_are_sorted_by_start() {
        let cues = vec![
            SubtitleCue { start: 4_000, duration: 1_000, text: "b".into() },
            SubtitleCue { start: 1_000, duration: 1_000, text: "a".into() },
        ];
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
}
