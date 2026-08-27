//! WebVTT output: the subtitle rendition of an HLS package.
//!
//! An HLS package carries subtitles as a separate rendition of segmented
//! WebVTT (RFC 8216 §3.5), not as a `tx3g` track: one `.vtt` file per video
//! segment, each a complete WebVTT document with an `X-TIMESTAMP-MAP` header
//! tying its cue times to the media timeline, listed by a media playlist
//! whose segment durations are the video's.
//!
//! [`write_webvtt_rendition`] takes a video track's segment grid — the
//! [`CmafTrackManifest`] the segmenter produced — and writes the cues into
//! matching segments: a cue lands in every segment its display time overlaps
//! (§3.5: a cue that spans a boundary is repeated on both sides), with its
//! original absolute times, and a segment with nothing on screen is still
//! written so the rendition covers the whole timeline.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::cmaf::{CmafTrackManifest, SegmentInfo};
use crate::demux::subtitle::SubtitleTrack;

/// What [`write_webvtt_rendition`] produced: the segment list the media
/// playlist is written from. `duration_ticks` is on `timescale`, which is the
/// video grid's timescale, so the playlist's `EXTINF` lines agree with the
/// video's to the tick.
#[derive(Debug, Clone)]
pub struct WebVttManifest {
    pub segments: Vec<SegmentInfo>,
    pub timescale: u32,
}

impl WebVttManifest {
    /// Total duration across all segments, in seconds.
    pub fn duration_seconds(&self) -> f64 {
        let total_ticks: u64 = self.segments.iter().map(|s| s.duration_ticks).sum();
        total_ticks as f64 / self.timescale.max(1) as f64
    }
}

/// The `X-TIMESTAMP-MAP` every segment carries. Cue times are absolute on the
/// asset timeline and the CMAF video's first `tfdt` is 0, so the map is the
/// identity: local 0 is MPEG-2 timestamp 0. A player computes its offset
/// from the video's own first timestamp, so this holds across every segment.
pub const TIMESTAMP_MAP: &str = "X-TIMESTAMP-MAP=MPEGTS:0,LOCAL:00:00:00.000";

/// `HH:MM:SS.mmm` for a time in milliseconds — the long form the WebVTT
/// timestamp grammar allows, chosen over `MM:SS.mmm` so an hour-plus asset
/// needs no special case.
pub fn format_timestamp(ms: u64) -> String {
    let h = ms / 3_600_000;
    let m = (ms / 60_000) % 60;
    let s = (ms / 1_000) % 60;
    let frac = ms % 1_000;
    format!("{h:02}:{m:02}:{s:02}.{frac:03}")
}

/// Cue payload text from the stripped source text. `&`, `<` and `>` become
/// character references (WebVTT reads a bare `<` as the start of a tag), and
/// a blank line inside the text is collapsed, because a blank line ends a
/// cue.
pub fn escape_cue_text(text: &str) -> String {
    let escaped = text.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;");
    escaped
        .lines()
        .map(str::trim_end)
        .filter(|l| !l.trim().is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

/// One segment's WebVTT document: every cue of `track` whose display time
/// overlaps `[seg_start, seg_end)` (in `grid_timescale` ticks), with absolute
/// times. Empty segments are a header alone.
pub fn render_segment(
    track: &SubtitleTrack,
    seg_start: u64,
    seg_end: u64,
    grid_timescale: u32,
) -> String {
    let mut out = String::from("WEBVTT\n");
    out.push_str(TIMESTAMP_MAP);
    out.push_str("\n\n");

    // Overlap in a common unit without rounding: cross-multiply by the two
    // timescales.
    let g = grid_timescale.max(1) as u128;
    let t = track.timescale.max(1) as u128;
    let window_start = seg_start as u128 * t;
    let window_end = seg_end as u128 * t;
    for cue in &track.cues {
        let start = cue.start as u128 * g;
        let end = cue.end() as u128 * g;
        if start >= window_end || end <= window_start {
            continue;
        }
        let text = escape_cue_text(&cue.text);
        if text.is_empty() {
            continue;
        }
        out.push_str(&format_timestamp(ms_of(cue.start, track.timescale)));
        out.push_str(" --> ");
        out.push_str(&format_timestamp(ms_of(cue.end(), track.timescale)));
        out.push('\n');
        out.push_str(&text);
        out.push_str("\n\n");
    }
    out
}

/// Ticks on `timescale` as milliseconds, rounded to nearest.
fn ms_of(ticks: u64, timescale: u32) -> u64 {
    let ts = timescale.max(1) as u128;
    ((ticks as u128 * 1_000 + ts / 2) / ts) as u64
}

/// Write `track` as segmented WebVTT under `dir`, one `seg-NNNNN.vtt` per
/// segment of `grid`, and return the segment list for the media playlist.
///
/// The grid is a video rendition's manifest: the subtitle segments take its
/// boundaries and durations exactly, so `EXT-X-TARGETDURATION` and the
/// per-segment `EXTINF` values are the video's, and a player switching
/// variants never sees the subtitle rendition disagree about where a segment
/// starts.
pub fn write_webvtt_rendition(
    dir: &Path,
    track: &SubtitleTrack,
    grid: &CmafTrackManifest,
) -> Result<WebVttManifest> {
    fs::create_dir_all(dir).with_context(|| format!("creating subtitle dir {}", dir.display()))?;
    let mut segments = Vec::with_capacity(grid.segments.len());
    let mut at: u64 = 0;
    for (i, g) in grid.segments.iter().enumerate() {
        let seg_start = at;
        let seg_end = at + g.duration_ticks;
        at = seg_end;
        let body = render_segment(track, seg_start, seg_end, grid.timescale);
        let path: PathBuf = dir.join(format!("seg-{:05}.vtt", i + 1));
        let mut f = fs::File::create(&path)
            .with_context(|| format!("creating subtitle segment {}", path.display()))?;
        f.write_all(body.as_bytes())
            .with_context(|| format!("writing subtitle segment {}", path.display()))?;
        segments.push(SegmentInfo {
            sequence_number: (i + 1) as u32,
            path,
            byte_size: body.len() as u64,
            duration_ticks: g.duration_ticks,
        });
    }
    Ok(WebVttManifest { segments, timescale: grid.timescale })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::demux::subtitle::SubtitleCue;

    fn cue(start: u64, duration: u32, text: &str) -> SubtitleCue {
        SubtitleCue { start, duration, text: text.into() }
    }

    fn track(cues: Vec<SubtitleCue>) -> SubtitleTrack {
        SubtitleTrack { codec: "subrip".into(), cues, timescale: 1_000, language: "eng".into() }
    }

    /// A small reader for the documents this module writes: the header line,
    /// the timestamp map, and `(start, end, text)` per cue. Used so the tests
    /// check what a player would parse, not what a string contains.
    fn parse(doc: &str) -> (bool, Option<String>, Vec<(String, String, String)>) {
        let mut lines = doc.lines();
        let magic = lines.next() == Some("WEBVTT");
        let mut map = None;
        let mut cues = Vec::new();
        let mut pending: Option<(String, String, Vec<String>)> = None;
        for line in lines {
            if let Some(m) = line.strip_prefix("X-TIMESTAMP-MAP=") {
                map = Some(m.to_string());
                continue;
            }
            if line.is_empty() {
                if let Some((s, e, text)) = pending.take() {
                    cues.push((s, e, text.join("\n")));
                }
                continue;
            }
            if let Some((s, e)) = line.split_once(" --> ") {
                pending = Some((s.to_string(), e.to_string(), Vec::new()));
            } else if let Some(p) = pending.as_mut() {
                p.2.push(line.to_string());
            }
        }
        if let Some((s, e, text)) = pending.take() {
            cues.push((s, e, text.join("\n")));
        }
        (magic, map, cues)
    }

    #[test]
    fn timestamps_are_hh_mm_ss_mmm() {
        assert_eq!(format_timestamp(0), "00:00:00.000");
        assert_eq!(format_timestamp(1_500), "00:00:01.500");
        assert_eq!(format_timestamp(61_001), "00:01:01.001");
        assert_eq!(format_timestamp(3_600_000 + 59_999), "01:00:59.999");
    }

    #[test]
    fn cue_text_is_escaped_and_blank_lines_collapse() {
        assert_eq!(escape_cue_text("Tom & Jerry <3"), "Tom &amp; Jerry &lt;3");
        assert_eq!(escape_cue_text("a\n\nb"), "a\nb", "a blank line would end the cue");
        assert_eq!(escape_cue_text("x --> y"), "x --&gt; y", "an arrow can't be mistaken for a timing line");
    }

    #[test]
    fn a_cue_lands_in_every_segment_it_overlaps_with_absolute_times() {
        // 4 s segments at a 30 kHz grid; a cue from 3.0 s to 5.5 s crosses
        // the first boundary and must appear in segments 1 and 2, unchanged.
        let t = track(vec![cue(500, 1_500, "first"), cue(3_000, 2_500, "crossing"), cue(9_000, 500, "late")]);
        let seg = |i: u64| render_segment(&t, i * 120_000, (i + 1) * 120_000, 30_000);

        let (magic, map, cues) = parse(&seg(0));
        assert!(magic);
        assert_eq!(map.as_deref(), Some("MPEGTS:0,LOCAL:00:00:00.000"));
        assert_eq!(
            cues,
            vec![
                ("00:00:00.500".to_string(), "00:00:02.000".to_string(), "first".to_string()),
                ("00:00:03.000".to_string(), "00:00:05.500".to_string(), "crossing".to_string()),
            ]
        );
        let (_, _, cues) = parse(&seg(1));
        assert_eq!(cues, vec![("00:00:03.000".to_string(), "00:00:05.500".to_string(), "crossing".to_string())]);
        // Segment 3 (8–12 s) holds only the late cue; a cue ending exactly at
        // a boundary does not spill into the next segment.
        let (_, _, cues) = parse(&seg(2));
        assert_eq!(cues.len(), 1);
        assert_eq!(cues[0].2, "late");
        let t2 = track(vec![cue(0, 4_000, "edge")]);
        assert!(parse(&render_segment(&t2, 120_000, 240_000, 30_000)).2.is_empty());
    }

    #[test]
    fn an_empty_segment_is_a_valid_document() {
        let t = track(vec![cue(10_000, 1_000, "far")]);
        let doc = render_segment(&t, 0, 120_000, 30_000);
        let (magic, map, cues) = parse(&doc);
        assert!(magic && map.is_some() && cues.is_empty());
        assert_eq!(doc, "WEBVTT\nX-TIMESTAMP-MAP=MPEGTS:0,LOCAL:00:00:00.000\n\n");
    }

    #[test]
    fn rendition_follows_the_video_grid_exactly() {
        let dir = tempfile::tempdir().unwrap();
        let grid = CmafTrackManifest {
            init_path: dir.path().join("init.mp4"),
            segments: [120_000u64, 120_000, 87_500]
                .iter()
                .enumerate()
                .map(|(i, &d)| SegmentInfo {
                    sequence_number: (i + 1) as u32,
                    path: dir.path().join(format!("seg-{:05}.m4s", i + 1)),
                    byte_size: 1,
                    duration_ticks: d,
                })
                .collect(),
            timescale: 30_000,
        };
        let t = track(vec![cue(3_500, 1_000, "one"), cue(9_000, 500, "two")]);
        let m = write_webvtt_rendition(&dir.path().join("subs"), &t, &grid).unwrap();
        assert_eq!(m.timescale, 30_000);
        let durations: Vec<u64> = m.segments.iter().map(|s| s.duration_ticks).collect();
        assert_eq!(durations, vec![120_000, 120_000, 87_500], "segment durations are the video's");
        assert_eq!(m.segments.len(), grid.segments.len());
        for (i, s) in m.segments.iter().enumerate() {
            assert_eq!(s.sequence_number as usize, i + 1);
            assert!(s.path.exists(), "{} written", s.path.display());
            assert_eq!(s.byte_size, fs::metadata(&s.path).unwrap().len());
        }
        let cues_in = |i: usize| parse(&fs::read_to_string(&m.segments[i].path).unwrap()).2.len();
        assert_eq!((cues_in(0), cues_in(1), cues_in(2)), (1, 1, 1), "cue 'one' straddles 4 s, so it is in segments 1 and 2");
    }
}
