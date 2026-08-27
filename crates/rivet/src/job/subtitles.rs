//! Subtitles in the job engine: the HLS rendition builder, and the re-basing
//! a trim or a splice applies to cue timelines — the subtitle counterpart of
//! [`super::splice::trim_audio`] and [`super::audio::PreparedAudio::extend`].

use std::path::Path;

use anyhow::{Context, Result};

use container::demux::subtitle::{REBASE_TIMESCALE, SubtitleTrack};
use container::hls::{SubtitleVariantSpec, VideoVariantSpec};
use container::language::{bcp47_tag, display_name, same_language};
use container::webvtt::write_webvtt_rendition;

/// Clip every track to the window `[start, end)` seconds and re-base it to
/// zero, the way `trim_audio` keeps the packets inside the window and lets
/// the muxer re-time them from zero. `None`/`None` is the identity.
pub(super) fn trim_subtitles(
    tracks: &[&SubtitleTrack],
    start: Option<f64>,
    end: Option<f64>,
) -> Vec<SubtitleTrack> {
    tracks
        .iter()
        .map(|t| {
            if start.is_none() && end.is_none() {
                (*t).clone()
            } else {
                t.window(t.ticks(start.unwrap_or(0.0)), end.map(|e| t.ticks(e)))
            }
        })
        .collect()
}

/// Place one clip's (already trimmed) tracks on the joined timeline of a
/// splice, `offset_seconds` in, and merge them into `joined` by language:
/// the clip's first `eng` track continues the joined first `eng` track, and
/// a language no earlier clip had starts a new track. Everything lands on
/// [`REBASE_TIMESCALE`] so tracks from sources with different clocks join.
///
/// This is the re-basing the cues need and the audio gets for free: audio
/// packets carry durations and the muxer times them from the running total,
/// but a cue carries an absolute start, so a clip's cues have to be moved by
/// the length of everything before it or they all show up at the start of
/// the output.
pub(super) fn append_clip_subtitles(
    joined: &mut Vec<SubtitleTrack>,
    clip: &[SubtitleTrack],
    offset_seconds: f64,
) {
    let offset_ticks = (offset_seconds.max(0.0) * REBASE_TIMESCALE as f64).round() as u64;
    for (i, t) in clip.iter().enumerate() {
        let ordinal = clip[..i].iter().filter(|o| same_language(&o.language, &t.language)).count();
        let placed = t.rescaled(REBASE_TIMESCALE).shifted(offset_ticks);
        match joined.iter_mut().filter(|j| same_language(&j.language, &t.language)).nth(ordinal) {
            Some(j) => j.append(&placed),
            None => joined.push(placed),
        }
    }
}

/// Write one segmented-WebVTT rendition per track under `<root>/subs/<lang>`
/// on the first video variant's segment grid, and describe them for the
/// master playlist. Two tracks of one language get distinct names and
/// directories (`English`, `English (2)`; `subs/en`, `subs/en-2`), because
/// RFC 8216 §4.3.4.1 wants `NAME` unique within a group and the two
/// playlists cannot share a path. The first rendition is the default.
pub(super) fn build_subtitle_renditions(
    root: &Path,
    tracks: &[SubtitleTrack],
    video: &[VideoVariantSpec],
) -> Result<Vec<SubtitleVariantSpec>> {
    let Some(grid) = video.first().map(|v| &v.manifest) else {
        return Ok(Vec::new());
    };
    let mut specs = Vec::with_capacity(tracks.len());
    for (i, t) in tracks.iter().enumerate() {
        let tag = bcp47_tag(&t.language);
        let dup = tracks[..i].iter().filter(|o| bcp47_tag(&o.language) == tag).count();
        let (name, dir_key) = if dup == 0 {
            (display_name(&t.language), tag.clone())
        } else {
            (format!("{} ({})", display_name(&t.language), dup + 1), format!("{tag}-{}", dup + 1))
        };
        let relative_dir = format!("subs/{dir_key}");
        let manifest = write_webvtt_rendition(&root.join(&relative_dir), t, grid)
            .with_context(|| format!("writing WebVTT rendition {relative_dir}"))?;
        tracing::info!(
            language = %tag,
            name = %name,
            cues = t.cues.len(),
            segments = manifest.segments.len(),
            dir = %relative_dir,
            "HLS subtitle rendition written on the video segment grid"
        );
        specs.push(SubtitleVariantSpec { language: tag, name, relative_dir, default: i == 0, manifest });
    }
    Ok(specs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use container::demux::subtitle::SubtitleCue;

    fn cue(start: u64, duration: u32, text: &str) -> SubtitleCue {
        SubtitleCue { start, duration, text: text.into() }
    }

    fn track(lang: &str, cues: Vec<SubtitleCue>) -> SubtitleTrack {
        SubtitleTrack { codec: "subrip".into(), cues, timescale: 1_000, language: lang.into() }
    }

    #[test]
    fn trim_subtitles_clips_to_the_window_and_rebases_to_zero() {
        let t = track("eng", vec![cue(1_000, 1_000, "a"), cue(3_000, 2_000, "b"), cue(6_000, 1_000, "c")]);
        let out = trim_subtitles(&[&t], Some(2.5), Some(6.5));
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].cues, vec![cue(500, 2_000, "b"), cue(3_500, 500, "c")]);
        // No window: unchanged.
        assert_eq!(trim_subtitles(&[&t], None, None)[0], t);
        // Open end.
        assert_eq!(trim_subtitles(&[&t], Some(5.0), None)[0].cues, vec![cue(1_000, 1_000, "c")]);
    }

    /// The splice re-basing: clip 2's cues must be moved by clip 1's length.
    /// This is the test that fails when the shift is dropped.
    #[test]
    fn spliced_cues_land_at_clip_offset_plus_their_own_time() {
        let clip1 = vec![track("eng", vec![cue(500, 1_000, "one"), cue(3_000, 1_000, "two")])];
        let clip2 = vec![track("eng", vec![cue(250, 1_000, "three")])];
        let mut joined = Vec::new();
        append_clip_subtitles(&mut joined, &clip1, 0.0);
        // Clip 1 is 8 s long on the output timeline.
        append_clip_subtitles(&mut joined, &clip2, 8.0);
        assert_eq!(joined.len(), 1, "same language joins one track");
        assert_eq!(
            joined[0].cues,
            vec![cue(500, 1_000, "one"), cue(3_000, 1_000, "two"), cue(8_250, 1_000, "three")],
            "clip 2's cue is 8 s + 0.25 s in, not 0.25 s"
        );
        assert_eq!(joined[0].timescale, REBASE_TIMESCALE);
    }

    #[test]
    fn languages_join_by_language_and_new_ones_start_new_tracks() {
        let clip1 = vec![track("eng", vec![cue(0, 500, "e1")]), track("deu", vec![cue(0, 500, "d1")])];
        // Clip 2 spells English the BCP-47 way and has no German.
        let clip2 = vec![track("en", vec![cue(100, 500, "e2")]), track("fra", vec![cue(100, 500, "f2")])];
        let mut joined = Vec::new();
        append_clip_subtitles(&mut joined, &clip1, 0.0);
        append_clip_subtitles(&mut joined, &clip2, 4.0);
        let langs: Vec<&str> = joined.iter().map(|t| t.language.as_str()).collect();
        assert_eq!(langs, vec!["eng", "deu", "fra"]);
        assert_eq!(joined[0].cues, vec![cue(0, 500, "e1"), cue(4_100, 500, "e2")]);
        assert_eq!(joined[1].cues, vec![cue(0, 500, "d1")]);
        assert_eq!(joined[2].cues, vec![cue(4_100, 500, "f2")]);
    }

    #[test]
    fn two_tracks_of_one_language_pair_up_by_ordinal() {
        let clip1 = vec![track("eng", vec![cue(0, 500, "a1")]), track("eng", vec![cue(0, 500, "b1")])];
        let clip2 = vec![track("eng", vec![cue(0, 500, "a2")]), track("eng", vec![cue(0, 500, "b2")])];
        let mut joined = Vec::new();
        append_clip_subtitles(&mut joined, &clip1, 0.0);
        append_clip_subtitles(&mut joined, &clip2, 1.0);
        assert_eq!(joined.len(), 2);
        assert_eq!(joined[0].cues.iter().map(|c| c.text.as_str()).collect::<Vec<_>>(), ["a1", "a2"]);
        assert_eq!(joined[1].cues.iter().map(|c| c.text.as_str()).collect::<Vec<_>>(), ["b1", "b2"]);
    }

    #[test]
    fn a_clip_on_another_timescale_is_rescaled_before_joining() {
        let clip1 = vec![track("eng", vec![cue(0, 500, "a")])];
        let mut clip2 = vec![track("eng", vec![cue(90_000, 45_000, "b")])];
        clip2[0].timescale = 90_000;
        let mut joined = Vec::new();
        append_clip_subtitles(&mut joined, &clip1, 0.0);
        append_clip_subtitles(&mut joined, &clip2, 2.0);
        assert_eq!(joined[0].cues, vec![cue(0, 500, "a"), cue(3_000, 500, "b")]);
    }

    #[test]
    fn renditions_get_unique_names_and_dirs_and_one_default() {
        use container::cmaf::{CmafTrackManifest, SegmentInfo};
        let dir = tempfile::tempdir().unwrap();
        let grid = CmafTrackManifest {
            init_path: dir.path().join("video/360p/init.mp4"),
            segments: vec![SegmentInfo {
                sequence_number: 1,
                path: dir.path().join("video/360p/seg-00001.m4s"),
                byte_size: 1,
                duration_ticks: 120_000,
            }],
            timescale: 30_000,
        };
        let video = vec![VideoVariantSpec {
            width: 640,
            height: 360,
            frame_rate: 30.0,
            average_bandwidth_bps: 1,
            bandwidth_bps: 1,
            codec_string: "avc1.64001e".into(),
            supplemental_codecs: None,
            video_range: None,
            relative_dir: "video/360p".into(),
            manifest: grid,
        }];
        let tracks = vec![
            track("eng", vec![cue(0, 500, "a")]),
            track("deu", vec![cue(0, 500, "b")]),
            track("en", vec![cue(0, 500, "c")]),
        ];
        let specs = build_subtitle_renditions(dir.path(), &tracks, &video).unwrap();
        assert_eq!(specs.len(), 3);
        assert_eq!((specs[0].language.as_str(), specs[0].name.as_str(), specs[0].relative_dir.as_str()), ("en", "English", "subs/en"));
        assert_eq!((specs[1].language.as_str(), specs[1].name.as_str(), specs[1].relative_dir.as_str()), ("de", "German", "subs/de"));
        assert_eq!((specs[2].language.as_str(), specs[2].name.as_str(), specs[2].relative_dir.as_str()), ("en", "English (2)", "subs/en-2"));
        assert_eq!(specs.iter().filter(|s| s.default).count(), 1);
        assert!(specs[0].default);
        for s in &specs {
            assert_eq!(s.manifest.segments.len(), 1);
            assert!(s.manifest.segments[0].path.exists());
        }
        // No video grid: nothing to segment on.
        assert!(build_subtitle_renditions(dir.path(), &tracks, &[]).unwrap().is_empty());
    }
}
