//! Programmatic CMAF segment-alignment validator.
//!
//! Run AFTER all video renditions + the audio track have finalized
//! and BEFORE the master playlist is written. Catches the silent
//! ABR-failure mode where one rendition's segments are off by a
//! fraction of a second relative to the others — the master playlist
//! looks fine, hls.js plays the lowest rendition, but every quality
//! switch glitches because the segments don't share timestamps at
//! the boundary.
//!
//! Two checks:
//!   1. every video rendition has the same segment count
//!   2. cumulative duration per segment agrees across renditions
//!      within 1 ms (the drift tolerance is configurable)
//!
//! Audio is allowed to have a different segment count and timestamps
//! — its samples don't need to align frame-by-frame with video, only
//! to share the asset's overall duration.

use anyhow::{Result, bail};

use container::cmaf::CmafTrackManifest;

/// Maximum cumulative-duration drift allowed between any two video
/// renditions at any segment boundary, expressed in seconds.
pub const DEFAULT_DRIFT_TOLERANCE_SECONDS: f64 = 0.001; // 1 ms

/// Validate that every video rendition's segment count and cumulative
/// boundaries match. Returns Ok(()) on success, an Error describing
/// the divergence on failure.
pub fn verify_video_segment_alignment(
    variants: &[&CmafTrackManifest],
    drift_tolerance_seconds: f64,
) -> Result<()> {
    if variants.is_empty() {
        bail!("no video renditions to validate — at least one required");
    }
    let baseline = variants[0];
    let baseline_count = baseline.segments.len();

    for (i, v) in variants.iter().enumerate().skip(1) {
        if v.segments.len() != baseline_count {
            bail!(
                "video rendition {} has {} segments, baseline (rendition 0) has {} — \
                 segment counts must agree across the ladder for ABR switching to work",
                i,
                v.segments.len(),
                baseline_count
            );
        }
    }

    // Cumulative-duration check at each boundary index.
    for boundary_idx in 0..baseline_count {
        let baseline_cum = cumulative_seconds(baseline, boundary_idx);
        for (i, v) in variants.iter().enumerate().skip(1) {
            let v_cum = cumulative_seconds(v, boundary_idx);
            let drift = (v_cum - baseline_cum).abs();
            if drift > drift_tolerance_seconds {
                bail!(
                    "video rendition {} drifts {:.6}s from baseline at segment boundary {} \
                     ({:.6}s vs baseline {:.6}s); tolerance is {:.6}s. ABR switching \
                     between these renditions will glitch at the boundary.",
                    i,
                    drift,
                    boundary_idx + 1,
                    v_cum,
                    baseline_cum,
                    drift_tolerance_seconds
                );
            }
        }
    }

    Ok(())
}

/// Validate that audio + video durations are sane relative to each
/// other. Differs from `verify_video_segment_alignment` in two
/// material ways:
///
///   - **Asymmetry is normal.** UGC routinely lands with audio
///     extending past video (recordings where the camera ends but the
///     mic keeps going) or video past audio (silent intros, files
///     where the audio decoder bailed near the end of the track). The
///     player handles this gracefully: the backend's
///     `CmafPlaylistController` caps the audio playlist to the
///     video's segment count when building `audio.m3u8` (commit
///     `8773fc50`), so audio truncates at video EOF; if video
///     overruns audio, the tail just plays silent. Neither is a
///     failure mode worth bailing the encode for.
///
///   - **Pathological drift IS still a failure**, since drifts of
///     minutes or more usually indicate a probe/decode bug that
///     produced a fundamentally wrong output (the actual recent
///     example: probe reported 30 fps for a 24-fps VFR source, so the
///     encoder produced 51 s less video than the source's audio
///     track). The hard-bail tolerance lives at `pathological_seconds`
///     and is loose by design — small drifts are warning-only.
///
/// Returns Ok(()) when drift is within `pathological_seconds`,
/// emitting a `tracing::warn!` for any drift exceeding the soft
/// `tolerance_seconds` threshold so operators can spot a probe
/// regression in their logs without forcing every re-upload.
///
/// Bails only when drift exceeds `pathological_seconds` — by then the
/// drift is large enough that no player-side handling will produce
/// reasonable output.
pub fn verify_audio_video_duration_match(
    video_variants: &[&CmafTrackManifest],
    audio: &CmafTrackManifest,
    tolerance_seconds: f64,
) -> Result<()> {
    verify_audio_video_duration_match_with_pathological_ceiling(
        video_variants,
        audio,
        tolerance_seconds,
        DEFAULT_PATHOLOGICAL_DRIFT_SECONDS,
    )
}

/// Drift above this many seconds is treated as fatal — no realistic
/// UGC workflow produces audio/video drift over two minutes; if we
/// see it, something deeper than "tail trimming" is wrong (probe
/// reported wrong FPS, decoder bailed catastrophically, or the
/// container is split-track where one stream is from a totally
/// different recording). Surfacing as a typed worker failure is the
/// right call so the user sees a clean "couldn't process this video"
/// rather than a published file with a 5-minute silent tail.
pub const DEFAULT_PATHOLOGICAL_DRIFT_SECONDS: f64 = 120.0;

/// Inner form with explicit pathological ceiling — kept as a separate
/// function so test fixtures with a 4 s overshoot can still exercise
/// the bail branch deterministically without needing to construct a
/// minutes-long synthetic manifest.
pub fn verify_audio_video_duration_match_with_pathological_ceiling(
    video_variants: &[&CmafTrackManifest],
    audio: &CmafTrackManifest,
    tolerance_seconds: f64,
    pathological_seconds: f64,
) -> Result<()> {
    if video_variants.is_empty() {
        return Ok(());
    }
    let video_total = video_variants[0].duration_seconds();
    let audio_total = audio.duration_seconds();
    let drift = (video_total - audio_total).abs();
    if drift <= tolerance_seconds {
        return Ok(());
    }
    if drift > pathological_seconds {
        bail!(
            "audio total {:.6}s differs from video total {:.6}s by {:.6}s; \
             pathological-drift ceiling is {:.6}s. Likely cause: probe \
             reported the wrong fps for a VFR source, or the audio decoder \
             bailed mid-track. No player-side handling produces reasonable \
             output for drifts this large.",
            audio_total,
            video_total,
            drift,
            pathological_seconds
        );
    }
    if audio_total > video_total {
        tracing::warn!(
            video_seconds = video_total,
            audio_seconds = audio_total,
            drift_seconds = drift,
            tolerance_seconds = tolerance_seconds,
            "audio extends {:.3}s past video end — a playlist writer that caps \
             the audio segment count to the video's makes the player truncate \
             audio at video EOF. Continuing.",
            drift,
        );
    } else {
        tracing::warn!(
            video_seconds = video_total,
            audio_seconds = audio_total,
            drift_seconds = drift,
            tolerance_seconds = tolerance_seconds,
            "video extends {:.3}s past audio end — player plays silent \
             video for the tail. Continuing.",
            drift,
        );
    }
    Ok(())
}

/// Cumulative duration in seconds through and including segment index `n`.
fn cumulative_seconds(manifest: &CmafTrackManifest, through_idx: usize) -> f64 {
    let total_ticks: u64 = manifest.segments[..=through_idx]
        .iter()
        .map(|s| s.duration_ticks)
        .sum();
    total_ticks as f64 / manifest.timescale as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use container::cmaf::SegmentInfo;
    use std::path::PathBuf;

    fn synth(timescale: u32, durations_ticks: &[u64]) -> CmafTrackManifest {
        let segments: Vec<SegmentInfo> = durations_ticks
            .iter()
            .enumerate()
            .map(|(i, &d)| SegmentInfo {
                sequence_number: (i + 1) as u32,
                path: PathBuf::from(format!("seg-{:05}.m4s", i + 1)),
                byte_size: 1024,
                duration_ticks: d,
            })
            .collect();
        CmafTrackManifest {
            init_path: PathBuf::from("init.mp4"),
            segments,
            timescale,
        }
    }

    #[test]
    fn aligned_renditions_pass() {
        // Three renditions, identical segment durations on different
        // timescales.
        let v1080 = synth(30000, &[120_000, 120_000, 120_000]); // 4s × 3
        let v720 = synth(30000, &[120_000, 120_000, 120_000]);
        let v480 = synth(30000, &[120_000, 120_000, 120_000]);
        let result = verify_video_segment_alignment(
            &[&v1080, &v720, &v480],
            DEFAULT_DRIFT_TOLERANCE_SECONDS,
        );
        assert!(result.is_ok(), "{:?}", result);
    }

    #[test]
    fn mismatched_segment_counts_fail() {
        let v1 = synth(30000, &[120_000, 120_000, 120_000]); // 3 segments
        let v2 = synth(30000, &[120_000, 120_000]); // 2 segments
        let err = verify_video_segment_alignment(&[&v1, &v2], DEFAULT_DRIFT_TOLERANCE_SECONDS)
            .expect_err("must fail with mismatched segment counts");
        assert!(err.to_string().contains("segments"));
        assert!(err.to_string().contains("counts must agree"));
    }

    #[test]
    fn tiny_drift_within_tolerance_passes() {
        // 1 frame at 30000 timescale = 1/30 s ≈ 33ms. Setting tolerance
        // to 50ms should pass; setting it to 1ms (default) should fail.
        let v1 = synth(30000, &[120_000, 120_000, 120_000]); // 4s × 3 = 12s
        let v2 = synth(30000, &[120_000, 119_000, 121_000]); // total still 360k ticks but
        // mid-segment cumulative drifts
        let err = verify_video_segment_alignment(&[&v1, &v2], DEFAULT_DRIFT_TOLERANCE_SECONDS)
            .expect_err("must fail with sub-frame drift at boundaries");
        assert!(err.to_string().contains("drifts"), "got: {err}");

        // With a generous 50ms tolerance the same renditions pass.
        let result = verify_video_segment_alignment(&[&v1, &v2], 0.050);
        assert!(result.is_ok(), "{:?}", result);
    }

    #[test]
    fn large_drift_at_first_boundary_fails_immediately() {
        // First boundary drift = 1 frame difference (33ms at 30fps),
        // well outside 1ms tolerance.
        let v1 = synth(30000, &[120_000, 120_000]);
        let v2 = synth(30000, &[121_000, 119_000]); // 33ms ahead at boundary 0
        let err = verify_video_segment_alignment(&[&v1, &v2], DEFAULT_DRIFT_TOLERANCE_SECONDS)
            .expect_err("first-boundary drift must fail");
        assert!(err.to_string().contains("boundary 1"), "got: {err}");
    }

    #[test]
    fn audio_video_duration_within_tolerance_passes() {
        let video = synth(30000, &[120_000, 120_000]); // 8s total
        let audio = synth(48000, &[192_000, 192_000]); // also 8s (192_000/48000=4)
        let result = verify_audio_video_duration_match(&[&video], &audio, 0.001);
        assert!(result.is_ok(), "{:?}", result);
    }

    #[test]
    fn audio_video_duration_overshoot_within_pathological_ceiling_warns_and_passes() {
        // 4 s overshoot — outside the lipsync-perceptibility tolerance,
        // well inside the pathological-drift ceiling (120 s by default).
        // Backend caps audio.m3u8 to video segment count downstream;
        // worker passes through with a warn log.
        let video = synth(30000, &[120_000, 120_000]); // 8 s
        let audio = synth(48000, &[192_000, 192_000, 192_000]); // 12 s — 4 s over
        let result = verify_audio_video_duration_match(&[&video], &audio, 0.050);
        assert!(result.is_ok(), "{:?}", result);
    }

    #[test]
    fn audio_video_duration_pathological_overshoot_still_fails() {
        // 130 s overshoot — past the default pathological ceiling.
        // Probe-fps regressions on VFR sources land here; we want the
        // worker to surface a typed failure rather than publish a file
        // with two minutes of orphan audio segments.
        let video = synth(30000, &[300_000]); // 10 s
        let mut durations = Vec::with_capacity(35);
        durations.resize(35, 192_000); // 35 × 4 s = 140 s
        let audio = synth(48000, &durations);
        let err = verify_audio_video_duration_match(&[&video], &audio, 0.050)
            .expect_err("130 s overshoot must trip the pathological ceiling");
        assert!(err.to_string().contains("pathological"), "got: {err}");
        assert!(err.to_string().contains("differs from video"));
    }

    #[test]
    fn audio_video_duration_explicit_ceiling_lets_tight_tests_keep_bailing() {
        // The `_with_pathological_ceiling` form lets tests pin a tight
        // ceiling so a 4 s overshoot still fails deterministically without
        // having to construct a multi-minute synthetic manifest.
        let video = synth(30000, &[120_000, 120_000]); // 8 s
        let audio = synth(48000, &[192_000, 192_000, 192_000]); // 12 s
        let err = verify_audio_video_duration_match_with_pathological_ceiling(
            &[&video],
            &audio,
            0.050,
            1.000,
        )
        .expect_err("4 s overshoot exceeds the pinned 1 s ceiling");
        assert!(err.to_string().contains("pathological"), "got: {err}");
    }

    #[test]
    fn empty_variants_list_fails() {
        let empty: Vec<&CmafTrackManifest> = Vec::new();
        let err = verify_video_segment_alignment(&empty, DEFAULT_DRIFT_TOLERANCE_SECONDS)
            .expect_err("empty variant list must fail");
        assert!(err.to_string().contains("at least one"));
    }

    #[test]
    fn single_variant_trivially_passes() {
        // One variant has no peer to disagree with — trivially aligned
        // (boundary check loops over .skip(1) which is empty).
        let v = synth(30000, &[120_000, 120_000, 87_500]);
        assert!(verify_video_segment_alignment(&[&v], DEFAULT_DRIFT_TOLERANCE_SECONDS).is_ok());
    }
}
