//! Frame-rate estimation from a PTS window.

/// Estimate source frame rate from a window of video-PID PTSes.
///
/// Uses the **median** of sorted inter-PTS deltas rather than span /
/// count: the span method is off-by-one-period sensitive to the
/// boundary conditions of the scan (an extra stray PTS on the video
/// PID, a stuffing PES, a mid-stream split) and consistently produced
/// 23.625 instead of 24.000 on the BBB test sample. Median handles
/// outliers uniformly — one spurious 2× delta leaves a run of
/// correct-period deltas around it, and sorting picks the correct
/// one as the middle.
///
/// PTS is 90 kHz; median_delta = ticks-per-frame; fps = 90000 /
/// median_delta. Zero deltas (duplicate PTSes, e.g. if a frame's
/// AU is split across multiple PES packets on the same PID) drop
/// out — they would otherwise force fps → ∞.
///
/// Returns `None` when fewer than two PTSes are present, all deltas
/// are zero, or the estimate lands outside `[1.0, 240.0]` (protects
/// against 33-bit wraparound or a fixed-value PTS injection).
pub(super) fn estimate_frame_rate_from_ptses(ptses: &[u64]) -> Option<f64> {
    if ptses.len() < 2 {
        return None;
    }
    let mut sorted: Vec<u64> = ptses.to_vec();
    sorted.sort_unstable();
    let mut deltas: Vec<u64> = sorted.windows(2).map(|w| w[1] - w[0]).collect();
    deltas.retain(|&d| d > 0);
    if deltas.is_empty() {
        return None;
    }
    deltas.sort_unstable();
    let median = deltas[deltas.len() / 2];
    if median == 0 {
        return None;
    }
    let fps = 90000.0 / median as f64;
    if !fps.is_finite() || !(1.0..=240.0).contains(&fps) {
        return None;
    }
    Some(fps)
}
