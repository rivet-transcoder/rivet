//! ABR ladder computation — derive a sensible set of [`Rung`]s from a source
//! resolution.
//!
//! Callers who want full control build [`Rung`]s by hand and skip this module
//! entirely. [`standard_ladder`] is the convenience path: it snaps to standard
//! short-side quantizations (2160/1440/1080/720/480/360/240), preserves the
//! source aspect ratio, even-aligns every dimension, caps the top rung, and
//! drops the standard rung just under a source that is barely above it (see
//! [`SOURCE_SNAP_TOLERANCE`]).

use crate::spec::{Quality, Rung};

/// Standard short-side quantizations, descending. The "p" number always refers
/// to the **short** side of the frame regardless of orientation.
const STANDARD_SHORT_SIDES: &[u32] = &[2160, 1440, 1080, 720, 480, 360, 240];

/// Default cap on the short side of any rung. Sources above this have their
/// top rung clamped down (aspect preserved); rungs above it never appear.
pub const DEFAULT_MAX_SHORT_SIDE: u32 = 1080;

/// Smallest allowed dimension on a ladder rung (excluding the source rung).
const MIN_DIMENSION: u32 = 200;

/// How close the source's short side may be to the standard rung beneath it
/// before that rung is dropped as a near-duplicate.
///
/// A 1920×818 source used to emit an 818 rung *and* a 720 rung — 12% apart in
/// height, and the 818 one is the most expensive thing in the ladder. Nobody
/// picks between two rungs that close; one of them is pure storage. Within
/// this tolerance the standard rung is absorbed and the source rung ships
/// alone, at its full resolution.
///
/// # Why the source rung is kept and the standard one dropped
///
/// The other way round — snapping the source *down* onto the standard rung —
/// also removes the duplicate, but it costs 24% of the source's pixels on the
/// rung that sets the ceiling for the whole asset, and that is not recoverable
/// by spending bits. Measured on a 1920×818 source snapped to 1690×720:
///
/// | quality delta | bytes | VMAF |
/// |---|---|---|
/// | 0 | 977,227 | 81.71 |
/// | -8 | 3,539,771 | 84.58 |
/// | -16 | 12,485,098 | 85.54 |
///
/// **12.8× the bytes for +3.8 VMAF**, still short of the 93–96 range anything
/// ships at, because the rung was resolution-limited rather than
/// quality-limited. A 1080p source, which never snapped, reached 99.4.
///
/// Absorbing the standard rung from below keeps the source at full resolution
/// and still ships one rung rather than two, so the tolerance costs nothing: it
/// only ever drops a rung that would have been within 15% of the one above it.
pub const SOURCE_SNAP_TOLERANCE: f64 = 0.15;

/// Build a standard ladder for a source clip, every rung at default quality.
///
/// Pass `max_short_side = None` for the default cap (1080). Lifting the cap
/// unlocks the corresponding higher rungs (1440 → QHD, 2160 → 4K).
pub fn standard_ladder(src_width: u32, src_height: u32, max_short_side: Option<u32>) -> Vec<Rung> {
    dims_for(src_width, src_height, max_short_side)
        .into_iter()
        .map(|(w, h)| Rung::new(w, h))
        .collect()
}

/// Same as [`standard_ladder`] but stamps every rung with `quality`.
pub fn standard_ladder_with_quality(
    src_width: u32,
    src_height: u32,
    max_short_side: Option<u32>,
    quality: Quality,
) -> Vec<Rung> {
    dims_for(src_width, src_height, max_short_side)
        .into_iter()
        .map(|(w, h)| Rung::new(w, h).with_quality(quality.clone()))
        .collect()
}

/// Core algorithm: source-aspect-preserving, even-aligned, capped rung dims.
fn dims_for(src_width: u32, src_height: u32, max_short_side: Option<u32>) -> Vec<(u32, u32)> {
    let cap = max_short_side.unwrap_or(DEFAULT_MAX_SHORT_SIDE);
    if src_width == 0 || src_height == 0 {
        return Vec::new();
    }

    let is_landscape = src_width >= src_height;
    let src_short = src_width.min(src_height);
    let src_long = src_width.max(src_height);
    let src_aspect = src_long as f64 / src_short as f64;

    // Source rung — clamp the short side to `cap`, scale the long side
    // proportionally, even-align both axes.
    let (top_short, top_long) = if src_short > cap {
        let s = cap;
        let l = ((s as f64 * src_aspect).round() as u32) & !1;
        (s & !1, l)
    } else {
        (src_short & !1, src_long & !1)
    };
    let (top_w, top_h) = if is_landscape {
        (top_long, top_short)
    } else {
        (top_short, top_long)
    };

    // The standard rung just under the top, when the top is barely above it —
    // dropped rather than shipped as a near-duplicate. See
    // [`SOURCE_SNAP_TOLERANCE`] for why it is this rung that goes and not the
    // source's.
    let absorbed_short = STANDARD_SHORT_SIDES
        .iter()
        .copied()
        .filter(|&std_short| std_short < top_short && std_short <= cap)
        .max()
        .filter(|&std_short| {
            f64::from(top_short) <= f64::from(std_short) * (1.0 + SOURCE_SNAP_TOLERANCE)
        });

    let mut out: Vec<(u32, u32)> = vec![(top_w, top_h)];

    for &short in STANDARD_SHORT_SIDES {
        if Some(short) == absorbed_short {
            continue;
        }
        if short >= top_short || short > cap {
            continue;
        }
        let long = ((short as f64 * src_aspect).round() as u32) & !1;
        let short_even = short & !1;
        if short_even < MIN_DIMENSION || long < MIN_DIMENSION {
            continue;
        }
        let (w, h) = if is_landscape {
            (long, short_even)
        } else {
            (short_even, long)
        };
        if !out.iter().any(|&(pw, ph)| pw == w && ph == h) {
            out.push((w, h));
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dims(rungs: &[Rung]) -> Vec<(u32, u32)> {
        rungs.iter().map(|r| (r.width, r.height)).collect()
    }

    #[test]
    fn ladder_16_9_1080p_source() {
        let v = standard_ladder(1920, 1080, None);
        assert_eq!(
            dims(&v),
            vec![(1920, 1080), (1280, 720), (852, 480), (640, 360), (426, 240)]
        );
    }

    #[test]
    fn ladder_4k_clamps_to_1080p_by_default() {
        let v = standard_ladder(3840, 2160, None);
        assert_eq!(v.first().map(|r| (r.width, r.height)), Some((1920, 1080)));
    }

    #[test]
    fn ladder_4k_with_2160_cap_keeps_full_quality() {
        let v = standard_ladder(3840, 2160, Some(2160));
        assert_eq!(v.first().map(|r| (r.width, r.height)), Some((3840, 2160)));
        assert_eq!(v.len(), 7);
    }

    #[test]
    fn ladder_portrait_short_side_labels() {
        let v = standard_ladder(1080, 1920, None);
        assert_eq!(dims(&v), vec![(1080, 1920), (720, 1280), (480, 852), (360, 640), (240, 426)]);
        assert_eq!(v[0].label, "1080p");
        assert_eq!(v[1].label, "720p");
    }

    #[test]
    fn ladder_below_floor_keeps_only_source() {
        assert_eq!(dims(&standard_ladder(320, 240, None)), vec![(320, 240)]);
    }

    #[test]
    fn ladder_zero_dims_empty() {
        assert!(standard_ladder(0, 1080, None).is_empty());
    }

    #[test]
    fn every_rung_is_even() {
        for r in standard_ladder(1921, 1081, None) {
            assert_eq!(r.width % 2, 0);
            assert_eq!(r.height % 2, 0);
        }
    }

    #[test]
    fn a_source_just_above_a_standard_rung_absorbs_it() {
        // 1920×818 used to ship an 818 rung *and* a 720 rung, 12% apart. The
        // source keeps its resolution; the near-identical rung underneath goes.
        let heights: Vec<u32> =
            standard_ladder(1920, 818, None).iter().map(|r| r.height).collect();
        assert_eq!(heights.first(), Some(&818), "the source lost resolution: {heights:?}");
        assert!(
            !heights.contains(&720),
            "shipped a near-duplicate 12% below the top: {heights:?}"
        );
        // Only the neighbour goes; the rest of the ladder is untouched.
        assert!(heights.contains(&480), "absorbed more than the neighbour: {heights:?}");
    }

    #[test]
    fn a_source_far_from_a_standard_rung_keeps_its_own() {
        // 1920×960 is 33% above 720 — a real rung of its own.
        let heights: Vec<u32> =
            standard_ladder(1920, 960, None).iter().map(|r| r.height).collect();
        assert_eq!(heights.first(), Some(&960), "a genuine top rung was collapsed: {heights:?}");
        assert!(heights.contains(&720), "the standard rung below it went missing");
    }

    #[test]
    fn a_source_already_on_a_standard_rung_is_unchanged() {
        let heights: Vec<u32> =
            standard_ladder(1280, 720, None).iter().map(|r| r.height).collect();
        assert_eq!(heights.first(), Some(&720));
        assert_eq!(heights.iter().filter(|&&h| h == 720).count(), 1);
    }

    #[test]
    fn portrait_absorbs_on_the_short_side_too() {
        let widths: Vec<u32> =
            standard_ladder(818, 1920, None).iter().map(|r| r.width).collect();
        assert_eq!(widths.first(), Some(&818), "portrait lost resolution: {widths:?}");
        assert!(!widths.contains(&720), "portrait kept the near-duplicate: {widths:?}");
    }

    #[test]
    fn absorption_never_produces_a_duplicate_or_an_empty_ladder() {
        for short in 200..=1200u32 {
            let ladder = standard_ladder(short * 2, short, None);
            assert!(!ladder.is_empty(), "empty ladder at short side {short}");
            let mut seen = dims(&ladder);
            let before = seen.len();
            seen.sort_unstable();
            seen.dedup();
            assert_eq!(before, seen.len(), "duplicate rung at short side {short}");
        }
    }
}
