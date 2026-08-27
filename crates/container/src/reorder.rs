//! Composition offsets for samples that arrive in decode order.
//!
//! An encoder hands the muxer its packets in **decode** order, and each packet
//! carries the presentation timestamp of the picture it codes. Without B
//! pictures the two orders coincide; with them the muxer has to say, per
//! sample, how far its presentation time sits from its decode time — the
//! `ctts` table of a plain MP4, the `sample_composition_time_offset` column
//! of a CMAF `trun`.
//!
//! # The design: offsets are a pure function of arrival order and `pts`
//!
//! The muxers do not take a decode timestamp from the encoder. The decode
//! timeline is the muxer's own — a fixed tick per sample, in arrival order —
//! and a sample's presentation time on that timeline is the decode time of
//! the sample that holds its **display rank**: the sample whose `pts` is the
//! `r`-th smallest is presented at the `r`-th decode instant. So the offset
//! for the `i`-th arrival with display rank `r` is `DT(r) − DT(i)`.
//!
//! That makes a wrong timestamp *impossible to write* rather than merely
//! detected: there is no second clock for an encoder to get out of step with,
//! the only input is the timestamp the frame went in with, and the only way
//! to lie is to put a frame's `pts` on a different frame's packet — which
//! nothing at the container level could see either way. Two things *are*
//! checked and refused: a duplicated `pts` (a rank is then undefined), and —
//! where a fragment must stand alone — a first sample that is not its
//! fragment's earliest-presented one.
//!
//! No B pictures ⇒ every rank equals its index ⇒ every offset is zero ⇒ no
//! table is written, and the file is byte-identical to one from a muxer that
//! never had the table.

use anyhow::{Result, bail};

/// Per-sample composition offsets, in ticks, for samples given in decode
/// order with their presentation timestamps and their decode durations.
///
/// `pts[i]` is the timestamp of the `i`-th sample to arrive (any monotone
/// clock — frame numbers do); `durations[i]` is that sample's decode
/// duration in track ticks. The result is `CT(i) − DT(i)` for each sample,
/// where `DT(i)` is the running sum of the durations before `i` and `CT(i)`
/// is the decode time of the sample whose display rank `i` holds.
///
/// Fails on a duplicated `pts`, or on an offset that would not fit the
/// 32-bit field.
pub fn composition_offsets(pts: &[u64], durations: &[u32]) -> Result<Vec<i32>> {
    if pts.len() != durations.len() {
        bail!(
            "composition offsets: {} timestamps but {} durations",
            pts.len(),
            durations.len()
        );
    }
    let mut sorted = pts.to_vec();
    sorted.sort_unstable();
    if let Some(w) = sorted.windows(2).find(|w| w[0] == w[1]) {
        bail!(
            "composition offsets: presentation timestamp {} appears on two samples; \
             a display order is undefined",
            w[0]
        );
    }
    // Decode time of each arrival: prefix sums of the durations.
    let mut dt: Vec<i64> = Vec::with_capacity(pts.len());
    let mut acc: i64 = 0;
    for &d in durations {
        dt.push(acc);
        acc += i64::from(d);
    }
    let mut out = Vec::with_capacity(pts.len());
    for (i, &p) in pts.iter().enumerate() {
        // Present — `sorted` is exactly `pts` reordered.
        let rank = sorted.binary_search(&p).expect("every pts is in its own sorted copy");
        let offset = dt[rank] - dt[i];
        out.push(i32::try_from(offset).map_err(|_| {
            anyhow::anyhow!(
                "composition offsets: sample {i} is presented {offset} ticks from its decode \
                 time, outside the 32-bit field"
            )
        })?);
    }
    Ok(out)
}

/// Whether any sample is presented away from its decode time — i.e. whether
/// a composition-offset table has to be written at all.
pub fn is_reordered(offsets: &[i32]) -> bool {
    offsets.iter().any(|&o| o != 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn in_order_samples_have_no_offset_at_all() {
        let pts = [0u64, 1, 2, 3, 4];
        let offsets = composition_offsets(&pts, &[3000; 5]).unwrap();
        assert_eq!(offsets, vec![0; 5]);
        assert!(!is_reordered(&offsets));
    }

    /// One IDR, then anchors each followed by the two B pictures they
    /// release: display 0 3 1 2 6 4 5. Every anchor is presented two ticks
    /// late, every B one tick early.
    #[test]
    fn two_b_pictures_between_anchors() {
        let pts = [0u64, 3, 1, 2, 6, 4, 5];
        let t = 3000i32;
        let offsets = composition_offsets(&pts, &[t as u32; 7]).unwrap();
        assert_eq!(offsets, vec![0, 2 * t, -t, -t, 2 * t, -t, -t]);
        assert!(is_reordered(&offsets));
        // The presentation times the offsets produce are the input
        // timestamps, each exactly once, on the muxer's own clock.
        let mut ct: Vec<i64> = offsets
            .iter()
            .enumerate()
            .map(|(i, &o)| i as i64 * t as i64 + o as i64)
            .collect();
        ct.sort_unstable();
        assert_eq!(ct, (0..7).map(|k| k * t as i64).collect::<Vec<_>>());
    }

    /// Timestamps need not start at zero or step by one — only their order
    /// matters, because the decode timeline is the muxer's.
    #[test]
    fn only_the_order_of_the_timestamps_matters() {
        let a = composition_offsets(&[10, 13, 11, 12], &[1; 4]).unwrap();
        let b = composition_offsets(&[1000, 4000, 2000, 3000], &[1; 4]).unwrap();
        assert_eq!(a, b);
        assert_eq!(a, vec![0, 2, -1, -1]);
    }

    #[test]
    fn unequal_durations_are_summed_not_multiplied() {
        // Decode times 0, 10, 30; display order 0, 2, 1 → the sample at
        // decode 10 is shown at 30 (+20), the one at 30 is shown at 10 (−20).
        let offsets = composition_offsets(&[0, 2, 1], &[10, 20, 5]).unwrap();
        assert_eq!(offsets, vec![0, 20, -20]);
    }

    #[test]
    fn a_duplicated_timestamp_is_refused() {
        let err = composition_offsets(&[0, 1, 1, 2], &[1; 4]).unwrap_err();
        assert!(err.to_string().contains("appears on two samples"), "{err}");
    }

    #[test]
    fn a_length_mismatch_is_refused() {
        assert!(composition_offsets(&[0, 1], &[1]).is_err());
    }

    #[test]
    fn empty_is_empty() {
        assert!(composition_offsets(&[], &[]).unwrap().is_empty());
    }
}
