use super::*;

/// A sweep whose rows are made up, so the selection logic can be tested
/// without hardware.
///
/// Encoding is the expensive, GPU-bound half; choosing between results is
/// ordinary arithmetic and is where the mistakes that silently pick the wrong
/// setting live. Those are worth testing on every machine, not only on one
/// with an encoder.
fn sweep_of(rows: &[(i16, u64, f64)]) -> Sweep {
    Sweep {
        samples: rows
            .iter()
            .map(|(quality_delta, bytes, ssim)| Sample {
                quality_delta: *quality_delta,
                bytes: *bytes,
                psnr: 40.0,
                ssim: *ssim,
            })
            .collect(),
    }
}

#[test]
fn the_cheapest_candidate_clearing_the_floor_wins() {
    // Not the highest quality, and not the smallest — the smallest that is
    // still good enough, which is the only question a ladder actually asks.
    let sweep = sweep_of(&[(-2, 900, 0.985), (0, 700, 0.975), (2, 500, 0.960), (4, 350, 0.930)]);

    let best = sweep.best_at_or_above(0.955).expect("something clears 0.955");
    assert_eq!(best.quality_delta, 2, "picked {:?}", best);
    assert_eq!(best.bytes, 500);
}

#[test]
fn a_floor_nothing_reaches_is_answered_honestly() {
    // "This content needs more bits than anything you offered" is a real
    // answer. Returning the best available would look like success and quietly
    // ship a rung below the floor the caller set.
    let sweep = sweep_of(&[(0, 700, 0.90), (2, 500, 0.88)]);

    assert!(sweep.best_at_or_above(0.99).is_none());
}

#[test]
fn ties_on_quality_are_broken_by_size() {
    // Two settings that look identical should not be a coin flip: the cheaper
    // one is strictly better and picking the other wastes bytes forever.
    let sweep = sweep_of(&[(0, 800, 0.97), (2, 600, 0.97), (4, 900, 0.97)]);

    assert_eq!(sweep.best_at_or_above(0.97).expect("all clear").bytes, 600);
}

#[test]
fn the_knee_is_where_more_bytes_stop_buying_quality() {
    // A curve that climbs steeply and then flattens. The knee is the last
    // candidate before the flat part — past it, bytes are being spent on
    // quality nobody gets.
    let sweep = sweep_of(&[
        (6, 200, 0.900),
        (4, 300, 0.940),
        (2, 450, 0.968),
        (0, 900, 0.972),
        (-2, 1800, 0.974),
    ]);

    let knee = sweep.knee().expect("five points is enough for a knee");
    assert!(
        knee.quality_delta >= 0,
        "the knee landed on the expensive side of the curve: {knee:?}",
    );
}

#[test]
fn too_few_points_have_no_knee() {
    // Two points are a line. Reporting a knee from them would be inventing a
    // shape the data does not have.
    assert!(sweep_of(&[(0, 700, 0.97), (2, 500, 0.96)]).knee().is_none());
}

#[test]
fn value_per_quality_point_is_measured_above_a_floor() {
    // Near SSIM 1.0 the denominator collapses and every ratio becomes huge and
    // meaningless, so the comparison is against a floor of worth-having
    // quality rather than against zero.
    let cheap = Sample { quality_delta: 4, bytes: 400, psnr: 38.0, ssim: 0.96 };
    let dear = Sample { quality_delta: -2, bytes: 1600, psnr: 44.0, ssim: 0.98 };

    assert!(
        cheap.bytes_per_ssim_above(0.90) < dear.bytes_per_ssim_above(0.90),
        "the expensive candidate was rated better value",
    );
}

#[test]
fn an_empty_slice_sweeps_to_nothing() {
    // A caller that hands over no frames gets no rows rather than an error:
    // there is nothing wrong, there is simply nothing to say.
    let config = crate::encode::EncoderConfig::default();
    let sweep = sweep(&config, &[], &[0, 2, 4]).expect("an empty slice is not a failure");

    assert!(sweep.samples.is_empty());
}
