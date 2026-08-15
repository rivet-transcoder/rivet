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
                trimmed_bytes: *bytes,
                psnr: 40.0,
                ssim: *ssim,
                packets: 0,
                largest_packet: 0,
                mean_other_packet: 0,
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
    let cheap = Sample { quality_delta: 4, bytes: 400, trimmed_bytes: 400, psnr: 38.0, ssim: 0.96, packets: 0, largest_packet: 0, mean_other_packet: 0 };
    let dear = Sample { quality_delta: -2, bytes: 1600, trimmed_bytes: 1600, psnr: 44.0, ssim: 0.98, packets: 0, largest_packet: 0, mean_other_packet: 0 };

    assert!(
        cheap.bytes_per_ssim_above(0.90) < dear.bytes_per_ssim_above(0.90),
        "the expensive candidate was rated better value",
    );
}

#[test]
fn ssim_in_db_separates_what_raw_ssim_crushes_together() {
    // The two clips this scale exists for, at identical encoder settings.
    // 0.9989 and 0.9804 look like neighbours; they are 12 dB apart, and their
    // delivered VMAF was 99.4 and 81.7. Any threshold set in raw SSIM is
    // describing one of them wrongly.
    let flat = Sample { quality_delta: 0, bytes: 1, trimmed_bytes: 1, psnr: 0.0, ssim: 0.9989, packets: 0, largest_packet: 0, mean_other_packet: 0 };
    let noisy = Sample { quality_delta: 0, bytes: 1, trimmed_bytes: 1, psnr: 0.0, ssim: 0.9804, packets: 0, largest_packet: 0, mean_other_packet: 0 };

    assert!((flat.ssim_db() - 29.6).abs() < 0.5, "{}", flat.ssim_db());
    assert!((noisy.ssim_db() - 17.1).abs() < 0.5, "{}", noisy.ssim_db());
    assert!(flat.ssim_db() - noisy.ssim_db() > 10.0, "the scale still crushes them together");
}

#[test]
fn a_perfect_slice_does_not_return_infinity() {
    // One flat frame can be reconstructed exactly. Left uncapped that is an
    // infinite dB value, and it propagates into every comparison it touches.
    let perfect = Sample { quality_delta: 0, bytes: 1, trimmed_bytes: 1, psnr: 0.0, ssim: 1.0, packets: 0, largest_packet: 0, mean_other_packet: 0 };

    assert!(perfect.ssim_db().is_finite(), "{}", perfect.ssim_db());
    assert!(perfect.ssim_db() <= 60.0);
}

#[test]
fn the_budget_is_spent_against_the_clips_own_base() {
    // The whole point: the same budget applied to two clips with very
    // different bases picks sensibly for both, where one fixed floor could not
    // pick sensibly for either.
    //
    // Easy clip — base is far above transparency, so a 1 dB budget should buy
    // a much cheaper encode.
    let easy = sweep_of(&[
        (0, 1000, 0.9989),   // 29.6 dB — base
        (4, 600, 0.9975),    // 26.0 dB — 3.6 dB down, too far
        (2, 800, 0.9986),    // 28.5 dB — 1.1 dB down, just outside
    ]);
    let picked = easy.best_within_drop(1.0, None).expect("the base itself always qualifies");
    assert_eq!(picked.quality_delta, 0, "gave away more than the budget: {picked:?}");

    // Same budget, wider spacing — now there is something inside it.
    let easy2 = sweep_of(&[
        (0, 1000, 0.9989),   // 29.6 dB
        (2, 700, 0.99875),   // 29.0 dB — 0.6 dB down, inside a 1 dB budget
    ]);
    assert_eq!(easy2.best_within_drop(1.0, None).expect("inside budget").quality_delta, 2);
}

#[test]
fn a_clip_with_no_headroom_keeps_its_base() {
    // The hard case. Every cheaper candidate costs more than the budget, so
    // the honest answer is the base — not the least-bad alternative. This is
    // the direction that ruins videos when it goes wrong.
    let hard = sweep_of(&[
        (0, 1000, 0.9804),   // 17.1 dB
        (6, 500, 0.9703),    // 15.3 dB — 1.8 dB down
        (8, 350, 0.9619),    // 14.2 dB — 2.9 dB down
    ]);

    let picked = hard.best_within_drop(0.5, None).expect("the base qualifies");
    assert_eq!(picked.quality_delta, 0, "shipped a visible quality loss: {picked:?}");
    assert_eq!(picked.bytes, 1000);
}

#[test]
fn the_absolute_floor_still_vetoes_a_bad_base() {
    // A clip whose base is already poor should not spend a budget on top of
    // it. The relative rule alone would happily approve 0.90 → 0.89.
    let poor = sweep_of(&[(0, 1000, 0.90), (4, 400, 0.895)]);

    assert!(poor.best_within_drop(1.0, Some(0.95)).is_none(), "a poor base passed the floor");
    assert!(poor.best_within_drop(1.0, None).is_some(), "the floor is meant to be optional");
}

#[test]
fn a_sweep_without_a_base_row_cannot_judge_a_drop() {
    // Every relative decision is measured from delta 0. Without it there is no
    // reference, and inventing one from the best available would silently
    // change what the budget means.
    let no_base = sweep_of(&[(2, 800, 0.99), (4, 600, 0.98)]);

    assert!(no_base.base().is_none());
    assert!(no_base.best_within_drop(1.0, None).is_none());
}

/// A 1080p-ish sample: one frame's pixels, and how many frames were encoded.
const PX: u64 = 1920 * 1080;
const FR: usize = 60;

#[test]
fn on_flat_content_the_cheapest_candidate_wins() {
    // The case both thresholds got wrong. Quality is effectively identical at
    // every setting — the encoder reproduces this content exactly — so there
    // is nothing to protect and the only sensible answer is the smallest file.
    // An absolute floor kept it because everything cleared; a relative dB
    // budget kept the *base* because dB explodes near SSIM 1.0. The slope sees
    // a flat curve and takes the saving.
    let flat = sweep_of(&[
        (0, 200_000, 0.99999),
        (4, 150_000, 0.99997),
        (8, 90_000, 0.99995),
    ]);

    let picked = flat.rd_optimal(0.15, PX, FR).expect("a non-empty sweep");
    assert_eq!(picked.quality_delta, 8, "left the saving on the table: {picked:?}");
}

#[test]
fn on_hard_content_the_same_lambda_refuses_the_saving() {
    // Same constant, opposite answer. Quality falls steeply here, so the bytes
    // saved stop paying for the distortion added. This is the direction that
    // ruins videos, and it has to be handled by the same number that allowed
    // the flat clip's saving — otherwise there are two knobs and neither is
    // calibrated.
    let hard = sweep_of(&[
        (0, 200_000, 0.9804),
        (6, 100_000, 0.9703),
        (8, 70_000, 0.9619),
    ]);

    let picked = hard.rd_optimal(0.15, PX, FR).expect("a non-empty sweep");
    assert_eq!(picked.quality_delta, 0, "shipped a visible loss: {picked:?}");
}

#[test]
fn lambda_orders_the_tradeoff_monotonically() {
    // A larger lambda values bytes more, so it can never choose a *dearer*
    // encode than a smaller one. If this inverts, the knob is not a knob.
    let sweep = sweep_of(&[
        (0, 200_000, 0.995),
        (4, 120_000, 0.990),
        (8, 60_000, 0.975),
    ]);

    let cheapskate = sweep.rd_optimal(5.0, PX, FR).expect("non-empty").quality_delta;
    let spendthrift = sweep.rd_optimal(0.001, PX, FR).expect("non-empty").quality_delta;

    assert!(
        cheapskate >= spendthrift,
        "a bigger lambda picked a dearer encode: {cheapskate} vs {spendthrift}",
    );
}

#[test]
fn rate_is_normalised_so_lambda_survives_a_different_sample_size() {
    // Without per-pixel normalisation, doubling the sample length halves the
    // effective lambda and silently changes every decision — the kind of bug
    // that looks like the encoder behaving differently on long videos.
    let sweep = sweep_of(&[(0, 200_000, 0.995), (4, 120_000, 0.990), (8, 60_000, 0.975)]);

    let short = sweep.rd_optimal(0.15, PX, 30).expect("non-empty").quality_delta;
    let long = sweep_of(&[(0, 400_000, 0.995), (4, 240_000, 0.990), (8, 120_000, 0.975)])
        .rd_optimal(0.15, PX, 60)
        .expect("non-empty")
        .quality_delta;

    assert_eq!(short, long, "twice the sample at the same bitrate changed the answer");
}

#[test]
fn an_empty_sweep_has_no_optimum() {
    assert!(Sweep::default().rd_optimal(0.15, PX, FR).is_none());
    // Zero pixels would divide by zero and rank every candidate as equal.
    assert!(sweep_of(&[(0, 100, 0.99)]).rd_optimal(0.15, 0, FR).is_none());
}

#[test]
fn scoring_a_smaller_rung_goes_through_the_upscale() {
    // The property that makes a sweep predictive: a rung is compared at the
    // size a viewer watches it, not at its own. This checks the scaling
    // helpers agree on that round trip — down to the rung, back to the
    // reference — because if they do not, `score_frame` sees mismatched
    // dimensions, returns `None`, and every candidate reports "no decodable
    // frames" rather than a bad score.
    use crate::colorspace::scale_frame;
    use crate::frame::{ColorSpace, PixelFormat, VideoFrame};

    let (sw, sh) = (128u32, 64u32);
    let mut data = vec![128u8; (sw * sh) as usize * 3 / 2];
    for y in 0..sh as usize {
        for x in 0..sw as usize {
            data[y * sw as usize + x] = ((x * 3 + y * 7) % 255) as u8;
        }
    }
    let reference =
        VideoFrame::new(bytes::Bytes::from(data), sw, sh, PixelFormat::Yuv420p, ColorSpace::Bt709, 0);

    let rung = scale_frame(&reference, 64, 32).expect("down to the rung");
    assert_eq!((rung.width, rung.height), (64, 32));

    let shown = scale_frame(&rung, sw, sh).expect("back up to the reference");
    assert_eq!((shown.width, shown.height), (sw, sh));

    let score = quality::score_frame(&reference, &shown).expect("dimensions now agree");

    // A real loss, since the round trip through half resolution cannot be
    // free. The assertion is that it is *visible to the metric* at all — the
    // native-resolution comparison this replaced would have reported a
    // near-perfect score for the same rung.
    assert!(score.ssim < 0.999, "the upscale round trip was free: {score:?}");
    assert!(score.ssim > 0.0, "the comparison collapsed: {score:?}");
}

#[test]
fn an_empty_slice_sweeps_to_nothing() {
    // A caller that hands over no frames gets no rows rather than an error:
    // there is nothing wrong, there is simply nothing to say.
    let config = crate::encode::EncoderConfig::default();
    let sweep = sweep(&config, &[], &[0, 2, 4]).expect("an empty slice is not a failure");

    assert!(sweep.samples.is_empty());
}
