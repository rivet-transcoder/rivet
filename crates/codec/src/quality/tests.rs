use super::*;

/// A flat plane of `value`, `w * h` bytes.
fn flat(value: u8, w: usize, h: usize) -> Vec<u8> {
    vec![value; w * h]
}

#[test]
fn identical_planes_score_perfectly() {
    // The anchor for everything else. If this drifts, no other number in this
    // module means anything.
    let a = flat(128, 32, 32);

    assert!(
        psnr_8bit(&a, &a).is_infinite(),
        "identical planes are not lossless"
    );
    assert!((ssim_8bit(&a, &a, 32, 32) - 1.0).abs() < 1e-9);
}

#[test]
fn psnr_matches_the_closed_form_for_a_constant_offset() {
    // A constant offset of `d` gives MSE = d², so PSNR = 10·log10(255²/d²).
    // Checking against the algebra rather than against a recorded number: a
    // recorded number only says the code still does what it did.
    let a = flat(100, 16, 16);
    let b = flat(110, 16, 16);

    let expected = 10.0 * (255.0f64 * 255.0 / 100.0).log10();
    assert!(
        (psnr_8bit(&a, &b) - expected).abs() < 1e-9,
        "{} vs {expected}",
        psnr_8bit(&a, &b)
    );
}

#[test]
fn ssim_falls_when_structure_is_destroyed() {
    // Two planes with the same mean and wildly different structure. SSIM is
    // supposed to notice; that is the whole reason it exists next to PSNR.
    let w = 32;
    let h = 32;
    let smooth: Vec<u8> = (0..w * h).map(|i| ((i % w) * 8) as u8).collect();
    let scrambled: Vec<u8> = (0..w * h)
        .map(|i| if (i / w) % 2 == 0 { 0 } else { 255 })
        .collect();

    let same = ssim_8bit(&smooth, &smooth, w, h);
    let different = ssim_8bit(&smooth, &scrambled, w, h);

    assert!(
        different < same,
        "scrambling did not reduce SSIM: {different} vs {same}"
    );
    assert!(
        different < 0.5,
        "SSIM stayed high across a structural change: {different}"
    );
}

#[test]
fn a_small_plane_falls_back_rather_than_panicking() {
    // Smaller than the 11x11 window. The fallback is degenerate and that is
    // fine; indexing past the end would not be.
    let a = flat(10, 4, 4);
    let b = flat(20, 4, 4);

    let s = ssim_8bit(&a, &b, 4, 4);
    assert!(s.is_finite(), "small-plane SSIM is not a number: {s}");
    assert!((-1.0..=1.0).contains(&s), "SSIM outside its range: {s}");
}

#[test]
fn noise_scores_worse_than_a_small_offset() {
    // Ordering is the property that matters for choosing between encode
    // settings: a worse reconstruction must score lower, whatever the
    // absolute numbers are.
    let w = 32;
    let h = 32;
    let reference: Vec<u8> = (0..w * h).map(|i| (i % 256) as u8).collect();

    let offset: Vec<u8> = reference.iter().map(|v| v.saturating_add(2)).collect();
    let noisy: Vec<u8> = reference
        .iter()
        .enumerate()
        .map(|(i, v)| if i % 3 == 0 { v.saturating_add(60) } else { *v })
        .collect();

    assert!(
        psnr_8bit(&reference, &noisy) < psnr_8bit(&reference, &offset),
        "noise did not score worse than a small offset",
    );
}

/// A frame whose luma plane is built by `f`, 64x64, YUV420p.
fn frame_of(f: impl Fn(usize, usize) -> u8) -> crate::frame::VideoFrame {
    use crate::frame::{ColorSpace, PixelFormat, VideoFrame};

    let (w, h) = (64usize, 64usize);
    let mut data = vec![128u8; w * h * 3 / 2];
    for y in 0..h {
        for x in 0..w {
            data[y * w + x] = f(x, y);
        }
    }
    VideoFrame::new(
        bytes::Bytes::from(data),
        w as u32,
        h as u32,
        PixelFormat::Yuv420p,
        ColorSpace::Bt709,
        0,
    )
}

#[test]
fn tv_range_black_reads_as_blank() {
    // The frame a fade-in opens on. Y=16 is black in TV range, which is what
    // decoded frames are in — testing against 0 would find nothing and the
    // sweep would happily measure the fade.
    let black = frame_of(|_, _| 16);
    let activity = luma_activity(&black);

    assert!((activity.mean - 16.0).abs() < 1e-9, "{activity:?}");
    assert!(
        activity.std_dev < 1e-9,
        "flat plane has variation: {activity:?}"
    );
    assert!(
        activity.looks_blank(),
        "TV-range black was not called blank: {activity:?}"
    );
}

#[test]
fn a_dim_night_scene_is_content() {
    // The case that must not be rejected. Dark, but with real detail in it —
    // exactly the content that most needs its own quality decision, and the
    // reason the brightness threshold is generous and the variation one is not.
    let night = frame_of(|x, y| 18 + (((x * 7 + y * 13) % 40) as u8));
    let activity = luma_activity(&night);

    assert!(
        activity.mean < 45.0,
        "the fixture is not dark: {activity:?}"
    );
    assert!(
        !activity.looks_blank(),
        "a dim scene with detail was called blank: {activity:?}"
    );
}

#[test]
fn a_flat_white_card_is_blank_despite_being_bright() {
    // Brightness is not content. A title card carries nothing for an encoder
    // to spend bits on, and sweeping across one says every setting is free.
    let card = frame_of(|_, _| 235);

    assert!(
        luma_activity(&card).looks_blank(),
        "a flat bright frame was called content"
    );
}

#[test]
fn one_live_frame_does_not_rescue_a_black_window() {
    // The end of a fade-in: mostly black, with the picture just arriving. Not
    // a sample of the content, and the averaged activity says otherwise —
    // the single busy frame's variance spreads across the window and lifts its
    // standard deviation clear of any sane flatness threshold. Counting frames
    // is what sees through that, and this test exists because averaging did
    // not.
    let mut window: Vec<_> = (0..8).map(|_| frame_of(|_, _| 16)).collect();
    window.push(frame_of(|x, y| ((x * 3 + y * 5) % 255) as u8));

    let averaged = window_activity(&window).expect("a non-empty window");
    assert!(
        !averaged.looks_blank(),
        "the fixture no longer demonstrates the averaging trap: {averaged:?}",
    );

    assert!(
        window_looks_blank(&window),
        "an 8-in-9 black window passed as content (blank fraction {})",
        blank_fraction(&window),
    );
}

#[test]
fn a_window_that_is_mostly_picture_is_content() {
    // The other side of the majority rule. A couple of dark frames in an
    // otherwise live window — a cut, a blink, a dark corner of a scene — must
    // not disqualify it, or nothing with contrast in it would ever be sampled.
    let mut window: Vec<_> = (0..8)
        .map(|i| frame_of(move |x, y| (((x * 3 + y * 5) + i * 11) % 255) as u8))
        .collect();
    window.push(frame_of(|_, _| 16));
    window.push(frame_of(|_, _| 16));

    assert!(
        !window_looks_blank(&window),
        "a mostly-live window was rejected (blank fraction {})",
        blank_fraction(&window),
    );
}

#[test]
fn blank_fraction_counts_what_it_says() {
    // The unit a caller sets a threshold in, so it has to mean literally
    // "proportion of frames that are blank" and not some weighted proxy.
    let mut window: Vec<_> = (0..3).map(|_| frame_of(|_, _| 16)).collect();
    window.push(frame_of(|x, y| ((x * 3 + y * 5) % 255) as u8));

    assert!(
        (blank_fraction(&window) - 0.75).abs() < 1e-9,
        "{}",
        blank_fraction(&window)
    );
    assert_eq!(
        blank_fraction(&[]),
        0.0,
        "an empty window has no blank frames"
    );
}

#[test]
fn an_empty_window_has_no_activity() {
    // Nothing to say rather than a made-up zero that would read as "black".
    assert!(window_activity(&[]).is_none());
}

#[test]
fn frames_of_different_sizes_do_not_score() {
    // A caller comparing a 720p decode against a 1080p reference has a bug,
    // and a number here would hide it — the comparison is not meaningful and
    // saying so is more useful than returning something plausible.
    use crate::frame::{ColorSpace, PixelFormat, VideoFrame};

    let make = |w: u32, h: u32| {
        VideoFrame::new(
            bytes::Bytes::from(flat(128, (w * h) as usize * 3 / 2, 1)),
            w,
            h,
            PixelFormat::Yuv420p,
            ColorSpace::Bt709,
            0,
        )
    };

    assert!(score_frame(&make(64, 64), &make(32, 32)).is_none());
}
