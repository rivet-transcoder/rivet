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

    assert!(psnr_8bit(&a, &a).is_infinite(), "identical planes are not lossless");
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
    assert!((psnr_8bit(&a, &b) - expected).abs() < 1e-9, "{} vs {expected}", psnr_8bit(&a, &b));
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

    assert!(different < same, "scrambling did not reduce SSIM: {different} vs {same}");
    assert!(different < 0.5, "SSIM stayed high across a structural change: {different}");
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
