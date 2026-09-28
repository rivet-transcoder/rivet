use super::*;

fn out(p: &Placement) -> (u32, u32) {
    p.canvas
}

fn contain(src: SourceShape, bx: (u32, u32)) -> Placement {
    place(src, bx, Fit::Contain, Orientation::Auto, false)
}

fn sq(w: u32, h: u32) -> SourceShape {
    SourceShape::square(w, h)
}

/// Width over height of what `p` puts on screen, from the crop it takes and
/// the size it scales that to — equal to the source's shape when nothing is
/// distorted.
fn shown_aspect_error(src: SourceShape, p: &Placement) -> f64 {
    let (_, _, cw, ch) = p.crop;
    let sar = f64::from(src.sample_aspect.0) / f64::from(src.sample_aspect.1);
    let crop_shape = f64::from(cw) * sar / f64::from(ch);
    let shown = f64::from(p.scaled.0) / f64::from(p.scaled.1);
    (shown / crop_shape - 1.0).abs()
}

#[test]
fn a_4_3_source_keeps_its_shape_and_its_size_in_a_720p_box() {
    // The reported defect: 640x480 through web-av1-720p came out 1280x720,
    // stretched sideways and upscaled.
    let p = contain(sq(640, 480), (1280, 720));
    assert_eq!(out(&p), (640, 480));
    assert!(!p.crops() && !p.pads());
    // Allowed to enlarge, it fills the box's height and keeps 4:3.
    let up = place(sq(640, 480), (1280, 720), Fit::Contain, Orientation::Auto, true);
    assert_eq!(out(&up), (960, 720));
}

#[test]
fn a_portrait_source_turns_a_landscape_box() {
    // The second defect: 720x1280 through a 1280x720 rung was squashed into
    // landscape. The box turns; the source already fits it.
    assert_eq!(out(&contain(sq(720, 1280), (1280, 720))), (720, 1280));
    // Through 1920x1080: a 1080x1920 box, no upscale — the source's size.
    assert_eq!(out(&contain(sq(720, 1280), (1920, 1080))), (720, 1280));
    let up = place(sq(720, 1280), (1920, 1080), Fit::Contain, Orientation::Auto, true);
    assert_eq!(out(&up), (1080, 1920));
    // A 1080p portrait source through a 720p rung: 720x1280, not 406x720.
    assert_eq!(out(&contain(sq(1080, 1920), (1280, 720))), (720, 1280));
}

#[test]
fn a_fixed_box_is_used_as_written() {
    let p = place(sq(1080, 1920), (1280, 720), Fit::Contain, Orientation::Fixed, false);
    let (w, h) = out(&p);
    assert_eq!(h, 720);
    assert!((404..=406).contains(&w), "{w}x{h}");
}

#[test]
fn wide_and_square_sources_fit_by_their_limiting_side() {
    // 21:9 in a 16:9 box: width-limited, letterbox-free.
    assert_eq!(out(&contain(sq(2560, 1080), (1920, 1080))), (1920, 810));
    assert_eq!(out(&contain(sq(3840, 1608), (1920, 1080))), (1920, 804));
    // 1:1: height-limited, and a square box stays square.
    assert_eq!(out(&contain(sq(1080, 1080), (1920, 1080))), (1080, 1080));
    assert_eq!(out(&contain(sq(2000, 2000), (1280, 720))), (720, 720));
    assert_eq!(out(&contain(sq(1920, 1080), (720, 720))), (720, 406));
}

#[test]
fn an_anamorphic_source_is_sized_by_its_display_shape() {
    // PAL 16:9: 720x576 stored, 64:45 samples, shown 1024x576.
    let pal_wide = SourceShape { width: 720, height: 576, sample_aspect: (64, 45) };
    assert_eq!(out(&contain(pal_wide, (1920, 1080))), (1024, 576));
    assert_eq!(out(&contain(pal_wide, (854, 480))), (854, 480));
    // PAL 4:3: 16:15 samples, shown 768x576.
    let pal_43 = SourceShape { width: 720, height: 576, sample_aspect: (16, 15) };
    assert_eq!(out(&contain(pal_43, (1920, 1080))), (768, 576));
    // NTSC 4:3: 10:11 — tall samples, shown 720x528.
    let ntsc = SourceShape { width: 720, height: 480, sample_aspect: (10, 11) };
    assert_eq!(out(&contain(ntsc, (1920, 1080))), (720, 528));
    // HDV: 1440x1080 at 4:3 is 1920x1080.
    let hdv = SourceShape { width: 1440, height: 1080, sample_aspect: (4, 3) };
    assert_eq!(out(&contain(hdv, (1920, 1080))), (1920, 1080));
    for (src, bx) in [(pal_wide, (1920, 1080)), (pal_43, (1280, 720)), (ntsc, (640, 360)), (hdv, (1280, 720))] {
        for fit in [Fit::Contain, Fit::Cover, Fit::Pad] {
            let p = place(src, bx, fit, Orientation::Auto, false);
            assert!(shown_aspect_error(src, &p) < 0.01, "{src:?} {bx:?} {fit}: {p:?}");
        }
    }
}

#[test]
fn an_odd_source_is_even_aligned_down() {
    assert_eq!(out(&contain(sq(853, 480), (1280, 720))), (852, 480));
    assert_eq!(out(&contain(sq(853, 480), (640, 360))), (640, 360));
    assert_eq!(out(&contain(sq(1921, 1081), (1920, 1080))), (1920, 1080));
    let p = contain(sq(853, 480), (1280, 720));
    assert_eq!(p.crop, (0, 0, 853, 480), "the whole picture is taken, odd column included");
}

#[test]
fn cover_fills_the_box_and_crops_the_centre() {
    // A landscape 1080p source into a fixed 9:16 box, allowed to enlarge:
    // exactly the box, from a centred 608x1080 window of the source.
    let p = place(sq(1920, 1080), (1080, 1920), Fit::Cover, Orientation::Fixed, true);
    assert_eq!(out(&p), (1080, 1920));
    let (x, y, w, h) = p.crop;
    assert_eq!((w, h), (608, 1080));
    assert_eq!((x, y), (656, 0), "centred");
    // Without upscale: the same window at its own size.
    let p = place(sq(1920, 1080), (1080, 1920), Fit::Cover, Orientation::Fixed, false);
    assert_eq!(out(&p), (608, 1080));
    // A 4:3 source into 16:9: the top and bottom are cut.
    let p = place(sq(1440, 1080), (1920, 1080), Fit::Cover, Orientation::Auto, false);
    assert_eq!(out(&p), (1440, 810));
    assert_eq!(p.crop, (0, 134, 1440, 810));
    // Auto orientation on a landscape source turns the 9:16 box: no crop.
    let p = place(sq(1920, 1080), (1080, 1920), Fit::Cover, Orientation::Auto, false);
    assert_eq!(out(&p), (1920, 1080));
    assert!(!p.crops());
}

#[test]
fn pad_is_exactly_the_box_with_the_picture_centred() {
    let p = place(sq(640, 480), (1280, 720), Fit::Pad, Orientation::Auto, false);
    assert_eq!((p.canvas, p.scaled, p.offset), ((1280, 720), (640, 480), (320, 120)));
    let p = place(sq(640, 480), (1280, 720), Fit::Pad, Orientation::Auto, true);
    assert_eq!((p.canvas, p.scaled, p.offset), ((1280, 720), (960, 720), (160, 0)));
    let p = place(sq(2560, 1080), (1920, 1080), Fit::Pad, Orientation::Auto, false);
    assert_eq!((p.canvas, p.scaled, p.offset), ((1920, 1080), (1920, 810), (0, 134)));
}

#[test]
fn stretch_is_the_old_behaviour() {
    let p = place(sq(720, 1280), (1280, 720), Fit::Stretch, Orientation::Auto, false);
    assert_eq!((p.canvas, p.scaled, p.crop), ((1280, 720), (1280, 720), (0, 0, 720, 1280)));
}

#[test]
fn every_fit_gives_even_sizes() {
    for (w, h) in [(853, 480), (641, 479), (1921, 817), (333, 777), (1080, 1081)] {
        for bx in [(1920, 1080), (1280, 720), (640, 360), (720, 720)] {
            for fit in Fit::ALL {
                for upscale in [false, true] {
                    let p = place(sq(w, h), bx, fit, Orientation::Auto, upscale);
                    for (a, b) in [p.canvas, p.scaled, p.offset] {
                        assert!(a % 2 == 0 && b % 2 == 0, "{w}x{h} {bx:?} {fit} {upscale}: {p:?}");
                    }
                    assert!(p.offset.0 + p.scaled.0 <= p.canvas.0 && p.offset.1 + p.scaled.1 <= p.canvas.1);
                    let (x, y, cw, ch) = p.crop;
                    assert!(x + cw <= w && y + ch <= h, "crop outside the source: {p:?}");
                    if fit != Fit::Stretch {
                        assert!(shown_aspect_error(sq(w, h), &p) < 0.02, "{w}x{h} {bx:?} {fit}: {p:?}");
                    }
                    if !upscale && fit != Fit::Stretch {
                        assert!(p.scaled.0 <= w + 1 && p.scaled.1 <= h + 1, "upscaled: {w}x{h} {p:?}");
                    }
                }
            }
        }
    }
}

#[test]
fn rungs_a_small_source_collapses_are_merged_and_reported() {
    let rungs = vec![Rung::new(1920, 1080), Rung::new(1280, 720), Rung::new(640, 360)];
    let (kept, report) = fit_rungs(&rungs, sq(640, 480), Fit::Contain, Orientation::Auto, false);
    let dims: Vec<_> = kept.iter().map(|r| (r.width, r.height, r.label.as_str())).collect();
    assert_eq!(dims, vec![(640, 480, "480p"), (480, 360, "360p")]);
    assert_eq!(report.len(), 3);
    assert_eq!(report[0].duplicate_of, None);
    assert_eq!(report[1].duplicate_of, Some(0), "1280x720 came out the same as 1920x1080");
    assert_eq!((report[1].requested, report[1].output, report[1].label.as_str()), ((1280, 720), (640, 480), "480p"));
    assert_eq!((report[2].output, report[2].duplicate_of), ((480, 360), None));
    // With upscale there is nothing to merge.
    let (kept, _) = fit_rungs(&rungs, sq(640, 480), Fit::Contain, Orientation::Auto, true);
    let dims: Vec<_> = kept.iter().map(|r| (r.width, r.height)).collect();
    assert_eq!(dims, vec![(1440, 1080), (960, 720), (480, 360)]);
}

#[test]
fn a_rung_asked_for_twice_is_not_merged() {
    // The same box twice is the caller's own choice (two qualities, say).
    let rungs = vec![Rung::new(1280, 720), Rung::new(1280, 720).with_label("720p-low")];
    let (kept, report) = fit_rungs(&rungs, sq(640, 480), Fit::Contain, Orientation::Auto, false);
    assert_eq!(kept.len(), 2);
    assert!(report.iter().all(|r| r.duplicate_of.is_none()));
    assert_eq!(kept[1].label, "720p-low", "a caller's label is kept");
}

#[test]
fn labels_follow_the_output_and_stay_unique() {
    let rungs = vec![Rung::new(1280, 720), Rung::new(1280, 720).with_fit(Fit::Pad)];
    let (kept, _) = fit_rungs(&rungs, sq(1440, 1080), Fit::Contain, Orientation::Auto, false);
    let got: Vec<_> = kept.iter().map(|r| (r.width, r.height, r.label.as_str())).collect();
    assert_eq!(got, vec![(960, 720, "720p"), (1280, 720, "720p-2")]);
    // A portrait output is labelled by its short side.
    let (kept, _) = fit_rungs(&[Rung::new(1920, 1080)], sq(720, 1280), Fit::Contain, Orientation::Auto, false);
    assert_eq!((kept[0].width, kept[0].height, kept[0].label.as_str()), (720, 1280, "720p"));
}

#[test]
fn a_rungs_own_settings_win_over_the_spec() {
    let rungs = vec![
        Rung::new(1920, 1080),
        Rung::new(1080, 1920).with_fit(Fit::Cover).with_orientation(Orientation::Fixed).with_upscale(true),
    ];
    let (kept, _) = fit_rungs(&rungs, sq(1920, 1080), Fit::Contain, Orientation::Auto, false);
    assert_eq!((kept[0].width, kept[0].height), (1920, 1080));
    assert_eq!((kept[1].width, kept[1].height), (1080, 1920));
    assert!(kept[1].placement.unwrap().crops());
}

#[test]
fn the_filters_that_change_size_are_followed() {
    use codec::filter::VideoFilter;
    let src = SourceShape { width: 1920, height: 1080, sample_aspect: (4, 3) };
    let cropped = filtered_shape(src, &[VideoFilter::Crop { w: 1000, h: 5000, x: None, y: None }]);
    assert_eq!((cropped.width, cropped.height), (1000, 1080));
    let turned = filtered_shape(src, &[VideoFilter::Rotate(90), VideoFilter::HFlip]);
    assert_eq!((turned.width, turned.height, turned.sample_aspect), (1080, 1920, (3, 4)));
    let padded = filtered_shape(sq(640, 480), &[VideoFilter::Pad { w: 640, h: 640, x: None, y: None }]);
    assert_eq!((padded.width, padded.height), (640, 640));
}

#[test]
fn a_frame_of_another_size_is_fitted_into_the_same_output() {
    use bytes::Bytes;
    use codec::frame::{ColorSpace, PixelFormat};
    let frame = |w: u32, h: u32| {
        VideoFrame::new(
            Bytes::from(vec![128u8; (w * h * 3 / 2) as usize]),
            w,
            h,
            PixelFormat::Yuv420p,
            ColorSpace::Bt709,
            0,
        )
    };
    let p = contain(sq(64, 48), (32, 32));
    assert_eq!(p.canvas, (32, 24));
    assert_eq!(dims(&p.apply(&frame(64, 48)).unwrap()), (32, 24));
    // The next clip of a splice is square: letterboxed into the same 32x24.
    assert_eq!(dims(&p.apply(&frame(48, 48)).unwrap()), (32, 24));
    let cover = place(sq(64, 48), (32, 32), Fit::Cover, Orientation::Auto, false);
    assert_eq!(dims(&cover.apply(&frame(20, 80)).unwrap()), cover.canvas);
}

fn dims(f: &VideoFrame) -> (u32, u32) {
    (f.width, f.height)
}

/// A frame with a disc drawn on it as it is *shown*: round on screen, so in
/// stored samples it is `1 / sar` as wide as it is tall. Luma 235 inside,
/// 16 outside; neutral chroma.
fn disc_frame(src: SourceShape, radius_shown: f64) -> VideoFrame {
    use bytes::Bytes;
    use codec::frame::{ColorSpace, PixelFormat};
    let (w, h) = (src.width, src.height);
    let sar = f64::from(src.sample_aspect.0) / f64::from(src.sample_aspect.1);
    let (cx, cy) = (f64::from(w) / 2.0, f64::from(h) / 2.0);
    let mut data = Vec::with_capacity((w * h) as usize * 3 / 2 + 4);
    for y in 0..h {
        for x in 0..w {
            // Distance on screen: a sample is `sar` wide.
            let dx = (f64::from(x) + 0.5 - cx) * sar;
            let dy = f64::from(y) + 0.5 - cy;
            data.push(if dx.hypot(dy) <= radius_shown { 235 } else { 16 });
        }
    }
    data.resize(data.len() + 2 * (w.div_ceil(2) * h.div_ceil(2)) as usize, 128);
    VideoFrame::new(Bytes::from(data), w, h, PixelFormat::Yuv420p, ColorSpace::Bt709, 0)
}

/// The width and height of the bright region's bounding box in `f`'s luma.
fn disc_extent(f: &VideoFrame) -> (u32, u32) {
    let (w, h) = (f.width as usize, f.height as usize);
    let (mut x0, mut x1, mut y0, mut y1) = (usize::MAX, 0, usize::MAX, 0);
    for y in 0..h {
        for x in 0..w {
            if f.data[y * w + x] > 125 {
                (x0, x1, y0, y1) = (x0.min(x), x1.max(x), y0.min(y), y1.max(y));
            }
        }
    }
    ((x1 + 1 - x0) as u32, (y1 + 1 - y0) as u32)
}

#[test]
fn a_disc_stays_round_through_every_fit_but_stretch() {
    let sources = [
        sq(640, 480),
        sq(1280, 548),
        sq(360, 640),
        sq(400, 400),
        sq(853, 480),
        SourceShape { width: 720, height: 576, sample_aspect: (64, 45) },
        SourceShape { width: 720, height: 480, sample_aspect: (10, 11) },
    ];
    for src in sources {
        let radius = f64::from(src.width.min(src.height)) * 0.2;
        let frame = disc_frame(src, radius);
        for bx in [(1280, 720), (640, 360), (480, 480)] {
            for fit in [Fit::Contain, Fit::Cover, Fit::Pad] {
                for upscale in [false, true] {
                    let p = place(src, bx, fit, Orientation::Auto, upscale);
                    let scaled = p.apply(&frame).unwrap();
                    assert_eq!((scaled.width, scaled.height), p.canvas);
                    let (dw, dh) = disc_extent(&scaled);
                    let roundness = f64::from(dw) / f64::from(dh);
                    assert!(
                        (roundness - 1.0).abs() <= 0.04 + 2.0 / f64::from(dh),
                        "{src:?} into {bx:?} by {fit} (upscale {upscale}): the disc came out {dw}x{dh} in {:?}",
                        p.canvas
                    );
                }
            }
        }
    }
    // The measure can see distortion: the old behaviour squashes the disc.
    let src = sq(640, 480);
    let stretched = place(src, (1280, 720), Fit::Stretch, Orientation::Auto, false).apply(&disc_frame(src, 168.0)).unwrap();
    let (dw, dh) = disc_extent(&stretched);
    assert!(f64::from(dw) / f64::from(dh) > 1.25, "stretch left the disc round: {dw}x{dh}");
}
