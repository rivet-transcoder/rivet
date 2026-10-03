//! Fitting a picture into each rendition's box, and resampling it there.
//!
//! The plan is [`crate::fit`]'s, the one every video rung gets — contain,
//! cover, pad, stretch; the box turning with the picture; never enlarged
//! unless asked — on a one-pixel grid, since a still has no 4:2:0 chroma to
//! keep even. Resampling is Lanczos-3 on premultiplied RGBA, so a transparent
//! edge does not pull its hidden colour into the pixels beside it.

use anyhow::Result;

use super::colour::Source;
use super::decode::Picture;
use super::raster::RgbaImage;
use super::{ImageSpec, MergedRendition};
use crate::fit::{Fit, Orientation, Placement, SourceShape, place_aligned};

/// One output size to make.
#[derive(Debug, Clone)]
pub(crate) struct Plan {
    /// Its position in the spec's renditions.
    pub(crate) rendition: usize,
    pub(crate) label: String,
    pub(crate) placement: Placement,
}

/// The pixels of one output, and the profile to embed with them.
pub(crate) struct Pixels<'a> {
    pub(crate) image: RgbaImage,
    /// Whether any pixel is less than opaque.
    pub(crate) alpha: bool,
    pub(crate) icc: Option<&'a [u8]>,
}

/// Plan every rendition of `spec` against `picture`: its output size and how
/// the picture meets it. A rendition that comes out the same as an earlier one
/// from a different box (a small picture under two larger boxes, with
/// `upscale` off) is made once, and reported.
pub(crate) fn plan(picture: &Picture, spec: &ImageSpec) -> (Vec<Plan>, Vec<MergedRendition>) {
    let (w, h) = picture.rgba.dimensions();
    let shape = SourceShape { width: w, height: h, sample_aspect: picture.sample_aspect };
    let requests: Vec<((u32, u32), Fit, Orientation, bool)> = if spec.renditions.is_empty() {
        // The picture at its own size — as shown, for an anamorphic frame.
        let (dw, dh) = shape.display_size();
        let size = ((dw.round() as u32).max(1), (dh.round() as u32).max(1));
        vec![(size, Fit::Contain, Orientation::Fixed, true)]
    } else {
        spec.renditions
            .iter()
            .map(|r| {
                (
                    (r.width, r.height),
                    r.fit.unwrap_or(spec.fit),
                    r.orientation.unwrap_or(spec.orientation),
                    r.upscale.unwrap_or(spec.upscale),
                )
            })
            .collect()
    };

    let mut plans: Vec<Plan> = Vec::with_capacity(requests.len());
    let mut merged = Vec::new();
    for (index, &(requested, fit, orientation, upscale)) in requests.iter().enumerate() {
        let placement = place_aligned(shape, requested, fit, orientation, upscale, 1);
        // Same output and the same picture in it, from a different box: two
        // renditions the picture collapsed into one. The same box asked for
        // twice is the caller's business, and both are made.
        if let Some(earlier) =
            plans.iter().find(|p| p.placement == placement && requests[p.rendition].0 != requested)
        {
            merged.push(MergedRendition { rendition: index, same_as: earlier.rendition, output: placement.canvas });
            continue;
        }
        let base = format!("{}x{}", placement.canvas.0, placement.canvas.1);
        let label = if plans.iter().any(|p| p.label == base) {
            (2..).map(|n| format!("{base}-{n}")).find(|l| plans.iter().all(|p| &p.label != l)).unwrap_or(base)
        } else {
            base
        };
        plans.push(Plan { rendition: index, label, placement });
    }
    (plans, merged)
}

/// Make one output's pixels: crop, resample, and place on its canvas. The
/// canvas's bars (`pad`) are transparent when the output can hold that and
/// the picture has transparency of its own, and black otherwise, as a video's
/// are.
pub(crate) fn apply<'a>(source: Source<'a>, plan: &Plan, keeps_alpha: bool) -> Result<Pixels<'a>> {
    let picture = source.picture;
    let p = &plan.placement;
    let (cx, cy, cw, ch) = p.crop;
    let whole = (cx, cy, cw, ch) == (0, 0, picture.rgba.width(), picture.rgba.height());
    let cropped;
    let region = if whole {
        &picture.rgba
    } else {
        cropped = picture.rgba.crop(cx, cy, cw, ch);
        &cropped
    };
    let scaled = if (cw, ch) == p.scaled {
        region.clone()
    } else {
        resample(region, p.scaled, picture.alpha)
    };
    let image = if p.scaled == p.canvas {
        scaled
    } else {
        let bar = if keeps_alpha && picture.alpha { [0, 0, 0, 0] } else { [0, 0, 0, u8::MAX] };
        let mut canvas = RgbaImage::from_pixel(p.canvas.0, p.canvas.1, bar);
        canvas.replace(&scaled, i64::from(p.offset.0), i64::from(p.offset.1));
        canvas
    };
    let alpha = picture.alpha || image.has_alpha();
    Ok(Pixels { image, alpha, icc: source.icc })
}

/// Lanczos-3, premultiplied when the picture has transparency.
fn resample(image: &RgbaImage, (w, h): (u32, u32), alpha: bool) -> RgbaImage {
    if !alpha {
        return image.resize(w, h);
    }
    let mut pre = image.clone();
    for px in pre.pixels_mut() {
        let a = u32::from(px[3]);
        for c in &mut px[..3] {
            *c = ((u32::from(*c) * a + 127) / 255) as u8;
        }
    }
    let mut out = pre.resize(w, h);
    for px in out.pixels_mut() {
        let a = u32::from(px[3]);
        if a == 0 {
            *px = [0, 0, 0, 0];
            continue;
        }
        for c in &mut px[..3] {
            *c = ((u32::from(*c) * 255 + a / 2) / a).min(255) as u8;
        }
    }
    out
}
