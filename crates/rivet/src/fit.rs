//! Fitting a source into a rung's box — how an explicit `WxH` rung is sized.
//!
//! A rung's width and height are a **bounding box**, not the output size. The
//! output keeps the source's display shape (non-square samples included) and,
//! unless asked, is never larger than the source. How the picture meets the
//! box is the [`Fit`]:
//!
//! | fit | output | picture |
//! |---|---|---|
//! | `contain` (default) | inside the box, the source's shape | whole |
//! | `cover` | the box's shape | centre-cropped to fill it |
//! | `pad` | exactly the box | whole, letterboxed or pillarboxed in black |
//! | `stretch` | exactly the box | whole, distorted to fill it |
//!
//! `stretch` is what every explicit rung did before fitting existed: a 640x480
//! source through a 1280x720 rung came out 1280x720, stretched sideways and
//! upscaled. It is kept for a caller who asks for it by name.
//!
//! The box turns with the source ([`Orientation::Auto`]): a 1920x1080 rung on
//! a portrait source is read as 1080x1920, so a phone video keeps its shape
//! through a landscape ladder. A rung that wants its box as written — a 9:16
//! social rung that crops a landscape source to vertical — is
//! [`Orientation::Fixed`].
//!
//! Without `upscale`, a source smaller than the box comes out at its own size
//! (even-aligned). Rungs that collapse onto the same output that way are
//! merged by [`fit_rungs`], which reports the ones it dropped.
//!
//! The output always has square samples: an anamorphic 720x576 at 64:45 is
//! resized as the 1024x576 picture it is shown as.

use std::fmt;

use anyhow::{Result, bail};
use codec::frame::VideoFrame;

use crate::spec::Rung;

/// How a source meets a rung's box. See the [module docs](self).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum Fit {
    /// Inside the box, keeping the source's shape. The default.
    #[default]
    Contain,
    /// Fill the box, keeping the source's shape, and centre-crop the overflow.
    Cover,
    /// Contain, then letterbox or pillarbox to exactly the box in black.
    Pad,
    /// Exactly the box, whatever the source's shape: the picture is distorted
    /// to fill it. Ignores orientation and `upscale`.
    Stretch,
}

impl Fit {
    /// Every fit, in the order they are documented.
    pub const ALL: [Fit; 4] = [Fit::Contain, Fit::Cover, Fit::Pad, Fit::Stretch];

    /// Read `contain`, `cover`, `pad` or `stretch`.
    pub fn parse(s: &str) -> Result<Self> {
        Ok(match s.trim().to_ascii_lowercase().as_str() {
            "contain" => Fit::Contain,
            "cover" => Fit::Cover,
            "pad" => Fit::Pad,
            "stretch" => Fit::Stretch,
            other => bail!("fit must be contain, cover, pad or stretch (got '{other}')"),
        })
    }

    /// The name [`Fit::parse`] reads.
    pub fn as_str(self) -> &'static str {
        match self {
            Fit::Contain => "contain",
            Fit::Cover => "cover",
            Fit::Pad => "pad",
            Fit::Stretch => "stretch",
        }
    }
}

impl fmt::Display for Fit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Whether a rung's box turns to match the source's orientation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum Orientation {
    /// The box is long side x short side: a landscape box on a portrait
    /// source is used portrait. The default.
    #[default]
    Auto,
    /// The box is used as written, whatever the source's orientation.
    Fixed,
}

impl Orientation {
    /// Read `auto` or `fixed`.
    pub fn parse(s: &str) -> Result<Self> {
        Ok(match s.trim().to_ascii_lowercase().as_str() {
            "auto" => Orientation::Auto,
            "fixed" => Orientation::Fixed,
            other => bail!("orientation must be auto or fixed (got '{other}')"),
        })
    }

    /// The name [`Orientation::parse`] reads.
    pub fn as_str(self) -> &'static str {
        match self {
            Orientation::Auto => "auto",
            Orientation::Fixed => "fixed",
        }
    }
}

impl fmt::Display for Orientation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The picture a rung is fitted to: the frames' size as they reach the
/// scaler (upright, after the filters) and the shape of one sample.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SourceShape {
    pub width: u32,
    pub height: u32,
    /// `(1, 1)` for square pixels.
    pub sample_aspect: (u32, u32),
}

impl SourceShape {
    /// A source with square pixels.
    pub fn square(width: u32, height: u32) -> Self {
        Self { width, height, sample_aspect: (1, 1) }
    }

    fn sar(&self) -> f64 {
        match self.sample_aspect {
            (w, h) if w > 0 && h > 0 => f64::from(w) / f64::from(h),
            _ => 1.0,
        }
    }

    /// The size the picture is shown at in square pixels. A wide sample
    /// widens the picture and a tall one heightens it, so neither side loses
    /// the resolution the source has: 720x576 at 64:45 is 1024x576.
    pub fn display_size(&self) -> (f64, f64) {
        let (w, h, sar) = (f64::from(self.width), f64::from(self.height), self.sar());
        if sar >= 1.0 { (w * sar, h) } else { (w, h / sar) }
    }

    /// Width over height as shown.
    pub fn display_aspect(&self) -> f64 {
        let (w, h) = self.display_size();
        if h > 0.0 { w / h } else { 0.0 }
    }
}

/// Where one rung's picture comes from and where it goes: crop `crop` out of
/// a `source`-sized frame, scale it to `scaled`, and place it at `offset` on a
/// `canvas` (the rung's output size) of black.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Placement {
    pub fit: Fit,
    /// The frame size this was planned for.
    pub source: (u32, u32),
    pub sample_aspect: (u32, u32),
    /// `x, y, w, h` in source samples; the whole frame when nothing is cut.
    pub crop: (u32, u32, u32, u32),
    pub scaled: (u32, u32),
    pub offset: (u32, u32),
    /// The output frame size.
    pub canvas: (u32, u32),
    pub upscale: bool,
}

impl Placement {
    /// A plain resize of a `source`-sized frame to exactly `canvas` — the
    /// placement of a rung nothing was fitted to.
    pub fn stretch(source: (u32, u32), canvas: (u32, u32)) -> Self {
        Self {
            fit: Fit::Stretch,
            source,
            sample_aspect: (1, 1),
            crop: (0, 0, source.0, source.1),
            scaled: canvas,
            offset: (0, 0),
            canvas,
            upscale: true,
        }
    }

    /// Whether any of the source is cut away.
    pub fn crops(&self) -> bool {
        self.crop != (0, 0, self.source.0, self.source.1)
    }

    /// Whether the output has bars.
    pub fn pads(&self) -> bool {
        self.scaled != self.canvas
    }

    /// Produce this rung's frame from a decoded one.
    ///
    /// A frame of another size than planned — the next clip of a splice, a
    /// transport stream that changes resolution — is fitted into the same
    /// canvas afresh, since the encoder's size is already fixed: `contain`
    /// and `pad` letterbox it, `cover` crops it, `stretch` stretches it.
    pub fn apply(&self, frame: &VideoFrame) -> Result<VideoFrame> {
        let p = if (frame.width, frame.height) == self.source {
            *self
        } else {
            let shape = SourceShape {
                width: frame.width,
                height: frame.height,
                sample_aspect: self.sample_aspect,
            };
            let fit = match self.fit {
                Fit::Contain | Fit::Pad => Fit::Pad,
                other => other,
            };
            let p = place(shape, self.canvas, fit, Orientation::Fixed, self.upscale || fit == Fit::Cover);
            debug_assert_eq!(p.canvas, self.canvas);
            p
        };
        codec::colorspace::scale_region(frame, p.crop, p.scaled, p.canvas, p.offset)
    }
}

/// `v` to the nearest even number, at least 2.
fn even_round(v: f64) -> u32 {
    (((v / 2.0).round() as u32) * 2).max(2)
}

/// `v` down to an even number, at least 2 — for a size that must not exceed
/// what the source has. The epsilon keeps 1023.9999… from a ratio at 1024.
fn even_floor(v: f64) -> u32 {
    (((v + 1e-6).floor() as u32) & !1).max(2)
}

/// Plan how a `source` meets a `box_w x box_h` box. See the [module docs](self).
pub fn place(source: SourceShape, (box_w, box_h): (u32, u32), fit: Fit, orientation: Orientation, upscale: bool) -> Placement {
    let whole = (0, 0, source.width, source.height);
    if fit == Fit::Stretch {
        return Placement {
            sample_aspect: source.sample_aspect,
            ..Placement::stretch((source.width, source.height), (box_w, box_h))
        };
    }
    let (dw, dh) = source.display_size();
    let aspect = dw / dh;
    // Long side x short side: the box takes the source's orientation. A
    // square source or a square box has none to take.
    let (bw, bh) = if orientation == Orientation::Auto && box_w != box_h && aspect != 1.0 && (aspect > 1.0) != (box_w > box_h) {
        (box_h, box_w)
    } else {
        (box_w, box_h)
    };
    let (bwf, bhf) = (f64::from(bw), f64::from(bh));
    let placement = |crop, scaled, offset, canvas| Placement {
        fit,
        source: (source.width, source.height),
        sample_aspect: source.sample_aspect,
        crop,
        scaled,
        offset,
        canvas,
        upscale,
    };

    match fit {
        Fit::Contain | Fit::Pad => {
            let mut s = (bwf / dw).min(bhf / dh);
            if !upscale {
                s = s.min(1.0);
            }
            let (mut w, mut h) = (even_round(dw * s).min(bw), even_round(dh * s).min(bh));
            if !upscale {
                (w, h) = (w.min(even_floor(dw)), h.min(even_floor(dh)));
            }
            if fit == Fit::Contain {
                placement(whole, (w, h), (0, 0), (w, h))
            } else {
                let offset = (((bw - w) / 2) & !1, ((bh - h) / 2) & !1);
                placement(whole, (w, h), offset, (bw, bh))
            }
        }
        Fit::Cover => {
            // The largest box-shaped output the source can fill without being
            // enlarged, when it cannot fill the box itself.
            let k = if upscale { 1.0 } else { (dw / bwf).min(dh / bhf).min(1.0) };
            let (cw, ch) = if k < 1.0 {
                (
                    even_round(bwf * k).min(even_floor(dw)).min(bw),
                    even_round(bhf * k).min(even_floor(dh)).min(bh),
                )
            } else {
                (bw, bh)
            };
            // The crop, in stored samples, with the canvas's display shape.
            let target = f64::from(cw) / f64::from(ch);
            let (sw, sh) = (f64::from(source.width), f64::from(source.height));
            let sar = source.sar();
            let (crop_w, crop_h) = if aspect > target {
                ((sh * target / sar).round().min(sw), sh)
            } else {
                (sw, (sw * sar / target).round().min(sh))
            };
            let (crop_w, crop_h) = (crop_w.max(1.0) as u32, crop_h.max(1.0) as u32);
            let x = ((source.width - crop_w) / 2) & !1;
            let y = ((source.height - crop_h) / 2) & !1;
            placement((x, y, crop_w, crop_h), (cw, ch), (0, 0), (cw, ch))
        }
        Fit::Stretch => unreachable!("returned above"),
    }
}

/// What fitting did to one requested rung.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FittedRung {
    /// The rung's label after fitting — the label of its output, or, for a
    /// dropped rung, the label it would have had.
    pub label: String,
    /// The box that was asked for.
    pub requested: (u32, u32),
    /// The output size.
    pub output: (u32, u32),
    pub fit: Fit,
    /// For a rung that came out the same as an earlier one and was dropped:
    /// that rung's position in the request. `None` for a rung that is
    /// produced.
    pub duplicate_of: Option<usize>,
}

/// Fit every rung to `source`: set each one's output size and
/// [`Placement`], relabel the ones carrying their automatic label, and drop a
/// rung whose output is the same as an earlier one's (the first is kept).
///
/// Returns the rungs to produce and, in request order, what became of each
/// requested rung. A rung that already has a placement (a caller that fitted
/// it) is left alone.
pub fn fit_rungs(
    rungs: &[Rung],
    source: SourceShape,
    fit: Fit,
    orientation: Orientation,
    upscale: bool,
) -> (Vec<Rung>, Vec<FittedRung>) {
    let mut kept: Vec<Rung> = Vec::with_capacity(rungs.len());
    // Request index of each kept rung, and the (output, placement) it got.
    let mut kept_from: Vec<usize> = Vec::with_capacity(rungs.len());
    let mut report = Vec::with_capacity(rungs.len());
    for (index, rung) in rungs.iter().enumerate() {
        let requested = (rung.width, rung.height);
        let placement = rung.placement.unwrap_or_else(|| {
            place(
                source,
                requested,
                rung.fit.unwrap_or(fit),
                rung.orientation.unwrap_or(orientation),
                rung.upscale.unwrap_or(upscale),
            )
        });
        let auto_label = rung.label == format!("{}p", rung.short_side());
        let mut fitted = rung.clone();
        fitted.width = placement.canvas.0;
        fitted.height = placement.canvas.1;
        fitted.placement = Some(placement);
        if auto_label {
            fitted.label = format!("{}p", fitted.short_side());
        }
        // Same output size and the same picture in it, from a different box:
        // two rungs the source collapsed into one. Rungs asked for with the
        // same box are the caller's business and both run.
        let duplicate = kept.iter().zip(&kept_from).find(|(k, from)| {
            k.placement == fitted.placement && (rungs[**from].width, rungs[**from].height) != requested
        });
        if let Some((k, &from)) = duplicate {
            report.push(FittedRung {
                label: k.label.clone(),
                requested,
                output: placement.canvas,
                fit: placement.fit,
                duplicate_of: Some(from),
            });
            continue;
        }
        // Labels name files and HLS renditions, so they stay unique.
        if kept.iter().any(|k| k.label == fitted.label) {
            let base = fitted.label.clone();
            fitted.label = (2..).map(|n| format!("{base}-{n}")).find(|l| kept.iter().all(|k| &k.label != l)).unwrap_or(base);
        }
        report.push(FittedRung {
            label: fitted.label.clone(),
            requested,
            output: placement.canvas,
            fit: placement.fit,
            duplicate_of: None,
        });
        kept.push(fitted);
        kept_from.push(index);
    }
    for r in report.iter().filter(|r| r.duplicate_of.is_some()) {
        tracing::info!(
            requested = %format!("{}x{}", r.requested.0, r.requested.1),
            output = %format!("{}x{}", r.output.0, r.output.1),
            same_as = %r.label,
            "rung dropped: the source is smaller than its box, and it comes out the same as another rung (upscale is off)"
        );
    }
    (kept, report)
}

/// The size frames reach the rung scalers at: `upright` (the source turned
/// upright) through the size-changing `filters` (crop, pad, rotate), with
/// the sample shape a quarter-turn rotate swaps.
pub fn filtered_shape(upright: SourceShape, filters: &[codec::filter::VideoFilter]) -> SourceShape {
    use codec::filter::VideoFilter;
    let even = |v: u32| v & !1;
    filters.iter().fold(upright, |s, f| match f {
        VideoFilter::Crop { w, h, x: Some(_), y: Some(_) } => SourceShape { width: even(*w), height: even(*h), ..s },
        VideoFilter::Crop { w, h, .. } => {
            SourceShape { width: even((*w).min(s.width)), height: even((*h).min(s.height)), ..s }
        }
        VideoFilter::Pad { w, h, .. } => {
            SourceShape { width: even((*w).max(s.width)), height: even((*h).max(s.height)), ..s }
        }
        VideoFilter::Rotate(90 | 270) => SourceShape {
            width: s.height,
            height: s.width,
            sample_aspect: (s.sample_aspect.1, s.sample_aspect.0),
        },
        _ => s,
    })
}

#[cfg(test)]
mod tests;
