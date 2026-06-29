//! Video filters — per-frame transforms applied to decoded frames **before**
//! per-rung scaling and encoding.
//!
//! ## Layout
//!
//! The canonical representation is a list of [`VideoFilter`] **values**. This
//! `mod.rs` owns the cross-cutting pieces — the enum, the textual / structured
//! parsers, the [`apply`] dispatch, the [`FilterChain`], and the shared plane
//! helpers — while **each filter's implementation lives in its own file**:
//! [`crop`], [`pad`], [`hflip`], [`vflip`], [`rotate`], [`grayscale`],
//! [`overlay`], [`invert`], [`brightness`], [`contrast`], [`saturation`], and
//! the [`denoise`] family (one file per algorithm under `denoise/`).
//!
//! Two kinds of filter:
//!
//! - **Stateless** ([`apply`] runs them directly): crop, pad, hflip, vflip,
//!   rotate, grayscale (geometry, any bit depth); invert, brightness, contrast,
//!   saturation (colour, 8-bit); and `denoise` (selectable algorithm, 8-bit).
//! - **Resource** filters need one-time setup — `overlay` loads its PNG and
//!   converts it to YUV + alpha. Build a [`FilterChain`] with
//!   [`FilterChain::prepare`] (loads overlays once) and call
//!   [`FilterChain::apply`] per frame.
//!
//! Two interchangeable serializations (they round-trip:
//! `parse_chain(&chain_to_string(c)) == c`):
//!
//! - **Structured** objects (serde feature) — a YAML/JSON DSL writes a chain as
//!   a list of objects: `[{crop: {w,h}}, hflip, {overlay: {image: "logo.png"}}]`.
//! - **Textual** ffmpeg-`-vf` style — [`parse_chain`] / [`Display`]:
//!   `crop=1280:720,hflip,overlay=logo.png:24:24`.

use std::fmt;

use anyhow::{Context, Result, bail};
use bytes::BytesMut;

use crate::frame::{PixelFormat, VideoFrame};

mod brightness;
mod contrast;
mod crop;
mod denoise;
mod grayscale;
mod hflip;
mod invert;
mod overlay;
mod pad;
mod rotate;
mod saturation;
mod vflip;

#[cfg(test)]
mod tests;

pub use denoise::DenoiseMethod;

/// One video-filter step. The canonical, code-interpreted representation.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum VideoFilter {
    /// Crop a `w×h` region. Centred when `x`/`y` are omitted, else at `(x, y)`.
    Crop {
        w: u32,
        h: u32,
        #[cfg_attr(feature = "serde", serde(default, skip_serializing_if = "Option::is_none"))]
        x: Option<u32>,
        #[cfg_attr(feature = "serde", serde(default, skip_serializing_if = "Option::is_none"))]
        y: Option<u32>,
    },
    /// Pad into a `w×h` canvas (neutral black). Centred when `x`/`y` are omitted.
    Pad {
        w: u32,
        h: u32,
        #[cfg_attr(feature = "serde", serde(default, skip_serializing_if = "Option::is_none"))]
        x: Option<u32>,
        #[cfg_attr(feature = "serde", serde(default, skip_serializing_if = "Option::is_none"))]
        y: Option<u32>,
    },
    /// Mirror horizontally (left↔right).
    #[cfg_attr(feature = "serde", serde(rename = "hflip"))]
    HFlip,
    /// Mirror vertically (top↔bottom).
    #[cfg_attr(feature = "serde", serde(rename = "vflip"))]
    VFlip,
    /// Rotate clockwise by 90, 180, or 270 degrees (90/270 swap width↔height).
    Rotate(u32),
    /// Drop chroma — set U/V to neutral so the image is grayscale.
    Grayscale,
    /// Alpha-composite a PNG (logo / watermark) at top-left `(x, y)`. 8-bit only.
    Overlay {
        /// Path to a PNG image (with or without an alpha channel).
        image: String,
        #[cfg_attr(feature = "serde", serde(default))]
        x: u32,
        #[cfg_attr(feature = "serde", serde(default))]
        y: u32,
    },
    /// Invert (negate) luma + chroma. 8-bit only.
    Invert,
    /// Add a luma offset (`-255..=255`); brighten/darken. 8-bit only.
    Brightness(i32),
    /// Scale luma contrast around mid-grey (`1.0` = unchanged). 8-bit only.
    Contrast(f32),
    /// Scale chroma saturation around neutral (`0` = grayscale, `1.0` = unchanged). 8-bit only.
    Saturation(f32),
    /// Spatial **denoise** with a selectable algorithm (see [`DenoiseMethod`])
    /// and a `strength` in `0.0..=1.0` (default `0.5`) that blends the filtered
    /// result back with the source (`0` = off, `1` = fully denoised). Applied to
    /// luma + chroma. 8-bit only.
    Denoise {
        #[cfg_attr(feature = "serde", serde(default))]
        method: DenoiseMethod,
        #[cfg_attr(feature = "serde", serde(default = "denoise::default_denoise_strength"))]
        strength: f32,
    },
}

impl fmt::Display for VideoFilter {
    /// The textual (ffmpeg-`-vf`) token for this filter.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VideoFilter::Crop { w, h, x: Some(x), y: Some(y) } => write!(f, "crop={w}:{h}:{x}:{y}"),
            VideoFilter::Crop { w, h, .. } => write!(f, "crop={w}:{h}"),
            VideoFilter::Pad { w, h, x: Some(x), y: Some(y) } => write!(f, "pad={w}:{h}:{x}:{y}"),
            VideoFilter::Pad { w, h, .. } => write!(f, "pad={w}:{h}"),
            VideoFilter::HFlip => write!(f, "hflip"),
            VideoFilter::VFlip => write!(f, "vflip"),
            VideoFilter::Rotate(d) => write!(f, "rotate={d}"),
            VideoFilter::Grayscale => write!(f, "grayscale"),
            VideoFilter::Overlay { image, x, y } => write!(f, "overlay={image}:{x}:{y}"),
            VideoFilter::Invert => write!(f, "invert"),
            VideoFilter::Brightness(b) => write!(f, "brightness={b}"),
            VideoFilter::Contrast(c) => write!(f, "contrast={c}"),
            VideoFilter::Saturation(s) => write!(f, "saturation={s}"),
            VideoFilter::Denoise { method, strength } => write!(f, "denoise={method}:{strength}"),
        }
    }
}

/// A whole chain as a comma-separated textual string (the inverse of
/// [`parse_chain`]).
pub fn chain_to_string(chain: &[VideoFilter]) -> String {
    chain.iter().map(|f| f.to_string()).collect::<Vec<_>>().join(",")
}

/// A filter chain in either form, for a DSL field that should accept both a
/// structured list or a string. Resolve with [`FilterSpec::resolve`].
#[cfg(feature = "serde")]
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[serde(untagged)]
pub enum FilterSpec {
    /// An ffmpeg-`-vf`-style chain string, e.g. `"crop=1280:720,hflip"`.
    Chain(String),
    /// A structured list of filters.
    List(Vec<VideoFilter>),
}

#[cfg(feature = "serde")]
impl FilterSpec {
    /// Resolve to the concrete, **validated** filter list. The string form is
    /// validated by [`parse_chain`]; the structured form is validated by
    /// round-tripping through its textual rendering, so e.g. `rotate: 45` is
    /// rejected at config time rather than at apply time.
    pub fn resolve(&self) -> Result<Vec<VideoFilter>> {
        match self {
            FilterSpec::Chain(s) => parse_chain(s),
            FilterSpec::List(v) => parse_chain(&chain_to_string(v)),
        }
    }

    /// Collapse to the chain-string form (for string-only surfaces).
    pub fn to_chain(&self) -> String {
        match self {
            FilterSpec::Chain(s) => s.clone(),
            FilterSpec::List(v) => chain_to_string(v),
        }
    }
}

/// Parse an ffmpeg-`-vf`-style chain, e.g. `"crop=1280:720,hflip"`.
pub fn parse_chain(s: &str) -> Result<Vec<VideoFilter>> {
    let mut out = Vec::new();
    for part in s.split(',').map(str::trim).filter(|p| !p.is_empty()) {
        out.push(parse_one(part)?);
    }
    if out.is_empty() {
        bail!("empty filter chain");
    }
    Ok(out)
}

fn parse_one(spec: &str) -> Result<VideoFilter> {
    let (name, args) = match spec.split_once('=') {
        Some((n, a)) => (n.trim(), a.trim()),
        None => (spec.trim(), ""),
    };
    let parts: Vec<&str> = args.split(':').map(str::trim).filter(|s| !s.is_empty()).collect();
    let nums = || -> Result<Vec<u32>> {
        parts
            .iter()
            .map(|s| s.parse::<u32>().map_err(|_| anyhow::anyhow!("bad number '{s}' in '{spec}'")))
            .collect()
    };
    let one_f32 = || -> Result<f32> {
        parts
            .first()
            .ok_or_else(|| anyhow::anyhow!("'{name}' needs a value"))?
            .parse::<f32>()
            .map_err(|_| anyhow::anyhow!("bad number in '{spec}'"))
    };
    let f = match name {
        "crop" => match nums()?.as_slice() {
            [w, h] => VideoFilter::Crop { w: *w, h: *h, x: None, y: None },
            [w, h, x, y] => VideoFilter::Crop { w: *w, h: *h, x: Some(*x), y: Some(*y) },
            _ => bail!("crop wants W:H or W:H:X:Y, got '{args}'"),
        },
        "pad" => match nums()?.as_slice() {
            [w, h] => VideoFilter::Pad { w: *w, h: *h, x: None, y: None },
            [w, h, x, y] => VideoFilter::Pad { w: *w, h: *h, x: Some(*x), y: Some(*y) },
            _ => bail!("pad wants W:H or W:H:X:Y, got '{args}'"),
        },
        "hflip" => VideoFilter::HFlip,
        "vflip" => VideoFilter::VFlip,
        "rotate" | "transpose" => {
            let deg = if name == "transpose" { 90 } else { *nums()?.first().unwrap_or(&90) };
            if !matches!(deg, 90 | 180 | 270) {
                bail!("rotate wants 90|180|270, got {deg}");
            }
            VideoFilter::Rotate(deg)
        }
        "grayscale" | "gray" => VideoFilter::Grayscale,
        "overlay" => {
            // overlay=PATH[:X:Y] — PATH must not contain ':'.
            let image =
                parts.first().ok_or_else(|| anyhow::anyhow!("overlay needs a PATH"))?.to_string();
            let x = parts.get(1).map(|s| s.parse::<u32>()).transpose().map_err(|_| anyhow::anyhow!("bad overlay x in '{spec}'"))?.unwrap_or(0);
            let y = parts.get(2).map(|s| s.parse::<u32>()).transpose().map_err(|_| anyhow::anyhow!("bad overlay y in '{spec}'"))?.unwrap_or(0);
            VideoFilter::Overlay { image, x, y }
        }
        "invert" | "negate" => VideoFilter::Invert,
        "brightness" => {
            let b: i32 = parts.first().ok_or_else(|| anyhow::anyhow!("brightness needs a value"))?.parse().map_err(|_| anyhow::anyhow!("bad brightness in '{spec}'"))?;
            VideoFilter::Brightness(b)
        }
        "contrast" => VideoFilter::Contrast(one_f32()?),
        "saturation" => VideoFilter::Saturation(one_f32()?),
        "denoise" | "nr" => {
            // denoise[=METHOD][:STRENGTH] — METHOD is bilateral|gaussian|median|
            // mean|nlmeans|anisotropic (default bilateral); STRENGTH is 0..=1
            // (default 0.5). The two args are order-free: a token that parses as
            // a number is the strength, anything else is the method.
            let mut method = DenoiseMethod::Bilateral;
            let mut strength = 0.5f32;
            for &p in &parts {
                match p.parse::<f32>() {
                    Ok(s) => strength = s,
                    Err(_) => {
                        method = match p.to_ascii_lowercase().as_str() {
                            "bilateral" | "bl" => DenoiseMethod::Bilateral,
                            "gaussian" | "gauss" | "gs" => DenoiseMethod::Gaussian,
                            "median" | "md" => DenoiseMethod::Median,
                            "mean" | "box" | "average" => DenoiseMethod::Mean,
                            "nlmeans" | "nlm" => DenoiseMethod::Nlmeans,
                            "anisotropic" | "diffusion" | "pm" => DenoiseMethod::Anisotropic,
                            o => bail!(
                                "unknown denoise method '{o}' (want bilateral|gaussian|median|\
                                 mean|nlmeans|anisotropic)"
                            ),
                        };
                    }
                }
            }
            if !(0.0..=1.0).contains(&strength) {
                bail!("denoise strength must be 0.0..=1.0, got {strength}");
            }
            VideoFilter::Denoise { method, strength }
        }
        o => bail!("unknown filter '{o}'"),
    };
    Ok(f)
}

/// Apply a whole **stateless** chain to a frame, in order. Returns an error if
/// the chain contains an `overlay` (use [`FilterChain`] for that).
pub fn apply_chain(frame: VideoFrame, chain: &[VideoFilter]) -> Result<VideoFrame> {
    let mut f = frame;
    for filter in chain {
        f = apply(&f, filter)?;
    }
    Ok(f)
}

/// Apply one **stateless** filter, dispatching to its module. (`Overlay` errors
/// here — use [`FilterChain`].)
pub fn apply(frame: &VideoFrame, filter: &VideoFilter) -> Result<VideoFrame> {
    match filter {
        VideoFilter::Crop { w, h, x, y } => crop::apply(frame, *w, *h, *x, *y),
        VideoFilter::Pad { w, h, x, y } => pad::apply(frame, *w, *h, *x, *y),
        VideoFilter::HFlip => hflip::apply(frame),
        VideoFilter::VFlip => vflip::apply(frame),
        VideoFilter::Rotate(deg) => rotate::apply(frame, *deg),
        VideoFilter::Grayscale => grayscale::apply(frame),
        VideoFilter::Invert => invert::apply(frame),
        VideoFilter::Brightness(delta) => brightness::apply(frame, *delta),
        VideoFilter::Contrast(c) => contrast::apply(frame, *c),
        VideoFilter::Saturation(s) => saturation::apply(frame, *s),
        VideoFilter::Denoise { method, strength } => denoise::apply(frame, *method, *strength),
        VideoFilter::Overlay { .. } => {
            bail!("overlay is a resource filter — build a FilterChain::prepare(..) and call .apply()")
        }
    }
}

// ── shared plane helpers (used by the per-filter modules via `super::`) ───────

/// Bytes-per-sample for the supported 4:2:0 formats.
fn bps(format: PixelFormat) -> Result<usize> {
    match format {
        PixelFormat::Yuv420p => Ok(1),
        PixelFormat::Yuv420p10le => Ok(2),
        other => bail!("video filters need Yuv420p / Yuv420p10le, got {other:?}"),
    }
}

/// Split a frame into its (Y, U, V) plane byte slices for a `w×h` 4:2:0 frame.
fn planes(frame: &VideoFrame, bps: usize) -> Result<(&[u8], &[u8], &[u8])> {
    let w = frame.width as usize;
    let h = frame.height as usize;
    let y_len = w * h * bps;
    let c_len = (w / 2) * (h / 2) * bps;
    if frame.data.len() < y_len + 2 * c_len {
        bail!("frame data too small: {} < {} for {}x{}", frame.data.len(), y_len + 2 * c_len, w, h);
    }
    let (y, rest) = frame.data.split_at(y_len);
    let (u, v) = rest.split_at(c_len);
    Ok((y, &u[..c_len], &v[..c_len]))
}

/// Reassemble a frame from new Y/U/V planes + new dims.
fn assemble(src: &VideoFrame, w: u32, h: u32, y: Vec<u8>, u: Vec<u8>, v: Vec<u8>) -> VideoFrame {
    let mut data = BytesMut::with_capacity(y.len() + u.len() + v.len());
    data.extend_from_slice(&y);
    data.extend_from_slice(&u);
    data.extend_from_slice(&v);
    VideoFrame::new(data.freeze(), w, h, src.format, src.color_space, src.pts)
}

/// Require 8-bit `Yuv420p` for the colour / overlay / denoise filters and return
/// the planes (owned, so callers can mutate them).
fn planes_8bit(frame: &VideoFrame, what: &str) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    if frame.format != PixelFormat::Yuv420p {
        bail!("the `{what}` filter needs an 8-bit Yuv420p frame (got {:?}); it applies to SDR output", frame.format);
    }
    let (y, u, v) = planes(frame, 1)?;
    Ok((y.to_vec(), u.to_vec(), v.to_vec()))
}

/// Round `n` down to even (4:2:0 chroma alignment).
fn even(n: u32) -> u32 {
    n & !1
}

// ── prepared chain (loads overlays once, then applies per frame) ─────────────

enum Step {
    Plain(VideoFilter),
    Overlay(overlay::PreparedOverlay),
}

/// A filter chain with its resources prepared (overlay PNGs loaded + converted).
/// Build once with [`prepare`](FilterChain::prepare), then [`apply`](FilterChain::apply)
/// per frame.
pub struct FilterChain {
    steps: Vec<Step>,
}

impl FilterChain {
    /// Prepare a chain: load + convert every `overlay` image (the rest pass
    /// through). Fails if an overlay image can't be read or decoded.
    pub fn prepare(filters: &[VideoFilter]) -> Result<Self> {
        let mut steps = Vec::with_capacity(filters.len());
        for f in filters {
            match f {
                VideoFilter::Overlay { image, x, y } => {
                    let img = image::ImageReader::open(image)
                        .with_context(|| format!("opening overlay image '{image}'"))?
                        .decode()
                        .with_context(|| format!("decoding overlay image '{image}'"))?
                        .to_rgba8();
                    let (w, h) = (img.width(), img.height());
                    steps.push(Step::Overlay(overlay::PreparedOverlay::from_rgba(img.as_raw(), w, h, *x, *y)?));
                }
                other => steps.push(Step::Plain(other.clone())),
            }
        }
        Ok(Self { steps })
    }

    /// Apply the whole chain to a frame, in order.
    pub fn apply(&self, frame: VideoFrame) -> Result<VideoFrame> {
        let mut f = frame;
        for step in &self.steps {
            f = match step {
                Step::Plain(filt) => apply(&f, filt)?,
                Step::Overlay(ov) => ov.composite(&f)?,
            };
        }
        Ok(f)
    }

    /// No filters → applying is a no-op.
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }
}
