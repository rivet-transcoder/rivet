//! Video filters — per-frame transforms applied to decoded frames **before**
//! per-rung scaling and encoding.
//!
//! The canonical representation is a list of [`VideoFilter`] **values**. Two
//! kinds:
//!
//! - **Stateless** filters ([`apply`] runs them directly): crop, pad, hflip,
//!   vflip, rotate, grayscale (geometry, any bit depth); invert, brightness,
//!   contrast, saturation (colour, 8-bit); and `denoise` — a spatial denoise
//!   with a **selectable algorithm** (bilateral / gaussian / median / mean /
//!   nlmeans / anisotropic — see [`DenoiseMethod`]) and a strength blend, 8-bit.
//! - **Resource** filters need a one-time setup before they can run per frame —
//!   `overlay` loads its PNG and converts it to YUV + alpha. Build a
//!   [`FilterChain`] with [`FilterChain::prepare`] (loads overlays once) and
//!   call [`FilterChain::apply`] per frame.
//!
//! Two interchangeable serializations (they round-trip:
//! `parse_chain(&chain_to_string(c)) == c`):
//!
//! - **Structured** objects (serde feature) — a YAML/JSON DSL writes a chain as
//!   a list of objects: `[{crop: {w,h}}, hflip, {overlay: {image: "logo.png"}}]`.
//! - **Textual** ffmpeg-`-vf` style — [`parse_chain`] / [`Display`]:
//!   `crop=1280:720,hflip,overlay=logo.png:24:24`.
//!
//! Geometric ops are pure sample rearrangement (run on raw bytes, any bit
//! depth). Colour + overlay ops work on 8-bit `Yuv420p` (the default SDR output).

use std::fmt;

use anyhow::{Context, Result, bail};
use bytes::BytesMut;

use crate::frame::{PixelFormat, VideoFrame};

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
        #[cfg_attr(feature = "serde", serde(default = "default_denoise_strength"))]
        strength: f32,
    },
}

/// Which spatial denoise algorithm [`VideoFilter::Denoise`] runs. Each suits a
/// different kind of noise; `strength` then blends the result with the source.
/// (Temporal denoisers — hqdn3d / NLM-temporal — need frame history and don't
/// fit this stateless per-frame filter; a future extension.)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum DenoiseMethod {
    /// Edge-preserving **bilateral** filter (5×5): smooths flat / sensor noise
    /// while keeping edges sharp. The general-purpose default.
    #[default]
    Bilateral,
    /// **Gaussian** low-pass blur (separable 5×5): smooths everything, so it
    /// softens fine detail along with the noise.
    Gaussian,
    /// **Median** filter (3×3): best for salt-and-pepper / impulse noise; also
    /// edge-preserving.
    Median,
    /// **Mean** (box) blur over a 3×3 window — the cheapest smoother; blurs noise
    /// and detail equally.
    Mean,
    /// **Non-local means**: averages samples weighted by how similar their
    /// surrounding patch is, so repeating texture denoises without blurring.
    /// Highest classical quality — and by far the slowest (7×7 search, 3×3 patch).
    Nlmeans,
    /// **Anisotropic diffusion** (Perona–Malik): iteratively diffuses the image
    /// but gates the flow by the local gradient, so it smooths flat regions while
    /// stopping at edges. Edge-preserving like bilateral, different character.
    Anisotropic,
}

impl fmt::Display for DenoiseMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            DenoiseMethod::Bilateral => "bilateral",
            DenoiseMethod::Gaussian => "gaussian",
            DenoiseMethod::Median => "median",
            DenoiseMethod::Mean => "mean",
            DenoiseMethod::Nlmeans => "nlmeans",
            DenoiseMethod::Anisotropic => "anisotropic",
        })
    }
}

#[cfg(feature = "serde")]
fn default_denoise_strength() -> f32 {
    0.5
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
            let deg = if name == "transpose" {
                90
            } else {
                *nums()?.first().unwrap_or(&90)
            };
            if !matches!(deg, 90 | 180 | 270) {
                bail!("rotate wants 90|180|270, got {deg}");
            }
            VideoFilter::Rotate(deg)
        }
        "grayscale" | "gray" => VideoFilter::Grayscale,
        "overlay" => {
            // overlay=PATH[:X:Y] — PATH must not contain ':'.
            let image = parts.first().ok_or_else(|| anyhow::anyhow!("overlay needs a PATH"))?.to_string();
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
            // denoise[=METHOD][:STRENGTH] — METHOD is bilateral|gaussian|median
            // (default bilateral); STRENGTH is 0..=1 (default 0.5). The two args
            // are order-free: a token that parses as a number is the strength,
            // anything else is the method (so `denoise=0.7` and `denoise=median`
            // both work).
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

/// Require 8-bit `Yuv420p` for the colour / overlay filters and return the planes.
fn planes_8bit(frame: &VideoFrame, what: &str) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    if frame.format != PixelFormat::Yuv420p {
        bail!("the `{what}` filter needs an 8-bit Yuv420p frame (got {:?}); it applies to SDR output", frame.format);
    }
    let (y, u, v) = planes(frame, 1)?;
    Ok((y.to_vec(), u.to_vec(), v.to_vec()))
}

/// Apply one **stateless** filter. (`Overlay` errors here — use [`FilterChain`].)
pub fn apply(frame: &VideoFrame, filter: &VideoFilter) -> Result<VideoFrame> {
    let bps = bps(frame.format)?;
    let w = frame.width as usize;
    let h = frame.height as usize;

    match filter {
        VideoFilter::Crop { w: cw, h: ch, x, y: cy } => match (x, cy) {
            (Some(x), Some(cy)) => crop(frame, *x, *cy, *cw, *ch),
            _ => {
                let cw = even((*cw).min(frame.width));
                let ch = even((*ch).min(frame.height));
                let cx = even(frame.width.saturating_sub(cw) / 2);
                let cyc = even(frame.height.saturating_sub(ch) / 2);
                crop(frame, cx, cyc, cw, ch)
            }
        },
        VideoFilter::Pad { w: pw, h: ph, x, y: py } => {
            let pw = even((*pw).max(frame.width));
            let ph = even((*ph).max(frame.height));
            let px = x.map(even).unwrap_or_else(|| even(pw.saturating_sub(frame.width) / 2));
            let pyc = py.map(even).unwrap_or_else(|| even(ph.saturating_sub(frame.height) / 2));
            pad(frame, pw, ph, px, pyc)
        }
        VideoFilter::HFlip | VideoFilter::VFlip | VideoFilter::Rotate(_) | VideoFilter::Grayscale => {
            let (y, u, v) = planes(frame, bps)?;
            geometric(frame, filter, y, u, v, w, h, bps)
        }
        VideoFilter::Invert => {
            let (mut y, mut u, mut v) = planes_8bit(frame, "invert")?;
            for b in y.iter_mut().chain(u.iter_mut()).chain(v.iter_mut()) {
                *b = 255 - *b;
            }
            Ok(assemble(frame, frame.width, frame.height, y, u, v))
        }
        VideoFilter::Brightness(delta) => {
            let (mut y, u, v) = planes_8bit(frame, "brightness")?;
            for p in y.iter_mut() {
                *p = (*p as i32 + delta).clamp(0, 255) as u8;
            }
            Ok(assemble(frame, frame.width, frame.height, y, u, v))
        }
        VideoFilter::Contrast(c) => {
            let (mut y, u, v) = planes_8bit(frame, "contrast")?;
            for p in y.iter_mut() {
                *p = (((*p as f32 - 128.0) * c) + 128.0).round().clamp(0.0, 255.0) as u8;
            }
            Ok(assemble(frame, frame.width, frame.height, y, u, v))
        }
        VideoFilter::Saturation(s) => {
            let (y, mut u, mut v) = planes_8bit(frame, "saturation")?;
            for p in u.iter_mut().chain(v.iter_mut()) {
                *p = (((*p as f32 - 128.0) * s) + 128.0).round().clamp(0.0, 255.0) as u8;
            }
            Ok(assemble(frame, frame.width, frame.height, y, u, v))
        }
        VideoFilter::Denoise { method, strength } => {
            let (yp, up, vp) = planes_8bit(frame, "denoise")?;
            let s = strength.clamp(0.0, 1.0);
            let (cw, ch) = (w / 2, h / 2);
            Ok(assemble(
                frame,
                frame.width,
                frame.height,
                denoise_plane(*method, &yp, w, h, s),
                denoise_plane(*method, &up, cw, ch, s),
                denoise_plane(*method, &vp, cw, ch, s),
            ))
        }
        VideoFilter::Overlay { .. } => {
            bail!("overlay is a resource filter — build a FilterChain::prepare(..) and call .apply()")
        }
    }
}

/// The geometric filters (flip / rotate / grayscale), given the planes.
fn geometric(
    frame: &VideoFrame,
    filter: &VideoFilter,
    y: &[u8],
    u: &[u8],
    v: &[u8],
    w: usize,
    h: usize,
    bps: usize,
) -> Result<VideoFrame> {
    Ok(match filter {
        VideoFilter::HFlip => assemble(
            frame, frame.width, frame.height,
            hflip(y, w, h, bps), hflip(u, w / 2, h / 2, bps), hflip(v, w / 2, h / 2, bps),
        ),
        VideoFilter::VFlip => assemble(
            frame, frame.width, frame.height,
            vflip(y, w, h, bps), vflip(u, w / 2, h / 2, bps), vflip(v, w / 2, h / 2, bps),
        ),
        VideoFilter::Rotate(180) => assemble(
            frame, frame.width, frame.height,
            vflip(&hflip(y, w, h, bps), w, h, bps),
            vflip(&hflip(u, w / 2, h / 2, bps), w / 2, h / 2, bps),
            vflip(&hflip(v, w / 2, h / 2, bps), w / 2, h / 2, bps),
        ),
        VideoFilter::Rotate(90) => assemble(
            frame, frame.height, frame.width,
            rot90(y, w, h, bps), rot90(u, w / 2, h / 2, bps), rot90(v, w / 2, h / 2, bps),
        ),
        VideoFilter::Rotate(270) => assemble(
            frame, frame.height, frame.width,
            rot270(y, w, h, bps), rot270(u, w / 2, h / 2, bps), rot270(v, w / 2, h / 2, bps),
        ),
        VideoFilter::Rotate(d) => bail!("rotate must be 90|180|270, got {d}"),
        VideoFilter::Grayscale => {
            let neutral = neutral_chroma(frame.format);
            let mut uu = u.to_vec();
            let mut vv = v.to_vec();
            fill(&mut uu, &neutral);
            fill(&mut vv, &neutral);
            assemble(frame, frame.width, frame.height, y.to_vec(), uu, vv)
        }
        _ => unreachable!("geometric() called with a non-geometric filter"),
    })
}

fn even(n: u32) -> u32 {
    n & !1
}

fn crop(frame: &VideoFrame, x: u32, y: u32, w: u32, h: u32) -> Result<VideoFrame> {
    let (x, y, w, h) = (even(x), even(y), even(w), even(h));
    if w == 0 || h == 0 || x + w > frame.width || y + h > frame.height {
        bail!("crop {w}x{h}+{x}+{y} out of bounds for {}x{}", frame.width, frame.height);
    }
    let bps = bps(frame.format)?;
    let (yp, up, vp) = planes(frame, bps)?;
    let fw = frame.width as usize;
    let y_new = crop_plane(yp, fw, x as usize, y as usize, w as usize, h as usize, bps);
    let u_new = crop_plane(up, fw / 2, (x / 2) as usize, (y / 2) as usize, (w / 2) as usize, (h / 2) as usize, bps);
    let v_new = crop_plane(vp, fw / 2, (x / 2) as usize, (y / 2) as usize, (w / 2) as usize, (h / 2) as usize, bps);
    Ok(assemble(frame, w, h, y_new, u_new, v_new))
}

fn pad(frame: &VideoFrame, pw: u32, ph: u32, x: u32, y: u32) -> Result<VideoFrame> {
    let (pw, ph, x, y) = (even(pw), even(ph), even(x), even(y));
    if x + frame.width > pw || y + frame.height > ph {
        bail!("pad {pw}x{ph} with frame {}x{} at +{x}+{y} overflows", frame.width, frame.height);
    }
    let bps = bps(frame.format)?;
    let (yp, up, vp) = planes(frame, bps)?;
    let (luma_fill, chroma_fill) = black_fill(frame.format);
    let fw = frame.width as usize;
    let fh = frame.height as usize;
    let y_new = pad_plane(yp, fw, fh, pw as usize, ph as usize, x as usize, y as usize, bps, &luma_fill);
    let u_new = pad_plane(up, fw / 2, fh / 2, (pw / 2) as usize, (ph / 2) as usize, (x / 2) as usize, (y / 2) as usize, bps, &chroma_fill);
    let v_new = pad_plane(vp, fw / 2, fh / 2, (pw / 2) as usize, (ph / 2) as usize, (x / 2) as usize, (y / 2) as usize, bps, &chroma_fill);
    Ok(assemble(frame, pw, ph, y_new, u_new, v_new))
}

// ── overlay (image with alpha) ──────────────────────────────────────────────

/// A loaded overlay image, pre-converted to 8-bit YUV 4:2:0 + per-sample alpha,
/// ready to alpha-composite onto frames. Built once by [`FilterChain::prepare`].
#[derive(Debug, Clone)]
struct PreparedOverlay {
    w: usize,
    h: usize,
    x: usize,
    y: usize,
    y_o: Vec<u8>,
    u_o: Vec<u8>,
    v_o: Vec<u8>,
    a_y: Vec<u8>, // luma-resolution alpha
    a_c: Vec<u8>, // chroma-resolution alpha (2×2 averaged)
}

fn clamp8(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

impl PreparedOverlay {
    /// Convert a row-major RGBA8 buffer (`src_w × src_h`) to a prepared overlay
    /// positioned at `(x, y)`. BT.709 limited-range YUV.
    fn from_rgba(rgba: &[u8], src_w: u32, src_h: u32, x: u32, y: u32) -> Result<Self> {
        let w = (src_w & !1) as usize; // even for 4:2:0
        let h = (src_h & !1) as usize;
        if w == 0 || h == 0 {
            bail!("overlay image is too small ({src_w}x{src_h})");
        }
        let stride = src_w as usize * 4;
        let mut y_o = vec![0u8; w * h];
        let mut a_y = vec![0u8; w * h];
        let (cw, ch) = (w / 2, h / 2);
        let mut u_o = vec![0u8; cw * ch];
        let mut v_o = vec![0u8; cw * ch];
        let mut a_c = vec![0u8; cw * ch];
        for r in 0..h {
            for c in 0..w {
                let p = r * stride + c * 4;
                let (rr, gg, bb) = (rgba[p] as i32, rgba[p + 1] as i32, rgba[p + 2] as i32);
                y_o[r * w + c] = clamp8(16 + ((47 * rr + 157 * gg + 16 * bb) >> 8));
                a_y[r * w + c] = rgba[p + 3];
            }
        }
        for r in 0..ch {
            for c in 0..cw {
                let (mut sr, mut sg, mut sb, mut sa) = (0i32, 0i32, 0i32, 0i32);
                for dy in 0..2 {
                    for dx in 0..2 {
                        let p = (r * 2 + dy) * stride + (c * 2 + dx) * 4;
                        sr += rgba[p] as i32;
                        sg += rgba[p + 1] as i32;
                        sb += rgba[p + 2] as i32;
                        sa += rgba[p + 3] as i32;
                    }
                }
                let (rr, gg, bb) = (sr / 4, sg / 4, sb / 4);
                u_o[r * cw + c] = clamp8(128 + ((-26 * rr - 87 * gg + 112 * bb) >> 8));
                v_o[r * cw + c] = clamp8(128 + ((112 * rr - 102 * gg - 10 * bb) >> 8));
                a_c[r * cw + c] = (sa / 4) as u8;
            }
        }
        Ok(Self { w, h, x: (x & !1) as usize, y: (y & !1) as usize, y_o, u_o, v_o, a_y, a_c })
    }

    /// Alpha-composite onto an 8-bit Yuv420p frame: `out = src·(1−α) + ovl·α`.
    fn composite(&self, frame: &VideoFrame) -> Result<VideoFrame> {
        let (mut y, mut u, mut v) = planes_8bit(frame, "overlay")?;
        let (fw, fh) = (frame.width as usize, frame.height as usize);
        for r in 0..self.h {
            let fy = self.y + r;
            if fy >= fh {
                break;
            }
            for c in 0..self.w {
                let fx = self.x + c;
                if fx >= fw {
                    continue;
                }
                let a = self.a_y[r * self.w + c] as u32;
                if a == 0 {
                    continue;
                }
                let i = fy * fw + fx;
                y[i] = ((y[i] as u32 * (255 - a) + self.y_o[r * self.w + c] as u32 * a + 127) / 255) as u8;
            }
        }
        let (cw, ch) = (self.w / 2, self.h / 2);
        let (fcw, fch) = (fw / 2, fh / 2);
        let (ocx, ocy) = (self.x / 2, self.y / 2);
        for r in 0..ch {
            let fy = ocy + r;
            if fy >= fch {
                break;
            }
            for c in 0..cw {
                let fx = ocx + c;
                if fx >= fcw {
                    continue;
                }
                let a = self.a_c[r * cw + c] as u32;
                if a == 0 {
                    continue;
                }
                let i = fy * fcw + fx;
                u[i] = ((u[i] as u32 * (255 - a) + self.u_o[r * cw + c] as u32 * a + 127) / 255) as u8;
                v[i] = ((v[i] as u32 * (255 - a) + self.v_o[r * cw + c] as u32 * a + 127) / 255) as u8;
            }
        }
        Ok(assemble(frame, frame.width, frame.height, y, u, v))
    }
}

// ── prepared chain (loads overlays once, then applies per frame) ─────────────

enum Step {
    Plain(VideoFilter),
    Overlay(PreparedOverlay),
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
                    steps.push(Step::Overlay(PreparedOverlay::from_rgba(img.as_raw(), w, h, *x, *y)?));
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

// ── plane primitives (sample = `bps` bytes; pure rearrangement) ──

fn crop_plane(src: &[u8], pw: usize, x: usize, y: usize, cw: usize, ch: usize, bps: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(cw * ch * bps);
    for row in 0..ch {
        let start = ((y + row) * pw + x) * bps;
        out.extend_from_slice(&src[start..start + cw * bps]);
    }
    out
}

fn pad_plane(src: &[u8], sw: usize, sh: usize, dw: usize, dh: usize, ox: usize, oy: usize, bps: usize, fill_sample: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(dw * dh * bps);
    for _ in 0..dw * dh {
        out.extend_from_slice(fill_sample);
    }
    for row in 0..sh {
        let s = row * sw * bps;
        let d = ((oy + row) * dw + ox) * bps;
        out[d..d + sw * bps].copy_from_slice(&src[s..s + sw * bps]);
    }
    out
}

fn hflip(src: &[u8], w: usize, h: usize, bps: usize) -> Vec<u8> {
    let mut out = vec![0u8; w * h * bps];
    for row in 0..h {
        let base = row * w * bps;
        for col in 0..w {
            let s = base + col * bps;
            let d = base + (w - 1 - col) * bps;
            out[d..d + bps].copy_from_slice(&src[s..s + bps]);
        }
    }
    out
}

fn vflip(src: &[u8], w: usize, h: usize, bps: usize) -> Vec<u8> {
    let rb = w * bps;
    let mut out = vec![0u8; w * h * bps];
    for row in 0..h {
        let s = row * rb;
        let d = (h - 1 - row) * rb;
        out[d..d + rb].copy_from_slice(&src[s..s + rb]);
    }
    out
}

/// Rotate 90° clockwise: src `w×h` → dst `h×w`. dst(r,c) = src(h-1-c, r).
fn rot90(src: &[u8], w: usize, h: usize, bps: usize) -> Vec<u8> {
    let (dw, dh) = (h, w);
    let mut out = vec![0u8; dw * dh * bps];
    for r in 0..dh {
        for c in 0..dw {
            let s = ((h - 1 - c) * w + r) * bps;
            let d = (r * dw + c) * bps;
            out[d..d + bps].copy_from_slice(&src[s..s + bps]);
        }
    }
    out
}

/// Rotate 270° clockwise: src `w×h` → dst `h×w`. dst(r,c) = src(c, w-1-r).
fn rot270(src: &[u8], w: usize, h: usize, bps: usize) -> Vec<u8> {
    let (dw, dh) = (h, w);
    let mut out = vec![0u8; dw * dh * bps];
    for r in 0..dh {
        for c in 0..dw {
            let s = (c * w + (w - 1 - r)) * bps;
            let d = (r * dw + c) * bps;
            out[d..d + bps].copy_from_slice(&src[s..s + bps]);
        }
    }
    out
}

fn fill(buf: &mut [u8], sample: &[u8]) {
    for chunk in buf.chunks_exact_mut(sample.len()) {
        chunk.copy_from_slice(sample);
    }
}

/// Neutral chroma sample bytes (mid-range): 128 for 8-bit, 512 for 10-bit LE.
fn neutral_chroma(format: PixelFormat) -> Vec<u8> {
    match format {
        PixelFormat::Yuv420p => vec![128],
        _ => (512u16).to_le_bytes().to_vec(),
    }
}

/// Limited-range black: luma 16, chroma 128 (8-bit); luma 64, chroma 512 (10-bit).
fn black_fill(format: PixelFormat) -> (Vec<u8>, Vec<u8>) {
    match format {
        PixelFormat::Yuv420p => (vec![16], vec![128]),
        _ => ((64u16).to_le_bytes().to_vec(), (512u16).to_le_bytes().to_vec()),
    }
}

// ── denoise (spatial; one 8-bit plane at a time) ─────────────────────────────

/// Denoise one 8-bit plane with `method`, then blend the filtered plane back
/// with the source by `strength` (`0` ⇒ source, `1` ⇒ fully filtered). `strength
/// == 0` and degenerate sizes short-circuit to a copy.
fn denoise_plane(method: DenoiseMethod, src: &[u8], w: usize, h: usize, strength: f32) -> Vec<u8> {
    if w == 0 || h == 0 || strength <= 0.0 {
        return src.to_vec();
    }
    let filtered = match method {
        DenoiseMethod::Bilateral => bilateral_plane(src, w, h),
        DenoiseMethod::Gaussian => gaussian_plane(src, w, h),
        DenoiseMethod::Median => median_plane(src, w, h),
        DenoiseMethod::Mean => mean_plane(src, w, h),
        DenoiseMethod::Nlmeans => nlmeans_plane(src, w, h),
        DenoiseMethod::Anisotropic => anisotropic_plane(src, w, h),
    };
    if strength >= 1.0 {
        return filtered;
    }
    let inv = 1.0 - strength;
    src.iter()
        .zip(&filtered)
        .map(|(&s, &f)| (s as f32 * inv + f as f32 * strength).round().clamp(0.0, 255.0) as u8)
        .collect()
}

/// Clamp `v` to `0..hi` (edge-replicate border addressing).
fn clamp_idx(v: isize, hi: usize) -> usize {
    v.clamp(0, hi as isize - 1) as usize
}

/// Edge-preserving bilateral filter over a 5×5 window. Each output sample is a
/// weighted average of its neighbourhood where the weight is `spatial(distance)
/// × range(|intensity − centre|)` — so samples across a strong intensity step
/// (an edge) barely contribute and edges stay sharp while flat noise averages
/// out. Border samples shrink the window (out-of-range neighbours are skipped).
fn bilateral_plane(src: &[u8], w: usize, h: usize) -> Vec<u8> {
    const R: isize = 2; // 5×5
    let spatial_sigma = 2.0f32;
    let range_sigma = 20.0f32;
    // Precompute the 5×5 spatial weights and a 256-entry range LUT.
    let mut spatial = [[0f32; 5]; 5];
    for dy in -R..=R {
        for dx in -R..=R {
            let d2 = (dx * dx + dy * dy) as f32;
            spatial[(dy + R) as usize][(dx + R) as usize] =
                (-d2 / (2.0 * spatial_sigma * spatial_sigma)).exp();
        }
    }
    let mut range_lut = [0f32; 256];
    for (d, wt) in range_lut.iter_mut().enumerate() {
        *wt = (-((d * d) as f32) / (2.0 * range_sigma * range_sigma)).exp();
    }
    let mut out = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            let centre = src[y * w + x] as i32;
            let mut sum = 0f32;
            let mut wsum = 0f32;
            for dy in -R..=R {
                let yy = y as isize + dy;
                if yy < 0 || yy >= h as isize {
                    continue;
                }
                for dx in -R..=R {
                    let xx = x as isize + dx;
                    if xx < 0 || xx >= w as isize {
                        continue;
                    }
                    let s = src[yy as usize * w + xx as usize] as i32;
                    let wt = spatial[(dy + R) as usize][(dx + R) as usize]
                        * range_lut[(s - centre).unsigned_abs() as usize];
                    sum += wt * s as f32;
                    wsum += wt;
                }
            }
            out[y * w + x] = (sum / wsum).round().clamp(0.0, 255.0) as u8;
        }
    }
    out
}

/// Separable 5-tap Gaussian blur (σ≈1.0, kernel `[1,4,6,4,1]/16`) — a plain
/// low-pass that smooths noise and detail alike. Border uses edge-replicate.
fn gaussian_plane(src: &[u8], w: usize, h: usize) -> Vec<u8> {
    const K: [f32; 5] = [1.0, 4.0, 6.0, 4.0, 1.0];
    const KSUM: f32 = 16.0;
    const R: isize = 2;
    // Horizontal pass → f32 scratch.
    let mut tmp = vec![0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0f32;
            for (k, &kw) in K.iter().enumerate() {
                let xx = clamp_idx(x as isize + k as isize - R, w);
                acc += kw * src[y * w + xx] as f32;
            }
            tmp[y * w + x] = acc / KSUM;
        }
    }
    // Vertical pass → u8.
    let mut out = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0f32;
            for (k, &kw) in K.iter().enumerate() {
                let yy = clamp_idx(y as isize + k as isize - R, h);
                acc += kw * tmp[yy * w + x];
            }
            out[y * w + x] = (acc / KSUM).round().clamp(0.0, 255.0) as u8;
        }
    }
    out
}

/// 3×3 median filter — replaces each sample with the median of its 3×3
/// neighbourhood, which removes isolated impulse (salt-and-pepper) samples
/// outright while leaving edges intact. Border uses edge-replicate.
fn median_plane(src: &[u8], w: usize, h: usize) -> Vec<u8> {
    let mut out = vec![0u8; w * h];
    let mut window = [0u8; 9];
    for y in 0..h {
        for x in 0..w {
            let mut n = 0;
            for dy in -1isize..=1 {
                for dx in -1isize..=1 {
                    let yy = clamp_idx(y as isize + dy, h);
                    let xx = clamp_idx(x as isize + dx, w);
                    window[n] = src[yy * w + xx];
                    n += 1;
                }
            }
            window.sort_unstable();
            out[y * w + x] = window[4]; // median of 9
        }
    }
    out
}

/// Plain 3×3 **mean** (box) blur, separable. Cheapest smoother; blurs noise and
/// detail alike. Border uses edge-replicate.
fn mean_plane(src: &[u8], w: usize, h: usize) -> Vec<u8> {
    // Horizontal 3-sum into u16 scratch, then vertical 3-sum / 9.
    let mut tmp = vec![0u16; w * h];
    for y in 0..h {
        for x in 0..w {
            let l = clamp_idx(x as isize - 1, w);
            let r = clamp_idx(x as isize + 1, w);
            tmp[y * w + x] = src[y * w + l] as u16 + src[y * w + x] as u16 + src[y * w + r] as u16;
        }
    }
    let mut out = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            let u = clamp_idx(y as isize - 1, h);
            let d = clamp_idx(y as isize + 1, h);
            out[y * w + x] = ((tmp[u * w + x] + tmp[y * w + x] + tmp[d * w + x] + 4) / 9) as u8;
        }
    }
    out
}

/// **Non-local means**: each output sample is an average of the samples in a 7×7
/// search window, weighted by the SSD between the 3×3 patch around the centre and
/// the 3×3 patch around each candidate — so samples whose *surroundings* look
/// like the centre's contribute most. Denoises repeating texture without
/// blurring it, at the cost of being the slowest method here (~`49 × 9` ops per
/// output sample). Border uses edge-replicate.
fn nlmeans_plane(src: &[u8], w: usize, h: usize) -> Vec<u8> {
    const SR: isize = 3; // 7×7 search window
    const PR: isize = 1; // 3×3 patch
    const PN: f32 = ((2 * PR + 1) * (2 * PR + 1)) as f32;
    let h_param = 10.0f32; // filter strength (decay of the patch-distance weight)
    let h2 = h_param * h_param;
    let at = |xx: isize, yy: isize| src[clamp_idx(yy, h) * w + clamp_idx(xx, w)] as i32;
    let patch_ssd = |x1: isize, y1: isize, x2: isize, y2: isize| -> f32 {
        let mut s = 0i32;
        for py in -PR..=PR {
            for px in -PR..=PR {
                let d = at(x1 + px, y1 + py) - at(x2 + px, y2 + py);
                s += d * d;
            }
        }
        s as f32 / PN
    };
    let mut out = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            let (xi, yi) = (x as isize, y as isize);
            let mut sum = 0f32;
            let mut wsum = 0f32;
            for dy in -SR..=SR {
                for dx in -SR..=SR {
                    let dist = patch_ssd(xi, yi, xi + dx, yi + dy);
                    let wt = (-dist / h2).exp();
                    sum += wt * at(xi + dx, yi + dy) as f32;
                    wsum += wt;
                }
            }
            out[y * w + x] = (sum / wsum).round().clamp(0.0, 255.0) as u8;
        }
    }
    out
}

/// **Anisotropic diffusion** (Perona–Malik): iterate `u += λ·Σ g(∇)·∇` over the
/// 4-neighbour gradients, where the conduction `g(∇) = exp(−(∇/κ)²)` falls to
/// ~0 at strong gradients — so the image diffuses (smooths) inside flat regions
/// but the flow stops at edges. 8 iterations, `λ = 0.20` (≤ ¼ for 4-neighbour
/// stability), `κ = 20`. Border uses edge-replicate.
fn anisotropic_plane(src: &[u8], w: usize, h: usize) -> Vec<u8> {
    const ITERS: usize = 8;
    let kappa = 20.0f32;
    let lambda = 0.20f32;
    let g = |grad: f32| {
        let q = grad / kappa;
        (-(q * q)).exp()
    };
    let mut img: Vec<f32> = src.iter().map(|&v| v as f32).collect();
    let mut next = img.clone();
    for _ in 0..ITERS {
        for y in 0..h {
            for x in 0..w {
                let c = img[y * w + x];
                let n = img[clamp_idx(y as isize - 1, h) * w + x] - c;
                let s = img[clamp_idx(y as isize + 1, h) * w + x] - c;
                let e = img[y * w + clamp_idx(x as isize + 1, w)] - c;
                let we = img[y * w + clamp_idx(x as isize - 1, w)] - c;
                next[y * w + x] = c + lambda * (g(n) * n + g(s) * s + g(e) * e + g(we) * we);
            }
        }
        std::mem::swap(&mut img, &mut next);
    }
    img.iter().map(|&v| v.round().clamp(0.0, 255.0) as u8).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::ColorSpace;
    use bytes::Bytes;

    fn frame(w: u32, h: u32) -> VideoFrame {
        let (wu, hu) = (w as usize, h as usize);
        let mut data = Vec::new();
        for r in 0..hu {
            for c in 0..wu {
                data.push((r * wu + c) as u8);
            }
        }
        data.extend(std::iter::repeat(100).take((wu / 2) * (hu / 2)));
        data.extend(std::iter::repeat(200).take((wu / 2) * (hu / 2)));
        VideoFrame::new(Bytes::from(data), w, h, PixelFormat::Yuv420p, ColorSpace::Bt709, 0)
    }
    fn flat(w: u32, h: u32, yv: u8, uv: u8, vv: u8) -> VideoFrame {
        let (wu, hu) = (w as usize, h as usize);
        let mut data = vec![yv; wu * hu];
        data.extend(std::iter::repeat(uv).take((wu / 2) * (hu / 2)));
        data.extend(std::iter::repeat(vv).take((wu / 2) * (hu / 2)));
        VideoFrame::new(Bytes::from(data), w, h, PixelFormat::Yuv420p, ColorSpace::Bt709, 0)
    }
    fn luma(f: &VideoFrame) -> &[u8] {
        &f.data[..(f.width * f.height) as usize]
    }

    #[test]
    fn parse_and_display_round_trip() {
        let c = parse_chain("crop=1280:720,hflip,overlay=logo.png:24:24,brightness=10,saturation=1.5,invert").unwrap();
        assert_eq!(c[0], VideoFilter::Crop { w: 1280, h: 720, x: None, y: None });
        assert_eq!(c[2], VideoFilter::Overlay { image: "logo.png".into(), x: 24, y: 24 });
        assert_eq!(c[3], VideoFilter::Brightness(10));
        assert_eq!(c[4], VideoFilter::Saturation(1.5));
        assert_eq!(c[5], VideoFilter::Invert);
        assert_eq!(chain_to_string(&c), "crop=1280:720,hflip,overlay=logo.png:24:24,brightness=10,saturation=1.5,invert");
        assert_eq!(parse_chain("overlay=a.png").unwrap()[0], VideoFilter::Overlay { image: "a.png".into(), x: 0, y: 0 });
        assert_eq!(parse_chain("negate").unwrap()[0], VideoFilter::Invert);
        assert_eq!(parse_chain("contrast=1.2").unwrap()[0], VideoFilter::Contrast(1.2));
        assert!(parse_chain("brightness=x").is_err());
        assert!(parse_chain("rotate=45").is_err());
    }

    #[cfg(feature = "serde")]
    #[test]
    fn structured_json_round_trips() {
        let json = r#"[{"crop":{"w":1280,"h":720}},"hflip",{"overlay":{"image":"logo.png","x":24,"y":24}},{"brightness":10},"invert"]"#;
        let from_list: FilterSpec = serde_json::from_str(json).unwrap();
        let expect = vec![
            VideoFilter::Crop { w: 1280, h: 720, x: None, y: None },
            VideoFilter::HFlip,
            VideoFilter::Overlay { image: "logo.png".into(), x: 24, y: 24 },
            VideoFilter::Brightness(10),
            VideoFilter::Invert,
        ];
        assert_eq!(from_list.resolve().unwrap(), expect);
        assert_eq!(parse_chain(&chain_to_string(&expect)).unwrap(), expect);
    }

    #[test]
    fn hflip_reverses_rows() {
        let out = apply(&frame(4, 2), &VideoFilter::HFlip).unwrap();
        assert_eq!(&luma(&out)[..4], &[3, 2, 1, 0]);
    }

    #[test]
    fn rotate_dims_and_roundtrip() {
        let f = frame(4, 2);
        let r90 = apply(&f, &VideoFilter::Rotate(90)).unwrap();
        assert_eq!((r90.width, r90.height), (2, 4));
        let back = apply(&r90, &VideoFilter::Rotate(270)).unwrap();
        assert_eq!(luma(&back), luma(&f));
        assert!(apply(&f, &VideoFilter::Rotate(45)).is_err());
    }

    #[test]
    fn color_filters() {
        // brightness: +20 on a flat-100 luma → 120
        let b = apply(&flat(4, 4, 100, 128, 128), &VideoFilter::Brightness(20)).unwrap();
        assert!(luma(&b).iter().all(|&p| p == 120));
        // invert: 100 → 155, chroma 128 → 127
        let inv = apply(&flat(2, 2, 100, 128, 128), &VideoFilter::Invert).unwrap();
        assert_eq!(luma(&inv)[0], 155);
        assert_eq!(inv.data[4], 127);
        // saturation 0 → chroma collapses to 128 (grayscale)
        let s0 = apply(&flat(4, 4, 100, 200, 60), &VideoFilter::Saturation(0.0)).unwrap();
        assert!(s0.data[16..].iter().all(|&p| p == 128));
        // brightness on a 10-bit frame is rejected
        let ten = VideoFrame::new(Bytes::from(vec![0u8; 2 * (4 * 4 + 2 * 4)]), 4, 4, PixelFormat::Yuv420p10le, ColorSpace::Bt709, 0);
        assert!(apply(&ten, &VideoFilter::Brightness(10)).is_err());
    }

    #[test]
    fn overlay_composites_with_alpha() {
        // 2×2 RGBA overlay: top row opaque red, bottom row fully transparent.
        let red = [255u8, 0, 0, 255];
        let clear = [0u8, 0, 0, 0];
        let mut rgba = Vec::new();
        rgba.extend_from_slice(&red);
        rgba.extend_from_slice(&red);
        rgba.extend_from_slice(&clear);
        rgba.extend_from_slice(&clear);
        let ov = PreparedOverlay::from_rgba(&rgba, 2, 2, 0, 0).unwrap();
        // composite onto a 4×4 flat grey frame
        let base = flat(4, 4, 100, 128, 128);
        let out = ov.composite(&base).unwrap();
        let y = luma(&out);
        // opaque red top-left → red's luma (≈ 16 + 0.183*255 ≈ 63), NOT 100
        assert!(y[0] > 50 && y[0] < 90, "opaque red luma was {}", y[0]);
        // transparent bottom row → unchanged grey 100
        assert_eq!(y[2 * 4], 100);
        // out-of-overlay region (col ≥ 2) unchanged
        assert_eq!(y[2], 100);
    }

    #[test]
    fn overlay_via_apply_errors_without_prepare() {
        let r = apply(&flat(4, 4, 100, 128, 128), &VideoFilter::Overlay { image: "x.png".into(), x: 0, y: 0 });
        assert!(r.is_err());
    }

    #[test]
    fn filter_chain_prepare_missing_image_errors() {
        let r = FilterChain::prepare(&[VideoFilter::Overlay { image: "/nope/missing.png".into(), x: 0, y: 0 }]);
        assert!(r.is_err());
    }

    #[test]
    fn filter_chain_applies_stateless() {
        let chain = FilterChain::prepare(&[VideoFilter::HFlip, VideoFilter::Brightness(10)]).unwrap();
        assert!(!chain.is_empty());
        let out = chain.apply(frame(4, 2)).unwrap();
        assert_eq!((out.width, out.height), (4, 2));
    }

    #[test]
    fn ten_bit_geometric_still_works() {
        let mut data: Vec<u8> = Vec::new();
        for s in [0u16, 1, 2, 3] {
            data.extend_from_slice(&s.to_le_bytes());
        }
        data.extend_from_slice(&(512u16).to_le_bytes());
        data.extend_from_slice(&(512u16).to_le_bytes());
        let f = VideoFrame::new(Bytes::from(data), 2, 2, PixelFormat::Yuv420p10le, ColorSpace::Bt709, 0);
        let out = apply(&f, &VideoFilter::HFlip).unwrap();
        assert_eq!(&out.data[0..2], &1u16.to_le_bytes());
    }

    // ── denoise family ──────────────────────────────────────────────────────

    const DENOISE_METHODS: [DenoiseMethod; 6] = [
        DenoiseMethod::Bilateral,
        DenoiseMethod::Gaussian,
        DenoiseMethod::Median,
        DenoiseMethod::Mean,
        DenoiseMethod::Nlmeans,
        DenoiseMethod::Anisotropic,
    ];

    /// Build a `w×h` Yuv420p frame with the given luma + flat neutral chroma.
    fn frame_with_luma(luma: Vec<u8>, w: u32, h: u32) -> VideoFrame {
        let (wu, hu) = (w as usize, h as usize);
        assert_eq!(luma.len(), wu * hu);
        let mut data = luma;
        data.extend(std::iter::repeat(128).take(2 * (wu / 2) * (hu / 2)));
        VideoFrame::new(Bytes::from(data), w, h, PixelFormat::Yuv420p, ColorSpace::Bt709, 0)
    }

    /// Denoise a luma pattern, return the output luma plane.
    fn denoise_luma(plane: Vec<u8>, w: u32, h: u32, method: DenoiseMethod, strength: f32) -> Vec<u8> {
        let f = frame_with_luma(plane, w, h);
        let out = apply(&f, &VideoFilter::Denoise { method, strength }).unwrap();
        luma(&out).to_vec()
    }

    #[test]
    fn denoise_parse_and_display() {
        let bil = |s| VideoFilter::Denoise { method: DenoiseMethod::Bilateral, strength: s };
        assert_eq!(parse_chain("denoise").unwrap()[0], bil(0.5));
        assert_eq!(parse_chain("denoise=0.7").unwrap()[0], bil(0.7));
        assert_eq!(
            parse_chain("denoise=median").unwrap()[0],
            VideoFilter::Denoise { method: DenoiseMethod::Median, strength: 0.5 }
        );
        assert_eq!(
            parse_chain("denoise=nlmeans:0.3").unwrap()[0],
            VideoFilter::Denoise { method: DenoiseMethod::Nlmeans, strength: 0.3 }
        );
        // args are order-free, and `nr` + aliases work
        assert_eq!(
            parse_chain("denoise=0.3:gaussian").unwrap()[0],
            VideoFilter::Denoise { method: DenoiseMethod::Gaussian, strength: 0.3 }
        );
        assert_eq!(
            parse_chain("nr=pm").unwrap()[0],
            VideoFilter::Denoise { method: DenoiseMethod::Anisotropic, strength: 0.5 }
        );
        // round-trip through Display
        assert_eq!(chain_to_string(&parse_chain("denoise=median:0.8").unwrap()), "denoise=median:0.8");
        assert!(parse_chain("denoise=2.0").is_err()); // strength out of range
        assert!(parse_chain("denoise=foo").is_err()); // unknown method
    }

    #[test]
    fn denoise_flat_is_unchanged() {
        for m in DENOISE_METHODS {
            let out = denoise_luma(vec![100u8; 64], 8, 8, m, 1.0);
            assert!(
                out.iter().all(|&p| (p as i32 - 100).abs() <= 1),
                "{m:?} altered a flat plane"
            );
        }
    }

    #[test]
    fn denoise_strength_zero_is_identity() {
        let luma: Vec<u8> = (0..64).map(|i| (i * 3) as u8).collect();
        for m in DENOISE_METHODS {
            assert_eq!(denoise_luma(luma.clone(), 8, 8, m, 0.0), luma, "{m:?} @ strength 0 must be identity");
        }
    }

    #[test]
    fn denoise_smooths_checkerboard() {
        // ±6 checkerboard around 128 — the smoothing methods pull it toward 128.
        let luma: Vec<u8> =
            (0..64).map(|i| if (i / 8 + i % 8) % 2 == 0 { 122 } else { 134 }).collect();
        for m in [
            DenoiseMethod::Bilateral,
            DenoiseMethod::Gaussian,
            DenoiseMethod::Mean,
            DenoiseMethod::Nlmeans,
            DenoiseMethod::Anisotropic,
        ] {
            let out = denoise_luma(luma.clone(), 8, 8, m, 1.0);
            let maxdev = out.iter().map(|&p| (p as i32 - 128).abs()).max().unwrap();
            assert!(maxdev < 6, "{m:?} didn't smooth the checkerboard (maxdev {maxdev})");
        }
    }

    #[test]
    fn denoise_median_removes_impulse() {
        // A single bright spike on a flat field is exactly what median kills.
        let mut luma = vec![100u8; 64];
        luma[3 * 8 + 3] = 250;
        let out = denoise_luma(luma, 8, 8, DenoiseMethod::Median, 1.0);
        assert_eq!(out[3 * 8 + 3], 100, "median should remove the impulse");
    }

    #[test]
    fn denoise_bilateral_preserves_edge() {
        // Left half 50, right half 200 — the edge must survive.
        let luma: Vec<u8> = (0..64).map(|i| if (i % 8) < 4 { 50 } else { 200 }).collect();
        let out = denoise_luma(luma, 8, 8, DenoiseMethod::Bilateral, 1.0);
        for r in 0..8 {
            assert!(out[r * 8 + 1] < 80, "left edge blurred: {}", out[r * 8 + 1]);
            assert!(out[r * 8 + 6] > 170, "right edge blurred: {}", out[r * 8 + 6]);
        }
    }

    #[test]
    fn denoise_rejects_10bit() {
        let ten = VideoFrame::new(
            Bytes::from(vec![0u8; 2 * (4 * 4 + 2 * 4)]),
            4,
            4,
            PixelFormat::Yuv420p10le,
            ColorSpace::Bt709,
            0,
        );
        assert!(apply(&ten, &VideoFilter::Denoise { method: DenoiseMethod::Gaussian, strength: 0.5 }).is_err());
    }
}
