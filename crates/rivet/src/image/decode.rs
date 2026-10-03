//! Reading an input into pixels: a still image, or stills from a video.
//!
//! Every still-image decoder is this workspace's own, written clean-room from
//! its format's specification: rivet-jpeg (`jpeg`), rivet-png (`rpng`),
//! rivet-gif, rivet-bmp and rivet-tiff (`gif`, `bmp`, `tiff`), and for AVIF and
//! HEIC rivet's HEIF reader ([`heif`]) in front of the AV1 and HEVC decoders,
//! and rivet-webp for WebP ([`webp`](super::webp)).

use anyhow::{Context, Result, anyhow, bail};
use bytes::Bytes;

use super::colour::Profile;
use super::raster::RgbaImage;
use super::{FrameSelection, SourceFormat, heif};
use crate::thumbnail;

/// The most pixels a source may have: 100 megapixels, a little over the
/// largest phone and camera sensors (a 48 MP phone, a 61 MP full-frame body,
/// a 100 MP medium-format one). Checked from the header, before anything is
/// allocated, so an image that declares a billion pixels is refused rather
/// than attempted.
pub const MAX_SOURCE_PIXELS: u64 = 100_000_000;

/// A decoded picture: upright, 8-bit RGBA, with what its colours mean.
#[derive(Debug, Clone)]
pub(crate) struct Picture {
    pub(crate) rgba: RgbaImage,
    /// Whether any pixel is less than opaque.
    pub(crate) alpha: bool,
    /// `None` is sRGB (or untagged, which a browser reads as sRGB).
    pub(crate) profile: Option<Profile>,
    /// The shape of one pixel: `(1, 1)` for every still image; a frame of an
    /// anamorphic video carries its own.
    pub(crate) sample_aspect: (u32, u32),
}

impl Picture {
    pub(crate) fn new(rgba: RgbaImage, profile: Option<Profile>) -> Self {
        let alpha = rgba.has_alpha();
        Self { rgba, alpha, profile, sample_aspect: (1, 1) }
    }
}

/// What [`super::probe`] reports of a still image.
pub(crate) struct Header {
    /// Upright.
    pub(crate) width: u32,
    pub(crate) height: u32,
    /// As stored, before the orientation is applied.
    pub(crate) stored_width: u32,
    pub(crate) stored_height: u32,
    pub(crate) pixel_format: String,
}

/// Sniff a still image from its first bytes.
pub(crate) fn sniff(data: &[u8]) -> Option<SourceFormat> {
    if data.starts_with(&[0xFF, 0xD8, 0xFF]) {
        return Some(SourceFormat::Jpeg);
    }
    if data.starts_with(b"\x89PNG\r\n\x1a\n") {
        return Some(SourceFormat::Png);
    }
    if data.starts_with(b"GIF87a") || data.starts_with(b"GIF89a") {
        return Some(SourceFormat::Gif);
    }
    if data.len() >= 12 && &data[..4] == b"RIFF" && &data[8..12] == b"WEBP" {
        return Some(SourceFormat::Webp);
    }
    // Classic TIFF and BigTIFF, either byte order.
    if data.starts_with(b"II*\0") || data.starts_with(b"MM\0*") || data.starts_with(b"II+\0") || data.starts_with(b"MM\0+") {
        return Some(SourceFormat::Tiff);
    }
    // `BM` is two bytes of pattern, so the DIB header's own size has to be
    // one of the sizes the format defines as well.
    if data.len() >= 18 && data.starts_with(b"BM") {
        let dib = u32::from_le_bytes([data[14], data[15], data[16], data[17]]);
        if matches!(dib, 12 | 16 | 40 | 52 | 56 | 64 | 108 | 124) {
            return Some(SourceFormat::Bmp);
        }
    }
    heif::sniff(data)
}

/// The EXIF orientation (tag 0x0112, 1-8) in a TIFF-structured EXIF block,
/// with or without its `Exif\0\0` prefix. `None` when there is none.
pub(crate) fn exif_orientation(exif: &[u8]) -> Option<u16> {
    let tiff = exif.strip_prefix(b"Exif\0\0").unwrap_or(exif);
    let le = match tiff.get(..2)? {
        b"II" => true,
        b"MM" => false,
        _ => return None,
    };
    let u16_at = |at: usize| -> Option<u16> {
        let b: [u8; 2] = tiff.get(at..at + 2)?.try_into().ok()?;
        Some(if le { u16::from_le_bytes(b) } else { u16::from_be_bytes(b) })
    };
    let u32_at = |at: usize| -> Option<u32> {
        let b: [u8; 4] = tiff.get(at..at + 4)?.try_into().ok()?;
        Some(if le { u32::from_le_bytes(b) } else { u32::from_be_bytes(b) })
    };
    let ifd = u32_at(4)? as usize;
    let count = usize::from(u16_at(ifd)?);
    (0..count.min(512))
        .find_map(|i| {
            let at = ifd + 2 + i * 12;
            // SHORT (3), one value, in the first two bytes of the value field.
            if u16_at(at)? == 0x0112 && u16_at(at + 2)? == 3 { u16_at(at + 8) } else { None }
        })
        .filter(|o| (1..=8).contains(o))
}

/// Whether an orientation turns the picture on its side.
fn turned(orientation: u16) -> bool {
    (5..=8).contains(&orientation)
}

fn header(w: u32, h: u32, orientation: u16, pixel_format: impl Into<String>) -> Header {
    let t = turned(orientation);
    Header {
        width: if t { h } else { w },
        height: if t { w } else { h },
        stored_width: w,
        stored_height: h,
        pixel_format: pixel_format.into(),
    }
}

fn png_format(c: rpng::ColorType, depth: u8) -> String {
    let name = match c {
        rpng::ColorType::Grayscale => "L",
        rpng::ColorType::GrayscaleAlpha => "La",
        rpng::ColorType::Rgb => "Rgb",
        rpng::ColorType::Rgba => "Rgba",
        rpng::ColorType::Indexed => return format!("Indexed{depth}"),
    };
    format!("{name}{}", if depth > 8 { 16 } else { 8 })
}

/// The header of a still image: its size, upright and as stored.
pub(crate) fn read_header(data: &[u8], format: SourceFormat) -> Result<Header> {
    Ok(match format {
        SourceFormat::Avif | SourceFormat::Heic => {
            let h = heif::read_header(data)?;
            Header {
                width: h.width,
                height: h.height,
                stored_width: h.stored_width,
                stored_height: h.stored_height,
                pixel_format: h.pixel_format,
            }
        }
        SourceFormat::Jpeg => {
            let info = jpeg::read_info(data).map_err(|e| anyhow!("{e}"))?;
            let orientation = info.orientation.map_or(1, |o| o.to_exif());
            let format = match (info.colour_space, info.precision > 8) {
                (jpeg::ColourSpace::Grey, false) => "L8",
                (jpeg::ColourSpace::Grey, true) => "L16",
                (jpeg::ColourSpace::Cmyk | jpeg::ColourSpace::Ycck, _) => "Cmyk8",
                (_, false) => "Rgb8",
                (_, true) => "Rgb16",
            };
            header(info.width, info.height, orientation, format)
        }
        SourceFormat::Png => {
            let h = rpng::read_header(data).map_err(|e| anyhow!("{e}"))?;
            // The orientation is in an `eXIf` chunk, which is read with the
            // chunks; the pixels are not decoded for it.
            let orientation = png_exif(data).as_deref().and_then(exif_orientation).unwrap_or(1);
            header(h.width, h.height, orientation, png_format(h.color_type, h.bit_depth))
        }
        SourceFormat::Gif => {
            let info = gif::read_info(data).map_err(|e| anyhow!("{e}"))?;
            header(u32::from(info.width), u32::from(info.height), 1, "Rgba8")
        }
        SourceFormat::Tiff => {
            let page = tiff::read_info(data).map_err(|e| anyhow!("{e}"))?;
            let bits = page.bits_per_sample.first().copied().unwrap_or(8);
            let channels = match page.samples_per_pixel {
                1 => "L",
                2 => "La",
                3 => "Rgb",
                _ => "Rgba",
            };
            header(page.width, page.height, page.orientation, format!("{channels}{}", if bits > 8 { 16 } else { 8 }))
        }
        SourceFormat::Bmp => {
            let info = bmp::read_info(data).map_err(|e| anyhow!("{e}"))?;
            header(info.width, info.height, 1, if info.bits_per_pixel == 32 { "Rgba8" } else { "Rgb8" })
        }
        SourceFormat::Webp => {
            let (w, h) = super::webp::read_header(data)?;
            header(w, h, 1, "Rgba8")
        }
    })
}

/// A PNG's `eXIf` chunk, found by walking the chunk list (no inflating).
fn png_exif(data: &[u8]) -> Option<Vec<u8>> {
    let mut at = 8;
    while at + 8 <= data.len() {
        let len = u32::from_be_bytes(data[at..at + 4].try_into().ok()?) as usize;
        let kind = &data[at + 4..at + 8];
        let body = data.get(at + 8..at + 8 + len)?;
        match kind {
            b"eXIf" => return Some(body.to_vec()),
            b"IDAT" | b"IEND" => return None,
            _ => {}
        }
        at += 12 + len;
    }
    None
}

/// Decode a still image, upright.
pub(crate) fn decode(data: &[u8], format: SourceFormat) -> Result<Picture> {
    // The header first, so a picture over the limit is refused in its own
    // words before a decoder allocates for it.
    let h = read_header(data, format)?;
    check_size(h.stored_width, h.stored_height)?;
    let limit = MAX_SOURCE_PIXELS;
    // A profile or an orientation that cannot be read is treated as absent,
    // as a browser would, rather than failing a picture that displays fine.
    let (rgba, icc, orientation) = match format {
        SourceFormat::Avif | SourceFormat::Heic => return heif::decode(data, format),
        SourceFormat::Webp => return super::webp::decode(data),
        SourceFormat::Jpeg => {
            let options = jpeg::DecodeOptions { max_pixels: Some(limit), ..Default::default() };
            let img = jpeg::decode_with(data, &options).map_err(|e| anyhow!("{e}"))?;
            for w in &img.warnings {
                tracing::debug!(warning = %w, "JPEG decode");
            }
            let orientation = img.info.orientation.map_or(1, |o| o.to_exif());
            let rgba = RgbaImage::from_raw(img.info.width, img.info.height, img.to_rgba8())
                .context("the JPEG decoder returned the wrong number of pixels")?;
            (rgba, img.info.icc_profile.clone(), orientation)
        }
        SourceFormat::Png => {
            let png = rpng::Decoder::new().max_pixels(limit).animation(false).decode(data).map_err(|e| anyhow!("{e}"))?;
            let orientation = png.metadata.exif.as_deref().and_then(exif_orientation).unwrap_or(1);
            let icc = png.metadata.icc_profile.as_ref().map(|p| p.profile.clone());
            let rgba = RgbaImage::from_raw(png.image.width, png.image.height, png.image.to_rgba8())
                .context("the PNG decoder returned the wrong number of pixels")?;
            (rgba, icc, orientation)
        }
        SourceFormat::Gif => {
            // The first frame, composited onto the logical screen: what a
            // viewer shows before the animation moves.
            let limits = gif::Limits { max_pixels: limit, max_frames: 1, ..Default::default() };
            let mut decoder = gif::Decoder::with_limits(data, limits).map_err(|e| anyhow!("{e}"))?;
            let (w, h) = (u32::from(decoder.info().width), u32::from(decoder.info().height));
            let frame = decoder.next_frame().map_err(|e| anyhow!("{e}"))?.context("the GIF has no image")?;
            let rgba = RgbaImage::from_raw(w, h, frame.rgba).context("the GIF decoder returned the wrong number of pixels")?;
            (rgba, None, 1)
        }
        SourceFormat::Tiff => {
            let limits = tiff::Limits { max_pixels: limit, ..Default::default() };
            let page = tiff::decode_with_limits(data, limits).map_err(|e| anyhow!("{e}"))?;
            let rgba = RgbaImage::from_raw(page.width, page.height, page.to_rgba8())
                .context("the TIFF decoder returned the wrong number of pixels")?;
            (rgba, page.info.icc_profile.clone(), page.info.orientation)
        }
        SourceFormat::Bmp => {
            let img = bmp::decode_with_limits(data, bmp::Limits { max_pixels: limit }).map_err(|e| anyhow!("{e}"))?;
            let rgba = RgbaImage::from_raw(img.width, img.height, img.rgba)
                .context("the BMP decoder returned the wrong number of pixels")?;
            (rgba, img.icc_profile, 1)
        }
    };
    let icc = icc.filter(|p| !p.is_empty());
    Ok(Picture::new(rgba.oriented(orientation), icc.map(Profile::Icc)))
}

/// Refuse a picture larger than [`MAX_SOURCE_PIXELS`].
pub(crate) fn check_size(width: u32, height: u32) -> Result<()> {
    if width == 0 || height == 0 {
        bail!("the image has no pixels ({width}x{height})");
    }
    if u64::from(width) * u64::from(height) > MAX_SOURCE_PIXELS {
        bail!(
            "unsupported input: the image is {width}x{height}, over the {} megapixel limit",
            MAX_SOURCE_PIXELS / 1_000_000
        );
    }
    Ok(())
}

/// The stills `selection` picks from a video, each with its position in the
/// selection and its time in seconds, and the video's codec.
pub(crate) fn video_stills(input: &Bytes, selection: &FrameSelection) -> Result<(String, Vec<(usize, f64, Picture)>)> {
    // Filled in by `pick`, which is handed the stream before any frame is
    // decoded: which requested still each frame index serves.
    let mut wanted: Vec<u64> = Vec::new();
    let mut rate = 0.0;
    let (source, frames) = thumbnail::capture_frames(input, |source| {
        rate = frame_rate(source);
        wanted = frame_indices(source, rate, selection)?;
        Ok(wanted.clone())
    })?;
    if frames.is_empty() {
        bail!("the video gave no frames to take stills from");
    }

    let mut stills = Vec::with_capacity(wanted.len());
    for (position, &index) in wanted.iter().enumerate() {
        // The frame taken for this index: itself, or the last one when the
        // stream ended short of it.
        let (taken, captured) = frames
            .iter()
            .rev()
            .find(|(i, _)| *i <= index)
            .or_else(|| frames.first())
            .ok_or_else(|| anyhow!("no frame for still {position}"))?;
        let (rgb, w, h) = thumbnail::frame_to_rgb8(&captured.frame, captured.color).context("converting the frame to RGB")?;
        let rgba = RgbaImage::from_rgb(w, h, &rgb).context("the frame converted to the wrong number of pixels")?;
        let mut picture = Picture::new(rgba, None);
        picture.sample_aspect = source.sample_aspect;
        let seconds = if rate > 0.0 { *taken as f64 / rate } else { 0.0 };
        stills.push((position, seconds, picture));
    }
    Ok((source.codec, stills))
}

/// Frames per second, from the stream's rate or, lacking one, its frame count
/// over its duration.
fn frame_rate(source: &thumbnail::StillSource) -> f64 {
    if source.frame_rate.is_finite() && source.frame_rate > 0.0 {
        source.frame_rate
    } else if source.duration > 0.0 {
        source.total_frames as f64 / source.duration
    } else {
        0.0
    }
}

/// The frame index of each requested still, in request order.
pub(crate) fn frame_indices(source: &thumbnail::StillSource, rate: f64, selection: &FrameSelection) -> Result<Vec<u64>> {
    let total = source.total_frames.max(1);
    let last = total - 1;
    let at_fraction = |f: f64| (((total as f64) * f) as u64).min(last);
    Ok(match selection {
        FrameSelection::Poster => vec![at_fraction(thumbnail::DEFAULT_THUMBNAIL_FRACTION)],
        FrameSelection::Count(n) => (0..*n).map(|i| at_fraction((f64::from(i) + 0.5) / f64::from(*n))).collect(),
        FrameSelection::At(times) => {
            let duration = if source.duration > 0.0 {
                source.duration
            } else if rate > 0.0 {
                total as f64 / rate
            } else {
                0.0
            };
            let mut indices = Vec::with_capacity(times.len());
            for &t in times {
                if duration > 0.0 && t > duration {
                    bail!("invalid output spec: a frame at {t}s is past the end of the video ({duration:.3}s)");
                }
                let index = if rate > 0.0 { (t * rate) as u64 } else { 0 };
                indices.push(index.min(last));
            }
            indices
        }
    })
}
