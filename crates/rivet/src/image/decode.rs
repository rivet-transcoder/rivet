//! Reading an input into pixels: a still image, or stills from a video.

use std::io::Cursor;

use anyhow::{Context, Result, anyhow, bail};
use bytes::Bytes;
use image::{DynamicImage, ImageDecoder, RgbaImage};

use super::colour::Profile;
use super::{FrameSelection, SourceFormat, heif};
use crate::thumbnail;

/// The most pixels a source may have: 100 megapixels, a little over the
/// largest phone and camera sensors (a 48 MP phone, a 61 MP full-frame body,
/// a 100 MP medium-format one). Checked from the header, before anything is
/// allocated, so an image that declares a billion pixels is refused rather
/// than attempted.
pub const MAX_SOURCE_PIXELS: u64 = 100_000_000;

/// The most memory a raster decoder may allocate: the largest source's RGBA
/// at 16 bits a sample, and some.
const MAX_DECODE_ALLOC: u64 = MAX_SOURCE_PIXELS * 8 + (64 << 20);

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
        let alpha = rgba.pixels().any(|p| p.0[3] != u8::MAX);
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
    if data.starts_with(b"II*\0") || data.starts_with(b"MM\0*") {
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

/// The header of a still image: its size, upright and as stored.
pub(crate) fn read_header(data: &[u8], format: SourceFormat) -> Result<Header> {
    if matches!(format, SourceFormat::Avif | SourceFormat::Heic) {
        let h = heif::read_header(data)?;
        return Ok(Header {
            width: h.width,
            height: h.height,
            stored_width: h.stored_width,
            stored_height: h.stored_height,
            pixel_format: h.pixel_format,
        });
    }
    let mut decoder = raster_decoder(data, format)?;
    let (w, h) = decoder.dimensions();
    let turned = decoder.orientation().is_ok_and(|o| {
        use image::metadata::Orientation::*;
        matches!(o, Rotate90 | Rotate270 | Rotate90FlipH | Rotate270FlipH)
    });
    Ok(Header {
        width: if turned { h } else { w },
        height: if turned { w } else { h },
        stored_width: w,
        stored_height: h,
        pixel_format: format!("{:?}", decoder.color_type()),
    })
}

/// Decode a still image, upright.
pub(crate) fn decode(data: &[u8], format: SourceFormat) -> Result<Picture> {
    if matches!(format, SourceFormat::Avif | SourceFormat::Heic) {
        return heif::decode(data, format);
    }
    let mut decoder = raster_decoder(data, format)?;
    let (w, h) = decoder.dimensions();
    check_size(w, h)?;
    let mut limits = image::Limits::default();
    limits.max_alloc = Some(MAX_DECODE_ALLOC);
    decoder.set_limits(limits)?;
    // A profile or an orientation that cannot be read is treated as absent,
    // as a browser would, rather than failing a picture that displays fine.
    let icc = decoder.icc_profile().ok().flatten().filter(|p| !p.is_empty());
    let orientation = decoder.orientation().unwrap_or(image::metadata::Orientation::NoTransforms);
    let mut img = DynamicImage::from_decoder(decoder)?;
    img.apply_orientation(orientation);
    Ok(Picture::new(img.to_rgba8(), icc.map(Profile::Icc)))
}

fn raster_decoder(data: &[u8], format: SourceFormat) -> Result<impl ImageDecoder + '_> {
    let format = match format {
        SourceFormat::Jpeg => image::ImageFormat::Jpeg,
        SourceFormat::Png => image::ImageFormat::Png,
        SourceFormat::Webp => image::ImageFormat::WebP,
        SourceFormat::Gif => image::ImageFormat::Gif,
        SourceFormat::Tiff => image::ImageFormat::Tiff,
        SourceFormat::Bmp => image::ImageFormat::Bmp,
        SourceFormat::Avif | SourceFormat::Heic => unreachable!("read by heif"),
    };
    // Headers are read without limits, so a picture over the pixel limit is
    // refused in its own words; the decode then runs under them.
    let mut reader = image::ImageReader::with_format(Cursor::new(data), format);
    reader.no_limits();
    Ok(reader.into_decoder()?)
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
        let rgba = RgbaImage::from_fn(w, h, |x, y| {
            let at = ((y * w + x) * 3) as usize;
            image::Rgba([rgb[at], rgb[at + 1], rgb[at + 2], u8::MAX])
        });
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
