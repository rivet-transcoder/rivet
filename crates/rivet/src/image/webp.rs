//! WebP, in and out, through this workspace's own codec (`crates/webp`, the
//! rivet-webp repository), written clean-room from RFC 9649: lossy stills
//! through rivet-vp8, lossless VP8L, alpha (`ALPH`), animation, ICC / EXIF /
//! XMP chunks.
//!
//! In: a still, or an animation's first frame as composited on its canvas,
//! with its ICC profile. Out: lossy at the job's quality, or lossless
//! (`image-lossless`), with the ICC profile when it is kept; the effort is
//! the job's `image-speed` mapped onto the codec's 0-6 ([`effort`]).

use anyhow::{Context, Result, anyhow};

use super::colour::Profile;
use super::decode::Picture;
use super::raster::RgbaImage;
use super::scale::Pixels;

/// Whether this build reads and writes WebP.
pub const AVAILABLE: bool = true;

/// Why WebP would be refused, were it unavailable.
pub const UNAVAILABLE: &str = "WebP is not available in this build";

/// The picture's size.
pub(crate) fn read_header(data: &[u8]) -> Result<(u32, u32)> {
    let info = webp::probe(data).map_err(|e| anyhow!("{e}"))?;
    Ok((info.width, info.height))
}

/// Decode a WebP still (an animation's first frame), with its ICC profile.
pub(crate) fn decode(data: &[u8]) -> Result<Picture> {
    let info = webp::probe(data).map_err(|e| anyhow!("{e}"))?;
    let image = webp::decode(data).map_err(|e| anyhow!("{e}"))?;
    let rgba = RgbaImage::from_raw(image.width, image.height, image.rgba)
        .context("the WebP decoder returned the wrong number of pixels")?;
    let icc = info.icc_profile.filter(|p| !p.is_empty());
    Ok(Picture::new(rgba, icc.map(Profile::Icc)))
}

/// The codec's effort (0 fastest - 6 smallest) for an `image-speed` of
/// 1 (slowest) - 10 (fastest): 6 (default) is the codec's own default, 4.
pub(crate) fn effort(speed: u8) -> u8 {
    match speed {
        0..=2 => 6,
        3..=4 => 5,
        5..=6 => 4,
        7..=8 => 2,
        _ => 0,
    }
}

/// Encode lossy at `quality` (1-100) or lossless, with the ICC profile.
pub(crate) fn encode(pixels: &Pixels<'_>, quality: u8, lossless: bool, speed: u8) -> Result<Vec<u8>> {
    let (w, h) = pixels.image.dimensions();
    let image = webp::Image::new(w, h, pixels.image.as_raw().to_vec()).map_err(|e| anyhow!("{e}"))?;
    let config = webp::EncoderConfig {
        lossless,
        quality: quality.clamp(1, 100),
        effort: effort(speed),
        // Keep the colour under fully transparent pixels in a lossless
        // file, as PNG does; lossy output may clear it, which is smaller.
        exact: lossless,
        icc_profile: pixels.icc.map(<[u8]>::to_vec),
        ..Default::default()
    };
    webp::encode(&image, &config).map_err(|e| anyhow!("WebP encode failed: {e}"))
}
