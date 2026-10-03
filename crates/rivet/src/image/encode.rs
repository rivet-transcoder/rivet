//! The four web formats' encoders, each this workspace's own: AVIF (rivet's
//! AV1 encoder in rivet's HEIF writer, [`crate::avif`]), JPEG (rivet-jpeg),
//! PNG (rivet-png), and WebP (rivet-webp, pending: [`super::webp`]).

use anyhow::{Result, anyhow};

use super::ImageFormat;
use super::scale::Pixels;

/// Encode `pixels` as `format`. `quality` is 1–100 for the lossy formats;
/// `lossless` makes WebP lossless; `speed` (1 slowest to 10 fastest) is the
/// effort PNG spends on compression ([`png_level`]).
pub(crate) fn encode(pixels: &Pixels<'_>, format: ImageFormat, quality: u8, lossless: bool, speed: u8) -> Result<Vec<u8>> {
    match format {
        ImageFormat::Avif => avif(pixels, quality),
        ImageFormat::Webp => super::webp::encode(pixels, quality, lossless),
        ImageFormat::Jpeg => jpeg(pixels, quality),
        ImageFormat::Png => png(pixels, png_level(speed)),
    }
}

/// AVIF, with an alpha item when the picture has transparency. The writer
/// tags its output sRGB and writes no ICC, which is why AVIF output is always
/// converted.
fn avif(pixels: &Pixels<'_>, quality: u8) -> Result<Vec<u8>> {
    let (w, h) = pixels.image.dimensions();
    crate::avif::encode_rgba(pixels.image.as_raw(), w, h, pixels.alpha, quality)
}

/// Progressive JPEG with optimised Huffman tables and 4:2:0 chroma, as a web
/// JPEG is. A transparent picture is flattened onto white.
fn jpeg(pixels: &Pixels<'_>, quality: u8) -> Result<Vec<u8>> {
    let (w, h) = pixels.image.dimensions();
    let rgb: Vec<u8> = pixels
        .image
        .pixels()
        .flat_map(|p| {
            let a = u32::from(p[3]);
            let over_white = |c: u8| ((u32::from(c) * a + 255 * (255 - a) + 127) / 255) as u8;
            [over_white(p[0]), over_white(p[1]), over_white(p[2])]
        })
        .collect();
    let settings = jpeg::EncodeSettings {
        quality: quality.clamp(1, 100),
        subsampling: jpeg::Subsampling::S420,
        progressive: true,
        optimize_huffman: true,
        icc_profile: pixels.icc.map(<[u8]>::to_vec),
        ..Default::default()
    };
    jpeg::encode(&rgb, w, h, jpeg::PixelFormat::Rgb, &settings).map_err(|e| anyhow!("JPEG encode failed: {e}"))
}

/// The DEFLATE level a PNG is written at for an encoder effort of `speed`
/// (1 slowest to 10 fastest; 6, the default, is level 6).
///
/// Level 9 is what the `image` crate's "Best" meant, and in rivet-png it
/// also tries a second parse and keeps the smaller: up to 8.5 s on a
/// 2048x2048 picture. On a photograph-like 2048x2048 one
/// ([`png_level_timings`]) levels 1 / 3 / 6 / 9 took 0.25 / 0.34 / 1.12 /
/// 1.59 s for 6.44 / 6.33 / 6.107 / 6.107 MB: past 6 the time buys
/// nothing on noisy content. So 9 is kept for the slowest setting, and the
/// default is 6 — the zlib default, lazy matching over the whole window.
pub(crate) fn png_level(speed: u8) -> u8 {
    match speed {
        0 | 1 => 9,
        2 => 8,
        3 => 7,
        4..=6 => 6,
        7 => 5,
        8 => 4,
        9 => 3,
        _ => 1,
    }
}

/// PNG: RGB, or RGBA when the picture has transparency; adaptive filtering.
fn png(pixels: &Pixels<'_>, level: u8) -> Result<Vec<u8>> {
    let (w, h) = pixels.image.dimensions();
    let image = if pixels.alpha {
        rpng::Image::from_rgba8(w, h, pixels.image.as_raw().to_vec())
    } else {
        rpng::Image::new(w, h, rpng::ColorType::Rgb, 8, pixels.image.to_rgb())
    }
    .map_err(|e| anyhow!("PNG encode failed: {e}"))?;
    let mut encoder = rpng::Encoder::with_level(level);
    if let Some(icc) = pixels.icc {
        encoder.metadata.icc_profile = Some(rpng::IccProfile { name: "ICC profile".into(), profile: icc.to_vec() });
    }
    encoder.encode(&image).map_err(|e| anyhow!("PNG encode failed: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn png_levels_follow_the_speed() {
        assert_eq!(png_level(super::super::DEFAULT_AVIF_SPEED), 6);
        let levels: Vec<u8> = (1..=10).map(png_level).collect();
        assert!(levels.windows(2).all(|w| w[1] <= w[0]), "{levels:?}");
        assert_eq!((levels[0], levels[9]), (9, 1));
    }
}

/// What each PNG level costs on a 2048x2048 photograph-like picture (smooth
/// gradients under sensor-like noise), printed (`--release --ignored
/// --nocapture`): the measurement behind [`png_level`].
#[cfg(test)]
#[test]
#[ignore = "a measurement: run with --release --ignored --nocapture"]
fn png_level_timings() {
    let (w, h) = (2048u32, 2048u32);
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let mut rgb = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            for c in 0..3u32 {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                let base = ((x * (c + 1) / 9 + y / (c + 3)) % 200) as i32 + 20;
                rgb.push((base + (seed % 7) as i32 - 3).clamp(0, 255) as u8);
            }
        }
    }
    let image = rpng::Image::new(w, h, rpng::ColorType::Rgb, 8, rgb).unwrap();
    for level in [1u8, 3, 6, 9] {
        let start = std::time::Instant::now();
        let bytes = rpng::Encoder::with_level(level).encode(&image).unwrap();
        eprintln!("PNG level {level}: {:.2} s, {} bytes", start.elapsed().as_secs_f64(), bytes.len());
    }
}
