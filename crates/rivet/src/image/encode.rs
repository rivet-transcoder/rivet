//! The four web formats' encoders.

use anyhow::{Context, Result, anyhow};
use image::ImageEncoder;

use super::ImageFormat;
use super::scale::Pixels;

/// Encode `pixels` as `format`. `quality` is 1–100 for the lossy formats;
/// `lossless` makes WebP lossless; `speed` is AVIF's effort.
pub(crate) fn encode(pixels: &Pixels<'_>, format: ImageFormat, quality: u8, lossless: bool, speed: u8) -> Result<Vec<u8>> {
    match format {
        ImageFormat::Avif => avif(pixels, quality, speed),
        ImageFormat::Webp => webp(pixels, quality, lossless),
        ImageFormat::Jpeg => jpeg(pixels, quality),
        ImageFormat::Png => png(pixels),
    }
}

/// AVIF through ravif (rav1e): the AV1 encoder the thumbnail path uses, with
/// an alpha plane when the picture has transparency. ravif tags its output
/// sRGB and writes no ICC, which is why AVIF output is always converted.
fn avif(pixels: &Pixels<'_>, quality: u8, speed: u8) -> Result<Vec<u8>> {
    let (w, h) = (pixels.image.width() as usize, pixels.image.height() as usize);
    let encoder = ravif::Encoder::new().with_quality(f32::from(quality)).with_alpha_quality(f32::from(quality)).with_speed(speed);
    if !pixels.alpha {
        let rgb: Vec<rgb::RGB8> = pixels.image.pixels().map(|p| rgb::RGB8::new(p.0[0], p.0[1], p.0[2])).collect();
        return encoder
            .encode_rgb(ravif::Img::new(rgb.as_slice(), w, h))
            .map(|e| e.avif_file)
            .map_err(|e| anyhow!("rav1e (AVIF) encode failed: {e}"));
    }
    let rgba: Vec<rgb::RGBA8> = pixels.image.pixels().map(|p| rgb::RGBA8::new(p.0[0], p.0[1], p.0[2], p.0[3])).collect();
    encoder
        .encode_rgba(ravif::Img::new(rgba.as_slice(), w, h))
        .map(|e| e.avif_file)
        .map_err(|e| anyhow!("rav1e (AVIF) encode failed: {e}"))
}

/// WebP through libwebp: lossy (VP8, with a separately coded alpha plane) or
/// lossless (VP8L). An ICC profile goes in an extended (`VP8X`) container,
/// which libwebp's one-shot encoder does not write.
fn webp(pixels: &Pixels<'_>, quality: u8, lossless: bool) -> Result<Vec<u8>> {
    let (w, h) = pixels.image.dimensions();
    let rgb: Vec<u8>;
    let encoder = if pixels.alpha {
        webp::Encoder::from_rgba(pixels.image.as_raw(), w, h)
    } else {
        rgb = pixels.image.pixels().flat_map(|p| [p.0[0], p.0[1], p.0[2]]).collect();
        webp::Encoder::from_rgb(&rgb, w, h)
    };
    let mut config = webp::WebPConfig::new().map_err(|()| anyhow!("libwebp would not give a configuration"))?;
    config.lossless = i32::from(lossless);
    config.quality = if lossless { 75.0 } else { f32::from(quality) };
    // 0 (fastest) to 6 (smallest); 4 is libwebp's own default.
    config.method = 4;
    // Keep the colour under fully transparent pixels in a lossless file, as
    // PNG does; lossy output may clear it, which is smaller.
    config.exact = i32::from(lossless);
    let encoded = encoder.encode_advanced(&config).map_err(|e| anyhow!("libwebp (WebP) encode failed: {e:?}"))?;
    let bytes = encoded.to_vec();
    match pixels.icc {
        Some(icc) => webp_with_icc(&bytes, w, h, pixels.alpha, icc),
        None => Ok(bytes),
    }
}

/// Rewrap a simple-format WebP (`RIFF WEBP` + one `VP8 ` / `VP8L` chunk, and
/// `ALPH` for lossy with alpha) as the extended format with an `ICCP` chunk.
fn webp_with_icc(simple: &[u8], width: u32, height: u32, alpha: bool, icc: &[u8]) -> Result<Vec<u8>> {
    let body = simple.get(12..).context("a WebP shorter than its header")?;
    // Chunks as libwebp wrote them. An encoder that already wrote `VP8X`
    // (lossy with alpha) has its flags rewritten rather than duplicated.
    let mut chunks = Vec::new();
    let mut at = 0;
    while at + 8 <= body.len() {
        let size = u32::from_le_bytes(body[at + 4..at + 8].try_into().unwrap()) as usize;
        let end = at + 8 + size + (size & 1);
        let chunk = body.get(at..end.min(body.len())).context("a WebP chunk runs past the file")?;
        if &chunk[..4] != b"VP8X" {
            chunks.push(chunk);
        }
        at = end;
    }
    let chunk = |fourcc: &[u8; 4], data: &[u8]| {
        let mut c = Vec::with_capacity(8 + data.len() + 1);
        c.extend_from_slice(fourcc);
        c.extend_from_slice(&(data.len() as u32).to_le_bytes());
        c.extend_from_slice(data);
        if data.len() % 2 == 1 {
            c.push(0);
        }
        c
    };
    // VP8X: flags (ICC 0x20, alpha 0x10), three reserved bytes, then the
    // canvas size minus one in 24 bits each.
    let mut vp8x = vec![0x20 | if alpha { 0x10 } else { 0 }, 0, 0, 0];
    vp8x.extend_from_slice(&(width - 1).to_le_bytes()[..3]);
    vp8x.extend_from_slice(&(height - 1).to_le_bytes()[..3]);

    let mut payload = b"WEBP".to_vec();
    payload.extend(chunk(b"VP8X", &vp8x));
    payload.extend(chunk(b"ICCP", icc));
    for c in chunks {
        payload.extend_from_slice(c);
    }
    let mut out = b"RIFF".to_vec();
    out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    out.extend(payload);
    Ok(out)
}

/// Progressive JPEG with optimised Huffman tables and 4:2:0 chroma, as a web
/// JPEG is. A transparent picture is flattened onto white.
fn jpeg(pixels: &Pixels<'_>, quality: u8) -> Result<Vec<u8>> {
    let (w, h) = pixels.image.dimensions();
    let rgb: Vec<u8> = pixels
        .image
        .pixels()
        .flat_map(|p| {
            let a = u32::from(p.0[3]);
            let over_white = |c: u8| ((u32::from(c) * a + 255 * (255 - a) + 127) / 255) as u8;
            [over_white(p.0[0]), over_white(p.0[1]), over_white(p.0[2])]
        })
        .collect();
    let mut out = Vec::new();
    let mut encoder = jpeg_encoder::Encoder::new(&mut out, quality);
    encoder.set_sampling_factor(jpeg_encoder::SamplingFactor::R_4_2_0);
    encoder.set_progressive(true);
    encoder.set_optimized_huffman_tables(true);
    if let Some(icc) = pixels.icc {
        encoder.add_icc_profile(icc).map_err(|e| anyhow!("adding the ICC profile to the JPEG: {e}"))?;
    }
    encoder
        .encode(&rgb, u16::try_from(w)?, u16::try_from(h)?, jpeg_encoder::ColorType::Rgb)
        .map_err(|e| anyhow!("JPEG encode failed: {e}"))?;
    Ok(out)
}

/// PNG: RGB, or RGBA when the picture has transparency.
fn png(pixels: &Pixels<'_>) -> Result<Vec<u8>> {
    let (w, h) = pixels.image.dimensions();
    let mut out = Vec::new();
    let mut encoder = image::codecs::png::PngEncoder::new_with_quality(
        &mut out,
        image::codecs::png::CompressionType::Best,
        image::codecs::png::FilterType::Adaptive,
    );
    if let Some(icc) = pixels.icc {
        encoder.set_icc_profile(icc.to_vec()).map_err(|e| anyhow!("adding the ICC profile to the PNG: {e}"))?;
    }
    if pixels.alpha {
        encoder.write_image(pixels.image.as_raw(), w, h, image::ExtendedColorType::Rgba8)?;
    } else {
        let rgb: Vec<u8> = pixels.image.pixels().flat_map(|p| [p.0[0], p.0[1], p.0[2]]).collect();
        encoder.write_image(&rgb, w, h, image::ExtendedColorType::Rgb8)?;
    }
    Ok(out)
}
