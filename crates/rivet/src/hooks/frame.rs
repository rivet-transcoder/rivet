//! Views of a [`VideoFrame`] a hook can use whatever the decoder produced:
//! 8-bit luma, 8-bit RGB, and the two plainest image files there are (PGM and
//! PPM, which every image tool reads).
//!
//! The pipeline's frames are planar YUV (4:2:0 / 4:2:2 / 4:4:4, 8 to 12 bits,
//! LE 16-bit samples above 8), semi-planar NV12 / NV21, or packed RGB(A) — a
//! still image is RGBA. Planes are tightly packed, luma first.

use anyhow::{Result, bail};

use codec::frame::{ColorSpace, PixelFormat, VideoFrame};

/// An image encoding of a frame, for an integration that wants a file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FrameFormat {
    /// Binary PPM (`P6`): 8-bit RGB.
    #[default]
    Ppm,
    /// Binary PGM (`P5`): 8-bit luma.
    Pgm,
    /// The frame's own planes, as decoded (see the event's `pixel_format`).
    Raw,
}

impl FrameFormat {
    pub fn as_str(self) -> &'static str {
        match self {
            FrameFormat::Ppm => "ppm",
            FrameFormat::Pgm => "pgm",
            FrameFormat::Raw => "raw",
        }
    }

    pub fn media_type(self) -> &'static str {
        match self {
            FrameFormat::Ppm => "image/x-portable-pixmap",
            FrameFormat::Pgm => "image/x-portable-graymap",
            FrameFormat::Raw => "application/octet-stream",
        }
    }

    pub fn extension(self) -> &'static str {
        match self {
            FrameFormat::Ppm => "ppm",
            FrameFormat::Pgm => "pgm",
            FrameFormat::Raw => "yuv",
        }
    }
}

impl std::str::FromStr for FrameFormat {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        Ok(match s.trim().to_ascii_lowercase().as_str() {
            "ppm" | "rgb" => FrameFormat::Ppm,
            "pgm" | "gray" | "grey" | "luma" => FrameFormat::Pgm,
            "raw" | "yuv" => FrameFormat::Raw,
            other => bail!("unknown frame format `{other}` (ppm, pgm, raw)"),
        })
    }
}

/// `frame` encoded as `format`.
pub fn encode(frame: &VideoFrame, format: FrameFormat) -> Result<Vec<u8>> {
    Ok(match format {
        FrameFormat::Raw => frame.data.to_vec(),
        FrameFormat::Pgm => {
            let luma = luma8(frame)?;
            let mut out = format!("P5\n{} {}\n255\n", frame.width, frame.height).into_bytes();
            out.extend_from_slice(&luma);
            out
        }
        FrameFormat::Ppm => {
            let rgb = rgb8(frame)?;
            let mut out = format!("P6\n{} {}\n255\n", frame.width, frame.height).into_bytes();
            out.extend_from_slice(&rgb);
            out
        }
    })
}

/// Bits per sample, and whether samples are 16-bit LE words.
fn depth(format: PixelFormat) -> u32 {
    match format {
        PixelFormat::Yuv420p10le | PixelFormat::Yuv422p10le | PixelFormat::Yuv444p10le | PixelFormat::Yuva444p10le => 10,
        PixelFormat::Yuv420p12le | PixelFormat::Yuv422p12le | PixelFormat::Yuv444p12le => 12,
        _ => 8,
    }
}

/// One plane's samples as 8-bit.
fn plane8(data: &[u8], samples: usize, bits: u32) -> Result<Vec<u8>> {
    if bits == 8 {
        if data.len() < samples {
            bail!("frame plane is {} bytes, {} expected", data.len(), samples);
        }
        return Ok(data[..samples].to_vec());
    }
    if data.len() < samples * 2 {
        bail!("frame plane is {} bytes, {} expected", data.len(), samples * 2);
    }
    let shift = bits - 8;
    Ok(data[..samples * 2]
        .chunks_exact(2)
        .map(|c| (u16::from_le_bytes([c[0], c[1]]) >> shift).min(255) as u8)
        .collect())
}

/// The frame's luma, one byte a pixel, row-major, `width * height` long.
/// RGB frames are weighted BT.601 (the weights perceptual hashes are
/// conventionally computed with).
pub fn luma8(frame: &VideoFrame) -> Result<Vec<u8>> {
    let (w, h) = (frame.width as usize, frame.height as usize);
    let pixels = w * h;
    if pixels == 0 {
        bail!("frame has no pixels ({}x{})", frame.width, frame.height);
    }
    let data = &frame.data[..];
    match frame.format {
        PixelFormat::Rgb24 | PixelFormat::Rgba32 => {
            let step = if frame.format == PixelFormat::Rgb24 { 3 } else { 4 };
            if data.len() < pixels * step {
                bail!("{} frame is {} bytes, {} expected", frame.format.as_ffmpeg_str(), data.len(), pixels * step);
            }
            Ok(data
                .chunks_exact(step)
                .take(pixels)
                .map(|p| ((77 * p[0] as u32 + 150 * p[1] as u32 + 29 * p[2] as u32 + 128) >> 8) as u8)
                .collect())
        }
        f => plane8(data, pixels, depth(f)),
    }
}

/// The frame as 8-bit RGB, three bytes a pixel, row-major. YUV is converted
/// with the frame's matrix, limited range; chroma is replicated (nearest).
pub fn rgb8(frame: &VideoFrame) -> Result<Vec<u8>> {
    let (w, h) = (frame.width as usize, frame.height as usize);
    let pixels = w * h;
    if pixels == 0 {
        bail!("frame has no pixels ({}x{})", frame.width, frame.height);
    }
    let data = &frame.data[..];
    match frame.format {
        PixelFormat::Rgb24 => {
            if data.len() < pixels * 3 {
                bail!("rgb24 frame is {} bytes, {} expected", data.len(), pixels * 3);
            }
            return Ok(data[..pixels * 3].to_vec());
        }
        PixelFormat::Rgba32 => {
            if data.len() < pixels * 4 {
                bail!("rgba frame is {} bytes, {} expected", data.len(), pixels * 4);
            }
            return Ok(data.chunks_exact(4).take(pixels).flat_map(|p| [p[0], p[1], p[2]]).collect());
        }
        _ => {}
    }
    let bits = depth(frame.format);
    let bps = if bits == 8 { 1 } else { 2 };
    let y = plane8(data, pixels, bits)?;
    let rest = &data[pixels * bps..];
    // Chroma: (cw, ch) and the U / V planes as 8-bit, `cw * ch` each.
    let (cw, ch, u, v) = match frame.format {
        PixelFormat::Nv12 | PixelFormat::Nv21 => {
            let (cw, ch) = chroma_dims(w, h, 2, 2, rest.len() / 2);
            let n = cw * ch;
            if rest.len() < n * 2 {
                bail!("{} chroma is {} bytes, {} expected", frame.format.as_ffmpeg_str(), rest.len(), n * 2);
            }
            let (mut u, mut v) = (Vec::with_capacity(n), Vec::with_capacity(n));
            for pair in rest[..n * 2].chunks_exact(2) {
                u.push(pair[0]);
                v.push(pair[1]);
            }
            if frame.format == PixelFormat::Nv21 {
                std::mem::swap(&mut u, &mut v);
            }
            (cw, ch, u, v)
        }
        f => {
            let (sx, sy) = match f {
                PixelFormat::Yuv420p | PixelFormat::Yuv420p10le | PixelFormat::Yuv420p12le => (2, 2),
                PixelFormat::Yuv422p | PixelFormat::Yuv422p10le | PixelFormat::Yuv422p12le => (2, 1),
                _ => (1, 1),
            };
            let (cw, ch) = chroma_dims(w, h, sx, sy, rest.len() / (2 * bps));
            let n = cw * ch;
            let u = plane8(rest, n, bits)?;
            let v = plane8(&rest[(n * bps).min(rest.len())..], n, bits)?;
            (cw, ch, u, v)
        }
    };
    let (kr, kb) = match frame.color_space {
        ColorSpace::Bt601 => (0.299, 0.114),
        ColorSpace::Bt709 => (0.2126, 0.0722),
        ColorSpace::Bt2020 => (0.2627, 0.0593),
    };
    let kg = 1.0 - kr - kb;
    let (sx, sy) = (w.div_ceil(cw.max(1)).max(1), h.div_ceil(ch.max(1)).max(1));
    let mut out = Vec::with_capacity(pixels * 3);
    for row in 0..h {
        let crow = (row / sy).min(ch - 1);
        for col in 0..w {
            let ccol = (col / sx).min(cw - 1);
            let yy = (y[row * w + col] as f32 - 16.0) * (255.0 / 219.0);
            let cb = (u[crow * cw + ccol] as f32 - 128.0) * (255.0 / 224.0);
            let cr = (v[crow * cw + ccol] as f32 - 128.0) * (255.0 / 224.0);
            let r = yy + 2.0 * (1.0 - kr) * cr;
            let b = yy + 2.0 * (1.0 - kb) * cb;
            let g = (yy - kr * r - kb * b) / kg;
            out.extend([r, g, b].map(|c| c.round().clamp(0.0, 255.0) as u8));
        }
    }
    Ok(out)
}

/// The chroma plane's size for subsampling `(sx, sy)`: rounded up when the
/// data holds a rounded-up plane, else down.
fn chroma_dims(w: usize, h: usize, sx: usize, sy: usize, available: usize) -> (usize, usize) {
    let up = (w.div_ceil(sx), h.div_ceil(sy));
    if up.0 * up.1 <= available { up } else { ((w / sx).max(1), (h / sy).max(1)) }
}

/// How a letterboxed picture maps onto its source: the source was scaled by
/// `scale` and placed `pad_x` / `pad_y` pixels in. What a vision model's
/// coordinates are brought back through ([`Letterbox::to_source`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Letterbox {
    pub scale: f32,
    pub pad_x: u32,
    pub pad_y: u32,
    /// The source's size, which mapped points are clamped to.
    pub source: (u32, u32),
}

impl Letterbox {
    /// A point in the letterboxed picture → the same point in the source,
    /// clamped to it.
    pub fn to_source(&self, x: f32, y: f32) -> (f32, f32) {
        let sx = ((x - self.pad_x as f32) / self.scale).clamp(0.0, self.source.0 as f32);
        let sy = ((y - self.pad_y as f32) / self.scale).clamp(0.0, self.source.1 as f32);
        (sx, sy)
    }

    /// A box `(x, y, w, h)` in the letterboxed picture → the source.
    pub fn box_to_source(&self, x: f32, y: f32, w: f32, h: f32) -> (f32, f32, f32, f32) {
        let (x0, y0) = self.to_source(x, y);
        let (x1, y1) = self.to_source(x + w, y + h);
        (x0, y0, x1 - x0, y1 - y0)
    }
}

/// The frame as 8-bit RGB scaled to exactly `width × height` (aspect ratio
/// not kept), bilinear. For a model that takes a fixed input size and was
/// trained on stretched pictures.
pub fn rgb8_resized(frame: &VideoFrame, width: u32, height: u32) -> Result<Vec<u8>> {
    let rgb = rgb8(frame)?;
    Ok(resize_rgb(&rgb, frame.width, frame.height, width, height))
}

/// The frame as 8-bit RGB fitted inside `width × height` with its aspect ratio
/// kept, centred, the rest filled with `fill` — the "letterbox" most detection
/// models (YOLO among them) expect — and the [`Letterbox`] that maps the
/// model's coordinates back onto the frame.
pub fn rgb8_letterboxed(frame: &VideoFrame, width: u32, height: u32, fill: [u8; 3]) -> Result<(Vec<u8>, Letterbox)> {
    if width == 0 || height == 0 {
        bail!("a letterbox needs a size (got {width}x{height})");
    }
    let rgb = rgb8(frame)?;
    let scale = (width as f32 / frame.width as f32).min(height as f32 / frame.height as f32);
    let (sw, sh) = (
        ((frame.width as f32 * scale).round() as u32).clamp(1, width),
        ((frame.height as f32 * scale).round() as u32).clamp(1, height),
    );
    let (pad_x, pad_y) = ((width - sw) / 2, (height - sh) / 2);
    let scaled = resize_rgb(&rgb, frame.width, frame.height, sw, sh);
    let mut out: Vec<u8> = fill.iter().copied().cycle().take((width * height * 3) as usize).collect();
    for row in 0..sh as usize {
        let dst = ((row + pad_y as usize) * width as usize + pad_x as usize) * 3;
        out[dst..dst + sw as usize * 3].copy_from_slice(&scaled[row * sw as usize * 3..(row + 1) * sw as usize * 3]);
    }
    Ok((out, Letterbox { scale, pad_x, pad_y, source: (frame.width, frame.height) }))
}

/// Interleaved 8-bit RGB (`width × height`) → planar `f32` in `0.0..=1.0`,
/// channel by channel (R plane, G plane, B plane): the NCHW layout (batch of
/// one) most vision models take.
pub fn rgb8_to_planar_f32(rgb: &[u8], width: u32, height: u32) -> Vec<f32> {
    let n = (width * height) as usize;
    let mut out = vec![0f32; n * 3];
    for (i, px) in rgb.chunks_exact(3).take(n).enumerate() {
        for c in 0..3 {
            out[c * n + i] = px[c] as f32 / 255.0;
        }
    }
    out
}

/// Bilinear resize of interleaved 8-bit RGB, sampling pixel centres.
fn resize_rgb(rgb: &[u8], sw: u32, sh: u32, dw: u32, dh: u32) -> Vec<u8> {
    let (sw_, sh_) = (sw as usize, sh as usize);
    let mut out = Vec::with_capacity((dw * dh * 3) as usize);
    let (fx, fy) = (sw as f32 / dw as f32, sh as f32 / dh as f32);
    for y in 0..dh {
        let sy = ((y as f32 + 0.5) * fy - 0.5).clamp(0.0, (sh_ - 1) as f32);
        let (y0, ty) = (sy.floor() as usize, sy.fract());
        let y1 = (y0 + 1).min(sh_ - 1);
        for x in 0..dw {
            let sx = ((x as f32 + 0.5) * fx - 0.5).clamp(0.0, (sw_ - 1) as f32);
            let (x0, tx) = (sx.floor() as usize, sx.fract());
            let x1 = (x0 + 1).min(sw_ - 1);
            for c in 0..3 {
                let p = |xx: usize, yy: usize| rgb[(yy * sw_ + xx) * 3 + c] as f32;
                let top = p(x0, y0) * (1.0 - tx) + p(x1, y0) * tx;
                let bottom = p(x0, y1) * (1.0 - tx) + p(x1, y1) * tx;
                out.push((top * (1.0 - ty) + bottom * ty).round().clamp(0.0, 255.0) as u8);
            }
        }
    }
    out
}
