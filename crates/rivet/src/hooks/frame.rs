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
