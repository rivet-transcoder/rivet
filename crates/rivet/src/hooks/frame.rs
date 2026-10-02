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
///
/// Scales and converts in one pass, straight from the decoder's planes, so the
/// cost follows the output's size rather than the frame's: a 4K frame costs
/// about what a 720p one does.
pub fn rgb8_resized(frame: &VideoFrame, width: u32, height: u32) -> Result<Vec<u8>> {
    if width == 0 || height == 0 {
        bail!("a resize needs a size (got {width}x{height})");
    }
    let mut out = vec![0u8; (width * height * 3) as usize];
    let w = width as usize;
    Planes::of(frame)?.scaled(width, height, |x, y, rgb| out[(y * w + x) * 3..(y * w + x) * 3 + 3].copy_from_slice(&rgb));
    Ok(out)
}

/// The size `frame` is scaled to inside a `width × height` letterbox, and
/// where it is placed.
fn fit(frame: &VideoFrame, width: u32, height: u32) -> Result<(u32, u32, Letterbox)> {
    if width == 0 || height == 0 {
        bail!("a letterbox needs a size (got {width}x{height})");
    }
    let scale = (width as f32 / frame.width.max(1) as f32).min(height as f32 / frame.height.max(1) as f32);
    let (sw, sh) = (
        ((frame.width as f32 * scale).round() as u32).clamp(1, width),
        ((frame.height as f32 * scale).round() as u32).clamp(1, height),
    );
    let (pad_x, pad_y) = ((width - sw) / 2, (height - sh) / 2);
    Ok((sw, sh, Letterbox { scale, pad_x, pad_y, source: (frame.width, frame.height) }))
}

/// The frame as 8-bit RGB fitted inside `width × height` with its aspect ratio
/// kept, centred, the rest filled with `fill` — the "letterbox" most detection
/// models (YOLO among them) expect — and the [`Letterbox`] that maps the
/// model's coordinates back onto the frame. Like [`rgb8_resized`], one pass
/// from the decoder's planes.
pub fn rgb8_letterboxed(frame: &VideoFrame, width: u32, height: u32, fill: [u8; 3]) -> Result<(Vec<u8>, Letterbox)> {
    let planes = Planes::of(frame)?;
    let (sw, sh, letterbox) = fit(frame, width, height)?;
    let mut out: Vec<u8> = fill.iter().copied().cycle().take((width * height * 3) as usize).collect();
    let (w, px, py) = (width as usize, letterbox.pad_x as usize, letterbox.pad_y as usize);
    planes.scaled(sw, sh, |x, y, rgb| {
        let at = ((y + py) * w + px + x) * 3;
        out[at..at + 3].copy_from_slice(&rgb);
    });
    Ok((out, letterbox))
}

/// [`rgb8_letterboxed`] straight into the tensor a vision model takes: planar
/// `f32` in `0.0..=1.0`, channel by channel (the NCHW layout, batch of one).
/// The same values as `rgb8_to_planar_f32(&rgb8_letterboxed(..).0, ..)`,
/// without the interleaved picture in between.
pub fn planar_f32_letterboxed(frame: &VideoFrame, width: u32, height: u32, fill: [u8; 3]) -> Result<(Vec<f32>, Letterbox)> {
    let planes = Planes::of(frame)?;
    let (sw, sh, letterbox) = fit(frame, width, height)?;
    let n = (width * height) as usize;
    let mut out = vec![0f32; n * 3];
    for (c, plane) in out.chunks_exact_mut(n).enumerate() {
        plane.fill(UNIT[fill[c] as usize]);
    }
    let (w, px, py) = (width as usize, letterbox.pad_x as usize, letterbox.pad_y as usize);
    planes.scaled(sw, sh, |x, y, rgb| {
        let at = (y + py) * w + px + x;
        for (c, v) in rgb.into_iter().enumerate() {
            out[c * n + at] = UNIT[v as usize];
        }
    });
    Ok((out, letterbox))
}

/// `v / 255.0` for every 8-bit `v`: a load instead of a division per sample.
const UNIT: [f32; 256] = {
    let mut table = [0f32; 256];
    let mut v = 0;
    while v < 256 {
        table[v] = v as f32 / 255.0;
        v += 1;
    }
    table
};

/// Interleaved 8-bit RGB (`width × height`) → planar `f32` in `0.0..=1.0`,
/// channel by channel (R plane, G plane, B plane): the NCHW layout (batch of
/// one) most vision models take.
pub fn rgb8_to_planar_f32(rgb: &[u8], width: u32, height: u32) -> Vec<f32> {
    let n = (width * height) as usize;
    let mut out = vec![0f32; n * 3];
    for (i, px) in rgb.chunks_exact(3).take(n).enumerate() {
        for c in 0..3 {
            out[c * n + i] = UNIT[px[c] as usize];
        }
    }
    out
}

/// One plane of a frame, read in place: sample `(x, y)` is sample number
/// `y * stride + x * step + offset`, of `bytes` bytes each.
#[derive(Clone, Copy)]
struct PlaneView<'a> {
    data: &'a [u8],
    w: usize,
    h: usize,
    /// In samples.
    stride: usize,
    step: usize,
    offset: usize,
    /// 1, or 2 for 16-bit LE words.
    bytes: usize,
    /// What a sample is multiplied by to bring it to 8 bits.
    to8: f32,
}

/// Where a bilinear sample falls along one axis: the two neighbours (as
/// sample offsets: row × stride, or column × step) and the weight of the
/// second.
#[derive(Clone, Copy)]
struct Tap {
    a: usize,
    b: usize,
    t: f32,
}

/// The taps of `count` output positions over `len` samples `scale` apart,
/// output `i` landing at sample coordinate `at(i)` (centres at integers),
/// clamped to the edges.
fn taps(count: u32, len: usize, scale: usize, at: impl Fn(f32) -> f32) -> Vec<Tap> {
    (0..count)
        .map(|i| {
            let p = at(i as f32).clamp(0.0, (len - 1) as f32);
            let a = p as usize;
            Tap { a: a * scale, b: (a + 1).min(len - 1) * scale, t: p - a as f32 }
        })
        .collect()
}

/// How a plane's samples are stored. Monomorphised, so the inner loop has no
/// per-sample branch on the bit depth.
trait Samples {
    fn read(plane: &PlaneView, sample: usize) -> f32;
}

/// 8-bit samples.
struct Narrow;
/// 16-bit LE samples (10 or 12 significant bits).
struct Wide;

impl Samples for Narrow {
    #[inline(always)]
    fn read(plane: &PlaneView, sample: usize) -> f32 {
        plane.data[sample] as f32
    }
}

impl Samples for Wide {
    #[inline(always)]
    fn read(plane: &PlaneView, sample: usize) -> f32 {
        u16::from_le_bytes([plane.data[sample * 2], plane.data[sample * 2 + 1]]) as f32 * plane.to8
    }
}

impl PlaneView<'_> {
    /// Bilinear between rows `row` and columns `col` (taps of this plane).
    #[inline(always)]
    fn bilinear<S: Samples>(&self, row: Tap, col: Tap) -> f32 {
        let (r0, r1) = (row.a + self.offset, row.b + self.offset);
        let top = S::read(self, r0 + col.a) * (1.0 - col.t) + S::read(self, r0 + col.b) * col.t;
        let bottom = S::read(self, r1 + col.a) * (1.0 - col.t) + S::read(self, r1 + col.b) * col.t;
        top * (1.0 - row.t) + bottom * row.t
    }
}

/// A frame's three planes in place, and how to turn a sample of each into RGB.
struct Planes<'a> {
    p: [PlaneView<'a>; 3],
    /// `Some((kr, kb))` for limited-range YUV with that matrix; `None` for RGB.
    yuv: Option<(f32, f32)>,
}

impl<'a> Planes<'a> {
    fn of(frame: &'a VideoFrame) -> Result<Planes<'a>> {
        let (w, h) = (frame.width as usize, frame.height as usize);
        let pixels = w * h;
        if pixels == 0 {
            bail!("frame has no pixels ({}x{})", frame.width, frame.height);
        }
        let data = &frame.data[..];
        if let PixelFormat::Rgb24 | PixelFormat::Rgba32 = frame.format {
            let step = if frame.format == PixelFormat::Rgb24 { 3 } else { 4 };
            if data.len() < pixels * step {
                bail!("{} frame is {} bytes, {} expected", frame.format.as_ffmpeg_str(), data.len(), pixels * step);
            }
            let plane = |offset| PlaneView { data, w, h, stride: w * step, step, offset, bytes: 1, to8: 1.0 };
            return Ok(Planes { p: [plane(0), plane(1), plane(2)], yuv: None });
        }
        let bits = depth(frame.format);
        let bytes = if bits == 8 { 1 } else { 2 };
        let to8 = 1.0 / (1u32 << (bits - 8)) as f32;
        if data.len() < pixels * bytes {
            bail!("frame plane is {} bytes, {} expected", data.len(), pixels * bytes);
        }
        let y = PlaneView { data, w, h, stride: w, step: 1, offset: 0, bytes, to8 };
        let rest = &data[pixels * bytes..];
        let (u, v) = match frame.format {
            PixelFormat::Nv12 | PixelFormat::Nv21 => {
                let (cw, ch) = chroma_dims(w, h, 2, 2, rest.len() / 2);
                if rest.len() < cw * ch * 2 {
                    bail!("{} chroma is {} bytes, {} expected", frame.format.as_ffmpeg_str(), rest.len(), cw * ch * 2);
                }
                let c = |offset| PlaneView { data: rest, w: cw, h: ch, stride: cw * 2, step: 2, offset, bytes: 1, to8: 1.0 };
                if frame.format == PixelFormat::Nv12 { (c(0), c(1)) } else { (c(1), c(0)) }
            }
            f => {
                let (sx, sy) = match f {
                    PixelFormat::Yuv420p | PixelFormat::Yuv420p10le | PixelFormat::Yuv420p12le => (2, 2),
                    PixelFormat::Yuv422p | PixelFormat::Yuv422p10le | PixelFormat::Yuv422p12le => (2, 1),
                    _ => (1, 1),
                };
                let (cw, ch) = chroma_dims(w, h, sx, sy, rest.len() / (2 * bytes));
                let n = cw * ch * bytes;
                if rest.len() < n * 2 {
                    bail!("frame chroma is {} bytes, {} expected", rest.len(), n * 2);
                }
                let c = |data| PlaneView { data, w: cw, h: ch, stride: cw, step: 1, offset: 0, bytes, to8 };
                (c(&rest[..n]), c(&rest[n..]))
            }
        };
        let (kr, kb) = match frame.color_space {
            ColorSpace::Bt601 => (0.299, 0.114),
            ColorSpace::Bt709 => (0.2126, 0.0722),
            ColorSpace::Bt2020 => (0.2627, 0.0593),
        };
        Ok(Planes { p: [y, u, v], yuv: Some((kr, kb)) })
    }

    /// The picture scaled to `dw × dh`, bilinear, each pixel's 8-bit RGB
    /// handed to `put(x, y, rgb)` in row-major order. Only the samples the
    /// output needs are read, and only those are converted.
    fn scaled(&self, dw: u32, dh: u32, put: impl FnMut(usize, usize, [u8; 3])) {
        if self.p.iter().all(|p| p.bytes == 1) {
            self.scaled_as::<Narrow>(dw, dh, put)
        } else {
            self.scaled_as::<Wide>(dw, dh, put)
        }
    }

    fn scaled_as<S: Samples>(&self, dw: u32, dh: u32, mut put: impl FnMut(usize, usize, [u8; 3])) {
        let [y, u, v] = &self.p;
        // Round half up, as `f32::round` does for these non-negative values,
        // but a truncating conversion rather than a libm call on baseline x86.
        let to8 = |c: f32| (c.clamp(0.0, 255.0) + 0.5) as u8;
        let (fx, fy) = (y.w as f32 / dw as f32, y.h as f32 / dh as f32);
        let cols = taps(dw, y.w, y.step, |c| (c + 0.5) * fx - 0.5);
        let rows = taps(dh, y.h, y.stride, |r| (r + 0.5) * fy - 0.5);
        let Some((kr, kb)) = self.yuv else {
            for (oy, &row) in rows.iter().enumerate() {
                for (ox, &col) in cols.iter().enumerate() {
                    put(ox, oy, [y, u, v].map(|p| to8(p.bilinear::<S>(row, col))));
                }
            }
            return;
        };
        // Chroma sits on its own, coarser grid.
        let (cx, cy) = (u.w as f32 / y.w as f32, u.h as f32 / y.h as f32);
        let ccols = taps(dw, u.w, u.step, |c| (c + 0.5) * fx * cx - 0.5);
        let crows = taps(dh, u.h, u.stride, |r| (r + 0.5) * fy * cy - 0.5);
        // Limited range to full, and the matrix, folded together.
        let (ys, cs) = (255.0 / 219.0, 255.0 / 224.0);
        let kg = 1.0 - kr - kb;
        let (rv, bu) = (2.0 * (1.0 - kr) * cs, 2.0 * (1.0 - kb) * cs);
        let (gu, gv) = (kb * bu / kg, kr * rv / kg);
        for (oy, (&row, &crow)) in rows.iter().zip(&crows).enumerate() {
            for (ox, (&col, &ccol)) in cols.iter().zip(&ccols).enumerate() {
                let yy = (y.bilinear::<S>(row, col) - 16.0) * ys;
                let cb = u.bilinear::<S>(crow, ccol) - 128.0;
                let cr = v.bilinear::<S>(crow, ccol) - 128.0;
                put(ox, oy, [to8(yy + rv * cr), to8(yy - gu * cb - gv * cr), to8(yy + bu * cb)]);
            }
        }
    }
}
