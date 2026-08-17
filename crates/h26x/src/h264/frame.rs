//! The decoded frame buffer: three planes with a replicated border, plus the
//! per-4x4-block motion data a later picture reads back (direct mode,
//! deblocking) and the per-macroblock facts the deblocker needs.

use crate::picture::{ChromaFormat, Picture, Plane};

/// Luma border in samples on every side. Motion compensation reads up to
/// 3 samples beyond a block on each side (the six-tap filter), and vectors
/// pointing outside the picture read the replicated edge; anything further
/// out than the border is clamped by the fetch path.
pub const LUMA_PAD: usize = 32;
/// Chroma border.
pub const CHROMA_PAD: usize = 16;

/// A motion vector in quarter-sample units.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Mv {
    /// Horizontal component.
    pub x: i16,
    /// Vertical component.
    pub y: i16,
}

impl Mv {
    /// The zero vector.
    pub const ZERO: Mv = Mv { x: 0, y: 0 };
    /// Construct.
    pub const fn new(x: i16, y: i16) -> Mv {
        Mv { x, y }
    }
}

/// Motion data of one 4x4 block, for one reference list.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockMotion {
    /// The vector.
    pub mv: Mv,
    /// Reference index into the slice's list, or -1.
    pub ref_idx: i8,
    /// POC of the referenced picture (identifies it for direct mode and
    /// deblocking); `i32::MIN` when `ref_idx < 0`.
    pub ref_poc: i32,
    /// Whether the referenced picture was a long-term reference.
    pub ref_long_term: bool,
}

impl Default for BlockMotion {
    fn default() -> Self {
        Self { mv: Mv::ZERO, ref_idx: -1, ref_poc: i32::MIN, ref_long_term: false }
    }
}

/// One plane with a border.
#[derive(Debug, Clone)]
pub struct PaddedPlane {
    /// Samples, `(width + 2*pad) * (height + 2*pad)`.
    pub data: Vec<u8>,
    /// Visible width.
    pub width: usize,
    /// Visible height.
    pub height: usize,
    /// Border on each side.
    pub pad: usize,
    /// Row stride.
    pub stride: usize,
}

impl PaddedPlane {
    fn new(width: usize, height: usize, pad: usize) -> Self {
        let stride = width + 2 * pad;
        Self { data: vec![0; stride * (height + 2 * pad)], width, height, pad, stride }
    }
    /// Offset of visible sample (0, 0).
    #[inline(always)]
    pub fn origin(&self) -> usize {
        self.pad * self.stride + self.pad
    }
    /// Offset of visible sample (x, y) — x and y may be negative within the
    /// border.
    #[inline(always)]
    pub fn offset(&self, x: isize, y: isize) -> usize {
        (self.origin() as isize + y * self.stride as isize + x) as usize
    }
    /// Sample at (x, y) — x and y may reach into the border.
    #[inline(always)]
    pub fn at(&self, x: isize, y: isize) -> u8 {
        self.data[self.offset(x, y)]
    }
    /// Replicate the visible edge samples into the border.
    pub fn extend_edges(&mut self) {
        let (w, h, pad, stride) = (self.width, self.height, self.pad, self.stride);
        let origin = self.origin();
        // Left and right.
        for y in 0..h {
            let row = origin + y * stride;
            let l = self.data[row];
            let r = self.data[row + w - 1];
            for i in 1..=pad {
                self.data[row - i] = l;
                self.data[row + w - 1 + i] = r;
            }
        }
        // Top and bottom: whole rows, corners included (the left/right
        // extension above already filled them for the visible rows).
        let first = origin - pad;
        for i in 1..=pad {
            self.data.copy_within(first..first + stride, first - i * stride);
        }
        let last = origin + (h - 1) * stride - pad;
        for i in 1..=pad {
            self.data.copy_within(last..last + stride, last + i * stride);
        }
    }
}

/// A decoded frame.
#[derive(Debug, Clone)]
pub struct Frame {
    /// Luma.
    pub y: PaddedPlane,
    /// Cb, Cr (empty planes for monochrome).
    pub cb: PaddedPlane,
    /// See `cb`.
    pub cr: PaddedPlane,
    /// Chroma format.
    pub chroma: ChromaFormat,
    /// Width in macroblocks.
    pub mb_width: usize,
    /// Height in macroblocks.
    pub mb_height: usize,
    /// Per-4x4-block motion, per list: index `mb_addr * 16 + blk4x4_raster`
    /// (raster within the macroblock: `by * 4 + bx`).
    pub motion: [Vec<BlockMotion>; 2],
    /// Per-macroblock: intra coded (for temporal/spatial direct — an intra
    /// colocated block behaves as if it had no motion).
    pub mb_intra: Vec<bool>,
    /// POC of this frame.
    pub poc: i32,
    /// Whether this frame is/was a long-term reference (read by direct mode
    /// through the colocated picture).
    pub long_term: bool,
}

impl Frame {
    /// Allocate a frame for `mb_width x mb_height` macroblocks.
    pub fn new(mb_width: usize, mb_height: usize, chroma: ChromaFormat) -> Self {
        let w = mb_width * 16;
        let h = mb_height * 16;
        let (cw, ch) = match chroma {
            ChromaFormat::Monochrome => (0, 0),
            ChromaFormat::Yuv420 => (w / 2, h / 2),
            ChromaFormat::Yuv422 => (w / 2, h),
            ChromaFormat::Yuv444 => (w, h),
        };
        let n = mb_width * mb_height;
        Self {
            y: PaddedPlane::new(w, h, LUMA_PAD),
            cb: PaddedPlane::new(cw, ch, CHROMA_PAD),
            cr: PaddedPlane::new(cw, ch, CHROMA_PAD),
            chroma,
            mb_width,
            mb_height,
            motion: [vec![BlockMotion::default(); n * 16], vec![BlockMotion::default(); n * 16]],
            mb_intra: vec![false; n],
            poc: 0,
            long_term: false,
        }
    }

    /// Replicate edges of all planes (call once the picture is fully
    /// decoded and filtered, before it is used as a reference).
    pub fn extend_edges(&mut self) {
        self.y.extend_edges();
        if self.chroma != ChromaFormat::Monochrome {
            self.cb.extend_edges();
            self.cr.extend_edges();
        }
    }

    /// Copy out the visible picture, cropped by `(left, right, top, bottom)`
    /// luma samples.
    pub fn to_picture(&self, crop: (u32, u32, u32, u32), poc: i32, decode_index: u64) -> Picture {
        let (l, r, t, b) = (crop.0 as usize, crop.1 as usize, crop.2 as usize, crop.3 as usize);
        let width = self.y.width - l - r;
        let height = self.y.height - t - b;
        let mut planes = Vec::with_capacity(3);
        let mut plane = |p: &PaddedPlane, x0: usize, y0: usize, w: usize, h: usize| {
            let mut data = Vec::with_capacity(w * h);
            for yy in 0..h {
                let off = p.offset((x0) as isize, (y0 + yy) as isize);
                data.extend_from_slice(&p.data[off..off + w]);
            }
            planes.push(Plane { data, width: w as u32, height: h as u32 });
        };
        plane(&self.y, l, t, width, height);
        if self.chroma != ChromaFormat::Monochrome {
            let (sw, sh) = self.chroma.subsampling();
            let (sw, sh) = (sw as usize, sh as usize);
            plane(&self.cb, l / sw, t / sh, width / sw, height / sh);
            plane(&self.cr, l / sw, t / sh, width / sw, height / sh);
        }
        Picture { width: width as u32, height: height as u32, bit_depth: 8, chroma: self.chroma, planes, poc, decode_index }
    }
}
