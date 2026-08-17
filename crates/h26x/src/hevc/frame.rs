//! HEVC picture buffers: 16-bit sample planes with a replicated border (one
//! representation for 8- and 10-bit), and the per-picture side data later
//! pictures and the loop filters read.

use crate::picture::{ChromaFormat, Picture, Plane};
use crate::threading::Progress;

/// Luma border in samples on every side (the 8-tap filter needs 3/4; the
/// rest absorbs vectors that leave the picture, with a clamped slow path
/// beyond it).
pub const LUMA_PAD: usize = 80;
/// Chroma border.
pub const CHROMA_PAD: usize = 40;

/// A motion vector in quarter-sample units.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Mv {
    /// Horizontal.
    pub x: i16,
    /// Vertical.
    pub y: i16,
}

impl Mv {
    /// Zero.
    pub const ZERO: Mv = Mv { x: 0, y: 0 };
    /// Construct.
    pub const fn new(x: i16, y: i16) -> Mv {
        Mv { x, y }
    }
}

/// Motion of one 4x4 block: both lists.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MotionInfo {
    /// Vectors per list.
    pub mv: [Mv; 2],
    /// Reference index per list (-1 = list unused).
    pub ref_idx: [i8; 2],
    /// POC of the referenced picture per list (for TMVP and deblocking).
    pub ref_poc: [i32; 2],
    /// Whether the referenced picture is long-term, per list.
    pub ref_long_term: [bool; 2],
    /// Intra (no motion) — TMVP treats it as unavailable.
    pub intra: bool,
}

impl Default for MotionInfo {
    fn default() -> Self {
        MotionInfo { mv: [Mv::ZERO; 2], ref_idx: [-1; 2], ref_poc: [0; 2], ref_long_term: [false; 2], intra: true }
    }
}

impl MotionInfo {
    /// `predFlagLX`.
    #[inline]
    pub fn uses(&self, list: usize) -> bool {
        self.ref_idx[list] >= 0
    }
}

/// One plane of `u16` samples with a border.
#[derive(Debug, Clone)]
pub struct Plane16 {
    /// Samples.
    pub data: Vec<u16>,
    /// Visible width.
    pub width: usize,
    /// Visible height.
    pub height: usize,
    /// Border.
    pub pad: usize,
    /// Row stride.
    pub stride: usize,
}

impl Plane16 {
    fn new(width: usize, height: usize, pad: usize) -> Self {
        let stride = width + 2 * pad;
        Plane16 { data: vec![0; stride * (height + 2 * pad)], width, height, pad, stride }
    }
    /// Offset of visible sample (0, 0).
    #[inline(always)]
    pub fn origin(&self) -> usize {
        self.pad * self.stride + self.pad
    }
    /// Offset of sample (x, y); may reach into the border.
    #[inline(always)]
    pub fn offset(&self, x: isize, y: isize) -> usize {
        (self.origin() as isize + y * self.stride as isize + x) as usize
    }
    /// Sample at (x, y) with coordinates clamped to the visible picture.
    #[inline(always)]
    pub fn at_clamped(&self, x: i32, y: i32) -> u16 {
        let xx = x.clamp(0, self.width as i32 - 1) as isize;
        let yy = y.clamp(0, self.height as i32 - 1) as isize;
        self.data[self.offset(xx, yy)]
    }
    /// Sample at (x, y) — the coordinates must be inside the padded area.
    #[inline(always)]
    pub fn at(&self, x: isize, y: isize) -> u16 {
        self.data[self.offset(x, y)]
    }
    /// Replicate the left/right edge samples of rows `y0..y1` into the border.
    pub fn extend_rows(&mut self, y0: usize, y1: usize) {
        let (w, pad, stride) = (self.width, self.pad, self.stride);
        if w == 0 {
            return;
        }
        let origin = self.origin();
        for y in y0..y1.min(self.height) {
            let row = origin + y * stride;
            let l = self.data[row];
            let r = self.data[row + w - 1];
            self.data[row - pad..row].fill(l);
            self.data[row + w..row + w + pad].fill(r);
        }
    }

    /// Replicate the (already row-extended) first row upwards into the border.
    pub fn extend_top(&mut self) {
        let (pad, stride) = (self.pad, self.stride);
        let first = self.origin() - pad;
        for i in 1..=pad {
            self.data.copy_within(first..first + stride, first - i * stride);
        }
    }

    /// Replicate the (already row-extended) last row downwards into the border.
    pub fn extend_bottom(&mut self) {
        let (h, pad, stride) = (self.height, self.pad, self.stride);
        if h == 0 {
            return;
        }
        let last = self.origin() + (h - 1) * stride - pad;
        for i in 1..=pad {
            self.data.copy_within(last..last + stride, last + i * stride);
        }
    }

    /// Replicate the visible edges into the border.
    pub fn extend_edges(&mut self) {
        let (w, h, pad, stride) = (self.width, self.height, self.pad, self.stride);
        if w == 0 || h == 0 {
            return;
        }
        let origin = self.origin();
        for y in 0..h {
            let row = origin + y * stride;
            let l = self.data[row];
            let r = self.data[row + w - 1];
            for i in 1..=pad {
                self.data[row - i] = l;
                self.data[row + w - 1 + i] = r;
            }
        }
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

/// A decoded HEVC picture.
#[derive(Debug, Clone)]
pub struct Frame {
    /// Luma.
    pub y: Plane16,
    /// Cb.
    pub cb: Plane16,
    /// Cr.
    pub cr: Plane16,
    /// Chroma format.
    pub chroma: ChromaFormat,
    /// Bit depth (luma; chroma equal in the profiles supported).
    pub bit_depth: u32,
    /// Luma width / height in samples.
    pub width: usize,
    /// See `width`.
    pub height: usize,
    /// Width in 4x4 blocks (`ceil(width / 4)`).
    pub w4: usize,
    /// Height in 4x4 blocks.
    pub h4: usize,
    /// Motion per 4x4 block, raster over the picture: `y4 * w4 + x4`.
    pub motion: Vec<MotionInfo>,
    /// POC.
    pub poc: i32,
    /// Long-term reference (as seen when used as a collocated picture).
    pub long_term: bool,
}

impl Frame {
    /// A zero-size placeholder (no buffers).
    pub fn empty() -> Self {
        let none = || Plane16 { data: Vec::new(), width: 0, height: 0, pad: 0, stride: 0 };
        Frame { y: none(), cb: none(), cr: none(), chroma: ChromaFormat::Yuv420, bit_depth: 8, width: 0, height: 0, w4: 0, h4: 0, motion: Vec::new(), poc: 0, long_term: false }
    }

    /// Allocate.
    pub fn new(width: usize, height: usize, chroma: ChromaFormat, bit_depth: u32) -> Self {
        let (cw, ch) = match chroma {
            ChromaFormat::Monochrome => (0, 0),
            ChromaFormat::Yuv420 => (width.div_ceil(2), height.div_ceil(2)),
            ChromaFormat::Yuv422 => (width.div_ceil(2), height),
            ChromaFormat::Yuv444 => (width, height),
        };
        let w4 = width.div_ceil(4);
        let h4 = height.div_ceil(4);
        Frame {
            y: Plane16::new(width, height, LUMA_PAD),
            cb: Plane16::new(cw, ch, CHROMA_PAD),
            cr: Plane16::new(cw, ch, CHROMA_PAD),
            chroma,
            bit_depth,
            width,
            height,
            w4,
            h4,
            motion: vec![MotionInfo::default(); w4 * h4],
            poc: 0,
            long_term: false,
        }
    }

    /// Motion of the 4x4 block containing luma sample (x, y).
    #[inline]
    pub fn motion_at(&self, x: usize, y: usize) -> &MotionInfo {
        &self.motion[(y / 4) * self.w4 + x / 4]
    }

    /// Replicate edges of every plane.
    pub fn extend_edges(&mut self) {
        self.y.extend_edges();
        if self.chroma != ChromaFormat::Monochrome {
            self.cb.extend_edges();
            self.cr.extend_edges();
        }
    }

    /// Extend the borders of luma rows `y0..y1` (and the matching chroma rows)
    /// left/right; the top border once `y0 == 0`, the bottom once `y1 >= height`.
    pub fn extend_rows(&mut self, y0: usize, y1: usize) {
        let y1 = y1.min(self.height);
        self.y.extend_rows(y0, y1);
        let (sw, sh) = self.chroma.subsampling();
        let _ = sw;
        if self.chroma != ChromaFormat::Monochrome {
            let (cy0, cy1) = (y0 / sh as usize, y1.div_ceil(sh as usize));
            self.cb.extend_rows(cy0, cy1);
            self.cr.extend_rows(cy0, cy1);
        }
        if y0 == 0 {
            self.y.extend_top();
            if self.chroma != ChromaFormat::Monochrome {
                self.cb.extend_top();
                self.cr.extend_top();
            }
        }
        if y1 >= self.height {
            self.y.extend_bottom();
            if self.chroma != ChromaFormat::Monochrome {
                self.cb.extend_bottom();
                self.cr.extend_bottom();
            }
        }
    }

    /// Copy the visible, cropped picture out. 8-bit output as bytes, higher
    /// depths as little-endian `u16`.
    pub fn to_picture(&self, crop: (u32, u32, u32, u32), poc: i32, decode_index: u64) -> Picture {
        let (l, r, t, b) = (crop.0 as usize, crop.1 as usize, crop.2 as usize, crop.3 as usize);
        let width = self.width.saturating_sub(l + r).max(1);
        let height = self.height.saturating_sub(t + b).max(1);
        let mut planes = Vec::with_capacity(3);
        let eight = self.bit_depth == 8;
        let mut plane = |p: &Plane16, x0: usize, y0: usize, w: usize, h: usize| {
            let mut data = Vec::with_capacity(w * h * if eight { 1 } else { 2 });
            for yy in 0..h {
                let off = p.offset(x0 as isize, (y0 + yy) as isize);
                if eight {
                    data.extend(p.data[off..off + w].iter().map(|&v| v as u8));
                } else {
                    for &v in &p.data[off..off + w] {
                        data.extend_from_slice(&v.to_le_bytes());
                    }
                }
            }
            planes.push(Plane { data, width: w as u32, height: h as u32 });
        };
        plane(&self.y, l, t, width, height);
        if self.chroma != ChromaFormat::Monochrome {
            let (sw, sh) = self.chroma.subsampling();
            let (sw, sh) = (sw as usize, sh as usize);
            plane(&self.cb, l / sw, t / sh, width.div_ceil(sw), height.div_ceil(sh));
            plane(&self.cr, l / sw, t / sh, width.div_ceil(sw), height.div_ceil(sh));
        }
        Picture { width: width as u32, height: height as u32, bit_depth: self.bit_depth, chroma: self.chroma, planes, poc, decode_index }
    }
}

/// A picture shared between the thread decoding it and the threads
/// decoding later pictures that reference it.
///
/// The writer holds the only `&mut Frame` (through [`SharedFrame::get_mut`])
/// for the picture's lifetime as the current picture; readers take `&Frame`
/// through [`SharedFrame::get`] and only touch rows below what
/// [`SharedFrame::progress`] says is ready (samples: `done`; motion:
/// `decoded`). That row discipline plus the acquire/release publication in
/// [`Progress`] is what makes the concurrent access sound in practice — the
/// same contract libavcodec's frame threading relies on.
pub struct SharedFrame {
    inner: std::cell::UnsafeCell<Frame>,
    /// Row progress.
    pub progress: Progress,
    /// POC (fixed at creation).
    pub poc: i32,
    /// Unique id.
    pub id: u64,
    /// Where the buffers go back to when this picture is dropped.
    pool: Option<FramePool>,
}

/// Recycled frame buffers: allocation and zeroing of a picture's planes cost
/// as much as decoding a few CTB rows, so pictures leaving the DPB hand
/// their buffers to the next picture of the same geometry.
#[derive(Clone, Default)]
pub struct FramePool(std::sync::Arc<std::sync::Mutex<Vec<Frame>>>);

impl FramePool {
    /// Empty pool.
    pub fn new() -> Self {
        Self::default()
    }

    /// A frame of the given geometry, recycled if one is available (its
    /// samples are stale — every sample gets written before it is read).
    pub fn take(&self, width: usize, height: usize, chroma: ChromaFormat, bit_depth: u32) -> Frame {
        let mut g = self.0.lock().unwrap();
        if let Some(i) = g.iter().position(|f| f.width == width && f.height == height && f.chroma == chroma && f.bit_depth == bit_depth) {
            let mut f = g.swap_remove(i);
            f.motion.fill(MotionInfo::default());
            f.poc = 0;
            return f;
        }
        drop(g);
        Frame::new(width, height, chroma, bit_depth)
    }

    /// Return a frame.
    pub fn give(&self, f: Frame) {
        if f.width == 0 {
            return;
        }
        let mut g = self.0.lock().unwrap();
        if g.len() < 32 {
            g.push(f);
        }
    }
}

impl Drop for SharedFrame {
    fn drop(&mut self) {
        if let Some(pool) = self.pool.take() {
            let f = std::mem::replace(self.inner.get_mut(), Frame::empty());
            pool.give(f);
        }
    }
}

// SAFETY: see the type documentation — access is partitioned by rows and
// synchronised through `progress`.
unsafe impl Sync for SharedFrame {}
unsafe impl Send for SharedFrame {}

impl SharedFrame {
    /// Wrap a fresh frame.
    pub fn new(frame: Frame, poc: i32, id: u64, complete: bool) -> Self {
        SharedFrame { inner: std::cell::UnsafeCell::new(frame), progress: if complete { Progress::complete() } else { Progress::new() }, poc, id, pool: None }
    }

    /// Wrap a frame whose buffers return to `pool` on drop.
    pub fn with_pool(frame: Frame, poc: i32, id: u64, pool: FramePool) -> Self {
        SharedFrame { inner: std::cell::UnsafeCell::new(frame), progress: Progress::new(), poc, id, pool: Some(pool) }
    }

    /// Shared view; only rows the progress covers may be read.
    ///
    /// # Safety
    /// The caller must not touch rows the writer has not published.
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn get(&self) -> &Frame {
        unsafe { &*self.inner.get() }
    }

    /// The writer's view.
    ///
    /// # Safety
    /// Only the thread decoding this picture may call this, and only one
    /// such reference may exist at a time.
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn get_mut(&self) -> &mut Frame {
        unsafe { &mut *self.inner.get() }
    }

    /// The frame once complete (waits).
    pub fn wait_and_get(&self) -> &Frame {
        self.progress.wait_complete();
        // SAFETY: complete — no writer remains.
        unsafe { self.get() }
    }
}
