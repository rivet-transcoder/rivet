//! The decoded frame buffer: three planes with a replicated border, plus the
//! per-4x4-block motion data a later picture reads back (direct mode,
//! deblocking) and the per-macroblock facts the deblocker needs.

use crate::picture::{ChromaFormat, Picture, Plane};
use crate::threading::Progress;

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

    /// Replicate the (row-extended) first row upwards into the border.
    pub fn extend_top(&mut self) {
        let (pad, stride) = (self.pad, self.stride);
        if self.width == 0 {
            return;
        }
        let first = self.origin() - pad;
        for i in 1..=pad {
            self.data.copy_within(first..first + stride, first - i * stride);
        }
    }

    /// Replicate the (row-extended) last row downwards into the border.
    pub fn extend_bottom(&mut self) {
        let (h, pad, stride) = (self.height, self.pad, self.stride);
        if self.width == 0 || h == 0 {
            return;
        }
        let last = self.origin() + (h - 1) * stride - pad;
        for i in 1..=pad {
            self.data.copy_within(last..last + stride, last + i * stride);
        }
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
    /// A zero-size placeholder (no buffers).
    pub fn empty() -> Self {
        let none = || PaddedPlane { data: Vec::new(), width: 0, height: 0, pad: 0, stride: 0 };
        Frame { y: none(), cb: none(), cr: none(), chroma: ChromaFormat::Yuv420, mb_width: 0, mb_height: 0, motion: [Vec::new(), Vec::new()], mb_intra: Vec::new(), poc: 0, long_term: false }
    }

    /// Extend the borders of luma rows `y0..y1` (and the matching chroma
    /// rows); the top border once `y0 == 0`, the bottom once `y1` reaches
    /// the height.
    pub fn extend_rows(&mut self, y0: usize, y1: usize) {
        let h = self.mb_height * 16;
        let y1 = y1.min(h);
        self.y.extend_rows(y0, y1);
        let has_chroma = self.chroma != ChromaFormat::Monochrome;
        if has_chroma {
            let (_, sh) = self.chroma.subsampling();
            let sh = sh as usize;
            self.cb.extend_rows(y0 / sh, y1.div_ceil(sh));
            self.cr.extend_rows(y0 / sh, y1.div_ceil(sh));
        }
        if y0 == 0 {
            self.y.extend_top();
            if has_chroma {
                self.cb.extend_top();
                self.cr.extend_top();
            }
        }
        if y1 >= h {
            self.y.extend_bottom();
            if has_chroma {
                self.cb.extend_bottom();
                self.cr.extend_bottom();
            }
        }
    }

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

/// A picture shared between the thread decoding it and the threads
/// decoding later pictures that reference it — see the HEVC twin
/// [`crate::hevc::frame::SharedFrame`] for the access contract (rows are
/// partitioned by [`Progress`]).
pub struct SharedFrame {
    inner: std::cell::UnsafeCell<Frame>,
    /// Row progress.
    pub progress: Progress,
    /// POC (fixed at creation).
    pub poc: i32,
    /// Unique id.
    pub id: u64,
    pool: Option<FramePool>,
}

// SAFETY: access is partitioned by rows and synchronised through `progress`.
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

impl Drop for SharedFrame {
    fn drop(&mut self) {
        if let Some(pool) = self.pool.take() {
            let f = std::mem::replace(self.inner.get_mut(), Frame::empty());
            pool.give(f);
        }
    }
}

/// Recycled frame buffers (see the HEVC twin).
#[derive(Clone, Default)]
pub struct FramePool(std::sync::Arc<std::sync::Mutex<Vec<Frame>>>);

impl FramePool {
    /// Empty pool.
    pub fn new() -> Self {
        Self::default()
    }

    /// A frame of the given geometry, recycled if available. Samples, motion
    /// and intra flags are stale: every macroblock that decodes writes all
    /// three before anyone reads them, and a macroblock that never decodes
    /// (a lost slice) leaves the previous picture's — no worse than zeros for
    /// the neighbours that read it, and not paid for on every picture.
    pub fn take(&self, mb_width: usize, mb_height: usize, chroma: ChromaFormat) -> Frame {
        let mut g = self.0.lock().unwrap();
        if let Some(i) = g.iter().position(|f| f.mb_width == mb_width && f.mb_height == mb_height && f.chroma == chroma) {
            let mut f = g.swap_remove(i);
            f.poc = 0;
            f.long_term = false;
            return f;
        }
        drop(g);
        Frame::new(mb_width, mb_height, chroma)
    }

    /// Return a frame.
    pub fn give(&self, f: Frame) {
        if f.mb_width == 0 {
            return;
        }
        let mut g = self.0.lock().unwrap();
        if g.len() < 32 {
            g.push(f);
        }
    }
}
