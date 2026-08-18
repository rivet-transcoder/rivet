//! HEVC picture buffers: sample planes with a replicated border — `u8` for
//! 8-bit streams, `u16` for 9–12-bit — and the per-picture side data later
//! pictures and the loop filters read. Everything below the NAL layer is
//! generic over the sample type ([`Sample`]), so the 8-bit decode moves half
//! the bytes and the SIMD kernels work on twice the lanes.

use crate::picture::{ChromaFormat, Picture, Plane};
use crate::threading::Progress;

/// A picture sample: `u8` (8-bit streams) or `u16` (up to 12-bit).
pub trait Sample: Copy + Default + Send + Sync + PartialEq + Eq + std::fmt::Debug + 'static {
    /// Bytes per sample.
    const BYTES: usize;
    /// Widen.
    fn to_i32(self) -> i32;
    /// Narrow (the value must be in range).
    fn from_i32(v: i32) -> Self;
    /// Fill the SIMD entries of the kernel table for this sample type.
    fn install_simd(dsp: &mut crate::dsp::hevc::HevcDsp<Self>, cpu: crate::dsp::Cpu);
}

impl Sample for u8 {
    const BYTES: usize = 1;
    #[inline(always)]
    fn to_i32(self) -> i32 {
        self as i32
    }
    #[inline(always)]
    fn from_i32(v: i32) -> Self {
        v as u8
    }
    fn install_simd(dsp: &mut crate::dsp::hevc::HevcDsp<Self>, cpu: crate::dsp::Cpu) {
        crate::dsp::hevc::install_simd_u8(dsp, cpu);
    }
}

impl Sample for u16 {
    const BYTES: usize = 2;
    #[inline(always)]
    fn to_i32(self) -> i32 {
        self as i32
    }
    #[inline(always)]
    fn from_i32(v: i32) -> Self {
        v as u16
    }
    fn install_simd(dsp: &mut crate::dsp::hevc::HevcDsp<Self>, cpu: crate::dsp::Cpu) {
        crate::dsp::hevc::install_simd_u16(dsp, cpu);
    }
}

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

/// Motion of one 4x4 block: both lists. Sixteen bytes — a picture's motion
/// is written per 4x4 block and read by the deblocking filter and by later
/// pictures' TMVP, so it is kept small: the referenced picture is recorded
/// as its POC distance from this picture (`DiffPicOrderCnt(cur, ref)`,
/// which the standard bounds to 16 bits) rather than its POC.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct MotionInfo {
    /// Vectors per list.
    pub mv: [Mv; 2],
    /// `POC(this picture) - POC(reference)` per list (0 when unused).
    pub ref_delta: [i16; 2],
    /// Reference index per list (-1 = list unused).
    pub ref_idx: [i8; 2],
    /// Bit 0 / 1: the list-0 / list-1 reference is long-term; bit 2: intra
    /// (no motion — TMVP treats it as unavailable).
    pub flags: u8,
    /// Unused.
    pub pad: u8,
}

const _: () = assert!(std::mem::size_of::<MotionInfo>() == 16);

impl Default for MotionInfo {
    fn default() -> Self {
        MotionInfo::INTRA
    }
}

impl MotionInfo {
    /// An intra block.
    pub const INTRA: MotionInfo = MotionInfo { mv: [Mv::ZERO; 2], ref_delta: [0; 2], ref_idx: [-1; 2], flags: 4, pad: 0 };

    /// `predFlagLX`.
    #[inline]
    pub fn uses(&self, list: usize) -> bool {
        self.ref_idx[list] >= 0
    }

    /// Intra (no motion).
    #[inline]
    pub fn intra(&self) -> bool {
        self.flags & 4 != 0
    }

    /// Whether the list's reference is a long-term picture.
    #[inline]
    pub fn long_term(&self, list: usize) -> bool {
        self.flags & (1 << list) != 0
    }

    /// The reference's POC, given this picture's.
    #[inline]
    pub fn ref_poc(&self, list: usize, cur_poc: i32) -> i32 {
        cur_poc - self.ref_delta[list] as i32
    }
}

/// One plane of samples with a border.
#[derive(Debug, Clone)]
pub struct Plane16<S: Sample = u16> {
    /// Samples.
    pub data: Vec<S>,
    /// Visible width.
    pub width: usize,
    /// Visible height.
    pub height: usize,
    /// Border.
    pub pad: usize,
    /// Row stride.
    pub stride: usize,
}

impl<S: Sample> Plane16<S> {
    fn new(width: usize, height: usize, pad: usize) -> Self {
        let stride = width + 2 * pad;
        Plane16 { data: vec![S::default(); stride * (height + 2 * pad)], width, height, pad, stride }
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
    pub fn at_clamped(&self, x: i32, y: i32) -> S {
        let xx = x.clamp(0, self.width as i32 - 1) as isize;
        let yy = y.clamp(0, self.height as i32 - 1) as isize;
        self.data[self.offset(xx, yy)]
    }
    /// Sample at (x, y) — the coordinates must be inside the padded area.
    #[inline(always)]
    pub fn at(&self, x: isize, y: isize) -> S {
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
pub struct Frame<S: Sample = u16> {
    /// Luma.
    pub y: Plane16<S>,
    /// Cb.
    pub cb: Plane16<S>,
    /// Cr.
    pub cr: Plane16<S>,
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

impl<S: Sample> Frame<S> {
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
        let mut plane = |p: &Plane16<S>, x0: usize, y0: usize, w: usize, h: usize| {
            let bps = S::BYTES;
            // A zeroed allocation: for a picture-sized buffer that is fresh
            // pages from the OS (already zero, no memset), and glibc keeps
            // recycling the mapping — a plain `with_capacity` + fill was
            // measured to take four times the page faults.
            let mut data = vec![0u8; w * h * bps];
            for yy in 0..h {
                let off = p.offset(x0 as isize, (y0 + yy) as isize);
                let src = &p.data[off..off + w];
                let dst = &mut data[yy * w * bps..(yy + 1) * w * bps];
                if bps == 1 || cfg!(target_endian = "little") {
                    // Bytes, or little-endian u16 already in memory order.
                    // SAFETY: `src` is `w` samples of `bps` bytes; `dst` is `bps * w` bytes.
                    unsafe { std::ptr::copy_nonoverlapping(src.as_ptr() as *const u8, dst.as_mut_ptr(), bps * w) };
                } else {
                    for (d, s) in dst.chunks_exact_mut(2).zip(src) {
                        d.copy_from_slice(&(s.to_i32() as u16).to_le_bytes());
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
pub struct SharedFrame<S: Sample = u16> {
    inner: std::cell::UnsafeCell<Frame<S>>,
    /// Row progress.
    pub progress: Progress,
    /// POC (fixed at creation).
    pub poc: i32,
    /// Unique id.
    pub id: u64,
    /// Where the buffers go back to when this picture is dropped.
    pool: Option<FramePool<S>>,
}

/// Recycled frame buffers: allocation and zeroing of a picture's planes cost
/// as much as decoding a few CTB rows, so pictures leaving the DPB hand
/// their buffers to the next picture of the same geometry.
pub struct FramePool<S: Sample = u16>(std::sync::Arc<std::sync::Mutex<Vec<Frame<S>>>>);

impl<S: Sample> Clone for FramePool<S> {
    fn clone(&self) -> Self {
        FramePool(self.0.clone())
    }
}

impl<S: Sample> Default for FramePool<S> {
    fn default() -> Self {
        FramePool(std::sync::Arc::new(std::sync::Mutex::new(Vec::new())))
    }
}

impl<S: Sample> FramePool<S> {
    /// Empty pool.
    pub fn new() -> Self {
        Self::default()
    }

    /// A frame of the given geometry, recycled if one is available (its
    /// samples are stale — every sample gets written before it is read).
    pub fn take(&self, width: usize, height: usize, chroma: ChromaFormat, bit_depth: u32) -> Frame<S> {
        let mut g = self.0.lock().unwrap();
        if let Some(i) = g.iter().position(|f| f.width == width && f.height == height && f.chroma == chroma && f.bit_depth == bit_depth) {
            let mut f = g.swap_remove(i);
            // Motion is rewritten for every coded block; stale values only
            // remain under lost slices, where they are as good as anything.
            f.poc = 0;
            return f;
        }
        drop(g);
        Frame::new(width, height, chroma, bit_depth)
    }

    /// Return a frame.
    pub fn give(&self, f: Frame<S>) {
        if f.width == 0 {
            return;
        }
        let mut g = self.0.lock().unwrap();
        if g.len() < 32 {
            g.push(f);
        }
    }
}

impl<S: Sample> Drop for SharedFrame<S> {
    fn drop(&mut self) {
        if let Some(pool) = self.pool.take() {
            let f = std::mem::replace(self.inner.get_mut(), Frame::empty());
            pool.give(f);
        }
    }
}

// SAFETY: see the type documentation — access is partitioned by rows and
// synchronised through `progress`.
unsafe impl<S: Sample> Sync for SharedFrame<S> {}
unsafe impl<S: Sample> Send for SharedFrame<S> {}

impl<S: Sample> SharedFrame<S> {
    /// Wrap a fresh frame.
    pub fn new(frame: Frame<S>, poc: i32, id: u64, complete: bool) -> Self {
        SharedFrame { inner: std::cell::UnsafeCell::new(frame), progress: if complete { Progress::complete() } else { Progress::new() }, poc, id, pool: None }
    }

    /// Wrap a frame whose buffers return to `pool` on drop.
    pub fn with_pool(frame: Frame<S>, poc: i32, id: u64, pool: FramePool<S>) -> Self {
        SharedFrame { inner: std::cell::UnsafeCell::new(frame), progress: Progress::new(), poc, id, pool: Some(pool) }
    }

    /// Shared view; only rows the progress covers may be read.
    ///
    /// # Safety
    /// The caller must not touch rows the writer has not published.
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn get(&self) -> &Frame<S> {
        unsafe { &*self.inner.get() }
    }

    /// The writer's view.
    ///
    /// # Safety
    /// Only the thread decoding this picture may call this, and only one
    /// such reference may exist at a time.
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn get_mut(&self) -> &mut Frame<S> {
        unsafe { &mut *self.inner.get() }
    }

    /// The frame once complete (waits).
    pub fn wait_and_get(&self) -> &Frame<S> {
        self.progress.wait_complete();
        // SAFETY: complete — no writer remains.
        unsafe { self.get() }
    }
}
