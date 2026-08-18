//! The decoded frame buffer: three planes with a replicated border, plus the
//! per-4x4-block motion data a later picture reads back (direct mode,
//! deblocking) and the per-macroblock facts the deblocker needs.

use crate::picture::{ChromaFormat, Picture, Plane};
use crate::sample::Sample;
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

/// `ref_parity` of a [`BlockMotion`] whose reference is a frame.
pub const PARITY_FRAME: u8 = 2;
/// `ref_parity` of a [`BlockMotion`] without a reference.
pub const PARITY_NONE: u8 = 3;

/// Motion data of one 4x4 block, for one reference list.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockMotion {
    /// The vector.
    pub mv: Mv,
    /// Reference index into the slice's list, or -1.
    pub ref_idx: i8,
    /// Which picture the reference is — 0 / 1 the top / bottom field of the
    /// frame `ref_id`, [`PARITY_FRAME`] the frame itself, [`PARITY_NONE`]
    /// when `ref_idx < 0`. With `ref_id` it identifies the reference for
    /// direct mode and the deblocking filter.
    pub ref_parity: u8,
    /// The low bits of the referenced frame's id ([`SharedFrame::id`]):
    /// unique among the pictures alive together.
    pub ref_id: u16,
}

impl Default for BlockMotion {
    fn default() -> Self {
        Self { mv: Mv::ZERO, ref_idx: -1, ref_parity: PARITY_NONE, ref_id: 0 }
    }
}

impl BlockMotion {
    /// Whether two blocks reference the same picture (frame or field).
    #[inline]
    pub fn same_ref(&self, other: &BlockMotion) -> bool {
        self.ref_id == other.ref_id && self.ref_parity == other.ref_parity
    }
}

/// One plane with a border.
#[derive(Debug, Clone)]
pub struct PaddedPlane<S: Sample = u8> {
    /// Samples, `(width + 2*pad) * (height + 2*pad)`.
    pub data: Vec<S>,
    /// Visible width.
    pub width: usize,
    /// Visible height.
    pub height: usize,
    /// Border on each side.
    pub pad: usize,
    /// Row stride.
    pub stride: usize,
}

impl<S: Sample> PaddedPlane<S> {
    fn new(width: usize, height: usize, pad: usize) -> Self {
        let stride = width + 2 * pad;
        Self { data: vec![S::default(); stride * (height + 2 * pad)], width, height, pad, stride }
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
    pub fn at(&self, x: isize, y: isize) -> S {
        self.data[self.offset(x, y)]
    }
    /// Replicate the left/right edge samples of rows `y0..y1` into the border.
    pub fn extend_rows(&mut self, y0: usize, y1: usize) {
        self.extend_rows_step(y0, y1, 1);
    }

    /// [`Self::extend_rows`] for every `step`-th row from `y0`.
    pub fn extend_rows_step(&mut self, y0: usize, y1: usize, step: usize) {
        let (w, pad, stride) = (self.width, self.pad, self.stride);
        if w == 0 {
            return;
        }
        let origin = self.origin();
        let mut y = y0;
        while y < y1.min(self.height) {
            let row = origin + y * stride;
            let l = self.data[row];
            let r = self.data[row + w - 1];
            self.data[row - pad..row].fill(l);
            self.data[row + w..row + w + pad].fill(r);
            y += step;
        }
    }

    /// The top border rows of parity `parity` (border row `-k` with `k % 2
    /// == parity`) get the first row of that field.
    pub fn extend_top_parity(&mut self, parity: usize) {
        let (pad, stride) = (self.pad, self.stride);
        if self.width == 0 || self.height < 2 {
            return;
        }
        let first = self.origin() - pad;
        let src = first + parity * stride;
        for i in 1..=pad {
            if i % 2 == parity {
                self.data.copy_within(src..src + stride, first - i * stride);
            }
        }
    }

    /// The bottom border rows of parity `parity` get the last row of that
    /// field (`h` is even: border row `h - 1 + k` has parity `(1 + k) % 2`).
    pub fn extend_bottom_parity(&mut self, parity: usize) {
        let (h, pad, stride) = (self.height, self.pad, self.stride);
        if self.width == 0 || h < 2 {
            return;
        }
        let last = self.origin() + (h - 1) * stride - pad;
        let src = last - (1 - parity) * stride;
        for i in 1..=pad {
            if (1 + i) % 2 == parity {
                self.data.copy_within(src..src + stride, last + i * stride);
            }
        }
    }

    /// Replicate the (row-extended) first row upwards into the border —
    /// or, `per_field`, the first row of each field into the border rows
    /// of its parity (border row `-k` gets row `k % 2`), which is what a
    /// field read of the plane needs.
    pub fn extend_top(&mut self, per_field: bool) {
        let (pad, stride) = (self.pad, self.stride);
        if self.width == 0 || (per_field && self.height < 2) {
            return;
        }
        let first = self.origin() - pad;
        for i in 1..=pad {
            let src = if per_field { first + (i % 2) * stride } else { first };
            self.data.copy_within(src..src + stride, first - i * stride);
        }
    }

    /// Replicate the (row-extended) last row downwards into the border —
    /// or, `per_field`, the last row of each field into the rows of its
    /// parity (border row `h - 1 + k` gets row `h - 1 - (k % 2)`; `h` is
    /// even).
    pub fn extend_bottom(&mut self, per_field: bool) {
        let (h, pad, stride) = (self.height, self.pad, self.stride);
        if self.width == 0 || h == 0 || (per_field && h < 2) {
            return;
        }
        let last = self.origin() + (h - 1) * stride - pad;
        for i in 1..=pad {
            let src = if per_field { last - (i % 2) * stride } else { last };
            self.data.copy_within(src..src + stride, last + i * stride);
        }
    }

    /// Replicate the visible edge samples into the border (`per_field` as
    /// in [`Self::extend_top`]).
    pub fn extend_edges(&mut self, per_field: bool) {
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
        self.extend_top(per_field);
        self.extend_bottom(per_field);
    }
}

/// A decoded frame.
#[derive(Debug, Clone)]
pub struct Frame<S: Sample = u8> {
    /// Luma.
    pub y: PaddedPlane<S>,
    /// Cb, Cr (empty planes for monochrome).
    pub cb: PaddedPlane<S>,
    /// See `cb`.
    pub cr: PaddedPlane<S>,
    /// Chroma format.
    pub chroma: ChromaFormat,
    /// Luma bit depth (chroma has its own in the SPS; the two are equal in
    /// every stream accepted).
    pub bit_depth: u32,
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
    /// Interlaced sequence (`frame_mbs_only_flag` 0): the top / bottom
    /// borders are extended per field, so a field read of the frame sees
    /// its own field's edge samples (a frame read then has no vertical
    /// border to lean on and clamps instead — see [`super::inter`]).
    pub field_borders: bool,
    /// Decoded as two field pictures (else as a frame).
    pub field_coded: bool,
    /// Decoded as an MBAFF frame (macroblock pairs, `mb_field` per pair).
    pub mbaff: bool,
    /// Per macroblock (frame address `frame_row * mb_width + x`; a field
    /// picture's row `r` of parity `p` is frame row `2r + p`): coded as a
    /// field macroblock, i.e. its motion is in field units. All false for a
    /// progressive frame.
    pub mb_field: Vec<bool>,
    /// The POC of each field.
    pub field_poc: [i32; 2],
    /// `separate_colour_plane_flag` pictures: the Cb and Cr colour planes,
    /// each decoded as its own monochrome picture (own samples in `y`, own
    /// motion and intra flags — 8.1: three monochrome decoding processes).
    /// `chroma` is then `Monochrome` and `cb` / `cr` are empty; the picture
    /// is assembled as 4:4:4 on output.
    pub colour_planes: Option<Box<[Frame<S>; 2]>>,
}

impl<S: Sample> Frame<S> {
    /// A zero-size placeholder (no buffers).
    pub fn empty() -> Self {
        let none = || PaddedPlane { data: Vec::new(), width: 0, height: 0, pad: 0, stride: 0 };
        Frame {
            y: none(),
            cb: none(),
            cr: none(),
            chroma: ChromaFormat::Yuv420,
            bit_depth: 8,
            mb_width: 0,
            mb_height: 0,
            motion: [Vec::new(), Vec::new()],
            mb_intra: Vec::new(),
            poc: 0,
            long_term: false,
            field_borders: false,
            field_coded: false,
            mbaff: false,
            mb_field: Vec::new(),
            field_poc: [0; 2],
            colour_planes: None,
        }
    }

    /// The frame decoding colour plane `k`: this one for 0 (and for every
    /// plane of a picture without separate colour planes), else the Cb / Cr
    /// plane's own monochrome frame.
    #[inline]
    pub fn plane_frame(&self, k: usize) -> &Frame<S> {
        match (k, &self.colour_planes) {
            (1..=2, Some(p)) => &p[k - 1],
            _ => self,
        }
    }

    /// See [`Self::plane_frame`].
    #[inline]
    pub fn plane_frame_mut(&mut self, k: usize) -> &mut Frame<S> {
        if (1..=2).contains(&k) && self.colour_planes.is_some() {
            return &mut self.colour_planes.as_mut().unwrap()[k - 1];
        }
        self
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
        let pf = self.field_borders;
        if y0 == 0 {
            self.y.extend_top(pf);
            if has_chroma {
                self.cb.extend_top(pf);
                self.cr.extend_top(pf);
            }
        }
        if y1 >= h {
            self.y.extend_bottom(pf);
            if has_chroma {
                self.cb.extend_bottom(pf);
                self.cr.extend_bottom(pf);
            }
        }
        if let Some(planes) = &mut self.colour_planes {
            for f in planes.iter_mut() {
                f.extend_rows(y0, y1);
            }
        }
    }

    /// Allocate a frame for `mb_width x mb_height` macroblocks; with
    /// `separate_planes`, a monochrome frame plus two more for the Cb and
    /// Cr colour planes (`chroma` is then ignored).
    pub fn new(mb_width: usize, mb_height: usize, chroma: ChromaFormat, bit_depth: u32, separate_planes: bool) -> Self {
        if separate_planes {
            let mut f = Self::new(mb_width, mb_height, ChromaFormat::Monochrome, bit_depth, false);
            f.colour_planes = Some(Box::new([
                Self::new(mb_width, mb_height, ChromaFormat::Monochrome, bit_depth, false),
                Self::new(mb_width, mb_height, ChromaFormat::Monochrome, bit_depth, false),
            ]));
            return f;
        }
        let w = mb_width * 16;
        let h = mb_height * 16;
        let (cw, ch) = match chroma {
            ChromaFormat::Monochrome => (0, 0),
            ChromaFormat::Yuv420 => (w / 2, h / 2),
            ChromaFormat::Yuv422 => (w / 2, h),
            ChromaFormat::Yuv444 => (w, h),
        };
        let n = mb_width * mb_height;
        // 4:4:4 chroma is interpolated with the luma six-tap filter, so it
        // needs the luma border.
        let cpad = if chroma == ChromaFormat::Yuv444 { LUMA_PAD } else { CHROMA_PAD };
        Self {
            y: PaddedPlane::new(w, h, LUMA_PAD),
            cb: PaddedPlane::new(cw, ch, cpad),
            cr: PaddedPlane::new(cw, ch, cpad),
            chroma,
            bit_depth,
            mb_width,
            mb_height,
            motion: [vec![BlockMotion::default(); n * 16], vec![BlockMotion::default(); n * 16]],
            mb_intra: vec![false; n],
            poc: 0,
            long_term: false,
            field_borders: false,
            field_coded: false,
            mbaff: false,
            mb_field: vec![false; n],
            field_poc: [0; 2],
            colour_planes: None,
        }
    }

    /// Extend the borders of the rows of field parity `parity` among luma
    /// rows `y0..y1` (frame rows), and that field's top / bottom border
    /// once the range reaches the picture's top / bottom — what a field
    /// picture publishes, its rows interleaved with the other field's.
    pub fn extend_rows_parity(&mut self, y0: usize, y1: usize, parity: usize) {
        let h = self.mb_height * 16;
        let y1 = y1.min(h);
        let first = y0 + ((y0 % 2) != parity) as usize;
        self.y.extend_rows_step(first, y1, 2);
        let has_chroma = self.chroma != ChromaFormat::Monochrome;
        if has_chroma {
            let (_, sh) = self.chroma.subsampling();
            let sh = sh as usize;
            let (cy0, cy1) = (y0 / sh, y1.div_ceil(sh));
            let cfirst = cy0 + ((cy0 % 2) != parity) as usize;
            self.cb.extend_rows_step(cfirst, cy1, 2);
            self.cr.extend_rows_step(cfirst, cy1, 2);
        }
        if y0 == 0 {
            self.y.extend_top_parity(parity);
            if has_chroma {
                self.cb.extend_top_parity(parity);
                self.cr.extend_top_parity(parity);
            }
        }
        if y1 >= h {
            self.y.extend_bottom_parity(parity);
            if has_chroma {
                self.cb.extend_bottom_parity(parity);
                self.cr.extend_bottom_parity(parity);
            }
        }
        if let Some(planes) = &mut self.colour_planes {
            for f in planes.iter_mut() {
                f.extend_rows_parity(y0, y1, parity);
            }
        }
    }

    /// Copy luma rows `y0..y1` (and the matching chroma rows) of the field
    /// picture `field` (a half-height frame) into this frame's rows of
    /// parity `parity` (field row `y` is frame row `2y + parity`).
    pub fn interleave_field_rows(&mut self, field: &Frame<S>, y0: usize, y1: usize, parity: usize) {
        fn rows<S: Sample>(dst: &mut PaddedPlane<S>, src: &PaddedPlane<S>, y0: usize, y1: usize, parity: usize) {
            let w = src.width;
            if w == 0 {
                return;
            }
            for y in y0..y1.min(src.height) {
                let s = src.offset(0, y as isize);
                let d = dst.offset(0, (2 * y + parity) as isize);
                dst.data[d..d + w].copy_from_slice(&src.data[s..s + w]);
            }
        }
        rows(&mut self.y, &field.y, y0, y1, parity);
        if self.chroma != ChromaFormat::Monochrome {
            let (_, sh) = self.chroma.subsampling();
            let sh = sh as usize;
            rows(&mut self.cb, &field.cb, y0 / sh, y1.div_ceil(sh), parity);
            rows(&mut self.cr, &field.cr, y0 / sh, y1.div_ceil(sh), parity);
        }
        if let (Some(dp), Some(sp)) = (&mut self.colour_planes, &field.colour_planes) {
            for k in 0..2 {
                dp[k].interleave_field_rows(&sp[k], y0, y1, parity);
            }
        }
    }

    /// Copy the motion of field-picture macroblock row `r` of `field` into
    /// this frame's macroblock rows (frame row `2r + parity`), flagging them
    /// as field macroblocks.
    pub fn take_field_motion_row(&mut self, field: &Frame<S>, r: usize, parity: usize) {
        let mbw = self.mb_width;
        let src = r * mbw;
        let dst = (2 * r + parity) * mbw;
        for l in 0..2 {
            self.motion[l][dst * 16..(dst + mbw) * 16].copy_from_slice(&field.motion[l][src * 16..(src + mbw) * 16]);
        }
        self.mb_intra[dst..dst + mbw].copy_from_slice(&field.mb_intra[src..src + mbw]);
        self.mb_field[dst..dst + mbw].fill(true);
        if let (Some(dp), Some(sp)) = (&mut self.colour_planes, &field.colour_planes) {
            for k in 0..2 {
                dp[k].take_field_motion_row(&sp[k], r, parity);
            }
        }
    }

    /// A field never got its pair: fill the missing field's rows with the
    /// decoded field's (line doubling), so the frame is presentable.
    pub fn double_field(&mut self, present: usize) {
        fn dbl<S: Sample>(p: &mut PaddedPlane<S>, present: usize) {
            let (w, h) = (p.width, p.height);
            if w == 0 {
                return;
            }
            let mut y = 1 - present;
            while y < h {
                let src_y = if present == 0 { y - 1 } else { (y + 1).min(h - 1) };
                let s = p.offset(0, src_y as isize);
                let d = p.offset(0, y as isize);
                p.data.copy_within(s..s + w, d);
                y += 2;
            }
        }
        dbl(&mut self.y, present);
        if self.chroma != ChromaFormat::Monochrome {
            dbl(&mut self.cb, present);
            dbl(&mut self.cr, present);
        }
        if let Some(planes) = &mut self.colour_planes {
            for f in planes.iter_mut() {
                f.double_field(present);
            }
        }
        self.extend_edges();
    }

    /// Replicate edges of all planes (call once the picture is fully
    /// decoded and filtered, before it is used as a reference).
    pub fn extend_edges(&mut self) {
        let pf = self.field_borders;
        self.y.extend_edges(pf);
        if self.chroma != ChromaFormat::Monochrome {
            self.cb.extend_edges(pf);
            self.cr.extend_edges(pf);
        }
        if let Some(planes) = &mut self.colour_planes {
            for f in planes.iter_mut() {
                f.extend_edges();
            }
        }
    }

    /// Copy out the visible picture, cropped by `(left, right, top, bottom)`
    /// luma samples.
    pub fn to_picture(&self, crop: (u32, u32, u32, u32), poc: i32, decode_index: u64) -> Picture {
        let (l, r, t, b) = (crop.0 as usize, crop.1 as usize, crop.2 as usize, crop.3 as usize);
        let width = self.y.width - l - r;
        let height = self.y.height - t - b;
        let mut planes = Vec::with_capacity(3);
        let bps = S::BYTES;
        let mut plane = |p: &PaddedPlane<S>, x0: usize, y0: usize, w: usize, h: usize| {
            let mut data = vec![0u8; w * h * bps];
            for yy in 0..h {
                let off = p.offset((x0) as isize, (y0 + yy) as isize);
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
        let mut chroma = self.chroma;
        if let Some(cp) = &self.colour_planes {
            // Separate colour planes: three luma-like planes make a 4:4:4 picture.
            plane(&cp[0].y, l, t, width, height);
            plane(&cp[1].y, l, t, width, height);
            chroma = ChromaFormat::Yuv444;
        } else if self.chroma != ChromaFormat::Monochrome {
            let (sw, sh) = self.chroma.subsampling();
            let (sw, sh) = (sw as usize, sh as usize);
            plane(&self.cb, l / sw, t / sh, width / sw, height / sh);
            plane(&self.cr, l / sw, t / sh, width / sw, height / sh);
        }
        Picture { width: width as u32, height: height as u32, bit_depth: self.bit_depth, chroma, planes, poc, decode_index }
    }
}

/// A picture shared between the thread decoding it and the threads
/// decoding later pictures that reference it — see the HEVC twin
/// [`crate::hevc::frame::SharedFrame`] for the access contract (rows are
/// partitioned by [`Progress`]).
pub struct SharedFrame<S: Sample = u8> {
    inner: std::cell::UnsafeCell<Frame<S>>,
    /// Row progress per field parity, in frame luma rows: a frame picture
    /// advances both together; a field picture advances its own (a field
    /// row covers two frame rows). Readers of a frame wait on both, readers
    /// of a field on its parity.
    pub progress: [Progress; 2],
    /// POC (fixed at creation).
    pub poc: i32,
    /// Unique id.
    pub id: u64,
    pool: Option<FramePool<S>>,
}

// SAFETY: access is partitioned by rows and synchronised through `progress`.
unsafe impl<S: Sample> Sync for SharedFrame<S> {}
unsafe impl<S: Sample> Send for SharedFrame<S> {}

impl<S: Sample> SharedFrame<S> {
    /// Wrap a fresh frame.
    pub fn new(frame: Frame<S>, poc: i32, id: u64, complete: bool) -> Self {
        let mk = || if complete { Progress::complete() } else { Progress::new() };
        SharedFrame { inner: std::cell::UnsafeCell::new(frame), progress: [mk(), mk()], poc, id, pool: None }
    }

    /// Wrap a frame whose buffers return to `pool` on drop.
    pub fn with_pool(frame: Frame<S>, poc: i32, id: u64, pool: FramePool<S>) -> Self {
        SharedFrame { inner: std::cell::UnsafeCell::new(frame), progress: [Progress::new(), Progress::new()], poc, id, pool: Some(pool) }
    }

    /// The parities a picture of `parity` (0 / 1 field, [`PARITY_FRAME`]) covers.
    #[inline]
    fn parities(parity: u8) -> std::ops::Range<usize> {
        if parity == PARITY_FRAME { 0..2 } else { parity as usize..parity as usize + 1 }
    }

    /// Frame rows `< y` of the picture `parity` are reconstructed.
    pub fn set_decoded(&self, parity: u8, y: i32) {
        for p in Self::parities(parity) {
            self.progress[p].set_decoded(y);
        }
    }

    /// Frame rows `< y` of the picture `parity` are final.
    pub fn set_done(&self, parity: u8, y: i32) {
        for p in Self::parities(parity) {
            self.progress[p].set_done(y);
        }
    }

    /// The picture `parity` is finished (or abandoned).
    pub fn finish(&self, parity: u8) {
        for p in Self::parities(parity) {
            self.progress[p].finish();
        }
    }

    /// Block until frame rows `< y` of the picture `parity` are final.
    pub fn wait_done(&self, parity: u8, y: i32) {
        for p in Self::parities(parity) {
            self.progress[p].wait_done(y);
        }
    }

    /// Block until frame rows `< y` of the picture `parity` are reconstructed.
    pub fn wait_decoded(&self, parity: u8, y: i32) {
        for p in Self::parities(parity) {
            self.progress[p].wait_decoded(y);
        }
    }

    /// Both fields (the frame) finished.
    pub fn is_complete(&self) -> bool {
        self.progress[0].is_complete() && self.progress[1].is_complete()
    }

    /// The picture `parity` finished.
    pub fn is_parity_complete(&self, parity: u8) -> bool {
        Self::parities(parity).all(|p| self.progress[p].is_complete())
    }

    /// Record a decoding error.
    pub fn set_error(&self) {
        self.progress[0].error.store(true, std::sync::atomic::Ordering::Relaxed);
    }

    /// Whether decoding hit an error.
    pub fn has_error(&self) -> bool {
        self.progress[0].error.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Shared view; only rows the progress covers may be read.
    ///
    /// # Safety
    /// The caller must not touch rows the writer has not published.
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
        self.progress[0].wait_complete();
        self.progress[1].wait_complete();
        // SAFETY: complete — no writer remains.
        unsafe { self.get() }
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

/// Recycled frame buffers (see the HEVC twin).
pub struct FramePool<S: Sample = u8>(std::sync::Arc<std::sync::Mutex<Vec<Frame<S>>>>);

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

    /// A frame of the given geometry, recycled if available. Samples, motion
    /// and intra flags are stale: every macroblock that decodes writes all
    /// three before anyone reads them, and a macroblock that never decodes
    /// (a lost slice) leaves the previous picture's — no worse than zeros for
    /// the neighbours that read it, and not paid for on every picture.
    pub fn take(&self, mb_width: usize, mb_height: usize, chroma: ChromaFormat, bit_depth: u32, separate_planes: bool) -> Frame<S> {
        let mut g = self.0.lock().unwrap();
        if let Some(i) = g.iter().position(|f| {
            f.mb_width == mb_width && f.mb_height == mb_height && f.chroma == chroma && f.bit_depth == bit_depth && f.colour_planes.is_some() == separate_planes
        }) {
            let mut f = g.swap_remove(i);
            f.poc = 0;
            f.long_term = false;
            f.field_coded = false;
            f.mbaff = false;
            f.field_poc = [0; 2];
            f.mb_field.fill(false);
            return f;
        }
        drop(g);
        Frame::new(mb_width, mb_height, chroma, bit_depth, separate_planes)
    }

    /// Return a frame.
    pub fn give(&self, f: Frame<S>) {
        if f.mb_width == 0 {
            return;
        }
        let mut g = self.0.lock().unwrap();
        if g.len() < 32 {
            g.push(f);
        }
    }
}
