//! Macroblock-level definitions shared by the CAVLC and CABAC parsers and
//! the reconstruction: macroblock types and partitions, the per-picture
//! macroblock info arrays, neighbour derivation (6.4.11), motion vector
//! prediction (8.4.1) including P_Skip and the B direct modes.

use crate::sample::Sample;
use super::frame::{BlockMotion, Frame, Mv};
use super::slice::SliceType;
use super::tables::{BLK4X4_X, BLK4X4_Y};

/// Macroblock prediction kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MbKind {
    /// Intra 4x4 (`I_NxN` with `transform_size_8x8_flag` 0).
    I4x4,
    /// Intra 8x8 (`I_NxN` with `transform_size_8x8_flag` 1).
    I8x8,
    /// Intra 16x16.
    I16x16,
    /// Raw samples.
    IPcm,
    /// One 16x16 inter partition (P_L0_16x16 or a B 16x16 with any direction).
    Inter16x16,
    /// Two 16x8 partitions.
    Inter16x8,
    /// Two 8x16 partitions.
    Inter8x16,
    /// Four 8x8 partitions with sub-macroblock types.
    Inter8x8,
    /// P_Skip.
    PSkip,
    /// B_Skip.
    BSkip,
    /// B_Direct_16x16.
    BDirect16x16,
}

impl MbKind {
    /// Intra?
    #[inline]
    pub fn is_intra(self) -> bool {
        matches!(self, MbKind::I4x4 | MbKind::I8x8 | MbKind::I16x16 | MbKind::IPcm)
    }
    /// Skipped (no residual, inferred motion)?
    #[inline]
    pub fn is_skip(self) -> bool {
        matches!(self, MbKind::PSkip | MbKind::BSkip)
    }
    /// Uses whole-macroblock direct prediction (B_Skip / B_Direct_16x16)?
    #[inline]
    pub fn is_direct16x16(self) -> bool {
        matches!(self, MbKind::BSkip | MbKind::BDirect16x16)
    }
}

/// Sub-macroblock partitioning of an 8x8 partition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubMbShape {
    /// One 8x8.
    S8x8,
    /// Two 8x4.
    S8x4,
    /// Two 4x8.
    S4x8,
    /// Four 4x4.
    S4x4,
    /// B_Direct_8x8.
    Direct,
}

impl SubMbShape {
    /// Number of sub-partitions.
    pub fn count(self) -> usize {
        match self {
            SubMbShape::S8x8 | SubMbShape::Direct => 1,
            SubMbShape::S8x4 | SubMbShape::S4x8 => 2,
            SubMbShape::S4x4 => 4,
        }
    }
    /// Width and height in samples of each sub-partition.
    pub fn size(self) -> (usize, usize) {
        match self {
            SubMbShape::S8x8 | SubMbShape::Direct => (8, 8),
            SubMbShape::S8x4 => (8, 4),
            SubMbShape::S4x8 => (4, 8),
            SubMbShape::S4x4 => (4, 4),
        }
    }
}

/// Prediction direction flags of a partition: bit 0 = list 0, bit 1 = list 1.
pub const PRED_L0: u8 = 1;
/// See [`PRED_L0`].
pub const PRED_L1: u8 = 2;
/// See [`PRED_L0`].
pub const PRED_BI: u8 = 3;

/// One 4x4 block's worth of parsed motion syntax.
#[derive(Debug, Clone, Copy, Default)]
pub struct MvdEntry {
    /// `mvd_lX` for the block (zero for blocks that did not carry one).
    pub mvd: [Mv; 2],
}

/// The parsed macroblock (`macroblock_layer()`), before reconstruction.
#[derive(Debug, Clone)]
pub struct MbLayer {
    /// The kind.
    pub kind: MbKind,
    /// `transform_size_8x8_flag`.
    pub transform_8x8: bool,
    /// Intra 16x16 prediction mode (0..=3).
    pub intra16_mode: u8,
    /// Intra 4x4 (or 8x8, replicated over its four 4x4s) prediction modes in
    /// **raster** 4x4 order within the macroblock.
    pub intra_modes: [u8; 16],
    /// `intra_chroma_pred_mode`.
    pub chroma_mode: u8,
    /// `coded_block_pattern`: bits 0..3 luma 8x8 blocks, bits 4..5 chroma.
    pub cbp: u8,
    /// Per 8x8 partition: prediction direction flags (`PRED_*`).
    pub pred_dir: [u8; 4],
    /// Per 8x8 partition: sub-macroblock shape (for `Inter8x8`).
    pub sub_shape: [SubMbShape; 4],
    /// Per list, per 8x8 partition: reference index (-1 = none).
    pub ref_idx: [[i8; 4]; 2],
    /// Per 4x4 block (raster): the mvds carried.
    pub mvd: [MvdEntry; 16],
    /// `mb_qp_delta`.
    pub qp_delta: i32,
    /// Luma coefficients: 16 blocks of 16 (4x4 mode, `blk_raster * 16`, each
    /// in raster order within the block) or 4 blocks of 64 (8x8 mode,
    /// `blk8 * 64`, raster within the block). Intra 16x16 AC blocks keep
    /// position 0 free (the DC lives in `luma_dc`).
    pub luma: [i32; 256],
    /// Intra 16x16 DC coefficients (raster 4x4 over the macroblock).
    pub luma_dc: [i32; 16],
    /// Chroma DC (Cb, Cr): 4 each for 4:2:0 (2x2 raster), 8 for 4:2:2 (4x2
    /// raster, rows of two).
    pub chroma_dc: [[i32; 8]; 2],
    /// Chroma AC: [component][block raster (2 columns; 2 or 4 rows)][raster
    /// within block, 0 unused].
    pub chroma_ac: [[[i32; 16]; 8]; 2],
    /// Which luma 4x4 blocks (raster) have any nonzero coefficient
    /// (deblocking bS 2, CAVLC nC, CABAC cbf).
    pub luma_nz: [u8; 16],
    /// Chroma AC nonzero counts: [component][block raster] (4 or 8 blocks).
    pub chroma_nz: [[u8; 8]; 2],
    /// Coded-block flags of the DC blocks: bit 0 luma DC, bit 1 Cb DC, bit 2 Cr DC (CABAC).
    pub dc_cbf: u8,
    /// PCM samples: 256 luma then the Cb then the Cr samples (64 each for
    /// 4:2:0, 128 for 4:2:2).
    pub pcm: Vec<u16>,
}

impl MbLayer {
    /// A blank macroblock of `kind`.
    pub fn new(kind: MbKind) -> Self {
        MbLayer {
            kind,
            transform_8x8: false,
            intra16_mode: 0,
            intra_modes: [2; 16],
            chroma_mode: 0,
            cbp: 0,
            pred_dir: [0; 4],
            sub_shape: [SubMbShape::S8x8; 4],
            ref_idx: [[-1; 4]; 2],
            mvd: [MvdEntry::default(); 16],
            qp_delta: 0,
            luma: [0; 256],
            luma_dc: [0; 16],
            chroma_dc: [[0; 8]; 2],
            chroma_ac: [[[0; 16]; 8]; 2],
            luma_nz: [0; 16],
            chroma_nz: [[0; 8]; 2],
            dc_cbf: 0,
            pcm: Vec::new(),
        }
    }
    /// Make this a blank macroblock of `kind` again, cheaply.
    ///
    /// The layer is ~1.7 KiB and is reused across a slice: zeroing all of it
    /// per macroblock was a measurable share of the decode. The parsers write
    /// only nonzero coefficients into zeroed blocks and record which blocks
    /// they touched (`luma_nz`, `chroma_nz`, the kind), so only those blocks
    /// are cleared here and every untouched block is still zero.
    pub fn reset(&mut self, kind: MbKind, cabac: bool) {
        if self.transform_8x8 || self.kind == MbKind::I8x8 {
            for blk8 in 0..4 {
                let (bx, by) = ((blk8 & 1) * 2, (blk8 >> 1) * 2);
                if self.luma_nz[by * 4 + bx] != 0
                    || self.luma_nz[by * 4 + bx + 1] != 0
                    || self.luma_nz[(by + 1) * 4 + bx] != 0
                    || self.luma_nz[(by + 1) * 4 + bx + 1] != 0
                {
                    self.luma[blk8 * 64..blk8 * 64 + 64].fill(0);
                }
            }
        } else {
            for r in 0..16 {
                if self.luma_nz[r] != 0 {
                    self.luma[r * 16..r * 16 + 16].fill(0);
                }
            }
        }
        if self.kind == MbKind::I16x16 {
            self.luma_dc = [0; 16];
        }
        self.chroma_dc = [[0; 8]; 2];
        for comp in 0..2 {
            for b in 0..8 {
                if self.chroma_nz[comp][b] != 0 {
                    self.chroma_ac[comp][b] = [0; 16];
                }
            }
        }
        self.kind = kind;
        self.transform_8x8 = false;
        self.intra16_mode = 0;
        // `intra_modes` is written for every block of the kinds that read
        // it (I4x4: all sixteen; I8x8: all four quads); no reset needed.
        self.chroma_mode = 0;
        self.cbp = 0;
        self.pred_dir = [0; 4];
        self.sub_shape = [SubMbShape::S8x8; 4];
        self.ref_idx = [[-1; 4]; 2];
        // Only CABAC reads a macroblock's mvds (its neighbours' contexts).
        if cabac {
            self.mvd = [MvdEntry::default(); 16];
        }
        self.qp_delta = 0;
        self.luma_nz = [0; 16];
        self.chroma_nz = [[0; 8]; 2];
        self.dc_cbf = 0;
    }

    /// Whether the macroblock carries any luma residual (cbp luma bits) — the
    /// condition under which `mb_qp_delta` and residual are present, together
    /// with the chroma bits and I16x16.
    pub fn has_residual(&self) -> bool {
        self.cbp != 0 || self.kind == MbKind::I16x16
    }
}

/// What the picture remembers about each decoded macroblock, for its
/// neighbours and for the deblocking filter.
#[derive(Debug, Clone, Copy)]
pub struct MbInfo {
    /// Kind (`IPcm`, skip and intra matter to neighbours).
    pub kind: MbKind,
    /// The slice the macroblock belongs to (index in decode order).
    pub slice: u16,
    /// Decoded (available as a neighbour)?
    pub decoded: bool,
    /// `QP_Y`.
    pub qp: i8,
    /// `QP_C` for Cb and Cr (from `qp` and the PPS offsets).
    pub qpc: [i8; 2],
    /// `coded_block_pattern`.
    pub cbp: u8,
    /// `transform_size_8x8_flag`.
    pub transform_8x8: bool,
    /// `intra_chroma_pred_mode` (CABAC context).
    pub chroma_mode: u8,
    /// `mb_qp_delta != 0` (CABAC context for the next macroblock).
    pub qp_delta_nonzero: bool,
    /// Coded-block flags of the DC blocks (CABAC): bit 0 luma DC (I16x16),
    /// bit 1 Cb DC, bit 2 Cr DC.
    pub dc_cbf: u8,
    /// For `Inter8x8`: which 8x8 partitions are B_Direct_8x8 (bitmask).
    pub sub_direct: u8,
}

impl Default for MbInfo {
    fn default() -> Self {
        MbInfo {
            kind: MbKind::PSkip,
            slice: 0,
            decoded: false,
            qp: 0,
            qpc: [0; 2],
            cbp: 0,
            transform_8x8: false,
            chroma_mode: 0,
            qp_delta_nonzero: false,
            dc_cbf: 0,
            sub_direct: 0,
        }
    }
}

/// The per-picture arrays of macroblock and block information.
pub struct PicInfo {
    /// Width in macroblocks.
    pub mb_width: usize,
    /// Height in macroblocks.
    pub mb_height: usize,
    /// Per macroblock.
    pub mbs: Vec<MbInfo>,
    /// Per 4x4 luma block (raster within MB): nonzero coefficient count
    /// (`TotalCoeff` for CAVLC nC; `!= 0` is the CABAC coded_block_flag).
    pub luma_nz: Vec<u8>,
    /// Per 4x4 chroma block: [Cb blocks 0..8 then Cr blocks 0..8] per MB
    /// (four blocks per component in 4:2:0, eight in 4:2:2).
    pub chroma_nz: Vec<u8>,
    /// Per 4x4 luma block (raster): intra prediction mode (2 = DC for
    /// macroblocks that are not I4x4/I8x8, which is what neighbours read).
    pub intra_modes: Vec<u8>,
    /// Per 4x4 block, per list: the mvd (CABAC contexts).
    pub mvd: [Vec<Mv>; 2],
}

/// Recycled [`PicInfo`]s: the per-picture arrays are several hundred KiB at
/// HD, and allocating them fresh per picture was page faults on every one.
#[derive(Clone, Default)]
pub struct InfoPool(std::sync::Arc<std::sync::Mutex<Vec<PicInfo>>>);

impl InfoPool {
    /// A reset info block for the geometry, recycled when one is available.
    pub fn take(&self, mb_width: usize, mb_height: usize) -> PicInfo {
        let mut g = self.0.lock().unwrap();
        let mut info = match g.iter().position(|i| i.mb_width == mb_width && i.mb_height == mb_height) {
            Some(i) => g.swap_remove(i),
            None => {
                drop(g);
                PicInfo::new(mb_width, mb_height)
            }
        };
        info.reset();
        info
    }

    /// Return one.
    pub fn give(&self, info: PicInfo) {
        let mut g = self.0.lock().unwrap();
        if g.len() < 32 {
            g.push(info);
        }
    }
}

impl PicInfo {
    /// Fresh info arrays for a picture size.
    pub fn new(mb_width: usize, mb_height: usize) -> Self {
        let n = mb_width * mb_height;
        PicInfo {
            mb_width,
            mb_height,
            mbs: vec![MbInfo::default(); n],
            luma_nz: vec![0; n * 16],
            chroma_nz: vec![0; n * 16],
            intra_modes: vec![2; n * 16],
            mvd: [vec![Mv::ZERO; n * 16], vec![Mv::ZERO; n * 16]],
        }
    }
    /// Reset for a new picture.
    pub fn reset(&mut self) {
        for m in &mut self.mbs {
            *m = MbInfo::default();
        }
        self.luma_nz.fill(0);
        self.chroma_nz.fill(0);
        self.intra_modes.fill(2);
        self.mvd[0].fill(Mv::ZERO);
        self.mvd[1].fill(Mv::ZERO);
    }
}

/// The macroblock neighbours A (left), B (above), C (above-right), D
/// (above-left) of the current macroblock, as addresses when available
/// (decoded and in the same slice).
#[derive(Debug, Clone, Copy)]
pub struct MbNeighbours {
    /// Current macroblock address.
    pub addr: usize,
    /// Left.
    pub a: Option<usize>,
    /// Above.
    pub b: Option<usize>,
    /// Above-right.
    pub c: Option<usize>,
    /// Above-left.
    pub d: Option<usize>,
}

impl MbNeighbours {
    /// Derive for `addr` (6.4.11.1) with the availability rule of 6.4.9:
    /// a macroblock is available if it is decoded and in the same slice.
    pub fn derive(info: &PicInfo, addr: usize, slice: u16) -> Self {
        let w = info.mb_width;
        let avail = |a: usize| -> Option<usize> {
            let m = &info.mbs[a];
            if m.decoded && m.slice == slice { Some(a) } else { None }
        };
        let x = addr % w;
        let a = if x > 0 { avail(addr - 1) } else { None };
        let b = if addr >= w { avail(addr - w) } else { None };
        let c = if addr >= w && x + 1 < w { avail(addr - w + 1) } else { None };
        let d = if addr >= w && x > 0 { avail(addr - w - 1) } else { None };
        MbNeighbours { addr, a, b, c, d }
    }

    /// The 4x4-block neighbour at 4x4-unit offset `(bx + dx, by + dy)` of a
    /// block in the current macroblock (6.4.12), as `(mb_addr, raster 4x4
    /// index in that macroblock)`. `dx` in `-1..=4`, `dy` in `-1..=3`.
    /// Positions inside the current macroblock come back as the current
    /// address (whether they are decoded yet is the caller's business —
    /// see [`Self::block_available`]).
    #[inline]
    pub fn block(&self, bx: i32, by: i32) -> Option<(usize, usize)> {
        if by < 0 {
            if bx < 0 {
                self.d.map(|a| (a, 15))
            } else if bx < 4 {
                self.b.map(|a| (a, (12 + bx) as usize))
            } else {
                self.c.map(|a| (a, 12))
            }
        } else if by < 4 {
            if bx < 0 {
                self.a.map(|a| (a, (by * 4 + 3) as usize))
            } else if bx < 4 {
                Some((self.addr, (by * 4 + bx) as usize))
            } else {
                None
            }
        } else {
            None
        }
    }
}

/// Availability of a 4x4 block at raster `(bx, by)` inside the *current*
/// macroblock given which blocks have already been decoded (`done` is a
/// bitmask over the 16 raster positions).
#[inline]
pub fn block_available(done: u16, bx: i32, by: i32) -> bool {
    (0..4).contains(&bx) && (0..4).contains(&by) && (done >> (by * 4 + bx)) & 1 != 0
}

// ---------------------------------------------------------------------------
// Motion vector prediction (8.4.1.3)
// ---------------------------------------------------------------------------

/// A neighbour's motion for one list, for prediction.
#[derive(Debug, Clone, Copy)]
pub struct NbMotion {
    /// Available (partition exists and is decoded)?
    pub avail: bool,
    /// Reference index (-1 when unavailable or intra / not using the list).
    pub ref_idx: i8,
    /// Motion vector (zero when unavailable).
    pub mv: Mv,
}

impl NbMotion {
    /// The unavailable neighbour.
    pub const NONE: NbMotion = NbMotion { avail: false, ref_idx: -1, mv: Mv::ZERO };
}

/// Read the motion of the 4x4 block at 4x4-unit offset `(bx, by)` relative
/// to the current macroblock, for `list`. Blocks in the current macroblock
/// must be in `done` to count as available.
pub fn neighbour_motion<S: Sample>(
    nb: &MbNeighbours,
    frame: &Frame<S>,
    info: &PicInfo,
    done: u16,
    list: usize,
    bx: i32,
    by: i32,
) -> NbMotion {
    let Some((addr, blk)) = nb.block(bx, by) else { return NbMotion::NONE };
    if addr == nb.addr && !block_available(done, bx, by) {
        return NbMotion::NONE;
    }
    if info.mbs[addr].kind.is_intra() && addr != nb.addr {
        // Intra neighbour: available, but "not used for inter prediction":
        // refIdx -1, mv 0.
        return NbMotion { avail: true, ref_idx: -1, mv: Mv::ZERO };
    }
    let m = frame.motion[list][addr * 16 + blk];
    NbMotion { avail: true, ref_idx: m.ref_idx, mv: m.mv }
}

/// The three neighbours used for a partition's motion vector prediction:
/// A (left of the top-left block), B (above the top-left block), C (above
/// the block right of the partition's top-right block, falling back to D
/// (above-left of the top-left block) when C is unavailable) — 8.4.1.3.2.
/// `(bx, by)` is the partition's top-left 4x4 block, `w4` its width in 4x4
/// units.
pub fn prediction_neighbours<S: Sample>(
    nb: &MbNeighbours,
    frame: &Frame<S>,
    info: &PicInfo,
    done: u16,
    list: usize,
    bx: i32,
    by: i32,
    w4: i32,
) -> (NbMotion, NbMotion, NbMotion) {
    let a = neighbour_motion(nb, frame, info, done, list, bx - 1, by);
    let b = neighbour_motion(nb, frame, info, done, list, bx, by - 1);
    let mut c = neighbour_motion(nb, frame, info, done, list, bx + w4, by - 1);
    if !c.avail {
        c = neighbour_motion(nb, frame, info, done, list, bx - 1, by - 1);
    }
    (a, b, c)
}

/// Median motion vector prediction (8.4.1.3.1) given the three neighbours,
/// with the "only A available" and "exactly one matching reference" rules.
pub fn median_mvp(mut a: NbMotion, mut b: NbMotion, mut c: NbMotion, ref_idx: i8) -> Mv {
    if !b.avail && !c.avail && a.avail {
        b = a;
        c = a;
    }
    let ma = a.ref_idx == ref_idx;
    let mb = b.ref_idx == ref_idx;
    let mc = c.ref_idx == ref_idx;
    match (ma, mb, mc) {
        (true, false, false) => a.mv,
        (false, true, false) => b.mv,
        (false, false, true) => c.mv,
        _ => Mv::new(median3(a.mv.x, b.mv.x, c.mv.x), median3(a.mv.y, b.mv.y, c.mv.y)),
    }
}

#[inline]
fn median3(a: i16, b: i16, c: i16) -> i16 {
    a.max(b).min(a.min(b).max(c))
}

/// Motion vector prediction for a partition of the given shape (8.4.1.3):
/// directional for 16x8 / 8x16, median otherwise. `part_w`/`part_h` in
/// samples, `(x, y)` the partition's top-left in samples within the MB.
#[allow(clippy::too_many_arguments)]
pub fn predict_mv<S: Sample>(
    nb: &MbNeighbours,
    frame: &Frame<S>,
    info: &PicInfo,
    done: u16,
    list: usize,
    ref_idx: i8,
    x: usize,
    y: usize,
    part_w: usize,
    part_h: usize,
) -> Mv {
    let bx = (x / 4) as i32;
    let by = (y / 4) as i32;
    let w4 = (part_w / 4) as i32;
    let (a, b, c) = prediction_neighbours(nb, frame, info, done, list, bx, by, w4);
    if part_w == 16 && part_h == 8 {
        if y == 0 {
            if b.ref_idx == ref_idx {
                return b.mv;
            }
        } else if a.ref_idx == ref_idx {
            return a.mv;
        }
    } else if part_w == 8 && part_h == 16 {
        if x == 0 {
            if a.ref_idx == ref_idx {
                return a.mv;
            }
        } else if c.ref_idx == ref_idx {
            return c.mv;
        }
    }
    median_mvp(a, b, c, ref_idx)
}

/// P_Skip motion (8.4.1.1): reference index 0 and either the zero vector or
/// the 16x16 median prediction.
pub fn p_skip_mv<S: Sample>(nb: &MbNeighbours, frame: &Frame<S>, info: &PicInfo) -> Mv {
    let a = neighbour_motion(nb, frame, info, 0, 0, -1, 0);
    let b = neighbour_motion(nb, frame, info, 0, 0, 0, -1);
    if !a.avail || !b.avail || (a.ref_idx == 0 && a.mv == Mv::ZERO) || (b.ref_idx == 0 && b.mv == Mv::ZERO) {
        return Mv::ZERO;
    }
    predict_mv(nb, frame, info, 0, 0, 0, 0, 0, 16, 16)
}

/// Write `motion` for list `list` into the 4x4 blocks of the rectangle
/// `(x, y, w, h)` (samples within the macroblock) of macroblock `addr`.
pub fn fill_motion<S: Sample>(frame: &mut Frame<S>, addr: usize, list: usize, x: usize, y: usize, w: usize, h: usize, motion: BlockMotion) {
    for by in y / 4..(y + h) / 4 {
        for bx in x / 4..(x + w) / 4 {
            frame.motion[list][addr * 16 + by * 4 + bx] = motion;
        }
    }
}

/// The reference index of the neighbour, or -1, treated the way
/// `MinPositive` needs it.
#[inline]
fn min_positive(a: i8, b: i8) -> i8 {
    if a >= 0 && b >= 0 { a.min(b) } else { a.max(b) }
}

/// Reference indices for spatial direct prediction (8.4.1.2.2): the
/// `MinPositive` over the whole-macroblock neighbours A, B, C for each list.
pub fn spatial_direct_ref_idx<S: Sample>(nb: &MbNeighbours, frame: &Frame<S>, info: &PicInfo) -> [i8; 2] {
    let mut out = [0i8; 2];
    for list in 0..2 {
        let (a, b, c) = prediction_neighbours(nb, frame, info, 0, list, 0, 0, 4);
        out[list] = min_positive(a.ref_idx, min_positive(b.ref_idx, c.ref_idx));
    }
    out
}

/// The 4x4 raster index of the colocated block for direct prediction of
/// 8x8 partition `part` / sub-partition `sub` (8.4.1.2.1): with
/// `direct_8x8_inference` the corner 4x4 of the 8x8 (0, 3, 12, 15 in raster),
/// else the actual sub-block.
pub fn colocated_block(direct_8x8_inference: bool, part: usize, sub: usize) -> usize {
    let px = (part & 1) * 2;
    let py = (part >> 1) * 2;
    if direct_8x8_inference {
        // Corner: for part 0 the top-left 4x4 (0,0); part 1 top-right (3,0);
        // part 2 bottom-left (0,3); part 3 bottom-right (3,3).
        let cx = if part & 1 == 0 { 0 } else { 3 };
        let cy = if part >> 1 == 0 { 0 } else { 3 };
        cy * 4 + cx
    } else {
        let sx = px + (sub & 1);
        let sy = py + (sub >> 1);
        sy * 4 + sx
    }
}

/// The colocated motion for direct prediction: `(mvCol, refIdxCol,
/// ref_poc, ref_long_term, list_used)`; intra colocated → refIdx -1.
pub fn colocated_motion<S: Sample>(col: &Frame<S>, addr: usize, blk: usize) -> (Mv, i8, i32, bool) {
    if col.mb_intra[addr] {
        return (Mv::ZERO, -1, i32::MIN, false);
    }
    let m0 = col.motion[0][addr * 16 + blk];
    if m0.ref_idx >= 0 {
        return (m0.mv, m0.ref_idx, m0.ref_poc, m0.ref_long_term);
    }
    let m1 = col.motion[1][addr * 16 + blk];
    (m1.mv, m1.ref_idx, m1.ref_poc, m1.ref_long_term)
}

/// The raster 4x4 index from the standard's `luma4x4BlkIdx`.
#[inline]
pub fn raster_of_blk(blk: usize) -> usize {
    (BLK4X4_Y[blk] as usize) * 4 + BLK4X4_X[blk] as usize
}

/// Slice-level facts the parsers need at every macroblock.
#[derive(Debug, Clone, Copy)]
pub struct SliceCtx {
    /// Slice type.
    pub slice_type: SliceType,
    /// Slice index (for neighbour availability).
    pub slice_num: u16,
    /// Active reference count per list.
    pub num_ref_idx: [u32; 2],
    /// `direct_spatial_mv_pred_flag`.
    pub direct_spatial: bool,
    /// `transform_8x8_mode_flag`.
    pub transform_8x8_mode: bool,
    /// `constrained_intra_pred_flag`.
    pub constrained_intra_pred: bool,
    /// `direct_8x8_inference_flag`.
    pub direct_8x8_inference: bool,
    /// Chroma format idc.
    pub chroma_format_idc: u32,
    /// Whether the slice is CABAC-coded (mvd bookkeeping is only for it).
    pub cabac: bool,
    /// Sample bit depth (luma; chroma is the same in every stream accepted).
    pub bit_depth: u32,
}
