//! Macroblock reconstruction: motion vector derivation (8.4.1), inter and
//! intra prediction, residual reconstruction (8.5), and the bookkeeping the
//! neighbours and the deblocking filter read.

use crate::picture::ChromaFormat;
use crate::sample::Sample;
use crate::{Error, Result};

use super::cavlc::{mb_partitions, part_index_of, sub_partition_rect};
use super::frame::{BlockMotion, Frame, Mv, PARITY_FRAME, SharedFrame};
use super::inter::{MbGeom, Weighting, predict_partition};
use super::intra::{IntraAvail, predict_4x4, predict_8x8, predict_16x16, predict_chroma};
use super::mb::{
    MbKind, MbLayer, MbNeighbours, PicInfo, SliceCtx, SubMbShape, block_available, colocated_block,
    colocated_motion, fill_motion, median_mvp, p_skip_mv, predict_mv, prediction_neighbours,
    spatial_direct_ref_idx,
};
use super::slice::PredWeightTable;
use super::tables::{BLK4X4_FROM_RASTER, CHROMA_QP};
use super::transform::{
    Dequant, chroma_dc_transform_420, chroma_dc_transform_422, luma_dc_transform,
};
use crate::dsp::h264::{H264Dsp, NO_DC};

/// The reference pictures of a slice, resolved to frames.
pub struct SliceRefs<'a, S: Sample = u8> {
    /// Per list, per index: the frame (a grey stand-in for a missing one).
    pub frames: [Vec<&'a Frame<S>>; 2],
    /// The same references with their progress, for waiting on rows still
    /// being decoded by another thread.
    pub shared: [Vec<&'a SharedFrame<S>>; 2],
    /// The colocated picture's progress.
    pub col_shared: Option<&'a SharedFrame<S>>,
    /// Which picture of the colocated frame RefPicList1[0] is (0 / 1 field,
    /// [`super::frame::PARITY_FRAME`]).
    pub col_parity: u8,
    /// Per list, per index: the picture's POC (a field's when the entry is
    /// a field).
    pub pocs: [Vec<i32>; 2],
    /// Per list, per index: long-term?
    pub long_term: [Vec<bool>; 2],
    /// Per list, per index: the referenced frame's id (low bits) and which
    /// picture of it (0 / 1 field, [`super::frame::PARITY_FRAME`]).
    pub ids: [Vec<u16>; 2],
    /// See `ids`.
    pub parity: [Vec<u8>; 2],
    /// The colocated picture (RefPicList1[0]) for direct prediction.
    pub col: Option<&'a Frame<S>>,
    /// Whether RefPicList1[0] is a long-term reference.
    pub col_long_term: bool,
    /// Explicit weights, when the slice has them.
    pub explicit: Option<&'a PredWeightTable>,
    /// Implicit bi-prediction weights `[ref0][ref1] -> (w0, w1)`, when
    /// `weighted_bipred_idc == 2`.
    pub implicit: Option<Vec<Vec<(i32, i32)>>>,
    /// POC of the current picture.
    pub cur_poc: i32,
    /// The POCs of the current frame's fields (MBAFF field macroblocks).
    pub cur_field_poc: [i32; 2],
    /// Which picture the current one is (0 / 1 field, [`super::frame::PARITY_FRAME`]).
    pub cur_parity: u8,
    /// The kernels.
    pub dsp: H264Dsp<S>,
    /// Sample bit depth (weighted-prediction offsets scale with it).
    pub bit_depth: u32,
}

impl<'a, S: Sample> SliceRefs<'a, S> {
    /// The list entry and picture parity a reference index names: the entry
    /// itself for a frame macroblock (or a field picture); for a field
    /// macroblock of an MBAFF frame, entry `ref_idx >> 1` read as the field
    /// of the macroblock's parity (even index) or the opposite one (odd).
    #[inline]
    pub fn resolve(&self, list: usize, ref_idx: i8, field_mb: bool, mb_parity: u8) -> (usize, u8) {
        if field_mb {
            let i = (ref_idx >> 1) as usize;
            let par = if ref_idx & 1 == 0 {
                mb_parity
            } else {
                1 - mb_parity
            };
            (i, par)
        } else {
            (ref_idx as usize, self.parity[list][ref_idx as usize])
        }
    }

    /// The POC of the picture `ref_idx` names (see [`Self::resolve`]).
    #[inline]
    pub fn ref_poc(&self, list: usize, ref_idx: i8, field_mb: bool, mb_parity: u8) -> i32 {
        let (i, par) = self.resolve(list, ref_idx, field_mb, mb_parity);
        if field_mb {
            self.frames[list][i].field_poc[par as usize]
        } else {
            self.pocs[list][i]
        }
    }

    fn motion(&self, list: usize, ref_idx: i8, mv: Mv) -> BlockMotion {
        BlockMotion {
            mv,
            ref_idx,
            ref_parity: self.parity[list][ref_idx as usize],
            ref_id: self.ids[list][ref_idx as usize],
        }
    }

    /// [`Self::motion`] for a field macroblock of an MBAFF frame.
    fn motion_field_mb(&self, list: usize, ref_idx: i8, mv: Mv, mb_parity: u8) -> BlockMotion {
        let (i, par) = self.resolve(list, ref_idx, true, mb_parity);
        BlockMotion {
            mv,
            ref_idx,
            ref_parity: par,
            ref_id: self.ids[list][i],
        }
    }

    /// [`Self::motion`] for whichever kind of macroblock.
    #[inline]
    fn motion_of(
        &self,
        list: usize,
        ref_idx: i8,
        mv: Mv,
        field_mb: bool,
        mb_parity: u8,
    ) -> BlockMotion {
        if field_mb {
            self.motion_field_mb(list, ref_idx, mv, mb_parity)
        } else {
            self.motion(list, ref_idx, mv)
        }
    }

    /// The weighting for a block predicted from `r0` (list 0) and/or `r1`
    /// (`field_mb` / `mb_parity`: an MBAFF field macroblock, whose indices
    /// name fields — explicit weights come from entry `r >> 1`, implicit
    /// ones from field POC distances).
    fn weighting(&self, r0: i8, r1: i8, field_mb: bool, mb_parity: u8) -> Weighting {
        if let Some(t) = self.explicit {
            let mut w = [[1i32; 2]; 3];
            let mut o = [[0i32; 2]; 3];
            let log_wd = [
                t.luma_log2_denom as i32,
                t.chroma_log2_denom as i32,
                t.chroma_log2_denom as i32,
            ];
            for (list, r) in [(0usize, r0), (1usize, r1)] {
                if r < 0 {
                    continue;
                }
                let r = if field_mb { r >> 1 } else { r };
                // Offsets are in 8-bit units in the syntax: `o << (BitDepth - 8)` (8-278).
                let e = &t.lists[list][r as usize];
                let sh = self.bit_depth - 8;
                w[0][list] = e.luma.0;
                o[0][list] = e.luma.1 << sh;
                for c in 0..2 {
                    w[1 + c][list] = e.chroma[c].0;
                    o[1 + c][list] = e.chroma[c].1 << sh;
                }
            }
            return Weighting::Weighted { log_wd, w, o };
        }
        if self.implicit.is_some() {
            if r0 >= 0 && r1 >= 0 {
                let (w0, w1) = if field_mb {
                    self.implicit_weight_field(r0, r1, mb_parity)
                } else {
                    self.implicit.as_ref().unwrap()[r0 as usize][r1 as usize]
                };
                return Weighting::Weighted {
                    log_wd: [5; 3],
                    w: [[w0, w1]; 3],
                    o: [[0; 2]; 3],
                };
            }
        }
        Weighting::Default
    }

    /// One implicit weight pair (8.4.2.3.1) from POC distances.
    fn implicit_pair(cur_poc: i32, poc0: i32, poc1: i32, long0: bool, long1: bool) -> (i32, i32) {
        let tb = (cur_poc - poc0).clamp(-128, 127);
        let td = (poc1 - poc0).clamp(-128, 127);
        if td == 0 || long0 || long1 {
            return (32, 32);
        }
        let tx = (16384 + (td / 2).abs()) / td;
        let dsf = ((tb * tx + 32) >> 6).clamp(-1024, 1023);
        let w1 = dsf >> 2;
        if !(-64..=128).contains(&w1) {
            return (32, 32);
        }
        (64 - w1, w1)
    }

    /// Implicit weights for a field macroblock of an MBAFF frame: the
    /// current field's POC against the referenced fields' (8.4.2.3.1 —
    /// currPicOrField, pic0, pic1 are fields then).
    fn implicit_weight_field(&self, r0: i8, r1: i8, mb_parity: u8) -> (i32, i32) {
        let cur = self.cur_field_poc[mb_parity as usize];
        let (i0, p0) = self.resolve(0, r0, true, mb_parity);
        let (i1, p1) = self.resolve(1, r1, true, mb_parity);
        let poc0 = self.frames[0][i0].field_poc[p0 as usize];
        let poc1 = self.frames[1][i1].field_poc[p1 as usize];
        Self::implicit_pair(
            cur,
            poc0,
            poc1,
            self.long_term[0][i0],
            self.long_term[1][i1],
        )
    }

    /// Implicit weights (8.4.2.3.1) for the whole active list pair.
    pub fn build_implicit(&mut self) {
        let n0 = self.frames[0].len();
        let n1 = self.frames[1].len();
        let mut t = vec![vec![(32i32, 32i32); n1]; n0];
        for i in 0..n0 {
            for j in 0..n1 {
                t[i][j] = Self::implicit_pair(
                    self.cur_poc,
                    self.pocs[0][i],
                    self.pocs[1][j],
                    self.long_term[0][i],
                    self.long_term[1][j],
                );
            }
        }
        self.implicit = Some(t);
    }
}

/// Per-slice quantisation state.
pub struct QpState {
    /// `QP_Y,PRED` — the previous macroblock's QP_Y in decoding order.
    pub prev_qp: i32,
    /// `chroma_qp_index_offset`, `second_chroma_qp_index_offset`.
    pub chroma_offset: [i32; 2],
}

/// `QPC` from `QPY` and the chroma QP offset (8.5.8 / Table 8-15): `qPI`
/// clips at `-QpBdOffsetC` below (a negative `qPI` maps to itself).
fn chroma_qp(qp: i32, offset: i32, qp_bd_offset: i32) -> i32 {
    let qpi = (qp + offset).clamp(-qp_bd_offset, 51);
    if qpi < 0 {
        qpi
    } else {
        CHROMA_QP[qpi as usize] as i32
    }
}

/// Whether the macroblock at `addr` may supply intra prediction samples
/// (available, and intra when constrained_intra_pred is on).
fn intra_ok(info: &PicInfo, ctx: &SliceCtx, addr: Option<usize>) -> bool {
    match addr {
        Some(a) => !ctx.constrained_intra_pred || info.mbs[a].kind.is_intra(),
        None => false,
    }
}

/// Reconstruct one parsed macroblock into the current picture and record
/// what later macroblocks need to know about it.
#[allow(clippy::too_many_arguments)]
pub fn reconstruct<S: Sample>(
    ctx: &SliceCtx,
    qps: &mut QpState,
    dq: &Dequant,
    cur: &mut Frame<S>,
    info: &mut PicInfo,
    nb: &MbNeighbours,
    layer: &MbLayer,
    refs: &SliceRefs<S>,
) -> Result<()> {
    let addr = nb.addr;
    let mbx = addr % info.mb_width;
    let mby = addr / info.mb_width;
    // Where the macroblock's samples go. A field macroblock of an MBAFF
    // frame (storage row `2 * pair_row + parity`) covers every other line
    // of its pair's 32 rows and works in field coordinates for motion
    // compensation.
    let field_mb = ctx.mbaff && layer.field;
    let geom = if field_mb {
        let (pr, parity) = (mby / 2, mby % 2);
        MbGeom {
            x: mbx * 16,
            y_pic: pr * 16,
            y_dst: pr * 32 + parity,
            yc_dst: pr * 2 * cur.chroma_mb_height() + parity,
            step: 2,
            parity: parity as u8,
        }
    } else {
        MbGeom {
            x: mbx * 16,
            y_pic: mby * 16,
            y_dst: mby * 16,
            yc_dst: mby * cur.chroma_mb_height(),
            step: 1,
            parity: refs.cur_parity,
        }
    };
    let (px, py) = (geom.x, geom.y_dst);
    let step = geom.step;
    let bit_depth = cur.bit_depth;
    let max = (1i32 << bit_depth) - 1;

    // QP (7.4.5): QPY wraps in −QpBdOffsetY..=51; the dequantiser takes
    // QP'Y = QPY + QpBdOffsetY (and QP'C likewise); the deblocking filter
    // reads the unshifted values.
    let bd_off = 6 * (ctx.bit_depth as i32 - 8);
    let qp = if layer.kind.is_skip() || layer.kind == MbKind::IPcm || !layer.has_residual() {
        qps.prev_qp
    } else {
        ((qps.prev_qp + layer.qp_delta + 52 + 2 * bd_off) % (52 + bd_off)) - bd_off
    };
    qps.prev_qp = qp;
    let deblock_qp = if layer.kind == MbKind::IPcm { 0 } else { qp };
    let qpc_raw = [
        chroma_qp(qp, qps.chroma_offset[0], bd_off),
        chroma_qp(qp, qps.chroma_offset[1], bd_off),
    ];
    let deblock_qpc = [
        chroma_qp(deblock_qp, qps.chroma_offset[0], bd_off),
        chroma_qp(deblock_qp, qps.chroma_offset[1], bd_off),
    ];
    // From here on `qp` / `qpc` are the primed values the scaling uses.
    let qp = qp + bd_off;
    let qpc = [qpc_raw[0] + bd_off, qpc_raw[1] + bd_off];

    let intra = layer.kind.is_intra();
    cur.mb_intra[addr] = intra;
    let dsp = &refs.dsp;
    // `H26X_TRACE=<mbaddr>`, read once: getenv per macroblock was measurable.
    static TRACE_MB: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    let trace_mb = *TRACE_MB.get_or_init(|| {
        std::env::var("H26X_TRACE").ok().and_then(|t| {
            if t == "all" {
                Some(usize::MAX)
            } else {
                t.parse().ok()
            }
        })
    });
    if trace_mb == Some(addr) || trace_mb == Some(usize::MAX) {
        {
            eprintln!(
                "mb {addr}: kind {:?} field {} qp {qp} cbp {:#x} t8x8 {} i16 {} modes {:?} chroma {} nz {:?} nb a{:?} b{:?} c{:?} d{:?} refs {:?} dc {:?}",
                layer.kind,
                layer.field,
                layer.cbp,
                layer.transform_8x8,
                layer.intra16_mode,
                layer.intra_modes,
                layer.chroma_mode,
                layer.nz[0],
                nb.a,
                nb.b,
                nb.c,
                nb.d,
                layer.ref_idx,
                &layer.dc[0]
            );
        }
    }

    // The planes coded luma-style: luma alone, or all three in 4:4:4 (Cb
    // and Cr then use the luma prediction modes, transforms and scaling
    // lists, at their own QP).
    let planes = if cur.chroma == ChromaFormat::Yuv444 {
        3
    } else {
        1
    };
    // TransformBypassModeFlag (7.4.5): lossless macroblocks at QP'Y 0.
    let bypass = ctx.transform_bypass && qp == 0;
    let plane_qp = [qp, qpc[0], qpc[1]];

    if intra {
        // No motion for the deblocking / direct-mode readers.
        for l in 0..2 {
            for b in 0..16 {
                cur.motion[l][addr * 16 + b] = BlockMotion::default();
            }
        }
        match layer.kind {
            MbKind::IPcm => {
                let stride = cur.y.stride * step;
                let off = cur.y.offset(px as isize, py as isize);
                for y in 0..16 {
                    for (d, &v) in cur.y.data[off + y * stride..off + y * stride + 16]
                        .iter_mut()
                        .zip(&layer.pcm[y * 16..y * 16 + 16])
                    {
                        *d = S::from_i32(v as i32);
                    }
                }
                let (mbw_c, mbh_c) = match cur.chroma {
                    ChromaFormat::Yuv420 => (8usize, 8usize),
                    ChromaFormat::Yuv422 => (8, 16),
                    ChromaFormat::Yuv444 => (16, 16),
                    ChromaFormat::Monochrome => (0, 0),
                };
                if mbw_c > 0 {
                    let cstride = cur.cb.stride * step;
                    let coff = cur
                        .cb
                        .offset((px / 16 * mbw_c) as isize, geom.yc_dst as isize);
                    let n = mbw_c * mbh_c;
                    for y in 0..mbh_c {
                        for (d, &v) in cur.cb.data[coff + y * cstride..coff + y * cstride + mbw_c]
                            .iter_mut()
                            .zip(&layer.pcm[256 + y * mbw_c..256 + (y + 1) * mbw_c])
                        {
                            *d = S::from_i32(v as i32);
                        }
                        for (d, &v) in cur.cr.data[coff + y * cstride..coff + y * cstride + mbw_c]
                            .iter_mut()
                            .zip(&layer.pcm[256 + n + y * mbw_c..256 + n + (y + 1) * mbw_c])
                        {
                            *d = S::from_i32(v as i32);
                        }
                    }
                }
            }
            MbKind::I16x16 => {
                let av = IntraAvail {
                    top: intra_ok(info, ctx, nb.b),
                    left: intra_left_ok(info, ctx, nb, 0, 16),
                    top_left: intra_ok(info, ctx, nb.d),
                    top_right: false,
                };
                for p in 0..planes {
                    let plane = plane_mut(cur, p);
                    let off = plane.offset(px as isize, py as isize);
                    let stride = plane.stride * step;
                    predict_16x16(plane, off, stride, layer.intra16_mode, av, bit_depth)?;
                    let qp = plane_qp[p];
                    let scale = &dq.scale4[p + ctx.scaling_plane][(qp % 6) as usize];
                    if bypass {
                        // The whole 16x16 residual at once: DC into position
                        // 0 of each block, then 8.5.15's DPCM for the
                        // vertical / horizontal modes.
                        let mut r = [0i32; 256];
                        for blk in 0..16 {
                            let (bx, by) = (blk % 4, blk / 4);
                            for i in 0..4 {
                                for j in 0..4 {
                                    r[(by * 4 + i) * 16 + bx * 4 + j] = if i == 0 && j == 0 {
                                        layer.dc[p][blk]
                                    } else {
                                        layer.coef[p][blk * 16 + i * 4 + j]
                                    };
                                }
                            }
                        }
                        let hor = match layer.intra16_mode {
                            0 => Some(false),
                            1 => Some(true),
                            _ => None,
                        };
                        add_bypass(&mut plane.data[off..], stride, &mut r, 16, hor, max);
                    } else {
                        // Residual: DC transform then per-4x4 blocks.
                        let mut dc = layer.dc[p];
                        luma_dc_transform(&mut dc, scale[0], qp);
                        for blk in 0..16 {
                            let (bx, by) = (blk % 4, blk / 4);
                            let boff = off + by * 4 * stride + bx * 4;
                            residual4(
                                dsp,
                                &mut plane.data[boff..],
                                stride,
                                &layer.coef[p][blk * 16..blk * 16 + 16],
                                scale,
                                qp,
                                Some(dc[blk]),
                                max,
                            );
                        }
                    }
                }
                predict_and_add_chroma(
                    dsp, cur, info, ctx, nb, layer, geom, qpc, dq, true, bypass,
                )?;
            }
            MbKind::I4x4 => {
                for p in 0..planes {
                    let plane = plane_mut(cur, p);
                    let stride = plane.stride * step;
                    let off = plane.offset(px as isize, py as isize);
                    let qp = plane_qp[p];
                    let scale = &dq.scale4[p + ctx.scaling_plane][(qp % 6) as usize];
                    for blk_idx in 0..16 {
                        let raster = super::mb::raster_of_blk(blk_idx);
                        let (bx, by) = (raster % 4, raster / 4);
                        let av = intra_avail_4x4(info, ctx, nb, bx, by);
                        let boff = off + by * 4 * stride + bx * 4;
                        let mode = layer.intra_modes[raster];
                        predict_4x4(plane, boff, stride, mode, av, bit_depth)?;
                        if layer.nz[p][raster] != 0 {
                            if bypass {
                                let mut r = [0i32; 16];
                                r.copy_from_slice(&layer.coef[p][raster * 16..raster * 16 + 16]);
                                add_bypass(
                                    &mut plane.data[boff..],
                                    stride,
                                    &mut r,
                                    4,
                                    hv_mode(mode),
                                    max,
                                );
                            } else {
                                residual4(
                                    dsp,
                                    &mut plane.data[boff..],
                                    stride,
                                    &layer.coef[p][raster * 16..raster * 16 + 16],
                                    scale,
                                    qp,
                                    None,
                                    max,
                                );
                            }
                        }
                    }
                }
                predict_and_add_chroma(
                    dsp, cur, info, ctx, nb, layer, geom, qpc, dq, true, bypass,
                )?;
            }
            MbKind::I8x8 => {
                for p in 0..planes {
                    let plane = plane_mut(cur, p);
                    let stride = plane.stride * step;
                    let off = plane.offset(px as isize, py as isize);
                    let qp = plane_qp[p];
                    let scale = &dq.scale8[2 * (p + ctx.scaling_plane)][(qp % 6) as usize];
                    for blk8 in 0..4 {
                        let (bx, by) = ((blk8 & 1) * 2, (blk8 >> 1) * 2);
                        let av = intra_avail_8x8(info, ctx, nb, blk8);
                        let boff = off + by * 4 * stride + bx * 4;
                        let mode = layer.intra_modes[by * 4 + bx];
                        predict_8x8(plane, boff, stride, mode, av, bit_depth)?;
                        if layer.nz[p][by * 4 + bx] != 0
                            || layer.nz[p][by * 4 + bx + 1] != 0
                            || layer.nz[p][(by + 1) * 4 + bx] != 0
                            || layer.nz[p][(by + 1) * 4 + bx + 1] != 0
                        {
                            if bypass {
                                let mut r = [0i32; 64];
                                r.copy_from_slice(&layer.coef[p][blk8 * 64..blk8 * 64 + 64]);
                                add_bypass(
                                    &mut plane.data[boff..],
                                    stride,
                                    &mut r,
                                    8,
                                    hv_mode(mode),
                                    max,
                                );
                            } else {
                                residual8(
                                    dsp,
                                    &mut plane.data[boff..],
                                    stride,
                                    &layer.coef[p][blk8 * 64..blk8 * 64 + 64],
                                    scale,
                                    qp,
                                    max,
                                );
                            }
                        }
                    }
                }
                predict_and_add_chroma(
                    dsp, cur, info, ctx, nb, layer, geom, qpc, dq, true, bypass,
                )?;
            }
            _ => unreachable!(),
        }
    } else {
        derive_motion_and_predict(ctx, cur, info, nb, layer, refs, geom)?;
        // Residual.
        for p in 0..planes {
            let plane = plane_mut(cur, p);
            let stride = plane.stride * step;
            let off = plane.offset(px as isize, py as isize);
            let qp = plane_qp[p];
            if layer.transform_8x8 {
                let scale = &dq.scale8[2 * (p + ctx.scaling_plane) + 1][(qp % 6) as usize];
                for blk8 in 0..4 {
                    if layer.cbp & (1 << blk8) == 0 {
                        continue;
                    }
                    let (bx, by) = ((blk8 & 1) * 2, (blk8 >> 1) * 2);
                    if layer.nz[p][by * 4 + bx] == 0
                        && layer.nz[p][by * 4 + bx + 1] == 0
                        && layer.nz[p][(by + 1) * 4 + bx] == 0
                        && layer.nz[p][(by + 1) * 4 + bx + 1] == 0
                    {
                        continue;
                    }
                    let boff = off + by * 4 * stride + bx * 4;
                    if bypass {
                        let mut r = [0i32; 64];
                        r.copy_from_slice(&layer.coef[p][blk8 * 64..blk8 * 64 + 64]);
                        add_bypass(&mut plane.data[boff..], stride, &mut r, 8, None, max);
                    } else {
                        residual8(
                            dsp,
                            &mut plane.data[boff..],
                            stride,
                            &layer.coef[p][blk8 * 64..blk8 * 64 + 64],
                            scale,
                            qp,
                            max,
                        );
                    }
                }
            } else {
                let scale = &dq.scale4[3 + p + ctx.scaling_plane][(qp % 6) as usize];
                for raster in 0..16 {
                    if layer.nz[p][raster] == 0 {
                        continue;
                    }
                    let (bx, by) = (raster % 4, raster / 4);
                    let boff = off + by * 4 * stride + bx * 4;
                    if bypass {
                        let mut r = [0i32; 16];
                        r.copy_from_slice(&layer.coef[p][raster * 16..raster * 16 + 16]);
                        add_bypass(&mut plane.data[boff..], stride, &mut r, 4, None, max);
                    } else {
                        residual4(
                            dsp,
                            &mut plane.data[boff..],
                            stride,
                            &layer.coef[p][raster * 16..raster * 16 + 16],
                            scale,
                            qp,
                            None,
                            max,
                        );
                    }
                }
            }
        }
        if matches!(cur.chroma, ChromaFormat::Yuv420 | ChromaFormat::Yuv422)
            && layer.cbp & 0x30 != 0
        {
            add_chroma_residual(dsp, cur, layer, geom, qpc, dq, false, bypass);
        }
    }

    // Bookkeeping.
    let m = &mut info.mbs[addr];
    m.kind = layer.kind;
    m.slice = ctx.slice_num;
    m.decoded = true;
    m.qp = deblock_qp as i8;
    m.qpc = [deblock_qpc[0] as i8, deblock_qpc[1] as i8];
    m.cbp = layer.cbp;
    m.transform_8x8 = layer.transform_8x8;
    m.chroma_mode = layer.chroma_mode;
    m.qp_delta_nonzero = layer.has_residual() && layer.qp_delta != 0;
    m.dc_cbf = layer.dc_cbf;
    m.sub_direct = if layer.kind == MbKind::Inter8x8 {
        (0..4)
            .map(|p| ((layer.sub_shape[p] == SubMbShape::Direct) as u8) << p)
            .sum()
    } else {
        0
    };
    m.field = field_mb || ctx.field_pic;
    if ctx.mbaff {
        cur.mb_field[addr] = field_mb;
    }
    let base = addr * 16;
    if layer.kind == MbKind::IPcm {
        info.luma_nz[base..base + 16].fill(16);
        info.chroma_nz[addr * 32..addr * 32 + 32].fill(16);
    } else {
        info.luma_nz[base..base + 16].copy_from_slice(&layer.nz[0]);
        if planes == 3 {
            info.chroma_nz[addr * 32..addr * 32 + 16].copy_from_slice(&layer.nz[1]);
            info.chroma_nz[addr * 32 + 16..addr * 32 + 32].copy_from_slice(&layer.nz[2]);
        } else {
            for comp in 0..2 {
                info.chroma_nz[addr * 32 + comp * 16..addr * 32 + comp * 16 + 8]
                    .copy_from_slice(&layer.chroma_nz[comp]);
            }
        }
    }
    if matches!(layer.kind, MbKind::I4x4 | MbKind::I8x8) {
        info.intra_modes[base..base + 16].copy_from_slice(&layer.intra_modes);
    } else {
        info.intra_modes[base..base + 16].fill(2);
    }
    // Only CABAC reads the neighbours' mvds (context selection).
    if ctx.cabac {
        for l in 0..2 {
            for b in 0..16 {
                info.mvd[l][base + b] = layer.mvd[b].mvd[l];
            }
        }
    }
    Ok(())
}

/// Availability for Intra_4x4 block `(bx, by)` (raster) of the current MB.
fn intra_avail_4x4(
    info: &PicInfo,
    ctx: &SliceCtx,
    nb: &MbNeighbours,
    bx: usize,
    by: usize,
) -> IntraAvail {
    let cur = BLK4X4_FROM_RASTER[by * 4 + bx];
    let (bx, by) = (bx as i32, by as i32);
    // A neighbouring block outside the macroblock: its macroblock, intra-ok.
    let ok = |b: Option<(usize, usize)>| b.is_some_and(|(a, _)| intra_ok(info, ctx, Some(a)));
    let left = if bx > 0 {
        true
    } else {
        intra_left_ok(info, ctx, nb, by * 4, by * 4 + 4)
    };
    let top = if by > 0 { true } else { ok(nb.block(bx, -1)) };
    // The top-left sample (bx*4 - 1, by*4 - 1): an odd row when by > 0, which
    // in an MBAFF frame may be the other macroblock of the left pair.
    let top_left = if bx > 0 && by > 0 {
        true
    } else {
        intra_ok(
            info,
            ctx,
            nb.locate(bx * 4 - 1, by * 4 - 1, 16, 16).map(|(a, _, _)| a),
        )
    };
    let top_right = if by == 0 {
        ok(nb.block(bx + 1, -1))
    } else if bx == 3 {
        false
    } else {
        BLK4X4_FROM_RASTER[((by - 1) * 4 + bx + 1) as usize] < cur
    };
    IntraAvail {
        top,
        left,
        top_left,
        top_right,
    }
}

/// Availability for Intra_8x8 block `blk8`.
fn intra_avail_8x8(info: &PicInfo, ctx: &SliceCtx, nb: &MbNeighbours, blk8: usize) -> IntraAvail {
    let (bx, by) = ((blk8 as i32 & 1) * 2, (blk8 as i32 >> 1) * 2);
    let ok = |b: Option<(usize, usize)>| b.is_some_and(|(a, _)| intra_ok(info, ctx, Some(a)));
    let left = if bx > 0 {
        true
    } else {
        intra_left_ok(info, ctx, nb, by * 4, by * 4 + 8)
    };
    let top = if by > 0 { true } else { ok(nb.block(bx, -1)) };
    let top_left = if bx > 0 && by > 0 {
        true
    } else {
        intra_ok(
            info,
            ctx,
            nb.locate(bx * 4 - 1, by * 4 - 1, 16, 16).map(|(a, _, _)| a),
        )
    };
    let top_right = if by == 0 {
        ok(nb.block(bx + 2, -1))
    } else {
        bx == 0
    };
    IntraAvail {
        top,
        left,
        top_left,
        top_right,
    }
}

/// Whether the whole left column of luma rows `y0..y1` of the current
/// macroblock may supply intra prediction samples: every macroblock it
/// borders (two in an MBAFF frame when the left pair is of the other kind)
/// is available and intra-ok.
fn intra_left_ok(info: &PicInfo, ctx: &SliceCtx, nb: &MbNeighbours, y0: i32, y1: i32) -> bool {
    let (a, b) = nb.left_mbs(y0, y1);
    intra_ok(info, ctx, a) && intra_ok(info, ctx, b)
}

/// Chroma intra prediction and residual for an intra macroblock.
#[allow(clippy::too_many_arguments)]
fn predict_and_add_chroma<S: Sample>(
    dsp: &H264Dsp<S>,
    cur: &mut Frame<S>,
    info: &PicInfo,
    ctx: &SliceCtx,
    nb: &MbNeighbours,
    layer: &MbLayer,
    geom: MbGeom,
    qpc: [i32; 2],
    dq: &Dequant,
    intra: bool,
    bypass: bool,
) -> Result<()> {
    let (mbw_c, mbh_c) = match cur.chroma {
        ChromaFormat::Yuv420 => (8usize, 8usize),
        ChromaFormat::Yuv422 => (8, 16),
        _ => return Ok(()),
    };
    let px = geom.x;
    let av = IntraAvail {
        top: intra_ok(info, ctx, nb.b),
        left: intra_left_ok(info, ctx, nb, 0, 16),
        top_left: intra_ok(info, ctx, nb.d),
        top_right: false,
    };
    // Per chroma block row (4 chroma rows = 4 * (16 / mbh_c) luma rows), the
    // availability of its left samples.
    let rows_per_blk = 4 * 16 / mbh_c;
    let mut left_rows = [false; 4];
    for (i, lr) in left_rows.iter_mut().enumerate().take(mbh_c / 4) {
        *lr = intra_left_ok(
            info,
            ctx,
            nb,
            (i * rows_per_blk) as i32,
            ((i + 1) * rows_per_blk) as i32,
        );
    }
    let coff = cur
        .cb
        .offset((px / 16 * mbw_c) as isize, geom.yc_dst as isize);
    let cstride = cur.cb.stride * geom.step;
    predict_chroma(
        &mut cur.cb,
        coff,
        cstride,
        layer.chroma_mode,
        av,
        left_rows,
        cur.bit_depth,
        mbh_c,
    )?;
    predict_chroma(
        &mut cur.cr,
        coff,
        cstride,
        layer.chroma_mode,
        av,
        left_rows,
        cur.bit_depth,
        mbh_c,
    )?;
    if layer.cbp & 0x30 != 0 {
        add_chroma_residual(dsp, cur, layer, geom, qpc, dq, intra, bypass);
    }
    Ok(())
}

/// Chroma residual (DC transform + per-block AC) added to the prediction.
#[allow(clippy::too_many_arguments)]
fn add_chroma_residual<S: Sample>(
    dsp: &H264Dsp<S>,
    cur: &mut Frame<S>,
    layer: &MbLayer,
    geom: MbGeom,
    qpc: [i32; 2],
    dq: &Dequant,
    intra: bool,
    bypass: bool,
) {
    let max = (1i32 << cur.bit_depth) - 1;
    let c422 = cur.chroma == ChromaFormat::Yuv422;
    let (mbw_c, mbh_c) = if c422 { (8usize, 16usize) } else { (8, 8) };
    let px = geom.x;
    let cstride = cur.cb.stride * geom.step;
    let coff = cur
        .cb
        .offset((px / 16 * mbw_c) as isize, geom.yc_dst as isize);
    if bypass {
        // Lossless: the levels are the residual (the DC as parsed, in
        // position 0 of its block); an intra macroblock predicted
        // horizontally / vertically accumulates it over the whole chroma
        // block (8.5.15 with nW = MbWidthC, nH = MbHeightC).
        let hor = match (intra, layer.chroma_mode) {
            (true, 1) => Some(true),
            (true, 2) => Some(false),
            _ => None,
        };
        for comp in 0..2 {
            let mut r = [0i32; 8 * 16];
            for blk in 0..(mbh_c / 4) * 2 {
                let (bx, by) = (blk % 2, blk / 2);
                for i in 0..4 {
                    for j in 0..4 {
                        r[(by * 4 + i) * 8 + bx * 4 + j] = if i == 0 && j == 0 {
                            layer.chroma_dc[comp][blk]
                        } else {
                            layer.chroma_ac[comp][blk][i * 4 + j]
                        };
                    }
                }
            }
            let plane = if comp == 0 { &mut cur.cb } else { &mut cur.cr };
            add_bypass_rect(
                &mut plane.data[coff..],
                cstride,
                &mut r[..mbw_c * mbh_c],
                mbw_c,
                mbh_c,
                hor,
                max,
            );
        }
        return;
    }
    for comp in 0..2 {
        let list = if intra { 1 + comp } else { 4 + comp };
        let qp = qpc[comp];
        let mut dc = layer.chroma_dc[comp];
        if c422 {
            // The DC scaling uses QP'c + 3, whose scale factor is the table row
            // of that QP.
            chroma_dc_transform_422(&mut dc, dq.scale4[list][((qp + 3) % 6) as usize][0], qp);
        } else {
            let mut dc4 = [dc[0], dc[1], dc[2], dc[3]];
            chroma_dc_transform_420(&mut dc4, dq.scale4[list][(qp % 6) as usize][0], qp);
            dc[..4].copy_from_slice(&dc4);
        }
        let plane = if comp == 0 { &mut cur.cb } else { &mut cur.cr };
        for blk in 0..(mbh_c / 4) * 2 {
            let (bx, by) = (blk % 2, blk / 2);
            let boff = coff + by * 4 * cstride + bx * 4;
            residual4(
                dsp,
                &mut plane.data[boff..],
                cstride,
                &layer.chroma_ac[comp][blk],
                &dq.scale4[list][(qp % 6) as usize],
                qp,
                Some(dc[blk]),
                max,
            );
        }
    }
}

/// One prediction to run: `(x, y, w, h, ref0, mv0, ref1, mv1)` in
/// macroblock coordinates.
type Job = (usize, usize, usize, usize, i8, Mv, i8, Mv);

/// The predictions of one macroblock, on the stack (at most sixteen 4x4s).
#[derive(Default)]
struct Jobs {
    items: [Job; 16],
    len: usize,
}

impl Jobs {
    #[inline(always)]
    fn push(&mut self, j: Job) {
        self.items[self.len] = j;
        self.len += 1;
    }
    #[inline(always)]
    fn as_slice(&self) -> &[Job] {
        &self.items[..self.len]
    }
}

/// Derive every partition's motion (8.4.1), store it, and motion-compensate
/// the whole macroblock.
fn derive_motion_and_predict<S: Sample>(
    ctx: &SliceCtx,
    cur: &mut Frame<S>,
    info: &PicInfo,
    nb: &MbNeighbours,
    layer: &MbLayer,
    refs: &SliceRefs<S>,
    geom: MbGeom,
) -> Result<()> {
    let addr = nb.addr;
    let py = geom.y_pic;
    // An MBAFF field macroblock: reference indices name fields of the
    // frame list (twice as many entries).
    let field_mb = geom.step == 2;
    let mb_parity = geom.parity;
    let list_mult = if field_mb { 2 } else { 1 };
    // Blocks of the current MB whose motion is final (for neighbour prediction).
    let mut done: u16 = 0;
    // The predictions to run: (x, y, w, h, ref0, mv0, ref1, mv1) in MB coords.
    let mut jobs = Jobs::default();

    match layer.kind {
        MbKind::PSkip => {
            let mv = p_skip_mv(nb, cur, info);
            fill_motion(
                cur,
                addr,
                0,
                0,
                0,
                16,
                16,
                refs.motion_of(0, 0, mv, field_mb, mb_parity),
            );
            fill_motion(cur, addr, 1, 0, 0, 16, 16, BlockMotion::default());
            jobs.push((0, 0, 16, 16, 0, mv, -1, Mv::ZERO));
        }
        MbKind::BSkip | MbKind::BDirect16x16 => {
            direct_partitions(ctx, cur, info, nb, refs, &[0, 1, 2, 3], &mut jobs, geom)?;
        }
        MbKind::Inter16x16 | MbKind::Inter16x8 | MbKind::Inter8x16 => {
            let parts = mb_partitions(layer.kind);
            for &(x, y, w, h) in parts {
                let part = part_index_of(x, y);
                let dir = layer.pred_dir[part];
                let mut mvs = [Mv::ZERO; 2];
                let mut rids = [-1i8; 2];
                for list in 0..2 {
                    if dir & (1 << list) == 0 {
                        fill_motion(cur, addr, list, x, y, w, h, BlockMotion::default());
                        continue;
                    }
                    let ri = layer.ref_idx[list][part];
                    if ri < 0 || ri as usize >= refs.frames[list].len() * list_mult {
                        return Err(Error::bitstream("ref_idx beyond the reference list"));
                    }
                    let mvp = predict_mv(nb, cur, info, done, list, ri, x, y, w, h);
                    let mvd = layer.mvd[(y / 4) * 4 + x / 4].mvd[list];
                    let mv = Mv::new(mvp.x.wrapping_add(mvd.x), mvp.y.wrapping_add(mvd.y));
                    fill_motion(
                        cur,
                        addr,
                        list,
                        x,
                        y,
                        w,
                        h,
                        refs.motion_of(list, ri, mv, field_mb, mb_parity),
                    );
                    mvs[list] = mv;
                    rids[list] = ri;
                }
                for by in y / 4..(y + h) / 4 {
                    for bx in x / 4..(x + w) / 4 {
                        done |= 1 << (by * 4 + bx);
                    }
                }
                jobs.push((x, y, w, h, rids[0], mvs[0], rids[1], mvs[1]));
            }
        }
        MbKind::Inter8x8 => {
            for part in 0..4 {
                let shape = layer.sub_shape[part];
                if shape == SubMbShape::Direct {
                    direct_partitions(ctx, cur, info, nb, refs, &[part], &mut jobs, geom)?;
                    let (x, y, w, h) = sub_partition_rect(part, shape, 0);
                    for by in y / 4..(y + h) / 4 {
                        for bx in x / 4..(x + w) / 4 {
                            done |= 1 << (by * 4 + bx);
                        }
                    }
                    continue;
                }
                let dir = layer.pred_dir[part];
                for sub in 0..shape.count() {
                    let (x, y, w, h) = sub_partition_rect(part, shape, sub);
                    let mut mvs = [Mv::ZERO; 2];
                    let mut rids = [-1i8; 2];
                    for list in 0..2 {
                        if dir & (1 << list) == 0 {
                            fill_motion(cur, addr, list, x, y, w, h, BlockMotion::default());
                            continue;
                        }
                        let ri = layer.ref_idx[list][part];
                        if ri < 0 || ri as usize >= refs.frames[list].len() * list_mult {
                            return Err(Error::bitstream("ref_idx beyond the reference list"));
                        }
                        let mvp = predict_mv(nb, cur, info, done, list, ri, x, y, w, h);
                        let mvd = layer.mvd[(y / 4) * 4 + x / 4].mvd[list];
                        let mv = Mv::new(mvp.x.wrapping_add(mvd.x), mvp.y.wrapping_add(mvd.y));
                        fill_motion(
                            cur,
                            addr,
                            list,
                            x,
                            y,
                            w,
                            h,
                            refs.motion_of(list, ri, mv, field_mb, mb_parity),
                        );
                        mvs[list] = mv;
                        rids[list] = ri;
                    }
                    for by in y / 4..(y + h) / 4 {
                        for bx in x / 4..(x + w) / 4 {
                            done |= 1 << (by * 4 + bx);
                        }
                    }
                    jobs.push((x, y, w, h, rids[0], mvs[0], rids[1], mvs[1]));
                }
            }
        }
        _ => unreachable!(),
    }

    // Motion compensation: wait for the reference rows the filters reach
    // (six-tap luma: 2 above / 3 below; bilinear chroma: 1 below, in luma
    // rows), then predict. A field macroblock of an MBAFF frame reads the
    // fields of the frame list (index >> 1, parity from index & 1) in field
    // coordinates.
    let field_mb = geom.step == 2;
    let pic_h = (cur.mb_height * 16 / if field_mb { 2 } else { 1 }) as i32;
    for &(x, y, w, h, r0, mv0, r1, mv1) in jobs.as_slice() {
        let mut fr: [Option<(&Frame<S>, Mv, u8)>; 2] = [None, None];
        for (list, ri, mv) in [(0usize, r0, mv0), (1, r1, mv1)] {
            if ri < 0 {
                continue;
            }
            let (fi, rpar) = refs.resolve(list, ri, field_mb, geom.parity);
            let yb = (py + y) as i32;
            let yi = yb + (mv.y as i32 >> 2);
            let need_l = yi + h as i32 + 3;
            let yci = (yb >> 1) + (mv.y as i32 >> 3);
            let need_c = 2 * (yci + (h as i32 >> 1) + 1);
            let need = need_l.max(need_c).clamp(1, pic_h);
            // A field reference's progress counts frame rows (two per field row).
            refs.shared[list][fi]
                .wait_done(rpar, if rpar == PARITY_FRAME { need } else { 2 * need });
            fr[list] = Some((refs.frames[list][fi], mv, rpar));
        }
        let weighting = refs.weighting(r0, r1, field_mb, geom.parity);
        predict_partition(&refs.dsp, cur, geom, x, y, w, h, fr[0], fr[1], weighting);
    }
    Ok(())
}

/// How the colocated vertical vector relates to the current picture's
/// units (Table 8-8's vertMvScale).
#[derive(Clone, Copy, PartialEq, Eq)]
enum VertScale {
    OneToOne,
    FrmToFld,
    FldToFrm,
}

/// A colocated 4x4 block: the macroblock (in the colocated frame's
/// frame-row addressing) and the 4x4 in it, and the vector scaling.
struct ColBlock {
    addr: usize,
    blk: usize,
    scale: VertScale,
}

/// For a frame picture whose RefPicList1[0] is a field-coded frame: the
/// field of it that is colPic (Table 8-6 — the one closer in POC).
fn col_field_of_frame<S: Sample>(refs: &SliceRefs<S>, col: &Frame<S>) -> u8 {
    let top = (col.field_poc[0] - refs.cur_poc).abs();
    let bottom = (col.field_poc[1] - refs.cur_poc).abs();
    if top < bottom { 0 } else { 1 }
}

/// The colocated block for direct prediction of 8x8 partition `part` /
/// sub-partition `sub` of macroblock `addr` (8.4.1.2.1, Tables 8-6 and 8-8),
/// across the frame / field combinations of the current picture and the
/// colocated frame (which stores its macroblocks by frame row, a field
/// picture's row `r` at frame rows `2r + parity`, with `mb_field` flags).
#[allow(clippy::too_many_arguments)]
fn colocated<S: Sample>(
    refs: &SliceRefs<S>,
    col: &Frame<S>,
    cur: &Frame<S>,
    addr: usize,
    field_mb: bool,
    mb_parity: u8,
    inference: bool,
    part: usize,
    sub: usize,
) -> ColBlock {
    let mbw = cur.mb_width;
    let (x, row) = (addr % mbw, addr / mbw);
    let blk = colocated_block(inference, part, sub);
    let (bx, by) = (blk % 4, blk / 4);
    let y_col = by * 4;
    let cur_field = refs.cur_parity != PARITY_FRAME;
    if cur.mbaff {
        // AFRM current (Table 8-8, last six rows): the colocated pair is at
        // this pair's position; which of its two macroblocks depends on the
        // kinds on both sides.
        let pr = row / 2;
        let bottom = row % 2;
        if col.field_coded {
            // FLD colocated (a complementary field pair): the field of this
            // macroblock's parity for a field macroblock, the closer one in
            // POC for a frame macroblock (Table 8-6).
            let cp = if field_mb {
                mb_parity as usize
            } else {
                col_field_of_frame(refs, col) as usize
            };
            let frow = 2 * pr + cp;
            if field_mb {
                ColBlock {
                    addr: frow * mbw + x,
                    blk,
                    scale: VertScale::OneToOne,
                }
            } else {
                let y_m = 8 * bottom + 4 * (y_col / 8);
                ColBlock {
                    addr: frow * mbw + x,
                    blk: (y_m / 4) * 4 + bx,
                    scale: VertScale::FldToFrm,
                }
            }
        } else {
            let col_field = col.mb_field[addr];
            match (field_mb, col_field) {
                (false, false) | (true, true) => ColBlock {
                    addr,
                    blk,
                    scale: VertScale::OneToOne,
                },
                (false, true) => {
                    // mbAddrCol6: the field macroblock (of the pair) closer in POC.
                    let cp = col_field_of_frame(refs, col) as usize;
                    let y_m = 8 * bottom + 4 * (y_col / 8);
                    ColBlock {
                        addr: (2 * pr + cp) * mbw + x,
                        blk: (y_m / 4) * 4 + bx,
                        scale: VertScale::FldToFrm,
                    }
                }
                (true, false) => {
                    // mbAddrCol7: the frame macroblock of the pair covering this half.
                    let y_m = (2 * y_col) % 16;
                    ColBlock {
                        addr: (2 * pr + y_col / 8) * mbw + x,
                        blk: (y_m / 4) * 4 + bx,
                        scale: VertScale::FrmToFld,
                    }
                }
            }
        }
    } else if cur_field {
        if col.field_coded {
            // FLD / FLD: the same position in the colocated field.
            let frow = 2 * row + refs.col_parity as usize;
            ColBlock {
                addr: frow * mbw + x,
                blk,
                scale: VertScale::OneToOne,
            }
        } else if col.mbaff && col.mb_field[(2 * row) * mbw + x] {
            // FLD / AFRM with a field pair: the field macroblock of the
            // current parity.
            let frow = 2 * row + refs.cur_parity as usize;
            ColBlock {
                addr: frow * mbw + x,
                blk,
                scale: VertScale::OneToOne,
            }
        } else {
            // FLD / FRM (or a frame pair of an AFRM): the frame macroblock
            // covering the current field macroblock's half.
            let frow = 2 * row + y_col / 8;
            let y_m = (2 * y_col) % 16;
            ColBlock {
                addr: frow * mbw + x,
                blk: (y_m / 4) * 4 + bx,
                scale: VertScale::FrmToFld,
            }
        }
    } else if col.field_coded {
        // FRM / FLD: the field macroblock (of the field chosen by POC
        // distance) covering this frame macroblock's rows.
        let cp = col_field_of_frame(refs, col) as usize;
        let frow = 2 * (row / 2) + cp;
        let y_m = 8 * (row % 2) + 4 * (y_col / 8);
        ColBlock {
            addr: frow * mbw + x,
            blk: (y_m / 4) * 4 + bx,
            scale: VertScale::FldToFrm,
        }
    } else {
        ColBlock {
            addr,
            blk,
            scale: VertScale::OneToOne,
        }
    }
}

/// Direct-mode motion (8.4.1.2) for the given 8x8 partitions of the
/// current macroblock: stores it and queues the prediction jobs.
fn direct_partitions<S: Sample>(
    ctx: &SliceCtx,
    cur: &mut Frame<S>,
    info: &PicInfo,
    nb: &MbNeighbours,
    refs: &SliceRefs<S>,
    parts: &[usize],
    jobs: &mut Jobs,
    geom: MbGeom,
) -> Result<()> {
    let addr = nb.addr;
    if refs.frames[1].is_empty() || refs.frames[0].is_empty() {
        return Err(Error::bitstream(
            "direct prediction without both reference lists",
        ));
    }
    let field_mb = geom.step == 2;
    let mb_parity = geom.parity;
    let list_mult = if field_mb { 2 } else { 1 };
    let cur_field = refs.cur_parity != PARITY_FRAME;
    let cur_frame_mbh = cur.mb_height * if cur_field { 2 } else { 1 };
    let col_avail = refs
        .col
        .is_some_and(|c| c.mb_width == cur.mb_width && c.mb_height == cur_frame_mbh);
    // The colocated macroblock's motion is read: wait for its rows (in
    // frame rows of the colocated frame; a field's row is two frame rows).
    let mby = addr / info.mb_width;
    if col_avail {
        if let Some(cs) = refs.col_shared {
            let col = refs.col.unwrap();
            let (wait_parity, frow_end) = if cur_field {
                (
                    if col.field_coded {
                        refs.col_parity
                    } else {
                        PARITY_FRAME
                    },
                    2 * mby + 2,
                )
            } else if cur.mbaff {
                // Frame macroblocks read the closer field, field macroblocks
                // their own parity: wait for both.
                (PARITY_FRAME, (mby / 2) * 2 + 2)
            } else if col.field_coded {
                (col_field_of_frame(refs, col), (mby / 2) * 2 + 2)
            } else {
                (PARITY_FRAME, mby + 1)
            };
            cs.wait_decoded(wait_parity, (frow_end * 16) as i32);
        }
    }
    if ctx.direct_spatial {
        let mut ref_idx = spatial_direct_ref_idx(nb, cur, info);
        let mut mvp = [Mv::ZERO; 2];
        if ref_idx[0] < 0 && ref_idx[1] < 0 {
            ref_idx = [0, 0];
        } else {
            for list in 0..2 {
                if ref_idx[list] >= 0 {
                    let (a, b, c) = prediction_neighbours(nb, cur, info, 0, list, 0, 0, 4);
                    mvp[list] = median_mvp(a, b, c, ref_idx[list]);
                }
            }
        }
        for l in 0..2 {
            if ref_idx[l] >= 0 && ref_idx[l] as usize >= refs.frames[l].len() * list_mult {
                return Err(Error::bitstream(
                    "direct spatial ref_idx beyond the reference list",
                ));
            }
        }
        for &part in parts {
            let subs = if ctx.direct_8x8_inference { 1 } else { 4 };
            for sub in 0..subs {
                let (x, y, w, h) = if ctx.direct_8x8_inference {
                    sub_partition_rect(part, SubMbShape::S8x8, 0)
                } else {
                    sub_partition_rect(part, SubMbShape::S4x4, sub)
                };
                // colZeroFlag from the colocated block (its vector unscaled,
                // NOTE 2 of 8.4.1.2.2).
                let mut col_zero = false;
                if col_avail && !refs.col_long_term {
                    let col = refs.col.unwrap();
                    let cb = colocated(
                        refs,
                        col,
                        cur,
                        addr,
                        field_mb,
                        mb_parity,
                        ctx.direct_8x8_inference,
                        part,
                        sub,
                    );
                    let (mv_col, ref_col, _, _) = colocated_motion(col, cb.addr, cb.blk);
                    col_zero = ref_col == 0
                        && (-1..=1).contains(&mv_col.x)
                        && (-1..=1).contains(&mv_col.y);
                }
                let mut mvs = [Mv::ZERO; 2];
                let mut rids = [-1i8; 2];
                for list in 0..2 {
                    if ref_idx[list] < 0 {
                        fill_motion(cur, addr, list, x, y, w, h, BlockMotion::default());
                        continue;
                    }
                    let mv = if ref_idx[list] == 0 && col_zero {
                        Mv::ZERO
                    } else {
                        mvp[list]
                    };
                    fill_motion(
                        cur,
                        addr,
                        list,
                        x,
                        y,
                        w,
                        h,
                        refs.motion_of(list, ref_idx[list], mv, field_mb, mb_parity),
                    );
                    mvs[list] = mv;
                    rids[list] = ref_idx[list];
                }
                jobs.push((x, y, w, h, rids[0], mvs[0], rids[1], mvs[1]));
            }
        }
    } else {
        // Temporal.
        for &part in parts {
            let subs = if ctx.direct_8x8_inference { 1 } else { 4 };
            for sub in 0..subs {
                let (x, y, w, h) = if ctx.direct_8x8_inference {
                    sub_partition_rect(part, SubMbShape::S8x8, 0)
                } else {
                    sub_partition_rect(part, SubMbShape::S4x4, sub)
                };
                let (mv_col, ref_col, ref_id_col, ref_par_col, scale) = if col_avail {
                    let col = refs.col.unwrap();
                    let cb = colocated(
                        refs,
                        col,
                        cur,
                        addr,
                        field_mb,
                        mb_parity,
                        ctx.direct_8x8_inference,
                        part,
                        sub,
                    );
                    let (mv, r, id, par) = colocated_motion(col, cb.addr, cb.blk);
                    (mv, r, id, par, cb.scale)
                } else {
                    (
                        Mv::ZERO,
                        -1,
                        0,
                        super::frame::PARITY_NONE,
                        VertScale::OneToOne,
                    )
                };
                // The vertical vector in the current picture's units (8-193 /
                // 8-194; "/" truncates toward zero).
                let mv_col = match scale {
                    VertScale::OneToOne => mv_col,
                    VertScale::FrmToFld => Mv::new(mv_col.x, mv_col.y / 2),
                    VertScale::FldToFrm => Mv::new(mv_col.x, mv_col.y.wrapping_mul(2)),
                };
                // refIdxL0: the lowest index in list 0 referencing refPicCol —
                // as a field of the current parity when the colocated frame
                // macroblock referenced a frame, as the frame when the
                // colocated field macroblock referenced a field (MapColToList0).
                let ref0: i8 = if ref_col < 0 {
                    0
                } else if field_mb {
                    // A field macroblock of an MBAFF frame: list 0 holds
                    // frames; the index names a field of the frame holding
                    // refPicCol — the same parity as this macroblock (even)
                    // or the other one (odd; a colocated field of the other
                    // parity referenced a field of that parity).
                    let frm = refs.ids[0]
                        .iter()
                        .position(|&id| id == ref_id_col)
                        .unwrap_or(0) as i8;
                    let opposite = scale == VertScale::OneToOne
                        && ref_par_col != PARITY_FRAME
                        && ref_par_col != mb_parity;
                    (frm << 1) + opposite as i8
                } else {
                    let want_par = match scale {
                        VertScale::OneToOne => ref_par_col,
                        VertScale::FrmToFld => refs.cur_parity,
                        VertScale::FldToFrm => PARITY_FRAME,
                    };
                    refs.ids[0]
                        .iter()
                        .zip(refs.parity[0].iter())
                        .position(|(&id, &par)| id == ref_id_col && par == want_par)
                        .unwrap_or(0) as i8
                };
                let poc0 = refs.ref_poc(0, ref0, field_mb, mb_parity);
                let poc1 = refs.ref_poc(1, 0, field_mb, mb_parity);
                let cur_poc = if field_mb {
                    refs.cur_field_poc[mb_parity as usize]
                } else {
                    refs.cur_poc
                };
                let lt0 = refs.long_term[0][refs.resolve(0, ref0, field_mb, mb_parity).0];
                let (mv0, mv1) = if lt0 || poc1 - poc0 == 0 {
                    (mv_col, Mv::ZERO)
                } else {
                    let tb = (cur_poc - poc0).clamp(-128, 127);
                    let td = (poc1 - poc0).clamp(-128, 127);
                    let tx = (16384 + (td / 2).abs()) / td;
                    let dsf = ((tb * tx + 32) >> 6).clamp(-1024, 1023);
                    let sx = ((dsf * mv_col.x as i32 + 128) >> 8) as i16;
                    let sy = ((dsf * mv_col.y as i32 + 128) >> 8) as i16;
                    (
                        Mv::new(sx, sy),
                        Mv::new(sx.wrapping_sub(mv_col.x), sy.wrapping_sub(mv_col.y)),
                    )
                };
                fill_motion(
                    cur,
                    addr,
                    0,
                    x,
                    y,
                    w,
                    h,
                    refs.motion_of(0, ref0, mv0, field_mb, mb_parity),
                );
                fill_motion(
                    cur,
                    addr,
                    1,
                    x,
                    y,
                    w,
                    h,
                    refs.motion_of(1, 0, mv1, field_mb, mb_parity),
                );
                jobs.push((x, y, w, h, ref0, mv0, 0, mv1));
            }
        }
    }
    // A whole-macroblock direct prediction whose four 8x8s came out with the
    // same motion (the common B_Skip / B_Direct_16x16 case) is one 16x16 job:
    // a quarter of the interpolation calls and one combine.
    if parts.len() == 4 && jobs.len == 4 {
        let all = jobs.as_slice();
        let j0 = all[0];
        if all[1..]
            .iter()
            .all(|j| (j.4, j.5, j.6, j.7) == (j0.4, j0.5, j0.6, j0.7))
            && all.iter().all(|j| j.2 == 8 && j.3 == 8)
        {
            jobs.len = 0;
            jobs.push((0, 0, 16, 16, j0.4, j0.5, j0.6, j0.7));
        }
    }
    let _ = block_available;
    Ok(())
}

/// Add the inverse transform of a dequantised 4x4 block of `levels` to `dst`
/// (`dc`: an already-scaled DC replacing position 0, or `NO_DC`).
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn residual4<S: Sample>(
    dsp: &H264Dsp<S>,
    dst: &mut [S],
    stride: usize,
    levels: &[i32],
    scale: &[i32; 16],
    qp: i32,
    dc: Option<i32>,
    max: i32,
) {
    let levels: &[i32; 16] = levels.try_into().expect("16 levels");
    (dsp.residual4)(dst, stride, levels, scale, qp, dc.unwrap_or(NO_DC), max);
}

/// The colour plane `p` of a frame (0 luma, 1 Cb, 2 Cr).
#[inline(always)]
fn plane_mut<S: Sample>(cur: &mut Frame<S>, p: usize) -> &mut super::frame::PaddedPlane<S> {
    match p {
        0 => &mut cur.y,
        1 => &mut cur.cb,
        _ => &mut cur.cr,
    }
}

/// Whether an Intra_NxN / Intra_16x16 mode is vertical (`Some(false)`) or
/// horizontal (`Some(true)`) — the modes whose transform-bypass residual is
/// accumulated (8.5.15).
#[inline(always)]
fn hv_mode(mode: u8) -> Option<bool> {
    match mode {
        0 => Some(false),
        1 => Some(true),
        _ => None,
    }
}

/// Transform-bypass residual (lossless, TransformBypassModeFlag): add the
/// `n x n` raster of levels `r` to `dst` as it is, after the DPCM of 8.5.15
/// when `hor` says the block was predicted horizontally (`Some(true)`) or
/// vertically (`Some(false)`).
#[inline(always)]
fn add_bypass<S: Sample>(
    dst: &mut [S],
    stride: usize,
    r: &mut [i32],
    n: usize,
    hor: Option<bool>,
    max: i32,
) {
    add_bypass_rect(dst, stride, r, n, n, hor, max);
}

/// [`add_bypass`] for a `w x h` block.
fn add_bypass_rect<S: Sample>(
    dst: &mut [S],
    stride: usize,
    r: &mut [i32],
    w: usize,
    h: usize,
    hor: Option<bool>,
    max: i32,
) {
    match hor {
        Some(false) => {
            for i in 1..h {
                for j in 0..w {
                    r[i * w + j] += r[(i - 1) * w + j];
                }
            }
        }
        Some(true) => {
            for i in 0..h {
                for j in 1..w {
                    r[i * w + j] += r[i * w + j - 1];
                }
            }
        }
        None => {}
    }
    for i in 0..h {
        for j in 0..w {
            let d = &mut dst[i * stride + j];
            *d = S::from_i32((d.to_i32() + r[i * w + j]).clamp(0, max));
        }
    }
}

/// Add the inverse transform of a dequantised 8x8 block of `levels` to `dst`.
#[inline(always)]
fn residual8<S: Sample>(
    dsp: &H264Dsp<S>,
    dst: &mut [S],
    stride: usize,
    levels: &[i32],
    scale: &[i32; 64],
    qp: i32,
    max: i32,
) {
    let levels: &[i32; 64] = levels.try_into().expect("64 levels");
    (dsp.residual8)(dst, stride, levels, scale, qp, max);
}
