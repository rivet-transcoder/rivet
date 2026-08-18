//! Coding tree unit decoding (H.265 7.3.8): SAO syntax, the coding
//! quadtree, coding units, prediction units (motion derivation and
//! compensation), transform trees and units (residual + reconstruction),
//! and the per-CU bookkeeping the loop filters and later CUs read.

use crate::cabac::Cabac;
use crate::picture::ChromaFormat;
use crate::{Error, Result};

use super::ctx::*;
use super::frame::{Frame, MotionInfo, Mv, SharedFrame, Sample};
use super::inter::{McScratch, Weighting, predict_block};
use super::intra::{RefAvail, predict as intra_predict};
use super::mvpred::{Cand, PuPos, RefCtx, amvp, merge_candidate};
use super::pic::{PicInfo, SaoParams};
use super::pps::Pps;
use super::residual::{ResidualParams, ScalingSource, parse_residual, scale_coefficients, transform_skip_residual};
use crate::dsp::hevc::HevcDsp;
use super::slice::{SliceHeader, SliceType};
use super::sps::{ScalingList, Sps};

/// Partition modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PartMode {
    /// 2Nx2N.
    P2Nx2N,
    /// 2NxN.
    P2NxN,
    /// Nx2N.
    PNx2N,
    /// NxN.
    PNxN,
    /// 2NxnU.
    P2NxnU,
    /// 2NxnD.
    P2NxnD,
    /// nLx2N.
    PnLx2N,
    /// nRx2N.
    PnRx2N,
}

impl PartMode {
    /// The prediction blocks `(x, y, w, h)` relative to the CB of size `n`
    /// (the first `count` entries are valid).
    fn pus(self, n: i32) -> Pus {
        let z = (0, 0, 0, 0);
        match self {
            PartMode::P2Nx2N => Pus { list: [(0, 0, n, n), z, z, z], count: 1 },
            PartMode::P2NxN => Pus { list: [(0, 0, n, n / 2), (0, n / 2, n, n / 2), z, z], count: 2 },
            PartMode::PNx2N => Pus { list: [(0, 0, n / 2, n), (n / 2, 0, n / 2, n), z, z], count: 2 },
            PartMode::PNxN => Pus { list: [(0, 0, n / 2, n / 2), (n / 2, 0, n / 2, n / 2), (0, n / 2, n / 2, n / 2), (n / 2, n / 2, n / 2, n / 2)], count: 4 },
            PartMode::P2NxnU => Pus { list: [(0, 0, n, n / 4), (0, n / 4, n, n * 3 / 4), z, z], count: 2 },
            PartMode::P2NxnD => Pus { list: [(0, 0, n, n * 3 / 4), (0, n * 3 / 4, n, n / 4), z, z], count: 2 },
            PartMode::PnLx2N => Pus { list: [(0, 0, n / 4, n), (n / 4, 0, n * 3 / 4, n), z, z], count: 2 },
            PartMode::PnRx2N => Pus { list: [(0, 0, n * 3 / 4, n), (n * 3 / 4, 0, n / 4, n), z, z], count: 2 },
        }
    }
}

/// The prediction blocks of a partitioning, without a heap allocation.
struct Pus {
    list: [(i32, i32, i32, i32); 4],
    count: usize,
}

impl Pus {
    #[inline(always)]
    fn iter(&self) -> impl Iterator<Item = &(i32, i32, i32, i32)> {
        self.list[..self.count].iter()
    }
}

/// Everything one slice segment's CTUs need.
pub struct SliceDec<'a, S: Sample = u16> {
    /// SPS.
    pub sps: &'a Sps,
    /// PPS.
    pub pps: &'a Pps,
    /// Slice header (of the independent segment).
    pub hdr: &'a SliceHeader,
    /// The picture being decoded.
    pub frame: &'a mut Frame<S>,
    /// Per-picture side data.
    pub info: &'a mut PicInfo,
    /// The arithmetic decoder over the current substream.
    pub cabac: Cabac<'a>,
    /// Context variables.
    pub cx: Contexts,
    /// Reference picture context.
    pub refs: RefCtx<'a, S>,
    /// Reference frames per list.
    pub ref_frames: [Vec<&'a Frame<S>>; 2],
    /// The same references with their progress (for waiting on rows still
    /// being decoded by another thread).
    pub ref_shared: [Vec<&'a SharedFrame<S>>; 2],
    /// The collocated picture's progress.
    pub col_shared: Option<&'a SharedFrame<S>>,
    /// Slice index (into `info.slices`).
    pub slice_idx: u16,
    /// `SliceAddrRs`.
    pub slice_addr: u32,
    /// Resolved scaling lists (None = flat).
    pub scaling: Option<ScalingList>,
    /// `QpY` of the current CU (running).
    pub qp_y: i32,
    /// `qPY_PREV` for the next quantisation group.
    pub qp_y_prev: i32,
    /// `CuQpDeltaVal`.
    pub cu_qp_delta_val: i32,
    /// `IsCuQpDeltaCoded`.
    pub is_cu_qp_delta_coded: bool,
    /// The current quantisation group's top-left.
    pub qg: (i32, i32),
    /// `qPY_PREV` resolved for the current QG.
    pub qg_qp_prev: i32,
    /// Whether the next QG is the first in the slice / tile / CTB row (WPP).
    pub first_qg: bool,
    /// The last PU parsed used merge (for `rqt_root_cbf` presence).
    pub last_pu_merged: bool,
    /// Current CTB address (raster).
    pub ctb_addr_rs: usize,
    /// Current CTB address (tile scan).
    pub ctb_addr_ts: usize,
    /// Scratch coefficient buffer.
    pub coeffs: Vec<i16>,
    /// The kernels.
    pub dsp: HevcDsp<S>,
    /// Motion compensation scratch.
    pub mc: McScratch<S>,
    /// Non-fatal problems seen.
    pub warnings: u64,
    /// Debug tracing (from the `H26X_TRACE_*` environment variables).
    pub trace: TraceCfg,
}

/// What to print while decoding (debugging aid; all off by default).
#[derive(Debug, Clone, Copy, Default)]
pub struct TraceCfg {
    /// `H26X_TRACE_CU`: one line per coding unit.
    pub cu: bool,
    /// `H26X_TRACE_PU=x,y`: the prediction units covering luma (x, y).
    pub pu: Option<(i32, i32)>,
    /// `H26X_TRACE_TB=c,x,y`: the transform blocks of component c covering (x, y).
    pub tb: Option<(usize, usize, usize)>,
    /// `H26X_TRACE_CTB`: a checksum per CTB after reconstruction.
    pub ctb: bool,
}

impl TraceCfg {
    /// Read the environment once.
    pub fn from_env() -> Self {
        let pair = |name: &str| -> Option<Vec<i64>> {
            let v = std::env::var(name).ok()?;
            Some(v.split(',').filter_map(|t| t.parse().ok()).collect())
        };
        TraceCfg {
            cu: std::env::var_os("H26X_TRACE_CU").is_some(),
            ctb: std::env::var_os("H26X_TRACE_CTB").is_some(),
            pu: pair("H26X_TRACE_PU").filter(|p| p.len() == 2).map(|p| (p[0] as i32, p[1] as i32)),
            tb: pair("H26X_TRACE_TB").filter(|p| p.len() == 3).map(|p| (p[0] as usize, p[1] as usize, p[2] as usize)),
        }
    }
    fn tb_hit(&self, c_idx: usize, x: usize, y: usize, n: usize) -> bool {
        matches!(self.tb, Some((c, tx, ty)) if c == c_idx && tx >= x && tx < x + n && ty >= y && ty < y + n)
    }
}

#[inline(always)]
fn bin(c: &mut Cabac, cx: &mut Contexts, ctx: usize) -> u32 {
    c.decision(&mut cx.c[ctx])
}

impl<'a, S: Sample> SliceDec<'a, S> {
    fn bit_depth(&self) -> u32 {
        self.sps.bit_depth_luma
    }

    /// Whether the neighbouring luma location is available for the current
    /// block at `(xc, yc)` (z-scan availability, 6.4.1).
    fn avail(&self, xc: i32, yc: i32, xn: i32, yn: i32) -> bool {
        self.info.available(xc, yc, xn, yn, self.frame.width as i32, self.frame.height as i32)
    }

    // ------------------------------------------------------------------
    // SAO syntax (7.3.8.3)
    // ------------------------------------------------------------------
    fn parse_sao(&mut self, rx: usize, ry: usize) -> Result<()> {
        let ctb = self.ctb_addr_rs;
        let wc = self.info.wc;
        let mut merge_left = false;
        let mut merge_up = false;
        if rx > 0 {
            let left_in_slice = ctb as u32 > self.slice_addr;
            let left_in_tile = self.info.ctb_tile[ctb] == self.info.ctb_tile[ctb - 1];
            if left_in_slice && left_in_tile {
                merge_left = bin(&mut self.cabac, &mut self.cx, SAO_MERGE_FLAG_OFFSET) != 0;
            }
        }
        if ry > 0 && !merge_left {
            let up_in_slice = (ctb - wc) as u32 >= self.slice_addr;
            let up_in_tile = self.info.ctb_tile[ctb] == self.info.ctb_tile[ctb - wc];
            if up_in_slice && up_in_tile {
                merge_up = bin(&mut self.cabac, &mut self.cx, SAO_MERGE_FLAG_OFFSET) != 0;
            }
        }
        if merge_left {
            self.info.sao[ctb] = self.info.sao[ctb - 1];
            return Ok(());
        }
        if merge_up {
            self.info.sao[ctb] = self.info.sao[ctb - wc];
            return Ok(());
        }
        let mut params = [SaoParams::default(); 3];
        let ncomp = if self.sps.chroma_format_idc != 0 { 3 } else { 1 };
        let bd = self.bit_depth();
        let cmax = (1u32 << (bd.min(10) - 5)) - 1;
        for c_idx in 0..ncomp {
            if !((self.hdr.sao_luma && c_idx == 0) || (self.hdr.sao_chroma && c_idx > 0)) {
                continue;
            }
            if c_idx == 0 || c_idx == 1 {
                // sao_type_idx: TR cMax 2, first bin ctx, second bypass.
                let t = if bin(&mut self.cabac, &mut self.cx, SAO_TYPE_IDX_OFFSET) == 0 {
                    0
                } else if self.cabac.bypass() == 0 {
                    1
                } else {
                    2
                };
                params[c_idx].type_idx = t;
                if c_idx == 1 {
                    params[2].type_idx = t;
                }
            }
            if params[c_idx].type_idx == 0 {
                continue;
            }
            // SaoOffsetVal = sign * abs << log2OffsetScale (7-72); the scale
            // comes from the PPS range extension (0 without it).
            let shift = if c_idx == 0 { self.pps.log2_sao_offset_scale.0 } else { self.pps.log2_sao_offset_scale.1 };
            let mut abs = [0i32; 4];
            for a in abs.iter_mut() {
                let mut v = 0u32;
                while v < cmax && self.cabac.bypass() != 0 {
                    v += 1;
                }
                *a = v as i32;
            }
            if params[c_idx].type_idx == 1 {
                for a in abs.iter_mut() {
                    if *a != 0 && self.cabac.bypass() != 0 {
                        *a = -*a;
                    }
                }
                params[c_idx].band_or_class = self.cabac.bypass_bits(5) as u8;
                for i in 0..4 {
                    params[c_idx].offsets[i] = (abs[i] << shift) as i16;
                }
            } else {
                // Edge: offsets 0,1 positive, 2,3 negative.
                params[c_idx].offsets = [(abs[0] << shift) as i16, (abs[1] << shift) as i16, (-(abs[2] << shift)) as i16, (-(abs[3] << shift)) as i16];
                if c_idx == 0 {
                    params[0].band_or_class = self.cabac.bypass_bits(2) as u8;
                } else if c_idx == 1 {
                    let cls = self.cabac.bypass_bits(2) as u8;
                    params[1].band_or_class = cls;
                    params[2].band_or_class = cls;
                } else {
                    // Cr shares the class with Cb (already set).
                }
            }
        }
        self.info.sao[ctb] = params;
        Ok(())
    }

    // ------------------------------------------------------------------
    // Coding tree
    // ------------------------------------------------------------------

    /// Decode one CTU (SAO + coding quadtree). Returns Ok(()) after
    /// `end_of_slice_segment_flag` is left for the caller.
    pub fn decode_ctu(&mut self, ctb_addr_rs: usize, ctb_addr_ts: usize) -> Result<()> {
        self.ctb_addr_rs = ctb_addr_rs;
        self.ctb_addr_ts = ctb_addr_ts;
        let wc = self.info.wc;
        let rx = ctb_addr_rs % wc;
        let ry = ctb_addr_rs / wc;
        self.info.ctb_slice[ctb_addr_rs] = self.slice_idx;
        self.info.ctb_slice_addr[ctb_addr_rs] = self.slice_addr;
        if self.hdr.sao_luma || self.hdr.sao_chroma {
            self.parse_sao(rx, ry)?;
        }
        let log2 = self.sps.log2_ctb_size;
        let x0 = (rx << log2) as i32;
        let y0 = (ry << log2) as i32;
        self.coding_quadtree(x0, y0, log2, 0)
    }

    fn coding_quadtree(&mut self, x0: i32, y0: i32, log2_cb: u32, depth: u32) -> Result<()> {
        let size = 1i32 << log2_cb;
        let (pw, ph) = (self.frame.width as i32, self.frame.height as i32);
        let split = if x0 + size <= pw && y0 + size <= ph && log2_cb > self.sps.log2_min_cb_size {
            // split_cu_flag with context from the neighbours' depth.
            let mut inc = 0usize;
            if self.avail(x0, y0, x0 - 1, y0) && self.info.ct_depth[self.info.idx4((x0 - 1) as usize, y0 as usize)] as u32 > depth {
                inc += 1;
            }
            if self.avail(x0, y0, x0, y0 - 1) && self.info.ct_depth[self.info.idx4(x0 as usize, (y0 - 1) as usize)] as u32 > depth {
                inc += 1;
            }
            bin(&mut self.cabac, &mut self.cx, SPLIT_CODING_UNIT_FLAG_OFFSET + inc) != 0
        } else {
            log2_cb > self.sps.log2_min_cb_size
        };
        if self.pps.cu_qp_delta_enabled && log2_cb >= self.sps.log2_ctb_size - self.pps.diff_cu_qp_delta_depth {
            self.is_cu_qp_delta_coded = false;
            self.cu_qp_delta_val = 0;
            self.qg = (x0, y0);
            // qPY_PREV: SliceQpY for the first QG of a slice / tile / WPP row,
            // else the QpY of the last CU of the previous QG.
            self.qg_qp_prev = if self.first_qg { self.hdr.slice_qp } else { self.qp_y_prev };
            self.first_qg = false;
        }
        if split {
            let half = size / 2;
            let x1 = x0 + half;
            let y1 = y0 + half;
            self.coding_quadtree(x0, y0, log2_cb - 1, depth + 1)?;
            if x1 < pw {
                self.coding_quadtree(x1, y0, log2_cb - 1, depth + 1)?;
            }
            if y1 < ph {
                self.coding_quadtree(x0, y1, log2_cb - 1, depth + 1)?;
            }
            if x1 < pw && y1 < ph {
                self.coding_quadtree(x1, y1, log2_cb - 1, depth + 1)?;
            }
        } else {
            self.coding_unit(x0, y0, log2_cb, depth)?;
        }
        // End of a quantisation group: remember QpY for qPY_PREV.
        if self.pps.cu_qp_delta_enabled {
            let mask = (1i32 << (self.sps.log2_ctb_size - self.pps.diff_cu_qp_delta_depth)) - 1;
            if ((x0 + size) & mask) == 0 && ((y0 + size) & mask) == 0 {
                self.qp_y_prev = self.qp_y;
            }
        }
        Ok(())
    }

    /// `qPY_PRED` for the CU at `(x_cb, y_cb)` in the current QG (8.6.1).
    fn qp_y_pred(&mut self, x_cb: i32, y_cb: i32) -> i32 {
        let (xq, yq) = self.qg;
        let prev = self.qg_qp_prev;
        let ctb_cur = self.info.ctb_of(x_cb as usize, y_cb as usize);
        let qa = if self.avail(x_cb, y_cb, xq - 1, yq) && self.info.ctb_of((xq - 1) as usize, yq as usize) == ctb_cur {
            self.info.qp_y[self.info.idx4((xq - 1) as usize, yq as usize)] as i32
        } else {
            prev
        };
        let qb = if self.avail(x_cb, y_cb, xq, yq - 1) && self.info.ctb_of(xq as usize, (yq - 1) as usize) == ctb_cur {
            self.info.qp_y[self.info.idx4(xq as usize, (yq - 1) as usize)] as i32
        } else {
            prev
        };
        (qa + qb + 1) >> 1
    }

    fn set_qp(&mut self, x_cb: i32, y_cb: i32) {
        let pred = self.qp_y_pred(x_cb, y_cb);
        let bd_off = 6 * (self.sps.bit_depth_luma as i32 - 8);
        self.qp_y = ((pred + self.cu_qp_delta_val + 52 + 2 * bd_off) % (52 + bd_off)) - bd_off;
    }

    fn coding_unit(&mut self, x0: i32, y0: i32, log2_cb: u32, depth: u32) -> Result<()> {
        let n = 1i32 << log2_cb;
        let (pw, ph) = (self.frame.width as i32, self.frame.height as i32);
        let w4 = self.info.w4;
        let cw = (n.min(pw - x0)) as usize;
        let ch = (n.min(ph - y0)) as usize;
        let mut bypass = false;
        if self.pps.transquant_bypass_enabled {
            bypass = bin(&mut self.cabac, &mut self.cx, CU_TRANSQUANT_BYPASS_FLAG_OFFSET) != 0;
        }
        let mut skip = false;
        if self.hdr.slice_type != SliceType::I {
            let mut inc = 0usize;
            if self.avail(x0, y0, x0 - 1, y0) && self.info.skip[self.info.idx4((x0 - 1) as usize, y0 as usize)] != 0 {
                inc += 1;
            }
            if self.avail(x0, y0, x0, y0 - 1) && self.info.skip[self.info.idx4(x0 as usize, (y0 - 1) as usize)] != 0 {
                inc += 1;
            }
            skip = bin(&mut self.cabac, &mut self.cx, SKIP_FLAG_OFFSET + inc) != 0;
        }
        // Record depth / skip / pred mode for the CU area now (neighbours in
        // this CU need them; availability uses pred_mode != 2).
        PicInfo::fill4(&mut self.info.ct_depth, w4, x0 as usize, y0 as usize, cw, ch, depth as u8);
        PicInfo::fill4(&mut self.info.skip, w4, x0 as usize, y0 as usize, cw, ch, skip as u8);

        // QP for this CU (delta may still arrive in a TU; recomputed then).
        self.set_qp(x0, y0);

        let intra;
        let mut part_mode = PartMode::P2Nx2N;
        let mut pcm = false;
        if skip {
            intra = false;
        } else {
            intra = if self.hdr.slice_type != SliceType::I {
                bin(&mut self.cabac, &mut self.cx, PRED_MODE_FLAG_OFFSET) != 0
            } else {
                true
            };
            if !intra || log2_cb == self.sps.log2_min_cb_size {
                part_mode = self.parse_part_mode(intra, log2_cb)?;
            }
        }
        PicInfo::fill4(&mut self.info.pred_mode, w4, x0 as usize, y0 as usize, cw, ch, intra as u8);
        // Motion of an intra CU: none (TMVP treats it as unavailable). An
        // inter CU's prediction units cover it entirely and write their own.
        if intra {
            let w4 = self.frame.w4;
            let (bx0, bx1) = (x0 as usize >> 2, (x0 as usize + cw) >> 2);
            for by in (y0 as usize >> 2)..((y0 as usize + ch) >> 2) {
                self.frame.motion[by * w4 + bx0..by * w4 + bx1].fill(MotionInfo::default());
            }
        }

        let mut intra_modes = [1u32; 4];
        let mut chroma_mode_syntax = 0u32;
        if intra {
            if part_mode == PartMode::P2Nx2N
                && self.sps.pcm_enabled
                && log2_cb >= self.sps.pcm.2
                && log2_cb <= self.sps.pcm.3
            {
                pcm = self.cabac.terminate() != 0;
            }
            if pcm {
                self.decode_pcm(x0, y0, log2_cb)?;
            } else {
                let npu = if part_mode == PartMode::PNxN { 4 } else { 1 };
                let pb = if part_mode == PartMode::PNxN { n / 2 } else { n };
                let mut prev_flags = [false; 4];
                for i in 0..npu {
                    prev_flags[i] = bin(&mut self.cabac, &mut self.cx, PREV_INTRA_LUMA_PRED_FLAG_OFFSET) != 0;
                }
                for i in 0..npu {
                    let xp = x0 + (i as i32 % 2) * pb;
                    let yp = y0 + (i as i32 / 2) * pb;
                    let mode = if prev_flags[i] {
                        // mpm_idx: TR cMax 2 bypass.
                        let mut idx = 0usize;
                        while idx < 2 && self.cabac.bypass() != 0 {
                            idx += 1;
                        }
                        let cands = self.mpm_candidates(xp, yp);
                        cands[idx]
                    } else {
                        let rem = self.cabac.bypass_bits(5);
                        let mut cands = self.mpm_candidates(xp, yp);
                        cands.sort_unstable();
                        let mut m = rem;
                        for c in cands {
                            if m >= c {
                                m += 1;
                            }
                        }
                        m
                    };
                    intra_modes[i] = mode;
                    PicInfo::fill4(&mut self.info.intra_mode, w4, xp as usize, yp as usize, pb as usize, pb as usize, mode as u8);
                }
                if self.sps.chroma_format_idc != 0 {
                    // intra_chroma_pred_mode: bin0 ctx; 0 -> 4, else 2 bypass bits.
                    chroma_mode_syntax = if bin(&mut self.cabac, &mut self.cx, INTRA_CHROMA_PRED_MODE_OFFSET) == 0 {
                        4
                    } else {
                        self.cabac.bypass_bits(2)
                    };
                }
            }
        } else {
            // Inter: prediction units.
            let pus = part_mode.pus(n);
            for (part_idx, &(px, py, pwid, phei)) in pus.iter().enumerate() {
                self.prediction_unit(x0, y0, n, x0 + px, y0 + py, pwid, phei, part_idx as u32, skip)?;
            }
        }

        // Residual.
        let mut rqt_root_cbf = true;
        if !pcm {
            if !intra && !(part_mode == PartMode::P2Nx2N && self.last_pu_merged) && !skip {
                rqt_root_cbf = bin(&mut self.cabac, &mut self.cx, NO_RESIDUAL_DATA_FLAG_OFFSET) != 0;
            } else if skip {
                rqt_root_cbf = false;
            }
            // Chroma intra mode (IntraPredModeC, 8.4.3) for the CU.
            let chroma_mode = if intra {
                let luma0 = intra_modes[0];
                let m = match chroma_mode_syntax {
                    0 => 0,
                    1 => 26,
                    2 => 10,
                    3 => 1,
                    _ => luma0,
                };
                if chroma_mode_syntax < 4 && m == luma0 { 34 } else { m }
            } else {
                0
            };
            if rqt_root_cbf {
                let intra_split = intra && part_mode == PartMode::PNxN;
                let max_depth = if intra {
                    self.sps.max_th_depth_intra + intra_split as u32
                } else {
                    self.sps.max_th_depth_inter
                };
                let cu = CuCtx { x0, y0, log2_cb, intra, part_mode, intra_split, max_depth, chroma_mode, bypass, intra_modes };
                self.transform_tree(&cu, x0, y0, x0, y0, log2_cb, 0, 0, [true, true])?;
            } else if intra {
                // Intra CU with no residual still needs its prediction.
                // (rqt_root_cbf is not present for intra CUs: always 1.)
                unreachable!("intra CUs always carry a transform tree");
            }
        }

        if self.trace.cu {
            eprintln!("cu poc={} x={} y={} n={} intra={} skip={} pcm={} bypass={} qp={} part={:?}", self.refs.cur_poc, x0, y0, n, intra, skip, pcm, bypass, self.qp_y, part_mode);
        }
        // Bookkeeping: QP over the CU, filter exemption, PU edges.
        PicInfo::fill4(&mut self.info.qp_y, w4, x0 as usize, y0 as usize, cw, ch, self.qp_y as i8);
        let exempt = ((pcm && self.sps.pcm.4) as u8) | ((bypass as u8) << 1) | (bypass as u8);
        if exempt != 0 {
            PicInfo::fill4(&mut self.info.filter_exempt, w4, x0 as usize, y0 as usize, cw, ch, exempt);
        }
        // Prediction block edges (for deblocking): left/top edge of every PU.
        let pus = part_mode.pus(n);
        for &(px, py, pwid, phei) in pus.iter() {
            let (ax, ay) = ((x0 + px) as usize, (y0 + py) as usize);
            for yy in (ay..(ay + phei as usize).min(ph as usize)).step_by(4) {
                let i = self.info.idx4(ax, yy);
                self.info.edges[i] |= 2;
            }
            for xx in (ax..(ax + pwid as usize).min(pw as usize)).step_by(4) {
                let i = self.info.idx4(xx, ay);
                self.info.edges[i] |= 8;
            }
        }
        // Coding block edges are transform block edges too (marked by the
        // transform tree); a skipped CU has no transform tree, so mark here.
        for yy in (y0 as usize..(y0 as usize + ch)).step_by(4) {
            let i = self.info.idx4(x0 as usize, yy);
            self.info.edges[i] |= 1;
        }
        for xx in (x0 as usize..(x0 as usize + cw)).step_by(4) {
            let i = self.info.idx4(xx, y0 as usize);
            self.info.edges[i] |= 4;
        }
        Ok(())
    }

    fn parse_part_mode(&mut self, intra: bool, log2_cb: u32) -> Result<PartMode> {
        if bin(&mut self.cabac, &mut self.cx, PART_MODE_OFFSET) != 0 {
            return Ok(PartMode::P2Nx2N);
        }
        if intra {
            return Ok(PartMode::PNxN);
        }
        let min = log2_cb == self.sps.log2_min_cb_size;
        if min {
            if bin(&mut self.cabac, &mut self.cx, PART_MODE_OFFSET + 1) != 0 {
                return Ok(PartMode::P2NxN);
            }
            if log2_cb == 3 {
                return Ok(PartMode::PNx2N);
            }
            if bin(&mut self.cabac, &mut self.cx, PART_MODE_OFFSET + 2) != 0 {
                return Ok(PartMode::PNx2N);
            }
            return Ok(PartMode::PNxN);
        }
        if !self.sps.amp_enabled {
            if bin(&mut self.cabac, &mut self.cx, PART_MODE_OFFSET + 1) != 0 {
                return Ok(PartMode::P2NxN);
            }
            return Ok(PartMode::PNx2N);
        }
        if bin(&mut self.cabac, &mut self.cx, PART_MODE_OFFSET + 1) != 0 {
            // Horizontal.
            if bin(&mut self.cabac, &mut self.cx, PART_MODE_OFFSET + 3) != 0 {
                return Ok(PartMode::P2NxN);
            }
            return Ok(if self.cabac.bypass() != 0 { PartMode::P2NxnD } else { PartMode::P2NxnU });
        }
        if bin(&mut self.cabac, &mut self.cx, PART_MODE_OFFSET + 3) != 0 {
            return Ok(PartMode::PNx2N);
        }
        Ok(if self.cabac.bypass() != 0 { PartMode::PnRx2N } else { PartMode::PnLx2N })
    }

    /// The three MPM candidates (8.4.2) for the PU at `(xp, yp)`.
    fn mpm_candidates(&self, xp: i32, yp: i32) -> [u32; 3] {
        let cand = |xn: i32, yn: i32, is_above: bool| -> u32 {
            if !self.avail(xp, yp, xn, yn) {
                return 1;
            }
            let i = self.info.idx4(xn as usize, yn as usize);
            if self.info.pred_mode[i] != 1 {
                return 1; // not intra (or pcm counts as intra with mode DC: pcm CUs store DC)
            }
            if is_above && yn < ((yp >> self.sps.log2_ctb_size) << self.sps.log2_ctb_size) {
                return 1;
            }
            self.info.intra_mode[i] as u32
        };
        let a = cand(xp - 1, yp, false);
        let b = cand(xp, yp - 1, true);
        if a == b {
            if a < 2 {
                [0, 1, 26]
            } else {
                [a, 2 + ((a + 29) % 32), 2 + ((a - 2 + 1) % 32)]
            }
        } else {
            let c = if a != 0 && b != 0 {
                0
            } else if a != 1 && b != 1 {
                1
            } else {
                26
            };
            [a, b, c]
        }
    }

    fn decode_pcm(&mut self, x0: i32, y0: i32, log2_cb: u32) -> Result<()> {
        let n = 1usize << log2_cb;
        let r = self.cabac.reader();
        r.align();
        let (bdy, bdc) = (self.sps.pcm.0, self.sps.pcm.1);
        let shift_y = self.sps.bit_depth_luma - bdy;
        let shift_c = self.sps.bit_depth_chroma - bdc;
        let stride = self.frame.y.stride;
        let off = self.frame.y.offset(x0 as isize, y0 as isize);
        for y in 0..n {
            for x in 0..n {
                let v = r.bits(bdy) << shift_y;
                self.frame.y.data[off + y * stride + x] = S::from_i32(v as i32);
            }
        }
        if self.frame.chroma == ChromaFormat::Yuv420 {
            let cs = self.frame.cb.stride;
            let coff = self.frame.cb.offset((x0 / 2) as isize, (y0 / 2) as isize);
            for y in 0..n / 2 {
                for x in 0..n / 2 {
                    let v = r.bits(bdc) << shift_c;
                    self.frame.cb.data[coff + y * cs + x] = S::from_i32(v as i32);
                }
            }
            for y in 0..n / 2 {
                for x in 0..n / 2 {
                    let v = r.bits(bdc) << shift_c;
                    self.frame.cr.data[coff + y * cs + x] = S::from_i32(v as i32);
                }
            }
        }
        if r.overrun() {
            return Err(Error::bitstream("PCM samples truncated"));
        }
        self.cabac.reinit();
        // PCM CUs report intra mode DC for their neighbours.
        let w4 = self.info.w4;
        PicInfo::fill4(&mut self.info.intra_mode, w4, x0 as usize, y0 as usize, n, n, 1);
        Ok(())
    }

    // ------------------------------------------------------------------
    // Prediction units
    // ------------------------------------------------------------------

    #[allow(clippy::too_many_arguments)]
    fn prediction_unit(&mut self, x_cb: i32, y_cb: i32, n_cb: i32, x_pb: i32, y_pb: i32, w: i32, h: i32, part_idx: u32, skip: bool) -> Result<()> {
        let pu = PuPos { x_cb, y_cb, n_cb, x_pb, y_pb, w, h, part_idx };
        // TMVP reads the collocated picture's motion within this CTB row.
        if let Some(col) = self.col_shared {
            let row_end = ((y_cb >> self.sps.log2_ctb_size) + 1) << self.sps.log2_ctb_size;
            col.progress.wait_decoded(row_end.min(self.frame.height as i32));
        }
        let cand: Cand;
        let mut merged = false;
        if skip {
            let idx = self.parse_merge_idx();
            cand = merge_candidate(self.info, self.frame, &self.refs, &pu, idx);
            merged = true;
        } else {
            let merge_flag = bin(&mut self.cabac, &mut self.cx, MERGE_FLAG_OFFSET) != 0;
            if merge_flag {
                let idx = self.parse_merge_idx();
                cand = merge_candidate(self.info, self.frame, &self.refs, &pu, idx);
                merged = true;
            } else {
                // inter_pred_idc
                let mut pred_idc = 0u32; // 0 L0, 1 L1, 2 BI
                if self.hdr.slice_type == SliceType::B {
                    if w + h != 12 {
                        let depth = self.info.ct_depth[self.info.idx4(x_cb as usize, y_cb as usize)] as usize;
                        if bin(&mut self.cabac, &mut self.cx, INTER_PRED_IDC_OFFSET + depth) != 0 {
                            pred_idc = 2;
                        } else {
                            pred_idc = bin(&mut self.cabac, &mut self.cx, INTER_PRED_IDC_OFFSET + 4);
                        }
                    } else {
                        pred_idc = bin(&mut self.cabac, &mut self.cx, INTER_PRED_IDC_OFFSET + 4);
                    }
                }
                let mut c = Cand { mv: [Mv::ZERO; 2], ref_idx: [-1; 2] };
                let mut mvds = [Mv::ZERO; 2];
                let mut mvp_flags = [0u32; 2];
                for list in 0..2 {
                    let uses = match pred_idc {
                        0 => list == 0,
                        1 => list == 1,
                        _ => true,
                    };
                    if !uses {
                        continue;
                    }
                    let nref = self.hdr.num_ref_idx[list];
                    let ri = if nref > 1 { self.parse_ref_idx(nref) } else { 0 };
                    c.ref_idx[list] = ri as i8;
                    if list == 1 && self.hdr.mvd_l1_zero && pred_idc == 2 {
                        mvds[1] = Mv::ZERO;
                    } else {
                        mvds[list] = self.parse_mvd()?;
                    }
                    mvp_flags[list] = bin(&mut self.cabac, &mut self.cx, MVP_LX_FLAG_OFFSET);
                }
                for list in 0..2 {
                    if c.ref_idx[list] < 0 {
                        continue;
                    }
                    if c.ref_idx[list] as usize >= self.ref_frames[list].len() {
                        return Err(Error::bitstream("ref_idx beyond the reference list"));
                    }
                    let mvp = amvp(self.info, self.frame, &self.refs, &pu, list, c.ref_idx[list], mvp_flags[list]);
                    // uLX = (mvpLX + mvdLX + 2^16) % 2^16 -> wrapping i16 add.
                    c.mv[list] = Mv::new(mvp.x.wrapping_add(mvds[list].x), mvp.y.wrapping_add(mvds[list].y));
                }
                cand = c;
            }
        }
        self.last_pu_merged = merged;
        // Store motion.
        let mut mi = MotionInfo { mv: cand.mv, ref_idx: cand.ref_idx, ref_poc: [0; 2], ref_long_term: [false; 2], intra: false };
        for list in 0..2 {
            if cand.ref_idx[list] >= 0 {
                let ri = cand.ref_idx[list] as usize;
                if ri >= self.refs.pocs[list].len() {
                    return Err(Error::bitstream("merge candidate references beyond the list"));
                }
                mi.ref_poc[list] = self.refs.pocs[list][ri];
                mi.ref_long_term[list] = self.refs.long_term[list][ri];
            }
        }
        let (pw, ph) = (self.frame.width as i32, self.frame.height as i32);
        let cw = w.min(pw - x_pb) as usize;
        let ch = h.min(ph - y_pb) as usize;
        {
            let w4 = self.frame.w4;
            let (bx0, bx1) = (x_pb as usize >> 2, (x_pb as usize + cw) >> 2);
            for by in (y_pb as usize >> 2)..((y_pb as usize + ch) >> 2) {
                self.frame.motion[by * w4 + bx0..by * w4 + bx1].fill(mi);
            }
        }
        // Motion compensation: wait for the reference rows the filters reach
        // (8-tap luma: 3 above / 4 below; 4-tap chroma: 1 / 2, in luma rows).
        let pic_h = self.frame.height as i32;
        for list in 0..2 {
            if cand.ref_idx[list] < 0 {
                continue;
            }
            let mv = cand.mv[list];
            let yi = y_pb + (mv.y as i32 >> 2);
            let need_l = yi + h + 4;
            let yci = (y_pb >> 1) + (mv.y as i32 >> 3);
            let need_c = 2 * (yci + (h >> 1) + 2);
            // Reads above the picture clamp to (or pad from) row 0, which is
            // only ready once row 0 is published; reads below need it all.
            let need = need_l.max(need_c).clamp(1, pic_h);
            self.ref_shared[list][cand.ref_idx[list] as usize].progress.wait_done(need);
        }
        let weighting = self.weighting_for(cand.ref_idx);
        if let Some((tx, ty)) = self.trace.pu {
            if tx >= x_pb && tx < x_pb + w && ty >= y_pb && ty < y_pb + h {
                eprintln!(
                    "pu poc={} x={x_pb} y={y_pb} w={w} h={h} merged={merged} mv={:?} ref_idx={:?} ref_poc={:?} weighting={:?}",
                    self.refs.cur_poc, cand.mv, cand.ref_idx, mi.ref_poc, weighting
                );
            }
        }
        let f0 = if cand.ref_idx[0] >= 0 { Some((self.ref_frames[0][cand.ref_idx[0] as usize], cand.mv[0])) } else { None };
        let f1 = if cand.ref_idx[1] >= 0 { Some((self.ref_frames[1][cand.ref_idx[1] as usize], cand.mv[1])) } else { None };
        // Blocks may extend past the picture edge (the last CTB row/col);
        // predict the whole PB — the border absorbs it.
        predict_block(&self.dsp, &mut self.mc, self.frame, x_pb as usize, y_pb as usize, w as usize, h as usize, f0, f1, weighting);
        Ok(())
    }

    fn weighting_for(&self, ref_idx: [i8; 2]) -> [Weighting; 3] {
        let explicit = match self.hdr.slice_type {
            SliceType::P => self.pps.weighted_pred,
            SliceType::B => self.pps.weighted_bipred,
            SliceType::I => false,
        };
        let Some(t) = (if explicit { self.hdr.pred_weights.as_ref() } else { None }) else {
            return [Weighting::Default; 3];
        };
        let shift1 = 14 - self.bit_depth() as i32;
        let mut out = [Weighting::Default; 3];
        for c in 0..3 {
            let log2_wd = if c == 0 { t.luma_log2_denom as i32 } else { t.chroma_log2_denom as i32 } + shift1;
            let mut w = [1i32; 2];
            let mut o = [0i32; 2];
            for list in 0..2 {
                if ref_idx[list] < 0 {
                    continue;
                }
                let e = &t.lists[list][ref_idx[list] as usize];
                if c == 0 {
                    w[list] = e.luma.0;
                    o[list] = e.luma.1;
                } else {
                    w[list] = e.chroma[c - 1].0;
                    o[list] = e.chroma[c - 1].1;
                }
            }
            out[c] = Weighting::Explicit { log2_wd, w, o };
        }
        out
    }

    fn parse_merge_idx(&mut self) -> usize {
        let max = self.hdr.max_num_merge_cand as usize;
        if max <= 1 {
            return 0;
        }
        let mut idx = 0usize;
        if bin(&mut self.cabac, &mut self.cx, MERGE_IDX_OFFSET) != 0 {
            idx = 1;
            while idx < max - 1 && self.cabac.bypass() != 0 {
                idx += 1;
            }
        }
        idx
    }

    fn parse_ref_idx(&mut self, nref: u32) -> u32 {
        let cmax = nref - 1;
        let mut v = 0u32;
        while v < cmax {
            let b = if v < 2 { bin(&mut self.cabac, &mut self.cx, REF_IDX_L0_OFFSET + v as usize) } else { self.cabac.bypass() };
            if b == 0 {
                break;
            }
            v += 1;
        }
        v
    }

    fn parse_mvd(&mut self) -> Result<Mv> {
        let g0x = bin(&mut self.cabac, &mut self.cx, ABS_MVD_GREATER0_FLAG_OFFSET) != 0;
        let g0y = bin(&mut self.cabac, &mut self.cx, ABS_MVD_GREATER0_FLAG_OFFSET) != 0;
        // The generated table (FFmpeg element order) keeps greater1 at slot +1.
        let g1x = if g0x { bin(&mut self.cabac, &mut self.cx, ABS_MVD_GREATER1_FLAG_OFFSET + 1) != 0 } else { false };
        let g1y = if g0y { bin(&mut self.cabac, &mut self.cx, ABS_MVD_GREATER1_FLAG_OFFSET + 1) != 0 } else { false };
        let mut out = [0i32; 2];
        for (i, (g0, g1)) in [(g0x, g1x), (g0y, g1y)].iter().enumerate() {
            if !g0 {
                continue;
            }
            let mut abs = 1i32;
            if *g1 {
                // abs_mvd_minus2: EG1 bypass.
                let mut k = 1u32;
                let mut v = 0i32;
                loop {
                    if self.cabac.bypass() != 0 {
                        v += 1 << k;
                        k += 1;
                        if k > 24 {
                            return Err(Error::bitstream("mvd runaway"));
                        }
                    } else {
                        break;
                    }
                }
                v += self.cabac.bypass_bits(k) as i32;
                abs = v + 2;
            }
            let sign = self.cabac.bypass();
            out[i] = if sign != 0 { -abs } else { abs };
            if !(-32768..=32767).contains(&out[i]) {
                return Err(Error::bitstream("mvd out of range"));
            }
        }
        Ok(Mv::new(out[0] as i16, out[1] as i16))
    }

    // ------------------------------------------------------------------
    // Transform tree
    // ------------------------------------------------------------------

    #[allow(clippy::too_many_arguments)]
    fn transform_tree(
        &mut self,
        cu: &CuCtx,
        x0: i32,
        y0: i32,
        x_base: i32,
        y_base: i32,
        log2: u32,
        depth: u32,
        blk_idx: u32,
        parent_cbf_c: [bool; 2],
    ) -> Result<()> {
        let split = if log2 <= self.sps.log2_max_tb_size
            && log2 > self.sps.log2_min_tb_size
            && depth < cu.max_depth
            && !(cu.intra_split && depth == 0)
        {
            bin(&mut self.cabac, &mut self.cx, SPLIT_TRANSFORM_FLAG_OFFSET + (5 - log2) as usize) != 0
        } else {
            let inter_split = self.sps.max_th_depth_inter == 0 && !cu.intra && cu.part_mode != PartMode::P2Nx2N && depth == 0;
            log2 > self.sps.log2_max_tb_size || (cu.intra_split && depth == 0) || inter_split
        };
        let mut cbf_c = [false; 2];
        if log2 > 2 && self.sps.chroma_format_idc != 0 {
            for c in 0..2 {
                if depth == 0 || parent_cbf_c[c] {
                    cbf_c[c] = bin(&mut self.cabac, &mut self.cx, CBF_CB_CR_OFFSET + depth as usize) != 0;
                }
            }
        } else if log2 == 2 && self.sps.chroma_format_idc != 0 {
            // Inferred from the parent (chroma is coded at the parent size).
            cbf_c = parent_cbf_c;
        }
        if split {
            let half = 1i32 << (log2 - 1);
            self.transform_tree(cu, x0, y0, x0, y0, log2 - 1, depth + 1, 0, cbf_c)?;
            self.transform_tree(cu, x0 + half, y0, x0, y0, log2 - 1, depth + 1, 1, cbf_c)?;
            self.transform_tree(cu, x0, y0 + half, x0, y0, log2 - 1, depth + 1, 2, cbf_c)?;
            self.transform_tree(cu, x0 + half, y0 + half, x0, y0, log2 - 1, depth + 1, 3, cbf_c)?;
            return Ok(());
        }
        let cbf_luma = if cu.intra || depth != 0 || cbf_c[0] || cbf_c[1] {
            bin(&mut self.cabac, &mut self.cx, CBF_LUMA_OFFSET + (depth == 0) as usize) != 0
        } else {
            true
        };
        self.transform_unit(cu, x0, y0, x_base, y_base, log2, depth, blk_idx, cbf_luma, cbf_c)
    }

    #[allow(clippy::too_many_arguments)]
    fn transform_unit(
        &mut self,
        cu: &CuCtx,
        x0: i32,
        y0: i32,
        x_base: i32,
        y_base: i32,
        log2: u32,
        depth: u32,
        blk_idx: u32,
        cbf_luma: bool,
        cbf_c: [bool; 2],
    ) -> Result<()> {
        let n = 1usize << log2;
        let (pw, ph) = (self.frame.width as usize, self.frame.height as usize);
        // Transform block edges + cbf for the deblocking filter.
        {
            let w4 = self.info.w4;
            let _ = w4;
            for yy in (y0 as usize..(y0 as usize + n).min(ph)).step_by(4) {
                let i = self.info.idx4(x0 as usize, yy);
                self.info.edges[i] |= 1;
            }
            for xx in (x0 as usize..(x0 as usize + n).min(pw)).step_by(4) {
                let i = self.info.idx4(xx, y0 as usize);
                self.info.edges[i] |= 4;
            }
            if cbf_luma {
                let w4 = self.info.w4;
                PicInfo::fill4(&mut self.info.cbf_luma, w4, x0 as usize, y0 as usize, n.min(pw - x0 as usize), n.min(ph - y0 as usize), 1);
            }
        }
        let cbf_chroma = cbf_c[0] || cbf_c[1];
        if (cbf_luma || cbf_chroma) && self.pps.cu_qp_delta_enabled && !self.is_cu_qp_delta_coded {
            // cu_qp_delta_abs / sign.
            let mut prefix = 0u32;
            let mut ctx = CU_QP_DELTA_OFFSET;
            while prefix < 5 && bin(&mut self.cabac, &mut self.cx, ctx) != 0 {
                prefix += 1;
                ctx = CU_QP_DELTA_OFFSET + 1;
            }
            let mut val = prefix as i32;
            if prefix >= 5 {
                let mut k = 0u32;
                let mut v = 0i32;
                loop {
                    if self.cabac.bypass() != 0 {
                        v += 1 << k;
                        k += 1;
                        if k > 24 {
                            return Err(Error::bitstream("cu_qp_delta runaway"));
                        }
                    } else {
                        break;
                    }
                }
                v += self.cabac.bypass_bits(k) as i32;
                val += v;
            }
            if val != 0 && self.cabac.bypass() != 0 {
                val = -val;
            }
            self.is_cu_qp_delta_coded = true;
            self.cu_qp_delta_val = val;
            let bd_off = 6 * (self.sps.bit_depth_luma as i32 - 8);
            if val < -(26 + bd_off / 2) || val > 25 + bd_off / 2 {
                return Err(Error::bitstream("CuQpDeltaVal out of range"));
            }
            self.set_qp(cu.x0, cu.y0);
        }

        // Luma: intra prediction (per TB) then residual.
        if cu.intra {
            let mode = self.info.intra_mode[self.info.idx4(x0 as usize, y0 as usize)] as u32;
            self.intra_predict_block(0, x0 as usize, y0 as usize, n, mode);
        }
        if cbf_luma {
            let mode = if cu.intra { self.info.intra_mode[self.info.idx4(x0 as usize, y0 as usize)] as u32 } else { 0 };
            self.residual_block(cu, x0 as usize, y0 as usize, log2, 0, mode)?;
        }
        // Chroma (4:2:0).
        if self.sps.chroma_format_idc == 1 {
            if log2 > 2 {
                let (xc, yc, nc) = ((x0 / 2) as usize, (y0 / 2) as usize, n / 2);
                for c in 0..2usize {
                    if cu.intra {
                        self.intra_predict_block(1 + c, xc, yc, nc, cu.chroma_mode);
                    }
                    if cbf_c[c] {
                        self.residual_block(cu, xc, yc, log2 - 1, 1 + c, cu.chroma_mode)?;
                    }
                }
            } else if blk_idx == 3 {
                let (xc, yc) = ((x_base / 2) as usize, (y_base / 2) as usize);
                for c in 0..2usize {
                    if cu.intra {
                        self.intra_predict_block(1 + c, xc, yc, 4, cu.chroma_mode);
                    }
                    if cbf_c[c] {
                        self.residual_block(cu, xc, yc, 2, 1 + c, cu.chroma_mode)?;
                    }
                }
            }
        }
        let _ = depth;
        Ok(())
    }

    /// Intra prediction of one transform block of component `c_idx` at
    /// component coordinates `(x, y)`, size `n`.
    fn intra_predict_block(&mut self, c_idx: usize, x: usize, y: usize, n: usize, mode: u32) {
        // Availability of the neighbouring samples in luma coordinates.
        let scale = if c_idx == 0 { 1 } else { 2 };
        let xl = (x * scale) as i32;
        let yl = (y * scale) as i32;
        let cip = self.pps.constrained_intra_pred;
        let mut av = RefAvail { corner: false, left: [false; 64], top: [false; 64] };
        let check = |s: &Self, xn: i32, yn: i32| -> bool {
            if !s.info.available(xl, yl, xn, yn, s.frame.width as i32, s.frame.height as i32) {
                return false;
            }
            !cip || s.info.pred_mode[s.info.idx4(xn as usize, yn as usize)] == 1
        };
        av.corner = check(self, xl - 1, yl - 1);
        // Left samples y = 0..2n and top samples x = 0..2n, in units of 4x4
        // luma blocks. Without constrained intra prediction, availability is
        // uniform over each aligned n-block of neighbours (the left n-block,
        // the below-left n-block: z-scan order decides for the whole block)
        // as long as it lies inside the picture — one check per half.
        let unit = 4 / scale; // samples per 4x4 luma block along the edge
        let (pw, ph) = (self.frame.width as i32, self.frame.height as i32);
        let span = (n * scale) as i32;
        let mut side = |vertical: bool, av_side: &mut [bool; 64]| {
            for half in 0..2 {
                let start = half * n; // in component samples
                let (nx, ny) = if vertical { (xl - 1, yl + half as i32 * span) } else { (xl + half as i32 * span, yl - 1) };
                let inside = if vertical { ny + span <= ph } else { nx + span <= pw };
                if !cip && inside {
                    let a = check(self, nx, ny);
                    for k in 0..n {
                        if start + k < 64 {
                            av_side[start + k] = a;
                        }
                    }
                } else {
                    let mut i = 0;
                    while i < n {
                        let (cx, cy) = if vertical { (nx, ny + (i * scale) as i32) } else { (nx + (i * scale) as i32, ny) };
                        let a = check(self, cx, cy);
                        for k in 0..unit {
                            if start + i + k < 64 {
                                av_side[start + i + k] = a;
                            }
                        }
                        i += unit;
                    }
                }
            }
        };
        side(true, &mut av.left);
        side(false, &mut av.top);
        let bd = self.bit_depth();
        let strong = self.sps.strong_intra_smoothing;
        let plane = match c_idx {
            0 => &mut self.frame.y,
            1 => &mut self.frame.cb,
            _ => &mut self.frame.cr,
        };
        intra_predict(plane, x, y, n, mode, c_idx, bd, strong, &av);
    }

    /// Parse and add the residual of one transform block of component
    /// `c_idx` at component coordinates `(x, y)`.
    fn residual_block(&mut self, cu: &CuCtx, x: usize, y: usize, log2: u32, c_idx: usize, pred_mode: u32) -> Result<()> {
        let n = 1usize << log2;
        // scanIdx (7.4.9.11).
        let scan_idx = if cu.intra && (log2 == 2 || (log2 == 3 && c_idx == 0)) {
            if (6..=14).contains(&pred_mode) {
                2
            } else if (22..=30).contains(&pred_mode) {
                1
            } else {
                0
            }
        } else {
            0
        };
        let params = ResidualParams {
            log2_size: log2,
            c_idx,
            scan_idx,
            bypass: cu.bypass,
            transform_skip_allowed: self.pps.transform_skip_enabled && log2 <= self.pps.log2_max_transform_skip_size,
            sign_hiding: self.pps.sign_data_hiding,
            trace: self.trace.tb_hit(c_idx, x, y, n),
        };
        let mut coeffs = std::mem::take(&mut self.coeffs);
        if coeffs.len() < n * n {
            coeffs.resize(1024, 0);
        }
        let ri = parse_residual(&mut self.cabac, &mut self.cx, &params, &mut coeffs)?;
        let ts = ri.transform_skip;
        // QP for the component.
        let bd_off = 6 * (self.sps.bit_depth_luma as i32 - 8);
        let qp = if c_idx == 0 {
            self.qp_y + bd_off
        } else {
            let off = if c_idx == 1 { self.pps.cb_qp_offset + self.hdr.cb_qp_offset } else { self.pps.cr_qp_offset + self.hdr.cr_qp_offset };
            let qpi = (self.qp_y + off).clamp(-bd_off, 57);
            let qpc = chroma_qp_420(qpi);
            qpc + bd_off
        };
        let bd = if c_idx == 0 { self.sps.bit_depth_luma } else { self.sps.bit_depth_chroma };
        if !cu.bypass {
            let scaling = match &self.scaling {
                None => ScalingSource::Flat,
                Some(sl) => {
                    let size_id = (log2 - 2) as usize;
                    let matrix_id = if cu.intra { c_idx } else { 3 + c_idx };
                    let list = &sl.lists[size_id][matrix_id];
                    let dc = if size_id >= 2 { sl.dc[size_id - 2][matrix_id] } else { 16 };
                    ScalingSource::List(&list[..if size_id == 0 { 16 } else { 64 }], dc)
                }
            };
            scale_coefficients(&mut coeffs, log2, qp, bd, scaling, ts, ri.max_x, ri.max_y);
            let bd_shift = 20 - bd as i32;
            if ts {
                transform_skip_residual(&mut coeffs, log2, bd);
            } else if cu.intra && log2 == 2 && c_idx == 0 {
                (self.dsp.idst4)(&mut coeffs, bd_shift, ri.max_x, ri.max_y);
            } else {
                (self.dsp.idct[(log2 - 2) as usize])(&mut coeffs, bd_shift, ri.max_x, ri.max_y);
            }
        }
        // Add to the prediction.
        let max = (1i32 << bd) - 1;
        let plane = match c_idx {
            0 => &mut self.frame.y,
            1 => &mut self.frame.cb,
            _ => &mut self.frame.cr,
        };
        let stride = plane.stride;
        let off = plane.offset(x as isize, y as isize);
        if params.trace {
            eprintln!("tb c={c_idx} x={x} y={y} n={n} bypass={} ts={ts} qp={qp} scan={scan_idx}", cu.bypass);
            for yy in 0..n {
                let pred: Vec<i32> = (0..n).map(|xx| plane.data[off + yy * stride + xx].to_i32()).collect();
                let res: Vec<i16> = (0..n).map(|xx| coeffs[yy * n + xx]).collect();
                eprintln!("  pred {pred:?} res {res:?}");
            }
        }
        (self.dsp.add_residual)(&mut plane.data[off..], stride, &coeffs, n, max);
        self.coeffs = coeffs;
        Ok(())
    }
}

/// `QpC` as a function of `qPi` for ChromaArrayType 1 (Table 8-10).
pub fn chroma_qp_420(qpi: i32) -> i32 {
    if qpi < 30 {
        qpi
    } else if qpi >= 43 {
        qpi - 6
    } else {
        const T: [i32; 14] = [29, 30, 31, 32, 33, 33, 34, 34, 35, 35, 36, 36, 37, 37];
        T[(qpi - 30) as usize]
    }
}

/// The coding unit facts the transform tree needs.
pub struct CuCtx {
    /// CB position.
    pub x0: i32,
    /// See `x0`.
    pub y0: i32,
    /// `log2CbSize`.
    pub log2_cb: u32,
    /// Intra?
    pub intra: bool,
    /// Partition mode.
    pub part_mode: PartMode,
    /// `IntraSplitFlag`.
    pub intra_split: bool,
    /// `MaxTrafoDepth`.
    pub max_depth: u32,
    /// `IntraPredModeC`.
    pub chroma_mode: u32,
    /// `cu_transquant_bypass_flag`.
    pub bypass: bool,
    /// Luma intra modes per PU (NxN: 4).
    pub intra_modes: [u32; 4],
}
