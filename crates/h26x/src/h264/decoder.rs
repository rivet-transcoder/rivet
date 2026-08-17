//! The H.264 decoder: NAL dispatch, access-unit boundaries, slice decoding
//! (CAVLC or CABAC), picture completion (deblocking, marking, output).

use std::collections::VecDeque;

use crate::bitreader::BitReader;
use crate::cabac::Cabac;
use crate::nal::{H264NalHeader, annexb_nals, unescape_rbsp};
use crate::picture::{ChromaFormat, Picture};
use crate::{Error, Result};

use super::cabac_mb::{CabacState, decode_end_of_slice, decode_mb_skip, parse_mb_cabac};
use super::cavlc::parse_mb_cavlc;
use super::deblock::{DeblockParams, deblock_picture};
use super::dpb::{DecodedPic, Dpb, PocState, RefMark, build_ref_lists, compute_poc};
use super::frame::Frame;
use super::mb::{MbKind, MbLayer, MbNeighbours, PicInfo, SliceCtx};
use super::pps::Pps;
use super::recon::{QpState, SliceRefs, reconstruct};
use super::slice::{Mmco, SliceHeader, SliceType};
use super::sps::{ScalingLists, Sps};
use super::transform::Dequant;

/// The picture being decoded.
struct Current {
    frame: Frame,
    info: PicInfo,
    /// The first slice's header (picture-level facts).
    hdr: SliceHeader,
    sps: Sps,
    pps_id: u32,
    poc: i32,
    slices: Vec<DeblockParams>,
    had_mmco5: bool,
    decode_index: u64,
}

/// A native H.264 decoder.
///
/// Feed Annex-B data with [`H264Decoder::push_annexb`] (or single NAL units
/// with [`H264Decoder::push_nal`]), pull frames in output order with
/// [`H264Decoder::next_picture`], and call [`H264Decoder::flush`] at the end
/// to drain the reorder buffer.
pub struct H264Decoder {
    sps: Vec<Option<Sps>>,
    pps: Vec<Option<Pps>>,
    dpb: Dpb,
    poc_state: PocState,
    cur: Option<Current>,
    dequant_cache: Option<(u32, u32, Dequant)>,
    decode_index: u64,
    grey: Option<Frame>,
    output: VecDeque<Picture>,
    /// Non-fatal problems seen so far (concealed references, dropped slices).
    warnings: u64,
}

impl Default for H264Decoder {
    fn default() -> Self {
        Self::new()
    }
}

impl H264Decoder {
    /// A fresh decoder.
    pub fn new() -> Self {
        H264Decoder {
            sps: (0..32).map(|_| None).collect(),
            pps: (0..256).map(|_| None).collect(),
            dpb: Dpb::new(),
            poc_state: PocState::default(),
            cur: None,
            dequant_cache: None,
            decode_index: 0,
            grey: None,
            output: VecDeque::new(),
            warnings: 0,
        }
    }

    /// Number of non-fatal problems concealed so far.
    pub fn warnings(&self) -> u64 {
        self.warnings
    }

    /// Feed an Annex-B chunk (one or more NAL units with start codes).
    pub fn push_annexb(&mut self, data: &[u8]) -> Result<()> {
        for nal in annexb_nals(data) {
            self.push_nal(nal)?;
        }
        Ok(())
    }

    /// Feed one NAL unit (with its header byte, emulation prevention still
    /// in place).
    pub fn push_nal(&mut self, nal: &[u8]) -> Result<()> {
        let Some(hdr) = H264NalHeader::parse(nal) else {
            return Ok(()); // forbidden bit / empty: ignore
        };
        match hdr.unit_type {
            1 | 5 => {
                let rbsp = unescape_rbsp(nal);
                self.decode_slice(hdr, &rbsp)
            }
            7 => {
                let rbsp = unescape_rbsp(&nal[1..]);
                let sps = Sps::parse(&rbsp)?;
                let id = sps.id as usize;
                // A changed SPS with the same id while a picture is open:
                // finish the picture first.
                if let Some(old) = &self.sps[id] {
                    if *old != sps && self.cur.is_some() {
                        self.finish_picture()?;
                    }
                }
                self.sps[id] = Some(sps);
                Ok(())
            }
            8 => {
                let rbsp = unescape_rbsp(&nal[1..]);
                let sps_tab = &self.sps;
                let pps = Pps::parse(&rbsp, &|id| sps_tab.get(id as usize).cloned().flatten())?;
                let id = pps.id as usize;
                if let Some(old) = &self.pps[id] {
                    if *old != pps && self.cur.is_some() {
                        self.finish_picture()?;
                    }
                }
                self.pps[id] = Some(pps);
                Ok(())
            }
            9 | 10 | 11 => {
                // Access unit delimiter / end of sequence / end of stream:
                // the current picture is complete.
                if self.cur.is_some() {
                    self.finish_picture()?;
                }
                Ok(())
            }
            2 | 3 | 4 => Err(Error::unsupported("H.264 slice data partitioning (nal_unit_type 2..4)")),
            20 | 21 => Ok(()), // SVC/MVC extension slices: ignore (base layer decodes)
            _ => Ok(()),       // SEI, filler, SPS extension, prefix NALs...
        }
    }

    /// End of stream: finish the open picture and drain the reorder buffer.
    pub fn flush(&mut self) -> Result<()> {
        if self.cur.is_some() {
            self.finish_picture()?;
        }
        self.dpb.flush_output();
        self.output.extend(self.dpb.output.drain(..));
        Ok(())
    }

    /// The next decoded picture in output order, if one is ready.
    pub fn next_picture(&mut self) -> Option<Picture> {
        self.output.extend(self.dpb.output.drain(..));
        self.output.pop_front()
    }

    fn check_supported(sps: &Sps, pps: &Pps) -> Result<()> {
        if !sps.frame_mbs_only {
            return Err(Error::unsupported("H.264 interlaced coding (frame_mbs_only_flag = 0)"));
        }
        if sps.chroma_format_idc != 1 {
            return Err(Error::unsupported(format!(
                "H.264 chroma_format_idc {} (only 4:2:0 is implemented)",
                sps.chroma_format_idc
            )));
        }
        if sps.bit_depth_luma != 8 || sps.bit_depth_chroma != 8 {
            return Err(Error::unsupported(format!(
                "H.264 bit depth {}/{} (only 8-bit is implemented)",
                sps.bit_depth_luma, sps.bit_depth_chroma
            )));
        }
        if sps.transform_bypass {
            return Err(Error::unsupported("H.264 lossless transform bypass"));
        }
        if pps.num_slice_groups > 1 {
            return Err(Error::unsupported("H.264 slice groups (FMO)"));
        }
        Ok(())
    }

    /// Whether `hdr` starts a new picture relative to the open one (7.4.1.2.4).
    fn is_new_picture(cur: &Current, hdr: &SliceHeader, sps: &Sps) -> bool {
        let a = &cur.hdr;
        if hdr.first_mb_in_slice == 0 && !(a.frame_num == hdr.frame_num && a.pps_id == hdr.pps_id) {
            return true;
        }
        if a.frame_num != hdr.frame_num || a.pps_id != hdr.pps_id || a.field_pic != hdr.field_pic {
            return true;
        }
        if (a.nal_ref_idc == 0) != (hdr.nal_ref_idc == 0) {
            return true;
        }
        if sps.poc_type == 0 && (a.poc_lsb != hdr.poc_lsb || a.delta_poc_bottom != hdr.delta_poc_bottom) {
            return true;
        }
        if sps.poc_type == 1 && a.delta_poc != hdr.delta_poc {
            return true;
        }
        if a.is_idr() != hdr.is_idr() {
            return true;
        }
        if a.is_idr() && hdr.is_idr() && a.idr_pic_id != hdr.idr_pic_id {
            return true;
        }
        // Same picture unless the slice starts at MB 0 again with the same
        // parameters (a repeated first slice — treat as new to stay safe).
        hdr.first_mb_in_slice == 0 && cur.info.mbs[0].decoded
    }

    fn decode_slice(&mut self, nal: H264NalHeader, rbsp: &[u8]) -> Result<()> {
        let sps_tab = &self.sps;
        let pps_tab = &self.pps;
        let (hdr, pps, sps) = SliceHeader::parse(
            rbsp,
            nal,
            &|id| pps_tab.get(id as usize).cloned().flatten(),
            &|id| sps_tab.get(id as usize).cloned().flatten(),
        )?;
        if hdr.redundant_pic_cnt > 0 {
            return Ok(()); // redundant slice: the primary is enough
        }
        if matches!(hdr.slice_type, SliceType::Sp | SliceType::Si) {
            return Err(Error::unsupported("H.264 SP/SI slices"));
        }
        Self::check_supported(&sps, &pps)?;

        // Picture boundary.
        let new_pic = match &self.cur {
            None => true,
            Some(cur) => Self::is_new_picture(cur, &hdr, &sps),
        };
        if new_pic {
            if self.cur.is_some() {
                self.finish_picture()?;
            }
            self.start_picture(&hdr, &sps, &pps)?;
        }
        let cur = self.cur.as_mut().unwrap();
        if cur.sps != sps {
            // A slice of the same picture referencing a different SPS: refuse.
            return Err(Error::bitstream("slices of one picture reference different SPSs"));
        }

        // Dequantisation tables for this PPS/SPS pair.
        let lists: ScalingLists = pps
            .scaling_lists
            .clone()
            .or_else(|| sps.scaling_lists.clone())
            .unwrap_or_else(ScalingLists::flat);
        let cache_ok = matches!(&self.dequant_cache, Some((s, p, _)) if *s == sps.id && *p == pps.id);
        if !cache_ok {
            self.dequant_cache = Some((sps.id, pps.id, Dequant::new(&lists)));
        }
        let dq: &Dequant = &self.dequant_cache.as_ref().unwrap().2;

        // Reference lists.
        let cur_poc = cur.poc;
        let lists_idx = if hdr.slice_type.is_intra() {
            None
        } else {
            Some(build_ref_lists(&mut self.dpb, &sps, &hdr, cur_poc)?)
        };
        // Grey frame for missing references (concealment).
        if self.grey.is_none() {
            let mut g = Frame::new(cur.frame.mb_width, cur.frame.mb_height, cur.frame.chroma);
            g.y.data.fill(128);
            g.cb.data.fill(128);
            g.cr.data.fill(128);
            g.mb_intra.fill(true);
            self.grey = Some(g);
        }
        let grey = self.grey.as_ref().unwrap();
        let mut refs = SliceRefs {
            frames: [Vec::new(), Vec::new()],
            pocs: [Vec::new(), Vec::new()],
            long_term: [Vec::new(), Vec::new()],
            col: None,
            col_long_term: false,
            explicit: hdr.pred_weights.as_ref(),
            implicit: None,
            cur_poc,
        };
        if let Some(rl) = &lists_idx {
            for l in 0..2 {
                for &i in &rl.lists[l] {
                    if i == usize::MAX || i >= self.dpb.pics.len() {
                        self.warnings += 1;
                        refs.frames[l].push(grey);
                        refs.pocs[l].push(i32::MIN / 2);
                        refs.long_term[l].push(false);
                    } else {
                        let p = &self.dpb.pics[i];
                        refs.frames[l].push(&p.frame);
                        refs.pocs[l].push(p.poc);
                        refs.long_term[l].push(p.mark == RefMark::Long);
                    }
                }
            }
            if hdr.slice_type.is_b() {
                if let Some(&i) = rl.lists[1].first() {
                    if i != usize::MAX && i < self.dpb.pics.len() {
                        refs.col = Some(&self.dpb.pics[i].frame);
                        refs.col_long_term = self.dpb.pics[i].mark == RefMark::Long;
                    }
                }
                if pps.weighted_bipred_idc == 2 {
                    refs.build_implicit();
                }
            }
        }

        let slice_num = cur.slices.len() as u16;
        cur.slices.push(DeblockParams {
            disable_idc: hdr.disable_deblocking_filter_idc,
            offset_a: hdr.filter_offset_a,
            offset_b: hdr.filter_offset_b,
        });
        let ctx = SliceCtx {
            slice_type: hdr.slice_type,
            slice_num,
            num_ref_idx: hdr.num_ref_idx_active,
            direct_spatial: hdr.direct_spatial_mv_pred,
            transform_8x8_mode: pps.transform_8x8_mode,
            constrained_intra_pred: pps.constrained_intra_pred,
            direct_8x8_inference: sps.direct_8x8_inference,
            chroma_format_idc: sps.chroma_format_idc,
        };
        let mut qps = QpState {
            prev_qp: hdr.slice_qp,
            chroma_offset: [pps.chroma_qp_index_offset, pps.second_chroma_qp_index_offset],
        };
        if hdr.marking.ops.iter().any(|o| *o == Mmco::UnmarkAll) {
            cur.had_mmco5 = true;
        }

        let total_mbs = cur.frame.mb_width * cur.frame.mb_height;
        let mut addr = hdr.first_mb_in_slice as usize;
        if addr >= total_mbs {
            return Err(Error::bitstream("first_mb_in_slice beyond the picture"));
        }
        let data_start = (hdr.data_bit_offset / 8) as usize;

        if pps.cabac {
            let mut cabac = Cabac::new(&rbsp[data_start..]);
            let mut st = CabacState::new(hdr.slice_type, hdr.cabac_init_idc, hdr.slice_qp);
            loop {
                if addr >= total_mbs {
                    return Err(Error::bitstream("slice data runs past the picture"));
                }
                let nb = MbNeighbours::derive(&cur.info, addr, slice_num);
                let mut layer: Option<MbLayer> = None;
                if !hdr.slice_type.is_intra() {
                    let skip = decode_mb_skip(&mut cabac, &mut st, &cur.info, &nb, hdr.slice_type.is_b());
                    if skip {
                        let kind = if hdr.slice_type.is_b() { MbKind::BSkip } else { MbKind::PSkip };
                        layer = Some(MbLayer::new(kind));
                        st.prev_qp_delta_nonzero = false;
                    }
                }
                let layer = match layer {
                    Some(l) => l,
                    None => parse_mb_cabac(&mut cabac, &mut st, &ctx, &cur.info, &nb, &cur.frame.motion)?,
                };
                reconstruct(&ctx, &mut qps, dq, &mut cur.frame, &mut cur.info, &nb, &layer, &refs)?;
                addr += 1;
                if decode_end_of_slice(&mut cabac) {
                    break;
                }
                if cabac.overrun() {
                    return Err(Error::bitstream("CABAC slice data exhausted before end_of_slice_flag"));
                }
            }
        } else {
            let mut r = BitReader::new(rbsp);
            r.skip(hdr.data_bit_offset as u32);
            loop {
                if !hdr.slice_type.is_intra() {
                    let run = r.ue() as usize;
                    if run > total_mbs {
                        return Err(Error::bitstream("mb_skip_run out of range"));
                    }
                    for _ in 0..run {
                        if addr >= total_mbs {
                            return Err(Error::bitstream("slice data runs past the picture"));
                        }
                        let nb = MbNeighbours::derive(&cur.info, addr, slice_num);
                        let kind = if hdr.slice_type.is_b() { MbKind::BSkip } else { MbKind::PSkip };
                        let layer = MbLayer::new(kind);
                        reconstruct(&ctx, &mut qps, dq, &mut cur.frame, &mut cur.info, &nb, &layer, &refs)?;
                        addr += 1;
                    }
                    if run > 0 && !r.more_rbsp_data() {
                        break;
                    }
                }
                if addr >= total_mbs {
                    return Err(Error::bitstream("slice data runs past the picture"));
                }
                let nb = MbNeighbours::derive(&cur.info, addr, slice_num);
                let t = r.ue();
                let layer = parse_mb_cavlc(&mut r, &ctx, &cur.info, &nb, t)?;
                reconstruct(&ctx, &mut qps, dq, &mut cur.frame, &mut cur.info, &nb, &layer, &refs)?;
                addr += 1;
                if r.overrun() {
                    return Err(Error::bitstream("CAVLC slice data exhausted"));
                }
                if !r.more_rbsp_data() {
                    break;
                }
            }
        }
        Ok(())
    }

    fn start_picture(&mut self, hdr: &SliceHeader, sps: &Sps, pps: &Pps) -> Result<()> {
        // Activate the SPS: (re)size the DPB and frame buffers.
        let mbw = sps.pic_width_in_mbs as usize;
        let mbh = sps.frame_height_in_mbs() as usize;
        let size_changed = self.dpb.pics.first().is_some_and(|p| p.frame.mb_width != mbw || p.frame.mb_height != mbh);
        if hdr.is_idr() || size_changed {
            if size_changed || hdr.is_idr() {
                // New coded video sequence.
                self.dpb.configure(sps);
            }
            if size_changed {
                self.dpb.flush_output();
                self.dpb.clear();
                self.grey = None;
            }
        }
        if self.dpb.pics.is_empty() {
            self.dpb.configure(sps);
        }
        self.dpb.crop = sps.crop;

        // frame_num gap (8.2.5.2).
        if !hdr.is_idr() {
            let prev = self.poc_state.prev_ref_frame_num;
            let max = sps.max_frame_num();
            if hdr.frame_num != prev && hdr.frame_num != (prev + 1) % max {
                let template = Frame::new(mbw, mbh, ChromaFormat::Yuv420);
                self.warnings += (!sps.gaps_in_frame_num_allowed) as u64;
                self.dpb.fill_frame_num_gap(sps, prev, hdr.frame_num, &template, &mut self.decode_index);
            }
        }

        // POC.
        let (top, bottom) = compute_poc(sps, hdr, &mut self.poc_state);
        let poc = top.min(bottom);

        let frame = Frame::new(mbw, mbh, ChromaFormat::Yuv420);
        let mut info = PicInfo::new(mbw, mbh);
        info.reset();
        self.cur = Some(Current {
            frame,
            info,
            hdr: hdr.clone(),
            sps: sps.clone(),
            pps_id: pps.id,
            poc,
            slices: Vec::new(),
            had_mmco5: false,
            decode_index: self.decode_index,
        });
        self.decode_index += 1;
        Ok(())
    }

    fn finish_picture(&mut self) -> Result<()> {
        let Some(mut cur) = self.cur.take() else { return Ok(()) };
        let sps = cur.sps.clone();
        let hdr = cur.hdr.clone();
        let _ = cur.pps_id;

        // Undecoded macroblocks (a lost slice): conceal by leaving them as
        // they are (zeros) — flag it.
        let missing = cur.info.mbs.iter().filter(|m| !m.decoded).count();
        if missing > 0 {
            self.warnings += 1;
        }

        if std::env::var_os("H26X_NO_DEBLOCK").is_none() {
            deblock_picture(&mut cur.frame, &cur.info, &cur.slices);
        }
        cur.frame.extend_edges();

        // POC / frame_num bookkeeping.
        let mut poc = cur.poc;
        let mut frame_num = hdr.frame_num;
        if cur.had_mmco5 {
            // 8.2.1: after MMCO 5 the picture's POC becomes 0 and frame_num 0.
            let top = poc; // frame: top == bottom for our purposes after subtraction
            self.poc_state.prev_ref_top_poc_after_mmco5 = top - poc;
            poc = 0;
            frame_num = 0;
        }
        self.poc_state.prev_frame_num = hdr.frame_num;
        self.poc_state.prev_had_mmco5 = cur.had_mmco5;
        if hdr.is_reference() {
            self.poc_state.prev_ref_frame_num = frame_num;
            self.poc_state.prev_ref_had_mmco5 = cur.had_mmco5;
            if cur.had_mmco5 {
                self.poc_state.prev_msb = 0;
                self.poc_state.prev_lsb = 0;
            }
        }
        if cur.had_mmco5 {
            self.poc_state.prev_frame_num_offset = 0;
        }
        cur.frame.poc = poc;

        let pic = DecodedPic {
            frame: cur.frame,
            poc,
            frame_num,
            frame_num_wrap: frame_num as i32,
            long_term_frame_idx: 0,
            mark: RefMark::Unused,
            needed_for_output: true,
            non_existing: false,
            decode_index: cur.decode_index,
        };
        self.dpb.store(pic, &hdr, &sps, cur.had_mmco5)?;
        // Long-term flag on the stored frame (for direct mode readers).
        if let Some(p) = self.dpb.pics.last_mut() {
            p.frame.long_term = p.mark == RefMark::Long;
        }
        self.output.extend(self.dpb.output.drain(..));
        Ok(())
    }
}
