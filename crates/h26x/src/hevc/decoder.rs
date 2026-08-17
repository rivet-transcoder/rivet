//! The HEVC decoder: NAL dispatch, picture boundaries, POC and RPS handling,
//! slice segment decoding (substreams, WPP, tiles), picture completion
//! (deblocking, SAO, DPB storage and output).

use std::collections::HashMap;

use crate::cabac::Cabac;
use crate::nal::{HevcNalHeader, annexb_nals, escaped_offset, unescape_rbsp, unescape_rbsp_positions, unescaped_offset};
use crate::picture::{ChromaFormat, Picture};
use crate::{Error, Result};

use super::ctu::SliceDec;
use super::ctx::Contexts;
use super::deblock::deblock_picture;
use super::dpb::{Dpb, DpbPic, RefSets};
use super::frame::Frame;
use super::mvpred::RefCtx;
use super::pic::{PicInfo, SliceFilterParams};
use super::pps::Pps;
use super::sao::sao_picture;
use super::slice::{SliceHeader, SliceType, nal_type};
use super::sps::{ScalingList, Sps, Vps};

/// The picture being decoded.
struct Current {
    frame: Frame,
    info: PicInfo,
    sps: Sps,
    pps: Pps,
    poc: i32,
    pic_output: bool,
    sets: RefSets,
    decode_index: u64,
    /// The independent slice segment header in force (for dependents).
    independent: Option<SliceHeader>,
    /// Contexts saved at the end of the previous slice segment (for a
    /// dependent one continuing it).
    saved_ds: Option<Contexts>,
    /// Contexts saved after the second CTB of a row (WPP), keyed by CTB row.
    saved_wpp: HashMap<usize, Contexts>,
    /// QpY of the last CU decoded (qPY_PREV across dependent segments).
    last_qp_y: i32,
    /// Resolved scaling lists (None = flat).
    scaling: Option<ScalingList>,
    /// The nal type of the picture.
    nal_type: u8,
}

/// A native HEVC (H.265) decoder — Main and Main 10 profiles, 4:2:0.
///
/// The API mirrors [`crate::h264::H264Decoder`]: feed Annex-B data or single
/// NAL units, pull frames in output order, flush at the end.
pub struct HevcDecoder {
    vps: HashMap<u32, Vps>,
    sps: HashMap<u32, Sps>,
    pps: HashMap<u32, Pps>,
    dpb: Dpb,
    cur: Option<Current>,
    /// POC of the previous TemporalId-0 non-RASL/RADL/SLNR picture.
    prev_tid0_poc: i32,
    /// The next IRAP starts a new coded video sequence (start of stream or
    /// after an end-of-sequence NAL).
    first_in_sequence: bool,
    /// NoRaslOutputFlag of the associated IRAP picture.
    no_rasl_output: bool,
    /// The current picture is being skipped (RASL after a NoRaslOutput
    /// IRAP, or undecodable).
    skipping: bool,
    decode_index: u64,
    warnings: u64,
}

impl Default for HevcDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl HevcDecoder {
    /// A fresh decoder.
    pub fn new() -> Self {
        HevcDecoder {
            vps: HashMap::new(),
            sps: HashMap::new(),
            pps: HashMap::new(),
            dpb: Dpb::new(),
            cur: None,
            prev_tid0_poc: 0,
            first_in_sequence: true,
            no_rasl_output: true,
            skipping: false,
            decode_index: 0,
            warnings: 0,
        }
    }

    /// Non-fatal problems seen so far.
    pub fn warnings(&self) -> u64 {
        self.warnings + self.dpb.warnings
    }

    /// Feed a chunk of Annex-B bytes (whole NAL units).
    pub fn push_annexb(&mut self, data: &[u8]) -> Result<()> {
        for nal in annexb_nals(data) {
            self.push_nal(nal)?;
        }
        Ok(())
    }

    /// Feed one NAL unit (with its two header bytes, without start code).
    pub fn push_nal(&mut self, nal: &[u8]) -> Result<()> {
        let Some(hdr) = HevcNalHeader::parse(nal) else {
            return Err(Error::bitstream("bad NAL header"));
        };
        if hdr.layer_id != 0 {
            return Ok(()); // enhancement layers are not decoded
        }
        match hdr.unit_type {
            nal_type::VPS => {
                let rbsp = unescape_rbsp(nal);
                let v = Vps::parse(&rbsp[2..])?;
                self.vps.insert(v.id, v);
            }
            nal_type::SPS => {
                let rbsp = unescape_rbsp(nal);
                let s = Sps::parse(&rbsp[2..])?;
                self.sps.insert(s.id, s);
            }
            nal_type::PPS => {
                let rbsp = unescape_rbsp(nal);
                let p = Pps::parse(&rbsp[2..])?;
                self.pps.insert(p.id, p);
            }
            nal_type::EOS | nal_type::EOB => {
                self.finish_picture()?;
                self.first_in_sequence = true;
            }
            nal_type::AUD | nal_type::SEI_PREFIX | nal_type::SEI_SUFFIX | nal_type::FD => {}
            t if nal_type::is_slice(t) => self.slice_nal(nal, hdr)?,
            _ => {} // reserved / unspecified
        }
        Ok(())
    }

    /// End of stream: finish the current picture and drain the DPB.
    pub fn flush(&mut self) -> Result<()> {
        self.finish_picture()?;
        self.dpb.flush();
        Ok(())
    }

    /// The next picture in output order, if any.
    pub fn next_picture(&mut self) -> Option<Picture> {
        self.dpb.output.pop_front()
    }

    fn slice_nal(&mut self, nal: &[u8], nh: HevcNalHeader) -> Result<()> {
        let (rbsp, removed) = unescape_rbsp_positions(nal);
        let first_flag = rbsp.get(2).is_some_and(|b| b & 0x80 != 0);
        if first_flag {
            self.finish_picture()?;
            self.skipping = false;
        } else if self.cur.is_none() && !self.skipping {
            self.warnings += 1;
            return Ok(()); // slice of a picture whose first segment was lost
        }
        if self.skipping && !first_flag {
            return Ok(());
        }
        let sps_map = &self.sps;
        let pps_map = &self.pps;
        let independent = self.cur.as_ref().and_then(|c| c.independent.as_ref());
        let (hdr, mut pps, sps) = SliceHeader::parse(
            &rbsp,
            nh,
            &|id| pps_map.get(&id).cloned(),
            &|id| sps_map.get(&id).cloned(),
            independent,
        )?;
        if first_flag {
            // Constraints of this decoder.
            if sps.chroma_format_idc != 1 || sps.separate_colour_plane {
                return Err(Error::unsupported(format!("chroma_format_idc {} (only 4:2:0)", sps.chroma_format_idc)));
            }
            if sps.bit_depth_luma != sps.bit_depth_chroma {
                return Err(Error::unsupported("different luma and chroma bit depths"));
            }
            if sps.bit_depth_luma > 12 {
                return Err(Error::unsupported(format!("bit depth {}", sps.bit_depth_luma)));
            }
            if let Some(ext) = &sps.range_ext {
                if ext.iter().any(|&f| f) {
                    return Err(Error::unsupported("range extension tools"));
                }
            }
            if pps.range_ext {
                return Err(Error::unsupported("PPS range extension"));
            }
            pps.resolve_tiles(&sps)?;
            self.start_picture(&hdr, sps, pps, nh)?;
            if self.skipping {
                return Ok(());
            }
        }
        // Dependent segments need the PPS/SPS of the picture (same ids).
        let nal_data = rbsp;
        self.decode_slice_segment(&hdr, &nal_data, &removed)
    }

    fn start_picture(&mut self, hdr: &SliceHeader, sps: Sps, pps: Pps, nh: HevcNalHeader) -> Result<()> {
        let t = nh.unit_type;
        let irap = nal_type::is_irap(t);
        if irap {
            self.no_rasl_output = nal_type::is_idr(t) || nal_type::is_bla(t) || self.first_in_sequence;
        }
        if nal_type::is_rasl(t) && self.no_rasl_output {
            // Not decodable (its references precede the random access
            // point) and not output: skip.
            self.skipping = true;
            return Ok(());
        }
        // POC (8.3.1).
        let max_poc_lsb = sps.max_poc_lsb();
        let lsb = hdr.poc_lsb as i32;
        let msb = if irap && self.no_rasl_output {
            0
        } else {
            let prev_lsb = self.prev_tid0_poc & (max_poc_lsb - 1);
            let prev_msb = self.prev_tid0_poc - prev_lsb;
            if lsb < prev_lsb && (prev_lsb - lsb) >= max_poc_lsb / 2 {
                prev_msb + max_poc_lsb
            } else if lsb > prev_lsb && (lsb - prev_lsb) > max_poc_lsb / 2 {
                prev_msb - max_poc_lsb
            } else {
                prev_msb
            }
        };
        let poc = msb + lsb;
        if nh.temporal_id == 0 && !nal_type::is_rasl(t) && !nal_type::is_radl(t) && !nal_type::is_sub_layer_non_ref(t) {
            self.prev_tid0_poc = poc;
        }
        let first_pic = self.decode_index == 0;
        self.first_in_sequence = false;

        // The DPB adopts the SPS limits.
        self.dpb.configure(&sps);
        let chroma = ChromaFormat::Yuv420;
        let bit_depth = sps.bit_depth_luma;
        let crop = sps.conf_win;
        // RPS + marking + missing references.
        let idr = nal_type::is_idr(t);
        let sets = self.dpb.apply_rps(hdr, &sps, poc, idr, chroma, bit_depth, self.decode_index, crop);
        // C.5.2.2.
        if irap && self.no_rasl_output && !first_pic {
            let no_output = if t == nal_type::CRA { true } else { hdr.no_output_of_prior_pics };
            self.dpb.before_decode(true, no_output);
        } else {
            self.dpb.before_decode(false, false);
        }
        // pic_output_flag: RASL with NoRaslOutputFlag are skipped above.
        let pic_output = hdr.pic_output;

        let frame = Frame::new(sps.width as usize, sps.height as usize, chroma, bit_depth);
        let info = PicInfo::new(&sps, &pps);
        let scaling = if sps.scaling_list_enabled {
            Some(match (&pps.scaling_list, &sps.scaling_list) {
                (Some(p), _) => p.clone(),
                (None, Some(s)) => s.clone(),
                (None, None) => ScalingList::default_lists(),
            })
        } else {
            None
        };
        self.cur = Some(Current {
            frame,
            info,
            sps,
            pps,
            poc,
            pic_output,
            sets,
            decode_index: self.decode_index,
            independent: None,
            saved_ds: None,
            saved_wpp: HashMap::new(),
            last_qp_y: 0,
            scaling,
            nal_type: t,
        });
        self.decode_index += 1;
        Ok(())
    }

    fn decode_slice_segment(&mut self, hdr: &SliceHeader, rbsp: &[u8], removed: &[usize]) -> Result<()> {
        let Some(cur) = self.cur.as_mut() else { return Ok(()) };
        let Current { frame, info, sps, pps, poc, sets, independent, saved_ds, saved_wpp, last_qp_y, scaling, .. } = cur;
        let sps: &Sps = sps;
        let pps: &Pps = pps;
        let cur_poc = *poc;
        // The slice-level header this segment belongs to.
        if !hdr.dependent {
            *independent = Some(hdr.clone());
            info.slices.push(SliceFilterParams {
                deblocking_disabled: hdr.deblocking_disabled,
                beta_offset: hdr.beta_offset,
                tc_offset: hdr.tc_offset,
                loop_filter_across_slices: hdr.loop_filter_across_slices,
                slice_addr: hdr.segment_address,
                cb_qp_offset: pps.cb_qp_offset,
                cr_qp_offset: pps.cr_qp_offset,
            });
        }
        let Some(ind) = independent.as_ref() else {
            self.warnings += 1;
            return Ok(());
        };
        let slice_idx = (info.slices.len() - 1) as u16;
        let slice_addr = ind.segment_address;
        // Reference picture lists.
        let lists = if ind.slice_type != SliceType::I { self.dpb.build_ref_lists(ind, sets)? } else { [Vec::new(), Vec::new()] };
        let ref_frames: [Vec<&Frame>; 2] = [
            lists[0].iter().map(|&i| &self.dpb.pics[i].frame).collect(),
            lists[1].iter().map(|&i| &self.dpb.pics[i].frame).collect(),
        ];
        let pocs: [Vec<i32>; 2] = [lists[0].iter().map(|&i| self.dpb.pics[i].poc).collect(), lists[1].iter().map(|&i| self.dpb.pics[i].poc).collect()];
        let long_term: [Vec<bool>; 2] =
            [lists[0].iter().map(|&i| self.dpb.pics[i].long_term).collect(), lists[1].iter().map(|&i| self.dpb.pics[i].long_term).collect()];
        let no_backward_pred = pocs[0].iter().chain(pocs[1].iter()).all(|&p| p <= cur_poc);
        let col = if ind.temporal_mvp_enabled && ind.slice_type != SliceType::I {
            let list = if ind.slice_type == SliceType::B && !ind.collocated_from_l0 { 1 } else { 0 };
            match ref_frames[list].get(ind.collocated_ref_idx as usize) {
                Some(f) => Some(*f),
                None => return Err(Error::bitstream("collocated_ref_idx out of range")),
            }
        } else {
            None
        };
        let refs = RefCtx {
            pocs,
            long_term,
            col,
            cur_poc,
            no_backward_pred,
            tmvp: ind.temporal_mvp_enabled,
            max_merge_cand: ind.max_num_merge_cand as usize,
            log2_par_mrg_level: pps.log2_parallel_merge_level,
            is_b: ind.slice_type == SliceType::B,
            num_ref_idx: [ind.num_ref_idx[0] as usize, ind.num_ref_idx[1] as usize],
            col_from_l0: ind.collocated_from_l0,
        };

        // Substreams: byte offsets in the escaped NAL, relative to the start
        // of the slice segment data.
        let data_start_unesc = (hdr.data_bit_offset / 8) as usize;
        let data_start_esc = escaped_offset(data_start_unesc, removed);
        let mut substreams: Vec<usize> = vec![data_start_unesc.min(rbsp.len())];
        for &ep in &hdr.entry_points {
            let esc = data_start_esc + ep as usize;
            substreams.push(unescaped_offset(esc, removed).min(rbsp.len()));
        }
        let init_type = match ind.slice_type {
            SliceType::I => 0,
            SliceType::P => {
                if ind.cabac_init {
                    2
                } else {
                    1
                }
            }
            SliceType::B => {
                if ind.cabac_init {
                    1
                } else {
                    2
                }
            }
        };
        let wc = info.wc;
        let n_ctbs = info.wc * info.hc;
        let mut ctb_addr_rs = hdr.segment_address as usize;
        if ctb_addr_rs >= n_ctbs {
            return Err(Error::bitstream("slice_segment_address out of range"));
        }
        let mut ctb_addr_ts = info.ctb_rs_to_ts[ctb_addr_rs] as usize;
        let mut sub = 0usize;
        let cabac = Cabac::new(&rbsp[substreams[0]..]);
        // First CTB column of the tile containing a CTB.
        let tile_col_start = |rs: usize| -> usize {
            let rx = rs % wc;
            let mut start = 0;
            for &b in &pps.col_bd {
                if (b as usize) <= rx {
                    start = b as usize;
                }
            }
            start
        };

        // Contexts at the start of the segment (9.3.1).
        let first_in_tile = ctb_addr_ts == 0 || info.tile_id_ts[ctb_addr_ts] != info.tile_id_ts[ctb_addr_ts - 1];
        let row_start = pps.entropy_coding_sync && ctb_addr_rs % wc == tile_col_start(ctb_addr_rs);
        let mut cx = Contexts::new(init_type, ind.slice_qp);
        let mut first_qg = true;
        let mut qp_prev_init = ind.slice_qp;
        if first_in_tile {
            // init
        } else if row_start {
            if let Some(saved) = wpp_sync_source(info, saved_wpp, ctb_addr_rs, slice_addr) {
                cx = saved.clone();
            }
        } else if hdr.dependent {
            if let Some(saved) = saved_ds.as_ref() {
                cx = saved.clone();
            }
            first_qg = false;
            qp_prev_init = *last_qp_y;
        }

        let mut dec = SliceDec {
            sps,
            pps,
            hdr: ind,
            frame,
            info,
            cabac,
            cx,
            refs,
            ref_frames,
            slice_idx,
            slice_addr,
            scaling: scaling.clone(),
            qp_y: ind.slice_qp,
            qp_y_prev: qp_prev_init,
            cu_qp_delta_val: 0,
            is_cu_qp_delta_coded: false,
            qg: (0, 0),
            qg_qp_prev: qp_prev_init,
            first_qg,
            last_pu_merged: false,
            ctb_addr_rs,
            ctb_addr_ts,
            coeffs: vec![0; 1024],
            warnings: 0,
        };

        loop {
            let rx = ctb_addr_rs % wc;
            dec.decode_ctu(ctb_addr_rs, ctb_addr_ts)?;
            let end_of_slice_segment = dec.cabac.terminate() != 0;
            // WPP storage after the second CTB of a (tile) row, keyed by the
            // CTB address so the row below finds it as its above-right.
            if pps.entropy_coding_sync && rx == tile_col_start(ctb_addr_rs) + 1 {
                saved_wpp.insert(ctb_addr_rs, dec.cx.clone());
            }
            if end_of_slice_segment {
                break;
            }
            ctb_addr_ts += 1;
            if ctb_addr_ts >= n_ctbs {
                return Err(Error::bitstream("slice segment runs past the picture"));
            }
            ctb_addr_rs = dec.info.ctb_ts_to_rs[ctb_addr_ts] as usize;
            let new_tile = dec.info.tile_id_ts[ctb_addr_ts] != dec.info.tile_id_ts[ctb_addr_ts - 1];
            let new_row = pps.entropy_coding_sync && ctb_addr_rs % wc == tile_col_start(ctb_addr_rs);
            if new_tile || new_row {
                // end_of_subset_one_bit + byte_alignment: the next substream
                // starts at the next entry point.
                sub += 1;
                let Some(&start) = substreams.get(sub) else {
                    return Err(Error::bitstream("missing entry point for a new substream"));
                };
                dec.cabac = Cabac::new(&rbsp[start..]);
                dec.cx = if new_tile {
                    Contexts::new(init_type, ind.slice_qp)
                } else {
                    match wpp_sync_source(dec.info, saved_wpp, ctb_addr_rs, slice_addr) {
                        Some(c) => c.clone(),
                        None => Contexts::new(init_type, ind.slice_qp),
                    }
                };
                dec.first_qg = true;
                dec.qp_y_prev = ind.slice_qp;
            }
            if dec.cabac.overrun() {
                return Err(Error::bitstream("slice data exhausted"));
            }
        }
        *last_qp_y = dec.qp_y;
        let cx_end = dec.cx.clone();
        let warn = dec.warnings;
        drop(dec);
        self.warnings += warn;
        if pps.dependent_slice_segments_enabled {
            *saved_ds = Some(cx_end);
        }
        Ok(())
    }

    fn finish_picture(&mut self) -> Result<()> {
        let Some(mut cur) = self.cur.take() else { return Ok(()) };
        // Loop filters.
        if std::env::var_os("H26X_NO_DEBLOCK").is_none() {
            deblock_picture(&mut cur.frame, &cur.info, &cur.pps, cur.sps.bit_depth_luma, cur.sps.bit_depth_chroma);
        }
        if std::env::var_os("H26X_NO_SAO").is_none() {
            sao_picture(&mut cur.frame, &cur.info, &cur.sps, &cur.pps);
        }
        cur.frame.extend_edges();
        cur.frame.poc = cur.poc;
        cur.frame.long_term = false;
        let pic = DpbPic {
            frame: cur.frame,
            poc: cur.poc,
            is_ref: true,
            long_term: false,
            needed_for_output: false,
            latency: 0,
            decode_index: cur.decode_index,
            crop: cur.sps.conf_win,
            generated: false,
            id: self.dpb.alloc_id(),
        };
        let _ = cur.nal_type;
        self.dpb.store(pic, cur.pic_output);
        Ok(())
    }
}

/// The WPP synchronisation source for the first CTB of a row at
/// `ctb_addr_rs`: the contexts saved after the second CTB of the row above
/// (its above-right neighbour), if that CTB is available (same slice and
/// tile, decoded).
fn wpp_sync_source<'a>(info: &PicInfo, saved: &'a HashMap<usize, Contexts>, ctb_addr_rs: usize, slice_addr: u32) -> Option<&'a Contexts> {
    let wc = info.wc;
    let row = ctb_addr_rs / wc;
    if row == 0 {
        return None;
    }
    let above_right = ctb_addr_rs + 1 - wc;
    if above_right / wc != row - 1 {
        return None;
    }
    if info.ctb_slice_addr[above_right] != slice_addr || info.ctb_tile[above_right] != info.ctb_tile[ctb_addr_rs] {
        return None;
    }
    saved.get(&above_right)
}
