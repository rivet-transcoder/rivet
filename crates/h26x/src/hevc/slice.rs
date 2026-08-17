//! Slice segment header (H.265 clause 7.3.6).

use crate::bitreader::BitReader;
use crate::nal::HevcNalHeader;
use crate::{Error, Result};

use super::pps::Pps;
use super::sps::{Sps, StRps, parse_st_rps};

/// NAL unit types the decoder distinguishes.
pub mod nal_type {
    /// TRAIL_N.
    pub const TRAIL_N: u8 = 0;
    /// TRAIL_R.
    pub const TRAIL_R: u8 = 1;
    /// RADL_N.
    pub const RADL_N: u8 = 6;
    /// RADL_R.
    pub const RADL_R: u8 = 7;
    /// RASL_N.
    pub const RASL_N: u8 = 8;
    /// RASL_R.
    pub const RASL_R: u8 = 9;
    /// BLA_W_LP.
    pub const BLA_W_LP: u8 = 16;
    /// BLA_W_RADL.
    pub const BLA_W_RADL: u8 = 17;
    /// BLA_N_LP.
    pub const BLA_N_LP: u8 = 18;
    /// IDR_W_RADL.
    pub const IDR_W_RADL: u8 = 19;
    /// IDR_N_LP.
    pub const IDR_N_LP: u8 = 20;
    /// CRA_NUT.
    pub const CRA: u8 = 21;
    /// VPS_NUT.
    pub const VPS: u8 = 32;
    /// SPS_NUT.
    pub const SPS: u8 = 33;
    /// PPS_NUT.
    pub const PPS: u8 = 34;
    /// AUD_NUT.
    pub const AUD: u8 = 35;
    /// EOS_NUT.
    pub const EOS: u8 = 36;
    /// EOB_NUT.
    pub const EOB: u8 = 37;
    /// FD_NUT.
    pub const FD: u8 = 38;
    /// PREFIX_SEI_NUT.
    pub const SEI_PREFIX: u8 = 39;
    /// SUFFIX_SEI_NUT.
    pub const SEI_SUFFIX: u8 = 40;

    /// IRAP (16..=23).
    pub fn is_irap(t: u8) -> bool {
        (16..=23).contains(&t)
    }
    /// IDR.
    pub fn is_idr(t: u8) -> bool {
        t == IDR_W_RADL || t == IDR_N_LP
    }
    /// BLA.
    pub fn is_bla(t: u8) -> bool {
        (BLA_W_LP..=BLA_N_LP).contains(&t)
    }
    /// RASL.
    pub fn is_rasl(t: u8) -> bool {
        t == RASL_N || t == RASL_R
    }
    /// RADL.
    pub fn is_radl(t: u8) -> bool {
        t == RADL_N || t == RADL_R
    }
    /// A sub-layer non-reference picture (even types below 16, except reserved).
    pub fn is_sub_layer_non_ref(t: u8) -> bool {
        t < 16 && t % 2 == 0
    }
    /// A VCL NAL unit (a slice segment) this decoder handles.
    pub fn is_slice(t: u8) -> bool {
        t <= 9 || (16..=21).contains(&t)
    }
}

/// `slice_type`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SliceType {
    /// B.
    B,
    /// P.
    P,
    /// I.
    I,
}

impl SliceType {
    /// Intra?
    pub fn is_intra(self) -> bool {
        self == SliceType::I
    }
    /// B?
    pub fn is_b(self) -> bool {
        self == SliceType::B
    }
}

/// A weighted-prediction entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WeightEntry {
    /// Luma `(weight, offset)`.
    pub luma: (i32, i32),
    /// Chroma Cb / Cr `(weight, offset)`.
    pub chroma: [(i32, i32); 2],
}

/// `pred_weight_table()`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PredWeightTable {
    /// `luma_log2_weight_denom`.
    pub luma_log2_denom: u32,
    /// `ChromaLog2WeightDenom`.
    pub chroma_log2_denom: u32,
    /// Per list, per reference index.
    pub lists: [Vec<WeightEntry>; 2],
}

/// The long-term reference picture entries of a slice: `(poc, msb_present,
/// used_by_curr_pic)` — `poc` is the full POC when `msb_present`, else the
/// LSB value (matched on LSBs).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LtEntry {
    /// `PocLsbLt` or the full POC.
    pub poc: i32,
    /// `delta_poc_msb_present_flag`.
    pub msb_present: bool,
    /// `UsedByCurrPicLt`.
    pub used: bool,
}

/// A parsed slice segment header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SliceHeader {
    /// NAL unit type.
    pub nal_type: u8,
    /// `nuh_temporal_id_plus1 - 1`.
    pub temporal_id: u8,
    /// `first_slice_segment_in_pic_flag`.
    pub first_slice_segment_in_pic: bool,
    /// `no_output_of_prior_pics_flag` (IRAP).
    pub no_output_of_prior_pics: bool,
    /// `slice_pic_parameter_set_id`.
    pub pps_id: u32,
    /// `dependent_slice_segment_flag`.
    pub dependent: bool,
    /// `slice_segment_address` (in CTBs, raster scan).
    pub segment_address: u32,
    /// `slice_type`.
    pub slice_type: SliceType,
    /// `pic_output_flag`.
    pub pic_output: bool,
    /// `colour_plane_id`.
    pub colour_plane_id: u32,
    /// `slice_pic_order_cnt_lsb`.
    pub poc_lsb: u32,
    /// The short-term RPS in effect (from the SPS or coded here).
    pub st_rps: StRps,
    /// Number of bits the coded st_ref_pic_set took (for the entry-point-less
    /// path nothing needs it; kept for completeness).
    pub st_rps_bits: u32,
    /// Long-term entries.
    pub lt: Vec<LtEntry>,
    /// `slice_temporal_mvp_enabled_flag`.
    pub temporal_mvp_enabled: bool,
    /// `slice_sao_luma_flag`.
    pub sao_luma: bool,
    /// `slice_sao_chroma_flag`.
    pub sao_chroma: bool,
    /// Active reference counts per list.
    pub num_ref_idx: [u32; 2],
    /// `ref_pic_lists_modification`: `list_entry_lX` per list when present.
    pub list_entry: [Option<Vec<u32>>; 2],
    /// `mvd_l1_zero_flag`.
    pub mvd_l1_zero: bool,
    /// `cabac_init_flag`.
    pub cabac_init: bool,
    /// `collocated_from_l0_flag`.
    pub collocated_from_l0: bool,
    /// `collocated_ref_idx`.
    pub collocated_ref_idx: u32,
    /// Weighted prediction table.
    pub pred_weights: Option<PredWeightTable>,
    /// `MaxNumMergeCand`.
    pub max_num_merge_cand: u32,
    /// `SliceQpY`.
    pub slice_qp: i32,
    /// `slice_cb_qp_offset`.
    pub cb_qp_offset: i32,
    /// `slice_cr_qp_offset`.
    pub cr_qp_offset: i32,
    /// `slice_deblocking_filter_disabled_flag`.
    pub deblocking_disabled: bool,
    /// `slice_beta_offset_div2 * 2`.
    pub beta_offset: i32,
    /// `slice_tc_offset_div2 * 2`.
    pub tc_offset: i32,
    /// `slice_loop_filter_across_slices_enabled_flag`.
    pub loop_filter_across_slices: bool,
    /// Entry point offsets (byte offsets of substream starts relative to the
    /// first byte of slice segment data): cumulative, `entry_point_offset_minus1 + 1`.
    pub entry_points: Vec<u32>,
    /// Bit position where `slice_segment_data()` starts (after `byte_alignment()`).
    pub data_bit_offset: u64,
}

impl SliceHeader {
    /// Whether this is an IRAP picture.
    pub fn is_irap(&self) -> bool {
        nal_type::is_irap(self.nal_type)
    }
    /// IDR?
    pub fn is_idr(&self) -> bool {
        nal_type::is_idr(self.nal_type)
    }

    /// Parse a slice segment header. `rbsp` is the whole NAL RBSP with its
    /// two header bytes. For a dependent slice segment, `independent` supplies
    /// the fields copied from the preceding independent segment.
    pub fn parse(
        rbsp: &[u8],
        nal: HevcNalHeader,
        pps_lookup: &dyn Fn(u32) -> Option<Pps>,
        sps_lookup: &dyn Fn(u32) -> Option<Sps>,
        independent: Option<&SliceHeader>,
    ) -> Result<(SliceHeader, Pps, Sps)> {
        let mut r = BitReader::new(rbsp);
        r.bits(16); // NAL header
        let first_slice_segment_in_pic = r.flag();
        let mut no_output_of_prior_pics = false;
        if nal_type::is_irap(nal.unit_type) {
            no_output_of_prior_pics = r.flag();
        }
        let pps_id = r.ue();
        let pps = pps_lookup(pps_id).ok_or_else(|| Error::bitstream(format!("slice references unknown PPS {pps_id}")))?;
        let sps = sps_lookup(pps.sps_id)
            .ok_or_else(|| Error::bitstream(format!("PPS {pps_id} references unknown SPS {}", pps.sps_id)))?;
        let mut dependent = false;
        let mut segment_address = 0;
        if !first_slice_segment_in_pic {
            if pps.dependent_slice_segments_enabled {
                dependent = r.flag();
            }
            let pic_size_in_ctbs = sps.pic_width_in_ctbs() * sps.pic_height_in_ctbs();
            // Ceil(Log2(PicSizeInCtbsY)) bits (zero of them for a 1-CTB picture).
            let bits = 32 - (pic_size_in_ctbs - 1).leading_zeros();
            if bits > 0 {
                segment_address = r.bits(bits);
            }
            if segment_address >= pic_size_in_ctbs {
                return Err(Error::bitstream("slice_segment_address out of range"));
            }
        }
        if pic_size_is_one(&sps) && !first_slice_segment_in_pic {
            segment_address = 0;
        }

        // A dependent slice segment copies everything else from the
        // independent one it follows.
        if dependent {
            let Some(ind) = independent else {
                return Err(Error::bitstream("dependent slice segment without a preceding independent one"));
            };
            let mut hdr = ind.clone();
            hdr.first_slice_segment_in_pic = false;
            hdr.dependent = true;
            hdr.segment_address = segment_address;
            hdr.nal_type = nal.unit_type;
            hdr.temporal_id = nal.temporal_id;
            hdr.entry_points = parse_entry_points(&mut r, &pps)?;
            if pps.slice_header_extension_present {
                let n = r.ue();
                for _ in 0..n {
                    r.bits(8);
                }
            }
            byte_alignment(&mut r)?;
            hdr.data_bit_offset = r.position();
            r.finish("dependent slice segment header")?;
            return Ok((hdr, pps, sps));
        }

        for _ in 0..pps.num_extra_slice_header_bits {
            r.bit();
        }
        let slice_type = match r.ue() {
            0 => SliceType::B,
            1 => SliceType::P,
            2 => SliceType::I,
            _ => return Err(Error::bitstream("slice_type out of range")),
        };
        let mut pic_output = true;
        if pps.output_flag_present {
            pic_output = r.flag();
        }
        let mut colour_plane_id = 0;
        if sps.separate_colour_plane {
            colour_plane_id = r.bits(2);
        }
        let mut poc_lsb = 0;
        let mut st_rps = StRps::default();
        let mut st_rps_bits = 0;
        let mut lt = Vec::new();
        let mut temporal_mvp_enabled = false;
        if !nal_type::is_idr(nal.unit_type) {
            poc_lsb = r.bits(sps.log2_max_poc_lsb);
            let short_term_ref_pic_set_sps_flag = r.flag();
            let num_sets = sps.st_rps.len();
            if !short_term_ref_pic_set_sps_flag {
                let before = r.position();
                st_rps = parse_st_rps(&mut r, num_sets, num_sets, &sps.st_rps)?;
                st_rps_bits = (r.position() - before) as u32;
            } else {
                if num_sets == 0 {
                    return Err(Error::bitstream("short_term_ref_pic_set_sps_flag with no SPS sets"));
                }
                let bits = 32 - ((num_sets as u32).saturating_sub(1)).leading_zeros();
                let idx = if num_sets > 1 { r.bits(bits) as usize } else { 0 };
                if idx >= num_sets {
                    return Err(Error::bitstream("short_term_ref_pic_set_idx out of range"));
                }
                st_rps = sps.st_rps[idx].clone();
            }
            if sps.long_term_ref_pics_present {
                let mut num_lt_sps = 0;
                if !sps.lt_ref_pics.is_empty() {
                    num_lt_sps = r.ue();
                    if num_lt_sps as usize > sps.lt_ref_pics.len() {
                        return Err(Error::bitstream("num_long_term_sps out of range"));
                    }
                }
                let num_lt_pics = r.ue();
                if num_lt_sps + num_lt_pics > 32 {
                    return Err(Error::bitstream("too many long-term pictures"));
                }
                let mut prev_msb_cycle: i32 = 0;
                for i in 0..(num_lt_sps + num_lt_pics) {
                    let (lsb, used) = if i < num_lt_sps {
                        let bits = 32 - ((sps.lt_ref_pics.len() as u32).saturating_sub(1)).leading_zeros();
                        let idx = if sps.lt_ref_pics.len() > 1 { r.bits(bits) as usize } else { 0 };
                        if idx >= sps.lt_ref_pics.len() {
                            return Err(Error::bitstream("lt_idx_sps out of range"));
                        }
                        sps.lt_ref_pics[idx]
                    } else {
                        let lsb = r.bits(sps.log2_max_poc_lsb);
                        let used = r.flag();
                        (lsb, used)
                    };
                    let msb_present = r.flag();
                    // 7-52: DeltaPocMsbCycleLt accumulates, restarting at
                    // i == 0 and i == num_long_term_sps; an entry without the
                    // flag contributes 0.
                    let cycle = if msb_present { r.ue() as i32 } else { 0 };
                    let delta_msb_cycle = if i == 0 || i == num_lt_sps { cycle } else { cycle + prev_msb_cycle };
                    prev_msb_cycle = delta_msb_cycle;
                    // With the flag, the full POC is
                    //   PicOrderCntVal - DeltaPocMsbCycleLt * MaxPocLsb - slice_pic_order_cnt_lsb + PocLsbLt;
                    // stored here as (PocLsbLt - cycle * MaxPocLsb) and completed
                    // by the DPB, which knows PicOrderCntVal.
                    let poc = if msb_present { lsb as i32 - delta_msb_cycle * sps.max_poc_lsb() } else { lsb as i32 };
                    lt.push(LtEntry { poc, msb_present, used });
                }
            }
            if sps.temporal_mvp_enabled {
                temporal_mvp_enabled = r.flag();
            }
        }
        let mut sao_luma = false;
        let mut sao_chroma = false;
        if sps.sao_enabled {
            sao_luma = r.flag();
            if sps.chroma_format_idc != 0 {
                sao_chroma = r.flag();
            }
        }
        let mut num_ref_idx = [0u32; 2];
        let mut list_entry: [Option<Vec<u32>>; 2] = [None, None];
        let mut mvd_l1_zero = false;
        let mut cabac_init = false;
        let mut collocated_from_l0 = true;
        let mut collocated_ref_idx = 0;
        let mut pred_weights = None;
        let mut max_num_merge_cand = 5;
        if slice_type != SliceType::I {
            num_ref_idx = [pps.num_ref_idx_l0_default, if slice_type == SliceType::B { pps.num_ref_idx_l1_default } else { 0 }];
            if r.flag() {
                num_ref_idx[0] = r.ue() + 1;
                if slice_type == SliceType::B {
                    num_ref_idx[1] = r.ue() + 1;
                }
            }
            if num_ref_idx[0] > 15 || num_ref_idx[1] > 15 {
                return Err(Error::bitstream("num_ref_idx_active out of range"));
            }
            // NumPicTotalCurr for the list modification bit width.
            let mut num_pic_total_curr = 0u32;
            for &(_, used) in st_rps.neg.iter().chain(st_rps.pos.iter()) {
                if used {
                    num_pic_total_curr += 1;
                }
            }
            for e in &lt {
                if e.used {
                    num_pic_total_curr += 1;
                }
            }
            if pps.lists_modification_present && num_pic_total_curr > 1 {
                let bits = 32 - (num_pic_total_curr - 1).leading_zeros();
                if r.flag() {
                    let mut v = Vec::new();
                    for _ in 0..num_ref_idx[0] {
                        v.push(r.bits(bits));
                    }
                    list_entry[0] = Some(v);
                }
                if slice_type == SliceType::B && r.flag() {
                    let mut v = Vec::new();
                    for _ in 0..num_ref_idx[1] {
                        v.push(r.bits(bits));
                    }
                    list_entry[1] = Some(v);
                }
            }
            if slice_type == SliceType::B {
                mvd_l1_zero = r.flag();
            }
            if pps.cabac_init_present {
                cabac_init = r.flag();
            }
            if temporal_mvp_enabled {
                if slice_type == SliceType::B {
                    collocated_from_l0 = r.flag();
                }
                if (collocated_from_l0 && num_ref_idx[0] > 1) || (!collocated_from_l0 && num_ref_idx[1] > 1) {
                    collocated_ref_idx = r.ue();
                    let n = if collocated_from_l0 { num_ref_idx[0] } else { num_ref_idx[1] };
                    if collocated_ref_idx >= n {
                        return Err(Error::bitstream("collocated_ref_idx out of range"));
                    }
                }
            }
            if (pps.weighted_pred && slice_type == SliceType::P) || (pps.weighted_bipred && slice_type == SliceType::B) {
                pred_weights = Some(parse_pred_weight_table(&mut r, &sps, slice_type, num_ref_idx)?);
            }
            let five_minus = r.ue();
            if five_minus > 4 {
                return Err(Error::bitstream("five_minus_max_num_merge_cand out of range"));
            }
            max_num_merge_cand = 5 - five_minus;
        }
        let slice_qp = pps.init_qp + r.se();
        let qp_bd_offset = 6 * (sps.bit_depth_luma as i32 - 8);
        if slice_qp < -qp_bd_offset || slice_qp > 51 {
            return Err(Error::bitstream("SliceQpY out of range"));
        }
        let mut cb_qp_offset = 0;
        let mut cr_qp_offset = 0;
        if pps.slice_chroma_qp_offsets_present {
            cb_qp_offset = r.se();
            cr_qp_offset = r.se();
        }
        // (cu_chroma_qp_offset_enabled_flag: range extension only.)
        let mut deblocking_disabled = pps.deblocking_disabled;
        let mut beta_offset = pps.beta_offset;
        let mut tc_offset = pps.tc_offset;
        let mut override_flag = false;
        if pps.deblocking_override_enabled {
            override_flag = r.flag();
        }
        if override_flag {
            deblocking_disabled = r.flag();
            if !deblocking_disabled {
                beta_offset = r.se() * 2;
                tc_offset = r.se() * 2;
                if !(-12..=12).contains(&beta_offset) || !(-12..=12).contains(&tc_offset) {
                    return Err(Error::bitstream("slice deblocking offsets out of range"));
                }
            }
        }
        let mut loop_filter_across_slices = pps.loop_filter_across_slices;
        if pps.loop_filter_across_slices && (sao_luma || sao_chroma || !deblocking_disabled) {
            loop_filter_across_slices = r.flag();
        }
        let entry_points = parse_entry_points(&mut r, &pps)?;
        if pps.slice_header_extension_present {
            let n = r.ue();
            if n > 256 {
                return Err(Error::bitstream("slice_segment_header_extension_length out of range"));
            }
            for _ in 0..n {
                r.bits(8);
            }
        }
        byte_alignment(&mut r)?;
        let data_bit_offset = r.position();
        r.finish("slice segment header")?;

        Ok((
            SliceHeader {
                nal_type: nal.unit_type,
                temporal_id: nal.temporal_id,
                first_slice_segment_in_pic,
                no_output_of_prior_pics,
                pps_id,
                dependent: false,
                segment_address,
                slice_type,
                pic_output,
                colour_plane_id,
                poc_lsb,
                st_rps,
                st_rps_bits,
                lt,
                temporal_mvp_enabled,
                sao_luma,
                sao_chroma,
                num_ref_idx,
                list_entry,
                mvd_l1_zero,
                cabac_init,
                collocated_from_l0,
                collocated_ref_idx,
                pred_weights,
                max_num_merge_cand,
                slice_qp,
                cb_qp_offset,
                cr_qp_offset,
                deblocking_disabled,
                beta_offset,
                tc_offset,
                loop_filter_across_slices,
                entry_points,
                data_bit_offset,
            },
            pps,
            sps,
        ))
    }
}

fn pic_size_is_one(sps: &Sps) -> bool {
    sps.pic_width_in_ctbs() * sps.pic_height_in_ctbs() == 1
}

fn parse_entry_points(r: &mut BitReader, pps: &Pps) -> Result<Vec<u32>> {
    let mut entry_points = Vec::new();
    if pps.tiles_enabled || pps.entropy_coding_sync {
        let n = r.ue();
        if n > 440 * 4 {
            return Err(Error::bitstream("num_entry_point_offsets out of range"));
        }
        if n > 0 {
            let len = r.ue() + 1;
            if len > 32 {
                return Err(Error::bitstream("offset_len_minus1 out of range"));
            }
            let mut acc = 0u32;
            for _ in 0..n {
                let v = r.bits(len) + 1;
                acc = acc.wrapping_add(v);
                entry_points.push(acc);
            }
        }
    }
    Ok(entry_points)
}

fn byte_alignment(r: &mut BitReader) -> Result<()> {
    if r.bit() != 1 {
        return Err(Error::bitstream("byte_alignment(): alignment_bit_equal_to_one is 0"));
    }
    while !r.byte_aligned() {
        if r.bit() != 0 {
            return Err(Error::bitstream("byte_alignment(): alignment_bit_equal_to_zero is 1"));
        }
    }
    Ok(())
}

fn parse_pred_weight_table(r: &mut BitReader, sps: &Sps, slice_type: SliceType, num_ref_idx: [u32; 2]) -> Result<PredWeightTable> {
    let luma_log2_denom = r.ue();
    if luma_log2_denom > 7 {
        return Err(Error::bitstream("luma_log2_weight_denom out of range"));
    }
    let mut chroma_log2_denom = luma_log2_denom;
    if sps.chroma_format_idc != 0 {
        let d = luma_log2_denom as i32 + r.se();
        if !(0..=7).contains(&d) {
            return Err(Error::bitstream("ChromaLog2WeightDenom out of range"));
        }
        chroma_log2_denom = d as u32;
    }
    let mut lists: [Vec<WeightEntry>; 2] = [Vec::new(), Vec::new()];
    let nlists = if slice_type == SliceType::B { 2 } else { 1 };
    // Non-high-precision offsets (the range extension is refused): the
    // stored offsets are already shifted to the sample bit depth
    // (WpOffsetBdShift), so the prediction process adds them directly.
    let shift_y = sps.bit_depth_luma as i32 - 8;
    let shift_c = sps.bit_depth_chroma as i32 - 8;
    let half_c: i32 = 128; // WpOffsetHalfRangeC without high precision
    for (l, list) in lists.iter_mut().enumerate().take(nlists) {
        let n = num_ref_idx[l] as usize;
        let mut luma_flags = vec![false; n];
        let mut chroma_flags = vec![false; n];
        for f in luma_flags.iter_mut() {
            *f = r.flag();
        }
        if sps.chroma_format_idc != 0 {
            for f in chroma_flags.iter_mut() {
                *f = r.flag();
            }
        }
        for i in 0..n {
            let mut e = WeightEntry { luma: (1 << luma_log2_denom, 0), chroma: [(1 << chroma_log2_denom, 0); 2] };
            if luma_flags[i] {
                let dw = r.se();
                let off = r.se();
                if !(-128..=127).contains(&dw) || !(-128..=127).contains(&off) {
                    return Err(Error::bitstream("luma weight/offset out of range"));
                }
                e.luma = ((1 << luma_log2_denom) + dw, off << shift_y);
            }
            if chroma_flags[i] {
                for c in 0..2 {
                    let dw = r.se();
                    let doff = r.se();
                    if !(-128..=127).contains(&dw) || !(-4 * half_c..=4 * half_c - 1).contains(&doff) {
                        return Err(Error::bitstream("chroma weight/offset out of range"));
                    }
                    let w = (1 << chroma_log2_denom) + dw;
                    let o = (half_c + doff - ((half_c * w) >> chroma_log2_denom)).clamp(-half_c, half_c - 1);
                    e.chroma[c] = (w, o << shift_c);
                }
            }
            list.push(e);
        }
    }
    Ok(PredWeightTable { luma_log2_denom, chroma_log2_denom, lists })
}
