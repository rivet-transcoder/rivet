//! Slice header (H.264 clause 7.3.3 / 7.4.3).

use crate::bitreader::BitReader;
use crate::nal::H264NalHeader;
use crate::{Error, Result};

use super::pps::Pps;
use super::sps::Sps;

/// `slice_type % 5`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SliceType {
    /// P slice.
    P,
    /// B slice.
    B,
    /// I slice.
    I,
    /// SP slice.
    Sp,
    /// SI slice.
    Si,
}

impl SliceType {
    fn from_raw(v: u32) -> Option<Self> {
        Some(match v % 5 {
            0 => SliceType::P,
            1 => SliceType::B,
            2 => SliceType::I,
            3 => SliceType::Sp,
            4 => SliceType::Si,
            _ => return None,
        })
    }
    /// Intra-only slice types.
    pub fn is_intra(self) -> bool {
        matches!(self, SliceType::I | SliceType::Si)
    }
    /// P or SP.
    pub fn is_p(self) -> bool {
        matches!(self, SliceType::P | SliceType::Sp)
    }
    /// B.
    pub fn is_b(self) -> bool {
        matches!(self, SliceType::B)
    }
}

/// One `ref_pic_list_modification` operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefListMod {
    /// `modification_of_pic_nums_idc` 0: subtract `abs_diff_pic_num_minus1 + 1`.
    SubtractPicNum(u32),
    /// idc 1: add `abs_diff_pic_num_minus1 + 1`.
    AddPicNum(u32),
    /// idc 2: `long_term_pic_num`.
    LongTerm(u32),
}

/// A weighted-prediction entry for one reference index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WeightEntry {
    /// Luma `(weight, offset)`; the default when `luma_weight_flag` is 0.
    pub luma: (i32, i32),
    /// Chroma Cb and Cr `(weight, offset)`.
    pub chroma: [(i32, i32); 2],
    /// Whether an explicit luma weight was sent (matters for the
    /// `implicit`/`explicit` distinction only through the values).
    pub luma_flag: bool,
    /// Whether explicit chroma weights were sent.
    pub chroma_flag: bool,
}

/// `pred_weight_table()`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PredWeightTable {
    /// `luma_log2_weight_denom`.
    pub luma_log2_denom: u32,
    /// `chroma_log2_weight_denom`.
    pub chroma_log2_denom: u32,
    /// Per list, per reference index.
    pub lists: [Vec<WeightEntry>; 2],
}

/// One memory-management control operation (`dec_ref_pic_marking`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mmco {
    /// 1: mark a short-term picture (by `difference_of_pic_nums_minus1`) unused.
    UnmarkShortTerm(u32),
    /// 2: mark a long-term picture (by `long_term_pic_num`) unused.
    UnmarkLongTerm(u32),
    /// 3: convert a short-term picture (`difference_of_pic_nums_minus1`) to
    /// long-term with `long_term_frame_idx`.
    ShortToLong(u32, u32),
    /// 4: `max_long_term_frame_idx_plus1`.
    MaxLongTermIdx(u32),
    /// 5: mark everything unused, reset frame_num / POC.
    UnmarkAll,
    /// 6: mark the current picture long-term with `long_term_frame_idx`.
    CurrentToLong(u32),
}

/// `dec_ref_pic_marking()`.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RefPicMarking {
    /// IDR: `no_output_of_prior_pics_flag`.
    pub no_output_of_prior_pics: bool,
    /// IDR: `long_term_reference_flag`.
    pub long_term_reference: bool,
    /// Non-IDR: `adaptive_ref_pic_marking_mode_flag`.
    pub adaptive: bool,
    /// The MMCO operations, in order.
    pub ops: Vec<Mmco>,
}

/// A parsed slice header plus the NAL-level facts that go with it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SliceHeader {
    /// `nal_ref_idc`.
    pub nal_ref_idc: u8,
    /// `nal_unit_type` (5 for IDR).
    pub nal_unit_type: u8,
    /// `first_mb_in_slice`.
    pub first_mb_in_slice: u32,
    /// Slice type (mod 5).
    pub slice_type: SliceType,
    /// The raw `slice_type` (5..=9 means every slice of the picture has this type).
    pub slice_type_raw: u32,
    /// `pic_parameter_set_id`.
    pub pps_id: u32,
    /// `colour_plane_id` (separate colour planes only).
    pub colour_plane_id: u32,
    /// `frame_num`.
    pub frame_num: u32,
    /// `field_pic_flag`.
    pub field_pic: bool,
    /// `bottom_field_flag`.
    pub bottom_field: bool,
    /// `idr_pic_id` (IDR only).
    pub idr_pic_id: u32,
    /// `pic_order_cnt_lsb` (POC type 0).
    pub poc_lsb: u32,
    /// `delta_pic_order_cnt_bottom` (POC type 0).
    pub delta_poc_bottom: i32,
    /// `delta_pic_order_cnt[0..2]` (POC type 1).
    pub delta_poc: [i32; 2],
    /// `redundant_pic_cnt`.
    pub redundant_pic_cnt: u32,
    /// `direct_spatial_mv_pred_flag` (B).
    pub direct_spatial_mv_pred: bool,
    /// Active reference count for list 0 / list 1 (after the override).
    pub num_ref_idx_active: [u32; 2],
    /// `ref_pic_list_modification` for list 0 and list 1.
    pub ref_list_mods: [Vec<RefListMod>; 2],
    /// `pred_weight_table`, when the PPS calls for one.
    pub pred_weights: Option<PredWeightTable>,
    /// `dec_ref_pic_marking` (reference pictures only).
    pub marking: RefPicMarking,
    /// `cabac_init_idc`.
    pub cabac_init_idc: u32,
    /// `SliceQPY = 26 + pic_init_qp_minus26 + slice_qp_delta`.
    pub slice_qp: i32,
    /// `sp_for_switch_flag`.
    pub sp_for_switch: bool,
    /// `QSY`.
    pub slice_qs: i32,
    /// `disable_deblocking_filter_idc` (0: on, 1: off, 2: on except slice edges).
    pub disable_deblocking_filter_idc: u32,
    /// `FilterOffsetA = slice_alpha_c0_offset_div2 << 1`.
    pub filter_offset_a: i32,
    /// `FilterOffsetB = slice_beta_offset_div2 << 1`.
    pub filter_offset_b: i32,
    /// Bit position in the RBSP where `slice_data()` starts (for CABAC, after
    /// `cabac_alignment_one_bit`s; for CAVLC, right after the header).
    pub data_bit_offset: u64,
}

impl SliceHeader {
    /// Whether this is an IDR picture's slice.
    pub fn is_idr(&self) -> bool {
        self.nal_unit_type == 5
    }
    /// `MbaffFrameFlag`.
    pub fn mbaff(&self, sps: &Sps) -> bool {
        sps.mb_adaptive_frame_field && !self.field_pic
    }
    /// Whether the picture is a reference picture.
    pub fn is_reference(&self) -> bool {
        self.nal_ref_idc != 0
    }

    /// Parse a slice header. `rbsp` is the whole NAL RBSP with the one-byte
    /// NAL header still in front (so bit offsets index the same buffer the
    /// slice data will be read from).
    pub fn parse(
        rbsp: &[u8],
        nal: H264NalHeader,
        pps_lookup: &dyn Fn(u32) -> Option<Pps>,
        sps_lookup: &dyn Fn(u32) -> Option<Sps>,
    ) -> Result<(SliceHeader, Pps, Sps)> {
        let mut r = BitReader::new(rbsp);
        r.bits(8); // the NAL header
        let first_mb_in_slice = r.ue();
        let slice_type_raw = r.ue();
        if slice_type_raw > 9 {
            return Err(Error::bitstream("slice: slice_type out of range"));
        }
        let slice_type = SliceType::from_raw(slice_type_raw).unwrap();
        let pps_id = r.ue();
        let pps = pps_lookup(pps_id)
            .ok_or_else(|| Error::bitstream(format!("slice references unknown PPS {pps_id}")))?;
        let sps = sps_lookup(pps.sps_id)
            .ok_or_else(|| Error::bitstream(format!("PPS {pps_id} references unknown SPS {}", pps.sps_id)))?;

        let mut colour_plane_id = 0;
        if sps.separate_colour_plane {
            colour_plane_id = r.bits(2);
        }
        let frame_num = r.bits(sps.log2_max_frame_num);
        let mut field_pic = false;
        let mut bottom_field = false;
        if !sps.frame_mbs_only {
            field_pic = r.flag();
            if field_pic {
                bottom_field = r.flag();
            }
        }
        let is_idr = nal.unit_type == 5;
        let mut idr_pic_id = 0;
        if is_idr {
            idr_pic_id = r.ue();
        }
        let mut poc_lsb = 0;
        let mut delta_poc_bottom = 0;
        let mut delta_poc = [0i32; 2];
        if sps.poc_type == 0 {
            poc_lsb = r.bits(sps.log2_max_poc_lsb);
            if pps.bottom_field_pic_order_in_frame_present && !field_pic {
                delta_poc_bottom = r.se();
            }
        }
        if sps.poc_type == 1 && !sps.delta_pic_order_always_zero {
            delta_poc[0] = r.se();
            if pps.bottom_field_pic_order_in_frame_present && !field_pic {
                delta_poc[1] = r.se();
            }
        }
        let mut redundant_pic_cnt = 0;
        if pps.redundant_pic_cnt_present {
            redundant_pic_cnt = r.ue();
        }
        let mut direct_spatial_mv_pred = false;
        if slice_type.is_b() {
            direct_spatial_mv_pred = r.flag();
        }
        let mut num_ref_idx_active = [pps.num_ref_idx_l0_default, pps.num_ref_idx_l1_default];
        if slice_type.is_p() || slice_type.is_b() {
            if r.flag() {
                // num_ref_idx_active_override_flag
                num_ref_idx_active[0] = r.ue() + 1;
                if slice_type.is_b() {
                    num_ref_idx_active[1] = r.ue() + 1;
                }
            }
            let max = if field_pic { 32 } else { 16 };
            if num_ref_idx_active[0] > max || (slice_type.is_b() && num_ref_idx_active[1] > max) {
                return Err(Error::bitstream("slice: num_ref_idx_active out of range"));
            }
        }
        if !slice_type.is_b() {
            num_ref_idx_active[1] = 0;
        }
        if slice_type.is_intra() {
            num_ref_idx_active = [0, 0];
        }

        // ref_pic_list_modification()
        let mut ref_list_mods: [Vec<RefListMod>; 2] = [Vec::new(), Vec::new()];
        if !slice_type.is_intra() {
            let lists = if slice_type.is_b() { 2 } else { 1 };
            for (l, mods) in ref_list_mods.iter_mut().enumerate().take(lists) {
                if r.flag() {
                    loop {
                        let idc = r.ue();
                        match idc {
                            0 => mods.push(RefListMod::SubtractPicNum(r.ue() + 1)),
                            1 => mods.push(RefListMod::AddPicNum(r.ue() + 1)),
                            2 => mods.push(RefListMod::LongTerm(r.ue())),
                            3 => break,
                            _ => {
                                return Err(Error::bitstream(format!(
                                    "slice: modification_of_pic_nums_idc {idc} out of range (list {l})"
                                )));
                            }
                        }
                        if mods.len() > 64 || r.overrun() {
                            return Err(Error::bitstream("slice: runaway ref_pic_list_modification"));
                        }
                    }
                }
            }
        }

        // pred_weight_table()
        let mut pred_weights = None;
        if (pps.weighted_pred && slice_type.is_p()) || (pps.weighted_bipred_idc == 1 && slice_type.is_b()) {
            let luma_log2_denom = r.ue();
            let mut chroma_log2_denom = 0;
            if sps.chroma_format_idc != 0 {
                chroma_log2_denom = r.ue();
            }
            if luma_log2_denom > 7 || chroma_log2_denom > 7 {
                return Err(Error::bitstream("slice: log2_weight_denom out of range"));
            }
            let mut lists: [Vec<WeightEntry>; 2] = [Vec::new(), Vec::new()];
            let nlists = if slice_type.is_b() { 2 } else { 1 };
            for (l, list) in lists.iter_mut().enumerate().take(nlists) {
                for _ in 0..num_ref_idx_active[l] {
                    let mut e = WeightEntry {
                        luma: (1 << luma_log2_denom, 0),
                        chroma: [(1 << chroma_log2_denom, 0); 2],
                        luma_flag: false,
                        chroma_flag: false,
                    };
                    e.luma_flag = r.flag();
                    if e.luma_flag {
                        let w = r.se();
                        let o = r.se();
                        if !(-128..=127).contains(&w) || !(-128..=127).contains(&o) {
                            return Err(Error::bitstream("slice: luma weight/offset out of range"));
                        }
                        e.luma = (w, o);
                    }
                    if sps.chroma_format_idc != 0 {
                        e.chroma_flag = r.flag();
                        if e.chroma_flag {
                            for c in 0..2 {
                                let w = r.se();
                                let o = r.se();
                                if !(-128..=127).contains(&w) || !(-128..=127).contains(&o) {
                                    return Err(Error::bitstream("slice: chroma weight/offset out of range"));
                                }
                                e.chroma[c] = (w, o);
                            }
                        }
                    }
                    list.push(e);
                }
            }
            pred_weights = Some(PredWeightTable { luma_log2_denom, chroma_log2_denom, lists });
        }

        // dec_ref_pic_marking()
        let mut marking = RefPicMarking::default();
        if nal.ref_idc != 0 {
            if is_idr {
                marking.no_output_of_prior_pics = r.flag();
                marking.long_term_reference = r.flag();
            } else {
                marking.adaptive = r.flag();
                if marking.adaptive {
                    loop {
                        let op = r.ue();
                        let mmco = match op {
                            0 => break,
                            1 => Mmco::UnmarkShortTerm(r.ue()),
                            2 => Mmco::UnmarkLongTerm(r.ue()),
                            3 => {
                                let d = r.ue();
                                Mmco::ShortToLong(d, r.ue())
                            }
                            4 => Mmco::MaxLongTermIdx(r.ue()),
                            5 => Mmco::UnmarkAll,
                            6 => Mmco::CurrentToLong(r.ue()),
                            _ => {
                                return Err(Error::bitstream(format!(
                                    "slice: memory_management_control_operation {op} out of range"
                                )));
                            }
                        };
                        marking.ops.push(mmco);
                        if marking.ops.len() > 66 || r.overrun() {
                            return Err(Error::bitstream("slice: runaway dec_ref_pic_marking"));
                        }
                    }
                }
            }
        }

        let mut cabac_init_idc = 0;
        if pps.cabac && !slice_type.is_intra() {
            cabac_init_idc = r.ue();
            if cabac_init_idc > 2 {
                return Err(Error::bitstream("slice: cabac_init_idc out of range"));
            }
        }
        let slice_qp_delta = r.se();
        let slice_qp = pps.pic_init_qp + slice_qp_delta;
        if !(0..=51).contains(&slice_qp) && sps.bit_depth_luma == 8 {
            return Err(Error::bitstream(format!("slice: SliceQPY {slice_qp} out of range")));
        }
        let mut sp_for_switch = false;
        let mut slice_qs = 0;
        if matches!(slice_type, SliceType::Sp | SliceType::Si) {
            if slice_type == SliceType::Sp {
                sp_for_switch = r.flag();
            }
            slice_qs = pps.pic_init_qs + r.se();
        }
        let mut disable_deblocking_filter_idc = 0;
        let mut filter_offset_a = 0;
        let mut filter_offset_b = 0;
        if pps.deblocking_filter_control_present {
            disable_deblocking_filter_idc = r.ue();
            if disable_deblocking_filter_idc > 2 {
                return Err(Error::bitstream("slice: disable_deblocking_filter_idc out of range"));
            }
            if disable_deblocking_filter_idc != 1 {
                let a = r.se();
                let b = r.se();
                if !(-6..=6).contains(&a) || !(-6..=6).contains(&b) {
                    return Err(Error::bitstream("slice: deblocking offsets out of range"));
                }
                filter_offset_a = a << 1;
                filter_offset_b = b << 1;
            }
        }
        // (slice_group_change_cycle would go here; slice groups are refused
        // at the PPS.)
        r.finish("slice header")?;

        let mut data_bit_offset = r.position();
        if pps.cabac {
            // cabac_alignment_one_bit until byte aligned.
            let rem = data_bit_offset % 8;
            if rem != 0 {
                data_bit_offset += 8 - rem;
            }
        }

        Ok((
            SliceHeader {
                nal_ref_idc: nal.ref_idc,
                nal_unit_type: nal.unit_type,
                first_mb_in_slice,
                slice_type,
                slice_type_raw,
                pps_id,
                colour_plane_id,
                frame_num,
                field_pic,
                bottom_field,
                idr_pic_id,
                poc_lsb,
                delta_poc_bottom,
                delta_poc,
                redundant_pic_cnt,
                direct_spatial_mv_pred,
                num_ref_idx_active,
                ref_list_mods,
                pred_weights,
                marking,
                cabac_init_idc,
                slice_qp,
                sp_for_switch,
                slice_qs,
                disable_deblocking_filter_idc,
                filter_offset_a,
                filter_offset_b,
                data_bit_offset,
            },
            pps,
            sps,
        ))
    }
}
