//! Picture parameter set (H.264 clause 7.3.2.2 / 7.4.2.2).

use crate::bitreader::BitReader;
use crate::{Error, Result};

use super::sps::{ScalingLists, Sps, parse_scaling_matrix};

/// A parsed PPS.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pps {
    /// `pic_parameter_set_id` (0..=255).
    pub id: u32,
    /// `seq_parameter_set_id`.
    pub sps_id: u32,
    /// `entropy_coding_mode_flag`: CABAC (true) or CAVLC.
    pub cabac: bool,
    /// `bottom_field_pic_order_in_frame_present_flag`.
    pub bottom_field_pic_order_in_frame_present: bool,
    /// `num_slice_groups_minus1 + 1`.
    pub num_slice_groups: u32,
    /// `num_ref_idx_l0_default_active_minus1 + 1`.
    pub num_ref_idx_l0_default: u32,
    /// `num_ref_idx_l1_default_active_minus1 + 1`.
    pub num_ref_idx_l1_default: u32,
    /// `weighted_pred_flag`.
    pub weighted_pred: bool,
    /// `weighted_bipred_idc` (0: default, 1: explicit, 2: implicit).
    pub weighted_bipred_idc: u32,
    /// `pic_init_qp_minus26 + 26`.
    pub pic_init_qp: i32,
    /// `pic_init_qs_minus26 + 26`.
    pub pic_init_qs: i32,
    /// `chroma_qp_index_offset`.
    pub chroma_qp_index_offset: i32,
    /// `second_chroma_qp_index_offset` (equal to the first when absent).
    pub second_chroma_qp_index_offset: i32,
    /// `deblocking_filter_control_present_flag`.
    pub deblocking_filter_control_present: bool,
    /// `constrained_intra_pred_flag`.
    pub constrained_intra_pred: bool,
    /// `redundant_pic_cnt_present_flag`.
    pub redundant_pic_cnt_present: bool,
    /// `transform_8x8_mode_flag`.
    pub transform_8x8_mode: bool,
    /// The scaling lists the PPS sends, when `pic_scaling_matrix_present_flag`
    /// (already resolved against the SPS per Table 7-2).
    pub scaling_lists: Option<ScalingLists>,
}

impl Pps {
    /// Parse a PPS RBSP. Needs the SPS table because the extension part
    /// (scaling matrices) depends on the SPS's chroma format and lists.
    pub fn parse(rbsp: &[u8], sps_lookup: &dyn Fn(u32) -> Option<Sps>) -> Result<Pps> {
        let mut r = BitReader::new(rbsp);
        let id = r.ue();
        if id > 255 {
            return Err(Error::bitstream("PPS: pic_parameter_set_id out of range"));
        }
        let sps_id = r.ue();
        if sps_id > 31 {
            return Err(Error::bitstream("PPS: seq_parameter_set_id out of range"));
        }
        let cabac = r.flag();
        let bottom_field_pic_order_in_frame_present = r.flag();
        let num_slice_groups = r.ue() + 1;
        if num_slice_groups > 1 {
            // Flexible macroblock ordering (Baseline/Extended only). Parse
            // past it so the rest of the PPS is understood, then refuse the
            // stream: FMO/ASO decoding is not implemented.
            let map_type = r.ue();
            match map_type {
                0 => {
                    for _ in 0..num_slice_groups {
                        r.ue();
                    }
                }
                2 => {
                    for _ in 0..num_slice_groups - 1 {
                        r.ue();
                        r.ue();
                    }
                }
                3..=5 => {
                    r.flag();
                    r.ue();
                }
                6 => {
                    let n = r.ue() + 1;
                    let bits = 32 - (num_slice_groups - 1).leading_zeros();
                    for _ in 0..n {
                        r.bits(bits.max(1));
                    }
                }
                _ => {}
            }
            return Err(Error::unsupported(format!(
                "H.264 slice groups (FMO, num_slice_groups={num_slice_groups}) are not supported"
            )));
        }
        let num_ref_idx_l0_default = r.ue() + 1;
        let num_ref_idx_l1_default = r.ue() + 1;
        if num_ref_idx_l0_default > 32 || num_ref_idx_l1_default > 32 {
            return Err(Error::bitstream("PPS: num_ref_idx_default out of range"));
        }
        let weighted_pred = r.flag();
        let weighted_bipred_idc = r.bits(2);
        let pic_init_qp = r.se() + 26;
        let pic_init_qs = r.se() + 26;
        let chroma_qp_index_offset = r.se();
        if !(-12..=12).contains(&chroma_qp_index_offset) {
            return Err(Error::bitstream("PPS: chroma_qp_index_offset out of range"));
        }
        let deblocking_filter_control_present = r.flag();
        let constrained_intra_pred = r.flag();
        let redundant_pic_cnt_present = r.flag();
        let mut transform_8x8_mode = false;
        let mut scaling_lists = None;
        let mut second_chroma_qp_index_offset = chroma_qp_index_offset;
        if r.more_rbsp_data() {
            transform_8x8_mode = r.flag();
            if r.flag() {
                // pic_scaling_matrix_present_flag. Rule A falls back to the
                // SPS lists when the SPS sent any, else the defaults.
                let sps = sps_lookup(sps_id)
                    .ok_or_else(|| Error::bitstream(format!("PPS {id} references unknown SPS {sps_id}")))?;
                let fallback = sps.scaling_lists.clone().unwrap_or_else(ScalingLists::default_lists);
                let count8x8 = if transform_8x8_mode { if sps.chroma_format_idc == 3 { 6 } else { 2 } } else { 0 };
                let mut lists = parse_scaling_matrix(&mut r, &fallback, count8x8);
                if !transform_8x8_mode {
                    lists.list8x8 = fallback.list8x8;
                }
                scaling_lists = Some(lists);
            }
            second_chroma_qp_index_offset = r.se();
            if !(-12..=12).contains(&second_chroma_qp_index_offset) {
                return Err(Error::bitstream("PPS: second_chroma_qp_index_offset out of range"));
            }
        }
        r.finish("PPS")?;
        Ok(Pps {
            id,
            sps_id,
            cabac,
            bottom_field_pic_order_in_frame_present,
            num_slice_groups,
            num_ref_idx_l0_default,
            num_ref_idx_l1_default,
            weighted_pred,
            weighted_bipred_idc,
            pic_init_qp,
            pic_init_qs,
            chroma_qp_index_offset,
            second_chroma_qp_index_offset,
            deblocking_filter_control_present,
            constrained_intra_pred,
            redundant_pic_cnt_present,
            transform_8x8_mode,
            scaling_lists,
        })
    }
}
