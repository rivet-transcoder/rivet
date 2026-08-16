//! H.264 / AVC pixel-format detection and SPS/PPS/slice-header parsers.
//! See ITU-T H.264 §7.3.2.x.

use super::bitreader::{
    BitReader, clamp_to_i8, find_next_start_code, more_rbsp_data, remove_h264_rbsp_stuffing,
};
use crate::PixelFormat;

// ─── H.264 SPS parser ─────────────────────────────────────────────
// See ITU-T H.264 §7.3.2.1.1. Profile-gated fields: only profile_idc
// values in { 100, 110, 122, 244, 44, 83, 86, 118, 128, 138, 139,
// 134, 135 } carry the chroma_format_idc + bit_depth fields we want.
// Everything else is 4:2:0 8-bit by spec.
pub(super) fn detect_h264(sample: &[u8]) -> Option<PixelFormat> {
    let sps = find_h264_sps(sample)?;
    let rbsp = remove_h264_rbsp_stuffing(sps);
    let mut br = BitReader::new(&rbsp);

    let profile_idc = br.read_bits(8)? as u8;
    let _constraint_flags = br.read_bits(8)?;
    let _level_idc = br.read_bits(8)?;
    let _seq_parameter_set_id = br.read_ue()?;

    let profile_gates_chroma = matches!(
        profile_idc,
        100 | 110 | 122 | 244 | 44 | 83 | 86 | 118 | 128 | 138 | 139 | 134 | 135
    );

    let (chroma_format_idc, bit_depth_luma) = if profile_gates_chroma {
        let chroma_format_idc = br.read_ue()? as u8;
        if chroma_format_idc == 3 {
            let _separate_colour_plane_flag = br.read_bits(1)?;
        }
        let bit_depth_luma_minus8 = br.read_ue()? as u8;
        (chroma_format_idc, bit_depth_luma_minus8 + 8)
    } else {
        // Baseline / Main / Extended: spec-guaranteed 4:2:0 8-bit.
        (1, 8)
    };

    Some(PixelFormat::from_chroma_and_depth(
        chroma_format_idc,
        bit_depth_luma,
    ))
}

/// Return the SPS RBSP bytes (everything after the nal_unit_type byte,
/// up to but not including the next start code). Handles both 3-byte
/// and 4-byte start codes.
fn find_h264_sps(data: &[u8]) -> Option<&[u8]> {
    let mut i = 0;
    while i + 4 < data.len() {
        let (start_len, nal_byte) = if data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1 {
            (3, i + 3)
        } else if data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 0 && data[i + 3] == 1 {
            (4, i + 4)
        } else {
            i += 1;
            continue;
        };
        if nal_byte >= data.len() {
            return None;
        }
        let nal_unit_type = data[nal_byte] & 0x1F;
        if nal_unit_type == 7 {
            // Skip the NAL unit type byte itself; caller parses the RBSP.
            let start = nal_byte + 1;
            let end = find_next_start_code(&data[start..])
                .map(|off| start + off)
                .unwrap_or(data.len());
            return Some(&data[start..end]);
        }
        i += start_len;
    }
    None
}

// ─── H264SpsInfo struct ────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct H264SpsInfo {
    pub profile_idc: u8,
    /// Packed 8-bit constraint_set_flags (Ch) — constraint_set0..5_flag
    /// in the high 6 bits, 2 reserved_zero bits. Preserved verbatim for
    /// Std struct output.
    pub constraint_set_flags: u8,
    pub level_idc: u8,
    pub chroma_format_idc: u8,
    pub separate_colour_plane_flag: bool,
    pub bit_depth_luma: u8,
    pub bit_depth_chroma: u8,
    pub frame_mbs_only: bool,
    /// Post-crop width in luma samples, or None if the parse stopped
    /// before reaching the cropping fields.
    pub width: Option<u32>,
    pub height: Option<u32>,
    // ─── Slice-header branching predicates (filled by full parse) ─
    /// `log2_max_frame_num_minus4` — slice headers carry
    /// `frame_num` as `u(log2_max_frame_num_minus4 + 4)` bits.
    /// `None` if the dims parse bailed before reaching this field.
    pub log2_max_frame_num_minus4: Option<u8>,
    /// 0 / 1 / 2 per §7.4.2.1. Controls which POC fields the slice
    /// header carries.
    pub pic_order_cnt_type: Option<u8>,
    /// Valid when `pic_order_cnt_type == 0`. Bit width of
    /// `pic_order_cnt_lsb` in the slice header: `log2_max_pic_order_cnt_lsb_minus4 + 4`.
    pub log2_max_pic_order_cnt_lsb_minus4: Option<u8>,
    /// Valid when `pic_order_cnt_type == 1`. Gates the slice header's
    /// `delta_pic_order_cnt[0..1]` branch.
    pub delta_pic_order_always_zero_flag: Option<bool>,
    // ─── Fields needed to build StdVideoH264SequenceParameterSet ──
    pub qpprime_y_zero_transform_bypass_flag: Option<bool>,
    pub seq_scaling_matrix_present_flag: Option<bool>,
    pub max_num_ref_frames: Option<u8>,
    pub gaps_in_frame_num_value_allowed_flag: Option<bool>,
    /// Only meaningful when `!frame_mbs_only`.
    pub mb_adaptive_frame_field_flag: Option<bool>,
    pub direct_8x8_inference_flag: Option<bool>,
    pub frame_cropping_flag: Option<bool>,
    pub frame_crop_left_offset: Option<u32>,
    pub frame_crop_right_offset: Option<u32>,
    pub frame_crop_top_offset: Option<u32>,
    pub frame_crop_bottom_offset: Option<u32>,
    /// Valid when `pic_order_cnt_type == 1`.
    pub offset_for_non_ref_pic: Option<i32>,
    pub offset_for_top_to_bottom_field: Option<i32>,
    pub num_ref_frames_in_pic_order_cnt_cycle: Option<u8>,
    /// Populated only when `pic_order_cnt_type == 1`. Length equals
    /// `num_ref_frames_in_pic_order_cnt_cycle` (0..=255). Spec allows
    /// up to 256 entries but no real-world stream exercises the full
    /// range.
    pub offset_for_ref_frame: Vec<i32>,
}

// ─── Full SPS walker ───────────────────────────────────────────────

/// Full H.264 SPS walker — see §7.3.2.1.1 + §7.4.2.1.1. The parse is
/// greedy: profile_idc + chroma fields are populated first, then we
/// walk the variable-length sections (scaling lists,
/// pic_order_cnt_type branch) to reach pic_width_in_mbs_minus1 etc.
/// If any of those sections hit end-of-buffer the dims come back as
/// None but the early fields are returned.
pub fn parse_h264_sps(sample: &[u8]) -> Option<H264SpsInfo> {
    let sps = find_h264_sps(sample)?;
    let rbsp = remove_h264_rbsp_stuffing(sps);
    let mut br = BitReader::new(&rbsp);

    let profile_idc = br.read_bits(8)? as u8;
    let constraint_set_flags = br.read_bits(8)? as u8;
    let level_idc = br.read_bits(8)? as u8;
    let _seq_parameter_set_id = br.read_ue()?;

    let profile_gates_chroma = matches!(
        profile_idc,
        100 | 110 | 122 | 244 | 44 | 83 | 86 | 118 | 128 | 138 | 139 | 134 | 135
    );

    let (
        chroma_format_idc,
        separate_colour_plane_flag,
        bit_depth_luma,
        bit_depth_chroma,
        qpprime_y_zero,
        scaling_matrix,
    ) = if profile_gates_chroma {
        let chroma = br.read_ue()? as u8;
        let separate = if chroma == 3 {
            br.read_bits(1)? == 1
        } else {
            false
        };
        let bit_depth_luma_m8 = br.read_ue()?;
        let bit_depth_chroma_m8 = br.read_ue()?;
        let qpprime = br.read_bits(1)? == 1;
        let scaling_matrix_present = br.read_bits(1)? == 1;
        if scaling_matrix_present {
            // 8 scaling lists for chroma_format_idc != 3, 12 otherwise
            // (§7.3.2.1.1.1). Each list is size 16 for i<6, 64 otherwise.
            // Deltas are se(v); missing-list flag is u(1).
            let num_lists = if chroma == 3 { 12 } else { 8 };
            for i in 0..num_lists {
                if br.read_bits(1)? == 1 {
                    let size = if i < 6 { 16 } else { 64 };
                    let mut last_scale: i32 = 8;
                    let mut next_scale: i32 = 8;
                    for _j in 0..size {
                        if next_scale != 0 {
                            let delta = br.read_se()?;
                            next_scale = (last_scale + delta + 256).rem_euclid(256);
                        }
                        if next_scale != 0 {
                            last_scale = next_scale;
                        }
                    }
                }
            }
        }
        (
            chroma,
            separate,
            bit_depth_luma_m8 as u8 + 8,
            bit_depth_chroma_m8 as u8 + 8,
            qpprime,
            scaling_matrix_present,
        )
    } else {
        (1u8, false, 8u8, 8u8, false, false)
    };

    // At this point we've cleared the chroma/depth prefix. Everything
    // from here on is what we need for width/height and the slice-
    // header branching predicates. Any read failure below returns the
    // partial info with width/height = None.
    let info_prefix = H264SpsInfo {
        profile_idc,
        constraint_set_flags,
        level_idc,
        chroma_format_idc,
        separate_colour_plane_flag,
        bit_depth_luma,
        bit_depth_chroma,
        frame_mbs_only: true,
        width: None,
        height: None,
        log2_max_frame_num_minus4: None,
        pic_order_cnt_type: None,
        log2_max_pic_order_cnt_lsb_minus4: None,
        delta_pic_order_always_zero_flag: None,
        qpprime_y_zero_transform_bypass_flag: Some(qpprime_y_zero),
        seq_scaling_matrix_present_flag: Some(scaling_matrix),
        max_num_ref_frames: None,
        gaps_in_frame_num_value_allowed_flag: None,
        mb_adaptive_frame_field_flag: None,
        direct_8x8_inference_flag: None,
        frame_cropping_flag: None,
        frame_crop_left_offset: None,
        frame_crop_right_offset: None,
        frame_crop_top_offset: None,
        frame_crop_bottom_offset: None,
        offset_for_non_ref_pic: None,
        offset_for_top_to_bottom_field: None,
        num_ref_frames_in_pic_order_cnt_cycle: None,
        offset_for_ref_frame: Vec::new(),
    };

    let Some(dims) = parse_h264_sps_dims(&mut br, chroma_format_idc, separate_colour_plane_flag)
    else {
        return Some(info_prefix);
    };

    Some(H264SpsInfo {
        frame_mbs_only: dims.frame_mbs_only,
        width: Some(dims.width),
        height: Some(dims.height),
        log2_max_frame_num_minus4: Some(dims.log2_max_frame_num_minus4),
        pic_order_cnt_type: Some(dims.pic_order_cnt_type),
        log2_max_pic_order_cnt_lsb_minus4: dims.log2_max_pic_order_cnt_lsb_minus4,
        delta_pic_order_always_zero_flag: dims.delta_pic_order_always_zero_flag,
        max_num_ref_frames: Some(dims.max_num_ref_frames),
        gaps_in_frame_num_value_allowed_flag: Some(dims.gaps_in_frame_num_value_allowed_flag),
        mb_adaptive_frame_field_flag: dims.mb_adaptive_frame_field_flag,
        direct_8x8_inference_flag: Some(dims.direct_8x8_inference_flag),
        frame_cropping_flag: Some(dims.frame_cropping_flag),
        frame_crop_left_offset: Some(dims.crop_left),
        frame_crop_right_offset: Some(dims.crop_right),
        frame_crop_top_offset: Some(dims.crop_top),
        frame_crop_bottom_offset: Some(dims.crop_bottom),
        offset_for_non_ref_pic: dims.offset_for_non_ref_pic,
        offset_for_top_to_bottom_field: dims.offset_for_top_to_bottom_field,
        num_ref_frames_in_pic_order_cnt_cycle: dims.num_ref_frames_in_pic_order_cnt_cycle,
        offset_for_ref_frame: dims.offset_for_ref_frame,
        ..info_prefix
    })
}

struct H264Dims {
    width: u32,
    height: u32,
    frame_mbs_only: bool,
    log2_max_frame_num_minus4: u8,
    pic_order_cnt_type: u8,
    log2_max_pic_order_cnt_lsb_minus4: Option<u8>,
    delta_pic_order_always_zero_flag: Option<bool>,
    offset_for_non_ref_pic: Option<i32>,
    offset_for_top_to_bottom_field: Option<i32>,
    num_ref_frames_in_pic_order_cnt_cycle: Option<u8>,
    offset_for_ref_frame: Vec<i32>,
    max_num_ref_frames: u8,
    gaps_in_frame_num_value_allowed_flag: bool,
    mb_adaptive_frame_field_flag: Option<bool>,
    direct_8x8_inference_flag: bool,
    frame_cropping_flag: bool,
    crop_left: u32,
    crop_right: u32,
    crop_top: u32,
    crop_bottom: u32,
}

fn parse_h264_sps_dims(
    br: &mut BitReader,
    chroma_format_idc: u8,
    separate_colour_plane_flag: bool,
) -> Option<H264Dims> {
    let log2_max_frame_num_minus4 = br.read_ue()? as u8;
    let pic_order_cnt_type = br.read_ue()? as u8;
    let mut log2_max_pic_order_cnt_lsb_minus4 = None;
    let mut delta_pic_order_always_zero_flag = None;
    let mut offset_for_non_ref_pic = None;
    let mut offset_for_top_to_bottom_field = None;
    let mut num_ref_frames_in_pic_order_cnt_cycle: Option<u8> = None;
    let mut offset_for_ref_frame: Vec<i32> = Vec::new();
    match pic_order_cnt_type {
        0 => {
            log2_max_pic_order_cnt_lsb_minus4 = Some(br.read_ue()? as u8);
        }
        1 => {
            delta_pic_order_always_zero_flag = Some(br.read_bits(1)? == 1);
            offset_for_non_ref_pic = Some(br.read_se()?);
            offset_for_top_to_bottom_field = Some(br.read_se()?);
            let cycle_len = br.read_ue()?;
            // Cap at 255 to fit u8 + bound the loop — spec allows up
            // to 255, so no real loss of precision.
            let capped = cycle_len.min(255) as u8;
            num_ref_frames_in_pic_order_cnt_cycle = Some(capped);
            offset_for_ref_frame.reserve(capped as usize);
            for _ in 0..capped {
                offset_for_ref_frame.push(br.read_se()?);
            }
        }
        2 => { /* no fields */ }
        _ => return None, // reserved; spec says no other values are valid
    }
    let max_num_ref_frames = br.read_ue()?.min(u8::MAX as u32) as u8;
    let gaps_in_frame_num_value_allowed_flag = br.read_bits(1)? == 1;
    let pic_width_in_mbs_minus1 = br.read_ue()?;
    let pic_height_in_map_units_minus1 = br.read_ue()?;
    let frame_mbs_only_flag = br.read_bits(1)?;
    let mut mb_adaptive_frame_field_flag = None;
    if frame_mbs_only_flag == 0 {
        mb_adaptive_frame_field_flag = Some(br.read_bits(1)? == 1);
    }
    let direct_8x8_inference_flag = br.read_bits(1)? == 1;
    let frame_cropping_flag = br.read_bits(1)? == 1;
    let (cl, cr, ct, cb) = if frame_cropping_flag {
        (br.read_ue()?, br.read_ue()?, br.read_ue()?, br.read_ue()?)
    } else {
        (0, 0, 0, 0)
    };

    let pic_width_in_mbs = pic_width_in_mbs_minus1.saturating_add(1);
    let pic_height_in_map_units = pic_height_in_map_units_minus1.saturating_add(1);
    let frame_mbs_only = frame_mbs_only_flag == 1;
    let frame_height_in_mbs = if frame_mbs_only {
        pic_height_in_map_units
    } else {
        pic_height_in_map_units.saturating_mul(2)
    };

    // §6.2 Table 6-1 + §7.4.2.1.1
    let chroma_array_type = if separate_colour_plane_flag {
        0
    } else {
        chroma_format_idc
    };
    let (sub_w, sub_h) = match chroma_array_type {
        0 => (1u32, 1u32), // monochrome (cropping units below use 1,2-flag)
        1 => (2, 2),       // 4:2:0
        2 => (2, 1),       // 4:2:2
        3 => (1, 1),       // 4:4:4
        _ => (1, 1),
    };
    let (crop_x, crop_y) = if chroma_array_type == 0 {
        (1u32, 2u32 - frame_mbs_only_flag)
    } else {
        (sub_w, sub_h * (2 - frame_mbs_only_flag))
    };

    let width = pic_width_in_mbs
        .saturating_mul(16)
        .saturating_sub(crop_x.saturating_mul(cl.saturating_add(cr)));
    let height = frame_height_in_mbs
        .saturating_mul(16)
        .saturating_sub(crop_y.saturating_mul(ct.saturating_add(cb)));

    Some(H264Dims {
        width,
        height,
        frame_mbs_only,
        log2_max_frame_num_minus4,
        pic_order_cnt_type,
        log2_max_pic_order_cnt_lsb_minus4,
        delta_pic_order_always_zero_flag,
        offset_for_non_ref_pic,
        offset_for_top_to_bottom_field,
        num_ref_frames_in_pic_order_cnt_cycle,
        offset_for_ref_frame,
        max_num_ref_frames,
        gaps_in_frame_num_value_allowed_flag,
        mb_adaptive_frame_field_flag,
        direct_8x8_inference_flag,
        frame_cropping_flag,
        crop_left: cl,
        crop_right: cr,
        crop_top: ct,
        crop_bottom: cb,
    })
}

// ─── H.264 first-slice offset ─────────────────────────────────────

/// Scan an Annex-B H.264 sample for the first coded-slice NAL
/// (types 1 / 5 / 19) and return its byte offset within `data`.
/// Parallel to `hevc_first_slice_nal_offset`.
pub fn h264_first_slice_nal_offset(data: &[u8]) -> Option<u32> {
    let mut i = 0;
    while i + 4 < data.len() {
        let (start_len, nal_byte) = if data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1 {
            (3usize, i + 3)
        } else if data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 0 && data[i + 3] == 1 {
            (4usize, i + 4)
        } else {
            i += 1;
            continue;
        };
        if nal_byte >= data.len() {
            return None;
        }
        let t = data[nal_byte] & 0x1F;
        if matches!(t, 1 | 5 | 19) {
            return Some(nal_byte as u32);
        }
        i += start_len;
    }
    None
}

// ─── H.264 PPS parse (Vulkan Video + slice-header support) ────────
//
// Vulkan Video H.264 decode requires the app to build a
// `StdVideoH264PictureParameterSet` from the PPS NAL (type 8) — the
// driver does not parse bitstreams. Every field below lands in the
// Std header struct; the flags pack into bitfields per the Std video
// spec. See ITU-T H.264 §7.3.2.2 + §7.4.2.2.

/// Parsed H.264 PPS fields. Consumers: Vulkan Video decoder (fills
/// `StdVideoH264PictureParameterSet`), slice-header parser (needs
/// `bottom_field_pic_order_in_frame_present_flag` +
/// `redundant_pic_cnt_present_flag` as branching predicates).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct H264PpsInfo {
    pub pic_parameter_set_id: u8,
    pub seq_parameter_set_id: u8,
    pub entropy_coding_mode_flag: bool,
    /// Aka `pic_order_present_flag` in older spec editions. Controls
    /// whether slice headers carry `delta_pic_order_cnt_bottom` and
    /// `delta_pic_order_cnt[1]`.
    pub bottom_field_pic_order_in_frame_present_flag: bool,
    pub num_slice_groups_minus1: u8,
    pub num_ref_idx_l0_default_active_minus1: u8,
    pub num_ref_idx_l1_default_active_minus1: u8,
    pub weighted_pred_flag: bool,
    pub weighted_bipred_idc: u8,
    pub pic_init_qp_minus26: i8,
    pub pic_init_qs_minus26: i8,
    pub chroma_qp_index_offset: i8,
    pub deblocking_filter_control_present_flag: bool,
    pub constrained_intra_pred_flag: bool,
    pub redundant_pic_cnt_present_flag: bool,
    /// Extended fields — present only when the PPS RBSP has trailing
    /// data beyond the baseline syntax. All three were added in the
    /// 2005 amendment alongside High profile.
    pub transform_8x8_mode_flag: Option<bool>,
    pub pic_scaling_matrix_present_flag: Option<bool>,
    pub second_chroma_qp_index_offset: Option<i8>,
}

/// Walk an Annex-B sample looking for the first NAL of type 8 (PPS)
/// and decode its syntax elements. Returns `None` when no PPS is in
/// the sample or the syntax is truncated before
/// `redundant_pic_cnt_present_flag` (the last required field).
///
/// The FMO (Flexible Macroblock Ordering) sub-branches for
/// `num_slice_groups_minus1 > 0` / `slice_group_map_type`=0/2/3..5/6
/// are skipped correctly but not reported — no consumer today needs
/// the slice-group map (FMO is forbidden in Main and High profiles,
/// and every stream our decoder touches is Main/High).
pub fn parse_h264_pps(sample: &[u8]) -> Option<H264PpsInfo> {
    let pps = find_h264_nal_by_type(sample, 8)?;
    let rbsp = remove_h264_rbsp_stuffing(pps);
    let mut br = BitReader::new(&rbsp);

    let pic_parameter_set_id = br.read_ue()? as u8;
    let seq_parameter_set_id = br.read_ue()? as u8;
    let entropy_coding_mode_flag = br.read_bits(1)? == 1;
    let bottom_field_pic_order_in_frame_present_flag = br.read_bits(1)? == 1;

    let num_slice_groups_minus1 = br.read_ue()?;
    if num_slice_groups_minus1 > 0 {
        // FMO sub-branches — skip.
        let slice_group_map_type = br.read_ue()?;
        match slice_group_map_type {
            0 => {
                for _ in 0..=num_slice_groups_minus1 {
                    let _run_length_minus1 = br.read_ue()?;
                }
            }
            2 => {
                for _ in 0..num_slice_groups_minus1 {
                    let _top_left = br.read_ue()?;
                    let _bottom_right = br.read_ue()?;
                }
            }
            3..=5 => {
                let _slice_group_change_direction_flag = br.read_bits(1)?;
                let _slice_group_change_rate_minus1 = br.read_ue()?;
            }
            6 => {
                let pic_size_in_map_units_minus1 = br.read_ue()?;
                let bits = ((num_slice_groups_minus1 + 1) as f64).log2().ceil() as usize;
                let bits = bits.max(1);
                for _ in 0..=pic_size_in_map_units_minus1 {
                    let _slice_group_id = br.read_bits(bits)?;
                }
            }
            _ => {}
        }
    }

    let num_ref_idx_l0_default_active_minus1 = br.read_ue()? as u8;
    let num_ref_idx_l1_default_active_minus1 = br.read_ue()? as u8;
    let weighted_pred_flag = br.read_bits(1)? == 1;
    let weighted_bipred_idc = br.read_bits(2)? as u8;
    let pic_init_qp_minus26 = clamp_to_i8(br.read_se()?);
    let pic_init_qs_minus26 = clamp_to_i8(br.read_se()?);
    let chroma_qp_index_offset = clamp_to_i8(br.read_se()?);
    let deblocking_filter_control_present_flag = br.read_bits(1)? == 1;
    let constrained_intra_pred_flag = br.read_bits(1)? == 1;
    let redundant_pic_cnt_present_flag = br.read_bits(1)? == 1;

    // Extended fields — present only when more_rbsp_data() indicates
    // the PPS carried them. Detect by checking if any bits remain
    // beyond the rbsp_trailing_bits stop. We do a best-effort read:
    // fill from Some(...) on success, fall back to None if the trailer
    // runs out mid-field.
    let (transform_8x8_mode_flag, pic_scaling_matrix_present_flag, second_chroma_qp_index_offset) =
        if more_rbsp_data(&br, &rbsp) {
            let t8 = br.read_bits(1).map(|v| v == 1);
            let psm = br.read_bits(1).map(|v| v == 1);
            // If pic_scaling_matrix_present_flag is set, scaling_list
            // blocks follow before second_chroma_qp_index_offset. Skip
            // them (conservative — we don't consume these values).
            if let Some(true) = psm {
                // Number of scaling lists per §7.3.2.2:
                //   6 + ((chroma_format_idc != 3) ? 2 : 6) * transform_8x8_mode_flag
                // We don't know chroma_format_idc from the PPS alone;
                // assume 4:2:0 (most common) → 8 total lists when t8=1.
                let count = 6 + if let Some(true) = t8 { 2 } else { 0 };
                for i in 0..count {
                    if br.read_bits(1) == Some(1) {
                        let size = if i < 6 { 16 } else { 64 };
                        let mut last_scale: i32 = 8;
                        let mut next_scale: i32 = 8;
                        for _ in 0..size {
                            if next_scale != 0 {
                                let delta = br.read_se().unwrap_or(0);
                                next_scale = (last_scale + delta + 256).rem_euclid(256);
                            }
                            if next_scale != 0 {
                                last_scale = next_scale;
                            }
                        }
                    }
                }
            }
            let s2 = br.read_se().map(clamp_to_i8);
            (t8, psm, s2)
        } else {
            (None, None, None)
        };

    Some(H264PpsInfo {
        pic_parameter_set_id,
        seq_parameter_set_id,
        entropy_coding_mode_flag,
        bottom_field_pic_order_in_frame_present_flag,
        num_slice_groups_minus1: num_slice_groups_minus1.min(u8::MAX as u32) as u8,
        num_ref_idx_l0_default_active_minus1,
        num_ref_idx_l1_default_active_minus1,
        weighted_pred_flag,
        weighted_bipred_idc,
        pic_init_qp_minus26,
        pic_init_qs_minus26,
        chroma_qp_index_offset,
        deblocking_filter_control_present_flag,
        constrained_intra_pred_flag,
        redundant_pic_cnt_present_flag,
        transform_8x8_mode_flag,
        pic_scaling_matrix_present_flag,
        second_chroma_qp_index_offset,
    })
}

// ─── H.264 slice header ───────────────────────────────────────────

/// Slice type name (decoded from `slice_type` ue(v) value). Per
/// H.264 §7.4.3 Table 7-6, values 0..=4 are one iteration of the
/// slice types; values 5..=9 are the same types but mark "all
/// slices in the current picture have this type" (aka
/// `slice_type_all_same`). Both halves collapse to the same enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum H264SliceType {
    P,
    B,
    I,
    SP,
    SI,
}

impl H264SliceType {
    pub(super) fn from_ue(v: u32) -> Option<Self> {
        match v % 5 {
            0 => Some(Self::P),
            1 => Some(Self::B),
            2 => Some(Self::I),
            3 => Some(Self::SP),
            4 => Some(Self::SI),
            _ => None,
        }
    }
}

/// Parsed H.264 slice header — just the fields the Vulkan Video
/// decoder + our DPB manager need. See ITU-T H.264 §7.3.3. Full slice
/// header has ref_pic_list_modification, weighted_prediction tables,
/// dec_ref_pic_marking, etc., which we don't consume (the driver
/// re-derives them from the PPS + `StdVideoDecodeH264PictureInfo`
/// flags).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct H264SliceHeader {
    pub first_mb_in_slice: u32,
    pub slice_type: H264SliceType,
    pub pic_parameter_set_id: u8,
    /// From the NAL header: `nal_unit_type == 5` — set by the caller
    /// when it picks the NAL to parse. Affects whether `idr_pic_id` is
    /// carried.
    pub is_idr: bool,
    pub frame_num: u32,
    /// True when the slice encodes a single field of an interlaced
    /// frame (spec: `!frame_mbs_only_flag && field_pic_flag`). False
    /// for progressive frames or MBAFF pairs.
    pub field_pic_flag: bool,
    pub bottom_field_flag: bool,
    pub colour_plane_id: Option<u8>,
    /// Set when `is_idr`; otherwise `None`.
    pub idr_pic_id: Option<u32>,
    /// Set when SPS `pic_order_cnt_type == 0`.
    pub pic_order_cnt_lsb: Option<u32>,
    pub delta_pic_order_cnt_bottom: Option<i32>,
    /// Set when SPS `pic_order_cnt_type == 1` and
    /// `!delta_pic_order_always_zero_flag`. `[0]` always present in
    /// that branch, `[1]` present only when the PPS carried
    /// `bottom_field_pic_order_in_frame_present_flag` and we're in a
    /// frame (not field) slice.
    pub delta_pic_order_cnt: [Option<i32>; 2],
}

/// Parse the first slice-NAL in `sample`, using the SPS + PPS for
/// branch predicates. The NAL header's `nal_unit_type` gates which
/// slice types we accept: 1 (non-IDR), 5 (IDR), 19 (auxiliary coded
/// slice) all share the same syntax. Returns `None` when the sample
/// contains no slice NAL or the SPS/PPS didn't provide the required
/// context (e.g., SPS `pic_order_cnt_type` was `None` so we can't
/// branch into the POC reads).
pub fn parse_h264_slice_header(
    sample: &[u8],
    sps: &H264SpsInfo,
    pps: &H264PpsInfo,
) -> Option<H264SliceHeader> {
    // nal_unit_type values for coded slices: 1 (non-IDR), 2/3/4
    // (partition A/B/C, deprecated), 5 (IDR), 19 (aux). We accept
    // 1, 5, 19 — the common cases.
    let (nal_type, rbsp) = find_h264_slice_nal(sample)?;
    let is_idr = nal_type == 5;

    let mut br = BitReader::new(&rbsp);
    let first_mb_in_slice = br.read_ue()?;
    let slice_type_code = br.read_ue()?;
    let slice_type = H264SliceType::from_ue(slice_type_code)?;
    let pic_parameter_set_id = br.read_ue()? as u8;

    let colour_plane_id = if sps.separate_colour_plane_flag {
        Some(br.read_bits(2)? as u8)
    } else {
        None
    };

    let frame_num_bits = (sps.log2_max_frame_num_minus4? as usize) + 4;
    let frame_num = br.read_bits(frame_num_bits)?;

    let (field_pic_flag, bottom_field_flag) = if !sps.frame_mbs_only {
        let f = br.read_bits(1)? == 1;
        let b = if f { br.read_bits(1)? == 1 } else { false };
        (f, b)
    } else {
        (false, false)
    };

    let idr_pic_id = if is_idr { Some(br.read_ue()?) } else { None };

    let poc_type = sps.pic_order_cnt_type?;
    let mut pic_order_cnt_lsb = None;
    let mut delta_pic_order_cnt_bottom = None;
    let mut delta_pic_order_cnt: [Option<i32>; 2] = [None, None];
    match poc_type {
        0 => {
            let bits = (sps.log2_max_pic_order_cnt_lsb_minus4? as usize) + 4;
            pic_order_cnt_lsb = Some(br.read_bits(bits)?);
            if pps.bottom_field_pic_order_in_frame_present_flag && !field_pic_flag {
                delta_pic_order_cnt_bottom = Some(br.read_se()?);
            }
        }
        1 => {
            let always_zero = sps.delta_pic_order_always_zero_flag.unwrap_or(false);
            if !always_zero {
                delta_pic_order_cnt[0] = Some(br.read_se()?);
                if pps.bottom_field_pic_order_in_frame_present_flag && !field_pic_flag {
                    delta_pic_order_cnt[1] = Some(br.read_se()?);
                }
            }
        }
        2 => { /* implicit POC derivation; no fields */ }
        _ => return None,
    }

    Some(H264SliceHeader {
        first_mb_in_slice,
        slice_type,
        pic_parameter_set_id,
        is_idr,
        frame_num,
        field_pic_flag,
        bottom_field_flag,
        colour_plane_id,
        idr_pic_id,
        pic_order_cnt_lsb,
        delta_pic_order_cnt_bottom,
        delta_pic_order_cnt,
    })
}

/// Find the first coded-slice NAL (nal_unit_type ∈ {1, 5, 19}) in
/// `data` and return `(nal_unit_type, rbsp_bytes_with_stuffing_removed)`.
fn find_h264_slice_nal(data: &[u8]) -> Option<(u8, Vec<u8>)> {
    let mut i = 0;
    while i + 4 < data.len() {
        let (start_len, nal_byte) = if data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1 {
            (3, i + 3)
        } else if data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 0 && data[i + 3] == 1 {
            (4, i + 4)
        } else {
            i += 1;
            continue;
        };
        if nal_byte >= data.len() {
            return None;
        }
        let nal_unit_type = data[nal_byte] & 0x1F;
        if matches!(nal_unit_type, 1 | 5 | 19) {
            let start = nal_byte + 1;
            let end = find_next_start_code(&data[start..])
                .map(|off| start + off)
                .unwrap_or(data.len());
            let rbsp = remove_h264_rbsp_stuffing(&data[start..end]);
            return Some((nal_unit_type, rbsp));
        }
        i += start_len;
    }
    None
}

/// Generic "find the first Annex-B NAL whose `nal_unit_type` matches
/// `target_type`" helper. Factored out of `find_h264_sps` so the PPS
/// parser and future consumers (slice header, SEI) share one scanner.
fn find_h264_nal_by_type(data: &[u8], target_type: u8) -> Option<&[u8]> {
    let mut i = 0;
    while i + 4 < data.len() {
        let (start_len, nal_byte) = if data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1 {
            (3, i + 3)
        } else if data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 0 && data[i + 3] == 1 {
            (4, i + 4)
        } else {
            i += 1;
            continue;
        };
        if nal_byte >= data.len() {
            return None;
        }
        let nal_unit_type = data[nal_byte] & 0x1F;
        if nal_unit_type == target_type {
            let start = nal_byte + 1;
            let end = find_next_start_code(&data[start..])
                .map(|off| start + off)
                .unwrap_or(data.len());
            return Some(&data[start..end]);
        }
        i += start_len;
    }
    None
}
