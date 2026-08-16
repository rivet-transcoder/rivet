//! HEVC / H.265 pixel-format detection and SPS/VPS/PPS/slice-header parsers.
//! See ITU-T H.265 §7.3.2.x.

use super::bitreader::{BitReader, clamp_to_i8, find_next_start_code, remove_h264_rbsp_stuffing};

// ─── HEVC SPS parser ──────────────────────────────────────────────
// See ITU-T H.265 §7.3.2.2.1. We skip profile_tier_level and jump to
// chroma_format_idc + bit_depth_luma_minus8 + bit_depth_chroma_minus8.
pub(super) fn detect_hevc(sample: &[u8]) -> Option<crate::PixelFormat> {
    let sps = find_hevc_sps(sample)?;
    let rbsp = remove_h264_rbsp_stuffing(sps);
    let mut br = BitReader::new(&rbsp);

    let _sps_video_parameter_set_id = br.read_bits(4)?;
    let sps_max_sub_layers_minus1 = br.read_bits(3)? as usize;
    let _sps_temporal_id_nesting_flag = br.read_bits(1)?;

    // profile_tier_level: 88 bits for general, plus sub-layer loops.
    // The widths are fixed — we skip by the exact bit count instead
    // of semantically parsing.
    skip_hevc_profile_tier_level(&mut br, sps_max_sub_layers_minus1)?;

    let _sps_seq_parameter_set_id = br.read_ue()?;
    let chroma_format_idc = br.read_ue()? as u8;
    if chroma_format_idc == 3 {
        let _separate_colour_plane_flag = br.read_bits(1)?;
    }
    let _pic_width = br.read_ue()?;
    let _pic_height = br.read_ue()?;
    let conformance_window_flag = br.read_bits(1)?;
    if conformance_window_flag == 1 {
        let _ = br.read_ue()?;
        let _ = br.read_ue()?;
        let _ = br.read_ue()?;
        let _ = br.read_ue()?;
    }
    let bit_depth_luma = br.read_ue()? as u8 + 8;
    let _bit_depth_chroma_minus8 = br.read_ue()?;

    Some(crate::PixelFormat::from_chroma_and_depth(
        chroma_format_idc,
        bit_depth_luma,
    ))
}

fn find_hevc_sps(data: &[u8]) -> Option<&[u8]> {
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
        if nal_byte + 1 >= data.len() {
            return None;
        }
        // HEVC NAL header is 2 bytes; nal_unit_type is bits 1..7 of byte 0.
        let nal_unit_type = (data[nal_byte] >> 1) & 0x3F;
        if nal_unit_type == 33 {
            // Skip the 2-byte NAL header; RBSP starts after.
            let start = nal_byte + 2;
            let end = find_next_start_code(&data[start..])
                .map(|off| start + off)
                .unwrap_or(data.len());
            return Some(&data[start..end]);
        }
        i += start_len;
    }
    None
}

pub(super) fn skip_hevc_profile_tier_level(
    br: &mut BitReader,
    max_sub_layers_minus1: usize,
) -> Option<()> {
    // general_profile_space(2) + general_tier_flag(1) + general_profile_idc(5)
    let _ = br.read_bits(8)?;
    // general_profile_compatibility_flag[32]
    let _ = br.read_bits(32)?;
    // general_progressive_source_flag + interlaced + non_packed + frame_only +
    // 43 reserved + general_inbld/one_picture_only + level_idc
    let _ = br.read_bits(48)?;
    let _ = br.read_bits(8)?;

    // Sub-layer flags
    let mut sub_layer_profile_present = Vec::with_capacity(max_sub_layers_minus1);
    let mut sub_layer_level_present = Vec::with_capacity(max_sub_layers_minus1);
    for _ in 0..max_sub_layers_minus1 {
        sub_layer_profile_present.push(br.read_bits(1)?);
        sub_layer_level_present.push(br.read_bits(1)?);
    }
    if max_sub_layers_minus1 > 0 {
        // 2 bits reserved × (8 - max_sub_layers_minus1) — spec-mandated padding
        for _ in max_sub_layers_minus1..8 {
            let _ = br.read_bits(2)?;
        }
    }
    for i in 0..max_sub_layers_minus1 {
        if sub_layer_profile_present[i] == 1 {
            let _ = br.read_bits(8)?;
            let _ = br.read_bits(32)?;
            let _ = br.read_bits(48)?;
        }
        if sub_layer_level_present[i] == 1 {
            let _ = br.read_bits(8)?;
        }
    }
    Some(())
}

// ─── HevcSpsInfo struct ────────────────────────────────────────────

/// Parsed HEVC SPS fields relevant to the pipeline.
///
/// Width/height are post-conformance-window (§7.4.3.2.1): per the spec,
/// output luma dimensions = `pic_width_in_luma_samples - SubWidthC *
/// (conf_win_left + conf_win_right)` (and analogously for height).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct HevcSpsInfo {
    pub sps_video_parameter_set_id: u8,
    pub sps_seq_parameter_set_id: u8,
    pub sps_max_sub_layers_minus1: u8,
    pub sps_temporal_id_nesting_flag: bool,
    pub chroma_format_idc: u8,
    pub separate_colour_plane_flag: bool,
    pub bit_depth_luma: u8,
    pub bit_depth_chroma: u8,
    pub width: Option<u32>,
    pub height: Option<u32>,
    /// Post-conformance-window crop offsets in chroma samples.
    pub conf_win_left_offset: u32,
    pub conf_win_right_offset: u32,
    pub conf_win_top_offset: u32,
    pub conf_win_bottom_offset: u32,
    pub log2_max_pic_order_cnt_lsb_minus4: u8,
    pub log2_min_luma_coding_block_size_minus3: u8,
    pub log2_diff_max_min_luma_coding_block_size: u8,
    pub log2_min_luma_transform_block_size_minus2: u8,
    pub log2_diff_max_min_luma_transform_block_size: u8,
    pub max_transform_hierarchy_depth_inter: u8,
    pub max_transform_hierarchy_depth_intra: u8,
    pub scaling_list_enabled_flag: bool,
    pub sps_sub_layer_ordering_info_present_flag: bool,
    pub amp_enabled_flag: bool,
    pub sample_adaptive_offset_enabled_flag: bool,
    pub pcm_enabled_flag: bool,
    /// Only meaningful when pcm_enabled_flag is set; defaults to false.
    pub pcm_loop_filter_disabled_flag: bool,
    pub num_short_term_ref_pic_sets: u8,
    pub long_term_ref_pics_present_flag: bool,
    pub sps_temporal_mvp_enabled_flag: bool,
    pub strong_intra_smoothing_enabled_flag: bool,
    pub profile_idc: u8,
    pub level_idc: u8,
    pub tier_flag: bool,
    /// Sub-layer DPB management triple, one per sub-layer. Index 0..=sps_max_sub_layers_minus1
    /// are populated; indices above are left at defaults. Vulkan Video
    /// requires these to be conveyed via `StdVideoH265DecPicBufMgr`.
    pub max_dec_pic_buffering_minus1: [u8; 7],
    pub max_num_reorder_pics: [u8; 7],
    pub max_latency_increase_plus1: [u32; 7],
    /// `profile_compatibility_flag[32]` — high bit at index 0. Needed
    /// for the Std PTL struct.
    pub profile_compatibility_flags: u32,
    /// `general_profile_space` (2 bits) — almost always 0. Part of the
    /// `hvc1.*` codec string prefix.
    pub general_profile_space: u8,
    /// `general_constraint_indicator_flags` (48 bits), right-aligned in a u64
    /// (byte 0 = bits 47..40). Emitted as the trailing `.XX` segments of the
    /// `hvc1.*` codec string.
    pub general_constraint_flags: u64,
}

/// Parsed HEVC VPS — minimum fields needed for StdVideoH265VideoParameterSet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct H265VpsInfo {
    pub vps_video_parameter_set_id: u8,
    pub vps_max_sub_layers_minus1: u8,
    pub vps_temporal_id_nesting_flag: bool,
    pub profile_idc: u8,
    pub level_idc: u8,
    pub tier_flag: bool,
}

/// Parsed HEVC PPS.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct H265PpsInfo {
    pub pps_pic_parameter_set_id: u8,
    pub pps_seq_parameter_set_id: u8,
    pub dependent_slice_segments_enabled_flag: bool,
    pub output_flag_present_flag: bool,
    pub num_extra_slice_header_bits: u8,
    pub sign_data_hiding_enabled_flag: bool,
    pub cabac_init_present_flag: bool,
    pub num_ref_idx_l0_default_active_minus1: u8,
    pub num_ref_idx_l1_default_active_minus1: u8,
    pub init_qp_minus26: i8,
    pub constrained_intra_pred_flag: bool,
    pub transform_skip_enabled_flag: bool,
    pub cu_qp_delta_enabled_flag: bool,
    pub diff_cu_qp_delta_depth: u8,
    pub pps_cb_qp_offset: i8,
    pub pps_cr_qp_offset: i8,
    pub pps_slice_chroma_qp_offsets_present_flag: bool,
    pub weighted_pred_flag: bool,
    pub weighted_bipred_flag: bool,
    pub transquant_bypass_enabled_flag: bool,
    pub tiles_enabled_flag: bool,
    pub entropy_coding_sync_enabled_flag: bool,
    // Tile layout (§7.3.2.3) — only meaningful when tiles_enabled_flag.
    // Defaults below model a 1×1 uniform tile spanning the frame.
    pub num_tile_columns_minus1: u8,
    pub num_tile_rows_minus1: u8,
    pub uniform_spacing_flag: bool,
    pub loop_filter_across_tiles_enabled_flag: bool,
    // Slice / deblocking / merge controls
    pub pps_loop_filter_across_slices_enabled_flag: bool,
    pub deblocking_filter_control_present_flag: bool,
    pub deblocking_filter_override_enabled_flag: bool,
    pub pps_deblocking_filter_disabled_flag: bool,
    pub pps_beta_offset_div2: i8,
    pub pps_tc_offset_div2: i8,
    pub pps_scaling_list_data_present_flag: bool,
    pub lists_modification_present_flag: bool,
    pub log2_parallel_merge_level_minus2: u8,
    pub slice_segment_header_extension_present_flag: bool,
    pub pps_extension_present_flag: bool,
}

/// HEVC slice header — subset needed for StdVideoDecodeH265PictureInfo.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct H265SliceHeader {
    pub first_slice_segment_in_pic_flag: bool,
    pub nal_unit_type: u8,
    pub slice_pic_parameter_set_id: u8,
    pub slice_type: H265SliceType,
    pub pic_order_cnt_lsb: u32,
    pub short_term_ref_pic_set_sps_flag: bool,
    pub short_term_ref_pic_set_idx: Option<u8>,
    /// True for IRAP pictures (IDR / CRA / BLA): nal_unit_type ∈ 16..=23.
    pub is_irap: bool,
    /// True for IDR specifically: nal_unit_type ∈ 19..=20.
    pub is_idr: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum H265SliceType {
    B,
    P,
    I,
}

impl H265SliceType {
    fn from_ue(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::B),
            1 => Some(Self::P),
            2 => Some(Self::I),
            _ => None,
        }
    }
}

// ─── Full SPS walker ───────────────────────────────────────────────

/// Full HEVC SPS walker — see H.265 §7.3.2.2.1 + §7.4.3.2.1. Consumes
/// `profile_tier_level` via the existing `skip_hevc_profile_tier_level`
/// helper, then reads pic_width_in_luma_samples + pic_height_in_luma_samples
/// and applies the conformance window crop if present.
pub fn parse_hevc_sps(sample: &[u8]) -> Option<HevcSpsInfo> {
    let sps = find_hevc_sps(sample)?;
    let rbsp = remove_h264_rbsp_stuffing(sps);
    let mut br = BitReader::new(&rbsp);

    let sps_video_parameter_set_id = br.read_bits(4)? as u8;
    let sps_max_sub_layers_minus1 = br.read_bits(3)? as u8;
    let sps_temporal_id_nesting_flag = br.read_bits(1)? == 1;
    // profile_tier_level: capture general_profile_idc + tier + level
    // for the VPS mirror + Std struct. The rest is skipped via the
    // same helper we already had.
    let general_profile_space = br.read_bits(2)? as u8;
    let tier_flag = br.read_bits(1)? == 1;
    let profile_idc = br.read_bits(5)? as u8;
    // general_profile_compatibility_flag[32] — captured for Std PTL + codec str.
    let profile_compatibility_flags = br.read_bits(32)?;
    // general_constraint_indicator_flags (48 bits) — captured for the hvc1.*
    // codec string's trailing constraint bytes. Read as two halves because
    // BitReader::read_bits returns a u32.
    let constraint_hi = br.read_bits(24)? as u64;
    let constraint_lo = br.read_bits(24)? as u64;
    let general_constraint_flags = (constraint_hi << 24) | constraint_lo;
    let level_idc = br.read_bits(8)? as u8;
    // Skip sub-layer profile/level blocks — matches
    // skip_hevc_profile_tier_level's tail logic.
    let mut spl = Vec::with_capacity(sps_max_sub_layers_minus1 as usize);
    let mut sll = Vec::with_capacity(sps_max_sub_layers_minus1 as usize);
    for _ in 0..sps_max_sub_layers_minus1 {
        spl.push(br.read_bits(1)?);
        sll.push(br.read_bits(1)?);
    }
    if sps_max_sub_layers_minus1 > 0 {
        for _ in sps_max_sub_layers_minus1 as usize..8 {
            let _ = br.read_bits(2)?;
        }
    }
    for i in 0..sps_max_sub_layers_minus1 as usize {
        if spl[i] == 1 {
            let _ = br.read_bits(8)?;
            let _ = br.read_bits(32)?;
            let _ = br.read_bits(48)?;
        }
        if sll[i] == 1 {
            let _ = br.read_bits(8)?;
        }
    }

    let sps_seq_parameter_set_id = br.read_ue()? as u8;
    let chroma_format_idc = br.read_ue()? as u8;
    let separate_colour_plane_flag = if chroma_format_idc == 3 {
        br.read_bits(1)? == 1
    } else {
        false
    };
    let pic_width = br.read_ue()?;
    let pic_height = br.read_ue()?;
    let conformance_window_flag = br.read_bits(1)?;
    let (cl, cr, ct, cb) = if conformance_window_flag == 1 {
        (br.read_ue()?, br.read_ue()?, br.read_ue()?, br.read_ue()?)
    } else {
        (0u32, 0u32, 0u32, 0u32)
    };
    let bit_depth_luma_m8 = br.read_ue()?;
    let bit_depth_chroma_m8 = br.read_ue()?;
    let log2_max_pic_order_cnt_lsb_minus4 = br.read_ue()? as u8;

    // sps_sub_layer_ordering_info_present_flag branch.
    // Spec §7.3.2.2.1: when the flag is 0 only the top sub-layer's
    // triple is signalled, but the DPB buf-mgr should mirror that
    // value across all sub-layers i < max_sub_layers_minus1. We do
    // that unification here so Std DecPicBufMgr has all entries
    // populated regardless of how the bitstream flagged them.
    let sps_sub_layer_ordering_info_present_flag = br.read_bits(1)? == 1;
    let mut max_dec_pic_buffering_minus1 = [0u8; 7];
    let mut max_num_reorder_pics = [0u8; 7];
    let mut max_latency_increase_plus1 = [0u32; 7];
    let start = if sps_sub_layer_ordering_info_present_flag {
        0
    } else {
        sps_max_sub_layers_minus1
    };
    for i in start..=sps_max_sub_layers_minus1 {
        let dec = br.read_ue()?;
        let nro = br.read_ue()?;
        let latency = br.read_ue()?;
        let idx = (i as usize).min(6);
        max_dec_pic_buffering_minus1[idx] = dec.min(u8::MAX as u32) as u8;
        max_num_reorder_pics[idx] = nro.min(u8::MAX as u32) as u8;
        max_latency_increase_plus1[idx] = latency;
    }
    // Fill unsignalled lower sub-layers with the top-layer values.
    if !sps_sub_layer_ordering_info_present_flag {
        let top = sps_max_sub_layers_minus1 as usize;
        for i in 0..top {
            max_dec_pic_buffering_minus1[i] = max_dec_pic_buffering_minus1[top];
            max_num_reorder_pics[i] = max_num_reorder_pics[top];
            max_latency_increase_plus1[i] = max_latency_increase_plus1[top];
        }
    }

    let log2_min_luma_coding_block_size_minus3 = br.read_ue()? as u8;
    let log2_diff_max_min_luma_coding_block_size = br.read_ue()? as u8;
    let log2_min_luma_transform_block_size_minus2 = br.read_ue()? as u8;
    let log2_diff_max_min_luma_transform_block_size = br.read_ue()? as u8;
    let max_transform_hierarchy_depth_inter = br.read_ue()? as u8;
    let max_transform_hierarchy_depth_intra = br.read_ue()? as u8;

    let scaling_list_enabled_flag = br.read_bits(1)? == 1;
    if scaling_list_enabled_flag {
        let sps_scaling_list_data_present_flag = br.read_bits(1)? == 1;
        if sps_scaling_list_data_present_flag {
            skip_hevc_scaling_list_data(&mut br)?;
        }
    }
    let amp_enabled_flag = br.read_bits(1)? == 1;
    let sample_adaptive_offset_enabled_flag = br.read_bits(1)? == 1;
    let pcm_enabled_flag = br.read_bits(1)? == 1;
    let mut pcm_loop_filter_disabled_flag = false;
    if pcm_enabled_flag {
        let _pcm_sample_bit_depth_luma_minus1 = br.read_bits(4)?;
        let _pcm_sample_bit_depth_chroma_minus1 = br.read_bits(4)?;
        let _log2_min_pcm_luma_cb_size_minus3 = br.read_ue()?;
        let _log2_diff_max_min_pcm_luma_cb_size = br.read_ue()?;
        pcm_loop_filter_disabled_flag = br.read_bits(1)? == 1;
    }
    let num_short_term_ref_pic_sets = br.read_ue()? as u8;
    // Skip the short-term RPS syntax parsing — we don't need the
    // values to build Std SPS, but we do need to advance past them.
    // The full parse is complex; use a conservative skip that
    // tolerates simple streams. For a production decoder, this needs
    // a proper RPS parser — this is a scaffold.
    let mut st_rps_offsets: Vec<()> = Vec::with_capacity(num_short_term_ref_pic_sets as usize);
    for rps_idx in 0..num_short_term_ref_pic_sets {
        skip_hevc_short_term_rps(&mut br, rps_idx, num_short_term_ref_pic_sets)?;
        st_rps_offsets.push(());
    }
    let long_term_ref_pics_present_flag = br.read_bits(1)? == 1;
    if long_term_ref_pics_present_flag {
        let num_long_term_ref_pics_sps = br.read_ue()?;
        let lsb_bits = (log2_max_pic_order_cnt_lsb_minus4 as usize) + 4;
        for _ in 0..num_long_term_ref_pics_sps {
            let _lt_ref_pic_poc_lsb_sps = br.read_bits(lsb_bits)?;
            let _used_by_curr_pic_lt_sps_flag = br.read_bits(1)?;
        }
    }
    let sps_temporal_mvp_enabled_flag = br.read_bits(1)? == 1;
    let strong_intra_smoothing_enabled_flag = br.read_bits(1)? == 1;
    // vui / extension — stop here.

    let chroma_array_type = if separate_colour_plane_flag {
        0
    } else {
        chroma_format_idc
    };
    let (sub_w, sub_h) = match chroma_array_type {
        0 => (1u32, 1u32),
        1 => (2, 2),
        2 => (2, 1),
        3 => (1, 1),
        _ => (1, 1),
    };
    let width = pic_width.saturating_sub(sub_w.saturating_mul(cl.saturating_add(cr)));
    let height = pic_height.saturating_sub(sub_h.saturating_mul(ct.saturating_add(cb)));

    Some(HevcSpsInfo {
        sps_video_parameter_set_id,
        sps_seq_parameter_set_id,
        sps_max_sub_layers_minus1,
        sps_temporal_id_nesting_flag,
        chroma_format_idc,
        separate_colour_plane_flag,
        bit_depth_luma: bit_depth_luma_m8 as u8 + 8,
        bit_depth_chroma: bit_depth_chroma_m8 as u8 + 8,
        width: Some(width),
        height: Some(height),
        conf_win_left_offset: cl,
        conf_win_right_offset: cr,
        conf_win_top_offset: ct,
        conf_win_bottom_offset: cb,
        log2_max_pic_order_cnt_lsb_minus4,
        log2_min_luma_coding_block_size_minus3,
        log2_diff_max_min_luma_coding_block_size,
        log2_min_luma_transform_block_size_minus2,
        log2_diff_max_min_luma_transform_block_size,
        max_transform_hierarchy_depth_inter,
        max_transform_hierarchy_depth_intra,
        scaling_list_enabled_flag,
        sps_sub_layer_ordering_info_present_flag,
        amp_enabled_flag,
        sample_adaptive_offset_enabled_flag,
        pcm_enabled_flag,
        pcm_loop_filter_disabled_flag,
        num_short_term_ref_pic_sets,
        long_term_ref_pics_present_flag,
        sps_temporal_mvp_enabled_flag,
        strong_intra_smoothing_enabled_flag,
        profile_idc,
        level_idc,
        tier_flag,
        max_dec_pic_buffering_minus1,
        max_num_reorder_pics,
        max_latency_increase_plus1,
        profile_compatibility_flags,
        general_profile_space,
        general_constraint_flags,
    })
}

/// Skip HEVC scaling_list_data() syntax — §7.3.4. Four size IDs,
/// each size 4..=64 depending on sizeId + matrixId. For Std SPS
/// construction we skip the values; they're only needed when we
/// convey them in StdVideoH265ScalingLists (not currently wired).
fn skip_hevc_scaling_list_data(br: &mut BitReader) -> Option<()> {
    for size_id in 0..4 {
        let matrix_count = if size_id == 3 { 2 } else { 6 };
        for _matrix_id in 0..matrix_count {
            let scaling_list_pred_mode_flag = br.read_bits(1)? == 1;
            if !scaling_list_pred_mode_flag {
                let _scaling_list_pred_matrix_id_delta = br.read_ue()?;
            } else {
                let coef_num: usize = (1 << (4 + (size_id << 1))).min(64);
                if size_id > 1 {
                    let _scaling_list_dc_coef_minus8 = br.read_se()?;
                }
                for _ in 0..coef_num {
                    let _scaling_list_delta_coef = br.read_se()?;
                }
            }
        }
    }
    Some(())
}

/// Skip HEVC short_term_ref_pic_set(stRpsIdx) — §7.3.7. Complex;
/// we advance past the bits without populating state (we don't
/// need the values to build Std SPS).
fn skip_hevc_short_term_rps(br: &mut BitReader, st_rps_idx: u8, num_st_rps: u8) -> Option<()> {
    let inter_ref_pic_set_prediction_flag = if st_rps_idx != 0 {
        br.read_bits(1)? == 1
    } else {
        false
    };
    if inter_ref_pic_set_prediction_flag {
        if st_rps_idx == num_st_rps {
            let _delta_idx_minus1 = br.read_ue()?;
        }
        let _delta_rps_sign = br.read_bits(1)?;
        let _abs_delta_rps_minus1 = br.read_ue()?;
        // Per spec, NumDeltaPocs[RefRpsIdx] — we don't track that.
        // Approximation: assume up to 16 entries; each entry is
        // 1-2 bits. This works for typical streams but is a
        // known gap. A production parser needs real state tracking.
        for _ in 0..16 {
            let used = br.read_bits(1)?;
            if used == 0 {
                let _use_delta_flag = br.read_bits(1)?;
            }
        }
    } else {
        let num_negative_pics = br.read_ue()?;
        let num_positive_pics = br.read_ue()?;
        for _ in 0..num_negative_pics {
            let _delta_poc_s0_minus1 = br.read_ue()?;
            let _used_by_curr_pic_s0_flag = br.read_bits(1)?;
        }
        for _ in 0..num_positive_pics {
            let _delta_poc_s1_minus1 = br.read_ue()?;
            let _used_by_curr_pic_s1_flag = br.read_bits(1)?;
        }
    }
    Some(())
}

// ─── VPS parser ────────────────────────────────────────────────────

/// Parse the HEVC VPS (NAL type 32). Minimum fields for Std VPS.
pub fn parse_h265_vps(sample: &[u8]) -> Option<H265VpsInfo> {
    let nal = find_hevc_nal_by_type(sample, 32)?;
    let rbsp = remove_h264_rbsp_stuffing(nal);
    let mut br = BitReader::new(&rbsp);
    let vps_video_parameter_set_id = br.read_bits(4)? as u8;
    let _vps_base_layer_internal_flag = br.read_bits(1)?;
    let _vps_base_layer_available_flag = br.read_bits(1)?;
    let _vps_max_layers_minus1 = br.read_bits(6)?;
    let vps_max_sub_layers_minus1 = br.read_bits(3)? as u8;
    let vps_temporal_id_nesting_flag = br.read_bits(1)? == 1;
    let _vps_reserved_0xffff_16bits = br.read_bits(16)?;
    // profile_tier_level — reuse the pattern. We only need profile/
    // tier/level for the Std VPS + for our own info.
    let _gp_space = br.read_bits(2)?;
    let tier_flag = br.read_bits(1)? == 1;
    let profile_idc = br.read_bits(5)? as u8;
    let _ = br.read_bits(32)?; // profile_compatibility_flag
    let _ = br.read_bits(48)?; // constraint flags
    let level_idc = br.read_bits(8)? as u8;
    Some(H265VpsInfo {
        vps_video_parameter_set_id,
        vps_max_sub_layers_minus1,
        vps_temporal_id_nesting_flag,
        profile_idc,
        level_idc,
        tier_flag,
    })
}

// ─── PPS parser ────────────────────────────────────────────────────

/// Parse the HEVC PPS (NAL type 34). Subset needed for Std PPS.
pub fn parse_h265_pps(sample: &[u8]) -> Option<H265PpsInfo> {
    let nal = find_hevc_nal_by_type(sample, 34)?;
    let rbsp = remove_h264_rbsp_stuffing(nal);
    let mut br = BitReader::new(&rbsp);
    let pps_pic_parameter_set_id = br.read_ue()? as u8;
    let pps_seq_parameter_set_id = br.read_ue()? as u8;
    let dependent_slice_segments_enabled_flag = br.read_bits(1)? == 1;
    let output_flag_present_flag = br.read_bits(1)? == 1;
    let num_extra_slice_header_bits = br.read_bits(3)? as u8;
    let sign_data_hiding_enabled_flag = br.read_bits(1)? == 1;
    let cabac_init_present_flag = br.read_bits(1)? == 1;
    let num_ref_idx_l0_default_active_minus1 = br.read_ue()? as u8;
    let num_ref_idx_l1_default_active_minus1 = br.read_ue()? as u8;
    let init_qp_minus26 = clamp_to_i8(br.read_se()?);
    let constrained_intra_pred_flag = br.read_bits(1)? == 1;
    let transform_skip_enabled_flag = br.read_bits(1)? == 1;
    let cu_qp_delta_enabled_flag = br.read_bits(1)? == 1;
    let diff_cu_qp_delta_depth = if cu_qp_delta_enabled_flag {
        br.read_ue()? as u8
    } else {
        0
    };
    let pps_cb_qp_offset = clamp_to_i8(br.read_se()?);
    let pps_cr_qp_offset = clamp_to_i8(br.read_se()?);
    let pps_slice_chroma_qp_offsets_present_flag = br.read_bits(1)? == 1;
    let weighted_pred_flag = br.read_bits(1)? == 1;
    let weighted_bipred_flag = br.read_bits(1)? == 1;
    let transquant_bypass_enabled_flag = br.read_bits(1)? == 1;
    let tiles_enabled_flag = br.read_bits(1)? == 1;
    let entropy_coding_sync_enabled_flag = br.read_bits(1)? == 1;

    // ─── Past the original parse boundary (§7.3.2.3 continuation) ───
    // Tile layout — only present when tiles_enabled_flag.
    // Defaults below model the single-tile-spanning-frame case, which
    // is what the Vulkan Std PPS needs when tiles are disabled.
    let mut num_tile_columns_minus1 = 0u8;
    let mut num_tile_rows_minus1 = 0u8;
    let mut uniform_spacing_flag = true;
    let mut loop_filter_across_tiles_enabled_flag = true;
    if tiles_enabled_flag {
        num_tile_columns_minus1 = br.read_ue().unwrap_or(0) as u8;
        num_tile_rows_minus1 = br.read_ue().unwrap_or(0) as u8;
        uniform_spacing_flag = br.read_bits(1).unwrap_or(1) == 1;
        if !uniform_spacing_flag {
            // column_width_minus1[0..num_tile_columns_minus1] + row_height_minus1[]
            // — we skip but must advance the bit cursor exactly.
            for _ in 0..num_tile_columns_minus1 {
                let _ = br.read_ue();
            }
            for _ in 0..num_tile_rows_minus1 {
                let _ = br.read_ue();
            }
        }
        loop_filter_across_tiles_enabled_flag = br.read_bits(1).unwrap_or(1) == 1;
    }
    let pps_loop_filter_across_slices_enabled_flag = br.read_bits(1)? == 1;

    // Deblocking control
    let deblocking_filter_control_present_flag = br.read_bits(1)? == 1;
    let mut deblocking_filter_override_enabled_flag = false;
    let mut pps_deblocking_filter_disabled_flag = false;
    let mut pps_beta_offset_div2 = 0i8;
    let mut pps_tc_offset_div2 = 0i8;
    if deblocking_filter_control_present_flag {
        deblocking_filter_override_enabled_flag = br.read_bits(1)? == 1;
        pps_deblocking_filter_disabled_flag = br.read_bits(1)? == 1;
        if !pps_deblocking_filter_disabled_flag {
            pps_beta_offset_div2 = clamp_to_i8(br.read_se()?);
            pps_tc_offset_div2 = clamp_to_i8(br.read_se()?);
        }
    }

    // Scaling list
    let pps_scaling_list_data_present_flag = br.read_bits(1)? == 1;
    // If present, scaling_list_data() is a sub-syntax we skip —
    // the Vulkan Std PPS exposes scaling lists via pScalingLists
    // which we leave null for now (FFmpeg populates; we don't
    // and accept the silent driver fallback risk until a scaling-
    // list builder is wired).

    let lists_modification_present_flag = br.read_bits(1)? == 1;
    let log2_parallel_merge_level_minus2 = br.read_ue().unwrap_or(0) as u8;
    let slice_segment_header_extension_present_flag = br.read_bits(1)? == 1;
    let pps_extension_present_flag = br.read_bits(1).unwrap_or(0) == 1;

    Some(H265PpsInfo {
        pps_pic_parameter_set_id,
        pps_seq_parameter_set_id,
        dependent_slice_segments_enabled_flag,
        output_flag_present_flag,
        num_extra_slice_header_bits,
        sign_data_hiding_enabled_flag,
        cabac_init_present_flag,
        num_ref_idx_l0_default_active_minus1,
        num_ref_idx_l1_default_active_minus1,
        init_qp_minus26,
        constrained_intra_pred_flag,
        transform_skip_enabled_flag,
        cu_qp_delta_enabled_flag,
        diff_cu_qp_delta_depth,
        pps_cb_qp_offset,
        pps_cr_qp_offset,
        pps_slice_chroma_qp_offsets_present_flag,
        weighted_pred_flag,
        weighted_bipred_flag,
        transquant_bypass_enabled_flag,
        tiles_enabled_flag,
        entropy_coding_sync_enabled_flag,
        num_tile_columns_minus1,
        num_tile_rows_minus1,
        uniform_spacing_flag,
        loop_filter_across_tiles_enabled_flag,
        pps_loop_filter_across_slices_enabled_flag,
        deblocking_filter_control_present_flag,
        deblocking_filter_override_enabled_flag,
        pps_deblocking_filter_disabled_flag,
        pps_beta_offset_div2,
        pps_tc_offset_div2,
        pps_scaling_list_data_present_flag,
        lists_modification_present_flag,
        log2_parallel_merge_level_minus2,
        slice_segment_header_extension_present_flag,
        pps_extension_present_flag,
    })
}

// ─── Slice header parser ───────────────────────────────────────────

/// Parse the HEVC slice header — subset needed for StdVideoDecodeH265PictureInfo.
/// `sps` / `pps` provide context for bit-width of POC lsb and branch
/// predicates.
pub fn parse_h265_slice_header(
    sample: &[u8],
    sps: &HevcSpsInfo,
    pps: &H265PpsInfo,
) -> Option<H265SliceHeader> {
    let (nal_unit_type, rbsp) = find_hevc_slice_nal(sample)?;
    let mut br = BitReader::new(&rbsp);
    let first_slice_segment_in_pic_flag = br.read_bits(1)? == 1;
    let is_irap = (16..=23).contains(&nal_unit_type);
    let is_idr = matches!(nal_unit_type, 19 | 20);
    if is_irap {
        let _no_output_of_prior_pics_flag = br.read_bits(1)?;
    }
    let slice_pic_parameter_set_id = br.read_ue()? as u8;
    let dependent_slice_segment_flag =
        if !first_slice_segment_in_pic_flag && pps.dependent_slice_segments_enabled_flag {
            br.read_bits(1)? == 1
        } else {
            false
        };
    if !first_slice_segment_in_pic_flag {
        // slice_segment_address — ceil(log2(PicSizeInCtbsY)) bits.
        // For our purposes this is a skip; we don't need the value.
        // Conservative upper bound: 32 bits. In practice streams don't
        // have streams this large. If this bit width is wrong, the
        // rest of our parse will be misaligned — which is why we
        // bail early on non-first-slice headers for now.
        return None;
    }
    let _ = dependent_slice_segment_flag;
    // num_extra_slice_header_bits
    for _ in 0..pps.num_extra_slice_header_bits {
        let _ = br.read_bits(1)?;
    }
    let slice_type_code = br.read_ue()?;
    let slice_type = H265SliceType::from_ue(slice_type_code)?;
    if pps.output_flag_present_flag {
        let _pic_output_flag = br.read_bits(1)?;
    }
    if sps.separate_colour_plane_flag {
        let _colour_plane_id = br.read_bits(2)?;
    }

    let (pic_order_cnt_lsb, short_term_ref_pic_set_sps_flag, short_term_ref_pic_set_idx) =
        if !is_idr {
            let lsb_bits = (sps.log2_max_pic_order_cnt_lsb_minus4 as usize) + 4;
            let lsb = br.read_bits(lsb_bits)?;
            let sps_flag = br.read_bits(1)? == 1;
            let idx = if sps_flag {
                if sps.num_short_term_ref_pic_sets > 1 {
                    let bits =
                        ((sps.num_short_term_ref_pic_sets as f64).log2().ceil() as usize).max(1);
                    Some(br.read_bits(bits)? as u8)
                } else {
                    Some(0)
                }
            } else {
                None
            };
            (lsb, sps_flag, idx)
        } else {
            (0, false, None)
        };

    Some(H265SliceHeader {
        first_slice_segment_in_pic_flag,
        nal_unit_type,
        slice_pic_parameter_set_id,
        slice_type,
        pic_order_cnt_lsb,
        short_term_ref_pic_set_sps_flag,
        short_term_ref_pic_set_idx,
        is_irap,
        is_idr,
    })
}

// ─── NAL scanners ─────────────────────────────────────────────────

/// Find the first HEVC NAL with `nal_unit_type == target` in `data`.
fn find_hevc_nal_by_type(data: &[u8], target: u8) -> Option<&[u8]> {
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
        if nal_byte + 1 >= data.len() {
            return None;
        }
        let nal_unit_type = (data[nal_byte] >> 1) & 0x3F;
        if nal_unit_type == target {
            let start = nal_byte + 2; // 2-byte NAL header
            let end = find_next_start_code(&data[start..])
                .map(|off| start + off)
                .unwrap_or(data.len());
            return Some(&data[start..end]);
        }
        i += start_len;
    }
    None
}

/// Scan an Annex-B HEVC sample and return the offset, in bytes from
/// the start of `data`, where the first coded-slice NAL begins (the
/// byte AFTER the start code). Vulkan `slice_segment_offsets` wants
/// offsets to NAL-unit first bytes, not to start codes.
pub fn hevc_first_slice_nal_offset(data: &[u8]) -> Option<u32> {
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
        if nal_byte + 1 >= data.len() {
            return None;
        }
        let t = (data[nal_byte] >> 1) & 0x3F;
        if (0..=9).contains(&t) || (16..=23).contains(&t) {
            return Some(nal_byte as u32);
        }
        i += start_len;
    }
    None
}

/// Find the first HEVC coded-slice NAL: types 0..=9 (regular slices)
/// or 16..=23 (IRAP slices). Returns (nal_unit_type, RBSP bytes).
fn find_hevc_slice_nal(data: &[u8]) -> Option<(u8, Vec<u8>)> {
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
        if nal_byte + 1 >= data.len() {
            return None;
        }
        let t = (data[nal_byte] >> 1) & 0x3F;
        if (0..=9).contains(&t) || (16..=23).contains(&t) {
            let start = nal_byte + 2;
            let end = find_next_start_code(&data[start..])
                .map(|off| start + off)
                .unwrap_or(data.len());
            return Some((t, remove_h264_rbsp_stuffing(&data[start..end])));
        }
        i += start_len;
    }
    None
}
