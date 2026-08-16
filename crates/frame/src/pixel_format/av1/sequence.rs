//! AV1 sequence header parser and `Av1SequenceHeader` type.
//! See AV1 specification §5.5.2.

use super::super::bitreader::BitReader;
use super::obu::find_av1_obu;

// ─── Av1SequenceHeader ─────────────────────────────────────────────

/// Parsed AV1 sequence header fields (from OBU type 1, §5.5.2).
/// Minimum subset needed to build `StdVideoAV1SequenceHeader` for
/// Vulkan AV1 decode session parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Av1SequenceHeader {
    pub seq_profile: u8,
    pub still_picture: bool,
    pub reduced_still_picture_header: bool,
    pub max_frame_width_minus1: u32,
    pub max_frame_height_minus1: u32,
    pub seq_level_idx_0: u8,
    /// `seq_tier[0]` from AV1 §5.5.1. Only carried in the bitstream
    /// when `seq_level_idx_0 > 7` (i.e. level >= 4.0); below that the
    /// spec says tier is implicitly 0 (Main). 0 = Main, 1 = High.
    /// Required for the AV1 ISOBMFF codec string `av01.P.LLT.DD...`
    /// (the `T` character).
    pub seq_tier_0: u8,
    pub bit_depth: u8,
    pub monochrome: bool,
    pub color_primaries: u8,
    pub transfer_characteristics: u8,
    pub matrix_coefficients: u8,
    pub color_range: bool,
    pub chroma_subsampling_x: bool,
    pub chroma_subsampling_y: bool,
    pub film_grain_params_present: bool,
    pub enable_filter_intra: bool,
    pub enable_intra_edge_filter: bool,
    pub enable_interintra_compound: bool,
    pub enable_masked_compound: bool,
    pub enable_warped_motion: bool,
    pub enable_dual_filter: bool,
    pub enable_order_hint: bool,
    pub enable_jnt_comp: bool,
    pub enable_ref_frame_mvs: bool,
    pub enable_superres: bool,
    pub enable_cdef: bool,
    pub enable_restoration: bool,
    pub order_hint_bits: u8,
    /// Per AV1 §5.5.1: 0 = all frames block screen-content tools,
    /// 1 = all frames enable them, 2 = SELECT (each frame signals
    /// its own bit in the uncompressed_header). Our frame-header
    /// parser reads a per-frame bit only when this field == 2.
    pub seq_force_screen_content_tools: u8,
    /// 0 = all frames force non-integer MV, 1 = all force integer,
    /// 2 = SELECT. Only relevant when screen-content tools allow.
    pub seq_force_integer_mv: u8,
    /// Bit-width of max_frame_width_minus_1 / max_frame_height_minus_1
    /// fields in the sequence header. Vulkan's Std SPS requires these
    /// to match so the session parameters object is byte-compatible
    /// with what the driver re-parses from the bitstream.
    pub frame_width_bits_minus_1: u8,
    pub frame_height_bits_minus_1: u8,
    pub use_128x128_superblock: bool,
    /// AV1 §5.5.2 color_config bit — signals that U and V planes
    /// carry separate q-delta values. Feeds
    /// `StdVideoAV1ColorConfigFlags.separate_uv_delta_q` which the
    /// Vulkan AV1 decoder reads at session-parameters creation.
    pub separate_uv_delta_q: bool,
}

// ─── AV1 uvlc helper ───────────────────────────────────────────────

/// AV1 uvlc (unsigned variable-length code) — count leading zero bits
/// up to 31; then read that many bits as the suffix; value = (1<<N)-1+suffix.
fn read_av1_uvlc(br: &mut BitReader) -> Option<u32> {
    let mut leading_zeros = 0;
    while leading_zeros < 32 {
        if br.read_bits(1)? == 1 {
            break;
        }
        leading_zeros += 1;
    }
    if leading_zeros >= 32 {
        return None;
    }
    if leading_zeros == 0 {
        return Some(0);
    }
    let suffix = br.read_bits(leading_zeros)?;
    Some((1u32 << leading_zeros) - 1 + suffix)
}

// ─── Full sequence header parser ───────────────────────────────────

/// Parse the AV1 sequence header OBU (obu_type=1). Returns the
/// subset of §5.5.2 fields needed for Vulkan decode-session-params.
/// Partial parse: we stop after color_config + film_grain_params_present
/// (everything Vulkan's StdVideoAV1SequenceHeader cares about).
pub fn parse_av1_sequence_header(sample: &[u8]) -> Option<Av1SequenceHeader> {
    let obu = find_av1_obu(sample, 1)?;
    let mut br = BitReader::new(obu);
    let seq_profile = br.read_bits(3)? as u8;
    let still_picture = br.read_bits(1)? == 1;
    let reduced_still_picture_header = br.read_bits(1)? == 1;

    let mut seq_level_idx_0 = 0u8;
    let mut seq_tier_0 = 0u8;
    let (_operating_points_cnt_minus_1, _timing_info_present_flag);
    let mut order_hint_bits = 0u8;
    let mut enable_order_hint = false;

    if reduced_still_picture_header {
        seq_level_idx_0 = br.read_bits(5)? as u8;
        _operating_points_cnt_minus_1 = 0;
        _timing_info_present_flag = false;
    } else {
        let timing_info_present_flag = br.read_bits(1)? == 1;
        _timing_info_present_flag = timing_info_present_flag;
        let mut decoder_model_info_present_flag = false;
        let mut buffer_delay_length_minus_1 = 0u32;
        if timing_info_present_flag {
            let _num_units_in_display_tick = br.read_bits(32)?;
            let _time_scale = br.read_bits(32)?;
            let equal_picture_interval = br.read_bits(1)? == 1;
            if equal_picture_interval {
                let _num_ticks_per_picture_minus_1 = read_av1_uvlc(&mut br)?;
            }
            decoder_model_info_present_flag = br.read_bits(1)? == 1;
            if decoder_model_info_present_flag {
                buffer_delay_length_minus_1 = br.read_bits(5)?;
                let _num_units_in_decoding_tick = br.read_bits(32)?;
                let _buffer_removal_time_length_minus_1 = br.read_bits(5)?;
                let _frame_presentation_time_length_minus_1 = br.read_bits(5)?;
            }
        }
        // initial_display_delay_present_flag lives OUTSIDE the
        // timing-info-present branch per AV1 §5.5.1 — my earlier
        // parse had it nested, which desynced every field that
        // followed on streams with timing_info absent.
        let initial_display_delay_present_flag = br.read_bits(1)? == 1;
        let operating_points_cnt_minus_1 = br.read_bits(5)? as u8;
        _operating_points_cnt_minus_1 = operating_points_cnt_minus_1;
        for i in 0..=operating_points_cnt_minus_1 {
            let _operating_point_idc = br.read_bits(12)?;
            let seq_level_idx_i = br.read_bits(5)? as u8;
            // Per AV1 §5.5.1, seq_tier is present only for levels
            // >= 4.0 (level_idx > 7); below that it's implicitly 0.
            let seq_tier_i = if seq_level_idx_i > 7 {
                br.read_bits(1)? as u8
            } else {
                0
            };
            if i == 0 {
                seq_level_idx_0 = seq_level_idx_i;
                seq_tier_0 = seq_tier_i;
            }
            // operating_parameters_info(i) — one per-op-point
            // decoder_model_present_for_this_op gate.
            if decoder_model_info_present_flag {
                let decoder_model_present_for_this_op = br.read_bits(1)? == 1;
                if decoder_model_present_for_this_op {
                    let n = (buffer_delay_length_minus_1 + 1) as usize;
                    let _buffer_delay = br.read_bits(n)?;
                    let _encoder_buffer_delay = br.read_bits(n)?;
                    let _low_delay_mode_flag = br.read_bits(1)?;
                }
            }
            if initial_display_delay_present_flag {
                let idd_present_for_this_op = br.read_bits(1)? == 1;
                if idd_present_for_this_op {
                    let _initial_display_delay_minus_1 = br.read_bits(4)?;
                }
            }
        }
    }
    let frame_width_bits_minus_1 = br.read_bits(4)? as usize;
    let frame_height_bits_minus_1 = br.read_bits(4)? as usize;
    let max_frame_width_minus1 = br.read_bits(frame_width_bits_minus_1 + 1)?;
    let max_frame_height_minus1 = br.read_bits(frame_height_bits_minus_1 + 1)?;

    let frame_id_numbers_present_flag = if reduced_still_picture_header {
        false
    } else {
        br.read_bits(1)? == 1
    };
    if frame_id_numbers_present_flag {
        let _delta_frame_id_length_minus_2 = br.read_bits(4)?;
        let _additional_frame_id_length_minus_1 = br.read_bits(3)?;
    }
    let use_128x128_superblock = br.read_bits(1)? == 1;
    let enable_filter_intra = br.read_bits(1)? == 1;
    let enable_intra_edge_filter = br.read_bits(1)? == 1;
    let mut enable_interintra_compound = false;
    let mut enable_masked_compound = false;
    let mut enable_warped_motion = false;
    let mut enable_dual_filter = false;
    let mut enable_jnt_comp = false;
    let mut enable_ref_frame_mvs = false;
    let mut seq_force_screen_content_tools: u8 = 2; // SELECT when reduced_still_picture_header
    let mut seq_force_integer_mv: u8 = 2;
    if !reduced_still_picture_header {
        enable_interintra_compound = br.read_bits(1)? == 1;
        enable_masked_compound = br.read_bits(1)? == 1;
        enable_warped_motion = br.read_bits(1)? == 1;
        enable_dual_filter = br.read_bits(1)? == 1;
        enable_order_hint = br.read_bits(1)? == 1;
        if enable_order_hint {
            enable_jnt_comp = br.read_bits(1)? == 1;
            enable_ref_frame_mvs = br.read_bits(1)? == 1;
        }
        let seq_choose_screen_content_tools = br.read_bits(1)? == 1;
        seq_force_screen_content_tools = if seq_choose_screen_content_tools {
            2u8
        } else {
            br.read_bits(1)? as u8
        };
        if seq_force_screen_content_tools > 0 {
            let seq_choose_integer_mv = br.read_bits(1)? == 1;
            seq_force_integer_mv = if seq_choose_integer_mv {
                2u8
            } else {
                br.read_bits(1)? as u8
            };
        }
        if enable_order_hint {
            order_hint_bits = br.read_bits(3)? as u8 + 1;
        }
    }
    let enable_superres = br.read_bits(1)? == 1;
    let enable_cdef = br.read_bits(1)? == 1;
    let enable_restoration = br.read_bits(1)? == 1;

    // color_config(seq_profile)
    let high_bitdepth = br.read_bits(1)? == 1;
    let bit_depth = if seq_profile == 2 && high_bitdepth {
        if br.read_bits(1)? == 1 { 12 } else { 10 }
    } else if high_bitdepth {
        10
    } else {
        8
    };
    let monochrome = if seq_profile == 1 {
        false
    } else {
        br.read_bits(1)? == 1
    };
    let color_description_present_flag = br.read_bits(1)? == 1;
    let (color_primaries, transfer_characteristics, matrix_coefficients) =
        if color_description_present_flag {
            (
                br.read_bits(8)? as u8,
                br.read_bits(8)? as u8,
                br.read_bits(8)? as u8,
            )
        } else {
            (2u8, 2u8, 2u8) // unspecified
        };
    let color_range;
    let (subx, suby);
    let mut separate_uv_delta_q = false;
    if monochrome {
        color_range = br.read_bits(1)? == 1;
        subx = true;
        suby = true;
    } else if color_primaries == 1 && transfer_characteristics == 13 && matrix_coefficients == 0 {
        color_range = true;
        subx = false;
        suby = false;
    } else {
        color_range = br.read_bits(1)? == 1;
        match seq_profile {
            0 => {
                subx = true;
                suby = true;
            }
            1 => {
                subx = false;
                suby = false;
            }
            2 => {
                if bit_depth == 12 {
                    subx = br.read_bits(1)? == 1;
                    suby = if subx { br.read_bits(1)? == 1 } else { false };
                } else {
                    subx = true;
                    suby = false;
                }
            }
            _ => {
                subx = true;
                suby = true;
            }
        }
        if subx && suby {
            let _chroma_sample_position = br.read_bits(2)?;
        }
        separate_uv_delta_q = br.read_bits(1)? == 1;
    }
    let film_grain_params_present = br.read_bits(1)? == 1;

    Some(Av1SequenceHeader {
        seq_profile,
        still_picture,
        reduced_still_picture_header,
        max_frame_width_minus1,
        max_frame_height_minus1,
        seq_level_idx_0,
        seq_tier_0,
        bit_depth,
        monochrome,
        color_primaries,
        transfer_characteristics,
        matrix_coefficients,
        color_range,
        chroma_subsampling_x: subx,
        chroma_subsampling_y: suby,
        film_grain_params_present,
        enable_filter_intra,
        enable_intra_edge_filter,
        enable_interintra_compound,
        enable_masked_compound,
        enable_warped_motion,
        enable_dual_filter,
        enable_order_hint,
        enable_jnt_comp,
        enable_ref_frame_mvs,
        enable_superres,
        enable_cdef,
        enable_restoration,
        order_hint_bits,
        seq_force_screen_content_tools,
        seq_force_integer_mv,
        frame_width_bits_minus_1: frame_width_bits_minus_1 as u8,
        frame_height_bits_minus_1: frame_height_bits_minus_1 as u8,
        use_128x128_superblock,
        separate_uv_delta_q,
    })
}
