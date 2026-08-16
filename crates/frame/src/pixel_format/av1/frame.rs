//! AV1 frame-header parser, `Av1FrameHeader`, and `Av1FrameType`.
//! See AV1 specification §5.9.1 (uncompressed_header).

use super::super::bitreader::BitReader;
use super::obu::find_av1_obu;
use super::sequence::Av1SequenceHeader;

// ─── Av1FrameHeader ────────────────────────────────────────────────

/// Parsed AV1 frame header — full §5.9.1 uncompressed_header parse.
/// Provides everything needed to populate `StdVideoDecodeAV1PictureInfo`
/// + its 7 sub-struct pointers for a Vulkan Video AV1 decode submit.
/// Vec fields (tile MI-unit arrays) forced the drop of `Copy`.
#[derive(Debug, Clone)]
pub struct Av1FrameHeader {
    pub show_frame: bool,
    pub showable_frame: bool,
    pub frame_type: Av1FrameType,
    pub error_resilient_mode: bool,
    pub disable_cdf_update: bool,
    pub allow_screen_content_tools: bool,
    pub force_integer_mv: bool,
    pub order_hint: u32,
    pub primary_ref_frame: u8,
    pub refresh_frame_flags: u8,
    pub frame_width: u32,
    pub frame_height: u32,
    pub render_width: u32,
    pub render_height: u32,
    pub use_ref_frame_mvs: bool,
    pub allow_high_precision_mv: bool,
    pub is_filter_switchable: bool,
    pub disable_frame_end_update_cdf: bool,
    pub allow_warped_motion: bool,
    pub reduced_tx_set: bool,
    // ─── Extended fields (full §5.9.1 parse) ────────────────────
    pub allow_intrabc: bool,
    pub frame_size_override_flag: bool,
    pub use_superres: bool,
    pub is_motion_mode_switchable: bool,
    pub reference_select: bool,
    pub skip_mode_present: bool,
    // Tile info (§5.9.15) — derived MI-unit arrays feed
    // `StdVideoAV1TileInfo.pMi{Col,Row}Starts` etc.
    pub tile_cols: u8,
    pub tile_rows: u8,
    pub uniform_tile_spacing_flag: bool,
    pub tile_cols_log2: u8,
    pub tile_rows_log2: u8,
    pub mi_col_starts: Vec<u16>,         // len = tile_cols + 1
    pub mi_row_starts: Vec<u16>,         // len = tile_rows + 1
    pub width_in_sbs_minus_1: Vec<u16>,  // len = tile_cols
    pub height_in_sbs_minus_1: Vec<u16>, // len = tile_rows
    pub context_update_tile_id: u16,
    pub tile_size_bytes_minus_1: u8,
    // Quantization (§5.9.12)
    pub base_q_idx: u8,
    pub delta_q_y_dc: i8,
    pub delta_q_u_dc: i8,
    pub delta_q_u_ac: i8,
    pub delta_q_v_dc: i8,
    pub delta_q_v_ac: i8,
    pub using_qmatrix: bool,
    pub qm_y: u8,
    pub qm_u: u8,
    pub qm_v: u8,
    // Delta-Q / delta-LF (§5.9.17 / §5.9.18)
    pub delta_q_present: bool,
    pub delta_q_res: u8,
    pub delta_lf_present: bool,
    pub delta_lf_res: u8,
    pub delta_lf_multi: bool,
    // Segmentation (§5.9.14) — scaffolded as "disabled" for the
    // Vulkan scope; real feature arrays populated when
    // segmentation_enabled is 1.
    pub segmentation_enabled: bool,
    pub segmentation_update_map: bool,
    pub segmentation_temporal_update: bool,
    pub segmentation_update_data: bool,
    pub feature_enabled: [[bool; 8]; 8],
    pub feature_data: [[i16; 8]; 8],
    // Loop filter (§5.9.11)
    pub loop_filter_level: [u8; 4],
    pub loop_filter_sharpness: u8,
    pub loop_filter_delta_enabled: bool,
    pub loop_filter_delta_update: bool,
    pub update_ref_delta_mask: u8, // 8 bits
    pub loop_filter_ref_deltas: [i8; 8],
    pub update_mode_delta_mask: u8, // 2 bits (modes 0..=1)
    pub loop_filter_mode_deltas: [i8; 2],
    // CDEF (§5.9.19)
    pub cdef_damping_minus_3: u8,
    pub cdef_bits: u8,
    pub cdef_y_pri_strength: [u8; 8],
    pub cdef_y_sec_strength: [u8; 8],
    pub cdef_uv_pri_strength: [u8; 8],
    pub cdef_uv_sec_strength: [u8; 8],
    // Loop restoration (§5.9.20)
    pub lr_type: [u8; 3], // per-plane: 0=None, 1=Wiener, 2=SGrproj, 3=Switchable
    pub lr_unit_shift: u8,
    pub lr_uv_shift: u8,
    // TX mode (§5.9.22) — 0=ONLY_4X4, 1=LARGEST, 2=SELECT
    pub tx_mode: u8,
    pub interpolation_filter: u8,
    // Byte offset from the start of the OBU payload (NOT from the
    // start of the sample buffer) at which tile_group data begins.
    // For a Frame OBU (type 6) this is after uncompressed_header +
    // byte_alignment. For a pair of separate frame_header + tile_group
    // OBUs (types 3 and 4), the caller looks up the type 4 OBU's
    // payload start directly and ignores this value.
    pub tile_group_offset_in_obu: u32,
    // Coded lossless flag (derived from q-idx 0 + deltas all zero)
    pub coded_lossless: bool,
}

impl Default for Av1FrameHeader {
    fn default() -> Self {
        Self {
            show_frame: false,
            showable_frame: false,
            frame_type: Av1FrameType::Key,
            error_resilient_mode: false,
            disable_cdf_update: false,
            allow_screen_content_tools: false,
            force_integer_mv: false,
            order_hint: 0,
            primary_ref_frame: 7,
            refresh_frame_flags: 0,
            frame_width: 0,
            frame_height: 0,
            render_width: 0,
            render_height: 0,
            use_ref_frame_mvs: false,
            allow_high_precision_mv: false,
            is_filter_switchable: false,
            disable_frame_end_update_cdf: false,
            allow_warped_motion: false,
            reduced_tx_set: false,
            allow_intrabc: false,
            frame_size_override_flag: false,
            use_superres: false,
            is_motion_mode_switchable: false,
            reference_select: false,
            skip_mode_present: false,
            tile_cols: 1,
            tile_rows: 1,
            uniform_tile_spacing_flag: true,
            tile_cols_log2: 0,
            tile_rows_log2: 0,
            mi_col_starts: Vec::new(),
            mi_row_starts: Vec::new(),
            width_in_sbs_minus_1: Vec::new(),
            height_in_sbs_minus_1: Vec::new(),
            context_update_tile_id: 0,
            tile_size_bytes_minus_1: 3,
            base_q_idx: 0,
            delta_q_y_dc: 0,
            delta_q_u_dc: 0,
            delta_q_u_ac: 0,
            delta_q_v_dc: 0,
            delta_q_v_ac: 0,
            using_qmatrix: false,
            qm_y: 0,
            qm_u: 0,
            qm_v: 0,
            delta_q_present: false,
            delta_q_res: 0,
            delta_lf_present: false,
            delta_lf_res: 0,
            delta_lf_multi: false,
            segmentation_enabled: false,
            segmentation_update_map: false,
            segmentation_temporal_update: false,
            segmentation_update_data: false,
            feature_enabled: [[false; 8]; 8],
            feature_data: [[0; 8]; 8],
            loop_filter_level: [0; 4],
            loop_filter_sharpness: 0,
            loop_filter_delta_enabled: false,
            loop_filter_delta_update: false,
            update_ref_delta_mask: 0,
            loop_filter_ref_deltas: [0; 8],
            update_mode_delta_mask: 0,
            loop_filter_mode_deltas: [0; 2],
            cdef_damping_minus_3: 0,
            cdef_bits: 0,
            cdef_y_pri_strength: [0; 8],
            cdef_y_sec_strength: [0; 8],
            cdef_uv_pri_strength: [0; 8],
            cdef_uv_sec_strength: [0; 8],
            lr_type: [0; 3],
            lr_unit_shift: 0,
            lr_uv_shift: 0,
            tx_mode: 0,
            interpolation_filter: 0,
            tile_group_offset_in_obu: 0,
            coded_lossless: false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Av1FrameType {
    Key,
    Inter,
    IntraOnly,
    Switch,
}

// ─── Full frame header parser ───────────────────────────────────────

/// Parse an AV1 frame_header_obu (or the frame_header part of a
/// frame_obu) from the given sample. Requires the sequence header
/// for branch predicates (order_hint_bits, enable flags).
///
/// Returns an `Av1FrameHeader` with just enough fields populated for
/// Vulkan Video decode to build `StdVideoDecodeAV1PictureInfo` +
/// sub-structs. Does NOT fully parse the bitstream (skips large
/// parts of the uncompressed_header — tile_info, segmentation,
/// global motion, etc. — that can be defaulted for key frames).
///
/// Per AV1 spec §5.9.1 — complex, branching parse. Only handles
/// single-tile key frames at first; inter frames need more work
/// on ref_frame_idx + delta_frame_id resolution.
pub fn parse_av1_frame_header(sample: &[u8], seq: &Av1SequenceHeader) -> Option<Av1FrameHeader> {
    let obu_bytes = find_av1_obu(sample, 3).or_else(|| find_av1_obu(sample, 6))?;
    let mut br = BitReader::new(obu_bytes);
    let mut h = Av1FrameHeader::default();

    // ─── Phase 1: frame-level flags ────────────────────────────
    if seq.reduced_still_picture_header {
        h.frame_type = Av1FrameType::Key;
        h.show_frame = true;
        h.showable_frame = false;
        h.error_resilient_mode = true;
    } else {
        let show_existing_frame = br.read_bits(1)? == 1;
        if show_existing_frame {
            // Early-out: a show-existing-frame OBU is a thin pointer
            // to a previously-decoded DPB slot. No new bitstream to
            // decode, no uncompressed_header payload past this point.
            // Return a minimal header marked with show_frame=true so
            // callers know to skip bitstream decode.
            let _frame_to_show_map_idx = br.read_bits(3)?;
            h.show_frame = true;
            h.showable_frame = true;
            h.frame_type = Av1FrameType::Key;
            h.frame_width = seq.max_frame_width_minus1 + 1;
            h.frame_height = seq.max_frame_height_minus1 + 1;
            h.render_width = h.frame_width;
            h.render_height = h.frame_height;
            return Some(h);
        }
        let ft_code = br.read_bits(2)?;
        h.frame_type = match ft_code {
            0 => Av1FrameType::Key,
            1 => Av1FrameType::Inter,
            2 => Av1FrameType::IntraOnly,
            3 => Av1FrameType::Switch,
            _ => return None,
        };
        h.show_frame = br.read_bits(1)? == 1;
        h.showable_frame = if h.show_frame {
            !matches!(h.frame_type, Av1FrameType::Key)
        } else {
            br.read_bits(1)? == 1
        };
        let is_key = matches!(h.frame_type, Av1FrameType::Key);
        let is_switch = matches!(h.frame_type, Av1FrameType::Switch);
        h.error_resilient_mode = if is_switch || (is_key && h.show_frame) {
            true
        } else {
            br.read_bits(1)? == 1
        };
    }

    let frame_is_intra = matches!(h.frame_type, Av1FrameType::Key | Av1FrameType::IntraOnly);

    h.disable_cdf_update = br.read_bits(1)? == 1;
    // Per AV1 §5.9.1 — when seq_force_screen_content_tools == SELECT (2),
    // each frame signals its own bit; otherwise the seq-level force
    // fully determines the frame-level value.
    h.allow_screen_content_tools = if seq.seq_force_screen_content_tools == 2 {
        br.read_bits(1)? == 1
    } else {
        seq.seq_force_screen_content_tools == 1
    };
    if h.allow_screen_content_tools {
        h.force_integer_mv = if seq.seq_force_integer_mv == 2 {
            br.read_bits(1)? == 1
        } else {
            seq.seq_force_integer_mv == 1
        };
    } else {
        h.force_integer_mv = false;
    }
    if frame_is_intra {
        h.force_integer_mv = true;
    }

    // frame_size_override_flag
    let is_switch = matches!(h.frame_type, Av1FrameType::Switch);
    h.frame_size_override_flag = if is_switch {
        true
    } else if seq.reduced_still_picture_header {
        false
    } else {
        br.read_bits(1)? == 1
    };

    // order_hint
    if seq.enable_order_hint && seq.order_hint_bits > 0 {
        h.order_hint = br.read_bits(seq.order_hint_bits as usize)?;
    }

    // primary_ref_frame (only for non-intra, non-error-resilient)
    h.primary_ref_frame = if frame_is_intra || h.error_resilient_mode {
        7 // PRIMARY_REF_NONE
    } else {
        br.read_bits(3)? as u8
    };

    // refresh_frame_flags
    let all_frames = 0xFFu8;
    h.refresh_frame_flags = if matches!(h.frame_type, Av1FrameType::Key) && h.show_frame {
        all_frames
    } else if is_switch {
        all_frames
    } else {
        br.read_bits(8)? as u8
    };

    // ─── Phase 2: size / render size / ref frames ──────────────
    let (frame_width, frame_height) = if frame_is_intra {
        let (w, h2) = parse_av1_frame_size(&mut br, seq, h.frame_size_override_flag)?;
        // superres_params() is INSIDE frame_size() per §5.9.5 /
        // §5.9.6 — before render_size().
        h.use_superres = if seq.enable_superres {
            br.read_bits(1)? == 1
        } else {
            false
        };
        if h.use_superres {
            let _superres_denom_minus9 = br.read_bits(3)?;
        }
        parse_av1_render_size(&mut br, w, h2, &mut h.render_width, &mut h.render_height)?;
        if h.allow_screen_content_tools
        /* && UpscaledWidth == FrameWidth */
        {
            h.allow_intrabc = br.read_bits(1)? == 1;
        }
        (w, h2)
    } else {
        // Inter-frame path: ref_frame_idx[], frame_size_with_refs,
        // interpolation_filter, is_motion_mode_switchable,
        // use_ref_frame_mvs. For our key-frame-focused scope, this
        // branch ISN'T the critical path — but we still read bits
        // to keep the parser position in sync.
        let frame_refs_short_signaling = if seq.enable_order_hint {
            br.read_bits(1)? == 1
        } else {
            false
        };
        if frame_refs_short_signaling {
            let _last_frame_idx = br.read_bits(3)?;
            let _gold_frame_idx = br.read_bits(3)?;
        }
        for _ in 0..7u8
        /* REFS_PER_FRAME */
        {
            if !frame_refs_short_signaling {
                let _ref_frame_idx = br.read_bits(3)?;
            }
            // frame_id_numbers_present_flag is false in our minimal
            // seq, so no delta_frame_id read.
        }
        let (w, h2) = if h.frame_size_override_flag && !h.error_resilient_mode {
            parse_av1_frame_size_with_refs(&mut br, seq)?
        } else {
            let (w, h2) = parse_av1_frame_size(&mut br, seq, h.frame_size_override_flag)?;
            // superres_params() inside frame_size() per spec.
            h.use_superres = if seq.enable_superres {
                br.read_bits(1)? == 1
            } else {
                false
            };
            if h.use_superres {
                let _superres_denom_minus9 = br.read_bits(3)?;
            }
            parse_av1_render_size(&mut br, w, h2, &mut h.render_width, &mut h.render_height)?;
            (w, h2)
        };
        h.allow_high_precision_mv = if h.force_integer_mv {
            false
        } else {
            br.read_bits(1)? == 1
        };
        // read_interpolation_filter (§5.9.10)
        h.is_filter_switchable = br.read_bits(1)? == 1;
        h.interpolation_filter = if h.is_filter_switchable {
            4 // SWITCHABLE
        } else {
            br.read_bits(2)? as u8
        };
        h.is_motion_mode_switchable = br.read_bits(1)? == 1;
        h.use_ref_frame_mvs = if h.error_resilient_mode || !seq.enable_ref_frame_mvs {
            false
        } else {
            br.read_bits(1)? == 1
        };
        (w, h2)
    };
    h.frame_width = frame_width;
    h.frame_height = frame_height;
    if h.render_width == 0 {
        h.render_width = frame_width;
    }
    if h.render_height == 0 {
        h.render_height = frame_height;
    }

    h.disable_frame_end_update_cdf = if seq.reduced_still_picture_header {
        true
    } else {
        br.read_bits(1)? == 1
    };

    // ─── Phase 5: tile_info() (§5.9.15) ────────────────────────
    // MI (mode-info) units = 4 luma samples. SB (superblock) size in
    // MI units = 16 (64x64 SB) or 32 (128x128 SB). Our seq parser
    // doesn't capture `use_128x128_superblock` yet; default to 16 MI
    // per SB — the common case for current AV1 streams.
    let sb_size_log2: u32 = 4; // log2(16)
    let mi_cols_raw = 2 * ((frame_width.saturating_sub(1) + 8) >> 3);
    let mi_rows_raw = 2 * ((frame_height.saturating_sub(1) + 8) >> 3);
    // Align MI dims to SB boundaries for tile-spacing math.
    let sb_cols = (mi_cols_raw + (1 << sb_size_log2) - 1) >> sb_size_log2;
    let sb_rows = (mi_rows_raw + (1 << sb_size_log2) - 1) >> sb_size_log2;
    parse_av1_tile_info(
        &mut br,
        &mut h,
        sb_cols,
        sb_rows,
        sb_size_log2,
        mi_cols_raw,
        mi_rows_raw,
    )?;

    // ─── Phase 6: quantization_params() (§5.9.12) ──────────────
    parse_av1_quantization_params(&mut br, &mut h, seq)?;

    // ─── Phase 7: segmentation_params() (§5.9.14) ──────────────
    parse_av1_segmentation_params(&mut br, &mut h)?;

    // ─── Phase 8: delta_q_params / delta_lf_params ─────────────
    h.delta_q_present = if h.base_q_idx > 0 {
        br.read_bits(1)? == 1
    } else {
        false
    };
    h.delta_q_res = if h.delta_q_present {
        br.read_bits(2)? as u8
    } else {
        0
    };
    h.delta_lf_present = if h.delta_q_present && !h.allow_intrabc {
        br.read_bits(1)? == 1
    } else {
        false
    };
    if h.delta_lf_present {
        h.delta_lf_res = br.read_bits(2)? as u8;
        h.delta_lf_multi = br.read_bits(1)? == 1;
    }

    // ─── Compute CodedLossless (§5.9.1) ─────────────────────────
    // lossless requires base_q_idx=0 and ALL delta-q values == 0.
    // We don't iterate segment features for per-seg q deltas here;
    // coded_lossless only affects the later cdef_params gate and
    // tx_mode coding (both set to 0 when lossless).
    h.coded_lossless = h.base_q_idx == 0
        && h.delta_q_y_dc == 0
        && h.delta_q_u_dc == 0
        && h.delta_q_u_ac == 0
        && h.delta_q_v_dc == 0
        && h.delta_q_v_ac == 0;

    // ─── Phase 9: loop_filter_params() (§5.9.11) ───────────────
    parse_av1_loop_filter_params(&mut br, &mut h, frame_is_intra)?;

    // ─── Phase 10: cdef_params() (§5.9.19) ─────────────────────
    let num_planes_u32: u32 = if seq.monochrome { 1 } else { 3 };
    if !h.coded_lossless && !h.allow_intrabc && seq.enable_cdef {
        parse_av1_cdef_params(&mut br, &mut h, num_planes_u32)?;
    } else {
        // Spec defaults when cdef is skipped (§5.9.19 conformance).
        h.cdef_bits = 0;
        h.cdef_damping_minus_3 = 0;
        h.cdef_y_pri_strength = [0; 8];
        h.cdef_y_sec_strength = [0; 8];
        h.cdef_uv_pri_strength = [0; 8];
        h.cdef_uv_sec_strength = [0; 8];
    }

    // ─── Phase 11: lr_params() (§5.9.20) ───────────────────────
    if !h.coded_lossless && !h.allow_intrabc && seq.enable_restoration {
        parse_av1_lr_params(&mut br, &mut h, num_planes_u32, seq)?;
    }

    // ─── Phase 12: read_tx_mode (§5.9.22) ──────────────────────
    h.tx_mode = if h.coded_lossless {
        0 // ONLY_4X4
    } else if br.read_bits(1)? == 1 {
        2 // TX_MODE_SELECT
    } else {
        1 // TX_MODE_LARGEST
    };

    // ─── Phase 13: frame_reference_mode (§5.9.23) ──────────────
    h.reference_select = if !frame_is_intra {
        br.read_bits(1)? == 1
    } else {
        false
    };

    // ─── Phase 14: skip_mode_params (§5.9.24) ──────────────────
    let skip_mode_allowed = false; // For KEY/INTRA_ONLY, skip_mode is
    // implicitly disabled (requires 2
    // forward/backward refs). Inter
    // would derive from ref_frame_idx
    // + order hints — scaffolded.
    h.skip_mode_present = if skip_mode_allowed {
        br.read_bits(1)? == 1
    } else {
        false
    };

    // allow_warped_motion (§5.9.1) — 1 bit gated by seq.enable_warped_motion
    // AND !error_resilient_mode AND !FrameIsIntra.
    h.allow_warped_motion =
        if !frame_is_intra && !h.error_resilient_mode && seq.enable_warped_motion {
            br.read_bits(1)? == 1
        } else {
            false
        };

    // reduced_tx_set (1 bit) — the last bitstream bit we care about
    // for Vulkan Std picture info. global_motion_params,
    // film_grain_params, and the tile_group_obu() that follows the
    // byte_alignment() at the end are all parsed by the driver from
    // the bitstream, not from our Std struct.
    h.reduced_tx_set = br.read_bits(1)? == 1;

    // ─── Phase 15: global_motion_params (§5.9.21) ──────────────
    // Read-only — we don't carry gm params across into Vulkan's
    // StdVideoAV1GlobalMotion at this time (zero-init GmType[]
    // → IDENTITY for every ref, matching the implicit default).
    if !frame_is_intra {
        skip_av1_global_motion_params(&mut br)?;
    }

    // ─── Phase 16: film_grain_params (§5.9.25) ─────────────────
    if seq.film_grain_params_present && (h.show_frame || h.showable_frame) {
        skip_av1_film_grain_params(&mut br, seq)?;
    }

    // ─── Phase 17: byte_align() + record tile_group_offset ────
    // Per §5.3.5, uncompressed_header ends with byte_alignment. The
    // tile_group_obu starts at the next byte boundary in the same
    // Frame OBU (type 6) or in a separate Tile Group OBU (type 4).
    br.byte_align();
    h.tile_group_offset_in_obu = (br.bit_pos() / 8) as u32;

    Some(h)
}

// ─── AV1 frame-header helper functions ─────────────────────────────

/// §5.9.5 frame_size()
fn parse_av1_frame_size(
    br: &mut BitReader,
    seq: &Av1SequenceHeader,
    frame_size_override_flag: bool,
) -> Option<(u32, u32)> {
    if frame_size_override_flag {
        let w_bits = av1_bits_for_max(seq.max_frame_width_minus1 + 1);
        let h_bits = av1_bits_for_max(seq.max_frame_height_minus1 + 1);
        let w = br.read_bits(w_bits)? + 1;
        let hgt = br.read_bits(h_bits)? + 1;
        Some((w, hgt))
    } else {
        Some((
            seq.max_frame_width_minus1 + 1,
            seq.max_frame_height_minus1 + 1,
        ))
    }
}

/// §5.9.6 render_size()
fn parse_av1_render_size(
    br: &mut BitReader,
    frame_w: u32,
    frame_h: u32,
    out_w: &mut u32,
    out_h: &mut u32,
) -> Option<()> {
    let render_and_frame_size_different = br.read_bits(1)? == 1;
    if render_and_frame_size_different {
        *out_w = br.read_bits(16)? + 1;
        *out_h = br.read_bits(16)? + 1;
    } else {
        *out_w = frame_w;
        *out_h = frame_h;
    }
    Some(())
}

/// §5.9.7 frame_size_with_refs() — for inter frames with size override.
/// Returns (frame_width, frame_height). The per-ref "found_ref" loop
/// here requires access to the ref frames' dims, which our scaffold
/// doesn't track. We treat `found_ref=0` uniformly (falls back to
/// frame_size()).
fn parse_av1_frame_size_with_refs(
    br: &mut BitReader,
    seq: &Av1SequenceHeader,
) -> Option<(u32, u32)> {
    let mut found_ref = false;
    for _ in 0..7u8 {
        if br.read_bits(1)? == 1 {
            found_ref = true;
        }
    }
    if !found_ref {
        let (w, hgt) = parse_av1_frame_size(br, seq, true)?;
        let mut rw = 0;
        let mut rh = 0;
        parse_av1_render_size(br, w, hgt, &mut rw, &mut rh)?;
        // superres_params inlined
        if seq.enable_superres && br.read_bits(1)? == 1 {
            let _denom = br.read_bits(3)?;
        }
        Some((w, hgt))
    } else {
        // found_ref branch: dims come from one of the refs. No ref
        // tracking → fall back to the sequence header max.
        Some((
            seq.max_frame_width_minus1 + 1,
            seq.max_frame_height_minus1 + 1,
        ))
    }
}

fn av1_bits_for_max(v: u32) -> usize {
    // Inclusive ceil-log2 for a max-value field (AV1 uses
    // `n_bits = ceil(log2(max + 1))`).
    let mut bits = 0usize;
    let mut x = v.saturating_sub(1);
    while x > 0 {
        bits += 1;
        x >>= 1;
    }
    bits.max(1)
}

/// §5.9.15 tile_info()
fn parse_av1_tile_info(
    br: &mut BitReader,
    h: &mut Av1FrameHeader,
    sb_cols: u32,
    sb_rows: u32,
    sb_size_log2: u32,
    mi_cols: u32,
    mi_rows: u32,
) -> Option<()> {
    // Derive MAX_TILE_AREA_SB, MAX_TILE_WIDTH_SB etc. (§5.9.15)
    // Constants from AV1 spec for 64x64 SB (log2=4).
    let max_tile_width_sb = 4096 >> (sb_size_log2 + 2); // typically 64
    let max_tile_area_sb = (4096 * 2304) >> (2 * sb_size_log2 + 4); // 4608
    let min_log2_tile_cols = av1_tile_log2(max_tile_width_sb, sb_cols);
    let max_log2_tile_cols = av1_tile_log2(1, sb_cols.min(64));
    let max_log2_tile_rows = av1_tile_log2(1, sb_rows.min(64));
    let min_log2_tiles = min_log2_tile_cols.max(av1_tile_log2(max_tile_area_sb, sb_rows * sb_cols));

    h.uniform_tile_spacing_flag = br.read_bits(1)? == 1;
    let tile_cols_log2: u32;
    let tile_rows_log2: u32;
    h.mi_col_starts.clear();
    h.mi_row_starts.clear();
    h.width_in_sbs_minus_1.clear();
    h.height_in_sbs_minus_1.clear();

    if h.uniform_tile_spacing_flag {
        let mut tcl = min_log2_tile_cols;
        while tcl < max_log2_tile_cols {
            if br.read_bits(1)? == 0 {
                break;
            }
            tcl += 1;
        }
        tile_cols_log2 = tcl;
        let tile_width_sb = (sb_cols + (1 << tile_cols_log2) - 1) >> tile_cols_log2;
        let mut start_sb = 0u32;
        let mut mi_starts: Vec<u16> = vec![0];
        let mut widths: Vec<u16> = Vec::new();
        while start_sb < sb_cols {
            let size_sb = tile_width_sb.min(sb_cols - start_sb);
            widths.push((size_sb - 1) as u16);
            start_sb += size_sb;
            mi_starts.push(((start_sb << sb_size_log2).min(mi_cols)) as u16);
        }
        h.mi_col_starts = mi_starts;
        h.width_in_sbs_minus_1 = widths;
        h.tile_cols = h.width_in_sbs_minus_1.len() as u8;

        let min_log2_tile_rows = min_log2_tiles.saturating_sub(tile_cols_log2);
        let mut trl = min_log2_tile_rows;
        while trl < max_log2_tile_rows {
            if br.read_bits(1)? == 0 {
                break;
            }
            trl += 1;
        }
        tile_rows_log2 = trl;
        let tile_height_sb = (sb_rows + (1 << tile_rows_log2) - 1) >> tile_rows_log2;
        let mut start_sb_r = 0u32;
        let mut mi_starts_r: Vec<u16> = vec![0];
        let mut heights: Vec<u16> = Vec::new();
        while start_sb_r < sb_rows {
            let size_sb = tile_height_sb.min(sb_rows - start_sb_r);
            heights.push((size_sb - 1) as u16);
            start_sb_r += size_sb;
            mi_starts_r.push(((start_sb_r << sb_size_log2).min(mi_rows)) as u16);
        }
        h.mi_row_starts = mi_starts_r;
        h.height_in_sbs_minus_1 = heights;
        h.tile_rows = h.height_in_sbs_minus_1.len() as u8;
    } else {
        // Non-uniform tile spacing
        let mut start_sb = 0u32;
        let mut mi_starts: Vec<u16> = vec![0];
        let mut widths: Vec<u16> = Vec::new();
        while start_sb < sb_cols {
            let max_width = (sb_cols - start_sb).min(max_tile_width_sb);
            let size_minus_1 = av1_read_ns(br, max_width)?;
            let size = size_minus_1 + 1;
            widths.push(size_minus_1 as u16);
            start_sb += size;
            mi_starts.push(((start_sb << sb_size_log2).min(mi_cols)) as u16);
        }
        h.mi_col_starts = mi_starts;
        h.width_in_sbs_minus_1 = widths;
        h.tile_cols = h.width_in_sbs_minus_1.len() as u8;
        tile_cols_log2 = av1_tile_log2(1, h.tile_cols as u32);

        let tile_cols = h.tile_cols as u32;
        let max_tile_area_sb_r = if min_log2_tiles > 0 {
            (sb_rows * sb_cols) >> (min_log2_tiles + 1)
        } else {
            sb_rows * sb_cols
        };
        let max_tile_height_sb = (max_tile_area_sb_r / tile_cols).max(1);

        let mut start_sb_r = 0u32;
        let mut mi_starts_r: Vec<u16> = vec![0];
        let mut heights: Vec<u16> = Vec::new();
        while start_sb_r < sb_rows {
            let max_height = (sb_rows - start_sb_r).min(max_tile_height_sb);
            let size_minus_1 = av1_read_ns(br, max_height)?;
            let size = size_minus_1 + 1;
            heights.push(size_minus_1 as u16);
            start_sb_r += size;
            mi_starts_r.push(((start_sb_r << sb_size_log2).min(mi_rows)) as u16);
        }
        h.mi_row_starts = mi_starts_r;
        h.height_in_sbs_minus_1 = heights;
        h.tile_rows = h.height_in_sbs_minus_1.len() as u8;
        tile_rows_log2 = av1_tile_log2(1, h.tile_rows as u32);
    }
    h.tile_cols_log2 = tile_cols_log2 as u8;
    h.tile_rows_log2 = tile_rows_log2 as u8;

    if (tile_cols_log2 + tile_rows_log2) > 0 {
        let n = (tile_cols_log2 + tile_rows_log2) as usize;
        h.context_update_tile_id = br.read_bits(n)? as u16;
        h.tile_size_bytes_minus_1 = br.read_bits(2)? as u8;
    } else {
        h.context_update_tile_id = 0;
        h.tile_size_bytes_minus_1 = 0;
    }
    Some(())
}

/// AV1 tile_log2 helper (§5.9.15) — smallest k s.t. (blksize << k) >= target.
fn av1_tile_log2(blksize: u32, target: u32) -> u32 {
    let mut k = 0u32;
    while (blksize << k) < target {
        k += 1;
    }
    k
}

/// AV1 ns(n) — non-symmetric fixed-length code (§4.10.6)
fn av1_read_ns(br: &mut BitReader, n: u32) -> Option<u32> {
    if n == 0 {
        return Some(0);
    }
    let w = av1_ceil_log2(n);
    if w == 0 {
        return Some(0);
    }
    let m = (1u32 << w) - n;
    let v = br.read_bits((w - 1) as usize)?;
    if v < m {
        Some(v)
    } else {
        let extra = br.read_bits(1)?;
        Some((v << 1) - m + extra)
    }
}

fn av1_ceil_log2(n: u32) -> u32 {
    if n <= 1 {
        return 1;
    }
    let mut k = 0;
    let mut x = n - 1;
    while x > 0 {
        k += 1;
        x >>= 1;
    }
    k
}

/// §5.9.12 quantization_params()
fn parse_av1_quantization_params(
    br: &mut BitReader,
    h: &mut Av1FrameHeader,
    seq: &Av1SequenceHeader,
) -> Option<()> {
    h.base_q_idx = br.read_bits(8)? as u8;
    h.delta_q_y_dc = read_delta_q(br)?;
    let (diff_uv_delta, num_planes) = if seq.monochrome {
        (false, 1u32)
    } else {
        let diff = if seq.seq_profile == 2 {
            br.read_bits(1)? == 1
        } else {
            false
        };
        (diff, 3u32)
    };
    if num_planes > 1 {
        h.delta_q_u_dc = read_delta_q(br)?;
        h.delta_q_u_ac = read_delta_q(br)?;
        if diff_uv_delta {
            h.delta_q_v_dc = read_delta_q(br)?;
            h.delta_q_v_ac = read_delta_q(br)?;
        } else {
            h.delta_q_v_dc = h.delta_q_u_dc;
            h.delta_q_v_ac = h.delta_q_u_ac;
        }
    }
    h.using_qmatrix = br.read_bits(1)? == 1;
    if h.using_qmatrix {
        h.qm_y = br.read_bits(4)? as u8;
        h.qm_u = br.read_bits(4)? as u8;
        h.qm_v = if seq.monochrome {
            h.qm_u
        } else if br.read_bits(1)? == 0 {
            h.qm_u
        } else {
            br.read_bits(4)? as u8
        };
    }
    Some(())
}

fn read_delta_q(br: &mut BitReader) -> Option<i8> {
    let present = br.read_bits(1)? == 1;
    if present {
        Some(br.read_su(7)? as i8)
    } else {
        Some(0)
    }
}

/// §5.9.14 segmentation_params()
fn parse_av1_segmentation_params(br: &mut BitReader, h: &mut Av1FrameHeader) -> Option<()> {
    h.segmentation_enabled = br.read_bits(1)? == 1;
    if h.segmentation_enabled {
        if h.primary_ref_frame == 7 {
            // PRIMARY_REF_NONE → forced-fresh segment tree
            h.segmentation_update_map = true;
            h.segmentation_temporal_update = false;
            h.segmentation_update_data = true;
        } else {
            h.segmentation_update_map = br.read_bits(1)? == 1;
            if h.segmentation_update_map {
                h.segmentation_temporal_update = br.read_bits(1)? == 1;
            }
            h.segmentation_update_data = br.read_bits(1)? == 1;
        }
        if h.segmentation_update_data {
            // SEG_FEATURE_DATA table (§5.9.14) — per-feature bit counts
            // and sign flags.
            // (bits, signed)
            const FEAT_INFO: [(u32, bool); 8] = [
                (8, true),  // SEG_LVL_ALT_Q
                (6, true),  // SEG_LVL_ALT_LF_Y_V
                (6, true),  // SEG_LVL_ALT_LF_Y_H
                (6, true),  // SEG_LVL_ALT_LF_U
                (6, true),  // SEG_LVL_ALT_LF_V
                (3, false), // SEG_LVL_REF_FRAME
                (0, false), // SEG_LVL_SKIP
                (0, false), // SEG_LVL_GLOBALMV
            ];
            for seg in 0..8 {
                for (feat, &(bits, signed)) in FEAT_INFO.iter().enumerate() {
                    let enabled = br.read_bits(1)? == 1;
                    h.feature_enabled[seg][feat] = enabled;
                    if enabled {
                        if bits == 0 {
                            h.feature_data[seg][feat] = 1;
                        } else if signed {
                            h.feature_data[seg][feat] = br.read_su(bits as usize + 1)? as i16;
                        } else {
                            h.feature_data[seg][feat] = br.read_bits(bits as usize)? as i16;
                        }
                    }
                }
            }
        }
    }
    Some(())
}

/// §5.9.11 loop_filter_params()
fn parse_av1_loop_filter_params(
    br: &mut BitReader,
    h: &mut Av1FrameHeader,
    frame_is_intra: bool,
) -> Option<()> {
    if h.coded_lossless || h.allow_intrabc {
        h.loop_filter_level = [0; 4];
        h.loop_filter_sharpness = 0;
        h.loop_filter_delta_enabled = false;
        h.loop_filter_ref_deltas = [1, 0, 0, 0, -1, 0, -1, -1];
        h.loop_filter_mode_deltas = [0, 0];
        return Some(());
    }
    h.loop_filter_level[0] = br.read_bits(6)? as u8;
    h.loop_filter_level[1] = br.read_bits(6)? as u8;
    if h.loop_filter_level[0] > 0 || h.loop_filter_level[1] > 0 {
        h.loop_filter_level[2] = br.read_bits(6)? as u8;
        h.loop_filter_level[3] = br.read_bits(6)? as u8;
    }
    h.loop_filter_sharpness = br.read_bits(3)? as u8;
    h.loop_filter_delta_enabled = br.read_bits(1)? == 1;
    // Defaults for ref/mode deltas (§5.9.11)
    h.loop_filter_ref_deltas = [1, 0, 0, 0, -1, 0, -1, -1];
    h.loop_filter_mode_deltas = [0, 0];
    if h.loop_filter_delta_enabled {
        h.loop_filter_delta_update = br.read_bits(1)? == 1;
        if h.loop_filter_delta_update {
            let mut update_mask = 0u8;
            for i in 0..8 {
                let update = br.read_bits(1)? == 1;
                if update {
                    update_mask |= 1 << i;
                    h.loop_filter_ref_deltas[i] = br.read_su(7)? as i8;
                }
            }
            h.update_ref_delta_mask = update_mask;
            let mut mode_mask = 0u8;
            for i in 0..2 {
                let update = br.read_bits(1)? == 1;
                if update {
                    mode_mask |= 1 << i;
                    h.loop_filter_mode_deltas[i] = br.read_su(7)? as i8;
                }
            }
            h.update_mode_delta_mask = mode_mask;
        }
    }
    let _ = frame_is_intra; // reserved for future spec tweaks
    Some(())
}

/// §5.9.19 cdef_params()
fn parse_av1_cdef_params(
    br: &mut BitReader,
    h: &mut Av1FrameHeader,
    num_planes: u32,
) -> Option<()> {
    h.cdef_damping_minus_3 = br.read_bits(2)? as u8;
    h.cdef_bits = br.read_bits(2)? as u8;
    let count = 1usize << h.cdef_bits;
    for i in 0..count {
        h.cdef_y_pri_strength[i] = br.read_bits(4)? as u8;
        let y_sec = br.read_bits(2)? as u8;
        // Spec §5.9.19: after reading cdef_y_sec_strength, if the
        // decoded value == 3 it's remapped to 4 (the "== 3 → 4" gap
        // in the 2-bit encoding). Same for chroma below.
        h.cdef_y_sec_strength[i] = if y_sec == 3 { 4 } else { y_sec };
        if num_planes > 1 {
            h.cdef_uv_pri_strength[i] = br.read_bits(4)? as u8;
            let uv_sec = br.read_bits(2)? as u8;
            h.cdef_uv_sec_strength[i] = if uv_sec == 3 { 4 } else { uv_sec };
        }
    }
    Some(())
}

/// §5.9.20 lr_params()
fn parse_av1_lr_params(
    br: &mut BitReader,
    h: &mut Av1FrameHeader,
    num_planes: u32,
    seq: &Av1SequenceHeader,
) -> Option<()> {
    let mut uses_lr = false;
    let mut uses_chroma_lr = false;
    for i in 0..(num_planes as usize) {
        let lr_type = br.read_bits(2)? as u8;
        h.lr_type[i] = lr_type;
        if lr_type != 0 {
            uses_lr = true;
            if i > 0 {
                uses_chroma_lr = true;
            }
        }
    }
    if uses_lr {
        // 64x64 SB path (use_128x128_superblock=0 — we assume this):
        // read 1 bit; if set, read another for lr_unit_extra_shift.
        // 128x128 SB path: read 1 bit and add 1 (to get 128/256).
        // We don't track use_128x128_superblock — stick to 64x64.
        let base = br.read_bits(1)? as u8;
        h.lr_unit_shift = if base != 0 {
            let extra = br.read_bits(1)? as u8;
            base + extra
        } else {
            0
        };
        // lr_uv_shift only present when chroma is 4:2:0 (subx && suby)
        // AND chroma plane has LR enabled.
        if num_planes > 1 && uses_chroma_lr && seq.chroma_subsampling_x && seq.chroma_subsampling_y
        {
            h.lr_uv_shift = br.read_bits(1)? as u8;
        }
    }
    Some(())
}

/// §5.9.21 global_motion_params() — read-only; we don't populate
/// StdVideoAV1GlobalMotion so just consume the bits to keep the
/// parser position in sync.
fn skip_av1_global_motion_params(br: &mut BitReader) -> Option<()> {
    for _ in 0..7 {
        let is_global = br.read_bits(1)? == 1;
        let is_rot_zoom = if is_global {
            br.read_bits(1)? == 1
        } else {
            false
        };
        let _is_translation = if is_global && !is_rot_zoom {
            br.read_bits(1)? == 1
        } else {
            false
        };
        let gm_type = if is_global && !is_rot_zoom {
            2u8 /*TRANSLATION*/
        } else if is_rot_zoom {
            3u8 /*ROTZOOM*/
        } else if is_global {
            4u8 /*AFFINE*/
        } else {
            0u8 /*IDENTITY*/
        };
        if gm_type >= 3 {
            // 2 × 6 subexp params
            for _ in 0..2 {
                let _a = av1_read_subexp(br, 12, 0)?;
                let _b = av1_read_subexp(br, 12, 0)?;
            }
        }
        if gm_type >= 2 {
            // 2 × 6 subexp params for translation
            for _ in 0..2 {
                let _a = av1_read_subexp(br, 12, 0)?;
            }
        }
    }
    Some(())
}

fn av1_read_subexp(br: &mut BitReader, num_syms: u32, _ref: i32) -> Option<i32> {
    // Simplified: read the inv_remap_and_deltaAV1 signed field. We
    // only need to advance the bit cursor — value is discarded.
    // §5.11.21: inv_remap_and_delta recurrence. The simplified "skip
    // enough bits" form reads ceil(log2(num_syms)) + sign bits.
    let bits = av1_ceil_log2(num_syms) as usize + 1; // value + sign
    let _ = br.read_bits(bits.min(16))?;
    Some(0)
}

/// §5.9.25 film_grain_params() — we don't ship film-grain support in
/// the Vulkan scope; skip past the bits to keep parser position in
/// sync for byte_align().
fn skip_av1_film_grain_params(br: &mut BitReader, seq: &Av1SequenceHeader) -> Option<()> {
    let apply_grain = br.read_bits(1)? == 1;
    if !apply_grain {
        return Some(());
    }
    let _grain_seed = br.read_bits(16)?;
    let update_grain = br.read_bits(1)? == 1;
    if !update_grain {
        let _film_grain_params_ref_idx = br.read_bits(3)?;
        return Some(());
    }
    let num_y_points = br.read_bits(4)?;
    for _ in 0..num_y_points {
        let _point_y_value = br.read_bits(8)?;
        let _point_y_scaling = br.read_bits(8)?;
    }
    let chroma_scaling_from_luma = if seq.monochrome {
        false
    } else {
        br.read_bits(1)? == 1
    };
    let num_cb_points: u32;
    let num_cr_points: u32;
    if seq.monochrome
        || chroma_scaling_from_luma
        || (seq.chroma_subsampling_x && seq.chroma_subsampling_y && num_y_points == 0)
    {
        num_cb_points = 0;
        num_cr_points = 0;
    } else {
        num_cb_points = br.read_bits(4)?;
        for _ in 0..num_cb_points {
            let _point_cb_value = br.read_bits(8)?;
            let _point_cb_scaling = br.read_bits(8)?;
        }
        num_cr_points = br.read_bits(4)?;
        for _ in 0..num_cr_points {
            let _point_cr_value = br.read_bits(8)?;
            let _point_cr_scaling = br.read_bits(8)?;
        }
    }
    let _grain_scaling_minus_8 = br.read_bits(2)?;
    let ar_coeff_lag = br.read_bits(2)?;
    let num_pos_y = 2 * ar_coeff_lag * (ar_coeff_lag + 1);
    let num_pos_chroma = if num_y_points > 0 {
        num_pos_y + 1
    } else {
        num_pos_y
    };
    for _ in 0..num_pos_y {
        let _ar_coeff_y_plus_128 = br.read_bits(8)?;
    }
    if chroma_scaling_from_luma || num_cb_points > 0 {
        for _ in 0..num_pos_chroma {
            let _ar_coeff_cb_plus_128 = br.read_bits(8)?;
        }
    }
    if chroma_scaling_from_luma || num_cr_points > 0 {
        for _ in 0..num_pos_chroma {
            let _ar_coeff_cr_plus_128 = br.read_bits(8)?;
        }
    }
    let _ar_coeff_shift_minus_6 = br.read_bits(2)?;
    let _grain_scale_shift = br.read_bits(2)?;
    if num_cb_points > 0 {
        let _cb_mult = br.read_bits(8)?;
        let _cb_luma_mult = br.read_bits(8)?;
        let _cb_offset = br.read_bits(9)?;
    }
    if num_cr_points > 0 {
        let _cr_mult = br.read_bits(8)?;
        let _cr_luma_mult = br.read_bits(8)?;
        let _cr_offset = br.read_bits(9)?;
    }
    let _overlap_flag = br.read_bits(1)?;
    let _clip_to_restricted_range = br.read_bits(1)?;
    Some(())
}
