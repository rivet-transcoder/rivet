//! Picture parameter set (H.265 clause 7.3.2.3), including the tile
//! layout it defines.

use crate::bitreader::BitReader;
use crate::{Error, Result};

use super::sps::{ScalingList, Sps, parse_scaling_list_data};

/// A parsed PPS.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pps {
    /// `pps_pic_parameter_set_id`.
    pub id: u32,
    /// `pps_seq_parameter_set_id`.
    pub sps_id: u32,
    /// `dependent_slice_segments_enabled_flag`.
    pub dependent_slice_segments_enabled: bool,
    /// `output_flag_present_flag`.
    pub output_flag_present: bool,
    /// `num_extra_slice_header_bits`.
    pub num_extra_slice_header_bits: u32,
    /// `sign_data_hiding_enabled_flag`.
    pub sign_data_hiding: bool,
    /// `cabac_init_present_flag`.
    pub cabac_init_present: bool,
    /// `num_ref_idx_l0_default_active_minus1 + 1`.
    pub num_ref_idx_l0_default: u32,
    /// `num_ref_idx_l1_default_active_minus1 + 1`.
    pub num_ref_idx_l1_default: u32,
    /// `init_qp_minus26 + 26`.
    pub init_qp: i32,
    /// `constrained_intra_pred_flag`.
    pub constrained_intra_pred: bool,
    /// `transform_skip_enabled_flag`.
    pub transform_skip_enabled: bool,
    /// `cu_qp_delta_enabled_flag`.
    pub cu_qp_delta_enabled: bool,
    /// `diff_cu_qp_delta_depth`.
    pub diff_cu_qp_delta_depth: u32,
    /// `pps_cb_qp_offset`.
    pub cb_qp_offset: i32,
    /// `pps_cr_qp_offset`.
    pub cr_qp_offset: i32,
    /// `pps_slice_chroma_qp_offsets_present_flag`.
    pub slice_chroma_qp_offsets_present: bool,
    /// `weighted_pred_flag`.
    pub weighted_pred: bool,
    /// `weighted_bipred_flag`.
    pub weighted_bipred: bool,
    /// `transquant_bypass_enabled_flag`.
    pub transquant_bypass_enabled: bool,
    /// `tiles_enabled_flag`.
    pub tiles_enabled: bool,
    /// `entropy_coding_sync_enabled_flag` (WPP).
    pub entropy_coding_sync: bool,
    /// Tile column boundaries in CTBs (`colBd`, `num_tile_columns + 1` entries),
    /// filled in by [`Pps::resolve_tiles`].
    pub col_bd: Vec<u32>,
    /// Tile row boundaries in CTBs.
    pub row_bd: Vec<u32>,
    /// Raw tile signalling: `(num_columns, num_rows, uniform_spacing, column_widths_minus1, row_heights_minus1)`.
    pub tiles_raw: (u32, u32, bool, Vec<u32>, Vec<u32>),
    /// `loop_filter_across_tiles_enabled_flag`.
    pub loop_filter_across_tiles: bool,
    /// `pps_loop_filter_across_slices_enabled_flag`.
    pub loop_filter_across_slices: bool,
    /// `deblocking_filter_override_enabled_flag`.
    pub deblocking_override_enabled: bool,
    /// `pps_deblocking_filter_disabled_flag`.
    pub deblocking_disabled: bool,
    /// `pps_beta_offset_div2 * 2`.
    pub beta_offset: i32,
    /// `pps_tc_offset_div2 * 2`.
    pub tc_offset: i32,
    /// PPS scaling lists, when `pps_scaling_list_data_present_flag`.
    pub scaling_list: Option<ScalingList>,
    /// `lists_modification_present_flag`.
    pub lists_modification_present: bool,
    /// `log2_parallel_merge_level_minus2 + 2`.
    pub log2_parallel_merge_level: u32,
    /// `slice_segment_header_extension_present_flag`.
    pub slice_header_extension_present: bool,
    /// Range extension present.
    pub range_ext: bool,
    /// `Log2MaxTransformSkipSize` (2 without the range extension).
    pub log2_max_transform_skip_size: u32,
    /// `cross_component_prediction_enabled_flag` (refused).
    pub cross_component_prediction: bool,
    /// `chroma_qp_offset_list_enabled_flag` (refused).
    pub chroma_qp_offset_list: bool,
    /// `log2_sao_offset_scale_luma`, `log2_sao_offset_scale_chroma`.
    pub log2_sao_offset_scale: (u32, u32),
}

impl Pps {
    /// Parse a PPS RBSP.
    pub fn parse(rbsp: &[u8]) -> Result<Pps> {
        let mut r = BitReader::new(rbsp);
        let id = r.ue();
        if id > 63 {
            return Err(Error::bitstream("pps_pic_parameter_set_id out of range"));
        }
        let sps_id = r.ue();
        if sps_id > 15 {
            return Err(Error::bitstream("pps_seq_parameter_set_id out of range"));
        }
        let dependent_slice_segments_enabled = r.flag();
        let output_flag_present = r.flag();
        let num_extra_slice_header_bits = r.bits(3);
        let sign_data_hiding = r.flag();
        let cabac_init_present = r.flag();
        let num_ref_idx_l0_default = r.ue() + 1;
        let num_ref_idx_l1_default = r.ue() + 1;
        if num_ref_idx_l0_default > 15 || num_ref_idx_l1_default > 15 {
            return Err(Error::bitstream("num_ref_idx_default_active out of range"));
        }
        let init_qp = r.se() + 26;
        let constrained_intra_pred = r.flag();
        let transform_skip_enabled = r.flag();
        let cu_qp_delta_enabled = r.flag();
        let mut diff_cu_qp_delta_depth = 0;
        if cu_qp_delta_enabled {
            diff_cu_qp_delta_depth = r.ue();
        }
        let cb_qp_offset = r.se();
        let cr_qp_offset = r.se();
        if !(-12..=12).contains(&cb_qp_offset) || !(-12..=12).contains(&cr_qp_offset) {
            return Err(Error::bitstream("pps chroma qp offset out of range"));
        }
        let slice_chroma_qp_offsets_present = r.flag();
        let weighted_pred = r.flag();
        let weighted_bipred = r.flag();
        let transquant_bypass_enabled = r.flag();
        let tiles_enabled = r.flag();
        let entropy_coding_sync = r.flag();
        let mut tiles_raw = (1, 1, true, Vec::new(), Vec::new());
        let mut loop_filter_across_tiles = true;
        if tiles_enabled {
            let cols = r.ue() + 1;
            let rows = r.ue() + 1;
            if cols > 20 || rows > 22 {
                return Err(Error::bitstream("tile count out of range"));
            }
            let uniform = r.flag();
            let mut cw = Vec::new();
            let mut rh = Vec::new();
            if !uniform {
                for _ in 0..cols - 1 {
                    cw.push(r.ue());
                }
                for _ in 0..rows - 1 {
                    rh.push(r.ue());
                }
            }
            loop_filter_across_tiles = r.flag();
            tiles_raw = (cols, rows, uniform, cw, rh);
        }
        let loop_filter_across_slices = r.flag();
        let mut deblocking_override_enabled = false;
        let mut deblocking_disabled = false;
        let mut beta_offset = 0;
        let mut tc_offset = 0;
        if r.flag() {
            // deblocking_filter_control_present_flag
            deblocking_override_enabled = r.flag();
            deblocking_disabled = r.flag();
            if !deblocking_disabled {
                beta_offset = r.se() * 2;
                tc_offset = r.se() * 2;
                if !(-12..=12).contains(&beta_offset) || !(-12..=12).contains(&tc_offset) {
                    return Err(Error::bitstream("pps deblocking offsets out of range"));
                }
            }
        }
        let mut scaling_list = None;
        if r.flag() {
            scaling_list = Some(parse_scaling_list_data(&mut r)?);
        }
        let lists_modification_present = r.flag();
        let log2_parallel_merge_level = r.ue() + 2;
        let slice_header_extension_present = r.flag();
        let mut range_ext = false;
        let mut log2_max_transform_skip_size = 2;
        let mut cross_component_prediction = false;
        let mut chroma_qp_offset_list = false;
        let mut log2_sao_offset_scale = (0, 0);
        if r.flag() {
            range_ext = r.flag();
            let _multilayer = r.flag();
            let _3d = r.flag();
            let _scc = r.flag();
            r.bits(4);
            if range_ext {
                // pps_range_extension()
                if transform_skip_enabled {
                    log2_max_transform_skip_size = r.ue() + 2;
                }
                cross_component_prediction = r.flag();
                chroma_qp_offset_list = r.flag();
                if chroma_qp_offset_list {
                    let _diff_cu_chroma_qp_offset_depth = r.ue();
                    let len = r.ue() + 1;
                    for _ in 0..len {
                        let _cb = r.se();
                        let _cr = r.se();
                    }
                }
                log2_sao_offset_scale = (r.ue(), r.ue());
                if log2_max_transform_skip_size > 5 || log2_sao_offset_scale.0 > 6 || log2_sao_offset_scale.1 > 6 {
                    return Err(Error::bitstream("pps_range_extension values out of range"));
                }
            }
            // Other extensions are not parsed: their data sits before the
            // trailing bits, so do not insist on rbsp_trailing_bits here.
            return Ok(Pps {
                id,
                sps_id,
                dependent_slice_segments_enabled,
                output_flag_present,
                num_extra_slice_header_bits,
                sign_data_hiding,
                cabac_init_present,
                num_ref_idx_l0_default,
                num_ref_idx_l1_default,
                init_qp,
                constrained_intra_pred,
                transform_skip_enabled,
                cu_qp_delta_enabled,
                diff_cu_qp_delta_depth,
                cb_qp_offset,
                cr_qp_offset,
                slice_chroma_qp_offsets_present,
                weighted_pred,
                weighted_bipred,
                transquant_bypass_enabled,
                tiles_enabled,
                entropy_coding_sync,
                col_bd: Vec::new(),
                row_bd: Vec::new(),
                tiles_raw,
                loop_filter_across_tiles,
                loop_filter_across_slices,
                deblocking_override_enabled,
                deblocking_disabled,
                beta_offset,
                tc_offset,
                scaling_list,
                lists_modification_present,
                log2_parallel_merge_level,
                slice_header_extension_present,
                range_ext,
                log2_max_transform_skip_size,
                cross_component_prediction,
                chroma_qp_offset_list,
                log2_sao_offset_scale,
            });
        }
        r.finish("PPS")?;
        Ok(Pps {
            id,
            sps_id,
            dependent_slice_segments_enabled,
            output_flag_present,
            num_extra_slice_header_bits,
            sign_data_hiding,
            cabac_init_present,
            num_ref_idx_l0_default,
            num_ref_idx_l1_default,
            init_qp,
            constrained_intra_pred,
            transform_skip_enabled,
            cu_qp_delta_enabled,
            diff_cu_qp_delta_depth,
            cb_qp_offset,
            cr_qp_offset,
            slice_chroma_qp_offsets_present,
            weighted_pred,
            weighted_bipred,
            transquant_bypass_enabled,
            tiles_enabled,
            entropy_coding_sync,
            col_bd: Vec::new(),
            row_bd: Vec::new(),
            tiles_raw,
            loop_filter_across_tiles,
            loop_filter_across_slices,
            deblocking_override_enabled,
            deblocking_disabled,
            beta_offset,
            tc_offset,
            scaling_list,
            lists_modification_present,
            log2_parallel_merge_level,
            slice_header_extension_present,
            range_ext,
            log2_max_transform_skip_size,
            cross_component_prediction,
            chroma_qp_offset_list,
            log2_sao_offset_scale,
        })
    }

    /// Resolve the tile boundaries against an SPS (6.5.1): `colBd`, `rowBd`.
    pub fn resolve_tiles(&mut self, sps: &Sps) -> Result<()> {
        let wc = sps.pic_width_in_ctbs();
        let hc = sps.pic_height_in_ctbs();
        let (cols, rows, uniform, cw, rh) = &self.tiles_raw;
        let (cols, rows) = (*cols as usize, *rows as usize);
        let mut col_widths = vec![0u32; cols];
        let mut row_heights = vec![0u32; rows];
        if *uniform {
            for i in 0..cols {
                col_widths[i] = ((i as u32 + 1) * wc) / cols as u32 - (i as u32 * wc) / cols as u32;
            }
            for j in 0..rows {
                row_heights[j] = ((j as u32 + 1) * hc) / rows as u32 - (j as u32 * hc) / rows as u32;
            }
        } else {
            let mut sum = 0;
            for i in 0..cols - 1 {
                col_widths[i] = cw[i] + 1;
                sum += col_widths[i];
            }
            if sum >= wc {
                return Err(Error::bitstream("tile columns wider than the picture"));
            }
            col_widths[cols - 1] = wc - sum;
            let mut sum = 0;
            for j in 0..rows - 1 {
                row_heights[j] = rh[j] + 1;
                sum += row_heights[j];
            }
            if sum >= hc {
                return Err(Error::bitstream("tile rows taller than the picture"));
            }
            row_heights[rows - 1] = hc - sum;
        }
        let mut col_bd = vec![0u32; cols + 1];
        for i in 0..cols {
            col_bd[i + 1] = col_bd[i] + col_widths[i];
        }
        let mut row_bd = vec![0u32; rows + 1];
        for j in 0..rows {
            row_bd[j + 1] = row_bd[j] + row_heights[j];
        }
        if col_widths.iter().any(|&w| w == 0) || row_heights.iter().any(|&h| h == 0) {
            return Err(Error::bitstream("empty tile"));
        }
        self.col_bd = col_bd;
        self.row_bd = row_bd;
        Ok(())
    }
}
