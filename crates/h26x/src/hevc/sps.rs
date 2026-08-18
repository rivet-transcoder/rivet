//! Video and sequence parameter sets (H.265 clauses 7.3.2.1 / 7.3.2.2), the
//! short-term reference picture set syntax (7.3.7) and scaling list data
//! (7.3.4).

use crate::bitreader::BitReader;
use crate::{Error, Result};

use super::tables::{DEFAULT_SCALING_INTER, DEFAULT_SCALING_INTRA, DIAG_SCAN4X4_X, DIAG_SCAN4X4_Y, DIAG_SCAN8X8_X, DIAG_SCAN8X8_Y};

/// `profile_tier_level()` — the general (top) level only.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ProfileTierLevel {
    /// `general_profile_idc`.
    pub profile_idc: u8,
    /// `general_tier_flag`.
    pub tier: bool,
    /// `general_level_idc`.
    pub level_idc: u8,
    /// `general_profile_compatibility_flag[]` as bits (bit j = flag j).
    pub compat: u32,
}

/// Parse `profile_tier_level(profilePresentFlag, maxNumSubLayersMinus1)`.
fn parse_ptl(r: &mut BitReader, profile_present: bool, max_sub_layers_minus1: u32) -> ProfileTierLevel {
    let mut ptl = ProfileTierLevel::default();
    if profile_present {
        r.bits(2); // profile_space
        ptl.tier = r.flag();
        ptl.profile_idc = r.bits(5) as u8;
        ptl.compat = r.bits(32);
        r.bits(4); // progressive, interlaced, non_packed, frame_only
        r.bits(32); // reserved / range extension flags
        r.bits(11);
        r.bit(); // general_inbld_flag / reserved
    }
    ptl.level_idc = r.bits(8) as u8;
    let mut sub_profile_present = [false; 8];
    let mut sub_level_present = [false; 8];
    for i in 0..max_sub_layers_minus1 as usize {
        sub_profile_present[i] = r.flag();
        sub_level_present[i] = r.flag();
    }
    if max_sub_layers_minus1 > 0 {
        for _ in max_sub_layers_minus1..8 {
            r.bits(2); // reserved_zero_2bits
        }
    }
    for i in 0..max_sub_layers_minus1 as usize {
        if sub_profile_present[i] {
            r.bits(2);
            r.bit();
            r.bits(5);
            r.bits(32);
            r.bits(4);
            r.bits(32);
            r.bits(11);
            r.bit();
        }
        if sub_level_present[i] {
            r.bits(8);
        }
    }
    ptl
}

/// A parsed VPS (only what a single-layer decoder needs: that it exists).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Vps {
    /// `vps_video_parameter_set_id`.
    pub id: u32,
    /// `vps_max_sub_layers_minus1 + 1`.
    pub max_sub_layers: u32,
    /// The general profile/tier/level.
    pub ptl: ProfileTierLevel,
}

impl Vps {
    /// Parse a VPS RBSP.
    pub fn parse(rbsp: &[u8]) -> Result<Vps> {
        let mut r = BitReader::new(rbsp);
        let id = r.bits(4);
        r.bit(); // vps_base_layer_internal_flag
        r.bit(); // vps_base_layer_available_flag
        r.bits(6); // vps_max_layers_minus1
        let max_sub_layers = r.bits(3) + 1;
        r.bit(); // temporal_id_nesting
        r.bits(16); // reserved 0xffff
        let ptl = parse_ptl(&mut r, true, max_sub_layers - 1);
        // The rest (sub-layer ordering, layer id, timing, hrd) is not needed.
        if r.overrun() {
            return Err(Error::bitstream("VPS truncated"));
        }
        Ok(Vps { id, max_sub_layers, ptl })
    }
}

/// Scaling lists per `sizeId` (0: 4x4, 1: 8x8, 2: 16x16, 3: 32x32) and
/// `matrixId` (0..5; for 32x32 only 0 and 3 are signalled, the rest copied
/// per 7.3.4), stored in raster order at their coded size (4x4 has 16, the
/// rest 64 entries plus a DC value for 16x16/32x32).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScalingList {
    /// `[sizeId][matrixId][coef]` (16 or 64 values, raster at coded size).
    pub lists: [[[u8; 64]; 6]; 4],
    /// `scaling_list_dc_coef` for sizeId 2 and 3: `[sizeId-2][matrixId]`.
    pub dc: [[u8; 6]; 2],
}

impl ScalingList {
    /// The default lists (Tables 7-5/7-6).
    pub fn default_lists() -> Self {
        let mut s = ScalingList { lists: [[[16; 64]; 6]; 4], dc: [[16; 6]; 2] };
        for size in 1..4 {
            for m in 0..6 {
                s.lists[size][m] = if m < 3 { DEFAULT_SCALING_INTRA } else { DEFAULT_SCALING_INTER };
            }
        }
        s
    }
    /// Flat 16.
    pub fn flat() -> Self {
        ScalingList { lists: [[[16; 64]; 6]; 4], dc: [[16; 6]; 2] }
    }
}

/// Parse `scaling_list_data()` (7.3.4) into raster-ordered lists.
pub fn parse_scaling_list_data(r: &mut BitReader) -> Result<ScalingList> {
    let mut sl = ScalingList::default_lists();
    for size_id in 0..4usize {
        let step = if size_id == 3 { 3 } else { 1 };
        let mut matrix_id = 0usize;
        while matrix_id < 6 {
            let pred_mode_flag = r.flag();
            if !pred_mode_flag {
                let delta = r.ue() as usize * step;
                if delta == 0 {
                    // Default list.
                    sl.lists[size_id][matrix_id] = if size_id == 0 {
                        [16u8; 64]
                    } else if matrix_id < 3 {
                        DEFAULT_SCALING_INTRA
                    } else {
                        DEFAULT_SCALING_INTER
                    };
                    if size_id >= 2 {
                        sl.dc[size_id - 2][matrix_id] = 16;
                    }
                } else {
                    if delta > matrix_id {
                        return Err(Error::bitstream("scaling_list_pred_matrix_id_delta out of range"));
                    }
                    let ref_id = matrix_id - delta;
                    sl.lists[size_id][matrix_id] = sl.lists[size_id][ref_id];
                    if size_id >= 2 {
                        sl.dc[size_id - 2][matrix_id] = sl.dc[size_id - 2][ref_id];
                    }
                }
            } else {
                let coef_num = 64.min(1 << (4 + (size_id << 1)));
                let mut next: i32 = 8;
                if size_id > 1 {
                    let dc = r.se();
                    if !(-7..=247).contains(&dc) {
                        return Err(Error::bitstream("scaling_list_dc_coef out of range"));
                    }
                    next = dc + 8;
                    sl.dc[size_id - 2][matrix_id] = next as u8;
                }
                let mut list = [0u8; 64];
                for i in 0..coef_num {
                    let delta = r.se();
                    if !(-128..=127).contains(&delta) {
                        return Err(Error::bitstream("scaling_list_delta_coef out of range"));
                    }
                    next = (next + delta + 256) % 256;
                    // Up-right diagonal scan position i -> raster.
                    let pos = if size_id == 0 {
                        (DIAG_SCAN4X4_Y[i] as usize) * 4 + DIAG_SCAN4X4_X[i] as usize
                    } else {
                        (DIAG_SCAN8X8_Y[i] as usize) * 8 + DIAG_SCAN8X8_X[i] as usize
                    };
                    list[pos] = next as u8;
                }
                sl.lists[size_id][matrix_id] = list;
            }
            matrix_id += step;
        }
    }
    // 32x32: matrixId 1, 2, 4, 5 are copied from the 16x16... no — per
    // 7.4.5, for sizeId 3 only matrixId 0 and 3 exist; the chroma 32x32
    // lists (used only for 4:4:4) take the 16x16 values. Copy so lookups
    // never hit an unset slot.
    for m in [1usize, 2, 4, 5] {
        sl.lists[3][m] = sl.lists[2][m];
        sl.dc[1][m] = sl.dc[0][m];
    }
    Ok(sl)
}

/// A short-term reference picture set (7.4.8), as `(delta_poc,
/// used_by_curr_pic)` lists: negative pictures in decreasing POC order (i.e.
/// closest first), positive pictures in increasing order.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct StRps {
    /// `DeltaPocS0[i]`, `UsedByCurrPicS0[i]`.
    pub neg: Vec<(i32, bool)>,
    /// `DeltaPocS1[i]`, `UsedByCurrPicS1[i]`.
    pub pos: Vec<(i32, bool)>,
}

impl StRps {
    /// `NumDeltaPocs`.
    pub fn num_delta_pocs(&self) -> usize {
        self.neg.len() + self.pos.len()
    }
}

/// Parse `st_ref_pic_set(stRpsIdx)` given the sets already parsed (for
/// inter-RPS prediction). `num_short_term_ref_pic_sets` is the SPS count
/// (the slice-header set has index equal to it).
pub fn parse_st_rps(r: &mut BitReader, idx: usize, num_sets: usize, sets: &[StRps]) -> Result<StRps> {
    let mut inter_rps = false;
    if idx != 0 {
        inter_rps = r.flag();
    }
    if inter_rps {
        let mut delta_idx = 1usize;
        if idx == num_sets {
            delta_idx = r.ue() as usize + 1;
        }
        if delta_idx > idx {
            return Err(Error::bitstream("delta_idx_minus1 out of range"));
        }
        let ref_idx = idx - delta_idx;
        let sign = r.bit() as i32;
        let abs = r.ue() as i32 + 1;
        let delta_rps = (1 - 2 * sign) * abs;
        let reference = &sets[ref_idx];
        let n = reference.num_delta_pocs();
        let mut used = vec![false; n + 1];
        let mut use_delta = vec![true; n + 1];
        for j in 0..=n {
            used[j] = r.flag();
            if !used[j] {
                use_delta[j] = r.flag();
            }
        }
        // 7-61 / 7-62.
        let mut neg = Vec::new();
        let mut pos = Vec::new();
        // dPoc of the reference set's entries, in the order used by the
        // derivation: negatives (S0), positives (S1); the last entry (index
        // n) is the reference picture itself (delta 0).
        let ref_neg = &reference.neg;
        let ref_pos = &reference.pos;
        // Negative pictures.
        {
            let mut i = 0usize;
            for j in (0..ref_pos.len()).rev() {
                let d = ref_pos[j].0 + delta_rps;
                let jj = ref_neg.len() + j;
                if d < 0 && use_delta[jj] {
                    neg.push((d, used[jj]));
                    i += 1;
                }
            }
            if delta_rps < 0 && use_delta[n] {
                neg.push((delta_rps, used[n]));
                i += 1;
            }
            for j in 0..ref_neg.len() {
                let d = ref_neg[j].0 + delta_rps;
                if d < 0 && use_delta[j] {
                    neg.push((d, used[j]));
                    i += 1;
                }
            }
            let _ = i;
        }
        // Positive pictures.
        {
            for j in (0..ref_neg.len()).rev() {
                let d = ref_neg[j].0 + delta_rps;
                if d > 0 && use_delta[j] {
                    pos.push((d, used[j]));
                }
            }
            if delta_rps > 0 && use_delta[n] {
                pos.push((delta_rps, used[n]));
            }
            for j in 0..ref_pos.len() {
                let d = ref_pos[j].0 + delta_rps;
                let jj = ref_neg.len() + j;
                if d > 0 && use_delta[jj] {
                    pos.push((d, used[jj]));
                }
            }
        }
        if neg.len() + pos.len() > 16 {
            return Err(Error::bitstream("reference picture set too large"));
        }
        Ok(StRps { neg, pos })
    } else {
        let num_neg = r.ue() as usize;
        let num_pos = r.ue() as usize;
        if num_neg > 16 || num_pos > 16 || num_neg + num_pos > 16 {
            return Err(Error::bitstream("num_negative/positive_pics out of range"));
        }
        let mut neg = Vec::with_capacity(num_neg);
        let mut poc = 0i32;
        for _ in 0..num_neg {
            let d = r.ue() as i32 + 1;
            poc -= d;
            let used = r.flag();
            neg.push((poc, used));
        }
        let mut pos = Vec::with_capacity(num_pos);
        poc = 0;
        for _ in 0..num_pos {
            let d = r.ue() as i32 + 1;
            poc += d;
            let used = r.flag();
            pos.push((poc, used));
        }
        Ok(StRps { neg, pos })
    }
}

/// VUI fields the decoder uses.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Vui {
    /// `video_full_range_flag`.
    pub full_range: bool,
    /// `(colour_primaries, transfer_characteristics, matrix_coeffs)`.
    pub colour_description: Option<(u8, u8, u8)>,
    /// `(num_units_in_tick, time_scale)`.
    pub timing: Option<(u32, u32)>,
    /// `default_display_window` offsets in luma samples (l, r, t, b) — kept
    /// separate from the conformance window; not applied by default.
    pub default_display_window: (u32, u32, u32, u32),
}

fn parse_sub_layer_hrd(r: &mut BitReader, cpb_cnt: u32, sub_pic: bool) {
    for _ in 0..cpb_cnt {
        r.ue();
        r.ue();
        if sub_pic {
            r.ue();
            r.ue();
        }
        r.flag();
    }
}

fn parse_hrd(r: &mut BitReader, common_inf: bool, max_sub_layers_minus1: u32) {
    let mut nal_hrd = false;
    let mut vcl_hrd = false;
    let mut sub_pic = false;
    if common_inf {
        nal_hrd = r.flag();
        vcl_hrd = r.flag();
        if nal_hrd || vcl_hrd {
            sub_pic = r.flag();
            if sub_pic {
                r.bits(8);
                r.bits(5);
                r.flag();
                r.bits(5);
            }
            r.bits(4);
            r.bits(4);
            if sub_pic {
                r.bits(4);
            }
            r.bits(5);
            r.bits(5);
            r.bits(5);
        }
    }
    for _ in 0..=max_sub_layers_minus1 {
        let fixed_general = r.flag();
        let mut fixed_within_cvs = true;
        if !fixed_general {
            fixed_within_cvs = r.flag();
        }
        let mut low_delay = false;
        if fixed_within_cvs {
            r.ue(); // elemental_duration_in_tc_minus1
        } else {
            low_delay = r.flag();
        }
        let mut cpb_cnt = 1;
        if !low_delay {
            cpb_cnt = r.ue() + 1;
        }
        if nal_hrd {
            parse_sub_layer_hrd(r, cpb_cnt.min(32), sub_pic);
        }
        if vcl_hrd {
            parse_sub_layer_hrd(r, cpb_cnt.min(32), sub_pic);
        }
    }
}

fn parse_vui(r: &mut BitReader, max_sub_layers_minus1: u32) -> Vui {
    let mut vui = Vui::default();
    if r.flag() {
        let idc = r.bits(8);
        if idc == 255 {
            r.bits(16);
            r.bits(16);
        }
    }
    if r.flag() {
        r.flag(); // overscan_appropriate
    }
    if r.flag() {
        r.bits(3);
        vui.full_range = r.flag();
        if r.flag() {
            let p = r.bits(8) as u8;
            let t = r.bits(8) as u8;
            let m = r.bits(8) as u8;
            vui.colour_description = Some((p, t, m));
        }
    }
    if r.flag() {
        r.ue();
        r.ue();
    }
    r.flag(); // neutral_chroma_indication
    r.flag(); // field_seq
    r.flag(); // frame_field_info_present
    if r.flag() {
        let l = r.ue();
        let rr = r.ue();
        let t = r.ue();
        let b = r.ue();
        vui.default_display_window = (l, rr, t, b);
    }
    if r.flag() {
        let n = r.bits(32);
        let t = r.bits(32);
        vui.timing = Some((n, t));
        if r.flag() {
            r.ue(); // num_ticks_poc_diff_one_minus1
        }
        if r.flag() {
            parse_hrd(r, true, max_sub_layers_minus1);
        }
    }
    if r.flag() {
        // bitstream_restriction
        r.flag();
        r.flag();
        r.flag();
        r.ue();
        r.ue();
        r.ue();
        r.ue();
        r.ue();
    }
    vui
}

/// A parsed SPS.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sps {
    /// `sps_seq_parameter_set_id`.
    pub id: u32,
    /// `sps_video_parameter_set_id`.
    pub vps_id: u32,
    /// `sps_max_sub_layers_minus1 + 1`.
    pub max_sub_layers: u32,
    /// Profile/tier/level.
    pub ptl: ProfileTierLevel,
    /// `chroma_format_idc`.
    pub chroma_format_idc: u32,
    /// `separate_colour_plane_flag`.
    pub separate_colour_plane: bool,
    /// `pic_width_in_luma_samples`.
    pub width: u32,
    /// `pic_height_in_luma_samples`.
    pub height: u32,
    /// Conformance window in luma samples (left, right, top, bottom).
    pub conf_win: (u32, u32, u32, u32),
    /// `bit_depth_luma_minus8 + 8`.
    pub bit_depth_luma: u32,
    /// `bit_depth_chroma_minus8 + 8`.
    pub bit_depth_chroma: u32,
    /// `log2_max_pic_order_cnt_lsb_minus4 + 4`.
    pub log2_max_poc_lsb: u32,
    /// Per sub-layer (highest used): `sps_max_dec_pic_buffering_minus1 + 1`.
    pub max_dec_pic_buffering: u32,
    /// `sps_max_num_reorder_pics` of the highest sub-layer.
    pub max_num_reorder_pics: u32,
    /// `sps_max_latency_increase_plus1` of the highest sub-layer.
    pub max_latency_increase_plus1: u32,
    /// `log2_min_luma_coding_block_size_minus3 + 3`.
    pub log2_min_cb_size: u32,
    /// `log2_min_cb_size + log2_diff_max_min_luma_coding_block_size` (= CtbLog2SizeY).
    pub log2_ctb_size: u32,
    /// `log2_min_luma_transform_block_size_minus2 + 2`.
    pub log2_min_tb_size: u32,
    /// Max transform block size log2.
    pub log2_max_tb_size: u32,
    /// `max_transform_hierarchy_depth_inter`.
    pub max_th_depth_inter: u32,
    /// `max_transform_hierarchy_depth_intra`.
    pub max_th_depth_intra: u32,
    /// `scaling_list_enabled_flag`.
    pub scaling_list_enabled: bool,
    /// The SPS scaling lists (default when enabled without data).
    pub scaling_list: Option<ScalingList>,
    /// `amp_enabled_flag`.
    pub amp_enabled: bool,
    /// `sample_adaptive_offset_enabled_flag`.
    pub sao_enabled: bool,
    /// `pcm_enabled_flag`.
    pub pcm_enabled: bool,
    /// PCM: `(bit_depth_luma, bit_depth_chroma, log2_min_pcm_cb, log2_max_pcm_cb, loop_filter_disabled)`.
    pub pcm: (u32, u32, u32, u32, bool),
    /// The short-term RPS candidates.
    pub st_rps: Vec<StRps>,
    /// `long_term_ref_pics_present_flag`.
    pub long_term_ref_pics_present: bool,
    /// `(lt_ref_pic_poc_lsb_sps, used_by_curr_pic_lt_sps_flag)`.
    pub lt_ref_pics: Vec<(u32, bool)>,
    /// `sps_temporal_mvp_enabled_flag`.
    pub temporal_mvp_enabled: bool,
    /// `strong_intra_smoothing_enabled_flag`.
    pub strong_intra_smoothing: bool,
    /// VUI, when present.
    pub vui: Option<Vui>,
    /// Range extension flags: `(transform_skip_rotation, transform_skip_context,
    /// implicit_rdpcm, explicit_rdpcm, extended_precision, intra_smoothing_disabled,
    /// high_precision_offsets, persistent_rice_adaptation, cabac_bypass_alignment)`.
    pub range_ext: Option<[bool; 9]>,
}

impl Sps {
    /// `ChromaArrayType`: the chroma format, or 0 when the colour planes
    /// are coded separately.
    pub fn chroma_array_type(&self) -> u32 {
        if self.separate_colour_plane {
            0
        } else {
            self.chroma_format_idc
        }
    }

    /// `(SubWidthC, SubHeightC)`.
    pub fn sub_wh(&self) -> (usize, usize) {
        match self.chroma_array_type() {
            1 => (2, 2),
            2 => (2, 1),
            _ => (1, 1),
        }
    }

    /// The picture's chroma format.
    pub fn chroma_format(&self) -> crate::picture::ChromaFormat {
        use crate::picture::ChromaFormat;
        match self.chroma_array_type() {
            0 => ChromaFormat::Monochrome,
            1 => ChromaFormat::Yuv420,
            2 => ChromaFormat::Yuv422,
            _ => ChromaFormat::Yuv444,
        }
    }

    /// `MinCbSizeY`.
    pub fn min_cb_size(&self) -> u32 {
        1 << self.log2_min_cb_size
    }
    /// `CtbSizeY`.
    pub fn ctb_size(&self) -> u32 {
        1 << self.log2_ctb_size
    }
    /// `PicWidthInCtbsY`.
    pub fn pic_width_in_ctbs(&self) -> u32 {
        self.width.div_ceil(self.ctb_size())
    }
    /// `PicHeightInCtbsY`.
    pub fn pic_height_in_ctbs(&self) -> u32 {
        self.height.div_ceil(self.ctb_size())
    }
    /// `PicWidthInMinCbsY`.
    pub fn pic_width_in_min_cbs(&self) -> u32 {
        self.width / self.min_cb_size()
    }
    /// `PicHeightInMinCbsY`.
    pub fn pic_height_in_min_cbs(&self) -> u32 {
        self.height / self.min_cb_size()
    }
    /// `(SubWidthC, SubHeightC)`.
    pub fn chroma_subsampling(&self) -> (u32, u32) {
        match self.chroma_format_idc {
            1 => (2, 2),
            2 => (2, 1),
            _ => (1, 1),
        }
    }
    /// `MaxPicOrderCntLsb`.
    pub fn max_poc_lsb(&self) -> i32 {
        1 << self.log2_max_poc_lsb
    }

    /// Parse an SPS RBSP.
    pub fn parse(rbsp: &[u8]) -> Result<Sps> {
        let mut r = BitReader::new(rbsp);
        let vps_id = r.bits(4);
        let max_sub_layers_minus1 = r.bits(3);
        if max_sub_layers_minus1 > 6 {
            return Err(Error::bitstream("sps_max_sub_layers_minus1 out of range"));
        }
        r.flag(); // temporal_id_nesting
        let ptl = parse_ptl(&mut r, true, max_sub_layers_minus1);
        let id = r.ue();
        if id > 15 {
            return Err(Error::bitstream("sps_seq_parameter_set_id out of range"));
        }
        let chroma_format_idc = r.ue();
        if chroma_format_idc > 3 {
            return Err(Error::bitstream("chroma_format_idc out of range"));
        }
        let mut separate_colour_plane = false;
        if chroma_format_idc == 3 {
            separate_colour_plane = r.flag();
        }
        let width = r.ue();
        let height = r.ue();
        if width == 0 || height == 0 || width > 16888 || height > 16888 {
            return Err(Error::bitstream("picture size out of range"));
        }
        let mut conf_win = (0, 0, 0, 0);
        if r.flag() {
            let (sw, sh) = match chroma_format_idc {
                1 => (2, 2),
                2 => (2, 1),
                _ => (1, 1),
            };
            let l = r.ue();
            let rr = r.ue();
            let t = r.ue();
            let b = r.ue();
            conf_win = (l * sw, rr * sw, t * sh, b * sh);
        }
        let bit_depth_luma = r.ue() + 8;
        let bit_depth_chroma = r.ue() + 8;
        if bit_depth_luma > 16 || bit_depth_chroma > 16 {
            return Err(Error::bitstream("bit depth out of range"));
        }
        let log2_max_poc_lsb = r.ue() + 4;
        if log2_max_poc_lsb > 16 {
            return Err(Error::bitstream("log2_max_pic_order_cnt_lsb out of range"));
        }
        let sub_layer_ordering_info_present = r.flag();
        let start = if sub_layer_ordering_info_present { 0 } else { max_sub_layers_minus1 };
        let mut max_dec_pic_buffering = 1;
        let mut max_num_reorder_pics = 0;
        let mut max_latency_increase_plus1 = 0;
        for _ in start..=max_sub_layers_minus1 {
            max_dec_pic_buffering = r.ue() + 1;
            max_num_reorder_pics = r.ue();
            max_latency_increase_plus1 = r.ue();
        }
        let log2_min_cb_size = r.ue() + 3;
        let log2_ctb_size = log2_min_cb_size + r.ue();
        let log2_min_tb_size = r.ue() + 2;
        let log2_max_tb_size = log2_min_tb_size + r.ue();
        if log2_ctb_size > 6 || log2_ctb_size < 4 || log2_min_cb_size > log2_ctb_size {
            return Err(Error::bitstream("coding block sizes out of range"));
        }
        if log2_max_tb_size > 5 || log2_min_tb_size >= log2_min_cb_size || log2_max_tb_size > log2_ctb_size {
            return Err(Error::bitstream("transform block sizes out of range"));
        }
        let max_th_depth_inter = r.ue();
        let max_th_depth_intra = r.ue();
        let scaling_list_enabled = r.flag();
        let mut scaling_list = None;
        if scaling_list_enabled {
            if r.flag() {
                scaling_list = Some(parse_scaling_list_data(&mut r)?);
            } else {
                scaling_list = Some(ScalingList::default_lists());
            }
        }
        let amp_enabled = r.flag();
        let sao_enabled = r.flag();
        let pcm_enabled = r.flag();
        let mut pcm = (8, 8, 3, 3, false);
        if pcm_enabled {
            let bl = r.bits(4) + 1;
            let bc = r.bits(4) + 1;
            let lmin = r.ue() + 3;
            let lmax = lmin + r.ue();
            let lf = r.flag();
            pcm = (bl, bc, lmin, lmax, lf);
        }
        let num_st = r.ue() as usize;
        if num_st > 64 {
            return Err(Error::bitstream("num_short_term_ref_pic_sets out of range"));
        }
        let mut st_rps: Vec<StRps> = Vec::with_capacity(num_st);
        for i in 0..num_st {
            let s = parse_st_rps(&mut r, i, num_st, &st_rps)?;
            st_rps.push(s);
        }
        let long_term_ref_pics_present = r.flag();
        let mut lt_ref_pics = Vec::new();
        if long_term_ref_pics_present {
            let n = r.ue();
            if n > 32 {
                return Err(Error::bitstream("num_long_term_ref_pics_sps out of range"));
            }
            for _ in 0..n {
                let lsb = r.bits(log2_max_poc_lsb);
                let used = r.flag();
                lt_ref_pics.push((lsb, used));
            }
        }
        let temporal_mvp_enabled = r.flag();
        let strong_intra_smoothing = r.flag();
        let vui = if r.flag() { Some(parse_vui(&mut r, max_sub_layers_minus1)) } else { None };
        let mut range_ext = None;
        if r.flag() {
            // sps_extension_present_flag
            let range = r.flag();
            let _multilayer = r.flag();
            let _3d = r.flag();
            let _scc = r.flag();
            r.bits(4); // sps_extension_4bits
            if range {
                let mut f = [false; 9];
                for x in f.iter_mut() {
                    *x = r.flag();
                }
                range_ext = Some(f);
            }
            // Other extensions are not parsed; the flags above suffice to
            // refuse them.
        }
        r.finish("SPS")?;
        Ok(Sps {
            id,
            vps_id,
            max_sub_layers: max_sub_layers_minus1 + 1,
            ptl,
            chroma_format_idc,
            separate_colour_plane,
            width,
            height,
            conf_win,
            bit_depth_luma,
            bit_depth_chroma,
            log2_max_poc_lsb,
            max_dec_pic_buffering,
            max_num_reorder_pics,
            max_latency_increase_plus1,
            log2_min_cb_size,
            log2_ctb_size,
            log2_min_tb_size,
            log2_max_tb_size,
            max_th_depth_inter,
            max_th_depth_intra,
            scaling_list_enabled,
            scaling_list,
            amp_enabled,
            sao_enabled,
            pcm_enabled,
            pcm,
            st_rps,
            long_term_ref_pics_present,
            lt_ref_pics,
            temporal_mvp_enabled,
            strong_intra_smoothing,
            vui,
            range_ext,
        })
    }
}
