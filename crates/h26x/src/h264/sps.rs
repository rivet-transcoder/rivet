//! Sequence parameter set (H.264 clause 7.3.2.1 / 7.4.2.1).

use crate::bitreader::BitReader;
use crate::{Error, Result};

use super::tables::{DEFAULT_SCALING4, DEFAULT_SCALING8, ZIGZAG4X4, ZIGZAG8X8};

/// The scaling lists a parameter set carries, in raster order, ready for
/// the dequantiser: six 4x4 lists (Y/Cb/Cr intra, Y/Cb/Cr inter) and up to
/// six 8x8 lists (Y intra, Y inter, and for 4:4:4 Cb/Cr intra/inter).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScalingLists {
    /// `ScalingList4x4[i][raster]`.
    pub list4x4: [[u8; 16]; 6],
    /// `ScalingList8x8[i][raster]` (index 0: Y intra, 1: Y inter, 2: Cb intra,
    /// 3: Cb inter, 4: Cr intra, 5: Cr inter).
    pub list8x8: [[u8; 64]; 6],
}

impl ScalingLists {
    /// Flat_4x4_16 / Flat_8x8_16 everywhere.
    pub fn flat() -> Self {
        Self { list4x4: [[16; 16]; 6], list8x8: [[16; 64]; 6] }
    }

    /// The default lists (Tables 7-3 and 7-4).
    pub fn default_lists() -> Self {
        let mut s = Self::flat();
        for i in 0..6 {
            s.list4x4[i] = DEFAULT_SCALING4[if i < 3 { 0 } else { 1 }];
            s.list8x8[i] = DEFAULT_SCALING8[i & 1];
        }
        s
    }
}

/// VUI fields the decoder itself uses (the rest are parsed and skipped so the
/// bit position stays right).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Vui {
    /// `bitstream_restriction_flag` was present.
    pub bitstream_restriction: bool,
    /// `max_num_reorder_frames`, when signalled.
    pub max_num_reorder_frames: Option<u32>,
    /// `max_dec_frame_buffering`, when signalled.
    pub max_dec_frame_buffering: Option<u32>,
    /// `video_full_range_flag`.
    pub full_range: bool,
    /// `colour_primaries`, `transfer_characteristics`, `matrix_coefficients`
    /// when `colour_description_present_flag`.
    pub colour_description: Option<(u8, u8, u8)>,
    /// `(num_units_in_tick, time_scale)` when timing_info_present.
    pub timing: Option<(u32, u32)>,
    /// `fixed_frame_rate_flag`.
    pub fixed_frame_rate: bool,
}

/// A parsed SPS.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sps {
    /// `profile_idc`.
    pub profile_idc: u8,
    /// The six constraint_set flags packed as bits 7..2 (constraint_set0 in
    /// bit 7).
    pub constraint_flags: u8,
    /// `level_idc`.
    pub level_idc: u8,
    /// `seq_parameter_set_id` (0..=31).
    pub id: u32,
    /// `chroma_format_idc` (1 unless a High profile says otherwise).
    pub chroma_format_idc: u32,
    /// `separate_colour_plane_flag` (4:4:4 only).
    pub separate_colour_plane: bool,
    /// `bit_depth_luma_minus8 + 8`.
    pub bit_depth_luma: u32,
    /// `bit_depth_chroma_minus8 + 8`.
    pub bit_depth_chroma: u32,
    /// `qpprime_y_zero_transform_bypass_flag`.
    pub transform_bypass: bool,
    /// The scaling lists in effect at the sequence level (`None` when the
    /// SPS sends none: Flat_16 unless the PPS overrides).
    pub scaling_lists: Option<ScalingLists>,
    /// `log2_max_frame_num_minus4 + 4`.
    pub log2_max_frame_num: u32,
    /// `pic_order_cnt_type` (0, 1 or 2).
    pub poc_type: u32,
    /// `log2_max_pic_order_cnt_lsb_minus4 + 4` (POC type 0).
    pub log2_max_poc_lsb: u32,
    /// `delta_pic_order_always_zero_flag` (POC type 1).
    pub delta_pic_order_always_zero: bool,
    /// `offset_for_non_ref_pic` (POC type 1).
    pub offset_for_non_ref_pic: i32,
    /// `offset_for_top_to_bottom_field` (POC type 1).
    pub offset_for_top_to_bottom_field: i32,
    /// `offset_for_ref_frame[]` (POC type 1).
    pub offset_for_ref_frame: Vec<i32>,
    /// `max_num_ref_frames`.
    pub max_num_ref_frames: u32,
    /// `gaps_in_frame_num_value_allowed_flag`.
    pub gaps_in_frame_num_allowed: bool,
    /// `pic_width_in_mbs_minus1 + 1`.
    pub pic_width_in_mbs: u32,
    /// `pic_height_in_map_units_minus1 + 1`.
    pub pic_height_in_map_units: u32,
    /// `frame_mbs_only_flag`.
    pub frame_mbs_only: bool,
    /// `mb_adaptive_frame_field_flag`.
    pub mb_adaptive_frame_field: bool,
    /// `direct_8x8_inference_flag`.
    pub direct_8x8_inference: bool,
    /// Frame cropping offsets in *luma samples* (already scaled by the
    /// chroma subsampling / field factors): left, right, top, bottom.
    pub crop: (u32, u32, u32, u32),
    /// VUI parameters, when present.
    pub vui: Option<Vui>,
}

impl Sps {
    /// `PicHeightInMbs` for a frame.
    pub fn frame_height_in_mbs(&self) -> u32 {
        self.pic_height_in_map_units * if self.frame_mbs_only { 1 } else { 2 }
    }
    /// Luma width in samples (uncropped).
    pub fn width(&self) -> u32 {
        self.pic_width_in_mbs * 16
    }
    /// Luma height in samples (uncropped, frame).
    pub fn height(&self) -> u32 {
        self.frame_height_in_mbs() * 16
    }
    /// `MaxFrameNum`.
    pub fn max_frame_num(&self) -> u32 {
        1 << self.log2_max_frame_num
    }
    /// `MaxPicOrderCntLsb`.
    pub fn max_poc_lsb(&self) -> u32 {
        1 << self.log2_max_poc_lsb
    }
    /// The (SubWidthC, SubHeightC) of the chroma format.
    pub fn chroma_subsampling(&self) -> (u32, u32) {
        match self.chroma_format_idc {
            0 => (1, 1),
            1 => (2, 2),
            2 => (2, 1),
            _ => (1, 1),
        }
    }
    /// The DPB size in frames the level allows for this picture size
    /// (Table A-1 `MaxDpbMbs`), capped at 16.
    pub fn level_max_dpb_frames(&self) -> u32 {
        let max_dpb_mbs: u32 = match self.level_idc {
            9 => 396,
            10 => 396,
            11 => {
                if self.constraint_flags & (1 << 4) != 0 && self.profile_idc != 100 {
                    // level 1b via constraint_set3 on Baseline/Main/Extended
                    396
                } else {
                    900
                }
            }
            12 => 2376,
            13 => 2376,
            20 => 2376,
            21 => 4752,
            22 => 8100,
            30 => 8100,
            31 => 18000,
            32 => 20480,
            40 => 32768,
            41 => 32768,
            42 => 34816,
            50 => 110400,
            51 => 184320,
            52 => 184320,
            _ => 184320,
        };
        let frame_mbs = self.pic_width_in_mbs * self.frame_height_in_mbs();
        (max_dpb_mbs / frame_mbs.max(1)).clamp(1, 16)
    }
}

/// Parse `scaling_list()` (7.3.2.1.1.1) into `out` in **raster** order.
/// Returns whether the "use default" fallback (rule B) was signalled.
fn parse_scaling_list(r: &mut BitReader, out: &mut [u8], scan: &[u8]) -> bool {
    let size = out.len();
    let mut last: i32 = 8;
    let mut next: i32 = 8;
    let mut use_default = false;
    for j in 0..size {
        if next != 0 {
            let delta = r.se();
            next = (last + delta + 256) % 256;
            use_default = j == 0 && next == 0;
        }
        let v = if next == 0 { last } else { next };
        out[scan[j] as usize] = v as u8;
        last = v;
    }
    use_default
}

/// Parse the scaling matrices of an SPS or PPS (7.3.2.1.1 / 7.3.2.2), with
/// the fall-back rules of Table 7-2. `fallback` is what a list not present in
/// the bitstream falls back to for the *first* list of each size (rule A:
/// the defaults for an SPS, the SPS lists for a PPS); later lists fall back
/// to the previous list of the same kind (rule B). `count8x8` is how many 8x8
/// lists to read (2, or 6 for 4:4:4).
pub(crate) fn parse_scaling_matrix(
    r: &mut BitReader,
    fallback: &ScalingLists,
    count8x8: usize,
) -> ScalingLists {
    let mut lists = fallback.clone();
    let mut tmp4 = [0u8; 16];
    let mut tmp8 = [0u8; 64];
    for i in 0..6 {
        let present = r.flag();
        if present {
            let use_default = parse_scaling_list(r, &mut tmp4, &ZIGZAG4X4);
            lists.list4x4[i] = if use_default { DEFAULT_SCALING4[if i < 3 { 0 } else { 1 }] } else { tmp4 };
        } else {
            // Rule A for i == 0 and 3, rule B otherwise.
            lists.list4x4[i] = match i {
                0 | 3 => fallback.list4x4[i],
                _ => lists.list4x4[i - 1],
            };
        }
    }
    for i in 0..count8x8 {
        let present = r.flag();
        if present {
            let use_default = parse_scaling_list(r, &mut tmp8, &ZIGZAG8X8);
            lists.list8x8[i] = if use_default { DEFAULT_SCALING8[i & 1] } else { tmp8 };
        } else {
            lists.list8x8[i] = match i {
                0 | 1 => fallback.list8x8[i],
                _ => lists.list8x8[i - 2],
            };
        }
    }
    lists
}

fn parse_hrd(r: &mut BitReader) {
    let cpb_cnt = r.ue() + 1;
    r.bits(4); // bit_rate_scale
    r.bits(4); // cpb_size_scale
    for _ in 0..cpb_cnt.min(32) {
        r.ue(); // bit_rate_value_minus1
        r.ue(); // cpb_size_value_minus1
        r.flag(); // cbr_flag
    }
    r.bits(5); // initial_cpb_removal_delay_length_minus1
    r.bits(5); // cpb_removal_delay_length_minus1
    r.bits(5); // dpb_output_delay_length_minus1
    r.bits(5); // time_offset_length
}

fn parse_vui(r: &mut BitReader) -> Vui {
    let mut vui = Vui::default();
    if r.flag() {
        // aspect_ratio_info_present_flag
        let idc = r.bits(8);
        if idc == 255 {
            r.bits(16);
            r.bits(16);
        }
    }
    if r.flag() {
        // overscan_info_present_flag
        r.flag();
    }
    if r.flag() {
        // video_signal_type_present_flag
        r.bits(3); // video_format
        vui.full_range = r.flag();
        if r.flag() {
            let p = r.bits(8) as u8;
            let t = r.bits(8) as u8;
            let m = r.bits(8) as u8;
            vui.colour_description = Some((p, t, m));
        }
    }
    if r.flag() {
        // chroma_loc_info_present_flag
        r.ue();
        r.ue();
    }
    if r.flag() {
        // timing_info_present_flag
        let num_units = r.bits(32);
        let time_scale = r.bits(32);
        vui.timing = Some((num_units, time_scale));
        vui.fixed_frame_rate = r.flag();
    }
    let nal_hrd = r.flag();
    if nal_hrd {
        parse_hrd(r);
    }
    let vcl_hrd = r.flag();
    if vcl_hrd {
        parse_hrd(r);
    }
    if nal_hrd || vcl_hrd {
        r.flag(); // low_delay_hrd_flag
    }
    r.flag(); // pic_struct_present_flag
    if r.flag() {
        // bitstream_restriction_flag
        vui.bitstream_restriction = true;
        r.flag(); // motion_vectors_over_pic_boundaries_flag
        r.ue(); // max_bytes_per_pic_denom
        r.ue(); // max_bits_per_mb_denom
        r.ue(); // log2_max_mv_length_horizontal
        r.ue(); // log2_max_mv_length_vertical
        vui.max_num_reorder_frames = Some(r.ue());
        vui.max_dec_frame_buffering = Some(r.ue());
    }
    vui
}

impl Sps {
    /// Parse an SPS RBSP (the bytes after the NAL header, emulation
    /// prevention removed).
    pub fn parse(rbsp: &[u8]) -> Result<Sps> {
        let mut r = BitReader::new(rbsp);
        let profile_idc = r.bits(8) as u8;
        let constraint_flags = r.bits(8) as u8;
        let level_idc = r.bits(8) as u8;
        let id = r.ue();
        if id > 31 {
            return Err(Error::bitstream(format!("SPS: seq_parameter_set_id {id} out of range")));
        }

        let mut chroma_format_idc = 1;
        let mut separate_colour_plane = false;
        let mut bit_depth_luma = 8;
        let mut bit_depth_chroma = 8;
        let mut transform_bypass = false;
        let mut scaling_lists = None;
        if matches!(profile_idc, 100 | 110 | 122 | 244 | 44 | 83 | 86 | 118 | 128 | 138 | 139 | 134 | 135) {
            chroma_format_idc = r.ue();
            if chroma_format_idc > 3 {
                return Err(Error::bitstream("SPS: chroma_format_idc out of range"));
            }
            if chroma_format_idc == 3 {
                separate_colour_plane = r.flag();
            }
            bit_depth_luma = r.ue() + 8;
            bit_depth_chroma = r.ue() + 8;
            if bit_depth_luma > 14 || bit_depth_chroma > 14 {
                return Err(Error::bitstream("SPS: bit depth out of range"));
            }
            transform_bypass = r.flag();
            if r.flag() {
                // seq_scaling_matrix_present_flag: rule A falls back to the
                // default lists.
                let count8x8 = if chroma_format_idc != 3 { 2 } else { 6 };
                let defaults = ScalingLists::default_lists();
                scaling_lists = Some(parse_scaling_matrix(&mut r, &defaults, count8x8));
            }
        }

        let log2_max_frame_num = r.ue() + 4;
        if log2_max_frame_num > 16 {
            return Err(Error::bitstream("SPS: log2_max_frame_num out of range"));
        }
        let poc_type = r.ue();
        let mut log2_max_poc_lsb = 0;
        let mut delta_pic_order_always_zero = false;
        let mut offset_for_non_ref_pic = 0;
        let mut offset_for_top_to_bottom_field = 0;
        let mut offset_for_ref_frame = Vec::new();
        match poc_type {
            0 => {
                log2_max_poc_lsb = r.ue() + 4;
                if log2_max_poc_lsb > 16 {
                    return Err(Error::bitstream("SPS: log2_max_pic_order_cnt_lsb out of range"));
                }
            }
            1 => {
                delta_pic_order_always_zero = r.flag();
                offset_for_non_ref_pic = r.se();
                offset_for_top_to_bottom_field = r.se();
                let n = r.ue();
                if n > 255 {
                    return Err(Error::bitstream("SPS: num_ref_frames_in_pic_order_cnt_cycle out of range"));
                }
                for _ in 0..n {
                    offset_for_ref_frame.push(r.se());
                }
            }
            2 => {}
            _ => return Err(Error::bitstream("SPS: pic_order_cnt_type out of range")),
        }
        let max_num_ref_frames = r.ue();
        let gaps_in_frame_num_allowed = r.flag();
        let pic_width_in_mbs = r.ue() + 1;
        let pic_height_in_map_units = r.ue() + 1;
        if pic_width_in_mbs > 1024 || pic_height_in_map_units > 1024 {
            return Err(Error::bitstream("SPS: picture size out of range"));
        }
        let frame_mbs_only = r.flag();
        let mut mb_adaptive_frame_field = false;
        if !frame_mbs_only {
            mb_adaptive_frame_field = r.flag();
        }
        let direct_8x8_inference = r.flag();
        let mut crop = (0, 0, 0, 0);
        if r.flag() {
            let (sub_w, sub_h) = match chroma_format_idc {
                0 => (1, 1),
                1 => (2, 2),
                2 => (2, 1),
                _ => (1, 1),
            };
            let crop_unit_x = if chroma_format_idc == 0 { 1 } else { sub_w };
            let crop_unit_y = (if chroma_format_idc == 0 { 1 } else { sub_h }) * if frame_mbs_only { 1 } else { 2 };
            let l = r.ue();
            let rr = r.ue();
            let t = r.ue();
            let b = r.ue();
            crop = (l * crop_unit_x, rr * crop_unit_x, t * crop_unit_y, b * crop_unit_y);
        }
        let vui = if r.flag() { Some(parse_vui(&mut r)) } else { None };
        r.finish("SPS")?;

        let sps = Sps {
            profile_idc,
            constraint_flags,
            level_idc,
            id,
            chroma_format_idc,
            separate_colour_plane,
            bit_depth_luma,
            bit_depth_chroma,
            transform_bypass,
            scaling_lists,
            log2_max_frame_num,
            poc_type,
            log2_max_poc_lsb,
            delta_pic_order_always_zero,
            offset_for_non_ref_pic,
            offset_for_top_to_bottom_field,
            offset_for_ref_frame,
            max_num_ref_frames,
            gaps_in_frame_num_allowed,
            pic_width_in_mbs,
            pic_height_in_map_units,
            frame_mbs_only,
            mb_adaptive_frame_field,
            direct_8x8_inference,
            crop,
            vui,
        };
        if sps.crop.0 + sps.crop.1 >= sps.width() || sps.crop.2 + sps.crop.3 >= sps.height() {
            return Err(Error::bitstream("SPS: cropping window larger than the picture"));
        }
        Ok(sps)
    }
}
