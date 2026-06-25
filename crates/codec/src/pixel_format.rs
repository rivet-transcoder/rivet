//! Pixel-format detection from codec sequence headers.
//!
//! Given raw bitstream samples (the same Vec<Vec<u8>> our decoders
//! consume), parse just enough of the first sequence header to
//! extract chroma subsampling + luma bit depth, then map to our
//! PixelFormat enum.
//!
//! Why not use the full decoder: our CPU decoders (H.264 openh264,
//! HEVC Rust, VP9 Rust, rav1d AV1) each have their own parser
//! entry points, but none of them expose a "just probe the format"
//! API. NVDEC's sequence_callback tells us, but only after decode
//! starts. This module gives the pipeline a fast, codec-agnostic
//! probe path that runs before decoder construction.

use crate::frame::PixelFormat;

/// Detect pixel format from the first sequence header in `samples`.
/// Falls back to Yuv420p on any parse failure — that matches the
/// previous hard-coded behavior so a bad probe doesn't block the
/// transcode, just the probe payload accuracy.
pub fn detect(codec: &str, samples: &[Vec<u8>]) -> PixelFormat {
    if samples.is_empty() {
        return PixelFormat::Yuv420p;
    }

    let result = match codec.to_lowercase().as_str() {
        "h264" | "avc1" | "avc" => detect_h264(&samples[0]),
        "h265" | "hevc" | "hvc1" | "hev1" => detect_hevc(&samples[0]),
        "vp9" | "vp09" => detect_vp9(&samples[0]),
        "av1" | "av01" => detect_av1(&samples[0]),
        _ => None,
    };

    result.unwrap_or(PixelFormat::Yuv420p)
}

// ─── Bit reader ────────────────────────────────────────────────────
struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn read_bits(&mut self, n: usize) -> Option<u32> {
        let mut val = 0u32;
        for _ in 0..n {
            let byte_idx = self.pos / 8;
            let bit_idx = 7 - (self.pos % 8);
            if byte_idx >= self.data.len() {
                return None;
            }
            val = (val << 1) | (((self.data[byte_idx] >> bit_idx) & 1) as u32);
            self.pos += 1;
        }
        Some(val)
    }

    /// Exp-Golomb unsigned — used by H.264 and HEVC SPS fields.
    fn read_ue(&mut self) -> Option<u32> {
        let mut zeros = 0;
        while self.read_bits(1)? == 0 {
            zeros += 1;
            if zeros > 31 {
                // Cap before `1u32 << 32` would panic. 31 zeros already
                // allow any value up to u32::MAX; any SPS field we care
                // about fits within ~10 zeros.
                return None;
            }
        }
        if zeros == 0 {
            return Some(0);
        }
        let suffix = self.read_bits(zeros)?;
        Some((1u32 << zeros) - 1 + suffix)
    }

    /// Signed Exp-Golomb (se(v)). H.264 §9.1.1: `codeNum` from `read_ue`,
    /// then `(-1)^(codeNum+1) * ceil(codeNum/2)` — odd codeNums map to
    /// positive values, even to negative (or zero for codeNum=0).
    /// Used by H.264 SPS `scaling_list` deltas and `pic_order_cnt_type==1`
    /// offsets.
    fn read_se(&mut self) -> Option<i32> {
        let code = self.read_ue()? as i64;
        let signed = if code & 1 == 1 {
            ((code + 1) / 2) as i32
        } else {
            -((code / 2) as i32)
        };
        Some(signed)
    }

    /// Current bit position within `data`. Used by AV1 parsers to find
    /// the byte-aligned end of uncompressed_header().
    fn bit_pos(&self) -> usize {
        self.pos
    }

    /// Advance the bit cursor to the next byte boundary. AV1 spec
    /// byte_alignment() per §5.3.5: skip bits until `pos % 8 == 0`.
    fn byte_align(&mut self) {
        let rem = self.pos & 7;
        if rem != 0 {
            self.pos += 8 - rem;
        }
    }

    /// AV1 signed `su(n)` — n-bit two's-complement signed integer
    /// (§4.10.5). Read n bits, sign-extend from bit n-1.
    fn read_su(&mut self, n: usize) -> Option<i32> {
        let raw = self.read_bits(n)?;
        let sign_bit = 1u32 << (n - 1);
        let signed = if raw & sign_bit != 0 {
            (raw as i32) - (1i32 << n)
        } else {
            raw as i32
        };
        Some(signed)
    }
}

// ─── H.264 SPS parser ─────────────────────────────────────────────
// See ITU-T H.264 §7.3.2.1.1. Profile-gated fields: only profile_idc
// values in { 100, 110, 122, 244, 44, 83, 86, 118, 128, 138, 139,
// 134, 135 } carry the chroma_format_idc + bit_depth fields we want.
// Everything else is 4:2:0 8-bit by spec.
fn detect_h264(sample: &[u8]) -> Option<PixelFormat> {
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

fn find_next_start_code(data: &[u8]) -> Option<usize> {
    (0..data.len().saturating_sub(3)).find(|&i| {
        data[i] == 0
            && data[i + 1] == 0
            && (data[i + 2] == 1 || (data[i + 2] == 0 && data[i + 3] == 1))
    })
}

/// Strip H.264 / HEVC emulation-prevention bytes (0x00 0x00 0x03 → 0x00 0x00).
fn remove_h264_rbsp_stuffing(sps: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(sps.len());
    let mut i = 0;
    while i < sps.len() {
        if i + 2 < sps.len() && sps[i] == 0 && sps[i + 1] == 0 && sps[i + 2] == 3 {
            out.push(0);
            out.push(0);
            i += 3;
        } else {
            out.push(sps[i]);
            i += 1;
        }
    }
    out
}

// ─── HEVC SPS parser ──────────────────────────────────────────────
// See ITU-T H.265 §7.3.2.2.1. We skip profile_tier_level and jump to
// chroma_format_idc + bit_depth_luma_minus8 + bit_depth_chroma_minus8.
fn detect_hevc(sample: &[u8]) -> Option<PixelFormat> {
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

    Some(PixelFormat::from_chroma_and_depth(
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

fn skip_hevc_profile_tier_level(br: &mut BitReader, max_sub_layers_minus1: usize) -> Option<()> {
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

// ─── VP9 uncompressed header parser ───────────────────────────────
// See VP9 bitstream specification §6.2. We only need profile + bit_depth
// + subsampling_x + subsampling_y from the header.
fn detect_vp9(sample: &[u8]) -> Option<PixelFormat> {
    if sample.len() < 2 {
        return None;
    }
    let mut br = BitReader::new(sample);
    let frame_marker = br.read_bits(2)?;
    if frame_marker != 2 {
        return None;
    }
    let profile_low = br.read_bits(1)?;
    let profile_high = br.read_bits(1)?;
    let profile = (profile_high << 1) | profile_low;
    if profile == 3 {
        let _reserved_zero = br.read_bits(1)?;
    }
    let show_existing_frame = br.read_bits(1)?;
    if show_existing_frame == 1 {
        return None;
    }
    let frame_type = br.read_bits(1)?;
    let _show_frame = br.read_bits(1)?;
    let _error_resilient = br.read_bits(1)?;

    // color_config only appears on keyframes.
    if frame_type != 0 {
        return None;
    }

    // Keyframe sync code: 3 bytes {0x49, 0x83, 0x42}. 24 bits.
    let sync = br.read_bits(24)?;
    if sync != 0x498342 {
        return None;
    }

    let bit_depth = if profile >= 2 {
        if br.read_bits(1)? == 0 { 10 } else { 12 }
    } else {
        8
    };
    let _color_space = br.read_bits(3)?;
    // color_range + subsampling — layout depends on color_space
    // For simplicity: for Profile 0/2 the subsampling is 4:2:0. Profile
    // 1/3 read subsampling_x/y fields to distinguish 4:2:2 vs 4:4:4.
    let (sx, sy) = if profile == 1 || profile == 3 {
        let _color_range = br.read_bits(1)?;
        let sx = br.read_bits(1)?;
        let sy = br.read_bits(1)?;
        (sx, sy)
    } else {
        (1, 1) // 4:2:0
    };

    let chroma_idc = match (sx, sy) {
        (1, 1) => 1, // 4:2:0
        (1, 0) => 2, // 4:2:2
        (0, 0) => 3, // 4:4:4
        _ => 1,
    };

    Some(PixelFormat::from_chroma_and_depth(chroma_idc, bit_depth))
}

// ─── AV1 sequence header parser ───────────────────────────────────
// See AV1 spec §5.5. Full parse is long; we hop through enough fields
// to reach color_config. Most AV1 content in the wild is 4:2:0 8-bit
// (Main profile), and 4:2:0 10-bit for HDR (Main-10).
fn detect_av1(sample: &[u8]) -> Option<PixelFormat> {
    // AV1 wraps sequence headers in an OBU with obu_type == 1.
    let obu = find_av1_obu(sample, 1)?;
    let mut br = BitReader::new(obu);

    let _seq_profile = br.read_bits(3)?;
    let _still_picture = br.read_bits(1)?;
    let reduced_still_picture_header = br.read_bits(1)?;

    if reduced_still_picture_header == 0 {
        // timing_info_present, decoder_model_info, initial_display_delay,
        // operating_points — a lot to skip. Abort safely if any read
        // fails; fallback to Yuv420p.
        let timing_info_present = br.read_bits(1)?;
        if timing_info_present == 1 {
            let _num_units_in_display_tick = br.read_bits(32)?;
            let _time_scale = br.read_bits(32)?;
            let equal_picture_interval = br.read_bits(1)?;
            if equal_picture_interval == 1 {
                let _num_ticks_per_picture = br.read_ue()?; // uvlc, not ue(v), but reuse
            }
            let decoder_model_info_present = br.read_bits(1)?;
            if decoder_model_info_present == 1 {
                let _buffer_delay_length_minus_1 = br.read_bits(5)?;
                let _num_units_in_decoding_tick = br.read_bits(32)?;
                let _buffer_removal_time_length_minus_1 = br.read_bits(5)?;
                let _frame_presentation_time_length_minus_1 = br.read_bits(5)?;
            }
        }
        // Bail out to default — the full operating-points loop is long
        // and rarely worth the maintenance cost vs accepting that
        // non-trivial AV1 probes return Yuv420p for now. If the MP4
        // container advertises codec profile in its track box, we can
        // use that instead (future follow-up if the data shows 10-bit
        // AV1 slipping through).
        return Some(PixelFormat::Yuv420p);
    }

    // Reduced still-picture path is simpler: go straight to
    // seq_level_idx + bit depth fields.
    let _seq_level_idx_0 = br.read_bits(5)?;

    // For full correctness we'd continue into color_config. Since the
    // reduced path is rare for VOD content we take the safe default
    // and let downstream validation surface anything unexpected.
    Some(PixelFormat::Yuv420p)
}

/// Find the first AV1 OBU of the given obu_type. AV1 OBU header:
///   obu_forbidden_bit(1) | obu_type(4) | obu_extension_flag(1)
///   | obu_has_size_field(1) | obu_reserved_1bit(1)
/// followed by an optional 1-byte extension, optional LEB128 size,
/// then payload. For simplicity we require obu_has_size_field=1 which
/// all muxed AV1 satisfies.
fn find_av1_obu(data: &[u8], target_type: u8) -> Option<&[u8]> {
    find_av1_obu_with_offset(data, target_type).map(|(bytes, _)| bytes)
}

/// Public re-export so the Vulkan Video decoder can extract the byte
/// range of an OBU from a demuxed sample.
pub fn find_av1_obu_with_offset_pub(data: &[u8], target_type: u8) -> Option<(&[u8], usize)> {
    find_av1_obu_with_offset(data, target_type)
}

/// Returns the OBU payload slice AND the byte offset at which it
/// starts inside `data`. The offset is what callers need to translate
/// an in-OBU bit/byte position (e.g. tile_group start after
/// byte_alignment()) to an absolute position in the sample buffer.
fn find_av1_obu_with_offset(data: &[u8], target_type: u8) -> Option<(&[u8], usize)> {
    let mut i = 0;
    while i < data.len() {
        let header = data[i];
        let obu_type = (header >> 3) & 0x0F;
        let extension_flag = (header >> 2) & 0x01;
        let has_size_field = (header >> 1) & 0x01;
        i += 1;
        if extension_flag == 1 {
            i += 1;
        }
        if has_size_field == 0 {
            return None;
        }
        let (size, leb_bytes) = read_leb128(&data[i..])?;
        i += leb_bytes;
        if obu_type == target_type {
            let end = (i + size as usize).min(data.len());
            return Some((&data[i..end], i));
        }
        i += size as usize;
    }
    None
}

fn read_leb128(data: &[u8]) -> Option<(u64, usize)> {
    let mut value = 0u64;
    for i in 0..8 {
        if i >= data.len() {
            return None;
        }
        let byte = data[i];
        value |= ((byte & 0x7F) as u64) << (i * 7);
        if byte & 0x80 == 0 {
            return Some((value, i + 1));
        }
    }
    None
}

// ─── Deep sequence-header parse: width / height extraction ────────
//
// The `detect` entry points above stop at chroma_format_idc + bit_depth,
// which is all they need for pixel-format mapping. MPEG-TS can't carry
// width/height at the container layer (no sample-entry atom; SPS is the
// only source), so we need parsers that go deeper — through scaling
// lists, pic_order_cnt_type branches, and frame cropping — to extract
// the displayable width/height.
//
// Consumers: `container::ts` calls `detect_dims` during demux to populate
// `StreamInfo.width` / `.height`; the H.264 decoder's chroma-reject sniff
// (`codec::decode::h264`) uses `parse_h264_sps` to read profile +
// chroma_format_idc in a single pass instead of a second scan.

/// Parsed H.264 SPS fields relevant to the pipeline.
///
/// Populated by `parse_h264_sps` which walks the SPS through the frame
/// cropping offsets per ITU-T H.264 §7.3.2.1.1 and applies the display
/// rectangle derivation of §7.4.2.1.1 + Table 6-1 (SubWidthC /
/// SubHeightC) to produce the post-crop displayable width/height.
///
/// Width/height are `Option<u32>` because the full SPS walk can bail on
/// a malformed scaling list or an exotic `pic_order_cnt_type` branch;
/// `profile_idc` / `chroma_format_idc` are always populated on a
/// successful `Some(_)` return since they live in the SPS prefix before
/// any of the variable-length sections.
#[derive(Debug, Clone, PartialEq, Eq)]
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

/// Parsed HEVC SPS fields relevant to the pipeline.
///
/// Width/height are post-conformance-window (§7.4.3.2.1): per the spec,
/// output luma dimensions = `pic_width_in_luma_samples - SubWidthC *
/// (conf_win_left + conf_win_right)` (and analogously for height).
#[derive(Debug, Clone, PartialEq, Eq)]
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

/// Locate the byte offset, within `sample`, of the uncompressed_header
/// payload of the first Frame OBU (obu_type 3 or 6). Returns None if
/// no such OBU is found.
///
/// AV1 OBU layout: 1-byte header + optional 1-byte extension + LEB128
/// size + payload. For a Frame OBU (type 6), the payload begins with
/// uncompressed_header_obu() — so the byte offset we return is the
/// first byte of uncompressed_header() in the original sample buffer.
/// Vulkan `VkVideoDecodeAV1PictureInfoKHR::frameHeaderOffset` wants
/// exactly this value.
pub fn av1_frame_header_offset(sample: &[u8]) -> Option<u32> {
    let mut i = 0usize;
    while i < sample.len() {
        let header = sample[i];
        let obu_type = (header >> 3) & 0x0F;
        let extension_flag = (header >> 2) & 0x01;
        let has_size_field = (header >> 1) & 0x01;
        let mut p = i + 1;
        if extension_flag == 1 {
            p += 1;
        }
        let (size, leb) = if has_size_field == 1 {
            let (s, n) = read_leb128(&sample[p..])?;
            p += n;
            (s as usize, n)
        } else {
            // OBU has_size_field=0 is legal but we don't handle it
            // (AV1 in MP4 always sets it).
            return None;
        };
        let _ = leb;
        if obu_type == 3 || obu_type == 6 {
            return Some(p as u32);
        }
        p += size;
        i = p;
    }
    None
}

/// Locate the byte offset of the first tile_group_obu payload within
/// the sample buffer, used for
/// `VkVideoDecodeAV1PictureInfoKHR::pTileOffsets`. Two shapes:
/// - Separate Frame Header OBU (type 3) + Tile Group OBU (type 4):
///   return the type-4 OBU payload start.
/// - Frame OBU (type 6) (frame header + tile group in one OBU):
///   return `frame_OBU_payload_start + tile_group_offset_in_obu`
///   where the in-OBU offset comes from `parse_av1_frame_header`
///   (the byte-aligned position after uncompressed_header).
///
/// Returns None when neither shape is found or the parser bails.
pub fn av1_tile_group_offset(sample: &[u8], seq: &Av1SequenceHeader) -> Option<u32> {
    // If a standalone Tile Group OBU (type 4) exists, use its payload
    // start directly — no uncompressed_header to skip past.
    let mut i = 0usize;
    while i < sample.len() {
        let header = sample[i];
        let obu_type = (header >> 3) & 0x0F;
        let extension_flag = (header >> 2) & 0x01;
        let has_size_field = (header >> 1) & 0x01;
        let mut p = i + 1;
        if extension_flag == 1 {
            p += 1;
        }
        let size = if has_size_field == 1 {
            let (s, n) = read_leb128(&sample[p..])?;
            p += n;
            s as usize
        } else {
            return None;
        };
        if obu_type == 4 {
            return Some(p as u32);
        }
        p += size;
        i = p;
    }
    // Frame OBU (type 6): combine the OBU payload start with the
    // in-OBU offset from the parsed frame header.
    let (_obu_bytes, payload_offset) = find_av1_obu_with_offset(sample, 6)?;
    let hdr = parse_av1_frame_header(sample, seq)?;
    Some(payload_offset as u32 + hdr.tile_group_offset_in_obu)
}

/// Backwards-compatible shim — uses an empty-ish sequence header
/// default that only works for the fallback path (standalone type-4
/// OBU). Callers with access to the parsed sequence header should
/// use `av1_tile_group_offset` (the seq-aware form) instead.
pub fn av1_tile_group_offset_fallback(sample: &[u8]) -> Option<u32> {
    let mut i = 0usize;
    while i < sample.len() {
        let header = sample[i];
        let obu_type = (header >> 3) & 0x0F;
        let extension_flag = (header >> 2) & 0x01;
        let has_size_field = (header >> 1) & 0x01;
        let mut p = i + 1;
        if extension_flag == 1 {
            p += 1;
        }
        let size = if has_size_field == 1 {
            let (s, n) = read_leb128(&sample[p..])?;
            p += n;
            s as usize
        } else {
            return None;
        };
        if obu_type == 4 {
            return Some(p as u32);
        }
        p += size;
        i = p;
    }
    av1_frame_header_offset(sample)
}

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

/// Parsed MPEG-2 sequence header + (optional) sequence extension.
///
/// MPEG-2 video §6.2.2.1/§6.2.2.3 (ISO/IEC 13818-2): the 12-bit
/// `horizontal_size_value` / `vertical_size_value` from the sequence
/// header, optionally extended to 14 bits by the 2-bit
/// `horizontal_size_extension` / `vertical_size_extension` fields in a
/// `sequence_extension()` start-code-prefixed NAL. Pure MPEG-1
/// (start code 0xB3 but no 0xB5 extension) stays 12-bit — produces
/// the same 12-bit result via the extension-less path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Mpeg2SeqInfo {
    pub width: u32,
    pub height: u32,
}

/// Public entry point — dispatch by codec and return `Some((width,
/// height))` if the sequence header in `samples[0]` is parseable,
/// `None` otherwise.
///
/// Callers should treat `None` as "keep the existing width/height" —
/// it's load-bearing for MPEG-TS where `StreamInfo` would otherwise
/// carry `0×0`, but a parse failure on MP4/MKV (which already have
/// width/height in the sample-entry / track-header) is a no-op.
pub fn detect_dims(codec: &str, samples: &[Vec<u8>]) -> Option<(u32, u32)> {
    if samples.is_empty() {
        return None;
    }
    let sample = &samples[0];
    match codec.to_lowercase().as_str() {
        "h264" | "avc1" | "avc" | "avc3" => {
            let info = parse_h264_sps(sample)?;
            Some((info.width?, info.height?))
        }
        "h265" | "hevc" | "hvc1" | "hev1" | "hvc2" | "hev2" => {
            let info = parse_hevc_sps(sample)?;
            Some((info.width?, info.height?))
        }
        "mpeg2" | "mpeg2video" | "mp2v" => {
            let info = parse_mpeg2_sequence_header(sample)?;
            Some((info.width, info.height))
        }
        _ => None,
    }
}

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
    let general_profile_space = br.read_bits(2)?;
    let tier_flag = br.read_bits(1)? == 1;
    let profile_idc = br.read_bits(5)? as u8;
    let _ = general_profile_space;
    // general_profile_compatibility_flag[32] — captured for Std PTL.
    let profile_compatibility_flags = br.read_bits(32)?;
    // constraint flags (48 bits) — ignored here; Std PTL has them as
    // individual flag bits we're not reporting (conservative default).
    let _ = br.read_bits(48)?;
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

/// MPEG-2 sequence header scan — ISO/IEC 13818-2 §6.2.2.1 (sequence
/// header, start code `00 00 01 B3`) + §6.2.2.3 (sequence extension,
/// start code `00 00 01 B5` with `extension_start_code_identifier==1`).
///
/// The sequence header carries 12-bit `horizontal_size_value` and
/// `vertical_size_value`, tight for sizes ≤ 4095. The optional sequence
/// extension prepends 2-bit `_extension` fields that, when combined,
/// bring the total to 14 bits (sizes ≤ 16383). Pure MPEG-1 (start code
/// 0xB3 only, no 0xB5) never has the extension and stays 12-bit.
pub fn parse_mpeg2_sequence_header(sample: &[u8]) -> Option<Mpeg2SeqInfo> {
    // Walk bytes looking for 00 00 01 B3 (sequence_header_code). The
    // following 3 bytes carry horizontal(12) + vertical(12).
    let seq_hdr_start = find_mpeg2_start_code(sample, 0xB3)?;
    let hdr_body_off = seq_hdr_start + 4;
    if hdr_body_off + 3 > sample.len() {
        return None;
    }
    let b = &sample[hdr_body_off..hdr_body_off + 3];
    let mut width = (((b[0] as u32) << 4) | ((b[1] as u32) >> 4)) & 0x0FFF;
    let mut height = (((b[1] as u32 & 0x0F) << 8) | (b[2] as u32)) & 0x0FFF;

    // Look for a subsequent sequence_extension that upgrades the 12-bit
    // values to 14-bit. Only scan forward from seq_hdr_start; a
    // sequence_extension before the first sequence_header is
    // nonsensical and we shouldn't confuse the parse.
    let search_from = hdr_body_off + 3;
    if search_from < sample.len()
        && let Some(ext_start) = find_mpeg2_start_code(&sample[search_from..], 0xB5)
    {
        let ext_body_off = search_from + ext_start + 4;
        if ext_body_off + 3 <= sample.len() {
            let mut br = BitReader::new(&sample[ext_body_off..]);
            if let Some(id) = br.read_bits(4)
                && id == 1
            {
                // sequence_extension §6.2.2.3:
                //   extension_start_code_identifier  u(4) = 0001   (already read)
                //   profile_and_level_indication     u(8)
                //   progressive_sequence             u(1)
                //   chroma_format                    u(2)
                //   horizontal_size_extension        u(2)
                //   vertical_size_extension          u(2)
                let _profile_level = br.read_bits(8)?;
                let _progressive = br.read_bits(1)?;
                let _chroma = br.read_bits(2)?;
                let h_ext = br.read_bits(2)?;
                let v_ext = br.read_bits(2)?;
                width |= h_ext << 12;
                height |= v_ext << 12;
            }
        }
    }

    if width == 0 || height == 0 {
        return None;
    }
    Some(Mpeg2SeqInfo { width, height })
}

/// Scan for an MPEG-2 start code (0x00 0x00 0x01 <target>) byte-aligned.
/// Returns the file offset of the leading 0x00 on success.
fn find_mpeg2_start_code(data: &[u8], target: u8) -> Option<usize> {
    let mut i = 0;
    while i + 4 <= data.len() {
        if data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1 && data[i + 3] == target {
            return Some(i);
        }
        i += 1;
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

/// Heuristic more_rbsp_data() check — the spec defines it precisely
/// (position in RBSP trailing bits) but needs byte-alignment awareness
/// we don't expose from BitReader. Approximation: at least one more
/// full byte of input remains after the current cursor. Good enough
/// for the PPS extended-field branch since the trailing byte is a
/// stop bit + zero pad — parsing a spurious bit from that gives
/// `transform_8x8_mode_flag = true` which the caller tolerates.
fn more_rbsp_data(br: &BitReader, rbsp: &[u8]) -> bool {
    let pos = br.pos;
    let total_bits = rbsp.len() * 8;
    // We need at least 1 payload bit + the 1-bit stop + up to 7 zero
    // pad bits. "More data" = at least 9 bits remain.
    total_bits.saturating_sub(pos) > 8
}

fn clamp_to_i8(v: i32) -> i8 {
    v.clamp(i8::MIN as i32, i8::MAX as i32) as i8
}

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
    fn from_ue(v: u32) -> Option<Self> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_h264_baseline_yuv420p() {
        // Minimal H.264 baseline SPS: profile=66 → spec-forced 4:2:0 8-bit.
        let sps_rbsp = vec![
            66, // profile_idc = 66 (baseline)
            0,  // constraints + reserved
            30, // level_idc
            // seq_parameter_set_id = ue(0) = 1 bit, value 0 → bit "1"
            0b1000_0000,
        ];
        let mut sample = vec![0, 0, 0, 1, 0x27]; // start code + NAL header (type=7)
        sample.extend_from_slice(&sps_rbsp);
        let pf = detect_h264(&sample).unwrap();
        assert_eq!(pf, PixelFormat::Yuv420p);
    }

    #[test]
    fn empty_samples_returns_default() {
        let pf = detect("h264", &[]);
        assert_eq!(pf, PixelFormat::Yuv420p);
    }

    #[test]
    fn unknown_codec_returns_default() {
        let pf = detect("prores", &[vec![1, 2, 3]]);
        assert_eq!(pf, PixelFormat::Yuv420p);
    }

    #[test]
    fn from_chroma_and_depth_420_8bit() {
        assert_eq!(
            PixelFormat::from_chroma_and_depth(1, 8),
            PixelFormat::Yuv420p
        );
        assert_eq!(
            PixelFormat::from_chroma_and_depth(1, 10),
            PixelFormat::Yuv420p10le
        );
        assert_eq!(
            PixelFormat::from_chroma_and_depth(2, 8),
            PixelFormat::Yuv422p
        );
        assert_eq!(
            PixelFormat::from_chroma_and_depth(3, 8),
            PixelFormat::Yuv444p
        );
    }

    #[test]
    fn as_ffmpeg_str_matches_python_names() {
        assert_eq!(PixelFormat::Yuv420p.as_ffmpeg_str(), "yuv420p");
        assert_eq!(PixelFormat::Yuv420p10le.as_ffmpeg_str(), "yuv420p10le");
        assert_eq!(PixelFormat::Yuv444p.as_ffmpeg_str(), "yuv444p");
    }

    // ─── Deep-parse: BitWriter + SPS synthesis helpers ────────────
    //
    // `BitWriter` mirrors `BitReader` MSB-first layout so synthesised
    // samples round-trip through `parse_h264_sps` / `parse_hevc_sps`
    // with byte-for-byte fidelity. `write_ue` inverts `read_ue` by
    // encoding codeNum `v` as `z` leading zeros (where `z =
    // floor(log2(v+1))`) + a `1` marker + `z` suffix bits equal to
    // `v + 1 - (1 << z)`.

    struct BitWriter {
        bytes: Vec<u8>,
        bit_pos: usize, // 0..=8 (when ==8 we allocate a fresh byte on next write)
    }

    impl BitWriter {
        fn new() -> Self {
            Self {
                bytes: Vec::new(),
                bit_pos: 8,
            }
        }
        fn write_bit(&mut self, b: u8) {
            if self.bit_pos == 8 {
                self.bytes.push(0);
                self.bit_pos = 0;
            }
            if b != 0 {
                let idx = self.bytes.len() - 1;
                self.bytes[idx] |= 1 << (7 - self.bit_pos);
            }
            self.bit_pos += 1;
        }
        fn write_bits(&mut self, val: u64, n: usize) {
            // u64 is wide enough for every H.264 / HEVC SPS field we
            // synthesise (longest contiguous run is the 48-bit HEVC
            // profile_tier_level constraint-flags block).
            for i in 0..n {
                let bit = ((val >> (n - 1 - i)) & 1) as u8;
                self.write_bit(bit);
            }
        }
        fn write_ue(&mut self, v: u32) {
            let z = if v == 0 { 0 } else { (v + 1).ilog2() as usize };
            for _ in 0..z {
                self.write_bit(0);
            }
            self.write_bit(1);
            if z > 0 {
                let suffix = (v + 1) - (1u32 << z);
                self.write_bits(suffix as u64, z);
            }
        }
        fn bytes(self) -> Vec<u8> {
            self.bytes
        }
    }

    /// Build a minimal H.264 baseline SPS RBSP for the given dims with
    /// no scaling lists, no pic_order_cnt_type==1 branch, no cropping.
    /// Profile=66 skips the chroma_format / bit_depth / scaling_matrix
    /// block entirely per §7.3.2.1.1. `width_in_mbs` / `height_in_mbs`
    /// are the `pic_width_in_mbs_minus1 + 1` and
    /// `pic_height_in_map_units_minus1 + 1` values — the helper
    /// encodes the minus1 forms on the wire.
    fn build_h264_baseline_sps(width_in_mbs: u32, height_in_mbs: u32) -> Vec<u8> {
        let mut w = BitWriter::new();
        w.write_bits(66, 8); // profile_idc = Baseline
        w.write_bits(0, 8); // constraint_set_flags + reserved
        w.write_bits(30, 8); // level_idc
        w.write_ue(0); // seq_parameter_set_id
        w.write_ue(0); // log2_max_frame_num_minus4
        w.write_ue(0); // pic_order_cnt_type
        w.write_ue(0); // log2_max_pic_order_cnt_lsb_minus4
        w.write_ue(1); // max_num_ref_frames
        w.write_bit(0); // gaps_in_frame_num_value_allowed_flag
        w.write_ue(width_in_mbs - 1); // pic_width_in_mbs_minus1
        w.write_ue(height_in_mbs - 1); // pic_height_in_map_units_minus1
        w.write_bit(1); // frame_mbs_only_flag
        w.write_bit(1); // direct_8x8_inference_flag
        w.write_bit(0); // frame_cropping_flag
        w.write_bit(0); // vui_parameters_present_flag
        w.write_bit(1); // rbsp_trailing_bits stop bit
        // zero-align is implicit — trailing bits in partial last byte are 0
        let mut sample = vec![0x00, 0x00, 0x00, 0x01, 0x67]; // Annex-B + NAL header type=7
        sample.extend_from_slice(&w.bytes());
        sample
    }

    #[test]
    fn parse_h264_sps_baseline_1280x720() {
        let sample = build_h264_baseline_sps(1280 / 16, 720 / 16);
        let info = parse_h264_sps(&sample).expect("parse");
        assert_eq!(info.profile_idc, 66);
        assert_eq!(info.chroma_format_idc, 1); // spec-forced for Baseline
        assert_eq!(info.width, Some(1280));
        assert_eq!(info.height, Some(720));
        assert!(info.frame_mbs_only);
    }

    #[test]
    fn parse_h264_sps_baseline_640x480() {
        let sample = build_h264_baseline_sps(640 / 16, 480 / 16);
        let info = parse_h264_sps(&sample).expect("parse");
        assert_eq!(info.width, Some(640));
        assert_eq!(info.height, Some(480));
    }

    #[test]
    fn parse_h264_sps_with_cropping_1920x1080() {
        // 1920×1088 coded → cropped to 1920×1080 via crop_bottom=4 chroma
        // samples (SubHeightC=2, CropUnitY=2 → 8 luma samples cropped).
        let mut w = BitWriter::new();
        w.write_bits(66, 8);
        w.write_bits(0, 8);
        w.write_bits(40, 8);
        w.write_ue(0);
        w.write_ue(0);
        w.write_ue(0);
        w.write_ue(0);
        w.write_ue(1);
        w.write_bit(0);
        w.write_ue(1920 / 16 - 1); // pic_width_in_mbs_minus1
        w.write_ue(1088 / 16 - 1); // pic_height_in_map_units_minus1
        w.write_bit(1); // frame_mbs_only_flag
        w.write_bit(1); // direct_8x8_inference_flag
        w.write_bit(1); // frame_cropping_flag
        w.write_ue(0); // frame_crop_left_offset
        w.write_ue(0); // frame_crop_right_offset
        w.write_ue(0); // frame_crop_top_offset
        w.write_ue(4); // frame_crop_bottom_offset (chroma samples)
        w.write_bit(0); // vui_parameters_present_flag
        w.write_bit(1); // rbsp trailing stop bit
        let mut sample = vec![0, 0, 0, 1, 0x67];
        sample.extend_from_slice(&w.bytes());
        let info = parse_h264_sps(&sample).expect("parse");
        assert_eq!(info.width, Some(1920));
        assert_eq!(info.height, Some(1080));
    }

    #[test]
    fn parse_h264_sps_high_profile_422_returns_chroma_even_without_dims() {
        // Profile=122 (High 4:2:2) gates chroma_format_idc=2. We don't
        // synthesise scaling lists or the rest of the SPS (would be
        // significantly larger), so width/height come back as None but
        // chroma_format_idc must still be populated — this is the
        // contract that `decode::h264`'s reject path depends on.
        let mut w = BitWriter::new();
        w.write_bits(122, 8); // profile_idc = High 4:2:2
        w.write_bits(0, 8);
        w.write_bits(40, 8);
        w.write_ue(0); // sps_id
        w.write_ue(2); // chroma_format_idc = 4:2:2
        w.write_ue(0); // bit_depth_luma_minus8
        w.write_ue(0); // bit_depth_chroma_minus8
        w.write_bit(0); // qpprime_y_zero_transform_bypass_flag
        w.write_bit(0); // seq_scaling_matrix_present_flag
        // Truncate here — the remainder would be log2_max_frame_num etc
        // but this is enough for the chroma-reject contract to hold.
        let mut sample = vec![0, 0, 0, 1, 0x67];
        sample.extend_from_slice(&w.bytes());
        let info = parse_h264_sps(&sample).expect("parse");
        assert_eq!(info.profile_idc, 122);
        assert_eq!(info.chroma_format_idc, 2);
        // width/height may be None here — we truncated the SPS; that's OK.
    }

    /// Build a minimal HEVC SPS at given pic_width / pic_height with
    /// chroma_format_idc=1 (4:2:0), bit_depth=8, no conformance window.
    /// profile_tier_level is a default Main 8-bit profile with
    /// max_sub_layers_minus1=0 (no sub-layer loop).
    fn build_hevc_sps(pic_width: u32, pic_height: u32) -> Vec<u8> {
        build_hevc_sps_full(pic_width, pic_height, false, 0, 0, 0, 0)
    }

    fn build_hevc_sps_full(
        pic_width: u32,
        pic_height: u32,
        conformance_window: bool,
        cwl: u32,
        cwr: u32,
        cwt: u32,
        cwb: u32,
    ) -> Vec<u8> {
        let mut w = BitWriter::new();
        w.write_bits(0, 4); // sps_video_parameter_set_id
        w.write_bits(0, 3); // sps_max_sub_layers_minus1 = 0
        w.write_bits(1, 1); // sps_temporal_id_nesting_flag
        w.write_bits(0b0_0_00001, 8); // profile_space=0, tier=0, profile_idc=1 (Main)
        w.write_bits(0x40000000, 32); // profile_compatibility_flag[32]
        w.write_bits(0, 48); // constraint flags
        w.write_bits(93, 8); // general_level_idc

        w.write_ue(0); // sps_seq_parameter_set_id
        w.write_ue(1); // chroma_format_idc = 4:2:0
        w.write_ue(pic_width);
        w.write_ue(pic_height);
        if conformance_window {
            w.write_bit(1);
            w.write_ue(cwl);
            w.write_ue(cwr);
            w.write_ue(cwt);
            w.write_ue(cwb);
        } else {
            w.write_bit(0); // conformance_window_flag
        }
        w.write_ue(0); // bit_depth_luma_minus8
        w.write_ue(0); // bit_depth_chroma_minus8
        w.write_ue(4); // log2_max_pic_order_cnt_lsb_minus4 (8-bit POC)
        w.write_bit(1); // sps_sub_layer_ordering_info_present_flag
        // Single entry for max_sub_layers_minus1 == 0:
        w.write_ue(4); // sps_max_dec_pic_buffering_minus1
        w.write_ue(0); // sps_max_num_reorder_pics
        w.write_ue(0); // sps_max_latency_increase_plus1
        w.write_ue(0); // log2_min_luma_coding_block_size_minus3
        w.write_ue(3); // log2_diff_max_min_luma_coding_block_size
        w.write_ue(0); // log2_min_luma_transform_block_size_minus2
        w.write_ue(3); // log2_diff_max_min_luma_transform_block_size
        w.write_ue(2); // max_transform_hierarchy_depth_inter
        w.write_ue(2); // max_transform_hierarchy_depth_intra
        w.write_bit(0); // scaling_list_enabled_flag
        w.write_bit(1); // amp_enabled_flag
        w.write_bit(1); // sample_adaptive_offset_enabled_flag
        w.write_bit(0); // pcm_enabled_flag
        w.write_ue(0); // num_short_term_ref_pic_sets (none)
        w.write_bit(0); // long_term_ref_pics_present_flag
        w.write_bit(1); // sps_temporal_mvp_enabled_flag
        w.write_bit(0); // strong_intra_smoothing_enabled_flag
        w.write_bit(0); // vui_parameters_present_flag
        w.write_bit(0); // sps_extension_present_flag
        w.write_bit(1); // rbsp trailing stop bit
        let mut sample = vec![0, 0, 0, 1, 0x42, 0x01];
        sample.extend_from_slice(&w.bytes());
        sample
    }

    #[test]
    fn parse_hevc_sps_1920x1080_no_crop() {
        let sample = build_hevc_sps(1920, 1080);
        let info = parse_hevc_sps(&sample).expect("parse");
        assert_eq!(info.chroma_format_idc, 1);
        assert_eq!(info.bit_depth_luma, 8);
        assert_eq!(info.width, Some(1920));
        assert_eq!(info.height, Some(1080));
    }

    #[test]
    fn parse_hevc_sps_with_conformance_window() {
        // Coded 1920×1088, conformance window crops 8 luma samples
        // off the bottom → 1920×1080 output.
        let sample = build_hevc_sps_full(1920, 1088, true, 0, 0, 0, 4);
        let info = parse_hevc_sps(&sample).expect("parse");
        assert_eq!(info.width, Some(1920));
        assert_eq!(info.height, Some(1080));
    }

    #[test]
    fn parse_mpeg2_sequence_header_no_extension_640x480() {
        // start code 00 00 01 B3 + 3-byte body: 12 bits h + 12 bits v.
        // 640 = 0x280 → high 8 bits = 0x28, low 4 = 0. 480 = 0x1E0 →
        // high 4 = 1, low 8 = 0xE0. So bytes: 0x28 0x01 0xE0.
        let sample = vec![0x00, 0x00, 0x01, 0xB3, 0x28, 0x01, 0xE0, 0x13, 0xFF, 0xFF];
        let info = parse_mpeg2_sequence_header(&sample).expect("parse");
        assert_eq!(info.width, 640);
        assert_eq!(info.height, 480);
    }

    #[test]
    fn parse_mpeg2_sequence_header_with_extension_upgrades_to_14bit() {
        // 1920 = 0x780 (fits in 12 bits: high 8=0x78 low 4=0). 1080 =
        // 0x438 (fits 12 bits: high 4=4 low 8=0x38). So sequence header
        // alone would return 1920×1080 — same as the extended form with
        // h_ext=0 v_ext=0. Set h_ext=1, v_ext=0 so the extension MUST
        // flip the value (otherwise the test would pass even if the
        // extension parse was broken).
        let mut bytes = vec![0x00, 0x00, 0x01, 0xB3, 0x78, 0x04, 0x38, 0x13, 0xFF, 0xFF];
        // Now tack on 00 00 01 B5 + extension body:
        // Extension body (bit layout, MSB first within each byte):
        //   ext_id(4)=0001 | profile_level(8)=0 | progressive(1)=1 |
        //   chroma(2)=01 (4:2:0) | h_ext(2)=01 | v_ext(2)=10
        // = 19 bits. Use BitWriter to avoid manual packing errors.
        let mut w = BitWriter::new();
        w.write_bits(1, 4); // extension_start_code_identifier = 0001 (seq ext)
        w.write_bits(0, 8); // profile_and_level_indication
        w.write_bit(1); // progressive_sequence
        w.write_bits(1, 2); // chroma_format = 01 (4:2:0)
        w.write_bits(1, 2); // horizontal_size_extension = 01 (h |= 1<<12 = 4096)
        w.write_bits(2, 2); // vertical_size_extension = 10 (v |= 2<<12 = 8192)
        w.write_bits(0, 1); // pad to byte
        bytes.extend_from_slice(&[0x00, 0x00, 0x01, 0xB5]);
        bytes.extend_from_slice(&w.bytes());
        let info = parse_mpeg2_sequence_header(&bytes).expect("parse");
        assert_eq!(info.width, 1920 | (1 << 12)); // 6016
        assert_eq!(info.height, 1080 | (2 << 12)); // 9272
    }

    #[test]
    fn parse_mpeg2_sequence_header_none_when_no_start_code() {
        let sample = vec![0xFFu8; 128];
        assert!(parse_mpeg2_sequence_header(&sample).is_none());
    }

    #[test]
    fn detect_dims_dispatches_by_codec() {
        let h264 = build_h264_baseline_sps(1280 / 16, 720 / 16);
        let hevc = build_hevc_sps(1920, 1080);
        let mpeg2 = vec![0x00, 0x00, 0x01, 0xB3, 0x28, 0x01, 0xE0, 0x13, 0xFF, 0xFF];
        assert_eq!(detect_dims("h264", &[h264.clone()]), Some((1280, 720)));
        assert_eq!(detect_dims("avc1", &[h264]), Some((1280, 720)));
        assert_eq!(detect_dims("h265", &[hevc.clone()]), Some((1920, 1080)));
        assert_eq!(detect_dims("hevc", &[hevc]), Some((1920, 1080)));
        assert_eq!(detect_dims("mpeg2", &[mpeg2]), Some((640, 480)));
        assert_eq!(detect_dims("unknown", &[vec![0u8; 8]]), None);
        assert_eq!(detect_dims("h264", &[]), None);
    }

    /// Build a minimal H.264 PPS NAL (type 8) with the baseline set of
    /// fields — no FMO, no extended (High-profile) trailer. Returns
    /// the Annex-B sample (start code + NAL header byte + RBSP).
    fn build_h264_baseline_pps(pps_id: u32, sps_id: u32) -> Vec<u8> {
        let mut w = BitWriter::new();
        w.write_ue(pps_id); // pic_parameter_set_id
        w.write_ue(sps_id); // seq_parameter_set_id
        w.write_bit(0); // entropy_coding_mode_flag = CAVLC
        w.write_bit(0); // bottom_field_pic_order_in_frame_present_flag
        w.write_ue(0); // num_slice_groups_minus1 = 0 (no FMO)
        w.write_ue(0); // num_ref_idx_l0_default_active_minus1 = 0
        w.write_ue(0); // num_ref_idx_l1_default_active_minus1 = 0
        w.write_bit(0); // weighted_pred_flag
        w.write_bits(0, 2); // weighted_bipred_idc = 0
        w.write_ue(0); // pic_init_qp_minus26 = 0 (encoded as se(v)=0 → ue 0)
        w.write_ue(0); // pic_init_qs_minus26 = 0
        w.write_ue(0); // chroma_qp_index_offset = 0
        w.write_bit(1); // deblocking_filter_control_present_flag
        w.write_bit(0); // constrained_intra_pred_flag
        w.write_bit(0); // redundant_pic_cnt_present_flag
        w.write_bit(1); // rbsp trailing stop bit
        let mut sample = vec![0x00, 0x00, 0x00, 0x01, 0x68]; // NAL header: type=8 (PPS), nal_ref_idc=3
        sample.extend_from_slice(&w.bytes());
        sample
    }

    #[test]
    fn parse_h264_pps_baseline_roundtrip() {
        let sample = build_h264_baseline_pps(0, 0);
        let info = parse_h264_pps(&sample).expect("PPS parses");
        assert_eq!(info.pic_parameter_set_id, 0);
        assert_eq!(info.seq_parameter_set_id, 0);
        assert!(!info.entropy_coding_mode_flag);
        assert!(!info.bottom_field_pic_order_in_frame_present_flag);
        assert_eq!(info.num_slice_groups_minus1, 0);
        assert_eq!(info.num_ref_idx_l0_default_active_minus1, 0);
        assert_eq!(info.num_ref_idx_l1_default_active_minus1, 0);
        assert!(!info.weighted_pred_flag);
        assert_eq!(info.weighted_bipred_idc, 0);
        assert_eq!(info.pic_init_qp_minus26, 0);
        assert_eq!(info.pic_init_qs_minus26, 0);
        assert_eq!(info.chroma_qp_index_offset, 0);
        assert!(info.deblocking_filter_control_present_flag);
        assert!(!info.constrained_intra_pred_flag);
        assert!(!info.redundant_pic_cnt_present_flag);
    }

    #[test]
    fn parse_h264_pps_nonzero_ids_and_flags() {
        let mut w = BitWriter::new();
        w.write_ue(3); // pps_id
        w.write_ue(7); // sps_id
        w.write_bit(1); // entropy_coding_mode_flag = CABAC
        w.write_bit(1); // bottom_field_pic_order_in_frame_present_flag
        w.write_ue(0); // num_slice_groups_minus1
        w.write_ue(2); // num_ref_idx_l0_default_active_minus1 = 2
        w.write_ue(1); // num_ref_idx_l1_default_active_minus1 = 1
        w.write_bit(1); // weighted_pred_flag
        w.write_bits(2, 2); // weighted_bipred_idc = 2
        // pic_init_qp_minus26 = -5 (valid se range). codeNum for -5 = 2*5 = 10.
        w.write_ue(10);
        // pic_init_qs_minus26 = 3. codeNum for +3 = 2*3 - 1 = 5.
        w.write_ue(5);
        // chroma_qp_index_offset = 0
        w.write_ue(0);
        w.write_bit(0); // deblocking_filter_control_present_flag
        w.write_bit(1); // constrained_intra_pred_flag
        w.write_bit(1); // redundant_pic_cnt_present_flag
        w.write_bit(1); // rbsp stop
        let mut sample = vec![0x00, 0x00, 0x00, 0x01, 0x68];
        sample.extend_from_slice(&w.bytes());
        let info = parse_h264_pps(&sample).expect("parse");
        assert_eq!(info.pic_parameter_set_id, 3);
        assert_eq!(info.seq_parameter_set_id, 7);
        assert!(info.entropy_coding_mode_flag);
        assert!(info.bottom_field_pic_order_in_frame_present_flag);
        assert_eq!(info.num_ref_idx_l0_default_active_minus1, 2);
        assert_eq!(info.num_ref_idx_l1_default_active_minus1, 1);
        assert!(info.weighted_pred_flag);
        assert_eq!(info.weighted_bipred_idc, 2);
        assert_eq!(info.pic_init_qp_minus26, -5);
        assert_eq!(info.pic_init_qs_minus26, 3);
        assert!(!info.deblocking_filter_control_present_flag);
        assert!(info.constrained_intra_pred_flag);
        assert!(info.redundant_pic_cnt_present_flag);
    }

    #[test]
    fn parse_h264_pps_returns_none_when_no_pps_in_sample() {
        // Sample contains only an SPS NAL — PPS parser should bail.
        let sample = build_h264_baseline_sps(80, 45); // just a SPS
        assert!(parse_h264_pps(&sample).is_none());
    }

    /// Build a minimal H.264 slice NAL (type 5 for IDR) with:
    /// - first_mb_in_slice = 0
    /// - slice_type = I (codeNum 2)
    /// - pic_parameter_set_id = 0
    /// - frame_num = 0 (4 bits, since log2_max_frame_num_minus4 = 0)
    /// - idr_pic_id = 0 (only when is_idr)
    /// - pic_order_cnt_lsb = 0 (4 bits, since log2_max_pic_order_cnt_lsb_minus4 = 0)
    fn build_h264_idr_slice_header_rbsp() -> Vec<u8> {
        let mut w = BitWriter::new();
        w.write_ue(0); // first_mb_in_slice
        w.write_ue(7); // slice_type = 7 → 7 % 5 = 2 → I, "all I" variant
        w.write_ue(0); // pic_parameter_set_id
        w.write_bits(0, 4); // frame_num (4 bits)
        w.write_ue(0); // idr_pic_id
        w.write_bits(0, 4); // pic_order_cnt_lsb (4 bits)
        // Don't need rbsp trailing bits — caller doesn't look past the
        // fields we care about and the BitReader tolerates short data.
        w.bytes()
    }

    #[test]
    fn parse_h264_slice_header_idr_i_slice() {
        let sps = parse_h264_sps(&build_h264_baseline_sps(1280 / 16, 720 / 16)).expect("sps");
        let pps = parse_h264_pps(&build_h264_baseline_pps(0, 0)).expect("pps");
        let rbsp = build_h264_idr_slice_header_rbsp();
        // NAL header byte for IDR slice: forbidden_zero=0, nal_ref_idc=3, type=5 → 0x65
        let mut sample = vec![0x00, 0x00, 0x00, 0x01, 0x65];
        sample.extend_from_slice(&rbsp);

        let sh = parse_h264_slice_header(&sample, &sps, &pps).expect("slice");
        assert_eq!(sh.first_mb_in_slice, 0);
        assert_eq!(sh.slice_type, H264SliceType::I);
        assert_eq!(sh.pic_parameter_set_id, 0);
        assert!(sh.is_idr);
        assert_eq!(sh.frame_num, 0);
        assert!(!sh.field_pic_flag);
        assert_eq!(sh.idr_pic_id, Some(0));
        assert_eq!(sh.pic_order_cnt_lsb, Some(0));
    }

    #[test]
    fn parse_h264_slice_header_returns_none_without_sps_context() {
        // Build an SPS with profile 122 (High 4:2:2) — chroma-reject
        // path stops parsing before pic_order_cnt_type is reached, so
        // sps.pic_order_cnt_type = None. Slice header parser should
        // gracefully bail.
        let mut w = BitWriter::new();
        w.write_bits(122, 8);
        w.write_bits(0, 8);
        w.write_bits(40, 8);
        w.write_ue(0); // sps_id
        w.write_ue(2); // chroma_format_idc
        w.write_ue(0);
        w.write_ue(0);
        w.write_bit(0); // qpprime
        w.write_bit(0); // scaling_matrix_present = 0
        let mut sample = vec![0, 0, 0, 1, 0x67];
        sample.extend_from_slice(&w.bytes());
        let sps = parse_h264_sps(&sample).expect("sps parses");
        assert!(sps.pic_order_cnt_type.is_none());

        let pps = parse_h264_pps(&build_h264_baseline_pps(0, 0)).expect("pps");
        let rbsp = build_h264_idr_slice_header_rbsp();
        let mut slice_sample = vec![0x00, 0x00, 0x00, 0x01, 0x65];
        slice_sample.extend_from_slice(&rbsp);
        // sps.pic_order_cnt_type is None → parser bails via `?`.
        // Technically this tests the _early-exit_ because log2_max_frame_num_minus4
        // is None too. Either way, slice header parsing requires a full SPS.
        assert!(parse_h264_slice_header(&slice_sample, &sps, &pps).is_none());
    }

    #[test]
    fn parse_h264_slice_type_ue_mapping_covers_both_halves() {
        // codeNum 0..=4 → {P, B, I, SP, SI}, codeNum 5..=9 → same
        // five types ("all same" annotation). Both map identically.
        for (code, expected) in [
            (0, H264SliceType::P),
            (5, H264SliceType::P),
            (1, H264SliceType::B),
            (6, H264SliceType::B),
            (2, H264SliceType::I),
            (7, H264SliceType::I),
            (3, H264SliceType::SP),
            (8, H264SliceType::SP),
            (4, H264SliceType::SI),
            (9, H264SliceType::SI),
        ] {
            assert_eq!(
                H264SliceType::from_ue(code),
                Some(expected),
                "code {}",
                code
            );
        }
    }

    #[test]
    fn bit_reader_read_se_exp_golomb_mapping() {
        // codeNum → signed: 0→0, 1→+1, 2→-1, 3→+2, 4→-2.
        // Encode each via BitWriter::write_ue and verify read_se.
        for (code, expected) in [(0u32, 0i32), (1, 1), (2, -1), (3, 2), (4, -2), (5, 3)] {
            let mut w = BitWriter::new();
            w.write_ue(code);
            let bytes = w.bytes();
            let mut br = BitReader::new(&bytes);
            assert_eq!(
                br.read_se(),
                Some(expected),
                "codeNum={} expected={}",
                code,
                expected
            );
        }
    }
}
