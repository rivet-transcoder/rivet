//! Unit tests for pixel_format parsers and detectors.
//! Declared by `#[cfg(test)] mod tests;` in mod.rs — this file is the
//! inner content only (no outer wrapper needed).

use super::bitreader::BitReader;
use super::h264::detect_h264;
use super::*;
use crate::PixelFormat;

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
