// ─── Tests: the H.264 / H.265 (and AV1) property sequences ────────
//
// Each `apply_*_properties` is driven against the recording mock component
// from `tests.rs`, and the recorded `(name, value)` pairs are checked
// against the header-cited constants — so a renamed property or a wrong
// enum value fails here, not on a customer's GPU. The level tables and the
// parameter folding are checked directly.

use super::h26x::*;
use super::tests::{make_mock_pair, recorded};
use super::{
    AmfVariant, apply_av1_properties, apply_avc_properties, apply_hevc_properties,
    av1_quality_preset, effective_keyframe_interval,
};
use crate::encode::tuning::{AmfQualityPreset, AmfRateControl, QualityTarget, SpeedTier};
use crate::encode::{AUTO_FROM_TARGET, EncoderConfig};
use crate::frame::{ColorMetadata, PixelFormat, TransferFn, VideoCodec};
use std::collections::HashMap;
use std::ffi::c_void;

fn config(codec: VideoCodec, pixel_format: PixelFormat) -> EncoderConfig {
    EncoderConfig {
        width: 1920,
        height: 1080,
        frame_rate: 30.0,
        codec,
        pixel_format,
        keyframe_interval: 120,
        ..Default::default()
    }
}

/// Drive one property sequence against the mock and index the result.
fn run(apply: unsafe fn(*mut c_void, &EncoderConfig) -> anyhow::Result<String>, cfg: &EncoderConfig) -> Recorded {
    // The mock records into a thread-local; clear it via the pair helper's
    // reset by taking a fresh snapshot after the run.
    super::tests::RECORDED.with(|r| r.borrow_mut().clear());
    let (_, mut component) = make_mock_pair();
    let component_ptr: *mut c_void = component.as_mut() as *mut _ as *mut c_void;
    let summary = unsafe { apply(component_ptr, cfg) }.expect("property sequence");
    let list = recorded();
    let mut map = HashMap::new();
    for (name, v) in &list {
        map.insert(name.clone(), *v);
    }
    Recorded { order: list.into_iter().map(|(n, _)| n).collect(), map, summary }
}

struct Recorded {
    order: Vec<String>,
    map: HashMap<String, AmfVariant>,
    summary: String,
}

impl Recorded {
    fn int(&self, name: &str) -> i64 {
        self.map
            .get(name)
            .unwrap_or_else(|| panic!("property {name} was not set; set: {:?}", self.order))
            .as_int64()
            .unwrap_or_else(|| panic!("property {name} is not int64"))
    }
    fn bool_(&self, name: &str) -> bool {
        self.map
            .get(name)
            .unwrap_or_else(|| panic!("property {name} was not set; set: {:?}", self.order))
            .as_bool()
            .unwrap_or_else(|| panic!("property {name} is not bool"))
    }
    fn rate(&self, name: &str) -> (u32, u32) {
        self.map
            .get(name)
            .unwrap_or_else(|| panic!("property {name} was not set; set: {:?}", self.order))
            .as_rate()
            .unwrap_or_else(|| panic!("property {name} is not AMFRate"))
    }
    fn has(&self, name: &str) -> bool {
        self.map.contains_key(name)
    }
}

// ── Levels ────────────────────────────────────────────────────

/// H.264 (High): 1080p30 needs 4.0 (8160 MBs ≤ 8192, 244 800 MB/s ≤ 245 760),
/// 1080p60 needs 4.2, 4K30 needs 5.1, 4K60 needs 5.2, 720p30 needs 3.1.
#[test]
fn h264_levels_by_size_and_rate() {
    assert_eq!(h264_level_for(1920, 1080, 30.0).amf_value, 40);
    assert_eq!(h264_level_for(1920, 1080, 60.0).amf_value, 42);
    assert_eq!(h264_level_for(3840, 2160, 30.0).amf_value, 51);
    assert_eq!(h264_level_for(3840, 2160, 60.0).amf_value, 52);
    assert_eq!(h264_level_for(1280, 720, 30.0).amf_value, 31);
    assert_eq!(h264_level_for(640, 480, 30.0).amf_value, 30, "the table starts at 3.0");
    assert_eq!(h264_level_for(7680, 4320, 120.0).amf_value, 62, "8K120 tops out at 6.2");
    // High profile's ceiling is 1.25 × the Main figure: 4.0 → 25 Mbit/s.
    assert_eq!(h264_level_for(1920, 1080, 30.0).max_bitrate, 25_000_000);
}

/// H.265 (Main tier): 1080p30 → 4.0 (120), 1080p60 → 4.1 (123), 4K30 → 5.0
/// (150), 4K60 → 5.1 (153), 720p30 → 3.1 (93). AMF values are 30 × level.
#[test]
fn h265_levels_by_size_and_rate() {
    assert_eq!(h265_level_for(1920, 1080, 30.0).amf_value, 120);
    assert_eq!(h265_level_for(1920, 1080, 60.0).amf_value, 123);
    assert_eq!(h265_level_for(3840, 2160, 30.0).amf_value, 150);
    assert_eq!(h265_level_for(3840, 2160, 60.0).amf_value, 153);
    assert_eq!(h265_level_for(1280, 720, 30.0).amf_value, 93);
    assert_eq!(h265_level_for(7680, 4320, 120.0).amf_value, 186, "8K120 tops out at 6.2");
    assert_eq!(h265_level_for(3840, 2160, 60.0).max_bitrate, 40_000_000, "5.1 Main tier");
}

/// The QVBR ceiling: 0.25 bit/pixel/s, floored at 2 Mbit/s, capped by the
/// level.
#[test]
fn qvbr_ceiling_scales_floors_and_caps() {
    assert_eq!(qvbr_bitrate_ceiling(1920, 1080, 30.0, None), 15_552_000);
    assert_eq!(qvbr_bitrate_ceiling(320, 240, 15.0, None), 2_000_000, "floor");
    assert_eq!(qvbr_bitrate_ceiling(3840, 2160, 60.0, None), 124_416_000);
    assert_eq!(qvbr_bitrate_ceiling(3840, 2160, 60.0, Some(40_000_000)), 40_000_000, "level cap");
    assert_eq!(qvbr_bitrate_ceiling(1920, 1080, f64::NAN, None), 15_552_000, "bad fps → 30");
}

// ── Parameter folding ─────────────────────────────────────────

/// The derived quantiser set: `Standard` is QP 26 / 28, QVBR level 26,
/// `Quality` preset; `VisuallyLossless` is CQP at 18; `Draft` is `Balanced`;
/// `Archive` is `HighQuality`.
#[test]
fn h26x_quant_from_target_and_tier() {
    let q = h26x_quant(&config(VideoCodec::H264, PixelFormat::Yuv420p));
    assert_eq!(q.rc, AmfRateControl::QualityVbr);
    assert_eq!((q.qp_i, q.qp_p, q.qvbr_level), (26, 28, 26));
    assert_eq!(q.preset, AmfQualityPreset::Quality);

    let mut cfg = config(VideoCodec::H265, PixelFormat::Yuv420p);
    cfg.target = QualityTarget::VisuallyLossless;
    cfg.tier = SpeedTier::Archive;
    let q = h26x_quant(&cfg);
    assert_eq!(q.rc, AmfRateControl::Cqp);
    assert_eq!((q.qp_i, q.qp_p, q.qvbr_level), (18, 20, 18));
    assert_eq!(q.preset, AmfQualityPreset::HighQuality);

    cfg.tier = SpeedTier::Draft;
    assert_eq!(h26x_quant(&cfg).preset, AmfQualityPreset::Balanced);
}

/// The legacy CRF escape hatch replaces the derived QP outright (it is
/// already in this codec's 0..51 currency), and `constant_qp` forces CQP.
#[test]
fn h26x_quant_legacy_crf_and_constant_qp() {
    let mut cfg = config(VideoCodec::H264, PixelFormat::Yuv420p);
    cfg.quality = 20;
    let q = h26x_quant(&cfg);
    assert_eq!((q.qp_i, q.qp_p, q.qvbr_level), (20, 22, 20));
    assert_eq!(q.rc, AmfRateControl::QualityVbr, "a CRF alone does not change the mode");

    cfg.quality = 0;
    let q = h26x_quant(&cfg);
    assert_eq!((q.qp_i, q.qp_p, q.qvbr_level), (0, 2, 1), "QVBR level is 1-51, QP may be 0");

    cfg.quality = 80;
    assert_eq!(h26x_quant(&cfg).qp_i, 51, "clamped to the codec's scale");

    cfg.quality = AUTO_FROM_TARGET;
    cfg.constant_qp = true;
    assert_eq!(h26x_quant(&cfg).rc, AmfRateControl::Cqp);
}

/// Per-rung overrides reach the quantiser through the adapter.
#[test]
fn h26x_quant_applies_overrides() {
    let mut cfg = config(VideoCodec::H265, PixelFormat::Yuv420p);
    cfg.overrides.quality_delta = 4;
    let q = h26x_quant(&cfg);
    assert_eq!((q.qp_i, q.qp_p, q.qvbr_level), (30, 32, 30));
    cfg.overrides.quality_delta = -40;
    let q = h26x_quant(&cfg);
    assert_eq!((q.qp_i, q.qp_p, q.qvbr_level), (0, 0, 1), "clamped, QVBR floor is 1");
    cfg.overrides.quality_delta = 0;
    cfg.overrides.speed_tier = Some(SpeedTier::Archive);
    assert_eq!(h26x_quant(&cfg).preset, AmfQualityPreset::HighQuality);
}

#[test]
fn h26x_format_check_refuses_by_name() {
    assert!(check_h26x_format(VideoCodec::H264, PixelFormat::Yuv420p).is_ok());
    let e = check_h26x_format(VideoCodec::H264, PixelFormat::Yuv420p10le).unwrap_err();
    assert!(e.to_string().contains("8-bit"), "{e}");
    assert!(check_h26x_format(VideoCodec::H265, PixelFormat::Yuv420p).is_ok());
    assert!(check_h26x_format(VideoCodec::H265, PixelFormat::Yuv420p10le).is_ok());
    assert!(check_h26x_format(VideoCodec::H265, PixelFormat::Yuv444p10le).is_err());
    assert!(check_h26x_format(VideoCodec::Av1, PixelFormat::Yuv420p).is_err());
}

#[test]
fn effective_keyframe_interval_defaults_zero_to_240() {
    assert_eq!(effective_keyframe_interval(0), 240);
    assert_eq!(effective_keyframe_interval(48), 48);
}

// ── Preset enums, per codec ───────────────────────────────────

/// AVC (VideoEncoderVCE.h:112-115): BALANCED 0, SPEED 1, QUALITY 2,
/// HIGH_QUALITY 3. HEVC (VideoEncoderHEVC.h:107-110): QUALITY 0, BALANCED 5,
/// SPEED 10, HIGH_QUALITY 15. AV1 (VideoEncoderAV1.h:128-131): HIGH_QUALITY
/// 0, QUALITY 30, BALANCED 70, SPEED 100.
#[test]
fn quality_presets_are_numbered_per_codec() {
    use AmfQualityPreset::*;
    assert_eq!([HighQuality, Quality, Balanced, Speed].map(avc_quality_preset), [3, 2, 0, 1]);
    assert_eq!([HighQuality, Quality, Balanced, Speed].map(hevc_quality_preset), [15, 0, 5, 10]);
    assert_eq!([HighQuality, Quality, Balanced, Speed].map(av1_quality_preset), [0, 30, 70, 100]);
}

// ── Recorded property sequences ───────────────────────────────

/// H.264, 1080p30 Standard/Standard: every property the header names, with
/// the header's values.
#[test]
fn avc_property_sequence_matches_header() {
    let cfg = config(VideoCodec::H264, PixelFormat::Yuv420p);
    let r = run(apply_avc_properties, &cfg);

    assert_eq!(r.order[0], "Usage", "USAGE first: it fully configures the parameter set");
    assert_eq!(r.int("Usage"), 0, "AMF_VIDEO_ENCODER_USAGE_TRANSCODING");
    assert_eq!(r.int("Profile"), 100, "AMF_VIDEO_ENCODER_PROFILE_HIGH");
    assert_eq!(r.int("ProfileLevel"), 40, "AMF_H264_LEVEL__4");
    assert_eq!(r.int("QualityPreset"), 2, "AMF_VIDEO_ENCODER_QUALITY_PRESET_QUALITY");
    assert_eq!(r.int("CABACEnable"), 1, "AMF_VIDEO_ENCODER_CABAC");
    assert_eq!(r.rate("FrameRate"), (30, 1));
    assert_eq!(r.int("BPicturesPattern"), 0, "no B frames");
    assert_eq!(r.int("IDRPeriod"), 120);
    assert_eq!(r.int("OutputMode"), 0, "AMF_VIDEO_ENCODER_OUTPUT_MODE_FRAME");
    assert_eq!(r.int("RateControlMethod"), 4, "AMF_VIDEO_ENCODER_RATE_CONTROL_METHOD_QUALITY_VBR");
    assert_eq!(r.int("QvbrQualityLevel"), 26);
    assert_eq!(r.int("TargetBitrate"), 15_552_000);
    assert_eq!(r.int("PeakBitrate"), 15_552_000);
    assert_eq!(r.int("VBVBufferSize"), 15_552_000);
    assert!(r.bool_("EnforceHRD"), "the ceiling is enforced");
    assert!(!r.bool_("FillerDataEnable"));
    assert_eq!((r.int("QPI"), r.int("QPP"), r.int("QPB")), (26, 28, 28));
    assert_eq!(r.int("ColorBitDepth"), 8, "AMF_COLOR_BIT_DEPTH_8 is the literal 8");
    assert!(!r.bool_("FullRangeColor"));
    assert_eq!(r.int("InColorProfile"), 1, "AMF_VIDEO_CONVERTER_COLOR_PROFILE_709");
    assert_eq!(r.int("OutColorProfile"), 1);
    assert_eq!(r.int("InColorTransferChar"), 1);
    assert_eq!(r.int("OutColorTransferChar"), 1);
    assert_eq!(r.int("InColorPrimaries"), 1);
    assert_eq!(r.int("OutColorPrimaries"), 1);
    assert!(!r.has("HevcUsage") && !r.has("Av1Usage"), "no other codec's names");
    assert!(r.summary.contains("level=40"), "{}", r.summary);
}

/// H.264 at VisuallyLossless is CQP, so the QVBR level and the bitrate
/// constraints are not set at all (a stale constraint under CQP would still
/// be honoured by the driver).
#[test]
fn avc_cqp_sets_no_bitrate_constraints() {
    let mut cfg = config(VideoCodec::H264, PixelFormat::Yuv420p);
    cfg.target = QualityTarget::VisuallyLossless;
    let r = run(apply_avc_properties, &cfg);
    assert_eq!(r.int("RateControlMethod"), 0, "AMF_VIDEO_ENCODER_RATE_CONTROL_METHOD_CONSTANT_QP");
    assert!(!r.has("QvbrQualityLevel"));
    assert!(!r.has("TargetBitrate") && !r.has("PeakBitrate") && !r.has("VBVBufferSize"));
    assert!(!r.has("EnforceHRD"));
    assert_eq!((r.int("QPI"), r.int("QPP")), (18, 20));
}

/// 10-bit H.264 is refused before any property is set.
#[test]
fn avc_refuses_ten_bit() {
    super::tests::RECORDED.with(|r| r.borrow_mut().clear());
    let (_, mut component) = make_mock_pair();
    let component_ptr: *mut c_void = component.as_mut() as *mut _ as *mut c_void;
    let cfg = config(VideoCodec::H264, PixelFormat::Yuv420p10le);
    let err = unsafe { apply_avc_properties(component_ptr, &cfg) }.unwrap_err();
    assert!(err.to_string().contains("8-bit"), "{err}");
    assert!(recorded().is_empty(), "nothing was set");
}

/// H.265 Main, 1080p30 Standard/Standard.
#[test]
fn hevc_property_sequence_matches_header() {
    let cfg = config(VideoCodec::H265, PixelFormat::Yuv420p);
    let r = run(apply_hevc_properties, &cfg);

    assert_eq!(r.order[0], "HevcUsage");
    assert_eq!(r.int("HevcUsage"), 0, "AMF_VIDEO_ENCODER_HEVC_USAGE_TRANSCODING");
    assert_eq!(r.int("HevcProfile"), 1, "AMF_VIDEO_ENCODER_HEVC_PROFILE_MAIN");
    assert_eq!(r.int("HevcTier"), 0, "AMF_VIDEO_ENCODER_HEVC_TIER_MAIN");
    assert_eq!(r.int("HevcProfileLevel"), 120, "AMF_LEVEL_4");
    assert_eq!(r.int("HevcQualityPreset"), 0, "AMF_VIDEO_ENCODER_HEVC_QUALITY_PRESET_QUALITY");
    assert_eq!(r.rate("HevcFrameRate"), (30, 1));
    assert_eq!(r.int("HevcGOPSize"), 120);
    assert_eq!(r.int("HevcGOPSPerIDR"), 1);
    assert_eq!(r.int("HevcHeaderInsertionMode"), 2, "IDR_ALIGNED");
    assert_eq!(r.int("HevcOutputMode"), 0);
    assert_eq!(r.int("HevcRateControlMethod"), 4, "AMF_VIDEO_ENCODER_HEVC_RATE_CONTROL_METHOD_QUALITY_VBR");
    assert_eq!(r.int("HevcQvbrQualityLevel"), 26);
    // Level 4.0 Main tier caps the ceiling at 12 Mbit/s.
    assert_eq!(r.int("HevcTargetBitrate"), 12_000_000);
    assert_eq!(r.int("HevcPeakBitrate"), 12_000_000);
    assert_eq!(r.int("HevcVBVBufferSize"), 12_000_000);
    assert!(r.bool_("HevcEnforceHRD"));
    assert!(!r.bool_("HevcFillerDataEnable"));
    assert_eq!((r.int("HevcQP_I"), r.int("HevcQP_P")), (26, 28));
    assert_eq!(r.int("HevcColorBitDepth"), 8);
    assert_eq!(r.int("HevcNominalRange"), 0, "STUDIO");
    assert_eq!(r.int("HevcInColorProfile"), 1);
    assert_eq!(r.int("HevcOutColorProfile"), 1);
    assert_eq!(r.int("HevcInColorTransferChar"), 1);
    assert_eq!(r.int("HevcOutColorTransferChar"), 1);
    assert_eq!(r.int("HevcInColorPrimaries"), 1);
    assert_eq!(r.int("HevcOutColorPrimaries"), 1);
    assert!(!r.has("QPI") && !r.has("Usage"), "no AVC names");
    assert!(r.summary.contains("profile=Main "), "{}", r.summary);
}

/// H.265 Main 10 with HDR10 metadata, 4K60: profile 2, depth 10, BT.2020
/// profile, PQ transfer, BT.2020 primaries, full range, level 5.1 and its
/// 40 Mbit/s ceiling.
#[test]
fn hevc_main10_hdr_property_sequence() {
    let mut cfg = config(VideoCodec::H265, PixelFormat::Yuv420p10le);
    cfg.width = 3840;
    cfg.height = 2160;
    cfg.frame_rate = 60.0;
    cfg.color_metadata = ColorMetadata {
        transfer: TransferFn::St2084,
        matrix_coefficients: 9,
        colour_primaries: 9,
        full_range: true,
        mastering_display: None,
        content_light_level: None,
    };
    let r = run(apply_hevc_properties, &cfg);
    assert_eq!(r.int("HevcProfile"), 2, "AMF_VIDEO_ENCODER_HEVC_PROFILE_MAIN_10");
    assert_eq!(r.int("HevcColorBitDepth"), 10);
    assert_eq!(r.int("HevcProfileLevel"), 153, "AMF_LEVEL_5_1");
    assert_eq!(r.int("HevcPeakBitrate"), 40_000_000);
    assert_eq!(r.int("HevcNominalRange"), 1, "FULL");
    assert_eq!(r.int("HevcOutColorProfile"), 8, "FULL_2020");
    assert_eq!(r.int("HevcInColorProfile"), 8);
    assert_eq!(r.int("HevcOutColorTransferChar"), 16, "ST 2084 / PQ");
    assert_eq!(r.int("HevcOutColorPrimaries"), 9, "BT.2020");
    assert!(r.summary.contains("profile=Main10"), "{}", r.summary);
}

/// AV1, 1080p30 Standard/Standard: the corrected names and values.
#[test]
fn av1_property_sequence_matches_header() {
    let cfg = config(VideoCodec::Av1, PixelFormat::Yuv420p);
    let r = run(apply_av1_properties, &cfg);
    assert_eq!(r.order[0], "Av1Usage");
    assert_eq!(r.int("Av1Usage"), 0);
    assert_eq!(r.int("Av1RateControlMethod"), 4, "QUALITY_VBR is 4, not 5");
    assert_eq!(r.int("Av1QualityPreset"), 30, "QUALITY is 30");
    assert_eq!(r.int("Av1QIndex_Intra"), 120, "the tuning table's Standard q-index");
    assert_eq!(r.int("Av1QIndex_Inter"), 128);
    assert_eq!(r.int("Av1QvbrQualityLevel"), 30, "1-51 scale: q-index / 4");
    assert_eq!(r.rate("Av1FrameRate"), (30, 1));
    assert_eq!(r.int("Av1GOPSize"), 120);
    assert_eq!(r.int("Av1AQMode"), 1, "CAQ");
    assert_eq!(r.int("Av1NumTilesPerFrame"), 4, "2×2 at 1080p");
    assert_eq!(r.int("AV1OutputMode"), 0, "capital AV1 in this one name");
    assert_eq!(r.int("Av1ColorBitDepth"), 8);
    assert_eq!(r.int("Av1OutputColorProfile"), 1);
    assert_eq!(r.int("Av1OutputColorTransferChar"), 1);
    assert_eq!(r.int("Av1OutputColorPrimaries"), 1);
    assert_eq!(r.int("Av1InputColorPrimaries"), 1);
    assert!(r.has("Av1TargetBitrate") && r.has("Av1PeakBitrate") && r.has("Av1VBVBufferSize"));
    assert!(r.bool_("Av1EnforceHRD"));
    assert!(!r.has("Av1OutColorRange"), "no such property in the header");
}

/// AV1 with the legacy CQ escape hatch at 0 must not send q-index 0 (the
/// header's range is 1-255).
#[test]
fn av1_q_index_floor_is_one() {
    let mut cfg = config(VideoCodec::Av1, PixelFormat::Yuv420p);
    cfg.quality = 0;
    let r = run(apply_av1_properties, &cfg);
    assert_eq!(r.int("Av1QIndex_Intra"), 1);
    assert_eq!(r.int("Av1QvbrQualityLevel"), 1);
}

// ── End to end on this machine ────────────────────────────────

/// NAL unit types in an Annex-B access unit, in order.
fn annexb_nal_types(data: &[u8], hevc: bool) -> Vec<u8> {
    let mut out = Vec::new();
    let mut i = 0;
    while i + 3 <= data.len() {
        if data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1 {
            if let Some(&hdr) = data.get(i + 3) {
                out.push(if hevc { (hdr >> 1) & 0x3f } else { hdr & 0x1f });
            }
            i += 3;
        } else {
            i += 1;
        }
    }
    out
}

/// Encode synthetic frames through the real component on this machine's
/// AMD GPU: every pushed frame comes back as one Annex-B access unit with
/// the frame's own pts; frame 0, the GOP boundary and the frame after
/// `force_keyframe_next` are IDRs that carry their parameter sets in band
/// and are tagged as keyframes; the rest are P slices and are not. Skipped
/// (loudly) where no AMF-capable AMD GPU is present.
fn h26x_roundtrip_on_this_machine(codec: VideoCodec) {
    use crate::encode::Encoder;
    use crate::frame::{ColorSpace, VideoFrame};
    let (w, h) = (320u32, 240u32);
    let cfg = EncoderConfig {
        width: w,
        height: h,
        frame_rate: 30.0,
        codec,
        keyframe_interval: 12,
        ..Default::default()
    };
    let mut enc = match super::AmfEncoder::new(cfg, 0) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("SKIPPED ({codec:?}): no AMF-capable AMD GPU on this machine: {e:#}");
            return;
        }
    };
    let hevc = codec == VideoCodec::H265;
    let n = 30u64;
    let forced_at = 7u64;
    for i in 0..n {
        // A moving gradient, so P frames have something to predict.
        let mut data = vec![0u8; (w * h * 3 / 2) as usize];
        for y in 0..h as usize {
            for x in 0..w as usize {
                data[y * w as usize + x] = ((x + y + i as usize * 4) & 0xff) as u8;
            }
        }
        for c in data[(w * h) as usize..].iter_mut() {
            *c = 128;
        }
        let frame = VideoFrame::new(bytes::Bytes::from(data), w, h, PixelFormat::Yuv420p, ColorSpace::Bt709, i);
        if i == forced_at {
            enc.force_keyframe_next().unwrap();
        }
        enc.send_frame(&frame).unwrap();
    }
    enc.flush().unwrap();
    let mut packets = Vec::new();
    while let Some(p) = enc.receive_packet().unwrap() {
        packets.push(p);
    }
    assert_eq!(packets.len(), n as usize, "one access unit per frame, none lost at the flush");
    let (sps, pps, idr_types, p_types): (u8, u8, &[u8], &[u8]) = if hevc {
        (33, 34, &[19, 20], &[0, 1])
    } else {
        (7, 8, &[5], &[1])
    };
    for (i, p) in packets.iter().enumerate() {
        assert_eq!(p.pts, i as u64, "pts is the frame's own timestamp");
        let types = annexb_nal_types(&p.data, hevc);
        assert!(!types.is_empty(), "packet {i} is Annex-B");
        let expect_idr = i == 0 || i as u64 == forced_at || (i as u64).is_multiple_of(12);
        let has_idr = types.iter().any(|t| idr_types.contains(t));
        let has_ps = types.contains(&sps) && types.contains(&pps);
        assert_eq!(p.is_keyframe, expect_idr, "packet {i} keyframe tag; NALs {types:?}");
        assert_eq!(has_idr, expect_idr, "packet {i} IDR slice; NALs {types:?}");
        if expect_idr {
            assert!(has_ps, "packet {i} carries SPS+PPS in band; NALs {types:?}");
            if hevc {
                assert!(types.contains(&32), "packet {i} carries the VPS; NALs {types:?}");
            }
        } else {
            assert!(types.iter().any(|t| p_types.contains(t)), "packet {i} is a P slice; NALs {types:?}");
        }
    }
    eprintln!("{codec:?} roundtrip on this machine: {n} frames, IDR at 0/{forced_at}/12/24 with in-band parameter sets");
}

#[test]
fn h264_roundtrip_on_this_machine() {
    h26x_roundtrip_on_this_machine(VideoCodec::H264);
}

#[test]
fn h265_roundtrip_on_this_machine() {
    h26x_roundtrip_on_this_machine(VideoCodec::H265);
}
