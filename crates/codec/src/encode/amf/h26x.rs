//! The H.264 and H.265 components: `AMFVideoEncoderVCE_AVC` and
//! `AMFVideoEncoderHW_HEVC`, their property names and enum values from
//! `components/VideoEncoderVCE.h` and `components/VideoEncoderHEVC.h` (AMF
//! SDK v1.4.36), the level tables the two specs impose, and the pre-`Init`
//! property sequence for each.
//!
//! # What the two components share with the AV1 one
//!
//! The session flow in `mod.rs` — runtime, factory, context, component,
//! `Init(NV12|P010, w, h)`, per-frame `AllocSurface` → copy → `SubmitInput`
//! → `QueryOutput`, `Drain` — is codec-agnostic. What differs per codec is
//! the component id, the property names (AVC's are bare, HEVC's carry a
//! `Hevc` prefix, AV1's an `Av1` prefix), the enum numbering (each codec's
//! `QUALITY_PRESET` enum is numbered differently), the per-surface
//! "force an IDR" property and the output buffer's frame-type property. That
//! is the [`CodecPlan`] plus one `apply_*_properties` function per codec.
//!
//! # Output
//!
//! Both components emit **Annex-B** NAL streams (start-code delimited), one
//! access unit per output buffer in `OUTPUT_MODE_FRAME`
//! (`VideoEncoderVCE.h:172`, `VideoEncoderHEVC.h:141`), which is what the
//! muxer's `nal_mux` consumes: it captures the in-band SPS/PPS(/VPS) for the
//! `avcC`/`hvcC` box and length-prefixes the slices. Parameter sets travel
//! **in band**: every IDR this module forces (the first frame, every GOP
//! boundary, and every `force_keyframe_next`) also carries `InsertSPS` +
//! `InsertPPS` (AVC, `:289-290`) or `HevcInsertHeader` (HEVC, `:255`), and
//! HEVC's `HevcHeaderInsertionMode` is `IDR_ALIGNED` (`:117`) besides — so
//! every segment a packager cuts at an IDR is self-describing. The read-only
//! `ExtraData` / `HevcExtraData` property (`VideoEncoderVCE.h:187`,
//! `VideoEncoderHEVC.h:164`) is therefore not needed and not read.
//!
//! # Picture structure
//!
//! No B pictures (`BPicturesPattern = 0`, `VideoEncoderVCE.h:259`; the HEVC
//! component has none), so coding order is display order and a packet's
//! `pts` is the one its frame came in with — the same contract as every
//! other H.26x backend in this crate; the muxer carries no composition
//! offsets.
//!
//! # Rate control
//!
//! `VisuallyLossless` (and `ChunkSeamMode::ParallelConstQp`) use constant QP
//! (`RATE_CONTROL_METHOD_CONSTANT_QP = 0` in both headers). Everything else
//! uses `QUALITY_VBR = 4` with a `QvbrQualityLevel` on the same 1-51 scale as
//! the QP (`VideoEncoderVCE.h:204`, `VideoEncoderHEVC.h:181`: "default = 23;
//! range = 1-51" — a CRF-like level, lower is better). QVBR keeps quality
//! *within* the bitrate constraints, so the constraints are set explicitly
//! rather than left to the USAGE default: a resolution-scaled ceiling, capped
//! at what the chosen level allows (see [`qvbr_bitrate_ceiling`] and the
//! level tables), with a one-second VBV and the HRD enforced (without which
//! the peak is advisory — measured), filler data off.
//!
//! # Bit depth
//!
//! H.264 is 8-bit only (there is no `AMF_VIDEO_ENCODER_COLOR_BIT_DEPTH_10`
//! path for AVC in any AMD driver, and the `Profile` enum has no High 10);
//! H.265 takes 8-bit NV12 as Main and 10-bit P010 as Main 10
//! (`HevcProfile = 2`, `VideoEncoderHEVC.h:51`, with
//! `HevcColorBitDepth = 10`, `:200`). A 10-bit H.264 request is refused by
//! name, as it is on QSV.
//!
//! **Not verified on hardware.** The only AMD silicon on the development box
//! is the Ryzen desktop iGPU. What *is* proven: every vtable slot and struct
//! offset against the header (`ffi.rs`), every property name and enum value
//! by citation here, the parameter mapping and level tables by unit test,
//! the property-storage ABI (`SetProperty` / `GetProperty` / `HasProperty` /
//! `QueryInterface`) against the installed `amfrt64.dll`, and that a job on
//! this box fails through to the next tier with a clear log line.

use anyhow::{Result, bail};
use std::ffi::c_void;

use crate::encode::tuning::{self, AmfQualityPreset, AmfRateControl};
use crate::encode::{AUTO_FROM_TARGET, EncoderConfig};
use super::{
    AMF_COLOR_BIT_DEPTH_8, AMF_COLOR_BIT_DEPTH_10, CodecPlan, amf_color_bit_depth_for,
    amf_color_profile_for, frame_rate_rational, set_bool_property, set_int_property,
    set_rate_property, transfer_to_h273,
};
use crate::frame::{PixelFormat, VideoCodec};

// ─── Component ids ────────────────────────────────────────────────

/// `AMFVideoEncoderVCE_AVC` (`VideoEncoderVCE.h:45`).
pub(super) const AVC_COMPONENT_ID: &str = "AMFVideoEncoderVCE_AVC";
/// `AMFVideoEncoder_HEVC` (`VideoEncoderHEVC.h:35`).
pub(super) const HEVC_COMPONENT_ID: &str = "AMFVideoEncoderHW_HEVC";

// ─── AVC property names (VideoEncoderVCE.h) ───────────────────────

/// `AMF_VIDEO_ENCODER_USAGE` (`:188`).
pub(super) const AVC_USAGE: &str = "Usage";
/// `AMF_VIDEO_ENCODER_PROFILE` (`:189`).
pub(super) const AVC_PROFILE: &str = "Profile";
/// `AMF_VIDEO_ENCODER_PROFILE_LEVEL` (`:190`).
pub(super) const AVC_PROFILE_LEVEL: &str = "ProfileLevel";
/// `AMF_VIDEO_ENCODER_FULL_RANGE_COLOR` (`:198`), bool.
pub(super) const AVC_FULL_RANGE_COLOR: &str = "FullRangeColor";
/// `AMF_VIDEO_ENCODER_RATE_CONTROL_METHOD` (`:203`).
pub(super) const AVC_RATE_CONTROL_METHOD: &str = "RateControlMethod";
/// `AMF_VIDEO_ENCODER_QVBR_QUALITY_LEVEL` (`:204`), range 1-51.
pub(super) const AVC_QVBR_QUALITY_LEVEL: &str = "QvbrQualityLevel";
/// `AMF_VIDEO_ENCODER_QUALITY_PRESET` (`:212`).
pub(super) const AVC_QUALITY_PRESET: &str = "QualityPreset";
/// `AMF_VIDEO_ENCODER_COLOR_BIT_DEPTH` (`:215`).
pub(super) const AVC_COLOR_BIT_DEPTH: &str = "ColorBitDepth";
/// `AMF_VIDEO_ENCODER_INPUT_COLOR_PROFILE` (`:217`).
pub(super) const AVC_INPUT_COLOR_PROFILE: &str = "InColorProfile";
/// `AMF_VIDEO_ENCODER_INPUT_TRANSFER_CHARACTERISTIC` (`:218`).
pub(super) const AVC_INPUT_TRANSFER_CHAR: &str = "InColorTransferChar";
/// `AMF_VIDEO_ENCODER_INPUT_COLOR_PRIMARIES` (`:219`).
pub(super) const AVC_INPUT_COLOR_PRIMARIES: &str = "InColorPrimaries";
/// `AMF_VIDEO_ENCODER_OUTPUT_COLOR_PROFILE` (`:222`).
pub(super) const AVC_OUTPUT_COLOR_PROFILE: &str = "OutColorProfile";
/// `AMF_VIDEO_ENCODER_OUTPUT_TRANSFER_CHARACTERISTIC` (`:223`).
pub(super) const AVC_OUTPUT_TRANSFER_CHAR: &str = "OutColorTransferChar";
/// `AMF_VIDEO_ENCODER_OUTPUT_COLOR_PRIMARIES` (`:224`).
pub(super) const AVC_OUTPUT_COLOR_PRIMARIES: &str = "OutColorPrimaries";
/// `AMF_VIDEO_ENCODER_OUTPUT_MODE` (`:228`).
pub(super) const AVC_OUTPUT_MODE: &str = "OutputMode";
/// `AMF_VIDEO_ENCODER_FRAMERATE` (`:233`), `AMFRate`.
pub(super) const AVC_FRAMERATE: &str = "FrameRate";
/// `AMF_VIDEO_ENCODER_ENFORCE_HRD` (`:237`), bool.
pub(super) const AVC_ENFORCE_HRD: &str = "EnforceHRD";
/// `AMF_VIDEO_ENCODER_FILLER_DATA_ENABLE` (`:238`), bool.
pub(super) const AVC_FILLER_DATA_ENABLE: &str = "FillerDataEnable";
/// `AMF_VIDEO_ENCODER_VBV_BUFFER_SIZE` (`:243`), bits.
pub(super) const AVC_VBV_BUFFER_SIZE: &str = "VBVBufferSize";
/// `AMF_VIDEO_ENCODER_QP_I` / `QP_P` / `QP_B` (`:250-252`), range 0-51.
pub(super) const AVC_QP_I: &str = "QPI";
pub(super) const AVC_QP_P: &str = "QPP";
pub(super) const AVC_QP_B: &str = "QPB";
/// `AMF_VIDEO_ENCODER_TARGET_BITRATE` / `PEAK_BITRATE` (`:253-254`), bits/s.
pub(super) const AVC_TARGET_BITRATE: &str = "TargetBitrate";
pub(super) const AVC_PEAK_BITRATE: &str = "PeakBitrate";
/// `AMF_VIDEO_ENCODER_B_PIC_PATTERN` (`:259`): number of B frames.
pub(super) const AVC_B_PIC_PATTERN: &str = "BPicturesPattern";
/// `AMF_VIDEO_ENCODER_IDR_PERIOD` (`:262`), frames.
pub(super) const AVC_IDR_PERIOD: &str = "IDRPeriod";
/// `AMF_VIDEO_ENCODER_CABAC_ENABLE` (`:266`).
pub(super) const AVC_CABAC_ENABLE: &str = "CABACEnable";
/// `AMF_VIDEO_ENCODER_FORCE_PICTURE_TYPE` (`:287`), per-surface.
pub(super) const AVC_FORCE_PICTURE_TYPE: &str = "ForcePictureType";
/// `AMF_VIDEO_ENCODER_INSERT_SPS` / `INSERT_PPS` (`:289-290`), per-surface bools.
pub(super) const AVC_INSERT_SPS: &str = "InsertSPS";
pub(super) const AVC_INSERT_PPS: &str = "InsertPPS";
/// `AMF_VIDEO_ENCODER_OUTPUT_DATA_TYPE` (`:303`), on the output buffer.
pub(super) const AVC_OUTPUT_DATA_TYPE: &str = "OutputDataType";

// ─── AVC enum values (VideoEncoderVCE.h) ──────────────────────────

/// `AMF_VIDEO_ENCODER_USAGE_TRANSCODING` (`:51`).
pub(super) const AVC_USAGE_TRANSCODING: i64 = 0;
/// `AMF_VIDEO_ENCODER_PROFILE_HIGH` (`:64`).
pub(super) const AVC_PROFILE_HIGH: i64 = 100;
/// `AMF_VIDEO_ENCODER_RATE_CONTROL_METHOD_CONSTANT_QP` (`:101`).
pub(super) const AVC_RC_CONSTANT_QP: i64 = 0;
/// `AMF_VIDEO_ENCODER_RATE_CONTROL_METHOD_QUALITY_VBR` (`:105`; CQP, CBR,
/// PCVBR, LCVBR, QVBR — the AVC order, which differs from HEVC's).
pub(super) const AVC_RC_QUALITY_VBR: i64 = 4;
/// `AMF_VIDEO_ENCODER_QUALITY_PRESET_*` (`:112-115`).
pub(super) const AVC_QUALITY_PRESET_BALANCED: i64 = 0;
pub(super) const AVC_QUALITY_PRESET_SPEED: i64 = 1;
pub(super) const AVC_QUALITY_PRESET_QUALITY: i64 = 2;
pub(super) const AVC_QUALITY_PRESET_HIGH_QUALITY: i64 = 3;
/// `AMF_VIDEO_ENCODER_PICTURE_TYPE_IDR` (`:130`).
pub(super) const AVC_PICTURE_TYPE_IDR: i64 = 2;
/// `AMF_VIDEO_ENCODER_OUTPUT_DATA_TYPE_IDR` (`:138`).
pub(super) const AVC_OUTPUT_DATA_TYPE_IDR: i64 = 0;
/// `AMF_VIDEO_ENCODER_CABAC` (`:153`).
pub(super) const AVC_CODING_CABAC: i64 = 1;
/// `AMF_VIDEO_ENCODER_OUTPUT_MODE_FRAME` (`:172`).
pub(super) const AVC_OUTPUT_MODE_FRAME: i64 = 0;

// ─── HEVC property names (VideoEncoderHEVC.h) ─────────────────────

/// `AMF_VIDEO_ENCODER_HEVC_USAGE` (`:156`).
pub(super) const HEVC_USAGE: &str = "HevcUsage";
/// `AMF_VIDEO_ENCODER_HEVC_PROFILE` (`:157`).
pub(super) const HEVC_PROFILE: &str = "HevcProfile";
/// `AMF_VIDEO_ENCODER_HEVC_TIER` (`:158`).
pub(super) const HEVC_TIER: &str = "HevcTier";
/// `AMF_VIDEO_ENCODER_HEVC_PROFILE_LEVEL` (`:159`).
pub(super) const HEVC_PROFILE_LEVEL: &str = "HevcProfileLevel";
/// `AMF_VIDEO_ENCODER_HEVC_QUALITY_PRESET` (`:163`).
pub(super) const HEVC_QUALITY_PRESET: &str = "HevcQualityPreset";
/// `AMF_VIDEO_ENCODER_HEVC_NOMINAL_RANGE` (`:168`).
pub(super) const HEVC_NOMINAL_RANGE: &str = "HevcNominalRange";
/// `AMF_VIDEO_ENCODER_HEVC_NUM_GOPS_PER_IDR` (`:172`).
pub(super) const HEVC_NUM_GOPS_PER_IDR: &str = "HevcGOPSPerIDR";
/// `AMF_VIDEO_ENCODER_HEVC_GOP_SIZE` (`:173`).
pub(super) const HEVC_GOP_SIZE: &str = "HevcGOPSize";
/// `AMF_VIDEO_ENCODER_HEVC_HEADER_INSERTION_MODE` (`:176`).
pub(super) const HEVC_HEADER_INSERTION_MODE: &str = "HevcHeaderInsertionMode";
/// `AMF_VIDEO_ENCODER_HEVC_RATE_CONTROL_METHOD` (`:180`).
pub(super) const HEVC_RATE_CONTROL_METHOD: &str = "HevcRateControlMethod";
/// `AMF_VIDEO_ENCODER_HEVC_QVBR_QUALITY_LEVEL` (`:181`), range 1-51.
pub(super) const HEVC_QVBR_QUALITY_LEVEL: &str = "HevcQvbrQualityLevel";
/// `AMF_VIDEO_ENCODER_HEVC_VBV_BUFFER_SIZE` (`:182`), bits.
pub(super) const HEVC_VBV_BUFFER_SIZE: &str = "HevcVBVBufferSize";
/// `AMF_VIDEO_ENCODER_HEVC_ENFORCE_HRD` (`:218`), bool.
pub(super) const HEVC_ENFORCE_HRD: &str = "HevcEnforceHRD";
/// `AMF_VIDEO_ENCODER_HEVC_FILLER_DATA_ENABLE` (`:219`), bool.
pub(super) const HEVC_FILLER_DATA_ENABLE: &str = "HevcFillerDataEnable";
/// `AMF_VIDEO_ENCODER_HEVC_COLOR_BIT_DEPTH` (`:200`).
pub(super) const HEVC_COLOR_BIT_DEPTH: &str = "HevcColorBitDepth";
/// `AMF_VIDEO_ENCODER_HEVC_INPUT_COLOR_PROFILE` (`:202`).
pub(super) const HEVC_INPUT_COLOR_PROFILE: &str = "HevcInColorProfile";
/// `AMF_VIDEO_ENCODER_HEVC_INPUT_TRANSFER_CHARACTERISTIC` (`:203`).
pub(super) const HEVC_INPUT_TRANSFER_CHAR: &str = "HevcInColorTransferChar";
/// `AMF_VIDEO_ENCODER_HEVC_INPUT_COLOR_PRIMARIES` (`:204`).
pub(super) const HEVC_INPUT_COLOR_PRIMARIES: &str = "HevcInColorPrimaries";
/// `AMF_VIDEO_ENCODER_HEVC_OUTPUT_COLOR_PROFILE` (`:206`).
pub(super) const HEVC_OUTPUT_COLOR_PROFILE: &str = "HevcOutColorProfile";
/// `AMF_VIDEO_ENCODER_HEVC_OUTPUT_TRANSFER_CHARACTERISTIC` (`:207`).
pub(super) const HEVC_OUTPUT_TRANSFER_CHAR: &str = "HevcOutColorTransferChar";
/// `AMF_VIDEO_ENCODER_HEVC_OUTPUT_COLOR_PRIMARIES` (`:208`).
pub(super) const HEVC_OUTPUT_COLOR_PRIMARIES: &str = "HevcOutColorPrimaries";
/// `AMF_VIDEO_ENCODER_HEVC_OUTPUT_MODE` (`:211`).
pub(super) const HEVC_OUTPUT_MODE: &str = "HevcOutputMode";
/// `AMF_VIDEO_ENCODER_HEVC_FRAMERATE` (`:216`), `AMFRate`.
pub(super) const HEVC_FRAMERATE: &str = "HevcFrameRate";
/// `AMF_VIDEO_ENCODER_HEVC_TARGET_BITRATE` / `PEAK_BITRATE` (`:220-221`), bits/s.
pub(super) const HEVC_TARGET_BITRATE: &str = "HevcTargetBitrate";
pub(super) const HEVC_PEAK_BITRATE: &str = "HevcPeakBitrate";
/// `AMF_VIDEO_ENCODER_HEVC_QP_I` / `QP_P` (`:230-231`), range 0-51.
pub(super) const HEVC_QP_I: &str = "HevcQP_I";
pub(super) const HEVC_QP_P: &str = "HevcQP_P";
/// `AMF_VIDEO_ENCODER_HEVC_FORCE_PICTURE_TYPE` (`:253`), per-surface.
pub(super) const HEVC_FORCE_PICTURE_TYPE: &str = "HevcForcePictureType";
/// `AMF_VIDEO_ENCODER_HEVC_INSERT_HEADER` (`:255`), per-surface bool
/// ("insert header(SPS, PPS, VPS)").
pub(super) const HEVC_INSERT_HEADER: &str = "HevcInsertHeader";
/// `AMF_VIDEO_ENCODER_HEVC_OUTPUT_DATA_TYPE` (`:267`), on the output buffer.
pub(super) const HEVC_OUTPUT_DATA_TYPE: &str = "HevcOutputDataType";

// ─── HEVC enum values (VideoEncoderHEVC.h) ────────────────────────

/// `AMF_VIDEO_ENCODER_HEVC_USAGE_TRANSCODING` (`:40`).
pub(super) const HEVC_USAGE_TRANSCODING: i64 = 0;
/// `AMF_VIDEO_ENCODER_HEVC_PROFILE_MAIN` / `MAIN_10` (`:50-51`).
pub(super) const HEVC_PROFILE_MAIN: i64 = 1;
pub(super) const HEVC_PROFILE_MAIN_10: i64 = 2;
/// `AMF_VIDEO_ENCODER_HEVC_TIER_MAIN` (`:56`).
pub(super) const HEVC_TIER_MAIN: i64 = 0;
/// `AMF_VIDEO_ENCODER_HEVC_RATE_CONTROL_METHOD_CONSTANT_QP` (`:80`).
pub(super) const HEVC_RC_CONSTANT_QP: i64 = 0;
/// `AMF_VIDEO_ENCODER_HEVC_RATE_CONTROL_METHOD_QUALITY_VBR` (`:84`; CQP,
/// LCVBR, PCVBR, CBR, QVBR — the HEVC order).
pub(super) const HEVC_RC_QUALITY_VBR: i64 = 4;
/// `AMF_VIDEO_ENCODER_HEVC_PICTURE_TYPE_IDR` (`:93`).
pub(super) const HEVC_PICTURE_TYPE_IDR: i64 = 2;
/// `AMF_VIDEO_ENCODER_HEVC_OUTPUT_DATA_TYPE_IDR` (`:100`).
pub(super) const HEVC_OUTPUT_DATA_TYPE_IDR: i64 = 0;
/// `AMF_VIDEO_ENCODER_HEVC_QUALITY_PRESET_*` (`:107-110`).
pub(super) const HEVC_QUALITY_PRESET_QUALITY: i64 = 0;
pub(super) const HEVC_QUALITY_PRESET_BALANCED: i64 = 5;
pub(super) const HEVC_QUALITY_PRESET_SPEED: i64 = 10;
pub(super) const HEVC_QUALITY_PRESET_HIGH_QUALITY: i64 = 15;
/// `AMF_VIDEO_ENCODER_HEVC_HEADER_INSERTION_MODE_IDR_ALIGNED` (`:117`).
pub(super) const HEVC_HEADER_INSERTION_MODE_IDR_ALIGNED: i64 = 2;
/// `AMF_VIDEO_ENCODER_HEVC_NOMINAL_RANGE_STUDIO` / `FULL` (`:129-130`).
pub(super) const HEVC_NOMINAL_RANGE_STUDIO: i64 = 0;
pub(super) const HEVC_NOMINAL_RANGE_FULL: i64 = 1;
/// `AMF_VIDEO_ENCODER_HEVC_OUTPUT_MODE_FRAME` (`:141`).
pub(super) const HEVC_OUTPUT_MODE_FRAME: i64 = 0;

// ─── Preset mapping ───────────────────────────────────────────────

/// `AMF_VIDEO_ENCODER_QUALITY_PRESET_ENUM` value (`VideoEncoderVCE.h:112-115`).
pub(super) fn avc_quality_preset(preset: AmfQualityPreset) -> i64 {
    match preset {
        AmfQualityPreset::HighQuality => AVC_QUALITY_PRESET_HIGH_QUALITY,
        AmfQualityPreset::Quality => AVC_QUALITY_PRESET_QUALITY,
        AmfQualityPreset::Balanced => AVC_QUALITY_PRESET_BALANCED,
        AmfQualityPreset::Speed => AVC_QUALITY_PRESET_SPEED,
    }
}

/// `AMF_VIDEO_ENCODER_HEVC_QUALITY_PRESET_ENUM` value (`VideoEncoderHEVC.h:107-110`).
pub(super) fn hevc_quality_preset(preset: AmfQualityPreset) -> i64 {
    match preset {
        AmfQualityPreset::HighQuality => HEVC_QUALITY_PRESET_HIGH_QUALITY,
        AmfQualityPreset::Quality => HEVC_QUALITY_PRESET_QUALITY,
        AmfQualityPreset::Balanced => HEVC_QUALITY_PRESET_BALANCED,
        AmfQualityPreset::Speed => HEVC_QUALITY_PRESET_SPEED,
    }
}

// ─── Codec plans ──────────────────────────────────────────────────

pub(super) const AVC_PLAN: CodecPlan = CodecPlan {
    component_id: AVC_COMPONENT_ID,
    force_key: (AVC_FORCE_PICTURE_TYPE, AVC_PICTURE_TYPE_IDR),
    key_extras: &[AVC_INSERT_SPS, AVC_INSERT_PPS],
    output_type: AVC_OUTPUT_DATA_TYPE,
    is_keyframe: |v| v == AVC_OUTPUT_DATA_TYPE_IDR,
};

pub(super) const HEVC_PLAN: CodecPlan = CodecPlan {
    component_id: HEVC_COMPONENT_ID,
    force_key: (HEVC_FORCE_PICTURE_TYPE, HEVC_PICTURE_TYPE_IDR),
    key_extras: &[HEVC_INSERT_HEADER],
    output_type: HEVC_OUTPUT_DATA_TYPE,
    is_keyframe: |v| v == HEVC_OUTPUT_DATA_TYPE_IDR,
};

// ─── Levels ───────────────────────────────────────────────────────

/// H.264 levels 3.0 and up: `(AMF_H264_LEVEL__x, MaxMBPS, MaxFS, MaxBR kbit/s
/// for Main)`. Table A-1 of ITU-T H.264; the AMF enum value is `10 × level`
/// (`VideoEncoderVCE.h:78-89`). High profile's bitrate limit is 1.25 × Main
/// (`cpbBrVclFactor` 1250 vs 1000, H.264 A.3.1).
const H264_LEVELS: &[(i64, u64, u64, u64)] = &[
    (30, 40_500, 1_620, 10_000),
    (31, 108_000, 3_600, 14_000),
    (32, 216_000, 5_120, 20_000),
    (40, 245_760, 8_192, 20_000),
    (41, 245_760, 8_192, 50_000),
    (42, 522_240, 8_704, 50_000),
    (50, 589_824, 22_080, 135_000),
    (51, 983_040, 36_864, 240_000),
    (52, 2_073_600, 36_864, 240_000),
    (60, 4_177_920, 139_264, 240_000),
    (61, 8_355_840, 139_264, 480_000),
    (62, 16_711_680, 139_264, 800_000),
];

/// H.265 levels 3.0 and up: `(AMF_LEVEL_x, MaxLumaPs, MaxLumaSr, MaxBR
/// kbit/s for the Main tier)`. Tables A.8 / A.9 of ITU-T H.265; the AMF enum
/// value is `30 × level` (`VideoEncoderHEVC.h:65-74`).
const H265_LEVELS: &[(i64, u64, u64, u64)] = &[
    (90, 552_960, 16_588_800, 6_000),
    (93, 983_040, 33_177_600, 10_000),
    (120, 2_228_224, 66_846_720, 12_000),
    (123, 2_228_224, 133_693_440, 20_000),
    (150, 8_912_896, 267_386_880, 25_000),
    (153, 8_912_896, 534_773_760, 40_000),
    (156, 8_912_896, 1_069_547_520, 60_000),
    (180, 35_651_584, 1_069_547_520, 60_000),
    (183, 35_651_584, 2_139_095_040, 120_000),
    (186, 35_651_584, 4_278_190_080, 240_000),
];

/// The level a picture size and rate need, and that level's bitrate ceiling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Level {
    /// The value for `ProfileLevel` / `HevcProfileLevel`.
    pub(super) amf_value: i64,
    /// The level's maximum bitrate for the profile we ask for, bits/s.
    pub(super) max_bitrate: u64,
}

/// The lowest H.264 level (High profile) whose frame-size and macroblock-rate
/// limits admit `width × height` at `fps`; the top level if none does (the
/// encoder will then reject or clamp, which is the honest outcome).
pub(super) fn h264_level_for(width: u32, height: u32, fps: f64) -> Level {
    let mbs = u64::from(width.div_ceil(16)) * u64::from(height.div_ceil(16));
    let mbps = (mbs as f64 * fps.max(1.0)).ceil() as u64;
    let (idc, _, _, kbps) = H264_LEVELS
        .iter()
        .copied()
        .find(|&(_, max_mbps, max_fs, _)| mbs <= max_fs && mbps <= max_mbps)
        .unwrap_or(*H264_LEVELS.last().expect("non-empty level table"));
    Level {
        amf_value: idc,
        // High profile: 1.25 × the Main-profile figure.
        max_bitrate: kbps * 1000 * 5 / 4,
    }
}

/// The lowest H.265 level (Main tier) whose luma picture-size and
/// sample-rate limits admit `width × height` at `fps`; the top level if none
/// does.
pub(super) fn h265_level_for(width: u32, height: u32, fps: f64) -> Level {
    let luma_ps = u64::from(width) * u64::from(height);
    let luma_sr = (luma_ps as f64 * fps.max(1.0)).ceil() as u64;
    let (amf, _, _, kbps) = H265_LEVELS
        .iter()
        .copied()
        .find(|&(_, max_ps, max_sr, _)| luma_ps <= max_ps && luma_sr <= max_sr)
        .unwrap_or(*H265_LEVELS.last().expect("non-empty level table"));
    Level {
        amf_value: amf,
        max_bitrate: kbps * 1000,
    }
}

/// The bitrate ceiling handed to QVBR as target, peak and one-second VBV:
/// a quarter of a bit per pixel per second (1080p30 → 15.5 Mbit/s, 4K60 →
/// 124 Mbit/s), never below 2 Mbit/s, and never above what `level_cap` (the
/// chosen level's maximum) allows. QVBR rarely approaches it at the QPs the
/// targets map to; it is there so the USAGE default cannot silently cap a
/// rung. By review — the constant has not been swept on hardware.
pub(super) fn qvbr_bitrate_ceiling(width: u32, height: u32, fps: f64, level_cap: Option<u64>) -> i64 {
    let fps = if fps.is_finite() && fps > 0.0 { fps } else { 30.0 };
    let per_second = f64::from(width) * f64::from(height) * fps * 0.25;
    let mut bits = (per_second.round() as u64).max(2_000_000);
    if let Some(cap) = level_cap {
        bits = bits.min(cap);
    }
    i64::try_from(bits).unwrap_or(i64::MAX)
}

// ─── Property sequences ───────────────────────────────────────────

/// The resolved quantiser set for one H.26x session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct H26xQuant {
    pub(super) rc: AmfRateControl,
    pub(super) qp_i: u8,
    pub(super) qp_p: u8,
    pub(super) qvbr_level: u8,
    pub(super) preset: AmfQualityPreset,
}

/// Fold the tuning adapter and the legacy CRF escape hatch into one
/// quantiser set. `config.quality` is already in this codec's 0..51 currency
/// (and `resolve_overrides` has applied any per-rung delta to it), so, as in
/// `h26x_sw.rs` and the QSV path, it replaces the derived QP outright.
pub(super) fn h26x_quant(config: &EncoderConfig) -> H26xQuant {
    let tp = tuning::amf_h26x_params_with(config.codec, config.target, config.tier, &config.overrides);
    let (qp_i, qp_p, qvbr_level) = if config.quality == AUTO_FROM_TARGET {
        (tp.qp_i, tp.qp_p, tp.qvbr_quality)
    } else {
        let crf = config.quality.min(51);
        (crf, crf.saturating_add(2).min(51), crf.max(1))
    };
    H26xQuant {
        rc: if config.constant_qp { AmfRateControl::Cqp } else { tp.rc_mode },
        qp_i,
        qp_p,
        qvbr_level: qvbr_level.clamp(1, 51),
        preset: tp.quality_preset,
    }
}

/// Refuse the (codec, pixel format) pairs the components cannot take, by
/// name, before any AMF call is made.
pub(super) fn check_h26x_format(codec: VideoCodec, fmt: PixelFormat) -> Result<()> {
    match (codec, fmt) {
        (VideoCodec::H264, PixelFormat::Yuv420p) => Ok(()),
        (VideoCodec::H264, other) => bail!(
            "AMF H.264 (VCE_AVC) encodes 8-bit 4:2:0 only, got {other:?}; H.264 is 8-bit on \
             every backend — for 10-bit output use H.265 (Main 10) or AV1"
        ),
        (VideoCodec::H265, PixelFormat::Yuv420p | PixelFormat::Yuv420p10le) => Ok(()),
        (VideoCodec::H265, other) => bail!(
            "AMF H.265 encodes Yuv420p (Main) or Yuv420p10le (Main 10), got {other:?}"
        ),
        (VideoCodec::Av1, _) => bail!("not an H.26x codec"),
    }
}

/// Every `SetProperty` for an H.264 session, before `Init`. Returns a short
/// summary for the "tuning applied" log line.
pub(super) unsafe fn apply_avc_properties(encoder: *mut c_void, config: &EncoderConfig) -> Result<String> {
    unsafe {
        check_h26x_format(VideoCodec::H264, config.pixel_format)?;
        let q = h26x_quant(config);
        let level = h264_level_for(config.width, config.height, config.frame_rate);
        let (fps_num, fps_den) = frame_rate_rational(config.frame_rate);
        let gop = i64::from(super::effective_keyframe_interval(config.keyframe_interval));

        // USAGE first: it "fully configures parameter set" (`:188`), so
        // everything after it is an override that survives a driver that
        // re-tunes the preset internals.
        set_int_property(encoder, AVC_USAGE, AVC_USAGE_TRANSCODING)?;
        set_int_property(encoder, AVC_PROFILE, AVC_PROFILE_HIGH)?;
        set_int_property(encoder, AVC_PROFILE_LEVEL, level.amf_value)?;
        set_int_property(encoder, AVC_QUALITY_PRESET, avc_quality_preset(q.preset))?;
        set_int_property(encoder, AVC_CABAC_ENABLE, AVC_CODING_CABAC)?;
        set_rate_property(encoder, AVC_FRAMERATE, fps_num, fps_den)?;
        // No B pictures: display order is coding order (see the module doc).
        set_int_property(encoder, AVC_B_PIC_PATTERN, 0)?;
        set_int_property(encoder, AVC_IDR_PERIOD, gop)?;
        set_int_property(encoder, AVC_OUTPUT_MODE, AVC_OUTPUT_MODE_FRAME)?;

        // Rate control.
        let ceiling = qvbr_bitrate_ceiling(config.width, config.height, config.frame_rate, Some(level.max_bitrate));
        match q.rc {
            AmfRateControl::Cqp => {
                set_int_property(encoder, AVC_RATE_CONTROL_METHOD, AVC_RC_CONSTANT_QP)?;
            }
            AmfRateControl::QualityVbr => {
                set_int_property(encoder, AVC_RATE_CONTROL_METHOD, AVC_RC_QUALITY_VBR)?;
                set_int_property(encoder, AVC_QVBR_QUALITY_LEVEL, i64::from(q.qvbr_level))?;
                set_int_property(encoder, AVC_TARGET_BITRATE, ceiling)?;
                set_int_property(encoder, AVC_PEAK_BITRATE, ceiling)?;
                set_int_property(encoder, AVC_VBV_BUFFER_SIZE, ceiling)?;
                // The ceiling is only a ceiling with the HRD enforced
                // (measured on the Ryzen iGPU: without it a 720p Main 10
                // QVBR encode ran at 17 Mbit/s past a 6.9 Mbit/s peak). No
                // filler: a quiet scene may stay under the target.
                set_bool_property(encoder, AVC_ENFORCE_HRD, true)?;
                set_bool_property(encoder, AVC_FILLER_DATA_ENABLE, false)?;
            }
        }
        // The QPs are set in both modes: CQP uses them directly, and QVBR's
        // default QP set is what it starts from.
        set_int_property(encoder, AVC_QP_I, i64::from(q.qp_i))?;
        set_int_property(encoder, AVC_QP_P, i64::from(q.qp_p))?;
        set_int_property(encoder, AVC_QP_B, i64::from(q.qp_p))?;

        // Colour. 8-bit only for AVC (`check_h26x_format`); the VUI gets the
        // range through the bool and the matrix through the profile.
        set_int_property(encoder, AVC_COLOR_BIT_DEPTH, AMF_COLOR_BIT_DEPTH_8)?;
        let cm = &config.color_metadata;
        let profile = amf_color_profile_for(cm.matrix_coefficients, cm.full_range);
        let transfer = transfer_to_h273(cm.transfer);
        let primaries = i64::from(cm.colour_primaries);
        set_bool_property(encoder, AVC_FULL_RANGE_COLOR, cm.full_range)?;
        set_int_property(encoder, AVC_INPUT_COLOR_PROFILE, profile)?;
        set_int_property(encoder, AVC_INPUT_TRANSFER_CHAR, transfer)?;
        set_int_property(encoder, AVC_INPUT_COLOR_PRIMARIES, primaries)?;
        set_int_property(encoder, AVC_OUTPUT_COLOR_PROFILE, profile)?;
        set_int_property(encoder, AVC_OUTPUT_TRANSFER_CHAR, transfer)?;
        set_int_property(encoder, AVC_OUTPUT_COLOR_PRIMARIES, primaries)?;

        Ok(format!(
            "profile=High level={} qp_i={} qp_p={} qvbr_level={} rc={:?} preset={:?} ceiling_bps={ceiling} gop={gop}",
            level.amf_value, q.qp_i, q.qp_p, q.qvbr_level, q.rc, q.preset
        ))
    }
}

/// Every `SetProperty` for an H.265 session, before `Init`.
pub(super) unsafe fn apply_hevc_properties(encoder: *mut c_void, config: &EncoderConfig) -> Result<String> {
    unsafe {
        check_h26x_format(VideoCodec::H265, config.pixel_format)?;
        let q = h26x_quant(config);
        let level = h265_level_for(config.width, config.height, config.frame_rate);
        let (fps_num, fps_den) = frame_rate_rational(config.frame_rate);
        let gop = i64::from(super::effective_keyframe_interval(config.keyframe_interval));
        let depth = amf_color_bit_depth_for(config.pixel_format);
        let profile = if depth == AMF_COLOR_BIT_DEPTH_10 { HEVC_PROFILE_MAIN_10 } else { HEVC_PROFILE_MAIN };

        set_int_property(encoder, HEVC_USAGE, HEVC_USAGE_TRANSCODING)?;
        set_int_property(encoder, HEVC_PROFILE, profile)?;
        set_int_property(encoder, HEVC_TIER, HEVC_TIER_MAIN)?;
        set_int_property(encoder, HEVC_PROFILE_LEVEL, level.amf_value)?;
        set_int_property(encoder, HEVC_QUALITY_PRESET, hevc_quality_preset(q.preset))?;
        set_rate_property(encoder, HEVC_FRAMERATE, fps_num, fps_den)?;
        // One GOP per IDR, GOP = the keyframe interval: an IDR opens every
        // GOP, and parameter sets precede every IDR.
        set_int_property(encoder, HEVC_GOP_SIZE, gop)?;
        set_int_property(encoder, HEVC_NUM_GOPS_PER_IDR, 1)?;
        set_int_property(encoder, HEVC_HEADER_INSERTION_MODE, HEVC_HEADER_INSERTION_MODE_IDR_ALIGNED)?;
        set_int_property(encoder, HEVC_OUTPUT_MODE, HEVC_OUTPUT_MODE_FRAME)?;

        let ceiling = qvbr_bitrate_ceiling(config.width, config.height, config.frame_rate, Some(level.max_bitrate));
        match q.rc {
            AmfRateControl::Cqp => {
                set_int_property(encoder, HEVC_RATE_CONTROL_METHOD, HEVC_RC_CONSTANT_QP)?;
            }
            AmfRateControl::QualityVbr => {
                set_int_property(encoder, HEVC_RATE_CONTROL_METHOD, HEVC_RC_QUALITY_VBR)?;
                set_int_property(encoder, HEVC_QVBR_QUALITY_LEVEL, i64::from(q.qvbr_level))?;
                set_int_property(encoder, HEVC_TARGET_BITRATE, ceiling)?;
                set_int_property(encoder, HEVC_PEAK_BITRATE, ceiling)?;
                set_int_property(encoder, HEVC_VBV_BUFFER_SIZE, ceiling)?;
                set_bool_property(encoder, HEVC_ENFORCE_HRD, true)?;
                set_bool_property(encoder, HEVC_FILLER_DATA_ENABLE, false)?;
            }
        }
        set_int_property(encoder, HEVC_QP_I, i64::from(q.qp_i))?;
        set_int_property(encoder, HEVC_QP_P, i64::from(q.qp_p))?;

        // Colour + depth. The range is its own enum here (`:129-130`) rather
        // than AVC's bool; the matrix travels in the profile as for AVC.
        set_int_property(encoder, HEVC_COLOR_BIT_DEPTH, depth)?;
        let cm = &config.color_metadata;
        let color_profile = amf_color_profile_for(cm.matrix_coefficients, cm.full_range);
        let transfer = transfer_to_h273(cm.transfer);
        let primaries = i64::from(cm.colour_primaries);
        set_int_property(
            encoder,
            HEVC_NOMINAL_RANGE,
            if cm.full_range { HEVC_NOMINAL_RANGE_FULL } else { HEVC_NOMINAL_RANGE_STUDIO },
        )?;
        set_int_property(encoder, HEVC_INPUT_COLOR_PROFILE, color_profile)?;
        set_int_property(encoder, HEVC_INPUT_TRANSFER_CHAR, transfer)?;
        set_int_property(encoder, HEVC_INPUT_COLOR_PRIMARIES, primaries)?;
        set_int_property(encoder, HEVC_OUTPUT_COLOR_PROFILE, color_profile)?;
        set_int_property(encoder, HEVC_OUTPUT_TRANSFER_CHAR, transfer)?;
        set_int_property(encoder, HEVC_OUTPUT_COLOR_PRIMARIES, primaries)?;

        Ok(format!(
            "profile={} level={} depth={depth} qp_i={} qp_p={} qvbr_level={} rc={:?} preset={:?} ceiling_bps={ceiling} gop={gop}",
            if profile == HEVC_PROFILE_MAIN_10 { "Main10" } else { "Main" },
            level.amf_value,
            q.qp_i,
            q.qp_p,
            q.qvbr_level,
            q.rc,
            q.preset
        ))
    }
}
