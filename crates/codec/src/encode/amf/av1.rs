//! The AV1 component: `AMFVideoEncoderHW_AV1` property names and enum values
//! from `components/VideoEncoderAV1.h` (AMF SDK v1.4.36), and the
//! pre-`Init` property sequence for it.
//!
//! Every string and number here is copied from the header line cited beside
//! it. The names are case-sensitive on the runtime side (`SetProperty` on an
//! unknown name returns `AMF_NOT_FOUND`, and the encoder then runs on its
//! USAGE defaults), so a typo is a silent quality regression rather than an
//! error — hence the line-by-line citations.

use anyhow::Result;
use std::ffi::c_void;

use crate::encode::tuning::{self, AmfQualityPreset, AmfRateControl};
use crate::encode::{AUTO_FROM_TARGET, EncoderConfig};
use super::{
    AMF_COLOR_BIT_DEPTH_8, AMF_COLOR_BIT_DEPTH_10, CodecPlan, amf_color_bit_depth_for,
    amf_color_profile_for, frame_rate_rational, qvbr_bitrate_ceiling, set_int_property,
    set_rate_property, transfer_to_h273,
};

// ─── Component id ─────────────────────────────────────────────────

/// `AMFVideoEncoder_AV1` (`VideoEncoderAV1.h:35`).
pub(super) const AV1_COMPONENT_ID: &str = "AMFVideoEncoderHW_AV1";

// ─── Property names (VideoEncoderAV1.h) ───────────────────────────

/// `AMF_VIDEO_ENCODER_AV1_USAGE` (`:199`).
pub(super) const AV1_USAGE: &str = "Av1Usage";
/// `AMF_VIDEO_ENCODER_AV1_COLOR_BIT_DEPTH` (`:203`).
pub(super) const AV1_COLOR_BIT_DEPTH: &str = "Av1ColorBitDepth";
/// `AMF_VIDEO_ENCODER_AV1_TILES_PER_FRAME` (`:206`) — note the `Num`.
pub(super) const AV1_TILES_PER_FRAME: &str = "Av1NumTilesPerFrame";
/// `AMF_VIDEO_ENCODER_AV1_QUALITY_PRESET` (`:207`).
pub(super) const AV1_QUALITY_PRESET: &str = "Av1QualityPreset";
/// `AMF_VIDEO_ENCODER_AV1_RATE_CONTROL_METHOD` (`:218`).
pub(super) const AV1_RATE_CONTROL_METHOD: &str = "Av1RateControlMethod";
/// `AMF_VIDEO_ENCODER_AV1_QVBR_QUALITY_LEVEL` (`:219`), range 1-51.
pub(super) const AV1_QVBR_QUALITY_LEVEL: &str = "Av1QvbrQualityLevel";
/// `AMF_VIDEO_ENCODER_AV1_AQ_MODE` (`:228`).
pub(super) const AV1_AQ_MODE: &str = "Av1AQMode";
/// `AMF_VIDEO_ENCODER_AV1_OUTPUT_MODE` (`:245`) — capital `AV1`, unlike
/// every other AV1 property.
pub(super) const AV1_OUTPUT_MODE: &str = "AV1OutputMode";
/// `AMF_VIDEO_ENCODER_AV1_VBV_BUFFER_SIZE` (`:257`), bits.
pub(super) const AV1_VBV_BUFFER_SIZE: &str = "Av1VBVBufferSize";
/// `AMF_VIDEO_ENCODER_AV1_FRAMERATE` (`:258`), `AMFRate`.
pub(super) const AV1_FRAMERATE: &str = "Av1FrameRate";
/// `AMF_VIDEO_ENCODER_AV1_TARGET_BITRATE` (`:261`), bits/s.
pub(super) const AV1_TARGET_BITRATE: &str = "Av1TargetBitrate";
/// `AMF_VIDEO_ENCODER_AV1_PEAK_BITRATE` (`:262`), bits/s.
pub(super) const AV1_PEAK_BITRATE: &str = "Av1PeakBitrate";
/// `AMF_VIDEO_ENCODER_AV1_Q_INDEX_INTRA` (`:273`), range **1**-255.
pub(super) const AV1_Q_INDEX_INTRA: &str = "Av1QIndex_Intra";
/// `AMF_VIDEO_ENCODER_AV1_Q_INDEX_INTER` (`:274`), range 1-255.
pub(super) const AV1_Q_INDEX_INTER: &str = "Av1QIndex_Inter";
/// `AMF_VIDEO_ENCODER_AV1_GOP_SIZE` (`:281`).
pub(super) const AV1_GOP_SIZE: &str = "Av1GOPSize";
/// `AMF_VIDEO_ENCODER_AV1_INPUT_COLOR_PROFILE` (`:292`).
pub(super) const AV1_INPUT_COLOR_PROFILE: &str = "Av1InputColorProfile";
/// `AMF_VIDEO_ENCODER_AV1_INPUT_TRANSFER_CHARACTERISTIC` (`:293`).
pub(super) const AV1_INPUT_TRANSFER_CHAR: &str = "Av1InputColorTransferChar";
/// `AMF_VIDEO_ENCODER_AV1_INPUT_COLOR_PRIMARIES` (`:294`).
pub(super) const AV1_INPUT_COLOR_PRIMARIES: &str = "Av1InputColorPrimaries";
/// `AMF_VIDEO_ENCODER_AV1_OUTPUT_COLOR_PROFILE` (`:296`).
pub(super) const AV1_OUTPUT_COLOR_PROFILE: &str = "Av1OutputColorProfile";
/// `AMF_VIDEO_ENCODER_AV1_OUTPUT_TRANSFER_CHARACTERISTIC` (`:297`).
pub(super) const AV1_OUTPUT_TRANSFER_CHAR: &str = "Av1OutputColorTransferChar";
/// `AMF_VIDEO_ENCODER_AV1_OUTPUT_COLOR_PRIMARIES` (`:298`).
pub(super) const AV1_OUTPUT_COLOR_PRIMARIES: &str = "Av1OutputColorPrimaries";
/// `AMF_VIDEO_ENCODER_AV1_FORCE_FRAME_TYPE` (`:302`), per-surface.
pub(super) const AV1_FORCE_FRAME_TYPE: &str = "Av1ForceFrameType";
/// `AMF_VIDEO_ENCODER_AV1_OUTPUT_FRAME_TYPE` (`:313`), on the output buffer.
pub(super) const AV1_OUTPUT_FRAME_TYPE: &str = "Av1OutputFrameType";

// ─── Enum values (VideoEncoderAV1.h) ──────────────────────────────

/// `AMF_VIDEO_ENCODER_AV1_USAGE_TRANSCODING` (`:47`).
pub(super) const AV1_USAGE_TRANSCODING: i64 = 0;
/// `AMF_VIDEO_ENCODER_AV1_RATE_CONTROL_METHOD_CONSTANT_QP` (`:91`).
pub(super) const AV1_RC_CONSTANT_QP: i64 = 0;
/// `AMF_VIDEO_ENCODER_AV1_RATE_CONTROL_METHOD_QUALITY_VBR` (`:95`).
pub(super) const AV1_RC_QUALITY_VBR: i64 = 4;
/// `AMF_VIDEO_ENCODER_AV1_FORCE_FRAME_TYPE_KEY` (`:111`).
pub(super) const AV1_FORCE_FRAME_TYPE_KEY: i64 = 1;
/// `AMF_VIDEO_ENCODER_AV1_OUTPUT_FRAME_TYPE_KEY` (`:119`).
pub(super) const AV1_OUTPUT_FRAME_TYPE_KEY: i64 = 0;
/// `AMF_VIDEO_ENCODER_AV1_QUALITY_PRESET_*` (`:128-131`).
pub(super) const AV1_QUALITY_PRESET_HIGH_QUALITY: i64 = 0;
pub(super) const AV1_QUALITY_PRESET_QUALITY: i64 = 30;
pub(super) const AV1_QUALITY_PRESET_BALANCED: i64 = 70;
pub(super) const AV1_QUALITY_PRESET_SPEED: i64 = 100;
/// `AMF_VIDEO_ENCODER_AV1_AQ_MODE_CAQ` (`:163`).
pub(super) const AV1_AQ_MODE_CAQ: i64 = 1;
/// `AMF_VIDEO_ENCODER_AV1_OUTPUT_MODE_FRAME` (`:180`).
pub(super) const AV1_OUTPUT_MODE_FRAME: i64 = 0;

/// Map the tuning preset onto `AMF_VIDEO_ENCODER_AV1_QUALITY_PRESET_ENUM`.
/// Each AMF codec numbers its presets differently (AV1 `0/30/70/100`, AVC
/// `3/2/0/1`, HEVC `15/0/5/10`), so the mapping is per codec.
pub(super) fn av1_quality_preset(preset: AmfQualityPreset) -> i64 {
    match preset {
        AmfQualityPreset::HighQuality => AV1_QUALITY_PRESET_HIGH_QUALITY,
        AmfQualityPreset::Quality => AV1_QUALITY_PRESET_QUALITY,
        AmfQualityPreset::Balanced => AV1_QUALITY_PRESET_BALANCED,
        AmfQualityPreset::Speed => AV1_QUALITY_PRESET_SPEED,
    }
}

/// The codec plan `mod.rs` drives the shared session with.
pub(super) const AV1_PLAN: CodecPlan = CodecPlan {
    component_id: AV1_COMPONENT_ID,
    force_key: (AV1_FORCE_FRAME_TYPE, AV1_FORCE_FRAME_TYPE_KEY),
    key_extras: &[],
    output_type: AV1_OUTPUT_FRAME_TYPE,
    is_keyframe: |v| v == AV1_OUTPUT_FRAME_TYPE_KEY,
};

/// Every `SetProperty` for an AV1 session, before `Init`. Returns a short
/// summary for the "tuning applied" log line.
pub(super) unsafe fn apply_av1_properties(
    encoder: *mut c_void,
    config: &EncoderConfig,
) -> Result<String> {
    unsafe {
        let tp = tuning::amf_av1_params_with(
            config.target,
            config.tier,
            &tuning::RungContext::standalone(config.width, config.height),
            &config.overrides,
        );

        // Legacy quality override: a concrete `config.quality` is a 0..63
        // libaom-style CQ; AMF's q-index is the 0..255 AV1 scale (×4).
        let q_intra = if config.quality == AUTO_FROM_TARGET {
            tp.q_index_intra
        } else {
            ((config.quality as u32 * 4).min(255)) as u8
        };
        // The header's range is 1-255 (`:273-274`): 0 is rejected, not
        // "lossless".
        let q_intra = q_intra.max(1);
        let q_inter = q_intra.saturating_add(8).max(1);
        // QVBR quality level is a 1-51 CRF-like scale (`:219`, "default =
        // 23"), the same currency as the H.26x QP; the AV1 q-index is four
        // times that, so divide back down.
        let qvbr_level = i64::from(q_intra / 4).clamp(1, 51);

        let rc = if config.constant_qp {
            AmfRateControl::Cqp
        } else {
            tp.rc_mode
        };

        // Baseline: USAGE_TRANSCODING picks driver-tuned defaults, then
        // override every knob we care about so the behaviour does not drift
        // when AMD ships a driver that tweaks the USAGE preset internals.
        set_int_property(encoder, AV1_USAGE, AV1_USAGE_TRANSCODING)?;
        set_int_property(
            encoder,
            AV1_RATE_CONTROL_METHOD,
            match rc {
                AmfRateControl::Cqp => AV1_RC_CONSTANT_QP,
                AmfRateControl::QualityVbr => AV1_RC_QUALITY_VBR,
            },
        )?;
        set_int_property(encoder, AV1_QUALITY_PRESET, av1_quality_preset(tp.quality_preset))?;
        set_int_property(encoder, AV1_Q_INDEX_INTRA, i64::from(q_intra))?;
        set_int_property(encoder, AV1_Q_INDEX_INTER, i64::from(q_inter))?;
        let (fps_num, fps_den) = frame_rate_rational(config.frame_rate);
        set_rate_property(encoder, AV1_FRAMERATE, fps_num, fps_den)?;
        if rc == AmfRateControl::QualityVbr {
            set_int_property(encoder, AV1_QVBR_QUALITY_LEVEL, qvbr_level)?;
            // QVBR keeps quality *under* the bitrate constraints, so leaving
            // them at the USAGE default would cap every rung at whatever the
            // driver assumed. Give it a generous, resolution-scaled ceiling.
            let ceiling = qvbr_bitrate_ceiling(config.width, config.height, config.frame_rate, None);
            set_int_property(encoder, AV1_TARGET_BITRATE, ceiling)?;
            set_int_property(encoder, AV1_PEAK_BITRATE, ceiling)?;
            set_int_property(encoder, AV1_VBV_BUFFER_SIZE, ceiling)?;
        }
        set_int_property(
            encoder,
            AV1_GOP_SIZE,
            i64::from(super::effective_keyframe_interval(config.keyframe_interval)),
        )?;
        set_int_property(
            encoder,
            AV1_AQ_MODE,
            if tp.aq_mode != 0 { AV1_AQ_MODE_CAQ } else { 0 },
        )?;
        set_int_property(encoder, AV1_TILES_PER_FRAME, i64::from(tp.tiles_per_frame))?;
        // Frame-level output — one buffer per frame, which is what the MP4
        // muxer's AV1 sample builder expects.
        set_int_property(encoder, AV1_OUTPUT_MODE, AV1_OUTPUT_MODE_FRAME)?;

        // Bit depth + colour signalling. The bit-depth enum is the literal
        // depth (8 / 10, `ColorSpace.h:106-107`). Input and output profiles
        // are set to the same values so the runtime has no reason to insert
        // a colour conversion between them.
        let depth = amf_color_bit_depth_for(config.pixel_format);
        debug_assert!(depth == AMF_COLOR_BIT_DEPTH_8 || depth == AMF_COLOR_BIT_DEPTH_10);
        set_int_property(encoder, AV1_COLOR_BIT_DEPTH, depth)?;
        let cm = &config.color_metadata;
        let profile = amf_color_profile_for(cm.matrix_coefficients, cm.full_range);
        let transfer = transfer_to_h273(cm.transfer);
        let primaries = i64::from(cm.colour_primaries);
        set_int_property(encoder, AV1_INPUT_COLOR_PROFILE, profile)?;
        set_int_property(encoder, AV1_INPUT_TRANSFER_CHAR, transfer)?;
        set_int_property(encoder, AV1_INPUT_COLOR_PRIMARIES, primaries)?;
        set_int_property(encoder, AV1_OUTPUT_COLOR_PROFILE, profile)?;
        set_int_property(encoder, AV1_OUTPUT_TRANSFER_CHAR, transfer)?;
        set_int_property(encoder, AV1_OUTPUT_COLOR_PRIMARIES, primaries)?;

        Ok(format!(
            "q_index_intra={q_intra} q_index_inter={q_inter} qvbr_level={qvbr_level} rc={rc:?} \
             preset={:?} tiles={} depth={depth}",
            tp.quality_preset, tp.tiles_per_frame
        ))
    }
}
