//! Per-encoder parameter structs, rate-control enums, quality presets,
//! and associated constants.
//!
//! These are the concrete knob-sets that the adapter functions in
//! `adapters.rs` produce. Each encoder backend (rav1e, NVENC, AMF, QSV)
//! consumes the matching struct directly.

// ─── rav1e ───────────────────────────────────────────────────────

/// Concrete parameters for rav1e's `EncoderConfig`.
///
/// Consumed in `crates/codec/src/encode/rav1e_enc.rs::build_rav1e_config`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rav1eParams {
    /// rav1e quantizer: 0–255, lower = higher quality. Default 100.
    pub quantizer: usize,
    /// rav1e speed preset 0 (slowest/best) – 10 (fastest). Archive=4,
    /// Standard=6, Draft=8.
    pub speed_preset: u8,
    /// Number of tile rows (literal, not log2). Resolution-dependent.
    pub tile_rows: usize,
    /// Number of tile columns (literal). Resolution-dependent.
    pub tile_cols: usize,
}

// ─── NVENC ───────────────────────────────────────────────────────

/// Concrete parameters for NVENC AV1 (NV_ENC_CONFIG + NV_ENC_RC_PARAMS).
///
/// Consumed in `crates/codec/src/encode/nvenc.rs` when populating
/// `NV_ENC_INITIALIZE_PARAMS.encode_config` (currently null — see
/// `reviews/codec-review-3.md` issues 1-3).
///
/// GUID is returned as its raw 16-byte form so the caller can splat
/// it into the SDK's `#[repr(C)] Guid { data1: u32, data2: u16,
/// data3: u16, data4: [u8;8] }` without this module depending on the
/// FFI struct definitions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NvencAv1Params {
    /// Rate control mode. Values are the SDK constants
    /// `NV_ENC_PARAMS_RC_CONSTQP = 0`, `NV_ENC_PARAMS_RC_VBR = 1`,
    /// `NV_ENC_PARAMS_RC_CBR = 2`. We only emit CONSTQP (archive) or
    /// VBR+targetQuality (all other tiers) — CBR is never used by
    /// this service.
    pub rc_mode: NvencRateControl,
    /// AV1 CQ target (for VBR mode) or constant QP (for CONSTQP mode).
    /// Range 0–63 for AV1 (NOT 0-51 — that range is H.264/HEVC).
    pub cq: u8,
    /// Preset GUID raw bytes, ready to splat into a `#[repr(C)] Guid`.
    /// Order: data1 (4 bytes, u32 LE), data2 (2 bytes u16 LE),
    /// data3 (2 bytes u16 LE), data4 (8 raw bytes).
    pub preset_guid: [u8; 16],
    /// `NV_ENC_TUNING_INFO` — always `HIGH_QUALITY (1)` for this
    /// service; never low-latency.
    pub tuning_info: u32,
    /// Adaptive quantization strength 0–15. 0 disables AQ. ~8 is
    /// a reasonable default under HIGH_QUALITY tuning.
    pub aq_strength: u8,
    /// Lookahead depth (frames). 0 disables. Higher = better quality
    /// bias at cost of latency.
    pub lookahead_depth: u32,
    /// `NV_ENC_CONFIG_AV1.numTileColumns`.
    pub num_tile_columns: u32,
    /// `NV_ENC_CONFIG_AV1.numTileRows`.
    pub num_tile_rows: u32,
    /// `NV_ENC_CONFIG_AV1.outputAnnexBFormat`. Always 0 (LOB) for
    /// MP4 muxing — AV1-ISOBMFF requires `obu_has_size_field = 1`.
    pub output_annex_b_format: u32,
    /// `NV_ENC_CONFIG_AV1.repeatSeqHdr`. Always 1 so every IDR
    /// carries a sequence header for seeking.
    pub repeat_seq_hdr: u32,
}

/// NVENC rate control modes actually used by this service. The numeric
/// value matches the SDK's `NV_ENC_PARAMS_RC_MODE`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum NvencRateControl {
    /// `NV_ENC_PARAMS_RC_CONSTQP = 0`. Every frame gets the same QP.
    /// Strict archival mode — bitrate floats.
    ConstQp = 0,
    /// `NV_ENC_PARAMS_RC_VBR = 1` with `targetQuality` set. NVENC's
    /// CQ mode — quality-stable across content.
    VbrTargetQuality = 1,
}

// ─── AMF ─────────────────────────────────────────────────────────

/// Concrete parameters for AMD AMF AV1 (VCN on RDNA3+).
///
/// AMF is property-driven: every knob is set via
/// `AMFComponent::SetProperty(name, value)` using wide-string names
/// defined in `vendor/amd/VideoEncoderAV1.h`. The adapter emits integer
/// ranges that exactly match the property-value ranges the AMF runtime
/// accepts — out-of-range values return `AMF_INVALID_ARG`.
///
/// Consumed in `crates/codec/src/encode/amf.rs::AmfEncoder::new`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AmfAv1Params {
    /// `AMF_VIDEO_ENCODER_AV1_RATE_CONTROL_METHOD`. CQP for archive,
    /// QVBR (quality-VBR) for the common quality-target tiers.
    pub rc_mode: AmfRateControl,
    /// `AMF_VIDEO_ENCODER_AV1_Q_INDEX_INTRA`. AV1 QP index 0..255 (the
    /// full AV1 quantizer range — NOT 0..63; that's NVENC's scale).
    pub q_index_intra: u8,
    /// `AMF_VIDEO_ENCODER_AV1_Q_INDEX_INTER`. Usually +8 on intra so
    /// P-frames spend fewer bits.
    pub q_index_inter: u8,
    /// `AMF_VIDEO_ENCODER_AV1_QVBR_QUALITY_LEVEL`. 1..100 when
    /// rc_mode == `QualityVbr`; ignored for CQP. Higher = better.
    pub qvbr_quality: u8,
    /// `AMF_VIDEO_ENCODER_AV1_QUALITY_PRESET`. Lower = better quality.
    pub quality_preset: AmfQualityPreset,
    /// `AMF_VIDEO_ENCODER_AV1_GOP_SIZE`. Frames between keyframes.
    pub gop_size: u32,
    /// `AMF_VIDEO_ENCODER_AV1_AQ_MODE`. 0=off, 1=CAQ (content-adaptive).
    pub aq_mode: u32,
    /// `AMF_VIDEO_ENCODER_AV1_TILES_PER_FRAME`. AMF picks the grid;
    /// we specify the total. 1 tile at ≤1080p, 4 at 1080p+, 4 at 4K
    /// (VCN is less tile-parallel than rav1e — more tiles hurts HQ).
    pub tiles_per_frame: u32,
}

/// AMF AV1 quality presets. Values match `AMF_VIDEO_ENCODER_AV1_QUALITY_PRESET_*`
/// constants from the GPUOpen AMF wiki. Lower = better quality / more wall-clock.
/// The transcode service never picks `Speed` (same rationale as NVENC: no
/// low-latency presets in this service — see research §2.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i64)]
pub enum AmfQualityPreset {
    HighQuality = 10,
    Quality = 30,
    Balanced = 50,
    /// Not used by this service; kept in the enum so the mapping table
    /// stays complete for any future ultra-low-latency path.
    #[allow(dead_code)]
    Speed = 70,
}

/// AMF AV1 rate control modes actually used by this service.
/// Values match `AMF_VIDEO_ENCODER_AV1_RATE_CONTROL_METHOD_*`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i64)]
pub enum AmfRateControl {
    /// `AMF_VIDEO_ENCODER_AV1_RATE_CONTROL_METHOD_CQP = 1`. Every
    /// frame gets the same q-index. Archival.
    Cqp = 1,
    /// `AMF_VIDEO_ENCODER_AV1_RATE_CONTROL_METHOD_QUALITY_VBR = 5`.
    /// Quality-target VBR — bitrate floats to hit a perceptual level.
    QualityVbr = 5,
}

// ─── QSV ─────────────────────────────────────────────────────────

/// Concrete parameters for Intel QSV AV1 (oneVPL on Arc / Meteor Lake+).
///
/// oneVPL is struct-driven: `mfxVideoParam` carries every knob in fixed
/// fields (no property bag). The adapter produces the exact values we
/// splat into the struct in `crates/codec/src/encode/qsv.rs`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QsvAv1Params {
    /// `mfxVideoParam.mfx.RateControlMethod`. ICQ for the common
    /// quality targets; CQP for archive.
    pub rc_mode: QsvRateControl,
    /// `mfxVideoParam.mfx.ICQQuality` (ICQ mode) — 1..51 for AV1 per
    /// oneVPL 2.8+ dispatcher. 1=best, 51=worst. Mapped from libaom CQ.
    pub icq_quality: u16,
    /// `mfxVideoParam.mfx.QPI` (CQP mode) — AV1 q-index 0..255.
    pub qp_i: u16,
    /// `mfxVideoParam.mfx.QPP` (CQP mode) — inter-frame QP.
    pub qp_p: u16,
    /// `mfxVideoParam.mfx.TargetUsage`. 1=best quality, 7=best speed.
    pub target_usage: u16,
    /// `mfxVideoParam.mfx.GopPicSize`. Frames between keyframes.
    pub gop_pic_size: u16,
    /// Tile grid — `mfxExtAV1TileParam.NumTileColumns` / `NumTileRows`.
    pub num_tile_columns: u8,
    pub num_tile_rows: u8,
    /// `mfxVideoParam.mfx.LowPower`. Always
    /// `MFX_CODINGOPTION_OFF = 32` for this service — the low-power
    /// path on older Arc silicon has documented quality regressions;
    /// leaving it explicitly OFF sidesteps that.
    pub low_power: u16,
}

/// oneVPL tri-state option values (from `MFX_CODINGOPTION_*`).
/// Used for `LowPower` and a handful of other `mfxU16` toggles.
pub const MFX_CODINGOPTION_OFF: u16 = 32;
/// Not currently used but named so the value shows up next to `OFF`
/// whenever a future code path wants explicit on-switching.
#[allow(dead_code)]
pub const MFX_CODINGOPTION_ON: u16 = 16;

/// QSV AV1 rate control mode values match `MFX_RATECONTROL_*`
/// in `oneVPL/include/vpl/mfxstructs.h`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
pub enum QsvRateControl {
    /// `MFX_RATECONTROL_CQP = 3`.
    Cqp = 3,
    /// `MFX_RATECONTROL_ICQ = 8`. Intelligent constant quality — the
    /// QSV equivalent of CRF. Best match for a perceptual target.
    Icq = 8,
}
