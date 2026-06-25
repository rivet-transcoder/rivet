//! AV1 encoder tuning adapter.
//!
//! Translates a single backend-agnostic perceptual quality target into
//! per-encoder parameters so identical inputs yield visually consistent
//! output across rav1e, NVENC AV1, and future backends (SVT-AV1, AMF,
//! QSV).
//!
//! See `docs/av1-tuning-research.md` for the source tables and
//! `docs/av1-tuning-methodology.md` for how to re-calibrate when a new
//! encoder lands.
//!
//! # Design
//!
//! The user picks two things:
//! 1. A `QualityTarget` — perceptual goal expressed in VMAF/SSIMULACRA2
//!    bands, not encoder-native CRF. Every backend must reach roughly
//!    the same VMAF for a given target (±2 VMAF band).
//! 2. A `SpeedTier` — how much wallclock to spend getting there. Maps
//!    to encoder-native speed presets.
//!
//! The adapter also takes `(width, height)` because tile grid and
//! lookahead sizing depend on frame size.

// ─── Public types ────────────────────────────────────────────────

/// A single perceptual quality target, backend-agnostic.
///
/// Maps to VMAF / SSIMULACRA2 bands, NOT encoder CRF values:
///
/// | Variant             | Target VMAF | Target SSIMULACRA2 | Use case                         |
/// |---------------------|:-----------:|:------------------:|----------------------------------|
/// | `VisuallyLossless`  | ~98         | ~90                | Archive, master                  |
/// | `High`              | ~95         | ~80                | Premium OTT / top ABR rung       |
/// | `Standard`          | ~90         | ~70                | Default web / streaming          |
/// | `Low`               | ~85         | ~60                | Mobile / bandwidth-constrained   |
/// | `Vmaf(u8)`          | explicit    | n/a                | A/B testing escape hatch         |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum QualityTarget {
    VisuallyLossless,
    High,
    #[default]
    Standard,
    Low,
    Vmaf(u8),
}

/// User-facing speed tier — maps to encoder-native speed presets.
///
/// | Variant    | rav1e | NVENC preset | SVT-AV1 preset | libaom cpu-used |
/// |------------|:-----:|:------------:|:--------------:|:---------------:|
/// | `Draft`    | 8     | P5           | 12             | 8               |
/// | `Standard` | 6     | P6           | 8              | 6               |
/// | `Archive`  | 4     | P7           | 4              | 4               |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SpeedTier {
    Draft,
    #[default]
    Standard,
    Archive,
}

// ─── Per-encoder parameter structs ───────────────────────────────

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

// NVENC preset GUIDs from Video Codec SDK 12.2 headers. Bytes are the
// raw #[repr(C)] serialization: u32 LE, u16 LE, u16 LE, [u8;8].
//
// Only P5, P6, P7 are exposed — the transcode service has no use for
// the low-latency presets P1-P4.

// NVENC SDK 13.0 preset GUIDs (vendor/nvidia/nvEncodeAPI.h:226-251).
//
// CRITICAL: SDK 12.2 had ENTIRELY DIFFERENT preset GUIDs for P5/P6/P7
// — when we vendored SDK 13's nvEncodeAPI.h on 2026-05-01 we updated
// the NvEncFunctionList ordering + struct layouts but missed that the
// preset-GUID values themselves were also reshuffled. Sending SDK
// 12.2 P5/P6/P7 GUIDs to a SDK 13 driver returned NV_ENC_ERR_UNSUPPORTED_PARAM
// (rc=12) from NvEncGetEncodePresetConfigEx (the driver doesn't
// recognise the old GUIDs and rejects the lookup). For reference, the
// 12.2 → 13 GUID rotation:
//   P5: d0918ee2-a509-4681-af96-e9c3c45b7aa7 → 21c6e6b4-297a-4cba-998f-b6cbde72ade3
//   P6: fc8ebf15-6e19-47b4-8ea7-b1917f379eed → 8e75c279-6299-4ab6-8302-0b215a335cf5
//   P7: 84bdda58-33cb-4895-a372-ddeddb013ac4 → 84848c12-6f71-4c13-931b-53e283f57974
const NV_ENC_PRESET_P5_GUID_BYTES: [u8; 16] = [
    0xb4, 0xe6, 0xc6, 0x21, // data1 = 0x21c6e6b4
    0x7a, 0x29, // data2 = 0x297a
    0xba, 0x4c, // data3 = 0x4cba
    0x99, 0x8f, 0xb6, 0xcb, 0xde, 0x72, 0xad, 0xe3,
];

const NV_ENC_PRESET_P6_GUID_BYTES: [u8; 16] = [
    0x79, 0xc2, 0x75, 0x8e, // data1 = 0x8e75c279
    0x99, 0x62, // data2 = 0x6299
    0xb6, 0x4a, // data3 = 0x4ab6
    0x83, 0x02, 0x0b, 0x21, 0x5a, 0x33, 0x5c, 0xf5,
];

const NV_ENC_PRESET_P7_GUID_BYTES: [u8; 16] = [
    0x12, 0x8c, 0x84, 0x84, // data1 = 0x84848c12
    0x71, 0x6f, // data2 = 0x6f71
    0x13, 0x4c, // data3 = 0x4c13
    0x93, 0x1b, 0x53, 0xe2, 0x83, 0xf5, 0x79, 0x74,
];

/// SDK constant `NV_ENC_TUNING_INFO_HIGH_QUALITY = 1`.
pub const NVENC_TUNING_HIGH_QUALITY: u32 = 1;

// ─── Adapter functions ───────────────────────────────────────────

/// Derive rav1e params for a given quality target + speed tier +
/// resolution.
pub fn rav1e_params(
    target: QualityTarget,
    tier: SpeedTier,
    width: u32,
    height: u32,
) -> Rav1eParams {
    // rav1e quantizer ≈ 4 × libaom cq-level (well-known rule of thumb;
    // see docs/av1-tuning-research.md §2.3).
    let libaom_cq = libaom_cq_for_target(target);
    let quantizer = (libaom_cq as usize) * 4;

    let speed_preset = match tier {
        SpeedTier::Archive => 4,
        SpeedTier::Standard => 6,
        SpeedTier::Draft => 8,
    };

    // rav1e has high per-tile overhead and benefits from parallelism;
    // use the generous tile grid at 4K (4x4 = 16 tiles).
    let (tile_cols, tile_rows) = tile_grid_rav1e(width, height);

    Rav1eParams {
        quantizer,
        speed_preset,
        tile_rows,
        tile_cols,
    }
}

/// Derive NVENC AV1 params for a given quality target + speed tier +
/// resolution.
pub fn nvenc_av1_params(
    target: QualityTarget,
    tier: SpeedTier,
    width: u32,
    height: u32,
) -> NvencAv1Params {
    // Calibrated CQ values: NVENC AV1 needs ~3-4 lower CQ to hit the
    // same VMAF as libaom, compensating for its lower compression
    // efficiency. See research §2.4.
    let cq = nvenc_cq_for_target(target);

    let (preset_guid, lookahead_depth, aq_strength) = match tier {
        SpeedTier::Archive => (NV_ENC_PRESET_P7_GUID_BYTES, 32, 10),
        SpeedTier::Standard => (NV_ENC_PRESET_P6_GUID_BYTES, 16, 8),
        SpeedTier::Draft => (NV_ENC_PRESET_P5_GUID_BYTES, 0, 6),
    };

    // Archive tier uses CONSTQP for reproducible bitstreams; every
    // other tier uses VBR with targetQuality so bitrate floats by
    // content complexity.
    let rc_mode = match target {
        QualityTarget::VisuallyLossless => NvencRateControl::ConstQp,
        _ => NvencRateControl::VbrTargetQuality,
    };

    // NVENC AV1 HQ tuning: fewer tiles = better compression because
    // tile boundaries break loop-filter continuity and AV1 tiles are
    // independently entropy-coded. Published measurements show ~0.6%
    // VMAF loss at 2 tiles, ~1.3% at 4+ tiles on libaom; NVENC HQ
    // exhibits the same scaling. NVENC has enough internal parallelism
    // that it doesn't need 16-tile grids for throughput the way rav1e
    // does — cap at 2x2 even at 4K.
    //   Reference: research §3 and
    //   https://streaminglearningcenter.com/codecs/av1-encoding-and-4k.html
    let (num_tile_columns, num_tile_rows) = tile_grid_nvenc(width, height);

    NvencAv1Params {
        rc_mode,
        cq,
        preset_guid,
        tuning_info: NVENC_TUNING_HIGH_QUALITY,
        aq_strength,
        lookahead_depth,
        num_tile_columns: num_tile_columns as u32,
        num_tile_rows: num_tile_rows as u32,
        output_annex_b_format: 0, // LOB for MP4
        repeat_seq_hdr: 1,
    }
}

/// Derive AMD AMF AV1 params for a given quality target + speed tier +
/// resolution.
///
/// AMF's AV1 q-index scale is 0..255 (the full AV1 quantizer range, not
/// the NVENC-style 0..63 CQ band). Start point is rav1e's `4 × libaom_cq`
/// rule, then apply an 8-point calibration shift down to compensate for
/// VCN's documented compression-efficiency gap vs libaom (same goughlui
/// study that calibrated NVENC's 3-4-point CQ shift tested AMF VCN and
/// reported an analogous ~2-point CQ-equivalent shift; 2 points × 4 ≈ 8
/// in the 0..255 space).
///
/// TODO(calibrate): replace these seed anchors with calibrated values
/// once av1-tuning-eng runs the offline VMAF pass on RDNA3 hardware.
/// See `docs/av1-tuning-research.md` §2.5 for the calibration protocol.
pub fn amf_av1_params(
    target: QualityTarget,
    tier: SpeedTier,
    width: u32,
    height: u32,
) -> AmfAv1Params {
    let q_index_intra = amf_q_index_for_target(target);
    // Inter-frames get a slightly higher QP so P/B frames spend fewer
    // bits — biases bit allocation toward keyframes, which matches how
    // rav1e and NVENC CONSTQP mode behave.
    let q_index_inter = q_index_intra.saturating_add(8);

    // QVBR quality 1..100; higher = better. Map our VMAF-band targets
    // to the AMF-native band: VL=95, High=85, Standard=70, Low=55.
    let qvbr_quality = match target {
        QualityTarget::VisuallyLossless => 95,
        QualityTarget::High => 85,
        QualityTarget::Standard => 70,
        QualityTarget::Low => 55,
        QualityTarget::Vmaf(v) => vmaf_to_qvbr_quality(v),
    };

    // AMF quality preset per SpeedTier. Archive → HighQuality (best
    // but slowest), Standard → Quality, Draft → Balanced. `Speed`
    // preset deliberately unused — same rule as NVENC's P1-P4
    // exclusion (see research §2.4: no low-latency tunings for batch
    // transcode).
    let quality_preset = match tier {
        SpeedTier::Archive => AmfQualityPreset::HighQuality,
        SpeedTier::Standard => AmfQualityPreset::Quality,
        SpeedTier::Draft => AmfQualityPreset::Balanced,
    };

    // CQP for archival-lossless runs (reproducible bitstream); QVBR
    // for everything else — matches the NVENC branch structure.
    let rc_mode = match target {
        QualityTarget::VisuallyLossless => AmfRateControl::Cqp,
        _ => AmfRateControl::QualityVbr,
    };

    // AMF VCN tile parallelism is similar to NVENC — fewer tiles =
    // better compression. Share the NVENC 2×2 cap via `tile_grid_hw`
    // (both are "HQ-equivalent HW encoders that don't need aggressive
    // tiling for throughput"). Total tiles = cols × rows; at 1×1 that's
    // one, at 2×2 that's 4.
    let (tile_cols, tile_rows) = tile_grid_hw(width, height);
    let tiles_per_frame = (tile_cols * tile_rows) as u32;

    AmfAv1Params {
        rc_mode,
        q_index_intra,
        q_index_inter,
        qvbr_quality,
        quality_preset,
        gop_size: 0, // caller fills from keyframe_interval
        aq_mode: 1,  // CAQ — content-adaptive QP on
        tiles_per_frame,
    }
}

/// Derive Intel QSV AV1 params for a given quality target + speed tier +
/// resolution.
///
/// oneVPL exposes two sensible modes for quality-driven encoding: ICQ
/// (intelligent constant quality, 1..51 for AV1 — 1=best) and CQP
/// (constant q-index, 0..255). ICQ is the default; CQP is the archival
/// path. ICQ quality maps near-linearly to libaom cq-level at the range
/// we care about (research §2.6, calibrated from Intel's public
/// oneVPL sample_encode benchmarks).
pub fn qsv_av1_params(
    target: QualityTarget,
    tier: SpeedTier,
    width: u32,
    height: u32,
) -> QsvAv1Params {
    // ICQ quality 1..51; 1=best. QSV maps AV1's native 0..63 CQ range
    // into the 0..51 scale for API parity with H.264/HEVC (oneVPL
    // idiosyncrasy), so we scale libaom cq-level by 51/63 ≈ 0.81.
    //   VL: libaom 20 × 51/63 ≈ 16
    //   Hi: libaom 27 × 51/63 ≈ 22
    //   Std: libaom 32 × 51/63 ≈ 26
    //   Low: libaom 38 × 51/63 ≈ 31
    let icq_quality = match target {
        QualityTarget::VisuallyLossless => 16,
        QualityTarget::High => 22,
        QualityTarget::Standard => 26,
        QualityTarget::Low => 31,
        QualityTarget::Vmaf(v) => vmaf_to_qsv_icq(v),
    };
    // CQP q-index for archival — QSV uses the full AV1 0..255 range
    // via `mfx.QPI`. Same 4× libaom mapping as rav1e/AMF.
    let libaom_cq = libaom_cq_for_target(target);
    let qp_i = (libaom_cq as u16 * 4).min(255);
    let qp_p = qp_i.saturating_add(8).min(255);

    // oneVPL TargetUsage: 1=best quality, 7=best speed. Per
    // av1-tuning-eng review: Archive=1, Standard=4, Draft=6
    // (not 7 — 6 still leaves headroom for the driver's
    // "adaptive speed" selections without falling into the explicit
    // "worst-quality" bucket).
    let target_usage = match tier {
        SpeedTier::Archive => 1,
        SpeedTier::Standard => 4,
        SpeedTier::Draft => 6,
    };

    let rc_mode = match target {
        QualityTarget::VisuallyLossless => QsvRateControl::Cqp,
        _ => QsvRateControl::Icq,
    };

    let (num_tile_columns, num_tile_rows) = tile_grid_hw(width, height);

    QsvAv1Params {
        rc_mode,
        icq_quality,
        qp_i,
        qp_p,
        target_usage,
        gop_pic_size: 0, // caller fills from keyframe_interval
        num_tile_columns: num_tile_columns as u8,
        num_tile_rows: num_tile_rows as u8,
        low_power: MFX_CODINGOPTION_OFF,
    }
}

// ─── Internal helpers ────────────────────────────────────────────

/// libaom `cq-level` that corresponds to a given QualityTarget. libaom
/// is the cross-encoder reference: we equalize other encoders *to*
/// libaom's VMAF at each CQ.
///
/// Exposed `pub` so the FFmpeg-wrapper encoder path
/// (`encode::ffmpeg_enc`) can route `libsvtav1` / `libaom-av1` through
/// the same adapter tables as the native encoders.
pub fn libaom_cq_for_target(target: QualityTarget) -> u8 {
    match target {
        QualityTarget::VisuallyLossless => 20,
        QualityTarget::High => 27,
        QualityTarget::Standard => 32,
        QualityTarget::Low => 38,
        QualityTarget::Vmaf(v) => vmaf_to_libaom_cq(v),
    }
}

/// NVENC CQ that hits the same VMAF as `libaom_cq_for_target`, per
/// the research doc §2.4.
fn nvenc_cq_for_target(target: QualityTarget) -> u8 {
    match target {
        QualityTarget::VisuallyLossless => 19,
        QualityTarget::High => 25,
        QualityTarget::Standard => 30,
        QualityTarget::Low => 36,
        QualityTarget::Vmaf(v) => vmaf_to_nvenc_cq(v),
    }
}

/// Anchor points for libaom VMAF↔cq-level (research §2.1). Must stay
/// in descending VMAF order; `piecewise_cq` below depends on it.
const LIBAOM_ANCHORS: &[(i32, i32)] = &[
    (100, 10), // asymptote beyond VisuallyLossless
    (98, 20),
    (95, 27),
    (90, 32),
    (85, 38),
    (70, 55), // low-quality extrapolation
];

/// Anchor points for NVENC AV1 VMAF↔CQ. Calibrated down from libaom
/// to compensate for NVENC's documented compression-efficiency gap
/// (research §2.4). Same VMAF → lower CQ than libaom.
const NVENC_ANCHORS: &[(i32, i32)] = &[(100, 10), (98, 19), (95, 25), (90, 30), (85, 36), (70, 52)];

/// Piecewise-linear interpolation between anchors. Anchors are
/// `(vmaf, cq)` pairs in descending VMAF order. Out-of-range VMAF
/// values clamp to the nearest anchor's CQ.
fn piecewise_cq(vmaf: u8, anchors: &[(i32, i32)]) -> u8 {
    let v = vmaf as i32;
    // Above the top anchor: return its CQ (asymptote).
    if v >= anchors[0].0 {
        return anchors[0].1.clamp(0, 63) as u8;
    }
    // Below the bottom anchor: return its CQ.
    let last = anchors.len() - 1;
    if v <= anchors[last].0 {
        return anchors[last].1.clamp(0, 63) as u8;
    }
    // Linear interpolation between surrounding anchors.
    for pair in anchors.windows(2) {
        let (v_hi, cq_hi) = pair[0];
        let (v_lo, cq_lo) = pair[1];
        if v <= v_hi && v >= v_lo {
            let span = v_hi - v_lo;
            if span == 0 {
                return cq_hi.clamp(0, 63) as u8;
            }
            let t = v_hi - v; // 0 at high anchor, span at low anchor
            let cq = cq_hi + (cq_lo - cq_hi) * t / span;
            return cq.clamp(0, 63) as u8;
        }
    }
    anchors[last].1.clamp(0, 63) as u8
}

fn vmaf_to_libaom_cq(vmaf: u8) -> u8 {
    piecewise_cq(vmaf, LIBAOM_ANCHORS)
}

fn vmaf_to_nvenc_cq(vmaf: u8) -> u8 {
    piecewise_cq(vmaf, NVENC_ANCHORS)
}

/// Tile grid for rav1e (CPU). Returns `(columns, rows)`, literal counts.
/// rav1e is memory-bandwidth-limited and benefits from aggressive tiling
/// even at the cost of a small quality hit, because tile parallelism is
/// most of its throughput story at 4K+.
fn tile_grid_rav1e(width: u32, height: u32) -> (usize, usize) {
    let max_dim = width.max(height);
    if max_dim >= 3840 {
        (4, 4) // 16 tiles at 4K — rav1e fans out across cores
    } else if max_dim >= 1920 {
        (2, 2)
    } else {
        (1, 1)
    }
}

/// Tile grid for NVENC AV1. Returns `(columns, rows)`. NVENC has enough
/// internal parallelism that it does not need large tile grids for
/// throughput — and its HIGH_QUALITY tuning is sensitive to the ~1%
/// quality cost per extra tile row/column (tile boundaries break loop
/// filter continuity, and AV1 tiles are entropy-coded independently).
/// Cap at 2×2 even at 4K.
fn tile_grid_nvenc(width: u32, height: u32) -> (usize, usize) {
    let max_dim = width.max(height);
    if max_dim >= 1920 { (2, 2) } else { (1, 1) }
}

/// Shared HW-encoder tile grid. Used by NVENC, AMF, and QSV — all
/// three are "HQ-equivalent hardware encoders" that don't need rav1e's
/// aggressive tiling for throughput and are sensitive to the ~1%
/// quality cost per extra tile row/column. Cap at 2×2 even at 4K.
///
/// This is an alias over `tile_grid_nvenc` so the shared rule is
/// explicit at call sites. Changing the shared cap is a one-line
/// change here.
fn tile_grid_hw(width: u32, height: u32) -> (usize, usize) {
    tile_grid_nvenc(width, height)
}

/// AMF CQP q-index (0..255) for a given QualityTarget. Starts from
/// `libaom_cq × 4` and subtracts an 8-point calibration shift to
/// compensate for VCN's compression-efficiency gap — analogous to
/// NVENC's 3-4-point CQ shift in 0..63 space.
///
/// TODO(calibrate): replace with anchors from the offline VMAF pass
/// on RDNA3 hardware. Seed values come from av1-tuning-eng's research
/// doc §2.5 and GPUOpen AMF tuning guide.
fn amf_q_index_for_target(target: QualityTarget) -> u8 {
    let base = match target {
        QualityTarget::VisuallyLossless => 72, // libaom 20 × 4 - 8
        QualityTarget::High => 100,            // libaom 27 × 4 - 8
        QualityTarget::Standard => 120,        // libaom 32 × 4 - 8
        QualityTarget::Low => 144,             // libaom 38 × 4 - 8
        QualityTarget::Vmaf(v) => vmaf_to_amf_q_index(v),
    };
    base.min(255) as u8
}

/// Anchors for AMF q-index interpolation when a caller passes an
/// explicit Vmaf target. Descending VMAF → ascending q-index.
const AMF_Q_INDEX_ANCHORS: &[(i32, i32)] = &[
    (100, 50), // asymptote below VisuallyLossless
    (98, 72),
    (95, 100),
    (90, 120),
    (85, 144),
    (70, 200),
];

fn vmaf_to_amf_q_index(vmaf: u8) -> u16 {
    piecewise_quality(vmaf, AMF_Q_INDEX_ANCHORS, 0, 255) as u16
}

/// AMF anchors: AMF's QVBR quality scale is 1..100 (higher = better).
/// Calibrated from research §2.5 against libaom at matched VMAF.
const AMF_QVBR_ANCHORS: &[(i32, i32)] =
    &[(100, 100), (98, 95), (95, 85), (90, 70), (85, 55), (70, 35)];

fn vmaf_to_qvbr_quality(vmaf: u8) -> u8 {
    piecewise_quality(vmaf, AMF_QVBR_ANCHORS, 1, 100)
}

/// QSV ICQ scale is 1..51 (lower = better), inverted from AMF's QVBR.
/// Anchor table reflects Intel's public oneVPL sample benchmarks.
const QSV_ICQ_ANCHORS: &[(i32, i32)] =
    &[(100, 8), (98, 18), (95, 24), (90, 30), (85, 36), (70, 48)];

fn vmaf_to_qsv_icq(vmaf: u8) -> u16 {
    piecewise_quality(vmaf, QSV_ICQ_ANCHORS, 1, 51) as u16
}

/// Generic piecewise-linear interpolator for non-CQ scales. Mirrors
/// `piecewise_cq` but with configurable clamp bounds so the same logic
/// serves AMF's 1..100 and QSV's 1..51.
fn piecewise_quality(vmaf: u8, anchors: &[(i32, i32)], lo: i32, hi: i32) -> u8 {
    let v = vmaf as i32;
    if v >= anchors[0].0 {
        return anchors[0].1.clamp(lo, hi) as u8;
    }
    let last = anchors.len() - 1;
    if v <= anchors[last].0 {
        return anchors[last].1.clamp(lo, hi) as u8;
    }
    for pair in anchors.windows(2) {
        let (v_hi, q_hi) = pair[0];
        let (v_lo, q_lo) = pair[1];
        if v <= v_hi && v >= v_lo {
            let span = v_hi - v_lo;
            if span == 0 {
                return q_hi.clamp(lo, hi) as u8;
            }
            let t = v_hi - v;
            let q = q_hi + (q_lo - q_hi) * t / span;
            return q.clamp(lo, hi) as u8;
        }
    }
    anchors[last].1.clamp(lo, hi) as u8
}

// ─── Unit tests ──────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const RESOLUTIONS: &[(u32, u32)] = &[
        (640, 360),   // 360p — single tile
        (854, 480),   // 480p — single tile
        (1280, 720),  // 720p — single tile
        (1920, 1080), // 1080p — 2x2
        (2560, 1440), // 1440p — 2x2
        (3840, 2160), // 4K — 4x4
    ];

    const TARGETS: &[QualityTarget] = &[
        QualityTarget::VisuallyLossless,
        QualityTarget::High,
        QualityTarget::Standard,
        QualityTarget::Low,
    ];

    const TIERS: &[SpeedTier] = &[SpeedTier::Draft, SpeedTier::Standard, SpeedTier::Archive];

    #[test]
    fn rav1e_every_combination_returns_valid_params() {
        for (w, h) in RESOLUTIONS {
            for target in TARGETS {
                for tier in TIERS {
                    let p = rav1e_params(*target, *tier, *w, *h);
                    assert!(p.quantizer <= 255, "quantizer {} oob", p.quantizer);
                    assert!(p.speed_preset <= 10, "speed_preset {} oob", p.speed_preset);
                    assert!(p.tile_rows >= 1);
                    assert!(p.tile_cols >= 1);
                }
            }
        }
    }

    #[test]
    fn nvenc_every_combination_returns_valid_params() {
        for (w, h) in RESOLUTIONS {
            for target in TARGETS {
                for tier in TIERS {
                    let p = nvenc_av1_params(*target, *tier, *w, *h);
                    assert!(p.cq <= 63, "cq {} exceeds AV1 max 63", p.cq);
                    assert_eq!(p.tuning_info, NVENC_TUNING_HIGH_QUALITY);
                    assert_eq!(p.output_annex_b_format, 0, "must be LOB for MP4");
                    assert_eq!(p.repeat_seq_hdr, 1, "every IDR needs seq hdr");
                    assert!(p.aq_strength <= 15);
                }
            }
        }
    }

    #[test]
    fn rav1e_quantizer_monotonic_in_quality() {
        // Higher-quality targets must produce lower (stricter) quantizer.
        let sd1080 = (1920, 1080);
        let vl = rav1e_params(
            QualityTarget::VisuallyLossless,
            SpeedTier::Standard,
            sd1080.0,
            sd1080.1,
        );
        let hi = rav1e_params(QualityTarget::High, SpeedTier::Standard, sd1080.0, sd1080.1);
        let std = rav1e_params(
            QualityTarget::Standard,
            SpeedTier::Standard,
            sd1080.0,
            sd1080.1,
        );
        let lo = rav1e_params(QualityTarget::Low, SpeedTier::Standard, sd1080.0, sd1080.1);
        assert!(vl.quantizer < hi.quantizer);
        assert!(hi.quantizer < std.quantizer);
        assert!(std.quantizer < lo.quantizer);
    }

    #[test]
    fn nvenc_cq_monotonic_in_quality() {
        let sd = (1920, 1080);
        let vl = nvenc_av1_params(
            QualityTarget::VisuallyLossless,
            SpeedTier::Standard,
            sd.0,
            sd.1,
        );
        let hi = nvenc_av1_params(QualityTarget::High, SpeedTier::Standard, sd.0, sd.1);
        let std = nvenc_av1_params(QualityTarget::Standard, SpeedTier::Standard, sd.0, sd.1);
        let lo = nvenc_av1_params(QualityTarget::Low, SpeedTier::Standard, sd.0, sd.1);
        assert!(vl.cq < hi.cq);
        assert!(hi.cq < std.cq);
        assert!(std.cq < lo.cq);
    }

    #[test]
    fn rav1e_speed_preset_monotonic_in_tier() {
        let vl = QualityTarget::Standard;
        let (w, h) = (1920, 1080);
        let arc = rav1e_params(vl, SpeedTier::Archive, w, h);
        let std = rav1e_params(vl, SpeedTier::Standard, w, h);
        let drf = rav1e_params(vl, SpeedTier::Draft, w, h);
        // Faster tiers -> higher preset number in rav1e.
        assert!(arc.speed_preset < std.speed_preset);
        assert!(std.speed_preset < drf.speed_preset);
    }

    #[test]
    fn tile_grid_rav1e_by_resolution() {
        assert_eq!(tile_grid_rav1e(640, 360), (1, 1));
        assert_eq!(tile_grid_rav1e(1280, 720), (1, 1));
        assert_eq!(tile_grid_rav1e(1920, 1080), (2, 2));
        assert_eq!(tile_grid_rav1e(2560, 1440), (2, 2));
        assert_eq!(tile_grid_rav1e(3840, 2160), (4, 4));
        assert_eq!(tile_grid_rav1e(4096, 2160), (4, 4));
        // Portrait 1080x1920 still deserves tiling — use max dim.
        assert_eq!(tile_grid_rav1e(1080, 1920), (2, 2));
    }

    #[test]
    fn tile_grid_nvenc_caps_at_2x2() {
        // NVENC HQ prefers fewer tiles — no 4x4 even at 4K.
        assert_eq!(tile_grid_nvenc(640, 360), (1, 1));
        assert_eq!(tile_grid_nvenc(1280, 720), (1, 1));
        assert_eq!(tile_grid_nvenc(1920, 1080), (2, 2));
        assert_eq!(tile_grid_nvenc(2560, 1440), (2, 2));
        assert_eq!(tile_grid_nvenc(3840, 2160), (2, 2));
        assert_eq!(tile_grid_nvenc(4096, 2160), (2, 2));
        assert_eq!(tile_grid_nvenc(1080, 1920), (2, 2));
    }

    #[test]
    fn archive_tier_uses_constqp_at_lossless() {
        let p = nvenc_av1_params(
            QualityTarget::VisuallyLossless,
            SpeedTier::Archive,
            1920,
            1080,
        );
        assert_eq!(p.rc_mode, NvencRateControl::ConstQp);
    }

    #[test]
    fn non_archive_tiers_use_vbr_cq() {
        for target in [
            QualityTarget::High,
            QualityTarget::Standard,
            QualityTarget::Low,
        ] {
            for tier in TIERS {
                let p = nvenc_av1_params(target, *tier, 1920, 1080);
                assert_eq!(
                    p.rc_mode,
                    NvencRateControl::VbrTargetQuality,
                    "target={:?} tier={:?} should use VBR+CQ",
                    target,
                    tier
                );
            }
        }
    }

    #[test]
    fn vmaf_escape_hatch_matches_named_targets() {
        // VMAF 98 should map to roughly VisuallyLossless's CQ.
        let vl = nvenc_cq_for_target(QualityTarget::VisuallyLossless);
        let v98 = nvenc_cq_for_target(QualityTarget::Vmaf(98));
        assert!(
            (vl as i32 - v98 as i32).abs() <= 2,
            "VMAF 98 escape hatch CQ={} should be within 2 of named VL CQ={}",
            v98,
            vl
        );

        // VMAF 90 should map near Standard's CQ.
        let std = nvenc_cq_for_target(QualityTarget::Standard);
        let v90 = nvenc_cq_for_target(QualityTarget::Vmaf(90));
        assert!((std as i32 - v90 as i32).abs() <= 2);
    }

    #[test]
    fn vmaf_escape_hatch_clamps_oob() {
        // 0 and 255 shouldn't panic; clamp to valid CQ range.
        let lo_cq = nvenc_cq_for_target(QualityTarget::Vmaf(0));
        let hi_cq = nvenc_cq_for_target(QualityTarget::Vmaf(255));
        assert!(lo_cq <= 63);
        assert!(hi_cq <= 63);
        // Low VMAF target -> high CQ. High VMAF target -> low CQ.
        assert!(lo_cq > hi_cq);
    }

    #[test]
    fn preset_guids_are_distinct() {
        assert_ne!(NV_ENC_PRESET_P5_GUID_BYTES, NV_ENC_PRESET_P6_GUID_BYTES);
        assert_ne!(NV_ENC_PRESET_P6_GUID_BYTES, NV_ENC_PRESET_P7_GUID_BYTES);
        assert_ne!(NV_ENC_PRESET_P5_GUID_BYTES, NV_ENC_PRESET_P7_GUID_BYTES);
    }

    #[test]
    fn rav1e_quantizer_matches_libaom_4x_rule() {
        // docs rule: rav1e quantizer ≈ 4 × libaom cq-level.
        let p = rav1e_params(QualityTarget::High, SpeedTier::Standard, 1920, 1080);
        assert_eq!(p.quantizer, 27 * 4); // libaom cq-level for High = 27
        let p = rav1e_params(QualityTarget::Standard, SpeedTier::Standard, 1920, 1080);
        assert_eq!(p.quantizer, 32 * 4);
    }

    #[test]
    fn default_quality_is_standard() {
        let q: QualityTarget = Default::default();
        assert_eq!(q, QualityTarget::Standard);
        let t: SpeedTier = Default::default();
        assert_eq!(t, SpeedTier::Standard);
    }

    #[test]
    fn amf_every_combination_returns_valid_params() {
        for (w, h) in RESOLUTIONS {
            for target in TARGETS {
                for tier in TIERS {
                    let p = amf_av1_params(*target, *tier, *w, *h);
                    // AV1 QP range is 0..255 — q_index_inter uses
                    // saturating add so it never wraps.
                    // q_index fields are u8, so the <=255 bound is
                    // structurally guaranteed — the meaningful check is
                    // that inter is at least as large as intra.
                    assert!(p.q_index_inter >= p.q_index_intra);
                    assert!((1..=100).contains(&p.qvbr_quality));
                    assert!(p.tiles_per_frame >= 1);
                    // Speed preset is not used by this service — any
                    // combination must stay in the HighQuality..Balanced
                    // band.
                    assert!(matches!(
                        p.quality_preset,
                        AmfQualityPreset::HighQuality
                            | AmfQualityPreset::Quality
                            | AmfQualityPreset::Balanced
                    ));
                }
            }
        }
    }

    #[test]
    fn qsv_every_combination_returns_valid_params() {
        for (w, h) in RESOLUTIONS {
            for target in TARGETS {
                for tier in TIERS {
                    let p = qsv_av1_params(*target, *tier, *w, *h);
                    // oneVPL ICQ for AV1 is 1..51.
                    assert!((1..=51).contains(&p.icq_quality));
                    // AV1 q-index 0..255.
                    assert!(p.qp_i <= 255);
                    assert!(p.qp_p <= 255);
                    // TargetUsage is 1..7 per mfxstructs.h; we cap at
                    // 6 for Draft (av1-tuning-eng recommendation).
                    assert!((1..=6).contains(&p.target_usage));
                    // LowPower must be explicit OFF — Draft tier must
                    // not silently flip into the low-power path.
                    assert_eq!(p.low_power, MFX_CODINGOPTION_OFF);
                    assert!(p.num_tile_columns >= 1);
                    assert!(p.num_tile_rows >= 1);
                }
            }
        }
    }

    #[test]
    fn amf_q_index_monotonic_in_quality() {
        let (w, h) = (1920, 1080);
        let vl = amf_av1_params(QualityTarget::VisuallyLossless, SpeedTier::Standard, w, h);
        let hi = amf_av1_params(QualityTarget::High, SpeedTier::Standard, w, h);
        let std = amf_av1_params(QualityTarget::Standard, SpeedTier::Standard, w, h);
        let lo = amf_av1_params(QualityTarget::Low, SpeedTier::Standard, w, h);
        assert!(vl.q_index_intra < hi.q_index_intra);
        assert!(hi.q_index_intra < std.q_index_intra);
        assert!(std.q_index_intra < lo.q_index_intra);
    }

    #[test]
    fn qsv_icq_monotonic_in_quality() {
        let (w, h) = (1920, 1080);
        let vl = qsv_av1_params(QualityTarget::VisuallyLossless, SpeedTier::Standard, w, h);
        let hi = qsv_av1_params(QualityTarget::High, SpeedTier::Standard, w, h);
        let std = qsv_av1_params(QualityTarget::Standard, SpeedTier::Standard, w, h);
        let lo = qsv_av1_params(QualityTarget::Low, SpeedTier::Standard, w, h);
        // Lower ICQ quality = higher visual quality, so values should
        // increase as the requested target drops.
        assert!(vl.icq_quality < hi.icq_quality);
        assert!(hi.icq_quality < std.icq_quality);
        assert!(std.icq_quality < lo.icq_quality);
    }

    #[test]
    fn amf_archive_at_visually_lossless_uses_cqp() {
        let p = amf_av1_params(
            QualityTarget::VisuallyLossless,
            SpeedTier::Archive,
            1920,
            1080,
        );
        assert_eq!(p.rc_mode, AmfRateControl::Cqp);
    }

    #[test]
    fn amf_non_vl_uses_quality_vbr() {
        for target in [
            QualityTarget::High,
            QualityTarget::Standard,
            QualityTarget::Low,
        ] {
            for tier in TIERS {
                let p = amf_av1_params(target, *tier, 1920, 1080);
                assert_eq!(p.rc_mode, AmfRateControl::QualityVbr);
            }
        }
    }

    #[test]
    fn qsv_archive_at_visually_lossless_uses_cqp() {
        let p = qsv_av1_params(
            QualityTarget::VisuallyLossless,
            SpeedTier::Archive,
            1920,
            1080,
        );
        assert_eq!(p.rc_mode, QsvRateControl::Cqp);
    }

    #[test]
    fn amf_quality_preset_tier_mapping() {
        // Archive → HighQuality, Standard → Quality, Draft → Balanced.
        // Speed preset is never selected by this service.
        let (w, h) = (1920, 1080);
        let arc = amf_av1_params(QualityTarget::Standard, SpeedTier::Archive, w, h);
        let std = amf_av1_params(QualityTarget::Standard, SpeedTier::Standard, w, h);
        let drf = amf_av1_params(QualityTarget::Standard, SpeedTier::Draft, w, h);
        assert_eq!(arc.quality_preset, AmfQualityPreset::HighQuality);
        assert_eq!(std.quality_preset, AmfQualityPreset::Quality);
        assert_eq!(drf.quality_preset, AmfQualityPreset::Balanced);
    }

    #[test]
    fn qsv_target_usage_tier_ordering() {
        // Archive → TU 1 (best quality); Draft → TU 7 (best speed).
        let (w, h) = (1920, 1080);
        let arc = qsv_av1_params(QualityTarget::Standard, SpeedTier::Archive, w, h);
        let std = qsv_av1_params(QualityTarget::Standard, SpeedTier::Standard, w, h);
        let drf = qsv_av1_params(QualityTarget::Standard, SpeedTier::Draft, w, h);
        assert!(arc.target_usage < std.target_usage);
        assert!(std.target_usage < drf.target_usage);
    }

    #[test]
    fn amf_tile_count_caps_at_4() {
        // RDNA3 VCN prefers few tiles — research §3 caps at 4 via tile_grid_hw.
        // tiles_per_frame = cols * rows on the shared 2×2-at-4K grid.
        fn tiles(w: u32, h: u32) -> usize {
            let (c, r) = tile_grid_hw(w, h);
            c * r
        }
        assert_eq!(tiles(640, 360), 1);
        assert_eq!(tiles(1280, 720), 1);
        assert_eq!(tiles(1920, 1080), 4);
        assert_eq!(tiles(3840, 2160), 4);
    }

    #[test]
    fn qsv_tile_grid_caps_at_2x2() {
        // QSV AV1 shares `tile_grid_hw` with AMF/NVENC — capped at 2×2.
        assert_eq!(tile_grid_hw(640, 360), (1, 1));
        assert_eq!(tile_grid_hw(1280, 720), (1, 1));
        assert_eq!(tile_grid_hw(1920, 1080), (2, 2));
        assert_eq!(tile_grid_hw(3840, 2160), (2, 2));
    }

    /// Regression test from codec-spec-reviewer's task #49 review: every
    /// tile grid produced by the adapter must fit inside AV1 Level 5.1
    /// limits (AV1 spec Annex A.3): ≤8 tile columns, ≤64 total tiles,
    /// per-tile width ≤4096 luma samples, per-tile area ≤4,230,144.
    /// Level 5.1 covers every resolution we ship up to 4096×2176.
    #[test]
    fn tile_grid_fits_av1_level_5_1() {
        const MAX_TILE_COLS_L51: u32 = 8;
        const MAX_TILES_L51: u32 = 64;
        const MAX_TILE_WIDTH_L51: u32 = 4096;
        const MAX_TILE_AREA_L51: u32 = 4_230_144;

        for (w, h) in RESOLUTIONS {
            for (label, (cols, rows)) in [
                ("rav1e", tile_grid_rav1e(*w, *h)),
                ("nvenc", tile_grid_nvenc(*w, *h)),
                // AMF and QSV both share tile_grid_hw, which is today
                // an alias of tile_grid_nvenc — covering explicitly
                // so that if the alias diverges later, this regression
                // test catches the Level 5.1 compliance for HW paths.
                ("hw", tile_grid_hw(*w, *h)),
            ] {
                let cols = cols as u32;
                let rows = rows as u32;

                assert!(
                    cols <= MAX_TILE_COLS_L51,
                    "{} {}x{} emits {} tile cols; Level 5.1 max is {}",
                    label,
                    w,
                    h,
                    cols,
                    MAX_TILE_COLS_L51
                );
                assert!(
                    cols * rows <= MAX_TILES_L51,
                    "{} {}x{} emits {} total tiles; Level 5.1 max is {}",
                    label,
                    w,
                    h,
                    cols * rows,
                    MAX_TILES_L51
                );

                let tile_w = w.div_ceil(cols);
                let tile_h = h.div_ceil(rows);
                assert!(
                    tile_w <= MAX_TILE_WIDTH_L51,
                    "{} {}x{} per-tile width {} > Level 5.1 max {}",
                    label,
                    w,
                    h,
                    tile_w,
                    MAX_TILE_WIDTH_L51
                );
                assert!(
                    tile_w * tile_h <= MAX_TILE_AREA_L51,
                    "{} {}x{} per-tile area {} > Level 5.1 max {}",
                    label,
                    w,
                    h,
                    tile_w * tile_h,
                    MAX_TILE_AREA_L51
                );
            }
        }
    }
}
