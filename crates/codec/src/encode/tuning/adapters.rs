//! Per-encoder adapter functions.
//!
//! Each public function translates a `(QualityTarget, SpeedTier, width, height)`
//! tuple into the concrete parameter struct for a specific encoder backend.
//! Backend-private helpers (anchors, q-index mappers) live beside the
//! function that uses them.

use super::{
    libaom_cq_for_target, nvenc_cq_for_target, piecewise_quality, tile_grid_hw, tile_grid_nvenc,
    tile_grid_rav1e, NV_ENC_PRESET_P5_GUID_BYTES, NV_ENC_PRESET_P6_GUID_BYTES,
    NV_ENC_PRESET_P7_GUID_BYTES, NVENC_TUNING_HIGH_QUALITY,
};
use super::params::{
    AmfAv1Params, AmfQualityPreset, AmfRateControl, MFX_CODINGOPTION_ON, NvencAv1Params,
    NvencRateControl, QsvAv1Params, QsvRateControl, Rav1eParams,
};
use super::{QualityTarget, SpeedTier};

// ─── rav1e ───────────────────────────────────────────────────────

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

// ─── NVENC ───────────────────────────────────────────────────────

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

// ─── AMF ─────────────────────────────────────────────────────────

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

// ─── QSV ─────────────────────────────────────────────────────────

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
        // AV1 QSV encode is VDENC (low-power) only on Arc / Meteor Lake+.
        low_power: MFX_CODINGOPTION_ON,
    }
}

/// QSV ICQ scale is 1..51 (lower = better), inverted from AMF's QVBR.
/// Anchor table reflects Intel's public oneVPL sample benchmarks.
const QSV_ICQ_ANCHORS: &[(i32, i32)] =
    &[(100, 8), (98, 18), (95, 24), (90, 30), (85, 36), (70, 48)];

fn vmaf_to_qsv_icq(vmaf: u8) -> u16 {
    piecewise_quality(vmaf, QSV_ICQ_ANCHORS, 1, 51) as u16
}
