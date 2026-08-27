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
    AmfAv1Params, AmfH26xParams, AmfQualityPreset, AmfRateControl, H26xSwParams, MFX_CODINGOPTION_ON,
    NvencAv1Params, NvencRateControl, QsvAv1Params, QsvRateControl, Rav1eParams,
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

// ─── AMF H.264 / H.265 ───────────────────────────────────────────

/// Derive AMD AMF H.264 / H.265 params for a quality target + speed tier.
///
/// The quantiser is [`h26x_qp_for_target`] — the same 0..51 anchors as the
/// QSV H.26x path and the native software encoders, so a job that lands on
/// an AMD card instead of an Arc keeps its QP. The QVBR quality level
/// (`VideoEncoderVCE.h:204` / `VideoEncoderHEVC.h:181`: "default = 23;
/// range = 1-51") runs the **other way** from a QP — higher is better —
/// measured on a Ryzen 9700X iGPU (H.264 1080p: level 1 → 35.9 dB at
/// 1.3 Mbit/s, 26 → 41.1 dB at 4.1 Mbit/s, 51 → 47.1 dB at 8.2 Mbit/s;
/// H.265 720p the same shape), so it is [`qvbr_level_for_qp`]: `52 - QP`,
/// which puts the Standard target's QP 26 at level 26, the driver's own
/// default neighbourhood. Presets follow the AV1 adapter's tier rule; the
/// numeric header value is assigned per codec in `encode/amf/h26x.rs`.
///
/// Not swept for VMAF — see TODO.md; the anchors are the x264 / x265 CRF
/// conventions the other H.26x tables share.
pub fn amf_h26x_params(
    codec: crate::frame::VideoCodec,
    target: QualityTarget,
    tier: SpeedTier,
) -> AmfH26xParams {
    debug_assert!(codec != crate::frame::VideoCodec::Av1, "AV1 has its own AMF adapter");
    let _ = codec; // the two codecs share every knob this struct carries
    let qp = h26x_qp_for_target(target).clamp(0, 51) as u8;
    AmfH26xParams {
        rc_mode: match target {
            QualityTarget::VisuallyLossless => AmfRateControl::Cqp,
            _ => AmfRateControl::QualityVbr,
        },
        qp_i: qp,
        // Inter frames tolerate a slightly coarser QP; +2 is the conventional
        // step, as on QSV.
        qp_p: (qp + 2).min(51),
        qvbr_quality: qvbr_level_for_qp(qp),
        quality_preset: match tier {
            SpeedTier::Archive => AmfQualityPreset::HighQuality,
            SpeedTier::Standard => AmfQualityPreset::Quality,
            SpeedTier::Draft => AmfQualityPreset::Balanced,
        },
    }
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
/// Derive Intel QSV params for a **specific output codec**.
///
/// The three codecs QSV encodes here don't share a quantizer scale, so the
/// AV1 table can't stand in for the others:
///
/// | | ICQ (`ICQQuality`) | CQP (`QPI`/`QPP`) |
/// |---|---|---|
/// | AV1 | 1..51 | 0..255 (q-index) |
/// | HEVC | 1..51 | 0..51 |
/// | H.264 | 1..51 | 0..51 |
///
/// ICQ happens to be a uniform 1..51 across all three (a oneVPL API
/// convention), but CQP is not: AV1 takes a native 0..255 q-index while
/// H.264/HEVC take an ordinary 0..51 QP. Feeding the AV1 table's
/// `libaom_cq * 4` (up to 152) into an HEVC job puts `QPI` far outside the
/// legal range, which the driver either clamps or rejects.
///
/// **Calibration provenance.** The AV1 numbers are measured against libaom as
/// the cross-encoder reference (`docs/av1-tuning-research.md`). The H.264 /
/// HEVC anchors below are *not* measured — they're the long-standing x264 /
/// x265 CRF conventions for each quality tier, which is the honest starting
/// point given the same VMAF sweep hasn't been run for them. See TODO.md.
pub fn qsv_params(
    codec: crate::frame::VideoCodec,
    target: QualityTarget,
    tier: SpeedTier,
    width: u32,
    height: u32,
) -> QsvAv1Params {
    match codec {
        crate::frame::VideoCodec::Av1 => qsv_av1_params(target, tier, width, height),
        crate::frame::VideoCodec::H265 | crate::frame::VideoCodec::H264 => {
            qsv_h26x_params(target, tier)
        }
    }
}

/// QSV params for H.264 / H.265, whose quantizer is an ordinary 0..51 QP.
///
/// Anchors are the familiar x264 / x265 CRF values per tier — 18 is the
/// "visually lossless" rule of thumb, 23 the x264 default, 28 the x265
/// default, and ~34 a deliberately lossy tier.
fn qsv_h26x_params(target: QualityTarget, tier: SpeedTier) -> QsvAv1Params {
    let qp = h26x_qp_for_target(target);
    QsvAv1Params {
        rc_mode: match target {
            QualityTarget::VisuallyLossless => QsvRateControl::Cqp,
            _ => QsvRateControl::Icq,
        },
        icq_quality: qp.clamp(1, 51),
        // Same 0..51 scale as ICQ for these codecs — no q-index conversion.
        qp_i: qp.clamp(0, 51),
        // Inter frames tolerate a slightly coarser QP; +2 is the conventional
        // step (the AV1 path's +8 is on a 4x-wider scale).
        qp_p: (qp + 2).clamp(0, 51),
        target_usage: match tier {
            SpeedTier::Archive => 1,
            SpeedTier::Standard => 4,
            SpeedTier::Draft => 6,
        },
        gop_pic_size: 0, // caller fills from keyframe_interval
        // Tiles are an AV1-only ext buffer here; leave the grid empty so the
        // caller has nothing to attach.
        num_tile_columns: 0,
        num_tile_rows: 0,
        // VDENC on Arc covers H.264 and HEVC as well as AV1.
        low_power: MFX_CODINGOPTION_ON,
    }
}

/// The H.26x quantiser (0..51) a quality target means, shared by every
/// backend whose H.264 / H.265 quantiser is on that scale — QSV in hardware
/// and the native `h26x` encoders in software — so a target lands at the same
/// QP whichever of them runs it.
///
/// Anchors are the familiar x264 / x265 CRF values per tier — 18 is the
/// "visually lossless" rule of thumb, 23 the x264 default, 28 the x265
/// default, and ~34 a deliberately lossy tier. Convention rather than
/// measurement; see TODO.md for the VMAF sweep still owed.
fn h26x_qp_for_target(target: QualityTarget) -> u16 {
    match target {
        QualityTarget::VisuallyLossless => 18,
        QualityTarget::High => 22,
        QualityTarget::Standard => 26,
        QualityTarget::Low => 32,
        // The ICQ anchor table is already on a 1..51 scale, which is the same
        // scale H.26x QP uses, so it transfers directly here.
        QualityTarget::Vmaf(v) => vmaf_to_qsv_icq(v),
    }
}

// ─── h26x (software H.264 / H.265) ───────────────────────────────

/// Derive the native software H.264 / H.265 encoder's params for a quality
/// target + speed tier.
///
/// The quantiser is [`h26x_qp_for_target`], so a job that moves between the
/// QSV hardware path and this one keeps its QP. The tier chooses the coding
/// tools that cost search time: the 8x8 transform is cheap and on from
/// `Standard` up; sub-16x16 partitions multiply the motion search and buy
/// little outside content whose macroblock halves move differently, so only
/// `Archive` pays for them; SAO (H.265) is an in-loop filter that costs bits
/// per CTB and only pays where there is quantisation noise to shape, which at
/// these QPs there is — on from `Standard` up.
pub fn h26x_sw_params(
    codec: crate::frame::VideoCodec,
    target: QualityTarget,
    tier: SpeedTier,
) -> H26xSwParams {
    let qp = h26x_qp_for_target(target).clamp(0, 51) as u8;
    let is_h264 = codec == crate::frame::VideoCodec::H264;
    H26xSwParams {
        qp,
        transform_8x8: is_h264 && tier != SpeedTier::Draft,
        subparts: is_h264 && tier == SpeedTier::Archive,
        sao: !is_h264 && tier != SpeedTier::Draft,
    }
}

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

// ─── Override-aware variants ─────────────────────────────────────
//
// The functions above answer "what does this quality target mean for this
// backend". These answer "…and then what did the caller ask for on top". They
// are separate rather than defaulted parameters so that every existing call
// site keeps its exact behaviour, and so the inert-empty-override property is
// something a test can state directly: `*_params_with(t, s, ctx, &default())`
// must equal `*_params(t, s, w, h)` for every backend.

use super::overrides::{EncodeOverrides, RungContext};

/// Apply a libaom-CQ-equivalent delta to a value on libaom's own scale.
fn shift_libaom(base: u8, delta: i16, max: u8) -> u8 {
    (i32::from(base) + i32::from(delta)).clamp(0, i32::from(max)) as u8
}

/// The tile grid the caller asked for, or the resolution-derived default.
fn tiles_or(overrides: &EncodeOverrides, derived: (usize, usize)) -> (usize, usize) {
    match overrides.tiles {
        Some(grid) => (usize::from(grid.columns).max(1), usize::from(grid.rows).max(1)),
        None => derived,
    }
}

/// [`rav1e_params`], with caller overrides applied.
pub fn rav1e_params_with(
    target: QualityTarget,
    tier: SpeedTier,
    rung: &RungContext,
    overrides: &EncodeOverrides,
) -> Rav1eParams {
    let target = overrides.quality_target.unwrap_or(target);
    let tier = overrides.speed_tier.unwrap_or(tier);
    let mut params = rav1e_params(target, tier, rung.width, rung.height);

    // rav1e's quantizer runs 0-255 at roughly four times libaom's scale, which
    // is the ratio `rav1e_params` itself uses to derive it.
    let shift = i32::from(overrides.quality_delta) * 4;
    params.quantizer = (params.quantizer as i32 + shift).clamp(0, 255) as usize;

    let (cols, rows) = tiles_or(overrides, (params.tile_cols, params.tile_rows));
    params.tile_cols = cols;
    params.tile_rows = rows;
    params
}

/// [`nvenc_av1_params`], with caller overrides applied.
pub fn nvenc_av1_params_with(
    target: QualityTarget,
    tier: SpeedTier,
    rung: &RungContext,
    overrides: &EncodeOverrides,
) -> NvencAv1Params {
    let target = overrides.quality_target.unwrap_or(target);
    let tier = overrides.speed_tier.unwrap_or(tier);
    let mut params = nvenc_av1_params(target, tier, rung.width, rung.height);

    // `cq` is AV1's 0-63 index, same direction as libaom.
    params.cq = shift_libaom(params.cq, overrides.quality_delta, 63);

    // Lookahead is a request, not an instruction: the encoder only honours it
    // if its surface pool can survive the runtime holding frames. See
    // `EncodeOverrides::lookahead_frames`.
    if let Some(frames) = overrides.lookahead_frames {
        params.lookahead_depth = frames;
    }

    let (cols, rows) =
        tiles_or(overrides, (params.num_tile_columns as usize, params.num_tile_rows as usize));
    params.num_tile_columns = cols as u32;
    params.num_tile_rows = rows as u32;
    params
}

/// [`amf_av1_params`], with caller overrides applied.
pub fn amf_av1_params_with(
    target: QualityTarget,
    tier: SpeedTier,
    rung: &RungContext,
    overrides: &EncodeOverrides,
) -> AmfAv1Params {
    let target = overrides.quality_target.unwrap_or(target);
    let tier = overrides.speed_tier.unwrap_or(tier);
    let mut params = amf_av1_params(target, tier, rung.width, rung.height);

    // AMF's q_index is `libaom * 4 - 8`, so a libaom step is four here.
    let shift = i32::from(overrides.quality_delta) * 4;
    params.q_index_intra = (i32::from(params.q_index_intra) + shift).clamp(0, 255) as u8;
    params.q_index_inter = (i32::from(params.q_index_inter) + shift).clamp(0, 255) as u8;

    // AMF carries only a tile count, not a grid, so an explicit grid collapses
    // to its product here — the shape is the caller's business, the total is
    // all this backend can act on.
    if let Some(grid) = overrides.tiles {
        params.tiles_per_frame = grid.tiles();
    }
    params
}

/// [`qsv_params`], with caller overrides applied.
pub fn qsv_params_with(
    codec: crate::frame::VideoCodec,
    target: QualityTarget,
    tier: SpeedTier,
    rung: &RungContext,
    overrides: &EncodeOverrides,
) -> QsvAv1Params {
    let target = overrides.quality_target.unwrap_or(target);
    let tier = overrides.speed_tier.unwrap_or(tier);
    let mut params = qsv_params(codec, target, tier, rung.width, rung.height);
    apply_qsv_overrides(&mut params, overrides);
    params
}

/// [`h26x_sw_params`], with caller overrides applied.
pub fn h26x_sw_params_with(
    codec: crate::frame::VideoCodec,
    target: QualityTarget,
    tier: SpeedTier,
    overrides: &EncodeOverrides,
) -> H26xSwParams {
    let target = overrides.quality_target.unwrap_or(target);
    let tier = overrides.speed_tier.unwrap_or(tier);
    let mut params = h26x_sw_params(codec, target, tier);
    // The delta is denominated in libaom CQ steps, and the H.26x QP scale
    // runs at about the same pitch (the QSV H.26x table applies it one for
    // one, too), so a step is a step.
    params.qp = shift_libaom(params.qp, overrides.quality_delta, 51);
    params
}

/// AMF's QVBR quality level (1..=51, higher = better) for an H.26x QP
/// (0..=51, lower = better): `52 - QP`, so QP 26 ↔ level 26, QP 1 ↔ 51.
/// Direction measured, see [`amf_h26x_params`].
pub fn qvbr_level_for_qp(qp: u8) -> u8 {
    (52u8.saturating_sub(qp)).clamp(1, 51)
}

/// [`amf_h26x_params`], with caller overrides applied.
pub fn amf_h26x_params_with(
    codec: crate::frame::VideoCodec,
    target: QualityTarget,
    tier: SpeedTier,
    overrides: &EncodeOverrides,
) -> AmfH26xParams {
    let target = overrides.quality_target.unwrap_or(target);
    let tier = overrides.speed_tier.unwrap_or(tier);
    let mut params = amf_h26x_params(codec, target, tier);
    // A libaom step is a QP step on this scale (as for the QSV and software
    // H.26x tables); the QVBR level is derived from the shifted QP so the
    // two cannot disagree about direction.
    params.qp_i = shift_libaom(params.qp_i, overrides.quality_delta, 51);
    params.qp_p = shift_libaom(params.qp_p, overrides.quality_delta, 51);
    params.qvbr_quality = qvbr_level_for_qp(params.qp_i);
    params
}

/// [`qsv_av1_params`], with caller overrides applied.
pub fn qsv_av1_params_with(
    target: QualityTarget,
    tier: SpeedTier,
    rung: &RungContext,
    overrides: &EncodeOverrides,
) -> QsvAv1Params {
    let target = overrides.quality_target.unwrap_or(target);
    let tier = overrides.speed_tier.unwrap_or(tier);
    let mut params = qsv_av1_params(target, tier, rung.width, rung.height);
    apply_qsv_overrides(&mut params, overrides);
    params
}

fn apply_qsv_overrides(params: &mut QsvAv1Params, overrides: &EncodeOverrides) {
    // ICQ is 1..51, 1 = best, and the adapter derives it on roughly libaom's
    // scale — so a libaom step is one ICQ step.
    params.icq_quality =
        (i32::from(params.icq_quality) + i32::from(overrides.quality_delta)).clamp(1, 51) as u16;

    let (cols, rows) = tiles_or(
        overrides,
        (usize::from(params.num_tile_columns), usize::from(params.num_tile_rows)),
    );
    params.num_tile_columns = cols as u8;
    params.num_tile_rows = rows as u8;

    // Say so, rather than ignoring it.
    //
    // oneVPL's lookahead is `mfxExtCodingOption2::LookAheadDepth` and needs the
    // LA rate-control mode; this backend sets neither, so a caller asking for
    // lookahead here gets exactly nothing. That silence cost a full
    // measurement cycle: the request was made on a live fleet, the output came
    // back byte-for-byte identical, and the only way to find out why was to
    // read the adapter.
    //
    // A knob that cannot be honoured has to say so at the point it is dropped.
    if overrides.lookahead_frames.is_some_and(|frames| frames > 0) {
        tracing::warn!(
            requested = overrides.lookahead_frames,
            "oneVPL: lookahead is not implemented in this backend and is being ignored —              the encode will be identical to one that never asked for it",
        );
    }
}
