// Private items from mod.rs that the tests call directly (not
// brought in by the `*` glob, which only imports pub items).
use super::{
    nvenc_cq_for_target, tile_grid_hw, tile_grid_nvenc, tile_grid_rav1e,
    NV_ENC_PRESET_P5_GUID_BYTES, NV_ENC_PRESET_P6_GUID_BYTES, NV_ENC_PRESET_P7_GUID_BYTES,
};
// All pub items (QualityTarget, SpeedTier, NVENC_TUNING_HIGH_QUALITY,
// libaom_cq_for_target, re-exported param structs/enums, adapter fns).
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
                // LowPower must be ON — AV1 QSV encode is VDENC-only on
                // Intel (the only AV1 encode entry point the iHD driver
                // exposes); OFF makes Query reject with MFX_ERR_UNSUPPORTED.
                assert_eq!(p.low_power, MFX_CODINGOPTION_ON);
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
