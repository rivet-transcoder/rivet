use super::*;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use codec::frame::{ColorMetadata, PixelFormat, VideoCodec};

#[test]
fn config_clone_preserves_fields() {
    let cfg = EncoderWorkerConfig {
        overrides: Default::default(),
        rung_idx: 2,
        codec: VideoCodec::Av1,
        width: 1280,
        height: 720,
        frame_rate: 30.0,
        quality: 32,
        speed_preset: u8::MAX,
        target: codec::encode::tuning::QualityTarget::Standard,
        tier: codec::encode::tuning::SpeedTier::Standard,
        threads: 4,
        gpu_index: Some(1),
        gpu_vendor: None,
        backend: None,
        output_color_metadata: ColorMetadata::default(),
        output_pixel_format: PixelFormat::Yuv420p,
        constant_qp: false,
        timescale: 30000,
        per_frame_ticks: 1000,
        keyframe_interval: 60,
        segment_target_ticks: 60_000,
        output_dir: PathBuf::from("/tmp/x"),
        rung_invariant: Arc::new(RwLock::new(None)),
    };
    let copy = cfg.clone();
    assert_eq!(copy.rung_idx, 2);
    assert_eq!(copy.keyframe_interval, 60);
}

#[test]
fn invariant_matches_itself() {
    let a = RungCodecInvariant::Av1(Av1Invariant {
        seq_profile: 0,
        seq_level_idx_0: 8,
        seq_tier_0: 0,
        bit_depth: 8,
        monochrome: false,
        chroma_subsampling_x: true,
        chroma_subsampling_y: true,
        color_primaries: 1,
        transfer_characteristics: 1,
        matrix_coefficients: 1,
        color_range: false,
        max_frame_width_minus1: 1919,
        max_frame_height_minus1: 1079,
        still_picture: false,
    });
    assert_eq!(a.clone(), a);
    assert_eq!(a.describe_diff(&a), "");
}

#[test]
fn invariant_diff_lists_changed_fields() {
    let inner = Av1Invariant {
        seq_profile: 0,
        seq_level_idx_0: 8,
        seq_tier_0: 0,
        bit_depth: 8,
        monochrome: false,
        chroma_subsampling_x: true,
        chroma_subsampling_y: true,
        color_primaries: 1,
        transfer_characteristics: 1,
        matrix_coefficients: 1,
        color_range: false,
        max_frame_width_minus1: 1919,
        max_frame_height_minus1: 1079,
        still_picture: false,
    };
    let mut inner_b = inner.clone();
    inner_b.bit_depth = 10;
    inner_b.color_primaries = 9;
    let a = RungCodecInvariant::Av1(inner);
    let b = RungCodecInvariant::Av1(inner_b);
    let diff = a.describe_diff(&b);
    assert!(diff.contains("bit_depth"));
    assert!(diff.contains("color_primaries"));
    assert!(!diff.contains("seq_profile"));
}

#[test]
fn validator_parse_error_returns_err_not_mismatch() {
    // Junk bytes — no recognisable AV1 sequence header OBU.
    // Distinct from a mismatch: this is a malformed-bitstream
    // condition that nothing downstream can recover from. The
    // worker propagates this Err and fails the run, unlike the
    // soft-fail Mismatched case.
    let slot: RwLock<Option<RungCodecInvariant>> = RwLock::new(None);
    let junk = vec![0u8; 8];
    let err = validate_or_set_rung_invariant(
        0,
        Some(codec::gpu::GpuVendor::Intel),
        &slot,
        &junk,
        VideoCodec::Av1,
    )
    .unwrap_err();
    assert!(
        err.to_string()
            .contains("could not parse AV1 sequence header")
    );
    assert!(slot.read().unwrap().is_none());
}

#[test]
fn mismatched_diff_includes_changed_field() {
    let inner = Av1Invariant {
        seq_profile: 0,
        seq_level_idx_0: 8,
        seq_tier_0: 0,
        bit_depth: 8,
        monochrome: false,
        chroma_subsampling_x: true,
        chroma_subsampling_y: true,
        color_primaries: 1,
        transfer_characteristics: 1,
        matrix_coefficients: 1,
        color_range: false,
        max_frame_width_minus1: 1919,
        max_frame_height_minus1: 1079,
        still_picture: false,
    };
    let mut other_inner = inner.clone();
    other_inner.bit_depth = 10;
    let existing = RungCodecInvariant::Av1(inner);
    let other = RungCodecInvariant::Av1(other_inner);
    let diff = existing.describe_diff(&other);
    assert!(
        diff.contains("bit_depth"),
        "diff should mention bit_depth; got {diff}"
    );
}

#[test]
fn h26x_invariant_equality_and_diff() {
    // Two H.264/H.265 chunks agree iff their SPS-derived fields match.
    let a = RungCodecInvariant::H26x(H26xInvariant {
        profile_idc: 100,
        level_idc: 31,
        chroma_format_idc: 1,
        bit_depth_luma: 8,
        bit_depth_chroma: 8,
        width: 1280,
        height: 720,
    });
    assert_eq!(a.clone(), a);
    assert_eq!(a.describe_diff(&a), "");
    let b = RungCodecInvariant::H26x(H26xInvariant {
        profile_idc: 100,
        level_idc: 31,
        chroma_format_idc: 1,
        bit_depth_luma: 10, // a 10-bit chunk must NOT match an 8-bit one
        bit_depth_chroma: 10,
        width: 1280,
        height: 720,
    });
    assert_ne!(a, b);
    assert!(!a.describe_diff(&b).is_empty());
    // An AV1 invariant never equals an H26x one (defensive — single-codec rungs).
    let av1 = RungCodecInvariant::Av1(Av1Invariant {
        seq_profile: 0,
        seq_level_idx_0: 8,
        seq_tier_0: 0,
        bit_depth: 8,
        monochrome: false,
        chroma_subsampling_x: true,
        chroma_subsampling_y: true,
        color_primaries: 1,
        transfer_characteristics: 1,
        matrix_coefficients: 1,
        color_range: false,
        max_frame_width_minus1: 1279,
        max_frame_height_minus1: 719,
        still_picture: false,
    });
    assert_ne!(a, av1);
}
