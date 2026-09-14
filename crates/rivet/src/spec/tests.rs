use super::*;
use super::hdr_metadata;
use codec::encode::EncoderConfig;
use codec::frame::{PixelFormat, TransferFn};

#[test]
fn decode_policy_parses_and_resolves() {
    // `--decode-gpu` value space.
    assert_eq!("auto".parse::<DecodePolicy>().unwrap(), DecodePolicy::Auto);
    assert_eq!("".parse::<DecodePolicy>().unwrap(), DecodePolicy::Auto);
    assert_eq!("AUTO".parse::<DecodePolicy>().unwrap(), DecodePolicy::Auto);
    assert_eq!("fastest".parse::<DecodePolicy>().unwrap(), DecodePolicy::FastestGpu);
    assert_eq!(" Fastest ".parse::<DecodePolicy>().unwrap(), DecodePolicy::FastestGpu);
    assert_eq!("2".parse::<DecodePolicy>().unwrap(), DecodePolicy::SpecificGpu(2));
    assert_eq!("gpu:2".parse::<DecodePolicy>().unwrap(), DecodePolicy::SpecificGpu(2));
    assert_eq!("whole".parse::<DecodePolicy>().unwrap(), DecodePolicy::Whole);
    assert_eq!("ranges:3".parse::<DecodePolicy>().unwrap(), DecodePolicy::Ranges(3));
    assert!("bogus".parse::<DecodePolicy>().is_err());
    // One enum, so a pinned decode is never split: it asks for one range.
    assert_eq!(DecodePolicy::SpecificGpu(2).ranges_for(3), 1);
    assert_eq!(DecodePolicy::FastestGpu.ranges_for(3), 1);
    assert_eq!(DecodePolicy::Whole.ranges_for(3), 1);
    assert_eq!(DecodePolicy::Auto.ranges_for(3), 3);
    assert_eq!(DecodePolicy::Ranges(5).ranges_for(3), 5);
    // Resolution to a concrete pin (Auto / unresolved Fastest ⇒ None).
    assert_eq!(DecodePolicy::Auto.gpu_index(), None);
    assert_eq!(DecodePolicy::FastestGpu.gpu_index(), None);
    assert_eq!(DecodePolicy::SpecificGpu(3).gpu_index(), Some(3));
    assert!(DecodePolicy::FastestGpu.is_fastest());
    assert!(!DecodePolicy::SpecificGpu(0).is_fastest());
    assert_eq!(DecodePolicy::default(), DecodePolicy::Auto);
}

#[test]
fn single_file_sets_coherent_fields() {
    let s = OutputSpec::single_file(vec![Rung::new(1280, 720)]);
    assert_eq!(s.mode, OutputMode::SingleFile);
    assert_eq!(s.container, Container::Mp4);
    assert_eq!(s.muxer, Muxer::Mp4File);
    assert!(s.validate().is_ok());
}

#[test]
fn encode_policy_defaults_to_all_gpus() {
    let s = OutputSpec::single_file(vec![Rung::new(640, 360)]);
    assert_eq!(s.encode_policy, EncodePolicy::AllGpus);
    assert_eq!(s.gpu_index, None);
}

#[test]
fn chunk_seam_mode_defaults_parallel_and_builder_sets_it() {
    let s = OutputSpec::single_file(vec![Rung::new(640, 360)]);
    assert_eq!(s.chunk_seam_mode, ChunkSeamMode::Parallel);
    let s = s.chunk_seam_mode(ChunkSeamMode::ParallelConstQp);
    assert_eq!(s.chunk_seam_mode, ChunkSeamMode::ParallelConstQp);
    let s = OutputSpec::single_file(vec![Rung::new(640, 360)])
        .chunk_seam_mode(ChunkSeamMode::ParallelConstQp);
    assert_eq!(s.chunk_seam_mode, ChunkSeamMode::ParallelConstQp);
    assert!(s.validate().is_ok());
}

#[test]
fn encode_policy_single_gpu_syncs_gpu_index() {
    let s = OutputSpec::single_file(vec![Rung::new(640, 360)])
        .encode_policy(EncodePolicy::SingleGpu(Some(2)));
    assert_eq!(s.encode_policy, EncodePolicy::SingleGpu(Some(2)));
    assert_eq!(s.gpu_index, Some(2));
}

#[test]
fn with_gpu_index_implies_single_gpu_policy() {
    let s = OutputSpec::single_file(vec![Rung::new(640, 360)]).with_gpu_index(1);
    assert_eq!(s.encode_policy, EncodePolicy::SingleGpu(Some(1)));
    assert_eq!(s.gpu_index, Some(1));
}

#[test]
fn encode_policy_family_does_not_pin_gpu_index() {
    let s = OutputSpec::single_file(vec![Rung::new(640, 360)])
        .encode_policy(EncodePolicy::Family(GpuFamily::Nvidia));
    assert_eq!(s.encode_policy, EncodePolicy::Family(GpuFamily::Nvidia));
    // Family is multi-GPU within a vendor — no single-GPU pin.
    assert_eq!(s.gpu_index, None);
}

#[test]
fn decode_policy_defaults_to_auto_and_is_settable() {
    let s = OutputSpec::single_file(vec![Rung::new(640, 360)]);
    assert_eq!(s.decode_policy, DecodePolicy::Auto);
    let s = s.decode_policy(DecodePolicy::SpecificGpu(0));
    assert_eq!(s.decode_policy, DecodePolicy::SpecificGpu(0));
    // decode_policy is independent of the encode policy.
    assert_eq!(s.encode_policy, EncodePolicy::AllGpus);
}

#[test]
fn encode_policy_all_gpus_leaves_gpu_index_untouched() {
    let s = OutputSpec::single_file(vec![Rung::new(640, 360)])
        .with_gpu_index(3)
        .encode_policy(EncodePolicy::AllGpus);
    // AllGpus doesn't clear an explicit pin; it just won't single-pin.
    assert_eq!(s.encode_policy, EncodePolicy::AllGpus);
    assert_eq!(s.gpu_index, Some(3));
}

#[test]
fn hls_sets_coherent_fields() {
    let s = OutputSpec::hls(vec![Rung::new(1920, 1080), Rung::new(640, 360)], 4.0);
    assert!(matches!(s.mode, OutputMode::Hls { .. }));
    assert_eq!(s.container, Container::Cmaf);
    assert_eq!(s.muxer, Muxer::CmafHls);
    assert!(s.validate().is_ok());
}

#[test]
fn validate_rejects_empty_rungs() {
    assert!(OutputSpec::single_file(vec![]).validate().is_err());
}

#[test]
fn validate_rejects_odd_dimensions() {
    assert!(OutputSpec::single_file(vec![Rung::new(1281, 720)]).validate().is_err());
}

#[test]
fn validate_rejects_incoherent_mode_muxer() {
    let mut s = OutputSpec::single_file(vec![Rung::new(640, 360)]);
    s.muxer = Muxer::CmafHls; // mismatched with SingleFile mode
    assert!(s.validate().is_err());
}

#[test]
fn rung_label_uses_short_side() {
    assert_eq!(Rung::new(1920, 1080).label, "1080p");
    assert_eq!(Rung::new(1080, 1920).label, "1080p");
    assert_eq!(Rung::new(640, 360).short_side(), 360);
}

#[test]
fn color_and_pixel_format_default_to_sdr_8bit() {
    let s = OutputSpec::single_file(vec![Rung::new(640, 360)]);
    assert_eq!(s.color, ColorPolicy::TonemapToSdr);
    assert_eq!(s.bit_depth, BitDepth::Auto);
    assert!(s.tonemaps());
    assert!(s.validate().is_ok());
}

#[test]
fn resolve_output_default_folds_hdr_source_to_sdr_8bit() {
    let s = OutputSpec::single_file(vec![Rung::new(640, 360)]);
    let hdr_src = hdr_metadata(TransferFn::St2084);
    let (color, pix) = s.resolve_output(hdr_src, PixelFormat::Yuv420p10le);
    // Default TonemapToSdr collapses an HDR 10-bit source to 8-bit SDR.
    assert_eq!(color.transfer, TransferFn::Bt709);
    assert_eq!(pix, PixelFormat::Yuv420p);
}

/// The tag describes the picture after the pump. The 8-bit SDR path
/// re-derives a BT.601 (or BT.2020) matrix to BT.709, so an smpte170m-tagged
/// source comes out tagged bt709 on the matrix — before this held it came
/// out with BT.709 pixels and a smpte170m tag (ffprobe `tv,smpte170m`,
/// raw-decode PSNR 31.6 dB against a BT.709 rendering, 21.8 against the
/// BT.601 one). Range, primaries and transfer are not converted by that
/// path and keep the source's values; a 10-bit source is not matrixed at
/// all and keeps everything; a BT.709 source is untouched.
#[test]
fn resolve_output_sdr_tags_a_rederived_matrix_bt709() {
    let s = OutputSpec::single_file(vec![Rung::new(640, 360)]);
    let bt601 = codec::frame::ColorMetadata {
        transfer: TransferFn::Bt709, // what `from_h273(6)` folds SMPTE 170M's transfer onto
        matrix_coefficients: 6,
        colour_primaries: 6,
        full_range: false,
        ..Default::default()
    };
    let (color, pix) = s.resolve_output(bt601, PixelFormat::Yuv420p);
    assert_eq!(color.matrix_coefficients, 1, "the matrix the pump re-derived");
    assert_eq!(color.colour_primaries, 6, "primaries are not converted, so not re-tagged");
    assert_eq!(color.transfer, TransferFn::Bt709);
    assert!(!color.full_range);
    assert_eq!(pix, PixelFormat::Yuv420p);
    // BT.470BG (PAL, matrix 5) and 8-bit BT.2020 (matrix 9) take the same path.
    for m in [5u8, 9, 10] {
        let src = codec::frame::ColorMetadata { matrix_coefficients: m, colour_primaries: m, ..bt601 };
        let (color, _) = s.resolve_output(src, PixelFormat::Yuv420p);
        assert_eq!(color.matrix_coefficients, 1, "matrix {m}");
    }
    // A 10-bit BT.601 source is layout-normalised only: its tags stay.
    let (color, pix) = s.resolve_output(bt601, PixelFormat::Yuv420p10le);
    assert_eq!(color.matrix_coefficients, 6, "10-bit: no matrix conversion, no re-tag");
    assert_eq!(pix, PixelFormat::Yuv420p10le);
    // A full-range BT.709 source is untouched: nothing to re-tag, range kept.
    let full = codec::frame::ColorMetadata { full_range: true, ..Default::default() };
    let (color, _) = s.resolve_output(full, PixelFormat::Yuv420p);
    assert_eq!(color, full);
}

#[test]
fn resolve_output_passthrough_keeps_source() {
    let s = OutputSpec::single_file(vec![Rung::new(640, 360)]).with_color(ColorPolicy::Passthrough);
    assert!(!s.tonemaps());
    let src = hdr_metadata(TransferFn::St2084);
    let (color, pix) = s.resolve_output(src, PixelFormat::Yuv420p10le);
    assert_eq!(color.transfer, TransferFn::St2084);
    assert_eq!(pix, PixelFormat::Yuv420p10le);
}

/// An HDR policy re-tags the gamut and transfer but keeps what the source
/// said about its content — the mastering display and the content light
/// level — because those are what the encoders' SEIs and the container's
/// `mdcv` / `clli` are written from. Before this held, `--color hdr10` on
/// an HDR10 source produced a file with no mastering display while
/// `passthrough` on the same source kept it.
#[test]
fn resolve_output_hdr_policies_keep_the_sources_static_metadata() {
    use codec::frame::{ContentLightLevel, MasteringDisplay};
    let md = MasteringDisplay {
        primaries_r_x: 34000,
        primaries_r_y: 16000,
        primaries_g_x: 13250,
        primaries_g_y: 34500,
        primaries_b_x: 7500,
        primaries_b_y: 3000,
        white_point_x: 15635,
        white_point_y: 16450,
        max_luminance: 10_000_000,
        min_luminance: 1,
    };
    let cll = ContentLightLevel { max_cll: 1000, max_fall: 400 };
    let src = codec::frame::ColorMetadata {
        mastering_display: Some(md),
        content_light_level: Some(cll),
        ..hdr_metadata(TransferFn::St2084)
    };
    for (policy, transfer) in [(ColorPolicy::Hdr10, TransferFn::St2084), (ColorPolicy::Hlg, TransferFn::AribStdB67)] {
        let s = OutputSpec::single_file(vec![Rung::new(640, 360)]).with_color(policy);
        let (color, pix) = s.resolve_output(src, PixelFormat::Yuv420p10le);
        assert_eq!(color.transfer, transfer, "{policy:?}");
        assert_eq!((color.colour_primaries, color.matrix_coefficients), (9, 9), "{policy:?}: BT.2020");
        assert_eq!(color.mastering_display, Some(md), "{policy:?}: the source's mastering display");
        assert_eq!(color.content_light_level, Some(cll), "{policy:?}: the source's content light level");
        assert_eq!(pix, PixelFormat::Yuv420p10le);
    }
    // A source without any says nothing either way.
    let s = OutputSpec::single_file(vec![Rung::new(640, 360)]).with_color(ColorPolicy::Hdr10);
    let (color, _) = s.resolve_output(codec::frame::ColorMetadata::default(), PixelFormat::Yuv420p);
    assert_eq!(color.mastering_display, None);
    assert_eq!(color.content_light_level, None);
}

#[test]
fn validate_rejects_hdr_without_a_10bit_encoder() {
    // HDR10 implies 10-bit AND HDR signalling. A default build is 8-bit; a
    // `rav1e-fallback`-only build is 8-bit too; the hardware encoders and
    // the `h26x-fallback` tier (H.265 Main 10 with the VUI colour
    // description) report both. Validation must reject unless the build
    // has both.
    let s = OutputSpec::single_file(vec![Rung::new(640, 360)]).with_color(ColorPolicy::Hdr10);
    let caps = codec::encode::build_output_caps();
    if caps.max_bit_depth < 10 || !caps.hdr {
        assert!(s.validate().is_err(), "HDR must be rejected on a build without a 10-bit HDR encoder ({caps:?})");
    } else {
        assert!(s.validate().is_ok());
    }
}

#[test]
fn validate_rejects_hdr_forced_8bit() {
    let s = OutputSpec::single_file(vec![Rung::new(640, 360)])
        .with_color(ColorPolicy::Hdr10)
        .with_bit_depth(BitDepth::EightBit);
    assert!(s.validate().is_err());
}

#[test]
fn quality_crf_applies_to_encoder_config() {
    let q = Quality::crf(28);
    let mut cfg = EncoderConfig::default();
    q.apply(&mut cfg, 30.0);
    assert_eq!(cfg.quality, 28);
    assert_eq!(cfg.keyframe_interval, 60); // 2 * 30
}

#[test]
fn rung_policy_resolves_by_position_and_the_rungs_own_knobs_win() {
    use codec::encode::tuning::{EncodeOverrides, RungPolicy, TileGrid};

    let rungs = vec![
        Rung::new(1920, 1080),
        // A per-title style shift on this rung alone, plus its own tile grid.
        Rung::new(1280, 720).with_quality(Quality::default().with_overrides(EncodeOverrides {
            quality_delta: 4,
            tiles: Some(TileGrid { columns: 2, rows: 1 }),
            ..Default::default()
        })),
        Rung::new(640, 360),
    ];
    let spec = OutputSpec::hls(rungs, 4.0).with_rung_policy(RungPolicy::recommended());
    let resolved = spec.with_rung_policy_resolved();

    // Folded away, so nothing downstream applies it twice.
    assert!(resolved.rung_policy.rules.is_empty() && resolved.rung_policy.global.is_empty());

    let top = resolved.rungs[0].quality.overrides;
    let mid = resolved.rungs[1].quality.overrides;
    let low = resolved.rungs[2].quality.overrides;

    // Softer going down: 0, +2, +4 from the policy — and the middle rung's own
    // +4 accumulates on top of its positional +2.
    assert_eq!(top.quality_delta, 0);
    assert_eq!(mid.quality_delta, 6);
    assert_eq!(low.quality_delta, 4);
    // The rung's own tile grid beats the policy's single tile.
    assert_eq!(mid.tiles, Some(TileGrid { columns: 2, rows: 1 }));
    assert_eq!(low.tiles, Some(TileGrid::SINGLE));
    // Global knobs reach every rung.
    assert_eq!(top.reference_frames, Some(3));

    // An empty policy is the identity.
    let plain = OutputSpec::hls(vec![Rung::new(1920, 1080)], 4.0);
    assert!(plain.with_rung_policy_resolved().rungs[0].quality.overrides.is_empty());
}

#[test]
fn encode_policy_parses_the_whole_plan() {
    // `--encode` value space: one enum for "which cards" and "how".
    assert_eq!("all".parse::<EncodePolicy>().unwrap(), EncodePolicy::AllGpus);
    assert_eq!("".parse::<EncodePolicy>().unwrap(), EncodePolicy::AllGpus);
    assert_eq!("per-rung".parse::<EncodePolicy>().unwrap(), EncodePolicy::PerRung);
    assert_eq!("single".parse::<EncodePolicy>().unwrap(), EncodePolicy::SingleGpu(None));
    // The old seam-mode spelling of "one encoder" lands where it belongs.
    assert_eq!("serial".parse::<EncodePolicy>().unwrap(), EncodePolicy::SingleGpu(None));
    assert_eq!("gpu:1".parse::<EncodePolicy>().unwrap(), EncodePolicy::SingleGpu(Some(1)));
    assert_eq!("family:intel".parse::<EncodePolicy>().unwrap(), EncodePolicy::Family(GpuFamily::Intel));
    assert!("family:voodoo".parse::<EncodePolicy>().is_err());
    assert!("bogus".parse::<EncodePolicy>().is_err());

    assert!(EncodePolicy::AllGpus.spreads());
    assert!(EncodePolicy::PerRung.spreads());
    assert!(EncodePolicy::Family(GpuFamily::Nvidia).spreads());
    assert!(!EncodePolicy::SingleGpu(None).spreads());
    assert!(EncodePolicy::PerRung.pins_rungs());
    assert!(!EncodePolicy::AllGpus.pins_rungs());
}

#[test]
fn resolve_output_folds_every_source_layout_onto_the_encoder_formats() {
    use crate::spec::encoder_input_format;
    let s = OutputSpec::single_file(vec![Rung::new(640, 360)]);
    let sdr = ColorMetadata::default();
    // 12-bit and 4:2:2 / 4:4:4 sources: no encoder takes them, the pump
    // narrows / downsamples, so the encoder is configured for 10-bit 4:2:0.
    for fmt in [
        PixelFormat::Yuv420p12le,
        PixelFormat::Yuv422p10le,
        PixelFormat::Yuv422p12le,
        PixelFormat::Yuv444p10le,
        PixelFormat::Yuv444p12le,
        PixelFormat::Yuva444p10le,
    ] {
        assert_eq!(s.resolve_output(sdr, fmt).1, PixelFormat::Yuv420p10le, "{fmt:?}");
        assert_eq!(encoder_input_format(fmt), PixelFormat::Yuv420p10le);
    }
    for fmt in [
        PixelFormat::Yuv422p,
        PixelFormat::Yuv444p,
        PixelFormat::Nv12,
        PixelFormat::Nv21,
        PixelFormat::Rgb24,
        PixelFormat::Rgba32,
    ] {
        assert_eq!(s.resolve_output(sdr, fmt).1, PixelFormat::Yuv420p, "{fmt:?}");
    }
    // An explicit 8-bit output narrows a 12-bit source all the way.
    let eight = OutputSpec::single_file(vec![Rung::new(640, 360)]).with_bit_depth(BitDepth::EightBit);
    assert_eq!(eight.resolve_output(sdr, PixelFormat::Yuv420p12le).1, PixelFormat::Yuv420p);
    // Passthrough of a 12-bit HDR source still lands on the 10-bit ceiling.
    let pt = OutputSpec::single_file(vec![Rung::new(640, 360)]).with_color(ColorPolicy::Passthrough);
    let (color, pix) = pt.resolve_output(hdr_metadata(TransferFn::St2084), PixelFormat::Yuv444p12le);
    assert_eq!(color.transfer, TransferFn::St2084);
    assert_eq!(pix, PixelFormat::Yuv420p10le);
}
