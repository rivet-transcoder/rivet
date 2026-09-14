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
    // An HDR source without any says nothing either way. (An SDR source mapped
    // into PQ is signalled with its mapped colour volume instead — see
    // `hdr10_on_an_sdr_source_signals_the_mapped_colour_volume`.)
    let s = OutputSpec::single_file(vec![Rung::new(640, 360)]).with_color(ColorPolicy::Hdr10);
    let (color, _) = s.resolve_output(hdr_metadata(TransferFn::St2084), PixelFormat::Yuv420p10le);
    assert_eq!(color.mastering_display, None);
    assert_eq!(color.content_light_level, None);
}

/// What `validate` checks a job's colour and depth against: the caps for the
/// job's codec on this build, never the codec-agnostic union. The two differ
/// on the builds that matter — `h26x-fallback` (AV1 8-bit, H.264 / H.265
/// 10-bit) and a hardware feature without it (H.264 8-bit, AV1 / H.265
/// 10-bit) — and a check against the union accepts a job that then fails
/// building its encoder after the job has started. On a default build the
/// two agree (8-bit everywhere); the backend-set tests below carry the
/// per-codec rule there.
#[test]
fn validate_checks_the_output_policy_against_the_jobs_codec_on_this_build() {
    use codec::encode::build_output_caps_for;
    for codec in [VideoCodecPolicy::Av1, VideoCodecPolicy::H264, VideoCodecPolicy::H265] {
        let caps = build_output_caps_for(codec.codec());
        for (color, depth) in TEN_BIT_POLICIES {
            let s = OutputSpec::single_file(vec![Rung::new(640, 360)])
                .with_video_codec(codec)
                .with_color(color)
                .with_bit_depth(depth);
            let producible = caps.max_bit_depth >= 10 && (!color.is_hdr() || caps.hdr);
            assert_eq!(
                s.validate().is_ok(),
                producible,
                "{codec:?} {color:?} {depth:?} on {caps:?}: {:?}",
                s.validate().err()
            );
        }
        let sdr = OutputSpec::single_file(vec![Rung::new(640, 360)])
            .with_video_codec(codec)
            .with_bit_depth(BitDepth::EightBit);
        assert!(sdr.validate().is_ok(), "{codec:?}: 8-bit SDR is never refused for capability");
    }
    // The two cases the codec-agnostic check got wrong, by name.
    let h264_hdr = OutputSpec::single_file(vec![Rung::new(640, 360)])
        .with_video_codec(VideoCodecPolicy::H264)
        .hdr10();
    if cfg!(feature = "h26x-fallback") {
        assert!(h264_hdr.validate().is_ok(), "h26x-fallback encodes H.264 High 10");
    } else {
        let err = h264_hdr.validate().expect_err("no 10-bit H.264 encoder without h26x-fallback").to_string();
        assert!(err.contains("h264 at 10 bits") && err.contains("`h26x-fallback`"), "{err}");
    }
    let av1_hdr = OutputSpec::single_file(vec![Rung::new(640, 360)]).hdr10();
    let hardware = cfg!(any(feature = "nvidia", feature = "amd", feature = "qsv"));
    assert_eq!(av1_hdr.validate().is_ok(), hardware, "10-bit AV1 is hardware-only: {:?}", av1_hdr.validate().err());
}

/// Every policy that needs 10 bits: the HDR two, and a forced 10-bit depth
/// with and without HDR.
const TEN_BIT_POLICIES: [(ColorPolicy, BitDepth); 5] = [
    (ColorPolicy::Hdr10, BitDepth::Auto),
    (ColorPolicy::Hlg, BitDepth::Auto),
    (ColorPolicy::Hdr10, BitDepth::TenBit),
    (ColorPolicy::TonemapToSdr, BitDepth::TenBit),
    (ColorPolicy::Passthrough, BitDepth::TenBit),
];

fn refusal(
    color: ColorPolicy,
    depth: BitDepth,
    codec: VideoCodec,
    backends: &[codec::encode::EncoderBackend],
) -> Option<String> {
    super::caps::check_output_caps(color, depth, codec, backends, None).err().map(|e| e.to_string())
}

fn refusal_pinned(
    color: ColorPolicy,
    depth: BitDepth,
    codec: VideoCodec,
    backends: &[codec::encode::EncoderBackend],
    pinned: codec::encode::EncoderBackend,
) -> Option<String> {
    super::caps::check_output_caps(color, depth, codec, backends, Some(pinned)).err().map(|e| e.to_string())
}

/// A backend asked for by name (`TRANSCODE_ENCODER_BACKEND`) is built whether
/// or not its `-fallback` feature is on (`h26x_sw`: "a caller that wants
/// software encoding can always ask for it by name, feature or no feature"),
/// so it counts for its codec: `h26x` pinned on a hardware-only set serves
/// 10-bit H.264 and H.265. A pin that cannot serve the request adds nothing,
/// and the refusal says what the pin is.
#[test]
fn a_backend_pinned_by_name_counts_without_its_fallback_feature() {
    use codec::encode::EncoderBackend::{Amf, H26x, Nvenc, Qsv, Rav1e};
    for (color, depth) in TEN_BIT_POLICIES {
        // The pin is what serves: without it the same set refuses.
        assert!(refusal(color, depth, VideoCodec::H264, &[Nvenc, Amf, Qsv]).is_some());
        assert_eq!(refusal_pinned(color, depth, VideoCodec::H264, &[Nvenc, Amf, Qsv], H26x), None);
        assert_eq!(refusal_pinned(color, depth, VideoCodec::H264, &[], H26x), None);
        assert_eq!(refusal_pinned(color, depth, VideoCodec::H265, &[], H26x), None);

        // rav1e pinned: 8-bit AV1, and no H.264 at all.
        let err = refusal_pinned(color, depth, VideoCodec::Av1, &[H26x], Rav1e).expect("rav1e is 8-bit");
        assert!(err.contains("this build encodes av1 with rav1e (8-bit SDR)"), "{err}");
        assert!(err.contains("; TRANSCODE_ENCODER_BACKEND=rav1e pins rav1e, which is 8-bit SDR for av1. "), "{err}");
        let err = refusal_pinned(color, depth, VideoCodec::H264, &[Nvenc], Rav1e).expect("rav1e has no H.264");
        assert!(
            err.contains(
                "this build encodes h264 with nvenc (8-bit SDR); TRANSCODE_ENCODER_BACKEND=rav1e pins rav1e, \
                 which does not encode h264. h264 at 10 bits needs the software tier (build with `h26x-fallback`)"
            ),
            "{err}"
        );
        // A pin already in the compiled set is listed once.
        let err = refusal_pinned(color, depth, VideoCodec::H264, &[Nvenc], Nvenc).expect("nvenc H.264 is 8-bit");
        assert_eq!(err.matches("nvenc (8-bit SDR)").count(), 1, "{err}");
        assert!(err.contains("TRANSCODE_ENCODER_BACKEND=nvenc pins nvenc, which is 8-bit SDR for h264"), "{err}");
    }
    // Without a pin the wording is exactly what it was.
    let err = refusal(ColorPolicy::Hdr10, BitDepth::Auto, VideoCodec::H264, &[Nvenc]).unwrap();
    assert!(!err.contains("TRANSCODE_ENCODER_BACKEND"), "{err}");
}

/// The spec-level rule, on this build: pinning `h26x` makes 10-bit H.264 and
/// H.265 valid whatever the features (it is built by name), and pinning
/// `rav1e` never makes 10-bit AV1 valid where the build could not already.
#[test]
fn check_encoder_caps_honours_the_pinned_backend_on_this_build() {
    use codec::encode::EncoderBackend::{H26x, Rav1e};
    for (color, depth) in TEN_BIT_POLICIES {
        for codec in [VideoCodecPolicy::H264, VideoCodecPolicy::H265] {
            let s = OutputSpec::single_file(vec![Rung::new(640, 360)])
                .with_video_codec(codec)
                .with_color(color)
                .with_bit_depth(depth);
            assert!(s.check_encoder_caps(Some(H26x)).is_ok(), "{codec:?} {color:?} {depth:?}: {:?}", s.check_encoder_caps(Some(H26x)).err());
        }
        let av1 = OutputSpec::single_file(vec![Rung::new(640, 360)]).with_color(color).with_bit_depth(depth);
        assert_eq!(av1.check_encoder_caps(Some(Rav1e)).is_ok(), av1.check_encoder_caps(None).is_ok(), "{color:?} {depth:?}");
    }
    // The env spellings the serial encode path accepts.
    for b in ENCODE_BACKENDS {
        assert_eq!(super::caps::encoder_backend_from_name(encode_backend_name(b)), Some(b));
        assert_eq!(super::caps::encoder_backend_from_name(&encode_backend_name(b).to_ascii_uppercase()), Some(b));
    }
    assert_eq!(super::caps::encoder_backend_from_name("x264"), None);
    assert_eq!(super::caps::encoder_backend_from_name(""), None);
}

/// A pin counts only for the mode whose encode path builds it: single-file
/// (the serial encoder reads `TRANSCODE_ENCODER_BACKEND`), never HLS (the
/// ladder leases from the pool). On a build without `h26x-fallback`, HLS
/// H.264 HDR10 with `h26x` pinned is refused with the unpinned wording, where
/// it used to pass and then fail building NVENC after decode had started.
#[test]
fn a_pin_counts_for_single_file_validation_and_not_for_hls() {
    use codec::encode::EncoderBackend::H26x;
    for (color, depth) in TEN_BIT_POLICIES {
        let single = OutputSpec::single_file(vec![Rung::new(640, 360)])
            .with_video_codec(VideoCodecPolicy::H264)
            .with_color(color)
            .with_bit_depth(depth);
        let hls = OutputSpec::hls(vec![Rung::new(640, 360)], 4.0)
            .with_video_codec(VideoCodecPolicy::H264)
            .with_color(color)
            .with_bit_depth(depth);
        assert_eq!(single.pin_honoured(Some(H26x)), Some(H26x));
        assert_eq!(hls.pin_honoured(Some(H26x)), None);
        assert_eq!(hls.pin_honoured(None), None);

        // Single-file: the pin serves 10-bit H.264 whatever the features.
        assert!(
            single.validate_with_pin(Some(H26x)).is_ok(),
            "{color:?} {depth:?}: {:?}",
            single.validate_with_pin(Some(H26x)).err()
        );
        // HLS: exactly what the build says without a pin.
        let pinned = hls.validate_with_pin(Some(H26x)).map_err(|e| format!("{e:#}"));
        let unpinned = hls.validate_with_pin(None).map_err(|e| format!("{e:#}"));
        assert_eq!(pinned, unpinned, "{color:?} {depth:?}");
        if cfg!(feature = "h26x-fallback") {
            assert!(pinned.is_ok(), "{color:?} {depth:?}: {pinned:?}");
        } else {
            let err = pinned.expect_err("HLS has no 10-bit H.264 encoder without h26x-fallback");
            assert!(err.contains("h264 at 10 bits") && err.contains("`h26x-fallback`"), "{err}");
            assert!(!err.contains("TRANSCODE_ENCODER_BACKEND"), "{err}");
        }
    }
}

/// H.264 at 10 bits is the software tier's alone: a set of hardware backends
/// refuses it, says each is 8-bit SDR for H.264, and points at
/// `h26x-fallback` — never at a GPU feature, though the same backends are
/// 10-bit HDR for AV1 and H.265.
#[test]
fn ten_bit_h264_is_refused_on_hardware_and_pointed_at_the_software_tier() {
    use codec::encode::EncoderBackend::{Amf, H26x, Nvenc, Qsv, Rav1e};
    for (color, depth) in TEN_BIT_POLICIES {
        let err = refusal(color, depth, VideoCodec::H264, &[Nvenc, Amf, Qsv]).expect("hardware H.264 is 8-bit");
        assert!(err.starts_with("h264 at 10 bits ("), "{err}");
        assert!(
            err.contains("this build encodes h264 with nvenc (8-bit SDR), amf (8-bit SDR), qsv (8-bit SDR)"),
            "{err}"
        );
        assert!(err.contains("h264 at 10 bits needs the software tier (build with `h26x-fallback`)"), "{err}");
        assert!(err.contains("no hardware backend encodes h264 at 10 bits"), "{err}");
        assert!(!err.contains("`nvidia`") && !err.contains("`amd`") && !err.contains("`qsv`"), "{err}");
        for hw in [Nvenc, Amf, Qsv] {
            assert!(refusal(color, depth, VideoCodec::H264, &[hw]).is_some(), "{hw:?}");
            // The same backend is 10-bit for the other two codecs.
            assert!(refusal(color, depth, VideoCodec::H265, &[hw]).is_none(), "{hw:?}");
        }
        assert!(refusal(color, depth, VideoCodec::H264, &[H26x]).is_none());
        assert!(refusal(color, depth, VideoCodec::H264, &[Nvenc, H26x]).is_none());
        // rav1e does not encode H.264 at all.
        let err = refusal(color, depth, VideoCodec::H264, &[Rav1e]).expect("rav1e has no H.264");
        assert!(err.contains("this build has no h264 encoder"), "{err}");
    }
}

/// AV1 at 10 bits is hardware-only: h26x does not encode AV1 and rav1e is
/// 8-bit, so a software-only set refuses it and names the GPU features and
/// the silicon, and says why the software tier does not count.
#[test]
fn ten_bit_av1_is_refused_without_a_hardware_backend() {
    use codec::encode::EncoderBackend::{Amf, H26x, Nvenc, Qsv, Rav1e};
    for (color, depth) in TEN_BIT_POLICIES {
        let err = refusal(color, depth, VideoCodec::Av1, &[H26x]).expect("h26x has no AV1");
        assert!(err.contains("this build has no av1 encoder"), "{err}");
        assert!(
            err.contains(
                "av1 at 10 bits needs a hardware encoder (build with `nvidia`, `amd` or `qsv`, \
                 on a GPU with AV1 encode: NVIDIA Ada+, AMD RDNA3+, Intel Arc / Meteor Lake+)"
            ),
            "{err}"
        );
        assert!(err.contains("the software av1 tier (`rav1e-fallback`) is 8-bit SDR"), "{err}");
        assert!(!err.contains("h26x-fallback"), "{err}");
        let err = refusal(color, depth, VideoCodec::Av1, &[Rav1e, H26x]).expect("rav1e is 8-bit");
        assert!(err.contains("this build encodes av1 with rav1e (8-bit SDR)"), "{err}");
        for hw in [Nvenc, Amf, Qsv] {
            assert!(refusal(color, depth, VideoCodec::Av1, &[hw, Rav1e]).is_none(), "{hw:?}");
        }
    }
}

/// H.265 at 10 bits has both tiers, and a set with neither names both.
#[test]
fn ten_bit_h265_names_the_hardware_and_the_software_tier() {
    use codec::encode::EncoderBackend::{Amf, H26x, Nvenc, Qsv, Rav1e};
    for (color, depth) in TEN_BIT_POLICIES {
        for set in [&[][..], &[Rav1e][..]] {
            let err = refusal(color, depth, VideoCodec::H265, set).expect("no 10-bit H.265 encoder");
            assert!(err.contains("this build has no h265 encoder"), "{err}");
            assert!(
                err.contains(
                    "h265 at 10 bits needs a hardware encoder (build with `nvidia`, `amd` or `qsv`) \
                     or the software tier (build with `h26x-fallback`)"
                ),
                "{err}"
            );
            assert!(!err.contains("no hardware backend"), "{err}");
        }
        for b in [Nvenc, Amf, Qsv, H26x] {
            assert!(refusal(color, depth, VideoCodec::H265, &[b]).is_none(), "{b:?}");
        }
    }
}

/// Only 10 bits and HDR are capability checks. An 8-bit SDR policy passes on
/// any backend set, one with no encoder for the codec included — as before,
/// that is found when the job builds its encoder.
#[test]
fn eight_bit_policies_are_not_capability_checked() {
    for codec in OUTPUT_CODECS {
        for (color, depth) in [
            (ColorPolicy::TonemapToSdr, BitDepth::Auto),
            (ColorPolicy::TonemapToSdr, BitDepth::EightBit),
            (ColorPolicy::Passthrough, BitDepth::Auto),
            (ColorPolicy::Passthrough, BitDepth::EightBit),
        ] {
            assert!(refusal(color, depth, codec, &[]).is_none(), "{codec:?} {color:?} {depth:?}");
        }
    }
}

/// The per-codec answer rivet reports and validates with is the codec crate's:
/// the build union is `build_output_caps_for`, a one-backend set is
/// `backend_output_caps_for`, a backend that does not encode the codec
/// contributes nothing, and the names are the ones `encode_backends` uses.
#[test]
fn codec_output_caps_agree_with_the_codec_crate() {
    use codec::encode::{
        OutputCaps, backend_output_caps_for, build_output_caps_for, compiled_encode_backends,
        encode_backends,
    };
    let floor = OutputCaps { max_bit_depth: 8, hdr: false };
    for codec in OUTPUT_CODECS {
        assert_eq!(CodecOutputCaps::of_this_build(codec).caps, build_output_caps_for(codec), "{codec:?}");
        for b in ENCODE_BACKENDS {
            let one = CodecOutputCaps::over(codec, &[b]);
            assert_eq!(one.caps, backend_output_caps_for(b, codec), "{b:?} {codec:?}");
            if encode_backend_serves(b, codec) {
                assert_eq!(one.backends, vec![(b, backend_output_caps_for(b, codec))]);
            } else {
                assert!(one.backends.is_empty(), "{b:?} {codec:?}");
                assert_eq!(backend_output_caps_for(b, codec), floor, "{b:?} {codec:?}");
            }
        }
    }
    let names: Vec<&str> = ENCODE_BACKENDS.iter().map(|&b| encode_backend_name(b)).collect();
    assert_eq!(names, ["nvenc", "amf", "qsv", "rav1e", "h26x"]);
    let compiled: Vec<&str> = compiled_encode_backends().into_iter().map(encode_backend_name).collect();
    assert_eq!(compiled, encode_backends());
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

#[test]
fn an_hdr_policy_maps_an_sdr_source_and_refuses_the_other_hdr_transfer() {
    let sdr = ColorMetadata::default();
    let pq = ColorMetadata {
        transfer: TransferFn::St2084,
        matrix_coefficients: 9,
        colour_primaries: 9,
        ..ColorMetadata::default()
    };
    let hlg = ColorMetadata {
        transfer: TransferFn::AribStdB67,
        ..pq
    };
    let spec = || OutputSpec::single_file(vec![Rung::new(1280, 720)]);
    let (hdr10, hlg_spec, sdr_spec, pass) =
        (spec().hdr10(), spec().hlg(), spec(), spec().passthrough());

    // Which transfer the pump maps an SDR source into, if any.
    assert_eq!(hdr10.sdr_to_hdr(&sdr), Some(TransferFn::St2084));
    assert_eq!(hlg_spec.sdr_to_hdr(&sdr), Some(TransferFn::AribStdB67));
    assert_eq!(hdr10.sdr_to_hdr(&pq), None, "PQ under hdr10 passes through");
    assert_eq!(sdr_spec.sdr_to_hdr(&sdr), None);
    assert_eq!(pass.sdr_to_hdr(&sdr), None, "passthrough keeps SDR as SDR");
    // Concat: an SDR clip joined to a PQ first clip under passthrough goes into PQ.
    assert_eq!(sdr_into_hdr(false, &sdr, &pq), Some(TransferFn::St2084));
    assert_eq!(
        sdr_into_hdr(true, &sdr, &pq),
        None,
        "a tonemapping pump maps nothing into HDR"
    );

    // What is refused before decode, by name.
    assert!(
        hdr10.check_source_colour(&sdr).is_ok(),
        "SDR is mapped, not refused"
    );
    assert!(hdr10.check_source_colour(&pq).is_ok());
    assert!(hlg_spec.check_source_colour(&hlg).is_ok());
    let e = format!("{:#}", hlg_spec.check_source_colour(&pq).unwrap_err());
    assert!(
        e.contains("--color hlg on a PQ") && e.contains("--color passthrough"),
        "{e}"
    );
    let e = format!("{:#}", hdr10.check_source_colour(&hlg).unwrap_err());
    assert!(e.contains("--color hdr10 on a HLG"), "{e}");
    let linear = ColorMetadata {
        transfer: TransferFn::Linear,
        ..sdr
    };
    let e = format!("{:#}", hdr10.check_source_colour(&linear).unwrap_err());
    assert!(
        e.contains("--color hdr10 on an SDR source") && e.contains("linear light"),
        "{e}"
    );
    assert!(
        sdr_spec.check_source_colour(&pq).is_ok(),
        "sdr tonemaps any HDR source"
    );
    assert!(
        pass.check_source_colour(&linear).is_ok(),
        "passthrough takes anything"
    );
}

#[test]
fn hdr10_on_an_sdr_source_signals_the_mapped_colour_volume() {
    let spec = || OutputSpec::single_file(vec![Rung::new(1280, 720)]);
    let sdr = ColorMetadata::default();
    let (color, _) = spec().hdr10().resolve_output(sdr, PixelFormat::Yuv420p);
    assert_eq!(color.mastering_display, Some(SDR_IN_PQ_MASTERING_DISPLAY));
    assert_eq!(
        color.content_light_level,
        Some(SDR_IN_PQ_CONTENT_LIGHT_LEVEL)
    );
    let md = SDR_IN_PQ_MASTERING_DISPLAY;
    assert_eq!(
        (md.primaries_r_x, md.primaries_g_y, md.max_luminance),
        (32000, 30000, 2_030_000),
        "BT.709 red x 0.64, green y 0.60 (units of 0.00002), 203 cd/m2 (units of 0.0001)"
    );

    // HLG: none, the signal is scene-referred.
    let (color, _) = spec().hlg().resolve_output(sdr, PixelFormat::Yuv420p);
    assert_eq!(
        (color.mastering_display, color.content_light_level),
        (None, None)
    );

    // A PQ source keeps its own, and one without any is not given the SDR values.
    let pq = hdr_metadata(TransferFn::St2084);
    let (color, _) = spec().hdr10().resolve_output(pq, PixelFormat::Yuv420p10le);
    assert_eq!(
        (color.mastering_display, color.content_light_level),
        (None, None)
    );
    let own = ColorMetadata {
        content_light_level: Some(codec::frame::ContentLightLevel {
            max_cll: 1000,
            max_fall: 400,
        }),
        ..pq
    };
    let (color, _) = spec().hdr10().resolve_output(own, PixelFormat::Yuv420p10le);
    assert_eq!(color.content_light_level.map(|c| c.max_cll), Some(1000));
    assert_eq!(color.mastering_display, None);

    // SDR output: nothing.
    let (color, _) = spec().resolve_output(sdr, PixelFormat::Yuv420p);
    assert_eq!(
        (color.mastering_display, color.content_light_level),
        (None, None)
    );
}
