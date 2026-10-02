//! ProRes through `decode::create_decoder`.
//!
//! No backend in this workspace decodes ProRes. libavcodec was the only one
//! that took the label, and it was removed for good on 2026-10-02 (the
//! project takes no dependency on FFmpeg; see `crates/codec/Cargo.toml`).
//! NVDEC, AMF, QSV and h26x all decline it. What this pins is that the
//! report and the dispatch agree about that.
//!
//! The fourcc-to-label mapping for the six Apple ProRes fourccs is covered
//! by the container demuxer's unit tests; this is the layer after it.

use frame::{ColorSpace, PixelFormat, StreamInfo};

fn prores_info() -> StreamInfo {
    StreamInfo {
        codec: "prores".into(),
        width: 1280,
        height: 720,
        frame_rate: 24.0,
        duration: 0.0,
        pixel_format: PixelFormat::Yuv422p10le,
        color_space: ColorSpace::Bt709,
        total_frames: 0,
        bitrate: 0,
        color_metadata: Default::default(),
    }
}

/// Every build: `rivet capabilities` lists a ProRes backend exactly when
/// `create_decoder` can build one. Advertising a decoder that is never
/// constructed is how the first FFmpeg integration was lost; refusing one
/// that is advertised is the same lie the other way round.
#[test]
fn prores_is_advertised_exactly_when_create_decoder_builds_it() {
    let backends = codec::decode::decode_capabilities()
        .into_iter()
        .find(|s| s.codec == "prores")
        .expect("prores row in decode_capabilities")
        .backends;
    let built = codec::decode::create_decoder("prores", prores_info());
    match &built {
        Ok(_) => assert!(
            !backends.is_empty(),
            "create_decoder built a ProRes decoder that capabilities does not list"
        ),
        Err(e) => {
            assert!(
                backends.is_empty(),
                "capabilities lists ProRes backends {backends:?} but create_decoder refused: {e:#}"
            );
            assert!(
                format!("{e:#}").contains("'prores'"),
                "refusal must name the codec: {e:#}"
            );
        }
    }
    #[cfg(not(any(feature = "nvidia", feature = "amd", feature = "qsv")))]
    assert!(
        built.is_err(),
        "no backend in this build decodes ProRes, so create_decoder must refuse it"
    );
}
