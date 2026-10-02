//! ProRes through `decode::create_decoder`.
//!
//! ProRes is decoded by this workspace's own decoder (`crates/prores`, the
//! rivet-prores repository, written clean-room from SMPTE RDD 36), always
//! compiled. Before it, libavcodec was the only backend that took the label,
//! and it was removed on 2026-10-02 (no FFmpeg; see `crates/codec/Cargo.toml`).
//! NVDEC, AMF, QSV and h26x all decline it. What this pins is that the report
//! and the dispatch agree, and that the ProRes tier is there.
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
    assert!(built.is_ok(), "the ProRes tier is always compiled, so create_decoder must build it");
    assert!(backends.contains(&"prores"), "capabilities must list the prores backend: {backends:?}");
}
