//! ProRes through `decode::create_decoder`.
//!
//! Moved here from `crates/container/tests/ts_avi_mov_demux.rs`: the test
//! never touched the container crate, and ProRes decode exists only behind
//! this crate's `ffmpeg` feature (libavcodec is the one backend that takes
//! the label — NVDEC, AMF, QSV and h26x all decline it). In the container
//! crate it could not be feature-gated, since that crate has no `ffmpeg`
//! feature, so it failed on every build without one.
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

/// With libavcodec compiled in, the label reaches a decoder: construction
/// succeeds, and an empty stream finishes with no frame.
#[cfg(feature = "ffmpeg")]
#[test]
fn create_decoder_accepts_prores_codec_label() {
    let mut dec = codec::decode::create_decoder("prores", prores_info())
        .expect("ProRes decoder must be wired in create_decoder dispatch");
    dec.finish().expect("finish");
    let frame = dec.decode_next().expect("decode_next on empty input");
    assert!(frame.is_none(), "no samples → no frame");
}

/// Every build: `rivet capabilities` lists a ProRes backend exactly when
/// `create_decoder` can build one. Advertising a decoder that is never
/// constructed is how the first FFmpeg integration was lost (see the
/// `ffmpeg` feature in Cargo.toml); refusing one that is advertised is the
/// same lie the other way round.
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
    #[cfg(feature = "ffmpeg")]
    assert!(
        backends.contains(&"ffmpeg"),
        "ffmpeg build must list ffmpeg for ProRes"
    );
    #[cfg(not(any(
        feature = "ffmpeg",
        feature = "nvidia",
        feature = "amd",
        feature = "qsv"
    )))]
    assert!(
        built.is_err(),
        "no backend in this build decodes ProRes, so create_decoder must refuse it"
    );
}
