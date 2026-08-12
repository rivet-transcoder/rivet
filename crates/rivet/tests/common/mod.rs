//! Shared helpers for the round-trip fidelity / e2e integration tests.
//!
//! rivet encodes on the GPU (NVENC / AMF / QSV), or in software when built
//! with `rav1e-fallback`. A round-trip test can't assume an encoder exists: it
//! must skip cleanly on a host with no AV1-encode silicon and no software
//! fallback compiled in. These helpers centralize that "encoder-or-skip"
//! decision so every test gates the same way.

#![allow(dead_code)]

use codec::decode::{self, Decoder};
use codec::encode::{self, Encoder, EncoderConfig};
use codec::frame::StreamInfo;

/// Try to build an AV1 encoder for `config`. Returns `None` (the caller should
/// `return` — i.e. skip) when this build/host has no AV1-encode path, printing
/// a clear SKIP line. Returns `Some(encoder)` on a host with NVENC (Ada+) /
/// AMF (RDNA3+) / QSV (Arc+) or a build with `rav1e-fallback`.
pub fn try_av1_encoder(config: EncoderConfig) -> Option<Box<dyn Encoder>> {
    match encode::select_encoder(config, None) {
        Ok(e) => Some(e),
        Err(e) => {
            eprintln!(
                "SKIP: no AV1 encoder on this host/build ({e}); needs NVENC (Ada+) / AMF \
                 (RDNA3+) / QSV (Arc+) or the `rav1e-fallback` feature"
            );
            None
        }
    }
}

/// Build an AV1 decoder for the given stream info (NVDEC / rav1d, whichever the
/// dispatch picks). Returns `None` to skip if no AV1 decoder can be constructed.
pub fn try_av1_decoder(info: StreamInfo) -> Option<Box<dyn Decoder>> {
    match decode::create_decoder("av1", info) {
        Ok(d) => Some(d),
        Err(e) => {
            eprintln!("SKIP: no AV1 decoder on this host/build ({e})");
            None
        }
    }
}

/// `RIVET_TEST_MEDIA` env override, else the workspace `test_media/` dir.
pub fn test_media_dir() -> std::path::PathBuf {
    if let Some(dir) = std::env::var_os("RIVET_TEST_MEDIA") {
        return std::path::PathBuf::from(dir);
    }
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("test_media")
}

/// Read a named file from the test-media dir, or `None` if absent.
pub fn read_test_media(name: &str) -> Option<Vec<u8>> {
    std::fs::read(test_media_dir().join(name)).ok()
}
