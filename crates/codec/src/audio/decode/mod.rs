//! Audio decoder implementations.
//!
//! See `audio::create_decoder` for the routing entry point.

pub mod aac;
pub mod ac3;
pub mod alac;
pub mod dts;
pub mod flac;
pub mod mp3;
pub mod opus;
pub mod pcm;
pub mod vorbis;

pub use aac::AacDecoder;
pub use ac3::Ac3Decoder;
pub use alac::AlacDecoder;
pub use dts::DtsDecoder;
pub use flac::FlacDecoder;
pub use mp3::Mp3Decoder;
pub use opus::OpusDecoder;
pub use pcm::{PcmDecoder, PcmFormat};
pub use vorbis::VorbisDecoder;
