//! Audio decoder implementations.
//!
//! See `audio::create_decoder` for the routing entry point.

pub mod ac3;
pub mod dts;
pub mod mp3;
pub mod pcm;
pub mod vorbis;

pub use ac3::Ac3Decoder;
pub use dts::DtsDecoder;
pub use mp3::Mp3Decoder;
pub use pcm::{PcmDecoder, PcmFormat};
pub use vorbis::VorbisDecoder;
