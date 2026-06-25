//! Audio decoder implementations.
//!
//! See `audio::create_decoder` for the routing entry point.

pub mod mp3;
pub mod vorbis;

pub use mp3::Mp3Decoder;
pub use vorbis::VorbisDecoder;
