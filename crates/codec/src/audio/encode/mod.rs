//! Audio encoder implementations.

pub mod aac;
pub mod alac;
pub mod flac;
#[cfg(feature = "lame")]
pub mod mp3;
pub mod opus;

pub use alac::AlacEncoder;
pub use flac::FlacEncoder;
pub use opus::OpusEncoder;
