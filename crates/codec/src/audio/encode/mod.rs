//! Audio encoder implementations.

#[cfg(feature = "lame")]
pub mod mp3;
pub mod opus;

pub use opus::OpusEncoder;
