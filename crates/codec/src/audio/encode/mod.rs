//! Audio encoder implementations.

pub mod aac;
#[cfg(feature = "lame")]
pub mod mp3;
pub mod opus;

pub use opus::OpusEncoder;
