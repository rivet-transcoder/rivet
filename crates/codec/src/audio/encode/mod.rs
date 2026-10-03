//! Audio encoder implementations, each an adapter over one of the
//! workspace's own codec crates.

pub mod aac;
pub mod ac3;
pub mod alac;
pub mod dts;
pub mod flac;
pub mod mp3;
pub mod opus;
pub mod vorbis;

pub use alac::AlacEncoder;
pub use flac::FlacEncoder;
pub use opus::OpusEncoder;
