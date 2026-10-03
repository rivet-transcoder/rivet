//! WebP, in and out — waiting for rivet-webp.
//!
//! WebP was read by the `image` crate and written by libwebp (through the
//! `webp` / `libwebp-sys` crates) until 2026-10-03, when every still-image
//! codec moved to this workspace's own clean-room crates. The WebP one,
//! rivet-webp (RFC 9649: lossy through rivet-vp8, lossless VP8L, alpha,
//! animation, ICC / EXIF / XMP), is being written and is not published yet,
//! so until it lands a WebP input or a WebP output is refused by name — up
//! front, when the spec is checked, not after a decode.
//!
//! TODO(rivet-webp): when `rivet-transcoder/rivet-webp` is published, add it
//! as a submodule (`crates/webp`, imported as `webp`; patch its `rivet-vp8`
//! git dependency to `crates/vp8` as rivet-png's is patched for rivet-tiff),
//! set [`AVAILABLE`] and fill in the three functions below. The surface the
//! rest of the module needs is exactly this:
//!
//! ```text
//! read_header(data)  -> webp::probe(data)          -> (width, height)
//! decode(data)       -> webp::decode(data)         -> RGBA + probe().icc
//! encode(px, q, ll)  -> webp::encode(&image, &webp::EncoderConfig {
//!                           quality, icc_profile, ..lossless() or default })
//! ```
//!
//! `ImageFormat::Webp`, `SourceFormat::Webp`, the `image-lossless` setting
//! and WebP's EXIF writer (`container::metadata::write::still`) are all still
//! in place; nothing else changes when the codec arrives.

use anyhow::{Result, bail};

use super::decode::Picture;
use super::scale::Pixels;

/// Whether this build reads and writes WebP.
pub const AVAILABLE: bool = false;

/// Why WebP is refused, in words a caller can match on (`rivet-webp`).
pub const UNAVAILABLE: &str = "WebP is not available in this build: rivet's own WebP codec (rivet-webp) has not \
                               landed yet. Ask for avif, jpeg or png";

/// The picture's size.
pub(crate) fn read_header(_data: &[u8]) -> Result<(u32, u32)> {
    bail!("{UNAVAILABLE}")
}

/// Decode a WebP still (an animation's first frame).
pub(crate) fn decode(_data: &[u8]) -> Result<Picture> {
    bail!("{UNAVAILABLE}")
}

/// Encode lossy at `quality` (1-100) or lossless, with the ICC profile.
pub(crate) fn encode(_pixels: &Pixels<'_>, _quality: u8, _lossless: bool) -> Result<Vec<u8>> {
    bail!("{UNAVAILABLE}")
}
