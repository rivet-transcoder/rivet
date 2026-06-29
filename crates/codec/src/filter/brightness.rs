//! `brightness` — add a luma offset (brighten / darken). 8-bit `Yuv420p` only.

use anyhow::Result;

use super::{assemble, planes_8bit};
use crate::frame::VideoFrame;

/// Add `delta` (`-255..=255`) to every luma sample, clamped to `0..=255`.
/// Chroma is untouched, so only perceived brightness changes, not hue. Requires
/// 8-bit SDR output.
pub(super) fn apply(frame: &VideoFrame, delta: i32) -> Result<VideoFrame> {
    let (mut y, u, v) = planes_8bit(frame, "brightness")?;
    for p in y.iter_mut() {
        *p = (*p as i32 + delta).clamp(0, 255) as u8;
    }
    Ok(assemble(frame, frame.width, frame.height, y, u, v))
}
