//! `invert` (alias `negate`) — photo-negative the frame. 8-bit `Yuv420p` only.

use anyhow::Result;

use super::{assemble, planes_8bit};
use crate::frame::VideoFrame;

/// Negate every sample (`out = 255 − in`) on luma **and** chroma. Requires
/// 8-bit SDR output — `planes_8bit` errors on a 10-bit frame.
pub(super) fn apply(frame: &VideoFrame) -> Result<VideoFrame> {
    let (mut y, mut u, mut v) = planes_8bit(frame, "invert")?;
    for b in y.iter_mut().chain(u.iter_mut()).chain(v.iter_mut()) {
        *b = 255 - *b;
    }
    Ok(assemble(frame, frame.width, frame.height, y, u, v))
}
