//! `saturation` — scale chroma intensity around neutral. 8-bit `Yuv420p` only.

use anyhow::Result;

use super::{assemble, planes_8bit};
use crate::frame::VideoFrame;

/// Scale both chroma planes away from / toward neutral (128) by `factor`
/// (`0` = grayscale, `1.0` = unchanged, `>1` = more saturated): `out = (in −
/// 128) · factor + 128`, clamped. Luma untouched. Requires 8-bit SDR output.
pub(super) fn apply(frame: &VideoFrame, factor: f32) -> Result<VideoFrame> {
    let (y, mut u, mut v) = planes_8bit(frame, "saturation")?;
    for p in u.iter_mut().chain(v.iter_mut()) {
        *p = (((*p as f32 - 128.0) * factor) + 128.0).round().clamp(0.0, 255.0) as u8;
    }
    Ok(assemble(frame, frame.width, frame.height, y, u, v))
}
