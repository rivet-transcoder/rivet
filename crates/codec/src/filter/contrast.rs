//! `contrast` — scale luma contrast around mid-grey. 8-bit `Yuv420p` only.

use anyhow::Result;

use super::{assemble, planes_8bit};
use crate::frame::VideoFrame;

/// Scale each luma sample away from / toward mid-grey (128) by `factor`
/// (`1.0` = unchanged, `>1` = more contrast, `<1` = flatter): `out = (in − 128)
/// · factor + 128`, clamped. Chroma untouched. Requires 8-bit SDR output.
pub(super) fn apply(frame: &VideoFrame, factor: f32) -> Result<VideoFrame> {
    let (mut y, u, v) = planes_8bit(frame, "contrast")?;
    for p in y.iter_mut() {
        *p = (((*p as f32 - 128.0) * factor) + 128.0).round().clamp(0.0, 255.0) as u8;
    }
    Ok(assemble(frame, frame.width, frame.height, y, u, v))
}
