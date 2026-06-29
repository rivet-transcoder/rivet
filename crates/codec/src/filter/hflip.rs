//! `hflip` — mirror horizontally (left ↔ right). Pure sample rearrangement, so
//! it works at any bit depth (8- or 10-bit). `rotate=180` reuses [`flip`].

use anyhow::Result;

use super::{assemble, bps, planes};
use crate::frame::VideoFrame;

/// Mirror each plane left-to-right.
pub(super) fn apply(frame: &VideoFrame) -> Result<VideoFrame> {
    let bps = bps(frame.format)?;
    let (y, u, v) = planes(frame, bps)?;
    let (w, h) = (frame.width as usize, frame.height as usize);
    Ok(assemble(
        frame,
        frame.width,
        frame.height,
        flip(y, w, h, bps),
        flip(u, w / 2, h / 2, bps),
        flip(v, w / 2, h / 2, bps),
    ))
}

/// Reverse every row of a `w×h` plane (`bps`-byte samples).
pub(super) fn flip(src: &[u8], w: usize, h: usize, bps: usize) -> Vec<u8> {
    let mut out = vec![0u8; w * h * bps];
    for row in 0..h {
        let base = row * w * bps;
        for col in 0..w {
            let s = base + col * bps;
            let d = base + (w - 1 - col) * bps;
            out[d..d + bps].copy_from_slice(&src[s..s + bps]);
        }
    }
    out
}
