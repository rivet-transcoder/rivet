//! `rotate` (alias `transpose` = 90) — rotate clockwise by 90 / 180 / 270°.
//! 90 / 270 swap width ↔ height. Pure sample rearrangement, any bit depth.
//! 180° is composed from [`hflip`](super::hflip) + [`vflip`](super::vflip).

use anyhow::{Result, bail};

use super::{assemble, bps, hflip, planes, vflip};
use crate::frame::VideoFrame;

/// Rotate clockwise by `deg` (must be 90 / 180 / 270).
pub(super) fn apply(frame: &VideoFrame, deg: u32) -> Result<VideoFrame> {
    let bps = bps(frame.format)?;
    let (y, u, v) = planes(frame, bps)?;
    let (w, h) = (frame.width as usize, frame.height as usize);
    let (cw, ch) = (w / 2, h / 2);
    Ok(match deg {
        180 => assemble(
            frame,
            frame.width,
            frame.height,
            vflip::flip(&hflip::flip(y, w, h, bps), w, h, bps),
            vflip::flip(&hflip::flip(u, cw, ch, bps), cw, ch, bps),
            vflip::flip(&hflip::flip(v, cw, ch, bps), cw, ch, bps),
        ),
        90 => assemble(
            frame,
            frame.height,
            frame.width,
            rot90(y, w, h, bps),
            rot90(u, cw, ch, bps),
            rot90(v, cw, ch, bps),
        ),
        270 => assemble(
            frame,
            frame.height,
            frame.width,
            rot270(y, w, h, bps),
            rot270(u, cw, ch, bps),
            rot270(v, cw, ch, bps),
        ),
        d => bail!("rotate must be 90|180|270, got {d}"),
    })
}

/// Rotate 90° clockwise: src `w×h` → dst `h×w`. `dst(r,c) = src(h−1−c, r)`.
fn rot90(src: &[u8], w: usize, h: usize, bps: usize) -> Vec<u8> {
    let (dw, dh) = (h, w);
    let mut out = vec![0u8; dw * dh * bps];
    for r in 0..dh {
        for c in 0..dw {
            let s = ((h - 1 - c) * w + r) * bps;
            let d = (r * dw + c) * bps;
            out[d..d + bps].copy_from_slice(&src[s..s + bps]);
        }
    }
    out
}

/// Rotate 270° clockwise: src `w×h` → dst `h×w`. `dst(r,c) = src(c, w−1−r)`.
fn rot270(src: &[u8], w: usize, h: usize, bps: usize) -> Vec<u8> {
    let (dw, dh) = (h, w);
    let mut out = vec![0u8; dw * dh * bps];
    for r in 0..dh {
        for c in 0..dw {
            let s = (c * w + (w - 1 - r)) * bps;
            let d = (r * dw + c) * bps;
            out[d..d + bps].copy_from_slice(&src[s..s + bps]);
        }
    }
    out
}
