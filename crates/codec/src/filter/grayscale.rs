//! `grayscale` (alias `gray`) — drop colour by setting both chroma planes to
//! neutral, keeping luma. Pure sample rewrite, any bit depth.

use anyhow::Result;

use super::{assemble, bps, planes};
use crate::frame::{PixelFormat, VideoFrame};

/// Set U and V to their neutral value (mid-range), leaving luma intact.
pub(super) fn apply(frame: &VideoFrame) -> Result<VideoFrame> {
    let bps = bps(frame.format)?;
    let (y, u, v) = planes(frame, bps)?;
    let neutral = neutral_chroma(frame.format);
    let mut uu = u.to_vec();
    let mut vv = v.to_vec();
    fill(&mut uu, &neutral);
    fill(&mut vv, &neutral);
    Ok(assemble(frame, frame.width, frame.height, y.to_vec(), uu, vv))
}

/// Overwrite `buf` with repeats of `sample` (one chroma value per sample slot).
fn fill(buf: &mut [u8], sample: &[u8]) {
    for chunk in buf.chunks_exact_mut(sample.len()) {
        chunk.copy_from_slice(sample);
    }
}

/// Neutral chroma sample bytes (mid-range): 128 for 8-bit, 512 for 10-bit LE.
fn neutral_chroma(format: PixelFormat) -> Vec<u8> {
    match format {
        PixelFormat::Yuv420p => vec![128],
        _ => (512u16).to_le_bytes().to_vec(),
    }
}
