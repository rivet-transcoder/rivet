//! `crop` — cut out a `w×h` region. With explicit `x`/`y` it crops at that
//! offset; without, it crops centred (and clamps `w`/`h` to the frame). All
//! values round to even for 4:2:0 chroma alignment. Any bit depth.

use anyhow::{Result, bail};

use super::{assemble, bps, even, planes};
use crate::frame::VideoFrame;

/// Crop to `cw×ch`. `x`/`y` give the top-left offset; when either is `None` the
/// crop is centred and the size is clamped to the frame.
pub(super) fn apply(
    frame: &VideoFrame,
    cw: u32,
    ch: u32,
    x: Option<u32>,
    y: Option<u32>,
) -> Result<VideoFrame> {
    match (x, y) {
        (Some(x), Some(y)) => crop(frame, x, y, cw, ch),
        _ => {
            let cw = even(cw.min(frame.width));
            let ch = even(ch.min(frame.height));
            let cx = even(frame.width.saturating_sub(cw) / 2);
            let cy = even(frame.height.saturating_sub(ch) / 2);
            crop(frame, cx, cy, cw, ch)
        }
    }
}

fn crop(frame: &VideoFrame, x: u32, y: u32, w: u32, h: u32) -> Result<VideoFrame> {
    let (x, y, w, h) = (even(x), even(y), even(w), even(h));
    if w == 0 || h == 0 || x + w > frame.width || y + h > frame.height {
        bail!("crop {w}x{h}+{x}+{y} out of bounds for {}x{}", frame.width, frame.height);
    }
    let bps = bps(frame.format)?;
    let (yp, up, vp) = planes(frame, bps)?;
    let fw = frame.width as usize;
    let y_new = crop_plane(yp, fw, x as usize, y as usize, w as usize, h as usize, bps);
    let cargs = ((x / 2) as usize, (y / 2) as usize, (w / 2) as usize, (h / 2) as usize);
    let u_new = crop_plane(up, fw / 2, cargs.0, cargs.1, cargs.2, cargs.3, bps);
    let v_new = crop_plane(vp, fw / 2, cargs.0, cargs.1, cargs.2, cargs.3, bps);
    Ok(assemble(frame, w, h, y_new, u_new, v_new))
}

/// Copy a `cw×ch` window at `(x, y)` out of a `pw`-wide plane (`bps`-byte samples).
fn crop_plane(src: &[u8], pw: usize, x: usize, y: usize, cw: usize, ch: usize, bps: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(cw * ch * bps);
    for row in 0..ch {
        let start = ((y + row) * pw + x) * bps;
        out.extend_from_slice(&src[start..start + cw * bps]);
    }
    out
}
