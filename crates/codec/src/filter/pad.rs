//! `pad` — letterbox / pillarbox the frame into a larger `w×h` canvas filled
//! with neutral (limited-range) black. With explicit `x`/`y` the source is
//! placed there; without, it is centred. Even-aligned for 4:2:0. Any bit depth.

use anyhow::{Result, bail};

use super::{assemble, bps, even, planes};
use crate::frame::{PixelFormat, VideoFrame};

/// Pad into a `pw×ph` canvas. `x`/`y` give the source's top-left position; when
/// either is `None` the source is centred. `pw`/`ph` are clamped to be at least
/// the frame size.
pub(super) fn apply(
    frame: &VideoFrame,
    pw: u32,
    ph: u32,
    x: Option<u32>,
    y: Option<u32>,
) -> Result<VideoFrame> {
    let pw = even(pw.max(frame.width));
    let ph = even(ph.max(frame.height));
    let px = x.map(even).unwrap_or_else(|| even(pw.saturating_sub(frame.width) / 2));
    let py = y.map(even).unwrap_or_else(|| even(ph.saturating_sub(frame.height) / 2));
    pad(frame, pw, ph, px, py)
}

fn pad(frame: &VideoFrame, pw: u32, ph: u32, x: u32, y: u32) -> Result<VideoFrame> {
    let (pw, ph, x, y) = (even(pw), even(ph), even(x), even(y));
    if x + frame.width > pw || y + frame.height > ph {
        bail!("pad {pw}x{ph} with frame {}x{} at +{x}+{y} overflows", frame.width, frame.height);
    }
    let bps = bps(frame.format)?;
    let (yp, up, vp) = planes(frame, bps)?;
    let (luma_fill, chroma_fill) = black_fill(frame.format);
    let (fw, fh) = (frame.width as usize, frame.height as usize);
    let y_new = pad_plane(yp, fw, fh, pw as usize, ph as usize, x as usize, y as usize, bps, &luma_fill);
    let ca = (fw / 2, fh / 2, (pw / 2) as usize, (ph / 2) as usize, (x / 2) as usize, (y / 2) as usize);
    let u_new = pad_plane(up, ca.0, ca.1, ca.2, ca.3, ca.4, ca.5, bps, &chroma_fill);
    let v_new = pad_plane(vp, ca.0, ca.1, ca.2, ca.3, ca.4, ca.5, bps, &chroma_fill);
    Ok(assemble(frame, pw, ph, y_new, u_new, v_new))
}

/// Place an `sw×sh` plane at `(ox, oy)` inside a `dw×dh` canvas pre-filled with
/// `fill_sample` (`bps`-byte samples).
fn pad_plane(
    src: &[u8],
    sw: usize,
    sh: usize,
    dw: usize,
    dh: usize,
    ox: usize,
    oy: usize,
    bps: usize,
    fill_sample: &[u8],
) -> Vec<u8> {
    let mut out = Vec::with_capacity(dw * dh * bps);
    for _ in 0..dw * dh {
        out.extend_from_slice(fill_sample);
    }
    for row in 0..sh {
        let s = row * sw * bps;
        let d = ((oy + row) * dw + ox) * bps;
        out[d..d + sw * bps].copy_from_slice(&src[s..s + sw * bps]);
    }
    out
}

/// Limited-range black: luma 16, chroma 128 (8-bit); luma 64, chroma 512 (10-bit).
fn black_fill(format: PixelFormat) -> (Vec<u8>, Vec<u8>) {
    match format {
        PixelFormat::Yuv420p => (vec![16], vec![128]),
        _ => ((64u16).to_le_bytes().to_vec(), (512u16).to_le_bytes().to_vec()),
    }
}
