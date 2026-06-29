//! `overlay` — alpha-composite a PNG (logo / watermark) onto the frame. This is
//! a **resource** filter: the image is loaded + converted to YUV 4:2:0 + alpha
//! once by [`FilterChain::prepare`](super::FilterChain::prepare) into a
//! [`PreparedOverlay`], which then composites per frame. 8-bit `Yuv420p` only.

use anyhow::{Result, bail};

use super::{assemble, planes_8bit};
use crate::frame::VideoFrame;

/// A loaded overlay image, pre-converted to 8-bit YUV 4:2:0 + per-sample alpha,
/// ready to alpha-composite onto frames.
#[derive(Debug, Clone)]
pub(super) struct PreparedOverlay {
    w: usize,
    h: usize,
    x: usize,
    y: usize,
    y_o: Vec<u8>,
    u_o: Vec<u8>,
    v_o: Vec<u8>,
    a_y: Vec<u8>, // luma-resolution alpha
    a_c: Vec<u8>, // chroma-resolution alpha (2×2 averaged)
}

fn clamp8(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

impl PreparedOverlay {
    /// Convert a row-major RGBA8 buffer (`src_w × src_h`) to a prepared overlay
    /// positioned at `(x, y)`. BT.709 limited-range YUV.
    pub(super) fn from_rgba(rgba: &[u8], src_w: u32, src_h: u32, x: u32, y: u32) -> Result<Self> {
        let w = (src_w & !1) as usize; // even for 4:2:0
        let h = (src_h & !1) as usize;
        if w == 0 || h == 0 {
            bail!("overlay image is too small ({src_w}x{src_h})");
        }
        let stride = src_w as usize * 4;
        let mut y_o = vec![0u8; w * h];
        let mut a_y = vec![0u8; w * h];
        let (cw, ch) = (w / 2, h / 2);
        let mut u_o = vec![0u8; cw * ch];
        let mut v_o = vec![0u8; cw * ch];
        let mut a_c = vec![0u8; cw * ch];
        for r in 0..h {
            for c in 0..w {
                let p = r * stride + c * 4;
                let (rr, gg, bb) = (rgba[p] as i32, rgba[p + 1] as i32, rgba[p + 2] as i32);
                y_o[r * w + c] = clamp8(16 + ((47 * rr + 157 * gg + 16 * bb) >> 8));
                a_y[r * w + c] = rgba[p + 3];
            }
        }
        for r in 0..ch {
            for c in 0..cw {
                let (mut sr, mut sg, mut sb, mut sa) = (0i32, 0i32, 0i32, 0i32);
                for dy in 0..2 {
                    for dx in 0..2 {
                        let p = (r * 2 + dy) * stride + (c * 2 + dx) * 4;
                        sr += rgba[p] as i32;
                        sg += rgba[p + 1] as i32;
                        sb += rgba[p + 2] as i32;
                        sa += rgba[p + 3] as i32;
                    }
                }
                let (rr, gg, bb) = (sr / 4, sg / 4, sb / 4);
                u_o[r * cw + c] = clamp8(128 + ((-26 * rr - 87 * gg + 112 * bb) >> 8));
                v_o[r * cw + c] = clamp8(128 + ((112 * rr - 102 * gg - 10 * bb) >> 8));
                a_c[r * cw + c] = (sa / 4) as u8;
            }
        }
        Ok(Self { w, h, x: (x & !1) as usize, y: (y & !1) as usize, y_o, u_o, v_o, a_y, a_c })
    }

    /// Alpha-composite onto an 8-bit Yuv420p frame: `out = src·(1−α) + ovl·α`.
    pub(super) fn composite(&self, frame: &VideoFrame) -> Result<VideoFrame> {
        let (mut y, mut u, mut v) = planes_8bit(frame, "overlay")?;
        let (fw, fh) = (frame.width as usize, frame.height as usize);
        for r in 0..self.h {
            let fy = self.y + r;
            if fy >= fh {
                break;
            }
            for c in 0..self.w {
                let fx = self.x + c;
                if fx >= fw {
                    continue;
                }
                let a = self.a_y[r * self.w + c] as u32;
                if a == 0 {
                    continue;
                }
                let i = fy * fw + fx;
                y[i] = ((y[i] as u32 * (255 - a) + self.y_o[r * self.w + c] as u32 * a + 127) / 255) as u8;
            }
        }
        let (cw, ch) = (self.w / 2, self.h / 2);
        let (fcw, fch) = (fw / 2, fh / 2);
        let (ocx, ocy) = (self.x / 2, self.y / 2);
        for r in 0..ch {
            let fy = ocy + r;
            if fy >= fch {
                break;
            }
            for c in 0..cw {
                let fx = ocx + c;
                if fx >= fcw {
                    continue;
                }
                let a = self.a_c[r * cw + c] as u32;
                if a == 0 {
                    continue;
                }
                let i = fy * fcw + fx;
                u[i] = ((u[i] as u32 * (255 - a) + self.u_o[r * cw + c] as u32 * a + 127) / 255) as u8;
                v[i] = ((v[i] as u32 * (255 - a) + self.v_o[r * cw + c] as u32 * a + 127) / 255) as u8;
            }
        }
        Ok(assemble(frame, frame.width, frame.height, y, u, v))
    }
}
