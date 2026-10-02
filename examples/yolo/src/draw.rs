//! Boxes drawn onto a frame, to see what the detector saw.

use std::path::Path;

use anyhow::{Context, Result};
use rivet::codec::frame::VideoFrame;

use crate::yolo::Detection;

/// One colour per class, cycling.
const PALETTE: [[u8; 3]; 8] = [
    [255, 56, 56],
    [255, 157, 151],
    [255, 178, 29],
    [72, 249, 10],
    [26, 147, 52],
    [0, 194, 255],
    [52, 69, 147],
    [203, 56, 255],
];

/// `frame` with an outline round each detection, written to `path` as PNG.
pub fn boxes(frame: &VideoFrame, detections: &[Detection], path: &Path) -> Result<()> {
    let (w, h) = (frame.width, frame.height);
    let mut rgb = rivet::hooks::frame::rgb8(frame)?;
    let thickness = (w.max(h) / 400).max(2) as i64;
    for d in detections {
        let colour = PALETTE[d.class % PALETTE.len()];
        let (x0, y0) = (d.x.round() as i64, d.y.round() as i64);
        let (x1, y1) = ((d.x + d.w).round() as i64, (d.y + d.h).round() as i64);
        for t in 0..thickness {
            for x in x0..=x1 {
                put(&mut rgb, w, h, x, y0 + t, colour);
                put(&mut rgb, w, h, x, y1 - t, colour);
            }
            for y in y0..=y1 {
                put(&mut rgb, w, h, x0 + t, y, colour);
                put(&mut rgb, w, h, x1 - t, y, colour);
            }
        }
    }
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    image::save_buffer(path, &rgb, w, h, image::ExtendedColorType::Rgb8).with_context(|| format!("writing {}", path.display()))
}

fn put(rgb: &mut [u8], w: u32, h: u32, x: i64, y: i64, colour: [u8; 3]) {
    if x >= 0 && y >= 0 && (x as u32) < w && (y as u32) < h {
        let i = (y as usize * w as usize + x as usize) * 3;
        rgb[i..i + 3].copy_from_slice(&colour);
    }
}
