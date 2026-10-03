//! An 8-bit RGBA picture in memory, and the few operations the still-image
//! path does to one: crop, place on a canvas, turn and mirror, and resample
//! (Lanczos-3).
//!
//! The codecs (rivet-png, rivet-jpeg, rivet-gif, rivet-bmp, rivet-tiff, the
//! AV1 decoder through `heif`) each hand back their own pixel layout; this is
//! the one the rest of the module works in.

/// A `width` x `height` picture, four bytes a pixel (R, G, B, A), rows top to
/// bottom, no padding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RgbaImage {
    width: u32,
    height: u32,
    data: Vec<u8>,
}

impl RgbaImage {
    /// Transparent black.
    pub(crate) fn new(width: u32, height: u32) -> Self {
        Self { width, height, data: vec![0; width as usize * height as usize * 4] }
    }

    /// Every pixel `px`.
    pub(crate) fn from_pixel(width: u32, height: u32, px: [u8; 4]) -> Self {
        let n = width as usize * height as usize;
        let mut data = Vec::with_capacity(n * 4);
        for _ in 0..n {
            data.extend_from_slice(&px);
        }
        Self { width, height, data }
    }

    /// Each pixel from `f(x, y)`.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn from_fn(width: u32, height: u32, mut f: impl FnMut(u32, u32) -> [u8; 4]) -> Self {
        let mut data = Vec::with_capacity(width as usize * height as usize * 4);
        for y in 0..height {
            for x in 0..width {
                data.extend_from_slice(&f(x, y));
            }
        }
        Self { width, height, data }
    }

    /// `data` as the picture's pixels, or `None` when it is not
    /// `width * height * 4` bytes.
    pub(crate) fn from_raw(width: u32, height: u32, data: Vec<u8>) -> Option<Self> {
        (data.len() == width as usize * height as usize * 4).then_some(Self { width, height, data })
    }

    /// RGB triplets, opaque.
    pub(crate) fn from_rgb(width: u32, height: u32, rgb: &[u8]) -> Option<Self> {
        if rgb.len() != width as usize * height as usize * 3 {
            return None;
        }
        let mut data = Vec::with_capacity(rgb.len() / 3 * 4);
        for p in rgb.chunks_exact(3) {
            data.extend_from_slice(&[p[0], p[1], p[2], u8::MAX]);
        }
        Some(Self { width, height, data })
    }

    pub(crate) fn width(&self) -> u32 {
        self.width
    }

    pub(crate) fn height(&self) -> u32 {
        self.height
    }

    pub(crate) fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// The pixels, R G B A, row by row.
    pub(crate) fn as_raw(&self) -> &[u8] {
        &self.data
    }

    /// RGB triplets, alpha dropped.
    pub(crate) fn to_rgb(&self) -> Vec<u8> {
        self.pixels().flat_map(|p| [p[0], p[1], p[2]]).collect()
    }

    pub(crate) fn pixels(&self) -> impl Iterator<Item = &[u8; 4]> {
        self.data.as_chunks::<4>().0.iter()
    }

    pub(crate) fn pixels_mut(&mut self) -> impl Iterator<Item = &mut [u8; 4]> {
        self.data.as_chunks_mut::<4>().0.iter_mut()
    }

    /// Every pixel with its position.
    pub(crate) fn enumerate_pixels_mut(&mut self) -> impl Iterator<Item = (u32, u32, &mut [u8; 4])> {
        let w = self.width.max(1);
        self.pixels_mut().enumerate().map(move |(i, p)| (i as u32 % w, i as u32 / w, p))
    }

    pub(crate) fn get_pixel(&self, x: u32, y: u32) -> [u8; 4] {
        let at = (y as usize * self.width as usize + x as usize) * 4;
        [self.data[at], self.data[at + 1], self.data[at + 2], self.data[at + 3]]
    }

    /// Whether any pixel is less than opaque.
    pub(crate) fn has_alpha(&self) -> bool {
        self.pixels().any(|p| p[3] != u8::MAX)
    }

    /// The `w` x `h` region at (`x`, `y`), which must lie inside.
    pub(crate) fn crop(&self, x: u32, y: u32, w: u32, h: u32) -> Self {
        let (x, y) = (x.min(self.width), y.min(self.height));
        let (w, h) = (w.min(self.width - x), h.min(self.height - y));
        let mut data = Vec::with_capacity(w as usize * h as usize * 4);
        for row in y..y + h {
            let at = (row as usize * self.width as usize + x as usize) * 4;
            data.extend_from_slice(&self.data[at..at + w as usize * 4]);
        }
        Self { width: w, height: h, data }
    }

    /// Copy `top` over this picture with its top-left at (`x`, `y`), clipped
    /// to this picture's edges.
    pub(crate) fn replace(&mut self, top: &RgbaImage, x: i64, y: i64) {
        for ty in 0..i64::from(top.height) {
            let dy = y + ty;
            if dy < 0 || dy >= i64::from(self.height) {
                continue;
            }
            let x0 = x.max(0);
            let x1 = (x + i64::from(top.width)).min(i64::from(self.width));
            if x0 >= x1 {
                continue;
            }
            let src = ((ty * i64::from(top.width) + (x0 - x)) * 4) as usize;
            let dst = ((dy * i64::from(self.width) + x0) * 4) as usize;
            let n = ((x1 - x0) * 4) as usize;
            self.data[dst..dst + n].copy_from_slice(&top.data[src..src + n]);
        }
    }

    /// A new picture, `w` x `h`, whose pixel (x, y) is this one's `at(x, y)`.
    fn remap(&self, w: u32, h: u32, at: impl Fn(u32, u32) -> (u32, u32)) -> Self {
        let mut data = Vec::with_capacity(self.data.len());
        for y in 0..h {
            for x in 0..w {
                let (sx, sy) = at(x, y);
                data.extend_from_slice(&self.get_pixel(sx, sy));
            }
        }
        Self { width: w, height: h, data }
    }

    /// Turned a quarter turn clockwise.
    pub(crate) fn rotate90(&self) -> Self {
        let h = self.height;
        self.remap(self.height, self.width, |x, y| (y, h - 1 - x))
    }

    /// Turned half a turn.
    pub(crate) fn rotate180(&self) -> Self {
        let (w, h) = (self.width, self.height);
        self.remap(w, h, |x, y| (w - 1 - x, h - 1 - y))
    }

    /// Turned three quarter turns clockwise (a quarter turn anticlockwise).
    pub(crate) fn rotate270(&self) -> Self {
        let w = self.width;
        self.remap(self.height, self.width, |x, y| (w - 1 - y, x))
    }

    /// Mirrored left to right.
    pub(crate) fn flip_horizontal(&self) -> Self {
        let w = self.width;
        self.remap(w, self.height, |x, y| (w - 1 - x, y))
    }

    /// Mirrored top to bottom.
    pub(crate) fn flip_vertical(&self) -> Self {
        let h = self.height;
        self.remap(self.width, h, |x, y| (x, h - 1 - y))
    }

    /// Upright for an EXIF / TIFF orientation (1-8: where the stored first
    /// row and column belong; anything else is 1).
    pub(crate) fn oriented(self, orientation: u16) -> Self {
        match orientation {
            2 => self.flip_horizontal(),
            3 => self.rotate180(),
            4 => self.flip_vertical(),
            5 => self.rotate90().flip_horizontal(),
            6 => self.rotate90(),
            7 => self.rotate270().flip_horizontal(),
            8 => self.rotate270(),
            _ => self,
        }
    }

    /// Resampled to `w` x `h` with a Lanczos-3 filter, widened by the scale
    /// factor when shrinking so every source pixel contributes. Separable:
    /// rows, then columns, in `f32`.
    pub(crate) fn resize(&self, w: u32, h: u32) -> Self {
        if (w, h) == (self.width, self.height) {
            return self.clone();
        }
        if w == 0 || h == 0 || self.width == 0 || self.height == 0 {
            return Self::new(w, h);
        }
        let (sw, sh) = (self.width as usize, self.height as usize);
        let (dw, dh) = (w as usize, h as usize);
        let xw = weights(sw, dw);
        let yw = weights(sh, dh);
        // Rows: sh x dw.
        let mut mid = vec![0f32; sh * dw * 4];
        for y in 0..sh {
            let row = &self.data[y * sw * 4..(y + 1) * sw * 4];
            for (x, (start, taps)) in xw.iter().enumerate() {
                let mut acc = [0f32; 4];
                for (k, &wt) in taps.iter().enumerate() {
                    let p = &row[(start + k) * 4..(start + k) * 4 + 4];
                    for c in 0..4 {
                        acc[c] += wt * f32::from(p[c]);
                    }
                }
                mid[(y * dw + x) * 4..(y * dw + x) * 4 + 4].copy_from_slice(&acc);
            }
        }
        // Columns: dh x dw.
        let mut data = vec![0u8; dw * dh * 4];
        for (y, (start, taps)) in yw.iter().enumerate() {
            for x in 0..dw {
                let mut acc = [0f32; 4];
                for (k, &wt) in taps.iter().enumerate() {
                    let at = ((start + k) * dw + x) * 4;
                    for c in 0..4 {
                        acc[c] += wt * mid[at + c];
                    }
                }
                for c in 0..4 {
                    data[(y * dw + x) * 4 + c] = acc[c].round().clamp(0.0, 255.0) as u8;
                }
            }
        }
        Self { width: w, height: h, data }
    }
}

fn lanczos3(x: f32) -> f32 {
    let x = x.abs();
    if x < 1e-6 {
        return 1.0;
    }
    if x >= 3.0 {
        return 0.0;
    }
    let px = std::f32::consts::PI * x;
    3.0 * px.sin() * (px / 3.0).sin() / (px * px)
}

/// For each of `dst` output samples along an axis of `src`: the first source
/// sample it reads and the normalised weights of the samples from there.
fn weights(src: usize, dst: usize) -> Vec<(usize, Vec<f32>)> {
    let ratio = src as f32 / dst as f32;
    let scale = ratio.max(1.0);
    let support = 3.0 * scale;
    (0..dst)
        .map(|i| {
            let centre = (i as f32 + 0.5) * ratio;
            let lo = ((centre - support).floor() as i64).max(0) as usize;
            let hi = ((centre + support).ceil() as i64).min(src as i64) as usize;
            let mut taps: Vec<f32> = (lo..hi).map(|j| lanczos3((j as f32 + 0.5 - centre) / scale)).collect();
            let sum: f32 = taps.iter().sum();
            if sum.abs() > 1e-6 {
                for t in &mut taps {
                    *t /= sum;
                }
            } else {
                // Nothing in reach (cannot happen for src >= 1): the nearest.
                let near = (centre as usize).min(src - 1);
                return (near, vec![1.0]);
            }
            (lo, taps)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn numbered(w: u32, h: u32) -> RgbaImage {
        RgbaImage::from_fn(w, h, |x, y| [x as u8, y as u8, 0, 255])
    }

    #[test]
    fn turns_and_mirrors_put_each_pixel_where_it_belongs() {
        let img = numbered(3, 2);
        // A quarter turn clockwise: the bottom-left corner goes to the top-left.
        let r = img.rotate90();
        assert_eq!(r.dimensions(), (2, 3));
        assert_eq!(r.get_pixel(0, 0)[..2], [0, 1]);
        assert_eq!(r.get_pixel(1, 0)[..2], [0, 0]);
        let r = img.rotate270();
        assert_eq!(r.get_pixel(0, 0)[..2], [2, 0]);
        assert_eq!(img.rotate180().get_pixel(0, 0)[..2], [2, 1]);
        assert_eq!(img.flip_horizontal().get_pixel(0, 1)[..2], [2, 1]);
        assert_eq!(img.flip_vertical().get_pixel(0, 0)[..2], [0, 1]);
        assert_eq!(img.rotate90().rotate270(), img);
        // EXIF 6 is "turn 90 clockwise to be upright".
        assert_eq!(img.clone().oriented(6), img.rotate90());
    }

    #[test]
    fn crop_and_replace_clip_at_the_edges() {
        let img = numbered(4, 4);
        let c = img.crop(1, 2, 2, 2);
        assert_eq!(c.get_pixel(0, 0)[..2], [1, 2]);
        let mut canvas = RgbaImage::new(3, 3);
        canvas.replace(&img, -2, 1);
        assert_eq!(canvas.get_pixel(0, 1)[..2], [2, 0]);
        assert_eq!(canvas.get_pixel(0, 0), [0, 0, 0, 0]);
    }

    #[test]
    fn resampling_keeps_flat_colour_and_an_edge() {
        let flat = RgbaImage::from_pixel(37, 23, [10, 200, 90, 255]);
        let small = flat.resize(9, 5);
        assert!(small.pixels().all(|p| *p == [10, 200, 90, 255]));
        let big = flat.resize(80, 41);
        assert!(big.pixels().all(|p| *p == [10, 200, 90, 255]));
        let edge = RgbaImage::from_fn(64, 8, |x, _| if x < 32 { [0, 0, 0, 255] } else { [255, 255, 255, 255] });
        let half = edge.resize(32, 4);
        assert!(half.get_pixel(4, 2)[0] < 10 && half.get_pixel(28, 2)[0] > 245);
    }
}
