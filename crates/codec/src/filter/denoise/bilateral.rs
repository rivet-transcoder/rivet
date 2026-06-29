//! Bilateral denoise (edge-preserving).

/// Edge-preserving bilateral filter over a 5×5 window. Each output sample is a
/// weighted average of its neighbourhood where the weight is `spatial(distance)
/// × range(|intensity − centre|)` — so samples across a strong intensity step
/// (an edge) barely contribute and edges stay sharp while flat noise averages
/// out. Border samples shrink the window (out-of-range neighbours are skipped).
pub(super) fn plane(src: &[u8], w: usize, h: usize) -> Vec<u8> {
    const R: isize = 2; // 5×5
    let spatial_sigma = 2.0f32;
    let range_sigma = 20.0f32;
    // Precompute the 5×5 spatial weights and a 256-entry range LUT.
    let mut spatial = [[0f32; 5]; 5];
    for dy in -R..=R {
        for dx in -R..=R {
            let d2 = (dx * dx + dy * dy) as f32;
            spatial[(dy + R) as usize][(dx + R) as usize] =
                (-d2 / (2.0 * spatial_sigma * spatial_sigma)).exp();
        }
    }
    let mut range_lut = [0f32; 256];
    for (d, wt) in range_lut.iter_mut().enumerate() {
        *wt = (-((d * d) as f32) / (2.0 * range_sigma * range_sigma)).exp();
    }
    let mut out = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            let centre = src[y * w + x] as i32;
            let mut sum = 0f32;
            let mut wsum = 0f32;
            for dy in -R..=R {
                let yy = y as isize + dy;
                if yy < 0 || yy >= h as isize {
                    continue;
                }
                for dx in -R..=R {
                    let xx = x as isize + dx;
                    if xx < 0 || xx >= w as isize {
                        continue;
                    }
                    let s = src[yy as usize * w + xx as usize] as i32;
                    let wt = spatial[(dy + R) as usize][(dx + R) as usize]
                        * range_lut[(s - centre).unsigned_abs() as usize];
                    sum += wt * s as f32;
                    wsum += wt;
                }
            }
            out[y * w + x] = (sum / wsum).round().clamp(0.0, 255.0) as u8;
        }
    }
    out
}
