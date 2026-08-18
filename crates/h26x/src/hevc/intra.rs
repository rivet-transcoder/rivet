//! Intra sample prediction (H.265 clause 8.4.4.2): reference sample
//! availability and substitution, the reference smoothing filter (with
//! strong intra smoothing), and the planar, DC and angular predictors.

use super::frame::Plane16;

/// `intraPredAngle` for modes 2..=34 (Table 8-4), indexed by `mode - 2`.
const INTRA_PRED_ANGLE: [i32; 33] = [
    32, 26, 21, 17, 13, 9, 5, 2, 0, -2, -5, -9, -13, -17, -21, -26, -32, -26, -21, -17, -13, -9, -5, -2, 0, 2, 5, 9, 13, 17, 21, 26, 32,
];
/// `invAngle` for modes 11..=25 (Table 8-5), indexed by `mode - 11`.
const INV_ANGLE: [i32; 15] = [-4096, -1638, -910, -630, -482, -390, -315, -256, -315, -390, -482, -630, -910, -1638, -4096];

/// Availability of each reference sample of a block: `left[0..2n]` for
/// `p[-1][y]`, `top[0..2n]` for `p[x][-1]`, and the corner.
pub struct RefAvail {
    /// `p[-1][-1]`.
    pub corner: bool,
    /// `p[-1][y]`, y in 0..2n.
    pub left: [bool; 64],
    /// `p[x][-1]`, x in 0..2n.
    pub top: [bool; 64],
}

/// Predict an `n x n` block at `(x0, y0)` (plane coordinates) into the
/// plane. `mode` is the intra prediction mode (0 planar, 1 DC, 2..=34
/// angular), `c_idx` the component (filters are luma-only for 4:2:0),
/// `bit_depth` the sample depth, `strong` the SPS strong intra smoothing
/// flag, `avail` says which neighbouring samples may be used.
#[allow(clippy::too_many_arguments)]
pub fn predict(
    plane: &mut Plane16,
    x0: usize,
    y0: usize,
    n: usize,
    mode: u32,
    c_idx: usize,
    bit_depth: u32,
    strong: bool,
    avail: &RefAvail,
) {
    let stride = plane.stride;
    let base = plane.offset(x0 as isize, y0 as isize);
    // Gather p[-1][-1..2n-1] (index 0 = corner, 1..=2n = left samples y=0..)
    // and p[-1..2n-1][-1] (index 0 = corner, 1..=2n = top samples).
    // We use two arrays: left[y] for y in 0..2n, top[x] for x in 0..2n, corner.
    let n2 = 2 * n;
    let mut left = [0i32; 64];
    let mut top = [0i32; 64];
    let mut corner: i32 = 0;
    let mut any = false;
    if avail.corner {
        corner = plane.data[base - stride - 1] as i32;
        any = true;
    }
    for y in 0..n2 {
        if avail.left[y] {
            left[y] = plane.data[base + y * stride - 1] as i32;
            any = true;
        }
    }
    for x in 0..n2 {
        if avail.top[x] {
            top[x] = plane.data[base - stride + x] as i32;
            any = true;
        }
    }
    // Substitution (8.4.4.2.2). Order: p[-1][2n-1] .. p[-1][-1] (bottom-left
    // upwards), then p[0][-1] .. p[2n-1][-1].
    if !any {
        let v = 1i32 << (bit_depth - 1);
        left[..n2].fill(v);
        top[..n2].fill(v);
        corner = v;
    } else {
        // Search from bottom-left for the first available sample.
        let mut cur: Option<i32> = None;
        // Sequence: left[2n-1] down to left[0], corner, top[0..2n].
        // First pass: find the first available in that order.
        if !avail.left[n2 - 1] {
            let mut found = None;
            for y in (0..n2 - 1).rev() {
                if avail.left[y] {
                    found = Some(left[y]);
                    break;
                }
            }
            if found.is_none() && avail.corner {
                found = Some(corner);
            }
            if found.is_none() {
                for x in 0..n2 {
                    if avail.top[x] {
                        found = Some(top[x]);
                        break;
                    }
                }
            }
            left[n2 - 1] = found.unwrap();
        }
        cur = Some(left[n2 - 1]);
        for y in (0..n2 - 1).rev() {
            if !avail.left[y] {
                left[y] = cur.unwrap();
            } else {
                cur = Some(left[y]);
            }
        }
        if !avail.corner {
            corner = cur.unwrap();
        } else {
            cur = Some(corner);
        }
        for x in 0..n2 {
            if !avail.top[x] {
                top[x] = cur.unwrap();
            } else {
                cur = Some(top[x]);
            }
        }
    }

    // Filtering (8.4.4.2.3): luma (or 4:4:4) only, not for DC / 4x4.
    if c_idx == 0 && mode != 1 && n != 4 {
        let min_dist = (mode as i32 - 26).abs().min((mode as i32 - 10).abs());
        let thres = match n {
            8 => 7,
            16 => 1,
            32 => 0,
            _ => 10, // never filtered
        };
        if min_dist > thres {
            let mut fl = [0i32; 64];
            let mut ft = [0i32; 64];
            let bi = strong
                && n == 32
                && (corner + top[n2 - 1] - 2 * top[n - 1]).abs() < (1 << (bit_depth - 5))
                && (corner + left[n2 - 1] - 2 * left[n - 1]).abs() < (1 << (bit_depth - 5));
            let fc;
            if bi {
                fc = corner;
                for i in 0..n2 - 1 {
                    // pF[-1][i] = ((63-i)*p[-1][-1] + (i+1)*p[-1][63] + 32) >> 6, i=0..62
                    fl[i] = ((63 - i as i32) * corner + (i as i32 + 1) * left[63] + 32) >> 6;
                    ft[i] = ((63 - i as i32) * corner + (i as i32 + 1) * top[63] + 32) >> 6;
                }
                fl[63] = left[63];
                ft[63] = top[63];
            } else {
                fc = (left[0] + 2 * corner + top[0] + 2) >> 2;
                fl[0] = (corner + 2 * left[0] + left[1] + 2) >> 2;
                for y in 1..n2 - 1 {
                    fl[y] = (left[y - 1] + 2 * left[y] + left[y + 1] + 2) >> 2;
                }
                fl[n2 - 1] = left[n2 - 1];
                ft[0] = (corner + 2 * top[0] + top[1] + 2) >> 2;
                for x in 1..n2 - 1 {
                    ft[x] = (top[x - 1] + 2 * top[x] + top[x + 1] + 2) >> 2;
                }
                ft[n2 - 1] = top[n2 - 1];
            }
            left = fl;
            top = ft;
            corner = fc;
        }
    }

    let max = (1i32 << bit_depth) - 1;
    let log2n = n.trailing_zeros();
    match mode {
        0 => {
            // Planar.
            for y in 0..n {
                let row = &mut plane.data[base + y * stride..base + y * stride + n];
                let (ly, ln, tn) = (left[y], left[n], top[n]);
                let ry = n as i32 - 1 - y as i32;
                for (x, d) in row.iter_mut().enumerate() {
                    let v = ((n as i32 - 1 - x as i32) * ly + (x as i32 + 1) * tn + ry * top[x] + (y as i32 + 1) * ln + n as i32) >> (log2n + 1);
                    *d = v as u16;
                }
            }
        }
        1 => {
            // DC.
            let mut sum = n as i32;
            for i in 0..n {
                sum += top[i] + left[i];
            }
            let dc = sum >> (log2n + 1);
            for y in 0..n {
                plane.data[base + y * stride..base + y * stride + n].fill(dc as u16);
            }
            if c_idx == 0 && n < 32 {
                plane.data[base] = ((left[0] + 2 * dc + top[0] + 2) >> 2) as u16;
                for x in 1..n {
                    plane.data[base + x] = ((top[x] + 3 * dc + 2) >> 2) as u16;
                }
                for y in 1..n {
                    plane.data[base + y * stride] = ((left[y] + 3 * dc + 2) >> 2) as u16;
                }
            }
        }
        _ => {
            let angle = INTRA_PRED_ANGLE[(mode - 2) as usize];
            // ref[] with an offset so negative indices work: ref_buf[i + n]
            let mut ref_buf = [0i32; 3 * 64 + 1];
            let off = n as i32; // ref[k] at ref_buf[(k + off) as usize]
            if mode >= 18 {
                // ref[x] = p[-1+x][-1] for x = 0..n  (ref[0] = corner)
                ref_buf[off as usize] = corner;
                for x in 1..=n {
                    ref_buf[off as usize + x] = top[x - 1];
                }
                if angle < 0 {
                    let last = (n as i32 * angle) >> 5;
                    if last < -1 {
                        let inv = INV_ANGLE[(mode - 11) as usize];
                        for x in last..=-1 {
                            // ref[x] = p[-1][-1 + ((x*invAngle+128)>>8)]
                            let idx = -1 + ((x * inv + 128) >> 8);
                            let v = if idx < 0 { corner } else { left[idx as usize] };
                            ref_buf[(x + off) as usize] = v;
                        }
                    }
                } else {
                    for x in n + 1..=2 * n {
                        ref_buf[off as usize + x] = top[x - 1];
                    }
                }
                for y in 0..n {
                    let i_idx = ((y as i32 + 1) * angle) >> 5;
                    let i_fact = ((y as i32 + 1) * angle) & 31;
                    let row = &mut plane.data[base + y * stride..base + y * stride + n];
                    let start = (i_idx + 1 + off) as usize;
                    // Contiguous reference reads along the row: the compiler
                    // vectorises both loops.
                    if i_fact != 0 {
                        let ra = &ref_buf[start..start + n];
                        let rb = &ref_buf[start + 1..start + 1 + n];
                        for ((d, &a), &b) in row.iter_mut().zip(ra).zip(rb) {
                            *d = (((32 - i_fact) * a + i_fact * b + 16) >> 5) as u16;
                        }
                    } else {
                        for (d, &a) in row.iter_mut().zip(&ref_buf[start..start + n]) {
                            *d = a as u16;
                        }
                    }
                }
                if mode == 26 && c_idx == 0 && n < 32 {
                    for y in 0..n {
                        let v = (top[0] + ((left[y] - corner) >> 1)).clamp(0, max);
                        plane.data[base + y * stride] = v as u16;
                    }
                }
            } else {
                // ref[x] = p[-1][-1+x] for x = 0..n
                ref_buf[off as usize] = corner;
                for x in 1..=n {
                    ref_buf[off as usize + x] = left[x - 1];
                }
                if angle < 0 {
                    let last = (n as i32 * angle) >> 5;
                    if last < -1 {
                        let inv = INV_ANGLE[(mode - 11) as usize];
                        for x in last..=-1 {
                            let idx = -1 + ((x * inv + 128) >> 8);
                            let v = if idx < 0 { corner } else { top[idx as usize] };
                            ref_buf[(x + off) as usize] = v;
                        }
                    }
                } else {
                    for x in n + 1..=2 * n {
                        ref_buf[off as usize + x] = left[x - 1];
                    }
                }
                // Horizontal-ish modes run along columns: predict the
                // transposed block row-wise (vectorisable), then transpose
                // into the plane.
                // (Transform blocks are at most 32x32; the temp is not
                // zeroed — every used entry is written before it is read.)
                debug_assert!(n <= 32);
                let mut tmp: [std::mem::MaybeUninit<u16>; 32 * 32] = [std::mem::MaybeUninit::uninit(); 32 * 32];
                for x in 0..n {
                    let i_idx = ((x as i32 + 1) * angle) >> 5;
                    let i_fact = ((x as i32 + 1) * angle) & 31;
                    let start = (i_idx + 1 + off) as usize;
                    let col = &mut tmp[x * n..x * n + n];
                    if i_fact != 0 {
                        let ra = &ref_buf[start..start + n];
                        let rb = &ref_buf[start + 1..start + 1 + n];
                        for ((d, &a), &b) in col.iter_mut().zip(ra).zip(rb) {
                            d.write((((32 - i_fact) * a + i_fact * b + 16) >> 5) as u16);
                        }
                    } else {
                        for (d, &a) in col.iter_mut().zip(&ref_buf[start..start + n]) {
                            d.write(a as u16);
                        }
                    }
                }
                for y in 0..n {
                    let row = &mut plane.data[base + y * stride..base + y * stride + n];
                    for x in 0..n {
                        // SAFETY: tmp[x * n + y] was written above (x, y < n).
                        row[x] = unsafe { tmp[x * n + y].assume_init() };
                    }
                }
                if mode == 10 && c_idx == 0 && n < 32 {
                    for x in 0..n {
                        let v = (left[0] + ((top[x] - corner) >> 1)).clamp(0, max);
                        plane.data[base + x] = v as u16;
                    }
                }
            }
        }
    }
}
