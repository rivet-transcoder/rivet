//! Sample adaptive offset (H.265 8.7.3), applied per CTB per component to
//! the deblocked picture; a copy of the deblocked planes serves as input so
//! neighbouring CTBs already filtered do not feed back. The interior of a
//! CTB goes through the [`crate::dsp::hevc`] kernels; the one-sample ring
//! next to an unusable neighbour, and CTBs holding filter-exempt (PCM /
//! lossless) blocks, take the per-sample path.

use crate::dsp::hevc::HevcDsp;

use super::frame::{Frame, Plane16};
use super::pic::{PicInfo, SaoParams};
use super::pps::Pps;
use super::sps::Sps;

/// Apply SAO to the whole picture in place.
pub fn sao_picture(dsp: &HevcDsp, frame: &mut Frame, info: &PicInfo, sps: &Sps, pps: &Pps) {
    if !sps.sao_enabled {
        return;
    }
    if !info.sao.iter().any(|s| s.iter().any(|c| c.type_idx != 0)) {
        return;
    }
    let src_y = frame.y.clone();
    let src_cb = frame.cb.clone();
    let src_cr = frame.cr.clone();
    let ctb = 1usize << sps.log2_ctb_size;
    let (pw, ph) = (frame.width, frame.height);
    for ry in 0..info.hc {
        for rx in 0..info.wc {
            let addr = ry * info.wc + rx;
            if info.ctb_slice[addr] == u16::MAX {
                continue;
            }
            let params = &info.sao[addr];
            if params.iter().all(|p| p.type_idx == 0) {
                continue;
            }
            // Which of the eight neighbouring CTBs may be read across.
            let nb = Neighbours::of(info, pps, rx, ry);
            // Any exempt block in this CTB → per-sample path throughout.
            let exempt_any = {
                let (x0, y0) = (rx * ctb, ry * ctb);
                let (w, h) = (ctb.min(pw - x0), ctb.min(ph - y0));
                let mut any = false;
                'scan: for by in (y0 >> 2)..((y0 + h) >> 2) {
                    for bx in (x0 >> 2)..((x0 + w) >> 2) {
                        if info.filter_exempt[by * info.w4 + bx] & 1 != 0 {
                            any = true;
                            break 'scan;
                        }
                    }
                }
                any
            };
            for c in 0..3usize {
                let p = &params[c];
                if p.type_idx == 0 || (c > 0 && sps.chroma_format_idc == 0) {
                    continue;
                }
                let scale = if c == 0 { 1 } else { 2 };
                let bd = if c == 0 { sps.bit_depth_luma } else { sps.bit_depth_chroma };
                let (src, dst): (&Plane16, &mut Plane16) = match c {
                    0 => (&src_y, &mut frame.y),
                    1 => (&src_cb, &mut frame.cb),
                    _ => (&src_cr, &mut frame.cr),
                };
                let x0 = rx * ctb / scale;
                let y0 = ry * ctb / scale;
                let w = (ctb / scale).min(pw / scale - x0);
                let h = (ctb / scale).min(ph / scale - y0);
                sao_ctb(dsp, src, dst, info, x0, y0, w, h, scale, bd, p, &nb, exempt_any);
            }
        }
    }
}

/// Whether each neighbouring CTB (l, r, t, b, tl, tr, bl, br) may be read by
/// this CTB's SAO (8.7.3.2: inside the picture, same slice or the flags of
/// the later slice allow it, same tile or loop_filter_across_tiles).
struct Neighbours {
    l: bool,
    r: bool,
    t: bool,
    b: bool,
    tl: bool,
    tr: bool,
    bl: bool,
    br: bool,
}

impl Neighbours {
    fn of(info: &PicInfo, pps: &Pps, rx: usize, ry: usize) -> Self {
        let addr = ry * info.wc + rx;
        let cur_slice_addr = info.ctb_slice_addr[addr];
        let cur_slice = &info.slices[info.ctb_slice[addr] as usize];
        let cur_tile = info.ctb_tile[addr];
        let cur_ts = info.ctb_rs_to_ts[addr];
        let usable = |dx: i32, dy: i32| -> bool {
            let nx = rx as i32 + dx;
            let ny = ry as i32 + dy;
            if nx < 0 || ny < 0 || nx as usize >= info.wc || ny as usize >= info.hc {
                return false;
            }
            let n = ny as usize * info.wc + nx as usize;
            if info.ctb_slice[n] == u16::MAX {
                return false;
            }
            if info.ctb_slice_addr[n] != cur_slice_addr {
                let n_slice = &info.slices[info.ctb_slice[n] as usize];
                let n_ts = info.ctb_rs_to_ts[n];
                // Both CTBs are whole units of z-scan order, so the MinTbAddrZs
                // comparison is a tile-scan comparison.
                if n_ts < cur_ts && !cur_slice.loop_filter_across_slices {
                    return false;
                }
                if cur_ts < n_ts && !n_slice.loop_filter_across_slices {
                    return false;
                }
            }
            if !pps.loop_filter_across_tiles && info.ctb_tile[n] != cur_tile {
                return false;
            }
            true
        };
        Neighbours {
            l: usable(-1, 0),
            r: usable(1, 0),
            t: usable(0, -1),
            b: usable(0, 1),
            tl: usable(-1, -1),
            tr: usable(1, -1),
            bl: usable(-1, 1),
            br: usable(1, 1),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn sao_ctb(
    dsp: &HevcDsp,
    src: &Plane16,
    dst: &mut Plane16,
    info: &PicInfo,
    x0: usize,
    y0: usize,
    w: usize,
    h: usize,
    scale: usize,
    bit_depth: u32,
    p: &SaoParams,
    nb: &Neighbours,
    exempt_any: bool,
) {
    let max = (1i32 << bit_depth) - 1;
    let stride = src.stride;
    let exempt = |x: usize, y: usize| -> bool { exempt_any && info.filter_exempt[info.idx4(x * scale, y * scale)] & 1 != 0 };
    match p.type_idx {
        1 => {
            let shift = bit_depth as i32 - 5;
            let mut table = [0i16; 32];
            for k in 0..4 {
                table[(k + p.band_or_class as usize) & 31] = p.offsets[k];
            }
            let off = src.offset(x0 as isize, y0 as isize);
            if !exempt_any {
                (dsp.sao_band)(&mut dst.data[off..], stride, &src.data[off..], stride, w, h, &table, shift, max);
            } else {
                for y in 0..h {
                    for x in 0..w {
                        if exempt(x0 + x, y0 + y) {
                            continue;
                        }
                        let i = off + y * stride + x;
                        let v = src.data[i] as i32;
                        dst.data[i] = (v + table[(v >> shift) as usize] as i32).clamp(0, max) as u16;
                    }
                }
            }
        }
        2 => {
            let (hp, vp): ([i32; 2], [i32; 2]) = match p.band_or_class {
                0 => ([-1, 1], [0, 0]),
                1 => ([0, 0], [-1, 1]),
                2 => ([-1, 1], [-1, 1]),
                _ => ([1, -1], [-1, 1]),
            };
            // SaoOffsetVal indexed by raw edgeIdx 0..=4 (2 = no change).
            let off_tab: [i16; 5] = [p.offsets[0], p.offsets[1], 0, p.offsets[2], p.offsets[3]];
            // The interior where both neighbours are certainly usable.
            let (mut xs, mut xe, mut ys, mut ye) = (x0, x0 + w, y0, y0 + h);
            let uses_x = hp[0] != 0;
            let uses_y = vp[0] != 0;
            let diag = uses_x && uses_y;
            if (uses_x && !nb.l) || (diag && (!nb.tl || !nb.bl)) {
                xs += 1;
            }
            if (uses_x && !nb.r) || (diag && (!nb.tr || !nb.br)) {
                xe -= 1;
            }
            if (uses_y && !nb.t) || (diag && (!nb.tl || !nb.tr)) {
                ys += 1;
            }
            if (uses_y && !nb.b) || (diag && (!nb.bl || !nb.br)) {
                ye -= 1;
            }
            let na = vp[0] as isize * stride as isize + hp[0] as isize;
            let nbb = vp[1] as isize * stride as isize + hp[1] as isize;
            let interior_ok = !exempt_any && xs < xe && ys < ye;
            if interior_ok {
                let off = src.offset(xs as isize, ys as isize);
                (dsp.sao_edge)(&mut dst.data, &src.data, off, stride, xe - xs, ye - ys, na, nbb, &off_tab, max);
            }
            // The ring (or everything, with exempt blocks): per sample, with
            // the exact neighbour rules.
            let (pic_w, pic_h) = (src.width, src.height);
            let usable = |x: usize, y: usize, xn: i32, yn: i32| -> bool {
                if xn < 0 || yn < 0 || xn as usize >= pic_w || yn as usize >= pic_h {
                    return false;
                }
                let (xn, yn) = (xn as usize, yn as usize);
                let cx = (xn * scale) >> info.log2_ctb;
                let cy = (yn * scale) >> info.log2_ctb;
                let (cx0, cy0) = ((x * scale) >> info.log2_ctb, (y * scale) >> info.log2_ctb);
                match (cx as i32 - cx0 as i32, cy as i32 - cy0 as i32) {
                    (0, 0) => true,
                    (-1, 0) => nb.l,
                    (1, 0) => nb.r,
                    (0, -1) => nb.t,
                    (0, 1) => nb.b,
                    (-1, -1) => nb.tl,
                    (1, -1) => nb.tr,
                    (-1, 1) => nb.bl,
                    _ => nb.br,
                }
            };
            for y in y0..y0 + h {
                for x in x0..x0 + w {
                    let interior = interior_ok && x >= xs && x < xe && y >= ys && y < ye;
                    if interior || exempt(x, y) {
                        continue;
                    }
                    let (xa, ya) = (x as i32 + hp[0], y as i32 + vp[0]);
                    let (xb, yb) = (x as i32 + hp[1], y as i32 + vp[1]);
                    if !usable(x, y, xa, ya) || !usable(x, y, xb, yb) {
                        continue;
                    }
                    let i = src.offset(x as isize, y as isize);
                    let v = src.data[i] as i32;
                    let a = src.data[(i as isize + na) as usize] as i32;
                    let b = src.data[(i as isize + nbb) as usize] as i32;
                    let e = (2 + (v - a).signum() + (v - b).signum()) as usize;
                    dst.data[i] = (v + off_tab[e] as i32).clamp(0, max) as u16;
                }
            }
        }
        _ => {}
    }
}
