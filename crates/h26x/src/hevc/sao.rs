//! Sample adaptive offset (H.265 8.7.3), applied per CTB per component to
//! the deblocked picture; a copy of the deblocked planes serves as input
//! so neighbouring CTBs already filtered do not feed back.

use super::frame::{Frame, Plane16};
use super::pic::PicInfo;
use super::pps::Pps;
use super::sps::Sps;

/// Apply SAO to the whole picture in place.
pub fn sao_picture(frame: &mut Frame, info: &PicInfo, sps: &Sps, pps: &Pps) {
    if !sps.sao_enabled {
        return;
    }
    // Any CTB with SAO on?
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
            let params = &info.sao[addr];
            if info.ctb_slice[addr] == u16::MAX {
                continue;
            }
            for c in 0..3usize {
                let p = &params[c];
                if p.type_idx == 0 {
                    continue;
                }
                if c > 0 && sps.chroma_format_idc == 0 {
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
                sao_ctb(src, dst, info, pps, x0, y0, w, h, scale, bd, p, pw / scale, ph / scale, addr);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn sao_ctb(
    src: &Plane16,
    dst: &mut Plane16,
    info: &PicInfo,
    pps: &Pps,
    x0: usize,
    y0: usize,
    w: usize,
    h: usize,
    scale: usize,
    bit_depth: u32,
    p: &super::pic::SaoParams,
    pic_w: usize,
    pic_h: usize,
    ctb_addr: usize,
) {
    let max = (1i32 << bit_depth) - 1;
    let stride = src.stride;
    let cur_slice = &info.slices[info.ctb_slice[ctb_addr] as usize];
    let cur_slice_addr = info.ctb_slice_addr[ctb_addr];
    let cur_tile = info.ctb_tile[ctb_addr];
    // Exemption per 4x4 luma block: pcm + pcm_loop_filter_disabled, or bypass.
    let exempt = |x: usize, y: usize| -> bool { info.filter_exempt[info.idx4(x * scale, y * scale)] & 1 != 0 };
    match p.type_idx {
        1 => {
            // Band offset.
            let shift = bit_depth - 5;
            let mut band_table = [0i32; 32];
            for k in 0..4 {
                band_table[(k + p.band_or_class as usize) & 31] = k as i32 + 1;
            }
            for y in y0..y0 + h {
                for x in x0..x0 + w {
                    if exempt(x, y) {
                        continue;
                    }
                    let off = src.offset(x as isize, y as isize);
                    let v = src.data[off] as i32;
                    let idx = band_table[(v >> shift) as usize];
                    if idx > 0 {
                        dst.data[off] = (v + p.offsets[idx as usize - 1] as i32).clamp(0, max) as u16;
                    }
                }
            }
        }
        2 => {
            // Edge offset.
            let (hp, vp): ([i32; 2], [i32; 2]) = match p.band_or_class {
                0 => ([-1, 1], [0, 0]),
                1 => ([0, 0], [-1, 1]),
                2 => ([-1, 1], [-1, 1]),
                _ => ([1, -1], [-1, 1]),
            };
            // Whether the neighbour at (xn, yn) (component coords) may be used
            // by the sample at (x, y).
            let usable = |x: usize, y: usize, xn: i32, yn: i32| -> bool {
                if xn < 0 || yn < 0 || xn as usize >= pic_w || yn as usize >= pic_h {
                    return false;
                }
                let (xl, yl) = (x * scale, y * scale);
                let (xnl, ynl) = (xn as usize * scale, yn as usize * scale);
                let cn = info.ctb_of(xnl, ynl);
                let n_slice_addr = info.ctb_slice_addr[cn];
                if n_slice_addr == u32::MAX {
                    return false;
                }
                if n_slice_addr != cur_slice_addr {
                    // Different slice: the earlier one's flag decides... no —
                    // 8.7.3: current-first vs neighbour-first, each with the
                    // flag of the *later* sample's slice.
                    let zc = info.min_tb_addr_zs[info.idx4(xl, yl)];
                    let zn = info.min_tb_addr_zs[info.idx4(xnl, ynl)];
                    let n_slice = &info.slices[info.ctb_slice[cn] as usize];
                    if zn < zc && !cur_slice.loop_filter_across_slices {
                        return false;
                    }
                    if zc < zn && !n_slice.loop_filter_across_slices {
                        return false;
                    }
                }
                if !pps.loop_filter_across_tiles && info.ctb_tile[cn] != cur_tile {
                    return false;
                }
                true
            };
            for y in y0..y0 + h {
                for x in x0..x0 + w {
                    if exempt(x, y) {
                        continue;
                    }
                    let xa = x as i32 + hp[0];
                    let ya = y as i32 + vp[0];
                    let xb = x as i32 + hp[1];
                    let yb = y as i32 + vp[1];
                    if !usable(x, y, xa, ya) || !usable(x, y, xb, yb) {
                        continue;
                    }
                    let off = src.offset(x as isize, y as isize);
                    let v = src.data[off] as i32;
                    let a = src.data[(off as isize + (ya - y as i32) as isize * stride as isize + (xa - x as i32) as isize) as usize] as i32;
                    let b = src.data[(off as isize + (yb - y as i32) as isize * stride as isize + (xb - x as i32) as isize) as usize] as i32;
                    let e = 2 + (v - a).signum() + (v - b).signum();
                    let idx = match e {
                        0 => 1,
                        1 => 2,
                        2 => 0,
                        3 => 3,
                        _ => 4,
                    };
                    if idx > 0 {
                        dst.data[off] = (v + p.offsets[idx - 1] as i32).clamp(0, max) as u16;
                    }
                }
            }
        }
        _ => {}
    }
}
