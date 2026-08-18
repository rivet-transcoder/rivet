//! Sample adaptive offset (H.265 8.7.3), applied per CTB per component to
//! the deblocked picture; a copy of the deblocked samples serves as input
//! so neighbouring CTBs already filtered do not feed back — a band of one
//! CTB row plus a line above and below, kept small enough to stay in cache.
//! The interior of a CTB goes through the [`crate::dsp::hevc`] kernels; the
//! one-sample ring next to an unusable neighbour, and CTBs holding
//! filter-exempt (PCM / lossless) blocks, take the per-sample path.

use crate::dsp::hevc::HevcDsp;

use super::frame::{Frame, Plane16, Sample};
use super::pic::{PicInfo, SaoParams};
use super::pps::Pps;
use super::sps::Sps;

/// Apply SAO to the whole picture in place.
pub fn sao_picture<S: Sample>(dsp: &HevcDsp<S>, frame: &mut Frame<S>, info: &PicInfo, sps: &Sps, pps: &Pps) {
    if !sps.sao_enabled {
        return;
    }
    if !info.sao.iter().any(|s| s.iter().any(|c| c.type_idx != 0)) {
        return;
    }
    let src = frame.clone();
    let band = SaoBand::<S>::new();
    for ry in 0..info.hc {
        sao_ctb_row(dsp, frame, &src, &band, info, sps, pps, ry);
    }
}

/// The deblocked source samples for one CTB row's SAO: a copy of the row
/// plus a line above and below (see [`sao_ctb_row`]), and which picture
/// rows its first lines are.
pub struct SaoBand<S: Sample> {
    /// Luma picture row held in the band's luma row 0.
    pub luma_row0: usize,
    /// Chroma picture row held in the band's chroma row 0.
    pub chroma_row0: usize,
    /// The last line of the previous CTB row before its SAO (the picture
    /// holds the filtered one by now), per plane, full stride.
    last: [Vec<S>; 3],
}

impl<S: Sample> SaoBand<S> {
    /// Nothing saved yet.
    pub fn new() -> Self {
        SaoBand { luma_row0: 0, chroma_row0: 0, last: [Vec::new(), Vec::new(), Vec::new()] }
    }

    /// Copy the deblocked lines CTB row `ry` needs from `frame` into
    /// `band` (a frame at least `ctb + 2` luma rows tall): the row itself
    /// plus the line above (as it was before that row's SAO) and below.
    /// Rows must be filled in order from 0.
    pub fn fill(&mut self, frame: &Frame<S>, band: &mut Frame<S>, ctb: usize, ry: usize) {
        let mono = frame.chroma == crate::picture::ChromaFormat::Monochrome;
        let y0 = ry * ctb;
        let ya = y0.saturating_sub(1);
        let yb = (y0 + ctb + 1).min(frame.height + 1);
        self.luma_row0 = ya;
        copy_lines(&frame.y, &mut band.y, ya, yb);
        let (cy0, ca, cb) = (y0 / 2, (y0 / 2).saturating_sub(1), (y0 / 2 + ctb / 2 + 1).min(frame.height / 2 + 1));
        if !mono {
            self.chroma_row0 = ca;
            copy_lines(&frame.cb, &mut band.cb, ca, cb);
            copy_lines(&frame.cr, &mut band.cr, ca, cb);
        }
        // The line above came from the picture already filtered: put back
        // the saved one; then save this row's last line for the next row.
        let planes: [(&mut Plane16<S>, usize, usize, usize); 3] = [(&mut band.y, ya, y0, (y0 + ctb - 1).min(frame.height - 1)), (&mut band.cb, ca, cy0, (cy0 + ctb / 2 - 1).min(frame.height / 2 - 1)), (&mut band.cr, ca, cy0, (cy0 + ctb / 2 - 1).min(frame.height / 2 - 1))];
        for (c, (plane, row0, first, last)) in planes.into_iter().enumerate() {
            if c > 0 && mono {
                break;
            }
            let stride = plane.stride;
            let pad = plane.pad;
            let line = |y: usize| (pad + y - row0) * stride;
            if ry > 0 && first > 0 {
                let d = line(first - 1);
                plane.data[d..d + stride].copy_from_slice(&self.last[c]);
            }
            let sidx = line(last);
            self.last[c].clear();
            self.last[c].extend_from_slice(&plane.data[sidx..sidx + stride]);
        }
    }
}

impl<S: Sample> Default for SaoBand<S> {
    fn default() -> Self {
        Self::new()
    }
}

/// Copy plane rows `y0..y1` of `from` (full stride, borders included) into
/// rows `0..` of `to`.
fn copy_lines<S: Sample>(from: &Plane16<S>, to: &mut Plane16<S>, y0: usize, y1: usize) {
    if y0 >= y1 {
        return;
    }
    let n = (y1 - y0) * from.stride;
    let s = from.offset(0, y0 as isize) - from.pad;
    let d = to.offset(0, 0) - to.pad;
    to.data[d..d + n].copy_from_slice(&from.data[s..s + n]);
}

/// Apply SAO to CTB row `ry` of `frame`, reading the deblocked samples from
/// `src` (filled by [`SaoBand::fill`] for this row: rows `ry - 1 ..= ry + 1`
/// hold final deblocked values — at least the last line of the row above
/// and the first line of the row below).
#[allow(clippy::too_many_arguments)]
pub fn sao_ctb_row<S: Sample>(dsp: &HevcDsp<S>, frame: &mut Frame<S>, src: &Frame<S>, band: &SaoBand<S>, info: &PicInfo, sps: &Sps, pps: &Pps, ry: usize) {
    if !sps.sao_enabled {
        return;
    }
    let (src_y, src_cb, src_cr) = (&src.y, &src.cb, &src.cr);
    let ctb = 1usize << sps.log2_ctb_size;
    let (pw, ph) = (frame.width, frame.height);
    {
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
                let (src, dst, src_row0): (&Plane16<S>, &mut Plane16<S>, usize) = match c {
                    0 => (src_y, &mut frame.y, band.luma_row0),
                    1 => (src_cb, &mut frame.cb, band.chroma_row0),
                    _ => (src_cr, &mut frame.cr, band.chroma_row0),
                };
                let x0 = rx * ctb / scale;
                let y0 = ry * ctb / scale;
                let w = (ctb / scale).min(pw / scale - x0);
                let h = (ctb / scale).min(ph / scale - y0);
                sao_ctb(dsp, src, src_row0, dst, info, x0, y0, w, h, scale, bd, p, &nb, exempt_any);
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

/// One CTB, one component. `src` holds picture rows from `src_row0` on in
/// its rows from 0; `dst` is the picture plane. Both have the same stride,
/// so `dst` is addressed through a slice shifted by `src_row0` rows and the
/// kernels see one index for a sample in both.
#[allow(clippy::too_many_arguments)]
fn sao_ctb<S: Sample>(
    dsp: &HevcDsp<S>,
    src: &Plane16<S>,
    src_row0: usize,
    dst: &mut Plane16<S>,
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
    debug_assert_eq!(stride, dst.stride);
    debug_assert_eq!(src.pad, dst.pad);
    let (pic_w, pic_h) = (dst.width, dst.height);
    // Picture (x, y) in the band's data, and the same index in `dst`.
    let at = |x: usize, y: usize| src.offset(x as isize, y as isize - src_row0 as isize);
    let dst = &mut dst.data[src_row0 * stride..];
    let exempt = |x: usize, y: usize| -> bool { exempt_any && info.filter_exempt[info.idx4(x * scale, y * scale)] & 1 != 0 };
    match p.type_idx {
        1 => {
            let shift = bit_depth as i32 - 5;
            let mut table = [0i16; 32];
            for k in 0..4 {
                table[(k + p.band_or_class as usize) & 31] = p.offsets[k];
            }
            let off = at(x0, y0);
            if !exempt_any {
                (dsp.sao_band)(&mut dst[off..], stride, &src.data[off..], stride, w, h, &table, shift, max);
            } else {
                for y in 0..h {
                    for x in 0..w {
                        if exempt(x0 + x, y0 + y) {
                            continue;
                        }
                        let i = off + y * stride + x;
                        let v = src.data[i].to_i32();
                        dst[i] = S::from_i32((v + table[(v >> shift) as usize] as i32).clamp(0, max));
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
                let off = at(xs, ys);
                (dsp.sao_edge)(dst, &src.data, off, stride, xe - xs, ye - ys, na, nbb, &off_tab, max);
            }
            // The ring (or everything, with exempt blocks): per sample, with
            // the exact neighbour rules.
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
            let mut sample = |x: usize, y: usize| {
                if exempt(x, y) {
                    return;
                }
                let (xa, ya) = (x as i32 + hp[0], y as i32 + vp[0]);
                let (xb, yb) = (x as i32 + hp[1], y as i32 + vp[1]);
                if !usable(x, y, xa, ya) || !usable(x, y, xb, yb) {
                    return;
                }
                let i = at(x, y);
                let v = src.data[i].to_i32();
                let a = src.data[(i as isize + na) as usize].to_i32();
                let b = src.data[(i as isize + nbb) as usize].to_i32();
                let e = (2 + (v - a).signum() + (v - b).signum()) as usize;
                dst[i] = S::from_i32((v + off_tab[e] as i32).clamp(0, max));
            };
            if interior_ok {
                // Only the ring around the interior: the rows above and
                // below it in full, the columns beside it in between.
                for y in y0..ys {
                    for x in x0..x0 + w {
                        sample(x, y);
                    }
                }
                for y in ys..ye {
                    for x in x0..xs {
                        sample(x, y);
                    }
                    for x in xe..x0 + w {
                        sample(x, y);
                    }
                }
                for y in ye..y0 + h {
                    for x in x0..x0 + w {
                        sample(x, y);
                    }
                }
            } else {
                for y in y0..y0 + h {
                    for x in x0..x0 + w {
                        sample(x, y);
                    }
                }
            }
        }
        _ => {}
    }
}
