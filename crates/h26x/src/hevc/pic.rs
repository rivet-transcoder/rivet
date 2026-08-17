//! Per-picture decoding state: block-level side data at 4x4 granularity,
//! CTB-level slice/tile membership, the z-scan order tables (6.5.2), SAO
//! parameters and the per-slice loop-filter parameters.

use super::pps::Pps;
use super::sps::Sps;

/// SAO parameters of one CTB for one component.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SaoParams {
    /// `SaoTypeIdx`: 0 off, 1 band, 2 edge.
    pub type_idx: u8,
    /// `sao_band_position` (band) or `SaoEoClass` (edge).
    pub band_or_class: u8,
    /// `SaoOffsetVal[1..=4]` (already shifted by `log2OffsetScale`).
    pub offsets: [i16; 4],
}

/// Per-slice loop filter facts, indexed by the slice number stored per CTB.
#[derive(Debug, Clone, Copy, Default)]
pub struct SliceFilterParams {
    /// `slice_deblocking_filter_disabled_flag`.
    pub deblocking_disabled: bool,
    /// `slice_beta_offset_div2 * 2`.
    pub beta_offset: i32,
    /// `slice_tc_offset_div2 * 2`.
    pub tc_offset: i32,
    /// `slice_loop_filter_across_slices_enabled_flag`.
    pub loop_filter_across_slices: bool,
    /// `SliceAddrRs` — the address of the slice's first (independent) segment.
    pub slice_addr: u32,
    /// `pps_cb_qp_offset`, `pps_cr_qp_offset` (chroma deblocking uses the PPS
    /// offsets, not the slice ones).
    pub cb_qp_offset: i32,
    /// See `cb_qp_offset`.
    pub cr_qp_offset: i32,
}

/// The per-picture arrays.
pub struct PicInfo {
    /// Width / height in 4x4 units.
    pub w4: usize,
    /// See `w4`.
    pub h4: usize,
    /// Width / height in CTBs.
    pub wc: usize,
    /// See `wc`.
    pub hc: usize,
    /// `CtbLog2SizeY`.
    pub log2_ctb: u32,
    /// Per 4x4: 1 = intra CU, 0 = inter, 2 = not decoded yet.
    pub pred_mode: Vec<u8>,
    /// Per 4x4: `cu_skip_flag`.
    pub skip: Vec<u8>,
    /// Per 4x4: `CtDepth`.
    pub ct_depth: Vec<u8>,
    /// Per 4x4: luma intra prediction mode (1 = DC where not intra).
    pub intra_mode: Vec<u8>,
    /// Per 4x4: `QpY` of the CU.
    pub qp_y: Vec<i8>,
    /// Per 4x4: bit 0 = samples are exempt from the loop filters (pcm with
    /// `pcm_loop_filter_disabled_flag`, or `cu_transquant_bypass`); bit 1 =
    /// `cu_transquant_bypass_flag` (for the deblocking nDp/nDq rule too).
    pub filter_exempt: Vec<u8>,
    /// Per 4x4: the luma transform block covering it has nonzero coefficients.
    pub cbf_luma: Vec<u8>,
    /// Per 4x4: bit 0 = a transform block edge lies on this block's left side,
    /// bit 1 = a prediction block edge does; bits 2/3 the same for the top.
    pub edges: Vec<u8>,
    /// Per CTB: index into `slices` (u16::MAX = not decoded).
    pub ctb_slice: Vec<u16>,
    /// Per CTB: `SliceAddrRs` (u32::MAX = not decoded).
    pub ctb_slice_addr: Vec<u32>,
    /// Per CTB: tile id.
    pub ctb_tile: Vec<u16>,
    /// `MinTbAddrZs` per 4x4.
    pub min_tb_addr_zs: Vec<u32>,
    /// `CtbAddrRsToTs`.
    pub ctb_rs_to_ts: Vec<u32>,
    /// `CtbAddrTsToRs`.
    pub ctb_ts_to_rs: Vec<u32>,
    /// `TileId` per CTB in TS order.
    pub tile_id_ts: Vec<u16>,
    /// SAO parameters per CTB per component.
    pub sao: Vec<[SaoParams; 3]>,
    /// The slices decoded so far.
    pub slices: Vec<SliceFilterParams>,
}

impl PicInfo {
    /// Build the tables for an SPS/PPS pair.
    pub fn new(sps: &Sps, pps: &Pps) -> Self {
        let w4 = (sps.width as usize).div_ceil(4);
        let h4 = (sps.height as usize).div_ceil(4);
        let wc = sps.pic_width_in_ctbs() as usize;
        let hc = sps.pic_height_in_ctbs() as usize;
        let n4 = w4 * h4;
        let nc = wc * hc;
        let log2_ctb = sps.log2_ctb_size;

        // 6.5.1: CtbAddrRsToTs / TsToRs / TileId.
        let cols = pps.col_bd.len() - 1;
        let rows = pps.row_bd.len() - 1;
        let mut rs_to_ts = vec![0u32; nc];
        let mut ts_to_rs = vec![0u32; nc];
        let mut tile_id_ts = vec![0u16; nc];
        for ctb_rs in 0..nc {
            let tbx = ctb_rs % wc;
            let tby = ctb_rs / wc;
            let mut tile_x = 0;
            for i in 0..cols {
                if tbx >= pps.col_bd[i] as usize {
                    tile_x = i;
                }
            }
            let mut tile_y = 0;
            for j in 0..rows {
                if tby >= pps.row_bd[j] as usize {
                    tile_y = j;
                }
            }
            let mut ts = 0usize;
            for i in 0..tile_x {
                ts += (pps.row_bd[tile_y + 1] - pps.row_bd[tile_y]) as usize * (pps.col_bd[i + 1] - pps.col_bd[i]) as usize;
            }
            for j in 0..tile_y {
                ts += wc * (pps.row_bd[j + 1] - pps.row_bd[j]) as usize;
            }
            ts += (tby - pps.row_bd[tile_y] as usize) * (pps.col_bd[tile_x + 1] - pps.col_bd[tile_x]) as usize
                + tbx
                - pps.col_bd[tile_x] as usize;
            rs_to_ts[ctb_rs] = ts as u32;
            ts_to_rs[ts] = ctb_rs as u32;
            tile_id_ts[ts] = (tile_y * cols + tile_x) as u16;
        }
        // 6.5.2: MinTbAddrZs at 4x4 granularity (log2 min tb = 2 here; the
        // standard's table is at MinTbLog2SizeY, and z-order at a finer
        // granularity is a refinement of the same order).
        let shift = log2_ctb - 2;
        let mut min_tb_addr_zs = vec![0u32; n4];
        for y in 0..h4 {
            for x in 0..w4 {
                let tb_x = (x << 2) >> log2_ctb;
                let tb_y = (y << 2) >> log2_ctb;
                let ctb_rs = tb_y * wc + tb_x;
                let mut v = rs_to_ts[ctb_rs] << (2 * shift);
                for i in 0..shift {
                    let m = 1usize << i;
                    let mut add = 0u32;
                    if m & x != 0 {
                        add += (m * m) as u32;
                    }
                    if m & y != 0 {
                        add += (2 * m * m) as u32;
                    }
                    v += add;
                }
                min_tb_addr_zs[y * w4 + x] = v;
            }
        }
        let mut ctb_tile = vec![0u16; nc];
        for rs in 0..nc {
            ctb_tile[rs] = tile_id_ts[rs_to_ts[rs] as usize];
        }
        PicInfo {
            w4,
            h4,
            wc,
            hc,
            log2_ctb,
            pred_mode: vec![2; n4],
            skip: vec![0; n4],
            ct_depth: vec![0; n4],
            intra_mode: vec![1; n4],
            qp_y: vec![0; n4],
            filter_exempt: vec![0; n4],
            cbf_luma: vec![0; n4],
            edges: vec![0; n4],
            ctb_slice: vec![u16::MAX; nc],
            ctb_slice_addr: vec![u32::MAX; nc],
            ctb_tile,
            min_tb_addr_zs,
            ctb_rs_to_ts: rs_to_ts,
            ctb_ts_to_rs: ts_to_rs,
            tile_id_ts,
            sao: vec![[SaoParams::default(); 3]; nc],
            slices: Vec::new(),
        }
    }

    /// Reset the per-picture (not per-sequence) arrays.
    pub fn reset(&mut self) {
        self.pred_mode.fill(2);
        self.skip.fill(0);
        self.ct_depth.fill(0);
        self.intra_mode.fill(1);
        self.qp_y.fill(0);
        self.filter_exempt.fill(0);
        self.cbf_luma.fill(0);
        self.edges.fill(0);
        self.ctb_slice.fill(u16::MAX);
        self.ctb_slice_addr.fill(u32::MAX);
        for s in &mut self.sao {
            *s = [SaoParams::default(); 3];
        }
        self.slices.clear();
    }

    /// Index of the 4x4 block containing luma sample (x, y).
    #[inline(always)]
    pub fn idx4(&self, x: usize, y: usize) -> usize {
        (y >> 2) * self.w4 + (x >> 2)
    }

    /// CTB raster address of luma sample (x, y).
    #[inline(always)]
    pub fn ctb_of(&self, x: usize, y: usize) -> usize {
        (y >> self.log2_ctb) * self.wc + (x >> self.log2_ctb)
    }

    /// z-scan availability (6.4.1) of the block containing `(xn, yn)` for a
    /// current block at `(xc, yc)`.
    pub fn available(&self, xc: i32, yc: i32, xn: i32, yn: i32, pic_w: i32, pic_h: i32) -> bool {
        if xn < 0 || yn < 0 || xn >= pic_w || yn >= pic_h {
            return false;
        }
        let (xc, yc, xn, yn) = (xc as usize, yc as usize, xn as usize, yn as usize);
        if self.min_tb_addr_zs[self.idx4(xn, yn)] > self.min_tb_addr_zs[self.idx4(xc, yc)] {
            return false;
        }
        let cn = self.ctb_of(xn, yn);
        let cc = self.ctb_of(xc, yc);
        if self.ctb_slice_addr[cn] != self.ctb_slice_addr[cc] || self.ctb_slice_addr[cn] == u32::MAX {
            return false;
        }
        if self.ctb_tile[cn] != self.ctb_tile[cc] {
            return false;
        }
        // Not yet decoded (a hole from a lost slice, or the block is later in
        // the same CTB): pred_mode 2 means unwritten.
        self.pred_mode[self.idx4(xn, yn)] != 2
    }

    /// Fill a rectangle of 4x4 entries in a per-4x4 array.
    pub fn fill4<T: Copy>(arr: &mut [T], w4: usize, x: usize, y: usize, w: usize, h: usize, v: T) {
        for by in (y >> 2)..((y + h) >> 2) {
            for bx in (x >> 2)..((x + w) >> 2) {
                arr[by * w4 + bx] = v;
            }
        }
    }
}
