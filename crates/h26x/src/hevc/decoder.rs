//! The HEVC decoder: NAL dispatch, picture boundaries, POC and RPS handling
//! on the caller's thread; slice segment decoding (substreams, WPP, tiles),
//! row-pipelined loop filters and progress publication on a worker per
//! picture (frame threading, see [`crate::threading`]).

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::Arc;

use crate::cabac::Cabac;
use crate::dsp::Cpu;
use crate::dsp::hevc::HevcDsp;
use crate::nal::{HevcNalHeader, annexb_nals, escaped_offset, unescape_rbsp, unescape_rbsp_positions, unescaped_offset};
use crate::picture::{ChromaFormat, Picture};
use crate::threading::{Pool, default_threads};
use crate::{Error, Result};

use super::ctu::{SliceDec, TraceCfg};
use super::ctx::Contexts;
use super::deblock::deblock_rows;
use super::dpb::{Dpb, DpbPic, RefSets};
use super::frame::{Frame, FramePool, SharedFrame};
use super::inter::McScratch;
use super::mvpred::RefCtx;
use super::pic::{PicInfo, SliceFilterParams};
use super::pps::Pps;
use super::sao::sao_ctb_rows;
use super::slice::{SliceHeader, SliceType, nal_type};
use super::sps::{ScalingList, Sps, Vps};

/// One slice segment handed to the picture's decoder.
struct SliceJob {
    hdr: SliceHeader,
    rbsp: Vec<u8>,
    removed: Vec<usize>,
}

/// The state of one picture's decoding: lives on the worker (or inline).
struct PictureDecoder {
    frame: Arc<SharedFrame>,
    /// Built on first use (on the worker).
    info: Option<PicInfo>,
    frames: FramePool,
    sps: Sps,
    pps: Pps,
    poc: i32,
    sets: RefSets,
    scaling: Option<ScalingList>,
    dsp: HevcDsp,
    /// The independent slice segment header in force (for dependents).
    independent: Option<SliceHeader>,
    /// Contexts saved at the end of the previous slice segment.
    saved_ds: Option<Contexts>,
    /// Contexts saved after the second CTB of a row (WPP), by CTB address.
    saved_wpp: HashMap<usize, Contexts>,
    /// QpY of the last CU decoded (qPY_PREV across dependent segments).
    last_qp_y: i32,
    /// Decoded CTBs per CTB row (a row is complete at `wc`).
    row_ctbs: Vec<u16>,
    /// The next CTB row to deblock (rows below it are complete and filtered).
    next_filter_row: usize,
    /// Deblocked copy of the picture, the source SAO reads from.
    sao_src: Option<Box<Frame>>,
    deblock: bool,
    sao: bool,
    mc: McScratch,
    trace: TraceCfg,
    warnings: Arc<AtomicU64>,
}

impl PictureDecoder {
    /// First touch on the worker: take the sample buffers from the pool and
    /// build the per-picture side data.
    fn ensure_buffers(&mut self) {
        if self.info.is_none() {
            // SAFETY: nobody else touches the frame before progress > 0.
            let frame: &mut Frame = unsafe { self.frame.get_mut() };
            if frame.width == 0 {
                let mut f = self.frames.take(self.sps.width as usize, self.sps.height as usize, ChromaFormat::Yuv420, self.sps.bit_depth_luma);
                f.poc = self.poc;
                *frame = f;
            }
            self.info = Some(PicInfo::new(&self.sps, &self.pps));
        }
    }

    /// Decode one slice segment.
    fn decode_slice(&mut self, job: SliceJob) -> Result<()> {
        let SliceJob { hdr, rbsp, removed } = job;
        self.ensure_buffers();
        // SAFETY: this PictureDecoder is the picture's only writer; other
        // threads read only rows below the published progress. The Arc does
        // not move while `self` is borrowed, so the raw pointer stays valid.
        let shared: &SharedFrame = unsafe { &*(Arc::as_ptr(&self.frame)) };
        let frame: &mut Frame = unsafe { shared.get_mut() };
        let PictureDecoder { info, sps, pps, poc, sets, scaling, dsp, independent, saved_ds, saved_wpp, last_qp_y, row_ctbs, next_filter_row, sao_src, deblock, sao, mc, trace, warnings, frames, .. } = self;
        let info: &mut PicInfo = info.as_mut().expect("buffers ensured");
        let frames: &FramePool = frames;
        let sps: &Sps = sps;
        let pps: &Pps = pps;
        let cur_poc = *poc;
        if !hdr.dependent {
            *independent = Some(hdr.clone());
            info.slices.push(SliceFilterParams {
                deblocking_disabled: hdr.deblocking_disabled,
                beta_offset: hdr.beta_offset,
                tc_offset: hdr.tc_offset,
                loop_filter_across_slices: hdr.loop_filter_across_slices,
                slice_addr: hdr.segment_address,
                cb_qp_offset: pps.cb_qp_offset,
                cr_qp_offset: pps.cr_qp_offset,
            });
        }
        let Some(ind) = independent.as_ref() else {
            warnings.fetch_add(1, Ordering::Relaxed);
            return Ok(());
        };
        let slice_idx = (info.slices.len() - 1) as u16;
        let slice_addr = ind.segment_address;
        // Reference picture lists.
        let lists = if ind.slice_type != SliceType::I { sets.build_ref_lists(ind)? } else { [Vec::new(), Vec::new()] };
        // SAFETY: reads of reference samples/motion wait on their progress.
        let ref_frames: [Vec<&Frame>; 2] = [lists[0].iter().map(|e| unsafe { e.frame.get() }).collect(), lists[1].iter().map(|e| unsafe { e.frame.get() }).collect()];
        let ref_shared: [Vec<&SharedFrame>; 2] = [lists[0].iter().map(|e| &*e.frame).collect(), lists[1].iter().map(|e| &*e.frame).collect()];
        let pocs: [Vec<i32>; 2] = [lists[0].iter().map(|e| e.poc).collect(), lists[1].iter().map(|e| e.poc).collect()];
        let long_term: [Vec<bool>; 2] = [lists[0].iter().map(|e| e.long_term).collect(), lists[1].iter().map(|e| e.long_term).collect()];
        let no_backward_pred = pocs[0].iter().chain(pocs[1].iter()).all(|&p| p <= cur_poc);
        let (col, col_shared) = if ind.temporal_mvp_enabled && ind.slice_type != SliceType::I {
            let list = if ind.slice_type == SliceType::B && !ind.collocated_from_l0 { 1 } else { 0 };
            match lists[list].get(ind.collocated_ref_idx as usize) {
                Some(e) => (Some(unsafe { e.frame.get() }), Some(&*e.frame)),
                None => return Err(Error::bitstream("collocated_ref_idx out of range")),
            }
        } else {
            (None, None)
        };
        let refs = RefCtx {
            pocs,
            long_term,
            col,
            cur_poc,
            no_backward_pred,
            tmvp: ind.temporal_mvp_enabled,
            max_merge_cand: ind.max_num_merge_cand as usize,
            log2_par_mrg_level: pps.log2_parallel_merge_level,
            is_b: ind.slice_type == SliceType::B,
            num_ref_idx: [ind.num_ref_idx[0] as usize, ind.num_ref_idx[1] as usize],
            col_from_l0: ind.collocated_from_l0,
        };

        // Substreams: byte offsets in the escaped NAL, relative to the start
        // of the slice segment data.
        let data_start_unesc = (hdr.data_bit_offset / 8) as usize;
        let data_start_esc = escaped_offset(data_start_unesc, &removed);
        let mut substreams: Vec<usize> = vec![data_start_unesc.min(rbsp.len())];
        for &ep in &hdr.entry_points {
            let esc = data_start_esc + ep as usize;
            substreams.push(unescaped_offset(esc, &removed).min(rbsp.len()));
        }
        let init_type = match ind.slice_type {
            SliceType::I => 0,
            SliceType::P => {
                if ind.cabac_init {
                    2
                } else {
                    1
                }
            }
            SliceType::B => {
                if ind.cabac_init {
                    1
                } else {
                    2
                }
            }
        };
        let wc = info.wc;
        let n_ctbs = info.wc * info.hc;
        let mut ctb_addr_rs = hdr.segment_address as usize;
        if ctb_addr_rs >= n_ctbs {
            return Err(Error::bitstream("slice_segment_address out of range"));
        }
        let mut ctb_addr_ts = info.ctb_rs_to_ts[ctb_addr_rs] as usize;
        let mut sub = 0usize;
        let cabac = Cabac::new(&rbsp[substreams[0]..]);
        let tile_col_start = |rs: usize| -> usize {
            let rx = rs % wc;
            let mut start = 0;
            for &b in &pps.col_bd {
                if (b as usize) <= rx {
                    start = b as usize;
                }
            }
            start
        };

        // Contexts at the start of the segment (9.3.1).
        let first_in_tile = ctb_addr_ts == 0 || info.tile_id_ts[ctb_addr_ts] != info.tile_id_ts[ctb_addr_ts - 1];
        let row_start = pps.entropy_coding_sync && ctb_addr_rs % wc == tile_col_start(ctb_addr_rs);
        let mut cx = Contexts::new(init_type, ind.slice_qp);
        let mut first_qg = true;
        let mut qp_prev_init = ind.slice_qp;
        if first_in_tile {
            // init
        } else if row_start {
            if let Some(saved) = wpp_sync_source(info, saved_wpp, ctb_addr_rs, slice_addr) {
                cx = saved.clone();
            }
        } else if hdr.dependent {
            if let Some(saved) = saved_ds.as_ref() {
                cx = saved.clone();
            }
            first_qg = false;
            qp_prev_init = *last_qp_y;
        }

        let mut dec = SliceDec {
            sps,
            pps,
            hdr: ind,
            frame,
            info,
            cabac,
            cx,
            refs,
            ref_frames,
            ref_shared,
            col_shared,
            slice_idx,
            slice_addr,
            scaling: scaling.clone(),
            qp_y: ind.slice_qp,
            qp_y_prev: qp_prev_init,
            cu_qp_delta_val: 0,
            is_cu_qp_delta_coded: false,
            qg: (0, 0),
            qg_qp_prev: qp_prev_init,
            first_qg,
            last_pu_merged: false,
            ctb_addr_rs,
            ctb_addr_ts,
            coeffs: vec![0; 1024],
            dsp: *dsp,
            mc: std::mem::take(mc),
            warnings: 0,
            trace: *trace,
        };
        let mut filters = RowFilters { row_ctbs, next_filter_row, sao_src, deblock: *deblock, sao: *sao, dsp, sps, pps, shared, frames };

        let result = (|| -> Result<()> {
            loop {
                let rx = ctb_addr_rs % wc;
                dec.decode_ctu(ctb_addr_rs, ctb_addr_ts)?;
                let end_of_slice_segment = dec.cabac.terminate() != 0;
                if pps.entropy_coding_sync && rx == tile_col_start(ctb_addr_rs) + 1 {
                    saved_wpp.insert(ctb_addr_rs, dec.cx.clone());
                }
                filters.ctb_done(ctb_addr_rs, dec.frame, dec.info);
                if end_of_slice_segment {
                    break;
                }
                ctb_addr_ts += 1;
                if ctb_addr_ts >= n_ctbs {
                    return Err(Error::bitstream("slice segment runs past the picture"));
                }
                ctb_addr_rs = dec.info.ctb_ts_to_rs[ctb_addr_ts] as usize;
                let new_tile = dec.info.tile_id_ts[ctb_addr_ts] != dec.info.tile_id_ts[ctb_addr_ts - 1];
                let new_row = pps.entropy_coding_sync && ctb_addr_rs % wc == tile_col_start(ctb_addr_rs);
                if new_tile || new_row {
                    sub += 1;
                    let Some(&start) = substreams.get(sub) else {
                        return Err(Error::bitstream("missing entry point for a new substream"));
                    };
                    dec.cabac = Cabac::new(&rbsp[start..]);
                    dec.cx = if new_tile {
                        Contexts::new(init_type, ind.slice_qp)
                    } else {
                        match wpp_sync_source(dec.info, saved_wpp, ctb_addr_rs, slice_addr) {
                            Some(c) => c.clone(),
                            None => Contexts::new(init_type, ind.slice_qp),
                        }
                    };
                    dec.first_qg = true;
                    dec.qp_y_prev = ind.slice_qp;
                }
                if dec.cabac.overrun() {
                    return Err(Error::bitstream("slice data exhausted"));
                }
            }
            Ok(())
        })();
        *last_qp_y = dec.qp_y;
        let cx_end = dec.cx.clone();
        warnings.fetch_add(dec.warnings, Ordering::Relaxed);
        *mc = std::mem::take(&mut dec.mc);
        drop(dec);
        if pps.dependent_slice_segments_enabled {
            *saved_ds = Some(cx_end);
        }
        result
    }

    /// The picture is over: finish the filters and publish completion.
    fn finish(mut self) {
        self.ensure_buffers();
        // SAFETY: as in decode_slice.
        let shared: &SharedFrame = unsafe { &*(Arc::as_ptr(&self.frame)) };
        let frame: &mut Frame = unsafe { shared.get_mut() };
        let PictureDecoder { info, sps, pps, row_ctbs, next_filter_row, sao_src, deblock, sao, dsp, poc, frames, .. } = &mut self;
        let info: &PicInfo = info.as_ref().expect("buffers ensured");
        let mut filters = RowFilters { row_ctbs, next_filter_row, sao_src, deblock: *deblock, sao: *sao, dsp, sps, pps, shared, frames };
        filters.finish(frame, info);
        frame.poc = *poc;
        shared.progress.finish();
        // The SAO source copy goes back to the pool for the next picture.
        if let Some(src) = sao_src.take() {
            frames.give(*src);
        }
    }
}

/// The row-pipelined loop filters of one picture. Intra prediction reads
/// the *unfiltered* samples of the row above, so a CTB row may only be
/// deblocked once the row below it is fully decoded (libavcodec filters
/// with the same lag, one CTB diagonal). So when CTB row `r` completes: row
/// `r - 1` is deblocked (vertical edges then horizontal — that settles row
/// `r - 2`), row `r - 2` is SAO'd from the deblocked copy, its borders are
/// extended and its rows are published as final for the pictures waiting on
/// this one.
struct RowFilters<'a> {
    row_ctbs: &'a mut Vec<u16>,
    /// The next CTB row whose completion has not been acted on.
    next_filter_row: &'a mut usize,
    sao_src: &'a mut Option<Box<Frame>>,
    deblock: bool,
    sao: bool,
    dsp: &'a HevcDsp,
    sps: &'a Sps,
    pps: &'a Pps,
    shared: &'a SharedFrame,
    frames: &'a FramePool,
}

impl RowFilters<'_> {
    fn ctb_done(&mut self, ctb_addr_rs: usize, frame: &mut Frame, info: &PicInfo) {
        let wc = info.wc;
        let ry = ctb_addr_rs / wc;
        self.row_ctbs[ry] += 1;
        while *self.next_filter_row < info.hc && self.row_ctbs[*self.next_filter_row] as usize >= wc {
            let r = *self.next_filter_row;
            self.row_complete(r, frame, info);
            *self.next_filter_row += 1;
        }
    }

    fn row_span(&self, r: usize, frame: &Frame) -> (usize, usize) {
        let ctb = 1usize << self.sps.log2_ctb_size;
        (r * ctb, ((r + 1) * ctb).min(frame.height))
    }

    /// CTB row `r` is complete (and all rows above it were completed before).
    fn row_complete(&mut self, r: usize, frame: &mut Frame, info: &PicInfo) {
        let (_, y1) = self.row_span(r, frame);
        self.shared.progress.set_decoded(y1 as i32);
        if r >= 1 {
            self.deblock_row(r - 1, frame, info);
        }
        if r >= 2 {
            self.sao_and_publish(r - 2, frame, info);
        }
    }

    fn deblock_row(&mut self, r: usize, frame: &mut Frame, info: &PicInfo) {
        if !self.deblock {
            return;
        }
        let (y0, y1) = self.row_span(r, frame);
        deblock_rows(frame, info, self.pps, self.sps.bit_depth_luma, self.sps.bit_depth_chroma, y0 / 4, y1.div_ceil(4));
    }

    /// Row `r` is deblocked-final (row `r + 1` has been deblocked, or there is
    /// none): SAO it, extend its borders, publish.
    fn sao_and_publish(&mut self, r: usize, frame: &mut Frame, info: &PicInfo) {
        let (y0, y1) = self.row_span(r, frame);
        if self.sao && self.sps.sao_enabled {
            let frames = self.frames;
            let src = self.sao_src.get_or_insert_with(|| Box::new(frames.take(frame.width, frame.height, frame.chroma, frame.bit_depth)));
            // All of row r plus the first line of row r + 1 (deblocked-final:
            // only row r + 1's own vertical edges and top edge touch it, both
            // done). The last line of row r - 1 was copied one step earlier.
            copy_rows(frame, src, y0, (y1 + 1).min(frame.height));
            sao_ctb_rows(self.dsp, frame, src, info, self.sps, self.pps, r, r + 1);
        }
        frame.extend_rows(y0, y1);
        self.shared.progress.set_done(y1 as i32);
    }

    /// End of picture: rows never completed (lost slices) count as complete;
    /// then the last row is deblocked and the last two rows are finished.
    fn finish(&mut self, frame: &mut Frame, info: &PicInfo) {
        let wc = info.wc;
        for r in *self.next_filter_row..info.hc {
            self.row_ctbs[r] = wc as u16;
            self.row_complete(r, frame, info);
            *self.next_filter_row = r + 1;
        }
        if info.hc > 0 {
            let last = info.hc - 1;
            self.deblock_row(last, frame, info);
            if last >= 1 {
                self.sao_and_publish(last - 1, frame, info);
            }
            self.sao_and_publish(last, frame, info);
        }
    }
}

/// Copy luma rows `y0..y1` (and the matching chroma rows) of `from` into `to`.
fn copy_rows(from: &Frame, to: &mut Frame, y0: usize, y1: usize) {
    if y0 >= y1 {
        return;
    }
    let (s, e) = (from.y.offset(0, y0 as isize) - from.y.pad, from.y.offset(0, y1 as isize) - from.y.pad);
    to.y.data[s..e].copy_from_slice(&from.y.data[s..e]);
    if from.chroma != ChromaFormat::Monochrome {
        let (cy0, cy1) = (y0 / 2, y1.div_ceil(2));
        let (s, e) = (from.cb.offset(0, cy0 as isize) - from.cb.pad, from.cb.offset(0, cy1 as isize) - from.cb.pad);
        to.cb.data[s..e].copy_from_slice(&from.cb.data[s..e]);
        to.cr.data[s..e].copy_from_slice(&from.cr.data[s..e]);
    }
}

/// The picture currently being fed (main-thread view).
struct Current {
    id: u64,
    pic_output: bool,
    /// The independent slice header in force (dependent segments copy it).
    independent: Option<SliceHeader>,
    /// Slices go to the worker through this...
    tx: Option<Sender<SliceJob>>,
    /// ...or are decoded here when there is no pool.
    inline: Option<PictureDecoder>,
}

/// A native HEVC (H.265) decoder — Main / Main 10 / Main 12, 4:2:0.
///
/// Frame-threaded: pictures are decoded on a worker pool as their slices
/// arrive, with row-level dependency tracking between a picture and its
/// references. The caller's thread only parses headers and manages the
/// DPB, so `push_nal` returns quickly; [`HevcDecoder::next_picture`] hands
/// back pictures in output order, waiting for each to finish.
pub struct HevcDecoder {
    vps: HashMap<u32, Vps>,
    sps: HashMap<u32, Sps>,
    pps: HashMap<u32, Pps>,
    dpb: Dpb,
    cur: Option<Current>,
    /// POC of the previous TemporalId-0 non-RASL/RADL/SLNR picture.
    prev_tid0_poc: i32,
    /// The next IRAP starts a new coded video sequence.
    first_in_sequence: bool,
    /// NoRaslOutputFlag of the associated IRAP picture.
    no_rasl_output: bool,
    /// The current picture is being skipped (RASL after a NoRaslOutput IRAP).
    skipping: bool,
    decode_index: u64,
    warnings: Arc<AtomicU64>,
    dsp: HevcDsp,
    pool: Option<Arc<Pool>>,
    frames: FramePool,
    deblock: bool,
    sao: bool,
}

impl Default for HevcDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl HevcDecoder {
    /// A decoder with one worker per hardware thread (capped), or as
    /// `H26X_THREADS` says.
    pub fn new() -> Self {
        let n = std::env::var("H26X_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or_else(default_threads);
        Self::with_threads(n)
    }

    /// A decoder with `threads` workers; 0 or 1 decodes on the caller's thread.
    pub fn with_threads(threads: usize) -> Self {
        let pool = if threads > 1 { Some(Arc::new(Pool::new(threads, threads))) } else { None };
        HevcDecoder {
            vps: HashMap::new(),
            sps: HashMap::new(),
            pps: HashMap::new(),
            dpb: Dpb::new(),
            cur: None,
            prev_tid0_poc: 0,
            first_in_sequence: true,
            no_rasl_output: true,
            skipping: false,
            decode_index: 0,
            warnings: Arc::new(AtomicU64::new(0)),
            dsp: HevcDsp::new(Cpu::detect_honouring_env()),
            pool,
            frames: FramePool::new(),
            deblock: std::env::var_os("H26X_NO_DEBLOCK").is_none(),
            sao: std::env::var_os("H26X_NO_SAO").is_none(),
        }
    }

    /// Non-fatal problems seen so far.
    pub fn warnings(&self) -> u64 {
        self.warnings.load(Ordering::Relaxed) + self.dpb.warnings
    }

    /// Feed a chunk of Annex-B bytes (whole NAL units).
    pub fn push_annexb(&mut self, data: &[u8]) -> Result<()> {
        for nal in annexb_nals(data) {
            self.push_nal(nal)?;
        }
        Ok(())
    }

    /// Feed one NAL unit (with its two header bytes, without start code).
    pub fn push_nal(&mut self, nal: &[u8]) -> Result<()> {
        let Some(hdr) = HevcNalHeader::parse(nal) else {
            return Err(Error::bitstream("bad NAL header"));
        };
        if hdr.layer_id != 0 {
            return Ok(());
        }
        match hdr.unit_type {
            nal_type::VPS => {
                let rbsp = unescape_rbsp(nal);
                let v = Vps::parse(&rbsp[2..])?;
                self.vps.insert(v.id, v);
            }
            nal_type::SPS => {
                let rbsp = unescape_rbsp(nal);
                let s = Sps::parse(&rbsp[2..])?;
                self.sps.insert(s.id, s);
            }
            nal_type::PPS => {
                let rbsp = unescape_rbsp(nal);
                let p = Pps::parse(&rbsp[2..])?;
                self.pps.insert(p.id, p);
            }
            nal_type::EOS | nal_type::EOB => {
                self.finish_picture();
                self.first_in_sequence = true;
            }
            nal_type::AUD | nal_type::SEI_PREFIX | nal_type::SEI_SUFFIX | nal_type::FD => {}
            t if nal_type::is_slice(t) => self.slice_nal(nal, hdr)?,
            _ => {}
        }
        Ok(())
    }

    /// End of stream: finish the current picture and drain the DPB.
    pub fn flush(&mut self) -> Result<()> {
        self.finish_picture();
        self.dpb.flush();
        Ok(())
    }

    /// The next picture in output order, if any (waits for it to finish).
    pub fn next_picture(&mut self) -> Option<Picture> {
        self.dpb.output.pop_front().map(|p| p.into_picture())
    }

    /// The next picture in output order if it is already finished.
    pub fn try_next_picture(&mut self) -> Option<Picture> {
        if self.dpb.output.front().is_some_and(|p| p.frame.progress.is_complete()) {
            return self.next_picture();
        }
        None
    }

    fn slice_nal(&mut self, nal: &[u8], nh: HevcNalHeader) -> Result<()> {
        let (rbsp, removed) = unescape_rbsp_positions(nal);
        let first_flag = rbsp.get(2).is_some_and(|b| b & 0x80 != 0);
        if first_flag {
            self.finish_picture();
            self.skipping = false;
        } else if self.cur.is_none() && !self.skipping {
            self.warnings.fetch_add(1, Ordering::Relaxed);
            return Ok(());
        }
        if self.skipping && !first_flag {
            return Ok(());
        }
        let sps_map = &self.sps;
        let pps_map = &self.pps;
        let independent = self.cur.as_ref().and_then(|c| c.independent.as_ref());
        let (hdr, mut pps, sps) = SliceHeader::parse(&rbsp, nh, &|id| pps_map.get(&id).cloned(), &|id| sps_map.get(&id).cloned(), independent)?;
        if first_flag {
            if sps.chroma_format_idc != 1 || sps.separate_colour_plane {
                return Err(Error::unsupported(format!("chroma_format_idc {} (only 4:2:0)", sps.chroma_format_idc)));
            }
            if sps.bit_depth_luma != sps.bit_depth_chroma {
                return Err(Error::unsupported("different luma and chroma bit depths"));
            }
            if sps.bit_depth_luma > 12 {
                return Err(Error::unsupported(format!("bit depth {}", sps.bit_depth_luma)));
            }
            if let Some(ext) = &sps.range_ext {
                if ext.iter().any(|&f| f) {
                    return Err(Error::unsupported("range extension tools"));
                }
            }
            if pps.cross_component_prediction || pps.chroma_qp_offset_list {
                return Err(Error::unsupported("PPS range extension tools (cross-component prediction / chroma QP offset lists)"));
            }
            pps.resolve_tiles(&sps)?;
            self.start_picture(&hdr, sps, pps, nh)?;
            if self.skipping {
                return Ok(());
            }
        }
        let Some(cur) = self.cur.as_mut() else { return Ok(()) };
        if !hdr.dependent {
            cur.independent = Some(hdr.clone());
        }
        let job = SliceJob { hdr, rbsp, removed };
        if let Some(tx) = &cur.tx {
            let _ = tx.send(job);
        } else if let Some(pd) = cur.inline.as_mut() {
            if let Err(e) = pd.decode_slice(job) {
                pd.frame.progress.error.store(true, Ordering::Relaxed);
                self.warnings.fetch_add(1, Ordering::Relaxed);
                return Err(e);
            }
        }
        Ok(())
    }

    fn start_picture(&mut self, hdr: &SliceHeader, sps: Sps, pps: Pps, nh: HevcNalHeader) -> Result<()> {
        let t = nh.unit_type;
        let irap = nal_type::is_irap(t);
        if irap {
            self.no_rasl_output = nal_type::is_idr(t) || nal_type::is_bla(t) || self.first_in_sequence;
        }
        if nal_type::is_rasl(t) && self.no_rasl_output {
            self.skipping = true;
            return Ok(());
        }
        // POC (8.3.1).
        let max_poc_lsb = sps.max_poc_lsb();
        let lsb = hdr.poc_lsb as i32;
        let msb = if irap && self.no_rasl_output {
            0
        } else {
            let prev_lsb = self.prev_tid0_poc & (max_poc_lsb - 1);
            let prev_msb = self.prev_tid0_poc - prev_lsb;
            if lsb < prev_lsb && (prev_lsb - lsb) >= max_poc_lsb / 2 {
                prev_msb + max_poc_lsb
            } else if lsb > prev_lsb && (lsb - prev_lsb) > max_poc_lsb / 2 {
                prev_msb - max_poc_lsb
            } else {
                prev_msb
            }
        };
        let poc = msb + lsb;
        if nh.temporal_id == 0 && !nal_type::is_rasl(t) && !nal_type::is_radl(t) && !nal_type::is_sub_layer_non_ref(t) {
            self.prev_tid0_poc = poc;
        }
        let first_pic = self.decode_index == 0;
        self.first_in_sequence = false;

        self.dpb.configure(&sps);
        let chroma = ChromaFormat::Yuv420;
        let bit_depth = sps.bit_depth_luma;
        let crop = sps.conf_win;
        let idr = nal_type::is_idr(t);
        let sets = self.dpb.apply_rps(hdr, &sps, poc, idr, chroma, bit_depth, self.decode_index, crop);
        if irap && self.no_rasl_output && !first_pic {
            let no_output = if t == nal_type::CRA { true } else { hdr.no_output_of_prior_pics };
            self.dpb.before_decode(true, no_output);
        } else {
            self.dpb.before_decode(false, false);
        }
        let pic_output = hdr.pic_output;

        // The buffers are taken from the pool by the worker (allocation and
        // zeroing off this thread); nobody reads them before progress says so.
        let id = self.dpb.alloc_id();
        let shared = Arc::new(SharedFrame::with_pool(Frame::empty(), poc, id, self.frames.clone()));
        self.dpb.insert_current(DpbPic {
            frame: shared.clone(),
            poc,
            is_ref: true,
            long_term: false,
            needed_for_output: false,
            latency: 0,
            decode_index: self.decode_index,
            crop,
            generated: false,
        });
        let scaling = if sps.scaling_list_enabled {
            Some(match (&pps.scaling_list, &sps.scaling_list) {
                (Some(p), _) => p.clone(),
                (None, Some(s)) => s.clone(),
                (None, None) => ScalingList::default_lists(),
            })
        } else {
            None
        };
        let hc = sps.pic_height_in_ctbs() as usize;
        let pd = PictureDecoder {
            frame: shared,
            info: None,
            frames: self.frames.clone(),
            sps,
            pps,
            poc,
            sets,
            scaling,
            dsp: self.dsp,
            independent: None,
            saved_ds: None,
            saved_wpp: HashMap::new(),
            last_qp_y: 0,
            row_ctbs: vec![0; hc],
            next_filter_row: 0,
            sao_src: None,
            deblock: self.deblock,
            sao: self.sao,
            mc: McScratch::new(),
            trace: TraceCfg::from_env(),
            warnings: self.warnings.clone(),
        };
        let cur = match &self.pool {
            Some(pool) => {
                let (tx, rx): (Sender<SliceJob>, Receiver<SliceJob>) = channel();
                let mut pd = pd;
                pool.submit(Box::new(move || {
                    for job in rx {
                        if pd.decode_slice(job).is_err() {
                            pd.frame.progress.error.store(true, Ordering::Relaxed);
                            pd.warnings.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                    pd.finish();
                }));
                Current { id, pic_output, independent: None, tx: Some(tx), inline: None }
            }
            None => Current { id, pic_output, independent: None, tx: None, inline: Some(pd) },
        };
        self.cur = Some(cur);
        self.decode_index += 1;
        Ok(())
    }

    fn finish_picture(&mut self) {
        let Some(cur) = self.cur.take() else { return };
        // Closing the channel ends the worker's slice loop; inline, finish now.
        drop(cur.tx);
        if let Some(pd) = cur.inline {
            pd.finish();
        }
        self.dpb.finish_current(cur.id, cur.pic_output);
    }
}

impl Drop for HevcDecoder {
    fn drop(&mut self) {
        // Let in-flight pictures finish before the pool goes away.
        self.finish_picture();
        if let Some(p) = &self.pool {
            p.wait_idle();
        }
    }
}

/// The WPP synchronisation source for the first CTB of a row at
/// `ctb_addr_rs`: the contexts saved after the second CTB of the row above
/// (its above-right neighbour), if that CTB is available (same slice and
/// tile, decoded).
fn wpp_sync_source<'a>(info: &PicInfo, saved: &'a HashMap<usize, Contexts>, ctb_addr_rs: usize, slice_addr: u32) -> Option<&'a Contexts> {
    let wc = info.wc;
    let row = ctb_addr_rs / wc;
    if row == 0 {
        return None;
    }
    let above_right = ctb_addr_rs + 1 - wc;
    if above_right / wc != row - 1 {
        return None;
    }
    if info.ctb_slice_addr[above_right] != slice_addr || info.ctb_tile[above_right] != info.ctb_tile[ctb_addr_rs] {
        return None;
    }
    saved.get(&above_right)
}
