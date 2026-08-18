//! The H.264 decoder: NAL dispatch, access-unit boundaries, POC and
//! reference marking on the caller's thread; slice decoding (CAVLC or
//! CABAC), row-pipelined deblocking and progress publication on a worker
//! per picture (frame threading, see [`crate::threading`]).

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::Arc;

use crate::bitreader::BitReader;
use crate::cabac::Cabac;
use crate::nal::{H264NalHeader, annexb_nals, unescape_rbsp};
use crate::picture::{ChromaFormat, Picture};
use crate::threading::{Pool, default_threads};
use crate::{Error, Result};

use super::cabac_mb::{CabacState, decode_end_of_slice, decode_mb_skip, parse_mb_cabac};
use super::cavlc::parse_mb_cavlc;
use super::deblock::{DeblockParams, deblock_mb_rows};
use super::dpb::{DecodedPic, Dpb, PocState, RefEntry, RefMark, build_ref_lists, compute_poc};
use super::frame::{Frame, FramePool, SharedFrame};
use super::mb::{MbKind, MbLayer, MbNeighbours, PicInfo, SliceCtx};
use super::pps::Pps;
use super::recon::{QpState, SliceRefs, reconstruct};
use super::slice::{Mmco, SliceHeader, SliceType};
use super::sps::{ScalingLists, Sps};
use super::transform::Dequant;

/// One slice handed to the picture's decoder, with its references resolved.
struct SliceJob {
    hdr: SliceHeader,
    rbsp: Vec<u8>,
    pps: Arc<Pps>,
    dequant: Arc<Dequant>,
    /// RefPicList0/1 (grey stand-ins already substituted).
    refs: [Vec<RefEntry>; 2],
    /// RefPicList1[0] for direct prediction.
    col: Option<RefEntry>,
}

/// The state of one picture's decoding: lives on the worker (or inline).
struct PictureDecoder {
    frame: Arc<SharedFrame>,
    info: Option<PicInfo>,
    frames: FramePool,
    sps: Sps,
    poc: i32,
    slices: Vec<DeblockParams>,
    /// Decoded macroblocks per MB row (a row is complete at `mb_width`).
    row_mbs: Vec<u16>,
    /// The next MB row whose completion has not been acted on.
    next_filter_row: usize,
    deblock: bool,
    warnings: Arc<AtomicU64>,
}

impl PictureDecoder {
    fn ensure_buffers(&mut self) {
        if self.info.is_none() {
            let mbw = self.sps.pic_width_in_mbs as usize;
            let mbh = self.sps.frame_height_in_mbs() as usize;
            let mut info = PicInfo::new(mbw, mbh);
            info.reset();
            self.info = Some(info);
        }
    }

    fn decode_slice(&mut self, job: SliceJob) -> Result<()> {
        self.ensure_buffers();
        let SliceJob { hdr, rbsp, pps, dequant, refs: ref_lists, col } = job;
        // SAFETY: this PictureDecoder is the picture's only writer; other
        // threads read only rows below the published progress. The Arc does
        // not move while `self` is borrowed.
        let shared: &SharedFrame = unsafe { &*(Arc::as_ptr(&self.frame)) };
        let cur: &mut Frame = unsafe { shared.get_mut() };
        let PictureDecoder { info, sps, poc, slices, row_mbs, next_filter_row, deblock, warnings, .. } = self;
        let info: &mut PicInfo = info.as_mut().expect("buffers ensured");
        let cur_poc = *poc;

        // SAFETY: reference reads wait on progress.
        let mut refs = SliceRefs {
            frames: [ref_lists[0].iter().map(|e| unsafe { e.frame.get() }).collect(), ref_lists[1].iter().map(|e| unsafe { e.frame.get() }).collect()],
            shared: [ref_lists[0].iter().map(|e| &*e.frame).collect(), ref_lists[1].iter().map(|e| &*e.frame).collect()],
            col_shared: col.as_ref().map(|e| &*e.frame),
            pocs: [ref_lists[0].iter().map(|e| e.poc).collect(), ref_lists[1].iter().map(|e| e.poc).collect()],
            long_term: [ref_lists[0].iter().map(|e| e.long_term).collect(), ref_lists[1].iter().map(|e| e.long_term).collect()],
            col: col.as_ref().map(|e| unsafe { e.frame.get() }),
            col_long_term: col.as_ref().is_some_and(|e| e.long_term),
            explicit: hdr.pred_weights.as_ref(),
            implicit: None,
            cur_poc,
        };
        if hdr.slice_type.is_b() && pps.weighted_bipred_idc == 2 {
            refs.build_implicit();
        }

        let slice_num = slices.len() as u16;
        slices.push(DeblockParams { disable_idc: hdr.disable_deblocking_filter_idc, offset_a: hdr.filter_offset_a, offset_b: hdr.filter_offset_b });
        let ctx = SliceCtx {
            slice_type: hdr.slice_type,
            slice_num,
            num_ref_idx: hdr.num_ref_idx_active,
            direct_spatial: hdr.direct_spatial_mv_pred,
            transform_8x8_mode: pps.transform_8x8_mode,
            constrained_intra_pred: pps.constrained_intra_pred,
            direct_8x8_inference: sps.direct_8x8_inference,
            chroma_format_idc: sps.chroma_format_idc,
        };
        let mut qps = QpState { prev_qp: hdr.slice_qp, chroma_offset: [pps.chroma_qp_index_offset, pps.second_chroma_qp_index_offset] };
        let total_mbs = cur.mb_width * cur.mb_height;
        let mbw = cur.mb_width;
        let mut addr = hdr.first_mb_in_slice as usize;
        if addr >= total_mbs {
            return Err(Error::bitstream("first_mb_in_slice beyond the picture"));
        }
        let data_start = (hdr.data_bit_offset / 8) as usize;
        let mut filters = RowFilters { row_mbs, next_filter_row, deblock: *deblock, shared, slices };
        let dq: &Dequant = &dequant;

        // Decode one macroblock and account for it.
        macro_rules! mb_done {
            () => {{
                filters.mb_done(addr, cur, info);
                addr += 1;
            }};
        }

        if pps.cabac {
            let mut cabac = Cabac::new(&rbsp[data_start..]);
            let mut st = CabacState::new(hdr.slice_type, hdr.cabac_init_idc, hdr.slice_qp);
            loop {
                if addr >= total_mbs {
                    return Err(Error::bitstream("slice data runs past the picture"));
                }
                let nb = MbNeighbours::derive(info, addr, slice_num);
                let mut layer: Option<MbLayer> = None;
                if !hdr.slice_type.is_intra() {
                    let skip = decode_mb_skip(&mut cabac, &mut st, info, &nb, hdr.slice_type.is_b());
                    if skip {
                        let kind = if hdr.slice_type.is_b() { MbKind::BSkip } else { MbKind::PSkip };
                        layer = Some(MbLayer::new(kind));
                        st.prev_qp_delta_nonzero = false;
                    }
                }
                let layer = match layer {
                    Some(l) => l,
                    None => parse_mb_cabac(&mut cabac, &mut st, &ctx, info, &nb, &cur.motion)?,
                };
                reconstruct(&ctx, &mut qps, dq, cur, info, &nb, &layer, &refs)?;
                mb_done!();
                if decode_end_of_slice(&mut cabac) {
                    break;
                }
                if cabac.overrun() {
                    return Err(Error::bitstream("CABAC slice data exhausted before end_of_slice_flag"));
                }
            }
        } else {
            let mut r = BitReader::new(&rbsp);
            r.skip(hdr.data_bit_offset as u32);
            loop {
                if !hdr.slice_type.is_intra() {
                    let run = r.ue() as usize;
                    if run > total_mbs {
                        return Err(Error::bitstream("mb_skip_run out of range"));
                    }
                    for _ in 0..run {
                        if addr >= total_mbs {
                            return Err(Error::bitstream("slice data runs past the picture"));
                        }
                        let nb = MbNeighbours::derive(info, addr, slice_num);
                        let kind = if hdr.slice_type.is_b() { MbKind::BSkip } else { MbKind::PSkip };
                        let layer = MbLayer::new(kind);
                        reconstruct(&ctx, &mut qps, dq, cur, info, &nb, &layer, &refs)?;
                        mb_done!();
                    }
                    if run > 0 && !r.more_rbsp_data() {
                        break;
                    }
                }
                if addr >= total_mbs {
                    return Err(Error::bitstream("slice data runs past the picture"));
                }
                let nb = MbNeighbours::derive(info, addr, slice_num);
                let t = r.ue();
                let layer = parse_mb_cavlc(&mut r, &ctx, info, &nb, t)?;
                reconstruct(&ctx, &mut qps, dq, cur, info, &nb, &layer, &refs)?;
                mb_done!();
                if r.overrun() {
                    return Err(Error::bitstream("CAVLC slice data exhausted"));
                }
                if !r.more_rbsp_data() {
                    break;
                }
            }
        }
        let _ = (mbw, warnings);
        Ok(())
    }

    /// The picture is over: finish the filters and publish completion.
    fn finish(mut self) {
        self.ensure_buffers();
        // SAFETY: as in decode_slice.
        let shared: &SharedFrame = unsafe { &*(Arc::as_ptr(&self.frame)) };
        let cur: &mut Frame = unsafe { shared.get_mut() };
        let PictureDecoder { info, poc, slices, row_mbs, next_filter_row, deblock, warnings, .. } = &mut self;
        let info: &PicInfo = info.as_ref().expect("buffers ensured");
        let missing = info.mbs.iter().filter(|m| !m.decoded).count();
        if missing > 0 {
            warnings.fetch_add(1, Ordering::Relaxed);
        }
        let mut filters = RowFilters { row_mbs, next_filter_row, deblock: *deblock, shared, slices };
        filters.finish(cur, info);
        cur.poc = *poc;
        shared.progress.finish();
    }
}

/// Row-pipelined deblocking of one picture: MB row `r` is filtered once row
/// `r + 1` is fully decoded (intra prediction reads unfiltered neighbours),
/// which settles row `r - 1` (its bottom lines are touched by row `r`'s top
/// edges); row `r - 1` is then edge-extended and published.
struct RowFilters<'a> {
    row_mbs: &'a mut Vec<u16>,
    next_filter_row: &'a mut usize,
    deblock: bool,
    shared: &'a SharedFrame,
    slices: &'a Vec<DeblockParams>,
}

impl RowFilters<'_> {
    fn mb_done(&mut self, addr: usize, frame: &mut Frame, info: &PicInfo) {
        let mbw = info.mb_width;
        let r = addr / mbw;
        self.row_mbs[r] += 1;
        while *self.next_filter_row < info.mb_height && self.row_mbs[*self.next_filter_row] as usize >= mbw {
            let row = *self.next_filter_row;
            self.row_complete(row, frame, info);
            *self.next_filter_row += 1;
        }
    }

    fn row_complete(&mut self, r: usize, frame: &mut Frame, info: &PicInfo) {
        self.shared.progress.set_decoded(((r + 1) * 16) as i32);
        if r >= 1 {
            if self.deblock {
                deblock_mb_rows(frame, info, self.slices, r - 1, r);
            }
            if r >= 2 {
                self.publish(r - 2, frame);
            }
        }
    }

    fn publish(&mut self, r: usize, frame: &mut Frame) {
        frame.extend_rows(r * 16, (r + 1) * 16);
        self.shared.progress.set_done(((r + 1) * 16) as i32);
    }

    fn finish(&mut self, frame: &mut Frame, info: &PicInfo) {
        let mbw = info.mb_width;
        for r in *self.next_filter_row..info.mb_height {
            self.row_mbs[r] = mbw as u16;
            self.row_complete(r, frame, info);
            *self.next_filter_row = r + 1;
        }
        if info.mb_height > 0 {
            let last = info.mb_height - 1;
            if self.deblock {
                deblock_mb_rows(frame, info, self.slices, last, last + 1);
            }
            if last >= 1 {
                self.publish(last - 1, frame);
            }
            self.publish(last, frame);
        }
    }
}

/// The picture currently being fed (main-thread view).
struct Current {
    /// The first slice's header (picture-level facts).
    hdr: SliceHeader,
    sps: Sps,
    poc: i32,
    had_mmco5: bool,
    decode_index: u64,
    frame: Arc<SharedFrame>,
    /// A slice starting at MB 0 has been seen (repeated first slice = new picture).
    any_slice: bool,
    tx: Option<Sender<SliceJob>>,
    inline: Option<PictureDecoder>,
}

/// A native H.264 decoder.
///
/// Frame-threaded like [`crate::hevc::HevcDecoder`]: `push_nal` parses
/// headers and manages the DPB on the caller's thread and hands slices to
/// a worker per picture; [`H264Decoder::next_picture`] returns pictures in
/// output order, waiting for each to finish.
pub struct H264Decoder {
    sps: Vec<Option<Sps>>,
    pps: Vec<Option<Arc<Pps>>>,
    dpb: Dpb,
    poc_state: PocState,
    cur: Option<Current>,
    dequant_cache: Option<(u32, u32, Arc<Dequant>)>,
    decode_index: u64,
    /// A complete grey frame of the current size, for missing references.
    grey: Option<Arc<SharedFrame>>,
    output: VecDeque<Picture>,
    warnings: Arc<AtomicU64>,
    pool: Option<Arc<Pool>>,
    frames: FramePool,
    deblock: bool,
    next_id: u64,
    /// Geometry of the pictures in the DPB (macroblocks).
    dpb_dims: (usize, usize),
}

impl Default for H264Decoder {
    fn default() -> Self {
        Self::new()
    }
}

impl H264Decoder {
    /// A decoder with one worker per hardware thread (capped), or as
    /// `H26X_THREADS` says.
    pub fn new() -> Self {
        let n = std::env::var("H26X_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or_else(default_threads);
        Self::with_threads(n)
    }

    /// A decoder with `threads` workers; 0 or 1 decodes on the caller's thread.
    pub fn with_threads(threads: usize) -> Self {
        let pool = if threads > 1 { Some(Pool::new(threads, threads)) } else { None };
        H264Decoder {
            sps: (0..32).map(|_| None).collect(),
            pps: (0..256).map(|_| None).collect(),
            dpb: Dpb::new(),
            poc_state: PocState::default(),
            cur: None,
            dequant_cache: None,
            decode_index: 0,
            grey: None,
            output: VecDeque::new(),
            warnings: Arc::new(AtomicU64::new(0)),
            pool,
            frames: FramePool::new(),
            deblock: std::env::var_os("H26X_NO_DEBLOCK").is_none(),
            next_id: 1,
            dpb_dims: (0, 0),
        }
    }

    /// Non-fatal problems seen so far (concealed references, dropped slices).
    pub fn warnings(&self) -> u64 {
        self.warnings.load(Ordering::Relaxed)
    }

    /// Feed a chunk of Annex-B bytes (whole NAL units).
    pub fn push_annexb(&mut self, data: &[u8]) -> Result<()> {
        for nal in annexb_nals(data) {
            self.push_nal(nal)?;
        }
        Ok(())
    }

    /// Feed one NAL unit (with its header byte, without start code).
    pub fn push_nal(&mut self, nal: &[u8]) -> Result<()> {
        let Some(hdr) = H264NalHeader::parse(nal) else {
            return Ok(());
        };
        match hdr.unit_type {
            1 | 5 => {
                let rbsp = unescape_rbsp(nal);
                self.decode_slice(hdr, rbsp)
            }
            7 => {
                let rbsp = unescape_rbsp(&nal[1..]);
                let sps = Sps::parse(&rbsp)?;
                let id = sps.id as usize;
                if let Some(old) = &self.sps[id] {
                    if *old != sps && self.cur.is_some() {
                        self.finish_picture()?;
                    }
                }
                self.sps[id] = Some(sps);
                Ok(())
            }
            8 => {
                let rbsp = unescape_rbsp(&nal[1..]);
                let sps_tab = &self.sps;
                let pps = Pps::parse(&rbsp, &|id| sps_tab.get(id as usize).cloned().flatten())?;
                let id = pps.id as usize;
                if let Some(old) = &self.pps[id] {
                    if **old != pps && self.cur.is_some() {
                        self.finish_picture()?;
                    }
                }
                self.pps[id] = Some(Arc::new(pps));
                Ok(())
            }
            9 | 10 | 11 => {
                if self.cur.is_some() {
                    self.finish_picture()?;
                }
                Ok(())
            }
            2 | 3 | 4 => Err(Error::unsupported("H.264 slice data partitioning (nal_unit_type 2..4)")),
            20 | 21 => Ok(()),
            _ => Ok(()),
        }
    }

    /// End of stream: finish the open picture and drain the reorder buffer.
    pub fn flush(&mut self) -> Result<()> {
        if self.cur.is_some() {
            self.finish_picture()?;
        }
        self.dpb.flush_output();
        Ok(())
    }

    /// The next decoded picture in output order, if one is ready (waits for
    /// it to finish decoding).
    pub fn next_picture(&mut self) -> Option<Picture> {
        if let Some(p) = self.output.pop_front() {
            return Some(p);
        }
        self.dpb.output.pop_front().map(|p| p.into_picture())
    }

    /// The next picture in output order if it has finished decoding.
    pub fn try_next_picture(&mut self) -> Option<Picture> {
        if let Some(p) = self.output.pop_front() {
            return Some(p);
        }
        if self.dpb.output.front().is_some_and(|p| p.frame.progress.is_complete()) {
            return self.dpb.output.pop_front().map(|p| p.into_picture());
        }
        None
    }

    fn check_supported(sps: &Sps, pps: &Pps) -> Result<()> {
        if !sps.frame_mbs_only {
            return Err(Error::unsupported("H.264 interlaced coding (frame_mbs_only_flag = 0)"));
        }
        if sps.chroma_format_idc != 1 {
            return Err(Error::unsupported(format!("H.264 chroma_format_idc {} (only 4:2:0 is implemented)", sps.chroma_format_idc)));
        }
        if sps.bit_depth_luma != 8 || sps.bit_depth_chroma != 8 {
            return Err(Error::unsupported(format!("H.264 bit depth {}/{} (only 8-bit is implemented)", sps.bit_depth_luma, sps.bit_depth_chroma)));
        }
        if sps.transform_bypass {
            return Err(Error::unsupported("H.264 lossless transform bypass"));
        }
        if pps.num_slice_groups > 1 {
            return Err(Error::unsupported("H.264 slice groups (FMO)"));
        }
        Ok(())
    }

    /// Whether `hdr` starts a new picture relative to the open one (7.4.1.2.4).
    fn is_new_picture(cur: &Current, hdr: &SliceHeader, sps: &Sps) -> bool {
        let a = &cur.hdr;
        if hdr.first_mb_in_slice == 0 && !(a.frame_num == hdr.frame_num && a.pps_id == hdr.pps_id) {
            return true;
        }
        if a.frame_num != hdr.frame_num || a.pps_id != hdr.pps_id || a.field_pic != hdr.field_pic {
            return true;
        }
        if (a.nal_ref_idc == 0) != (hdr.nal_ref_idc == 0) {
            return true;
        }
        if sps.poc_type == 0 && (a.poc_lsb != hdr.poc_lsb || a.delta_poc_bottom != hdr.delta_poc_bottom) {
            return true;
        }
        if sps.poc_type == 1 && a.delta_poc != hdr.delta_poc {
            return true;
        }
        if a.is_idr() != hdr.is_idr() {
            return true;
        }
        if a.is_idr() && hdr.is_idr() && a.idr_pic_id != hdr.idr_pic_id {
            return true;
        }
        hdr.first_mb_in_slice == 0 && cur.any_slice
    }

    fn grey_frame(&mut self, mbw: usize, mbh: usize) -> Arc<SharedFrame> {
        // SAFETY: complete frames are read-only.
        let ok = self.grey.as_ref().is_some_and(|g| {
            let f = unsafe { g.get() };
            f.mb_width == mbw && f.mb_height == mbh
        });
        if !ok {
            let mut g = Frame::new(mbw, mbh, ChromaFormat::Yuv420);
            g.y.data.fill(128);
            g.cb.data.fill(128);
            g.cr.data.fill(128);
            g.mb_intra.fill(true);
            let id = self.next_id;
            self.next_id += 1;
            self.grey = Some(Arc::new(SharedFrame::new(g, 0, id, true)));
        }
        self.grey.clone().unwrap()
    }

    fn decode_slice(&mut self, nal: H264NalHeader, rbsp: Vec<u8>) -> Result<()> {
        let sps_tab = &self.sps;
        let pps_tab = &self.pps;
        let (hdr, pps, sps) = SliceHeader::parse(
            &rbsp,
            nal,
            &|id| pps_tab.get(id as usize).cloned().flatten().map(|p| (*p).clone()),
            &|id| sps_tab.get(id as usize).cloned().flatten(),
        )?;
        if hdr.redundant_pic_cnt > 0 {
            return Ok(());
        }
        if matches!(hdr.slice_type, SliceType::Sp | SliceType::Si) {
            return Err(Error::unsupported("H.264 SP/SI slices"));
        }
        Self::check_supported(&sps, &pps)?;
        let pps: Arc<Pps> = self.pps[pps.id as usize].clone().expect("parsed from the table");

        // Picture boundary.
        let new_pic = match &self.cur {
            None => true,
            Some(cur) => Self::is_new_picture(cur, &hdr, &sps),
        };
        if new_pic {
            if self.cur.is_some() {
                self.finish_picture()?;
            }
            self.start_picture(&hdr, &sps)?;
        }
        if self.cur.as_ref().is_some_and(|c| c.sps != sps) {
            return Err(Error::bitstream("slices of one picture reference different SPSs"));
        }

        // Dequantisation tables for this PPS/SPS pair.
        let cache_ok = matches!(&self.dequant_cache, Some((s, p, _)) if *s == sps.id && *p == pps.id);
        if !cache_ok {
            let lists: ScalingLists = pps.scaling_lists.clone().or_else(|| sps.scaling_lists.clone()).unwrap_or_else(ScalingLists::flat);
            self.dequant_cache = Some((sps.id, pps.id, Arc::new(Dequant::new(&lists))));
        }
        let dequant = self.dequant_cache.as_ref().unwrap().2.clone();

        // Reference lists, resolved to pictures.
        let cur_poc = self.cur.as_ref().unwrap().poc;
        let mut refs: [Vec<RefEntry>; 2] = [Vec::new(), Vec::new()];
        let mut col: Option<RefEntry> = None;
        if !hdr.slice_type.is_intra() {
            let rl = build_ref_lists(&mut self.dpb, &sps, &hdr, cur_poc)?;
            let mbw = sps.pic_width_in_mbs as usize;
            let mbh = sps.frame_height_in_mbs() as usize;
            let grey = self.grey_frame(mbw, mbh);
            for l in 0..2 {
                for &i in &rl.lists[l] {
                    if i == usize::MAX || i >= self.dpb.pics.len() {
                        self.warnings.fetch_add(1, Ordering::Relaxed);
                        refs[l].push(RefEntry { frame: grey.clone(), poc: i32::MIN / 2, long_term: false });
                    } else {
                        let p = &self.dpb.pics[i];
                        refs[l].push(RefEntry { frame: p.frame.clone(), poc: p.poc, long_term: p.mark == RefMark::Long });
                    }
                }
            }
            if hdr.slice_type.is_b() {
                if let Some(&i) = rl.lists[1].first() {
                    if i != usize::MAX && i < self.dpb.pics.len() {
                        let p = &self.dpb.pics[i];
                        col = Some(RefEntry { frame: p.frame.clone(), poc: p.poc, long_term: p.mark == RefMark::Long });
                    }
                }
            }
        }

        let cur = self.cur.as_mut().unwrap();
        if hdr.marking.ops.iter().any(|o| *o == Mmco::UnmarkAll) {
            cur.had_mmco5 = true;
        }
        if hdr.first_mb_in_slice == 0 {
            cur.any_slice = true;
        }
        let job = SliceJob { hdr, rbsp, pps, dequant, refs, col };
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

    fn start_picture(&mut self, hdr: &SliceHeader, sps: &Sps) -> Result<()> {
        let mbw = sps.pic_width_in_mbs as usize;
        let mbh = sps.frame_height_in_mbs() as usize;
        let size_changed = !self.dpb.pics.is_empty() && self.dpb_dims != (mbw, mbh);
        self.dpb_dims = (mbw, mbh);
        if hdr.is_idr() || size_changed {
            self.dpb.configure(sps);
            if size_changed {
                self.dpb.flush_output();
                self.dpb.clear();
                self.grey = None;
            }
        }
        if self.dpb.pics.is_empty() {
            self.dpb.configure(sps);
        }
        self.dpb.crop = sps.crop;

        // frame_num gap (8.2.5.2).
        if !hdr.is_idr() {
            let prev = self.poc_state.prev_ref_frame_num;
            let max = sps.max_frame_num();
            if hdr.frame_num != prev && hdr.frame_num != (prev + 1) % max {
                let grey = self.grey_frame(mbw, mbh);
                if !sps.gaps_in_frame_num_allowed {
                    self.warnings.fetch_add(1, Ordering::Relaxed);
                }
                self.dpb.fill_frame_num_gap(sps, prev, hdr.frame_num, &grey, &mut self.decode_index);
            }
        }

        // POC.
        let (top, bottom) = compute_poc(sps, hdr, &mut self.poc_state);
        let poc = top.min(bottom);

        let id = self.next_id;
        self.next_id += 1;
        // The buffers are attached here, before the picture can be seen by
        // anyone (a later picture reads a reference's geometry — the
        // colocated size check — before it waits on its rows).
        let mut f = self.frames.take(mbw, mbh, ChromaFormat::Yuv420);
        f.poc = poc;
        let shared = Arc::new(SharedFrame::with_pool(f, poc, id, self.frames.clone()));
        let pd = PictureDecoder {
            frame: shared.clone(),
            info: None,
            frames: self.frames.clone(),
            sps: sps.clone(),
            poc,
            slices: Vec::new(),
            row_mbs: vec![0; mbh],
            next_filter_row: 0,
            deblock: self.deblock,
            warnings: self.warnings.clone(),
        };
        let (tx, inline) = match &self.pool {
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
                (Some(tx), None)
            }
            None => (None, Some(pd)),
        };
        self.cur = Some(Current {
            hdr: hdr.clone(),
            sps: sps.clone(),
            poc,
            had_mmco5: false,
            decode_index: self.decode_index,
            frame: shared,
            any_slice: false,
            tx,
            inline,
        });
        self.decode_index += 1;
        Ok(())
    }

    fn finish_picture(&mut self) -> Result<()> {
        let Some(cur) = self.cur.take() else { return Ok(()) };
        // End the worker's slice loop (or finish inline).
        drop(cur.tx);
        if let Some(pd) = cur.inline {
            pd.finish();
        }
        let sps = cur.sps;
        let hdr = cur.hdr;

        // POC / frame_num bookkeeping.
        let mut poc = cur.poc;
        let mut frame_num = hdr.frame_num;
        if cur.had_mmco5 {
            poc = 0;
            frame_num = 0;
        }
        self.poc_state.prev_frame_num = hdr.frame_num;
        self.poc_state.prev_had_mmco5 = cur.had_mmco5;
        if hdr.is_reference() {
            self.poc_state.prev_ref_frame_num = frame_num;
            if cur.had_mmco5 {
                self.poc_state.prev_msb = 0;
                self.poc_state.prev_lsb = 0;
            }
        }
        if cur.had_mmco5 {
            self.poc_state.prev_frame_num_offset = 0;
        }

        let pic = DecodedPic {
            frame: cur.frame,
            poc,
            frame_num,
            frame_num_wrap: frame_num as i32,
            long_term_frame_idx: 0,
            mark: RefMark::Unused,
            needed_for_output: true,
            non_existing: false,
            decode_index: cur.decode_index,
        };
        self.dpb.store(pic, &hdr, &sps, cur.had_mmco5)?;
        Ok(())
    }
}

impl Drop for H264Decoder {
    fn drop(&mut self) {
        let _ = self.finish_picture();
        if let Some(p) = &self.pool {
            p.wait_idle();
        }
    }
}
