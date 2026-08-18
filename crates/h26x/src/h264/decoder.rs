//! The H.264 decoder: NAL dispatch, access-unit boundaries, POC and
//! reference marking on the caller's thread; slice decoding (CAVLC or
//! CABAC), row-pipelined deblocking and progress publication on a worker
//! per picture (frame threading, see [`crate::threading`]).

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::Arc;

use crate::bitreader::BitReader;
use crate::dsp::h264::H264Dsp;
use crate::cabac::Cabac;
use crate::nal::{H264NalHeader, annexb_nals, unescape_rbsp};
use crate::picture::{ChromaFormat, Picture};
use crate::threading::{Pool, default_threads};
use crate::{Error, Result};

use super::cabac_mb::{CabacState, decode_end_of_slice, decode_mb_skip, parse_mb_cabac};
use super::cavlc::parse_mb_cavlc;
use super::deblock::{DeblockParams, deblock_mb_rows};
use super::dpb::{DecodedPic, Dpb, PocState, RefEntry, RefMark, build_ref_lists, compute_poc};
#[allow(unused_imports)]
use super::dpb::MISSING_REF;
use super::frame::{Frame, FramePool, PARITY_FRAME, SharedFrame};
use super::mb::{InfoPool, MbKind, MbLayer, MbNeighbours, PicInfo, SliceCtx};
use super::pps::Pps;
use super::recon::{QpState, SliceRefs, reconstruct};
use super::slice::{Mmco, SliceHeader, SliceType};
use super::sps::{ScalingLists, Sps};
use crate::sample::Sample;
use super::transform::Dequant;

/// One slice handed to the picture's decoder, with its references resolved.
struct SliceJob<S: Sample> {
    hdr: SliceHeader,
    rbsp: Vec<u8>,
    pps: Arc<Pps>,
    dequant: Arc<Dequant>,
    /// The stream was made by x264 before build 151 (its user-data SEI
    /// says so): its 4:4:4 CABAC 8x8 coded_block_flag contexts followed a
    /// bug, reproduced on decode like libavcodec does.
    x264_old_444: bool,
    /// RefPicList0/1 (grey stand-ins already substituted).
    refs: [Vec<RefEntry<S>>; 2],
    /// RefPicList1[0] for direct prediction.
    col: Option<RefEntry<S>>,
}

/// The state of one picture's decoding: lives on the worker (or inline).
struct PictureDecoder<S: Sample> {
    frame: Arc<SharedFrame<S>>,
    /// Which picture this is: [`PARITY_FRAME`], or a field's parity.
    parity: u8,
    /// A field picture decodes into this half-height frame; its rows are
    /// interleaved into `frame` as they are published, its motion copied
    /// into `frame`'s frame-row layout as rows complete.
    field: Option<Frame<S>>,
    frames: FramePool<S>,
    info: Option<PicInfo>,
    /// The Cb / Cr colour planes' macroblock info when the picture is coded
    /// as separate colour planes (each plane is its own monochrome decode).
    plane_infos: [Option<PicInfo>; 2],
    separate_planes: bool,
    infos: InfoPool,
    sps: Sps,
    poc: i32,
    slices: Vec<DeblockParams>,
    /// Decoded macroblocks per MB row (a row is complete at `mb_width`).
    row_mbs: Vec<u16>,
    /// The next MB row whose completion has not been acted on.
    next_filter_row: usize,
    deblock: bool,
    warnings: Arc<AtomicU64>,
    dsp: H264Dsp<S>,
}

impl<S: Sample> PictureDecoder<S> {
    fn ensure_buffers(&mut self) {
        if self.info.is_none() {
            let mbw = self.sps.pic_width_in_mbs as usize;
            let mbh = self.sps.frame_height_in_mbs() as usize / if self.parity == PARITY_FRAME { 1 } else { 2 };
            self.info = Some(self.infos.take(mbw, mbh));
            if self.separate_planes {
                self.plane_infos = [Some(self.infos.take(mbw, mbh)), Some(self.infos.take(mbw, mbh))];
            }
        }
    }

    fn decode_slice(&mut self, job: SliceJob<S>) -> Result<()> {
        self.ensure_buffers();
        let SliceJob { hdr, rbsp, pps, dequant, refs: ref_lists, col, x264_old_444 } = job;
        // SAFETY: this PictureDecoder is the picture's only writer; other
        // threads read only rows below the published progress. The Arc does
        // not move while `self` is borrowed.
        let shared: &SharedFrame<S> = unsafe { &*(Arc::as_ptr(&self.frame)) };
        let PictureDecoder { info, plane_infos, separate_planes, sps, poc, slices, row_mbs, next_filter_row, deblock, warnings, dsp, field, parity, .. } = self;
        let parity = *parity;
        // A field picture works in its own half-height frame.
        let cur_main: &mut Frame<S> = match field {
            Some(f) => f,
            None => unsafe { shared.get_mut() },
        };
        let separate = *separate_planes;
        // The colour plane this slice decodes (8.1: separate colour planes
        // are three monochrome decodes; the plane picks its frame, its
        // macroblock info and its references' plane).
        let plane = if separate { hdr.colour_plane_id as usize } else { 0 };
        let n_planes = if separate { 3 } else { 1 };
        let cur_poc = *poc;

        // SAFETY: reference reads wait on progress.
        let mut refs = SliceRefs {
            frames: [
                ref_lists[0].iter().map(|e| unsafe { e.frame.get() }.plane_frame(plane)).collect(),
                ref_lists[1].iter().map(|e| unsafe { e.frame.get() }.plane_frame(plane)).collect(),
            ],
            shared: [ref_lists[0].iter().map(|e| &*e.frame).collect(), ref_lists[1].iter().map(|e| &*e.frame).collect()],
            col_shared: col.as_ref().map(|e| &*e.frame),
            col_parity: col.as_ref().map_or(PARITY_FRAME, |e| e.parity),
            pocs: [ref_lists[0].iter().map(|e| e.poc).collect(), ref_lists[1].iter().map(|e| e.poc).collect()],
            long_term: [ref_lists[0].iter().map(|e| e.long_term).collect(), ref_lists[1].iter().map(|e| e.long_term).collect()],
            ids: [ref_lists[0].iter().map(|e| e.frame.id as u16).collect(), ref_lists[1].iter().map(|e| e.frame.id as u16).collect()],
            parity: [ref_lists[0].iter().map(|e| e.parity).collect(), ref_lists[1].iter().map(|e| e.parity).collect()],
            col: col.as_ref().map(|e| unsafe { e.frame.get() }.plane_frame(plane)),
            col_long_term: col.as_ref().is_some_and(|e| e.long_term),
            explicit: hdr.pred_weights.as_ref(),
            implicit: None,
            cur_poc,
            cur_parity: parity,
            dsp: *dsp,
            bit_depth: sps.bit_depth_luma,
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
            // ChromaArrayType: 0 for a colour plane coded on its own.
            chroma_format_idc: if separate { 0 } else { sps.chroma_format_idc },
            cabac: pps.cabac,
            bit_depth: sps.bit_depth_luma,
            transform_bypass: sps.transform_bypass,
            scaling_plane: plane,
            x264_old_444,
            field_pic: hdr.field_pic,
        };
        let mut qps = QpState { prev_qp: hdr.slice_qp, chroma_offset: [pps.chroma_qp_index_offset, pps.second_chroma_qp_index_offset] };
        let total_mbs = cur_main.mb_width * cur_main.mb_height;
        let mbw = cur_main.mb_width;
        let mut addr = hdr.first_mb_in_slice as usize;
        if addr >= total_mbs {
            return Err(Error::bitstream("first_mb_in_slice beyond the picture"));
        }
        let data_start = (hdr.data_bit_offset / 8) as usize;
        let mut filters = RowFilters { row_mbs, next_filter_row, deblock: *deblock, shared, slices, dsp, parity };
        let dq: &Dequant = &dequant;

        // The macroblock info of this slice's plane (a fresh borrow each
        // time, so the row filters can read every plane's in between).
        macro_rules! info {
            () => {
                (if plane == 0 { info.as_mut() } else { plane_infos[plane - 1].as_mut() }).expect("buffers ensured")
            };
        }
        // Decode one macroblock and account for it.
        macro_rules! mb_done {
            () => {{
                let all: [&PicInfo; 3] = [
                    info.as_ref().expect("buffers ensured"),
                    plane_infos[0].as_ref().unwrap_or(info.as_ref().expect("buffers ensured")),
                    plane_infos[1].as_ref().unwrap_or(info.as_ref().expect("buffers ensured")),
                ];
                filters.mb_done(addr, cur_main, &all[..n_planes]);
                addr += 1;
            }};
        }

        // One macroblock layer for the whole slice, reset per macroblock.
        let mut layer = MbLayer::new(MbKind::I4x4);
        if pps.cabac {
            let mut cabac = Cabac::new(&rbsp[data_start..]);
            let mut st = CabacState::new(hdr.slice_type, hdr.cabac_init_idc, hdr.slice_qp);
            loop {
                if addr >= total_mbs {
                    return Err(Error::bitstream("slice data runs past the picture"));
                }
                let info: &mut PicInfo = info!();
                let cur: &mut Frame<S> = cur_main.plane_frame_mut(plane);
                let nb = MbNeighbours::derive(info, addr, slice_num);
                let mut skipped = false;
                if !hdr.slice_type.is_intra() {
                    let skip = decode_mb_skip(&mut cabac, &mut st, info, &nb, hdr.slice_type.is_b());
                    if skip {
                        let kind = if hdr.slice_type.is_b() { MbKind::BSkip } else { MbKind::PSkip };
                        layer.reset(kind, true);
                        st.prev_qp_delta_nonzero = false;
                        skipped = true;
                    }
                }
                if !skipped {
                    parse_mb_cabac(&mut cabac, &mut st, &ctx, info, &nb, &cur.motion, &mut layer)?;
                }
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
                        let info: &mut PicInfo = info!();
                        let cur: &mut Frame<S> = cur_main.plane_frame_mut(plane);
                        let nb = MbNeighbours::derive(info, addr, slice_num);
                        let kind = if hdr.slice_type.is_b() { MbKind::BSkip } else { MbKind::PSkip };
                        layer.reset(kind, false);
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
                let info: &mut PicInfo = info!();
                let cur: &mut Frame<S> = cur_main.plane_frame_mut(plane);
                let nb = MbNeighbours::derive(info, addr, slice_num);
                let t = r.ue();
                parse_mb_cavlc(&mut r, &ctx, info, &nb, t, &mut layer)?;
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
        let shared: &SharedFrame<S> = unsafe { &*(Arc::as_ptr(&self.frame)) };
        let PictureDecoder { info, plane_infos, separate_planes, poc, slices, row_mbs, next_filter_row, deblock, warnings, dsp, field, parity, .. } = &mut self;
        let parity = *parity;
        let cur: &mut Frame<S> = match field {
            Some(f) => f,
            None => unsafe { shared.get_mut() },
        };
        let info: &PicInfo = info.as_ref().expect("buffers ensured");
        let all: [&PicInfo; 3] = [info, plane_infos[0].as_ref().unwrap_or(info), plane_infos[1].as_ref().unwrap_or(info)];
        let n_planes = if *separate_planes { 3 } else { 1 };
        let missing = all[..n_planes].iter().map(|i| i.mbs.iter().filter(|m| !m.decoded).count()).sum::<usize>();
        if missing > 0 {
            warnings.fetch_add(1, Ordering::Relaxed);
        }
        let mut filters = RowFilters { row_mbs, next_filter_row, deblock: *deblock, shared, slices, dsp, parity };
        filters.finish(cur, &all[..n_planes]);
        cur.poc = *poc;
        shared.finish(parity);
        if let Some(info) = self.info.take() {
            self.infos.give(info);
        }
        for pi in self.plane_infos.iter_mut() {
            if let Some(info) = pi.take() {
                self.infos.give(info);
            }
        }
        if let Some(f) = self.field.take() {
            self.frames.give(f);
        }
    }
}

/// Row-pipelined deblocking of one picture: MB row `r` is filtered once row
/// `r + 1` is fully decoded (intra prediction reads unfiltered neighbours),
/// which settles row `r - 1` (its bottom lines are touched by row `r`'s top
/// edges); row `r - 1` is then edge-extended and published.
struct RowFilters<'a, S: Sample> {
    row_mbs: &'a mut Vec<u16>,
    next_filter_row: &'a mut usize,
    deblock: bool,
    shared: &'a SharedFrame<S>,
    slices: &'a Vec<DeblockParams>,
    dsp: &'a H264Dsp<S>,
    /// [`PARITY_FRAME`], or the field being decoded: its rows are then
    /// interleaved into the shared frame as they publish, and its progress
    /// counts frame rows (two per field row).
    parity: u8,
}

impl<S: Sample> RowFilters<'_, S> {
    /// Frame rows covered once macroblock row `r` of this picture is done.
    #[inline]
    fn frame_rows(&self, r: usize) -> i32 {
        (((r + 1) * 16) * if self.parity == PARITY_FRAME { 1 } else { 2 }) as i32
    }

    /// `infos` is the macroblock info per colour plane (one entry, or three
    /// for separate colour planes — a row is then complete once all three
    /// planes have decoded it, and each plane is filtered as its own frame).
    fn mb_done(&mut self, addr: usize, frame: &mut Frame<S>, infos: &[&PicInfo]) {
        let info = infos[0];
        let mbw = info.mb_width;
        let r = addr / mbw;
        self.row_mbs[r] += 1;
        let target = mbw * infos.len();
        while *self.next_filter_row < info.mb_height && self.row_mbs[*self.next_filter_row] as usize >= target {
            let row = *self.next_filter_row;
            self.row_complete(row, frame, infos);
            *self.next_filter_row += 1;
        }
    }

    fn row_complete(&mut self, r: usize, frame: &mut Frame<S>, infos: &[&PicInfo]) {
        if self.parity != PARITY_FRAME {
            // The field's motion goes into the shared frame's frame-row
            // layout before the row is announced (colocated readers).
            // SAFETY: this decoder owns the rows of its parity.
            let dst: &mut Frame<S> = unsafe { self.shared.get_mut() };
            dst.take_field_motion_row(frame, r, self.parity as usize);
        }
        self.shared.set_decoded(self.parity, self.frame_rows(r));
        if r >= 1 {
            if self.deblock {
                for (k, info) in infos.iter().enumerate() {
                    deblock_mb_rows(self.dsp, frame.plane_frame_mut(k), info, self.slices, r - 1, r);
                }
            }
            if r >= 2 {
                self.publish(r - 2, frame);
            }
        }
    }

    fn publish(&mut self, r: usize, frame: &mut Frame<S>) {
        if self.parity == PARITY_FRAME {
            frame.extend_rows(r * 16, (r + 1) * 16);
        } else {
            // Interleave the field's rows into the frame and extend that
            // field's borders there.
            // SAFETY: this decoder owns the rows of its parity.
            let dst: &mut Frame<S> = unsafe { self.shared.get_mut() };
            dst.interleave_field_rows(frame, r * 16, (r + 1) * 16, self.parity as usize);
            dst.extend_rows_parity(r * 32, (r + 1) * 32, self.parity as usize);
        }
        self.shared.set_done(self.parity, self.frame_rows(r));
    }

    fn finish(&mut self, frame: &mut Frame<S>, infos: &[&PicInfo]) {
        let info = infos[0];
        let mbw = info.mb_width;
        for r in *self.next_filter_row..info.mb_height {
            self.row_mbs[r] = (mbw * infos.len()) as u16;
            self.row_complete(r, frame, infos);
            *self.next_filter_row = r + 1;
        }
        if info.mb_height > 0 {
            let last = info.mb_height - 1;
            if self.deblock {
                for (k, info) in infos.iter().enumerate() {
                    deblock_mb_rows(self.dsp, frame.plane_frame_mut(k), info, self.slices, last, last + 1);
                }
            }
            if last >= 1 {
                self.publish(last - 1, frame);
            }
            self.publish(last, frame);
        }
    }
}

/// A first field waiting for its second (main-thread view).
struct OpenField<S: Sample> {
    frame: Arc<SharedFrame<S>>,
    parity: u8,
    frame_num: u32,
    decode_index: u64,
}

/// The picture currently being fed (main-thread view).
struct Current<S: Sample> {
    /// The first slice's header (picture-level facts).
    hdr: SliceHeader,
    sps: Sps,
    poc: i32,
    had_mmco5: bool,
    decode_index: u64,
    frame: Arc<SharedFrame<S>>,
    /// [`PARITY_FRAME`], or the field's parity.
    parity: u8,
    /// Colour planes (bit `colour_plane_id`; bit 0 alone without separate
    /// planes) whose slice starting at MB 0 has been seen — a repeated first
    /// slice of a plane is a new picture.
    planes_started: u8,
    tx: Option<Sender<SliceJob<S>>>,
    inline: Option<PictureDecoder<S>>,
}

/// The decoder for one sample type (see [`H264Decoder`], which picks it).
pub(crate) struct H264DecoderImpl<S: Sample> {
    sps: Vec<Option<Sps>>,
    pps: Vec<Option<Arc<Pps>>>,
    dpb: Dpb<S>,
    poc_state: PocState,
    cur: Option<Current<S>>,
    dequant_cache: Option<(u32, u32, Arc<Dequant>)>,
    decode_index: u64,
    /// A complete grey frame of the current size, for missing references.
    grey: Option<Arc<SharedFrame<S>>>,
    output: VecDeque<Picture>,
    warnings: Arc<AtomicU64>,
    pool: Option<Arc<Pool>>,
    frames: FramePool<S>,
    infos: InfoPool,
    deblock: bool,
    next_id: u64,
    /// Geometry of the pictures in the DPB (macroblocks).
    dpb_dims: (usize, usize),
    dsp: H264Dsp<S>,
    /// The x264 build number from its user-data-unregistered SEI, when seen.
    x264_build: Option<u32>,
    /// A decoded first field whose second field may follow.
    open_field: Option<OpenField<S>>,
}

impl<S: Sample> H264DecoderImpl<S> {
    /// A decoder with `threads` workers; 0 or 1 decodes on the caller's thread.
    pub fn with_threads(threads: usize) -> Self {
        let pool = if threads > 1 { Some(Pool::new(threads, threads)) } else { None };
        H264DecoderImpl {
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
            infos: InfoPool::default(),
            deblock: std::env::var_os("H26X_NO_DEBLOCK").is_none(),
            next_id: 1,
            dpb_dims: (0, 0),
            x264_build: None,
            open_field: None,
            dsp: H264Dsp::new(crate::dsp::Cpu::detect_honouring_env()),
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
                // The tables derive from the parameter sets' scaling lists,
                // which a re-sent set may change without changing its id.
                self.dequant_cache = None;
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
                self.dequant_cache = None;
                Ok(())
            }
            6 => {
                self.parse_sei(&unescape_rbsp(&nal[1..]));
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

    /// SEI messages: only `user_data_unregistered` from x264 matters (its
    /// build number selects a decoding quirk); everything else is skipped.
    fn parse_sei(&mut self, rbsp: &[u8]) {
        const X264_UUID: [u8; 16] = [0xdc, 0x45, 0xe9, 0xbd, 0xe6, 0xd9, 0x48, 0xb7, 0x96, 0x2c, 0xd8, 0x20, 0xd9, 0x23, 0xee, 0xef];
        let mut i = 0usize;
        while i < rbsp.len() && rbsp[i] != 0x80 {
            let mut ptype = 0usize;
            while i < rbsp.len() && rbsp[i] == 0xff {
                ptype += 255;
                i += 1;
            }
            if i >= rbsp.len() {
                return;
            }
            ptype += rbsp[i] as usize;
            i += 1;
            let mut size = 0usize;
            while i < rbsp.len() && rbsp[i] == 0xff {
                size += 255;
                i += 1;
            }
            if i >= rbsp.len() {
                return;
            }
            size += rbsp[i] as usize;
            i += 1;
            let end = (i + size).min(rbsp.len());
            let payload = &rbsp[i..end];
            if ptype == 5 && payload.len() > 16 && payload[..16] == X264_UUID {
                // "x264 - core NNN ..."
                let text = &payload[16..];
                if let Some(rest) = text.strip_prefix(b"x264 - core ") {
                    let digits: Vec<u8> = rest.iter().copied().take_while(|c| c.is_ascii_digit()).collect();
                    if let Ok(n) = std::str::from_utf8(&digits).unwrap_or("").parse::<u32>() {
                        self.x264_build = Some(n);
                    }
                }
            }
            i = end;
        }
    }

    /// End of stream: finish the open picture and drain the reorder buffer.
    pub fn flush(&mut self) -> Result<()> {
        if self.cur.is_some() {
            self.finish_picture()?;
        }
        if let Some(of) = self.open_field.take() {
            self.close_unpaired(of);
        }
        self.dpb.flush_output();
        Ok(())
    }

    /// A first field never got its second: give the frame a second field
    /// (a copy of the first) once the first is decoded, and let it be output.
    fn close_unpaired(&mut self, of: OpenField<S>) {
        self.warnings.fetch_add(1, Ordering::Relaxed);
        let frame = of.frame.clone();
        let present = of.parity as usize;
        let fill = move || {
            frame.progress[present].wait_complete();
            // SAFETY: the decoded field is complete; nobody writes the frame.
            let f: &mut Frame<S> = unsafe { frame.get_mut() };
            f.double_field(present);
            frame.finish(1 - present as u8);
        };
        match &self.pool {
            Some(pool) => pool.spawn(Box::new(fill)),
            None => fill(),
        }
        self.dpb.close_unpaired(&of.frame);
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
        if self.dpb.output.front().is_some_and(|p| p.frame.is_complete()) {
            return self.dpb.output.pop_front().map(|p| p.into_picture());
        }
        None
    }

    fn check_supported(sps: &Sps, pps: &Pps) -> Result<()> {
        if sps.mb_adaptive_frame_field {
            return Err(Error::unsupported("H.264 macroblock-adaptive frame/field coding (MBAFF)"));
        }
        if sps.bit_depth_luma != sps.bit_depth_chroma {
            return Err(Error::unsupported(format!("H.264 different luma / chroma bit depths ({} / {})", sps.bit_depth_luma, sps.bit_depth_chroma)));
        }
        if sps.bit_depth_luma > 14 {
            return Err(Error::unsupported(format!("H.264 bit depth {}", sps.bit_depth_luma)));
        }
        if pps.num_slice_groups > 1 {
            return Err(Error::unsupported("H.264 slice groups (FMO)"));
        }
        Ok(())
    }

    /// Whether `hdr` starts a new picture relative to the open one (7.4.1.2.4).
    fn is_new_picture(cur: &Current<S>, hdr: &SliceHeader, sps: &Sps) -> bool {
        let a = &cur.hdr;
        if hdr.first_mb_in_slice == 0 && !(a.frame_num == hdr.frame_num && a.pps_id == hdr.pps_id) {
            return true;
        }
        if a.frame_num != hdr.frame_num || a.pps_id != hdr.pps_id || a.field_pic != hdr.field_pic || a.bottom_field != hdr.bottom_field {
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
        hdr.first_mb_in_slice == 0 && cur.planes_started & (1 << hdr.colour_plane_id) != 0
    }

    fn grey_frame(&mut self, mbw: usize, mbh: usize, chroma: ChromaFormat, bit_depth: u32, separate: bool) -> Arc<SharedFrame<S>> {
        // SAFETY: complete frames are read-only.
        let ok = self.grey.as_ref().is_some_and(|g| {
            let f = unsafe { g.get() };
            f.mb_width == mbw && f.mb_height == mbh && f.chroma == chroma && f.bit_depth == bit_depth && f.colour_planes.is_some() == separate
        });
        if !ok {
            let mut g = Frame::new(mbw, mbh, chroma, bit_depth, separate);
            let mid = S::from_i32(1 << (bit_depth - 1));
            g.y.data.fill(mid);
            g.cb.data.fill(mid);
            g.cr.data.fill(mid);
            g.mb_intra.fill(true);
            if let Some(planes) = &mut g.colour_planes {
                for f in planes.iter_mut() {
                    f.y.data.fill(mid);
                    f.mb_intra.fill(true);
                }
            }
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
        let (cur_poc, cur_parity) = {
            let c = self.cur.as_ref().unwrap();
            (c.poc, c.parity)
        };
        let mut refs: [Vec<RefEntry<S>>; 2] = [Vec::new(), Vec::new()];
        let mut col: Option<RefEntry<S>> = None;
        if !hdr.slice_type.is_intra() {
            let rl = build_ref_lists(&mut self.dpb, &sps, &hdr, cur_poc, cur_parity)?;
            let mbw = sps.pic_width_in_mbs as usize;
            let mbh = sps.frame_height_in_mbs() as usize;
            let grey = self.grey_frame(mbw, mbh, sps.frame_chroma(), sps.bit_depth_luma, sps.separate_colour_plane);
            let entry = |dpb: &Dpb<S>, e: (usize, u8)| -> Option<RefEntry<S>> {
                let (i, par) = e;
                if i == usize::MAX || i >= dpb.pics.len() {
                    return None;
                }
                let p = &dpb.pics[i];
                Some(RefEntry { frame: p.frame.clone(), poc: p.poc_of(par), long_term: p.mark_of(par) == RefMark::Long, parity: par })
            };
            for l in 0..2 {
                for &e in &rl.lists[l] {
                    match entry(&self.dpb, e) {
                        Some(r) => refs[l].push(r),
                        None => {
                            self.warnings.fetch_add(1, Ordering::Relaxed);
                            refs[l].push(RefEntry { frame: grey.clone(), poc: i32::MIN / 2, long_term: false, parity: cur_parity });
                        }
                    }
                }
            }
            if hdr.slice_type.is_b() {
                if let Some(&e) = rl.lists[1].first() {
                    col = entry(&self.dpb, e);
                }
            }
        }

        let cur = self.cur.as_mut().unwrap();
        if hdr.marking.ops.iter().any(|o| *o == Mmco::UnmarkAll) {
            cur.had_mmco5 = true;
        }
        if hdr.first_mb_in_slice == 0 {
            cur.planes_started |= 1 << hdr.colour_plane_id;
        }
        let x264_old_444 = self.x264_build.is_some_and(|b| b < 151);
        let job = SliceJob { hdr, rbsp, pps, dequant, refs, col, x264_old_444 };
        if let Some(tx) = &cur.tx {
            let _ = tx.send(job);
        } else if let Some(pd) = cur.inline.as_mut() {
            if let Err(e) = pd.decode_slice(job) {
                pd.frame.set_error();
                self.warnings.fetch_add(1, Ordering::Relaxed);
                return Err(e);
            }
        }
        Ok(())
    }

    fn start_picture(&mut self, hdr: &SliceHeader, sps: &Sps) -> Result<()> {
        let mbw = sps.pic_width_in_mbs as usize;
        let mbh = sps.frame_height_in_mbs() as usize;
        let field_pic = hdr.field_pic;
        let parity = if field_pic { hdr.bottom_field as u8 } else { PARITY_FRAME };
        // A field: the second of a complementary pair when the previous
        // picture was a first field of the opposite parity with the same
        // frame_num (3.35 / 7.4.1.2.4); else any open first field stays
        // unpaired.
        let mut second: Option<OpenField<S>> = None;
        if let Some(of) = self.open_field.take() {
            if field_pic && of.parity != parity && of.frame_num == hdr.frame_num && !hdr.is_idr() {
                second = Some(of);
            } else {
                self.close_unpaired(of);
            }
        }
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

        // frame_num gap (8.2.5.2) — not between the fields of a pair.
        if !hdr.is_idr() && second.is_none() {
            let prev = self.poc_state.prev_ref_frame_num;
            let max = sps.max_frame_num();
            if hdr.frame_num != prev && hdr.frame_num != (prev + 1) % max {
                let grey = self.grey_frame(mbw, mbh, sps.frame_chroma(), sps.bit_depth_luma, sps.separate_colour_plane);
                if !sps.gaps_in_frame_num_allowed {
                    self.warnings.fetch_add(1, Ordering::Relaxed);
                }
                self.dpb.fill_frame_num_gap(sps, prev, hdr.frame_num, &grey, &mut self.decode_index);
            }
        }

        // POC.
        let (top, bottom) = compute_poc(sps, hdr, &mut self.poc_state);
        let poc = match parity {
            0 => top,
            1 => bottom,
            _ => top.min(bottom),
        };

        // The frame buffer: the first field's for a second field, else a
        // fresh one. The buffers are attached here, before the picture can
        // be seen by anyone (a later picture reads a reference's geometry —
        // the colocated size check — before it waits on its rows).
        let (shared, decode_index) = match &second {
            Some(of) => {
                // SAFETY: only this thread touches these fields; the first
                // field's worker never reads them.
                let f: &mut Frame<S> = unsafe { of.frame.get_mut() };
                f.field_poc[parity as usize] = poc;
                (of.frame.clone(), of.decode_index)
            }
            None => {
                let id = self.next_id;
                self.next_id += 1;
                let mut f = self.frames.take(mbw, mbh, sps.frame_chroma(), sps.bit_depth_luma, sps.separate_colour_plane);
                f.poc = poc;
                f.field_borders = !sps.frame_mbs_only;
                f.field_coded = field_pic;
                if field_pic {
                    f.field_poc[parity as usize] = poc;
                } else {
                    f.field_poc = [top, bottom];
                }
                (Arc::new(SharedFrame::with_pool(f, poc, id, self.frames.clone())), self.decode_index)
            }
        };
        // A field decodes into a half-height working frame of its own.
        let field = if field_pic {
            let mut f = self.frames.take(mbw, mbh / 2, sps.frame_chroma(), sps.bit_depth_luma, sps.separate_colour_plane);
            f.field_coded = true;
            f.poc = poc;
            Some(f)
        } else {
            None
        };
        let pic_mbh = if field_pic { mbh / 2 } else { mbh };
        let pd = PictureDecoder {
            frame: shared.clone(),
            parity,
            field,
            frames: self.frames.clone(),
            info: None,
            plane_infos: [None, None],
            separate_planes: sps.separate_colour_plane,
            infos: self.infos.clone(),
            sps: sps.clone(),
            poc,
            slices: Vec::new(),
            row_mbs: vec![0; pic_mbh],
            next_filter_row: 0,
            deblock: self.deblock,
            warnings: self.warnings.clone(),
            dsp: self.dsp,
        };
        let (tx, inline) = match &self.pool {
            Some(pool) => {
                let (tx, rx): (Sender<SliceJob<S>>, Receiver<SliceJob<S>>) = channel();
                let mut pd = pd;
                pool.submit(Box::new(move || {
                    for job in rx {
                        if pd.decode_slice(job).is_err() {
                            pd.frame.set_error();
                            pd.warnings.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                    pd.finish();
                }));
                (Some(tx), None)
            }
            None => (None, Some(pd)),
        };
        if field_pic && second.is_none() {
            self.open_field = Some(OpenField { frame: shared.clone(), parity, frame_num: hdr.frame_num, decode_index });
        }
        self.cur = Some(Current {
            hdr: hdr.clone(),
            sps: sps.clone(),
            poc,
            had_mmco5: false,
            decode_index,
            frame: shared,
            parity,
            planes_started: 0,
            tx,
            inline,
        });
        if second.is_none() {
            self.decode_index += 1;
        }
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
        // A picture with an MMCO 5 is treated as having had frame_num 0
        // afterwards (7.4.3), for the wrap check of the next FrameNumOffset.
        self.poc_state.prev_frame_num = frame_num;
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

        let parity = cur.parity;
        let field_poc = match parity {
            0 => [poc, i32::MAX],
            1 => [i32::MAX, poc],
            _ => {
                // SAFETY: the frame's POCs were written before decoding started.
                let f = unsafe { cur.frame.get() };
                if cur.had_mmco5 { [poc, poc] } else { f.field_poc }
            }
        };
        let pic = DecodedPic {
            frame: cur.frame,
            poc,
            field_poc,
            fields: if parity == PARITY_FRAME { 3 } else { 1 << parity },
            frame_num,
            frame_num_wrap: frame_num as i32,
            long_term_frame_idx: 0,
            mark: [RefMark::Unused; 2],
            needed_for_output: true,
            awaiting_field: false,
            non_existing: false,
            decode_index: cur.decode_index,
        };
        if cur.had_mmco5 {
            if let Some(of) = &mut self.open_field {
                of.frame_num = 0;
            }
        }
        self.dpb.store(pic, &hdr, &sps, cur.had_mmco5, parity)?;
        Ok(())
    }
}

impl<S: Sample> Drop for H264DecoderImpl<S> {
    fn drop(&mut self) {
        let _ = self.finish_picture();
        if let Some(p) = &self.pool {
            p.wait_idle();
        }
    }
}

// ----------------------------------------------------------------------
// The public decoder: picks the sample type from the SPS
// ----------------------------------------------------------------------

/// A native H.264 decoder.
///
/// Frame-threaded like [`crate::hevc::HevcDecoder`]: `push_nal` parses
/// headers and manages the DPB on the caller's thread and hands slices to
/// a worker per picture; [`H264Decoder::next_picture`] returns pictures in
/// output order, waiting for each to finish. 8-bit streams decode into
/// 8-bit planes, deeper ones into 16-bit planes; the implementation for the
/// sample type is created at the first SPS.
pub struct H264Decoder {
    threads: usize,
    inner: Option<Inner>,
    /// NAL units seen before the sample type was known (SEI, AUD...),
    /// replayed into the implementation once it exists.
    pending: Vec<Vec<u8>>,
    /// Output left over from an implementation replaced mid-stream.
    leftover: VecDeque<Picture>,
    warnings_before: u64,
}

enum Inner {
    U8(H264DecoderImpl<u8>),
    U16(H264DecoderImpl<u16>),
}

macro_rules! with_inner {
    ($self:expr, $d:ident => $body:expr) => {
        match $self {
            Inner::U8($d) => $body,
            Inner::U16($d) => $body,
        }
    };
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
        H264Decoder { threads, inner: None, pending: Vec::new(), leftover: VecDeque::new(), warnings_before: 0 }
    }

    /// Non-fatal problems seen so far (concealed references, dropped slices).
    pub fn warnings(&self) -> u64 {
        self.warnings_before + self.inner.as_ref().map_or(0, |i| with_inner!(i, d => d.warnings()))
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
        if hdr.unit_type == 7 {
            let rbsp = unescape_rbsp(&nal[1..]);
            let sps = Sps::parse(&rbsp)?;
            let want_u8 = sps.bit_depth_luma == 8 && sps.bit_depth_chroma == 8;
            let matches = match &self.inner {
                None => false,
                Some(Inner::U8(_)) => want_u8,
                Some(Inner::U16(_)) => !want_u8,
            };
            if !matches {
                // Drain what the previous implementation still owes, then
                // switch. (Only the very first SPS normally gets here.)
                if let Some(mut old) = self.inner.take() {
                    let _ = with_inner!(&mut old, d => d.flush());
                    while let Some(p) = with_inner!(&mut old, d => d.next_picture()) {
                        self.leftover.push_back(p);
                    }
                    self.warnings_before += with_inner!(&old, d => d.warnings());
                }
                let mut inner = if want_u8 { Inner::U8(H264DecoderImpl::with_threads(self.threads)) } else { Inner::U16(H264DecoderImpl::with_threads(self.threads)) };
                for p in std::mem::take(&mut self.pending) {
                    let _ = with_inner!(&mut inner, d => d.push_nal(&p));
                }
                self.inner = Some(inner);
            }
        }
        match &mut self.inner {
            Some(inner) => with_inner!(inner, d => d.push_nal(nal)),
            None => {
                // Anything before the first SPS: keep it for the
                // implementation; slices without an SPS are an error.
                if hdr.unit_type == 1 || hdr.unit_type == 5 {
                    return Err(Error::bitstream("slice before any SPS"));
                }
                self.pending.push(nal.to_vec());
                Ok(())
            }
        }
    }

    /// End of stream: finish the current picture and drain the DPB.
    pub fn flush(&mut self) -> Result<()> {
        match &mut self.inner {
            Some(inner) => with_inner!(inner, d => d.flush()),
            None => Ok(()),
        }
    }

    /// The next picture in output order, if any (waits for it to finish).
    pub fn next_picture(&mut self) -> Option<Picture> {
        if let Some(p) = self.leftover.pop_front() {
            return Some(p);
        }
        match &mut self.inner {
            Some(inner) => with_inner!(inner, d => d.next_picture()),
            None => None,
        }
    }

    /// The next picture in output order if it is already finished.
    pub fn try_next_picture(&mut self) -> Option<Picture> {
        if let Some(p) = self.leftover.pop_front() {
            return Some(p);
        }
        match &mut self.inner {
            Some(inner) => with_inner!(inner, d => d.try_next_picture()),
            None => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn x264_build_from_user_data_sei() {
        let mut d = H264DecoderImpl::<u8>::with_threads(1);
        let text = b"x264 - core 148 r2708 86b7198 - H.264/MPEG-4 AVC codec - Copyleft 2003-2016";
        let mut payload = vec![0xdc, 0x45, 0xe9, 0xbd, 0xe6, 0xd9, 0x48, 0xb7, 0x96, 0x2c, 0xd8, 0x20, 0xd9, 0x23, 0xee, 0xef];
        payload.extend_from_slice(text);
        // A leading unrelated message (type 1, one byte), then ours (type 5).
        let mut sei = vec![1u8, 1, 0x00, 5];
        let mut size = payload.len();
        while size >= 255 {
            sei.push(255);
            size -= 255;
        }
        sei.push(size as u8);
        sei.extend_from_slice(&payload);
        sei.push(0x80);
        d.parse_sei(&sei);
        assert_eq!(d.x264_build, Some(148));
    }
}
