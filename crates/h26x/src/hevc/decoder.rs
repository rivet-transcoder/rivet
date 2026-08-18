//! The HEVC decoder.
//!
//! The caller's thread parses NAL headers, decides picture boundaries,
//! derives POC / RPS / reference lists and drives the DPB. Everything
//! sample-related runs on worker threads at two levels of parallelism:
//!
//! - **pictures** in flight concurrently (frame threading), each waiting on
//!   exactly the rows of its references that its motion vectors reach;
//! - **substreams** within a picture (WPP CTB rows, tiles, slice segments)
//!   as independent tasks, each CTB waiting for the neighbours the standard
//!   lets it read (left, above, above-right) — the wavefront.
//!
//! Every task depends only on tasks submitted before it, and the task queue
//! is FIFO, so a task that waits is always waiting on something already
//! running or done. The in-loop filters run one CTB row behind, from
//! whichever task completes a row, and a per-picture finisher publishes the
//! last rows once the main thread has closed the picture.

use std::cell::UnsafeCell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};

use crate::cabac::Cabac;
use crate::dsp::Cpu;
use crate::dsp::hevc::HevcDsp;
use crate::nal::{HevcNalHeader, annexb_nals, escaped_offset, unescape_rbsp, unescape_rbsp_positions, unescaped_offset};
use crate::picture::{ChromaFormat, Picture};
use crate::threading::{Pool, default_threads, prof};
use crate::{Error, Result};

use super::ctu::{SliceDec, TraceCfg};
use super::ctx::Contexts;
use super::deblock::{DeblockScratch, deblock_rows};
use super::dpb::{Dpb, DpbPic, RefSets};
use super::frame::{Frame, FramePool, Sample, SharedFrame};
use super::inter::McScratch;
use super::mvpred::RefCtx;
use super::pic::{Geometry, PicInfo, SliceFilterParams};
use super::pps::Pps;
use super::sao::{sao_ctb_row, SaoBand};
use super::slice::{SliceHeader, SliceType, nal_type};
use super::sps::{ScalingList, Sps, Vps};

// ----------------------------------------------------------------------
// Per-picture shared state
// ----------------------------------------------------------------------

/// What one slice segment's substream tasks share.
struct Segment {
    hdr: SliceHeader,
    /// The independent header this segment belongs to (itself if independent).
    ind: Arc<SliceHeader>,
    rbsp: Arc<Vec<u8>>,
    /// Substream start offsets into `rbsp`.
    substreams: Vec<usize>,
    /// Per substream: the CTB (raster address) it starts at, and whether it
    /// starts a new tile (as opposed to a WPP row inside a tile).
    starts: Vec<(usize, bool)>,
    /// Per substream: already handed to the pool.
    submitted: Vec<AtomicBool>,
    /// Substreams handed to the pool so far.
    spawned: AtomicUsize,
    slice_idx: u16,
    /// Contexts + qPY at the end of the segment (for a dependent successor).
    end_state: Mutex<Option<(Contexts, i32)>>,
    finished: AtomicBool,
}

/// The state of a picture being decoded, shared by its tasks.
struct PicShared<S: Sample> {
    frame: Arc<SharedFrame<S>>,
    info: UnsafeCell<PicInfo>,
    sps: Sps,
    pps: Pps,
    poc: i32,
    sets: RefSets<S>,
    scaling: Option<ScalingList>,
    dsp: HevcDsp<S>,
    /// Per CTB: decoded.
    ctb_done: Vec<AtomicBool>,
    /// Decoded CTB count.
    done_count: AtomicUsize,
    /// Tasks submitted so far / finished so far / currently blocked in a wait.
    tasks_submitted: AtomicUsize,
    tasks_finished: AtomicUsize,
    tasks_waiting: AtomicUsize,
    /// Tasks run on the pool (false = inline on the caller's thread, where a
    /// wait can never be satisfied later and returns at once).
    parallel: bool,
    /// The main thread has sent everything.
    closed: AtomicBool,
    /// The tail (last filter rows, completion) has been run.
    tail_done: AtomicBool,
    /// Per CTB row: contexts after its second CTB (WPP storage).
    wpp_ctx: Vec<UnsafeCell<Option<Contexts>>>,
    /// Tile columns (the stride of `wpp_ctx`).
    wpp_cols: usize,
    segments: Mutex<Vec<Arc<Segment>>>,
    /// Waiting on CTBs / tasks.
    lock: Mutex<()>,
    cv: Condvar,
    /// Decoded CTBs per CTB row (a row is complete at `wc`).
    row_ctbs: Vec<AtomicUsize>,
    /// CTBs per row.
    wc: usize,
    /// Rows complete from the top (the `decoded` frontier), see
    /// `publish_decoded_frontier`.
    decoded_rows: AtomicUsize,
    /// The task pool (None = inline).
    pool: Option<Arc<Pool>>,
    /// The filter pool: its tasks never wait, so they always make progress
    /// even when every decoding worker is blocked on a dependency (which is
    /// why they cannot share the decoding workers' queue).
    filter_pool: Option<Arc<Pool>>,
    /// Inline mode: tasks queued in FIFO order and run one after another by
    /// the caller's thread (a task must not run in the middle of another).
    inline_queue: Mutex<std::collections::VecDeque<Box<dyn FnOnce() + Send>>>,
    /// The row-pipelined filters (locked only when a row completes).
    filters: Mutex<RowFilterState<S>>,
    /// A row completed while another thread held `filters`.
    filter_pending: AtomicBool,
    frames: FramePool<S>,
    deblock: bool,
    sao: bool,
    trace: TraceCfg,
    warnings: Arc<AtomicU64>,
    /// When the picture was created (profiling).
    created: std::time::Instant,
    /// When its first CTB was decoded (profiling).
    first_ctb: Mutex<Option<std::time::Instant>>,
}

// SAFETY: `info` and the frame are written by tasks in disjoint CTB regions
// and read only after the writing task published `ctb_done` (Release/Acquire
// through the atomics); the row filters take a mutex.
unsafe impl<S: Sample> Sync for PicShared<S> {}
unsafe impl<S: Sample> Send for PicShared<S> {}

impl<S: Sample> PicShared<S> {
    #[allow(clippy::mut_from_ref)]
    unsafe fn info(&self) -> &mut PicInfo {
        unsafe { &mut *self.info.get() }
    }
    #[allow(clippy::mut_from_ref)]
    unsafe fn frame_mut(&self) -> &mut Frame<S> {
        unsafe { self.frame.get_mut() }
    }

    /// Advance the "rows decoded from the top" frontier over every complete
    /// row and publish it as the frame's `decoded` progress.
    fn publish_decoded_frontier(&self, height: usize) {
        let _g = self.lock.lock().unwrap();
        let mut r = self.decoded_rows.load(Ordering::Acquire);
        let hc = self.row_ctbs.len();
        let width_ctbs = self.wc;
        while r < hc && self.row_ctbs[r].load(Ordering::Acquire) >= width_ctbs {
            r += 1;
        }
        self.decoded_rows.store(r, Ordering::Release);
        drop(_g);
        let ctb = 1usize << self.sps.log2_ctb_size;
        self.frame.progress.set_decoded(((r * ctb).min(height)) as i32);
    }

    fn mark_ctb_done(&self, addr: usize) {
        self.ctb_done[addr].store(true, Ordering::Release);
        self.done_count.fetch_add(1, Ordering::AcqRel);
        // Only wake sleepers when there are any (spinners see the flag).
        if self.tasks_waiting.load(Ordering::Acquire) > 0 {
            let _g = self.lock.lock().unwrap();
            self.cv.notify_all();
        }
    }

    fn wait_ctb(&self, addr: usize) {
        if self.ctb_done[addr].load(Ordering::Acquire) || !self.parallel {
            return;
        }
        struct T(std::time::Instant);
        impl Drop for T {
            fn drop(&mut self) {
                if prof::enabled() {
                    prof::add(&prof::WAIT_NEIGHBOUR, self.0);
                }
            }
        }
        let _t = T(std::time::Instant::now());
        // Spin briefly: the producer is usually one CTB away.
        let start = std::time::Instant::now();
        loop {
            for _ in 0..64 {
                std::hint::spin_loop();
            }
            if self.ctb_done[addr].load(Ordering::Acquire) {
                return;
            }
            if start.elapsed() > std::time::Duration::from_micros(15) {
                break;
            }
        }
        self.blocking_wait(|| self.ctb_done[addr].load(Ordering::Acquire));
    }

    /// Block until `ready()`, or until the picture is closed and every
    /// remaining task is blocked too (lost data: nobody will produce what we
    /// wait for — proceed, the missing blocks read as undecoded).
    fn blocking_wait(&self, ready: impl Fn() -> bool) {
        let mut g = self.lock.lock().unwrap();
        self.tasks_waiting.fetch_add(1, Ordering::AcqRel);
        // A waiter that has just been satisfied is still counted until it
        // wakes, so "everyone is blocked" is only believed after it has held
        // continuously for a while (real progress satisfies someone quickly).
        let mut stuck_since: Option<std::time::Instant> = None;
        loop {
            if ready() {
                break;
            }
            let closed = self.closed.load(Ordering::Acquire);
            let submitted = self.tasks_submitted.load(Ordering::Acquire);
            let finished = self.tasks_finished.load(Ordering::Acquire);
            let waiting = self.tasks_waiting.load(Ordering::Acquire);
            if closed && finished + waiting >= submitted {
                let since = *stuck_since.get_or_insert_with(std::time::Instant::now);
                if since.elapsed() > std::time::Duration::from_millis(200) {
                    break;
                }
                let (g2, _) = self.cv.wait_timeout(g, std::time::Duration::from_millis(50)).unwrap();
                g = g2;
                continue;
            }
            stuck_since = None;
            g = self.cv.wait(g).unwrap();
        }
        self.tasks_waiting.fetch_sub(1, Ordering::AcqRel);
        self.cv.notify_all();
    }

    /// Wait for the CTBs a CTB at `(rx, ry)` may read: left, above,
    /// above-right, above-left — when they are in the same tile.
    fn wait_neighbours(&self, rx: usize, ry: usize) {
        let info = unsafe { self.info() };
        let wc = info.wc;
        let addr = ry * wc + rx;
        let tile = info.ctb_tile[addr];
        let mut need = |a: usize| {
            if info.ctb_tile[a] == tile {
                self.wait_ctb(a);
            }
        };
        if ry > 0 {
            if rx + 1 < wc {
                need(addr - wc + 1);
            }
            need(addr - wc);
            if rx > 0 {
                need(addr - wc - 1);
            }
        }
        if rx > 0 {
            need(addr - 1);
        }
    }

    /// A row completed: filter every complete row in order, unless another
    /// thread is already doing so (it will pick the new row up).
    fn run_filters(&self, frame: &mut Frame<S>, info: &PicInfo) {
        loop {
            let Ok(mut f) = self.filters.try_lock() else {
                self.filter_pending.store(true, Ordering::Release);
                // The holder may have just released without seeing the flag.
                if self.filters.try_lock().is_err() {
                    return;
                }
                continue;
            };
            self.filter_pending.store(false, Ordering::Release);
            f.row_done(self, frame, info);
            drop(f);
            if !self.filter_pending.load(Ordering::Acquire) {
                return;
            }
        }
    }

    fn task_finished(&self) {
        self.tasks_finished.fetch_add(1, Ordering::AcqRel);
        {
            let _g = self.lock.lock().unwrap();
            self.cv.notify_all();
        }
        self.maybe_finish();
    }

    /// Whether the picture is closed and every task has run.
    fn all_done(&self) -> bool {
        self.closed.load(Ordering::Acquire) && self.tasks_finished.load(Ordering::Acquire) >= self.tasks_submitted.load(Ordering::Acquire)
    }

    /// Run the tail exactly once, on whichever thread observes the picture
    /// to be complete (the last task, or the closer).
    fn maybe_finish(&self) {
        if self.all_done() && !self.tail_done.swap(true, Ordering::AcqRel) {
            finish_picture_tasks(self);
        }
    }
}

thread_local! {
    static SCRATCH: std::cell::RefCell<Vec<Box<dyn std::any::Any + Send>>> = const { std::cell::RefCell::new(Vec::new()) };
}

fn take_scratch<S: Sample>() -> McScratch<S> {
    SCRATCH.with(|s| {
        let mut v = s.borrow_mut();
        let pos = v.iter().position(|b| b.is::<McScratch<S>>());
        pos.map(|i| *v.swap_remove(i).downcast::<McScratch<S>>().expect("checked"))
    })
    .unwrap_or_default()
}

fn give_scratch<S: Sample>(m: McScratch<S>) {
    SCRATCH.with(|s| s.borrow_mut().push(Box::new(m)));
}

// ----------------------------------------------------------------------
// Substream tasks
// ----------------------------------------------------------------------

/// Decode one substream (`sub` of segment `seg`): from its first CTB until
/// `end_of_slice_segment_flag`, the end of the tile, or the end of the CTB
/// row (with WPP).
fn run_substream<S: Sample>(pic_arc: &Arc<PicShared<S>>, seg_arc: &Arc<Segment>, sub: usize) -> Result<()> {
    let pic: &PicShared<S> = pic_arc;
    let seg: &Segment = seg_arc;
    let sps = &pic.sps;
    let pps = &pic.pps;
    let hdr = &seg.hdr;
    let ind = &*seg.ind;
    // SAFETY: see PicShared.
    let info: &mut PicInfo = unsafe { pic.info() };
    let frame: &mut Frame<S> = unsafe { pic.frame_mut() };
    let wc = info.wc;
    let n_ctbs = wc * info.hc;

    // Where this substream starts.
    let seg_start_rs = hdr.segment_address as usize;
    let seg_start_ts = info.ctb_rs_to_ts[seg_start_rs] as usize;
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
    let _ = seg_start_ts;
    let Some(&(start_rs, _)) = seg.starts.get(sub) else {
        return Err(Error::bitstream("entry point beyond the picture"));
    };
    let mut ctb_addr_rs = start_rs;
    let mut ctb_addr_ts = info.ctb_rs_to_ts[ctb_addr_rs] as usize;
    let Some(&data_start) = seg.substreams.get(sub) else {
        return Err(Error::bitstream("missing entry point"));
    };
    let mut cabac = Cabac::new(&seg.rbsp[data_start..]);
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
    let slice_addr = ind.segment_address;

    // Reference lists.
    let lists = if ind.slice_type != SliceType::I { pic.sets.build_ref_lists(ind)? } else { [Vec::new(), Vec::new()] };
    // SAFETY: reference reads wait on progress.
    let ref_frames: [Vec<&Frame<S>>; 2] = [lists[0].iter().map(|e| unsafe { e.frame.get() }).collect(), lists[1].iter().map(|e| unsafe { e.frame.get() }).collect()];
    let ref_shared: [Vec<&SharedFrame<S>>; 2] = [lists[0].iter().map(|e| &*e.frame).collect(), lists[1].iter().map(|e| &*e.frame).collect()];
    let pocs: [Vec<i32>; 2] = [lists[0].iter().map(|e| e.poc).collect(), lists[1].iter().map(|e| e.poc).collect()];
    let long_term: [Vec<bool>; 2] = [lists[0].iter().map(|e| e.long_term).collect(), lists[1].iter().map(|e| e.long_term).collect()];
    let no_backward_pred = pocs[0].iter().chain(pocs[1].iter()).all(|&p| p <= pic.poc);
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
        cur_poc: pic.poc,
        no_backward_pred,
        tmvp: ind.temporal_mvp_enabled,
        max_merge_cand: ind.max_num_merge_cand as usize,
        log2_par_mrg_level: pps.log2_parallel_merge_level,
        is_b: ind.slice_type == SliceType::B,
        num_ref_idx: [ind.num_ref_idx[0] as usize, ind.num_ref_idx[1] as usize],
        col_from_l0: ind.collocated_from_l0,
    };

    // Contexts at the start (9.3.1).
    let first_in_tile = ctb_addr_ts == 0 || info.tile_id_ts[ctb_addr_ts] != info.tile_id_ts[ctb_addr_ts - 1];
    let row_start = pps.entropy_coding_sync && ctb_addr_rs % wc == tile_col_start(ctb_addr_rs);
    let mut cx = Contexts::new(init_type, ind.slice_qp);
    let mut first_qg = true;
    let mut qp_prev_init = ind.slice_qp;
    if first_in_tile {
        // init
    } else if row_start {
        // WPP: the row above must have passed its second CTB (we wait for
        // above-right anyway before the first CTB; do it now to read the
        // stored contexts).
        if let Some(saved) = wpp_sync_source(pic, info, ctb_addr_rs, slice_addr) {
            cx = saved;
        }
    } else if hdr.dependent && sub == 0 {
        // Continues the previous segment: wait for it to end.
        let prev = {
            let segs = pic.segments.lock().unwrap();
            let idx = segs.iter().position(|s| std::ptr::eq(&**s, seg)).unwrap_or(0);
            if idx > 0 { Some(segs[idx - 1].clone()) } else { None }
        };
        if let Some(prev) = prev {
            wait_segment(pic, &prev);
            if let Some((c, q)) = prev.end_state.lock().unwrap().clone() {
                cx = c;
                qp_prev_init = q;
                first_qg = false;
            }
        }
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
        slice_idx: seg.slice_idx,
        slice_addr,
        scaling: pic.scaling.clone(),
        qp_y: ind.slice_qp,
        qp_y_prev: qp_prev_init,
        cu_qp_delta_val: 0,
        is_cu_qp_delta_coded: false,
        is_cu_chroma_qp_offset_coded: false,
        cu_qp_offset_c: [0, 0],
        qg: (0, 0),
        qg_qp_prev: qp_prev_init,
        first_qg,
        last_pu_merged: false,
        ctb_addr_rs,
        ctb_addr_ts,
        coeffs: vec![0; 1024],
        luma_res: Vec::new(),
        luma_res_valid: false,
        dsp: pic.dsp,
        mc: {
            let mut m = take_scratch();
            if m.tmp.is_empty() {
                m = McScratch::new();
            }
            m
        },
        warnings: 0,
        trace: pic.trace,
    };

    let result = (|| -> Result<()> {
        loop {
            let rx = ctb_addr_rs % wc;
            let ry = ctb_addr_rs / wc;
            pic.wait_neighbours(rx, ry);
            let t_dec = std::time::Instant::now();
            if prof::enabled() {
                let mut f = pic.first_ctb.lock().unwrap();
                if f.is_none() {
                    *f = Some(t_dec);
                }
            }
            dec.decode_ctu(ctb_addr_rs, ctb_addr_ts)?;
            let end_of_slice_segment = dec.cabac.terminate() != 0;
            if prof::enabled() {
                prof::add(&prof::DECODE, t_dec);
            }
            if pic.trace.ctb {
                // Checksum of the CTB's luma right after reconstruction.
                let ctb = 1usize << sps.log2_ctb_size;
                let (x0, y0) = (rx * ctb, ry * ctb);
                let (w, h) = (ctb.min(dec.frame.width - x0), ctb.min(dec.frame.height - y0));
                let mut sum: u64 = 0;
                for yy in 0..h {
                    let off = dec.frame.y.offset(x0 as isize, (y0 + yy) as isize);
                    for xx in 0..w {
                        sum = sum.wrapping_mul(31).wrapping_add(dec.frame.y.data[off + xx].to_i32() as u64);
                    }
                }
                eprintln!("ctb poc={} addr={} sum={sum:x} qp={} cx0={}", pic.poc, ctb_addr_rs, dec.qp_y, dec.cx.c[0]);
            }
            if pps.entropy_coding_sync && rx == tile_col_start(ctb_addr_rs) + 1 {
                // WPP storage: written before this CTB is published; one
                // slot per (CTB row, tile column).
                // SAFETY: the slot is written by this task only.
                unsafe { *pic.wpp_ctx[ry * pic.wpp_cols + tile_col_idx(&pps, rx)].get() = Some(dec.cx.clone()) };
            }
            pic.mark_ctb_done(ctb_addr_rs);
            // The row below may start once we are two CTBs in (its first CTB
            // needs our second): hand its substream to the pool now, so tasks
            // start when their dependencies are met instead of blocking.
            if pps.entropy_coding_sync && rx >= tile_col_start(ctb_addr_rs) + 1 {
                spawn_substream(pic_arc, seg_arc, sub + 1);
            }
            // The filters advance from whichever task completes a row — as
            // a task of their own when there is a pool, so decoding does not
            // stall behind deblocking / SAO of the rows above (a picture
            // decoded as one substream then overlaps its filtering with its
            // parsing, and with the other pictures in flight).
            if pic.row_ctbs[ry].fetch_add(1, Ordering::AcqRel) + 1 == wc {
                // The rows decoded from the top of the picture are final in
                // motion and side data: say so before the filters get to them
                // (TMVP of later pictures waits on this, not on filtering).
                // Rows can complete out of order — slices are independent and
                // run concurrently — so the frontier is the first incomplete
                // row, advanced under the picture lock.
                pic.publish_decoded_frontier(dec.frame.height);
                match &pic_arc.filter_pool {
                    Some(pool) => spawn_filter_task(pic_arc, pool),
                    None => {
                        let t_f = std::time::Instant::now();
                        pic.run_filters(dec.frame, dec.info);
                        if prof::enabled() {
                            prof::add(&prof::FILTER, t_f);
                        }
                    }
                }
            }
            if end_of_slice_segment {
                // The segment's end state for a dependent successor.
                *seg.end_state.lock().unwrap() = Some((dec.cx.clone(), dec.qp_y));
                break;
            }
            // A one-CTB row: the row below starts as soon as this one ends.
            spawn_substream(pic_arc, seg_arc, sub + 1);
            ctb_addr_ts += 1;
            if ctb_addr_ts >= n_ctbs {
                return Err(Error::bitstream("slice segment runs past the picture"));
            }
            ctb_addr_rs = dec.info.ctb_ts_to_rs[ctb_addr_ts] as usize;
            let new_tile = dec.info.tile_id_ts[ctb_addr_ts] != dec.info.tile_id_ts[ctb_addr_ts - 1];
            let new_row = pps.entropy_coding_sync && ctb_addr_rs % wc == tile_col_start(ctb_addr_rs);
            if new_tile || new_row {
                // The next substream is another task's.
                break;
            }
            if dec.cabac.overrun() {
                return Err(Error::bitstream("slice data exhausted"));
            }
        }
        Ok(())
    })();
    pic.warnings.fetch_add(dec.warnings, Ordering::Relaxed);
    give_scratch(std::mem::take(&mut dec.mc));
    result
}

/// Hand substream `sub` of `seg` to the pool (or run it inline), once.
fn spawn_substream<S: Sample>(pic: &Arc<PicShared<S>>, seg: &Arc<Segment>, sub: usize) {
    let Some(flag) = seg.submitted.get(sub) else { return };
    if flag.swap(true, Ordering::AcqRel) {
        return;
    }
    pic.tasks_submitted.fetch_add(1, Ordering::AcqRel);
    // Queued in order: FIFO position = dependency order (see Pool).
    seg.spawned.fetch_add(1, Ordering::AcqRel);
    {
        let _g = pic.lock.lock().unwrap();
        pic.cv.notify_all();
    }
    let pic_arc = pic.clone();
    let seg_arc = seg.clone();
    let last = sub + 1 == seg.substreams.len();
    let job = move || {
        // Whatever happens (including a panic), the task counts as finished
        // so the picture can complete.
        struct Done<'a, S: Sample>(&'a PicShared<S>, &'a Segment, bool);
        impl<S: Sample> Drop for Done<'_, S> {
            fn drop(&mut self) {
                if self.2 {
                    self.1.finished.store(true, Ordering::Release);
                }
                self.0.task_finished();
            }
        }
        let _done = Done(&pic_arc, &seg_arc, last);
        let r = run_substream(&pic_arc, &seg_arc, sub);
        if let Err(e) = r {
            if pic_arc.trace.cu || std::env::var_os("H26X_DEBUG").is_some() {
                eprintln!("substream error: poc={} sub={sub}: {e}", pic_arc.poc);
            }
            pic_arc.frame.progress.error.store(true, Ordering::Relaxed);
            pic_arc.warnings.fetch_add(1, Ordering::Relaxed);
        }
    };
    match &pic.pool {
        Some(pool) => pool.spawn(Box::new(job)),
        None => pic.inline_queue.lock().unwrap().push_back(Box::new(job)),
    }
}

/// Queue a task that runs the row filters for whatever rows are complete.
/// It waits for nothing, so its position in the pool's FIFO is harmless; it
/// counts as a task so the picture's tail runs after it.
fn spawn_filter_task<S: Sample>(pic: &Arc<PicShared<S>>, pool: &Arc<Pool>) {
    pic.tasks_submitted.fetch_add(1, Ordering::AcqRel);
    let pic_arc = pic.clone();
    pool.spawn(Box::new(move || {
        struct Done<'a, S: Sample>(&'a PicShared<S>);
        impl<S: Sample> Drop for Done<'_, S> {
            fn drop(&mut self) {
                self.0.task_finished();
            }
        }
        let _done = Done(&pic_arc);
        let t_f = std::time::Instant::now();
        // SAFETY: the filters touch only rows the decoding tasks have
        // finished (row_ctbs), the same discipline as when a decoding task
        // runs them; the filter state is behind its mutex.
        let frame: &mut Frame<S> = unsafe { pic_arc.frame_mut() };
        let info: &PicInfo = unsafe { pic_arc.info() };
        pic_arc.run_filters(frame, info);
        if prof::enabled() {
            prof::add(&prof::FILTER, t_f);
        }
    }));
}

/// Inline mode: run queued tasks in order until none is left.
fn drain_inline<S: Sample>(pic: &PicShared<S>) {
    loop {
        let job = pic.inline_queue.lock().unwrap().pop_front();
        match job {
            Some(j) => j(),
            None => break,
        }
    }
}

/// The WPP synchronisation source for the first CTB of a row: the contexts
/// stored after the second CTB of the row above, if that CTB is in the same
/// slice and tile.
fn wpp_sync_source<S: Sample>(pic: &PicShared<S>, info: &PicInfo, ctb_addr_rs: usize, slice_addr: u32) -> Option<Contexts> {
    let wc = info.wc;
    let row = ctb_addr_rs / wc;
    if row == 0 {
        return None;
    }
    let above_right = ctb_addr_rs + 1 - wc;
    if above_right / wc != row - 1 {
        return None;
    }
    // Availability needs the CTB decoded; wait for it (it precedes us).
    if info.ctb_tile[above_right] != info.ctb_tile[ctb_addr_rs] {
        return None;
    }
    pic.wait_ctb(above_right);
    if info.ctb_slice_addr[above_right] != slice_addr {
        return None;
    }
    // SAFETY: written by the row above before it published that CTB.
    let col = tile_col_idx(&pic.pps, ctb_addr_rs % wc);
    unsafe { (*pic.wpp_ctx[(row - 1) * pic.wpp_cols + col].get()).clone() }
}

/// The tile column holding CTB column `rx`.
fn tile_col_idx(pps: &Pps, rx: usize) -> usize {
    let mut idx = 0;
    for (i, &b) in pps.col_bd.iter().enumerate().skip(1) {
        if (b as usize) <= rx {
            idx = i;
        }
    }
    idx
}

fn wait_segment<S: Sample>(pic: &PicShared<S>, seg: &Segment) {
    if seg.finished.load(Ordering::Acquire) || !pic.parallel {
        return;
    }
    pic.blocking_wait(|| seg.finished.load(Ordering::Acquire));
}

// ----------------------------------------------------------------------
// Row-pipelined loop filters
// ----------------------------------------------------------------------

/// Intra prediction reads the *unfiltered* samples of the row above, so a
/// CTB row may only be deblocked once the row below it is fully decoded. So
/// when CTB row `r` completes: row `r - 1` is deblocked (vertical edges then
/// horizontal — that settles row `r - 2`), row `r - 2` is SAO'd from the
/// deblocked copy, its borders are extended and its rows are published as
/// final for the pictures waiting on this one.
struct RowFilterState<S: Sample> {
    next_filter_row: usize,
    /// The deblocked source lines of the row being SAO'd (a CTB row plus a
    /// line above and below), and which picture rows they are.
    sao_src: Option<Box<Frame<S>>>,
    sao_band: SaoBand<S>,
    finished: bool,
    deblock_scratch: DeblockScratch,
}

impl<S: Sample> RowFilterState<S> {
    /// Some row just completed: act on every complete row in order.
    fn row_done(&mut self, pic: &PicShared<S>, frame: &mut Frame<S>, info: &PicInfo) {
        let wc = info.wc;
        while self.next_filter_row < info.hc && pic.row_ctbs[self.next_filter_row].load(Ordering::Acquire) >= wc {
            let r = self.next_filter_row;
            self.row_complete(pic, r, frame, info);
            self.next_filter_row += 1;
        }
    }

    fn row_span(pic: &PicShared<S>, r: usize, frame: &Frame<S>) -> (usize, usize) {
        let ctb = 1usize << pic.sps.log2_ctb_size;
        (r * ctb, ((r + 1) * ctb).min(frame.height))
    }

    fn row_complete(&mut self, pic: &PicShared<S>, r: usize, frame: &mut Frame<S>, info: &PicInfo) {
        let (_, y1) = Self::row_span(pic, r, frame);
        pic.frame.progress.set_decoded(y1 as i32);
        if r >= 1 {
            self.deblock_row(pic, r - 1, frame, info);
        }
        if r >= 2 {
            self.sao_and_publish(pic, r - 2, frame, info);
        }
    }

    fn deblock_row(&mut self, pic: &PicShared<S>, r: usize, frame: &mut Frame<S>, info: &PicInfo) {
        if !pic.deblock {
            return;
        }
        let (y0, y1) = Self::row_span(pic, r, frame);
        deblock_rows(&pic.dsp, &mut self.deblock_scratch, frame, info, &pic.pps, pic.sps.bit_depth_luma, pic.sps.bit_depth_chroma, y0 / 4, y1.div_ceil(4));
    }

    fn sao_and_publish(&mut self, pic: &PicShared<S>, r: usize, frame: &mut Frame<S>, info: &PicInfo) {
        let (y0, y1) = Self::row_span(pic, r, frame);
        if pic.sao && pic.sps.sao_enabled {
            let ctb = 1usize << pic.sps.log2_ctb_size;
            // A band frame from the pool (recycled picture to picture: an
            // allocation per picture here was measurable in page faults).
            let frames = &pic.frames;
            let src = self.sao_src.get_or_insert_with(|| Box::new(frames.take(frame.width, ctb + 4, frame.chroma, frame.bit_depth)));
            self.sao_band.fill(frame, src, ctb, r);
            sao_ctb_row(&pic.dsp, frame, src, &self.sao_band, info, &pic.sps, &pic.pps, r);
        }
        frame.extend_rows(y0, y1);
        pic.frame.progress.set_done(y1 as i32);
    }

    /// End of picture: rows never completed count as complete; the last row
    /// is deblocked and the last two rows are finished.
    fn finish(&mut self, pic: &PicShared<S>, frame: &mut Frame<S>, info: &PicInfo) {
        if self.finished {
            return;
        }
        self.finished = true;
        for r in self.next_filter_row..info.hc {
            self.row_complete(pic, r, frame, info);
            self.next_filter_row = r + 1;
        }
        if info.hc > 0 {
            let last = info.hc - 1;
            self.deblock_row(pic, last, frame, info);
            if last >= 1 {
                self.sao_and_publish(pic, last - 1, frame, info);
            }
            self.sao_and_publish(pic, last, frame, info);
        }
        if let Some(src) = self.sao_src.take() {
            pic.frames.give(*src);
        }
    }
}

/// Copy luma rows `y0..y1` (and the matching chroma rows) of `from` into `to`.
fn copy_rows<S: Sample>(from: &Frame<S>, to: &mut Frame<S>, y0: usize, y1: usize) {
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

/// The tail of a picture: run once every task has finished — the last
/// filter rows, then completion.
fn finish_picture_tasks<S: Sample>(pic: &PicShared<S>) {
    let t = std::time::Instant::now();
    // SAFETY: all tasks are done; this is now the only accessor.
    let frame: &mut Frame<S> = unsafe { pic.frame_mut() };
    let info: &PicInfo = unsafe { pic.info() };
    if frame.width != 0 {
        let mut f = pic.filters.lock().unwrap();
        f.finish(pic, frame, info);
    }
    frame.poc = pic.poc;
    pic.frame.progress.finish();
    if prof::enabled() {
        prof::add(&prof::WAIT_TASKS, t);
        let first = pic.first_ctb.lock().unwrap().map(|f| f.duration_since(pic.created).as_micros()).unwrap_or(0);
        eprintln!("pic poc={} created+{}us first-ctb, +{}us complete", pic.poc, first, pic.created.elapsed().as_micros());
    }
}

// ----------------------------------------------------------------------
// The decoder (main thread)
// ----------------------------------------------------------------------

/// The picture currently being fed (main-thread view).
struct Current<S: Sample> {
    id: u64,
    pic_output: bool,
    shared: Arc<PicShared<S>>,
    /// The independent slice header in force (dependent segments copy it).
    independent: Option<Arc<SliceHeader>>,
    slice_count: u16,
    /// Buffers taken (first slice seen).
    buffers_ready: bool,
}

/// `H26X_TRACE_PS=1`: print every parsed SPS / PPS.
fn trace_ps() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("H26X_TRACE_PS").is_some_and(|v| v == "1"))
}

/// The decoder for one sample type (see [`HevcDecoder`], which picks it).
pub(crate) struct HevcDecoderImpl<S: Sample> {
    vps: HashMap<u32, Vps>,
    sps: HashMap<u32, Sps>,
    pps: HashMap<u32, Pps>,
    dpb: Dpb<S>,
    cur: Option<Current<S>>,
    prev_tid0_poc: i32,
    first_in_sequence: bool,
    no_rasl_output: bool,
    skipping: bool,
    decode_index: u64,
    warnings: Arc<AtomicU64>,
    dsp: HevcDsp<S>,
    /// Substream tasks (FIFO, dependency order = submission order).
    tasks: Option<Arc<Pool>>,
    /// Row filter tasks (deblocking / SAO / publish); see `PicShared::filter_pool`.
    filter_tasks: Option<Arc<Pool>>,
    /// The last segment handed out (its rows are queued lazily by the rows
    /// above; the next segment waits until they are all queued so that FIFO
    /// order stays dependency order).
    last_segment: Option<(Arc<PicShared<S>>, Arc<Segment>)>,
    frames: FramePool<S>,
    deblock: bool,
    sao: bool,
    /// Scan/tile tables of the last picture's parameter sets, reused while
    /// the geometry (size, CTB size, tile boundaries) stays the same.
    geometry: Option<(GeoKey, Arc<Geometry>)>,
    /// Pictures started and possibly still decoding, oldest first: the
    /// caller's thread waits for the oldest before starting more than
    /// `max_in_flight` (back-pressure without blocking any worker).
    in_flight: std::collections::VecDeque<Arc<SharedFrame<S>>>,
    max_in_flight: usize,
}

/// What the geometry tables depend on.
#[derive(Clone, PartialEq, Eq)]
struct GeoKey {
    width: u32,
    height: u32,
    log2_ctb: u32,
    col_bd: Vec<u32>,
    row_bd: Vec<u32>,
}

impl<S: Sample> HevcDecoderImpl<S> {
    /// A decoder with one worker per hardware thread (capped), or as
    /// `H26X_THREADS` says.
    pub fn new() -> Self {
        let n = std::env::var("H26X_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or_else(default_threads);
        Self::with_threads(n)
    }

    /// A decoder with `threads` workers; 0 or 1 decodes on the caller's thread.
    pub fn with_threads(threads: usize) -> Self {
        // Substream tasks block on their neighbours and references while
        // holding a worker, so run more workers than hardware threads; the
        // queue is unbounded (pictures in flight are what is bounded).
        let tasks = if threads > 1 { Some(Pool::new(threads * 2, usize::MAX)) } else { None };
        // Filtering is roughly a fifth of the work and never blocks: a few
        // threads of their own keep it off the decoding tasks' critical path
        // without ever being starved by them.
        let filter_tasks = if threads > 1 { Some(Pool::new((threads / 4).max(1), usize::MAX)) } else { None };
        HevcDecoderImpl {
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
            tasks,
            filter_tasks,
            last_segment: None,
            frames: FramePool::new(),
            deblock: std::env::var_os("H26X_NO_DEBLOCK").is_none(),
            sao: std::env::var_os("H26X_NO_SAO").is_none(),
            geometry: None,
            in_flight: std::collections::VecDeque::new(),
            max_in_flight: std::env::var("H26X_INFLIGHT").ok().and_then(|v| v.parse().ok()).unwrap_or(threads.clamp(2, 16)),
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
        let t = std::time::Instant::now();
        let r = self.push_nal_inner(nal);
        if prof::enabled() {
            prof::add(&prof::MAIN, t);
        }
        r
    }

    fn push_nal_inner(&mut self, nal: &[u8]) -> Result<()> {
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
                if trace_ps() {
                    eprintln!("{s:#?}");
                }
                self.sps.insert(s.id, s);
            }
            nal_type::PPS => {
                let rbsp = unescape_rbsp(nal);
                let p = Pps::parse(&rbsp[2..])?;
                if trace_ps() {
                    eprintln!("{p:#?}");
                }
                self.pps.insert(p.id, p);
            }
            nal_type::EOS | nal_type::EOB => {
                self.finish_picture();
                self.first_in_sequence = true;
            }
            nal_type::SEI_SUFFIX => {
                // A decoded picture hash for the picture being decoded.
                if super::hash::verify_enabled() {
                    if let Some(cur) = &self.cur {
                        let rbsp = unescape_rbsp(nal);
                        if let Some(h) = super::hash::parse_sei(&rbsp[2..], cur.shared.sps.chroma_format_idc) {
                            *cur.shared.frame.hash.lock().unwrap() = Some(h);
                        }
                    }
                }
            }
            nal_type::AUD | nal_type::SEI_PREFIX | nal_type::FD => {}
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
        let (pic, ok) = self.dpb.output.pop_front()?.into_picture();
        if !ok {
            self.warnings.fetch_add(1, Ordering::Relaxed);
        }
        Some(pic)
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
        let independent = self.cur.as_ref().and_then(|c| c.independent.as_deref());
        let (hdr, mut pps, sps) = SliceHeader::parse(&rbsp, nh, &|id| pps_map.get(&id).cloned(), &|id| sps_map.get(&id).cloned(), independent)?;
        if first_flag {
            if sps.separate_colour_plane {
                return Err(Error::unsupported("separate_colour_plane_flag"));
            }
            if sps.bit_depth_luma != sps.bit_depth_chroma {
                return Err(Error::unsupported("different luma and chroma bit depths"));
            }
            if sps.bit_depth_luma > 12 {
                return Err(Error::unsupported(format!("bit depth {}", sps.bit_depth_luma)));
            }
            if sps.extended_precision() {
                return Err(Error::unsupported("extended_precision_processing_flag"));
            }
            if sps.cabac_bypass_alignment() {
                return Err(Error::unsupported("cabac_bypass_alignment_enabled_flag"));
            }
            pps.resolve_tiles(&sps)?;
            self.start_picture(&hdr, sps, pps, nh)?;
            if self.skipping {
                return Ok(());
            }
        }
        let Some(cur) = self.cur.as_mut() else { return Ok(()) };
        let pic = &cur.shared;
        // First slice: take the buffers (allocation happens here, on the
        // caller's thread, only when the pool has no spare frame).
        if !cur.buffers_ready {
            // SAFETY: nothing else touches the frame before progress > 0.
            let frame: &mut Frame<S> = unsafe { pic.frame_mut() };
            if frame.width == 0 {
                let mut f = self.frames.take(pic.sps.width as usize, pic.sps.height as usize, pic.sps.chroma_format(), pic.sps.bit_depth_luma);
                f.poc = pic.poc;
                *frame = f;
            }
            cur.buffers_ready = true;
        }
        // Slice bookkeeping (main thread, before any task of it runs).
        let ind: Arc<SliceHeader> = if hdr.dependent {
            match &cur.independent {
                Some(i) => i.clone(),
                None => {
                    self.warnings.fetch_add(1, Ordering::Relaxed);
                    return Ok(());
                }
            }
        } else {
            let a = Arc::new(hdr.clone());
            cur.independent = Some(a.clone());
            let idx = cur.slice_count as usize;
            cur.slice_count += 1;
            // SAFETY: the slot is written before any task reads it.
            let info: &mut PicInfo = unsafe { pic.info() };
            if idx < info.slices.len() {
                info.slices[idx] = SliceFilterParams {
                    deblocking_disabled: hdr.deblocking_disabled,
                    beta_offset: hdr.beta_offset,
                    tc_offset: hdr.tc_offset,
                    loop_filter_across_slices: hdr.loop_filter_across_slices,
                    slice_addr: hdr.segment_address,
                    cb_qp_offset: pic.pps.cb_qp_offset,
                    cr_qp_offset: pic.pps.cr_qp_offset,
                };
            }
            a
        };
        let slice_idx = cur.slice_count.saturating_sub(1);
        // Substream offsets in the unescaped data.
        let data_start_unesc = (hdr.data_bit_offset / 8) as usize;
        let data_start_esc = escaped_offset(data_start_unesc, &removed);
        let mut substreams: Vec<usize> = vec![data_start_unesc.min(rbsp.len())];
        for &ep in &hdr.entry_points {
            let esc = data_start_esc + ep as usize;
            substreams.push(unescaped_offset(esc, &removed).min(rbsp.len()));
        }
        let n_sub = substreams.len();
        // Where each substream starts (6.5: substream k > 0 begins at the
        // k-th tile / WPP-row boundary after the segment start).
        let mut starts: Vec<(usize, bool)> = Vec::with_capacity(n_sub);
        {
            // SAFETY: geometry tables are read-only.
            let info: &PicInfo = unsafe { pic.info() };
            let wc = info.wc;
            let n_ctbs = wc * info.hc;
            let pps = &pic.pps;
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
            let seg_start_rs = (hdr.segment_address as usize).min(n_ctbs.saturating_sub(1));
            starts.push((seg_start_rs, true));
            let mut ts = info.ctb_rs_to_ts[seg_start_rs] as usize + 1;
            while starts.len() < n_sub && ts < n_ctbs {
                let rs = info.ctb_ts_to_rs[ts] as usize;
                let new_tile = info.tile_id_ts[ts] != info.tile_id_ts[ts - 1];
                let new_row = pps.entropy_coding_sync && rs % wc == tile_col_start(rs);
                if new_tile || new_row {
                    starts.push((rs, new_tile));
                }
                ts += 1;
            }
        }
        let n_sub = starts.len().min(n_sub);
        let seg = Arc::new(Segment {
            hdr,
            ind,
            rbsp: Arc::new(rbsp),
            substreams,
            starts,
            submitted: (0..n_sub).map(|_| AtomicBool::new(false)).collect(),
            spawned: AtomicUsize::new(0),
            slice_idx,
            end_state: Mutex::new(None),
            finished: AtomicBool::new(false),
        });
        pic.segments.lock().unwrap().push(seg.clone());
        // FIFO gate: everything of the previous segment must be queued before
        // anything of this one (a task only waits on earlier queue entries).
        if let Some((prev_pic, prev_seg)) = self.last_segment.take() {
            let n = prev_seg.substreams.len().min(prev_seg.starts.len());
            if prev_seg.spawned.load(Ordering::Acquire) < n {
                let mut g = prev_pic.lock.lock().unwrap();
                while prev_seg.spawned.load(Ordering::Acquire) < n {
                    // If its rows can no longer be spawned (lost data), stop
                    // waiting once the picture is finished.
                    if prev_pic.frame.progress.is_complete() {
                        break;
                    }
                    let (g2, _) = prev_pic.cv.wait_timeout(g, std::time::Duration::from_millis(5)).unwrap();
                    g = g2;
                }
            }
        }
        // Submit the first substream and every tile start now; WPP rows are
        // submitted by the row above once it is two CTBs in.
        for sub in 0..n_sub {
            if sub == 0 || seg.starts[sub].1 {
                spawn_substream(&cur.shared, &seg, sub);
            }
        }
        if self.tasks.is_none() {
            drain_inline(&cur.shared);
        }
        self.last_segment = Some((cur.shared.clone(), seg));
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
        let chroma = sps.chroma_format();
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

        // Back-pressure: no more than max_in_flight pictures decoding.
        while self.in_flight.len() >= self.max_in_flight {
            if let Some(f) = self.in_flight.pop_front() {
                f.progress.wait_complete();
            }
        }
        self.in_flight.retain(|f| !f.progress.is_complete());
        let id = self.dpb.alloc_id();
        let shared_frame = Arc::new(SharedFrame::with_pool(Frame::empty(), poc, id, self.frames.clone()));
        self.in_flight.push_back(shared_frame.clone());
        self.dpb.insert_current(DpbPic {
            frame: shared_frame.clone(),
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
        let key = GeoKey { width: sps.width, height: sps.height, log2_ctb: sps.log2_ctb_size, col_bd: pps.col_bd.clone(), row_bd: pps.row_bd.clone() };
        let geo = match &self.geometry {
            Some((k, g)) if *k == key => g.clone(),
            _ => {
                let g = Arc::new(Geometry::new(&sps, &pps));
                self.geometry = Some((key, g.clone()));
                g
            }
        };
        let info = PicInfo::new(geo);
        let nc = info.wc * info.hc;
        let hc = info.hc;
        let wc = info.wc;
        let wpp_cols = pps.col_bd.len().saturating_sub(1).max(1);
        let pic = Arc::new(PicShared {
            frame: shared_frame,
            info: UnsafeCell::new(info),
            sps,
            pps,
            poc,
            sets,
            scaling,
            dsp: self.dsp,
            ctb_done: (0..nc).map(|_| AtomicBool::new(false)).collect(),
            done_count: AtomicUsize::new(0),
            tasks_submitted: AtomicUsize::new(0),
            tasks_finished: AtomicUsize::new(0),
            tasks_waiting: AtomicUsize::new(0),
            parallel: self.tasks.is_some(),
            closed: AtomicBool::new(false),
            tail_done: AtomicBool::new(false),
            wpp_cols,
            wpp_ctx: (0..hc * wpp_cols).map(|_| UnsafeCell::new(None)).collect(),
            segments: Mutex::new(Vec::new()),
            lock: Mutex::new(()),
            cv: Condvar::new(),
            row_ctbs: (0..hc).map(|_| AtomicUsize::new(0)).collect(),
            wc,
            decoded_rows: AtomicUsize::new(0),
            pool: self.tasks.clone(),
            filter_pool: self.filter_tasks.clone(),
            inline_queue: Mutex::new(std::collections::VecDeque::new()),
            filters: Mutex::new(RowFilterState { next_filter_row: 0, sao_src: None, sao_band: SaoBand::new(), finished: false, deblock_scratch: DeblockScratch::default() }),
            filter_pending: AtomicBool::new(false),
            frames: self.frames.clone(),
            deblock: self.deblock,
            sao: self.sao,
            trace: TraceCfg::from_env(),
            warnings: self.warnings.clone(),
            created: std::time::Instant::now(),
            first_ctb: Mutex::new(None),
        });
        self.cur = Some(Current { id, pic_output, shared: pic, independent: None, slice_count: 0, buffers_ready: false });
        self.decode_index += 1;
        Ok(())
    }

    fn finish_picture(&mut self) {
        let Some(cur) = self.cur.take() else { return };
        {
            let _g = cur.shared.lock.lock().unwrap();
            cur.shared.closed.store(true, Ordering::Release);
            cur.shared.cv.notify_all();
        }
        // If every task has already run, the tail is ours to trigger; hand
        // it to the pool so this thread stays light (inline mode runs it here).
        if cur.shared.all_done() && !cur.shared.tail_done.load(Ordering::Acquire) {
            match &self.tasks {
                Some(pool) => {
                    let p = cur.shared.clone();
                    pool.submit(Box::new(move || p.maybe_finish()));
                }
                None => cur.shared.maybe_finish(),
            }
        }
        self.dpb.finish_current(cur.id, cur.pic_output);
    }
}

impl<S: Sample> Drop for HevcDecoderImpl<S> {
    fn drop(&mut self) {
        self.finish_picture();
        if let Some(p) = &self.tasks {
            p.wait_idle();
        }
        if let Some(p) = &self.filter_tasks {
            p.wait_idle();
        }
        prof::report();
    }
}

// ----------------------------------------------------------------------
// The public decoder: picks the sample type from the SPS
// ----------------------------------------------------------------------

/// A native HEVC (H.265) decoder — Main / Main 10 / Main 12, 4:2:0.
///
/// Threaded at picture level (frame threading with row-progress dependency
/// tracking) and within pictures (WPP rows, tiles and slice segments as
/// wavefront tasks). The caller's thread only parses headers and manages
/// the DPB, so `push_nal` returns quickly; [`HevcDecoder::next_picture`]
/// hands back pictures in output order, waiting for each to finish, and
/// [`HevcDecoder::try_next_picture`] does not wait.
///
/// 8-bit streams decode into 8-bit planes, higher bit depths into 16-bit
/// planes: the implementation is chosen when the first SPS arrives (and
/// re-chosen, after draining, should a later SPS change the bit depth).
pub struct HevcDecoder {
    threads: usize,
    inner: Option<Inner>,
    /// NAL units seen before the sample type was known (VPS/SEI/...),
    /// replayed into the implementation once it exists.
    pending: Vec<Vec<u8>>,
    /// Output left over from an implementation replaced mid-stream.
    leftover: std::collections::VecDeque<Picture>,
    warnings_before: u64,
}

enum Inner {
    U8(HevcDecoderImpl<u8>),
    U16(HevcDecoderImpl<u16>),
}

macro_rules! with_inner {
    ($self:expr, $d:ident => $body:expr) => {
        match $self {
            Inner::U8($d) => $body,
            Inner::U16($d) => $body,
        }
    };
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
        HevcDecoder { threads, inner: None, pending: Vec::new(), leftover: std::collections::VecDeque::new(), warnings_before: 0 }
    }

    /// Non-fatal problems seen so far.
    pub fn warnings(&self) -> u64 {
        self.warnings_before + self.inner.as_ref().map_or(0, |i| with_inner!(i, d => d.warnings()))
    }

    /// Feed Annex-B data (any number of NAL units with start codes).
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
        if hdr.unit_type == nal_type::SPS && hdr.layer_id == 0 {
            let rbsp = unescape_rbsp(nal);
            let sps = Sps::parse(&rbsp[2..])?;
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
                let mut inner = if want_u8 { Inner::U8(HevcDecoderImpl::with_threads(self.threads)) } else { Inner::U16(HevcDecoderImpl::with_threads(self.threads)) };
                for p in std::mem::take(&mut self.pending) {
                    let _ = with_inner!(&mut inner, d => d.push_nal(&p));
                }
                self.inner = Some(inner);
            }
        }
        match &mut self.inner {
            Some(inner) => with_inner!(inner, d => d.push_nal(nal)),
            None => {
                // Parameter sets and the like before the first SPS: keep them
                // for the implementation; slices without an SPS are an error.
                if nal_type::is_slice(hdr.unit_type) {
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
