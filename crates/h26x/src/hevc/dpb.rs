//! The decoded picture buffer: reference picture set marking (8.3.2),
//! generation of unavailable references (8.3.3), reference picture list
//! construction (8.3.4) and output-order bumping (C.5.2). Runs on the
//! decoder's main thread; pictures are `Arc<SharedFrame<S>>` so the worker
//! decoding a picture and the workers reading it as a reference share it.

use std::collections::VecDeque;
use std::sync::Arc;

use crate::picture::{ChromaFormat, Picture};

use super::frame::{Frame, Sample, SharedFrame};
use super::slice::SliceHeader;
use super::sps::Sps;

/// A picture held in the DPB.
pub struct DpbPic<S: Sample = u16> {
    /// Samples and motion (possibly still being decoded).
    pub frame: Arc<SharedFrame<S>>,
    /// `PicOrderCntVal`.
    pub poc: i32,
    /// Marked "used for reference" (short- or long-term).
    pub is_ref: bool,
    /// Marked "used for long-term reference".
    pub long_term: bool,
    /// Marked "needed for output".
    pub needed_for_output: bool,
    /// `PicLatencyCount`.
    pub latency: u32,
    /// Decode order index.
    pub decode_index: u64,
    /// Conformance window (left, right, top, bottom) in luma samples.
    pub crop: (u32, u32, u32, u32),
    /// Generated to stand in for a missing reference (never output).
    pub generated: bool,
}

/// One entry of a resolved reference set: the picture plus what the current
/// picture needs to know about it.
#[derive(Clone)]
pub struct RefEntry<S: Sample = u16> {
    /// The picture.
    pub frame: Arc<SharedFrame<S>>,
    /// Its POC.
    pub poc: i32,
    /// Marked long-term for the current picture.
    pub long_term: bool,
}

/// The reference picture set of the current picture, resolved to pictures.
#[derive(Default, Clone)]
pub struct RefSets<S: Sample = u16> {
    /// `RefPicSetStCurrBefore`.
    pub st_curr_before: Vec<RefEntry<S>>,
    /// `RefPicSetStCurrAfter`.
    pub st_curr_after: Vec<RefEntry<S>>,
    /// `RefPicSetLtCurr`.
    pub lt_curr: Vec<RefEntry<S>>,
}

impl<S: Sample> RefSets<S> {
    /// `NumPicTotalCurr`.
    pub fn num_pic_total_curr(&self) -> usize {
        self.st_curr_before.len() + self.st_curr_after.len() + self.lt_curr.len()
    }

    /// 8.3.4: build `RefPicList0/1` for a slice.
    pub fn build_ref_lists(&self, hdr: &SliceHeader) -> Result<[Vec<RefEntry<S>>; 2], crate::Error> {
        let total = self.num_pic_total_curr();
        let mut out: [Vec<RefEntry<S>>; 2] = [Vec::new(), Vec::new()];
        let nlists = if hdr.slice_type.is_b() { 2 } else { 1 };
        for (list, out_list) in out.iter_mut().enumerate().take(nlists) {
            let n_active = hdr.num_ref_idx[list] as usize;
            if n_active == 0 {
                continue;
            }
            if total == 0 {
                return Err(crate::Error::bitstream("P/B slice with an empty reference picture set"));
            }
            let num_temp = n_active.max(total);
            let mut temp: Vec<&RefEntry<S>> = Vec::with_capacity(num_temp);
            let (first, second) = if list == 0 { (&self.st_curr_before, &self.st_curr_after) } else { (&self.st_curr_after, &self.st_curr_before) };
            while temp.len() < num_temp {
                for e in first.iter().chain(second.iter()).chain(self.lt_curr.iter()) {
                    if temp.len() < num_temp {
                        temp.push(e);
                    }
                }
            }
            let mut l = Vec::with_capacity(n_active);
            for i in 0..n_active {
                let idx = match &hdr.list_entry[list] {
                    Some(entries) => *entries.get(i).ok_or_else(|| crate::Error::bitstream("list_entry too short"))? as usize,
                    None => i,
                };
                if idx >= temp.len() {
                    return Err(crate::Error::bitstream("list_entry out of range"));
                }
                l.push(temp[idx].clone());
            }
            *out_list = l;
        }
        Ok(out)
    }
}

/// A picture that has been bumped out for display and is waiting to be
/// collected (it may still be decoding).
pub struct PendingOutput<S: Sample = u16> {
    /// The picture.
    pub frame: Arc<SharedFrame<S>>,
    /// Decode order index.
    pub decode_index: u64,
    /// Conformance window.
    pub crop: (u32, u32, u32, u32),
}

impl<S: Sample> PendingOutput<S> {
    /// Wait for the picture to finish and copy it out. The flag is false when
    /// the picture carried a hash SEI that does not match (verification on).
    pub fn into_picture(self) -> (Picture, bool) {
        let f = self.frame.wait_and_get();
        let mut ok = true;
        if let Some(h) = self.frame.hash.lock().unwrap().take() {
            if let Err(msg) = super::hash::verify(f, &h) {
                eprintln!("h26x: picture poc={} decode_index={}: {msg}", self.frame.poc, self.decode_index);
                ok = false;
            }
        }
        (f.to_picture(self.crop, self.frame.poc, self.decode_index), ok)
    }
}

/// The DPB with its output queue.
pub struct Dpb<S: Sample = u16> {
    /// Pictures.
    pub pics: Vec<DpbPic<S>>,
    /// Pictures output but not yet collected.
    pub output: VecDeque<PendingOutput<S>>,
    max_num_reorder: u32,
    /// `SpsMaxLatencyPictures` (None = no limit).
    max_latency: Option<u32>,
    max_dec_pic_buffering: u32,
    /// Warnings (missing references generated).
    pub warnings: u64,
    next_id: u64,
}

impl<S: Sample> Dpb<S> {
    /// Empty.
    pub fn new() -> Self {
        Dpb { pics: Vec::new(), output: VecDeque::new(), max_num_reorder: 0, max_latency: None, max_dec_pic_buffering: 1, warnings: 0, next_id: 1 }
    }

    /// A fresh picture id.
    pub fn alloc_id(&mut self) -> u64 {
        self.next_id += 1;
        self.next_id - 1
    }

    /// Adopt the limits of the active SPS.
    pub fn configure(&mut self, sps: &Sps) {
        self.max_num_reorder = sps.max_num_reorder_pics;
        self.max_latency = if sps.max_latency_increase_plus1 != 0 {
            Some(sps.max_num_reorder_pics + sps.max_latency_increase_plus1 - 1)
        } else {
            None
        };
        self.max_dec_pic_buffering = sps.max_dec_pic_buffering.max(1);
    }

    fn output_pic(&mut self, i: usize) {
        let p = &mut self.pics[i];
        p.needed_for_output = false;
        if !p.generated {
            self.output.push_back(PendingOutput { frame: p.frame.clone(), decode_index: p.decode_index, crop: p.crop });
        }
    }

    /// C.5.2.4: output the smallest-POC picture needed for output; drop it if
    /// unreferenced. Returns false if nothing was waiting.
    pub fn bump_one(&mut self) -> bool {
        let mut best: Option<usize> = None;
        for (i, p) in self.pics.iter().enumerate() {
            if p.needed_for_output && best.is_none_or(|b| p.poc < self.pics[b].poc) {
                best = Some(i);
            }
        }
        let Some(i) = best else { return false };
        self.output_pic(i);
        if !self.pics[i].is_ref {
            self.pics.remove(i);
        }
        true
    }

    fn remove_unused(&mut self) {
        self.pics.retain(|p| p.needed_for_output || p.is_ref);
    }

    fn num_needed_for_output(&self) -> usize {
        self.pics.iter().filter(|p| p.needed_for_output).count()
    }

    fn latency_exceeded(&self) -> bool {
        match self.max_latency {
            Some(m) => self.pics.iter().any(|p| p.needed_for_output && p.latency >= m),
            None => false,
        }
    }

    /// C.5.2.2: output/removal before decoding the current picture (after
    /// the RPS has been applied).
    pub fn before_decode(&mut self, irap_no_rasl_output: bool, no_output_of_prior_pics: bool) {
        if irap_no_rasl_output {
            if no_output_of_prior_pics {
                self.pics.clear();
            } else {
                self.remove_unused();
                while self.bump_one() {}
                self.pics.clear();
            }
            return;
        }
        self.remove_unused();
        while self.num_needed_for_output() > self.max_num_reorder as usize
            || self.latency_exceeded()
            || self.pics.len() >= self.max_dec_pic_buffering as usize
        {
            if !self.bump_one() {
                break;
            }
        }
    }

    /// Insert the current picture (as it starts decoding); C.5.2.3's marking
    /// and bumping happen in [`Dpb::finish_current`].
    pub fn insert_current(&mut self, pic: DpbPic<S>) {
        self.pics.push(pic);
    }

    /// C.5.2.3 for the picture `id`: mark for output and bump.
    pub fn finish_current(&mut self, id: u64, pic_output: bool) {
        let Some(idx) = self.pics.iter().position(|p| p.id() == id) else { return };
        let poc = self.pics[idx].poc;
        if pic_output {
            for (i, p) in self.pics.iter_mut().enumerate() {
                if i != idx && p.needed_for_output && p.poc > poc {
                    p.latency += 1;
                }
            }
        }
        {
            let p = &mut self.pics[idx];
            p.needed_for_output = pic_output;
            p.latency = 0;
            p.is_ref = true;
            p.long_term = false;
        }
        while self.num_needed_for_output() > self.max_num_reorder as usize || self.latency_exceeded() {
            if !self.bump_one() {
                break;
            }
        }
    }

    /// Output everything (end of stream).
    pub fn flush(&mut self) {
        while self.bump_one() {}
        self.pics.retain(|p| p.is_ref);
    }

    /// Drop everything.
    pub fn clear(&mut self) {
        self.pics.clear();
    }

    /// 8.3.2: derive the RPS of the current picture from its slice header,
    /// mark the DPB accordingly, generate missing references (8.3.3), and
    /// return the "Curr" sets. `idr` marks everything unused.
    #[allow(clippy::too_many_arguments)]
    pub fn apply_rps(
        &mut self,
        hdr: &SliceHeader,
        sps: &Sps,
        cur_poc: i32,
        idr: bool,
        chroma: ChromaFormat,
        bit_depth: u32,
        decode_index: u64,
        crop: (u32, u32, u32, u32),
    ) -> RefSets<S> {
        if idr {
            for p in &mut self.pics {
                p.is_ref = false;
                p.long_term = false;
            }
            return RefSets::default();
        }
        let max_poc_lsb = sps.max_poc_lsb();
        let mut poc_st_curr_before = Vec::new();
        let mut poc_st_curr_after = Vec::new();
        let mut poc_st_foll = Vec::new();
        for &(d, used) in &hdr.st_rps.neg {
            if used {
                poc_st_curr_before.push(cur_poc + d);
            } else {
                poc_st_foll.push(cur_poc + d);
            }
        }
        for &(d, used) in &hdr.st_rps.pos {
            if used {
                poc_st_curr_after.push(cur_poc + d);
            } else {
                poc_st_foll.push(cur_poc + d);
            }
        }
        let mut lt_curr: Vec<(i32, bool)> = Vec::new();
        let mut lt_foll: Vec<(i32, bool)> = Vec::new();
        for e in &hdr.lt {
            // e.poc is PocLsbLt - DeltaPocMsbCycleLt * MaxPocLsb; with the MSB
            // present the full POC adds the current picture's MSB (8-5).
            let poc = if e.msb_present { e.poc + (cur_poc - (cur_poc & (max_poc_lsb - 1))) } else { e.poc };
            let entry = (poc, e.msb_present);
            if e.used {
                lt_curr.push(entry);
            } else {
                lt_foll.push(entry);
            }
        }
        let find_lt = |pics: &[DpbPic<S>], poc: i32, msb_present: bool| -> Option<usize> {
            pics.iter().position(|p| {
                p.is_ref
                    && if msb_present { p.poc == poc } else { (p.poc & (max_poc_lsb - 1)) == (poc & (max_poc_lsb - 1)) }
            })
        };
        let mut lt_curr_idx: Vec<Option<usize>> = lt_curr.iter().map(|&(poc, msb)| find_lt(&self.pics, poc, msb)).collect();
        let lt_foll_idx: Vec<Option<usize>> = lt_foll.iter().map(|&(poc, msb)| find_lt(&self.pics, poc, msb)).collect();
        let find_st = |pics: &[DpbPic<S>], poc: i32| -> Option<usize> { pics.iter().position(|p| p.is_ref && !p.long_term && p.poc == poc) };
        let mut st_before_idx: Vec<Option<usize>> = poc_st_curr_before.iter().map(|&p| find_st(&self.pics, p)).collect();
        let mut st_after_idx: Vec<Option<usize>> = poc_st_curr_after.iter().map(|&p| find_st(&self.pics, p)).collect();
        let st_foll_idx: Vec<Option<usize>> = poc_st_foll.iter().map(|&p| find_st(&self.pics, p)).collect();

        // Marking.
        let mut keep = vec![false; self.pics.len()];
        let mut make_lt = vec![false; self.pics.len()];
        for i in lt_curr_idx.iter().chain(lt_foll_idx.iter()).flatten() {
            keep[*i] = true;
            make_lt[*i] = true;
        }
        for i in st_before_idx.iter().chain(st_after_idx.iter()).chain(st_foll_idx.iter()).flatten() {
            keep[*i] = true;
        }
        for (i, p) in self.pics.iter_mut().enumerate() {
            if !keep[i] {
                p.is_ref = false;
                p.long_term = false;
            } else if make_lt[i] {
                p.long_term = true;
            }
        }
        // Generate missing "Curr" references (8.3.3.2).
        let mut next_id = self.next_id;
        let mut generate = |poc: i32, long_term: bool, pics: &mut Vec<DpbPic<S>>| -> usize {
            let mut f = Frame::<S>::new(sps.width as usize, sps.height as usize, chroma, bit_depth);
            let mid = S::from_i32(1 << (bit_depth - 1));
            f.y.data.fill(mid);
            f.cb.data.fill(mid);
            f.cr.data.fill(mid);
            f.poc = poc;
            pics.push(DpbPic {
                frame: Arc::new(SharedFrame::new(f, poc, next_id, true)),
                poc,
                is_ref: true,
                long_term,
                needed_for_output: false,
                latency: 0,
                decode_index,
                crop,
                generated: true,
            });
            next_id += 1;
            pics.len() - 1
        };
        let mut warned = false;
        for (k, slot) in st_before_idx.iter_mut().enumerate() {
            if slot.is_none() {
                *slot = Some(generate(poc_st_curr_before[k], false, &mut self.pics));
                warned = true;
            }
        }
        for (k, slot) in st_after_idx.iter_mut().enumerate() {
            if slot.is_none() {
                *slot = Some(generate(poc_st_curr_after[k], false, &mut self.pics));
                warned = true;
            }
        }
        for (k, slot) in lt_curr_idx.iter_mut().enumerate() {
            if slot.is_none() {
                *slot = Some(generate(lt_curr[k].0, true, &mut self.pics));
                warned = true;
            }
        }
        self.next_id = next_id;
        if warned {
            self.warnings += 1;
        }
        let entries = |v: Vec<Option<usize>>, pics: &[DpbPic<S>]| -> Vec<RefEntry<S>> {
            v.into_iter().flatten().map(|i| RefEntry { frame: pics[i].frame.clone(), poc: pics[i].poc, long_term: pics[i].long_term }).collect()
        };
        RefSets { st_curr_before: entries(st_before_idx, &self.pics), st_curr_after: entries(st_after_idx, &self.pics), lt_curr: entries(lt_curr_idx, &self.pics) }
    }
}

impl<S: Sample> DpbPic<S> {
    /// The picture's id.
    pub fn id(&self) -> u64 {
        self.frame.id
    }
}

impl<S: Sample> Default for Dpb<S> {
    fn default() -> Self {
        Self::new()
    }
}
