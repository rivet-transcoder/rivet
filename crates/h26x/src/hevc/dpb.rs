//! The decoded picture buffer: reference picture set marking (8.3.2),
//! generation of unavailable references (8.3.3), reference picture list
//! construction (8.3.4) and output-order bumping (C.5.2).

use std::collections::VecDeque;

use crate::picture::{ChromaFormat, Picture};

use super::frame::Frame;
use super::slice::SliceHeader;
use super::sps::Sps;

/// A picture held in the DPB.
pub struct DpbPic {
    /// Samples and motion.
    pub frame: Frame,
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
    /// Unique id (stable while indices shift as pictures leave).
    pub id: u64,
}

/// The reference picture set of the current picture, as DPB picture ids
/// (indices shift when pictures are bumped out, ids do not).
#[derive(Debug, Default, Clone)]
pub struct RefSets {
    /// `RefPicSetStCurrBefore`.
    pub st_curr_before: Vec<u64>,
    /// `RefPicSetStCurrAfter`.
    pub st_curr_after: Vec<u64>,
    /// `RefPicSetLtCurr`.
    pub lt_curr: Vec<u64>,
}

impl RefSets {
    /// `NumPicTotalCurr`.
    pub fn num_pic_total_curr(&self) -> usize {
        self.st_curr_before.len() + self.st_curr_after.len() + self.lt_curr.len()
    }
}

/// The DPB with its output queue.
pub struct Dpb {
    /// Pictures.
    pub pics: Vec<DpbPic>,
    /// Pictures output but not yet collected.
    pub output: VecDeque<Picture>,
    max_num_reorder: u32,
    /// `SpsMaxLatencyPictures` (None = no limit).
    max_latency: Option<u32>,
    max_dec_pic_buffering: u32,
    /// Warnings (missing references generated).
    pub warnings: u64,
    next_id: u64,
}

impl Dpb {
    /// Empty.
    pub fn new() -> Self {
        Dpb { pics: Vec::new(), output: VecDeque::new(), max_num_reorder: 0, max_latency: None, max_dec_pic_buffering: 1, warnings: 0, next_id: 1 }
    }

    /// A fresh picture id.
    pub fn alloc_id(&mut self) -> u64 {
        self.next_id += 1;
        self.next_id - 1
    }

    /// Index of the picture with `id`, if still present.
    pub fn index_of(&self, id: u64) -> Option<usize> {
        self.pics.iter().position(|p| p.id == id)
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
            let pic = p.frame.to_picture(p.crop, p.poc, p.decode_index);
            self.output.push_back(pic);
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
                // Full of references only: nothing to bump.
                break;
            }
        }
    }

    /// C.5.2.3: store the decoded picture and do the additional bumping.
    pub fn store(&mut self, mut pic: DpbPic, pic_output: bool) {
        if pic_output {
            for p in &mut self.pics {
                if p.needed_for_output && p.poc > pic.poc {
                    p.latency += 1;
                }
            }
        }
        pic.needed_for_output = pic_output;
        pic.latency = 0;
        pic.is_ref = true;
        pic.long_term = false;
        self.pics.push(pic);
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

    /// Drop everything (a new sequence with no output of prior pictures, or
    /// a reset).
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
    ) -> RefSets {
        if idr {
            for p in &mut self.pics {
                p.is_ref = false;
                p.long_term = false;
            }
            return RefSets::default();
        }
        let max_poc_lsb = sps.max_poc_lsb();
        // POC lists.
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
        // Long-term: (poc, msb_present, used).
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
        // Long-term entries first (they may steal a picture from the
        // short-term candidates).
        let find_lt = |pics: &[DpbPic], poc: i32, msb_present: bool| -> Option<usize> {
            pics.iter().position(|p| {
                p.is_ref
                    && if msb_present { p.poc == poc } else { (p.poc & (max_poc_lsb - 1)) == (poc & (max_poc_lsb - 1)) }
            })
        };
        let mut lt_curr_idx: Vec<Option<usize>> = Vec::new();
        let mut lt_foll_idx: Vec<Option<usize>> = Vec::new();
        for &(poc, msb) in &lt_curr {
            lt_curr_idx.push(find_lt(&self.pics, poc, msb));
        }
        for &(poc, msb) in &lt_foll {
            lt_foll_idx.push(find_lt(&self.pics, poc, msb));
        }
        let find_st = |pics: &[DpbPic], poc: i32| -> Option<usize> { pics.iter().position(|p| p.is_ref && !p.long_term && p.poc == poc) };
        let mut st_before_idx: Vec<Option<usize>> = poc_st_curr_before.iter().map(|&p| find_st(&self.pics, p)).collect();
        let mut st_after_idx: Vec<Option<usize>> = poc_st_curr_after.iter().map(|&p| find_st(&self.pics, p)).collect();
        let st_foll_idx: Vec<Option<usize>> = poc_st_foll.iter().map(|&p| find_st(&self.pics, p)).collect();

        // Marking: everything in the RPS keeps its status (long-term ones
        // become long-term); everything else becomes unused.
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
                p.frame.long_term = true;
            }
        }
        // Generate missing "Curr" references (8.3.3.2); Foll ones may be
        // absent legitimately.
        let mut next_id = self.next_id;
        let mut generate = |poc: i32, long_term: bool, pics: &mut Vec<DpbPic>| -> usize {
            let mut f = Frame::new(sps.width as usize, sps.height as usize, chroma, bit_depth);
            let mid = 1u16 << (bit_depth - 1);
            f.y.data.fill(mid);
            f.cb.data.fill(mid);
            f.cr.data.fill(mid);
            f.poc = poc;
            f.long_term = long_term;
            pics.push(DpbPic {
                frame: f,
                poc,
                is_ref: true,
                long_term,
                needed_for_output: false,
                latency: 0,
                decode_index,
                crop,
                generated: true,
                id: next_id,
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
        let ids = |v: Vec<Option<usize>>, pics: &[DpbPic]| -> Vec<u64> { v.into_iter().flatten().map(|i| pics[i].id).collect() };
        RefSets {
            st_curr_before: ids(st_before_idx, &self.pics),
            st_curr_after: ids(st_after_idx, &self.pics),
            lt_curr: ids(lt_curr_idx, &self.pics),
        }
    }

    /// 8.3.4: build `RefPicList0/1` for a slice as DPB indices.
    pub fn build_ref_lists(&self, hdr: &SliceHeader, sets: &RefSets) -> Result<[Vec<usize>; 2], crate::Error> {
        let total = sets.num_pic_total_curr();
        let mut out: [Vec<usize>; 2] = [Vec::new(), Vec::new()];
        let nlists = if hdr.slice_type.is_b() { 2 } else { 1 };
        for list in 0..nlists {
            let n_active = hdr.num_ref_idx[list] as usize;
            if n_active == 0 {
                continue;
            }
            if total == 0 {
                return Err(crate::Error::bitstream("P/B slice with an empty reference picture set"));
            }
            let num_temp = n_active.max(total);
            let mut temp = Vec::with_capacity(num_temp);
            let (first, second) = if list == 0 { (&sets.st_curr_before, &sets.st_curr_after) } else { (&sets.st_curr_after, &sets.st_curr_before) };
            while temp.len() < num_temp {
                for &i in first {
                    if temp.len() < num_temp {
                        temp.push(i);
                    }
                }
                for &i in second {
                    if temp.len() < num_temp {
                        temp.push(i);
                    }
                }
                for &i in &sets.lt_curr {
                    if temp.len() < num_temp {
                        temp.push(i);
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
                let pic_idx = self.index_of(temp[idx]).ok_or_else(|| crate::Error::bitstream("reference picture left the DPB"))?;
                l.push(pic_idx);
            }
            out[list] = l;
        }
        Ok(out)
    }
}

impl Default for Dpb {
    fn default() -> Self {
        Self::new()
    }
}
