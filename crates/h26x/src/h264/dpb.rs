//! Decoded picture buffer: picture order count (8.2.1), frame_num gaps
//! (8.2.5.2), reference picture marking (8.2.5), reference picture list
//! construction (8.2.4) and output ordering (Annex C bumping).
//!
//! Every entry is a frame buffer; a picture coded as two fields fills its
//! entry in two steps (the entry waits for its second field), and its
//! fields are marked and referenced separately, as 8.2.4 / 8.2.5 say.

use std::sync::Arc;

use super::frame::{PARITY_FRAME, SharedFrame};
use crate::sample::Sample;
use super::slice::{Mmco, RefListMod, SliceHeader, SliceType};
use super::sps::Sps;
use crate::picture::Picture;
use crate::{Error, Result};

/// How a stored picture (field) is marked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefMark {
    /// Not used for reference.
    Unused,
    /// Short-term reference.
    Short,
    /// Long-term reference.
    Long,
}

/// A frame in the DPB: a decoded frame, a complementary field pair, or a
/// non-paired field.
pub struct DecodedPic<S: Sample = u8> {
    /// The samples and motion (possibly still being decoded).
    pub frame: Arc<SharedFrame<S>>,
    /// PicOrderCnt of the frame: the smaller of its decoded fields' POCs.
    pub poc: i32,
    /// The POC of each field (of the decoded ones).
    pub field_poc: [i32; 2],
    /// Which fields are decoded: bit 0 top, bit 1 bottom (3: both, or a
    /// frame picture).
    pub fields: u8,
    /// `frame_num` (0 after MMCO 5).
    pub frame_num: u32,
    /// `FrameNumWrap`, computed per slice.
    pub frame_num_wrap: i32,
    /// `LongTermFrameIdx` when any field is long-term.
    pub long_term_frame_idx: u32,
    /// Marking per field (both entries alike for a frame picture).
    pub mark: [RefMark; 2],
    /// Still to be output.
    pub needed_for_output: bool,
    /// A first field whose second field has not arrived (not output yet;
    /// kept even when it is not a reference).
    pub awaiting_field: bool,
    /// Inferred by frame_num gap processing (never output).
    pub non_existing: bool,
    /// Decode order index.
    pub decode_index: u64,
}

impl<S: Sample> DecodedPic<S> {
    /// Any field marked short-term.
    #[inline]
    pub fn any_short(&self) -> bool {
        self.mark[0] == RefMark::Short || self.mark[1] == RefMark::Short
    }
    /// Any field marked long-term.
    #[inline]
    pub fn any_long(&self) -> bool {
        self.mark[0] == RefMark::Long || self.mark[1] == RefMark::Long
    }
    /// Any field marked as a reference.
    #[inline]
    pub fn is_ref(&self) -> bool {
        self.mark[0] != RefMark::Unused || self.mark[1] != RefMark::Unused
    }
    /// Both fields short-term (what a frame's reference list wants).
    #[inline]
    pub fn both_short(&self) -> bool {
        self.mark[0] == RefMark::Short && self.mark[1] == RefMark::Short
    }
    /// Both fields long-term.
    #[inline]
    pub fn both_long(&self) -> bool {
        self.mark[0] == RefMark::Long && self.mark[1] == RefMark::Long
    }
    /// The frame's mark for a frame reference (both fields agree).
    #[inline]
    pub fn frame_long(&self) -> bool {
        self.both_long()
    }
    fn set_all(&mut self, m: RefMark) {
        self.mark = [m, m];
    }
    /// The POC of the picture `parity` (a field's, or the frame's).
    #[inline]
    pub fn poc_of(&self, parity: u8) -> i32 {
        if parity == PARITY_FRAME { self.poc } else { self.field_poc[parity as usize] }
    }
    /// The mark of the picture `parity`.
    #[inline]
    pub fn mark_of(&self, parity: u8) -> RefMark {
        if parity == PARITY_FRAME { self.mark[0] } else { self.mark[parity as usize] }
    }
    /// Field `p` decoded?
    #[inline]
    pub fn has_field(&self, p: u8) -> bool {
        self.fields & (1 << p) != 0
    }
}

/// POC bookkeeping across pictures.
#[derive(Debug, Clone, Default)]
pub struct PocState {
    /// prevPicOrderCntMsb / Lsb (POC type 0).
    pub prev_msb: i32,
    /// See `prev_msb`.
    pub prev_lsb: i32,
    /// prevFrameNumOffset (types 1/2).
    pub prev_frame_num_offset: i32,
    /// frame_num of the previous picture in decoding order.
    pub prev_frame_num: u32,
    /// The previous reference picture's frame_num (`PrevRefFrameNum`).
    pub prev_ref_frame_num: u32,
    /// The previous picture had an MMCO 5.
    pub prev_had_mmco5: bool,
    /// The previous *reference* picture had an MMCO 5 (POC type 0 uses it).
    pub prev_ref_had_mmco5: bool,
    /// The previous reference picture's TopFieldOrderCnt after MMCO 5 (0
    /// when that picture was a bottom field).
    pub prev_ref_top_poc_after_mmco5: i32,
}

/// The POC of a picture: `(top, bottom)` for a frame; for a field picture
/// the field's POC in its slot (the other slot repeats it).
pub fn compute_poc(sps: &Sps, hdr: &SliceHeader, st: &mut PocState) -> (i32, i32) {
    let field = hdr.field_pic;
    let bottom_field = hdr.bottom_field;
    match sps.poc_type {
        0 => {
            let (prev_msb, prev_lsb) = if hdr.is_idr() {
                (0, 0)
            } else if st.prev_ref_had_mmco5 {
                (0, st.prev_ref_top_poc_after_mmco5)
            } else {
                (st.prev_msb, st.prev_lsb)
            };
            let max_lsb = sps.max_poc_lsb() as i32;
            let lsb = hdr.poc_lsb as i32;
            let msb = if lsb < prev_lsb && (prev_lsb - lsb) >= max_lsb / 2 {
                prev_msb + max_lsb
            } else if lsb > prev_lsb && (lsb - prev_lsb) > max_lsb / 2 {
                prev_msb - max_lsb
            } else {
                prev_msb
            };
            if hdr.is_reference() {
                st.prev_msb = msb;
                st.prev_lsb = lsb;
            }
            let v = msb + lsb;
            if field {
                (v, v)
            } else {
                (v, v + hdr.delta_poc_bottom)
            }
        }
        1 | 2 => {
            let prev_offset = if st.prev_had_mmco5 { 0 } else { st.prev_frame_num_offset };
            let frame_num_offset = if hdr.is_idr() {
                0
            } else if st.prev_frame_num > hdr.frame_num {
                prev_offset + sps.max_frame_num() as i32
            } else {
                prev_offset
            };
            st.prev_frame_num_offset = frame_num_offset;
            if sps.poc_type == 1 {
                let cycle = sps.offset_for_ref_frame.len() as i32;
                let mut abs_frame_num = if cycle != 0 { frame_num_offset + hdr.frame_num as i32 } else { 0 };
                if hdr.nal_ref_idc == 0 && abs_frame_num > 0 {
                    abs_frame_num -= 1;
                }
                let mut expected = 0i32;
                if abs_frame_num > 0 {
                    let cnt = (abs_frame_num - 1) / cycle;
                    let in_cycle = (abs_frame_num - 1) % cycle;
                    let delta_per_cycle: i32 = sps.offset_for_ref_frame.iter().sum();
                    expected = cnt * delta_per_cycle;
                    for i in 0..=in_cycle {
                        expected += sps.offset_for_ref_frame[i as usize];
                    }
                }
                if hdr.nal_ref_idc == 0 {
                    expected += sps.offset_for_non_ref_pic;
                }
                if !field {
                    let top = expected + hdr.delta_poc[0];
                    let bottom = top + sps.offset_for_top_to_bottom_field + hdr.delta_poc[1];
                    (top, bottom)
                } else if !bottom_field {
                    let top = expected + hdr.delta_poc[0];
                    (top, top)
                } else {
                    let bottom = expected + sps.offset_for_top_to_bottom_field + hdr.delta_poc[0];
                    (bottom, bottom)
                }
            } else {
                let temp = if hdr.is_idr() {
                    0
                } else if hdr.nal_ref_idc == 0 {
                    2 * (frame_num_offset + hdr.frame_num as i32) - 1
                } else {
                    2 * (frame_num_offset + hdr.frame_num as i32)
                };
                (temp, temp)
            }
        }
        _ => (0, 0),
    }
}

/// The DPB with its size limits.
pub struct Dpb<S: Sample = u8> {
    /// The pictures.
    pub pics: Vec<DecodedPic<S>>,
    /// Frame buffers the level / VUI allow.
    pub capacity: usize,
    /// Pictures that may precede one in output order (`num_reorder_frames`).
    pub num_reorder: usize,
    /// `MaxLongTermFrameIdx` (None = "no long-term frame indices").
    pub max_long_term_frame_idx: Option<u32>,
    /// Pictures ready to be handed out, in output order (each may still be
    /// decoding; collecting one waits for it).
    pub output: std::collections::VecDeque<PendingOutput<S>>,
    /// Cropping applied on output.
    pub crop: (u32, u32, u32, u32),
}

impl<S: Sample> Dpb<S> {
    /// An empty DPB.
    pub fn new() -> Self {
        Dpb {
            pics: Vec::new(),
            capacity: 16,
            num_reorder: 16,
            max_long_term_frame_idx: None,
            output: std::collections::VecDeque::new(),
            crop: (0, 0, 0, 0),
        }
    }

    /// Size the DPB from the SPS: VUI `max_dec_frame_buffering` when
    /// present, else the level's limit; the reorder depth from VUI when
    /// present, else the same as the size (output only when full).
    pub fn configure(&mut self, sps: &Sps) {
        let level_frames = sps.level_max_dpb_frames() as usize;
        let (mut cap, mut reorder) = (level_frames, level_frames);
        if let Some(vui) = &sps.vui {
            if let Some(m) = vui.max_dec_frame_buffering {
                cap = (m as usize).clamp(1, 16).max(sps.max_num_ref_frames as usize);
            }
            if let Some(r) = vui.max_num_reorder_frames {
                reorder = (r as usize).min(cap);
            } else {
                reorder = cap;
            }
        }
        // Never fewer buffers than references the stream may keep.
        cap = cap.max(sps.max_num_ref_frames as usize).clamp(1, 16);
        self.capacity = cap;
        self.num_reorder = reorder.min(cap);
        self.crop = sps.crop;
    }

    /// Number of frames with any field marked as reference.
    pub fn num_refs(&self) -> usize {
        self.pics.iter().filter(|p| p.is_ref()).count()
    }

    fn remove_unneeded(&mut self) {
        self.pics.retain(|p| p.is_ref() || p.needed_for_output || p.awaiting_field);
    }

    /// The entry holding `frame`, if any.
    pub fn index_of(&self, frame: &Arc<SharedFrame<S>>) -> Option<usize> {
        self.pics.iter().position(|p| Arc::ptr_eq(&p.frame, frame))
    }

    /// Output the picture with the smallest POC among those waiting.
    fn bump_one(&mut self) -> bool {
        let mut best: Option<usize> = None;
        for (i, p) in self.pics.iter().enumerate() {
            if p.needed_for_output && best.is_none_or(|b| p.poc < self.pics[b].poc) {
                best = Some(i);
            }
        }
        let Some(i) = best else { return false };
        let p = &mut self.pics[i];
        p.needed_for_output = false;
        if std::env::var_os("H26X_TRACE_DPB").is_some() {
            eprintln!("  bump poc {}", p.poc);
        }
        if !p.non_existing {
            self.output.push_back(PendingOutput { frame: p.frame.clone(), poc: p.poc, decode_index: p.decode_index, crop: self.crop });
        }
        self.remove_unneeded();
        true
    }

    /// Output every picture still waiting, in POC order.
    pub fn flush_output(&mut self) {
        while self.bump_one() {}
        self.remove_unneeded();
    }

    /// Discard everything (IDR with `no_output_of_prior_pics_flag`).
    pub fn clear(&mut self) {
        self.pics.clear();
    }

    /// A first field never got its second: the entry becomes a complete
    /// (non-paired) picture, output like any other.
    pub fn close_unpaired(&mut self, frame: &Arc<SharedFrame<S>>) {
        if let Some(i) = self.index_of(frame) {
            let p = &mut self.pics[i];
            if p.awaiting_field {
                p.awaiting_field = false;
                p.needed_for_output = !p.non_existing;
            }
        }
        // The buffer it holds may now be one too many.
        while self.pics.len() > self.capacity {
            if !self.bump_one() {
                break;
            }
        }
        self.reorder_bump();
    }

    /// FrameNumWrap of every short-term entry relative to `frame_num`.
    fn update_frame_num_wrap(&mut self, frame_num: u32, sps: &Sps) {
        let curr = frame_num as i32;
        let max = sps.max_frame_num() as i32;
        for p in &mut self.pics {
            if p.any_short() {
                p.frame_num_wrap = if p.frame_num as i32 > curr { p.frame_num as i32 - max } else { p.frame_num as i32 };
            }
        }
    }

    /// Store a decoded picture — a frame, or one field of a frame (its
    /// first field opens the entry, its second completes it) — marking it
    /// per `hdr` and outputting what the bumping rules say (C.4.5.3, plus
    /// the reorder-depth early output). `parity` is [`PARITY_FRAME`] or the
    /// field's.
    pub fn store(&mut self, mut pic: DecodedPic<S>, hdr: &SliceHeader, sps: &Sps, had_mmco5: bool, parity: u8) -> Result<()> {
        let field = parity != PARITY_FRAME;
        let trace = std::env::var_os("H26X_TRACE_DPB").is_some();
        // The second field of a frame already in the DPB?
        let second = if field { self.index_of(&pic.frame).filter(|&i| self.pics[i].awaiting_field) } else { None };
        if let Some(i) = second {
            // Complete the entry (it stays "awaiting" through the marking so
            // nothing below drops it before it is flagged for output).
            let e = &mut self.pics[i];
            e.fields |= 1 << parity;
            e.field_poc[parity as usize] = pic.field_poc[parity as usize];
            e.poc = e.field_poc[0].min(e.field_poc[1]);
        }
        self.update_frame_num_wrap(hdr.frame_num, sps);

        // Reference marking of the current picture (8.2.5.1).
        // The mark to give the current picture (its field, or both fields).
        let mut cur_mark = RefMark::Unused;
        let mut cur_lt_idx: Option<u32> = None;
        if hdr.is_reference() {
            if hdr.is_idr() {
                for p in &mut self.pics {
                    p.set_all(RefMark::Unused);
                }
                if hdr.marking.long_term_reference {
                    cur_mark = RefMark::Long;
                    cur_lt_idx = Some(0);
                    self.max_long_term_frame_idx = Some(0);
                } else {
                    cur_mark = RefMark::Short;
                    self.max_long_term_frame_idx = None;
                }
            } else if hdr.marking.adaptive {
                cur_mark = RefMark::Short;
                let (curr_pic_num, max_pic_num) = if field {
                    (2 * hdr.frame_num as i32 + 1, 2 * sps.max_frame_num() as i32)
                } else {
                    (hdr.frame_num as i32, sps.max_frame_num() as i32)
                };
                let _ = max_pic_num;
                // The pair partner (second field): MMCO 3 / 6 leave it alone
                // when it holds the index being assigned.
                let partner = second;
                for op in &hdr.marking.ops {
                    match *op {
                        Mmco::UnmarkShortTerm(diff) => {
                            let pic_num = curr_pic_num - (diff as i32 + 1);
                            if let Some((i, q)) = self.find_short(pic_num, field, parity) {
                                self.pics[i].mark[q] = RefMark::Unused;
                                if !field {
                                    self.pics[i].mark[1 - q] = RefMark::Unused;
                                }
                            }
                        }
                        Mmco::UnmarkLongTerm(lt) => {
                            if let Some((i, q)) = self.find_long(lt as i32, field, parity) {
                                self.pics[i].mark[q] = RefMark::Unused;
                                if !field {
                                    self.pics[i].mark[1 - q] = RefMark::Unused;
                                }
                            }
                        }
                        Mmco::ShortToLong(diff, idx) => {
                            let pic_num = curr_pic_num - (diff as i32 + 1);
                            if let Some((i, q)) = self.find_short(pic_num, field, parity) {
                                self.release_long_term_idx(idx, Some(i));
                                let p = &mut self.pics[i];
                                p.mark[q] = RefMark::Long;
                                if !field {
                                    p.mark[1 - q] = RefMark::Long;
                                }
                                p.long_term_frame_idx = idx;
                            }
                        }
                        Mmco::MaxLongTermIdx(plus1) => {
                            self.max_long_term_frame_idx = if plus1 == 0 { None } else { Some(plus1 - 1) };
                            for p in &mut self.pics {
                                if p.any_long() && (plus1 == 0 || p.long_term_frame_idx > plus1 - 1) {
                                    for q in 0..2 {
                                        if p.mark[q] == RefMark::Long {
                                            p.mark[q] = RefMark::Unused;
                                        }
                                    }
                                }
                            }
                        }
                        Mmco::UnmarkAll => {
                            for p in &mut self.pics {
                                p.set_all(RefMark::Unused);
                            }
                            self.max_long_term_frame_idx = None;
                        }
                        Mmco::CurrentToLong(idx) => {
                            self.release_long_term_idx(idx, partner);
                            cur_mark = RefMark::Long;
                            cur_lt_idx = Some(idx);
                        }
                    }
                }
            } else {
                // Sliding window (8.2.5.3): not for the second field of a
                // reference pair whose first field is short-term.
                let first_short = second.is_some_and(|i| self.pics[i].mark[1 - parity as usize] == RefMark::Short);
                if !first_short {
                    self.sliding_window(sps, second);
                }
                cur_mark = RefMark::Short;
            }
        }
        if trace {
            eprintln!("  marking: idr {} adaptive {} ops {:?} mmco5 {} parity {parity} second {:?}", hdr.is_idr(), hdr.marking.adaptive, hdr.marking.ops, had_mmco5, second);
        }
        // Apply the current picture's mark.
        {
            let target: &mut DecodedPic<S> = match second {
                Some(i) => &mut self.pics[i],
                None => &mut pic,
            };
            if field {
                target.mark[parity as usize] = cur_mark;
            } else {
                target.set_all(cur_mark);
            }
            if let Some(idx) = cur_lt_idx {
                target.long_term_frame_idx = idx;
            }
        }
        if had_mmco5 {
            // Everything before is output before the current picture.
            self.flush_output();
        }
        // Make room: an IDR (without no_output_of_prior_pics) or a full DPB
        // bumps until there is space.
        if hdr.is_idr() {
            if hdr.marking.no_output_of_prior_pics {
                self.pics.clear();
            } else {
                self.flush_output();
                self.pics.clear();
            }
        }
        self.remove_unneeded();
        if let Some(i) = second {
            // The buffer is already held; the frame is complete now.
            let e = &mut self.pics[i];
            e.awaiting_field = false;
            e.needed_for_output = !e.non_existing;
            if trace {
                let (poc, fnum, marks) = (e.poc, e.frame_num, e.mark);
                eprintln!("store 2nd field poc {} fn {} marks {:?} | dpb: {:?}", poc, fnum, marks, self.pics.iter().map(|p| (p.poc, p.mark[0] as u8, p.mark[1] as u8, p.needed_for_output as u8)).collect::<Vec<_>>());
            }
            self.reorder_bump();
            return Ok(());
        }
        if trace {
            eprintln!(
                "  before capacity loop: len {} cap {} list {:?}",
                self.pics.len(),
                self.capacity,
                self.pics.iter().map(|p| (p.poc, p.mark[0] as u8, p.mark[1] as u8, p.needed_for_output as u8)).collect::<Vec<_>>()
            );
        }
        // C.4.5.2: a non-reference frame with no free frame buffer is output
        // directly once its POC is the lowest among those waiting — it never
        // needs storing. (A first field needs its buffer for the second.)
        if !hdr.is_reference() && !hdr.is_idr() && !field {
            while self.pics.len() >= self.capacity {
                let min_waiting = self.pics.iter().filter(|p| p.needed_for_output).map(|p| p.poc).min();
                match min_waiting {
                    Some(m) if pic.poc > m => {
                        self.bump_one();
                    }
                    _ => {
                        if !pic.non_existing {
                            self.output.push_back(PendingOutput { frame: pic.frame.clone(), poc: pic.poc, decode_index: pic.decode_index, crop: self.crop });
                        }
                        return Ok(());
                    }
                }
            }
        }
        while self.pics.len() >= self.capacity {
            if !self.bump_one() {
                // Full of references nobody outputs: drop the oldest
                // non-existing / unused first, else give up gracefully.
                if let Some(i) = self.pics.iter().position(|p| p.non_existing) {
                    self.pics.remove(i);
                    continue;
                }
                // Over-full with references (a broken stream): evict the
                // oldest short-term reference.
                if let Some((i, _)) = self
                    .pics
                    .iter()
                    .enumerate()
                    .filter(|(_, p)| p.any_short() && !p.awaiting_field)
                    .min_by_key(|(_, p)| p.decode_index)
                {
                    self.pics.remove(i);
                    continue;
                }
                return Err(Error::bitstream("DPB overflow"));
            }
        }
        pic.awaiting_field = field;
        pic.needed_for_output = !pic.non_existing && !field;
        if trace {
            eprintln!(
                "store poc {} fn {} marks {:?} fields {} | cap {} reorder {} | dpb: {:?}",
                pic.poc,
                pic.frame_num,
                pic.mark,
                pic.fields,
                self.capacity,
                self.num_reorder,
                self.pics.iter().map(|p| (p.poc, p.mark[0] as u8, p.mark[1] as u8, p.needed_for_output as u8)).collect::<Vec<_>>()
            );
        }
        self.pics.push(pic);
        self.reorder_bump();
        Ok(())
    }

    /// Reorder-depth early output.
    fn reorder_bump(&mut self) {
        loop {
            let waiting = self.pics.iter().filter(|p| p.needed_for_output).count();
            if waiting > self.num_reorder {
                if !self.bump_one() {
                    break;
                }
            } else {
                break;
            }
        }
    }

    /// The short-term picture with PicNum `pic_num`: for a frame picture the
    /// entry with that FrameNumWrap (both fields short-term); for a field
    /// picture the field whose PicNum is `2 * FrameNumWrap + 1` (same
    /// parity as the current field) or `2 * FrameNumWrap` (opposite).
    /// Returns `(entry, field)`.
    fn find_short(&self, pic_num: i32, field: bool, cur_parity: u8) -> Option<(usize, usize)> {
        if !field {
            self.pics.iter().position(|p| p.both_short() && p.frame_num_wrap == pic_num && !p.awaiting_field).map(|i| (i, 0))
                .or_else(|| self.pics.iter().position(|p| p.any_short() && p.frame_num_wrap == pic_num).map(|i| (i, 0)))
        } else {
            let same = pic_num & 1 == 1;
            let q = if same { cur_parity as usize } else { 1 - cur_parity as usize };
            let fnw = pic_num >> 1;
            self.pics.iter().position(|p| p.mark[q] == RefMark::Short && p.has_field(q as u8) && p.frame_num_wrap == fnw).map(|i| (i, q))
        }
    }

    /// The long-term picture with LongTermPicNum `lt_pic_num`, as
    /// [`Self::find_short`].
    fn find_long(&self, lt_pic_num: i32, field: bool, cur_parity: u8) -> Option<(usize, usize)> {
        if !field {
            self.pics.iter().position(|p| p.any_long() && p.long_term_frame_idx as i32 == lt_pic_num).map(|i| (i, 0))
        } else {
            let same = lt_pic_num & 1 == 1;
            let q = if same { cur_parity as usize } else { 1 - cur_parity as usize };
            let idx = lt_pic_num >> 1;
            self.pics.iter().position(|p| p.mark[q] == RefMark::Long && p.has_field(q as u8) && p.long_term_frame_idx as i32 == idx).map(|i| (i, q))
        }
    }

    /// Free LongTermFrameIdx `idx` for a new holder: every other frame's
    /// long-term fields with that index become unused (8.2.5.4.3 /
    /// 8.2.5.4.6), `keep` (the frame receiving it) excepted.
    fn release_long_term_idx(&mut self, idx: u32, keep: Option<usize>) {
        for (i, p) in self.pics.iter_mut().enumerate() {
            if Some(i) == keep {
                continue;
            }
            if p.any_long() && p.long_term_frame_idx == idx {
                for q in 0..2 {
                    if p.mark[q] == RefMark::Long {
                        p.mark[q] = RefMark::Unused;
                    }
                }
            }
        }
    }

    /// Sliding window marking (8.2.5.3). `current` is the entry of the
    /// current picture (a second field), never the one unmarked.
    pub fn sliding_window(&mut self, sps: &Sps, current: Option<usize>) {
        let max_refs = sps.max_num_ref_frames.max(1) as usize;
        loop {
            let num_short = self.pics.iter().filter(|p| p.any_short()).count();
            let num_long = self.pics.iter().filter(|p| p.any_long()).count();
            if num_short + num_long < max_refs {
                break;
            }
            // Unmark the short-term frame with the smallest FrameNumWrap.
            let mut best: Option<usize> = None;
            for (i, p) in self.pics.iter().enumerate() {
                if Some(i) == current {
                    continue;
                }
                if p.any_short() && best.is_none_or(|b| p.frame_num_wrap < self.pics[b].frame_num_wrap) {
                    best = Some(i);
                }
            }
            match best {
                Some(i) => {
                    for q in 0..2 {
                        if self.pics[i].mark[q] == RefMark::Short {
                            self.pics[i].mark[q] = RefMark::Unused;
                        }
                    }
                }
                None => break,
            }
        }
        self.remove_unneeded();
    }

    /// Insert "non-existing" frames for a gap in frame_num (8.2.5.2).
    /// `grey` is a complete grey frame of the right size to stand in.
    pub fn fill_frame_num_gap(&mut self, sps: &Sps, prev_ref_frame_num: u32, frame_num: u32, grey: &Arc<SharedFrame<S>>, decode_index: &mut u64) {
        let max = sps.max_frame_num();
        let mut unused = (prev_ref_frame_num + 1) % max;
        let mut guard = 0;
        while unused != frame_num && guard < 64 {
            guard += 1;
            // FrameNumWrap of the existing short-terms relative to this frame_num.
            self.update_frame_num_wrap(unused, sps);
            self.sliding_window(sps, None);
            self.remove_unneeded();
            while self.pics.len() >= self.capacity {
                if !self.bump_one() {
                    break;
                }
            }
            // A grey picture: what a decoder is expected to show if this ever
            // gets referenced.
            let pic = DecodedPic {
                frame: grey.clone(),
                poc: 0,
                field_poc: [0, 0],
                fields: 3,
                frame_num: unused,
                frame_num_wrap: unused as i32,
                long_term_frame_idx: 0,
                mark: [RefMark::Short; 2],
                needed_for_output: false,
                awaiting_field: false,
                non_existing: true,
                decode_index: *decode_index,
            };
            *decode_index += 1;
            self.pics.push(pic);
            unused = (unused + 1) % max;
        }
    }
}

impl<S: Sample> Default for Dpb<S> {
    fn default() -> Self {
        Self::new()
    }
}

/// Reference picture lists for a slice: `(index into dpb.pics, parity)`
/// per entry — the parity is [`PARITY_FRAME`] for a frame's lists, a
/// field's for a field's.
pub struct RefLists {
    /// List 0 and list 1.
    pub lists: [Vec<(usize, u8)>; 2],
}

/// A missing entry (the modification named a picture that is not there).
pub const MISSING_REF: (usize, u8) = (usize::MAX, PARITY_FRAME);

/// 8.2.4.2.5: the fields of the frames `frames` (in order), alternating
/// parity starting with `cur_parity`, taking only fields that are decoded
/// and marked `want`.
fn alternate_fields<S: Sample>(dpb: &Dpb<S>, frames: &[usize], cur_parity: u8, want: RefMark) -> Vec<(usize, u8)> {
    let pick = |q: u8| -> Vec<(usize, u8)> {
        frames.iter().copied().filter(|&i| dpb.pics[i].has_field(q) && dpb.pics[i].mark[q as usize] == want).map(|i| (i, q)).collect()
    };
    let same = pick(cur_parity);
    let opp = pick(1 - cur_parity);
    let mut out = Vec::with_capacity(same.len() + opp.len());
    let (mut a, mut b) = (0usize, 0usize);
    loop {
        if a < same.len() {
            out.push(same[a]);
            a += 1;
        }
        if b < opp.len() {
            out.push(opp[b]);
            b += 1;
        }
        if a >= same.len() && b >= opp.len() {
            break;
        }
    }
    out
}

/// Build the reference picture lists for a P or B slice (8.2.4).
/// `cur_parity` is [`PARITY_FRAME`] for a frame picture.
pub fn build_ref_lists<S: Sample>(dpb: &mut Dpb<S>, sps: &Sps, hdr: &SliceHeader, cur_poc: i32, cur_parity: u8) -> Result<RefLists> {
    let field = cur_parity != PARITY_FRAME;
    let max_frame_num = sps.max_frame_num() as i32;
    dpb.update_frame_num_wrap(hdr.frame_num, sps);
    let (curr_pic_num, max_pic_num) = if field { (2 * hdr.frame_num as i32 + 1, 2 * max_frame_num) } else { (hdr.frame_num as i32, max_frame_num) };

    let mut lists: [Vec<(usize, u8)>; 2] = [Vec::new(), Vec::new()];
    if !field {
        // Frames: entries with both fields marked alike.
        let shorts: Vec<usize> = dpb.pics.iter().enumerate().filter(|(_, p)| p.both_short() && !p.awaiting_field).map(|(i, _)| i).collect();
        let mut longs: Vec<usize> = dpb.pics.iter().enumerate().filter(|(_, p)| p.both_long() && !p.awaiting_field).map(|(i, _)| i).collect();
        longs.sort_by_key(|&i| dpb.pics[i].long_term_frame_idx);
        match hdr.slice_type {
            SliceType::P | SliceType::Sp => {
                let mut s = shorts.clone();
                s.sort_by(|&a, &b| dpb.pics[b].frame_num_wrap.cmp(&dpb.pics[a].frame_num_wrap));
                lists[0].extend(s.iter().map(|&i| (i, PARITY_FRAME)));
                lists[0].extend(longs.iter().map(|&i| (i, PARITY_FRAME)));
            }
            SliceType::B => {
                let mut before: Vec<usize> = shorts.iter().copied().filter(|&i| dpb.pics[i].poc < cur_poc).collect();
                let mut after: Vec<usize> = shorts.iter().copied().filter(|&i| dpb.pics[i].poc > cur_poc).collect();
                before.sort_by(|&a, &b| dpb.pics[b].poc.cmp(&dpb.pics[a].poc));
                after.sort_by_key(|&i| dpb.pics[i].poc);
                lists[0].extend(before.iter().map(|&i| (i, PARITY_FRAME)));
                lists[0].extend(after.iter().map(|&i| (i, PARITY_FRAME)));
                lists[0].extend(longs.iter().map(|&i| (i, PARITY_FRAME)));
                lists[1].extend(after.iter().map(|&i| (i, PARITY_FRAME)));
                lists[1].extend(before.iter().map(|&i| (i, PARITY_FRAME)));
                lists[1].extend(longs.iter().map(|&i| (i, PARITY_FRAME)));
                if lists[1].len() > 1 && lists[0] == lists[1] {
                    lists[1].swap(0, 1);
                }
            }
            _ => {}
        }
    } else {
        // Fields (8.2.4.2.2 / 8.2.4.2.4 / 8.2.4.2.5): frames with any field
        // marked, ordered as frames, then their fields alternating parity.
        let shorts: Vec<usize> = dpb.pics.iter().enumerate().filter(|(_, p)| p.any_short()).map(|(i, _)| i).collect();
        let mut longs: Vec<usize> = dpb.pics.iter().enumerate().filter(|(_, p)| p.any_long()).map(|(i, _)| i).collect();
        longs.sort_by_key(|&i| dpb.pics[i].long_term_frame_idx);
        match hdr.slice_type {
            SliceType::P | SliceType::Sp => {
                let mut s = shorts.clone();
                s.sort_by(|&a, &b| dpb.pics[b].frame_num_wrap.cmp(&dpb.pics[a].frame_num_wrap));
                lists[0].extend(alternate_fields(dpb, &s, cur_parity, RefMark::Short));
                lists[0].extend(alternate_fields(dpb, &longs, cur_parity, RefMark::Long));
            }
            SliceType::B => {
                let mut before: Vec<usize> = shorts.iter().copied().filter(|&i| dpb.pics[i].poc <= cur_poc).collect();
                let mut after: Vec<usize> = shorts.iter().copied().filter(|&i| dpb.pics[i].poc > cur_poc).collect();
                before.sort_by(|&a, &b| dpb.pics[b].poc.cmp(&dpb.pics[a].poc));
                after.sort_by_key(|&i| dpb.pics[i].poc);
                let mut f0 = before.clone();
                f0.extend(after.iter().copied());
                let mut f1 = after.clone();
                f1.extend(before.iter().copied());
                lists[0].extend(alternate_fields(dpb, &f0, cur_parity, RefMark::Short));
                lists[0].extend(alternate_fields(dpb, &longs, cur_parity, RefMark::Long));
                lists[1].extend(alternate_fields(dpb, &f1, cur_parity, RefMark::Short));
                lists[1].extend(alternate_fields(dpb, &longs, cur_parity, RefMark::Long));
                if lists[1].len() > 1 && lists[0] == lists[1] {
                    lists[1].swap(0, 1);
                }
            }
            _ => {}
        }
    }

    // PicNum / LongTermPicNum of an entry (frame or field), for the
    // modification process; None when it is not marked that way.
    let pic_num_of = |e: (usize, u8)| -> Option<i32> {
        let p = &dpb.pics[e.0];
        if !field {
            if p.both_short() { Some(p.frame_num_wrap) } else { None }
        } else if p.mark[e.1 as usize] == RefMark::Short {
            Some(2 * p.frame_num_wrap + (e.1 == cur_parity) as i32)
        } else {
            None
        }
    };
    let lt_pic_num_of = |e: (usize, u8)| -> Option<i32> {
        let p = &dpb.pics[e.0];
        if !field {
            if p.both_long() { Some(p.long_term_frame_idx as i32) } else { None }
        } else if p.mark[e.1 as usize] == RefMark::Long {
            Some(2 * p.long_term_frame_idx as i32 + (e.1 == cur_parity) as i32)
        } else {
            None
        }
    };
    // The entry with a given PicNum / LongTermPicNum.
    let find_pic_num = |pic_num: i32| -> Option<(usize, u8)> {
        if !field {
            dpb.pics.iter().position(|p| p.both_short() && !p.awaiting_field && p.frame_num_wrap == pic_num).map(|i| (i, PARITY_FRAME))
        } else {
            let q = if pic_num & 1 == 1 { cur_parity } else { 1 - cur_parity };
            let fnw = pic_num >> 1;
            dpb.pics.iter().position(|p| p.has_field(q) && p.mark[q as usize] == RefMark::Short && p.frame_num_wrap == fnw).map(|i| (i, q))
        }
    };
    let find_lt_pic_num = |lt: i32| -> Option<(usize, u8)> {
        if !field {
            dpb.pics.iter().position(|p| p.both_long() && !p.awaiting_field && p.long_term_frame_idx as i32 == lt).map(|i| (i, PARITY_FRAME))
        } else {
            let q = if lt & 1 == 1 { cur_parity } else { 1 - cur_parity };
            let idx = lt >> 1;
            dpb.pics.iter().position(|p| p.has_field(q) && p.mark[q as usize] == RefMark::Long && p.long_term_frame_idx as i32 == idx).map(|i| (i, q))
        }
    };

    for l in 0..2 {
        let n = hdr.num_ref_idx_active[l] as usize;
        if n == 0 {
            lists[l].clear();
            continue;
        }
        // Modification (8.2.4.3), on a list one longer than active.
        if !hdr.ref_list_mods[l].is_empty() {
            let mut list: Vec<Option<(usize, u8)>> = lists[l].iter().map(|&i| Some(i)).collect();
            list.resize(n + 1, None);
            let mut pic_num_pred = curr_pic_num;
            let mut ref_idx = 0usize;
            for m in &hdr.ref_list_mods[l] {
                match *m {
                    RefListMod::SubtractPicNum(d) | RefListMod::AddPicNum(d) => {
                        let d = d as i32;
                        let mut no_wrap = if matches!(m, RefListMod::SubtractPicNum(_)) { pic_num_pred - d } else { pic_num_pred + d };
                        if no_wrap < 0 {
                            no_wrap += max_pic_num;
                        } else if no_wrap >= max_pic_num {
                            no_wrap -= max_pic_num;
                        }
                        pic_num_pred = no_wrap;
                        let pic_num = if no_wrap > curr_pic_num { no_wrap - max_pic_num } else { no_wrap };
                        let Some(t) = find_pic_num(pic_num) else {
                            return Err(Error::bitstream(format!("ref_pic_list_modification names a missing short-term picture (PicNum {pic_num})")));
                        };
                        // Insert and remove the duplicate.
                        for c in (ref_idx + 1..=n).rev() {
                            list[c] = list[c - 1];
                        }
                        list[ref_idx] = Some(t);
                        ref_idx += 1;
                        let mut n_idx = ref_idx;
                        for c in ref_idx..=n {
                            let keep = match list[c] {
                                Some(e) => pic_num_of(e) != Some(pic_num),
                                None => true,
                            };
                            if keep {
                                list[n_idx] = list[c];
                                n_idx += 1;
                            }
                        }
                        for c in n_idx..=n {
                            list[c] = None;
                        }
                    }
                    RefListMod::LongTerm(lt) => {
                        let Some(t) = find_lt_pic_num(lt as i32) else {
                            return Err(Error::bitstream(format!("ref_pic_list_modification names a missing long-term picture ({lt})")));
                        };
                        for c in (ref_idx + 1..=n).rev() {
                            list[c] = list[c - 1];
                        }
                        list[ref_idx] = Some(t);
                        ref_idx += 1;
                        let mut n_idx = ref_idx;
                        for c in ref_idx..=n {
                            let keep = match list[c] {
                                Some(e) => lt_pic_num_of(e) != Some(lt as i32),
                                None => true,
                            };
                            if keep {
                                list[n_idx] = list[c];
                                n_idx += 1;
                            }
                        }
                        for c in n_idx..=n {
                            list[c] = None;
                        }
                    }
                }
            }
            lists[l] = list[..n].iter().map(|x| x.unwrap_or(MISSING_REF)).collect();
        } else {
            lists[l].truncate(n);
            while lists[l].len() < n {
                lists[l].push(MISSING_REF);
            }
        }
    }
    Ok(RefLists { lists })
}

/// A picture bumped for output, waiting to be collected.
pub struct PendingOutput<S: Sample = u8> {
    /// The picture.
    pub frame: Arc<SharedFrame<S>>,
    /// POC.
    pub poc: i32,
    /// Decode order index.
    pub decode_index: u64,
    /// Cropping window.
    pub crop: (u32, u32, u32, u32),
}

impl<S: Sample> PendingOutput<S> {
    /// Wait for the picture to finish and copy it out.
    pub fn into_picture(self) -> Picture {
        let f = self.frame.wait_and_get();
        f.to_picture(self.crop, self.poc, self.decode_index)
    }
}

/// A resolved reference: what a slice needs to know about one entry of its
/// reference picture list.
pub struct RefEntry<S: Sample = u8> {
    /// The picture (possibly still decoding).
    pub frame: Arc<SharedFrame<S>>,
    /// POC (of the field, for a field entry).
    pub poc: i32,
    /// Long-term?
    pub long_term: bool,
    /// Which picture of the frame: 0 / 1 a field, [`PARITY_FRAME`] the frame.
    pub parity: u8,
}

impl<S: Sample> Clone for RefEntry<S> {
    fn clone(&self) -> Self {
        RefEntry { frame: self.frame.clone(), poc: self.poc, long_term: self.long_term, parity: self.parity }
    }
}
