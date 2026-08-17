//! Decoded picture buffer: picture order count (8.2.1), frame_num gaps
//! (8.2.5.2), reference picture marking (8.2.5), reference picture list
//! construction (8.2.4) and output ordering (Annex C bumping).
//!
//! Frames only (progressive) — a field picture never reaches here.

use super::frame::Frame;
use super::slice::{Mmco, RefListMod, SliceHeader, SliceType};
use super::sps::Sps;
use crate::picture::Picture;
use crate::{Error, Result};

/// How a stored picture is marked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefMark {
    /// Not used for reference.
    Unused,
    /// Short-term reference.
    Short,
    /// Long-term reference.
    Long,
}

/// A picture in the DPB.
pub struct DecodedPic {
    /// The samples and motion.
    pub frame: Frame,
    /// PicOrderCnt (frames: min of the field POCs).
    pub poc: i32,
    /// `frame_num` (0 after MMCO 5).
    pub frame_num: u32,
    /// `FrameNumWrap`, computed per slice.
    pub frame_num_wrap: i32,
    /// `LongTermFrameIdx` when long-term.
    pub long_term_frame_idx: u32,
    /// Marking.
    pub mark: RefMark,
    /// Still to be output.
    pub needed_for_output: bool,
    /// Inferred by frame_num gap processing (never output).
    pub non_existing: bool,
    /// Decode order index.
    pub decode_index: u64,
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
    /// The previous reference picture's TopFieldOrderCnt after MMCO 5.
    pub prev_ref_top_poc_after_mmco5: i32,
}

/// The POC of a picture (frame): `(top, bottom)`.
pub fn compute_poc(sps: &Sps, hdr: &SliceHeader, st: &mut PocState) -> (i32, i32) {
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
            let top = msb + lsb;
            let bottom = top + hdr.delta_poc_bottom;
            if hdr.is_reference() {
                st.prev_msb = msb;
                st.prev_lsb = lsb;
            }
            (top, bottom)
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
                let top = expected + hdr.delta_poc[0];
                let bottom = top + sps.offset_for_top_to_bottom_field + hdr.delta_poc[1];
                (top, bottom)
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
pub struct Dpb {
    /// The pictures.
    pub pics: Vec<DecodedPic>,
    /// Frame buffers the level / VUI allow.
    pub capacity: usize,
    /// Pictures that may precede one in output order (`num_reorder_frames`).
    pub num_reorder: usize,
    /// `MaxLongTermFrameIdx` (None = "no long-term frame indices").
    pub max_long_term_frame_idx: Option<u32>,
    /// Pictures ready to be handed out, in output order.
    pub output: std::collections::VecDeque<Picture>,
    /// Cropping applied on output.
    pub crop: (u32, u32, u32, u32),
}

impl Dpb {
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

    /// Number of pictures marked as reference.
    pub fn num_refs(&self) -> usize {
        self.pics.iter().filter(|p| p.mark != RefMark::Unused).count()
    }

    fn remove_unneeded(&mut self) {
        self.pics.retain(|p| p.mark != RefMark::Unused || p.needed_for_output);
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
            let pic = p.frame.to_picture(self.crop, p.poc, p.decode_index);
            self.output.push_back(pic);
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

    /// Store a decoded picture, marking it per `hdr`, and output what the
    /// bumping rules say (C.4.5.3, plus the reorder-depth early output).
    pub fn store(&mut self, mut pic: DecodedPic, hdr: &SliceHeader, sps: &Sps, had_mmco5: bool) -> Result<()> {
        // FrameNumWrap of every short-term picture relative to this one, for
        // the sliding window and the MMCO arithmetic.
        {
            let curr = hdr.frame_num as i32;
            let max = sps.max_frame_num() as i32;
            for p in &mut self.pics {
                if p.mark == RefMark::Short {
                    p.frame_num_wrap = if p.frame_num as i32 > curr { p.frame_num as i32 - max } else { p.frame_num as i32 };
                }
            }
        }
        // Reference marking of the current picture (8.2.5.1).
        if hdr.is_reference() {
            if hdr.is_idr() {
                for p in &mut self.pics {
                    p.mark = RefMark::Unused;
                }
                if hdr.marking.long_term_reference {
                    pic.mark = RefMark::Long;
                    pic.long_term_frame_idx = 0;
                    self.max_long_term_frame_idx = Some(0);
                } else {
                    pic.mark = RefMark::Short;
                    self.max_long_term_frame_idx = None;
                }
            } else if hdr.marking.adaptive {
                let mut current_long: Option<u32> = None;
                let curr_pic_num = hdr.frame_num as i32;
                let max_frame_num = sps.max_frame_num() as i32;
                // FrameNumWrap for the MMCO pic-number arithmetic.
                for p in &mut self.pics {
                    if p.mark == RefMark::Short {
                        p.frame_num_wrap = if p.frame_num as i32 > curr_pic_num {
                            p.frame_num as i32 - max_frame_num
                        } else {
                            p.frame_num as i32
                        };
                    }
                }
                for op in &hdr.marking.ops {
                    match *op {
                        Mmco::UnmarkShortTerm(diff) => {
                            let pic_num = curr_pic_num - (diff as i32 + 1);
                            for p in &mut self.pics {
                                if p.mark == RefMark::Short && p.frame_num_wrap == pic_num {
                                    p.mark = RefMark::Unused;
                                }
                            }
                        }
                        Mmco::UnmarkLongTerm(lt) => {
                            for p in &mut self.pics {
                                if p.mark == RefMark::Long && p.long_term_frame_idx == lt {
                                    p.mark = RefMark::Unused;
                                }
                            }
                        }
                        Mmco::ShortToLong(diff, idx) => {
                            let pic_num = curr_pic_num - (diff as i32 + 1);
                            for p in &mut self.pics {
                                if p.mark == RefMark::Long && p.long_term_frame_idx == idx {
                                    p.mark = RefMark::Unused;
                                }
                            }
                            for p in &mut self.pics {
                                if p.mark == RefMark::Short && p.frame_num_wrap == pic_num {
                                    p.mark = RefMark::Long;
                                    p.long_term_frame_idx = idx;
                                }
                            }
                        }
                        Mmco::MaxLongTermIdx(plus1) => {
                            self.max_long_term_frame_idx = if plus1 == 0 { None } else { Some(plus1 - 1) };
                            for p in &mut self.pics {
                                if p.mark == RefMark::Long && (plus1 == 0 || p.long_term_frame_idx > plus1 - 1) {
                                    p.mark = RefMark::Unused;
                                }
                            }
                        }
                        Mmco::UnmarkAll => {
                            for p in &mut self.pics {
                                p.mark = RefMark::Unused;
                            }
                            self.max_long_term_frame_idx = None;
                        }
                        Mmco::CurrentToLong(idx) => {
                            for p in &mut self.pics {
                                if p.mark == RefMark::Long && p.long_term_frame_idx == idx {
                                    p.mark = RefMark::Unused;
                                }
                            }
                            current_long = Some(idx);
                        }
                    }
                }
                if let Some(idx) = current_long {
                    pic.mark = RefMark::Long;
                    pic.long_term_frame_idx = idx;
                } else {
                    pic.mark = RefMark::Short;
                }
            } else {
                // Sliding window (8.2.5.3).
                self.sliding_window(sps);
                pic.mark = RefMark::Short;
            }
        }
        if std::env::var_os("H26X_TRACE_DPB").is_some() {
            eprintln!("  marking: idr {} adaptive {} ops {:?} mmco5 {}", hdr.is_idr(), hdr.marking.adaptive, hdr.marking.ops, had_mmco5);
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
        if std::env::var_os("H26X_TRACE_DPB").is_some() {
            eprintln!(
                "  before capacity loop: len {} cap {} list {:?}",
                self.pics.len(),
                self.capacity,
                self.pics.iter().map(|p| (p.poc, p.mark as u8, p.needed_for_output as u8)).collect::<Vec<_>>()
            );
        }
        // C.4.5.2: a non-reference picture with no free frame buffer is
        // output directly once its POC is the lowest among those waiting —
        // it never needs storing.
        if !hdr.is_reference() && !hdr.is_idr() {
            while self.pics.len() >= self.capacity {
                let min_waiting = self.pics.iter().filter(|p| p.needed_for_output).map(|p| p.poc).min();
                match min_waiting {
                    Some(m) if pic.poc > m => {
                        self.bump_one();
                    }
                    _ => {
                        if !pic.non_existing {
                            self.output.push_back(pic.frame.to_picture(self.crop, pic.poc, pic.decode_index));
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
                    .filter(|(_, p)| p.mark == RefMark::Short)
                    .min_by_key(|(_, p)| p.decode_index)
                {
                    self.pics.remove(i);
                    continue;
                }
                return Err(Error::bitstream("DPB overflow"));
            }
        }
        pic.needed_for_output = !pic.non_existing;
        if std::env::var_os("H26X_TRACE_DPB").is_some() {
            eprintln!(
                "store poc {} fn {} ref {:?} | cap {} reorder {} | dpb: {:?}",
                pic.poc,
                pic.frame_num,
                pic.mark,
                self.capacity,
                self.num_reorder,
                self.pics.iter().map(|p| (p.poc, p.mark as u8, p.needed_for_output as u8)).collect::<Vec<_>>()
            );
        }
        self.pics.push(pic);
        // Reorder-depth early output.
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
        Ok(())
    }

    /// Sliding window marking (8.2.5.3).
    pub fn sliding_window(&mut self, sps: &Sps) {
        let max_refs = sps.max_num_ref_frames.max(1) as usize;
        while self.num_refs() >= max_refs {
            // Unmark the short-term picture with the smallest FrameNumWrap.
            let mut best: Option<usize> = None;
            for (i, p) in self.pics.iter().enumerate() {
                if p.mark == RefMark::Short && best.is_none_or(|b| p.frame_num_wrap < self.pics[b].frame_num_wrap) {
                    best = Some(i);
                }
            }
            match best {
                Some(i) => self.pics[i].mark = RefMark::Unused,
                None => break,
            }
        }
        self.remove_unneeded();
    }

    /// Insert "non-existing" frames for a gap in frame_num (8.2.5.2).
    pub fn fill_frame_num_gap(&mut self, sps: &Sps, prev_ref_frame_num: u32, frame_num: u32, template: &Frame, decode_index: &mut u64) {
        let max = sps.max_frame_num();
        let mut unused = (prev_ref_frame_num + 1) % max;
        let mut guard = 0;
        while unused != frame_num && guard < 64 {
            guard += 1;
            // FrameNumWrap of the existing short-terms relative to this frame_num.
            for p in &mut self.pics {
                if p.mark == RefMark::Short {
                    p.frame_num_wrap =
                        if p.frame_num > unused { p.frame_num as i32 - max as i32 } else { p.frame_num as i32 };
                }
            }
            self.sliding_window(sps);
            self.remove_unneeded();
            while self.pics.len() >= self.capacity {
                if !self.bump_one() {
                    break;
                }
            }
            let mut frame = template.clone();
            // A grey picture: what a decoder is expected to show if this ever
            // gets referenced.
            frame.y.data.fill(128);
            frame.cb.data.fill(128);
            frame.cr.data.fill(128);
            frame.poc = 0;
            let pic = DecodedPic {
                frame,
                poc: 0,
                frame_num: unused,
                frame_num_wrap: unused as i32,
                long_term_frame_idx: 0,
                mark: RefMark::Short,
                needed_for_output: false,
                non_existing: true,
                decode_index: *decode_index,
            };
            *decode_index += 1;
            self.pics.push(pic);
            unused = (unused + 1) % max;
        }
    }
}

impl Default for Dpb {
    fn default() -> Self {
        Self::new()
    }
}

/// Reference picture lists for a slice: indices into `dpb.pics`.
pub struct RefLists {
    /// List 0 and list 1.
    pub lists: [Vec<usize>; 2],
}

/// Build the reference picture lists for a P or B slice (8.2.4).
pub fn build_ref_lists(dpb: &mut Dpb, sps: &Sps, hdr: &SliceHeader, cur_poc: i32) -> Result<RefLists> {
    let curr_pic_num = hdr.frame_num as i32;
    let max_frame_num = sps.max_frame_num() as i32;
    for p in &mut dpb.pics {
        if p.mark == RefMark::Short {
            p.frame_num_wrap = if p.frame_num as i32 > curr_pic_num { p.frame_num as i32 - max_frame_num } else { p.frame_num as i32 };
        }
    }
    let shorts: Vec<usize> = dpb.pics.iter().enumerate().filter(|(_, p)| p.mark == RefMark::Short).map(|(i, _)| i).collect();
    let mut longs: Vec<usize> = dpb.pics.iter().enumerate().filter(|(_, p)| p.mark == RefMark::Long).map(|(i, _)| i).collect();
    longs.sort_by_key(|&i| dpb.pics[i].long_term_frame_idx);

    let mut lists: [Vec<usize>; 2] = [Vec::new(), Vec::new()];
    match hdr.slice_type {
        SliceType::P | SliceType::Sp => {
            let mut s = shorts.clone();
            s.sort_by(|&a, &b| dpb.pics[b].frame_num_wrap.cmp(&dpb.pics[a].frame_num_wrap));
            lists[0].extend(s);
            lists[0].extend(longs.iter().copied());
        }
        SliceType::B => {
            let mut before: Vec<usize> = shorts.iter().copied().filter(|&i| dpb.pics[i].poc < cur_poc).collect();
            let mut after: Vec<usize> = shorts.iter().copied().filter(|&i| dpb.pics[i].poc > cur_poc).collect();
            before.sort_by(|&a, &b| dpb.pics[b].poc.cmp(&dpb.pics[a].poc));
            after.sort_by_key(|&i| dpb.pics[i].poc);
            lists[0].extend(before.iter().copied());
            lists[0].extend(after.iter().copied());
            lists[0].extend(longs.iter().copied());
            lists[1].extend(after.iter().copied());
            lists[1].extend(before.iter().copied());
            lists[1].extend(longs.iter().copied());
            if lists[1].len() > 1 && lists[0] == lists[1] {
                lists[1].swap(0, 1);
            }
        }
        _ => {}
    }
    for l in 0..2 {
        let n = hdr.num_ref_idx_active[l] as usize;
        if n == 0 {
            lists[l].clear();
            continue;
        }
        // Modification (8.2.4.3), on a list one longer than active.
        if !hdr.ref_list_mods[l].is_empty() {
            let mut list: Vec<Option<usize>> = lists[l].iter().map(|&i| Some(i)).collect();
            list.resize(n + 1, None);
            let mut pic_num_pred = curr_pic_num;
            let mut ref_idx = 0usize;
            for m in &hdr.ref_list_mods[l] {
                let target: Option<usize>;
                match *m {
                    RefListMod::SubtractPicNum(d) | RefListMod::AddPicNum(d) => {
                        let d = d as i32;
                        let mut no_wrap = if matches!(m, RefListMod::SubtractPicNum(_)) {
                            pic_num_pred - d
                        } else {
                            pic_num_pred + d
                        };
                        if no_wrap < 0 {
                            no_wrap += max_frame_num;
                        } else if no_wrap >= max_frame_num {
                            no_wrap -= max_frame_num;
                        }
                        pic_num_pred = no_wrap;
                        let pic_num = if no_wrap > curr_pic_num { no_wrap - max_frame_num } else { no_wrap };
                        target = dpb
                            .pics
                            .iter()
                            .position(|p| p.mark == RefMark::Short && p.frame_num_wrap == pic_num);
                        if target.is_none() {
                            return Err(Error::bitstream(format!(
                                "ref_pic_list_modification names a missing short-term picture (PicNum {pic_num})"
                            )));
                        }
                        // Insert and remove the duplicate.
                        let t = target.unwrap();
                        for c in (ref_idx + 1..=n).rev() {
                            list[c] = list[c - 1];
                        }
                        list[ref_idx] = Some(t);
                        ref_idx += 1;
                        let mut n_idx = ref_idx;
                        for c in ref_idx..=n {
                            let keep = match list[c] {
                                Some(i) => !(dpb.pics[i].mark == RefMark::Short && dpb.pics[i].frame_num_wrap == pic_num),
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
                        target = dpb.pics.iter().position(|p| p.mark == RefMark::Long && p.long_term_frame_idx == lt);
                        let Some(t) = target else {
                            return Err(Error::bitstream(format!(
                                "ref_pic_list_modification names a missing long-term picture ({lt})"
                            )));
                        };
                        for c in (ref_idx + 1..=n).rev() {
                            list[c] = list[c - 1];
                        }
                        list[ref_idx] = Some(t);
                        ref_idx += 1;
                        let mut n_idx = ref_idx;
                        for c in ref_idx..=n {
                            let keep = match list[c] {
                                Some(i) => !(dpb.pics[i].mark == RefMark::Long && dpb.pics[i].long_term_frame_idx == lt),
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
            lists[l] = list[..n].iter().map(|x| x.unwrap_or(usize::MAX)).collect();
        } else {
            lists[l].truncate(n);
            while lists[l].len() < n {
                lists[l].push(usize::MAX);
            }
        }
    }
    Ok(RefLists { lists })
}
