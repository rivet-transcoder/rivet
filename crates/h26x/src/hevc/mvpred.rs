//! Motion vector prediction (H.265 8.5.3.2): the merge candidate list
//! (spatial, temporal, combined bi-predictive, zero) and AMVP (spatial with
//! scaling, temporal), on top of the collocated motion vector derivation.

use super::frame::{Frame, MotionInfo, Mv, Sample};
use super::pic::PicInfo;

/// The reference pictures of the current slice, as the predictor needs them.
pub struct RefCtx<'a, S: Sample = u16> {
    /// POC of RefPicListX[i].
    pub pocs: [Vec<i32>; 2],
    /// Long-term flags of RefPicListX[i].
    pub long_term: [Vec<bool>; 2],
    /// The collocated picture (`ColPic`), if temporal MVP is on.
    pub col: Option<&'a Frame<S>>,
    /// POC of the current picture.
    pub cur_poc: i32,
    /// `NoBackwardPredFlag`.
    pub no_backward_pred: bool,
    /// `slice_temporal_mvp_enabled_flag`.
    pub tmvp: bool,
    /// `MaxNumMergeCand`.
    pub max_merge_cand: usize,
    /// `Log2ParMrgLevel`.
    pub log2_par_mrg_level: u32,
    /// Slice type is B.
    pub is_b: bool,
    /// Active reference count per list.
    pub num_ref_idx: [usize; 2],
    /// `collocated_from_l0_flag`.
    pub col_from_l0: bool,
}

/// A merge/AMVP candidate: motion of a PU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cand {
    /// Vectors.
    pub mv: [Mv; 2],
    /// Reference indices (-1 = list unused).
    pub ref_idx: [i8; 2],
}

impl Cand {
    const NONE: Cand = Cand { mv: [Mv::ZERO; 2], ref_idx: [-1; 2] };
    fn same_motion(&self, o: &Cand) -> bool {
        self.ref_idx == o.ref_idx && self.mv == o.mv
    }
}

/// The current PU and its coding block, for availability (6.4.2).
#[derive(Debug, Clone, Copy)]
pub struct PuPos {
    /// Coding block.
    pub x_cb: i32,
    /// See `x_cb`.
    pub y_cb: i32,
    /// `nCbS`.
    pub n_cb: i32,
    /// Prediction block.
    pub x_pb: i32,
    /// See `x_pb`.
    pub y_pb: i32,
    /// `nPbW`.
    pub w: i32,
    /// `nPbH`.
    pub h: i32,
    /// `partIdx`.
    pub part_idx: u32,
}

/// Prediction block availability (6.4.2) of the neighbour at `(xn, yn)`,
/// returning its motion when available and inter.
fn neighbour_pb<S: Sample>(info: &PicInfo, cur: &Frame<S>, pu: &PuPos, xn: i32, yn: i32) -> Option<MotionInfo> {
    let (pw, ph) = (cur.width as i32, cur.height as i32);
    let same_cb = xn >= pu.x_cb && xn < pu.x_cb + pu.n_cb && yn >= pu.y_cb && yn < pu.y_cb + pu.n_cb;
    let avail = if !same_cb {
        info.available(pu.x_pb, pu.y_pb, xn, yn, pw, ph)
    } else if (pu.w << 1) == pu.n_cb && (pu.h << 1) == pu.n_cb && pu.part_idx == 1 && pu.y_cb + pu.h <= yn && pu.x_cb + pu.w > xn {
        false
    } else {
        // Same CB: an earlier partition (the candidate positions never fall
        // in the current or a later partition except the case above).
        true
    };
    if !avail {
        return None;
    }
    let m = cur.motion_at(xn as usize, yn as usize);
    if m.intra {
        return None;
    }
    Some(*m)
}

/// `LongTermRefPic` for the current slice's list X index.
#[inline]
fn cur_is_long_term<S: Sample>(refs: &RefCtx<S>, list: usize, idx: i8) -> bool {
    refs.long_term[list].get(idx as usize).copied().unwrap_or(false)
}

/// Collocated motion vector (8.5.3.2.9) for list `list` / `ref_idx`, at
/// collocated position `(xc, yc)` (already 16-aligned) in `col`.
fn collocated_mv<S: Sample>(refs: &RefCtx<S>, col: &Frame<S>, xc: i32, yc: i32, list: usize, ref_idx: i8) -> Option<Mv> {
    if xc < 0 || yc < 0 || xc >= col.width as i32 || yc >= col.height as i32 {
        return None;
    }
    let m = col.motion_at(xc as usize, yc as usize);
    if m.intra {
        return None;
    }
    let (mv_col, ref_poc_col, ref_lt_col) = if !m.uses(0) {
        (m.mv[1], m.ref_poc[1], m.ref_long_term[1])
    } else if !m.uses(1) {
        (m.mv[0], m.ref_poc[0], m.ref_long_term[0])
    } else {
        let n = if refs.no_backward_pred { list } else { refs.collocated_list_n() };
        (m.mv[n], m.ref_poc[n], m.ref_long_term[n])
    };
    let cur_lt = cur_is_long_term(refs, list, ref_idx);
    if cur_lt != ref_lt_col {
        return None;
    }
    let col_poc_diff = col.poc - ref_poc_col;
    let cur_poc_diff = refs.cur_poc - refs.pocs[list][ref_idx as usize];
    if cur_lt || col_poc_diff == cur_poc_diff {
        return Some(mv_col);
    }
    if col_poc_diff == 0 {
        return Some(mv_col);
    }
    let td = col_poc_diff.clamp(-128, 127);
    let tb = cur_poc_diff.clamp(-128, 127);
    let tx = (16384 + (td.abs() >> 1)) / td;
    let dsf = ((tb * tx + 32) >> 6).clamp(-4096, 4095);
    let scale = |v: i16| -> i16 {
        let p = dsf * v as i32;
        let s = if p < 0 { -1 } else { 1 };
        (s * ((p.abs() + 127) >> 8)).clamp(-32768, 32767) as i16
    };
    Some(Mv::new(scale(mv_col.x), scale(mv_col.y)))
}

impl<S: Sample> RefCtx<'_, S> {
    /// `collocated_from_l0_flag` as a list index N for the "both lists"
    /// case: mvLNCol with N = collocated_from_l0_flag.
    fn collocated_list_n(&self) -> usize {
        self.col_from_l0 as usize
    }
}


/// Temporal candidate (8.5.3.2.8) for list `list` and `ref_idx`.
pub fn temporal_mv<S: Sample>(refs: &RefCtx<S>, info: &PicInfo, pu: &PuPos, list: usize, ref_idx: i8) -> Option<Mv> {
    if !refs.tmvp {
        return None;
    }
    let col = refs.col?;
    let log2_ctb = info.log2_ctb;
    let x_br = pu.x_pb + pu.w;
    let y_br = pu.y_pb + pu.h;
    let mut result = None;
    if (pu.y_pb >> log2_ctb) == (y_br >> log2_ctb) && y_br < col.height as i32 && x_br < col.width as i32 {
        result = collocated_mv(refs, col, (x_br >> 4) << 4, (y_br >> 4) << 4, list, ref_idx);
    }
    if result.is_none() {
        let x_ctr = pu.x_pb + (pu.w >> 1);
        let y_ctr = pu.y_pb + (pu.h >> 1);
        result = collocated_mv(refs, col, (x_ctr >> 4) << 4, (y_ctr >> 4) << 4, list, ref_idx);
    }
    result
}

/// The merge candidate list (8.5.3.2.2 – 8.5.3.2.5) and the selected
/// candidate `merge_idx`.
pub fn merge_candidate<S: Sample>(info: &PicInfo, cur: &Frame<S>, refs: &RefCtx<S>, pu_in: &PuPos, merge_idx: usize) -> Cand {
    // Parallel merge level: a CU of size 8 shares one candidate list.
    let mut pu = *pu_in;
    if refs.log2_par_mrg_level > 2 && pu.n_cb == 8 {
        pu.x_pb = pu.x_cb;
        pu.y_pb = pu.y_cb;
        pu.w = pu.n_cb;
        pu.h = pu.n_cb;
        pu.part_idx = 0;
    }
    let (x_pb, y_pb, w, h) = (pu.x_pb, pu.y_pb, pu.w, pu.h);
    let pml = refs.log2_par_mrg_level;
    let same_mer = |xn: i32, yn: i32| (x_pb >> pml) == (xn >> pml) && (y_pb >> pml) == (yn >> pml);
    // The partition-mode exclusions need the original PU shape/index.
    let orig = pu_in;
    let part_mode_second_vertical = orig.part_idx == 1 && orig.w < orig.n_cb && orig.h == orig.n_cb; // Nx2N, nLx2N, nRx2N
    let part_mode_second_horizontal = orig.part_idx == 1 && orig.h < orig.n_cb && orig.w == orig.n_cb; // 2NxN, 2NxnU, 2NxnD
    let single = refs.log2_par_mrg_level > 2 && pu.n_cb == 8;

    let mut list: Vec<Cand> = Vec::with_capacity(5);
    let to_cand = |m: MotionInfo| Cand { mv: m.mv, ref_idx: m.ref_idx };

    // A1
    let (xa1, ya1) = (x_pb - 1, y_pb + h - 1);
    let a1 = if same_mer(xa1, ya1) || (part_mode_second_vertical && !single) {
        None
    } else {
        neighbour_pb(info, cur, &pu, xa1, ya1).map(to_cand)
    };
    if let Some(c) = a1 {
        list.push(c);
    }
    // B1
    let (xb1, yb1) = (x_pb + w - 1, y_pb - 1);
    let b1 = if same_mer(xb1, yb1) || (part_mode_second_horizontal && !single) {
        None
    } else {
        neighbour_pb(info, cur, &pu, xb1, yb1).map(to_cand)
    };
    if let Some(c) = b1 {
        if !a1.is_some_and(|a| a.same_motion(&c)) {
            list.push(c);
        }
    }
    // B0
    let (xb0, yb0) = (x_pb + w, y_pb - 1);
    let b0 = if same_mer(xb0, yb0) { None } else { neighbour_pb(info, cur, &pu, xb0, yb0).map(to_cand) };
    if let Some(c) = b0 {
        if !b1.is_some_and(|b| b.same_motion(&c)) {
            list.push(c);
        }
    }
    // A0
    let (xa0, ya0) = (x_pb - 1, y_pb + h);
    let a0 = if same_mer(xa0, ya0) { None } else { neighbour_pb(info, cur, &pu, xa0, ya0).map(to_cand) };
    if let Some(c) = a0 {
        if !a1.is_some_and(|a| a.same_motion(&c)) {
            list.push(c);
        }
    }
    // B2 (only if fewer than four so far)
    if list.len() < 4 {
        let (xb2, yb2) = (x_pb - 1, y_pb - 1);
        let b2 = if same_mer(xb2, yb2) { None } else { neighbour_pb(info, cur, &pu, xb2, yb2).map(to_cand) };
        if let Some(c) = b2 {
            if !a1.is_some_and(|a| a.same_motion(&c)) && !b1.is_some_and(|b| b.same_motion(&c)) {
                list.push(c);
            }
        }
    }
    if list.len() > merge_idx {
        // Only what is needed: the merge index picks an early candidate.
        // (The list must still be complete for later candidates, so only
        // return early when the index is already covered.)
        return finalize(list[merge_idx], orig);
    }
    // Temporal.
    if refs.tmvp && list.len() < refs.max_merge_cand {
        let mv0 = temporal_mv(refs, info, &pu, 0, 0);
        let mv1 = if refs.is_b { temporal_mv(refs, info, &pu, 1, 0) } else { None };
        if mv0.is_some() || mv1.is_some() {
            list.push(Cand {
                mv: [mv0.unwrap_or(Mv::ZERO), mv1.unwrap_or(Mv::ZERO)],
                ref_idx: [if mv0.is_some() { 0 } else { -1 }, if mv1.is_some() { 0 } else { -1 }],
            });
        }
    }
    if list.len() > merge_idx {
        return finalize(list[merge_idx], orig);
    }
    // Combined bi-predictive (B slices).
    if refs.is_b && list.len() > 1 && list.len() < refs.max_merge_cand {
        const COMB: [(usize, usize); 12] = [(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1), (0, 3), (3, 0), (1, 3), (3, 1), (2, 3), (3, 2)];
        let num_orig = list.len();
        let mut comb_idx = 0;
        while comb_idx < num_orig * (num_orig - 1) && list.len() < refs.max_merge_cand {
            let (l0i, l1i) = COMB[comb_idx];
            let l0c = list[l0i];
            let l1c = list[l1i];
            if l0c.ref_idx[0] >= 0 && l1c.ref_idx[1] >= 0 {
                let poc0 = refs.pocs[0][l0c.ref_idx[0] as usize];
                let poc1 = refs.pocs[1][l1c.ref_idx[1] as usize];
                if poc0 != poc1 || l0c.mv[0] != l1c.mv[1] {
                    list.push(Cand { mv: [l0c.mv[0], l1c.mv[1]], ref_idx: [l0c.ref_idx[0], l1c.ref_idx[1]] });
                }
            }
            comb_idx += 1;
        }
    }
    // Zero candidates.
    let num_ref = if refs.is_b { refs.num_ref_idx[0].min(refs.num_ref_idx[1]) } else { refs.num_ref_idx[0] };
    let mut zero_idx = 0i8;
    while list.len() < refs.max_merge_cand {
        let r = if (zero_idx as usize) < num_ref { zero_idx } else { 0 };
        list.push(Cand { mv: [Mv::ZERO; 2], ref_idx: [r, if refs.is_b { r } else { -1 }] });
        zero_idx += 1;
    }
    finalize(list[merge_idx.min(list.len() - 1)], orig)
}

/// The 8x4 / 4x8 bi-prediction restriction (8.5.3.2.2 step 10).
fn finalize(mut c: Cand, orig: &PuPos) -> Cand {
    if c.ref_idx[0] >= 0 && c.ref_idx[1] >= 0 && orig.w + orig.h == 12 {
        c.ref_idx[1] = -1;
        c.mv[1] = Mv::ZERO;
    }
    c
}

/// AMVP: the motion vector predictor (8.5.3.2.6 / 8.5.3.2.7) for list
/// `list`, reference `ref_idx`, selected by `mvp_flag`.
pub fn amvp<S: Sample>(info: &PicInfo, cur: &Frame<S>, refs: &RefCtx<S>, pu: &PuPos, list: usize, ref_idx: i8, mvp_flag: u32) -> Mv {
    let target_poc = refs.pocs[list][ref_idx as usize];
    let target_lt = refs.long_term[list][ref_idx as usize];
    let (x_pb, y_pb, w, h) = (pu.x_pb, pu.y_pb, pu.w, pu.h);
    // Neighbours A0, A1 (below-left, left-bottom), B0, B1, B2.
    let a0 = neighbour_pb(info, cur, pu, x_pb - 1, y_pb + h);
    let a1 = neighbour_pb(info, cur, pu, x_pb - 1, y_pb + h - 1);
    let b0 = neighbour_pb(info, cur, pu, x_pb + w, y_pb - 1);
    let b1 = neighbour_pb(info, cur, pu, x_pb + w - 1, y_pb - 1);
    let b2 = neighbour_pb(info, cur, pu, x_pb - 1, y_pb - 1);
    let is_scaled = a0.is_some() || a1.is_some();

    // First pass on a candidate: same reference picture (any list).
    let direct = |m: &MotionInfo| -> Option<Mv> {
        // predFlagLX and same POC in list X first, then the other list Y.
        if m.uses(list) && m.ref_poc[list] == target_poc && m.ref_long_term[list] == target_lt {
            return Some(m.mv[list]);
        }
        let y = 1 - list;
        if m.uses(y) && m.ref_poc[y] == target_poc && m.ref_long_term[y] == target_lt {
            return Some(m.mv[y]);
        }
        None
    };
    // Second pass: any reference, scaled unless long-term mismatch.
    let scaled = |m: &MotionInfo| -> Option<Mv> {
        for l in [list, 1 - list] {
            if m.uses(l) && m.ref_long_term[l] == target_lt {
                let mv = m.mv[l];
                if target_lt {
                    return Some(mv);
                }
                let td = (refs.cur_poc - m.ref_poc[l]).clamp(-128, 127);
                let tb = (refs.cur_poc - target_poc).clamp(-128, 127);
                if td == 0 || td == tb {
                    return Some(mv);
                }
                let tx = (16384 + (td.abs() >> 1)) / td;
                let dsf = ((tb * tx + 32) >> 6).clamp(-4096, 4095);
                let sc = |v: i16| -> i16 {
                    let p = dsf * v as i32;
                    let s = if p < 0 { -1 } else { 1 };
                    (s * ((p.abs() + 127) >> 8)).clamp(-32768, 32767) as i16
                };
                return Some(Mv::new(sc(mv.x), sc(mv.y)));
            }
        }
        None
    };

    // A.
    let mut mv_a: Option<Mv> = None;
    for m in [a0.as_ref(), a1.as_ref()].into_iter().flatten() {
        if let Some(mv) = direct(m) {
            mv_a = Some(mv);
            break;
        }
    }
    if mv_a.is_none() {
        for m in [a0.as_ref(), a1.as_ref()].into_iter().flatten() {
            if let Some(mv) = scaled(m) {
                mv_a = Some(mv);
                break;
            }
        }
    }
    // B.
    let mut mv_b: Option<Mv> = None;
    for m in [b0.as_ref(), b1.as_ref(), b2.as_ref()].into_iter().flatten() {
        if let Some(mv) = direct(m) {
            mv_b = Some(mv);
            break;
        }
    }
    if !is_scaled && mv_b.is_some() && mv_a.is_none() {
        // 8.5.3.2.7 step 7: when isScaledFlagLX is 0 and B was found,
        // A takes B's value and B is re-derived with scaling.
        mv_a = mv_b;
        mv_b = None;
    }
    if !is_scaled {
        // Re-derive B with scaling (step 8).
        if mv_b.is_none() {
            for m in [b0.as_ref(), b1.as_ref(), b2.as_ref()].into_iter().flatten() {
                if let Some(mv) = scaled(m) {
                    mv_b = Some(mv);
                    break;
                }
            }
        }
    }
    let mut cands: Vec<Mv> = Vec::with_capacity(3);
    if let Some(a) = mv_a {
        cands.push(a);
    }
    if let Some(b) = mv_b {
        if !(mv_a == Some(b)) {
            cands.push(b);
        }
    }
    if cands.len() < 2 {
        if let Some(t) = temporal_mv(refs, info, pu, list, ref_idx) {
            cands.push(t);
        }
    }
    while cands.len() < 2 {
        cands.push(Mv::ZERO);
    }
    cands[mvp_flag as usize]
}
