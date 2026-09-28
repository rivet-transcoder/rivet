//! Noiseless coding: spectrum n-tuples through the Annex A codebooks with
//! their sign bits and escape sequences (subclause 9.3), scalefactor
//! differences, and the choice of sections.
//!
//! Sectioning is chosen by dynamic programming over scalefactor bands —
//! the exact minimum of (section headers + Huffman bits) over every way to
//! split a group into sections — rather than Annex C.8.3's greedy merge; the
//! search space is at most 51 bands, so the exact answer is cheap.

use super::bits::BitWriter;
use super::codebooks::{SCALEFACTOR, SPECTRUM};

/// Table 59: `(unsigned, dimension, largest absolute value)` per codebook.
const CODEBOOK: [(bool, usize, i32); 12] = [
    (false, 0, 0),
    (false, 4, 1),
    (false, 4, 1),
    (true, 4, 2),
    (true, 4, 2),
    (false, 2, 4),
    (false, 2, 4),
    (true, 2, 7),
    (true, 2, 7),
    (true, 2, 12),
    (true, 2, 12),
    (true, 2, 16),
];

/// The escape codebook.
pub(super) const ESC_HCB: u8 = 11;
/// Codebook numbers that can code any band (0 only codes all-zero bands).
pub(super) const NUM_CODEBOOKS: usize = 12;
/// Largest quantized magnitude the syntax can carry (subclause 10.3).
pub(super) const MAX_QUANT: i32 = 8191;

/// Whether `cb` can code a band whose largest magnitude is `max_abs`.
pub(super) fn codebook_covers(cb: u8, max_abs: i32) -> bool {
    match cb {
        0 => max_abs == 0,
        ESC_HCB => true,
        _ => CODEBOOK[usize::from(cb)].2 >= max_abs,
    }
}

fn tuple_index(cb: u8, vals: &[i32]) -> usize {
    let (unsigned, _, lav) = CODEBOOK[usize::from(cb)];
    let (modulus, offset) = if unsigned {
        (lav + 1, 0)
    } else {
        (2 * lav + 1, lav)
    };
    vals.iter().fold(0i32, |idx, &v| {
        let v = if unsigned { v.abs().min(16) } else { v };
        idx * modulus + v + offset
    }) as usize
}

/// Length of the escape sequence for magnitude `v >= 16`: N ones, a zero and
/// an N+4 bit word, where 2^(N+4) <= v < 2^(N+5).
fn escape_bits(v: i32) -> u32 {
    let n = 31 - (v as u32).leading_zeros() - 4;
    2 * n + 5
}

fn tuple_bits(cb: u8, vals: &[i32]) -> u32 {
    let (unsigned, _, _) = CODEBOOK[usize::from(cb)];
    let mut bits = u32::from(SPECTRUM[usize::from(cb)][tuple_index(cb, vals)].0);
    if unsigned {
        for &v in vals {
            if v != 0 {
                bits += 1;
            }
            if cb == ESC_HCB && v.abs() >= 16 {
                bits += escape_bits(v.abs());
            }
        }
    }
    bits
}

/// Bits to code the quantized values `q` (a whole scalefactor band) with
/// codebook `cb`, which must cover them.
pub(super) fn band_bits(cb: u8, q: &[i32]) -> u32 {
    if cb == 0 {
        return 0;
    }
    let dim = CODEBOOK[usize::from(cb)].1;
    q.chunks_exact(dim).map(|t| tuple_bits(cb, t)).sum()
}

pub(super) fn write_band(w: &mut BitWriter, cb: u8, q: &[i32]) {
    if cb == 0 {
        return;
    }
    let (unsigned, dim, _) = CODEBOOK[usize::from(cb)];
    for t in q.chunks_exact(dim) {
        let (len, code) = SPECTRUM[usize::from(cb)][tuple_index(cb, t)];
        w.put(code, u32::from(len));
        if unsigned {
            for &v in t {
                if v != 0 {
                    w.put(u32::from(v < 0), 1);
                }
            }
            if cb == ESC_HCB {
                for &v in t {
                    let a = v.abs();
                    if a >= 16 {
                        let n = 31 - (a as u32).leading_zeros() - 4;
                        // N ones then the separating zero, then the N+4 bit word.
                        w.put(((1u32 << n) - 1) << 1, n + 1);
                        w.put(a as u32 - (1 << (n + 4)), n + 4);
                    }
                }
            }
        }
    }
}

/// Bits of the scalefactor codeword for a difference `d` in -60..=60.
pub(super) fn sf_bits(d: i32) -> u32 {
    u32::from(SCALEFACTOR[(d + 60) as usize].0)
}

pub(super) fn write_sf(w: &mut BitWriter, d: i32) {
    let (len, code) = SCALEFACTOR[(d + 60) as usize];
    w.put(code, u32::from(len));
}

/// One section: a codebook over bands `start..end` of a group.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Section {
    pub cb: u8,
    pub start: usize,
    pub end: usize,
}

/// sect_len field width and escape value: 3 bits / 7 for short windows,
/// 5 bits / 31 otherwise (Table 17).
pub(super) fn section_len_params(short: bool) -> (u32, usize) {
    if short { (3, 7) } else { (5, 31) }
}

fn section_header_bits(len: usize, short: bool) -> u32 {
    let (bits, esc) = section_len_params(short);
    4 + bits * (len / esc + 1) as u32
}

/// Optimal sectioning of one group's bands.
///
/// `cost[b][cb]` is the cost in bits of coding band `b` with codebook `cb`
/// (`u32::MAX` when `cb` cannot code it). Returns the sections and their
/// total cost including the section headers.
pub(super) fn choose_sections(cost: &[[u32; NUM_CODEBOOKS]], short: bool) -> (Vec<Section>, u32) {
    let n = cost.len();
    let mut best = vec![u32::MAX; n + 1];
    let mut back = vec![(0usize, 0u8); n + 1];
    best[0] = 0;
    for end in 1..=n {
        for cb in 0..NUM_CODEBOOKS {
            let mut run = 0u32;
            for (start, row) in cost[..end].iter().enumerate().rev() {
                let c = row[cb];
                if c == u32::MAX {
                    break;
                }
                run += c;
                if best[start] == u32::MAX {
                    continue;
                }
                let total = best[start] + run + section_header_bits(end - start, short);
                if total < best[end] {
                    best[end] = total;
                    back[end] = (start, cb as u8);
                }
            }
        }
    }
    let mut sections = Vec::new();
    let mut end = n;
    while end > 0 {
        let (start, cb) = back[end];
        sections.push(Section { cb, start, end });
        end = start;
    }
    sections.reverse();
    (sections, best[n])
}

pub(super) fn write_sections(w: &mut BitWriter, sections: &[Section], short: bool) {
    let (bits, esc) = section_len_params(short);
    for s in sections {
        w.put(u32::from(s.cb), 4);
        let mut len = s.end - s.start;
        while len >= esc {
            w.put(esc as u32, bits);
            len -= esc;
        }
        w.put(len as u32, bits);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::encode::aac::codebooks;

    fn check_complete_prefix_code(name: &str, book: &[(u8, u32)]) {
        let mut kraft = 0f64;
        let mut codes: Vec<String> = Vec::new();
        for &(len, code) in book {
            assert!(
                len > 0 && len <= 32 && u64::from(code) < 1u64 << len,
                "{name}"
            );
            kraft += 0.5f64.powi(i32::from(len));
            codes.push(format!("{code:0width$b}", width = usize::from(len)));
        }
        assert!((kraft - 1.0).abs() < 1e-12, "{name}: Kraft sum {kraft}");
        codes.sort();
        for pair in codes.windows(2) {
            assert!(
                !pair[1].starts_with(&pair[0]),
                "{name}: {} prefixes {}",
                pair[0],
                pair[1]
            );
        }
    }

    #[test]
    fn every_codebook_is_a_complete_prefix_code_of_the_right_size() {
        check_complete_prefix_code("scalefactor", &codebooks::SCALEFACTOR);
        let sizes = [0, 81, 81, 81, 81, 81, 81, 64, 64, 169, 169, 289];
        for cb in 1..12 {
            assert_eq!(SPECTRUM[cb].len(), sizes[cb]);
            check_complete_prefix_code(&format!("spectrum {cb}"), SPECTRUM[cb]);
        }
        // Spot checks against Table A.1: a zero difference is the one-bit '0'.
        assert_eq!(codebooks::SCALEFACTOR[60], (1, 0));
    }

    #[test]
    fn tuple_indices_follow_subclause_9_3() {
        // Signed 4-tuples, LAV 1: offset 1, modulus 3.
        assert_eq!(tuple_index(1, &[-1, -1, -1, -1]), 0);
        assert_eq!(tuple_index(1, &[0, 0, 0, 0]), 40);
        assert_eq!(tuple_index(1, &[1, 1, 1, 1]), 80);
        // Unsigned pairs, LAV 7: modulus 8, magnitudes only.
        assert_eq!(tuple_index(7, &[-3, 5]), 3 * 8 + 5);
        // The escape codebook clips to the escape flag 16.
        assert_eq!(tuple_index(11, &[100, 2]), 16 * 17 + 2);
    }

    #[test]
    fn escape_sequences_match_the_worked_examples() {
        // Subclause 9.3: 00000 is 16, 01111 is 31, 1000000 is 32, 1011111 is 63.
        for (v, expect) in [
            (16, "00000"),
            (31, "01111"),
            (32, "1000000"),
            (63, "1011111"),
        ] {
            let mut w = BitWriter::default();
            write_band(&mut w, ESC_HCB, &[v, 0]);
            let (len, _) = SPECTRUM[11][tuple_index(11, &[v, 0])];
            let total = w.len_bits();
            assert_eq!(total as u32, band_bits(ESC_HCB, &[v, 0]));
            assert_eq!(total - usize::from(len) - 1, expect.len(), "{v}");
            w.align();
            let bytes = w.into_bytes();
            let bitstr: String = bytes.iter().map(|b| format!("{b:08b}")).collect();
            let start = usize::from(len) + 1;
            assert_eq!(&bitstr[start..start + expect.len()], expect, "{v}");
        }
    }

    #[test]
    fn sectioning_merges_when_headers_cost_more_than_the_codebook_change() {
        let inf = u32::MAX;
        let mut a = [inf; NUM_CODEBOOKS];
        let mut b = [inf; NUM_CODEBOOKS];
        for cb in 1..12 {
            a[cb] = 20;
            b[cb] = if cb >= 5 { 21 } else { inf };
        }
        let (secs, cost) = choose_sections(&[a, b], false);
        assert_eq!(
            secs,
            vec![Section {
                cb: 5,
                start: 0,
                end: 2
            }]
        );
        assert_eq!(cost, 20 + 21 + 9);
    }
}
