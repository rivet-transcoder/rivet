//! Entropy decoding for the DTS core: the Annex D.5 Huffman code books, the
//! Annex D.6 4-element block codes, and the (`ABITS`, `SEL`) → code book
//! selection of Table 5-26.

use std::sync::LazyLock;

use super::bits::BitReader;
use super::tables::{self, HuffEntry};
use super::DtsError;

/// A Huffman code book indexed for decoding: for every code length, the
/// `(code, level)` pairs of that length. The books are small (≤ 129
/// entries, ≤ 16-bit codes), so reading one bit at a time and scanning the
/// entries of the current length is fast enough for audio rates.
pub struct Codebook {
    by_len: Vec<Vec<(u32, i32)>>,
    max_len: u32,
}

impl Codebook {
    pub fn new(entries: &[HuffEntry]) -> Self {
        let max_len = entries.iter().map(|e| e.1 as u32).max().unwrap_or(0);
        let mut by_len = vec![Vec::new(); max_len as usize + 1];
        for &(level, len, code) in entries {
            by_len[len as usize].push((code, level));
        }
        Self { by_len, max_len }
    }

    /// Decode one symbol. The books are complete prefix codes (the generator
    /// checks Kraft sum == 1), so every bit string of `max_len` bits has a
    /// match; not finding one means the table is wrong, not the stream.
    pub fn decode(&self, r: &mut BitReader) -> Result<i32, DtsError> {
        let mut code = 0u32;
        for len in 1..=self.max_len {
            code = (code << 1) | r.bits(1)?;
            if let Some(&(_, level)) = self.by_len[len as usize].iter().find(|(c, _)| *c == code) {
                return Ok(level);
            }
        }
        Err(DtsError::Invalid("Huffman code not in book (complete code book cannot fail — table defect)"))
    }
}

macro_rules! book {
    ($name:ident, $table:ident) => {
        static $name: LazyLock<Codebook> = LazyLock::new(|| Codebook::new(&tables::$table));
    };
}

book!(A3, HUFF_A3);
book!(A4, HUFF_A4);
book!(B4, HUFF_B4);
book!(C4, HUFF_C4);
book!(D4, HUFF_D4);
book!(A5, HUFF_A5);
book!(B5, HUFF_B5);
book!(C5, HUFF_C5);
book!(A7, HUFF_A7);
book!(B7, HUFF_B7);
book!(C7, HUFF_C7);
book!(A9, HUFF_A9);
book!(B9, HUFF_B9);
book!(C9, HUFF_C9);
book!(A12, HUFF_A12);
book!(B12, HUFF_B12);
book!(C12, HUFF_C12);
book!(D12, HUFF_D12);
book!(E12, HUFF_E12);
book!(A13, HUFF_A13);
book!(B13, HUFF_B13);
book!(C13, HUFF_C13);
book!(A17, HUFF_A17);
book!(B17, HUFF_B17);
book!(C17, HUFF_C17);
book!(D17, HUFF_D17);
book!(E17, HUFF_E17);
book!(F17, HUFF_F17);
book!(G17, HUFF_G17);
book!(A25, HUFF_A25);
book!(B25, HUFF_B25);
book!(C25, HUFF_C25);
book!(D25, HUFF_D25);
book!(E25, HUFF_E25);
book!(F25, HUFF_F25);
book!(G25, HUFF_G25);
book!(A33, HUFF_A33);
book!(B33, HUFF_B33);
book!(C33, HUFF_C33);
book!(D33, HUFF_D33);
book!(E33, HUFF_E33);
book!(F33, HUFF_F33);
book!(G33, HUFF_G33);
book!(A65, HUFF_A65);
book!(B65, HUFF_B65);
book!(C65, HUFF_C65);
book!(D65, HUFF_D65);
book!(E65, HUFF_E65);
book!(F65, HUFF_F65);
book!(G65, HUFF_G65);
book!(SA129, HUFF_SA129);
book!(SB129, HUFF_SB129);
book!(SC129, HUFF_SC129);
book!(SD129, HUFF_SD129);
book!(SE129, HUFF_SE129);
book!(A129, HUFF_A129);
book!(B129, HUFF_B129);
book!(C129, HUFF_C129);
book!(D129, HUFF_D129);
book!(E129, HUFF_E129);
book!(F129, HUFF_F129);
book!(G129, HUFF_G129);

/// Table 5-23: `THUFF` → code book for `TMODE` (A4..D4).
pub fn tmode_book(thuff: u32) -> &'static Codebook {
    match thuff {
        0 => &A4,
        1 => &B4,
        2 => &C4,
        _ => &D4,
    }
}

/// Table 5-25: `BHUFF` 0..=4 → Huffman code book for `ABITS` (A12..E12);
/// 5 and 6 are linear 4/5-bit fields and 7 is invalid.
pub fn abits_book(bhuff: u32) -> Option<&'static Codebook> {
    match bhuff {
        0 => Some(&A12),
        1 => Some(&B12),
        2 => Some(&C12),
        3 => Some(&D12),
        4 => Some(&E12),
        _ => None,
    }
}

/// Table 5-24: `SHUFF` 0..=4 → Huffman code book for scale-factor index
/// differences (SA129..SE129); 5 and 6 are linear 6/7-bit fields.
pub fn scale_book(shuff: u32) -> Option<&'static Codebook> {
    match shuff {
        0 => Some(&SA129),
        1 => Some(&SB129),
        2 => Some(&SC129),
        3 => Some(&SD129),
        4 => Some(&SE129),
        _ => None,
    }
}

/// How the eight quantisation indices of one subband subsubframe are coded,
/// per Table 5-26 and the `nQType` logic of Table 5-29.
#[derive(Clone, Copy)]
pub enum SampleCoding {
    /// `ABITS` == 0: no bits allocated, the samples are zero.
    None,
    /// Huffman code, one symbol per sample.
    Huffman(&'static Codebook),
    /// 4-element block code: two codes of `bits` bits each cover 8 samples.
    Block { levels: u32, bits: u32 },
    /// "No further encoding": `bits`-bit two's complement per sample.
    Raw { bits: u32 },
}

/// Resolve `(ABITS, SEL)` to a coding. `sel` is `SEL[ch][ABITS-1]` — for
/// `ABITS` > 10 it is not transmitted and must be 0.
pub fn sample_coding(abits: u32, sel: u32) -> Result<SampleCoding, DtsError> {
    fn pick(books: &[&'static Codebook], sel: u32, last: SampleCoding) -> Result<SampleCoding, DtsError> {
        match books.get(sel as usize) {
            Some(b) => Ok(SampleCoding::Huffman(b)),
            None if sel as usize == books.len() => Ok(last),
            None => Err(DtsError::Invalid("SEL out of range for this ABITS")),
        }
    }
    let block = |levels: u32| {
        let bits = tables::BLOCK_CODE_BITS
            .iter()
            .find(|(l, _)| *l == levels)
            .map(|(_, b)| *b as u32)
            .expect("block code width for every Table 5-26 block code");
        SampleCoding::Block { levels, bits }
    };
    match abits {
        0 => Ok(SampleCoding::None),
        1 => pick(&[&A3], sel, block(3)),
        2 => pick(&[&A5, &B5, &C5], sel, block(5)),
        3 => pick(&[&A7, &B7, &C7], sel, block(7)),
        4 => pick(&[&A9, &B9, &C9], sel, block(9)),
        5 => pick(&[&A13, &B13, &C13], sel, block(13)),
        6 => pick(&[&A17, &B17, &C17, &D17, &E17, &F17, &G17], sel, block(17)),
        7 => pick(&[&A25, &B25, &C25, &D25, &E25, &F25, &G25], sel, block(25)),
        // "33 or 32", "65 or 64", "129 or 128" levels: the NFE form is the
        // even count, i.e. ABITS-3 bits of two's complement.
        8 => pick(&[&A33, &B33, &C33, &D33, &E33, &F33, &G33], sel, SampleCoding::Raw { bits: 5 }),
        9 => pick(&[&A65, &B65, &C65, &D65, &E65, &F65, &G65], sel, SampleCoding::Raw { bits: 6 }),
        10 => pick(&[&A129, &B129, &C129, &D129, &E129, &F129, &G129], sel, SampleCoding::Raw { bits: 7 }),
        11..=26 => Ok(SampleCoding::Raw { bits: abits - 3 }),
        _ => Err(DtsError::Invalid("ABITS above 26")),
    }
}

/// Decode one 4-element block code (Annex C.3.2, arithmetic form): the code
/// is the mixed-radix number `i0 + L·i1 + L²·i2 + L³·i3` of the four
/// zero-centred indices.
pub fn decode_block(mut code: u32, levels: u32, out: &mut [i32; 4]) -> Result<(), DtsError> {
    let offset = ((levels - 1) / 2) as i32;
    for v in out.iter_mut() {
        *v = (code % levels) as i32 - offset;
        code /= levels;
    }
    if code != 0 {
        return Err(DtsError::Invalid("block code out of range"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pack `(len, code)` pairs MSB-first into bytes.
    fn pack(codes: &[(u8, u32)]) -> Vec<u8> {
        let mut bits: Vec<u8> = Vec::new();
        for &(len, code) in codes {
            for i in (0..len).rev() {
                bits.push(((code >> i) & 1) as u8);
            }
        }
        let mut out = vec![0u8; bits.len().div_ceil(8)];
        for (i, b) in bits.iter().enumerate() {
            out[i / 8] |= b << (7 - (i % 8));
        }
        out
    }

    #[test]
    fn every_book_round_trips_every_entry() {
        // Each table entry, encoded as its own code, decodes to its level —
        // and in sequence, so a wrong-length entry would derail the rest.
        let all: &[(&str, &[HuffEntry])] = &[
            ("A3", &tables::HUFF_A3),
            ("D4", &tables::HUFF_D4),
            ("C5", &tables::HUFF_C5),
            ("B7", &tables::HUFF_B7),
            ("A12", &tables::HUFF_A12),
            ("E12", &tables::HUFF_E12),
            ("C13", &tables::HUFF_C13),
            ("G17", &tables::HUFF_G17),
            ("F25", &tables::HUFF_F25),
            ("A33", &tables::HUFF_A33),
            ("D65", &tables::HUFF_D65),
            ("SA129", &tables::HUFF_SA129),
            ("D129", &tables::HUFF_D129),
            ("G129", &tables::HUFF_G129),
        ];
        for (name, entries) in all {
            let book = Codebook::new(entries);
            let packed = pack(&entries.iter().map(|e| (e.1, e.2)).collect::<Vec<_>>());
            let mut r = BitReader::new(&packed);
            for e in entries.iter() {
                assert_eq!(book.decode(&mut r).unwrap(), e.0, "{name}: entry {e:?}");
            }
        }
    }

    #[test]
    #[allow(clippy::unusual_byte_groupings)] // grouped per code word
    fn a3_matches_the_printed_codes() {
        // D.5.1 Table A3: 0 → "0" (1 bit), 1 → "10", -1 → "11".
        let mut r = BitReader::new(&[0b0_10_11_0_00]);
        let b = Codebook::new(&tables::HUFF_A3);
        assert_eq!(b.decode(&mut r).unwrap(), 0);
        assert_eq!(b.decode(&mut r).unwrap(), 1);
        assert_eq!(b.decode(&mut r).unwrap(), -1);
        assert_eq!(b.decode(&mut r).unwrap(), 0);
    }

    #[test]
    fn block_code_decodes_the_spec_example() {
        // Annex C.3.2: three-level code 64 → (0, -1, 0, +1).
        let mut out = [0i32; 4];
        decode_block(64, 3, &mut out).unwrap();
        assert_eq!(out, [0, -1, 0, 1]);
        // 3^4 = 81 is the first code out of range.
        assert!(decode_block(81, 3, &mut out).is_err());
        assert!(decode_block(80, 3, &mut out).is_ok());
        assert_eq!(out, [1, 1, 1, 1]);
    }

    #[test]
    fn table_5_26_selection() {
        assert!(matches!(sample_coding(0, 0).unwrap(), SampleCoding::None));
        assert!(matches!(sample_coding(1, 0).unwrap(), SampleCoding::Huffman(_)));
        assert!(matches!(sample_coding(1, 1).unwrap(), SampleCoding::Block { levels: 3, bits: 7 }));
        assert!(matches!(sample_coding(2, 3).unwrap(), SampleCoding::Block { levels: 5, bits: 10 }));
        assert!(matches!(sample_coding(7, 7).unwrap(), SampleCoding::Block { levels: 25, bits: 19 }));
        assert!(matches!(sample_coding(7, 6).unwrap(), SampleCoding::Huffman(_)));
        assert!(matches!(sample_coding(8, 7).unwrap(), SampleCoding::Raw { bits: 5 }));
        assert!(matches!(sample_coding(10, 7).unwrap(), SampleCoding::Raw { bits: 7 }));
        assert!(matches!(sample_coding(11, 0).unwrap(), SampleCoding::Raw { bits: 8 }));
        assert!(matches!(sample_coding(26, 0).unwrap(), SampleCoding::Raw { bits: 23 }));
        assert!(sample_coding(27, 0).is_err());
        assert!(sample_coding(1, 2).is_err(), "SEL past the group is invalid");
    }
}
