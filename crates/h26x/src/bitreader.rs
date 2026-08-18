//! MSB-first bit reader over an RBSP (emulation-prevention bytes already
//! removed — see [`crate::nal::unescape_rbsp`]).
//!
//! Reads past the end return zeros and set a sticky overrun flag rather than
//! failing at every call: header parsers read dozens of fields in a row, and
//! checking once at the end (`finish`) keeps them readable. The overrun is
//! never silent — it turns into a bitstream error.

use crate::{Error, Result};

/// A bit reader with a 64-bit cache.
#[derive(Clone)]
pub struct BitReader<'a> {
    data: &'a [u8],
    /// Next byte to load into the cache.
    pos: usize,
    /// Left-aligned bits not yet consumed.
    cache: u64,
    /// How many bits of `cache` are valid.
    bits: u32,
    /// Bits consumed so far (from the start of `data`).
    consumed: u64,
    overrun: bool,
}

impl<'a> BitReader<'a> {
    /// A reader positioned at the first bit of `data`.
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0, cache: 0, bits: 0, consumed: 0, overrun: false }
    }

    /// Total number of bits in the underlying data.
    pub fn len_bits(&self) -> u64 {
        (self.data.len() as u64) * 8
    }

    /// Bits consumed so far.
    pub fn position(&self) -> u64 {
        self.consumed
    }

    /// Bits left, saturating at zero.
    pub fn bits_left(&self) -> u64 {
        self.len_bits().saturating_sub(self.consumed)
    }

    /// Whether any read went past the end.
    pub fn overrun(&self) -> bool {
        self.overrun
    }

    /// The underlying bytes.
    pub fn data(&self) -> &'a [u8] {
        self.data
    }

    #[inline(always)]
    fn refill(&mut self) {
        // Top up to at least 57 valid bits: whole words while eight bytes are
        // there, then one byte at a time. Past the end of the data, zeros
        // come in and the overrun flag is set only when those zeros are
        // actually consumed (see `bits`).
        if self.bits <= 32 && self.pos + 8 <= self.data.len() {
            // Bring in as many whole bytes as fit: (64 - bits) / 8.
            let n = ((64 - self.bits) / 8) as usize;
            let take = 8 * n as u32;
            let w = u64::from_be_bytes(self.data[self.pos..self.pos + 8].try_into().unwrap());
            // The top `take` bits of the word, right below the valid bits;
            // nothing below them (the next refill relies on zeros there).
            let top = if take == 64 { w } else { w >> (64 - take) };
            self.cache |= top << (64 - self.bits - take);
            self.pos += n;
            self.bits += take;
            return;
        }
        while self.bits <= 56 {
            let byte = if self.pos < self.data.len() {
                let b = self.data[self.pos];
                self.pos += 1;
                b
            } else {
                // Virtual zero byte; count its position so `consumed` tracking
                // stays exact.
                self.pos += 1;
                0
            };
            self.cache |= (byte as u64) << (56 - self.bits);
            self.bits += 8;
        }
    }

    /// Read `n` bits (0..=32) as an unsigned value.
    #[inline(always)]
    pub fn bits(&mut self, n: u32) -> u32 {
        debug_assert!(n <= 32);
        if n == 0 {
            return 0;
        }
        if self.bits < n {
            self.refill();
        }
        let v = (self.cache >> (64 - n)) as u32;
        self.cache <<= n;
        self.bits -= n;
        self.consumed += n as u64;
        if self.consumed > self.len_bits() {
            self.overrun = true;
        }
        v
    }

    /// Read `n` bits (0..=64) as an unsigned 64-bit value.
    pub fn bits64(&mut self, n: u32) -> u64 {
        if n <= 32 {
            return self.bits(n) as u64;
        }
        let hi = self.bits(n - 32) as u64;
        let lo = self.bits(32) as u64;
        (hi << 32) | lo
    }

    /// Read one bit as a bool.
    #[inline(always)]
    pub fn flag(&mut self) -> bool {
        self.bits(1) != 0
    }

    /// Read one bit as 0/1.
    #[inline(always)]
    pub fn bit(&mut self) -> u32 {
        self.bits(1)
    }

    /// Peek `n` bits (0..=32) without consuming.
    #[inline(always)]
    pub fn peek(&mut self, n: u32) -> u32 {
        if n == 0 {
            return 0;
        }
        if self.bits < n {
            self.refill();
        }
        (self.cache >> (64 - n)) as u32
    }

    /// Skip `n` bits.
    #[inline(always)]
    pub fn skip(&mut self, mut n: u32) {
        while n > 32 {
            self.bits(32);
            n -= 32;
        }
        self.bits(n);
    }

    /// Exp-Golomb `ue(v)`.
    #[inline]
    pub fn ue(&mut self) -> u32 {
        // Count leading zeros using the cache directly.
        if self.bits < 32 {
            self.refill();
        }
        let top = (self.cache >> 32) as u32;
        if top == 0 {
            // More than 32 leading zeros: malformed. Consume and flag.
            self.overrun = true;
            self.skip(32);
            return 0;
        }
        let zeros = top.leading_zeros();
        if zeros == 0 {
            self.bits(1);
            return 0;
        }
        // Skip the zeros and the terminating one, then read `zeros` bits.
        self.skip(zeros + 1);
        let suffix = self.bits(zeros);
        ((1u64 << zeros) - 1 + suffix as u64) as u32
    }

    /// Exp-Golomb `se(v)`.
    #[inline]
    pub fn se(&mut self) -> i32 {
        let k = self.ue() as u64;
        if k & 1 == 1 {
            ((k + 1) / 2) as i32
        } else {
            -((k / 2) as i32)
        }
    }

    /// Truncated Exp-Golomb `te(v)` with range `max`.
    pub fn te(&mut self, max: u32) -> u32 {
        if max > 1 {
            self.ue()
        } else {
            1 - self.bit()
        }
    }

    /// Whether the reader is at a byte boundary.
    pub fn byte_aligned(&self) -> bool {
        self.consumed % 8 == 0
    }

    /// Advance to the next byte boundary (no-op if already aligned).
    pub fn align(&mut self) {
        let rem = (self.consumed % 8) as u32;
        if rem != 0 {
            self.skip(8 - rem);
        }
    }

    /// The byte offset of the current position (must be byte aligned).
    pub fn byte_position(&self) -> usize {
        (self.consumed / 8) as usize
    }

    /// The remaining bytes from the current (byte-aligned) position.
    pub fn remaining_bytes(&self) -> &'a [u8] {
        let p = self.byte_position().min(self.data.len());
        &self.data[p..]
    }

    /// `more_rbsp_data()`: true if there is more data before the
    /// `rbsp_trailing_bits`. The RBSP ends with a stop bit `1` followed by
    /// zero bits to the byte boundary (and possibly trailing zero bytes,
    /// which callers strip); so: are there any `1` bits after the current
    /// position other than the last one?
    pub fn more_rbsp_data(&self) -> bool {
        // Position of the last set bit in the data.
        let mut last = self.data.len();
        while last > 0 && self.data[last - 1] == 0 {
            last -= 1;
        }
        if last == 0 {
            return false;
        }
        let byte = self.data[last - 1];
        let tz = byte.trailing_zeros() as u64;
        let stop_bit_pos = (last as u64) * 8 - 1 - tz;
        self.consumed < stop_bit_pos
    }

    /// Fail with a bitstream error if any read overran the data.
    pub fn finish(&self, what: &str) -> Result<()> {
        if self.overrun {
            Err(Error::bitstream(format!("{what}: truncated (read past the end of the NAL unit)")))
        } else {
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_width_and_golomb() {
        // 1 | 010 | 011 | 00100 | 00101 | 1 | 0000
        // ue: 0, 1, 2, 3, 4 ; then a 1 bit
        let bytes = [0b1010_0110, 0b0100_0010, 0b1100_0000];
        let mut r = BitReader::new(&bytes);
        assert_eq!(r.ue(), 0);
        assert_eq!(r.ue(), 1);
        assert_eq!(r.ue(), 2);
        assert_eq!(r.ue(), 3);
        assert_eq!(r.ue(), 4);
        assert_eq!(r.bit(), 1);
        assert_eq!(r.position(), 18);
        assert!(!r.overrun());
        r.finish("test").unwrap();
    }

    #[test]
    fn signed_golomb() {
        // se: codeNum 1 -> +1, 2 -> -1, 3 -> +2, 4 -> -2
        let bytes = [0b0100_1100, 0b1000_0101, 0b0000_0000];
        let mut r = BitReader::new(&bytes);
        assert_eq!(r.se(), 1);
        assert_eq!(r.se(), -1);
        assert_eq!(r.se(), 2);
        assert_eq!(r.se(), -2);
    }

    #[test]
    fn overrun_is_sticky() {
        let bytes = [0xff];
        let mut r = BitReader::new(&bytes);
        assert_eq!(r.bits(8), 0xff);
        assert!(!r.overrun());
        assert_eq!(r.bits(4), 0);
        assert!(r.overrun());
        assert!(r.finish("x").is_err());
    }

    #[test]
    fn more_rbsp_data_sees_the_stop_bit() {
        // payload bits: 1 0 1, then stop bit 1, then zeros.
        let bytes = [0b1011_0000];
        let mut r = BitReader::new(&bytes);
        assert!(r.more_rbsp_data());
        r.bits(3);
        assert!(!r.more_rbsp_data());
        // With trailing zero bytes (cabac_zero_words) too.
        let bytes = [0b1011_0000, 0, 0];
        let mut r = BitReader::new(&bytes);
        r.bits(3);
        assert!(!r.more_rbsp_data());
    }

    #[test]
    fn peek_and_align() {
        let bytes = [0b1100_0000, 0b1010_1010];
        let mut r = BitReader::new(&bytes);
        assert_eq!(r.peek(2), 0b11);
        r.bits(2);
        assert!(!r.byte_aligned());
        r.align();
        assert!(r.byte_aligned());
        assert_eq!(r.byte_position(), 1);
        assert_eq!(r.bits(8), 0b1010_1010);
    }
}
