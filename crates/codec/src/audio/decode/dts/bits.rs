//! MSB-first bit reader over one DTS core frame.
//!
//! The core frame is a single continuous bit string (ETSI TS 102 114 §5.2):
//! header, side information and audio arrays follow each other with no byte
//! alignment until the optional trailer, so one cursor over the whole frame
//! is all the decoder needs.

use super::DtsError;

pub struct BitReader<'a> {
    data: &'a [u8],
    /// Position in bits from the start of `data`.
    pos: usize,
}

impl<'a> BitReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    /// Bits still available.
    pub fn remaining(&self) -> usize {
        (self.data.len() * 8).saturating_sub(self.pos)
    }

    /// Read `n` bits (0..=32) as an unsigned value. Running past the end of
    /// the frame is an error: DTS frames carry their byte size in the header,
    /// so a short read means a corrupt or truncated frame, not an EOS.
    pub fn bits(&mut self, n: u32) -> Result<u32, DtsError> {
        debug_assert!(n <= 32);
        if n == 0 {
            return Ok(0);
        }
        if self.remaining() < n as usize {
            return Err(DtsError::Truncated {
                at_bit: self.pos,
                wanted: n,
            });
        }
        let mut v: u64 = 0;
        let mut left = n as usize;
        while left > 0 {
            let byte = self.data[self.pos >> 3] as u64;
            let bit_in_byte = self.pos & 7;
            let avail = 8 - bit_in_byte;
            let take = avail.min(left);
            let chunk = (byte >> (avail - take)) & ((1u64 << take) - 1);
            v = (v << take) | chunk;
            self.pos += take;
            left -= take;
        }
        Ok(v as u32)
    }

    /// Read a single bit as a flag.
    pub fn flag(&mut self) -> Result<bool, DtsError> {
        Ok(self.bits(1)? == 1)
    }

    /// Read `n` bits (1..=32) as a two's-complement signed value — the form
    /// the spec's `SignExtension(nCode)` produces for "no further encoding"
    /// quantisation indices and for the 8-bit LFE samples.
    pub fn sbits(&mut self, n: u32) -> Result<i32, DtsError> {
        debug_assert!((1..=32).contains(&n));
        let v = self.bits(n)?;
        let shift = 32 - n;
        Ok(((v << shift) as i32) >> shift)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::unusual_byte_groupings)] // grouped as the reads split them
    fn reads_msb_first_across_byte_boundaries() {
        let data = [0b1010_1100, 0b0101_0011, 0xFF];
        let mut r = BitReader::new(&data);
        assert_eq!(r.bits(3).unwrap(), 0b101);
        assert_eq!(r.bits(7).unwrap(), 0b0_1100_01);
        assert_eq!(r.bits(6).unwrap(), 0b01_0011);
        assert_eq!(r.remaining(), 8);
        assert_eq!(r.bits(8).unwrap(), 0xFF);
        assert_eq!(r.remaining(), 0);
    }

    #[test]
    fn sign_extends() {
        let data = [0b1111_1111, 0b0000_0001];
        let mut r = BitReader::new(&data);
        assert_eq!(r.sbits(8).unwrap(), -1);
        assert_eq!(r.sbits(8).unwrap(), 1);
        let data = [0b1000_0000];
        let mut r = BitReader::new(&data);
        assert_eq!(r.sbits(2).unwrap(), -2, "10b is -2 in two's complement");
    }

    #[test]
    fn overrun_is_an_error_not_zero_fill() {
        let data = [0xAB];
        let mut r = BitReader::new(&data);
        assert_eq!(r.bits(4).unwrap(), 0xA);
        assert!(matches!(
            r.bits(5),
            Err(DtsError::Truncated { at_bit: 4, wanted: 5 })
        ));
        // A failed read consumes nothing.
        assert_eq!(r.bits(4).unwrap(), 0xB);
    }

    #[test]
    fn thirty_two_bit_reads_work_unaligned() {
        let data = [0x0F, 0xFF, 0xFF, 0xFF, 0xF0];
        let mut r = BitReader::new(&data);
        assert_eq!(r.bits(4).unwrap(), 0);
        assert_eq!(r.bits(32).unwrap(), 0xFFFF_FFFF);
        assert_eq!(r.bits(4).unwrap(), 0);
    }
}
