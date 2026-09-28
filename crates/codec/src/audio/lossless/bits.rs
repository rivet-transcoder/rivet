//! MSB-first bit reader and writer for the FLAC and ALAC bitstreams.
//!
//! Both formats pack fields most-significant bit first, both code residuals
//! with a unary prefix, and both run past 32 bits in places (a FLAC side
//! channel of 32-bit audio is 33 bits wide), so the reader hands out up to 64
//! bits and the writer takes as many.

use crate::audio::AudioError;

pub(crate) struct BitReader<'a> {
    data: &'a [u8],
    /// Bits consumed from the start of `data`.
    pos: usize,
    /// Which codec is reading, for the error text.
    codec: &'static str,
}

impl<'a> BitReader<'a> {
    pub fn new(data: &'a [u8], codec: &'static str) -> Self {
        Self { data, pos: 0, codec }
    }

    /// Bits consumed so far.
    pub fn pos(&self) -> usize {
        self.pos
    }

    /// Bits left before the end of the data.
    pub fn remaining(&self) -> usize {
        self.data.len() * 8 - self.pos
    }

    fn overrun(&self, n: u32) -> AudioError {
        AudioError::Decode(format!(
            "{}: read of {n} bits at bit {} runs past the end of a {}-byte packet",
            self.codec,
            self.pos,
            self.data.len()
        ))
    }

    /// Read `n` (≤ 64) bits as an unsigned value.
    pub fn read(&mut self, n: u32) -> Result<u64, AudioError> {
        debug_assert!(n <= 64);
        if n == 0 {
            return Ok(0);
        }
        if n as usize > self.remaining() {
            return Err(self.overrun(n));
        }
        let mut v: u64 = 0;
        let mut left = n;
        while left > 0 {
            let byte = self.data[self.pos / 8];
            let off = (self.pos % 8) as u32;
            let avail = 8 - off;
            let take = avail.min(left);
            let bits = (u64::from(byte) >> (avail - take)) & ((1u64 << take) - 1);
            v = if take == 64 { bits } else { (v << take) | bits };
            left -= take;
            self.pos += take as usize;
        }
        Ok(v)
    }

    /// Read `n` (≤ 32) bits as an unsigned value.
    pub fn read_u32(&mut self, n: u32) -> Result<u32, AudioError> {
        debug_assert!(n <= 32);
        Ok(self.read(n)? as u32)
    }

    pub fn read_bit(&mut self) -> Result<bool, AudioError> {
        Ok(self.read(1)? == 1)
    }

    /// Read `n` (1..=64) bits as a two's-complement value.
    pub fn read_signed(&mut self, n: u32) -> Result<i64, AudioError> {
        if n == 0 {
            return Ok(0);
        }
        let v = self.read(n)?;
        let shift = 64 - n;
        Ok(((v << shift) as i64) >> shift)
    }

    /// Count `0` bits up to the next `1`, and consume that `1` (FLAC's
    /// unary: the quotient of a Rice code, the wasted-bits count).
    pub fn read_unary_zeros(&mut self) -> Result<u32, AudioError> {
        let mut n = 0u32;
        loop {
            if self.pos >= self.data.len() * 8 {
                return Err(self.overrun(1));
            }
            let byte = self.data[self.pos / 8];
            let off = self.pos % 8;
            let rest = byte << off;
            if rest != 0 {
                let z = rest.leading_zeros();
                self.pos += z as usize + 1;
                return Ok(n + z);
            }
            let skipped = 8 - off as u32;
            n += skipped;
            self.pos += skipped as usize;
        }
    }

    /// Count `1` bits up to the next `0` or until `limit` of them have been
    /// read, consuming the `0` when one ends the run (ALAC's unary prefix,
    /// which has an escape at `limit`).
    pub fn read_unary_ones(&mut self, limit: u32) -> Result<u32, AudioError> {
        let mut n = 0u32;
        while n < limit {
            if !self.read_bit()? {
                return Ok(n);
            }
            n += 1;
        }
        Ok(n)
    }

    /// Give back the last `n` bits read.
    pub fn unread(&mut self, n: usize) {
        debug_assert!(n <= self.pos);
        self.pos -= n;
    }

    pub fn skip(&mut self, n: usize) -> Result<(), AudioError> {
        if n > self.remaining() {
            return Err(self.overrun(n as u32));
        }
        self.pos += n;
        Ok(())
    }

    /// Skip to the next byte boundary.
    pub fn align(&mut self) {
        self.pos = self.pos.div_ceil(8) * 8;
    }

    /// The whole bytes consumed so far (after [`align`](Self::align)).
    pub fn byte_pos(&self) -> usize {
        self.pos / 8
    }
}

/// Accumulates bits MSB first into a byte vector.
#[derive(Default)]
pub(crate) struct BitWriter {
    bytes: Vec<u8>,
    /// Bits waiting to fill a byte, right-aligned in `acc`.
    acc: u64,
    /// How many bits of `acc` are live (always < 8 between calls).
    live: u32,
}

impl BitWriter {
    pub fn with_capacity(bytes: usize) -> Self {
        Self { bytes: Vec::with_capacity(bytes), acc: 0, live: 0 }
    }

    /// Write the low `n` (≤ 64) bits of `v`.
    pub fn write(&mut self, v: u64, n: u32) {
        debug_assert!(n <= 64);
        if n == 0 {
            return;
        }
        // Split so the accumulator (≤ 7 live bits) never takes more than 56.
        if n > 32 {
            self.write(v >> 32, n - 32);
            self.write(v & 0xFFFF_FFFF, 32);
            return;
        }
        let v = v & ((1u64 << n) - 1);
        self.acc = (self.acc << n) | v;
        self.live += n;
        while self.live >= 8 {
            self.live -= 8;
            self.bytes.push((self.acc >> self.live) as u8);
        }
        self.acc &= (1u64 << self.live) - 1;
    }

    pub fn write_bit(&mut self, b: bool) {
        self.write(u64::from(b), 1);
    }

    /// Write `v` as an `n`-bit two's-complement field.
    pub fn write_signed(&mut self, v: i64, n: u32) {
        self.write(v as u64, n);
    }

    /// `n` zeros then a `1` (FLAC's unary).
    pub fn write_unary_zeros(&mut self, mut n: u32) {
        while n >= 32 {
            self.write(0, 32);
            n -= 32;
        }
        self.write(1, n + 1);
    }

    /// Bits written so far.
    pub fn len_bits(&self) -> usize {
        self.bytes.len() * 8 + self.live as usize
    }

    /// Pad with zeros to a byte boundary.
    pub fn align(&mut self) {
        if self.live > 0 {
            self.write(0, 8 - self.live);
        }
    }

    /// The bytes so far; only whole bytes (call [`align`](Self::align) first).
    pub fn bytes(&self) -> &[u8] {
        debug_assert_eq!(self.live, 0);
        &self.bytes
    }

    pub fn into_bytes(mut self) -> Vec<u8> {
        self.align();
        self.bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_every_width() {
        let mut w = BitWriter::default();
        let fields: Vec<(u64, u32)> = (1..=64).map(|n| (0xA5A5_5A5A_F00F_0FF0u64.rotate_left(n), n)).collect();
        for &(v, n) in &fields {
            w.write(v, n);
        }
        let bytes = w.into_bytes();
        let mut r = BitReader::new(&bytes, "test");
        for &(v, n) in &fields {
            let mask = if n == 64 { u64::MAX } else { (1u64 << n) - 1 };
            assert_eq!(r.read(n).unwrap(), v & mask, "{n} bits");
        }
    }

    #[test]
    fn signed_and_unary() {
        let mut w = BitWriter::default();
        w.write_signed(-5, 7);
        w.write_unary_zeros(0);
        w.write_unary_zeros(13);
        w.write_unary_zeros(70);
        w.write(0b1110, 4);
        w.write_signed(-(1i64 << 32), 33);
        let bytes = w.into_bytes();
        let mut r = BitReader::new(&bytes, "test");
        assert_eq!(r.read_signed(7).unwrap(), -5);
        assert_eq!(r.read_unary_zeros().unwrap(), 0);
        assert_eq!(r.read_unary_zeros().unwrap(), 13);
        assert_eq!(r.read_unary_zeros().unwrap(), 70);
        assert_eq!(r.read_unary_ones(9).unwrap(), 3);
        assert_eq!(r.read_signed(33).unwrap(), -(1i64 << 32));
    }

    #[test]
    fn overrun_is_an_error() {
        let mut r = BitReader::new(&[0x00], "test");
        assert!(r.read_unary_zeros().is_err());
        let mut r = BitReader::new(&[0xFF], "test");
        assert!(r.read(9).is_err());
    }
}
