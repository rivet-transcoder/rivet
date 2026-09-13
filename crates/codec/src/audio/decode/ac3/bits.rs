//! MSB-first bit reader for AC-3 syncframes.
//!
//! Reads past the end of the frame are an error, not zero padding: a
//! mantissa field that runs off the frame means the side information was
//! mis-parsed, and silently reading zeros would turn that into plausible
//! garbage instead of a diagnosable failure.

use crate::audio::AudioError;

pub(super) struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> BitReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    /// Bits consumed so far from the start of the frame.
    pub fn pos(&self) -> usize {
        self.pos
    }

    fn overrun(&self, n: u32) -> AudioError {
        AudioError::Decode(format!(
            "ac3: read of {n} bits at bit {} runs past the end of a {}-byte frame",
            self.pos,
            self.data.len()
        ))
    }

    /// Read `n` (≤ 32) bits, MSB first.
    pub fn read(&mut self, n: u32) -> Result<u32, AudioError> {
        debug_assert!(n <= 32);
        if n == 0 {
            return Ok(0);
        }
        if self.pos + n as usize > self.data.len() * 8 {
            return Err(self.overrun(n));
        }
        let mut v: u64 = 0;
        let mut got = 0u32;
        while got < n {
            let byte = self.data[self.pos / 8];
            let off = (self.pos % 8) as u32;
            let avail = 8 - off;
            let take = avail.min(n - got);
            let bits = (u32::from(byte) >> (avail - take)) & ((1 << take) - 1);
            v = (v << take) | u64::from(bits);
            got += take;
            self.pos += take as usize;
        }
        Ok(v as u32)
    }

    pub fn read_bit(&mut self) -> Result<bool, AudioError> {
        Ok(self.read(1)? == 1)
    }

    /// Read `n` (1..=32) bits as a two's-complement signed value.
    pub fn read_signed(&mut self, n: u32) -> Result<i32, AudioError> {
        let v = self.read(n)?;
        let shift = 32 - n;
        Ok(((v << shift) as i32) >> shift)
    }

    pub fn skip(&mut self, n: usize) -> Result<(), AudioError> {
        if self.pos + n > self.data.len() * 8 {
            return Err(self.overrun(n as u32));
        }
        self.pos += n;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_msb_first_across_byte_boundaries() {
        let data = [0b1011_0011, 0b0101_1111, 0xff];
        let mut br = BitReader::new(&data);
        assert_eq!(br.read(3).unwrap(), 0b101);
        assert_eq!(br.read(7).unwrap(), 0b1_0011_01);
        assert_eq!(br.read_signed(3).unwrap(), 3); // next bits 011 → +3
        assert_eq!(br.read_signed(3).unwrap(), -1); // 111 → −1
    }

    #[test]
    fn signed_reads_sign_extend() {
        let data = [0b1000_0000, 0b0111_1111];
        let mut br = BitReader::new(&data);
        assert_eq!(br.read_signed(3).unwrap(), -4);
        assert_eq!(br.read_signed(5).unwrap(), 0);
        assert_eq!(br.read_signed(8).unwrap(), 127);
    }

    #[test]
    fn overrun_is_an_error_not_zero_padding() {
        let data = [0xab];
        let mut br = BitReader::new(&data);
        assert_eq!(br.read(8).unwrap(), 0xab);
        assert!(br.read(1).is_err());
        assert!(br.skip(1).is_err());
    }
}
