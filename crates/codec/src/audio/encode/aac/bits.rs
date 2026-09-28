//! MSB-first bit writer for the AAC bitstream syntax (`uimsbf` / `bslbf`
//! fields and Huffman codewords are all written most significant bit first).

#[derive(Default)]
pub(super) struct BitWriter {
    buf: Vec<u8>,
    /// Pending bits, right-aligned; `acc_bits` of them are valid.
    acc: u64,
    acc_bits: u32,
}

impl BitWriter {
    pub fn with_capacity(bytes: usize) -> Self {
        Self {
            buf: Vec::with_capacity(bytes),
            acc: 0,
            acc_bits: 0,
        }
    }

    /// Append the low `n` bits of `value` (`n <= 32`).
    pub fn put(&mut self, value: u32, n: u32) {
        debug_assert!(n <= 32);
        debug_assert!(
            n == 32 || value >> n == 0,
            "{value:#x} does not fit {n} bits"
        );
        if n == 0 {
            return;
        }
        self.acc = (self.acc << n) | u64::from(value);
        self.acc_bits += n;
        while self.acc_bits >= 8 {
            self.acc_bits -= 8;
            self.buf.push((self.acc >> self.acc_bits) as u8);
        }
        self.acc &= (1u64 << self.acc_bits) - 1;
    }

    /// Bits written so far.
    pub fn len_bits(&self) -> usize {
        self.buf.len() * 8 + self.acc_bits as usize
    }

    /// Pad with zero bits to the next byte boundary.
    pub fn align(&mut self) {
        let pad = (8 - self.acc_bits % 8) % 8;
        self.put(0, pad);
    }

    /// The written bytes; the writer must be byte-aligned.
    pub fn into_bytes(self) -> Vec<u8> {
        debug_assert_eq!(self.acc_bits, 0, "BitWriter finished mid-byte");
        self.buf
    }
}

#[cfg(test)]
mod tests {
    use super::BitWriter;

    #[test]
    fn writes_msb_first_across_byte_boundaries() {
        let mut w = BitWriter::default();
        w.put(0b101, 3);
        w.put(0x3ffe8, 18);
        w.put(1, 1);
        assert_eq!(w.len_bits(), 22);
        w.align();
        // 101 | 11 1111 1111 1110 1000 | 1 | 00
        assert_eq!(w.into_bytes(), vec![0b1011_1111, 0b1111_1111, 0b0100_0100]);
    }
}
