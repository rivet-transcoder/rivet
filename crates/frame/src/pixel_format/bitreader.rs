//! MSB-first bit reader + shared low-level helpers used by multiple
//! codec parsers in this module.

// ─── Bit reader ────────────────────────────────────────────────────
pub(super) struct BitReader<'a> {
    pub(super) data: &'a [u8],
    pub(super) pos: usize,
}

impl<'a> BitReader<'a> {
    pub(super) fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    pub(super) fn read_bits(&mut self, n: usize) -> Option<u32> {
        let mut val = 0u32;
        for _ in 0..n {
            let byte_idx = self.pos / 8;
            let bit_idx = 7 - (self.pos % 8);
            if byte_idx >= self.data.len() {
                return None;
            }
            val = (val << 1) | (((self.data[byte_idx] >> bit_idx) & 1) as u32);
            self.pos += 1;
        }
        Some(val)
    }

    /// Exp-Golomb unsigned — used by H.264 and HEVC SPS fields.
    pub(super) fn read_ue(&mut self) -> Option<u32> {
        let mut zeros = 0;
        while self.read_bits(1)? == 0 {
            zeros += 1;
            if zeros > 31 {
                // Cap before `1u32 << 32` would panic. 31 zeros already
                // allow any value up to u32::MAX; any SPS field we care
                // about fits within ~10 zeros.
                return None;
            }
        }
        if zeros == 0 {
            return Some(0);
        }
        let suffix = self.read_bits(zeros)?;
        Some((1u32 << zeros) - 1 + suffix)
    }

    /// Signed Exp-Golomb (se(v)). H.264 §9.1.1: `codeNum` from `read_ue`,
    /// then `(-1)^(codeNum+1) * ceil(codeNum/2)` — odd codeNums map to
    /// positive values, even to negative (or zero for codeNum=0).
    /// Used by H.264 SPS `scaling_list` deltas and `pic_order_cnt_type==1`
    /// offsets.
    pub(super) fn read_se(&mut self) -> Option<i32> {
        let code = self.read_ue()? as i64;
        let signed = if code & 1 == 1 {
            ((code + 1) / 2) as i32
        } else {
            -((code / 2) as i32)
        };
        Some(signed)
    }

    /// Current bit position within `data`. Used by AV1 parsers to find
    /// the byte-aligned end of uncompressed_header().
    pub(super) fn bit_pos(&self) -> usize {
        self.pos
    }

    /// Advance the bit cursor to the next byte boundary. AV1 spec
    /// byte_alignment() per §5.3.5: skip bits until `pos % 8 == 0`.
    pub(super) fn byte_align(&mut self) {
        let rem = self.pos & 7;
        if rem != 0 {
            self.pos += 8 - rem;
        }
    }

    /// AV1 signed `su(n)` — n-bit two's-complement signed integer
    /// (§4.10.5). Read n bits, sign-extend from bit n-1.
    pub(super) fn read_su(&mut self, n: usize) -> Option<i32> {
        let raw = self.read_bits(n)?;
        let sign_bit = 1u32 << (n - 1);
        let signed = if raw & sign_bit != 0 {
            (raw as i32) - (1i32 << n)
        } else {
            raw as i32
        };
        Some(signed)
    }
}

// ─── Shared low-level helpers ───────────────────────────────────────

pub(super) fn find_next_start_code(data: &[u8]) -> Option<usize> {
    (0..data.len().saturating_sub(3)).find(|&i| {
        data[i] == 0
            && data[i + 1] == 0
            && (data[i + 2] == 1 || (data[i + 2] == 0 && data[i + 3] == 1))
    })
}

/// Strip H.264 / HEVC emulation-prevention bytes (0x00 0x00 0x03 → 0x00 0x00).
pub(super) fn remove_h264_rbsp_stuffing(sps: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(sps.len());
    let mut i = 0;
    while i < sps.len() {
        if i + 2 < sps.len() && sps[i] == 0 && sps[i + 1] == 0 && sps[i + 2] == 3 {
            out.push(0);
            out.push(0);
            i += 3;
        } else {
            out.push(sps[i]);
            i += 1;
        }
    }
    out
}

pub(super) fn clamp_to_i8(v: i32) -> i8 {
    v.clamp(i8::MIN as i32, i8::MAX as i32) as i8
}

/// Heuristic more_rbsp_data() check — the spec defines it precisely
/// (position in RBSP trailing bits) but needs byte-alignment awareness
/// we don't expose from BitReader. Approximation: at least one more
/// full byte of input remains after the current cursor. Good enough
/// for the PPS extended-field branch since the trailing byte is a
/// stop bit + zero pad — parsing a spurious bit from that gives
/// `transform_8x8_mode_flag = true` which the caller tolerates.
pub(super) fn more_rbsp_data(br: &BitReader, rbsp: &[u8]) -> bool {
    let pos = br.pos;
    let total_bits = rbsp.len() * 8;
    // We need at least 1 payload bit + the 1-bit stop + up to 7 zero
    // pad bits. "More data" = at least 9 bits remain.
    total_bits.saturating_sub(pos) > 8
}
