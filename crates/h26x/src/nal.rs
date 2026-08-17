//! Annex-B framing and emulation prevention, shared by both codecs.
//!
//! Samples reach the decoders as Annex-B access units: NAL units separated by
//! `00 00 01` (or `00 00 00 01`) start codes, each NAL escaped so no
//! `00 00 0x` (x ≤ 3) pattern occurs inside it. The decoders want the reverse:
//! one NAL at a time, unescaped, with the header split off.

/// Iterate over the NAL units of an Annex-B byte stream (payloads *with*
/// their NAL header bytes, still escaped). Leading zeros before a start code
/// and trailing zero bytes of a NAL (`trailing_zero_8bits`) are dropped.
pub fn annexb_nals(data: &[u8]) -> impl Iterator<Item = &[u8]> {
    AnnexBIter { data, pos: 0 }
}

struct AnnexBIter<'a> {
    data: &'a [u8],
    pos: usize,
}

/// Find the next `00 00 01` at or after `from`; returns the index of the
/// first `00`.
fn find_start_code(data: &[u8], from: usize) -> Option<usize> {
    if data.len() < 3 {
        return None;
    }
    let mut i = from;
    while i + 2 < data.len() {
        if data[i + 2] > 1 {
            i += 3;
        } else if data[i + 2] == 0 {
            i += 1;
        } else if data[i] == 0 && data[i + 1] == 0 {
            return Some(i);
        } else {
            i += 3;
        }
    }
    None
}

impl<'a> Iterator for AnnexBIter<'a> {
    type Item = &'a [u8];
    fn next(&mut self) -> Option<&'a [u8]> {
        let data = self.data;
        if self.pos >= data.len() {
            return None;
        }
        // Skip to the first start code if we are at the beginning, else we
        // are already positioned right after one.
        let mut start = self.pos;
        if self.pos == 0 {
            match find_start_code(data, 0) {
                Some(sc) => start = sc + 3,
                None => {
                    // No start code at all: treat the whole buffer as one NAL
                    // (a caller that hands us a raw NAL).
                    self.pos = data.len();
                    let nal = trim_trailing_zeros(data);
                    return if nal.is_empty() { None } else { Some(nal) };
                }
            }
        }
        if start >= data.len() {
            self.pos = data.len();
            return None;
        }
        let end = match find_start_code(data, start) {
            Some(sc) => sc,
            None => data.len(),
        };
        self.pos = end + 3;
        let nal = trim_trailing_zeros(&data[start..end]);
        if nal.is_empty() {
            // Two start codes back to back (or a 4-byte one seen as `00` +
            // `00 00 01`); keep going.
            return self.next();
        }
        Some(nal)
    }
}

fn trim_trailing_zeros(mut nal: &[u8]) -> &[u8] {
    while let Some((&0, rest)) = nal.split_last() {
        nal = rest;
    }
    nal
}

/// Remove emulation-prevention bytes: every `00 00 03` becomes `00 00`.
/// Returns the RBSP (with the NAL header bytes still at the front — the
/// header is never escaped, so callers can slice it off before or after).
pub fn unescape_rbsp(nal: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(nal.len());
    let mut zeros = 0usize;
    let mut i = 0;
    while i < nal.len() {
        let b = nal[i];
        if zeros >= 2 && b == 3 {
            // Emulation prevention byte: drop it, unless it is the very last
            // byte (then it is a `cabac_zero_word` artefact and harmless
            // either way).
            zeros = 0;
            i += 1;
            continue;
        }
        out.push(b);
        if b == 0 {
            zeros += 1;
        } else {
            zeros = 0;
        }
        i += 1;
    }
    out
}

/// H.264 NAL unit header (one byte).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct H264NalHeader {
    /// `nal_ref_idc`: nonzero means the picture is a reference picture.
    pub ref_idc: u8,
    /// `nal_unit_type`.
    pub unit_type: u8,
}

impl H264NalHeader {
    /// Parse the first byte of an H.264 NAL unit. `None` if the forbidden bit
    /// is set (or the slice is empty).
    pub fn parse(nal: &[u8]) -> Option<Self> {
        let b = *nal.first()?;
        if b & 0x80 != 0 {
            return None;
        }
        Some(Self { ref_idc: (b >> 5) & 3, unit_type: b & 0x1f })
    }
}

/// HEVC NAL unit header (two bytes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HevcNalHeader {
    /// `nal_unit_type` (0..=63).
    pub unit_type: u8,
    /// `nuh_layer_id`.
    pub layer_id: u8,
    /// `nuh_temporal_id_plus1 - 1`.
    pub temporal_id: u8,
}

impl HevcNalHeader {
    /// Parse the first two bytes of an HEVC NAL unit.
    pub fn parse(nal: &[u8]) -> Option<Self> {
        if nal.len() < 2 || nal[0] & 0x80 != 0 {
            return None;
        }
        let tid_plus1 = nal[1] & 7;
        if tid_plus1 == 0 {
            return None;
        }
        Some(Self {
            unit_type: (nal[0] >> 1) & 0x3f,
            layer_id: ((nal[0] & 1) << 5) | (nal[1] >> 3),
            temporal_id: tid_plus1 - 1,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_three_and_four_byte_start_codes() {
        let data = [0, 0, 0, 1, 0x67, 0xaa, 0, 0, 1, 0x68, 0xbb, 0xcc, 0, 0, 0, 1, 0x65, 1, 2, 0, 0];
        let nals: Vec<&[u8]> = annexb_nals(&data).collect();
        assert_eq!(nals, vec![&[0x67, 0xaa][..], &[0x68, 0xbb, 0xcc][..], &[0x65, 1, 2][..]]);
    }

    #[test]
    fn a_raw_nal_without_start_code_is_one_nal() {
        let data = [0x67, 0x42, 0, 0x1e];
        let nals: Vec<&[u8]> = annexb_nals(&data).collect();
        assert_eq!(nals, vec![&data[..]]);
    }

    #[test]
    fn unescapes_emulation_prevention() {
        assert_eq!(unescape_rbsp(&[0x65, 0, 0, 3, 1, 0, 0, 3, 0, 5]), vec![0x65, 0, 0, 1, 0, 0, 0, 5]);
        // 00 00 03 03: the second 03 is data.
        assert_eq!(unescape_rbsp(&[0, 0, 3, 3]), vec![0, 0, 3]);
        // A lone 03 is data.
        assert_eq!(unescape_rbsp(&[0, 3, 0, 3]), vec![0, 3, 0, 3]);
    }

    #[test]
    fn nal_headers() {
        assert_eq!(H264NalHeader::parse(&[0x65]), Some(H264NalHeader { ref_idc: 3, unit_type: 5 }));
        assert_eq!(H264NalHeader::parse(&[0x41]), Some(H264NalHeader { ref_idc: 2, unit_type: 1 }));
        assert!(H264NalHeader::parse(&[0x85]).is_none());
        // HEVC IDR_W_RADL (19), layer 0, tid 0: 0x26 0x01
        assert_eq!(
            HevcNalHeader::parse(&[0x26, 0x01]),
            Some(HevcNalHeader { unit_type: 19, layer_id: 0, temporal_id: 0 })
        );
        // VPS (32): 0x40 0x01
        assert_eq!(HevcNalHeader::parse(&[0x40, 0x01]).unwrap().unit_type, 32);
    }
}
