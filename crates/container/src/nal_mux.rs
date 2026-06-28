//! Mux-side H.264 / H.265 NAL handling: take the encoder's **Annex-B** output
//! (start-code-delimited NAL units), strip the out-of-band parameter sets
//! (SPS/PPS, plus HEVC VPS) for the `avcC`/`hvcC` config box, and repackage the
//! remaining NALs (slices, SEI) as **length-prefixed** (4-byte) samples for the
//! MP4 `mdat`. This is the inverse of the demux path in
//! [`annexb`](crate::annexb), which reads length-prefixed → Annex-B.
//!
//! `avc1`/`hvc1` carry the parameter sets in the sample-entry config box, not
//! in-band, so the per-sample data must NOT repeat them.

/// Which NAL codec the bitstream is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NalMuxCodec {
    H264,
    H265,
}

/// What a NAL unit is, for the mux split.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NalClass {
    Vps,
    Sps,
    Pps,
    /// Slice / SEI / AUD / anything else that belongs in the sample data.
    Sample,
}

/// Classify a NAL unit (payload only, no start code) for the given codec.
fn classify(nal: &[u8], codec: NalMuxCodec) -> NalClass {
    if nal.is_empty() {
        return NalClass::Sample;
    }
    match codec {
        // H.264 §7.3.1: nal_unit_type in the low 5 bits of byte 0.
        NalMuxCodec::H264 => match nal[0] & 0x1F {
            7 => NalClass::Sps,
            8 => NalClass::Pps,
            _ => NalClass::Sample,
        },
        // H.265 §7.3.1.2: nal_unit_type in bits 1..=6 of byte 0 (2-byte header).
        NalMuxCodec::H265 => match (nal[0] >> 1) & 0x3F {
            32 => NalClass::Vps,
            33 => NalClass::Sps,
            34 => NalClass::Pps,
            _ => NalClass::Sample,
        },
    }
}

/// Split an Annex-B buffer into its NAL units (payloads, start codes removed).
/// Handles both 3-byte (`00 00 01`) and 4-byte (`00 00 00 01`) start codes.
pub fn split_annexb_nals(data: &[u8]) -> Vec<&[u8]> {
    let mut nals = Vec::new();
    let n = data.len();
    // Position just past the first start code.
    let mut cursor = match find_start_code(data, 0) {
        Some((pos, len)) => pos + len,
        None => return nals, // no start code → not Annex-B / empty
    };
    loop {
        let (next_pos, next_len) = match find_start_code(data, cursor) {
            Some(x) => x,
            None => {
                // last NAL runs to the end (trim trailing zero bytes)
                let end = trim_trailing_zeros(&data[cursor..n]);
                if end > 0 {
                    nals.push(&data[cursor..cursor + end]);
                }
                break;
            }
        };
        let end = trim_trailing_zeros(&data[cursor..next_pos]);
        if end > 0 {
            nals.push(&data[cursor..cursor + end]);
        }
        cursor = next_pos + next_len;
    }
    nals
}

/// Find the next Annex-B start code at/after `from`; returns (offset, length).
fn find_start_code(data: &[u8], from: usize) -> Option<(usize, usize)> {
    let n = data.len();
    let mut i = from;
    while i + 3 <= n {
        if data[i] == 0 && data[i + 1] == 0 {
            if data[i + 2] == 1 {
                return Some((i, 3));
            }
            if i + 4 <= n && data[i + 2] == 0 && data[i + 3] == 1 {
                return Some((i, 4));
            }
        }
        i += 1;
    }
    None
}

/// Trailing-zero trim so a 4-byte start code's leading `00` (which the 3-byte
/// scan attributes to the previous NAL) isn't kept as NAL payload.
fn trim_trailing_zeros(s: &[u8]) -> usize {
    let mut end = s.len();
    while end > 0 && s[end - 1] == 0 {
        end -= 1;
    }
    end
}

/// Repackages Annex-B encoder frames into length-prefixed mdat samples while
/// collecting the out-of-band parameter sets for the `avcC`/`hvcC` config box.
#[derive(Debug)]
pub struct NalSampleWriter {
    codec: NalMuxCodec,
    /// HEVC VPS NAL units (empty for H.264), first-seen order, de-duplicated.
    pub vps: Vec<Vec<u8>>,
    pub sps: Vec<Vec<u8>>,
    pub pps: Vec<Vec<u8>>,
}

impl NalSampleWriter {
    pub fn new(codec: NalMuxCodec) -> Self {
        Self { codec, vps: Vec::new(), sps: Vec::new(), pps: Vec::new() }
    }

    /// Convert one Annex-B encoded frame to a **length-prefixed** (4-byte
    /// big-endian length + NAL) mdat sample. SPS/PPS/VPS are captured (for the
    /// config box) and stripped from the returned sample.
    pub fn push_frame(&mut self, annexb: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(annexb.len());
        for nal in split_annexb_nals(annexb) {
            match classify(nal, self.codec) {
                NalClass::Vps => dedup_push(&mut self.vps, nal),
                NalClass::Sps => dedup_push(&mut self.sps, nal),
                NalClass::Pps => dedup_push(&mut self.pps, nal),
                NalClass::Sample => {
                    out.extend_from_slice(&(nal.len() as u32).to_be_bytes());
                    out.extend_from_slice(nal);
                }
            }
        }
        out
    }

    /// Whether the parameter sets needed for the config box have been seen.
    pub fn has_param_sets(&self) -> bool {
        let vps_ok = matches!(self.codec, NalMuxCodec::H264) || !self.vps.is_empty();
        vps_ok && !self.sps.is_empty() && !self.pps.is_empty()
    }
}

fn dedup_push(set: &mut Vec<Vec<u8>>, nal: &[u8]) {
    if !set.iter().any(|n| n.as_slice() == nal) {
        set.push(nal.to_vec());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sc4(nal: &[u8]) -> Vec<u8> {
        let mut v = vec![0, 0, 0, 1];
        v.extend_from_slice(nal);
        v
    }

    #[test]
    fn splits_3_and_4_byte_start_codes() {
        // 4-byte SC, then 3-byte SC
        let mut buf = vec![0, 0, 0, 1, 0xAA, 0xBB];
        buf.extend_from_slice(&[0, 0, 1, 0xCC]);
        let nals = split_annexb_nals(&buf);
        assert_eq!(nals.len(), 2);
        assert_eq!(nals[0], &[0xAA, 0xBB]);
        assert_eq!(nals[1], &[0xCC]);
    }

    #[test]
    fn h264_strips_sps_pps_keeps_slice() {
        // SPS (type 7), PPS (type 8), IDR slice (type 5)
        let sps = [0x67u8, 0x42, 0x00, 0x1e, 0xAA];
        let pps = [0x68u8, 0xCE, 0x3C];
        let idr = [0x65u8, 0x88, 0x11, 0x22];
        let mut frame = sc4(&sps);
        frame.extend(sc4(&pps));
        frame.extend(sc4(&idr));
        let mut w = NalSampleWriter::new(NalMuxCodec::H264);
        let sample = w.push_frame(&frame);
        // captured param sets
        assert_eq!(w.sps, vec![sps.to_vec()]);
        assert_eq!(w.pps, vec![pps.to_vec()]);
        assert!(w.has_param_sets());
        // sample = length-prefixed IDR only
        let mut expect = (idr.len() as u32).to_be_bytes().to_vec();
        expect.extend_from_slice(&idr);
        assert_eq!(sample, expect);
    }

    #[test]
    fn h265_captures_vps_sps_pps() {
        let vps = [0x40u8, 0x01, 0x0c]; // type 32
        let sps = [0x42u8, 0x01, 0x01]; // type 33
        let pps = [0x44u8, 0x01, 0xc1]; // type 34
        let slice = [0x26u8, 0x01, 0xaf]; // type 19 (IDR_W_RADL)
        let mut frame = sc4(&vps);
        frame.extend(sc4(&sps));
        frame.extend(sc4(&pps));
        frame.extend(sc4(&slice));
        let mut w = NalSampleWriter::new(NalMuxCodec::H265);
        let sample = w.push_frame(&frame);
        assert_eq!(w.vps, vec![vps.to_vec()]);
        assert_eq!(w.sps, vec![sps.to_vec()]);
        assert_eq!(w.pps, vec![pps.to_vec()]);
        assert!(w.has_param_sets());
        let mut expect = (slice.len() as u32).to_be_bytes().to_vec();
        expect.extend_from_slice(&slice);
        assert_eq!(sample, expect);
    }

    #[test]
    fn dedups_repeated_param_sets() {
        let sps = [0x67u8, 0x42, 0x00, 0x1e];
        let pps = [0x68u8, 0xCE, 0x3C];
        let idr = [0x65u8, 0x88];
        let mut w = NalSampleWriter::new(NalMuxCodec::H264);
        // two frames each repeating SPS/PPS (HW encoders often do this)
        for _ in 0..2 {
            let mut f = sc4(&sps);
            f.extend(sc4(&pps));
            f.extend(sc4(&idr));
            w.push_frame(&f);
        }
        assert_eq!(w.sps.len(), 1);
        assert_eq!(w.pps.len(), 1);
    }
}
