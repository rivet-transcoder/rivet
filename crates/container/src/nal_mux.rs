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

/// `nal_unit_type` for the given codec (0 for an empty NAL).
fn nal_type(nal: &[u8], codec: NalMuxCodec) -> u8 {
    if nal.is_empty() {
        return 0;
    }
    match codec {
        NalMuxCodec::H264 => nal[0] & 0x1F,           // H.264 §7.3.1
        NalMuxCodec::H265 => (nal[0] >> 1) & 0x3F,    // H.265 §7.3.1.2 (2-byte header)
    }
}

/// Classify a NAL unit (payload only, no start code) for the given codec.
fn classify(nal: &[u8], codec: NalMuxCodec) -> NalClass {
    match (codec, nal_type(nal, codec)) {
        (NalMuxCodec::H264, 7) => NalClass::Sps,
        (NalMuxCodec::H264, 8) => NalClass::Pps,
        (NalMuxCodec::H265, 32) => NalClass::Vps,
        (NalMuxCodec::H265, 33) => NalClass::Sps,
        (NalMuxCodec::H265, 34) => NalClass::Pps,
        _ => NalClass::Sample,
    }
}

/// Access-unit delimiter (H.264 type 9 / H.265 type 35) — starts a new frame.
fn is_aud(nal: &[u8], codec: NalMuxCodec) -> bool {
    match codec {
        NalMuxCodec::H264 => nal_type(nal, codec) == 9,
        NalMuxCodec::H265 => nal_type(nal, codec) == 35,
    }
}

/// Whether this NAL is an IDR / IRAP slice (a keyframe's VCL NAL).
fn is_idr(nal: &[u8], codec: NalMuxCodec) -> bool {
    match codec {
        NalMuxCodec::H264 => nal_type(nal, codec) == 5,              // IDR slice
        NalMuxCodec::H265 => matches!(nal_type(nal, codec), 16..=23), // BLA..IRAP
    }
}

/// One muxed access unit (frame): its length-prefixed sample bytes + whether
/// it is a keyframe.
#[derive(Debug, Clone)]
pub struct AuSample {
    pub data: Vec<u8>,
    pub is_keyframe: bool,
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
        // `find_start_code` reports a 4-byte start code at its first `00`, so the
        // NAL ends exactly at the next start code — legitimate trailing zero
        // bytes in the slice RBSP (cabac_zero_words, rbsp trailing) are kept.
        let (next_pos, next_len) = match find_start_code(data, cursor) {
            Some(x) => x,
            None => {
                if n > cursor {
                    nals.push(&data[cursor..n]); // last NAL runs to the end
                }
                break;
            }
        };
        if next_pos > cursor {
            nals.push(&data[cursor..next_pos]);
        }
        cursor = next_pos + next_len;
    }
    nals
}

/// Find the next start-code **prefix** `00 00 01` at/after `from`; returns
/// (offset, 3). We deliberately match only the 3-byte prefix: a 4-byte start
/// code `00 00 00 01` is then seen as `[zero_byte] [00 00 01]`, so the leading
/// `00` stays with the *previous* NAL as a harmless trailing zero (decoders
/// ignore it) rather than being greedily consumed — which would otherwise eat a
/// slice's own trailing `0x00` byte and corrupt it.
fn find_start_code(data: &[u8], from: usize) -> Option<(usize, usize)> {
    let n = data.len();
    let mut i = from;
    while i + 3 <= n {
        if data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1 {
            return Some((i, 3));
        }
        i += 1;
    }
    None
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

    /// Convert one encoder packet — which may carry **multiple access units**
    /// (HW encoders return several frames per buffer) — into one
    /// **length-prefixed** mdat sample *per access unit*. Access units are
    /// delimited by the AUD NAL (a packet with no AUD is treated as one unit).
    /// SPS/PPS/VPS are captured (for the config box) and stripped from samples.
    pub fn push_packet(&mut self, annexb: &[u8]) -> Vec<AuSample> {
        // Group NALs into access units: a new unit begins at each AUD.
        let mut units: Vec<Vec<&[u8]>> = vec![Vec::new()];
        for nal in split_annexb_nals(annexb) {
            if is_aud(nal, self.codec) && !units.last().unwrap().is_empty() {
                units.push(Vec::new());
            }
            units.last_mut().unwrap().push(nal);
        }

        let mut samples = Vec::new();
        for unit in units {
            let mut data = Vec::new();
            let mut is_keyframe = false;
            for nal in unit {
                match classify(nal, self.codec) {
                    NalClass::Vps => dedup_push(&mut self.vps, nal),
                    NalClass::Sps => dedup_push(&mut self.sps, nal),
                    NalClass::Pps => dedup_push(&mut self.pps, nal),
                    NalClass::Sample => {
                        if is_idr(nal, self.codec) {
                            is_keyframe = true;
                        }
                        data.extend_from_slice(&(nal.len() as u32).to_be_bytes());
                        data.extend_from_slice(nal);
                    }
                }
            }
            if !data.is_empty() {
                samples.push(AuSample { data, is_keyframe });
            }
        }
        samples
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
        let samples = w.push_packet(&frame);
        assert_eq!(samples.len(), 1, "no AUD → one access unit");
        assert!(samples[0].is_keyframe, "contains an IDR slice");
        // captured param sets (a 4-byte next start code may add a harmless
        // trailing 0x00, so check the param set is a prefix of what was captured)
        assert_eq!(w.sps.len(), 1);
        assert!(w.sps[0].starts_with(&sps));
        assert!(w.pps[0].starts_with(&pps));
        assert!(w.has_param_sets());
        // sample = length-prefixed IDR (the last NAL, no trailing start code → exact)
        let mut expect = (idr.len() as u32).to_be_bytes().to_vec();
        expect.extend_from_slice(&idr);
        assert_eq!(samples[0].data, expect);
    }

    #[test]
    fn splits_multi_au_packet_by_aud() {
        // A packet with two AUDs (type 9) → two access-unit samples.
        let aud = [0x09u8, 0x10];
        let idr = [0x65u8, 0x11];
        let p = [0x41u8, 0x22];
        let mut frame = sc4(&aud);
        frame.extend(sc4(&idr)); // AU 1: AUD + IDR
        frame.extend(sc4(&aud));
        frame.extend(sc4(&p)); // AU 2: AUD + P-slice
        let mut w = NalSampleWriter::new(NalMuxCodec::H264);
        let samples = w.push_packet(&frame);
        assert_eq!(samples.len(), 2, "two AUDs → two samples");
        assert!(samples[0].is_keyframe, "AU1 has the IDR");
        assert!(!samples[1].is_keyframe, "AU2 is a P-frame");
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
        let samples = w.push_packet(&frame);
        assert_eq!(samples.len(), 1);
        assert!(samples[0].is_keyframe, "type 19 is an IRAP/IDR");
        assert!(w.vps[0].starts_with(&vps));
        assert!(w.sps[0].starts_with(&sps));
        assert!(w.pps[0].starts_with(&pps));
        assert!(w.has_param_sets());
        let mut expect = (slice.len() as u32).to_be_bytes().to_vec();
        expect.extend_from_slice(&slice);
        assert_eq!(samples[0].data, expect);
    }

    #[test]
    fn preserves_slice_trailing_zero_bytes() {
        // A slice NAL whose RBSP legitimately ends in zero bytes (cabac_zero_words)
        // must NOT be truncated — that corrupts the slice and breaks decode.
        let slice = [0x65u8, 0x88, 0x00, 0x00, 0x00];
        let next = [0x41u8, 0x9a]; // a following P-slice
        let mut frame = sc4(&slice);
        frame.extend(sc4(&next));
        let nals = split_annexb_nals(&frame);
        assert_eq!(nals.len(), 2);
        // The slice's own bytes (incl. its trailing zeros) are never eaten; a
        // 4-byte next start code may leave one harmless extra trailing 0x00.
        assert!(nals[0].starts_with(&slice), "slice trailing zeros must survive: {:?}", nals[0]);
        assert!(nals[1].starts_with(&next));
        // 3-byte next start code: the slice is preserved exactly.
        let mut f2 = sc4(&slice);
        f2.extend_from_slice(&[0, 0, 1]);
        f2.extend_from_slice(&next);
        let n2 = split_annexb_nals(&f2);
        assert_eq!(n2[0], &slice, "trailing zeros kept exactly with a 3-byte next start code");
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
            w.push_packet(&f);
        }
        assert_eq!(w.sps.len(), 1);
        assert_eq!(w.pps.len(), 1);
    }
}
