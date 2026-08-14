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
        NalMuxCodec::H264 => nal[0] & 0x1F,        // H.264 §7.3.1
        NalMuxCodec::H265 => (nal[0] >> 1) & 0x3F, // H.265 §7.3.1.2 (2-byte header)
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
        NalMuxCodec::H264 => nal_type(nal, codec) == 5, // IDR slice
        NalMuxCodec::H265 => matches!(nal_type(nal, codec), 16..=23), // BLA..IRAP
    }
}

/// Whether this NAL is a VCL (slice) NAL.
fn is_vcl(nal: &[u8], codec: NalMuxCodec) -> bool {
    let t = nal_type(nal, codec);
    match codec {
        NalMuxCodec::H264 => (1..=5).contains(&t),
        NalMuxCodec::H265 => t <= 31,
    }
}

/// Whether a VCL slice begins a new picture — the access-unit boundary signal
/// when the encoder emits no AUD. H.264: `first_mb_in_slice == 0` ⟺ the slice
/// header's leading `ue(v)` is the single bit `1` (top bit set). H.265:
/// `first_slice_segment_in_pic_flag` is the first bit after the 2-byte header.
fn first_slice_in_pic(nal: &[u8], codec: NalMuxCodec) -> bool {
    match codec {
        NalMuxCodec::H264 => nal.len() > 1 && (nal[1] & 0x80) != 0,
        NalMuxCodec::H265 => nal.len() > 2 && (nal[2] & 0x80) != 0,
    }
}

/// One muxed access unit (frame): its length-prefixed sample bytes + whether
/// it is a keyframe.
#[derive(Debug, Clone)]
pub struct AuSample {
    pub data: Vec<u8>,
    pub is_keyframe: bool,
}

/// Whether a demuxed Annex-B sample can be decoded without anything before it.
///
/// True when the sample carries an IDR (H.264) or IRAP (H.265) slice, which is
/// the definition of a point a decoder may be started at cold.
///
/// # Why this reads the bitstream instead of the container
///
/// Containers carry their own answer — mp4 `stss`, fragmented mp4 `trun` sample
/// flags, Matroska's SimpleBlock keyframe bit — and those are the obvious
/// source. They are also, in the wild, sometimes wrong: a remux that rebuilds
/// the sample table can mark every sample sync, and files produced by segmenters
/// routinely disagree with their own slice headers.
///
/// For splitting decode across GPUs the cost of that being wrong is not a
/// warning, it is a chunk whose first frame references a picture the decoder
/// never saw — silently corrupt output that no size or duration check notices.
/// The slice header cannot disagree with itself, so it is what this asks.
pub fn sample_is_keyframe(annexb: &[u8], codec: NalMuxCodec) -> bool {
    split_annexb_nals(annexb)
        .iter()
        .any(|nal| is_vcl(nal, codec) && is_idr(nal, codec))
}

/// The parameter-set NALs in an Annex-B buffer, re-emitted as Annex-B.
///
/// SPS/PPS for H.264, VPS/SPS/PPS for H.265. Empty when the buffer carries
/// none, which is the usual case for a non-IDR sample.
///
/// # Why a decoder started mid-stream needs these
///
/// An IDR is a point the *codec* can resume from, but only once the decoder
/// knows the frame geometry and reference setup — and that lives in the
/// parameter sets. mp4 keeps them in `avcC`/`hvcC` extradata rather than in the
/// stream, so a demuxer emits them in-band once, at the start. A decoder handed
/// the middle of such a stream sees an IDR it has no parameter sets for and
/// decodes nothing at all: not an error, just zero frames out.
///
/// So a decode range that does not begin at sample 0 has to be given the
/// parameter sets that were in force where it starts.
pub fn extract_parameter_sets(annexb: &[u8], codec: NalMuxCodec) -> Vec<u8> {
    let mut out = Vec::new();
    for nal in split_annexb_nals(annexb) {
        let is_param = match codec {
            NalMuxCodec::H264 => matches!(nal_type(nal, codec), 7 | 8),
            NalMuxCodec::H265 => matches!(nal_type(nal, codec), 32..=34),
        };
        if is_param {
            out.extend_from_slice(&[0, 0, 0, 1]);
            out.extend_from_slice(nal);
        }
    }
    out
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
/// collecting the parameter sets for the `avcC`/`hvcC` config box.
///
/// Two modes:
/// - **out-of-band** (default): SPS/PPS/VPS are stripped from samples and stored
///   in the config box. Correct for a single encoder (`avc1`/`hvc1`).
/// - **inline** ([`new_inline`]): SPS/PPS/VPS are ALSO kept inline in each
///   access unit (each IDR self-describes). Used by the multi-GPU stitch, where
///   chunks come from independent encoders (possibly different vendors): the
///   inline parameter sets let each chunk decode with its own SPS/PPS even if
///   they differ cosmetically. Pairs with the `avc3`/`hev1` sample entry. The
///   config box still gets the FIRST set as a default hint.
#[derive(Debug)]
pub struct NalSampleWriter {
    codec: NalMuxCodec,
    /// HEVC VPS NAL units (empty for H.264), first-seen order, de-duplicated.
    pub vps: Vec<Vec<u8>>,
    pub sps: Vec<Vec<u8>>,
    pub pps: Vec<Vec<u8>>,
    inline_param_sets: bool,
}

impl NalSampleWriter {
    pub fn new(codec: NalMuxCodec) -> Self {
        Self {
            codec,
            vps: Vec::new(),
            sps: Vec::new(),
            pps: Vec::new(),
            inline_param_sets: false,
        }
    }

    /// Inline-parameter-set mode (for the multi-GPU stitch). Keeps SPS/PPS/VPS
    /// inline in each access unit AND records the first set for the config box.
    pub fn new_inline(codec: NalMuxCodec) -> Self {
        Self {
            codec,
            vps: Vec::new(),
            sps: Vec::new(),
            pps: Vec::new(),
            inline_param_sets: true,
        }
    }

    /// Convert one encoder packet — which may carry **multiple access units**
    /// (HW encoders return several frames per buffer) — into one
    /// **length-prefixed** mdat sample *per access unit*. Access units are
    /// delimited by the AUD NAL (a packet with no AUD is treated as one unit).
    /// SPS/PPS/VPS are captured (for the config box) and stripped from samples.
    pub fn push_packet(&mut self, annexb: &[u8]) -> Vec<AuSample> {
        // Group NALs into access units. A new unit begins at an AUD, or — when
        // the encoder emits no AUD (QSV H.265) — at the first VCL slice of a new
        // picture once the current unit already holds a slice.
        let mut units: Vec<Vec<&[u8]>> = vec![Vec::new()];
        let mut cur_has_vcl = false;
        for nal in split_annexb_nals(annexb) {
            let new_au = is_aud(nal, self.codec)
                || (is_vcl(nal, self.codec) && cur_has_vcl && first_slice_in_pic(nal, self.codec));
            if new_au && !units.last().unwrap().is_empty() {
                units.push(Vec::new());
                cur_has_vcl = false;
            }
            if is_vcl(nal, self.codec) {
                cur_has_vcl = true;
            }
            units.last_mut().unwrap().push(nal);
        }

        let codec = self.codec;
        let inline = self.inline_param_sets;
        let mut samples = Vec::new();
        for unit in units {
            let mut data = Vec::new();
            let mut is_keyframe = false;
            for nal in unit {
                let push_inline = |data: &mut Vec<u8>| {
                    data.extend_from_slice(&(nal.len() as u32).to_be_bytes());
                    data.extend_from_slice(nal);
                };
                match classify(nal, codec) {
                    NalClass::Sample => {
                        if is_idr(nal, codec) {
                            is_keyframe = true;
                        }
                        push_inline(&mut data);
                        continue;
                    }
                    NalClass::Vps | NalClass::Sps | NalClass::Pps => {}
                }
                // A parameter set (SPS/PPS/VPS):
                let store = match classify(nal, codec) {
                    NalClass::Vps => &mut self.vps,
                    NalClass::Sps => &mut self.sps,
                    NalClass::Pps => &mut self.pps,
                    NalClass::Sample => unreachable!(),
                };
                if inline {
                    // Record the first of each kind for the config-box default,
                    // and keep every parameter set inline in the access unit.
                    if store.is_empty() {
                        store.push(nal.to_vec());
                    }
                    push_inline(&mut data);
                } else {
                    dedup_push(store, nal);
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
    fn inline_mode_keeps_param_sets_in_sample() {
        // Multi-GPU stitch: each access unit must self-describe with its own
        // SPS/PPS (avc3/hev1), so a chunk decodes with its own parameter sets.
        let sps = [0x67u8, 0x42, 0x00, 0x1e, 0xAA];
        let pps = [0x68u8, 0xCE, 0x3C];
        let idr = [0x65u8, 0x88, 0x11, 0x22];
        let mut frame = sc4(&sps);
        frame.extend(sc4(&pps));
        frame.extend(sc4(&idr));

        let mut w = NalSampleWriter::new_inline(NalMuxCodec::H264);
        let inline = w.push_packet(&frame);
        assert_eq!(inline.len(), 1);
        assert!(inline[0].is_keyframe);
        // Config box still records the first SPS/PPS as a default hint.
        assert_eq!(w.sps.len(), 1);
        assert!(w.sps[0].starts_with(&sps));
        assert_eq!(w.pps.len(), 1);

        // Out-of-band mode strips the params, so its sample is smaller.
        let mut w2 = NalSampleWriter::new(NalMuxCodec::H264);
        let oob = w2.push_packet(&frame);
        assert!(
            inline[0].data.len() > oob[0].data.len(),
            "inline sample (SPS+PPS+IDR) must be larger than the stripped one ({} vs {})",
            inline[0].data.len(),
            oob[0].data.len()
        );
        // The inline sample begins with the length-prefixed SPS bytes.
        assert_eq!(&inline[0].data[4..4 + sps.len()], &sps);
    }

    #[test]
    fn h265_splits_multi_picture_packet_without_aud() {
        // QSV H.265 emits no AUD: split on VCL slices with first_slice flag set.
        let idr = [0x26u8, 0x01, 0xA0]; // type 19 (IDR), first_slice_segment=1
        let trail = [0x02u8, 0x01, 0xA0]; // type 1 (TRAIL_R), first_slice_segment=1
        let mut frame = sc4(&idr);
        frame.extend(sc4(&trail));
        let mut w = NalSampleWriter::new(NalMuxCodec::H265);
        let samples = w.push_packet(&frame);
        assert_eq!(
            samples.len(),
            2,
            "two first-slice VCL NALs → two access units"
        );
        assert!(samples[0].is_keyframe);
        assert!(!samples[1].is_keyframe);
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
        assert!(
            nals[0].starts_with(&slice),
            "slice trailing zeros must survive: {:?}",
            nals[0]
        );
        assert!(nals[1].starts_with(&next));
        // 3-byte next start code: the slice is preserved exactly.
        let mut f2 = sc4(&slice);
        f2.extend_from_slice(&[0, 0, 1]);
        f2.extend_from_slice(&next);
        let n2 = split_annexb_nals(&f2);
        assert_eq!(
            n2[0], &slice,
            "trailing zeros kept exactly with a 3-byte next start code"
        );
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
