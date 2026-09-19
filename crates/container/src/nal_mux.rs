//! Mux-side H.264 / H.265 NAL handling: take the encoder's **Annex-B** output
//! (start-code-delimited NAL units), strip the out-of-band parameter sets
//! (SPS/PPS, plus HEVC VPS) for the `avcC`/`hvcC` config box, and repackage the
//! remaining NALs (slices, SEI) as **length-prefixed** (4-byte) samples for the
//! MP4 `mdat`. This is the inverse of the demux path in
//! the crate-internal `annexb` module, which reads length-prefixed → Annex-B.
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
/// - **inline** ([`NalSampleWriter::new_inline`]): SPS/PPS/VPS are ALSO kept inline in each
///   access unit (each IDR self-describes). Used by the multi-GPU stitch, where
///   chunks come from independent encoders (possibly different vendors): the
///   inline parameter sets let each chunk decode with its own SPS/PPS even if
///   they differ cosmetically. Pairs with the `avc3`/`hev1` sample entry. The
///   config box still gets the FIRST set of each id as a default hint.
///
/// A stream may carry several sets of a kind under different ids — an H.264
/// encoder that codes its B pictures with a second PPS (`pps_id` 1, weighted
/// bi-prediction) re-sends it in the access units that use it. Both modes
/// keep one set per id, in id order (`avcC` numOfPictureParameterSets, the
/// `hvcC` arrays); a byte-identical repeat is only a repeat.
#[derive(Debug)]
pub struct NalSampleWriter {
    codec: NalMuxCodec,
    /// HEVC VPS NAL units (empty for H.264), one per id in id order.
    pub vps: Vec<Vec<u8>>,
    pub sps: Vec<Vec<u8>>,
    pub pps: Vec<Vec<u8>>,
    inline_param_sets: bool,
    /// The (kind, id) pairs whose set arrived again with different contents,
    /// out of band — each warned about once.
    conflicts: Vec<(NalClass, u32)>,
}

impl NalSampleWriter {
    pub fn new(codec: NalMuxCodec) -> Self {
        Self {
            codec,
            vps: Vec::new(),
            sps: Vec::new(),
            pps: Vec::new(),
            inline_param_sets: false,
            conflicts: Vec::new(),
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
            conflicts: Vec::new(),
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
                let kind = classify(nal, codec);
                let store = match kind {
                    NalClass::Vps => &mut self.vps,
                    NalClass::Sps => &mut self.sps,
                    NalClass::Pps => &mut self.pps,
                    NalClass::Sample => unreachable!(),
                };
                if inline {
                    // Keep every parameter set inline in the access unit, where
                    // a re-sent set replaces the one before it, and record the
                    // first of each id for the config-box default.
                    keep_by_id(store, nal, codec, kind);
                    push_inline(&mut data);
                } else if let Kept::Conflict(id) = keep_by_id(store, nal, codec, kind)
                    && !self.conflicts.contains(&(kind, id))
                {
                    // Out-of-band parameter sets are the whole stream's: a set
                    // that CHANGED under its id — the encoder re-sent its PPS
                    // with a different `pic_init_qp`, say — is one Annex-B
                    // tolerates (the re-sent set replaces the old one) and
                    // this box cannot. A decoder reading `avcC` / `hvcC` knows
                    // one set per id, so the pictures written under the other
                    // decode to garbage from their first macroblock. Found by
                    // exactly that (the native H.264 encoder's I-vs-P PPS,
                    // 2026-08-27); the fix is in the encoder, and this says so.
                    self.conflicts.push((kind, id));
                    tracing::warn!(
                        codec = ?codec,
                        kind = ?kind,
                        id,
                        "a parameter set arrived again under its id with different contents; \
                         this sample entry carries one set per id out of band, so the pictures \
                         coded after the change decode wrong. The first is kept — the encoder \
                         should give a changed set a new id (or the writer should use inline \
                         mode, avc3/hev1)"
                    );
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

/// What [`keep_by_id`] did with a parameter set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kept {
    /// A new id (or a set whose id does not parse): held.
    New,
    /// The same set again: nothing to hold.
    Repeat,
    /// Different contents under an id already held, which stays.
    Conflict(u32),
}

/// Hold parameter set `nal` in `store`, which keeps one set per id in id
/// order. A set whose id does not parse is held once by its bytes, after the
/// ordered ones — the order the sets arrived in, as before ids were read.
fn keep_by_id(store: &mut Vec<Vec<u8>>, nal: &[u8], codec: NalMuxCodec, kind: NalClass) -> Kept {
    if store.iter().any(|held| same_nal(held, nal)) {
        return Kept::Repeat;
    }
    let Some(id) = param_set_id(nal, codec, kind) else {
        store.push(nal.to_vec());
        return Kept::New;
    };
    let ids: Vec<Option<u32>> = store.iter().map(|held| param_set_id(held, codec, kind)).collect();
    if ids.contains(&Some(id)) {
        return Kept::Conflict(id);
    }
    let at = ids.iter().position(|held| held.is_none_or(|h| h > id)).unwrap_or(store.len());
    store.insert(at, nal.to_vec());
    Kept::New
}

/// Whether two NAL units are the same once trailing zero bytes are set aside:
/// a 4-byte start code after a NAL leaves its first `00` on the NAL
/// ([`find_start_code`]), and a NAL's own RBSP ends in its stop bit.
fn same_nal(a: &[u8], b: &[u8]) -> bool {
    let trim = |n: &[u8]| n.len() - n.iter().rev().take_while(|&&x| x == 0).count();
    a[..trim(a)] == b[..trim(b)]
}

/// The id a parameter set NAL unit (header included) declares: H.264
/// `seq_parameter_set_id` / `pic_parameter_set_id`, H.265
/// `vps_video_parameter_set_id` / `sps_seq_parameter_set_id` /
/// `pps_pic_parameter_set_id`. `None` when it does not parse.
fn param_set_id(nal: &[u8], codec: NalMuxCodec, kind: NalClass) -> Option<u32> {
    let header = match codec {
        NalMuxCodec::H264 => 1,
        NalMuxCodec::H265 => 2,
    };
    let rbsp = unescape(nal.get(header..)?);
    let mut r = Bits { data: &rbsp, pos: 0 };
    match (codec, kind) {
        // profile_idc, the constraint flags and level_idc come first.
        (NalMuxCodec::H264, NalClass::Sps) => {
            r.skip(24)?;
            r.ue()
        }
        (_, NalClass::Pps) => r.ue(),
        (NalMuxCodec::H265, NalClass::Vps) => r.bits(4),
        (NalMuxCodec::H265, NalClass::Sps) => {
            r.skip(4)?; // sps_video_parameter_set_id
            let max_sub_layers_minus1 = r.bits(3)? as usize;
            r.skip(1)?; // sps_temporal_id_nesting_flag
            skip_profile_tier_level(&mut r, max_sub_layers_minus1)?;
            r.ue()
        }
        _ => None,
    }
}

/// Skip H.265 `profile_tier_level(1, max_sub_layers_minus1)` (§7.3.3).
fn skip_profile_tier_level(r: &mut Bits, max_sub_layers_minus1: usize) -> Option<()> {
    r.skip(88 + 8)?; // the general profile, then general_level_idc
    let mut present = [(false, false); 8];
    for p in present.iter_mut().take(max_sub_layers_minus1) {
        *p = (r.bits(1)? == 1, r.bits(1)? == 1);
    }
    if max_sub_layers_minus1 > 0 {
        r.skip(2 * (8 - max_sub_layers_minus1))?; // reserved_zero_2bits
    }
    for &(profile, level) in present.iter().take(max_sub_layers_minus1) {
        r.skip(if profile { 88 } else { 0 } + if level { 8 } else { 0 })?;
    }
    Some(())
}

/// A NAL unit's payload with its emulation-prevention bytes (`00 00 03`)
/// removed.
fn unescape(ebsp: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(ebsp.len());
    let mut zeros = 0;
    for &b in ebsp {
        if zeros >= 2 && b == 3 {
            zeros = 0;
            continue;
        }
        zeros = if b == 0 { zeros + 1 } else { 0 };
        out.push(b);
    }
    out
}

/// MSB-first bit reader over an RBSP, with Exp-Golomb.
struct Bits<'a> {
    data: &'a [u8],
    pos: usize,
}

impl Bits<'_> {
    fn bits(&mut self, n: usize) -> Option<u32> {
        let mut v = 0u32;
        for _ in 0..n {
            let byte = *self.data.get(self.pos / 8)?;
            v = (v << 1) | u32::from((byte >> (7 - self.pos % 8)) & 1);
            self.pos += 1;
        }
        Some(v)
    }

    fn skip(&mut self, n: usize) -> Option<()> {
        self.pos += n;
        (self.pos <= self.data.len() * 8).then_some(())
    }

    fn ue(&mut self) -> Option<u32> {
        let mut zeros = 0;
        while self.bits(1)? == 0 {
            zeros += 1;
            if zeros > 31 {
                return None;
            }
        }
        Some((1u32 << zeros) - 1 + self.bits(zeros)?)
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

    /// A NAL unit: `header`, then `fields` MSB-first — `(value, bits)`, with
    /// `bits == 0` meaning ue(v) — then the stop bit, emulation-prevented.
    fn nal(header: &[u8], fields: &[(u32, usize)]) -> Vec<u8> {
        let mut bits: Vec<u8> = Vec::new();
        for &(v, n) in fields {
            if n == 0 {
                let len = 32 - (v + 1).leading_zeros() as usize;
                bits.extend(std::iter::repeat_n(0, len - 1));
                bits.extend((0..len).rev().map(|i| ((v + 1) >> i & 1) as u8));
            } else {
                bits.extend((0..n).rev().map(|i| (v >> i & 1) as u8));
            }
        }
        bits.push(1);
        while !bits.len().is_multiple_of(8) {
            bits.push(0);
        }
        let mut out = header.to_vec();
        let mut zeros = 0;
        for byte in bits.chunks(8).map(|c| c.iter().fold(0u8, |b, &x| b << 1 | x)) {
            if zeros >= 2 && byte <= 3 {
                out.push(3);
                zeros = 0;
            }
            zeros = if byte == 0 { zeros + 1 } else { 0 };
            out.push(byte);
        }
        out
    }

    /// An H.264 PPS: `pic_parameter_set_id`, SPS 0, then `pic_init_qp_minus26`
    /// (as its ue(v) code number) standing for the rest of the set.
    fn h264_pps(id: u32, qp_code: u32) -> Vec<u8> {
        // entropy, bottom_field_pic_order, slice groups, l0/l1 defaults,
        // weighted_pred, weighted_bipred_idc, then pic_init_qp_minus26.
        let fields = [(id, 0), (0, 0), (0, 1), (0, 1), (0, 0), (0, 0), (0, 0), (0, 1), (0, 2), (qp_code, 0)];
        nal(&[0x68], &fields)
    }

    /// Annex-B access unit of the given NAL units, 3-byte start codes (so a
    /// NAL carries no trailing zero from the next one).
    fn au(nals: &[&[u8]]) -> Vec<u8> {
        let mut v = Vec::new();
        for n in nals {
            v.extend_from_slice(&[0, 0, 1]);
            v.extend_from_slice(n);
        }
        v
    }

    fn length_prefixed(nals: &[&[u8]]) -> Vec<u8> {
        let mut v = Vec::new();
        for n in nals {
            v.extend_from_slice(&(n.len() as u32).to_be_bytes());
            v.extend_from_slice(n);
        }
        v
    }

    const H264_SPS: [u8; 5] = [0x67, 0x64, 0x00, 0x1f, 0xAC];
    const H264_IDR: [u8; 3] = [0x65, 0x88, 0x84];
    const H264_P: [u8; 3] = [0x41, 0x9a, 0x02];

    #[test]
    fn a_second_pps_id_is_kept_silently_in_id_order() {
        // The first access unit brings PPS 1 before PPS 0; the B pictures'
        // access units re-send PPS 1 in-band, as the H.264 encoder does for
        // its weighted-bi-prediction set.
        let (pps0, pps1) = (h264_pps(0, 0), h264_pps(1, 2));
        let mut w = NalSampleWriter::new(NalMuxCodec::H264);
        let mut samples = w.push_packet(&au(&[&H264_SPS, &pps1, &pps0, &H264_IDR]));
        samples.extend(w.push_packet(&au(&[&pps1, &H264_P])));
        samples.extend(w.push_packet(&au(&[&H264_P])));
        assert_eq!(w.pps, vec![pps0, pps1], "one PPS per id, in id order");
        assert!(w.conflicts.is_empty(), "two ids are no conflict: {:?}", w.conflicts);
        assert_eq!(samples.len(), 3);
        assert_eq!(samples[0].data, length_prefixed(&[&H264_IDR]));
        assert_eq!(samples[1].data, length_prefixed(&[&H264_P]), "avc1: the re-sent PPS lives in avcC");
        // And the record lists both: numOfPictureParameterSets = 2, id 0 first.
        let avcc = crate::mux::build_avcc(&w.sps, &w.pps);
        let at = 8 + 6 + 2 + H264_SPS.len();
        assert_eq!(avcc[at], 2, "numOfPictureParameterSets");
        let first_len = u16::from_be_bytes([avcc[at + 1], avcc[at + 2]]) as usize;
        assert_eq!(&avcc[at + 3..at + 3 + first_len], w.pps[0].as_slice());
    }

    #[test]
    fn inline_mode_keeps_the_repeats_in_band_and_every_id_in_the_box() {
        let (pps0, pps1) = (h264_pps(0, 0), h264_pps(1, 2));
        let mut w = NalSampleWriter::new_inline(NalMuxCodec::H264);
        w.push_packet(&au(&[&H264_SPS, &pps0, &H264_IDR]));
        let b = w.push_packet(&au(&[&pps1, &H264_P]));
        let again = w.push_packet(&au(&[&pps1, &H264_P]));
        assert_eq!(b[0].data, length_prefixed(&[&pps1, &H264_P]), "avc3: the set stays in its access unit");
        assert_eq!(again[0].data, b[0].data, "every repeat too");
        assert_eq!(w.pps, vec![pps0, pps1], "the box's default hint has both ids");
    }

    #[test]
    fn a_changed_set_under_its_id_is_named_once_and_the_first_kept() {
        let (first, changed) = (h264_pps(0, 0), h264_pps(0, 4));
        let mut w = NalSampleWriter::new(NalMuxCodec::H264);
        w.push_packet(&au(&[&H264_SPS, &first, &H264_IDR]));
        w.push_packet(&au(&[&changed, &H264_P]));
        w.push_packet(&au(&[&changed, &H264_P]));
        assert_eq!(w.pps, vec![first], "one set per id: the one the first pictures use");
        assert_eq!(w.conflicts, vec![(NalClass::Pps, 0)], "named once, by kind and id");

        // Inline, a re-sent set replaces the one before it in-band: no conflict.
        let mut w = NalSampleWriter::new_inline(NalMuxCodec::H264);
        w.push_packet(&au(&[&H264_SPS, &h264_pps(0, 0), &H264_IDR]));
        w.push_packet(&au(&[&h264_pps(0, 4), &H264_P]));
        assert!(w.conflicts.is_empty());
        assert_eq!(w.pps, vec![h264_pps(0, 0)]);
    }

    #[test]
    fn a_trailing_zero_from_a_4_byte_start_code_is_not_a_new_set() {
        let pps = h264_pps(0, 0);
        let mut w = NalSampleWriter::new(NalMuxCodec::H264);
        // 4-byte start codes: the PPS is split off with the next code's `00`.
        let mut first = sc4(&H264_SPS);
        first.extend(sc4(&pps));
        first.extend(sc4(&H264_IDR));
        w.push_packet(&first);
        w.push_packet(&au(&[&pps, &H264_P]));
        assert_eq!(w.pps.len(), 1, "{:02x?}", w.pps);
        assert!(w.conflicts.is_empty(), "a repeat, not a changed set");
    }

    /// An H.265 SPS declaring `id`, with `sub_layers` temporal sub-layers whose
    /// profile and level are both signalled (so the id sits past them).
    fn h265_sps(id: u32, sub_layers: u32) -> Vec<u8> {
        let general = [(0, 2), (0, 1), (1, 5), (0x6000_0000, 32), (0b1001, 4), (0, 32), (0, 11), (0, 1)];
        let mut fields = vec![(0, 4), (sub_layers, 3), (1, 1)];
        fields.extend(general);
        fields.push((93, 8)); // general_level_idc
        for _ in 0..sub_layers {
            fields.extend([(1, 1), (1, 1)]);
        }
        if sub_layers > 0 {
            fields.extend((sub_layers..8).map(|_| (0, 2)));
        }
        for _ in 0..sub_layers {
            fields.extend(general);
            fields.push((90, 8));
        }
        fields.extend([(id, 0), (1, 0), (160, 0), (96, 0)]); // id, chroma, width, height
        nal(&[0x42, 0x01], &fields)
    }

    #[test]
    fn h265_sets_are_kept_one_per_id_in_id_order() {
        let vps = |id: u32| nal(&[0x40, 0x01], &[(id, 4), (3, 2), (0, 6), (0, 3), (1, 1), (0xFFFF, 16)]);
        let pps = |id: u32| nal(&[0x44, 0x01], &[(id, 0), (0, 0), (0, 1), (0, 1), (0, 3)]);
        let idr = [0x26u8, 0x01, 0xAF];
        for sub_layers in [0, 2] {
            let (sps0, sps1) = (h265_sps(0, sub_layers), h265_sps(1, sub_layers));
            assert_eq!(param_set_id(&sps1, NalMuxCodec::H265, NalClass::Sps), Some(1));
            let mut w = NalSampleWriter::new(NalMuxCodec::H265);
            let s = w.push_packet(&au(&[&vps(1), &vps(0), &sps1, &sps0, &pps(1), &pps(0), &idr]));
            w.push_packet(&au(&[&pps(1), &sps1, &idr]));
            assert_eq!(w.vps, vec![vps(0), vps(1)]);
            assert_eq!(w.sps, vec![sps0, sps1], "{sub_layers} sub-layers");
            assert_eq!(w.pps, vec![pps(0), pps(1)]);
            assert!(w.conflicts.is_empty());
            assert_eq!(s[0].data, length_prefixed(&[&idr]), "hvc1: every set out of band");
        }
    }

    #[test]
    fn emulation_prevention_is_removed_before_the_id_is_read() {
        assert_eq!(unescape(&[0, 0, 3, 3, 0, 0, 3]), vec![0, 0, 3, 0, 0]);
        // The profile_tier_level's zero bits are escaped on the wire; the id
        // past them reads only from the unescaped payload.
        let sps = h265_sps(5, 0);
        assert!(sps.windows(3).any(|w| w == [0, 0, 3]), "{sps:02x?}");
        assert_eq!(param_set_id(&sps, NalMuxCodec::H265, NalClass::Sps), Some(5));
    }
}
