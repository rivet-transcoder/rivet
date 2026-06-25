//! AVCC / HVCC → Annex-B conversion, shared by MP4 and MKV demuxers.
//!
//! MP4 and MKV both store H.264 / HEVC as length-prefixed NAL units with
//! the parameter sets sitting out-of-band in an AVCDecoderConfigurationRecord
//! (avcC) or HEVCDecoderConfigurationRecord (hvcC) box. Downstream decoders
//! (openh264, libde265, NVDEC) expect Annex-B streams: `00 00 00 01` start
//! codes between NAL units, with VPS/SPS/PPS prepended to the first sample.
//!
//! Key subtlety: the length-prefix size is NOT always 4 bytes. The
//! configuration record's `lengthSizeMinusOne` field (low 2 bits of a
//! config byte) specifies 0, 1, or 3 → 1, 2, or 4 bytes. Real files
//! using length_size=2 exist (MP4 streaming profiles), so we honor
//! the recorded value.

/// Parsed AVC configuration: SPS + PPS NAL units plus the sample
/// length-prefix size in bytes.
pub(crate) struct AvcConfig {
    /// 1, 2, or 4 bytes per NAL length prefix.
    pub length_size: u8,
    /// SPS NAL units followed by PPS NAL units, payload only (no start code,
    /// no length prefix). Ready to be emitted with an Annex-B start code.
    pub parameter_sets: Vec<Vec<u8>>,
}

/// Parsed HEVC configuration: VPS + SPS + PPS (+ optional SEI) NAL units
/// in the array order declared by the hvcC record, plus length-prefix size.
pub(crate) struct HevcConfig {
    pub length_size: u8,
    pub parameter_sets: Vec<Vec<u8>>,
}

/// Parse H.264 AVCDecoderConfigurationRecord (ISO/IEC 14496-15 §5.3.3.1).
/// Layout:
///   u8  configurationVersion = 1
///   u8  AVCProfileIndication
///   u8  profile_compatibility
///   u8  AVCLevelIndication
///   u8  reserved(6)|lengthSizeMinusOne(2)
///   u8  reserved(3)|numOfSequenceParameterSets(5)
///   // per SPS: u16 nalUnitLength, u8[nalUnitLength]
///   u8  numOfPictureParameterSets
///   // per PPS: u16 nalUnitLength, u8[nalUnitLength]
///
/// Returns `None` on truncation or an impossible record (so callers can
/// fall back to a 4-byte length-prefix default rather than panicking).
pub(crate) fn parse_avcc(avcc: &[u8]) -> Option<AvcConfig> {
    if avcc.len() < 7 {
        return None;
    }
    let length_size = (avcc[4] & 0x03) + 1;
    if !matches!(length_size, 1 | 2 | 4) {
        // length_size=3 is reserved; fall back to 4 for robustness.
        return None;
    }
    let num_sps = (avcc[5] & 0x1F) as usize;
    let mut out: Vec<Vec<u8>> = Vec::new();
    let mut cur = 6;
    for _ in 0..num_sps {
        if cur + 2 > avcc.len() {
            return None;
        }
        let nalu_len = u16::from_be_bytes([avcc[cur], avcc[cur + 1]]) as usize;
        cur += 2;
        if cur + nalu_len > avcc.len() {
            return None;
        }
        out.push(avcc[cur..cur + nalu_len].to_vec());
        cur += nalu_len;
    }
    if cur >= avcc.len() {
        return Some(AvcConfig {
            length_size,
            parameter_sets: out,
        });
    }
    let num_pps = avcc[cur] as usize;
    cur += 1;
    for _ in 0..num_pps {
        if cur + 2 > avcc.len() {
            return None;
        }
        let nalu_len = u16::from_be_bytes([avcc[cur], avcc[cur + 1]]) as usize;
        cur += 2;
        if cur + nalu_len > avcc.len() {
            return None;
        }
        out.push(avcc[cur..cur + nalu_len].to_vec());
        cur += nalu_len;
    }
    Some(AvcConfig {
        length_size,
        parameter_sets: out,
    })
}

/// Parse HEVC HEVCDecoderConfigurationRecord (ISO/IEC 14496-15 §8.3.3.1.2).
/// Layout (23 bytes fixed header + arrays):
///   u8  configurationVersion = 1
///   u8  general_profile_space(2)|tier(1)|profile_idc(5)
///   u32 general_profile_compatibility_flags
///   u48 general_constraint_indicator_flags
///   u8  general_level_idc
///   u16 reserved(4)|min_spatial_segmentation_idc(12)
///   u8  reserved(6)|parallelismType(2)
///   u8  reserved(6)|chromaFormat(2)
///   u8  reserved(5)|bitDepthLumaMinus8(3)
///   u8  reserved(5)|bitDepthChromaMinus8(3)
///   u16 avgFrameRate
///   u8  constantFrameRate(2)|numTemporalLayers(3)|temporalIdNested(1)|lengthSizeMinusOne(2)
///   u8  numOfArrays
///   // per array:
///   //   u8  array_completeness(1)|reserved(1)|NAL_unit_type(6)
///   //   u16 numNalus
///   //   // per nalu:  u16 nalUnitLength, u8[nalUnitLength]
///
/// Parameter sets are emitted in the order the arrays appear in the record
/// (typically VPS=32, SPS=33, PPS=34, optional prefix-SEI=39 / suffix-SEI=40).
pub(crate) fn parse_hvcc(hvcc: &[u8]) -> Option<HevcConfig> {
    if hvcc.len() < 23 {
        return None;
    }
    let length_size = (hvcc[21] & 0x03) + 1;
    if !matches!(length_size, 1 | 2 | 4) {
        return None;
    }
    let num_arrays = hvcc[22] as usize;
    let mut out: Vec<Vec<u8>> = Vec::new();
    let mut cur = 23;
    for _ in 0..num_arrays {
        if cur + 3 > hvcc.len() {
            return None;
        }
        let _array_hdr = hvcc[cur];
        let num_nalus = u16::from_be_bytes([hvcc[cur + 1], hvcc[cur + 2]]) as usize;
        cur += 3;
        for _ in 0..num_nalus {
            if cur + 2 > hvcc.len() {
                return None;
            }
            let nalu_len = u16::from_be_bytes([hvcc[cur], hvcc[cur + 1]]) as usize;
            cur += 2;
            if cur + nalu_len > hvcc.len() {
                return None;
            }
            out.push(hvcc[cur..cur + nalu_len].to_vec());
            cur += nalu_len;
        }
    }
    Some(HevcConfig {
        length_size,
        parameter_sets: out,
    })
}

/// Codec dispatch for inline NAL-type inspection. AVC has a 1-byte NAL
/// header with `nal_unit_type` in the low 5 bits; HEVC has a 2-byte
/// header with `nal_unit_type` in bits 1..7 of byte 0.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NaluCodec {
    Avc,
    Hevc,
}

/// Per-stream emission tracker for AVC/HEVC parameter sets.
///
/// Keeps the demuxer honest about prepending SPS/PPS (and HEVC VPS) when
/// the bitstream needs them. Two failure modes the older
/// `prepend on sample_idx==1` heuristic missed:
///
/// 1. **ExoPlayer open-GOP MP4** (#67/#68): sample 0 is SPS-only with
///    a non-IDR slice. The decoder cannot start mid-GOP without
///    parameter sets at the *next* IRAP — but that IRAP carries only
///    a slice NAL, so the stream stalls. We now prepend on the first
///    IRAP that is missing parameter sets.
/// 2. **avcC carries SPS but PPS arrives inline late**. We watch the
///    inline NAL types and only prepend the parts that haven't been
///    seen yet.
///
/// State is per-stream (one tracker per `samples` iteration). Reusing
/// across streams would conflate emission state and mis-skip prepend.
pub(crate) struct ParamSetTracker {
    codec: NaluCodec,
    /// Whether SPS has been emitted (either via `param_sets` prepend or
    /// inline in a previous sample we already converted).
    sps_emitted: bool,
    /// Whether PPS has been emitted.
    pps_emitted: bool,
    /// Whether HEVC VPS has been emitted. Always `true` for AVC since
    /// AVC has no VPS and we want the IRAP-prepend path to ignore it.
    vps_emitted: bool,
}

impl ParamSetTracker {
    pub(crate) fn new(codec: NaluCodec) -> Self {
        Self {
            codec,
            sps_emitted: false,
            pps_emitted: false,
            vps_emitted: matches!(codec, NaluCodec::Avc),
        }
    }

    /// Mark a parameter set type as already emitted (e.g. when the
    /// caller pre-emits the avcC/hvcC contents to prime the decoder
    /// before any samples). Avoids redundant prepending on the first
    /// IRAP seen in the bitstream.
    #[allow(dead_code)]
    pub(crate) fn note_param_sets_emitted(&mut self, nalus: &[Vec<u8>]) {
        for n in nalus {
            self.observe(n);
        }
    }

    fn observe(&mut self, nalu: &[u8]) {
        match self.classify(nalu) {
            Some(NalKind::Vps) => self.vps_emitted = true,
            Some(NalKind::Sps) => self.sps_emitted = true,
            Some(NalKind::Pps) => self.pps_emitted = true,
            _ => {}
        }
    }

    fn classify(&self, nalu: &[u8]) -> Option<NalKind> {
        if nalu.is_empty() {
            return None;
        }
        match self.codec {
            NaluCodec::Avc => {
                let t = nalu[0] & 0x1F;
                match t {
                    5 => Some(NalKind::Idr),
                    7 => Some(NalKind::Sps),
                    8 => Some(NalKind::Pps),
                    _ => Some(NalKind::Other),
                }
            }
            NaluCodec::Hevc => {
                if nalu.is_empty() {
                    return None;
                }
                let t = (nalu[0] >> 1) & 0x3F;
                // IRAP = BLA_W_LP..=RSV_IRAP_VCL23 (16..=23) per H.265 §7.4.2.2.
                // Cleanly covers IDR_W_RADL(19), IDR_N_LP(20), CRA(21), BLA(16-18).
                match t {
                    16..=23 => Some(NalKind::Idr),
                    32 => Some(NalKind::Vps),
                    33 => Some(NalKind::Sps),
                    34 => Some(NalKind::Pps),
                    _ => Some(NalKind::Other),
                }
            }
        }
    }

    fn fully_emitted(&self) -> bool {
        self.sps_emitted && self.pps_emitted && self.vps_emitted
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NalKind {
    Vps,
    Sps,
    Pps,
    Idr,
    Other,
}

/// Convert one length-prefixed AVCC/HVCC sample to Annex-B with a
/// stateful per-stream tracker. The tracker scans inline NAL types
/// and prepends parameter sets from `param_sets` (parsed from the
/// avcC / hvcC config record) on the **first IRAP we see that does
/// not already carry the parameter sets it needs**.
///
/// This is the correct fix for #67/#68 (MP4 ExoPlayer Main 720p):
/// sample 0 is `SPS + non-IDR-slice`. Old code prepended avcC SPS+PPS
/// on sample 0, which decoder-side looked like
/// `SPS PPS SPS slice` — the slice still references PPS, but the
/// decoder may discard the second SPS as a no-op and try to decode
/// the slice as if the GOP had started, which fails.
///
/// New behavior:
/// - Sample 0 has SPS inline + non-IDR slice → emit as-is, mark SPS
///   as observed. Decoder discards the slice silently (no IRAP yet).
/// - First IRAP sample arrives → if SPS or PPS hasn't been emitted
///   yet, prepend the missing one(s) from `param_sets` before the
///   IRAP NAL.
///
/// Inline parameter sets continue to count: a sample that contains
/// `SPS PPS IDR` will emit verbatim and mark both sets as fresh.
pub(crate) fn length_prefixed_to_annexb_tracked(
    sample: &[u8],
    length_size: u8,
    tracker: &mut ParamSetTracker,
    param_sets: &[Vec<u8>],
) -> Vec<u8> {
    const START_CODE: [u8; 4] = [0x00, 0x00, 0x00, 0x01];
    let ls = length_size as usize;
    debug_assert!(ls == 1 || ls == 2 || ls == 4, "invalid length_size {ls}");

    // Pass 1: scan the sample's NAL types so we know whether to prepend.
    // We cannot rewrite while we read because we may need to insert
    // parameter sets *before* the first IRAP NAL (not at the very front
    // of the sample — there may be SEI / AUD NALs ahead of it). So we
    // record (offset, length, kind) tuples and emit in pass 2.
    let mut nalus: Vec<(usize, usize, NalKind)> = Vec::new();
    {
        let mut pos = 0;
        while pos + ls <= sample.len() {
            let nal_size = read_be_uint(&sample[pos..pos + ls], ls);
            pos += ls;
            if nal_size == 0 || pos + nal_size > sample.len() {
                break;
            }
            let kind = tracker
                .classify(&sample[pos..pos + nal_size])
                .unwrap_or(NalKind::Other);
            nalus.push((pos, nal_size, kind));
            pos += nal_size;
        }
    }

    let has_irap = nalus.iter().any(|(_, _, k)| matches!(k, NalKind::Idr));
    let prepend_now = has_irap && !tracker.fully_emitted();

    let mut out = Vec::with_capacity(sample.len() + 4 * param_sets.len() + 16);

    if prepend_now {
        // Find the offset of the first IRAP NAL in the *output* stream
        // (i.e. emit non-IRAP NALs that came before it first, then the
        // missing parameter sets, then the IRAP and the rest). This
        // preserves AUD/SEI ordering before the IRAP.
        let irap_idx = nalus
            .iter()
            .position(|(_, _, k)| matches!(k, NalKind::Idr))
            .expect("has_irap implies position is Some");

        // Emit any leading non-IRAP NALs verbatim (typically AUD / SEI).
        for (off, len, kind) in &nalus[..irap_idx] {
            out.extend_from_slice(&START_CODE);
            out.extend_from_slice(&sample[*off..*off + *len]);
            tracker_observe(tracker, *kind);
        }

        // Emit only the parameter-set kinds that haven't been observed yet.
        // For HEVC the param_sets array is in hvcC array order (typically
        // VPS, SPS, PPS); we filter by kind so we never duplicate one that
        // already showed up inline above.
        for nalu in param_sets {
            let kind = tracker.classify(nalu).unwrap_or(NalKind::Other);
            let needed = match kind {
                NalKind::Vps => !tracker.vps_emitted,
                NalKind::Sps => !tracker.sps_emitted,
                NalKind::Pps => !tracker.pps_emitted,
                _ => false,
            };
            if needed {
                out.extend_from_slice(&START_CODE);
                out.extend_from_slice(nalu);
                tracker_observe(tracker, kind);
            }
        }

        // Emit the IRAP NAL and everything after it.
        for (off, len, kind) in &nalus[irap_idx..] {
            out.extend_from_slice(&START_CODE);
            out.extend_from_slice(&sample[*off..*off + *len]);
            tracker_observe(tracker, *kind);
        }
    } else {
        // No IRAP in this sample, or we've already emitted all parameter
        // sets — just convert verbatim and let the tracker absorb any
        // inline parameter sets that may be present.
        for (off, len, kind) in &nalus {
            out.extend_from_slice(&START_CODE);
            out.extend_from_slice(&sample[*off..*off + *len]);
            tracker_observe(tracker, *kind);
        }
    }

    out
}

fn tracker_observe(tracker: &mut ParamSetTracker, kind: NalKind) {
    match kind {
        NalKind::Vps => tracker.vps_emitted = true,
        NalKind::Sps => tracker.sps_emitted = true,
        NalKind::Pps => tracker.pps_emitted = true,
        _ => {}
    }
}

fn read_be_uint(buf: &[u8], n: usize) -> usize {
    let mut v: usize = 0;
    for &b in &buf[..n] {
        v = (v << 8) | b as usize;
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn avcc_roundtrip_length_size_four() {
        let sps = [0x67u8, 0x42, 0x00, 0x1e, 0xab];
        let pps = [0x68u8, 0xce, 0x3c, 0x80];
        let mut avcc = vec![
            0x01, 0x42, 0x00, 0x1e,
            0xff, // reserved(6)=1|lengthSizeMinusOne(2)=3 → 4-byte prefix
            0xe1, // reserved(3)=7|num_sps=1
        ];
        avcc.extend_from_slice(&(sps.len() as u16).to_be_bytes());
        avcc.extend_from_slice(&sps);
        avcc.push(0x01);
        avcc.extend_from_slice(&(pps.len() as u16).to_be_bytes());
        avcc.extend_from_slice(&pps);

        let cfg = parse_avcc(&avcc).expect("parse avcc");
        assert_eq!(cfg.length_size, 4);
        assert_eq!(cfg.parameter_sets.len(), 2);
        assert_eq!(&cfg.parameter_sets[0], &sps);
        assert_eq!(&cfg.parameter_sets[1], &pps);
    }

    #[test]
    fn avcc_honors_length_size_two() {
        let mut avcc = vec![
            0x01, 0x42, 0x00, 0x1e, 0xfd, // length_size_minus_one = 1 → 2-byte prefix
            0xe0, // num_sps = 0
        ];
        avcc.push(0x00); // num_pps = 0
        let cfg = parse_avcc(&avcc).expect("parse avcc ls=2");
        assert_eq!(cfg.length_size, 2);
        assert!(cfg.parameter_sets.is_empty());
    }

    /// Helper: build a length-prefixed sample (4-byte length) from a list of NALs.
    fn lp4_sample(nalus: &[&[u8]]) -> Vec<u8> {
        let mut s = Vec::new();
        for n in nalus {
            s.extend_from_slice(&(n.len() as u32).to_be_bytes());
            s.extend_from_slice(n);
        }
        s
    }

    fn avc_sps_nal() -> Vec<u8> {
        vec![0x67, 0x42, 0x00, 0x1e, 0xaa]
    }
    fn avc_pps_nal() -> Vec<u8> {
        vec![0x68, 0xce, 0x3c, 0x80]
    }
    fn avc_idr_nal() -> Vec<u8> {
        vec![0x65, 0x88, 0x84, 0x00]
    }
    fn avc_p_slice_nal() -> Vec<u8> {
        vec![0x41, 0x9a, 0x00, 0x00]
    }
    fn avc_sei_nal() -> Vec<u8> {
        vec![0x06, 0x00, 0x80]
    }

    /// MP4 / ExoPlayer regression (#67/#68): sample 0 carries inline SPS
    /// only with a non-IDR slice (open-GOP recovery), avcC has SPS+PPS.
    /// The first IRAP-bearing sample later must get PPS prepended (SPS
    /// already emitted inline upstream so we only patch the missing one).
    #[test]
    fn tracked_avc_exoplayer_sps_only_then_idr() {
        let sps = avc_sps_nal();
        let pps = avc_pps_nal();
        let avc_param_sets = vec![sps.clone(), pps.clone()];

        let mut tracker = ParamSetTracker::new(NaluCodec::Avc);

        // Sample 0: SPS + non-IDR P-slice (no PPS, no IDR). No prepend.
        let s0 = lp4_sample(&[&sps, &avc_p_slice_nal()]);
        let out0 = length_prefixed_to_annexb_tracked(&s0, 4, &mut tracker, &avc_param_sets);
        // Output should not contain PPS (no IRAP yet).
        assert!(
            !out0.windows(pps.len()).any(|w| w == pps.as_slice()),
            "PPS must not be prepended on a non-IRAP sample"
        );
        assert!(tracker.sps_emitted);
        assert!(!tracker.pps_emitted);

        // Sample 1: IDR alone (no inline params). PPS must be prepended;
        // SPS already emitted from sample 0 so we don't re-emit it.
        let idr = avc_idr_nal();
        let s1 = lp4_sample(&[&idr]);
        let out1 = length_prefixed_to_annexb_tracked(&s1, 4, &mut tracker, &avc_param_sets);
        // Should contain the PPS NAL exactly once.
        let pps_count = out1
            .windows(pps.len())
            .filter(|w| *w == pps.as_slice())
            .count();
        assert_eq!(
            pps_count, 1,
            "PPS must be prepended exactly once on first IDR"
        );
        // Should not duplicate the SPS already seen inline at sample 0.
        let sps_count = out1
            .windows(sps.len())
            .filter(|w| *w == sps.as_slice())
            .count();
        assert_eq!(
            sps_count, 0,
            "SPS must not be re-emitted (already inline at sample 0)"
        );
        assert!(tracker.pps_emitted);
    }

    /// AVC with no inline parameter sets at all (a Main-profile track
    /// where the muxer relies entirely on avcC). First IDR triggers
    /// prepending of both SPS and PPS in avcC declaration order.
    #[test]
    fn tracked_avc_avcc_only_first_idr_prepends_both() {
        let sps = avc_sps_nal();
        let pps = avc_pps_nal();
        let avc_param_sets = vec![sps.clone(), pps.clone()];

        let mut tracker = ParamSetTracker::new(NaluCodec::Avc);
        // Sample 0 IS the first IDR (most well-formed MP4s).
        let idr = avc_idr_nal();
        let s0 = lp4_sample(&[&idr]);
        let out0 = length_prefixed_to_annexb_tracked(&s0, 4, &mut tracker, &avc_param_sets);
        // Expect: SPS, PPS, IDR — each with 4-byte start code.
        let sps_idx = out0.windows(sps.len()).position(|w| w == sps.as_slice());
        let pps_idx = out0.windows(pps.len()).position(|w| w == pps.as_slice());
        let idr_idx = out0.windows(idr.len()).position(|w| w == idr.as_slice());
        assert!(sps_idx.is_some() && pps_idx.is_some() && idr_idx.is_some());
        // Order: SPS < PPS < IDR.
        assert!(sps_idx.unwrap() < pps_idx.unwrap());
        assert!(pps_idx.unwrap() < idr_idx.unwrap());
    }

    /// Inline SPS+PPS+IDR — typical Jellyfin / FFmpeg output.
    /// Must emit verbatim and **not duplicate** parameter sets.
    #[test]
    fn tracked_avc_inline_sps_pps_idr_no_duplication() {
        let sps = avc_sps_nal();
        let pps = avc_pps_nal();
        let idr = avc_idr_nal();
        let avc_param_sets = vec![sps.clone(), pps.clone()];

        let mut tracker = ParamSetTracker::new(NaluCodec::Avc);
        // Sample 0: SPS + PPS + IDR all inline.
        let s0 = lp4_sample(&[&sps, &pps, &idr]);
        let out0 = length_prefixed_to_annexb_tracked(&s0, 4, &mut tracker, &avc_param_sets);

        let sps_count = out0
            .windows(sps.len())
            .filter(|w| *w == sps.as_slice())
            .count();
        let pps_count = out0
            .windows(pps.len())
            .filter(|w| *w == pps.as_slice())
            .count();
        assert_eq!(sps_count, 1, "SPS must appear exactly once when inline");
        assert_eq!(pps_count, 1, "PPS must appear exactly once when inline");
    }

    /// IDR sample with leading SEI NAL — params must be inserted *between*
    /// the SEI and the IDR, not before the SEI. (The decoder is otherwise
    /// fine, but standards-compliant Annex-B order is SEI → params → IRAP
    /// when the SEI carries buffering-period info that depends on the SPS.)
    #[test]
    fn tracked_avc_sei_then_idr_inserts_params_between() {
        let sps = avc_sps_nal();
        let pps = avc_pps_nal();
        let sei = avc_sei_nal();
        let idr = avc_idr_nal();
        let avc_param_sets = vec![sps.clone(), pps.clone()];

        let mut tracker = ParamSetTracker::new(NaluCodec::Avc);
        let s0 = lp4_sample(&[&sei, &idr]);
        let out0 = length_prefixed_to_annexb_tracked(&s0, 4, &mut tracker, &avc_param_sets);

        let sei_pos = out0
            .windows(sei.len())
            .position(|w| w == sei.as_slice())
            .unwrap();
        let sps_pos = out0
            .windows(sps.len())
            .position(|w| w == sps.as_slice())
            .unwrap();
        let pps_pos = out0
            .windows(pps.len())
            .position(|w| w == pps.as_slice())
            .unwrap();
        let idr_pos = out0
            .windows(idr.len())
            .position(|w| w == idr.as_slice())
            .unwrap();
        assert!(sei_pos < sps_pos);
        assert!(sps_pos < pps_pos);
        assert!(pps_pos < idr_pos);
    }

    /// HEVC IRAP detection (IDR_W_RADL, type=19) with VPS+SPS+PPS in hvcC
    /// — first IRAP sample without inline parameter sets must get all
    /// three prepended.
    #[test]
    fn tracked_hevc_first_idr_prepends_vps_sps_pps() {
        // HEVC NAL header is 2 bytes; we set nal_unit_type via byte[0]
        // bits 1..7. type=32 → byte[0]=0b0100_0000=0x40, type=33→0x42, type=34→0x44.
        let vps: Vec<u8> = vec![0x40, 0x01, 0xAA];
        let sps: Vec<u8> = vec![0x42, 0x01, 0xBB];
        let pps: Vec<u8> = vec![0x44, 0x01, 0xCC];
        // type=19 (IDR_W_RADL) → byte[0] = 19<<1 = 0x26.
        let idr: Vec<u8> = vec![0x26, 0x01, 0xDD];
        let hevc_param_sets = vec![vps.clone(), sps.clone(), pps.clone()];

        let mut tracker = ParamSetTracker::new(NaluCodec::Hevc);
        let s0 = lp4_sample(&[&idr]);
        let out0 = length_prefixed_to_annexb_tracked(&s0, 4, &mut tracker, &hevc_param_sets);
        let vps_pos = out0
            .windows(vps.len())
            .position(|w| w == vps.as_slice())
            .unwrap();
        let sps_pos = out0
            .windows(sps.len())
            .position(|w| w == sps.as_slice())
            .unwrap();
        let pps_pos = out0
            .windows(pps.len())
            .position(|w| w == pps.as_slice())
            .unwrap();
        let idr_pos = out0
            .windows(idr.len())
            .position(|w| w == idr.as_slice())
            .unwrap();
        assert!(vps_pos < sps_pos);
        assert!(sps_pos < pps_pos);
        assert!(pps_pos < idr_pos);
    }

    /// Mixed: avcC has SPS+PPS, sample 0 has inline SPS only (no IDR),
    /// sample 1 has the actual IDR. Should emit avcC PPS but not SPS
    /// (already inline) on sample 1.
    #[test]
    fn tracked_avc_mixed_inline_sps_avcc_pps_only() {
        let sps = avc_sps_nal();
        let pps = avc_pps_nal();
        let avc_param_sets = vec![sps.clone(), pps.clone()];

        let mut tracker = ParamSetTracker::new(NaluCodec::Avc);

        // Sample 0: inline SPS + non-IDR. Mark SPS as observed.
        let s0 = lp4_sample(&[&sps, &avc_p_slice_nal()]);
        let _ = length_prefixed_to_annexb_tracked(&s0, 4, &mut tracker, &avc_param_sets);
        assert!(tracker.sps_emitted && !tracker.pps_emitted);

        // Sample 1: IDR alone. PPS gets prepended; SPS does not.
        let idr = avc_idr_nal();
        let s1 = lp4_sample(&[&idr]);
        let out1 = length_prefixed_to_annexb_tracked(&s1, 4, &mut tracker, &avc_param_sets);
        let sps_count = out1
            .windows(sps.len())
            .filter(|w| *w == sps.as_slice())
            .count();
        let pps_count = out1
            .windows(pps.len())
            .filter(|w| *w == pps.as_slice())
            .count();
        assert_eq!(sps_count, 0);
        assert_eq!(pps_count, 1);
    }

    /// Subsequent IDRs in the same stream must NOT re-prepend (decoder
    /// already has the parameter sets — repeating them is harmless but
    /// wastes bytes and could confuse strict parsers).
    #[test]
    fn tracked_avc_second_idr_no_reprepend() {
        let sps = avc_sps_nal();
        let pps = avc_pps_nal();
        let idr = avc_idr_nal();
        let avc_param_sets = vec![sps.clone(), pps.clone()];

        let mut tracker = ParamSetTracker::new(NaluCodec::Avc);

        // First IDR (sample 0) gets prepend.
        let s0 = lp4_sample(&[&idr]);
        let _ = length_prefixed_to_annexb_tracked(&s0, 4, &mut tracker, &avc_param_sets);
        // Second IDR (sample N) — params already emitted.
        let s1 = lp4_sample(&[&idr]);
        let out1 = length_prefixed_to_annexb_tracked(&s1, 4, &mut tracker, &avc_param_sets);
        let sps_count = out1
            .windows(sps.len())
            .filter(|w| *w == sps.as_slice())
            .count();
        let pps_count = out1
            .windows(pps.len())
            .filter(|w| *w == pps.as_slice())
            .count();
        assert_eq!(sps_count, 0);
        assert_eq!(pps_count, 0);
    }

    /// Honors length_size=2 (less common but spec-legal).
    #[test]
    fn tracked_avc_length_size_two() {
        let sps = avc_sps_nal();
        let pps = avc_pps_nal();
        let idr = avc_idr_nal();
        let avc_param_sets = vec![sps.clone(), pps.clone()];

        // Build a length=2 sample.
        let mut s = Vec::new();
        s.extend_from_slice(&(idr.len() as u16).to_be_bytes());
        s.extend_from_slice(&idr);

        let mut tracker = ParamSetTracker::new(NaluCodec::Avc);
        let out = length_prefixed_to_annexb_tracked(&s, 2, &mut tracker, &avc_param_sets);
        assert!(out.windows(sps.len()).any(|w| w == sps.as_slice()));
        assert!(out.windows(pps.len()).any(|w| w == pps.as_slice()));
        assert!(out.windows(idr.len()).any(|w| w == idr.as_slice()));
    }

    #[test]
    fn hvcc_extracts_vps_sps_pps_in_array_order() {
        let vps = [0x40u8, 0x01, 0x0c];
        let sps = [0x42u8, 0x01, 0x01];
        let pps = [0x44u8, 0x01];

        let mut hvcc = vec![0u8; 23];
        hvcc[0] = 1; // configurationVersion
        hvcc[21] = 0xf3; // lengthSizeMinusOne=3 → 4-byte prefix
        hvcc[22] = 3; // numOfArrays

        // Array 1: VPS (nal_unit_type=32) — top bit not set.
        hvcc.push(32);
        hvcc.extend_from_slice(&1u16.to_be_bytes()); // num_nalus
        hvcc.extend_from_slice(&(vps.len() as u16).to_be_bytes());
        hvcc.extend_from_slice(&vps);

        // Array 2: SPS (nal_unit_type=33)
        hvcc.push(33);
        hvcc.extend_from_slice(&1u16.to_be_bytes());
        hvcc.extend_from_slice(&(sps.len() as u16).to_be_bytes());
        hvcc.extend_from_slice(&sps);

        // Array 3: PPS (nal_unit_type=34)
        hvcc.push(34);
        hvcc.extend_from_slice(&1u16.to_be_bytes());
        hvcc.extend_from_slice(&(pps.len() as u16).to_be_bytes());
        hvcc.extend_from_slice(&pps);

        let cfg = parse_hvcc(&hvcc).expect("parse hvcc");
        assert_eq!(cfg.length_size, 4);
        assert_eq!(cfg.parameter_sets.len(), 3);
        assert_eq!(&cfg.parameter_sets[0], &vps);
        assert_eq!(&cfg.parameter_sets[1], &sps);
        assert_eq!(&cfg.parameter_sets[2], &pps);
    }
}
