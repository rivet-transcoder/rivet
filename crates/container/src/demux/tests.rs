use super::detect_container;
use super::mkv::{mkv_codec_needs_annexb, read_id_vint, read_size_vint};
use super::mp4::{has_av01_sample_entry, parse_avcc_param_sets, prores_sample_entry_fourcc};

#[test]
fn mkv_annexb_guard_flags_avc_and_hevc() {
    assert!(mkv_codec_needs_annexb("V_MPEG4/ISO/AVC"));
    assert!(mkv_codec_needs_annexb("V_MPEGH/ISO/HEVC"));
}

#[test]
fn mkv_annexb_guard_passes_self_contained_codecs() {
    assert!(!mkv_codec_needs_annexb("V_VP9"));
    assert!(!mkv_codec_needs_annexb("V_VP8"));
    assert!(!mkv_codec_needs_annexb("V_AV1"));
    assert!(!mkv_codec_needs_annexb("V_UNKNOWN"));
}

#[test]
fn parse_avcc_extracts_sps_and_pps() {
    // One SPS (6 bytes) + one PPS (4 bytes), no extension fields.
    let sps: [u8; 6] = [0x67, 0x42, 0x00, 0x1e, 0xab, 0x40];
    let pps: [u8; 4] = [0x68, 0xce, 0x3c, 0x80];
    let mut avcc = Vec::new();
    avcc.push(0x01); // configurationVersion
    avcc.push(0x42); // AVCProfileIndication = 66 (Baseline)
    avcc.push(0x00); // profile_compatibility
    avcc.push(0x1e); // AVCLevelIndication = 3.0
    avcc.push(0xff); // reserved(6)=1|lengthSizeMinusOne(2)=3
    avcc.push(0xe1); // reserved(3)=7|numOfSequenceParameterSets(5)=1
    avcc.extend_from_slice(&(sps.len() as u16).to_be_bytes());
    avcc.extend_from_slice(&sps);
    avcc.push(0x01); // numOfPictureParameterSets = 1
    avcc.extend_from_slice(&(pps.len() as u16).to_be_bytes());
    avcc.extend_from_slice(&pps);

    let sets = parse_avcc_param_sets(&avcc);
    assert_eq!(sets.len(), 2, "expected SPS + PPS");
    assert_eq!(&sets[0], &sps);
    assert_eq!(&sets[1], &pps);
}

#[test]
fn parse_avcc_truncated_returns_partial() {
    // Truncation mid-SPS should not panic; returns whatever was fully read.
    let avcc: [u8; 6] = [0x01, 0x42, 0x00, 0x1e, 0xff, 0xe1];
    let sets = parse_avcc_param_sets(&avcc);
    assert!(sets.is_empty());
}

#[test]
fn parse_avcc_empty_record_returns_empty() {
    assert!(parse_avcc_param_sets(&[]).is_empty());
    assert!(parse_avcc_param_sets(&[0x01]).is_empty());
}

/// Build a minimal box: `[size u32 BE][fourcc 4][payload]`.
fn mkbox(fourcc: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    let size = (8 + payload.len()) as u32;
    let mut out = Vec::with_capacity(size as usize);
    out.extend_from_slice(&size.to_be_bytes());
    out.extend_from_slice(fourcc);
    out.extend_from_slice(payload);
    out
}

#[test]
fn av01_detector_finds_sample_entry_in_nested_stsd() {
    // Minimal valid stsd body: version/flags (4 B) + entry_count=1 (4 B)
    // + one 16-byte av01 box header.
    let mut stsd_body = vec![0u8; 8];
    stsd_body.extend_from_slice(&mkbox(b"av01", &[0u8; 8]));
    let stsd = mkbox(b"stsd", &stsd_body);
    let stbl = mkbox(b"stbl", &stsd);
    let minf = mkbox(b"minf", &stbl);
    let mdia = mkbox(b"mdia", &minf);
    let trak = mkbox(b"trak", &mdia);
    let moov = mkbox(b"moov", &trak);
    assert!(has_av01_sample_entry(&moov));
}

#[test]
fn av01_detector_ignores_av01_in_wrong_place() {
    // av01 bytes floating in mdat must not trigger the detector.
    let mdat = mkbox(b"mdat", b"...av01... garbage");
    assert!(!has_av01_sample_entry(&mdat));
}

#[test]
fn read_size_vint_8_byte_encoding() {
    // size_vint_8 form used by the MKV test builder: `(1 << 56) | size`
    // encoded as 8 bytes big-endian. First byte is 0x01.
    let size: u64 = 1000;
    let v = (1u64 << 56) | size;
    let bytes = v.to_be_bytes();
    let (read, len) = read_size_vint(&bytes).expect("parse 8-byte size");
    assert_eq!(len, 8);
    assert_eq!(read, 1000);
}

#[test]
fn read_size_vint_1_byte_encoding() {
    // 1-byte VInt for value 1: 0x81.
    let (v, l) = read_size_vint(&[0x81]).expect("1-byte size");
    assert_eq!(l, 1);
    assert_eq!(v, 1);
}

#[test]
fn read_id_vint_parses_matroska_ids() {
    assert_eq!(read_id_vint(&[0xAE]), Some((0xAE, 1)));
    assert_eq!(
        read_id_vint(&[0x1A, 0x45, 0xDF, 0xA3, 0xFF]),
        Some((0x1A45DFA3, 4))
    );
    assert_eq!(read_id_vint(&[0x55, 0xB0, 0xFF]), Some((0x55B0, 2)));
}

#[test]
fn av01_detector_returns_false_for_avc1_sample_entry() {
    let mut stsd_body = vec![0u8; 8];
    stsd_body.extend_from_slice(&mkbox(b"avc1", &[0u8; 8]));
    let stsd = mkbox(b"stsd", &stsd_body);
    let stbl = mkbox(b"stbl", &stsd);
    let minf = mkbox(b"minf", &stbl);
    let mdia = mkbox(b"mdia", &minf);
    let trak = mkbox(b"trak", &mdia);
    let moov = mkbox(b"moov", &trak);
    assert!(!has_av01_sample_entry(&moov));
}

/// Helper: build a minimal MOV box tree carrying a single sample
/// entry with the supplied fourcc, nested moov/trak/mdia/minf/stbl/stsd.
/// The sample entry payload itself is zeros — the prores detector
/// only looks at the fourcc, not at any internal fields.
fn mov_with_sample_entry(fourcc: &[u8; 4]) -> Vec<u8> {
    let mut stsd_body = vec![0u8; 8]; // version/flags + entry_count
    stsd_body.extend_from_slice(&mkbox(fourcc, &[0u8; 8]));
    let stsd = mkbox(b"stsd", &stsd_body);
    let stbl = mkbox(b"stbl", &stsd);
    let minf = mkbox(b"minf", &stbl);
    let mdia = mkbox(b"mdia", &minf);
    let trak = mkbox(b"trak", &mdia);
    mkbox(b"moov", &trak)
}

#[test]
fn prores_detector_finds_all_six_fourccs() {
    for fcc in [b"apco", b"apcs", b"apcn", b"apch", b"ap4h", b"ap4x"] {
        let moov = mov_with_sample_entry(fcc);
        let detected = prores_sample_entry_fourcc(&moov)
            .unwrap_or_else(|| panic!("did not detect ProRes fourcc {fcc:?}"));
        assert_eq!(&detected, fcc, "fourcc round-trip for {fcc:?}");
    }
}

#[test]
fn prores_detector_ignores_non_prores_fourccs() {
    // A sample entry whose fourcc is something else (h264, hevc, etc.)
    // must NOT trigger the ProRes detector even when nested correctly.
    for fcc in [b"avc1", b"hvc1", b"av01", b"vp09", b"mp4v"] {
        let moov = mov_with_sample_entry(fcc);
        assert!(
            prores_sample_entry_fourcc(&moov).is_none(),
            "false positive on fourcc {fcc:?}"
        );
    }
}

#[test]
fn prores_detector_returns_none_when_no_stsd() {
    // Bare moov with no stsd path — must safely return None,
    // never panic.
    let moov = mkbox(b"moov", &[0u8; 4]);
    assert!(prores_sample_entry_fourcc(&moov).is_none());
}

#[test]
fn detect_container_recognises_mpeg_ts_sync_pattern() {
    // detect_container is package-private here; we exercise it via
    // a buffer whose first three sync points all land on 0x47.
    let mut buf = vec![0xFFu8; 12];
    buf[0] = 0x47;
    // Pad to length so detect_container can probe offsets 188 and 376.
    while buf.len() < 400 {
        buf.push(0x00);
    }
    buf[188] = 0x47;
    buf[376] = 0x47;
    assert_eq!(detect_container(&buf), "ts");
}

#[test]
fn detect_container_rejects_lone_0x47_byte() {
    // A single 0x47 sync byte must not be enough — random payloads
    // routinely contain it. Demand at least two confirming hits.
    let mut buf = vec![0u8; 400];
    buf[0] = 0x47;
    buf[188] = 0x00; // miss the second probe
    assert_ne!(detect_container(&buf), "ts");
}

#[test]
fn detect_container_recognises_avi_riff_signature() {
    let mut buf: Vec<u8> = b"RIFF".to_vec();
    buf.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]);
    buf.extend_from_slice(b"AVI ");
    buf.extend_from_slice(&[0u8; 32]);
    assert_eq!(detect_container(&buf), "avi");
}
