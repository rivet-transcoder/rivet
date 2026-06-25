//! Integration tests for MKV demux: colour metadata, bitrate sources,
//! and AVC/HEVC Annex-B conversion. We build minimal but valid MKV
//! byte streams in-test (EBML encoder below) rather than shipping
//! fixture files — keeps the tests hermetic and easy to extend.
//!
//! EBML encoding recap (see <https://www.matroska.org/technical/elements.html>):
//!   * Each element is `[VInt ID][VInt size][payload]`.
//!   * IDs are written verbatim from the Matroska spec (they already have
//!     the VInt leading-1 marker baked in).
//!   * Size VInts are the same format: an 8-byte upper bound works for
//!     all our payloads (leading 0x01 marker + 7 big-endian bytes).
//!   * Unsigned integers: shortest big-endian encoding (1..=8 bytes).
//!   * Floats: 8 bytes IEEE-754 big-endian.
//!   * Strings: raw UTF-8 bytes, no terminator.

use codec::frame::{ColorSpace, TransferFn};
use container::demux::{self, demux_mkv, probe_mkv_color_info};

/// Big-endian encode an unsigned using the minimum bytes (1..=8), or
/// 1 byte of `0x00` when the value is zero (matches Matroska practice).
fn be_min_bytes(v: u64) -> Vec<u8> {
    if v == 0 {
        return vec![0];
    }
    let mut bytes: Vec<u8> = Vec::new();
    let mut x = v;
    while x != 0 {
        bytes.push((x & 0xff) as u8);
        x >>= 8;
    }
    bytes.reverse();
    bytes
}

/// Encode a size VInt as an 8-byte value. The first byte has the marker
/// 0x01 in its top bit; the remaining 7 bytes carry the size big-endian.
/// This is sufficient for any payload we build here (< 2^56 bytes).
fn size_vint_8(size: u64) -> [u8; 8] {
    // Marker 0x01 at bit 56 (top bit of byte 0) leaves 56 bits of length.
    let v = (1u64 << 56) | (size & ((1u64 << 56) - 1));
    v.to_be_bytes()
}

/// Write an EBML element: `[id bytes][size vint (8-byte form)][payload]`.
fn el(id: &[u8], payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(id.len() + 8 + payload.len());
    out.extend_from_slice(id);
    out.extend_from_slice(&size_vint_8(payload.len() as u64));
    out.extend_from_slice(payload);
    out
}

fn el_uint(id: &[u8], v: u64) -> Vec<u8> {
    el(id, &be_min_bytes(v))
}

fn el_float(id: &[u8], v: f64) -> Vec<u8> {
    el(id, &v.to_be_bytes())
}

fn el_str(id: &[u8], s: &str) -> Vec<u8> {
    el(id, s.as_bytes())
}

// EBML element IDs (spec-verbatim, including VInt leading-1 markers).
const ID_EBML: &[u8] = &[0x1A, 0x45, 0xDF, 0xA3];
const ID_DOC_TYPE: &[u8] = &[0x42, 0x82];
const ID_DOC_TYPE_VERSION: &[u8] = &[0x42, 0x87];
const ID_DOC_TYPE_READ_VERSION: &[u8] = &[0x42, 0x85];

const ID_SEGMENT: &[u8] = &[0x18, 0x53, 0x80, 0x67];
const ID_INFO: &[u8] = &[0x15, 0x49, 0xA9, 0x66];
const ID_TIMESTAMP_SCALE: &[u8] = &[0x2A, 0xD7, 0xB1];
const ID_DURATION: &[u8] = &[0x44, 0x89];
const ID_MUXING_APP: &[u8] = &[0x4D, 0x80];
const ID_WRITING_APP: &[u8] = &[0x57, 0x41];

const ID_TRACKS: &[u8] = &[0x16, 0x54, 0xAE, 0x6B];
const ID_TRACK_ENTRY: &[u8] = &[0xAE];
const ID_TRACK_NUMBER: &[u8] = &[0xD7];
const ID_TRACK_UID: &[u8] = &[0x73, 0xC5];
const ID_TRACK_TYPE: &[u8] = &[0x83];
const ID_CODEC_ID: &[u8] = &[0x86];
const ID_CODEC_PRIVATE: &[u8] = &[0x63, 0xA2];
// DefaultDuration (per-frame ns) — Matroska element 0x23E383, VInt-encoded.
const ID_DEFAULT_DURATION: &[u8] = &[0x23, 0xE3, 0x83];
const ID_VIDEO: &[u8] = &[0xE0];
const ID_PIXEL_WIDTH: &[u8] = &[0xB0];
const ID_PIXEL_HEIGHT: &[u8] = &[0xBA];

const ID_COLOUR: &[u8] = &[0x55, 0xB0];
const ID_MATRIX_COEFFICIENTS: &[u8] = &[0x55, 0xB1];
const ID_BITS_PER_CHANNEL: &[u8] = &[0x55, 0xB2];
const ID_RANGE: &[u8] = &[0x55, 0xB9];
const ID_TRANSFER_CHARACTERISTICS: &[u8] = &[0x55, 0xBA];
const ID_PRIMARIES: &[u8] = &[0x55, 0xBB];
const ID_MAX_CLL: &[u8] = &[0x55, 0xBC];
const ID_MAX_FALL: &[u8] = &[0x55, 0xBD];
const ID_MASTERING_METADATA: &[u8] = &[0x55, 0xD0];
const ID_PRIMARY_R_CHROMATICITY_X: &[u8] = &[0x55, 0xD1];
const ID_PRIMARY_R_CHROMATICITY_Y: &[u8] = &[0x55, 0xD2];
const ID_PRIMARY_G_CHROMATICITY_X: &[u8] = &[0x55, 0xD3];
const ID_PRIMARY_G_CHROMATICITY_Y: &[u8] = &[0x55, 0xD4];
const ID_PRIMARY_B_CHROMATICITY_X: &[u8] = &[0x55, 0xD5];
const ID_PRIMARY_B_CHROMATICITY_Y: &[u8] = &[0x55, 0xD6];
const ID_WHITE_POINT_CHROMATICITY_X: &[u8] = &[0x55, 0xD7];
const ID_WHITE_POINT_CHROMATICITY_Y: &[u8] = &[0x55, 0xD8];
const ID_LUMINANCE_MAX: &[u8] = &[0x55, 0xD9];
const ID_LUMINANCE_MIN: &[u8] = &[0x55, 0xDA];

const ID_TAGS: &[u8] = &[0x12, 0x54, 0xC3, 0x67];
const ID_TAG: &[u8] = &[0x73, 0x73];
const ID_TARGETS: &[u8] = &[0x63, 0xC0];
const ID_TARGET_TYPE_VALUE: &[u8] = &[0x68, 0xCA];
const ID_TAG_TRACK_UID: &[u8] = &[0x63, 0xC5];
const ID_SIMPLE_TAG: &[u8] = &[0x67, 0xC8];
const ID_TAG_NAME: &[u8] = &[0x45, 0xA3];
const ID_TAG_STRING: &[u8] = &[0x44, 0x87];

const ID_CLUSTER: &[u8] = &[0x1F, 0x43, 0xB6, 0x75];
const ID_TIMESTAMP: &[u8] = &[0xE7];
const ID_SIMPLE_BLOCK: &[u8] = &[0xA3];

/// EBML header declaring a Matroska document the parser will accept.
fn ebml_header() -> Vec<u8> {
    let mut body = Vec::new();
    body.extend(el_str(ID_DOC_TYPE, "matroska"));
    body.extend(el_uint(ID_DOC_TYPE_VERSION, 4));
    body.extend(el_uint(ID_DOC_TYPE_READ_VERSION, 2));
    el(ID_EBML, &body)
}

fn info_element(duration_ticks: f64, timestamp_scale: u64) -> Vec<u8> {
    let mut body = Vec::new();
    body.extend(el_uint(ID_TIMESTAMP_SCALE, timestamp_scale));
    body.extend(el_float(ID_DURATION, duration_ticks));
    body.extend(el_str(ID_MUXING_APP, "test"));
    body.extend(el_str(ID_WRITING_APP, "test"));
    el(ID_INFO, &body)
}

fn video_element(width: u64, height: u64, colour: Option<Vec<u8>>) -> Vec<u8> {
    let mut body = Vec::new();
    body.extend(el_uint(ID_PIXEL_WIDTH, width));
    body.extend(el_uint(ID_PIXEL_HEIGHT, height));
    if let Some(c) = colour {
        body.extend(c);
    }
    el(ID_VIDEO, &body)
}

fn colour_element_hdr10() -> Vec<u8> {
    let mut mastering = Vec::new();
    mastering.extend(el_float(ID_PRIMARY_R_CHROMATICITY_X, 0.680));
    mastering.extend(el_float(ID_PRIMARY_R_CHROMATICITY_Y, 0.320));
    mastering.extend(el_float(ID_PRIMARY_G_CHROMATICITY_X, 0.265));
    mastering.extend(el_float(ID_PRIMARY_G_CHROMATICITY_Y, 0.690));
    mastering.extend(el_float(ID_PRIMARY_B_CHROMATICITY_X, 0.150));
    mastering.extend(el_float(ID_PRIMARY_B_CHROMATICITY_Y, 0.060));
    mastering.extend(el_float(ID_WHITE_POINT_CHROMATICITY_X, 0.3127));
    mastering.extend(el_float(ID_WHITE_POINT_CHROMATICITY_Y, 0.3290));
    mastering.extend(el_float(ID_LUMINANCE_MAX, 1000.0));
    mastering.extend(el_float(ID_LUMINANCE_MIN, 0.005));

    let mut body = Vec::new();
    body.extend(el_uint(ID_MATRIX_COEFFICIENTS, 9)); // BT.2020 NCL
    body.extend(el_uint(ID_BITS_PER_CHANNEL, 10));
    body.extend(el_uint(ID_RANGE, 1)); // Broadcast / studio
    body.extend(el_uint(ID_TRANSFER_CHARACTERISTICS, 16)); // ST 2084 (PQ)
    body.extend(el_uint(ID_PRIMARIES, 9)); // BT.2020
    body.extend(el_uint(ID_MAX_CLL, 1000));
    body.extend(el_uint(ID_MAX_FALL, 400));
    body.extend(el(ID_MASTERING_METADATA, &mastering));
    el(ID_COLOUR, &body)
}

fn colour_element_full_range_bt709() -> Vec<u8> {
    let mut body = Vec::new();
    body.extend(el_uint(ID_MATRIX_COEFFICIENTS, 1)); // BT.709
    body.extend(el_uint(ID_RANGE, 2)); // Full
    body.extend(el_uint(ID_TRANSFER_CHARACTERISTICS, 1));
    body.extend(el_uint(ID_PRIMARIES, 1));
    el(ID_COLOUR, &body)
}

fn tracks_element(codec_id: &str, codec_private: &[u8], colour: Option<Vec<u8>>) -> Vec<u8> {
    let mut video = video_element(640, 480, colour);

    let mut track_entry = Vec::new();
    track_entry.extend(el_uint(ID_TRACK_NUMBER, 1));
    track_entry.extend(el_uint(ID_TRACK_UID, 42));
    track_entry.extend(el_uint(ID_TRACK_TYPE, 1)); // video
    track_entry.extend(el_str(ID_CODEC_ID, codec_id));
    if !codec_private.is_empty() {
        track_entry.extend(el(ID_CODEC_PRIVATE, codec_private));
    }
    track_entry.append(&mut video);

    let tracks_body = el(ID_TRACK_ENTRY, &track_entry);
    el(ID_TRACKS, &tracks_body)
}

fn tag_bit_rate(track_uid: u64, bit_rate: u64) -> Vec<u8> {
    let mut targets = Vec::new();
    targets.extend(el_uint(ID_TARGET_TYPE_VALUE, 30));
    targets.extend(el_uint(ID_TAG_TRACK_UID, track_uid));

    let mut simple_tag = Vec::new();
    simple_tag.extend(el_str(ID_TAG_NAME, "BIT_RATE"));
    simple_tag.extend(el_str(ID_TAG_STRING, &bit_rate.to_string()));

    let mut tag_body = Vec::new();
    tag_body.extend(el(ID_TARGETS, &targets));
    tag_body.extend(el(ID_SIMPLE_TAG, &simple_tag));

    let tag = el(ID_TAG, &tag_body);
    el(ID_TAGS, &tag)
}

/// A SimpleBlock header: `[track vint][int16 timestamp][flags u8]`. Track
/// 1 fits in a 1-byte VInt of value `0x81`. We emit a keyframe (flags 0x80).
fn simple_block_header(track: u8) -> [u8; 4] {
    [0x80 | track, 0x00, 0x00, 0x80]
}

fn cluster_with_sample(timestamp_ms: u64, track: u8, payload: &[u8]) -> Vec<u8> {
    let mut block = Vec::new();
    block.extend(&simple_block_header(track));
    block.extend(payload);

    let mut body = Vec::new();
    body.extend(el_uint(ID_TIMESTAMP, timestamp_ms));
    body.extend(el(ID_SIMPLE_BLOCK, &block));
    el(ID_CLUSTER, &body)
}

/// Build a full MKV byte stream with the chosen track config, an
/// optional colour element, an optional tag, and a single Cluster
/// carrying the supplied sample payload on track 1.
struct MkvBuilder {
    codec_id: String,
    codec_private: Vec<u8>,
    colour: Option<Vec<u8>>,
    tag: Option<Vec<u8>>,
    sample_payload: Vec<u8>,
    duration_ticks: f64,
    timestamp_scale: u64,
}

impl MkvBuilder {
    fn new(codec_id: &str) -> Self {
        Self {
            codec_id: codec_id.to_string(),
            codec_private: Vec::new(),
            colour: None,
            tag: None,
            sample_payload: vec![0u8; 4], // default: one empty length-prefixed NAL
            duration_ticks: 1000.0,
            timestamp_scale: 1_000_000, // 1 ms per tick → 1 s duration
        }
    }

    fn build(self) -> Vec<u8> {
        let info = info_element(self.duration_ticks, self.timestamp_scale);
        let tracks = tracks_element(&self.codec_id, &self.codec_private, self.colour);
        let cluster = cluster_with_sample(0, 1, &self.sample_payload);

        let mut segment_body = Vec::new();
        segment_body.extend(info);
        segment_body.extend(tracks);
        if let Some(tag) = self.tag {
            segment_body.extend(tag);
        }
        segment_body.extend(cluster);

        let mut out = Vec::new();
        out.extend(ebml_header());
        out.extend(el(ID_SEGMENT, &segment_body));
        out
    }
}

#[test]
fn mkv_parses_colour_element() {
    let data = MkvBuilder {
        codec_id: "V_VP9".into(),
        codec_private: Vec::new(),
        colour: Some(colour_element_hdr10()),
        tag: None,
        sample_payload: vec![0xAA, 0xBB, 0xCC, 0xDD],
        duration_ticks: 1000.0,
        timestamp_scale: 1_000_000,
    }
    .build();

    let res = demux_mkv(&data).expect("demux MKV");
    // BT.2020 NCL (matrix 9) → ColorSpace::Bt2020.
    assert_eq!(res.info.color_space, ColorSpace::Bt2020);
    // The core H.273 quartet rides on `StreamInfo.color_metadata`; the
    // MKV demuxer populates every field from the Colour element.
    assert_eq!(res.info.color_metadata.matrix_coefficients, 9);
    assert_eq!(res.info.color_metadata.colour_primaries, 9);
    assert_eq!(
        res.info.color_metadata.transfer,
        TransferFn::St2084,
        "ST 2084 (PQ) transfer must round-trip"
    );
    // Range=1 (Broadcast) must NOT set full_range.
    assert!(!res.info.color_metadata.full_range);

    // Extended HDR10 side-data surfaces via `probe_mkv_color_info` so
    // HDR10-aware mux paths can round-trip it.
    let extra = probe_mkv_color_info(&data).expect("probe");
    assert_eq!(extra.bits_per_channel, Some(10));
    assert_eq!(extra.max_cll, Some(1000));
    assert_eq!(extra.max_fall, Some(400));
    let m = extra.mastering.expect("mastering metadata populated");
    assert_eq!(m.luminance_max, Some(1000.0));
    assert_eq!(m.luminance_min, Some(0.005));
    assert_eq!(m.white_point_chromaticity_x, Some(0.3127));

    // Full-range sibling must round-trip `full_range = true`.
    let full_range_data = MkvBuilder {
        codec_id: "V_VP9".into(),
        codec_private: Vec::new(),
        colour: Some(colour_element_full_range_bt709()),
        tag: None,
        sample_payload: vec![0xAA],
        duration_ticks: 1000.0,
        timestamp_scale: 1_000_000,
    }
    .build();
    let res2 = demux_mkv(&full_range_data).expect("demux full-range MKV");
    assert_eq!(res2.info.color_space, ColorSpace::Bt709);
    assert!(res2.info.color_metadata.full_range);
    assert_eq!(res2.info.color_metadata.matrix_coefficients, 1);
}

#[test]
fn mkv_bitrate_from_tag_or_computed() {
    // --- Path 1: explicit BIT_RATE tag wins ---------------------------
    let data = MkvBuilder {
        codec_id: "V_VP9".into(),
        codec_private: Vec::new(),
        colour: None,
        tag: Some(tag_bit_rate(42, 5_000_000)),
        sample_payload: vec![0u8; 100],
        duration_ticks: 1000.0,
        timestamp_scale: 1_000_000,
    }
    .build();
    let res = demux_mkv(&data).expect("demux MKV with tag");
    assert_eq!(
        res.info.bitrate, 5_000_000,
        "tag-scoped BIT_RATE must override the computed fallback"
    );

    // --- Path 2: no tag → computed from total sample bytes / duration -
    // 1000 bytes × 8 bits / 1.0 s = 8000 bps.
    let data_no_tag = MkvBuilder {
        codec_id: "V_VP9".into(),
        codec_private: Vec::new(),
        colour: None,
        tag: None,
        sample_payload: vec![0u8; 1000],
        duration_ticks: 1000.0,
        timestamp_scale: 1_000_000,
    }
    .build();
    let res2 = demux_mkv(&data_no_tag).expect("demux MKV no tag");
    assert!(
        res2.info.bitrate > 0,
        "bitrate must never be zero when samples + duration are known"
    );
    // Allow a little slack for EBML overhead in sample accounting.
    assert!(
        res2.info.bitrate >= 7_900 && res2.info.bitrate <= 8_100,
        "expected ≈8000 bps, got {}",
        res2.info.bitrate
    );

    // --- Path 3: BPS alias accepted (mkvmerge writes BPS-eng) ---------
    let mut bps_tag_body = Vec::new();
    bps_tag_body.extend(el(ID_TARGETS, &el_uint(ID_TARGET_TYPE_VALUE, 30)));
    let mut st = Vec::new();
    st.extend(el_str(ID_TAG_NAME, "BPS"));
    st.extend(el_str(ID_TAG_STRING, "1234567"));
    bps_tag_body.extend(el(ID_SIMPLE_TAG, &st));
    let bps_tag = el(ID_TAGS, &el(ID_TAG, &bps_tag_body));

    let data_bps = MkvBuilder {
        codec_id: "V_VP9".into(),
        codec_private: Vec::new(),
        colour: None,
        tag: Some(bps_tag),
        sample_payload: vec![0u8; 100],
        duration_ticks: 1000.0,
        timestamp_scale: 1_000_000,
    }
    .build();
    let res3 = demux_mkv(&data_bps).expect("demux MKV with BPS");
    assert_eq!(res3.info.bitrate, 1_234_567);
}

/// Build a minimal avcC with exactly one SPS and one PPS, length_size=4.
fn make_minimal_avcc(sps: &[u8], pps: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    out.push(0x01); // configurationVersion
    out.push(0x42); // profile
    out.push(0x00); // compat
    out.push(0x1e); // level
    out.push(0xff); // reserved(6)=1|lengthSizeMinusOne=3
    out.push(0xe1); // reserved(3)=7|num_sps=1
    out.extend_from_slice(&(sps.len() as u16).to_be_bytes());
    out.extend_from_slice(sps);
    out.push(0x01); // num_pps=1
    out.extend_from_slice(&(pps.len() as u16).to_be_bytes());
    out.extend_from_slice(pps);
    out
}

#[test]
fn mkv_avcc_to_annexb_first_sample_has_sps_pps() {
    let sps = [0x67u8, 0x42, 0x00, 0x1e, 0xab, 0x40];
    let pps = [0x68u8, 0xce, 0x3c, 0x80];
    let avcc = make_minimal_avcc(&sps, &pps);

    let slice_nalu = [0x65u8, 0x11, 0x22, 0x33, 0x44, 0x55];
    // Sample: 4-byte big-endian length prefix + NAL unit payload.
    let mut sample = Vec::new();
    sample.extend_from_slice(&(slice_nalu.len() as u32).to_be_bytes());
    sample.extend_from_slice(&slice_nalu);

    let data = MkvBuilder {
        codec_id: "V_MPEG4/ISO/AVC".into(),
        codec_private: avcc,
        colour: None,
        tag: None,
        sample_payload: sample,
        duration_ticks: 1000.0,
        timestamp_scale: 1_000_000,
    }
    .build();

    let res = demux_mkv(&data).expect("demux H.264 MKV");
    assert_eq!(res.codec, "h264");
    assert_eq!(res.samples.len(), 1);
    let out = &res.samples[0];

    let start = [0u8, 0, 0, 1];
    // Expected layout: SPS NAL, PPS NAL, slice NAL — each preceded by
    // the 4-byte Annex-B start code.
    let mut expected = Vec::new();
    expected.extend_from_slice(&start);
    expected.extend_from_slice(&sps);
    expected.extend_from_slice(&start);
    expected.extend_from_slice(&pps);
    expected.extend_from_slice(&start);
    expected.extend_from_slice(&slice_nalu);
    assert_eq!(out, &expected);

    // Also sanity-check the specific ordering / start codes requested
    // in the task brief.
    assert_eq!(&out[0..4], &start, "first NAL must start with Annex-B code");
    assert_eq!(out[4], sps[0], "first NAL must be SPS");
    let pps_offset = 4 + sps.len();
    assert_eq!(&out[pps_offset..pps_offset + 4], &start);
    assert_eq!(out[pps_offset + 4], pps[0], "second NAL must be PPS");
}

/// Build an hvcC with VPS + SPS + PPS in separate arrays (three arrays).
fn make_multi_array_hvcc(vps: &[u8], sps: &[u8], pps: &[u8]) -> Vec<u8> {
    let mut out = vec![0u8; 23];
    out[0] = 1; // configurationVersion
    out[21] = 0xf3; // reserved(6)=0xFC | lengthSizeMinusOne=3 → 4-byte prefix
    out[22] = 3; // numOfArrays

    // Array 1: VPS (nal_unit_type = 32)
    out.push(32);
    out.extend_from_slice(&1u16.to_be_bytes());
    out.extend_from_slice(&(vps.len() as u16).to_be_bytes());
    out.extend_from_slice(vps);

    // Array 2: SPS (33)
    out.push(33);
    out.extend_from_slice(&1u16.to_be_bytes());
    out.extend_from_slice(&(sps.len() as u16).to_be_bytes());
    out.extend_from_slice(sps);

    // Array 3: PPS (34)
    out.push(34);
    out.extend_from_slice(&1u16.to_be_bytes());
    out.extend_from_slice(&(pps.len() as u16).to_be_bytes());
    out.extend_from_slice(pps);

    out
}

#[test]
fn mkv_hvcc_to_annexb_multi_array() {
    let vps = [0x40u8, 0x01, 0x0c];
    let sps = [0x42u8, 0x01, 0x01, 0x22];
    let pps = [0x44u8, 0x01];
    let hvcc = make_multi_array_hvcc(&vps, &sps, &pps);

    let slice_nalu = [0x26u8, 0x01, 0xAA, 0xBB];
    let mut sample = Vec::new();
    sample.extend_from_slice(&(slice_nalu.len() as u32).to_be_bytes());
    sample.extend_from_slice(&slice_nalu);

    let data = MkvBuilder {
        codec_id: "V_MPEGH/ISO/HEVC".into(),
        codec_private: hvcc,
        colour: None,
        tag: None,
        sample_payload: sample,
        duration_ticks: 1000.0,
        timestamp_scale: 1_000_000,
    }
    .build();

    let res = demux_mkv(&data).expect("demux HEVC MKV");
    assert_eq!(res.codec, "h265");
    assert_eq!(res.samples.len(), 1);
    let out = &res.samples[0];

    let start = [0u8, 0, 0, 1];
    let mut expected = Vec::new();
    expected.extend_from_slice(&start);
    expected.extend_from_slice(&vps);
    expected.extend_from_slice(&start);
    expected.extend_from_slice(&sps);
    expected.extend_from_slice(&start);
    expected.extend_from_slice(&pps);
    expected.extend_from_slice(&start);
    expected.extend_from_slice(&slice_nalu);
    assert_eq!(
        out, &expected,
        "VPS, SPS, PPS must be prepended in hvcC array order before the slice NAL"
    );
}

#[test]
fn mkv_dispatch_routes_to_demux_mkv() {
    // Ensure the top-level `demux` dispatcher still picks the MKV path
    // for an EBML-signed input.
    let data = MkvBuilder::new("V_VP9").build();
    let res = demux::demux(&data).expect("top-level demux");
    assert_eq!(res.codec, "vp9");
}

/// Squad-32 regression: when an MKV ships without a segment-level `Duration`
/// element (live recordings, some streaming WebMs, MKV-from-screen-recorder
/// software), the frame-rate fallback chain must consult the per-track
/// `DefaultDuration` (Matroska 0x23E383, ns per frame) rather than silently
/// returning 30.0. Historically this was the source of the "MKV
/// DefaultDuration parsing issue" listed as a pre-existing fail in
/// CLAUDE.md / TODO.md — the parser surfaced the field correctly, but
/// `demux_mkv` never consumed it.
#[test]
fn mkv_default_duration_is_used_when_segment_duration_missing() {
    // 16,666,667 ns ≈ 60 fps. Build a custom MKV inline (the standard
    // MkvBuilder always emits a Duration element).
    let dd_ns: u64 = 16_666_667;

    // Info: TimestampScale only — no Duration.
    let mut info_body = Vec::new();
    info_body.extend(el_uint(ID_TIMESTAMP_SCALE, 1_000_000));
    info_body.extend(el_str(ID_MUXING_APP, "test"));
    info_body.extend(el_str(ID_WRITING_APP, "test"));
    let info = el(ID_INFO, &info_body);

    // Tracks: VP9 video with DefaultDuration set.
    let video = video_element(640, 480, None);
    let mut track_entry = Vec::new();
    track_entry.extend(el_uint(ID_TRACK_NUMBER, 1));
    track_entry.extend(el_uint(ID_TRACK_UID, 42));
    track_entry.extend(el_uint(ID_TRACK_TYPE, 1));
    track_entry.extend(el_str(ID_CODEC_ID, "V_VP9"));
    track_entry.extend(el_uint(ID_DEFAULT_DURATION, dd_ns));
    track_entry.extend(video);
    let tracks_body = el(ID_TRACK_ENTRY, &track_entry);
    let tracks = el(ID_TRACKS, &tracks_body);

    let cluster = cluster_with_sample(0, 1, &[0xDE, 0xAD, 0xBE, 0xEF]);

    let mut segment_body = Vec::new();
    segment_body.extend(info);
    segment_body.extend(tracks);
    segment_body.extend(cluster);

    let mut data = Vec::new();
    data.extend(ebml_header());
    data.extend(el(ID_SEGMENT, &segment_body));

    let res = demux_mkv(&data).expect("demux MKV with DefaultDuration but no Duration");
    assert_eq!(res.codec, "vp9");
    assert_eq!(
        res.info.duration, 0.0,
        "no segment Duration ⇒ duration is 0"
    );
    let expected_fps = 1_000_000_000.0 / dd_ns as f64;
    let delta = (res.info.frame_rate - expected_fps).abs();
    assert!(
        delta < 0.01,
        "frame_rate from DefaultDuration: got {}, expected ~{} (Δ {})",
        res.info.frame_rate,
        expected_fps,
        delta
    );
}

/// Companion: when neither Duration nor DefaultDuration is present, we
/// keep the historical 30.0 sentinel. Belt-and-braces against accidentally
/// regressing to a panic / NaN.
#[test]
fn mkv_frame_rate_falls_back_to_30_when_both_durations_missing() {
    let mut info_body = Vec::new();
    info_body.extend(el_uint(ID_TIMESTAMP_SCALE, 1_000_000));
    info_body.extend(el_str(ID_MUXING_APP, "test"));
    info_body.extend(el_str(ID_WRITING_APP, "test"));
    let info = el(ID_INFO, &info_body);

    let video = video_element(640, 480, None);
    let mut track_entry = Vec::new();
    track_entry.extend(el_uint(ID_TRACK_NUMBER, 1));
    track_entry.extend(el_uint(ID_TRACK_UID, 42));
    track_entry.extend(el_uint(ID_TRACK_TYPE, 1));
    track_entry.extend(el_str(ID_CODEC_ID, "V_VP9"));
    track_entry.extend(video);
    let tracks_body = el(ID_TRACK_ENTRY, &track_entry);
    let tracks = el(ID_TRACKS, &tracks_body);

    let cluster = cluster_with_sample(0, 1, &[0u8; 4]);

    let mut segment_body = Vec::new();
    segment_body.extend(info);
    segment_body.extend(tracks);
    segment_body.extend(cluster);

    let mut data = Vec::new();
    data.extend(ebml_header());
    data.extend(el(ID_SEGMENT, &segment_body));

    let res = demux_mkv(&data).expect("demux MKV with neither duration");
    assert_eq!(res.info.frame_rate, 30.0);
}
