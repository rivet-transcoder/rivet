//! Integration tests for HDR static-metadata extraction across all
//! source containers (Squad-21).
//!
//! Coverage:
//!   * MP4 visual sample-entry `mdcv` + `clli` boxes — synthetic MP4
//!     built end-to-end so the test exercises `demux_mp4` and the
//!     stsd box-walk it depends on.
//!   * MKV `MasteringMetadata` + `MaxCLL` / `MaxFALL` — synthetic MKV
//!     built end-to-end so the test exercises `demux_mkv` and the
//!     unified ColorMetadata wiring.
//!   * Real HDR10 media (`bbb_hdr10.mp4` or any HEVC HDR10 sample) —
//!     opt-in: skips with a notice when no fixture is present.

use codec::frame::{ContentLightLevel, MasteringDisplay};
use container::demux::{demux_mkv, demux_mp4};

// === Box helpers ===

fn box_(fourcc: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    let size = 8 + payload.len();
    let mut out = Vec::with_capacity(size);
    out.extend_from_slice(&(size as u32).to_be_bytes());
    out.extend_from_slice(fourcc);
    out.extend_from_slice(payload);
    out
}

fn full_box_(fourcc: &[u8; 4], version: u8, flags: u32, payload: &[u8]) -> Vec<u8> {
    let mut body = Vec::with_capacity(4 + payload.len());
    body.push(version);
    let f = flags & 0x00FF_FFFF;
    body.push(((f >> 16) & 0xFF) as u8);
    body.push(((f >> 8) & 0xFF) as u8);
    body.push((f & 0xFF) as u8);
    body.extend_from_slice(payload);
    box_(fourcc, &body)
}

/// Build an `mdcv` box per ISO/IEC 23001-17 §7.3.
/// Wire order is GBR.
fn build_mdcv(md: &MasteringDisplay) -> Vec<u8> {
    let mut p = Vec::new();
    p.extend_from_slice(&md.primaries_g_x.to_be_bytes());
    p.extend_from_slice(&md.primaries_g_y.to_be_bytes());
    p.extend_from_slice(&md.primaries_b_x.to_be_bytes());
    p.extend_from_slice(&md.primaries_b_y.to_be_bytes());
    p.extend_from_slice(&md.primaries_r_x.to_be_bytes());
    p.extend_from_slice(&md.primaries_r_y.to_be_bytes());
    p.extend_from_slice(&md.white_point_x.to_be_bytes());
    p.extend_from_slice(&md.white_point_y.to_be_bytes());
    p.extend_from_slice(&md.max_luminance.to_be_bytes());
    p.extend_from_slice(&md.min_luminance.to_be_bytes());
    box_(b"mdcv", &p)
}

fn build_clli(cll: &ContentLightLevel) -> Vec<u8> {
    let mut p = Vec::new();
    p.extend_from_slice(&cll.max_cll.to_be_bytes());
    p.extend_from_slice(&cll.max_fall.to_be_bytes());
    box_(b"clli", &p)
}

/// Build a minimal MP4 carrying an HEVC video track with optional
/// `mdcv` and `clli` boxes nested inside the `hvc1` visual sample
/// entry. Returns the full byte stream demux_mp4 can read.
fn build_synthetic_mp4_with_hdr_boxes(
    width: u16,
    height: u16,
    mdcv_box: Option<Vec<u8>>,
    clli_box: Option<Vec<u8>>,
) -> Vec<u8> {
    // Minimal hvcC body — Configuration_version + flags only. mp4
    // crate's parser just walks descendants of the visual sample
    // entry; we don't need a valid hvcC body for the box-walk test.
    let hvcc_body = vec![
        0x01, 0x01, 0x60, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xF0, 0x00, 0xFC, 0xFD, 0xFA,
        0xFA, 0x00, 0x00,
    ];
    let hvcc = box_(b"hvcC", &hvcc_body);

    let ftyp = box_(b"ftyp", &{
        let mut p = Vec::new();
        p.extend_from_slice(b"isom");
        p.extend_from_slice(&0u32.to_be_bytes());
        p.extend_from_slice(b"isom");
        p.extend_from_slice(b"mp41");
        p
    });

    let mvhd = full_box_(b"mvhd", 0, 0, &{
        let mut p = Vec::new();
        p.extend_from_slice(&0u32.to_be_bytes());
        p.extend_from_slice(&0u32.to_be_bytes());
        p.extend_from_slice(&1000u32.to_be_bytes());
        p.extend_from_slice(&1000u32.to_be_bytes());
        p.extend_from_slice(&0x00010000u32.to_be_bytes());
        p.extend_from_slice(&0x0100u16.to_be_bytes());
        p.extend_from_slice(&0u16.to_be_bytes());
        p.extend_from_slice(&[0u8; 8]);
        p.extend_from_slice(&0x00010000u32.to_be_bytes());
        p.extend_from_slice(&0u32.to_be_bytes());
        p.extend_from_slice(&0u32.to_be_bytes());
        p.extend_from_slice(&0u32.to_be_bytes());
        p.extend_from_slice(&0x00010000u32.to_be_bytes());
        p.extend_from_slice(&0u32.to_be_bytes());
        p.extend_from_slice(&0u32.to_be_bytes());
        p.extend_from_slice(&0u32.to_be_bytes());
        p.extend_from_slice(&0x40000000u32.to_be_bytes());
        p.extend_from_slice(&[0u8; 24]);
        p.extend_from_slice(&2u32.to_be_bytes());
        p
    });

    let tkhd = full_box_(b"tkhd", 0, 0x000007, &{
        let mut p = Vec::new();
        p.extend_from_slice(&0u32.to_be_bytes());
        p.extend_from_slice(&0u32.to_be_bytes());
        p.extend_from_slice(&1u32.to_be_bytes());
        p.extend_from_slice(&0u32.to_be_bytes());
        p.extend_from_slice(&1000u32.to_be_bytes());
        p.extend_from_slice(&[0u8; 8]);
        p.extend_from_slice(&0u16.to_be_bytes());
        p.extend_from_slice(&0u16.to_be_bytes());
        p.extend_from_slice(&0u16.to_be_bytes());
        p.extend_from_slice(&0u16.to_be_bytes());
        p.extend_from_slice(&0x00010000u32.to_be_bytes());
        p.extend_from_slice(&0u32.to_be_bytes());
        p.extend_from_slice(&0u32.to_be_bytes());
        p.extend_from_slice(&0u32.to_be_bytes());
        p.extend_from_slice(&0x00010000u32.to_be_bytes());
        p.extend_from_slice(&0u32.to_be_bytes());
        p.extend_from_slice(&0u32.to_be_bytes());
        p.extend_from_slice(&0u32.to_be_bytes());
        p.extend_from_slice(&0x40000000u32.to_be_bytes());
        p.extend_from_slice(&((width as u32) << 16).to_be_bytes());
        p.extend_from_slice(&((height as u32) << 16).to_be_bytes());
        p
    });

    let mdhd = full_box_(b"mdhd", 0, 0, &{
        let mut p = Vec::new();
        p.extend_from_slice(&0u32.to_be_bytes());
        p.extend_from_slice(&0u32.to_be_bytes());
        p.extend_from_slice(&1000u32.to_be_bytes());
        p.extend_from_slice(&1000u32.to_be_bytes());
        p.extend_from_slice(&0x55C4u16.to_be_bytes());
        p.extend_from_slice(&0u16.to_be_bytes());
        p
    });
    let hdlr = full_box_(b"hdlr", 0, 0, &{
        let mut p = Vec::new();
        p.extend_from_slice(&0u32.to_be_bytes());
        p.extend_from_slice(b"vide");
        p.extend_from_slice(&[0u8; 12]);
        p.push(0);
        p
    });
    let vmhd = full_box_(b"vmhd", 0, 0x000001, &{
        let mut p = Vec::new();
        p.extend_from_slice(&0u16.to_be_bytes());
        p.extend_from_slice(&[0u8; 6]);
        p
    });
    let url = full_box_(b"url ", 0, 0x000001, &[]);
    let dref = full_box_(b"dref", 0, 0, &{
        let mut p = Vec::new();
        p.extend_from_slice(&1u32.to_be_bytes());
        p.extend_from_slice(&url);
        p
    });
    let dinf = box_(b"dinf", &dref);

    // hvc1 sample entry — VisualSampleEntry header + hvcC + optional
    // mdcv + optional clli.
    let hvc1 = {
        let mut p = Vec::new();
        p.extend_from_slice(&[0u8; 6]); // reserved
        p.extend_from_slice(&1u16.to_be_bytes()); // data_ref_index
        p.extend_from_slice(&0u16.to_be_bytes()); // pre_defined
        p.extend_from_slice(&0u16.to_be_bytes()); // reserved
        p.extend_from_slice(&[0u8; 12]); // pre_defined[3]
        p.extend_from_slice(&width.to_be_bytes());
        p.extend_from_slice(&height.to_be_bytes());
        p.extend_from_slice(&0x00480000u32.to_be_bytes()); // horizres
        p.extend_from_slice(&0x00480000u32.to_be_bytes()); // vertres
        p.extend_from_slice(&0u32.to_be_bytes()); // reserved
        p.extend_from_slice(&1u16.to_be_bytes()); // frame_count
        p.extend_from_slice(&[0u8; 32]); // compressorname (32 bytes)
        p.extend_from_slice(&0x0018u16.to_be_bytes()); // depth
        p.extend_from_slice(&0xFFFFu16.to_be_bytes()); // pre_defined
        p.extend_from_slice(&hvcc);
        if let Some(b) = mdcv_box {
            p.extend_from_slice(&b);
        }
        if let Some(b) = clli_box {
            p.extend_from_slice(&b);
        }
        box_(b"hvc1", &p)
    };
    let stsd = full_box_(b"stsd", 0, 0, &{
        let mut p = Vec::new();
        p.extend_from_slice(&1u32.to_be_bytes());
        p.extend_from_slice(&hvc1);
        p
    });

    // One sample (placeholder) so demux_mp4 has a sample to walk.
    let samples = vec![vec![0u8, 0, 0, 4, 0x40, 0x01, 0xFF, 0xFF]];

    let stts = full_box_(b"stts", 0, 0, &{
        let mut p = Vec::new();
        p.extend_from_slice(&1u32.to_be_bytes());
        p.extend_from_slice(&(samples.len() as u32).to_be_bytes());
        p.extend_from_slice(&1000u32.to_be_bytes());
        p
    });
    let stsc = full_box_(b"stsc", 0, 0, &{
        let mut p = Vec::new();
        p.extend_from_slice(&1u32.to_be_bytes());
        p.extend_from_slice(&1u32.to_be_bytes());
        p.extend_from_slice(&1u32.to_be_bytes());
        p.extend_from_slice(&1u32.to_be_bytes());
        p
    });
    let stsz = full_box_(b"stsz", 0, 0, &{
        let mut p = Vec::new();
        p.extend_from_slice(&0u32.to_be_bytes());
        p.extend_from_slice(&(samples.len() as u32).to_be_bytes());
        for s in &samples {
            p.extend_from_slice(&(s.len() as u32).to_be_bytes());
        }
        p
    });

    let build_moov = |stco_offsets: &[u32]| -> Vec<u8> {
        let stco = full_box_(b"stco", 0, 0, &{
            let mut p = Vec::new();
            p.extend_from_slice(&(stco_offsets.len() as u32).to_be_bytes());
            for off in stco_offsets {
                p.extend_from_slice(&off.to_be_bytes());
            }
            p
        });
        let stbl = box_(
            b"stbl",
            &[stsd.clone(), stts.clone(), stsc.clone(), stsz.clone(), stco].concat(),
        );
        let minf = box_(b"minf", &[vmhd.clone(), dinf.clone(), stbl].concat());
        let mdia = box_(b"mdia", &[mdhd.clone(), hdlr.clone(), minf].concat());
        let trak = box_(b"trak", &[tkhd.clone(), mdia].concat());
        box_(b"moov", &[mvhd.clone(), trak].concat())
    };

    let mut stco_offsets = vec![0u32; samples.len()];
    let moov_v1 = build_moov(&stco_offsets);
    let mdat_payload_start = ftyp.len() + moov_v1.len() + 8;
    let mut cur = mdat_payload_start;
    for (i, s) in samples.iter().enumerate() {
        stco_offsets[i] = cur as u32;
        cur += s.len();
    }
    let moov_v2 = build_moov(&stco_offsets);
    assert_eq!(
        moov_v1.len(),
        moov_v2.len(),
        "moov size must be deterministic"
    );
    let mdat_payload: Vec<u8> = samples.iter().flatten().copied().collect();
    let mdat = box_(b"mdat", &mdat_payload);

    let mut out = Vec::new();
    out.extend_from_slice(&ftyp);
    out.extend_from_slice(&moov_v2);
    out.extend_from_slice(&mdat);
    out
}

fn canonical_hdr10_mastering_display() -> MasteringDisplay {
    MasteringDisplay {
        primaries_r_x: 34000,
        primaries_r_y: 16000,
        primaries_g_x: 13250,
        primaries_g_y: 34500,
        primaries_b_x: 7500,
        primaries_b_y: 3000,
        white_point_x: 15635,
        white_point_y: 16450,
        max_luminance: 10_000_000,
        min_luminance: 50,
    }
}

#[test]
fn mp4_mdcv_box_in_hvc1_sample_entry_extracts_to_color_metadata() {
    let md = canonical_hdr10_mastering_display();
    let mp4 = build_synthetic_mp4_with_hdr_boxes(1920, 1080, Some(build_mdcv(&md)), None);
    let res = demux_mp4(&mp4).expect("demux MP4");
    let extracted = res
        .info
        .color_metadata
        .mastering_display
        .expect("mastering display populated by demux");
    assert_eq!(extracted, md, "mdcv must round-trip exact wire bytes");
    assert!(
        res.info.color_metadata.content_light_level.is_none(),
        "no clli written → must remain None"
    );
}

#[test]
fn mp4_clli_box_in_hvc1_sample_entry_extracts_to_color_metadata() {
    let cll = ContentLightLevel {
        max_cll: 4000,
        max_fall: 800,
    };
    let mp4 = build_synthetic_mp4_with_hdr_boxes(1920, 1080, None, Some(build_clli(&cll)));
    let res = demux_mp4(&mp4).expect("demux MP4");
    let extracted = res
        .info
        .color_metadata
        .content_light_level
        .expect("clli populated by demux");
    assert_eq!(extracted, cll);
    assert!(res.info.color_metadata.mastering_display.is_none());
}

#[test]
fn mp4_both_mdcv_and_clli_round_trip() {
    let md = canonical_hdr10_mastering_display();
    let cll = ContentLightLevel {
        max_cll: 1000,
        max_fall: 400,
    };
    let mp4 = build_synthetic_mp4_with_hdr_boxes(
        3840,
        2160,
        Some(build_mdcv(&md)),
        Some(build_clli(&cll)),
    );
    let res = demux_mp4(&mp4).expect("demux MP4");
    assert_eq!(res.info.color_metadata.mastering_display, Some(md));
    assert_eq!(res.info.color_metadata.content_light_level, Some(cll));
}

#[test]
fn mp4_without_hdr_boxes_leaves_color_metadata_unset() {
    let mp4 = build_synthetic_mp4_with_hdr_boxes(640, 480, None, None);
    let res = demux_mp4(&mp4).expect("demux MP4");
    assert!(res.info.color_metadata.mastering_display.is_none());
    assert!(res.info.color_metadata.content_light_level.is_none());
}

// === MKV synthetic build ===

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

fn size_vint_8(size: u64) -> [u8; 8] {
    let v = (1u64 << 56) | (size & ((1u64 << 56) - 1));
    v.to_be_bytes()
}

fn ebml_el(id: &[u8], payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(id.len() + 8 + payload.len());
    out.extend_from_slice(id);
    out.extend_from_slice(&size_vint_8(payload.len() as u64));
    out.extend_from_slice(payload);
    out
}

fn ebml_uint(id: &[u8], v: u64) -> Vec<u8> {
    ebml_el(id, &be_min_bytes(v))
}

fn ebml_float(id: &[u8], v: f64) -> Vec<u8> {
    ebml_el(id, &v.to_be_bytes())
}

fn ebml_str(id: &[u8], s: &str) -> Vec<u8> {
    ebml_el(id, s.as_bytes())
}

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

const ID_CLUSTER: &[u8] = &[0x1F, 0x43, 0xB6, 0x75];
const ID_TIMESTAMP: &[u8] = &[0xE7];
const ID_SIMPLE_BLOCK: &[u8] = &[0xA3];

fn ebml_header() -> Vec<u8> {
    let mut body = Vec::new();
    body.extend(ebml_str(ID_DOC_TYPE, "matroska"));
    body.extend(ebml_uint(ID_DOC_TYPE_VERSION, 4));
    body.extend(ebml_uint(ID_DOC_TYPE_READ_VERSION, 2));
    ebml_el(ID_EBML, &body)
}

fn build_mkv_with_full_hdr_metadata() -> Vec<u8> {
    // Mastering display: BT.2020 / ST 2086 / 1000 cd/m² peak.
    let mut mastering = Vec::new();
    mastering.extend(ebml_float(ID_PRIMARY_R_CHROMATICITY_X, 0.680));
    mastering.extend(ebml_float(ID_PRIMARY_R_CHROMATICITY_Y, 0.320));
    mastering.extend(ebml_float(ID_PRIMARY_G_CHROMATICITY_X, 0.265));
    mastering.extend(ebml_float(ID_PRIMARY_G_CHROMATICITY_Y, 0.690));
    mastering.extend(ebml_float(ID_PRIMARY_B_CHROMATICITY_X, 0.150));
    mastering.extend(ebml_float(ID_PRIMARY_B_CHROMATICITY_Y, 0.060));
    mastering.extend(ebml_float(ID_WHITE_POINT_CHROMATICITY_X, 0.3127));
    mastering.extend(ebml_float(ID_WHITE_POINT_CHROMATICITY_Y, 0.3290));
    mastering.extend(ebml_float(ID_LUMINANCE_MAX, 1000.0));
    mastering.extend(ebml_float(ID_LUMINANCE_MIN, 0.005));

    let mut colour = Vec::new();
    colour.extend(ebml_uint(ID_MATRIX_COEFFICIENTS, 9)); // BT.2020 NCL
    colour.extend(ebml_uint(ID_BITS_PER_CHANNEL, 10));
    colour.extend(ebml_uint(ID_RANGE, 1));
    colour.extend(ebml_uint(ID_TRANSFER_CHARACTERISTICS, 16)); // ST 2084
    colour.extend(ebml_uint(ID_PRIMARIES, 9));
    colour.extend(ebml_uint(ID_MAX_CLL, 1000));
    colour.extend(ebml_uint(ID_MAX_FALL, 400));
    colour.extend(ebml_el(ID_MASTERING_METADATA, &mastering));
    let colour_el = ebml_el(ID_COLOUR, &colour);

    let mut video = Vec::new();
    video.extend(ebml_uint(ID_PIXEL_WIDTH, 3840));
    video.extend(ebml_uint(ID_PIXEL_HEIGHT, 2160));
    video.extend(colour_el);
    let video_el = ebml_el(ID_VIDEO, &video);

    let mut track = Vec::new();
    track.extend(ebml_uint(ID_TRACK_NUMBER, 1));
    track.extend(ebml_uint(ID_TRACK_UID, 99));
    track.extend(ebml_uint(ID_TRACK_TYPE, 1)); // video
    track.extend(ebml_str(ID_CODEC_ID, "V_VP9"));
    track.extend(video_el);
    let tracks = ebml_el(ID_TRACKS, &ebml_el(ID_TRACK_ENTRY, &track));

    let mut info = Vec::new();
    info.extend(ebml_uint(ID_TIMESTAMP_SCALE, 1_000_000));
    info.extend(ebml_float(ID_DURATION, 1000.0));
    info.extend(ebml_str(ID_MUXING_APP, "test"));
    info.extend(ebml_str(ID_WRITING_APP, "test"));
    let info_el = ebml_el(ID_INFO, &info);

    // One placeholder cluster + sample so demux_mkv finds a frame to walk.
    let mut block = Vec::new();
    block.push(0x81); // track 1, vint
    block.extend_from_slice(&0i16.to_be_bytes());
    block.push(0x80); // keyframe flag
    block.push(0xAA); // payload
    let mut cluster = Vec::new();
    cluster.extend(ebml_uint(ID_TIMESTAMP, 0));
    cluster.extend(ebml_el(ID_SIMPLE_BLOCK, &block));
    let cluster_el = ebml_el(ID_CLUSTER, &cluster);

    let mut segment_body = Vec::new();
    segment_body.extend(info_el);
    segment_body.extend(tracks);
    segment_body.extend(cluster_el);

    let mut out = Vec::new();
    out.extend(ebml_header());
    out.extend(ebml_el(ID_SEGMENT, &segment_body));
    out
}

#[test]
fn mkv_mastering_metadata_and_cll_flow_to_unified_color_metadata() {
    let mkv = build_mkv_with_full_hdr_metadata();
    let res = demux_mkv(&mkv).expect("demux MKV");

    let cll = res
        .info
        .color_metadata
        .content_light_level
        .expect("MKV MaxCLL/MaxFALL must populate ContentLightLevel");
    assert_eq!(cll.max_cll, 1000);
    assert_eq!(cll.max_fall, 400);

    let md = res
        .info
        .color_metadata
        .mastering_display
        .expect("MKV MasteringMetadata must populate MasteringDisplay");
    // 0.680 * 50000 = 34000, 0.320 * 50000 = 16000, etc.
    assert_eq!(md.primaries_r_x, 34000);
    assert_eq!(md.primaries_r_y, 16000);
    assert_eq!(md.primaries_g_x, 13250);
    assert_eq!(md.primaries_g_y, 34500);
    assert_eq!(md.primaries_b_x, 7500);
    assert_eq!(md.primaries_b_y, 3000);
    assert_eq!(md.white_point_x, 15635);
    assert_eq!(md.white_point_y, 16450);
    // 1000.0 * 10000 = 10_000_000 ; 0.005 * 10000 = 50.
    assert_eq!(md.max_luminance, 10_000_000);
    assert_eq!(md.min_luminance, 50);
}

#[test]
fn real_hdr10_sample_round_trips_mdcv_clli() {
    let candidates = [
        "../../test_media/bbb_hdr10.mp4",
        "../../test_media/jellyfin_hevc_main10_hdr10_1080p.mp4",
        "test_media/bbb_hdr10.mp4",
    ];
    let path = candidates
        .iter()
        .map(std::path::Path::new)
        .find(|p| p.exists());
    let Some(path) = path else {
        eprintln!("[skip] no real HDR10 sample at any of {:?}", candidates);
        return;
    };
    let bytes = std::fs::read(path).expect("read HDR10 sample");
    let res = demux_mp4(&bytes).expect("demux real HDR10");
    println!(
        "real HDR10 sample {} → mastering_display={} content_light_level={}",
        path.display(),
        res.info.color_metadata.mastering_display.is_some(),
        res.info.color_metadata.content_light_level.is_some(),
    );
    // Don't assert hard — the file may carry the metadata via SEI in
    // the bitstream rather than container-side `mdcv`/`clli` boxes.
}
