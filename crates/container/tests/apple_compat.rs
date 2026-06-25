//! Apple QuickTime / iOS Safari compatibility tests for `Av1Mp4Muxer` output.
//!
//! Covers the three Apple-compat fixes that landed with audio (Squad-18):
//!   1. `ftyp` `compatible_brands` lists `av01` (REQUIRED per AV1-ISOBMFF
//!      v1.3.0 §2.1) plus `iso6` (covers co64 / largesize from 14496-12 v6)
//!      and `mp42` (AAC parsing rules).
//!   2. `colr nclx` atom present inside the `av01` visual sample entry so
//!      Apple's player doesn't silently fall back to BT.709 limited.
//!   3. AV1 sample entry fourcc is `av01` (no `hev1`/`hvc1`-style variant
//!      exists for AV1 — Apple support docs and AV1-ISOBMFF agree).
//!
//! Plus #9: 64-bit `largesize` mdat header for outputs >4 GiB. Tested via
//! the `force_largesize_mdat_for_test` knob to avoid producing a real 4 GiB
//! tempfile in unit-test time.

use bytes::Bytes;
use codec::encode::EncodedPacket;
use container::mux::Av1Mp4Muxer;

fn minimal_av1_first_packet() -> Bytes {
    let header: u8 = (1 << 3) | (1 << 1);
    let payload = [0u8; 5];
    let mut out = Vec::with_capacity(2 + payload.len());
    out.push(header);
    out.push(payload.len() as u8);
    out.extend_from_slice(&payload);
    Bytes::from(out)
}

fn opaque_packet(size: usize) -> Bytes {
    Bytes::from(vec![0xAA; size])
}

fn find_fourcc(data: &[u8], tag: &[u8; 4]) -> Option<usize> {
    data.windows(4).position(|w| w == tag)
}

fn count_fourcc(data: &[u8], tag: &[u8; 4]) -> usize {
    data.windows(4).filter(|w| *w == tag).count()
}

fn build_video_only(packets: u32, fps: f64, packet_size: usize) -> Bytes {
    let mut muxer = Av1Mp4Muxer::new(640, 480, fps).expect("muxer");
    muxer
        .add_packet(EncodedPacket {
            data: minimal_av1_first_packet(),
            pts: 0,
            is_keyframe: true,
        })
        .expect("first packet");
    for i in 1..packets {
        muxer
            .add_packet(EncodedPacket {
                data: opaque_packet(packet_size),
                pts: i as u64,
                is_keyframe: false,
            })
            .expect("packet");
    }
    muxer.finalize().expect("finalize")
}

#[test]
fn output_ftyp_includes_av01_brand() {
    let out = build_video_only(30, 30.0, 256);
    // ftyp lives at file start: size(4) + 'ftyp' + major(4) + minor(4) + brands.
    assert_eq!(&out[4..8], b"ftyp", "file must start with ftyp");
    let major = &out[8..12];
    assert_eq!(
        major, b"iso6",
        "Apple-compat: major_brand should be iso6 (covers co64/largesize from 14496-12 v6)"
    );
    let ftyp_size = u32::from_be_bytes([out[0], out[1], out[2], out[3]]) as usize;
    let compat = &out[16..ftyp_size];
    let brands: Vec<[u8; 4]> = compat
        .chunks_exact(4)
        .map(|c| [c[0], c[1], c[2], c[3]])
        .collect();
    assert!(
        brands.iter().any(|b| b == b"av01"),
        "AV1-ISOBMFF v1.3.0 §2.1 mandates av01 in compatible_brands"
    );
    assert!(
        brands.iter().any(|b| b == b"iso6"),
        "Apple QuickTime / iOS Safari needs structural ISOBMFF brand"
    );
    assert!(
        brands.iter().any(|b| b == b"mp42"),
        "mp42 brand needed for AAC parsing rules"
    );
}

#[test]
fn output_video_sample_entry_uses_av01_fourcc() {
    let out = build_video_only(30, 30.0, 256);
    // AV1-ISOBMFF defines exactly one sample entry fourcc for AV1: 'av01'.
    // No hev1/hvc1-style transport variant exists. Verify we emit it (and
    // not, say, 'avc1' or 'av1c' by typo). The token 'av01' also appears
    // in `ftyp.compatible_brands`, so there are exactly two occurrences in
    // the file: one in ftyp, one as the sample entry inside stsd.
    assert!(
        find_fourcc(&out, b"av01").is_some(),
        "AV1 sample entry must use 'av01' fourcc per AV1-ISOBMFF"
    );
    assert_eq!(
        count_fourcc(&out, b"av01"),
        2,
        "expected 'av01' once in ftyp.compatible_brands and once as the stsd sample entry; \
         got {}",
        count_fourcc(&out, b"av01")
    );
    // Also assert the sample entry one specifically lives inside stsd.
    let stsd_pos = find_fourcc(&out, b"stsd").expect("stsd missing");
    let stsd_size = u32::from_be_bytes([
        out[stsd_pos - 4],
        out[stsd_pos - 3],
        out[stsd_pos - 2],
        out[stsd_pos - 1],
    ]) as usize;
    let stsd_end = stsd_pos - 4 + stsd_size;
    let av01_in_stsd = out[stsd_pos..stsd_end]
        .windows(4)
        .position(|w| w == b"av01")
        .map(|rel| stsd_pos + rel)
        .expect("av01 sample entry must live inside stsd");
    assert!(
        av01_in_stsd > stsd_pos && av01_in_stsd < stsd_end,
        "av01 must be the sample entry inside stsd"
    );
    // av1C config record always sits inside av01 — sanity-check it is there
    // (its presence alone proves the av01 entry decoded the OBU stream
    // correctly during build).
    assert!(
        find_fourcc(&out, b"av1C").is_some(),
        "av1C decoder config record should be nested inside av01"
    );
}

#[test]
fn output_includes_colr_nclx_atom_with_default_bt709() {
    let out = build_video_only(30, 30.0, 256);
    let colr_pos = find_fourcc(&out, b"colr")
        .expect("colr atom missing — Apple silently assumes BT.709 limited otherwise");
    // Layout: size(4) | 'colr' | colour_type[4] | u16 cp | u16 tc | u16 mc | packed range byte
    assert_eq!(
        &out[colr_pos + 4..colr_pos + 8],
        b"nclx",
        "colour_type must be 'nclx' for video-distribution color signalling"
    );
    let cp = u16::from_be_bytes([out[colr_pos + 8], out[colr_pos + 9]]);
    let tc = u16::from_be_bytes([out[colr_pos + 10], out[colr_pos + 11]]);
    let mc = u16::from_be_bytes([out[colr_pos + 12], out[colr_pos + 13]]);
    assert_eq!(cp, 1, "default colour_primaries=1 (BT.709)");
    assert_eq!(tc, 1, "default transfer_characteristics=1 (BT.709)");
    assert_eq!(mc, 1, "default matrix_coefficients=1 (BT.709)");
    let range_byte = out[colr_pos + 14];
    assert_eq!(
        range_byte & 0x80,
        0x00,
        "default full_range_flag=0 (limited)"
    );
    // Reserved low 7 bits must be zero per ISO 23001-8.
    assert_eq!(
        range_byte & 0x7F,
        0x00,
        "colr nclx reserved bits must be zero"
    );
}

#[test]
fn faststart_moov_precedes_mdat() {
    // Apple's player accepts moov-after-mdat when seeking, but +faststart
    // is required for HLS / progressive playback, and the project's CLAUDE.md
    // states it's already done — guard against regression.
    let out = build_video_only(30, 30.0, 256);
    let moov_pos = find_fourcc(&out, b"moov").expect("moov missing");
    let mdat_pos = find_fourcc(&out, b"mdat").expect("mdat missing");
    assert!(
        moov_pos < mdat_pos,
        "moov ({}) must precede mdat ({}) for faststart / progressive playback",
        moov_pos,
        mdat_pos
    );
}

// ---- mdat 64-bit largesize (#9) ------------------------------------------

#[test]
fn mdat_largesize_header_layout_is_correct() {
    // Force largesize on a small payload to exercise the bit layout
    // without producing a 4 GiB file. Production callers leave
    // force_largesize_mdat off; the natural threshold is u32::MAX - 8.
    let mut muxer = Av1Mp4Muxer::new(320, 240, 30.0).expect("muxer");
    muxer.force_largesize_mdat_for_test();
    muxer
        .add_packet(EncodedPacket {
            data: minimal_av1_first_packet(),
            pts: 0,
            is_keyframe: true,
        })
        .expect("first");
    for i in 1..15 {
        muxer
            .add_packet(EncodedPacket {
                data: opaque_packet(128),
                pts: i,
                is_keyframe: false,
            })
            .expect("packet");
    }
    let out = muxer.finalize().expect("finalize");
    let mdat_pos = find_fourcc(&out, b"mdat").expect("mdat missing");
    // Largesize mdat layout per ISO/IEC 14496-12 §4.2:
    //   [pos-4..pos]   size = 0x00000001 (sentinel)
    //   [pos..pos+4]   type = 'mdat'
    //   [pos+4..pos+12] largesize u64be = total box length (header + payload)
    let size_field = u32::from_be_bytes([
        out[mdat_pos - 4],
        out[mdat_pos - 3],
        out[mdat_pos - 2],
        out[mdat_pos - 1],
    ]);
    assert_eq!(
        size_field, 1,
        "largesize sentinel: size field must be 0x00000001 to signal 64-bit length"
    );
    let largesize = u64::from_be_bytes([
        out[mdat_pos + 4],
        out[mdat_pos + 5],
        out[mdat_pos + 6],
        out[mdat_pos + 7],
        out[mdat_pos + 8],
        out[mdat_pos + 9],
        out[mdat_pos + 10],
        out[mdat_pos + 11],
    ]);
    // Payload bytes: first packet (1 + 1 + 5 = 7) + 14 * 128 = 7 + 1792 = 1799.
    // largesize = 16 (header) + 1799 (payload) = 1815.
    let expected = 16 + 7 + 14 * 128;
    assert_eq!(
        largesize,
        expected,
        "largesize must equal header(16) + payload({}); got {}",
        expected - 16,
        largesize
    );
}

#[test]
fn mdat_largesize_offsets_account_for_16_byte_header() {
    // When largesize is in use the chunk offsets in stco/co64 must point
    // past the 16-byte header (8 size + 'mdat' + 8 largesize), not the
    // 8-byte short-header start. If we got that wrong every chunk would
    // start 8 bytes early and the player would read into the largesize
    // field instead of the first sample.
    let mut muxer = Av1Mp4Muxer::new(320, 240, 30.0).expect("muxer");
    muxer.force_largesize_mdat_for_test();
    muxer
        .add_packet(EncodedPacket {
            data: minimal_av1_first_packet(),
            pts: 0,
            is_keyframe: true,
        })
        .expect("first");
    for i in 1..30 {
        muxer
            .add_packet(EncodedPacket {
                data: opaque_packet(64),
                pts: i,
                is_keyframe: false,
            })
            .expect("packet");
    }
    let out = muxer.finalize().expect("finalize");
    let mdat_pos = find_fourcc(&out, b"mdat").expect("mdat present");
    // Payload starts at mdat_pos + 4 (type) + 8 (largesize) = mdat_pos + 12.
    // mdat_pos itself is offset of the type field, which is at file offset
    // (mdat_pos - 4) + 4 = mdat_pos, so payload offset is mdat_pos + 12.
    let expected_first_sample_offset = (mdat_pos as u64) + 12;

    // Find stco — this is a small file so use_co64 is off.
    let stco_pos = find_fourcc(&out, b"stco").expect("stco present");
    let count = u32::from_be_bytes([
        out[stco_pos + 8],
        out[stco_pos + 9],
        out[stco_pos + 10],
        out[stco_pos + 11],
    ]);
    assert!(count >= 1, "expected at least one stco entry");
    let first_offset = u32::from_be_bytes([
        out[stco_pos + 12],
        out[stco_pos + 13],
        out[stco_pos + 14],
        out[stco_pos + 15],
    ]) as u64;
    assert_eq!(
        first_offset, expected_first_sample_offset,
        "first chunk offset must point past the 16-byte largesize header: \
         expected {}, got {}",
        expected_first_sample_offset, first_offset
    );
}

#[test]
fn mdat_short_header_used_for_small_payloads() {
    // Without the test override, small outputs must still use the 8-byte
    // form (size + type) — switching to largesize on every output would
    // waste 8 bytes and confuse fragmented players.
    let out = build_video_only(15, 30.0, 100);
    let mdat_pos = find_fourcc(&out, b"mdat").expect("mdat missing");
    let size_field = u32::from_be_bytes([
        out[mdat_pos - 4],
        out[mdat_pos - 3],
        out[mdat_pos - 2],
        out[mdat_pos - 1],
    ]);
    // Short-header: size carries the actual length; largesize sentinel
    // would be exactly 1.
    assert_ne!(
        size_field, 1,
        "small payload should not use largesize sentinel; size field was 1"
    );
    assert!(
        size_field > 8,
        "short-header size must include header (8 bytes) plus payload; got {}",
        size_field
    );
}

// ---- Squad-20: HDR atoms (mdcv + clli) end-to-end through Av1Mp4Muxer ----

/// Build a video-only output with a populated HDR10 ColorMetadata
/// (BT.2020 primaries, PQ transfer, mastering display, content light
/// level). Used to assert mdcv/clli round-trip through the muxer's
/// public surface (`set_color_metadata`).
fn build_video_only_hdr10(packets: u32, fps: f64, packet_size: usize) -> Bytes {
    use codec::frame::{ColorMetadata, ContentLightLevel, MasteringDisplay, TransferFn};
    let mut muxer = Av1Mp4Muxer::new(640, 480, fps).expect("muxer");
    muxer.set_color_metadata(ColorMetadata {
        transfer: TransferFn::St2084,
        matrix_coefficients: 9, // BT.2020 NCL
        colour_primaries: 9,    // BT.2020
        full_range: false,
        mastering_display: Some(MasteringDisplay {
            primaries_r_x: 35400,
            primaries_r_y: 14600,
            primaries_g_x: 8500,
            primaries_g_y: 39850,
            primaries_b_x: 6550,
            primaries_b_y: 2300,
            white_point_x: 15635,
            white_point_y: 16450,
            max_luminance: 10_000_000,
            min_luminance: 1,
        }),
        content_light_level: Some(ContentLightLevel {
            max_cll: 1000,
            max_fall: 400,
        }),
    });
    muxer
        .add_packet(EncodedPacket {
            data: minimal_av1_first_packet(),
            pts: 0,
            is_keyframe: true,
        })
        .expect("first packet");
    for i in 1..packets {
        muxer
            .add_packet(EncodedPacket {
                data: opaque_packet(packet_size),
                pts: i as u64,
                is_keyframe: false,
            })
            .expect("packet");
    }
    muxer.finalize().expect("finalize")
}

#[test]
fn output_includes_mdcv_atom_when_hdr_metadata_set() {
    let out = build_video_only_hdr10(30, 30.0, 256);
    let mdcv_pos = find_fourcc(&out, b"mdcv")
        .expect("mdcv atom missing — HDR mastering display lost in mux output");
    // size sits at mdcv_pos - 4; box-type 'mdcv' at mdcv_pos..mdcv_pos+4
    let size = u32::from_be_bytes([
        out[mdcv_pos - 4],
        out[mdcv_pos - 3],
        out[mdcv_pos - 2],
        out[mdcv_pos - 1],
    ]) as usize;
    assert_eq!(
        size, 32,
        "mdcv must be exactly 32 bytes (8 hdr + 24 payload)"
    );
    // Spot-check a few fields against the BT.2020 / D65 / 1000-nit canonical:
    let primaries_r_x = u16::from_be_bytes([out[mdcv_pos + 4], out[mdcv_pos + 5]]);
    assert_eq!(primaries_r_x, 35400, "BT.2020 R primary x");
    let max_lum_off = mdcv_pos + 4 + 16; // type + 8 u16
    let max_lum = u32::from_be_bytes([
        out[max_lum_off],
        out[max_lum_off + 1],
        out[max_lum_off + 2],
        out[max_lum_off + 3],
    ]);
    assert_eq!(
        max_lum, 10_000_000,
        "max_luminance = 1000 cd/m² in 0.0001 steps"
    );
}

#[test]
fn output_includes_clli_atom_when_hdr_metadata_set() {
    let out = build_video_only_hdr10(30, 30.0, 256);
    let clli_pos =
        find_fourcc(&out, b"clli").expect("clli atom missing — HDR CLL lost in mux output");
    let size = u32::from_be_bytes([
        out[clli_pos - 4],
        out[clli_pos - 3],
        out[clli_pos - 2],
        out[clli_pos - 1],
    ]) as usize;
    assert_eq!(
        size, 12,
        "clli must be exactly 12 bytes (8 hdr + 4 payload)"
    );
    let max_cll = u16::from_be_bytes([out[clli_pos + 4], out[clli_pos + 5]]);
    let max_fall = u16::from_be_bytes([out[clli_pos + 6], out[clli_pos + 7]]);
    assert_eq!(max_cll, 1000, "MaxCLL");
    assert_eq!(max_fall, 400, "MaxFALL");
}

#[test]
fn hdr_atoms_absent_for_default_sdr_metadata() {
    // Default ColorMetadata has mastering_display=None and
    // content_light_level=None. The SDR output (default test fixture) must
    // not emit mdcv or clli — they would mislead HDR-aware players into
    // applying PQ/HLG tone mapping to BT.709 SDR pixels.
    let out = build_video_only(30, 30.0, 256);
    assert!(
        find_fourcc(&out, b"mdcv").is_none(),
        "SDR output must not emit mdcv atom (would corrupt SDR rendering)"
    );
    assert!(
        find_fourcc(&out, b"clli").is_none(),
        "SDR output must not emit clli atom"
    );
}

#[test]
fn hdr_atoms_nest_inside_av01_after_colr() {
    // Spec order is colr → mdcv → clli inside the av01 sample entry.
    // Verify the byte positions follow that order in a real muxed file.
    let out = build_video_only_hdr10(30, 30.0, 256);
    let av01_pos = find_fourcc(&out, b"av01")
        .filter(|&p| {
            // The 4cc 'av01' appears in ftyp.compatible_brands too — the
            // sample-entry one lives inside stsd. Walk to the second
            // occurrence (the first is in ftyp).
            let stsd = find_fourcc(&out, b"stsd").expect("stsd missing");
            p > stsd
        })
        .or_else(|| {
            // Fall back: search past any earlier 'av01' (in ftyp brands).
            let stsd = find_fourcc(&out, b"stsd").expect("stsd missing");
            out[stsd..]
                .windows(4)
                .position(|w| w == b"av01")
                .map(|rel| stsd + rel)
        })
        .expect("av01 sample entry inside stsd");
    let colr_pos = find_fourcc(&out, b"colr").expect("colr missing");
    let mdcv_pos = find_fourcc(&out, b"mdcv").expect("mdcv missing");
    let clli_pos = find_fourcc(&out, b"clli").expect("clli missing");
    assert!(
        av01_pos < colr_pos,
        "av01 ({}) must precede colr ({})",
        av01_pos,
        colr_pos
    );
    assert!(
        colr_pos < mdcv_pos,
        "colr ({}) must precede mdcv ({})",
        colr_pos,
        mdcv_pos
    );
    assert!(
        mdcv_pos < clli_pos,
        "mdcv ({}) must precede clli ({})",
        mdcv_pos,
        clli_pos
    );
}
