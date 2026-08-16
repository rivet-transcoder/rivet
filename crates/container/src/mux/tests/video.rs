// Video-track tests: ftyp brands, av01 sample entry, colr/nclx,
// HDR atoms (mdcv + clli), and H.273 transfer-code coverage.
// 15 #[test] functions.

use frame::{ColorMetadata, VideoCodec};
use super::super::boxes::{build_ftyp, build_moov_any};
use super::super::video_track::{
    build_av01, build_colr_nclx, build_mdcv, build_clli, transfer_to_h273,
};
use super::{find_fourcc, count_fourcc_occurrences, hdr10_mastering_display};

// ---- Apple-compat: ftyp brands -------------------------------------------

/// AV1-ISOBMFF v1.3.0 §2.1 mandates `av01` in `compatible_brands`. Apple
/// QuickTime / iOS Safari additionally need a structural ISOBMFF brand
/// (`iso6` covers co64 / largesize from 14496-12 sixth edition). `mp42`
/// is conventional for AAC parsing rules.
#[test]
fn ftyp_lists_av01_and_iso6_and_mp42_brands() {
    let ftyp = build_ftyp(VideoCodec::Av1);
    // major_brand at offset 8..12 (after size + 'ftyp')
    assert_eq!(&ftyp[8..12], b"iso6", "major_brand should be iso6");
    // After major(4) + minor(4) the compatible_brands list runs to end.
    let compat = &ftyp[16..];
    let brands: Vec<&[u8]> = compat.chunks_exact(4).collect();
    assert!(
        brands.contains(&b"av01".as_ref()),
        "compatible_brands must list av01 per AV1-ISOBMFF §2.1; got {:?}",
        brands
    );
    assert!(
        brands.contains(&b"iso6".as_ref()),
        "compatible_brands must list iso6 (14496-12 v6 — covers co64/largesize)"
    );
    assert!(
        brands.contains(&b"mp42".as_ref()),
        "compatible_brands should list mp42 for AAC parsing rules"
    );
}

// ---- Apple-compat: colr nclx atom ----------------------------------------

#[test]
fn av01_sample_entry_includes_colr_nclx_box() {
    let cm = ColorMetadata::default();
    let sample_sizes = vec![100u32; 30];
    let chunk_offsets: Vec<u64> = vec![1000];
    let config_obus = vec![0x0Au8, 0x03, 0x00, 0x00, 0x00];
    let _ = (&sample_sizes, &chunk_offsets);
    let moov = build_av01(1920, 1080, &config_obus, &cm);
    let colr_pos = find_fourcc(&moov, b"colr").expect("colr atom missing");
    // Body layout: [pos-4..pos] = size, [pos..pos+4] = 'colr',
    // [pos+4..pos+8] = colour_type, then 6 bytes nclx fields.
    assert_eq!(
        &moov[colr_pos + 4..colr_pos + 8],
        b"nclx",
        "colour_type must be 'nclx' per ISO/IEC 23001-8"
    );
    // colour_primaries (u16 BE) at +8..+10
    let cp = u16::from_be_bytes([moov[colr_pos + 8], moov[colr_pos + 9]]);
    assert_eq!(cp, 1, "default BT.709 colour_primaries=1");
    // transfer_characteristics at +10..+12
    let tc = u16::from_be_bytes([moov[colr_pos + 10], moov[colr_pos + 11]]);
    assert_eq!(tc, 1, "default BT.709 transfer_characteristics=1");
    // matrix_coefficients at +12..+14
    let mc = u16::from_be_bytes([moov[colr_pos + 12], moov[colr_pos + 13]]);
    assert_eq!(mc, 1, "default BT.709 matrix_coefficients=1");
    // full_range_flag is the high bit of the byte at +14
    let fr = moov[colr_pos + 14];
    assert_eq!(fr & 0x80, 0x00, "default limited-range full_range_flag=0");
}

#[test]
fn colr_nclx_carries_hdr10_metadata() {
    // HDR10: BT.2020 NCL primaries (9), ST 2084 PQ transfer (16),
    // BT.2020 NCL matrix (9), limited range. This is the canonical
    // HDR10 nclx triple — Apple's player needs it to apply PQ tone
    // mapping correctly.
    let cm = ColorMetadata {
        transfer: frame::TransferFn::St2084,
        matrix_coefficients: 9,
        colour_primaries: 9,
        full_range: false,
        ..ColorMetadata::default()
    };
    let colr = build_colr_nclx(&cm);
    assert_eq!(&colr[4..8], b"colr");
    assert_eq!(&colr[8..12], b"nclx");
    let cp = u16::from_be_bytes([colr[12], colr[13]]);
    let tc = u16::from_be_bytes([colr[14], colr[15]]);
    let mc = u16::from_be_bytes([colr[16], colr[17]]);
    let fr = colr[18];
    assert_eq!(cp, 9, "BT.2020 NCL primaries");
    assert_eq!(tc, 16, "ST 2084 PQ transfer");
    assert_eq!(mc, 9, "BT.2020 NCL matrix");
    assert_eq!(fr & 0x80, 0x00, "HDR10 typically signals limited range");
}

#[test]
fn colr_nclx_full_range_sets_high_bit() {
    let cm = ColorMetadata {
        transfer: frame::TransferFn::Bt709,
        matrix_coefficients: 1,
        colour_primaries: 1,
        full_range: true,
        ..ColorMetadata::default()
    };
    let colr = build_colr_nclx(&cm);
    assert_eq!(colr[18] & 0x80, 0x80, "full_range high bit must be set");
    // Low 7 bits are reserved-zero per ISO 23001-8.
    assert_eq!(colr[18] & 0x7F, 0x00, "reserved bits must be zero");
}

#[test]
fn colr_nclx_box_size_matches_layout() {
    // Box: 4 size + 4 'colr' + 4 colour_type + 2 cp + 2 tc + 2 mc + 1 packed = 19 bytes.
    let colr = build_colr_nclx(&ColorMetadata::default());
    let size = u32::from_be_bytes([colr[0], colr[1], colr[2], colr[3]]) as usize;
    assert_eq!(
        size,
        colr.len(),
        "colr box size field must equal box length"
    );
    assert_eq!(size, 19, "colr nclx must be exactly 19 bytes");
}

/// Sanity: the `colr` atom must live inside the visual sample entry,
/// not float at the moov / trak / stbl level. Players look for it
/// nested inside `av01` (or `avc1`/`hvc1`) in `stsd`.
#[test]
fn colr_lives_inside_av01_sample_entry() {
    let cm = ColorMetadata::default();
    let sample_sizes = vec![100u32; 30];
    let chunk_offsets: Vec<u64> = vec![1000];
    let config_obus = vec![0x0Au8, 0x03, 0x00, 0x00, 0x00];
    let _ = (&sample_sizes, &chunk_offsets);
    let moov = build_av01(1920, 1080, &config_obus, &cm);
    let av01_pos = find_fourcc(&moov, b"av01").expect("av01 sample entry missing");
    let av01_size = u32::from_be_bytes([
        moov[av01_pos - 4],
        moov[av01_pos - 3],
        moov[av01_pos - 2],
        moov[av01_pos - 1],
    ]) as usize;
    let av01_end = av01_pos - 4 + av01_size;
    let colr_pos = find_fourcc(&moov, b"colr").expect("colr missing");
    assert!(
        colr_pos > av01_pos && colr_pos < av01_end,
        "colr must be nested inside av01 sample entry: av01@{}..{} colr@{}",
        av01_pos,
        av01_end,
        colr_pos
    );
    assert_eq!(
        count_fourcc_occurrences(&moov, b"colr"),
        1,
        "exactly one colr atom expected"
    );
}

// ---- mdat 64-bit largesize / transfer_to_h273 ----------------------------

/// transfer_to_h273 should round-trip through the H.273 codes the
/// pipeline knows about. The Bt709 enum variant collapses 4 H.273
/// codes (1, 6, 14, 15) — we always emit the canonical 1 on write.
#[test]
fn transfer_to_h273_emits_canonical_codes() {
    use frame::TransferFn;
    assert_eq!(transfer_to_h273(TransferFn::Bt709), 1);
    assert_eq!(transfer_to_h273(TransferFn::Bt470Bg), 4);
    assert_eq!(transfer_to_h273(TransferFn::Linear), 8);
    assert_eq!(transfer_to_h273(TransferFn::St2084), 16);
    assert_eq!(transfer_to_h273(TransferFn::AribStdB67), 18);
    assert_eq!(transfer_to_h273(TransferFn::Unspecified), 2);
}

// ---- HDR atoms: mdcv (Mastering Display Color Volume) --------------------

/// 24-byte payload + 8-byte header = 32 bytes. Bytes laid out big-endian.
/// Box-type is `'mdcv'` (NOT `'SmDm'`).
#[test]
fn mdcv_box_24_byte_payload_layout() {
    let md = hdr10_mastering_display();
    let mdcv = build_mdcv(&md);
    assert_eq!(
        mdcv.len(),
        32,
        "mdcv box must be exactly 32 bytes (8 header + 24 payload)"
    );
    let size = u32::from_be_bytes([mdcv[0], mdcv[1], mdcv[2], mdcv[3]]) as usize;
    assert_eq!(size, mdcv.len(), "size field must equal box length");
    assert_eq!(&mdcv[4..8], b"mdcv", "box type must be 'mdcv' (not 'SmDm')");
    // Body fields, all u16 BE except the trailing two u32s.
    let u16_at = |off: usize| u16::from_be_bytes([mdcv[off], mdcv[off + 1]]);
    let u32_at = |off: usize| {
        u32::from_be_bytes([mdcv[off], mdcv[off + 1], mdcv[off + 2], mdcv[off + 3]])
    };
    assert_eq!(u16_at(8), 35400, "primaries_r_x");
    assert_eq!(u16_at(10), 14600, "primaries_r_y");
    assert_eq!(u16_at(12), 8500, "primaries_g_x");
    assert_eq!(u16_at(14), 39850, "primaries_g_y");
    assert_eq!(u16_at(16), 6550, "primaries_b_x");
    assert_eq!(u16_at(18), 2300, "primaries_b_y");
    assert_eq!(u16_at(20), 15635, "white_point_x");
    assert_eq!(u16_at(22), 16450, "white_point_y");
    assert_eq!(u32_at(24), 10_000_000, "max_luminance (0.0001 cd/m² steps)");
    assert_eq!(u32_at(28), 1, "min_luminance");
}

/// 4-byte payload + 8-byte header = 12 bytes. Box-type is `'clli'`
/// (NOT `'CoLL'`).
#[test]
fn clli_box_4_byte_payload_layout() {
    let cll = frame::ContentLightLevel {
        max_cll: 1000,
        max_fall: 400,
    };
    let clli = build_clli(&cll);
    assert_eq!(
        clli.len(),
        12,
        "clli box must be exactly 12 bytes (8 header + 4 payload)"
    );
    let size = u32::from_be_bytes([clli[0], clli[1], clli[2], clli[3]]) as usize;
    assert_eq!(size, clli.len(), "size field must equal box length");
    assert_eq!(&clli[4..8], b"clli", "box type must be 'clli' (not 'CoLL')");
    let max_cll = u16::from_be_bytes([clli[8], clli[9]]);
    let max_fall = u16::from_be_bytes([clli[10], clli[11]]);
    assert_eq!(max_cll, 1000, "max_cll");
    assert_eq!(max_fall, 400, "max_fall");
}

/// When mastering_display is None, the av01 sample entry must omit
/// the `mdcv` box entirely. SDR sources should produce a moov with
/// no `mdcv` 4cc anywhere.
#[test]
fn mdcv_omitted_when_none() {
    let cm = ColorMetadata::default(); // None, None
    let sample_sizes = vec![100u32; 30];
    let chunk_offsets: Vec<u64> = vec![1000];
    let config_obus = vec![0x0Au8, 0x03, 0x00, 0x00, 0x00];
    let moov = build_moov_any(
        1920,
        1080,
        90_000,
        90_000,
        30 * 3000,
        30 * 3000,
        3000,
        &sample_sizes,
        &[],
        &config_obus,
        &chunk_offsets,
        30,
        None,
        &[],
        None,
        &[],
        false,
        &cm,
    );
    assert!(
        find_fourcc(&moov, b"mdcv").is_none(),
        "SDR (mastering_display=None) moov must NOT contain mdcv box"
    );
}

/// When content_light_level is None, the av01 sample entry must omit
/// the `clli` box entirely.
#[test]
fn clli_omitted_when_none() {
    let cm = ColorMetadata::default();
    let sample_sizes = vec![100u32; 30];
    let chunk_offsets: Vec<u64> = vec![1000];
    let config_obus = vec![0x0Au8, 0x03, 0x00, 0x00, 0x00];
    let moov = build_moov_any(
        1920,
        1080,
        90_000,
        90_000,
        30 * 3000,
        30 * 3000,
        3000,
        &sample_sizes,
        &[],
        &config_obus,
        &chunk_offsets,
        30,
        None,
        &[],
        None,
        &[],
        false,
        &cm,
    );
    assert!(
        find_fourcc(&moov, b"clli").is_none(),
        "SDR (content_light_level=None) moov must NOT contain clli box"
    );
}

/// AV1-ISOBMFF v1.3.0 §2.3.4 + §2.3.5 prescribe the order
/// `colr → mdcv → clli` inside the visual sample entry. Players
/// scan by 4cc so order is recommended-not-required, but matching
/// the spec keeps us defensible against strict validators
/// (mp4parser, GPAC's mp4box -info).
#[test]
fn av01_sample_entry_emits_mdcv_and_clli_in_order() {
    let cm = ColorMetadata {
        transfer: frame::TransferFn::St2084,
        matrix_coefficients: 9,
        colour_primaries: 9,
        full_range: false,
        mastering_display: Some(hdr10_mastering_display()),
        content_light_level: Some(frame::ContentLightLevel {
            max_cll: 1000,
            max_fall: 400,
        }),
    };
    let sample_sizes = vec![100u32; 30];
    let chunk_offsets: Vec<u64> = vec![1000];
    let config_obus = vec![0x0Au8, 0x03, 0x00, 0x00, 0x00];
    let _ = (&sample_sizes, &chunk_offsets);
    let moov = build_av01(1920, 1080, &config_obus, &cm);
    let av01_pos = find_fourcc(&moov, b"av01").expect("av01 sample entry missing");
    let av01_size = u32::from_be_bytes([
        moov[av01_pos - 4],
        moov[av01_pos - 3],
        moov[av01_pos - 2],
        moov[av01_pos - 1],
    ]) as usize;
    let av01_end = av01_pos - 4 + av01_size;
    let av01_body = &moov[av01_pos..av01_end];
    let colr_rel = av01_body
        .windows(4)
        .position(|w| w == b"colr")
        .expect("colr nested in av01");
    let mdcv_rel = av01_body
        .windows(4)
        .position(|w| w == b"mdcv")
        .expect("mdcv nested in av01");
    let clli_rel = av01_body
        .windows(4)
        .position(|w| w == b"clli")
        .expect("clli nested in av01");
    assert!(
        colr_rel < mdcv_rel,
        "colr ({}) must precede mdcv ({})",
        colr_rel,
        mdcv_rel
    );
    assert!(
        mdcv_rel < clli_rel,
        "mdcv ({}) must precede clli ({})",
        mdcv_rel,
        clli_rel
    );
    // Exactly one of each, all under av01.
    assert_eq!(
        count_fourcc_occurrences(&moov, b"mdcv"),
        1,
        "exactly one mdcv expected"
    );
    assert_eq!(
        count_fourcc_occurrences(&moov, b"clli"),
        1,
        "exactly one clli expected"
    );
}

// ---- colr nclx HDR transfer-code coverage (Squad-18 verification) --------

/// PQ transfer (HDR10) is H.273 transfer_characteristics = 16. Apple
/// + browsers key off this code to apply the ST 2084 EOTF; emitting
/// 1 (BT.709) here would render HDR10 as washed-out SDR.
#[test]
fn colr_handles_pq_transfer_code_16() {
    let cm = ColorMetadata {
        transfer: frame::TransferFn::St2084,
        matrix_coefficients: 9,
        colour_primaries: 9,
        full_range: false,
        ..ColorMetadata::default()
    };
    let colr = build_colr_nclx(&cm);
    let tc = u16::from_be_bytes([colr[14], colr[15]]);
    assert_eq!(tc, 16, "PQ transfer must encode as H.273 code 16");
}

/// HLG transfer is H.273 transfer_characteristics = 18. Same role as
/// PQ but for broadcast HDR; players that support HLG read 18 to
/// activate the ARIB STD-B67 OETF.
#[test]
fn colr_handles_hlg_transfer_code_18() {
    let cm = ColorMetadata {
        transfer: frame::TransferFn::AribStdB67,
        matrix_coefficients: 9,
        colour_primaries: 9,
        full_range: false,
        ..ColorMetadata::default()
    };
    let colr = build_colr_nclx(&cm);
    let tc = u16::from_be_bytes([colr[14], colr[15]]);
    assert_eq!(tc, 18, "HLG transfer must encode as H.273 code 18");
}

/// BT.2020 colour_primaries = 9, matrix_coefficients = 9 (NCL) or 10
/// (CL). Both must round-trip verbatim — the pipeline preserves the
/// raw u8 from the source SPS so the encode side can pick the right
/// matrix back out.
#[test]
fn colr_bt2020_primaries_matrix() {
    // NCL variant (most common — matrix_coefficients = 9)
    let cm_ncl = ColorMetadata {
        transfer: frame::TransferFn::St2084,
        matrix_coefficients: 9,
        colour_primaries: 9,
        full_range: false,
        ..ColorMetadata::default()
    };
    let colr_ncl = build_colr_nclx(&cm_ncl);
    let cp_ncl = u16::from_be_bytes([colr_ncl[12], colr_ncl[13]]);
    let mc_ncl = u16::from_be_bytes([colr_ncl[16], colr_ncl[17]]);
    assert_eq!(cp_ncl, 9, "BT.2020 colour_primaries must be 9");
    assert_eq!(mc_ncl, 9, "BT.2020 NCL matrix must be 9");

    // CL variant (matrix_coefficients = 10)
    let cm_cl = ColorMetadata {
        matrix_coefficients: 10,
        ..cm_ncl
    };
    let colr_cl = build_colr_nclx(&cm_cl);
    let mc_cl = u16::from_be_bytes([colr_cl[16], colr_cl[17]]);
    assert_eq!(
        mc_cl, 10,
        "BT.2020 CL matrix must be 10 (preserved verbatim)"
    );
}
