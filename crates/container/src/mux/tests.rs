use crate::AudioInfo;
use crate::ac3_sync::{Ac3SyncInfo, Eac3SyncInfo};
use codec::frame::{ColorMetadata, VideoCodec};
use super::Av1Mp4Muxer;
use super::boxes::{BoxBuilder, build_ftyp, build_moov, build_moov_any};
use super::video_track::{
    build_av01, build_colr_nclx, build_mdcv, build_clli, transfer_to_h273,
};
use super::audio_track::{
    build_audio_stsd, build_chan_box, build_mp4a, build_opus_sample_entry,
    build_dops, build_dac3, build_dec3, build_ac3_sample_entry, build_ec3_sample_entry,
    dac3_body_from_sync, dec3_body_from_sync,
};
use super::sample_table::{build_stsc, build_stco, build_co64, compute_chunk_offsets};

#[test]
fn ftyp_starts_with_size_and_type() {
    let ftyp = build_ftyp(VideoCodec::Av1);
    let size = u32::from_be_bytes([ftyp[0], ftyp[1], ftyp[2], ftyp[3]]);
    assert_eq!(size as usize, ftyp.len());
    assert_eq!(&ftyp[4..8], b"ftyp");
}

#[test]
fn leb128_roundtrip() {
    let mut buf = Vec::new();
    super::boxes::write_leb128(&mut buf, 300);
    let (v, n) = super::boxes::read_leb128(&buf).unwrap();
    assert_eq!(v, 300);
    assert_eq!(n, buf.len());
}

#[test]
fn box_builder_sizes_correctly() {
    let mut b = BoxBuilder::new(b"test");
    b.u32(0xDEADBEEF);
    let out = b.finish();
    assert_eq!(out.len(), 12);
    assert_eq!(&out[4..8], b"test");
    assert_eq!(u32::from_be_bytes([out[0], out[1], out[2], out[3]]), 12);
}

// ---- stsc chunk-run tests --------------------------------------------

/// Parse a `stsc` box bytes → Vec<(first_chunk, samples_per_chunk, sdi)>.
fn parse_stsc_entries(stsc: &[u8]) -> Vec<(u32, u32, u32)> {
    assert_eq!(&stsc[4..8], b"stsc");
    // size(4) type(4) ver(1) flags(3) count(4)
    let count = u32::from_be_bytes([stsc[12], stsc[13], stsc[14], stsc[15]]) as usize;
    let mut out = Vec::with_capacity(count);
    let mut p = 16usize;
    for _ in 0..count {
        let fc = u32::from_be_bytes([stsc[p], stsc[p + 1], stsc[p + 2], stsc[p + 3]]);
        let spc = u32::from_be_bytes([stsc[p + 4], stsc[p + 5], stsc[p + 6], stsc[p + 7]]);
        let sdi = u32::from_be_bytes([stsc[p + 8], stsc[p + 9], stsc[p + 10], stsc[p + 11]]);
        out.push((fc, spc, sdi));
        p += 12;
    }
    out
}

#[test]
fn mux_stsc_emits_multiple_chunk_runs() {
    // 120 samples at spc=24 → 5 full chunks of 24, no remainder.
    let stsc = build_stsc(120, 24);
    let entries = parse_stsc_entries(&stsc);
    assert_eq!(entries, vec![(1, 24, 1)]);
}

#[test]
fn mux_stsc_last_chunk_under_spc_emits_tail_entry() {
    // 121 samples at spc=24 → 5 full chunks + 1 tail of 1.
    let stsc = build_stsc(121, 24);
    let entries = parse_stsc_entries(&stsc);
    assert_eq!(entries, vec![(1, 24, 1), (6, 1, 1)]);
}

#[test]
fn mux_stsc_all_under_spc_single_entry() {
    // 10 samples at spc=24 → one partial chunk.
    let stsc = build_stsc(10, 24);
    let entries = parse_stsc_entries(&stsc);
    assert_eq!(entries, vec![(1, 10, 1)]);
}

// ---- chunk offset computation ----------------------------------------

#[test]
fn compute_chunk_offsets_walks_sample_sizes() {
    let sizes = vec![100u32, 200, 300, 400, 500, 600, 700];
    let offs = compute_chunk_offsets(1000, &sizes, 3);
    // chunks: [0..3]=1000, [3..6]=1000+600=1600, [6..7]=1600+1500=3100
    assert_eq!(offs, vec![1000, 1600, 3100]);
}

#[test]
fn compute_chunk_offsets_single_chunk() {
    let sizes = vec![10u32; 5];
    let offs = compute_chunk_offsets(42, &sizes, 120);
    assert_eq!(offs, vec![42]);
}

// ---- stco / co64 ------------------------------------------------------

#[test]
fn build_stco_emits_32bit_offsets() {
    let offs = vec![8u64, 1_000_000, u32::MAX as u64];
    let box_bytes = build_stco(&offs);
    assert_eq!(&box_bytes[4..8], b"stco");
    let count =
        u32::from_be_bytes([box_bytes[12], box_bytes[13], box_bytes[14], box_bytes[15]]);
    assert_eq!(count, 3);
    // 3 × 4 = 12 entry bytes. Header: 4 size + 4 type + 1 ver + 3 flags + 4 count = 16.
    assert_eq!(box_bytes.len(), 16 + 12);
    let last = u32::from_be_bytes([box_bytes[24], box_bytes[25], box_bytes[26], box_bytes[27]]);
    assert_eq!(last, u32::MAX);
}

#[test]
fn build_co64_emits_64bit_offsets() {
    let big = (u32::MAX as u64) + 100;
    let offs = vec![8u64, big, big + 1_000_000];
    let box_bytes = build_co64(&offs);
    assert_eq!(&box_bytes[4..8], b"co64");
    let count =
        u32::from_be_bytes([box_bytes[12], box_bytes[13], box_bytes[14], box_bytes[15]]);
    assert_eq!(count, 3);
    // 3 × 8 = 24 entry bytes. Header = 16.
    assert_eq!(box_bytes.len(), 16 + 24);
    // Second entry: bytes 24..32.
    let got = u64::from_be_bytes([
        box_bytes[24],
        box_bytes[25],
        box_bytes[26],
        box_bytes[27],
        box_bytes[28],
        box_bytes[29],
        box_bytes[30],
        box_bytes[31],
    ]);
    assert_eq!(got, big);
}

#[test]
fn build_co64_offsets_are_monotonic_and_be() {
    // Craft a descending payload input to guard against accidental
    // little-endian or re-sort bugs.
    let offs: Vec<u64> = (0..5)
        .map(|i| 10_000_000_000u64 + i as u64 * 4096)
        .collect();
    let box_bytes = build_co64(&offs);
    let mut prev = 0u64;
    for i in 0..5 {
        let p = 16 + i * 8;
        let v = u64::from_be_bytes([
            box_bytes[p],
            box_bytes[p + 1],
            box_bytes[p + 2],
            box_bytes[p + 3],
            box_bytes[p + 4],
            box_bytes[p + 5],
            box_bytes[p + 6],
            box_bytes[p + 7],
        ]);
        assert!(v > prev, "offsets not monotonic: {v} after {prev}");
        prev = v;
    }
}

// ---- moov-level stco vs co64 -----------------------------------------

/// Find a 4-cc occurrence in a byte slice. Used to assert presence of
/// `co64`/`stco` in built moov blobs. Returns None if absent.
fn find_fourcc(data: &[u8], tag: &[u8; 4]) -> Option<usize> {
    data.windows(4).position(|w| w == tag)
}

#[test]
fn moov_with_use_co64_true_emits_co64_not_stco() {
    let sample_sizes = vec![1000u32; 120];
    // Offsets span past u32::MAX — representative of a 5 GiB file.
    let chunk_offsets: Vec<u64> = (0..5)
        .map(|i| (u32::MAX as u64) + i * 1_000_000_000)
        .collect();
    // Minimal config_obus — content is opaque to stbl layout.
    let config_obus = vec![0x0Au8, 0x03, 0x00, 0x00, 0x00];
    let moov = build_moov(
        1920,
        1080,
        90_000,
        120 * 3750,
        3750,
        &sample_sizes,
        &[],
        &config_obus,
        &chunk_offsets,
        24,
        true,
    );
    assert!(find_fourcc(&moov, b"co64").is_some(), "co64 box missing");
    // NB: must check for standalone `stco` not a substring — `stco` can
    // appear in payload or other labels. Use exact 4-byte box-type match.
    assert!(
        find_fourcc(&moov, b"stco").is_none(),
        "stco present when co64 chosen"
    );
}

#[test]
fn moov_with_use_co64_false_emits_stco_not_co64() {
    let sample_sizes = vec![1000u32; 120];
    let chunk_offsets: Vec<u64> = (0..5).map(|i| 1000 + i * 24_000).collect();
    let config_obus = vec![0x0Au8, 0x03, 0x00, 0x00, 0x00];
    let moov = build_moov(
        1920,
        1080,
        90_000,
        120 * 3750,
        3750,
        &sample_sizes,
        &[],
        &config_obus,
        &chunk_offsets,
        24,
        false,
    );
    assert!(find_fourcc(&moov, b"stco").is_some(), "stco box missing");
    assert!(
        find_fourcc(&moov, b"co64").is_none(),
        "co64 present when stco chosen"
    );
}

// ---- Apple-compat: ftyp brands ---------------------------------------

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

// ---- Apple-compat: colr nclx atom ------------------------------------

/// Find every occurrence of the 4-byte tag (used for assertions where
/// the tag may legitimately appear inside payload too).
fn count_fourcc_occurrences(data: &[u8], tag: &[u8; 4]) -> usize {
    data.windows(4).filter(|w| *w == tag).count()
}

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
        transfer: codec::frame::TransferFn::St2084,
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
        transfer: codec::frame::TransferFn::Bt709,
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

// ---- mdat 64-bit largesize -------------------------------------------

/// transfer_to_h273 should round-trip through the H.273 codes the
/// pipeline knows about. The Bt709 enum variant collapses 4 H.273
/// codes (1, 6, 14, 15) — we always emit the canonical 1 on write.
#[test]
fn transfer_to_h273_emits_canonical_codes() {
    use codec::frame::TransferFn;
    assert_eq!(transfer_to_h273(TransferFn::Bt709), 1);
    assert_eq!(transfer_to_h273(TransferFn::Bt470Bg), 4);
    assert_eq!(transfer_to_h273(TransferFn::Linear), 8);
    assert_eq!(transfer_to_h273(TransferFn::St2084), 16);
    assert_eq!(transfer_to_h273(TransferFn::AribStdB67), 18);
    assert_eq!(transfer_to_h273(TransferFn::Unspecified), 2);
}

// ---- HDR atoms: mdcv (Mastering Display Color Volume) ----------------

/// HDR10-canonical mastering display values: BT.2020 primaries +
/// D65 white point + 1000 nits / 0.0001 nits luminance, all in the
/// HEVC SEI 137 / SMPTE ST 2086 spec-domain integer encoding.
///
/// Cross-references for the wire numbers (so future reviewers can
/// re-derive without chasing a spec PDF):
///   BT.2020 R primary  (0.708 , 0.292)  → (35400, 14600)
///   BT.2020 G primary  (0.170 , 0.797)  → ( 8500, 39850)
///   BT.2020 B primary  (0.131 , 0.046)  → ( 6550,  2300)
///   D65 white point    (0.3127, 0.3290) → (15635, 16450)
///   max luminance       1000 cd/m²      → 10_000_000  (0.0001 cd/m² steps)
///   min luminance       0.0001 cd/m²    →          1
fn hdr10_mastering_display() -> codec::frame::MasteringDisplay {
    codec::frame::MasteringDisplay {
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
    }
}

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
    let cll = codec::frame::ContentLightLevel {
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
        transfer: codec::frame::TransferFn::St2084,
        matrix_coefficients: 9,
        colour_primaries: 9,
        full_range: false,
        mastering_display: Some(hdr10_mastering_display()),
        content_light_level: Some(codec::frame::ContentLightLevel {
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

// ---- colr nclx HDR transfer-code coverage (Squad-18 verification) ----

/// PQ transfer (HDR10) is H.273 transfer_characteristics = 16. Apple
/// + browsers key off this code to apply the ST 2084 EOTF; emitting
/// 1 (BT.709) here would render HDR10 as washed-out SDR.
#[test]
fn colr_handles_pq_transfer_code_16() {
    let cm = ColorMetadata {
        transfer: codec::frame::TransferFn::St2084,
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
        transfer: codec::frame::TransferFn::AribStdB67,
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
        transfer: codec::frame::TransferFn::St2084,
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

// ---- Squad-23: Opus + dOps box layout (RFC 7845) ---------------------

/// Standard OpusHead body for stereo @ 48 kHz with PreSkip = 312
/// (the typical libopus encoder lookahead at 48 kHz). Output gain = 0,
/// ChannelMappingFamily = 0 (stereo).
///
/// Layout (post-magic body, 11 bytes; LE numeric fields per RFC 7845
/// §5.1):
///   [0]    Version=1
///   [1]    OutputChannelCount=2
///   [2..4] PreSkip=312 LE → 38 01
///   [4..8] InputSampleRate=48000 LE → 80 BB 00 00
///   [8..10] OutputGain=0 LE → 00 00
///   [10]   ChannelMappingFamily=0
fn opus_head_stereo_48k_preskip_312() -> Vec<u8> {
    let mut head = Vec::with_capacity(11);
    head.push(1u8); // Version
    head.push(2u8); // OutputChannelCount
    head.extend_from_slice(&312u16.to_le_bytes()); // PreSkip
    head.extend_from_slice(&48_000u32.to_le_bytes()); // InputSampleRate
    head.extend_from_slice(&0i16.to_le_bytes()); // OutputGain
    head.push(0u8); // ChannelMappingFamily
    head
}

fn opus_info_stereo_48k() -> AudioInfo {
    AudioInfo {
        codec: "opus".into(),
        sample_rate: 48_000,
        channels: 2,
        timescale: 48_000,
        asc_bytes: Vec::new(),
        codec_private: opus_head_stereo_48k_preskip_312(),
    }
}

/// `dOps` body layout per RFC 7845 §4.5: 11-byte minimum. Box wrapper
/// adds 8-byte ISOBMFF header → total 19 bytes for ChannelMappingFamily=0.
/// Numeric fields are big-endian (NOT the little-endian convention of
/// the OpusHead source bytes).
#[test]
fn dops_box_11_byte_payload_layout() {
    let info = opus_info_stereo_48k();
    let dops = build_dops(&info);
    assert_eq!(
        dops.len(),
        19,
        "dOps must be exactly 19 bytes (8 header + 11 payload)"
    );
    let size = u32::from_be_bytes([dops[0], dops[1], dops[2], dops[3]]) as usize;
    assert_eq!(size, dops.len(), "size field must equal box length");
    assert_eq!(
        &dops[4..8],
        b"dOps",
        "box type must be 'dOps' (capital O lowercase ps)"
    );
    // Body fields, all BE per §4.5.
    assert_eq!(dops[8], 0, "Version (RFC 7845 §4.5: MUST be 0)");
    assert_eq!(dops[9], 2, "OutputChannelCount = stereo");
    let pre_skip = u16::from_be_bytes([dops[10], dops[11]]);
    assert_eq!(pre_skip, 312, "PreSkip = 312 (BE)");
    let input_sample_rate = u32::from_be_bytes([dops[12], dops[13], dops[14], dops[15]]);
    assert_eq!(input_sample_rate, 48_000, "InputSampleRate = 48000 (BE)");
    let output_gain = i16::from_be_bytes([dops[16], dops[17]]);
    assert_eq!(output_gain, 0, "OutputGain = 0 (Q8 dB, BE)");
    assert_eq!(dops[18], 0, "ChannelMappingFamily = 0 (mono/stereo)");
}

/// The byte-order conversion between OpusHead (LE) and dOps (BE) is
/// the load-bearing piece — easy to mess up. PreSkip=312 in LE is
/// `38 01`; in BE it must come back out as `01 38`.
#[test]
fn dops_byte_order_flipped_from_opushead() {
    let info = opus_info_stereo_48k();
    // Sanity check the input is in LE.
    assert_eq!(
        info.codec_private[2..4],
        [0x38, 0x01],
        "OpusHead PreSkip must be LE"
    );
    let dops = build_dops(&info);
    // PreSkip in dOps body = bytes 10..12 of the box (after 8-byte header).
    assert_eq!(
        dops[10..12],
        [0x01, 0x38],
        "dOps PreSkip must be BE — got {:02X?}",
        &dops[10..12]
    );
}

/// `Opus` sample entry per RFC 7845 §4.4. Same generic AudioSampleEntry
/// preamble as `mp4a` (36 bytes including header) plus the dOps child.
/// Total = 36 + 19 = 55 bytes for the minimum-channel-count case.
/// 4-cc is `Opus` exactly (capital O).
#[test]
fn opus_sample_entry_size_and_fourcc() {
    let info = opus_info_stereo_48k();
    let entry = build_opus_sample_entry(&info);
    let size = u32::from_be_bytes([entry[0], entry[1], entry[2], entry[3]]) as usize;
    assert_eq!(size, entry.len(), "size field must equal box length");
    assert_eq!(&entry[4..8], b"Opus", "4-cc MUST be 'Opus' (capital O)");
    assert_ne!(&entry[4..8], b"opus", "lowercase 'opus' is non-conformant");
    // Total = 36 (sample entry preamble inc 8-byte header) + 19 (dOps) = 55.
    assert_eq!(
        entry.len(),
        55,
        "Opus sample entry should be 55 bytes for stereo + dOps minimum"
    );
}

/// AudioSampleEntry-level samplerate field inside `Opus` MUST be
/// 48000 << 16 — RFC 7845 §3 mandates 48 kHz internally; emitting
/// the source's nominal rate (e.g. 44100) would mismatch dOps and
/// confuse strict validators.
#[test]
fn opus_sample_entry_samplerate_is_48000_q16() {
    let info = AudioInfo {
        // Source nominal sample_rate is 44100, but the sample-entry
        // and mdhd MUST report 48000.
        sample_rate: 44_100,
        ..opus_info_stereo_48k()
    };
    let entry = build_opus_sample_entry(&info);
    // Layout offsets inside the sample entry (after the 8-byte box header):
    //   reserved[6]+data_ref(2)=8, reserved2(8)=16, channelcount(2)=18,
    //   sample_size(2)=20, pre_def(2)=22, reserved3(2)=24,
    //   samplerate u32 16.16 at +24..+28.
    // So box-relative offset 8 + 24 = 32.
    let sr_q16 = u32::from_be_bytes([entry[32], entry[33], entry[34], entry[35]]);
    assert_eq!(
        sr_q16,
        48_000u32 << 16,
        "samplerate field MUST be 48000<<16 (Q16); got 0x{:08X}",
        sr_q16
    );
}

/// `dOps` must nest inside the `Opus` sample entry. The build_audio_stsd
/// dispatcher routes Opus → build_opus_sample_entry → dOps child.
#[test]
fn dops_nests_inside_opus_sample_entry() {
    let info = opus_info_stereo_48k();
    let entry = build_opus_sample_entry(&info);
    let dops_pos = entry
        .windows(4)
        .position(|w| w == b"dOps")
        .expect("dOps child missing inside Opus sample entry");
    // dOps must come AFTER the 36-byte AudioSampleEntry preamble.
    assert!(
        dops_pos > 28,
        "dOps must come after the AudioSampleEntry preamble; got pos={}",
        dops_pos
    );
}

/// stsd dispatcher: AAC info → mp4a; Opus info → Opus. The dispatcher
/// must NEVER produce mp4a for Opus or Opus for AAC.
#[test]
fn stsd_dispatcher_routes_codec_to_correct_sample_entry() {
    let aac = AudioInfo {
        codec: "aac".into(),
        sample_rate: 44_100,
        channels: 2,
        timescale: 44_100,
        asc_bytes: vec![0x12, 0x10],
        codec_private: Vec::new(),
    };
    let stsd_aac = build_audio_stsd(&aac);
    assert!(
        stsd_aac.windows(4).any(|w| w == b"mp4a"),
        "AAC stsd must contain mp4a"
    );
    assert!(
        !stsd_aac.windows(4).any(|w| w == b"Opus"),
        "AAC stsd must NOT contain Opus"
    );
    assert!(
        stsd_aac.windows(4).any(|w| w == b"esds"),
        "AAC stsd must contain esds"
    );

    let opus = opus_info_stereo_48k();
    let stsd_opus = build_audio_stsd(&opus);
    assert!(
        stsd_opus.windows(4).any(|w| w == b"Opus"),
        "Opus stsd must contain Opus"
    );
    assert!(
        !stsd_opus.windows(4).any(|w| w == b"mp4a"),
        "Opus stsd must NOT contain mp4a"
    );
    assert!(
        stsd_opus.windows(4).any(|w| w == b"dOps"),
        "Opus stsd must contain dOps"
    );
    assert!(
        !stsd_opus.windows(4).any(|w| w == b"esds"),
        "Opus stsd must NOT contain esds"
    );
}

/// Negative output gain (-3 dB Q8 = -768) round-trips correctly through
/// the i16-as-u16 BE conversion.
#[test]
fn dops_handles_negative_output_gain() {
    let mut head = opus_head_stereo_48k_preskip_312();
    // OutputGain at offset 8..10. Set to -768 (i.e. -3 dB Q8).
    let gain: i16 = -768;
    head[8..10].copy_from_slice(&gain.to_le_bytes());
    let info = AudioInfo {
        codec_private: head,
        ..opus_info_stereo_48k()
    };
    let dops = build_dops(&info);
    let recovered = i16::from_be_bytes([dops[16], dops[17]]);
    assert_eq!(
        recovered, -768,
        "negative OutputGain must survive LE→BE roundtrip"
    );
}

/// PreSkip from the encoder's actual `OPUS_GET_LOOKAHEAD` (often
/// non-default like 156, 312, 480) must round-trip verbatim — we
/// don't normalize to 312.
#[test]
fn dops_preserves_arbitrary_preskip() {
    for &expected in &[0u16, 156, 312, 480, 1024, 65535] {
        let mut head = opus_head_stereo_48k_preskip_312();
        head[2..4].copy_from_slice(&expected.to_le_bytes());
        let info = AudioInfo {
            codec_private: head,
            ..opus_info_stereo_48k()
        };
        let dops = build_dops(&info);
        let got = u16::from_be_bytes([dops[10], dops[11]]);
        assert_eq!(got, expected, "PreSkip {} must survive LE→BE", expected);
    }
}

// ---- Squad-28: multichannel Opus dOps family=1 ----------------------

/// Build an OpusHead body for an N-channel surround layout per
/// RFC 7845 §5.1. Layout matches what Squad-28's
/// `OpusEncoder::extra_data()` emits and what an MKV/WebM
/// `CodecPrivate` carries verbatim. All multi-byte fields LE.
fn opus_head_surround(
    channels: u8,
    pre_skip: u16,
    input_sample_rate: u32,
    streams: u8,
    coupled: u8,
    mapping: &[u8],
) -> Vec<u8> {
    assert_eq!(mapping.len(), channels as usize);
    let mut h = Vec::with_capacity(11 + 2 + channels as usize);
    h.push(1u8); // Version
    h.push(channels);
    h.extend_from_slice(&pre_skip.to_le_bytes());
    h.extend_from_slice(&input_sample_rate.to_le_bytes());
    h.extend_from_slice(&0i16.to_le_bytes()); // OutputGain
    h.push(1u8); // ChannelMappingFamily=1
    h.push(streams);
    h.push(coupled);
    h.extend_from_slice(mapping);
    h
}

fn opus_info_5_1() -> AudioInfo {
    // RFC 7845 §5.1.1.2 5.1 layout: streams=4, coupled=2,
    // mapping = [0, 4, 1, 2, 3, 5]. PreSkip=312 (typical libopus
    // lookahead).
    let cp = opus_head_surround(6, 312, 48_000, 4, 2, &[0, 4, 1, 2, 3, 5]);
    AudioInfo {
        codec: "opus".into(),
        sample_rate: 48_000,
        channels: 6,
        timescale: 48_000,
        asc_bytes: Vec::new(),
        codec_private: cp,
    }
}

/// 5.1 dOps box payload = 11 + 2 + 6 = 19 bytes; with the 8-byte
/// box header the total is 27 bytes. All numeric fields BE inside
/// the box; the trailing channel-mapping bytes are u8 each so no
/// endianness conversion needed.
#[test]
fn dops_box_5_1_payload_is_19_bytes_total_27() {
    let info = opus_info_5_1();
    let dops = build_dops(&info);
    assert_eq!(
        dops.len(),
        27,
        "5.1 dOps box = 8 header + 19 payload = 27 bytes; got {}",
        dops.len()
    );
    let size = u32::from_be_bytes([dops[0], dops[1], dops[2], dops[3]]) as usize;
    assert_eq!(size, dops.len());
    assert_eq!(&dops[4..8], b"dOps");
    // Body
    assert_eq!(dops[8], 0, "Version");
    assert_eq!(dops[9], 6, "OutputChannelCount = 6 for 5.1");
    let pre_skip = u16::from_be_bytes([dops[10], dops[11]]);
    assert_eq!(pre_skip, 312);
    let isr = u32::from_be_bytes([dops[12], dops[13], dops[14], dops[15]]);
    assert_eq!(isr, 48_000);
    assert_eq!(i16::from_be_bytes([dops[16], dops[17]]), 0);
    assert_eq!(dops[18], 1, "ChannelMappingFamily = 1 for surround");
    assert_eq!(dops[19], 4, "StreamCount = 4 for 5.1");
    assert_eq!(dops[20], 2, "CoupledCount = 2 for 5.1");
    assert_eq!(
        &dops[21..27],
        &[0u8, 4, 1, 2, 3, 5][..],
        "ChannelMapping for 5.1"
    );
}

/// 7.1 layout: streams=5, coupled=3, mapping = [0, 6, 1, 2, 3, 4, 5, 7].
/// dOps box = 8 header + 11 preamble + 2 stream/coupled + 8 mapping = 29 bytes.
#[test]
fn dops_box_7_1_payload_is_21_bytes_total_29() {
    let cp = opus_head_surround(8, 312, 48_000, 5, 3, &[0, 6, 1, 2, 3, 4, 5, 7]);
    let info = AudioInfo {
        codec: "opus".into(),
        sample_rate: 48_000,
        channels: 8,
        timescale: 48_000,
        asc_bytes: Vec::new(),
        codec_private: cp,
    };
    let dops = build_dops(&info);
    assert_eq!(dops.len(), 29);
    assert_eq!(dops[18], 1, "Family = 1");
    assert_eq!(dops[19], 5, "StreamCount = 5 for 7.1");
    assert_eq!(dops[20], 3, "CoupledCount = 3 for 7.1");
    assert_eq!(&dops[21..29], &[0u8, 6, 1, 2, 3, 4, 5, 7][..]);
}

/// Hex-dump the 5.1 dOps box for the deliverables report.
#[test]
fn dops_box_5_1_hex_dump() {
    let info = opus_info_5_1();
    let dops = build_dops(&info);
    let hex: String = dops.iter().map(|b| format!("{b:02x} ")).collect();
    println!("5.1 dOps box hex (27 bytes total): {}", hex.trim_end());
}

/// `Opus` sample entry containing a family-1 dOps for 5.1. Total
/// size = 36 (sample-entry preamble) + 27 (5.1 dOps) = 63 bytes.
#[test]
fn opus_sample_entry_5_1_size_and_dops_nesting() {
    let info = opus_info_5_1();
    let entry = build_opus_sample_entry(&info);
    assert_eq!(
        entry.len(),
        36 + 27,
        "Opus sample entry for 5.1 = 36 + 27 = 63 bytes; got {}",
        entry.len()
    );
    // Sample-entry channel_count field is at offset 24 inside the
    // sample entry (after 8-byte box header + 6 reserved + 2 dri +
    // 8 reserved = 24).
    let entry_channels = u16::from_be_bytes([entry[24], entry[25]]);
    assert_eq!(
        entry_channels, 6,
        "channel_count in AudioSampleEntry must reflect 5.1"
    );
    // The dOps child should appear after the 36-byte preamble.
    assert!(entry[36..].windows(4).any(|w| w == b"dOps"));
    // Family byte inside the dOps child = entry[36 + 8 + 10] = entry[54].
    // (8-byte dOps box header + 11-byte preamble offset 10 = family).
    assert_eq!(
        entry[36 + 8 + 10],
        1,
        "dOps inside Opus sample entry must carry family=1 for 5.1"
    );
}

/// `with_audio()` family=1 validation: stream count + coupled +
/// mapping must all be sane. Each negative case below is rejected
/// loudly with a clear error message.
#[test]
fn with_audio_rejects_family_1_with_truncated_codec_private() {
    let mut muxer = Av1Mp4Muxer::new(640, 480, 30.0).unwrap();
    let mut info = opus_info_5_1();
    // Truncate so the channel-mapping table is missing.
    info.codec_private.truncate(13); // header + 2 stream/coupled, no mapping
    let err = match muxer.with_audio(info) {
        Ok(_) => panic!("truncated family=1 codec_private must reject"),
        Err(e) => e,
    };
    let msg = format!("{}", err);
    assert!(
        msg.contains("≥") && msg.contains("preamble"),
        "error message must explain the size requirement; got: {msg}"
    );
}

#[test]
fn with_audio_rejects_family_1_with_zero_streams() {
    let mut muxer = Av1Mp4Muxer::new(640, 480, 30.0).unwrap();
    let mut info = opus_info_5_1();
    // Zero out StreamCount byte (offset 11).
    info.codec_private[11] = 0;
    let r = muxer.with_audio(info);
    assert!(r.is_err(), "StreamCount = 0 must reject");
}

#[test]
fn with_audio_rejects_family_1_with_coupled_exceeding_streams() {
    let mut muxer = Av1Mp4Muxer::new(640, 480, 30.0).unwrap();
    let mut info = opus_info_5_1();
    // Make CoupledCount > StreamCount (offset 12 vs 11).
    info.codec_private[11] = 2;
    info.codec_private[12] = 5;
    let r = muxer.with_audio(info);
    assert!(r.is_err(), "CoupledCount > StreamCount must reject");
}

#[test]
fn with_audio_rejects_family_1_with_mapping_index_out_of_range() {
    let mut muxer = Av1Mp4Muxer::new(640, 480, 30.0).unwrap();
    let mut info = opus_info_5_1();
    // Streams=4, coupled=2 → max valid mapping index = 5. Set first
    // mapping byte to 99 to force the out-of-range branch.
    info.codec_private[13] = 99;
    let r = muxer.with_audio(info);
    assert!(r.is_err(), "ChannelMapping out-of-range must reject");
}

#[test]
fn with_audio_rejects_family_0_with_5_1_channels() {
    let mut muxer = Av1Mp4Muxer::new(640, 480, 30.0).unwrap();
    // Build a hand-crafted family-0 head but claim 6 channels.
    // Family 0 only supports 1..=2 channels per RFC 7845 §5.1.1.
    let mut head = Vec::with_capacity(11);
    head.push(1u8);
    head.push(6u8);
    head.extend_from_slice(&312u16.to_le_bytes());
    head.extend_from_slice(&48_000u32.to_le_bytes());
    head.extend_from_slice(&0i16.to_le_bytes());
    head.push(0u8); // family=0
    let info = AudioInfo {
        codec: "opus".into(),
        sample_rate: 48_000,
        channels: 6,
        timescale: 48_000,
        asc_bytes: Vec::new(),
        codec_private: head,
    };
    let r = muxer.with_audio(info);
    assert!(r.is_err(), "family=0 + 6 channels must reject");
}

#[test]
fn with_audio_accepts_5_1_opus() {
    let mut muxer = Av1Mp4Muxer::new(640, 480, 30.0).unwrap();
    let info = opus_info_5_1();
    muxer
        .with_audio(info)
        .expect("5.1 Opus with valid family=1 trailer must accept");
}

#[test]
fn with_audio_rejects_9_channel_opus() {
    let mut muxer = Av1Mp4Muxer::new(640, 480, 30.0).unwrap();
    // 9 channels has no defined family-1 layout.
    let mut head = Vec::with_capacity(11 + 2 + 9);
    head.push(1u8);
    head.push(9u8);
    head.extend_from_slice(&312u16.to_le_bytes());
    head.extend_from_slice(&48_000u32.to_le_bytes());
    head.extend_from_slice(&0i16.to_le_bytes());
    head.push(1u8); // family=1
    head.push(5);
    head.push(3);
    head.extend_from_slice(&[0u8, 1, 2, 3, 4, 5, 6, 7, 0]);
    let info = AudioInfo {
        codec: "opus".into(),
        sample_rate: 48_000,
        channels: 9,
        timescale: 48_000,
        asc_bytes: Vec::new(),
        codec_private: head,
    };
    let r = muxer.with_audio(info);
    assert!(
        r.is_err(),
        "9-channel Opus must reject (no family-1 layout above 8)"
    );
}

// ---- Squad-25: Apple `chan` (Channel Layout) box -----------------------

/// Mono / stereo: no `chan` box (Apple's default layouts are correct).
#[test]
fn chan_box_omitted_for_mono_and_stereo() {
    assert!(build_chan_box(1).is_none(), "mono should not emit chan");
    assert!(build_chan_box(2).is_none(), "stereo should not emit chan");
}

/// Unsupported channel counts return None — defence-in-depth (the
/// caller's `with_audio` gate already rejects them, so seeing 8/Atmos
/// here means a code path bypassed that gate).
#[test]
fn chan_box_omitted_for_unsupported_counts() {
    for &c in &[0u16, 3, 4, 5, 8, 9, 16] {
        assert!(
            build_chan_box(c).is_none(),
            "channels={c} must not emit chan"
        );
    }
}

/// 5.1 → kAudioChannelLayoutTag_MPEG_5_1_C = (114 << 16) | 6 = 0x00720006.
/// Body layout: tag u32 (4) | bitmap u32 (4) | num_descriptions u32 (4)
/// = 12 bytes. Total box = 8-byte header + 12-byte body = 20 bytes.
#[test]
fn chan_box_5_1_layout_and_size() {
    let chan = build_chan_box(6).expect("5.1 must emit chan");
    assert_eq!(
        chan.len(),
        20,
        "5.1 chan box must be 20 bytes (8 header + 12 body)"
    );
    let size = u32::from_be_bytes([chan[0], chan[1], chan[2], chan[3]]);
    assert_eq!(
        size as usize,
        chan.len(),
        "size field must equal box length"
    );
    assert_eq!(&chan[4..8], b"chan", "fourcc must be 'chan'");
    let tag = u32::from_be_bytes([chan[8], chan[9], chan[10], chan[11]]);
    assert_eq!(
        tag, 0x00720006u32,
        "5.1 tag must be kAudioChannelLayoutTag_MPEG_5_1_C = 0x00720006; got 0x{tag:08X}"
    );
    let bitmap = u32::from_be_bytes([chan[12], chan[13], chan[14], chan[15]]);
    assert_eq!(bitmap, 0, "mChannelBitmap must be 0 for tag form");
    let ndescs = u32::from_be_bytes([chan[16], chan[17], chan[18], chan[19]]);
    assert_eq!(
        ndescs, 0,
        "mNumberChannelDescriptions must be 0 for tag form"
    );
}

/// 7.1 → kAudioChannelLayoutTag_MPEG_7_1_C = (127 << 16) | 8 = 0x007F0008.
#[test]
fn chan_box_7_1_layout_and_size() {
    let chan = build_chan_box(7).expect("7.1 must emit chan");
    assert_eq!(chan.len(), 20);
    let tag = u32::from_be_bytes([chan[8], chan[9], chan[10], chan[11]]);
    assert_eq!(
        tag, 0x007F0008u32,
        "7.1 tag must be kAudioChannelLayoutTag_MPEG_7_1_C = 0x007F0008; got 0x{tag:08X}"
    );
}

/// `chan` nests inside the `mp4a` AudioSampleEntry (alongside `esds`)
/// per QuickTime File Format Spec. Multichannel mp4a should contain
/// both an esds AND a chan child.
#[test]
fn chan_nests_inside_mp4a_for_5_1() {
    // 5.1 ASC: AOT=2 SFI=3 chan=6 → 0x11 0xB0.
    let info = AudioInfo {
        codec: "aac".into(),
        sample_rate: 48_000,
        channels: 6,
        timescale: 48_000,
        asc_bytes: vec![0x11, 0xB0],
        codec_private: Vec::new(),
    };
    let mp4a = build_mp4a(&info);
    assert_eq!(&mp4a[4..8], b"mp4a", "outer box must be mp4a");
    let chan_pos = mp4a
        .windows(4)
        .position(|w| w == b"chan")
        .expect("multichannel mp4a must contain chan child");
    let esds_pos = mp4a
        .windows(4)
        .position(|w| w == b"esds")
        .expect("mp4a must always contain esds child");
    // chan should come AFTER esds (we append chan last in build_mp4a).
    assert!(
        chan_pos > esds_pos,
        "chan should come after esds in mp4a (esds @ {}, chan @ {})",
        esds_pos,
        chan_pos
    );
}

/// Stereo mp4a must NOT carry a `chan` box — Apple's default L+R
/// stereo layout is correct without one, and emitting a stereo `chan`
/// would just bloat the output.
#[test]
fn chan_absent_from_stereo_mp4a() {
    let info = AudioInfo {
        codec: "aac".into(),
        sample_rate: 48_000,
        channels: 2,
        timescale: 48_000,
        asc_bytes: vec![0x11, 0x90],
        codec_private: Vec::new(),
    };
    let mp4a = build_mp4a(&info);
    assert!(
        mp4a.windows(4).all(|w| w != b"chan"),
        "stereo mp4a must not contain a chan box"
    );
}

// ---- Squad-26: AC-3 + E-AC-3 mux box layout (ETSI TS 102 366 §F) ----

/// Canonical 5.1 384 kbps 48 kHz AC-3:
///   fscod=0, bsid=8, bsmod=0, acmod=7 (3/2), lfeon=1, bit_rate_code=14.
fn ac3_sync_5_1_384k_48k() -> Ac3SyncInfo {
    Ac3SyncInfo {
        fscod: 0,
        bit_rate_code: 14,
        bsid: 8,
        bsmod: 0,
        acmod: 7,
        lfeon: true,
    }
}

fn ac3_info_5_1_384k() -> AudioInfo {
    let body = dac3_body_from_sync(&ac3_sync_5_1_384k_48k());
    AudioInfo::ac3(48_000, 6, body.to_vec())
}

/// Vanilla 5.1 E-AC-3 single independent substream, 48 kHz, 384 kbps.
fn eac3_sync_5_1_48k() -> Eac3SyncInfo {
    Eac3SyncInfo {
        strmtyp: 0,
        substreamid: 0,
        // frmsiz arbitrary for box-layout tests; choose 191 → frame
        // size = 384 bytes which corresponds to 384 kbps @ 48 kHz / 1536
        // samples-per-frame.
        frmsiz: 191,
        fscod: 0,
        fscod2: 0,
        numblkscod: 3,
        acmod: 7,
        lfeon: true,
        bsid: 16,
        dialnorm: 0,
        bsmod: 0,
    }
}

fn eac3_info_5_1_384k() -> AudioInfo {
    // 384 kbps → data_rate field = 192 (the "kbps / 2" encoding).
    let body = dec3_body_from_sync(&eac3_sync_5_1_48k(), 192);
    AudioInfo::eac3(48_000, 6, body.to_vec())
}

/// `dac3` is exactly 11 bytes total (8-byte box header + 3-byte body).
/// Body field positions per ETSI TS 102 366 §F.4: fscod 2b | bsid 5b |
/// bsmod 3b | acmod 3b | lfeon 1b | bit_rate_code 5b | reserved 5b.
#[test]
fn dac3_box_3_byte_payload_layout() {
    let info = ac3_info_5_1_384k();
    let dac3 = build_dac3(&info);
    assert_eq!(dac3.len(), 11, "dac3 = 8-byte header + 3-byte body");
    let size = u32::from_be_bytes([dac3[0], dac3[1], dac3[2], dac3[3]]) as usize;
    assert_eq!(size, dac3.len(), "size field equals box length");
    assert_eq!(&dac3[4..8], b"dac3", "box type 'dac3'");
    // Body bit-extract (24 bits, MSB-first across 3 bytes 8..11).
    let raw = ((dac3[8] as u32) << 16) | ((dac3[9] as u32) << 8) | dac3[10] as u32;
    assert_eq!((raw >> 22) & 0x03, 0, "fscod = 0 (48 kHz)");
    assert_eq!((raw >> 17) & 0x1F, 8, "bsid = 8 (AC-3)");
    assert_eq!((raw >> 14) & 0x07, 0, "bsmod = 0");
    assert_eq!((raw >> 11) & 0x07, 7, "acmod = 7 (3/2 = 5.1 with LFE)");
    assert_eq!((raw >> 10) & 0x01, 1, "lfeon = 1");
    assert_eq!((raw >> 5) & 0x1F, 14, "bit_rate_code = 14 (= 384 kbps)");
    assert_eq!(raw & 0x1F, 0, "reserved 5 bits = 0");
}

/// `ac-3` AudioSampleEntry per ETSI TS 102 366 §F.2.
/// Total = 36-byte sample-entry preamble + 11-byte dac3 = 47 bytes.
/// 4cc is `ac-3` exactly (with the hyphen at byte index 6 = 0x2D).
#[test]
fn ac3_sample_entry_size_and_fourcc() {
    let info = ac3_info_5_1_384k();
    let entry = build_ac3_sample_entry(&info);
    let size = u32::from_be_bytes([entry[0], entry[1], entry[2], entry[3]]) as usize;
    assert_eq!(size, entry.len(), "size field equals box length");
    assert_eq!(&entry[4..8], b"ac-3", "4cc MUST be 'ac-3' (with hyphen)");
    // Reject the dehyphenated form
    assert_ne!(
        &entry[4..8],
        b"ac3\0",
        "4cc 'ac3' (3-char) is non-conformant"
    );
    assert_eq!(
        entry.len(),
        47,
        "ac-3 sample entry = 36 (preamble) + 11 (dac3)"
    );
    // dac3 must nest inside.
    let dac3_pos = entry
        .windows(4)
        .position(|w| w == b"dac3")
        .expect("dac3 child missing");
    assert!(
        dac3_pos > 28,
        "dac3 must come after AudioSampleEntry preamble"
    );
    // samplerate field at box-relative offset 8 + 24 = 32.
    let sr_q16 = u32::from_be_bytes([entry[32], entry[33], entry[34], entry[35]]);
    assert_eq!(sr_q16, 48_000u32 << 16, "samplerate = 48000 << 16 (Q16)");
}

/// `dec3` for a single independent substream (Squad-26's scope) is
/// 13 bytes total = 8-byte box header + 5-byte body (no dependent
/// substreams = no chan_loc tail). Body layout per ETSI TS 102 366
/// §F.6.
#[test]
fn dec3_box_5_byte_payload_layout() {
    let info = eac3_info_5_1_384k();
    let dec3 = build_dec3(&info);
    assert_eq!(dec3.len(), 13, "dec3 = 8-byte header + 5-byte body");
    let size = u32::from_be_bytes([dec3[0], dec3[1], dec3[2], dec3[3]]) as usize;
    assert_eq!(size, dec3.len(), "size field equals box length");
    assert_eq!(&dec3[4..8], b"dec3", "box type 'dec3'");
    // Body header: data_rate(13) + num_ind_sub-1(3) packed in bytes 8..10.
    let header = ((dec3[8] as u16) << 8) | dec3[9] as u16;
    let data_rate = (header >> 3) & 0x1FFF;
    assert_eq!(data_rate, 192, "data_rate = 192 (= 384 kbps / 2)");
    let num_ind_sub_minus_1 = header & 0x07;
    assert_eq!(num_ind_sub_minus_1, 0, "single substream → field = 0");
    // Per-independent-substream block: bits 16..40 (3 bytes 10..13).
    // Layout shifts within the 24-bit window:
    //   bit 23..22 fscod
    //   bit 21..17 bsid (=16)
    //   bit 16     reserved
    //   bit 15     asvc
    //   bit 14..12 bsmod
    //   bit 11..9  acmod
    //   bit 8      lfeon
    //   bit 7..5   reserved
    //   bit 4..1   num_dep_sub (=0)
    //   bit 0      reserved
    let sub = ((dec3[10] as u32) << 16) | ((dec3[11] as u32) << 8) | dec3[12] as u32;
    assert_eq!((sub >> 22) & 0x03, 0, "fscod = 0 (48 kHz)");
    assert_eq!((sub >> 17) & 0x1F, 16, "bsid = 16 (E-AC-3 marker)");
    assert_eq!((sub >> 12) & 0x07, 0, "bsmod = 0");
    assert_eq!((sub >> 9) & 0x07, 7, "acmod = 7 (3/2 = 5.1 with LFE)");
    assert_eq!((sub >> 8) & 0x01, 1, "lfeon = 1");
    assert_eq!((sub >> 1) & 0x0F, 0, "num_dep_sub = 0 (single substream)");
}

/// `ec-3` AudioSampleEntry per ETSI TS 102 366 §F.5.
/// Total = 36-byte sample-entry preamble + 13-byte dec3 = 49 bytes.
#[test]
fn ec3_sample_entry_size_and_fourcc() {
    let info = eac3_info_5_1_384k();
    let entry = build_ec3_sample_entry(&info);
    let size = u32::from_be_bytes([entry[0], entry[1], entry[2], entry[3]]) as usize;
    assert_eq!(size, entry.len(), "size field equals box length");
    assert_eq!(&entry[4..8], b"ec-3", "4cc MUST be 'ec-3' (with hyphen)");
    assert_eq!(
        entry.len(),
        49,
        "ec-3 sample entry = 36 (preamble) + 13 (dec3)"
    );
    let dec3_pos = entry
        .windows(4)
        .position(|w| w == b"dec3")
        .expect("dec3 child missing");
    assert!(
        dec3_pos > 28,
        "dec3 must come after AudioSampleEntry preamble"
    );
}

/// stsd dispatcher: ac3 info → ac-3 entry; eac3 info → ec-3 entry.
/// Must NOT cross-pollinate with mp4a / Opus.
#[test]
fn stsd_dispatcher_routes_ac3_eac3() {
    let stsd_ac3 = build_audio_stsd(&ac3_info_5_1_384k());
    assert!(
        stsd_ac3.windows(4).any(|w| w == b"ac-3"),
        "AC-3 stsd has 'ac-3'"
    );
    assert!(
        stsd_ac3.windows(4).any(|w| w == b"dac3"),
        "AC-3 stsd has 'dac3'"
    );
    assert!(
        !stsd_ac3.windows(4).any(|w| w == b"mp4a"),
        "AC-3 stsd MUST NOT have mp4a"
    );
    assert!(
        !stsd_ac3.windows(4).any(|w| w == b"Opus"),
        "AC-3 stsd MUST NOT have Opus"
    );
    assert!(
        !stsd_ac3.windows(4).any(|w| w == b"esds"),
        "AC-3 stsd MUST NOT have esds"
    );

    let stsd_eac3 = build_audio_stsd(&eac3_info_5_1_384k());
    assert!(
        stsd_eac3.windows(4).any(|w| w == b"ec-3"),
        "E-AC-3 stsd has 'ec-3'"
    );
    assert!(
        stsd_eac3.windows(4).any(|w| w == b"dec3"),
        "E-AC-3 stsd has 'dec3'"
    );
    assert!(
        !stsd_eac3.windows(4).any(|w| w == b"mp4a"),
        "E-AC-3 stsd MUST NOT have mp4a"
    );
    assert!(
        !stsd_eac3.windows(4).any(|w| w == b"esds"),
        "E-AC-3 stsd MUST NOT have esds"
    );
    assert!(
        !stsd_eac3.windows(4).any(|w| w == b"dac3"),
        "E-AC-3 stsd MUST NOT have dac3"
    );
}

/// `with_audio` must accept a 5.1 AC-3 info and reject obvious shape
/// errors (wrong dac3 body length, wrong sample rate).
#[test]
fn with_audio_accepts_ac3_5_1_and_rejects_bad_shape() {
    let mut muxer = Av1Mp4Muxer::new(320, 240, 30.0).unwrap();
    muxer
        .with_audio(ac3_info_5_1_384k())
        .expect("5.1 AC-3 must be accepted");

    // Wrong body length
    let mut muxer2 = Av1Mp4Muxer::new(320, 240, 30.0).unwrap();
    let mut bad = ac3_info_5_1_384k();
    bad.codec_private = vec![0u8; 2];
    let err = muxer2
        .with_audio(bad)
        .err()
        .expect("must reject 2-byte dac3");
    assert!(format!("{err:#}").contains("3 bytes"));

    // Wrong sample rate
    let mut muxer3 = Av1Mp4Muxer::new(320, 240, 30.0).unwrap();
    let bad_sr = AudioInfo {
        sample_rate: 22_050,
        timescale: 22_050,
        ..ac3_info_5_1_384k()
    };
    let err = muxer3
        .with_audio(bad_sr)
        .err()
        .expect("must reject 22050 for AC-3");
    assert!(format!("{err:#}").contains("32000"));
}

/// `with_audio` must accept a single-substream E-AC-3 info and reject
/// an under-sized dec3 body.
#[test]
fn with_audio_accepts_eac3_5_1_and_rejects_short_dec3() {
    let mut muxer = Av1Mp4Muxer::new(320, 240, 30.0).unwrap();
    muxer
        .with_audio(eac3_info_5_1_384k())
        .expect("5.1 E-AC-3 must be accepted");

    let mut muxer2 = Av1Mp4Muxer::new(320, 240, 30.0).unwrap();
    let mut bad = eac3_info_5_1_384k();
    bad.codec_private = vec![0u8; 4];
    let err = muxer2
        .with_audio(bad)
        .err()
        .expect("must reject short dec3");
    assert!(format!("{err:#}").contains("≥5"));
}

/// AC-3 / E-AC-3 channel count gate: must reject >6.
#[test]
fn with_audio_rejects_ac3_more_than_6_channels() {
    let mut muxer = Av1Mp4Muxer::new(320, 240, 30.0).unwrap();
    let bad = AudioInfo {
        channels: 8,
        ..ac3_info_5_1_384k()
    };
    let err = muxer.with_audio(bad).err().expect("must reject 8 channels");
    assert!(format!("{err:#}").contains("1..=6"));
}

/// Round-trip: parse a synthetic 5.1 AC-3 sync header → derive dac3
/// body → pack into an `ac-3` sample entry → walk the bytes back out
/// and recover fscod / acmod / lfeon / bit_rate_code unchanged.
#[test]
fn ac3_sync_to_dac3_to_sample_entry_roundtrip() {
    let sync = ac3_sync_5_1_384k_48k();
    let body = dac3_body_from_sync(&sync);
    let info = AudioInfo::ac3(48_000, 6, body.to_vec());
    let entry = build_ac3_sample_entry(&info);
    // Find dac3 box body (8-byte box header inside the entry then 3
    // body bytes).
    let dac3_pos = entry.windows(4).position(|w| w == b"dac3").unwrap();
    let dac3_body_start = dac3_pos + 4;
    let raw = ((entry[dac3_body_start] as u32) << 16)
        | ((entry[dac3_body_start + 1] as u32) << 8)
        | entry[dac3_body_start + 2] as u32;
    assert_eq!((raw >> 22) & 0x03, sync.fscod as u32);
    assert_eq!((raw >> 17) & 0x1F, sync.bsid as u32);
    assert_eq!((raw >> 11) & 0x07, sync.acmod as u32);
    assert_eq!((raw >> 10) & 0x01, sync.lfeon as u32);
    assert_eq!((raw >> 5) & 0x1F, sync.bit_rate_code as u32);
}
