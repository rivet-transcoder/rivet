// Box primitives, sample-table builders, chunk-offset switching.
// 13 #[test] functions.

use frame::VideoCodec;
use super::super::boxes::{BoxBuilder, build_ftyp, build_moov, write_leb128, read_leb128};
use super::super::sample_table::{build_stsc, build_stco, build_co64, compute_chunk_offsets};
use super::find_fourcc;

// ---- ftyp / leb128 / BoxBuilder ------------------------------------------

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
    write_leb128(&mut buf, 300);
    let (v, n) = read_leb128(&buf).unwrap();
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

// ---- stsc chunk-run tests -------------------------------------------------

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

// ---- chunk offset computation ---------------------------------------------

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

// ---- stco / co64 ----------------------------------------------------------

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

// ---- moov-level stco vs co64 switching ------------------------------------

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
