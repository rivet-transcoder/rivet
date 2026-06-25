//! Integration tests for MP4 avcC → Annex-B PPS prepend (#67/#68).
//!
//! These verify the demux wiring for the bug Squad-14 fixed: the old
//! `prepend on sample_idx==1` heuristic mishandled ExoPlayer open-GOP
//! MP4s where sample 0 is `SPS + non-IDR slice`. The new tracker
//! prepends on the first IRAP that is missing parameter sets.
//!
//! Two test layers:
//!   1. Synthetic MP4 we build by hand — runs everywhere, covers the
//!      box-tree-walk + `length_prefixed_to_annexb_tracked` glue.
//!   2. Real `exoplayer_h264_main_720p.mp4` if present — gracefully
//!      skipped when the test_media corpus isn't downloaded.

use std::path::Path;

use container::demux;

// === Synthetic MP4 builder ===
//
// ISOBMFF is recursive boxes: `[u32 size BE][fourcc 4][payload...]`.
// We ship the smallest set of boxes needed for `mp4-rust` to find the
// avc1 sample entry, the avcC config, and one mdat sample.

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

/// Build an avcC config record per ISO/IEC 14496-15 §5.3.3.1.
fn build_avcc(sps: &[u8], pps: &[u8], length_size_minus_one: u8) -> Vec<u8> {
    let mut out = Vec::new();
    out.push(0x01); // configurationVersion
    out.push(0x42); // AVCProfileIndication = 66 (Baseline) — value irrelevant for the demux test
    out.push(0x00); // profile_compatibility
    out.push(0x1E); // AVCLevelIndication = 3.0
    out.push(0xFC | (length_size_minus_one & 0x03));
    out.push(0xE1); // reserved(3)=7|num_sps=1
    out.extend_from_slice(&(sps.len() as u16).to_be_bytes());
    out.extend_from_slice(sps);
    out.push(0x01); // num_pps
    out.extend_from_slice(&(pps.len() as u16).to_be_bytes());
    out.extend_from_slice(pps);
    out
}

/// Build a length-prefixed sample (always 4-byte length here).
fn lp4_sample(nalus: &[&[u8]]) -> Vec<u8> {
    let mut s = Vec::new();
    for n in nalus {
        s.extend_from_slice(&(n.len() as u32).to_be_bytes());
        s.extend_from_slice(n);
    }
    s
}

/// Build a minimal MP4 with one video track (avc1 sample entry) holding
/// `samples` mdat payloads. `avcc` is embedded under avc1.
///
/// This mimics ExoPlayer's open-GOP layout closely enough that the
/// `mp4-rust` reader walks the trak, finds the avc1 sample entry, reads
/// stsc/stco/stsz, and yields each sample bytewise to our demuxer.
fn build_synthetic_mp4(width: u16, height: u16, avcc: &[u8], samples: &[Vec<u8>]) -> Vec<u8> {
    // ftyp
    let ftyp = box_(b"ftyp", &{
        let mut p = Vec::new();
        p.extend_from_slice(b"isom"); // major_brand
        p.extend_from_slice(&0u32.to_be_bytes()); // minor_version
        p.extend_from_slice(b"isom"); // compatible_brands
        p.extend_from_slice(b"mp41");
        p.extend_from_slice(b"avc1");
        p
    });

    // mvhd: timescale=1000, duration=1000 (1s of clip)
    let mvhd = full_box_(b"mvhd", 0, 0, &{
        let mut p = Vec::new();
        p.extend_from_slice(&0u32.to_be_bytes()); // creation_time
        p.extend_from_slice(&0u32.to_be_bytes()); // modification_time
        p.extend_from_slice(&1000u32.to_be_bytes()); // timescale
        p.extend_from_slice(&1000u32.to_be_bytes()); // duration
        p.extend_from_slice(&0x00010000u32.to_be_bytes()); // rate 1.0
        p.extend_from_slice(&0x0100u16.to_be_bytes()); // volume 1.0
        p.extend_from_slice(&0u16.to_be_bytes()); // reserved
        p.extend_from_slice(&[0u8; 8]); // reserved 2x u32
        // Unity matrix
        p.extend_from_slice(&0x00010000u32.to_be_bytes());
        p.extend_from_slice(&0u32.to_be_bytes());
        p.extend_from_slice(&0u32.to_be_bytes());
        p.extend_from_slice(&0u32.to_be_bytes());
        p.extend_from_slice(&0x00010000u32.to_be_bytes());
        p.extend_from_slice(&0u32.to_be_bytes());
        p.extend_from_slice(&0u32.to_be_bytes());
        p.extend_from_slice(&0u32.to_be_bytes());
        p.extend_from_slice(&0x40000000u32.to_be_bytes());
        p.extend_from_slice(&[0u8; 24]); // pre_defined
        p.extend_from_slice(&2u32.to_be_bytes()); // next_track_ID
        p
    });

    // tkhd: track_id=1
    let tkhd = full_box_(b"tkhd", 0, 0x000007, &{
        // flags: track_enabled | track_in_movie | track_in_preview
        let mut p = Vec::new();
        p.extend_from_slice(&0u32.to_be_bytes()); // creation_time
        p.extend_from_slice(&0u32.to_be_bytes()); // modification_time
        p.extend_from_slice(&1u32.to_be_bytes()); // track_ID
        p.extend_from_slice(&0u32.to_be_bytes()); // reserved
        p.extend_from_slice(&1000u32.to_be_bytes()); // duration
        p.extend_from_slice(&[0u8; 8]); // reserved
        p.extend_from_slice(&0u16.to_be_bytes()); // layer
        p.extend_from_slice(&0u16.to_be_bytes()); // alternate_group
        p.extend_from_slice(&0u16.to_be_bytes()); // volume
        p.extend_from_slice(&0u16.to_be_bytes()); // reserved
        // Unity matrix
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

    // mdhd: timescale=1000, duration=1000, lang=und
    let mdhd = full_box_(b"mdhd", 0, 0, &{
        let mut p = Vec::new();
        p.extend_from_slice(&0u32.to_be_bytes());
        p.extend_from_slice(&0u32.to_be_bytes());
        p.extend_from_slice(&1000u32.to_be_bytes());
        p.extend_from_slice(&1000u32.to_be_bytes());
        p.extend_from_slice(&0x55C4u16.to_be_bytes()); // 'und'
        p.extend_from_slice(&0u16.to_be_bytes());
        p
    });

    // hdlr: vide
    let hdlr = full_box_(b"hdlr", 0, 0, &{
        let mut p = Vec::new();
        p.extend_from_slice(&0u32.to_be_bytes()); // pre_defined
        p.extend_from_slice(b"vide");
        p.extend_from_slice(&[0u8; 12]); // reserved
        p.push(0); // empty name (null-terminated)
        p
    });

    // vmhd
    let vmhd = full_box_(b"vmhd", 0, 0x000001, &{
        let mut p = Vec::new();
        p.extend_from_slice(&0u16.to_be_bytes()); // graphicsmode
        p.extend_from_slice(&[0u8; 6]); // opcolor
        p
    });

    // dref → url with self-contained flag
    let url = full_box_(b"url ", 0, 0x000001, &[]);
    let dref = full_box_(b"dref", 0, 0, &{
        let mut p = Vec::new();
        p.extend_from_slice(&1u32.to_be_bytes()); // entry_count
        p.extend_from_slice(&url);
        p
    });
    let dinf = box_(b"dinf", &dref);

    // stsd → avc1 sample entry
    let avc1 = {
        let mut p = Vec::new();
        // SampleEntry header: 6 reserved + 2 data_ref_index
        p.extend_from_slice(&[0u8; 6]);
        p.extend_from_slice(&1u16.to_be_bytes());
        // VisualSampleEntry: 2 pre_defined + 2 reserved + 12 pre_defined3
        p.extend_from_slice(&0u16.to_be_bytes());
        p.extend_from_slice(&0u16.to_be_bytes());
        p.extend_from_slice(&[0u8; 12]);
        // width, height
        p.extend_from_slice(&width.to_be_bytes());
        p.extend_from_slice(&height.to_be_bytes());
        // horizresolution, vertresolution = 72.0 (16.16 fixed)
        p.extend_from_slice(&0x00480000u32.to_be_bytes());
        p.extend_from_slice(&0x00480000u32.to_be_bytes());
        p.extend_from_slice(&0u32.to_be_bytes()); // reserved
        p.extend_from_slice(&1u16.to_be_bytes()); // frame_count
        // 32-byte compressorname (1-byte length + 31 bytes)
        p.extend_from_slice(&[0u8; 32]);
        p.extend_from_slice(&0x0018u16.to_be_bytes()); // depth = 24
        p.extend_from_slice(&0xFFFFu16.to_be_bytes()); // pre_defined
        // child boxes: avcC
        let avcc_box = box_(b"avcC", avcc);
        p.extend_from_slice(&avcc_box);
        box_(b"avc1", &p)
    };
    let stsd = full_box_(b"stsd", 0, 0, &{
        let mut p = Vec::new();
        p.extend_from_slice(&1u32.to_be_bytes()); // entry_count
        p.extend_from_slice(&avc1);
        p
    });

    // stts: 1 entry, sample_count=N, sample_delta=1000/N
    let per_sample_dur = 1000u32 / (samples.len() as u32).max(1);
    let stts = full_box_(b"stts", 0, 0, &{
        let mut p = Vec::new();
        p.extend_from_slice(&1u32.to_be_bytes()); // entry_count
        p.extend_from_slice(&(samples.len() as u32).to_be_bytes());
        p.extend_from_slice(&per_sample_dur.to_be_bytes());
        p
    });

    // stsc: every chunk has the same number of samples (1 here for simplicity)
    let stsc = full_box_(b"stsc", 0, 0, &{
        let mut p = Vec::new();
        p.extend_from_slice(&1u32.to_be_bytes()); // entry_count
        p.extend_from_slice(&1u32.to_be_bytes()); // first_chunk
        p.extend_from_slice(&1u32.to_be_bytes()); // samples_per_chunk
        p.extend_from_slice(&1u32.to_be_bytes()); // sample_description_index
        p
    });

    // stsz: variable-size samples
    let stsz = full_box_(b"stsz", 0, 0, &{
        let mut p = Vec::new();
        p.extend_from_slice(&0u32.to_be_bytes()); // sample_size = 0 (use entries)
        p.extend_from_slice(&(samples.len() as u32).to_be_bytes());
        for s in samples {
            p.extend_from_slice(&(s.len() as u32).to_be_bytes());
        }
        p
    });

    // Build moov with placeholder stco offsets to discover its size, then
    // rebuild with real offsets. Box sizes are content-length-determined,
    // so a placeholder of zero produces the same byte length as a real
    // offset (both u32). The two-pass approach lets us compute mdat's
    // payload start without parsing.
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

    // Layout: ftyp | moov | mdat. mdat payload starts at:
    //   ftyp.len() + moov.len() + 8 (mdat box header: 4 size + 4 fourcc).
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
        "moov v1 and v2 must be same size — offset bytes are u32 in both"
    );

    let mdat_payload: Vec<u8> = samples.iter().flatten().copied().collect();
    let mdat = box_(b"mdat", &mdat_payload);

    let mut out = Vec::new();
    out.extend_from_slice(&ftyp);
    out.extend_from_slice(&moov_v2);
    out.extend_from_slice(&mdat);
    out
}

// === Synthetic MP4 tests (hermetic, no test_media required) ===

/// Locate a NAL unit (4-byte start-code framed Annex-B) in a buffer.
/// Returns the index of the first byte of the NAL header (after the
/// 4-byte 0x00000001). Both 3-byte and 4-byte start codes are accepted.
fn find_nal_after_start_code(buf: &[u8], nal_first_byte: u8) -> Option<usize> {
    let mut i = 0;
    while i + 4 < buf.len() {
        let is_4_sc = buf[i] == 0 && buf[i + 1] == 0 && buf[i + 2] == 0 && buf[i + 3] == 1;
        let is_3_sc = buf[i] == 0 && buf[i + 1] == 0 && buf[i + 2] == 1;
        if is_4_sc {
            if buf[i + 4] == nal_first_byte {
                return Some(i + 4);
            }
            i += 4;
        } else if is_3_sc {
            if buf[i + 3] == nal_first_byte {
                return Some(i + 3);
            }
            i += 3;
        } else {
            i += 1;
        }
    }
    None
}

/// Walk an Annex-B buffer and collect every NAL header byte (the byte
/// immediately following the start code).
fn collect_nal_first_bytes(buf: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut i = 0;
    while i + 4 < buf.len() {
        let is_4_sc = buf[i] == 0 && buf[i + 1] == 0 && buf[i + 2] == 0 && buf[i + 3] == 1;
        let is_3_sc = buf[i] == 0 && buf[i + 1] == 0 && buf[i + 2] == 1;
        if is_4_sc {
            out.push(buf[i + 4]);
            i += 4;
        } else if is_3_sc {
            out.push(buf[i + 3]);
            i += 3;
        } else {
            i += 1;
        }
    }
    out
}

/// Synthetic ExoPlayer-shape MP4: avcC has SPS+PPS, sample 0 has
/// `SPS + non-IDR slice` inline, sample 1 is the first IDR. After
/// demux we expect:
///   - sample 0 Annex-B has SPS but NO PPS (no IRAP yet → no prepend)
///   - sample 1 Annex-B has PPS prepended (first IRAP), but NOT SPS
///     (already emitted inline upstream)
#[test]
fn mp4_exoplayer_sps_only_then_idr_pps_prepended_on_irap() {
    // SPS NAL header byte: 0x67 (forbidden_zero=0, nal_ref_idc=3, type=7)
    let sps: Vec<u8> = vec![0x67, 0x42, 0x00, 0x1E, 0xAA, 0xBB];
    // PPS NAL header byte: 0x68 (type=8)
    let pps: Vec<u8> = vec![0x68, 0xCE, 0x3C, 0x80];
    // P-slice NAL header byte: 0x41 (type=1 — non-IDR slice, ref_idc=2)
    let p_slice: Vec<u8> = vec![0x41, 0x9A, 0x00];
    // IDR NAL header byte: 0x65 (type=5, ref_idc=3)
    let idr: Vec<u8> = vec![0x65, 0x88, 0x84, 0x10];

    let avcc = build_avcc(&sps, &pps, /* lengthSizeMinusOne = */ 3);
    let samples = vec![
        lp4_sample(&[&sps, &p_slice]), // sample 0: SPS + non-IDR
        lp4_sample(&[&idr]),           // sample 1: first IDR
    ];

    let data = build_synthetic_mp4(320, 240, &avcc, &samples);
    let demuxed = demux::demux(&data).expect("demux synthetic ExoPlayer-shape MP4");
    assert_eq!(demuxed.codec, "h264");
    assert_eq!(demuxed.samples.len(), 2);

    // Sample 0: contains SPS + slice; PPS must NOT have been prepended
    // because there is no IRAP in this sample.
    let s0 = &demuxed.samples[0];
    assert!(
        find_nal_after_start_code(s0, 0x67).is_some(),
        "sample 0 must still contain its inline SPS"
    );
    assert!(
        find_nal_after_start_code(s0, 0x68).is_none(),
        "sample 0 must NOT contain PPS (no IRAP yet — old buggy code prepended here)"
    );

    // Sample 1: IDR — PPS gets prepended; SPS does not (already emitted
    // upstream).
    let s1 = &demuxed.samples[1];
    let nals = collect_nal_first_bytes(s1);
    let pps_count = nals.iter().filter(|b| **b == 0x68).count();
    let sps_count = nals.iter().filter(|b| **b == 0x67).count();
    let idr_count = nals.iter().filter(|b| **b == 0x65).count();
    assert_eq!(
        pps_count, 1,
        "PPS must be prepended exactly once on the first IRAP"
    );
    assert_eq!(
        sps_count, 0,
        "SPS must not be re-emitted (already inline at sample 0)"
    );
    assert_eq!(idr_count, 1, "IDR must appear exactly once");

    // Order in sample 1: PPS appears before IDR.
    let pps_pos = find_nal_after_start_code(s1, 0x68).unwrap();
    let idr_pos = find_nal_after_start_code(s1, 0x65).unwrap();
    assert!(
        pps_pos < idr_pos,
        "PPS must come before IDR in the prepended output"
    );
}

/// Synthetic well-formed MP4: avcC has SPS+PPS, sample 0 IS the first
/// IDR with no inline parameter sets (typical FFmpeg output). Demux
/// must prepend SPS+PPS on sample 0.
#[test]
fn mp4_avcc_only_first_sample_idr_prepends_both() {
    let sps: Vec<u8> = vec![0x67, 0x42, 0x00, 0x1E, 0xAA];
    let pps: Vec<u8> = vec![0x68, 0xCE, 0x3C, 0x80];
    let idr: Vec<u8> = vec![0x65, 0x88, 0x84, 0x00];

    let avcc = build_avcc(&sps, &pps, 3);
    let samples = vec![lp4_sample(&[&idr])];

    let data = build_synthetic_mp4(320, 240, &avcc, &samples);
    let demuxed = demux::demux(&data).expect("demux");
    assert_eq!(demuxed.samples.len(), 1);
    let s0 = &demuxed.samples[0];

    let sps_pos = find_nal_after_start_code(s0, 0x67).expect("SPS prepended");
    let pps_pos = find_nal_after_start_code(s0, 0x68).expect("PPS prepended");
    let idr_pos = find_nal_after_start_code(s0, 0x65).expect("IDR present");
    assert!(sps_pos < pps_pos);
    assert!(pps_pos < idr_pos);
}

/// Synthetic Jellyfin-like MP4: avcC has SPS+PPS *and* sample 0 has
/// `SPS + PPS + IDR` inline. The demuxer must NOT duplicate them.
#[test]
fn mp4_inline_sps_pps_idr_no_duplication() {
    let sps: Vec<u8> = vec![0x67, 0x42, 0x00, 0x1E, 0xAA];
    let pps: Vec<u8> = vec![0x68, 0xCE, 0x3C, 0x80];
    let idr: Vec<u8> = vec![0x65, 0x88, 0x84, 0x00];

    let avcc = build_avcc(&sps, &pps, 3);
    let samples = vec![lp4_sample(&[&sps, &pps, &idr])];

    let data = build_synthetic_mp4(320, 240, &avcc, &samples);
    let demuxed = demux::demux(&data).expect("demux");
    let s0 = &demuxed.samples[0];

    let nals = collect_nal_first_bytes(s0);
    let sps_count = nals.iter().filter(|b| **b == 0x67).count();
    let pps_count = nals.iter().filter(|b| **b == 0x68).count();
    assert_eq!(sps_count, 1, "SPS must appear exactly once");
    assert_eq!(pps_count, 1, "PPS must appear exactly once");
}

// === Real-file test (gracefully skipped when test_media is absent) ===

fn test_media(name: &str) -> Option<Vec<u8>> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("test_media")
        .join(name);
    std::fs::read(&path).ok()
}

/// Real ExoPlayer file (#67/#68 reproducer). When present, asserts that
/// the demuxer produces a stream where the first IRAP has both SPS and
/// PPS available before it (either inline in earlier samples or
/// prepended). This is the demux-layer half of the fix; the full decode
/// path is gated by openh264 err 16 (Squad-16 territory).
#[test]
fn mp4_exoplayer_h264_main_720p_first_irap_has_pps() {
    let Some(data) = test_media("exoplayer_h264_main_720p.mp4") else {
        eprintln!("SKIP mp4_exoplayer_h264_main_720p_first_irap_has_pps: test_media not available");
        return;
    };
    let demuxed = match demux::demux(&data) {
        Ok(d) => d,
        Err(e) => {
            panic!("demux failed: {e}");
        }
    };
    assert_eq!(demuxed.codec, "h264");
    assert!(!demuxed.samples.is_empty());

    // Walk samples, classifying NALs. By the time we see the first IDR
    // (or first slice if there's no IDR — open GOP), both SPS (type 7)
    // and PPS (type 8) must have appeared at least once.
    let mut saw_sps = false;
    let mut saw_pps = false;
    let mut first_irap_idx: Option<usize> = None;
    for (i, s) in demuxed.samples.iter().enumerate() {
        let nals = collect_nal_first_bytes(s);
        for n in &nals {
            let t = n & 0x1F;
            if t == 7 {
                saw_sps = true;
            }
            if t == 8 {
                saw_pps = true;
            }
            if t == 5 && first_irap_idx.is_none() {
                first_irap_idx = Some(i);
            }
        }
        if first_irap_idx.is_some() {
            break;
        }
    }

    if let Some(idx) = first_irap_idx {
        assert!(
            saw_sps && saw_pps,
            "by the first IDR (sample {}), SPS and PPS must have been seen \
             — saw_sps={}, saw_pps={}",
            idx,
            saw_sps,
            saw_pps
        );
    } else {
        // Open GOP / no IDR — at minimum SPS+PPS must exist somewhere
        // in the prefix so the decoder has parameter sets to attempt
        // recovery on a CRA-equivalent picture.
        assert!(
            saw_sps && saw_pps,
            "ExoPlayer stream lacks both an IDR and the SPS+PPS prefix \
             — saw_sps={}, saw_pps={}",
            saw_sps,
            saw_pps
        );
    }
}
