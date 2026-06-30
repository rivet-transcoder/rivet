use codec::frame::{ColorMetadata, VideoCodec};

use crate::AudioInfo;

use super::*;

fn read_be_u32(buf: &[u8], pos: usize) -> u32 {
    u32::from_be_bytes(buf[pos..pos + 4].try_into().unwrap())
}

fn read_be_u64(buf: &[u8], pos: usize) -> u64 {
    u64::from_be_bytes(buf[pos..pos + 8].try_into().unwrap())
}

fn box_size_and_type(buf: &[u8]) -> (u32, &[u8]) {
    let size = read_be_u32(buf, 0);
    let kind = &buf[4..8];
    (size, kind)
}

#[test]
fn mfhd_layout_is_16_bytes_with_sequence_number() {
    let bytes = build_mfhd(42);
    assert_eq!(bytes.len(), 16);
    let (size, kind) = box_size_and_type(&bytes);
    assert_eq!(size, 16);
    assert_eq!(kind, b"mfhd");
    assert_eq!(bytes[8], 0); // version
    assert_eq!(&bytes[9..12], &[0, 0, 0]); // flags
    assert_eq!(read_be_u32(&bytes, 12), 42);
}

#[test]
fn tfhd_minimal_track_id_only_is_16_bytes() {
    let bytes = build_tfhd(1, None, None, None);
    // 8 (header) + 1 (version) + 3 (flags) + 4 (track_id) = 16.
    assert_eq!(bytes.len(), 16);
    let (size, kind) = box_size_and_type(&bytes);
    assert_eq!(size, 16);
    assert_eq!(kind, b"tfhd");
    // tf_flags should ONLY have default-base-is-moof (0x020000) set.
    let flag_bytes = [0u8, bytes[9], bytes[10], bytes[11]];
    let flags = u32::from_be_bytes(flag_bytes);
    assert_eq!(flags, 0x020000);
    assert_eq!(read_be_u32(&bytes, 12), 1);
}

#[test]
fn tfhd_with_default_flags_only_packs_correct_bits() {
    let bytes = build_tfhd(1, None, None, Some(SampleFlags::delta_frame().pack()));
    // 8 header + 1 version + 3 flags + 4 track_id + 4 default_sample_flags = 20.
    assert_eq!(bytes.len(), 20);
    let flag_bytes = [0u8, bytes[9], bytes[10], bytes[11]];
    let flags = u32::from_be_bytes(flag_bytes);
    // default-base-is-moof (0x020000) | default-sample-flags (0x000020).
    assert_eq!(flags, 0x020020);
    assert_eq!(read_be_u32(&bytes, 12), 1);
    assert_eq!(read_be_u32(&bytes, 16), SampleFlags::delta_frame().pack());
}

#[test]
fn tfhd_with_all_defaults_packs_in_spec_order() {
    let bytes = build_tfhd(1, Some(1024), Some(2048), Some(0x01010000));
    // 8 + 1 + 3 + 4 + 4 + 4 + 4 = 28.
    assert_eq!(bytes.len(), 28);
    let flag_bytes = [0u8, bytes[9], bytes[10], bytes[11]];
    let flags = u32::from_be_bytes(flag_bytes);
    // default-base-is-moof (0x020000) | dur (0x000008) | size (0x000010) | flags (0x000020).
    assert_eq!(flags, 0x020038);
    assert_eq!(read_be_u32(&bytes, 12), 1);
    assert_eq!(read_be_u32(&bytes, 16), 1024); // duration
    assert_eq!(read_be_u32(&bytes, 20), 2048); // size
    assert_eq!(read_be_u32(&bytes, 24), 0x01010000); // flags
}

#[test]
fn tfdt_v1_carries_u64_decode_time() {
    let bytes = build_tfdt(0x0123_4567_89AB_CDEF);
    // 8 header + 1 version + 3 flags + 8 decode_time = 20.
    assert_eq!(bytes.len(), 20);
    assert_eq!(box_size_and_type(&bytes), (20, b"tfdt".as_slice()));
    assert_eq!(bytes[8], 1); // version 1
    assert_eq!(read_be_u64(&bytes, 12), 0x0123_4567_89AB_CDEF);
}

#[test]
fn mehd_v1_carries_u64_fragment_duration() {
    let bytes = build_mehd(1_000_000);
    assert_eq!(bytes.len(), 20);
    assert_eq!(box_size_and_type(&bytes), (20, b"mehd".as_slice()));
    assert_eq!(bytes[8], 1);
    assert_eq!(read_be_u64(&bytes, 12), 1_000_000);
}

#[test]
fn trex_layout_is_32_bytes_with_track_id_and_flags() {
    let default_flags = SampleFlags::delta_frame().pack();
    let bytes = build_trex(2, default_flags);
    // 8 + 1 + 3 + 4 + 4 + 4 + 4 + 4 = 32.
    assert_eq!(bytes.len(), 32);
    assert_eq!(box_size_and_type(&bytes), (32, b"trex".as_slice()));
    assert_eq!(read_be_u32(&bytes, 12), 2); // track_id
    assert_eq!(read_be_u32(&bytes, 16), 1); // default_sample_description_index
    assert_eq!(read_be_u32(&bytes, 20), 0); // default_sample_duration
    assert_eq!(read_be_u32(&bytes, 24), 0); // default_sample_size
    assert_eq!(read_be_u32(&bytes, 28), default_flags);
}

#[test]
fn sample_flags_pack_distinguishes_sync_from_delta() {
    let sync = SampleFlags::keyframe().pack();
    let delta = SampleFlags::delta_frame().pack();
    assert_ne!(sync, delta);
    // Sync: depends_on=2 in bits 24-25, is_non_sync=0 in bit 16.
    assert_eq!(sync, 0x02_00_00_00);
    // Delta: depends_on=1, is_non_sync=1.
    assert_eq!(delta, 0x01_01_00_00);
}

#[test]
fn moof_video_one_keyframe_sample_round_trip() {
    let samples = vec![CmafSample {
        duration: 1500,
        size: 4096,
        flags: SampleFlags::keyframe(),
    }];
    let mut moof = build_moof_video(1, 1, 0, &samples);
    moof.patch_default_no_gap();

    let (size, kind) = box_size_and_type(&moof.bytes);
    assert_eq!(size as usize, moof.bytes.len());
    assert_eq!(kind, b"moof");

    // mfhd starts at offset 8 (after moof header).
    let (mfhd_size, mfhd_kind) = box_size_and_type(&moof.bytes[8..]);
    assert_eq!(mfhd_size, 16);
    assert_eq!(mfhd_kind, b"mfhd");
    assert_eq!(read_be_u32(&moof.bytes, 8 + 12), 1); // sequence_number

    // traf starts after mfhd.
    let traf_start = 8 + mfhd_size as usize;
    let (_, traf_kind) = box_size_and_type(&moof.bytes[traf_start..]);
    assert_eq!(traf_kind, b"traf");

    // The patched data_offset should equal moof.len() + 8.
    let patched = read_be_u32(&moof.bytes, moof.data_offset_pos);
    assert_eq!(patched as usize, moof.bytes.len() + 8);

    // The first_sample_flags slot in trun should equal the keyframe flags.
    // It sits 4 bytes after the data_offset field per the trun layout.
    let first_flags = read_be_u32(&moof.bytes, moof.data_offset_pos + 4);
    assert_eq!(first_flags, SampleFlags::keyframe().pack());
}

#[test]
fn moof_video_three_samples_records_per_sample_dur_and_size() {
    let samples = vec![
        CmafSample {
            duration: 1500,
            size: 4096,
            flags: SampleFlags::keyframe(),
        },
        CmafSample {
            duration: 1500,
            size: 1024,
            flags: SampleFlags::delta_frame(),
        },
        CmafSample {
            duration: 1500,
            size: 1024,
            flags: SampleFlags::delta_frame(),
        },
    ];
    let mut moof = build_moof_video(2, 1, 6000, &samples);
    moof.patch_default_no_gap();

    // Walk into trun and read sample_count.
    // moof header(8) + mfhd(16) + traf header(8) = 32.
    // Then tfhd: 8 + 1 + 3 + 4 + 4 = 20 bytes (track_id + default_flags).
    // Then tfdt v1: 20 bytes.
    // trun starts at 32 + 20 + 20 = 72.
    let trun_start = 8 + 16 + 8 + 20 + 20;
    let (_, trun_kind) = box_size_and_type(&moof.bytes[trun_start..]);
    assert_eq!(trun_kind, b"trun");
    let sample_count = read_be_u32(&moof.bytes, trun_start + 12);
    assert_eq!(sample_count, 3);

    // Per-sample table starts after data_offset(4) + first_sample_flags(4):
    //   trun_start + 8(header) + 1(version) + 3(flags) + 4(count) +
    //                4(data_offset) + 4(first_sample_flags) = trun_start + 24.
    let table_start = trun_start + 24;
    // sample 0: dur=1500, size=4096
    assert_eq!(read_be_u32(&moof.bytes, table_start), 1500);
    assert_eq!(read_be_u32(&moof.bytes, table_start + 4), 4096);
    // sample 1: dur=1500, size=1024
    assert_eq!(read_be_u32(&moof.bytes, table_start + 8), 1500);
    assert_eq!(read_be_u32(&moof.bytes, table_start + 12), 1024);
    // sample 2: dur=1500, size=1024
    assert_eq!(read_be_u32(&moof.bytes, table_start + 16), 1500);
    assert_eq!(read_be_u32(&moof.bytes, table_start + 20), 1024);
}

#[test]
fn moof_audio_does_not_emit_first_sample_flags() {
    let samples = vec![
        CmafSample {
            duration: 1024,
            size: 256,
            flags: SampleFlags::keyframe(),
        },
        CmafSample {
            duration: 1024,
            size: 256,
            flags: SampleFlags::keyframe(),
        },
    ];
    let mut moof = build_moof_audio(1, 2, 0, &samples);
    moof.patch_default_no_gap();

    // Audio trun flags = 0x000001 | 0x000100 | 0x000200 = 0x000301
    // (no first-sample-flags bit, no per-sample-flags bit).
    let trun_start = 8 + 16 + 8 + 20 + 20;
    let flag_bytes = [
        0u8,
        moof.bytes[trun_start + 9],
        moof.bytes[trun_start + 10],
        moof.bytes[trun_start + 11],
    ];
    let flags = u32::from_be_bytes(flag_bytes);
    assert_eq!(flags, 0x000001 | 0x000100 | 0x000200);

    // Per-sample table starts after data_offset(4) only — no
    // first_sample_flags this time.
    //   trun_start + 8 + 1 + 3 + 4 + 4 = trun_start + 20.
    let table_start = trun_start + 20;
    assert_eq!(read_be_u32(&moof.bytes, table_start), 1024); // sample 0 dur
    assert_eq!(read_be_u32(&moof.bytes, table_start + 4), 256); // sample 0 size
    assert_eq!(read_be_u32(&moof.bytes, table_start + 8), 1024); // sample 1 dur
    assert_eq!(read_be_u32(&moof.bytes, table_start + 12), 256); // sample 1 size
}

#[test]
fn moof_data_offset_patch_is_at_correct_position() {
    // Keyframe-only fragment of 1 sample. Data offset is at a
    // computable position; verify patch_data_offset writes there.
    let samples = vec![CmafSample {
        duration: 1500,
        size: 1234,
        flags: SampleFlags::keyframe(),
    }];
    let mut moof = build_moof_video(1, 1, 0, &samples);
    moof.patch_data_offset(0xDEAD_BEEF);
    let read_back = read_be_u32(&moof.bytes, moof.data_offset_pos);
    assert_eq!(read_back, 0xDEAD_BEEF);
}

// Synthetic AV1 OBU bytes that contain exactly one
// OBU_SEQUENCE_HEADER (type=1, has_size=1, ext=0). This is what
// `extract_sequence_header` sniffs out of the first encoded packet
// to build the av1C config record. Payload is 1 byte (0xAA) — the
// value is irrelevant for our shape tests; the muxer just round-
// trips it as bytes inside av1C.
fn synthetic_seq_header_packet() -> Vec<u8> {
    let header_byte: u8 = (1 << 3) | (1 << 1); // obu_type=1, has_size=1
    vec![header_byte, 0x01, 0xAA]
}

fn find_box<'a>(buf: &'a [u8], box_type: &[u8; 4]) -> Option<&'a [u8]> {
    let mut pos = 0;
    while pos + 8 <= buf.len() {
        let size = read_be_u32(buf, pos) as usize;
        if size < 8 || pos + size > buf.len() {
            return None;
        }
        let kind = &buf[pos + 4..pos + 8];
        if kind == box_type {
            return Some(&buf[pos..pos + size]);
        }
        pos += size;
    }
    None
}

fn ftyp_compatible_brands(ftyp: &[u8]) -> Vec<&[u8]> {
    // size:4 + 'ftyp' + major:4 + minor:4 = 16, then brands[]
    let mut brands = Vec::new();
    let mut p = 16;
    while p + 4 <= ftyp.len() {
        brands.push(&ftyp[p..p + 4]);
        p += 4;
    }
    brands
}

#[test]
fn init_segment_video_lists_cmfc_and_av01_brands() {
    let init = build_init_segment_video(
        1920,
        1080,
        30000,
        &synthetic_seq_header_packet(),
        &ColorMetadata::default(),
    );
    let ftyp = find_box(&init, b"ftyp").expect("init has ftyp");
    let brands = ftyp_compatible_brands(ftyp);
    assert!(
        brands.contains(&b"cmfc".as_slice()),
        "cmfc brand missing: {brands:?}"
    );
    assert!(
        brands.contains(&b"av01".as_slice()),
        "av01 brand missing: {brands:?}"
    );
    assert!(
        brands.contains(&b"iso6".as_slice()),
        "iso6 brand missing: {brands:?}"
    );
}

#[test]
fn init_segment_audio_lists_cmfa_brand() {
    // ASC bytes for AAC-LC: object_type=2 (LC), sample_rate_index=3 (48 kHz),
    // channelConfiguration=2 (stereo).
    let info = AudioInfo::aac_lc(48000, 2, vec![0x11, 0x90]);
    let init = build_init_segment_audio(&info);
    let ftyp = find_box(&init, b"ftyp").expect("init has ftyp");
    let brands = ftyp_compatible_brands(ftyp);
    assert!(
        brands.contains(&b"cmfa".as_slice()),
        "cmfa brand missing: {brands:?}"
    );
    assert!(
        !brands.contains(&b"cmfc".as_slice()),
        "cmfc should not appear in audio init"
    );
}

#[test]
fn init_segment_video_moov_contains_mvex_with_trex() {
    let init = build_init_segment_video(
        1280,
        720,
        30000,
        &synthetic_seq_header_packet(),
        &ColorMetadata::default(),
    );
    let moov = find_box(&init, b"moov").expect("init has moov");
    let mvex = find_box(&moov[8..], b"mvex").expect("moov has mvex");
    assert!(
        find_box(&mvex[8..], b"trex").is_some(),
        "mvex must contain trex"
    );
    assert!(
        find_box(&mvex[8..], b"mehd").is_some(),
        "mvex must contain mehd"
    );
}

#[test]
fn init_segment_video_stbl_has_empty_sample_tables() {
    let init = build_init_segment_video(
        1280,
        720,
        30000,
        &synthetic_seq_header_packet(),
        &ColorMetadata::default(),
    );
    let moov = find_box(&init, b"moov").expect("init has moov");
    let trak = find_box(&moov[8..], b"trak").expect("moov has trak");
    let mdia = find_box(&trak[8..], b"mdia").expect("trak has mdia");
    let minf = find_box(&mdia[8..], b"minf").expect("mdia has minf");
    let stbl = find_box(&minf[8..], b"stbl").expect("minf has stbl");

    // stsz: sample_size=0 (variable), sample_count=0 (no samples in init)
    let stsz = find_box(&stbl[8..], b"stsz").expect("stbl has stsz");
    // 8 (header) + 1 (version) + 3 (flags) + 4 (sample_size) + 4 (sample_count) = 20.
    assert_eq!(stsz.len(), 20);
    assert_eq!(read_be_u32(stsz, 12), 0); // sample_size
    assert_eq!(read_be_u32(stsz, 16), 0); // sample_count

    // stts/stsc/stco: entry_count=0
    for box_type in [b"stts", b"stsc", b"stco"] {
        let bx = find_box(&stbl[8..], box_type).expect("stbl has empty full box");
        assert_eq!(
            bx.len(),
            16,
            "{:?} should be 16-byte empty FullBox",
            std::str::from_utf8(box_type).unwrap()
        );
        assert_eq!(read_be_u32(bx, 12), 0); // entry_count
    }

    // stsd has exactly one entry — the av01 sample entry.
    let stsd = find_box(&stbl[8..], b"stsd").expect("stbl has stsd");
    assert_eq!(read_be_u32(stsd, 12), 1); // entry_count
    // First sample entry should be av01.
    let av01 = &stsd[16..];
    assert_eq!(&av01[4..8], b"av01");
}

#[test]
fn cmaf_video_muxer_emits_init_then_segment_files() {
    let dir = tempfile::tempdir().unwrap();
    let mut muxer =
        CmafVideoMuxer::new(dir.path(), 1280, 720, 30000, ColorMetadata::default()).unwrap();

    // Two-packet "fragment": one keyframe, one delta. Each "payload"
    // starts with the synthetic sequence header (so the muxer's
    // first-packet OBU sniff succeeds) but the muxer doesn't care
    // about the rest of the payload bytes — it just round-trips
    // them through mdat.
    let mut k = synthetic_seq_header_packet();
    k.extend_from_slice(&[0xDE, 0xAD]);
    muxer.add_packet(k, 1500, true).unwrap();
    muxer
        .add_packet(synthetic_seq_header_packet(), 1500, false)
        .unwrap();

    let info = muxer
        .flush_segment()
        .unwrap()
        .expect("flush emits a segment");
    assert_eq!(info.sequence_number, 1);
    assert_eq!(info.duration_ticks, 3000);
    assert!(info.path.exists());
    assert_eq!(info.path.file_name().unwrap(), "seg-00001.m4s");

    // init.mp4 was written lazily on first flush.
    let init_path = dir.path().join("init.mp4");
    assert!(init_path.exists(), "init.mp4 must exist after first flush");

    // Segment file starts with `moof` and contains an `mdat` after.
    let seg_bytes = std::fs::read(&info.path).unwrap();
    assert_eq!(&seg_bytes[4..8], b"moof");
    let moof_size = read_be_u32(&seg_bytes, 0) as usize;
    assert_eq!(&seg_bytes[moof_size + 4..moof_size + 8], b"mdat");

    // Manifest finalize covers the empty-pending case (we already flushed).
    let manifest = muxer.finalize().unwrap();
    assert_eq!(manifest.segments.len(), 1);
    assert_eq!(manifest.timescale, 30000);
    assert!((manifest.duration_seconds() - 0.1).abs() < 1e-6); // 3000/30000 = 0.1s
}

#[test]
fn cmaf_h264_init_segment_is_avc3_with_inline_params() {
    let dir = tempfile::tempdir().unwrap();
    let mut muxer = CmafVideoMuxer::new_with_codec_options(
        dir.path(),
        1280,
        720,
        30000,
        ColorMetadata::default(),
        VideoCodec::H264,
        CmafVideoMuxerOptions::default(),
    )
    .unwrap();
    // Synthetic Annex-B keyframe AU: SPS (7) + PPS (8) + IDR (5).
    let mut kf = vec![0, 0, 0, 1, 0x67, 0x42, 0x00, 0x1e, 0xAA]; // SPS
    kf.extend_from_slice(&[0, 0, 0, 1, 0x68, 0xCE, 0x3C]); // PPS
    kf.extend_from_slice(&[0, 0, 0, 1, 0x65, 0x88, 0x11, 0x22]); // IDR slice
    muxer.add_packet(kf, 1000, true).unwrap();
    muxer
        .add_packet(vec![0, 0, 0, 1, 0x41, 0x9a, 0x33], 1000, false) // P-slice
        .unwrap();
    let info = muxer.flush_segment().unwrap().expect("segment flushed");
    assert!(info.path.exists());
    let manifest = muxer.finalize().unwrap();
    assert_eq!(manifest.segments.len(), 1);

    let has = |buf: &[u8], pat: &[u8; 4]| buf.windows(4).any(|w| w == pat);
    let init = std::fs::read(dir.path().join("init.mp4")).unwrap();
    assert!(has(&init, b"avc3"), "H.264 CMAF init must use the avc3 sample entry");
    assert!(has(&init, b"avcC"), "init must carry the avcC config box");
    assert!(!has(&init, b"av01"), "must NOT contain an av01 box");
    let seg = std::fs::read(&info.path).unwrap();
    assert!(has(&seg, b"moof") && has(&seg, b"mdat"));
}

#[test]
fn cmaf_h265_init_segment_is_hev1() {
    let dir = tempfile::tempdir().unwrap();
    let mut muxer = CmafVideoMuxer::new_with_codec_options(
        dir.path(),
        1280,
        720,
        30000,
        ColorMetadata::default(),
        VideoCodec::H265,
        CmafVideoMuxerOptions::default(),
    )
    .unwrap();
    // Synthetic HEVC keyframe AU: VPS (32) + SPS (33) + PPS (34) + IDR (19).
    let mut kf = vec![0, 0, 0, 1, 0x40, 0x01, 0x0c]; // VPS
    kf.extend_from_slice(&[0, 0, 0, 1, 0x42, 0x01, 0x01, 0x60, 0x00, 0x00, 0x03]); // SPS
    kf.extend_from_slice(&[0, 0, 0, 1, 0x44, 0x01, 0xc1]); // PPS
    kf.extend_from_slice(&[0, 0, 0, 1, 0x26, 0x01, 0xaf]); // IDR_W_RADL slice (type 19)
    muxer.add_packet(kf, 1000, true).unwrap();
    let info = muxer.flush_segment().unwrap().expect("segment flushed");
    let _ = muxer.finalize().unwrap();
    let has = |buf: &[u8], pat: &[u8; 4]| buf.windows(4).any(|w| w == pat);
    let init = std::fs::read(dir.path().join("init.mp4")).unwrap();
    assert!(has(&init, b"hev1"), "H.265 CMAF init must use the hev1 sample entry");
    assert!(has(&init, b"hvcC"), "init must carry the hvcC config box");
    assert!(info.path.exists());
}

#[test]
fn cmaf_video_muxer_options_default_matches_legacy_new() {
    // Calling `new()` and `new_with_options(..., default())` must
    // produce byte-identical first-segment output. This is the
    // contract that lets every existing call site stay on `new()`
    // unmodified.
    let dir_a = tempfile::tempdir().unwrap();
    let dir_b = tempfile::tempdir().unwrap();
    let mut ma = CmafVideoMuxer::new(
        dir_a.path(),
        1280,
        720,
        30000,
        ColorMetadata::default(),
    )
    .unwrap();
    let mut mb = CmafVideoMuxer::new_with_options(
        dir_b.path(),
        1280,
        720,
        30000,
        ColorMetadata::default(),
        CmafVideoMuxerOptions::default(),
    )
    .unwrap();

    let mut kf = synthetic_seq_header_packet();
    kf.extend_from_slice(&[0xDE, 0xAD]);
    ma.add_packet(kf.clone(), 1500, true).unwrap();
    mb.add_packet(kf, 1500, true).unwrap();

    let info_a = ma.flush_segment().unwrap().unwrap();
    let info_b = mb.flush_segment().unwrap().unwrap();
    assert_eq!(info_a.sequence_number, info_b.sequence_number);
    assert_eq!(info_a.duration_ticks, info_b.duration_ticks);
    assert_eq!(
        info_a.path.file_name().unwrap(),
        info_b.path.file_name().unwrap(),
    );
    // Byte-identical moof+mdat — proves no observable difference.
    let bytes_a = std::fs::read(&info_a.path).unwrap();
    let bytes_b = std::fs::read(&info_b.path).unwrap();
    assert_eq!(bytes_a, bytes_b);
    // init.mp4 written in both cases.
    assert!(dir_a.path().join("init.mp4").exists());
    assert!(dir_b.path().join("init.mp4").exists());
}

#[test]
fn cmaf_video_muxer_first_segment_index_offset_writes_correct_filename() {
    // A helper muxer attached at segment 5 of an in-progress rung
    // must produce `seg-00005.m4s` as its first output, not 00001.
    let dir = tempfile::tempdir().unwrap();
    let mut muxer = CmafVideoMuxer::new_with_options(
        dir.path(),
        1280,
        720,
        30000,
        ColorMetadata::default(),
        CmafVideoMuxerOptions {
            first_segment_index: 5,
            first_segment_base_decode_time: 4 * 3000, // 4 prior segments × 3000-tick duration
            write_init_segment: true,
        },
    )
    .unwrap();

    let mut kf = synthetic_seq_header_packet();
    kf.extend_from_slice(&[0xCA, 0xFE]);
    muxer.add_packet(kf, 1500, true).unwrap();
    muxer
        .add_packet(synthetic_seq_header_packet(), 1500, false)
        .unwrap();

    let info = muxer.flush_segment().unwrap().unwrap();
    assert_eq!(
        info.sequence_number, 5,
        "first flush of an offset muxer must produce segment number 5",
    );
    assert_eq!(info.path.file_name().unwrap(), "seg-00005.m4s");

    // Second flush continues the sequence at 6.
    let mut kf2 = synthetic_seq_header_packet();
    kf2.extend_from_slice(&[0xBE, 0xEF]);
    muxer.add_packet(kf2, 1500, true).unwrap();
    let info2 = muxer.flush_segment().unwrap().unwrap();
    assert_eq!(info2.sequence_number, 6);
    assert_eq!(info2.path.file_name().unwrap(), "seg-00006.m4s");
}

#[test]
fn cmaf_video_muxer_offset_base_decode_time_propagates_to_tfdt() {
    // Verifies the `tfdt` box of the offset muxer's first segment
    // carries the configured base_decode_time. Without this, an
    // HLS player would see segment 5 starting at decode-time 0,
    // producing a buffer underrun at the cut from primary's
    // segment 4 to helper's segment 5.
    let dir = tempfile::tempdir().unwrap();
    let mut muxer = CmafVideoMuxer::new_with_options(
        dir.path(),
        1280,
        720,
        30000,
        ColorMetadata::default(),
        CmafVideoMuxerOptions {
            first_segment_index: 5,
            first_segment_base_decode_time: 4 * 3000,
            write_init_segment: true,
        },
    )
    .unwrap();

    let mut kf = synthetic_seq_header_packet();
    kf.extend_from_slice(&[0x01, 0x02]);
    muxer.add_packet(kf, 1500, true).unwrap();
    let info = muxer.flush_segment().unwrap().unwrap();

    // Walk the segment bytes: moof > traf > tfdt. tfdt v1 layout:
    //   8 bytes box header (size + 'tfdt')
    //   1 byte version (=1) + 3 bytes flags
    //   8 bytes base_media_decode_time (u64 BE)
    let bytes = std::fs::read(&info.path).unwrap();
    let moof_size = read_be_u32(&bytes, 0) as usize;
    let moof = &bytes[..moof_size];
    let traf = find_box(&moof[8..], b"traf").expect("moof has traf");
    let tfdt = find_box(&traf[8..], b"tfdt").expect("traf has tfdt");
    let version = tfdt[8];
    assert_eq!(version, 1, "tfdt should be version 1 (u64 decode time)");
    let dt = u64::from_be_bytes([
        tfdt[12], tfdt[13], tfdt[14], tfdt[15], tfdt[16], tfdt[17], tfdt[18], tfdt[19],
    ]);
    assert_eq!(
        dt, 12000,
        "tfdt base_media_decode_time must equal configured offset (4×3000)",
    );
}

#[test]
fn cmaf_video_muxer_write_init_false_skips_init_file() {
    // A helper muxer must NOT write init.mp4 — the primary owns
    // that file. Verify that flush_segment + finalize do not
    // create init.mp4 in the output directory.
    let dir = tempfile::tempdir().unwrap();
    let mut muxer = CmafVideoMuxer::new_with_options(
        dir.path(),
        1280,
        720,
        30000,
        ColorMetadata::default(),
        CmafVideoMuxerOptions {
            first_segment_index: 5,
            first_segment_base_decode_time: 4 * 3000,
            write_init_segment: false,
        },
    )
    .unwrap();

    let mut kf = synthetic_seq_header_packet();
    kf.extend_from_slice(&[0x03, 0x04]);
    muxer.add_packet(kf, 1500, true).unwrap();
    let info = muxer.flush_segment().unwrap().unwrap();
    assert!(
        info.path.exists(),
        "segment file must be written even when init is skipped",
    );
    let init_path = dir.path().join("init.mp4");
    assert!(
        !init_path.exists(),
        "init.mp4 must NOT be written when write_init_segment=false",
    );

    // finalize must also not write init.
    let _ = muxer.finalize().unwrap();
    assert!(
        !init_path.exists(),
        "finalize must not retroactively write init.mp4 when disabled",
    );
}

#[test]
fn cmaf_video_muxer_two_writers_share_output_dir_with_distinct_indices() {
    // The actual helper-task contract: primary writes segments
    // 1..3 + init.mp4 into dir/. Helper writes segments 3..5 into
    // the same dir with write_init_segment=false. After both
    // finalize, all 4 segment files plus init.mp4 exist.
    let dir = tempfile::tempdir().unwrap();

    let mut primary = CmafVideoMuxer::new(
        dir.path(),
        1280,
        720,
        30000,
        ColorMetadata::default(),
    )
    .unwrap();
    let mut helper = CmafVideoMuxer::new_with_options(
        dir.path(),
        1280,
        720,
        30000,
        ColorMetadata::default(),
        CmafVideoMuxerOptions {
            first_segment_index: 3,
            first_segment_base_decode_time: 2 * 3000,
            write_init_segment: false,
        },
    )
    .unwrap();

    // Primary writes segments 1 and 2.
    for _ in 0..2 {
        let mut kf = synthetic_seq_header_packet();
        kf.extend_from_slice(&[0xAA, 0xBB]);
        primary.add_packet(kf, 1500, true).unwrap();
        primary
            .add_packet(synthetic_seq_header_packet(), 1500, false)
            .unwrap();
        primary.flush_segment().unwrap().unwrap();
    }
    // Helper writes segments 3 and 4.
    for _ in 0..2 {
        let mut kf = synthetic_seq_header_packet();
        kf.extend_from_slice(&[0xCC, 0xDD]);
        helper.add_packet(kf, 1500, true).unwrap();
        helper
            .add_packet(synthetic_seq_header_packet(), 1500, false)
            .unwrap();
        helper.flush_segment().unwrap().unwrap();
    }

    primary.finalize().unwrap();
    helper.finalize().unwrap();

    // All four segments + one init.mp4 present.
    for seg_idx in 1..=4 {
        let p = dir.path().join(format!("seg-{seg_idx:05}.m4s"));
        assert!(p.exists(), "segment {seg_idx} missing at {}", p.display());
    }
    let init_path = dir.path().join("init.mp4");
    assert!(init_path.exists(), "primary's init.mp4 must be present");
}

#[test]
#[should_panic(expected = "first_segment_index is 1-based")]
fn cmaf_video_muxer_first_segment_index_zero_panics() {
    let dir = tempfile::tempdir().unwrap();
    let _ = CmafVideoMuxer::new_with_options(
        dir.path(),
        1280,
        720,
        30000,
        ColorMetadata::default(),
        CmafVideoMuxerOptions {
            first_segment_index: 0,
            first_segment_base_decode_time: 0,
            write_init_segment: true,
        },
    );
}

#[test]
fn cmaf_video_muxer_rejects_segment_starting_on_non_keyframe() {
    let dir = tempfile::tempdir().unwrap();
    let mut muxer =
        CmafVideoMuxer::new(dir.path(), 640, 360, 30000, ColorMetadata::default()).unwrap();
    muxer
        .add_packet(synthetic_seq_header_packet(), 1500, false)
        .unwrap();
    let err = muxer
        .flush_segment()
        .expect_err("must fail when first sample is not sync");
    assert!(err.to_string().contains("must start with a sync sample"));
}

#[test]
fn cmaf_audio_muxer_emits_init_and_segments_with_correct_durations() {
    let info = AudioInfo {
        codec: "aac".into(),
        sample_rate: 48000,
        channels: 2,
        timescale: 48000,
        asc_bytes: vec![0x12, 0x10],
        codec_private: vec![],
    };
    let dir = tempfile::tempdir().unwrap();
    let mut muxer = CmafAudioMuxer::new(dir.path(), info).unwrap();

    // 5 AAC frames at 1024 samples each = 5120 ticks @ 48 kHz =
    // ~107 ms total.
    for _ in 0..5 {
        muxer.add_packet(vec![0xDE; 256], 1024).unwrap();
    }
    let seg = muxer
        .flush_segment()
        .unwrap()
        .expect("audio segment emitted");
    assert_eq!(seg.duration_ticks, 5 * 1024);
    assert!(seg.path.exists());
    let init_path = dir.path().join("init.mp4");
    assert!(init_path.exists());

    // Audio segment moof should NOT contain a first_sample_flags
    // slot — the trun layout for audio omits that flag bit. We
    // already cover this in `moof_audio_does_not_emit_first_sample_flags`;
    // here we just verify the file shape is valid.
    let bytes = std::fs::read(&seg.path).unwrap();
    assert_eq!(&bytes[4..8], b"moof");

    let manifest = muxer.finalize().unwrap();
    assert_eq!(manifest.timescale, 48000);
    assert!((manifest.duration_seconds() - (5.0 * 1024.0 / 48000.0)).abs() < 1e-6);
}

#[test]
fn mvex_wraps_mehd_and_one_or_more_trex_in_order() {
    let mehd = build_mehd(10_000);
    let trex_v = build_trex(1, SampleFlags::delta_frame().pack());
    let trex_a = build_trex(2, SampleFlags::keyframe().pack());
    let mvex = build_mvex(&mehd, &[trex_v.clone(), trex_a.clone()]);
    let (size, kind) = box_size_and_type(&mvex);
    assert_eq!(size as usize, mvex.len());
    assert_eq!(kind, b"mvex");
    // 8 (header) + mehd(20) + trex(32) + trex(32) = 92.
    assert_eq!(mvex.len(), 8 + mehd.len() + trex_v.len() + trex_a.len());
    // First child is mehd.
    let (_, child0_kind) = box_size_and_type(&mvex[8..]);
    assert_eq!(child0_kind, b"mehd");
    // Second child is the first trex.
    let (_, child1_kind) = box_size_and_type(&mvex[8 + mehd.len()..]);
    assert_eq!(child1_kind, b"trex");
}
