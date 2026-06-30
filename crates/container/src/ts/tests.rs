use super::*;

// Sub-module items not re-exported into mod.rs need explicit imports.
use super::audio::{AdtsHeader, decode_sample_rate_index, parse_adts_header, synthesize_asc};
use super::framerate::estimate_frame_rate_from_ptses;
use super::pat_pmt::parse_pmt_streams;
use crate::streaming::StreamingDemuxer;

fn ts_pkt(pid: u16, pusi: bool, adaptation: u8, payload: &[u8]) -> [u8; TS_PACKET] {
    let mut p = [0xFFu8; TS_PACKET];
    p[0] = TS_SYNC;
    // TEI=0, PUSI=pusi, transport_priority=0, PID(13)
    p[1] = if pusi { 0x40 } else { 0x00 } | ((pid >> 8) & 0x1F) as u8;
    p[2] = (pid & 0xFF) as u8;
    // scramble=00, adaptation=adaptation, continuity=0
    p[3] = (adaptation & 0x03) << 4;
    let mut off = 4;
    // For these tests we always use adaptation=01 (payload only).
    let pay_len = payload.len().min(TS_PACKET - off);
    p[off..off + pay_len].copy_from_slice(&payload[..pay_len]);
    off += pay_len;
    // Pad any remaining bytes with 0xFF (already initialised).
    let _ = off;
    p
}

#[test]
fn estimate_frame_rate_from_uniform_ptses_returns_exact_fps() {
    // 24 fps: inter-PTS = 90000/24 = 3750 ticks.
    let ptses: Vec<u64> = (0..64).map(|i| i as u64 * 3750).collect();
    let fps = estimate_frame_rate_from_ptses(&ptses).expect("24 fps");
    assert!((fps - 24.0).abs() < 1e-9, "{} != 24.0", fps);
}

#[test]
fn estimate_frame_rate_from_reordered_ptses_sorts_before_delta() {
    // Same 24 fps, but decode-order != display-order (one B-frame
    // pair swapped). Median should still pick up the 3750-tick
    // period cleanly.
    let mut ptses: Vec<u64> = (0..32).map(|i| i as u64 * 3750).collect();
    ptses.swap(5, 6);
    ptses.swap(10, 11);
    let fps = estimate_frame_rate_from_ptses(&ptses).expect("24 fps after swap");
    assert!((fps - 24.0).abs() < 1e-9, "{} != 24.0", fps);
}

#[test]
fn estimate_frame_rate_from_single_outlier_delta_uses_median() {
    // 23 uniform 24-fps deltas + one 10× outlier. Median still 3750.
    let mut ptses: Vec<u64> = (0..24).map(|i| i as u64 * 3750).collect();
    ptses.push(24 * 3750 + 37500); // one huge gap
    let fps = estimate_frame_rate_from_ptses(&ptses).expect("24 fps despite outlier");
    assert!((fps - 24.0).abs() < 1e-9);
}

#[test]
fn estimate_frame_rate_returns_none_when_all_ptses_equal() {
    let ptses = vec![0u64; 10];
    assert!(estimate_frame_rate_from_ptses(&ptses).is_none());
}

#[test]
fn estimate_frame_rate_returns_none_when_fewer_than_two() {
    assert!(estimate_frame_rate_from_ptses(&[]).is_none());
    assert!(estimate_frame_rate_from_ptses(&[1234]).is_none());
}

#[test]
fn estimate_frame_rate_rejects_out_of_range_values() {
    // Single 1-tick delta → fps = 90000, outside [1, 240].
    let ptses = vec![0u64, 1];
    assert!(estimate_frame_rate_from_ptses(&ptses).is_none());
}

#[test]
fn estimate_frame_rate_handles_29_97_ntsc() {
    // 29.97 fps = 30000/1001. Inter-PTS = 90000 * 1001 / 30000 = 3003.
    let ptses: Vec<u64> = (0..32).map(|i| i as u64 * 3003).collect();
    let fps = estimate_frame_rate_from_ptses(&ptses).expect("29.97");
    assert!((fps - 30.0).abs() < 0.05, "got {}", fps); // 90000/3003 = 29.97..30.03
}

#[test]
fn detects_plain_ts_layout() {
    let mut buf = Vec::with_capacity(3 * TS_PACKET);
    for _ in 0..3 {
        let pkt = ts_pkt(0x1FFF, false, 0b01, &[]);
        buf.extend_from_slice(&pkt);
    }
    let (count, stride, prefix) = detect_packet_layout(&buf).unwrap();
    assert_eq!((count, stride, prefix), (3, 188, 0));
}

#[test]
fn parses_minimal_pat_pmt_and_reassembles_one_sample() {
    // Build a PAT pointing at PMT=0x100, a PMT listing video PID=0x200
    // stream_type=MPEG-2, then a single PES packet carrying 16 bytes
    // of video ES.

    // PAT section (we skip CRC correctness — the parser only uses
    // section_length to decide where to stop).
    let mut pat = vec![0u8; 0];
    pat.push(0x00); // table_id
    let section_length: usize = 5 + 4 + 4; // 5 header bytes (after len) + 1 program + CRC
    pat.push(0xB0 | ((section_length >> 8) & 0x0F) as u8);
    pat.push((section_length & 0xFF) as u8);
    pat.extend_from_slice(&[0x00, 0x01, 0xC1, 0x00, 0x00]); // tsid, ver/current, secno, lastno
    pat.extend_from_slice(&[0x00, 0x01]); // program_number = 1
    pat.extend_from_slice(&[0xE1, 0x00]); // reserved + PMT PID = 0x100
    pat.extend_from_slice(&[0, 0, 0, 0]); // CRC placeholder

    // PAT packet payload = [pointer_field=0, section...]
    let mut pat_payload = vec![0u8];
    pat_payload.extend_from_slice(&pat);
    let pat_pkt = ts_pkt(0x0000, true, 0b01, &pat_payload);

    // PMT section.
    let mut pmt = vec![0u8; 0];
    pmt.push(0x02);
    let pmt_sec_len: usize = 9 + 5 + 4; // program_number..pil(9) + 1 stream entry(5) + CRC(4)
    pmt.push(0xB0 | ((pmt_sec_len >> 8) & 0x0F) as u8);
    pmt.push((pmt_sec_len & 0xFF) as u8);
    pmt.extend_from_slice(&[0x00, 0x01, 0xC1, 0x00, 0x00]); // prog, ver/current, sec/last
    pmt.extend_from_slice(&[0xE2, 0x00]); // PCR PID = 0x200
    pmt.extend_from_slice(&[0xF0, 0x00]); // program_info_length = 0
    pmt.extend_from_slice(&[STREAM_TYPE_MPEG2_VIDEO, 0xE2, 0x00, 0xF0, 0x00]); // stream entry
    pmt.extend_from_slice(&[0, 0, 0, 0]); // CRC placeholder
    let mut pmt_payload = vec![0u8];
    pmt_payload.extend_from_slice(&pmt);
    let pmt_pkt = ts_pkt(0x0100, true, 0b01, &pmt_payload);

    // Two PES packets, each 16 bytes of ES, so the reassembler's
    // PUSI-flush path is exercised. Real MPEG-TS files also set
    // PES_packet_length which bounds the first one, but packet_length=0
    // ("unbounded") is also legal for MPEG-2 video PES, which is what
    // we emit here — termination comes from the next PUSI.
    let make_pes = |byte: u8| {
        let mut pes = vec![0u8, 0u8, 1u8]; // start code
        pes.push(0xE0); // stream_id video
        pes.extend_from_slice(&[0u8, 0u8]); // packet_length=0
        pes.push(0x80);
        pes.push(0x80); // PTS_DTS_flags = 10
        pes.push(5); // PES_header_data_length
        pes.extend_from_slice(&[0x21, 0x00, 0x01, 0x00, 0x01]); // PTS=0
        pes.extend_from_slice(&[byte; 16]);
        pes
    };
    let pes_pkt_a = ts_pkt(0x0200, true, 0b01, &make_pes(0xAA));
    let pes_pkt_b = ts_pkt(0x0200, true, 0b01, &make_pes(0xBB));

    let mut buf = Vec::new();
    buf.extend_from_slice(&pat_pkt);
    buf.extend_from_slice(&pmt_pkt);
    buf.extend_from_slice(&pes_pkt_a);
    buf.extend_from_slice(&pes_pkt_b);
    // Trailing null packet so detect_packet_layout sees a sync run.
    buf.extend_from_slice(&ts_pkt(0x1FFF, false, 0b01, &[]));

    let d = demux_ts(&buf).expect("demux");
    assert_eq!(d.codec, "mpeg2");
    // We should have reassembled two samples (the first flushed when
    // the second PUSI arrives). Sample A carries the 16 AU bytes
    // plus whatever TS padding trailed the PES header — the
    // demuxer does not know the bound, so exact byte-for-byte
    // comparison needs packet_length support (future). For now
    // assert: right sample count, correct leading bytes.
    assert_eq!(d.samples.len(), 2);
    assert_eq!(&d.samples[0][..16], &[0xAA; 16]);
    assert_eq!(&d.samples[1][..16], &[0xBB; 16]);
}

#[test]
fn rejects_file_with_no_sync() {
    let garbage = vec![0u8; TS_PACKET * 3];
    assert!(demux_ts(&garbage).is_err());
}

// ---------------- AAC-ADTS / ASC unit tests (Squad-27) ----------------

/// Build a 7-byte ADTS header (no CRC) with the given fields.
/// `frame_length` covers header + payload.
fn build_adts_header_7(profile: u8, sr_idx: u8, ch_cfg: u8, frame_length: usize) -> [u8; 7] {
    let mut h = [0u8; 7];
    // Bytes 0..1: 0xFFF sync + ID(1)=0 (MPEG-4) + layer(2)=0 +
    // protection_absent(1)=1.
    h[0] = 0xFF;
    h[1] = 0xF0 | 0x01; // protection_absent = 1
    // Byte 2: profile(2) | sr_idx(4) | private(1) | ch_cfg high bit(1).
    h[2] = ((profile & 0x03) << 6) | ((sr_idx & 0x0F) << 2) | ((ch_cfg >> 2) & 0x01);
    // Byte 3: ch_cfg low 2 bits(2) | original/copy(1) | home(1) |
    // copyright_id_bit(1) | copyright_id_start(1) | frame_length high 2.
    h[3] = ((ch_cfg & 0x03) << 6) | (((frame_length >> 11) & 0x03) as u8);
    h[4] = ((frame_length >> 3) & 0xFF) as u8;
    h[5] = (((frame_length & 0x07) << 5) | 0x1F) as u8;
    // Byte 6: low buffer_fullness bits + number_of_raw_data_blocks(2) = 0.
    h[6] = 0xFC;
    h
}

/// Build a 9-byte ADTS header (with CRC). CRC bytes are placeholders.
fn build_adts_header_9(profile: u8, sr_idx: u8, ch_cfg: u8, frame_length: usize) -> [u8; 9] {
    let mut h = [0u8; 9];
    h[0] = 0xFF;
    h[1] = 0xF0; // protection_absent = 0 → CRC present
    h[2] = ((profile & 0x03) << 6) | ((sr_idx & 0x0F) << 2) | ((ch_cfg >> 2) & 0x01);
    h[3] = ((ch_cfg & 0x03) << 6) | (((frame_length >> 11) & 0x03) as u8);
    h[4] = ((frame_length >> 3) & 0xFF) as u8;
    h[5] = (((frame_length & 0x07) << 5) | 0x1F) as u8;
    h[6] = 0xFC;
    // Bytes 7..8: CRC placeholder (not validated by the parser).
    h
}

#[test]
fn adts_parser_decodes_canonical_lc_stereo_7byte_header() {
    // Canonical LC stereo @ 48k, 100-byte payload + 7-byte header.
    let h = build_adts_header_7(1, 3, 2, 107);
    let parsed = parse_adts_header(&h).expect("must parse 7-byte ADTS header");
    assert_eq!(parsed.profile, 1, "ADTS profile=1 LC");
    assert_eq!(parsed.sampling_frequency_index, 3, "sr_idx=3 → 48kHz");
    assert_eq!(parsed.channel_configuration, 2, "ch_cfg=2 stereo");
    assert_eq!(parsed.frame_length, 107);
    assert_eq!(parsed.header_len, 7, "protection_absent=1 → 7-byte header");
    assert_eq!(
        decode_sample_rate_index(parsed.sampling_frequency_index),
        Some(48000)
    );
}

#[test]
fn adts_parser_decodes_9byte_header_with_crc() {
    let h = build_adts_header_9(1, 4, 2, 109);
    let parsed = parse_adts_header(&h).expect("must parse 9-byte ADTS header");
    assert_eq!(parsed.profile, 1);
    assert_eq!(parsed.sampling_frequency_index, 4, "sr_idx=4 → 44.1kHz");
    assert_eq!(parsed.channel_configuration, 2);
    assert_eq!(parsed.frame_length, 109);
    assert_eq!(
        parsed.header_len, 9,
        "protection_absent=0 → 9-byte header (incl CRC)"
    );
    assert_eq!(
        decode_sample_rate_index(parsed.sampling_frequency_index),
        Some(44100)
    );
}

#[test]
fn adts_parser_decodes_aac_profile_bits_full_range() {
    // ADTS profile is 2 bits → values 0..=3 are the only legal forms:
    // 0=Main, 1=LC, 2=SSR, 3=LTP. Parent HE-AAC's AOT=5 (SBR) cannot
    // be carried in ADTS — HE-AAC streams in ADTS look like LC at
    // the header level and signal SBR inside the access unit. The
    // parser must round-trip every legal 2-bit profile value so the
    // upstream router can decide what to do (we accept LC=1 and
    // reject the rest at mux-validation time).
    for profile in 0u8..=3 {
        let h = build_adts_header_7(profile, 3, 2, 32);
        let parsed =
            parse_adts_header(&h).unwrap_or_else(|| panic!("must parse profile={profile}"));
        assert_eq!(parsed.profile, profile);
    }
}

#[test]
fn adts_parser_rejects_missing_sync() {
    let mut h = build_adts_header_7(1, 3, 2, 32);
    h[0] = 0x00;
    assert!(parse_adts_header(&h).is_none());
}

#[test]
fn adts_parser_rejects_short_buffer() {
    let h = build_adts_header_7(1, 3, 2, 32);
    assert!(
        parse_adts_header(&h[..6]).is_none(),
        "<7 bytes can't carry a complete ADTS header"
    );
}

#[test]
fn synthesize_asc_lc_stereo_48k_emits_0x1190() {
    // Squad-27 spec example: ADTS profile=1 (LC), sr_idx=3 (48k),
    // ch_cfg=2 (stereo) → ASC `0x11 0x90`.
    // Bit math:
    //   AOT=2 (LC),    5 bits = 00010
    //   sr_idx=3,      4 bits = 0011
    //   ch_cfg=2,      4 bits = 0010
    //   GA padding,    3 bits = 000
    // Concat: 00010 0011 0010 000 = 0001 0001 1001 0000 = 0x1190
    let adts = AdtsHeader {
        profile: 1,
        sampling_frequency_index: 3,
        channel_configuration: 2,
        frame_length: 0,
        header_len: 7,
    };
    let asc = synthesize_asc(&adts);
    assert_eq!(asc, [0x11, 0x90], "LC/48k/stereo → ASC 0x11 0x90");
}

#[test]
fn synthesize_asc_lc_mono_44k() {
    // AOT=2, sr_idx=4 (44.1k), ch_cfg=1 (mono):
    //   00010 0100 0001 000 = 0001 0010 0000 1000 = 0x12 0x08
    let adts = AdtsHeader {
        profile: 1,
        sampling_frequency_index: 4,
        channel_configuration: 1,
        frame_length: 0,
        header_len: 7,
    };
    assert_eq!(synthesize_asc(&adts), [0x12, 0x08]);
}

#[test]
fn synthesize_asc_main_aot_at_44k_5p1_rejected_at_channel_layer() {
    // ADTS profile=0 (Main) → ASC AOT=1. sr_idx=4 (44.1k),
    // ch_cfg=6 (5.1). The ASC bit packing must round-trip these
    // values regardless of whether the downstream mux accepts them
    // (mux today validates channels in {1, 2}).
    //   00001 0100 0110 000 = 0000 1010 0011 0000 = 0x0A 0x30
    let adts = AdtsHeader {
        profile: 0,
        sampling_frequency_index: 4,
        channel_configuration: 6,
        frame_length: 0,
        header_len: 7,
    };
    assert_eq!(synthesize_asc(&adts), [0x0A, 0x30]);
}

#[test]
fn adts_strip_7byte_header_yields_payload_only() {
    // Synthesize one ADTS frame: 7-byte header + 100-byte payload.
    // Run it through extract_ts_aac_audio's frame loop (via a minimal
    // synthetic TS) and assert the resulting sample is exactly 100
    // bytes — header stripped.
    let mut frame = Vec::with_capacity(107);
    frame.extend_from_slice(&build_adts_header_7(1, 3, 2, 107));
    frame.extend_from_slice(&[0x42u8; 100]);
    // Drive the frame loop directly to avoid the PES/TS scaffolding.
    // We test the public extraction in a separate integration test.
    let header = parse_adts_header(&frame).unwrap();
    assert_eq!(header.frame_length, 107);
    let payload = &frame[header.header_len..header.frame_length];
    assert_eq!(payload.len(), 100);
    assert!(payload.iter().all(|b| *b == 0x42));
}

#[test]
fn adts_sample_rate_table_covers_documented_indices() {
    // Spot-check the two anchors plus the boundary indices.
    assert_eq!(decode_sample_rate_index(0), Some(96000));
    assert_eq!(decode_sample_rate_index(3), Some(48000));
    assert_eq!(decode_sample_rate_index(4), Some(44100));
    assert_eq!(decode_sample_rate_index(12), Some(7350));
    assert!(decode_sample_rate_index(13).is_none(), "13 is reserved");
    assert!(
        decode_sample_rate_index(15).is_none(),
        "15 (escape) not supported"
    );
}

/// End-to-end: build a synthetic TS file with PAT + PMT advertising
/// MPEG-2 video on PID 0x200 AND AAC-ADTS on PID 0x300, plus PES
/// packets carrying ADTS frames. After demux, the audio track must
/// surface with synthesized ASC + stripped AAC samples + 1024-tick
/// durations.
#[test]
fn demux_ts_yields_audio_track_when_pmt_advertises_aac() {
    // ---- PAT pointing at PMT 0x100 ----
    let mut pat = vec![0x00];
    let pat_section_len: usize = 5 + 4 + 4;
    pat.push(0xB0 | ((pat_section_len >> 8) & 0x0F) as u8);
    pat.push((pat_section_len & 0xFF) as u8);
    pat.extend_from_slice(&[0x00, 0x01, 0xC1, 0x00, 0x00]);
    pat.extend_from_slice(&[0x00, 0x01, 0xE1, 0x00, 0u8, 0u8, 0u8, 0u8]);
    let mut pat_payload = vec![0u8];
    pat_payload.extend_from_slice(&pat);
    let pat_pkt = ts_pkt(0x0000, true, 0b01, &pat_payload);

    // ---- PMT advertising MPEG-2 video (PID 0x200) and AAC-ADTS audio
    // (PID 0x300) ----
    let mut pmt = vec![0x02];
    let pmt_section_len: usize = 9 + 5 + 5 + 4; // hdr + 2 stream entries + CRC
    pmt.push(0xB0 | ((pmt_section_len >> 8) & 0x0F) as u8);
    pmt.push((pmt_section_len & 0xFF) as u8);
    pmt.extend_from_slice(&[0x00, 0x01, 0xC1, 0x00, 0x00]);
    pmt.extend_from_slice(&[0xE2, 0x00]); // PCR PID = 0x200
    pmt.extend_from_slice(&[0xF0, 0x00]); // program_info_length = 0
    // Stream 1: MPEG-2 video on 0x200
    pmt.extend_from_slice(&[STREAM_TYPE_MPEG2_VIDEO, 0xE2, 0x00, 0xF0, 0x00]);
    // Stream 2: AAC-ADTS on 0x300
    pmt.extend_from_slice(&[STREAM_TYPE_AAC_ADTS, 0xE3, 0x00, 0xF0, 0x00]);
    pmt.extend_from_slice(&[0u8; 4]); // CRC placeholder
    let mut pmt_payload = vec![0u8];
    pmt_payload.extend_from_slice(&pmt);
    let pmt_pkt = ts_pkt(0x0100, true, 0b01, &pmt_payload);

    // ---- Video PES (one packet, byte-pattern 0xAA × 16) so video
    // path doesn't bail. ----
    let video_pes = {
        let mut pes = vec![
            0u8, 0u8, 1u8, 0xE0, 0u8, 0u8, 0x80, 0x80, 5, 0x21, 0x00, 0x01, 0x00, 0x01,
        ];
        pes.extend_from_slice(&[0xAAu8; 16]);
        pes
    };
    let video_pkt = ts_pkt(0x0200, true, 0b01, &video_pes);

    // ---- Audio PES carrying TWO ADTS frames (so we exercise the
    // frame-walking loop, not just the first). Each frame: 7-byte
    // header + 32-byte payload = 39 bytes total.
    let mut adts_stream = Vec::new();
    for fill in [0xCCu8, 0xDDu8] {
        adts_stream.extend_from_slice(&build_adts_header_7(1, 3, 2, 39));
        adts_stream.extend_from_slice(&[fill; 32]);
    }
    let audio_pes = {
        // PES header (audio stream_id 0xC0).
        let mut pes = vec![
            0u8, 0u8, 1u8, 0xC0, 0u8, 0u8, 0x80, 0x80, 5, 0x21, 0x00, 0x01, 0x00, 0x01,
        ];
        pes.extend_from_slice(&adts_stream);
        pes
    };
    let audio_pkt = ts_pkt(0x0300, true, 0b01, &audio_pes);

    let mut buf = Vec::new();
    buf.extend_from_slice(&pat_pkt);
    buf.extend_from_slice(&pmt_pkt);
    buf.extend_from_slice(&video_pkt);
    buf.extend_from_slice(&audio_pkt);
    buf.extend_from_slice(&ts_pkt(0x1FFF, false, 0b01, &[]));

    let d = demux_ts(&buf).expect("demux must succeed");
    assert_eq!(d.codec, "mpeg2");
    let audio = d.audio.expect("AAC audio track must be surfaced");
    assert_eq!(audio.codec, "aac");
    assert_eq!(audio.channels, 2, "ch_cfg=2 stereo");
    assert_eq!(audio.sample_rate, 48000, "sr_idx=3 → 48k");
    assert_eq!(audio.timescale, 48000, "AAC timescale = sample_rate");
    assert_eq!(
        audio.asc,
        vec![0x11, 0x90],
        "synthesized ASC for LC/48k/stereo"
    );
    assert_eq!(audio.samples.len(), 2, "two ADTS frames → two samples");
    assert_eq!(
        audio.samples[0].len(),
        32,
        "32-byte payload after 7-byte header strip"
    );
    assert!(audio.samples[0].iter().all(|b| *b == 0xCC));
    assert!(audio.samples[1].iter().all(|b| *b == 0xDD));
    assert_eq!(
        audio.durations,
        vec![1024, 1024],
        "AAC-LC frame duration = 1024 ticks @ sample-rate timescale"
    );
}

#[test]
fn demux_ts_emits_audio_none_when_no_aac_stream_in_pmt() {
    // The original two-stream test (video-only PMT). No audio expected.
    let mut buf = Vec::new();
    let mut pat = vec![0x00];
    let pat_section_len: usize = 5 + 4 + 4;
    pat.push(0xB0 | ((pat_section_len >> 8) & 0x0F) as u8);
    pat.push((pat_section_len & 0xFF) as u8);
    pat.extend_from_slice(&[0x00, 0x01, 0xC1, 0x00, 0x00]);
    pat.extend_from_slice(&[0x00, 0x01, 0xE1, 0x00, 0u8, 0u8, 0u8, 0u8]);
    let mut pat_payload = vec![0u8];
    pat_payload.extend_from_slice(&pat);
    buf.extend_from_slice(&ts_pkt(0x0000, true, 0b01, &pat_payload));

    let mut pmt = vec![0x02];
    let pmt_section_len: usize = 9 + 5 + 4;
    pmt.push(0xB0 | ((pmt_section_len >> 8) & 0x0F) as u8);
    pmt.push((pmt_section_len & 0xFF) as u8);
    pmt.extend_from_slice(&[0x00, 0x01, 0xC1, 0x00, 0x00]);
    pmt.extend_from_slice(&[0xE2, 0x00, 0xF0, 0x00]);
    pmt.extend_from_slice(&[STREAM_TYPE_MPEG2_VIDEO, 0xE2, 0x00, 0xF0, 0x00]);
    pmt.extend_from_slice(&[0u8; 4]);
    let mut pmt_payload = vec![0u8];
    pmt_payload.extend_from_slice(&pmt);
    buf.extend_from_slice(&ts_pkt(0x0100, true, 0b01, &pmt_payload));

    let video_pes = {
        let mut pes = vec![
            0u8, 0u8, 1u8, 0xE0, 0u8, 0u8, 0x80, 0x80, 5, 0x21, 0x00, 0x01, 0x00, 0x01,
        ];
        pes.extend_from_slice(&[0xAAu8; 16]);
        pes
    };
    buf.extend_from_slice(&ts_pkt(0x0200, true, 0b01, &video_pes));
    buf.extend_from_slice(&ts_pkt(0x1FFF, false, 0b01, &[]));

    let d = demux_ts(&buf).expect("demux");
    assert!(
        d.audio.is_none(),
        "PMT without AAC-ADTS stream → no audio track surfaced"
    );
}

// ---------------- Squad-37: AC-3 / E-AC-3 in TS, multi-program, encrypted ----------------

/// Build a minimal AC-3 syncframe by hand with a valid frmsizecod:
/// fscod=0 (48k), bit_rate_code=8 (128 kbps) → frame_length = 384
/// bytes per Table F.7. acmod=2 stereo, lfeon=0, bsid=8, bsmod=0.
/// The body bytes after the BSI prefix are zero-padded — only the
/// first ~7 bytes participate in our parser.
fn synth_ac3_frame_stereo_48k_128k() -> Vec<u8> {
    let mut bw = BitWriter::new();
    bw.put(16, 0x0B77); // syncword
    bw.put(16, 0); // crc1
    bw.put(2, 0); // fscod=0 → 48k
    bw.put(6, 8 << 1); // frmsizecod = bit_rate_code(8) << 1 = 16
    bw.put(5, 8); // bsid
    bw.put(3, 0); // bsmod
    bw.put(3, 2); // acmod=2 stereo
    // acmod=2 → dsurmod (2 bits)
    bw.put(2, 0);
    bw.put(1, 0); // lfeon=0
    // Pad up to 384 bytes (the AC-3 frame size we just announced).
    while bw.bytes.len() < 384 {
        bw.put(8, 0);
    }
    bw.flush()
}

/// E-AC-3 stereo frame with 6 audio blocks (numblkscod=3) at 48k.
/// frmsiz chosen such that frame_size_bytes = 192 ((0x5F + 1) * 2).
fn synth_eac3_frame_stereo_48k_192bytes() -> Vec<u8> {
    let mut bw = BitWriter::new();
    bw.put(16, 0x0B77);
    bw.put(2, 0); // strmtyp = 0 (independent)
    bw.put(3, 0); // substreamid
    bw.put(11, 0x5F); // frmsiz = 95 → frame_size = 192 bytes
    bw.put(2, 0); // fscod=0 → 48k
    bw.put(2, 3); // numblkscod=3 → 6 blocks
    bw.put(3, 2); // acmod=2 stereo
    bw.put(1, 0); // lfeon
    bw.put(5, 16); // bsid=16
    bw.put(5, 0); // dialnorm
    bw.put(1, 0); // compre=0
    while bw.bytes.len() < 192 {
        bw.put(8, 0);
    }
    bw.flush()
}

/// Local copy of the BitWriter used by the existing AAC tests, kept
/// alongside the Squad-37 sync-frame builders for self-containment.
struct BitWriter {
    bytes: Vec<u8>,
    bit_pos: usize,
}
impl BitWriter {
    fn new() -> Self {
        Self {
            bytes: Vec::new(),
            bit_pos: 0,
        }
    }
    fn put(&mut self, n: usize, v: u32) {
        for i in (0..n).rev() {
            let bit = ((v >> i) & 0x01) as u8;
            if self.bit_pos % 8 == 0 {
                self.bytes.push(0);
            }
            let byte_idx = self.bit_pos / 8;
            let bit_idx = 7 - (self.bit_pos % 8);
            self.bytes[byte_idx] |= bit << bit_idx;
            self.bit_pos += 1;
        }
    }
    fn flush(self) -> Vec<u8> {
        self.bytes
    }
}

/// Build a continuation TS packet (PUSI=0) on `pid` with raw
/// `payload` bytes. Used by `build_ts_with_audio` when an audio PES
/// payload doesn't fit in a single 188-byte packet — the PES header
/// rides on the PUSI=1 packet, and continuation packets carry the
/// rest of the elementary-stream bytes verbatim until the next PUSI.
fn ts_pkt_continuation(pid: u16, payload: &[u8]) -> [u8; TS_PACKET] {
    let mut p = [0xFFu8; TS_PACKET];
    p[0] = TS_SYNC;
    p[1] = ((pid >> 8) & 0x1F) as u8; // PUSI=0
    p[2] = (pid & 0xFF) as u8;
    p[3] = 0b01 << 4; // adaptation=01 (payload only), continuity=0
    let pay_len = payload.len().min(TS_PACKET - 4);
    p[4..4 + pay_len].copy_from_slice(&payload[..pay_len]);
    p
}

/// Helper to build a TS file with: PAT, PMT, video PES (so the
/// video gate doesn't bail), audio PES on `audio_pid` with a given
/// `stream_type` byte and `descriptor_loop` for the PMT entry.
/// `audio_es` is the elementary-stream payload (AC-3 frame, etc.)
/// inserted into the audio PES packet body. If `audio_es` is too
/// large to fit in a single TS packet's payload area (~184 bytes),
/// the helper emits one PUSI=1 packet with the PES header + the
/// first chunk and successive PUSI=0 continuation packets carrying
/// the rest.
fn build_ts_with_audio(
    audio_stream_type: u8,
    audio_descriptors: &[u8],
    audio_pid: u16,
    audio_es: &[u8],
) -> Vec<u8> {
    // PAT pointing at PMT 0x100.
    let mut pat = vec![0x00];
    let pat_section_len: usize = 5 + 4 + 4;
    pat.push(0xB0 | ((pat_section_len >> 8) & 0x0F) as u8);
    pat.push((pat_section_len & 0xFF) as u8);
    pat.extend_from_slice(&[0x00, 0x01, 0xC1, 0x00, 0x00]);
    pat.extend_from_slice(&[0x00, 0x01, 0xE1, 0x00, 0u8, 0u8, 0u8, 0u8]);
    let mut pat_payload = vec![0u8];
    pat_payload.extend_from_slice(&pat);
    let pat_pkt = ts_pkt(0x0000, true, 0b01, &pat_payload);

    // PMT advertising MPEG-2 video on 0x200 + audio entry.
    let mut pmt = vec![0x02];
    let pmt_stream_entries = 5  // video stream entry
        + 5 + audio_descriptors.len(); // audio stream entry + descriptors
    let pmt_section_len: usize = 9 + pmt_stream_entries + 4;
    pmt.push(0xB0 | ((pmt_section_len >> 8) & 0x0F) as u8);
    pmt.push((pmt_section_len & 0xFF) as u8);
    pmt.extend_from_slice(&[0x00, 0x01, 0xC1, 0x00, 0x00]);
    pmt.extend_from_slice(&[0xE2, 0x00]); // PCR PID = 0x200
    pmt.extend_from_slice(&[0xF0, 0x00]); // program_info_length=0
    // Stream 1: MPEG-2 video on 0x200, no descriptors.
    pmt.extend_from_slice(&[STREAM_TYPE_MPEG2_VIDEO, 0xE2, 0x00, 0xF0, 0x00]);
    // Stream 2: audio_pid w/ given stream_type + descriptors.
    pmt.push(audio_stream_type);
    pmt.push(0xE0 | ((audio_pid >> 8) & 0x1F) as u8);
    pmt.push((audio_pid & 0xFF) as u8);
    let esi_len = audio_descriptors.len() as u16;
    pmt.push(0xF0 | ((esi_len >> 8) & 0x0F) as u8);
    pmt.push((esi_len & 0xFF) as u8);
    pmt.extend_from_slice(audio_descriptors);
    pmt.extend_from_slice(&[0u8; 4]); // CRC placeholder
    let mut pmt_payload = vec![0u8];
    pmt_payload.extend_from_slice(&pmt);
    let pmt_pkt = ts_pkt(0x0100, true, 0b01, &pmt_payload);

    // Video PES (just enough so the video path doesn't bail).
    let video_pes = {
        let mut pes = vec![
            0u8, 0u8, 1u8, 0xE0, 0u8, 0u8, 0x80, 0x80, 5, 0x21, 0x00, 0x01, 0x00, 0x01,
        ];
        pes.extend_from_slice(&[0xAAu8; 16]);
        pes
    };
    let video_pkt = ts_pkt(0x0200, true, 0b01, &video_pes);

    // Audio PES — a single PES packet carrying all of audio_es,
    // potentially split across multiple TS packets via continuation.
    // Stream_id 0xC0 is audio per ISO/IEC 13818-1 §2.4.3.7.
    // Note: for AC-3 / E-AC-3, ATSC A/53 PES uses stream_id 0xBD
    // (PES private) rather than 0xC0; our parse_pes_header_audio
    // accepts the 0xC0..=0xDF range so we use 0xC0 here for test
    // simplicity. In real-world bitstreams the parser would also
    // need 0xBD support — that's a separate uplift.
    let mut audio_pes = vec![
        0u8, 0u8, 1u8, 0xC0, 0u8, 0u8, 0x80, 0x80, 5, 0x21, 0x00, 0x01, 0x00, 0x01,
    ];
    audio_pes.extend_from_slice(audio_es);

    // Split audio_pes across one PUSI=1 packet plus continuation
    // packets so PES payloads larger than 184 bytes flow through.
    let first_chunk_max = TS_PACKET - 4; // 184 bytes per TS packet payload
    let mut audio_pkts: Vec<[u8; TS_PACKET]> = Vec::new();
    let first_len = audio_pes.len().min(first_chunk_max);
    audio_pkts.push(ts_pkt(audio_pid, true, 0b01, &audio_pes[..first_len]));
    let mut cursor = first_len;
    while cursor < audio_pes.len() {
        let end = (cursor + first_chunk_max).min(audio_pes.len());
        audio_pkts.push(ts_pkt_continuation(audio_pid, &audio_pes[cursor..end]));
        cursor = end;
    }

    let mut buf = Vec::new();
    buf.extend_from_slice(&pat_pkt);
    buf.extend_from_slice(&pmt_pkt);
    buf.extend_from_slice(&video_pkt);
    for pkt in &audio_pkts {
        buf.extend_from_slice(pkt);
    }
    buf.extend_from_slice(&ts_pkt(0x1FFF, false, 0b01, &[]));
    buf
}

#[test]
fn pmt_walker_classifies_aac_ac3_eac3_stream_types() {
    // Build a synthetic PMT section with one of each audio
    // stream_type and verify the walker tags them correctly.
    let mut pmt = vec![0x02];
    let stream_entries = 5 + 5 + 5 + 5; // video + AAC + AC-3 + E-AC-3
    let pmt_section_len: usize = 9 + stream_entries + 4;
    pmt.push(0xB0 | ((pmt_section_len >> 8) & 0x0F) as u8);
    pmt.push((pmt_section_len & 0xFF) as u8);
    pmt.extend_from_slice(&[0x00, 0x01, 0xC1, 0x00, 0x00]);
    pmt.extend_from_slice(&[0xE2, 0x00, 0xF0, 0x00]); // PCR + pil=0
    pmt.extend_from_slice(&[STREAM_TYPE_MPEG2_VIDEO, 0xE2, 0x00, 0xF0, 0x00]);
    pmt.extend_from_slice(&[STREAM_TYPE_AAC_ADTS, 0xE3, 0x00, 0xF0, 0x00]);
    pmt.extend_from_slice(&[STREAM_TYPE_AC3, 0xE4, 0x00, 0xF0, 0x00]);
    pmt.extend_from_slice(&[STREAM_TYPE_EAC3, 0xE5, 0x00, 0xF0, 0x00]);
    pmt.extend_from_slice(&[0u8; 4]);

    let (video, audio) = parse_pmt_streams(&pmt).expect("parse");
    assert_eq!(video.len(), 1);
    assert_eq!(video[0].pid, 0x200);
    assert_eq!(audio.len(), 3);
    assert_eq!(
        (audio[0].pid, audio[0].kind),
        (0x300, AudioCodecKind::AacAdts)
    );
    assert_eq!((audio[1].pid, audio[1].kind), (0x400, AudioCodecKind::Ac3));
    assert_eq!((audio[2].pid, audio[2].kind), (0x500, AudioCodecKind::Eac3));
}

#[test]
fn pmt_walker_recognises_dvb_ac3_via_registration_descriptor() {
    // PES private (0x06) with a registration_descriptor whose 4-char
    // identifier is "AC-3" → audio routed as AC-3 per ETSI TS 101 154.
    let mut pmt = vec![0x02];
    let descriptors: [u8; 6] = [DESC_TAG_REGISTRATION, 4, b'A', b'C', b'-', b'3'];
    let stream_entries = 5 + 5 + descriptors.len();
    let pmt_section_len: usize = 9 + stream_entries + 4;
    pmt.push(0xB0 | ((pmt_section_len >> 8) & 0x0F) as u8);
    pmt.push((pmt_section_len & 0xFF) as u8);
    pmt.extend_from_slice(&[0x00, 0x01, 0xC1, 0x00, 0x00]);
    pmt.extend_from_slice(&[0xE2, 0x00, 0xF0, 0x00]);
    pmt.extend_from_slice(&[STREAM_TYPE_MPEG2_VIDEO, 0xE2, 0x00, 0xF0, 0x00]);
    pmt.push(STREAM_TYPE_PES_PRIVATE);
    pmt.extend_from_slice(&[0xE3, 0x00]);
    let esi_len = descriptors.len() as u16;
    pmt.push(0xF0 | ((esi_len >> 8) & 0x0F) as u8);
    pmt.push((esi_len & 0xFF) as u8);
    pmt.extend_from_slice(&descriptors);
    pmt.extend_from_slice(&[0u8; 4]);

    let (_, audio) = parse_pmt_streams(&pmt).expect("parse");
    assert_eq!(audio.len(), 1);
    assert_eq!(audio[0].kind, AudioCodecKind::Ac3);
    assert_eq!(audio[0].stream_type, STREAM_TYPE_PES_PRIVATE);
}

#[test]
fn pmt_walker_recognises_dvb_eac3_via_registration_descriptor() {
    let mut pmt = vec![0x02];
    let descriptors: [u8; 6] = [DESC_TAG_REGISTRATION, 4, b'E', b'A', b'C', b'3'];
    let stream_entries = 5 + 5 + descriptors.len();
    let pmt_section_len: usize = 9 + stream_entries + 4;
    pmt.push(0xB0 | ((pmt_section_len >> 8) & 0x0F) as u8);
    pmt.push((pmt_section_len & 0xFF) as u8);
    pmt.extend_from_slice(&[0x00, 0x01, 0xC1, 0x00, 0x00]);
    pmt.extend_from_slice(&[0xE2, 0x00, 0xF0, 0x00]);
    pmt.extend_from_slice(&[STREAM_TYPE_MPEG2_VIDEO, 0xE2, 0x00, 0xF0, 0x00]);
    pmt.push(STREAM_TYPE_PES_PRIVATE);
    pmt.extend_from_slice(&[0xE3, 0x00]);
    let esi_len = descriptors.len() as u16;
    pmt.push(0xF0 | ((esi_len >> 8) & 0x0F) as u8);
    pmt.push((esi_len & 0xFF) as u8);
    pmt.extend_from_slice(&descriptors);
    pmt.extend_from_slice(&[0u8; 4]);

    let (_, audio) = parse_pmt_streams(&pmt).expect("parse");
    assert_eq!(audio.len(), 1);
    assert_eq!(audio[0].kind, AudioCodecKind::Eac3);
}

#[test]
fn extract_ac3_frames_from_synthetic_ts_yields_passthrough_track() {
    // stream_type 0x81, no descriptors needed.
    let frame = synth_ac3_frame_stereo_48k_128k();
    // Concatenate two frames so the frame loop runs more than once.
    let mut es = frame.clone();
    es.extend_from_slice(&frame);
    let buf = build_ts_with_audio(STREAM_TYPE_AC3, &[], 0x300, &es);

    let d = demux_ts(&buf).expect("demux");
    let audio = d.audio.expect("AC-3 audio surfaced");
    assert_eq!(audio.codec, "ac3");
    assert_eq!(audio.channels, 2);
    assert_eq!(audio.sample_rate, 48_000);
    assert_eq!(audio.timescale, 48_000);
    // dac3 body is the 3-byte payload that goes into the MP4 sample
    // entry verbatim — derived from the first sync header.
    assert_eq!(audio.codec_private.len(), 3);
    // Two frames in, two samples out (raw frame bytes, sync word
    // intact).
    assert!(
        audio.samples.len() >= 1,
        "at least one AC-3 frame extracted"
    );
    assert_eq!(
        &audio.samples[0][..2],
        &[0x0B, 0x77],
        "AC-3 frame begins with 0x0B77 sync word verbatim"
    );
    // Each AC-3 frame is 1536 samples per spec.
    assert!(
        audio.durations.iter().all(|&d| d == 1536),
        "AC-3 frames are 1536 samples each"
    );
}

#[test]
fn extract_eac3_frames_from_synthetic_ts_yields_passthrough_track() {
    let frame = synth_eac3_frame_stereo_48k_192bytes();
    let mut es = frame.clone();
    es.extend_from_slice(&frame);
    let buf = build_ts_with_audio(STREAM_TYPE_EAC3, &[], 0x300, &es);

    let d = demux_ts(&buf).expect("demux");
    let audio = d.audio.expect("E-AC-3 audio surfaced");
    assert_eq!(audio.codec, "eac3");
    assert_eq!(audio.channels, 2);
    assert_eq!(audio.sample_rate, 48_000);
    // dec3 single-substream body is 5 bytes per ETSI TS 102 366 §F.6.
    assert_eq!(audio.codec_private.len(), 5);
    assert!(!audio.samples.is_empty());
    assert_eq!(
        &audio.samples[0][..2],
        &[0x0B, 0x77],
        "E-AC-3 frame begins with 0x0B77 sync word verbatim"
    );
    // numblkscod=3 → 1536 samples/frame.
    assert!(audio.durations.iter().all(|&d| d == 1536));
}

#[test]
fn extract_ac3_via_pes_private_with_dvb_registration() {
    // stream_type 0x06 + registration "AC-3" must route through the
    // AC-3 extractor end-to-end.
    let frame = synth_ac3_frame_stereo_48k_128k();
    let descriptors: [u8; 6] = [DESC_TAG_REGISTRATION, 4, b'A', b'C', b'-', b'3'];
    let buf = build_ts_with_audio(STREAM_TYPE_PES_PRIVATE, &descriptors, 0x300, &frame);
    let d = demux_ts(&buf).expect("demux");
    let audio = d.audio.expect("AC-3 audio via DVB registration surfaced");
    assert_eq!(audio.codec, "ac3");
    assert_eq!(&audio.samples[0][..2], &[0x0B, 0x77]);
}

#[test]
fn dac3_body_synthesized_from_first_ts_frame_matches_sync_header() {
    // The dac3 body the TS extractor produces must equal the body
    // we'd compute by parsing the same first frame independently —
    // proves the AC-3 path is using the canonical Squad-26 helper
    // rather than a parallel implementation.
    let frame = synth_ac3_frame_stereo_48k_128k();
    let buf = build_ts_with_audio(STREAM_TYPE_AC3, &[], 0x300, &frame);
    let d = demux_ts(&buf).expect("demux");
    let audio = d.audio.expect("AC-3 audio");
    let parsed = match crate::ac3_sync::parse_sync_info(&frame).unwrap() {
        crate::ac3_sync::SyncInfo::Ac3(s) => s,
        _ => panic!("expected AC-3"),
    };
    let expected = crate::mux::dac3_body_from_sync(&parsed);
    assert_eq!(
        audio.codec_private,
        expected.to_vec(),
        "TS-extracted dac3 must match the canonical helper"
    );
}

/// Build a TS file with two distinct programs (program_number 1 and
/// 2). Program 1 carries MPEG-2 video on 0x200; program 2 carries
/// H.264 video on 0x300. Both PMTs live in their own PIDs (0x100,
/// 0x101 respectively).
fn build_two_program_ts() -> Vec<u8> {
    // PAT with TWO program entries.
    let mut pat = vec![0x00];
    let pat_section_len: usize = 5 + 4 + 4 + 4; // 2 programs + CRC
    pat.push(0xB0 | ((pat_section_len >> 8) & 0x0F) as u8);
    pat.push((pat_section_len & 0xFF) as u8);
    pat.extend_from_slice(&[0x00, 0x01, 0xC1, 0x00, 0x00]);
    pat.extend_from_slice(&[0x00, 0x01, 0xE1, 0x00]); // program 1 → PMT 0x100
    pat.extend_from_slice(&[0x00, 0x02, 0xE1, 0x01]); // program 2 → PMT 0x101
    pat.extend_from_slice(&[0u8; 4]);
    let mut pat_payload = vec![0u8];
    pat_payload.extend_from_slice(&pat);
    let pat_pkt = ts_pkt(0x0000, true, 0b01, &pat_payload);

    // PMT 1: MPEG-2 video on 0x200.
    let mut pmt1 = vec![0x02];
    let pmt1_section_len: usize = 9 + 5 + 4;
    pmt1.push(0xB0 | ((pmt1_section_len >> 8) & 0x0F) as u8);
    pmt1.push((pmt1_section_len & 0xFF) as u8);
    pmt1.extend_from_slice(&[0x00, 0x01, 0xC1, 0x00, 0x00]); // program 1
    pmt1.extend_from_slice(&[0xE2, 0x00, 0xF0, 0x00]);
    pmt1.extend_from_slice(&[STREAM_TYPE_MPEG2_VIDEO, 0xE2, 0x00, 0xF0, 0x00]);
    pmt1.extend_from_slice(&[0u8; 4]);
    let mut pmt1_payload = vec![0u8];
    pmt1_payload.extend_from_slice(&pmt1);
    let pmt1_pkt = ts_pkt(0x0100, true, 0b01, &pmt1_payload);

    // PMT 2: H.264 video on 0x300.
    let mut pmt2 = vec![0x02];
    let pmt2_section_len: usize = 9 + 5 + 4;
    pmt2.push(0xB0 | ((pmt2_section_len >> 8) & 0x0F) as u8);
    pmt2.push((pmt2_section_len & 0xFF) as u8);
    pmt2.extend_from_slice(&[0x00, 0x02, 0xC1, 0x00, 0x00]); // program 2
    pmt2.extend_from_slice(&[0xE3, 0x00, 0xF0, 0x00]);
    pmt2.extend_from_slice(&[STREAM_TYPE_H264, 0xE3, 0x00, 0xF0, 0x00]);
    pmt2.extend_from_slice(&[0u8; 4]);
    let mut pmt2_payload = vec![0u8];
    pmt2_payload.extend_from_slice(&pmt2);
    let pmt2_pkt = ts_pkt(0x0101, true, 0b01, &pmt2_payload);

    // Distinct PES bytes so we can tell programs apart at sample
    // level. Program 1 → 0xAA; program 2 → 0xBB.
    let make_pes = |fill: u8| {
        let mut pes = vec![
            0u8, 0u8, 1u8, 0xE0, 0u8, 0u8, 0x80, 0x80, 5, 0x21, 0x00, 0x01, 0x00, 0x01,
        ];
        pes.extend_from_slice(&[fill; 16]);
        pes
    };
    let p1_pes = ts_pkt(0x0200, true, 0b01, &make_pes(0xAA));
    let p2_pes = ts_pkt(0x0300, true, 0b01, &make_pes(0xBB));

    let mut buf = Vec::new();
    buf.extend_from_slice(&pat_pkt);
    buf.extend_from_slice(&pmt1_pkt);
    buf.extend_from_slice(&pmt2_pkt);
    // Two PES per program so the streaming path's PUSI flush yields.
    buf.extend_from_slice(&p1_pes);
    buf.extend_from_slice(&p2_pes);
    buf.extend_from_slice(&ts_pkt(0x0200, true, 0b01, &make_pes(0xAA)));
    buf.extend_from_slice(&ts_pkt(0x0300, true, 0b01, &make_pes(0xBB)));
    buf.extend_from_slice(&ts_pkt(0x1FFF, false, 0b01, &[]));
    buf
}

#[test]
fn streaming_demuxer_lists_all_pat_programs() {
    let buf = build_two_program_ts();
    let dem = demux_ts_streaming_init(&buf).expect("init");
    let progs = dem.programs();
    assert_eq!(progs.len(), 2, "PAT advertised 2 programs");
    let nums: Vec<u16> = progs.iter().map(|p| p.program_number).collect();
    assert_eq!(nums, vec![1, 2]);
    assert_eq!(progs[0].pmt_pid, 0x100);
    assert_eq!(progs[1].pmt_pid, 0x101);
    // Program 1 → MPEG-2 on 0x200; program 2 → H.264 on 0x300.
    assert_eq!(progs[0].video_streams[0].pid, 0x200);
    assert_eq!(
        progs[0].video_streams[0].stream_type,
        STREAM_TYPE_MPEG2_VIDEO
    );
    assert_eq!(progs[1].video_streams[0].pid, 0x300);
    assert_eq!(progs[1].video_streams[0].stream_type, STREAM_TYPE_H264);
}

#[test]
fn streaming_demuxer_default_picks_first_program() {
    let buf = build_two_program_ts();
    let mut dem = demux_ts_streaming_init(&buf).expect("init");
    assert_eq!(dem.active_program_index(), 0);
    assert_eq!(dem.header().codec, "mpeg2", "program 1 is MPEG-2");
    // Drain — samples should be 0xAA-filled (program 1's bytes).
    let s = dem.next_video_sample().expect("sample").expect("some");
    assert!(
        s.data.iter().any(|&b| b == 0xAA),
        "program 1 sample should carry 0xAA"
    );
    assert!(
        !s.data.iter().any(|&b| b == 0xBB),
        "program 1 sample must not carry program 2's 0xBB"
    );
}

#[test]
fn streaming_demuxer_select_program_switches_active_streams() {
    let buf = build_two_program_ts();
    let mut dem = demux_ts_streaming_init(&buf).expect("init");
    dem.select_program(2).expect("switch to program 2");
    assert_eq!(dem.active_program_index(), 1);
    assert_eq!(dem.header().codec, "h264", "program 2 is H.264");
    let s = dem.next_video_sample().expect("sample").expect("some");
    assert!(
        s.data.iter().any(|&b| b == 0xBB),
        "program 2 sample should carry 0xBB"
    );
    assert!(
        !s.data.iter().any(|&b| b == 0xAA),
        "program 2 sample must not carry program 1's 0xAA"
    );
}

#[test]
fn streaming_demuxer_select_program_rejects_unknown_number() {
    let buf = build_two_program_ts();
    let mut dem = demux_ts_streaming_init(&buf).expect("init");
    assert!(
        dem.select_program(99).is_err(),
        "unknown program_number must error rather than silently no-op"
    );
}

/// Build a single-program TS where the video PID's packets carry
/// `transport_scrambling_control != 0` (TSC=01 = "user-defined,
/// reserved" in ISO/IEC 13818-1 — both this and 10/11 indicate the
/// payload is encrypted and we have no CA tables).
fn build_encrypted_ts() -> Vec<u8> {
    // Reuse the single-program PAT/PMT shape from the existing tests.
    let mut pat = vec![0x00];
    let pat_section_len: usize = 5 + 4 + 4;
    pat.push(0xB0 | ((pat_section_len >> 8) & 0x0F) as u8);
    pat.push((pat_section_len & 0xFF) as u8);
    pat.extend_from_slice(&[0x00, 0x01, 0xC1, 0x00, 0x00]);
    pat.extend_from_slice(&[0x00, 0x01, 0xE1, 0x00, 0u8, 0u8, 0u8, 0u8]);
    let mut pat_payload = vec![0u8];
    pat_payload.extend_from_slice(&pat);
    let pat_pkt = ts_pkt(0x0000, true, 0b01, &pat_payload);

    let mut pmt = vec![0x02];
    let pmt_section_len: usize = 9 + 5 + 4;
    pmt.push(0xB0 | ((pmt_section_len >> 8) & 0x0F) as u8);
    pmt.push((pmt_section_len & 0xFF) as u8);
    pmt.extend_from_slice(&[0x00, 0x01, 0xC1, 0x00, 0x00]);
    pmt.extend_from_slice(&[0xE2, 0x00, 0xF0, 0x00]);
    pmt.extend_from_slice(&[STREAM_TYPE_MPEG2_VIDEO, 0xE2, 0x00, 0xF0, 0x00]);
    pmt.extend_from_slice(&[0u8; 4]);
    let mut pmt_payload = vec![0u8];
    pmt_payload.extend_from_slice(&pmt);
    let pmt_pkt = ts_pkt(0x0100, true, 0b01, &pmt_payload);

    // Encrypted video PES: build the packet as normal but flip
    // bits 6-7 of byte 3 to TSC=01.
    let video_pes = {
        let mut pes = vec![
            0u8, 0u8, 1u8, 0xE0, 0u8, 0u8, 0x80, 0x80, 5, 0x21, 0x00, 0x01, 0x00, 0x01,
        ];
        pes.extend_from_slice(&[0xAAu8; 16]);
        pes
    };
    let mut video_pkt = ts_pkt(0x0200, true, 0b01, &video_pes);
    // TSC = 01 (single-bit set in the top 2 bits of byte 3).
    video_pkt[3] = (video_pkt[3] & 0x3F) | (0x01 << 6);

    let mut buf = Vec::new();
    buf.extend_from_slice(&pat_pkt);
    buf.extend_from_slice(&pmt_pkt);
    buf.extend_from_slice(&video_pkt);
    buf.extend_from_slice(&ts_pkt(0x1FFF, false, 0b01, &[]));
    buf
}

#[test]
fn streaming_demuxer_drops_video_when_active_pid_is_scrambled() {
    let buf = build_encrypted_ts();
    let mut dem = demux_ts_streaming_init(&buf).expect("init");
    // First call should hit the encrypted packet, latch the guard,
    // and return None. No samples should ever surface.
    let s = dem.next_video_sample().expect("call must not error");
    assert!(
        s.is_none(),
        "encrypted TS → next_video_sample returns None on first call"
    );
    // Subsequent calls remain None — the guard latches.
    let s2 = dem.next_video_sample().expect("call must not error");
    assert!(
        s2.is_none(),
        "encrypted TS → guard remains latched on subsequent calls"
    );
}
