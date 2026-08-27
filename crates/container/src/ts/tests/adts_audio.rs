use super::super::audio::{AdtsHeader, decode_sample_rate_index, parse_adts_header, synthesize_asc};
use super::super::{demux_ts, STREAM_TYPE_AAC_ADTS, STREAM_TYPE_MPEG2_VIDEO};
use super::{build_adts_header_7, build_adts_header_9, ts_pkt};

// ---------------- AAC-ADTS / ASC unit tests (Squad-27) ----------------

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

/// `channel_configuration = 0`: the layout lives in a PCE at the head of
/// the raw data block (ffmpeg writes 7.1 this way). The TS path must read
/// it, count eight channels, and hand the MP4 an ASC that carries the same
/// PCE — the ASC's `byte_alignment()` pads differently from the in-band
/// one, so the ASC is re-serialised, not copied. Before 2026-08-27 this
/// stream bailed ("PCE-defined; not supported") and the job went video-only.
#[test]
fn adts_channel_configuration_zero_takes_the_layout_from_the_in_band_pce() {
    use crate::aac_asc::{BitWriter, PceElement, ProgramConfig, parse_aac_asc, write_pce};
    let pce = ProgramConfig {
        element_instance_tag: 0,
        object_type: 1,
        sampling_frequency_index: 3,
        front: vec![PceElement { is_cpe: false, tag: 0 }, PceElement { is_cpe: true, tag: 0 }],
        side: vec![PceElement { is_cpe: true, tag: 1 }],
        back: vec![PceElement { is_cpe: true, tag: 2 }],
        lfe: vec![0],
        ..Default::default()
    };
    // Raw data block: id_syn_ele = ID_PCE (5), the PCE, then filler.
    let mut bw = BitWriter::new();
    bw.bits(5, 3);
    write_pce(&mut bw, &pce);
    let mut block = bw.into_bytes();
    block.extend_from_slice(&[0x5Au8; 40]);
    let mut frame = build_adts_header_7(1, 3, 0, 7 + block.len()).to_vec();
    frame.extend_from_slice(&block);
    let mut es = frame.clone();
    es.extend_from_slice(&frame);
    let buf = super::build_ts_with_audio(STREAM_TYPE_AAC_ADTS, &[], 0x300, &es);

    let d = demux_ts(&buf).expect("demux");
    let audio = d.audio.expect("a PCE-described AAC track surfaces");
    assert_eq!(audio.codec, "aac");
    assert_eq!(audio.channels, 8, "C + L/R + Ls/Rs + Lb/Rb + LFE");
    assert_eq!(audio.sample_rate, 48_000);
    assert_eq!(audio.samples.len(), 2);
    assert_eq!(audio.samples[0], block, "frames stay verbatim, in-band PCE included");
    // The ASC says channelConfiguration = 0 and carries the same PCE.
    let parsed = parse_aac_asc(&audio.asc).expect("the synthesised ASC parses");
    assert_eq!(parsed.aot, 2);
    assert_eq!(parsed.sample_rate, 48_000);
    assert_eq!(parsed.channels, 8);
    assert_eq!(parsed.pce.as_ref(), Some(&pce));
    assert_eq!(audio.asc[1] & 0x78, 0, "channelConfiguration field is 0");
    assert_ne!(&audio.asc[2..], &block[..audio.asc.len() - 2], "re-serialised, not copied");

    // channel_configuration = 0 with no PCE in the block is refused by name.
    let mut bad = build_adts_header_7(1, 3, 0, 7 + 40).to_vec();
    bad.extend_from_slice(&[0x00u8; 40]);
    let buf = super::build_ts_with_audio(STREAM_TYPE_AAC_ADTS, &[], 0x300, &bad);
    let err = demux_ts(&buf).err().expect("no PCE → error").to_string();
    assert!(err.contains("PCE"), "{err}");

    // Table 1.19 config 7 is eight channels too.
    let mut seven = build_adts_header_7(1, 3, 7, 7 + 8).to_vec();
    seven.extend_from_slice(&[0x21u8; 8]);
    let buf = super::build_ts_with_audio(STREAM_TYPE_AAC_ADTS, &[], 0x300, &seven);
    assert_eq!(demux_ts(&buf).unwrap().audio.unwrap().channels, 8);
}
