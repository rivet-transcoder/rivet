// ---------------- Squad-37: AC-3 / E-AC-3 in TS, multi-program, encrypted ----------------

use super::super::pat_pmt::parse_pmt_streams;
use super::super::{
    AudioCodecKind, DESC_TAG_REGISTRATION, STREAM_TYPE_AAC_ADTS, STREAM_TYPE_AC3,
    STREAM_TYPE_EAC3, STREAM_TYPE_MPEG2_VIDEO, STREAM_TYPE_PES_PRIVATE, demux_ts,
};
use super::{build_ts_with_audio, synth_ac3_frame_stereo_48k_128k,
            synth_eac3_frame_stereo_48k_192bytes};

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
        !audio.samples.is_empty(),
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
