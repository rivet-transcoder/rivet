// ---------------- MPEG audio (MP3 / MP2) in TS: stream types 0x03 / 0x04 ----------------

use super::super::pat_pmt::parse_pmt_streams;
use super::super::{
    AudioCodecKind, STREAM_TYPE_MPEG1_AUDIO, STREAM_TYPE_MPEG2_AUDIO, STREAM_TYPE_MPEG2_VIDEO, demux_ts,
};
use super::build_ts_with_audio;

/// A syntactically valid MPEG audio frame for `header`, the body zero.
fn frame(header: [u8; 4]) -> Vec<u8> {
    let len = crate::mp3::FrameHeader::parse(&header).unwrap().frame_len();
    let mut f = vec![0u8; len];
    f[..4].copy_from_slice(&header);
    f
}

#[test]
fn pmt_walker_classifies_mpeg1_and_mpeg2_audio_stream_types() {
    let mut pmt = vec![0x02];
    let pmt_section_len: usize = 9 + 15 + 4;
    pmt.push(0xB0 | ((pmt_section_len >> 8) & 0x0F) as u8);
    pmt.push((pmt_section_len & 0xFF) as u8);
    pmt.extend_from_slice(&[0x00, 0x01, 0xC1, 0x00, 0x00]);
    pmt.extend_from_slice(&[0xE2, 0x00, 0xF0, 0x00]);
    pmt.extend_from_slice(&[STREAM_TYPE_MPEG2_VIDEO, 0xE2, 0x00, 0xF0, 0x00]);
    pmt.extend_from_slice(&[STREAM_TYPE_MPEG1_AUDIO, 0xE3, 0x00, 0xF0, 0x00]);
    pmt.extend_from_slice(&[STREAM_TYPE_MPEG2_AUDIO, 0xE4, 0x00, 0xF0, 0x00]);
    pmt.extend_from_slice(&[0u8; 4]);
    let (video, audio) = parse_pmt_streams(&pmt).expect("parse");
    assert_eq!(video.len(), 1);
    assert_eq!(
        audio.iter().map(|a| (a.pid, a.stream_type, a.kind)).collect::<Vec<_>>(),
        vec![
            (0x300, 0x03, AudioCodecKind::MpegAudio),
            (0x400, 0x04, AudioCodecKind::MpegAudio),
        ]
    );
}

#[test]
fn mp3_frames_in_a_transport_stream_come_out_one_per_sample() {
    // MPEG-1 Layer III, 128 kbps, 44.1 kHz, joint stereo; one frame padded.
    let (a, b) = (frame([0xFF, 0xFB, 0x90, 0x44]), frame([0xFF, 0xFB, 0x92, 0x44]));
    let es = [a.clone(), b.clone(), a.clone()].concat();
    let d = demux_ts(&build_ts_with_audio(STREAM_TYPE_MPEG1_AUDIO, &[], 0x300, &es)).expect("demux");
    let audio = d.audio.expect("MP3 audio surfaced");
    assert_eq!((audio.codec.as_str(), audio.sample_rate, audio.channels, audio.timescale), ("mp3", 44_100, 2, 44_100));
    assert_eq!(audio.samples, vec![a.clone(), b, a]);
    assert_eq!(audio.durations, vec![1152; 3]);
}

#[test]
fn mpeg2_layer_ii_is_labelled_mp2() {
    // MPEG-2 Layer II, 64 kbps, 24 kHz, mono.
    let f = frame([0xFF, 0xF5, 0x84, 0xC0]);
    let es = [f.clone(), f.clone()].concat();
    let d = demux_ts(&build_ts_with_audio(STREAM_TYPE_MPEG2_AUDIO, &[], 0x300, &es)).expect("demux");
    let audio = d.audio.expect("MP2 audio surfaced");
    assert_eq!((audio.codec.as_str(), audio.sample_rate, audio.channels), ("mp2", 24_000, 1));
    assert_eq!(audio.durations, vec![1152; 2]);
}
