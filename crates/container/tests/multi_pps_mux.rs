//! Streams carrying more than one parameter set of a kind, through the MP4
//! writer's out-of-band (`avc1` / `hvc1`) sample entries.
//!
//! The fixtures are ffmpeg's 64x64 H.264 and H.265 rewritten by
//! `tests/fixtures/multi_pps/make_fixtures.py`: `two_pps.h264` codes every
//! odd picture with a second PPS (id 1) that its access unit re-sends in-band,
//! `two_pps.h265` sends an unused PPS 1 beside PPS 0, and `conflict.h264`
//! re-sends PPS 0 mid-stream with a different `pic_init_qp_minus26`. Each is
//! muxed as the encoder pump would (one packet per access unit) and the
//! config box read back. The files are kept in `CARGO_TARGET_TMPDIR` as
//! `<fixture>.mp4` for decoding with other tools.

use std::path::Path;

use bytes::Bytes;
use container::mux::Av1Mp4Muxer;
use container::nal_mux::{NalMuxCodec, sample_is_keyframe, split_annexb_nals};
use container::streaming::demux_streaming;
use frame::{EncodedPacket, VideoCodec};

fn fixture(name: &str) -> Vec<u8> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/multi_pps").join(name);
    std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn nal_type(nal: &[u8], codec: NalMuxCodec) -> u8 {
    match codec {
        NalMuxCodec::H264 => nal[0] & 0x1F,
        NalMuxCodec::H265 => (nal[0] >> 1) & 0x3F,
    }
}

/// A NAL unit without the zero byte a following 4-byte start code leaves it.
fn trimmed(nal: &[u8]) -> Vec<u8> {
    let end = nal.len() - nal.iter().rev().take_while(|&&b| b == 0).count();
    nal[..end].to_vec()
}

/// The fixture's access units (split at its delimiters), as Annex-B.
fn access_units(data: &[u8], codec: NalMuxCodec) -> Vec<Vec<u8>> {
    let aud = match codec {
        NalMuxCodec::H264 => 9,
        NalMuxCodec::H265 => 35,
    };
    let mut units: Vec<Vec<u8>> = Vec::new();
    for nal in split_annexb_nals(data) {
        if nal_type(nal, codec) == aud || units.is_empty() {
            units.push(Vec::new());
        }
        let unit = units.last_mut().unwrap();
        unit.extend_from_slice(&[0, 0, 0, 1]);
        unit.extend_from_slice(nal);
    }
    units
}

/// The fixture's distinct parameter sets of NAL type `kind`, in arrival order.
fn sets_in(data: &[u8], codec: NalMuxCodec, kind: u8) -> Vec<Vec<u8>> {
    let mut sets: Vec<Vec<u8>> = Vec::new();
    for nal in split_annexb_nals(data) {
        let nal = trimmed(nal);
        if nal_type(&nal, codec) == kind && !sets.contains(&nal) {
            sets.push(nal);
        }
    }
    sets
}

fn mux(name: &str, codec: VideoCodec, nal_codec: NalMuxCodec) -> Bytes {
    let mut m = Av1Mp4Muxer::new_with_codec(64, 64, 25.0, codec).unwrap();
    for (i, au) in access_units(&fixture(name), nal_codec).into_iter().enumerate() {
        let is_keyframe = sample_is_keyframe(&au, nal_codec);
        m.add_packet(EncodedPacket { data: Bytes::from(au), pts: i as u64, is_keyframe })
            .unwrap();
    }
    let mp4 = m.finalize().unwrap();
    std::fs::write(Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("{name}.mp4")), &mp4).unwrap();
    mp4
}

/// The body of the first box of type `tag`.
fn box_body<'a>(mp4: &'a [u8], tag: &[u8; 4]) -> &'a [u8] {
    let at = mp4.windows(4).position(|w| w == tag).expect("box present");
    let size = u32::from_be_bytes(mp4[at - 4..at].try_into().unwrap()) as usize;
    &mp4[at + 4..at - 4 + size]
}

/// `count` length-prefixed (u16) NAL units from `body[*at..]`.
fn nal_list(body: &[u8], at: &mut usize, count: usize) -> Vec<Vec<u8>> {
    (0..count)
        .map(|_| {
            let len = u16::from_be_bytes([body[*at], body[*at + 1]]) as usize;
            let nal = trimmed(&body[*at + 2..*at + 2 + len]);
            *at += 2 + len;
            nal
        })
        .collect()
}

/// `avcC`'s PPS list (ISO 14496-15 §5.3.3.1).
fn avcc_pps(mp4: &[u8]) -> Vec<Vec<u8>> {
    let body = box_body(mp4, b"avcC");
    let mut at = 6;
    nal_list(body, &mut at, (body[5] & 0x1F) as usize);
    let count = body[at] as usize;
    at += 1;
    nal_list(body, &mut at, count)
}

/// `hvcC`'s array of NAL type `kind` (ISO 14496-15 §8.3.3.1).
fn hvcc_array(mp4: &[u8], kind: u8) -> Vec<Vec<u8>> {
    let body = box_body(mp4, b"hvcC");
    let mut at = 23;
    for _ in 0..body[22] {
        let array_kind = body[at] & 0x3F;
        let count = u16::from_be_bytes([body[at + 1], body[at + 2]]) as usize;
        at += 3;
        let nals = nal_list(body, &mut at, count);
        if array_kind == kind {
            return nals;
        }
    }
    Vec::new()
}

/// Whether any sample after the first carries a NAL of type `kind` — the
/// demuxer puts the config box's sets in front of the first one only.
fn later_samples_carry(mp4: &[u8], codec: NalMuxCodec, kind: u8) -> bool {
    let mut d = demux_streaming(mp4).expect("our demuxer reads what we wrote");
    let mut n = 0;
    let mut found = false;
    while let Some(s) = d.next_video_sample().expect("next_video_sample") {
        found |= n > 0 && split_annexb_nals(&s.data).iter().any(|nal| nal_type(nal, codec) == kind);
        n += 1;
    }
    assert_eq!(n, 12, "every access unit is a sample");
    found
}

#[test]
fn a_second_h264_pps_goes_in_avcc_after_the_first() {
    let mp4 = mux("two_pps.h264", VideoCodec::H264, NalMuxCodec::H264);
    // The IDR's access unit brings PPS 1 before PPS 0.
    let arrived = sets_in(&fixture("two_pps.h264"), NalMuxCodec::H264, 8);
    assert_eq!(arrived.len(), 2);
    assert_eq!(avcc_pps(&mp4), vec![arrived[1].clone(), arrived[0].clone()], "id order, both ids");
    assert!(!later_samples_carry(&mp4, NalMuxCodec::H264, 8), "avc1: the re-sent PPS 1 is out of band");
}

#[test]
fn a_changed_h264_pps_keeps_the_first_under_its_id() {
    let mp4 = mux("conflict.h264", VideoCodec::H264, NalMuxCodec::H264);
    let arrived = sets_in(&fixture("conflict.h264"), NalMuxCodec::H264, 8);
    assert_eq!(arrived.len(), 2, "PPS 0 and its changed re-send");
    assert_eq!(avcc_pps(&mp4), vec![arrived[0].clone()], "one PPS per id: the first");
}

#[test]
fn a_second_h265_pps_goes_in_hvcc_after_the_first() {
    let mp4 = mux("two_pps.h265", VideoCodec::H265, NalMuxCodec::H265);
    let arrived = sets_in(&fixture("two_pps.h265"), NalMuxCodec::H265, 34);
    assert_eq!(arrived.len(), 2);
    assert_eq!(hvcc_array(&mp4, 34), vec![arrived[1].clone(), arrived[0].clone()], "id order, both ids");
    assert_eq!(hvcc_array(&mp4, 32).len(), 1);
    assert_eq!(hvcc_array(&mp4, 33).len(), 1);
    assert!(!later_samples_carry(&mp4, NalMuxCodec::H265, 34), "hvc1: array_completeness=1");
}
