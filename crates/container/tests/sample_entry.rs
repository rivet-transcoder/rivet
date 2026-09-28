//! The single-file MP4 writer's H.264 / H.265 sample entry across the packet
//! shapes the encoder backends hand it.
//!
//! Every backend — the software `h26x` encoder, QSV, NVENC, AMF — reaches the
//! writer as Annex-B packets, one access unit each, and what differs is the
//! shape: whether access-unit delimiters are there (QSV's H.265 writes none),
//! whether the parameter sets come once or again at every IDR (the software
//! encoder and NVENC's repeated headers), and whether several ids of a kind
//! appear. Whatever the shape, one encoder's stream is written `avc1` /
//! `hvc1` — the entry Safari's `<video>` element on iOS requires, where it
//! refuses `avc3` — with a config box that parses and holds exactly the
//! stream's parameter sets, `hvcC`'s arrays complete, and no set left in a
//! sample.
//!
//! The streams are the `multi_pps` fixtures (ffmpeg's 64x64 H.264 and H.265,
//! AUDs and the sets at every IDR), reshaped here.

use std::path::Path;

use bytes::Bytes;
use container::mux::Av1Mp4Muxer;
use container::nal_mux::{NalMuxCodec, sample_is_keyframe, split_annexb_nals};
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

fn is_aud(nal: &[u8], codec: NalMuxCodec) -> bool {
    nal_type(nal, codec) == if codec == NalMuxCodec::H264 { 9 } else { 35 }
}

fn is_param_set(nal: &[u8], codec: NalMuxCodec) -> bool {
    match codec {
        NalMuxCodec::H264 => matches!(nal_type(nal, codec), 7 | 8),
        NalMuxCodec::H265 => matches!(nal_type(nal, codec), 32..=34),
    }
}

/// A NAL unit without the zero byte a following 4-byte start code leaves it.
fn trimmed(nal: &[u8]) -> Vec<u8> {
    nal[..nal.len() - nal.iter().rev().take_while(|&&b| b == 0).count()].to_vec()
}

/// The fixture's access units (split at its delimiters), each a list of NALs.
fn access_units(data: &[u8], codec: NalMuxCodec) -> Vec<Vec<Vec<u8>>> {
    let mut units: Vec<Vec<Vec<u8>>> = Vec::new();
    for nal in split_annexb_nals(data) {
        if is_aud(nal, codec) || units.is_empty() {
            units.push(Vec::new());
        }
        units.last_mut().unwrap().push(trimmed(nal));
    }
    units
}

/// How a backend packages the same access units.
#[derive(Debug, Clone, Copy)]
enum Shape {
    /// As the fixture has them: AUDs, the sets at every IDR.
    Repeated,
    /// No delimiters (QSV's H.265).
    NoDelimiters,
    /// The sets only in the first access unit.
    SetsOnce,
}

fn packets(units: &[Vec<Vec<u8>>], codec: NalMuxCodec, shape: Shape) -> Vec<Vec<u8>> {
    units
        .iter()
        .enumerate()
        .map(|(i, unit)| {
            let mut au = Vec::new();
            for nal in unit {
                let drop = match shape {
                    Shape::Repeated => false,
                    Shape::NoDelimiters => is_aud(nal, codec),
                    Shape::SetsOnce => i > 0 && is_param_set(nal, codec),
                };
                if !drop {
                    au.extend_from_slice(&[0, 0, 0, 1]);
                    au.extend_from_slice(nal);
                }
            }
            au
        })
        .collect()
}

/// The stream's distinct parameter sets, sorted.
fn stream_sets(units: &[Vec<Vec<u8>>], codec: NalMuxCodec) -> Vec<Vec<u8>> {
    let mut sets: Vec<Vec<u8>> = units.iter().flatten().filter(|n| is_param_set(n, codec)).cloned().collect();
    sets.sort();
    sets.dedup();
    sets
}

/// The visual sample entry: its fourcc, and its body from the config box on.
fn sample_entry(mp4: &[u8]) -> ([u8; 4], &[u8]) {
    let at = mp4.windows(4).position(|w| w == b"stsd").expect("an stsd box") + 4 + 8;
    let size = u32::from_be_bytes(mp4[at..at + 4].try_into().unwrap()) as usize;
    (mp4[at + 4..at + 8].try_into().unwrap(), &mp4[at + 8 + 78..at + size])
}

/// The config box's sets, sorted, and each `hvcC` array's completeness bit —
/// read field by field to its last byte, which is the check that it parses.
fn config_record(mp4: &[u8], codec: NalMuxCodec) -> (Vec<Vec<u8>>, Vec<u8>) {
    let (_, children) = sample_entry(mp4);
    let size = u32::from_be_bytes(children[..4].try_into().unwrap()) as usize;
    let tag: &[u8; 4] = if codec == NalMuxCodec::H264 { b"avcC" } else { b"hvcC" };
    assert_eq!(&children[4..8], tag, "the config box comes first");
    let body = &children[8..size];
    let mut sets = Vec::new();
    let mut complete = Vec::new();
    let mut take = |at: &mut usize| {
        let len = u16::from_be_bytes([body[*at], body[*at + 1]]) as usize;
        sets.push(trimmed(&body[*at + 2..*at + 2 + len]));
        *at += 2 + len;
    };
    let end = match codec {
        NalMuxCodec::H264 => {
            let mut at = 6;
            for _ in 0..body[5] & 0x1F {
                take(&mut at);
            }
            let n = body[at];
            at += 1;
            for _ in 0..n {
                take(&mut at);
            }
            if !matches!(body[1], 66 | 77 | 88) {
                at += 4;
            }
            at
        }
        NalMuxCodec::H265 => {
            let mut at = 23;
            for _ in 0..body[22] {
                complete.push(body[at] >> 7);
                let n = u16::from_be_bytes([body[at + 1], body[at + 2]]);
                at += 3;
                for _ in 0..n {
                    take(&mut at);
                }
            }
            at
        }
    };
    assert_eq!(end, body.len(), "the record ends where its fields do");
    sets.sort();
    (sets, complete)
}

/// Whether any sample carries a parameter set: the mdat read as the
/// length-prefixed NAL units its samples are.
fn samples_carry_sets(mp4: &[u8], codec: NalMuxCodec) -> bool {
    let at = mp4.windows(4).position(|w| w == b"mdat").expect("an mdat");
    let size = u32::from_be_bytes(mp4[at - 4..at].try_into().unwrap()) as usize;
    let mut data = &mp4[at + 4..at - 4 + size];
    while data.len() >= 4 {
        let len = u32::from_be_bytes(data[..4].try_into().unwrap()) as usize;
        if is_param_set(&data[4..4 + len], codec) {
            return true;
        }
        data = &data[4 + len..];
    }
    false
}

fn mux(packets: Vec<Vec<u8>>, codec: VideoCodec, nal_codec: NalMuxCodec) -> Bytes {
    let mut m = Av1Mp4Muxer::new_with_codec(64, 64, 25.0, codec).unwrap();
    for (i, au) in packets.into_iter().enumerate() {
        let is_keyframe = sample_is_keyframe(&au, nal_codec);
        m.add_packet(EncodedPacket { data: Bytes::from(au), pts: i as u64, is_keyframe }).unwrap();
    }
    m.finalize().unwrap()
}

#[test]
fn every_packet_shape_writes_avc1_and_hvc1_with_the_streams_sets() {
    let cases = [
        ("two_pps.h264", VideoCodec::H264, NalMuxCodec::H264, *b"avc1"),
        ("two_pps.h265", VideoCodec::H265, NalMuxCodec::H265, *b"hvc1"),
    ];
    for (name, codec, nal_codec, entry) in cases {
        let units = access_units(&fixture(name), nal_codec);
        let want = stream_sets(&units, nal_codec);
        for shape in [Shape::Repeated, Shape::NoDelimiters, Shape::SetsOnce] {
            let mp4 = mux(packets(&units, nal_codec, shape), codec, nal_codec);
            let case = format!("{name} {shape:?}");
            assert_eq!(sample_entry(&mp4).0, entry, "{case}: sample entry");
            let (sets, complete) = config_record(&mp4, nal_codec);
            assert_eq!(sets, want, "{case}: the config box holds the stream's sets");
            if codec == VideoCodec::H265 {
                assert_eq!(complete, vec![1, 1, 1], "{case}: hvc1 arrays are complete");
            }
            assert!(!samples_carry_sets(&mp4, nal_codec), "{case}: no set in band");
        }
    }
}

#[test]
fn a_set_changed_under_its_id_goes_in_band_under_avc3() {
    // `conflict.h264` re-sends PPS 0 before picture 8 with other contents.
    let units = access_units(&fixture("conflict.h264"), NalMuxCodec::H264);
    let mp4 = mux(packets(&units, NalMuxCodec::H264, Shape::Repeated), VideoCodec::H264, NalMuxCodec::H264);
    assert_eq!(&sample_entry(&mp4).0, b"avc3", "the sets change: avc3");
    assert!(samples_carry_sets(&mp4, NalMuxCodec::H264), "from the change on, in band");
}
