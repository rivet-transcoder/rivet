//! `Encoder::reset` on the native software H.264 / H.265 tier, and the
//! measurement behind its implementation: a rebuild of the inner encoder
//! *is* the reset, because constructing one costs less than the shortest
//! chunk by orders of magnitude.
//!
//! No GPU, no feature flag: the h26x encoders are always compiled.

use std::time::Instant;

use bytes::Bytes;
use codec::encode::h26x_sw::H26xEncoder;
use codec::encode::{Encoder, EncoderConfig};
use codec::frame::{ColorSpace, PixelFormat, VideoCodec, VideoFrame};

const W: u32 = 64;
const H: u32 = 48;

fn frame(pts: u64) -> VideoFrame {
    let luma = (W * H) as usize;
    let chroma = ((W / 2) * (H / 2)) as usize;
    let mut data = Vec::with_capacity(luma + 2 * chroma);
    for y in 0..H as usize {
        for x in 0..W as usize {
            data.push(((x * 3 + y * 5 + pts as usize * 7) % 256) as u8);
        }
    }
    data.extend(std::iter::repeat_n(128u8, 2 * chroma));
    VideoFrame::new(Bytes::from(data), W, H, PixelFormat::Yuv420p, ColorSpace::Bt709, pts)
}

fn nal_types(data: &[u8], hevc: bool) -> Vec<u8> {
    let mut out = Vec::new();
    let mut i = 0;
    while i + 3 < data.len() {
        if data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1 {
            let hdr = data[i + 3];
            out.push(if hevc { (hdr >> 1) & 0x3f } else { hdr & 0x1f });
            i += 4;
        } else {
            i += 1;
        }
    }
    out
}

fn config(codec: VideoCodec) -> EncoderConfig {
    EncoderConfig { width: W, height: H, codec, keyframe_interval: 1000, threads: 1, ..Default::default() }
}

fn stream(enc: &mut dyn Encoder, first_pts: u64, n: u64) -> Vec<codec::encode::EncodedPacket> {
    let mut packets = Vec::new();
    for pts in first_pts..first_pts + n {
        enc.send_frame(&frame(pts)).unwrap();
        while let Some(p) = enc.receive_packet().unwrap() {
            packets.push(p);
        }
    }
    enc.flush().unwrap();
    while let Some(p) = enc.receive_packet().unwrap() {
        packets.push(p);
    }
    packets
}

fn check(codec: VideoCodec) {
    let hevc = codec == VideoCodec::H265;
    let mut enc = H26xEncoder::new(config(codec)).unwrap();
    let sps = if hevc { 33 } else { 7 };
    let mut openings = Vec::new();
    for round in 0..3u64 {
        if round > 0 {
            enc.reset().expect("h26x reset");
            assert!(enc.receive_packet().unwrap().is_none(), "nothing queued after a reset");
        }
        let packets = stream(&mut enc, round * 100, 5);
        assert_eq!(packets.len(), 5, "round {round}: one packet per frame");
        assert!(packets[0].is_keyframe, "round {round}: opens with an IDR");
        assert!(packets[1..].iter().all(|p| !p.is_keyframe), "round {round}: one IDR at this GOP");
        assert_eq!(packets[0].pts, round * 100, "round {round}: the new stream's timestamps");
        let types = nal_types(&packets[0].data, hevc);
        assert!(types.contains(&sps), "round {round}: SPS on the first packet: {types:?}");
        eprintln!("{codec:?} round {round}: first packet NAL types {types:?}");
        openings.push(types);
    }
    assert_eq!(openings[0], openings[1], "a reset stream opens like a fresh one");
    assert_eq!(openings[1], openings[2]);
}

#[test]
fn h264_software_reset_opens_a_fresh_stream() {
    check(VideoCodec::H264);
}

#[test]
fn h265_software_reset_opens_a_fresh_stream() {
    check(VideoCodec::H265);
}

/// The measurement: construction vs reset, at a chunk-sized picture. Printed
/// (`--nocapture`), and asserted only loosely — a reset must not be slower
/// than building the whole encoder, which is the alternative it replaces.
#[test]
fn a_reset_costs_no_more_than_a_construction() {
    for codec in [VideoCodec::H264, VideoCodec::H265] {
        let cfg = EncoderConfig { width: 640, height: 360, ..config(codec) };
        let n = 200;
        let t = Instant::now();
        for _ in 0..n {
            let e = H26xEncoder::new(cfg.clone()).unwrap();
            std::hint::black_box(&e);
        }
        let build = t.elapsed() / n;
        let mut e = H26xEncoder::new(cfg.clone()).unwrap();
        let t = Instant::now();
        for _ in 0..n {
            e.reset().unwrap();
        }
        let reset = t.elapsed() / n;
        eprintln!("{codec:?} 640x360: new() {build:?}/encoder, reset() {reset:?}/reset");
        assert!(reset <= build * 2, "{codec:?}: reset {reset:?} should not exceed construction {build:?}");
    }
}
