//! `Encoder::reset` on real NVENC hardware: one session, several streams.
//!
//! Requires the `nvidia` feature and a usable NVIDIA GPU at index 0; skips
//! cleanly otherwise. What it checks is the contract the chunked path leans
//! on after a reset: the packet queue is empty, the first packet of the new
//! stream is a keyframe **carrying its parameter sets** (the stitcher and the
//! rung invariant both read the SPS off the first packet of every chunk, the
//! way a fresh session's first packet has it), and every submitted frame
//! comes back as exactly one packet. Twice over, because "resettable once"
//! is not "reusable".
#![cfg(feature = "nvidia")]

use bytes::Bytes;
use codec::encode::nvenc::NvencEncoder;
use codec::encode::{Encoder, EncoderConfig};
use codec::frame::{ColorSpace, PixelFormat, VideoCodec, VideoFrame};

const W: u32 = 320;
const H: u32 = 240;

fn frame(pts: u64) -> VideoFrame {
    let luma = (W * H) as usize;
    let chroma = ((W / 2) * (H / 2)) as usize;
    let mut data = Vec::with_capacity(luma + 2 * chroma);
    for y in 0..H as usize {
        for x in 0..W as usize {
            data.push(((x + y + pts as usize * 5) % 256) as u8);
        }
    }
    data.extend(std::iter::repeat_n(128u8, 2 * chroma));
    VideoFrame::new(Bytes::from(data), W, H, PixelFormat::Yuv420p, ColorSpace::Bt709, pts)
}

/// Annex-B NAL unit types in a packet, in order.
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

fn drain(enc: &mut dyn Encoder) -> Vec<codec::encode::EncodedPacket> {
    let mut v = Vec::new();
    while let Some(p) = enc.receive_packet().unwrap() {
        v.push(p);
    }
    v
}

fn stream(enc: &mut dyn Encoder, first_pts: u64, n: u64) -> Vec<codec::encode::EncodedPacket> {
    let mut packets = Vec::new();
    for pts in first_pts..first_pts + n {
        enc.send_frame(&frame(pts)).unwrap();
        packets.extend(drain(enc));
    }
    enc.flush().unwrap();
    packets.extend(drain(enc));
    packets
}

fn check(codec: VideoCodec) {
    let hevc = codec == VideoCodec::H265;
    let cfg = EncoderConfig { width: W, height: H, codec, keyframe_interval: 1000, ..Default::default() };
    let mut enc = match NvencEncoder::new(cfg, 0) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("skip: no usable NVENC on GPU 0 ({e})");
            return;
        }
    };
    let (sps, pps, idr): (Vec<u8>, Vec<u8>, Vec<u8>) =
        if hevc { (vec![33], vec![34], vec![19, 20]) } else { (vec![7], vec![8], vec![5]) };

    let mut first_types = Vec::new();
    let mut first_headers: Vec<Vec<u8>> = Vec::new();
    for round in 0..3 {
        if round > 0 {
            enc.reset().expect("NVENC reset");
            assert!(enc.receive_packet().unwrap().is_none(), "nothing queued right after a reset");
        }
        let packets = stream(&mut enc, round * 100, 6);
        assert_eq!(packets.len(), 6, "round {round}: one packet per frame");
        assert!(packets[0].is_keyframe, "round {round}: first packet must be a keyframe");
        assert!(packets[1..].iter().all(|p| !p.is_keyframe), "round {round}: one IDR per stream at this GOP");
        assert_eq!(packets[0].pts, round * 100, "round {round}: the new stream's own timestamps");
        let types = nal_types(&packets[0].data, hevc);
        eprintln!("{codec:?} round {round}: first packet {} bytes, NAL types {types:?}", packets[0].data.len());
        assert!(types.iter().any(|t| idr.contains(t)), "round {round}: first packet is an IDR AU: {types:?}");
        assert!(types.iter().any(|t| sps.contains(t)), "round {round}: first packet carries the SPS: {types:?}");
        assert!(types.iter().any(|t| pps.contains(t)), "round {round}: first packet carries the PPS: {types:?}");
        first_types.push(types);
        first_headers.push(parameter_sets(&packets[0].data, hevc));
    }
    assert_eq!(first_types[0], first_types[1], "a reset stream opens exactly like a fresh session");
    assert_eq!(first_types[1], first_types[2]);
    // Byte-identical parameter sets: what a reset stream is given up front
    // is exactly what the driver wrote in-band for the fresh session.
    assert!(!first_headers[0].is_empty());
    assert_eq!(first_headers[0], first_headers[1], "reset stream's parameter sets differ from the session's");
    assert_eq!(first_headers[1], first_headers[2]);
    eprintln!("{codec:?}: parameter sets identical across 3 streams ({} bytes)", first_headers[0].len());
}

/// The bytes of a packet up to its first VCL NAL unit — the parameter sets.
fn parameter_sets(data: &[u8], hevc: bool) -> Vec<u8> {
    let mut i = 0;
    while i + 3 < data.len() {
        if data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1 {
            let t = if hevc { (data[i + 3] >> 1) & 0x3f } else { data[i + 3] & 0x1f };
            let vcl = if hevc { t < 32 } else { t <= 5 };
            if vcl {
                // Include a preceding zero byte of a 4-byte start code.
                let cut = if i > 0 && data[i - 1] == 0 { i - 1 } else { i };
                return data[..cut].to_vec();
            }
            i += 4;
        } else {
            i += 1;
        }
    }
    data.to_vec()
}

#[test]
fn nvenc_h264_reset_opens_a_new_stream_with_sps_pps_idr() {
    check(VideoCodec::H264);
}

#[test]
fn nvenc_h265_reset_opens_a_new_stream_with_vps_sps_pps_idr() {
    check(VideoCodec::H265);
}
