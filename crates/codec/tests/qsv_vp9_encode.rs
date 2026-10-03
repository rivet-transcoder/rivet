//! VP9 encoded on an Intel card (QSV, `MFX_CODEC_VP9`), checked against
//! rivet's own VP9 decoder and the source.
//!
//! For profile 0 (8-bit, NV12) and profile 2 (10-bit, P010):
//!
//! - every frame in comes out as one packet, every packet decodes through
//!   rivet's own decoder (`crates/vp9`, bit-exact on the WebM project's
//!   vectors) to one picture of the source's size and depth;
//! - the first packet is a key frame, and every packet's sync flag is what
//!   its own header says (a key frame where the GOP puts one);
//! - each picture's luma PSNR against its source frame clears a floor;
//! - the packets go through rivet's WebM and MP4 muxers and come back out
//!   of its demuxer unchanged;
//! - QSV's own decoder decodes the stream byte for byte as rivet's does.
//!
//! And a constant-rate rung (`rate=cbr`) holds its average within 15%.
//!
//! Needs the `qsv` feature and an Intel card with VP9 encode (Arc A-series,
//! Meteor Lake). Without one it prints `SKIPPED` and passes — unless
//! `RIVET_REQUIRE_QSV=1`, which the Intel CI runner sets: there a skip is a
//! failure.
#![cfg(feature = "qsv")]

use bytes::Bytes;
use codec::decode::Decoder;
use codec::encode::tuning::{EncodeOverrides, RateMode};
use codec::encode::{EncodedPacket, EncoderBackend, EncoderConfig, select_encoder};
use codec::frame::{ColorSpace, PixelFormat, StreamInfo, VideoCodec, VideoFrame};

const W: u32 = 640;
const H: u32 = 360;
const FPS: f64 = 30.0;
const GOP: u32 = 30;

fn intel_present() -> bool {
    let present = codec::gpu::detect_gpus().iter().any(|g| g.vendor == codec::gpu::GpuVendor::Intel);
    if !present {
        assert!(std::env::var("RIVET_REQUIRE_QSV").ok().as_deref() != Some("1"), "RIVET_REQUIRE_QSV=1 and no Intel GPU");
        eprintln!("SKIPPED: no Intel GPU");
    }
    present
}

/// Frame `t`: a scrolling ramp, a moving box, textured chroma — motion and
/// detail for an inter coder, nothing a decent encoder should miss by much.
fn source(t: u64, ten_bit: bool) -> VideoFrame {
    let (w, h) = (W as usize, H as usize);
    let mut v: Vec<u16> = Vec::with_capacity(w * h * 3 / 2);
    for y in 0..h {
        for x in 0..w {
            let in_box = (x as i64 - 40 - 6 * t as i64).rem_euclid(w as i64) < 96 && (120..240).contains(&y);
            v.push(if in_box { 220 } else { (24 + (x / 2 + y + 2 * t as usize) % 200) as u16 });
        }
    }
    for c in 0..2 {
        for y in 0..h / 2 {
            for x in 0..w / 2 {
                v.push((80 + (x * (c + 1) / 2 + y / 2 + t as usize) % 96) as u16);
            }
        }
    }
    let (data, format) = if ten_bit {
        (v.iter().flat_map(|s| (s << 2).to_le_bytes()).collect::<Vec<u8>>(), PixelFormat::Yuv420p10le)
    } else {
        (v.iter().map(|&s| s as u8).collect(), PixelFormat::Yuv420p)
    };
    VideoFrame::new(Bytes::from(data), W, H, format, ColorSpace::Bt709, t)
}

fn config(ten_bit: bool, overrides: EncodeOverrides) -> EncoderConfig {
    EncoderConfig {
        width: W,
        height: H,
        frame_rate: FPS,
        codec: VideoCodec::Vp9,
        keyframe_interval: GOP,
        pixel_format: if ten_bit { PixelFormat::Yuv420p10le } else { PixelFormat::Yuv420p },
        overrides,
        ..Default::default()
    }
}

fn encode(cfg: EncoderConfig, frames: u64, ten_bit: bool) -> Vec<EncodedPacket> {
    let mut enc = select_encoder(cfg, Some(EncoderBackend::Qsv)).expect("QSV VP9 encoder on this card");
    let mut out = Vec::new();
    for t in 0..frames {
        enc.send_frame(&source(t, ten_bit)).expect("send_frame");
        while let Some(p) = enc.receive_packet().expect("packet") {
            out.push(p);
        }
    }
    enc.flush().expect("flush");
    while let Some(p) = enc.receive_packet().expect("packet") {
        out.push(p);
    }
    out
}

fn info(ten_bit: bool) -> StreamInfo {
    StreamInfo {
        codec: "vp9".into(),
        width: W,
        height: H,
        frame_rate: FPS,
        duration: 0.0,
        pixel_format: if ten_bit { PixelFormat::Yuv420p10le } else { PixelFormat::Yuv420p },
        color_space: ColorSpace::Bt709,
        total_frames: 0,
        bitrate: 0,
        color_metadata: Default::default(),
    }
}

fn decode(mut dec: Box<dyn Decoder>, packets: &[Vec<u8>]) -> Vec<VideoFrame> {
    let mut out = Vec::new();
    for p in packets {
        dec.push_sample(p).expect("decode");
        while let Some(f) = dec.decode_next().unwrap() {
            out.push(f);
        }
    }
    dec.finish().unwrap();
    while let Some(f) = dec.decode_next().unwrap() {
        out.push(f);
    }
    out
}

/// Luma PSNR of `got` against `want`, on the sample scale of the format.
fn luma_psnr(got: &VideoFrame, want: &VideoFrame) -> f64 {
    let n = (W * H) as usize;
    let (se, peak) = if got.format == PixelFormat::Yuv420p10le {
        let se: f64 = (0..n)
            .map(|i| {
                let a = f64::from(u16::from_le_bytes([got.data[2 * i], got.data[2 * i + 1]]));
                let b = f64::from(u16::from_le_bytes([want.data[2 * i], want.data[2 * i + 1]]));
                (a - b) * (a - b)
            })
            .sum();
        (se, 1023.0)
    } else {
        let se: f64 = (0..n).map(|i| (f64::from(got.data[i]) - f64::from(want.data[i])).powi(2)).sum();
        (se, 255.0)
    };
    10.0 * (peak * peak / (se / n as f64).max(1e-9)).log10()
}

fn check_profile(ten_bit: bool) {
    let label = if ten_bit { "profile 2 (10-bit)" } else { "profile 0 (8-bit)" };
    let frames = 60u64;
    let packets = encode(config(ten_bit, EncodeOverrides::default()), frames, ten_bit);
    assert_eq!(packets.len() as u64, frames, "{label}: one packet per frame");
    assert!(packets[0].is_keyframe, "{label}: the first packet is a key frame");
    let mut keys = Vec::new();
    for (i, p) in packets.iter().enumerate() {
        assert_eq!(p.is_keyframe, codec::vp9_header::packet_is_keyframe(&p.data), "{label}: packet {i}'s sync flag");
        assert!(codec::vp9_header::packet_shows(&p.data), "{label}: packet {i} shows a picture");
        assert_eq!(p.pts, i as u64, "{label}: packet {i} pts");
        if p.is_keyframe {
            keys.push(i);
        }
        // Profile in the header: 0 or 2.
        if let Some(codec::vp9_header::FrameHeader::Coded(c)) =
            vp9::superframe::split(&p.data).last().and_then(|f| codec::vp9_header::peek(f))
        {
            assert_eq!(c.profile, if ten_bit { 2 } else { 0 }, "{label}: packet {i} profile");
        }
    }
    eprintln!("{label}: {} packets, key frames at {keys:?}, {} bytes", packets.len(), packets.iter().map(|p| p.data.len()).sum::<usize>());

    let raw: Vec<Vec<u8>> = packets.iter().map(|p| p.data.to_vec()).collect();
    let pictures = decode(Box::new(codec::decode::vp9_sw::Vp9Decoder::new(info(ten_bit)).unwrap()), &raw);
    assert_eq!(pictures.len() as u64, frames, "{label}: rivet's decoder shows every frame");
    let mut worst = f64::INFINITY;
    let mut sum = 0.0;
    for (t, pic) in pictures.iter().enumerate() {
        assert_eq!((pic.width, pic.height), (W, H), "{label}: picture {t} size");
        assert_eq!(pic.format, if ten_bit { PixelFormat::Yuv420p10le } else { PixelFormat::Yuv420p });
        let p = luma_psnr(pic, &source(t as u64, ten_bit));
        worst = worst.min(p);
        sum += p;
    }
    eprintln!("{label}: luma PSNR vs source: mean {:.2} dB, worst {worst:.2} dB", sum / frames as f64);
    assert!(worst >= 30.0, "{label}: worst luma PSNR {worst:.2} dB");

    // Through rivet's muxers and back.
    let mut webm = container::webm::WebmMuxer::new(W, H, FPS, VideoCodec::Vp9).unwrap();
    let mut mp4 = container::mux::Av1Mp4Muxer::new_with_codec(W, H, FPS, VideoCodec::Vp9).unwrap();
    for p in &packets {
        webm.add_packet(p.clone()).unwrap();
        mp4.add_packet(p.clone()).unwrap();
    }
    for (name, file) in [("webm", webm.finalize().unwrap()), ("mp4", mp4.finalize().unwrap().to_vec())] {
        let d = container::demux::demux(&file).expect("demux");
        assert!(d.codec.eq_ignore_ascii_case("vp9"), "{label} {name}: {}", d.codec);
        assert_eq!(d.samples.len(), packets.len(), "{label} {name}: samples");
        for (i, (s, p)) in d.samples.iter().zip(&packets).enumerate() {
            assert!(s[..] == p.data[..], "{label} {name}: sample {i} changed in the container");
        }
    }

    // QSV's own decoder on QSV's stream, against rivet's.
    let qsv = codec::decode::qsv_dec::QsvDecoder::new(info(ten_bit), 0).expect("QSV decoder");
    let hw = decode(Box::new(qsv), &raw);
    assert_eq!(hw.len(), pictures.len(), "{label}: QSV decode frame count");
    for (i, (a, b)) in hw.iter().zip(&pictures).enumerate() {
        assert!(a.data[..] == b.data[..], "{label}: QSV decode of picture {i} differs from rivet's");
    }
    eprintln!("{label}: QSV decode bit-exact with rivet's on all {} pictures", hw.len());
}

#[test]
fn qsv_vp9_profile0_decodes_in_rivet_and_tracks_the_source() {
    if intel_present() {
        check_profile(false);
    }
}

#[test]
fn qsv_vp9_profile2_decodes_in_rivet_and_tracks_the_source() {
    if intel_present() {
        check_profile(true);
    }
}

/// A constant-rate rung: CBR at 1.5 Mbit/s with a one-second buffer, the
/// average over four seconds within 15%.
#[test]
fn qsv_vp9_cbr_holds_its_rate() {
    if !intel_present() {
        return;
    }
    let bps = 1_500_000u32;
    let overrides = EncodeOverrides {
        rate_mode: Some(RateMode::Constant),
        bitrate: Some(bps),
        buffer_ms: Some(1000),
        ..Default::default()
    };
    let frames = 120u64;
    let packets = encode(config(false, overrides), frames, false);
    assert_eq!(packets.len() as u64, frames);
    let bits: u64 = packets.iter().map(|p| p.data.len() as u64 * 8).sum();
    let avg = bits as f64 / (frames as f64 / FPS);
    eprintln!("CBR {bps} bit/s: averaged {avg:.0} bit/s over {frames} frames");
    assert!((avg - f64::from(bps)).abs() <= 0.15 * f64::from(bps), "average {avg:.0} vs {bps}");
    let raw: Vec<Vec<u8>> = packets.iter().map(|p| p.data.to_vec()).collect();
    let pictures = decode(Box::new(codec::decode::vp9_sw::Vp9Decoder::new(info(false)).unwrap()), &raw);
    assert_eq!(pictures.len() as u64, frames, "the CBR stream decodes in rivet");
}
