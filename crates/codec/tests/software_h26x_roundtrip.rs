//! The software H.264 / H.265 pair, end to end: the native `h26x` encoder
//! writes it, the native `h26x` decoder reads it back — through rivet's own
//! adapters on both sides, which is the part nothing in the `h26x` crate's
//! gate covers.
//!
//! The `h26x` gate (`tools/verify_encode.sh`) already proves the encoder's
//! bitstreams decode to its own reconstruction and that libavcodec agrees.
//! What it cannot see is the plumbing here: the frame-buffer prefix handed to
//! the encoder, the timestamp table, the keyframe flag, the packet order, and
//! the decoder adapter's plane packing. Each of those can be wrong in a way
//! that decodes without error, so this checks structure rather than bytes
//! having come out — the same reasoning as the AV1 round trip beside it.
//!
//! Small and fast on purpose: 96×80 (deliberately not a multiple of 16, so
//! the encoder's padding and crop are exercised) at the fastest tier. A
//! correctness guard on the plumbing, not a quality measurement.

use codec::decode::Decoder;
use codec::decode::h26x_sw::H26xDecoder;
use codec::encode::h26x_sw::H26xEncoder;
use codec::encode::{Encoder, EncoderConfig, QualityTarget, SpeedTier};
use codec::frame::{ColorMetadata, ColorSpace, PixelFormat, StreamInfo, VideoCodec, VideoFrame};

const W: u32 = 96;
const H: u32 = 80;
const FRAMES: u64 = 6;
/// Well inside the clip, so the forced IDR is one the GOP cadence would not
/// have placed.
const FORCED_IDR_AT: u64 = 3;

/// A frame with a hard vertical edge whose position moves one pixel per
/// frame.
///
/// The edge is what a stride mistake destroys — a sheared picture moves it by
/// a few pixels on each successive row — and the per-frame motion is what a
/// packet-order or timestamp mistake destroys: every frame is
/// distinguishable, so a swapped or duplicated one cannot pass.
fn edge_frame(pts: u64) -> VideoFrame {
    let (w, h) = (W as usize, H as usize);
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    let edge = edge_x(pts);

    let mut data = Vec::with_capacity(w * h + 2 * cw * ch);
    for _ in 0..h {
        for x in 0..w {
            data.push(if x < edge { 40 } else { 210 });
        }
    }
    data.extend(std::iter::repeat_n(128u8, 2 * cw * ch));

    VideoFrame::new(data.into(), W, H, PixelFormat::Yuv420p, ColorSpace::Bt709, pts)
}

fn edge_x(pts: u64) -> usize {
    W as usize / 2 + pts as usize
}

fn encoder_config(codec: VideoCodec) -> EncoderConfig {
    EncoderConfig {
        width: W,
        height: H,
        frame_rate: 30.0,
        quality: u8::MAX,
        speed_preset: u8::MAX,
        keyframe_interval: 30,
        target: QualityTarget::Standard,
        tier: SpeedTier::Draft,
        threads: 1,
        pixel_format: PixelFormat::Yuv420p,
        color_metadata: ColorMetadata::default(),
        gpu_index: None,
        gpu_vendor: None,
        codec,
        constant_qp: false,
        overrides: Default::default(),
    }
}

/// Encode `FRAMES` frames, forcing an IDR part way through, and return the
/// packets in the order the encoder produced them.
fn encode(codec: VideoCodec) -> Vec<codec::encode::EncodedPacket> {
    let mut enc = H26xEncoder::new(encoder_config(codec)).expect("build the software encoder");
    let mut packets = Vec::new();
    for pts in 0..FRAMES {
        if pts == FORCED_IDR_AT {
            enc.force_keyframe_next().expect("force an IDR");
        }
        enc.send_frame(&edge_frame(pts)).expect("send a frame");
        while let Some(p) = enc.receive_packet().expect("receive") {
            packets.push(p);
        }
    }
    enc.flush().expect("flush");
    while let Some(p) = enc.receive_packet().expect("receive after flush") {
        packets.push(p);
    }
    packets
}

fn decode(codec: VideoCodec, packets: &[codec::encode::EncodedPacket]) -> Vec<VideoFrame> {
    let info = StreamInfo {
        codec: codec.label().to_string(),
        width: W,
        height: H,
        frame_rate: 30.0,
        duration: 0.0,
        pixel_format: PixelFormat::Yuv420p,
        color_space: ColorSpace::Bt709,
        total_frames: FRAMES,
        bitrate: 0,
        color_metadata: ColorMetadata::default(),
    };
    let mut dec = H26xDecoder::new(info).expect("build the software decoder");
    let mut frames = Vec::new();
    for p in packets {
        dec.push_sample(&p.data).expect("push a packet");
        while let Some(f) = dec.decode_next().expect("receive a frame") {
            frames.push(f);
        }
    }
    dec.finish().expect("finish");
    while let Some(f) = dec.decode_next().expect("receive after finish") {
        frames.push(f);
    }
    frames
}

fn round_trip(codec: VideoCodec) {
    let packets = encode(codec);

    // One packet per frame, in the order the frames went in, carrying the
    // timestamps they went in with. No B pictures, so nothing reorders.
    assert_eq!(packets.len() as u64, FRAMES, "{codec:?}: one packet per frame");
    let pts: Vec<u64> = packets.iter().map(|p| p.pts).collect();
    assert_eq!(pts, (0..FRAMES).collect::<Vec<u64>>(), "{codec:?}: timestamps in order");

    // Keyframes exactly where they must be: the first picture by rule, the
    // forced one by request, and nowhere else in a clip shorter than the
    // interval.
    let keys: Vec<u64> = packets.iter().filter(|p| p.is_keyframe).map(|p| p.pts).collect();
    assert_eq!(keys, vec![0, FORCED_IDR_AT], "{codec:?}: keyframe positions");

    // Every packet is Annex-B, and the keyframes carry the parameter sets the
    // muxer will lift into avcC / hvcC.
    for p in &packets {
        assert!(p.data.starts_with(&[0, 0, 0, 1]) || p.data.starts_with(&[0, 0, 1]),
            "{codec:?}: packet at pts {} is not Annex-B", p.pts);
    }
    let has_nal = |data: &[u8], pred: &dyn Fn(u8) -> bool| {
        h26x::nal::annexb_nals(data).any(|n| !n.is_empty() && pred(n[0]))
    };
    for p in packets.iter().filter(|p| p.is_keyframe) {
        let (sps, pps) = match codec {
            // nal_unit_type 7 / 8.
            VideoCodec::H264 => (
                has_nal(&p.data, &|b| b & 0x1f == 7),
                has_nal(&p.data, &|b| b & 0x1f == 8),
            ),
            // nal_unit_type 33 / 34 in the high six bits.
            VideoCodec::H265 => (
                has_nal(&p.data, &|b| (b >> 1) & 0x3f == 33),
                has_nal(&p.data, &|b| (b >> 1) & 0x3f == 34),
            ),
            VideoCodec::Av1 => unreachable!(),
        };
        assert!(sps && pps, "{codec:?}: keyframe at pts {} lacks its parameter sets", p.pts);
    }

    // Exactly one parameter set of each kind across the WHOLE stream. The
    // muxer writes `avc1` / `hvc1`: parameter sets live out of band in the
    // config box and are stripped from the samples, so a set that changes
    // mid-stream — even legally, re-sent under the same id, which Annex-B
    // tolerates — leaves the box holding two under one id and a decoder
    // reading the pictures written under the other one as garbage from their
    // first macroblock. The H.264 encoder did exactly that (a different
    // `pic_init_qp` for I and P pictures) and every Annex-B check passed.
    let distinct = |pred: &dyn Fn(u8) -> bool| -> std::collections::BTreeSet<Vec<u8>> {
        packets
            .iter()
            .flat_map(|p| h26x::nal::annexb_nals(&p.data).map(|n| n.to_vec()).collect::<Vec<_>>())
            .filter(|n| !n.is_empty() && pred(n[0]))
            .collect()
    };
    let (sps_set, pps_set, vps_set) = match codec {
        VideoCodec::H264 => (
            distinct(&|b| b & 0x1f == 7),
            distinct(&|b| b & 0x1f == 8),
            std::collections::BTreeSet::new(),
        ),
        VideoCodec::H265 => (
            distinct(&|b| (b >> 1) & 0x3f == 33),
            distinct(&|b| (b >> 1) & 0x3f == 34),
            distinct(&|b| (b >> 1) & 0x3f == 32),
        ),
        VideoCodec::Av1 => unreachable!(),
    };
    assert_eq!(sps_set.len(), 1, "{codec:?}: one SPS for the stream, got {sps_set:?}");
    assert_eq!(pps_set.len(), 1, "{codec:?}: one PPS for the stream, got {pps_set:?}");
    if codec == VideoCodec::H265 {
        assert_eq!(vps_set.len(), 1, "{codec:?}: one VPS for the stream, got {vps_set:?}");
    }

    let frames = decode(codec, &packets);
    assert_eq!(frames.len() as u64, FRAMES, "{codec:?}: every frame decodes");

    for (i, f) in frames.iter().enumerate() {
        assert_eq!((f.width, f.height), (W, H), "{codec:?}: frame {i} is cropped to size");
        assert_eq!(f.format, PixelFormat::Yuv420p, "{codec:?}: frame {i} format");
        let w = W as usize;
        let luma = &f.data[..w * H as usize];
        // The edge for THIS frame — a frame decoded in the wrong place lands
        // on a neighbour's edge position and fails here.
        let edge = edge_x(i as u64);
        for (y, row) in luma.chunks_exact(w).enumerate() {
            // Sample well clear of the edge on both sides: quantisation
            // ringing lives within a few pixels of it.
            let dark = row[edge - 6];
            let bright = row[edge + 6];
            let inside_dark = row[4];
            let inside_bright = row[w - 5];
            assert!(dark < 100 && bright > 150,
                "{codec:?}: frame {i} row {y}: edge not at x={edge} (dark={dark}, bright={bright})");
            assert!(inside_dark < 100 && inside_bright > 150,
                "{codec:?}: frame {i} row {y}: plane sheared (left={inside_dark}, right={inside_bright})");
        }
    }
}

#[test]
fn h264_round_trips_through_the_native_pair() {
    round_trip(VideoCodec::H264);
}

#[test]
fn h265_round_trips_through_the_native_pair() {
    round_trip(VideoCodec::H265);
}

/// The same edge at 10 bits, as little-endian `u16` planes (`yuv420p10le`).
fn edge_frame_10(pts: u64) -> VideoFrame {
    let (w, h) = (W as usize, H as usize);
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    let edge = edge_x(pts);
    let mut data = Vec::with_capacity((w * h + 2 * cw * ch) * 2);
    for _ in 0..h {
        for x in 0..w {
            // 40 and 210 at 8 bits, scaled into the 10-bit range with real
            // low bits set so a truncation to 8 bits would be visible.
            let v: u16 = if x < edge { 40 * 4 + 3 } else { 210 * 4 + 1 };
            data.extend_from_slice(&v.to_le_bytes());
        }
    }
    for _ in 0..2 * cw * ch {
        data.extend_from_slice(&512u16.to_le_bytes());
    }
    VideoFrame::new(data.into(), W, H, PixelFormat::Yuv420p10le, ColorSpace::Bt709, pts)
}

/// H.265 Main 10 through the native pair: 10-bit in, 10-bit out, the edge
/// intact on every row and the low bits genuinely 10-bit (not 8-bit
/// samples shifted up).
#[test]
fn h265_ten_bit_round_trips_through_the_native_pair() {
    let codec = VideoCodec::H265;
    let cfg = EncoderConfig { pixel_format: PixelFormat::Yuv420p10le, ..encoder_config(codec) };
    let mut enc = H26xEncoder::new(cfg).expect("build the 10-bit software encoder");
    let mut packets = Vec::new();
    for pts in 0..FRAMES {
        enc.send_frame(&edge_frame_10(pts)).expect("send a 10-bit frame");
        while let Some(p) = enc.receive_packet().expect("receive") {
            packets.push(p);
        }
    }
    enc.flush().expect("flush");
    while let Some(p) = enc.receive_packet().expect("receive after flush") {
        packets.push(p);
    }
    assert_eq!(packets.len() as u64, FRAMES);

    let frames = decode(codec, &packets);
    assert_eq!(frames.len() as u64, FRAMES, "every 10-bit frame decodes");
    for (i, f) in frames.iter().enumerate() {
        assert_eq!(f.format, PixelFormat::Yuv420p10le, "frame {i} comes back as 10-bit");
        let w = W as usize;
        let luma: Vec<u16> = f.data[..w * H as usize * 2]
            .chunks_exact(2)
            .map(|p| u16::from_le_bytes([p[0], p[1]]))
            .collect();
        let edge = edge_x(i as u64);
        let mut odd_low_bits = 0usize;
        for (y, row) in luma.chunks_exact(w).enumerate() {
            let (dark, bright) = (row[edge - 6], row[edge + 6]);
            assert!(dark < 400 && bright > 600,
                "frame {i} row {y}: 10-bit edge not at x={edge} (dark={dark}, bright={bright})");
            odd_low_bits += row.iter().filter(|&&v| v & 3 != 0).count();
        }
        // A picture that had been coded at 8 bits and shifted up would have
        // zero low bits everywhere; the source has them set on every sample.
        assert!(odd_low_bits > 0, "frame {i}: no sample carries low bits — narrowed to 8-bit?");
    }
}

/// H.264 stays 8-bit on every backend; a 10-bit request is refused by name,
/// not narrowed.
#[test]
fn ten_bit_h264_is_refused_by_name() {
    let cfg = EncoderConfig {
        pixel_format: PixelFormat::Yuv420p10le,
        ..encoder_config(VideoCodec::H264)
    };
    let err = H26xEncoder::new(cfg).err().expect("10-bit H.264 must be refused");
    assert!(err.to_string().contains("8-bit"), "{err}");
}

#[test]
fn av1_is_not_this_tier() {
    let err = H26xEncoder::new(encoder_config(VideoCodec::Av1)).err().expect("AV1 refused");
    assert!(err.to_string().contains("H.264 and H.265"), "{err}");
}
