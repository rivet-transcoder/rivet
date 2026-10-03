//! The software AV1 decoder's end-of-stream drain, frame by frame.
//!
//! The adapter (`decode::av1_sw`) decodes on a worker thread a few temporal
//! units ahead of the caller, so when `finish` is called frames are still in
//! flight. A drain that stopped collecting too early would drop the tail of
//! the stream silently: the file would just be shorter.
//!
//! This decodes enough frames to fill every frame context several times over
//! and checks each one came back, in order, by an index painted into the
//! picture itself rather than by counting.

use codec::decode::Decoder;
use codec::decode::av1_sw::Av1Decoder;
use codec::encode::av1_sw::Av1Encoder;
use codec::encode::{Encoder, EncoderConfig, QualityTarget, SpeedTier};
use codec::frame::{ColorMetadata, ColorSpace, PixelFormat, StreamInfo, VideoCodec, VideoFrame};

const W: u32 = 176;
const H: u32 = 144;
const FRAMES: u64 = 40;
/// Vertical stripes, one per bit of the frame index.
const BITS: usize = 8;

/// A frame whose index is written as eight black or white stripes.
///
/// Stripes survive quantisation where a subtle brightness ramp would not, and
/// every frame differs from its neighbours by whole stripes, so a dropped or
/// repeated frame shows up as the wrong number rather than as noise.
fn indexed_frame(index: u64) -> VideoFrame {
    let (w, h) = (W as usize, H as usize);
    let (cw, ch) = (w / 2, h / 2);
    let stripe = w / BITS;

    let mut data = Vec::with_capacity(w * h + 2 * cw * ch);
    for _ in 0..h {
        for x in 0..w {
            let bit = (x / stripe).min(BITS - 1);
            data.push(if index >> bit & 1 == 1 { 220 } else { 30 });
        }
    }
    data.extend(std::iter::repeat_n(128u8, 2 * cw * ch));

    VideoFrame::new(
        data.into(),
        W,
        H,
        PixelFormat::Yuv420p,
        ColorSpace::Bt709,
        index,
    )
}

/// Read the index back from the middle of each stripe on the middle row.
fn read_index(frame: &VideoFrame) -> u64 {
    let w = W as usize;
    let stripe = w / BITS;
    let row = &frame.data[(H as usize / 2) * w..(H as usize / 2 + 1) * w];
    (0..BITS).fold(0, |acc, bit| {
        acc | (u64::from(row[bit * stripe + stripe / 2] > 128) << bit)
    })
}

fn encoder_config() -> EncoderConfig {
    EncoderConfig {
        width: W,
        height: H,
        frame_rate: 30.0,
        quality: u8::MAX,
        speed_preset: u8::MAX,
        keyframe_interval: 30,
        target: QualityTarget::Standard,
        tier: SpeedTier::Draft,
        threads: 0,
        pixel_format: PixelFormat::Yuv420p,
        color_metadata: ColorMetadata::default(),
        gpu_index: None,
        gpu_vendor: None,
        codec: VideoCodec::Av1,
        constant_qp: false,
        overrides: Default::default(),
    }
}

fn stream_info() -> StreamInfo {
    StreamInfo {
        codec: "av1".to_string(),
        width: W,
        height: H,
        frame_rate: 30.0,
        duration: FRAMES as f64 / 30.0,
        pixel_format: PixelFormat::Yuv420p,
        color_space: ColorSpace::Bt709,
        total_frames: FRAMES,
        bitrate: 0,
        color_metadata: ColorMetadata::default(),
    }
}

#[test]
fn every_frame_comes_back_in_order_through_the_threaded_drain() {
    let mut enc = Av1Encoder::new(encoder_config()).expect("the AV1 encoder should construct");
    let mut packets = Vec::new();
    for i in 0..FRAMES {
        enc.send_frame(&indexed_frame(i))
            .expect("the encoder accepts a frame");
        while let Some(pkt) = enc.receive_packet().expect("receive") {
            packets.push(pkt);
        }
    }
    enc.flush().expect("flush");
    while let Some(pkt) = enc.receive_packet().expect("receive") {
        packets.push(pkt);
    }
    assert_eq!(
        packets.len() as u64,
        FRAMES,
        "the encoder returned {} packets",
        packets.len()
    );

    let mut dec = Av1Decoder::new(stream_info()).expect("the AV1 decoder should construct");
    let mut decoded = Vec::new();
    for pkt in &packets {
        dec.push_sample(&pkt.data).expect("the decoder accepts a packet");
        while let Some(frame) = dec.decode_next().expect("decode") {
            decoded.push(read_index(&frame));
        }
    }
    let before_finish = decoded.len();
    dec.finish().expect("finish");
    while let Some(frame) = dec.decode_next().expect("drain") {
        decoded.push(read_index(&frame));
    }

    let expected: Vec<u64> = (0..FRAMES).collect();
    assert_eq!(
        decoded, expected,
        "decoded frame indices differ from the input ({before_finish} frames had come out \
         before finish)"
    );
}
