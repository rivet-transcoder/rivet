//! The software AV1 pair, end to end: rav1e encodes it, rav1d decodes it back.
//!
//! Both halves have their own reasons to be wrong in ways a unit test on either
//! alone would not catch, and both reasons are about **stride**. rav1e pads
//! each plane row to its own alignment, and rav1d hands decoded planes back the
//! same way; copy either flat and you get a picture that shears progressively
//! down the frame. It still decodes, it still has the right byte count, and it
//! looks enough like a decoder bug to send somebody looking in the wrong place
//! for an afternoon.
//!
//! So this encodes a frame with known structure and checks the structure
//! survives the round trip, rather than merely checking that bytes came out.
//!
//! Small and fast on purpose — 128×128 at the fastest speed preset. This is a
//! correctness guard on the plumbing, not a quality or throughput measurement.

use codec::decode::Decoder;
use codec::decode::rav1d_sw::Rav1dDecoder;
use codec::encode::rav1e_sw::Rav1eEncoder;
use codec::encode::{Encoder, EncoderConfig, QualityTarget, SpeedTier};
use codec::frame::{ColorMetadata, ColorSpace, PixelFormat, StreamInfo, VideoCodec, VideoFrame};

const W: u32 = 128;
const H: u32 = 128;

/// A frame with a hard vertical edge down the middle.
///
/// Chosen because it is exactly what a stride mistake destroys: a sheared
/// picture moves the edge by a few pixels on each successive row, so comparing
/// one row against another catches it. A flat grey frame would survive every
/// stride bug ever written.
fn split_frame(pts: u64) -> VideoFrame {
    let (w, h) = (W as usize, H as usize);
    let (cw, ch) = (w / 2, h / 2);

    let mut data = Vec::with_capacity(w * h + 2 * cw * ch);
    for _ in 0..h {
        for x in 0..w {
            data.push(if x < w / 2 { 40 } else { 210 });
        }
    }
    // Neutral chroma — the luma edge is what is being checked, and flat chroma
    // keeps the encode cheap.
    data.extend(std::iter::repeat_n(128u8, 2 * cw * ch));

    VideoFrame::new(
        data.into(),
        W,
        H,
        PixelFormat::Yuv420p,
        ColorSpace::Bt709,
        pts,
    )
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
        // Fastest tier: this test is about plumbing, and Archive would make it
        // slow enough that somebody would eventually mark it `#[ignore]`.
        tier: SpeedTier::Draft,
        threads: 1,
        pixel_format: PixelFormat::Yuv420p,
        color_metadata: ColorMetadata::default(),
        gpu_index: None,
        gpu_vendor: None,
        codec: VideoCodec::Av1,
        constant_qp: false,
        // No per-rung policy: this test is about plumbing, and an empty
        // override is required to be inert anyway.
        overrides: Default::default(),
    }
}

fn stream_info() -> StreamInfo {
    StreamInfo {
        codec: "av1".to_string(),
        width: W,
        height: H,
        frame_rate: 30.0,
        duration: 1.0,
        pixel_format: PixelFormat::Yuv420p,
        color_space: ColorSpace::Bt709,
        total_frames: 1,
        bitrate: 0,
        color_metadata: ColorMetadata::default(),
    }
}

#[test]
fn rav1e_encodes_and_rav1d_decodes_it_back() {
    let mut enc = Rav1eEncoder::new(encoder_config()).expect("rav1e should construct");

    // A handful of frames: one is enough to exercise the plumbing, several
    // confirm the pts queue stays in step rather than drifting by one.
    const FRAMES: u64 = 5;
    for pts in 0..FRAMES {
        enc.send_frame(&split_frame(pts))
            .expect("rav1e accepts a frame");
    }
    enc.flush().expect("flush");

    let mut packets = Vec::new();
    while let Some(pkt) = enc.receive_packet().expect("receive") {
        packets.push(pkt);
    }

    // Count, not merely non-empty: a truncated drain returns one packet and
    // looks like success until the decode side comes up short.
    assert_eq!(
        packets.len() as u64,
        FRAMES,
        "rav1e returned {} packets for {FRAMES} frames",
        packets.len()
    );
    assert!(
        packets[0].is_keyframe,
        "the first packet must be a keyframe or nothing can start decoding here"
    );
    // Timestamps are the caller's, not rav1e's frame counter — the distinction
    // matters to any container writing in its own timebase.
    let stamps: Vec<u64> = packets.iter().map(|p| p.pts).collect();
    let mut sorted = stamps.clone();
    sorted.sort_unstable();
    assert_eq!(
        stamps, sorted,
        "packet timestamps came back out of order: {stamps:?}"
    );

    let mut dec = Rav1dDecoder::new(stream_info()).expect("rav1d should construct");
    let mut decoded = Vec::new();
    for pkt in &packets {
        dec.push_sample(&pkt.data).expect("rav1d accepts a packet");
        while let Some(frame) = dec.decode_next().expect("decode") {
            decoded.push(frame);
        }
    }
    dec.finish().expect("finish");
    while let Some(frame) = dec.decode_next().expect("drain") {
        decoded.push(frame);
    }

    assert_eq!(
        decoded.len() as u64,
        FRAMES,
        "expected {FRAMES} frames back, got {}",
        decoded.len()
    );

    let first = &decoded[0];
    assert_eq!((first.width, first.height), (W, H));
    assert_eq!(first.format, PixelFormat::Yuv420p);

    let (w, h) = (W as usize, H as usize);
    let (cw, ch) = (w / 2, h / 2);
    assert_eq!(
        first.data.len(),
        w * h + 2 * cw * ch,
        "decoded buffer is not tightly packed 4:2:0"
    );

    // The edge, on every row. A stride bug walks it sideways as the frame
    // progresses, so checking row 0 alone would pass while the picture sheared.
    for row in 0..h {
        let line = &first.data[row * w..(row + 1) * w];
        let left = line[w / 4] as i32;
        let right = line[3 * w / 4] as i32;
        assert!(
            right - left > 100,
            "row {row}: expected a dark-to-light edge, got left={left} right={right}. \
             A value that drifts with the row number means a stride was mishandled."
        );
    }
}

#[test]
fn the_encoder_refuses_a_format_it_cannot_encode() {
    // Better a clear error than a picture with the chroma planes misread. By
    // the time this tier is reached the caller has exhausted every hardware
    // backend, so a wrong answer here is the one that ships.
    let mut enc = Rav1eEncoder::new(encoder_config()).expect("construct");

    let mut wrong = split_frame(0);
    wrong.format = PixelFormat::Yuv420p10le;

    let err = enc.send_frame(&wrong).expect_err("10-bit must be refused");
    assert!(
        err.to_string().contains("4:2:0"),
        "the error should name the format it wanted: {err}"
    );
}

#[test]
fn the_encoder_refuses_a_frame_of_the_wrong_size() {
    let mut enc = Rav1eEncoder::new(encoder_config()).expect("construct");

    let mut wrong = split_frame(0);
    wrong.width = W * 2;

    let err = enc
        .send_frame(&wrong)
        .expect_err("a mismatched frame must be refused");
    assert!(
        err.to_string().contains("configured for"),
        "the error should say what it was configured for: {err}"
    );
}
