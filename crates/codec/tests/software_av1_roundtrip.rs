//! The software AV1 pair, end to end: rivet's own AV1 encoder encodes it,
//! rivet's own AV1 decoder decodes it back, through the codec crate's
//! adapters (`encode::av1_sw`, `decode::av1_sw`).
//!
//! Both adapters copy planes between the pipeline's packed frames and the
//! codec's, and a stride or plane-offset mistake on either side gives a
//! picture that shears progressively down the frame. It still decodes, it
//! still has the right byte count, and it looks enough like a decoder bug to
//! send somebody looking in the wrong place for an afternoon.
//!
//! So this encodes a frame with known structure and checks the structure
//! survives the round trip, rather than merely checking that bytes came out.
//!
//! Small and fast on purpose — 128×128 at the fastest speed tier. This is a
//! correctness guard on the plumbing, not a quality or throughput measurement.

use codec::decode::Decoder;
use codec::decode::av1_sw::Av1Decoder;
use codec::encode::av1_sw::Av1Encoder;
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
fn the_software_encoder_encodes_and_the_software_decoder_decodes_it_back() {
    let mut enc = Av1Encoder::new(encoder_config()).expect("the AV1 encoder should construct");

    // A handful of frames: one is enough to exercise the plumbing, several
    // confirm the pts queue stays in step rather than drifting by one.
    const FRAMES: u64 = 5;
    for pts in 0..FRAMES {
        enc.send_frame(&split_frame(pts))
            .expect("the encoder accepts a frame");
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
        "the encoder returned {} packets for {FRAMES} frames",
        packets.len()
    );
    assert!(
        packets[0].is_keyframe,
        "the first packet must be a keyframe or nothing can start decoding here"
    );
    // Timestamps are the caller's, not the encoder's frame counter — the distinction
    // matters to any container writing in its own timebase.
    let stamps: Vec<u64> = packets.iter().map(|p| p.pts).collect();
    let mut sorted = stamps.clone();
    sorted.sort_unstable();
    assert_eq!(
        stamps, sorted,
        "packet timestamps came back out of order: {stamps:?}"
    );

    let mut dec = Av1Decoder::new(stream_info()).expect("the AV1 decoder should construct");
    let mut decoded = Vec::new();
    for pkt in &packets {
        dec.push_sample(&pkt.data).expect("the decoder accepts a packet");
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
    let mut enc = Av1Encoder::new(encoder_config()).expect("construct");

    let mut wrong = split_frame(0);
    wrong.format = PixelFormat::Yuv420p10le;

    let err = enc.send_frame(&wrong).expect_err("10-bit must be refused");
    assert!(
        err.to_string().contains("Yuv420p"),
        "the error should name the format it wanted: {err}"
    );
}

#[test]
fn the_encoder_refuses_a_frame_of_the_wrong_size() {
    let mut enc = Av1Encoder::new(encoder_config()).expect("construct");

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

/// Throughput of the software pair at 1280x720, printed (`--ignored
/// --nocapture`, release): what `docs/codec-decode.md` and
/// `docs/codec-encode.md` quote. Not a gate — a machine's speed is not a
/// property of the code.
#[test]
#[ignore = "a measurement: run with --release --ignored --nocapture"]
fn throughput_at_720p() {
    let (w, h, n) = (1280u32, 720u32, 12u64);
    let frames: Vec<VideoFrame> = (0..n)
        .map(|t| {
            let (wu, hu) = (w as usize, h as usize);
            let mut data = vec![128u8; wu * hu * 3 / 2];
            let mut seed = 0x2545_f491_4f6c_dd1du64 ^ t;
            for y in 0..hu {
                for x in 0..wu {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    let base = ((x / 3 + y / 2 + 4 * t as usize) % 180) as u8 + 30;
                    data[y * wu + x] = base.wrapping_add((seed & 7) as u8);
                }
            }
            for (i, v) in data[wu * hu..].iter_mut().enumerate() {
                *v = 100 + ((i / 37) % 50) as u8;
            }
            VideoFrame::new(data.into(), w, h, PixelFormat::Yuv420p, ColorSpace::Bt709, t)
        })
        .collect();
    let config = EncoderConfig { width: w, height: h, tier: SpeedTier::Standard, ..encoder_config() };
    let mut enc = Av1Encoder::new(config).expect("encoder");
    let start = std::time::Instant::now();
    let mut packets = Vec::new();
    for f in &frames {
        enc.send_frame(f).unwrap();
        while let Some(p) = enc.receive_packet().unwrap() {
            packets.push(p);
        }
    }
    let encode = start.elapsed().as_secs_f64();
    let mp = f64::from(w * h) * n as f64 / 1e6;
    eprintln!(
        "encode 1280x720, {n} frames: {encode:.2} s, {:.2} frames/s, {:.2} MP/s, {} bytes",
        n as f64 / encode,
        mp / encode,
        packets.iter().map(|p| p.data.len()).sum::<usize>()
    );
    let info = StreamInfo { width: w, height: h, ..stream_info() };
    for threaded in [false, true] {
        // SAFETY: a test process; nothing else reads the variable concurrently.
        unsafe { std::env::set_var("RIVET_AV1_DECODE_THREAD", if threaded { "1" } else { "0" }) };
        let mut dec = Av1Decoder::new(info.clone()).unwrap();
        let start = std::time::Instant::now();
        let mut got = 0;
        for p in &packets {
            dec.push_sample(&p.data).unwrap();
            while dec.decode_next().unwrap().is_some() {
                got += 1;
            }
        }
        dec.finish().unwrap();
        while dec.decode_next().unwrap().is_some() {
            got += 1;
        }
        let t = start.elapsed().as_secs_f64();
        assert_eq!(got, n);
        eprintln!(
            "decode 1280x720 ({}): {t:.2} s, {:.2} frames/s, {:.2} MP/s",
            if threaded { "worker thread" } else { "caller's thread" },
            n as f64 / t,
            mp / t
        );
    }
}
