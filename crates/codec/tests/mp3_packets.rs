//! The MP3 decoder fed the way containers hand MP3 over: in packets that
//! are one frame, or byte runs that cut frames anywhere.
//!
//! The stream is half a second of a 1 kHz tone, 48 kHz mono, 64 kb/s
//! (192 bytes a frame; no ID3, no Xing frame), made here by the workspace's
//! own MP3 encoder. It decodes to one 1152-sample frame per frame written.

use codec::audio::decode::Mp3Decoder;
use codec::audio::{AudioCodec, AudioDecoder, AudioEncoderConfig, AudioFrame, create_encoder};

/// The stream, and how many frames it holds.
fn stream() -> (Vec<u8>, usize) {
    let mut enc = create_encoder(AudioEncoderConfig::new(AudioCodec::Mp3, 48_000, 1, 64_000)).expect("the MP3 encoder");
    let samples = (0..24_000).map(|i| 0.4 * (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / 48_000.0).sin()).collect();
    let mut packets = enc.encode(&AudioFrame { samples, sample_rate: 48_000, channels: 1, pts: 0 }).expect("encode");
    packets.extend(enc.flush().expect("flush"));
    assert!(packets.iter().all(|p| p.data.len() == 192), "64 kb/s at 48 kHz: 192-byte frames");
    (packets.iter().flat_map(|p| p.data.clone()).collect(), packets.len())
}

/// Every sample the decoder produces from `stream` cut into `packet`-byte
/// packets, then flushed.
fn decode_in(stream: &[u8], packet: usize) -> Vec<f32> {
    let mut dec = Mp3Decoder::new(48_000, 1).expect("constructs");
    let mut out = Vec::new();
    for p in stream.chunks(packet) {
        for f in dec.decode(p, 0).expect("decode") {
            assert_eq!((f.sample_rate, f.channels), (48_000, 1));
            out.extend(f.samples);
        }
    }
    for f in dec.flush().expect("flush") {
        out.extend(f.samples);
    }
    out
}

/// One frame to a packet (as AVI and Matroska store MP3) and byte runs that
/// cut frames anywhere both decode every frame, to the same samples as the
/// whole stream in one go. (A decoder that confirms a frame against the next
/// one's header must hold a lone frame back, not discard it as unsynced.)
#[test]
fn packets_decode_every_frame_whatever_their_size() {
    let (stream, frames) = stream();
    let whole = decode_in(&stream, stream.len());
    assert_eq!(whole.len(), frames * 1152, "every frame of the stream");
    assert!(whole.iter().any(|s| s.abs() > 0.1), "the tone, not silence");
    for packet in [192, 100, 7, 1] {
        let got = decode_in(&stream, packet);
        assert_eq!(got.len(), whole.len(), "{packet}-byte packets: samples decoded");
        assert!(got == whole, "{packet}-byte packets: not the whole stream's samples");
    }
}

/// Timestamps run on from the first packet's, a frame's length apart.
#[test]
fn frame_timestamps_step_by_the_frame_length() {
    let (stream, frames) = stream();
    let mut dec = Mp3Decoder::new(48_000, 1).expect("constructs");
    let mut pts = Vec::new();
    for p in stream.chunks(192) {
        pts.extend(dec.decode(p, 5_000).expect("decode").iter().map(|f| f.pts));
    }
    pts.extend(dec.flush().expect("flush").iter().map(|f| f.pts));
    assert_eq!(pts.len(), frames);
    for (i, p) in pts.iter().enumerate() {
        assert_eq!(*p, 5_000 + i as i64 * 24_000, "frame {i}");
    }
}
