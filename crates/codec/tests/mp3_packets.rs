//! The MP3 decoder fed the way containers hand MP3 over: in packets that
//! are one frame, or byte runs that cut frames anywhere.
//!
//! `tests/data/mp3_tone_48k_mono.mp3` is 22 MPEG-1 Layer III frames (0.5 s
//! of a 1 kHz tone, 48 kHz mono, 64 kb/s, 192 bytes a frame; no ID3, no
//! Xing frame), made with
//! `ffmpeg -f lavfi -i sine=frequency=1000:duration=0.5:sample_rate=48000
//! -c:a libmp3lame -b:a 64k -id3v2_version 0 -write_xing 0`. ffmpeg decodes it
//! to 22 × 1152 samples.

use codec::audio::AudioDecoder;
use codec::audio::decode::Mp3Decoder;

const STREAM: &[u8] = include_bytes!("data/mp3_tone_48k_mono.mp3");
const FRAMES: usize = 22;

/// Every sample the decoder produces from `STREAM` cut into `packet`-byte
/// packets, then flushed.
fn decode_in(packet: usize) -> Vec<f32> {
    let mut dec = Mp3Decoder::new(48_000, 1).expect("constructs");
    let mut out = Vec::new();
    for p in STREAM.chunks(packet) {
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
/// whole stream in one go. Driving the minimp3 crate's reader a packet at a
/// time decoded none of them (at most one): minimp3 confirms a frame against
/// the next frame's header and, handed a single frame, discarded it as
/// unsynced.
#[test]
fn packets_decode_every_frame_whatever_their_size() {
    let whole = decode_in(STREAM.len());
    assert_eq!(whole.len(), FRAMES * 1152, "every frame of the stream");
    assert!(whole.iter().any(|s| s.abs() > 0.1), "the tone, not silence");
    for packet in [192, 100, 7, 1] {
        let got = decode_in(packet);
        assert_eq!(got.len(), whole.len(), "{packet}-byte packets: samples decoded");
        assert!(got == whole, "{packet}-byte packets: not the whole stream's samples");
    }
}

/// Timestamps run on from the first packet's, a frame's length apart.
#[test]
fn frame_timestamps_step_by_the_frame_length() {
    let mut dec = Mp3Decoder::new(48_000, 1).expect("constructs");
    let mut pts = Vec::new();
    for p in STREAM.chunks(192) {
        pts.extend(dec.decode(p, 5_000).expect("decode").iter().map(|f| f.pts));
    }
    pts.extend(dec.flush().expect("flush").iter().map(|f| f.pts));
    assert_eq!(pts.len(), FRAMES);
    for (i, p) in pts.iter().enumerate() {
        assert_eq!(*p, 5_000 + i as i64 * 24_000, "frame {i}");
    }
}
