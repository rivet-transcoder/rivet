use super::*;
use crate::audio::decode::flac::{FlacDecoder, decode_frame};

fn signal(frames: usize, channels: usize, bits: u32, seed: u32) -> Vec<i32> {
    let full = ((1i64 << (bits - 1)) - 1) as f64;
    let mut rng = seed.max(1);
    let mut noise = move || {
        rng ^= rng << 13;
        rng ^= rng >> 17;
        rng ^= rng << 5;
        f64::from(rng) / f64::from(u32::MAX) - 0.5
    };
    let mut out = Vec::with_capacity(frames * channels);
    for i in 0..frames {
        let t = i as f64 / 44_100.0;
        let common = (t * 2.0 * std::f64::consts::PI * 440.0).sin() * 0.5;
        for c in 0..channels {
            let v = if i % 9000 < 700 {
                0.0
            } else {
                common * (1.0 - 0.2 * c as f64) + noise() * 0.01 * (c + 1) as f64
            };
            out.push((v * full).round().clamp(-full - 1.0, full) as i32);
        }
    }
    out
}

fn round_trip(pcm: &[i32], channels: u8, bits: u8, level: FlacLevel) -> (Vec<u8>, StreamInfo) {
    let mut enc =
        FlacEncoder::new(FlacEncoderConfig { sample_rate: 44_100, channels, bits_per_sample: bits, level }).unwrap();
    let mut frames = enc.encode_int(pcm);
    frames.extend(enc.finish());
    let info = enc.stream_info();
    let mut dec = FlacDecoder::new(Some(&enc.metadata_blocks()), 44_100, channels).unwrap();
    let mut got = Vec::new();
    let mut stream = Vec::new();
    for (f, n) in &frames {
        let (s, ch, b) = dec.decode_int(f).unwrap();
        assert_eq!((ch, b), (channels, u32::from(bits)));
        assert_eq!(s.len(), *n as usize * usize::from(channels));
        got.extend(s);
        stream.extend_from_slice(f);
    }
    assert!(got == pcm, "{channels}ch {bits}-bit {level:?}: round trip differs");
    assert_eq!(dec.md5_matches(), Some(true), "MD5");
    (stream, info)
}

#[test]
fn every_level_depth_and_layout_round_trips() {
    for level in [FlacLevel::Fast, FlacLevel::Default, FlacLevel::Best] {
        for (channels, bits) in [(1u8, 16u8), (2, 16), (2, 24), (6, 24), (8, 16)] {
            let pcm = signal(10_000, usize::from(channels), u32::from(bits), 7);
            round_trip(&pcm, channels, bits, level);
        }
    }
    for bits in [4u8, 8, 12, 20, 32] {
        let pcm = signal(5_000, 2, u32::from(bits), 3);
        round_trip(&pcm, 2, bits, FlacLevel::Default);
    }
}

#[test]
fn edge_shapes_round_trip() {
    // Shorter than one block, shorter than 16 samples, exactly one block,
    // all silence, a constant, wasted low bits, full-scale square waves.
    let cases: Vec<(Vec<i32>, u8)> = vec![
        (signal(1_000, 2, 16, 1), 2),
        (signal(5, 1, 16, 1), 1),
        (signal(BLOCK_SIZE, 2, 16, 2), 2),
        (vec![0; 9_000], 2),
        (vec![1234; 9_000], 1),
        (signal(9_000, 2, 16, 4).iter().map(|s| s & !0xFF).collect(), 2),
        ((0..9_000).map(|i| if (i / 3) % 2 == 0 { 32_767 } else { -32_768 }).collect(), 1),
    ];
    for (pcm, ch) in cases {
        let (_, info) = round_trip(&pcm, ch, 16, FlacLevel::Default);
        assert_eq!(info.total_samples, (pcm.len() / usize::from(ch)) as u64);
    }
}

#[test]
fn stereo_decorrelation_is_used_on_correlated_channels() {
    // Identical channels: the side channel is all zeros.
    let mono = signal(BLOCK_SIZE, 1, 16, 9);
    let pcm: Vec<i32> = mono.iter().flat_map(|&s| [s, s]).collect();
    let mut enc = FlacEncoder::new(FlacEncoderConfig {
        sample_rate: 48_000,
        channels: 2,
        bits_per_sample: 16,
        level: FlacLevel::Default,
    })
    .unwrap();
    let frames = enc.encode_int(&pcm);
    let frame = decode_frame(&frames[0].0, None).unwrap();
    assert!(matches!(frame.header.assignment, 8..=10), "assignment {}", frame.header.assignment);
    assert_eq!(frame.samples, pcm);
}

#[test]
fn coded_numbers_use_the_utf8_form() {
    for (v, want) in [
        (0x7Fu64, vec![0x7F]),
        (0x80, vec![0xC2, 0x80]),
        (0x7FF, vec![0xDF, 0xBF]),
        (0x800, vec![0xE0, 0xA0, 0x80]),
        ((1 << 36) - 1, vec![0xFE, 0xBF, 0xBF, 0xBF, 0xBF, 0xBF, 0xBF]),
    ] {
        let mut bw = BitWriter::default();
        write_coded_number(&mut bw, v);
        assert_eq!(bw.into_bytes(), want, "{v:#x}");
    }
}

#[test]
fn streaminfo_describes_the_stream() {
    let pcm = signal(10_000, 2, 16, 5);
    let (_, info) = round_trip(&pcm, 2, 16, FlacLevel::Default);
    assert_eq!((info.min_block_size, info.max_block_size), (4096, 4096));
    assert_eq!(info.total_samples, 10_000);
    assert!(info.min_frame_size > 0 && info.min_frame_size <= info.max_frame_size);
    assert_ne!(info.md5, [0; 16]);
}
