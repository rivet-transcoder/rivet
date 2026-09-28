use super::*;
use crate::audio::decode::alac::AlacDecoder;

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
        let t = i as f64 / 48_000.0;
        let common = (t * 2.0 * std::f64::consts::PI * 300.0).sin() * 0.6;
        for c in 0..channels {
            let v = if i % 10_000 < 900 {
                0.0
            } else if i % 10_000 < 1000 {
                if (i / 2 + c) % 2 == 0 { 1.0 } else { -1.0 }
            } else {
                common * (1.0 - 0.1 * c as f64) + noise() * 0.02
            };
            out.push((v * full).round().clamp(-full - 1.0, full) as i32);
        }
    }
    out
}

fn round_trip(pcm: &[i32], channels: u8, bits: u8) -> usize {
    let mut enc = AlacEncoder::new(48_000, channels, bits).unwrap();
    let mut frames = enc.encode_int(pcm);
    frames.extend(enc.finish());
    let cookie = enc.cookie();
    assert!(frames.iter().all(|(f, _)| f.len() as u32 <= cookie.max_frame_bytes));
    let mut dec = AlacDecoder::new(Some(&cookie.to_bytes())).unwrap();
    let mut got = Vec::new();
    let mut size = 0;
    for (f, n) in &frames {
        let s = dec.decode_int(f).unwrap();
        assert_eq!(s.len(), *n as usize * usize::from(channels));
        got.extend(s);
        size += f.len();
    }
    assert!(got == pcm, "{channels}ch {bits}-bit: round trip differs");
    size
}

#[test]
fn every_depth_and_layout_round_trips() {
    for channels in 1..=8u8 {
        let pcm = signal(9_000, usize::from(channels), 16, u32::from(channels));
        round_trip(&pcm, channels, 16);
    }
    for bits in [20u8, 24, 32] {
        for channels in [1u8, 2, 6] {
            let pcm = signal(9_000, usize::from(channels), u32::from(bits), 11);
            round_trip(&pcm, channels, bits);
        }
    }
}

#[test]
fn it_compresses_and_escapes_noise() {
    let pcm = signal(4096 * 4, 2, 16, 3);
    let size = round_trip(&pcm, 2, 16);
    assert!(size < pcm.len() * 2 * 3 / 4, "{size} bytes for {} raw", pcm.len() * 2);
    // White noise at full scale does not compress: the frames escape, and
    // are barely larger than the samples.
    let mut rng = 12345u32;
    let noise: Vec<i32> = (0..4096 * 2)
        .map(|_| {
            rng = rng.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            (rng >> 16) as i16 as i32
        })
        .collect();
    let size = round_trip(&noise, 2, 16);
    assert!(size <= noise.len() * 2 + 16, "{size}");
}

#[test]
fn short_and_silent_streams_round_trip() {
    round_trip(&[5, -5, 7], 1, 16);
    round_trip(&vec![0; 20_000], 2, 24);
    round_trip(&signal(100, 2, 16, 1), 2, 16);
}

