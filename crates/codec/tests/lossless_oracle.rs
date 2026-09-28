//! FLAC and ALAC against independent implementations, used as black boxes.
//!
//! Decode: synthetic PCM → the `flac` command-line encoder or ffmpeg's ALAC
//! encoder → this crate's decoder, which must give back the PCM exactly.
//! Encode: synthetic PCM → this crate's encoder → `flac -d` and ffmpeg,
//! which must give back the PCM exactly.
//!
//! Each test SKIPs (passes, printing why) when the tool it needs is not on
//! PATH, so the suite runs anywhere; CI images with `flac` and `ffmpeg`
//! installed run the comparisons.

use std::path::{Path, PathBuf};
use std::process::Command;

use codec::audio::decode::{AlacDecoder, FlacDecoder};
use codec::audio::encode::flac::{FlacEncoderConfig, FlacLevel};
use codec::audio::encode::{AlacEncoder, FlacEncoder};

fn have(tool: &str) -> bool {
    let arg = if tool == "flac" { "--version" } else { "-version" };
    Command::new(tool).arg(arg).output().is_ok_and(|o| o.status.success())
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("rivet-lossless-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(name)
}

fn run(cmd: &mut Command) {
    let out = cmd.output().expect("spawn");
    assert!(out.status.success(), "{cmd:?} failed: {}", String::from_utf8_lossy(&out.stderr));
}

/// Deterministic test audio: per channel, a mix of a few tones and noise,
/// the channels partly correlated (so every stereo mode has something to
/// win), a stretch of digital silence, a stretch of a constant, and a burst
/// at full scale.
fn signal(frames: usize, channels: usize, bits: u32, seed: u32) -> Vec<i32> {
    let full = f64::from((1u32 << (bits - 1)) - 1);
    let mut rng = seed.wrapping_mul(2_654_435_761).max(1);
    let mut noise = move || {
        rng ^= rng << 13;
        rng ^= rng >> 17;
        rng ^= rng << 5;
        f64::from(rng) / f64::from(u32::MAX) - 0.5
    };
    let mut out = Vec::with_capacity(frames * channels);
    for i in 0..frames {
        let t = i as f64 / 48_000.0;
        let common = (t * 2.0 * std::f64::consts::PI * 220.0).sin() * 0.4
            + (t * 2.0 * std::f64::consts::PI * 1_375.0).sin() * 0.1;
        let n = noise();
        for c in 0..channels {
            let own = (t * 2.0 * std::f64::consts::PI * (330.0 + 110.0 * c as f64)).sin() * 0.2;
            let v = if i > frames / 3 && i < frames / 3 + 5_000 {
                0.0
            } else if i >= frames / 2 && i < frames / 2 + 3_000 {
                0.25
            } else if i >= frames * 3 / 4 && i < frames * 3 / 4 + 500 {
                if (i / 7 + c) % 2 == 0 { 1.0 } else { -1.0 }
            } else {
                common + own * (c as f64 * 0.3) + n * 0.05 + noise() * 0.02
            };
            let s = (v * full).round().clamp(-full - 1.0, full) as i64;
            out.push(s as i32);
        }
    }
    out
}

/// Little-endian raw PCM in the fewest whole bytes per sample.
fn raw_bytes(samples: &[i32], bits: u32) -> Vec<u8> {
    let width = bits.div_ceil(8) as usize;
    samples.iter().flat_map(|s| s.to_le_bytes()[..width].to_vec()).collect()
}

fn read_raw(path: &Path, bits: u32) -> Vec<i32> {
    let width = bits.div_ceil(8) as usize;
    std::fs::read(path)
        .unwrap()
        .chunks_exact(width)
        .map(|b| {
            let mut v = [0u8; 4];
            v[4 - width..].copy_from_slice(b);
            i32::from_le_bytes(v) >> (32 - 8 * width)
        })
        .collect()
}

fn decode_flac_track(track: &container::demux::AudioTrack) -> (Vec<i32>, FlacDecoder) {
    let mut dec = FlacDecoder::new(Some(&track.codec_private), track.sample_rate, track.channels as u8).unwrap();
    let mut out = Vec::new();
    for p in &track.samples {
        out.extend(dec.decode_int(p).unwrap().0);
    }
    (out, dec)
}

fn decode_alac_track(track: &container::demux::AudioTrack) -> Vec<i32> {
    let mut dec = AlacDecoder::new(Some(&track.codec_private)).unwrap();
    let mut out = Vec::new();
    for p in &track.samples {
        out.extend(dec.decode_int(p).unwrap());
    }
    out
}

fn first_mismatch(a: &[i32], b: &[i32]) -> String {
    match a.iter().zip(b).position(|(x, y)| x != y) {
        Some(i) => format!("first difference at sample {i}: got {} want {}", a[i], b[i]),
        None => format!("lengths differ: got {} want {}", a.len(), b.len()),
    }
}

#[test]
fn flac_cli_streams_decode_bit_exact() {
    if !have("flac") {
        eprintln!("SKIP: no `flac` on PATH");
        return;
    }
    let cases: &[(u32, usize, u32, &[&str])] = &[
        (44_100, 2, 16, &["-5"]),
        (44_100, 2, 16, &["-0"]),
        (44_100, 2, 16, &["-8"]),
        (48_000, 1, 16, &["-5"]),
        (48_000, 2, 24, &["-8"]),
        (96_000, 2, 24, &["-5", "--no-mid-side"]),
        (48_000, 6, 16, &["-5"]),
        (96_000, 6, 24, &["-8"]),
        (48_000, 8, 24, &["-5"]),
        (44_100, 2, 8, &["-5"]),
        (48_000, 2, 32, &["-5"]),
        (44_100, 2, 16, &["-5", "-b", "1152"]),
        (22_050, 3, 16, &["-3"]),
        (32_000, 5, 16, &["-5"]),
        (48_000, 7, 16, &["-5"]),
        (48_000, 4, 16, &["-5", "-b", "576"]),
        (192_000, 2, 24, &["-8", "-l", "32"]),
    ];
    for (i, &(rate, channels, bits, args)) in cases.iter().enumerate() {
        let pcm = signal(rate as usize * 3 / 2 + 1_234, channels, bits, i as u32 + 1);
        let raw = scratch(&format!("f{i}.raw"));
        let flac = scratch(&format!("f{i}.flac"));
        std::fs::write(&raw, raw_bytes(&pcm, bits)).unwrap();
        run(Command::new("flac")
            .args(["--silent", "-f", "--force-raw-format", "--endian=little", "--sign=signed"])
            .arg(format!("--channels={channels}"))
            .arg(format!("--bps={bits}"))
            .arg(format!("--sample-rate={rate}"))
            .args(args)
            .arg("-o")
            .arg(&flac)
            .arg(&raw));
        let data = std::fs::read(&flac).unwrap();
        let src = container::demux::demux_audio(&data).unwrap();
        assert_eq!(src.track.codec, "flac");
        let (got, dec) = decode_flac_track(&src.track);
        let label = format!("{rate} Hz {channels}ch {bits}-bit flac {args:?}");
        assert!(got == pcm, "{label}: {}", first_mismatch(&got, &pcm));
        assert_eq!(dec.md5_matches(), Some(true), "{label}: MD5");
        assert_eq!(src.track.durations.iter().map(|&d| d as usize).sum::<usize>(), pcm.len() / channels);
        eprintln!("ok: {label}");
    }
}

#[test]
fn flac_in_mp4_and_matroska_decodes_bit_exact() {
    if !have("flac") || !have("ffmpeg") {
        eprintln!("SKIP: needs `flac` and `ffmpeg` on PATH");
        return;
    }
    for (i, &(rate, channels, bits)) in [(48_000u32, 2usize, 16u32), (96_000, 6, 24)].iter().enumerate() {
        let pcm = signal(rate as usize + 777, channels, bits, 40 + i as u32);
        let raw = scratch(&format!("m{i}.raw"));
        let flac = scratch(&format!("m{i}.flac"));
        std::fs::write(&raw, raw_bytes(&pcm, bits)).unwrap();
        run(Command::new("flac")
            .args(["--silent", "-f", "--force-raw-format", "--endian=little", "--sign=signed"])
            .arg(format!("--channels={channels}"))
            .arg(format!("--bps={bits}"))
            .arg(format!("--sample-rate={rate}"))
            .arg("-o")
            .arg(&flac)
            .arg(&raw));
        for ext in ["mp4", "mkv"] {
            let out = scratch(&format!("m{i}.{ext}"));
            run(Command::new("ffmpeg")
                .args(["-loglevel", "error", "-y", "-i"])
                .arg(&flac)
                .args(["-c:a", "copy", "-strict", "-2"])
                .arg(&out));
            let src = container::demux::demux_audio(&std::fs::read(&out).unwrap()).unwrap();
            assert_eq!(src.track.codec, "flac", "{ext}");
            let (got, _) = decode_flac_track(&src.track);
            assert!(got == pcm, "FLAC in {ext} {channels}ch {bits}-bit: {}", first_mismatch(&got, &pcm));
            assert_eq!(src.track.durations.iter().map(|&d| u64::from(d)).sum::<u64>(), (pcm.len() / channels) as u64);
            eprintln!("ok: FLAC in {ext} {rate} Hz {channels}ch {bits}-bit");
        }
    }
}

#[test]
fn ffmpeg_alac_decodes_bit_exact() {
    if !have("ffmpeg") {
        eprintln!("SKIP: no `ffmpeg` on PATH");
        return;
    }
    let cases: &[(u32, usize, u32)] = &[
        (44_100, 2, 16),
        (44_100, 1, 16),
        (48_000, 2, 24),
        (96_000, 2, 24),
        (48_000, 6, 16),
        (96_000, 6, 24),
        (48_000, 8, 16),
        (48_000, 3, 16),
        (48_000, 4, 16),
        (48_000, 5, 24),
        (48_000, 7, 16),
    ];
    for (i, &(rate, channels, bits)) in cases.iter().enumerate() {
        let pcm = signal(rate as usize * 3 / 2 + 99, channels, bits, 80 + i as u32);
        let raw = scratch(&format!("a{i}.raw"));
        std::fs::write(&raw, raw_bytes(&pcm, bits)).unwrap();
        let fmt = if bits == 16 { "s16le" } else { "s24le" };
        for ext in ["m4a", "mkv"] {
            let out = scratch(&format!("a{i}.{ext}"));
            let mut cmd = Command::new("ffmpeg");
            cmd.args(["-loglevel", "error", "-y", "-f", fmt, "-ar", &rate.to_string(), "-ac", &channels.to_string()]);
            // Name the layouts ALAC has for these counts: given ffmpeg's
            // defaults (2.1, quad, 7.1) its encoder silently remixes.
            if let Some(layout) = match channels {
                3 => Some("3.0"),
                4 => Some("4.0"),
                5 => Some("5.0"),
                8 => Some("7.1(wide)"),
                _ => None,
            } {
                cmd.args(["-channel_layout", layout]);
            }
            run(cmd
                .arg("-i")
                .arg(&raw)
                .args(["-c:a", "alac"])
                .arg(&out));
            let src = container::demux::demux_audio(&std::fs::read(&out).unwrap()).unwrap();
            assert_eq!(src.track.codec, "alac", "{ext}");
            let got = decode_alac_track(&src.track);
            let label = format!("ALAC in {ext} {rate} Hz {channels}ch {bits}-bit");
            assert!(got == pcm, "{label}: {}", first_mismatch(&got, &pcm));
            eprintln!("ok: {label}");
        }
    }
}

fn rivet_flac(pcm: &[i32], rate: u32, channels: u8, bits: u8, level: FlacLevel) -> (Vec<u8>, Vec<(Vec<u8>, u32)>) {
    let mut enc =
        FlacEncoder::new(FlacEncoderConfig { sample_rate: rate, channels, bits_per_sample: bits, level }).unwrap();
    let mut frames = enc.encode_int(pcm);
    frames.extend(enc.finish());
    (enc.metadata_blocks(), frames)
}

fn rivet_alac(pcm: &[i32], rate: u32, channels: u8, bits: u8) -> (Vec<u8>, Vec<(Vec<u8>, u32)>) {
    let mut enc = AlacEncoder::new(rate, channels, bits).unwrap();
    let mut frames = enc.encode_int(pcm);
    frames.extend(enc.finish());
    (enc.cookie().to_bytes().to_vec(), frames)
}

/// ffmpeg's decode of `path` as raw PCM at `bits`.
fn ffmpeg_decode(path: &Path, bits: u32) -> Vec<i32> {
    let out = path.with_extension("ffdec.raw");
    let fmt = match bits {
        16 => "s16le",
        24 => "s24le",
        _ => "s32le",
    };
    run(Command::new("ffmpeg")
        .args(["-loglevel", "error", "-y", "-i"])
        .arg(path)
        .args(["-c:a", &format!("pcm_{fmt}"), "-f", fmt])
        .arg(&out));
    read_raw(&out, bits)
}

#[test]
fn rivet_flac_decodes_bit_exact_in_flac_and_ffmpeg() {
    if !have("flac") || !have("ffmpeg") {
        eprintln!("SKIP: needs `flac` and `ffmpeg` on PATH");
        return;
    }
    let cases: &[(u32, u8, u8, FlacLevel)] = &[
        (44_100, 2, 16, FlacLevel::Fast),
        (44_100, 2, 16, FlacLevel::Default),
        (44_100, 2, 16, FlacLevel::Best),
        (48_000, 1, 16, FlacLevel::Default),
        (48_000, 2, 24, FlacLevel::Default),
        (96_000, 2, 24, FlacLevel::Best),
        (48_000, 6, 16, FlacLevel::Default),
        (96_000, 6, 24, FlacLevel::Default),
        (48_000, 8, 24, FlacLevel::Default),
        (192_000, 2, 24, FlacLevel::Fast),
        (48_000, 2, 32, FlacLevel::Default),
        (22_050, 3, 16, FlacLevel::Default),
    ];
    for (i, &(rate, channels, bits, level)) in cases.iter().enumerate() {
        let pcm = signal(rate as usize * 3 / 2 + 555, usize::from(channels), u32::from(bits), 200 + i as u32);
        let (blocks, frames) = rivet_flac(&pcm, rate, channels, bits, level);
        let label = format!("rivet FLAC {rate} Hz {channels}ch {bits}-bit {level:?}");
        let native = scratch(&format!("e{i}.flac"));
        std::fs::write(&native, container::mux::write_native_flac(&blocks, &frames).unwrap()).unwrap();
        // The reference decoder, with its MD5 check.
        run(Command::new("flac").args(["--silent", "-t"]).arg(&native));
        let raw = scratch(&format!("e{i}.dec.raw"));
        run(Command::new("flac")
            .args(["--silent", "-f", "-d", "--force-raw-format", "--endian=little", "--sign=signed", "-o"])
            .arg(&raw)
            .arg(&native));
        let got = read_raw(&raw, u32::from(bits));
        assert!(got == pcm, "{label} via flac -d: {}", first_mismatch(&got, &pcm));
        if bits != 32 {
            let got = ffmpeg_decode(&native, u32::from(bits));
            assert!(got == pcm, "{label} via ffmpeg: {}", first_mismatch(&got, &pcm));
            // FLAC in MP4.
            let info = container::AudioInfo::flac(rate, u16::from(channels), blocks.clone());
            let mp4 = scratch(&format!("e{i}.mp4"));
            std::fs::write(&mp4, container::mux::write_audio_mp4(&info, &frames, Default::default()).unwrap())
                .unwrap();
            let got = ffmpeg_decode(&mp4, u32::from(bits));
            assert!(got == pcm, "{label} in MP4 via ffmpeg: {}", first_mismatch(&got, &pcm));
        }
        eprintln!("ok: {label}");
    }
}

#[test]
fn rivet_alac_decodes_bit_exact_in_ffmpeg() {
    if !have("ffmpeg") {
        eprintln!("SKIP: no `ffmpeg` on PATH");
        return;
    }
    let cases: &[(u32, u8, u8)] = &[
        (44_100, 2, 16),
        (44_100, 1, 16),
        (48_000, 2, 24),
        (96_000, 2, 24),
        (48_000, 6, 16),
        (96_000, 6, 24),
        (48_000, 8, 16),
        (48_000, 3, 16),
        (48_000, 4, 24),
        (48_000, 5, 16),
        (48_000, 7, 16),
        (48_000, 2, 20),
        (48_000, 2, 32),
    ];
    for (i, &(rate, channels, bits)) in cases.iter().enumerate() {
        let pcm = signal(rate as usize * 3 / 2 + 321, usize::from(channels), u32::from(bits), 300 + i as u32);
        let (cookie, frames) = rivet_alac(&pcm, rate, channels, bits);
        let info = container::AudioInfo::alac(rate, u16::from(channels), cookie);
        let m4a = scratch(&format!("r{i}.m4a"));
        std::fs::write(&m4a, container::mux::write_audio_mp4(&info, &frames, Default::default()).unwrap()).unwrap();
        // 20-bit audio comes out of ffmpeg left-justified in 24 bits.
        let (out_bits, shift) = match bits {
            20 => (24, 4),
            b => (u32::from(b), 0),
        };
        let mut got: Vec<i32> = ffmpeg_decode(&m4a, out_bits).into_iter().map(|s| s >> shift).collect();
        // ffmpeg names ALAC's seven channels 6.1(back) (… BL BR BC); the
        // pipeline's 6.1 is FL FR FC LFE BC SL SR.
        if channels == 7 {
            for f in got.chunks_exact_mut(7) {
                let (ls, rs, cs) = (f[4], f[5], f[6]);
                f[4..7].copy_from_slice(&[cs, ls, rs]);
            }
        }
        let label = format!("rivet ALAC {rate} Hz {channels}ch {bits}-bit");
        assert!(got == pcm, "{label} via ffmpeg: {}", first_mismatch(&got, &pcm));
        eprintln!("ok: {label}");
    }
}

/// Sizes against the reference encoders on a few synthetic signals, for
/// the record: printed, with nothing asserted beyond beating raw PCM.
#[test]
fn compression_against_the_reference_encoders() {
    if !have("flac") || !have("ffmpeg") {
        eprintln!("SKIP: needs `flac` and `ffmpeg` on PATH");
        return;
    }
    let rate = 44_100u32;
    let frames = rate as usize * 10;
    let mut rng = 99u32;
    let mut noise = move || {
        rng ^= rng << 13;
        rng ^= rng >> 17;
        rng ^= rng << 5;
        f64::from(rng) / f64::from(u32::MAX) - 0.5
    };
    let mut brown = [0.0f64; 2];
    let sine: Vec<i32> = (0..frames)
        .flat_map(|i| {
            let v = ((i as f64 / 44_100.0 * 2.0 * std::f64::consts::PI * 1000.0).sin() * 20_000.0) as i32;
            [v, v / 2]
        })
        .collect();
    let brownian: Vec<i32> = (0..frames)
        .flat_map(|_| {
            let mut out = [0i32; 2];
            for (c, p) in brown.iter_mut().enumerate() {
                *p = (*p * 0.995 + noise() * 800.0).clamp(-32_000.0, 32_000.0);
                out[c] = *p as i32;
            }
            out
        })
        .collect();
    let signals: Vec<(&str, u32, Vec<i32>)> = vec![
        ("tones+noise 16-bit", 16, signal(frames, 2, 16, 1)),
        ("tones+noise 24-bit", 24, signal(frames, 2, 24, 2)),
        ("sine 1 kHz 16-bit", 16, sine),
        ("brown noise 16-bit", 16, brownian),
    ];
    eprintln!(
        "| signal | PCM bytes | rivet FLAC fast | rivet FLAC default | rivet FLAC best | flac -5 | rivet ALAC | ffmpeg ALAC |"
    );
    for (i, (name, bits, pcm)) in signals.iter().enumerate() {
        let raw_len = pcm.len() * (*bits as usize / 8);
        let raw = scratch(&format!("c{i}.raw"));
        std::fs::write(&raw, raw_bytes(pcm, *bits)).unwrap();
        let rivet: Vec<usize> = [FlacLevel::Fast, FlacLevel::Default, FlacLevel::Best]
            .iter()
            .map(|&l| {
                let (blocks, frames) = rivet_flac(pcm, rate, 2, *bits as u8, l);
                container::mux::write_native_flac(&blocks, &frames).unwrap().len()
            })
            .collect();
        let reference = scratch(&format!("c{i}.flac"));
        run(Command::new("flac")
            .args(["--silent", "-f", "-5", "--force-raw-format", "--endian=little", "--sign=signed", "--channels=2"])
            .arg(format!("--bps={bits}"))
            .arg(format!("--sample-rate={rate}"))
            .arg("-o")
            .arg(&reference)
            .arg(&raw));
        let flac5 = std::fs::metadata(&reference).unwrap().len() as usize;
        let (cookie, frames_a) = rivet_alac(pcm, rate, 2, *bits as u8);
        let info = container::AudioInfo::alac(rate, 2, cookie);
        let rivet_alac_len = container::mux::write_audio_mp4(&info, &frames_a, Default::default()).unwrap().len();
        let ff = scratch(&format!("c{i}.m4a"));
        let fmt = if *bits == 16 { "s16le" } else { "s24le" };
        run(Command::new("ffmpeg")
            .args(["-loglevel", "error", "-y", "-f", fmt, "-ar", &rate.to_string(), "-ac", "2", "-i"])
            .arg(&raw)
            .args(["-c:a", "alac"])
            .arg(&ff));
        let ff_alac = std::fs::metadata(&ff).unwrap().len() as usize;
        let pct = |n: usize| format!("{:.1}%", 100.0 * n as f64 / raw_len as f64);
        eprintln!(
            "| {name} | {raw_len} | {} | {} | {} | {} | {} | {} |",
            pct(rivet[0]),
            pct(rivet[1]),
            pct(rivet[2]),
            pct(flac5),
            pct(rivet_alac_len),
            pct(ff_alac)
        );
        assert!(rivet.iter().all(|&n| n < raw_len) && rivet_alac_len < raw_len + raw_len / 50);
    }
}
