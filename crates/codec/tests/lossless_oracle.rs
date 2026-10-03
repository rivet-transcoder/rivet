//! FLAC and ALAC through rivet's adapters and containers against
//! independent implementations, used as black boxes. The same checks as the
//! rivet-lossless crate's `tests/oracle.rs`, with the streams going in and
//! out through rivet-container's demuxer and muxers.
//!
//! - FLAC: `flac` (Xiph.Org's reference command-line tool). Its streams,
//!   native and muxed into Matroska by `mkvmerge` (MKVToolNix), must decode
//!   here to the PCM that went in; this crate's streams, native and in
//!   rivet's MP4 (taken apart again by `mkvmerge` / `mkvextract`), must
//!   decode in `flac -t` / `flac -d` to the PCM that went in.
//! - ALAC: `alacconvert`, Apple's reference encoder and decoder, built from
//!   Apple's open-source ALAC release (`crates/lossless/tools/
//!   build-alacconvert.sh`). Its CAF files, muxed into Matroska by
//!   `mkvmerge`, must decode here; this crate's ALAC in rivet's MP4, taken
//!   back to CAF by `mkvmerge` / `mkvextract`, must decode in Apple's
//!   decoder. Both bit-exact.
//!
//! - Both, above 65535 Hz: rivet's MP4s as `mkvmerge` and MediaInfo
//!   (MediaArea's `mediainfo`) read them — the rate and channel count each
//!   reports, where the sample entry's 16.16 rate field cannot hold the
//!   rate.
//!
//! Each test SKIPs (passes, printing why) when a tool it needs is missing,
//! so the suite runs anywhere — unless `RIVET_REQUIRE_LOSSLESS_ORACLES` is
//! set, as CI sets it, where a missing tool is a failure. `FLAC`,
//! `MKVMERGE`, `MKVEXTRACT`, `ALACCONVERT` and `MEDIAINFO` name the binaries; by default
//! each is looked for on PATH.

use std::path::{Path, PathBuf};
use std::process::Command;

use codec::audio::decode::{AlacDecoder, FlacDecoder};
use codec::audio::encode::flac::{FlacEncoderConfig, FlacLevel};
use codec::audio::encode::{AlacEncoder, FlacEncoder};

/// The tool `name` (or the binary its environment variable names) if it
/// runs; a panic instead of `None` under `RIVET_REQUIRE_LOSSLESS_ORACLES`.
fn tool(name: &str) -> Option<PathBuf> {
    let var = name.to_ascii_uppercase();
    let path = std::env::var_os(&var).map_or_else(|| PathBuf::from(name), PathBuf::from);
    let found = match name {
        // Run bare, `alacconvert` prints its usage and exits non-zero.
        "alacconvert" => Command::new(&path).output().is_ok_and(|o| {
            String::from_utf8_lossy(&[o.stdout, o.stderr].concat()).contains("alacconvert")
        }),
        "mediainfo" => Command::new(&path).arg("--Version").output().is_ok_and(|o| o.status.success()),
        _ => Command::new(&path).arg("--version").output().is_ok_and(|o| o.status.success()),
    };
    if !found && std::env::var_os("RIVET_REQUIRE_LOSSLESS_ORACLES").is_some() {
        panic!("RIVET_REQUIRE_LOSSLESS_ORACLES is set, and `{name}` does not run (set {var} or put it on PATH)");
    }
    found.then_some(path)
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("rivet-lossless-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(name)
}

fn run(cmd: &mut Command) {
    let out = cmd.output().expect("spawn");
    assert!(
        out.status.success(),
        "{cmd:?} failed: {}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
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


/// The audio track of a file, as the job engine reads it.
fn demux_audio(data: &[u8]) -> container::streaming::AudioSource {
    container::streaming::demux_audio(bytes::Bytes::copy_from_slice(data)).unwrap().expect("an audio track")
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

/// `flac`'s encode of `pcm` (raw, little-endian, signed) with `args`.
fn flac_encode(flac: &Path, pcm: &[i32], rate: u32, channels: usize, bits: u32, args: &[&str], name: &str) -> PathBuf {
    let raw = scratch(&format!("{name}.raw"));
    let out = scratch(&format!("{name}.flac"));
    std::fs::write(&raw, raw_bytes(pcm, bits)).unwrap();
    run(Command::new(flac)
        .args(["--silent", "-f", "--force-raw-format", "--endian=little", "--sign=signed"])
        .arg(format!("--channels={channels}"))
        .arg(format!("--bps={bits}"))
        .arg(format!("--sample-rate={rate}"))
        .args(args)
        .arg("-o")
        .arg(&out)
        .arg(&raw));
    out
}

/// `mkvmerge`'s Matroska of the one track in `input` (any container it
/// reads: native FLAC, CAF, MP4).
fn mkvmerge(mkvmerge: &Path, input: &Path, name: &str) -> PathBuf {
    let out = scratch(&format!("{name}.mkv"));
    run(Command::new(mkvmerge).args(["--quiet", "-o"]).arg(&out).arg(input));
    out
}

/// The one track of `input`, remuxed by `mkvmerge` and written out by
/// `mkvextract` in the codec's own file format (`ext`: `flac`, or `caf` for
/// ALAC).
fn remux_and_extract(mkvmerge_bin: &Path, mkvextract: &Path, input: &Path, name: &str, ext: &str) -> PathBuf {
    let mkv = mkvmerge(mkvmerge_bin, input, name);
    let out = scratch(&format!("{name}.extracted.{ext}"));
    run(Command::new(mkvextract).arg(&mkv).arg("tracks").arg(format!("0:{}", out.display())));
    out
}

#[test]
fn flac_cli_streams_decode_bit_exact() {
    let Some(flac) = tool("flac") else {
        eprintln!("SKIP: no `flac`");
        return;
    };
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
        let path = flac_encode(&flac, &pcm, rate, channels, bits, args, &format!("f{i}"));
        let src = demux_audio(&std::fs::read(&path).unwrap());
        assert_eq!(src.track.codec, "flac");
        let (got, dec) = decode_flac_track(&src.track);
        let label = format!("{rate} Hz {channels}ch {bits}-bit flac {args:?}");
        assert!(got == pcm, "{label}: {}", first_mismatch(&got, &pcm));
        assert_eq!(dec.md5_matches(), Some(true), "{label}: MD5");
        assert_eq!(src.track.durations.iter().map(|&d| d as usize).sum::<usize>(), pcm.len() / channels);
        eprintln!("ok: {label}");
    }
}

/// `flac`'s streams in Matroska as `mkvmerge` writes them (laced, as it
/// writes audio by default), through rivet's Matroska demuxer.
#[test]
fn flac_in_matroska_decodes_bit_exact() {
    let (Some(flac), Some(mkvmerge_bin)) = (tool("flac"), tool("mkvmerge")) else {
        eprintln!("SKIP: needs `flac` and `mkvmerge`");
        return;
    };
    for (i, &(rate, channels, bits)) in [(48_000u32, 2usize, 16u32), (96_000, 6, 24), (44_100, 1, 16)].iter().enumerate() {
        let pcm = signal(rate as usize + 777, channels, bits, 40 + i as u32);
        let native = flac_encode(&flac, &pcm, rate, channels, bits, &[], &format!("m{i}"));
        let mkv = mkvmerge(&mkvmerge_bin, &native, &format!("m{i}"));
        let src = demux_audio(&std::fs::read(&mkv).unwrap());
        assert_eq!(src.track.codec, "flac");
        let (got, _) = decode_flac_track(&src.track);
        assert!(got == pcm, "FLAC in Matroska {channels}ch {bits}-bit: {}", first_mismatch(&got, &pcm));
        assert_eq!(src.track.durations.iter().map(|&d| u64::from(d)).sum::<u64>(), (pcm.len() / channels) as u64);
        eprintln!("ok: FLAC in Matroska (mkvmerge) {rate} Hz {channels}ch {bits}-bit");
    }
}

/// For each native-order channel, the slot it takes in ALAC's own channel
/// order, as Apple's ALAC release documents the orders: 3 C L R; 4 C L R
/// Cs; 5 C L R Ls Rs; 6 C L R Ls Rs LFE; 7 C L R Ls Rs Cs LFE; 8 C Lc Rc L R
/// Ls Rs LFE. `alacconvert` does no channel reordering, so its PCM is in
/// these orders.
fn alac_slot(channels: usize) -> &'static [usize] {
    match channels {
        1 => &[0],
        2 => &[0, 1],
        3 => &[1, 2, 0],
        4 => &[1, 2, 0, 3],
        5 => &[1, 2, 0, 3, 4],
        6 => &[1, 2, 0, 5, 3, 4],
        7 => &[1, 2, 0, 6, 5, 3, 4],
        8 => &[3, 4, 0, 7, 5, 6, 1, 2],
        n => panic!("{n} channels"),
    }
}

/// Native-order interleaved PCM to ALAC order.
fn to_alac_order(pcm: &[i32], channels: usize) -> Vec<i32> {
    let slot = alac_slot(channels);
    let mut out = vec![0; pcm.len()];
    for (src, dst) in pcm.chunks_exact(channels).zip(out.chunks_exact_mut(channels)) {
        for (native, &s) in slot.iter().enumerate() {
            dst[s] = src[native];
        }
    }
    out
}

/// ALAC-order interleaved PCM to native order.
fn from_alac_order(pcm: &[i32], channels: usize) -> Vec<i32> {
    let slot = alac_slot(channels);
    pcm.chunks_exact(channels).flat_map(|f| slot.iter().map(|&s| f[s])).collect()
}

/// Apple's encoder takes 16-, 24- and 32-bit PCM of one to eight channels;
/// its decoder gives the same back. (Neither handles 20-bit ALAC: that
/// depth is checked by the round trips only.)
const APPLE_CASES: &[(u32, usize, u32)] = &[
    (44_100, 2, 16),
    (44_100, 1, 16),
    (48_000, 2, 24),
    (96_000, 2, 24),
    (48_000, 3, 16),
    (48_000, 4, 24),
    (48_000, 5, 16),
    (48_000, 6, 16),
    (96_000, 6, 24),
    (192_000, 2, 24),
    (48_000, 7, 16),
    (48_000, 8, 16),
    (48_000, 8, 24),
    (48_000, 2, 32),
];

/// Apple's ALAC, in Matroska as `mkvmerge` writes it from Apple's CAF,
/// through rivet's Matroska demuxer.
#[test]
fn apple_alac_decodes_bit_exact() {
    let (Some(alacconvert), Some(mkvmerge_bin)) = (tool("alacconvert"), tool("mkvmerge")) else {
        eprintln!("SKIP: needs `alacconvert` and `mkvmerge`");
        return;
    };
    for (i, &(rate, channels, bits)) in APPLE_CASES.iter().enumerate() {
        let pcm = signal(rate as usize * 3 / 2 + 99, channels, bits, 80 + i as u32);
        let src = scratch(&format!("a{i}.pcm.caf"));
        let caf = scratch(&format!("a{i}.alac.caf"));
        std::fs::write(&src, caf_pcm(&to_alac_order(&pcm, channels), rate, channels, bits)).unwrap();
        run(Command::new(&alacconvert).arg(&src).arg(&caf));
        let mkv = mkvmerge(&mkvmerge_bin, &caf, &format!("a{i}"));
        let track = demux_audio(&std::fs::read(&mkv).unwrap()).track;
        assert_eq!(track.codec, "alac");
        let got = decode_alac_track(&track);
        let label = format!("Apple ALAC in Matroska {rate} Hz {channels}ch {bits}-bit");
        assert!(got == pcm, "{label}: {}", first_mismatch(&got, &pcm));
        eprintln!("ok: {label}");
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

/// `flac -d` of `path` as raw PCM, after its MD5 check (`flac -t`).
fn flac_decode(flac: &Path, path: &Path, bits: u32) -> Vec<i32> {
    run(Command::new(flac).args(["--silent", "-t"]).arg(path));
    let raw = path.with_extension("dec.raw");
    run(Command::new(flac)
        .args(["--silent", "-f", "-d", "--force-raw-format", "--endian=little", "--sign=signed", "-o"])
        .arg(&raw)
        .arg(path));
    read_raw(&raw, bits)
}

#[test]
fn rivet_flac_decodes_bit_exact_in_flac() {
    let (Some(flac), Some(mkvmerge_bin), Some(mkvextract)) = (tool("flac"), tool("mkvmerge"), tool("mkvextract")) else {
        eprintln!("SKIP: needs `flac`, `mkvmerge` and `mkvextract`");
        return;
    };
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
        let got = flac_decode(&flac, &native, u32::from(bits));
        assert!(got == pcm, "{label} via flac -d: {}", first_mismatch(&got, &pcm));
        // FLAC in rivet's MP4, read by mkvmerge and written back out as a
        // native stream by mkvextract.
        let info = container::AudioInfo::flac(rate, u16::from(channels), blocks.clone());
        let mp4 = scratch(&format!("e{i}.mp4"));
        std::fs::write(&mp4, container::mux::write_audio_mp4(&info, &frames, Default::default()).unwrap()).unwrap();
        let extracted = remux_and_extract(&mkvmerge_bin, &mkvextract, &mp4, &format!("e{i}"), "flac");
        let got = flac_decode(&flac, &extracted, u32::from(bits));
        assert!(got == pcm, "{label} in MP4 via mkvmerge and flac -d: {}", first_mismatch(&got, &pcm));
        eprintln!("ok: {label}");
    }
}

/// This crate's ALAC in rivet's MP4, read by `mkvmerge` into Matroska, its
/// packets and cookie taken back out (by rivet's Matroska demuxer, which
/// `apple_alac_decodes_bit_exact` checks against `mkvmerge`'s output) into a
/// CAF file, and decoded by Apple's decoder. (`mkvextract` writes CAF too,
/// but not one `alacconvert` takes — no depth in its `desc`, the cookie in
/// QuickTime atoms — and not at all when the cookie carries its channel
/// layout, as rivet's MP4 does above two channels.)
#[test]
fn rivet_alac_decodes_bit_exact_in_apple_alac() {
    let (Some(alacconvert), Some(mkvmerge_bin)) = (tool("alacconvert"), tool("mkvmerge")) else {
        eprintln!("SKIP: needs `alacconvert` and `mkvmerge`");
        return;
    };
    for (i, &(rate, channels, bits)) in APPLE_CASES.iter().enumerate() {
        let pcm = signal(rate as usize * 3 / 2 + 321, channels, bits, 700 + i as u32);
        let (cookie, frames) = rivet_alac(&pcm, rate, channels as u8, bits as u8);
        let label = format!("rivet ALAC {rate} Hz {channels}ch {bits}-bit");
        let valid = (pcm.len() / channels) as u64;
        let caf = scratch(&format!("r{i}.alac.caf"));
        let info = container::AudioInfo::alac(rate, channels as u16, cookie.clone());
        let m4a = scratch(&format!("r{i}.m4a"));
        std::fs::write(&m4a, container::mux::write_audio_mp4(&info, &frames, Default::default()).unwrap())
            .unwrap();
        let mkv = mkvmerge(&mkvmerge_bin, &m4a, &format!("r{i}"));
        let track = demux_audio(&std::fs::read(&mkv).unwrap()).track;
        assert_eq!(track.codec, "alac", "{label}");
        assert_eq!((track.sample_rate, track.channels), (rate, channels as u16), "{label}: rate and channels in Matroska");
        assert_eq!(track.codec_private, cookie, "{label}: the cookie through MP4 and Matroska");
        assert_eq!(track.samples.len(), frames.len(), "{label}: packets through MP4 and Matroska");
        for (n, (got, (want, _))) in track.samples.iter().zip(&frames).enumerate() {
            assert!(got[..] == want[..], "{label}: packet {n} through MP4 and Matroska");
        }
        let packets: Vec<(Vec<u8>, u32)> = track.samples.into_iter().map(|p| (p.to_vec(), 0)).collect();
        std::fs::write(&caf, caf_alac(&cookie, rate, channels, bits, valid, &packets)).unwrap();
        let out = scratch(&format!("r{i}.pcm.caf"));
        run(Command::new(&alacconvert).arg(&caf).arg(&out));
        let decoded = read_caf(&std::fs::read(&out).unwrap()).pcm.unwrap_or_else(|| panic!("{label}: no PCM out"));
        let got = from_alac_order(&decoded, channels);
        assert!(got == pcm, "{label} via alacconvert: {}", first_mismatch(&got, &pcm));
        eprintln!("ok: {label}");
    }
}

/// rivet's FLAC and ALAC MP4s at rates the sample entry's 16.16 field
/// cannot hold, as two independent readers see them: `mkvmerge -J` and
/// MediaInfo must each report the true rate and channel count, and
/// `mkvmerge` must take the file without a warning. (Once the field held 0
/// there, and `mkvmerge` refused the track as broken header atoms.)
#[test]
fn high_rate_mp4_reads_right_in_mkvmerge_and_mediainfo() {
    let (Some(mkvmerge_bin), Some(mediainfo)) = (tool("mkvmerge"), tool("mediainfo")) else {
        eprintln!("SKIP: needs `mkvmerge` and `mediainfo`");
        return;
    };
    let mut n = 0;
    for &rate in &[44_100u32, 88_200, 96_000, 176_400, 192_000, 352_800, 384_000] {
        for &channels in &[2usize, 6] {
            for flac in [false, true] {
                let bits = 24u32;
                let pcm = signal(rate as usize / 4, channels, bits, 900 + n);
                let (info, frames) = if flac {
                    let (blocks, frames) = rivet_flac(&pcm, rate, channels as u8, bits as u8, FlacLevel::Fast);
                    (container::AudioInfo::flac(rate, channels as u16, blocks), frames)
                } else {
                    let (cookie, frames) = rivet_alac(&pcm, rate, channels as u8, bits as u8);
                    (container::AudioInfo::alac(rate, channels as u16, cookie), frames)
                };
                let label = format!("rivet {} MP4 {rate} Hz {channels}ch", if flac { "FLAC" } else { "ALAC" });
                let mp4 = scratch(&format!("h{n}.m4a"));
                n += 1;
                std::fs::write(&mp4, container::mux::write_audio_mp4(&info, &frames, Default::default()).unwrap())
                    .unwrap();
                let out = Command::new(&mkvmerge_bin).arg("-J").arg(&mp4).output().expect("spawn mkvmerge");
                let json = String::from_utf8_lossy(&out.stdout).replace(char::is_whitespace, "");
                assert!(json.contains("\"warnings\":[]") && json.contains("\"errors\":[]"), "{label}: mkvmerge -J: {json}");
                assert!(json.contains(&format!("\"audio_sampling_frequency\":{rate}")), "{label}: mkvmerge's rate: {json}");
                assert!(json.contains(&format!("\"audio_channels\":{channels}")), "{label}: mkvmerge's channels: {json}");
                let mkv = mkvmerge(&mkvmerge_bin, &mp4, &format!("h{n}"));
                let track = demux_audio(&std::fs::read(&mkv).unwrap()).track;
                assert_eq!((track.sample_rate, track.channels), (rate, channels as u16), "{label}: in mkvmerge's Matroska");
                let out = Command::new(&mediainfo)
                    .arg("--Inform=Audio;%Format%|%SamplingRate%|%Channel(s)%")
                    .arg(&mp4)
                    .output()
                    .expect("spawn mediainfo");
                let got = String::from_utf8_lossy(&out.stdout).trim().to_string();
                let want = format!("{}|{rate}|{channels}", if flac { "FLAC" } else { "ALAC" });
                assert_eq!(got, want, "{label}: MediaInfo's format|rate|channels");
                eprintln!("ok: {label}");
            }
        }
    }
}

/// Sizes against the reference encoders on a few synthetic signals, for
/// the record: printed, with nothing asserted beyond beating raw PCM.
#[test]
fn compression_against_the_reference_encoders() {
    let (Some(flac), Some(alacconvert)) = (tool("flac"), tool("alacconvert")) else {
        eprintln!("SKIP: needs `flac` and `alacconvert`");
        return;
    };
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
        "| signal | PCM bytes | rivet FLAC fast | rivet FLAC default | rivet FLAC best | flac -5 | rivet ALAC | Apple ALAC |"
    );
    for (i, (name, bits, pcm)) in signals.iter().enumerate() {
        let raw_len = pcm.len() * (*bits as usize / 8);
        let rivet: Vec<usize> = [FlacLevel::Fast, FlacLevel::Default, FlacLevel::Best]
            .iter()
            .map(|&l| {
                let (blocks, frames) = rivet_flac(pcm, rate, 2, *bits as u8, l);
                container::mux::write_native_flac(&blocks, &frames).unwrap().len()
            })
            .collect();
        let reference = flac_encode(&flac, pcm, rate, 2, *bits, &["-5"], &format!("c{i}"));
        let flac5 = std::fs::metadata(&reference).unwrap().len() as usize;
        // Both ALAC sizes are the packets alone.
        let (_, frames_a) = rivet_alac(pcm, rate, 2, *bits as u8);
        let rivet_alac_len: usize = frames_a.iter().map(|(f, _)| f.len()).sum();
        let src = scratch(&format!("c{i}.pcm.caf"));
        let out = scratch(&format!("c{i}.alac.caf"));
        std::fs::write(&src, caf_pcm(pcm, rate, 2, *bits)).unwrap();
        run(Command::new(&alacconvert).arg(&src).arg(&out));
        let apple_len: usize = read_caf(&std::fs::read(&out).unwrap()).alac.expect("ALAC").iter().map(Vec::len).sum();
        let pct = |n: usize| format!("{:.1}%", 100.0 * n as f64 / raw_len as f64);
        eprintln!(
            "| {name} | {raw_len} | {} | {} | {} | {} | {} | {} |",
            pct(rivet[0]),
            pct(rivet[1]),
            pct(rivet[2]),
            pct(flac5),
            pct(rivet_alac_len),
            pct(apple_len)
        );
        assert!(rivet.iter().all(|&n| n < raw_len) && rivet_alac_len < raw_len + raw_len / 50);
    }
}
// ---------------------------------------------------------------------------
// Core Audio Format (Apple's published CAF specification): what
// `alacconvert` reads and writes. A file header, then chunks of a type and a
// 64-bit big-endian size: `desc` (the stream format), `kuki` (the codec's
// magic cookie), `pakt` (the packet table) and `data` (a 32-bit edit count,
// then the audio).
// ---------------------------------------------------------------------------

const CAF_LPCM: u32 = u32::from_be_bytes(*b"lpcm");
const CAF_ALAC: u32 = u32::from_be_bytes(*b"alac");
/// `kCAFLinearPCMFormatFlagIsLittleEndian`.
const CAF_LITTLE_ENDIAN: u32 = 2;

fn caf_chunk(kind: &[u8; 4], body: &[u8]) -> Vec<u8> {
    let mut out = kind.to_vec();
    out.extend_from_slice(&(body.len() as i64).to_be_bytes());
    out.extend_from_slice(body);
    out
}

fn caf_desc(rate: u32, format: u32, flags: u32, bytes_per_packet: u32, frames_per_packet: u32, channels: usize, bits: u32) -> Vec<u8> {
    let mut d = f64::from(rate).to_be_bytes().to_vec();
    for v in [format, flags, bytes_per_packet, frames_per_packet, channels as u32, bits] {
        d.extend_from_slice(&v.to_be_bytes());
    }
    caf_chunk(b"desc", &d)
}

fn caf_data(audio: &[u8]) -> Vec<u8> {
    let mut body = 0u32.to_be_bytes().to_vec();
    body.extend_from_slice(audio);
    caf_chunk(b"data", &body)
}

/// Little-endian signed integer PCM, `bits` in whole bytes.
fn caf_pcm(pcm: &[i32], rate: u32, channels: usize, bits: u32) -> Vec<u8> {
    let width = bits.div_ceil(8);
    let mut out = b"caff\x00\x01\x00\x00".to_vec();
    out.extend(caf_desc(rate, CAF_LPCM, CAF_LITTLE_ENDIAN, width * channels as u32, 1, channels, bits));
    out.extend(caf_data(&raw_bytes(pcm, bits)));
    out
}

/// ALAC packets: `desc` names the depth in its flags (1–4 for 16, 20, 24
/// and 32 bits), `kuki` is the 24-byte cookie, `pakt` lists each packet's
/// size as a variable-length integer and the frames the last one pads.
fn caf_alac(cookie: &[u8], rate: u32, channels: usize, bits: u32, valid: u64, frames: &[(Vec<u8>, u32)]) -> Vec<u8> {
    let depth_flag = match bits {
        16 => 1,
        20 => 2,
        24 => 3,
        32 => 4,
        b => panic!("{b}-bit ALAC"),
    };
    let per_packet = u32::from_be_bytes(cookie[..4].try_into().unwrap());
    let mut pakt = (frames.len() as i64).to_be_bytes().to_vec();
    pakt.extend_from_slice(&(valid as i64).to_be_bytes());
    pakt.extend_from_slice(&0i32.to_be_bytes());
    pakt.extend_from_slice(&((u64::from(per_packet) * frames.len() as u64 - valid) as i32).to_be_bytes());
    for (f, _) in frames {
        let mut v = f.len() as u64;
        let mut bytes = vec![(v & 0x7F) as u8];
        v >>= 7;
        while v > 0 {
            bytes.push((v & 0x7F) as u8 | 0x80);
            v >>= 7;
        }
        pakt.extend(bytes.iter().rev());
    }
    let audio: Vec<u8> = frames.iter().flat_map(|(f, _)| f.iter().copied()).collect();
    let mut out = b"caff\x00\x01\x00\x00".to_vec();
    out.extend(caf_desc(rate, CAF_ALAC, depth_flag, 0, per_packet, channels, 0));
    out.extend(caf_chunk(b"kuki", cookie));
    out.extend(caf_chunk(b"pakt", &pakt));
    out.extend(caf_data(&audio));
    out
}

/// What a CAF file holds: integer PCM (interleaved, in the file's channel
/// order) or ALAC packets.
struct Caf {
    pcm: Option<Vec<i32>>,
    alac: Option<Vec<Vec<u8>>>,
}

fn read_caf(data: &[u8]) -> Caf {
    assert_eq!(&data[..4], b"caff", "a CAF file");
    let mut chunks = std::collections::HashMap::new();
    let mut at = 8;
    while at + 12 <= data.len() {
        let kind: [u8; 4] = data[at..at + 4].try_into().unwrap();
        let size = i64::from_be_bytes(data[at + 4..at + 12].try_into().unwrap());
        // A size of -1 is a `data` chunk running to the end of the file.
        let end = if size < 0 { data.len() } else { at + 12 + size as usize };
        chunks.insert(kind, &data[at + 12..end]);
        at = end;
    }
    let desc = chunks[b"desc"];
    let field = |i: usize| be32(desc, 8 + 4 * i);
    let (format, flags, bytes_per_packet, channels, bits) = (field(0), field(1), field(2), field(4), field(5));
    let audio = &chunks[b"data"][4..];
    if format == CAF_LPCM {
        assert_eq!(flags & 1, 0, "integer PCM");
        let width = (bytes_per_packet / channels) as usize;
        assert!(width * 8 >= bits as usize && width <= 4, "{bits} bits in {width} bytes");
        let pcm = audio
            .chunks_exact(width)
            .map(|b| {
                let mut v = [0u8; 4];
                if flags & CAF_LITTLE_ENDIAN != 0 {
                    v[4 - width..].copy_from_slice(b);
                    i32::from_le_bytes(v) >> (32 - 8 * width)
                } else {
                    v[..width].copy_from_slice(b);
                    i32::from_be_bytes(v) >> (32 - 8 * width)
                }
            })
            .collect();
        return Caf { pcm: Some(pcm), alac: None };
    }
    assert_eq!(format, CAF_ALAC, "lpcm or alac");
    let pakt = chunks[b"pakt"];
    let count = i64::from_be_bytes(pakt[..8].try_into().unwrap()) as usize;
    let mut at = 24;
    let mut packets = Vec::with_capacity(count);
    let mut offset = 0;
    for _ in 0..count {
        let mut len = 0usize;
        loop {
            let b = pakt[at];
            at += 1;
            len = (len << 7) | usize::from(b & 0x7F);
            if b & 0x80 == 0 {
                break;
            }
        }
        packets.push(audio[offset..offset + len].to_vec());
        offset += len;
    }
    Caf { pcm: None, alac: Some(packets) }
}

fn be32(b: &[u8], at: usize) -> u32 {
    u32::from_be_bytes(b[at..at + 4].try_into().unwrap())
}
