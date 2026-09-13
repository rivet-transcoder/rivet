//! Cross-check of the in-tree AC-3 / E-AC-3 decoder against libavcodec.
//!
//! The vectors are made by `crates/codec/tests/data/ac3_make_vectors.sh`
//! with the real ffmpeg binary: for each case an elementary stream and
//! libavcodec's f32le decode of it (`<name>.drc1.f32` with dynrng applied,
//! `<name>.drc0.f32` with `-drc_scale 0`). Point `RIVET_AC3_VECTORS` at that
//! directory to run the full sweep; without it the sweep is skipped and
//! only the committed 5.1 fixture (`tests/data/ac3_51_448k.*`) runs.
//!
//! Pass criterion
//! --------------
//! A/52:2018 defines the bit allocation in exact integer arithmetic (§7.2.1)
//! but leaves the transform and dequantisation to floating point, and lets
//! the dither and spectral-extension noise be "any reasonably random
//! sequence" (§7.3.4). Two conformant decoders therefore agree to float
//! rounding wherever the bit stream is deterministic, and differ by the
//! dither wherever it is not. So the gate (see `check`) is relative to a
//! measured *noise floor*: how much of our own output is noise fill, found
//! by decoding once more with the fill switched off. Numbers are in 16-bit LSBs
//! (1 LSB16 = 1/32768 ≈ 3.05e-5). A mis-transcribed table does not nudge
//! these numbers — it desynchronises the mantissa parse and the error jumps
//! to signal level (see `ac3_table_mutation` in this crate's tests).

use std::path::{Path, PathBuf};

use codec::audio::decode::ac3::{Ac3Decoder, Ac3Options, Features, FrameDecoder, frame_crc_ok, parse_header};
use codec::audio::AudioDecoder;

const LSB16: f32 = 1.0 / 32768.0;

struct Stats {
    channels: usize,
    rms: Vec<f32>,
    peak: Vec<f32>,
    compared: usize,
    /// Blocks left out of the comparison (see `FrameDecoder::mixed_transform_blocks`).
    masked_blocks: usize,
}

/// Per-channel RMS and peak difference over the common length, skipping
/// the 256-sample blocks listed in `masked` (global block indices).
fn compare(ours: &[f32], reference: &[f32], channels: usize, masked: &[u64]) -> Stats {
    let n = ours.len().min(reference.len()) / channels;
    let mut rms = vec![0.0f64; channels];
    let mut peak = vec![0.0f32; channels];
    let mut counted = 0usize;
    let mut masked_blocks = 0usize;
    for i in 0..n {
        if i % 256 == 0 && masked.contains(&((i / 256) as u64)) {
            masked_blocks += 1;
        }
        if masked.contains(&((i / 256) as u64)) {
            continue;
        }
        counted += 1;
        for c in 0..channels {
            let d = ours[i * channels + c] - reference[i * channels + c];
            rms[c] += f64::from(d) * f64::from(d);
            peak[c] = peak[c].max(d.abs());
        }
    }
    Stats {
        channels,
        rms: rms.iter().map(|s| (s / counted.max(1) as f64).sqrt() as f32).collect(),
        peak,
        compared: counted,
        masked_blocks,
    }
}

fn read_f32le(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn read_s16le(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    bytes.chunks_exact(2).map(|c| f32::from(i16::from_le_bytes([c[0], c[1]])) / 32768.0).collect()
}

struct Decoded {
    pcm: Vec<f32>,
    channels: usize,
    frames: usize,
    /// Zero-bit mantissas that were noise-filled, out of all mantissas.
    dithered: (u64, u64),
    /// Blocks carrying a non-unity `dynrng`, out of all blocks.
    drc: (u64, u64),
    /// Which coding tools the stream exercised.
    features: Features,
    /// The stream ended inside a syncframe (dropped, as libavcodec drops it).
    truncated_tail: bool,
    /// Blocks where the fbw channels used different transform lengths.
    /// libavcodec overlap-adds the switched channel's previous tail onto a
    /// neighbouring channel there (seen on two Dolby-encoded FATE streams,
    /// `millers_crossing_4.0` and `monsters_inc_2.0_192`), contrary to
    /// §7.9.4 step 6, so these blocks are masked out of the comparison and
    /// counted in the report.
    mixed_blocks: Vec<u64>,
}

/// Decode a whole elementary stream with the `FrameDecoder`, frame by
/// frame, checking every frame's CRC on the way. Leading junk before the
/// first syncword is skipped (some captured streams start mid-frame) and a
/// truncated final frame is dropped, both exactly as libavcodec does, so the
/// sample counts still line up.
fn decode_es(es: &[u8], drc_scale: f32, noise_fill: bool) -> Decoded {
    let mut dec = FrameDecoder::new(drc_scale);
    dec.set_noise_fill(noise_fill);
    let mut pcm = Vec::new();
    let mut pos = 0;
    let mut frames = 0;
    let mut channels = 0;
    let mut truncated_tail = false;
    while pos + 8 <= es.len() {
        let hdr = match parse_header(&es[pos..]) {
            Ok(h) => h,
            Err(e) => {
                assert_eq!(frames, 0, "frame {frames} at {pos}: {e}");
                pos += 1;
                continue;
            }
        };
        if pos + hdr.frame_len > es.len() {
            truncated_tail = true;
            break;
        }
        let frame = &es[pos..pos + hdr.frame_len];
        assert!(frame_crc_ok(frame), "frame {frames}: CRC-16 remainder is not zero");
        if let Some(h) = dec.decode(frame, &mut pcm).unwrap_or_else(|e| panic!("frame {frames}: {e}")) {
            channels = h.channels();
        }
        frames += 1;
        pos += hdr.frame_len;
    }
    Decoded {
        pcm,
        channels,
        frames,
        dithered: dec.dither_stats(),
        drc: dec.drc_stats(),
        features: dec.features(),
        truncated_tail,
        mixed_blocks: dec.mixed_transform_blocks().to_vec(),
    }
}

/// How much of our own output is §7.3.4 noise fill: the difference between
/// a normal decode and one with the noise fill switched off. Where nothing
/// is dithered it is 0. libavcodec's noise has the same amplitude and is
/// independent of ours, so the expected disagreement is √2 × this RMS.
fn noise_floor(es: &[u8], drc_scale: f32, ours: &Decoded) -> Stats {
    let silent = decode_es(es, drc_scale, false);
    compare(&ours.pcm, &silent.pcm, ours.channels, &ours.mixed_blocks)
}

fn report(name: &str, s: &Stats) -> String {
    let mut line = format!("{name}: {} samples/ch, {} ch;", s.compared, s.channels);
    if s.masked_blocks > 0 {
        line.push_str(&format!(" {} mixed-transform blocks masked;", s.masked_blocks));
    }
    for c in 0..s.channels {
        line.push_str(&format!(" ch{c} rms={:.3} peak={:.2} LSB16;", s.rms[c] / LSB16, s.peak[c] / LSB16));
    }
    line
}

/// The gate. Per channel, against libavcodec:
/// - RMS error ≤ max(1 LSB16, 1.5 × √2 × our noise-fill RMS). Two
///   independent ±0.707 noise sequences differ by √2 × one of them in
///   expectation; the 1.5 covers the realisation variance of short files
///   whose noise is dominated by a few high-exponent bins, and libavcodec's
///   freedom to use ±0.5 or ±0.75 instead (§7.3.4). 1 LSB16 is float
///   rounding plus a 16-bit reference's own rounding.
/// - peak error ≤ max(8 LSB16, 2.5 × our noise-fill peak): the peaks of two
///   independent noises add at most, and rarely coincide.
/// Everything deterministic in the decode — exponents, bit allocation,
/// mantissas, coupling, rematrixing, the transform — contributes only
/// float rounding, so a real bug shows up as a jump far beyond either bound
/// (the `hth` mutation in the report raises the RMS ~1000×).
fn check(name: &str, s: &Stats, noise: &Stats, d: &Decoded) {
    let mut line = report(name, s);
    line.push_str(&format!(
        " noise rms={} peak={} LSB16; dithered {}/{} bins; dynrng blocks {}/{}; features: {}",
        noise.rms.iter().map(|f| format!("{:.3}", f / LSB16)).collect::<Vec<_>>().join("/"),
        noise.peak.iter().map(|f| format!("{:.2}", f / LSB16)).collect::<Vec<_>>().join("/"),
        d.dithered.0,
        d.dithered.1,
        d.drc.0,
        d.drc.1,
        d.features
    ));
    println!("{line}");
    assert!(s.compared > 0, "{name}: nothing compared");
    // `RIVET_AC3_REPORT_ONLY` turns the gate into a report, to see every
    // vector's numbers in one run while tuning.
    let report_only = std::env::var_os("RIVET_AC3_REPORT_ONLY").is_some();
    for c in 0..s.channels {
        let expected = std::f32::consts::SQRT_2 * noise.rms[c];
        let rms_limit = (1.5 * expected).max(LSB16);
        let peak_limit = (2.5 * noise.peak[c]).max(8.0 * LSB16);
        println!(
            "  ch{c}: rms/expected = {:.2}, peak/noise-peak = {:.2}",
            s.rms[c] / expected.max(1e-12),
            s.peak[c] / noise.peak[c].max(1e-12)
        );
        let ok = s.rms[c] <= rms_limit && s.peak[c] <= peak_limit;
        if report_only && !ok {
            println!("  ch{c} OVER: rms limit {:.3}, peak limit {:.3} LSB16", rms_limit / LSB16, peak_limit / LSB16);
            continue;
        }
        assert!(s.rms[c] <= rms_limit, "{line}\n  ch{c} RMS over {:.3} LSB16", rms_limit / LSB16);
        assert!(s.peak[c] <= peak_limit, "{line}\n  ch{c} peak over {:.3} LSB16", peak_limit / LSB16);
    }
}

/// The committed fixture: 250 ms of 5.1 AC-3 at 448 kbit/s from ffmpeg's
/// encoder, with libavcodec's decode as s16le. Always runs.
#[test]
fn committed_5_1_fixture_matches_libavcodec() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data");
    let es = std::fs::read(dir.join("ac3_51_448k.ac3")).expect("fixture ac3_51_448k.ac3");
    let reference = read_s16le(&dir.join("ac3_51_448k.s16le"));
    let ours = decode_es(&es, 1.0, true);
    assert_eq!(ours.channels, 6);
    assert!(ours.frames >= 7, "expected ≥ 7 syncframes, got {}", ours.frames);
    assert_eq!(ours.pcm.len(), reference.len(), "sample count differs from libavcodec");
    let s = compare(&ours.pcm, &reference, 6, &ours.mixed_blocks);
    let noise = noise_floor(&es, 1.0, &ours);
    // The reference is 16-bit, so its own rounding contributes up to 0.5 LSB16.
    check("ac3_51_448k (s16 ref)", &s, &noise, &ours);
}

/// The `AudioDecoder` adapter must produce the same PCM as the frame
/// decoder when the stream arrives as arbitrary packet boundaries.
#[test]
fn audio_decoder_adapter_reassembles_split_frames() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data");
    let es = std::fs::read(dir.join("ac3_51_448k.ac3")).expect("fixture");
    let direct = decode_es(&es, 1.0, true).pcm;
    let mut dec = Ac3Decoder::with_options(48_000, 6, Ac3Options { drc_scale: 1.0 }).unwrap();
    let mut out = Vec::new();
    let mut pts_seen = Vec::new();
    for (i, chunk) in es.chunks(1000).enumerate() {
        for f in dec.decode(chunk, if i == 0 { 5_000 } else { 0 }).unwrap() {
            assert_eq!(f.channels, 6);
            assert_eq!(f.sample_rate, 48_000);
            pts_seen.push(f.pts);
            out.extend_from_slice(&f.samples);
        }
    }
    out.extend(dec.flush().unwrap().into_iter().flat_map(|f| f.samples));
    assert_eq!(out, direct);
    assert_eq!(pts_seen[0], 5_000);
    assert_eq!(pts_seen[1], 5_000 + 32_000, "one 1536-sample syncframe is 32 ms");
}

fn vectors_dir() -> Option<PathBuf> {
    let dir = PathBuf::from(std::env::var_os("RIVET_AC3_VECTORS")?);
    dir.is_dir().then_some(dir)
}

/// Every vector in `RIVET_AC3_VECTORS`, both with and without dynamic range
/// compression, against libavcodec's f32 decode.
#[test]
fn ffmpeg_vector_sweep_matches_libavcodec() {
    let Some(dir) = vectors_dir() else {
        eprintln!("RIVET_AC3_VECTORS not set — skipping the libavcodec sweep");
        return;
    };
    let mut names: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| matches!(p.extension().and_then(|e| e.to_str()), Some("ac3" | "eac3")))
        .collect();
    names.sort();
    assert!(!names.is_empty(), "no .ac3/.eac3 vectors in {}", dir.display());
    let mut lines = Vec::new();
    for es_path in &names {
        let stem = es_path.file_stem().unwrap().to_string_lossy().to_string();
        let es = std::fs::read(es_path).unwrap();
        for (tag, drc) in [("drc1", 1.0f32), ("drc0", 0.0)] {
            let ref_path = dir.join(format!("{stem}.{tag}.f32"));
            if !ref_path.is_file() {
                continue;
            }
            let reference = read_f32le(&ref_path);
            let ours = decode_es(&es, drc, true);
            if ours.truncated_tail {
                // libavcodec conceals a cut-off final frame (it emits one
                // frame of concealment); we drop it. Compare the common part.
                let extra = reference.len().saturating_sub(ours.pcm.len());
                assert!(
                    extra <= 6 * 256 * ours.channels,
                    "{stem}: libavcodec output is {extra} samples longer than ours, more than one frame"
                );
            } else {
                assert_eq!(ours.pcm.len(), reference.len(), "{stem}: sample count differs from libavcodec");
            }
            let s = compare(&ours.pcm, &reference, ours.channels, &ours.mixed_blocks);
            let noise = noise_floor(&es, drc, &ours);
            let name = format!("{stem}.{tag}");
            check(&name, &s, &noise, &ours);
            lines.push(format!(
                "{} noise rms={} peak={}; dithered {}/{}; dynrng {}/{}; features: {}",
                report(&name, &s),
                noise.rms.iter().map(|f| format!("{:.3}", f / LSB16)).collect::<Vec<_>>().join("/"),
                noise.peak.iter().map(|f| format!("{:.2}", f / LSB16)).collect::<Vec<_>>().join("/"),
                ours.dithered.0,
                ours.dithered.1,
                ours.drc.0,
                ours.drc.1,
                ours.features
            ));
        }
    }
    // A summary the report can quote.
    std::fs::write(dir.join("ac3_sweep_report.txt"), lines.join("\n") + "\n").unwrap();
}
