//! DTS core decoder against libavcodec.
//!
//! ffmpeg's `dca` encoder makes the vectors from known lavfi signals, its
//! decoder turns them back into float PCM, and this crate's decoder has to
//! agree with that PCM per channel. ffmpeg is not a dependency of rivet: the
//! tests skip with a message when it is not on PATH, the same way
//! `rivet/tests/fidelity_ffprobe.rs` does. They run on the developer boxes
//! and the CI images that carry ffmpeg.
//!
//! What ffmpeg's encoder exercises: Huffman, block-coded and linear
//! quantisation indices, 6/7-bit scale factors, joint-intensity-free 5.1 and
//! stereo, the LFE 64× path, both QMF prototypes. What it never produces:
//! ADPCM prediction and high-frequency VQ subbands, whose code books ETSI
//! does not print (see the decoder's module docs).

use std::path::PathBuf;
use std::process::Command;

use codec::audio::create_decoder;

fn ffmpeg_available() -> bool {
    Command::new("ffmpeg")
        .arg("-version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn scratch_dir() -> PathBuf {
    let dir = std::env::temp_dir().join(format!("rivet_dts_core_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("scratch dir");
    dir
}

fn run_ffmpeg(args: &[&str]) {
    let out = Command::new("ffmpeg")
        .args(["-hide_banner", "-loglevel", "error", "-y"])
        .args(args)
        .output()
        .expect("spawn ffmpeg");
    assert!(
        out.status.success(),
        "ffmpeg {} failed: {}",
        args.join(" "),
        String::from_utf8_lossy(&out.stderr)
    );
}

/// Encode a lavfi `graph` (one audio output) with ffmpeg's DTS encoder, and
/// decode the result back with libavcodec: `(dts bytes, reference f32 PCM)`.
fn make_vector(name: &str, graph: &str, bitrate: &str) -> (Vec<u8>, Vec<f32>) {
    let dir = scratch_dir();
    let dts = dir.join(format!("{name}.dts"));
    let raw = dir.join(format!("{name}.f32"));
    run_ffmpeg(&[
        "-f", "lavfi", "-i", graph, "-c:a", "dca", "-strict", "-2", "-b:a", bitrate,
        dts.to_str().unwrap(),
    ]);
    run_ffmpeg(&["-i", dts.to_str().unwrap(), "-c:a", "pcm_f32le", "-f", "f32le", raw.to_str().unwrap()]);
    let dts_bytes = std::fs::read(&dts).unwrap();
    let raw_bytes = std::fs::read(&raw).unwrap();
    let reference: Vec<f32> = raw_bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    let _ = std::fs::remove_file(&dts);
    let _ = std::fs::remove_file(&raw);
    let _ = std::fs::remove_dir(&dir);
    (dts_bytes, reference)
}

/// `FSIZE + 1` of the core frame at the start of `b` (14 bits at bit 46).
fn core_frame_len(b: &[u8]) -> usize {
    assert_eq!(&b[..4], &[0x7F, 0xFE, 0x80, 0x01], "core sync word");
    let fsize = ((b[5] as usize & 0x03) << 12) | ((b[6] as usize) << 4) | (b[7] as usize >> 4);
    fsize + 1
}

/// Feed the raw stream one core frame per packet, as a container would, and
/// return `(interleaved PCM, channels, sample rate)`.
fn decode_all(dts: &[u8], container_channels: u8) -> (Vec<f32>, usize, u32) {
    let mut dec = create_decoder("dts", None, 48_000, container_channels).expect("dts decoder");
    let mut pcm = Vec::new();
    let (mut channels, mut rate) = (0usize, 0u32);
    let mut off = 0;
    let mut pts = 0i64;
    while off + 8 <= dts.len() {
        let len = core_frame_len(&dts[off..]);
        let frames = dec.decode(&dts[off..off + len], pts).unwrap_or_else(|e| panic!("frame at byte {off}: {e}"));
        for f in frames {
            channels = f.channels as usize;
            rate = f.sample_rate;
            pts = f.pts + (f.samples.len() / channels) as i64 * 1_000_000 / rate as i64;
            pcm.extend_from_slice(&f.samples);
        }
        off += len;
    }
    pcm.extend(dec.flush().unwrap().into_iter().flat_map(|f| f.samples));
    (pcm, channels, rate)
}

struct ChannelStats {
    max_abs_err: f32,
    rms_err: f64,
    rms_ref: f64,
}

fn compare(ours: &[f32], reference: &[f32], channels: usize) -> Vec<ChannelStats> {
    let frames = ours.len().min(reference.len()) / channels;
    (0..channels)
        .map(|c| {
            let (mut max_abs, mut se, mut sr) = (0.0f32, 0.0f64, 0.0f64);
            for i in 0..frames {
                let a = ours[i * channels + c];
                let b = reference[i * channels + c];
                max_abs = max_abs.max((a - b).abs());
                se += ((a - b) as f64).powi(2);
                sr += (b as f64).powi(2);
            }
            ChannelStats {
                max_abs_err: max_abs,
                rms_err: (se / frames as f64).sqrt(),
                rms_ref: (sr / frames as f64).sqrt(),
            }
        })
        .collect()
}

/// Where channel 0 of `ours` lines up best with the reference, for the
/// diagnostic line when things disagree: `(lag in samples, gain ratio)`.
fn best_alignment(ours: &[f32], reference: &[f32], channels: usize) -> (i64, f64) {
    let frames = (ours.len().min(reference.len()) / channels) as i64;
    let take = |v: &[f32], i: i64| if i >= 0 && i < frames { v[i as usize * channels] as f64 } else { 0.0 };
    let mut best = (0i64, f64::MIN);
    for lag in (-2048..=2048).step_by(1) {
        let mut acc = 0.0;
        for i in (0..frames).step_by(3) {
            acc += take(ours, i) * take(reference, i + lag);
        }
        if acc > best.1 {
            best = (lag, acc);
        }
    }
    let energy = |v: &[f32]| (0..frames).map(|i| take(v, i).powi(2)).sum::<f64>().sqrt();
    (best.0, energy(reference) / energy(ours).max(1e-30))
}

fn check(name: &str, graph: &str, bitrate: &str, expect_channels: usize, expect_rate: u32, tolerance: f64) {
    if !ffmpeg_available() {
        eprintln!("{name}: ffmpeg not on PATH — skipping (optional cross-check)");
        return;
    }
    let (dts, reference) = make_vector(name, graph, bitrate);
    let (ours, channels, rate) = decode_all(&dts, expect_channels as u8);
    assert_eq!(channels, expect_channels, "{name}: channel count");
    assert_eq!(rate, expect_rate, "{name}: sample rate");
    eprintln!(
        "{name}: {} bytes of DTS; ours {} frames, libavcodec {} frames",
        dts.len(),
        ours.len() / channels,
        reference.len() / channels
    );
    let stats = compare(&ours, &reference, channels);
    let mut worst = 0.0f64;
    for (c, s) in stats.iter().enumerate() {
        let rel = s.rms_err / s.rms_ref.max(1e-9);
        worst = worst.max(rel);
        eprintln!(
            "{name}: ch{c}: ref rms {:.4}  err rms {:.2e}  rel {:.2e}  max |err| {:.2e}",
            s.rms_ref, s.rms_err, rel, s.max_abs_err
        );
    }
    if worst > tolerance {
        let (lag, gain) = best_alignment(&ours, &reference, channels);
        eprintln!("{name}: best alignment of ch0: lag {lag} samples, reference/ours gain {gain:.4}");
    }
    assert!(
        worst <= tolerance,
        "{name}: worst per-channel relative RMS error {worst:.3e} exceeds {tolerance:.1e}"
    );
    // The stream length must agree to within one core frame (512 samples).
    let diff = (ours.len() as i64 - reference.len() as i64).abs() / channels as i64;
    assert!(diff <= 512, "{name}: length differs by {diff} samples");
}

#[test]
fn five_one_48k_matches_libavcodec() {
    check(
        "dts_5_1_48k",
        "sine=frequency=440:sample_rate=48000:duration=3[a];\
         sine=frequency=660:sample_rate=48000:duration=3[b];\
         sine=frequency=880:sample_rate=48000:duration=3[c];\
         sine=frequency=60:sample_rate=48000:duration=3[d];\
         anoisesrc=color=pink:sample_rate=48000:duration=3:amplitude=0.3:seed=1[e];\
         anoisesrc=color=brown:sample_rate=48000:duration=3:amplitude=0.3:seed=2[f];\
         [a][b][c][d][e][f]join=inputs=6:channel_layout=5.1(side):map=0.0-FL|1.0-FR|2.0-FC|3.0-LFE|4.0-SL|5.0-SR,volume=0.5",
        "1536k",
        6,
        48_000,
        1e-3,
    );
}

#[test]
fn stereo_48k_matches_libavcodec() {
    check(
        "dts_stereo_48k",
        "sine=frequency=1000:sample_rate=48000:duration=2[a];\
         anoisesrc=color=white:sample_rate=48000:duration=2:amplitude=0.2:seed=3[b];\
         [a][b]join=inputs=2:channel_layout=stereo",
        "768k",
        2,
        48_000,
        1e-3,
    );
}

#[test]
fn stereo_44k1_low_rate_matches_libavcodec() {
    // The lowest rate ffmpeg's encoder accepts for stereo at 44.1 kHz makes
    // it reach for the coarse quantisers and block codes; 44.1 kHz is the
    // other common core rate.
    check(
        "dts_stereo_44k1",
        "sine=frequency=1000:sample_rate=44100:duration=2[a];\
         anoisesrc=color=white:sample_rate=44100:duration=2:amplitude=0.2:seed=3[b];\
         [a][b]join=inputs=2:channel_layout=stereo",
        "256k",
        2,
        44_100,
        1e-3,
    );
}

/// Diagnostic, not a gate: point `RIVET_DTS_SAMPLE` at a raw core stream (16-bit
/// big-endian framing; `ffmpeg -i x.mkv -map 0:a -c copy -bsf:a dca_core -f dts
/// x.dts`) and this reports what the decoder makes of it. Real DTS tracks from
/// commercial encoders use ADPCM prediction in most frames, which the decoder
/// refuses by name (see the module docs); such frames are skipped here (their
/// PCM stands in as silence) so the frames the decoder *does* take can still
/// be compared with libavcodec one by one. A frame counts as matching when its
/// worst sample is within 1e-3 of the reference; the frame right after a skip
/// is not counted either way, as its filter-bank history is missing.
#[test]
fn real_sample_if_pointed_at_one() {
    let Some(path) = std::env::var_os("RIVET_DTS_SAMPLE") else {
        eprintln!("real_sample: RIVET_DTS_SAMPLE not set — skipping");
        return;
    };
    let dts = std::fs::read(&path).expect("read RIVET_DTS_SAMPLE");
    let mut dec = create_decoder("dts", None, 48_000, 6).unwrap();
    let (mut off, mut pcm, mut channels) = (0usize, Vec::<f32>::new(), 0usize);
    // Per frame: `Some(sample offset)` for a decoded frame, `None` for a
    // refused one.
    let mut frames: Vec<Option<usize>> = Vec::new();
    let mut refusals: std::collections::BTreeMap<String, usize> = Default::default();
    let mut first_error = None;
    while off + 8 <= dts.len() {
        let len = core_frame_len(&dts[off..]);
        let start = pcm.len();
        match dec.decode(&dts[off..off + len], 0) {
            Ok(out) => {
                for f in out {
                    channels = f.channels as usize;
                    pcm.extend_from_slice(&f.samples);
                }
                frames.push(Some(start));
            }
            Err(codec::audio::AudioError::Unsupported(reason)) => {
                // Keep the reason without the per-frame count so alike
                // refusals group.
                let key = reason.split(" (PMODE").next().unwrap_or(&reason).to_string();
                *refusals.entry(key).or_default() += 1;
                let per_frame = channels.max(1) * 512;
                pcm.extend(std::iter::repeat_n(0.0f32, per_frame));
                frames.push(None);
            }
            Err(e) => {
                first_error = Some((frames.len(), e.to_string()));
                break;
            }
        }
        off += len;
    }
    let decoded = frames.iter().filter(|f| f.is_some()).count();
    eprintln!(
        "real_sample: {} bytes, {} frames: {decoded} decoded, {} refused by name",
        dts.len(),
        frames.len(),
        frames.len() - decoded
    );
    for (reason, n) in &refusals {
        eprintln!("real_sample:   {n} × {reason}");
    }
    if let Some((at, e)) = first_error {
        eprintln!("real_sample: parse error at frame {at}: {e}");
    }
    if decoded > 0 && ffmpeg_available() {
        let dir = scratch_dir();
        let raw = dir.join("real_sample.f32");
        run_ffmpeg(&["-i", path.to_str().unwrap(), "-c:a", "pcm_f32le", "-f", "f32le", raw.to_str().unwrap()]);
        let reference: Vec<f32> = std::fs::read(&raw)
            .unwrap()
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        let _ = std::fs::remove_file(&raw);
        let _ = std::fs::remove_dir(&dir);
        let per_frame = channels * 512;
        let (mut counted, mut matching, mut worst) = (0usize, 0usize, 0.0f32);
        for (i, f) in frames.iter().enumerate() {
            let Some(start) = f else { continue };
            if i > 0 && frames[i - 1].is_none() {
                continue;
            }
            if start + per_frame > reference.len() {
                break;
            }
            let err = pcm[*start..start + per_frame]
                .iter()
                .zip(&reference[*start..start + per_frame])
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            counted += 1;
            matching += usize::from(err < 1e-3);
            worst = worst.max(err);
        }
        eprintln!(
            "real_sample: of {counted} decoded frames with intact history, {matching} match libavcodec within 1e-3 \
             (worst frame max |err| {worst:.2e}; VQ-coded subbands decode as silence here)"
        );
    }
}

#[test]
fn stereo_32k_lowest_rate_matches_libavcodec() {
    check(
        "dts_stereo_32k",
        "sine=frequency=300:sample_rate=32000:duration=2[a];\
         anoisesrc=color=pink:sample_rate=32000:duration=2:amplitude=0.3:seed=4[b];\
         [a][b]join=inputs=2:channel_layout=stereo",
        "192k",
        2,
        32_000,
        1e-3,
    );
}
