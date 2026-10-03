//! Every audio output rivet writes, end to end through the job engine: a
//! source is transcoded by `run_job_blocking` to each codec in each file it
//! goes in — a single-file MP4, a QuickTime movie, a WebM, an HLS package, an
//! audio-only `.m4a`, `.ogg` and `.mp3` — and the output is read back with
//! rivet's own demuxers and decoded with rivet's own decoders. No other
//! implementation is run, and every codec is the workspace's own.
//!
//! What is checked, per output: the codec the demuxer reports, the channel
//! count and rate, how many samples the file presents (its edit list, Ogg
//! granule positions or tag frame applied) against the source's, and each
//! channel's level and waveform SNR against the source as rivet decodes it
//! (the best alignment within a few samples). The figures are printed
//! (`--nocapture`).
//!
//! Sources: native FLAC files made here (stereo, one tone per channel, and
//! 5.1, one tone per speaker, at 48 kHz), and a synthetic H.264 clip with a
//! stereo AAC track for the outputs that carry video.

mod common;

use std::sync::Arc;

use bytes::Bytes;
use codec::audio::{AudioCodec, AudioEncoderConfig, AudioFrame, create_decoder, create_encoder};
use container::streaming::demux_audio;
use rivet::{RungArtifact, TranscodeSettings};

const RATE: u32 = 48_000;

/// A tone per channel, `seconds` long, at 0.25 of full scale.
fn tones(freqs: &[f64], seconds: f64) -> Vec<f32> {
    let n = (seconds * f64::from(RATE)) as usize;
    let ch = freqs.len();
    (0..n * ch)
        .map(|i| (0.25 * (std::f64::consts::TAU * freqs[i % ch] * (i / ch) as f64 / f64::from(RATE)).sin()) as f32)
        .collect()
}

/// `pcm` as a native FLAC file (rivet's own FLAC encoder), 16-bit.
fn native_flac(pcm: &[f32], channels: u8) -> Vec<u8> {
    let codec = AudioCodec::Flac { bits_per_sample: 16, level: Default::default() };
    let mut enc = create_encoder(AudioEncoderConfig::new(codec, RATE, channels, 0)).unwrap();
    let mut frames = Vec::new();
    for (i, c) in pcm.chunks(4096 * usize::from(channels)).enumerate() {
        let f = AudioFrame { samples: c.to_vec(), sample_rate: RATE, channels, pts: i as i64 };
        frames.extend(enc.encode(&f).unwrap().into_iter().map(|p| (p.data, p.duration as u32)));
    }
    frames.extend(enc.flush().unwrap().into_iter().map(|p| (p.data, p.duration as u32)));
    container::mux::write_native_flac(&enc.extra_data(), &frames).unwrap()
}

/// What a file presents: its audio track decoded by rivet, the edit applied.
struct Presented {
    codec: String,
    rate: u32,
    channels: usize,
    /// Interleaved samples, every channel.
    pcm: Vec<f32>,
}

impl Presented {
    fn len(&self) -> usize {
        self.pcm.len() / self.channels
    }

    fn channel(&self, c: usize) -> Vec<f32> {
        self.pcm.iter().skip(c).step_by(self.channels).copied().collect()
    }
}

fn presented(file: Bytes) -> Presented {
    let src = demux_audio(file).expect("rivet demuxes the file").expect("an audio track");
    let t = src.track;
    let private = if t.codec == "aac" { &t.asc } else { &t.codec_private };
    let extra = (!private.is_empty()).then_some(private.as_slice());
    let mut dec = create_decoder(&t.codec, extra, t.sample_rate, t.channels as u8).expect("a decoder");
    let (mut pcm, mut rate, mut channels) = (Vec::new(), 0u32, 0usize);
    for p in &t.samples {
        for f in dec.decode(p, 0).expect("every packet decodes") {
            (rate, channels) = (f.sample_rate, usize::from(f.channels));
            pcm.extend(f.samples);
        }
    }
    for f in dec.flush().expect("flush") {
        pcm.extend(f.samples);
    }
    // The edit the file states: an MP4 edit list, Ogg granule positions, the
    // MP3 tag frame, Matroska's `CodecDelay` and `DiscardPadding`.
    if let Some(e) = src.edit {
        assert_eq!(e.delay, 0, "the audio starts with the file");
        let at = |ticks: u64| (u128::from(ticks) * u128::from(rate)).div_ceil(u128::from(t.timescale)) as usize * channels;
        if let Some(end) = e.media_end {
            pcm.truncate(at(end).min(pcm.len()));
        }
        pcm.drain(..at(e.media_start).min(pcm.len()));
    }
    Presented { codec: t.codec, rate, channels, pcm }
}

fn rms(x: &[f32]) -> f64 {
    (x.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>() / x.len().max(1) as f64).sqrt()
}

/// SNR of `got` against `want` at the best lag within ±4 samples, over the
/// middle (a tenth of a second off either end).
fn snr(want: &[f32], got: &[f32]) -> f64 {
    let n = want.len().min(got.len());
    let margin = (RATE / 10) as usize;
    (-4i64..=4)
        .map(|lag| {
            let (mut s, mut e) = (0.0f64, 0.0f64);
            for i in margin..n.saturating_sub(margin) {
                let j = i as i64 + lag;
                if j < 0 || j as usize >= got.len() {
                    continue;
                }
                let (a, b) = (f64::from(want[i]), f64::from(got[j as usize]));
                s += a * a;
                e += (a - b) * (a - b);
            }
            10.0 * (s / e.max(1e-30)).log10()
        })
        .fold(f64::NEG_INFINITY, f64::max)
}

/// The output against the source, channel for channel: the length the file
/// presents exactly the source's, the level within 1.5 dB (a 50 Hz LFE tone
/// through a low-passed LFE channel loses about 1), the SNR above `floor`.
fn compare(name: &str, source: &Presented, out: &Presented, floor: f64) {
    assert_eq!((out.rate, out.channels), (source.rate, source.channels), "{name}: rate and channels");
    assert_eq!(out.len(), source.len(), "{name}: the samples presented");
    let mut worst = f64::INFINITY;
    for c in 0..source.channels {
        let (want, got) = (source.channel(c), out.channel(c));
        let level = 20.0 * (rms(&got[..want.len()]) / rms(&want)).log10();
        let s = snr(&want, &got);
        assert!(level.abs() < 1.5, "{name}: channel {c} is {level:+.2} dB off the source");
        assert!(s > floor, "{name}: channel {c} at {s:.1} dB SNR (floor {floor})");
        worst = worst.min(s);
    }
    eprintln!(
        "{name}: {} ch at {} Hz, {} samples (source {}), worst channel {worst:.1} dB",
        out.channels,
        out.rate,
        out.len(),
        source.len()
    );
}

/// The one file a single-file or audio-only job made.
fn run(source: &[u8], settings: &str, w: u32, h: u32) -> (Vec<u8>, rivet::JobOutput) {
    let spec = TranscodeSettings::parse_kv_line(settings).unwrap().into_spec(w, h).unwrap_or_else(|e| panic!("{settings}: {e:#}"));
    let out = rivet::run_job_blocking(source, &spec, None, Arc::new(rivet::fn_sink(|_| {})))
        .unwrap_or_else(|e| panic!("{settings}: {e:#}"));
    let bytes = match &out.rungs[0].artifact {
        RungArtifact::File(b) => b.clone(),
        other => panic!("{settings}: a single file, not {other:?}"),
    };
    (bytes, out)
}

/// Audio-only outputs of a stereo source: every codec in the files it goes
/// in, the length exact (edit lists, granule positions, the MP3 tag frame).
#[test]
fn audio_only_outputs_of_a_stereo_source() {
    let src = native_flac(&tones(&[440.0, 660.0], 1.5), 2);
    let source = presented(Bytes::from(src.clone()));
    for (settings, codec, ext, floor) in [
        ("mode=audio audio=opus", "opus", "opus", 15.0),
        ("mode=audio audio=vorbis", "vorbis", "ogg", 12.0),
        ("mode=audio audio=vorbis audio-quality=9", "vorbis", "ogg", 15.0),
        ("mode=audio audio=opus audio-container=mp4", "opus", "m4a", 15.0),
        ("mode=audio audio=mp3", "mp3", "mp3", 25.0),
        ("mode=audio audio=mp3 audio-container=mp4", "mp3", "m4a", 25.0),
        ("mode=audio audio=aac", "aac", "m4a", 25.0),
        ("mode=audio audio=he-aac", "aac", "m4a", 15.0),
        ("mode=audio audio=ac3", "ac3", "m4a", 25.0),
        ("mode=audio audio=eac3", "eac3", "m4a", 25.0),
        ("mode=audio audio=dts", "dts", "m4a", 25.0),
    ] {
        let (file, out) = run(&src, settings, 0, 0);
        assert_eq!(rivet::single_file_extension(&file), ext, "{settings}");
        let got = presented(Bytes::from(file));
        assert_eq!(got.codec, codec, "{settings}");
        eprintln!("{settings}: {}", out.audio_handling);
        compare(settings, &source, &got, floor);
    }
}

/// HE-AAC v2 codes a stereo image parametrically: the waveform of each side
/// is not kept, but the length, the levels and which tone is on which side
/// are.
#[test]
fn he_aac_v2_keeps_the_stereo_image() {
    let src = native_flac(&tones(&[440.0, 3000.0], 1.5), 2);
    let source = presented(Bytes::from(src.clone()));
    let (file, out) = run(&src, "mode=audio audio=he-aacv2", 0, 0);
    assert_eq!(out.audio_codecs.as_deref(), Some("mp4a.40.29"));
    let got = presented(Bytes::from(file));
    assert_eq!((got.codec.as_str(), got.channels, got.rate, got.len()), ("aac", 2, RATE, source.len()));
    // Tones in different parametric stereo bands, so their sides can be told
    // apart.
    for (c, (own, other)) in [(440.0, 3000.0), (3000.0, 440.0)].into_iter().enumerate() {
        let ch = got.channel(c);
        let (a, b) = (goertzel(&ch, own), goertzel(&ch, other));
        eprintln!("he-aacv2 channel {c}: {own} Hz at {a:.3}, {other} Hz at {b:.3}");
        assert!((a / 0.25 - 1.0).abs() < 0.25, "channel {c}: its own tone at {a:.3}");
        assert!(b < a / 4.0, "channel {c}: the other side's tone at {b:.3}");
    }
}

/// The amplitude of `freq` in `x` (48 kHz).
fn goertzel(x: &[f32], freq: f64) -> f64 {
    let w = std::f64::consts::TAU * freq / f64::from(RATE);
    let (mut s1, mut s2) = (0.0f64, 0.0f64);
    for &v in x {
        let s = f64::from(v) + 2.0 * w.cos() * s1 - s2;
        s2 = s1;
        s1 = s;
    }
    let p = s1 * s1 + s2 * s2 - 2.0 * w.cos() * s1 * s2;
    2.0 * p.sqrt() / x.len() as f64
}

/// 5.1 through every codec that carries it: each tone stays on its own
/// speaker (AC-3, E-AC-3 and DTS name the surrounds side ones, in the same
/// slots).
#[test]
fn five_one_outputs_keep_every_speaker() {
    let src = native_flac(&tones(&[400.0, 600.0, 800.0, 50.0, 1000.0, 1200.0], 1.0), 6);
    let source = presented(Bytes::from(src.clone()));
    for (settings, codec, floor) in [
        ("mode=audio audio=opus", "opus", 10.0),
        ("mode=audio audio=vorbis", "vorbis", 10.0),
        ("mode=audio audio=aac", "aac", 20.0),
        ("mode=audio audio=he-aac audio-bitrate=128k", "aac", 10.0),
        ("mode=audio audio=ac3", "ac3", 20.0),
        ("mode=audio audio=eac3", "eac3", 20.0),
        ("mode=audio audio=dts", "dts", 20.0),
    ] {
        let (file, out) = run(&src, settings, 0, 0);
        let got = presented(Bytes::from(file));
        assert_eq!(got.codec, codec, "{settings}");
        eprintln!("{settings}: {}", out.audio_handling);
        compare(settings, &source, &got, floor);
    }
}

/// The outputs with video: the clip's AAC re-encoded into an MP4, a
/// QuickTime movie, a WebM and an HLS package, against the clip's own audio
/// as rivet decodes it.
#[test]
fn outputs_with_video_carry_each_codec() {
    let src = common::synth::clip(128, 96, 24, 1.5, 0, 0, true);
    let source = presented(Bytes::from(src.clone()));
    for (settings, codec, floor) in [
        ("codec=mpeg4 audio=he-aac", "aac", 15.0),
        ("codec=mpeg4 audio=ac3", "ac3", 25.0),
        ("codec=mpeg4 audio=eac3", "eac3", 25.0),
        ("codec=mpeg4 audio=dts", "dts", 25.0),
        ("codec=mpeg4 audio=mp3", "mp3", 25.0),
        ("codec=mpeg4 audio=opus", "opus", 15.0),
        ("codec=mpeg4 container=mov audio=ac3", "ac3", 25.0),
        // WebM: `CodecDelay` and the last block's `DiscardPadding`.
        ("codec=vp9 container=webm audio=vorbis", "vorbis", 12.0),
        ("codec=vp9 container=webm audio=opus", "opus", 15.0),
    ] {
        let (file, out) = run(&src, settings, 128, 96);
        let got = presented(Bytes::from(file));
        assert_eq!(got.codec, codec, "{settings}");
        eprintln!("{settings}: {}", out.audio_handling);
        compare(settings, &source, &got, floor);
    }
}

/// A WebM's audio trim (`CodecDelay`, the last block's `DiscardPadding`)
/// survives a WebM-to-WebM passthrough, and MKVToolNix — a black box here —
/// reads the file, keeps the trim when it remuxes it, and its remux presents
/// the same samples to rivet. The MKVToolNix half SKIPs without `mkvmerge`
/// and `mkvinfo` (`MKVMERGE` / `MKVINFO` name them) unless
/// `RIVET_REQUIRE_MKVTOOLNIX` is set, as CI sets it.
#[test]
fn webm_audio_trim_survives_passthrough_and_mkvtoolnix() {
    let src = common::synth::clip(128, 96, 24, 1.5, 0, 0, true);
    let source = presented(Bytes::from(src.clone()));
    let mkvtoolnix = mkvtoolnix();
    for (settings, codec) in
        [("codec=vp9 container=webm audio=opus", "opus"), ("codec=vp9 container=webm audio=vorbis", "vorbis")]
    {
        let (first, _) = run(&src, settings, 128, 96);
        let (second, out) = run(&first, settings, 128, 96);
        assert_eq!(out.audio_handling, format!("{codec} passthrough"), "{settings}");
        let a = presented(Bytes::from(first.clone()));
        let b = presented(Bytes::from(second));
        assert_eq!((a.len(), b.len()), (source.len(), source.len()), "{settings}: the samples presented");
        assert_eq!(a.pcm, b.pcm, "{settings}: the same audio, sample for sample");
        let Some((mkvmerge, mkvinfo)) = &mkvtoolnix else { continue };
        let dir = tempfile::tempdir().unwrap();
        let (file, remux) = (dir.path().join(format!("{codec}.webm")), dir.path().join(format!("{codec}-remux.webm")));
        std::fs::write(&file, &first).unwrap();
        let info = std::process::Command::new(mkvinfo).arg("-v").arg(&file).output().expect("mkvinfo runs");
        let text = String::from_utf8_lossy(&info.stdout);
        assert!(info.status.success(), "{settings}: mkvinfo: {text}{}", String::from_utf8_lossy(&info.stderr));
        assert!(text.contains("Discard padding"), "{settings}: mkvinfo sees no DiscardPadding:
{text}");
        if codec == "opus" {
            assert!(text.contains("Codec-inherent delay"), "{settings}: mkvinfo sees no CodecDelay:
{text}");
        }
        // Exit status 0: no warning either (1 is "warnings").
        let merge = std::process::Command::new(mkvmerge).arg("-o").arg(&remux).arg(&file).output().expect("mkvmerge runs");
        assert_eq!(merge.status.code(), Some(0), "{settings}: mkvmerge: {}", String::from_utf8_lossy(&merge.stdout));
        let c = presented(Bytes::from(std::fs::read(&remux).unwrap()));
        assert_eq!(c.len(), source.len(), "{settings}: mkvmerge's remux presents the source's samples");
        assert_eq!(c.pcm, a.pcm, "{settings}: mkvmerge's remux decodes to the same audio");
        eprintln!("{settings}: mkvinfo and mkvmerge take it; the remux presents {} samples", c.len());
    }
}

/// `mkvmerge` and `mkvinfo`, when both run; a panic instead of `None` under
/// `RIVET_REQUIRE_MKVTOOLNIX`.
fn mkvtoolnix() -> Option<(std::path::PathBuf, std::path::PathBuf)> {
    let find = |name: &str| {
        let path = std::env::var_os(name.to_ascii_uppercase()).map_or_else(|| name.into(), std::path::PathBuf::from);
        let runs = std::process::Command::new(&path).arg("--version").output().is_ok_and(|o| o.status.success());
        runs.then_some(path)
    };
    let found = find("mkvmerge").zip(find("mkvinfo"));
    if found.is_none() {
        assert!(
            std::env::var_os("RIVET_REQUIRE_MKVTOOLNIX").is_none(),
            "RIVET_REQUIRE_MKVTOOLNIX is set, and mkvmerge / mkvinfo do not run (set MKVMERGE / MKVINFO or put them on PATH)"
        );
        eprintln!("SKIP the MKVToolNix half: mkvmerge / mkvinfo not found");
    }
    found
}

/// HLS: the audio rendition, its init and media segments joined as a player
/// fetches them, for the codecs CMAF carries.
#[test]
fn hls_audio_renditions_carry_each_codec() {
    let src = common::synth::clip(128, 96, 24, 1.5, 0, 0, true);
    let source = presented(Bytes::from(src.clone()));
    for (settings, codec, codecs, floor) in [
        ("mode=hls segment-seconds=0.5 codec=vp9 audio=he-aac", "aac", "mp4a.40.5", 15.0),
        ("mode=hls segment-seconds=0.5 codec=vp9 audio=he-aacv2", "aac", "mp4a.40.29", -100.0),
        ("mode=hls segment-seconds=0.5 codec=vp9 audio=ac3", "ac3", "ac-3", 25.0),
        ("mode=hls segment-seconds=0.5 codec=vp9 audio=eac3", "eac3", "ec-3", 25.0),
        ("mode=hls segment-seconds=0.5 codec=vp9 audio=dts", "dts", "dtsc", 25.0),
        ("mode=hls segment-seconds=0.5 codec=vp9 audio=opus", "opus", "opus", 15.0),
    ] {
        let dir = tempfile::tempdir().unwrap();
        let spec = TranscodeSettings::parse_kv_line(settings).unwrap().into_spec(128, 96).unwrap();
        let out = rivet::run_job_blocking(&src, &spec, Some(dir.path()), Arc::new(rivet::fn_sink(|_| {}))).unwrap();
        assert_eq!(out.audio_codecs.as_deref(), Some(codecs), "{settings}");
        let master_path = out.master_playlist.expect("a master playlist");
        let master = std::fs::read_to_string(&master_path).unwrap();
        assert!(master.contains(codecs), "{settings}: {master}");
        let uri = master
            .lines()
            .filter(|l| l.starts_with("#EXT-X-MEDIA:TYPE=AUDIO"))
            .find_map(|l| l.split("URI=\"").nth(1)?.split('"').next())
            .expect("an audio rendition");
        let playlist = master_path.parent().unwrap().join(uri);
        let rendition = rendition_bytes(&playlist);
        let got = presented(rendition);
        assert_eq!(got.codec, codec, "{settings}");
        if floor > 0.0 {
            compare(settings, &source, &got, floor);
        } else {
            // Parametric stereo: the length and the level.
            assert_eq!((got.channels, got.len()), (2, source.len()), "{settings}");
            let level = 20.0 * (rms(&got.pcm) / rms(&source.pcm)).log10();
            assert!(level.abs() < 1.5, "{settings}: {level:+.2} dB");
            eprintln!("{settings}: {} samples, level {level:+.2} dB", got.len());
        }
    }
}

/// An HLS audio rendition joined as a player fetches it: the init segment
/// and every media segment its playlist lists, in order.
fn rendition_bytes(playlist: &std::path::Path) -> Bytes {
    let dir = playlist.parent().unwrap();
    let text = std::fs::read_to_string(playlist).unwrap();
    let init = text
        .lines()
        .find_map(|l| l.strip_prefix("#EXT-X-MAP:URI=\"")?.split('"').next())
        .expect("an EXT-X-MAP");
    let mut joined = std::fs::read(dir.join(init)).unwrap();
    for seg in text.lines().filter(|l| !l.starts_with('#') && !l.trim().is_empty()) {
        joined.extend(std::fs::read(dir.join(seg.trim())).unwrap());
    }
    Bytes::from(joined)
}

/// The new outputs passed through: a file rivet wrote in each codec, into
/// the same kind of file again, is copied, not re-encoded.
#[test]
fn rivets_own_outputs_pass_through() {
    let src = native_flac(&tones(&[440.0, 660.0], 1.0), 2);
    for (make, again, handling) in [
        ("mode=audio audio=vorbis", "mode=audio audio=vorbis", "vorbis passthrough"),
        ("mode=audio audio=opus", "mode=audio audio=opus", "opus passthrough"),
        ("mode=audio audio=ac3", "mode=audio audio=ac3", "ac3 passthrough"),
        ("mode=audio audio=eac3", "mode=audio audio=eac3", "eac3 passthrough"),
        ("mode=audio audio=dts", "mode=audio audio=dts", "dts passthrough"),
        ("mode=audio audio=he-aac", "mode=audio audio=he-aac", "aac passthrough"),
        ("mode=audio audio=mp3", "mode=audio audio=mp3", "mp3 passthrough"),
    ] {
        let (first, _) = run(&src, make, 0, 0);
        let (second, out) = run(&first, again, 0, 0);
        assert_eq!(out.audio_handling, handling, "{again}");
        let (a, b) = (presented(Bytes::from(first)), presented(Bytes::from(second)));
        assert_eq!(a.len(), b.len(), "{again}: the same presentation");
        assert_eq!(a.pcm, b.pcm, "{again}: the same audio, sample for sample");
    }
}
