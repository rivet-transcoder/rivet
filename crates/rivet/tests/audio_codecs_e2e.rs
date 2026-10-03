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
    native_flac_at(pcm, channels, RATE)
}

/// [`native_flac`] at `rate`.
fn native_flac_at(pcm: &[f32], channels: u8, rate: u32) -> Vec<u8> {
    let codec = AudioCodec::Flac { bits_per_sample: 16, level: Default::default() };
    let mut enc = create_encoder(AudioEncoderConfig::new(codec, rate, channels, 0)).unwrap();
    let mut frames = Vec::new();
    for (i, c) in pcm.chunks(4096 * usize::from(channels)).enumerate() {
        let f = AudioFrame { samples: c.to_vec(), sample_rate: rate, channels, pts: i as i64 };
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
    goertzel_at(x, freq, RATE)
}

/// [`goertzel`] for `x` at `rate`.
fn goertzel_at(x: &[f32], freq: f64, rate: u32) -> f64 {
    let w = std::f64::consts::TAU * freq / f64::from(rate);
    let (mut s1, mut s2) = (0.0f64, 0.0f64);
    for &v in x {
        let s = f64::from(v) + 2.0 * w.cos() * s1 - s2;
        s2 = s1;
        s1 = s;
    }
    let p = s1 * s1 + s2 * s2 - 2.0 * w.cos() * s1 * s2;
    2.0 * p.sqrt() / x.len() as f64
}

/// AAC-LC at the reduced rates, into an `.m4a`: a source at 8, 11.025, 12,
/// 16, 22.05 or 24 kHz is coded at its own rate (not resampled up to 22.05 or
/// 24 kHz), within 5 % of the bit rate asked for; the file
/// states the rate (`samplingFrequencyIndex`, the sample entry, `mdhd`) and
/// that no SBR follows, reads back at it, presents the source's length at
/// it, and keeps each tone on its side. MediaInfo, where it runs, reads the
/// rate and AAC-LC (not HE-AAC) from the file.
#[test]
fn aac_at_the_reduced_rates() {
    let mediainfo = std::env::var("MEDIAINFO").unwrap_or_else(|_| "mediainfo".into());
    let mediainfo = std::process::Command::new(&mediainfo)
        .arg("--Version")
        .output()
        .is_ok_and(|o| o.status.success())
        .then_some(mediainfo);
    assert!(
        mediainfo.is_some() || std::env::var_os("RIVET_REQUIRE_MEDIAINFO").is_none(),
        "RIVET_REQUIRE_MEDIAINFO is set and MediaInfo does not run"
    );
    for rate in [8_000u32, 11_025, 12_000, 16_000, 22_050, 24_000] {
        let (left, right) = (440.0, 1_000.0);
        let n = (1.5 * f64::from(rate)) as usize;
        let pcm: Vec<f32> = (0..2 * n)
            .map(|i| {
                let f = if i % 2 == 0 { left } else { right };
                (0.25 * (std::f64::consts::TAU * f * (i / 2) as f64 / f64::from(rate)).sin()) as f32
            })
            .collect();
        let src = native_flac_at(&pcm, 2, rate);
        let source = presented(Bytes::from(src.clone()));
        let (file, out) = run(&src, "mode=audio audio=aac", 0, 0);
        assert_eq!(rivet::single_file_extension(&file), "m4a");
        let coded = codec::audio::encode::aac::coding_rate(rate);
        let got = presented(Bytes::from(file.clone()));
        let asc = container::streaming::demux_audio(Bytes::from(file.clone())).unwrap().unwrap().track.asc;
        let parsed = container::aac_asc::parse_aac_asc(&asc).expect("the ASC");
        assert_eq!((parsed.aot, parsed.sample_rate, parsed.sbr_present), (2, coded, false), "{rate} Hz");
        assert_eq!(parsed.signaling, container::aac_asc::AscSignaling::NoExtension, "{rate} Hz: no SBR, said");
        assert_eq!((got.codec.as_str(), got.rate, got.channels), ("aac", coded, 2), "{rate} Hz");
        let want_len = (source.len() as u64 * u64::from(coded)).div_ceil(u64::from(rate)) as usize;
        assert!(got.len().abs_diff(want_len) <= 1, "{rate} Hz: {} samples presented, {want_len} expected", got.len());
        for (c, (own, other)) in [(left, right), (right, left)].into_iter().enumerate() {
            let ch = got.channel(c);
            let (a, b) = (goertzel_at(&ch, own, coded), goertzel_at(&ch, other, coded));
            assert!((a / 0.25 - 1.0).abs() < 0.1, "{rate} Hz channel {c}: its own tone at {a:.3}");
            assert!(b < 0.0025, "{rate} Hz channel {c}: the other side's tone at {b:.4}");
        }
        if coded == rate {
            compare(&format!("aac at {rate} Hz"), &source, &got, 20.0);
        }
        eprintln!("{rate} Hz source: {} → AAC-LC at {coded} Hz, {} samples", out.audio_handling, got.len());
        if let Some(bin) = &mediainfo {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join(format!("aac-{rate}.m4a"));
            std::fs::write(&path, &file).unwrap();
            let o = std::process::Command::new(bin).arg("--Inform=Audio;%Format%|%Format_AdditionalFeatures%|%SamplingRate%").arg(&path).output().unwrap();
            let text = String::from_utf8_lossy(&o.stdout).trim().to_string();
            assert_eq!(text, format!("AAC|LC|{coded}"), "{rate} Hz: MediaInfo reads {text}");
        }
    }
}

/// AAC-LC at the speech-band rates over ten seconds: the stream keeps the
/// source's rate and its bit rate lands within 5 % of the one asked for
/// (music-like tones with noise, stereo, two rates each).
#[test]
fn aac_speech_band_rates_hit_the_bit_rate() {
    for (rate, kbps) in [(8_000u32, [16u32, 32]), (11_025, [24, 48]), (12_000, [24, 48]), (16_000, [32, 64])] {
        let n = 10 * rate as usize;
        let mut seed = 0x1234_5678u32;
        let pcm: Vec<f32> = (0..2 * n)
            .map(|i| {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let noise = (seed >> 8) as f32 / (1u32 << 23) as f32 - 1.0;
                let t = (i / 2) as f64 / f64::from(rate);
                let f = if i % 2 == 0 { [220.0, 330.0, 1100.0] } else { [277.0, 415.0, 1500.0] };
                let tone: f64 = f.iter().enumerate().map(|(k, f)| 0.15 / (k + 1) as f64 * (std::f64::consts::TAU * f * t).sin()).sum();
                tone as f32 + 0.01 * noise
            })
            .collect();
        let src = native_flac_at(&pcm, 2, rate);
        for k in kbps {
            let (file, _) = run(&src, &format!("mode=audio audio=aac audio-bitrate={k}k"), 0, 0);
            let track = container::streaming::demux_audio(Bytes::from(file)).unwrap().unwrap().track;
            assert_eq!(track.sample_rate, rate, "{rate} Hz at {k} kb/s: coded at the source's rate");
            let bits: usize = track.samples.iter().map(|s| 8 * s.len()).sum();
            let seconds = (track.samples.len() * 1024) as f64 / f64::from(rate);
            let got = bits as f64 / seconds;
            let err = (got / f64::from(k * 1000) - 1.0) * 100.0;
            eprintln!("AAC-LC {rate} Hz stereo, {k} kb/s asked: {:.2} kb/s ({err:+.2} %)", got / 1000.0);
            assert!(err.abs() <= 5.0, "{rate} Hz at {k} kb/s: {got:.0} b/s ({err:+.2} %)");
        }
    }
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

/// Each of the eight speakers of a decoded 7.1 programme: its own tone at
/// 0 dB (within 1), every other speaker's below −60 dB. Measured over
/// 0.1–0.9 s, whole periods of every tone, clear of where the tones start
/// and stop (a transport stream states no edit, so its decode keeps the
/// encoder's delay and its last frame's padding).
fn assert_each_speaker_in_place(name: &str, got: &Presented, tones_hz: &[f64]) {
    assert_eq!(got.channels, tones_hz.len(), "{name}");
    for (c, &own) in tones_hz.iter().enumerate() {
        let ch = got.channel(c);
        let ch = &ch[4800..43_200];
        let a = goertzel(ch, own);
        let worst = tones_hz.iter().filter(|&&t| t != own).map(|&t| goertzel(ch, t)).fold(0.0, f64::max);
        let (own_db, worst_db) = (20.0 * (a / 0.25).log10(), 20.0 * (worst / 0.25).log10());
        eprintln!("{name} channel {c}: own {own} Hz {own_db:+.2} dB, worst other {worst_db:.1} dB");
        assert!(own_db.abs() < 1.0, "{name} channel {c}: its own tone at {own_db:+.2} dB");
        assert!(worst_db < -60.0, "{name} channel {c}: another speaker's tone at {worst_db:.1} dB");
    }
}

/// 7.1 to E-AC-3, as ETSI TS 102 366 §E.2.8.2 lays it out: independent
/// substream 0 a 5.1 downmix of the programme and a dependent substream
/// whose side surrounds replace the downmixed ones and whose back surrounds
/// add to them, one access unit to an MP4 sample, the `dec3` naming both
/// (`num_dep_sub` 1, `chan_loc` Lrs/Rrs). Read back, each of the eight
/// speakers carries its own tone at its level and the others' far below;
/// passed through again it is copied. MediaInfo, where it runs, reads eight
/// channels from the file.
#[test]
fn seven_one_e_ac3_keeps_every_speaker() {
    let tones_hz = [400.0, 600.0, 800.0, 50.0, 1000.0, 1200.0, 1400.0, 1600.0];
    let src = native_flac(&tones(&tones_hz, 1.0), 8);
    let source = presented(Bytes::from(src.clone()));
    let (file, out) = run(&src, "mode=audio audio=eac3", 0, 0);
    assert_eq!(out.audio_handling, "flac → eac3 (8ch)");
    let track = demux_audio(Bytes::from(file.clone())).unwrap().unwrap().track;
    assert_eq!((track.codec.as_str(), track.channels), ("eac3", 8));
    assert_eq!(track.codec_private.len(), 6, "dec3 with a dependent substream: {:02x?}", track.codec_private);
    let got = presented(Bytes::from(file.clone()));
    compare("7.1 e-ac-3", &source, &got, 20.0);
    assert_each_speaker_in_place("7.1 e-ac-3", &got, &tones_hz);
    let (again, out) = run(&file, "mode=audio audio=eac3", 0, 0);
    assert_eq!(out.audio_handling, "eac3 passthrough");
    assert_eq!(presented(Bytes::from(again)).pcm, got.pcm, "passed through sample for sample");
    let mediainfo = std::env::var("MEDIAINFO").unwrap_or_else(|_| "mediainfo".into());
    if std::process::Command::new(&mediainfo).arg("--Version").output().is_ok_and(|o| o.status.success()) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("eac3-7.1.m4a");
        std::fs::write(&path, &file).unwrap();
        let o = std::process::Command::new(&mediainfo)
            .arg("--Inform=Audio;%Format%|%Channel(s)%|%ChannelLayout%")
            .arg(&path)
            .output()
            .unwrap();
        let text = String::from_utf8_lossy(&o.stdout).trim().to_string();
        eprintln!("MediaInfo: {text}");
        assert!(text.starts_with("E-AC-3|8|"), "MediaInfo reads {text}");
    } else {
        assert!(std::env::var_os("RIVET_REQUIRE_MEDIAINFO").is_none(), "RIVET_REQUIRE_MEDIAINFO is set and MediaInfo does not run");
    }
}

/// 7.1 E-AC-3 in an MPEG transport stream (stream_type 0x87, PES
/// private_stream_1, one access unit per PES, beside an H.264 picture
/// track): the TS demuxer joins each dependent syncframe to its independent
/// one, so the track reads as eight channels with the same `dec3` as the
/// MP4 and the same access units byte for byte; decoded, each speaker is in
/// its place; passed through to an `.m4a`, it is copied.
#[test]
fn seven_one_e_ac3_through_a_transport_stream() {
    use common::synth;
    let tones_hz = [400.0, 600.0, 800.0, 50.0, 1000.0, 1200.0, 1400.0, 1600.0];
    let src = native_flac(&tones(&tones_hz, 1.0), 8);
    let (m4a, _) = run(&src, "mode=audio audio=eac3", 0, 0);
    let coded = demux_audio(Bytes::from(m4a.clone())).unwrap().unwrap().track;
    let units: Vec<(Vec<u8>, u32)> = coded.samples.iter().cloned().zip(coded.durations.iter().copied()).collect();
    let fps = 25;
    let pictures = (0..fps).map(|t| synth::test_pattern(128, 96, u64::from(t), h26x::ChromaFormat::Yuv420));
    let video = synth::encode_h264(&synth::H264::new(128, 96, fps), pictures);
    let audio = synth::TsAudio { stream_type: 0x87, stream_id: 0xBD, rate: RATE, units: &units };
    let ts = synth::ts_av(&video, 0x1B, fps, Some(audio));
    let track = demux_audio(Bytes::from(ts.clone())).unwrap().unwrap().track;
    assert_eq!((track.codec.as_str(), track.channels), ("eac3", 8));
    assert_eq!(track.codec_private, coded.codec_private, "the TS track's dec3 is the MP4's");
    assert_eq!(track.samples, coded.samples, "each access unit, independent and dependent syncframes, whole");
    let got = presented(Bytes::from(ts.clone()));
    assert_each_speaker_in_place("7.1 e-ac-3 in TS", &got, &tones_hz);
    let (copied, out) = run(&ts, "mode=audio audio=eac3", 0, 0);
    assert_eq!(out.audio_handling, "eac3 passthrough");
    let copied = demux_audio(Bytes::from(copied)).unwrap().unwrap().track;
    assert_eq!((copied.channels, &copied.codec_private, &copied.samples), (8, &coded.codec_private, &coded.samples));
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
