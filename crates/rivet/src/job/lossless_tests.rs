//! FLAC / ALAC through the job engine: audio-only outputs end to end, the
//! settings' checks, and the HLS rendition's codec string. Every output is
//! decoded back and compared with the source PCM sample for sample.

use std::sync::Arc;

use codec::audio::decode::{AlacDecoder, FlacDecoder};
use codec::audio::encode::flac::{FlacEncoderConfig, FlacLevel};
use codec::audio::encode::{AlacEncoder, FlacEncoder};

use super::audio::{AudioOutput, AudioRequest, build_audio_rendition, prepare_audio};
use crate::progress::NullSink;
use crate::settings::TranscodeSettings;
use crate::spec::AudioCodecPolicy;
use crate::{JobOutput, RungArtifact};

fn signal(frames: usize, channels: usize, bits: u32) -> Vec<i32> {
    let full = ((1i64 << (bits - 1)) - 1) as f64;
    (0..frames)
        .flat_map(|i| {
            let t = i as f64 / 48_000.0;
            (0..channels).map(move |c| {
                let v = (t * 2.0 * std::f64::consts::PI * (440.0 + 110.0 * c as f64)).sin() * 0.5
                    + ((i * 7919 + c * 104_729) % 1000) as f64 / 1000.0 * 0.01;
                (v * full).round() as i32
            })
        })
        .collect()
}

fn native_flac(pcm: &[i32], channels: u8, bits: u8) -> Vec<u8> {
    let mut enc = FlacEncoder::new(FlacEncoderConfig {
        sample_rate: 48_000,
        channels,
        bits_per_sample: bits,
        level: FlacLevel::Default,
    })
    .unwrap();
    let mut frames = enc.encode_int(pcm);
    frames.extend(enc.finish());
    container::mux::write_native_flac(&enc.metadata_blocks(), &frames).unwrap()
}

fn alac_m4a(pcm: &[i32], channels: u8, bits: u8) -> Vec<u8> {
    let mut enc = AlacEncoder::new(48_000, channels, bits).unwrap();
    let mut frames = enc.encode_int(pcm);
    frames.extend(enc.finish());
    let info = container::AudioInfo::alac(48_000, u16::from(channels), enc.cookie().to_bytes().to_vec());
    container::mux::write_audio_mp4(&info, &frames, Default::default()).unwrap()
}

fn audio_track(file: &[u8]) -> container::demux::AudioTrack {
    container::streaming::demux_audio(bytes::Bytes::copy_from_slice(file)).unwrap().expect("an audio track").track
}

/// Decode a file's audio track to integers: (codec, samples, bit depth).
fn decode(file: &[u8]) -> (String, Vec<i32>, u32) {
    let t = audio_track(file);
    let mut out = Vec::new();
    let bits = match t.codec.as_str() {
        "flac" => {
            let mut d = FlacDecoder::new(Some(&t.codec_private), t.sample_rate, t.channels as u8).unwrap();
            let mut bits = 0;
            for p in &t.samples {
                let (s, _, b) = d.decode_int(p).unwrap();
                out.extend(s);
                bits = b;
            }
            assert_ne!(d.md5_matches(), Some(false), "the output's STREAMINFO MD5 matches its audio");
            bits
        }
        "alac" => {
            let mut d = AlacDecoder::new(Some(&t.codec_private)).unwrap();
            for p in &t.samples {
                out.extend(d.decode_int(p).unwrap());
            }
            u32::from(d.config().bit_depth)
        }
        other => panic!("unexpected output codec {other}"),
    };
    (t.codec.clone(), out, bits)
}

fn run(input: &[u8], line: &str) -> JobOutput {
    let settings = TranscodeSettings::parse_kv_line(line).unwrap();
    let spec = settings.into_spec(0, 0).unwrap();
    crate::run_job_blocking(input, &spec, None, Arc::new(NullSink)).unwrap()
}

fn file(out: &JobOutput) -> &[u8] {
    match &out.rungs[0].artifact {
        RungArtifact::File(b) => b,
        other => panic!("expected a file, got {other:?}"),
    }
}

#[test]
fn a_flac_source_to_native_flac_is_copied() {
    let pcm = signal(30_000, 2, 16);
    let out = run(&native_flac(&pcm, 2, 16), "mode=audio audio=flac");
    assert_eq!(out.audio_handling, "flac passthrough");
    assert_eq!(out.audio_codecs.as_deref(), Some("fLaC"));
    let bytes = file(&out);
    assert_eq!(&bytes[..4], b"fLaC");
    assert_eq!(decode(bytes), ("flac".into(), pcm, 16));
}

#[test]
fn flac_to_alac_in_m4a_is_bit_exact() {
    let pcm = signal(30_000, 6, 24);
    let out = run(&native_flac(&pcm, 6, 24), "mode=audio audio=alac");
    assert_eq!(out.audio_handling, "flac → alac (6ch, 24-bit)");
    let bytes = file(&out);
    assert_eq!(&bytes[4..12], b"ftypM4A ");
    assert_eq!(decode(bytes), ("alac".into(), pcm, 24));
}

#[test]
fn alac_to_flac_in_mp4_is_bit_exact() {
    let pcm = signal(20_000, 2, 16);
    let out = run(&alac_m4a(&pcm, 2, 16), "mode=audio audio=flac audio-container=mp4 flac-compression=best");
    assert_eq!(out.audio_handling, "alac → flac (2ch, 16-bit)");
    assert_eq!(decode(file(&out)), ("flac".into(), pcm, 16));
}

#[test]
fn a_requested_depth_re_encodes_with_rounding() {
    let pcm = signal(10_000, 2, 24);
    let out = run(&native_flac(&pcm, 2, 24), "mode=audio audio=flac audio-bit-depth=16");
    assert_eq!(out.audio_handling, "flac → flac (2ch, 16-bit)");
    let want: Vec<i32> = pcm.iter().map(|&s| (f64::from(s) / 256.0).round().clamp(-32_768.0, 32_767.0) as i32).collect();
    assert_eq!(decode(file(&out)), ("flac".into(), want, 16));
    // The source's own depth, named, is still a copy.
    let out = run(&native_flac(&pcm, 2, 24), "mode=audio audio=flac audio-bit-depth=24");
    assert_eq!(out.audio_handling, "flac passthrough");
}

#[test]
fn a_flac_input_with_no_video_becomes_its_audio_only_form() {
    // A single-file job of a file with no video: FLAC asked for, so a .flac.
    let pcm = signal(12_000, 2, 16);
    let settings = TranscodeSettings::parse_kv_line("audio=flac").unwrap();
    let probed = crate::probe_bytes(&native_flac(&pcm, 2, 16)).expect("a FLAC file probes");
    let spec = settings.into_spec_for(&probed).unwrap();
    assert_eq!(spec.file_extension(), "flac");
}

#[test]
fn lossless_settings_are_checked() {
    let err = |line: &str| {
        let e = TranscodeSettings::parse_kv_line(line).and_then(|s| s.into_spec(1280, 720)).unwrap_err();
        format!("{e:#}")
    };
    assert!(err("audio=flac audio-bitrate=128k").contains("lossless"), "{}", err("audio=flac audio-bitrate=128k"));
    assert!(err("audio=opus audio-bit-depth=24").contains("audio-bit-depth applies"));
    assert!(err("audio=alac flac-compression=best").contains("flac-compression applies"));
    assert!(err("mode=audio audio=alac audio-container=flac").contains("holds FLAC only"));
    assert!(err("mode=audio audio=flac audio-container=mp3").contains("cannot hold lossless"));
    assert!(err("audio-container=flac").contains("mode=audio"));
    assert!(err("audio-bit-depth=20").contains("source|16|24"));
    // Accepted: lossless audio beside video, in either output.
    for line in ["audio=flac", "audio=alac audio-bit-depth=24", "mode=hls audio=flac flac-compression=fast"] {
        TranscodeSettings::parse_kv_line(line).unwrap().into_spec(1280, 720).unwrap();
    }
    let ext = |line: &str| TranscodeSettings::parse_kv_line(line).unwrap().into_spec(0, 0).unwrap().file_extension();
    assert_eq!(ext("mode=audio audio=flac"), "flac");
    assert_eq!(ext("mode=audio audio=flac audio-container=mp4"), "m4a");
    assert_eq!(ext("mode=audio audio=alac"), "m4a");
    assert_eq!(ext("mode=audio"), "mp3");
}

#[test]
fn an_hls_rendition_names_the_lossless_codecs() {
    let dir = tempfile::tempdir().unwrap();
    let pcm = signal(20_000, 2, 16);
    let track = audio_track(&native_flac(&pcm, 2, 16));
    for (policy, want) in [(AudioCodecPolicy::Flac, "fLaC"), (AudioCodecPolicy::Alac, "alac")] {
        let req = AudioRequest { output: AudioOutput::Cmaf, ..AudioRequest::plain(policy) };
        let prepared = prepare_audio(Some(&track), None, &[], req).unwrap().unwrap();
        let root = dir.path().join(want);
        let variant = build_audio_rendition(&root, &prepared, 4.0, "audio", "Audio").unwrap().unwrap();
        assert_eq!(variant.codec_string, want);
        let init = std::fs::read(root.join("audio/init.mp4")).unwrap();
        let config: &[u8] = if want == "fLaC" { b"dfLa" } else { b"alac" };
        assert!(init.windows(4).any(|w| w == config), "{want} init segment carries its config box");
    }
}
