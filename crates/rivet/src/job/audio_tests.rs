//! The audio job end to end on real streams: surround layouts through Opus,
//! downmixes and their levels, MP3 into an MP4 and into a bare `.mp3`,
//! passthroughs that must keep their layout, and the HLS signalling.
//!
//! The 5.1 fixtures (`tests/data/audio/make_fixtures.sh`) carry one tone per
//! channel — FL 400 Hz, FR 600, FC 800, LFE 50, SL 1000, SR 1200, each at
//! 0.25 — so an output channel is identified by decoding it and measuring
//! which tones it holds. Tests that encode MP3 need LAME at run time: they
//! skip without it, unless `RIVET_REQUIRE_LAME` is set (CI sets it).

use std::sync::Arc;

use bytes::Bytes;
use container::demux::AudioTrack;
use container::streaming::demux_audio;

use super::audio::{AudioOutput, AudioRequest, PreparedAudio, audio_codec_string, build_audio_rendition, prepare_audio};
use crate::progress::NullSink;
use crate::spec::{AudioChannels, AudioCodecPolicy, OutputMode, OutputSpec, Rung};

const FL: f32 = 400.0;
const FR: f32 = 600.0;
const FC: f32 = 800.0;
const LFE: f32 = 50.0;
const SL: f32 = 1000.0;
const SR: f32 = 1200.0;
const TONES: [f32; 6] = [FL, FR, FC, LFE, SL, SR];
const LEVEL: f32 = 0.25;

fn fixture(name: &str) -> Bytes {
    let path = format!("{}/tests/data/audio/{name}", env!("CARGO_MANIFEST_DIR"));
    Bytes::from(std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e}")))
}

fn track(name: &str) -> AudioTrack {
    demux_audio(fixture(name)).expect("demux").expect("an audio track").track
}

/// Whether LAME can be loaded; panics instead when `RIVET_REQUIRE_LAME` is set.
fn lame() -> bool {
    match codec::audio::mp3_encoder_name() {
        Some(_) => true,
        None => {
            assert!(
                std::env::var_os("RIVET_REQUIRE_LAME").is_none(),
                "RIVET_REQUIRE_LAME is set but MP3 cannot be encoded (a build without `lame`, or no libmp3lame)"
            );
            eprintln!("skipping: no MP3 encoder (build with `lame` and install libmp3lame)");
            false
        }
    }
}

fn request(policy: AudioCodecPolicy, channels: AudioChannels, output: AudioOutput) -> AudioRequest<'static> {
    AudioRequest { policy, bitrate: None, filters: &[], channels, output }
}

/// The prepared track decoded back to interleaved PCM, with its channel count.
fn decode(a: &PreparedAudio) -> (Vec<f32>, usize) {
    let ch = a.info.channels as u8;
    let extra = (!a.info.codec_private.is_empty()).then_some(a.info.codec_private.as_slice());
    let mut dec = codec::audio::create_decoder(&a.info.codec, extra, a.info.sample_rate, ch).expect("decoder");
    let mut pcm = Vec::new();
    for (packet, _) in &a.samples {
        for f in dec.decode(packet, 0).expect("decode") {
            assert_eq!(usize::from(f.channels), usize::from(ch));
            pcm.extend_from_slice(&f.samples);
        }
    }
    for f in dec.flush().expect("flush") {
        pcm.extend_from_slice(&f.samples);
    }
    (pcm, usize::from(ch))
}

/// Amplitude of the `freq` tone in channel `c` of a steady stretch of `pcm`
/// (Goertzel over a whole number of 50 Hz periods, which every test tone
/// divides).
fn amplitude(pcm: &[f32], ch: usize, c: usize, freq: f32, rate: f32) -> f32 {
    let frames = pcm.len() / ch;
    // Skip the codecs' lead-in and tail; keep a whole number of 20 ms.
    let (start, len) = (frames / 4, ((frames / 2) as f32 / (rate / 50.0)).floor() as usize * (rate / 50.0) as usize);
    let w = 2.0 * std::f32::consts::PI * freq / rate;
    let (mut s1, mut s2) = (0.0f32, 0.0f32);
    for i in start..start + len {
        let s = pcm[i * ch + c] + 2.0 * w.cos() * s1 - s2;
        s2 = s1;
        s1 = s;
    }
    let power = s1 * s1 + s2 * s2 - 2.0 * w.cos() * s1 * s2;
    2.0 * power.max(0.0).sqrt() / len as f32
}

fn close(got: f32, want: f32, what: &str) {
    assert!((got - want).abs() <= want * 0.1 + 0.002, "{what}: {got:.4}, want {want:.4}");
}

#[test]
fn ac3_5_1_to_opus_keeps_every_channel_in_its_place() {
    let t = track("tones_51_ac3.mka");
    assert_eq!((t.codec.as_str(), t.channels), ("ac3", 6));
    let a = prepare_audio(Some(&t), None, &[], request(AudioCodecPolicy::ForceOpus, AudioChannels::Source, AudioOutput::Mp4))
        .unwrap()
        .unwrap();
    assert_eq!(a.handling, "ac3 → opus (6ch)");
    assert_eq!((a.info.codec.as_str(), a.info.channels), ("opus", 6));
    // dOps family 1, four streams (two coupled), the RFC 7845 5.1 mapping.
    assert_eq!(a.info.codec_private[10..], [1, 4, 2, 0, 4, 1, 2, 3, 5]);
    let (pcm, ch) = decode(&a);
    // 5.1(side) arrives as 5.1: the side pair in the surround slots.
    for (c, &tone) in TONES.iter().enumerate() {
        let own = amplitude(&pcm, ch, c, tone, 48_000.0);
        close(own, LEVEL, &format!("channel {c}'s own {tone} Hz"));
        for &other in TONES.iter().filter(|&&o| o != tone) {
            let leak = amplitude(&pcm, ch, c, other, 48_000.0);
            assert!(leak < 0.01, "channel {c} carries {other} Hz at {leak:.4}");
        }
    }
    assert_eq!(audio_codec_string(&a.info), "opus");
}

/// ITU-R BS.775, normalised: L = 0.414·FL + 0.293·FC + 0.293·SL, the LFE
/// dropped, nothing crossing sides.
#[test]
fn ac3_5_1_downmixes_to_stereo_at_bs775_levels() {
    let t = track("tones_51_ac3.mka");
    let a = prepare_audio(Some(&t), None, &[], request(AudioCodecPolicy::Auto, AudioChannels::Stereo, AudioOutput::Mp4))
        .unwrap()
        .unwrap();
    assert_eq!(a.handling, "ac3 → opus (6ch → 2ch)", "a layout change forces the transcode");
    assert_eq!(a.info.channels, 2);
    let (pcm, ch) = decode(&a);
    let front = 1.0 / (1.0 + std::f32::consts::SQRT_2);
    let centre = std::f32::consts::FRAC_1_SQRT_2 * front;
    let amp = |c, f| amplitude(&pcm, ch, c, f, 48_000.0);
    close(amp(0, FL), LEVEL * front, "FL in L");
    close(amp(1, FR), LEVEL * front, "FR in R");
    close(amp(0, FC), LEVEL * centre, "FC in L");
    close(amp(1, FC), LEVEL * centre, "FC in R");
    close(amp(0, SL), LEVEL * centre, "SL in L");
    close(amp(1, SR), LEVEL * centre, "SR in R");
    for (c, f, what) in [(0, FR, "FR in L"), (1, FL, "FL in R"), (0, SR, "SR in L"), (1, SL, "SL in R"), (0, LFE, "LFE in L"), (1, LFE, "LFE in R")] {
        assert!(amp(c, f) < 0.005, "{what}: {}", amp(c, f));
    }
}

#[test]
fn ac3_5_1_to_mono_folds_the_stereo_downmix() {
    let t = track("tones_51_ac3.mka");
    let a = prepare_audio(Some(&t), None, &[], request(AudioCodecPolicy::Auto, AudioChannels::Mono, AudioOutput::Mp4))
        .unwrap()
        .unwrap();
    assert_eq!(a.info.channels, 1);
    let (pcm, _) = decode(&a);
    // 0.707·(L + R) + C + 0.5·(Ls + Rs), over 3.414.
    let n = std::f32::consts::SQRT_2 + 2.0;
    close(amplitude(&pcm, 1, 0, FL, 48_000.0), LEVEL * std::f32::consts::FRAC_1_SQRT_2 / n, "FL");
    close(amplitude(&pcm, 1, 0, FC, 48_000.0), LEVEL / n, "FC");
    close(amplitude(&pcm, 1, 0, SR, 48_000.0), LEVEL * 0.5 / n, "SR");
}

#[test]
fn rivet_does_not_upmix() {
    let t = track("tones_51_ac3.mka");
    let err = prepare_audio(Some(&t), None, &[], request(AudioCodecPolicy::Auto, AudioChannels::Surround71, AudioOutput::Mp4))
        .err()
        .expect("7.1 from 5.1 is refused");
    assert!(err.to_string().contains("rivet does not upmix"), "{err:#}");
    // Asking for the layout the source already has changes nothing.
    let a = prepare_audio(Some(&t), None, &[], request(AudioCodecPolicy::Auto, AudioChannels::Surround51, AudioOutput::Mp4))
        .unwrap()
        .unwrap();
    assert_eq!(a.handling, "ac3 passthrough");
}

#[test]
fn aac_5_1_passes_through_with_its_layout_and_cannot_be_downmixed() {
    let t = track("tones_51_aac.m4a");
    let a = prepare_audio(Some(&t), None, &[], request(AudioCodecPolicy::Auto, AudioChannels::Source, AudioOutput::Mp4))
        .unwrap()
        .unwrap();
    assert_eq!((a.handling.as_str(), a.info.channels), ("aac passthrough", 6));
    let asc = container::aac_asc::parse_aac_asc(&a.info.asc_bytes).expect("the ASC");
    assert_eq!(asc.channel_configuration, 6, "5.1 signalled as Table 1.19's configuration 6");
    assert_eq!(audio_codec_string(&a.info), "mp4a.40.2");
    // Keeping the width is still a passthrough; changing it needs a decoder.
    let kept = prepare_audio(Some(&t), None, &[], request(AudioCodecPolicy::Auto, AudioChannels::Surround51, AudioOutput::Mp4));
    assert_eq!(kept.unwrap().unwrap().handling, "aac passthrough");
    let err = prepare_audio(Some(&t), None, &[], request(AudioCodecPolicy::Auto, AudioChannels::Stereo, AudioOutput::Mp4))
        .err()
        .expect("no AAC decoder, so no downmix");
    assert!(err.to_string().contains("has no decoder"), "{err:#}");
    // MP3 asked for, AAC not decodable: kept in an MP4 (as Opus is), refused
    // for a bare .mp3 which cannot hold it.
    let mp4 = prepare_audio(Some(&t), None, &[], request(AudioCodecPolicy::ForceMp3, AudioChannels::Source, AudioOutput::Mp4));
    assert_eq!(mp4.unwrap().unwrap().handling, "aac passthrough (mp3 requested; no aac decoder)");
    let err = prepare_audio(Some(&t), None, &[], request(AudioCodecPolicy::Auto, AudioChannels::Source, AudioOutput::Mp3File))
        .err()
        .expect("an .mp3 file cannot hold AAC");
    assert!(err.to_string().contains("an .mp3 file holds MP3"), "{err:#}");
}

/// An Opus 5.1 track (family 1) is decoded now, so it can be downmixed.
#[test]
fn opus_5_1_downmixes_to_stereo() {
    let frames = 48_000 / 2;
    let samples = (0..frames * 6)
        .map(|i| {
            let (t, c) = (i / 6, i % 6);
            LEVEL * (2.0 * std::f32::consts::PI * TONES[c] * t as f32 / 48_000.0).sin()
        })
        .collect();
    let mut enc = codec::audio::create_encoder(codec::audio::AudioEncoderConfig {
        codec: codec::audio::AudioCodec::Opus,
        sample_rate: 48_000,
        channels: 6,
        bitrate: 0,
    })
    .unwrap();
    let mut packets = enc.encode(&codec::audio::AudioFrame { samples, sample_rate: 48_000, channels: 6, pts: 0 }).unwrap();
    packets.extend(enc.flush().unwrap());
    let t = AudioTrack {
        codec: "opus".into(),
        samples: packets.iter().map(|p| p.data.clone()).collect(),
        sample_rate: 48_000,
        channels: 6,
        asc: Vec::new(),
        codec_private: enc.extra_data(),
        timescale: 48_000,
        durations: vec![960; packets.len()],
    };
    let a = prepare_audio(Some(&t), None, &[], request(AudioCodecPolicy::Auto, AudioChannels::Stereo, AudioOutput::Mp4))
        .unwrap()
        .unwrap();
    assert_eq!(a.handling, "opus → opus (6ch → 2ch)");
    // The source's pre-skip is hidden (from its OpusHead), not encoded.
    assert_eq!(a.edit.duration, Some(frames as u64 - u64::from(enc.pre_skip()) + (960 - frames as u64 % 960) % 960));
    let (pcm, ch) = decode(&a);
    let front = 1.0 / (1.0 + std::f32::consts::SQRT_2);
    close(amplitude(&pcm, ch, 0, FL, 48_000.0), LEVEL * front, "FL in L");
    close(amplitude(&pcm, ch, 1, FR, 48_000.0), LEVEL * front, "FR in R");
    assert!(amplitude(&pcm, ch, 0, FR, 48_000.0) < 0.005);
}

#[test]
fn ac3_5_1_to_mp3_is_a_stereo_downmix() {
    if !lame() {
        return;
    }
    let t = track("tones_51_ac3.mka");
    let a = prepare_audio(Some(&t), None, &[], request(AudioCodecPolicy::ForceMp3, AudioChannels::Source, AudioOutput::Mp4))
        .unwrap()
        .unwrap();
    assert_eq!(a.handling, "ac3 → mp3 (6ch → 2ch)");
    assert_eq!((a.info.codec.as_str(), a.info.sample_rate, a.info.channels, a.info.timescale), ("mp3", 48_000, 2, 48_000));
    assert!(a.encoder.as_deref().is_some_and(|e| e.starts_with("LAME")));
    assert_eq!(a.edit.media_time, 576 + 529, "LAME's delay and the decoder's, hidden by the edit");
    let coded = a.samples.len() as u64 * 1152;
    let presented = a.edit.duration.unwrap();
    assert!(coded >= presented + 1105 && coded < presented + 1105 + 2 * 1152, "{coded} coded, {presented} presented");
    assert!(a.samples.iter().all(|(f, d)| *d == 1152 && f[..2] == [0xFF, 0xFB] && f[2] >> 4 == 9), "128k MPEG-1 L3 frames");
    assert_eq!(audio_codec_string(&a.info), "mp3");
    let (pcm, ch) = decode(&a);
    let front = 1.0 / (1.0 + std::f32::consts::SQRT_2);
    close(amplitude(&pcm, ch, 0, FL, 48_000.0), LEVEL * front, "FL in L");
    close(amplitude(&pcm, ch, 1, FR, 48_000.0), LEVEL * front, "FR in R");
    assert!(amplitude(&pcm, ch, 0, FR, 48_000.0) < 0.005);
    // The MP4 muxer takes the track.
    container::mux::Av1Mp4Muxer::check_audio(&a.info).expect("MP3 in MP4");
}

/// An MP3 source goes into a single-file MP4 as it is under `auto` — and is
/// transcoded for HLS, whose CMAF has no MP3.
#[test]
fn an_mp3_source_passes_into_an_mp4_but_not_into_hls() {
    let bytes = std::fs::read(format!("{}/../codec/tests/data/mp3_tone_48k_mono.mp3", env!("CARGO_MANIFEST_DIR"))).unwrap();
    let src = demux_audio(Bytes::from(bytes)).unwrap().unwrap();
    assert!(!src.has_video);
    let mp4 = prepare_audio(Some(&src.track), src.edit, &[], request(AudioCodecPolicy::Auto, AudioChannels::Source, AudioOutput::Mp4))
        .unwrap()
        .unwrap();
    assert_eq!((mp4.handling.as_str(), mp4.info.codec.as_str()), ("mp3 passthrough", "mp3"));
    let hls = prepare_audio(Some(&src.track), src.edit, &[], request(AudioCodecPolicy::Auto, AudioChannels::Source, AudioOutput::Cmaf))
        .unwrap()
        .unwrap();
    assert_eq!(hls.handling, "mp3 → opus (1ch)");
    let file = prepare_audio(Some(&src.track), src.edit, &[], request(AudioCodecPolicy::Auto, AudioChannels::Source, AudioOutput::Mp3File))
        .unwrap()
        .unwrap();
    assert_eq!(file.handling, "mp3 passthrough");
}

/// `mode=audio` on a 5.1 AC-3 source: a stereo `.mp3` whose Info frame's
/// LAME extension says exactly how many samples are audio.
#[test]
fn an_audio_only_job_writes_a_gapless_mp3() {
    if !lame() {
        return;
    }
    let out = super::run_job_blocking(&fixture("tones_51_ac3.mka"), &OutputSpec::audio_only(), None, Arc::new(NullSink))
        .expect("the job");
    assert_eq!(out.audio_codecs.as_deref(), Some("mp3"));
    assert_eq!(out.audio_handling, "ac3 → mp3 (6ch → 2ch)");
    let [r] = &out.rungs[..] else { panic!("one output") };
    let super::RungArtifact::File(bytes) = &r.artifact else { panic!("a file") };
    assert_eq!(container::sniff_container(bytes), container::ContainerKind::Mp3);
    let (mp3, edit) = container::mp3::read_file(bytes).expect("the .mp3 reads back");
    assert_eq!((mp3.codec.as_str(), mp3.sample_rate, mp3.channels), ("mp3", 48_000, 2));
    // Every sample the AC-3 decodes to is presented, after the 1105-sample delay.
    let t = track("tones_51_ac3.mka");
    let mut dec = codec::audio::create_decoder("ac3", None, 48_000, 6).unwrap();
    let mut decoded = 0u64;
    for p in &t.samples {
        decoded += dec.decode(p, 0).unwrap().iter().map(|f| (f.samples.len() / 6) as u64).sum::<u64>();
    }
    let edit = edit.expect("the LAME tag's gapless info");
    assert_eq!((edit.media_start, edit.media_end), (1105, Some(1105 + decoded)));
    // That .mp3 through another audio-only job: a passthrough, which states
    // the same gapless information under the same encoder name.
    let again = super::run_job_blocking(bytes, &OutputSpec::audio_only(), None, Arc::new(NullSink)).expect("the job");
    assert_eq!(again.audio_handling, "mp3 passthrough");
    let super::RungArtifact::File(copy) = &again.rungs[0].artifact else { panic!("a file") };
    let (_, copy_edit) = container::mp3::read_file(copy).unwrap();
    assert_eq!(copy_edit, Some(edit));
}

/// A single-file job of an input with no video is its audio-only form.
#[test]
fn a_video_less_input_under_a_single_file_spec_writes_audio_only() {
    if !lame() {
        return;
    }
    let spec = OutputSpec::single_file(vec![Rung::new(640, 360)]);
    let out = super::run_job_blocking(&fixture("tones_51_ac3.mka"), &spec, None, Arc::new(NullSink)).expect("the job");
    assert_eq!(out.rungs.len(), 1);
    assert_eq!(out.rungs[0].label, "audio");
    // Opus has no audio-only form: said so, not quietly changed.
    let opus = OutputSpec::single_file(vec![Rung::new(640, 360)]).with_audio(AudioCodecPolicy::ForceOpus);
    let err = super::run_job_blocking(&fixture("tones_51_ac3.mka"), &opus, None, Arc::new(NullSink)).err().unwrap();
    assert!(format!("{err:#}").contains("cannot hold Opus"), "{err:#}");
}

/// The master playlist's audio group: CHANNELS from the prepared track (6
/// for Opus 5.1), and with a stereo fallback a second rendition beside it.
#[test]
fn hls_signals_surround_channels_and_the_stereo_fallback() {
    let t = track("tones_51_ac3.mka");
    let surround = prepare_audio(Some(&t), None, &[], request(AudioCodecPolicy::Auto, AudioChannels::Source, AudioOutput::Cmaf))
        .unwrap()
        .unwrap();
    assert_eq!(surround.handling, "ac3 passthrough", "AC-3 goes into CMAF as it is");
    let opus = prepare_audio(Some(&t), None, &[], request(AudioCodecPolicy::ForceOpus, AudioChannels::Source, AudioOutput::Cmaf))
        .unwrap()
        .unwrap();
    let stereo = prepare_audio(Some(&t), None, &[], request(AudioCodecPolicy::ForceOpus, AudioChannels::Stereo, AudioOutput::Cmaf))
        .unwrap()
        .unwrap();
    let dir = tempfile::tempdir().unwrap();
    let main = build_audio_rendition(dir.path(), &opus, 4.0, "audio", "Surround").unwrap().unwrap();
    let fallback = build_audio_rendition(dir.path(), &stereo, 4.0, "audio-stereo", "Stereo").unwrap().unwrap();
    let ac3 = build_audio_rendition(dir.path(), &surround, 4.0, "audio-ac3", "AC-3").unwrap().unwrap();
    assert_eq!((main.channels, main.codec_string.as_str()), (6, "opus"));
    assert_eq!((fallback.channels, fallback.codec_string.as_str()), (2, "opus"));
    assert_eq!((ac3.channels, ac3.codec_string.as_str()), (6, "ac-3"), "not mp4a.40.2, which it used to be called");
    let video = container::hls::VideoVariantSpec {
        width: 640,
        height: 360,
        frame_rate: 30.0,
        bandwidth_bps: 800_000,
        average_bandwidth_bps: 600_000,
        codec_string: "av01.0.04M.08".into(),
        supplemental_codecs: None,
        video_range: None,
        relative_dir: "video/360p".into(),
        manifest: main.manifest.clone(),
    };
    let paths = container::hls::write_hls_package(dir.path(), &[video], &[fallback, main], &[], 4).unwrap();
    assert_eq!(paths.audio_playlist_paths.len(), 2);
    let master = std::fs::read_to_string(paths.master_path).unwrap();
    assert!(master.contains(r#"NAME="Stereo",DEFAULT=YES,AUTOSELECT=YES,LANGUAGE="und",CHANNELS="2",URI="audio-stereo/audio.m3u8""#), "{master}");
    assert!(master.contains(r#"NAME="Surround",DEFAULT=NO,AUTOSELECT=YES,LANGUAGE="und",CHANNELS="6",URI="audio/audio.m3u8""#), "{master}");
    assert!(master.contains(r#"CODECS="av01.0.04M.08,opus""#), "{master}");
}

#[test]
fn codec_strings_follow_the_track() {
    let info = |codec: &str, asc: Vec<u8>| container::AudioInfo {
        codec: codec.into(),
        sample_rate: 48_000,
        channels: 2,
        timescale: 48_000,
        asc_bytes: asc,
        codec_private: Vec::new(),
    };
    // AAC-LC 48 kHz stereo; HE-AAC explicitly signalled (AOT 5, 24 kHz core).
    assert_eq!(audio_codec_string(&info("aac", vec![0x11, 0x90])), "mp4a.40.2");
    assert_eq!(audio_codec_string(&info("aac", vec![0x2B, 0x11, 0x88, 0x00])), "mp4a.40.5");
    assert_eq!(audio_codec_string(&info("ac3", Vec::new())), "ac-3");
    assert_eq!(audio_codec_string(&info("eac3", Vec::new())), "ec-3");
    assert_eq!(audio_codec_string(&info("opus", Vec::new())), "opus");
    assert_eq!(audio_codec_string(&info("mp3", Vec::new())), "mp3");
}

#[test]
fn an_audio_only_spec_is_its_own_output_mode() {
    let spec = OutputSpec::audio_only();
    assert_eq!(spec.mode, OutputMode::AudioOnly);
    spec.validate().expect("no rungs needed");
    assert_eq!(spec.audio_encode_codec(), codec::audio::AudioCodec::Mp3);
}

// ---- AAC output ----------------------------------------------------------
//
// rivet has no AAC decoder, so the AAC tests read their output back with
// ffmpeg / ffprobe — strangers to this codebase, used only as black-box
// decoders — and skip (saying so) on a host without them.

fn ffmpeg_available() -> bool {
    let ok = |tool: &str| {
        std::process::Command::new(tool)
            .arg("-version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    };
    let there = ok("ffmpeg") && ok("ffprobe");
    if !there {
        eprintln!("skipping: ffmpeg / ffprobe not on PATH (they read the AAC output back)");
    }
    there
}

/// ffprobe's view of the file's audio stream, as `key=value` pairs.
fn probe_audio(path: &std::path::Path) -> std::collections::HashMap<String, String> {
    let out = std::process::Command::new("ffprobe")
        .args(["-v", "error", "-select_streams", "a:0", "-show_entries"])
        .arg("stream=codec_name,profile,sample_rate,channels,channel_layout")
        .args(["-of", "default=noprint_wrappers=1"])
        .arg(path)
        .output()
        .expect("ffprobe");
    assert!(out.status.success(), "ffprobe: {}", String::from_utf8_lossy(&out.stderr));
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|l| l.split_once('=').map(|(k, v)| (k.to_string(), v.to_string())))
        .collect()
}

/// The file's audio decoded by ffmpeg to interleaved f32 (in ffmpeg's
/// native channel order, which is the pipeline's), refusing any decode
/// error.
fn ffmpeg_pcm(path: &std::path::Path) -> Vec<f32> {
    let out = std::process::Command::new("ffmpeg")
        .args(["-v", "error", "-xerror", "-i"])
        .arg(path)
        .args(["-map", "0:a:0", "-f", "f32le", "-"])
        .output()
        .expect("ffmpeg");
    // The stand-in video track is filler, and ffmpeg's probe tries to decode
    // it: its AV1 decoder's complaints are about that, not about the audio.
    let stderr = String::from_utf8_lossy(&out.stderr);
    let err: Vec<&str> = stderr.lines().filter(|l| !l.contains("dav1d") && !l.contains("av1")).collect();
    assert!(out.status.success() && err.is_empty(), "ffmpeg decode: {stderr}");
    out.stdout.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect()
}

/// A single-file MP4 around `a`: the crate's muxer with a stand-in AV1 track
/// of filler samples (never decoded here), the audio with its edit list.
fn mp4_with(a: &PreparedAudio, dir: &std::path::Path, name: &str) -> std::path::PathBuf {
    use codec::encode::EncodedPacket;
    let mut muxer = container::mux::Av1Mp4Muxer::new(64, 64, 30.0).unwrap();
    muxer.with_audio(a.info.clone()).unwrap();
    muxer.set_audio_edit(a.edit);
    let header: u8 = (1 << 3) | (1 << 1);
    let mut first = vec![header, 5];
    first.extend_from_slice(&[0u8; 5]);
    muxer.add_packet(EncodedPacket { data: Bytes::from(first), pts: 0, is_keyframe: true }).unwrap();
    for i in 1..15u64 {
        muxer.add_packet(EncodedPacket { data: Bytes::from(vec![0xAA; 64]), pts: i, is_keyframe: false }).unwrap();
    }
    for (sample, dur) in &a.samples {
        muxer.add_audio_sample(sample, 0, *dur).unwrap();
    }
    let path = dir.join(name);
    std::fs::write(&path, muxer.finalize().unwrap()).unwrap();
    path
}

/// Every channel of 5.1 AC-3 comes out of the AAC encoder in its own slot:
/// the MP4 (esds, channelConfiguration 6, `mp4a.40.2`) decoded by ffmpeg
/// has each channel's tone where it belongs and nothing else, and the edit
/// list hides the encoder's priming so the length is the source's.
#[test]
fn ac3_5_1_to_aac_keeps_every_channel_in_its_place() {
    let t = track("tones_51_ac3.mka");
    let a = prepare_audio(Some(&t), None, &[], request(AudioCodecPolicy::ForceAac, AudioChannels::Source, AudioOutput::Mp4))
        .unwrap()
        .unwrap();
    assert_eq!(a.handling, "ac3 → aac (6ch)");
    assert_eq!((a.info.codec.as_str(), a.info.channels, a.info.sample_rate), ("aac", 6, 48_000));
    let asc = container::aac_asc::parse_aac_asc(&a.info.asc_bytes).expect("the ASC");
    assert_eq!((asc.aot, asc.channel_configuration), (2, 6));
    assert_eq!(audio_codec_string(&a.info), "mp4a.40.2");
    assert_eq!(a.edit.media_time, 1024, "one frame of priming, hidden by the edit list");
    container::mux::Av1Mp4Muxer::check_audio(&a.info).expect("AAC in MP4");
    if !ffmpeg_available() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let path = mp4_with(&a, dir.path(), "surround.mp4");
    let p = probe_audio(&path);
    assert_eq!(p["codec_name"], "aac");
    assert_eq!(p["profile"], "LC");
    assert_eq!(p["channels"], "6");
    assert_eq!(p["channel_layout"], "5.1");
    let pcm = ffmpeg_pcm(&path);
    let source_len = a.edit.duration.expect("an encode states its length") as usize;
    assert_eq!(pcm.len() / 6, source_len, "the edit list presents exactly the source's samples");
    // The source is 5.1(side); 5.1 carries its side pair in the back slots.
    for (c, &tone) in TONES.iter().enumerate() {
        let own = amplitude(&pcm, 6, c, tone, 48_000.0);
        close(own, LEVEL, &format!("channel {c}'s own {tone} Hz"));
        for &other in TONES.iter().filter(|&&o| o != tone) {
            let leak = amplitude(&pcm, 6, c, other, 48_000.0);
            assert!(leak < 0.01, "channel {c} carries {other} Hz at {leak:.4}");
        }
    }
}

/// `audio-channels=stereo` with AAC: the BS.775 downmix, in a stereo MP4.
#[test]
fn ac3_5_1_to_stereo_aac_mp4() {
    let t = track("tones_51_ac3.mka");
    let a = prepare_audio(Some(&t), None, &[], request(AudioCodecPolicy::ForceAac, AudioChannels::Stereo, AudioOutput::Mp4))
        .unwrap()
        .unwrap();
    assert_eq!(a.handling, "ac3 → aac (6ch → 2ch)");
    assert_eq!(container::aac_asc::parse_aac_asc(&a.info.asc_bytes).unwrap().channel_configuration, 2);
    if !ffmpeg_available() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let path = mp4_with(&a, dir.path(), "stereo.mp4");
    let p = probe_audio(&path);
    assert_eq!((p["codec_name"].as_str(), p["channels"].as_str(), p["channel_layout"].as_str()), ("aac", "2", "stereo"));
    let pcm = ffmpeg_pcm(&path);
    // L = 0.414·FL + 0.293·FC + 0.293·SL, the LFE dropped, nothing across.
    let n = 1.0 + 2.0 * std::f32::consts::FRAC_1_SQRT_2;
    close(amplitude(&pcm, 2, 0, FL, 48_000.0), LEVEL / n, "FL in L");
    close(amplitude(&pcm, 2, 0, FC, 48_000.0), LEVEL * std::f32::consts::FRAC_1_SQRT_2 / n, "FC in L");
    close(amplitude(&pcm, 2, 1, SR, 48_000.0), LEVEL * std::f32::consts::FRAC_1_SQRT_2 / n, "SR in R");
    assert!(amplitude(&pcm, 2, 0, FR, 48_000.0) < 0.01, "FR leaks into L");
    assert!(amplitude(&pcm, 2, 0, LFE, 48_000.0) < 0.01, "the LFE is dropped");
}

/// HLS with AAC: each rendition's CHANNELS and `mp4a.40.2`, and its CMAF
/// segments decode cleanly with every channel in place.
#[test]
fn hls_aac_renditions_signal_their_channels() {
    let t = track("tones_51_ac3.mka");
    let surround = prepare_audio(Some(&t), None, &[], request(AudioCodecPolicy::ForceAac, AudioChannels::Source, AudioOutput::Cmaf))
        .unwrap()
        .unwrap();
    let stereo = prepare_audio(Some(&t), None, &[], request(AudioCodecPolicy::ForceAac, AudioChannels::Stereo, AudioOutput::Cmaf))
        .unwrap()
        .unwrap();
    let dir = tempfile::tempdir().unwrap();
    let main = build_audio_rendition(dir.path(), &surround, 4.0, "audio", "Surround").unwrap().unwrap();
    let fallback = build_audio_rendition(dir.path(), &stereo, 4.0, "audio-stereo", "Stereo").unwrap().unwrap();
    assert_eq!((main.channels, main.codec_string.as_str()), (6, "mp4a.40.2"));
    assert_eq!((fallback.channels, fallback.codec_string.as_str()), (2, "mp4a.40.2"));
    let video = container::hls::VideoVariantSpec {
        width: 640,
        height: 360,
        frame_rate: 30.0,
        bandwidth_bps: 800_000,
        average_bandwidth_bps: 600_000,
        codec_string: "av01.0.04M.08".into(),
        supplemental_codecs: None,
        video_range: None,
        relative_dir: "video/360p".into(),
        manifest: main.manifest.clone(),
    };
    let paths = container::hls::write_hls_package(dir.path(), &[video], &[fallback, main], &[], 4).unwrap();
    let master = std::fs::read_to_string(&paths.master_path).unwrap();
    assert!(master.contains(r#"CHANNELS="6",URI="audio/audio.m3u8""#), "{master}");
    assert!(master.contains(r#"CHANNELS="2",URI="audio-stereo/audio.m3u8""#), "{master}");
    assert!(master.contains(r#"CODECS="av01.0.04M.08,mp4a.40.2""#), "{master}");
    if !ffmpeg_available() {
        return;
    }
    let playlist = dir.path().join("audio/audio.m3u8");
    let p = probe_audio(&playlist);
    assert_eq!((p["codec_name"].as_str(), p["channels"].as_str()), ("aac", "6"));
    let pcm = ffmpeg_pcm(&playlist);
    for (c, &tone) in TONES.iter().enumerate() {
        close(amplitude(&pcm, 6, c, tone, 48_000.0), LEVEL, &format!("HLS channel {c}'s own {tone} Hz"));
    }
}

#[test]
fn aac_passes_aac_through_and_is_refused_where_it_cannot_go() {
    // An AAC source asked for AAC is a passthrough, not a re-encode.
    let t = track("tones_51_aac.m4a");
    let a = prepare_audio(Some(&t), None, &[], request(AudioCodecPolicy::ForceAac, AudioChannels::Source, AudioOutput::Mp4))
        .unwrap()
        .unwrap();
    assert_eq!(a.handling, "aac passthrough");
    // A bare .mp3 cannot hold AAC.
    let err = OutputSpec::audio_only().with_audio(AudioCodecPolicy::ForceAac).validate().unwrap_err();
    assert!(format!("{err:#}").contains("cannot hold AAC"), "{err:#}");
    let spec = OutputSpec::single_file(vec![Rung::new(640, 360)]).with_audio(AudioCodecPolicy::ForceAac);
    assert_eq!(spec.audio_encode_codec(), codec::audio::AudioCodec::Aac);
    spec.validate().expect("AAC into a single-file MP4");
    let mut low = spec.clone();
    low.audio_bitrate = Some(1_000);
    let err = low.validate().unwrap_err();
    assert!(format!("{err:#}").contains("outside AAC's range"), "{err:#}");
}
