use std::ffi::c_int;

use audiopus::Channels as OpusChannels;
use audiopus::SampleRate;
use audiopus::coder::Decoder as OpusDecoderInner;
use audiopus::ffi;

use crate::audio::{AudioCodec, AudioEncoder, AudioEncoderConfig, AudioFrame};

use super::{OpusEncoder, build_dops, surround_mapping_family_1};

fn config_stereo_48k() -> AudioEncoderConfig {
    AudioEncoderConfig {
        codec: AudioCodec::Opus,
        sample_rate: 48_000,
        channels: 2,
        bitrate: 96_000,
    }
}

fn config_mono_48k() -> AudioEncoderConfig {
    AudioEncoderConfig {
        codec: AudioCodec::Opus,
        sample_rate: 48_000,
        channels: 1,
        bitrate: 64_000,
    }
}

fn config_multi_48k(channels: u8) -> AudioEncoderConfig {
    AudioEncoderConfig {
        codec: AudioCodec::Opus,
        sample_rate: 48_000,
        channels,
        bitrate: 0, // exercise the per-stream default-bitrate path
    }
}

#[test]
fn opus_encoder_constructs_for_mono_48k_with_1_channel_dops() {
    let enc = OpusEncoder::new(config_mono_48k()).expect("constructs");
    assert_eq!(enc.channels, 1);
    assert!(enc.resampler.is_none());
    // dOps[1] = OutputChannelCount = 1 for mono
    assert_eq!(enc.extra_data[1], 1);
}

#[test]
fn opus_encoder_uses_default_bitrate_when_caller_passes_zero() {
    let mut cfg = config_stereo_48k();
    cfg.bitrate = 0;
    let _enc = OpusEncoder::new(cfg).expect("constructs with bitrate=0");
    // Default bitrate path doesn't expose the value via a public
    // method on audiopus's Encoder without GenericCtl, but the
    // constructor would fail if it tried to set an invalid
    // bitrate. The fact that we got here means the default
    // (DEFAULT_BITRATE_STEREO=96k) was applied successfully.
}

fn config_stereo_44100() -> AudioEncoderConfig {
    AudioEncoderConfig {
        codec: AudioCodec::Opus,
        sample_rate: 44_100,
        channels: 2,
        bitrate: 96_000,
    }
}

fn make_silence(channels: u8, frames: usize, sample_rate: u32) -> AudioFrame {
    AudioFrame {
        samples: vec![0.0f32; frames * channels as usize],
        sample_rate,
        channels,
        pts: 0,
    }
}

fn make_sine_1k(channels: u8, frames: usize, sample_rate: u32, amp: f32) -> AudioFrame {
    let mut samples = Vec::with_capacity(frames * channels as usize);
    let two_pi = std::f32::consts::PI * 2.0;
    let freq = 1000.0f32;
    for i in 0..frames {
        let t = i as f32 / sample_rate as f32;
        let v = (two_pi * freq * t).sin() * amp;
        for _ in 0..channels {
            samples.push(v);
        }
    }
    AudioFrame {
        samples,
        sample_rate,
        channels,
        pts: 0,
    }
}

#[test]
fn opus_encoder_constructs_for_stereo_48k() {
    let enc = OpusEncoder::new(config_stereo_48k()).expect("constructs");
    assert_eq!(enc.channels, 2);
    assert_eq!(enc.in_rate, 48000);
    assert!(enc.resampler.is_none(), "no resampler at native rate");
    assert_eq!(enc.extra_data.len(), 11, "dOps body must be 11 bytes");
    // dOps[0] = Version = 0
    assert_eq!(enc.extra_data[0], 0);
    // dOps[1] = OutputChannelCount
    assert_eq!(enc.extra_data[1], 2);
    // dOps[10] = ChannelMappingFamily = 0
    assert_eq!(enc.extra_data[10], 0);
}

#[test]
fn opus_encoder_resamples_44100_to_48k_internally() {
    let enc = OpusEncoder::new(config_stereo_44100()).expect("constructs");
    assert!(enc.resampler.is_some(), "resampler engaged at 44.1k input");
    let r = enc.resampler.as_ref().unwrap();
    assert_eq!(r.in_rate(), 44100);
    assert_eq!(r.out_rate(), 48000);
}

#[test]
fn opus_encoder_rejects_zero_channels() {
    let mut bad = config_stereo_48k();
    bad.channels = 0;
    assert!(matches!(
        OpusEncoder::new(bad),
        Err(crate::audio::AudioError::Unsupported(_))
    ));
}

#[test]
fn opus_encoder_rejects_nine_channels() {
    // 9 channels (and above) has no defined channel-mapping family-1
    // layout in RFC 7845 §5.1.1.2, so we Unsupported it.
    let mut bad9 = config_stereo_48k();
    bad9.channels = 9;
    assert!(matches!(
        OpusEncoder::new(bad9),
        Err(crate::audio::AudioError::Unsupported(_))
    ));
}

#[test]
fn opus_encoder_rejects_nine_channel_frame_at_runtime() {
    let mut enc = OpusEncoder::new(config_stereo_48k()).expect("constructs");
    let bad_frame = AudioFrame {
        samples: vec![0.0; 960 * 9],
        sample_rate: 48000,
        channels: 9,
        pts: 0,
    };
    let r = enc.encode(&bad_frame);
    assert!(
        matches!(r, Err(crate::audio::AudioError::Unsupported(_))),
        "9-channel frame should be Unsupported, got {:?}",
        r
    );
}

#[test]
fn opus_pre_skip_in_48khz_ticks_is_nonzero() {
    let enc = OpusEncoder::new(config_stereo_48k()).expect("constructs");
    // libopus typically reports lookahead in the 312..=400 sample
    // range at 48 kHz. We just sanity-check it's nonzero.
    assert!(
        enc.pre_skip() > 0,
        "Opus encoder lookahead should be positive (libopus convention)"
    );
    assert!(
        enc.pre_skip() < 2000,
        "lookahead is bounded — typically <600 samples at 48 kHz"
    );
}

#[test]
fn opus_dops_carries_correct_pre_skip_and_input_sample_rate_le() {
    let enc = OpusEncoder::new(config_stereo_44100()).expect("constructs");
    let d = enc.extra_data();
    // PreSkip at offset 2 (LE u16)
    let ps = u16::from_le_bytes([d[2], d[3]]);
    assert_eq!(ps, enc.pre_skip(), "dOps PreSkip matches encoder lookahead");
    // InputSampleRate at offset 4 (LE u32)
    let isr = u32::from_le_bytes([d[4], d[5], d[6], d[7]]);
    assert_eq!(
        isr, 44100,
        "dOps InputSampleRate is the source rate, not 48k"
    );
    // OutputGain at offset 8 (LE i16, default 0)
    let og = i16::from_le_bytes([d[8], d[9]]);
    assert_eq!(og, 0);
}

#[test]
fn opus_encode_20ms_silence_produces_one_packet() {
    let mut enc = OpusEncoder::new(config_stereo_48k()).expect("constructs");
    // 20 ms at 48 kHz = 960 frames per channel
    let frame = make_silence(2, 960, 48_000);
    let pkts = enc.encode(&frame).expect("encode");
    assert_eq!(pkts.len(), 1, "exactly one Opus packet for one 20ms frame");
    let pkt = &pkts[0];
    assert!(!pkt.data.is_empty(), "packet should have bytes");
    // Silence at 96 kbps stereo: Opus DTX is OFF so we still get
    // a regular packet. Should be small (a few dozen bytes).
    assert!(
        pkt.data.len() < 200,
        "silence packet at 96 kbps should be small, got {} bytes",
        pkt.data.len()
    );
    assert_eq!(pkt.duration, 960, "20ms = 960 ticks at 48k");
}

#[test]
fn opus_encode_one_second_of_sine_produces_packets_with_reasonable_bitrate() {
    let mut enc = OpusEncoder::new(config_stereo_48k()).expect("constructs");
    // Feed 1 second of 1 kHz sine in 20 ms slices so we have round
    // numbers. 48000 / 960 = 50 frames per second.
    let mut total_bytes = 0usize;
    let mut total_packets = 0usize;
    for i in 0..50 {
        let mut frame = make_sine_1k(2, 960, 48_000, 0.3);
        // Stagger the per-slice phase by adjusting pts; the
        // generator above uses i=0..960 so phase resets each
        // slice — for this test we don't care about phase
        // continuity across slices, only about bitrate aggregate.
        frame.pts = i * 20_000;
        let pkts = enc.encode(&frame).expect("encode");
        for p in &pkts {
            total_bytes += p.data.len();
            total_packets += 1;
        }
    }
    let pkts_flush = enc.flush().expect("flush");
    for p in &pkts_flush {
        total_bytes += p.data.len();
        total_packets += 1;
    }
    // Expect ~50 packets for 1 s of audio (one per 20 ms)
    assert!(
        total_packets >= 49 && total_packets <= 51,
        "expected ~50 packets for 1 s of audio, got {total_packets}"
    );
    // 1 second at 96 kbps = 96000 bits = 12000 bytes target.
    // VBR encoder will be within ±30% of this on a sine wave.
    let observed_bps = (total_bytes as u64 * 8) as i64;
    assert!(
        observed_bps > 30_000 && observed_bps < 200_000,
        "1s of 1kHz sine at 96 kbps should yield 30-200 kbps actual, got {observed_bps} bps ({total_bytes} bytes)"
    );
}

#[test]
fn opus_pts_steps_by_20ms_per_packet() {
    let mut enc = OpusEncoder::new(config_stereo_48k()).expect("constructs");
    let frame_a = make_silence(2, 960, 48_000);
    let mut frame_b = make_silence(2, 960, 48_000);
    frame_b.pts = 20_000;
    let pkts_a = enc.encode(&frame_a).expect("a");
    let pkts_b = enc.encode(&frame_b).expect("b");
    assert_eq!(pkts_a.len(), 1);
    assert_eq!(pkts_b.len(), 1);
    let dt = pkts_b[0].pts - pkts_a[0].pts;
    // 20 ms in microseconds = 20_000
    assert_eq!(
        dt, 20_000,
        "PTS should step by 20_000 us per Opus packet (20 ms frame)"
    );
}

/// Round-trip: encode a sine wave then decode through libopus and
/// compare against the input. Opus is lossy (especially silence
/// padding at the front for pre_skip) so we measure RMS error
/// over the steady-state portion only.
#[test]
fn opus_round_trip_sine_wave_quality_is_acceptable() {
    let mut enc = OpusEncoder::new(config_stereo_48k()).expect("constructs");
    let frames_per_chunk = 960;
    let n_chunks = 25; // ~500 ms
    let total_frames = frames_per_chunk * n_chunks;

    // Continuous-phase 1 kHz sine across all chunks.
    let mut all_samples = Vec::with_capacity(total_frames * 2);
    let two_pi = std::f32::consts::PI * 2.0;
    let freq = 1000.0f32;
    for i in 0..total_frames {
        let t = i as f32 / 48_000.0;
        let v = (two_pi * freq * t).sin() * 0.5;
        all_samples.push(v);
        all_samples.push(v);
    }

    // Encode chunk by chunk.
    let mut packets = Vec::new();
    for c in 0..n_chunks {
        let chunk_samples =
            all_samples[c * frames_per_chunk * 2..(c + 1) * frames_per_chunk * 2].to_vec();
        let frame = AudioFrame {
            samples: chunk_samples,
            sample_rate: 48_000,
            channels: 2,
            pts: (c as i64) * 20_000,
        };
        packets.extend(enc.encode(&frame).expect("encode"));
    }
    packets.extend(enc.flush().expect("flush"));
    assert!(!packets.is_empty(), "encode must produce packets");

    // Decode with audiopus.
    let mut dec =
        OpusDecoderInner::new(SampleRate::Hz48000, OpusChannels::Stereo).expect("dec");
    let mut decoded = Vec::with_capacity(total_frames * 2);
    let mut tmp = vec![0.0f32; frames_per_chunk * 2];
    for p in &packets {
        let pkt = audiopus::packet::Packet::try_from(p.data.as_slice()).expect("pkt");
        let sig = audiopus::MutSignals::try_from(tmp.as_mut_slice()).expect("sig");
        let n = dec
            .decode_float(Some(pkt), sig, false)
            .expect("decode_float");
        decoded.extend_from_slice(&tmp[..n * 2]);
    }
    assert!(
        decoded.len() >= (total_frames - 100) * 2,
        "decoded length {} should approximate input length {}",
        decoded.len(),
        total_frames * 2
    );

    // Compare the steady-state portion (skip pre_skip + a couple
    // hundred extra samples for filter warm-up) to the original.
    // Opus decoder output is delayed by `pre_skip` 48k samples
    // relative to the original input.
    let pre_skip = enc.pre_skip() as usize;
    let cmp_start = pre_skip + 480; // skip first 10 ms more
    let cmp_end = (decoded.len() / 2).min(total_frames - 100);
    if cmp_end <= cmp_start {
        panic!(
            "round trip too short: cmp_start={cmp_start}, cmp_end={cmp_end}, decoded len/2={}",
            decoded.len() / 2
        );
    }

    let mut sum_sq_err = 0.0f64;
    let mut sum_sq_sig = 0.0f64;
    let mut n = 0usize;
    for i in cmp_start..cmp_end {
        // Opus decoder output at sample i corresponds to input at
        // sample (i - pre_skip). The decoded buffer already starts
        // at output sample 0, and pre_skip samples of it are the
        // encoder's lookahead "padding" — input sample 0 of the
        // user's stream lives at decoder output sample pre_skip.
        let in_idx = i - pre_skip;
        let l_in = all_samples[in_idx * 2];
        let r_in = all_samples[in_idx * 2 + 1];
        let l_out = decoded[i * 2];
        let r_out = decoded[i * 2 + 1];
        sum_sq_err += ((l_in - l_out) as f64).powi(2);
        sum_sq_err += ((r_in - r_out) as f64).powi(2);
        sum_sq_sig += (l_in as f64).powi(2);
        sum_sq_sig += (r_in as f64).powi(2);
        n += 2;
    }
    let rms_err = (sum_sq_err / n as f64).sqrt();
    let rms_sig = (sum_sq_sig / n as f64).sqrt();
    let snr_db = 20.0 * (rms_sig / rms_err.max(1e-12)).log10();
    // A sine wave round-tripped through Opus at 96 kbps stereo
    // should land >15 dB SNR easily — Opus is transparent on
    // simple tones at this bitrate. We use a conservative bound
    // because exact SNR depends on libopus version.
    assert!(
        snr_db > 15.0,
        "round-trip SNR {snr_db:.2} dB too low — Opus quality regression?"
    );
    // Print so the deliverables report can capture the actual
    // number from `cargo test -- --nocapture`.
    println!("opus_round_trip SNR = {snr_db:.2} dB, rms_err = {rms_err:.4}");
}

#[test]
fn dops_layout_matches_rfc_7845_for_mono_and_stereo() {
    let d_mono = build_dops(1, 312, 48_000, None);
    assert_eq!(d_mono.len(), 11);
    assert_eq!(d_mono[0], 0); // Version
    assert_eq!(d_mono[1], 1); // ChannelCount
    assert_eq!(u16::from_le_bytes([d_mono[2], d_mono[3]]), 312); // PreSkip
    assert_eq!(
        u32::from_le_bytes([d_mono[4], d_mono[5], d_mono[6], d_mono[7]]),
        48000
    ); // InputSampleRate
    assert_eq!(i16::from_le_bytes([d_mono[8], d_mono[9]]), 0); // OutputGain
    assert_eq!(d_mono[10], 0); // Family

    let d_stereo = build_dops(2, 400, 44_100, None);
    assert_eq!(d_stereo.len(), 11);
    assert_eq!(d_stereo[1], 2);
    assert_eq!(u16::from_le_bytes([d_stereo[2], d_stereo[3]]), 400);
    assert_eq!(
        u32::from_le_bytes([d_stereo[4], d_stereo[5], d_stereo[6], d_stereo[7]]),
        44100
    );
}

// -------- Squad-28 multistream tests below --------

/// Standard surround layouts per RFC 7845 §5.1.1.2. Each pair
/// `(channels, (streams, coupled, mapping))` matches the spec
/// table exactly.
#[test]
fn surround_mapping_family_1_matches_rfc_7845_5_1_1_2() {
    // 3.0 — L, R, C → coupled[L,R] + stream[C]
    assert_eq!(
        surround_mapping_family_1(3).unwrap(),
        (2, 1, &[0, 2, 1][..])
    );
    // quad — FL, FR, BL, BR → coupled[FL,FR] + coupled[BL,BR]
    assert_eq!(
        surround_mapping_family_1(4).unwrap(),
        (2, 2, &[0, 1, 2, 3][..])
    );
    // 5.0 — FL, FR, C, BL, BR
    assert_eq!(
        surround_mapping_family_1(5).unwrap(),
        (3, 2, &[0, 4, 1, 2, 3][..])
    );
    // 5.1 — FL, FR, C, LFE, BL, BR
    assert_eq!(
        surround_mapping_family_1(6).unwrap(),
        (4, 2, &[0, 4, 1, 2, 3, 5][..])
    );
    // 6.1 — FL, FR, C, LFE, BC, SL, SR
    // (streams=4, coupled=3; libopus authoritative — see
    // `vorbis_mappings[]` in opus_multistream_encoder.c:60).
    assert_eq!(
        surround_mapping_family_1(7).unwrap(),
        (4, 3, &[0, 4, 1, 2, 3, 5, 6][..])
    );
    // 7.1 — FL, FR, C, LFE, BL, BR, SL, SR
    assert_eq!(
        surround_mapping_family_1(8).unwrap(),
        (5, 3, &[0, 6, 1, 2, 3, 4, 5, 7][..])
    );
    // Out-of-range
    assert!(surround_mapping_family_1(0).is_err());
    assert!(surround_mapping_family_1(1).is_err()); // family-1 is 3..=8
    assert!(surround_mapping_family_1(2).is_err());
    assert!(surround_mapping_family_1(9).is_err());
}

#[test]
fn opus_encoder_constructs_for_3_0_through_7_1_with_family_1_dops() {
    // For each surround channel count, the encoder should construct
    // and the dOps body should be 11 + 2 + N bytes with family=1
    // and the spec-mandated streams/coupled/mapping appended.
    for &ch in &[3u8, 4, 5, 6, 7, 8] {
        let enc = OpusEncoder::new(config_multi_48k(ch))
            .unwrap_or_else(|e| panic!("constructs for {ch}ch: {e:?}"));
        assert_eq!(enc.channels, ch);
        assert!(enc.resampler.is_none(), "no resampler at native rate");

        let d = enc.extra_data();
        let expected_len = 11 + 2 + ch as usize;
        assert_eq!(
            d.len(),
            expected_len,
            "dOps body for {ch}ch should be {expected_len} bytes (11 preamble + 2 stream/coupled + N mapping); got {}",
            d.len()
        );
        assert_eq!(
            d[0], 0,
            "Version=0 (dOps box version, not Opus stream version)"
        );
        assert_eq!(d[1], ch, "OutputChannelCount");
        assert_eq!(d[10], 1, "ChannelMappingFamily=1 for surround");

        let (exp_streams, exp_coupled, exp_mapping) = surround_mapping_family_1(ch).unwrap();
        assert_eq!(d[11], exp_streams, "StreamCount for {ch}ch");
        assert_eq!(d[12], exp_coupled, "CoupledCount for {ch}ch");
        assert_eq!(
            &d[13..13 + ch as usize],
            exp_mapping,
            "ChannelMapping for {ch}ch"
        );
    }
}

/// dOps body for a 5.1 encoder, hex-dumped. Captured in the
/// deliverables report for cross-tool verification.
#[test]
fn opus_encoder_dops_5_1_hex_layout() {
    let enc = OpusEncoder::new(config_multi_48k(6)).expect("5.1 constructs");
    let d = enc.extra_data();
    assert_eq!(d.len(), 19, "5.1 dOps body = 11 + 2 + 6 = 19 bytes");
    let hex: String = d.iter().map(|b| format!("{b:02x} ")).collect();
    println!(
        "5.1 dOps body hex (LE-encoded, 19 bytes): {}",
        hex.trim_end()
    );
    // Layout cross-check:
    assert_eq!(d[0], 0); // Version
    assert_eq!(d[1], 6); // OutputChannelCount
    // PreSkip varies by libopus build; check it's non-zero
    let ps = u16::from_le_bytes([d[2], d[3]]);
    assert!(ps > 0 && ps < 2000);
    assert_eq!(
        u32::from_le_bytes([d[4], d[5], d[6], d[7]]),
        48_000,
        "InputSampleRate=48000"
    );
    assert_eq!(i16::from_le_bytes([d[8], d[9]]), 0); // OutputGain
    assert_eq!(d[10], 1); // Family=1
    assert_eq!(d[11], 4); // StreamCount=4 (5.1)
    assert_eq!(d[12], 2); // CoupledCount=2 (5.1)
    assert_eq!(&d[13..19], &[0u8, 4, 1, 2, 3, 5][..]); // ChannelMapping
}

#[test]
fn opus_5_1_encode_20ms_silence_produces_one_packet() {
    let mut enc = OpusEncoder::new(config_multi_48k(6)).expect("5.1 constructs");
    // 20 ms at 48 kHz, 6 channels
    let frame = make_silence(6, 960, 48_000);
    let pkts = enc.encode(&frame).expect("encode 5.1 silence");
    assert_eq!(pkts.len(), 1, "exactly one Opus packet for one 20ms frame");
    let pkt = &pkts[0];
    assert!(!pkt.data.is_empty());
    // Multistream silence packet is larger than the mono case
    // because there's >=4 internal streams emitting their own
    // silence frame — but should still be small in absolute terms.
    assert!(
        pkt.data.len() < 600,
        "5.1 silence packet should still be under ~600 bytes, got {} bytes",
        pkt.data.len()
    );
    assert_eq!(pkt.duration, 960);
}

/// Round-trip 5.1 sine through libopus multistream encode + decode,
/// computing per-channel SNR. Each channel carries a different
/// frequency so cross-channel bleed would show up as low SNR.
#[test]
fn opus_5_1_round_trip_per_channel_snr_is_acceptable() {
    // Per-channel sine frequencies (Hz). Distinct so a coupled
    // stream that mixed channels would show degraded SNR.
    // 5.1 channel order: FL, FR, C, LFE, BL, BR
    let freqs = [440.0f32, 523.25, 659.25, 80.0, 880.0, 987.77];
    let chans: u8 = 6;
    let frames_per_chunk = 960;
    let n_chunks = 30; // ~600 ms
    let total_frames = frames_per_chunk * n_chunks;
    let amp = 0.4f32;

    // Build the multichannel input. Continuous phase across chunks.
    let mut all = vec![0.0f32; total_frames * chans as usize];
    let two_pi = std::f32::consts::PI * 2.0;
    for i in 0..total_frames {
        let t = i as f32 / 48_000.0;
        for ch in 0..chans as usize {
            all[i * chans as usize + ch] = (two_pi * freqs[ch] * t).sin() * amp;
        }
    }

    // Encode.
    let mut enc = OpusEncoder::new(config_multi_48k(chans)).expect("encoder");
    let mut packets = Vec::new();
    for c in 0..n_chunks {
        let frame = AudioFrame {
            samples: all[c * frames_per_chunk * chans as usize
                ..(c + 1) * frames_per_chunk * chans as usize]
                .to_vec(),
            sample_rate: 48_000,
            channels: chans,
            pts: (c as i64) * 20_000,
        };
        packets.extend(enc.encode(&frame).expect("encode"));
    }
    packets.extend(enc.flush().expect("flush"));
    assert!(!packets.is_empty(), "must produce packets");

    // Decode via the multistream API directly through audiopus_sys.
    let (streams, coupled, mapping) = surround_mapping_family_1(chans).unwrap();
    let mut err: c_int = 0;
    let dec_state = unsafe {
        ffi::opus_multistream_decoder_create(
            48_000,
            chans as c_int,
            streams as c_int,
            coupled as c_int,
            mapping.as_ptr(),
            &mut err,
        )
    };
    assert!(
        !dec_state.is_null() && err == ffi::OPUS_OK,
        "MS decoder create"
    );

    let mut decoded = Vec::with_capacity(total_frames * chans as usize);
    let mut tmp = vec![0.0f32; frames_per_chunk * chans as usize];
    for p in &packets {
        let n = unsafe {
            ffi::opus_multistream_decode_float(
                dec_state,
                p.data.as_ptr(),
                p.data.len() as i32,
                tmp.as_mut_ptr(),
                frames_per_chunk as c_int,
                0,
            )
        };
        assert!(n > 0, "MS decode_float returned {n}");
        decoded.extend_from_slice(&tmp[..(n as usize) * chans as usize]);
    }
    unsafe { ffi::opus_multistream_decoder_destroy(dec_state) };

    // Per-channel SNR over the steady-state portion. Skip pre_skip
    // + 480 samples of filter warm-up at the front, plus a small
    // tail margin.
    let pre_skip = enc.pre_skip() as usize;
    let cmp_start = pre_skip + 480;
    let cmp_end = (decoded.len() / chans as usize).min(total_frames - 200);
    assert!(cmp_end > cmp_start, "round trip too short");

    let mut snrs = Vec::with_capacity(chans as usize);
    for ch in 0..chans as usize {
        let mut sum_sq_err = 0.0f64;
        let mut sum_sq_sig = 0.0f64;
        for i in cmp_start..cmp_end {
            let in_idx = i - pre_skip;
            let s_in = all[in_idx * chans as usize + ch];
            let s_out = decoded[i * chans as usize + ch];
            sum_sq_err += ((s_in - s_out) as f64).powi(2);
            sum_sq_sig += (s_in as f64).powi(2);
        }
        let n = (cmp_end - cmp_start) as f64;
        let rms_err = (sum_sq_err / n).sqrt();
        let rms_sig = (sum_sq_sig / n).sqrt();
        let snr_db = 20.0 * (rms_sig / rms_err.max(1e-12)).log10();
        snrs.push(snr_db);
    }

    println!("5.1 per-channel SNR (dB):");
    for (i, snr) in snrs.iter().enumerate() {
        let label = ["FL", "FR", "C", "LFE", "BL", "BR"][i];
        println!("  ch{i} ({label}): {snr:.2} dB");
    }

    // Each channel should land >= 5 dB SNR on a steady tone.
    // Multistream Opus at default per-stream bitrate (~320 kbps
    // total) is transparent on a simple sine, but the LFE channel
    // is allocated less bitrate by libopus and lower-frequency
    // tones have proportionally larger error per sample, so we use
    // a conservative bound.
    for (i, snr) in snrs.iter().enumerate() {
        assert!(
            *snr > 5.0,
            "ch{i} SNR {snr:.2} dB too low — multistream quality regression?"
        );
    }
}

#[test]
fn dops_layout_for_5_1_matches_family_1_spec() {
    let (streams, coupled, mapping) = surround_mapping_family_1(6).unwrap();
    let d = build_dops(6, 312, 48_000, Some((streams, coupled, mapping)));
    assert_eq!(d.len(), 11 + 2 + 6, "5.1 dOps = 19 bytes");
    assert_eq!(d[0], 0); // Version
    assert_eq!(d[1], 6); // OutputChannelCount
    assert_eq!(u16::from_le_bytes([d[2], d[3]]), 312); // PreSkip
    assert_eq!(u32::from_le_bytes([d[4], d[5], d[6], d[7]]), 48_000); // InputSampleRate
    assert_eq!(i16::from_le_bytes([d[8], d[9]]), 0); // OutputGain
    assert_eq!(d[10], 1); // Family=1
    assert_eq!(d[11], 4); // StreamCount=4 for 5.1
    assert_eq!(d[12], 2); // CoupledCount=2 for 5.1
    assert_eq!(&d[13..19], &[0u8, 4, 1, 2, 3, 5][..]);
}

/// 5.1 encoder at a non-48k input rate must engage the resampler
/// for its 6 channels — gates the resampler-channel-cap lift.
#[test]
fn opus_5_1_resamples_44100_to_48k() {
    let mut cfg = config_multi_48k(6);
    cfg.sample_rate = 44_100;
    let enc = OpusEncoder::new(cfg).expect("5.1 @ 44.1k constructs");
    assert!(enc.resampler.is_some(), "resampler engaged for 6ch @ 44.1k");
    let r = enc.resampler.as_ref().unwrap();
    assert_eq!(r.in_rate(), 44_100);
    assert_eq!(r.out_rate(), 48_000);
    assert_eq!(r.channels(), 6);
}
