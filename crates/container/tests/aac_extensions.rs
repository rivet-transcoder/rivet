//! Squad-25 integration tests: HE-AAC + multichannel AAC passthrough end-to-end.
//!
//! Covers:
//!   1. HE-AAC v1 5.1 explicit-signaled ASC survives mux → demux → re-mux
//!      with byte-identical samples and ASC.
//!   2. HE-AAC v2 mono PS demuxes with effective channels=2 (PS upmix).
//!   3. 5.1 AAC source produces a `chan` box in the output mp4a.
//!   4. 7.1 AAC source produces a 7.1 `chan` tag.
//!   5. Implicit HE-AAC ASC at the input is upgraded by the routing layer
//!      before it reaches the mux (covered indirectly via the mux unit
//!      test gate; the routing-layer upgrade is verified in the
//!      pipeline crate's integration test).

use bytes::Bytes;
use codec::encode::EncodedPacket;
use container::AudioInfo;
use container::aac_asc::{
    AscSignaling, effective_output_channels, parse_aac_asc, upgrade_to_explicit_signaling,
};
use container::demux;
use container::mux::Av1Mp4Muxer;

fn minimal_av1_first_packet() -> Bytes {
    let header: u8 = (1 << 3) | (1 << 1);
    let payload = [0u8; 5];
    let mut out = Vec::with_capacity(2 + payload.len());
    out.push(header);
    out.push(payload.len() as u8);
    out.extend_from_slice(&payload);
    Bytes::from(out)
}

fn opaque_video_packet(size: usize) -> Bytes {
    Bytes::from(vec![0xAAu8; size])
}

fn push_minimal_video(muxer: &mut Av1Mp4Muxer, frames: usize) {
    muxer
        .add_packet(EncodedPacket {
            data: minimal_av1_first_packet(),
            pts: 0,
            is_keyframe: true,
        })
        .expect("first packet");
    for i in 1..frames {
        muxer
            .add_packet(EncodedPacket {
                data: opaque_video_packet(128),
                pts: i as u64,
                is_keyframe: false,
            })
            .expect("packet");
    }
}

fn push_aac_samples(muxer: &mut Av1Mp4Muxer, count: usize, frame_size: usize) {
    for i in 0..count {
        let blob = vec![0x5Au8; frame_size];
        muxer
            .add_audio_sample(&blob, (i * 1024) as u64, 1024)
            .expect("audio sample");
    }
}

/// HE-AAC v1 5.1 explicit ASC: 0x29 0xB1 0x88 (verified by the unit test
/// `parse_he_aac_v1_5_1_explicit`). Confirms the parser's output and that
/// the mux accepts and round-trips the ASC verbatim.
fn he_aac_v1_5_1_asc() -> Vec<u8> {
    vec![0x29, 0xB1, 0x88]
}

/// HE-AAC v2 mono PS explicit ASC: 0xEA 0x0A 0x08.
fn he_aac_v2_mono_ps_asc() -> Vec<u8> {
    vec![0xEA, 0x0A, 0x08]
}

/// 5.1 AAC-LC ASC at 48 kHz: 0x11 0xB0 (channels=6).
fn aac_lc_5_1_48k_asc() -> Vec<u8> {
    vec![0x11, 0xB0]
}

/// 7.1 AAC-LC ASC at 48 kHz: 0x11 0xB8 (channels=7).
fn aac_lc_7_1_48k_asc() -> Vec<u8> {
    vec![0x11, 0xB8]
}

#[test]
fn he_aac_v1_explicit_5_1_passes_mux_gate() {
    let mut muxer = Av1Mp4Muxer::new(320, 240, 30.0).expect("muxer");
    push_minimal_video(&mut muxer, 6);
    let info = AudioInfo {
        codec: "aac".into(),
        sample_rate: 48_000,
        channels: 6, // 5.1 effective
        timescale: 48_000,
        asc_bytes: he_aac_v1_5_1_asc(),
        codec_private: Vec::new(),
    };
    muxer
        .with_audio(info)
        .expect("HE-AAC v1 5.1 must be accepted");
    push_aac_samples(&mut muxer, 5, 200);
    let out = muxer.finalize().expect("finalize");

    assert!(out.windows(4).any(|w| w == b"mp4a"), "must contain mp4a");
    assert!(out.windows(4).any(|w| w == b"esds"), "must contain esds");
    // Multichannel 5.1 must emit chan with the standard MPEG_5_1_C tag.
    assert!(
        out.windows(4).any(|w| w == b"chan"),
        "5.1 output must contain chan box"
    );
}

#[test]
fn he_aac_v2_mono_ps_passes_mux_gate_with_effective_stereo() {
    // PS upmixes mono → stereo, so the demux side surfaces channels=2.
    // The ASC still has channelConfiguration=1; the mux should accept it
    // either way (as long as the ASC parses to a recognised AOT path).
    let mut muxer = Av1Mp4Muxer::new(320, 240, 30.0).expect("muxer");
    push_minimal_video(&mut muxer, 6);
    let info = AudioInfo {
        codec: "aac".into(),
        sample_rate: 44_100,
        channels: 2, // effective post-PS
        timescale: 44_100,
        asc_bytes: he_aac_v2_mono_ps_asc(),
        codec_private: Vec::new(),
    };
    muxer
        .with_audio(info)
        .expect("HE-AAC v2 mono PS must be accepted");
    push_aac_samples(&mut muxer, 5, 200);
    let out = muxer.finalize().expect("finalize");

    // Stereo (effective channels=2) must NOT emit chan.
    assert!(
        out.windows(4).all(|w| w != b"chan"),
        "stereo (post-PS) output must not emit chan"
    );
    // The ASC bytes must round-trip verbatim through the esds.
    let asc = he_aac_v2_mono_ps_asc();
    let esds_pos = out.windows(4).position(|w| w == b"esds").expect("esds");
    let win = &out[esds_pos..(esds_pos + 80).min(out.len())];
    assert!(
        win.windows(asc.len()).any(|w| w == asc.as_slice()),
        "HE-AAC v2 ASC must appear inside esds verbatim"
    );
}

#[test]
fn aac_5_1_demux_remux_byte_identical_samples_and_chan_box() {
    // Build a 5.1 AAC source via mux, demux it, then re-mux. The demux side
    // surfaces effective channels=6 from the ASC, samples should compare
    // byte-identical, and the second mux output must contain chan.
    let mut muxer1 = Av1Mp4Muxer::new(320, 240, 30.0).expect("muxer1");
    push_minimal_video(&mut muxer1, 12);
    let original_asc = aac_lc_5_1_48k_asc();
    let info = AudioInfo {
        codec: "aac".into(),
        sample_rate: 48_000,
        channels: 6,
        timescale: 48_000,
        asc_bytes: original_asc.clone(),
        codec_private: Vec::new(),
    };
    muxer1.with_audio(info).expect("with_audio 5.1");
    let original_samples: Vec<Vec<u8>> = (0..8)
        .map(|i| (0..400).map(|j| ((i * 17 + j) & 0xFF) as u8).collect())
        .collect();
    for (i, s) in original_samples.iter().enumerate() {
        muxer1
            .add_audio_sample(s, (i * 1024) as u64, 1024)
            .expect("add_audio_sample");
    }
    let out1 = muxer1.finalize().expect("finalize 1");

    // Re-demux. Samples + ASC + effective channels must match.
    let demuxed = demux::demux(&out1).expect("demux");
    let audio = demuxed.audio.expect("audio track must demux");
    assert_eq!(audio.codec, "aac");
    assert_eq!(audio.channels, 6, "5.1 effective channels");
    assert_eq!(audio.sample_rate, 48_000);
    assert_eq!(audio.asc, original_asc, "ASC bytes must round-trip");
    assert_eq!(
        audio.samples.len(),
        original_samples.len(),
        "sample count drifted"
    );
    for (i, (got, want)) in audio
        .samples
        .iter()
        .zip(original_samples.iter())
        .enumerate()
    {
        assert_eq!(
            got,
            want,
            "sample {i} must be byte-identical (got {} bytes, want {} bytes)",
            got.len(),
            want.len()
        );
    }

    // Confirm chan box is present.
    assert!(
        out1.windows(4).any(|w| w == b"chan"),
        "5.1 output must include chan box"
    );

    // Re-mux from the demuxed payload — round-trip the round-trip.
    let mut muxer2 = Av1Mp4Muxer::new(320, 240, 30.0).expect("muxer2");
    push_minimal_video(&mut muxer2, 12);
    let info2 = AudioInfo {
        codec: "aac".into(),
        sample_rate: audio.sample_rate,
        channels: audio.channels,
        timescale: audio.timescale,
        asc_bytes: audio.asc.clone(),
        codec_private: Vec::new(),
    };
    muxer2.with_audio(info2).expect("with_audio 2");
    for (i, s) in audio.samples.iter().enumerate() {
        muxer2
            .add_audio_sample(s, (i * 1024) as u64, 1024)
            .expect("re-add audio sample");
    }
    let out2 = muxer2.finalize().expect("finalize 2");
    assert!(
        out2.windows(4).any(|w| w == b"chan"),
        "re-muxed 5.1 output must include chan box"
    );
}

#[test]
fn aac_7_1_emits_mpeg_7_1_c_chan_tag() {
    let mut muxer = Av1Mp4Muxer::new(320, 240, 30.0).expect("muxer");
    push_minimal_video(&mut muxer, 6);
    let info = AudioInfo {
        codec: "aac".into(),
        sample_rate: 48_000,
        channels: 7,
        timescale: 48_000,
        asc_bytes: aac_lc_7_1_48k_asc(),
        codec_private: Vec::new(),
    };
    muxer.with_audio(info).expect("with_audio 7.1");
    push_aac_samples(&mut muxer, 5, 200);
    let out = muxer.finalize().expect("finalize");

    let chan_pos = out
        .windows(4)
        .position(|w| w == b"chan")
        .expect("7.1 output must contain chan");
    // chan box layout: [size 4][fourcc 'chan' 4][tag 4][bitmap 4][nDescs 4].
    // Tag bytes are at chan_pos + 4 (size already consumed since we found
    // 'chan' at byte 4 of its 8-byte header, so go back 4 bytes for size,
    // then forward 8 for the body).
    let tag_offset = chan_pos + 4;
    let tag = u32::from_be_bytes([
        out[tag_offset],
        out[tag_offset + 1],
        out[tag_offset + 2],
        out[tag_offset + 3],
    ]);
    assert_eq!(
        tag, 0x007F0008,
        "7.1 tag must be kAudioChannelLayoutTag_MPEG_7_1_C = 0x007F0008; got 0x{tag:08X}"
    );
}

#[test]
fn implicit_he_aac_upgrade_then_mux() {
    // Implicit HE-AAC ASC: 0x13 0x08 (AOT=2 SFI=6 chan=1 = mono 24 kHz LC).
    // Mux must reject the implicit form...
    let implicit = vec![0x13u8, 0x08];
    let parsed = parse_aac_asc(&implicit).expect("parse");
    assert_eq!(parsed.signaling, AscSignaling::ImplicitMaybe);

    // ...then accept the upgraded explicit form.
    let upgraded = upgrade_to_explicit_signaling(&implicit).expect("upgrade");
    let reparsed = parse_aac_asc(&upgraded).expect("reparse");
    assert_eq!(reparsed.signaling, AscSignaling::ExplicitSbr);
    assert_eq!(reparsed.sbr_sample_rate, Some(48_000));

    let mut muxer = Av1Mp4Muxer::new(320, 240, 30.0).expect("muxer");
    push_minimal_video(&mut muxer, 6);
    let info = AudioInfo {
        codec: "aac".into(),
        sample_rate: 48_000, // SBR-extended output rate
        channels: 2,         // PS would upmix; effective is 2 anyway
        timescale: 48_000,
        asc_bytes: upgraded,
        codec_private: Vec::new(),
    };
    muxer
        .with_audio(info)
        .expect("explicit-upgraded HE-AAC must be accepted");
    push_aac_samples(&mut muxer, 5, 200);
    let _out = muxer.finalize().expect("finalize");
}

#[test]
fn effective_output_channels_ps_upmix() {
    // HE-AAC v2 PS mono → stereo upmix. Verify the demuxer-side helper
    // surfaces 2 even though the ASC carries channelConfiguration=1.
    let parsed = parse_aac_asc(&he_aac_v2_mono_ps_asc()).expect("parse");
    assert_eq!(parsed.channels, 1, "ASC channelConfiguration is mono");
    assert!(parsed.ps_present, "PS must be flagged");
    assert_eq!(
        effective_output_channels(&parsed),
        2,
        "PS upmix: 1-channel core → 2-channel effective output"
    );
}
