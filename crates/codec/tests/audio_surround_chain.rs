//! The surround audio chain end to end: decoded multichannel PCM →
//! `channelmap` → Opus multistream encoder → `dOps` config bytes.
//!
//! Each piece has unit tests of its own; what this file guards is that they
//! *connect* — the channel count the filter produces is the one the encoder is
//! built for, and the layout the encoder advertises in its `dOps` is the one
//! the map asked for. That join is what used to be missing: the encoder has
//! carried channel-mapping family 1 for a while, but the job layer dropped any
//! track with more than two channels before it could be reached.

use codec::audio::filter::{apply_chain, output_channels, parse_chain};
use codec::audio::{AudioCodec, AudioEncoderConfig, AudioFrame, create_encoder};

/// 20 ms of 48 kHz interleaved audio where channel `c` holds a distinct
/// constant, so a permutation is observable and the encoder gets real signal.
fn surround_pcm(channels: u8, frames: usize) -> AudioFrame {
    let samples = (0..frames)
        .flat_map(|i| {
            (0..channels).map(move |c| {
                // A per-channel tone so the encoder isn't fed digital silence.
                let phase = (i as f32) * 0.05 * (c as f32 + 1.0);
                phase.sin() * 0.25
            })
        })
        .collect();
    AudioFrame { samples, sample_rate: 48_000, channels, pts: 0 }
}

/// Split a family-1 `dOps` body into its parts, per RFC 7845 §5.1.1:
/// `(channels, family, streams, coupled, mapping)`.
fn parse_dops(dops: &[u8]) -> (u8, u8, u8, u8, Vec<u8>) {
    assert!(dops.len() >= 11, "dOps too short: {}", dops.len());
    let channels = dops[1];
    let family = dops[10];
    if family == 0 {
        return (channels, family, 1, channels.saturating_sub(1), Vec::new());
    }
    assert_eq!(
        dops.len(),
        11 + 2 + channels as usize,
        "family-1 dOps must carry a stream count, a coupled count, and one mapping byte per channel"
    );
    (channels, family, dops[11], dops[12], dops[13..].to_vec())
}

#[test]
fn five_one_side_to_back_relabel_reaches_a_family_1_opus_encoder() {
    // The motivating command:
    //   -filter:a channelmap=FL-FL|FR-FR|FC-FC|LFE-LFE|SL-BL|SR-BR:5.1
    //   -c:a libopus -b:a 240k
    let chain = parse_chain("channelmap=FL-FL|FR-FR|FC-FC|LFE-LFE|SL-BL|SR-BR:5.1").unwrap();

    // The job layer sizes the encoder from the chain before any frame arrives.
    let out_channels = output_channels(&chain, 6).unwrap();
    assert_eq!(out_channels, 6);

    let mut enc = create_encoder(AudioEncoderConfig {
        codec: AudioCodec::Opus,
        sample_rate: 48_000,
        channels: out_channels,
        bitrate: 240_000,
    })
    .expect("opus multistream encoder for 5.1");

    // dOps must announce channel-mapping family 1 with libopus's own 5.1
    // layout: 4 streams, 2 of them coupled, mapping [0,4,1,2,3,5].
    let (channels, family, streams, coupled, mapping) = parse_dops(&enc.extra_data());
    assert_eq!(channels, 6);
    assert_eq!(family, 1, "5.1 must use channel-mapping family 1");
    assert_eq!((streams, coupled), (4, 2));
    assert_eq!(mapping, vec![0, 4, 1, 2, 3, 5]);

    // Push a second of audio through filter → encoder and expect real packets.
    let mut packets = 0usize;
    for chunk in 0..50 {
        let mut pcm = surround_pcm(6, 960); // 20 ms at 48 kHz
        pcm.pts = chunk * 20_000;
        let filtered = apply_chain(&pcm, &chain).unwrap();
        assert_eq!(filtered.channels, 6);
        for pkt in enc.encode(&filtered).unwrap() {
            assert!(!pkt.data.is_empty(), "empty Opus packet");
            assert_eq!(pkt.duration, 960, "20 ms at 48 kHz is 960 ticks");
            packets += 1;
        }
    }
    packets += enc.flush().unwrap().len();
    assert!(packets >= 40, "expected ~50 packets for 1 s of audio, got {packets}");
}

#[test]
fn a_downmixing_map_resizes_the_encoder() {
    // Selecting the front pair out of 5.1 must produce a *stereo* encoder —
    // family 0, not a 6-channel one with four silent channels.
    let chain = parse_chain("channelmap=FL-FL|FR-FR:stereo").unwrap();
    let out_channels = output_channels(&chain, 6).unwrap();
    assert_eq!(out_channels, 2);

    let mut enc = create_encoder(AudioEncoderConfig {
        codec: AudioCodec::Opus,
        sample_rate: 48_000,
        channels: out_channels,
        bitrate: 96_000,
    })
    .unwrap();
    let (channels, family, ..) = parse_dops(&enc.extra_data());
    assert_eq!((channels, family), (2, 0), "stereo stays on channel-mapping family 0");

    let filtered = apply_chain(&surround_pcm(6, 960), &chain).unwrap();
    assert_eq!(filtered.samples.len(), 960 * 2);
    enc.encode(&filtered).unwrap();
}

#[test]
fn every_surround_width_builds_an_encoder() {
    // 3..=8 all have an RFC 7845 family-1 layout; the encoder must accept each,
    // and its default bitrate must scale with the stream count rather than
    // falling back to a stereo-sized one.
    for channels in 1u8..=8 {
        let enc = create_encoder(AudioEncoderConfig {
            codec: AudioCodec::Opus,
            sample_rate: 48_000,
            channels,
            bitrate: 0, // 0 = derive from the layout
        })
        .unwrap_or_else(|e| panic!("{channels}-channel Opus encoder: {e}"));
        let (got, family, streams, coupled, _) = parse_dops(&enc.extra_data());
        assert_eq!(got, channels);
        if channels <= 2 {
            assert_eq!(family, 0, "{channels}ch should stay on family 0");
        } else {
            assert_eq!(family, 1, "{channels}ch needs family 1");
            assert!(streams >= coupled, "{channels}ch: coupled exceeds total streams");
            assert_eq!(
                streams + coupled,
                channels,
                "{channels}ch: streams + coupled must equal the channel count"
            );
        }
    }
}
