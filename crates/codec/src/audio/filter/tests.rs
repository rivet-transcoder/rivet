//! Tests for the audio filter module — they drive the public [`apply`] /
//! [`parse_chain`] / [`output_channels`] surface.

use super::*;
use ChannelLabel::*;

/// An `frames`-long interleaved frame where channel `c` carries the constant
/// `c + 1` — so a remap is visible as a permutation of small integers.
fn ident_frame(channels: u8, frames: usize) -> AudioFrame {
    let samples = (0..frames)
        .flat_map(|_| (0..channels).map(|c| (c + 1) as f32))
        .collect();
    AudioFrame { samples, sample_rate: 48_000, channels, pts: 0 }
}

/// The per-channel constants a filtered frame carries, one entry per output
/// channel (asserting every frame agrees).
fn channel_values(f: &AudioFrame) -> Vec<f32> {
    let ch = f.channels as usize;
    let frames = f.samples.len() / ch;
    let first: Vec<f32> = f.samples[..ch].to_vec();
    for i in 1..frames {
        assert_eq!(&f.samples[i * ch..(i + 1) * ch], first.as_slice(), "frame {i} differs");
    }
    first
}

// ── layouts ─────────────────────────────────────────────────────────────────

#[test]
fn default_layouts_match_rfc7845_order() {
    // These are the orders `audio::encode::opus`'s multistream table assumes.
    // If they drift, a 5.1 encode silently swaps speakers.
    assert_eq!(ChannelLayout::default_for(1).unwrap().labels(), &[FC]);
    assert_eq!(ChannelLayout::default_for(2).unwrap().labels(), &[FL, FR]);
    assert_eq!(ChannelLayout::default_for(6).unwrap().labels(), &[FL, FR, FC, LFE, BL, BR]);
    assert_eq!(
        ChannelLayout::default_for(8).unwrap().labels(),
        &[FL, FR, FC, LFE, BL, BR, SL, SR]
    );
    assert!(ChannelLayout::default_for(9).is_err());
    assert!(ChannelLayout::default_for(0).is_err());
}

#[test]
fn layout_parses_and_displays() {
    assert_eq!("5.1".parse::<ChannelLayout>().unwrap().labels(), &[FL, FR, FC, LFE, BL, BR]);
    assert_eq!("5.1(side)".parse::<ChannelLayout>().unwrap().labels(), &[FL, FR, FC, LFE, SL, SR]);
    // A bare count is the default layout for that count.
    assert_eq!("6".parse::<ChannelLayout>().unwrap(), "5.1".parse().unwrap());
    // The explicit form round-trips, and a named layout renders by name.
    assert_eq!("FL+FR".parse::<ChannelLayout>().unwrap().to_string(), "stereo");
    assert_eq!("FC+FL".parse::<ChannelLayout>().unwrap().to_string(), "FC+FL");
    assert_eq!("FC+FL".parse::<ChannelLayout>().unwrap(), "FC+FL".parse().unwrap());
    assert!("FL+FL".parse::<ChannelLayout>().is_err(), "duplicate channel");
    assert!("nonsense".parse::<ChannelLayout>().is_err());
}

// ── parsing ─────────────────────────────────────────────────────────────────

#[test]
fn channelmap_parses_the_ffmpeg_spelling() {
    let c = parse_chain("channelmap=FL-FL|FR-FR|FC-FC|LFE-LFE|SL-BL|SR-BR:5.1").unwrap();
    assert_eq!(
        c[0],
        AudioFilter::ChannelMap {
            pairs: vec![(FL, FL), (FR, FR), (FC, FC), (LFE, LFE), (SL, BL), (SR, BR)],
            layout: Some("5.1".parse().unwrap()),
        }
    );
    // …and round-trips through its textual form.
    assert_eq!(chain_to_string(&c), "channelmap=FL-FL|FR-FR|FC-FC|LFE-LFE|SL-BL|SR-BR:5.1");
    assert_eq!(parse_chain(&chain_to_string(&c)).unwrap(), c);
}

#[test]
fn channelmap_parses_positional_entries() {
    // Positional form: the i-th entry names the input feeding output slot i,
    // so the output position comes from the declared layout.
    let c = parse_chain("channelmap=FR|FL:stereo").unwrap();
    assert_eq!(
        c[0],
        AudioFilter::ChannelMap {
            pairs: vec![(FR, FL), (FL, FR)],
            layout: Some("stereo".parse().unwrap()),
        }
    );
}

#[test]
fn channelmap_rejects_incoherent_maps() {
    assert!(parse_chain("channelmap=").is_err(), "no map");
    assert!(parse_chain("channelmap=FL-FL|FR-FL").is_err(), "output written twice");
    assert!(parse_chain("channelmap=XX-FL").is_err(), "unknown channel");
    assert!(parse_chain("channelmap=FL|FR").is_err(), "positional without a layout");
    assert!(parse_chain("bogus=1").is_err(), "unknown filter");
    assert!(parse_chain("").is_err(), "empty chain");
}

// ── applying ────────────────────────────────────────────────────────────────

#[test]
fn side_to_back_relabel_is_a_passthrough_permutation() {
    // The motivating case: a 5.1 source whose surrounds are labelled *side* gets
    // re-tagged to the *back* positions MP4/Opus expect. Both layouts put the
    // surrounds in slots 4/5, so the samples must come out untouched.
    let chain = parse_chain("channelmap=FL-FL|FR-FR|FC-FC|LFE-LFE|SL-BL|SR-BR:5.1").unwrap();
    let out = apply_chain(&ident_frame(6, 4), &chain).unwrap();
    assert_eq!(out.channels, 6);
    assert_eq!(channel_values(&out), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!(output_channels(&chain, 6).unwrap(), 6);
}

#[test]
fn channelmap_actually_permutes() {
    // Swap the front pair: output FL must carry what input FR held.
    let chain = parse_chain("channelmap=FR-FL|FL-FR:stereo").unwrap();
    let out = apply_chain(&ident_frame(2, 3), &chain).unwrap();
    assert_eq!(channel_values(&out), vec![2.0, 1.0]);
}

#[test]
fn channelmap_can_select_a_subset() {
    // 5.1 → stereo by picking the front pair. Channel count follows the map.
    let chain = parse_chain("channelmap=FL-FL|FR-FR:stereo").unwrap();
    let out = apply_chain(&ident_frame(6, 2), &chain).unwrap();
    assert_eq!(out.channels, 2);
    assert_eq!(channel_values(&out), vec![1.0, 2.0]);
    assert_eq!(output_channels(&chain, 6).unwrap(), 2);
}

#[test]
fn unmapped_output_channels_are_silent() {
    // The layout asks for 5.1 but only the fronts are fed — the rest must be
    // silence, not a neighbouring channel's samples.
    let chain = parse_chain("channelmap=FL-FL|FR-FR:5.1").unwrap();
    let out = apply_chain(&ident_frame(2, 2), &chain).unwrap();
    assert_eq!(out.channels, 6);
    assert_eq!(channel_values(&out), vec![1.0, 2.0, 0.0, 0.0, 0.0, 0.0]);
}

#[test]
fn layout_is_inferred_from_the_pairs_when_omitted() {
    let chain = parse_chain("channelmap=FL-FL|FR-FR").unwrap();
    assert_eq!(output_channels(&chain, 6).unwrap(), 2);
    assert_eq!(channel_values(&apply_chain(&ident_frame(6, 2), &chain).unwrap()), vec![1.0, 2.0]);
}

#[test]
fn reading_a_channel_the_input_lacks_is_an_error() {
    // Catching this when the encoder is configured beats discovering it on the
    // first frame — `output_channels` is what the job layer calls up front.
    let chain = parse_chain("channelmap=FL-FL|FR-FR|BL-BL|BR-BR:quad").unwrap();
    let err = output_channels(&chain, 2).unwrap_err().to_string();
    assert!(err.contains("BL"), "unhelpful error: {err}");
    assert!(apply_chain(&ident_frame(2, 2), &chain).is_err());
}

#[test]
fn input_layout_is_inferred_from_the_channels_the_map_reads() {
    // 6 channels is `5.1` (back surrounds) or `5.1(side)`, and the container
    // usually only says "6". A map that reads SL/SR can only mean the latter,
    // so it resolves rather than erroring — this is what makes the motivating
    // `SL-BL|SR-BR` command work on a plain 6-channel track.
    let side = parse_chain("channelmap=FL-FL|FR-FR|FC-FC|LFE-LFE|SL-BL|SR-BR:5.1").unwrap();
    assert_eq!(output_channels(&side, 6).unwrap(), 6);

    // Reading BL/BR on 6 channels still resolves the other way, to plain 5.1.
    let back = parse_chain("channelmap=BL-BL|BR-BR").unwrap();
    assert_eq!(output_channels(&back, 6).unwrap(), 2);

    // Only channels that exist somewhere in a 6-channel layout are accepted.
    let bogus = parse_chain("channelmap=BC-FC").unwrap();
    let err = output_channels(&bogus, 6).unwrap_err().to_string();
    assert!(err.contains("BC"), "unhelpful error: {err}");
}

#[test]
fn inference_does_not_mix_two_layouts() {
    // Reading both SL and BL means no single 6-channel layout fits; better to
    // say so than to silently invent a mapping.
    let chain = parse_chain("channelmap=SL-FL|BL-FR").unwrap();
    assert!(output_channels(&chain, 6).is_err());
}

#[test]
fn frame_metadata_survives_the_filter() {
    let chain = parse_chain("channelmap=FR-FL|FL-FR:stereo").unwrap();
    let mut src = ident_frame(2, 5);
    src.sample_rate = 44_100;
    src.pts = 12_345;
    let out = apply_chain(&src, &chain).unwrap();
    assert_eq!(out.sample_rate, 44_100);
    assert_eq!(out.pts, 12_345);
    assert_eq!(out.samples.len(), 10);
}
