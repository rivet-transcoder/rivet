// Opus + dOps box layout, multichannel surround (family 1), Opus
// sample-entry validation, stsd dispatcher, and Apple `chan` box.
// 25 #[test] functions.

use crate::AudioInfo;
use super::super::Av1Mp4Muxer;
use super::super::audio_track::{
    build_audio_stsd, build_chan_box, build_mp4a, build_opus_sample_entry, build_dops,
};

// ---- local fixtures -------------------------------------------------------

/// Standard OpusHead body for stereo @ 48 kHz with PreSkip = 312
/// (the typical libopus encoder lookahead at 48 kHz). Output gain = 0,
/// ChannelMappingFamily = 0 (stereo).
///
/// Layout (post-magic body, 11 bytes; LE numeric fields per RFC 7845
/// §5.1):
///   [0]    Version=1
///   [1]    OutputChannelCount=2
///   [2..4] PreSkip=312 LE → 38 01
///   [4..8] InputSampleRate=48000 LE → 80 BB 00 00
///   [8..10] OutputGain=0 LE → 00 00
///   [10]   ChannelMappingFamily=0
fn opus_head_stereo_48k_preskip_312() -> Vec<u8> {
    let mut head = Vec::with_capacity(11);
    head.push(1u8); // Version
    head.push(2u8); // OutputChannelCount
    head.extend_from_slice(&312u16.to_le_bytes()); // PreSkip
    head.extend_from_slice(&48_000u32.to_le_bytes()); // InputSampleRate
    head.extend_from_slice(&0i16.to_le_bytes()); // OutputGain
    head.push(0u8); // ChannelMappingFamily
    head
}

fn opus_info_stereo_48k() -> AudioInfo {
    AudioInfo {
        codec: "opus".into(),
        sample_rate: 48_000,
        channels: 2,
        timescale: 48_000,
        asc_bytes: Vec::new(),
        codec_private: opus_head_stereo_48k_preskip_312(),
    }
}

/// Build an OpusHead body for an N-channel surround layout per
/// RFC 7845 §5.1. Layout matches what Squad-28's
/// `OpusEncoder::extra_data()` emits and what an MKV/WebM
/// `CodecPrivate` carries verbatim. All multi-byte fields LE.
fn opus_head_surround(
    channels: u8,
    pre_skip: u16,
    input_sample_rate: u32,
    streams: u8,
    coupled: u8,
    mapping: &[u8],
) -> Vec<u8> {
    assert_eq!(mapping.len(), channels as usize);
    let mut h = Vec::with_capacity(11 + 2 + channels as usize);
    h.push(1u8); // Version
    h.push(channels);
    h.extend_from_slice(&pre_skip.to_le_bytes());
    h.extend_from_slice(&input_sample_rate.to_le_bytes());
    h.extend_from_slice(&0i16.to_le_bytes()); // OutputGain
    h.push(1u8); // ChannelMappingFamily=1
    h.push(streams);
    h.push(coupled);
    h.extend_from_slice(mapping);
    h
}

fn opus_info_5_1() -> AudioInfo {
    // RFC 7845 §5.1.1.2 5.1 layout: streams=4, coupled=2,
    // mapping = [0, 4, 1, 2, 3, 5]. PreSkip=312 (typical libopus
    // lookahead).
    let cp = opus_head_surround(6, 312, 48_000, 4, 2, &[0, 4, 1, 2, 3, 5]);
    AudioInfo {
        codec: "opus".into(),
        sample_rate: 48_000,
        channels: 6,
        timescale: 48_000,
        asc_bytes: Vec::new(),
        codec_private: cp,
    }
}

// ---- Squad-23: Opus + dOps box layout (RFC 7845) -------------------------

/// `dOps` body layout per RFC 7845 §4.5: 11-byte minimum. Box wrapper
/// adds 8-byte ISOBMFF header → total 19 bytes for ChannelMappingFamily=0.
/// Numeric fields are big-endian (NOT the little-endian convention of
/// the OpusHead source bytes).
#[test]
fn dops_box_11_byte_payload_layout() {
    let info = opus_info_stereo_48k();
    let dops = build_dops(&info);
    assert_eq!(
        dops.len(),
        19,
        "dOps must be exactly 19 bytes (8 header + 11 payload)"
    );
    let size = u32::from_be_bytes([dops[0], dops[1], dops[2], dops[3]]) as usize;
    assert_eq!(size, dops.len(), "size field must equal box length");
    assert_eq!(
        &dops[4..8],
        b"dOps",
        "box type must be 'dOps' (capital O lowercase ps)"
    );
    // Body fields, all BE per §4.5.
    assert_eq!(dops[8], 0, "Version (RFC 7845 §4.5: MUST be 0)");
    assert_eq!(dops[9], 2, "OutputChannelCount = stereo");
    let pre_skip = u16::from_be_bytes([dops[10], dops[11]]);
    assert_eq!(pre_skip, 312, "PreSkip = 312 (BE)");
    let input_sample_rate = u32::from_be_bytes([dops[12], dops[13], dops[14], dops[15]]);
    assert_eq!(input_sample_rate, 48_000, "InputSampleRate = 48000 (BE)");
    let output_gain = i16::from_be_bytes([dops[16], dops[17]]);
    assert_eq!(output_gain, 0, "OutputGain = 0 (Q8 dB, BE)");
    assert_eq!(dops[18], 0, "ChannelMappingFamily = 0 (mono/stereo)");
}

/// The byte-order conversion between OpusHead (LE) and dOps (BE) is
/// the load-bearing piece — easy to mess up. PreSkip=312 in LE is
/// `38 01`; in BE it must come back out as `01 38`.
#[test]
fn dops_byte_order_flipped_from_opushead() {
    let info = opus_info_stereo_48k();
    // Sanity check the input is in LE.
    assert_eq!(
        info.codec_private[2..4],
        [0x38, 0x01],
        "OpusHead PreSkip must be LE"
    );
    let dops = build_dops(&info);
    // PreSkip in dOps body = bytes 10..12 of the box (after 8-byte header).
    assert_eq!(
        dops[10..12],
        [0x01, 0x38],
        "dOps PreSkip must be BE — got {:02X?}",
        &dops[10..12]
    );
}

/// `Opus` sample entry per RFC 7845 §4.4. Same generic AudioSampleEntry
/// preamble as `mp4a` (36 bytes including header) plus the dOps child.
/// Total = 36 + 19 = 55 bytes for the minimum-channel-count case.
/// 4-cc is `Opus` exactly (capital O).
#[test]
fn opus_sample_entry_size_and_fourcc() {
    let info = opus_info_stereo_48k();
    let entry = build_opus_sample_entry(&info);
    let size = u32::from_be_bytes([entry[0], entry[1], entry[2], entry[3]]) as usize;
    assert_eq!(size, entry.len(), "size field must equal box length");
    assert_eq!(&entry[4..8], b"Opus", "4-cc MUST be 'Opus' (capital O)");
    assert_ne!(&entry[4..8], b"opus", "lowercase 'opus' is non-conformant");
    // Total = 36 (sample entry preamble inc 8-byte header) + 19 (dOps) = 55.
    assert_eq!(
        entry.len(),
        55,
        "Opus sample entry should be 55 bytes for stereo + dOps minimum"
    );
}

/// AudioSampleEntry-level samplerate field inside `Opus` MUST be
/// 48000 << 16 — RFC 7845 §3 mandates 48 kHz internally; emitting
/// the source's nominal rate (e.g. 44100) would mismatch dOps and
/// confuse strict validators.
#[test]
fn opus_sample_entry_samplerate_is_48000_q16() {
    let info = AudioInfo {
        // Source nominal sample_rate is 44100, but the sample-entry
        // and mdhd MUST report 48000.
        sample_rate: 44_100,
        ..opus_info_stereo_48k()
    };
    let entry = build_opus_sample_entry(&info);
    // Layout offsets inside the sample entry (after the 8-byte box header):
    //   reserved[6]+data_ref(2)=8, reserved2(8)=16, channelcount(2)=18,
    //   sample_size(2)=20, pre_def(2)=22, reserved3(2)=24,
    //   samplerate u32 16.16 at +24..+28.
    // So box-relative offset 8 + 24 = 32.
    let sr_q16 = u32::from_be_bytes([entry[32], entry[33], entry[34], entry[35]]);
    assert_eq!(
        sr_q16,
        48_000u32 << 16,
        "samplerate field MUST be 48000<<16 (Q16); got 0x{:08X}",
        sr_q16
    );
}

/// `dOps` must nest inside the `Opus` sample entry. The build_audio_stsd
/// dispatcher routes Opus → build_opus_sample_entry → dOps child.
#[test]
fn dops_nests_inside_opus_sample_entry() {
    let info = opus_info_stereo_48k();
    let entry = build_opus_sample_entry(&info);
    let dops_pos = entry
        .windows(4)
        .position(|w| w == b"dOps")
        .expect("dOps child missing inside Opus sample entry");
    // dOps must come AFTER the 36-byte AudioSampleEntry preamble.
    assert!(
        dops_pos > 28,
        "dOps must come after the AudioSampleEntry preamble; got pos={}",
        dops_pos
    );
}

/// stsd dispatcher: AAC info → mp4a; Opus info → Opus. The dispatcher
/// must NEVER produce mp4a for Opus or Opus for AAC.
#[test]
fn stsd_dispatcher_routes_codec_to_correct_sample_entry() {
    let aac = AudioInfo {
        codec: "aac".into(),
        sample_rate: 44_100,
        channels: 2,
        timescale: 44_100,
        asc_bytes: vec![0x12, 0x10],
        codec_private: Vec::new(),
    };
    let stsd_aac = build_audio_stsd(&aac);
    assert!(
        stsd_aac.windows(4).any(|w| w == b"mp4a"),
        "AAC stsd must contain mp4a"
    );
    assert!(
        !stsd_aac.windows(4).any(|w| w == b"Opus"),
        "AAC stsd must NOT contain Opus"
    );
    assert!(
        stsd_aac.windows(4).any(|w| w == b"esds"),
        "AAC stsd must contain esds"
    );

    let opus = opus_info_stereo_48k();
    let stsd_opus = build_audio_stsd(&opus);
    assert!(
        stsd_opus.windows(4).any(|w| w == b"Opus"),
        "Opus stsd must contain Opus"
    );
    assert!(
        !stsd_opus.windows(4).any(|w| w == b"mp4a"),
        "Opus stsd must NOT contain mp4a"
    );
    assert!(
        stsd_opus.windows(4).any(|w| w == b"dOps"),
        "Opus stsd must contain dOps"
    );
    assert!(
        !stsd_opus.windows(4).any(|w| w == b"esds"),
        "Opus stsd must NOT contain esds"
    );
}

/// Negative output gain (-3 dB Q8 = -768) round-trips correctly through
/// the i16-as-u16 BE conversion.
#[test]
fn dops_handles_negative_output_gain() {
    let mut head = opus_head_stereo_48k_preskip_312();
    // OutputGain at offset 8..10. Set to -768 (i.e. -3 dB Q8).
    let gain: i16 = -768;
    head[8..10].copy_from_slice(&gain.to_le_bytes());
    let info = AudioInfo {
        codec_private: head,
        ..opus_info_stereo_48k()
    };
    let dops = build_dops(&info);
    let recovered = i16::from_be_bytes([dops[16], dops[17]]);
    assert_eq!(
        recovered, -768,
        "negative OutputGain must survive LE→BE roundtrip"
    );
}

/// PreSkip from the encoder's actual `OPUS_GET_LOOKAHEAD` (often
/// non-default like 156, 312, 480) must round-trip verbatim — we
/// don't normalize to 312.
#[test]
fn dops_preserves_arbitrary_preskip() {
    for &expected in &[0u16, 156, 312, 480, 1024, 65535] {
        let mut head = opus_head_stereo_48k_preskip_312();
        head[2..4].copy_from_slice(&expected.to_le_bytes());
        let info = AudioInfo {
            codec_private: head,
            ..opus_info_stereo_48k()
        };
        let dops = build_dops(&info);
        let got = u16::from_be_bytes([dops[10], dops[11]]);
        assert_eq!(got, expected, "PreSkip {} must survive LE→BE", expected);
    }
}

// ---- Squad-28: multichannel Opus dOps family=1 ---------------------------

/// 5.1 dOps box payload = 11 + 2 + 6 = 19 bytes; with the 8-byte
/// box header the total is 27 bytes. All numeric fields BE inside
/// the box; the trailing channel-mapping bytes are u8 each so no
/// endianness conversion needed.
#[test]
fn dops_box_5_1_payload_is_19_bytes_total_27() {
    let info = opus_info_5_1();
    let dops = build_dops(&info);
    assert_eq!(
        dops.len(),
        27,
        "5.1 dOps box = 8 header + 19 payload = 27 bytes; got {}",
        dops.len()
    );
    let size = u32::from_be_bytes([dops[0], dops[1], dops[2], dops[3]]) as usize;
    assert_eq!(size, dops.len());
    assert_eq!(&dops[4..8], b"dOps");
    // Body
    assert_eq!(dops[8], 0, "Version");
    assert_eq!(dops[9], 6, "OutputChannelCount = 6 for 5.1");
    let pre_skip = u16::from_be_bytes([dops[10], dops[11]]);
    assert_eq!(pre_skip, 312);
    let isr = u32::from_be_bytes([dops[12], dops[13], dops[14], dops[15]]);
    assert_eq!(isr, 48_000);
    assert_eq!(i16::from_be_bytes([dops[16], dops[17]]), 0);
    assert_eq!(dops[18], 1, "ChannelMappingFamily = 1 for surround");
    assert_eq!(dops[19], 4, "StreamCount = 4 for 5.1");
    assert_eq!(dops[20], 2, "CoupledCount = 2 for 5.1");
    assert_eq!(
        &dops[21..27],
        &[0u8, 4, 1, 2, 3, 5][..],
        "ChannelMapping for 5.1"
    );
}

/// 7.1 layout: streams=5, coupled=3, mapping = [0, 6, 1, 2, 3, 4, 5, 7].
/// dOps box = 8 header + 11 preamble + 2 stream/coupled + 8 mapping = 29 bytes.
#[test]
fn dops_box_7_1_payload_is_21_bytes_total_29() {
    let cp = opus_head_surround(8, 312, 48_000, 5, 3, &[0, 6, 1, 2, 3, 4, 5, 7]);
    let info = AudioInfo {
        codec: "opus".into(),
        sample_rate: 48_000,
        channels: 8,
        timescale: 48_000,
        asc_bytes: Vec::new(),
        codec_private: cp,
    };
    let dops = build_dops(&info);
    assert_eq!(dops.len(), 29);
    assert_eq!(dops[18], 1, "Family = 1");
    assert_eq!(dops[19], 5, "StreamCount = 5 for 7.1");
    assert_eq!(dops[20], 3, "CoupledCount = 3 for 7.1");
    assert_eq!(&dops[21..29], &[0u8, 6, 1, 2, 3, 4, 5, 7][..]);
}

/// Hex-dump the 5.1 dOps box for the deliverables report.
#[test]
fn dops_box_5_1_hex_dump() {
    let info = opus_info_5_1();
    let dops = build_dops(&info);
    let hex: String = dops.iter().map(|b| format!("{b:02x} ")).collect();
    println!("5.1 dOps box hex (27 bytes total): {}", hex.trim_end());
}

/// `Opus` sample entry containing a family-1 dOps for 5.1. Total
/// size = 36 (sample-entry preamble) + 27 (5.1 dOps) = 63 bytes.
#[test]
fn opus_sample_entry_5_1_size_and_dops_nesting() {
    let info = opus_info_5_1();
    let entry = build_opus_sample_entry(&info);
    assert_eq!(
        entry.len(),
        36 + 27,
        "Opus sample entry for 5.1 = 36 + 27 = 63 bytes; got {}",
        entry.len()
    );
    // Sample-entry channel_count field is at offset 24 inside the
    // sample entry (after 8-byte box header + 6 reserved + 2 dri +
    // 8 reserved = 24).
    let entry_channels = u16::from_be_bytes([entry[24], entry[25]]);
    assert_eq!(
        entry_channels, 6,
        "channel_count in AudioSampleEntry must reflect 5.1"
    );
    // The dOps child should appear after the 36-byte preamble.
    assert!(entry[36..].windows(4).any(|w| w == b"dOps"));
    // Family byte inside the dOps child = entry[36 + 8 + 10] = entry[54].
    // (8-byte dOps box header + 11-byte preamble offset 10 = family).
    assert_eq!(
        entry[36 + 8 + 10],
        1,
        "dOps inside Opus sample entry must carry family=1 for 5.1"
    );
}

/// `with_audio()` family=1 validation: stream count + coupled +
/// mapping must all be sane. Each negative case below is rejected
/// loudly with a clear error message.
#[test]
fn with_audio_rejects_family_1_with_truncated_codec_private() {
    let mut muxer = Av1Mp4Muxer::new(640, 480, 30.0).unwrap();
    let mut info = opus_info_5_1();
    // Truncate so the channel-mapping table is missing.
    info.codec_private.truncate(13); // header + 2 stream/coupled, no mapping
    let err = match muxer.with_audio(info) {
        Ok(_) => panic!("truncated family=1 codec_private must reject"),
        Err(e) => e,
    };
    let msg = format!("{}", err);
    assert!(
        msg.contains("≥") && msg.contains("preamble"),
        "error message must explain the size requirement; got: {msg}"
    );
}

#[test]
fn with_audio_rejects_family_1_with_zero_streams() {
    let mut muxer = Av1Mp4Muxer::new(640, 480, 30.0).unwrap();
    let mut info = opus_info_5_1();
    // Zero out StreamCount byte (offset 11).
    info.codec_private[11] = 0;
    let r = muxer.with_audio(info);
    assert!(r.is_err(), "StreamCount = 0 must reject");
}

#[test]
fn with_audio_rejects_family_1_with_coupled_exceeding_streams() {
    let mut muxer = Av1Mp4Muxer::new(640, 480, 30.0).unwrap();
    let mut info = opus_info_5_1();
    // Make CoupledCount > StreamCount (offset 12 vs 11).
    info.codec_private[11] = 2;
    info.codec_private[12] = 5;
    let r = muxer.with_audio(info);
    assert!(r.is_err(), "CoupledCount > StreamCount must reject");
}

#[test]
fn with_audio_rejects_family_1_with_mapping_index_out_of_range() {
    let mut muxer = Av1Mp4Muxer::new(640, 480, 30.0).unwrap();
    let mut info = opus_info_5_1();
    // Streams=4, coupled=2 → max valid mapping index = 5. Set first
    // mapping byte to 99 to force the out-of-range branch.
    info.codec_private[13] = 99;
    let r = muxer.with_audio(info);
    assert!(r.is_err(), "ChannelMapping out-of-range must reject");
}

#[test]
fn with_audio_rejects_family_0_with_5_1_channels() {
    let mut muxer = Av1Mp4Muxer::new(640, 480, 30.0).unwrap();
    // Build a hand-crafted family-0 head but claim 6 channels.
    // Family 0 only supports 1..=2 channels per RFC 7845 §5.1.1.
    let mut head = Vec::with_capacity(11);
    head.push(1u8);
    head.push(6u8);
    head.extend_from_slice(&312u16.to_le_bytes());
    head.extend_from_slice(&48_000u32.to_le_bytes());
    head.extend_from_slice(&0i16.to_le_bytes());
    head.push(0u8); // family=0
    let info = AudioInfo {
        codec: "opus".into(),
        sample_rate: 48_000,
        channels: 6,
        timescale: 48_000,
        asc_bytes: Vec::new(),
        codec_private: head,
    };
    let r = muxer.with_audio(info);
    assert!(r.is_err(), "family=0 + 6 channels must reject");
}

#[test]
fn with_audio_accepts_5_1_opus() {
    let mut muxer = Av1Mp4Muxer::new(640, 480, 30.0).unwrap();
    let info = opus_info_5_1();
    muxer
        .with_audio(info)
        .expect("5.1 Opus with valid family=1 trailer must accept");
}

#[test]
fn with_audio_rejects_9_channel_opus() {
    let mut muxer = Av1Mp4Muxer::new(640, 480, 30.0).unwrap();
    // 9 channels has no defined family-1 layout.
    let mut head = Vec::with_capacity(11 + 2 + 9);
    head.push(1u8);
    head.push(9u8);
    head.extend_from_slice(&312u16.to_le_bytes());
    head.extend_from_slice(&48_000u32.to_le_bytes());
    head.extend_from_slice(&0i16.to_le_bytes());
    head.push(1u8); // family=1
    head.push(5);
    head.push(3);
    head.extend_from_slice(&[0u8, 1, 2, 3, 4, 5, 6, 7, 0]);
    let info = AudioInfo {
        codec: "opus".into(),
        sample_rate: 48_000,
        channels: 9,
        timescale: 48_000,
        asc_bytes: Vec::new(),
        codec_private: head,
    };
    let r = muxer.with_audio(info);
    assert!(
        r.is_err(),
        "9-channel Opus must reject (no family-1 layout above 8)"
    );
}

// ---- Squad-25: Apple `chan` (Channel Layout) box -------------------------

/// Mono / stereo: no `chan` box (Apple's default layouts are correct).
#[test]
fn chan_box_omitted_for_mono_and_stereo() {
    assert!(build_chan_box(1).is_none(), "mono should not emit chan");
    assert!(build_chan_box(2).is_none(), "stereo should not emit chan");
}

/// Unsupported channel counts return None — defence-in-depth (the
/// caller's `with_audio` gate already rejects them, so seeing 9+/Atmos
/// here means a code path bypassed that gate). 8 is 7.1, see
/// `aac_eight_channels_is_seven_point_one`.
#[test]
fn chan_box_omitted_for_unsupported_counts() {
    for &c in &[0u16, 3, 4, 5, 9, 16] {
        assert!(
            build_chan_box(c).is_none(),
            "channels={c} must not emit chan"
        );
    }
}

/// 5.1 → kAudioChannelLayoutTag_MPEG_5_1_C = (114 << 16) | 6 = 0x00720006.
/// Body layout: tag u32 (4) | bitmap u32 (4) | num_descriptions u32 (4)
/// = 12 bytes. Total box = 8-byte header + 12-byte body = 20 bytes.
#[test]
fn chan_box_5_1_layout_and_size() {
    let chan = build_chan_box(6).expect("5.1 must emit chan");
    assert_eq!(
        chan.len(),
        20,
        "5.1 chan box must be 20 bytes (8 header + 12 body)"
    );
    let size = u32::from_be_bytes([chan[0], chan[1], chan[2], chan[3]]);
    assert_eq!(
        size as usize,
        chan.len(),
        "size field must equal box length"
    );
    assert_eq!(&chan[4..8], b"chan", "fourcc must be 'chan'");
    let tag = u32::from_be_bytes([chan[8], chan[9], chan[10], chan[11]]);
    assert_eq!(
        tag, 0x00720006u32,
        "5.1 tag must be kAudioChannelLayoutTag_MPEG_5_1_C = 0x00720006; got 0x{tag:08X}"
    );
    let bitmap = u32::from_be_bytes([chan[12], chan[13], chan[14], chan[15]]);
    assert_eq!(bitmap, 0, "mChannelBitmap must be 0 for tag form");
    let ndescs = u32::from_be_bytes([chan[16], chan[17], chan[18], chan[19]]);
    assert_eq!(
        ndescs, 0,
        "mNumberChannelDescriptions must be 0 for tag form"
    );
}

/// 7.1 → kAudioChannelLayoutTag_MPEG_7_1_C = (127 << 16) | 8 = 0x007F0008.
#[test]
fn chan_box_7_1_layout_and_size() {
    let chan = build_chan_box(7).expect("7.1 must emit chan");
    assert_eq!(chan.len(), 20);
    let tag = u32::from_be_bytes([chan[8], chan[9], chan[10], chan[11]]);
    assert_eq!(
        tag, 0x007F0008u32,
        "7.1 tag must be kAudioChannelLayoutTag_MPEG_7_1_C = 0x007F0008; got 0x{tag:08X}"
    );
}

/// `chan` nests inside the `mp4a` AudioSampleEntry (alongside `esds`)
/// per QuickTime File Format Spec. Multichannel mp4a should contain
/// both an esds AND a chan child.
#[test]
fn chan_nests_inside_mp4a_for_5_1() {
    // 5.1 ASC: AOT=2 SFI=3 chan=6 → 0x11 0xB0.
    let info = AudioInfo {
        codec: "aac".into(),
        sample_rate: 48_000,
        channels: 6,
        timescale: 48_000,
        asc_bytes: vec![0x11, 0xB0],
        codec_private: Vec::new(),
    };
    let mp4a = build_mp4a(&info);
    assert_eq!(&mp4a[4..8], b"mp4a", "outer box must be mp4a");
    let chan_pos = mp4a
        .windows(4)
        .position(|w| w == b"chan")
        .expect("multichannel mp4a must contain chan child");
    let esds_pos = mp4a
        .windows(4)
        .position(|w| w == b"esds")
        .expect("mp4a must always contain esds child");
    // chan should come AFTER esds (we append chan last in build_mp4a).
    assert!(
        chan_pos > esds_pos,
        "chan should come after esds in mp4a (esds @ {}, chan @ {})",
        esds_pos,
        chan_pos
    );
}

/// Stereo mp4a must NOT carry a `chan` box — Apple's default L+R
/// stereo layout is correct without one, and emitting a stereo `chan`
/// would just bloat the output.
#[test]
fn chan_absent_from_stereo_mp4a() {
    let info = AudioInfo {
        codec: "aac".into(),
        sample_rate: 48_000,
        channels: 2,
        timescale: 48_000,
        asc_bytes: vec![0x11, 0x90],
        codec_private: Vec::new(),
    };
    let mp4a = build_mp4a(&info);
    assert!(
        mp4a.windows(4).all(|w| w != b"chan"),
        "stereo mp4a must not contain a chan box"
    );
}

/// 7.1 AAC is eight channels (Table 1.19 channelConfiguration 7, or a
/// PCE-described layout with channelConfiguration 0): the gate accepts 8 and
/// the Apple `chan` box carries the MPEG_7_1_C tag, same as the older `7`.
#[test]
fn aac_eight_channels_is_seven_point_one() {
    use crate::mux::audio_track::build_chan_box;
    let eight = build_chan_box(8).expect("8-channel AAC gets a chan box");
    let seven = build_chan_box(7).expect("legacy spelling still maps");
    assert_eq!(eight, seven);
    assert_eq!(&eight[8..12], &0x007F_0008u32.to_be_bytes());
    assert!(build_chan_box(9).is_none());
}
