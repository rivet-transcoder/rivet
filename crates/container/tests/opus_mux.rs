//! Integration tests for the Opus audio mux + demux paths (Squad-23).
//!
//! Covers:
//!   1. `with_audio` accepts AudioInfo with codec="opus" (RFC 7845 §4.4).
//!   2. `with_audio` rejects: empty/short codec_private, wrong timescale,
//!      ChannelMappingFamily != 0.
//!   3. `Opus` sample entry + `dOps` box appear in the muxed MP4 in the
//!      right nesting order, with no `mp4a` / `esds` leakage.
//!   4. Synthesised Opus packet stream → mux → demux roundtrip preserves
//!      OpusHead body (codec_private), sample byte equality, and channel /
//!      sample-rate metadata.
//!   5. Synthesised WebM-with-Opus → demux → mux roundtrip (passthrough
//!      end-to-end).
//!
//! Tests are byte-level — no real Opus decoder needed (we synthesise opaque
//! packet payloads). The mux + demux code paths are exercised end-to-end.

use bytes::Bytes;
use codec::encode::EncodedPacket;
use container::AudioInfo;
use container::demux;
use container::mux::Av1Mp4Muxer;

/// Minimal AV1 OBU_SEQUENCE_HEADER with obu_has_size_field=1. Required to
/// pass `extract_sequence_header` during finalize. (Mirrors the helper in
/// `audio_mux.rs` — kept private to each test file to avoid a shared
/// helper crate.)
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

/// OpusHead body for stereo @ 48 kHz, PreSkip=312, OutputGain=0,
/// ChannelMappingFamily=0. 11 bytes minimum per RFC 7845 §5.1.
/// LE numeric convention (the Ogg / WebM CodecPrivate form).
fn opus_head_stereo_48k() -> Vec<u8> {
    let mut head = Vec::with_capacity(11);
    head.push(1u8); // Version=1
    head.push(2u8); // OutputChannelCount=2
    head.extend_from_slice(&312u16.to_le_bytes()); // PreSkip=312 LE
    head.extend_from_slice(&48_000u32.to_le_bytes()); // InputSampleRate=48000 LE
    head.extend_from_slice(&0i16.to_le_bytes()); // OutputGain=0 LE
    head.push(0u8); // ChannelMappingFamily=0
    head
}

fn opus_info_stereo() -> AudioInfo {
    AudioInfo::opus(48_000, 2, opus_head_stereo_48k())
}

/// Synthesise an opaque Opus packet. Real Opus packets are TOC byte +
/// frame data; for byte-level mux/demux roundtrip tests, a unique opaque
/// payload is fine and lets us assert sample-byte equality.
fn opus_packet(seed: u8, size: usize) -> Vec<u8> {
    let mut p = Vec::with_capacity(size);
    p.push(seed); // pretend-TOC byte
    for i in 1..size {
        p.push(((seed as usize + i) & 0xFF) as u8);
    }
    p
}

fn push_opus_samples(muxer: &mut Av1Mp4Muxer, count: usize, packet_size: usize) -> Vec<Vec<u8>> {
    let mut emitted = Vec::with_capacity(count);
    for i in 0..count {
        let p = opus_packet((i & 0xFF) as u8, packet_size);
        emitted.push(p.clone());
        // 960-tick duration = 20 ms @ 48 kHz, the standard Opus encoder frame.
        muxer
            .add_audio_sample(&p, (i * 960) as u64, 960)
            .expect("audio sample");
    }
    emitted
}

fn find_fourcc(data: &[u8], tag: &[u8; 4]) -> Option<usize> {
    data.windows(4).position(|w| w == tag)
}

// ---- with_audio gate ---------------------------------------------------

#[test]
fn opus_with_audio_accepts_valid_stereo() {
    let mut muxer = Av1Mp4Muxer::new(320, 240, 30.0).expect("muxer");
    muxer
        .with_audio(opus_info_stereo())
        .expect("Opus stereo should be accepted");
}

#[test]
fn opus_with_audio_rejects_short_codec_private() {
    let mut muxer = Av1Mp4Muxer::new(320, 240, 30.0).expect("muxer");
    let mut info = opus_info_stereo();
    info.codec_private.truncate(5);
    let err = muxer
        .with_audio(info)
        .err()
        .expect("must reject short OpusHead");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("11 bytes"),
        "error should mention 11-byte minimum: {msg}"
    );
}

#[test]
fn opus_with_audio_rejects_wrong_timescale() {
    let mut muxer = Av1Mp4Muxer::new(320, 240, 30.0).expect("muxer");
    let mut info = opus_info_stereo();
    info.timescale = 44_100; // RFC 7845 §3 mandates 48000
    let err = muxer
        .with_audio(info)
        .err()
        .expect("must reject non-48k timescale");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("48000"),
        "error should cite RFC 7845 48000 requirement: {msg}"
    );
}

#[test]
fn opus_with_audio_rejects_channel_mapping_family_nonzero() {
    let mut muxer = Av1Mp4Muxer::new(320, 240, 30.0).expect("muxer");
    let mut head = opus_head_stereo_48k();
    head[10] = 1; // ChannelMappingFamily=1 (surround) — out of scope
    let info = AudioInfo {
        codec_private: head,
        ..opus_info_stereo()
    };
    let err = muxer
        .with_audio(info)
        .err()
        .expect("must reject family != 0");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("family"),
        "error should mention ChannelMappingFamily: {msg}"
    );
}

// ---- Sample entry presence in finalised MP4 ----------------------------

#[test]
fn opus_finalize_writes_opus_sample_entry_and_dops() {
    let mut muxer = Av1Mp4Muxer::new(320, 240, 30.0).expect("muxer");
    push_minimal_video(&mut muxer, 10);
    muxer.with_audio(opus_info_stereo()).expect("with_audio");
    push_opus_samples(&mut muxer, 8, 200);
    let out = muxer.finalize().expect("finalize");

    assert!(
        find_fourcc(&out, b"Opus").is_some(),
        "no Opus sample entry — capital O is load-bearing per RFC 7845 §4.4"
    );
    assert!(find_fourcc(&out, b"dOps").is_some(), "no dOps box");
    assert!(find_fourcc(&out, b"soun").is_some(), "no soun handler");
    assert!(find_fourcc(&out, b"smhd").is_some(), "no smhd");
    // No AAC leftovers.
    assert!(
        find_fourcc(&out, b"mp4a").is_none(),
        "Opus output must NOT include mp4a sample entry"
    );
    assert!(
        find_fourcc(&out, b"esds").is_none(),
        "Opus output must NOT include esds (esds is AAC-only here)"
    );
}

#[test]
fn opus_dops_lives_inside_opus_sample_entry() {
    let mut muxer = Av1Mp4Muxer::new(320, 240, 30.0).expect("muxer");
    push_minimal_video(&mut muxer, 10);
    muxer.with_audio(opus_info_stereo()).expect("with_audio");
    push_opus_samples(&mut muxer, 5, 120);
    let out = muxer.finalize().expect("finalize");
    let opus_pos = find_fourcc(&out, b"Opus").expect("Opus sample entry");
    // Box layout: [pos-4..pos]=size, [pos..+4]=fourcc.
    let opus_size = u32::from_be_bytes([
        out[opus_pos - 4],
        out[opus_pos - 3],
        out[opus_pos - 2],
        out[opus_pos - 1],
    ]) as usize;
    let opus_end = opus_pos - 4 + opus_size;
    let dops_pos = find_fourcc(&out, b"dOps").expect("dOps box");
    assert!(
        dops_pos > opus_pos && dops_pos < opus_end,
        "dOps must nest inside Opus: opus@{}..{} dops@{}",
        opus_pos,
        opus_end,
        dops_pos
    );
}

// ---- Roundtrip: mux + demux ------------------------------------------

#[test]
fn opus_mux_then_demux_preserves_codec_private_and_samples() {
    let mut muxer = Av1Mp4Muxer::new(320, 240, 30.0).expect("muxer");
    push_minimal_video(&mut muxer, 15);
    let info = opus_info_stereo();
    let expected_head = info.codec_private.clone();
    muxer.with_audio(info).expect("with_audio");
    let pushed = push_opus_samples(&mut muxer, 12, 180);
    let out = muxer.finalize().expect("finalize");

    let demuxed = demux::demux(&out).expect("demux roundtrip");
    let audio = demuxed.audio.expect("audio track missing after roundtrip");
    assert_eq!(audio.codec, "opus", "codec tag must round-trip as 'opus'");
    assert_eq!(audio.channels, 2);
    assert_eq!(audio.sample_rate, 48_000);
    assert_eq!(audio.timescale, 48_000);

    // codec_private MUST come back byte-identical: PreSkip + InputSampleRate
    // are translated LE→BE inside dOps then BE→LE on the demux side, so a
    // mismatch here means the byte-order conversion is broken.
    assert_eq!(
        audio.codec_private, expected_head,
        "OpusHead body must round-trip byte-identical: expected {:02X?}, got {:02X?}",
        expected_head, audio.codec_private
    );

    // Sample byte equality. Each Opus packet must come out of demux exactly
    // as it went in.
    assert_eq!(
        audio.samples.len(),
        pushed.len(),
        "sample count drifted: pushed {}, got {}",
        pushed.len(),
        audio.samples.len()
    );
    for (i, (got, expected)) in audio.samples.iter().zip(pushed.iter()).enumerate() {
        assert_eq!(got, expected, "sample {} byte mismatch", i);
    }

    // Opus path must NOT carry an ASC.
    assert!(
        audio.asc.is_empty(),
        "Opus track must have empty asc field; got {} bytes",
        audio.asc.len()
    );
}

#[test]
fn opus_mux_finalize_emits_av01_video_and_opus_audio_in_one_file() {
    // Both tracks present and discoverable. mvhd next_track_ID lifts past
    // both; trak count = 2.
    let mut muxer = Av1Mp4Muxer::new(320, 240, 30.0).expect("muxer");
    push_minimal_video(&mut muxer, 30);
    muxer.with_audio(opus_info_stereo()).expect("with_audio");
    push_opus_samples(&mut muxer, 50, 100);
    let out = muxer.finalize().expect("finalize");

    // Two trak boxes, one video (av01) and one audio (Opus).
    let trak_count = out.windows(4).filter(|w| *w == b"trak").count();
    assert_eq!(
        trak_count, 2,
        "expected 2 trak boxes (video + audio); got {}",
        trak_count
    );
    assert!(
        find_fourcc(&out, b"av01").is_some(),
        "av01 video sample entry missing"
    );
    assert!(
        find_fourcc(&out, b"Opus").is_some(),
        "Opus audio sample entry missing"
    );

    let demuxed = demux::demux(&out).expect("demux");
    assert!(demuxed.audio.is_some(), "demuxed audio track missing");
    assert_eq!(demuxed.audio.unwrap().codec, "opus");
}

// ---- WebM (synthesised) — demux + mux roundtrip ----------------------

/// Minimal WebM: EBML header + Segment[Tracks(VideoTrack + AudioTrack)
/// + Cluster + SimpleBlock + SimpleBlock + ...]. Hand-written EBML to
/// avoid pulling a webm encoder crate.
///
/// Uses fixed-known sizes and 1-byte VINT length fields (top bit set to
/// 1, low 7 bits carry length) where possible; larger fields use 4-byte
/// VINTs (0x10 prefix).
fn synth_webm_with_opus_track() -> Vec<u8> {
    // We construct individual EBML elements bottom-up so each parent's
    // size field can be computed from its already-built children.
    fn vint_4(value: u64) -> Vec<u8> {
        // 4-byte VINT: top byte is 0x10 | (value >> 24); rest are MSB.
        // Max value: 2^28 - 2.
        debug_assert!(value < (1u64 << 28));
        let mut v = Vec::with_capacity(4);
        v.push(0x10 | ((value >> 24) & 0xFF) as u8);
        v.push(((value >> 16) & 0xFF) as u8);
        v.push(((value >> 8) & 0xFF) as u8);
        v.push((value & 0xFF) as u8);
        v
    }
    fn elem_id(id: &[u8]) -> Vec<u8> {
        id.to_vec()
    }
    fn elem(id: &[u8], body: &[u8]) -> Vec<u8> {
        let mut out = elem_id(id);
        out.extend_from_slice(&vint_4(body.len() as u64));
        out.extend_from_slice(body);
        out
    }
    fn elem_u64(id: &[u8], val: u64) -> Vec<u8> {
        let bytes = val.to_be_bytes();
        // strip leading zeros but keep at least 1 byte
        let start = bytes.iter().position(|&b| b != 0).unwrap_or(7);
        let body = &bytes[start..];
        elem(id, body)
    }
    fn elem_f64(id: &[u8], val: f64) -> Vec<u8> {
        elem(id, &val.to_be_bytes())
    }
    fn elem_str(id: &[u8], s: &str) -> Vec<u8> {
        elem(id, s.as_bytes())
    }

    // EBML element IDs (matroska):
    let id_ebml = &[0x1A, 0x45, 0xDF, 0xA3];
    let id_doctype = &[0x42, 0x82];
    let id_doctype_version = &[0x42, 0x87];
    let id_doctype_read_version = &[0x42, 0x85];
    let id_segment = &[0x18, 0x53, 0x80, 0x67];
    let id_info = &[0x15, 0x49, 0xA9, 0x66];
    let id_timestamp_scale = &[0x2A, 0xD7, 0xB1];
    let id_muxing_app = &[0x4D, 0x80]; // matroska required field
    let id_writing_app = &[0x57, 0x41]; // matroska required field
    let id_duration = &[0x44, 0x89];
    let id_tracks = &[0x16, 0x54, 0xAE, 0x6B];
    let id_track_entry = &[0xAE];
    let id_track_number = &[0xD7];
    let id_track_uid = &[0x73, 0xC5];
    let id_track_type = &[0x83];
    let id_codec_id = &[0x86];
    let id_codec_private = &[0x63, 0xA2];
    let id_audio = &[0xE1];
    let id_sampling_freq = &[0xB5];
    let id_channels = &[0x9F];
    let id_video = &[0xE0];
    let id_pixel_width = &[0xB0];
    let id_pixel_height = &[0xBA];
    let id_default_duration = &[0x23, 0xE3, 0x83];
    let id_cluster = &[0x1F, 0x43, 0xB6, 0x75];
    let id_timestamp = &[0xE7];
    let id_simple_block = &[0xA3];

    // EBML header.
    let mut ebml_body = Vec::new();
    ebml_body.extend(elem_str(id_doctype, "matroska"));
    ebml_body.extend(elem_u64(id_doctype_version, 4));
    ebml_body.extend(elem_u64(id_doctype_read_version, 2));
    let ebml = elem(id_ebml, &ebml_body);

    // Tracks.
    // Video track (V_AV1 just to satisfy the demuxer's video lookup).
    let mut video_track_body = Vec::new();
    video_track_body.extend(elem_u64(id_track_number, 1));
    video_track_body.extend(elem_u64(id_track_uid, 0xCAFE));
    video_track_body.extend(elem_u64(id_track_type, 1)); // 1 = video
    video_track_body.extend(elem_str(id_codec_id, "V_AV1"));
    let mut video_subbody = Vec::new();
    video_subbody.extend(elem_u64(id_pixel_width, 320));
    video_subbody.extend(elem_u64(id_pixel_height, 240));
    video_track_body.extend(elem(id_video, &video_subbody));
    let video_track = elem(id_track_entry, &video_track_body);

    // Audio track (A_OPUS) with OpusHead in CodecPrivate.
    let mut audio_track_body = Vec::new();
    audio_track_body.extend(elem_u64(id_track_number, 2));
    audio_track_body.extend(elem_u64(id_track_uid, 0xBEEF));
    audio_track_body.extend(elem_u64(id_track_type, 2)); // 2 = audio
    audio_track_body.extend(elem_str(id_codec_id, "A_OPUS"));
    audio_track_body.extend(elem(id_codec_private, &opus_head_stereo_48k()));
    // DefaultDuration in nanoseconds: 20 ms = 20_000_000 ns (one Opus frame).
    audio_track_body.extend(elem_u64(id_default_duration, 20_000_000));
    let mut audio_subbody = Vec::new();
    audio_subbody.extend(elem_f64(id_sampling_freq, 48_000.0));
    audio_subbody.extend(elem_u64(id_channels, 2));
    audio_track_body.extend(elem(id_audio, &audio_subbody));
    let audio_track = elem(id_track_entry, &audio_track_body);

    let mut tracks_body = Vec::new();
    tracks_body.extend(video_track);
    tracks_body.extend(audio_track);
    let tracks = elem(id_tracks, &tracks_body);

    // Info (timestamp scale = 1ms = 1_000_000 ns; duration = 1s).
    // matroska-demuxer requires MuxingApp + WritingApp.
    let mut info_body = Vec::new();
    info_body.extend(elem_u64(id_timestamp_scale, 1_000_000));
    info_body.extend(elem_str(id_muxing_app, "squad-23-test"));
    info_body.extend(elem_str(id_writing_app, "squad-23-test"));
    info_body.extend(elem_f64(id_duration, 1000.0));
    let info = elem(id_info, &info_body);

    // Cluster: 5 audio SimpleBlocks at 0, 20, 40, 60, 80 ms relative to
    // the cluster's timestamp=0.
    let mut cluster_body = Vec::new();
    cluster_body.extend(elem_u64(id_timestamp, 0));
    let opus_packets: Vec<Vec<u8>> = (0..5).map(|i| opus_packet(i, 64)).collect();
    for (i, packet) in opus_packets.iter().enumerate() {
        // SimpleBlock header: track number VINT (1 byte for track 2 → 0x82),
        // i16 BE relative timestamp, 1 byte flags.
        let mut sb = Vec::with_capacity(4 + packet.len());
        sb.push(0x82); // track number 2 (VINT 1-byte form)
        sb.extend_from_slice(&((i as i16) * 20).to_be_bytes()); // ms relative
        sb.push(0x80); // flags: keyframe-ish
        sb.extend_from_slice(packet);
        cluster_body.extend(elem(id_simple_block, &sb));
    }
    let cluster = elem(id_cluster, &cluster_body);

    // Segment.
    let mut segment_body = Vec::new();
    segment_body.extend(info);
    segment_body.extend(tracks);
    segment_body.extend(cluster);
    let segment = elem(id_segment, &segment_body);

    let mut out = Vec::new();
    out.extend(ebml);
    out.extend(segment);
    out
}

#[test]
fn webm_opus_demux_extracts_codec_private_and_samples() {
    let webm = synth_webm_with_opus_track();
    let demuxed = demux::demux(&webm).expect("demux synth WebM");
    let audio = demuxed.audio.expect("WebM audio missing");
    assert_eq!(audio.codec, "opus");
    assert_eq!(audio.channels, 2);
    assert_eq!(audio.sample_rate, 48_000);
    assert_eq!(
        audio.timescale, 48_000,
        "Opus mdhd timescale must be pinned to 48000"
    );
    assert_eq!(
        audio.codec_private,
        opus_head_stereo_48k(),
        "WebM CodecPrivate must round-trip as OpusHead body verbatim"
    );
    assert_eq!(audio.samples.len(), 5, "WebM has 5 Opus SimpleBlocks");
    // Each sample must equal its synthesised packet.
    for (i, sample) in audio.samples.iter().enumerate() {
        let expected = opus_packet(i as u8, 64);
        assert_eq!(sample, &expected, "WebM sample {} mismatch", i);
    }
}

#[test]
fn webm_opus_demux_then_mux_preserves_byte_identity() {
    // WebM-with-Opus → demux → mux into MP4 → re-demux. The OpusHead
    // body and the Opus packet bytes must come through both stages
    // byte-identical (passthrough end-to-end without re-encode).
    let webm = synth_webm_with_opus_track();
    let demuxed = demux::demux(&webm).expect("WebM demux");
    let audio_in = demuxed.audio.expect("WebM audio");

    let mut muxer = Av1Mp4Muxer::new(320, 240, 30.0).expect("muxer");
    push_minimal_video(&mut muxer, 10);
    let info = AudioInfo {
        codec: audio_in.codec.clone(),
        sample_rate: audio_in.sample_rate,
        channels: audio_in.channels,
        timescale: audio_in.timescale,
        asc_bytes: audio_in.asc.clone(),
        codec_private: audio_in.codec_private.clone(),
    };
    let expected_head = info.codec_private.clone();
    let expected_samples = audio_in.samples.clone();
    muxer.with_audio(info).expect("WebM→MP4 with_audio");
    for (sample, dur) in audio_in.samples.iter().zip(audio_in.durations.iter()) {
        muxer
            .add_audio_sample(sample, 0, *dur)
            .expect("add Opus sample");
    }
    let mp4 = muxer.finalize().expect("MP4 finalize");

    let redemuxed = demux::demux(&mp4).expect("MP4 re-demux");
    let audio_out = redemuxed.audio.expect("re-demux audio missing");
    assert_eq!(
        audio_out.codec_private, expected_head,
        "OpusHead must survive WebM → MP4 → re-demux byte-identical"
    );
    assert_eq!(
        audio_out.samples, expected_samples,
        "Opus packets must survive the full passthrough byte-identical"
    );
}
