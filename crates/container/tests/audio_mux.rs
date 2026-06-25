//! Integration tests for the AAC audio trak path in `Av1Mp4Muxer` (#63).
//!
//! Covers:
//!   1. Accepting / rejecting AudioInfo based on codec / channel count.
//!   2. `mp4a` sample entry and `esds` descriptor presence.
//!   3. ASC verbatim byte preservation inside esds's DecoderSpecificInfo.
//!   4. `write_descriptor_length` encoding for both < 128 and >= 128 cases.
//!   5. Chunk-offset planning keeps video + audio offsets consistent with
//!      the mdat interleave layout the muxer emits.
//!   6. End-to-end roundtrip: mux video+audio, re-demux, assert the ASC
//!      and sample count survive.

use bytes::Bytes;
use codec::encode::EncodedPacket;
use container::AudioInfo;
use container::demux;
use container::mux::Av1Mp4Muxer;

/// Minimal AV1 OBU_SEQUENCE_HEADER with obu_has_size_field=1. Required to
/// pass `extract_sequence_header` during finalize.
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

/// ASC bytes for AAC-LC stereo @ 44100 Hz:
///   audioObjectType = 2 (5 bits: 00010)
///   samplingFrequencyIndex = 4 (4 bits: 0100)  -> 44100
///   channelConfiguration = 2 (4 bits: 0010)
///   => 00010 0100 0010 000 = 0001 0010 0001 0000 = 0x12 0x10
fn aac_lc_stereo_asc() -> Vec<u8> {
    vec![0x12, 0x10]
}

fn aac_lc_stereo_48k_asc() -> Vec<u8> {
    // AOT=2 (00010), SFI=3 (0011) -> 48000, chan=2 (0010)
    // 00010 0011 0010 000 = 0001 0001 1001 0000 = 0x11 0x90
    vec![0x11, 0x90]
}

fn aac_info_stereo_44100() -> AudioInfo {
    AudioInfo {
        codec: "aac".into(),
        sample_rate: 44100,
        channels: 2,
        timescale: 44100,
        asc_bytes: aac_lc_stereo_asc(),
        codec_private: Vec::new(),
    }
}

fn find_fourcc(data: &[u8], tag: &[u8; 4]) -> Option<usize> {
    data.windows(4).position(|w| w == tag)
}

fn find_all_fourcc(data: &[u8], tag: &[u8; 4]) -> Vec<usize> {
    let mut hits = Vec::new();
    let mut pos = 0;
    while let Some(rel) = data[pos..].windows(4).position(|w| w == tag) {
        hits.push(pos + rel);
        pos += rel + 1;
    }
    hits
}

fn push_minimal_video(muxer: &mut Av1Mp4Muxer, frames: usize) {
    // First packet carries the seq header; rest are opaque.
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
    // Each AAC access unit is fake-opaque but non-empty. Duration = 1024
    // (natural AAC frame size).
    for i in 0..count {
        let blob = vec![0x5Au8; frame_size];
        muxer
            .add_audio_sample(&blob, (i * 1024) as u64, 1024)
            .expect("audio sample");
    }
}

#[test]
fn audio_mux_bails_on_non_aac_codec() {
    let mut muxer = Av1Mp4Muxer::new(320, 240, 30.0).expect("muxer");
    let info = AudioInfo {
        codec: "mp3".into(),
        sample_rate: 48000,
        channels: 2,
        timescale: 48000,
        asc_bytes: vec![0x12, 0x10],
        codec_private: Vec::new(),
    };
    let err = muxer.with_audio(info).err().expect("should reject mp3");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("AAC") && msg.contains("Opus"),
        "error should mention both supported codecs: {msg}"
    );
}

#[test]
fn audio_mux_bails_on_extended_channel_layout() {
    // Squad-25: 5.1 (channels=6) and 7.1 (channels=7) are now accepted with
    // a `chan` box. Channel counts outside {1, 2, 6, 7} (e.g. 8 = 7.1 + Atmos
    // height channels, or non-standard quad) must still bail clearly.
    // Squad-28 lifted Opus separately to 1..=8 via Multistream — see
    // `with_audio` for the per-codec channel matrix.
    let mut muxer = Av1Mp4Muxer::new(320, 240, 30.0).expect("muxer");
    let info = AudioInfo {
        codec: "aac".into(),
        sample_rate: 48000,
        channels: 8, // Atmos / extended layout — unsupported
        timescale: 48000,
        asc_bytes: aac_lc_stereo_asc(),
        codec_private: Vec::new(),
    };
    let err = muxer
        .with_audio(info)
        .err()
        .expect("should reject 8-channel AAC");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("Atmos") || msg.contains("not supported"),
        "error should mention extended layouts: {msg}"
    );
}

#[test]
fn audio_mux_bails_on_implicit_he_aac_signaling() {
    // Squad-25: AAC-LC ASC at low core rate (≤24 kHz) is the canonical
    // implicit-HE-AAC signaling shape. Apple silently downgrades this to
    // mono 22.05 kHz core. The mux now rejects it loudly so the caller
    // upgrades to explicit signaling first.
    //   AOT=2 (00010), SFI=6 → 24000 (0110), chan=1 (0001).
    //   00010 0110 0001 000 = 0001 0011 0000 1000 = 0x13 0x08.
    let mut muxer = Av1Mp4Muxer::new(320, 240, 30.0).expect("muxer");
    let info = AudioInfo {
        codec: "aac".into(),
        sample_rate: 24000,
        channels: 1,
        timescale: 24000,
        asc_bytes: vec![0x13, 0x08],
        codec_private: Vec::new(),
    };
    let err = muxer
        .with_audio(info)
        .err()
        .expect("should reject implicit HE-AAC");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("implicit") || msg.contains("upgrade"),
        "error should mention implicit signaling: {msg}"
    );
}

#[test]
fn audio_mux_writes_mp4a_sample_entry() {
    let mut muxer = Av1Mp4Muxer::new(320, 240, 30.0).expect("muxer");
    push_minimal_video(&mut muxer, 10);
    muxer
        .with_audio(aac_info_stereo_44100())
        .expect("with_audio");
    push_aac_samples(&mut muxer, 8, 300);
    let out = muxer.finalize().expect("finalize");
    assert!(find_fourcc(&out, b"mp4a").is_some(), "no mp4a sample entry");
    assert!(find_fourcc(&out, b"soun").is_some(), "no soun handler");
    assert!(find_fourcc(&out, b"smhd").is_some(), "no smhd");
}

#[test]
fn audio_mux_writes_esds_with_asc_verbatim() {
    let mut muxer = Av1Mp4Muxer::new(320, 240, 30.0).expect("muxer");
    push_minimal_video(&mut muxer, 10);
    let info = aac_info_stereo_44100();
    let asc = info.asc_bytes.clone();
    muxer.with_audio(info).expect("with_audio");
    push_aac_samples(&mut muxer, 5, 200);
    let out = muxer.finalize().expect("finalize");
    // ASC must appear inside the esds box. Byte-scan is sufficient — no
    // other box body contains a 2-byte 0x12 0x10 pair at the DSI position.
    let esds_pos = find_fourcc(&out, b"esds").expect("no esds");
    let search_window = &out[esds_pos..(esds_pos + 80).min(out.len())];
    let found = search_window
        .windows(asc.len())
        .any(|w| w == asc.as_slice());
    assert!(found, "ASC bytes {:02x?} not found after esds", asc);
}

#[test]
fn audio_mux_esds_descriptor_length_encoding() {
    // Case A: ASC of 2 bytes → all descriptor lengths < 128 → single-byte
    // length fields. Case B: ASC padded to >= 128 bytes → length field has
    // to use the 4-byte continuation form for the DCD (which wraps the
    // DSI + 13-byte preamble).

    // Case A.
    let mut muxer_a = Av1Mp4Muxer::new(320, 240, 30.0).expect("muxer");
    push_minimal_video(&mut muxer_a, 4);
    muxer_a
        .with_audio(aac_info_stereo_44100())
        .expect("with_audio A");
    push_aac_samples(&mut muxer_a, 3, 80);
    let out_a = muxer_a.finalize().expect("finalize A");
    let esds_a = find_fourcc(&out_a, b"esds").expect("esds A");
    // esds FullBox header is 4 size + 4 type + 4 ver/flags = 12; ES_Descr
    // tag byte at +12, then length byte (should be < 128).
    let es_tag = out_a[esds_a + 8]; // box body starts 8 after fourcc position - 4 = size field
    // actually layout: data[esds_a - 4..esds_a] = size, [esds_a..+4] = "esds"
    // body starts at esds_a + 4; FullBox ver/flags = 4 bytes; so ES_Descriptor
    // tag byte is at esds_a + 8.
    assert_eq!(
        es_tag, 0x03,
        "ES_Descriptor tag mismatch (got 0x{:02x})",
        es_tag
    );
    let es_len_byte = out_a[esds_a + 9];
    assert!(
        es_len_byte < 128,
        "expected single-byte length in case A, got 0x{:02x}",
        es_len_byte
    );

    // Case B: large ASC > 127 bytes. This isn't a realistic AAC-LC ASC
    // (those are 2-5 bytes), but the length encoder must handle it per
    // 14496-1. Pad the ASC with zeros past the initial audioObjectType
    // field so the descriptor-length branch fires.
    let mut big_asc = aac_lc_stereo_asc();
    big_asc.extend(std::iter::repeat(0u8).take(200));
    let mut muxer_b = Av1Mp4Muxer::new(320, 240, 30.0).expect("muxer");
    push_minimal_video(&mut muxer_b, 4);
    muxer_b
        .with_audio(AudioInfo {
            codec: "aac".into(),
            sample_rate: 44100,
            channels: 2,
            timescale: 44100,
            asc_bytes: big_asc,
            codec_private: Vec::new(),
        })
        .expect("with_audio B");
    push_aac_samples(&mut muxer_b, 3, 80);
    let out_b = muxer_b.finalize().expect("finalize B");
    let esds_b = find_fourcc(&out_b, b"esds").expect("esds B");
    let es_tag_b = out_b[esds_b + 8];
    assert_eq!(es_tag_b, 0x03);
    // With a 200-byte DSI, the DCD body = 13 + 2 + 200 + 2 = 217 bytes (DSI
    // descriptor header is 1 tag + 1 length, since DSI itself is 200 bytes
    // -> 200 > 127, so DSI length uses 4-byte continuation form; DCD body
    // = 13 preamble + 1 DSI tag + 4 DSI len + 200 DSI data = 218). Plus the
    // ES_Descriptor wraps DCD (4-byte len) + SLC (3 bytes) + 3 bytes ES
    // preamble -> ES_Descr length also uses 4-byte encoding. So the ES
    // length byte at esds_b+9 must have its high bit set.
    assert!(
        out_b[esds_b + 9] & 0x80 != 0,
        "expected 4-byte descriptor-length encoding for ES_Descriptor, got 0x{:02x}",
        out_b[esds_b + 9]
    );
}

/// Helper: extract a (stco or co64) chunk-offset entry_count + entries for
/// the stbl at the given box position. `co64` stores u64, `stco` stores u32.
fn read_offsets(data: &[u8], box_pos: usize) -> Vec<u64> {
    // layout: [pos-4..pos] size, [pos..pos+4] fourcc, then ver/flags/count/entries
    let size = u32::from_be_bytes([
        data[box_pos - 4],
        data[box_pos - 3],
        data[box_pos - 2],
        data[box_pos - 1],
    ]) as usize;
    let body_end = box_pos - 4 + size;
    let ver = data[box_pos + 4];
    assert_eq!(ver, 0);
    let count = u32::from_be_bytes([
        data[box_pos + 8],
        data[box_pos + 9],
        data[box_pos + 10],
        data[box_pos + 11],
    ]) as usize;
    let entry_start = box_pos + 12;
    let is_co64 = &data[box_pos..box_pos + 4] == b"co64";
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        if is_co64 {
            let p = entry_start + i * 8;
            assert!(p + 8 <= body_end, "stco/co64 entry out of body bounds");
            out.push(u64::from_be_bytes([
                data[p],
                data[p + 1],
                data[p + 2],
                data[p + 3],
                data[p + 4],
                data[p + 5],
                data[p + 6],
                data[p + 7],
            ]));
        } else {
            let p = entry_start + i * 4;
            assert!(p + 4 <= body_end, "stco entry out of bounds");
            out.push(u32::from_be_bytes([data[p], data[p + 1], data[p + 2], data[p + 3]]) as u64);
        }
    }
    out
}

#[test]
fn audio_mux_interleaved_chunks_offsets_correct() {
    // Mux 30 video frames @ 30fps → 1 chunk (spc = 30). Audio spc for
    // 44100 Hz = round(44100/1024) = 43 samples/chunk. Push 43 samples →
    // 1 audio chunk. Interleave plan: [video_chunk_0, audio_chunk_0].
    // Each video packet after the first is 128 bytes; first packet ≈ 8
    // bytes (OBU header + LEB128 len + 5 payload = 8). So video chunk 0
    // = 8 + 29 * 128 = 3720.
    let mut muxer = Av1Mp4Muxer::new(320, 240, 30.0).expect("muxer");
    push_minimal_video(&mut muxer, 30);
    muxer
        .with_audio(aac_info_stereo_44100())
        .expect("with_audio");
    push_aac_samples(&mut muxer, 43, 100);
    let out = muxer.finalize().expect("finalize");

    // There must be exactly 2 stco boxes (one per trak) or equivalent
    // co64 (depending on upper bound). For small payloads stco is used.
    let stco_hits: Vec<usize> = find_all_fourcc(&out, b"stco");
    let co64_hits: Vec<usize> = find_all_fourcc(&out, b"co64");
    let total_chunk_boxes = stco_hits.len() + co64_hits.len();
    assert_eq!(
        total_chunk_boxes, 2,
        "expected 2 chunk-offset tables, got {}",
        total_chunk_boxes
    );

    let all: Vec<usize> = stco_hits.into_iter().chain(co64_hits.into_iter()).collect();
    let offsets_a = read_offsets(&out, all[0]);
    let offsets_b = read_offsets(&out, all[1]);
    assert_eq!(offsets_a.len(), 1, "video expected 1 chunk");
    assert_eq!(offsets_b.len(), 1, "audio expected 1 chunk");

    // Video chunk offset + video chunk size should equal audio chunk
    // offset. Video chunk size = 7 (first packet: 1 OBU header + 1 LEB128
    // len byte for 5 + 5 payload) + 29 * 128 = 3719.
    let video_chunk_size: u64 = 7 + 29 * 128;
    assert_eq!(
        offsets_a[0] + video_chunk_size,
        offsets_b[0],
        "audio offset should sit immediately after video chunk: v={} +{} != a={}",
        offsets_a[0],
        video_chunk_size,
        offsets_b[0]
    );

    // Both offsets must point inside mdat. mdat fourcc position in out.
    let mdat_pos = find_fourcc(&out, b"mdat").expect("mdat");
    let mdat_payload_start = (mdat_pos + 4) as u64; // fourcc is at pos; size is before
    // Actually: [pos-4..pos] = size(4), [pos..pos+4] = "mdat", payload starts at pos+4.
    assert!(
        offsets_a[0] >= mdat_payload_start,
        "video offset {} should be >= mdat payload start {}",
        offsets_a[0],
        mdat_payload_start
    );
    assert!(
        offsets_b[0] >= mdat_payload_start,
        "audio offset {} should be >= mdat payload start {}",
        offsets_b[0],
        mdat_payload_start
    );
}

#[test]
fn audio_mux_video_plus_audio_roundtrip() {
    // Build an MP4 with a video trak + audio trak, then demux it and
    // assert the audio round-trips: ASC bytes match and sample count is
    // preserved. This exercises the two builders end-to-end without
    // needing a real AAC decoder.
    let mut muxer = Av1Mp4Muxer::new(320, 240, 30.0).expect("muxer");
    push_minimal_video(&mut muxer, 15);
    let info = AudioInfo {
        codec: "aac".into(),
        sample_rate: 48000,
        channels: 2,
        timescale: 48000,
        asc_bytes: aac_lc_stereo_48k_asc(),
        codec_private: Vec::new(),
    };
    let expected_asc = info.asc_bytes.clone();
    let expected_count = 20;
    muxer.with_audio(info).expect("with_audio");
    push_aac_samples(&mut muxer, expected_count, 250);
    let out = muxer.finalize().expect("finalize");

    let demuxed = demux::demux(&out).expect("demux roundtrip");
    let audio = demuxed.audio.expect("audio track missing after roundtrip");
    assert_eq!(audio.codec, "aac");
    assert_eq!(audio.channels, 2);
    assert_eq!(audio.sample_rate, 48000);
    assert_eq!(audio.timescale, 48000);
    assert_eq!(
        audio.samples.len(),
        expected_count,
        "sample count drifted: expected {}, got {}",
        expected_count,
        audio.samples.len()
    );
    assert_eq!(
        audio.asc, expected_asc,
        "ASC bytes changed across mux roundtrip"
    );
}

#[test]
fn audio_mux_drops_empty_audio_track() {
    // Caller invokes with_audio then never pushes a sample. Finalize must
    // not emit a half-formed audio trak. Historically this would have
    // produced an audio trak with zero stsz entries, which breaks players.
    let mut muxer = Av1Mp4Muxer::new(320, 240, 30.0).expect("muxer");
    push_minimal_video(&mut muxer, 5);
    muxer
        .with_audio(aac_info_stereo_44100())
        .expect("with_audio");
    // NB: no add_audio_sample calls.
    let out = muxer.finalize().expect("finalize");
    assert!(
        find_fourcc(&out, b"mp4a").is_none(),
        "audio trak should be dropped when empty"
    );
    assert!(
        find_fourcc(&out, b"soun").is_none(),
        "soun handler should be absent"
    );
}
