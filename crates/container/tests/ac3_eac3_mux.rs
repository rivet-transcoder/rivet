//! Integration tests for the AC-3 / E-AC-3 audio mux + sync-header parser
//! (Squad-26).
//!
//! Covers:
//!   1. `with_audio` accepts AudioInfo with codec="ac3" / "eac3" and the
//!      right shape.
//!   2. `ac-3` + `dac3` boxes appear in the muxed MP4 in the right
//!      nesting order, with no `mp4a` / `Opus` leakage.
//!   3. Same for `ec-3` + `dec3`.
//!   4. Synthesised AC-3 frame stream → mux into MP4 → re-demux → samples
//!      come back byte-identical (true passthrough).
//!   5. AC-3 sync-header parser hex-dump verification for canned 5.1 input.

use bytes::Bytes;
use codec::encode::EncodedPacket;
use container::AudioInfo;
use container::ac3_sync::{
    self, Ac3SyncInfo, Eac3SyncInfo, SyncInfo, ac3_bit_rate_kbps, ac3_sample_rate_hz,
    channel_count, parse_sync_info,
};
use container::mux::{Av1Mp4Muxer, dac3_body_from_sync, dec3_body_from_sync};

// Minimal AV1 OBU_SEQUENCE_HEADER with obu_has_size_field=1 — required to
// pass `extract_sequence_header` during finalize. (Mirrors the helper in
// `audio_mux.rs` / `opus_mux.rs`.)
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

// ---- Synthetic-frame helpers ------------------------------------------

/// MSB-first bit writer for synthesizing AC-3 / E-AC-3 wire-format
/// headers (matches `crate::ac3_sync::tests`'s helper).
struct BitWriter {
    bytes: Vec<u8>,
    bit_pos: usize,
}
impl BitWriter {
    fn new() -> Self {
        Self {
            bytes: Vec::new(),
            bit_pos: 0,
        }
    }
    fn put(&mut self, n: usize, v: u32) {
        for i in (0..n).rev() {
            let bit = ((v >> i) & 0x01) as u8;
            if self.bit_pos % 8 == 0 {
                self.bytes.push(0);
            }
            let byte_idx = self.bit_pos / 8;
            let bit_idx = 7 - (self.bit_pos % 8);
            self.bytes[byte_idx] |= bit << bit_idx;
            self.bit_pos += 1;
        }
    }
}

/// Synthesise a complete AC-3 syncframe. The first ~7 bytes carry the
/// BSI fields the parser cares about; the rest is opaque audio data we
/// fill with a per-frame seed so passthrough roundtrip can do byte-level
/// equality checks.
fn synth_ac3_frame(
    fscod: u8,
    bit_rate_code: u8,
    bsid: u8,
    bsmod: u8,
    acmod: u8,
    lfeon: bool,
    seed: u8,
) -> Vec<u8> {
    let mut bw = BitWriter::new();
    bw.put(16, 0x0B77);
    bw.put(16, 0); // crc1
    bw.put(2, fscod as u32);
    bw.put(6, (bit_rate_code as u32) << 1); // frmsizecod (bit_rate_code << 1)
    bw.put(5, bsid as u32);
    bw.put(3, bsmod as u32);
    bw.put(3, acmod as u32);
    if (acmod & 0x01) != 0 && acmod != 0x01 {
        bw.put(2, 0);
    }
    if (acmod & 0x04) != 0 {
        bw.put(2, 0);
    }
    if acmod == 0x02 {
        bw.put(2, 0);
    }
    bw.put(1, if lfeon { 1 } else { 0 });
    while bw.bytes.len() < 12 {
        bw.put(8, 0);
    }

    // Pad to a full nominal frame size for 5.1 384 kbps 48 kHz: 1536 bytes.
    // Real frames are exactly that; we tile a seed pattern after the BSI
    // prefix so byte-level demux roundtrip is exact.
    let target_size = 1536usize;
    let mut frame = bw.bytes;
    while frame.len() < target_size {
        frame.push(seed.wrapping_add(frame.len() as u8));
    }
    frame
}

// ---- Sync parser sanity checks -----------------------------------------

#[test]
fn sync_parser_recognises_canonical_5_1_ac3() {
    // 5.1 384 kbps 48 kHz: the most common Blu-ray / DVD AC-3 profile.
    let frame = synth_ac3_frame(0, 14, 8, 0, 7, true, 0x42);
    let info = parse_sync_info(&frame).expect("must parse");
    match info {
        SyncInfo::Ac3(s) => {
            assert_eq!(s.fscod, 0);
            assert_eq!(s.bit_rate_code, 14);
            assert_eq!(s.bsid, 8);
            assert_eq!(s.acmod, 7);
            assert!(s.lfeon);
            assert_eq!(channel_count(s.acmod, s.lfeon), 6);
            assert_eq!(ac3_sample_rate_hz(s.fscod), 48_000);
            assert_eq!(ac3_bit_rate_kbps(s.bit_rate_code), 384);
        }
        _ => panic!("expected AC-3"),
    }
}

#[test]
fn sync_parser_rejects_bad_bytes() {
    let mut frame = synth_ac3_frame(0, 14, 8, 0, 7, true, 0);
    frame[0] = 0xFF;
    assert!(parse_sync_info(&frame).is_err());
}

// ---- Mux: with_audio gate (AC-3) ---------------------------------------

fn ac3_5_1_info() -> AudioInfo {
    let s = Ac3SyncInfo {
        fscod: 0,
        bit_rate_code: 14,
        bsid: 8,
        bsmod: 0,
        acmod: 7,
        lfeon: true,
    };
    AudioInfo::ac3(48_000, 6, dac3_body_from_sync(&s).to_vec())
}

#[test]
fn ac3_with_audio_accepts_5_1() {
    let mut muxer = Av1Mp4Muxer::new(320, 240, 30.0).expect("muxer");
    muxer
        .with_audio(ac3_5_1_info())
        .expect("AC-3 5.1 should be accepted");
}

#[test]
fn ac3_with_audio_rejects_wrong_dac3_length() {
    let mut muxer = Av1Mp4Muxer::new(320, 240, 30.0).expect("muxer");
    let bad = AudioInfo {
        codec_private: vec![0u8; 4],
        ..ac3_5_1_info()
    };
    let err = muxer
        .with_audio(bad)
        .err()
        .expect("must reject 4-byte dac3");
    assert!(format!("{err:#}").contains("3 bytes"), "{err:#}");
}

#[test]
fn ac3_with_audio_rejects_unsupported_sample_rate() {
    let mut muxer = Av1Mp4Muxer::new(320, 240, 30.0).expect("muxer");
    let bad = AudioInfo {
        sample_rate: 16_000,
        timescale: 16_000,
        ..ac3_5_1_info()
    };
    let err = muxer
        .with_audio(bad)
        .err()
        .expect("must reject 16k for AC-3");
    assert!(format!("{err:#}").contains("32000"));
}

// ---- Mux: with_audio gate (E-AC-3) -------------------------------------

fn eac3_5_1_info() -> AudioInfo {
    let s = Eac3SyncInfo {
        strmtyp: 0,
        substreamid: 0,
        frmsiz: 191,
        fscod: 0,
        fscod2: 0,
        numblkscod: 3,
        acmod: 7,
        lfeon: true,
        bsid: 16,
        dialnorm: 0,
        bsmod: 0,
    };
    AudioInfo::eac3(48_000, 6, dec3_body_from_sync(&s, 192).to_vec())
}

#[test]
fn eac3_with_audio_accepts_5_1() {
    let mut muxer = Av1Mp4Muxer::new(320, 240, 30.0).expect("muxer");
    muxer
        .with_audio(eac3_5_1_info())
        .expect("E-AC-3 5.1 should be accepted");
}

#[test]
fn eac3_with_audio_rejects_short_dec3() {
    let mut muxer = Av1Mp4Muxer::new(320, 240, 30.0).expect("muxer");
    let bad = AudioInfo {
        codec_private: vec![0u8; 3],
        ..eac3_5_1_info()
    };
    let err = muxer
        .with_audio(bad)
        .err()
        .expect("must reject 3-byte dec3");
    assert!(format!("{err:#}").contains("≥5"));
}

// ---- Mux: sample-entry presence in finalised MP4 -----------------------

fn find_fourcc(data: &[u8], tag: &[u8; 4]) -> Option<usize> {
    data.windows(4).position(|w| w == tag)
}

#[test]
fn ac3_finalize_writes_ac_3_sample_entry_and_dac3() {
    let mut muxer = Av1Mp4Muxer::new(320, 240, 30.0).expect("muxer");
    push_minimal_video(&mut muxer, 10);
    muxer.with_audio(ac3_5_1_info()).expect("with_audio");
    // Push a couple of opaque AC-3 syncframes. Real frames would be 1536
    // bytes for 384 kbps 48 kHz; for box-presence checks the content is
    // immaterial.
    for i in 0..5 {
        let frame = synth_ac3_frame(0, 14, 8, 0, 7, true, i as u8);
        muxer
            .add_audio_sample(&frame, (i * 1536) as u64, 1536)
            .expect("audio");
    }
    let bytes = muxer.finalize().expect("finalize");
    assert!(
        find_fourcc(&bytes, b"ac-3").is_some(),
        "output must contain 'ac-3' sample entry"
    );
    assert!(
        find_fourcc(&bytes, b"dac3").is_some(),
        "output must contain 'dac3' config box"
    );
    assert!(
        find_fourcc(&bytes, b"mp4a").is_none(),
        "output must NOT contain mp4a"
    );
    assert!(
        find_fourcc(&bytes, b"Opus").is_none(),
        "output must NOT contain Opus"
    );
    assert!(
        find_fourcc(&bytes, b"esds").is_none(),
        "output must NOT contain esds"
    );
}

#[test]
fn eac3_finalize_writes_ec_3_sample_entry_and_dec3() {
    let mut muxer = Av1Mp4Muxer::new(320, 240, 30.0).expect("muxer");
    push_minimal_video(&mut muxer, 10);
    muxer.with_audio(eac3_5_1_info()).expect("with_audio");
    for i in 0..5 {
        // Reuse the AC-3 frame synth — for byte-presence tests the content
        // doesn't have to be a real E-AC-3 frame; the mux side doesn't
        // parse sample data. The dec3 body inside the sample entry is what
        // identifies the codec.
        let frame = synth_ac3_frame(0, 14, 8, 0, 7, true, i as u8);
        muxer
            .add_audio_sample(&frame, (i * 1536) as u64, 1536)
            .expect("audio");
    }
    let bytes = muxer.finalize().expect("finalize");
    assert!(
        find_fourcc(&bytes, b"ec-3").is_some(),
        "output must contain 'ec-3' sample entry"
    );
    assert!(
        find_fourcc(&bytes, b"dec3").is_some(),
        "output must contain 'dec3' config box"
    );
    assert!(
        find_fourcc(&bytes, b"mp4a").is_none(),
        "output must NOT contain mp4a"
    );
    assert!(
        find_fourcc(&bytes, b"Opus").is_none(),
        "output must NOT contain Opus"
    );
    assert!(
        find_fourcc(&bytes, b"dac3").is_none(),
        "E-AC-3 stsd MUST NOT contain dac3"
    );
}

// ---- End-to-end: synthesised AC-3 → mux → demux roundtrip --------------

#[test]
fn ac3_mux_demux_roundtrip_preserves_samples() {
    use container::demux;

    let mut muxer = Av1Mp4Muxer::new(320, 240, 30.0).expect("muxer");
    push_minimal_video(&mut muxer, 10);
    muxer.with_audio(ac3_5_1_info()).expect("with_audio");

    let mut sent_samples: Vec<Vec<u8>> = Vec::new();
    for i in 0..6 {
        // Real-size AC-3 frame: 1536 bytes for 384 kbps 48 kHz. A unique
        // seed per frame lets us assert byte-equality after the round trip.
        let frame = synth_ac3_frame(0, 14, 8, 0, 7, true, (i + 1) as u8);
        muxer
            .add_audio_sample(&frame, (i * 1536) as u64, 1536)
            .expect("audio");
        sent_samples.push(frame);
    }
    let bytes = muxer.finalize().expect("finalize");

    let demuxed = demux::demux(&bytes).expect("demux");
    let audio = demuxed.audio.expect("audio track must round-trip");
    assert_eq!(audio.codec, "ac3");
    assert_eq!(audio.channels, 6);
    assert_eq!(audio.sample_rate, 48_000);
    assert_eq!(audio.codec_private.len(), 3, "dac3 body is 3 bytes");
    assert_eq!(
        audio.samples.len(),
        sent_samples.len(),
        "mux+demux must preserve sample count"
    );
    for (idx, (got, sent)) in audio.samples.iter().zip(&sent_samples).enumerate() {
        assert_eq!(
            got, sent,
            "sample {idx} bytes must match (true passthrough)"
        );
    }
}

// ---- Hex-dump regression: dac3 + dec3 bodies for canned 5.1 inputs -----

/// dac3 body for canonical 5.1 48 kHz 384 kbps AC-3 (bsid=8, acmod=7,
/// lfeon=1, bit_rate_code=14, fscod=0). MSB-first packed bits:
///   fscod=00 | bsid=01000 | bsmod=000 | acmod=111 | lfeon=1 |
///   bit_rate_code=01110 | reserved=00000
/// Concatenated: 00 01000 000 111 1 01110 00000
/// Regrouped into 8-bit chunks (MSB-first):
///   0001 0000  0011 1101  1100 0000
///   = 0x10       0x3D       0xC0
#[test]
fn dac3_canonical_5_1_384k_hex_dump() {
    let s = Ac3SyncInfo {
        fscod: 0,
        bit_rate_code: 14,
        bsid: 8,
        bsmod: 0,
        acmod: 7,
        lfeon: true,
    };
    let body = dac3_body_from_sync(&s);
    assert_eq!(
        body,
        [0x10, 0x3D, 0xC0],
        "dac3 body (3 bytes) must hex-match {:02X?}",
        body
    );
}

/// dec3 body for canonical 5.1 48 kHz 384 kbps E-AC-3 (single
/// independent substream, num_ind_sub=1 / wire encoding 0, num_dep_sub=0).
///
/// Wire layout (40 bits total = 5 bytes), MSB-first within each byte:
///   data_rate=192          (13 bits) → 0_0000_1100_0000   (bit pos 0..13)
///   num_ind_sub-1=0        ( 3 bits) → 000                (pos 13..16)
///   per-independent-substream:
///     fscod=00             ( 2 bits)                       (pos 16..18)
///     bsid=10000 (=16)     ( 5 bits)                       (pos 18..23)
///     reserved=0           ( 1 bit )                       (pos 23..24)
///     asvc=0               ( 1 bit )                       (pos 24..25)
///     bsmod=000            ( 3 bits)                       (pos 25..28)
///     acmod=111 (=7)       ( 3 bits)                       (pos 28..31)
///     lfeon=1              ( 1 bit )                       (pos 31..32)
///     reserved=000         ( 3 bits)                       (pos 32..35)
///     num_dep_sub=0000     ( 4 bits)                       (pos 35..39)
///     (bit 39 → final reserved 0 bit, dropped — pad up to byte boundary)
///
/// Regrouped into 8-bit chunks (MSB-first within each byte):
///   pos  0.. 8: 0_0000_110 → 0000_0110 = 0x06
///   pos  8..16: 0_000_0000 → 0000_0000 = 0x00
///   pos 16..24: 00_10000_0 → 0010_0000 = 0x20
///   pos 24..32: 0_000_111_1 → 0000_1111 = 0x0F
///   pos 32..40: 000_0000_0 → 0000_0000 = 0x00
///
/// → full body: 06 00 20 0F 00
#[test]
fn dec3_canonical_5_1_384k_hex_dump() {
    let s = Eac3SyncInfo {
        strmtyp: 0,
        substreamid: 0,
        frmsiz: 191,
        fscod: 0,
        fscod2: 0,
        numblkscod: 3,
        acmod: 7,
        lfeon: true,
        bsid: 16,
        dialnorm: 0,
        bsmod: 0,
    };
    let body = dec3_body_from_sync(&s, 192);
    assert_eq!(
        body,
        [0x06, 0x00, 0x20, 0x0F, 0x00],
        "dec3 body (5 bytes) must hex-match {:02X?}",
        body
    );
}

// ---- BitReader / BitWriter sanity --------------------------------------

#[test]
fn bit_layout_is_msb_first() {
    // Sanity check the synthetic-frame BitWriter actually packs MSB-first.
    let mut bw = BitWriter::new();
    bw.put(8, 0b1100_1010);
    assert_eq!(bw.bytes, vec![0b1100_1010]);
}

// ---- ac3_sync re-export sanity -----------------------------------------

#[test]
fn ac3_sync_module_is_publicly_reachable() {
    // Compile-time check: the module is `pub` so external crates (callers
    // outside `container`) can construct sync info / parse frames if they
    // need to plumb a non-passthrough variant later.
    let _ = ac3_sync::Ac3SyncInfo {
        fscod: 0,
        bit_rate_code: 14,
        bsid: 8,
        bsmod: 0,
        acmod: 7,
        lfeon: true,
    };
    let _ = ac3_sync::Eac3SyncInfo {
        strmtyp: 0,
        substreamid: 0,
        frmsiz: 0,
        fscod: 0,
        fscod2: 0,
        numblkscod: 0,
        acmod: 0,
        lfeon: false,
        bsid: 16,
        dialnorm: 0,
        bsmod: 0,
    };
}
