// AC-3 and E-AC-3 mux box layout (ETSI TS 102 366 §F).
// 9 #[test] functions.

use crate::AudioInfo;
use crate::ac3_sync::{Ac3SyncInfo, Eac3SyncInfo};
use super::super::Av1Mp4Muxer;
use super::super::audio_track::{
    build_audio_stsd, build_dac3, build_dec3, build_ac3_sample_entry,
    build_ec3_sample_entry, dac3_body_from_sync, dec3_body_from_sync,
};

// ---- local fixtures -------------------------------------------------------

/// Canonical 5.1 384 kbps 48 kHz AC-3:
///   fscod=0, bsid=8, bsmod=0, acmod=7 (3/2), lfeon=1, bit_rate_code=14.
fn ac3_sync_5_1_384k_48k() -> Ac3SyncInfo {
    Ac3SyncInfo {
        fscod: 0,
        bit_rate_code: 14,
        bsid: 8,
        bsmod: 0,
        acmod: 7,
        lfeon: true,
    }
}

fn ac3_info_5_1_384k() -> AudioInfo {
    let body = dac3_body_from_sync(&ac3_sync_5_1_384k_48k());
    AudioInfo::ac3(48_000, 6, body.to_vec())
}

/// Vanilla 5.1 E-AC-3 single independent substream, 48 kHz, 384 kbps.
fn eac3_sync_5_1_48k() -> Eac3SyncInfo {
    Eac3SyncInfo {
        strmtyp: 0,
        substreamid: 0,
        // frmsiz arbitrary for box-layout tests; choose 191 → frame
        // size = 384 bytes which corresponds to 384 kbps @ 48 kHz / 1536
        // samples-per-frame.
        frmsiz: 191,
        fscod: 0,
        fscod2: 0,
        numblkscod: 3,
        acmod: 7,
        lfeon: true,
        bsid: 16,
        dialnorm: 0,
        bsmod: 0,
    }
}

fn eac3_info_5_1_384k() -> AudioInfo {
    // 384 kbps → data_rate field = 384 (kbps, ETSI TS 102 366 F.6.2.2).
    let body = dec3_body_from_sync(&eac3_sync_5_1_48k(), 384);
    AudioInfo::eac3(48_000, 6, body.to_vec())
}

// ---- Squad-26: AC-3 + E-AC-3 mux box layout (ETSI TS 102 366 §F) --------

/// `dac3` is exactly 11 bytes total (8-byte box header + 3-byte body).
/// Body field positions per ETSI TS 102 366 §F.4: fscod 2b | bsid 5b |
/// bsmod 3b | acmod 3b | lfeon 1b | bit_rate_code 5b | reserved 5b.
#[test]
fn dac3_box_3_byte_payload_layout() {
    let info = ac3_info_5_1_384k();
    let dac3 = build_dac3(&info);
    assert_eq!(dac3.len(), 11, "dac3 = 8-byte header + 3-byte body");
    let size = u32::from_be_bytes([dac3[0], dac3[1], dac3[2], dac3[3]]) as usize;
    assert_eq!(size, dac3.len(), "size field equals box length");
    assert_eq!(&dac3[4..8], b"dac3", "box type 'dac3'");
    // Body bit-extract (24 bits, MSB-first across 3 bytes 8..11).
    let raw = ((dac3[8] as u32) << 16) | ((dac3[9] as u32) << 8) | dac3[10] as u32;
    assert_eq!((raw >> 22) & 0x03, 0, "fscod = 0 (48 kHz)");
    assert_eq!((raw >> 17) & 0x1F, 8, "bsid = 8 (AC-3)");
    assert_eq!((raw >> 14) & 0x07, 0, "bsmod = 0");
    assert_eq!((raw >> 11) & 0x07, 7, "acmod = 7 (3/2 = 5.1 with LFE)");
    assert_eq!((raw >> 10) & 0x01, 1, "lfeon = 1");
    assert_eq!((raw >> 5) & 0x1F, 14, "bit_rate_code = 14 (= 384 kbps)");
    assert_eq!(raw & 0x1F, 0, "reserved 5 bits = 0");
}

/// `ac-3` AudioSampleEntry per ETSI TS 102 366 §F.2.
/// Total = 36-byte sample-entry preamble + 11-byte dac3 = 47 bytes.
/// 4cc is `ac-3` exactly (with the hyphen at byte index 6 = 0x2D).
#[test]
fn ac3_sample_entry_size_and_fourcc() {
    let info = ac3_info_5_1_384k();
    let entry = build_ac3_sample_entry(&info);
    let size = u32::from_be_bytes([entry[0], entry[1], entry[2], entry[3]]) as usize;
    assert_eq!(size, entry.len(), "size field equals box length");
    assert_eq!(&entry[4..8], b"ac-3", "4cc MUST be 'ac-3' (with hyphen)");
    // Reject the dehyphenated form
    assert_ne!(
        &entry[4..8],
        b"ac3\0",
        "4cc 'ac3' (3-char) is non-conformant"
    );
    assert_eq!(
        entry.len(),
        47,
        "ac-3 sample entry = 36 (preamble) + 11 (dac3)"
    );
    // dac3 must nest inside.
    let dac3_pos = entry
        .windows(4)
        .position(|w| w == b"dac3")
        .expect("dac3 child missing");
    assert!(
        dac3_pos > 28,
        "dac3 must come after AudioSampleEntry preamble"
    );
    // samplerate field at box-relative offset 8 + 24 = 32.
    let sr_q16 = u32::from_be_bytes([entry[32], entry[33], entry[34], entry[35]]);
    assert_eq!(sr_q16, 48_000u32 << 16, "samplerate = 48000 << 16 (Q16)");
}

/// `dec3` for a single independent substream (Squad-26's scope) is
/// 13 bytes total = 8-byte box header + 5-byte body (no dependent
/// substreams = no chan_loc tail). Body layout per ETSI TS 102 366
/// §F.6.
#[test]
fn dec3_box_5_byte_payload_layout() {
    let info = eac3_info_5_1_384k();
    let dec3 = build_dec3(&info);
    assert_eq!(dec3.len(), 13, "dec3 = 8-byte header + 5-byte body");
    let size = u32::from_be_bytes([dec3[0], dec3[1], dec3[2], dec3[3]]) as usize;
    assert_eq!(size, dec3.len(), "size field equals box length");
    assert_eq!(&dec3[4..8], b"dec3", "box type 'dec3'");
    // Body header: data_rate(13) + num_ind_sub-1(3) packed in bytes 8..10.
    let header = ((dec3[8] as u16) << 8) | dec3[9] as u16;
    let data_rate = (header >> 3) & 0x1FFF;
    assert_eq!(data_rate, 384, "data_rate = 384 kbps (F.6.2.2)");
    let num_ind_sub_minus_1 = header & 0x07;
    assert_eq!(num_ind_sub_minus_1, 0, "single substream → field = 0");
    // Per-independent-substream block: bits 16..40 (3 bytes 10..13).
    // Layout shifts within the 24-bit window:
    //   bit 23..22 fscod
    //   bit 21..17 bsid (=16)
    //   bit 16     reserved
    //   bit 15     asvc
    //   bit 14..12 bsmod
    //   bit 11..9  acmod
    //   bit 8      lfeon
    //   bit 7..5   reserved
    //   bit 4..1   num_dep_sub (=0)
    //   bit 0      reserved
    let sub = ((dec3[10] as u32) << 16) | ((dec3[11] as u32) << 8) | dec3[12] as u32;
    assert_eq!((sub >> 22) & 0x03, 0, "fscod = 0 (48 kHz)");
    assert_eq!((sub >> 17) & 0x1F, 16, "bsid = 16 (E-AC-3 marker)");
    assert_eq!((sub >> 12) & 0x07, 0, "bsmod = 0");
    assert_eq!((sub >> 9) & 0x07, 7, "acmod = 7 (3/2 = 5.1 with LFE)");
    assert_eq!((sub >> 8) & 0x01, 1, "lfeon = 1");
    assert_eq!((sub >> 1) & 0x0F, 0, "num_dep_sub = 0 (single substream)");
}

/// `ec-3` AudioSampleEntry per ETSI TS 102 366 §F.5.
/// Total = 36-byte sample-entry preamble + 13-byte dec3 = 49 bytes.
#[test]
fn ec3_sample_entry_size_and_fourcc() {
    let info = eac3_info_5_1_384k();
    let entry = build_ec3_sample_entry(&info);
    let size = u32::from_be_bytes([entry[0], entry[1], entry[2], entry[3]]) as usize;
    assert_eq!(size, entry.len(), "size field equals box length");
    assert_eq!(&entry[4..8], b"ec-3", "4cc MUST be 'ec-3' (with hyphen)");
    assert_eq!(
        entry.len(),
        49,
        "ec-3 sample entry = 36 (preamble) + 13 (dec3)"
    );
    let dec3_pos = entry
        .windows(4)
        .position(|w| w == b"dec3")
        .expect("dec3 child missing");
    assert!(
        dec3_pos > 28,
        "dec3 must come after AudioSampleEntry preamble"
    );
}

/// stsd dispatcher: ac3 info → ac-3 entry; eac3 info → ec-3 entry.
/// Must NOT cross-pollinate with mp4a / Opus.
#[test]
fn stsd_dispatcher_routes_ac3_eac3() {
    let stsd_ac3 = build_audio_stsd(&ac3_info_5_1_384k());
    assert!(
        stsd_ac3.windows(4).any(|w| w == b"ac-3"),
        "AC-3 stsd has 'ac-3'"
    );
    assert!(
        stsd_ac3.windows(4).any(|w| w == b"dac3"),
        "AC-3 stsd has 'dac3'"
    );
    assert!(
        !stsd_ac3.windows(4).any(|w| w == b"mp4a"),
        "AC-3 stsd MUST NOT have mp4a"
    );
    assert!(
        !stsd_ac3.windows(4).any(|w| w == b"Opus"),
        "AC-3 stsd MUST NOT have Opus"
    );
    assert!(
        !stsd_ac3.windows(4).any(|w| w == b"esds"),
        "AC-3 stsd MUST NOT have esds"
    );

    let stsd_eac3 = build_audio_stsd(&eac3_info_5_1_384k());
    assert!(
        stsd_eac3.windows(4).any(|w| w == b"ec-3"),
        "E-AC-3 stsd has 'ec-3'"
    );
    assert!(
        stsd_eac3.windows(4).any(|w| w == b"dec3"),
        "E-AC-3 stsd has 'dec3'"
    );
    assert!(
        !stsd_eac3.windows(4).any(|w| w == b"mp4a"),
        "E-AC-3 stsd MUST NOT have mp4a"
    );
    assert!(
        !stsd_eac3.windows(4).any(|w| w == b"esds"),
        "E-AC-3 stsd MUST NOT have esds"
    );
    assert!(
        !stsd_eac3.windows(4).any(|w| w == b"dac3"),
        "E-AC-3 stsd MUST NOT have dac3"
    );
}

/// `with_audio` must accept a 5.1 AC-3 info and reject obvious shape
/// errors (wrong dac3 body length, wrong sample rate).
#[test]
fn with_audio_accepts_ac3_5_1_and_rejects_bad_shape() {
    let mut muxer = Av1Mp4Muxer::new(320, 240, 30.0).unwrap();
    muxer
        .with_audio(ac3_info_5_1_384k())
        .expect("5.1 AC-3 must be accepted");

    // Wrong body length
    let mut muxer2 = Av1Mp4Muxer::new(320, 240, 30.0).unwrap();
    let mut bad = ac3_info_5_1_384k();
    bad.codec_private = vec![0u8; 2];
    let err = muxer2
        .with_audio(bad)
        .err()
        .expect("must reject 2-byte dac3");
    assert!(format!("{err:#}").contains("3 bytes"));

    // Wrong sample rate
    let mut muxer3 = Av1Mp4Muxer::new(320, 240, 30.0).unwrap();
    let bad_sr = AudioInfo {
        sample_rate: 22_050,
        timescale: 22_050,
        ..ac3_info_5_1_384k()
    };
    let err = muxer3
        .with_audio(bad_sr)
        .err()
        .expect("must reject 22050 for AC-3");
    assert!(format!("{err:#}").contains("32000"));
}

/// `with_audio` must accept a single-substream E-AC-3 info and reject
/// an under-sized dec3 body.
#[test]
fn with_audio_accepts_eac3_5_1_and_rejects_short_dec3() {
    let mut muxer = Av1Mp4Muxer::new(320, 240, 30.0).unwrap();
    muxer
        .with_audio(eac3_info_5_1_384k())
        .expect("5.1 E-AC-3 must be accepted");

    let mut muxer2 = Av1Mp4Muxer::new(320, 240, 30.0).unwrap();
    let mut bad = eac3_info_5_1_384k();
    bad.codec_private = vec![0u8; 4];
    let err = muxer2
        .with_audio(bad)
        .err()
        .expect("must reject short dec3");
    assert!(format!("{err:#}").contains("≥5"));
}

/// AC-3 / E-AC-3 channel count gate: must reject >6.
#[test]
fn with_audio_rejects_ac3_more_than_6_channels() {
    let mut muxer = Av1Mp4Muxer::new(320, 240, 30.0).unwrap();
    let bad = AudioInfo {
        channels: 8,
        ..ac3_info_5_1_384k()
    };
    let err = muxer.with_audio(bad).err().expect("must reject 8 channels");
    assert!(format!("{err:#}").contains("1..=6"));
}

/// Round-trip: parse a synthetic 5.1 AC-3 sync header → derive dac3
/// body → pack into an `ac-3` sample entry → walk the bytes back out
/// and recover fscod / acmod / lfeon / bit_rate_code unchanged.
#[test]
fn ac3_sync_to_dac3_to_sample_entry_roundtrip() {
    let sync = ac3_sync_5_1_384k_48k();
    let body = dac3_body_from_sync(&sync);
    let info = AudioInfo::ac3(48_000, 6, body.to_vec());
    let entry = build_ac3_sample_entry(&info);
    // Find dac3 box body (8-byte box header inside the entry then 3
    // body bytes).
    let dac3_pos = entry.windows(4).position(|w| w == b"dac3").unwrap();
    let dac3_body_start = dac3_pos + 4;
    let raw = ((entry[dac3_body_start] as u32) << 16)
        | ((entry[dac3_body_start + 1] as u32) << 8)
        | entry[dac3_body_start + 2] as u32;
    assert_eq!((raw >> 22) & 0x03, sync.fscod as u32);
    assert_eq!((raw >> 17) & 0x1F, sync.bsid as u32);
    assert_eq!((raw >> 11) & 0x07, sync.acmod as u32);
    assert_eq!((raw >> 10) & 0x01, sync.lfeon as u32);
    assert_eq!((raw >> 5) & 0x1F, sync.bit_rate_code as u32);
}
