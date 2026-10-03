//! AC-3 / E-AC-3 helpers for MP4 audio extraction.
//!
//! Box-walking primitives live in `demux/mod.rs` and are reached via
//! `super::super::` (super = audio, super::super = demux).

/// Walk every `trak` looking for one whose `stsd` contains an `ac-3`
/// sample entry (ETSI TS 102 366 §F.2). Returns the body bytes of the
/// contained `dac3` box (without the 8-byte box header) or None.
pub(super) fn extract_mp4_ac3_dac3_body(data: &[u8]) -> Option<Vec<u8>> {
    extract_mp4_audio_config_body(data, b"ac-3", b"dac3")
}

/// Walk every `trak` looking for one whose `stsd` contains an `ec-3`
/// sample entry (ETSI TS 102 366 §F.5). Returns the body bytes of the
/// contained `dec3` box (without the 8-byte box header) or None.
pub(super) fn extract_mp4_eac3_dec3_body(data: &[u8]) -> Option<Vec<u8>> {
    extract_mp4_audio_config_body(data, b"ec-3", b"dec3")
}

/// Generic walker — find an audio sample-entry of `entry_fourcc`, return
/// the body of the named codec-config child (`dac3` / `dec3` / `ddts`)
/// inside. Mirrors `extract_mp4_opus_dops_body`'s shape but parameterised
/// on the entry / config 4-cc pair.
pub(super) fn extract_mp4_audio_config_body(
    data: &[u8],
    entry_fourcc: &[u8; 4],
    cfg_fourcc: &[u8; 4],
) -> Option<Vec<u8>> {
    let moov = super::super::find_direct_child(data, b"moov")?;
    super::super::direct_children(moov, b"trak")
        .find_map(|trak_body| extract_audio_cfg_from_trak(trak_body, entry_fourcc, cfg_fourcc))
}

fn extract_audio_cfg_from_trak(
    trak: &[u8],
    entry_fourcc: &[u8; 4],
    cfg_fourcc: &[u8; 4],
) -> Option<Vec<u8>> {
    let stsd = super::super::find_box_body(trak, &[b"mdia", b"minf", b"stbl", b"stsd"])?;
    if stsd.len() < 16 {
        return None;
    }
    let mut pos = 8; // skip version/flags/entry_count
    while pos + 8 <= stsd.len() {
        let entry_size =
            u32::from_be_bytes([stsd[pos], stsd[pos + 1], stsd[pos + 2], stsd[pos + 3]]) as usize;
        let entry_type: [u8; 4] = stsd[pos + 4..pos + 8].try_into().ok()?;
        if entry_size < 8 || pos.saturating_add(entry_size) > stsd.len() {
            break;
        }
        if &entry_type == entry_fourcc {
            let end = pos + entry_size;
            // AudioSampleEntry layout per ISO/IEC 14496-12 §8.5.2.2: after
            // the 8-byte box header there's a 28-byte fixed preamble
            // followed by nested codec-specific boxes.
            let child_start = pos + 8 + 28;
            if child_start >= end {
                return None;
            }
            return super::super::find_direct_child(&stsd[child_start..end], cfg_fourcc)
                .map(|b| b.to_vec());
        }
        pos += entry_size;
    }
    None
}

/// Decode (sample_rate, channel_count) from a 3-byte `dac3` body per
/// ETSI TS 102 366 §F.4. Bit layout (MSB-first across 24 bits):
///   bits 23..22 fscod          (shift=22)
///   bits 21..17 bsid           (shift=17)
///   bits 16..14 bsmod          (shift=14)
///   bits 13..11 acmod          (shift=11)
///   bit  10     lfeon          (shift=10)
///   bits  9.. 5 bit_rate_code  (shift= 5)
///   bits  4.. 0 reserved (=0)
pub(crate) fn ac3_sample_rate_channels_from_dac3(dac3: &[u8]) -> Option<(u32, u16)> {
    if dac3.len() < 3 {
        return None;
    }
    let raw = ((dac3[0] as u32) << 16) | ((dac3[1] as u32) << 8) | dac3[2] as u32;
    let fscod = ((raw >> 22) & 0x03) as u8;
    let acmod = ((raw >> 11) & 0x07) as u8;
    let lfeon = ((raw >> 10) & 0x01) == 1;
    let sr = match fscod {
        0 => 48_000,
        1 => 44_100,
        2 => 32_000,
        _ => return None,
    };
    Some((sr, crate::ac3_sync::channel_count(acmod, lfeon)))
}

/// Decode (sample_rate, channel_count) from a `dec3` body per ETSI TS 102
/// 366 F.6, for its first independent substream: 13 bits `data_rate`, 3
/// `num_ind_sub`, then `fscod` 2, `bsid` 5, reserved 1, `asvc` 1, `bsmod`
/// 3, `acmod` 3, `lfeon` 1, reserved 3, `num_dep_sub` 4 and, when that is
/// not zero, `chan_loc` 9 — the locations the dependent substreams add
/// (7.1: Lrs/Rrs), counted into the channels.
pub(crate) fn eac3_sample_rate_channels_from_dec3(dec3: &[u8]) -> Option<(u32, u16)> {
    if dec3.len() < 5 {
        return None;
    }
    let bits = |from: usize, n: usize| -> Option<u16> {
        let mut v = 0u16;
        for i in from..from + n {
            v = (v << 1) | u16::from((dec3.get(i / 8)? >> (7 - i % 8)) & 1);
        }
        Some(v)
    };
    let fscod = bits(16, 2)? as u8;
    let acmod = bits(28, 3)? as u8;
    let lfeon = bits(31, 1)? == 1;
    let num_dep_sub = bits(35, 4)?;
    let chan_loc = if num_dep_sub > 0 { bits(39, 9).unwrap_or(0) } else { 0 };
    let sr = crate::ac3_sync::eac3_sample_rate_hz(fscod, 0);
    if sr == 0 {
        return None;
    }
    Some((sr, crate::ac3_sync::channel_count(acmod, lfeon) + crate::ac3_sync::chan_loc_channels(chan_loc)))
}
