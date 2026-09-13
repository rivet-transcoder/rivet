/// AC-3 / E-AC-3 helpers for MP4 audio extraction.
///
/// Box-walking primitives live in `demux/mod.rs` and are reached via
/// `super::super::` (super = audio, super::super = demux).

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
    let mut pos = 0;
    while pos + 8 <= moov.len() {
        let size =
            u32::from_be_bytes([moov[pos], moov[pos + 1], moov[pos + 2], moov[pos + 3]]) as usize;
        let btype = &moov[pos + 4..pos + 8];
        if size < 8 || pos.checked_add(size).is_none_or(|end| end > moov.len()) {
            break;
        }
        if btype == b"trak" {
            let trak_body = &moov[pos + 8..pos + size];
            if let Some(cfg) = extract_audio_cfg_from_trak(trak_body, entry_fourcc, cfg_fourcc) {
                return Some(cfg);
            }
        }
        pos += size;
    }
    None
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
/// 366 §F.6. Squad-26 only emits / extracts the single-substream form
/// (5-byte body), which is what every vanilla 5.1 / 7.1 E-AC-3 file uses.
pub(crate) fn eac3_sample_rate_channels_from_dec3(dec3: &[u8]) -> Option<(u32, u16)> {
    if dec3.len() < 5 {
        return None;
    }
    // Header: data_rate(13b) + num_ind_sub-1(3b) packed in bytes 0..2.
    // Per-substream block starts at bit position 16.
    // bits 16..18 = fscod
    //  18..23 = bsid (=16)
    //  23..24 = reserved
    //  24..25 = asvc
    //  25..28 = bsmod
    //  28..31 = acmod
    //  31..32 = lfeon
    let raw_be = u64::from(dec3[0]) << 32
        | u64::from(dec3[1]) << 24
        | u64::from(dec3[2]) << 16
        | u64::from(dec3[3]) << 8
        | u64::from(dec3[4]);
    // dec3 is 5 bytes total (40 bits) for the single-substream case.
    // Adjust shifts: high bit is bit 39 in our 40-bit value.
    //   bit 39..27 = data_rate (13 bits)  shift=27
    //   bit 26..24 = num_ind_sub-1        shift=24
    //   bit 23..22 = fscod                shift=22
    //   bit 21..17 = bsid                 shift=17
    //   bit 16     = reserved
    //   bit 15     = asvc
    //   bit 14..12 = bsmod
    //   bit 11..9  = acmod                shift=9
    //   bit 8      = lfeon                shift=8
    //   bit 7..5   = reserved
    //   bit 4..1   = num_dep_sub
    //   bit 0      = reserved
    let fscod = ((raw_be >> 22) & 0x03) as u8;
    let acmod = ((raw_be >> 9) & 0x07) as u8;
    let lfeon = ((raw_be >> 8) & 0x01) == 1;
    let sr = crate::ac3_sync::eac3_sample_rate_hz(fscod, 0);
    if sr == 0 {
        return None;
    }
    Some((sr, crate::ac3_sync::channel_count(acmod, lfeon)))
}
