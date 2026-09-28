//! Bitstream syntax writers (ISO/IEC 13818-7 subclause 6.3 and ISO/IEC
//! 14496-3 subclause 1.6.2.1 / 4.4.1): raw_data_block elements, the
//! AudioSpecificConfig and the ADTS header.

use super::bits::BitWriter;
use super::huffman;
use super::quant::{ChannelFrame, Layout, Quantized};

/// id_syn_ele values (Table 36).
pub(super) const ID_SCE: u32 = 0;
pub(super) const ID_CPE: u32 = 1;
pub(super) const ID_LFE: u32 = 3;
pub(super) const ID_FIL: u32 = 6;
pub(super) const ID_END: u32 = 7;

/// ics_info() length in bits.
pub(super) fn ics_info_bits(layout: &Layout) -> usize {
    if layout.short() {
        1 + 2 + 1 + 4 + 7
    } else {
        1 + 2 + 1 + 6 + 1
    }
}

pub(super) fn write_ics_info(w: &mut BitWriter, layout: &Layout, max_sfb: usize) {
    w.put(0, 1); // ics_reserved_bit
    w.put(layout.seq as u32, 2);
    w.put(0, 1); // window_shape: sine
    if layout.short() {
        w.put(max_sfb as u32, 4);
        w.put(layout.grouping_bits(), 7);
    } else {
        w.put(max_sfb as u32, 6);
        w.put(0, 1); // predictor_data_present
    }
}

/// individual_channel_stream(common_window).
pub(super) fn write_ics(w: &mut BitWriter, cf: &ChannelFrame, qz: &Quantized, common_window: bool) {
    let start = w.len_bits();
    let layout = &cf.layout;
    let nswb = layout.num_swb();
    w.put(qz.global_gain as u32, 8);
    if !common_window {
        write_ics_info(w, layout, qz.max_sfb);
    }
    for sections in &qz.sections {
        huffman::write_sections(w, sections, layout.short());
    }
    // scale_factor_data(): every band of a non-zero section, differentially
    // along the chain; empty bands inside a coded section repeat the last.
    let mut last = qz.global_gain;
    for g in 0..layout.num_groups() {
        for sfb in 0..qz.max_sfb {
            if qz.band_codebook(g, sfb) == 0 {
                continue;
            }
            let i = g * nswb + sfb;
            let sf = if qz.active[i] { qz.sf[i] } else { last };
            huffman::write_sf(w, sf - last);
            last = sf;
        }
    }
    w.put(0, 1); // pulse_data_present
    w.put(0, 1); // tns_data_present
    w.put(0, 1); // gain_control_data_present
    for (g, sections) in qz.sections.iter().enumerate() {
        for s in sections {
            for sfb in s.start..s.end {
                huffman::write_band(w, s.cb, &qz.q[layout.band(g, sfb)]);
            }
        }
    }
    debug_assert_eq!(
        w.len_bits() - start,
        qz.body_bits
            + if common_window {
                0
            } else {
                ics_info_bits(layout)
            },
        "ICS bit count drifted from the rate loop's estimate"
    );
}

/// The ms_mask_present value and mask for a channel pair: 0 (no M/S), 2
/// (every band) or 1 (explicit mask).
pub(super) fn ms_mask_mode(ms: &[bool], layout: &Layout, max_sfb: usize) -> u32 {
    let nswb = layout.num_swb();
    let used = (0..layout.num_groups())
        .flat_map(|g| (0..max_sfb).map(move |sfb| g * nswb + sfb))
        .map(|i| ms[i]);
    let (mut any, mut all) = (false, true);
    for u in used {
        any |= u;
        all &= u;
    }
    if !any || max_sfb == 0 {
        0
    } else if all {
        2
    } else {
        1
    }
}

pub(super) fn ms_mask_bits(mode: u32, layout: &Layout, max_sfb: usize) -> usize {
    2 + if mode == 1 {
        layout.num_groups() * max_sfb
    } else {
        0
    }
}

pub(super) fn write_ms_mask(
    w: &mut BitWriter,
    mode: u32,
    ms: &[bool],
    layout: &Layout,
    max_sfb: usize,
) {
    w.put(mode, 2);
    if mode == 1 {
        let nswb = layout.num_swb();
        for g in 0..layout.num_groups() {
            for sfb in 0..max_sfb {
                w.put(u32::from(ms[g * nswb + sfb]), 1);
            }
        }
    }
}

/// Bits of a fill_element() carrying `payload` bytes (subclause 8.7): the
/// 4-bit count, its 8-bit escape from 15 bytes on, and the payload.
pub(super) fn fill_element_bits(payload: usize) -> usize {
    3 + 4 + if payload >= 15 { 8 } else { 0 } + 8 * payload
}

/// Largest payload one fill_element() can carry (count 15 + esc_count 255 - 1).
pub(super) const MAX_FILL_PAYLOAD: usize = 269;

/// fill_element() with an EXT_FILL_DATA extension_payload(): the type
/// nibble, the '0000' fill_nibble, then `payload - 1` bytes of '10100101'.
pub(super) fn write_fill_element(w: &mut BitWriter, payload: usize) {
    debug_assert!(payload <= MAX_FILL_PAYLOAD);
    w.put(ID_FIL, 3);
    if payload >= 15 {
        w.put(15, 4);
        w.put((payload - 14) as u32, 8);
    } else {
        w.put(payload as u32, 4);
    }
    if payload > 0 {
        w.put(0b0001, 4); // EXT_FILL_DATA
        w.put(0, 4); // fill_nibble
        for _ in 1..payload {
            w.put(0xA5, 8);
        }
    }
}

/// AudioSpecificConfig for AAC-LC (ISO/IEC 14496-3 1.6.2.1 and
/// GASpecificConfig 4.4.1): audioObjectType 2, the sampling frequency
/// index, the channel configuration, then frameLengthFlag (1024-sample
/// frames), dependsOnCoreCoder and extensionFlag, all zero.
pub fn audio_specific_config(sampling_index: u8, channel_configuration: u8) -> [u8; 2] {
    let v: u16 =
        (2 << 11) | (u16::from(sampling_index) << 7) | (u16::from(channel_configuration) << 3);
    v.to_be_bytes()
}

/// The 7-byte ADTS header (ISO/IEC 13818-7 6.2, no CRC) for one raw data
/// block of `payload_len` bytes. `buffer_fullness` is the 11-bit
/// adts_buffer_fullness (0x7FF for a variable-rate stream).
pub fn adts_header(
    sampling_index: u8,
    channel_configuration: u8,
    payload_len: usize,
    buffer_fullness: u16,
) -> [u8; 7] {
    let frame_len = payload_len + 7;
    assert!(
        frame_len < 1 << 13,
        "ADTS frame of {frame_len} bytes overflows frame_length"
    );
    let mut w = BitWriter::with_capacity(7);
    w.put(0xFFF, 12); // syncword
    w.put(0, 1); // ID: MPEG-4
    w.put(0, 2); // layer
    w.put(1, 1); // protection_absent
    w.put(1, 2); // profile: LC (audioObjectType - 1)
    w.put(u32::from(sampling_index), 4);
    w.put(0, 1); // private_bit
    w.put(u32::from(channel_configuration), 3);
    w.put(0, 1); // original/copy
    w.put(0, 1); // home
    w.put(0, 1); // copyright_identification_bit
    w.put(0, 1); // copyright_identification_start
    w.put(frame_len as u32, 13);
    w.put(u32::from(buffer_fullness.min(0x7FF)), 11);
    w.put(0, 2); // number_of_raw_data_blocks_in_frame
    let bytes = w.into_bytes();
    let mut out = [0u8; 7];
    out.copy_from_slice(&bytes);
    out
}

/// Prefix a raw data block with its ADTS header (for MPEG-TS, which carries
/// AAC as ADTS).
pub fn adts_frame(sampling_index: u8, channel_configuration: u8, raw: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(raw.len() + 7);
    out.extend_from_slice(&adts_header(
        sampling_index,
        channel_configuration,
        raw.len(),
        0x7FF,
    ));
    out.extend_from_slice(raw);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn asc_matches_the_well_known_lc_values() {
        // 48 kHz stereo LC is 0x11 0x90; 44.1 kHz stereo 0x12 0x10; 5.1 at
        // 48 kHz 0x11 0xB0.
        assert_eq!(audio_specific_config(3, 2), [0x11, 0x90]);
        assert_eq!(audio_specific_config(4, 2), [0x12, 0x10]);
        assert_eq!(audio_specific_config(3, 6), [0x11, 0xB0]);
    }

    #[test]
    fn adts_header_fields() {
        let h = adts_header(4, 2, 100, 0x7FF);
        assert_eq!(&h[..2], &[0xFF, 0xF1]);
        // profile 01, sf index 0100, private 0, channel config 010 (high bit 0).
        assert_eq!(h[2], 0b0101_0000);
        let frame_len =
            (u32::from(h[3] & 0x3) << 11) | (u32::from(h[4]) << 3) | u32::from(h[5] >> 5);
        assert_eq!(frame_len, 107);
        assert_eq!(h[3] >> 6, 0b10); // channel config low two bits
    }

    #[test]
    fn fill_element_lengths_match_their_count_field() {
        for payload in [0usize, 1, 14, 15, 16, 200, MAX_FILL_PAYLOAD] {
            let mut w = BitWriter::default();
            write_fill_element(&mut w, payload);
            assert_eq!(w.len_bits(), fill_element_bits(payload), "{payload}");
        }
    }
}
