//! HEVC SEI extractor for HDR static metadata.
//!
//! libde265 does not expose SEI messages through its public C API
//! (the `sei_message` type lives in `libde265/libde265/sei.h` as
//! internal C++; only the processing side that hashes pictures is
//! visible), so we vendor a minimal pure-Rust NAL/SEI parser here
//! that reads just the two payload types HDR10 pass-through needs:
//!
//!   * **Mastering display colour volume** — `payload_type=137`,
//!     per HEVC spec D.2.28.
//!   * **Content light level information** — `payload_type=144`,
//!     per HEVC spec D.2.35.
//!
//! Inputs are raw Annex-B samples (start-code delimited NAL units)
//! — the same bytes we push into libde265. Output merges into
//! `ColorMetadata.mastering_display` / `content_light_level`. The
//! parser does not touch the decode path at all; it just scans
//! the bitstream once at demux / decoder-construction time and
//! caches the two structs.
//!
//! Referenced normative text:
//!
//!   * ITU-T H.265 (2021) §7.3.5 "SEI payload syntax"
//!   * ITU-T H.265 (2021) D.2.28 mastering_display_colour_volume()
//!   * ITU-T H.265 (2021) D.2.35 content_light_level_info()
//!   * ITU-T H.265 (2021) §7.3.2.4 "SEI RBSP syntax" — NAL units 39/40
//!
//! Anti-emulation: the SEI RBSP uses emulation-prevention byte stuffing
//! (any `0x00 0x00 0x00`, `0x00 0x00 0x01`, `0x00 0x00 0x02`, or
//! `0x00 0x00 0x03` in the decoded payload is written as the original
//! first two bytes followed by a `0x03` in the coded bitstream). For
//! the two payload types we parse, all fields are single bytes or
//! 16/32-bit BE words whose payload lengths are fixed; we strip
//! emulation-prevention bytes out on a per-NAL basis before parsing.

use crate::frame::{ContentLightLevel, MasteringDisplay};

#[derive(Debug, Clone, Copy, Default)]
pub struct HevcHdrSei {
    pub mastering_display: Option<MasteringDisplay>,
    pub content_light_level: Option<ContentLightLevel>,
}

impl HevcHdrSei {
    /// Fold `other` into `self`: later samples overwrite earlier ones
    /// only when they populate a field the current state lacks OR when
    /// the payload differs (most streams repeat the SEI on every
    /// IRAP; folding keeps the newest).
    pub fn merge(&mut self, other: HevcHdrSei) {
        if other.mastering_display.is_some() {
            self.mastering_display = other.mastering_display;
        }
        if other.content_light_level.is_some() {
            self.content_light_level = other.content_light_level;
        }
    }

    pub fn is_empty(&self) -> bool {
        self.mastering_display.is_none() && self.content_light_level.is_none()
    }
}

/// Scan an Annex-B byte buffer for HEVC SEI NAL units (nal_unit_type 39
/// prefix, 40 suffix) and extract HDR static metadata payloads.
/// Returns a potentially-empty `HevcHdrSei`; callers should fold it
/// into `ColorMetadata` only when non-empty.
pub fn parse_annexb(buf: &[u8]) -> HevcHdrSei {
    let mut out = HevcHdrSei::default();
    for nal in annexb_split(buf) {
        if nal.is_empty() {
            continue;
        }
        // HEVC NAL unit header (2 bytes): forbidden_zero_bit(1) |
        // nal_unit_type(6) | nuh_layer_id(6) | nuh_temporal_id_plus1(3).
        // We care about types 39 (PREFIX_SEI_NUT) and 40 (SUFFIX_SEI_NUT).
        if nal.len() < 2 {
            continue;
        }
        let nal_unit_type = (nal[0] >> 1) & 0x3F;
        if nal_unit_type != 39 && nal_unit_type != 40 {
            continue;
        }
        let rbsp = strip_emulation_prevention(&nal[2..]);
        parse_sei_rbsp(&rbsp, &mut out);
    }
    out
}

/// Iterator: split an Annex-B byte buffer into NAL payloads (start
/// codes and trailing zero-byte fillers removed). Start codes are
/// `0x00 0x00 0x01` (3 bytes) or `0x00 0x00 0x00 0x01` (4 bytes).
fn annexb_split(buf: &[u8]) -> Vec<&[u8]> {
    let mut out = Vec::new();
    let mut i = 0;
    // Advance to first start code.
    while i + 2 < buf.len() {
        if buf[i] == 0 && buf[i + 1] == 0 && buf[i + 2] == 1 {
            i += 3;
            break;
        }
        if i + 3 < buf.len() && buf[i] == 0 && buf[i + 1] == 0 && buf[i + 2] == 0 && buf[i + 3] == 1
        {
            i += 4;
            break;
        }
        i += 1;
    }
    let mut nal_start = i;
    while i + 2 < buf.len() {
        if buf[i] == 0 && buf[i + 1] == 0 && (buf[i + 2] == 1 || buf[i + 2] == 0) {
            // Check for a true start code (may be 3 or 4 bytes).
            let is_3byte = buf[i + 2] == 1;
            let is_4byte = !is_3byte && i + 3 < buf.len() && buf[i + 3] == 1;
            if is_3byte || is_4byte {
                let mut end = i;
                // Trim trailing zero fill before the start code.
                while end > nal_start && buf[end - 1] == 0 {
                    end -= 1;
                }
                if end > nal_start {
                    out.push(&buf[nal_start..end]);
                }
                i += if is_3byte { 3 } else { 4 };
                nal_start = i;
                continue;
            }
        }
        i += 1;
    }
    if nal_start < buf.len() {
        let mut end = buf.len();
        while end > nal_start && buf[end - 1] == 0 {
            end -= 1;
        }
        if end > nal_start {
            out.push(&buf[nal_start..end]);
        }
    }
    out
}

/// Remove HEVC emulation-prevention bytes (a `0x03` inserted after any
/// `0x00 0x00` pair whose next original byte was ≤ 0x03). Input is an
/// EBSP slice; output is the underlying RBSP.
fn strip_emulation_prevention(ebsp: &[u8]) -> Vec<u8> {
    let mut rbsp = Vec::with_capacity(ebsp.len());
    let mut i = 0;
    while i < ebsp.len() {
        if i + 2 < ebsp.len() && ebsp[i] == 0 && ebsp[i + 1] == 0 && ebsp[i + 2] == 0x03 {
            rbsp.push(0);
            rbsp.push(0);
            i += 3;
            continue;
        }
        rbsp.push(ebsp[i]);
        i += 1;
    }
    rbsp
}

/// Parse one SEI RBSP: a concatenation of
///   `(payload_type, payload_size, payload_bytes)`
/// triples, ending with an `rbsp_trailing_bits()` byte. Each `_type` /
/// `_size` is variable-length via 0xFF-run encoding:
///   payload_type = sum of leading 0xFF bytes + final non-0xFF byte.
fn parse_sei_rbsp(rbsp: &[u8], out: &mut HevcHdrSei) {
    let mut cursor = 0;
    while cursor < rbsp.len() {
        // payload_type
        let (payload_type, after_type) = match read_sei_ff_byte_sum(rbsp, cursor) {
            Some(v) => v,
            None => return,
        };
        cursor = after_type;
        if cursor >= rbsp.len() {
            return;
        }
        // payload_size
        let (payload_size, after_size) = match read_sei_ff_byte_sum(rbsp, cursor) {
            Some(v) => v,
            None => return,
        };
        cursor = after_size;
        if cursor + payload_size > rbsp.len() {
            return;
        }
        let payload = &rbsp[cursor..cursor + payload_size];
        cursor += payload_size;

        match payload_type {
            137 => {
                if let Some(mdcv) = parse_mastering_display(payload) {
                    out.mastering_display = Some(mdcv);
                }
            }
            144 => {
                if let Some(clli) = parse_content_light_level(payload) {
                    out.content_light_level = Some(clli);
                }
            }
            _ => { /* ignore other payload types */ }
        }

        // Check for rbsp_trailing_bits: a single `1` bit followed by
        // zeros. Any remaining byte at `cursor` that's 0x80 or similar
        // marks the end; SEI RBSPs rarely concatenate multiple messages
        // without a trailing bit, but the spec allows it.
        if cursor < rbsp.len() && rbsp[cursor] == 0x80 {
            break;
        }
    }
}

/// Sum leading 0xFF bytes with the first non-0xFF byte, yielding the
/// SEI payload_type or payload_size field. Returns `(value, next_idx)`.
fn read_sei_ff_byte_sum(buf: &[u8], mut idx: usize) -> Option<(usize, usize)> {
    let mut acc = 0usize;
    while idx < buf.len() && buf[idx] == 0xFF {
        acc += 0xFF;
        idx += 1;
    }
    if idx >= buf.len() {
        return None;
    }
    acc += buf[idx] as usize;
    Some((acc, idx + 1))
}

/// Parse mastering_display_colour_volume() per HEVC D.2.28.
///
/// Payload (big-endian):
///   u16 display_primaries_x[0], u16 display_primaries_y[0]   // G
///   u16 display_primaries_x[1], u16 display_primaries_y[1]   // B
///   u16 display_primaries_x[2], u16 display_primaries_y[2]   // R
///   u16 white_point_x,          u16 white_point_y
///   u32 max_display_mastering_luminance
///   u32 min_display_mastering_luminance
///
/// = 24 bytes. Note the spec-defined wire order is GBR (not RGB); we
/// remap to the struct's R/G/B field order.
fn parse_mastering_display(p: &[u8]) -> Option<MasteringDisplay> {
    if p.len() < 24 {
        return None;
    }
    let u16be = |o: usize| u16::from_be_bytes([p[o], p[o + 1]]);
    let u32be = |o: usize| u32::from_be_bytes([p[o], p[o + 1], p[o + 2], p[o + 3]]);
    Some(MasteringDisplay {
        // Wire order GBR → struct field RGB remap.
        primaries_g_x: u16be(0),
        primaries_g_y: u16be(2),
        primaries_b_x: u16be(4),
        primaries_b_y: u16be(6),
        primaries_r_x: u16be(8),
        primaries_r_y: u16be(10),
        white_point_x: u16be(12),
        white_point_y: u16be(14),
        max_luminance: u32be(16),
        min_luminance: u32be(20),
    })
}

/// Parse content_light_level_info() per HEVC D.2.35.
///
/// Payload (big-endian):
///   u16 max_content_light_level
///   u16 max_pic_average_light_level
///
/// = 4 bytes.
fn parse_content_light_level(p: &[u8]) -> Option<ContentLightLevel> {
    if p.len() < 4 {
        return None;
    }
    Some(ContentLightLevel {
        max_cll: u16::from_be_bytes([p[0], p[1]]),
        max_fall: u16::from_be_bytes([p[2], p[3]]),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn emit_sei_payload(payload_type: u8, payload: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        out.push(payload_type);
        out.push(payload.len() as u8);
        out.extend_from_slice(payload);
        // rbsp_trailing_bits: single '1' bit then zero-fill to byte.
        out.push(0x80);
        out
    }

    fn wrap_as_prefix_sei_nal(rbsp: &[u8]) -> Vec<u8> {
        // NAL header type=39 (PREFIX_SEI_NUT), layer_id=0, tid+1=1.
        // byte[0] = (0<<7) | (39<<1) | ((0 >> 5) & 1)   = 0x4E
        // byte[1] = ((0 & 0x1F) << 3) | 1               = 0x01
        let mut v = Vec::with_capacity(2 + rbsp.len());
        v.push(0x4E);
        v.push(0x01);
        v.extend_from_slice(rbsp);
        v
    }

    fn mastering_display_sei_bytes() -> Vec<u8> {
        // BT.2020 primaries (HDR10 canonical):
        //   G = (0.265, 0.690) → (13250, 34500)
        //   B = (0.150, 0.060) → ( 7500,  3000)
        //   R = (0.680, 0.320) → (34000, 16000)
        //   W = (0.3127, 0.3290) → (15635, 16450)
        // max_luminance = 1000 cd/m² in 0.0001 units → 10_000_000
        // min_luminance = 0.005 cd/m² in 0.0001 units →         50
        let mut p = Vec::new();
        p.extend_from_slice(&13250u16.to_be_bytes());
        p.extend_from_slice(&34500u16.to_be_bytes());
        p.extend_from_slice(&7500u16.to_be_bytes());
        p.extend_from_slice(&3000u16.to_be_bytes());
        p.extend_from_slice(&34000u16.to_be_bytes());
        p.extend_from_slice(&16000u16.to_be_bytes());
        p.extend_from_slice(&15635u16.to_be_bytes());
        p.extend_from_slice(&16450u16.to_be_bytes());
        p.extend_from_slice(&10_000_000u32.to_be_bytes());
        p.extend_from_slice(&50u32.to_be_bytes());
        assert_eq!(p.len(), 24);
        p
    }

    fn content_light_level_sei_bytes() -> Vec<u8> {
        // MaxCLL = 1000, MaxFALL = 400.
        let mut p = Vec::new();
        p.extend_from_slice(&1000u16.to_be_bytes());
        p.extend_from_slice(&400u16.to_be_bytes());
        p
    }

    fn build_annexb(nals: &[&[u8]]) -> Vec<u8> {
        let mut out = Vec::new();
        for nal in nals {
            out.extend_from_slice(&[0, 0, 0, 1]);
            out.extend_from_slice(nal);
        }
        out
    }

    #[test]
    fn parses_mastering_display_sei_from_prefix_nal() {
        let rbsp = emit_sei_payload(137, &mastering_display_sei_bytes());
        let nal = wrap_as_prefix_sei_nal(&rbsp);
        let stream = build_annexb(&[&nal]);
        let sei = parse_annexb(&stream);
        let md = sei.mastering_display.expect("mastering display populated");
        assert_eq!(md.primaries_r_x, 34000);
        assert_eq!(md.primaries_r_y, 16000);
        assert_eq!(md.primaries_g_x, 13250);
        assert_eq!(md.primaries_g_y, 34500);
        assert_eq!(md.primaries_b_x, 7500);
        assert_eq!(md.primaries_b_y, 3000);
        assert_eq!(md.white_point_x, 15635);
        assert_eq!(md.white_point_y, 16450);
        assert_eq!(md.max_luminance, 10_000_000);
        assert_eq!(md.min_luminance, 50);
        assert!(sei.content_light_level.is_none());
    }

    #[test]
    fn parses_content_light_level_sei_from_prefix_nal() {
        let rbsp = emit_sei_payload(144, &content_light_level_sei_bytes());
        let nal = wrap_as_prefix_sei_nal(&rbsp);
        let stream = build_annexb(&[&nal]);
        let sei = parse_annexb(&stream);
        let cll = sei.content_light_level.expect("clli populated");
        assert_eq!(cll.max_cll, 1000);
        assert_eq!(cll.max_fall, 400);
        assert!(sei.mastering_display.is_none());
    }

    #[test]
    fn parses_both_sei_messages_in_same_nal() {
        let mut rbsp = emit_sei_payload(137, &mastering_display_sei_bytes());
        // Drop the earlier rbsp_trailing_bits — the second payload follows
        // directly — then emit the second message with its own trailing.
        rbsp.pop();
        rbsp.extend(emit_sei_payload(144, &content_light_level_sei_bytes()));
        let nal = wrap_as_prefix_sei_nal(&rbsp);
        let stream = build_annexb(&[&nal]);
        let sei = parse_annexb(&stream);
        assert!(sei.mastering_display.is_some());
        assert!(sei.content_light_level.is_some());
    }

    #[test]
    fn handles_emulation_prevention_bytes() {
        // Hand-craft a payload that contains an embedded 0x00 0x00 0x01
        // sequence — inject a 0x03 emulation-prevention byte before the
        // trailing 0x01 so the encoder output parses cleanly. The parser
        // must strip the 0x03 before reading the u32.
        //
        // Place the sensitive sequence inside the CLLI payload:
        //   MaxCLL = 0x0000 → 0x00 0x00
        //   MaxFALL = 0x0001 → 0x00 0x01
        // Full payload bytes: 0x00 0x00 0x00 0x01 → must be encoded as
        // 0x00 0x00 0x03 0x00 0x01 in the EBSP.
        let payload = vec![0x00, 0x00, 0x00, 0x01];
        let mut rbsp_without_prevention = Vec::new();
        rbsp_without_prevention.push(144); // payload_type
        rbsp_without_prevention.push(payload.len() as u8); // payload_size
        rbsp_without_prevention.extend_from_slice(&payload);
        rbsp_without_prevention.push(0x80); // rbsp_trailing_bits

        // Now produce the EBSP by inserting 0x03 after any two zero bytes
        // whose next byte is ≤ 0x03.
        let mut ebsp = Vec::new();
        let mut zero_run = 0;
        for &b in &rbsp_without_prevention {
            if zero_run >= 2 && b <= 0x03 {
                ebsp.push(0x03);
                zero_run = 0;
            }
            ebsp.push(b);
            if b == 0 {
                zero_run += 1;
            } else {
                zero_run = 0;
            }
        }

        // Wrap with NAL header and Annex-B start code.
        let mut nal = vec![0x4E, 0x01];
        nal.extend_from_slice(&ebsp);
        let stream = build_annexb(&[&nal]);
        let sei = parse_annexb(&stream);
        let cll = sei.content_light_level.expect("clli after emulation strip");
        assert_eq!(cll.max_cll, 0);
        assert_eq!(cll.max_fall, 1);
    }

    #[test]
    fn returns_empty_when_no_sei_nal_present() {
        // A random VCL NAL (type 1, non-IDR slice). Parser must skip.
        let mut nal = vec![0x02, 0x01]; // (1 << 1) = 0x02
        nal.extend_from_slice(&[0xFF, 0xFF, 0xFF]);
        let stream = build_annexb(&[&nal]);
        let sei = parse_annexb(&stream);
        assert!(sei.mastering_display.is_none());
        assert!(sei.content_light_level.is_none());
        assert!(sei.is_empty());
    }

    #[test]
    fn handles_start_code_4byte_variant() {
        let rbsp = emit_sei_payload(144, &content_light_level_sei_bytes());
        let nal = wrap_as_prefix_sei_nal(&rbsp);
        // 4-byte start code form: 0x00 0x00 0x00 0x01.
        let mut stream = vec![0, 0, 0, 1];
        stream.extend_from_slice(&nal);
        let sei = parse_annexb(&stream);
        assert!(sei.content_light_level.is_some());
    }

    #[test]
    fn suffix_sei_nal_type_40_also_parsed() {
        let rbsp = emit_sei_payload(144, &content_light_level_sei_bytes());
        // NAL type 40 (SUFFIX_SEI_NUT):
        //   byte[0] = (40 << 1) | 0 = 0x50
        //   byte[1] = 0x01
        let mut nal = vec![0x50, 0x01];
        nal.extend_from_slice(&rbsp);
        let stream = build_annexb(&[&nal]);
        let sei = parse_annexb(&stream);
        assert!(sei.content_light_level.is_some());
    }

    #[test]
    fn ff_byte_sum_handles_large_payload_type() {
        // payload_type = 255 + 7 = 262 (fictional type; parser must skip).
        // Verify the 0xFF run-length decode advances the cursor correctly.
        let mut rbsp = vec![0xFF, 7, /* size */ 0, /* trailing */ 0x80];
        // Append a valid clli after.
        rbsp.pop();
        rbsp.extend(emit_sei_payload(144, &content_light_level_sei_bytes()));
        let nal = wrap_as_prefix_sei_nal(&rbsp);
        let stream = build_annexb(&[&nal]);
        let sei = parse_annexb(&stream);
        assert!(sei.content_light_level.is_some());
    }
}
