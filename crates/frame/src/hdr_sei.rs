//! HDR10 static metadata carried as SEI: mastering display colour volume
//! (`payload_type` 137) and content light level (`payload_type` 144), in an
//! H.264 or an HEVC Annex-B byte stream.
//!
//! A container may say nothing about HDR static metadata — no MP4 `mdcv` /
//! `clli`, no Matroska `MasteringMetadata` / `MaxCLL`, and MPEG-TS or AVI have
//! nowhere to put it — while the stream itself carries both messages in its
//! first IRAP access unit (x265 `hdr10=1` does exactly this). The demuxers read
//! them from there, so the values reach the encoders' SEIs and the output's
//! boxes. It lives in `rivet-frame` rather than `rivet-codec` because the
//! demuxers need it and `rivet-container` does not depend on the codec crate;
//! `codec::hevc_sei` re-exports it at its old path.
//!
//! The two payloads have the same syntax in both codecs (H.265 D.2.28 / D.2.35,
//! H.264 D.1.29 / D.1.31); only the SEI NAL unit differs: HEVC types 39
//! (prefix) and 40 (suffix) behind a two-byte header, H.264 type 6 behind a
//! one-byte header.
//!
//! Anti-emulation: the SEI RBSP uses emulation-prevention byte stuffing (a
//! `0x03` after any `0x00 0x00` whose next original byte is `<= 0x03`); it is
//! stripped per NAL unit before parsing.

use crate::{ContentLightLevel, MasteringDisplay};

/// The HDR10 static metadata found in SEI messages.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct HdrSei {
    /// `mastering_display_colour_volume` (payload type 137).
    pub mastering_display: Option<MasteringDisplay>,
    /// `content_light_level_info` (payload type 144).
    pub content_light_level: Option<ContentLightLevel>,
}

/// The name this type had in `codec::hevc_sei`, when it read HEVC only.
pub type HevcHdrSei = HdrSei;

impl HdrSei {
    /// Fold `other` into `self`: each field `other` populates replaces the
    /// current one (streams repeat the SEI on every IRAP; the newest wins).
    pub fn merge(&mut self, other: HdrSei) {
        if other.mastering_display.is_some() {
            self.mastering_display = other.mastering_display;
        }
        if other.content_light_level.is_some() {
            self.content_light_level = other.content_light_level;
        }
    }

    /// Neither message was found.
    pub fn is_empty(&self) -> bool {
        self.mastering_display.is_none() && self.content_light_level.is_none()
    }
}

/// Scan an HEVC Annex-B byte buffer (SEI NAL unit types 39 and 40).
pub fn parse_annexb(buf: &[u8]) -> HdrSei {
    scan(buf, 2, |nal| matches!((nal[0] >> 1) & 0x3F, 39 | 40))
}

/// Scan an H.264 Annex-B byte buffer (SEI NAL unit type 6).
pub fn parse_h264_annexb(buf: &[u8]) -> HdrSei {
    scan(buf, 1, |nal| nal[0] & 0x1F == 6)
}

/// Scan an Annex-B byte buffer of the codec named by `codec` (the labels the
/// demuxers use: `h264` / `avc*`, `h265` / `hevc` / `hvc1` / `hev1`). Any other
/// codec has no H.264 / HEVC SEI to read and gives an empty result.
pub fn parse_annexb_for(codec: &str, buf: &[u8]) -> HdrSei {
    match codec.to_ascii_lowercase().as_str() {
        "h264" | "avc" | "avc1" | "avc3" => parse_h264_annexb(buf),
        "h265" | "hevc" | "hvc1" | "hev1" | "hvc2" | "hev2" => parse_annexb(buf),
        _ => HdrSei::default(),
    }
}

/// The HDR10 static metadata in an AV1 temporal unit's metadata OBUs
/// (`OBU_METADATA`, obu_type 5; AV1 §5.8.2 / §6.7.3 / §6.7.4) — the AV1
/// counterpart of SEI 137 / 144: `METADATA_TYPE_HDR_CLL` (1) and
/// `METADATA_TYPE_HDR_MDCV` (2). The values are given in the SEI's units,
/// which [`MasteringDisplay`] holds, as libavcodec converts them: a
/// chromaticity from AV1's 0.16 fixed point to 0.00002 steps, the maximum
/// luminance from 24.8 and the minimum from 18.14 fixed point to 0.0001
/// cd/m². AV1 lists the primaries R, G, B.
///
/// `buf` is a sequence of low-overhead-format OBUs (`obu_has_size_field`
/// set, as in MP4 / Matroska / IVF); an OBU without a size field is taken to
/// run to the end of `buf`.
pub fn parse_av1_obus(buf: &[u8]) -> HdrSei {
    let mut out = HdrSei::default();
    let mut at = 0usize;
    while at < buf.len() {
        let header = buf[at];
        let obu_type = (header >> 3) & 0x0F;
        let mut pos = at + 1 + usize::from(header & 0x04 != 0);
        let size = if header & 0x02 != 0 {
            let Some((size, len)) = leb128(&buf[pos.min(buf.len())..]) else {
                break;
            };
            pos += len;
            size
        } else {
            buf.len().saturating_sub(pos)
        };
        let Some(payload) = buf.get(pos..pos.saturating_add(size)) else {
            break;
        };
        if obu_type == 5 {
            av1_metadata(payload, &mut out);
        }
        at = pos + size;
    }
    out
}

/// One `metadata_obu()` payload into `out`.
fn av1_metadata(payload: &[u8], out: &mut HdrSei) {
    let Some((kind, len)) = leb128(payload) else {
        return;
    };
    let body = &payload[len..];
    let u16_at = |o: usize| body.get(o..o + 2).map(|b| u16::from_be_bytes([b[0], b[1]]));
    let u32_at = |o: usize| {
        body.get(o..o + 4)
            .map(|b| u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
    };
    match kind {
        1 => {
            if let (Some(max_cll), Some(max_fall)) = (u16_at(0), u16_at(2)) {
                out.content_light_level = Some(ContentLightLevel { max_cll, max_fall });
            }
        }
        2 => {
            // 0.16 fixed point -> 0.00002 steps.
            let xy = |o: usize| u16_at(o).map(|v| ((u32::from(v) * 50_000 + 32_768) >> 16) as u16);
            let (Some(rx), Some(ry), Some(gx), Some(gy), Some(bx), Some(by), Some(wx), Some(wy)) =
                (xy(0), xy(2), xy(4), xy(6), xy(8), xy(10), xy(12), xy(14))
            else {
                return;
            };
            let (Some(max), Some(min)) = (u32_at(16), u32_at(20)) else {
                return;
            };
            out.mastering_display = Some(MasteringDisplay {
                primaries_r_x: rx,
                primaries_r_y: ry,
                primaries_g_x: gx,
                primaries_g_y: gy,
                primaries_b_x: bx,
                primaries_b_y: by,
                white_point_x: wx,
                white_point_y: wy,
                // 24.8 -> 0.0001 cd/m2: x 10000 / 256; 18.14: x 10000 / 16384.
                max_luminance: ((u64::from(max) * 10_000 + 128) >> 8) as u32,
                min_luminance: ((u64::from(min) * 10_000 + 8_192) >> 14) as u32,
            });
        }
        _ => {}
    }
}

/// An unsigned LEB128 value at the start of `buf` and its length in bytes
/// (AV1 §4.10.5: at most eight bytes).
fn leb128(buf: &[u8]) -> Option<(usize, usize)> {
    let mut value = 0u64;
    for (i, &b) in buf.iter().take(8).enumerate() {
        value |= u64::from(b & 0x7F) << (7 * i);
        if b & 0x80 == 0 {
            return Some((usize::try_from(value).ok()?, i + 1));
        }
    }
    None
}

/// Every NAL unit of `buf` that `is_sei` accepts is unescaped past its
/// `header_len`-byte header and parsed as an SEI RBSP.
fn scan(buf: &[u8], header_len: usize, is_sei: fn(&[u8]) -> bool) -> HdrSei {
    let mut out = HdrSei::default();
    for nal in annexb_split(buf) {
        if nal.len() <= header_len || !is_sei(nal) {
            continue;
        }
        let rbsp = strip_emulation_prevention(&nal[header_len..]);
        parse_sei_rbsp(&rbsp, &mut out);
    }
    out
}

/// Split an Annex-B byte buffer into NAL payloads (start codes and trailing
/// zero-byte fillers removed). Start codes are `0x00 0x00 0x01` (3 bytes) or
/// `0x00 0x00 0x00 0x01` (4 bytes).
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

/// Remove emulation-prevention bytes (a `0x03` inserted after any `0x00 0x00`
/// pair whose next original byte was `<= 0x03`). Input is an EBSP slice;
/// output is the underlying RBSP.
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

/// Parse one SEI RBSP: a concatenation of `(payload_type, payload_size,
/// payload_bytes)` triples, ending with an `rbsp_trailing_bits()` byte. Each
/// `_type` / `_size` is variable-length via 0xFF-run encoding: the sum of the
/// leading 0xFF bytes plus the final non-0xFF byte.
fn parse_sei_rbsp(rbsp: &[u8], out: &mut HdrSei) {
    let mut cursor = 0;
    while cursor < rbsp.len() {
        let Some((payload_type, after_type)) = read_sei_ff_byte_sum(rbsp, cursor) else {
            return;
        };
        cursor = after_type;
        if cursor >= rbsp.len() {
            return;
        }
        let Some((payload_size, after_size)) = read_sei_ff_byte_sum(rbsp, cursor) else {
            return;
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
            _ => {}
        }

        // rbsp_trailing_bits: a single `1` bit followed by zeros.
        if cursor < rbsp.len() && rbsp[cursor] == 0x80 {
            break;
        }
    }
}

/// Sum leading 0xFF bytes with the first non-0xFF byte, yielding the SEI
/// payload_type or payload_size field. Returns `(value, next_idx)`.
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

/// Parse mastering_display_colour_volume() (H.265 D.2.28, H.264 D.1.29).
///
/// Payload (big-endian), 24 bytes: `display_primaries_x/y[0..3]` in wire
/// order G, B, R; `white_point_x/y`; u32 `max_display_mastering_luminance`;
/// u32 `min_display_mastering_luminance`.
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

/// Parse content_light_level_info() (H.265 D.2.35, H.264 D.1.31): u16
/// `max_content_light_level`, u16 `max_pic_average_light_level`.
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
        let mut v = Vec::with_capacity(2 + rbsp.len());
        v.push(0x4E);
        v.push(0x01);
        v.extend_from_slice(rbsp);
        v
    }

    fn mastering_display_sei_bytes() -> Vec<u8> {
        // BT.2020 primaries (HDR10 canonical), wire order G, B, R, W;
        // max 1000 cd/m² (10_000_000), min 0.005 cd/m² (50).
        let mut p = Vec::new();
        for v in [13250u16, 34500, 7500, 3000, 34000, 16000, 15635, 16450] {
            p.extend_from_slice(&v.to_be_bytes());
        }
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
        // Drop the first message's trailing bits; the second follows directly.
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
        // CLLI payload 0x00 0x00 0x00 0x01 must be coded 0x00 0x00 0x03 0x00 0x01.
        let payload = vec![0x00, 0x00, 0x00, 0x01];
        let mut rbsp_without_prevention = vec![144, payload.len() as u8];
        rbsp_without_prevention.extend_from_slice(&payload);
        rbsp_without_prevention.push(0x80);

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
        // A VCL NAL (type 1, non-IDR slice). Parser must skip.
        let mut nal = vec![0x02, 0x01];
        nal.extend_from_slice(&[0xFF, 0xFF, 0xFF]);
        let stream = build_annexb(&[&nal]);
        let sei = parse_annexb(&stream);
        assert!(sei.is_empty());
    }

    #[test]
    fn handles_start_code_4byte_variant() {
        let rbsp = emit_sei_payload(144, &content_light_level_sei_bytes());
        let nal = wrap_as_prefix_sei_nal(&rbsp);
        let mut stream = vec![0, 0, 0, 1];
        stream.extend_from_slice(&nal);
        let sei = parse_annexb(&stream);
        assert!(sei.content_light_level.is_some());
    }

    #[test]
    fn suffix_sei_nal_type_40_also_parsed() {
        let rbsp = emit_sei_payload(144, &content_light_level_sei_bytes());
        // NAL type 40 (SUFFIX_SEI_NUT): byte[0] = 40 << 1 = 0x50.
        let mut nal = vec![0x50, 0x01];
        nal.extend_from_slice(&rbsp);
        let stream = build_annexb(&[&nal]);
        let sei = parse_annexb(&stream);
        assert!(sei.content_light_level.is_some());
    }

    #[test]
    fn ff_byte_sum_handles_large_payload_type() {
        // payload_type = 255 + 7 = 262 (fictional; skipped), size 0, then a clli.
        let mut rbsp = vec![0xFF, 7, 0];
        rbsp.extend(emit_sei_payload(144, &content_light_level_sei_bytes()));
        let nal = wrap_as_prefix_sei_nal(&rbsp);
        let stream = build_annexb(&[&nal]);
        let sei = parse_annexb(&stream);
        assert!(sei.content_light_level.is_some());
    }

    /// H.264 carries the same two payloads in NAL type 6 behind a one-byte
    /// header; the HEVC scan must not read it and vice versa.
    #[test]
    fn h264_sei_nal_type_6_parses_and_the_codecs_do_not_cross() {
        let mut rbsp = emit_sei_payload(137, &mastering_display_sei_bytes());
        rbsp.pop();
        rbsp.extend(emit_sei_payload(144, &content_light_level_sei_bytes()));
        let mut h264_nal = vec![0x06];
        h264_nal.extend_from_slice(&rbsp);
        let h264 = build_annexb(&[&h264_nal]);
        let sei = parse_h264_annexb(&h264);
        assert_eq!(
            sei.mastering_display.map(|m| m.max_luminance),
            Some(10_000_000)
        );
        assert_eq!(
            sei.content_light_level.map(|c| (c.max_cll, c.max_fall)),
            Some((1000, 400))
        );
        assert_eq!(parse_annexb_for("h264", &h264), sei);
        assert_eq!(parse_annexb_for("avc1", &h264), sei);
        assert!(
            parse_annexb(&h264).is_empty(),
            "0x06 is HEVC type 3, not an SEI"
        );

        let hevc = build_annexb(&[&wrap_as_prefix_sei_nal(&rbsp)]);
        assert!(
            parse_h264_annexb(&hevc).is_empty(),
            "0x4E is H.264 type 14, not an SEI"
        );
        assert_eq!(parse_annexb_for("hevc", &hevc), sei);
        assert!(parse_annexb_for("av1", &hevc).is_empty());
    }

    /// One metadata OBU with a size field: `obu_type` 5, `metadata_type`, body.
    fn metadata_obu(kind: u8, body: &[u8]) -> Vec<u8> {
        let mut obu = vec![5 << 3 | 0x02, (body.len() + 1) as u8, kind];
        obu.extend_from_slice(body);
        obu
    }

    #[test]
    fn av1_metadata_obus_give_the_hdr10_values_in_sei_units() {
        // CLL 1234 / 567, then an MDCV of the HEVC fixtures' mastering display
        // in AV1's fixed point: R (0.680, 0.320), G (0.265, 0.690), B (0.150,
        // 0.060), white (0.3127, 0.3290), 4000 and 0.005 cd/m2.
        let fixed16 = |v: f64| ((v * 65536.0).round() as u16).to_be_bytes();
        let mut mdcv = Vec::new();
        for v in [0.680, 0.320, 0.265, 0.690, 0.150, 0.060, 0.3127, 0.3290] {
            mdcv.extend_from_slice(&fixed16(v));
        }
        mdcv.extend_from_slice(&(4000u32 << 8).to_be_bytes());
        mdcv.extend_from_slice(&((0.005f64 * 16384.0).round() as u32).to_be_bytes());
        // A temporal delimiter and a padding OBU around them are skipped.
        let mut tu = vec![0x12, 0x00];
        tu.extend(metadata_obu(1, &[0x04, 0xD2, 0x02, 0x37]));
        tu.extend([15 << 3 | 0x02, 2, 0xAA, 0xBB]);
        tu.extend(metadata_obu(2, &mdcv));
        let sei = parse_av1_obus(&tu);
        assert_eq!(
            sei.content_light_level,
            Some(ContentLightLevel {
                max_cll: 1234,
                max_fall: 567
            })
        );
        assert_eq!(
            sei.mastering_display,
            Some(MasteringDisplay {
                primaries_r_x: 34000,
                primaries_r_y: 16000,
                primaries_g_x: 13250,
                primaries_g_y: 34500,
                primaries_b_x: 7500,
                primaries_b_y: 3000,
                white_point_x: 15635,
                white_point_y: 16450,
                max_luminance: 40_000_000,
                min_luminance: 50,
            })
        );
        // A truncated OBU ends the walk without a value.
        assert!(parse_av1_obus(&[5 << 3 | 0x02, 40, 1, 0x04]).is_empty());
    }
}
