/// HDR static metadata pulled from the visual sample entry's `mdcv` and
/// `clli` boxes — Squad-21 wires this to ColorMetadata so Squad-20's
/// muxer can round-trip HDR10 mastering display + content light level
/// from any source MP4 / MOV that signals them.
use codec::frame::{ContentLightLevel, MasteringDisplay};

#[derive(Debug, Default, Clone, Copy)]
pub(super) struct Mp4VisualColorMetadata {
    pub(super) mastering_display: Option<MasteringDisplay>,
    pub(super) content_light_level: Option<ContentLightLevel>,
}

/// Walk `moov/trak/mdia/minf/stbl/stsd > {av01, hvc1, hev1, ...}` and
/// pick out the optional `mdcv` and `clli` child boxes.
///
/// Per ISO/IEC 23001-17 (Carriage of static and dynamic metadata in
/// ISOBMFF), `mdcv` and `clli` are direct children of the visual
/// sample entry — same nesting level as `colr`. Layouts:
///
///   `mdcv` body (24 bytes):
///     u16[2] display_primaries[3]   // wire order GBR
///     u16    white_point_x
///     u16    white_point_y
///     u32    max_display_mastering_luminance  (in 0.0001 cd/m²)
///     u32    min_display_mastering_luminance  (in 0.0001 cd/m²)
///
///   `clli` body (4 bytes):
///     u16    max_content_light_level
///     u16    max_pic_average_light_level
pub(super) fn extract_mp4_visual_color_metadata(data: &[u8]) -> Mp4VisualColorMetadata {
    let path: &[&[u8; 4]] = &[b"moov", b"trak", b"mdia", b"minf", b"stbl", b"stsd"];
    let Some(stsd_body) = super::find_box_body(data, path) else {
        return Mp4VisualColorMetadata::default();
    };
    if stsd_body.len() < 16 {
        return Mp4VisualColorMetadata::default();
    }

    let mut pos = 8; // skip version/flags/entry_count
    while pos + 8 <= stsd_body.len() {
        let entry_size = u32::from_be_bytes([
            stsd_body[pos],
            stsd_body[pos + 1],
            stsd_body[pos + 2],
            stsd_body[pos + 3],
        ]) as usize;
        if entry_size < 8 || pos.saturating_add(entry_size) > stsd_body.len() {
            break;
        }
        let entry_type: [u8; 4] = match stsd_body[pos + 4..pos + 8].try_into() {
            Ok(v) => v,
            Err(_) => break,
        };
        // Visual sample entries — mdcv/clli only live under these.
        let is_visual = matches!(
            &entry_type,
            b"av01"
                | b"avc1"
                | b"avc3"
                | b"hvc1"
                | b"hev1"
                | b"hvc2"
                | b"hev2"
                | b"dvh1"
                | b"dvhe"
                | b"vp08"
                | b"vp09"
                | b"apcn"
                | b"apch"
                | b"apcs"
                | b"apco"
                | b"ap4h"
                | b"ap4x"
        );
        if !is_visual {
            pos = pos.saturating_add(entry_size);
            continue;
        }
        let end = pos.saturating_add(entry_size);
        // VisualSampleEntry header: 8-byte box header + 78 bytes of fixed
        // VisualSampleEntry fields before the first child box. Same
        // offset for every visual sample entry kind.
        let child_start = pos + 8 + 78;
        if child_start >= end {
            return Mp4VisualColorMetadata::default();
        }
        let children = &stsd_body[child_start..end];
        let mut out = Mp4VisualColorMetadata::default();
        if let Some(mdcv) = super::find_direct_child(children, b"mdcv") {
            out.mastering_display = parse_mp4_mdcv(mdcv);
        }
        if let Some(clli) = super::find_direct_child(children, b"clli") {
            out.content_light_level = parse_mp4_clli(clli);
        }
        return out;
    }
    Mp4VisualColorMetadata::default()
}

fn parse_mp4_mdcv(body: &[u8]) -> Option<MasteringDisplay> {
    if body.len() < 24 {
        return None;
    }
    let u16be = |o: usize| u16::from_be_bytes([body[o], body[o + 1]]);
    let u32be = |o: usize| u32::from_be_bytes([body[o], body[o + 1], body[o + 2], body[o + 3]]);
    Some(MasteringDisplay {
        // Wire order is GBR per ISO/IEC 23001-17 §7.3.
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

fn parse_mp4_clli(body: &[u8]) -> Option<ContentLightLevel> {
    if body.len() < 4 {
        return None;
    }
    Some(ContentLightLevel {
        max_cll: u16::from_be_bytes([body[0], body[1]]),
        max_fall: u16::from_be_bytes([body[2], body[3]]),
    })
}
