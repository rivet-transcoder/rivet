/// Colour signalling pulled from an MP4 / MOV visual sample entry — the
/// `colr` box (`nclx` / `nclc`: primaries, transfer, matrix, range) and the
/// HDR static metadata in `mdcv` / `clli` — plus the fallback to the
/// bitstream's own SPS VUI when the container carries no `colr`.
///
/// Until 2026-08-27 only `mdcv` / `clli` were read and the transfer stayed
/// at the SDR default, so an HDR MP4 was never tonemapped while the same
/// clip remuxed to MKV (whose `Colour` element the MKV demuxer reads) was.
/// ffmpeg's MP4 muxer writes no `colr` unless asked (`-movflags
/// +write_colr`), so most HDR MP4s in the wild signal their transfer only
/// in the SPS VUI — hence the fallback.
use frame::{ColorSpace, ContentLightLevel, MasteringDisplay, StreamInfo, TransferFn};

#[derive(Debug, Default, Clone, Copy)]
pub(super) struct Mp4VisualColorMetadata {
    pub(super) mastering_display: Option<MasteringDisplay>,
    pub(super) content_light_level: Option<ContentLightLevel>,
    /// The `colr` box's H.273 triple, when the sample entry has one.
    pub(super) nclx: Option<Nclx>,
}

/// An H.273 colour description: `colour_primaries`,
/// `transfer_characteristics`, `matrix_coefficients`, `full_range_flag`.
/// From a `colr` box (`nclx` / `nclc`) or from an SPS VUI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Nclx {
    pub(crate) primaries: u8,
    pub(crate) transfer: u8,
    pub(crate) matrix: u8,
    pub(crate) full_range: bool,
}

impl Nclx {
    /// Whether the description says anything: a triple of all-`2`
    /// (unspecified) carries no information and must not override a
    /// better source.
    fn is_specified(&self) -> bool {
        self.primaries != 2 || self.transfer != 2 || self.matrix != 2
    }
}

/// Parse a `colr` box body. `nclx` (ISO/IEC 14496-12) and `nclc`
/// (QuickTime) share the layout: u16 primaries, u16 transfer, u16 matrix;
/// `nclx` adds one byte whose top bit is `full_range_flag`. Other colour
/// types (`rICC`, `prof`) carry an ICC profile and no H.273 triple.
fn parse_colr(body: &[u8]) -> Option<Nclx> {
    if body.len() < 10 || !matches!(&body[0..4], b"nclx" | b"nclc") {
        return None;
    }
    let u16be = |o: usize| u16::from_be_bytes([body[o], body[o + 1]]);
    let narrow = |v: u16| u8::try_from(v).unwrap_or(2);
    Some(Nclx {
        primaries: narrow(u16be(4)),
        transfer: narrow(u16be(6)),
        matrix: narrow(u16be(8)),
        full_range: &body[0..4] == b"nclx" && body.len() >= 11 && body[10] & 0x80 != 0,
    })
}

/// The H.273 colour description in the first SPS among `parameter_sets`
/// (one NAL unit per entry, with or without an Annex-B start code — the
/// avcC / hvcC extractors hand them out either way), for `codec` `"h264"`
/// or `"h265"`. `None` when there is no SPS, it does not parse, or its VUI
/// has no `colour_description_present_flag`.
pub(crate) fn colour_from_parameter_sets(codec: &str, parameter_sets: &[Vec<u8>]) -> Option<Nclx> {
    for entry in parameter_sets {
        let nal: &[u8] = if entry.starts_with(&[0, 0, 0, 1]) {
            &entry[4..]
        } else if entry.starts_with(&[0, 0, 1]) {
            &entry[3..]
        } else {
            entry
        };
        if nal.is_empty() {
            continue;
        }
        let found = match codec {
            "h265" | "hevc" if (nal[0] >> 1) & 0x3f == 33 => {
                let rbsp = h26x::nal::unescape_rbsp(nal);
                let sps = h26x::hevc::Sps::parse(rbsp.get(2..)?).ok()?;
                let vui = sps.vui?;
                let (p, t, m) = vui.colour_description?;
                Nclx { primaries: p, transfer: t, matrix: m, full_range: vui.full_range }
            }
            "h264" | "avc" | "avc1" if nal[0] & 0x1f == 7 => {
                let rbsp = h26x::nal::unescape_rbsp(&nal[1..]);
                let sps = h26x::h264::Sps::parse(&rbsp).ok()?;
                let vui = sps.vui?;
                let (p, t, m) = vui.colour_description?;
                Nclx { primaries: p, transfer: t, matrix: m, full_range: vui.full_range }
            }
            _ => continue,
        };
        return Some(found);
    }
    None
}

/// Apply a colour description to the demuxed `StreamInfo`: `colr` wins
/// when it says something (ISO/IEC 14496-12 §12.1.5: the box overrides
/// the bitstream), else the SPS VUI, else the SDR defaults stay. Sets the
/// H.273 fields, the transfer, and the pipeline `ColorSpace` from the
/// matrix (BT.601 for 5/6, BT.2020 for 9/10, BT.709 otherwise).
pub(crate) fn apply_colour_description(info: &mut StreamInfo, colr: Option<Nclx>, vui: Option<Nclx>) {
    let Some(nclx) = colr.filter(Nclx::is_specified).or(vui) else {
        return;
    };
    info.color_metadata.colour_primaries = nclx.primaries;
    info.color_metadata.matrix_coefficients = nclx.matrix;
    info.color_metadata.transfer = TransferFn::from_h273(nclx.transfer);
    info.color_metadata.full_range = nclx.full_range;
    info.color_space = match nclx.matrix {
        5 | 6 => ColorSpace::Bt601,
        9 | 10 => ColorSpace::Bt2020,
        _ => ColorSpace::Bt709,
    };
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
    let Some(stsd_body) = super::find_video_stsd(data) else {
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
        if let Some(colr) = super::find_direct_child(children, b"colr") {
            out.nclx = parse_colr(colr);
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

#[cfg(test)]
mod colour_tests {
    use super::*;
    use frame::ColorMetadata;

    /// The SPS of an x265 PQ encode (`colorprim=bt2020:transfer=smpte2084:
    /// colormatrix=bt2020nc`), Annex-B framed.
    const HEVC_PQ_SPS: &[u8] = &[
        0, 0, 0, 1, 0x42, 0x01, 0x01, 0x02, 0x20, 0x00, 0x00, 0x03, 0x00, 0x90, 0x00, 0x00, 0x03,
        0x00, 0x00, 0x03, 0x00, 0x78, 0xa0, 0x03, 0xc0, 0x80, 0x10, 0xe4, 0xd9, 0x65, 0x66, 0x92,
        0x4c, 0xaf, 0x01, 0x6a, 0x12, 0x20, 0x12, 0x08, 0x00, 0x00, 0x03, 0x00, 0x08, 0x00, 0x00,
        0x03, 0x00, 0xf0, 0x40,
    ];
    /// The SPS of an x264 BT.709 encode with the colour description written.
    const H264_709_SPS: &[u8] = &[
        0, 0, 0, 1, 0x67, 0x64, 0x00, 0x0d, 0xac, 0xd9, 0x41, 0x41, 0x9f, 0x9f, 0x01, 0x6a, 0x02,
        0x02, 0x02, 0x80, 0x00, 0x00, 0x03, 0x00, 0x80, 0x00, 0x00, 0x1e, 0x07, 0x8a, 0x14, 0xcb,
    ];

    fn sdr_info() -> StreamInfo {
        StreamInfo {
            codec: "h265".into(),
            width: 16,
            height: 16,
            frame_rate: 30.0,
            duration: 1.0,
            pixel_format: frame::PixelFormat::Yuv420p10le,
            color_space: ColorSpace::Bt709,
            total_frames: 30,
            bitrate: 0,
            color_metadata: ColorMetadata::default(),
        }
    }

    #[test]
    fn hevc_vui_gives_the_pq_bt2020_triple() {
        let n = colour_from_parameter_sets("h265", &[HEVC_PQ_SPS.to_vec()]).expect("vui colour description");
        assert_eq!(n, Nclx { primaries: 9, transfer: 16, matrix: 9, full_range: false });
        let mut info = sdr_info();
        apply_colour_description(&mut info, None, Some(n));
        assert_eq!(info.color_metadata.transfer, TransferFn::St2084);
        assert_eq!(info.color_metadata.colour_primaries, 9);
        assert_eq!(info.color_metadata.matrix_coefficients, 9);
        assert_eq!(info.color_space, ColorSpace::Bt2020);
    }

    #[test]
    fn h264_vui_gives_bt709_and_a_missing_sps_gives_nothing() {
        let n = colour_from_parameter_sets("h264", &[H264_709_SPS.to_vec()]).expect("vui colour description");
        assert_eq!(n, Nclx { primaries: 1, transfer: 1, matrix: 1, full_range: false });
        assert!(colour_from_parameter_sets("h265", &[H264_709_SPS.to_vec()]).is_none(), "wrong codec: no SPS of that kind");
        assert!(colour_from_parameter_sets("h264", &[vec![0, 0, 0, 1, 0x68, 0xce, 0x38, 0x80]]).is_none(), "a PPS alone");
        assert!(colour_from_parameter_sets("h264", &[]).is_none());
        // Without a start code too (the hvcC / avcC extractors may strip it).
        assert!(colour_from_parameter_sets("h264", &[H264_709_SPS[4..].to_vec()]).is_some());
    }

    #[test]
    fn colr_nclx_parses_and_overrides_the_vui_unless_unspecified() {
        // nclx: primaries 9, transfer 18 (HLG), matrix 9, full_range set.
        let body = [b'n', b'c', b'l', b'x', 0, 9, 0, 18, 0, 9, 0x80];
        let n = parse_colr(&body).expect("nclx");
        assert_eq!(n, Nclx { primaries: 9, transfer: 18, matrix: 9, full_range: true });
        // nclc (QuickTime) has no range byte.
        let n = parse_colr(&[b'n', b'c', b'l', b'c', 0, 1, 0, 1, 0, 1]).expect("nclc");
        assert_eq!(n, Nclx { primaries: 1, transfer: 1, matrix: 1, full_range: false });
        assert!(parse_colr(b"rICC\x00\x00\x00\x00\x00\x00").is_none());
        assert!(parse_colr(b"nclx\x00\x09").is_none(), "truncated");

        let vui = colour_from_parameter_sets("h265", &[HEVC_PQ_SPS.to_vec()]).unwrap();
        // colr says HLG, the VUI says PQ: the box wins.
        let mut info = sdr_info();
        apply_colour_description(&mut info, parse_colr(&body), Some(vui));
        assert_eq!(info.color_metadata.transfer, TransferFn::AribStdB67);
        assert!(info.color_metadata.full_range);
        // An all-unspecified colr defers to the VUI.
        let mut info = sdr_info();
        apply_colour_description(&mut info, parse_colr(&[b'n', b'c', b'l', b'x', 0, 2, 0, 2, 0, 2, 0]), Some(vui));
        assert_eq!(info.color_metadata.transfer, TransferFn::St2084);
        // Nothing at all: SDR defaults stay.
        let mut info = sdr_info();
        apply_colour_description(&mut info, None, None);
        assert_eq!(info.color_metadata.transfer, TransferFn::Bt709);
        assert_eq!(info.color_space, ColorSpace::Bt709);
    }

    /// A minimal `moov > trak > mdia > minf > stbl > stsd > hvc1 > colr`
    /// so the sample-entry walker finds the box where a real file puts it.
    #[test]
    fn the_sample_entry_walker_finds_colr_beside_mdcv() {
        fn bx(kind: &[u8; 4], body: &[u8]) -> Vec<u8> {
            let mut v = ((body.len() + 8) as u32).to_be_bytes().to_vec();
            v.extend_from_slice(kind);
            v.extend_from_slice(body);
            v
        }
        let colr = bx(b"colr", &[b'n', b'c', b'l', b'x', 0, 9, 0, 16, 0, 9, 0]);
        let mut clli_body = 1000u16.to_be_bytes().to_vec();
        clli_body.extend_from_slice(&400u16.to_be_bytes());
        let clli = bx(b"clli", &clli_body);
        let mut entry_body = vec![0u8; 78];
        entry_body.extend_from_slice(&colr);
        entry_body.extend_from_slice(&clli);
        let entry = bx(b"hvc1", &entry_body);
        let mut stsd_body = vec![0, 0, 0, 0, 0, 0, 0, 1];
        stsd_body.extend_from_slice(&entry);
        let stsd = bx(b"stsd", &stsd_body);
        let file = bx(b"moov", &bx(b"trak", &bx(b"mdia", &bx(b"minf", &bx(b"stbl", &stsd)))));
        let got = extract_mp4_visual_color_metadata(&file);
        assert_eq!(got.nclx, Some(Nclx { primaries: 9, transfer: 16, matrix: 9, full_range: false }));
        assert_eq!(got.content_light_level.map(|c| c.max_cll), Some(1000));
    }
}
