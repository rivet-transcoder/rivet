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
use std::sync::Mutex;

#[derive(Debug, Default, Clone, Copy)]
pub(super) struct Mp4VisualColorMetadata {
    pub(super) mastering_display: Option<MasteringDisplay>,
    pub(super) content_light_level: Option<ContentLightLevel>,
    /// The `colr` box's H.273 triple, when the sample entry has one.
    pub(super) nclx: Option<Nclx>,
}

/// An H.273 colour description: `colour_primaries`,
/// `transfer_characteristics`, `matrix_coefficients`, `full_range_flag`.
/// From a `colr` box (`nclx` / `nclc`) or from an SPS VUI. A VUI that signals
/// its video signal type without a colour description gives all three
/// unspecified (2) and its range.
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
/// has no `video_signal_type_present_flag` — a VUI that says nothing about
/// colour. With the flag and no `colour_description_present_flag` the triple
/// is unspecified (2, 2, 2) and the range is the stream's: `-color_range pc`
/// alone is a full-range stream, and reading it as nothing lost that.
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
                let vui = sps.vui.filter(|v| v.video_signal_type)?;
                let (p, t, m) = vui.colour_description.unwrap_or((2, 2, 2));
                Nclx {
                    primaries: p,
                    transfer: t,
                    matrix: m,
                    full_range: vui.full_range,
                }
            }
            "h264" | "avc" | "avc1" if nal[0] & 0x1f == 7 => {
                let rbsp = h26x::nal::unescape_rbsp(&nal[1..]);
                let sps = h26x::h264::Sps::parse(&rbsp).ok()?;
                let vui = sps.vui.filter(|v| v.video_signal_type)?;
                let (p, t, m) = vui.colour_description.unwrap_or((2, 2, 2));
                Nclx {
                    primaries: p,
                    transfer: t,
                    matrix: m,
                    full_range: vui.full_range,
                }
            }
            _ => continue,
        };
        return Some(found);
    }
    None
}

/// Whether `message` is new since the last one told under `last`. One job
/// opens its input several times (the header probe, the decode pump, each
/// spliced clip) and every open resolves the same colour, so the log says it
/// once instead of once per open. Consecutive repeats only: another source,
/// or the same one after something else was told in between, is told again.
fn first_telling(last: &Mutex<u64>, message: &str) -> bool {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    message.hash(&mut hasher);
    let key = hasher.finish();
    let mut last = last.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    if *last == key {
        return false;
    }
    *last = key;
    true
}

static VUI_FILLED: Mutex<u64> = Mutex::new(0);
static VUI_DISAGREES: Mutex<u64> = Mutex::new(0);
static SEI_FILLED: Mutex<u64> = Mutex::new(0);
static MASTERING_DISAGREES: Mutex<u64> = Mutex::new(0);
static CLL_DISAGREES: Mutex<u64> = Mutex::new(0);

/// The pipeline `ColorSpace` for an H.273 matrix: BT.601 for 5/6, BT.2020
/// for 9/10, BT.709 otherwise.
fn color_space_for_matrix(matrix: u8) -> ColorSpace {
    match matrix {
        5 | 6 => ColorSpace::Bt601,
        9 | 10 => ColorSpace::Bt2020,
        _ => ColorSpace::Bt709,
    }
}

/// What a container's own colour description says, field by field. `None`
/// is a container that is silent on the field (MP4: no `colr`, or one whose
/// triple is all unspecified; Matroska: the element is absent). A present
/// value of 2 is the container saying "unspecified" out loud — the bitstream
/// may still fill it.
///
/// Matroska's elements are individually optional, and ffmpeg's muxer writes
/// only those its codec context knows: a BT.601 x264 encode came out with
/// `MatrixCoefficients` and `Range` and no primaries or transfer, an x265
/// PQ encode with `Range` alone. So the bitstream fills per field, not per
/// description.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ContainerColour {
    pub(crate) primaries: Option<u8>,
    pub(crate) transfer: Option<u8>,
    pub(crate) matrix: Option<u8>,
    pub(crate) full_range: Option<bool>,
}

impl ContainerColour {
    /// An MP4 `colr` box's say. An all-unspecified triple says nothing
    /// (not even its range bit), exactly as before the per-field rule.
    pub(crate) fn from_colr(colr: Option<Nclx>) -> Self {
        match colr.filter(Nclx::is_specified) {
            Some(n) => Self {
                primaries: Some(n.primaries),
                transfer: Some(n.transfer),
                matrix: Some(n.matrix),
                full_range: Some(n.full_range),
            },
            None => Self::default(),
        }
    }
}

/// Apply a colour description to the demuxed `StreamInfo`: `colr` wins
/// when it says something (ISO/IEC 14496-12 §12.1.5: the box overrides
/// the bitstream), else the SPS VUI, else the SDR defaults stay — per field,
/// see [`fill_colour_from_vui`]. Sets the H.273 fields, the transfer, and the
/// pipeline `ColorSpace` from the matrix (BT.601 for 5/6, BT.2020 for 9/10,
/// BT.709 otherwise).
pub(crate) fn apply_colour_description(
    info: &mut StreamInfo,
    colr: Option<Nclx>,
    vui: Option<Nclx>,
) {
    if let Some(nclx) = colr.filter(Nclx::is_specified) {
        info.color_metadata.colour_primaries = nclx.primaries;
        info.color_metadata.matrix_coefficients = nclx.matrix;
        info.color_metadata.transfer = TransferFn::from_h273(nclx.transfer);
        info.color_metadata.full_range = nclx.full_range;
        info.color_space = color_space_for_matrix(nclx.matrix);
    }
    fill_colour_from_vui(info, ContainerColour::from_colr(colr), vui, "mp4");
}

/// The source's colour is a property of the stream: fill every field the
/// container left unsaid (`None`, or an explicit 2) from the SPS VUI, when the
/// VUI specifies it (not 2). The range follows the VUI whenever the container
/// did not signal one and the VUI supplied any field, or signalled a range
/// other than the one `info` holds — `vui` is only ever a VUI that signalled
/// its video signal type ([`colour_from_parameter_sets`]), so its range is
/// the stream's statement even with no colour description beside it. A field
/// the container set is never touched, and `info` is left exactly as the
/// container made it when the VUI has nothing to add — so a fully
/// container-tagged source reads the same as before. `ColorSpace` is
/// re-derived only when the matrix came from the VUI.
///
/// Returns whether anything was filled. Logs what was taken from the VUI, and
/// a field the container and the VUI disagree on (the container's is kept).
pub(crate) fn fill_colour_from_vui(
    info: &mut StreamInfo,
    container: ContainerColour,
    vui: Option<Nclx>,
    container_label: &str,
) -> bool {
    let Some(vui) = vui else { return false };
    let specified = |v: u8| v != 2;
    let wants = |c: Option<u8>, v: u8| !c.is_some_and(specified) && specified(v);
    let mut filled = Vec::new();
    if wants(container.primaries, vui.primaries) {
        info.color_metadata.colour_primaries = vui.primaries;
        filled.push("primaries");
    }
    if wants(container.transfer, vui.transfer) {
        info.color_metadata.transfer = TransferFn::from_h273(vui.transfer);
        filled.push("transfer");
    }
    if wants(container.matrix, vui.matrix) {
        info.color_metadata.matrix_coefficients = vui.matrix;
        info.color_space = color_space_for_matrix(vui.matrix);
        filled.push("matrix");
    }
    if container.full_range.is_none()
        && (!filled.is_empty() || vui.full_range != info.color_metadata.full_range)
    {
        info.color_metadata.full_range = vui.full_range;
        filled.push("range");
    }
    let disagree = |c: Option<u8>, v: u8| c.is_some_and(|c| specified(c) && specified(v) && c != v);
    if (disagree(container.primaries, vui.primaries)
        || disagree(container.transfer, vui.transfer)
        || disagree(container.matrix, vui.matrix))
        && first_telling(
            &VUI_DISAGREES,
            &format!("{container_label} {container:?} {vui:?}"),
        )
    {
        tracing::info!(
            container = container_label,
            container_primaries = ?container.primaries,
            container_transfer = ?container.transfer,
            container_matrix = ?container.matrix,
            vui_primaries = vui.primaries,
            vui_transfer = vui.transfer,
            vui_matrix = vui.matrix,
            "source colour: the container's colour description differs from the SPS VUI; keeping the container's"
        );
    }
    if filled.is_empty() {
        return false;
    }
    let told = format!(
        "{container_label} {filled:?} {:?} {:?}",
        info.color_metadata, info.color_space
    );
    if first_telling(&VUI_FILLED, &told) {
        tracing::info!(
            container = container_label,
            from_sps_vui = ?filled,
            primaries = info.color_metadata.colour_primaries,
            transfer = ?info.color_metadata.transfer,
            matrix = info.color_metadata.matrix_coefficients,
            full_range = info.color_metadata.full_range,
            color_space = ?info.color_space,
            "source colour: filled from the SPS VUI where the container is silent"
        );
    }
    true
}

/// Fill the HDR10 static metadata the container did not carry from the
/// stream's SEI messages (mastering display 137, content light level 144).
/// A value the container carries is kept; when the SEI carries a different
/// one, that is logged. Returns whether anything was filled.
pub(crate) fn fill_hdr_static_from_sei(
    info: &mut StreamInfo,
    sei: frame::hdr_sei::HdrSei,
    container_label: &str,
) -> bool {
    let meta = &mut info.color_metadata;
    let mut filled = Vec::new();
    match (meta.mastering_display, sei.mastering_display) {
        (None, Some(s)) => {
            meta.mastering_display = Some(s);
            filled.push("mastering_display");
        }
        (Some(c), Some(s))
            if c != s
                && first_telling(
                    &MASTERING_DISAGREES,
                    &format!("{container_label} {c:?} {s:?}"),
                ) =>
        {
            tracing::warn!(
                container = container_label,
                container_mastering_display = ?c,
                sei_mastering_display = ?s,
                "source HDR metadata: the container's mastering display differs from the stream's SEI 137; keeping the container's"
            )
        }
        _ => {}
    }
    match (meta.content_light_level, sei.content_light_level) {
        (None, Some(s)) => {
            meta.content_light_level = Some(s);
            filled.push("content_light_level");
        }
        (Some(c), Some(s))
            if c != s
                && first_telling(&CLL_DISAGREES, &format!("{container_label} {c:?} {s:?}")) =>
        {
            tracing::warn!(
                container = container_label,
                container_content_light_level = ?c,
                sei_content_light_level = ?s,
                "source HDR metadata: the container's content light level differs from the stream's SEI 144; keeping the container's"
            )
        }
        _ => {}
    }
    if filled.is_empty() {
        return false;
    }
    let told = format!(
        "{container_label} {filled:?} {:?} {:?}",
        meta.mastering_display, meta.content_light_level
    );
    if first_telling(&SEI_FILLED, &told) {
        tracing::info!(
            container = container_label,
            from_sei = ?filled,
            mastering_display = ?meta.mastering_display,
            content_light_level = ?meta.content_light_level,
            "source HDR metadata: filled from the stream's SEI where the container is silent"
        );
    }
    true
}

/// What the bitstream says about colour: the H.273 description of the first
/// SPS (from the out-of-band `parameter_sets`, else the in-band ones of
/// `first_au`), and the HDR10 static metadata SEIs among both. `first_au` is
/// the stream's first access unit, Annex-B — for a file that opens on an IRAP,
/// the unit x264 / x265 put those SEIs in.
pub(crate) fn bitstream_colour(
    codec: &str,
    parameter_sets: &[Vec<u8>],
    first_au: Option<&[u8]>,
) -> (Option<Nclx>, frame::hdr_sei::HdrSei) {
    let in_band: Vec<Vec<u8>> = first_au
        .map(|au| h26x::nal::annexb_nals(au).map(<[u8]>::to_vec).collect())
        .unwrap_or_default();
    let vui = colour_from_parameter_sets(codec, parameter_sets)
        .or_else(|| colour_from_parameter_sets(codec, &in_band));
    let mut annexb = Vec::new();
    for nal in parameter_sets.iter().chain(&in_band) {
        if !nal.starts_with(&[0, 0, 1]) && !nal.starts_with(&[0, 0, 0, 1]) {
            annexb.extend_from_slice(&[0, 0, 0, 1]);
        }
        annexb.extend_from_slice(nal);
    }
    (vui, frame::hdr_sei::parse_annexb_for(codec, &annexb))
}

/// The one rule every demuxer applies once it knows its container's say:
/// container colour description first, else the first SPS's VUI, else the
/// defaults stay (per field, [`fill_colour_from_vui`]); container mastering
/// display / content light level first, else the SEIs
/// ([`fill_hdr_static_from_sei`]). H.264 and HEVC only — no other codec's
/// bitstream is read here.
pub(crate) fn resolve_source_colour(
    info: &mut StreamInfo,
    container: ContainerColour,
    codec: &str,
    parameter_sets: &[Vec<u8>],
    first_au: Option<&[u8]>,
    container_label: &str,
) {
    if !matches!(codec, "h264" | "h265") {
        return;
    }
    let (vui, sei) = bitstream_colour(codec, parameter_sets, first_au);
    fill_colour_from_vui(info, container, vui, container_label);
    fill_hdr_static_from_sei(info, sei, container_label);
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

pub(crate) fn parse_mp4_mdcv(body: &[u8]) -> Option<MasteringDisplay> {
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
        let n = colour_from_parameter_sets("h265", &[HEVC_PQ_SPS.to_vec()])
            .expect("vui colour description");
        assert_eq!(
            n,
            Nclx {
                primaries: 9,
                transfer: 16,
                matrix: 9,
                full_range: false
            }
        );
        let mut info = sdr_info();
        apply_colour_description(&mut info, None, Some(n));
        assert_eq!(info.color_metadata.transfer, TransferFn::St2084);
        assert_eq!(info.color_metadata.colour_primaries, 9);
        assert_eq!(info.color_metadata.matrix_coefficients, 9);
        assert_eq!(info.color_space, ColorSpace::Bt2020);
    }

    #[test]
    fn h264_vui_gives_bt709_and_a_missing_sps_gives_nothing() {
        let n = colour_from_parameter_sets("h264", &[H264_709_SPS.to_vec()])
            .expect("vui colour description");
        assert_eq!(
            n,
            Nclx {
                primaries: 1,
                transfer: 1,
                matrix: 1,
                full_range: false
            }
        );
        assert!(
            colour_from_parameter_sets("h265", &[H264_709_SPS.to_vec()]).is_none(),
            "wrong codec: no SPS of that kind"
        );
        assert!(
            colour_from_parameter_sets("h264", &[vec![0, 0, 0, 1, 0x68, 0xce, 0x38, 0x80]])
                .is_none(),
            "a PPS alone"
        );
        assert!(colour_from_parameter_sets("h264", &[]).is_none());
        // Without a start code too (the hvcC / avcC extractors may strip it).
        assert!(colour_from_parameter_sets("h264", &[H264_709_SPS[4..].to_vec()]).is_some());
    }

    /// `-color_range pc` and nothing else: x264 and x265 write
    /// `video_signal_type_present_flag` with `video_full_range_flag` 1 and no
    /// colour description. Before, that read as no VUI colour at all and the
    /// source came out limited range.
    #[test]
    fn a_vui_range_without_a_colour_description_is_read() {
        // Each SPS NAL straight out of ffmpeg 8.1.1 (64x64 testsrc2), trace_headers
        // quoted beside it.
        // x264: video_signal_type_present_flag=1 video_full_range_flag=1
        // colour_description_present_flag=0
        const X264_FULL: &[u8] = &[
            0x67, 0x64, 0x00, 0x0a, 0xac, 0xd9, 0x44, 0x26, 0xc0, 0x5b, 0x20, 0x00, 0x00, 0x03,
            0x00, 0x20, 0x00, 0x00, 0x07, 0x81, 0xe2, 0x44, 0xb2, 0xc0,
        ];
        // x264, no colour options: video_signal_type_present_flag=0
        const X264_PLAIN: &[u8] = &[
            0x67, 0x64, 0x00, 0x0a, 0xac, 0xd9, 0x44, 0x26, 0xc0, 0x44, 0x00, 0x00, 0x03, 0x00,
            0x04, 0x00, 0x00, 0x03, 0x00, 0xf0, 0x3c, 0x48, 0x96, 0x58,
        ];
        // x265: video_signal_type_present_flag=1 video_full_range_flag=1
        // colour_description_present_flag=0
        const X265_FULL: &[u8] = &[
            0x42, 0x01, 0x01, 0x01, 0x60, 0x00, 0x00, 0x03, 0x00, 0x90, 0x00, 0x00, 0x03, 0x00,
            0x00, 0x03, 0x00, 0x1e, 0xa0, 0x20, 0x81, 0x05, 0x96, 0x56, 0x69, 0x24, 0xca, 0xf0,
            0x16, 0xc0, 0x80, 0x00, 0x00, 0x03, 0x00, 0x80, 0x00, 0x00, 0x0f, 0x04,
        ];
        // x265, no colour options: video_signal_type_present_flag=1
        // video_full_range_flag=0 colour_description_present_flag=0
        const X265_PLAIN: &[u8] = &[
            0x42, 0x01, 0x01, 0x01, 0x60, 0x00, 0x00, 0x03, 0x00, 0x90, 0x00, 0x00, 0x03, 0x00,
            0x00, 0x03, 0x00, 0x1e, 0xa0, 0x20, 0x81, 0x05, 0x96, 0x56, 0x69, 0x24, 0xca, 0xf0,
            0x16, 0x80, 0x80, 0x00, 0x00, 0x03, 0x00, 0x80, 0x00, 0x00, 0x0f, 0x04,
        ];
        let range_only = |full_range| Nclx {
            primaries: 2,
            transfer: 2,
            matrix: 2,
            full_range,
        };
        assert_eq!(
            colour_from_parameter_sets("h264", &[X264_FULL.to_vec()]),
            Some(range_only(true))
        );
        assert_eq!(
            colour_from_parameter_sets("h264", &[X264_PLAIN.to_vec()]),
            None,
            "no signal type: the VUI says nothing about colour"
        );
        assert_eq!(
            colour_from_parameter_sets("h265", &[X265_FULL.to_vec()]),
            Some(range_only(true))
        );
        assert_eq!(
            colour_from_parameter_sets("h265", &[X265_PLAIN.to_vec()]),
            Some(range_only(false))
        );

        for (codec, sps, full) in [
            ("h264", X264_FULL, true),
            ("h264", X264_PLAIN, false),
            ("h265", X265_FULL, true),
            ("h265", X265_PLAIN, false),
        ] {
            let mut info = sdr_info();
            let before = format!("{info:?}");
            resolve_source_colour(
                &mut info,
                ContainerColour::default(),
                codec,
                &[sps.to_vec()],
                None,
                "t",
            );
            assert_eq!(info.color_metadata.full_range, full, "{codec} {full}");
            if !full {
                assert_eq!(format!("{info:?}"), before, "{codec}: nothing to fill");
            }
            assert_eq!(
                info.color_space,
                ColorSpace::Bt709,
                "{codec}: no matrix came with it"
            );
        }
    }

    #[test]
    fn colr_nclx_parses_and_overrides_the_vui_unless_unspecified() {
        // nclx: primaries 9, transfer 18 (HLG), matrix 9, full_range set.
        let body = [b'n', b'c', b'l', b'x', 0, 9, 0, 18, 0, 9, 0x80];
        let n = parse_colr(&body).expect("nclx");
        assert_eq!(
            n,
            Nclx {
                primaries: 9,
                transfer: 18,
                matrix: 9,
                full_range: true
            }
        );
        // nclc (QuickTime) has no range byte.
        let n = parse_colr(&[b'n', b'c', b'l', b'c', 0, 1, 0, 1, 0, 1]).expect("nclc");
        assert_eq!(
            n,
            Nclx {
                primaries: 1,
                transfer: 1,
                matrix: 1,
                full_range: false
            }
        );
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
        apply_colour_description(
            &mut info,
            parse_colr(&[b'n', b'c', b'l', b'x', 0, 2, 0, 2, 0, 2, 0]),
            Some(vui),
        );
        assert_eq!(info.color_metadata.transfer, TransferFn::St2084);
        // Nothing at all: SDR defaults stay.
        let mut info = sdr_info();
        apply_colour_description(&mut info, None, None);
        assert_eq!(info.color_metadata.transfer, TransferFn::Bt709);
        assert_eq!(info.color_space, ColorSpace::Bt709);
    }

    /// ffmpeg's Matroska muxer wrote `MatrixCoefficients` (and `Range`) and
    /// nothing else for a BT.601 x264 encode: the VUI fills the silent fields,
    /// the container's own are untouched, its `ColorSpace` too.
    #[test]
    fn a_partial_container_description_is_filled_per_field_from_the_vui() {
        let vui = colour_from_parameter_sets("h265", &[HEVC_PQ_SPS.to_vec()]).unwrap();
        let mut info = sdr_info();
        info.color_metadata.matrix_coefficients = 6;
        info.color_space = ColorSpace::Bt601;
        let container = ContainerColour {
            matrix: Some(6),
            full_range: Some(false),
            ..Default::default()
        };
        assert!(fill_colour_from_vui(
            &mut info,
            container,
            Some(vui),
            "test"
        ));
        assert_eq!(info.color_metadata.colour_primaries, 9, "filled");
        assert_eq!(info.color_metadata.transfer, TransferFn::St2084, "filled");
        assert_eq!(
            info.color_metadata.matrix_coefficients, 6,
            "the container's"
        );
        assert_eq!(info.color_space, ColorSpace::Bt601, "the container's");
    }

    #[test]
    fn a_fully_tagged_container_is_left_exactly_as_it_was() {
        let vui = colour_from_parameter_sets("h265", &[HEVC_PQ_SPS.to_vec()]).unwrap();
        let mut info = sdr_info();
        info.color_metadata.full_range = true;
        let before = info.clone();
        let container = ContainerColour {
            primaries: Some(1),
            transfer: Some(1),
            matrix: Some(1),
            full_range: Some(true),
        };
        assert!(!fill_colour_from_vui(
            &mut info,
            container,
            Some(vui),
            "test"
        ));
        assert_eq!(format!("{info:?}"), format!("{before:?}"));
    }

    #[test]
    fn a_container_unspecified_is_filled_and_a_vui_unspecified_fills_nothing() {
        let mut info = sdr_info();
        let container = ContainerColour {
            primaries: Some(1),
            transfer: Some(2),
            matrix: Some(1),
            full_range: Some(false),
        };
        let pq = Nclx {
            primaries: 9,
            transfer: 16,
            matrix: 9,
            full_range: true,
        };
        assert!(fill_colour_from_vui(&mut info, container, Some(pq), "test"));
        assert_eq!(info.color_metadata.transfer, TransferFn::St2084);
        assert_eq!(info.color_metadata.colour_primaries, 1);
        assert!(
            !info.color_metadata.full_range,
            "the container signalled a range"
        );

        // A VUI with nothing specified but its range: the range is still the
        // stream's statement, and a silent container takes it.
        let mut info = sdr_info();
        let range_only = Nclx {
            primaries: 2,
            transfer: 2,
            matrix: 2,
            full_range: true,
        };
        assert!(fill_colour_from_vui(
            &mut info,
            ContainerColour::default(),
            Some(range_only),
            "t"
        ));
        assert!(
            info.color_metadata.full_range,
            "the VUI signalled a full range"
        );
        assert_eq!(
            info.color_metadata.matrix_coefficients, 1,
            "and nothing else"
        );
        // ...unless the container signalled one.
        let mut info = sdr_info();
        let ranged = ContainerColour {
            full_range: Some(false),
            ..Default::default()
        };
        assert!(!fill_colour_from_vui(
            &mut info,
            ranged,
            Some(range_only),
            "t"
        ));
        assert!(!info.color_metadata.full_range);
        // A range-only VUI that says what `info` already holds fills nothing.
        let mut info = sdr_info();
        assert!(!fill_colour_from_vui(
            &mut info,
            ContainerColour::default(),
            Some(Nclx {
                full_range: false,
                ..range_only
            }),
            "t"
        ));
        assert!(!fill_colour_from_vui(
            &mut info,
            ContainerColour::default(),
            None,
            "t"
        ));
    }

    #[test]
    fn the_container_static_metadata_wins_and_the_sei_fills_what_is_missing() {
        let md = |max_luminance| frame::MasteringDisplay {
            primaries_r_x: 1,
            primaries_r_y: 2,
            primaries_g_x: 3,
            primaries_g_y: 4,
            primaries_b_x: 5,
            primaries_b_y: 6,
            white_point_x: 7,
            white_point_y: 8,
            max_luminance,
            min_luminance: 9,
        };
        let cll = frame::ContentLightLevel {
            max_cll: 1234,
            max_fall: 567,
        };
        let mut info = sdr_info();
        info.color_metadata.mastering_display = Some(md(10_000_000));
        let sei = frame::hdr_sei::HdrSei {
            mastering_display: Some(md(40_000_000)),
            content_light_level: Some(cll),
        };
        assert!(fill_hdr_static_from_sei(&mut info, sei, "test"));
        assert_eq!(
            info.color_metadata.mastering_display,
            Some(md(10_000_000)),
            "container kept"
        );
        assert_eq!(
            info.color_metadata.content_light_level,
            Some(cll),
            "SEI fills"
        );
        assert!(!fill_hdr_static_from_sei(
            &mut info,
            frame::hdr_sei::HdrSei::default(),
            "t"
        ));
    }

    /// No out-of-band parameter sets (hev1 / TS / AVI): the first access
    /// unit's own SPS and SEIs are read. Other codecs are left alone.
    #[test]
    fn the_first_access_unit_supplies_the_vui_and_the_seis() {
        let mut au = HEVC_PQ_SPS.to_vec();
        // Prefix SEI (type 39): content_light_level_info 1234 / 567.
        au.extend_from_slice(&[0, 0, 0, 1, 0x4E, 0x01, 144, 4, 0x04, 0xD2, 0x02, 0x37, 0x80]);
        let (vui, sei) = bitstream_colour("h265", &[], Some(&au));
        assert_eq!(
            vui.map(|n| (n.primaries, n.transfer, n.matrix)),
            Some((9, 16, 9))
        );
        assert_eq!(
            sei.content_light_level.map(|c| (c.max_cll, c.max_fall)),
            Some((1234, 567))
        );

        let mut info = sdr_info();
        resolve_source_colour(
            &mut info,
            ContainerColour::default(),
            "h265",
            &[],
            Some(&au),
            "t",
        );
        assert_eq!(info.color_metadata.transfer, TransferFn::St2084);
        assert_eq!(info.color_space, ColorSpace::Bt2020);
        assert!(info.color_metadata.content_light_level.is_some());

        let mut info = sdr_info();
        resolve_source_colour(
            &mut info,
            ContainerColour::default(),
            "av1",
            &[],
            Some(&au),
            "t",
        );
        assert_eq!(
            info.color_metadata,
            ColorMetadata::default(),
            "not an H.264 / HEVC stream"
        );
    }

    #[test]
    fn a_repeated_telling_is_suppressed_until_something_else_is_told() {
        let last = Mutex::new(0);
        assert!(first_telling(&last, "mp4 [matrix] a"));
        assert!(
            !first_telling(&last, "mp4 [matrix] a"),
            "the same open again"
        );
        assert!(first_telling(&last, "ts [matrix] b"));
        assert!(
            first_telling(&last, "mp4 [matrix] a"),
            "told again after something else"
        );
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
        let file = bx(
            b"moov",
            &bx(b"trak", &bx(b"mdia", &bx(b"minf", &bx(b"stbl", &stsd)))),
        );
        let got = extract_mp4_visual_color_metadata(&file);
        assert_eq!(
            got.nclx,
            Some(Nclx {
                primaries: 9,
                transfer: 16,
                matrix: 9,
                full_range: false
            })
        );
        assert_eq!(got.content_light_level.map(|c| c.max_cll), Some(1000));
    }
}
