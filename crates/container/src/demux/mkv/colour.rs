/// Matroska `Colour` element → pipeline types mapping, mastering-display
/// conversions, matrix/transfer/primaries H.273 mappings, and the
/// tag-based bitrate resolver.

use frame::{ColorMetadata, ColorSpace, ContentLightLevel, MasteringDisplay, TransferFn};
use matroska_demuxer::{
    Colour as MkvColour, MasteringMetadata as MkvMastering, MatrixCoefficients, Primaries,
    Range as MkvRange, TransferCharacteristics,
};

use crate::{MkvColorInfo, MkvMasteringMetadata};

// ---------------------------------------------------------------------------
// Colour element → pipeline types
// ---------------------------------------------------------------------------

/// Map a Matroska `Colour` element into our pipeline's color-space,
/// per-H.273 `ColorMetadata`, and extended `MkvColorInfo`. Unspecified
/// sub-elements default to the SDR BT.709 baseline so decoders that
/// never read a Colour element keep behaving exactly as before.
pub(super) fn colour_to_pipeline(colour: &MkvColour) -> (ColorSpace, ColorMetadata, MkvColorInfo) {
    let matrix_u8 = colour
        .matrix_coefficients()
        .map(matrix_coefficients_to_h273);
    let primaries_u8 = colour.primaries().map(primaries_to_h273);
    let transfer_u8 = colour.transfer_characteristics().map(transfer_to_h273);
    let range = colour.range();

    let color_space = match colour.matrix_coefficients() {
        Some(MatrixCoefficients::Bt709) => ColorSpace::Bt709,
        Some(MatrixCoefficients::Bt470bg) | Some(MatrixCoefficients::Smpte170) => ColorSpace::Bt601,
        Some(MatrixCoefficients::Bt2020Ncl)
        | Some(MatrixCoefficients::Bt2020Cl)
        | Some(MatrixCoefficients::Bt2100) => ColorSpace::Bt2020,
        _ => ColorSpace::Bt709,
    };

    let mastering = colour.mastering_metadata().map(mkv_mastering_to_local);
    let mkv_max_cll = colour.max_cll().and_then(|v| u32::try_from(v).ok());
    let mkv_max_fall = colour.max_fall().and_then(|v| u32::try_from(v).ok());

    // Squad-21: also synthesize the unified ColorMetadata HDR fields from
    // the MKV `MasteringMetadata` + `MaxCLL` / `MaxFALL` so the muxer
    // (Squad-20) can write `mdcv`/`clli` without re-reading the
    // MKV-specific MkvColorInfo struct. matroska-demuxer 0.7's MaxCLL/
    // MaxFALL bug (see `probe_mkv_color_info`) means the values here
    // come from the typed accessor — for the canonical scan we re-read
    // raw bytes in `probe_mkv_color_info`. The two paths agree on
    // well-formed MKVs and disagree only on malformed ones (where the
    // raw scan wins). Pipeline plumbs the raw-scan path for MKV.
    let unified_mastering = mastering.as_ref().and_then(mkv_mastering_to_unified);
    let unified_cll = match (mkv_max_cll, mkv_max_fall) {
        (None, None) => None,
        (cll, fall) => Some(ContentLightLevel {
            max_cll: cll.unwrap_or(0).min(u16::MAX as u32) as u16,
            max_fall: fall.unwrap_or(0).min(u16::MAX as u32) as u16,
        }),
    };

    let color_metadata = ColorMetadata {
        transfer: transfer_u8.map(TransferFn::from_h273).unwrap_or_default(),
        matrix_coefficients: matrix_u8.unwrap_or(1),
        colour_primaries: primaries_u8.unwrap_or(1),
        // H.273 full_range_flag: Matroska Range=2 (Full) sets it; any
        // other value (Broadcast, Defined, Unknown) keeps the studio
        // 16..235 default.
        full_range: matches!(range, Some(MkvRange::Full)),
        // Squad-21 wires MKV float chromaticities + max_cll/fall into
        // the H.265-spec u16 encoding via `mkv_mastering_to_unified` and
        // the f64 → cd/m² conversion above (also recovers around two
        // matroska-demuxer 0.7 bugs that misread MaxCLL/MaxFALL and y
        // chromaticities at the wrong ElementIds).
        mastering_display: unified_mastering,
        content_light_level: unified_cll,
    };

    let extra = MkvColorInfo {
        bits_per_channel: colour.bits_per_channel().and_then(|v| u8::try_from(v).ok()),
        chroma_subsampling_horz: colour
            .chroma_subsampling_horz()
            .and_then(|v| u8::try_from(v).ok()),
        chroma_subsampling_vert: colour
            .chroma_subsampling_vert()
            .and_then(|v| u8::try_from(v).ok()),
        chroma_siting_horz: colour.chroma_sitting_horz().map(chroma_siting_horz_to_u8),
        chroma_siting_vert: colour.chroma_sitting_vert().map(chroma_siting_vert_to_u8),
        max_cll: mkv_max_cll,
        max_fall: mkv_max_fall,
        mastering,
    };

    (color_space, color_metadata, extra)
}

/// Convert the Matroska f64 chromaticities (range 0..=1) and luminance
/// (cd/m²) into the integer encoding the unified `MasteringDisplay`
/// uses (HEVC SEI D.2.28 wire format). Returns `None` when no
/// sub-element of the MasteringMetadata was populated.
fn mkv_mastering_to_unified(m: &MkvMasteringMetadata) -> Option<MasteringDisplay> {
    if m.primary_r_chromaticity_x.is_none()
        && m.primary_g_chromaticity_x.is_none()
        && m.primary_b_chromaticity_x.is_none()
        && m.white_point_chromaticity_x.is_none()
        && m.luminance_max.is_none()
        && m.luminance_min.is_none()
    {
        return None;
    }
    let chrom = |v: Option<f64>| -> u16 {
        // 0.00002 increments per HEVC SEI D.2.28 — map [0.0, ~1.31)
        // into a u16 with saturation.
        let scaled = (v.unwrap_or(0.0) * 50_000.0).round();
        scaled.clamp(0.0, u16::MAX as f64) as u16
    };
    let max_lum = (m.luminance_max.unwrap_or(0.0) * 10_000.0).round();
    let min_lum = (m.luminance_min.unwrap_or(0.0) * 10_000.0).round();
    Some(MasteringDisplay {
        primaries_r_x: chrom(m.primary_r_chromaticity_x),
        primaries_r_y: chrom(m.primary_r_chromaticity_y),
        primaries_g_x: chrom(m.primary_g_chromaticity_x),
        primaries_g_y: chrom(m.primary_g_chromaticity_y),
        primaries_b_x: chrom(m.primary_b_chromaticity_x),
        primaries_b_y: chrom(m.primary_b_chromaticity_y),
        white_point_x: chrom(m.white_point_chromaticity_x),
        white_point_y: chrom(m.white_point_chromaticity_y),
        max_luminance: max_lum.clamp(0.0, u32::MAX as f64) as u32,
        min_luminance: min_lum.clamp(0.0, u32::MAX as f64) as u32,
    })
}

fn mkv_mastering_to_local(m: &MkvMastering) -> MkvMasteringMetadata {
    MkvMasteringMetadata {
        primary_r_chromaticity_x: m.primary_r_chromaticity_x(),
        primary_r_chromaticity_y: m.primary_r_chromaticity_y(),
        primary_g_chromaticity_x: m.primary_g_chromaticity_x(),
        primary_g_chromaticity_y: m.primary_g_chromaticity_y(),
        primary_b_chromaticity_x: m.primary_b_chromaticity_x(),
        primary_b_chromaticity_y: m.primary_b_chromaticity_y(),
        white_point_chromaticity_x: m.white_point_chromaticity_x(),
        white_point_chromaticity_y: m.white_point_chromaticity_y(),
        luminance_max: m.luminance_max(),
        luminance_min: m.luminance_min(),
    }
}

// ---------------------------------------------------------------------------
// H.273 numeric mappings
// ---------------------------------------------------------------------------

/// MatroskaElement MatrixCoefficients (0x55B1) uses the H.273 numbering
/// 1:1, but the `matroska-demuxer` enum hides the raw u8. Reverse the
/// mapping so downstream (mux `colr nclx`, nvenc encode params) can
/// write the original numeric value back out without re-deriving it.
fn matrix_coefficients_to_h273(m: MatrixCoefficients) -> u8 {
    match m {
        MatrixCoefficients::Identity => 0,
        MatrixCoefficients::Bt709 => 1,
        MatrixCoefficients::Fcc73682 => 4,
        MatrixCoefficients::Bt470bg => 5,
        MatrixCoefficients::Smpte170 => 6,
        MatrixCoefficients::Smpte240 => 7,
        MatrixCoefficients::YCoCg => 8,
        MatrixCoefficients::Bt2020Ncl => 9,
        MatrixCoefficients::Bt2020Cl => 10,
        MatrixCoefficients::SmpteSt2085 => 11,
        MatrixCoefficients::ChromaDerivedNcl => 12,
        MatrixCoefficients::ChromaDerivedCl => 13,
        MatrixCoefficients::Bt2100 => 14,
        MatrixCoefficients::Unknown => 2, // H.273 "unspecified"
    }
}

fn transfer_to_h273(t: TransferCharacteristics) -> u8 {
    match t {
        TransferCharacteristics::Bt709 => 1,
        TransferCharacteristics::Bt407m => 4,
        TransferCharacteristics::Bt407bg => 5,
        TransferCharacteristics::Smpte170 => 6,
        TransferCharacteristics::Smpte240 => 7,
        TransferCharacteristics::Linear => 8,
        TransferCharacteristics::Log => 9,
        TransferCharacteristics::LogSqrt => 10,
        TransferCharacteristics::Iec61966_2_4 => 11,
        TransferCharacteristics::Bt1361 => 12,
        TransferCharacteristics::Iec61966_2_1 => 13,
        TransferCharacteristics::Bt220_10 => 14,
        TransferCharacteristics::Bt220_12 => 15,
        TransferCharacteristics::Bt2100 => 16,
        TransferCharacteristics::SmpteSt428_1 => 17,
        TransferCharacteristics::Hlg => 18,
        TransferCharacteristics::Unknown => 2,
    }
}

fn primaries_to_h273(p: Primaries) -> u8 {
    match p {
        Primaries::Bt709 => 1,
        Primaries::Bt470m => 4,
        Primaries::Bt601 => 5,
        Primaries::Smpte170 => 6,
        Primaries::Smpte240 => 7,
        Primaries::Film => 8,
        Primaries::Bt2020 => 9,
        Primaries::SmpteSt428_1 => 10,
        Primaries::SmpteRp432_2 => 11,
        Primaries::SmpteEg432_2 => 12,
        Primaries::JedecP22 => 22,
        Primaries::Unknown => 2,
    }
}

fn chroma_siting_horz_to_u8(s: matroska_demuxer::ChromaSitingHorz) -> u8 {
    match s {
        matroska_demuxer::ChromaSitingHorz::LeftCollated => 1,
        matroska_demuxer::ChromaSitingHorz::Half => 2,
        matroska_demuxer::ChromaSitingHorz::Unknown => 0,
    }
}

fn chroma_siting_vert_to_u8(s: matroska_demuxer::ChromaSitingVert) -> u8 {
    match s {
        matroska_demuxer::ChromaSitingVert::LeftCollated => 1,
        matroska_demuxer::ChromaSitingVert::Half => 2,
        matroska_demuxer::ChromaSitingVert::Unknown => 0,
    }
}

// ---------------------------------------------------------------------------
// Tag-based bitrate resolver
// ---------------------------------------------------------------------------

/// Resolve a track-scoped `BIT_RATE` Matroska Tag to a bits-per-second
/// value. Matroska's tag-scoping rules (spec §"Tagging") say: a Tag
/// applies to the track whose `TagTrackUID` matches, or to every track
/// in the segment if `TagTrackUID` is absent or 0. We prefer an exact
/// UID match, fall back to a segment-wide tag when no per-track value
/// exists.
///
/// `BIT_RATE` is the canonical Matroska target tag name (FFmpeg writes
/// it; the MKVToolNix matrix documents it). Some encoders emit
/// `BPS` / `BPS-eng` instead — we accept both for robustness. Values
/// are strings of base-10 digits in bits per second.
pub(super) fn bitrate_from_tags(tags: &[matroska_demuxer::Tag], track_uid: u64) -> Option<u64> {
    let matches_track = |tag: &matroska_demuxer::Tag| -> bool {
        match tag.targets() {
            None => true, // Segment-wide — applies to all tracks.
            Some(t) => match t.tag_track_uid() {
                None | Some(0) => true,
                Some(uid) => uid == track_uid,
            },
        }
    };
    let mut segment_wide: Option<u64> = None;
    let mut track_scoped: Option<u64> = None;
    for tag in tags {
        if !matches_track(tag) {
            continue;
        }
        for st in tag.simple_tags() {
            let name = st.name();
            let is_bitrate = name.eq_ignore_ascii_case("BIT_RATE")
                || name.eq_ignore_ascii_case("BPS")
                || name.to_ascii_uppercase().starts_with("BPS-");
            if !is_bitrate {
                continue;
            }
            let Some(val) = st.string() else {
                continue;
            };
            let Ok(parsed) = val.trim().parse::<u64>() else {
                continue;
            };
            let is_track_scoped = tag
                .targets()
                .and_then(|t| t.tag_track_uid())
                .map(|uid| uid == track_uid)
                .unwrap_or(false);
            if is_track_scoped {
                track_scoped = Some(parsed);
            } else if segment_wide.is_none() {
                segment_wide = Some(parsed);
            }
        }
    }
    track_scoped.or(segment_wide)
}
