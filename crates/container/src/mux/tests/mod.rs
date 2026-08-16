// Shared fixtures and helpers visible to all test sub-modules.
// Item names available to sub-files via `use super::<name>;`.

use frame::MasteringDisplay;

mod boxes;
mod video;
mod audio_opus;
mod audio_ac3;

// ---- shared helpers -------------------------------------------------------

/// Find the first occurrence of a 4-byte tag in a byte slice.
/// Returns the offset of the tag itself (i.e. the four matching bytes),
/// not the preceding size field.
pub(super) fn find_fourcc(data: &[u8], tag: &[u8; 4]) -> Option<usize> {
    data.windows(4).position(|w| w == tag)
}

/// Count every occurrence of a 4-byte tag (used where the tag may appear
/// inside payload bytes as well as in box headers).
pub(super) fn count_fourcc_occurrences(data: &[u8], tag: &[u8; 4]) -> usize {
    data.windows(4).filter(|w| *w == tag).count()
}

// ---- shared fixture -------------------------------------------------------

/// HDR10-canonical mastering display values: BT.2020 primaries +
/// D65 white point + 1000 nits / 0.0001 nits luminance, all in the
/// HEVC SEI 137 / SMPTE ST 2086 spec-domain integer encoding.
///
/// Cross-references for the wire numbers (so future reviewers can
/// re-derive without chasing a spec PDF):
///   BT.2020 R primary  (0.708 , 0.292)  → (35400, 14600)
///   BT.2020 G primary  (0.170 , 0.797)  → ( 8500, 39850)
///   BT.2020 B primary  (0.131 , 0.046)  → ( 6550,  2300)
///   D65 white point    (0.3127, 0.3290) → (15635, 16450)
///   max luminance       1000 cd/m²      → 10_000_000  (0.0001 cd/m² steps)
///   min luminance       0.0001 cd/m²    →          1
pub(super) fn hdr10_mastering_display() -> MasteringDisplay {
    MasteringDisplay {
        primaries_r_x: 35400,
        primaries_r_y: 14600,
        primaries_g_x: 8500,
        primaries_g_y: 39850,
        primaries_b_x: 6550,
        primaries_b_y: 2300,
        white_point_x: 15635,
        white_point_y: 16450,
        max_luminance: 10_000_000,
        min_luminance: 1,
    }
}
