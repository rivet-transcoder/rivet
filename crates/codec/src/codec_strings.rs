//! HLS / DASH `CODECS` attribute string generation.
//!
//! Generates the precise codec-string bytes that go into the
//! `#EXT-X-STREAM-INF:CODECS="..."` line of a HLS master playlist.
//! These strings are what hls.js (and Safari's native HLS, and DASH
//! players) use to decide whether the browser can play a given variant
//! BEFORE downloading any media bytes. A wrong string causes the
//! variant to be silently skipped, so they have to be parsed from the
//! actual bitstream — never composed from a config file.
//!
//! Sources of truth:
//! - AV1: AV1 Codec ISO Media File Format Binding v1.2.0 §A.3,
//!   "Codecs Parameter String"
//! - AAC-LC in MP4: ISO/IEC 14496-3 + RFC 6381 §3.3
//! - AVC: RFC 6381 §3.3 (`avc1.PPCCLL` hex from SPS)
//! - HEVC: ISO/IEC 14496-15 §A.5
//!
//! We currently emit AV1 and AAC strings. AVC / HEVC formatters are
//! sketched as future work for when those codecs ride through the
//! pipeline as outputs.

use crate::pixel_format::Av1SequenceHeader;

/// AV1 codec string — `av01.P.LLT.DD.M.CCC.TTT.MMM.F`.
///
/// Per the AV1 ISOBMFF binding §A.3:
///   - `P` = `seq_profile` (decimal, 1 char). Profile 0 (Main) is by
///     far the most common; 1 (High) and 2 (Professional) are rare.
///   - `LL` = `seq_level_idx_0` formatted as 2-digit decimal (00..31).
///   - `T` = `seq_tier_0` mapped to 'M' (Main, 0) or 'H' (High, 1).
///     Tier is only signaled in the bitstream for levels >= 4.0
///     (level_idx > 7); the parser implicitly sets it to 0 below
///     that.
///   - `DD` = bit depth as 2-digit decimal (08, 10, or 12).
///   - `M` = `monochrome` flag (0 or 1).
///   - `CCC.TTT.MMM` = `color_primaries`, `transfer_characteristics`,
///     `matrix_coefficients` formatted as 3-digit zero-padded
///     decimals. H.273 codes 1/1/1 = BT.709, 9/16/9 = BT.2020 PQ
///     (HDR10), 9/18/9 = BT.2020 HLG, etc.
///   - `F` = `color_range` flag (0 = limited / studio, 1 = full).
///
/// Per spec, the optional tail (`.M.CCC.TTT.MMM.F`) MAY be omitted
/// when ALL of these are at their defaults (M=0, CCC=001, TTT=001,
/// MMM=001, F=0 — i.e. SDR BT.709 limited). We emit the SHORT form
/// when at defaults and the LONG form otherwise.
///
/// The original posture was "always emit long for explicit
/// identification", but that broke playback in the browser MSE path:
/// some hls.js / Chrome / Edge versions reject the long form via
/// `MediaSource.isTypeSupported('video/mp4; codecs="av01.0.05M.08.0.001.001.001.0"')`
/// even though the underlying av1C bitstream is byte-identical to
/// what the same browser plays via direct rendition load (which
/// internally generates the short form by inferring codec from
/// init.mp4 — bypassing the long-form attribute path). Switching the
/// master playlist to short-form when at defaults makes the same
/// segments decode consistently across native HLS, hls.js, and
/// Safari.
///
/// The HDR / wide-gamut / monochrome / non-default-range case still
/// emits the full 9-component form — those values are NOT defaults
/// and short form would mean "BT.709 limited 8-bit" which is wrong.
pub fn av1_codec_string(h: &Av1SequenceHeader) -> String {
    let tier_char = if h.seq_tier_0 == 0 { 'M' } else { 'H' };
    let at_defaults = !h.monochrome
        && h.color_primaries == 1
        && h.transfer_characteristics == 1
        && h.matrix_coefficients == 1
        && !h.color_range;
    if at_defaults {
        format!(
            "av01.{}.{:02}{}.{:02}",
            h.seq_profile, h.seq_level_idx_0, tier_char, h.bit_depth,
        )
    } else {
        format!(
            "av01.{}.{:02}{}.{:02}.{}.{:03}.{:03}.{:03}.{}",
            h.seq_profile,
            h.seq_level_idx_0,
            tier_char,
            h.bit_depth,
            u8::from(h.monochrome),
            h.color_primaries,
            h.transfer_characteristics,
            h.matrix_coefficients,
            u8::from(h.color_range),
        )
    }
}

/// AAC-LC in MP4 codec string. Always `mp4a.40.2`:
///   - `mp4a` = ISO/IEC 14496 sample entry fourcc
///   - `40`   = ObjectTypeIndication for MPEG-4 Audio (decimal 64,
///              hex 0x40)
///   - `2`    = Audio Object Type 2 (AAC-LC) per ISO/IEC 14496-3
///              Table 1.16
///
/// HE-AAC v1 = `mp4a.40.5`, HE-AAC v2 = `mp4a.40.29`. We don't emit
/// those today — the audio rendition is always AAC-LC stereo at 48
/// kHz per the CMAF ladder defaults — but if the worker ever
/// passes-through HE-AAC source, this needs to inspect the AOT
/// signaled in the AudioSpecificConfig and switch. Until then,
/// callers using the constant string are correct.
pub const AAC_LC_CODEC_STRING: &str = "mp4a.40.2";

/// Convenience: pack an HLS `CODECS=` attribute value for a variant
/// that carries one video and one audio track. Order is
/// `<video>,<audio>` per RFC 8216 §4.3.4.2 and HLS-Authoring spec.
pub fn hls_codecs_attribute(video: &str, audio: &str) -> String {
    format!("{video},{audio}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synth_seq_header(
        seq_profile: u8,
        seq_level_idx_0: u8,
        seq_tier_0: u8,
        bit_depth: u8,
        monochrome: bool,
        color_primaries: u8,
        transfer_characteristics: u8,
        matrix_coefficients: u8,
        color_range: bool,
    ) -> Av1SequenceHeader {
        Av1SequenceHeader {
            seq_profile,
            still_picture: false,
            reduced_still_picture_header: false,
            max_frame_width_minus1: 0,
            max_frame_height_minus1: 0,
            seq_level_idx_0,
            seq_tier_0,
            bit_depth,
            monochrome,
            color_primaries,
            transfer_characteristics,
            matrix_coefficients,
            color_range,
            chroma_subsampling_x: true,
            chroma_subsampling_y: true,
            film_grain_params_present: false,
            enable_filter_intra: false,
            enable_intra_edge_filter: false,
            enable_interintra_compound: false,
            enable_masked_compound: false,
            enable_warped_motion: false,
            enable_dual_filter: false,
            enable_order_hint: false,
            enable_jnt_comp: false,
            enable_ref_frame_mvs: false,
            enable_superres: false,
            enable_cdef: false,
            enable_restoration: false,
            order_hint_bits: 0,
            seq_force_screen_content_tools: 0,
            seq_force_integer_mv: 0,
            frame_width_bits_minus_1: 0,
            frame_height_bits_minus_1: 0,
            use_128x128_superblock: false,
            separate_uv_delta_q: false,
        }
    }

    #[test]
    fn av1_string_short_form_at_bt709_defaults() {
        // Profile 0, level_idx 8 (level 4.0), Main tier, 8-bit, SDR BT.709 limited.
        // The "boring 1080p" baseline string — at all defaults so the
        // short form is correct. Long form here was rejected by Chrome
        // / hls.js MediaSource.isTypeSupported in 2026-05-02 testing
        // (manifest_url playback dropped video while audio worked).
        let h = synth_seq_header(0, 8, 0, 8, false, 1, 1, 1, false);
        assert_eq!(av1_codec_string(&h), "av01.0.08M.08");
    }

    #[test]
    fn av1_string_high_tier_renders_h_character() {
        // Level 6.0 (idx 16), High tier — tier_char swaps M -> H.
        // Bit depth + color codes deviate from defaults so long form is correct.
        let h = synth_seq_header(0, 16, 1, 10, false, 9, 16, 9, false);
        assert_eq!(av1_codec_string(&h), "av01.0.16H.10.0.009.016.009.0");
    }

    #[test]
    fn av1_string_hdr10_bt2020_pq_full_range() {
        // BT.2020 + PQ + BT.2020 NCL + full range = HDR10 limited PQ.
        // CCC=009, TTT=016, MMM=009, F=1. Long form REQUIRED — short
        // form at defaults would mis-signal as BT.709 SDR.
        let h = synth_seq_header(0, 12, 0, 10, false, 9, 16, 9, true);
        assert_eq!(av1_codec_string(&h), "av01.0.12M.10.0.009.016.009.1");
    }

    #[test]
    fn av1_string_monochrome_uses_long_form() {
        // Monochrome is non-default — long form required so the player
        // doesn't allocate a chroma buffer that won't get filled.
        let h = synth_seq_header(0, 8, 0, 8, true, 1, 1, 1, false);
        assert_eq!(av1_codec_string(&h), "av01.0.08M.08.1.001.001.001.0");
    }

    #[test]
    fn av1_string_full_range_at_8bit_bt709_uses_long_form() {
        // Full range != 0 so even with BT.709 / 8-bit the SDR-defaults
        // check fails — long form required so the player applies
        // full-range scaling.
        let h = synth_seq_header(0, 8, 0, 8, false, 1, 1, 1, true);
        assert_eq!(av1_codec_string(&h), "av01.0.08M.08.0.001.001.001.1");
    }

    #[test]
    fn av1_string_two_digit_level_padding() {
        // level_idx 0 must format as "00", not "0".
        let h = synth_seq_header(0, 0, 0, 8, false, 1, 1, 1, false);
        let s = av1_codec_string(&h);
        assert!(s.starts_with("av01.0.00M."), "got: {s}");
    }

    #[test]
    fn av1_string_two_digit_bit_depth_padding() {
        // 8-bit at defaults → short form; 10-bit + 12-bit deviate from
        // bit-depth=8 (which is the implicit default carried by short
        // form) but they're still valid as short form so long as
        // color codes are at default.
        let h_8 = synth_seq_header(0, 8, 0, 8, false, 1, 1, 1, false);
        let h_10 = synth_seq_header(0, 8, 0, 10, false, 1, 1, 1, false);
        let h_12 = synth_seq_header(2, 8, 0, 12, false, 1, 1, 1, false);
        assert_eq!(av1_codec_string(&h_8), "av01.0.08M.08");
        assert_eq!(av1_codec_string(&h_10), "av01.0.08M.10");
        assert_eq!(av1_codec_string(&h_12), "av01.2.08M.12");
    }

    #[test]
    fn aac_lc_constant_is_canonical() {
        assert_eq!(AAC_LC_CODEC_STRING, "mp4a.40.2");
    }

    #[test]
    fn hls_codecs_attribute_concatenates_video_then_audio() {
        let s = hls_codecs_attribute("av01.0.08M.08.0.001.001.001.0", AAC_LC_CODEC_STRING);
        assert_eq!(s, "av01.0.08M.08.0.001.001.001.0,mp4a.40.2");
    }
}
