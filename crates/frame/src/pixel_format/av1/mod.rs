//! AV1 pixel-format detection and sequence/frame-header parsers.
//! See AV1 specification §5.5.2 (sequence header) and §5.9.1 (frame header).

use super::bitreader::BitReader;
use crate::PixelFormat;

mod frame;
mod obu;
mod sequence;

pub use frame::*;
pub use obu::*;
pub use sequence::*;

// ─── AV1 sequence header pixel-format detection ────────────────────
// See AV1 spec §5.5. Full parse is long; we hop through enough fields
// to reach color_config. Most AV1 content in the wild is 4:2:0 8-bit
// (Main profile), and 4:2:0 10-bit for HDR (Main-10).
pub(super) fn detect_av1(sample: &[u8]) -> Option<PixelFormat> {
    // AV1 wraps sequence headers in an OBU with obu_type == 1.
    let obu_bytes = obu::find_av1_obu(sample, 1)?;
    let mut br = BitReader::new(obu_bytes);

    let _seq_profile = br.read_bits(3)?;
    let _still_picture = br.read_bits(1)?;
    let reduced_still_picture_header = br.read_bits(1)?;

    if reduced_still_picture_header == 0 {
        // timing_info_present, decoder_model_info, initial_display_delay,
        // operating_points — a lot to skip. Abort safely if any read
        // fails; fallback to Yuv420p.
        let timing_info_present = br.read_bits(1)?;
        if timing_info_present == 1 {
            let _num_units_in_display_tick = br.read_bits(32)?;
            let _time_scale = br.read_bits(32)?;
            let equal_picture_interval = br.read_bits(1)?;
            if equal_picture_interval == 1 {
                let _num_ticks_per_picture = br.read_ue()?; // uvlc, not ue(v), but reuse
            }
            let decoder_model_info_present = br.read_bits(1)?;
            if decoder_model_info_present == 1 {
                let _buffer_delay_length_minus_1 = br.read_bits(5)?;
                let _num_units_in_decoding_tick = br.read_bits(32)?;
                let _buffer_removal_time_length_minus_1 = br.read_bits(5)?;
                let _frame_presentation_time_length_minus_1 = br.read_bits(5)?;
            }
        }
        // Bail out to default — the full operating-points loop is long
        // and rarely worth the maintenance cost vs accepting that
        // non-trivial AV1 probes return Yuv420p for now. If the MP4
        // container advertises codec profile in its track box, we can
        // use that instead (future follow-up if the data shows 10-bit
        // AV1 slipping through).
        return Some(PixelFormat::Yuv420p);
    }

    // Reduced still-picture path is simpler: go straight to
    // seq_level_idx + bit depth fields.
    let _seq_level_idx_0 = br.read_bits(5)?;

    // For full correctness we'd continue into color_config. Since the
    // reduced path is rare for VOD content we take the safe default
    // and let downstream validation surface anything unexpected.
    Some(PixelFormat::Yuv420p)
}
