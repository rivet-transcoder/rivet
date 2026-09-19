//! AV1 pixel-format detection and sequence/frame-header parsers.
//! See AV1 specification §5.5.2 (sequence header) and §5.9.1 (frame header).

use crate::PixelFormat;

mod frame;
mod obu;
mod sequence;

pub use frame::*;
pub use obu::*;
pub use sequence::*;

// ─── AV1 sequence header pixel-format detection ────────────────────
// See AV1 spec §5.5: the bit depth and the chroma subsampling are in the
// sequence header's color_config, which `parse_av1_sequence_header` reaches
// through the operating points. This used to stop at the timing info and
// answer 8-bit 4:2:0 for every stream with a full sequence header, so a
// 10-bit AV1 source — every HDR one — read as 8-bit.
pub(super) fn detect_av1(sample: &[u8]) -> Option<PixelFormat> {
    let seq = parse_av1_sequence_header(sample)?;
    let chroma_idc = match (seq.chroma_subsampling_x, seq.chroma_subsampling_y) {
        (true, true) => 1,
        (true, false) => 2,
        (false, false) => 3,
        (false, true) => return None,
    };
    Some(PixelFormat::from_chroma_and_depth(
        chroma_idc,
        seq.bit_depth,
    ))
}
