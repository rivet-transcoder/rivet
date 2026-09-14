//! HEVC SEI extractor for HDR static metadata (mastering display colour volume,
//! content light level).
//!
//! Moved to `rivet-frame` as [`frame::hdr_sei`](::frame::hdr_sei), which reads
//! the H.264 SEIs too, so the demuxers can fill a source's static metadata from
//! the bitstream when the container carries none. Re-exported here unchanged.

pub use ::frame::hdr_sei::{HdrSei, HevcHdrSei, parse_annexb, parse_annexb_for, parse_h264_annexb};
