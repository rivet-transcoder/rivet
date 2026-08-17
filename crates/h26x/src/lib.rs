//! Native H.264/AVC and H.265/HEVC decoders.
//!
//! Pure Rust, no C, no system libraries: the software decode tier for the two
//! codecs every camera, phone and broadcast chain emits. Written from the
//! ITU-T specifications (H.264 and H.265, decoding processes in clauses 7–9)
//! with libavcodec's decoder *architecture* as the model — parameter-set
//! tables, a slice-driven decode loop over a decoded-picture buffer, entropy
//! decoding into per-block syntax, then prediction, inverse transform,
//! reconstruction and the in-loop filters, with the pixel kernels behind a
//! runtime-dispatched DSP layer ([`dsp`]) so AVX2 and NEON kernels can
//! replace the scalar reference paths without the decoders knowing.
//!
//! Both decoders are **bit-exact**: the standards define the decoding process
//! completely, so a correct decoder reproduces the reference decoder's output
//! to the sample. That is what the verification checks — MD5s of decoded
//! frames against libavcodec on the workspace test media, and against the
//! ITU/JCT-VC conformance bitstreams (see the crate README for the numbers).
//!
//! # Layout
//!
//! - [`bitreader`] — RBSP bit reader (Exp-Golomb, fixed-width, alignment).
//! - [`nal`] — Annex-B start-code splitting and emulation-prevention removal.
//! - [`cabac`] — the arithmetic decoding engine both standards share.
//! - [`h264`] — the H.264 decoder: parameter sets, slice header, POC,
//!   reference lists and DPB, CAVLC + CABAC macroblock parsing, intra/inter
//!   prediction, transforms, deblocking, output ordering.
//! - [`hevc`] — the H.265 decoder: VPS/SPS/PPS, slice segment header, RPS
//!   and DPB, CTU quadtree parsing (CABAC), intra/inter prediction with
//!   merge/AMVP/TMVP, transforms, deblocking, SAO, tiles and WPP entry points.
//! - [`dsp`] — the kernels: interpolation filters, inverse transforms,
//!   deblocking and SAO edges, scalar and SIMD.
//! - [`picture`] — the decoded picture type the two decoders hand back.
//!
//! # Provenance
//!
//! This is not a translation of libavcodec's C (which is LGPL and could not be
//! carried under this crate's license). The structure follows FFmpeg because
//! it is the right structure; the code follows the standard.

#![warn(missing_docs)]

pub mod bitreader;
pub mod cabac;
pub mod dsp;
pub mod h264;
pub mod hevc;
pub mod nal;
pub mod picture;

pub use picture::{ChromaFormat, Picture, Plane};

/// Errors a decoder can report.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The bitstream is malformed: a syntax element out of range, a NAL cut
    /// short, a reference to a parameter set that was never sent.
    #[error("bitstream error: {0}")]
    Bitstream(String),
    /// The stream is valid but uses a feature this decoder does not implement
    /// (yet). The message names it. A caller with another decoder available
    /// should hand the stream to it.
    #[error("unsupported: {0}")]
    Unsupported(String),
}

/// `Result` with this crate's [`Error`].
pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    pub(crate) fn bitstream(msg: impl Into<String>) -> Self {
        Error::Bitstream(msg.into())
    }
    pub(crate) fn unsupported(msg: impl Into<String>) -> Self {
        Error::Unsupported(msg.into())
    }
}
