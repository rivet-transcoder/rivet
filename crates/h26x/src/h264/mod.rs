//! The H.264 / AVC decoder.
//!
//! What is implemented: Baseline, Main and High profile *frame* pictures at
//! 8-bit 4:2:0 — CAVLC and CABAC entropy coding, I/P/B slices, every
//! macroblock and sub-macroblock type including P_Skip, B_Skip and the
//! spatial and temporal direct modes, multiple reference frames with
//! reordering, long-term references and adaptive marking (MMCO 1–6),
//! weighted prediction (explicit and implicit), the 8x8 transform and
//! Intra_8x8, scaling matrices, I_PCM, constrained intra prediction,
//! multiple slices per picture, frame_num gaps, POC types 0/1/2, cropping,
//! and the deblocking filter.
//!
//! What is refused with [`Error::Unsupported`](crate::Error::Unsupported)
//! (a caller with another decoder should hand the stream over): interlaced
//! coding (field pictures, MBAFF, PAFF), 4:2:2 / 4:4:4 and monochrome,
//! bit depths above 8, slice groups (FMO/ASO), data partitioning, SP/SI
//! slices, and lossless transform bypass.
//!
//! Module map: `sps` / `pps` / `slice` parse the parameter sets and slice
//! header; `cavlc` and `cabac_mb` parse a macroblock into an [`mb::MbLayer`];
//! `recon` turns it into samples (through `intra`, `inter`, `transform`);
//! `deblock` filters the finished picture; `dpb` owns POC, reference marking,
//! list construction and output order; `decoder` drives it all.

pub mod cabac_mb;
pub mod cavlc;
pub mod deblock;
pub mod decoder;
pub mod dpb;
pub mod frame;
pub mod inter;
pub mod intra;
pub mod mb;
pub mod pps;
pub mod recon;
pub mod slice;
pub mod sps;
pub mod tables;
#[allow(missing_docs, clippy::all)]
pub mod tables_gen;
pub mod transform;

pub use decoder::H264Decoder;
pub use pps::Pps;
pub use slice::{SliceHeader, SliceType};
pub use sps::Sps;
