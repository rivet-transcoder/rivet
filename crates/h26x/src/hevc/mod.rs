//! H.265 / HEVC decoding: Main and Main 10 profiles (4:2:0, 8..12-bit),
//! written from ITU-T H.265 (V11, 01/2026) with the same architecture as the
//! H.264 half of the crate: parameter sets, slice segments, CTU parsing and
//! reconstruction, deblocking + SAO, and an output-order DPB.

pub mod ctu;
pub mod ctx;
pub mod deblock;
pub mod decoder;
pub mod dpb;
pub mod frame;
pub mod inter;
pub mod intra;
pub mod mvpred;
pub mod pic;
pub mod pps;
pub mod residual;
pub mod sao;
pub mod slice;
pub mod sps;
pub mod tables;
pub mod tables_gen;

pub use decoder::HevcDecoder;
