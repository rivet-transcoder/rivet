//! AC-3 / E-AC-3 (Dolby Digital / Digital Plus) decoder, in-tree and
//! pure Rust, written from ATSC A/52:2018.
//!
//! The stages below the syncframe parser: the normative tables (with a
//! test per table), the bit reader, the parametric bit allocation and the
//! inverse transform.

pub mod bitalloc;
mod bits;
pub mod imdct;
pub mod tables;
