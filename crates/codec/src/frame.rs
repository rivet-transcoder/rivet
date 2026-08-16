//! Frame and stream value types.
//!
//! These live in the dependency-free `rivet-frame` crate so that
//! `rivet-container` can use them without pulling in this crate's GPU/audio
//! dependencies (which do not build for wasm32). Re-exported here unchanged.
pub use ::frame::*;
