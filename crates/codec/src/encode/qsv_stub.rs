//! Stub Intel QSV encoder, compiled when the `qsv` feature is **off**.
//!
//! Keeps `qsv::QsvEncoder` a real type so the dispatcher in
//! `encode/mod.rs` compiles unchanged, but construction always errors —
//! auto-select then skips the Intel tier, and an explicit
//! `EncoderBackend::Qsv` request surfaces the error. Enable `--features
//! qsv` to compile the real oneVPL-backed encoder (`qsv.rs`).

use anyhow::{Result, bail};

use super::{EncodedPacket, Encoder, EncoderConfig};
use crate::frame::VideoFrame;

pub struct QsvEncoder;

impl QsvEncoder {
    pub fn new(_config: EncoderConfig, _gpu_index: u32) -> Result<Self> {
        bail!(
            "QSV (Intel oneVPL) encode support was not compiled in; \
             rebuild with the `qsv` feature enabled to use Intel hardware encode"
        )
    }

    /// Mirror of the real encoder's unpinned constructor.
    ///
    /// The stub exists so `encode/mod.rs` compiles unchanged with the feature
    /// off, which only holds if it carries every constructor the dispatcher
    /// calls. When `new_unpinned` was added to the real encoder it was not
    /// added here, and `--no-default-features` stopped compiling — silently,
    /// because every build that mattered enabled `qsv`.
    pub fn new_unpinned(_config: EncoderConfig) -> Result<Self> {
        Self::new(_config, 0)
    }
}

impl Encoder for QsvEncoder {
    fn send_frame(&mut self, _frame: &VideoFrame) -> Result<()> {
        unreachable!("stub QSV encoder is never constructed")
    }
    fn flush(&mut self) -> Result<()> {
        unreachable!("stub QSV encoder is never constructed")
    }
    fn receive_packet(&mut self) -> Result<Option<EncodedPacket>> {
        unreachable!("stub QSV encoder is never constructed")
    }
}
