//! Stub NVENC encoder, compiled when the `nvidia` feature is **off**.
//!
//! Keeps `nvenc::NvencEncoder` a real type so the dispatcher in
//! `encode/mod.rs` compiles unchanged, but construction always errors —
//! auto-select then skips the NVIDIA tier. Enable `--features nvidia` to
//! compile the real `shiguredo_nvcodec`-backed encoder (`nvenc.rs`).

use anyhow::{Result, bail};

use super::{EncodedPacket, Encoder, EncoderConfig};
use crate::frame::VideoFrame;

pub struct NvencEncoder;

impl NvencEncoder {
    pub fn new(_config: EncoderConfig, _gpu_index: u32) -> Result<Self> {
        bail!(
            "NVENC encode support was not compiled in; rebuild with the `nvidia` feature \
             (shiguredo_nvcodec) to use NVIDIA hardware encode"
        )
    }
}

impl Encoder for NvencEncoder {
    fn send_frame(&mut self, _frame: &VideoFrame) -> Result<()> {
        unreachable!("stub NVENC encoder is never constructed")
    }
    fn flush(&mut self) -> Result<()> {
        unreachable!("stub NVENC encoder is never constructed")
    }
    fn receive_packet(&mut self) -> Result<Option<EncodedPacket>> {
        unreachable!("stub NVENC encoder is never constructed")
    }
}
