//! Stub AMF encoder, compiled when the `amd` feature is **off**.
//!
//! Keeps `amf::AmfEncoder` a real type so the dispatcher in `encode/mod.rs`
//! compiles unchanged, but construction always errors — auto-select then skips
//! the AMD tier. Enable `--features amd` to compile the real
//! `shiguredo_amf`-backed encoder (`amf.rs`).

use anyhow::{Result, bail};

use super::{EncodedPacket, Encoder, EncoderConfig};
use crate::frame::VideoFrame;

pub struct AmfEncoder;

impl AmfEncoder {
    pub fn new(_config: EncoderConfig, _gpu_index: u32) -> Result<Self> {
        bail!(
            "AMF encode support was not compiled in; rebuild with the `amd` feature \
             (shiguredo_amf) to use AMD hardware encode"
        )
    }
}

impl Encoder for AmfEncoder {
    fn send_frame(&mut self, _frame: &VideoFrame) -> Result<()> {
        unreachable!("stub AMF encoder is never constructed")
    }
    fn flush(&mut self) -> Result<()> {
        unreachable!("stub AMF encoder is never constructed")
    }
    fn receive_packet(&mut self) -> Result<Option<EncodedPacket>> {
        unreachable!("stub AMF encoder is never constructed")
    }
}
