//! ALAC (Apple Lossless) output through the workspace's lossless codecs
//! (`lossless::alac`, the `crates/lossless` submodule), adapted to
//! [`AudioEncoder`].
//!
//! 4096-sample frames in ALAC's channel order for the count (the encoder
//! reorders from the pipeline's), 16, 20, 24 or 32 bits. The magic cookie's
//! largest-frame and average-bit-rate fields are final once
//! [`AudioEncoder::flush`] has run, which is when a muxer asks for
//! [`AudioEncoder::extra_data`].

pub use lossless::alac::Encoder as AlacEncoder;

use crate::audio::{AudioEncoder, AudioEncoderConfig, AudioError, AudioFrame, EncodedAudioPacket};

/// ALAC through the [`AudioEncoder`] surface: pipeline f32 samples are
/// taken at `bits_per_sample`.
pub struct AlacAudioEncoder {
    inner: AlacEncoder,
    samples_out: u64,
}

impl AlacAudioEncoder {
    pub fn new(config: &AudioEncoderConfig, bits_per_sample: u8) -> Result<Self, AudioError> {
        Ok(Self { inner: AlacEncoder::new(config.sample_rate, config.channels, bits_per_sample)?, samples_out: 0 })
    }

    fn packets(&mut self, frames: Vec<(Vec<u8>, u32)>) -> Vec<EncodedAudioPacket> {
        let rate = i64::from(self.inner.config().sample_rate);
        frames
            .into_iter()
            .map(|(data, n)| {
                let pts = self.samples_out as i64 * 1_000_000 / rate;
                self.samples_out += u64::from(n);
                EncodedAudioPacket { data, pts, duration: i64::from(n) }
            })
            .collect()
    }
}

impl AudioEncoder for AlacAudioEncoder {
    fn encode(&mut self, frame: &AudioFrame) -> Result<Vec<EncodedAudioPacket>, AudioError> {
        let cfg = self.inner.config();
        if frame.channels != cfg.num_channels || frame.sample_rate != cfg.sample_rate {
            return Err(AudioError::Encode(format!(
                "alac: a {}-channel {} Hz frame into a {}-channel {} Hz encoder",
                frame.channels, frame.sample_rate, cfg.num_channels, cfg.sample_rate
            )));
        }
        let bits = u32::from(cfg.bit_depth);
        let ints: Vec<i32> = frame.samples.iter().map(|&x| lossless::pcm::f32_to_int(x, bits)).collect();
        let frames = self.inner.encode_int(&ints);
        Ok(self.packets(frames))
    }

    fn flush(&mut self) -> Result<Vec<EncodedAudioPacket>, AudioError> {
        let frames = self.inner.finish();
        Ok(self.packets(frames))
    }

    fn pre_skip(&self) -> u16 {
        0
    }

    /// The 24-byte magic cookie, with its frame-size and bit-rate fields
    /// final once flushed.
    fn extra_data(&self) -> Vec<u8> {
        self.inner.cookie().to_bytes().to_vec()
    }

    fn sample_rate(&self) -> u32 {
        self.inner.config().sample_rate
    }
}
