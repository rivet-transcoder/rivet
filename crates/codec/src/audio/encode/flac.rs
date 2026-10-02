//! FLAC output through the workspace's lossless codecs (`lossless::flac`,
//! the `crates/lossless` submodule), adapted to [`AudioEncoder`].
//!
//! 4096-sample frames; [`FlacLevel`] trades speed for size (`Fast`: fixed
//! predictors; `Default`: LPC to order 8; `Best`: every LPC order to 12).
//! The stream's STREAMINFO — frame size bounds, sample count and the MD5 of
//! the audio — is complete once [`AudioEncoder::flush`] has run, which is
//! when a muxer asks for [`AudioEncoder::extra_data`].

pub use lossless::flac::{
    BLOCK_SIZE, Encoder as FlacEncoder, EncoderConfig as FlacEncoderConfig, Level as FlacLevel,
};

use crate::audio::{AudioEncoder, AudioEncoderConfig, AudioError, AudioFrame, EncodedAudioPacket};

/// FLAC through the [`AudioEncoder`] surface: pipeline f32 samples are
/// taken at `bits_per_sample` (exactly, for audio that came from integers
/// of at most that depth).
pub struct FlacAudioEncoder {
    inner: FlacEncoder,
    samples_out: u64,
}

impl FlacAudioEncoder {
    pub fn new(config: &AudioEncoderConfig, bits_per_sample: u8, level: FlacLevel) -> Result<Self, AudioError> {
        Ok(Self {
            inner: FlacEncoder::new(FlacEncoderConfig {
                sample_rate: config.sample_rate,
                channels: config.channels,
                bits_per_sample,
                level,
            })?,
            samples_out: 0,
        })
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

impl AudioEncoder for FlacAudioEncoder {
    fn encode(&mut self, frame: &AudioFrame) -> Result<Vec<EncodedAudioPacket>, AudioError> {
        let config = *self.inner.config();
        if frame.channels != config.channels {
            return Err(AudioError::Encode(format!(
                "flac: a {}-channel frame into a {}-channel encoder",
                frame.channels, config.channels
            )));
        }
        if frame.sample_rate != config.sample_rate {
            return Err(AudioError::Encode(format!(
                "flac: a {} Hz frame into a {} Hz encoder",
                frame.sample_rate, config.sample_rate
            )));
        }
        let bits = u32::from(config.bits_per_sample);
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

    /// The metadata blocks (STREAMINFO), final once flushed.
    fn extra_data(&self) -> Vec<u8> {
        self.inner.metadata_blocks()
    }

    fn sample_rate(&self) -> u32 {
        self.inner.config().sample_rate
    }
}
