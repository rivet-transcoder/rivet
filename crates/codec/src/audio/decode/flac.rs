//! FLAC decode through the workspace's lossless codecs (`lossless::flac`,
//! the `crates/lossless` submodule), adapted to [`AudioDecoder`].
//!
//! One packet is one or more whole FLAC frames (an MP4 `fLaC` sample, a
//! Matroska `A_FLAC` block, a frame the native-stream reader cut). Every
//! subframe type, 4–32 bits, 1–8 channels; both CRCs are checked, and the
//! STREAMINFO MD5 at the end of the stream (a mismatch is logged at
//! [`AudioDecoder::flush`]). FLAC's channel order for every count is the
//! pipeline's native order, so [`AudioDecoder::layout`] is the default.
//!
//! The decoder's integer output ([`FlacDecoder::decode_int`]) is exact at
//! every depth; the f32 frames are exact up to 24 bits (see
//! [`lossless::pcm`]).

pub use lossless::flac::{DecodedFrame, FrameHeader, StreamInfo, decode_frame, parse_frame_header};

use crate::audio::{AudioDecoder, AudioError, AudioFrame};

/// FLAC through the [`AudioDecoder`] surface.
pub struct FlacDecoder {
    inner: lossless::flac::Decoder,
    first_pts_us: Option<i64>,
}

impl FlacDecoder {
    /// `extra_data` is the codec configuration: an MP4 `dfLa` body or a
    /// Matroska CodecPrivate (`fLaC` + metadata blocks). Without it, every
    /// frame header has to be self-describing.
    pub fn new(extra_data: Option<&[u8]>, sample_rate: u32, channels: u8) -> Result<Self, AudioError> {
        Ok(Self { inner: lossless::flac::Decoder::new(extra_data, sample_rate, channels)?, first_pts_us: None })
    }

    /// The stream's STREAMINFO, when the configuration carried one.
    pub fn stream_info(&self) -> Option<&StreamInfo> {
        self.inner.stream_info()
    }

    /// Decode every frame in `packet` to interleaved integer samples, with
    /// the channel count and bit depth they are at.
    pub fn decode_int(&mut self, packet: &[u8]) -> Result<(Vec<i32>, u8, u32), AudioError> {
        Ok(self.inner.decode_int(packet)?)
    }

    /// Whether the audio decoded so far hashes to STREAMINFO's MD5: `None`
    /// when there is none to compare, or the stream has not been decoded to
    /// its stated end.
    pub fn md5_matches(&self) -> Option<bool> {
        self.inner.md5_matches()
    }
}

impl AudioDecoder for FlacDecoder {
    fn decode(&mut self, packet: &[u8], pts: i64) -> Result<Vec<AudioFrame>, AudioError> {
        let first_pts_us = *self.first_pts_us.get_or_insert(pts);
        let before = self.inner.samples_decoded();
        let (samples, channels, bits) = self.decode_int(packet)?;
        if samples.is_empty() {
            return Ok(Vec::new());
        }
        let sample_rate = self.inner.sample_rate();
        let pts = first_pts_us + (before as i64 * 1_000_000) / i64::from(sample_rate.max(1));
        Ok(vec![AudioFrame { samples: lossless::pcm::ints_to_f32(&samples, bits), sample_rate, channels, pts }])
    }

    fn flush(&mut self) -> Result<Vec<AudioFrame>, AudioError> {
        if self.md5_matches() == Some(false) {
            tracing::warn!("flac: the decoded audio does not match the stream's MD5 signature");
        }
        Ok(Vec::new())
    }
}
