//! ALAC (Apple Lossless) decode through the workspace's lossless codecs
//! (`lossless::alac`, the `crates/lossless` submodule), adapted to
//! [`AudioDecoder`].
//!
//! One packet is one ALAC frame, under the magic cookie
//! (`ALACSpecificConfig`) the container carries: 16, 20, 24 or 32 bits,
//! 1–8 channels. The decoded channels come out reordered from ALAC's
//! layouts (which lead with the centre channel) to the pipeline's native
//! order; [`lossless::alac::layout`] names the layout each count lands on.

pub use lossless::alac::{Config as AlacConfig, decode_frame};

use crate::audio::filter::{ChannelLabel, ChannelLayout};
use crate::audio::{AudioDecoder, AudioError, AudioFrame};

/// ALAC through the [`AudioDecoder`] surface.
pub struct AlacDecoder {
    inner: lossless::alac::Decoder,
    first_pts_us: Option<i64>,
}

impl AlacDecoder {
    /// `extra_data` is the magic cookie, in any of the wrappings
    /// [`AlacConfig::parse`] takes; it is required.
    pub fn new(extra_data: Option<&[u8]>) -> Result<Self, AudioError> {
        Ok(Self { inner: lossless::alac::Decoder::new(extra_data)?, first_pts_us: None })
    }

    pub fn config(&self) -> &AlacConfig {
        self.inner.config()
    }

    /// Decode one packet to interleaved integer samples in the pipeline's
    /// channel order, at the cookie's bit depth.
    pub fn decode_int(&mut self, packet: &[u8]) -> Result<Vec<i32>, AudioError> {
        Ok(self.inner.decode_int(packet)?)
    }
}

/// The pipeline's label for a speaker the codec names; `None` for the two
/// it has no label for (ALAC's eight-channel front left- and
/// right-of-centre pair).
fn label(s: lossless::Speaker) -> Option<ChannelLabel> {
    use lossless::Speaker::*;
    Some(match s {
        FL => ChannelLabel::FL,
        FR => ChannelLabel::FR,
        FC => ChannelLabel::FC,
        LFE => ChannelLabel::LFE,
        BL => ChannelLabel::BL,
        BR => ChannelLabel::BR,
        BC => ChannelLabel::BC,
        SL => ChannelLabel::SL,
        SR => ChannelLabel::SR,
        FLC | FRC => return None,
    })
}

impl AudioDecoder for AlacDecoder {
    fn decode(&mut self, packet: &[u8], pts: i64) -> Result<Vec<AudioFrame>, AudioError> {
        let first_pts_us = *self.first_pts_us.get_or_insert(pts);
        let before = self.inner.samples_decoded();
        let samples = self.decode_int(packet)?;
        if samples.is_empty() {
            return Ok(Vec::new());
        }
        let config = self.inner.config();
        let rate = config.sample_rate.max(1);
        Ok(vec![AudioFrame {
            samples: lossless::pcm::ints_to_f32(&samples, u32::from(config.bit_depth)),
            sample_rate: rate,
            channels: config.num_channels,
            pts: first_pts_us + (before as i64 * 1_000_000) / i64::from(rate),
        }])
    }

    fn flush(&mut self) -> Result<Vec<AudioFrame>, AudioError> {
        Ok(Vec::new())
    }

    /// The codec's layout when it is not the pipeline's default for the
    /// count: in practice four channels, which ALAC lays out as 4.0
    /// (C L R Cs), not the quad the pipeline assumes. Eight channels'
    /// front left- and right-of-centre pair has no label here, so an
    /// eight-channel stream is left to the default 7.1, that pair riding in
    /// the SL / SR slots.
    fn layout(&self) -> Option<ChannelLayout> {
        let channels = self.inner.config().num_channels;
        let labels: Vec<ChannelLabel> =
            lossless::alac::layout(channels)?.iter().map(|&s| label(s)).collect::<Option<_>>()?;
        let layout = ChannelLayout::new(labels).ok()?;
        (ChannelLayout::default_for(channels).ok().as_ref() != Some(&layout)).then_some(layout)
    }
}
