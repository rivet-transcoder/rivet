//! DTS core decode through the workspace's DTS decoder (`dts`, the
//! `crates/dts` submodule, the rivet-dts repository), adapted to
//! [`AudioDecoder`].
//!
//! - Packets are whole core frames (Matroska `A_DTS`, MP4 `dtsc`), each
//!   optionally followed by a DTS-HD extension substream, which is skipped:
//!   a DTS-HD track decodes as its lossy core. Up to 5.1 at ≤ 48 kHz.
//! - Output is in the native order for the core's `AMODE` layout, which
//!   [`AudioDecoder::layout`] names (`5.1(side)`, `2.1`, `4.0`, …).
//! - A frame with ADPCM-predicted subbands is [`AudioError::Unsupported`]
//!   (the D.10.1 code book is printed in no ETSI edition); the job layer
//!   drops it with the reason. Most disc-sourced DTS predicts somewhere.
//!   High-frequency VQ subbands decode as silence, which the spec allows,
//!   with one warning per decoder.
//! - The container's rate and channel count are only a cross-check: the
//!   core frame header is authoritative, and a disagreement is logged once.

use crate::audio::filter::{ChannelLabel, ChannelLayout};
use crate::audio::{AudioDecoder, AudioError, AudioFrame};

pub struct DtsDecoder {
    inner: dts::Decoder,
    /// Declared by the container; checked against the stream's own header.
    declared_sample_rate: u32,
    declared_channels: u8,
    next_pts_us: Option<i64>,
    hf_vq_warned: bool,
    layout_logged: bool,
}

fn decode_error(e: dts::Error) -> AudioError {
    match e {
        dts::Error::Unsupported(_) => AudioError::Unsupported(e.to_string()),
        other => AudioError::Decode(other.to_string()),
    }
}

/// The rivet label of a speaker the decoder names; `None` for the speakers
/// only the extensions carry (wides, heights, centres of front), which this
/// adapter does not decode.
fn label(s: dts::Speaker) -> Option<ChannelLabel> {
    use dts::Speaker::*;
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
        _ => return None,
    })
}

impl DtsDecoder {
    pub fn new(sample_rate: u32, channels: u8) -> Result<Self, AudioError> {
        if channels == 0 || channels > 6 {
            return Err(AudioError::Unsupported(format!(
                "DTS core decoder handles 1..=6 channels (5.1), container says {channels}"
            )));
        }
        // The core alone, as before the decoder grew its extensions: up to
        // 5.1 at the core's rate, which is what the pipeline's layouts and
        // the container's declared channel count describe.
        let mut inner = dts::Decoder::new();
        inner.set_core_only(true);
        Ok(Self {
            inner,
            declared_sample_rate: sample_rate,
            declared_channels: channels,
            next_pts_us: None,
            hf_vq_warned: false,
            layout_logged: false,
        })
    }

    /// The log lines the decoder used to write itself: what it is decoding
    /// (once, from the first header whose layout mapped, even if that frame
    /// was then refused), whether the container agrees, and the one HF VQ
    /// warning.
    fn report(&mut self) {
        if let (false, Some(info)) = (self.layout_logged, self.inner.info()) {
            self.layout_logged = true;
            tracing::info!(
                amode = info.amode,
                lfe = info.lfe,
                layout = info.layout.name(),
                sample_rate = info.sample_rate,
                filts = if info.perfect_reconstruction { "perfect" } else { "non-perfect" },
                "DTS core: decoding"
            );
            let channels = info.layout.channels();
            if info.sample_rate != self.declared_sample_rate || channels != self.declared_channels as usize {
                tracing::warn!(
                    container_rate = self.declared_sample_rate,
                    container_channels = self.declared_channels,
                    stream_rate = info.sample_rate,
                    stream_channels = channels,
                    "DTS core: container metadata disagrees with the bitstream; the bitstream wins"
                );
            }
        }
        if !self.hf_vq_warned && self.inner.hf_vq_skipped() {
            self.hf_vq_warned = true;
            tracing::warn!(
                "DTS core: high-frequency VQ subbands present; decoded as silence (the D.10.2 \
                 codebook is not published; the spec allows ignoring these subbands)"
            );
        }
    }
}

impl AudioDecoder for DtsDecoder {
    fn decode(&mut self, packet: &[u8], pts: i64) -> Result<Vec<AudioFrame>, AudioError> {
        if self.next_pts_us.is_none() {
            self.next_pts_us = Some(pts);
        }
        let decoded = self.inner.decode(packet);
        self.report();
        let frames = decoded.map_err(decode_error)?;
        let mut out = Vec::with_capacity(frames.len());
        for f in frames {
            let pts_us = self.next_pts_us.unwrap_or(0);
            let dur_us = (f.samples_per_channel() as i64 * 1_000_000) / f.sample_rate as i64;
            self.next_pts_us = Some(pts_us.saturating_add(dur_us));
            out.push(AudioFrame {
                samples: f.samples,
                sample_rate: f.sample_rate,
                channels: f.channels as u8,
                pts: pts_us,
            });
        }
        Ok(out)
    }

    fn flush(&mut self) -> Result<Vec<AudioFrame>, AudioError> {
        // Every core frame decodes to its full block of PCM as it arrives; the
        // QMF delay is inherent to the format and not flushed (matches the
        // spec's decoder, which emits 32·(NBLKS+1) samples per frame).
        Ok(Vec::new())
    }

    fn layout(&self) -> Option<ChannelLayout> {
        let layout = self.inner.layout()?;
        ChannelLayout::new(layout.speakers().iter().copied().map(label).collect::<Option<_>>()?).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decoder_construction_bounds_the_channel_count() {
        assert!(DtsDecoder::new(48_000, 0).is_err());
        assert!(DtsDecoder::new(48_000, 7).is_err());
        assert!(DtsDecoder::new(48_000, 6).is_ok());
    }

    #[test]
    fn errors_map_onto_audio_errors_with_their_messages() {
        let mut d = DtsDecoder::new(48_000, 6).unwrap();
        let err = d.decode(&[0x0B, 0x77, 0, 0, 0, 0, 0, 0, 0, 0], 0).unwrap_err();
        assert!(matches!(err, AudioError::Decode(_)), "{err}");
        assert!(err.to_string().contains("0x7FFE8001"), "{err}");
        let err = decode_error(dts::Error::Unsupported("ADPCM prediction".into()));
        assert!(matches!(err, AudioError::Unsupported(_)), "{err}");
        assert_eq!(err.to_string(), "unsupported: DTS: unsupported: ADPCM prediction");
    }

    #[test]
    fn every_dts_layout_is_a_named_pipeline_layout() {
        use dts::Layout::*;
        for l in [Mono, Stereo, Stereo21, Surround30, Surround40, QuadSide, Surround50Side, Surround51Side] {
            let ours = ChannelLayout::new(l.speakers().iter().copied().map(|s| label(s).unwrap()).collect()).unwrap();
            assert_eq!(ours, ChannelLayout::named(l.name()), "{l}");
        }
    }
}
