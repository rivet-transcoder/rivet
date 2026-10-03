//! AAC decode through the workspace's AAC decoder (`aac::decode`, the
//! `crates/aac` submodule), adapted to [`AudioDecoder`].
//!
//! - Packets are raw access units under the AudioSpecificConfig the demuxer
//!   found (MP4 `esds`, Matroska CodecPrivate, or the one the TS demuxer
//!   synthesises from the first ADTS header), or, with no configuration,
//!   ADTS bytes in any chunking.
//! - AAC-LC is decoded fully: channel configurations 1–7 and
//!   program_config_element layouts, every AAC-LC tool. Output is in the
//!   native order for the layout, which [`AudioDecoder::layout`] names.
//! - HE-AAC and HE-AAC v2 decode in full: spectral band replication at the
//!   SBR rate (twice the core's), parametric stereo to two channels. A
//!   caller that wants the AAC-LC core alone (half the rate, the core's
//!   channels, the old behaviour, [`HE_AAC_CORE_NOTE`]) builds the decoder
//!   with [`AacDecoder::new_core_only`]. [`probe`] says what a stream is
//!   before a job commits to decoding it.
//! - AAC Main, SSR, LTP and the other object types, and coupling channel
//!   elements, are [`AudioError::Unsupported`].

pub use aac::decode::HE_AAC_CORE_NOTE;

use crate::audio::filter::{ChannelLabel, ChannelLayout};
use crate::audio::{AudioDecoder, AudioError, AudioFrame};

pub struct AacDecoder {
    inner: aac::decode::Decoder,
    layout: Option<ChannelLayout>,
}

fn decode_error(e: aac::Error) -> AudioError {
    match e {
        aac::Error::Unsupported(m) => AudioError::Unsupported(format!("aac: {m}")),
        aac::Error::Invalid(m) | aac::Error::Config(m) => AudioError::Decode(format!("aac: {m}")),
    }
}

/// The rivet label of a speaker the decoder names.
fn label(s: aac::decode::Speaker) -> ChannelLabel {
    use aac::decode::Speaker::*;
    match s {
        FL => ChannelLabel::FL,
        FR => ChannelLabel::FR,
        FC => ChannelLabel::FC,
        LFE => ChannelLabel::LFE,
        BL => ChannelLabel::BL,
        BR => ChannelLabel::BR,
        BC => ChannelLabel::BC,
        SL => ChannelLabel::SL,
        SR => ChannelLabel::SR,
    }
}

impl AacDecoder {
    /// `asc` is the AudioSpecificConfig for raw access units; `None` (or
    /// empty) reads the packets as ADTS. HE-AAC decodes in full.
    pub fn new(asc: Option<&[u8]>) -> Result<Self, AudioError> {
        Self::build(asc, false)
    }

    /// As [`Self::new`], but an HE-AAC stream decodes as its AAC-LC core
    /// alone: half the rate, a quarter of the full rate's bandwidth, and
    /// HE-AAC v2's core mono.
    pub fn new_core_only(asc: Option<&[u8]>) -> Result<Self, AudioError> {
        Self::build(asc, true)
    }

    fn build(asc: Option<&[u8]>, core_only: bool) -> Result<Self, AudioError> {
        let mut inner = match asc.filter(|a| !a.is_empty()) {
            Some(a) => aac::decode::Decoder::new_raw(a).map_err(decode_error)?,
            None => aac::decode::Decoder::new_adts(),
        };
        inner.set_core_only(core_only);
        Ok(Self { inner, layout: None })
    }

    /// Whether the stream turned out to be HE-AAC.
    pub fn he_aac(&self) -> bool {
        self.inner.he_aac().is_some()
    }
}

impl AudioDecoder for AacDecoder {
    fn decode(&mut self, packet: &[u8], pts: i64) -> Result<Vec<AudioFrame>, AudioError> {
        let frames = self.inner.decode(packet).map_err(decode_error)?;
        let mut out = Vec::with_capacity(frames.len());
        for (i, f) in frames.into_iter().enumerate() {
            self.layout = f
                .speakers
                .as_ref()
                .and_then(|s| ChannelLayout::new(s.iter().copied().map(label).collect()).ok());
            let len = (f.samples.len() / f.channels.max(1)) as i64;
            let step = (len * 1_000_000) / i64::from(f.sample_rate.max(1));
            out.push(AudioFrame {
                samples: f.samples,
                sample_rate: f.sample_rate,
                channels: f.channels as u8,
                pts: pts + i as i64 * step,
            });
        }
        Ok(out)
    }

    fn flush(&mut self) -> Result<Vec<AudioFrame>, AudioError> {
        // AAC holds nothing back: each access unit's samples come out when
        // it is decoded. A partial ADTS frame at the end is dropped.
        self.inner.flush();
        Ok(Vec::new())
    }

    fn layout(&self) -> Option<ChannelLayout> {
        self.layout.clone()
    }
}

/// What the first access unit of an AAC track says: the rate and channels
/// it decodes to (for HE-AAC the SBR rate, or the core's with `core_only`)
/// and whether it is HE-AAC. `asc` empty means the packets are ADTS.
pub fn probe(asc: &[u8], first: &[u8], core_only: bool) -> Result<aac::decode::StreamInfo, AudioError> {
    let asc = (!asc.is_empty()).then_some(asc);
    let mut d = if core_only { AacDecoder::new_core_only(asc)? } else { AacDecoder::new(asc)? };
    let frame = d
        .inner
        .decode(first)
        .map_err(decode_error)?
        .into_iter()
        .next()
        .ok_or_else(|| AudioError::Decode("aac: no complete access unit to probe".into()))?;
    Ok(aac::decode::StreamInfo {
        sample_rate: frame.sample_rate,
        channels: frame.channels,
        speakers: frame.speakers,
        he_aac: d.inner.he_aac(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::encode::aac::{AacConfig, AacEncoder, adts_frame};
    use crate::audio::AudioEncoder;

    fn encode(rate: u32, channels: u8) -> (AacEncoder, Vec<Vec<u8>>) {
        let mut enc = AacEncoder::new(AacConfig { sample_rate: rate, channels, bitrate: 0 }).unwrap();
        let n = usize::from(channels);
        let samples: Vec<f32> = (0..rate as usize * n / 2)
            .map(|i| 0.3 * ((i / n) as f32 * 0.03 * (1 + i % n) as f32).sin())
            .collect();
        let frame = AudioFrame { samples, sample_rate: rate, channels, pts: 0 };
        let mut aus: Vec<Vec<u8>> = enc.encode(&frame).unwrap().into_iter().map(|p| p.data).collect();
        aus.extend(enc.flush().unwrap().into_iter().map(|p| p.data));
        (enc, aus)
    }

    #[test]
    fn raw_and_adts_packets_decode_with_their_layout() {
        let (enc, aus) = encode(48_000, 6);
        let mut raw = AacDecoder::new(Some(&enc.audio_specific_config())).unwrap();
        let mut adts = AacDecoder::new(None).unwrap();
        for (k, au) in aus.iter().enumerate() {
            let a = raw.decode(au, k as i64 * 21_333).unwrap();
            let framed = adts_frame(enc.sampling_index(), enc.channel_configuration(), au);
            let b = adts.decode(&framed, k as i64 * 21_333).unwrap();
            assert_eq!(a.len(), 1);
            assert_eq!(a[0].samples, b[0].samples);
            assert_eq!((a[0].sample_rate, a[0].channels, a[0].pts), (48_000, 6, k as i64 * 21_333));
        }
        assert_eq!(raw.layout(), Some(ChannelLayout::named("5.1")));
        assert!(!raw.he_aac());
        let info = probe(&enc.audio_specific_config(), &aus[0], false).unwrap();
        assert_eq!((info.sample_rate, info.channels, info.he_aac), (48_000, 6, None));
    }

    #[test]
    fn broken_packets_are_errors() {
        let (enc, _) = encode(44_100, 2);
        let mut dec = AacDecoder::new(Some(&enc.audio_specific_config())).unwrap();
        // A single channel element that ends before its global gain.
        assert!(matches!(dec.decode(&[0x00, 0x00], 0), Err(AudioError::Decode(_))));
        assert!(matches!(AacDecoder::new(Some(&[0x0a, 0x10])), Err(AudioError::Unsupported(_))));
    }

    /// An HE-AAC stream decodes at its SBR rate, or with `new_core_only` as
    /// its core at half of it; the probe says which.
    #[test]
    fn he_aac_decodes_in_full_or_as_its_core() {
        use crate::audio::encode::aac::Profile;
        let mut enc =
            AacEncoder::with_profile(AacConfig { sample_rate: 48_000, channels: 2, bitrate: 0 }, Profile::HeAac).unwrap();
        let samples: Vec<f32> = (0..48_000 * 2).map(|i| 0.3 * ((i / 2) as f32 * 0.05).sin()).collect();
        let mut aus: Vec<Vec<u8>> =
            enc.encode(&AudioFrame { samples, sample_rate: 48_000, channels: 2, pts: 0 }).unwrap().into_iter().map(|p| p.data).collect();
        aus.extend(enc.flush().unwrap().into_iter().map(|p| p.data));
        let asc = enc.audio_specific_config();
        let full = probe(&asc, &aus[0], false).unwrap();
        let core = probe(&asc, &aus[0], true).unwrap();
        assert_eq!((full.sample_rate, core.sample_rate), (48_000, 24_000));
        assert!(full.he_aac.is_some() && core.he_aac.is_some());
        let mut dec = AacDecoder::new(Some(&asc)).unwrap();
        let f = dec.decode(&aus[1], 0).unwrap();
        assert_eq!((f[0].sample_rate, f[0].channels, f[0].samples.len()), (48_000, 2, 4096));
        assert!(dec.he_aac());
    }
}
