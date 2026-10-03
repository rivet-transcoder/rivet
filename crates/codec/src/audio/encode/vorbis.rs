//! Vorbis output through the workspace's Vorbis encoder (`vorbis`, the
//! `crates/vorbis` submodule, the rivet-vorbis repository), adapted to
//! [`AudioEncoder`].
//!
//! - **Quality, not bit rate.** Vorbis is variable-rate by design; the
//!   encoder takes one quality value, −1 (smallest) to 10 (best), 5 by
//!   default ([`DEFAULT_QUALITY`]). Stereo 44.1 kHz music comes out at about
//!   66 kb/s at −1, 141 at 4, 173 at 6, 234 at 10.
//! - **Rates and layouts.** 8 to 192 kHz as the input has it (no
//!   resampling), mono to 7.1. Vorbis orders a surround layout as RFC 7845
//!   §5.1.1.2 does, so the pipeline's native order is permuted on the way in.
//! - **Timing.** Packets are timed by their granule positions: the first
//!   packet only primes the overlap and lasts zero samples, each one after
//!   lasts from the previous granule to its own, and the last one's granule
//!   is the input's length, so the durations add up to exactly the input
//!   (an Ogg file's granule positions, which trim the end). No priming: the
//!   decoded stream starts with the first input sample.
//! - [`AudioEncoder::extra_data`] is the three header packets in Xiph lacing,
//!   Matroska's `CodecPrivate`.

use crate::audio::{AudioEncoder, AudioEncoderConfig, AudioError, AudioFrame, EncodedAudioPacket};

/// The quality used when none is given.
pub const DEFAULT_QUALITY: f32 = 5.0;
/// The rates the encoder takes, in Hz.
pub const RATES: std::ops::RangeInclusive<u32> = 8_000..=192_000;

fn encode_error(e: ::vorbis::Error) -> AudioError {
    match e {
        ::vorbis::Error::Config(m) => AudioError::Unsupported(format!("vorbis: {m}")),
        other => AudioError::Encode(format!("vorbis: {other}")),
    }
}

pub struct VorbisEncoder {
    inner: ::vorbis::Encoder,
    channels: u8,
    sample_rate: u32,
    /// For each Vorbis slot, the native slot it takes (3–8 channels).
    order: Option<&'static [usize]>,
    /// The granule position the next packet starts at.
    granule: i64,
    first_pts: Option<i64>,
    buf: Vec<f32>,
}

impl VorbisEncoder {
    pub fn new(config: &AudioEncoderConfig) -> Result<Self, AudioError> {
        if !(1..=8).contains(&config.channels) {
            return Err(AudioError::Unsupported(format!(
                "Vorbis output carries 1 to 8 channels (mono to 7.1); got {}",
                config.channels
            )));
        }
        if !RATES.contains(&config.sample_rate) {
            return Err(AudioError::Unsupported(format!(
                "Vorbis encodes 8 to 192 kHz; the input is {} Hz",
                config.sample_rate
            )));
        }
        let quality = config.quality.unwrap_or(DEFAULT_QUALITY);
        let inner = ::vorbis::Encoder::new(::vorbis::EncoderConfig {
            sample_rate: config.sample_rate,
            channels: config.channels,
            quality,
            comments: Vec::new(),
        })
        .map_err(encode_error)?;
        Ok(Self {
            inner,
            channels: config.channels,
            sample_rate: config.sample_rate,
            order: crate::audio::rfc7845_family1_order(config.channels),
            granule: 0,
            first_pts: None,
            buf: Vec::new(),
        })
    }

    fn packets(&mut self, packets: Vec<::vorbis::EncodedPacket>) -> Vec<EncodedAudioPacket> {
        let first = self.first_pts.unwrap_or(0);
        packets
            .into_iter()
            .map(|p| {
                let start = self.granule;
                let end = p.granule.max(start);
                self.granule = end;
                EncodedAudioPacket {
                    data: p.data,
                    pts: first + start * 1_000_000 / i64::from(self.sample_rate),
                    duration: end - start,
                }
            })
            .collect()
    }
}

impl AudioEncoder for VorbisEncoder {
    fn encode(&mut self, frame: &AudioFrame) -> Result<Vec<EncodedAudioPacket>, AudioError> {
        if frame.channels != self.channels {
            return Err(AudioError::Encode(format!(
                "channel count mismatch: encoder configured for {}, frame has {}",
                self.channels, frame.channels
            )));
        }
        if frame.sample_rate != self.sample_rate {
            return Err(AudioError::Encode(format!(
                "sample rate mismatch: encoder configured for {}, frame has {}",
                self.sample_rate, frame.sample_rate
            )));
        }
        if self.first_pts.is_none() {
            self.first_pts = Some(frame.pts);
        }
        let packets = match self.order {
            None => self.inner.encode(&frame.samples),
            Some(order) => {
                let ch = usize::from(self.channels);
                self.buf.clear();
                self.buf.resize(frame.samples.len(), 0.0);
                for (dst, src) in self.buf.chunks_exact_mut(ch).zip(frame.samples.chunks_exact(ch)) {
                    for (slot, &native) in order.iter().enumerate() {
                        dst[slot] = src[native];
                    }
                }
                self.inner.encode(&self.buf)
            }
        }
        .map_err(encode_error)?;
        Ok(self.packets(packets))
    }

    fn flush(&mut self) -> Result<Vec<EncodedAudioPacket>, AudioError> {
        let packets = self.inner.finish().map_err(encode_error)?;
        Ok(self.packets(packets))
    }

    /// None: the decoded stream starts with the first input sample.
    fn pre_skip(&self) -> u16 {
        0
    }

    /// The identification, comment and setup headers in Xiph lacing.
    fn extra_data(&self) -> Vec<u8> {
        self.inner.codec_private()
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::decode::vorbis::VorbisDecoder;
    use crate::audio::{AudioCodec, AudioDecoder};

    fn config(sample_rate: u32, channels: u8, quality: Option<f32>) -> AudioEncoderConfig {
        AudioEncoderConfig { codec: AudioCodec::Vorbis, sample_rate, channels, bitrate: 0, quality, layout: None }
    }

    #[test]
    fn bad_configurations_are_refused() {
        assert!(VorbisEncoder::new(&config(44_100, 9, None)).is_err());
        assert!(VorbisEncoder::new(&config(4_000, 2, None)).is_err());
        assert!(VorbisEncoder::new(&config(44_100, 2, Some(11.0))).is_err());
        assert!(VorbisEncoder::new(&config(44_100, 2, Some(-1.0))).is_ok());
    }

    /// The durations add up to the input exactly, and the decoder (rivet's
    /// own) gives the input back from the first sample, channel for channel
    /// in the native order.
    #[test]
    fn five_one_round_trips_in_time_and_in_order() {
        let freqs = [300.0f32, 500.0, 700.0, 60.0, 1100.0, 1300.0]; // native 5.1: FL FR FC LFE BL BR
        let n = 48_000;
        let pcm: Vec<f32> = (0..n * 6)
            .map(|i| 0.3 * (2.0 * std::f32::consts::PI * freqs[i % 6] * (i / 6) as f32 / 48_000.0).sin())
            .collect();
        let mut enc = VorbisEncoder::new(&config(48_000, 6, None)).unwrap();
        assert_eq!((enc.pre_skip(), enc.sample_rate()), (0, 48_000));
        let mut packets = Vec::new();
        for c in pcm.chunks(6 * 1500) {
            packets.extend(enc.encode(&AudioFrame { samples: c.to_vec(), sample_rate: 48_000, channels: 6, pts: 0 }).unwrap());
        }
        packets.extend(enc.flush().unwrap());
        assert_eq!(packets.iter().map(|p| p.duration).sum::<i64>(), n as i64);
        assert_eq!(packets[0].duration, 0, "the first packet only primes the overlap");
        let mut dec = VorbisDecoder::new(Some(&enc.extra_data()), 48_000, 6).unwrap();
        let mut out = Vec::new();
        for p in &packets {
            for f in dec.decode(&p.data, 0).unwrap() {
                out.extend(f.samples);
            }
        }
        assert!(out.len() >= n * 6, "{} samples", out.len() / 6);
        for (c, f) in freqs.iter().enumerate() {
            let (mut s, mut e) = (0.0f64, 0.0f64);
            for i in 4096..n - 4096 {
                let (a, b) = (pcm[i * 6 + c], out[i * 6 + c]);
                s += f64::from(a).powi(2);
                e += f64::from(a - b).powi(2);
            }
            let snr = 10.0 * (s / e).log10();
            eprintln!("vorbis 5.1 q5 channel {c} ({f} Hz): {snr:.1} dB");
            assert!(snr > 8.0, "channel {c}: {snr:.1} dB");
        }
    }
}
