//! AAC-LC output through the workspace's AAC encoder (`aac::encode`, the
//! `crates/aac` submodule), adapted to [`AudioEncoder`]: input at a rate the
//! encoder does not code is resampled here to [`coding_rate`] of it, the
//! resampler's delay trimmed so the output stays aligned with the input, and
//! each access unit is timed. See `docs/decisions.md` §26.

pub use aac::encode::{
    ENCODER_DELAY, FRAME_SAMPLES, SUPPORTED_RATES, adts_frame, adts_header, audio_specific_config,
    bitrate_range, coding_rate, default_bitrate,
};

use crate::audio::resample::AudioResampler;
use crate::audio::{AudioEncoder, AudioError, AudioFrame, EncodedAudioPacket};

/// Encoder settings.
#[derive(Clone, Debug)]
pub struct AacConfig {
    /// The input's sample rate; the stream is coded at [`coding_rate`] of it.
    pub sample_rate: u32,
    /// 1, 2, 3, 4, 5, 6 or 8, in rivet's native channel order.
    pub channels: u8,
    /// Target bit rate in bits per second for all channels together; 0
    /// picks [`default_bitrate`].
    pub bitrate: u32,
}

pub struct AacEncoder {
    inner: aac::encode::Encoder,
    /// The input's sample rate, and the resampler to the coding rate when
    /// they differ, with the output frames of its delay still to drop.
    in_rate: u32,
    resampler: Option<AudioResampler>,
    resample_skip: usize,
    resampled: Vec<f32>,
    /// Input sample frames received, per channel, at the input rate.
    samples_in: u64,
    frames_out: u64,
    first_pts: Option<i64>,
}

fn encode_error(e: aac::Error) -> AudioError {
    match e {
        aac::Error::Config(m) | aac::Error::Unsupported(m) => AudioError::Unsupported(m),
        aac::Error::Invalid(m) => AudioError::Encode(m),
    }
}

impl AacEncoder {
    pub fn new(config: AacConfig) -> Result<Self, AudioError> {
        if config.sample_rate == 0 {
            return Err(AudioError::Encode("input sample_rate is 0".to_string()));
        }
        let rate = coding_rate(config.sample_rate);
        let inner = aac::encode::Encoder::new(aac::encode::EncoderConfig {
            sample_rate: rate,
            channels: config.channels,
            bitrate: config.bitrate,
        })
        .map_err(encode_error)?;
        let (resampler, resample_skip) = if rate == config.sample_rate {
            (None, 0)
        } else {
            (
                Some(AudioResampler::new(config.sample_rate, rate, config.channels, 1024)?),
                resampler_delay(config.sample_rate, rate)?,
            )
        };
        Ok(Self {
            inner,
            in_rate: config.sample_rate,
            resampler,
            resample_skip,
            resampled: Vec::new(),
            samples_in: 0,
            frames_out: 0,
            first_pts: None,
        })
    }

    /// The AudioSpecificConfig (ISO/IEC 14496-3 1.6.2.1) for the MP4 `esds`.
    pub fn audio_specific_config(&self) -> [u8; 2] {
        self.inner.audio_specific_config()
    }

    /// The rate the stream is coded at (the timescale of its packets).
    pub fn coding_rate(&self) -> u32 {
        self.inner.coding_rate()
    }

    /// sampling_frequency_index, for an ADTS header.
    pub fn sampling_index(&self) -> u8 {
        self.inner.sampling_index()
    }

    /// The channel configuration signalled in the ASC / ADTS header.
    pub fn channel_configuration(&self) -> u8 {
        self.inner.channel_configuration()
    }

    fn packets(&mut self, aus: Vec<Vec<u8>>) -> Vec<EncodedAudioPacket> {
        let first = self.first_pts.unwrap_or(0);
        let rate = u64::from(self.inner.coding_rate());
        aus.into_iter()
            .map(|data| {
                let pts = first + (self.frames_out * FRAME_SAMPLES as u64 * 1_000_000 / rate) as i64;
                self.frames_out += 1;
                EncodedAudioPacket { data, pts, duration: FRAME_SAMPLES as i64 }
            })
            .collect()
    }

    /// Feed the resampler's output, less its delay.
    fn push_resampled(&mut self, samples: &[f32]) -> Vec<Vec<u8>> {
        let n = usize::from(self.inner.channels());
        let skip = self.resample_skip.min(samples.len() / n);
        self.resample_skip -= skip;
        self.inner.encode(&samples[skip * n..])
    }
}

/// The delay, in output samples, of the resampler from `in_rate` to
/// `out_rate`: where an impulse at the first input sample comes out. Measured
/// rather than computed, so it holds whatever the filter's design.
fn resampler_delay(in_rate: u32, out_rate: u32) -> Result<usize, AudioError> {
    let mut r = AudioResampler::new(in_rate, out_rate, 1, 1024)?;
    let mut impulse = vec![0.0f32; 1024];
    impulse[0] = 1.0;
    let mut out = Vec::new();
    for samples in [impulse, vec![0.0; 1024]] {
        let frame = AudioFrame { samples, sample_rate: in_rate, channels: 1, pts: 0 };
        r.process(&frame, &mut out)?;
    }
    Ok(out
        .iter()
        .enumerate()
        .fold((0, 0.0f32), |best, (i, &v)| if v.abs() > best.1 { (i, v.abs()) } else { best })
        .0)
}

impl AudioEncoder for AacEncoder {
    fn encode(&mut self, frame: &AudioFrame) -> Result<Vec<EncodedAudioPacket>, AudioError> {
        if frame.channels != self.inner.channels() {
            return Err(AudioError::Encode(format!(
                "channel count mismatch: encoder configured for {}, frame has {}",
                self.inner.channels(),
                frame.channels
            )));
        }
        if frame.sample_rate != self.in_rate {
            return Err(AudioError::Encode(format!(
                "sample rate mismatch: encoder configured for {}, frame has {}",
                self.in_rate, frame.sample_rate
            )));
        }
        if self.first_pts.is_none() {
            self.first_pts = Some(frame.pts);
        }
        self.samples_in += (frame.samples.len() / usize::from(self.inner.channels())) as u64;
        let aus = match self.resampler.as_mut() {
            None => self.inner.encode(&frame.samples),
            Some(r) => {
                let mut out = std::mem::take(&mut self.resampled);
                out.clear();
                r.process(frame, &mut out)?;
                let aus = self.push_resampled(&out);
                self.resampled = out;
                aus
            }
        };
        Ok(self.packets(aus))
    }

    fn flush(&mut self) -> Result<Vec<EncodedAudioPacket>, AudioError> {
        let mut aus = Vec::new();
        if let Some(mut r) = self.resampler.take() {
            // Silence behind the input pushes the filter's delayed tail out.
            let tail = AudioFrame {
                samples: vec![0.0; r.chunk_size() * usize::from(self.inner.channels())],
                sample_rate: self.in_rate,
                channels: self.inner.channels(),
                pts: 0,
            };
            let mut out = Vec::new();
            r.process(&tail, &mut out)?;
            r.flush(&mut out)?;
            aus = self.push_resampled(&out);
        }
        // Enough frames that the decoder's output covers the priming plus
        // every input sample (counted at the coding rate).
        let coded = (u128::from(self.samples_in) * u128::from(self.inner.coding_rate()))
            .div_ceil(u128::from(self.in_rate)) as u64;
        aus.extend(self.inner.finish(coded));
        Ok(self.packets(aus))
    }

    /// Priming samples at the stream's own rate (not 48 kHz ticks, as for
    /// Opus): the muxer's edit list skips them.
    fn pre_skip(&self) -> u16 {
        ENCODER_DELAY as u16
    }

    /// The AudioSpecificConfig.
    fn extra_data(&self) -> Vec<u8> {
        self.inner.audio_specific_config().to_vec()
    }

    fn sample_rate(&self) -> u32 {
        self.inner.coding_rate()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn sine(freq: f64, amp: f64, rate: u32, len: usize) -> Vec<f32> {
        (0..len).map(|i| (amp * (2.0 * PI * freq * i as f64 / f64::from(rate)).sin()) as f32).collect()
    }

    fn snr_db(reference: &[f32], decoded: &[f32]) -> f64 {
        let (mut s, mut n) = (0.0f64, 0.0f64);
        for (&a, &b) in reference.iter().zip(decoded) {
            s += f64::from(a) * f64::from(a);
            n += f64::from(a - b) * f64::from(a - b);
        }
        10.0 * (s / n.max(1e-30)).log10()
    }

    #[test]
    fn config_errors_keep_their_kinds() {
        let cfg = |sample_rate, channels, bitrate| AacConfig { sample_rate, channels, bitrate };
        assert!(matches!(AacEncoder::new(cfg(0, 2, 0)), Err(AudioError::Encode(_))));
        assert!(matches!(AacEncoder::new(cfg(48_000, 7, 0)), Err(AudioError::Unsupported(_))));
        assert!(matches!(AacEncoder::new(cfg(48_000, 2, 4_000)), Err(AudioError::Unsupported(_))));
        let e = AacEncoder::new(cfg(96_000, 2, 0)).unwrap();
        assert_eq!(e.coding_rate(), 48_000);
        assert_eq!(e.extra_data(), vec![0x11, 0x90]);
    }

    /// Input at a rate the encoder does not code goes through the
    /// resampler; the decoded output lines up with the same tone generated
    /// at the coding rate to within the resampler's fractional delay.
    #[test]
    fn other_input_rates_are_resampled_and_stay_in_time() {
        assert_eq!(coding_rate(48_000), 48_000);
        assert_eq!(coding_rate(96_000), 48_000);
        assert_eq!(coding_rate(88_200), 44_100);
        assert_eq!(coding_rate(16_000), 24_000);
        assert_eq!(coding_rate(11_025), 22_050);
        assert_eq!(coding_rate(8_000), 24_000);
        for input_rate in [8_000u32, 16_000, 11_025, 96_000, 88_200] {
            let len = input_rate as usize * 3 / 2;
            let x = sine(440.0, 0.5, input_rate, len);
            let mut enc = AacEncoder::new(AacConfig { sample_rate: input_rate, channels: 1, bitrate: 64_000 }).unwrap();
            let rate = enc.coding_rate();
            let mut aus = Vec::new();
            for (i, chunk) in x.chunks(999).enumerate() {
                let frame = AudioFrame { samples: chunk.to_vec(), sample_rate: input_rate, channels: 1, pts: i as i64 };
                aus.extend(enc.encode(&frame).unwrap());
            }
            aus.extend(enc.flush().unwrap());
            let coded_len = (len as u64 * u64::from(rate)).div_ceil(u64::from(input_rate)) as usize;
            assert_eq!(aus.len(), (coded_len + 1024).div_ceil(1024), "{input_rate}");
            // Packets are timed in steps of one access unit from the first PTS.
            assert_eq!(aus[0].pts, 0);
            assert_eq!(aus[1].pts, (1024 * 1_000_000 / u64::from(rate)) as i64);
            let mut dec = aac::decode::Decoder::new_raw(&enc.audio_specific_config()).unwrap();
            let mut out: Vec<f32> = aus.iter().flat_map(|p| dec.decode(&p.data).unwrap().remove(0).samples).collect();
            out.drain(..ENCODER_DELAY as usize);
            let (lag, snr) = (-4..=4)
                .map(|q| {
                    let shifted: Vec<f32> = (0..coded_len)
                        .map(|i| {
                            let t = (i as f64 + f64::from(q) * 0.25) / f64::from(rate);
                            (0.5 * (2.0 * PI * 440.0 * t).sin()) as f32
                        })
                        .collect();
                    let end = coded_len - 1024;
                    (f64::from(q) * 0.25, snr_db(&shifted[2048..end], &out[2048..end]))
                })
                .fold((0.0, f64::NEG_INFINITY), |a, b| if b.1 > a.1 { b } else { a });
            eprintln!("{input_rate} Hz input coded at {rate} Hz: SNR {snr:.1} dB, {lag:+} samples off the tone");
            assert!(snr > 50.0 && lag.abs() <= 0.5, "{input_rate}: {snr} dB at {lag}");
        }
    }
}
