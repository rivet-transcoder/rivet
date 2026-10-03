//! AAC-LC, HE-AAC and HE-AAC v2 output through the workspace's AAC encoder
//! (`aac::encode`, the `crates/aac` submodule), adapted to [`AudioEncoder`]:
//! input at a rate the profile does not code is resampled here (AAC-LC to
//! [`coding_rate`] of it, HE-AAC to [`he_aac_rate`]), the resampler's delay
//! trimmed so the output stays aligned with the input, and each access unit
//! is timed. See `docs/decisions.md` §26.
//!
//! - **AAC-LC**: 1024 samples per access unit, [`ENCODER_DELAY`] (1024) of
//!   priming; the AudioSpecificConfig is the plain two-byte one.
//! - **HE-AAC** (mono to 7.1) and **HE-AAC v2** (stereo): an AAC-LC core at
//!   half the rate plus SBR (and parametric stereo), 2048 output samples per
//!   access unit at 32, 44.1 or 48 kHz, [`HE_AAC_DELAY`] (3586) samples of
//!   priming at the output rate. The AudioSpecificConfig signals SBR / PS
//!   explicitly and hierarchically (audio object type 5 or 29 first, then the
//!   core's), the form ISO/IEC 14496-3 1.6.5.2 describes for MP4 and Apple's
//!   players need: `mp4a.40.5` / `mp4a.40.29`.

pub use aac::encode::{
    ENCODER_DELAY, FRAME_SAMPLES, HE_AAC_DELAY, HE_AAC_RATES, Profile, SUPPORTED_RATES, adts_frame, adts_header,
    audio_specific_config, bitrate_range, coding_rate, default_bitrate, default_he_aac_bitrate,
};

use crate::audio::resample::AlignedResampler;
use crate::audio::{AudioEncoder, AudioError, AudioFrame, EncodedAudioPacket};

/// Encoder settings.
#[derive(Clone, Debug)]
pub struct AacConfig {
    /// The input's sample rate; the stream is coded at [`coding_rate`] of it
    /// (AAC-LC) or [`he_aac_rate`] (HE-AAC).
    pub sample_rate: u32,
    /// AAC-LC and HE-AAC: 1, 2, 3, 4, 5, 6 or 8, in rivet's native channel
    /// order. HE-AAC v2: 2.
    pub channels: u8,
    /// Target bit rate in bits per second for all channels together; 0
    /// picks the profile's default ([`default_bitrate`],
    /// [`default_he_aac_bitrate`]).
    pub bitrate: u32,
}

/// The output rate an HE-AAC stream from `input` Hz is coded at: 32, 44.1
/// or 48 kHz, the input's own when it is one, else 44.1 kHz for its family
/// and 48 kHz for the rest.
pub fn he_aac_rate(input: u32) -> u32 {
    match input {
        r if HE_AAC_RATES.contains(&r) => r,
        r if r.is_multiple_of(11_025) => 44_100,
        _ => 48_000,
    }
}

/// The bit rates an HE-AAC profile takes for `channels` (the whole
/// stream's): what `aac::encode::Encoder::with_profile` accepts.
pub fn he_aac_bitrate_range(profile: Profile, channels: u8) -> (u32, u32) {
    match profile {
        Profile::Lc => bitrate_range(48_000, channels),
        Profile::HeAacV2 => (16_000, 64_000),
        Profile::HeAac => {
            let main = u32::from(channels) - u32::from(channels >= 6);
            (12_000 * main, 64_000 * main)
        }
    }
}

/// A two-byte AAC-LC AudioSpecificConfig (object type, sampling index,
/// channel configuration, GASpecificConfig's three flags) followed by the
/// sync extension of ISO/IEC 14496-3 1.6.2.1 that states no SBR:
/// `syncExtensionType` 0x2B7, `extensionAudioObjectType` 5,
/// `sbrPresentFlag` 0, then zero bits to the byte.
fn lc_without_sbr(asc: [u8; 2]) -> Vec<u8> {
    let bits: u64 = (u64::from(u16::from_be_bytes(asc)) << 17) | (0x2B7 << 6) | (5 << 1);
    // 33 bits, left-aligned in five bytes.
    (bits << 7).to_be_bytes()[3..].to_vec()
}

pub struct AacEncoder {
    inner: aac::encode::Encoder,
    /// The input's sample rate, and the resampler to the coded rate.
    in_rate: u32,
    resampler: AlignedResampler,
    resampled: Vec<f32>,
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
    /// An AAC-LC encoder.
    pub fn new(config: AacConfig) -> Result<Self, AudioError> {
        Self::with_profile(config, Profile::Lc)
    }

    /// An encoder of `profile`.
    pub fn with_profile(config: AacConfig, profile: Profile) -> Result<Self, AudioError> {
        if config.sample_rate == 0 {
            return Err(AudioError::Encode("input sample_rate is 0".to_string()));
        }
        let rate = match profile {
            Profile::Lc => coding_rate(config.sample_rate),
            _ => he_aac_rate(config.sample_rate),
        };
        let inner = aac::encode::Encoder::with_profile(
            aac::encode::EncoderConfig { sample_rate: rate, channels: config.channels, bitrate: config.bitrate },
            profile,
        )
        .map_err(encode_error)?;
        Ok(Self {
            resampler: AlignedResampler::new(config.sample_rate, rate, config.channels)?,
            inner,
            in_rate: config.sample_rate,
            resampled: Vec::new(),
            frames_out: 0,
            first_pts: None,
        })
    }

    /// The AudioSpecificConfig (ISO/IEC 14496-3 1.6.2.1) for the MP4 `esds`:
    /// AAC-LC's plain one, HE-AAC's with SBR / PS signalled hierarchically.
    /// At 24 kHz or less, where a plain AAC-LC configuration leaves a
    /// decoder to guess whether SBR follows in the access units (and some
    /// then play it at twice the rate), the AAC-LC one ends with the
    /// backward-compatible sync extension saying it does not
    /// (`sbrPresentFlag = 0`).
    pub fn audio_specific_config(&self) -> Vec<u8> {
        match self.inner.profile() {
            Profile::Lc if self.inner.coding_rate() <= 24_000 => lc_without_sbr(self.inner.audio_specific_config()),
            Profile::Lc => self.inner.audio_specific_config().to_vec(),
            _ => self.inner.audio_specific_config_with(aac::encode::Signalling::Hierarchical),
        }
    }

    /// The profile coded.
    pub fn profile(&self) -> Profile {
        self.inner.profile()
    }

    /// The rate the AAC-LC (core) stream is coded at: half the output rate
    /// for HE-AAC.
    pub fn coding_rate(&self) -> u32 {
        self.inner.coding_rate()
    }

    /// sampling_frequency_index, for an ADTS header (the core's).
    pub fn sampling_index(&self) -> u8 {
        self.inner.sampling_index()
    }

    /// The channel configuration signalled in the ASC / ADTS header.
    pub fn channel_configuration(&self) -> u8 {
        self.inner.channel_configuration()
    }

    fn packets(&mut self, aus: Vec<Vec<u8>>) -> Vec<EncodedAudioPacket> {
        let first = self.first_pts.unwrap_or(0);
        let rate = u64::from(self.inner.sample_rate());
        let step = self.inner.frame_samples() as u64;
        aus.into_iter()
            .map(|data| {
                let pts = first + (self.frames_out * step * 1_000_000 / rate) as i64;
                self.frames_out += 1;
                EncodedAudioPacket { data, pts, duration: step as i64 }
            })
            .collect()
    }
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
        let mut buf = std::mem::take(&mut self.resampled);
        buf.clear();
        self.resampler.process(frame, &mut buf)?;
        let aus = self.inner.encode(&buf);
        self.resampled = buf;
        Ok(self.packets(aus))
    }

    fn flush(&mut self) -> Result<Vec<EncodedAudioPacket>, AudioError> {
        let mut buf = Vec::new();
        self.resampler.flush(&mut buf)?;
        let mut aus = self.inner.encode(&buf);
        // Enough frames that the decoder's output covers the priming plus
        // every input sample (counted at the coded rate).
        aus.extend(self.inner.finish(self.resampler.target_len()));
        Ok(self.packets(aus))
    }

    /// Priming samples at the stream's output rate (not 48 kHz ticks, as for
    /// Opus): the muxer's edit list skips them.
    fn pre_skip(&self) -> u16 {
        self.inner.delay() as u16
    }

    /// The AudioSpecificConfig.
    fn extra_data(&self) -> Vec<u8> {
        self.audio_specific_config()
    }

    /// The output rate: the AAC-LC coding rate, or for HE-AAC twice the core's.
    fn sample_rate(&self) -> u32 {
        self.inner.sample_rate()
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
        assert!(matches!(AacEncoder::with_profile(cfg(48_000, 6, 0), Profile::HeAacV2), Err(AudioError::Unsupported(_))));
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
                    // Clear of the last frames, where the input's abrupt end is
                    // a transient the encoder spreads back a frame or two.
                    let end = coded_len - 3072;
                    (f64::from(q) * 0.25, snr_db(&shifted[2048..end], &out[2048..end]))
                })
                .fold((0.0, f64::NEG_INFINITY), |a, b| if b.1 > a.1 { b } else { a });
            eprintln!("{input_rate} Hz input coded at {rate} Hz: SNR {snr:.1} dB, {lag:+} samples off the tone");
            assert!(snr > 50.0 && lag.abs() <= 0.5, "{input_rate}: {snr} dB at {lag}");
        }
    }

    /// HE-AAC and HE-AAC v2: 2048 output samples per access unit at the
    /// output rate, a hierarchical AudioSpecificConfig (object type 5 / 29
    /// first), and the decoder's full-rate output carries the input back
    /// `HE_AAC_DELAY` samples late, the length covering every input sample.
    #[test]
    fn he_aac_profiles_round_trip_at_the_full_rate() {
        for (profile, channels, aot) in [(Profile::HeAac, 2u8, 5u8), (Profile::HeAacV2, 2, 29), (Profile::HeAac, 1, 5)] {
            let n = 44_100;
            let tone = sine(1000.0, 0.4, 44_100, n);
            let pcm: Vec<f32> = tone.iter().flat_map(|&v| std::iter::repeat_n(v, usize::from(channels))).collect();
            let mut enc =
                AacEncoder::with_profile(AacConfig { sample_rate: 44_100, channels, bitrate: 0 }, profile).unwrap();
            assert_eq!((enc.sample_rate(), enc.pre_skip()), (44_100, HE_AAC_DELAY as u16));
            let asc = enc.extra_data();
            assert_eq!(asc[0] >> 3, aot, "{profile:?}: explicit hierarchical signalling");
            let mut aus = enc.encode(&AudioFrame { samples: pcm, sample_rate: 44_100, channels, pts: 0 }).unwrap();
            aus.extend(enc.flush().unwrap());
            assert!(aus.iter().all(|p| p.duration == 2048));
            assert!(aus.len() * 2048 >= n + HE_AAC_DELAY as usize);
            let mut dec = aac::decode::Decoder::new_raw(&asc).unwrap();
            let mut out = Vec::new();
            for p in &aus {
                for f in dec.decode(&p.data).unwrap() {
                    assert_eq!((f.sample_rate, f.channels), (44_100, usize::from(channels)), "{profile:?}");
                    out.extend(f.samples.into_iter().step_by(usize::from(channels)));
                }
            }
            let d = HE_AAC_DELAY as usize;
            let snr = snr_db(&tone[4096..n - 4096], &out[d + 4096..d + n - 4096]);
            eprintln!("{profile:?} {channels} ch at 44.1 kHz: {snr:.1} dB");
            assert!(snr > 10.0, "{profile:?}: {snr:.1} dB");
        }
    }
}
