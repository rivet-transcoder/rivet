//! AC-3 (Dolby Digital) and E-AC-3 (Dolby Digital Plus) output through the
//! workspace's encoder (`ac3`, the `crates/ac3` submodule, the rivet-ac3
//! repository), adapted to [`AudioEncoder`].
//!
//! - **Layouts.** A/52's arrangements, 1/0 to 3/2, each with or without the
//!   LFE: mono, stereo, 2.1, 3.0, 3.0(back), 3.1, 4.0, quad(side), 4.1,
//!   5.0(side), 5.1(side) ([`crate::audio::remix::surround_core_layout`]
//!   picks one for a source); and for E-AC-3 7.1, a 3/2 + LFE independent
//!   substream with a 2/0 dependent substream on the back surrounds
//!   ([`crate::audio::remix::eac3_layout`]). The input's speakers come from
//!   [`AudioEncoderConfig::layout`] (or the count's default) and are
//!   reordered into the encoder's.
//! - **Rates.** 48, 44.1 and 32 kHz; other input is resampled to
//!   [`crate::audio::dolby_dts_sample_rate`] of it through an
//!   [`AlignedResampler`].
//! - **Bit rate.** AC-3: one of Table 5.18's ([`AC3_BITRATES`]), 32–640
//!   kb/s; E-AC-3: 32–6144 kb/s in whole kb/s. `0` picks [`default_bitrate`].
//! - **Delay.** The transform's overlap: 256 samples ([`AudioEncoder::pre_skip`]).
//!   The encoder pads its last frame so every input sample is in a frame.
//! - Packets are whole syncframes, 1536 samples each (an E-AC-3 frame at a
//!   rate too high for six blocks has fewer). [`AudioEncoder::extra_data`]
//!   is empty: the muxer derives `dac3` / `dec3` from the first syncframe.
//!
//! Not written: dynamic range metadata, Annex D (`bsid` 9/10). A packet
//! is an access unit: for 7.1 the independent syncframe and the dependent
//! one, as an MP4 sample holds them.

use crate::audio::filter::{ChannelLabel, ChannelLayout};
use crate::audio::resample::AlignedResampler;
use crate::audio::{
    AudioCodec, AudioEncoder, AudioEncoderConfig, AudioError, AudioFrame, EncodedAudioPacket, dolby_dts_sample_rate,
};

/// AC-3's bit rates (A/52 Table 5.18), bits per second.
pub const AC3_BITRATES: [u32; 19] = [
    32_000, 40_000, 48_000, 56_000, 64_000, 80_000, 96_000, 112_000, 128_000, 160_000, 192_000, 224_000, 256_000,
    320_000, 384_000, 448_000, 512_000, 576_000, 640_000,
];

/// E-AC-3's bit rate span, bits per second (whole kb/s).
pub const EAC3_BITRATE_RANGE: (u32, u32) = (32_000, 6_144_000);

/// The default bit rate of `codec` (AC-3 or E-AC-3) for a layout of
/// `channels`, the LFE counted: AC-3 96 kb/s mono, 192 stereo, 384 for three
/// or four channels, 448 for five or six (the DVD rate); E-AC-3 96 mono, 192
/// stereo, 256 for three or four, 384 for five or six, 512 for 7.1 (its
/// 5.1 independent substream keeps about 384 of it).
pub fn default_bitrate(codec: AudioCodec, channels: u8) -> u32 {
    let eac3 = codec == AudioCodec::Eac3;
    match channels {
        7 | 8 if eac3 => 512_000,
        0 | 1 => 96_000,
        2 => 192_000,
        3 | 4 => {
            if eac3 {
                256_000
            } else {
                384_000
            }
        }
        _ => {
            if eac3 {
                384_000
            } else {
                448_000
            }
        }
    }
}

/// Whether `bps` is a bit rate `codec` (AC-3 or E-AC-3) codes.
pub fn valid_bitrate(codec: AudioCodec, bps: u32) -> bool {
    match codec {
        AudioCodec::Eac3 => bps.is_multiple_of(1000) && (EAC3_BITRATE_RANGE.0..=EAC3_BITRATE_RANGE.1).contains(&bps),
        _ => AC3_BITRATES.contains(&bps),
    }
}

fn encode_error(e: ::ac3::Error) -> AudioError {
    match e {
        ::ac3::Error::InvalidInput(m) | ::ac3::Error::Unsupported(m) => AudioError::Unsupported(format!("ac3: {m}")),
        other => AudioError::Encode(format!("ac3: {other}")),
    }
}

fn speaker_label(s: ::ac3::Speaker) -> ChannelLabel {
    use ::ac3::Speaker::*;
    match s {
        FL => ChannelLabel::FL,
        FR => ChannelLabel::FR,
        FC => ChannelLabel::FC,
        LFE => ChannelLabel::LFE,
        BC => ChannelLabel::BC,
        SL => ChannelLabel::SL,
        SR => ChannelLabel::SR,
        BL => ChannelLabel::BL,
        BR => ChannelLabel::BR,
    }
}

/// The A/52 arrangement (and LFE) of `layout`, when it is one; for E-AC-3
/// (`eac3`) 7.1 too, as 3/4.
pub fn ac3_layout_of(layout: &ChannelLayout, eac3: bool) -> Option<(::ac3::Layout, bool)> {
    use ::ac3::Layout::*;
    let lfe = layout.has(ChannelLabel::LFE);
    let full = layout.len() - usize::from(lfe);
    let seven = eac3.then_some(ThreeFour);
    [Mono, Stereo, ThreeZero, TwoOne, ThreeOne, TwoTwo, ThreeTwo].into_iter().chain(seven).find_map(|l| {
        let speakers = l.speakers(lfe);
        (speakers.len() == full + usize::from(lfe) && speakers.iter().all(|&s| layout.has(speaker_label(s))))
            .then_some((l, lfe))
    })
}

pub struct Ac3Encoder {
    inner: ::ac3::Encoder,
    codec: AudioCodec,
    channels: u8,
    in_rate: u32,
    out_rate: u32,
    /// For each encoder input slot, the pipeline slot it takes.
    order: Vec<usize>,
    resampler: AlignedResampler,
    buf: Vec<f32>,
    reordered: Vec<f32>,
    first_pts: Option<i64>,
    samples_out: u64,
}

impl Ac3Encoder {
    pub fn new(config: &AudioEncoderConfig) -> Result<Self, AudioError> {
        let codec = config.codec;
        let format = match codec {
            AudioCodec::Ac3 => ::ac3::Format::Ac3,
            AudioCodec::Eac3 => ::ac3::Format::Eac3,
            other => return Err(AudioError::Encode(format!("Ac3Encoder constructed with codec {other:?}"))),
        };
        if config.sample_rate == 0 {
            return Err(AudioError::Encode("input sample_rate is 0".into()));
        }
        let layout = match &config.layout {
            Some(l) => l.clone(),
            None => ChannelLayout::default_for(config.channels)
                .map_err(|e| AudioError::Unsupported(format!("{} channels: {e}", config.channels)))?,
        };
        if layout.len() != usize::from(config.channels) {
            return Err(AudioError::Encode(format!("layout {layout} for {} channels", config.channels)));
        }
        let (arrangement, lfe) = ac3_layout_of(&layout, codec == AudioCodec::Eac3).ok_or_else(|| {
            AudioError::Unsupported(format!(
                "{layout} is not an AC-3 channel arrangement (1/0 to 3/2, with or without the LFE; 7.1 for E-AC-3)"
            ))
        })?;
        let bitrate = if config.bitrate == 0 { default_bitrate(codec, config.channels) } else { config.bitrate };
        if !valid_bitrate(codec, bitrate) {
            return Err(AudioError::Unsupported(match codec {
                AudioCodec::Eac3 => format!("{bitrate} bps is not an E-AC-3 bit rate (32k..6144k, whole kb/s)"),
                _ => format!(
                    "{bitrate} bps is not an AC-3 bit rate ({})",
                    AC3_BITRATES.map(|b| format!("{}k", b / 1000)).join(", ")
                ),
            }));
        }
        let out_rate = dolby_dts_sample_rate(config.sample_rate);
        let inner = ::ac3::Encoder::new(::ac3::Config::new(format, out_rate, arrangement, lfe, bitrate / 1000))
            .map_err(encode_error)?;
        let order = inner
            .speakers()
            .iter()
            .map(|&s| layout.index_of(speaker_label(s)).expect("the arrangement was matched on these speakers"))
            .collect();
        Ok(Self {
            inner,
            codec,
            channels: config.channels,
            in_rate: config.sample_rate,
            out_rate,
            order,
            resampler: AlignedResampler::new(config.sample_rate, out_rate, config.channels)?,
            buf: Vec::new(),
            reordered: Vec::new(),
            first_pts: None,
            samples_out: 0,
        })
    }

    /// AC-3 or E-AC-3.
    pub fn codec(&self) -> AudioCodec {
        self.codec
    }

    fn feed(&mut self, flush: bool) -> Result<Vec<EncodedAudioPacket>, AudioError> {
        let ch = usize::from(self.channels);
        self.reordered.clear();
        self.reordered.reserve(self.buf.len());
        for frame in self.buf.chunks_exact(ch) {
            self.reordered.extend(self.order.iter().map(|&i| frame[i]));
        }
        self.buf.clear();
        let mut frames = self.inner.encode(&self.reordered).map_err(encode_error)?;
        if flush {
            frames.extend(self.inner.flush().map_err(encode_error)?);
        }
        let step = self.inner.frame_samples() as u64;
        let first = self.first_pts.unwrap_or(0);
        Ok(frames
            .into_iter()
            .map(|data| {
                let pts = first + (self.samples_out * 1_000_000 / u64::from(self.out_rate)) as i64;
                self.samples_out += step;
                EncodedAudioPacket { data, pts, duration: step as i64 }
            })
            .collect())
    }
}

impl AudioEncoder for Ac3Encoder {
    fn encode(&mut self, frame: &AudioFrame) -> Result<Vec<EncodedAudioPacket>, AudioError> {
        if frame.channels != self.channels {
            return Err(AudioError::Encode(format!(
                "channel count mismatch: encoder configured for {}, frame has {}",
                self.channels, frame.channels
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
        self.resampler.process(frame, &mut self.buf)?;
        self.feed(false)
    }

    fn flush(&mut self) -> Result<Vec<EncodedAudioPacket>, AudioError> {
        self.resampler.flush(&mut self.buf)?;
        self.feed(true)
    }

    /// The transform's overlap, 256 samples at the coded rate.
    fn pre_skip(&self) -> u16 {
        self.inner.delay() as u16
    }

    /// Empty: `dac3` / `dec3` are derived from the first syncframe.
    fn extra_data(&self) -> Vec<u8> {
        Vec::new()
    }

    fn sample_rate(&self) -> u32 {
        self.out_rate
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::AudioDecoder;
    use crate::audio::decode::ac3::Ac3Decoder;

    fn config(codec: AudioCodec, rate: u32, layout: &str, bitrate: u32) -> AudioEncoderConfig {
        let layout = ChannelLayout::named(layout);
        AudioEncoderConfig {
            codec,
            sample_rate: rate,
            channels: layout.len() as u8,
            bitrate,
            quality: None,
            layout: Some(layout),
        }
    }

    #[test]
    fn arrangements_and_rates_are_checked() {
        for name in ["mono", "stereo", "2.1", "3.0", "3.0(back)", "3.1", "4.0", "quad(side)", "4.1", "5.0(side)", "5.1(side)"]
        {
            assert!(ac3_layout_of(&ChannelLayout::named(name), false).is_some(), "{name}");
        }
        assert!(ac3_layout_of(&ChannelLayout::named("5.1"), true).is_none(), "back surrounds are not A/52's");
        assert!(ac3_layout_of(&ChannelLayout::named("7.1"), false).is_none(), "7.1 is E-AC-3's");
        assert_eq!(ac3_layout_of(&ChannelLayout::named("7.1"), true), Some((::ac3::Layout::ThreeFour, true)));
        assert!(Ac3Encoder::new(&config(AudioCodec::Ac3, 48_000, "stereo", 100_000)).is_err());
        assert!(Ac3Encoder::new(&config(AudioCodec::Eac3, 48_000, "stereo", 100_000)).is_ok());
        assert!(Ac3Encoder::new(&config(AudioCodec::Ac3, 48_000, "7.1", 0)).is_err());
        let e = Ac3Encoder::new(&config(AudioCodec::Ac3, 96_000, "stereo", 0)).unwrap();
        assert_eq!((e.sample_rate(), e.pre_skip()), (48_000, 256));
        assert_eq!(default_bitrate(AudioCodec::Ac3, 6), 448_000);
    }

    /// 5.1(side) in the pipeline's order comes back from rivet's decoder in
    /// the same order, every tone in its own channel, 256 samples late, the
    /// frames covering every input sample.
    #[test]
    fn five_one_round_trips_channel_for_channel() {
        for codec in [AudioCodec::Ac3, AudioCodec::Eac3] {
            let freqs = [300.0f32, 500.0, 700.0, 60.0, 1100.0, 1300.0];
            let n = 48_000;
            let pcm: Vec<f32> = (0..n * 6)
                .map(|i| 0.3 * (2.0 * std::f32::consts::PI * freqs[i % 6] * (i / 6) as f32 / 48_000.0).sin())
                .collect();
            let mut enc = Ac3Encoder::new(&config(codec, 48_000, "5.1(side)", 0)).unwrap();
            let mut packets = Vec::new();
            for c in pcm.chunks(6 * 1000) {
                packets.extend(enc.encode(&AudioFrame { samples: c.to_vec(), sample_rate: 48_000, channels: 6, pts: 0 }).unwrap());
            }
            packets.extend(enc.flush().unwrap());
            assert!(packets.iter().all(|p| p.duration == 1536));
            assert!(packets.len() * 1536 >= n + 256);
            let mut dec = Ac3Decoder::new(48_000, 6).unwrap();
            let mut out = Vec::new();
            for p in &packets {
                for f in dec.decode(&p.data, 0).unwrap() {
                    out.extend(f.samples);
                }
            }
            assert_eq!(dec.layout(), Some(ChannelLayout::named("5.1(side)")));
            for (c, f) in freqs.iter().enumerate() {
                let (mut s, mut e) = (0.0f64, 0.0f64);
                for i in 2048..n - 2048 {
                    let (a, b) = (pcm[i * 6 + c], out[(i + 256) * 6 + c]);
                    s += f64::from(a).powi(2);
                    e += f64::from(a - b).powi(2);
                }
                let snr = 10.0 * (s / e).log10();
                eprintln!("{codec:?} 5.1 default rate, channel {c} ({f} Hz): {snr:.1} dB");
                assert!(snr > 15.0, "{codec:?} channel {c}: {snr:.1} dB");
            }
        }
    }
}
