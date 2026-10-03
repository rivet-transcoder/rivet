//! DTS Coherent Acoustics **core** output through the workspace's encoder
//! (`dts`, the `crates/dts` submodule, the rivet-dts repository), adapted to
//! [`AudioEncoder`].
//!
//! - **Layouts.** The core's arrangements, each with or without the LFE:
//!   mono, stereo, 3.0, 3.0(back), 4.0, quad(side), 5.0(side) (and 2.1, 3.1,
//!   4.1, 5.1(side)) — the same set as AC-3
//!   ([`crate::audio::remix::surround_core_layout`]). The input's speakers
//!   come from [`AudioEncoderConfig::layout`] and are reordered into the
//!   encoder's.
//! - **Rates.** 48, 44.1 and 32 kHz; other input is resampled to
//!   [`crate::audio::dolby_dts_sample_rate`] of it.
//! - **Bit rate.** One of ETSI TS 102 114 Table 5-7's ([`DTS_BITRATES`]),
//!   32 to 1536 kb/s; `0` is the full rate for the sample rate
//!   ([`default_bitrate`]: 1536 kb/s at 48 kHz, 1411.2 at 44.1, 1024 at 32).
//! - **Frames.** 512 samples per frame, a constant size for the rate, no
//!   ADPCM prediction (so every core decoder decodes them: the D.10.1 code
//!   book is not published) and no high-frequency VQ.
//! - **Delay.** The analysis and synthesis banks: 512 samples
//!   ([`AudioEncoder::pre_skip`]). The last frame is padded so the decoded
//!   output covers every input sample.
//! - [`AudioEncoder::extra_data`] is empty: the muxer derives `ddts` from the
//!   first frame's core header.

use crate::audio::filter::{ChannelLabel, ChannelLayout};
use crate::audio::resample::AlignedResampler;
use crate::audio::{
    AudioCodec, AudioEncoder, AudioEncoderConfig, AudioError, AudioFrame, EncodedAudioPacket, dolby_dts_sample_rate,
};

/// The bit rates of ETSI TS 102 114 Table 5-7, bits per second.
pub const DTS_BITRATES: [u32; 25] = [
    32_000, 56_000, 64_000, 96_000, 112_000, 128_000, 192_000, 224_000, 256_000, 320_000, 384_000, 448_000,
    512_000, 576_000, 640_000, 768_000, 960_000, 1_024_000, 1_152_000, 1_280_000, 1_344_000, 1_408_000,
    1_411_200, 1_472_000, 1_536_000,
];

/// The default bit rate at a coded `sample_rate`: the format's full rate.
pub fn default_bitrate(sample_rate: u32) -> u32 {
    match sample_rate {
        44_100 => 1_411_200,
        32_000 => 1_024_000,
        _ => 1_536_000,
    }
}

fn encode_error(e: ::dts::Error) -> AudioError {
    match e {
        ::dts::Error::Unsupported(m) => AudioError::Unsupported(format!("dts: {m}")),
        other => AudioError::Encode(other.to_string()),
    }
}

fn dts_speaker(l: ChannelLabel) -> ::dts::Speaker {
    use ::dts::Speaker;
    match l {
        ChannelLabel::FL => Speaker::FL,
        ChannelLabel::FR => Speaker::FR,
        ChannelLabel::FC => Speaker::FC,
        ChannelLabel::LFE => Speaker::LFE,
        ChannelLabel::BL => Speaker::BL,
        ChannelLabel::BR => Speaker::BR,
        ChannelLabel::BC => Speaker::BC,
        ChannelLabel::SL => Speaker::SL,
        ChannelLabel::SR => Speaker::SR,
    }
}

pub struct DtsEncoder {
    inner: ::dts::Encoder,
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

impl DtsEncoder {
    pub fn new(config: &AudioEncoderConfig) -> Result<Self, AudioError> {
        if config.codec != AudioCodec::Dts {
            return Err(AudioError::Encode(format!("DtsEncoder constructed with codec {:?}", config.codec)));
        }
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
        let speakers: Vec<::dts::Speaker> = layout.labels().iter().map(|&l| dts_speaker(l)).collect();
        let dts_layout = ::dts::Layout::from_speakers(&speakers);
        let out_rate = dolby_dts_sample_rate(config.sample_rate);
        let bitrate = if config.bitrate == 0 { default_bitrate(out_rate) } else { config.bitrate };
        if !DTS_BITRATES.contains(&bitrate) {
            return Err(AudioError::Unsupported(format!(
                "{bitrate} bps is not a DTS bit rate (ETSI TS 102 114 Table 5-7: 32k to 1536k)"
            )));
        }
        let inner = ::dts::Encoder::new(::dts::EncoderConfig::new(out_rate, dts_layout, bitrate)).map_err(encode_error)?;
        let order = dts_layout
            .speakers()
            .iter()
            .map(|&s| speakers.iter().position(|&x| x == s).expect("the layout was built from these speakers"))
            .collect();
        Ok(Self {
            inner,
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
        let step = ::dts::encoder::FRAME_SAMPLES as u64;
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

impl AudioEncoder for DtsEncoder {
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

    /// The filterbanks' delay: 512 samples at the coded rate.
    fn pre_skip(&self) -> u16 {
        self.inner.delay() as u16
    }

    /// Empty: `ddts` is derived from the first frame's core header.
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
    use crate::audio::decode::dts::DtsDecoder;

    fn config(rate: u32, layout: &str, bitrate: u32) -> AudioEncoderConfig {
        let layout = ChannelLayout::named(layout);
        AudioEncoderConfig {
            codec: AudioCodec::Dts,
            sample_rate: rate,
            channels: layout.len() as u8,
            bitrate,
            quality: None,
            layout: Some(layout),
        }
    }

    #[test]
    fn layouts_and_rates_are_checked() {
        for name in ["mono", "stereo", "2.1", "3.0", "3.0(back)", "3.1", "4.0", "quad(side)", "4.1", "5.0(side)", "5.1(side)"]
        {
            assert!(DtsEncoder::new(&config(48_000, name, 0)).is_ok(), "{name}");
        }
        assert!(DtsEncoder::new(&config(48_000, "7.1", 0)).is_err(), "not a core arrangement");
        assert!(DtsEncoder::new(&config(48_000, "stereo", 100_000)).is_err());
        let e = DtsEncoder::new(&config(88_200, "stereo", 0)).unwrap();
        assert_eq!((e.sample_rate(), e.pre_skip()), (44_100, 512));
    }

    /// 5.1(side) comes back from rivet's own decoder in the same order and
    /// layout, 512 samples late, every tone in its own channel.
    #[test]
    fn five_one_round_trips_channel_for_channel() {
        let freqs = [300.0f32, 500.0, 700.0, 60.0, 1100.0, 1300.0];
        let n = 48_000;
        let pcm: Vec<f32> = (0..n * 6)
            .map(|i| 0.3 * (2.0 * std::f32::consts::PI * freqs[i % 6] * (i / 6) as f32 / 48_000.0).sin())
            .collect();
        let mut enc = DtsEncoder::new(&config(48_000, "5.1(side)", 768_000)).unwrap();
        let mut packets = Vec::new();
        for c in pcm.chunks(6 * 1000) {
            packets.extend(enc.encode(&AudioFrame { samples: c.to_vec(), sample_rate: 48_000, channels: 6, pts: 0 }).unwrap());
        }
        packets.extend(enc.flush().unwrap());
        assert!(packets.iter().all(|p| p.duration == 512));
        assert!(packets.len() * 512 >= n + 512);
        let mut dec = DtsDecoder::new(48_000, 6).unwrap();
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
                let (a, b) = (pcm[i * 6 + c], out[(i + 512) * 6 + c]);
                s += f64::from(a).powi(2);
                e += f64::from(a - b).powi(2);
            }
            let snr = 10.0 * (s / e).log10();
            eprintln!("DTS 5.1 768k, channel {c} ({f} Hz): {snr:.1} dB");
            assert!(snr > 20.0, "channel {c}: {snr:.1} dB");
        }
    }
}
