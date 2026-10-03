//! MP3 (MPEG-1 Audio Layer III) output through the workspace's encoder
//! (`mp3`, the `crates/mp3` submodule, the rivet-mp3 repository), adapted to
//! [`AudioEncoder`]: constant bit rate, mono or stereo, at 32 / 44.1 / 48 kHz.
//!
//! - **Rate.** MPEG-1 Layer III codes 32, 44.1 and 48 kHz, the rates every
//!   player and every MP4 reader takes. Those pass through; anything else is
//!   resampled here ([`mp3_sample_rate`]): the 11.025 kHz family (22.05,
//!   88.2, 176.4 kHz) to 44.1 kHz, everything else to 48 kHz, through an
//!   [`AlignedResampler`] whose delay is trimmed.
//! - **Bit rate.** CBR on the MPEG-1 ladder ([`MP3_BITRATES`]); `0` is 128k
//!   stereo, 64k mono.
//! - **Channels.** One or two; joint stereo (mid/side chosen frame by frame)
//!   for two. A wider layout is the caller's to downmix first
//!   ([`crate::audio::remix::mp3_layout`]).
//! - **Delay.** The decoded stream starts [`AudioEncoder::pre_skip`] samples
//!   late: the encoder's delay ([`mp3::encode::ENCODER_DELAY`], 528) and the
//!   decoder's ([`mp3::xing::DECODER_DELAY`], 529). A container hides them
//!   with an edit list; a bare `.mp3` with the tag frame's delay field.
//! - **The tag frame.** [`AudioEncoder::file_header`] is the encoder's own
//!   `Info` frame with its LAME-style extension (encoder `rivetmp3`, delay,
//!   padding, music CRC), complete once the encoder is flushed. It goes at
//!   the head of an `.mp3` file only: in an MP4 it would be a sample that
//!   decodes to a frame of silence.

use crate::audio::resample::AlignedResampler;
use crate::audio::{
    AudioCodec, AudioEncoder, AudioEncoderConfig, AudioError, AudioFrame, EncodedAudioPacket, MP3_BITRATES,
    MP3_FRAME_SAMPLES, mp3_default_bitrate, mp3_sample_rate,
};

/// The name the tag frame's extension carries.
pub const ENCODER_NAME: &str = "rivetmp3";

fn encode_error(e: ::mp3::Error) -> AudioError {
    match e {
        ::mp3::Error::Config(m) | ::mp3::Error::Unsupported(m) => AudioError::Unsupported(format!("mp3: {m}")),
        other => AudioError::Encode(format!("mp3: {other}")),
    }
}

pub struct Mp3Encoder {
    inner: ::mp3::Encoder,
    in_rate: u32,
    out_rate: u32,
    channels: u8,
    resampler: AlignedResampler,
    /// Resampled input waiting for the encoder (reused).
    buf: Vec<f32>,
    first_pts: Option<i64>,
    frames_out: u64,
    flushed: bool,
}

impl Mp3Encoder {
    pub fn new(config: AudioEncoderConfig) -> Result<Self, AudioError> {
        if config.codec != AudioCodec::Mp3 {
            return Err(AudioError::Encode(format!("Mp3Encoder constructed with codec {:?}", config.codec)));
        }
        if !(1..=2).contains(&config.channels) {
            return Err(AudioError::Unsupported(format!(
                "MP3 carries one or two channels; got {} (downmix first)",
                config.channels
            )));
        }
        if config.sample_rate == 0 {
            return Err(AudioError::Encode("input sample_rate is 0".into()));
        }
        let bitrate = if config.bitrate == 0 { mp3_default_bitrate(config.channels) } else { config.bitrate };
        if !MP3_BITRATES.contains(&bitrate) {
            return Err(AudioError::Unsupported(format!(
                "{bitrate} bps is not an MPEG-1 Layer III bitrate (32k..320k: {})",
                MP3_BITRATES.map(|b| format!("{}k", b / 1000)).join(", ")
            )));
        }
        let out_rate = mp3_sample_rate(config.sample_rate);
        let inner = ::mp3::Encoder::new(::mp3::EncoderConfig {
            sample_rate: out_rate,
            channels: config.channels,
            bitrate: ::mp3::BitrateMode::Cbr(bitrate),
            ..::mp3::EncoderConfig::default()
        })
        .map_err(encode_error)?;
        debug_assert_eq!(inner.frame_samples(), MP3_FRAME_SAMPLES as usize);
        Ok(Self {
            inner,
            in_rate: config.sample_rate,
            out_rate,
            channels: config.channels,
            resampler: AlignedResampler::new(config.sample_rate, out_rate, config.channels)?,
            buf: Vec::new(),
            first_pts: None,
            frames_out: 0,
            flushed: false,
        })
    }

    fn packets(&mut self, frames: Vec<Vec<u8>>) -> Vec<EncodedAudioPacket> {
        let first = self.first_pts.unwrap_or(0);
        frames
            .into_iter()
            .map(|data| {
                let pts = first
                    + (self.frames_out * u64::from(MP3_FRAME_SAMPLES) * 1_000_000 / u64::from(self.out_rate)) as i64;
                self.frames_out += 1;
                EncodedAudioPacket { data, pts, duration: i64::from(MP3_FRAME_SAMPLES) }
            })
            .collect()
    }
}

impl AudioEncoder for Mp3Encoder {
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
        let mut buf = std::mem::take(&mut self.buf);
        buf.clear();
        self.resampler.process(frame, &mut buf)?;
        let frames = self.inner.encode(&buf);
        self.buf = buf;
        Ok(self.packets(frames))
    }

    fn flush(&mut self) -> Result<Vec<EncodedAudioPacket>, AudioError> {
        let mut buf = Vec::new();
        self.resampler.flush(&mut buf)?;
        let mut frames = self.inner.encode(&buf);
        frames.extend(self.inner.flush());
        self.flushed = true;
        Ok(self.packets(frames))
    }

    /// The encoder's delay plus the decoder's, in samples at
    /// [`Self::sample_rate`].
    fn pre_skip(&self) -> u16 {
        (self.inner.delay() + ::mp3::xing::DECODER_DELAY) as u16
    }

    /// MP3 has no decoder configuration: its `esds` carries none.
    fn extra_data(&self) -> Vec<u8> {
        Vec::new()
    }

    fn sample_rate(&self) -> u32 {
        self.out_rate
    }

    /// The `Info` tag frame, once flushed.
    fn file_header(&self) -> Option<Vec<u8>> {
        self.flushed.then(|| self.inner.tag_frame())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(sample_rate: u32, channels: u8, bitrate: u32) -> AudioEncoderConfig {
        AudioEncoderConfig { codec: AudioCodec::Mp3, sample_rate, channels, bitrate, quality: None, layout: None }
    }

    #[test]
    fn rates_outside_mpeg1_resample_to_the_nearest_family() {
        for (input, out) in [
            (48_000, 48_000),
            (44_100, 44_100),
            (32_000, 32_000),
            (22_050, 44_100),
            (11_025, 44_100),
            (88_200, 44_100),
            (96_000, 48_000),
            (16_000, 48_000),
            (8_000, 48_000),
        ] {
            assert_eq!(mp3_sample_rate(input), out, "{input}");
            assert_eq!(Mp3Encoder::new(config(input, 1, 0)).unwrap().sample_rate(), out);
        }
    }

    #[test]
    fn off_ladder_bitrates_and_surround_are_refused() {
        let e = Mp3Encoder::new(config(48_000, 2, 100_000)).err().unwrap();
        assert!(e.to_string().contains("not an MPEG-1 Layer III bitrate"), "{e}");
        let e = Mp3Encoder::new(config(48_000, 6, 0)).err().unwrap();
        assert!(e.to_string().contains("one or two channels"), "{e}");
    }

    /// A second of stereo 44.1 kHz comes back as whole CBR frames of the
    /// asked-for rate; decoded through the workspace's decoder with the tag
    /// frame in front, the gapless output is exactly the input, in time.
    #[test]
    fn encodes_whole_cbr_frames_and_a_tag_that_trims_exactly() {
        let mut enc = Mp3Encoder::new(config(44_100, 2, 0)).unwrap();
        assert_eq!(enc.pre_skip(), 528 + 529);
        assert!(enc.file_header().is_none(), "not before the flush");
        let n = 44_100;
        let pcm: Vec<f32> = (0..n * 2).map(|i| 0.25 * ((i / 2) as f32 * 0.0627).sin()).collect();
        let mut packets = Vec::new();
        for c in pcm.chunks(2000) {
            packets.extend(enc.encode(&AudioFrame { samples: c.to_vec(), sample_rate: 44_100, channels: 2, pts: 0 }).unwrap());
        }
        packets.extend(enc.flush().unwrap());
        for p in &packets {
            assert_eq!(p.duration, 1152);
            assert_eq!(p.data[0..2], [0xFF, 0xFB], "MPEG-1 Layer III, no CRC");
            assert_eq!(p.data[2] >> 4, 9, "128 kbps");
        }
        assert!(packets.len() * 1152 >= n + 528 + 529);
        let tag = enc.file_header().unwrap();
        assert_eq!(&tag[156..164], ENCODER_NAME.as_bytes(), "the extension's encoder string");
        let mut file = tag;
        for p in &packets {
            file.extend_from_slice(&p.data);
        }
        let decoded: Vec<f32> = ::mp3::Decoder::decode_all(&file).unwrap().into_iter().flat_map(|f| f.samples).collect();
        assert_eq!(decoded.len(), n * 2, "the tag's delay and padding give back exactly the input");
        let (mut s, mut e) = (0.0f64, 0.0f64);
        for (a, b) in pcm[4000..n * 2 - 4000].iter().zip(&decoded[4000..n * 2 - 4000]) {
            s += f64::from(*a).powi(2);
            e += f64::from(a - b).powi(2);
        }
        let snr = 10.0 * (s / e).log10();
        eprintln!("mp3 stereo 128k, gapless: {snr:.1} dB");
        assert!(snr > 20.0, "{snr:.1} dB");
    }

    /// Input at another rate is resampled in time: the tag still trims to
    /// the input's length at the coded rate.
    #[test]
    fn a_resampled_source_keeps_the_codec_delay_alone() {
        let mut enc = Mp3Encoder::new(config(22_050, 1, 0)).unwrap();
        assert_eq!(enc.sample_rate(), 44_100);
        assert_eq!(enc.pre_skip(), 528 + 529, "the resampler's delay is trimmed, not added");
        let pcm: Vec<f32> = (0..22_050).map(|i| 0.25 * (i as f32 * 0.1).sin()).collect();
        let mut packets = enc.encode(&AudioFrame { samples: pcm, sample_rate: 22_050, channels: 1, pts: 0 }).unwrap();
        packets.extend(enc.flush().unwrap());
        let mut file = enc.file_header().unwrap();
        for p in &packets {
            file.extend_from_slice(&p.data);
        }
        let decoded: usize = ::mp3::Decoder::decode_all(&file).unwrap().iter().map(|f| f.len()).sum();
        assert_eq!(decoded, 44_100);
    }
}
