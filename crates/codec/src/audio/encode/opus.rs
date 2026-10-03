//! Opus output through the workspace's Opus encoder (`opus`, the
//! `crates/opus` submodule, the rivet-opus repository), adapted to
//! [`AudioEncoder`].
//!
//! - **One code path for every layout.** Mono and stereo are a one-stream
//!   [`opus::MultistreamEncoder`] under channel-mapping family 0; 3.0 to 7.1
//!   are family 1 (RFC 7845 §5.1.1.2), whose streams take their input in the
//!   Vorbis channel order, so the pipeline's native order is permuted on the
//!   way in ([`crate::audio::rfc7845_family1_order`]).
//! - **48 kHz.** Opus is coded at 48 kHz whatever the input; other rates go
//!   through an [`AlignedResampler`] first, whose delay is trimmed, so the
//!   stream's pre-skip is the encoder's own lookahead alone
//!   ([`opus::LOOKAHEAD_48K`], 312 samples, 6.5 ms).
//! - **20 ms packets**, CELT (the `Audio` application), VBR. The bit rate is
//!   the total for all streams; `0` is 64 kb/s per mono stream and 96 kb/s
//!   per coupled pair (64k mono, 96k stereo, 320k 5.1, 416k 7.1).
//! - **Exact length.** The last packet is followed by as many silent ones as
//!   it takes for the decoded stream to cover the pre-skip and every input
//!   sample, so an edit list (or a granule position) can end it exactly.
//! - [`AudioEncoder::extra_data`] is the `OpusHead` body (RFC 7845 §5.1,
//!   without the magic; the `dOps` box's fields, little-endian as in Ogg).

use crate::audio::resample::AlignedResampler;
use crate::audio::{AudioCodec, AudioEncoder, AudioEncoderConfig, AudioError, AudioFrame, EncodedAudioPacket};

/// Samples per channel in one packet: 20 ms at 48 kHz.
pub const FRAME_SAMPLES: usize = 960;
/// The rate Opus is coded at, and the timescale of its packets.
pub const OPUS_RATE: u32 = 48_000;
const DEFAULT_BITRATE_MONO: u32 = 64_000;
const DEFAULT_BITRATE_STEREO: u32 = 96_000;
/// The lowest bit rate of one Opus stream (RFC 6716 §2.1.1: 6 kb/s).
pub const STREAM_BITRATE_MIN: u32 = 6_000;
/// The highest bit rate of one Opus stream (510 kb/s).
pub const STREAM_BITRATE_MAX: u32 = 510_000;

/// The default bit rate for `channels`: 96 kb/s per coupled stream of the
/// family 0 / 1 layout, 64 kb/s per mono one.
pub fn default_bitrate(channels: u8) -> u32 {
    match ::opus::family1_layout(channels) {
        Some((streams, coupled, _)) => {
            u32::from(coupled) * DEFAULT_BITRATE_STEREO + u32::from(streams - coupled) * DEFAULT_BITRATE_MONO
        }
        None => DEFAULT_BITRATE_STEREO,
    }
}

/// The total bit rates `channels` can be coded at: every stream of the
/// layout between 6 and 510 kb/s.
pub fn bitrate_range(channels: u8) -> (u32, u32) {
    let streams = ::opus::family1_layout(channels).map_or(1, |(s, _, _)| u32::from(s));
    (STREAM_BITRATE_MIN * streams, STREAM_BITRATE_MAX * streams)
}

fn encode_error(e: ::opus::Error) -> AudioError {
    match e {
        ::opus::Error::Config(m) | ::opus::Error::Unsupported(m) => AudioError::Unsupported(format!("opus: {m}")),
        other => AudioError::Encode(format!("opus: {other}")),
    }
}

pub struct OpusEncoder {
    inner: ::opus::MultistreamEncoder,
    channels: u8,
    in_rate: u32,
    resampler: AlignedResampler,
    /// Samples at 48 kHz (native order) not yet in a packet.
    carry: Vec<f32>,
    /// Samples per channel at 48 kHz handed to the encoder (padding included).
    coded: u64,
    /// The `OpusHead` body: `dOps`'s fields, version 0.
    head: Vec<u8>,
    pre_skip: u16,
    first_pts: Option<i64>,
    packets_out: u64,
}

impl OpusEncoder {
    pub fn new(config: AudioEncoderConfig) -> Result<Self, AudioError> {
        if config.codec != AudioCodec::Opus {
            return Err(AudioError::Encode(format!("OpusEncoder constructed with codec {:?}", config.codec)));
        }
        if !(1..=8).contains(&config.channels) {
            return Err(AudioError::Unsupported(format!(
                "Opus carries 1 to 8 channels (channel-mapping families 0 and 1, RFC 7845 §5.1.1.2); got {}",
                config.channels
            )));
        }
        if config.sample_rate == 0 {
            return Err(AudioError::Encode("input sample_rate is 0".to_string()));
        }
        let bitrate = if config.bitrate == 0 { default_bitrate(config.channels) } else { config.bitrate };
        let (lo, hi) = bitrate_range(config.channels);
        if !(lo..=hi).contains(&bitrate) {
            return Err(AudioError::Unsupported(format!(
                "Opus at {bitrate} bps for {} channels: the range is {lo}..={hi} (6 to 510 kb/s per stream)",
                config.channels
            )));
        }
        let inner = ::opus::MultistreamEncoder::new(::opus::EncoderConfig {
            sample_rate: OPUS_RATE,
            channels: usize::from(config.channels),
            bitrate,
            vbr: true,
            frame_size: FRAME_SAMPLES,
            ..::opus::EncoderConfig::default()
        })
        .map_err(encode_error)?;
        let pre_skip = u16::try_from(inner.lookahead()).unwrap_or(u16::MAX);
        let mut head = inner.head().clone();
        head.pre_skip = pre_skip;
        head.input_sample_rate = config.sample_rate;
        // The `dOps` box's version; the Ogg and Matroska writers put 1 back.
        head.version = 0;
        Ok(Self {
            channels: config.channels,
            in_rate: config.sample_rate,
            resampler: AlignedResampler::new(config.sample_rate, OPUS_RATE, config.channels)?,
            carry: Vec::with_capacity(FRAME_SAMPLES * usize::from(config.channels) * 2),
            coded: 0,
            head: head.body(),
            pre_skip,
            first_pts: None,
            packets_out: 0,
            inner,
        })
    }

    /// Encode every whole packet `carry` holds.
    fn drain(&mut self) -> Result<Vec<EncodedAudioPacket>, AudioError> {
        let ch = usize::from(self.channels);
        let len = FRAME_SAMPLES * ch;
        let order = crate::audio::rfc7845_family1_order(self.channels);
        let mut out = Vec::new();
        let mut at = 0;
        let mut frame = vec![0.0f32; len];
        while self.carry.len() - at >= len {
            let src = &self.carry[at..at + len];
            match order {
                // Family 1 takes the Vorbis order: slot `s` carries native `order[s]`.
                Some(order) => {
                    for (dst, src) in frame.chunks_exact_mut(ch).zip(src.chunks_exact(ch)) {
                        for (slot, &native) in order.iter().enumerate() {
                            dst[slot] = src[native];
                        }
                    }
                }
                None => frame.copy_from_slice(src),
            }
            let data = self.inner.encode(&frame).map_err(encode_error)?;
            let first = self.first_pts.unwrap_or(0);
            let pts = first + (self.packets_out * FRAME_SAMPLES as u64 * 1_000_000 / u64::from(OPUS_RATE)) as i64;
            self.packets_out += 1;
            self.coded += FRAME_SAMPLES as u64;
            out.push(EncodedAudioPacket { data, pts, duration: FRAME_SAMPLES as i64 });
            at += len;
        }
        self.carry.drain(..at);
        Ok(out)
    }
}

impl AudioEncoder for OpusEncoder {
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
        self.resampler.process(frame, &mut self.carry)?;
        self.drain()
    }

    fn flush(&mut self) -> Result<Vec<EncodedAudioPacket>, AudioError> {
        self.resampler.flush(&mut self.carry)?;
        // Silence until the decoded stream covers the pre-skip and every
        // input sample: whole packets past the end of the input.
        let ch = usize::from(self.channels);
        let need = self.resampler.target_len() + u64::from(self.pre_skip);
        let have = self.coded + (self.carry.len() / ch) as u64;
        let total = need.max(have).div_ceil(FRAME_SAMPLES as u64) * FRAME_SAMPLES as u64;
        self.carry.resize(self.carry.len() + (total - have) as usize * ch, 0.0);
        self.drain()
    }

    /// The encoder's lookahead, in 48 kHz samples (the `OpusHead` PreSkip).
    fn pre_skip(&self) -> u16 {
        self.pre_skip
    }

    /// The `OpusHead` body (RFC 7845 §5.1 without the magic; `dOps` version 0).
    fn extra_data(&self) -> Vec<u8> {
        self.head.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::AudioDecoder;
    use crate::audio::decode::opus::OpusDecoder;

    fn config(sample_rate: u32, channels: u8, bitrate: u32) -> AudioEncoderConfig {
        AudioEncoderConfig { codec: AudioCodec::Opus, sample_rate, channels, bitrate, quality: None, layout: None }
    }

    /// Interleaved frames where channel `c` is a tone at `freqs[c]`.
    fn tones(freqs: &[f32], rate: u32, frames: usize) -> Vec<f32> {
        let ch = freqs.len();
        (0..frames * ch)
            .map(|i| {
                let (t, c) = (i / ch, i % ch);
                0.3 * (2.0 * std::f32::consts::PI * freqs[c] * t as f32 / rate as f32).sin()
            })
            .collect()
    }

    fn encode_all(enc: &mut OpusEncoder, pcm: &[f32], rate: u32, channels: u8) -> Vec<EncodedAudioPacket> {
        let mut packets = Vec::new();
        for (i, chunk) in pcm.chunks(usize::from(channels) * 1000).enumerate() {
            let frame = AudioFrame { samples: chunk.to_vec(), sample_rate: rate, channels, pts: i as i64 * 1000 };
            packets.extend(enc.encode(&frame).unwrap());
        }
        packets.extend(enc.flush().unwrap());
        packets
    }

    fn decode_all(head: &[u8], channels: u8, packets: &[EncodedAudioPacket]) -> Vec<f32> {
        let mut dec = OpusDecoder::new(Some(head), channels).unwrap();
        let mut out = Vec::new();
        for p in packets {
            for f in dec.decode(&p.data, 0).unwrap() {
                out.extend(f.samples);
            }
        }
        out
    }

    fn snr_db(reference: &[f32], decoded: &[f32]) -> f64 {
        let (mut s, mut n) = (0.0f64, 0.0f64);
        for (&a, &b) in reference.iter().zip(decoded) {
            s += f64::from(a).powi(2);
            n += f64::from(a - b).powi(2);
        }
        10.0 * (s / n.max(1e-30)).log10()
    }

    #[test]
    fn heads_follow_rfc_7845_for_every_layout() {
        for ch in 1..=8u8 {
            let enc = OpusEncoder::new(config(44_100, ch, 0)).unwrap();
            let d = enc.extra_data();
            let family = if ch <= 2 { 0 } else { 1 };
            assert_eq!(d.len(), if ch <= 2 { 11 } else { 13 + usize::from(ch) }, "{ch}");
            assert_eq!((d[0], d[1], d[10]), (0, ch, family), "{ch}");
            assert_eq!(u16::from_le_bytes([d[2], d[3]]), enc.pre_skip());
            assert_eq!(u32::from_le_bytes([d[4], d[5], d[6], d[7]]), 44_100, "InputSampleRate is the source's");
            assert_eq!(i16::from_le_bytes([d[8], d[9]]), 0);
            if ch > 2 {
                let (streams, coupled, mapping) = ::opus::family1_layout(ch).unwrap();
                assert_eq!((d[11], d[12]), (streams, coupled));
                assert_eq!(&d[13..], mapping);
            }
        }
        assert_eq!(OpusEncoder::new(config(48_000, 2, 0)).unwrap().pre_skip(), 312);
        assert_eq!(default_bitrate(1), 64_000);
        assert_eq!(default_bitrate(2), 96_000);
        assert_eq!(default_bitrate(6), 320_000);
        assert_eq!(default_bitrate(8), 416_000);
    }

    #[test]
    fn bad_configurations_are_refused() {
        assert!(matches!(OpusEncoder::new(config(48_000, 0, 0)), Err(AudioError::Unsupported(_))));
        assert!(matches!(OpusEncoder::new(config(48_000, 9, 0)), Err(AudioError::Unsupported(_))));
        assert!(matches!(OpusEncoder::new(config(48_000, 2, 1_000)), Err(AudioError::Unsupported(_))));
        assert!(matches!(OpusEncoder::new(config(0, 2, 0)), Err(AudioError::Encode(_))));
        let mut enc = OpusEncoder::new(config(48_000, 2, 0)).unwrap();
        let nine = AudioFrame { samples: vec![0.0; 960 * 9], sample_rate: 48_000, channels: 9, pts: 0 };
        assert!(enc.encode(&nine).is_err());
    }

    /// Packets are 20 ms, timed from the first input PTS, and the stream
    /// decodes (through rivet's own decoder) to the pre-skip plus at least
    /// every input sample, the input coming back at the pre-skip.
    #[test]
    fn stereo_round_trip_is_exact_in_length_and_time() {
        let n = 48_000;
        let pcm = tones(&[440.0, 660.0], 48_000, n);
        let mut enc = OpusEncoder::new(config(48_000, 2, 128_000)).unwrap();
        let packets = encode_all(&mut enc, &pcm, 48_000, 2);
        assert!(packets.iter().all(|p| p.duration == 960));
        assert_eq!(packets[0].pts, 0);
        assert_eq!(packets[1].pts, 20_000);
        assert_eq!(packets.len(), (n + 312).div_ceil(960));
        let out = decode_all(&enc.extra_data(), 2, &packets);
        let skip = usize::from(enc.pre_skip()) * 2;
        let decoded = &out[skip..skip + n * 2];
        let snr = snr_db(&pcm[4800..n * 2 - 4800], &decoded[4800..n * 2 - 4800]);
        eprintln!("opus stereo 128k: {snr:.1} dB");
        assert!(snr > 15.0, "{snr:.1} dB");
    }

    /// 44.1 kHz input is resampled without moving it: the decoded tone lines
    /// up with the same tone generated at 48 kHz.
    #[test]
    fn other_rates_are_resampled_in_time() {
        let pcm = tones(&[440.0], 44_100, 44_100);
        let mut enc = OpusEncoder::new(config(44_100, 1, 96_000)).unwrap();
        let packets = encode_all(&mut enc, &pcm, 44_100, 1);
        assert_eq!(packets.len(), (48_000 + 312usize).div_ceil(960));
        let out = decode_all(&enc.extra_data(), 1, &packets);
        let reference = tones(&[440.0], 48_000, 48_000);
        let decoded = &out[312..312 + 48_000];
        let snr = snr_db(&reference[4800..43_200], &decoded[4800..43_200]);
        eprintln!("opus mono 44.1 kHz input: {snr:.1} dB against the tone at 48 kHz");
        assert!(snr > 15.0, "{snr:.1} dB");
    }

    /// 5.1 in the native order comes back channel for channel: the encoder's
    /// permutation into the Vorbis order and the decoder's out of it are
    /// inverses, and every channel keeps its own tone.
    #[test]
    fn five_one_round_trips_channel_for_channel() {
        let freqs = [300.0, 500.0, 700.0, 110.0, 1100.0, 1300.0];
        let mut enc = OpusEncoder::new(config(48_000, 6, 0)).unwrap();
        let pcm = tones(&freqs, 48_000, 48_000);
        let packets = encode_all(&mut enc, &pcm, 48_000, 6);
        let out = decode_all(&enc.extra_data(), 6, &packets);
        let skip = 312 * 6;
        for (c, f) in freqs.iter().enumerate() {
            let want: Vec<f32> = pcm.iter().skip(c).step_by(6).copied().collect();
            let got: Vec<f32> = out[skip..].iter().skip(c).step_by(6).copied().collect();
            let snr = snr_db(&want[4800..43_200], &got[4800..43_200]);
            eprintln!("opus 5.1 channel {c} ({f} Hz): {snr:.1} dB");
            assert!(snr > 8.0, "channel {c}: {snr:.1} dB");
        }
    }
}
