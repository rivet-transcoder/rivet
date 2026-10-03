//! MPEG audio decode (Layers I, II and III; MPEG-1, MPEG-2 LSF, MPEG-2.5)
//! through the workspace's decoder (`mp3`, the `crates/mp3` submodule, the
//! rivet-mp3 repository), adapted to [`AudioDecoder`].
//!
//! Packets are byte runs: an AVI or Matroska MP3 stream need not cut on
//! frames, so the bytes go to the decoder's stream interface, which buffers
//! them, confirms sync against the next frame's header and skips ID3 tags
//! and garbage. A Xing / Info / VBRI tag frame is recognised and not played.
//!
//! The decoder's own gapless trimming is **off**: which samples a track
//! presents is its container's business (an MP4 edit list, or the LAME tag
//! of a bare `.mp3`, which `container::mp3::read_file` turns into one), and
//! the job layer applies that exactly. The output therefore starts with the
//! encoder's delay plus the 529 samples every Layer III decoder adds.
//!
//! PTS: the caller's PTS on the first `decode` call seeds the clock; each
//! frame then steps it by its own length (1152 samples per MPEG-1 Layer II /
//! III frame, 576 per LSF Layer III frame, 384 per Layer I frame).

use crate::audio::{AudioDecoder, AudioError, AudioFrame};

pub struct Mp3Decoder {
    inner: ::mp3::Decoder,
    /// Caller-declared input sample rate, from the container (a fallback
    /// only: every frame header carries its own).
    declared_sample_rate: u32,
    /// Running PTS in microseconds, seeded by the first `decode` call.
    next_pts_us: Option<i64>,
}

fn decode_error(e: ::mp3::Error) -> AudioError {
    match e {
        ::mp3::Error::Unsupported(m) => AudioError::Unsupported(format!("mp3: {m}")),
        other => AudioError::Decode(format!("mp3: {other}")),
    }
}

impl Mp3Decoder {
    pub fn new(sample_rate: u32, channels: u8) -> Result<Self, AudioError> {
        if channels == 0 || channels > 2 {
            return Err(AudioError::Unsupported(format!("mp3 channel count {channels}")));
        }
        let inner = ::mp3::Decoder::with_options(::mp3::DecoderOptions {
            trim_gapless: false,
            ..::mp3::DecoderOptions::default()
        });
        Ok(Self { inner, declared_sample_rate: sample_rate.max(1), next_pts_us: None })
    }

    fn frames(&mut self, frames: Vec<::mp3::Frame>) -> Vec<AudioFrame> {
        let mut out = Vec::with_capacity(frames.len());
        for f in frames {
            if f.is_empty() {
                continue;
            }
            let sample_rate = if f.sample_rate > 0 { f.sample_rate } else { self.declared_sample_rate };
            let pts = self.next_pts_us.unwrap_or(0);
            self.next_pts_us = Some(pts + f.len() as i64 * 1_000_000 / i64::from(sample_rate));
            out.push(AudioFrame { samples: f.samples, sample_rate, channels: f.channels, pts });
        }
        out
    }
}

impl AudioDecoder for Mp3Decoder {
    fn decode(&mut self, packet: &[u8], pts: i64) -> Result<Vec<AudioFrame>, AudioError> {
        if self.next_pts_us.is_none() {
            self.next_pts_us = Some(pts);
        }
        let frames = self.inner.decode(packet).map_err(decode_error)?;
        Ok(self.frames(frames))
    }

    fn flush(&mut self) -> Result<Vec<AudioFrame>, AudioError> {
        let frames = self.inner.flush().map_err(decode_error)?;
        Ok(self.frames(frames))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Frames from the workspace's own encoder: a second of a stereo tone at
    /// 44.1 kHz, 128 kb/s.
    fn encoded(rate: u32) -> Vec<Vec<u8>> {
        let mut enc = ::mp3::Encoder::new(::mp3::EncoderConfig {
            sample_rate: rate,
            channels: 2,
            bitrate: ::mp3::BitrateMode::Cbr(128_000),
            ..Default::default()
        })
        .unwrap();
        let pcm: Vec<f32> = (0..rate as usize * 2).map(|i| 0.3 * ((i / 2) as f32 * 0.06).sin()).collect();
        let mut frames = enc.encode(&pcm);
        frames.extend(enc.flush());
        frames
    }

    #[test]
    fn rejects_zero_or_too_many_channels() {
        assert!(Mp3Decoder::new(44_100, 0).is_err());
        assert!(Mp3Decoder::new(44_100, 6).is_err());
    }

    #[test]
    fn garbage_and_empty_packets_decode_to_nothing() {
        let mut dec = Mp3Decoder::new(44_100, 2).unwrap();
        assert!(dec.decode(&[0u8; 4096], 0).unwrap().is_empty());
        assert!(dec.decode(&[], 12_345).unwrap().is_empty());
        assert!(dec.flush().unwrap().is_empty());
    }

    /// Packets in any chunking decode to every frame, timed from the first
    /// PTS, with the encoder's delay left in (no gapless trimming here).
    #[test]
    fn any_chunking_decodes_every_frame_in_order() {
        let frames = encoded(44_100);
        let bytes: Vec<u8> = frames.concat();
        for chunk in [frames[0].len(), 1000, 7] {
            let mut dec = Mp3Decoder::new(44_100, 2).unwrap();
            let mut out = Vec::new();
            for (i, c) in bytes.chunks(chunk).enumerate() {
                out.extend(dec.decode(c, 1_000 + i as i64).unwrap());
            }
            out.extend(dec.flush().unwrap());
            assert_eq!(out.len(), frames.len(), "chunk {chunk}");
            assert_eq!(out[0].pts, 1_000);
            assert_eq!(out[1].pts, 1_000 + 1152 * 1_000_000 / 44_100);
            assert!(out.iter().all(|f| f.channels == 2 && f.sample_rate == 44_100 && f.samples.len() == 2304));
        }
    }
}
