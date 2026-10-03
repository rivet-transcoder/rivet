//! Vorbis decode through the workspace's Vorbis decoder (`vorbis`, the
//! `crates/vorbis` submodule, the rivet-vorbis repository), adapted to
//! [`AudioDecoder`].
//!
//! Matroska / WebM carry no Ogg pages: `CodecPrivate` holds the three Vorbis
//! headers (identification, comment, setup) in Xiph lacing, and each block
//! one raw audio packet. An Ogg file's demuxer hands over the same: the
//! headers laced the same way as `extra_data`, then the audio packets.
//!
//! The first audio packet decodes to nothing (it only primes the overlap);
//! each one after to the overlapping halves of it and the one before.
//! Output is in the pipeline's native channel order: Vorbis orders a
//! surround layout as RFC 7845 §5.1.1.2 does (5.1 = FL FC FR RL RR LFE), so
//! the channels are permuted on the way out. Samples are clamped to ±1.0
//! (Vorbis can exceed full scale).

use crate::audio::{AudioDecoder, AudioError, AudioFrame};

pub struct VorbisDecoder {
    inner: ::vorbis::Decoder,
    sample_rate: u32,
    channels: u8,
    /// For each native slot, the Vorbis channel that carries it.
    source_of: Vec<usize>,
    /// Running PTS in microseconds. Set on first `decode` call.
    next_pts_us: Option<i64>,
}

fn decode_error(e: ::vorbis::Error) -> AudioError {
    AudioError::Decode(format!("vorbis: {e}"))
}

impl VorbisDecoder {
    /// `extra_data` is the Xiph-laced concatenation of the three header
    /// packets, as Matroska's `CodecPrivate` carries it. The container's
    /// rate and channel count are only a fallback: the identification
    /// header is authoritative.
    pub fn new(extra_data: Option<&[u8]>, _sample_rate: u32, _channels: u8) -> Result<Self, AudioError> {
        let extra = extra_data.ok_or_else(|| {
            AudioError::Decode("vorbis decoder needs CodecPrivate-style setup headers as extra_data".to_string())
        })?;
        let inner = ::vorbis::Decoder::from_xiph_lacing(extra).map_err(decode_error)?;
        let channels = inner.channels();
        // 1..=8: the layouts the pipeline names; Vorbis itself allows 255.
        if !(1..=8).contains(&channels) {
            return Err(AudioError::Unsupported(format!(
                "vorbis channel count {channels} (1..=8 supported: mono to 7.1)"
            )));
        }
        let channels = channels as u8;
        let source_of = match crate::audio::rfc7845_family1_order(channels) {
            Some(order) => (0..usize::from(channels))
                .map(|native| order.iter().position(|&n| n == native).unwrap_or(native))
                .collect(),
            None => (0..usize::from(channels)).collect(),
        };
        Ok(Self { sample_rate: inner.sample_rate(), inner, channels, source_of, next_pts_us: None })
    }
}

impl AudioDecoder for VorbisDecoder {
    fn decode(&mut self, packet: &[u8], pts: i64) -> Result<Vec<AudioFrame>, AudioError> {
        if self.next_pts_us.is_none() {
            self.next_pts_us = Some(pts);
        }
        if packet.is_empty() {
            return Ok(Vec::new());
        }
        let planar = self.inner.decode(packet).map_err(decode_error)?;
        let n = planar.first().map_or(0, Vec::len);
        if n == 0 {
            return Ok(Vec::new());
        }
        let mut interleaved = Vec::with_capacity(n * self.source_of.len());
        for i in 0..n {
            for &src in &self.source_of {
                interleaved.push(planar[src][i].clamp(-1.0, 1.0));
            }
        }
        let pts_us = self.next_pts_us.unwrap_or(pts);
        self.next_pts_us = Some(pts_us + n as i64 * 1_000_000 / i64::from(self.sample_rate.max(1)));
        Ok(vec![AudioFrame { samples: interleaved, sample_rate: self.sample_rate, channels: self.channels, pts: pts_us }])
    }

    fn flush(&mut self) -> Result<Vec<AudioFrame>, AudioError> {
        // Each packet's samples come out when the next one overlaps them;
        // the last packet's right half is never output (Vorbis I §4.3.8).
        Ok(Vec::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_or_garbage_headers_are_errors() {
        assert!(matches!(VorbisDecoder::new(None, 44_100, 2), Err(AudioError::Decode(_))));
        let extra = vec![2u8, 30, 19, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF];
        assert!(matches!(VorbisDecoder::new(Some(&extra), 44_100, 2), Err(AudioError::Decode(_))));
    }

    /// 5.1 from the workspace's own encoder, in Vorbis order, comes out in
    /// the native order: each tone in its own native slot.
    #[test]
    fn surround_comes_out_in_native_order() {
        let freqs = [300.0f32, 500.0, 700.0, 900.0, 1100.0, 60.0]; // Vorbis: FL C FR RL RR LFE
        let n = 44_100;
        let pcm: Vec<f32> = (0..n * 6)
            .map(|i| 0.3 * (2.0 * std::f32::consts::PI * freqs[i % 6] * (i / 6) as f32 / 44_100.0).sin())
            .collect();
        let mut enc = ::vorbis::Encoder::new(::vorbis::EncoderConfig {
            sample_rate: 44_100,
            channels: 6,
            quality: 6.0,
            comments: Vec::new(),
        })
        .unwrap();
        let mut packets = enc.encode(&pcm).unwrap();
        packets.extend(enc.finish().unwrap());
        let mut dec = VorbisDecoder::new(Some(&enc.codec_private()), 44_100, 6).unwrap();
        let mut out = Vec::new();
        for p in &packets {
            for f in dec.decode(&p.data, 0).unwrap() {
                assert_eq!((f.channels, f.sample_rate), (6, 44_100));
                out.extend(f.samples);
            }
        }
        // native FL FR FC LFE BL BR <- Vorbis FL FR(2) C(1) LFE(5) RL(3) RR(4)
        let vorbis_of_native = [0usize, 2, 1, 5, 3, 4];
        let frames = out.len() / 6;
        for (native, &v) in vorbis_of_native.iter().enumerate() {
            let want: Vec<f32> = pcm.iter().skip(v).step_by(6).take(frames).copied().collect();
            let got: Vec<f32> = out.iter().skip(native).step_by(6).copied().collect();
            let (mut s, mut e) = (0.0f64, 0.0f64);
            for (a, b) in want[4096..frames - 4096].iter().zip(&got[4096..frames - 4096]) {
                s += f64::from(*a).powi(2);
                e += f64::from(a - b).powi(2);
            }
            let snr = 10.0 * (s / e.max(1e-30)).log10();
            assert!(snr > 6.0, "native slot {native}: {snr:.1} dB");
        }
    }
}
