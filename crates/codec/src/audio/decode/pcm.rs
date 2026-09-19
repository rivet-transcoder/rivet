//! Linear PCM: samples as they lie in the container, converted to the
//! pipeline's interleaved f32 in `[-1.0, 1.0]`.
//!
//! Nothing is decoded — the bytes *are* the samples. What a container
//! supplies (AVI's `WAVE_FORMAT_PCM` / `WAVE_FORMAT_IEEE_FLOAT` /
//! `WAVE_FORMAT_EXTENSIBLE` chunks) is a run of little-endian sample frames,
//! one sample per channel each, in WAVE channel order (FL FR FC LFE BL BR …,
//! which is the pipeline's native order), cut into packets that need not
//! end on a frame. A partial frame at the end of a packet waits for the
//! next one.

use crate::audio::{AudioDecoder, AudioError, AudioFrame};

/// One PCM sample encoding, by the codec name the demuxers give it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PcmFormat {
    /// `pcm_u8`: unsigned 8-bit, 128 is silence.
    U8,
    /// `pcm_s16le`.
    S16Le,
    /// `pcm_s24le`: three bytes per sample.
    S24Le,
    /// `pcm_s32le`.
    S32Le,
    /// `pcm_f32le`: IEEE float.
    F32Le,
    /// `pcm_f64le`: IEEE double.
    F64Le,
}

impl PcmFormat {
    /// The format a codec name stands for (`pcm_s16le`, …); `None` for
    /// anything else.
    pub fn from_codec(codec: &str) -> Option<Self> {
        Some(match codec {
            "pcm_u8" => Self::U8,
            "pcm_s16le" => Self::S16Le,
            "pcm_s24le" => Self::S24Le,
            "pcm_s32le" => Self::S32Le,
            "pcm_f32le" => Self::F32Le,
            "pcm_f64le" => Self::F64Le,
            _ => return None,
        })
    }

    /// Bytes one sample of one channel takes.
    pub fn bytes_per_sample(self) -> usize {
        match self {
            Self::U8 => 1,
            Self::S16Le => 2,
            Self::S24Le => 3,
            Self::S32Le | Self::F32Le => 4,
            Self::F64Le => 8,
        }
    }

    /// One sample (`bytes_per_sample` bytes) as f32 in `[-1.0, 1.0]`.
    fn sample(self, b: &[u8]) -> f32 {
        match self {
            Self::U8 => (f32::from(b[0]) - 128.0) / 128.0,
            Self::S16Le => f32::from(i16::from_le_bytes([b[0], b[1]])) / 32_768.0,
            Self::S24Le => (i32::from_le_bytes([0, b[0], b[1], b[2]]) >> 8) as f32 / 8_388_608.0,
            Self::S32Le => i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32 / 2_147_483_648.0,
            Self::F32Le => f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
            Self::F64Le => f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as f32,
        }
    }
}

/// Linear PCM through the [`AudioDecoder`] surface.
pub struct PcmDecoder {
    format: PcmFormat,
    sample_rate: u32,
    channels: u8,
    /// Bytes of a sample frame the last packet ended inside of.
    pending: Vec<u8>,
    /// Microseconds of the first sample, from the first packet's PTS.
    first_pts_us: Option<i64>,
    /// Sample frames handed out so far.
    frames_out: u64,
}

impl PcmDecoder {
    pub fn new(codec: &str, sample_rate: u32, channels: u8) -> Result<Self, AudioError> {
        let format = PcmFormat::from_codec(codec)
            .ok_or_else(|| AudioError::Unsupported(format!("PCM sample format {codec}")))?;
        if channels == 0 || sample_rate == 0 {
            return Err(AudioError::Decode(format!(
                "PCM needs a channel count and a sample rate (got {channels} channel(s) at {sample_rate} Hz)"
            )));
        }
        Ok(Self { format, sample_rate, channels, pending: Vec::new(), first_pts_us: None, frames_out: 0 })
    }
}

impl AudioDecoder for PcmDecoder {
    fn decode(&mut self, packet: &[u8], pts: i64) -> Result<Vec<AudioFrame>, AudioError> {
        self.pending.extend_from_slice(packet);
        let first_pts_us = *self.first_pts_us.get_or_insert(pts);
        let frame_bytes = self.format.bytes_per_sample() * usize::from(self.channels);
        let whole = self.pending.len() / frame_bytes * frame_bytes;
        if whole == 0 {
            return Ok(Vec::new());
        }
        let samples: Vec<f32> = self.pending[..whole]
            .chunks_exact(self.format.bytes_per_sample())
            .map(|b| self.format.sample(b))
            .collect();
        self.pending.drain(..whole);
        let pts = first_pts_us + (self.frames_out as i64 * 1_000_000) / i64::from(self.sample_rate);
        self.frames_out += (whole / frame_bytes) as u64;
        Ok(vec![AudioFrame { samples, sample_rate: self.sample_rate, channels: self.channels, pts }])
    }

    fn flush(&mut self) -> Result<Vec<AudioFrame>, AudioError> {
        // A partial sample frame at the very end is not a sample.
        self.pending.clear();
        Ok(Vec::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_format_maps_full_scale_and_silence() {
        let cases: [(&str, Vec<u8>, f32); 8] = [
            ("pcm_u8", vec![128], 0.0),
            ("pcm_u8", vec![0], -1.0),
            ("pcm_s16le", 0x8000u16.to_le_bytes().to_vec(), -1.0),
            ("pcm_s16le", 0x4000u16.to_le_bytes().to_vec(), 0.5),
            ("pcm_s24le", vec![0x00, 0x00, 0xC0], -0.5),
            ("pcm_s32le", 0x4000_0000u32.to_le_bytes().to_vec(), 0.5),
            ("pcm_f32le", 0.25f32.to_le_bytes().to_vec(), 0.25),
            ("pcm_f64le", (-0.75f64).to_le_bytes().to_vec(), -0.75),
        ];
        for (codec, bytes, want) in cases {
            let mut dec = PcmDecoder::new(codec, 48_000, 1).unwrap();
            let frames = dec.decode(&bytes, 0).unwrap();
            assert_eq!(frames.len(), 1, "{codec}");
            assert_eq!(frames[0].samples, vec![want], "{codec} {bytes:02X?}");
        }
    }

    /// A packet may end inside a sample frame (AVI chunks are byte runs):
    /// the rest waits for the next packet, and the timestamps count frames.
    #[test]
    fn a_frame_split_across_packets_is_joined_and_timed() {
        let mut dec = PcmDecoder::new("pcm_s16le", 48_000, 2).unwrap();
        // Three stereo frames (12 bytes) cut 5 / 7.
        let bytes: Vec<u8> = [100i16, -100, 200, -200, 300, -300].iter().flat_map(|s| s.to_le_bytes()).collect();
        let a = dec.decode(&bytes[..5], 1_000).unwrap();
        assert_eq!(a[0].samples.len(), 2, "one whole frame");
        assert_eq!(a[0].pts, 1_000);
        let b = dec.decode(&bytes[5..], 9_999).unwrap();
        assert_eq!(b[0].samples, vec![200.0 / 32_768.0, -200.0 / 32_768.0, 300.0 / 32_768.0, -300.0 / 32_768.0]);
        assert_eq!(b[0].pts, 1_000 + 1_000_000 / 48_000, "one frame after the first packet's PTS");
        assert!(dec.decode(&[1], 0).unwrap().is_empty());
        assert!(dec.flush().unwrap().is_empty(), "a partial frame at the end is dropped");
    }

    #[test]
    fn other_codecs_are_refused_by_name() {
        let err = PcmDecoder::new("pcm_alaw", 8_000, 1).err().expect("not linear PCM");
        assert!(err.to_string().contains("pcm_alaw"), "{err}");
    }
}
