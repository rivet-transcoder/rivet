//! Opus decoder over libopus's multistream API (`audiopus_sys` FFI; libopus
//! is BSD). The same library the encoder links, so it costs no dependency.
//!
//! One code path for every layout: a mono or stereo stream (channel-mapping
//! family 0) is a multistream decoder with one stream, and families 1
//! (Vorbis layouts, 1–8 channels) and 255 (no defined layout) carry their
//! stream counts and mapping in the `OpusHead`. Family 1 output is in RFC
//! 7845 §5.1.1.2's (Vorbis) order and is permuted into the pipeline's
//! native order ([`crate::audio::rfc7845_family1_order`]); family 255 has no
//! speaker positions to give its channels, so it is refused by name.
//!
//! Output is always 48 kHz — the rate Opus codes at, whatever
//! `InputSampleRate` says — and **includes** the pre-skip: which samples a
//! track presents is its container's business (an MP4 edit list, a Matroska
//! `CodecDelay`), and the job layer applies that exactly, so dropping them
//! here as well would cut the start twice.

use std::ffi::c_int;
use std::ptr;

use audiopus::ffi;

use crate::audio::{AudioDecoder, AudioError, AudioFrame};

/// 120 ms at 48 kHz: the longest an Opus packet decodes to (RFC 6716 §3.2.1).
const MAX_FRAME_SAMPLES: usize = 5760;

/// What an `OpusHead` body (RFC 7845 §5.1, without the 8-byte magic — the
/// form the demuxers surface) says about the stream's layout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpusHead {
    pub channels: u8,
    /// Samples at 48 kHz the decoder's output starts with that are not audio.
    pub pre_skip: u16,
    pub family: u8,
    pub streams: u8,
    pub coupled: u8,
    pub mapping: Vec<u8>,
}

impl OpusHead {
    pub fn parse(body: &[u8]) -> Result<Self, AudioError> {
        let bad = |why: &str| AudioError::Decode(format!("OpusHead: {why}"));
        let body = body.strip_prefix(b"OpusHead").unwrap_or(body);
        if body.len() < 11 {
            return Err(bad(&format!("{} bytes, need 11", body.len())));
        }
        let channels = body[1];
        if channels == 0 {
            return Err(bad("zero channels"));
        }
        let pre_skip = u16::from_le_bytes([body[2], body[3]]);
        let family = body[10];
        if family == 0 {
            if channels > 2 {
                return Err(bad(&format!("family 0 with {channels} channels")));
            }
            return Ok(Self {
                channels,
                pre_skip,
                family,
                streams: 1,
                coupled: channels - 1,
                mapping: (0..channels).collect(),
            });
        }
        let need = 13 + usize::from(channels);
        if body.len() < need {
            return Err(bad(&format!("family {family} needs {need} bytes, has {}", body.len())));
        }
        let (streams, coupled) = (body[11], body[12]);
        if streams == 0 || coupled > streams || usize::from(streams) + usize::from(coupled) > 255 {
            return Err(bad(&format!("{streams} streams, {coupled} coupled")));
        }
        Ok(Self { channels, pre_skip, family, streams, coupled, mapping: body[13..need].to_vec() })
    }
}

pub struct OpusDecoder {
    state: *mut ffi::OpusMSDecoder,
    channels: u8,
    /// For family 1, the native slot each RFC slot carries.
    order: Option<&'static [usize]>,
    pcm: Vec<f32>,
    next_pts_us: Option<i64>,
}

// SAFETY: the libopus decoder state has no thread affinity and is only
// reached through `&mut self`.
unsafe impl Send for OpusDecoder {}

impl OpusDecoder {
    /// `extra_data` is the `OpusHead` body; without one, a mono or stereo
    /// stream of `channels` is assumed (family 0).
    pub fn new(extra_data: Option<&[u8]>, channels: u8) -> Result<Self, AudioError> {
        let head = match extra_data {
            Some(body) => OpusHead::parse(body)?,
            None if (1..=2).contains(&channels) => OpusHead {
                channels,
                pre_skip: 0,
                family: 0,
                streams: 1,
                coupled: channels - 1,
                mapping: (0..channels).collect(),
            },
            None => {
                return Err(AudioError::Decode(format!(
                    "opus: a {channels}-channel stream needs its OpusHead for the stream layout"
                )));
            }
        };
        let order = match head.family {
            0 => None,
            1 if head.channels <= 8 => crate::audio::rfc7845_family1_order(head.channels),
            f => {
                return Err(AudioError::Unsupported(format!(
                    "opus channel-mapping family {f} ({} channels) names no speaker positions",
                    head.channels
                )));
            }
        };
        let mut err: c_int = 0;
        // SAFETY: `mapping` holds `channels` bytes (checked by `parse`), read
        // synchronously; `err` is a valid out-pointer.
        let state = unsafe {
            ffi::opus_multistream_decoder_create(
                48_000,
                c_int::from(head.channels),
                c_int::from(head.streams),
                c_int::from(head.coupled),
                head.mapping.as_ptr(),
                &mut err,
            )
        };
        if state.is_null() || err != ffi::OPUS_OK {
            return Err(AudioError::Decode(format!("opus_multistream_decoder_create failed: code={err}")));
        }
        Ok(Self {
            state,
            channels: head.channels,
            order,
            pcm: vec![0.0; MAX_FRAME_SAMPLES * usize::from(head.channels)],
            next_pts_us: None,
        })
    }
}

impl AudioDecoder for OpusDecoder {
    fn decode(&mut self, packet: &[u8], pts: i64) -> Result<Vec<AudioFrame>, AudioError> {
        if self.next_pts_us.is_none() {
            self.next_pts_us = Some(pts);
        }
        if packet.is_empty() {
            return Ok(Vec::new());
        }
        // SAFETY: `packet` is valid for its length; `pcm` holds
        // MAX_FRAME_SAMPLES frames of `channels` floats, the bound passed.
        let n = unsafe {
            ffi::opus_multistream_decode_float(
                self.state,
                packet.as_ptr(),
                packet.len().min(i32::MAX as usize) as i32,
                self.pcm.as_mut_ptr(),
                MAX_FRAME_SAMPLES as c_int,
                0,
            )
        };
        if n < 0 {
            return Err(AudioError::Decode(format!("opus_multistream_decode_float failed: code={n}")));
        }
        let ch = usize::from(self.channels);
        let mut samples = self.pcm[..n as usize * ch].to_vec();
        if let Some(order) = self.order {
            let mut tmp = [0.0f32; 8];
            for f in samples.chunks_exact_mut(ch) {
                tmp[..ch].copy_from_slice(f);
                for (slot, &native) in order.iter().enumerate() {
                    f[native] = tmp[slot];
                }
            }
        }
        let pts = self.next_pts_us.unwrap_or(0);
        self.next_pts_us = Some(pts + i64::from(n) * 1_000_000 / 48_000);
        Ok(vec![AudioFrame { samples, sample_rate: 48_000, channels: self.channels, pts }])
    }

    fn flush(&mut self) -> Result<Vec<AudioFrame>, AudioError> {
        Ok(Vec::new())
    }
}

impl Drop for OpusDecoder {
    fn drop(&mut self) {
        if !self.state.is_null() {
            // SAFETY: created by opus_multistream_decoder_create, destroyed once.
            unsafe { ffi::opus_multistream_decoder_destroy(self.state) };
            self.state = ptr::null_mut();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::{AudioCodec, AudioEncoderConfig, create_encoder};

    /// Interleaved 48 kHz frames where channel `c` is a tone at `freqs[c]`.
    fn tones(freqs: &[f32], frames: usize) -> AudioFrame {
        let ch = freqs.len();
        let samples = (0..frames * ch)
            .map(|i| {
                let (t, c) = (i / ch, i % ch);
                0.3 * (2.0 * std::f32::consts::PI * freqs[c] * t as f32 / 48_000.0).sin()
            })
            .collect();
        AudioFrame { samples, sample_rate: 48_000, channels: ch as u8, pts: 0 }
    }

    /// Goertzel power of `freq` in channel `c` of interleaved `pcm`.
    fn power(pcm: &[f32], ch: usize, c: usize, freq: f32) -> f32 {
        let w = 2.0 * std::f32::consts::PI * freq / 48_000.0;
        let (mut s1, mut s2) = (0.0f32, 0.0f32);
        for x in pcm.iter().skip(c).step_by(ch) {
            let s = x + 2.0 * w.cos() * s1 - s2;
            s2 = s1;
            s1 = s;
        }
        s1 * s1 + s2 * s2 - 2.0 * w.cos() * s1 * s2
    }

    #[test]
    fn opus_head_reads_both_families() {
        let stereo = [1, 2, 0x38, 0x01, 0x80, 0xBB, 0, 0, 0, 0, 0];
        let h = OpusHead::parse(&stereo).unwrap();
        assert_eq!((h.channels, h.pre_skip, h.family, h.streams, h.coupled), (2, 312, 0, 1, 1));
        let mut surround = vec![1, 6, 0x38, 0x01, 0x80, 0xBB, 0, 0, 0, 0, 1, 4, 2];
        surround.extend_from_slice(&[0, 4, 1, 2, 3, 5]);
        let h = OpusHead::parse(&surround).unwrap();
        assert_eq!((h.channels, h.family, h.streams, h.coupled), (6, 1, 4, 2));
        assert_eq!(h.mapping, vec![0, 4, 1, 2, 3, 5]);
        assert!(OpusHead::parse(&surround[..15]).is_err(), "truncated mapping");
        let mut magic = b"OpusHead".to_vec();
        magic.extend_from_slice(&stereo);
        assert_eq!(OpusHead::parse(&magic).unwrap().channels, 2, "the magic is tolerated");
    }

    /// Encoding 5.1 in the native order and decoding it gives each tone back
    /// in its own slot: the encoder's permutation into the RFC order and the
    /// decoder's out of it are inverses.
    #[test]
    fn five_one_round_trips_channel_for_channel() {
        let freqs = [300.0, 500.0, 700.0, 110.0, 1100.0, 1300.0];
        let mut enc = create_encoder(AudioEncoderConfig {
            codec: AudioCodec::Opus,
            sample_rate: 48_000,
            channels: 6,
            bitrate: 0,
        })
        .unwrap();
        let mut packets = enc.encode(&tones(&freqs, 48_000)).unwrap();
        packets.extend(enc.flush().unwrap());
        let mut dec = OpusDecoder::new(Some(&enc.extra_data()), 6).unwrap();
        let mut pcm = Vec::new();
        for p in &packets {
            for f in dec.decode(&p.data, 0).unwrap() {
                assert_eq!((f.channels, f.sample_rate), (6, 48_000));
                pcm.extend_from_slice(&f.samples);
            }
        }
        // Skip the pre-skip and the attack, measure a steady stretch.
        let steady = &pcm[6 * 4800..6 * 43_200];
        for (c, &f) in freqs.iter().enumerate() {
            let own = power(steady, 6, c, f);
            for (other, &g) in freqs.iter().enumerate() {
                if other != c {
                    assert!(
                        own > 100.0 * power(steady, 6, c, g),
                        "channel {c} carries {g} Hz (channel {other}'s tone) as much as its own {f} Hz"
                    );
                }
            }
        }
    }

    #[test]
    fn family_255_is_refused_by_name() {
        let mut head = vec![1, 3, 0, 0, 0x80, 0xBB, 0, 0, 0, 0, 255, 3, 0];
        head.extend_from_slice(&[0, 1, 2]);
        let err = OpusDecoder::new(Some(&head), 3).err().unwrap();
        assert!(err.to_string().contains("family 255"), "{err}");
    }
}
