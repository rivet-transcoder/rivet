//! Opus decode through the workspace's Opus decoder (`opus`, the
//! `crates/opus` submodule, the rivet-opus repository), adapted to
//! [`AudioDecoder`].
//!
//! One code path for every layout: a mono or stereo stream (channel-mapping
//! family 0) is a multistream decoder with one stream, and families 1
//! (Vorbis layouts, 1–8 channels) and 255 (no defined layout) carry their
//! stream counts and mapping in the `OpusHead`. Family 1 output is in RFC
//! 7845 §5.1.1.2's (Vorbis) order and is permuted into the pipeline's native
//! order ([`crate::audio::rfc7845_family1_order`]); family 255 has no speaker
//! positions to give its channels, so it is refused by name. The head's
//! output gain is applied.
//!
//! Output is always 48 kHz — the rate Opus codes at, whatever
//! `InputSampleRate` says — and **includes** the pre-skip: which samples a
//! track presents is its container's business (an MP4 edit list, a Matroska
//! `CodecDelay`, an Ogg granule position), and the job layer applies that
//! exactly, so dropping them here as well would cut the start twice.

pub use ::opus::OpusHead;

use crate::audio::{AudioDecoder, AudioError, AudioFrame};

pub struct OpusDecoder {
    inner: ::opus::MultistreamDecoder,
    channels: u8,
    /// For family 1, the native slot each RFC slot carries.
    order: Option<&'static [usize]>,
    next_pts_us: Option<i64>,
}

fn decode_error(e: ::opus::Error) -> AudioError {
    match e {
        ::opus::Error::Unsupported(m) => AudioError::Unsupported(format!("opus: {m}")),
        other => AudioError::Decode(format!("opus: {other}")),
    }
}

impl OpusDecoder {
    /// `extra_data` is the `OpusHead` (with or without its magic); without
    /// one, a mono or stereo stream of `channels` is assumed (family 0).
    pub fn new(extra_data: Option<&[u8]>, channels: u8) -> Result<Self, AudioError> {
        let head = match extra_data {
            Some(body) => OpusHead::parse(body).map_err(decode_error)?,
            None if (1..=2).contains(&channels) => OpusHead::new(channels, 0, 48_000).map_err(decode_error)?,
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
        let inner = ::opus::MultistreamDecoder::from_head(&head, 48_000).map_err(decode_error)?;
        Ok(Self { inner, channels: head.channels, order, next_pts_us: None })
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
        let mut samples = self.inner.decode(Some(packet)).map_err(decode_error)?;
        let ch = usize::from(self.channels);
        if let Some(order) = self.order {
            let mut tmp = [0.0f32; 8];
            for f in samples.chunks_exact_mut(ch) {
                tmp[..ch].copy_from_slice(f);
                for (slot, &native) in order.iter().enumerate() {
                    f[native] = tmp[slot];
                }
            }
        }
        let n = (samples.len() / ch.max(1)) as i64;
        let pts = self.next_pts_us.unwrap_or(0);
        self.next_pts_us = Some(pts + n * 1_000_000 / 48_000);
        Ok(vec![AudioFrame { samples, sample_rate: 48_000, channels: self.channels, pts }])
    }

    fn flush(&mut self) -> Result<Vec<AudioFrame>, AudioError> {
        Ok(Vec::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn family_255_is_refused_by_name() {
        let mut head = vec![1, 3, 0, 0, 0x80, 0xBB, 0, 0, 0, 0, 255, 3, 0];
        head.extend_from_slice(&[0, 1, 2]);
        let err = OpusDecoder::new(Some(&head), 3).err().unwrap();
        assert!(err.to_string().contains("family 255"), "{err}");
    }

    #[test]
    fn a_damaged_packet_is_an_error_not_a_panic() {
        let mut dec = OpusDecoder::new(None, 2).unwrap();
        // Code 3 with a frame count of zero breaks [R5].
        assert!(dec.decode(&[0xFB, 0x00], 0).is_err());
        assert!(dec.decode(&[], 0).unwrap().is_empty());
    }
}
