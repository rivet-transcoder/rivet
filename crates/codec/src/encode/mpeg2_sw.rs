//! MPEG-2 Video encode — this workspace's own encoder (`crates/mpeg2`, the
//! rivet-mpeg2 repository), written clean-room from ITU-T H.262.
//!
//! The only MPEG-2 encoder in the tree, built directly by
//! [`select_encoder`](super::select_encoder) for an MPEG-2 job; see
//! [`native`](super::native).
//!
//! # What it takes and writes
//!
//! Main Profile, 8-bit 4:2:0, progressive frame pictures: I, P and B, in
//! GOPs of the configured keyframe interval (closed first, open after), a
//! sequence header before every GOP so any I picture is a place to start.
//! Interlaced coding, 4:2:2 and 10 bits are not in the crate's encoder, and
//! a rung that needs them is refused.
//!
//! # Order
//!
//! With B pictures (`overrides.bframes`, two when unset) the encoder holds
//! frames back and codes each reference picture before the B pictures that
//! precede it, so coding order is not display order. Its output is split
//! here into one packet per picture (the sequence and GOP headers ride with
//! the picture they precede), each stamped with the timestamp of the frame
//! it codes ([`ReferenceFirst`]); the muxer derives the composition offsets
//! from those.
//!
//! # Rate and quality
//!
//! A constant `quantiser_scale_code` (1-31): the rung's CRF, else the
//! quality target's ([`tuning::native_sw_quantizer`](super::tuning::native_sw_quantizer)).
//! A bitrate rung is coded to its rate by the encoder's own rate control
//! (Test Model 5's picture-level allocation, no VBV model); a constant rate
//! or a coded picture buffer is refused by name. The frame rate must be one
//! H.262 can signal (Table 6-4 and the rates its frame_rate_extension
//! reaches); another is refused when the encoder is built.

use std::collections::VecDeque;

use anyhow::{Context, Result, bail};
use bytes::Bytes;

use super::native::{ReferenceFirst, average_rate, check_frame, frame_rate_ratio, quantizer, tier};
use super::tuning::SpeedTier;
use super::{EncodedPacket, Encoder, EncoderConfig};
use crate::frame::{PixelFormat, VideoCodec, VideoFrame};

/// An MPEG-2 encoder behind rivet's [`Encoder`] trait.
pub struct Mpeg2Encoder {
    inner: mpeg2::Encoder,
    cfg: mpeg2::EncoderConfig,
    order: ReferenceFirst,
    ready: VecDeque<EncodedPacket>,
}

impl Mpeg2Encoder {
    /// Build an encoder for `config` (codec MPEG-2, `yuv420p`).
    pub fn new(config: EncoderConfig) -> Result<Self> {
        if config.codec != VideoCodec::Mpeg2 {
            bail!("the MPEG-2 encoder encodes MPEG-2, not {}", config.codec.label());
        }
        if config.pixel_format != PixelFormat::Yuv420p {
            bail!(
                "the MPEG-2 encoder writes Main Profile, 8-bit 4:2:0 only; this rung asks for {:?}",
                config.pixel_format
            );
        }
        let rate_control = match average_rate("MPEG-2", &config)? {
            Some(bps) => mpeg2::RateControl::Bitrate(bps),
            None => mpeg2::RateControl::ConstantQuantiser(quantizer(&config)),
        };
        let mut cfg = mpeg2::EncoderConfig::new(config.width, config.height);
        cfg.frame_rate = frame_rate_ratio(config.frame_rate);
        cfg.gop_size = config.keyframe_interval.max(1);
        cfg.b_frames = u32::from(config.overrides.bframes.unwrap_or(2));
        cfg.rate_control = rate_control;
        cfg.search_range = match tier(&config) {
            SpeedTier::Draft => 16,
            SpeedTier::Standard => 32,
            SpeedTier::Archive => 64,
        };
        let inner = mpeg2::Encoder::new(cfg.clone()).context("the MPEG-2 encoder rejected the configuration")?;
        Ok(Self { inner, cfg, order: ReferenceFirst::default(), ready: VecDeque::new() })
    }

    /// Split what the encoder returned into one packet per picture, stamped.
    fn collect(&mut self, bytes: Vec<u8>) -> Result<()> {
        let units = split_pictures(&bytes);
        let stamps = self.order.take(units.len())?;
        for (unit, pts) in units.into_iter().zip(stamps) {
            let is_keyframe = picture_type(&unit) == Some(1);
            self.ready.push_back(EncodedPacket { data: Bytes::from(unit), pts, is_keyframe });
        }
        Ok(())
    }
}

/// One access unit per picture: the headers before a picture ride with it.
fn split_pictures(data: &[u8]) -> Vec<Vec<u8>> {
    let codes = start_codes(data);
    let mut starts = Vec::new();
    let mut lead: Option<usize> = None;
    for (o, c) in codes {
        match c {
            0xb3 | 0xb8 => {
                lead.get_or_insert(o);
            }
            0x00 => starts.push(lead.take().unwrap_or(o)),
            _ => {}
        }
    }
    let mut out = Vec::with_capacity(starts.len());
    for (i, &s) in starts.iter().enumerate() {
        let begin = if i == 0 { 0 } else { s };
        let end = starts.get(i + 1).copied().unwrap_or(data.len());
        out.push(data[begin..end].to_vec());
    }
    out
}

fn start_codes(data: &[u8]) -> Vec<(usize, u8)> {
    let mut out = Vec::new();
    let mut i = 0;
    while i + 3 < data.len() {
        if data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1 {
            out.push((i, data[i + 3]));
            i += 3;
        } else {
            i += 1;
        }
    }
    out
}

/// The `picture_coding_type` of the picture in `unit` (1 I, 2 P, 3 B).
fn picture_type(unit: &[u8]) -> Option<u8> {
    let (o, _) = start_codes(unit).into_iter().find(|(_, c)| *c == 0x00)?;
    unit.get(o + 5).map(|b| (b >> 3) & 0x7)
}

impl Encoder for Mpeg2Encoder {
    fn send_frame(&mut self, frame: &VideoFrame) -> Result<()> {
        let want = check_frame("MPEG-2", frame, self.cfg.width, self.cfg.height, &[PixelFormat::Yuv420p])?;
        let mut picture = mpeg2::Frame::new(self.cfg.width, self.cfg.height, mpeg2::ChromaFormat::Yuv420);
        picture.data.copy_from_slice(&frame.data[..want]);
        self.order.push(frame.pts);
        let bytes = self.inner.encode(&picture).context("the MPEG-2 encoder refused a frame")?;
        self.collect(bytes)
    }

    fn flush(&mut self) -> Result<()> {
        let bytes = self.inner.finish().context("the MPEG-2 encoder failed to finish")?;
        self.collect(bytes)
    }

    fn receive_packet(&mut self) -> Result<Option<EncodedPacket>> {
        Ok(self.ready.pop_front())
    }

    /// Rebuild the encoder: the next frame opens a new sequence with a closed
    /// GOP.
    fn reset(&mut self) -> Result<()> {
        self.inner = mpeg2::Encoder::new(self.cfg.clone()).context("rebuilding the MPEG-2 encoder")?;
        self.order.clear();
        self.ready.clear();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One packet per picture in coding order, each with its own frame's
    /// timestamp; together they decode to every frame in display order.
    #[test]
    fn b_pictures_come_out_reordered_and_stamped() {
        let (w, h) = (64, 48);
        let config = EncoderConfig {
            width: w,
            height: h,
            frame_rate: 25.0,
            codec: VideoCodec::Mpeg2,
            keyframe_interval: 6,
            ..Default::default()
        };
        let mut enc = Mpeg2Encoder::new(config).unwrap();
        let mut packets = Vec::new();
        for n in 0..8 {
            enc.send_frame(&super::super::native::test_picture(w, h, n)).unwrap();
            while let Some(p) = enc.receive_packet().unwrap() {
                packets.push(p);
            }
        }
        enc.flush().unwrap();
        while let Some(p) = enc.receive_packet().unwrap() {
            packets.push(p);
        }
        let pts: Vec<u64> = packets.iter().map(|p| p.pts).collect();
        // I0, P3 B1 B2, then the GOP at 6: I6 B4 B5, then P7 at the flush.
        assert_eq!(pts, vec![0, 3, 1, 2, 6, 4, 5, 7]);
        let keys: Vec<bool> = packets.iter().map(|p| p.is_keyframe).collect();
        assert_eq!(keys, vec![true, false, false, false, true, false, false, false]);
        let mut dec = mpeg2::Decoder::new();
        let mut frames = Vec::new();
        for p in &packets {
            frames.extend(dec.decode(&p.data).unwrap());
        }
        frames.extend(dec.flush().unwrap());
        assert_eq!(frames.len(), 8);
    }

    #[test]
    fn a_rate_beside_a_buffer_is_refused_and_a_plain_rate_taken() {
        let base = EncoderConfig { width: 64, height: 48, frame_rate: 25.0, codec: VideoCodec::Mpeg2, ..Default::default() };
        let mut rate = base.clone();
        rate.overrides.bitrate = Some(2_000_000);
        assert!(Mpeg2Encoder::new(rate.clone()).is_ok());
        rate.overrides.buffer_ms = Some(1000);
        assert!(Mpeg2Encoder::new(rate).err().expect("refused").to_string().contains("buffer"));
        let odd = EncoderConfig { frame_rate: 17.3, ..base };
        assert!(Mpeg2Encoder::new(odd).is_err());
    }
}
