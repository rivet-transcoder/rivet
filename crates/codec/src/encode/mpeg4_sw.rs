//! MPEG-4 Part 2 Visual encode — this workspace's own encoder
//! (`crates/mpeg4`, the rivet-mpeg4 repository), written clean-room from
//! ISO/IEC 14496-2.
//!
//! The only MPEG-4 Part 2 encoder in the tree, built directly by
//! [`select_encoder`](super::select_encoder) for an MPEG-4 job; see
//! [`native`](super::native).
//!
//! # What it takes and writes
//!
//! 8-bit 4:2:0. Simple Profile (I- and P-VOPs) by default — what every
//! MPEG-4 Part 2 player decodes — and Advanced Simple Profile B-VOPs when the
//! rung asks for B frames (`overrides.bframes`, 1-8). The first packet opens
//! with the configuration headers (visual object sequence, visual object,
//! video object layer), which the MP4 muxer also copies into the `esds`.
//! Quarter-sample motion, GMC, interlace and the MPEG quantiser are not in
//! the crate's encoder (its decoder takes them all).
//!
//! # Order
//!
//! With B-VOPs the encoder codes each reference VOP before the B-VOPs that
//! precede it; its output is split here into one packet per VOP, each
//! stamped with its own frame's timestamp ([`ReferenceFirst`]).
//!
//! # Rate and quality
//!
//! A constant `vop_quant` (1-31; B-VOPs a quarter coarser): the rung's CRF,
//! else the quality target's ([`tuning::native_sw_quantizer`](super::tuning::native_sw_quantizer)).
//! A bitrate rung is coded to its rate by the encoder's own per-VOP
//! controller; a constant rate or a coded picture buffer is refused by name.
//! The `Archive` tier adds four-vector macroblocks.
//!
//! # Colour and key frames
//!
//! The visual object header's `video_signal_type()` carries the range and
//! the H.273 colour description (primaries, transfer, matrix) from
//! `color_metadata`, so a decoder that reads it — and the MP4 / QuickTime
//! muxers' `colr`, which say the same — agree on what the samples mean.
//! [`force_keyframe_next`](Encoder::force_keyframe_next) makes the next frame
//! an I-VOP (the chunked path's seam): B-VOPs still waiting are coded after
//! it, predicted from it.

use std::collections::VecDeque;

use anyhow::{Context, Result, bail};
use bytes::Bytes;

use super::native::{ReferenceFirst, average_rate, check_frame, frame_rate_ratio, quantizer, tier};
use super::tuning::SpeedTier;
use super::{EncodedPacket, Encoder, EncoderConfig};
use crate::frame::{ColorMetadata, PixelFormat, TransferFn, VideoCodec, VideoFrame};

/// The visual object header's `video_signal_type()` for `c`: the range and
/// the three H.273 codes (`video_format` 5, unspecified — the analogue
/// system a camera fed is not something the pipeline knows).
fn video_signal(c: &ColorMetadata) -> mpeg4::VideoSignal {
    let transfer = match c.transfer {
        TransferFn::Bt709 | TransferFn::Unspecified => 1,
        TransferFn::Bt470Bg => 4,
        TransferFn::Linear => 8,
        TransferFn::St2084 => 16,
        TransferFn::AribStdB67 => 18,
    };
    mpeg4::VideoSignal {
        video_format: 5,
        full_range: c.full_range,
        colour: Some(mpeg4::ColourDescription {
            colour_primaries: c.colour_primaries,
            transfer_characteristics: transfer,
            matrix_coefficients: c.matrix_coefficients,
        }),
    }
}

/// An MPEG-4 Part 2 encoder behind rivet's [`Encoder`] trait.
pub struct Mpeg4Encoder {
    inner: mpeg4::Encoder,
    cfg: mpeg4::EncoderConfig,
    order: ReferenceFirst,
    ready: VecDeque<EncodedPacket>,
}

impl Mpeg4Encoder {
    /// Build an encoder for `config` (codec MPEG-4, `yuv420p`).
    pub fn new(config: EncoderConfig) -> Result<Self> {
        if config.codec != VideoCodec::Mpeg4 {
            bail!("the MPEG-4 Part 2 encoder encodes MPEG-4 Part 2, not {}", config.codec.label());
        }
        if config.pixel_format != PixelFormat::Yuv420p {
            bail!(
                "the MPEG-4 Part 2 encoder writes 8-bit 4:2:0 only; this rung asks for {:?}",
                config.pixel_format
            );
        }
        let rate = match average_rate("MPEG-4 Part 2", &config)? {
            Some(bps) => mpeg4::RateControl::Bitrate(bps),
            None => mpeg4::RateControl::ConstantQuant(quantizer(&config)),
        };
        // vop_time_increment_resolution is 16 bits: an NTSC rate keeps its
        // 1001 duration where the resolution fits, else the rounded rate.
        let (mut num, mut den) = frame_rate_ratio(config.frame_rate);
        if num > 65_535 {
            (num, den) = (config.frame_rate.round().max(1.0) as u32, 1);
        }
        let mut cfg = mpeg4::EncoderConfig::new(config.width, config.height, num);
        cfg.frame_duration = den;
        cfg.gop_size = config.keyframe_interval.max(1);
        cfg.b_frames = u32::from(config.overrides.bframes.unwrap_or(0));
        cfg.rate = rate;
        let speed = tier(&config);
        cfg.search_range = match speed {
            SpeedTier::Draft => 8,
            SpeedTier::Standard => 15,
            SpeedTier::Archive => 31,
        };
        cfg.four_mv = speed == SpeedTier::Archive;
        cfg.video_signal = Some(video_signal(&config.color_metadata));
        let inner = mpeg4::Encoder::new(cfg.clone()).context("the MPEG-4 Part 2 encoder rejected the configuration")?;
        Ok(Self { inner, cfg, order: ReferenceFirst::default(), ready: VecDeque::new() })
    }

    /// Split what the encoder returned into one packet per VOP, stamped.
    fn collect(&mut self, bytes: Vec<u8>) -> Result<()> {
        let units = split_vops(&bytes);
        let stamps = self.order.take(units.len())?;
        for (unit, pts) in units.into_iter().zip(stamps) {
            let is_keyframe = vop_type(&unit) == Some(0);
            self.ready.push_back(EncodedPacket { data: Bytes::from(unit), pts, is_keyframe });
        }
        Ok(())
    }
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

/// One access unit per VOP (`vop_start_code` 0xB6): the headers before a
/// VOP — the configuration on the first, a GOV — ride with it.
fn split_vops(data: &[u8]) -> Vec<Vec<u8>> {
    let mut starts = Vec::new();
    let mut lead: Option<usize> = None;
    for (o, c) in start_codes(data) {
        if c == 0xb6 {
            starts.push(lead.take().unwrap_or(o));
        } else if c != 0xb2 {
            lead.get_or_insert(o);
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

/// The `vop_coding_type` of the VOP in `unit` (0 I, 1 P, 2 B).
fn vop_type(unit: &[u8]) -> Option<u8> {
    let (o, _) = start_codes(unit).into_iter().find(|(_, c)| *c == 0xb6)?;
    unit.get(o + 4).map(|b| b >> 6)
}

impl Encoder for Mpeg4Encoder {
    fn send_frame(&mut self, frame: &VideoFrame) -> Result<()> {
        let want = check_frame("MPEG-4 Part 2", frame, self.cfg.width, self.cfg.height, &[PixelFormat::Yuv420p])?;
        let mut picture = mpeg4::Frame::new(self.cfg.width, self.cfg.height);
        picture.data.copy_from_slice(&frame.data[..want]);
        self.order.push(frame.pts);
        let bytes = self.inner.encode(&picture).context("the MPEG-4 Part 2 encoder refused a frame")?;
        self.collect(bytes)
    }

    fn flush(&mut self) -> Result<()> {
        let bytes = self.inner.finish().context("the MPEG-4 Part 2 encoder failed to finish")?;
        self.collect(bytes)
    }

    fn receive_packet(&mut self) -> Result<Option<EncodedPacket>> {
        Ok(self.ready.pop_front())
    }

    fn force_keyframe_next(&mut self) -> Result<()> {
        self.inner.force_keyframe();
        Ok(())
    }

    /// Rebuild the encoder: the next frame is an I-VOP behind fresh
    /// configuration headers.
    fn reset(&mut self) -> Result<()> {
        self.inner = mpeg4::Encoder::new(self.cfg.clone()).context("rebuilding the MPEG-4 Part 2 encoder")?;
        self.order.clear();
        self.ready.clear();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encode_all(config: EncoderConfig, frames: u64) -> (Mpeg4Encoder, Vec<EncodedPacket>) {
        let (w, h) = (config.width, config.height);
        let mut enc = Mpeg4Encoder::new(config).unwrap();
        let mut packets = Vec::new();
        for n in 0..frames {
            enc.send_frame(&super::super::native::test_picture(w, h, n)).unwrap();
            while let Some(p) = enc.receive_packet().unwrap() {
                packets.push(p);
            }
        }
        enc.flush().unwrap();
        while let Some(p) = enc.receive_packet().unwrap() {
            packets.push(p);
        }
        (enc, packets)
    }

    /// Simple Profile: one VOP per frame, in order, the first an I-VOP behind
    /// the configuration; the decoder reproduces the encoder's
    /// reconstruction.
    #[test]
    fn simple_profile_vops_decode_to_the_reconstruction() {
        let config = EncoderConfig { width: 64, height: 48, frame_rate: 25.0, codec: VideoCodec::Mpeg4, ..Default::default() };
        let mut cfg = Mpeg4Encoder::new(config.clone()).unwrap().cfg;
        cfg.keep_reconstructions = true;
        let (mut enc, packets) = encode_all(config, 4);
        assert_eq!(packets.iter().map(|p| p.pts).collect::<Vec<_>>(), vec![0, 1, 2, 3]);
        assert!(packets[0].is_keyframe && !packets[1].is_keyframe);
        // The reconstruction check runs on an encoder that keeps them.
        enc.inner = mpeg4::Encoder::new(cfg).unwrap();
        let mut dec = mpeg4::Decoder::new();
        let mut recon = Vec::new();
        let mut decoded = Vec::new();
        for n in 0..4 {
            let mut picture = mpeg4::Frame::new(64, 48);
            picture.data.copy_from_slice(&super::super::native::test_picture(64, 48, n).data);
            let bytes = enc.inner.encode(&picture).unwrap();
            recon.extend(enc.inner.take_reconstructions());
            decoded.extend(dec.decode(&bytes).unwrap());
        }
        decoded.extend(dec.flush());
        assert_eq!(decoded.len(), 4);
        for (d, r) in decoded.iter().zip(&recon) {
            assert_eq!(d.data, r.data);
        }
    }

    /// Advanced Simple: B-VOPs come out after their reference, each with its
    /// own frame's timestamp.
    #[test]
    fn b_vops_are_stamped_with_their_own_frames() {
        let mut config = EncoderConfig { width: 64, height: 48, frame_rate: 25.0, codec: VideoCodec::Mpeg4, ..Default::default() };
        config.overrides.bframes = Some(2);
        let (_, packets) = encode_all(config, 7);
        let pts: Vec<u64> = packets.iter().map(|p| p.pts).collect();
        assert_eq!(pts, vec![0, 3, 1, 2, 6, 4, 5]);
        let mut dec = mpeg4::Decoder::new();
        let mut frames = Vec::new();
        for p in &packets {
            frames.extend(dec.decode(&p.data).unwrap());
        }
        frames.extend(dec.flush());
        assert_eq!(frames.len(), 7);
    }

    /// The colour description reaches the visual object header, and a forced
    /// key frame lands on the next frame.
    #[test]
    fn colour_is_signalled_and_a_forced_key_lands() {
        let color = ColorMetadata { colour_primaries: 9, matrix_coefficients: 9, transfer: TransferFn::AribStdB67, full_range: true, ..Default::default() };
        let config = EncoderConfig {
            width: 64,
            height: 48,
            frame_rate: 25.0,
            keyframe_interval: 100,
            codec: VideoCodec::Mpeg4,
            color_metadata: color,
            ..Default::default()
        };
        let mut enc = Mpeg4Encoder::new(config).unwrap();
        let mut packets = Vec::new();
        for n in 0..4 {
            if n == 2 {
                enc.force_keyframe_next().unwrap();
            }
            enc.send_frame(&super::super::native::test_picture(64, 48, n)).unwrap();
            while let Some(p) = enc.receive_packet().unwrap() {
                packets.push(p);
            }
        }
        enc.flush().unwrap();
        while let Some(p) = enc.receive_packet().unwrap() {
            packets.push(p);
        }
        let keys: Vec<bool> = packets.iter().map(|p| p.is_keyframe).collect();
        assert_eq!(keys, [true, false, true, false]);
        let mut dec = mpeg4::Decoder::new();
        dec.decode(&packets[0].data).unwrap();
        let signal = dec.vol().and_then(|v| v.video_signal).expect("a video_signal_type()");
        assert!(signal.full_range);
        let c = signal.colour.expect("a colour description");
        assert_eq!((c.colour_primaries, c.transfer_characteristics, c.matrix_coefficients), (9, 18, 9));
    }
}
