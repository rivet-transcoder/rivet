//! VP9 encode — this workspace's own encoder (`crates/vp9`, the rivet-vp9
//! repository), written clean-room from the VP9 bitstream specification.
//!
//! The only VP9 encoder in the tree (no hardware backend here is wired for
//! VP9 encode), so [`select_encoder`](super::select_encoder) builds it
//! directly for a VP9 job; see [`native`](super::native).
//!
//! # What it takes and writes
//!
//! Profile 0 — 8-bit 4:2:0, the profile every browser decodes. The crate's
//! encoder writes no other profile yet (its README lists what profiles 1-3
//! need), so a 10-bit or HDR VP9 rung is refused by
//! [`backend_output_caps_for`](super::backend_output_caps_for) before a
//! frame is decoded. One packet per frame, every frame shown and in display
//! order, so each packet carries its frame's timestamp. Key frames at the
//! interval and on [`force_keyframe_next`](Encoder::force_keyframe_next);
//! inter frames predict from the previous frame only.
//!
//! # Rate and quality
//!
//! A fixed `base_q_idx` (1-255): a CRF on libvpx's 0-63 `cq-level` scale
//! times four, else the quality target's
//! ([`tuning::native_sw_quantizer`](super::tuning::native_sw_quantizer));
//! never 0, which is VP9's lossless mode. No rate control: a bitrate rung is
//! refused by name. The speed tier picks the fixed partition's block size
//! (Draft 32x32, else 16x16) and the motion search range.
//!
//! # Colour
//!
//! The uncompressed header's `color_space` (BT.601 / BT.709 / BT.2020 /
//! SMPTE 170 / 240, from the matrix code) and `color_range` are written from
//! `color_metadata`; primaries and transfer are the container's to carry
//! (`vpcC`, `colr`, the WebM `Colour` element).

use std::collections::VecDeque;

use anyhow::{Context, Result, bail};
use bytes::Bytes;

use super::native::{check_frame, quantizer, refuse_any_rate, tier};
use super::tuning::SpeedTier;
use super::{EncodedPacket, Encoder, EncoderConfig};
use crate::frame::{ColorMetadata, PixelFormat, VideoCodec, VideoFrame};

/// A VP9 encoder behind rivet's [`Encoder`] trait.
pub struct Vp9Encoder {
    inner: vp9::Encoder,
    cfg: vp9::Config,
    ready: VecDeque<EncodedPacket>,
}

/// VP9's `color_space` for an H.273 matrix code.
fn color_space(c: &ColorMetadata) -> vp9::ColorSpace {
    match c.matrix_coefficients {
        1 => vp9::ColorSpace::Bt709,
        5 => vp9::ColorSpace::Bt601,
        6 => vp9::ColorSpace::Smpte170,
        7 => vp9::ColorSpace::Smpte240,
        9 | 10 => vp9::ColorSpace::Bt2020,
        _ => vp9::ColorSpace::Unknown,
    }
}

impl Vp9Encoder {
    /// Build an encoder for `config` (codec VP9, `yuv420p`).
    pub fn new(config: EncoderConfig) -> Result<Self> {
        if config.codec != VideoCodec::Vp9 {
            bail!("the VP9 encoder encodes VP9, not {}", config.codec.label());
        }
        if config.pixel_format != PixelFormat::Yuv420p {
            bail!(
                "the VP9 encoder writes profile 0 (8-bit 4:2:0) only; this rung asks for {:?}. Encode at \
                 8 bits (--pixel-format 8bit)",
                config.pixel_format
            );
        }
        refuse_any_rate("VP9", &config)?;
        let mut cfg = vp9::Config::new(config.width, config.height);
        cfg.quantizer = quantizer(&config);
        cfg.keyframe_interval = config.keyframe_interval.max(1);
        let speed = tier(&config);
        cfg.block_size = if speed == SpeedTier::Draft { 32 } else { 16 };
        cfg.search_range = match speed {
            SpeedTier::Draft => 8,
            SpeedTier::Standard => 16,
            SpeedTier::Archive => 32,
        };
        cfg.color_space = color_space(&config.color_metadata);
        cfg.full_range = config.color_metadata.full_range;
        cfg.validate().context("the VP9 encoder rejected the configuration")?;
        Ok(Self { inner: vp9::Encoder::new(cfg.clone()), cfg, ready: VecDeque::new() })
    }
}

impl Encoder for Vp9Encoder {
    fn send_frame(&mut self, frame: &VideoFrame) -> Result<()> {
        let want = check_frame("VP9", frame, self.cfg.width, self.cfg.height, &[PixelFormat::Yuv420p])?;
        let mut picture = vp9::Frame::new(self.cfg.width, self.cfg.height, 8, vp9::ChromaFormat::Yuv420);
        picture.data.copy_from_slice(&frame.data[..want]);
        picture.color_space = self.cfg.color_space;
        picture.full_range = self.cfg.full_range;
        let data = self.inner.encode(&picture).context("the VP9 encoder refused a frame")?;
        // frame_marker (2) profile (2) show_existing_frame (1) frame_type (1):
        // frame_type 0 is a key frame (profile 0, no reserved bit).
        let is_keyframe = data.first().is_some_and(|b| b & 0x04 == 0);
        self.ready.push_back(EncodedPacket { data: Bytes::from(data), pts: frame.pts, is_keyframe });
        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        Ok(())
    }

    fn receive_packet(&mut self) -> Result<Option<EncodedPacket>> {
        Ok(self.ready.pop_front())
    }

    fn force_keyframe_next(&mut self) -> Result<()> {
        self.inner.force_keyframe();
        Ok(())
    }

    /// Rebuild the encoder: no references, the next frame a key frame.
    fn reset(&mut self) -> Result<()> {
        self.inner = vp9::Encoder::new(self.cfg.clone());
        self.ready.clear();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every packet decodes, key frames where the interval and a forced one
    /// put them, timestamps unchanged, and the picture is near the source.
    #[test]
    fn packets_decode_near_the_source() {
        let (w, h) = (64, 48);
        let config = EncoderConfig { width: w, height: h, codec: VideoCodec::Vp9, keyframe_interval: 3, ..Default::default() };
        let mut enc = Vp9Encoder::new(config).unwrap();
        let mut dec = vp9::Decoder::new();
        for n in 0..4 {
            let src = super::super::native::test_picture(w, h, n);
            enc.send_frame(&src).unwrap();
            let p = enc.receive_packet().unwrap().expect("a packet per frame");
            assert_eq!((p.pts, p.is_keyframe), (n, n % 3 == 0), "frame {n}");
            let shown = dec.decode(&p.data).unwrap().expect("a shown frame");
            let luma = (w * h) as usize;
            let mse: f64 = shown.data[..luma]
                .iter()
                .zip(&src.data[..luma])
                .map(|(a, b)| (f64::from(*a) - f64::from(*b)).powi(2))
                .sum::<f64>()
                / luma as f64;
            let psnr = 10.0 * (255.0f64 * 255.0 / mse.max(1e-9)).log10();
            assert!(psnr > 30.0, "frame {n}: {psnr:.1} dB");
        }
        enc.force_keyframe_next().unwrap();
        enc.send_frame(&super::super::native::test_picture(w, h, 4)).unwrap();
        assert!(enc.receive_packet().unwrap().unwrap().is_keyframe);
    }

    #[test]
    fn ten_bit_is_refused() {
        let ten = EncoderConfig {
            width: 64,
            height: 48,
            codec: VideoCodec::Vp9,
            pixel_format: PixelFormat::Yuv420p10le,
            ..Default::default()
        };
        assert!(Vp9Encoder::new(ten).err().expect("refused").to_string().contains("profile 0"));
    }
}
