//! VP9 encode — this workspace's own encoder (`crates/vp9`, the rivet-vp9
//! repository), written clean-room from the VP9 bitstream specification.
//!
//! The only VP9 encoder in the tree (no hardware backend here is wired for
//! VP9 encode), so [`select_encoder`](super::select_encoder) builds it
//! directly for a VP9 job; see [`native`](super::native).
//!
//! # What it takes and writes
//!
//! Every profile: 8-bit 4:2:0 (profile 0, what every browser decodes),
//! 8-bit 4:2:2 / 4:4:4 (profile 1), 10- and 12-bit 4:2:0 (profile 2) and
//! 10- and 12-bit 4:2:2 / 4:4:4 (profile 3) — the profile follows from the
//! rung's pixel format. The pipeline itself encodes 4:2:0 at 8 or 10 bits
//! (`yuv420p`, `yuv420p10le`: profiles 0 and 2); the other formats are there
//! for a caller that hands this encoder its own frames. One packet per frame,
//! every frame shown and in display order, so each packet carries its
//! frame's timestamp. Key frames at the interval and on
//! [`force_keyframe_next`](Encoder::force_keyframe_next); inter frames
//! predict from LAST or GOLDEN.
//!
//! # Rate and quality
//!
//! A fixed `base_q_idx` (1-255): a CRF on libvpx's 0-63 `cq-level` scale
//! times four, else the quality target's
//! ([`tuning::native_sw_quantizer`](super::tuning::native_sw_quantizer));
//! never 0, which is VP9's lossless mode. A bitrate rung is coded to its
//! average rate by the encoder's one-pass rate controller (a budget per frame,
//! frames recoded when they miss it by more than 12%); a constant rate or a
//! coded picture buffer is refused by name.
//!
//! # Speed
//!
//! The crate's encoder searches partitions and transform sizes by real rate
//! and distortion at its speeds 0 and 1, which costs: single-threaded, about
//! 1.7 frames/s at 352x288 at speed 1, 10 at speed 2 (a fixed partition).
//! The speed tier (`encode-policy` `speed=`, or `video-speed`) picks:
//!
//! | tier | crate speed | partition | motion search |
//! |---|---|---|---|
//! | `draft` | 2 | fixed 32x32 | ±8 |
//! | `standard` (default) | 2 | fixed 16x16 | ±16 |
//! | `archive` | 1 | searched, NONE / SPLIT, two transform sizes | ±32 |
//!
//! GOLDEN (every 8 frames, coded finer) is on at every tier: it is nearly
//! free and worth 7-12% at the same quality.
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

use super::native::{average_rate, quantizer, tier};
use super::tuning::SpeedTier;
use super::{EncodedPacket, Encoder, EncoderConfig};
use crate::frame::{ColorMetadata, PixelFormat, VideoCodec, VideoFrame};

/// A VP9 encoder behind rivet's [`Encoder`] trait.
pub struct Vp9Encoder {
    inner: vp9::Encoder,
    cfg: vp9::Config,
    format: PixelFormat,
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

/// The bit depth and chroma format of a pixel format VP9 codes, or `None`.
pub fn vp9_layout(format: PixelFormat) -> Option<(u32, vp9::ChromaFormat)> {
    use vp9::ChromaFormat as C;
    Some(match format {
        PixelFormat::Yuv420p => (8, C::Yuv420),
        PixelFormat::Yuv420p10le => (10, C::Yuv420),
        PixelFormat::Yuv420p12le => (12, C::Yuv420),
        PixelFormat::Yuv422p => (8, C::Yuv422),
        PixelFormat::Yuv422p10le => (10, C::Yuv422),
        PixelFormat::Yuv422p12le => (12, C::Yuv422),
        PixelFormat::Yuv444p => (8, C::Yuv444),
        PixelFormat::Yuv444p10le => (10, C::Yuv444),
        PixelFormat::Yuv444p12le => (12, C::Yuv444),
        _ => return None,
    })
}

impl Vp9Encoder {
    /// Build an encoder for `config` (codec VP9; any planar 4:2:0, 4:2:2 or
    /// 4:4:4 format at 8, 10 or 12 bits).
    pub fn new(config: EncoderConfig) -> Result<Self> {
        if config.codec != VideoCodec::Vp9 {
            bail!("the VP9 encoder encodes VP9, not {}", config.codec.label());
        }
        let Some((bit_depth, chroma)) = vp9_layout(config.pixel_format) else {
            bail!(
                "the VP9 encoder codes planar 4:2:0, 4:2:2 or 4:4:4 at 8, 10 or 12 bits (profiles 0-3); this \
                 rung asks for {:?}",
                config.pixel_format
            );
        };
        let rate = average_rate("VP9", &config)?;
        let mut cfg = vp9::Config::new(config.width, config.height);
        cfg.bit_depth = bit_depth;
        cfg.chroma = chroma;
        cfg.quantizer = quantizer(&config);
        cfg.keyframe_interval = config.keyframe_interval.max(1);
        if let Some(bps) = rate {
            cfg.target_bitrate = Some(u64::from(bps));
            cfg.frame_rate = if config.frame_rate.is_finite() && config.frame_rate > 0.0 { config.frame_rate } else { 30.0 };
        }
        match tier(&config) {
            SpeedTier::Draft => {
                cfg.speed = 2;
                cfg.block_size = 32;
                cfg.search_range = 8;
            }
            SpeedTier::Standard => {
                cfg.speed = 2;
                cfg.block_size = 16;
                cfg.search_range = 16;
            }
            SpeedTier::Archive => {
                cfg.speed = 1;
                cfg.search_range = 32;
            }
        }
        cfg.color_space = color_space(&config.color_metadata);
        cfg.full_range = config.color_metadata.full_range;
        cfg.validate().context("the VP9 encoder rejected the configuration")?;
        Ok(Self { inner: vp9::Encoder::new(cfg.clone()), cfg, format: config.pixel_format, ready: VecDeque::new() })
    }
}

impl Encoder for Vp9Encoder {
    fn send_frame(&mut self, frame: &VideoFrame) -> Result<()> {
        if frame.format != self.format {
            bail!(
                "the VP9 encoder was configured for {:?} frames and got a {:?} one. Convert with the colorspace \
                 filter before the encoder.",
                self.format,
                frame.format
            );
        }
        if frame.width != self.cfg.width || frame.height != self.cfg.height {
            bail!(
                "frame is {}x{} but the VP9 encoder was configured for {}x{}",
                frame.width,
                frame.height,
                self.cfg.width,
                self.cfg.height
            );
        }
        let mut picture = vp9::Frame::new(self.cfg.width, self.cfg.height, self.cfg.bit_depth, self.cfg.chroma);
        let want = picture.data.len();
        if frame.data.len() < want {
            bail!("frame buffer is {} bytes, too short for a {:?} frame ({want} expected)", frame.data.len(), frame.format);
        }
        picture.data.copy_from_slice(&frame.data[..want]);
        picture.color_space = self.cfg.color_space;
        picture.full_range = self.cfg.full_range;
        let data = self.inner.encode(&picture).context("the VP9 encoder refused a frame")?;
        let is_keyframe = self.inner.last_was_keyframe();
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
    use crate::encode::tuning::{EncodeOverrides, RateMode};

    fn psnr8(a: &[u8], b: &[u8]) -> f64 {
        let mse = a.iter().zip(b).map(|(x, y)| (f64::from(*x) - f64::from(*y)).powi(2)).sum::<f64>() / a.len() as f64;
        10.0 * (255.0f64 * 255.0 / mse.max(1e-9)).log10()
    }

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
            let psnr = psnr8(&shown.data[..luma], &src.data[..luma]);
            assert!(psnr > 30.0, "frame {n}: {psnr:.1} dB");
        }
        enc.force_keyframe_next().unwrap();
        enc.send_frame(&super::super::native::test_picture(w, h, 4)).unwrap();
        assert!(enc.receive_packet().unwrap().unwrap().is_keyframe);
    }

    /// 10-bit 4:2:0 (profile 2) and 8-bit 4:4:4 (profile 1) frames are coded
    /// in their own profile and decode at their own depth and layout.
    #[test]
    fn high_bit_depth_and_full_chroma_are_coded() {
        let (w, h) = (48, 32);
        for (format, profile) in [
            (PixelFormat::Yuv420p10le, 2),
            (PixelFormat::Yuv444p, 1),
            (PixelFormat::Yuv422p12le, 3),
        ] {
            let (depth, chroma) = vp9_layout(format).unwrap();
            let config = EncoderConfig { width: w, height: h, codec: VideoCodec::Vp9, pixel_format: format, ..Default::default() };
            let mut enc = Vp9Encoder::new(config).unwrap();
            assert_eq!(enc.cfg.profile(), profile, "{format:?}");
            let mut src = vp9::Frame::new(w, h, depth, chroma);
            for p in 0..3 {
                let pl = src.planes[p];
                for y in 0..pl.height {
                    for x in 0..pl.width {
                        let v = ((x * 5 + y * 3 + p as u32 * 30) % 200 + 20) << (depth - 8);
                        let at = pl.offset + ((y * pl.width + x) as usize) * if depth > 8 { 2 } else { 1 };
                        if depth > 8 {
                            src.data[at..at + 2].copy_from_slice(&(v as u16).to_le_bytes());
                        } else {
                            src.data[at] = v as u8;
                        }
                    }
                }
            }
            let frame = VideoFrame::new(Bytes::from(src.data.clone()), w, h, format, crate::frame::ColorSpace::Bt709, 0);
            enc.send_frame(&frame).unwrap();
            let p = enc.receive_packet().unwrap().unwrap();
            let back = vp9::Decoder::new().decode(&p.data).unwrap().unwrap();
            assert_eq!((back.bit_depth, back.chroma), (depth, chroma), "{format:?}");
            assert_eq!(back.data.len(), src.data.len());
        }
    }

    #[test]
    fn a_bitrate_rung_is_coded_and_a_constant_rate_refused() {
        let base = EncoderConfig { width: 64, height: 48, codec: VideoCodec::Vp9, frame_rate: 25.0, ..Default::default() };
        let vbr = EncodeOverrides { bitrate: Some(300_000), ..Default::default() };
        let enc = Vp9Encoder::new(EncoderConfig { overrides: vbr, ..base.clone() }).unwrap();
        assert_eq!(enc.cfg.target_bitrate, Some(300_000));
        let cbr = EncodeOverrides { rate_mode: Some(RateMode::Constant), bitrate: Some(300_000), ..Default::default() };
        let msg = Vp9Encoder::new(EncoderConfig { overrides: cbr, ..base }).err().unwrap().to_string();
        assert!(msg.contains("constant"), "{msg}");
    }

    #[test]
    fn the_tier_picks_the_speed() {
        let speed = |t| {
            let o = EncodeOverrides { speed_tier: Some(t), ..Default::default() };
            let cfg = EncoderConfig { width: 64, height: 48, codec: VideoCodec::Vp9, overrides: o, ..Default::default() };
            let e = Vp9Encoder::new(cfg).unwrap();
            (e.cfg.speed, e.cfg.block_size)
        };
        assert_eq!(speed(SpeedTier::Draft), (2, 32));
        assert_eq!(speed(SpeedTier::Standard), (2, 16));
        assert_eq!(speed(SpeedTier::Archive).0, 1);
    }
}
