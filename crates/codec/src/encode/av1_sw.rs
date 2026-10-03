//! AV1 encode in software — this workspace's own encoder (`crates/av1`, the
//! rivet-av1 repository), written clean-room from the AV1 specification.
//!
//! Every other AV1 encoder in this crate needs silicon: NVENC wants Ada or
//! newer, AMF wants RDNA3, QSV wants Arc or Meteor Lake. This tier exists so
//! a host with none of them — a laptop, a CI runner, a container on a cloud
//! instance with no GPU attached — produces a file instead of a diagnostic.
//! (It replaced rav1e on 2026-10-03.)
//!
//! # Always built; the feature decides whether it is *reached*
//!
//! This module compiles unconditionally, so it is always testable and a caller
//! can always ask for it by name (`EncoderBackend::Av1`, `--encoder av1`). The
//! `av1-sw-fallback` feature gates whether
//! [`select_encoder`](super::select_encoder) **falls back** here on its own
//! once every hardware backend has declined: a fleet that quietly degraded
//! into a CPU encoder would look like a capacity problem rather than the
//! missing driver it is, while a workstation wants exactly the fallback.
//! When it engages it says so at `warn`.
//!
//! # What it takes and writes
//!
//! Profile 0: 8- or 10-bit 4:2:0 (`Yuv420p`, `Yuv420p10le`), one tile, one
//! temporal unit per frame, every frame shown and in display order — so each
//! packet carries its own frame's timestamp. Up to 4096 pixels wide (one
//! tile). Key frames at the interval; inter frames predict from the previous
//! frame. The colour description is not written into the sequence header
//! (the crate's encoder has no setting for it): the container's `colr`
//! carries it, so the tier reports no HDR.
//!
//! `force_keyframe_next` starts a fresh encoder for the next frame — the
//! crate has no forced-key call, and a key frame resets every reference
//! anyway — carrying the rate controller's quantiser over.
//!
//! # Rate and quality
//!
//! The quantiser (`base_q_idx`, 1-255) is the rung's CRF on AV1's 0-63 scale
//! times four, else the quality target's ([`tuning::av1_sw_params_with`]:
//! four times libaom's `cq-level`, the same table the hardware tiers are
//! equalised against). A bitrate rung is coded to its average rate by the
//! encoder's rate controller (bits per frame at the rung's frame rate); a
//! constant rate or a coded picture buffer is refused by name. The speed tier
//! picks the motion search range.
//!
//! # Speed
//!
//! Single-threaded scalar Rust: about 10 frames/s at 352x288 and 2 frames/s
//! (1.9 MP/s) at 1280x720 (`throughput_at_720p`), so about a second a frame
//! at 1080p. A fallback, not a production encoder; the GPU tiers come first.

use anyhow::{Context, Result, bail};
use bytes::Bytes;

use super::native::{average_rate, check_frame};
use super::tuning;
use super::{AUTO_FROM_TARGET, EncodedPacket, Encoder, EncoderConfig};
use crate::frame::{PixelFormat, VideoCodec, VideoFrame};

/// The widest frame the encoder codes: one tile, and AV1's widest tile.
pub const MAX_WIDTH: u32 = 4096;

/// rivet's software AV1 encoder.
pub struct Av1Encoder {
    inner: av1::Encoder,
    cfg: av1::Config,
    /// Frames into the current `inner`: the crate codes a key frame when
    /// this is a multiple of the interval.
    frames: u64,
    force_key: bool,
    ready: std::collections::VecDeque<EncodedPacket>,
}

impl Av1Encoder {
    /// Build an encoder for `config` (codec AV1, `yuv420p` or `yuv420p10le`).
    pub fn new(config: EncoderConfig) -> Result<Self> {
        if config.codec != VideoCodec::Av1 {
            bail!("the AV1 encoder encodes AV1, not {}", config.codec.label());
        }
        let bit_depth = match config.pixel_format {
            PixelFormat::Yuv420p => 8,
            PixelFormat::Yuv420p10le => 10,
            other => bail!(
                "the software AV1 encoder writes profile 0, 8- or 10-bit 4:2:0 (yuv420p, yuv420p10le); the encoder \
                 was configured for {other:?}"
            ),
        };
        if config.width == 0 || config.height == 0 {
            bail!("the software AV1 encoder needs a frame size, got {}x{}", config.width, config.height);
        }
        if config.width > MAX_WIDTH {
            bail!(
                "the software AV1 encoder codes one tile, at most {MAX_WIDTH} pixels wide; this rung is {}x{}. \
                 Encode it on a GPU (NVENC, AMF or QSV), or scale it down",
                config.width,
                config.height
            );
        }
        let rate = average_rate("software AV1", &config)?;
        let rung = tuning::RungContext::standalone(config.width, config.height);
        let p = tuning::av1_sw_params_with(config.target, config.tier, &rung, &config.overrides);
        let quantizer = if config.quality != AUTO_FROM_TARGET {
            (u32::from(config.quality) * 4).clamp(1, 255)
        } else {
            p.quantizer
        };

        let mut cfg = av1::Config::new(config.width, config.height);
        cfg.bit_depth = bit_depth;
        cfg.quantizer = quantizer;
        cfg.keyframe_interval = if config.keyframe_interval == 0 { 240 } else { config.keyframe_interval };
        cfg.search_range = p.search_range;
        if let Some(bps) = rate {
            let fps = if config.frame_rate.is_finite() && config.frame_rate > 0.0 { config.frame_rate } else { 30.0 };
            cfg.target_bits_per_frame = Some(((f64::from(bps) / fps).round() as u64).max(1));
        }

        tracing::warn!(
            width = config.width,
            height = config.height,
            quantizer,
            bit_depth,
            bitrate = ?rate,
            "no AV1 encode silicon available or asked for — encoding with rivet's own software AV1 encoder, \
             which is far slower than any hardware backend"
        );
        Ok(Self {
            inner: av1::Encoder::new(cfg.clone()),
            cfg,
            frames: 0,
            force_key: false,
            ready: Default::default(),
        })
    }

    /// The settings the encoder was built with.
    pub fn config(&self) -> &av1::Config {
        &self.cfg
    }
}

impl Encoder for Av1Encoder {
    fn send_frame(&mut self, frame: &VideoFrame) -> Result<()> {
        let format = if self.cfg.bit_depth == 8 { PixelFormat::Yuv420p } else { PixelFormat::Yuv420p10le };
        let want = check_frame("software AV1", frame, self.cfg.width, self.cfg.height, &[format])?;
        if std::mem::take(&mut self.force_key) && self.frames != 0 {
            // A fresh encoder's first frame is a key frame; the rate
            // controller resumes where it was.
            let mut cfg = self.cfg.clone();
            cfg.quantizer = self.inner.quantizer();
            self.inner = av1::Encoder::new(cfg);
            self.frames = 0;
        }
        let mut picture = av1::Frame::new(self.cfg.width, self.cfg.height, self.cfg.bit_depth, av1::ChromaFormat::Yuv420);
        picture.data.copy_from_slice(&frame.data[..want]);
        let is_keyframe = self.frames.is_multiple_of(u64::from(self.cfg.keyframe_interval.max(1)));
        let data = self.inner.encode(&picture).context("the software AV1 encoder refused a frame")?;
        self.frames += 1;
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
        // The chunked path discards a lead-in and needs the first kept frame
        // to be a key frame, or the chunk will not stand alone.
        self.force_key = true;
        Ok(())
    }

    /// Rebuild the encoder: no references, the next frame a key frame.
    fn reset(&mut self) -> Result<()> {
        self.inner = av1::Encoder::new(self.cfg.clone());
        self.frames = 0;
        self.force_key = false;
        self.ready.clear();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode::tuning::{EncodeOverrides, RateMode};

    fn config(w: u32, h: u32) -> EncoderConfig {
        EncoderConfig { width: w, height: h, keyframe_interval: 3, ..EncoderConfig::default() }
    }

    fn psnr(a: &[u8], b: &[u8]) -> f64 {
        let mse = a.iter().zip(b).map(|(x, y)| (f64::from(*x) - f64::from(*y)).powi(2)).sum::<f64>() / a.len() as f64;
        10.0 * (255.0f64 * 255.0 / mse.max(1e-9)).log10()
    }

    /// Every packet decodes with rivet's own decoder to the encoder's own
    /// reconstruction, near the source; key frames where the interval and a
    /// forced key put them, timestamps unchanged.
    #[test]
    fn packets_decode_near_the_source() {
        let (w, h) = (64, 48);
        let mut enc = Av1Encoder::new(config(w, h)).unwrap();
        let mut dec = av1::Decoder::new();
        let mut keys = Vec::new();
        for n in 0..5u64 {
            if n == 4 {
                enc.force_keyframe_next().unwrap();
            }
            let src = super::super::native::test_picture(w, h, n);
            enc.send_frame(&src).unwrap();
            let p = enc.receive_packet().unwrap().expect("a packet per frame");
            assert_eq!(p.pts, n);
            keys.push(p.is_keyframe);
            let shown = dec.decode(&p.data).unwrap().expect("a shown frame");
            assert_eq!(&shown, enc.inner.reconstruction().unwrap(), "frame {n}: decoder and encoder agree");
            let luma = (w * h) as usize;
            let db = psnr(&shown.data[..luma], &src.data[..luma]);
            assert!(db > 30.0, "frame {n}: {db:.1} dB");
        }
        assert_eq!(keys, [true, false, false, true, true]);
    }

    #[test]
    fn ten_bit_is_coded() {
        let cfg = EncoderConfig { pixel_format: PixelFormat::Yuv420p10le, ..config(32, 32) };
        let mut enc = Av1Encoder::new(cfg).unwrap();
        let mut data = Vec::new();
        for i in 0..32 * 32 + 2 * 16 * 16 {
            data.extend_from_slice(&((i % 900 + 50) as u16).to_le_bytes());
        }
        let frame = VideoFrame::new(Bytes::from(data), 32, 32, PixelFormat::Yuv420p10le, crate::frame::ColorSpace::Bt709, 7);
        enc.send_frame(&frame).unwrap();
        let p = enc.receive_packet().unwrap().unwrap();
        let back = av1::Decoder::new().decode(&p.data).unwrap().unwrap();
        assert_eq!(back.bit_depth, 10);
    }

    #[test]
    fn what_it_cannot_code_is_refused_by_name() {
        let wide = EncoderConfig { width: 4100, height: 64, ..EncoderConfig::default() };
        assert!(Av1Encoder::new(wide).err().unwrap().to_string().contains("4096"));
        let twelve = EncoderConfig { pixel_format: PixelFormat::Yuv444p, ..config(64, 64) };
        assert!(Av1Encoder::new(twelve).err().unwrap().to_string().contains("profile 0"));
        let cbr = EncodeOverrides { rate_mode: Some(RateMode::Constant), bitrate: Some(1_000_000), ..Default::default() };
        let msg = Av1Encoder::new(EncoderConfig { overrides: cbr, ..config(64, 64) }).err().unwrap().to_string();
        assert!(msg.contains("constant"), "{msg}");
    }

    /// Noisy frames, so a rate has something to spend its bits on.
    fn noisy(w: u32, h: u32, n: u64) -> VideoFrame {
        let mut seed = 0x9e37_79b9_7f4a_7c15u64 ^ n;
        let mut data = vec![128u8; (w * h * 3 / 2) as usize];
        for (i, v) in data.iter_mut().enumerate().take((w * h) as usize) {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let x = (i as u32 % w) as u64;
            *v = ((x * 2 + n * 3) % 160 + 40) as u8 ^ ((seed >> 59) as u8);
        }
        VideoFrame::new(Bytes::from(data), w, h, PixelFormat::Yuv420p, crate::frame::ColorSpace::Bt709, n)
    }

    /// A bitrate rung lands near its rate, and a higher rate spends more.
    #[test]
    fn a_bitrate_rung_is_coded_to_its_rate() {
        let (w, h) = (96, 64);
        let achieved = |bitrate: u32| {
            let overrides = EncodeOverrides { bitrate: Some(bitrate), ..Default::default() };
            let cfg = EncoderConfig { overrides, frame_rate: 25.0, keyframe_interval: 50, ..config(w, h) };
            let mut enc = Av1Encoder::new(cfg).unwrap();
            let mut bits = 0u64;
            let n = 40;
            for i in 0..n {
                enc.send_frame(&noisy(w, h, i)).unwrap();
                bits += enc.receive_packet().unwrap().unwrap().data.len() as u64 * 8;
            }
            bits as f64 * 25.0 / n as f64
        };
        let (low, high) = (achieved(150_000), achieved(400_000));
        for (asked, got) in [(150_000.0, low), (400_000.0, high)] {
            let ratio = got / asked;
            assert!((0.6..1.6).contains(&ratio), "asked {asked} b/s, got {got:.0} ({ratio:.2}x)");
        }
        assert!(high > low * 1.5, "{low:.0} vs {high:.0}");
    }
}
