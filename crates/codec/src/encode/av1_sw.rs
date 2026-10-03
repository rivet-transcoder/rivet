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
//! Profile 0: 8- or 10-bit 4:2:0 (`Yuv420p`, `Yuv420p10le`), one temporal
//! unit per frame, every frame shown and in display order — so each packet
//! carries its own frame's timestamp. Any width (frames wider than 4096 are
//! coded in several tile columns). Key frames at the interval, and wherever
//! `force_keyframe_next` asks (the encoder's own `force_keyframe`: the next
//! frame is a key frame with its sequence header, nothing else is reset);
//! inter frames predict from the last two frames and a golden frame, and may
//! average two of them.
//!
//! The colour description goes into the sequence header (`color_config()`:
//! primaries, transfer, matrix, range, from the job's `color_metadata`), and
//! HDR10 mastering display and content light level metadata into metadata
//! OBUs on every key frame — so a 10-bit PQ or HLG rung says what it is in
//! the bitstream as well as in the container's `colr` / `mdcv` / `clli`.
//!
//! # Rate and quality
//!
//! The quantiser (`base_q_idx`, 1-255) is the rung's CRF on AV1's 0-63 scale
//! times four, else the quality target's ([`tuning::av1_sw_params_with`]:
//! four times libaom's `cq-level`, the same table the hardware tiers are
//! equalised against). A bitrate rung is coded to its average rate by the
//! encoder's rate controller (bits per frame at the rung's frame rate); a
//! constant rate or a coded picture buffer is refused by name. The speed
//! tier picks the encoder's effort (`av1::Config::speed`: Draft 8, Standard
//! 6, Archive 4 — how much of its rate-distortion search runs, and which
//! tools) and the motion search range.
//!
//! # Speed
//!
//! A rate-distortion-searching encoder in Rust, SIMD in its hot kernels.
//! The frame is cut into tile columns (as many as the encoder's threads, a
//! power of two, each at least 256 pixels wide, unless `tiles=` asks) that
//! are coded in parallel: at the Standard tier about 0.25 frames/s
//! at 1280x720 on one thread, 0.8 with four (`throughput_at_720p`). A fallback,
//! not a production encoder; the GPU tiers come first.

use anyhow::{Context, Result, bail};
use bytes::Bytes;

use super::native::{average_rate, check_frame};
use super::tuning;
use super::{AUTO_FROM_TARGET, EncodedPacket, Encoder, EncoderConfig};
use crate::frame::{ColorMetadata, PixelFormat, TransferFn, VideoCodec, VideoFrame};

/// rivet's software AV1 encoder.
pub struct Av1Encoder {
    inner: av1::Encoder,
    cfg: av1::Config,
    ready: std::collections::VecDeque<EncodedPacket>,
}

/// The H.273 colour description of the job, as the encoder writes it.
pub fn color_info(m: &ColorMetadata) -> av1::ColorInfo {
    let transfer = match m.transfer {
        TransferFn::Bt709 => 1,
        TransferFn::Bt470Bg => 5,
        TransferFn::Linear => 8,
        TransferFn::St2084 => 16,
        TransferFn::AribStdB67 => 18,
        TransferFn::Unspecified => 2,
    };
    av1::ColorInfo {
        color_primaries: u32::from(m.colour_primaries),
        transfer_characteristics: transfer,
        matrix_coefficients: u32::from(m.matrix_coefficients),
        full_range: m.full_range,
        chroma_sample_position: 0,
    }
}

/// The job's HDR10 metadata in AV1's units: chromaticities from ST 2086's
/// 0.00002 steps to 0.16 fixed point, luminance from 0.0001 cd/m² to 24.8
/// (maximum) and 18.14 (minimum) fixed point.
pub fn hdr_metadata(m: &ColorMetadata) -> av1::HdrMetadata {
    let xy = |v: u16| ((u64::from(v) * 65_536 + 25_000) / 50_000).min(65_535) as u16;
    av1::HdrMetadata {
        content_light: m
            .content_light_level
            .map(|c| av1::ContentLightLevel { max_cll: c.max_cll, max_fall: c.max_fall }),
        mastering_display: m.mastering_display.map(|d| av1::MasteringDisplay {
            primaries: [
                [xy(d.primaries_r_x), xy(d.primaries_r_y)],
                [xy(d.primaries_g_x), xy(d.primaries_g_y)],
                [xy(d.primaries_b_x), xy(d.primaries_b_y)],
            ],
            white_point: [xy(d.white_point_x), xy(d.white_point_y)],
            luminance_max: ((u64::from(d.max_luminance) * 256 + 5_000) / 10_000).min(u64::from(u32::MAX)) as u32,
            luminance_min: ((u64::from(d.min_luminance) * 16_384 + 5_000) / 10_000).min(u64::from(u32::MAX)) as u32,
        }),
    }
}

/// Tile columns (log2) for a frame `width` wide coded on `threads` threads:
/// the largest power of two no more than the threads and no narrower than
/// 256 pixels a tile, unless `asked` names a count (rounded up to a power of
/// two). The encoder adds columns a frame wider than 4096 needs.
pub fn tile_cols_log2(width: u32, threads: usize, asked: Option<u32>) -> u32 {
    if let Some(n) = asked {
        return n.max(1).next_power_of_two().trailing_zeros();
    }
    let n = (threads.max(1) as u32).min((width / 256).max(1));
    31 - n.leading_zeros()
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
        if config.color_metadata.matrix_coefficients == 0 {
            bail!(
                "the software AV1 encoder writes 4:2:0, and this job's colour description has the identity matrix \
                 (RGB), which AV1 allows only at 4:4:4"
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
        let threads = if config.threads == 0 {
            std::thread::available_parallelism().map_or(1, |n| n.get())
        } else {
            config.threads
        };

        let mut cfg = av1::Config::new(config.width, config.height);
        cfg.bit_depth = bit_depth;
        cfg.quantizer = quantizer;
        cfg.keyframe_interval = if config.keyframe_interval == 0 { 240 } else { config.keyframe_interval };
        cfg.search_range = p.search_range;
        cfg.speed = p.speed;
        cfg.tools = av1::Tools::for_speed(p.speed);
        cfg.threads = threads;
        cfg.tile_cols_log2 = tile_cols_log2(config.width, threads, p.tile_columns);
        cfg.color = color_info(&config.color_metadata);
        cfg.hdr = hdr_metadata(&config.color_metadata);
        if let Some(bps) = rate {
            let fps = if config.frame_rate.is_finite() && config.frame_rate > 0.0 { config.frame_rate } else { 30.0 };
            cfg.target_bits_per_frame = Some(((f64::from(bps) / fps).round() as u64).max(1));
        }

        tracing::warn!(
            width = config.width,
            height = config.height,
            quantizer,
            bit_depth,
            speed = cfg.speed,
            tiles = 1u32 << cfg.tile_cols_log2,
            threads,
            bitrate = ?rate,
            "no AV1 encode silicon available or asked for — encoding with rivet's own software AV1 encoder, \
             which is far slower than any hardware backend"
        );
        Ok(Self { inner: av1::Encoder::new(cfg.clone()), cfg, ready: Default::default() })
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
        let mut picture = av1::Frame::new(self.cfg.width, self.cfg.height, self.cfg.bit_depth, av1::ChromaFormat::Yuv420);
        picture.data.copy_from_slice(&frame.data[..want]);
        let data = self.inner.encode(&picture).context("the software AV1 encoder refused a frame")?;
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
        // The chunked path discards a lead-in and needs the first kept frame
        // to be a key frame, or the chunk will not stand alone.
        self.inner.force_keyframe();
        Ok(())
    }

    /// Rebuild the encoder: no references, the next frame a key frame.
    fn reset(&mut self) -> Result<()> {
        self.inner = av1::Encoder::new(self.cfg.clone());
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
        let rgb = EncoderConfig {
            color_metadata: ColorMetadata { matrix_coefficients: 0, ..Default::default() },
            ..config(64, 64)
        };
        assert!(Av1Encoder::new(rgb).err().unwrap().to_string().contains("identity matrix"));
        let twelve = EncoderConfig { pixel_format: PixelFormat::Yuv444p, ..config(64, 64) };
        assert!(Av1Encoder::new(twelve).err().unwrap().to_string().contains("profile 0"));
        let cbr = EncodeOverrides { rate_mode: Some(RateMode::Constant), bitrate: Some(1_000_000), ..Default::default() };
        let msg = Av1Encoder::new(EncoderConfig { overrides: cbr, ..config(64, 64) }).err().unwrap().to_string();
        assert!(msg.contains("constant"), "{msg}");
    }

    /// A frame wider than one tile may be: coded in several tile columns.
    #[test]
    fn wide_frames_are_coded_in_tile_columns() {
        let (w, h) = (4160, 16);
        let mut enc = Av1Encoder::new(EncoderConfig { tier: crate::encode::SpeedTier::Draft, ..config(w, h) }).unwrap();
        let src = super::super::native::test_picture(w, h, 0);
        enc.send_frame(&src).unwrap();
        let p = enc.receive_packet().unwrap().unwrap();
        let shown = av1::Decoder::new().decode(&p.data).unwrap().unwrap();
        assert_eq!(&shown, enc.inner.reconstruction().unwrap());
        assert_eq!(shown.width, w);
    }

    /// A 10-bit PQ (HDR10) rung: BT.2020 PQ in the sequence header, the
    /// mastering display and content light level in metadata OBUs, in AV1's
    /// units — as a fresh decoder reports them.
    #[test]
    fn hdr10_is_signalled_in_the_bitstream() {
        use crate::frame::{ContentLightLevel, MasteringDisplay};
        let meta = ColorMetadata {
            transfer: TransferFn::St2084,
            matrix_coefficients: 9,
            colour_primaries: 9,
            full_range: false,
            // BT.2020 primaries, D65, 1000 / 0.005 cd/m2, in ST 2086's units.
            mastering_display: Some(MasteringDisplay {
                primaries_r_x: 35_400,
                primaries_r_y: 14_600,
                primaries_g_x: 8_500,
                primaries_g_y: 39_850,
                primaries_b_x: 6_550,
                primaries_b_y: 2_300,
                white_point_x: 15_635,
                white_point_y: 16_450,
                max_luminance: 10_000_000,
                min_luminance: 50,
            }),
            content_light_level: Some(ContentLightLevel { max_cll: 1000, max_fall: 400 }),
        };
        let cfg = EncoderConfig { pixel_format: PixelFormat::Yuv420p10le, color_metadata: meta, ..config(32, 32) };
        let mut enc = Av1Encoder::new(cfg).unwrap();
        let mut data = Vec::new();
        for i in 0..32 * 32 + 2 * 16 * 16 {
            data.extend_from_slice(&((i % 900 + 50) as u16).to_le_bytes());
        }
        let frame = VideoFrame::new(Bytes::from(data), 32, 32, PixelFormat::Yuv420p10le, crate::frame::ColorSpace::Bt2020, 0);
        let mut dec = av1::Decoder::new();
        for _ in 0..2 {
            enc.send_frame(&frame).unwrap();
            let p = enc.receive_packet().unwrap().unwrap();
            let back = dec.decode(&p.data).unwrap().unwrap();
            assert_eq!(back.color.color_primaries, 9);
            assert_eq!(back.color.transfer_characteristics, 16);
            assert_eq!(back.color.matrix_coefficients, 9);
            assert!(!back.color.full_range);
            let md = back.hdr.mastering_display.expect("mastering display");
            // 0.708 x 65536 = 46399.49; 1000 cd/m2 in 24.8; 0.005 in 18.14.
            assert_eq!(md.primaries[0][0], 46_399);
            assert_eq!(md.luminance_max, 1000 << 8);
            assert_eq!(md.luminance_min, 82);
            assert_eq!(back.hdr.content_light, Some(av1::ContentLightLevel { max_cll: 1000, max_fall: 400 }));
        }
    }

    #[test]
    fn tile_columns_follow_the_threads() {
        assert_eq!(tile_cols_log2(1280, 1, None), 0);
        assert_eq!(tile_cols_log2(1280, 4, None), 2);
        assert_eq!(tile_cols_log2(1280, 16, None), 2);
        assert_eq!(tile_cols_log2(640, 16, None), 1);
        assert_eq!(tile_cols_log2(3840, 16, None), 3);
        assert_eq!(tile_cols_log2(1920, 1, Some(3)), 2);
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
