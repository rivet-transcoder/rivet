//! What the adapters of this workspace's own clean-room encoders share —
//! ProRes ([`prores_sw`](super::prores_sw)), VP8 ([`vp8_sw`](super::vp8_sw)),
//! VP9 ([`vp9_sw`](super::vp9_sw)), MPEG-2 ([`mpeg2_sw`](super::mpeg2_sw)) and
//! MPEG-4 Part 2 ([`mpeg4_sw`](super::mpeg4_sw)): frame checks, the frame
//! rate as a ratio, the quantiser a rung asks for, and the timestamps of
//! pictures coded out of display order.
//!
//! # Why these codecs are software only, and always reachable
//!
//! No hardware backend here encodes them: NVENC, AMF and QSV are wired for
//! AV1, H.264 and H.265 alone. So for these five codecs the software encoder
//! is not a fallback below silicon — it is the encoder, and
//! [`select_encoder`](super::select_encoder) builds it directly, with no
//! feature to enable. The rule the `-fallback` features exist for ("never
//! silently trade a missing GPU for a slow CPU path") has nothing to guard:
//! there is no faster path being skipped.

use std::collections::VecDeque;

use anyhow::{Result, bail};

use super::{AUTO_FROM_TARGET, EncoderConfig, tuning};
use crate::frame::{PixelFormat, VideoCodec, VideoFrame};

/// The bytes of a `width` x `height` frame in `format` (4:2:0 planar, 8 or
/// 10 bits), or an error naming `name` for a frame the encoder cannot take.
pub(crate) fn check_frame(name: &str, frame: &VideoFrame, width: u32, height: u32, formats: &[PixelFormat]) -> Result<usize> {
    if !formats.contains(&frame.format) {
        bail!(
            "the {name} encoder takes {formats:?} frames and got a {:?} one. Convert with the colorspace \
             filter before the encoder.",
            frame.format
        );
    }
    if frame.width != width || frame.height != height {
        bail!(
            "frame is {}x{} but the {name} encoder was configured for {width}x{height}",
            frame.width,
            frame.height
        );
    }
    let (cw, ch) = (width.div_ceil(2) as usize, height.div_ceil(2) as usize);
    let samples = width as usize * height as usize + 2 * cw * ch;
    let want = if frame.format == PixelFormat::Yuv420p10le { samples * 2 } else { samples };
    if frame.data.len() < want {
        bail!("frame buffer is {} bytes, too short for {width}x{height} {:?} ({want} expected)", frame.data.len(), frame.format);
    }
    Ok(want)
}

/// `frame_rate` as a ratio: the NTSC rates as their `/1001` fractions, a
/// whole rate over 1, anything else to a thousandth.
pub(crate) fn frame_rate_ratio(frame_rate: f64) -> (u32, u32) {
    let fps = if frame_rate.is_finite() && frame_rate > 0.0 { frame_rate } else { 30.0 };
    for n in [24_000u32, 30_000, 48_000, 60_000, 120_000] {
        if (fps - f64::from(n) / 1001.0).abs() < 0.005 {
            return (n, 1001);
        }
    }
    if (fps - fps.round()).abs() < 0.001 {
        return (fps.round() as u32, 1);
    }
    let n = (fps * 1000.0).round() as u32;
    let g = gcd(n, 1000);
    (n / g, 1000 / g)
}

fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 { a.max(1) } else { gcd(b, a % b) }
}

/// The quantiser the rung asks for, on the codec's own scale: the caller's
/// CRF when it set one (the scale of each codec is in
/// [`crf_scale_max`](super::crf_scale_max): VP8 / VP9 0-63 like libvpx's
/// `cq-level`, MPEG-2 / MPEG-4 their 1-31 codes), else the quality target's
/// ([`tuning::native_sw_quantizer`]).
pub(crate) fn quantizer(config: &EncoderConfig) -> u8 {
    if config.quality != AUTO_FROM_TARGET {
        let crf = u32::from(config.quality);
        return match config.codec {
            VideoCodec::Vp8 => (crf * 2).min(127) as u8,
            VideoCodec::Vp9 => (crf * 4).clamp(1, 255) as u8,
            _ => crf.clamp(1, 31) as u8,
        };
    }
    tuning::native_sw_quantizer(config.codec, config.target, &config.overrides)
}

/// The speed tier the rung asks for.
pub(crate) fn tier(config: &EncoderConfig) -> tuning::SpeedTier {
    config.overrides.speed_tier.unwrap_or(config.tier)
}

/// Refuse, by name, a rung that asks `name` for a rate it does not code:
/// any rate at all for an encoder with a fixed quantiser (VP8, VP9) or a
/// fixed frame size (ProRes).
pub(crate) fn refuse_any_rate(name: &str, config: &EncoderConfig) -> Result<()> {
    let o = &config.overrides;
    if o.rate_mode.is_some() || o.bitrate.is_some() || o.buffer_ms.is_some_and(|ms| ms > 0) {
        bail!(
            "the {name} encoder codes to a fixed quantiser and this rung asks for a rate (bitrate={:?}, \
             buffer={:?}ms, mode={:?}): drop the bitrate and encode to a quality target (or a crf)",
            o.bitrate,
            o.buffer_ms,
            o.rate_mode.map(|m| m.name())
        );
    }
    Ok(())
}

/// The average bit rate an MPEG-2 / MPEG-4 rung asks for, or `None` for a
/// quality target. Their encoders code an average rate (a per-picture
/// quantiser from the bits spent so far); a constant rate, a coded picture
/// buffer, or a rate beside a CRF are refused by name.
pub(crate) fn average_rate(name: &str, config: &EncoderConfig) -> Result<Option<u32>> {
    let o = &config.overrides;
    if o.rate_mode == Some(tuning::RateMode::Constant) {
        bail!("the {name} encoder codes an average rate, not a constant one (rate=cbr): drop rate=cbr");
    }
    if o.buffer_ms.is_some_and(|ms| ms > 0) {
        bail!(
            "the {name} encoder has no coded picture buffer model, and this rung declares one ({:?} ms): \
             drop the buffer (video-buffer=0)",
            o.buffer_ms
        );
    }
    if o.bitrate.is_some() && config.quality != AUTO_FROM_TARGET {
        bail!("this rung asks the {name} encoder for both a crf and a bitrate; give one");
    }
    Ok(o.bitrate)
}

/// The presentation timestamps of pictures an encoder codes out of display
/// order, in the one pattern the MPEG-2 and MPEG-4 encoders use: each burst
/// of output is the newest frame held (coded as the reference picture) and
/// then every frame held before it, oldest first (the B pictures that wait
/// on it).
#[derive(Debug, Default)]
pub(crate) struct ReferenceFirst {
    held: VecDeque<u64>,
}

impl ReferenceFirst {
    /// A frame went in.
    pub(crate) fn push(&mut self, pts: u64) {
        self.held.push_back(pts);
    }

    /// The timestamps of `coded` pictures that came out, in coding order.
    pub(crate) fn take(&mut self, coded: usize) -> Result<Vec<u64>> {
        if coded == 0 {
            return Ok(Vec::new());
        }
        if coded != self.held.len() {
            bail!(
                "the encoder coded {coded} pictures while {} frames were waiting; its reordering is not the \
                 reference-then-B pattern this adapter times",
                self.held.len()
            );
        }
        let mut out = Vec::with_capacity(coded);
        out.extend(self.held.pop_back());
        out.extend(self.held.drain(..));
        Ok(out)
    }

    pub(crate) fn clear(&mut self) {
        self.held.clear();
    }
}

/// A moving 8-bit 4:2:0 test picture: frame `n` of a diagonal ramp.
#[cfg(test)]
pub(crate) fn test_picture(w: u32, h: u32, n: u64) -> VideoFrame {
    let mut data = vec![128u8; (w * h * 3 / 2) as usize];
    for y in 0..h {
        for x in 0..w {
            data[(y * w + x) as usize] = ((x * 3 + y * 2 + n as u32 * 4) % 220 + 16) as u8;
        }
    }
    VideoFrame::new(bytes::Bytes::from(data), w, h, PixelFormat::Yuv420p, crate::frame::ColorSpace::Bt709, n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_rates_become_exact_ratios() {
        assert_eq!(frame_rate_ratio(25.0), (25, 1));
        assert_eq!(frame_rate_ratio(30000.0 / 1001.0), (30000, 1001));
        assert_eq!(frame_rate_ratio(23.976), (24000, 1001));
        assert_eq!(frame_rate_ratio(12.5), (25, 2));
        assert_eq!(frame_rate_ratio(f64::NAN), (30, 1));
    }

    #[test]
    fn reference_first_puts_the_newest_first() {
        let mut o = ReferenceFirst::default();
        o.push(0);
        assert_eq!(o.take(1).unwrap(), vec![0]);
        o.push(1);
        o.push(2);
        assert_eq!(o.take(0).unwrap(), Vec::<u64>::new());
        o.push(3);
        assert_eq!(o.take(3).unwrap(), vec![3, 1, 2]);
        o.push(4);
        assert!(o.take(2).is_err());
    }
}
