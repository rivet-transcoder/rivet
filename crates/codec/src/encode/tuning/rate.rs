//! How a bitrate rung spends its rate: an average, or a constant rate (CBR).
//!
//! # The two modes
//!
//! [`RateMode::Average`] is what a bitrate rung has always been: the rate
//! controller spends the rate on average, within the optional coded picture
//! buffer the rung declares ([`EncodeOverrides::buffer_ms`]). The native
//! software H.264 / H.265 encoder codes it, and nothing else does.
//!
//! [`RateMode::Constant`] is CBR: the rate is also the maximum, an HRD
//! buffer is always declared (one second, [`CBR_DEFAULT_BUFFER_MS`], unless
//! the rung names another), and the encoder holds the rate, padding with
//! filler where the backend does that. Every hardware backend codes it —
//! QSV, NVENC and AMF, for every codec each of them encodes, AV1 included.
//! rav1e targets a bitrate but not a constant one and refuses it by name;
//! the native software H.264 / H.265 encoder refuses it until it codes one
//! (`h26x_sw::CODES_CONSTANT_RATE`).
//!
//! A constant-rate rung that names no bitrate of its own takes one from
//! [`default_cbr_bitrate`], by codec, size and frame rate. That happens
//! where the frame rate is known — the engine's job setup — so the encoders
//! themselves always see an explicit rate and refuse a constant rung without
//! one ([`constant_rate_refusal`]).

use std::str::FromStr;

use super::EncodeOverrides;
use crate::frame::VideoCodec;

/// How a bitrate rung ([`EncodeOverrides::bitrate`]) spends its rate.
///
/// `None` in [`EncodeOverrides::rate_mode`] is [`RateMode::Average`], so an
/// override that names no mode keeps the behaviour every bitrate rung has
/// always had.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RateMode {
    /// The rate on average, within the rung's optional buffer.
    #[default]
    Average,
    /// A constant rate (CBR): the rate is also the maximum, an HRD buffer is
    /// declared, and the encoder holds the rate.
    Constant,
}

impl RateMode {
    /// The spelling the settings key and the policy grammar write for it.
    pub fn name(self) -> &'static str {
        match self {
            Self::Average => "average",
            Self::Constant => "cbr",
        }
    }
}

/// The `rate-mode` / `rate=` word: `cbr` or `constant`, `average` or `abr`,
/// case-insensitively.
pub fn parse_rate_mode(value: &str) -> Option<RateMode> {
    match value.trim().to_ascii_lowercase().as_str() {
        "cbr" | "constant" => Some(RateMode::Constant),
        "average" | "abr" => Some(RateMode::Average),
        _ => None,
    }
}

impl FromStr for RateMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        parse_rate_mode(s).ok_or_else(|| format!("rate mode must be cbr|constant|average|abr; got '{s}'"))
    }
}

/// The HRD buffer a constant-rate rung declares when it names none, in
/// milliseconds of its rate: one second, the same default as a software
/// bitrate rung's buffer.
pub const CBR_DEFAULT_BUFFER_MS: u32 = 1000;

/// How full the decoder's buffer is, in 64ths, when the first picture is
/// removed from it — the initial delay every backend is handed with the
/// buffer (QSV `InitialDelayInKB`, NVENC `vbvInitialDelay`, AMF
/// `InitialVBVBufferFullness`, whose own scale is 0..=64). Three quarters:
/// room for the opening keyframe, which is the largest picture a segment
/// carries, without starting the buffer at the brim.
pub const CBR_INITIAL_FULLNESS_64THS: u32 = 48;

/// A constant-rate request, resolved: the rate and the buffer a backend is
/// handed. Built only for a [`RateMode::Constant`] rung that names a
/// positive bitrate — [`Self::from_overrides`] — so a backend that holds one
/// has nothing left to decide.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConstantRate {
    /// Target rate, which is also the maximum, in bits per second.
    pub bps: u32,
    /// The HRD buffer, in milliseconds of `bps`. Never 0: a constant-rate
    /// rung that names no buffer is refused ([`constant_rate_refusal`]).
    pub buffer_ms: u32,
}

impl ConstantRate {
    /// The constant rate `overrides` asks for: `Some` for a
    /// [`RateMode::Constant`] rung with a positive bitrate, its buffer the
    /// one it names or [`CBR_DEFAULT_BUFFER_MS`]. `None` for anything else.
    pub fn from_overrides(overrides: &EncodeOverrides) -> Option<Self> {
        if overrides.rate_mode != Some(RateMode::Constant) {
            return None;
        }
        let bps = overrides.bitrate.filter(|&bps| bps > 0)?;
        Some(Self { bps, buffer_ms: overrides.buffer_ms.unwrap_or(CBR_DEFAULT_BUFFER_MS) })
    }

    /// The buffer's size in bits.
    pub fn buffer_bits(self) -> u64 {
        u64::from(self.bps) * u64::from(self.buffer_ms) / 1000
    }

    /// The buffer's fullness at the first removal, in bits
    /// ([`CBR_INITIAL_FULLNESS_64THS`] of [`Self::buffer_bits`]).
    pub fn initial_fullness_bits(self) -> u64 {
        self.buffer_bits() * u64::from(CBR_INITIAL_FULLNESS_64THS) / 64
    }
}

/// Why a [`RateMode::Constant`] rung cannot be coded as asked, or `None`
/// when it can (or when the rung is not a constant-rate one). Backend-
/// agnostic: which backend codes it is decided where the pool is known.
///
/// `crf` is the rung's CRF when it names one, and `constant_qp` whether its
/// encode must be constant-QP (the chunked path's `--seam-mode constqp`).
pub fn constant_rate_refusal(overrides: &EncodeOverrides, crf: Option<u8>, constant_qp: bool) -> Option<String> {
    if overrides.rate_mode != Some(RateMode::Constant) {
        return None;
    }
    let Some(bps) = overrides.bitrate else {
        return Some(
            "rate=cbr codes the rung at a constant rate, and it has no bitrate: name one (`--video-bitrate`, \
             `--rung WxH@RATE` or `bitrate=`), or drop rate=cbr"
                .into(),
        );
    };
    if bps == 0 {
        return Some("bitrate=0 is not a rate: name a positive bitrate for a rate=cbr rung".into());
    }
    if let Some(q) = crf {
        return Some(format!(
            "crf={q} names a quantiser and rate=cbr a constant rate ({bps} bit/s), and a rung is coded to one \
             or the other: drop one of them"
        ));
    }
    if constant_qp {
        return Some(format!(
            "`--seam-mode constqp` codes every chunk at a constant quantiser, and this rung is coded at a \
             constant rate (rate=cbr, {bps} bit/s): drop one of them"
        ));
    }
    if overrides.buffer_ms == Some(0) {
        return Some(format!(
            "buffer=0 declares no coded picture buffer, and a rate=cbr rung ({bps} bit/s) is held to the rate \
             within one: name a buffer (`--video-buffer 1s`, `buffer=`) or leave it unset for one second"
        ));
    }
    None
}

/// The rate a [`RateMode::Constant`] rung with no rate of its own is coded
/// at, bits per second: a streaming rate by codec, the rung's short side,
/// and its frame rate.
///
/// H.264 at up to 30 fps, by short side:
///
/// | short side | 144 | 240 | 360 | 480 | 720 | 1080 | 1440 | 2160 |
/// |---|---|---|---|---|---|---|---|---|
/// | Mb/s | 0.2 | 0.4 | 0.8 | 1.2 | 3 | 5 | 9 | 16 |
///
/// - Between two rows the rate is interpolated linearly in the short side;
///   below 144 it is the 144 row; above 2160 it grows with the area.
/// - Above 30 fps it grows by half the extra frame rate — 60 fps is 1.5x,
///   120 fps (the cap) 2.5x — since a higher frame rate costs bits, but
///   less than in proportion: each picture is closer to the one before it.
///   At or below 30 fps it is the table as written.
/// - H.265 is 0.65x H.264 and AV1 0.5x, their usual efficiency against it.
///
/// Rounded to a whole kilobit. The anchors are common streaming-ladder
/// rates, not a measurement of this encoder; a rung that needs something
/// else names it.
pub fn default_cbr_bitrate(codec: VideoCodec, short_side: u32, fps: f64) -> u32 {
    const H264_30FPS: &[(u32, f64)] = &[
        (144, 200_000.0),
        (240, 400_000.0),
        (360, 800_000.0),
        (480, 1_200_000.0),
        (720, 3_000_000.0),
        (1080, 5_000_000.0),
        (1440, 9_000_000.0),
        (2160, 16_000_000.0),
    ];
    let side = short_side.max(1);
    let (first, last) = (H264_30FPS[0], H264_30FPS[H264_30FPS.len() - 1]);
    let base = if side <= first.0 {
        first.1
    } else if side >= last.0 {
        let scale = f64::from(side) / f64::from(last.0);
        last.1 * scale * scale
    } else {
        H264_30FPS
            .windows(2)
            .find(|w| side <= w[1].0)
            .map(|w| {
                let ((s0, r0), (s1, r1)) = (w[0], w[1]);
                r0 + (r1 - r0) * f64::from(side - s0) / f64::from(s1 - s0)
            })
            .unwrap_or(last.1)
    };
    let fps = if fps.is_finite() && fps > 30.0 { fps.min(120.0) } else { 30.0 };
    let frame_rate_scale = 1.0 + 0.5 * (fps / 30.0 - 1.0);
    let codec_scale = match codec {
        VideoCodec::H264 => 1.0,
        VideoCodec::H265 => 0.65,
        VideoCodec::Av1 => 0.5,
    };
    let bps = (base * frame_rate_scale * codec_scale / 1000.0).round() * 1000.0;
    bps.clamp(1000.0, f64::from(u32::MAX)) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_table_is_the_documented_one() {
        let h264 = |side| default_cbr_bitrate(VideoCodec::H264, side, 30.0);
        assert_eq!(h264(2160), 16_000_000);
        assert_eq!(h264(1440), 9_000_000);
        assert_eq!(h264(1080), 5_000_000);
        assert_eq!(h264(720), 3_000_000);
        assert_eq!(h264(480), 1_200_000);
        assert_eq!(h264(360), 800_000);
        assert_eq!(h264(240), 400_000);
        assert_eq!(h264(144), 200_000);
        // Interpolated between rows, floored below the table, by area above.
        assert_eq!(h264(540), 1_650_000);
        assert_eq!(h264(100), 200_000);
        assert_eq!(h264(4320), 64_000_000);
    }

    #[test]
    fn av1_is_below_h265_is_below_h264() {
        for side in [240, 360, 480, 720, 1080, 2160] {
            let (av1, h265, h264) = (
                default_cbr_bitrate(VideoCodec::Av1, side, 30.0),
                default_cbr_bitrate(VideoCodec::H265, side, 30.0),
                default_cbr_bitrate(VideoCodec::H264, side, 30.0),
            );
            assert!(av1 < h265 && h265 < h264, "{side}: {av1} {h265} {h264}");
        }
        assert_eq!(default_cbr_bitrate(VideoCodec::H265, 1080, 30.0), 3_250_000);
        assert_eq!(default_cbr_bitrate(VideoCodec::Av1, 1080, 30.0), 2_500_000);
    }

    #[test]
    fn a_higher_frame_rate_costs_more_but_less_than_in_proportion() {
        let at = |fps| default_cbr_bitrate(VideoCodec::H264, 1080, fps);
        // At or below 30 fps the table is as written.
        assert_eq!(at(24.0), 5_000_000);
        assert_eq!(at(30_000.0 / 1001.0), 5_000_000);
        assert_eq!(at(30.0), 5_000_000);
        assert_eq!(at(60.0), 7_500_000);
        assert_eq!(at(120.0), 12_500_000);
        assert_eq!(at(240.0), 12_500_000, "capped at 120 fps");
        assert_eq!(at(f64::NAN), 5_000_000);
    }

    #[test]
    fn the_mode_parses_in_both_spellings() {
        for s in ["cbr", "CBR", "constant", " Constant "] {
            assert_eq!(parse_rate_mode(s), Some(RateMode::Constant), "{s}");
        }
        for s in ["average", "abr", "ABR"] {
            assert_eq!(parse_rate_mode(s), Some(RateMode::Average), "{s}");
        }
        assert_eq!(parse_rate_mode("vbr"), None);
        assert!("vbr".parse::<RateMode>().unwrap_err().contains("cbr|constant|average|abr"));
        assert_eq!(RateMode::Constant.name().parse::<RateMode>(), Ok(RateMode::Constant));
        assert_eq!(RateMode::Average.name().parse::<RateMode>(), Ok(RateMode::Average));
    }

    fn cbr(bps: Option<u32>) -> EncodeOverrides {
        EncodeOverrides { rate_mode: Some(RateMode::Constant), bitrate: bps, ..Default::default() }
    }

    #[test]
    fn a_constant_rate_resolves_with_the_default_buffer() {
        assert_eq!(ConstantRate::from_overrides(&cbr(Some(3_000_000))), Some(ConstantRate { bps: 3_000_000, buffer_ms: 1000 }));
        let named = EncodeOverrides { buffer_ms: Some(500), ..cbr(Some(3_000_000)) };
        let rate = ConstantRate::from_overrides(&named).unwrap();
        assert_eq!(rate.buffer_bits(), 1_500_000);
        assert_eq!(rate.initial_fullness_bits(), 1_125_000);
        // An average rung, a rung with no rate, and a zero rate resolve to none.
        assert_eq!(ConstantRate::from_overrides(&EncodeOverrides { bitrate: Some(3_000_000), ..Default::default() }), None);
        assert_eq!(ConstantRate::from_overrides(&cbr(None)), None);
        assert_eq!(ConstantRate::from_overrides(&cbr(Some(0))), None);
    }

    #[test]
    fn a_constant_rate_request_that_cannot_be_coded_is_refused_by_name() {
        let refuse = |o: EncodeOverrides, crf, cqp, words: &[&str]| {
            let why = constant_rate_refusal(&o, crf, cqp).expect("refused");
            for w in words {
                assert!(why.contains(w), "{w} not in: {why}");
            }
        };
        refuse(cbr(None), None, false, &["rate=cbr", "no bitrate", "--video-bitrate"]);
        refuse(cbr(Some(0)), None, false, &["bitrate=0"]);
        refuse(cbr(Some(2_000_000)), Some(28), false, &["crf=28", "rate=cbr", "2000000"]);
        refuse(cbr(Some(2_000_000)), None, true, &["constqp", "rate=cbr"]);
        refuse(EncodeOverrides { buffer_ms: Some(0), ..cbr(Some(2_000_000)) }, None, false, &["buffer=0", "rate=cbr"]);
        // What can be coded, and what is not a constant-rate rung at all.
        assert_eq!(constant_rate_refusal(&cbr(Some(2_000_000)), None, false), None);
        let average = EncodeOverrides { bitrate: Some(2_000_000), buffer_ms: Some(0), ..Default::default() };
        assert_eq!(constant_rate_refusal(&average, Some(28), true), None);
    }
}
