//! Rung and quality types — one rendition of the output ladder plus the encoder
//! quality knobs that control it.

use codec::encode::tuning::{EncodeOverrides, QualityTarget, SpeedTier};
use codec::encode::{AUTO_FROM_TARGET, EncoderConfig};
use codec::frame::VideoFrame;

pub use crate::fit::{Fit, Orientation, Placement};

/// Encoder quality knobs for a rung.
#[derive(Debug, Clone)]
pub struct Quality {
    /// Constant rate factor in the encoder-native scale (the software AV1 encoder takes four times it as base_q_idx, 0..=255; NVENC scales it).
    /// `None` derives the quantizer from [`Quality::target`].
    pub crf: Option<u8>,
    /// Encoder-native speed preset. `None` derives it from [`Quality::tier`].
    pub speed_preset: Option<u8>,
    /// Perceptual quality target (used when `crf` is `None`).
    pub target: QualityTarget,
    /// Speed/efficiency tier (used when `speed_preset` is `None`).
    pub tier: SpeedTier,
    /// GOP length in frames. `None` → `2 × frame_rate` (a 2-second GOP).
    pub keyframe_interval: Option<u32>,
    /// Backend-agnostic per-rung knobs — a quality shift in libaom-CQ steps,
    /// tiles, reference frames, lookahead, B-frames, and so on — layered on
    /// top of the target/tier. The default is inert. Set per rung by a caller
    /// that knows the ladder (a per-title shift, say), or by the engine from
    /// [`OutputSpec::rung_policy`](super::OutputSpec::rung_policy).
    pub overrides: EncodeOverrides,
}

impl Default for Quality {
    fn default() -> Self {
        Self {
            crf: None,
            speed_preset: None,
            target: QualityTarget::Standard,
            tier: SpeedTier::Standard,
            keyframe_interval: None,
            overrides: EncodeOverrides::default(),
        }
    }
}

impl Quality {
    /// A constant-rate-factor quality.
    pub fn crf(crf: u8) -> Self {
        Self {
            crf: Some(crf),
            ..Default::default()
        }
    }

    /// A perceptual-target quality.
    pub fn target(target: QualityTarget) -> Self {
        Self {
            target,
            ..Default::default()
        }
    }

    /// A quality with these per-rung knobs.
    pub fn with_overrides(mut self, overrides: EncodeOverrides) -> Self {
        self.overrides = overrides;
        self
    }

    /// Apply these knobs onto an [`EncoderConfig`] for a given frame rate.
    pub(crate) fn apply(&self, cfg: &mut EncoderConfig, frame_rate: f64) {
        cfg.target = self.target;
        cfg.tier = self.tier;
        cfg.quality = self.crf.unwrap_or(AUTO_FROM_TARGET);
        cfg.speed_preset = self.speed_preset.unwrap_or(AUTO_FROM_TARGET);
        cfg.keyframe_interval = self
            .keyframe_interval
            .unwrap_or_else(|| (frame_rate * 2.0).round().max(1.0) as u32);
        cfg.overrides = self.overrides;
    }
}

/// One rendition of the output ladder.
///
/// `width x height` is a **box** the source is fitted into (see
/// [`crate::fit`]), not the output size: the engine replaces them with the
/// size it produces before encoding, and records how in
/// [`placement`](Self::placement).
#[derive(Debug, Clone)]
pub struct Rung {
    /// Box width in pixels (even); the output width once fitted.
    pub width: u32,
    /// Box height in pixels (even); the output height once fitted.
    pub height: u32,
    /// Human label, e.g. `"720p"` (short side). Auto-derived by [`Rung::new`],
    /// and re-derived from the output size when fitting changes it.
    pub label: String,
    /// Per-rung encoder quality.
    pub quality: Quality,
    /// This rung's [`Fit`]; `None` takes [`OutputSpec::fit`](super::OutputSpec::fit).
    pub fit: Option<Fit>,
    /// This rung's [`Orientation`]; `None` takes
    /// [`OutputSpec::orientation`](super::OutputSpec::orientation).
    pub orientation: Option<Orientation>,
    /// Whether this rung may be larger than the source; `None` takes
    /// [`OutputSpec::upscale`](super::OutputSpec::upscale).
    pub upscale: Option<bool>,
    /// How the source is cropped, scaled and padded into this rung — set by
    /// the engine when it fits the ladder to the source. `None` on a rung
    /// nothing fitted is a plain resize to `width x height`.
    pub placement: Option<Placement>,
    /// This rung's rate is the engine's standard one (`WxH@standard`): the
    /// rate it would have with none named anywhere. A spec-wide rate — the
    /// rung policy's `bitrate=` or `video-bitrate` — does not reach it, so a
    /// constant-rate rung takes the default for its codec, size and frame
    /// rate ([`default_cbr_bitrate`](codec::encode::tuning::default_cbr_bitrate)),
    /// and an average-rate one stays coded to its quality target. The
    /// rung's own [`Quality::overrides`] bitrate, when set, still wins.
    pub standard_rate: bool,
}

impl Rung {
    /// A rung at `width × height` with default quality and an auto label
    /// (`"<short-side>p"`).
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            label: format!("{}p", width.min(height)),
            quality: Quality::default(),
            fit: None,
            orientation: None,
            upscale: None,
            placement: None,
            standard_rate: false,
        }
    }

    /// Give this rung the engine's standard rate whatever spec-wide rate is
    /// set. See [`Rung::standard_rate`].
    pub fn with_standard_rate(mut self) -> Self {
        self.standard_rate = true;
        self
    }

    /// Fit the source into this rung's box this way, whatever the spec says.
    pub fn with_fit(mut self, fit: Fit) -> Self {
        self.fit = Some(fit);
        self
    }

    /// Turn (or not) this rung's box to the source's orientation, whatever
    /// the spec says.
    pub fn with_orientation(mut self, orientation: Orientation) -> Self {
        self.orientation = Some(orientation);
        self
    }

    /// Allow (or not) this rung to be larger than the source, whatever the
    /// spec says.
    pub fn with_upscale(mut self, upscale: bool) -> Self {
        self.upscale = Some(upscale);
        self
    }

    /// Produce this rung's frame from a decoded one: its [`Placement`], or a
    /// plain resize to `width x height` when it has none.
    pub fn scale(&self, frame: &VideoFrame) -> anyhow::Result<VideoFrame> {
        match &self.placement {
            Some(p) => p.apply(frame),
            None => codec::colorspace::scale_frame(frame, self.width, self.height),
        }
    }

    /// Override the per-rung quality.
    pub fn with_quality(mut self, quality: Quality) -> Self {
        self.quality = quality;
        self
    }

    /// Override the label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = label.into();
        self
    }

    /// Short side (the "p" number).
    pub fn short_side(&self) -> u32 {
        self.width.min(self.height)
    }
}
