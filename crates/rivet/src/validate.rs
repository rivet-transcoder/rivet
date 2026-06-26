//! Optional input-policy validation.
//!
//! These are **advisory** helpers — the job engine ([`crate::job::run_job`])
//! does *not* enforce them, so rivet transcodes whatever it's given. They
//! exist so policy-bearing callers (e.g. a hosted service) can gate uploads
//! with the same limits the reference transcoder uses.

use codec::frame::{PixelFormat, StreamInfo};

/// Minimum accepted short side (pixels).
pub const MIN_RESOLUTION: u32 = 360;
/// Minimum accepted frame rate (fps).
pub const MIN_FRAME_RATE: f64 = 15.0;
/// Maximum accepted duration (seconds).
pub const MAX_DURATION_SECS: f64 = 900.0;

/// Why a stream was rejected by [`validate_stream`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationErrorKind {
    ResolutionTooSmall,
    FrameRateTooSmall,
    DurationTooLong,
    UnsupportedPixelFormat,
}

/// A validation rejection: a machine-readable [`ValidationErrorKind`] plus a
/// human message.
#[derive(Debug, Clone)]
pub struct ValidationError {
    pub kind: ValidationErrorKind,
    pub message: String,
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for ValidationError {}

/// Gate a demuxed stream against the reference resolution/frame-rate/duration/
/// pixel-format policy. Accepts `Yuv420p`, `Yuv420p10le`, `Yuv444p10le`,
/// `Yuva444p10le` (the 4:4:4 formats are downsampled to 4:2:0 by the engine).
pub fn validate_stream(info: &StreamInfo) -> Result<(), ValidationError> {
    if info.width < MIN_RESOLUTION || info.height < MIN_RESOLUTION {
        return Err(ValidationError {
            kind: ValidationErrorKind::ResolutionTooSmall,
            message: format!(
                "Video resolution {}x{} is below the minimum {}x{}.",
                info.width, info.height, MIN_RESOLUTION, MIN_RESOLUTION
            ),
        });
    }
    if info.frame_rate < MIN_FRAME_RATE {
        return Err(ValidationError {
            kind: ValidationErrorKind::FrameRateTooSmall,
            message: format!(
                "Video frame rate {:.1} fps is below the minimum {:.0} fps.",
                info.frame_rate, MIN_FRAME_RATE
            ),
        });
    }
    if info.duration > MAX_DURATION_SECS {
        return Err(ValidationError {
            kind: ValidationErrorKind::DurationTooLong,
            message: format!(
                "Video duration {:.0}s exceeds the maximum {}s.",
                info.duration, MAX_DURATION_SECS
            ),
        });
    }
    if !matches!(
        info.pixel_format,
        PixelFormat::Yuv420p
            | PixelFormat::Yuv420p10le
            | PixelFormat::Yuv444p10le
            | PixelFormat::Yuva444p10le
    ) {
        return Err(ValidationError {
            kind: ValidationErrorKind::UnsupportedPixelFormat,
            message: format!(
                "Pixel format {} is not supported.",
                info.pixel_format.as_ffmpeg_str()
            ),
        });
    }
    Ok(())
}

/// Whether a source pixel format needs the per-frame 4:4:4 → 4:2:0 chroma
/// downsample before encode. The engine consults this to set up the pump.
pub fn needs_chroma_downsample(format: PixelFormat) -> bool {
    matches!(
        format,
        PixelFormat::Yuv444p10le | PixelFormat::Yuva444p10le | PixelFormat::Yuv444p
    )
}
