//! Optional input-policy validation.
//!
//! These are **advisory** helpers — the job engine ([`crate::job::run_job`])
//! does *not* enforce them, so rivet transcodes whatever it's given. They
//! exist so policy-bearing callers (e.g. a hosted service) can gate uploads
//! with the same limits the reference transcoder uses.

use codec::frame::{PixelFormat, StreamInfo};
use container::{ContainerKind, sniff_container};

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
    /// The bytes open with no container this crate demuxes.
    UnrecognizedContainer,
    /// A container this crate demuxes, but not one the caller accepts.
    UnsupportedContainer,
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

/// Gate an upload on its container, by structure and not by name.
///
/// Sniffs the first bytes ([`sniff_container`]) and accepts the file when its
/// family is in `accepted`. The sniff recognises ISO BMFF by the box that
/// opens the file, never by the `ftyp` major brand — the brand space is open,
/// and a brand a list has not heard of is not evidence of anything; a real
/// recording with major brand `nvr1` was once refused as "unrecognized" on
/// exactly that basis while the demuxer would have read it fine. Whatever is
/// ISO BMFF goes on to the demuxer, which reads what is in the file.
///
/// Returns the detected family so a caller can name it in its own message.
pub fn validate_container(
    data: &[u8],
    accepted: &[ContainerKind],
) -> Result<ContainerKind, ValidationError> {
    if data.len() < 12 {
        return Err(ValidationError {
            kind: ValidationErrorKind::UnrecognizedContainer,
            message: "Uploaded file is too small to be a valid video container.".into(),
        });
    }
    let kind = sniff_container(data);
    if kind == ContainerKind::Unknown {
        return Err(ValidationError {
            kind: ValidationErrorKind::UnrecognizedContainer,
            message: "Unrecognized container format.".into(),
        });
    }
    if !accepted.contains(&kind) {
        return Err(ValidationError {
            kind: ValidationErrorKind::UnsupportedContainer,
            message: format!("{} container is not supported.", kind.label()),
        });
    }
    Ok(kind)
}

/// Whether a source pixel format needs the per-frame 4:4:4 → 4:2:0 chroma
/// downsample before encode. The engine consults this to set up the pump.
pub fn needs_chroma_downsample(format: PixelFormat) -> bool {
    matches!(
        format,
        PixelFormat::Yuv444p10le | PixelFormat::Yuva444p10le | PixelFormat::Yuv444p
    )
}

#[cfg(test)]
mod container_tests {
    use super::*;

    const WEB: &[ContainerKind] = &[ContainerKind::IsoBmff, ContainerKind::Matroska];

    #[test]
    fn an_unfamiliar_ftyp_brand_passes_a_structural_gate() {
        let mut data = vec![0x00, 0x00, 0x00, 0x20];
        data.extend_from_slice(b"ftypnvr1");
        data.extend_from_slice(&[0, 0, 0, 0]);
        data.extend_from_slice(b"isommp42");
        assert_eq!(validate_container(&data, WEB).unwrap(), ContainerKind::IsoBmff);
    }

    #[test]
    fn a_known_but_unaccepted_container_says_which() {
        let mut ts = vec![0u8; 190];
        ts[0] = 0x47;
        ts[188] = 0x47;
        let err = validate_container(&ts, WEB).unwrap_err();
        assert_eq!(err.kind, ValidationErrorKind::UnsupportedContainer);
        assert!(err.message.contains("ts"), "{}", err.message);
    }

    #[test]
    fn junk_and_short_input_are_unrecognized() {
        assert_eq!(
            validate_container(b"hello, this is plain text, not a video", WEB).unwrap_err().kind,
            ValidationErrorKind::UnrecognizedContainer
        );
        assert_eq!(
            validate_container(&[0u8; 4], WEB).unwrap_err().kind,
            ValidationErrorKind::UnrecognizedContainer
        );
    }
}
