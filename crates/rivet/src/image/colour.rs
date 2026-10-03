//! Colour: what a source's pixels mean, and turning them into sRGB.
//!
//! A browser shows an untagged picture as sRGB. A source tagged with anything
//! else — an ICC profile (a phone's Display P3, a camera's Adobe RGB) or an
//! H.273 description in a HEIF/AVIF `nclx` — is converted here, with moxcms,
//! before it is scaled and encoded. With `keep_icc` the pixels are left alone
//! and the profile travels into the outputs that can carry one; AVIF output is
//! converted all the same, since its encoder here writes no ICC.

use anyhow::Result;
use super::raster::RgbaImage;
use moxcms::{
    CicpColorPrimaries, CicpProfile, ColorProfile, Layout, MatrixCoefficients, TransferCharacteristics,
    TransformOptions,
};

use super::ImageFormat;
use super::decode::Picture;

/// How a source says what its pixels mean.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Profile {
    /// An embedded ICC profile.
    Icc(Vec<u8>),
    /// An H.273 `colour_primaries` / `transfer_characteristics` pair, from a
    /// HEIF `nclx` or a coded stream's own colour description.
    Cicp { primaries: u8, transfer: u8 },
}

impl Profile {
    /// Whether this already is sRGB, or close enough that converting would
    /// only add rounding: BT.709 / sRGB primaries (or unspecified) with the
    /// sRGB, BT.709, BT.601 or unspecified transfer — the curves an sRGB
    /// display is assumed to apply to video-range content anyway.
    fn is_srgb(&self) -> bool {
        match self {
            Profile::Cicp { primaries, transfer } => {
                matches!(primaries, 1 | 2) && matches!(transfer, 1 | 2 | 6 | 13 | 14 | 15)
            }
            Profile::Icc(_) => false,
        }
    }

    fn to_moxcms(&self) -> Option<ColorProfile> {
        match self {
            Profile::Icc(bytes) => ColorProfile::new_from_slice(bytes).ok(),
            Profile::Cicp { primaries, transfer } => {
                let primaries = CicpColorPrimaries::try_from(*primaries).ok()?;
                let transfer = TransferCharacteristics::try_from(*transfer).ok()?;
                Some(ColorProfile::new_from_cicp(CicpProfile {
                    color_primaries: primaries,
                    transfer_characteristics: transfer,
                    matrix_coefficients: MatrixCoefficients::Identity,
                    full_range: true,
                }))
            }
        }
    }

    /// The bytes of an ICC profile saying the same thing, for an output that
    /// keeps it.
    fn icc_bytes(&self) -> Option<Vec<u8>> {
        match self {
            Profile::Icc(bytes) => Some(bytes.clone()),
            Profile::Cicp { .. } => self.to_moxcms()?.encode().ok(),
        }
    }
}

/// A decoded picture, ready for each output format: as decoded (with the
/// profile to embed) for the outputs that keep it, and in sRGB for the rest.
pub(crate) struct Prepared {
    kept: Option<(Picture, Option<Vec<u8>>)>,
    srgb: Option<Picture>,
}

/// The pixels an output is made from, and the ICC profile to embed with
/// them.
pub(crate) struct Source<'a> {
    pub(crate) picture: &'a Picture,
    pub(crate) icc: Option<&'a [u8]>,
}

impl Prepared {
    /// Prepare `picture` for every format in `formats`: converted to sRGB
    /// unless `keep_icc` (and then still for AVIF).
    pub(crate) fn new(picture: Picture, keep_icc: bool, formats: &[ImageFormat]) -> Result<Self> {
        let tagged = picture.profile.as_ref().is_some_and(|p| !p.is_srgb());
        if !tagged {
            let mut picture = picture;
            picture.profile = None;
            return Ok(Self { kept: None, srgb: Some(picture) });
        }
        let needs_srgb = !keep_icc || formats.contains(&ImageFormat::Avif);
        let needs_kept = keep_icc && formats.iter().any(|f| *f != ImageFormat::Avif);
        let srgb = needs_srgb.then(|| to_srgb(&picture));
        let kept = needs_kept.then(|| {
            let icc = picture.profile.as_ref().and_then(Profile::icc_bytes);
            (picture, icc)
        });
        Ok(Self { kept, srgb })
    }

    pub(crate) fn for_format(&self, format: ImageFormat) -> Source<'_> {
        match (&self.kept, &self.srgb) {
            (Some((picture, icc)), _) if format != ImageFormat::Avif => {
                Source { picture, icc: icc.as_deref() }
            }
            (_, Some(picture)) => Source { picture, icc: None },
            (Some((picture, icc)), None) => Source { picture, icc: icc.as_deref() },
            (None, None) => unreachable!("Prepared always holds one of the two"),
        }
    }
}

/// `picture` converted from its profile to sRGB. A profile moxcms cannot read
/// or build a transform from (a CMYK or greyscale ICC on RGB pixels, a
/// corrupt one) leaves the pixels as they are, as a browser would show them:
/// a warning, not a failed job.
fn to_srgb(picture: &Picture) -> Picture {
    let mut out = Picture { profile: None, ..picture.clone() };
    let Some(profile) = picture.profile.as_ref() else {
        return out;
    };
    let transform = profile.to_moxcms().and_then(|src| {
        src.create_transform_8bit(Layout::Rgba, &ColorProfile::new_srgb(), Layout::Rgba, TransformOptions::default())
            .ok()
    });
    let Some(transform) = transform else {
        tracing::warn!("the source's colour profile could not be read; its pixels are used as they are");
        return out;
    };
    let mut converted = vec![0u8; picture.rgba.as_raw().len()];
    if let Err(e) = transform.transform(picture.rgba.as_raw(), &mut converted) {
        tracing::warn!(error = %e, "converting to sRGB failed; the source's pixels are used as they are");
        return out;
    }
    out.rgba = RgbaImage::from_raw(picture.rgba.width(), picture.rgba.height(), converted)
        .expect("the same size as the source");
    out
}
