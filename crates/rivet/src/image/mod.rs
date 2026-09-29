//! Still images: the web's picture formats, in and out.
//!
//! A page is mostly pictures, and they are web media in exactly the sense the
//! video path is: they have to arrive in a format every browser decodes, at the
//! sizes the layout asks for, in the right colours, and with nothing in them the
//! uploader did not mean to publish. This module is that path for stills:
//!
//! - **In** ([`SourceFormat`]): what people upload — JPEG, PNG, WebP, AVIF, GIF
//!   (its first frame), TIFF, BMP, and HEIC/HEIF (what an iPhone takes). Or a
//!   video, from which [`FrameSelection`] picks the stills.
//! - **Out** ([`ImageFormat`]): AVIF (AV1, royalty-free, the smallest), WebP
//!   (lossy or lossless), JPEG and PNG — the formats `<picture>` and `srcset`
//!   are built from.
//! - **Sizes**: any number of renditions, each a box the picture is fitted into
//!   exactly as a video rung is ([`crate::fit`]: contain / cover / pad /
//!   stretch, the box turning with the picture, never enlarged unless asked),
//!   but on a one-pixel grid rather than video's two.
//!
//! What every output gets, whatever was asked:
//!
//! - **Upright.** A JPEG's EXIF orientation, and a HEIF's `irot` / `imir`, are
//!   applied to the pixels; no output carries an orientation of its own.
//! - **No metadata.** Outputs are encoded from pixels, so EXIF, XMP, GPS
//!   positions, camera serials and thumbnails never reach them, unless
//!   [`ImageSpec::metadata_keep`] names a category: then a fresh EXIF block
//!   holding only that is written. The colour profile can be kept too
//!   ([`ImageSpec::keep_icc`]).
//! - **sRGB.** A source tagged with another colour space (an ICC profile —
//!   Display P3 from a phone, Adobe RGB from a camera — or a HEIF/AVIF `nclx`)
//!   is converted to sRGB, which is what a browser assumes of an untagged
//!   picture. With `keep_icc` the pixels stay as they are and the profile goes
//!   into the output instead (PNG, JPEG and WebP carry one; AVIF output is
//!   always converted, since the encoder here writes no ICC).
//!
//! # HEIC, and why it is behind a switch
//!
//! HEIC is an HEVC picture in a HEIF box structure. Decoding it is decoding
//! HEVC, with the same patent position as decoding an HEVC video, and it runs
//! through the same decoder dispatch (the GPU's HEVC decoder where there is
//! one, rivet's own software HEVC decoder where there is not).
//! `image-decode-deny=heic` refuses it — for a deployment that does not decode
//! HEVC — as `audio-decode-deny` refuses an audio codec: said up front, in
//! words a caller can match, never silently skipped. AVIF is read the same way
//! through the AV1 decoders.

mod colour;
mod decode;
mod encode;
mod heif;
mod scale;
#[cfg(test)]
mod tests;

use std::fmt;

use anyhow::{Context, Result, bail};
use bytes::Bytes;

use crate::fit::{Fit, Orientation};

pub use decode::MAX_SOURCE_PIXELS;

/// An output image format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ImageFormat {
    /// AV1 in HEIF. The smallest of the four at a given quality, and
    /// royalty-free. Safari 16+ (iOS 16+), Chrome, Firefox, Edge.
    Avif,
    /// Lossy (VP8) or lossless (VP8L). Every current browser.
    Webp,
    /// Baseline-compatible progressive JPEG, 4:2:0. Everything, everywhere.
    Jpeg,
    /// Lossless. Everything, everywhere.
    Png,
}

impl ImageFormat {
    pub const ALL: [ImageFormat; 4] = [ImageFormat::Avif, ImageFormat::Webp, ImageFormat::Jpeg, ImageFormat::Png];

    /// Read `avif`, `webp`, `jpeg` (or `jpg`) or `png`.
    pub fn parse(s: &str) -> Result<Self> {
        Ok(match s.trim().to_ascii_lowercase().as_str() {
            "avif" => ImageFormat::Avif,
            "webp" => ImageFormat::Webp,
            "jpeg" | "jpg" => ImageFormat::Jpeg,
            "png" => ImageFormat::Png,
            other => bail!("image format must be avif, webp, jpeg or png (got '{other}')"),
        })
    }

    /// The name [`ImageFormat::parse`] reads.
    pub fn as_str(self) -> &'static str {
        match self {
            ImageFormat::Avif => "avif",
            ImageFormat::Webp => "webp",
            ImageFormat::Jpeg => "jpeg",
            ImageFormat::Png => "png",
        }
    }

    /// The file extension.
    pub fn extension(self) -> &'static str {
        match self {
            ImageFormat::Avif => "avif",
            ImageFormat::Webp => "webp",
            ImageFormat::Jpeg => "jpg",
            ImageFormat::Png => "png",
        }
    }

    /// The media type.
    pub fn content_type(self) -> &'static str {
        match self {
            ImageFormat::Avif => "image/avif",
            ImageFormat::Webp => "image/webp",
            ImageFormat::Jpeg => "image/jpeg",
            ImageFormat::Png => "image/png",
        }
    }

    /// Whether `quality` means anything to it. WebP is lossy unless asked
    /// otherwise; PNG never is.
    pub fn is_lossy(self) -> bool {
        self != ImageFormat::Png
    }

    /// Whether it can carry transparency. A transparent source made into a
    /// JPEG is flattened onto white.
    pub fn has_alpha(self) -> bool {
        self != ImageFormat::Jpeg
    }

    /// The quality used when none is asked for, on the 1–100 scale every
    /// lossy format here reads: chosen so each lands near the same visual
    /// quality on photographic content (AVIF's scale runs lower for the same
    /// picture).
    pub fn default_quality(self) -> u8 {
        match self {
            ImageFormat::Avif => 60,
            ImageFormat::Webp => 80,
            ImageFormat::Jpeg => 82,
            ImageFormat::Png => 100,
        }
    }

    /// The largest side the format can hold.
    pub fn max_side(self) -> u32 {
        match self {
            // VP8 / VP8L carry 14-bit dimensions.
            ImageFormat::Webp => 16_383,
            // SOF carries 16-bit dimensions.
            ImageFormat::Jpeg => 65_535,
            ImageFormat::Avif | ImageFormat::Png => MAX_OUTPUT_SIDE,
        }
    }
}

impl fmt::Display for ImageFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The largest side any output may have.
pub const MAX_OUTPUT_SIDE: u32 = 16_384;

/// A still-image input format, as sniffed from its first bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SourceFormat {
    Jpeg,
    Png,
    Webp,
    /// AV1 in HEIF, decoded by the AV1 decoders.
    Avif,
    /// The first frame of a GIF.
    Gif,
    Tiff,
    Bmp,
    /// HEVC in HEIF (`.heic`, `.heif`), decoded by the HEVC decoders.
    Heic,
}

impl SourceFormat {
    pub const ALL: [SourceFormat; 8] = [
        SourceFormat::Jpeg,
        SourceFormat::Png,
        SourceFormat::Webp,
        SourceFormat::Avif,
        SourceFormat::Gif,
        SourceFormat::Tiff,
        SourceFormat::Bmp,
        SourceFormat::Heic,
    ];

    /// The container label [`crate::probe`] reports: `jpeg`, `png`, `webp`,
    /// `avif`, `gif`, `tiff`, `bmp`, `heic`.
    pub fn label(self) -> &'static str {
        match self {
            SourceFormat::Jpeg => "jpeg",
            SourceFormat::Png => "png",
            SourceFormat::Webp => "webp",
            SourceFormat::Avif => "avif",
            SourceFormat::Gif => "gif",
            SourceFormat::Tiff => "tiff",
            SourceFormat::Bmp => "bmp",
            SourceFormat::Heic => "heic",
        }
    }

    /// What the picture is coded with, as [`crate::probe`] reports it in
    /// `video_codec`: the format's own name, except that AVIF is `av1` and
    /// HEIC is `hevc` — the codecs their decoders, and their patent position,
    /// are those of.
    pub fn coded_label(self) -> &'static str {
        match self {
            SourceFormat::Avif => "av1",
            SourceFormat::Heic => "hevc",
            other => other.label(),
        }
    }

    /// Read a [`label`](Self::label), or `heif` for HEIC.
    pub fn parse(s: &str) -> Result<Self> {
        let s = s.trim().to_ascii_lowercase();
        if s == "heif" {
            return Ok(SourceFormat::Heic);
        }
        if s == "jpg" {
            return Ok(SourceFormat::Jpeg);
        }
        SourceFormat::ALL.into_iter().find(|f| f.label() == s).with_context(|| {
            format!("image format must be one of jpeg, png, webp, avif, gif, tiff, bmp, heic (got '{s}')")
        })
    }

    /// Whether a container label from [`crate::probe`] names a still image.
    pub fn is_image_container(label: &str) -> bool {
        SourceFormat::ALL.iter().any(|f| f.label() == label)
    }
}

impl fmt::Display for SourceFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

/// Which still-image inputs may not be decoded (`image-decode-deny=heic`).
/// Empty restricts nothing.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ImageDecodeDeny(pub Vec<SourceFormat>);

impl ImageDecodeDeny {
    /// Read a comma-separated list of [`SourceFormat`] labels. Empty, or
    /// `none`, denies nothing.
    pub fn parse(s: &str) -> Result<Self> {
        let mut formats = Vec::new();
        for word in s.split(',').map(str::trim).filter(|w| !w.is_empty() && *w != "none") {
            let format = SourceFormat::parse(word).context("image-decode-deny")?;
            if !formats.contains(&format) {
                formats.push(format);
            }
        }
        Ok(Self(formats))
    }

    pub fn denies(&self, format: SourceFormat) -> bool {
        self.0.contains(&format)
    }

    /// The refusal for a denied source. The wording is the contract: a
    /// caller that maps refusals onto its own words matches on
    /// `image-decode-deny`.
    fn refusal(format: SourceFormat) -> anyhow::Error {
        anyhow::anyhow!("decoding {} images is denied by the image-decode-deny setting", format.label())
    }
}

/// Which stills a video input gives.
#[derive(Debug, Clone, PartialEq)]
pub enum FrameSelection {
    /// One frame, 10% of the way in (past most intros and fade-ins): a
    /// poster.
    Poster,
    /// The frames at these times, in seconds from the start.
    At(Vec<f64>),
    /// This many frames, evenly spaced through the video: the middles of
    /// `count` equal slices, so neither the first frame (often black) nor the
    /// last is one of them.
    Count(u32),
}

/// The most stills one video may give.
pub const MAX_FRAMES: usize = 1000;

/// One requested output size: a box, fitted as [`crate::fit`] describes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageRendition {
    pub width: u32,
    pub height: u32,
    /// This rendition's own fit, over [`ImageSpec::fit`].
    pub fit: Option<Fit>,
    pub orientation: Option<Orientation>,
    pub upscale: Option<bool>,
}

impl ImageRendition {
    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height, fit: None, orientation: None, upscale: None }
    }
}

/// What an image job makes.
#[derive(Debug, Clone, PartialEq)]
pub struct ImageSpec {
    /// Every rendition is made in each of these, in this order.
    pub formats: Vec<ImageFormat>,
    /// 1–100 for the lossy formats; `None` is each format's
    /// [`default_quality`](ImageFormat::default_quality).
    pub quality: Option<u8>,
    /// WebP lossless (VP8L) rather than lossy. Only with WebP and PNG, the
    /// formats that have a lossless form.
    pub lossless: bool,
    /// Keep the source's colour profile rather than converting to sRGB. See
    /// the [module docs](self).
    pub keep_icc: bool,
    /// AVIF encoder effort, 1 (slowest, smallest) to 10 (fastest).
    pub speed: u8,
    /// Output boxes; none is one output at the source's size.
    pub renditions: Vec<ImageRendition>,
    pub fit: Fit,
    pub orientation: Orientation,
    pub upscale: bool,
    /// Which stills a video input gives. `None` is [`FrameSelection::Poster`]
    /// for a video; for a still image anything else is refused.
    pub frames: Option<FrameSelection>,
    pub decode_deny: ImageDecodeDeny,
    /// Identifying source metadata to carry into every output as EXIF
    /// (`metadata-keep`): location, device, capture time, descriptive tags.
    /// None by default. A still from a video takes the video's.
    pub metadata_keep: container::metadata::Keep,
}

/// The AVIF effort used when none is asked for: a little slower than the
/// thumbnail path's 8, since a still is the job here rather than a side
/// product of one.
pub const DEFAULT_AVIF_SPEED: u8 = 6;

impl Default for ImageSpec {
    fn default() -> Self {
        Self {
            formats: vec![ImageFormat::Avif],
            quality: None,
            lossless: false,
            keep_icc: false,
            speed: DEFAULT_AVIF_SPEED,
            renditions: Vec::new(),
            fit: Fit::Contain,
            orientation: Orientation::Auto,
            upscale: false,
            frames: None,
            decode_deny: ImageDecodeDeny::default(),
            metadata_keep: container::metadata::Keep::NONE,
        }
    }
}

impl ImageSpec {
    /// Check the spec on its own, before any input is read.
    pub fn validate(&self) -> Result<()> {
        if self.formats.is_empty() {
            bail!("invalid output spec: an image job needs at least one format");
        }
        for (i, f) in self.formats.iter().enumerate() {
            if self.formats[..i].contains(f) {
                bail!("invalid output spec: image format {f} is listed twice");
            }
        }
        if let Some(q) = self.quality {
            if !(1..=100).contains(&q) {
                bail!("invalid output spec: image quality must be between 1 and 100 (got {q})");
            }
            let lossy = self.formats.iter().any(|f| f.is_lossy() && !(self.lossless && *f == ImageFormat::Webp));
            if !lossy {
                bail!("invalid output spec: image quality applies to lossy formats (avif, webp, jpeg), and none is being made");
            }
        }
        if self.lossless {
            if let Some(f) = self.formats.iter().find(|f| !matches!(f, ImageFormat::Webp | ImageFormat::Png)) {
                bail!("invalid output spec: lossless applies to webp (png is always lossless); {f} has no lossless form here");
            }
        }
        if !(1..=10).contains(&self.speed) {
            bail!("invalid output spec: image speed must be between 1 and 10 (got {})", self.speed);
        }
        for r in &self.renditions {
            if r.width == 0 || r.height == 0 || r.width > MAX_OUTPUT_SIDE || r.height > MAX_OUTPUT_SIDE {
                bail!(
                    "invalid output spec: an image rendition is between 1 and {MAX_OUTPUT_SIDE} on each side (got {}x{})",
                    r.width,
                    r.height
                );
            }
        }
        match &self.frames {
            Some(FrameSelection::At(times)) => {
                if times.is_empty() || times.len() > MAX_FRAMES {
                    bail!("invalid output spec: frames-at takes between 1 and {MAX_FRAMES} times");
                }
                if let Some(t) = times.iter().find(|t| !t.is_finite() || **t < 0.0) {
                    bail!("invalid output spec: frames-at times are seconds from the start, zero or more (got {t})");
                }
            }
            Some(FrameSelection::Count(n)) if *n == 0 || *n as usize > MAX_FRAMES => {
                bail!("invalid output spec: frames-count must be between 1 and {MAX_FRAMES} (got {n})");
            }
            _ => {}
        }
        Ok(())
    }

    fn quality_for(&self, format: ImageFormat) -> u8 {
        self.quality.unwrap_or_else(|| format.default_quality())
    }
}

/// One encoded output.
#[derive(Debug, Clone)]
pub struct ImageArtifact {
    /// Which of [`ImageSpec::renditions`] this is, by position; `0` for a job
    /// with none (one output at the source's size).
    pub rendition: usize,
    /// `WxH` of the output. Unique within a job, per frame and format: a
    /// second rendition coming out the same size another way gets `-2`.
    pub label: String,
    pub format: ImageFormat,
    pub width: u32,
    pub height: u32,
    /// For a still from a video: its position in the selection (0-based) and
    /// its time in seconds.
    pub frame: Option<(usize, f64)>,
    pub bytes: Vec<u8>,
}

impl ImageArtifact {
    /// `<label>.<ext>`, or `<label>-<nnn>.<ext>` (1-based) when the job took
    /// several frames.
    pub fn file_name(&self, several_frames: bool) -> String {
        match self.frame {
            Some((i, _)) if several_frames => format!("{}-{:03}.{}", self.label, i + 1, self.format.extension()),
            _ => format!("{}.{}", self.label, self.format.extension()),
        }
    }
}

/// A rendition that came out the same as an earlier one and was not made.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MergedRendition {
    /// Its position in [`ImageSpec::renditions`].
    pub rendition: usize,
    /// The earlier rendition it came out the same as.
    pub same_as: usize,
    pub output: (u32, u32),
}

/// What an image job read and made.
#[derive(Debug, Clone)]
pub struct ImageJobOutput {
    /// The input, as [`crate::probe`] labels it: an image format
    /// ([`SourceFormat::label`]), or the video's container.
    pub source_container: String,
    /// What was decoded: [`SourceFormat::coded_label`] for a still image
    /// (`av1` for AVIF, `hevc` for HEIC), the video codec for a video.
    pub decoded: String,
    /// Whether the input was a video the stills were taken from.
    pub from_video: bool,
    /// Whether more than one frame was taken, which is what names the files
    /// ([`ImageArtifact::file_name`]).
    pub several_frames: bool,
    pub artifacts: Vec<ImageArtifact>,
    pub merged: Vec<MergedRendition>,
}

/// Sniff a still image from its first bytes. `None` is not an image this
/// module reads — possibly a video.
pub fn sniff(data: &[u8]) -> Option<SourceFormat> {
    decode::sniff(data)
}

/// Probe a still image: `Ok(None)` when `data` is not one. The picture's
/// size is its upright size — EXIF orientation and HEIF `irot` applied — as
/// every output is sized against it.
pub fn probe(data: &[u8]) -> Result<Option<crate::probe::MediaInfo>> {
    let Some(format) = sniff(data) else {
        return Ok(None);
    };
    let header = decode::read_header(data, format).with_context(|| format!("reading the {format} header"))?;
    Ok(Some(crate::probe::MediaInfo {
        container: format.label().to_string(),
        video_codec: format.coded_label().to_string(),
        width: header.width,
        height: header.height,
        stored_width: header.stored_width,
        stored_height: header.stored_height,
        rotation_degrees: 0,
        sample_aspect: (1, 1),
        frame_rate: 0.0,
        duration: 0.0,
        pixel_format: header.pixel_format,
        audio: None,
        subtitles: Vec::new(),
    }))
}

/// Run an image job: decode `input` (a still image, or stills from a video),
/// and make every rendition in every format.
///
/// CPU-bound and blocking; call it from a blocking context
/// (`tokio::task::spawn_blocking`).
pub fn run_image_job(input: &Bytes, spec: &ImageSpec) -> Result<ImageJobOutput> {
    spec.validate()?;
    let (pictures, source_container, decoded, from_video) = match sniff(input) {
        Some(format) => {
            if spec.frames.is_some() {
                bail!(
                    "invalid output spec: frames pick stills from a video, and the input is a {} image",
                    format.label()
                );
            }
            if spec.decode_deny.denies(format) {
                return Err(ImageDecodeDeny::refusal(format));
            }
            let picture = decode::decode(input, format).with_context(|| format!("decoding the {format} image"))?;
            (vec![(None, picture)], format.label().to_string(), format.coded_label().to_string(), false)
        }
        None => {
            let container = container::sniff_container(input);
            if !container.is_known() {
                bail!("unrecognised container: the input is neither an image this service reads nor a video");
            }
            let selection = spec.frames.clone().unwrap_or(FrameSelection::Poster);
            let (codec, stills) = decode::video_stills(input, &selection)?;
            let pictures = stills.into_iter().map(|(i, t, p)| (Some((i, t)), p)).collect();
            (pictures, container.label().to_string(), codec, true)
        }
    };
    let several_frames = pictures.len() > 1;
    // What is kept, once, as the EXIF block every output gets; none unless
    // asked, and then only what was asked.
    let exif = if spec.metadata_keep.is_empty() {
        None
    } else {
        container::metadata::exif::build(&container::metadata::read(input).kept(spec.metadata_keep))
    };

    let mut artifacts = Vec::new();
    let mut merged = Vec::new();
    for (frame, picture) in pictures {
        let (planned, frame_merged) = scale::plan(&picture, spec);
        if frame.is_none_or(|(i, _)| i == 0) {
            merged = frame_merged;
        }
        let prepared = colour::Prepared::new(picture, spec.keep_icc, &spec.formats)?;
        for plan in planned {
            for &format in &spec.formats {
                let pixels = scale::apply(prepared.for_format(format), &plan, format.has_alpha())?;
                let (w, h) = (pixels.image.width(), pixels.image.height());
                if w > format.max_side() || h > format.max_side() {
                    bail!(
                        "invalid output spec: {format} holds at most {} pixels a side, and this output is {w}x{h}",
                        format.max_side()
                    );
                }
                let mut bytes = encode::encode(&pixels, format, spec.quality_for(format), spec.lossless, spec.speed)
                    .with_context(|| format!("encoding the {w}x{h} {format}"))?;
                if let Some(exif) = &exif {
                    bytes = container::metadata::write::still(&bytes, exif, w, h)
                        .with_context(|| format!("writing the kept metadata into the {format}"))?;
                }
                artifacts.push(ImageArtifact {
                    rendition: plan.rendition,
                    label: plan.label.clone(),
                    format,
                    width: w,
                    height: h,
                    frame,
                    bytes,
                });
            }
        }
    }
    Ok(ImageJobOutput { source_container, decoded, from_video, several_frames, artifacts, merged })
}
