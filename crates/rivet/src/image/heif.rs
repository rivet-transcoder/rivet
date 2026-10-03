//! HEIF still images (ISO/IEC 23008-12): AVIF (AV1) and HEIC (HEVC).
//!
//! A HEIF file is ISO BMFF with no movie: a `meta` box listing *items* — coded
//! pictures, derived pictures, metadata — with a primary one (`pitm`), where
//! each item's bytes are (`iloc`), and properties (`iprp`) associated with each
//! item: its size (`ispe`), its decoder configuration (`hvcC` / `av1C`), its
//! colour (`colr`), and transformations (`irot`, `imir`). This reads exactly
//! that much, and hands the coded items to the decoders the video path uses —
//! the HEVC dispatch for HEIC, the AV1 dispatch for AVIF.
//!
//! Handled, because real files have them:
//!
//! - **Grids.** A phone does not code a 12 MP photo as one picture: an iPhone
//!   HEIC is a `grid` item of 512x512 tiles, each its own coded item (`dimg`
//!   references, in raster order), cropped to the grid's declared size.
//! - **Alpha.** An auxiliary item (`auxl` reference, `auxC` naming the alpha
//!   URN) whose luma is the primary's transparency.
//! - **Orientation.** `irot` (anticlockwise quarter turns) and `imir`, applied
//!   in the order they are associated. HEIF's orientation is these, never EXIF.
//! - **Colour.** `colr` as `nclx` (the matrix to read the samples with, the
//!   range, and the primaries and transfer that make the pixels' meaning) or as
//!   an ICC profile; lacking an `nclx`, HEVC's own SPS VUI.
//!
//! Not handled, and refused by name: derived items other than `grid` (`iden`,
//! `iovl`), and any other coding (JPEG-in-HEIF, uncompressed).

use std::borrow::Cow;

use anyhow::{Context, Result, anyhow, bail};
use codec::frame::{ColorSpace, PixelFormat, StreamInfo, VideoFrame};

use super::SourceFormat;
use super::colour::Profile;
use super::decode::{Picture, check_size};
use super::raster::RgbaImage;
use crate::thumbnail::{SourceColor, YuvMatrix};

/// The auxiliary types that mark an item as its master's alpha plane: MPEG's
/// generic one (what AVIF uses) and HEVC's.
const ALPHA_URNS: [&str; 2] = ["urn:mpeg:mpegB:cicp:systems:auxiliary:alpha", "urn:mpeg:hevc:2015:auxid:1"];

/// Sniff a HEIF still: an `ftyp` naming a HEIF brand, and a top-level `meta`.
/// Which of AVIF and HEIC it is follows the primary item's coding when it can
/// be read, else the brands.
pub(crate) fn sniff(data: &[u8]) -> Option<SourceFormat> {
    if data.len() < 16 || &data[4..8] != b"ftyp" {
        return None;
    }
    let mut ftyp = None;
    let mut has_meta = false;
    for b in boxes(data) {
        let Ok(b) = b else { break };
        match &b.kind {
            b"ftyp" => ftyp = Some(b.body),
            b"meta" => has_meta = true,
            _ => {}
        }
    }
    let ftyp = ftyp?;
    if !has_meta || ftyp.len() < 8 {
        return None;
    }
    let brands: Vec<&[u8]> = std::iter::once(&ftyp[..4]).chain(ftyp[8..].chunks_exact(4)).collect();
    let has = |names: &[&[u8; 4]]| brands.iter().any(|b| names.iter().any(|n| *b == &n[..]));
    let avif = has(&[b"avif", b"avis"]);
    let heic = has(&[b"heic", b"heix", b"heim", b"heis", b"hevc", b"hevx"]);
    if !avif && !heic && !has(&[b"mif1", b"msf1"]) {
        return None;
    }
    let by_coding = Heif::parse(data).ok().and_then(|h| h.coding(h.primary).ok()).map(|c| match c {
        Coding::Av1 => SourceFormat::Avif,
        Coding::Hevc => SourceFormat::Heic,
    });
    Some(by_coding.unwrap_or(if avif { SourceFormat::Avif } else { SourceFormat::Heic }))
}

/// What [`super::probe`] reports.
pub(crate) struct Header {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) stored_width: u32,
    pub(crate) stored_height: u32,
    pub(crate) pixel_format: String,
}

pub(crate) fn read_header(data: &[u8]) -> Result<Header> {
    let heif = Heif::parse(data)?;
    let (w, h) = heif.ispe(heif.primary).context("the primary item has no size (ispe)")?;
    let turned = heif.props(heif.primary).filter(|p| matches!(p, Property::Irot(1 | 3))).count() % 2 == 1;
    let first_coded = heif.coded_items(heif.primary)?.first().copied().unwrap_or(heif.primary);
    let pixel_format = heif.config(first_coded).map(|c| format!("{:?}", c.pixel_format)).unwrap_or_default();
    Ok(Header {
        width: if turned { h } else { w },
        height: if turned { w } else { h },
        stored_width: w,
        stored_height: h,
        pixel_format,
    })
}

/// Decode the primary picture, with its alpha, turned upright.
pub(crate) fn decode(data: &[u8], format: SourceFormat) -> Result<Picture> {
    let heif = Heif::parse(data)?;
    let primary = heif.primary;
    let (w, h) = heif.ispe(primary).context("the primary item has no size (ispe)")?;
    check_size(w, h)?;
    let coding = heif.coding(primary)?;
    let expected = match format {
        SourceFormat::Avif => Coding::Av1,
        _ => Coding::Hevc,
    };
    if coding != expected {
        tracing::debug!(?coding, %format, "the HEIF brands and the primary item's coding disagree; going by the coding");
    }

    let colour = heif.colour(primary);
    let mut rgba = heif.decode_item(primary, Plane::Colour(colour.matrix))?;

    if let Some(alpha) = heif.alpha_item(primary) {
        match heif.decode_item(alpha, Plane::Luma) {
            Ok(a) if a.dimensions() == rgba.dimensions() => {
                for (px, a) in rgba.pixels_mut().zip(a.pixels()) {
                    px[3] = a[0];
                }
            }
            Ok(a) => tracing::warn!(
                alpha = ?a.dimensions(),
                picture = ?rgba.dimensions(),
                "the alpha plane is not the picture's size; the picture is used opaque"
            ),
            Err(e) => tracing::warn!(error = %e, "the alpha plane could not be decoded; the picture is used opaque"),
        }
    }

    for p in heif.props(primary) {
        rgba = match p {
            // Anticlockwise quarter turns.
            Property::Irot(1) => rgba.rotate270(),
            Property::Irot(2) => rgba.rotate180(),
            Property::Irot(3) => rgba.rotate90(),
            // Axis 0 is vertical: left and right swap.
            Property::Imir(0) => rgba.flip_horizontal(),
            Property::Imir(_) => rgba.flip_vertical(),
            _ => continue,
        };
    }

    Ok(Picture::new(rgba, colour.profile))
}

/// How an item was coded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Coding {
    Av1,
    Hevc,
}

impl Coding {
    fn decoder_name(self) -> &'static str {
        match self {
            Coding::Av1 => "av1",
            Coding::Hevc => "hevc",
        }
    }
}

/// What to make of a decoded item.
#[derive(Debug, Clone, Copy)]
enum Plane {
    /// RGB, read with this matrix (`None`: the codec's own, else BT.601).
    Colour(Option<Matrix>),
    /// The luma alone, for an alpha plane.
    Luma,
}

/// How a picture's samples become RGB.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Matrix {
    /// H.273 `matrix_coefficients`.
    coefficients: u8,
    full_range: bool,
}

/// An item's colour: how to read it, and what the result means.
struct Colour {
    matrix: Option<Matrix>,
    profile: Option<Profile>,
}

#[derive(Debug, Clone)]
enum Property<'a> {
    Ispe(u32, u32),
    HvcC(&'a [u8]),
    Av1C(&'a [u8]),
    Nclx(container::demux::Nclx),
    Icc(&'a [u8]),
    Irot(u8),
    Imir(u8),
    AuxC(String),
    Other,
}

struct Item {
    id: u32,
    kind: [u8; 4],
}

struct Location {
    /// 0: offsets into the file; 1: into `idat`.
    method: u8,
    base: u64,
    extents: Vec<(u64, u64)>,
}

struct Reference {
    kind: [u8; 4],
    from: u32,
    to: Vec<u32>,
}

struct Heif<'a> {
    data: &'a [u8],
    primary: u32,
    items: Vec<Item>,
    locations: Vec<(u32, Location)>,
    properties: Vec<Property<'a>>,
    /// Item → 1-based property indices, in association order.
    associations: Vec<(u32, Vec<u16>)>,
    references: Vec<Reference>,
    idat: Option<&'a [u8]>,
}

/// A decoder configuration, as far as a decoder needs telling.
struct Config {
    coding: Coding,
    pixel_format: PixelFormat,
    /// What goes ahead of the first coded item: Annex-B parameter sets for
    /// HEVC, the configuration OBUs for AV1.
    prefix: Vec<u8>,
    /// HEVC's NAL length-field size.
    length_size: usize,
    /// HEVC's parameter sets, for reading the SPS VUI.
    parameter_sets: Vec<Vec<u8>>,
}

impl<'a> Heif<'a> {
    fn parse(data: &'a [u8]) -> Result<Self> {
        let meta = boxes(data)
            .filter_map(Result::ok)
            .find(|b| &b.kind == b"meta")
            .context("no meta box: not a HEIF still")?;
        let meta_body = full_box(meta.body)?.1;
        let mut heif = Heif {
            data,
            primary: 0,
            items: Vec::new(),
            locations: Vec::new(),
            properties: Vec::new(),
            associations: Vec::new(),
            references: Vec::new(),
            idat: None,
        };
        let mut primary = None;
        for b in boxes(meta_body) {
            let b = b?;
            match &b.kind {
                b"pitm" => {
                    let (version, body) = full_box(b.body)?;
                    primary = Some(Reader::new(body).id(version == 0)?);
                }
                b"iinf" => heif.items = parse_iinf(b.body)?,
                b"iloc" => heif.locations = parse_iloc(b.body)?,
                b"iref" => heif.references = parse_iref(b.body)?,
                b"iprp" => (heif.properties, heif.associations) = parse_iprp(b.body)?,
                b"idat" => heif.idat = Some(b.body),
                _ => {}
            }
        }
        heif.primary = primary.context("no primary item (pitm)")?;
        Ok(heif)
    }

    fn item(&self, id: u32) -> Result<&Item> {
        self.items.iter().find(|i| i.id == id).ok_or_else(|| anyhow!("HEIF item {id} is not listed"))
    }

    fn props(&self, id: u32) -> impl Iterator<Item = &Property<'a>> {
        self.associations
            .iter()
            .filter(move |(item, _)| *item == id)
            .flat_map(|(_, indices)| indices.iter())
            .filter_map(|&i| self.properties.get(usize::from(i).checked_sub(1)?))
    }

    fn ispe(&self, id: u32) -> Option<(u32, u32)> {
        self.props(id).find_map(|p| match p {
            Property::Ispe(w, h) => Some((*w, *h)),
            _ => None,
        })
    }

    fn refs(&self, kind: &[u8; 4], from: u32) -> Vec<u32> {
        self.references.iter().filter(|r| &r.kind == kind && r.from == from).flat_map(|r| r.to.iter().copied()).collect()
    }

    /// The coded items behind `id`: itself, or a grid's tiles.
    fn coded_items(&self, id: u32) -> Result<Vec<u32>> {
        match &self.item(id)?.kind {
            b"grid" => {
                let tiles = self.refs(b"dimg", id);
                if tiles.is_empty() {
                    bail!("the HEIF grid names no tiles");
                }
                Ok(tiles)
            }
            _ => Ok(vec![id]),
        }
    }

    fn coding(&self, id: u32) -> Result<Coding> {
        let coded = *self.coded_items(id)?.first().expect("never empty");
        match &self.item(coded)?.kind {
            b"av01" => Ok(Coding::Av1),
            b"hvc1" => Ok(Coding::Hevc),
            other => bail!(
                "unsupported input: the HEIF picture is coded as '{}', and only AV1 (AVIF) and HEVC (HEIC) are read",
                String::from_utf8_lossy(other)
            ),
        }
    }

    fn config(&self, id: u32) -> Result<Config> {
        for p in self.props(id) {
            match p {
                Property::HvcC(body) => return hevc_config(body),
                Property::Av1C(body) => return av1_config(body),
                _ => {}
            }
        }
        bail!("HEIF item {id} has no decoder configuration")
    }

    /// The item's bytes, from the file or from `idat`.
    fn item_data(&self, id: u32) -> Result<Cow<'a, [u8]>> {
        let (_, loc) = self.locations.iter().find(|(i, _)| *i == id).ok_or_else(|| anyhow!("HEIF item {id} has no location"))?;
        let source = match loc.method {
            0 => self.data,
            1 => self.idat.context("an item is in idat, and there is none")?,
            m => bail!("unsupported input: HEIF construction method {m}"),
        };
        let slice = |offset: u64, length: u64| -> Result<&'a [u8]> {
            let start = usize::try_from(loc.base + offset)?;
            let end = if length == 0 { source.len() } else { start.checked_add(usize::try_from(length)?).context("extent")? };
            source.get(start..end).ok_or_else(|| anyhow!("HEIF item {id} runs past the end of the file"))
        };
        match loc.extents.as_slice() {
            [] => Ok(Cow::Borrowed(&[])),
            [(offset, length)] => Ok(Cow::Borrowed(slice(*offset, *length)?)),
            extents => {
                let mut joined = Vec::new();
                for (offset, length) in extents {
                    joined.extend_from_slice(slice(*offset, *length)?);
                }
                Ok(Cow::Owned(joined))
            }
        }
    }

    /// An item's colour, from its own `colr` properties or its first tile's.
    fn colour(&self, id: u32) -> Colour {
        let mut ids = vec![id];
        if let Ok(tiles) = self.coded_items(id) {
            ids.extend(tiles.first());
        }
        let mut nclx = None;
        let mut icc = None;
        for &i in &ids {
            for p in self.props(i) {
                match p {
                    Property::Nclx(n) if nclx.is_none() => nclx = Some(*n),
                    Property::Icc(bytes) if icc.is_none() => icc = Some(bytes.to_vec()),
                    _ => {}
                }
            }
        }
        // Lacking an `nclx`, HEVC says its own colour in the SPS VUI.
        if nclx.is_none()
            && let Some(first) = ids.last()
            && let Ok(config) = self.config(*first)
            && config.coding == Coding::Hevc
        {
            nclx = container::demux::colour_from_parameter_sets("hevc", &config.parameter_sets);
        }
        let matrix = nclx.map(|n| Matrix { coefficients: n.matrix, full_range: n.full_range });
        let profile = match (icc, nclx) {
            (Some(icc), _) => Some(Profile::Icc(icc)),
            (None, Some(n)) => Some(Profile::Cicp { primaries: n.primaries, transfer: n.transfer }),
            (None, None) => None,
        };
        Colour { matrix, profile }
    }

    /// The alpha plane of `id`, if it has one.
    fn alpha_item(&self, id: u32) -> Option<u32> {
        self.references.iter().filter(|r| &r.kind == b"auxl" && r.to.contains(&id)).map(|r| r.from).find(|&aux| {
            self.props(aux).any(|p| matches!(p, Property::AuxC(urn) if ALPHA_URNS.contains(&urn.as_str())))
        })
    }

    /// Decode `id` — a coded item, or a grid of them — into one picture.
    fn decode_item(&self, id: u32, plane: Plane) -> Result<RgbaImage> {
        let tiles = self.coded_items(id)?;
        let frames = self.decode_coded(&tiles)?;
        let colour = match plane {
            Plane::Colour(Some(m)) => Plane::Colour(Some(m)),
            // No `nclx` and no VUI: AVIF's writers default to full range, and
            // an unspecified matrix on a still is BT.601, as in JPEG.
            Plane::Colour(None) => Plane::Colour(Some(Matrix { coefficients: 2, full_range: true })),
            Plane::Luma => Plane::Luma,
        };
        let tile_images = frames
            .iter()
            .zip(&tiles)
            .map(|(frame, &tile)| {
                let (w, h) = self.ispe(tile).unwrap_or((frame.width, frame.height));
                to_rgba(frame, w.min(frame.width), h.min(frame.height), colour)
            })
            .collect::<Result<Vec<_>>>()?;

        if &self.item(id)?.kind != b"grid" {
            return tile_images.into_iter().next().context("the decoder gave no picture");
        }
        let grid = self.item_data(id)?;
        let mut r = Reader::new(&grid);
        let _version = r.u8()?;
        let flags = r.u8()?;
        let rows = u32::from(r.u8()?) + 1;
        let columns = u32::from(r.u8()?) + 1;
        let (out_w, out_h) = if flags & 1 == 0 { (u32::from(r.u16()?), u32::from(r.u16()?)) } else { (r.u32()?, r.u32()?) };
        check_size(out_w, out_h)?;
        if tile_images.len() != (rows * columns) as usize {
            bail!("the HEIF grid is {columns}x{rows} and names {} tiles", tile_images.len());
        }
        let (tw, th) = tile_images[0].dimensions();
        let mut canvas = RgbaImage::new(out_w, out_h);
        for (i, tile) in tile_images.iter().enumerate() {
            let (x, y) = ((i as u32 % columns) * tw, (i as u32 / columns) * th);
            // `replace` clips at the canvas edge: the grid's right and bottom
            // tiles overhang its declared size.
            canvas.replace(tile, i64::from(x), i64::from(y));
        }
        Ok(canvas)
    }

    /// Decode each of `ids` (all one coding and configuration, as a grid's
    /// tiles are) in turn. One decoder takes them all as a stream of
    /// keyframes; if it does not give back one picture per item, each is
    /// decoded on its own instead.
    fn decode_coded(&self, ids: &[u32]) -> Result<Vec<VideoFrame>> {
        let config = self.config(ids[0])?;
        let samples = ids
            .iter()
            .enumerate()
            .map(|(i, &id)| {
                let data = self.item_data(id)?;
                let mut sample = if i == 0 { config.prefix.clone() } else { Vec::new() };
                match config.coding {
                    Coding::Hevc => append_annexb(&mut sample, &data, config.length_size)?,
                    Coding::Av1 => sample.extend_from_slice(&data),
                }
                Ok(sample)
            })
            .collect::<Result<Vec<_>>>()?;
        let (w, h) = self.ispe(ids[0]).unwrap_or((0, 0));
        let info = StreamInfo {
            codec: config.coding.decoder_name().to_string(),
            width: w,
            height: h,
            frame_rate: 1.0,
            duration: 0.0,
            pixel_format: config.pixel_format,
            color_space: ColorSpace::Bt601,
            total_frames: ids.len() as u64,
            bitrate: 0,
            color_metadata: Default::default(),
        };
        let frames = decode_samples(config.coding, info.clone(), samples.iter().map(Vec::as_slice))?;
        if frames.len() == ids.len() {
            return Ok(frames);
        }
        tracing::debug!(tiles = ids.len(), pictures = frames.len(), "decoding each HEIF tile on its own");
        samples
            .iter()
            .enumerate()
            .map(|(i, sample)| {
                // Each on its own needs the configuration ahead of it.
                let with_prefix = if i == 0 { sample.clone() } else { [config.prefix.as_slice(), sample].concat() };
                decode_samples(config.coding, info.clone(), std::iter::once(with_prefix.as_slice()))?
                    .into_iter()
                    .next()
                    .ok_or_else(|| anyhow!("the decoder gave no picture for HEIF tile {i}"))
            })
            .collect()
    }
}

fn decode_samples<'s>(coding: Coding, info: StreamInfo, samples: impl Iterator<Item = &'s [u8]>) -> Result<Vec<VideoFrame>> {
    let mut decoder = codec::decode::create_decoder(coding.decoder_name(), info)
        .with_context(|| format!("no decoder available for {} images", coding.decoder_name()))?;
    let mut frames = Vec::new();
    for sample in samples {
        decoder.push_sample(sample).context("decoding a HEIF picture")?;
        while let Some(frame) = decoder.decode_next()? {
            frames.push(frame);
        }
    }
    decoder.finish()?;
    while let Some(frame) = decoder.decode_next()? {
        frames.push(frame);
    }
    Ok(frames)
}

/// Length-prefixed NAL units → Annex-B.
fn append_annexb(out: &mut Vec<u8>, data: &[u8], length_size: usize) -> Result<()> {
    let mut at = 0;
    while at < data.len() {
        let len = data
            .get(at..at + length_size)
            .ok_or_else(|| anyhow!("a truncated NAL length"))?
            .iter()
            .fold(0usize, |acc, b| (acc << 8) | usize::from(*b));
        at += length_size;
        let nal = data.get(at..at + len).ok_or_else(|| anyhow!("a NAL runs past its item"))?;
        out.extend_from_slice(&[0, 0, 0, 1]);
        out.extend_from_slice(nal);
        at += len;
    }
    Ok(())
}

/// `hvcC` (ISO/IEC 14496-15 §8.3.3): 22 bytes of profile and format, the NAL
/// length size, then arrays of parameter sets.
fn hevc_config(body: &[u8]) -> Result<Config> {
    let mut r = Reader::new(body);
    let head = r.take(22).context("a short hvcC")?;
    let chroma = head[16] & 3;
    let depth = (head[17] & 7) + 8;
    let length_size = usize::from(head[21] & 3) + 1;
    let arrays = r.u8()?;
    let mut prefix = Vec::new();
    let mut parameter_sets = Vec::new();
    for _ in 0..arrays {
        let _type = r.u8()?;
        let count = r.u16()?;
        for _ in 0..count {
            let len = usize::from(r.u16()?);
            let nal = r.take(len)?;
            prefix.extend_from_slice(&[0, 0, 0, 1]);
            prefix.extend_from_slice(nal);
            parameter_sets.push(nal.to_vec());
        }
    }
    Ok(Config {
        coding: Coding::Hevc,
        pixel_format: planar_format(chroma, depth),
        prefix,
        length_size,
        parameter_sets,
    })
}

/// `av1C` (AV1-ISOBMFF §2.3): four bytes of format, then configuration OBUs
/// (the sequence header, usually repeated in the item itself).
fn av1_config(body: &[u8]) -> Result<Config> {
    if body.len() < 4 {
        bail!("a short av1C");
    }
    let flags = body[2];
    let depth = match (flags & 0x40 != 0, flags & 0x20 != 0) {
        (true, true) => 12,
        (true, false) => 10,
        _ => 8,
    };
    let chroma = match (flags & 0x10 != 0, flags & 0x08 != 0, flags & 0x04 != 0) {
        (true, _, _) => 0,
        (false, true, true) => 1,
        (false, true, false) => 2,
        _ => 3,
    };
    Ok(Config {
        coding: Coding::Av1,
        pixel_format: planar_format(chroma, depth),
        prefix: body[4..].to_vec(),
        length_size: 0,
        parameter_sets: Vec::new(),
    })
}

/// The planar format for an H.273-style `chroma_format_idc` (0 monochrome,
/// 1 4:2:0, 2 4:2:2, 3 4:4:4) and bit depth. Monochrome decodes as 4:2:0.
fn planar_format(chroma: u8, depth: u8) -> PixelFormat {
    match (chroma, depth) {
        (0 | 1, 8) => PixelFormat::Yuv420p,
        (0 | 1, 10) => PixelFormat::Yuv420p10le,
        (0 | 1, _) => PixelFormat::Yuv420p12le,
        (2, 8) => PixelFormat::Yuv422p,
        (2, 10) => PixelFormat::Yuv422p10le,
        (2, _) => PixelFormat::Yuv422p12le,
        (_, 8) => PixelFormat::Yuv444p,
        (_, 10) => PixelFormat::Yuv444p10le,
        (_, _) => PixelFormat::Yuv444p12le,
    }
}

/// The top-left `w x h` of a decoded frame as RGBA, or its luma as a
/// greyscale RGBA (for an alpha plane).
///
/// Chroma planes are `ceil(width / 2)` wide (and high, for 4:2:0) when the
/// frame is odd-sized, as AV1 lays them out; an even frame is the same either
/// way.
fn to_rgba(frame: &VideoFrame, w: u32, h: u32, plane: Plane) -> Result<RgbaImage> {
    let (xs, ys, depth) = match frame.format {
        PixelFormat::Yuv420p => (1, 1, 8),
        PixelFormat::Yuv420p10le => (1, 1, 10),
        PixelFormat::Yuv420p12le => (1, 1, 12),
        PixelFormat::Yuv422p => (1, 0, 8),
        PixelFormat::Yuv422p10le => (1, 0, 10),
        PixelFormat::Yuv422p12le => (1, 0, 12),
        PixelFormat::Yuv444p => (0, 0, 8),
        PixelFormat::Yuv444p10le | PixelFormat::Yuva444p10le => (0, 0, 10),
        PixelFormat::Yuv444p12le => (0, 0, 12),
        PixelFormat::Nv12 | PixelFormat::Nv21 => return nv_to_rgba(frame, w, h, plane),
        other => bail!("the decoder returned {other:?}, which a HEIF picture is not decoded to"),
    };
    let fw = frame.width as usize;
    let fh = frame.height as usize;
    let bps = if depth > 8 { 2 } else { 1 };
    let cw = (fw + (1 << xs) - 1) >> xs;
    let ch = (fh + (1 << ys) - 1) >> ys;
    let y_len = fw * fh * bps;
    let c_len = cw * ch * bps;
    let data = frame.data.as_ref();
    if data.len() < y_len + 2 * c_len {
        bail!("the decoded picture is truncated ({} bytes for {fw}x{fh} {:?})", data.len(), frame.format);
    }
    let sample = |plane: &[u8], idx: usize| -> f32 {
        if bps == 2 {
            (u16::from_le_bytes([plane[idx * 2], plane[idx * 2 + 1]]) >> (depth - 8)) as f32
        } else {
            plane[idx] as f32
        }
    };
    let (yp, up, vp) = (&data[..y_len], &data[y_len..y_len + c_len], &data[y_len + c_len..y_len + 2 * c_len]);
    let mut out = RgbaImage::new(w, h);
    match plane {
        Plane::Luma => {
            for (x, y, px) in out.enumerate_pixels_mut() {
                let v = sample(yp, y as usize * fw + x as usize) as u8;
                *px = [v, v, v, u8::MAX];
            }
        }
        Plane::Colour(matrix) => {
            let matrix = matrix.expect("resolved by decode_item");
            let convert = Converter::new(matrix);
            let mut rgb = Vec::with_capacity(3);
            for (x, y, px) in out.enumerate_pixels_mut() {
                let (x, y) = (x as usize, y as usize);
                let ci = (y >> ys) * cw + (x >> xs);
                rgb.clear();
                convert.push(&mut rgb, sample(yp, y * fw + x), sample(up, ci), sample(vp, ci));
                *px = [rgb[0], rgb[1], rgb[2], u8::MAX];
            }
        }
    }
    Ok(out)
}

/// NV12 / NV21, which a hardware decoder hands back: one luma plane, then
/// chroma interleaved at 4:2:0.
fn nv_to_rgba(frame: &VideoFrame, w: u32, h: u32, plane: Plane) -> Result<RgbaImage> {
    let fw = frame.width as usize;
    let fh = frame.height as usize;
    let cw = fw.div_ceil(2);
    let data = frame.data.as_ref();
    if data.len() < fw * fh + cw * fh.div_ceil(2) * 2 {
        bail!("the decoded picture is truncated");
    }
    let (yp, cp) = data.split_at(fw * fh);
    let cb_first = frame.format == PixelFormat::Nv12;
    let mut out = RgbaImage::new(w, h);
    let convert = match plane {
        Plane::Colour(m) => Some(Converter::new(m.expect("resolved by decode_item"))),
        Plane::Luma => None,
    };
    let mut rgb = Vec::with_capacity(3);
    for (x, y, px) in out.enumerate_pixels_mut() {
        let (x, y) = (x as usize, y as usize);
        let luma = yp[y * fw + x];
        let Some(convert) = &convert else {
            *px = [luma, luma, luma, u8::MAX];
            continue;
        };
        let at = ((y / 2) * cw + x / 2) * 2;
        let (u, v) = if cb_first { (cp[at], cp[at + 1]) } else { (cp[at + 1], cp[at]) };
        rgb.clear();
        convert.push(&mut rgb, f32::from(luma), f32::from(u), f32::from(v));
        *px = [rgb[0], rgb[1], rgb[2], u8::MAX];
    }
    Ok(out)
}

/// Samples → RGB for one [`Matrix`]: the thumbnail path's matrices, and the
/// identity (GBR) a lossless AVIF is coded in.
enum Converter {
    Matrix(YuvMatrix),
    /// H.273 matrix 0: the "Y, Cb, Cr" planes are G, B, R.
    Identity,
}

impl Converter {
    fn new(m: Matrix) -> Self {
        let space = match m.coefficients {
            0 => return Converter::Identity,
            1 => ColorSpace::Bt709,
            9 | 10 => ColorSpace::Bt2020,
            // 5 and 6 are BT.601; so is anything unspecified on a still.
            _ => ColorSpace::Bt601,
        };
        Converter::Matrix(YuvMatrix::for_source(space, SourceColor { full_range: m.full_range }))
    }

    fn push(&self, rgb: &mut Vec<u8>, y: f32, u: f32, v: f32) {
        match self {
            Converter::Matrix(m) => m.push(rgb, y, u, v),
            Converter::Identity => rgb.extend([v as u8, y as u8, u as u8]),
        }
    }
}

// ── ISO BMFF reading ───────────────────────────────────────────────────────

struct BmffBox<'a> {
    kind: [u8; 4],
    body: &'a [u8],
}

/// The boxes laid end to end in `data`. A box whose size runs past the end is
/// an error, and ends the walk.
fn boxes(data: &[u8]) -> impl Iterator<Item = Result<BmffBox<'_>>> {
    let mut at = 0usize;
    let mut failed = false;
    std::iter::from_fn(move || {
        if failed || at + 8 > data.len() {
            return None;
        }
        let size = u32::from_be_bytes(data[at..at + 4].try_into().unwrap()) as u64;
        let kind: [u8; 4] = data[at + 4..at + 8].try_into().unwrap();
        let (header, size) = match size {
            0 => (8, (data.len() - at) as u64),
            1 => {
                let Some(large) = data.get(at + 8..at + 16) else {
                    failed = true;
                    return Some(Err(anyhow!("a truncated box header")));
                };
                (16, u64::from_be_bytes(large.try_into().unwrap()))
            }
            s => (8, s),
        };
        let end = usize::try_from(size).ok().and_then(|s| at.checked_add(s));
        match end {
            Some(end) if end <= data.len() && size >= header as u64 => {
                let b = BmffBox { kind, body: &data[at + header..end] };
                at = end;
                Some(Ok(b))
            }
            _ => {
                failed = true;
                Some(Err(anyhow!("the '{}' box runs past its parent", String::from_utf8_lossy(&kind))))
            }
        }
    })
}

/// A full box's version, and its body after the version and flags.
fn full_box(body: &[u8]) -> Result<(u8, &[u8])> {
    if body.len() < 4 {
        bail!("a truncated full box");
    }
    Ok((body[0], &body[4..]))
}

fn parse_iinf(body: &[u8]) -> Result<Vec<Item>> {
    let (version, body) = full_box(body)?;
    let skip = if version == 0 { 2 } else { 4 };
    let mut items = Vec::new();
    for b in boxes(body.get(skip..).unwrap_or_default()) {
        let b = b?;
        if &b.kind != b"infe" {
            continue;
        }
        let (version, body) = full_box(b.body)?;
        if version < 2 {
            // Item info entries before version 2 carry no item type.
            continue;
        }
        let mut r = Reader::new(body);
        let id = r.id(version == 2)?;
        let _protection = r.u16()?;
        let kind: [u8; 4] = r.take(4)?.try_into().unwrap();
        items.push(Item { id, kind });
    }
    Ok(items)
}

fn parse_iloc(body: &[u8]) -> Result<Vec<(u32, Location)>> {
    let (version, body) = full_box(body)?;
    let mut r = Reader::new(body);
    let sizes = r.u8()?;
    let (offset_size, length_size) = (sizes >> 4, sizes & 0xf);
    let sizes = r.u8()?;
    let base_size = sizes >> 4;
    let index_size = if version >= 1 { sizes & 0xf } else { 0 };
    let count = if version < 2 { u32::from(r.u16()?) } else { r.u32()? };
    let mut out = Vec::new();
    for _ in 0..count {
        let id = r.id(version < 2)?;
        let method = if version >= 1 { (r.u16()? & 0xf) as u8 } else { 0 };
        let _data_ref = r.u16()?;
        let base = r.uint(base_size)?;
        let extents = r.u16()?;
        let mut list = Vec::with_capacity(usize::from(extents));
        for _ in 0..extents {
            let _index = r.uint(index_size)?;
            let offset = r.uint(offset_size)?;
            let length = r.uint(length_size)?;
            list.push((offset, length));
        }
        out.push((id, Location { method, base, extents: list }));
    }
    Ok(out)
}

fn parse_iref(body: &[u8]) -> Result<Vec<Reference>> {
    let (version, body) = full_box(body)?;
    let mut refs = Vec::new();
    for b in boxes(body) {
        let b = b?;
        let mut r = Reader::new(b.body);
        let from = r.id(version == 0)?;
        let count = r.u16()?;
        let to = (0..count).map(|_| r.id(version == 0)).collect::<Result<Vec<_>>>()?;
        refs.push(Reference { kind: b.kind, from, to });
    }
    Ok(refs)
}

type Associations = Vec<(u32, Vec<u16>)>;

fn parse_iprp(body: &[u8]) -> Result<(Vec<Property<'_>>, Associations)> {
    let mut properties = Vec::new();
    let mut associations = Vec::new();
    for b in boxes(body) {
        let b = b?;
        match &b.kind {
            b"ipco" => {
                for p in boxes(b.body) {
                    properties.push(parse_property(p?));
                }
            }
            b"ipma" => {
                let (version, flags) = (b.body.first().copied().unwrap_or(0), b.body.get(3).copied().unwrap_or(0));
                let mut r = Reader::new(full_box(b.body)?.1);
                let count = r.u32()?;
                for _ in 0..count {
                    let item = r.id(version < 1)?;
                    let n = r.u8()?;
                    let mut list = Vec::with_capacity(usize::from(n));
                    for _ in 0..n {
                        let index = if flags & 1 != 0 { r.u16()? & 0x7fff } else { u16::from(r.u8()? & 0x7f) };
                        list.push(index);
                    }
                    associations.push((item, list));
                }
            }
            _ => {}
        }
    }
    Ok((properties, associations))
}

fn parse_property(b: BmffBox<'_>) -> Property<'_> {
    let parsed = (|| -> Result<Property<'_>> {
        Ok(match &b.kind {
            b"ispe" => {
                let mut r = Reader::new(full_box(b.body)?.1);
                Property::Ispe(r.u32()?, r.u32()?)
            }
            b"hvcC" => Property::HvcC(b.body),
            b"av1C" => Property::Av1C(b.body),
            b"colr" if b.body.len() >= 4 => match &b.body[..4] {
                b"nclx" => {
                    let mut r = Reader::new(&b.body[4..]);
                    let narrow = |v: u16| u8::try_from(v).unwrap_or(2);
                    Property::Nclx(container::demux::Nclx {
                        primaries: narrow(r.u16()?),
                        transfer: narrow(r.u16()?),
                        matrix: narrow(r.u16()?),
                        full_range: r.u8()? & 0x80 != 0,
                    })
                }
                b"prof" | b"rICC" => Property::Icc(&b.body[4..]),
                _ => Property::Other,
            },
            b"irot" => Property::Irot(b.body.first().copied().unwrap_or(0) & 3),
            b"imir" => Property::Imir(b.body.first().copied().unwrap_or(0) & 1),
            b"auxC" => {
                let urn = full_box(b.body)?.1;
                let end = urn.iter().position(|&c| c == 0).unwrap_or(urn.len());
                Property::AuxC(String::from_utf8_lossy(&urn[..end]).into_owned())
            }
            _ => Property::Other,
        })
    })();
    // A property that does not parse is one this reader does not use; it
    // keeps its place so the indices after it still line up.
    parsed.unwrap_or(Property::Other)
}

/// A big-endian cursor.
struct Reader<'a> {
    data: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, at: 0 }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let out = self.data.get(self.at..self.at + n).ok_or_else(|| anyhow!("a truncated HEIF box"))?;
        self.at += n;
        Ok(out)
    }

    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_be_bytes(self.take(2)?.try_into().unwrap()))
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_be_bytes(self.take(4)?.try_into().unwrap()))
    }

    /// An `n`-byte unsigned integer; `n` is 0, 4 or 8 in `iloc` (0 reads 0).
    fn uint(&mut self, n: u8) -> Result<u64> {
        Ok(self.take(usize::from(n))?.iter().fold(0u64, |acc, b| (acc << 8) | u64::from(*b)))
    }

    /// An item ID: 16 bits in the older box versions, 32 in the newer.
    fn id(&mut self, short: bool) -> Result<u32> {
        if short { Ok(u32::from(self.u16()?)) } else { self.u32() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bx(kind: &[u8; 4], body: &[u8]) -> Vec<u8> {
        let mut b = ((body.len() + 8) as u32).to_be_bytes().to_vec();
        b.extend_from_slice(kind);
        b.extend_from_slice(body);
        b
    }

    fn full(kind: &[u8; 4], version: u8, body: &[u8]) -> Vec<u8> {
        let mut b = vec![version, 0, 0, 0];
        b.extend_from_slice(body);
        bx(kind, &b)
    }

    /// A HEIF of a `columns` x 1 grid, `out_w` wide, every tile the HEVC
    /// picture of `testdata/small.heic`, with `transforms` (property boxes)
    /// on the grid. Everything is in `idat`, so no offset depends on layout.
    fn gridded(columns: u8, out_w: u16, transforms: &[Vec<u8>]) -> Vec<u8> {
        let source = include_bytes!("testdata/small.heic");
        let heif = Heif::parse(source).unwrap();
        // heif-enc writes even one tile as a grid of one.
        let coded = heif.coded_items(heif.primary).unwrap()[0];
        let hvcc = heif
            .props(coded)
            .find_map(|p| match p {
                Property::HvcC(b) => Some(b.to_vec()),
                _ => None,
            })
            .unwrap();
        let tile = heif.item_data(coded).unwrap().to_vec();
        let (tw, th) = heif.ispe(coded).unwrap();
        // The coded tile is padded to 64x64; the picture is 64x48.
        let (_, out_h) = heif.ispe(heif.primary).unwrap();

        let grid_id = 100u16;
        let tiles: Vec<u16> = (1..=u16::from(columns)).collect();
        let mut grid_data = vec![0, 0, 0, columns - 1];
        grid_data.extend_from_slice(&out_w.to_be_bytes());
        grid_data.extend_from_slice(&(out_h as u16).to_be_bytes());

        let mut idat = grid_data.clone();
        let tile_at = idat.len() as u32;
        idat.extend_from_slice(&tile);

        let infe = |id: u16, kind: &[u8; 4]| {
            let mut b = id.to_be_bytes().to_vec();
            b.extend_from_slice(&[0, 0]);
            b.extend_from_slice(kind);
            b.push(0);
            full(b"infe", 2, &b)
        };
        let mut iinf = ((tiles.len() + 1) as u16).to_be_bytes().to_vec();
        for &t in &tiles {
            iinf.extend(infe(t, b"hvc1"));
        }
        iinf.extend(infe(grid_id, b"grid"));

        let loc = |id: u16, offset: u32, length: u32| {
            let mut b = id.to_be_bytes().to_vec();
            b.extend_from_slice(&1u16.to_be_bytes()); // construction method 1: idat
            b.extend_from_slice(&0u16.to_be_bytes());
            b.extend_from_slice(&1u16.to_be_bytes());
            b.extend_from_slice(&offset.to_be_bytes());
            b.extend_from_slice(&length.to_be_bytes());
            b
        };
        let mut iloc = vec![0x44, 0x00];
        iloc.extend_from_slice(&((tiles.len() + 1) as u16).to_be_bytes());
        for &t in &tiles {
            iloc.extend(loc(t, tile_at, tile.len() as u32));
        }
        iloc.extend(loc(grid_id, 0, grid_data.len() as u32));

        let mut dimg = grid_id.to_be_bytes().to_vec();
        dimg.extend_from_slice(&(tiles.len() as u16).to_be_bytes());
        for &t in &tiles {
            dimg.extend_from_slice(&t.to_be_bytes());
        }

        let ispe = |w: u32, h: u32| {
            let mut b = w.to_be_bytes().to_vec();
            b.extend_from_slice(&h.to_be_bytes());
            full(b"ispe", 0, &b)
        };
        let mut ipco = bx(b"hvcC", &hvcc);
        ipco.extend(ispe(tw, th));
        ipco.extend(ispe(u32::from(out_w), out_h));
        for t in transforms {
            ipco.extend_from_slice(t);
        }
        let mut ipma = ((tiles.len() + 1) as u32).to_be_bytes().to_vec();
        for &t in &tiles {
            ipma.extend_from_slice(&t.to_be_bytes());
            ipma.extend_from_slice(&[2, 0x81, 2]);
        }
        ipma.extend_from_slice(&grid_id.to_be_bytes());
        ipma.push(1 + transforms.len() as u8);
        ipma.push(3);
        for i in 0..transforms.len() {
            ipma.push(0x80 | (4 + i as u8));
        }
        let mut iprp = bx(b"ipco", &ipco);
        iprp.extend(full(b"ipma", 0, &ipma));

        let mut hdlr = vec![0; 4];
        hdlr.extend_from_slice(b"pict");
        hdlr.extend_from_slice(&[0; 13]);
        let mut meta = full(b"hdlr", 0, &hdlr);
        meta.extend(full(b"pitm", 0, &grid_id.to_be_bytes()));
        meta.extend(full(b"iinf", 0, &iinf));
        meta.extend(full(b"iloc", 1, &iloc));
        meta.extend(full(b"iref", 0, &bx(b"dimg", &dimg)));
        meta.extend(bx(b"iprp", &iprp));
        meta.extend(bx(b"idat", &idat));

        let mut file = bx(b"ftyp", b"heic\0\0\0\0mif1heic");
        file.extend(full(b"meta", 0, &meta));
        file
    }

    fn is_red(img: &RgbaImage, x: u32, y: u32) -> bool {
        let p = img.get_pixel(x, y);
        p[0] > 200 && p[1] < 60 && p[2] < 60
    }

    #[test]
    fn a_grid_is_tiled_and_cropped_to_its_declared_size() {
        let file = gridded(2, 120, &[]);
        assert_eq!(sniff(&file), Some(SourceFormat::Heic));
        let h = read_header(&file).unwrap();
        assert_eq!((h.width, h.height), (120, 48));
        let picture = decode(&file, SourceFormat::Heic).unwrap();
        assert_eq!(picture.rgba.dimensions(), (120, 48));
        // Each tile's red corner, where its tile starts.
        assert!(is_red(&picture.rgba, 3, 3));
        assert!(is_red(&picture.rgba, 67, 3));
        assert!(!is_red(&picture.rgba, 40, 30));
    }

    #[test]
    fn irot_and_imir_turn_the_picture_upright() {
        // A quarter turn anticlockwise: the top-left corner goes to the
        // bottom-left, and the second tile's corner above it.
        let turned = gridded(2, 120, &[bx(b"irot", &[1])]);
        let h = read_header(&turned).unwrap();
        assert_eq!((h.width, h.height, h.stored_width, h.stored_height), (48, 120, 120, 48));
        let picture = decode(&turned, SourceFormat::Heic).unwrap();
        assert_eq!(picture.rgba.dimensions(), (48, 120));
        assert!(is_red(&picture.rgba, 3, 116));
        assert!(is_red(&picture.rgba, 3, 52));
        assert!(!is_red(&picture.rgba, 44, 3));

        // Mirrored about the vertical axis: the corner is top-right.
        let mirrored = gridded(1, 64, &[bx(b"imir", &[0])]);
        let picture = decode(&mirrored, SourceFormat::Heic).unwrap();
        assert!(is_red(&picture.rgba, 60, 3));
        assert!(!is_red(&picture.rgba, 3, 3));
    }

    #[test]
    fn a_truncated_file_is_an_error_not_a_panic() {
        let file = include_bytes!("testdata/small.heic");
        for cut in [20, 100, 300, file.len() - 1] {
            let _ = read_header(&file[..cut]);
            let _ = decode(&file[..cut], SourceFormat::Heic);
        }
    }
}
