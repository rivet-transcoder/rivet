//! AVIF stills: rivet's own AV1 encoder (`crates/av1`) in rivet's own HEIF
//! writer.
//!
//! An AVIF file is ISO BMFF with no movie (ISO/IEC 23008-12, HEIF; AV1 Image
//! File Format 1.1): an `ftyp` naming the `avif` brand, a `meta` box that
//! lists *items* and their properties, and an `mdat` holding the coded
//! pictures. This writes:
//!
//! - one `av01` item — a key frame, profile 0, 8-bit 4:2:0 — with its `ispe`
//!   (size), `pixi` (bit depth), `av1C` (the codec configuration, carrying
//!   the sequence header) and `colr` (`nclx`: sRGB primaries and transfer,
//!   BT.601 matrix, full range) properties;
//! - or, for a picture larger than one item should be (wider than the
//!   encoder's 4096 or over [`TILE_PIXELS`]), a `grid` item over equal tiles,
//!   each its own hidden `av01` item, which are encoded in parallel — the
//!   encoder is single-threaded, so a grid is also how a large still uses
//!   the machine;
//! - and, when the picture has transparency, an alpha item (or grid) of the
//!   same shape, linked to the colour item by an `auxl` reference and marked
//!   with the alpha `auxC` URN. Its luma is the alpha; its chroma is flat.
//!
//! What the reader in `image::heif` reads is exactly this, which is what
//! the round-trip tests check; browsers and libavif read it as well (the
//! structure is the one the AVIF specification's examples use).
//!
//! # Quality
//!
//! `quality` is the 1-100 scale the image settings use (higher is better),
//! mapped onto the encoder's `base_q_idx` 1-255 by
//! [`quantizer_for_quality`]. The encoder has no speed setting worth the
//! name for a key frame, so there is none here.

use anyhow::{Context, Result, anyhow, bail};

/// The largest single item, in pixels: above it the picture is split into a
/// grid of tiles encoded in parallel.
pub const TILE_PIXELS: u64 = 2048 * 2048;

/// The widest single AV1 item the encoder writes (one tile).
const MAX_ITEM_WIDTH: u32 = 4096;

/// The `auxC` URN that makes an auxiliary item an alpha plane.
const ALPHA_URN: &str = "urn:mpeg:mpegB:cicp:systems:auxiliary:alpha";

/// The encoder's `base_q_idx` for a 1-100 quality: 100 is near-lossless
/// (`base_q_idx` 1), 1 the coarsest (255); piecewise linear between anchors
/// chosen so the scale reads like the other lossy formats' (60, the AVIF
/// default, lands at 120: some 40 dB on photographic content).
pub fn quantizer_for_quality(quality: u8) -> u32 {
    const ANCHORS: [(f32, f32); 7] =
        [(1.0, 255.0), (20.0, 205.0), (40.0, 162.0), (60.0, 120.0), (80.0, 72.0), (90.0, 44.0), (100.0, 1.0)];
    let q = f32::from(quality.clamp(1, 100));
    let i = ANCHORS.iter().position(|a| a.0 >= q).unwrap_or(ANCHORS.len() - 1).max(1);
    let (a, b) = (ANCHORS[i - 1], ANCHORS[i]);
    let t = (q - a.0) / (b.0 - a.0);
    (a.1 + t * (b.1 - a.1)).round().clamp(1.0, 255.0) as u32
}

/// Encode `width` x `height` RGB triplets as an AVIF.
pub fn encode_rgb(rgb: &[u8], width: u32, height: u32, quality: u8) -> Result<Vec<u8>> {
    if rgb.len() != width as usize * height as usize * 3 {
        bail!("AVIF: {} bytes of RGB for {width}x{height}", rgb.len());
    }
    let rgba: Vec<u8> = rgb.chunks_exact(3).flat_map(|p| [p[0], p[1], p[2], u8::MAX]).collect();
    encode_rgba(&rgba, width, height, false, quality)
}

/// Encode `width` x `height` RGBA pixels as an AVIF, with an alpha item when
/// `alpha` (otherwise the alpha bytes are ignored).
pub fn encode_rgba(rgba: &[u8], width: u32, height: u32, alpha: bool, quality: u8) -> Result<Vec<u8>> {
    if width == 0 || height == 0 {
        bail!("AVIF: a picture has no pixels ({width}x{height})");
    }
    if rgba.len() != width as usize * height as usize * 4 {
        bail!("AVIF: {} bytes of RGBA for {width}x{height}", rgba.len());
    }
    let layout = Layout::for_size(width, height);
    let q = quantizer_for_quality(quality);
    let colour = encode_plane_set(rgba, width, &layout, q, Plane::Colour)?;
    let alpha = if alpha {
        // Alpha edges show more than colour does: a finer quantiser.
        Some(encode_plane_set(rgba, width, &layout, (q * 3 / 4).max(1), Plane::Alpha)?)
    } else {
        None
    };
    Ok(write_file(width, height, &layout, &colour, alpha.as_ref()))
}

/// How a picture is cut into items.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Layout {
    columns: u32,
    rows: u32,
    tile_w: u32,
    tile_h: u32,
}

impl Layout {
    fn for_size(w: u32, h: u32) -> Self {
        if w <= MAX_ITEM_WIDTH && u64::from(w) * u64::from(h) <= TILE_PIXELS {
            return Self { columns: 1, rows: 1, tile_w: w, tile_h: h };
        }
        // Tiles of at most 2048 a side, equal, even (4:2:0), the last row
        // and column overhanging the picture as little as possible.
        let columns = w.div_ceil(2048);
        let rows = h.div_ceil(2048);
        let even = |v: u32| v + (v & 1);
        Self { columns, rows, tile_w: even(w.div_ceil(columns)), tile_h: even(h.div_ceil(rows)) }
    }

    fn is_grid(&self) -> bool {
        self.columns * self.rows > 1
    }
}

#[derive(Clone, Copy)]
enum Plane {
    Colour,
    Alpha,
}

/// One coded item: its OBUs (no temporal delimiter) and its `av1C`.
struct Coded {
    obus: Vec<u8>,
    av1c: Vec<u8>,
}

/// Encode every tile of `layout` as its own key frame, in parallel.
fn encode_plane_set(rgba: &[u8], width: u32, layout: &Layout, q: u32, plane: Plane) -> Result<Vec<Coded>> {
    let height = (rgba.len() / 4 / width as usize) as u32;
    let tiles: Vec<(u32, u32)> =
        (0..layout.rows).flat_map(|r| (0..layout.columns).map(move |c| (c * layout.tile_w, r * layout.tile_h))).collect();
    let workers = std::thread::available_parallelism().map_or(1, |n| n.get()).min(tiles.len()).max(1);
    let next = std::sync::atomic::AtomicUsize::new(0);
    let mut out: Vec<Option<Result<Coded>>> = (0..tiles.len()).map(|_| None).collect();
    let slots = std::sync::Mutex::new(&mut out);
    std::thread::scope(|s| {
        for _ in 0..workers {
            s.spawn(|| {
                loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let Some(&(x0, y0)) = tiles.get(i) else { break };
                    let frame = tile_frame(rgba, width, height, x0, y0, layout.tile_w, layout.tile_h, plane);
                    let coded = encode_tile(&frame, q);
                    slots.lock().expect("no worker panics holding it")[i] = Some(coded);
                }
            });
        }
    });
    out.into_iter().map(|c| c.unwrap_or_else(|| Err(anyhow!("AVIF: a tile was not encoded")))).collect()
}

/// The `tw` x `th` tile at (`x0`, `y0`) as an 8-bit 4:2:0 frame: full-range
/// BT.601 Y'CbCr of the colour, or the alpha as luma over flat chroma.
/// Samples past the picture's edge repeat its last row and column.
#[allow(clippy::too_many_arguments)]
fn tile_frame(rgba: &[u8], w: u32, h: u32, x0: u32, y0: u32, tw: u32, th: u32, plane: Plane) -> av1::Frame {
    let mut f = av1::Frame::new(tw, th, 8, av1::ChromaFormat::Yuv420);
    let at = |x: u32, y: u32| -> [f32; 4] {
        let (x, y) = ((x0 + x).min(w - 1), (y0 + y).min(h - 1));
        let i = (y as usize * w as usize + x as usize) * 4;
        [f32::from(rgba[i]), f32::from(rgba[i + 1]), f32::from(rgba[i + 2]), f32::from(rgba[i + 3])]
    };
    let clamp = |v: f32| v.round().clamp(0.0, 255.0) as u8;
    let (yp, cp) = (f.planes[0], f.planes[1]);
    let (u_off, v_off) = (f.planes[1].offset, f.planes[2].offset);
    for y in 0..th {
        for x in 0..tw {
            let p = at(x, y);
            let luma = match plane {
                Plane::Colour => 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2],
                Plane::Alpha => p[3],
            };
            f.data[yp.offset + (y * tw + x) as usize] = clamp(luma);
        }
    }
    for cy in 0..cp.height {
        for cx in 0..cp.width {
            let (mut cb, mut cr) = (128.0, 128.0);
            if let Plane::Colour = plane {
                // The mean of the 2x2 block's chroma.
                let (mut sb, mut sr) = (0.0, 0.0);
                for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    let p = at((2 * cx + dx).min(tw - 1), (2 * cy + dy).min(th - 1));
                    sb += -0.168_736 * p[0] - 0.331_264 * p[1] + 0.5 * p[2];
                    sr += 0.5 * p[0] - 0.418_688 * p[1] - 0.081_312 * p[2];
                }
                cb = 128.0 + sb / 4.0;
                cr = 128.0 + sr / 4.0;
            }
            let i = (cy * cp.width + cx) as usize;
            f.data[u_off + i] = clamp(cb);
            f.data[v_off + i] = clamp(cr);
        }
    }
    f
}

/// The AV1 encoder's effort for AVIF items (`av1::Config::speed`, 0 slowest
/// to 10 fastest).
const AVIF_SPEED: u32 = 8;

/// One key frame, its OBUs less the temporal delimiter, and its `av1C`.
fn encode_tile(frame: &av1::Frame, q: u32) -> Result<Coded> {
    let mut cfg = av1::Config::new(frame.width, frame.height);
    cfg.quantizer = q;
    cfg.keyframe_interval = 1;
    // The encoder's fast rate-distortion search (palettes for screen
    // content included); the grid's tiles already run in parallel.
    cfg.speed = AVIF_SPEED;
    cfg.tools = av1::Tools::for_speed(AVIF_SPEED);
    let mut enc = av1::Encoder::new(cfg);
    let tu = enc.encode(frame).map_err(|e| anyhow!("AV1 encode of an AVIF item failed: {e}"))?;
    let mut obus = Vec::with_capacity(tu.len());
    let mut sequence_header = None;
    for (kind, whole, payload) in split_obus(&tu)? {
        match kind {
            OBU_TEMPORAL_DELIMITER => {}
            OBU_SEQUENCE_HEADER => {
                sequence_header = Some((whole.to_vec(), payload.to_vec()));
                obus.extend_from_slice(whole);
            }
            _ => obus.extend_from_slice(whole),
        }
    }
    let (sh_obu, sh) = sequence_header.context("the AV1 encoder wrote no sequence header on a key frame")?;
    Ok(Coded { obus, av1c: av1c(&sh_obu, &sh, frame.bit_depth) })
}

const OBU_SEQUENCE_HEADER: u8 = 1;
const OBU_TEMPORAL_DELIMITER: u8 = 2;

/// The OBUs of a temporal unit (low-overhead format, every OBU sized):
/// `(type, the whole OBU, its payload)`.
fn split_obus(data: &[u8]) -> Result<Vec<(u8, &[u8], &[u8])>> {
    let mut out = Vec::new();
    let mut at = 0;
    while at < data.len() {
        let start = at;
        let header = data[at];
        let kind = (header >> 3) & 15;
        let extension = header & 4 != 0;
        if header & 2 == 0 {
            bail!("an AV1 OBU without a size field");
        }
        at += 1 + usize::from(extension);
        let (size, n) = leb128(data.get(at..).unwrap_or_default())?;
        at += n;
        let end = at.checked_add(size).filter(|&e| e <= data.len()).context("an AV1 OBU runs past its temporal unit")?;
        out.push((kind, &data[start..end], &data[at..end]));
        at = end;
    }
    Ok(out)
}

fn leb128(data: &[u8]) -> Result<(usize, usize)> {
    let mut v = 0usize;
    for (i, b) in data.iter().take(8).enumerate() {
        v |= usize::from(b & 0x7f) << (7 * i);
        if b & 0x80 == 0 {
            return Ok((v, i + 1));
        }
    }
    bail!("a truncated leb128")
}

/// `AV1CodecConfigurationRecord` (AV1-ISOBMFF 2.3): marker and version, the
/// profile, level and tier of operating point 0, the colour format, then the
/// sequence header OBU as the configuration OBUs.
fn av1c(sequence_header_obu: &[u8], sh: &[u8], bit_depth: u32) -> Vec<u8> {
    let (profile, level, tier) = profile_level_tier(sh).unwrap_or((0, 31, 0));
    let high_bitdepth = u8::from(bit_depth > 8);
    let twelve_bit = u8::from(bit_depth == 12);
    let mut out = vec![
        0x81,
        (profile << 5) | (level & 31),
        // tier, high_bitdepth, twelve_bit, monochrome 0, subsampling 1 1,
        // chroma_sample_position 0 (unknown) — 4:2:0 as the encoder codes.
        (tier << 7) | (high_bitdepth << 6) | (twelve_bit << 5) | (1 << 3) | (1 << 2),
        // No initial_presentation_delay.
        0,
    ];
    out.extend_from_slice(sequence_header_obu);
    out
}

/// `seq_profile`, and `seq_level_idx[0]` and `seq_tier[0]`, from a sequence
/// header's first fields (5.5.1). `None` for a header with timing info, whose
/// operating points this does not walk.
fn profile_level_tier(sh: &[u8]) -> Option<(u8, u8, u8)> {
    let bit = |i: usize| -> Option<u8> { sh.get(i / 8).map(|b| (b >> (7 - i % 8)) & 1) };
    let bits = |from: usize, n: usize| -> Option<u32> { (from..from + n).try_fold(0u32, |v, i| Some(v << 1 | u32::from(bit(i)?))) };
    let profile = bits(0, 3)? as u8;
    let reduced = bit(4)? == 1;
    if reduced {
        return Some((profile, bits(5, 5)? as u8, 0));
    }
    if bit(5)? == 1 {
        return None; // timing_info_present_flag
    }
    // initial_display_delay_present_flag (6), operating_points_cnt_minus_1
    // (7..12), then operating point 0: idc (12 bits), seq_level_idx (5).
    let display_delay = bit(6)? == 1;
    let at = 12 + 12;
    let level = bits(at, 5)? as u8;
    let tier = if level > 7 { bit(at + 5)? } else { 0 };
    let _ = display_delay;
    Some((profile, level, tier))
}

// ── HEIF ─────────────────────────────────────────────────────────────────

/// A box: size, type, body.
fn bx(kind: &[u8; 4], body: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(8 + body.len());
    out.extend_from_slice(&((8 + body.len()) as u32).to_be_bytes());
    out.extend_from_slice(kind);
    out.extend_from_slice(body);
    out
}

/// A FullBox: version, flags, body.
fn full(kind: &[u8; 4], version: u8, flags: u32, body: &[u8]) -> Vec<u8> {
    let mut b = Vec::with_capacity(4 + body.len());
    b.push(version);
    b.extend_from_slice(&flags.to_be_bytes()[1..]);
    b.extend_from_slice(body);
    bx(kind, &b)
}

/// One item: its id, type, whether hidden, its data, and its properties
/// (indices into `ipco`, 1-based, with the essential flag).
struct Item {
    id: u16,
    kind: [u8; 4],
    hidden: bool,
    data: Vec<u8>,
    props: Vec<(u8, bool)>,
}

/// The property boxes, deduplicated.
#[derive(Default)]
struct Properties {
    boxes: Vec<Vec<u8>>,
}

impl Properties {
    /// The 1-based index of `b`, added if new.
    fn index(&mut self, b: Vec<u8>) -> u8 {
        if let Some(i) = self.boxes.iter().position(|x| *x == b) {
            return (i + 1) as u8;
        }
        self.boxes.push(b);
        self.boxes.len() as u8
    }
}

fn ispe(w: u32, h: u32) -> Vec<u8> {
    let mut b = w.to_be_bytes().to_vec();
    b.extend_from_slice(&h.to_be_bytes());
    full(b"ispe", 0, 0, &b)
}

fn pixi(channels: u8) -> Vec<u8> {
    let mut b = vec![channels];
    b.extend(std::iter::repeat_n(8u8, usize::from(channels)));
    full(b"pixi", 0, 0, &b)
}

/// `colr` `nclx`: BT.709 primaries, sRGB transfer, BT.601 matrix, full range.
fn colr() -> Vec<u8> {
    let mut b = b"nclx".to_vec();
    b.extend_from_slice(&1u16.to_be_bytes());
    b.extend_from_slice(&13u16.to_be_bytes());
    b.extend_from_slice(&6u16.to_be_bytes());
    b.push(0x80);
    bx(b"colr", &b)
}

fn auxc() -> Vec<u8> {
    let mut b = ALPHA_URN.as_bytes().to_vec();
    b.push(0);
    full(b"auxC", 0, 0, &b)
}

/// `ImageGrid` (HEIF 6.6.2.3.2): rows and columns over the output size.
fn grid_data(layout: &Layout, w: u32, h: u32) -> Vec<u8> {
    let wide = w > 0xFFFF || h > 0xFFFF;
    let mut b = vec![0, u8::from(wide), (layout.rows - 1) as u8, (layout.columns - 1) as u8];
    if wide {
        b.extend_from_slice(&w.to_be_bytes());
        b.extend_from_slice(&h.to_be_bytes());
    } else {
        b.extend_from_slice(&(w as u16).to_be_bytes());
        b.extend_from_slice(&(h as u16).to_be_bytes());
    }
    b
}

/// Lay the items out as a file: `ftyp`, `meta`, `mdat`.
fn write_file(w: u32, h: u32, layout: &Layout, colour: &[Coded], alpha: Option<&Vec<Coded>>) -> Vec<u8> {
    let mut props = Properties::default();
    let mut items: Vec<Item> = Vec::new();
    let mut dimg: Vec<(u16, Vec<u16>)> = Vec::new();
    let mut next_id = 1u16;

    // One picture (colour or alpha): its items, and the id that stands for it.
    let mut add_set = |coded: &[Coded], is_alpha: bool, items: &mut Vec<Item>, props: &mut Properties| -> u16 {
        let channels = if is_alpha { 1 } else { 3 };
        let mut extra: Vec<(u8, bool)> = Vec::new();
        if is_alpha {
            extra.push((props.index(auxc()), true));
        } else {
            extra.push((props.index(colr()), false));
        }
        if !layout.is_grid() {
            let id = next_id;
            next_id += 1;
            let mut p = vec![(props.index(ispe(w, h)), false), (props.index(pixi(channels)), false)];
            p.push((props.index(bx(b"av1C", &coded[0].av1c)), true));
            p.extend(extra);
            items.push(Item { id, kind: *b"av01", hidden: false, data: coded[0].obus.clone(), props: p });
            return id;
        }
        let grid_id = next_id;
        next_id += 1;
        let mut gp = vec![(props.index(ispe(w, h)), false), (props.index(pixi(channels)), false)];
        gp.extend(extra);
        items.push(Item { id: grid_id, kind: *b"grid", hidden: false, data: grid_data(layout, w, h), props: gp });
        let mut tiles = Vec::with_capacity(coded.len());
        for c in coded {
            let id = next_id;
            next_id += 1;
            let p = vec![
                (props.index(ispe(layout.tile_w, layout.tile_h)), false),
                (props.index(pixi(channels)), false),
                (props.index(bx(b"av1C", &c.av1c)), true),
            ];
            items.push(Item { id, kind: *b"av01", hidden: true, data: c.obus.clone(), props: p });
            tiles.push(id);
        }
        dimg.push((grid_id, tiles));
        grid_id
    };
    let primary = add_set(colour, false, &mut items, &mut props);
    let alpha_id = alpha.map(|a| add_set(a, true, &mut items, &mut props));

    let ftyp = {
        let mut b = b"avif".to_vec();
        b.extend_from_slice(&0u32.to_be_bytes());
        for brand in [b"avif", b"mif1", b"miaf"] {
            b.extend_from_slice(brand);
        }
        bx(b"ftyp", &b)
    };

    let meta = |offsets: &[u32]| -> Vec<u8> {
        let mut body = Vec::new();
        // hdlr: pre_defined, handler 'pict', three reserved, an empty name.
        let mut hdlr = 0u32.to_be_bytes().to_vec();
        hdlr.extend_from_slice(b"pict");
        hdlr.extend_from_slice(&[0; 12]);
        hdlr.push(0);
        body.extend(full(b"hdlr", 0, 0, &hdlr));
        body.extend(full(b"pitm", 0, 0, &primary.to_be_bytes()));
        // iloc v0: 4-byte offsets and lengths, no base offset; one extent each.
        let mut iloc = vec![0x44, 0x00];
        iloc.extend_from_slice(&(items.len() as u16).to_be_bytes());
        for (item, &offset) in items.iter().zip(offsets) {
            iloc.extend_from_slice(&item.id.to_be_bytes());
            iloc.extend_from_slice(&0u16.to_be_bytes()); // data_reference_index
            iloc.extend_from_slice(&1u16.to_be_bytes()); // extent_count
            iloc.extend_from_slice(&offset.to_be_bytes());
            iloc.extend_from_slice(&(item.data.len() as u32).to_be_bytes());
        }
        body.extend(full(b"iloc", 0, 0, &iloc));
        // iinf with an infe v2 per item (hidden items flagged).
        let mut iinf = (items.len() as u16).to_be_bytes().to_vec();
        for item in &items {
            let mut infe = item.id.to_be_bytes().to_vec();
            infe.extend_from_slice(&0u16.to_be_bytes()); // protection_index
            infe.extend_from_slice(&item.kind);
            infe.push(0); // item_name
            iinf.extend(full(b"infe", 2, u32::from(item.hidden), &infe));
        }
        body.extend(full(b"iinf", 0, 0, &iinf));
        // iref: each grid's tiles (dimg), and the alpha's master (auxl).
        let mut refs = Vec::new();
        for (from, to) in &dimg {
            let mut r = from.to_be_bytes().to_vec();
            r.extend_from_slice(&(to.len() as u16).to_be_bytes());
            for t in to {
                r.extend_from_slice(&t.to_be_bytes());
            }
            refs.extend(bx(b"dimg", &r));
        }
        if let Some(a) = alpha_id {
            let mut r = a.to_be_bytes().to_vec();
            r.extend_from_slice(&1u16.to_be_bytes());
            r.extend_from_slice(&primary.to_be_bytes());
            refs.extend(bx(b"auxl", &r));
        }
        if !refs.is_empty() {
            body.extend(full(b"iref", 0, 0, &refs));
        }
        // iprp: the properties, then each item's associations.
        let ipco = bx(b"ipco", &props.boxes.concat());
        let mut ipma = (items.len() as u32).to_be_bytes().to_vec();
        for item in &items {
            ipma.extend_from_slice(&item.id.to_be_bytes());
            ipma.push(item.props.len() as u8);
            for &(index, essential) in &item.props {
                ipma.push((u8::from(essential) << 7) | index);
            }
        }
        let mut iprp = ipco;
        iprp.extend(full(b"ipma", 0, 0, &ipma));
        body.extend(bx(b"iprp", &iprp));
        full(b"meta", 0, 0, &body)
    };

    // The offsets depend on the meta box's size, which does not depend on
    // the offsets: lay it out once to measure it.
    let meta_len = meta(&vec![0; items.len()]).len();
    let data_len: usize = items.iter().map(|i| i.data.len()).sum();
    let mut at = (ftyp.len() + meta_len + 8) as u32;
    let offsets: Vec<u32> = items
        .iter()
        .map(|i| {
            let o = at;
            at += i.data.len() as u32;
            o
        })
        .collect();
    let mut out = Vec::with_capacity(ftyp.len() + meta_len + 8 + data_len);
    out.extend(ftyp);
    out.extend(meta(&offsets));
    out.extend_from_slice(&((8 + data_len) as u32).to_be_bytes());
    out.extend_from_slice(b"mdat");
    for item in &items {
        out.extend_from_slice(&item.data);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quality_maps_monotonically_onto_the_quantiser() {
        let qs: Vec<u32> = (1..=100).map(quantizer_for_quality).collect();
        assert!(qs.windows(2).all(|w| w[1] <= w[0]), "{qs:?}");
        assert_eq!((qs[0], qs[59], qs[99]), (255, 120, 1));
    }

    #[test]
    fn small_pictures_are_one_item_and_large_ones_a_grid() {
        assert!(!Layout::for_size(2048, 2048).is_grid());
        assert!(!Layout::for_size(4096, 1024).is_grid());
        let l = Layout::for_size(4097, 100);
        assert_eq!((l.columns, l.rows), (3, 1));
        let l = Layout::for_size(4000, 3000);
        assert_eq!((l.columns, l.rows, l.tile_w, l.tile_h), (2, 2, 2000, 1500));
        let l = Layout::for_size(2101, 2101);
        assert_eq!((l.tile_w, l.tile_h), (1052, 1052), "even, with the least overhang");
    }

    #[test]
    fn the_configuration_record_names_profile_and_level() {
        let mut f = av1::Frame::new(16, 16, 8, av1::ChromaFormat::Yuv420);
        f.data.fill(100);
        let c = encode_tile(&f, 100).unwrap();
        assert_eq!(c.av1c[0], 0x81);
        assert_eq!(c.av1c[1] >> 5, 0, "profile 0");
        assert_eq!(c.av1c[2] & 0x0c, 0x0c, "4:2:0");
        assert!(c.obus.first().is_some_and(|b| (b >> 3) & 15 == OBU_SEQUENCE_HEADER), "no temporal delimiter");
    }
}
