//! Carrying kept metadata into an output, in the places players read it.
//!
//! Each writer takes a [`Metadata`] already narrowed to what is kept
//! ([`Metadata::kept`]) and writes all of it that the format has a standard
//! place for, and nothing else:
//!
//! | | MP4 / M4A | FLAC | MP3 |
//! |---|---|---|---|
//! | location | `mdta` `com.apple.quicktime.location.ISO6709` and `udta` `©xyz` | `LOCATION` (ISO 6709) | `TXXX:LOCATION` |
//! | device | `mdta` `…make`, `…model`, `…software`, `…camera.lens_model` | — | — |
//! | capture time | `mdta` `…creationdate`, `mvhd` creation time | `DATE` | `TDRC` |
//! | descriptive | `mdta` `…title`, `…artist`, `…album`, `…copyright`, `…comment`, `…description`, `…keywords`, `…genre`, `…composer` | the same names | `TIT2`, `TPE1`, `TALB`, `TCOP`, `COMM`, `TCON`, `TCOM` |
//!
//! Serial numbers and owner names have no standard place in a video or audio
//! file and are not written there; a still's EXIF ([`super::exif::build`])
//! carries them.

use anyhow::{Result, bail};

use super::isobmff::boxes;
use super::{Metadata, be16, be32, be64, iso6709};

/// The descriptive names every writer carries, and their QuickTime keys.
const DESCRIPTIVE: [&str; 9] = ["title", "artist", "album", "copyright", "comment", "description", "keywords", "genre", "composer"];

const QT_KEY: &str = "com.apple.quicktime.";

/// `m` as QuickTime metadata keys and their values, in writing order.
fn quicktime_items(m: &Metadata) -> Vec<(String, String)> {
    let mut items = Vec::new();
    let mut push = |key: &str, value: &Option<String>| {
        if let Some(v) = value.as_deref().map(str::trim).filter(|v| !v.is_empty()) {
            items.push((format!("{QT_KEY}{key}"), v.to_string()));
        }
    };
    if let Some(loc) = &m.location {
        push("location.ISO6709", &iso6709::format(loc));
        push("location.name", &loc.name);
    }
    push("make", &m.device.make);
    push("model", &m.device.model);
    push("software", &m.device.software);
    push("camera.lens_model", &m.device.lens);
    push("creationdate", &m.capture_time.as_deref().map(quicktime_date));
    for name in DESCRIPTIVE {
        push(name, &m.descriptive.get(name).cloned());
    }
    items
}

/// RFC 3339 in the form QuickTime writes `creationdate`: `…T12:34:56+0200`.
fn quicktime_date(t: &str) -> String {
    let b = t.as_bytes();
    let n = b.len();
    if n >= 25 && matches!(b[n - 6], b'+' | b'-') && b[n - 3] == b':' {
        return format!("{}{}", &t[..n - 3], &t[n - 2..]);
    }
    if t.ends_with('Z') && n >= 20 {
        return format!("{}+0000", &t[..n - 1]);
    }
    t.to_string()
}

struct Builder(Vec<u8>);

impl Builder {
    fn new(kind: &[u8; 4]) -> Self {
        let mut b = Vec::with_capacity(64);
        b.extend_from_slice(&[0, 0, 0, 0]);
        b.extend_from_slice(kind);
        Builder(b)
    }
    fn u32(&mut self, v: u32) -> &mut Self {
        self.0.extend_from_slice(&v.to_be_bytes());
        self
    }
    fn bytes(&mut self, v: &[u8]) -> &mut Self {
        self.0.extend_from_slice(v);
        self
    }
    fn finish(mut self) -> Vec<u8> {
        let n = self.0.len() as u32;
        self.0[..4].copy_from_slice(&n.to_be_bytes());
        self.0
    }
}

/// `moov/meta`: QuickTime metadata (no version and flags), an `mdta`
/// handler, `keys` and `ilst`.
fn build_mdta_meta(items: &[(String, String)]) -> Vec<u8> {
    let mut hdlr = Builder::new(b"hdlr");
    hdlr.u32(0).u32(0).bytes(b"mdta").u32(0).u32(0).u32(0).bytes(&[0]);
    let mut keys = Builder::new(b"keys");
    keys.u32(0).u32(items.len() as u32);
    let mut ilst = Builder::new(b"ilst");
    for (i, (key, value)) in items.iter().enumerate() {
        keys.u32(8 + key.len() as u32).bytes(b"mdta").bytes(key.as_bytes());
        let mut data = Builder::new(b"data");
        data.u32(1).u32(0).bytes(value.as_bytes()); // UTF-8, default locale
        let index = (i as u32 + 1).to_be_bytes();
        let mut item = Builder::new(&index);
        item.bytes(&data.finish());
        ilst.bytes(&item.finish());
    }
    let mut meta = Builder::new(b"meta");
    meta.bytes(&hdlr.finish()).bytes(&keys.finish()).bytes(&ilst.finish());
    meta.finish()
}

/// `moov/udta` with a `©xyz` location (what Android and ffmpeg write and
/// read).
fn build_udta(m: &Metadata) -> Option<Vec<u8>> {
    let xyz = iso6709::format(m.location.as_ref()?)?;
    let mut item = Builder::new(&[0xA9, b'x', b'y', b'z']);
    item.bytes(&(xyz.len() as u16).to_be_bytes()).bytes(&0x15C7u16.to_be_bytes()).bytes(xyz.as_bytes());
    let mut udta = Builder::new(b"udta");
    udta.bytes(&item.finish());
    Some(udta.finish())
}

/// An MP4 / M4A file with `m` written into its `moov`. The chunk offsets
/// move with the `moov` when it comes before the media. A fragmented file
/// (a CMAF segment or init) is refused: its metadata would have to be
/// repeated in every init, and no player reads it there.
pub fn mp4(file: &[u8], m: &Metadata) -> Result<Vec<u8>> {
    let items = quicktime_items(m);
    let udta = build_udta(m);
    let created = m.capture_time.as_deref().and_then(super::parse_unix_time).map(|s| s + 2_082_844_800).filter(|&s| s > 0);
    if items.is_empty() && udta.is_none() {
        return Ok(file.to_vec());
    }
    let top: Vec<_> = boxes(file).collect();
    if top.iter().any(|b| &b.kind == b"moof") {
        bail!("metadata cannot be written into a fragmented MP4");
    }
    let Some(moov) = top.iter().find(|b| &b.kind == b"moov") else { bail!("no moov box to write metadata into") };
    let mut body = Vec::with_capacity(moov.body.len() + 512);
    for c in boxes(moov.body) {
        match &c.kind {
            // Whatever was there is replaced, never merged.
            b"udta" | b"meta" => {}
            b"mvhd" => {
                let mut mvhd = moov.body[c.start..c.end].to_vec();
                if let Some(secs) = created {
                    set_mvhd_times(&mut mvhd, secs as u64)?;
                }
                body.extend_from_slice(&mvhd);
            }
            _ => body.extend_from_slice(&moov.body[c.start..c.end]),
        }
    }
    if let Some(u) = &udta {
        body.extend_from_slice(u);
    }
    if !items.is_empty() {
        body.extend_from_slice(&build_mdta_meta(&items));
    }
    let mut new_moov = Builder::new(b"moov");
    new_moov.bytes(&body);
    let mut new_moov = new_moov.finish();

    let old_len = moov.end - moov.start;
    let delta = new_moov.len() as i64 - old_len as i64;
    if delta != 0 {
        shift_chunk_offsets(&mut new_moov, moov.end as u64, delta)?;
    }
    let mut out = Vec::with_capacity(file.len() + new_moov.len() - old_len.min(new_moov.len()));
    out.extend_from_slice(&file[..moov.start]);
    out.extend_from_slice(&new_moov);
    out.extend_from_slice(&file[moov.end..]);
    Ok(out)
}

fn set_mvhd_times(mvhd: &mut [u8], secs: u64) -> Result<()> {
    match mvhd.get(8) {
        Some(0) => {
            let s = u32::try_from(secs).unwrap_or(0).to_be_bytes();
            mvhd[12..16].copy_from_slice(&s);
            mvhd[16..20].copy_from_slice(&s);
        }
        Some(1) => {
            mvhd[12..20].copy_from_slice(&secs.to_be_bytes());
            mvhd[20..28].copy_from_slice(&secs.to_be_bytes());
        }
        _ => bail!("mvhd: unknown version"),
    }
    Ok(())
}

/// Add `delta` to every `stco` / `co64` entry that points at or past
/// `from` (the old end of the `moov`: the media that moved).
fn shift_chunk_offsets(moov: &mut [u8], from: u64, delta: i64) -> Result<()> {
    let mut stack = vec![(8usize, moov.len())];
    while let Some((start, end)) = stack.pop() {
        let mut at = start;
        while at + 8 <= end {
            let size = be32(moov, at).unwrap_or(0) as usize;
            if size < 8 || at + size > end {
                break;
            }
            let kind: [u8; 4] = moov[at + 4..at + 8].try_into().unwrap_or_default();
            match &kind {
                b"trak" | b"mdia" | b"minf" | b"stbl" => stack.push((at + 8, at + size)),
                b"stco" => {
                    let n = be32(moov, at + 12).unwrap_or(0) as usize;
                    for i in 0..n {
                        let p = at + 16 + i * 4;
                        let Some(v) = be32(moov, p) else { break };
                        if u64::from(v) >= from {
                            let nv = u32::try_from(i64::from(v) + delta)
                                .map_err(|_| anyhow::anyhow!("chunk offset past 4 GiB after writing metadata"))?;
                            moov[p..p + 4].copy_from_slice(&nv.to_be_bytes());
                        }
                    }
                }
                b"co64" => {
                    let n = be32(moov, at + 12).unwrap_or(0) as usize;
                    for i in 0..n {
                        let p = at + 16 + i * 8;
                        let Some(v) = be64(moov, p) else { break };
                        if v >= from {
                            let nv = (v as i64 + delta) as u64;
                            moov[p..p + 8].copy_from_slice(&nv.to_be_bytes());
                        }
                    }
                }
                _ => {}
            }
            at += size;
        }
    }
    Ok(())
}

/// Vorbis comments for `m`: `NAME=value`.
fn vorbis_comments(m: &Metadata) -> Vec<String> {
    let mut out = Vec::new();
    for name in DESCRIPTIVE {
        if let Some(v) = m.descriptive.get(name) {
            out.push(format!("{}={v}", name.to_ascii_uppercase()));
        }
    }
    if let Some(t) = &m.capture_time {
        out.push(format!("DATE={t}"));
    }
    if let Some(loc) = &m.location {
        if let Some(s) = iso6709::format(loc).or_else(|| loc.name.clone()) {
            out.push(format!("LOCATION={s}"));
        }
    }
    out
}

/// A native FLAC file with its `VORBIS_COMMENT` block replaced by one
/// holding `m` (and no vendor string). Every other block stays.
pub fn flac(file: &[u8], m: &Metadata) -> Result<Vec<u8>> {
    let comments = vorbis_comments(m);
    let Some(blocks) = file.strip_prefix(b"fLaC") else { bail!("not a native FLAC stream") };
    let mut kept: Vec<(u8, &[u8])> = Vec::new();
    let mut at = 0;
    loop {
        let Some(&header) = blocks.get(at) else { bail!("FLAC: metadata ends early") };
        let Some(len_bytes) = blocks.get(at + 1..at + 4) else { bail!("FLAC: metadata ends early") };
        let len = (usize::from(len_bytes[0]) << 16) | (usize::from(len_bytes[1]) << 8) | usize::from(len_bytes[2]);
        let Some(body) = blocks.get(at + 4..at + 4 + len) else { bail!("FLAC: metadata ends early") };
        if header & 0x7F != 4 {
            kept.push((header & 0x7F, body));
        }
        at += 4 + len;
        if header & 0x80 != 0 {
            break;
        }
    }
    let mut vc = Vec::new();
    vc.extend_from_slice(&0u32.to_le_bytes());
    vc.extend_from_slice(&(comments.len() as u32).to_le_bytes());
    for c in &comments {
        vc.extend_from_slice(&(c.len() as u32).to_le_bytes());
        vc.extend_from_slice(c.as_bytes());
    }
    // After STREAMINFO (and SEEKTABLE), before PADDING / the rest.
    let insert_at = kept.iter().take_while(|(t, _)| *t == 0 || *t == 3).count();
    kept.insert(insert_at, (4, &vc));
    let mut out = b"fLaC".to_vec();
    for (i, (t, body)) in kept.iter().enumerate() {
        let last = if i + 1 == kept.len() { 0x80 } else { 0 };
        out.push(t | last);
        let n = body.len() as u32;
        if n >= 1 << 24 {
            bail!("FLAC: a metadata block is too large");
        }
        out.extend_from_slice(&n.to_be_bytes()[1..]);
        out.extend_from_slice(body);
    }
    out.extend_from_slice(&blocks[at..]);
    Ok(out)
}

/// An MP3 file with an ID3v2.4 tag holding `m` in front of it. Nothing to
/// write, no tag.
pub fn mp3(file: &[u8], m: &Metadata) -> Vec<u8> {
    let mut frames = Vec::new();
    let mut text_frame = |id: &[u8; 4], body: Vec<u8>| {
        frames.extend_from_slice(id);
        frames.extend_from_slice(&syncsafe(body.len() as u32));
        frames.extend_from_slice(&[0, 0]);
        frames.extend_from_slice(&body);
    };
    let utf8 = |s: &str| {
        let mut b = vec![3u8];
        b.extend_from_slice(s.as_bytes());
        b
    };
    for (name, id) in [("title", b"TIT2"), ("artist", b"TPE1"), ("album", b"TALB"), ("copyright", b"TCOP"), ("genre", b"TCON"), ("composer", b"TCOM")] {
        if let Some(v) = m.descriptive.get(name) {
            text_frame(id, utf8(v));
        }
    }
    for name in ["comment", "description"] {
        if let Some(v) = m.descriptive.get(name) {
            let mut b = vec![3u8];
            b.extend_from_slice(b"und");
            b.push(0);
            b.extend_from_slice(v.as_bytes());
            text_frame(b"COMM", b);
            break;
        }
    }
    if let Some(v) = m.descriptive.get("keywords") {
        text_frame(b"TXXX", utf8(&format!("KEYWORDS\0{v}")));
    }
    if let Some(t) = &m.capture_time {
        text_frame(b"TDRC", utf8(t));
    }
    if let Some(loc) = &m.location {
        if let Some(s) = iso6709::format(loc).or_else(|| loc.name.clone()) {
            text_frame(b"TXXX", utf8(&format!("LOCATION\0{s}")));
        }
    }
    if frames.is_empty() {
        return file.to_vec();
    }
    let mut out = b"ID3\x04\x00\x00".to_vec();
    out.extend_from_slice(&syncsafe(frames.len() as u32));
    out.extend_from_slice(&frames);
    out.extend_from_slice(file);
    out
}

fn syncsafe(n: u32) -> [u8; 4] {
    [(n >> 21) as u8 & 0x7F, (n >> 14) as u8 & 0x7F, (n >> 7) as u8 & 0x7F, n as u8 & 0x7F]
}

// ---- stills ----------------------------------------------------------------------

/// A still image with an EXIF block added: `tiff` is a TIFF structure, as
/// [`super::exif::build`] makes one. The format is read from the file's
/// first bytes: JPEG (an `APP1` after `SOI` / `APP0`), PNG (an `eXIf` chunk
/// after `IHDR`), WebP (an `EXIF` chunk, with a `VP8X` header made for a
/// simple file) or AVIF / HEIF (an `Exif` item describing the primary
/// item). `width` x `height` is the picture's size, for a WebP header.
pub fn still(file: &[u8], tiff: &[u8], width: u32, height: u32) -> Result<Vec<u8>> {
    if file.starts_with(&[0xFF, 0xD8]) {
        jpeg_exif(file, tiff)
    } else if file.starts_with(b"\x89PNG\r\n\x1a\n") {
        png_exif(file, tiff)
    } else if file.len() >= 12 && &file[..4] == b"RIFF" && &file[8..12] == b"WEBP" {
        webp_exif(file, tiff, width, height)
    } else if file.len() >= 8 && &file[4..8] == b"ftyp" {
        heif_exif(file, tiff)
    } else {
        bail!("EXIF can be written into a JPEG, PNG, WebP or AVIF, and this is none of them")
    }
}

fn jpeg_exif(file: &[u8], tiff: &[u8]) -> Result<Vec<u8>> {
    let len = 2 + 6 + tiff.len();
    if len > 0xFFFF {
        bail!("the EXIF block is too large for a JPEG segment");
    }
    // After SOI, and after a JFIF APP0 when there is one.
    let mut at = 2;
    if file.get(2..4) == Some(&[0xFF, 0xE0]) {
        at = 4 + usize::from(be16(file, 4).unwrap_or(0));
    }
    let mut out = Vec::with_capacity(file.len() + len + 2);
    out.extend_from_slice(&file[..at]);
    out.extend_from_slice(&[0xFF, 0xE1]);
    out.extend_from_slice(&(len as u16).to_be_bytes());
    out.extend_from_slice(b"Exif\0\0");
    out.extend_from_slice(tiff);
    out.extend_from_slice(&file[at..]);
    Ok(out)
}

fn png_exif(file: &[u8], tiff: &[u8]) -> Result<Vec<u8>> {
    // IHDR is first: 8 signature bytes, then 4 length + 4 type + 13 + 4 CRC.
    if file.get(12..16) != Some(b"IHDR") {
        bail!("PNG: IHDR is not the first chunk");
    }
    let at = 8 + 12 + be32(file, 8).unwrap_or(0) as usize;
    let mut chunk = Vec::with_capacity(tiff.len() + 12);
    chunk.extend_from_slice(&(tiff.len() as u32).to_be_bytes());
    chunk.extend_from_slice(b"eXIf");
    chunk.extend_from_slice(tiff);
    let crc = crc32(&chunk[4..]);
    chunk.extend_from_slice(&crc.to_be_bytes());
    let mut out = Vec::with_capacity(file.len() + chunk.len());
    out.extend_from_slice(&file[..at]);
    out.extend_from_slice(&chunk);
    out.extend_from_slice(&file[at..]);
    Ok(out)
}

/// CRC-32 (ISO 3309), as PNG chunks carry it.
fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &b in data {
        crc ^= u32::from(b);
        for _ in 0..8 {
            crc = if crc & 1 != 0 { (crc >> 1) ^ 0xEDB8_8320 } else { crc >> 1 };
        }
    }
    !crc
}

fn riff_chunk(kind: &[u8; 4], body: &[u8]) -> Vec<u8> {
    let mut c = kind.to_vec();
    c.extend_from_slice(&(body.len() as u32).to_le_bytes());
    c.extend_from_slice(body);
    if body.len() % 2 == 1 {
        c.push(0);
    }
    c
}

fn webp_exif(file: &[u8], tiff: &[u8], width: u32, height: u32) -> Result<Vec<u8>> {
    let chunks = &file[12..];
    let mut body = b"WEBP".to_vec();
    match chunks.get(..4) {
        Some(b"VP8X") => {
            let mut rest = chunks.to_vec();
            rest[8] |= 0x08; // EXIF present
            body.extend_from_slice(&rest);
        }
        Some(kind @ (b"VP8 " | b"VP8L")) => {
            // A simple file: a VP8X header first, which is where the EXIF
            // flag lives. A lossless bitstream says whether it has alpha.
            let alpha = kind == b"VP8L" && chunks.get(12).is_some_and(|b| b & 0x10 != 0);
            let mut vp8x = vec![0x08 | if alpha { 0x10 } else { 0 }, 0, 0, 0];
            vp8x.extend_from_slice(&(width.max(1) - 1).to_le_bytes()[..3]);
            vp8x.extend_from_slice(&(height.max(1) - 1).to_le_bytes()[..3]);
            body.extend(riff_chunk(b"VP8X", &vp8x));
            body.extend_from_slice(chunks);
        }
        _ => bail!("WebP: no image chunk where one was expected"),
    }
    body.extend(riff_chunk(b"EXIF", tiff));
    let mut out = b"RIFF".to_vec();
    out.extend_from_slice(&(body.len() as u32).to_le_bytes());
    out.extend(body);
    Ok(out)
}

/// An `iloc` item, as parsed and as rewritten.
struct IlocItem {
    id: u32,
    method: u16,
    data_ref: u16,
    base: u64,
    /// (index, offset, length)
    extents: Vec<(u64, u64, u64)>,
}

/// `iloc` as read: its field widths and items.
struct Iloc {
    head: Vec<u8>,
    version: u8,
    offset_size: usize,
    length_size: usize,
    base_size: usize,
    index_size: usize,
    items: Vec<IlocItem>,
}

fn read_iloc(b: &[u8]) -> Result<Iloc> {
    let (Some(&version), Some(&sizes), Some(&sizes2)) = (b.first(), b.get(4), b.get(5)) else {
        bail!("HEIF: iloc too short")
    };
    let (offset_size, length_size) = (usize::from(sizes >> 4), usize::from(sizes & 15));
    let base_size = usize::from(sizes2 >> 4);
    let index_size = if version >= 1 { usize::from(sizes2 & 15) } else { 0 };
    let read_n = |at: &mut usize, n: usize| -> Result<u64> {
        let mut v = 0u64;
        for i in 0..n {
            v = (v << 8) | u64::from(*b.get(*at + i).ok_or_else(|| anyhow::anyhow!("HEIF: iloc ends early"))?);
        }
        *at += n;
        Ok(v)
    };
    let id_size = if version < 2 { 2 } else { 4 };
    let mut at = 6;
    let count = read_n(&mut at, id_size)?;
    let mut items = Vec::new();
    for _ in 0..count {
        let id = read_n(&mut at, id_size)? as u32;
        let method = if version >= 1 { (read_n(&mut at, 2)? & 15) as u16 } else { 0 };
        let data_ref = read_n(&mut at, 2)? as u16;
        let base = read_n(&mut at, base_size)?;
        let n = read_n(&mut at, 2)?;
        let mut extents = Vec::new();
        for _ in 0..n {
            extents.push((read_n(&mut at, index_size)?, read_n(&mut at, offset_size)?, read_n(&mut at, length_size)?));
        }
        items.push(IlocItem { id, method, data_ref, base, extents });
    }
    Ok(Iloc { head: b[..6].to_vec(), version, offset_size, length_size, base_size, index_size, items })
}

fn write_n(out: &mut Vec<u8>, v: u64, n: usize) {
    out.extend_from_slice(&v.to_be_bytes()[8 - n..]);
}

impl Iloc {
    /// The box body, with file offsets at or past `moved_from` moved by
    /// `delta`.
    fn body(&self, moved_from: u64, delta: u64) -> Vec<u8> {
        let id_size = if self.version < 2 { 2 } else { 4 };
        let mut out = self.head.clone();
        write_n(&mut out, self.items.len() as u64, id_size);
        for item in &self.items {
            write_n(&mut out, u64::from(item.id), id_size);
            if self.version >= 1 {
                write_n(&mut out, u64::from(item.method), 2);
            }
            write_n(&mut out, u64::from(item.data_ref), 2);
            let in_file = item.method == 0 && item.data_ref == 0;
            let base_moves = in_file && item.base > 0 && item.base >= moved_from;
            write_n(&mut out, if base_moves { item.base + delta } else { item.base }, self.base_size);
            write_n(&mut out, item.extents.len() as u64, 2);
            for &(index, offset, length) in &item.extents {
                write_n(&mut out, index, self.index_size);
                let moves = in_file && !base_moves && item.base + offset >= moved_from;
                write_n(&mut out, if moves { offset + delta } else { offset }, self.offset_size);
                write_n(&mut out, length, self.length_size);
            }
        }
        out
    }
}

fn heif_exif(file: &[u8], tiff: &[u8]) -> Result<Vec<u8>> {
    let top: Vec<_> = boxes(file).collect();
    let Some(meta) = top.iter().find(|b| &b.kind == b"meta") else { bail!("HEIF: no meta box") };
    if meta.body.len() < 4 {
        bail!("HEIF: meta too short");
    }
    let (version_flags, children) = meta.body.split_at(4);
    let kids: Vec<_> = boxes(children).collect();
    let find = |k: &[u8; 4]| kids.iter().find(|b| &b.kind == k);
    let primary = match find(b"pitm") {
        Some(p) if p.body.first() == Some(&0) => u32::from(be16(p.body, 4).unwrap_or(1)),
        Some(p) => be32(p.body, 4).unwrap_or(1),
        None => bail!("HEIF: no primary item"),
    };
    let Some(iloc_box) = find(b"iloc") else { bail!("HEIF: no iloc") };
    let Some(iinf) = find(b"iinf") else { bail!("HEIF: no iinf") };
    let mut iloc = read_iloc(iloc_box.body)?;
    if iloc.offset_size < 4 || iloc.length_size < 4 {
        bail!("HEIF: iloc offsets too narrow to point at new data");
    }
    let exif_id = iloc.items.iter().map(|i| i.id).max().unwrap_or(0).max(primary) + 1;
    if iloc.version < 2 && exif_id > 0xFFFF {
        bail!("HEIF: no item id left");
    }
    // The item: a 4-byte offset to the TIFF header (0), then the TIFF.
    let mut payload = 0u32.to_be_bytes().to_vec();
    payload.extend_from_slice(tiff);
    iloc.items.push(IlocItem { id: exif_id, method: 0, data_ref: 0, base: 0, extents: vec![(0, 0, payload.len() as u64)] });

    // iinf with one more `infe` (version 2: 16-bit ids; 3: 32-bit).
    let iv = iinf.body.first().copied().unwrap_or(0);
    let head = if iv == 0 { 6 } else { 8 };
    let mut iinf_body = iinf.body[..4].to_vec();
    if iv == 0 {
        iinf_body.extend_from_slice(&(be16(iinf.body, 4).unwrap_or(0) + 1).to_be_bytes());
    } else {
        iinf_body.extend_from_slice(&(be32(iinf.body, 4).unwrap_or(0) + 1).to_be_bytes());
    }
    iinf_body.extend_from_slice(&iinf.body[head..]);
    let mut infe = Builder::new(b"infe");
    if exif_id > 0xFFFF {
        infe.u32(0x0300_0000).u32(exif_id).bytes(&0u16.to_be_bytes());
    } else {
        infe.u32(0x0200_0000).bytes(&(exif_id as u16).to_be_bytes()).bytes(&0u16.to_be_bytes());
    }
    infe.bytes(b"Exif").bytes(&[0]);
    iinf_body.extend(infe.finish());

    // iref: the Exif item describes (`cdsc`) the primary item.
    let wide = exif_id > 0xFFFF || primary > 0xFFFF;
    let mut cdsc = Builder::new(b"cdsc");
    if wide {
        cdsc.u32(exif_id).bytes(&1u16.to_be_bytes()).u32(primary);
    } else {
        cdsc.bytes(&(exif_id as u16).to_be_bytes()).bytes(&1u16.to_be_bytes()).bytes(&(primary as u16).to_be_bytes());
    }
    let cdsc = cdsc.finish();

    let build_meta = |iloc: &Iloc, delta: u64| -> Vec<u8> {
        let mut meta_body = version_flags.to_vec();
        let mut had_iref = false;
        for k in &kids {
            match &k.kind {
                b"iloc" => {
                    let mut bx = Builder::new(b"iloc");
                    bx.bytes(&iloc.body(meta.end as u64, delta));
                    meta_body.extend(bx.finish());
                }
                b"iinf" => {
                    let mut bx = Builder::new(b"iinf");
                    bx.bytes(&iinf_body);
                    meta_body.extend(bx.finish());
                }
                b"iref" if (k.body.first() == Some(&1)) == wide => {
                    had_iref = true;
                    let mut bx = Builder::new(b"iref");
                    bx.bytes(k.body).bytes(&cdsc);
                    meta_body.extend(bx.finish());
                }
                _ => meta_body.extend_from_slice(&children[k.start..k.end]),
            }
        }
        if !had_iref {
            let mut bx = Builder::new(b"iref");
            bx.u32(if wide { 0x0100_0000 } else { 0 }).bytes(&cdsc);
            meta_body.extend(bx.finish());
        }
        let mut bx = Builder::new(b"meta");
        bx.bytes(&meta_body);
        bx.finish()
    };
    // The meta box's size does not depend on the offsets in it. The new
    // item's data goes last, in an `mdat` of its own: at the old file's end
    // (plus that box's header), moved by the meta box's growth like every
    // other offset past it.
    let old_len = (meta.end - meta.start) as u64;
    let delta = build_meta(&iloc, 0).len() as u64 - old_len;
    if let Some(item) = iloc.items.last_mut() {
        item.extents[0].1 = file.len() as u64 + 8;
    }
    let new_meta = build_meta(&iloc, delta);
    let mut out = Vec::with_capacity(file.len() + new_meta.len() + payload.len() + 8);
    out.extend_from_slice(&file[..meta.start]);
    out.extend_from_slice(&new_meta);
    out.extend_from_slice(&file[meta.end..]);
    let mut mdat = Builder::new(b"mdat");
    mdat.bytes(&payload);
    out.extend(mdat.finish());
    Ok(out)
}
