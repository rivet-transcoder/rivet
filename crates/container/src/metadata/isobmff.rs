//! ISO base media files (MP4, MOV, M4A, 3GP, HEIF, AVIF): `moov` and track
//! times, `udta` (QuickTime `©xxx`, 3GPP strings and `loci`, maker boxes),
//! `meta` with `mdta` keys (QuickTime) or an iTunes `ilst`, HEIF `Exif` / XMP
//! items, XMP `uuid` boxes, and the timed metadata tracks a camera records
//! beside the picture.

use super::{Category, Categories, DeviceField, Location, Metadata, TimedTrack, TimedTrackKind, be16, be32, be64};

const XMP_UUID: [u8; 16] = [0xBE, 0x7A, 0xCF, 0xCB, 0x97, 0xA9, 0x42, 0xE8, 0x9C, 0x71, 0x99, 0x94, 0x91, 0xE3, 0xAF, 0xAC];

/// A box: its type, and its body (after the header, and after the 16-byte
/// extended type of a `uuid` box, which is `uuid`).
pub(crate) struct Mp4Box<'a> {
    pub kind: [u8; 4],
    pub uuid: Option<[u8; 16]>,
    pub body: &'a [u8],
    /// Offset of the box's first byte in the slice iterated.
    pub start: usize,
    /// Offset one past its last byte.
    pub end: usize,
}

pub(crate) fn boxes(data: &[u8]) -> impl Iterator<Item = Mp4Box<'_>> {
    let mut at = 0usize;
    std::iter::from_fn(move || {
        if at + 8 > data.len() {
            return None;
        }
        let size32 = be32(data, at)?;
        let kind: [u8; 4] = data[at + 4..at + 8].try_into().ok()?;
        let (header, size) = match size32 {
            0 => (8, data.len() - at),
            1 => (16, usize::try_from(be64(data, at + 8)?).ok()?),
            n => (8, n as usize),
        };
        if size < header || at.checked_add(size)? > data.len() {
            return None;
        }
        let start = at;
        let mut body = &data[at + header..at + size];
        let mut uuid = None;
        if &kind == b"uuid" {
            uuid = Some(body.get(..16)?.try_into().ok()?);
            body = &body[16..];
        }
        at += size;
        Some(Mp4Box { kind, uuid, body, start, end: at })
    })
}

fn child<'a>(data: &'a [u8], kind: &[u8; 4]) -> Option<&'a [u8]> {
    boxes(data).find(|b| &b.kind == kind).map(|b| b.body)
}

pub(crate) fn looks_like(data: &[u8]) -> bool {
    matches!(
        &data[4..8],
        b"ftyp" | b"moov" | b"mdat" | b"free" | b"wide" | b"skip" | b"styp" | b"moof" | b"sidx" | b"meta" | b"pnot"
    )
}

pub(crate) fn read(data: &[u8], m: &mut Metadata) {
    for b in boxes(data) {
        match &b.kind {
            b"moov" => read_moov(b.body, data, m),
            b"meta" => read_meta(b.body, data, m, "meta"),
            b"udta" => read_udta(b.body, data, m, "udta"),
            b"uuid" if b.uuid == Some(XMP_UUID) => super::xmp::read(b.body, m),
            b"uuid" => m.unclassified(String::from("top-level uuid box")),
            _ => {}
        }
    }
}

fn read_moov(moov: &[u8], file: &[u8], m: &mut Metadata) {
    // `mvhd`'s time is UTC with no offset: a `creationdate` key, where there
    // is one, says the same moment in the recorder's local time.
    let mut created = None;
    for b in boxes(moov) {
        match &b.kind {
            b"mvhd" => created = read_times(b.body, m),
            b"trak" => read_trak(b.body, file, m),
            b"udta" => read_udta(b.body, file, m, "udta"),
            b"meta" => read_meta(b.body, file, m, "meta"),
            b"uuid" if b.uuid == Some(XMP_UUID) => super::xmp::read(b.body, m),
            _ => {}
        }
    }
    if let Some(secs) = created {
        m.set_capture_time(&super::quicktime_time(secs));
    }
}

/// `mvhd` / `tkhd` / `mdhd`: creation and modification times, seconds since
/// 1904. Zero is "not recorded".
/// Returns the creation time when there is one.
fn read_times(body: &[u8], m: &mut Metadata) -> Option<u64> {
    let (created, modified) = match body.first() {
        Some(1) => (be64(body, 4), be64(body, 12)),
        Some(0) => (be32(body, 4).map(u64::from), be32(body, 8).map(u64::from)),
        _ => return None,
    };
    let (created, modified) = (created.unwrap_or(0), modified.unwrap_or(0));
    if created != 0 || modified != 0 {
        m.present.insert(Category::CaptureTime);
    }
    (created != 0).then_some(created)
}

fn read_trak(trak: &[u8], file: &[u8], m: &mut Metadata) {
    let mut handler = [0u8; 4];
    let mut entry: Option<([u8; 4], &[u8])> = None;
    let mut stbl: Option<&[u8]> = None;
    for b in boxes(trak) {
        match &b.kind {
            b"tkhd" => {
                read_times(b.body, m);
            }
            b"udta" => read_udta(b.body, file, m, "trak/udta"),
            b"meta" => read_meta(b.body, file, m, "trak/meta"),
            b"mdia" => {
                for c in boxes(b.body) {
                    match &c.kind {
                        b"mdhd" => {
                            read_times(c.body, m);
                        }
                        b"hdlr" => {
                            if let Some(h) = c.body.get(8..12) {
                                handler.copy_from_slice(h);
                            }
                        }
                        b"minf" => {
                            stbl = child(c.body, b"stbl");
                            let stsd = stbl.and_then(|stbl| child(stbl, b"stsd"));
                            if let Some(first) = stsd.and_then(|s| s.get(8..)).and_then(|s| boxes(s).next()) {
                                entry = Some((first.kind, first.body));
                            }
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }
    // An audio track's first and last packets: where an encoder names itself.
    if &handler == b"soun" {
        for (at, len) in stbl.map(first_last_samples).unwrap_or_default() {
            if let Some(packet) = file.get(at..at.saturating_add(len)) {
                for ident in super::scrub::encoder_idents(packet) {
                    if !m.embedded_software.contains(&ident) {
                        m.embedded_software.push(ident);
                    }
                }
            }
        }
    }
    let Some((fourcc, entry_body)) = entry else {
        if &handler == b"tmcd" {
            m.timed_tracks.push(timecode());
        }
        return;
    };
    // A FLAC track's `dfLa` holds the stream's metadata blocks, Vorbis
    // comments and pictures among them.
    if &fourcc == b"fLaC" {
        if let Some(blocks) = entry_body.get(28..).and_then(|c| child(c, b"dfLa")).and_then(|d| d.get(4..)) {
            super::audio::read_flac_blocks(blocks, m);
        }
    }
    let track = match (&handler, &fourcc) {
        (_, b"mebx") => Some(mebx_track(entry_body)),
        (_, b"gpmd") => Some(TimedTrack {
            kind: TimedTrackKind::Telemetry,
            label: "GoPro telemetry (GPMF), which can include GPS".into(),
            categories: Categories::NONE.with(Category::Location).with(Category::Device).with(Category::CaptureTime),
        }),
        (_, b"camm") => Some(TimedTrack {
            kind: TimedTrackKind::Telemetry,
            label: "Camera motion metadata (camm), which can include GPS".into(),
            categories: Categories::NONE.with(Category::Location),
        }),
        (_, b"rtmd") => Some(TimedTrack {
            kind: TimedTrackKind::Telemetry,
            label: "Sony real-time metadata, which can include GPS".into(),
            categories: Categories::NONE.with(Category::Location).with(Category::Device).with(Category::CaptureTime),
        }),
        (_, b"tmcd") | (b"tmcd", _) => Some(timecode()),
        (b"meta", other) => Some(TimedTrack {
            kind: TimedTrackKind::Other,
            label: format!("Timed metadata track ({})", String::from_utf8_lossy(other)),
            categories: Categories::NONE,
        }),
        _ => None,
    };
    if let Some(t) = track {
        m.timed_tracks.push(t);
    }
}

/// (offset, size) of a track's first two and last two samples, from its
/// `stsz`, `stsc` and `stco` / `co64`.
fn first_last_samples(stbl: &[u8]) -> Vec<(usize, usize)> {
    let (Some(stsz), Some(stsc)) = (child(stbl, b"stsz"), child(stbl, b"stsc")) else { return Vec::new() };
    let (offsets, wide) = match (child(stbl, b"stco"), child(stbl, b"co64")) {
        (Some(c), _) => (c, false),
        (None, Some(c)) => (c, true),
        _ => return Vec::new(),
    };
    let fixed = be32(stsz, 4).unwrap_or(0) as usize;
    let count = be32(stsz, 8).unwrap_or(0) as usize;
    let size = |i: usize| if fixed != 0 { Some(fixed) } else { be32(stsz, 12 + 4 * i).map(|v| v as usize) };
    let chunks = be32(offsets, 4).unwrap_or(0) as usize;
    let chunk_offset = |c: usize| if wide { be64(offsets, 8 + 8 * c).map(|v| v as usize) } else { be32(offsets, 8 + 4 * c).map(|v| v as usize) };
    let runs: Vec<(usize, usize)> = (0..be32(stsc, 4).unwrap_or(0) as usize)
        .filter_map(|r| Some((be32(stsc, 8 + 12 * r)? as usize, be32(stsc, 12 + 12 * r)? as usize)))
        .collect();
    let wanted: Vec<usize> = [0, 1, count.saturating_sub(2), count.saturating_sub(1)]
        .into_iter()
        .filter(|&i| i < count)
        .collect();
    let mut out = Vec::new();
    let mut sample = 0usize;
    for chunk in 0..chunks.min(1 << 20) {
        let per = runs.iter().take_while(|(first, _)| *first <= chunk + 1).last().map_or(0, |r| r.1);
        if per == 0 {
            break;
        }
        if wanted.iter().any(|&w| w >= sample && w < sample + per) {
            let Some(mut at) = chunk_offset(chunk) else { break };
            for i in sample..sample + per {
                let Some(len) = size(i) else { break };
                if wanted.contains(&i) && !out.contains(&(at, len)) {
                    out.push((at, len));
                }
                at += len;
            }
        }
        sample += per;
        if sample >= count {
            break;
        }
    }
    out
}

fn timecode() -> TimedTrack {
    TimedTrack {
        kind: TimedTrackKind::Timecode,
        label: "Timecode track".into(),
        categories: Categories::NONE.with(Category::CaptureTime),
    }
}

/// A QuickTime `mebx` sample entry: its `keys` name what the track carries.
fn mebx_track(entry: &[u8]) -> TimedTrack {
    let keys: Vec<String> = entry
        .get(8..)
        .and_then(|b| child(b, b"keys"))
        .map(|keys| {
            boxes(keys)
                .filter_map(|k| child(k.body, b"keyd").and_then(|d| d.get(4..)).map(super::text))
                .collect()
        })
        .unwrap_or_default();
    if keys.iter().any(|k| k.contains("location")) {
        TimedTrack {
            kind: TimedTrackKind::Location,
            label: "Apple location track".into(),
            categories: Categories::NONE.with(Category::Location),
        }
    } else {
        let what = if keys.iter().any(|k| k.contains("motion") || k.contains("orientation")) {
            "Apple motion track"
        } else if keys.iter().any(|k| k.contains("face")) {
            "Apple face-detection track"
        } else {
            "Apple timed metadata track"
        };
        TimedTrack { kind: TimedTrackKind::Other, label: what.into(), categories: Categories::NONE }
    }
}

fn read_udta(udta: &[u8], file: &[u8], m: &mut Metadata, path: &str) {
    for b in boxes(udta) {
        let k = &b.kind;
        match k {
            b"meta" => read_meta(b.body, file, m, &format!("{path}/meta")),
            b"XMP_" => super::xmp::read(b.body, m),
            b"uuid" if b.uuid == Some(XMP_UUID) => super::xmp::read(b.body, m),
            // 3GPP TS 26.244 location.
            b"loci" => read_loci(b.body, m),
            // 3GPP strings: FullBox, language, text.
            b"titl" | b"auth" | b"perf" | b"dscp" | b"cprt" | b"gnre" | b"albm" => {
                if let Some(t) = b.body.get(6..) {
                    let key = match k {
                        b"titl" => "title",
                        b"auth" | b"perf" => "artist",
                        b"dscp" => "description",
                        b"cprt" => "copyright",
                        b"gnre" => "genre",
                        _ => "album",
                    };
                    m.set_descriptive(key, &text_3gpp(t));
                }
            }
            b"kywd" | b"rtng" | b"clsf" => m.present.insert(Category::Descriptive),
            b"yrrc" => {
                if let Some(y) = be16(b.body, 4).filter(|&y| y != 0) {
                    m.set_capture_time(&y.to_string());
                }
            }
            // A track name.
            b"name" => m.set_descriptive("track_name", &super::text(b.body)),
            // GoPro.
            b"FIRM" => m.set_device(DeviceField::Software, &super::text(b.body)),
            b"LENS" => m.set_device(DeviceField::Lens, &super::text(b.body)),
            b"CAME" | b"MUID" | b"GUMI" | b"BCID" => m.set_device(DeviceField::Serial, &hex(b.body)),
            b"GPMF" | b"SETT" | b"MINF" | b"HMMT" | b"AMBA" | b"MTRX" => m.present.insert(Category::Device),
            // Hint information: not metadata.
            b"hnti" | b"hinf" | b"free" | b"skip" | b"wide" => {}
            _ if k[0] == 0xA9 => {
                let value = quicktime_text(b.body);
                copyright_atom(m, k, &value, path);
            }
            _ => m.unclassified(format!("{path}/{}", fourcc(k))),
        }
    }
}

/// A QuickTime user-data text item: one or more `(u16 length, u16 language,
/// text)` records, or an iTunes-style `data` child.
fn quicktime_text(body: &[u8]) -> String {
    if let Some(d) = child(body, b"data") {
        return data_value(d);
    }
    match be16(body, 0) {
        Some(len) if usize::from(len) + 4 <= body.len() => super::text(&body[4..4 + usize::from(len)]),
        _ => super::text(body),
    }
}

/// A `©xxx` item (in `udta` or an iTunes `ilst`).
fn copyright_atom(m: &mut Metadata, k: &[u8; 4], value: &str, path: &str) {
    match &k[1..] {
        b"xyz" => m.set_location_text(value),
        b"mak" => m.set_device(DeviceField::Make, value),
        b"mod" => m.set_device(DeviceField::Model, value),
        b"swr" | b"too" | b"swf" | b"fmt" | b"src" => m.set_device(DeviceField::Software, value),
        b"day" => m.set_capture_time(value),
        b"nam" => m.set_descriptive("title", value),
        b"ART" | b"aut" | b"prf" => m.set_descriptive("artist", value),
        b"cpy" => m.set_descriptive("copyright", value),
        b"cmt" => m.set_descriptive("comment", value),
        b"des" => m.set_descriptive("description", value),
        b"inf" => m.set_descriptive("information", value),
        b"alb" => m.set_descriptive("album", value),
        b"gen" => m.set_descriptive("genre", value),
        b"wrt" => m.set_descriptive("composer", value),
        b"dir" => m.set_descriptive("director", value),
        b"prd" => m.set_descriptive("producer", value),
        b"enc" => m.set_descriptive("encoded_by", value),
        b"lyr" => m.set_descriptive("lyrics", value),
        b"grp" => m.set_descriptive("grouping", value),
        b"key" => m.set_descriptive("keywords", value),
        _ => {
            if value.trim().is_empty() {
                m.unclassified(format!("{path}/{}", fourcc(k)));
            } else {
                m.set_descriptive(&format!("udta_{}", String::from_utf8_lossy(&k[1..]).to_ascii_lowercase()), value);
            }
        }
    }
}

/// 3GPP `loci`: FullBox, language, name, role, longitude, latitude and
/// altitude as 16.16 fixed point, astronomical body, notes.
fn read_loci(body: &[u8], m: &mut Metadata) {
    let Some(rest) = body.get(6..) else { return };
    let name_len = rest.iter().position(|&b| b == 0).unwrap_or(rest.len());
    let name = text_3gpp(&rest[..name_len]);
    let at = name_len + 1 + 1;
    let fixed = |i: usize| be32(rest, at + i * 4).map(|v| f64::from(v as i32) / 65536.0);
    let mut loc = match (fixed(0), fixed(1)) {
        (Some(lon), Some(lat)) => Location::coordinates(lat, lon, fixed(2)),
        _ => Location::default(),
    };
    if !name.trim().is_empty() {
        loc.name = Some(name);
    }
    m.set_location(loc);
    m.present.insert(Category::Location);
}

/// A 3GPP string: UTF-8, or UTF-16 after a byte-order mark.
fn text_3gpp(b: &[u8]) -> String {
    match b {
        [0xFE, 0xFF, rest @ ..] => super::utf16(rest, true),
        [0xFF, 0xFE, rest @ ..] => super::utf16(rest, false),
        _ => super::text(b),
    }
}

fn read_meta(meta: &[u8], file: &[u8], m: &mut Metadata, path: &str) {
    // QuickTime's `meta` has no version and flags; ISO's does.
    let body = if meta.get(4..8) == Some(b"hdlr") { meta } else { meta.get(4..).unwrap_or_default() };
    let handler: [u8; 4] = child(body, b"hdlr").and_then(|h| h.get(8..12)).and_then(|h| h.try_into().ok()).unwrap_or_default();
    let keys: Vec<String> = child(body, b"keys").map(read_keys).unwrap_or_default();
    for b in boxes(body) {
        match &b.kind {
            b"hdlr" | b"keys" | b"iloc" | b"idat" | b"pitm" | b"iprp" | b"iref" | b"dinf" | b"free" | b"grpl" => {}
            b"ilst" => read_ilst(b.body, &keys, &handler, m, path),
            b"iinf" => read_heif_items(body, file, m),
            b"ID32" => {
                if let Some(tag) = b.body.get(6..) {
                    super::audio::read_id3(tag, m);
                }
            }
            b"xml " | b"bxml" => {
                if let Some(x) = b.body.get(4..) {
                    super::xmp::read(x, m);
                }
                m.present.insert(Category::Descriptive);
            }
            b"uuid" if b.uuid == Some(XMP_UUID) => super::xmp::read(b.body, m),
            k => m.unclassified(format!("{path}/{}", fourcc(k))),
        }
    }
}

/// `keys`: FullBox, count, then (size, namespace, name) records.
fn read_keys(keys: &[u8]) -> Vec<String> {
    let mut out = Vec::new();
    let mut at = 8;
    while let Some(size) = be32(keys, at).map(|s| s as usize) {
        if size < 8 {
            break;
        }
        let Some(name) = keys.get(at + 8..at + size) else { break };
        out.push(super::text(name));
        at += size;
    }
    out
}

fn read_ilst(ilst: &[u8], keys: &[String], handler: &[u8; 4], m: &mut Metadata, path: &str) {
    for item in boxes(ilst) {
        let value = child(item.body, b"data").map(data_value).unwrap_or_default();
        let is_picture = child(item.body, b"data").and_then(|d| be32(d, 0)).is_some_and(|t| matches!(t & 0xFF_FFFF, 13 | 14 | 27));
        if handler == b"mdta" || (handler != b"mdir" && item.kind[0] == 0 && !keys.is_empty()) {
            let index = u32::from_be_bytes(item.kind) as usize;
            match index.checked_sub(1).and_then(|i| keys.get(i)) {
                Some(_) if is_picture => m.set_descriptive("picture", "present"),
                Some(key) => mdta_key(m, key, &value, path),
                None => m.unclassified(format!("{path}/ilst item {index}")),
            }
            continue;
        }
        let k = &item.kind;
        match k {
            _ if k[0] == 0xA9 => copyright_atom(m, k, &value, path),
            b"covr" => m.set_descriptive("picture", "present"),
            b"aART" => m.set_descriptive("album_artist", &value),
            b"cprt" => m.set_descriptive("copyright", &value),
            b"desc" | b"ldes" => m.set_descriptive("description", &value),
            b"purd" => m.set_descriptive("purchase_date", &value),
            b"apID" | b"ownr" => m.set_device(DeviceField::Owner, &value),
            b"----" => freeform(item.body, &value, m, path),
            _ => m.set_descriptive(&format!("itunes_{}", String::from_utf8_lossy(k).to_ascii_lowercase()), if value.is_empty() { "present" } else { &value }),
        }
    }
}

/// An iTunes `----` item: `mean`, `name`, `data`.
fn freeform(body: &[u8], value: &str, m: &mut Metadata, path: &str) {
    let name = child(body, b"name").and_then(|n| n.get(4..)).map(super::text).unwrap_or_default();
    match name.as_str() {
        // Gapless playback and loudness: how to play it, not who made it.
        "iTunSMPB" | "iTunNORM" | "iTunes_CDDB_IDs" => {}
        _ => {
            if !m.set_by_name(&name, value) {
                m.unclassified(format!("{path}/---- {name}"));
            }
        }
    }
}

/// A QuickTime metadata key (`com.apple.quicktime.make`, …) and its value.
pub(crate) fn mdta_key(m: &mut Metadata, key: &str, value: &str, path: &str) {
    let suffix = key.strip_prefix("com.apple.quicktime.").or_else(|| key.strip_prefix("com.android.")).unwrap_or(key);
    match suffix {
        "location.ISO6709" => m.set_location_text(value),
        "location.name" => m.set_location(Location { name: Some(value.to_string()), ..Default::default() }),
        s if s.starts_with("location.") => m.present.insert(Category::Location),
        "make" | "manufacturer" => m.set_device(DeviceField::Make, value),
        "model" => m.set_device(DeviceField::Model, value),
        "software" | "version" => m.set_device(DeviceField::Software, value),
        "camera.lens_model" => m.set_device(DeviceField::Lens, value),
        s if s.starts_with("camera.") => m.present.insert(Category::Device),
        "creationdate" => m.set_capture_time(value),
        "content.identifier" => m.set_device(DeviceField::Serial, value),
        // How to play the file, not who made it.
        "capture.fps" | "full-frame-rate-playback-intent" | "still-image-time" | "live-photo.auto" | "live-photo.vitality-score"
        | "live-photo.vitality-scoring-version" | "video-orientation" => {}
        s => {
            let short = s.rsplit('.').next().unwrap_or(s);
            if !m.set_by_name(short, value) {
                m.unclassified(format!("{path}/mdta {key}"));
            }
        }
    }
}

/// A `data` atom's value as text: type indicator, locale, payload.
fn data_value(d: &[u8]) -> String {
    let Some(kind) = be32(d, 0) else { return String::new() };
    let payload = d.get(8..).unwrap_or_default();
    match kind & 0xFF_FFFF {
        1 | 4 => super::text(payload),
        2 | 5 => super::utf16(payload, true),
        21 => match payload.len() {
            1 => (payload[0] as i8).to_string(),
            2 => (i16::from_be_bytes([payload[0], payload[1]])).to_string(),
            4 => (be32(payload, 0).unwrap_or(0) as i32).to_string(),
            8 => (be64(payload, 0).unwrap_or(0) as i64).to_string(),
            _ => hex(payload),
        },
        22 => match payload.len() {
            1 => payload[0].to_string(),
            2 => be16(payload, 0).unwrap_or(0).to_string(),
            4 => be32(payload, 0).unwrap_or(0).to_string(),
            8 => be64(payload, 0).unwrap_or(0).to_string(),
            _ => hex(payload),
        },
        23 => be32(payload, 0).map(|v| f32::from_bits(v).to_string()).unwrap_or_default(),
        24 => be64(payload, 0).map(|v| f64::from_bits(v).to_string()).unwrap_or_default(),
        13 | 14 | 27 => format!("{} bytes", payload.len()),
        _ => {
            let t = super::text(payload);
            if t.chars().all(|c| !c.is_control()) { t } else { hex(payload) }
        }
    }
}

fn hex(b: &[u8]) -> String {
    let b = match b.iter().rposition(|&x| x != 0) {
        Some(end) => &b[..=end],
        None => return String::new(),
    };
    if b.iter().all(|c| c.is_ascii_graphic() || *c == b' ') {
        return String::from_utf8_lossy(b).into_owned();
    }
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn fourcc(k: &[u8; 4]) -> String {
    k.iter().map(|&c| if c.is_ascii_graphic() || c == b' ' { c as char } else if c == 0xA9 { '©' } else { '?' }).collect()
}

// ---- HEIF / AVIF items -----------------------------------------------------------

/// The `Exif` and XMP items of a HEIF `meta`.
fn read_heif_items(meta: &[u8], file: &[u8], m: &mut Metadata) {
    let Some(iinf) = child(meta, b"iinf") else { return };
    let version = iinf.first().copied().unwrap_or(0);
    let entries = if version == 0 { iinf.get(6..) } else { iinf.get(8..) }.unwrap_or_default();
    let mut wanted: Vec<(u32, bool)> = Vec::new(); // (item id, is_exif)
    for infe in boxes(entries).filter(|b| &b.kind == b"infe") {
        let b = infe.body;
        let v = b.first().copied().unwrap_or(0);
        if v < 2 {
            continue;
        }
        let (id, at) = if v == 2 { (be16(b, 4).map(u32::from), 8) } else { (be32(b, 4), 10) };
        let (Some(id), Some(kind)) = (id, b.get(at..at + 4)) else { continue };
        match kind {
            b"Exif" => wanted.push((id, true)),
            b"mime" => {
                let rest = &b[at + 4..];
                let name_end = rest.iter().position(|&c| c == 0).unwrap_or(rest.len());
                let ct = super::text(rest.get(name_end + 1..).unwrap_or_default());
                if ct.contains("rdf+xml") || ct.contains("xmp") {
                    wanted.push((id, false));
                }
            }
            _ => {}
        }
    }
    if wanted.is_empty() {
        return;
    }
    let idat = child(meta, b"idat").unwrap_or_default();
    let Some(iloc) = child(meta, b"iloc") else { return };
    for (id, is_exif) in wanted {
        let Some(bytes) = heif_item(iloc, id, file, idat) else { continue };
        if is_exif {
            // A 4-byte offset to the TIFF header, then the block.
            let skip = be32(&bytes, 0).unwrap_or(0) as usize;
            if let Some(tiff) = bytes.get(4 + skip..) {
                super::exif::read_exif(tiff, m);
            }
        } else {
            super::xmp::read(&bytes, m);
        }
    }
}

/// An item's bytes, through `iloc` (ISO/IEC 14496-12 §8.11.3).
fn heif_item(iloc: &[u8], want: u32, file: &[u8], idat: &[u8]) -> Option<Vec<u8>> {
    let version = *iloc.first()?;
    let sizes = *iloc.get(4)?;
    let (offset_size, length_size) = (usize::from(sizes >> 4), usize::from(sizes & 15));
    let sizes2 = *iloc.get(5)?;
    let base_size = usize::from(sizes2 >> 4);
    let index_size = if version >= 1 { usize::from(sizes2 & 15) } else { 0 };
    let read_n = |at: &mut usize, n: usize| -> Option<u64> {
        let mut v = 0u64;
        for i in 0..n {
            v = (v << 8) | u64::from(*iloc.get(*at + i)?);
        }
        *at += n;
        Some(v)
    };
    let mut at = 6;
    let count = if version < 2 { read_n(&mut at, 2)? } else { read_n(&mut at, 4)? };
    for _ in 0..count.min(65_536) {
        let id = if version < 2 { read_n(&mut at, 2)? } else { read_n(&mut at, 4)? } as u32;
        let method = if version >= 1 { read_n(&mut at, 2)? & 15 } else { 0 };
        let _data_ref = read_n(&mut at, 2)?;
        let base = read_n(&mut at, base_size)?;
        let extents = read_n(&mut at, 2)?;
        let mut out = Vec::new();
        for _ in 0..extents {
            let _index = read_n(&mut at, index_size)?;
            let off = read_n(&mut at, offset_size)?;
            let len = read_n(&mut at, length_size)?;
            if id != want {
                continue;
            }
            let src = if method == 1 { idat } else { file };
            let start = usize::try_from(base + off).ok()?;
            let end = if len == 0 { src.len() } else { start.checked_add(usize::try_from(len).ok()?)? };
            out.extend_from_slice(src.get(start..end)?);
        }
        if id == want {
            return Some(out);
        }
    }
    None
}
