//! EXIF: a TIFF structure (header, IFD0, the Exif and GPS sub-IFDs) — read
//! for what it says about location, device, time and description, and
//! written fresh with only what is kept.

use super::{Category, DeviceField, Location, Metadata};

const TAG_IMAGE_DESCRIPTION: u16 = 0x010E;
const TAG_MAKE: u16 = 0x010F;
const TAG_MODEL: u16 = 0x0110;
const TAG_ORIENTATION: u16 = 0x0112;
const TAG_SOFTWARE: u16 = 0x0131;
const TAG_DATE_TIME: u16 = 0x0132;
const TAG_ARTIST: u16 = 0x013B;
const TAG_HOST_COMPUTER: u16 = 0x013C;
const TAG_COPYRIGHT: u16 = 0x8298;
const TAG_EXIF_IFD: u16 = 0x8769;
const TAG_GPS_IFD: u16 = 0x8825;
const TAG_XMP: u16 = 0x02BC;
const TAG_IPTC: u16 = 0x83BB;
const TAG_XP_TITLE: u16 = 0x9C9B;
const TAG_XP_COMMENT: u16 = 0x9C9C;
const TAG_XP_AUTHOR: u16 = 0x9C9D;
const TAG_XP_KEYWORDS: u16 = 0x9C9E;
const TAG_XP_SUBJECT: u16 = 0x9C9F;

const TAG_DATE_TIME_ORIGINAL: u16 = 0x9003;
const TAG_DATE_TIME_DIGITIZED: u16 = 0x9004;
const TAG_OFFSET_TIME: u16 = 0x9010;
const TAG_OFFSET_TIME_ORIGINAL: u16 = 0x9011;
const TAG_MAKER_NOTE: u16 = 0x927C;
const TAG_USER_COMMENT: u16 = 0x9286;
const TAG_IMAGE_UNIQUE_ID: u16 = 0xA420;
const TAG_CAMERA_OWNER_NAME: u16 = 0xA430;
const TAG_BODY_SERIAL_NUMBER: u16 = 0xA431;
const TAG_LENS_MAKE: u16 = 0xA433;
const TAG_LENS_MODEL: u16 = 0xA434;
const TAG_LENS_SERIAL_NUMBER: u16 = 0xA435;

const GPS_LATITUDE_REF: u16 = 1;
const GPS_LATITUDE: u16 = 2;
const GPS_LONGITUDE_REF: u16 = 3;
const GPS_LONGITUDE: u16 = 4;
const GPS_ALTITUDE_REF: u16 = 5;
const GPS_ALTITUDE: u16 = 6;

const TYPE_BYTE: u16 = 1;
const TYPE_ASCII: u16 = 2;
const TYPE_SHORT: u16 = 3;
const TYPE_LONG: u16 = 4;
const TYPE_RATIONAL: u16 = 5;

struct Tiff<'a> {
    data: &'a [u8],
    le: bool,
}

struct Entry<'a> {
    tag: u16,
    kind: u16,
    count: u32,
    value: &'a [u8],
}

impl<'a> Tiff<'a> {
    fn new(data: &'a [u8]) -> Option<Self> {
        let le = match data.get(..4)? {
            b"II*\0" => true,
            b"MM\0*" => false,
            _ => return None,
        };
        Some(Tiff { data, le })
    }
    fn u16(&self, at: usize) -> Option<u16> {
        let b: [u8; 2] = self.data.get(at..at + 2)?.try_into().ok()?;
        Some(if self.le { u16::from_le_bytes(b) } else { u16::from_be_bytes(b) })
    }
    fn u32(&self, at: usize) -> Option<u32> {
        let b: [u8; 4] = self.data.get(at..at + 4)?.try_into().ok()?;
        Some(if self.le { u32::from_le_bytes(b) } else { u32::from_be_bytes(b) })
    }

    fn entries(&self, ifd: usize) -> Vec<Entry<'a>> {
        let Some(n) = self.u16(ifd) else { return Vec::new() };
        let mut out = Vec::new();
        for i in 0..usize::from(n).min(1024) {
            let at = ifd + 2 + i * 12;
            let (Some(tag), Some(kind), Some(count)) = (self.u16(at), self.u16(at + 2), self.u32(at + 4)) else {
                break;
            };
            let unit = match kind {
                1 | 2 | 6 | 7 => 1,
                3 | 8 => 2,
                4 | 9 | 11 | 13 => 4,
                5 | 10 | 12 => 8,
                _ => continue,
            };
            let Some(len) = (count as usize).checked_mul(unit) else { continue };
            let value = if len <= 4 {
                self.data.get(at + 8..at + 8 + len)
            } else {
                self.u32(at + 8).and_then(|off| self.data.get(off as usize..(off as usize).checked_add(len)?))
            };
            if let Some(value) = value {
                out.push(Entry { tag, kind, count, value });
            }
        }
        out
    }

    fn rational(&self, v: &[u8], i: usize) -> Option<f64> {
        let t = Tiff { data: v, le: self.le };
        let (n, d) = (t.u32(i * 8)?, t.u32(i * 8 + 4)?);
        (d != 0).then(|| f64::from(n) / f64::from(d))
    }

    fn offset(&self, e: &Entry) -> Option<usize> {
        let t = Tiff { data: e.value, le: self.le };
        match e.kind {
            TYPE_LONG | 13 => t.u32(0).map(|v| v as usize),
            TYPE_SHORT => t.u16(0).map(usize::from),
            _ => None,
        }
    }
}

fn ascii(v: &[u8]) -> String {
    super::text(v)
}

/// EXIF `UserComment`: an 8-byte character code, then the text.
fn user_comment(v: &[u8]) -> String {
    let (code, body) = v.split_at(v.len().min(8));
    match code {
        b"UNICODE\0" => super::utf16(body, true),
        _ => super::text(body).trim().to_string(),
    }
}

/// An EXIF block: a TIFF structure, or the `Exif\0\0` APP1 form of one.
pub(crate) fn read_exif(data: &[u8], m: &mut Metadata) {
    let data = data.strip_prefix(b"Exif\0\0").unwrap_or(data);
    read_tiff(data, m);
}

/// A TIFF file or an EXIF TIFF structure.
pub(crate) fn read_tiff(data: &[u8], m: &mut Metadata) {
    let Some(t) = Tiff::new(data) else { return };
    let Some(ifd0) = t.u32(4) else { return };
    let mut offset_original = None;
    let mut date_original = None;
    let mut date_other = None;
    for e in t.entries(ifd0 as usize) {
        match e.tag {
            TAG_MAKE => m.set_device(DeviceField::Make, &ascii(e.value)),
            TAG_MODEL => m.set_device(DeviceField::Model, &ascii(e.value)),
            TAG_SOFTWARE => m.set_device(DeviceField::Software, &ascii(e.value)),
            TAG_HOST_COMPUTER => m.set_device(DeviceField::Software, &ascii(e.value)),
            TAG_DATE_TIME => date_other = Some(ascii(e.value)),
            TAG_ARTIST => m.set_descriptive("artist", &ascii(e.value)),
            TAG_COPYRIGHT => m.set_descriptive("copyright", &ascii(e.value)),
            TAG_IMAGE_DESCRIPTION => m.set_descriptive("description", &ascii(e.value)),
            TAG_XP_TITLE => m.set_descriptive("title", &super::utf16(e.value, false)),
            TAG_XP_COMMENT => m.set_descriptive("comment", &super::utf16(e.value, false)),
            TAG_XP_AUTHOR => m.set_descriptive("artist", &super::utf16(e.value, false)),
            TAG_XP_KEYWORDS => m.set_descriptive("keywords", &super::utf16(e.value, false)),
            TAG_XP_SUBJECT => m.set_descriptive("subject", &super::utf16(e.value, false)),
            TAG_XMP => super::xmp::read(e.value, m),
            TAG_IPTC => m.present.insert(Category::Descriptive),
            TAG_EXIF_IFD => {
                let Some(at) = t.offset(&e) else { continue };
                for x in t.entries(at) {
                    match x.tag {
                        TAG_DATE_TIME_ORIGINAL => date_original = Some(ascii(x.value)),
                        TAG_DATE_TIME_DIGITIZED => {
                            date_other.get_or_insert_with(|| ascii(x.value));
                        }
                        TAG_OFFSET_TIME_ORIGINAL => offset_original = Some(ascii(x.value)),
                        TAG_OFFSET_TIME => {
                            offset_original.get_or_insert_with(|| ascii(x.value));
                        }
                        TAG_CAMERA_OWNER_NAME => m.set_device(DeviceField::Owner, &ascii(x.value)),
                        TAG_BODY_SERIAL_NUMBER | TAG_LENS_SERIAL_NUMBER => {
                            m.set_device(DeviceField::Serial, &ascii(x.value))
                        }
                        TAG_IMAGE_UNIQUE_ID => m.set_device(DeviceField::Serial, &ascii(x.value)),
                        TAG_LENS_MODEL => m.set_device(DeviceField::Lens, &ascii(x.value)),
                        TAG_LENS_MAKE => {
                            if m.device.lens.is_none() {
                                m.present.insert(Category::Device);
                            }
                        }
                        // Maker notes hold serial numbers, firmware and more in
                        // each maker's own format.
                        TAG_MAKER_NOTE if !x.value.is_empty() => m.present.insert(Category::Device),
                        TAG_USER_COMMENT => m.set_descriptive("comment", &user_comment(x.value)),
                        _ => {}
                    }
                }
            }
            TAG_GPS_IFD => {
                let Some(at) = t.offset(&e) else { continue };
                read_gps(&t, at, m);
            }
            _ => {}
        }
    }
    if let Some(date) = date_original.or(date_other) {
        let date = date.trim().to_string();
        if !date.is_empty() && !date.starts_with("0000") && !date.chars().all(|c| c == ' ' || c == ':') {
            m.set_capture_time(&format!("{date}{}", offset_original.unwrap_or_default().trim()));
        }
    }
}

fn read_gps(t: &Tiff, at: usize, m: &mut Metadata) {
    let entries = t.entries(at);
    let get = |tag| entries.iter().find(|e| e.tag == tag);
    let dms = |e: &Entry| -> Option<f64> {
        (e.kind == TYPE_RATIONAL && e.count >= 3)
            .then(|| Some(t.rational(e.value, 0)? + t.rational(e.value, 1)? / 60.0 + t.rational(e.value, 2)? / 3600.0))
            .flatten()
    };
    let sign = |e: Option<&&Entry>, neg: u8| if e.is_some_and(|e| e.value.first() == Some(&neg)) { -1.0 } else { 1.0 };
    let lat = get(GPS_LATITUDE).and_then(dms);
    let lon = get(GPS_LONGITUDE).and_then(dms);
    if let (Some(lat), Some(lon)) = (lat, lon) {
        let lat = lat * sign(get(GPS_LATITUDE_REF).as_ref(), b'S');
        let lon = lon * sign(get(GPS_LONGITUDE_REF).as_ref(), b'W');
        let alt = get(GPS_ALTITUDE).and_then(|e| (e.kind == TYPE_RATIONAL).then(|| t.rational(e.value, 0)).flatten()).map(|a| {
            if get(GPS_ALTITUDE_REF).is_some_and(|e| e.value.first() == Some(&1)) { -a } else { a }
        });
        m.set_location(Location::coordinates(lat, lon, alt));
    }
    // Any GPS field beyond the version tag says something about where.
    if entries.iter().any(|e| e.tag != 0) {
        m.present.insert(Category::Location);
    }
}

// ---- writing -------------------------------------------------------------------

/// A value to write: a tag and its bytes, as TIFF type and count.
struct Field {
    tag: u16,
    kind: u16,
    count: u32,
    bytes: Vec<u8>,
}

fn ascii_field(tag: u16, s: &str) -> Field {
    let mut bytes: Vec<u8> = s.bytes().filter(|b| b.is_ascii() && *b != 0).collect();
    bytes.push(0);
    Field { tag, kind: TYPE_ASCII, count: bytes.len() as u32, bytes }
}

fn rationals_field(tag: u16, values: &[(u32, u32)]) -> Field {
    let bytes = values.iter().flat_map(|(n, d)| n.to_be_bytes().into_iter().chain(d.to_be_bytes())).collect();
    Field { tag, kind: TYPE_RATIONAL, count: values.len() as u32, bytes }
}

/// Serialise IFDs big-endian. `ifds[0]` is IFD0; a sub-IFD is linked from a
/// field whose tag is in `links` (tag → index into `ifds`).
fn serialise(mut ifds: Vec<Vec<Field>>, links: &[(u16, usize)]) -> Vec<u8> {
    for ifd in &mut ifds {
        ifd.sort_by_key(|f| f.tag);
    }
    // Lay the IFDs out one after the other, each followed by its own long
    // values, to know where each starts.
    let size = |ifd: &Vec<Field>| {
        2 + ifd.len() * 12 + 4 + ifd.iter().map(|f| if f.bytes.len() > 4 { (f.bytes.len() + 1) & !1 } else { 0 }).sum::<usize>()
    };
    let mut starts = Vec::with_capacity(ifds.len());
    let mut at = 8;
    for ifd in &ifds {
        starts.push(at);
        at += size(ifd);
    }
    let mut out = b"MM\0*".to_vec();
    out.extend_from_slice(&8u32.to_be_bytes());
    for (i, ifd) in ifds.iter().enumerate() {
        let mut extra_at = starts[i] + 2 + ifd.len() * 12 + 4;
        let mut extra = Vec::new();
        out.extend_from_slice(&(ifd.len() as u16).to_be_bytes());
        for f in ifd {
            out.extend_from_slice(&f.tag.to_be_bytes());
            let linked = links.iter().find(|(tag, _)| *tag == f.tag).map(|&(_, to)| starts[to] as u32);
            if let Some(off) = linked {
                out.extend_from_slice(&TYPE_LONG.to_be_bytes());
                out.extend_from_slice(&1u32.to_be_bytes());
                out.extend_from_slice(&off.to_be_bytes());
                continue;
            }
            out.extend_from_slice(&f.kind.to_be_bytes());
            out.extend_from_slice(&f.count.to_be_bytes());
            if f.bytes.len() <= 4 {
                let mut v = f.bytes.clone();
                v.resize(4, 0);
                out.extend_from_slice(&v);
            } else {
                out.extend_from_slice(&(extra_at as u32).to_be_bytes());
                extra.extend_from_slice(&f.bytes);
                if f.bytes.len() % 2 == 1 {
                    extra.push(0);
                }
                extra_at += (f.bytes.len() + 1) & !1;
            }
        }
        out.extend_from_slice(&0u32.to_be_bytes()); // no next IFD
        out.extend_from_slice(&extra);
    }
    out
}

/// A fresh EXIF TIFF structure holding `m`'s values — nothing else, and an
/// orientation of 1, since a still is written upright. `None` when there is
/// nothing to write.
pub fn build(m: &Metadata) -> Option<Vec<u8>> {
    if m.location.as_ref().is_none_or(|l| !l.has_coordinates())
        && m.device.is_empty()
        && m.capture_time.is_none()
        && m.descriptive.is_empty()
    {
        return None;
    }
    let mut ifd0 = vec![Field { tag: TAG_ORIENTATION, kind: TYPE_SHORT, count: 1, bytes: 1u16.to_be_bytes().to_vec() }];
    let mut exif = Vec::new();
    let mut gps = Vec::new();
    let d = &m.device;
    for (tag, v) in [(TAG_MAKE, &d.make), (TAG_MODEL, &d.model), (TAG_SOFTWARE, &d.software)] {
        if let Some(v) = v {
            ifd0.push(ascii_field(tag, v));
        }
    }
    for (tag, v) in [(TAG_LENS_MODEL, &d.lens), (TAG_BODY_SERIAL_NUMBER, &d.serial), (TAG_CAMERA_OWNER_NAME, &d.owner)] {
        if let Some(v) = v {
            exif.push(ascii_field(tag, v));
        }
    }
    if let Some(t) = &m.capture_time {
        if let Some((date, offset)) = exif_date(t) {
            ifd0.push(ascii_field(TAG_DATE_TIME, &date));
            exif.push(ascii_field(TAG_DATE_TIME_ORIGINAL, &date));
            exif.push(ascii_field(TAG_DATE_TIME_DIGITIZED, &date));
            if let Some(off) = offset {
                exif.push(ascii_field(TAG_OFFSET_TIME_ORIGINAL, &off));
            }
        }
    }
    for (key, tag) in [("artist", TAG_ARTIST), ("copyright", TAG_COPYRIGHT), ("description", TAG_IMAGE_DESCRIPTION)] {
        if let Some(v) = m.descriptive.get(key) {
            ifd0.push(ascii_field(tag, v));
        }
    }
    for (key, tag) in [("title", TAG_XP_TITLE), ("comment", TAG_XP_COMMENT), ("keywords", TAG_XP_KEYWORDS), ("subject", TAG_XP_SUBJECT)] {
        if let Some(v) = m.descriptive.get(key) {
            let mut bytes: Vec<u8> = v.encode_utf16().flat_map(u16::to_le_bytes).collect();
            bytes.extend_from_slice(&[0, 0]);
            ifd0.push(Field { tag, kind: TYPE_BYTE, count: bytes.len() as u32, bytes });
        }
    }
    if let Some(loc) = m.location.as_ref().filter(|l| l.has_coordinates()) {
        let (lat, lon) = (loc.latitude.unwrap_or_default(), loc.longitude.unwrap_or_default());
        gps.push(Field { tag: 0, kind: TYPE_BYTE, count: 4, bytes: vec![2, 3, 0, 0] });
        gps.push(ascii_field(GPS_LATITUDE_REF, if lat < 0.0 { "S" } else { "N" }));
        gps.push(rationals_field(GPS_LATITUDE, &dms(lat.abs())));
        gps.push(ascii_field(GPS_LONGITUDE_REF, if lon < 0.0 { "W" } else { "E" }));
        gps.push(rationals_field(GPS_LONGITUDE, &dms(lon.abs())));
        if let Some(alt) = loc.altitude {
            gps.push(Field { tag: GPS_ALTITUDE_REF, kind: TYPE_BYTE, count: 1, bytes: vec![u8::from(alt < 0.0)] });
            gps.push(rationals_field(GPS_ALTITUDE, &[((alt.abs() * 1000.0).round() as u32, 1000)]));
        }
    }
    let mut ifds = vec![ifd0];
    let mut links = Vec::new();
    for (tag, ifd) in [(TAG_EXIF_IFD, exif), (TAG_GPS_IFD, gps)] {
        if !ifd.is_empty() {
            ifds[0].push(Field { tag, kind: TYPE_LONG, count: 1, bytes: vec![0; 4] });
            links.push((tag, ifds.len()));
            ifds.push(ifd);
        }
    }
    Some(serialise(ifds, &links))
}

/// Degrees as EXIF's three rationals: degrees, minutes, seconds (to 1/10000″).
fn dms(deg: f64) -> [(u32, u32); 3] {
    let d = deg.trunc();
    let min_f = (deg - d) * 60.0;
    let min = min_f.trunc();
    let sec = ((min_f - min) * 60.0 * 10_000.0).round() as u32;
    [(d as u32, 1), (min as u32, 1), (sec, 10_000)]
}

/// RFC 3339 as EXIF's `YYYY:MM:DD HH:MM:SS` and a `±HH:MM` offset.
fn exif_date(t: &str) -> Option<(String, Option<String>)> {
    let b = t.as_bytes();
    if b.len() < 19 || b[4] != b'-' || b[7] != b'-' || !matches!(b[10], b'T' | b' ') {
        return None;
    }
    let date = format!("{}:{}:{} {}", &t[0..4], &t[5..7], &t[8..10], &t[11..19]);
    let tail = t[19..].trim_start_matches(|c: char| c == '.' || c.is_ascii_digit());
    let offset = match tail {
        "Z" | "z" => Some("+00:00".to_string()),
        s if s.len() == 6 && matches!(s.as_bytes()[0], b'+' | b'-') => Some(s.to_string()),
        _ => None,
    };
    Some((date, offset))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_then_read_round_trips() {
        let mut m = Metadata::default();
        m.location = Some(Location::coordinates(37.3349, -122.009, Some(10.0)));
        m.device.make = Some("Apple".into());
        m.device.model = Some("iPhone 15 Pro".into());
        m.device.lens = Some("iPhone 15 Pro back camera 6.765mm f/1.78".into());
        m.device.serial = Some("C39XK0Q1".into());
        m.capture_time = Some("2024-05-01T12:34:56+02:00".into());
        m.descriptive.insert("title".into(), "Beach".into());
        m.descriptive.insert("copyright".into(), "(c) Someone".into());
        let tiff = build(&m).unwrap();
        let back = {
            let mut b = Metadata::default();
            read_tiff(&tiff, &mut b);
            b
        };
        let loc = back.location.clone().unwrap();
        assert!((loc.latitude.unwrap() - 37.3349).abs() < 1e-5);
        assert!((loc.longitude.unwrap() + 122.009).abs() < 1e-5);
        assert_eq!(loc.altitude, Some(10.0));
        assert_eq!(back.device.make.as_deref(), Some("Apple"));
        assert_eq!(back.device.model.as_deref(), Some("iPhone 15 Pro"));
        assert_eq!(back.device.serial.as_deref(), Some("C39XK0Q1"));
        assert_eq!(back.capture_time.as_deref(), Some("2024-05-01T12:34:56+02:00"));
        assert_eq!(back.descriptive.get("title").map(String::as_str), Some("Beach"));
        assert_eq!(back.descriptive.get("copyright").map(String::as_str), Some("(c) Someone"));
    }

    #[test]
    fn nothing_kept_builds_nothing() {
        assert!(build(&Metadata::default()).is_none());
    }

    #[test]
    fn a_device_only_block_has_no_gps() {
        let mut m = Metadata::default();
        m.device.model = Some("Pixel 8".into());
        let tiff = build(&m).unwrap();
        let mut back = Metadata::default();
        read_tiff(&tiff, &mut back);
        assert_eq!(back.categories(), crate::metadata::Categories::NONE.with(Category::Device));
    }

    #[test]
    fn truncated_blocks_do_not_panic() {
        let mut m = Metadata::default();
        m.location = Some(Location::coordinates(1.0, 2.0, None));
        m.device.make = Some("X".into());
        let tiff = build(&m).unwrap();
        for n in 0..tiff.len() {
            let mut b = Metadata::default();
            read_tiff(&tiff[..n], &mut b);
        }
    }
}
