//! Identifying metadata in a media file: where it was made (location), what
//! made it (device), when (capture time), and what it is called (descriptive
//! tags).
//!
//! [`read`] finds it in a source or an output: QuickTime / MP4 `udta`, `meta`
//! (`mdta` keys and iTunes `ilst`), `mvhd` / `tkhd` / `mdhd` times and timed
//! metadata tracks (`mebx`, `gpmd`, `camm`, `tmcd`, `rtmd`); Matroska `Info`
//! and `Tags`; EXIF and XMP in JPEG, PNG, WebP, TIFF and HEIF / AVIF items;
//! FLAC Vorbis comments, ID3 and RIFF `INFO`. It never fails: what it cannot
//! parse it skips, and an item it finds but cannot place in a category is
//! listed in [`Metadata::unclassified`] rather than dropped, so a check that
//! an output is clean can refuse what it does not understand.
//!
//! [`write`] carries a kept subset into an output: MP4 `meta` keys and
//! `udta`, FLAC Vorbis comments, an ID3v2 tag, or an EXIF block for a still.
//!
//! Orientation, colour and the codec's own headers are not metadata here:
//! they describe how to show the picture, not who took it or where.

use std::collections::BTreeMap;
use std::fmt;

mod audio;
pub mod exif;
mod image;
mod isobmff;
pub mod iso6709;
mod matroska;
pub mod write;
mod xmp;

#[cfg(test)]
mod tests;

/// One kind of identifying metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Category {
    /// GPS coordinates and place names.
    Location,
    /// Make, model, lens, serial numbers, software and owner name.
    Device,
    /// When it was recorded or made.
    CaptureTime,
    /// Title, artist, copyright, comment, description, keywords, cover art.
    Descriptive,
}

impl Category {
    pub const ALL: [Category; 4] = [Category::Location, Category::Device, Category::CaptureTime, Category::Descriptive];

    /// The name the settings and the reports use.
    pub fn name(self) -> &'static str {
        match self {
            Category::Location => "location",
            Category::Device => "device",
            Category::CaptureTime => "capture_time",
            Category::Descriptive => "descriptive",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        Category::ALL.into_iter().find(|c| c.name() == s || c.name().replace('_', "-") == s)
    }

    fn bit(self) -> u8 {
        1 << self as u8
    }
}

/// A set of [`Category`]s.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Categories(u8);

impl Categories {
    pub const NONE: Categories = Categories(0);
    pub const ALL: Categories = Categories(0b1111);

    pub fn contains(self, c: Category) -> bool {
        self.0 & c.bit() != 0
    }
    pub fn insert(&mut self, c: Category) {
        self.0 |= c.bit();
    }
    pub fn with(mut self, c: Category) -> Self {
        self.insert(c);
        self
    }
    pub fn is_empty(self) -> bool {
        self.0 == 0
    }
    pub fn union(self, other: Categories) -> Categories {
        Categories(self.0 | other.0)
    }
    /// What is in `self` and not in `other`.
    pub fn minus(self, other: Categories) -> Categories {
        Categories(self.0 & !other.0)
    }
    pub fn iter(self) -> impl Iterator<Item = Category> {
        Category::ALL.into_iter().filter(move |c| self.contains(*c))
    }
    pub fn names(self) -> Vec<&'static str> {
        self.iter().map(Category::name).collect()
    }

    /// A comma-separated list of category names; `none` or empty is the
    /// empty set, `all` every category.
    pub fn parse_list(s: &str) -> Result<Categories, String> {
        let mut set = Categories::NONE;
        for word in s.split(',').map(str::trim).filter(|w| !w.is_empty()) {
            match word {
                "none" => {}
                "all" => set = Categories::ALL,
                _ => set.insert(Category::parse(word).ok_or_else(|| {
                    format!("unknown metadata category {word:?}: expected location, device, capture_time or descriptive")
                })?),
            }
        }
        Ok(set)
    }
}

impl fmt::Display for Categories {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_empty() {
            return f.write_str("none");
        }
        f.write_str(&self.names().join(","))
    }
}

impl FromIterator<Category> for Categories {
    fn from_iter<I: IntoIterator<Item = Category>>(iter: I) -> Self {
        let mut set = Categories::NONE;
        for c in iter {
            set.insert(c);
        }
        set
    }
}

/// Where a file was made: coordinates, a place name, or both.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Location {
    pub latitude: Option<f64>,
    pub longitude: Option<f64>,
    pub altitude: Option<f64>,
    /// A place name or a location written as text the reader could not turn
    /// into coordinates.
    pub name: Option<String>,
}

impl Location {
    pub fn coordinates(latitude: f64, longitude: f64, altitude: Option<f64>) -> Self {
        Location { latitude: Some(latitude), longitude: Some(longitude), altitude, name: None }
    }

    fn named(name: &str) -> Self {
        Location { name: Some(name.to_string()), ..Default::default() }
    }

    pub fn has_coordinates(&self) -> bool {
        self.latitude.is_some() && self.longitude.is_some()
    }
}

/// What made a file.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Device {
    pub make: Option<String>,
    pub model: Option<String>,
    pub software: Option<String>,
    pub lens: Option<String>,
    /// Camera body, lens or other serial numbers and unique identifiers.
    pub serial: Option<String>,
    /// The owner name a camera or account records.
    pub owner: Option<String>,
}

impl Device {
    pub fn is_empty(&self) -> bool {
        self.make.is_none()
            && self.model.is_none()
            && self.software.is_none()
            && self.lens.is_none()
            && self.serial.is_none()
            && self.owner.is_none()
    }
}

/// What a timed metadata track records.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimedTrackKind {
    /// Positions over time.
    Location,
    /// Motion, orientation or other sensor readings, which may include GPS.
    Telemetry,
    /// Time of day.
    Timecode,
    /// Any other timed metadata.
    Other,
}

impl TimedTrackKind {
    pub fn name(self) -> &'static str {
        match self {
            TimedTrackKind::Location => "location",
            TimedTrackKind::Telemetry => "telemetry",
            TimedTrackKind::Timecode => "timecode",
            TimedTrackKind::Other => "other",
        }
    }
}

/// A timed metadata track in a source. Never carried to an output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TimedTrack {
    pub kind: TimedTrackKind,
    /// What it is, for a person: "Apple location track", "GoPro telemetry
    /// (GPMF)".
    pub label: String,
    /// The categories it can hold.
    pub categories: Categories,
}

/// The identifying metadata [`read`] found in a file.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Metadata {
    pub location: Option<Location>,
    pub device: Device,
    /// As RFC 3339 where the source's form allows (`2024-05-01T12:34:56+02:00`,
    /// or with no offset when the source records none), else as written.
    pub capture_time: Option<String>,
    /// Lower-case names (`title`, `artist`, `copyright`, `comment`, …) to values.
    /// Cover art is `picture` with its size.
    pub descriptive: BTreeMap<String, String>,
    pub timed_tracks: Vec<TimedTrack>,
    /// Categories seen in a form the reader records no value for (a GPS block
    /// with no fix, a camera maker note, a date in a compressed text chunk).
    pub present: Categories,
    /// Metadata items found that fit no category, by where they were
    /// (`udta/XYZ `, `mdta/com.example.key`, `png/zTXt`). A source's are
    /// informative; an output's mean it cannot be shown clean.
    pub unclassified: Vec<String>,
}

impl Metadata {
    /// Every category this file carries, from values, timed tracks and
    /// presence alone.
    pub fn categories(&self) -> Categories {
        let mut set = self.present;
        if self.location.is_some() {
            set.insert(Category::Location);
        }
        if !self.device.is_empty() {
            set.insert(Category::Device);
        }
        if self.capture_time.is_some() {
            set.insert(Category::CaptureTime);
        }
        if !self.descriptive.is_empty() {
            set.insert(Category::Descriptive);
        }
        for t in &self.timed_tracks {
            set = set.union(t.categories);
        }
        set
    }

    pub fn is_empty(&self) -> bool {
        self.categories().is_empty() && self.unclassified.is_empty()
    }

    /// The values in `keep`, for writing into an output: no timed tracks, no
    /// presence-only categories, nothing unclassified.
    pub fn kept(&self, keep: Categories) -> Metadata {
        Metadata {
            location: self.location.clone().filter(|_| keep.contains(Category::Location)),
            device: if keep.contains(Category::Device) { self.device.clone() } else { Device::default() },
            capture_time: self.capture_time.clone().filter(|_| keep.contains(Category::CaptureTime)),
            descriptive: if keep.contains(Category::Descriptive) {
                self.descriptive.iter().filter(|(k, _)| k.as_str() != "picture").map(|(k, v)| (k.clone(), v.clone())).collect()
            } else {
                BTreeMap::new()
            },
            ..Default::default()
        }
    }

    fn set_location(&mut self, location: Location) {
        match &mut self.location {
            Some(have) if have.has_coordinates() || !location.has_coordinates() => {
                if have.name.is_none() {
                    have.name = location.name;
                }
            }
            _ => self.location = Some(location),
        }
    }

    /// A location written as text: ISO 6709 when it parses, else a name.
    fn set_location_text(&mut self, text: &str) {
        let text = text.trim();
        if text.is_empty() {
            return;
        }
        match iso6709::parse(text) {
            Some(loc) => self.set_location(loc),
            None => self.set_location(Location::named(text)),
        }
    }

    fn set_capture_time(&mut self, raw: &str) {
        let raw = raw.trim().trim_end_matches('\0');
        if raw.is_empty() || self.capture_time.is_some() {
            return;
        }
        self.capture_time = Some(normalize_date(raw));
    }

    fn set_descriptive(&mut self, key: &str, value: &str) {
        let value = value.trim().trim_end_matches('\0');
        if value.is_empty() {
            return;
        }
        self.descriptive.entry(key.to_ascii_lowercase()).or_insert_with(|| value.to_string());
    }

    fn set_device(&mut self, field: DeviceField, value: &str) {
        let value = value.trim().trim_end_matches('\0').trim();
        if value.is_empty() {
            return;
        }
        let slot = match field {
            DeviceField::Make => &mut self.device.make,
            DeviceField::Model => &mut self.device.model,
            DeviceField::Software => &mut self.device.software,
            DeviceField::Lens => &mut self.device.lens,
            DeviceField::Serial => &mut self.device.serial,
            DeviceField::Owner => &mut self.device.owner,
        };
        if slot.is_none() {
            *slot = Some(value.to_string());
        }
    }

    fn unclassified(&mut self, what: String) {
        if !self.unclassified.contains(&what) {
            self.unclassified.push(what);
        }
    }

    /// A text item by a common tag name (the vocabulary QuickTime keys,
    /// Matroska tags, Vorbis comments and PNG text share, give or take case).
    /// Returns false when the name means nothing here.
    fn set_by_name(&mut self, name: &str, value: &str) -> bool {
        let lower = name.to_ascii_lowercase();
        let key = lower.replace([' ', '-'], "_");
        match key.as_str() {
            "location" | "recording_location" | "location_iso6709" | "iso6709" | "gps" | "xyz" => {
                self.set_location_text(value)
            }
            "make" | "manufacturer" => self.set_device(DeviceField::Make, value),
            "model" | "camera_model" => self.set_device(DeviceField::Model, value),
            "software" | "encoder" | "encoding_tool" | "encoded_with" | "creator_tool" | "writing_app"
            | "muxing_app" | "firmware" | "android_version" | "version" | "source" | "host_computer" => {
                self.set_device(DeviceField::Software, value)
            }
            "lens" | "lens_model" | "lens_make" => self.set_device(DeviceField::Lens, value),
            "serial" | "serial_number" | "body_serial_number" | "camera_serial_number" | "lens_serial_number"
            | "identifier" | "device_identifier" | "unique_id" => self.set_device(DeviceField::Serial, value),
            "owner" | "owner_name" | "camera_owner_name" => self.set_device(DeviceField::Owner, value),
            "date" | "creation_time" | "creationdate" | "creation_date" | "date_recorded" | "date_released"
            | "date_encoded" | "date_tagged" | "date_time_original" | "datetime" | "year" | "recorded_date"
            | "date_time" | "time" => self.set_capture_time(value),
            "title" | "name" | "displayname" | "display_name" | "artist" | "author" | "album" | "album_artist"
            | "albumartist" | "performer" | "composer" | "copyright" | "comment" | "comments" | "description"
            | "keywords" | "genre" | "information" | "director" | "producer" | "publisher" | "license"
            | "organization" | "contact" | "subject" | "summary" | "synopsis" | "disclaimer" | "warning"
            | "collection_name" | "encoded_by" | "lyrics" | "grouping" | "tracknumber" | "track" | "discnumber"
            | "rating" | "credits" | "artist_url" | "url" | "isrc" | "label" => self.set_descriptive(&key, value),
            _ => return false,
        }
        true
    }
}

#[derive(Clone, Copy)]
enum DeviceField {
    Make,
    Model,
    Software,
    Lens,
    Serial,
    Owner,
}

/// The identifying metadata in `data`, whatever the container.
pub fn read(data: &[u8]) -> Metadata {
    let mut m = Metadata::default();
    if data.len() < 12 {
        return m;
    }
    if data.starts_with(&[0xFF, 0xD8]) {
        image::read_jpeg(data, &mut m);
    } else if data.starts_with(b"\x89PNG\r\n\x1a\n") {
        image::read_png(data, &mut m);
    } else if &data[..4] == b"RIFF" && &data[8..12] == b"WEBP" {
        image::read_webp(data, &mut m);
    } else if &data[..4] == b"RIFF" && &data[8..12] == b"AVI " {
        audio::read_riff_info(data, &mut m);
    } else if data.starts_with(b"II*\0") || data.starts_with(b"MM\0*") {
        exif::read_tiff(data, &mut m);
    } else if data.starts_with(&[0x1A, 0x45, 0xDF, 0xA3]) {
        matroska::read(data, &mut m);
    } else if isobmff::looks_like(data) {
        isobmff::read(data, &mut m);
    } else if data.starts_with(b"fLaC") || data.starts_with(b"ID3") || crate::mp3::sniff(data) {
        audio::read_id3(data, &mut m);
        if let Some(at) = crate::demux::audio::lossless::native_flac_offset(data) {
            audio::read_flac(&data[at..], &mut m);
        }
        audio::read_id3v1(data, &mut m);
    }
    m
}

// ---- shared helpers ------------------------------------------------------------

/// A date as RFC 3339 where it has a recognisable form: EXIF's
/// `YYYY:MM:DD HH:MM:SS`, ISO 8601 with a `-hhmm` offset, or a bare year.
pub(crate) fn normalize_date(raw: &str) -> String {
    let s = raw.trim();
    let b = s.as_bytes();
    let digits = |r: std::ops::Range<usize>| b.get(r.clone()).is_some_and(|d| d.iter().all(u8::is_ascii_digit));
    if b.len() >= 19 && digits(0..4) && digits(5..7) && digits(8..10) && digits(11..13) && digits(14..16) && digits(17..19) {
        let mut out = format!("{}-{}-{}T{}:{}:{}", &s[0..4], &s[5..7], &s[8..10], &s[11..13], &s[14..16], &s[17..19]);
        let mut rest = &s[19..];
        if let Some(frac) = rest.strip_prefix('.') {
            let n = frac.bytes().take_while(u8::is_ascii_digit).count();
            out.push('.');
            out.push_str(&frac[..n]);
            rest = &frac[n..];
        }
        let rest = rest.trim();
        match rest {
            "" => {}
            "Z" | "z" => out.push('Z'),
            _ => {
                let r = rest.as_bytes();
                if matches!(r.first(), Some(b'+' | b'-')) {
                    let hh = rest.get(1..3);
                    let mm = rest.get(3..).map(|m| m.trim_start_matches(':'));
                    if let (Some(hh), Some(mm)) = (hh, mm) {
                        if hh.len() == 2 && mm.len() >= 2 && hh.bytes().chain(mm[..2].bytes()).all(|c| c.is_ascii_digit()) {
                            out.push_str(&format!("{}{hh}:{}", &rest[..1], &mm[..2]));
                        }
                    }
                }
            }
        }
        return out;
    }
    s.to_string()
}

/// Seconds since 1904-01-01 (the QuickTime epoch) as RFC 3339 UTC.
pub(crate) fn quicktime_time(secs: u64) -> String {
    unix_time((secs as i64) - 2_082_844_800)
}

/// Seconds since 1970-01-01 as RFC 3339 UTC.
pub(crate) fn unix_time(secs: i64) -> String {
    let days = secs.div_euclid(86_400);
    let rem = secs.rem_euclid(86_400);
    let (y, m, d) = civil_from_days(days);
    format!("{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z", rem / 3600, rem / 60 % 60, rem % 60)
}

/// RFC 3339 (or the `YYYY-MM-DDTHH:MM:SS` prefix of one) as seconds since
/// 1970, the offset applied when there is one.
pub(crate) fn parse_unix_time(s: &str) -> Option<i64> {
    let b = s.as_bytes();
    if b.len() < 19 {
        return None;
    }
    let n = |r: std::ops::Range<usize>| s.get(r)?.parse::<i64>().ok();
    let days = days_from_civil(n(0..4)?, n(5..7)?, n(8..10)?);
    let mut secs = days * 86_400 + n(11..13)? * 3600 + n(14..16)? * 60 + n(17..19)?;
    let tail = s[19..].trim_start_matches(|c: char| c == '.' || c.is_ascii_digit());
    if tail.len() >= 6 && matches!(tail.as_bytes()[0], b'+' | b'-') {
        let off = tail.get(1..3)?.parse::<i64>().ok()? * 3600 + tail.get(4..6)?.parse::<i64>().ok()? * 60;
        secs -= if tail.starts_with('+') { off } else { -off };
    }
    Some(secs)
}

// Howard Hinnant's civil-date algorithms.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y.rem_euclid(400);
    let mp = if m > 2 { m - 3 } else { m + 9 };
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// Latin-1 or UTF-8 text, whichever it decodes as, with NULs trimmed.
pub(crate) fn text(bytes: &[u8]) -> String {
    let bytes = match bytes.iter().position(|&b| b == 0) {
        Some(end) if bytes[end..].iter().all(|&b| b == 0) => &bytes[..end],
        _ => bytes,
    };
    match std::str::from_utf8(bytes) {
        Ok(s) => s.to_string(),
        Err(_) => bytes.iter().map(|&b| b as char).collect(),
    }
}

pub(crate) fn utf16(bytes: &[u8], big_endian: bool) -> String {
    let units: Vec<u16> = bytes
        .chunks_exact(2)
        .map(|c| if big_endian { u16::from_be_bytes([c[0], c[1]]) } else { u16::from_le_bytes([c[0], c[1]]) })
        .take_while(|&u| u != 0)
        .collect();
    String::from_utf16_lossy(&units)
}

pub(crate) fn be16(b: &[u8], at: usize) -> Option<u16> {
    Some(u16::from_be_bytes(b.get(at..at + 2)?.try_into().ok()?))
}
pub(crate) fn be32(b: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_be_bytes(b.get(at..at + 4)?.try_into().ok()?))
}
pub(crate) fn be64(b: &[u8], at: usize) -> Option<u64> {
    Some(u64::from_be_bytes(b.get(at..at + 8)?.try_into().ok()?))
}
pub(crate) fn le32(b: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_le_bytes(b.get(at..at + 4)?.try_into().ok()?))
}
