//! Matroska / WebM: `Info` (title, date, muxing and writing application) and
//! `Tags` (`SimpleTag` name / value pairs, whose names follow the Matroska
//! tag list, or QuickTime's keys when a remux carried them over).

use super::{Category, Metadata};

const SEGMENT: u32 = 0x1853_8067;
const INFO: u32 = 0x1549_A966;
const TAGS: u32 = 0x1254_C367;
const ATTACHMENTS: u32 = 0x1941_A469;
const CLUSTER: u32 = 0x1F43_B675;
const TITLE: u32 = 0x7BA9;
const MUXING_APP: u32 = 0x4D80;
const WRITING_APP: u32 = 0x5741;
const DATE_UTC: u32 = 0x4461;
const TAG: u32 = 0x7373;
const SIMPLE_TAG: u32 = 0x67C8;
const TAG_NAME: u32 = 0x45A3;
const TAG_STRING: u32 = 0x4487;
const TAG_BINARY: u32 = 0x4485;

/// Seconds from 1970-01-01 to 2001-01-01, Matroska's `DateUTC` epoch.
const MKV_EPOCH: i64 = 978_307_200;

struct Element<'a> {
    id: u32,
    /// `None` for an unknown-size element (a live stream's segment or cluster).
    body: Option<&'a [u8]>,
    /// Where its body starts, in the slice walked.
    body_at: usize,
}

fn vint(data: &[u8], at: usize, keep_marker: bool) -> Option<(u64, usize)> {
    let first = *data.get(at)?;
    let len = first.leading_zeros() as usize + 1;
    if len > 8 {
        return None;
    }
    let mut v = if keep_marker { u64::from(first) } else { u64::from(first) & ((1u64 << (8 - len)) - 1) };
    let mut all_ones = v == (1u64 << (8 - len)) - 1;
    for i in 1..len {
        let b = *data.get(at + i)?;
        all_ones &= b == 0xFF;
        v = (v << 8) | u64::from(b);
    }
    if !keep_marker && all_ones {
        return Some((u64::MAX, len));
    }
    Some((v, len))
}

fn elements(data: &[u8]) -> impl Iterator<Item = Element<'_>> {
    let mut at = 0usize;
    std::iter::from_fn(move || {
        let (id, id_len) = vint(data, at, true)?;
        let (size, size_len) = vint(data, at + id_len, false)?;
        let body_at = at + id_len + size_len;
        if size == u64::MAX {
            at = data.len();
            return Some(Element { id: id as u32, body: None, body_at });
        }
        let end = body_at.checked_add(usize::try_from(size).ok()?)?;
        let body = data.get(body_at..end)?;
        at = end;
        Some(Element { id: id as u32, body: Some(body), body_at })
    })
}

pub(crate) fn read(data: &[u8], m: &mut Metadata) {
    for e in elements(data) {
        if e.id == SEGMENT {
            let segment = e.body.unwrap_or(&data[e.body_at..]);
            read_segment(segment, m);
        }
    }
}

fn read_segment(segment: &[u8], m: &mut Metadata) {
    for e in elements(segment) {
        let Some(body) = e.body else {
            // An unknown-size cluster runs to the end; a live stream carries
            // its tags before it, if at all.
            return;
        };
        match e.id {
            INFO => read_info(body, m),
            TAGS => {
                for tag in elements(body).filter(|t| t.id == TAG) {
                    for st in elements(tag.body.unwrap_or_default()).filter(|s| s.id == SIMPLE_TAG) {
                        read_simple_tag(st.body.unwrap_or_default(), m);
                    }
                }
            }
            ATTACHMENTS => m.set_descriptive("attachments", "present"),
            CLUSTER => {}
            _ => {}
        }
    }
}

fn read_info(info: &[u8], m: &mut Metadata) {
    for e in elements(info) {
        let body = e.body.unwrap_or_default();
        match e.id {
            TITLE => m.set_descriptive("title", &super::text(body)),
            MUXING_APP | WRITING_APP => m.set_device(super::DeviceField::Software, &super::text(body)),
            DATE_UTC if body.len() == 8 => {
                let ns = i64::from_be_bytes(body.try_into().unwrap_or_default());
                m.set_capture_time(&super::unix_time(MKV_EPOCH + ns.div_euclid(1_000_000_000)));
            }
            _ => {}
        }
    }
}

fn read_simple_tag(st: &[u8], m: &mut Metadata) {
    let mut name = String::new();
    let mut value = String::new();
    let mut binary = false;
    for e in elements(st) {
        let body = e.body.unwrap_or_default();
        match e.id {
            TAG_NAME => name = super::text(body),
            TAG_STRING => value = super::text(body),
            TAG_BINARY => binary = !body.is_empty(),
            SIMPLE_TAG => read_simple_tag(body, m),
            _ => {}
        }
    }
    let upper = name.to_ascii_uppercase();
    match upper.as_str() {
        "" => {}
        // How the streams are coded and muxed, not who made them.
        "DURATION" | "BPS" | "NUMBER_OF_FRAMES" | "NUMBER_OF_BYTES" | "LANGUAGE" | "MAJOR_BRAND" | "MINOR_VERSION"
        | "COMPATIBLE_BRANDS" | "HANDLER_NAME" | "VENDOR_ID" | "ENCODER_OPTIONS" => {}
        s if s.starts_with("_STATISTICS") => {}
        _ if name.starts_with("com.apple.") || name.starts_with("com.android.") => {
            super::isobmff::mdta_key(m, &name, &value, "mkv/Tags");
        }
        _ => {
            if binary && value.is_empty() {
                m.present.insert(Category::Descriptive);
            } else if !m.set_by_name(&name, &value) {
                m.unclassified(format!("mkv/Tags {name}"));
            }
        }
    }
}
