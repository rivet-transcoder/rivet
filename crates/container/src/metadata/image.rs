//! Metadata in still-image files: JPEG `APPn` / `COM` segments, PNG text and
//! `eXIf` chunks, WebP `EXIF` / `XMP ` chunks. ICC profiles, orientation and
//! the rest of how to show the picture are not metadata here.

use super::{Category, Metadata, be32, exif, le32, xmp};

const XMP_SIG: &[u8] = b"http://ns.adobe.com/xap/1.0/\0";
const XMP_EXT_SIG: &[u8] = b"http://ns.adobe.com/xmp/extension/\0";

pub(crate) fn read_jpeg(data: &[u8], m: &mut Metadata) {
    let mut at = 2;
    while at + 4 <= data.len() {
        if data[at] != 0xFF {
            return;
        }
        let marker = data[at + 1];
        // Fill bytes, and the markers with no length.
        if marker == 0xFF {
            at += 1;
            continue;
        }
        if marker == 0x01 || (0xD0..=0xD7).contains(&marker) {
            at += 2;
            continue;
        }
        // Start of scan or end of image: no more metadata segments before the
        // pixels.
        if marker == 0xDA || marker == 0xD9 {
            return;
        }
        let len = usize::from(u16::from_be_bytes([data[at + 2], data[at + 3]]));
        let Some(body) = data.get(at + 4..at + 2 + len).filter(|_| len >= 2) else { return };
        match marker {
            0xE1 if body.starts_with(b"Exif\0\0") => exif::read_exif(body, m),
            0xE1 if body.starts_with(XMP_SIG) => xmp::read(&body[XMP_SIG.len()..], m),
            0xE1 if body.starts_with(XMP_EXT_SIG) => xmp::read(body, m),
            // ICC profiles (APP2 `ICC_PROFILE`), MPF, JFIF, Adobe: not metadata.
            0xE0 | 0xE2 | 0xEE => {}
            // Photoshop IRB: IPTC captions, keywords, bylines and places.
            0xED if body.starts_with(b"Photoshop 3.0\0") => {
                m.present.insert(Category::Descriptive);
                m.unclassified(String::from("jpeg/APP13 Photoshop IRB"));
            }
            0xFE => m.set_descriptive("comment", &super::text(body)),
            0xE1 | 0xE3..=0xEF => {
                let sig: String = body.iter().take_while(|&&b| b != 0 && b.is_ascii_graphic()).take(24).map(|&b| b as char).collect();
                m.unclassified(format!("jpeg/APP{} {sig}", marker - 0xE0));
            }
            _ => {}
        }
        at += 2 + len;
    }
}

pub(crate) fn read_png(data: &[u8], m: &mut Metadata) {
    let mut at = 8;
    while at + 8 <= data.len() {
        let Some(len) = be32(data, at).map(|l| l as usize) else { return };
        let kind = &data[at + 4..at + 8];
        let Some(body) = data.get(at + 8..at + 8 + len) else { return };
        match kind {
            b"eXIf" => exif::read_exif(body, m),
            b"tEXt" => {
                if let Some(nul) = body.iter().position(|&b| b == 0) {
                    png_text(m, &super::text(&body[..nul]), &latin1(&body[nul + 1..]));
                }
            }
            b"iTXt" => {
                // keyword \0 compressed method \0 language \0 translated \0 text
                let mut parts = body.splitn(2, |&b| b == 0);
                let keyword = super::text(parts.next().unwrap_or_default());
                let rest = parts.next().unwrap_or_default();
                let compressed = rest.first() == Some(&1);
                let text = rest.get(2..).map(|r| {
                    let mut it = r.splitn(3, |&b| b == 0);
                    let _lang = it.next();
                    let _translated = it.next();
                    it.next().unwrap_or_default()
                });
                match (keyword.as_str(), compressed, text) {
                    ("XML:com.adobe.xmp", false, Some(t)) => xmp::read(t, m),
                    (_, false, Some(t)) => png_text(m, &keyword, &String::from_utf8_lossy(t)),
                    _ => png_text_presence(m, &keyword),
                }
            }
            // Compressed text: the keyword says what it is.
            b"zTXt" => {
                let keyword = super::text(body.split(|&b| b == 0).next().unwrap_or_default());
                png_text_presence(m, &keyword);
            }
            // The last-modified time.
            b"tIME" => m.present.insert(Category::CaptureTime),
            b"IDAT" | b"IEND" => {}
            _ => {}
        }
        at += 12 + len;
    }
}

fn latin1(b: &[u8]) -> String {
    b.iter().map(|&c| c as char).collect()
}

fn png_text(m: &mut Metadata, keyword: &str, value: &str) {
    match keyword {
        "Creation Time" => m.set_capture_time(value),
        "Raw profile type exif" | "Raw profile type APP1" => {
            m.present.insert(Category::Device);
            m.unclassified(format!("png/text {keyword}"));
        }
        "Raw profile type xmp" | "Raw profile type iptc" | "Raw profile type 8bim" => {
            m.present.insert(Category::Descriptive);
            m.unclassified(format!("png/text {keyword}"));
        }
        _ => {
            if !m.set_by_name(keyword, value) {
                m.set_descriptive(&keyword.to_ascii_lowercase().replace(' ', "_"), value);
            }
        }
    }
}

/// A text chunk whose value this reader does not decompress: its keyword
/// alone says which category it is.
fn png_text_presence(m: &mut Metadata, keyword: &str) {
    let category = match keyword.to_ascii_lowercase().replace(' ', "_").as_str() {
        "creation_time" => Category::CaptureTime,
        "software" | "source" | "raw_profile_type_exif" | "raw_profile_type_app1" => Category::Device,
        "xml:com.adobe.xmp" => {
            m.unclassified(String::from("png/compressed XMP"));
            Category::Descriptive
        }
        _ => Category::Descriptive,
    };
    m.present.insert(category);
}

pub(crate) fn read_webp(data: &[u8], m: &mut Metadata) {
    let mut at = 12;
    while at + 8 <= data.len() {
        let Some(len) = le32(data, at + 4).map(|l| l as usize) else { return };
        let Some(body) = data.get(at + 8..at + 8 + len) else { return };
        match &data[at..at + 4] {
            b"EXIF" => exif::read_exif(body, m),
            b"XMP " => xmp::read(body, m),
            _ => {}
        }
        at += 8 + len + (len & 1);
    }
}
