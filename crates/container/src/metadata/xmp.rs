//! XMP packets: RDF/XML, read by property name. A property may be written as
//! an attribute (`exif:GPSLatitude="37,20.094N"`) or an element
//! (`<tiff:Make>Apple</tiff:Make>`, or an `rdf:Alt` / `rdf:Seq` of `rdf:li`).
//! Only the properties that identify are looked for; the rest of the packet
//! (edit history, rating, colour) is not metadata here.

use super::{Category, DeviceField, Location, Metadata};

pub(crate) fn read(packet: &[u8], m: &mut Metadata) {
    let xml = String::from_utf8_lossy(packet);
    let xml = xml.as_ref();

    let lat = prop(xml, "exif:GPSLatitude").and_then(|v| gps_coordinate(&v));
    let lon = prop(xml, "exif:GPSLongitude").and_then(|v| gps_coordinate(&v));
    if let (Some(lat), Some(lon)) = (lat, lon) {
        let alt = prop(xml, "exif:GPSAltitude").and_then(|v| rational(&v)).map(|a| {
            if prop(xml, "exif:GPSAltitudeRef").as_deref() == Some("1") { -a } else { a }
        });
        m.set_location(Location::coordinates(lat, lon, alt));
    }
    if xml.contains("exif:GPS") || xml.contains("exifEX:GPS") {
        m.present.insert(Category::Location);
    }
    for name in ["Iptc4xmpCore:Location", "photoshop:City", "Iptc4xmpExt:LocationShown", "Iptc4xmpExt:LocationCreated"] {
        if let Some(v) = prop(xml, name) {
            m.set_location(Location { name: Some(v), ..Default::default() });
        } else if xml.contains(name) {
            m.present.insert(Category::Location);
        }
    }

    for (name, field) in [
        ("tiff:Make", DeviceField::Make),
        ("tiff:Model", DeviceField::Model),
        ("xmp:CreatorTool", DeviceField::Software),
        ("tiff:Software", DeviceField::Software),
        ("exifEX:LensModel", DeviceField::Lens),
        ("aux:Lens", DeviceField::Lens),
        ("exifEX:BodySerialNumber", DeviceField::Serial),
        ("aux:SerialNumber", DeviceField::Serial),
        ("exifEX:LensSerialNumber", DeviceField::Serial),
        ("exifEX:CameraOwnerName", DeviceField::Owner),
        ("aux:OwnerName", DeviceField::Owner),
    ] {
        if let Some(v) = prop(xml, name) {
            m.set_device(field, &v);
        }
    }

    for name in ["exif:DateTimeOriginal", "photoshop:DateCreated", "xmp:CreateDate", "exif:DateTimeDigitized", "tiff:DateTime"] {
        if let Some(v) = prop(xml, name) {
            m.set_capture_time(&v);
        }
    }
    if xml.contains("xmp:ModifyDate") || xml.contains("xmp:MetadataDate") {
        m.present.insert(Category::CaptureTime);
    }

    for (name, key) in [
        ("dc:title", "title"),
        ("dc:creator", "artist"),
        ("dc:rights", "copyright"),
        ("dc:description", "description"),
        ("dc:subject", "keywords"),
        ("xmpRights:UsageTerms", "license"),
        ("photoshop:Headline", "title"),
        ("photoshop:Credit", "credits"),
        ("Iptc4xmpCore:CreatorContactInfo", "contact"),
    ] {
        if let Some(v) = prop(xml, name) {
            m.set_descriptive(key, &v);
        } else if xml.contains(&format!("<{name}")) {
            m.present.insert(Category::Descriptive);
        }
    }
}

/// A property's value: its attribute, its element's text, or the first
/// `rdf:li` inside its element.
fn prop(xml: &str, name: &str) -> Option<String> {
    let attr = format!("{name}=");
    let mut from = 0;
    while let Some(i) = xml[from..].find(&attr) {
        let at = from + i;
        from = at + attr.len();
        // Must be a whole attribute name.
        if xml[..at].ends_with(|c: char| c.is_whitespace()) {
            let rest = &xml[from..];
            let quote = rest.chars().next()?;
            if quote == '"' || quote == '\'' {
                let body = &rest[1..];
                let end = body.find(quote)?;
                return non_empty(unescape(&body[..end]));
            }
        }
    }
    let open = format!("<{name}");
    let at = xml.find(&open)?;
    let after = &xml[at + open.len()..];
    // `<name/>` or `<name attr="..."/>` has no text.
    let tag_end = after.find('>')?;
    if after[..tag_end].ends_with('/') {
        return None;
    }
    let body = &after[tag_end + 1..];
    let close = format!("</{name}>");
    let body = &body[..body.find(&close)?];
    if let Some(li) = body.find("<rdf:li") {
        let li = &body[li..];
        let start = li.find('>')? + 1;
        let end = li[start..].find("</rdf:li>")? + start;
        return non_empty(unescape(&li[start..end]));
    }
    if body.contains('<') {
        // Structured (a contact-info struct): present, with no one value.
        return Some("(structured)".to_string());
    }
    non_empty(unescape(body))
}

fn non_empty(s: String) -> Option<String> {
    let t = s.trim();
    (!t.is_empty()).then(|| t.to_string())
}

fn unescape(s: &str) -> String {
    s.replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", "\"").replace("&apos;", "'").replace("&amp;", "&")
}

/// XMP's GPS coordinate: `DDD,MM.mmk` or `DDD,MM,SSk` with `k` one of NSEW.
fn gps_coordinate(v: &str) -> Option<f64> {
    let v = v.trim();
    let dir = v.chars().last()?;
    let sign = match dir.to_ascii_uppercase() {
        'N' | 'E' => 1.0,
        'S' | 'W' => -1.0,
        _ => return v.parse().ok(),
    };
    let parts: Vec<f64> = v[..v.len() - 1].split(',').map(|p| p.trim().parse::<f64>()).collect::<Result<_, _>>().ok()?;
    let deg = match parts.as_slice() {
        [d] => *d,
        [d, m] => d + m / 60.0,
        [d, m, s] => d + m / 60.0 + s / 3600.0,
        _ => return None,
    };
    Some(sign * deg)
}

fn rational(v: &str) -> Option<f64> {
    match v.split_once('/') {
        Some((n, d)) => {
            let (n, d) = (n.trim().parse::<f64>().ok()?, d.trim().parse::<f64>().ok()?);
            (d != 0.0).then(|| n / d)
        }
        None => v.trim().parse().ok(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attributes_elements_and_lists() {
        let packet = br#"<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF><rdf:Description rdf:about=""
            tiff:Make="Canon" xmp:CreateDate="2023-07-04T09:30:00-04:00"
            exif:GPSLatitude="40,26.767N" exif:GPSLongitude="79,58.933W">
            <tiff:Model>Canon EOS R5</tiff:Model>
            <dc:title><rdf:Alt><rdf:li xml:lang="x-default">Fireworks &amp; river</rdf:li></rdf:Alt></dc:title>
            </rdf:Description></rdf:RDF></x:xmpmeta>"#;
        let mut m = Metadata::default();
        read(packet, &mut m);
        assert_eq!(m.device.make.as_deref(), Some("Canon"));
        assert_eq!(m.device.model.as_deref(), Some("Canon EOS R5"));
        assert_eq!(m.capture_time.as_deref(), Some("2023-07-04T09:30:00-04:00"));
        assert_eq!(m.descriptive.get("title").map(String::as_str), Some("Fireworks & river"));
        let loc = m.location.unwrap();
        assert!((loc.latitude.unwrap() - (40.0 + 26.767 / 60.0)).abs() < 1e-9);
        assert!(loc.longitude.unwrap() < 0.0);
    }

    #[test]
    fn a_packet_with_nothing_identifying_is_empty() {
        let packet = br#"<x:xmpmeta><rdf:RDF><rdf:Description xmp:Rating="3" tiff:Orientation="1"/></rdf:RDF></x:xmpmeta>"#;
        let mut m = Metadata::default();
        read(packet, &mut m);
        assert!(m.is_empty(), "{m:?}");
    }
}
