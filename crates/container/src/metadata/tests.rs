//! Real-world shapes, built byte by byte: an iPhone-style MOV (QuickTime
//! keys with an ISO 6709 location, make and model, and an `mebx` location
//! track), an Android MP4 (`udta` `©xyz`, `com.android.version`), a JPEG
//! with EXIF GPS, a Matroska file with tags, FLAC and MP3 tags — and this
//! crate's own MP4 output, before and after writing kept metadata into it.

use super::*;

fn bx(kind: &[u8; 4], body: &[u8]) -> Vec<u8> {
    let mut b = ((body.len() + 8) as u32).to_be_bytes().to_vec();
    b.extend_from_slice(kind);
    b.extend_from_slice(body);
    b
}

fn cat(parts: &[&[u8]]) -> Vec<u8> {
    parts.concat()
}

/// 2024-05-01T10:34:56Z in QuickTime seconds.
const QT_2024_05_01: u32 = 3_797_404_496;

fn mvhd(created: u32) -> Vec<u8> {
    let mut b = vec![0u8; 4];
    b.extend_from_slice(&created.to_be_bytes());
    b.extend_from_slice(&created.to_be_bytes());
    b.extend_from_slice(&600u32.to_be_bytes());
    b.extend_from_slice(&[0u8; 84]);
    bx(b"mvhd", &b)
}

fn hdlr(kind: &[u8; 4]) -> Vec<u8> {
    let mut b = vec![0u8; 8];
    b.extend_from_slice(kind);
    b.extend_from_slice(&[0u8; 13]);
    bx(b"hdlr", &b)
}

fn stsd(entry: &[u8]) -> Vec<u8> {
    let mut b = vec![0, 0, 0, 0, 0, 0, 0, 1];
    b.extend_from_slice(entry);
    bx(b"stsd", &b)
}

fn trak(handler: &[u8; 4], entry: &[u8]) -> Vec<u8> {
    let stbl = bx(b"stbl", &stsd(entry));
    let minf = bx(b"minf", &stbl);
    let mdia = bx(b"mdia", &cat(&[&hdlr(handler), &minf]));
    bx(b"trak", &mdia)
}

/// QuickTime `meta` (no version) with `mdta` keys.
fn mdta_meta(items: &[(&str, &str)]) -> Vec<u8> {
    let mut keys = vec![0, 0, 0, 0];
    keys.extend_from_slice(&(items.len() as u32).to_be_bytes());
    let mut ilst = Vec::new();
    for (i, (k, v)) in items.iter().enumerate() {
        keys.extend_from_slice(&((k.len() + 8) as u32).to_be_bytes());
        keys.extend_from_slice(b"mdta");
        keys.extend_from_slice(k.as_bytes());
        let mut data = vec![0, 0, 0, 1, 0, 0, 0, 0];
        data.extend_from_slice(v.as_bytes());
        ilst.extend(bx(&(i as u32 + 1).to_be_bytes(), &bx(b"data", &data)));
    }
    bx(b"meta", &cat(&[&hdlr(b"mdta"), &bx(b"keys", &keys), &bx(b"ilst", &ilst)]))
}

/// An `mebx` sample entry whose one key is `key`.
fn mebx(key: &str) -> Vec<u8> {
    let mut keyd = b"mdta".to_vec();
    keyd.extend_from_slice(key.as_bytes());
    let local = bx(&1u32.to_be_bytes(), &bx(b"keyd", &keyd));
    let mut body = vec![0u8; 6];
    body.extend_from_slice(&1u16.to_be_bytes());
    body.extend(bx(b"keys", &local));
    bx(b"mebx", &body)
}

fn iphone_mov() -> Vec<u8> {
    let moov = cat(&[
        &mvhd(QT_2024_05_01),
        &trak(b"vide", &bx(b"hvc1", &[0u8; 78])),
        &trak(b"meta", &mebx("com.apple.quicktime.location.ISO6709")),
        &mdta_meta(&[
            ("com.apple.quicktime.location.accuracy.horizontal", "4.748"),
            ("com.apple.quicktime.location.ISO6709", "+37.3349-122.0090+010.000/"),
            ("com.apple.quicktime.make", "Apple"),
            ("com.apple.quicktime.model", "iPhone 15 Pro"),
            ("com.apple.quicktime.software", "17.4.1"),
            ("com.apple.quicktime.creationdate", "2024-05-01T12:34:56+0200"),
        ]),
    ]);
    let mut ftyp = b"qt  ".to_vec();
    ftyp.extend_from_slice(&[0, 0, 0, 0]);
    ftyp.extend_from_slice(b"qt  ");
    cat(&[&bx(b"ftyp", &ftyp), &bx(b"moov", &moov), &bx(b"mdat", &[0u8; 32])])
}

fn android_mp4() -> Vec<u8> {
    let xyz = "+37.4219-122.0840/";
    let mut item = (xyz.len() as u16).to_be_bytes().to_vec();
    item.extend_from_slice(&0x15C7u16.to_be_bytes());
    item.extend_from_slice(xyz.as_bytes());
    let udta = bx(b"udta", &bx(&[0xA9, b'x', b'y', b'z'], &item));
    let moov = cat(&[
        &mvhd(QT_2024_05_01),
        &trak(b"vide", &bx(b"avc1", &[0u8; 78])),
        &trak(b"soun", &bx(b"mp4a", &[0u8; 28])),
        &udta,
        &mdta_meta(&[("com.android.version", "14"), ("com.android.capture.fps", "30.0")]),
    ]);
    let mut ftyp = b"mp42".to_vec();
    ftyp.extend_from_slice(&[0, 0, 0, 0]);
    ftyp.extend_from_slice(b"isommp42");
    cat(&[&bx(b"ftyp", &ftyp), &bx(b"moov", &moov), &bx(b"mdat", &[0u8; 32])])
}

/// Little-endian EXIF the way a phone writes it: Make, Model, DateTime and
/// a GPS IFD at 51°30'2.52"N 0°7'28.56"W.
fn phone_exif() -> Vec<u8> {
    let mut t = b"II*\0".to_vec();
    t.extend_from_slice(&8u32.to_le_bytes());
    // IFD0 at 8: 4 entries.
    let ifd0_len = 2 + 4 * 12 + 4;
    let strings_at = 8 + ifd0_len;
    let make = b"Google\0";
    let model = b"Pixel 8\0";
    let date = b"2024:05:01 12:34:56\0";
    let model_at = strings_at + make.len();
    let date_at = model_at + model.len();
    let gps_at = date_at + date.len();
    let entry = |t: &mut Vec<u8>, tag: u16, kind: u16, count: u32, value: u32| {
        t.extend_from_slice(&tag.to_le_bytes());
        t.extend_from_slice(&kind.to_le_bytes());
        t.extend_from_slice(&count.to_le_bytes());
        t.extend_from_slice(&value.to_le_bytes());
    };
    t.extend_from_slice(&4u16.to_le_bytes());
    entry(&mut t, 0x010F, 2, make.len() as u32, strings_at as u32);
    entry(&mut t, 0x0110, 2, model.len() as u32, model_at as u32);
    entry(&mut t, 0x0132, 2, date.len() as u32, date_at as u32);
    entry(&mut t, 0x8825, 4, 1, gps_at as u32);
    t.extend_from_slice(&0u32.to_le_bytes());
    t.extend_from_slice(make);
    t.extend_from_slice(model);
    t.extend_from_slice(date);
    assert_eq!(t.len(), gps_at);
    // GPS IFD: 4 entries, then two 3-rational values.
    let gps_len = 2 + 4 * 12 + 4;
    let lat_at = gps_at + gps_len;
    let lon_at = lat_at + 24;
    t.extend_from_slice(&4u16.to_le_bytes());
    entry(&mut t, 1, 2, 2, u32::from(b'N'));
    entry(&mut t, 2, 5, 3, lat_at as u32);
    entry(&mut t, 3, 2, 2, u32::from(b'W'));
    entry(&mut t, 4, 5, 3, lon_at as u32);
    t.extend_from_slice(&0u32.to_le_bytes());
    for (n, d) in [(51u32, 1u32), (30, 1), (252, 100), (0, 1), (7, 1), (2856, 100)] {
        t.extend_from_slice(&n.to_le_bytes());
        t.extend_from_slice(&d.to_le_bytes());
    }
    t
}

fn jpeg_with_exif() -> Vec<u8> {
    let mut app1 = b"Exif\0\0".to_vec();
    app1.extend(phone_exif());
    let mut j = vec![0xFF, 0xD8];
    j.extend_from_slice(&[0xFF, 0xE0, 0, 16]);
    j.extend_from_slice(b"JFIF\0\x01\x01\0\0\x01\0\x01\0\0");
    j.extend_from_slice(&[0xFF, 0xE1]);
    j.extend_from_slice(&((app1.len() + 2) as u16).to_be_bytes());
    j.extend(app1);
    j.extend_from_slice(&[0xFF, 0xDA, 0, 2, 0x12, 0x34, 0xFF, 0xD9]);
    j
}

/// This crate's own MP4: an H.264 track of filler slices.
fn rivet_mp4() -> Vec<u8> {
    use frame::{EncodedPacket, VideoCodec};
    let unhex = |s: &str| (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap()).collect::<Vec<u8>>();
    let sps = unhex("676e001ea6cd940a02ff970110000003001000000303c0f162d960");
    let pps = unhex("68ebe1b2c8b0");
    let au = |nals: &[&[u8]]| -> bytes::Bytes {
        nals.iter().flat_map(|n| [&[0u8, 0, 0, 1][..], n].concat()).collect::<Vec<u8>>().into()
    };
    let mut muxer = crate::mux::Av1Mp4Muxer::new_with_codec(640, 360, 30.0, VideoCodec::H264).unwrap();
    muxer.add_packet(EncodedPacket { data: au(&[&sps, &pps, &[0x65, 0x88, 0x84, 0x00]]), pts: 0, is_keyframe: true }).unwrap();
    for i in 1..4u64 {
        muxer.add_packet(EncodedPacket { data: au(&[&[0x41, 0x9a, 0x02, 0x03]]), pts: i, is_keyframe: false }).unwrap();
    }
    muxer.finalize().unwrap().to_vec()
}

#[test]
fn an_iphone_mov_shows_location_device_time_and_its_location_track() {
    let m = read(&iphone_mov());
    let loc = m.location.clone().unwrap();
    assert_eq!((loc.latitude, loc.longitude, loc.altitude), (Some(37.3349), Some(-122.009), Some(10.0)));
    assert_eq!(m.device.make.as_deref(), Some("Apple"));
    assert_eq!(m.device.model.as_deref(), Some("iPhone 15 Pro"));
    assert_eq!(m.device.software.as_deref(), Some("17.4.1"));
    // The key's local time, over mvhd's UTC.
    assert_eq!(m.capture_time.as_deref(), Some("2024-05-01T12:34:56+02:00"));
    assert_eq!(m.timed_tracks.len(), 1);
    assert_eq!(m.timed_tracks[0].label, "Apple location track");
    assert_eq!(m.timed_tracks[0].kind, TimedTrackKind::Location);
    assert!(m.unclassified.is_empty(), "{:?}", m.unclassified);
    assert_eq!(m.categories(), Categories::ALL.minus(Categories::NONE.with(Category::Descriptive)));
}

#[test]
fn an_android_mp4_shows_its_xyz_location_and_version() {
    let m = read(&android_mp4());
    let loc = m.location.clone().unwrap();
    assert_eq!((loc.latitude, loc.longitude), (Some(37.4219), Some(-122.084)));
    assert_eq!(m.device.software.as_deref(), Some("14"));
    assert_eq!(m.capture_time.as_deref(), Some("2024-05-01T10:34:56Z"));
    assert!(m.timed_tracks.is_empty());
    assert!(m.unclassified.is_empty(), "{:?}", m.unclassified);
}

#[test]
fn a_jpeg_shows_its_exif_gps_and_camera() {
    let m = read(&jpeg_with_exif());
    let loc = m.location.clone().unwrap();
    assert!((loc.latitude.unwrap() - (51.0 + 30.0 / 60.0 + 2.52 / 3600.0)).abs() < 1e-9);
    assert!((loc.longitude.unwrap() + (7.0 / 60.0 + 28.56 / 3600.0)).abs() < 1e-9);
    assert_eq!(m.device.make.as_deref(), Some("Google"));
    assert_eq!(m.device.model.as_deref(), Some("Pixel 8"));
    assert_eq!(m.capture_time.as_deref(), Some("2024-05-01T12:34:56"));
}

#[test]
fn png_webp_and_tiff_carry_the_same_exif() {
    let exif = phone_exif();
    let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
    png.extend_from_slice(&13u32.to_be_bytes());
    png.extend_from_slice(b"IHDR");
    png.extend_from_slice(&[0u8; 13 + 4]);
    png.extend_from_slice(&(exif.len() as u32).to_be_bytes());
    png.extend_from_slice(b"eXIf");
    png.extend_from_slice(&exif);
    png.extend_from_slice(&[0u8; 4]);
    let mut text = b"Title\0Harbour".to_vec();
    png.extend_from_slice(&(text.len() as u32).to_be_bytes());
    png.extend_from_slice(b"tEXt");
    png.append(&mut text);
    png.extend_from_slice(&[0u8; 4]);

    let mut webp = b"RIFF\0\0\0\0WEBP".to_vec();
    webp.extend_from_slice(b"EXIF");
    webp.extend_from_slice(&(exif.len() as u32).to_le_bytes());
    webp.extend_from_slice(&exif);
    if exif.len() % 2 == 1 {
        webp.push(0);
    }

    for (name, file) in [("png", png), ("webp", webp), ("tiff", exif.clone())] {
        let m = read(&file);
        assert!(m.location.as_ref().is_some_and(Location::has_coordinates), "{name}");
        assert_eq!(m.device.model.as_deref(), Some("Pixel 8"), "{name}");
        if name == "png" {
            assert_eq!(m.descriptive.get("title").map(String::as_str), Some("Harbour"));
        }
    }
}

#[test]
fn a_heif_exif_item_is_found_through_iloc() {
    let exif = phone_exif();
    let mut item = 0u32.to_be_bytes().to_vec();
    item.extend_from_slice(&exif);
    // infe v2: item 1, Exif.
    let mut infe = vec![2, 0, 0, 0];
    infe.extend_from_slice(&1u16.to_be_bytes());
    infe.extend_from_slice(&0u16.to_be_bytes());
    infe.extend_from_slice(b"Exif");
    infe.push(0);
    let mut iinf = vec![0, 0, 0, 0];
    iinf.extend_from_slice(&1u16.to_be_bytes());
    iinf.extend(bx(b"infe", &infe));
    let ftyp = bx(b"ftyp", b"heic\0\0\0\0mif1heic");
    // Lay out: ftyp, meta, mdat(item). The iloc offset depends on meta's size,
    // which does not depend on the offset's value.
    let build = |offset: u32| {
        let mut iloc = vec![0, 0, 0, 0, 0x44, 0x00];
        iloc.extend_from_slice(&1u16.to_be_bytes());
        iloc.extend_from_slice(&1u16.to_be_bytes()); // item id
        iloc.extend_from_slice(&0u16.to_be_bytes()); // data ref
        iloc.extend_from_slice(&1u16.to_be_bytes()); // extents
        iloc.extend_from_slice(&offset.to_be_bytes());
        iloc.extend_from_slice(&(item.len() as u32).to_be_bytes());
        let mut meta = vec![0, 0, 0, 0];
        meta.extend(hdlr(b"pict"));
        meta.extend(bx(b"iinf", &iinf));
        meta.extend(bx(b"iloc", &iloc));
        bx(b"meta", &meta)
    };
    let meta_len = build(0).len();
    let offset = (ftyp.len() + meta_len + 8) as u32;
    let file = cat(&[&ftyp, &build(offset), &bx(b"mdat", &item)]);
    let m = read(&file);
    assert_eq!(m.device.model.as_deref(), Some("Pixel 8"));
    assert!(m.location.is_some());
}

fn ebml(id: u32, body: &[u8]) -> Vec<u8> {
    let mut out: Vec<u8> = id.to_be_bytes().iter().copied().skip_while(|&b| b == 0).collect();
    out.push(0x01); // an 8-byte size
    out.extend_from_slice(&(body.len() as u64).to_be_bytes()[1..]);
    out.extend_from_slice(body);
    out
}

#[test]
fn a_matroska_file_shows_its_tags_and_date() {
    let simple = |name: &str, value: &str| ebml(0x67C8, &cat(&[&ebml(0x45A3, name.as_bytes()), &ebml(0x4487, value.as_bytes())]));
    let tags = ebml(
        0x1254_C367,
        &ebml(
            0x7373,
            &cat(&[
                &simple("TITLE", "Holiday"),
                &simple("com.apple.quicktime.location.ISO6709", "+48.8584+002.2945/"),
                &simple("com.apple.quicktime.make", "Apple"),
                &simple("ENCODER", "Lavf61.7.100"),
                &simple("DURATION", "00:00:01.000000000"),
            ]),
        ),
    );
    let info = ebml(0x1549_A966, &cat(&[&ebml(0x4D80, b"Lavf61.7.100"), &ebml(0x4461, &0i64.to_be_bytes())]));
    let file = cat(&[&ebml(0x1A45_DFA3, &ebml(0x4282, b"webm")), &ebml(0x1853_8067, &cat(&[&info, &tags]))]);
    let m = read(&file);
    assert_eq!(m.descriptive.get("title").map(String::as_str), Some("Holiday"));
    assert_eq!(m.location.as_ref().and_then(|l| l.latitude), Some(48.8584));
    assert_eq!(m.device.make.as_deref(), Some("Apple"));
    assert_eq!(m.device.software.as_deref(), Some("Lavf61.7.100"));
    assert_eq!(m.capture_time.as_deref(), Some("2001-01-01T00:00:00Z"));
    assert!(m.unclassified.is_empty(), "{:?}", m.unclassified);
}

#[test]
fn flac_and_mp3_tags_round_trip_through_the_writers() {
    let mut kept = Metadata::default();
    kept.descriptive.insert("title".into(), "Song".into());
    kept.descriptive.insert("artist".into(), "Band".into());
    kept.capture_time = Some("2020-02-02T00:00:00Z".into());

    // STREAMINFO (34 bytes) flagged last, a frame's worth of bytes after.
    let mut flac_file = b"fLaC".to_vec();
    flac_file.extend_from_slice(&[0x80, 0, 0, 34]);
    let mut si = vec![0u8; 34];
    si[10] = 0x0A; // 44100 Hz, stereo, 16 bits
    si[11] = 0xC4;
    si[12] = 0x42;
    si[13] = 0xF0;
    flac_file.extend_from_slice(&si);
    flac_file.extend_from_slice(&[0xFF, 0xF8, 0x69, 0x08, 0x00, 0x00]);
    assert!(read(&flac_file).is_empty(), "a bare FLAC stream carries nothing");
    let written = write::flac(&flac_file, &kept).unwrap();
    assert!(written.ends_with(&[0xFF, 0xF8, 0x69, 0x08, 0x00, 0x00]), "the audio follows untouched");
    let back = read(&written);
    assert_eq!(back.descriptive.get("title").map(String::as_str), Some("Song"));
    assert_eq!(back.capture_time.as_deref(), Some("2020-02-02T00:00:00Z"));
    assert!(back.device.is_empty(), "no vendor string: {:?}", back.device);
    let none = write::flac(&flac_file, &Metadata::default()).unwrap();
    assert!(read(&none).is_empty());

    // Two MPEG-1 Layer III frame headers, 128 kbps 44.1 kHz: 417 bytes each.
    let mut frame = vec![0u8; 417];
    frame[..4].copy_from_slice(&[0xFF, 0xFB, 0x90, 0x64]);
    let mp3_file = cat(&[&frame, &frame, &frame]);
    let written = write::mp3(&mp3_file, &kept);
    assert!(written.starts_with(b"ID3\x04"));
    assert!(written.ends_with(&mp3_file));
    let back = read(&written);
    assert_eq!(back.descriptive.get("artist").map(String::as_str), Some("Band"));
    assert_eq!(back.capture_time.as_deref(), Some("2020-02-02T00:00:00Z"));
    assert_eq!(write::mp3(&mp3_file, &Metadata::default()), mp3_file);
}

#[test]
fn this_crates_mp4_output_carries_nothing() {
    let m = read(&rivet_mp4());
    assert!(m.is_empty(), "{m:?}");
}

#[test]
fn keeping_nothing_writes_nothing() {
    let out = rivet_mp4();
    let source = read(&iphone_mov());
    let written = write::mp4(&out, &source.kept(Keep::NONE)).unwrap();
    assert_eq!(written, out);
}

#[test]
fn each_kept_category_is_written_alone_and_the_media_still_lines_up() {
    let out = rivet_mp4();
    let source = read(&iphone_mov());
    let before = crate::streaming::demux_streaming_shared(bytes::Bytes::from(out.clone())).unwrap();
    let before_first = before.header().clone();
    for policy in ["location", "device", "capture_time", "all"] {
        let policy = Keep::parse(policy).unwrap();
        let keep = policy.categories();
        let written = write::mp4(&out, &source.kept(policy)).unwrap();
        let back = read(&written);
        let expect = keep.minus(Categories::NONE.with(Category::Descriptive));
        assert_eq!(back.categories(), expect, "keep {keep}: {back:?}");
        assert!(back.timed_tracks.is_empty(), "timed tracks never travel");
        assert!(back.unclassified.is_empty(), "{:?}", back.unclassified);
        if keep.contains(Category::Location) {
            let loc = back.location.clone().unwrap();
            assert_eq!((loc.latitude, loc.longitude), (Some(37.3349), Some(-122.009)));
        }
        if keep.contains(Category::CaptureTime) {
            assert_eq!(back.capture_time.as_deref(), Some("2024-05-01T12:34:56+02:00"));
        }
        // The samples are where the moved chunk offsets say.
        let demuxed = crate::streaming::demux_streaming_shared(bytes::Bytes::from(written.clone())).unwrap();
        assert_eq!(demuxed.header().info.width, before_first.info.width);
        let a = sample_bytes(&out);
        let b = sample_bytes(&written);
        assert_eq!(a, b, "keep {keep}: samples moved without their offsets");
    }
}

/// The bytes the first chunk offset points at.
fn sample_bytes(file: &[u8]) -> Vec<u8> {
    let moov = isobmff::boxes(file).find(|b| &b.kind == b"moov").unwrap().body;
    let stco = {
        let at = moov.windows(4).position(|w| w == b"stco" || w == b"co64").unwrap();
        (&moov[at..at + 4], &moov[at + 4..])
    };
    let off = match stco.0 {
        b"stco" => be32(stco.1, 8).unwrap() as usize,
        _ => be64(stco.1, 8).unwrap() as usize,
    };
    file[off..off + 16].to_vec()
}

#[test]
fn descriptive_tags_round_trip_through_mp4() {
    let mut kept = Metadata::default();
    kept.descriptive.insert("title".into(), "Harbour at dusk".into());
    kept.descriptive.insert("copyright".into(), "2024 Someone".into());
    let written = write::mp4(&rivet_mp4(), &kept).unwrap();
    let back = read(&written);
    assert_eq!(back.descriptive, kept.descriptive);
    assert_eq!(back.categories(), Categories::NONE.with(Category::Descriptive));
}

#[test]
fn a_fragmented_file_is_refused() {
    let mut kept = Metadata::default();
    kept.descriptive.insert("title".into(), "x".into());
    let frag = cat(&[&bx(b"ftyp", b"iso6\0\0\0\0iso6"), &bx(b"moov", &mvhd(0)), &bx(b"moof", &[]), &bx(b"mdat", &[])]);
    assert!(write::mp4(&frag, &kept).is_err());
}

#[test]
fn unknown_items_are_listed_not_dropped() {
    let moov = cat(&[&mvhd(0), &bx(b"udta", &bx(b"ZZZZ", b"secret"))]);
    let m = read(&cat(&[&bx(b"ftyp", b"isom\0\0\0\0isom"), &bx(b"moov", &moov)]));
    assert_eq!(m.unclassified, vec!["udta/ZZZZ".to_string()]);
    assert!(!m.is_empty());
}

#[test]
fn category_lists_parse() {
    assert_eq!(Categories::parse_list("").unwrap(), Categories::NONE);
    assert_eq!(Categories::parse_list("none").unwrap(), Categories::NONE);
    assert_eq!(Categories::parse_list("all").unwrap(), Categories::ALL);
    let c = Categories::parse_list("location, capture-time").unwrap();
    assert!(c.contains(Category::Location) && c.contains(Category::CaptureTime) && !c.contains(Category::Device));
    assert_eq!(c.to_string(), "location,capture_time");
    assert!(Categories::parse_list("gps").is_err());
}

#[test]
fn dates_normalise() {
    assert_eq!(normalize_date("2024:05:01 12:34:56"), "2024-05-01T12:34:56");
    assert_eq!(normalize_date("2024-05-01T12:34:56+0200"), "2024-05-01T12:34:56+02:00");
    assert_eq!(normalize_date("2024-05-01T12:34:56.123Z"), "2024-05-01T12:34:56.123Z");
    assert_eq!(normalize_date("1999"), "1999");
    assert_eq!(quicktime_time(u64::from(QT_2024_05_01)), "2024-05-01T10:34:56Z");
    assert_eq!(parse_unix_time("2024-05-01T12:34:56+02:00"), parse_unix_time("2024-05-01T10:34:56Z"));
}

#[test]
fn truncations_never_panic() {
    for file in [iphone_mov(), android_mp4(), jpeg_with_exif(), rivet_mp4()] {
        for n in (0..file.len()).step_by(3) {
            let _ = read(&file[..n]);
        }
    }
}

#[test]
fn an_avi_info_list_is_read() {
    let sub = |id: &[u8; 4], v: &[u8]| {
        let mut b = id.to_vec();
        b.extend_from_slice(&(v.len() as u32).to_le_bytes());
        b.extend_from_slice(v);
        if v.len() % 2 == 1 {
            b.push(0);
        }
        b
    };
    let info = cat(&[b"INFO", &sub(b"ISFT", b"Lavf58.76.100\0"), &sub(b"ICRD", b"2019-03-04\0")]);
    let list = sub(b"LIST", &info);
    let file = cat(&[b"RIFF", &((list.len() + 4) as u32).to_le_bytes(), b"AVI ", &list]);
    let m = read(&file);
    assert_eq!(m.device.software.as_deref(), Some("Lavf58.76.100"));
    assert_eq!(m.capture_time.as_deref(), Some("2019-03-04"));
}

#[test]
fn keep_policies_parse_and_print() {
    assert_eq!(Keep::parse("").unwrap(), Keep::NONE);
    assert_eq!(Keep::parse("none").unwrap(), Keep::NONE);
    assert_eq!(Keep::parse("all").unwrap(), Keep::ALL);
    let k = Keep::parse("location:approximate, capture_time:date,device,descriptive").unwrap();
    assert_eq!(
        k,
        Keep { location: LocationKeep::Approximate, capture_time: TimeKeep::Date, device: DeviceKeep::Keep, descriptive: true }
    );
    assert_eq!(k.to_string(), "location:approximate,capture_time:date,device,descriptive");
    assert_eq!(Keep::parse(&k.to_string()).unwrap(), k);
    assert_eq!(Keep::parse("device:all").unwrap().device, DeviceKeep::All);
    for bad in ["gps", "location:rough", "descriptive:some", "all:yes"] {
        assert!(Keep::parse(bad).is_err(), "{bad}");
    }
}

fn phone() -> Metadata {
    let mut m = read(&iphone_mov());
    m.location.as_mut().unwrap().name = Some("Apple Park".into());
    m.device.serial = Some("F2LXK0Q1".into());
    m.device.owner = Some("Ada".into());
    m
}

#[test]
fn an_approximate_location_is_rounded_and_nothing_finer_travels() {
    let kept = phone().kept(Keep::parse("location:approximate").unwrap());
    assert_eq!(kept.location, Some(Location { latitude: Some(37.33), longitude: Some(-122.01), altitude: None, name: None }));
    let written = write::mp4(&rivet_mp4(), &kept).unwrap();
    let back = read(&written);
    assert_eq!(back.location.as_ref().and_then(|l| l.latitude), Some(37.33));
    let keep = Keep::parse("location:approximate").unwrap();
    assert!(back.violations(keep, &[]).is_empty(), "{:?}", back.violations(keep, &[]));
    // The whole location, checked against approximate, is refused.
    let full = read(&write::mp4(&rivet_mp4(), &phone().kept(Keep::parse("location").unwrap())).unwrap());
    assert!(full.violations(keep, &[]).iter().any(|v| v.contains("finer than approximate")));
}

#[test]
fn a_date_keeps_the_day_and_zeroes_the_time() {
    let keep = Keep::parse("capture_time:date").unwrap();
    let kept = phone().kept(keep);
    assert_eq!(kept.capture_time.as_deref(), Some("2024-05-01T00:00:00"));
    let back = read(&write::mp4(&rivet_mp4(), &kept).unwrap());
    assert!(back.capture_time.as_deref().unwrap().starts_with("2024-05-01T00:00:00"), "{:?}", back.capture_time);
    assert!(back.violations(keep, &[]).is_empty(), "{:?}", back.violations(keep, &[]));
    let full = read(&write::mp4(&rivet_mp4(), &phone().kept(Keep::parse("capture_time").unwrap())).unwrap());
    assert!(full.violations(keep, &[]).iter().any(|v| v.contains("time of day")));
}

#[test]
fn device_keep_drops_serial_and_owner_and_keep_all_does_not() {
    let kept = phone().kept(Keep::parse("device").unwrap());
    assert_eq!(kept.device.model.as_deref(), Some("iPhone 15 Pro"));
    assert!(kept.device.serial.is_none() && kept.device.owner.is_none());
    let all = phone().kept(Keep::parse("device:all").unwrap());
    assert_eq!((all.device.serial.as_deref(), all.device.owner.as_deref()), (Some("F2LXK0Q1"), Some("Ada")));
    // In a still, where EXIF has a place for them.
    let tiff = exif::build(&all).unwrap();
    let mut back = Metadata::default();
    exif::read_tiff(&tiff, &mut back);
    assert!(back.violations(Keep::parse("device").unwrap(), &[]).iter().any(|v| v.contains("serial")));
    assert!(back.violations(Keep::parse("device:all").unwrap(), &[]).is_empty());
}

#[test]
fn encoder_names_in_audio_are_device_metadata_unless_allowed() {
    let mut m = Metadata::default();
    m.embedded_software.push("Lavc62.28.101".into());
    assert_eq!(m.categories(), Categories::NONE.with(Category::Device));
    assert!(m.violations(Keep::NONE, &[]).iter().any(|v| v.contains("encoder name")));
    assert!(m.violations(Keep::parse("device").unwrap(), &[]).is_empty());
    let mut ours = Metadata::default();
    ours.embedded_software.push("LAME3.100".into());
    assert!(ours.violations(Keep::NONE, &["LAME3.100".into()]).is_empty());
}
