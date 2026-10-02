//! Tags in audio files and RIFF: FLAC `VORBIS_COMMENT` and `PICTURE` blocks,
//! ID3v2 (2.2, 2.3, 2.4) and ID3v1, and a RIFF `LIST INFO` chunk.

use super::{Category, DeviceField, Metadata, be32, le32};

const FLAC_VORBIS_COMMENT: u8 = 4;
const FLAC_PICTURE: u8 = 6;
const FLAC_APPLICATION: u8 = 2;

/// A native FLAC stream from its `fLaC` marker.
pub(crate) fn read_flac(data: &[u8], m: &mut Metadata) {
    let Some(blocks) = data.strip_prefix(b"fLaC") else { return };
    read_flac_blocks(blocks, m);
}

/// FLAC metadata blocks (the `dfLa` body after its version and flags, or a
/// native stream after `fLaC`).
pub(crate) fn read_flac_blocks(blocks: &[u8], m: &mut Metadata) {
    let mut at = 0;
    while at + 4 <= blocks.len() {
        let header = blocks[at];
        let len = (usize::from(blocks[at + 1]) << 16) | (usize::from(blocks[at + 2]) << 8) | usize::from(blocks[at + 3]);
        let Some(body) = blocks.get(at + 4..at + 4 + len) else { return };
        match header & 0x7F {
            FLAC_VORBIS_COMMENT => read_vorbis_comment(body, m),
            FLAC_PICTURE => m.set_descriptive("picture", "present"),
            FLAC_APPLICATION => m.unclassified(format!("flac/APPLICATION {}", super::text(body.get(..4).unwrap_or_default()))),
            _ => {}
        }
        if header & 0x80 != 0 {
            return;
        }
        at += 4 + len;
    }
}

/// A Vorbis comment block: vendor string, then `NAME=value` comments.
pub(crate) fn read_vorbis_comment(body: &[u8], m: &mut Metadata) {
    let Some(vendor_len) = le32(body, 0).map(|l| l as usize) else { return };
    let Some(vendor) = body.get(4..4 + vendor_len) else { return };
    m.set_device(DeviceField::Software, &super::text(vendor));
    let mut at = 4 + vendor_len;
    let Some(count) = le32(body, at) else { return };
    at += 4;
    for _ in 0..count.min(4096) {
        let Some(len) = le32(body, at).map(|l| l as usize) else { return };
        let Some(c) = body.get(at + 4..at + 4 + len) else { return };
        at += 4 + len;
        let c = String::from_utf8_lossy(c);
        let Some((name, value)) = c.split_once('=') else { continue };
        match name.to_ascii_uppercase().as_str() {
            "METADATA_BLOCK_PICTURE" | "COVERART" => m.set_descriptive("picture", "present"),
            // ReplayGain and the like: how to play it.
            n if n.starts_with("REPLAYGAIN_") || n == "R128_TRACK_GAIN" || n == "R128_ALBUM_GAIN" || n == "WAVEFORMATEXTENSIBLE_CHANNEL_MASK" => {}
            _ => {
                if !m.set_by_name(name, value) {
                    m.set_descriptive(&name.to_ascii_lowercase(), value);
                }
            }
        }
    }
}

fn syncsafe(b: &[u8]) -> usize {
    b.iter().take(4).fold(0usize, |acc, &x| (acc << 7) | usize::from(x & 0x7F))
}

/// An ID3v2 tag at the start of `data`, and any that follow it.
pub(crate) fn read_id3(data: &[u8], m: &mut Metadata) {
    let mut at = 0;
    while data.get(at..at + 3) == Some(b"ID3") && at + 10 <= data.len() {
        let major = data[at + 3];
        let flags = data[at + 5];
        let size = syncsafe(&data[at + 6..at + 10]);
        let Some(tag) = data.get(at + 10..at + 10 + size) else { return };
        read_id3_frames(tag, major, flags, m);
        at += 10 + size + if flags & 0x10 != 0 { 10 } else { 0 };
    }
}

fn read_id3_frames(tag: &[u8], major: u8, flags: u8, m: &mut Metadata) {
    let mut at = 0;
    // An extended header.
    if flags & 0x40 != 0 && major >= 3 {
        let ext = if major == 4 { syncsafe(tag.get(0..4).unwrap_or_default()) } else { be32(tag, 0).unwrap_or(0) as usize + 4 };
        at = ext;
    }
    let (id_len, header_len) = if major == 2 { (3, 6) } else { (4, 10) };
    while at + header_len <= tag.len() {
        let id = &tag[at..at + id_len];
        if id[0] == 0 {
            return;
        }
        let size = match major {
            2 => (usize::from(tag[at + 3]) << 16) | (usize::from(tag[at + 4]) << 8) | usize::from(tag[at + 5]),
            3 => be32(tag, at + 4).unwrap_or(0) as usize,
            _ => syncsafe(&tag[at + 4..at + 8]),
        };
        let Some(body) = tag.get(at + header_len..at + header_len + size) else { return };
        at += header_len + size;
        let id = String::from_utf8_lossy(id).into_owned();
        id3_frame(&id, body, m);
    }
}

/// An ID3 text value: an encoding byte, then text in that encoding.
fn id3_text(body: &[u8]) -> String {
    let Some((&enc, rest)) = body.split_first() else { return String::new() };
    let s = match enc {
        1 => match rest {
            [0xFF, 0xFE, r @ ..] => super::utf16(r, false),
            [0xFE, 0xFF, r @ ..] => super::utf16(r, true),
            r => super::utf16(r, false),
        },
        2 => super::utf16(rest, true),
        3 => super::text(rest),
        _ => rest.iter().take_while(|&&b| b != 0).map(|&b| b as char).collect(),
    };
    // Multiple values are NUL-separated in v2.4.
    s.split('\0').filter(|p| !p.is_empty()).collect::<Vec<_>>().join("; ")
}

/// `COMM` / `USLT`: encoding, language, a description, then the text.
fn id3_comment(body: &[u8]) -> String {
    let Some(&enc) = body.first() else { return String::new() };
    let Some(rest) = body.get(4..) else { return String::new() };
    let wide = enc == 1 || enc == 2;
    let split = if wide {
        rest.chunks_exact(2).position(|c| c == [0, 0]).map(|i| i * 2 + 2)
    } else {
        rest.iter().position(|&b| b == 0).map(|i| i + 1)
    };
    let text = split.and_then(|s| rest.get(s..)).unwrap_or_default();
    let mut with_enc = vec![enc];
    with_enc.extend_from_slice(text);
    id3_text(&with_enc)
}

fn id3_frame(id: &str, body: &[u8], m: &mut Metadata) {
    match id {
        "TIT2" | "TT2" => m.set_descriptive("title", &id3_text(body)),
        "TPE1" | "TP1" | "TPE2" | "TP2" => m.set_descriptive("artist", &id3_text(body)),
        "TALB" | "TAL" => m.set_descriptive("album", &id3_text(body)),
        "TCOP" | "TCR" => m.set_descriptive("copyright", &id3_text(body)),
        "TCON" | "TCO" => m.set_descriptive("genre", &id3_text(body)),
        "TCOM" | "TCM" => m.set_descriptive("composer", &id3_text(body)),
        "TENC" | "TEN" => m.set_descriptive("encoded_by", &id3_text(body)),
        "COMM" | "COM" => m.set_descriptive("comment", &id3_comment(body)),
        "USLT" | "ULT" => m.set_descriptive("lyrics", &id3_comment(body)),
        "APIC" | "PIC" => m.set_descriptive("picture", "present"),
        "TSSE" | "TSS" => m.set_device(DeviceField::Software, &id3_text(body)),
        "TOWN" => m.set_device(DeviceField::Owner, &id3_text(body)),
        "TDRC" | "TYER" | "TYE" | "TDOR" | "TORY" | "TDRL" | "TDEN" | "TDTG" => m.set_capture_time(&id3_text(body)),
        "TDAT" | "TDA" | "TIME" | "TIM" => m.present.insert(Category::CaptureTime),
        "TXXX" | "TXX" => {
            // encoding, description, NUL, value.
            let full = id3_text(body);
            let (name, value) = full.split_once("; ").unwrap_or((full.as_str(), ""));
            if !m.set_by_name(name, value) && !name.is_empty() {
                m.set_descriptive(&name.to_ascii_lowercase(), if value.is_empty() { "present" } else { value });
            }
        }
        // Playback and technical frames.
        "TLEN" | "TLE" | "TFLT" | "TKEY" | "TBPM" | "TMED" | "RVA2" | "RVAD" | "EQU2" | "MCDI" | "ETCO" | "SEEK" | "ASPI"
        | "POSS" | "SYLT" | "RBUF" | "AENC" | "ENCR" | "GRID" | "SIGN" | "TSIZ" => {}
        "PRIV" | "GEOB" | "UFID" | "PRV" | "GEO" | "UFI" => m.unclassified(format!("id3/{id}")),
        s if s.starts_with('T') || s.starts_with('W') => m.set_descriptive(&format!("id3_{}", s.to_ascii_lowercase()), &id3_text(body)),
        _ => m.unclassified(format!("id3/{id}")),
    }
}

/// An ID3v1 tag: the last 128 bytes, `TAG` then fixed fields.
pub(crate) fn read_id3v1(data: &[u8], m: &mut Metadata) {
    let Some(tag) = data.len().checked_sub(128).map(|at| &data[at..]) else { return };
    if !tag.starts_with(b"TAG") {
        return;
    }
    let field = |r: std::ops::Range<usize>| super::text(&tag[r]).trim().to_string();
    m.set_descriptive("title", &field(3..33));
    m.set_descriptive("artist", &field(33..63));
    m.set_descriptive("album", &field(63..93));
    m.set_capture_time(&field(93..97));
    m.set_descriptive("comment", &field(97..125));
}

/// A RIFF file's `LIST` `INFO` chunk (AVI, WAV).
pub(crate) fn read_riff_info(data: &[u8], m: &mut Metadata) {
    let mut at = 12;
    while at + 8 <= data.len() {
        let Some(len) = le32(data, at + 4).map(|l| l as usize) else { return };
        let id = &data[at..at + 4];
        let Some(body) = data.get(at + 8..at + 8 + len) else { return };
        if id == b"LIST" && body.starts_with(b"INFO") {
            let mut i = 4;
            while i + 8 <= body.len() {
                let Some(l) = le32(body, i + 4).map(|l| l as usize) else { break };
                let Some(v) = body.get(i + 8..i + 8 + l) else { break };
                let v = super::text(v);
                match &body[i..i + 4] {
                    b"INAM" => m.set_descriptive("title", &v),
                    b"IART" => m.set_descriptive("artist", &v),
                    b"ICOP" => m.set_descriptive("copyright", &v),
                    b"ICMT" => m.set_descriptive("comment", &v),
                    b"ISBJ" => m.set_descriptive("subject", &v),
                    b"IKEY" => m.set_descriptive("keywords", &v),
                    b"IGNR" => m.set_descriptive("genre", &v),
                    b"ICRD" => m.set_capture_time(&v),
                    b"ISFT" => m.set_device(DeviceField::Software, &v),
                    b"ISRC" | b"IENG" | b"ITCH" => m.set_descriptive("credits", &v),
                    other => m.unclassified(format!("riff/INFO {}", String::from_utf8_lossy(other))),
                }
                i += 8 + l + (l & 1);
            }
        }
        at += 8 + len + (len & 1);
    }
}

/// Encoder names in an MP3 stream's first and last few frames (its LAME
/// tag, and the padding some encoders fill with their name).
pub(crate) fn read_mp3_idents(data: &[u8], m: &mut Metadata) {
    let mut start = 0;
    while data.get(start..start + 3) == Some(b"ID3") && start + 10 <= data.len() {
        start += 10 + syncsafe(&data[start + 6..start + 10]);
    }
    let end = if data.len() >= 128 && data[data.len() - 128..].starts_with(b"TAG") { data.len() - 128 } else { data.len() };
    let Some(audio) = data.get(start..end) else { return };
    const SPAN: usize = 4096;
    let head = &audio[..audio.len().min(SPAN)];
    let tail = &audio[audio.len().saturating_sub(SPAN)..];
    for part in [head, tail] {
        for ident in super::scrub::encoder_idents(part) {
            if !m.embedded_software.contains(&ident) {
                m.embedded_software.push(ident);
            }
        }
    }
}
