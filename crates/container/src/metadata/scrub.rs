//! Software names inside compressed audio: an encoder's identification in
//! the fill data of an AAC frame (`Lavc61.19.100` from ffmpeg's encoder) or
//! in the ancillary bytes of MP3 frames (LAME's `LAME3.100` padding). A copy
//! of the stream carries them unless they are cleared, and clearing them
//! changes nothing a decoder reads: the AAC fill element stays a fill
//! element, and MP3 ancillary data lies outside every frame's main data.
//!
//! [`encoder_idents`] finds such names, for a check that a file carries none.

use crate::mp3::{FrameHeader, Version};

/// The first bit of `buf` at bit position `at` onward, `n` bits (at most 32).
fn bits(buf: &[u8], at: usize, n: usize) -> Option<u32> {
    let mut v = 0u32;
    for i in at..at + n {
        let byte = *buf.get(i / 8)?;
        v = (v << 1) | u32::from((byte >> (7 - i % 8)) & 1);
    }
    Some(v)
}

fn clear_bits(buf: &mut [u8], from: usize, to: usize) {
    for i in from..to.min(buf.len() * 8) {
        buf[i / 8] &= !(1 << (7 - i % 8));
    }
}

/// Clear the payload of the fill element an AAC frame opens with, when it
/// is fill (`EXT_FILL`, `EXT_FILL_DATA` or `EXT_DATA_ELEMENT`): where an
/// encoder puts its name. `frame` is a raw data block, or one behind an
/// ADTS header. Returns whether anything was cleared.
pub fn aac_frame(frame: &mut [u8]) -> bool {
    // An ADTS header: 7 bytes, 9 with a CRC.
    let start = if frame.len() >= 7 && frame[0] == 0xFF && frame[1] & 0xF6 == 0xF0 {
        if frame[1] & 1 == 0 { 9 } else { 7 }
    } else {
        0
    };
    let base = start * 8;
    const ID_FIL: u32 = 6;
    if bits(frame, base, 3) != Some(ID_FIL) {
        return false;
    }
    let Some(mut count) = bits(frame, base + 3, 4) else { return false };
    let mut at = base + 7;
    if count == 15 {
        let Some(esc) = bits(frame, at, 8) else { return false };
        count += esc - 1;
        at += 8;
    }
    if count == 0 {
        return false;
    }
    let end = at + 8 * count as usize;
    if end > frame.len() * 8 {
        return false;
    }
    // Only fill: an SBR, dynamic-range or other extension payload is audio.
    match bits(frame, at, 4) {
        Some(0 | 1 | 2) => {}
        _ => return false,
    }
    let before = frame.to_vec();
    clear_bits(frame, at, end);
    before != frame
}

/// The side information of one Layer III frame: where its main data starts
/// (bytes back from the end of this frame's side information) and how many
/// bits it has.
fn layer3_main_data(frame: &[u8], h: &FrameHeader) -> Option<(usize, usize, usize)> {
    if h.layer != 3 {
        return None;
    }
    let mono = h.mode == 3;
    let channels = if mono { 1 } else { 2 };
    let mpeg1 = h.version == Version::Mpeg1;
    let side = match (mpeg1, mono) {
        (true, true) => 17,
        (true, false) => 32,
        (false, true) => 9,
        (false, false) => 17,
    };
    let side_at = 4 + if h.protected { 2 } else { 0 };
    let s = frame.get(side_at..side_at + side)?;
    let (begin, mut at, granules, per) = if mpeg1 {
        // main_data_begin 9, private 5 / 3, scfsi 4 per channel.
        (bits(s, 0, 9)? as usize, 9 + if mono { 5 } else { 3 } + 4 * channels, 2, 59)
    } else {
        (bits(s, 0, 8)? as usize, 8 + if mono { 1 } else { 2 }, 1, 63)
    };
    let mut main_bits = 0usize;
    for _ in 0..granules * channels {
        main_bits += bits(s, at, 12)? as usize;
        at += per;
    }
    Some((side_at + side, begin, main_bits))
}

/// Clear the ancillary bytes of a run of MP3 frames: those in the frames'
/// main-data areas that no frame's main data (bit reservoir included) takes.
/// Frames that are not Layer III, or a stream whose reservoir does not add
/// up, are left as they are. Returns the bytes cleared.
pub fn mp3_frames(frames: &mut [Vec<u8>]) -> usize {
    // Each frame's main-data area, as a span of one concatenated stream.
    let mut areas = Vec::with_capacity(frames.len()); // (frame, from, stream start, len)
    let mut used: Vec<(usize, usize)> = Vec::with_capacity(frames.len());
    let mut stream = 0usize;
    for (i, f) in frames.iter().enumerate() {
        let Some(h) = FrameHeader::parse(f) else { return 0 };
        let Some((area_at, begin, main_bits)) = layer3_main_data(f, &h) else { return 0 };
        let len = f.len().saturating_sub(area_at);
        let Some(start) = stream.checked_sub(begin) else { return 0 };
        used.push((start, start + main_bits.div_ceil(8)));
        areas.push((i, area_at, stream, len));
        stream += len;
    }
    if used.iter().any(|&(_, end)| end > stream) {
        return 0;
    }
    let mut taken = vec![false; stream];
    for (from, to) in used {
        taken[from..to].iter_mut().for_each(|t| *t = true);
    }
    let mut cleared = 0;
    for (i, area_at, start, len) in areas {
        for k in 0..len {
            if !taken[start + k] && frames[i][area_at + k] != 0 {
                frames[i][area_at + k] = 0;
                cleared += 1;
            }
        }
    }
    cleared
}

/// The names encoders leave in files, as the prefixes they start with.
const ENCODER_PREFIXES: [&[u8]; 10] =
    [b"Lavc", b"Lavf", b"LAME", b"libfaac", b"FAAC", b"Nero", b"FhG", b"iTunes", b"GPAC", b"libvorbis"];

/// Encoder names in `data`: each known prefix with the version that follows
/// it (`Lavc62.28.101`, `LAME3.100`; `LAME` alone counts too). What an
/// encoder pads after its version (LAME's `UUUU…`) is not part of the name.
pub fn encoder_idents(data: &[u8]) -> Vec<String> {
    let mut out = Vec::new();
    for prefix in ENCODER_PREFIXES {
        let mut from = 0;
        while let Some(i) = data[from..].windows(prefix.len()).position(|w| w == prefix) {
            let at = from + i;
            let version = data[at + prefix.len()..]
                .iter()
                .take(16)
                .take_while(|b| b.is_ascii_digit() || **b == b'.')
                .count();
            let ident = String::from_utf8_lossy(&data[at..at + prefix.len() + version]).into_owned();
            if !out.contains(&ident) {
                out.push(ident);
            }
            from = at + prefix.len();
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_leading_aac_fill_element_is_cleared_and_a_channel_element_is_not() {
        // ffmpeg's: FIL, count 15 + escape, EXT_FILL, then its name.
        let name = b"Lavc61.19.100";
        let cnt = name.len() + 2;
        let mut frame = vec![(6 << 5) | (15 << 1), ((cnt - 14) as u8) << 1, 0];
        // The payload starts at bit 15; put the name byte-aligned after the
        // extension type, as that encoder does.
        frame.extend_from_slice(name);
        frame.extend_from_slice(&[0, 0, 0x21, 0x10]); // then a CPE…
        let before = frame.clone();
        assert!(aac_frame(&mut frame));
        assert!(encoder_idents(&frame).is_empty(), "{frame:?}");
        assert_eq!(frame[frame.len() - 2..], before[before.len() - 2..], "the element after is untouched");
        assert!(!aac_frame(&mut frame), "cleared once");

        let mut cpe = vec![0x21, 0x10, 0x05, 0x00];
        assert!(!aac_frame(&mut cpe));
    }

    #[test]
    fn idents_are_found() {
        let found = encoder_idents(b"\x00\x01Lavc62.28.101\x00junk LAME3.100UUUU");
        assert_eq!(found, vec!["Lavc62.28.101".to_string(), "LAME3.100".to_string()]);
        assert!(encoder_idents(b"nothing here").is_empty());
    }
}
