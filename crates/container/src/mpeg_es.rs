//! MPEG-1 / MPEG-2 Video and MPEG-4 Part 2 Visual elementary-stream glue
//! for the muxers and demuxers: start codes, the configuration headers a
//! sample entry or a `CodecPrivate` carries, picture types.
//!
//! Both syntaxes are sequences of start-code units (`00 00 01 xx`). What a
//! container calls the decoder configuration is the run of units before the
//! first picture: MPEG-2's sequence header and its extensions (ISO/IEC
//! 14496-14 puts them in the `esds` of an `mp4v` entry with object type 0x61),
//! MPEG-4's visual object sequence, visual object and video object layer
//! (the `esds` DecoderSpecificInfo of object type 0x20, and a Matroska
//! `V_MPEG4/ISO/*` CodecPrivate).

/// MPEG-1 / MPEG-2 `picture_start_code` value.
pub const MPEG2_PICTURE: u8 = 0x00;
/// MPEG-1 / MPEG-2 `sequence_header_code` value.
pub const MPEG2_SEQUENCE_HEADER: u8 = 0xb3;
/// MPEG-1 / MPEG-2 `group_start_code` value.
pub const MPEG2_GOP: u8 = 0xb8;
/// MPEG-4 `vop_start_code` value.
pub const MPEG4_VOP: u8 = 0xb6;
/// MPEG-4 `group_of_vop_start_code` value.
pub const MPEG4_GOV: u8 = 0xb3;
/// MPEG-4 `visual_object_sequence_start_code` value.
pub const MPEG4_VOS: u8 = 0xb0;

/// Byte offsets of every start code prefix (`00 00 01`) in `data`, each with
/// the code value that follows it.
pub fn start_codes(data: &[u8]) -> Vec<(usize, u8)> {
    let mut out = Vec::new();
    let mut i = 0usize;
    while i + 3 < data.len() {
        if data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1 {
            out.push((i, data[i + 3]));
            i += 3;
        } else {
            i += 1;
        }
    }
    out
}

/// Whether an MPEG-4 start code value begins a video object layer
/// (`0x20..=0x2f`).
pub fn is_mpeg4_vol(code: u8) -> bool {
    (0x20..=0x2f).contains(&code)
}

/// The units before the first picture of an MPEG-1 / MPEG-2 stream — the
/// sequence header and its extensions, without the GOP header — or `None`
/// when `data` does not open with a sequence header.
pub fn mpeg2_config(data: &[u8]) -> Option<&[u8]> {
    let codes = start_codes(data);
    let (start, first) = *codes.first()?;
    if first != MPEG2_SEQUENCE_HEADER {
        return None;
    }
    let end = codes
        .iter()
        .find(|(_, c)| matches!(*c, MPEG2_PICTURE | MPEG2_GOP))
        .map_or(data.len(), |(o, _)| *o);
    Some(&data[start..end])
}

/// The units before the first VOP (or GOV) of an MPEG-4 Part 2 stream — the
/// visual object sequence, visual object and video object layer — or `None`
/// when they contain no video object layer.
pub fn mpeg4_config(data: &[u8]) -> Option<&[u8]> {
    let codes = start_codes(data);
    let (start, _) = *codes.first()?;
    let end = codes
        .iter()
        .find(|(_, c)| matches!(*c, MPEG4_VOP | MPEG4_GOV))
        .map_or(data.len(), |(o, _)| *o);
    codes.iter().any(|(o, c)| *o < end && is_mpeg4_vol(*c)).then(|| &data[start..end])
}

/// Whether `data` carries an MPEG-4 video object layer header in band.
pub fn has_mpeg4_vol(data: &[u8]) -> bool {
    start_codes(data).iter().any(|(_, c)| is_mpeg4_vol(*c))
}

/// Whether `data` carries an MPEG-1 / MPEG-2 sequence header.
pub fn has_mpeg2_sequence_header(data: &[u8]) -> bool {
    start_codes(data).iter().any(|(_, c)| *c == MPEG2_SEQUENCE_HEADER)
}

/// MPEG-1 / MPEG-2 `picture_coding_type` of each picture in `data`, in
/// order: 1 I, 2 P, 3 B, 4 D (MPEG-1).
pub fn mpeg2_picture_types(data: &[u8]) -> Vec<u8> {
    start_codes(data)
        .into_iter()
        .filter(|(_, c)| *c == MPEG2_PICTURE)
        // temporal_reference (10 bits), then picture_coding_type (3 bits):
        // the second byte after the code holds its top bits.
        .filter_map(|(o, _)| data.get(o + 5).map(|b| (b >> 3) & 0x7))
        .collect()
}

/// MPEG-4 `vop_coding_type` of each VOP in `data`, in order: 0 I, 1 P, 2 B,
/// 3 S.
pub fn mpeg4_vop_types(data: &[u8]) -> Vec<u8> {
    start_codes(data)
        .into_iter()
        .filter(|(_, c)| *c == MPEG4_VOP)
        .filter_map(|(o, _)| data.get(o + 4).map(|b| b >> 6))
        .collect()
}

/// Split `data` into pieces, each opening at one of `starts` (byte offsets of
/// start codes, ascending): bytes before the first start belong to the first
/// piece.
pub fn split_at_starts(data: &[u8], starts: &[usize]) -> Vec<Vec<u8>> {
    let mut out = Vec::with_capacity(starts.len());
    for (i, &s) in starts.iter().enumerate() {
        let begin = if i == 0 { 0 } else { s };
        let end = starts.get(i + 1).copied().unwrap_or(data.len());
        out.push(data[begin..end].to_vec());
    }
    out
}

/// One MPEG-1 / MPEG-2 access unit per picture: each piece holds the headers
/// before its picture (sequence header, GOP) and the picture's slices.
pub fn split_mpeg2_pictures(data: &[u8]) -> Vec<Vec<u8>> {
    let codes = start_codes(data);
    let mut starts = Vec::new();
    // A picture's access unit opens at the first header unit that leads to
    // it: a sequence header or GOP header directly before it, else the
    // picture start code itself.
    let mut lead: Option<usize> = None;
    for (o, c) in codes {
        match c {
            MPEG2_SEQUENCE_HEADER | MPEG2_GOP => {
                lead.get_or_insert(o);
            }
            MPEG2_PICTURE => starts.push(lead.take().unwrap_or(o)),
            _ => {}
        }
    }
    if starts.is_empty() {
        return Vec::new();
    }
    split_at_starts(data, &starts)
}

/// One MPEG-4 access unit per VOP: each piece holds the headers before its
/// VOP (configuration, GOV) and the VOP.
pub fn split_mpeg4_vops(data: &[u8]) -> Vec<Vec<u8>> {
    let codes = start_codes(data);
    let mut starts = Vec::new();
    let mut lead: Option<usize> = None;
    for (o, c) in codes {
        if c == MPEG4_VOP {
            starts.push(lead.take().unwrap_or(o));
        } else if c != 0xb2 {
            // Every header but user data (0xb2, which follows its owner) opens
            // the next VOP's unit.
            lead.get_or_insert(o);
        }
    }
    if starts.is_empty() {
        return Vec::new();
    }
    split_at_starts(data, &starts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mpeg2_pictures_keep_their_headers() {
        // seq hdr, ext, GOP, I picture, slice, P picture, slice.
        let mut s = vec![0, 0, 1, 0xb3, 9, 9, 0, 0, 1, 0xb5, 7, 0, 0, 1, 0xb8, 1];
        s.extend([0, 0, 1, 0x00, 0x00, 0x08, 0xff, 0, 0, 1, 0x01, 0xaa]);
        let p_at = s.len();
        s.extend([0, 0, 1, 0x00, 0x00, 0x50, 0xff, 0, 0, 1, 0x01, 0xbb]);
        let units = split_mpeg2_pictures(&s);
        assert_eq!(units.len(), 2);
        assert_eq!(units[0], s[..p_at]);
        assert_eq!(mpeg2_picture_types(&s), vec![1, 2]);
        assert_eq!(mpeg2_config(&s).unwrap(), &s[..11]);
    }

    #[test]
    fn mpeg4_config_is_everything_before_the_first_vop() {
        let mut s = vec![0, 0, 1, 0xb0, 1, 0, 0, 1, 0xb5, 9, 0, 0, 1, 0x00, 0, 0, 1, 0x20, 5, 6];
        let cfg_len = s.len();
        s.extend([0, 0, 1, 0xb6, 0x10, 0xff]);
        let second = s.len();
        s.extend([0, 0, 1, 0xb6, 0x50, 0xff]);
        assert_eq!(mpeg4_config(&s).unwrap(), &s[..cfg_len]);
        assert!(has_mpeg4_vol(&s));
        assert_eq!(mpeg4_vop_types(&s), vec![0, 1]);
        let units = split_mpeg4_vops(&s);
        assert_eq!(units.len(), 2);
        assert_eq!(units[1], s[second..]);
        assert!(mpeg4_config(&s[cfg_len..]).is_none());
    }
}
