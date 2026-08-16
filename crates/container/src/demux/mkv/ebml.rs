/// Raw EBML scanner for matroska-demuxer 0.7 bug workarounds.
///
/// Exposes `scan_mkv_colour_raw` (reads MaxCLL, MaxFALL, and the three
/// buggy y-chromaticity fields straight from the byte stream) and the two
/// `pub(super)` VInt readers that `demux/tests.rs` exercises directly.

// ---------------------------------------------------------------------------
// Workaround result type
// ---------------------------------------------------------------------------

/// Fields recovered by raw EBML scanning to work around two matroska-demuxer
/// 0.7 bugs:
///   * `Colour::new` reads MaxCLL / MaxFALL from the MatrixCoefficients
///     ElementId offset (lib.rs:725..728 in matroska-demuxer-0.7.0/src/lib.rs).
///   * `MasteringMetadata::new` reads `primary_{r,g,b}_chromaticity_y` from
///     the matching X ElementId — all three y values come back holding the
///     corresponding x value.
#[derive(Default)]
pub(super) struct RawColourFix {
    pub(super) max_cll: Option<u32>,
    pub(super) max_fall: Option<u32>,
    /// Mastering display y-chromaticity recoveries — Squad-21.
    pub(super) primary_r_chromaticity_y: Option<f64>,
    pub(super) primary_g_chromaticity_y: Option<f64>,
    pub(super) primary_b_chromaticity_y: Option<f64>,
}

// ---------------------------------------------------------------------------
// Raw EBML colour scan
// ---------------------------------------------------------------------------

/// Raw-bytes EBML walk for the Colour element's MaxCLL (0x55BC),
/// MaxFALL (0x55BD), and the mastering display chromaticity_y fields
/// (0x55D2 / 0x55D4 / 0x55D6). Used exclusively as a workaround for
/// matroska-demuxer 0.7 bugs (see `RawColourFix`).
/// Returns `None` when the file is not well-formed enough to reach the
/// Colour element, or when neither bug-recovery field is present.
/// The rotation a Matroska file asks a player to apply, in degrees clockwise.
///
/// # Why this is scanned rather than read
///
/// Matroska stores rotation in `Video > Projection > ProjectionPoseRoll`, and
/// `matroska-demuxer` 0.7 exposes no accessor for `Projection` at all — the
/// same shape of gap that `scan_mkv_colour_raw` above exists for.
///
/// # Why the sign is flipped
///
/// `ProjectionPoseRoll` is a **counter-clockwise** roll about the viewing axis,
/// while `tkhd`'s matrix and this function's callers speak clockwise. A file
/// recorded on a phone held one way round therefore reports `-90` where MP4
/// would say 90. Getting this backwards rotates the picture the wrong way,
/// which looks exactly as broken as not rotating it and is harder to spot,
/// because the video is at least the right way up on one axis.
///
/// Only the four right angles are honoured, for the same reason as the MP4
/// path: anything else needs a general transform per frame, and a ladder that
/// guesses is worse than one that leaves the picture alone.
pub(super) fn scan_mkv_rotation_raw(data: &[u8]) -> Option<u32> {
    let mut cursor = 0;
    let seg_body: &[u8] = loop {
        let (el, after) = next_ebml_element(data, cursor)?;
        if el.id == 0x18538067 {
            break &data[el.body_start..el.body_start + el.body_len];
        }
        cursor = after;
    };

    let tracks = find_ebml_child(seg_body, 0x1654AE6B)?;

    let mut cur = 0;
    while cur < tracks.len() {
        let (el, after) = next_ebml_element(tracks, cur)?;
        cur = after;
        if el.id != 0xAE {
            continue;
        }

        let entry = &tracks[el.body_start..el.body_start + el.body_len];
        // TrackType (0x83) == 1 is video. A file may carry several tracks and
        // only the video one's projection describes pixels.
        let is_video = find_ebml_child(entry, 0x83)
            .and_then(read_unsigned)
            .is_some_and(|t| t == 1);
        if !is_video {
            continue;
        }

        let Some(video) = find_ebml_child(entry, 0xE0) else { continue };
        let Some(projection) = find_ebml_child(video, 0x7670) else { continue };
        let Some(roll) = find_ebml_child(projection, 0x7775).and_then(read_float) else {
            continue;
        };

        // Counter-clockwise to clockwise, normalised into 0..360.
        let clockwise = (-roll).rem_euclid(360.0);
        return Some(match clockwise.round() as i64 {
            90 => 90,
            180 => 180,
            270 => 270,
            _ => 0,
        });
    }

    None
}

pub(super) fn scan_mkv_colour_raw(data: &[u8]) -> Option<RawColourFix> {
    // Top-level: EBML header (0x1A45DFA3) then Segment (0x18538067).
    // We walk linearly until we find the Segment element and grab its
    // payload bytes — all subsequent work is inside that slice.
    let mut cursor = 0;
    let seg_body: &[u8] = loop {
        let (el, after) = next_ebml_element(data, cursor)?;
        if el.id == 0x18538067 {
            break &data[el.body_start..el.body_start + el.body_len];
        }
        cursor = after;
    };

    // Segment → Tracks (0x1654AE6B). Segment may carry many top-level
    // elements in any order — walk them until we find Tracks.
    let tracks = find_ebml_child(seg_body, 0x1654AE6B)?;
    // Tracks → TrackEntry* (0xAE). Look for the first TrackEntry whose
    // Video sub-element has a Colour; that's the path we care about.
    let mut cur = 0;
    while cur < tracks.len() {
        let (el, after) = next_ebml_element(tracks, cur)?;
        cur = after;
        if el.id != 0xAE {
            continue;
        }
        let entry = &tracks[el.body_start..el.body_start + el.body_len];
        let Some(video) = find_ebml_child(entry, 0xE0) else {
            continue;
        };
        let Some(colour) = find_ebml_child(video, 0x55B0) else {
            continue;
        };

        let mut fix = RawColourFix::default();
        let mut c = 0;
        while c < colour.len() {
            let (ce, after_ce) = match next_ebml_element(colour, c) {
                Some(v) => v,
                None => break,
            };
            c = after_ce;
            let value_bytes = &colour[ce.body_start..ce.body_start + ce.body_len];
            match ce.id {
                0x55BC => {
                    fix.max_cll = read_unsigned(value_bytes).and_then(|v| u32::try_from(v).ok());
                }
                0x55BD => {
                    fix.max_fall = read_unsigned(value_bytes).and_then(|v| u32::try_from(v).ok());
                }
                // MasteringMetadata sub-element (0x55D0). Walk its children
                // and pull the three buggy y-chromaticities so callers can
                // override the typed-accessor reads.
                0x55D0 => {
                    let md = value_bytes;
                    let mut mc = 0;
                    while mc < md.len() {
                        let (mce, after_mce) = match next_ebml_element(md, mc) {
                            Some(v) => v,
                            None => break,
                        };
                        mc = after_mce;
                        let mv = &md[mce.body_start..mce.body_start + mce.body_len];
                        match mce.id {
                            0x55D2 => fix.primary_r_chromaticity_y = read_float(mv),
                            0x55D4 => fix.primary_g_chromaticity_y = read_float(mv),
                            0x55D6 => fix.primary_b_chromaticity_y = read_float(mv),
                            _ => {}
                        }
                    }
                }
                _ => {}
            }
        }
        if fix.max_cll.is_some()
            || fix.max_fall.is_some()
            || fix.primary_r_chromaticity_y.is_some()
            || fix.primary_g_chromaticity_y.is_some()
            || fix.primary_b_chromaticity_y.is_some()
        {
            return Some(fix);
        }
    }
    None
}

// ---------------------------------------------------------------------------
// EBML walking primitives
// ---------------------------------------------------------------------------

/// Walk the direct children of `buf` (assumed to be an EBML master
/// element body, NOT starting with the master's own header) and
/// return the payload slice of the first element with id `want`.
fn find_ebml_child(buf: &[u8], want: u32) -> Option<&[u8]> {
    let mut cur = 0;
    while cur < buf.len() {
        let (el, after) = next_ebml_element(buf, cur)?;
        cur = after;
        if el.id == want {
            return Some(&buf[el.body_start..el.body_start + el.body_len]);
        }
    }
    None
}

#[derive(Debug)]
struct RawEbmlElement {
    id: u32,
    body_start: usize,
    body_len: usize,
}

/// Read a single EBML element at `off` within `buf`. Returns the
/// element descriptor plus the byte offset immediately after the
/// element (header + body). Only handles up to 4-byte IDs (all
/// Matroska elements fit) and size VInts up to 8 bytes.
fn next_ebml_element(buf: &[u8], off: usize) -> Option<(RawEbmlElement, usize)> {
    if off >= buf.len() {
        return None;
    }
    let (id, id_len) = read_id_vint(&buf[off..])?;
    let body_off = off + id_len;
    if body_off >= buf.len() {
        return None;
    }
    let (size, size_len) = read_size_vint(&buf[body_off..])?;
    let body_start = body_off + size_len;
    if body_start + size as usize > buf.len() {
        return None;
    }
    let elem = RawEbmlElement {
        id,
        body_start,
        body_len: size as usize,
    };
    Some((elem, body_start + size as usize))
}

// ---------------------------------------------------------------------------
// VInt readers — pub(super) so demux/tests.rs can reach them via
// `super::mkv::{read_id_vint, read_size_vint}` through mod.rs's re-export.
// ---------------------------------------------------------------------------

/// Read an EBML Class A/B/C/D ID (top-bit marker determines width,
/// 1..=4 bytes). Returns (raw id with marker bits preserved, byte-count).
pub(crate) fn read_id_vint(buf: &[u8]) -> Option<(u32, usize)> {
    if buf.is_empty() {
        return None;
    }
    let first = buf[0];
    let len = if first & 0x80 != 0 {
        1
    } else if first & 0x40 != 0 {
        2
    } else if first & 0x20 != 0 {
        3
    } else if first & 0x10 != 0 {
        4
    } else {
        return None;
    };
    if buf.len() < len {
        return None;
    }
    let mut id: u32 = 0;
    for b in &buf[..len] {
        id = (id << 8) | (*b as u32);
    }
    Some((id, len))
}

/// Read an EBML size VInt (1..=8 bytes). Strips the marker bit and
/// returns the numeric value plus byte-count.
pub(crate) fn read_size_vint(buf: &[u8]) -> Option<(u64, usize)> {
    if buf.is_empty() {
        return None;
    }
    let first = buf[0];
    if first == 0 {
        return None;
    }
    let len = first.leading_zeros() as usize + 1;
    if len > 8 || buf.len() < len {
        return None;
    }
    // Mask off the leading marker bit. `len == 8` (first byte 0x01) has
    // *no* value bits in the first byte — all 56 value bits live in
    // bytes 1..8. `u8 >> 8` is UB, so branch explicitly.
    let mask: u8 = if len == 8 { 0 } else { 0xFFu8 >> len };
    let mut v: u64 = (first & mask) as u64;
    for b in &buf[1..len] {
        v = (v << 8) | (*b as u64);
    }
    Some((v, len))
}

// ---------------------------------------------------------------------------
// Primitive value readers (private — used only within this file)
// ---------------------------------------------------------------------------

/// Read a big-endian unsigned integer (1..=8 bytes) from a Matroska
/// value payload. Zero-length payloads encode 0.
fn read_unsigned(buf: &[u8]) -> Option<u64> {
    if buf.len() > 8 {
        return None;
    }
    let mut v: u64 = 0;
    for b in buf {
        v = (v << 8) | (*b as u64);
    }
    Some(v)
}

/// Read a big-endian Matroska float payload — 4 bytes encode an f32,
/// 8 bytes encode an f64. Anything else is malformed.
fn read_float(buf: &[u8]) -> Option<f64> {
    match buf.len() {
        4 => {
            let arr: [u8; 4] = buf.try_into().ok()?;
            Some(f32::from_be_bytes(arr) as f64)
        }
        8 => {
            let arr: [u8; 8] = buf.try_into().ok()?;
            Some(f64::from_be_bytes(arr))
        }
        _ => None,
    }
}

#[cfg(test)]
mod rotation_tests {
    use super::*;

    /// EBML element: id bytes, then a size vint, then the body.
    fn el(id: &[u8], body: &[u8]) -> Vec<u8> {
        let mut out = id.to_vec();
        // Sizes here are small, so the one-byte vint form (0x80 | len) is
        // enough and keeps the fixtures readable.
        assert!(body.len() < 0x7F, "fixture body outgrew the one-byte size vint");
        out.push(0x80 | body.len() as u8);
        out.extend_from_slice(body);
        out
    }

    /// A whole file: EBML header, Segment > Tracks > TrackEntry > Video >
    /// Projection > ProjectionPoseRoll.
    fn file_with_roll(roll: f64, track_type: u8) -> Vec<u8> {
        let pose = el(&[0x77, 0x75], &roll.to_be_bytes());
        let projection = el(&[0x76, 0x70], &pose);
        let video = el(&[0xE0], &projection);
        let mut entry_body = el(&[0x83], &[track_type]);
        entry_body.extend_from_slice(&video);
        let entry = el(&[0xAE], &entry_body);
        let tracks = el(&[0x16, 0x54, 0xAE, 0x6B], &entry);
        let segment = el(&[0x18, 0x53, 0x80, 0x67], &tracks);

        let mut file = el(&[0x1A, 0x45, 0xDF, 0xA3], &[0u8; 4]);
        file.extend_from_slice(&segment);
        file
    }

    #[test]
    fn the_roll_is_counter_clockwise_and_comes_back_clockwise() {
        // The detail most likely to be wrong, and the one that would look like
        // a working fix: `ProjectionPoseRoll` is counter-clockwise, and every
        // caller of this speaks clockwise. Flip the sign the wrong way and a
        // 90-degree file is rotated to 270 — still "rotated", still wrong, and
        // only visibly so if somebody watches it.
        assert_eq!(scan_mkv_rotation_raw(&file_with_roll(-90.0, 1)), Some(90));
        assert_eq!(scan_mkv_rotation_raw(&file_with_roll(90.0, 1)), Some(270));
    }

    #[test]
    fn a_half_turn_is_the_same_either_way() {
        // 180 is its own mirror, so it cannot catch a sign error — which is
        // exactly why the test above exists as well as this one.
        assert_eq!(scan_mkv_rotation_raw(&file_with_roll(180.0, 1)), Some(180));
        assert_eq!(scan_mkv_rotation_raw(&file_with_roll(-180.0, 1)), Some(180));
    }

    #[test]
    fn an_audio_tracks_projection_is_ignored() {
        // TrackType 2 is audio. Its projection describes nothing about pixels,
        // and reading the first track's would be the same mistake the MP4 path
        // made with `moov/trak`.
        assert_eq!(scan_mkv_rotation_raw(&file_with_roll(-90.0, 2)), None);
    }

    #[test]
    fn an_angle_that_is_not_a_right_one_is_left_alone() {
        // A ladder cannot honour 37 degrees without resampling every frame
        // through a general transform. 0 leaves the file exactly as it behaved
        // before any of this existed.
        assert_eq!(scan_mkv_rotation_raw(&file_with_roll(-37.0, 1)), Some(0));
    }

    #[test]
    fn a_file_with_no_projection_reports_nothing() {
        // Nearly every Matroska file. It must cost nothing and claim nothing.
        let video = el(&[0xE0], &el(&[0xB0], &[0x02, 0x80]));
        let mut entry_body = el(&[0x83], &[1]);
        entry_body.extend_from_slice(&video);
        let tracks = el(&[0x16, 0x54, 0xAE, 0x6B], &el(&[0xAE], &entry_body));
        let mut file = el(&[0x1A, 0x45, 0xDF, 0xA3], &[0u8; 4]);
        file.extend_from_slice(&el(&[0x18, 0x53, 0x80, 0x67], &tracks));

        assert_eq!(scan_mkv_rotation_raw(&file), None);
    }
}
