//! RIFF/AVI structural walk: hdrl/strl parsing, FourCC→codec map, movi
//! sample collection, and top-level RIFF segment scanner.

use anyhow::{Result, bail};

// ---------------------------------------------------------------------------
// VideoStream — parsed video stream descriptor from LIST strl
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub(super) struct VideoStream {
    pub(super) stream_index: u32,
    /// fccHandler from strh — usually the compressor identifier
    /// (DIV3/DIVX/DX50/XVID for Part 2, H264/X264 for AVC, etc.).
    pub(super) handler: [u8; 4],
    /// biCompression from strf's BITMAPINFOHEADER — sometimes the
    /// clearer codec tag when strh.fccHandler has been rewritten by
    /// editors to something generic like `vids` or zero.
    pub(super) compression: [u8; 4],
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) frame_rate: f64,
}

// ---------------------------------------------------------------------------
// hdrl / strl walk
// ---------------------------------------------------------------------------

pub(super) fn find_video_stream(hdrl: &[u8]) -> Option<VideoStream> {
    // hdrl contains avih (MainAVIHeader) followed by one LIST strl per
    // stream. We're only interested in the first video stream.
    let mut pos = 0;
    let mut stream_idx: u32 = 0;
    while pos + 8 <= hdrl.len() {
        let size = u32::from_le_bytes([hdrl[pos + 4], hdrl[pos + 5], hdrl[pos + 6], hdrl[pos + 7]])
            as usize;
        let fourcc = &hdrl[pos..pos + 4];
        let payload_start = pos + 8;
        let payload_end = payload_start + size;
        if payload_end > hdrl.len() {
            break;
        }
        if fourcc == b"LIST" && payload_start + 4 <= payload_end {
            let list_type = &hdrl[payload_start..payload_start + 4];
            if list_type == b"strl" {
                let strl = &hdrl[payload_start + 4..payload_end];
                if let Some(v) = parse_strl(strl, stream_idx) {
                    return Some(v);
                }
                stream_idx += 1;
            }
        }
        pos = payload_end + (payload_end & 1);
    }
    None
}

pub(super) fn parse_strl(strl: &[u8], stream_index: u32) -> Option<VideoStream> {
    // strl contains strh + strf (+ optionally strd, JUNK). strh layout:
    //   fccType          u32 ("vids"/"auds"/...)
    //   fccHandler       u32
    //   dwFlags          u32
    //   wPriority wLang  u16 u16
    //   dwInitialFrames  u32
    //   dwScale          u32   <-- frame rate = dwRate / dwScale
    //   dwRate           u32
    //   dwStart          u32
    //   dwLength         u32
    //   dwSuggestedBufSize u32
    //   dwQuality dwSampleSize u32 u32
    //   rcFrame          i16 i16 i16 i16
    let mut strh: Option<&[u8]> = None;
    let mut strf: Option<&[u8]> = None;
    let mut pos = 0;
    while pos + 8 <= strl.len() {
        let fourcc = &strl[pos..pos + 4];
        let size = u32::from_le_bytes([strl[pos + 4], strl[pos + 5], strl[pos + 6], strl[pos + 7]])
            as usize;
        let end = pos + 8 + size;
        if end > strl.len() {
            break;
        }
        let body = &strl[pos + 8..end];
        if fourcc == b"strh" {
            strh = Some(body);
        } else if fourcc == b"strf" {
            strf = Some(body);
        }
        pos = end + (end & 1);
    }
    let strh = strh?;
    let strf = strf?;
    if strh.len() < 32 {
        return None;
    }
    let fcc_type: [u8; 4] = strh[0..4].try_into().ok()?;
    if &fcc_type != b"vids" {
        return None;
    }
    let handler: [u8; 4] = strh[4..8].try_into().ok()?;
    let scale = u32::from_le_bytes([strh[20], strh[21], strh[22], strh[23]]);
    let rate = u32::from_le_bytes([strh[24], strh[25], strh[26], strh[27]]);
    let frame_rate = if scale > 0 {
        rate as f64 / scale as f64
    } else {
        30.0
    };

    // BITMAPINFOHEADER (strf for vids):
    //   biSize, biWidth (i32), biHeight (i32), biPlanes u16, biBitCount u16,
    //   biCompression u32 (fourcc), biSizeImage, biXPelsPerMeter,
    //   biYPelsPerMeter, biClrUsed, biClrImportant
    if strf.len() < 20 {
        return None;
    }
    let width = i32::from_le_bytes([strf[4], strf[5], strf[6], strf[7]]).unsigned_abs();
    let height = i32::from_le_bytes([strf[8], strf[9], strf[10], strf[11]]).unsigned_abs();
    let compression: [u8; 4] = strf[16..20].try_into().ok()?;

    Some(VideoStream {
        stream_index,
        handler,
        compression,
        width,
        height,
        frame_rate,
    })
}

// ---------------------------------------------------------------------------
// FourCC → codec label
// ---------------------------------------------------------------------------

/// Map an AVI fourcc (handler or biCompression) to one of the codec
/// labels the decoder factory recognises. Returns None for types we
/// don't support yet — the caller bails with a specific error listing
/// both fourccs tried.
pub(super) fn fourcc_to_codec(fcc: &[u8; 4]) -> Option<String> {
    // Case-fold so "xvid"/"XVID"/"XviD" all match.
    let mut norm = [0u8; 4];
    for (i, b) in fcc.iter().enumerate() {
        norm[i] = if (b'a'..=b'z').contains(b) {
            b - 32
        } else {
            *b
        };
    }
    match &norm {
        // MPEG-4 Part 2 family (DivX / XviD and friends)
        b"DIVX" | b"DX50" | b"XVID" | b"DIV3" | b"DIV4" | b"DIV5" | b"DIV6" | b"MP4V" | b"MP4S"
        | b"M4S2" | b"FMP4" | b"DM4V" | b"3IVX" | b"3IV2" | b"XVIX" => Some("mpeg4".into()),
        // H.264 in AVI (rare but real — older GoPro / legacy pipelines).
        b"H264" | b"X264" | b"AVC1" | b"DAVC" => Some("h264".into()),
        // MPEG-2 in AVI is unusual but not impossible.
        b"MPG2" | b"MPEG" => Some("mpeg2".into()),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// movi sample collection
// ---------------------------------------------------------------------------

/// Walk a movi LIST body pulling out every video sample (chunks whose
/// fourcc starts with `<stream_prefix>d`). `rec ` sub-LISTs (OpenDML
/// segmentation) recurse one level. Anything else is skipped.
pub(super) fn collect_movi_samples(
    movi: &[u8],
    stream_prefix: &str,
    out: &mut Vec<Vec<u8>>,
) -> Result<()> {
    let prefix = stream_prefix.as_bytes();
    if prefix.len() != 2 {
        bail!("stream prefix must be 2 chars, got {:?}", stream_prefix);
    }
    let mut pos = 0;
    while pos + 8 <= movi.len() {
        let fcc = &movi[pos..pos + 4];
        let size = u32::from_le_bytes([movi[pos + 4], movi[pos + 5], movi[pos + 6], movi[pos + 7]])
            as usize;
        let payload_start = pos + 8;
        let payload_end = payload_start + size;
        if payload_end > movi.len() {
            // Truncated tail — stop here rather than bail; gives us
            // whatever samples we've already picked up.
            break;
        }
        if fcc == b"LIST" && payload_start + 4 <= payload_end {
            let list_type = &movi[payload_start..payload_start + 4];
            if list_type == b"rec " {
                collect_movi_samples(&movi[payload_start + 4..payload_end], stream_prefix, out)?;
            }
        } else if fcc.len() == 4 && fcc[0] == prefix[0] && fcc[1] == prefix[1] {
            // `##dc` = compressed DIB, `##db` = uncompressed DIB — both
            // are legitimate video sample chunks. `##dd` is an OpenDML
            // keyframe index we ignore.
            let kind = fcc[3];
            if kind == b'c' || kind == b'b' {
                out.push(movi[payload_start..payload_end].to_vec());
            }
        }
        pos = payload_end + (payload_end & 1);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// ASCII helper
// ---------------------------------------------------------------------------

pub(super) fn ascii(b: &[u8; 4]) -> String {
    b.iter()
        .map(|c| {
            if c.is_ascii_graphic() {
                *c as char
            } else {
                '.'
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Top-level RIFF / AVI / AVIX segment scanner
// ---------------------------------------------------------------------------

/// Top-level scanner: walks the file picking out `LIST hdrl` (always
/// in the primary `RIFF AVI ` segment) and every `LIST movi` (which
/// in OpenDML files is split across the primary `RIFF AVI ` and one
/// or more `RIFF AVIX` continuation segments at file top level). The
/// hdrl record is single-occurrence; `movi_lists` accumulates in file
/// order so the caller can walk segments left-to-right.
pub(super) fn scan_top_level_records(
    data: &[u8],
    hdrl: &mut Option<(usize, usize)>,
    movi_lists: &mut Vec<(usize, usize)>,
) {
    let mut pos = 0;
    while pos + 8 <= data.len() {
        let fcc = &data[pos..pos + 4];
        let size = u32::from_le_bytes([data[pos + 4], data[pos + 5], data[pos + 6], data[pos + 7]])
            as usize;
        let payload_start = pos + 8;
        let claimed_end = payload_start.saturating_add(size);
        let payload_end = claimed_end.min(data.len());

        if fcc == b"RIFF" && payload_start + 4 <= payload_end {
            // Form-type follows the size: `AVI ` for the primary
            // segment, `AVIX` for OpenDML continuation segments. Both
            // wrap a sequence of LIST/chunk records inside.
            let form: [u8; 4] = data[payload_start..payload_start + 4].try_into().unwrap();
            if &form == b"AVI " || &form == b"AVIX" {
                scan_riff_segment(data, payload_start + 4, payload_end, hdrl, movi_lists);
            }
        } else if fcc == b"LIST" && payload_start + 4 <= payload_end {
            // Some single-RIFF files surface LIST records at file top
            // level if a stray byte preceded the outer RIFF — handle
            // defensively. (Real OpenDML files always use RIFF outer.)
            classify_list(data, payload_start, payload_end, hdrl, movi_lists);
        }
        if claimed_end > data.len() {
            break;
        }
        pos = payload_end + (payload_end & 1);
    }
}

/// Walk one `RIFF AVI ` / `RIFF AVIX` segment body recording any
/// `LIST hdrl` / `LIST movi` records found inside.
pub(super) fn scan_riff_segment(
    data: &[u8],
    body_start: usize,
    body_end: usize,
    hdrl: &mut Option<(usize, usize)>,
    movi_lists: &mut Vec<(usize, usize)>,
) {
    let mut p = body_start;
    while p + 8 <= body_end {
        let fcc = &data[p..p + 4];
        let size =
            u32::from_le_bytes([data[p + 4], data[p + 5], data[p + 6], data[p + 7]]) as usize;
        let payload_start = p + 8;
        let claimed_end = payload_start.saturating_add(size);
        let payload_end = claimed_end.min(body_end);
        if fcc == b"LIST" && payload_start + 4 <= payload_end {
            classify_list(data, payload_start, payload_end, hdrl, movi_lists);
        }
        if claimed_end > body_end {
            break;
        }
        p = payload_end + (payload_end & 1);
    }
}

/// Inspect a LIST's type field (4 bytes at `payload_start`) and, if
/// it's `hdrl` or `movi`, record its body range (after the type FOURCC).
pub(super) fn classify_list(
    data: &[u8],
    payload_start: usize,
    payload_end: usize,
    hdrl: &mut Option<(usize, usize)>,
    movi_lists: &mut Vec<(usize, usize)>,
) {
    let list_type: [u8; 4] = data[payload_start..payload_start + 4].try_into().unwrap();
    match &list_type {
        b"hdrl" => {
            // Only record the FIRST hdrl seen — it's defined to be
            // unique per file, and OpenDML AVIX segments don't carry
            // their own hdrl.
            if hdrl.is_none() {
                *hdrl = Some((payload_start + 4, payload_end));
            }
        }
        b"movi" => movi_lists.push((payload_start + 4, payload_end)),
        _ => {}
    }
}
