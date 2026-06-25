//! Minimal AVI (RIFF) demuxer.
//!
//! Scope: read the video stream out of a one-video-track AVI file, map
//! the stream's handler / fourcc to one of the codec labels the
//! transcoder knows how to decode, and emit per-frame samples in the
//! order the file lays them down (presentation order — AVI does not
//! have B-frame reordering at the container layer, stream samples are
//! already display-order). Secondary audio tracks are dropped with a
//! warning; the caller already handles that shape for MP4/MKV.
//!
//! OpenDML 1.0 super-indexes (Squad-38, 2026-04-17): files >1 GiB use
//! multiple `LIST movi` chunks (one per ~1 GiB RIFF segment) plus an
//! `indx` superindex chunk per stream that points at per-LIST `ix##`
//! standard indexes. Detection happens at construction: presence of an
//! `indx` chunk inside the video stream's `LIST strl` triggers the
//! OpenDML path, which precomputes a sample chunk-offset list from the
//! `indx` → `ix##` chain and `next_video_sample()` consumes that. When
//! `indx` is absent we fall back to the legacy single-`movi` cursor walk.
//! `dmlh.dwTotalFrames` from the `LIST odml` (sibling of `strl` LISTs in
//! `hdrl`) supersedes `avih.dwTotalFrames` for OpenDML files because
//! `avih` is a 32-bit field and gets truncated for clips longer than
//! `2^32 / fps` frames.
//!
//! What's intentionally not supported:
//! - Audio passthrough. AVI audio is usually MP3 or AC-3 anyway, not
//!   AAC — outside the passthrough scope.
//! - Variable-bitrate index reconstruction. We trust the sample order
//!   in the `movi` LIST itself; `idx1` is only used as a fallback when
//!   `movi` is missing (which real-world files don't exhibit).

use anyhow::{Context, Result, bail};
use codec::frame::{ColorSpace, PixelFormat, StreamInfo};

use crate::demux::{AudioTrack, DemuxResult};
use crate::streaming::{DemuxHeader, Sample, StreamingDemuxer};

pub(crate) fn demux_avi(data: &[u8]) -> Result<DemuxResult> {
    // RIFF header: "RIFF" u32-LE size "AVI ".  (Guaranteed present by
    // the dispatch in `demux::detect_container`, but re-validate in
    // case this is ever called directly.)
    if data.len() < 12 || &data[..4] != b"RIFF" || &data[8..12] != b"AVI " {
        bail!("not a RIFF/AVI file");
    }

    // The RIFF payload begins after the 12-byte header. From there we
    // see a sequence of LIST/chunk records and optionally further
    // top-level RIFF chunks (OpenDML 1.0 — multi-`movi` files split
    // every ~1 GiB into a fresh `RIFF AVIX` segment that itself
    // contains a `LIST movi`). The records we care about:
    //   LIST hdrl  -> avih + (LIST strl { strh + strf [+ indx] })*
    //                 + LIST odml { dmlh }
    //   LIST movi  -> stream sample chunks (##dc / ##db / ##wb)
    // For OpenDML the file is `RIFF AVI ` ... `RIFF AVIX` ... `RIFF AVIX` ...
    // We scan the entire file for every `LIST movi` regardless of
    // which RIFF segment it lives in.
    let mut hdrl: Option<(usize, usize)> = None;
    let mut movi_lists: Vec<(usize, usize)> = Vec::new();
    scan_top_level_records(data, &mut hdrl, &mut movi_lists);

    let (hdrl_start, hdrl_end) = hdrl.context("AVI: missing hdrl LIST")?;
    if movi_lists.is_empty() {
        bail!("AVI: missing movi LIST");
    }

    // Walk hdrl looking for the video stream's strl LIST. strl contains
    // strh (stream header: type, handler fourcc) and strf (stream
    // format: BITMAPINFOHEADER for video).
    let video = find_video_stream(&data[hdrl_start..hdrl_end])
        .context("AVI: no video stream found in hdrl")?;

    let codec = fourcc_to_codec(&video.handler)
        .or_else(|| fourcc_to_codec(&video.compression))
        .with_context(|| {
            format!(
                "AVI: unsupported video fourcc {:?}/{:?}",
                ascii(&video.handler),
                ascii(&video.compression)
            )
        })?;

    // Stream-id prefix for this video stream's sample chunks in movi.
    // Two ASCII digits (stream index, zero-padded) + 'd' for 'dc' / 'b'
    // for 'db'. E.g. stream 0 gives `00dc` (compressed DIB frame) or
    // `00db` (uncompressed DIB frame).
    let stream_idx = video.stream_index;
    let prefix = format!("{:02}", stream_idx);

    // Walk every movi LIST in file order (OpenDML splits one logical
    // movi across multiple RIFF AVIX segments). For each LIST, pull
    // every chunk whose fourcc starts with `<prefix>d` into samples in
    // order. Non-video chunks (audio `##wb`, JUNK, `rec ` LISTs for
    // OpenDML) are skipped.
    let mut samples: Vec<Vec<u8>> = Vec::new();
    for &(movi_start, movi_end) in &movi_lists {
        collect_movi_samples(&data[movi_start..movi_end], &prefix, &mut samples)?;
    }

    if samples.is_empty() {
        bail!(
            "AVI: movi LIST contained no video samples for stream {:02}",
            stream_idx
        );
    }

    // Prefer dmlh.dwTotalFrames over the materialized sample count when
    // OpenDML is present — for >1 GiB files, dmlh is the spec-mandated
    // accurate count; avih.dwTotalFrames is u32 and may have wrapped.
    // Falling back to samples.len() preserves legacy behaviour for
    // single-`movi` files without an odml LIST.
    let total_frames =
        read_dmlh_total_frames(&data[hdrl_start..hdrl_end]).unwrap_or(samples.len() as u64);
    let duration = if video.frame_rate > 0.0 {
        total_frames as f64 / video.frame_rate
    } else {
        0.0
    };

    let info = StreamInfo {
        codec: codec.clone(),
        width: video.width,
        height: video.height,
        frame_rate: video.frame_rate,
        duration,
        // AVI's BITMAPINFOHEADER does not carry a spec-grade pixel
        // format — the fourcc implies 4:2:0 for the codecs we
        // actually support (MPEG-4 Part 2 / DivX / H.264 Baseline
        // in AVI). Downstream `pixel_format::detect` can refine
        // this once a codec-level parse runs.
        pixel_format: PixelFormat::Yuv420p,
        color_space: ColorSpace::Bt709,
        color_metadata: Default::default(),
        total_frames,
        // AVI's dwRate/dwScale in strh gives a bps for audio, not
        // video. Real video bitrate requires codec-level inspection,
        // which we punt on. 0 means "unknown" to the pipeline — same
        // posture as the MKV demuxer.
        bitrate: 0,
    };

    // Refine pixel_format from the bitstream now that we have samples.
    let detected_pf = codec::pixel_format::detect(&codec, &samples);
    let info = StreamInfo {
        pixel_format: detected_pf,
        ..info
    };

    Ok(DemuxResult {
        codec,
        info,
        samples,
        audio: None,
    })
}

#[derive(Debug)]
struct VideoStream {
    stream_index: u32,
    /// fccHandler from strh — usually the compressor identifier
    /// (DIV3/DIVX/DX50/XVID for Part 2, H264/X264 for AVC, etc.).
    handler: [u8; 4],
    /// biCompression from strf's BITMAPINFOHEADER — sometimes the
    /// clearer codec tag when strh.fccHandler has been rewritten by
    /// editors to something generic like `vids` or zero.
    compression: [u8; 4],
    width: u32,
    height: u32,
    frame_rate: f64,
}

fn find_video_stream(hdrl: &[u8]) -> Option<VideoStream> {
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

fn parse_strl(strl: &[u8], stream_index: u32) -> Option<VideoStream> {
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

/// Map an AVI fourcc (handler or biCompression) to one of the codec
/// labels the decoder factory recognises. Returns None for types we
/// don't support yet — the caller bails with a specific error listing
/// both fourccs tried.
fn fourcc_to_codec(fcc: &[u8; 4]) -> Option<String> {
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

/// Walk a movi LIST body pulling out every video sample (chunks whose
/// fourcc starts with `<stream_prefix>d`). `rec ` sub-LISTs (OpenDML
/// segmentation) recurse one level. Anything else is skipped.
fn collect_movi_samples(movi: &[u8], stream_prefix: &str, out: &mut Vec<Vec<u8>>) -> Result<()> {
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

fn ascii(b: &[u8; 4]) -> String {
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
// AviStreamingDemuxer (Squad streaming-migration-55 P1)
// ---------------------------------------------------------------------------

/// Streaming AVI demuxer. Owns the input bytes and walks the `movi`
/// LIST(s) one chunk at a time. Two backends:
/// - **Legacy single-movi cursor walk** (`Backend::Cursor`): a stack of
///   (pos, end) frames over a single `LIST movi`. `rec ` sub-LISTs push
///   a new frame; we pop on EOF to resume the parent.
/// - **OpenDML index walk** (`Backend::OpenDml`): a precomputed list of
///   `(absolute byte offset, size)` sample chunks assembled from the
///   stream's `indx` superindex + each `ix##` sub-index. `next_video_sample`
///   advances `cursor` and reads `data[offset..offset+size]`.
/// The streaming impl never holds more than the current sample's bytes
/// regardless of backend.
pub struct AviStreamingDemuxer {
    data: Vec<u8>,
    header: DemuxHeader,
    backend: Backend,
    /// Two-character stream prefix derived from the video stream's
    /// index. e.g. stream 0 → "00". Only used by the cursor backend.
    prefix: [u8; 2],
    /// Frame index — used as a synthetic monotonic PTS in samples-since-
    /// start. AVI doesn't carry per-sample PTS at the container layer.
    next_idx: u64,
    /// Lazily set on first sample: `pixel_format::detect` is one-shot
    /// against the first sample, so we patch `header.info.pixel_format`
    /// in place once and skip the probe thereafter.
    pixel_format_detected: bool,
}

enum Backend {
    /// Walk one or more `LIST movi` records linearly. The Vec is
    /// initialised with one entry per top-level movi LIST in file
    /// order; `rec ` sub-LISTs push additional frames during walk and
    /// pop at EOF. We always operate on the LAST entry (top of stack).
    Cursor(Vec<(usize, usize)>),
    /// Precomputed (absolute_offset_of_chunk_data, data_size) list
    /// drawn from the indx → ix## chain. `cursor` indexes into it.
    OpenDml {
        samples: Vec<(usize, usize)>,
        cursor: usize,
    },
}

pub(crate) fn demux_avi_streaming_init(data: &[u8]) -> Result<AviStreamingDemuxer> {
    if data.len() < 12 || &data[..4] != b"RIFF" || &data[8..12] != b"AVI " {
        bail!("not a RIFF/AVI file");
    }
    let owned = data.to_vec();

    let mut hdrl: Option<(usize, usize)> = None;
    let mut movi_lists: Vec<(usize, usize)> = Vec::new();
    scan_top_level_records(&owned, &mut hdrl, &mut movi_lists);

    let (hdrl_start, hdrl_end) = hdrl.context("AVI: missing hdrl LIST")?;
    if movi_lists.is_empty() {
        bail!("AVI: missing movi LIST");
    }

    let video = find_video_stream(&owned[hdrl_start..hdrl_end])
        .context("AVI: no video stream found in hdrl")?;
    let codec = fourcc_to_codec(&video.handler)
        .or_else(|| fourcc_to_codec(&video.compression))
        .with_context(|| {
            format!(
                "AVI: unsupported video fourcc {:?}/{:?}",
                ascii(&video.handler),
                ascii(&video.compression)
            )
        })?;

    let stream_idx = video.stream_index;
    let prefix_str = format!("{:02}", stream_idx);
    let prefix_bytes = prefix_str.as_bytes();
    if prefix_bytes.len() != 2 {
        bail!("AVI: stream index out of range");
    }
    let prefix = [prefix_bytes[0], prefix_bytes[1]];

    // OpenDML detection: look for an `indx` superindex inside the
    // chosen stream's `LIST strl`. Presence triggers the ix##-walking
    // backend; absence falls back to the legacy cursor walk over each
    // `LIST movi` LIST in order.
    let backend =
        if let Some(ix_refs) = locate_stream_indx(&owned[hdrl_start..hdrl_end], stream_idx) {
            // Each `qwOffset` in ix_refs is an absolute file offset to an
            // `ix##` chunk's 8-byte header. Parse each in turn and append
            // its sample chunks to one big list, in superindex order.
            let mut samples: Vec<(usize, usize)> = Vec::new();
            for (ix_off, ix_size) in ix_refs {
                parse_ix_chunk(&owned, ix_off, ix_size, &prefix, &mut samples);
            }
            Backend::OpenDml { samples, cursor: 0 }
        } else {
            Backend::Cursor(movi_lists)
        };

    // total_frames priority for the OpenDML era:
    //   1. `dmlh.dwTotalFrames` inside `LIST hdrl > LIST odml > dmlh`
    //      — the spec-mandated 32-bit count for files that may have
    //      wrapped `avih.dwTotalFrames` (>1 GiB / very long clips).
    //   2. `avih.dwTotalFrames` for legacy single-RIFF files.
    //   3. 0 — same "unknown" sentinel as TS (pipeline tolerates).
    let total_frames = read_dmlh_total_frames(&owned[hdrl_start..hdrl_end])
        .or_else(|| read_avih_total_frames(&owned[hdrl_start..hdrl_end]))
        .unwrap_or(0);
    // Derive duration from total_frames + frame_rate when both are
    // populated — saves the legacy `samples.len() as f64 / frame_rate`
    // computation that needed the materialized Vec.
    let duration = if total_frames > 0 && video.frame_rate > 0.0 {
        total_frames as f64 / video.frame_rate
    } else {
        0.0
    };

    let info = StreamInfo {
        codec: codec.clone(),
        width: video.width,
        height: video.height,
        frame_rate: video.frame_rate,
        duration,
        pixel_format: PixelFormat::Yuv420p,
        color_space: ColorSpace::Bt709,
        color_metadata: Default::default(),
        total_frames,
        bitrate: 0,
    };

    Ok(AviStreamingDemuxer {
        data: owned,
        header: DemuxHeader { codec, info },
        backend,
        prefix,
        next_idx: 0,
        pixel_format_detected: false,
    })
}

/// Read `dwTotalFrames` from the MainAVIHeader (`avih`) chunk that lives
/// as the first chunk inside `LIST hdrl`. AVIMAINHEADER body layout
/// (Microsoft AVI RIFF reference):
///   u32 dwMicroSecPerFrame      // 0..4
///   u32 dwMaxBytesPerSec        // 4..8
///   u32 dwPaddingGranularity    // 8..12
///   u32 dwFlags                 // 12..16
///   u32 dwTotalFrames           // 16..20  ← what we want
///   ...
/// Returns `None` if the avih chunk is missing, shorter than 20 bytes,
/// or the field is zero (some encoders leave it unset; the pipeline
/// then falls back to "unknown" same as TS).
fn read_avih_total_frames(hdrl: &[u8]) -> Option<u64> {
    let mut pos = 0;
    while pos + 8 <= hdrl.len() {
        let fcc = &hdrl[pos..pos + 4];
        let size = u32::from_le_bytes([hdrl[pos + 4], hdrl[pos + 5], hdrl[pos + 6], hdrl[pos + 7]])
            as usize;
        let body_start = pos + 8;
        let body_end = body_start + size;
        if body_end > hdrl.len() {
            return None;
        }
        if fcc == b"avih" {
            if size < 20 {
                return None;
            }
            let body = &hdrl[body_start..body_end];
            let total = u32::from_le_bytes([body[16], body[17], body[18], body[19]]);
            return if total > 0 { Some(total as u64) } else { None };
        }
        pos = body_end + (body_end & 1);
    }
    None
}

/// Read `dwTotalFrames` from the OpenDML extension header chunk
/// (`dmlh`) which lives inside `LIST hdrl > LIST odml > dmlh`. For
/// >1 GiB / very long files the spec recommends using this in
/// preference to `avih.dwTotalFrames` because that field is u32 and
/// can wrap. `dmlh.dwTotalFrames` is the first (and for our purposes
/// only) field of the dmlh body. Returns None if the chunk is absent
/// or the field is zero.
fn read_dmlh_total_frames(hdrl: &[u8]) -> Option<u64> {
    let mut pos = 0;
    while pos + 8 <= hdrl.len() {
        let fcc = &hdrl[pos..pos + 4];
        let size = u32::from_le_bytes([hdrl[pos + 4], hdrl[pos + 5], hdrl[pos + 6], hdrl[pos + 7]])
            as usize;
        let body_start = pos + 8;
        let body_end = body_start + size;
        if body_end > hdrl.len() {
            return None;
        }
        if fcc == b"LIST" && size >= 4 && &hdrl[body_start..body_start + 4] == b"odml" {
            // Walk the odml LIST body looking for dmlh.
            let mut p = body_start + 4;
            while p + 8 <= body_end {
                let f = &hdrl[p..p + 4];
                let s = u32::from_le_bytes([hdrl[p + 4], hdrl[p + 5], hdrl[p + 6], hdrl[p + 7]])
                    as usize;
                let bs = p + 8;
                let be = bs + s;
                if be > body_end {
                    return None;
                }
                if f == b"dmlh" && s >= 4 {
                    let total =
                        u32::from_le_bytes([hdrl[bs], hdrl[bs + 1], hdrl[bs + 2], hdrl[bs + 3]]);
                    return if total > 0 { Some(total as u64) } else { None };
                }
                p = be + (be & 1);
            }
            return None;
        }
        pos = body_end + (body_end & 1);
    }
    None
}

/// Top-level scanner: walks the file picking out `LIST hdrl` (always
/// in the primary `RIFF AVI ` segment) and every `LIST movi` (which
/// in OpenDML files is split across the primary `RIFF AVI ` and one
/// or more `RIFF AVIX` continuation segments at file top level). The
/// hdrl record is single-occurrence; `movi_lists` accumulates in file
/// order so the caller can walk segments left-to-right.
fn scan_top_level_records(
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
fn scan_riff_segment(
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
fn classify_list(
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

/// Locate the `indx` superindex chunk inside the `LIST strl` for the
/// given video stream, and return a list of `(absolute file offset of
/// each ix## chunk header, ix## chunk body size)` references parsed
/// from its AVI_INDEX_OF_INDEXES entries. Returns None when:
/// - the strl LIST doesn't carry an indx (legacy single-`movi` file),
/// - the indx is the rare AVI_INDEX_OF_CHUNKS form (treated like a
///   fancy idx1; the cursor walk handles those files correctly), or
/// - the indx is malformed.
///
/// `indx` chunk body layout (24-byte header + N×16-byte entries for
/// AVI_INDEX_OF_INDEXES, per OpenDML 1.02 §3.7):
///   wLongsPerEntry     u16   // 4 for index-of-indexes
///   bIndexSubType      u8    // 0 for index-of-indexes
///   bIndexType         u8    // 0x00 = AVI_INDEX_OF_INDEXES
///   nEntriesInUse      u32
///   dwChunkId          u32   // fcc the entries refer to (e.g. "00dc")
///   dwReserved[3]      u32×3 // zero
///   then per entry:
///     qwOffset         u64   // absolute file offset of an ix## chunk
///     dwSize           u32   // ix## chunk size (excluding 8-byte hdr)
///     dwDuration       u32   // sample duration covered by this ix##
fn locate_stream_indx(hdrl: &[u8], target_stream_idx: u32) -> Option<Vec<(usize, usize)>> {
    let mut stream_idx: u32 = 0;
    let mut pos = 0;
    while pos + 8 <= hdrl.len() {
        let fcc = &hdrl[pos..pos + 4];
        let size = u32::from_le_bytes([hdrl[pos + 4], hdrl[pos + 5], hdrl[pos + 6], hdrl[pos + 7]])
            as usize;
        let body_start = pos + 8;
        let body_end = body_start + size;
        if body_end > hdrl.len() {
            return None;
        }
        if fcc == b"LIST" && size >= 4 && &hdrl[body_start..body_start + 4] == b"strl" {
            if stream_idx == target_stream_idx {
                return parse_indx_in_strl(&hdrl[body_start + 4..body_end]);
            }
            stream_idx += 1;
        }
        pos = body_end + (body_end & 1);
    }
    None
}

fn parse_indx_in_strl(strl: &[u8]) -> Option<Vec<(usize, usize)>> {
    let mut pos = 0;
    while pos + 8 <= strl.len() {
        let fcc = &strl[pos..pos + 4];
        let size = u32::from_le_bytes([strl[pos + 4], strl[pos + 5], strl[pos + 6], strl[pos + 7]])
            as usize;
        let body_start = pos + 8;
        let body_end = body_start + size;
        if body_end > strl.len() {
            return None;
        }
        if fcc == b"indx" {
            return parse_indx_body(&strl[body_start..body_end]);
        }
        pos = body_end + (body_end & 1);
    }
    None
}

fn parse_indx_body(body: &[u8]) -> Option<Vec<(usize, usize)>> {
    if body.len() < 24 {
        return None;
    }
    let longs_per_entry = u16::from_le_bytes([body[0], body[1]]);
    let _index_sub_type = body[2];
    let index_type = body[3];
    let n_entries = u32::from_le_bytes([body[4], body[5], body[6], body[7]]) as usize;
    // Index-of-chunks form (rare) is handled by the cursor backend.
    if index_type != 0x00 {
        return None;
    }
    if longs_per_entry != 4 {
        return None;
    } // 4 longs = 16 bytes per entry
    let entries_start = 24;
    let mut refs = Vec::with_capacity(n_entries);
    for i in 0..n_entries {
        let off = entries_start + i * 16;
        if off + 16 > body.len() {
            break;
        }
        let qw_offset = u64::from_le_bytes([
            body[off],
            body[off + 1],
            body[off + 2],
            body[off + 3],
            body[off + 4],
            body[off + 5],
            body[off + 6],
            body[off + 7],
        ]);
        let dw_size =
            u32::from_le_bytes([body[off + 8], body[off + 9], body[off + 10], body[off + 11]]);
        // _dw_duration at off+12..off+16 is informational; not needed
        // to walk the chunks.
        let off_us = qw_offset as usize;
        refs.push((off_us, dw_size as usize));
    }
    Some(refs)
}

/// Parse an `ix##` chunk at the given absolute file offset and append
/// each sample chunk's `(absolute payload offset, payload size)` to
/// `out`. Only entries whose chunk fourcc starts with `<prefix>` are
/// kept (filters out the rare case of a multi-stream ix## merged into
/// one file area).
///
/// `ix##` chunk body layout (24-byte header + N×8-byte entries for
/// AVI_INDEX_OF_CHUNKS, per OpenDML 1.02 §3.7):
///   wLongsPerEntry     u16   // 2 for index-of-chunks
///   bIndexSubType      u8    // 0 (frame index)
///   bIndexType         u8    // 0x01 = AVI_INDEX_OF_CHUNKS
///   nEntriesInUse      u32
///   dwChunkId          u32   // fcc the entries reference
///   qwBaseOffset       u64   // entries' dwOffset is relative to this
///   dwReserved         u32   // zero
///   then per entry:
///     dwOffset         u32   // chunk DATA offset from qwBaseOffset
///     dwSize           u32   // chunk DATA size; high bit = NOT keyframe
///
/// Note `dwOffset` points at the chunk DATA, NOT the chunk header
/// (FOURCC + size) per the OpenDML 1.02 conformance spec — the
/// reasoning being that the indx/ix## pre-locates the data so a player
/// can jump directly without re-reading the chunk header. We honor
/// that: the absolute offset we record is `qwBaseOffset + dwOffset`,
/// and `size` is the data-only payload size.
fn parse_ix_chunk(
    data: &[u8],
    ix_header_off: usize,
    _ix_size: usize,
    prefix: &[u8; 2],
    out: &mut Vec<(usize, usize)>,
) {
    // The ix_header_off given by the indx superindex points at the
    // ix## chunk's 8-byte RIFF header (FOURCC + LE size). Skip the
    // header then read the body.
    if ix_header_off + 8 > data.len() {
        return;
    }
    let body_start = ix_header_off + 8;
    let body_size = u32::from_le_bytes([
        data[ix_header_off + 4],
        data[ix_header_off + 5],
        data[ix_header_off + 6],
        data[ix_header_off + 7],
    ]) as usize;
    let body_end = body_start.saturating_add(body_size).min(data.len());
    if body_end < body_start + 24 {
        return;
    }
    let body = &data[body_start..body_end];
    let longs_per_entry = u16::from_le_bytes([body[0], body[1]]);
    let _index_sub_type = body[2];
    let index_type = body[3];
    let n_entries = u32::from_le_bytes([body[4], body[5], body[6], body[7]]) as usize;
    let chunk_id: [u8; 4] = body[8..12].try_into().unwrap();
    let qw_base_offset = u64::from_le_bytes([
        body[12], body[13], body[14], body[15], body[16], body[17], body[18], body[19],
    ]) as usize;
    // body[20..24] is dwReserved (zero).
    if index_type != 0x01 {
        return;
    } // not an index-of-chunks
    if longs_per_entry != 2 {
        return;
    }
    // Only this stream's chunks (`<prefix>dc` / `<prefix>db`).
    if chunk_id[0] != prefix[0] || chunk_id[1] != prefix[1] {
        return;
    }
    let kind = chunk_id[3];
    if kind != b'c' && kind != b'b' {
        return;
    }
    let entries_start = 24;
    for i in 0..n_entries {
        let off = entries_start + i * 8;
        if off + 8 > body.len() {
            break;
        }
        let dw_offset =
            u32::from_le_bytes([body[off], body[off + 1], body[off + 2], body[off + 3]]) as usize;
        let dw_size_raw =
            u32::from_le_bytes([body[off + 4], body[off + 5], body[off + 6], body[off + 7]]);
        // High bit = "this is NOT a keyframe" flag; mask off to get the
        // real payload size. (Not used here — we don't track keyframes
        // at the demux layer for AVI; the codec parses I/P/B itself.)
        let dw_size = (dw_size_raw & 0x7FFFFFFF) as usize;
        let abs_off = qw_base_offset.saturating_add(dw_offset);
        out.push((abs_off, dw_size));
    }
}

impl StreamingDemuxer for AviStreamingDemuxer {
    fn header(&self) -> &DemuxHeader {
        &self.header
    }

    fn next_video_sample(&mut self) -> Result<Option<Sample>> {
        let payload_range = match &mut self.backend {
            Backend::OpenDml { samples, cursor } => {
                loop {
                    if *cursor >= samples.len() {
                        return Ok(None);
                    }
                    let (off, size) = samples[*cursor];
                    *cursor += 1;
                    let end = off
                        .checked_add(size)
                        .ok_or_else(|| anyhow::anyhow!("AVI: ix## entry overflows usize"))?;
                    if end > self.data.len() {
                        // Truncated tail — skip rather than bail; matches
                        // the cursor-walk's "stop on EOF" posture.
                        continue;
                    }
                    break Some((off, end));
                }
            }
            Backend::Cursor(walk) => {
                loop {
                    // Pop empty frames off the walk stack.
                    while let Some(&(pos, end)) = walk.last() {
                        if pos + 8 <= end {
                            break;
                        }
                        walk.pop();
                    }
                    let Some(&mut (ref mut pos, end)) = walk.last_mut() else {
                        return Ok(None);
                    };

                    let fcc: [u8; 4] = self.data[*pos..*pos + 4].try_into()?;
                    let size = u32::from_le_bytes([
                        self.data[*pos + 4],
                        self.data[*pos + 5],
                        self.data[*pos + 6],
                        self.data[*pos + 7],
                    ]) as usize;
                    let payload_start = *pos + 8;
                    let payload_end = payload_start + size;
                    if payload_end > end || payload_end > self.data.len() {
                        // Truncated — pop this frame and resume parent.
                        walk.pop();
                        continue;
                    }

                    // Advance past this chunk on the cursor for the NEXT call.
                    *pos = payload_end + (payload_end & 1);

                    if &fcc == b"LIST" && payload_start + 4 <= payload_end {
                        let list_type: [u8; 4] =
                            self.data[payload_start..payload_start + 4].try_into()?;
                        if &list_type == b"rec " {
                            // Push the inner walk frame and recurse.
                            walk.push((payload_start + 4, payload_end));
                            continue;
                        }
                        continue; // unknown LIST — skip
                    }

                    if fcc[0] != self.prefix[0] || fcc[1] != self.prefix[1] {
                        continue; // wrong stream
                    }
                    let kind = fcc[3];
                    if kind != b'c' && kind != b'b' {
                        continue; // not a video sample chunk
                    }
                    break Some((payload_start, payload_end));
                }
            }
        };
        let Some((start, end)) = payload_range else {
            return Ok(None);
        };

        let pts_ticks = self.next_idx as i64;
        self.next_idx += 1;
        let data = self.data[start..end].to_vec();
        if !self.pixel_format_detected {
            let detected =
                codec::pixel_format::detect(&self.header.codec, std::slice::from_ref(&data));
            self.header.info.pixel_format = detected;
            self.pixel_format_detected = true;
        }
        Ok(Some(Sample {
            data,
            pts_ticks,
            duration_ticks: 0,
        }))
    }

    fn audio(&self) -> Option<&AudioTrack> {
        // AVI audio passthrough is not supported (the legacy path also
        // returns audio: None) — out of scope for this sprint.
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal RIFF chunk: little-endian 4-byte size header.
    fn chunk(fourcc: &[u8; 4], payload: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(8 + payload.len());
        out.extend_from_slice(fourcc);
        out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        out.extend_from_slice(payload);
        if out.len() & 1 == 1 {
            out.push(0);
        } // word-align
        out
    }

    /// Wrap a payload as `LIST <type> <payload>`.
    fn list(list_type: &[u8; 4], payload: &[u8]) -> Vec<u8> {
        let mut body = Vec::with_capacity(4 + payload.len());
        body.extend_from_slice(list_type);
        body.extend_from_slice(payload);
        chunk(b"LIST", &body)
    }

    /// Emit a strh + strf pair for one video stream using a given fcc.
    fn video_strl(
        handler: &[u8; 4],
        compression: &[u8; 4],
        w: u32,
        h: u32,
        rate: u32,
        scale: u32,
    ) -> Vec<u8> {
        let mut strh = Vec::with_capacity(56);
        strh.extend_from_slice(b"vids");
        strh.extend_from_slice(handler);
        strh.extend_from_slice(&[0u8; 12]); // flags/priority/lang/initial
        strh.extend_from_slice(&scale.to_le_bytes());
        strh.extend_from_slice(&rate.to_le_bytes());
        strh.extend_from_slice(&[0u8; 24]); // start/length/buf/quality/samplesize/rect
        let strh_chunk = chunk(b"strh", &strh);

        let mut strf = Vec::with_capacity(40);
        strf.extend_from_slice(&40u32.to_le_bytes()); // biSize
        strf.extend_from_slice(&(w as i32).to_le_bytes()); // biWidth
        strf.extend_from_slice(&(h as i32).to_le_bytes()); // biHeight
        strf.extend_from_slice(&1u16.to_le_bytes()); // biPlanes
        strf.extend_from_slice(&24u16.to_le_bytes()); // biBitCount
        strf.extend_from_slice(compression); // biCompression
        strf.extend_from_slice(&[0u8; 20]); // remaining BIH fields
        let strf_chunk = chunk(b"strf", &strf);

        let mut strl_body = Vec::new();
        strl_body.extend_from_slice(&strh_chunk);
        strl_body.extend_from_slice(&strf_chunk);
        list(b"strl", &strl_body)
    }

    #[test]
    fn demux_minimal_xvid_avi_emits_samples() {
        // hdrl LIST: dummy avih + one video strl with XVID fourcc.
        let mut hdrl_body = Vec::new();
        hdrl_body.extend_from_slice(&chunk(b"avih", &[0u8; 56])); // MainAVIHeader
        hdrl_body.extend_from_slice(&video_strl(b"XVID", b"XVID", 320, 240, 30, 1));
        let hdrl = list(b"hdrl", &hdrl_body);

        // movi LIST: three compressed DIB samples (00dc) of distinct payloads.
        let mut movi_body = Vec::new();
        movi_body.extend_from_slice(&chunk(b"00dc", b"frame-1-bytes"));
        movi_body.extend_from_slice(&chunk(b"01wb", b"audio-ignored"));
        movi_body.extend_from_slice(&chunk(b"00dc", b"frame-2"));
        movi_body.extend_from_slice(&chunk(b"00dc", b"frame-3-payload"));
        let movi = list(b"movi", &movi_body);

        // Outer RIFF.
        let mut riff_body = Vec::new();
        riff_body.extend_from_slice(b"AVI ");
        riff_body.extend_from_slice(&hdrl);
        riff_body.extend_from_slice(&movi);

        let mut file = Vec::with_capacity(8 + riff_body.len());
        file.extend_from_slice(b"RIFF");
        file.extend_from_slice(&(riff_body.len() as u32).to_le_bytes());
        file.extend_from_slice(&riff_body);

        let d = demux_avi(&file).expect("demux");
        assert_eq!(d.codec, "mpeg4");
        assert_eq!(d.info.width, 320);
        assert_eq!(d.info.height, 240);
        assert_eq!(d.samples.len(), 3);
        assert_eq!(d.samples[0], b"frame-1-bytes");
        assert_eq!(d.samples[1], b"frame-2");
        assert_eq!(d.samples[2], b"frame-3-payload");
    }

    #[test]
    fn demux_rejects_unknown_fourcc() {
        let mut hdrl_body = Vec::new();
        hdrl_body.extend_from_slice(&chunk(b"avih", &[0u8; 56]));
        hdrl_body.extend_from_slice(&video_strl(b"ZZZZ", b"ZZZZ", 100, 100, 30, 1));
        let hdrl = list(b"hdrl", &hdrl_body);
        let movi = list(b"movi", &chunk(b"00dc", b"x"));
        let mut body = Vec::new();
        body.extend_from_slice(b"AVI ");
        body.extend_from_slice(&hdrl);
        body.extend_from_slice(&movi);
        let mut file = Vec::new();
        file.extend_from_slice(b"RIFF");
        file.extend_from_slice(&(body.len() as u32).to_le_bytes());
        file.extend_from_slice(&body);
        assert!(demux_avi(&file).is_err());
    }

    #[test]
    fn demux_handles_divx_variants() {
        for fcc in [b"DIVX", b"DX50", b"DIV3", b"XviD"] {
            let mut hdrl_body = Vec::new();
            hdrl_body.extend_from_slice(&chunk(b"avih", &[0u8; 56]));
            hdrl_body.extend_from_slice(&video_strl(fcc, fcc, 640, 480, 25, 1));
            let hdrl = list(b"hdrl", &hdrl_body);
            let movi = list(b"movi", &chunk(b"00dc", b"sample"));
            let mut body = Vec::new();
            body.extend_from_slice(b"AVI ");
            body.extend_from_slice(&hdrl);
            body.extend_from_slice(&movi);
            let mut file = Vec::new();
            file.extend_from_slice(b"RIFF");
            file.extend_from_slice(&(body.len() as u32).to_le_bytes());
            file.extend_from_slice(&body);
            let d = demux_avi(&file).expect("should demux");
            assert_eq!(d.codec, "mpeg4", "fourcc {:?} did not map to mpeg4", fcc);
        }
    }

    // ----- OpenDML 1.0 super-index fixture tests (Squad-38) -----

    /// Build a synthetic OpenDML AVI: 2 movi LISTs each with 3 video
    /// chunks (XVID), an indx super-index pointing at 2 ix00 sub-indexes,
    /// each ix00 listing the 3 chunks in its movi, and `dmlh` reporting
    /// `dwTotalFrames=6`. Returns the assembled file bytes plus the six
    /// expected sample payloads in order, so tests can assert offsets +
    /// content.
    ///
    /// Layout (sizes computed bottom-up so absolute offsets work out):
    ///   `RIFF AVI ` segment
    ///     `LIST hdrl`
    ///       `avih` (dwTotalFrames=3 — only counts the first segment;
    ///                we expect dmlh's 6 to win)
    ///       `LIST strl`
    ///         strh (XVID), strf (320×240),
    ///         indx superindex pointing at the two ix00 chunks
    ///       `LIST odml` { dmlh (dwTotalFrames=6) }
    ///     `LIST movi` { 00dc×3 }
    ///     ix00 (3 entries pointing into movi#1)
    ///   `RIFF AVIX` segment
    ///     `LIST movi` { 00dc×3 }
    ///     ix00 (3 entries pointing into movi#2)
    fn build_opendml_two_movi_six_samples() -> (Vec<u8>, Vec<Vec<u8>>) {
        // The six sample payloads — distinct so we can assert ordering.
        let payloads: Vec<Vec<u8>> = (0..6)
            .map(|i| format!("opendml-frame-{i}").into_bytes())
            .collect();

        // ----- Inner movi bodies + ix00 stub layout planning -----
        // We build movi LISTs first, then plan ix00 chunks from the
        // resulting per-chunk offsets, then assemble outer RIFF segments
        // so we know the absolute file offsets of each ix00 chunk
        // (needed for the indx superindex entries).

        // movi#1 body: three 00dc chunks with payloads 0, 1, 2.
        // We'll record (offset_into_movi_body_of_chunk_data, size) for each.
        let mut movi1_body = Vec::new();
        let mut chunk_data_offsets_in_movi1 = Vec::new();
        for i in 0..3 {
            let cur_off = movi1_body.len();
            // Chunk header is 8 bytes; data starts at cur_off + 8.
            let c = chunk(b"00dc", &payloads[i]);
            movi1_body.extend_from_slice(&c);
            chunk_data_offsets_in_movi1.push((cur_off + 8, payloads[i].len()));
        }

        // movi#2 body: three 00dc chunks with payloads 3, 4, 5.
        let mut movi2_body = Vec::new();
        let mut chunk_data_offsets_in_movi2 = Vec::new();
        for i in 3..6 {
            let cur_off = movi2_body.len();
            let c = chunk(b"00dc", &payloads[i]);
            movi2_body.extend_from_slice(&c);
            chunk_data_offsets_in_movi2.push((cur_off + 8, payloads[i].len()));
        }

        // The movi LIST wraps a 4-byte type ("movi") + body. So the
        // body starts +12 from the LIST chunk's start (+8 chunk header
        // + 4 type fourcc).
        let movi1_chunk = list(b"movi", &movi1_body);
        let movi2_chunk = list(b"movi", &movi2_body);

        // Build the two ix00 chunks. Each ix## chunk body layout:
        //   wLongsPerEntry=2 (u16), bIndexSubType=0 (u8),
        //   bIndexType=0x01 (u8), nEntriesInUse=N (u32),
        //   dwChunkId="00dc" (u32), qwBaseOffset (u64),
        //   dwReserved=0 (u32), then per-entry (dwOffset, dwSize) u32×2.
        //
        // We point qwBaseOffset at the start of the corresponding movi
        // LIST's BODY (i.e. the byte right after `movi` type fourcc).
        // dwOffset for each entry is the offset of the chunk DATA from
        // qwBaseOffset, i.e. exactly `chunk_data_offsets_in_moviX[i].0`.
        let build_ix00 = |entries: &[(usize, usize)], qw_base_offset: u64| -> Vec<u8> {
            let mut body = Vec::new();
            body.extend_from_slice(&2u16.to_le_bytes()); // wLongsPerEntry
            body.push(0); // bIndexSubType
            body.push(0x01); // bIndexType=AVI_INDEX_OF_CHUNKS
            body.extend_from_slice(&(entries.len() as u32).to_le_bytes()); // nEntriesInUse
            body.extend_from_slice(b"00dc"); // dwChunkId
            body.extend_from_slice(&qw_base_offset.to_le_bytes()); // qwBaseOffset
            body.extend_from_slice(&0u32.to_le_bytes()); // dwReserved
            for (data_off, data_size) in entries {
                body.extend_from_slice(&(*data_off as u32).to_le_bytes()); // dwOffset
                body.extend_from_slice(&(*data_size as u32).to_le_bytes()); // dwSize
            }
            chunk(b"ix00", &body)
        };

        // We need the absolute file offsets of the two movi BODIES and
        // the two ix00 CHUNK HEADERS to fill the indx superindex.
        // Layout of the outer file is:
        //   [0..8]      "RIFF" + size32 of the AVI  segment payload
        //   [8..12]     "AVI " form type
        //   [12..]      LIST hdrl ... (size depends on indx contents
        //                — chicken/egg, we resolve below)
        //               LIST movi#1 ... (movi1_chunk)
        //               ix00#1 ... (ix1)
        //   then        "RIFF" + size32 of the AVIX segment payload
        //               "AVIX" form type
        //               LIST movi#2 ... (movi2_chunk)
        //               ix00#2 ... (ix2)
        //
        // To break the cycle, build hdrl with placeholder indx values
        // first, measure the resulting byte sizes, compute final
        // offsets, then rewrite the indx body and reassemble.

        // Build hdrl first with a PLACEHOLDER indx (zeroed offsets) so
        // we know the hdrl size — which doesn't change when we patch
        // the placeholder qwOffset values (size stays constant).
        let placeholder_indx = build_indx_placeholder();
        let hdrl_with_placeholder = build_hdrl(
            &placeholder_indx,
            /*dmlh_total*/ 6,
            /*avih_total*/ 3,
        );

        // Compute the absolute offsets we need to know AHEAD of writing
        // the real indx: positions of movi#1 body, movi#2 body,
        // ix00#1 chunk header, ix00#2 chunk header.

        // Position 0 of the file = "RIFF" header start. The AVI  segment
        // body begins at byte 12 (after RIFF/size/AVI ).
        let avi_body_start = 12usize;
        let hdrl_offset = avi_body_start; // hdrl is the first record
        let hdrl_end = hdrl_offset + hdrl_with_placeholder.len();

        let movi1_offset = hdrl_end; // movi LIST chunk header start
        // movi LIST body starts at movi1_offset + 8 (LIST hdr) + 4 (type "movi") = +12
        let movi1_body_offset = movi1_offset + 12;
        let movi1_end = movi1_offset + movi1_chunk.len();

        let ix1_offset = movi1_end; // ix00 chunk header for movi#1
        // ix00 chunk size doesn't depend on placeholder vs real values —
        // build a real one with the right qwBaseOffset to measure its byte
        // length (constant for fixed entries).
        let ix1_chunk_real = build_ix00(&chunk_data_offsets_in_movi1, movi1_body_offset as u64);
        let ix1_end = ix1_offset + ix1_chunk_real.len();

        // Now the second `RIFF AVIX` segment starts.
        let avix_outer_start = ix1_end;
        // RIFF chunk header (8) + form type "AVIX" (4) = 12 bytes before body.
        let avix_body_start = avix_outer_start + 12;

        let movi2_offset = avix_body_start;
        let movi2_body_offset = movi2_offset + 12;
        let movi2_end = movi2_offset + movi2_chunk.len();

        let ix2_offset = movi2_end;
        let ix2_chunk_real = build_ix00(&chunk_data_offsets_in_movi2, movi2_body_offset as u64);

        // Real indx superindex pointing at the two ix00 chunks.
        let real_indx = build_indx_real(&[
            (
                ix1_offset as u64,
                (ix1_chunk_real.len() - 8) as u32,
                /*dur*/ 3,
            ),
            (
                ix2_offset as u64,
                (ix2_chunk_real.len() - 8) as u32,
                /*dur*/ 3,
            ),
        ]);
        // Sanity: real and placeholder indx must be byte-identical in length.
        assert_eq!(
            real_indx.len(),
            placeholder_indx.len(),
            "indx size sanity — placeholder and real must match for offsets to stay valid"
        );

        let hdrl_real = build_hdrl(&real_indx, 6, 3);
        assert_eq!(
            hdrl_real.len(),
            hdrl_with_placeholder.len(),
            "hdrl size sanity — must not depend on indx values, only sizes"
        );

        // Assemble AVI  segment body (after the RIFF "AVI " 12-byte header).
        let mut avi_seg_body = Vec::new();
        avi_seg_body.extend_from_slice(b"AVI ");
        avi_seg_body.extend_from_slice(&hdrl_real);
        avi_seg_body.extend_from_slice(&movi1_chunk);
        avi_seg_body.extend_from_slice(&ix1_chunk_real);
        // RIFF wrapper for the AVI segment.
        let mut file = Vec::new();
        file.extend_from_slice(b"RIFF");
        file.extend_from_slice(&(avi_seg_body.len() as u32).to_le_bytes());
        file.extend_from_slice(&avi_seg_body);

        // Assemble AVIX segment body.
        let mut avix_seg_body = Vec::new();
        avix_seg_body.extend_from_slice(b"AVIX");
        avix_seg_body.extend_from_slice(&movi2_chunk);
        avix_seg_body.extend_from_slice(&ix2_chunk_real);
        file.extend_from_slice(b"RIFF");
        file.extend_from_slice(&(avix_seg_body.len() as u32).to_le_bytes());
        file.extend_from_slice(&avix_seg_body);

        // Sanity: confirm the actual byte positions match what we
        // computed (catches any off-by-one in the layout planning).
        assert_eq!(
            &file[movi1_offset..movi1_offset + 4],
            b"LIST",
            "movi#1 should start with LIST at the planned offset"
        );
        assert_eq!(
            &file[movi1_body_offset - 4..movi1_body_offset],
            b"movi",
            "movi#1 type fourcc should sit just before the body"
        );
        assert_eq!(&file[ix1_offset..ix1_offset + 4], b"ix00");
        assert_eq!(&file[movi2_offset..movi2_offset + 4], b"LIST");
        assert_eq!(&file[movi2_body_offset - 4..movi2_body_offset], b"movi");
        assert_eq!(&file[ix2_offset..ix2_offset + 4], b"ix00");

        (file, payloads)
    }

    /// Build a placeholder indx chunk with the right byte size for two
    /// AVI_INDEX_OF_INDEXES entries but zeroed qwOffset / dwSize so we
    /// can measure the chunk's overall size before knowing the real
    /// offsets of the ix00 chunks it points at.
    fn build_indx_placeholder() -> Vec<u8> {
        build_indx_real(&[(0, 0, 0), (0, 0, 0)])
    }

    /// Build a real indx (AVI_INDEX_OF_INDEXES) referring to the given
    /// `(qwOffset, dwSize, dwDuration)` triples.
    fn build_indx_real(entries: &[(u64, u32, u32)]) -> Vec<u8> {
        let mut body = Vec::new();
        body.extend_from_slice(&4u16.to_le_bytes()); // wLongsPerEntry=4
        body.push(0); // bIndexSubType
        body.push(0x00); // bIndexType=AVI_INDEX_OF_INDEXES
        body.extend_from_slice(&(entries.len() as u32).to_le_bytes()); // nEntriesInUse
        body.extend_from_slice(b"00dc"); // dwChunkId
        body.extend_from_slice(&[0u8; 12]); // dwReserved[3]
        for (qw_off, dw_size, dw_duration) in entries {
            body.extend_from_slice(&qw_off.to_le_bytes());
            body.extend_from_slice(&dw_size.to_le_bytes());
            body.extend_from_slice(&dw_duration.to_le_bytes());
        }
        chunk(b"indx", &body)
    }

    /// Build hdrl LIST containing avih (dwTotalFrames=avih_total),
    /// strl with XVID strh+strf+indx, and odml LIST with dmlh
    /// (dwTotalFrames=dmlh_total).
    fn build_hdrl(indx_chunk: &[u8], dmlh_total: u32, avih_total: u32) -> Vec<u8> {
        // avih: u32 dwMicroSecPerFrame, dwMaxBytesPerSec, dwPaddingGranularity,
        // dwFlags, dwTotalFrames, then enough zeros to fill 56 bytes.
        let mut avih_body = Vec::with_capacity(56);
        avih_body.extend_from_slice(&33333u32.to_le_bytes()); // ~30 fps
        avih_body.extend_from_slice(&[0u8; 12]); // bytes/sec, padding, flags
        avih_body.extend_from_slice(&avih_total.to_le_bytes());
        avih_body.extend_from_slice(&[0u8; 32]); // initial frames + remaining 7 fields
        let avih_chunk = chunk(b"avih", &avih_body);

        // strl with XVID + indx tacked on the end (lives inside strl per
        // the OpenDML spec).
        let strh_chunk = {
            let mut strh = Vec::with_capacity(56);
            strh.extend_from_slice(b"vids");
            strh.extend_from_slice(b"XVID");
            strh.extend_from_slice(&[0u8; 12]);
            strh.extend_from_slice(&1u32.to_le_bytes()); // dwScale
            strh.extend_from_slice(&30u32.to_le_bytes()); // dwRate
            strh.extend_from_slice(&[0u8; 24]);
            chunk(b"strh", &strh)
        };
        let strf_chunk = {
            let mut strf = Vec::with_capacity(40);
            strf.extend_from_slice(&40u32.to_le_bytes());
            strf.extend_from_slice(&320i32.to_le_bytes());
            strf.extend_from_slice(&240i32.to_le_bytes());
            strf.extend_from_slice(&1u16.to_le_bytes());
            strf.extend_from_slice(&24u16.to_le_bytes());
            strf.extend_from_slice(b"XVID");
            strf.extend_from_slice(&[0u8; 20]);
            chunk(b"strf", &strf)
        };
        let mut strl_body = Vec::new();
        strl_body.extend_from_slice(&strh_chunk);
        strl_body.extend_from_slice(&strf_chunk);
        strl_body.extend_from_slice(indx_chunk);
        let strl_chunk = list(b"strl", &strl_body);

        // odml LIST: contains dmlh chunk with the total frame count.
        let dmlh_chunk = {
            let mut body = Vec::new();
            body.extend_from_slice(&dmlh_total.to_le_bytes());
            // dmlh is allowed to contain more reserved fields; we keep
            // it minimal at 4 bytes — every parser only reads the first
            // u32.
            chunk(b"dmlh", &body)
        };
        let odml_chunk = list(b"odml", &dmlh_chunk);

        let mut hdrl_body = Vec::new();
        hdrl_body.extend_from_slice(&avih_chunk);
        hdrl_body.extend_from_slice(&strl_chunk);
        hdrl_body.extend_from_slice(&odml_chunk);
        list(b"hdrl", &hdrl_body)
    }

    #[test]
    fn opendml_streaming_walks_both_movi_lists_in_order() {
        let (file, expected) = build_opendml_two_movi_six_samples();
        let mut d = demux_avi_streaming_init(&file).expect("OpenDML init");
        // dmlh.dwTotalFrames=6 should win over avih.dwTotalFrames=3.
        assert_eq!(d.header.info.total_frames, 6);
        // Drain — six samples, in superindex (file) order.
        let mut got = Vec::new();
        while let Some(s) = d.next_video_sample().expect("next") {
            got.push(s.data);
        }
        assert_eq!(
            got.len(),
            6,
            "should pull all six samples across both movi LISTs"
        );
        for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            assert_eq!(
                g, e,
                "sample {i} mismatch — OpenDML walk lost ordering or content"
            );
        }
    }

    #[test]
    fn opendml_legacy_demux_also_walks_both_movi_lists() {
        // The legacy `demux_avi` (Vec materialization path) must also
        // pick up multi-movi for the bench / fidelity tests that don't
        // use streaming.
        let (file, expected) = build_opendml_two_movi_six_samples();
        let d = demux_avi(&file).expect("legacy demux");
        assert_eq!(d.samples.len(), 6);
        for (i, (g, e)) in d.samples.iter().zip(expected.iter()).enumerate() {
            assert_eq!(g, e, "legacy sample {i} mismatch");
        }
        assert_eq!(
            d.info.total_frames, 6,
            "legacy total_frames should honor dmlh"
        );
    }

    #[test]
    fn opendml_total_frames_prefers_dmlh_over_avih() {
        let (file, _) = build_opendml_two_movi_six_samples();
        let d = demux_avi_streaming_init(&file).expect("init");
        assert_eq!(
            d.header.info.total_frames, 6,
            "dmlh.dwTotalFrames (6) must win over avih.dwTotalFrames (3)"
        );
        // Duration sanity: 6 frames / 30 fps = 0.2s (frame_rate from strh).
        assert!(
            (d.header.info.duration - 0.2).abs() < 1e-6,
            "duration = total_frames / frame_rate, got {}",
            d.header.info.duration
        );
    }

    #[test]
    fn opendml_picks_indx_path_not_cursor_walk() {
        // White-box: the demuxer's backend should be OpenDml when the
        // input has an indx superindex. Confirms the dispatch took the
        // intended path and we're not accidentally running the cursor
        // walk over both movi LISTs (which would also pass the sample-
        // count test but defeats the streaming RSS goal for >1 GiB
        // files because the cursor walk reads through every byte).
        let (file, _) = build_opendml_two_movi_six_samples();
        let d = demux_avi_streaming_init(&file).expect("init");
        assert!(
            matches!(d.backend, Backend::OpenDml { .. }),
            "fixture has indx — backend must be OpenDml"
        );
    }

    #[test]
    fn legacy_single_movi_without_indx_uses_cursor_backend() {
        // Backward-compat: a single-movi AVI without indx must still
        // work via the legacy cursor path (Squad-13's contract).
        let mut hdrl_body = Vec::new();
        hdrl_body.extend_from_slice(&chunk(b"avih", &[0u8; 56]));
        hdrl_body.extend_from_slice(&video_strl(b"XVID", b"XVID", 320, 240, 30, 1));
        let hdrl = list(b"hdrl", &hdrl_body);
        let mut movi_body = Vec::new();
        movi_body.extend_from_slice(&chunk(b"00dc", b"f0"));
        movi_body.extend_from_slice(&chunk(b"00dc", b"f1"));
        let movi = list(b"movi", &movi_body);
        let mut riff_body = Vec::new();
        riff_body.extend_from_slice(b"AVI ");
        riff_body.extend_from_slice(&hdrl);
        riff_body.extend_from_slice(&movi);
        let mut file = Vec::new();
        file.extend_from_slice(b"RIFF");
        file.extend_from_slice(&(riff_body.len() as u32).to_le_bytes());
        file.extend_from_slice(&riff_body);

        let mut d = demux_avi_streaming_init(&file).expect("init");
        assert!(
            matches!(d.backend, Backend::Cursor(_)),
            "no indx → must take cursor backend (legacy path)"
        );
        let s0 = d.next_video_sample().unwrap().unwrap();
        let s1 = d.next_video_sample().unwrap().unwrap();
        assert_eq!(s0.data, b"f0");
        assert_eq!(s1.data, b"f1");
        assert!(d.next_video_sample().unwrap().is_none());
    }

    #[test]
    fn parse_indx_body_decodes_two_index_of_indexes_entries() {
        // Direct test of the indx body parser — wire layout regression.
        let entries = [
            (0xDEAD_BEEFu64, 0x1234u32, 100u32),
            (0xCAFE_F00Du64, 0x5678u32, 200u32),
        ];
        let chunk_bytes = build_indx_real(&entries);
        // Skip the 8-byte chunk header to get the body.
        let body = &chunk_bytes[8..8 + (chunk_bytes.len() - 8 - (chunk_bytes.len() & 1))];
        let parsed = parse_indx_body(body).expect("parse");
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0], (0xDEAD_BEEFusize, 0x1234usize));
        assert_eq!(parsed[1], (0xCAFE_F00Dusize, 0x5678usize));
    }

    #[test]
    fn read_dmlh_total_frames_finds_value_inside_odml_list() {
        let dmlh_chunk = {
            let mut body = Vec::new();
            body.extend_from_slice(&42u32.to_le_bytes());
            body.extend_from_slice(&[0u8; 244]); // pad to spec's 248-byte minimum
            chunk(b"dmlh", &body)
        };
        let odml = list(b"odml", &dmlh_chunk);
        let mut hdrl_body = Vec::new();
        hdrl_body.extend_from_slice(&chunk(b"avih", &[0u8; 56]));
        hdrl_body.extend_from_slice(&odml);
        // Strip the outer LIST header — read_dmlh_total_frames takes the
        // hdrl body (starts after `hdrl` type fourcc).
        assert_eq!(read_dmlh_total_frames(&hdrl_body), Some(42));
    }

    #[test]
    fn read_dmlh_total_frames_returns_none_when_odml_absent() {
        let mut hdrl_body = Vec::new();
        hdrl_body.extend_from_slice(&chunk(b"avih", &[0u8; 56]));
        // No odml LIST → fall through to None.
        assert_eq!(read_dmlh_total_frames(&hdrl_body), None);
    }
}
