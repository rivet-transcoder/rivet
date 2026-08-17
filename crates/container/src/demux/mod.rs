/// Demux dispatch + shared types + box-walking primitives.
///
/// The full demux implementation is split across concern-scoped submodules:
///   - `mp4`  — ISOBMFF / MP4 / MOV demux, fragmented MP4, streaming init
///   - `mkv`  — Matroska / WebM demux, Colour mapping, EBML scanner, streaming init
///   - `audio` — audio track extraction for all containers (AAC, Opus, AC-3, …)
///   - `hdr`  — HDR static metadata (`mdcv`/`clli`) pulled from visual sample entries
///   - `tests` — unit tests (compiled only under `#[cfg(test)]`)
use anyhow::{bail, Result};
use frame::StreamInfo;

use crate::avi::demux_avi;
use crate::ts::demux_ts;

pub mod mp4;
pub mod mkv;
pub(crate) mod audio;
pub(crate) mod hdr;
pub mod subtitle;

#[cfg(test)]
mod tests;

// Re-export every item that was `pub` on the old flat `demux` module so
// all existing `use crate::demux::…` call-sites remain valid.
// Public surface (matches the original flat module's `pub` items).
pub use mp4::{demux_mp4, Mp4StreamingDemuxer};
pub use mkv::{demux_mkv, probe_mkv_color_info, MkvStreamingDemuxer};
// Crate-internal entry points for the streaming dispatcher.
pub(crate) use mkv::demux_mkv_streaming_init;
pub(crate) use mp4::demux_mp4_streaming_init;
// The remaining helpers (has_av01_sample_entry, prores_sample_entry_fourcc,
// parse_avcc_param_sets, FragSample, mkv_codec_needs_annexb, extract_*_audio,
// {ac3,eac3}_sample_rate_channels_*) were private in the original flat module
// and stay internal — siblings reach them via `super::<sub>::`.

// ---------------------------------------------------------------------------
// Public shared types
// ---------------------------------------------------------------------------

pub struct DemuxResult {
    pub codec: String,
    pub info: StreamInfo,
    pub samples: Vec<Vec<u8>>,
    /// Optional audio track carried through for passthrough muxing. Populated
    /// when the input has an AAC track (MP4: `mp4a` sample entry; MKV codec
    /// id `A_AAC`). Other audio codecs log a warning and are dropped.
    pub audio: Option<AudioTrack>,
}

/// Audio track extracted for passthrough or transcode. Supports two codec
/// families today (Squad-18 + Squad-23):
/// - **AAC-LC**: `codec = "aac"`, `asc` holds the verbatim
///   AudioSpecificConfig bytes sourced from the MP4 esds descriptor (not
///   the mp4 crate's rebuilt form) or MKV `CodecPrivate`, so HE-AAC /
///   xHE-AAC signaling survives the copy. `codec_private` is empty.
/// - **Opus**: `codec = "opus"`, `codec_private` holds the RFC 7845 §5.1
///   `OpusHead` body verbatim — for MKV/WebM that's exactly the
///   `CodecPrivate` element bytes (post-magic — RFC 7845 §5.2 specifies
///   no magic prefix for the MKV CodecPrivate); for MP4-Opus that's the
///   `dOps` body re-serialised in OpusHead's LE numeric convention. `asc`
///   is empty.
///
/// `samples` are codec-native packets (AAC: ADTS-stripped raw access
/// units; Opus: TOC-prefixed Opus packets, one per frame). `durations`
/// are per-sample in `timescale` units.
#[derive(Debug, Clone)]
pub struct AudioTrack {
    pub codec: String,
    pub samples: Vec<Vec<u8>>,
    pub sample_rate: u32,
    pub channels: u16,
    /// AAC-only: AudioSpecificConfig bytes. Empty for non-AAC codecs.
    pub asc: Vec<u8>,
    /// Opus-only: OpusHead body bytes (RFC 7845 §5.1). Empty for non-Opus
    /// codecs. The 8-byte 'OpusHead' magic prefix is NOT included — only
    /// the post-magic body.
    pub codec_private: Vec<u8>,
    pub timescale: u32,
    pub durations: Vec<u32>,
}

// ---------------------------------------------------------------------------
// Public dispatch entry point
// ---------------------------------------------------------------------------

/// Dispatch to the right demuxer based on container magic bytes.
pub fn demux(data: &[u8]) -> Result<DemuxResult> {
    match detect_container(data) {
        // MOV shares its demuxer with MP4 — same ISOBMFF box tree, same
        // sample-entry structure. `detect_container` returns "mp4" for
        // both `ftyp mp4*` and `ftyp qt  ` / bare-moov MOVs.
        "mp4" => demux_mp4(data),
        "mkv" => demux_mkv(data),
        "avi" => demux_avi(data),
        "ts" => demux_ts(data),
        other => bail!("unsupported container: {other}"),
    }
}

/// [`crate::sniff_container`]'s label — one detector for every dispatch.
pub(crate) fn detect_container(data: &[u8]) -> &'static str {
    crate::sniff::sniff_container(data).label()
}

// ---------------------------------------------------------------------------
// Shared box-walking primitives (used by mp4.rs, hdr.rs, audio.rs)
// ---------------------------------------------------------------------------

/// Follow a box type path from `data` (top level) down and return the body
/// bytes (payload, excluding the 8-byte box header) of the last box in the
/// path, or None if any hop is missing. Handles 32-bit box sizes only —
/// adequate for moov/trak/stsd which are ~KB in practice.
pub(super) fn find_box_body<'a>(data: &'a [u8], path: &[&[u8; 4]]) -> Option<&'a [u8]> {
    let mut slice = data;
    for (i, target) in path.iter().enumerate() {
        let found = find_direct_child(slice, target)?;
        if i + 1 == path.len() {
            return Some(found);
        }
        slice = found;
    }
    None
}

/// The rotation the source declares, in degrees clockwise: 0, 90, 180 or 270.
///
/// # Why a transcode has to look at this
///
/// `tkhd` carries a 3x3 transformation matrix, and a player applies it before
/// showing a frame. A recorder mounted upside down writes ordinary top-down
/// pixels and a 180-degree matrix, and every player shows it the right way up.
/// A transcode that decodes the pixels and ignores the matrix re-encodes the
/// picture as stored and drops the matrix on the floor, so the output is
/// upright in the file and upside down on screen.
///
/// Found on a 289 MB `nvr1` recording whose rungs all came out inverted:
/// `a=-1.0, d=-1.0`, which is exactly 180 degrees.
///
/// Only the four right angles are recognised. The matrix can express shears
/// and arbitrary rotations, and a ladder cannot honour those without resampling
/// every frame through a general transform — so anything that is not a right
/// angle reads as 0 and is left alone, which is what happened before this
/// existed.
pub fn video_rotation_degrees(data: &[u8]) -> u32 {
    let Some(moov) = find_direct_child(data, b"moov") else { return 0 };

    for trak in direct_children(moov, b"trak") {
        // Video track only: an audio track's matrix is meaningless, and a
        // `tmcd` timecode track carries one that is not about pixels.
        let is_video = find_box_body(trak, &[b"mdia", b"hdlr"])
            .is_some_and(|hdlr| hdlr.len() >= 12 && &hdlr[8..12] == b"vide");
        if !is_video {
            continue;
        }

        let Some(tkhd) = find_direct_child(trak, b"tkhd") else { continue };
        if tkhd.is_empty() {
            continue;
        }

        // Version decides the width of the times that precede the matrix.
        // v0: 4 vf + 4 + 4 + 4 id + 4 resv + 4 dur; v1 widens the three times
        // to 8 bytes each. Then 8 reserved, layer, alternate_group, volume and
        // one reserved u16 before the matrix itself.
        let offset = if tkhd[0] == 1 { 4 + 8 + 8 + 4 + 4 + 8 } else { 4 + 4 + 4 + 4 + 4 + 4 };
        let offset = offset + 8 + 2 + 2 + 2 + 2;
        let Some(matrix) = tkhd.get(offset..offset + 36) else { continue };

        let fixed = |i: usize| -> i32 {
            i32::from_be_bytes([matrix[i], matrix[i + 1], matrix[i + 2], matrix[i + 3]])
        };
        // Only a, b, c, d matter; the third column is the perspective part and
        // is 0,0,1 for every rotation.
        const ONE: i32 = 65536;
        let (a, b, c, d) = (fixed(0), fixed(4), fixed(12), fixed(16));

        return match (a, b, c, d) {
            (ONE, 0, 0, ONE) => 0,
            (0, ONE, x, 0) if x == -ONE => 90,
            (x, 0, 0, y) if x == -ONE && y == -ONE => 180,
            (0, x, ONE, 0) if x == -ONE => 270,
            _ => 0,
        };
    }

    0
}

/// The `stsd` body of the **video** track.
///
/// # Why this is not `find_box_body(data, [moov, trak, ..., stsd])`
///
/// That path takes the *first* `trak`, and nothing requires the first track to
/// be the video one. Files that put audio first are ordinary — plenty of phone
/// and editor output does — and on those the whole codec-detection chain reads
/// the audio track's sample entry and concludes the video is something it has
/// never heard of.
///
/// Two production failures, one cause: an iPhone HEVC upload reported
/// `no decoder available for codec 'unknown'` (it read `mp4a` where it expected
/// a video fourcc) and another reported `avcc not found` (it looked for an
/// `avcC` box inside an audio sample entry). Neither file was malformed.
///
/// Falls back to the first `trak` when no handler says `vide`, which keeps
/// single-track files and anything with a missing or unusual `hdlr` working
/// exactly as before.
pub(super) fn find_video_stsd(data: &[u8]) -> Option<&[u8]> {
    let moov = find_direct_child(data, b"moov")?;

    let mut first_stsd = None;
    for trak in direct_children(moov, b"trak") {
        let Some(stsd) = find_box_body(trak, &[b"mdia", b"minf", b"stbl", b"stsd"]) else {
            continue;
        };
        if first_stsd.is_none() {
            first_stsd = Some(stsd);
        }

        // `hdlr`: 4 bytes version+flags, 4 pre_defined, then the handler type.
        let Some(hdlr) = find_box_body(trak, &[b"mdia", b"hdlr"]) else { continue };
        if hdlr.len() >= 12 && &hdlr[8..12] == b"vide" {
            return Some(stsd);
        }
    }

    first_stsd
}

/// Every direct child of `data` with the given type, in file order.
pub(super) fn direct_children<'a>(
    data: &'a [u8],
    target: &'a [u8; 4],
) -> impl Iterator<Item = &'a [u8]> + 'a {
    let mut pos = 0usize;
    std::iter::from_fn(move || {
        while pos + 8 <= data.len() {
            let size =
                u32::from_be_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]])
                    as usize;
            if size < 8 || pos.checked_add(size).is_none_or(|end| end > data.len()) {
                return None;
            }
            let btype = &data[pos + 4..pos + 8];
            let body = &data[pos + 8..pos + size];
            pos += size;
            if btype == target {
                return Some(body);
            }
        }
        None
    })
}

pub(super) fn find_direct_child<'a>(data: &'a [u8], target: &[u8; 4]) -> Option<&'a [u8]> {
    let mut pos = 0;
    while pos + 8 <= data.len() {
        let size =
            u32::from_be_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        let btype = &data[pos + 4..pos + 8];
        if size < 8 || pos.checked_add(size).is_none_or(|end| end > data.len()) {
            return None;
        }
        if btype == target {
            return Some(&data[pos + 8..pos + size]);
        }
        pos += size;
    }
    None
}
