/// Demux dispatch + shared types + box-walking primitives.
///
/// The full demux implementation is split across concern-scoped submodules:
///   - `mp4`  — ISOBMFF / MP4 / MOV demux, fragmented MP4, streaming init
///   - `mkv`  — Matroska / WebM demux, Colour mapping, EBML scanner, streaming init
///   - `audio` — audio track extraction for all containers (AAC, Opus, AC-3, …)
///   - `hdr`  — HDR static metadata (`mdcv`/`clli`) pulled from visual sample entries
///   - `tests` — unit tests (compiled only under `#[cfg(test)]`)
use anyhow::{bail, Result};
use codec::frame::StreamInfo;

use crate::avi::demux_avi;
use crate::ts::demux_ts;

pub mod mp4;
pub mod mkv;
pub(crate) mod audio;
pub(crate) mod hdr;

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

pub(crate) fn detect_container(data: &[u8]) -> &'static str {
    if data.len() < 12 {
        return "unknown";
    }
    // ISOBMFF: MP4 (`ftyp mp41`/`mp42`/`isom`/...) and MOV (`ftyp qt  `)
    // both land here. Older MOV files sometimes ship without a top-level
    // `ftyp` and lead with `moov` or `mdat` directly — accept those too.
    if &data[4..8] == b"ftyp" || &data[4..8] == b"moov" || &data[4..8] == b"mdat" {
        return "mp4";
    }
    // Matroska/WebM: EBML signature.
    if data[0] == 0x1A && data[1] == 0x45 && data[2] == 0xDF && data[3] == 0xA3 {
        return "mkv";
    }
    // RIFF-based AVI: "RIFF" <size> "AVI ".
    if &data[..4] == b"RIFF" && &data[8..12] == b"AVI " {
        return "avi";
    }
    // MPEG-TS: 0x47 sync byte at offset 0 AND at offset 188 (and 376 if
    // we have the bytes). A single 0x47 appears routinely in random
    // payloads, so require two confirming hits before committing.
    if data[0] == 0x47
        && data.len() > 188
        && data[188] == 0x47
        && (data.len() <= 376 || data[376] == 0x47)
    {
        return "ts";
    }
    "unknown"
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
