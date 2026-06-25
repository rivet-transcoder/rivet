//! Pull-based streaming demuxer (Squad streaming-migration-55 P1).
//!
//! Replaces the materialize-everything-upfront `demux()` shape with a
//! `next_video_sample()` iterator. Each per-format implementation
//! holds only the reader state it needs to produce ONE sample at a
//! time; nothing accumulates across samples. The legacy `demux()` is
//! preserved as a thin adapter that drains the iterator into a `Vec`
//! so existing callers keep working unchanged.
//!
//! Memory characteristic: peak heap from any one `next_video_sample()`
//! call is bounded by the sample size + the reader's internal cursor
//! state (mp4 0.14 keeps stbl indexes in the `Mp4Reader`; matroska-
//! demuxer keeps its own cluster cursor; the TS / AVI walks track
//! only an offset). Audio passthrough remains buffered per the
//! pinned contract — Squad-18's pattern is unchanged.

use anyhow::{Result, bail};
use codec::frame::StreamInfo;

use crate::avi::demux_avi_streaming_init;
use crate::demux::{AudioTrack, demux_mkv_streaming_init, demux_mp4_streaming_init};
use crate::ts::demux_ts_streaming_init;

/// Header information for a demuxed stream — codec label + the
/// `StreamInfo` shape every existing caller already consumes.
/// Available immediately after `demux_streaming()` returns; parsed
/// from the container header before any video samples are pulled.
#[derive(Debug, Clone)]
pub struct DemuxHeader {
    pub codec: String,
    pub info: StreamInfo,
}

/// One demuxed video sample with its container-level timing.
///
/// `data` is the codec-native bitstream for the sample — Annex-B for
/// AVC/HEVC (after AVCC→Annex-B conversion + Squad-14 parameter-set
/// tracking), raw OBU stream for AV1, IVF/raw frame for VP8/VP9,
/// self-contained frame for ProRes.
///
/// `pts_ticks` is in the container's native timescale (mp4 mvhd
/// timescale, MKV TimecodeScale-derived, TS 90 kHz, AVI samples-since-
/// start). The pipeline today does NOT consume per-sample PTS for
/// decode (decoders pull frames at their own cadence) — it's surfaced
/// for the muxer/QA bench to attribute durations.
///
/// `duration_ticks` defaults to 0 when the container does not record a
/// per-sample duration (TS PES, AVI movi walk). Callers should fall
/// back to `1 / frame_rate` from the header in that case.
#[derive(Debug, Clone)]
pub struct Sample {
    pub data: Vec<u8>,
    pub pts_ticks: i64,
    pub duration_ticks: u32,
}

/// Pull-based per-format demuxer. The trait is `Send` so the pipeline
/// can move the demuxer onto its dedicated decode thread (the existing
/// transcode pump pattern).
pub trait StreamingDemuxer: Send {
    /// Header info parsed from the container header. Cheap to call —
    /// returns a borrow of the cached `DemuxHeader` populated at
    /// construction time.
    fn header(&self) -> &DemuxHeader;

    /// Pull the next video sample. Returns `Ok(None)` at EOF.
    /// Allocates a fresh `Vec` per sample; nothing is retained
    /// internally beyond the reader's per-format cursor state.
    fn next_video_sample(&mut self) -> Result<Option<Sample>>;

    /// Audio is a single buffered slab populated at construction time
    /// (Squad-18/23/27 passthrough pattern). Streaming audio is out of
    /// scope for this sprint per the pinned design.
    fn audio(&self) -> Option<&AudioTrack>;
}

/// Magic-byte detect the container and dispatch to a per-format
/// streaming reader. Mirrors `demux::detect_container` exactly so the
/// streaming and legacy paths agree on every input.
pub fn demux_streaming(data: &[u8]) -> Result<Box<dyn StreamingDemuxer>> {
    match detect_container(data) {
        "mp4" => Ok(Box::new(demux_mp4_streaming_init(data)?)),
        "mkv" => Ok(Box::new(demux_mkv_streaming_init(data)?)),
        "avi" => Ok(Box::new(demux_avi_streaming_init(data)?)),
        "ts" => Ok(Box::new(demux_ts_streaming_init(data)?)),
        other => bail!("unsupported container: {other}"),
    }
}

/// Container magic-byte detector. Kept module-private + duplicated
/// from `demux::detect_container` so the streaming dispatch doesn't
/// reach into `demux::`'s private surface and so a future change to
/// either path stays a one-file edit.
fn detect_container(data: &[u8]) -> &'static str {
    if data.len() < 12 {
        return "unknown";
    }
    if &data[4..8] == b"ftyp" || &data[4..8] == b"moov" || &data[4..8] == b"mdat" {
        return "mp4";
    }
    if data[0] == 0x1A && data[1] == 0x45 && data[2] == 0xDF && data[3] == 0xA3 {
        return "mkv";
    }
    if &data[..4] == b"RIFF" && &data[8..12] == b"AVI " {
        return "avi";
    }
    if data[0] == 0x47
        && data.len() > 188
        && data[188] == 0x47
        && (data.len() <= 376 || data[376] == 0x47)
    {
        return "ts";
    }
    "unknown"
}
