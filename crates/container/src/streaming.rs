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
use frame::StreamInfo;

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
    /// Ticks per second for [`Sample::pts_ticks`] / [`Sample::duration_ticks`]:
    /// the video track's `mdhd` timescale for MP4, `1_000_000_000` for MKV
    /// (ticks are nanoseconds), `90_000` for TS. AVI ticks are frame indices,
    /// so this is the frame rate rounded to an integer there — pace AVI by
    /// `info.frame_rate` instead. `seconds = pts_ticks / timescale`.
    pub timescale: u32,
    /// Clockwise rotation the container asks a player to apply, in degrees:
    /// 0, 90, 180 or 270.
    ///
    /// The pixels are stored unrotated; this is the instruction that goes with
    /// them. A transcode that decodes the pixels and ignores this re-encodes
    /// the picture as stored, and the output plays upside down or on its side —
    /// correct in the file, wrong on screen. Containers with no such concept
    /// report 0.
    pub rotation_degrees: u32,
}

impl DemuxHeader {
    /// The picture's dimensions **as seen**: `info`'s width and height with the
    /// container's rotation applied, so a 90° or 270° source swaps them.
    ///
    /// `info.width`/`info.height` are the dimensions as *stored*, which is what
    /// the decoder has to be told. Everything that sizes an output from the
    /// source — a ladder, a single-file target, a thumbnail — wants these
    /// instead, or a portrait phone recording (stored landscape with a 90°
    /// matrix) gets a landscape ladder for a portrait picture and every rung
    /// is squashed onto its side.
    pub fn upright_dims(&self) -> (u32, u32) {
        if matches!(self.rotation_degrees, 90 | 270) {
            (self.info.height, self.info.width)
        } else {
            (self.info.width, self.info.height)
        }
    }

    /// `info` with [`upright_dims`](Self::upright_dims) in place of the stored
    /// dimensions — what a consumer of already-rotated frames should size by.
    pub fn upright_info(&self) -> StreamInfo {
        let (width, height) = self.upright_dims();
        StreamInfo { width, height, ..self.info.clone() }
    }

    /// [`Sample::pts_ticks`] in seconds.
    pub fn pts_seconds(&self, pts_ticks: i64) -> f64 {
        if self.timescale == 0 {
            return 0.0;
        }
        pts_ticks as f64 / self.timescale as f64
    }
}

/// One demuxed video sample with its container-level timing.
///
/// `data` is the codec-native bitstream for the sample — Annex-B for
/// AVC/HEVC (after AVCC→Annex-B conversion + Squad-14 parameter-set
/// tracking), raw OBU stream for AV1, IVF/raw frame for VP8/VP9,
/// self-contained frame for ProRes.
///
/// `pts_ticks` is in the container's native timescale — see
/// [`DemuxHeader::timescale`] (mp4 track timescale, MKV nanoseconds, TS
/// 90 kHz, AVI samples-since-start). The pipeline today does NOT consume per-sample PTS for
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

    /// Every text subtitle track the source carries that `tx3g` / WebVTT can
    /// represent, in source order. Buffered at construction like `audio`.
    ///
    /// Defaults to empty: Matroska and MP4 are the containers rivet reads
    /// text subtitles from, so the other readers inherit "no subtitles"
    /// rather than each restating it.
    fn subtitles(&self) -> &[crate::demux::subtitle::SubtitleTrack] {
        &[]
    }
}

/// Magic-byte detect the container and dispatch to a per-format
/// streaming reader. Mirrors `demux::detect_container` exactly so the
/// streaming and legacy paths agree on every input.
pub fn demux_streaming(data: &[u8]) -> Result<Box<dyn StreamingDemuxer>> {
    // Copies once, because a demuxer outlives the borrow. Callers that already
    // hold the input as `Bytes` — the job engine and the decode pump, the two
    // that run per transcode — should use [`demux_streaming_shared`] instead
    // and pay nothing.
    demux_streaming_shared(bytes::Bytes::copy_from_slice(data))
}

/// Same dispatch, but over a **shared** buffer.
///
/// Every demuxer holds the whole input for the life of the read, and a job
/// builds several of them (header probe, decode pump, one per spliced clip).
/// When each one owned a private `Vec<u8>` that meant a full copy apiece — on a
/// 9 GB Blu-ray remux the process reached 34 GB RSS and the OOM killer took it.
/// `Bytes` is refcounted, so N demuxers now cost one buffer.
pub fn demux_streaming_shared(data: bytes::Bytes) -> Result<Box<dyn StreamingDemuxer>> {
    match detect_container(&data) {
        "mp4" => Ok(Box::new(demux_mp4_streaming_init(data)?)),
        "mkv" => Ok(Box::new(demux_mkv_streaming_init(data)?)),
        "avi" => Ok(Box::new(demux_avi_streaming_init(data)?)),
        "ts" => Ok(Box::new(demux_ts_streaming_init(data)?)),
        other => bail!("unsupported container: {other}"),
    }
}

/// Container magic-byte detector — [`crate::sniff_container`], which every
/// dispatch in this crate reads, so no two of them can disagree about a file.
fn detect_container(data: &[u8]) -> &'static str {
    crate::sniff::sniff_container(data).label()
}
