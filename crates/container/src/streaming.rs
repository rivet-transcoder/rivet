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
    /// (ticks are nanoseconds), `90_000` for TS. AVI's is `strh.dwRate`, and
    /// a sample's `pts_ticks` is its chunk position × `dwScale` (empty chunks
    /// count) — pace AVI by `info.frame_rate` all the same, showing each
    /// frame for the periods [`StreamingDemuxer::frame_repeats`] gives it.
    /// `seconds = pts_ticks / timescale`.
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
    /// The shape of one stored sample, `(width, height)` in lowest terms:
    /// `(1, 1)` for square pixels, `(64, 45)` for a 16:9 PAL 720x576. From
    /// the container (`pasp`, Matroska's display size) or else the stream
    /// (SPS VUI, MPEG-2 sequence header); square when neither says. Of the
    /// picture as stored — see [`upright_sample_aspect`](Self::upright_sample_aspect).
    pub sample_aspect: (u32, u32),
}

impl DemuxHeader {
    /// [`sample_aspect`](Self::sample_aspect) as seen, after the rotation:
    /// a quarter turn swaps a sample's width and height with the picture's.
    pub fn upright_sample_aspect(&self) -> (u32, u32) {
        let (w, h) = self.sample_aspect;
        if matches!(self.rotation_degrees, 90 | 270) { (h, w) } else { (w, h) }
    }

    /// The width-over-height shape the picture is shown at: its
    /// [`upright_dims`](Self::upright_dims) with non-square samples
    /// accounted for. What an output sized "to the source's shape" keeps.
    pub fn display_aspect(&self) -> f64 {
        let (w, h) = self.upright_dims();
        let (sw, sh) = self.upright_sample_aspect();
        if h == 0 || sh == 0 {
            return 0.0;
        }
        (f64::from(w) * f64::from(sw)) / (f64::from(h) * f64::from(sh))
    }

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

    /// What the video track presents, when the container carries a
    /// presentation edit that changes anything (an MP4/MOV edit list hiding a
    /// trim's lead-in, ending early, or starting late). `None` presents every
    /// decoded frame from time zero.
    ///
    /// [`DemuxHeader::info`]'s `total_frames` and `duration` already describe
    /// the presented frames; this says which decoded frames those are. See
    /// [`crate::edit`].
    fn video_presentation(&self) -> Option<&crate::edit::VideoPresentation> {
        None
    }

    /// The audio track's presentation edit, in ticks of
    /// [`AudioTrack::timescale`] on the timeline of its sample durations, when
    /// it changes anything (encoder priming, a trim, a late start). `None`
    /// presents every sample from time zero.
    fn audio_edit(&self) -> Option<crate::edit::AudioEdit> {
        None
    }

    /// The silences inside the audio track, ascending, each already counted in
    /// the duration of the packet it follows ([`crate::edit::AudioGap`]).
    ///
    /// Defaults to none: a transport stream is the one source whose audio
    /// timing rivet reads from timestamps rather than from its packets.
    fn audio_gaps(&self) -> &[crate::edit::AudioGap] {
        &[]
    }

    /// How many frame periods of a constant-rate output each decoded frame
    /// fills, by decoded index, when the source holds some frames longer than
    /// one period: an AVI's empty video chunks are dropped frames' slots, and
    /// the frame before them is shown again for each. `None` shows every frame
    /// once. When `Some`, [`DemuxHeader::info`]'s `frame_rate` is the rate of
    /// those periods and `total_frames` their count.
    fn frame_repeats(&self) -> Option<&[u32]> {
        None
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
        "mp3" => bail!("an MP3 file has no video (audio-only output reads it: `demux_audio`)"),
        "flac" => bail!("a native FLAC stream has no video; read it with the audio-only output mode"),
        other => bail!("unsupported container: {other}"),
    }
}

/// An input read for its audio alone: the track, its presentation edit and
/// its holes, as a [`StreamingDemuxer`] reports them.
#[derive(Debug, Clone)]
pub struct AudioSource {
    pub track: AudioTrack,
    pub edit: Option<crate::edit::AudioEdit>,
    pub gaps: Vec<crate::edit::AudioGap>,
    /// Whether the input also has a video track (which is not read here).
    pub has_video: bool,
}

/// The audio of `data`, whether or not it has video: what the video demuxer
/// reads when there is a video track, else the audio-only readers — a bare
/// MP3 / MP2 file (its LAME tag's delay and padding as the edit), a native
/// FLAC stream, an MP4 /
/// M4A (its audio edit list), a Matroska / WebM. `None` when the input has
/// no audio track this crate reads.
pub fn demux_audio(data: bytes::Bytes) -> Result<Option<AudioSource>> {
    let kind = crate::sniff::sniff_container(&data);
    let video_error = match kind {
        crate::sniff::ContainerKind::Mp3 | crate::sniff::ContainerKind::Flac => None,
        _ => match demux_streaming_shared(data.clone()) {
            Ok(d) => {
                return Ok(d.audio().cloned().map(|track| AudioSource {
                    track,
                    edit: d.audio_edit(),
                    gaps: d.audio_gaps().to_vec(),
                    has_video: true,
                }));
            }
            Err(e) => Some(e),
        },
    };
    // A file whose video the demuxer refused is not an audio-only file: its
    // error stands, rather than the job quietly writing the audio alone.
    let has_video = match kind {
        crate::sniff::ContainerKind::IsoBmff => crate::demux::mp4::has_video_track(&data)?,
        crate::sniff::ContainerKind::Matroska => crate::demux::mkv::has_video_track(&data)?,
        _ => false,
    };
    if has_video && let Some(e) = video_error {
        return Err(e);
    }
    let audio_only = |track: AudioTrack, edit| Some(AudioSource { track, edit, gaps: Vec::new(), has_video: false });
    match kind {
        crate::sniff::ContainerKind::Mp3 => {
            let (track, edit) = crate::mp3::read_file(&data)?;
            Ok(audio_only(track, edit))
        }
        crate::sniff::ContainerKind::Flac => {
            Ok(audio_only(crate::demux::audio::lossless::read_native_flac(&data)?, None))
        }
        crate::sniff::ContainerKind::IsoBmff => {
            let Some(track) = crate::demux::audio::extract_mp4_audio(&data) else {
                return Ok(None);
            };
            let ids = crate::demux::mp4::audio_track_ids(&data)?;
            let edit = crate::demux::mp4::edit_list::resolve_audio_edit(&data, &ids, &track)?;
            Ok(audio_only(track, edit))
        }
        crate::sniff::ContainerKind::Matroska => {
            Ok(crate::demux::audio::extract_mkv_audio(&data).and_then(|t| audio_only(t, None)))
        }
        // A transport stream or AVI with no video: the demuxer's own error.
        _ => demux_streaming_shared(data).map(|_| None),
    }
}

/// Container magic-byte detector — [`crate::sniff_container`], which every
/// dispatch in this crate reads, so no two of them can disagree about a file.
fn detect_container(data: &[u8]) -> &'static str {
    crate::sniff::sniff_container(data).label()
}
