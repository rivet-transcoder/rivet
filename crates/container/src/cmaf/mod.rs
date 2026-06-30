//! Fragmented MP4 / CMAF box writers.
//!
//! Produces ISO/IEC 14496-12 §8.8 movie-fragment boxes (`moof` / `mfhd` /
//! `traf` / `tfhd` / `tfdt` / `trun`) and the corresponding `mvex` /
//! `mehd` / `trex` declarations that go inside a CMAF init segment's
//! `moov`. CMAF (ISO/IEC 23000-19) constrains the general 14496-12 model:
//! exactly one track per fragment (one `traf` per `moof`), exactly one
//! track per init segment, and a small set of mandatory boxes.
//!
//! This module is the box-level primitive layer. Higher-level callers
//! (`init_segment_video`, `media_segment_video`, etc. in subsequent
//! commits) compose these into init + media segments. The split lets us
//! unit-test each box's byte layout against the spec without having to
//! drive a full encode + segment pipeline.
//!
//! Spec citations are given by section number in the relevant box's doc
//! comment so future readers can cross-check against the standard.
//!
//! # CMAF brand
//!
//! Init segments for video tracks declare the `cmfc` brand (CMAF
//! constraints, per CMAF §7.3.4). Audio tracks use `cmfa`. Both brands
//! coexist in `compatible_brands` alongside the existing `iso6` / `mp42`
//! / `av01` brands so non-CMAF-aware tools that consume the same boxes
//! (e.g. an old ffprobe) can still demux them.
//!
//! # Sample-flags packing
//!
//! `default_sample_flags` (in `trex` / `tfhd`) and `first_sample_flags`
//! / per-sample flags (in `trun`) are packed per ISO/IEC 14496-12
//! §8.8.3.1. The 32 bits are laid out:
//!
//! ```text
//!   reserved[6]      = 0
//!   is_leading[2]    = 0
//!   sample_depends_on[2]
//!   sample_is_depended_on[2]
//!   sample_has_redundancy[2]
//!   sample_padding_value[3] = 0
//!   sample_is_non_sync_sample[1]
//!   sample_degradation_priority[16] = 0
//! ```
//!
//! For AV1 / AAC the meaningful values are `sample_depends_on = 1`
//! (this sample depends on others — i.e. P / B / non-IDR) or `2`
//! (independent — i.e. IDR / sync), and `sample_is_non_sync_sample = 1`
//! for non-key frames, `0` for keyframes. The helper
//! [`SampleFlags::pack`] handles this; callers shouldn't compose the
//! u32 by hand.

use anyhow::{Context, Result};
use codec::frame::{ColorMetadata, VideoCodec};
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use crate::AudioInfo;
use crate::mux::{build_avc1, build_avcc, build_hvc1, build_hvcc, extract_sequence_header};
use crate::nal_mux::{NalMuxCodec, NalSampleWriter};

mod fragment;
mod init;
#[cfg(test)]
mod tests;

pub use fragment::*;
pub use init::*;

// =====================================================================
// Shared types (re-used by fragment.rs, init.rs, and the muxers here)
// =====================================================================

/// CMAF brand identifiers used in `ftyp.compatible_brands`.
pub mod brand {
    /// CMAF video constraints brand (CMAF §7.3.4).
    pub const CMFC: &[u8; 4] = b"cmfc";
    /// CMAF audio constraints brand (CMAF §7.3.5).
    pub const CMFA: &[u8; 4] = b"cmfa";
}

/// Track type discriminator. CMAF places one track per init / fragment;
/// this enum is what higher-level orchestration uses to pick which
/// codec dispatch to take. The init / segment writers themselves don't
/// take this enum (they have type-specific entry points), so it stays
/// `#[allow(dead_code)]` until the pipeline orchestrator (Phase 4)
/// wires it through.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum CmafTrackKind {
    Video,
    Audio,
}

/// Sample flags as packed in `default_sample_flags` / `first_sample_flags` /
/// per-sample `sample_flags` in `trun`. ISO/IEC 14496-12 §8.8.3.1.
///
/// Defaults model an AV1 P-frame: depends-on=1, non-sync=1, no redundancy.
/// Override `is_sync` for IDR / key samples. The remaining fields aren't
/// meaningful for our pipeline (no DRM / leading samples / temporal layers
/// past Annex H), so they stay at their spec-default zero values.
#[derive(Debug, Clone, Copy)]
pub struct SampleFlags {
    /// `sample_is_non_sync_sample` flag. False ⇔ keyframe / IDR.
    pub is_sync: bool,
}

impl SampleFlags {
    /// Pack into the wire-format u32. See module docs for bit layout.
    pub fn pack(self) -> u32 {
        // For sync samples: sample_depends_on=2 (no other samples needed
        // to decode — i.e. independent), sample_is_non_sync_sample=0.
        // For non-sync: sample_depends_on=1 (depends on prior samples),
        // sample_is_non_sync_sample=1.
        if self.is_sync {
            // depends_on=2 in bits 24-25; is_non_sync=0 in bit 16.
            0x02_00_00_00
        } else {
            // depends_on=1 in bits 24-25; is_non_sync=1 in bit 16.
            0x01_01_00_00
        }
    }

    pub fn keyframe() -> Self {
        Self { is_sync: true }
    }
    pub fn delta_frame() -> Self {
        Self { is_sync: false }
    }
}

/// Per-sample fields written into `trun`. Each entry produces one row
/// of (duration, size, flags) in the fragment's sample table.
#[derive(Debug, Clone, Copy)]
pub struct CmafSample {
    /// Sample duration in track timescale ticks.
    pub duration: u32,
    /// Encoded sample size in bytes.
    pub size: u32,
    /// Sample flags (sync / non-sync). The very FIRST sample in a fragment
    /// uses `first_sample_flags` instead — see `build_trun_video`.
    pub flags: SampleFlags,
}

// =====================================================================
// Stateful per-rendition segmenter types
// =====================================================================

/// Per-segment metadata returned by [`CmafVideoMuxer::flush_segment`] /
/// [`CmafAudioMuxer::flush_segment`]. These records form the input to
/// the HLS playlist writer (Phase 3) and the segment-alignment validator
/// (Phase 5).
#[derive(Debug, Clone)]
pub struct SegmentInfo {
    /// 1-based monotonically increasing sequence number per track.
    pub sequence_number: u32,
    /// Path of the `seg-NNNNN.m4s` file on disk.
    pub path: PathBuf,
    /// Total file size in bytes (moof + mdat header + payload).
    pub byte_size: u64,
    /// Sum of per-sample durations in track-timescale ticks. The HLS
    /// `EXTINF` line is written from this divided by the timescale.
    pub duration_ticks: u64,
}

/// Output of a finalized track muxer: where the init segment lives,
/// the ordered list of media segments, and the timescale needed to
/// convert `duration_ticks` to seconds.
#[derive(Debug, Clone)]
pub struct CmafTrackManifest {
    pub init_path: PathBuf,
    pub segments: Vec<SegmentInfo>,
    pub timescale: u32,
}

impl CmafTrackManifest {
    /// Total duration across all segments, in seconds.
    pub fn duration_seconds(&self) -> f64 {
        let total_ticks: u64 = self.segments.iter().map(|s| s.duration_ticks).sum();
        total_ticks as f64 / self.timescale as f64
    }
}

/// One pending video sample inside the muxer's per-segment buffer.
struct PendingVideoSample {
    payload: Vec<u8>,
    duration: u32,
    is_keyframe: bool,
}

/// One pending audio sample.
struct PendingAudioSample {
    payload: Vec<u8>,
    duration: u32,
}

// =====================================================================
// CmafVideoMuxer
// =====================================================================

/// Optional construction parameters for [`CmafVideoMuxer`]. Defaults
/// match the original 5-arg `new()` behaviour: write init.mp4, start
/// segment numbering at 1, decode-time at 0.
///
/// Non-default values are used by the multi-GPU helper-task path
/// (see `pipeline::cmaf` helper variant): when multiple muxers share
/// a single per-rung output directory, each helper's muxer starts
/// at a non-1 `first_segment_index` and the corresponding decode-time
/// offset, and only the primary writes `init.mp4`.
#[derive(Debug, Clone)]
pub struct CmafVideoMuxerOptions {
    /// 1-based segment index the muxer's first `flush_segment()` will
    /// write. The output file is `seg-{first_segment_index:05}.m4s`.
    /// Defaults to `1` (the primary's first segment).
    pub first_segment_index: u32,
    /// Decode-time (in track-timescale ticks) of the muxer's first
    /// segment's first sample. Should equal
    /// `(first_segment_index - 1) * segment_duration_ticks` so that
    /// `tfdt` is byte-identical to what the primary would produce for
    /// the same segment index. Defaults to `0`.
    pub first_segment_base_decode_time: u64,
    /// When `false`, `flush_segment()` and `finalize()` skip writing
    /// `init.mp4`. Use when a sibling muxer (typically the primary)
    /// is responsible for the init segment and helpers must not race
    /// against it. Defaults to `true`.
    pub write_init_segment: bool,
}

impl Default for CmafVideoMuxerOptions {
    fn default() -> Self {
        Self {
            first_segment_index: 1,
            first_segment_base_decode_time: 0,
            write_init_segment: true,
        }
    }
}

/// Stateful CMAF video segmenter for one AV1 rendition.
///
/// Driven by the pipeline:
/// 1. Construct with rendition dimensions + output dir + timescale.
/// 2. Call `add_packet` for each encoded packet from the encoder.
///    The first packet's OBU stream MUST contain a sequence header;
///    the muxer extracts it and uses it for `av1C` in the init.mp4
///    (written lazily on the first `flush_segment` call).
/// 3. Call `flush_segment` whenever a CMAF fragment boundary is
///    reached (the orchestrator decides when based on accumulated
///    duration + the segment_duration knob).
/// 4. After the last packet is added and flushed, call `finalize`
///    to consume the muxer and get the [`CmafTrackManifest`].
///
/// Segment files are named `seg-00001.m4s`, `seg-00002.m4s`, ...
/// in the output dir.
pub struct CmafVideoMuxer {
    output_dir: PathBuf,
    width: u32,
    height: u32,
    timescale: u32,
    color_metadata: ColorMetadata,
    track_id: u32,
    /// Output codec. `Av1` stores OBUs verbatim + builds `av01`/`av1C`;
    /// `H264`/`H265` repackage Annex-B → length-prefixed via `nal_writer` and
    /// build `avc3`/`hev1` init segments with inline parameter sets.
    codec: VideoCodec,
    /// AV1 only: the OBU sequence header captured from the first packet.
    config_obus: Option<Vec<u8>>,
    /// H.264/H.265 only: Annex-B → length-prefixed repackaging + SPS/PPS(/VPS)
    /// capture (inline mode — each segment self-describes; `avc3`/`hev1`).
    nal_writer: Option<NalSampleWriter>,
    init_path: PathBuf,
    init_written: bool,
    sequence_number: u32,
    base_decode_time: u64,
    pending: Vec<PendingVideoSample>,
    segments: Vec<SegmentInfo>,
}

impl CmafVideoMuxer {
    /// Construct a new video muxer that writes init.mp4 + segments to
    /// `output_dir`. Creates the directory if it doesn't exist.
    ///
    /// Equivalent to `new_with_options(..., CmafVideoMuxerOptions::default())`.
    pub fn new(
        output_dir: impl AsRef<Path>,
        width: u32,
        height: u32,
        timescale: u32,
        color_metadata: ColorMetadata,
    ) -> Result<Self> {
        Self::new_with_options(
            output_dir,
            width,
            height,
            timescale,
            color_metadata,
            CmafVideoMuxerOptions::default(),
        )
    }

    /// Construct a muxer with non-default options. See
    /// [`CmafVideoMuxerOptions`].
    ///
    /// The helper-task path uses this to attach to an in-progress rung:
    /// the helper's muxer starts numbering segments at the helper's
    /// claim range start, advances `tfdt` to the corresponding decode
    /// time, and skips the init segment write that the primary owns.
    pub fn new_with_options(
        output_dir: impl AsRef<Path>,
        width: u32,
        height: u32,
        timescale: u32,
        color_metadata: ColorMetadata,
        options: CmafVideoMuxerOptions,
    ) -> Result<Self> {
        Self::new_with_codec_options(
            output_dir,
            width,
            height,
            timescale,
            color_metadata,
            VideoCodec::Av1,
            options,
        )
    }

    /// Codec-aware constructor. `Av1` matches the legacy behaviour; `H264` /
    /// `H265` build `avc3` / `hev1` init segments and repackage the encoder's
    /// Annex-B packets into length-prefixed samples with inline parameter sets
    /// (each segment self-describes — robust across the multi-GPU helper path).
    pub fn new_with_codec_options(
        output_dir: impl AsRef<Path>,
        width: u32,
        height: u32,
        timescale: u32,
        color_metadata: ColorMetadata,
        codec: VideoCodec,
        options: CmafVideoMuxerOptions,
    ) -> Result<Self> {
        assert!(
            options.first_segment_index >= 1,
            "first_segment_index is 1-based; got {}",
            options.first_segment_index,
        );
        let output_dir = output_dir.as_ref().to_path_buf();
        fs::create_dir_all(&output_dir)
            .with_context(|| format!("creating CMAF video output dir: {}", output_dir.display()))?;
        let init_path = output_dir.join("init.mp4");
        // H.264/H.265 use inline parameter sets (avc3/hev1) so each segment —
        // and each independently-encoded multi-GPU chunk — self-describes.
        let nal_writer = match codec {
            VideoCodec::Av1 => None,
            VideoCodec::H264 => Some(NalSampleWriter::new_inline(NalMuxCodec::H264)),
            VideoCodec::H265 => Some(NalSampleWriter::new_inline(NalMuxCodec::H265)),
        };
        Ok(Self {
            output_dir,
            width,
            height,
            timescale,
            color_metadata,
            track_id: 1,
            codec,
            config_obus: None,
            nal_writer,
            init_path,
            // When write_init_segment is false, mark init as already
            // written so `ensure_init_written` is a no-op. The primary
            // is expected to have written (or will write) init.mp4
            // separately.
            init_written: !options.write_init_segment,
            // `flush_segment` pre-increments `sequence_number` before
            // writing, so the on-disk segment number equals
            // `sequence_number` AFTER the increment. To produce
            // `seg-{first_segment_index:05}.m4s` as the first output,
            // start at `first_segment_index - 1`.
            sequence_number: options.first_segment_index - 1,
            base_decode_time: options.first_segment_base_decode_time,
            pending: Vec::new(),
            segments: Vec::new(),
        })
    }

    /// Add one encoded video packet to the current pending segment.
    /// `duration` is in track-timescale ticks. `is_keyframe` must be
    /// true for IDR / sync-sample packets — the muxer doesn't peek
    /// into the OBU stream to figure that out, and a wrong value
    /// will produce a CMAF segment that doesn't decode (the spec
    /// requires every segment to start with a sync sample).
    pub fn add_packet(&mut self, payload: Vec<u8>, duration: u32, is_keyframe: bool) -> Result<()> {
        match &mut self.nal_writer {
            None => {
                // AV1: capture the OBU sequence header once; store OBUs verbatim.
                if self.config_obus.is_none() {
                    self.config_obus = Some(extract_sequence_header(&payload).context(
                        "extracting AV1 sequence header from first packet for av1C config record",
                    )?);
                }
                self.pending.push(PendingVideoSample {
                    payload,
                    duration,
                    is_keyframe,
                });
            }
            Some(writer) => {
                // H.264/H.265: split the Annex-B packet into access units (one
                // per frame); each becomes a length-prefixed sample carrying its
                // own inline SPS/PPS. Per-AU keyframe (IDR) detection comes from
                // the bitstream, not the caller's flag. Each frame keeps the
                // full per-frame `duration` (a packet may hold several frames).
                for au in writer.push_packet(&payload) {
                    self.pending.push(PendingVideoSample {
                        payload: au.data,
                        duration,
                        is_keyframe: au.is_keyframe,
                    });
                }
            }
        }
        Ok(())
    }

    /// Whether the muxer is ready to flush a segment that starts on a
    /// sync sample. The first sample in `pending` must be a keyframe.
    /// CMAF requires every segment to begin with a sync sample
    /// (§7.3.2.1), so the orchestrator should ensure this invariant
    /// before calling `flush_segment`.
    pub fn first_pending_is_keyframe(&self) -> bool {
        self.pending.first().is_some_and(|s| s.is_keyframe)
    }

    /// Total duration of pending samples in track-timescale ticks. The
    /// orchestrator uses this to decide when a segment has reached
    /// its target duration.
    pub fn pending_duration_ticks(&self) -> u64 {
        self.pending.iter().map(|s| s.duration as u64).sum()
    }

    /// View of segments already flushed to disk. Each entry's
    /// `sequence_number` is the segment's 1-based index; `path` is
    /// the on-disk location. The helper-task path
    /// (`pipeline::cmaf::cmaf_transcode_rung_slice`) reads this
    /// between `add_packet` calls to detect "did the last add
    /// trigger an auto-flush?" — when `segments().len()` grows, the
    /// last entry is the newly-flushed segment.
    pub fn segments(&self) -> &[SegmentInfo] {
        &self.segments
    }

    /// Drop every sample currently in the pending buffer without
    /// writing them to disk. Used by the helper-task path when its
    /// claim has been shrunk by an `attach_helper` and the encoder's
    /// lookahead would otherwise produce a segment that conflicts
    /// with whichever helper now owns that range.
    ///
    /// Specifically: when a primary's claim is shrunk from `[0..N)`
    /// to `[0..K)`, the primary's encoder has already received
    /// frames `K*KI..K*KI+lookahead` by the time the claim-shrink
    /// is observed at the segment boundary. Those frames belong to
    /// the helper that took `[K..N)`. Discarding the muxer pending
    /// + dropping the encoder is the cleanest way to ensure no
    /// stale segment file is written for the helper's territory.
    pub fn clear_pending(&mut self) {
        self.pending.clear();
    }

    /// Flush pending samples to a new media segment file. Writes
    /// `init.mp4` first if it hasn't been written yet (the av1C config
    /// record needs the first packet's sequence header). Returns the
    /// segment's metadata and clears the pending buffer.
    ///
    /// No-op if `pending` is empty.
    pub fn flush_segment(&mut self) -> Result<Option<SegmentInfo>> {
        if self.pending.is_empty() {
            return Ok(None);
        }
        if !self.first_pending_is_keyframe() {
            anyhow::bail!(
                "CMAF segment must start with a sync sample; first pending sample is not a keyframe \
                 (segment_number={}, pending_count={})",
                self.sequence_number + 1,
                self.pending.len()
            );
        }
        self.ensure_init_written()?;

        self.sequence_number += 1;
        let seq = self.sequence_number;
        let samples_meta: Vec<CmafSample> = self
            .pending
            .iter()
            .map(|s| CmafSample {
                duration: s.duration,
                size: s.payload.len() as u32,
                flags: if s.is_keyframe {
                    SampleFlags::keyframe()
                } else {
                    SampleFlags::delta_frame()
                },
            })
            .collect();
        let segment_duration: u64 = samples_meta.iter().map(|s| s.duration as u64).sum();

        let mut moof = build_moof_video(seq, self.track_id, self.base_decode_time, &samples_meta);
        moof.patch_default_no_gap();

        let payload_total: u64 = self.pending.iter().map(|s| s.payload.len() as u64).sum();
        let mdat_box_size: u64 = 8 + payload_total;
        if mdat_box_size > u32::MAX as u64 {
            // Above u32::MAX we'd need a `largesize` mdat (16-byte header).
            // For 4-second segments at sane bitrates this is impossible; if
            // we ever hit it, bail with a clear error rather than silently
            // overflowing.
            anyhow::bail!(
                "CMAF media segment payload {} bytes exceeds 32-bit mdat size limit",
                payload_total
            );
        }

        let path = self.output_dir.join(format!("seg-{:05}.m4s", seq));
        let file = File::create(&path)
            .with_context(|| format!("creating CMAF segment file: {}", path.display()))?;
        let mut writer = BufWriter::new(file);
        writer.write_all(&moof.bytes).context("writing moof")?;
        writer
            .write_all(&(mdat_box_size as u32).to_be_bytes())
            .context("writing mdat size")?;
        writer.write_all(b"mdat").context("writing mdat type")?;
        for sample in &self.pending {
            writer
                .write_all(&sample.payload)
                .context("writing mdat payload")?;
        }
        writer.flush().context("flushing CMAF segment writer")?;
        let byte_size = moof.bytes.len() as u64 + mdat_box_size;

        self.base_decode_time += segment_duration;
        self.pending.clear();

        let info = SegmentInfo {
            sequence_number: seq,
            path,
            byte_size,
            duration_ticks: segment_duration,
        };
        self.segments.push(info.clone());
        Ok(Some(info))
    }

    /// Finalize the muxer: ensures the init segment is on disk (covers
    /// the edge case where add_packet was called but flush_segment
    /// never was — e.g. an empty source), drops any non-flushed
    /// pending samples (caller should have flushed them), and returns
    /// the manifest.
    pub fn finalize(mut self) -> Result<CmafTrackManifest> {
        if !self.pending.is_empty() {
            // Flush whatever's left. The caller should have done this
            // explicitly; we cover them defensively.
            self.flush_segment()?;
        }
        self.ensure_init_written()?;
        Ok(CmafTrackManifest {
            init_path: self.init_path,
            segments: self.segments,
            timescale: self.timescale,
        })
    }

    fn ensure_init_written(&mut self) -> Result<()> {
        if self.init_written {
            return Ok(());
        }
        let init = match self.codec {
            VideoCodec::Av1 => {
                let config = self.config_obus.as_ref().ok_or_else(|| {
                    anyhow::anyhow!(
                        "cannot write CMAF video init segment: no AV1 sequence header has been \
                         observed yet (call add_packet before flush_segment / finalize)"
                    )
                })?;
                build_init_segment_video(
                    self.width,
                    self.height,
                    self.timescale,
                    config,
                    &self.color_metadata,
                )
            }
            VideoCodec::H264 => {
                let w = self.nal_writer.as_ref().context("H.264 CMAF nal writer missing")?;
                if !w.has_param_sets() {
                    anyhow::bail!("cannot write CMAF H.264 init segment: no SPS/PPS observed yet");
                }
                let avcc = build_avcc(&w.sps, &w.pps);
                // avc3 sample entry (in-band parameter sets); avc1 ftyp brand.
                let entry = build_avc1(self.width, self.height, &avcc, &self.color_metadata, b"avc3");
                build_init_segment_video_with_entry(
                    self.width,
                    self.height,
                    self.timescale,
                    &entry,
                    b"avc1",
                )
            }
            VideoCodec::H265 => {
                let w = self.nal_writer.as_ref().context("H.265 CMAF nal writer missing")?;
                if !w.has_param_sets() {
                    anyhow::bail!(
                        "cannot write CMAF H.265 init segment: no VPS/SPS/PPS observed yet"
                    );
                }
                let hvcc = build_hvcc(&w.vps, &w.sps, &w.pps);
                // hev1 sample entry (in-band parameter sets); hvc1 ftyp brand.
                let entry = build_hvc1(self.width, self.height, &hvcc, &self.color_metadata, b"hev1");
                build_init_segment_video_with_entry(
                    self.width,
                    self.height,
                    self.timescale,
                    &entry,
                    b"hvc1",
                )
            }
        };
        let mut file = File::create(&self.init_path).with_context(|| {
            format!(
                "creating CMAF video init segment: {}",
                self.init_path.display()
            )
        })?;
        file.write_all(&init)
            .context("writing CMAF video init segment bytes")?;
        file.flush().context("flushing CMAF video init segment")?;
        self.init_written = true;
        Ok(())
    }
}

// =====================================================================
// CmafAudioMuxer
// =====================================================================

/// Stateful CMAF audio segmenter. Same model as the video muxer but
/// simpler — every audio sample is independently decodable, so there's
/// no first-sample-flags / sync-boundary requirement.
pub struct CmafAudioMuxer {
    output_dir: PathBuf,
    info: AudioInfo,
    track_id: u32,
    init_path: PathBuf,
    init_written: bool,
    sequence_number: u32,
    base_decode_time: u64,
    pending: Vec<PendingAudioSample>,
    segments: Vec<SegmentInfo>,
}

impl CmafAudioMuxer {
    pub fn new(output_dir: impl AsRef<Path>, info: AudioInfo) -> Result<Self> {
        let output_dir = output_dir.as_ref().to_path_buf();
        fs::create_dir_all(&output_dir)
            .with_context(|| format!("creating CMAF audio output dir: {}", output_dir.display()))?;
        let init_path = output_dir.join("init.mp4");
        Ok(Self {
            output_dir,
            info,
            track_id: 1,
            init_path,
            init_written: false,
            sequence_number: 0,
            base_decode_time: 0,
            pending: Vec::new(),
            segments: Vec::new(),
        })
    }

    pub fn add_packet(&mut self, payload: Vec<u8>, duration: u32) -> Result<()> {
        self.pending.push(PendingAudioSample { payload, duration });
        Ok(())
    }

    pub fn pending_duration_ticks(&self) -> u64 {
        self.pending.iter().map(|s| s.duration as u64).sum()
    }

    pub fn flush_segment(&mut self) -> Result<Option<SegmentInfo>> {
        if self.pending.is_empty() {
            return Ok(None);
        }
        self.ensure_init_written()?;

        self.sequence_number += 1;
        let seq = self.sequence_number;
        let samples_meta: Vec<CmafSample> = self
            .pending
            .iter()
            .map(|s| CmafSample {
                duration: s.duration,
                size: s.payload.len() as u32,
                flags: SampleFlags::keyframe(),
            })
            .collect();
        let segment_duration: u64 = samples_meta.iter().map(|s| s.duration as u64).sum();

        let mut moof = build_moof_audio(seq, self.track_id, self.base_decode_time, &samples_meta);
        moof.patch_default_no_gap();

        let payload_total: u64 = self.pending.iter().map(|s| s.payload.len() as u64).sum();
        let mdat_box_size: u64 = 8 + payload_total;
        if mdat_box_size > u32::MAX as u64 {
            anyhow::bail!(
                "CMAF audio media segment payload {} bytes exceeds 32-bit mdat size limit",
                payload_total
            );
        }

        let path = self.output_dir.join(format!("seg-{:05}.m4s", seq));
        let file = File::create(&path)
            .with_context(|| format!("creating CMAF audio segment file: {}", path.display()))?;
        let mut writer = BufWriter::new(file);
        writer
            .write_all(&moof.bytes)
            .context("writing audio moof")?;
        writer
            .write_all(&(mdat_box_size as u32).to_be_bytes())
            .context("writing audio mdat size")?;
        writer
            .write_all(b"mdat")
            .context("writing audio mdat type")?;
        for sample in &self.pending {
            writer
                .write_all(&sample.payload)
                .context("writing audio mdat payload")?;
        }
        writer
            .flush()
            .context("flushing CMAF audio segment writer")?;
        let byte_size = moof.bytes.len() as u64 + mdat_box_size;

        self.base_decode_time += segment_duration;
        self.pending.clear();

        let info = SegmentInfo {
            sequence_number: seq,
            path,
            byte_size,
            duration_ticks: segment_duration,
        };
        self.segments.push(info.clone());
        Ok(Some(info))
    }

    pub fn finalize(mut self) -> Result<CmafTrackManifest> {
        if !self.pending.is_empty() {
            self.flush_segment()?;
        }
        self.ensure_init_written()?;
        let timescale = self.info.timescale;
        Ok(CmafTrackManifest {
            init_path: self.init_path,
            segments: self.segments,
            timescale,
        })
    }

    fn ensure_init_written(&mut self) -> Result<()> {
        if self.init_written {
            return Ok(());
        }
        let init = build_init_segment_audio(&self.info);
        let mut file = File::create(&self.init_path).with_context(|| {
            format!(
                "creating CMAF audio init segment: {}",
                self.init_path.display()
            )
        })?;
        file.write_all(&init)
            .context("writing CMAF audio init segment bytes")?;
        file.flush().context("flushing CMAF audio init segment")?;
        self.init_written = true;
        Ok(())
    }
}
