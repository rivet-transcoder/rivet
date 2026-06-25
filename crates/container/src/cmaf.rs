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
use codec::frame::ColorMetadata;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use crate::AudioInfo;
use crate::mux::{BoxBuilder, build_audio_stsd, build_av01, write_unity_matrix};

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
// Box writers
// =====================================================================

/// `mfhd` — Movie Fragment Header (14496-12 §8.8.5).
///
/// Carries the per-fragment sequence number. CMAF requires
/// `sequence_number` to be monotonic and start at 1 for the first
/// fragment of each track.
///
/// Wire layout (16 bytes total):
/// ```text
///   size:u32          = 16
///   type:'mfhd'
///   version:u8        = 0
///   flags:u24         = 0
///   sequence_number:u32
/// ```
pub fn build_mfhd(sequence_number: u32) -> Vec<u8> {
    let mut b = BoxBuilder::new(b"mfhd");
    b.u8(0); // version
    b.extend(&[0, 0, 0]); // flags
    b.u32(sequence_number);
    b.finish()
}

/// `tfhd` — Track Fragment Header (14496-12 §8.8.7).
///
/// We always set the `default-base-is-moof` flag (`0x020000`) — required
/// by CMAF §7.3.2.1. With this flag, sample data offsets in `trun`
/// become relative to the start of the enclosing `moof`, which is
/// exactly what HLS-CMAF expects. We avoid emitting `base_data_offset`
/// (an absolute file offset that breaks segment portability).
///
/// Optional fields are emitted based on the bitwise combination of
/// `tf_flags`:
///   0x000001 base_data_offset            (NOT emitted; we use default-base-is-moof)
///   0x000002 sample_description_index    (only if non-default needed)
///   0x000008 default_sample_duration     (emitted when `default_duration.is_some()`)
///   0x000010 default_sample_size         (emitted when `default_size.is_some()`)
///   0x000020 default_sample_flags        (emitted when `default_flags.is_some()`)
///   0x010000 duration-is-empty
///   0x020000 default-base-is-moof        (always emitted)
pub fn build_tfhd(
    track_id: u32,
    default_duration: Option<u32>,
    default_size: Option<u32>,
    default_flags: Option<u32>,
) -> Vec<u8> {
    let mut tf_flags: u32 = 0x020000; // default-base-is-moof
    if default_duration.is_some() {
        tf_flags |= 0x000008;
    }
    if default_size.is_some() {
        tf_flags |= 0x000010;
    }
    if default_flags.is_some() {
        tf_flags |= 0x000020;
    }

    let mut b = BoxBuilder::new(b"tfhd");
    b.u8(0); // version
    let flag_bytes = tf_flags.to_be_bytes();
    b.extend(&flag_bytes[1..]); // 24-bit flags (drop high byte)
    b.u32(track_id);
    if let Some(d) = default_duration {
        b.u32(d);
    }
    if let Some(s) = default_size {
        b.u32(s);
    }
    if let Some(f) = default_flags {
        b.u32(f);
    }
    b.finish()
}

/// `tfdt` — Track Fragment Decode Time (14496-12 §8.8.12).
///
/// Carries the absolute decode time of the first sample in this
/// fragment, in track timescale ticks, accumulated from the start of
/// the track (NOT from the start of the fragment). Required by CMAF
/// §7.3.2.1.
///
/// We always emit version 1 (u64 decode time). Version 0's u32 wraps
/// at ~24h for a 48 kHz audio track; version 1 covers >12 million
/// years at the same rate. The 4 extra bytes are immaterial.
///
/// Wire layout (20 bytes total):
/// ```text
///   size:u32          = 20
///   type:'tfdt'
///   version:u8        = 1
///   flags:u24         = 0
///   base_media_decode_time:u64
/// ```
pub fn build_tfdt(base_media_decode_time: u64) -> Vec<u8> {
    let mut b = BoxBuilder::new(b"tfdt");
    b.u8(1); // version 1
    b.extend(&[0, 0, 0]); // flags
    b.u64(base_media_decode_time);
    b.finish()
}

/// `trun` — Track Run (14496-12 §8.8.8) for a video fragment.
///
/// Encodes the per-sample table for the fragment's run of samples.
/// CMAF allows multiple `trun`s per `traf` but we always emit exactly
/// one (cleaner manifest, no functional difference).
///
/// Flag bits we always set:
///   0x000001 data-offset-present       (offset from moof start to mdat data)
///   0x000004 first-sample-flags-present (override of default for sample 0)
///   0x000100 sample-duration-present
///   0x000200 sample-size-present
///
/// We don't emit per-sample-flags (0x000400) because all non-first
/// samples in a video fragment share the default (P-frame), and we
/// don't emit sample-composition-time-offsets (0x000800) because
/// AV1 has no B-frame reordering in our pipeline (PTS == DTS).
///
/// `data_offset` is the byte offset from the START of the enclosing
/// `moof` to the first byte of the fragment's `mdat` payload. It
/// CANNOT be filled in until the full `moof` size is known, so this
/// builder leaves it as 0 and returns the byte position to be patched.
/// See [`MoofData::patch_data_offset`].
fn build_trun_video(samples: &[CmafSample]) -> (Vec<u8>, usize) {
    let mut b = BoxBuilder::new(b"trun");
    b.u8(0); // version
    // Flags: data-offset (1) | first-sample-flags (4) | duration (0x100) | size (0x200)
    let flags: u32 = 0x000001 | 0x000004 | 0x000100 | 0x000200;
    let flag_bytes = flags.to_be_bytes();
    b.extend(&flag_bytes[1..]);
    b.u32(samples.len() as u32);
    // data_offset placeholder — final value patched in once moof size is
    // known. We track its absolute position WITHIN this trun box (header
    // 8 + version 1 + flags 3 + sample_count 4 = 16) so the caller can
    // translate to a position-within-moof later.
    let data_offset_pos_within_trun = b.current_len();
    b.u32(0); // placeholder

    // first_sample_flags: the spec's standard pattern is to mark sample
    // 0 explicitly (almost always a sync sample for the first fragment;
    // for subsequent fragments the first sample is whatever the GOP
    // boundary produced — typically also sync since CMAF segments must
    // start with a sync sample per §7.3.2.1).
    if let Some(first) = samples.first() {
        b.u32(first.flags.pack());
    } else {
        b.u32(0);
    }

    for s in samples {
        b.u32(s.duration);
        b.u32(s.size);
    }

    let bytes = b.finish();
    (bytes, data_offset_pos_within_trun)
}

/// `trun` for an audio fragment. Same shape as video but no sync-flags
/// distinction (every audio sample is independently decodable in
/// AAC-LC / Opus / AC-3 / E-AC-3), so we don't emit first-sample-flags
/// — the default in `trex` / `tfhd` covers them all.
fn build_trun_audio(samples: &[CmafSample]) -> (Vec<u8>, usize) {
    let mut b = BoxBuilder::new(b"trun");
    b.u8(0); // version
    // Flags: data-offset (1) | duration (0x100) | size (0x200)
    let flags: u32 = 0x000001 | 0x000100 | 0x000200;
    let flag_bytes = flags.to_be_bytes();
    b.extend(&flag_bytes[1..]);
    b.u32(samples.len() as u32);
    let data_offset_pos_within_trun = b.current_len();
    b.u32(0); // placeholder

    for s in samples {
        b.u32(s.duration);
        b.u32(s.size);
    }

    let bytes = b.finish();
    (bytes, data_offset_pos_within_trun)
}

/// `mehd` — Movie Extends Header (14496-12 §8.8.2).
///
/// Carries the total fragment duration of the longest track, in
/// movie timescale ticks. CMAF treats this as informational; players
/// derive actual duration from the sum of per-fragment `trun` rows.
/// We emit it for spec completeness.
///
/// Version 1 (u64 fragment_duration) — same rationale as `tfdt`.
///
/// Wire layout (20 bytes total):
/// ```text
///   size:u32          = 20
///   type:'mehd'
///   version:u8        = 1
///   flags:u24         = 0
///   fragment_duration:u64
/// ```
pub fn build_mehd(fragment_duration: u64) -> Vec<u8> {
    let mut b = BoxBuilder::new(b"mehd");
    b.u8(1); // version 1
    b.extend(&[0, 0, 0]); // flags
    b.u64(fragment_duration);
    b.finish()
}

/// `trex` — Track Extends (14496-12 §8.8.3).
///
/// Per-track defaults that apply to every `trun` in every `moof`
/// unless overridden via `tfhd`'s default-* fields or per-sample
/// values in `trun`. The point of `trex` is to keep `moof` boxes
/// small: if every sample has the same duration / size / flags, the
/// `trun` can omit them and just inherit from `trex`.
///
/// In practice we override `default_sample_duration` / `_size` per
/// fragment (durations vary slightly with rounding; sizes vary per
/// sample) so most of these fields just hold spec-zero values. We do
/// set `default_sample_description_index = 1` since every sample in
/// our pipeline references the single `stsd` entry built in the
/// init segment.
///
/// Wire layout (32 bytes total):
/// ```text
///   size:u32          = 32
///   type:'trex'
///   version:u8        = 0
///   flags:u24         = 0
///   track_id:u32
///   default_sample_description_index:u32 = 1
///   default_sample_duration:u32          = 0
///   default_sample_size:u32              = 0
///   default_sample_flags:u32             = 0 (or non-sync default)
/// ```
pub fn build_trex(track_id: u32, default_sample_flags: u32) -> Vec<u8> {
    let mut b = BoxBuilder::new(b"trex");
    b.u8(0); // version
    b.extend(&[0, 0, 0]); // flags
    b.u32(track_id);
    b.u32(1); // default_sample_description_index
    b.u32(0); // default_sample_duration (overridden per-fragment)
    b.u32(0); // default_sample_size (overridden per-sample)
    b.u32(default_sample_flags);
    b.finish()
}

/// `mvex` — Movie Extends container (14496-12 §8.8.1).
///
/// Goes inside `moov`. Wraps a single `mehd` plus one `trex` per
/// track. Presence of `mvex` is what tells a parser this MP4 is
/// fragmented (i.e. there will be `moof`s following).
pub fn build_mvex(mehd: &[u8], trexes: &[Vec<u8>]) -> Vec<u8> {
    let mut b = BoxBuilder::new(b"mvex");
    b.extend(mehd);
    for trex in trexes {
        b.extend(trex);
    }
    b.finish()
}

/// `traf` — Track Fragment (14496-12 §8.8.6).
///
/// Wraps `tfhd` + `tfdt` + `trun` for one track inside one `moof`.
/// CMAF mandates exactly one `traf` per `moof` (§7.3.2.1: "Each CMAF
/// Fragment SHALL contain exactly one Track Fragment Box.").
fn build_traf(tfhd: &[u8], tfdt: &[u8], trun: &[u8]) -> Vec<u8> {
    let mut b = BoxBuilder::new(b"traf");
    b.extend(tfhd);
    b.extend(tfdt);
    b.extend(trun);
    b.finish()
}

/// Full `moof` blob with the inner `trun.data_offset` patched up.
///
/// Returned by [`build_moof_video`] and [`build_moof_audio`]. Holds the
/// final byte vector AND knows where inside it the `data_offset` field
/// lives, so callers can either accept the default offset (immediately
/// after the moof — i.e. mdat starts right after this moof in the file)
/// OR substitute their own if they're writing some intervening bytes.
///
/// The default `data_offset` is `bytes.len() + 8`: full moof size plus
/// the 8-byte mdat header. That's the standard "moof immediately
/// followed by mdat" CMAF layout.
pub struct MoofData {
    pub bytes: Vec<u8>,
    /// Byte position WITHIN `bytes` of the 4-byte big-endian
    /// `data_offset` field inside `trun`. Use [`Self::patch_data_offset`]
    /// to overwrite it.
    pub data_offset_pos: usize,
}

impl MoofData {
    /// Patch the `trun.data_offset` field in place. Call once with the
    /// final byte offset from the START of the moof to the START of
    /// the mdat payload (i.e. moof_size + 8 for a no-gap layout).
    pub fn patch_data_offset(&mut self, data_offset: u32) {
        self.bytes[self.data_offset_pos..self.data_offset_pos + 4]
            .copy_from_slice(&data_offset.to_be_bytes());
    }

    /// Convenience: patch with the default no-gap offset (moof
    /// immediately followed by mdat). Use this in the common case
    /// where moof + mdat are written contiguously.
    pub fn patch_default_no_gap(&mut self) {
        let off = (self.bytes.len() + 8) as u32;
        self.patch_data_offset(off);
    }
}

/// Build a video `moof` for one CMAF fragment.
///
/// Composes `mfhd` + `traf{tfhd, tfdt, trun}` and tracks the byte
/// position of `trun.data_offset` so the caller can patch it once
/// the moof's final size is known (or accept the default no-gap
/// layout via [`MoofData::patch_default_no_gap`]).
pub fn build_moof_video(
    sequence_number: u32,
    track_id: u32,
    base_media_decode_time: u64,
    samples: &[CmafSample],
) -> MoofData {
    let mfhd = build_mfhd(sequence_number);
    // Default duration/size omitted — they'll vary per-sample, so
    // emitting them as defaults would be wrong. Default flags set to
    // delta-frame so per-sample flags are needed only on the first
    // (sync) sample, which we override via first_sample_flags in trun.
    let tfhd = build_tfhd(
        track_id,
        None,
        None,
        Some(SampleFlags::delta_frame().pack()),
    );
    let tfdt = build_tfdt(base_media_decode_time);
    let (trun, data_offset_pos_within_trun) = build_trun_video(samples);

    // Compute where `data_offset` lives within the eventual moof.
    // moof_header(8) + mfhd(16) + traf_header(8) + tfhd_len + tfdt(20) +
    //   data_offset_pos_within_trun.
    let moof_header = 8usize;
    let traf_header = 8usize;
    let pos_in_moof = moof_header
        + mfhd.len()
        + traf_header
        + tfhd.len()
        + tfdt.len()
        + data_offset_pos_within_trun;

    let traf = build_traf(&tfhd, &tfdt, &trun);
    let mut b = BoxBuilder::new(b"moof");
    b.extend(&mfhd);
    b.extend(&traf);
    let bytes = b.finish();

    MoofData {
        bytes,
        data_offset_pos: pos_in_moof,
    }
}

/// Build an audio `moof`. Same composition as video but without
/// first-sample-flags differentiation in `trun` (every audio sample
/// is independently decodable).
pub fn build_moof_audio(
    sequence_number: u32,
    track_id: u32,
    base_media_decode_time: u64,
    samples: &[CmafSample],
) -> MoofData {
    let mfhd = build_mfhd(sequence_number);
    // Audio default-flags: every sample is independently decodable,
    // so default to sync.
    let tfhd = build_tfhd(track_id, None, None, Some(SampleFlags::keyframe().pack()));
    let tfdt = build_tfdt(base_media_decode_time);
    let (trun, data_offset_pos_within_trun) = build_trun_audio(samples);

    let moof_header = 8usize;
    let traf_header = 8usize;
    let pos_in_moof = moof_header
        + mfhd.len()
        + traf_header
        + tfhd.len()
        + tfdt.len()
        + data_offset_pos_within_trun;

    let traf = build_traf(&tfhd, &tfdt, &trun);
    let mut b = BoxBuilder::new(b"moof");
    b.extend(&mfhd);
    b.extend(&traf);
    let bytes = b.finish();

    MoofData {
        bytes,
        data_offset_pos: pos_in_moof,
    }
}

// =====================================================================
// Init segment writers (Phase 1.2)
// =====================================================================
//
// CMAF init segments carry `ftyp + moov` only — no sample data. The
// `moov.trak.mdia.minf.stbl` has a populated `stsd` (the sample
// description) but EMPTY `stts/stsc/stsz/stco`. That's how the parser
// knows samples will arrive in subsequent `moof` boxes via the
// `mvex/trex` defaults set in this same `moov`.
//
// The track is one-per-init per CMAF §7.3.2.1 (each video init carries
// only the video track, each audio init only the audio track).
// `track_id = 1` in both cases since each init's `moov` is independent.

/// Build a CMAF video init segment for an AV1 track.
///
/// `config_obus` is the LOB-formatted OBU sequence header (with
/// `obu_has_size_field=1`) — call [`crate::mux::extract_sequence_header`]
/// against the first encoded packet to get this. `timescale` is the
/// track's mdhd/mvhd timescale in ticks per second; we recommend
/// `frame_rate × 1000` rounded to a clean number (e.g. 30000 for 30fps,
/// 24000 for 24fps) so per-frame durations divide evenly. The fragment
/// duration in `mehd` is left at 0 (informational; players derive
/// actual duration from `trun`).
pub fn build_init_segment_video(
    width: u32,
    height: u32,
    timescale: u32,
    config_obus: &[u8],
    color_metadata: &ColorMetadata,
) -> Vec<u8> {
    let track_id = 1u32;

    // ftyp — major_brand=iso6, brands include cmfc + av01
    let ftyp = build_ftyp_video();

    // moov children
    let mvhd = build_mvhd(timescale, /* duration */ 0, /* next_track_id */ 2);
    let trak = build_video_trak(
        width,
        height,
        timescale,
        track_id,
        config_obus,
        color_metadata,
    );
    let mvex_blob = {
        let mehd = build_mehd(0);
        // For video, default sample flags are delta-frame (most samples
        // in a fragment are P-frames); the IDR opening each fragment
        // overrides via trun's first_sample_flags. This matches what the
        // moof writer sets in tfhd.
        let trex = build_trex(track_id, SampleFlags::delta_frame().pack());
        build_mvex(&mehd, &[trex])
    };

    let mut moov = BoxBuilder::new(b"moov");
    moov.extend(&mvhd);
    moov.extend(&trak);
    moov.extend(&mvex_blob);
    let moov = moov.finish();

    let mut out = Vec::with_capacity(ftyp.len() + moov.len());
    out.extend_from_slice(&ftyp);
    out.extend_from_slice(&moov);
    out
}

/// Build a CMAF audio init segment.
///
/// `audio_info` carries codec / sample_rate / channels / asc_bytes (or
/// codec_private for Opus / AC-3 / E-AC-3). Same struct the existing
/// non-fragmented muxer's `with_audio` accepts — see crate::AudioInfo.
pub fn build_init_segment_audio(audio_info: &AudioInfo) -> Vec<u8> {
    let track_id = 1u32;

    let ftyp = build_ftyp_audio();

    let mvhd = build_mvhd(
        audio_info.timescale,
        /* duration */ 0,
        /* next_track_id */ 2,
    );
    let trak = build_audio_trak(audio_info, track_id);
    let mvex_blob = {
        let mehd = build_mehd(0);
        // Every audio sample is independently decodable — sync default.
        let trex = build_trex(track_id, SampleFlags::keyframe().pack());
        build_mvex(&mehd, &[trex])
    };

    let mut moov = BoxBuilder::new(b"moov");
    moov.extend(&mvhd);
    moov.extend(&trak);
    moov.extend(&mvex_blob);
    let moov = moov.finish();

    let mut out = Vec::with_capacity(ftyp.len() + moov.len());
    out.extend_from_slice(&ftyp);
    out.extend_from_slice(&moov);
    out
}

/// `ftyp` for a video init segment. Brands declare `cmfc` (CMAF video
/// constraints), `av01` (AV1-in-MP4), plus `iso6` / `mp42` / `iso2` for
/// broad parser compatibility. Major brand is `iso6` (CMAF / 14496-12
/// edition 6) — Apple's player and ffmpeg both honour it.
fn build_ftyp_video() -> Vec<u8> {
    let mut b = BoxBuilder::new(b"ftyp");
    b.extend(b"iso6"); // major_brand
    b.u32(0); // minor_version
    b.extend(b"iso6");
    b.extend(b"iso2");
    b.extend(b"mp42");
    b.extend(brand::CMFC);
    b.extend(b"av01");
    b.finish()
}

/// `ftyp` for an audio init segment. Same as video but `cmfa` brand
/// instead of `cmfc`, and no `av01` (irrelevant for an audio-only
/// segment).
fn build_ftyp_audio() -> Vec<u8> {
    let mut b = BoxBuilder::new(b"ftyp");
    b.extend(b"iso6"); // major_brand
    b.u32(0); // minor_version
    b.extend(b"iso6");
    b.extend(b"iso2");
    b.extend(b"mp42");
    b.extend(brand::CMFA);
    b.finish()
}

/// `mvhd` (14496-12 §8.2.2) — movie header. Same layout as the existing
/// non-fragmented muxer; reimplemented here because we need a slightly
/// different `next_track_id` (single-track init segments).
fn build_mvhd(timescale: u32, duration: u64, next_track_id: u32) -> Vec<u8> {
    let mut b = BoxBuilder::new(b"mvhd");
    b.u8(0);
    b.extend(&[0, 0, 0]);
    b.u32(0); // creation_time
    b.u32(0); // modification_time
    b.u32(timescale);
    b.u32(duration as u32);
    b.u32(0x00010000); // rate 1.0
    b.u16(0x0100); // volume 1.0
    b.u16(0); // reserved
    b.u32(0);
    b.u32(0);
    write_unity_matrix(&mut b);
    for _ in 0..6 {
        b.u32(0);
    } // pre_defined
    b.u32(next_track_id);
    b.finish()
}

fn build_video_trak(
    width: u32,
    height: u32,
    timescale: u32,
    track_id: u32,
    config_obus: &[u8],
    color_metadata: &ColorMetadata,
) -> Vec<u8> {
    let tkhd = build_video_tkhd(width, height, track_id);
    let mdia = build_video_mdia(width, height, timescale, config_obus, color_metadata);
    let mut b = BoxBuilder::new(b"trak");
    b.extend(&tkhd);
    b.extend(&mdia);
    b.finish()
}

fn build_video_tkhd(width: u32, height: u32, track_id: u32) -> Vec<u8> {
    let mut b = BoxBuilder::new(b"tkhd");
    b.u8(0);
    // flags = 0x000003 (track_enabled | track_in_movie). We don't set
    // 0x000004 (track_in_preview) — that's a QuickTime-flavored bit and
    // streaming players ignore it.
    b.extend(&[0, 0, 0x03]);
    b.u32(0); // creation_time
    b.u32(0); // modification_time
    b.u32(track_id);
    b.u32(0); // reserved
    b.u32(0); // duration (movie timescale; fragment muxer leaves this 0)
    b.u32(0);
    b.u32(0);
    b.u16(0); // layer
    b.u16(0); // alternate_group
    b.u16(0); // volume = 0 for video
    b.u16(0); // reserved
    write_unity_matrix(&mut b);
    b.u32(width << 16); // width 16.16
    b.u32(height << 16);
    b.finish()
}

fn build_video_mdia(
    width: u32,
    height: u32,
    timescale: u32,
    config_obus: &[u8],
    color_metadata: &ColorMetadata,
) -> Vec<u8> {
    let mdhd = build_mdhd(timescale, 0);
    let hdlr = build_hdlr(b"vide", "VideoHandler\0");
    let minf = build_video_minf(width, height, config_obus, color_metadata);
    let mut b = BoxBuilder::new(b"mdia");
    b.extend(&mdhd);
    b.extend(&hdlr);
    b.extend(&minf);
    b.finish()
}

fn build_mdhd(timescale: u32, duration: u64) -> Vec<u8> {
    let mut b = BoxBuilder::new(b"mdhd");
    b.u8(0);
    b.extend(&[0, 0, 0]);
    b.u32(0); // creation_time
    b.u32(0); // modification_time
    b.u32(timescale);
    b.u32(duration as u32);
    b.u16(0x55c4); // language 'und'
    b.u16(0); // pre_defined
    b.finish()
}

/// Generic handler box — `'vide'` for video, `'soun'` for audio. The
/// human-readable name string (with trailing NUL) is purely
/// informational; ffprobe surfaces it but no playback path consumes it.
fn build_hdlr(handler_type: &[u8; 4], name: &str) -> Vec<u8> {
    let mut b = BoxBuilder::new(b"hdlr");
    b.u8(0);
    b.extend(&[0, 0, 0]);
    b.u32(0); // pre_defined
    b.extend(handler_type);
    b.u32(0);
    b.u32(0);
    b.u32(0); // reserved[3]
    b.extend(name.as_bytes());
    b.finish()
}

fn build_video_minf(
    width: u32,
    height: u32,
    config_obus: &[u8],
    color_metadata: &ColorMetadata,
) -> Vec<u8> {
    let vmhd = build_vmhd();
    let dinf = build_dinf();
    let stbl = build_video_stbl_empty(width, height, config_obus, color_metadata);
    let mut b = BoxBuilder::new(b"minf");
    b.extend(&vmhd);
    b.extend(&dinf);
    b.extend(&stbl);
    b.finish()
}

fn build_vmhd() -> Vec<u8> {
    let mut b = BoxBuilder::new(b"vmhd");
    b.u8(0);
    b.extend(&[0, 0, 0x01]); // flags = 1 per spec
    b.u16(0); // graphicsmode (0 = copy)
    b.u16(0);
    b.u16(0);
    b.u16(0); // opcolor[3] (RGB, 0,0,0)
    b.finish()
}

fn build_smhd() -> Vec<u8> {
    let mut b = BoxBuilder::new(b"smhd");
    b.u8(0);
    b.extend(&[0, 0, 0]);
    b.u16(0); // balance (0 = center)
    b.u16(0); // reserved
    b.finish()
}

/// `dinf` containing a minimal `dref` with one `url ` self-reference.
/// Required by 14496-12 even when sample data is in the same file.
fn build_dinf() -> Vec<u8> {
    let url = {
        let mut b = BoxBuilder::new(b"url ");
        b.u8(0); // version
        b.extend(&[0, 0, 0x01]); // flags = 1 (data is in the same file)
        b.finish()
    };
    let dref = {
        let mut b = BoxBuilder::new(b"dref");
        b.u8(0);
        b.extend(&[0, 0, 0]);
        b.u32(1); // entry_count
        b.extend(&url);
        b.finish()
    };
    let mut b = BoxBuilder::new(b"dinf");
    b.extend(&dref);
    b.finish()
}

/// Empty sample tables for a CMAF video init: `stsd` has the av01
/// sample entry (with av1C, colr, optional mdcv/clli) and the rest of
/// the tables are empty boxes (entry_count=0).
fn build_video_stbl_empty(
    width: u32,
    height: u32,
    config_obus: &[u8],
    color_metadata: &ColorMetadata,
) -> Vec<u8> {
    let av01 = build_av01(width, height, config_obus, color_metadata);
    let stsd = {
        let mut b = BoxBuilder::new(b"stsd");
        b.u8(0);
        b.extend(&[0, 0, 0]);
        b.u32(1); // entry_count
        b.extend(&av01);
        b.finish()
    };
    let stts = build_empty_full_box(b"stts");
    let stsc = build_empty_full_box(b"stsc");
    let stsz = {
        let mut b = BoxBuilder::new(b"stsz");
        b.u8(0);
        b.extend(&[0, 0, 0]);
        b.u32(0); // sample_size = 0 → variable, per stsz (then sample_count must be 0 too)
        b.u32(0); // sample_count = 0
        b.finish()
    };
    let stco = build_empty_full_box(b"stco");

    let mut b = BoxBuilder::new(b"stbl");
    b.extend(&stsd);
    b.extend(&stts);
    b.extend(&stsc);
    b.extend(&stsz);
    b.extend(&stco);
    b.finish()
}

fn build_audio_trak(info: &AudioInfo, track_id: u32) -> Vec<u8> {
    let tkhd = build_audio_tkhd(track_id);
    let mdia = build_audio_mdia(info);
    let mut b = BoxBuilder::new(b"trak");
    b.extend(&tkhd);
    b.extend(&mdia);
    b.finish()
}

fn build_audio_tkhd(track_id: u32) -> Vec<u8> {
    let mut b = BoxBuilder::new(b"tkhd");
    b.u8(0);
    b.extend(&[0, 0, 0x03]);
    b.u32(0);
    b.u32(0);
    b.u32(track_id);
    b.u32(0);
    b.u32(0);
    b.u32(0);
    b.u32(0);
    b.u16(0); // layer
    b.u16(0); // alternate_group (audio init has only one track; 0 fine)
    b.u16(0x0100); // volume 1.0
    b.u16(0); // reserved
    write_unity_matrix(&mut b);
    b.u32(0);
    b.u32(0); // width / height = 0
    b.finish()
}

fn build_audio_mdia(info: &AudioInfo) -> Vec<u8> {
    let mdhd = build_mdhd(info.timescale, 0);
    let hdlr = build_hdlr(b"soun", "SoundHandler\0");
    let minf = build_audio_minf(info);
    let mut b = BoxBuilder::new(b"mdia");
    b.extend(&mdhd);
    b.extend(&hdlr);
    b.extend(&minf);
    b.finish()
}

fn build_audio_minf(info: &AudioInfo) -> Vec<u8> {
    let smhd = build_smhd();
    let dinf = build_dinf();
    let stbl = build_audio_stbl_empty(info);
    let mut b = BoxBuilder::new(b"minf");
    b.extend(&smhd);
    b.extend(&dinf);
    b.extend(&stbl);
    b.finish()
}

fn build_audio_stbl_empty(info: &AudioInfo) -> Vec<u8> {
    let stsd = build_audio_stsd(info);
    let stts = build_empty_full_box(b"stts");
    let stsc = build_empty_full_box(b"stsc");
    let stsz = {
        let mut b = BoxBuilder::new(b"stsz");
        b.u8(0);
        b.extend(&[0, 0, 0]);
        b.u32(0);
        b.u32(0);
        b.finish()
    };
    let stco = build_empty_full_box(b"stco");

    let mut b = BoxBuilder::new(b"stbl");
    b.extend(&stsd);
    b.extend(&stts);
    b.extend(&stsc);
    b.extend(&stsz);
    b.extend(&stco);
    b.finish()
}

/// Empty FullBox with version 0 + flags 0 + entry_count 0. Layout:
///   size:u32 = 16 | type | version:u8 = 0 | flags:u24 = 0 | entry_count:u32 = 0
fn build_empty_full_box(box_type: &[u8; 4]) -> Vec<u8> {
    let mut b = BoxBuilder::new(box_type);
    b.u8(0);
    b.extend(&[0, 0, 0]);
    b.u32(0);
    b.finish()
}

// =====================================================================
// Stateful per-rendition segmenter (Phase 1.3 + 1.4)
// =====================================================================
//
// Each `CmafVideoMuxer` (one per video rendition) and `CmafAudioMuxer`
// (one per audio rendition; usually a single instance per asset)
// accumulates encoded packets in memory and flushes them to disk as
// CMAF media segments (`seg-NNNNN.m4s` = `moof + mdat`) on demand.
//
// Memory ceiling: at most one segment's worth of payload bytes are
// held at a time (caller flushes at every keyframe boundary that
// crosses the segment-duration target). For a 4-second 1080p AV1
// segment at ~3 Mbps that's ~1.5 MB; not a concern at the per-job
// 4 GiB ceiling.
//
// The init segment (`init.mp4`) is written EAGERLY on construction
// for audio (we have everything we need) and LAZILY on first flush
// for video (we need the first packet's OBU sequence header to build
// the av1C config record).

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
    config_obus: Option<Vec<u8>>, // captured from the first packet
    init_path: PathBuf,
    init_written: bool,
    sequence_number: u32,
    base_decode_time: u64,
    pending: Vec<PendingVideoSample>,
    segments: Vec<SegmentInfo>,
}

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
        assert!(
            options.first_segment_index >= 1,
            "first_segment_index is 1-based; got {}",
            options.first_segment_index,
        );
        let output_dir = output_dir.as_ref().to_path_buf();
        fs::create_dir_all(&output_dir)
            .with_context(|| format!("creating CMAF video output dir: {}", output_dir.display()))?;
        let init_path = output_dir.join("init.mp4");
        Ok(Self {
            output_dir,
            width,
            height,
            timescale,
            color_metadata,
            track_id: 1,
            config_obus: None,
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
        if self.config_obus.is_none() {
            self.config_obus = Some(crate::mux::extract_sequence_header(&payload).context(
                "extracting AV1 sequence header from first packet for av1C config record",
            )?);
        }
        self.pending.push(PendingVideoSample {
            payload,
            duration,
            is_keyframe,
        });
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
        let config = self.config_obus.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "cannot write CMAF video init segment: no AV1 sequence header has been observed yet \
                 (must call add_packet at least once before flush_segment / finalize)"
            )
        })?;
        let init = build_init_segment_video(
            self.width,
            self.height,
            self.timescale,
            config,
            &self.color_metadata,
        );
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

#[cfg(test)]
mod tests {
    use super::*;

    fn read_be_u32(buf: &[u8], pos: usize) -> u32 {
        u32::from_be_bytes(buf[pos..pos + 4].try_into().unwrap())
    }

    fn read_be_u64(buf: &[u8], pos: usize) -> u64 {
        u64::from_be_bytes(buf[pos..pos + 8].try_into().unwrap())
    }

    fn box_size_and_type(buf: &[u8]) -> (u32, &[u8]) {
        let size = read_be_u32(buf, 0);
        let kind = &buf[4..8];
        (size, kind)
    }

    #[test]
    fn mfhd_layout_is_16_bytes_with_sequence_number() {
        let bytes = build_mfhd(42);
        assert_eq!(bytes.len(), 16);
        let (size, kind) = box_size_and_type(&bytes);
        assert_eq!(size, 16);
        assert_eq!(kind, b"mfhd");
        assert_eq!(bytes[8], 0); // version
        assert_eq!(&bytes[9..12], &[0, 0, 0]); // flags
        assert_eq!(read_be_u32(&bytes, 12), 42);
    }

    #[test]
    fn tfhd_minimal_track_id_only_is_16_bytes() {
        let bytes = build_tfhd(1, None, None, None);
        // 8 (header) + 1 (version) + 3 (flags) + 4 (track_id) = 16.
        assert_eq!(bytes.len(), 16);
        let (size, kind) = box_size_and_type(&bytes);
        assert_eq!(size, 16);
        assert_eq!(kind, b"tfhd");
        // tf_flags should ONLY have default-base-is-moof (0x020000) set.
        let flag_bytes = [0u8, bytes[9], bytes[10], bytes[11]];
        let flags = u32::from_be_bytes(flag_bytes);
        assert_eq!(flags, 0x020000);
        assert_eq!(read_be_u32(&bytes, 12), 1);
    }

    #[test]
    fn tfhd_with_default_flags_only_packs_correct_bits() {
        let bytes = build_tfhd(1, None, None, Some(SampleFlags::delta_frame().pack()));
        // 8 header + 1 version + 3 flags + 4 track_id + 4 default_sample_flags = 20.
        assert_eq!(bytes.len(), 20);
        let flag_bytes = [0u8, bytes[9], bytes[10], bytes[11]];
        let flags = u32::from_be_bytes(flag_bytes);
        // default-base-is-moof (0x020000) | default-sample-flags (0x000020).
        assert_eq!(flags, 0x020020);
        assert_eq!(read_be_u32(&bytes, 12), 1);
        assert_eq!(read_be_u32(&bytes, 16), SampleFlags::delta_frame().pack());
    }

    #[test]
    fn tfhd_with_all_defaults_packs_in_spec_order() {
        let bytes = build_tfhd(1, Some(1024), Some(2048), Some(0x01010000));
        // 8 + 1 + 3 + 4 + 4 + 4 + 4 = 28.
        assert_eq!(bytes.len(), 28);
        let flag_bytes = [0u8, bytes[9], bytes[10], bytes[11]];
        let flags = u32::from_be_bytes(flag_bytes);
        // default-base-is-moof (0x020000) | dur (0x000008) | size (0x000010) | flags (0x000020).
        assert_eq!(flags, 0x020038);
        assert_eq!(read_be_u32(&bytes, 12), 1);
        assert_eq!(read_be_u32(&bytes, 16), 1024); // duration
        assert_eq!(read_be_u32(&bytes, 20), 2048); // size
        assert_eq!(read_be_u32(&bytes, 24), 0x01010000); // flags
    }

    #[test]
    fn tfdt_v1_carries_u64_decode_time() {
        let bytes = build_tfdt(0x0123_4567_89AB_CDEF);
        // 8 header + 1 version + 3 flags + 8 decode_time = 20.
        assert_eq!(bytes.len(), 20);
        assert_eq!(box_size_and_type(&bytes), (20, b"tfdt".as_slice()));
        assert_eq!(bytes[8], 1); // version 1
        assert_eq!(read_be_u64(&bytes, 12), 0x0123_4567_89AB_CDEF);
    }

    #[test]
    fn mehd_v1_carries_u64_fragment_duration() {
        let bytes = build_mehd(1_000_000);
        assert_eq!(bytes.len(), 20);
        assert_eq!(box_size_and_type(&bytes), (20, b"mehd".as_slice()));
        assert_eq!(bytes[8], 1);
        assert_eq!(read_be_u64(&bytes, 12), 1_000_000);
    }

    #[test]
    fn trex_layout_is_32_bytes_with_track_id_and_flags() {
        let default_flags = SampleFlags::delta_frame().pack();
        let bytes = build_trex(2, default_flags);
        // 8 + 1 + 3 + 4 + 4 + 4 + 4 + 4 = 32.
        assert_eq!(bytes.len(), 32);
        assert_eq!(box_size_and_type(&bytes), (32, b"trex".as_slice()));
        assert_eq!(read_be_u32(&bytes, 12), 2); // track_id
        assert_eq!(read_be_u32(&bytes, 16), 1); // default_sample_description_index
        assert_eq!(read_be_u32(&bytes, 20), 0); // default_sample_duration
        assert_eq!(read_be_u32(&bytes, 24), 0); // default_sample_size
        assert_eq!(read_be_u32(&bytes, 28), default_flags);
    }

    #[test]
    fn sample_flags_pack_distinguishes_sync_from_delta() {
        let sync = SampleFlags::keyframe().pack();
        let delta = SampleFlags::delta_frame().pack();
        assert_ne!(sync, delta);
        // Sync: depends_on=2 in bits 24-25, is_non_sync=0 in bit 16.
        assert_eq!(sync, 0x02_00_00_00);
        // Delta: depends_on=1, is_non_sync=1.
        assert_eq!(delta, 0x01_01_00_00);
    }

    #[test]
    fn moof_video_one_keyframe_sample_round_trip() {
        let samples = vec![CmafSample {
            duration: 1500,
            size: 4096,
            flags: SampleFlags::keyframe(),
        }];
        let mut moof = build_moof_video(1, 1, 0, &samples);
        moof.patch_default_no_gap();

        let (size, kind) = box_size_and_type(&moof.bytes);
        assert_eq!(size as usize, moof.bytes.len());
        assert_eq!(kind, b"moof");

        // mfhd starts at offset 8 (after moof header).
        let (mfhd_size, mfhd_kind) = box_size_and_type(&moof.bytes[8..]);
        assert_eq!(mfhd_size, 16);
        assert_eq!(mfhd_kind, b"mfhd");
        assert_eq!(read_be_u32(&moof.bytes, 8 + 12), 1); // sequence_number

        // traf starts after mfhd.
        let traf_start = 8 + mfhd_size as usize;
        let (_, traf_kind) = box_size_and_type(&moof.bytes[traf_start..]);
        assert_eq!(traf_kind, b"traf");

        // The patched data_offset should equal moof.len() + 8.
        let patched = read_be_u32(&moof.bytes, moof.data_offset_pos);
        assert_eq!(patched as usize, moof.bytes.len() + 8);

        // The first_sample_flags slot in trun should equal the keyframe flags.
        // It sits 4 bytes after the data_offset field per the trun layout.
        let first_flags = read_be_u32(&moof.bytes, moof.data_offset_pos + 4);
        assert_eq!(first_flags, SampleFlags::keyframe().pack());
    }

    #[test]
    fn moof_video_three_samples_records_per_sample_dur_and_size() {
        let samples = vec![
            CmafSample {
                duration: 1500,
                size: 4096,
                flags: SampleFlags::keyframe(),
            },
            CmafSample {
                duration: 1500,
                size: 1024,
                flags: SampleFlags::delta_frame(),
            },
            CmafSample {
                duration: 1500,
                size: 1024,
                flags: SampleFlags::delta_frame(),
            },
        ];
        let mut moof = build_moof_video(2, 1, 6000, &samples);
        moof.patch_default_no_gap();

        // Walk into trun and read sample_count.
        // moof header(8) + mfhd(16) + traf header(8) = 32.
        // Then tfhd: 8 + 1 + 3 + 4 + 4 = 20 bytes (track_id + default_flags).
        // Then tfdt v1: 20 bytes.
        // trun starts at 32 + 20 + 20 = 72.
        let trun_start = 8 + 16 + 8 + 20 + 20;
        let (_, trun_kind) = box_size_and_type(&moof.bytes[trun_start..]);
        assert_eq!(trun_kind, b"trun");
        let sample_count = read_be_u32(&moof.bytes, trun_start + 12);
        assert_eq!(sample_count, 3);

        // Per-sample table starts after data_offset(4) + first_sample_flags(4):
        //   trun_start + 8(header) + 1(version) + 3(flags) + 4(count) +
        //                4(data_offset) + 4(first_sample_flags) = trun_start + 24.
        let table_start = trun_start + 24;
        // sample 0: dur=1500, size=4096
        assert_eq!(read_be_u32(&moof.bytes, table_start), 1500);
        assert_eq!(read_be_u32(&moof.bytes, table_start + 4), 4096);
        // sample 1: dur=1500, size=1024
        assert_eq!(read_be_u32(&moof.bytes, table_start + 8), 1500);
        assert_eq!(read_be_u32(&moof.bytes, table_start + 12), 1024);
        // sample 2: dur=1500, size=1024
        assert_eq!(read_be_u32(&moof.bytes, table_start + 16), 1500);
        assert_eq!(read_be_u32(&moof.bytes, table_start + 20), 1024);
    }

    #[test]
    fn moof_audio_does_not_emit_first_sample_flags() {
        let samples = vec![
            CmafSample {
                duration: 1024,
                size: 256,
                flags: SampleFlags::keyframe(),
            },
            CmafSample {
                duration: 1024,
                size: 256,
                flags: SampleFlags::keyframe(),
            },
        ];
        let mut moof = build_moof_audio(1, 2, 0, &samples);
        moof.patch_default_no_gap();

        // Audio trun flags = 0x000001 | 0x000100 | 0x000200 = 0x000301
        // (no first-sample-flags bit, no per-sample-flags bit).
        let trun_start = 8 + 16 + 8 + 20 + 20;
        let flag_bytes = [
            0u8,
            moof.bytes[trun_start + 9],
            moof.bytes[trun_start + 10],
            moof.bytes[trun_start + 11],
        ];
        let flags = u32::from_be_bytes(flag_bytes);
        assert_eq!(flags, 0x000001 | 0x000100 | 0x000200);

        // Per-sample table starts after data_offset(4) only — no
        // first_sample_flags this time.
        //   trun_start + 8 + 1 + 3 + 4 + 4 = trun_start + 20.
        let table_start = trun_start + 20;
        assert_eq!(read_be_u32(&moof.bytes, table_start), 1024); // sample 0 dur
        assert_eq!(read_be_u32(&moof.bytes, table_start + 4), 256); // sample 0 size
        assert_eq!(read_be_u32(&moof.bytes, table_start + 8), 1024); // sample 1 dur
        assert_eq!(read_be_u32(&moof.bytes, table_start + 12), 256); // sample 1 size
    }

    #[test]
    fn moof_data_offset_patch_is_at_correct_position() {
        // Keyframe-only fragment of 1 sample. Data offset is at a
        // computable position; verify patch_data_offset writes there.
        let samples = vec![CmafSample {
            duration: 1500,
            size: 1234,
            flags: SampleFlags::keyframe(),
        }];
        let mut moof = build_moof_video(1, 1, 0, &samples);
        moof.patch_data_offset(0xDEAD_BEEF);
        let read_back = read_be_u32(&moof.bytes, moof.data_offset_pos);
        assert_eq!(read_back, 0xDEAD_BEEF);
    }

    // Synthetic AV1 OBU bytes that contain exactly one
    // OBU_SEQUENCE_HEADER (type=1, has_size=1, ext=0). This is what
    // `extract_sequence_header` sniffs out of the first encoded packet
    // to build the av1C config record. Payload is 1 byte (0xAA) — the
    // value is irrelevant for our shape tests; the muxer just round-
    // trips it as bytes inside av1C.
    fn synthetic_seq_header_packet() -> Vec<u8> {
        let header_byte: u8 = (1 << 3) | (1 << 1); // obu_type=1, has_size=1
        vec![header_byte, 0x01, 0xAA]
    }

    fn find_box<'a>(buf: &'a [u8], box_type: &[u8; 4]) -> Option<&'a [u8]> {
        let mut pos = 0;
        while pos + 8 <= buf.len() {
            let size = read_be_u32(buf, pos) as usize;
            if size < 8 || pos + size > buf.len() {
                return None;
            }
            let kind = &buf[pos + 4..pos + 8];
            if kind == box_type {
                return Some(&buf[pos..pos + size]);
            }
            pos += size;
        }
        None
    }

    fn ftyp_compatible_brands(ftyp: &[u8]) -> Vec<&[u8]> {
        // size:4 + 'ftyp' + major:4 + minor:4 = 16, then brands[]
        let mut brands = Vec::new();
        let mut p = 16;
        while p + 4 <= ftyp.len() {
            brands.push(&ftyp[p..p + 4]);
            p += 4;
        }
        brands
    }

    #[test]
    fn init_segment_video_lists_cmfc_and_av01_brands() {
        let init = build_init_segment_video(
            1920,
            1080,
            30000,
            &synthetic_seq_header_packet(),
            &ColorMetadata::default(),
        );
        let ftyp = find_box(&init, b"ftyp").expect("init has ftyp");
        let brands = ftyp_compatible_brands(ftyp);
        assert!(
            brands.contains(&b"cmfc".as_slice()),
            "cmfc brand missing: {brands:?}"
        );
        assert!(
            brands.contains(&b"av01".as_slice()),
            "av01 brand missing: {brands:?}"
        );
        assert!(
            brands.contains(&b"iso6".as_slice()),
            "iso6 brand missing: {brands:?}"
        );
    }

    #[test]
    fn init_segment_audio_lists_cmfa_brand() {
        // ASC bytes for AAC-LC: object_type=2 (LC), sample_rate_index=3 (48 kHz),
        // channelConfiguration=2 (stereo).
        let info = AudioInfo::aac_lc(48000, 2, vec![0x11, 0x90]);
        let init = build_init_segment_audio(&info);
        let ftyp = find_box(&init, b"ftyp").expect("init has ftyp");
        let brands = ftyp_compatible_brands(ftyp);
        assert!(
            brands.contains(&b"cmfa".as_slice()),
            "cmfa brand missing: {brands:?}"
        );
        assert!(
            !brands.contains(&b"cmfc".as_slice()),
            "cmfc should not appear in audio init"
        );
    }

    #[test]
    fn init_segment_video_moov_contains_mvex_with_trex() {
        let init = build_init_segment_video(
            1280,
            720,
            30000,
            &synthetic_seq_header_packet(),
            &ColorMetadata::default(),
        );
        let moov = find_box(&init, b"moov").expect("init has moov");
        let mvex = find_box(&moov[8..], b"mvex").expect("moov has mvex");
        assert!(
            find_box(&mvex[8..], b"trex").is_some(),
            "mvex must contain trex"
        );
        assert!(
            find_box(&mvex[8..], b"mehd").is_some(),
            "mvex must contain mehd"
        );
    }

    #[test]
    fn init_segment_video_stbl_has_empty_sample_tables() {
        let init = build_init_segment_video(
            1280,
            720,
            30000,
            &synthetic_seq_header_packet(),
            &ColorMetadata::default(),
        );
        let moov = find_box(&init, b"moov").expect("init has moov");
        let trak = find_box(&moov[8..], b"trak").expect("moov has trak");
        let mdia = find_box(&trak[8..], b"mdia").expect("trak has mdia");
        let minf = find_box(&mdia[8..], b"minf").expect("mdia has minf");
        let stbl = find_box(&minf[8..], b"stbl").expect("minf has stbl");

        // stsz: sample_size=0 (variable), sample_count=0 (no samples in init)
        let stsz = find_box(&stbl[8..], b"stsz").expect("stbl has stsz");
        // 8 (header) + 1 (version) + 3 (flags) + 4 (sample_size) + 4 (sample_count) = 20.
        assert_eq!(stsz.len(), 20);
        assert_eq!(read_be_u32(stsz, 12), 0); // sample_size
        assert_eq!(read_be_u32(stsz, 16), 0); // sample_count

        // stts/stsc/stco: entry_count=0
        for box_type in [b"stts", b"stsc", b"stco"] {
            let bx = find_box(&stbl[8..], box_type).expect("stbl has empty full box");
            assert_eq!(
                bx.len(),
                16,
                "{:?} should be 16-byte empty FullBox",
                std::str::from_utf8(box_type).unwrap()
            );
            assert_eq!(read_be_u32(bx, 12), 0); // entry_count
        }

        // stsd has exactly one entry — the av01 sample entry.
        let stsd = find_box(&stbl[8..], b"stsd").expect("stbl has stsd");
        assert_eq!(read_be_u32(stsd, 12), 1); // entry_count
        // First sample entry should be av01.
        let av01 = &stsd[16..];
        assert_eq!(&av01[4..8], b"av01");
    }

    #[test]
    fn cmaf_video_muxer_emits_init_then_segment_files() {
        let dir = tempfile::tempdir().unwrap();
        let mut muxer =
            CmafVideoMuxer::new(dir.path(), 1280, 720, 30000, ColorMetadata::default()).unwrap();

        // Two-packet "fragment": one keyframe, one delta. Each "payload"
        // starts with the synthetic sequence header (so the muxer's
        // first-packet OBU sniff succeeds) but the muxer doesn't care
        // about the rest of the payload bytes — it just round-trips
        // them through mdat.
        let mut k = synthetic_seq_header_packet();
        k.extend_from_slice(&[0xDE, 0xAD]);
        muxer.add_packet(k, 1500, true).unwrap();
        muxer
            .add_packet(synthetic_seq_header_packet(), 1500, false)
            .unwrap();

        let info = muxer
            .flush_segment()
            .unwrap()
            .expect("flush emits a segment");
        assert_eq!(info.sequence_number, 1);
        assert_eq!(info.duration_ticks, 3000);
        assert!(info.path.exists());
        assert_eq!(info.path.file_name().unwrap(), "seg-00001.m4s");

        // init.mp4 was written lazily on first flush.
        let init_path = dir.path().join("init.mp4");
        assert!(init_path.exists(), "init.mp4 must exist after first flush");

        // Segment file starts with `moof` and contains an `mdat` after.
        let seg_bytes = std::fs::read(&info.path).unwrap();
        assert_eq!(&seg_bytes[4..8], b"moof");
        let moof_size = read_be_u32(&seg_bytes, 0) as usize;
        assert_eq!(&seg_bytes[moof_size + 4..moof_size + 8], b"mdat");

        // Manifest finalize covers the empty-pending case (we already flushed).
        let manifest = muxer.finalize().unwrap();
        assert_eq!(manifest.segments.len(), 1);
        assert_eq!(manifest.timescale, 30000);
        assert!((manifest.duration_seconds() - 0.1).abs() < 1e-6); // 3000/30000 = 0.1s
    }

    #[test]
    fn cmaf_video_muxer_options_default_matches_legacy_new() {
        // Calling `new()` and `new_with_options(..., default())` must
        // produce byte-identical first-segment output. This is the
        // contract that lets every existing call site stay on `new()`
        // unmodified.
        let dir_a = tempfile::tempdir().unwrap();
        let dir_b = tempfile::tempdir().unwrap();
        let mut ma = CmafVideoMuxer::new(
            dir_a.path(),
            1280,
            720,
            30000,
            ColorMetadata::default(),
        )
        .unwrap();
        let mut mb = CmafVideoMuxer::new_with_options(
            dir_b.path(),
            1280,
            720,
            30000,
            ColorMetadata::default(),
            CmafVideoMuxerOptions::default(),
        )
        .unwrap();

        let mut kf = synthetic_seq_header_packet();
        kf.extend_from_slice(&[0xDE, 0xAD]);
        ma.add_packet(kf.clone(), 1500, true).unwrap();
        mb.add_packet(kf, 1500, true).unwrap();

        let info_a = ma.flush_segment().unwrap().unwrap();
        let info_b = mb.flush_segment().unwrap().unwrap();
        assert_eq!(info_a.sequence_number, info_b.sequence_number);
        assert_eq!(info_a.duration_ticks, info_b.duration_ticks);
        assert_eq!(
            info_a.path.file_name().unwrap(),
            info_b.path.file_name().unwrap(),
        );
        // Byte-identical moof+mdat — proves no observable difference.
        let bytes_a = std::fs::read(&info_a.path).unwrap();
        let bytes_b = std::fs::read(&info_b.path).unwrap();
        assert_eq!(bytes_a, bytes_b);
        // init.mp4 written in both cases.
        assert!(dir_a.path().join("init.mp4").exists());
        assert!(dir_b.path().join("init.mp4").exists());
    }

    #[test]
    fn cmaf_video_muxer_first_segment_index_offset_writes_correct_filename() {
        // A helper muxer attached at segment 5 of an in-progress rung
        // must produce `seg-00005.m4s` as its first output, not 00001.
        let dir = tempfile::tempdir().unwrap();
        let mut muxer = CmafVideoMuxer::new_with_options(
            dir.path(),
            1280,
            720,
            30000,
            ColorMetadata::default(),
            CmafVideoMuxerOptions {
                first_segment_index: 5,
                first_segment_base_decode_time: 4 * 3000, // 4 prior segments × 3000-tick duration
                write_init_segment: true,
            },
        )
        .unwrap();

        let mut kf = synthetic_seq_header_packet();
        kf.extend_from_slice(&[0xCA, 0xFE]);
        muxer.add_packet(kf, 1500, true).unwrap();
        muxer
            .add_packet(synthetic_seq_header_packet(), 1500, false)
            .unwrap();

        let info = muxer.flush_segment().unwrap().unwrap();
        assert_eq!(
            info.sequence_number, 5,
            "first flush of an offset muxer must produce segment number 5",
        );
        assert_eq!(info.path.file_name().unwrap(), "seg-00005.m4s");

        // Second flush continues the sequence at 6.
        let mut kf2 = synthetic_seq_header_packet();
        kf2.extend_from_slice(&[0xBE, 0xEF]);
        muxer.add_packet(kf2, 1500, true).unwrap();
        let info2 = muxer.flush_segment().unwrap().unwrap();
        assert_eq!(info2.sequence_number, 6);
        assert_eq!(info2.path.file_name().unwrap(), "seg-00006.m4s");
    }

    #[test]
    fn cmaf_video_muxer_offset_base_decode_time_propagates_to_tfdt() {
        // Verifies the `tfdt` box of the offset muxer's first segment
        // carries the configured base_decode_time. Without this, an
        // HLS player would see segment 5 starting at decode-time 0,
        // producing a buffer underrun at the cut from primary's
        // segment 4 to helper's segment 5.
        let dir = tempfile::tempdir().unwrap();
        let mut muxer = CmafVideoMuxer::new_with_options(
            dir.path(),
            1280,
            720,
            30000,
            ColorMetadata::default(),
            CmafVideoMuxerOptions {
                first_segment_index: 5,
                first_segment_base_decode_time: 4 * 3000,
                write_init_segment: true,
            },
        )
        .unwrap();

        let mut kf = synthetic_seq_header_packet();
        kf.extend_from_slice(&[0x01, 0x02]);
        muxer.add_packet(kf, 1500, true).unwrap();
        let info = muxer.flush_segment().unwrap().unwrap();

        // Walk the segment bytes: moof > traf > tfdt. tfdt v1 layout:
        //   8 bytes box header (size + 'tfdt')
        //   1 byte version (=1) + 3 bytes flags
        //   8 bytes base_media_decode_time (u64 BE)
        let bytes = std::fs::read(&info.path).unwrap();
        let moof_size = read_be_u32(&bytes, 0) as usize;
        let moof = &bytes[..moof_size];
        let traf = find_box(&moof[8..], b"traf").expect("moof has traf");
        let tfdt = find_box(&traf[8..], b"tfdt").expect("traf has tfdt");
        let version = tfdt[8];
        assert_eq!(version, 1, "tfdt should be version 1 (u64 decode time)");
        let dt = u64::from_be_bytes([
            tfdt[12], tfdt[13], tfdt[14], tfdt[15], tfdt[16], tfdt[17], tfdt[18], tfdt[19],
        ]);
        assert_eq!(
            dt, 12000,
            "tfdt base_media_decode_time must equal configured offset (4×3000)",
        );
    }

    #[test]
    fn cmaf_video_muxer_write_init_false_skips_init_file() {
        // A helper muxer must NOT write init.mp4 — the primary owns
        // that file. Verify that flush_segment + finalize do not
        // create init.mp4 in the output directory.
        let dir = tempfile::tempdir().unwrap();
        let mut muxer = CmafVideoMuxer::new_with_options(
            dir.path(),
            1280,
            720,
            30000,
            ColorMetadata::default(),
            CmafVideoMuxerOptions {
                first_segment_index: 5,
                first_segment_base_decode_time: 4 * 3000,
                write_init_segment: false,
            },
        )
        .unwrap();

        let mut kf = synthetic_seq_header_packet();
        kf.extend_from_slice(&[0x03, 0x04]);
        muxer.add_packet(kf, 1500, true).unwrap();
        let info = muxer.flush_segment().unwrap().unwrap();
        assert!(
            info.path.exists(),
            "segment file must be written even when init is skipped",
        );
        let init_path = dir.path().join("init.mp4");
        assert!(
            !init_path.exists(),
            "init.mp4 must NOT be written when write_init_segment=false",
        );

        // finalize must also not write init.
        let _ = muxer.finalize().unwrap();
        assert!(
            !init_path.exists(),
            "finalize must not retroactively write init.mp4 when disabled",
        );
    }

    #[test]
    fn cmaf_video_muxer_two_writers_share_output_dir_with_distinct_indices() {
        // The actual helper-task contract: primary writes segments
        // 1..3 + init.mp4 into dir/. Helper writes segments 3..5 into
        // the same dir with write_init_segment=false. After both
        // finalize, all 4 segment files plus init.mp4 exist.
        let dir = tempfile::tempdir().unwrap();

        let mut primary = CmafVideoMuxer::new(
            dir.path(),
            1280,
            720,
            30000,
            ColorMetadata::default(),
        )
        .unwrap();
        let mut helper = CmafVideoMuxer::new_with_options(
            dir.path(),
            1280,
            720,
            30000,
            ColorMetadata::default(),
            CmafVideoMuxerOptions {
                first_segment_index: 3,
                first_segment_base_decode_time: 2 * 3000,
                write_init_segment: false,
            },
        )
        .unwrap();

        // Primary writes segments 1 and 2.
        for _ in 0..2 {
            let mut kf = synthetic_seq_header_packet();
            kf.extend_from_slice(&[0xAA, 0xBB]);
            primary.add_packet(kf, 1500, true).unwrap();
            primary
                .add_packet(synthetic_seq_header_packet(), 1500, false)
                .unwrap();
            primary.flush_segment().unwrap().unwrap();
        }
        // Helper writes segments 3 and 4.
        for _ in 0..2 {
            let mut kf = synthetic_seq_header_packet();
            kf.extend_from_slice(&[0xCC, 0xDD]);
            helper.add_packet(kf, 1500, true).unwrap();
            helper
                .add_packet(synthetic_seq_header_packet(), 1500, false)
                .unwrap();
            helper.flush_segment().unwrap().unwrap();
        }

        primary.finalize().unwrap();
        helper.finalize().unwrap();

        // All four segments + one init.mp4 present.
        for seg_idx in 1..=4 {
            let p = dir.path().join(format!("seg-{seg_idx:05}.m4s"));
            assert!(p.exists(), "segment {seg_idx} missing at {}", p.display());
        }
        let init_path = dir.path().join("init.mp4");
        assert!(init_path.exists(), "primary's init.mp4 must be present");
    }

    #[test]
    #[should_panic(expected = "first_segment_index is 1-based")]
    fn cmaf_video_muxer_first_segment_index_zero_panics() {
        let dir = tempfile::tempdir().unwrap();
        let _ = CmafVideoMuxer::new_with_options(
            dir.path(),
            1280,
            720,
            30000,
            ColorMetadata::default(),
            CmafVideoMuxerOptions {
                first_segment_index: 0,
                first_segment_base_decode_time: 0,
                write_init_segment: true,
            },
        );
    }

    #[test]
    fn cmaf_video_muxer_rejects_segment_starting_on_non_keyframe() {
        let dir = tempfile::tempdir().unwrap();
        let mut muxer =
            CmafVideoMuxer::new(dir.path(), 640, 360, 30000, ColorMetadata::default()).unwrap();
        muxer
            .add_packet(synthetic_seq_header_packet(), 1500, false)
            .unwrap();
        let err = muxer
            .flush_segment()
            .expect_err("must fail when first sample is not sync");
        assert!(err.to_string().contains("must start with a sync sample"));
    }

    #[test]
    fn cmaf_audio_muxer_emits_init_and_segments_with_correct_durations() {
        let info = AudioInfo {
            codec: "aac".into(),
            sample_rate: 48000,
            channels: 2,
            timescale: 48000,
            asc_bytes: vec![0x12, 0x10],
            codec_private: vec![],
        };
        let dir = tempfile::tempdir().unwrap();
        let mut muxer = CmafAudioMuxer::new(dir.path(), info).unwrap();

        // 5 AAC frames at 1024 samples each = 5120 ticks @ 48 kHz =
        // ~107 ms total.
        for _ in 0..5 {
            muxer.add_packet(vec![0xDE; 256], 1024).unwrap();
        }
        let seg = muxer
            .flush_segment()
            .unwrap()
            .expect("audio segment emitted");
        assert_eq!(seg.duration_ticks, 5 * 1024);
        assert!(seg.path.exists());
        let init_path = dir.path().join("init.mp4");
        assert!(init_path.exists());

        // Audio segment moof should NOT contain a first_sample_flags
        // slot — the trun layout for audio omits that flag bit. We
        // already cover this in `moof_audio_does_not_emit_first_sample_flags`;
        // here we just verify the file shape is valid.
        let bytes = std::fs::read(&seg.path).unwrap();
        assert_eq!(&bytes[4..8], b"moof");

        let manifest = muxer.finalize().unwrap();
        assert_eq!(manifest.timescale, 48000);
        assert!((manifest.duration_seconds() - (5.0 * 1024.0 / 48000.0)).abs() < 1e-6);
    }

    #[test]
    fn mvex_wraps_mehd_and_one_or_more_trex_in_order() {
        let mehd = build_mehd(10_000);
        let trex_v = build_trex(1, SampleFlags::delta_frame().pack());
        let trex_a = build_trex(2, SampleFlags::keyframe().pack());
        let mvex = build_mvex(&mehd, &[trex_v.clone(), trex_a.clone()]);
        let (size, kind) = box_size_and_type(&mvex);
        assert_eq!(size as usize, mvex.len());
        assert_eq!(kind, b"mvex");
        // 8 (header) + mehd(20) + trex(32) + trex(32) = 92.
        assert_eq!(mvex.len(), 8 + mehd.len() + trex_v.len() + trex_a.len());
        // First child is mehd.
        let (_, child0_kind) = box_size_and_type(&mvex[8..]);
        assert_eq!(child0_kind, b"mehd");
        // Second child is the first trex.
        let (_, child1_kind) = box_size_and_type(&mvex[8 + mehd.len()..]);
        assert_eq!(child1_kind, b"trex");
    }
}
