//! Fragment-level box writers: `mfhd`, `tfhd`, `tfdt`, `trun`, `traf`, `moof`.
//!
//! Every function here maps to one ISO 14496-12 §8.8 box.  The public ones
//! (`build_mfhd`, `build_tfhd`, `build_tfdt`, `build_moof_video`,
//! `build_moof_audio`) are re-exported from the parent module.  The private
//! helpers (`build_trun_video`, `build_trun_audio`, `build_traf`) are used
//! only by the two `build_moof_*` compositors and stay crate-private.

use crate::mux::BoxBuilder;

use super::{CmafSample, SampleFlags};

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
