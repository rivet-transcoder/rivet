//! Init-segment writers and `mvex` / `trex` / `mehd` helpers.
//!
//! Produces ISO 14496-12 `ftyp + moov` init segments for CMAF video and
//! audio tracks.  The `moov` contains a single `trak` plus an `mvex` box
//! (with `mehd` + one `trex`) that tells parsers this MP4 is fragmented.
//! Sample tables inside `moov` are left intentionally empty; samples arrive
//! in subsequent `moof` boxes.

use codec::frame::ColorMetadata;

use crate::mux::{
    build_audio_stsd, build_av01, write_unity_matrix, BoxBuilder,
};
use crate::AudioInfo;

use super::{brand, SampleFlags};

// =====================================================================
// mvex / mehd / trex
// =====================================================================

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

// =====================================================================
// Init-segment entry points
// =====================================================================

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
    let av01 = build_av01(width, height, config_obus, color_metadata);
    build_init_segment_video_with_entry(width, height, timescale, &av01, b"av01")
}

/// Build a CMAF video init segment from a pre-built **visual sample entry**
/// (`av01` / `avc1` / `avc3` / `hvc1` / `hev1`, with its config box + colr
/// already inside) and the `ftyp` codec brand. Codec-agnostic — the caller
/// constructs the sample entry for AV1 / H.264 / H.265.
pub fn build_init_segment_video_with_entry(
    width: u32,
    height: u32,
    timescale: u32,
    sample_entry: &[u8],
    codec_brand: &[u8; 4],
) -> Vec<u8> {
    let track_id = 1u32;

    let ftyp = build_ftyp_video(codec_brand);

    // moov children
    let mvhd = build_mvhd(timescale, /* duration */ 0, /* next_track_id */ 2);
    let trak = build_video_trak(width, height, timescale, track_id, sample_entry);
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

// =====================================================================
// ftyp helpers
// =====================================================================

/// `ftyp` for a video init segment. Brands declare `cmfc` (CMAF video
/// constraints), the codec brand (`av01` / `avc1` / `hvc1`), plus `iso6` /
/// `mp42` / `iso2` for broad parser compatibility. Major brand is `iso6` (CMAF /
/// 14496-12 edition 6) — Apple's player and ffmpeg both honour it.
fn build_ftyp_video(codec_brand: &[u8; 4]) -> Vec<u8> {
    let mut b = BoxBuilder::new(b"ftyp");
    b.extend(b"iso6"); // major_brand
    b.u32(0); // minor_version
    b.extend(b"iso6");
    b.extend(b"iso2");
    b.extend(b"mp42");
    b.extend(brand::CMFC);
    b.extend(codec_brand);
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

// =====================================================================
// moov / mvhd
// =====================================================================

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

// =====================================================================
// Video trak tree
// =====================================================================

fn build_video_trak(
    width: u32,
    height: u32,
    timescale: u32,
    track_id: u32,
    sample_entry: &[u8],
) -> Vec<u8> {
    let tkhd = build_video_tkhd(width, height, track_id);
    let mdia = build_video_mdia(timescale, sample_entry);
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

fn build_video_mdia(timescale: u32, sample_entry: &[u8]) -> Vec<u8> {
    let mdhd = build_mdhd(timescale, 0);
    let hdlr = build_hdlr(b"vide", "VideoHandler\0");
    let minf = build_video_minf(sample_entry);
    let mut b = BoxBuilder::new(b"mdia");
    b.extend(&mdhd);
    b.extend(&hdlr);
    b.extend(&minf);
    b.finish()
}

fn build_video_minf(sample_entry: &[u8]) -> Vec<u8> {
    let vmhd = build_vmhd();
    let dinf = build_dinf();
    let stbl = build_video_stbl_empty(sample_entry);
    let mut b = BoxBuilder::new(b"minf");
    b.extend(&vmhd);
    b.extend(&dinf);
    b.extend(&stbl);
    b.finish()
}

/// Empty sample tables for a CMAF video init: `stsd` has the av01
/// sample entry (with av1C, colr, optional mdcv/clli) and the rest of
/// the tables are empty boxes (entry_count=0).
fn build_video_stbl_empty(sample_entry: &[u8]) -> Vec<u8> {
    let stsd = {
        let mut b = BoxBuilder::new(b"stsd");
        b.u8(0);
        b.extend(&[0, 0, 0]);
        b.u32(1); // entry_count
        b.extend(sample_entry);
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

// =====================================================================
// Audio trak tree
// =====================================================================

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

// =====================================================================
// Shared structural helpers
// =====================================================================

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

/// Empty FullBox with version 0 + flags 0 + entry_count 0. Layout:
///   size:u32 = 16 | type | version:u8 = 0 | flags:u24 = 0 | entry_count:u32 = 0
fn build_empty_full_box(box_type: &[u8; 4]) -> Vec<u8> {
    let mut b = BoxBuilder::new(box_type);
    b.u8(0);
    b.extend(&[0, 0, 0]);
    b.u32(0);
    b.finish()
}
