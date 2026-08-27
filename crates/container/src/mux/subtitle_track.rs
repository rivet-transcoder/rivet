//! `tx3g` subtitle track — 3GPP timed text (ffmpeg's `mov_text`), MP4's native
//! subtitle format.
//!
//! Structure mirrors [`super::audio_track`]: a `trak` with its own `mdia` /
//! `minf` / `stbl`, sharing the file's single `mdat`. Two things differ from
//! the audio side:
//!
//! - **The timeline has no holes.** A `tx3g` track is a continuous run of
//!   samples, so the gaps between cues have to be filled with *empty* samples
//!   (a zero text length). [`SubtitleBuildPlan::from_cues`] does that.
//! - **It isn't interleaved.** A feature-length subtitle track is a few tens of
//!   kilobytes, so it's written as one contiguous chunk at the end of the mdat
//!   rather than rotated through it. Chunk layout is a seek-performance hint,
//!   not a correctness constraint, and one chunk keeps the interleave planner
//!   a two-track problem.

use crate::demux::subtitle::SubtitleCue;

use super::boxes::BoxBuilder;
use super::boxes::write_unity_matrix;
use super::video_track::build_dinf;
use super::sample_table::{build_stco, build_co64, build_stsc, build_stsz};

/// Track ID of the **first** subtitle track (video = 1, audio = 2); the
/// second subtitle track is 4, and so on. Fixed rather than "one past the
/// audio track" so a file without audio lays out exactly as it always has.
pub(super) const SUBTITLE_TRACK_ID: u32 = 3;

/// Everything `finalize` needs to lay out and describe the subtitle track.
#[derive(Debug, Clone)]
pub(super) struct SubtitleBuildPlan {
    /// Serialized `tx3g` samples, in presentation order.
    pub(super) samples: Vec<Vec<u8>>,
    /// Per-sample duration in `timescale` ticks; parallel to `samples`.
    pub(super) durations: Vec<u32>,
    pub(super) timescale: u32,
    /// ISO-639-2 language code, packed into `mdhd` as three 5-bit letters.
    pub(super) language: String,
}

impl SubtitleBuildPlan {
    /// Turn decoded cues into a gap-free run of `tx3g` samples.
    ///
    /// A `tx3g` timeline is contiguous: sample *n* starts where sample *n-1*
    /// ended. Cues are not — they have silence between them — so every gap
    /// becomes an empty sample. Without those, every cue after the first gap
    /// would show up early by the width of the gap.
    pub(super) fn from_cues(cues: &[SubtitleCue], timescale: u32, language: String) -> Option<Self> {
        if cues.is_empty() {
            return None;
        }
        let mut samples = Vec::with_capacity(cues.len() * 2);
        let mut durations = Vec::with_capacity(cues.len() * 2);
        let mut cursor: u64 = 0;
        for cue in cues {
            if cue.start > cursor {
                samples.push(encode_sample(""));
                durations.push((cue.start - cursor).min(u32::MAX as u64) as u32);
            }
            samples.push(encode_sample(&cue.text));
            durations.push(cue.duration.max(1));
            cursor = cue.start + cue.duration.max(1) as u64;
        }
        Some(Self { samples, durations, timescale, language })
    }

    pub(super) fn sample_sizes(&self) -> Vec<u32> {
        self.samples.iter().map(|s| s.len() as u32).collect()
    }

    pub(super) fn payload_bytes(&self) -> u64 {
        self.samples.iter().map(|s| s.len() as u64).sum()
    }

    /// Total duration in the track's own timescale.
    pub(super) fn total_duration(&self) -> u64 {
        self.durations.iter().map(|&d| d as u64).sum()
    }
}

/// Serialize one `tx3g` sample: a big-endian `u16` byte count followed by the
/// UTF-8 text (3GPP TS 26.245 §5.16). Style modifier boxes are optional and we
/// emit none — the markup was already stripped at demux, and the sample entry's
/// default style covers the whole string.
///
/// The length field counts *bytes*, not characters, so multi-byte UTF-8 is
/// handled by construction. A string longer than `u16::MAX` bytes is truncated
/// on a character boundary rather than corrupting the length prefix.
pub(super) fn encode_sample(text: &str) -> Vec<u8> {
    let mut bytes = text.as_bytes();
    if bytes.len() > u16::MAX as usize {
        let mut cut = u16::MAX as usize;
        while cut > 0 && !text.is_char_boundary(cut) {
            cut -= 1;
        }
        bytes = &bytes[..cut];
    }
    let mut out = Vec::with_capacity(2 + bytes.len());
    out.extend_from_slice(&(bytes.len() as u16).to_be_bytes());
    out.extend_from_slice(bytes);
    out
}

pub(super) fn build_subtitle_trak(
    plan: &SubtitleBuildPlan,
    track_id: u32,
    width: u32,
    height: u32,
    duration_in_movie_ts: u64,
    chunk_offsets: &[u64],
    use_co64: bool,
) -> Vec<u8> {
    let mut b = BoxBuilder::new(b"trak");
    b.extend(&build_subtitle_tkhd(track_id, width, height, duration_in_movie_ts));
    b.extend(&build_subtitle_mdia(plan, width, height, chunk_offsets, use_co64));
    b.finish()
}

fn build_subtitle_tkhd(track_id: u32, width: u32, height: u32, duration_in_movie_ts: u64) -> Vec<u8> {
    let mut b = BoxBuilder::new(b"tkhd");
    b.u8(0); // version
    b.extend(&[0, 0, 0x03]); // track_enabled | track_in_movie
    b.u32(0); // creation_time
    b.u32(0); // modification_time
    b.u32(track_id);
    b.u32(0); // reserved
    b.u32(duration_in_movie_ts as u32);
    b.u32(0); // reserved
    b.u32(0);
    // Subtitles render above the video, so layer is negative (front-most is
    // most negative in ISOBMFF). -1 as a signed 16-bit value.
    b.u16(0xFFFF);
    b.u16(0); // alternate_group
    b.u16(0); // volume (0 for non-audio)
    b.u16(0); // reserved
    write_unity_matrix(&mut b);
    // A text track carries a visual size so a player knows where to lay the
    // box out; match the video's, in 16.16 fixed point.
    b.u32(width << 16);
    b.u32(height << 16);
    b.finish()
}

fn build_subtitle_mdia(
    plan: &SubtitleBuildPlan,
    width: u32,
    height: u32,
    chunk_offsets: &[u64],
    use_co64: bool,
) -> Vec<u8> {
    let mut b = BoxBuilder::new(b"mdia");
    b.extend(&build_subtitle_mdhd(plan));
    b.extend(&build_subtitle_hdlr());
    b.extend(&build_subtitle_minf(plan, width, height, chunk_offsets, use_co64));
    b.finish()
}

/// `mdhd` with the track's language packed in. [`build_mdhd`] hardcodes
/// `undetermined`, and a subtitle track is the one place the language actually
/// matters to a player's track picker.
fn build_subtitle_mdhd(plan: &SubtitleBuildPlan) -> Vec<u8> {
    let mut b = BoxBuilder::new(b"mdhd");
    b.u8(0); // version
    b.extend(&[0, 0, 0]); // flags
    b.u32(0); // creation_time
    b.u32(0); // modification_time
    b.u32(plan.timescale);
    b.u32(plan.total_duration() as u32);
    b.u16(pack_language(&plan.language));
    b.u16(0); // pre_defined
    b.finish()
}

/// ISO-639-2/T as three 5-bit values, each letter offset from `0x60`, with the
/// top bit zero (ISO/IEC 14496-12 §8.4.2). Anything that isn't three ASCII
/// lowercase letters becomes `und`.
fn pack_language(lang: &str) -> u16 {
    let b = lang.as_bytes();
    let valid = b.len() == 3 && b.iter().all(|c| c.is_ascii_lowercase());
    let src: &[u8] = if valid { b } else { b"und" };
    ((src[0] as u16 - 0x60) << 10) | ((src[1] as u16 - 0x60) << 5) | (src[2] as u16 - 0x60)
}

fn build_subtitle_hdlr() -> Vec<u8> {
    let mut b = BoxBuilder::new(b"hdlr");
    b.u8(0); // version
    b.extend(&[0, 0, 0]); // flags
    b.u32(0); // pre_defined
    b.extend(b"sbtl"); // handler_type: subtitle
    b.u32(0); // reserved[0]
    b.u32(0); // reserved[1]
    b.u32(0); // reserved[2]
    b.extend(b"SubtitleHandler\0");
    b.finish()
}

fn build_subtitle_minf(
    plan: &SubtitleBuildPlan,
    width: u32,
    height: u32,
    chunk_offsets: &[u64],
    use_co64: bool,
) -> Vec<u8> {
    let mut b = BoxBuilder::new(b"minf");
    // `nmhd` — the null media header, for tracks that are neither video (vmhd)
    // nor sound (smhd). ISO/IEC 14496-12 §8.4.5.2.
    let mut nmhd = BoxBuilder::new(b"nmhd");
    nmhd.u8(0);
    nmhd.extend(&[0, 0, 0]);
    b.extend(&nmhd.finish());
    b.extend(&build_dinf());
    b.extend(&build_subtitle_stbl(plan, width, height, chunk_offsets, use_co64));
    b.finish()
}

fn build_subtitle_stbl(
    plan: &SubtitleBuildPlan,
    width: u32,
    height: u32,
    chunk_offsets: &[u64],
    use_co64: bool,
) -> Vec<u8> {
    let sizes = plan.sample_sizes();
    let mut b = BoxBuilder::new(b"stbl");
    b.extend(&build_subtitle_stsd(width, height));
    b.extend(&build_subtitle_stts(&plan.durations));
    // One chunk holding every sample — see the module note on why the subtitle
    // track sits outside the interleave rotation.
    b.extend(&build_stsc(sizes.len() as u32, sizes.len() as u32));
    b.extend(&build_stsz(&sizes));
    if use_co64 {
        b.extend(&build_co64(chunk_offsets));
    } else {
        b.extend(&build_stco(chunk_offsets));
    }
    b.finish()
}

/// `stts` from a per-sample duration list, run-length compressed. Subtitle
/// durations are all different, so this is mostly one entry per sample — but
/// the runs of equal-length gaps do compress.
fn build_subtitle_stts(durations: &[u32]) -> Vec<u8> {
    let mut runs: Vec<(u32, u32)> = Vec::new();
    for &d in durations {
        match runs.last_mut() {
            Some((count, dur)) if *dur == d => *count += 1,
            _ => runs.push((1, d)),
        }
    }
    let mut b = BoxBuilder::new(b"stts");
    b.u8(0);
    b.extend(&[0, 0, 0]);
    b.u32(runs.len() as u32);
    for (count, dur) in runs {
        b.u32(count);
        b.u32(dur);
    }
    b.finish()
}

fn build_subtitle_stsd(width: u32, height: u32) -> Vec<u8> {
    let mut b = BoxBuilder::new(b"stsd");
    b.u8(0);
    b.extend(&[0, 0, 0]);
    b.u32(1); // entry_count
    b.extend(&build_tx3g(width, height));
    b.finish()
}

/// The `tx3g` sample entry (3GPP TS 26.245 §5.16). Defaults chosen to match
/// what ffmpeg's `mov_text` muxer writes: bottom-centred white text in a box
/// spanning the video, no background fill.
fn build_tx3g(width: u32, height: u32) -> Vec<u8> {
    let mut b = BoxBuilder::new(b"tx3g");
    // SampleEntry base.
    b.extend(&[0u8; 6]); // reserved
    b.u16(1); // data_reference_index

    b.u32(0); // displayFlags
    b.u8(1); // horizontal-justification: centre
    b.u8(0xFF); // vertical-justification: bottom (-1)
    b.extend(&[0, 0, 0, 0]); // background-color-rgba: transparent

    // BoxRecord default-text-box: top, left, bottom, right. A box the size of
    // the frame lets the player place the text itself.
    b.u16(0);
    b.u16(0);
    b.u16(height.min(u16::MAX as u32) as u16);
    b.u16(width.min(u16::MAX as u32) as u16);

    // StyleRecord default-style: whole-string, font 1, plain, 18pt, opaque white.
    b.u16(0); // startChar
    b.u16(0); // endChar
    b.u16(1); // font-ID
    b.u8(0); // face-style-flags: plain
    b.u8(18); // font-size
    b.extend(&[0xFF, 0xFF, 0xFF, 0xFF]); // text-color-rgba

    // FontTableBox: one entry, "Serif" — the font name 3GPP names as the
    // baseline every renderer is expected to resolve.
    let mut ftab = BoxBuilder::new(b"ftab");
    ftab.u16(1); // entry-count
    ftab.u16(1); // font-ID
    ftab.u8(5); // font-name-length
    ftab.extend(b"Serif");
    b.extend(&ftab.finish());

    b.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cue(start: u64, duration: u32, text: &str) -> SubtitleCue {
        SubtitleCue { start, duration, text: text.into() }
    }

    #[test]
    fn sample_is_a_be_length_prefix_plus_utf8() {
        assert_eq!(encode_sample("hi"), vec![0x00, 0x02, b'h', b'i']);
        // Empty sample — what a gap looks like on the wire.
        assert_eq!(encode_sample(""), vec![0x00, 0x00]);
        // Length counts bytes, not chars.
        let s = encode_sample("é");
        assert_eq!(&s[..2], &[0x00, 0x02]);
        assert_eq!(s.len(), 4);
    }

    #[test]
    fn gaps_between_cues_become_empty_samples() {
        // A cue at t=0 and another at t=5000 with a hole between them.
        let plan = SubtitleBuildPlan::from_cues(
            &[cue(0, 1_000, "a"), cue(5_000, 1_000, "b")],
            1_000,
            "eng".into(),
        )
        .unwrap();
        // a | gap | b
        assert_eq!(plan.samples.len(), 3);
        assert_eq!(plan.durations, vec![1_000, 4_000, 1_000]);
        assert_eq!(plan.samples[1], encode_sample(""));
        // The timeline is contiguous and ends where the last cue ends.
        assert_eq!(plan.total_duration(), 6_000);
    }

    #[test]
    fn a_leading_gap_is_filled_too() {
        // Without this the first cue would appear at t=0 instead of t=2s.
        let plan =
            SubtitleBuildPlan::from_cues(&[cue(2_000, 500, "x")], 1_000, "und".into()).unwrap();
        assert_eq!(plan.samples.len(), 2);
        assert_eq!(plan.durations, vec![2_000, 500]);
        assert_eq!(plan.samples[0], encode_sample(""));
    }

    #[test]
    fn back_to_back_cues_need_no_filler() {
        let plan = SubtitleBuildPlan::from_cues(
            &[cue(0, 1_000, "a"), cue(1_000, 1_000, "b")],
            1_000,
            "und".into(),
        )
        .unwrap();
        assert_eq!(plan.samples.len(), 2);
        assert_eq!(plan.durations, vec![1_000, 1_000]);
    }

    #[test]
    fn language_packs_to_iso639_2() {
        // 'e'=0x65 → 5, 'n'=0x6E → 14, 'g'=0x67 → 7
        assert_eq!(pack_language("eng"), (5 << 10) | (14 << 5) | 7);
        assert_eq!(pack_language("und"), (21 << 10) | (14 << 5) | 4);
        // Anything malformed falls back to und rather than emitting garbage.
        assert_eq!(pack_language("EN"), pack_language("und"));
        assert_eq!(pack_language(""), pack_language("und"));
        assert_eq!(pack_language("english"), pack_language("und"));
        // The top bit must be clear.
        assert_eq!(pack_language("zzz") & 0x8000, 0);
    }

    #[test]
    fn stts_run_length_compresses_equal_durations() {
        // 3 samples of 1000 then 1 of 500 → two entries, not four.
        let stts = build_subtitle_stts(&[1_000, 1_000, 1_000, 500]);
        // header: size(4) type(4) version+flags(4) entry_count(4) = 16
        let entry_count = u32::from_be_bytes(stts[12..16].try_into().unwrap());
        assert_eq!(entry_count, 2);
        assert_eq!(u32::from_be_bytes(stts[16..20].try_into().unwrap()), 3);
        assert_eq!(u32::from_be_bytes(stts[20..24].try_into().unwrap()), 1_000);
    }

    #[test]
    fn tx3g_entry_is_well_formed() {
        let e = build_tx3g(1920, 1080);
        let size = u32::from_be_bytes(e[..4].try_into().unwrap()) as usize;
        assert_eq!(size, e.len(), "declared box size must match the bytes written");
        assert_eq!(&e[4..8], b"tx3g");
        // The nested ftab must also be self-consistent.
        let ftab_at = e.len() - 19; // ftab: 8 header + 2 + 2 + 1 + 5 = 18… find it
        let _ = ftab_at;
        let pos = e.windows(4).position(|w| w == b"ftab").expect("ftab present");
        let ftab_size = u32::from_be_bytes(e[pos - 4..pos].try_into().unwrap()) as usize;
        assert_eq!(pos - 4 + ftab_size, e.len(), "ftab must be the last box and fit exactly");
    }

    #[test]
    fn no_cues_means_no_plan() {
        assert!(SubtitleBuildPlan::from_cues(&[], 1_000, "und".into()).is_none());
    }
}
