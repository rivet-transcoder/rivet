//! FLAC and ALAC out: their MP4 sample entries, the audio-only MP4 file
//! (`.m4a`) and the native FLAC stream (`.flac`).
//!
//! - `fLaC` + `dfLa`: "Encapsulation of FLAC in ISO Base Media File Format"
//!   (xiph.org). The `dfLa` FullBox holds the stream's metadata blocks,
//!   STREAMINFO first, the last one flagged; the samples are FLAC frames.
//! - `alac` + `alac`: the sample entry holds an `alac` FullBox whose body
//!   is the 24-byte `ALACSpecificConfig`, and for more than two channels a
//!   `chan` box naming the layout.
//!
//! Both put the sample rate in `mdhd` (the timescale) and in their own
//! configuration (STREAMINFO, the cookie's `sampleRate`), which is
//! definitive. A rate above 65535 Hz does not fit the 16.16 `samplerate`
//! field of the sample entry; see [`entry_sample_rate`] for what goes
//! there instead.

use anyhow::{Context, Result, bail};

use super::audio_track::build_audio_trak;
use super::boxes::{BoxBuilder, build_edts};
use super::sample_table::AudioBuildPlan;
use crate::AudioInfo;
use crate::edit::TrackEdit;

/// The fields every AudioSampleEntry opens with (ISO/IEC 14496-12
/// §8.5.2.2), with the sample size and rate these codecs call for.
fn audio_sample_entry_head(b: &mut BoxBuilder, info: &AudioInfo, sample_size: u16) {
    for _ in 0..6 {
        b.u8(0);
    } // reserved
    b.u16(1); // data_reference_index
    b.u32(0); // reserved
    b.u32(0); // reserved
    b.u16(info.channels);
    b.u16(sample_size);
    b.u16(0); // pre_defined
    b.u16(0); // reserved
    b.u32(entry_sample_rate(info.sample_rate) << 16);
}

/// The integer part of the 16.16 `samplerate` field for `rate`: the rate
/// itself up to 65535 Hz; above, its greatest expressible regular division
/// (halved until it fits: 48000 for 96 and 192 kHz, 44100 for 88.2 and
/// 176.4 kHz), or 65535 for a rate with none.
///
/// That is the rule of "Encapsulation of FLAC in ISO Base Media File
/// Format" (xiph.org) for `fLaC`, and the one used for `alac` too: ISO/IEC
/// 14496-12 §12.2.3.1 takes the codec's own output configuration as
/// definitive where it has one — STREAMINFO, the ALAC cookie — and asks
/// only for "sensible values" in the sample entry. Not 0, which MKVToolNix
/// rejects as broken header atoms; and not the §12.2.3 alternative, an
/// `AudioSampleEntryV1` with a `srat` box in a version 1 `stsd`, which the
/// standard says to use only when needed (here it is not: the codec
/// configuration carries the rate) and which QuickTime-lineage readers
/// take for a QuickTime version 1 sound description: `mkvmerge` 102 reads
/// it only with `srat` the first box after the fields, and with `srat`
/// after the codec's box reports 0 channels and a nonsense rate. A
/// QuickTime version 2 sound description (a 64-bit float rate) MediaInfo
/// misreads as 1 Hz. The version 0 entry with the divided rate reads
/// right in both.
pub(super) fn entry_sample_rate(rate: u32) -> u32 {
    let mut r = rate;
    while r > 0xFFFF && r % 2 == 0 {
        r /= 2;
    }
    r.min(0xFFFF)
}

/// The bit depth STREAMINFO names (bits 4..8 of its 13th byte and the top
/// bit of the 14th), for the sample entry's `samplesize`.
fn flac_bits(blocks: &[u8]) -> u16 {
    let si = &blocks[4..];
    u16::from((((si[12] & 1) << 4) | (si[13] >> 4)) + 1)
}

pub(super) fn build_flac_sample_entry(info: &AudioInfo) -> Vec<u8> {
    let mut b = BoxBuilder::new(b"fLaC");
    audio_sample_entry_head(&mut b, info, flac_bits(&info.codec_private));
    let mut dfla = BoxBuilder::new(b"dfLa");
    dfla.u32(0); // version 0, flags 0
    dfla.extend(&info.codec_private);
    b.extend(&dfla.finish());
    b.finish()
}

/// Core Audio layout tags for ALAC's channel layouts (count in the low 16
/// bits): MPEG 3.0 B, MPEG 4.0 B, MPEG 5.0 D, MPEG 5.1 D, AAC 6.1, MPEG 7.1 B.
fn alac_layout_tag(channels: u16) -> Option<u32> {
    let layout = match channels {
        3 => 114,
        4 => 116,
        5 => 120,
        6 => 124,
        7 => 142,
        8 => 127,
        _ => return None,
    };
    Some((layout << 16) | u32::from(channels))
}

pub(super) fn build_alac_sample_entry(info: &AudioInfo) -> Vec<u8> {
    let mut b = BoxBuilder::new(b"alac");
    audio_sample_entry_head(&mut b, info, u16::from(info.codec_private[5]));
    let mut cookie = BoxBuilder::new(b"alac");
    cookie.u32(0); // version 0, flags 0
    cookie.extend(&info.codec_private);
    b.extend(&cookie.finish());
    if let Some(tag) = alac_layout_tag(info.channels) {
        let mut chan = BoxBuilder::new(b"chan");
        chan.u32(0); // version 0, flags 0
        chan.u32(tag);
        chan.u32(0); // channel bitmap
        chan.u32(0); // channel descriptions
        b.extend(&chan.finish());
    }
    b.finish()
}

/// The checks [`crate::mux::Av1Mp4Muxer::check_audio`] makes of a FLAC or
/// ALAC track.
pub(super) fn check_lossless(info: &AudioInfo, flac: bool) -> Result<()> {
    if !(1..=8).contains(&info.channels) {
        bail!("audio mux: {} supports 1..=8 channels; got {}", if flac { "FLAC" } else { "ALAC" }, info.channels);
    }
    if flac {
        let p = &info.codec_private;
        if p.len() < 4 + 34 || p[0] & 0x7F != 0 || p[1..4] != [0, 0, 34] {
            bail!("audio mux: FLAC codec_private must be the metadata blocks, STREAMINFO first");
        }
    } else if info.codec_private.len() != 24 {
        bail!("audio mux: ALAC codec_private (the magic cookie) must be 24 bytes; got {}", info.codec_private.len());
    }
    if info.timescale != info.sample_rate {
        bail!(
            "audio mux: a lossless track's timescale is its sample rate ({}), got {}",
            info.sample_rate,
            info.timescale
        );
    }
    Ok(())
}

/// An audio-only MP4 (`.m4a`): `ftyp`, `moov` with the one track, then
/// `mdat` — the header first, so it plays while it downloads. `samples` are
/// (packet, duration in `info.timescale` ticks); `edit` places them as
/// [`Av1Mp4Muxer::set_audio_edit`](crate::mux::Av1Mp4Muxer::set_audio_edit)
/// does.
pub fn write_audio_mp4(info: &AudioInfo, samples: &[(Vec<u8>, u32)], edit: TrackEdit) -> Result<Vec<u8>> {
    crate::mux::Av1Mp4Muxer::check_audio(info)?;
    if samples.is_empty() {
        bail!("audio mux: no audio samples to write");
    }
    let ts = info.timescale;
    let total: u64 = samples.iter().map(|(_, d)| u64::from(*d)).sum();
    // About a second of audio per chunk.
    let per_chunk = {
        let mean = (total / samples.len() as u64).max(1);
        (u64::from(ts) / mean).clamp(1, 200) as usize
    };
    let plan = AudioBuildPlan {
        info: info.clone(),
        sample_sizes: samples.iter().map(|(p, _)| p.len() as u32).collect(),
        durations: samples.iter().map(|(_, d)| *d).collect(),
        total_duration_in_own_ts: total,
        total_duration_in_movie_ts: total,
        samples_per_chunk: per_chunk as u32,
    };
    let edts = (!edit.is_identity()).then(|| {
        let presented = edit.duration.unwrap_or(total.saturating_sub(edit.media_time));
        (build_edts(edit.delay, edit.media_time, presented), edit.delay + presented)
    });
    let duration = edts.as_ref().map_or(total, |(_, d)| *d);
    let payload: u64 = samples.iter().map(|(p, _)| p.len() as u64).sum();
    let ftyp = {
        let mut b = BoxBuilder::new(b"ftyp");
        b.extend(b"M4A ");
        b.u32(0);
        b.extend(b"M4A ");
        b.extend(b"mp42");
        b.extend(b"isom");
        b.finish()
    };
    let use_co64 = ftyp.len() as u64 + payload + (1 << 20) > u64::from(u32::MAX);
    let mdat_header: u64 = if payload + 8 > u64::from(u32::MAX) { 16 } else { 8 };
    let chunk_starts: Vec<usize> = (0..samples.len()).step_by(per_chunk).collect();
    let build_moov = |mdat_start: u64| -> Vec<u8> {
        let mut offsets = Vec::with_capacity(chunk_starts.len());
        let mut at = mdat_start + mdat_header;
        let mut next = 0usize;
        for (i, (p, _)) in samples.iter().enumerate() {
            if chunk_starts.get(next) == Some(&i) {
                offsets.push(at);
                next += 1;
            }
            at += p.len() as u64;
        }
        let trak = build_audio_trak(&plan, duration, &offsets, use_co64, edts.as_ref().map(|(e, _)| e.as_slice()));
        let mut mvhd = BoxBuilder::new(b"mvhd");
        mvhd.u8(0);
        mvhd.extend(&[0, 0, 0]);
        mvhd.u32(0); // creation_time
        mvhd.u32(0); // modification_time
        mvhd.u32(ts);
        mvhd.u32(duration.min(u64::from(u32::MAX)) as u32);
        mvhd.u32(0x0001_0000); // rate 1.0
        mvhd.u16(0x0100); // volume 1.0
        mvhd.u16(0);
        mvhd.u32(0);
        mvhd.u32(0);
        super::boxes::write_unity_matrix(&mut mvhd);
        for _ in 0..6 {
            mvhd.u32(0);
        }
        mvhd.u32(3); // next_track_ID (the audio track is 2)
        let mut moov = BoxBuilder::new(b"moov");
        moov.extend(&mvhd.finish());
        moov.extend(&trak);
        moov.finish()
    };
    // The chunk offsets depend on the moov's size, which does not depend on
    // their values: size it once, then write it for real.
    let moov_len = build_moov(0).len() as u64;
    let moov = build_moov(ftyp.len() as u64 + moov_len);
    let mut out = Vec::with_capacity(ftyp.len() + moov.len() + payload as usize + 16);
    out.extend_from_slice(&ftyp);
    out.extend_from_slice(&moov);
    if mdat_header == 16 {
        out.extend_from_slice(&1u32.to_be_bytes());
        out.extend_from_slice(b"mdat");
        out.extend_from_slice(&(payload + 16).to_be_bytes());
    } else {
        out.extend_from_slice(&((payload + 8) as u32).to_be_bytes());
        out.extend_from_slice(b"mdat");
    }
    for (p, _) in samples {
        out.extend_from_slice(p);
    }
    Ok(out)
}

/// Seek points every this many seconds of a native FLAC stream.
const SEEK_POINT_SECONDS: u64 = 10;

/// A native FLAC stream: `fLaC`, the metadata blocks (STREAMINFO from
/// `blocks`, a SEEKTABLE with a point every ten seconds on a frame start,
/// and a VORBIS_COMMENT with an empty vendor string and no comments), then
/// the frames. `frames` are (frame, samples in it), in order.
pub fn write_native_flac(blocks: &[u8], frames: &[(Vec<u8>, u32)]) -> Result<Vec<u8>> {
    write_native_flac_with_vendor(blocks, frames, b"")
}

/// [`write_native_flac`] with `vendor` as the VORBIS_COMMENT vendor string,
/// for a caller that wants its files to name what wrote them.
pub fn write_native_flac_with_vendor(blocks: &[u8], frames: &[(Vec<u8>, u32)], vendor: &[u8]) -> Result<Vec<u8>> {
    if blocks.len() < 4 + 34 || blocks[0] & 0x7F != 0 {
        bail!("FLAC: the metadata blocks must open with STREAMINFO");
    }
    let streaminfo = &blocks[4..4 + 34];
    let rate = (u64::from(streaminfo[10]) << 12) | (u64::from(streaminfo[11]) << 4) | (u64::from(streaminfo[12]) >> 4);
    if rate == 0 {
        bail!("FLAC: STREAMINFO sample rate is 0");
    }
    // Seek points: the first frame at or after each multiple of the interval.
    let mut points: Vec<(u64, u64, u16)> = Vec::new();
    let (mut sample, mut offset) = (0u64, 0u64);
    for (f, n) in frames {
        let due = points.len() as u64 * SEEK_POINT_SECONDS * rate;
        if sample >= due {
            points.push((sample, offset, u16::try_from(*n).context("FLAC frame over 65535 samples")?));
        }
        sample += u64::from(*n);
        offset += f.len() as u64;
    }
    let mut out = Vec::with_capacity(offset as usize + 256);
    out.extend_from_slice(b"fLaC");
    out.extend_from_slice(&[0, 0, 0, 34]); // STREAMINFO, not last
    out.extend_from_slice(streaminfo);
    out.push(3); // SEEKTABLE
    out.extend_from_slice(&((points.len() * 18) as u32).to_be_bytes()[1..]);
    for (s, o, n) in &points {
        out.extend_from_slice(&s.to_be_bytes());
        out.extend_from_slice(&o.to_be_bytes());
        out.extend_from_slice(&n.to_be_bytes());
    }
    let comment_len = 4 + vendor.len() + 4;
    out.push(0x80 | 4); // VORBIS_COMMENT, last
    out.extend_from_slice(&(comment_len as u32).to_be_bytes()[1..]);
    out.extend_from_slice(&(vendor.len() as u32).to_le_bytes());
    out.extend_from_slice(vendor);
    out.extend_from_slice(&0u32.to_le_bytes()); // no comments
    for (f, _) in frames {
        out.extend_from_slice(f);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flac_info() -> AudioInfo {
        let mut blocks = vec![0x80, 0, 0, 34];
        let mut si = [0u8; 34];
        si[0..4].copy_from_slice(&[0x10, 0, 0x10, 0]);
        // 96000 Hz, 2 channels, 24 bits.
        let packed: u64 = (96_000u64 << 44) | (1 << 41) | (23 << 36);
        si[10..18].copy_from_slice(&packed.to_be_bytes());
        blocks.extend_from_slice(&si);
        AudioInfo::flac(96_000, 2, blocks)
    }

    #[test]
    fn a_flac_entry_carries_dfla_and_the_depth() {
        let e = build_flac_sample_entry(&flac_info());
        assert_eq!(&e[4..8], b"fLaC");
        assert_eq!(u16::from_be_bytes([e[24], e[25]]), 2, "channels");
        assert_eq!(u16::from_be_bytes([e[26], e[27]]), 24, "samplesize");
        assert_eq!(u32::from_be_bytes([e[32], e[33], e[34], e[35]]), 48_000 << 16, "96 kHz: its half");
        assert_eq!(&e[36 + 4..36 + 8], b"dfLa");
        assert_eq!(e.len(), 36 + 12 + 38);
    }

    #[test]
    fn rates_past_16_16_get_their_greatest_regular_division() {
        for (rate, field) in [
            (8_000, 8_000),
            (44_100, 44_100),
            (48_000, 48_000),
            (65_535, 65_535),
            (88_200, 44_100),
            (96_000, 48_000),
            (176_400, 44_100),
            (192_000, 48_000),
            (352_800, 44_100),
            (384_000, 48_000),
            (705_600, 44_100),
            (768_000, 48_000),
            (65_536, 32_768),
            (100_001, 65_535),
        ] {
            assert_eq!(entry_sample_rate(rate), field, "{rate} Hz");
        }
    }

    #[test]
    fn an_alac_entry_above_65535_hz_names_a_rate() {
        for (rate, field) in [(96_000u32, 48_000u32), (192_000, 48_000), (176_400, 44_100), (44_100, 44_100)] {
            let mut cookie = vec![0u8; 24];
            cookie[0..4].copy_from_slice(&4096u32.to_be_bytes());
            cookie[5] = 24;
            cookie[9] = 2;
            cookie[20..24].copy_from_slice(&rate.to_be_bytes());
            let e = build_alac_sample_entry(&AudioInfo::alac(rate, 2, cookie.clone()));
            assert_eq!(&e[4..8], b"alac");
            assert_eq!(&e[16..18], &[0, 0], "{rate}: a version 0 AudioSampleEntry");
            assert_eq!(u16::from_be_bytes([e[24], e[25]]), 2, "{rate}: channels");
            assert_eq!(u16::from_be_bytes([e[26], e[27]]), 24, "{rate}: samplesize");
            assert_eq!(u32::from_be_bytes([e[32], e[33], e[34], e[35]]), field << 16, "{rate}: samplerate");
            assert_eq!(&e[36 + 4..36 + 8], b"alac");
            assert_eq!(&e[36 + 12..36 + 36], &cookie[..], "{rate}: the cookie, with the true rate");
        }
    }

    #[test]
    fn an_audio_only_mp4_reads_back() {
        let info = flac_info();
        let samples: Vec<(Vec<u8>, u32)> = (0..50).map(|i| (vec![i as u8; 100 + i], 4096)).collect();
        let file = write_audio_mp4(&info, &samples, TrackEdit::default()).unwrap();
        let track = crate::demux::audio::lossless::extract_mp4_lossless(&file).expect("reads back");
        assert_eq!(track.codec, "flac");
        assert_eq!(track.timescale, 96_000);
        assert_eq!(track.samples.len(), 50);
        assert_eq!(track.samples[7], samples[7].0);
        assert!(track.durations.iter().all(|&d| d == 4096));
    }

    #[test]
    fn a_native_stream_gets_seek_points_on_frame_starts() {
        let info = flac_info();
        let frames: Vec<(Vec<u8>, u32)> = (0..600).map(|_| (vec![0u8; 1000], 4096)).collect();
        let file = write_native_flac(&info.codec_private, &frames).unwrap();
        assert_eq!(&file[..4], b"fLaC");
        assert_eq!(file[4 + 4 + 34], 3, "SEEKTABLE follows STREAMINFO");
        let n = u32::from_be_bytes([0, file[43], file[44], file[45]]) as usize / 18;
        // 600 frames of 4096 at 96 kHz is 25.6 s: points at 0, 10 and 20 s.
        assert_eq!(n, 3);
        let second = &file[46 + 18..46 + 36];
        let sample = u64::from_be_bytes(second[..8].try_into().unwrap());
        assert!(sample >= 960_000 && sample % 4096 == 0, "{sample}");
    }
}
