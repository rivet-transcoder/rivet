//! Text subtitle tracks out of MP4 / MOV: `tx3g` (3GPP timed text, ffmpeg's
//! `mov_text`) and `wvtt` (WebVTT in ISOBMFF, ISO/IEC 14496-30).
//!
//! The mp4 crate parses `tx3g` sample entries but has no `wvtt`, and either
//! way its sample iterator is built around the video/audio tracks the rest of
//! the demuxer uses. A subtitle track is a few hundred samples at most, so
//! this walks the sample table itself with the crate's box primitives: one
//! pass over `stts` / `stsz` / `stsc` / `stco` per track, then one read per
//! sample. Fragmented files fall back to the shared `moof`/`trun` walker.
//!
//! This is the reader beside [`crate::mux::subtitle_track`], the writer: the
//! round-trip test in this module feeds the muxer's own output back through
//! it.

use crate::demux::subtitle::{SubtitleCue, SubtitleTrack, finish, strip_markup};
use crate::demux::{direct_children, find_box_body, find_direct_child};

use super::streaming::build_fragmented_sample_table;

/// Every `tx3g` / `wvtt` track in the file, in `trak` order. Tracks with no
/// usable cues are left out.
pub(crate) fn extract_mp4_subtitle_tracks(data: &[u8]) -> Vec<SubtitleTrack> {
    let Some(moov) = find_direct_child(data, b"moov") else { return Vec::new() };
    direct_children(moov, b"trak")
        .filter_map(|trak| extract_track(data, trak))
        .collect()
}

fn extract_track(file: &[u8], trak: &[u8]) -> Option<SubtitleTrack> {
    let stbl = find_box_body(trak, &[b"mdia", b"minf", b"stbl"])?;
    let stsd = find_direct_child(stbl, b"stsd")?;
    // stsd: version/flags(4) entry_count(4) then the first sample entry:
    // size(4) fourcc(4).
    let fourcc = stsd.get(12..16)?;
    let codec = match fourcc {
        b"tx3g" => "tx3g",
        b"wvtt" => "webvtt",
        _ => return None,
    };

    let mdhd = find_box_body(trak, &[b"mdia", b"mdhd"])?;
    let (timescale, language) = parse_mdhd(mdhd)?;
    if timescale == 0 {
        return None;
    }

    let samples = match static_samples(stbl) {
        Some(s) if !s.is_empty() => s,
        _ => {
            // No static sample table: a fragmented file. `tkhd` carries the
            // track id the `traf`s are keyed by.
            let tkhd = find_direct_child(trak, b"tkhd")?;
            let track_id = tkhd_track_id(tkhd)?;
            build_fragmented_sample_table(file, track_id, 0, 0)?
                .into_iter()
                .map(|s| SampleRef {
                    offset: s.offset,
                    size: s.size,
                    start: s.pts_ticks.max(0) as u64,
                    duration: s.duration_ticks,
                })
                .collect()
        }
    };

    let mut cues = Vec::with_capacity(samples.len());
    for s in samples {
        let end = s.offset.checked_add(s.size as u64)?;
        let Some(bytes) = file.get(s.offset as usize..end as usize) else {
            tracing::warn!(offset = s.offset, size = s.size, "subtitle sample past end of file; stopping");
            break;
        };
        let text = match codec {
            "tx3g" => tx3g_sample_text(bytes),
            _ => wvtt_sample_text(bytes),
        };
        let Some(text) = text else { continue };
        let text = match codec {
            // tx3g text is literal: a `<` is a `<`.
            "tx3g" => text.trim().to_string(),
            _ => strip_markup(&text, "webvtt"),
        };
        if text.is_empty() {
            continue;
        }
        cues.push(SubtitleCue { start: s.start, duration: s.duration.max(1), text });
    }
    finish(codec, cues, timescale, language)
}

/// One sample's place in the file and on the timeline.
struct SampleRef {
    offset: u64,
    size: u32,
    start: u64,
    duration: u32,
}

/// `(timescale, language)` out of an `mdhd` body (ISO/IEC 14496-12 §8.4.2).
fn parse_mdhd(mdhd: &[u8]) -> Option<(u32, String)> {
    let version = *mdhd.first()?;
    // v0: 4 vf + 4 creation + 4 modification + 4 timescale + 4 duration;
    // v1 widens the times to 8 bytes.
    let (ts_at, lang_at) = if version == 1 { (20, 32) } else { (12, 20) };
    let timescale = u32::from_be_bytes(mdhd.get(ts_at..ts_at + 4)?.try_into().ok()?);
    let packed = u16::from_be_bytes(mdhd.get(lang_at..lang_at + 2)?.try_into().ok()?);
    Some((timescale, unpack_language(packed)))
}

/// The inverse of the muxer's `pack_language`: three 5-bit letters offset
/// from 0x60. Values below the smallest packed code (`aaa` = 0x421) are
/// QuickTime's Macintosh language numbers, which this doesn't table — those
/// read as `und` rather than as three control characters.
fn unpack_language(packed: u16) -> String {
    let packed = packed & 0x7FFF;
    if packed < 0x421 {
        return "und".to_string();
    }
    let letters = [(packed >> 10) & 0x1F, (packed >> 5) & 0x1F, packed & 0x1F];
    if letters.iter().any(|&l| l == 0 || l > 26) {
        return "und".to_string();
    }
    letters.iter().map(|&l| (0x60 + l as u8) as char).collect()
}

fn tkhd_track_id(tkhd: &[u8]) -> Option<u32> {
    let at = if *tkhd.first()? == 1 { 4 + 8 + 8 } else { 4 + 4 + 4 };
    Some(u32::from_be_bytes(tkhd.get(at..at + 4)?.try_into().ok()?))
}

/// Resolve the static sample table to per-sample `(offset, size, start,
/// duration)`. `None` when a required box is missing or inconsistent.
fn static_samples(stbl: &[u8]) -> Option<Vec<SampleRef>> {
    let be32 = |b: &[u8], at: usize| -> Option<u32> {
        Some(u32::from_be_bytes(b.get(at..at + 4)?.try_into().ok()?))
    };
    let be64 = |b: &[u8], at: usize| -> Option<u64> {
        Some(u64::from_be_bytes(b.get(at..at + 8)?.try_into().ok()?))
    };

    // stsz: version/flags, sample_size, sample_count, [entry_size…].
    let stsz = find_direct_child(stbl, b"stsz")?;
    let fixed_size = be32(stsz, 4)?;
    let count = be32(stsz, 8)? as usize;
    if count == 0 {
        return Some(Vec::new());
    }
    let sizes: Vec<u32> = if fixed_size != 0 {
        vec![fixed_size; count]
    } else {
        (0..count).map(|i| be32(stsz, 12 + 4 * i)).collect::<Option<_>>()?
    };

    // stts: run-length durations.
    let stts = find_direct_child(stbl, b"stts")?;
    let runs = be32(stts, 4)? as usize;
    let mut durations = Vec::with_capacity(count);
    for r in 0..runs {
        let n = be32(stts, 8 + 8 * r)? as usize;
        let d = be32(stts, 12 + 8 * r)?;
        durations.extend(std::iter::repeat_n(d, n.min(count - durations.len())));
        if durations.len() == count {
            break;
        }
    }
    durations.resize(count, 0);

    // Chunk offsets: stco (32-bit) or co64.
    let chunk_offsets: Vec<u64> = if let Some(stco) = find_direct_child(stbl, b"stco") {
        let n = be32(stco, 4)? as usize;
        (0..n).map(|i| be32(stco, 8 + 4 * i).map(u64::from)).collect::<Option<_>>()?
    } else {
        let co64 = find_direct_child(stbl, b"co64")?;
        let n = be32(co64, 4)? as usize;
        (0..n).map(|i| be64(co64, 8 + 8 * i)).collect::<Option<_>>()?
    };

    // stsc: (first_chunk, samples_per_chunk, description_index) runs.
    let stsc = find_direct_child(stbl, b"stsc")?;
    let entries = be32(stsc, 4)? as usize;
    let stsc_runs: Vec<(u32, u32)> = (0..entries)
        .map(|i| Some((be32(stsc, 8 + 12 * i)?, be32(stsc, 12 + 12 * i)?)))
        .collect::<Option<_>>()?;

    let mut out = Vec::with_capacity(count);
    let mut sample = 0usize;
    let mut start: u64 = 0;
    for (chunk_idx, &chunk_offset) in chunk_offsets.iter().enumerate() {
        let chunk_no = chunk_idx as u32 + 1;
        let per_chunk = stsc_runs
            .iter()
            .filter(|(first, _)| *first <= chunk_no)
            .last()
            .map(|(_, spc)| *spc)
            .unwrap_or(0) as usize;
        let mut offset = chunk_offset;
        for _ in 0..per_chunk {
            if sample >= count {
                break;
            }
            let size = sizes[sample];
            let duration = durations[sample];
            out.push(SampleRef { offset, size, start, duration });
            offset = offset.checked_add(size as u64)?;
            start = start.checked_add(duration as u64)?;
            sample += 1;
        }
        if sample >= count {
            break;
        }
    }
    if out.len() != count {
        tracing::warn!(
            expected = count,
            resolved = out.len(),
            "subtitle sample table is inconsistent (stsc/stco); using what resolved"
        );
    }
    Some(out)
}

/// The text of one `tx3g` sample: a big-endian `u16` byte count then UTF-8
/// (3GPP TS 26.245 §5.16); trailing modifier boxes are style and ignored.
/// `None` for an empty sample — the gap filler.
pub(crate) fn tx3g_sample_text(bytes: &[u8]) -> Option<String> {
    let len = u16::from_be_bytes(bytes.get(0..2)?.try_into().ok()?) as usize;
    if len == 0 {
        return None;
    }
    let text = bytes.get(2..2 + len)?;
    Some(String::from_utf8_lossy(text).into_owned())
}

/// The text of one `wvtt` sample (ISO/IEC 14496-30 §6): a run of boxes —
/// `vttc` (a cue, whose `payl` child is the cue text) or `vtte` (nothing on
/// screen). Several `vttc` in one sample are cues shown at once; they join
/// with a line break, since `tx3g` shows one string at a time.
pub(crate) fn wvtt_sample_text(bytes: &[u8]) -> Option<String> {
    let mut lines: Vec<String> = Vec::new();
    for vttc in direct_children(bytes, b"vttc") {
        if let Some(payl) = find_direct_child(vttc, b"payl") {
            let text = String::from_utf8_lossy(payl);
            let text = text.trim_end_matches('\0');
            if !text.is_empty() {
                lines.push(text.to_string());
            }
        }
    }
    (!lines.is_empty()).then(|| lines.join("\n"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mux::Av1Mp4Muxer;
    use bytes::Bytes;
    use frame::EncodedPacket;

    fn cue(start: u64, duration: u32, text: &str) -> SubtitleCue {
        SubtitleCue { start, duration, text: text.into() }
    }

    /// A file from the crate's own muxer: video plus the given subtitle
    /// tracks. The writer and this reader are exact inverses of each other.
    fn mux_with_subtitles(tracks: &[(Vec<SubtitleCue>, u32, &str)]) -> Vec<u8> {
        let mut muxer = Av1Mp4Muxer::new(640, 480, 30.0).unwrap();
        let header: u8 = (1 << 3) | (1 << 1);
        let mut first = vec![header, 5];
        first.extend_from_slice(&[0u8; 5]);
        muxer
            .add_packet(EncodedPacket { data: Bytes::from(first), pts: 0, is_keyframe: true })
            .unwrap();
        for i in 1..30u64 {
            muxer
                .add_packet(EncodedPacket {
                    data: Bytes::from(vec![0xAA; 64]),
                    pts: i,
                    is_keyframe: false,
                })
                .unwrap();
        }
        for (cues, timescale, language) in tracks {
            muxer.add_subtitle_track(cues, *timescale, language).unwrap();
        }
        muxer.finalize().unwrap().to_vec()
    }

    #[test]
    fn tx3g_round_trips_through_the_muxer() {
        let eng = vec![cue(500, 1_500, "Hello, world"), cue(3_000, 2_500, "Second\nline")];
        let deu = vec![cue(1_000, 1_500, "Hallo Welt")];
        let mp4 = mux_with_subtitles(&[(eng.clone(), 1_000, "eng"), (deu.clone(), 1_000, "deu")]);

        let tracks = extract_mp4_subtitle_tracks(&mp4);
        assert_eq!(tracks.len(), 2, "both tracks come back, in order");
        assert_eq!(tracks[0].codec, "tx3g");
        assert_eq!(tracks[0].language, "eng");
        assert_eq!(tracks[0].timescale, 1_000);
        // The gap-filling empty samples the muxer wrote are not cues.
        assert_eq!(tracks[0].cues, eng);
        assert_eq!(tracks[1].language, "deu");
        assert_eq!(tracks[1].cues, deu);
    }

    #[test]
    fn a_file_without_text_tracks_yields_nothing() {
        let mp4 = mux_with_subtitles(&[]);
        assert!(extract_mp4_subtitle_tracks(&mp4).is_empty());
        assert!(extract_mp4_subtitle_tracks(b"not an mp4 at all").is_empty());
    }

    #[test]
    fn mdhd_language_unpacks_both_versions() {
        // eng = (5<<10)|(14<<5)|7
        let packed: u16 = (5 << 10) | (14 << 5) | 7;
        let mut v0 = vec![0u8, 0, 0, 0];
        v0.extend_from_slice(&0u32.to_be_bytes());
        v0.extend_from_slice(&0u32.to_be_bytes());
        v0.extend_from_slice(&1_000u32.to_be_bytes());
        v0.extend_from_slice(&0u32.to_be_bytes());
        v0.extend_from_slice(&packed.to_be_bytes());
        v0.extend_from_slice(&0u16.to_be_bytes());
        assert_eq!(parse_mdhd(&v0), Some((1_000, "eng".into())));

        let mut v1 = vec![1u8, 0, 0, 0];
        v1.extend_from_slice(&0u64.to_be_bytes());
        v1.extend_from_slice(&0u64.to_be_bytes());
        v1.extend_from_slice(&600u32.to_be_bytes());
        v1.extend_from_slice(&0u64.to_be_bytes());
        v1.extend_from_slice(&packed.to_be_bytes());
        v1.extend_from_slice(&0u16.to_be_bytes());
        assert_eq!(parse_mdhd(&v1), Some((600, "eng".into())));

        // Zero and Macintosh language numbers are "undetermined".
        assert_eq!(unpack_language(0), "und");
        assert_eq!(unpack_language(0x0000), "und");
        assert_eq!(unpack_language(9), "und");
        assert_eq!(unpack_language((21 << 10) | (14 << 5) | 4), "und");
    }

    #[test]
    fn tx3g_sample_text_reads_the_length_prefix() {
        assert_eq!(tx3g_sample_text(&[0, 2, b'h', b'i']), Some("hi".into()));
        assert_eq!(tx3g_sample_text(&[0, 0]), None, "an empty sample is a gap");
        // Style modifier boxes after the text are ignored.
        assert_eq!(tx3g_sample_text(&[0, 1, b'x', 0, 0, 0, 8, b's', b't', b'y', b'l']), Some("x".into()));
        assert_eq!(tx3g_sample_text(&[0, 5, b'a']), None, "truncated sample is not text");
    }

    /// Build a `wvtt` sample per ISO/IEC 14496-30 §6.4: boxes with a 32-bit
    /// size and a 4-char type, `vttc { payl }` for a cue, `vtte` for none.
    fn boxed(kind: &[u8; 4], body: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(8 + body.len());
        out.extend_from_slice(&((8 + body.len()) as u32).to_be_bytes());
        out.extend_from_slice(kind);
        out.extend_from_slice(body);
        out
    }

    #[test]
    fn wvtt_sample_text_reads_vttc_payloads_and_skips_vtte() {
        let cue1 = boxed(b"vttc", &[boxed(b"iden", b"1"), boxed(b"payl", b"Hello &amp; <i>bye</i>")].concat());
        let cue2 = boxed(b"vttc", &boxed(b"payl", b"second"));
        let sample = [cue1, cue2].concat();
        assert_eq!(wvtt_sample_text(&sample), Some("Hello &amp; <i>bye</i>\nsecond".into()));
        assert_eq!(wvtt_sample_text(&boxed(b"vtte", &[])), None, "vtte is a gap");
        assert_eq!(wvtt_sample_text(&[]), None);
    }

    #[test]
    fn wvtt_tracks_are_read_from_a_hand_built_sample_table() {
        // No third-party `wvtt` muxer is on this box (ffmpeg's mp4 muxer
        // refuses `-c:s webvtt`), so the track is assembled by hand to the
        // spec's layout: one chunk, three samples — a cue, an empty `vtte`
        // gap, a cue with a tag and an entity — at a 1 kHz timescale.
        let s1 = boxed(b"vttc", &boxed(b"payl", b"First cue"));
        let s2 = boxed(b"vtte", &[]);
        let s3 = boxed(b"vttc", &boxed(b"payl", b"<b>Bold</b> &amp; plain"));
        let durations = [1_000u32, 2_000, 500];
        let mdat_payload = [s1.clone(), s2.clone(), s3.clone()].concat();

        let stsd = {
            // wvtt sample entry: 6 reserved + dref index, then a `vttC` config.
            let mut entry = vec![0u8; 6];
            entry.extend_from_slice(&1u16.to_be_bytes());
            entry.extend_from_slice(&boxed(b"vttC", b"WEBVTT"));
            let mut body = vec![0u8, 0, 0, 0];
            body.extend_from_slice(&1u32.to_be_bytes());
            body.extend_from_slice(&boxed(b"wvtt", &entry));
            boxed(b"stsd", &body)
        };
        let stts = {
            let mut body = vec![0u8, 0, 0, 0];
            body.extend_from_slice(&3u32.to_be_bytes());
            for d in durations {
                body.extend_from_slice(&1u32.to_be_bytes());
                body.extend_from_slice(&d.to_be_bytes());
            }
            boxed(b"stts", &body)
        };
        let stsc = {
            let mut body = vec![0u8, 0, 0, 0];
            body.extend_from_slice(&1u32.to_be_bytes());
            body.extend_from_slice(&1u32.to_be_bytes());
            body.extend_from_slice(&3u32.to_be_bytes());
            body.extend_from_slice(&1u32.to_be_bytes());
            boxed(b"stsc", &body)
        };
        let stsz = {
            let mut body = vec![0u8, 0, 0, 0];
            body.extend_from_slice(&0u32.to_be_bytes());
            body.extend_from_slice(&3u32.to_be_bytes());
            for s in [&s1, &s2, &s3] {
                body.extend_from_slice(&(s.len() as u32).to_be_bytes());
            }
            boxed(b"stsz", &body)
        };
        // The chunk offset is patched once the moov size is known.
        let build = |chunk_offset: u32| -> Vec<u8> {
            let mut stco_body = vec![0u8, 0, 0, 0];
            stco_body.extend_from_slice(&1u32.to_be_bytes());
            stco_body.extend_from_slice(&chunk_offset.to_be_bytes());
            let stbl = boxed(b"stbl", &[stsd.clone(), stts.clone(), stsc.clone(), stsz.clone(), boxed(b"stco", &stco_body)].concat());
            let mdhd = {
                let mut b = vec![0u8, 0, 0, 0];
                b.extend_from_slice(&0u32.to_be_bytes());
                b.extend_from_slice(&0u32.to_be_bytes());
                b.extend_from_slice(&1_000u32.to_be_bytes());
                b.extend_from_slice(&3_500u32.to_be_bytes());
                b.extend_from_slice(&(((4u16) << 10) | (5 << 5) | 21).to_be_bytes()); // deu
                b.extend_from_slice(&0u16.to_be_bytes());
                boxed(b"mdhd", &b)
            };
            let minf = boxed(b"minf", &stbl);
            let mdia = boxed(b"mdia", &[mdhd, minf].concat());
            let tkhd = boxed(b"tkhd", &[0u8; 84]);
            let trak = boxed(b"trak", &[tkhd, mdia].concat());
            let moov = boxed(b"moov", &trak);
            let ftyp = boxed(b"ftyp", b"isom\0\0\0\0isom");
            let mdat = boxed(b"mdat", &mdat_payload);
            [ftyp, moov, mdat].concat()
        };
        let probe = build(0);
        let mdat_start = probe.len() - mdat_payload.len();
        let file = build(mdat_start as u32);

        let tracks = extract_mp4_subtitle_tracks(&file);
        assert_eq!(tracks.len(), 1);
        let t = &tracks[0];
        assert_eq!(t.codec, "webvtt");
        assert_eq!(t.language, "deu");
        assert_eq!(t.timescale, 1_000);
        // The vtte gap is not a cue; the tag is stripped and the entity decoded.
        assert_eq!(t.cues, vec![cue(0, 1_000, "First cue"), cue(3_000, 500, "Bold & plain")]);
    }
}
