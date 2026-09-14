//! MP4 / MOV edit lists: `trak > edts > elst` (ISO/IEC 14496-12 §8.6.6),
//! read from the box bytes and reduced to [`crate::edit`]'s types.
//!
//! # What is implemented, and what is refused
//!
//! An edit list is a sequence of entries, each mapping `segment_duration` of
//! the movie timeline (in the `mvhd` timescale) to media starting at
//! `media_time` (in the track's `mdhd` timescale) played at `media_rate`;
//! `media_time = -1` is an *empty* edit, time with nothing presented. Two shapes
//! cover what real files carry:
//!
//! - one media edit: presentation starts at `media_time` (a trim's hidden
//!   lead-in, a B-frame composition shift, AAC priming) and lasts
//!   `segment_duration` (`0` = to the end of the media);
//! - one empty edit followed by one media edit: the same, starting late.
//!
//! Anything else — a rate other than 1 (slow motion, a dwell), a gap in the
//! middle, two media segments (an edit-decision list) — is refused with the
//! shape named. Guessing at those would put the wrong frames on screen with no
//! error; refusing says what to fix.

use anyhow::{Result, bail};

use crate::edit::{AudioEdit, VideoPresentation, rescale_round};

/// One `elst` entry as stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct EditEntry {
    /// Length of this edit on the movie timeline, `mvhd` ticks.
    pub(crate) segment_duration: u64,
    /// Start of this edit in the media, `mdhd` ticks; `-1` is an empty edit.
    pub(crate) media_time: i64,
    /// Integer part of the playback rate.
    pub(crate) rate_integer: i16,
    /// Fractional part of the playback rate (1/65536ths).
    pub(crate) rate_fraction: i16,
}

/// Parse an `elst` body (the bytes after the 8-byte box header).
pub(crate) fn parse_elst(body: &[u8]) -> Result<Vec<EditEntry>> {
    if body.len() < 8 {
        bail!("elst: {} bytes, shorter than its 8-byte header", body.len());
    }
    let version = body[0];
    let count = u32::from_be_bytes([body[4], body[5], body[6], body[7]]) as usize;
    let entry_len = match version {
        0 => 12,
        1 => 20,
        v => bail!("elst: version {v} is not defined (0 or 1)"),
    };
    let needed = count.checked_mul(entry_len).and_then(|n| n.checked_add(8));
    if needed.is_none_or(|n| n > body.len()) {
        bail!("elst: {count} entries need {} bytes, the box has {}", count.saturating_mul(entry_len) + 8, body.len());
    }
    let mut entries = Vec::with_capacity(count);
    for i in 0..count {
        let e = &body[8 + i * entry_len..8 + (i + 1) * entry_len];
        let (segment_duration, media_time, rate) = if version == 1 {
            (
                u64::from_be_bytes(e[0..8].try_into().expect("8 bytes")),
                i64::from_be_bytes(e[8..16].try_into().expect("8 bytes")),
                &e[16..20],
            )
        } else {
            (
                u64::from(u32::from_be_bytes(e[0..4].try_into().expect("4 bytes"))),
                i64::from(i32::from_be_bytes(e[4..8].try_into().expect("4 bytes"))),
                &e[8..12],
            )
        };
        entries.push(EditEntry {
            segment_duration,
            media_time,
            rate_integer: i16::from_be_bytes([rate[0], rate[1]]),
            rate_fraction: i16::from_be_bytes([rate[2], rate[3]]),
        });
    }
    Ok(entries)
}

/// A timescale field of a full box whose version-0 layout puts it after
/// creation and modification times of 4 bytes, and version-1 after 8
/// (`mvhd`, `mdhd`).
fn timescale_of(body: &[u8]) -> Option<u32> {
    let at = if *body.first()? == 1 { 4 + 8 + 8 } else { 4 + 4 + 4 };
    Some(u32::from_be_bytes(body.get(at..at + 4)?.try_into().ok()?))
}

/// The `tkhd` track id (after the same times as [`timescale_of`]).
fn track_id_of(tkhd: &[u8]) -> Option<u32> {
    timescale_of(tkhd)
}

/// A track's raw edit list with the two timescales it is written in.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TrackEditList {
    pub(crate) entries: Vec<EditEntry>,
    /// `mvhd` timescale: the unit of `segment_duration`.
    pub(crate) movie_timescale: u32,
    /// The track's `mdhd` timescale: the unit of `media_time`.
    pub(crate) media_timescale: u32,
}

/// The edit list of the track with `track_id` in an MP4's box tree, `None` when
/// that track has none (or the file has no such track).
pub(crate) fn track_edit_list(data: &[u8], track_id: u32) -> Result<Option<TrackEditList>> {
    use super::super::{direct_children, find_box_body, find_direct_child};
    let Some(moov) = find_direct_child(data, b"moov") else { return Ok(None) };
    let Some(movie_timescale) = find_direct_child(moov, b"mvhd").and_then(timescale_of) else {
        return Ok(None);
    };
    for trak in direct_children(moov, b"trak") {
        if find_direct_child(trak, b"tkhd").and_then(track_id_of) != Some(track_id) {
            continue;
        }
        let Some(elst) = find_box_body(trak, &[b"edts", b"elst"]) else { return Ok(None) };
        let Some(media_timescale) = find_box_body(trak, &[b"mdia", b"mdhd"]).and_then(timescale_of) else {
            return Ok(None);
        };
        let entries = parse_elst(elst)?;
        return Ok(Some(TrackEditList { entries, movie_timescale, media_timescale }));
    }
    Ok(None)
}

/// An edit list reduced to the implemented shapes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct EditTimeline {
    /// `mvhd` ticks per second.
    pub(crate) movie_timescale: u32,
    /// `mdhd` ticks per second.
    pub(crate) media_timescale: u32,
    /// Empty time before the first presented sample, movie ticks.
    pub(crate) delay: u64,
    /// First presented media time, media ticks.
    pub(crate) media_start: u64,
    /// Presented length, movie ticks; `None` = to the end of the media.
    pub(crate) duration: Option<u64>,
}

impl EditTimeline {
    /// Reduce `list` to a timeline, or refuse its shape by name. `None` for an
    /// empty list (no edit at all).
    pub(crate) fn from_list(list: &TrackEditList, track: &str) -> Result<Option<Self>> {
        let entries = &list.entries;
        if entries.is_empty() {
            return Ok(None);
        }
        if list.movie_timescale == 0 || list.media_timescale == 0 {
            bail!(
                "{track} edit list: movie timescale {} / media timescale {} — an edit cannot be placed on a zero clock",
                list.movie_timescale,
                list.media_timescale
            );
        }
        let mut delay = 0u64;
        let mut media: Option<&EditEntry> = None;
        for (i, e) in entries.iter().enumerate() {
            if e.media_time < -1 {
                bail!("{track} edit list entry {i}: media_time {} (only -1, an empty edit, may be negative)", e.media_time);
            }
            if e.media_time == -1 {
                if i != 0 {
                    bail!(
                        "{track} edit list: an empty edit at entry {i} of {} puts a gap inside the presentation; \
                         only a single leading empty edit (a late start) is implemented",
                        entries.len()
                    );
                }
                delay = e.segment_duration;
                continue;
            }
            if e.rate_integer == 0 && e.rate_fraction == 0 {
                bail!(
                    "{track} edit list entry {i}: media_rate 0 is a dwell (one frame held for {} movie ticks); not implemented",
                    e.segment_duration
                );
            }
            if e.rate_integer != 1 || e.rate_fraction != 0 {
                bail!(
                    "{track} edit list entry {i}: media_rate {}+{}/65536 plays the media at another speed; only rate 1 is implemented",
                    e.rate_integer,
                    e.rate_fraction
                );
            }
            if media.is_some() {
                bail!(
                    "{track} edit list: {} media segments; only one media segment (optionally after one empty edit) is implemented",
                    entries.iter().filter(|e| e.media_time >= 0).count()
                );
            }
            media = Some(e);
        }
        let Some(media) = media else {
            bail!("{track} edit list: {} empty edit(s) and no media segment, so nothing is presented; not implemented", entries.len());
        };
        Ok(Some(Self {
            movie_timescale: list.movie_timescale,
            media_timescale: list.media_timescale,
            delay,
            media_start: media.media_time as u64,
            duration: (media.segment_duration != 0).then_some(media.segment_duration),
        }))
    }

    /// End of the presented media, exclusive, media ticks; `None` = open.
    pub(crate) fn media_end(&self) -> Option<u64> {
        self.duration
            .map(|d| self.media_start + rescale_round(d, self.media_timescale, self.movie_timescale))
    }

    /// The audio edit on a track whose sample timeline is in media ticks.
    pub(crate) fn audio_edit(&self) -> AudioEdit {
        AudioEdit {
            delay: rescale_round(self.delay, self.media_timescale, self.movie_timescale),
            media_start: self.media_start,
            media_end: self.media_end(),
        }
    }

    /// What this edit presents of a video track whose samples, in decode
    /// order, have presentation times `pts` (media ticks): the frames before
    /// `media_start` are hidden, the frames from `media_end` on are past the
    /// end. Decoded frames come out in presentation order, so the hidden ones
    /// are the first [`VideoPresentation::hidden`] of them — for a track whose
    /// timestamps are its display order; see [`hidden_display_positions`] for
    /// one whose are not.
    pub(crate) fn video_presentation(&self, pts: &[i64]) -> VideoPresentation {
        let start = self.media_start as i64;
        let end = self.media_end().map(|e| e as i64);
        let hidden = pts.iter().filter(|&&t| t < start).count() as u64;
        let presented = pts.iter().filter(|&&t| t >= start && end.is_none_or(|e| t < e)).count() as u64;
        VideoPresentation {
            delay_ticks: self.delay,
            delay_timescale: self.movie_timescale,
            ..VideoPresentation::leading(hidden, presented, pts.len() as u64)
        }
    }
}

/// Whether a track's presentation times run in decode order: no picture is
/// shown before one decoded ahead of it. True for a track without composition
/// offsets, whatever its stream actually does.
pub(crate) fn presentation_in_decode_order(pts: &[i64]) -> bool {
    pts.windows(2).all(|w| w[0] <= w[1])
}

/// Where the pictures of the first `hidden` samples come out of a decoder:
/// `output_decode_indices` lists, in output (display) order, the decode-order
/// index of each picture a decoder produced. Returns their display positions,
/// ascending, or `None` when fewer than `hidden` of them are in the list.
pub(crate) fn hidden_display_positions(output_decode_indices: &[u64], hidden: u64) -> Option<Vec<u64>> {
    let positions: Vec<u64> = output_decode_indices
        .iter()
        .enumerate()
        .filter(|(_, d)| **d < hidden)
        .map(|(i, _)| i as u64)
        .collect();
    (positions.len() as u64 == hidden).then_some(positions)
}

/// Presentation times, in decode order, of a non-fragmented track's samples:
/// `stts` decode times plus `ctts` offsets — the same arithmetic the streaming
/// demuxer's `next_video_sample` reports per sample, done over the tables
/// without reading a byte of sample data.
pub(crate) fn static_sample_pts(track: &mp4::Mp4Track, sample_count: u32) -> Vec<i64> {
    let stbl = &track.trak.mdia.minf.stbl;
    let n = sample_count as usize;
    let mut pts = Vec::with_capacity(n);
    let mut at = 0i64;
    'stts: for e in &stbl.stts.entries {
        for _ in 0..e.sample_count {
            if pts.len() == n {
                break 'stts;
            }
            pts.push(at);
            at += i64::from(e.sample_delta);
        }
    }
    if let Some(ctts) = &stbl.ctts {
        let mut i = 0usize;
        for e in &ctts.entries {
            for _ in 0..e.sample_count {
                if let Some(t) = pts.get_mut(i) {
                    *t += i64::from(e.sample_offset);
                }
                i += 1;
            }
        }
    }
    pts
}

/// Whether an H.264 / HEVC stream may output pictures in a different order
/// than it decodes them, from its SPS (`parameter_sets` as the avcC / hvcC
/// extractors return them). HEVC signals `sps_max_num_reorder_pics`; H.264
/// signals `max_num_reorder_frames` in the VUI when it bothers, and without it
/// only Baseline (no B slices) is known not to. Unknown — no SPS, or one that
/// does not parse — reads as "may".
pub(crate) fn stream_may_reorder(codec: &str, parameter_sets: &[Vec<u8>]) -> bool {
    for entry in parameter_sets {
        let nal: &[u8] = if entry.starts_with(&[0, 0, 0, 1]) {
            &entry[4..]
        } else if entry.starts_with(&[0, 0, 1]) {
            &entry[3..]
        } else {
            entry
        };
        match codec {
            "h265" if nal.len() > 2 && (nal[0] >> 1) & 0x3f == 33 => {
                let rbsp = h26x::nal::unescape_rbsp(nal);
                if let Ok(sps) = h26x::hevc::Sps::parse(&rbsp[2..]) {
                    return sps.max_num_reorder_pics > 0;
                }
            }
            "h264" if nal.len() > 1 && nal[0] & 0x1f == 7 => {
                let rbsp = h26x::nal::unescape_rbsp(&nal[1..]);
                if let Ok(sps) = h26x::h264::Sps::parse(&rbsp) {
                    return match sps.vui.as_ref().and_then(|v| v.max_num_reorder_frames) {
                        Some(n) => n > 0,
                        None => sps.profile_idc != 66,
                    };
                }
            }
            _ => {}
        }
    }
    true
}

/// The decode-order index of every picture an H.264 / HEVC decoder outputs,
/// in output order, until the pictures of the first `hidden` samples are all
/// out — then their display positions. `next_sample` yields Annex-B samples in
/// decode order (`None` at the end).
///
/// This is how the pictures an edit hides are found when the container's
/// timestamps cannot say: rivet's own decoder, the reader every other decode
/// path is checked against, run over the first GOPs only.
pub(crate) fn hidden_pictures_by_decoding(
    codec: &str,
    hidden: u64,
    mut next_sample: impl FnMut() -> Result<Option<Vec<u8>>>,
) -> Result<Vec<u64>> {
    enum Dec {
        H264(h26x::h264::H264Decoder),
        Hevc(h26x::hevc::HevcDecoder),
    }
    let mut dec = match codec {
        "h264" => Dec::H264(h26x::h264::H264Decoder::new()),
        "h265" => Dec::Hevc(h26x::hevc::HevcDecoder::new()),
        other => bail!("no decoder to place an edit's hidden pictures for codec '{other}'"),
    };
    let mut out: Vec<u64> = Vec::new();
    let mut pushed = 0u64;
    loop {
        if let Some(found) = hidden_display_positions(&out, hidden) {
            return Ok(found);
        }
        let Some(sample) = next_sample()? else { break };
        pushed += 1;
        for nal in h26x::nal::annexb_nals(&sample) {
            let r = match &mut dec {
                Dec::H264(d) => d.push_nal(nal),
                Dec::Hevc(d) => d.push_nal(nal),
            };
            r.map_err(|e| anyhow::anyhow!("decoding sample {} to place the edit's hidden pictures: {e}", pushed - 1))?;
        }
        loop {
            let pic = match &mut dec {
                Dec::H264(d) => d.try_next_picture(),
                Dec::Hevc(d) => d.try_next_picture(),
            };
            let Some(pic) = pic else { break };
            out.push(pic.decode_index);
        }
    }
    let flushed = match &mut dec {
        Dec::H264(d) => d.flush(),
        Dec::Hevc(d) => d.flush(),
    };
    flushed.map_err(|e| anyhow::anyhow!("flushing the decoder placing the edit's hidden pictures: {e}"))?;
    loop {
        let pic = match &mut dec {
            Dec::H264(d) => d.next_picture(),
            Dec::Hevc(d) => d.next_picture(),
        };
        let Some(pic) = pic else { break };
        out.push(pic.decode_index);
    }
    hidden_display_positions(&out, hidden).ok_or_else(|| {
        anyhow::anyhow!(
            "the edit hides the pictures of the first {hidden} samples, but decoding all {pushed} samples \
             output only {} of them",
            out.iter().filter(|&&d| d < hidden).count()
        )
    })
}

/// What an edit `timeline` presents of a video track with sample presentation
/// times `pts` (decode order). `None` when it changes nothing.
///
/// When the times are the display order — any file with composition offsets,
/// or a stream that never reorders — the hidden frames are the first few.
/// When they are not (H.264 / HEVC with no composition offsets on a stream
/// that may reorder: a raw elementary stream remuxed with `-c copy`), the
/// hidden samples' pictures are found with `decode_hidden`, and an edit that
/// also ends early is refused: which pictures it cuts from the end cannot be
/// read from the container.
pub(crate) fn resolve_video_presentation(
    codec: &str,
    parameter_sets: &[Vec<u8>],
    timeline: EditTimeline,
    pts: &[i64],
    decode_hidden: impl FnOnce(u64) -> Result<Vec<u64>>,
) -> Result<Option<VideoPresentation>> {
    let mut p = timeline.video_presentation(pts);
    if p.is_identity() {
        return Ok(None);
    }
    let h26x = matches!(codec, "h264" | "h265");
    if h26x && presentation_in_decode_order(pts) && stream_may_reorder(codec, parameter_sets) {
        let tail = p.samples - p.hidden.len() as u64 - p.presented;
        if tail > 0 {
            bail!(
                "video edit list ends {tail} samples before the track does, but this {codec} track has no \
                 composition offsets on a stream that may reorder its pictures, so which pictures the end of \
                 the edit cuts cannot be read from the container; not implemented"
            );
        }
        if !p.hidden.is_empty() {
            let hidden = p.hidden.len() as u64;
            p.hidden = decode_hidden(hidden)?;
            tracing::info!(
                codec,
                hidden_samples = hidden,
                display_positions = ?p.hidden,
                "video edit list: the track's timestamps are in decode order on a stream that may reorder; \
                 placed the hidden samples' pictures by decoding them"
            );
        }
    }
    Ok(Some(p))
}

/// The audio edit for the carried audio track (`track`), from the edit lists of
/// every audio track in the file (`audio_track_ids`). The demuxer carries one
/// audio track without saying which when there are several, so their edits
/// must agree; if they do not, which one applies is not known and that is
/// refused. `None` when there is no edit or it changes nothing.
pub(crate) fn resolve_audio_edit(
    data: &[u8],
    audio_track_ids: &[u32],
    track: &crate::demux::AudioTrack,
) -> Result<Option<AudioEdit>> {
    let mut timelines = Vec::with_capacity(audio_track_ids.len());
    for &id in audio_track_ids {
        let timeline = match track_edit_list(data, id)? {
            Some(list) => EditTimeline::from_list(&list, "audio")?,
            None => None,
        };
        timelines.push(timeline);
    }
    timelines.dedup();
    if timelines.len() > 1 {
        bail!(
            "{} audio tracks with different edit lists; the audio carried is not tied to one of them, so \
             which edit applies is not known — not implemented",
            audio_track_ids.len()
        );
    }
    let total: u64 = track.durations.iter().map(|&d| u64::from(d)).sum();
    Ok(timelines.into_iter().flatten().next().map(|t| t.audio_edit()).filter(|e| !e.is_identity(total)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn elst_v0(entries: &[(u32, i32, i16, i16)]) -> Vec<u8> {
        let mut b = vec![0u8, 0, 0, 0];
        b.extend_from_slice(&(entries.len() as u32).to_be_bytes());
        for &(d, t, ri, rf) in entries {
            b.extend_from_slice(&d.to_be_bytes());
            b.extend_from_slice(&t.to_be_bytes());
            b.extend_from_slice(&ri.to_be_bytes());
            b.extend_from_slice(&rf.to_be_bytes());
        }
        b
    }

    fn list(entries: &[(u32, i32, i16, i16)], movie: u32, media: u32) -> TrackEditList {
        TrackEditList { entries: parse_elst(&elst_v0(entries)).expect("parse"), movie_timescale: movie, media_timescale: media }
    }

    #[test]
    fn parses_version_0_and_1_entries_including_an_empty_edit() {
        let v0 = elst_v0(&[(500, -1, 1, 0), (10_000, 1024, 1, 0)]);
        assert_eq!(
            parse_elst(&v0).unwrap(),
            vec![
                EditEntry { segment_duration: 500, media_time: -1, rate_integer: 1, rate_fraction: 0 },
                EditEntry { segment_duration: 10_000, media_time: 1024, rate_integer: 1, rate_fraction: 0 },
            ]
        );
        let mut v1 = vec![1u8, 0, 0, 0, 0, 0, 0, 1];
        v1.extend_from_slice(&(5_000_000_000u64).to_be_bytes());
        v1.extend_from_slice(&(-1i64).to_be_bytes());
        v1.extend_from_slice(&[0, 1, 0, 0]);
        assert_eq!(
            parse_elst(&v1).unwrap(),
            vec![EditEntry { segment_duration: 5_000_000_000, media_time: -1, rate_integer: 1, rate_fraction: 0 }]
        );
        assert!(format!("{:#}", parse_elst(&v0[..20]).unwrap_err()).contains("2 entries need 32 bytes"));
        assert!(format!("{:#}", parse_elst(&[2, 0, 0, 0, 0, 0, 0, 0]).unwrap_err()).contains("version 2"));
    }

    #[test]
    fn finds_the_edit_list_of_the_named_track_in_a_box_tree() {
        fn bx(kind: &[u8; 4], body: &[u8]) -> Vec<u8> {
            let mut b = ((body.len() + 8) as u32).to_be_bytes().to_vec();
            b.extend_from_slice(kind);
            b.extend_from_slice(body);
            b
        }
        fn full_v0(ts_or_id: u32) -> Vec<u8> {
            let mut b = vec![0u8; 12];
            b.extend_from_slice(&ts_or_id.to_be_bytes());
            b.extend_from_slice(&[0u8; 8]);
            b
        }
        let trak = |id: u32, media_ts: u32, elst: Option<Vec<u8>>| {
            let mut body = bx(b"tkhd", &full_v0(id));
            if let Some(e) = elst {
                body.extend(bx(b"edts", &bx(b"elst", &e)));
            }
            body.extend(bx(b"mdia", &bx(b"mdhd", &full_v0(media_ts))));
            bx(b"trak", &body)
        };
        let mut moov = bx(b"mvhd", &full_v0(1000));
        moov.extend(trak(1, 15_360, Some(elst_v0(&[(10_000, 8704, 1, 0)]))));
        moov.extend(trak(2, 48_000, None));
        let file = [bx(b"ftyp", b"isom"), bx(b"moov", &moov)].concat();

        let video = track_edit_list(&file, 1).unwrap().expect("track 1 has an edit list");
        assert_eq!((video.movie_timescale, video.media_timescale), (1000, 15_360));
        assert_eq!(video.entries[0].media_time, 8704);
        assert_eq!(track_edit_list(&file, 2).unwrap(), None, "track 2 has no edts");
        assert_eq!(track_edit_list(&file, 9).unwrap(), None, "no track 9");
    }

    #[test]
    fn the_implemented_shapes_reduce_to_a_timeline() {
        // One media edit: ffmpeg's `-ss 3.5 -c copy` video, 17 frames of 512 in.
        let t = EditTimeline::from_list(&list(&[(6500, 8704, 1, 0)], 1000, 15_360), "video").unwrap().unwrap();
        assert_eq!((t.delay, t.media_start, t.duration), (0, 8704, Some(6500)));
        assert_eq!(t.media_end(), Some(8704 + 99_840));
        // Empty then media: a late start.
        let t = EditTimeline::from_list(&list(&[(500, -1, 1, 0), (10_000, 1024, 1, 0)], 1000, 48_000), "audio")
            .unwrap()
            .unwrap();
        assert_eq!(t.audio_edit(), AudioEdit { delay: 24_000, media_start: 1024, media_end: Some(481_024) });
        // Zero duration runs to the end.
        let t = EditTimeline::from_list(&list(&[(0, 0, 1, 0)], 1000, 48_000), "audio").unwrap().unwrap();
        assert_eq!(t.media_end(), None);
        assert_eq!(EditTimeline::from_list(&list(&[], 1000, 48_000), "audio").unwrap(), None);
    }

    #[test]
    fn every_other_shape_is_refused_by_name() {
        let refuse = |entries: &[(u32, i32, i16, i16)]| {
            format!("{:#}", EditTimeline::from_list(&list(entries, 1000, 30_000), "video").unwrap_err())
        };
        assert!(refuse(&[(1000, 0, 2, 0)]).contains("media_rate 2+0/65536"));
        assert!(refuse(&[(1000, 0, 1, 16384)]).contains("only rate 1 is implemented"));
        assert!(refuse(&[(1000, 3000, 0, 0)]).contains("dwell"));
        assert!(refuse(&[(1000, 0, 1, 0), (1000, 60_000, 1, 0)]).contains("2 media segments"));
        assert!(refuse(&[(1000, 0, 1, 0), (500, -1, 1, 0)]).contains("empty edit at entry 1"));
        assert!(refuse(&[(500, -1, 1, 0)]).contains("no media segment"));
        assert!(refuse(&[(500, -7, 1, 0)]).contains("media_time -7"));
        let zero = EditTimeline::from_list(&list(&[(1, 0, 1, 0)], 0, 30_000), "video");
        assert!(format!("{:#}", zero.unwrap_err()).contains("zero clock"));
    }

    #[test]
    fn video_frames_before_the_edit_are_hidden_and_after_it_past_the_end() {
        // B pictures: decode order 0 3 1 2 6 4 5, presentation = pts order,
        // with the ffmpeg composition shift of 2 frames (media_time 2).
        let pts = [2i64, 5, 3, 4, 8, 6, 7, 11, 9, 10];
        let whole = EditTimeline { movie_timescale: 1, media_timescale: 1, delay: 0, media_start: 2, duration: Some(10) };
        assert!(whole.video_presentation(&pts).is_identity(), "a composition shift hides nothing");
        let trimmed = EditTimeline { media_start: 5, duration: Some(4), ..whole };
        let p = trimmed.video_presentation(&pts);
        assert_eq!((p.hidden.clone(), p.presented, p.samples), (vec![0, 1, 2], 4, 10));
        let late = EditTimeline { delay: 7, ..whole };
        let p = late.video_presentation(&pts);
        assert_eq!((p.delay_ticks, p.is_identity()), (7, false));
    }

    #[test]
    fn a_track_without_composition_offsets_reads_as_decode_order() {
        assert!(presentation_in_decode_order(&[0, 48_000, 96_000, 144_000]));
        assert!(!presentation_in_decode_order(&[1024, 2560, 1536, 2048]));
    }

    #[test]
    fn hidden_pictures_are_found_where_the_decoder_put_them() {
        // WPP_C: display order carries decode indices 0 4 3 5 1 ... (hierarchical
        // B); the first three samples' pictures land at display 0, 4 and 8.
        let out = [0u64, 3, 4, 5, 2, 7, 8, 9, 1, 11, 12];
        assert_eq!(hidden_display_positions(&out, 3), Some(vec![0, 4, 8]));
        assert_eq!(hidden_display_positions(&out[..6], 3), None, "picture 1 not out yet");
        assert_eq!(hidden_display_positions(&out, 0), Some(vec![]));
    }

    fn hex(s: &str) -> Vec<u8> {
        (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).expect("hex")).collect()
    }

    #[test]
    fn the_sps_says_whether_a_stream_may_reorder() {
        // SPS NALs taken from libx264 High `-bf 3`, libx264 Baseline and
        // libx265 `bframes=3` clips (editlist_sps_hex.py).
        let high = hex("6764001eacd940a02ff9610000030001000003003c0f162d96");
        let baseline = hex("6742c00bd9028df930110000030001000003003c0f142a48");
        let hevc = hex("42010101600000030090000003000003003fa0050201696595964932b9a020000003002000000303c1");
        assert!(stream_may_reorder("h264", std::slice::from_ref(&high)));
        assert!(!stream_may_reorder("h264", &[[&[0u8, 0, 0, 1][..], &baseline].concat()]));
        assert!(stream_may_reorder("h265", &[hevc]));
        assert!(stream_may_reorder("h265", &[]), "no SPS reads as may reorder");
        assert!(stream_may_reorder("h264", &[vec![0x67, 0xff]]), "an SPS that does not parse reads as may");
    }

    #[test]
    fn written_edit_lists_read_back_through_the_parser() {
        let edts = crate::mux::build_edts(45_000, 1024, 900_000);
        assert_eq!((&edts[4..8], &edts[12..16]), (&b"edts"[..], &b"elst"[..]));
        assert_eq!(edts[16], 0, "fits version 0");
        assert_eq!(
            parse_elst(&edts[16..]).unwrap(),
            vec![
                EditEntry { segment_duration: 45_000, media_time: -1, rate_integer: 1, rate_fraction: 0 },
                EditEntry { segment_duration: 900_000, media_time: 1024, rate_integer: 1, rate_fraction: 0 },
            ]
        );
        let wide = crate::mux::build_edts(0, 5_000_000_000, 0);
        assert_eq!(wide[16], 1, "a media time past i32 needs version 1");
        assert_eq!(
            parse_elst(&wide[16..]).unwrap(),
            vec![EditEntry { segment_duration: 0, media_time: 5_000_000_000, rate_integer: 1, rate_fraction: 0 }]
        );
    }

    #[test]
    fn a_track_whose_timestamps_are_decode_order_places_hidden_pictures_by_decoding() {
        let decode_order: Vec<i64> = (0..48).map(|i| i * 48_000).collect();
        let timeline = EditTimeline {
            movie_timescale: 1000,
            media_timescale: 1_200_000,
            delay: 0,
            media_start: 144_000,
            duration: Some(1800),
        };
        // HEVC with no SPS to say otherwise: the decoder is asked, and its answer used.
        let mut asked = None;
        let p = resolve_video_presentation("h265", &[], timeline, &decode_order, |n| {
            asked = Some(n);
            Ok(vec![0, 4, 8])
        })
        .unwrap()
        .expect("hides three");
        assert_eq!(asked, Some(3));
        assert_eq!((p.hidden, p.presented, p.samples), (vec![0, 4, 8], 45, 48));

        // A codec with no reorder question (or composition offsets present): the first frames.
        let never = |_: u64| -> Result<Vec<u64>> { panic!("must not decode") };
        let p = resolve_video_presentation("av1", &[], timeline, &decode_order, never).unwrap().unwrap();
        assert_eq!(p.hidden, vec![0, 1, 2]);
        let mut reordered = decode_order.clone();
        reordered.swap(4, 5);
        let p = resolve_video_presentation("h265", &[], timeline, &reordered, never).unwrap().unwrap();
        assert_eq!(p.hidden, vec![0, 1, 2]);

        // Ending early on such a track is refused; changing nothing is `None`.
        let short = EditTimeline { duration: Some(1700), ..timeline };
        let err = resolve_video_presentation("h264", &[], short, &decode_order, never).unwrap_err();
        assert!(format!("{err:#}").contains("ends 2 samples before the track does"), "{err:#}");
        let whole = EditTimeline { media_start: 0, duration: None, ..timeline };
        assert_eq!(resolve_video_presentation("h265", &[], whole, &decode_order, never).unwrap(), None);
    }

    #[test]
    fn the_audio_edit_must_agree_across_audio_tracks() {
        fn bx(kind: &[u8; 4], body: &[u8]) -> Vec<u8> {
            let mut b = ((body.len() + 8) as u32).to_be_bytes().to_vec();
            b.extend_from_slice(kind);
            b.extend_from_slice(body);
            b
        }
        fn full_v0(v: u32) -> Vec<u8> {
            [vec![0u8; 12], v.to_be_bytes().to_vec(), vec![0u8; 8]].concat()
        }
        let trak = |id: u32, media_time: i32| {
            let body = [
                bx(b"tkhd", &full_v0(id)),
                bx(b"edts", &bx(b"elst", &elst_v0(&[(10_000, media_time, 1, 0)]))),
                bx(b"mdia", &bx(b"mdhd", &full_v0(48_000))),
            ]
            .concat();
            bx(b"trak", &body)
        };
        let file = |traks: &[Vec<u8>]| {
            let moov = [vec![bx(b"mvhd", &full_v0(1000))], traks.to_vec()].concat().concat();
            [bx(b"ftyp", b"isom"), bx(b"moov", &moov)].concat()
        };
        let track = crate::demux::AudioTrack {
            codec: "aac".into(),
            // 468 packets: 479232 ticks, inside the 10 s (480000-tick) edits below.
            samples: vec![vec![0]; 468],
            sample_rate: 48_000,
            channels: 1,
            asc: vec![],
            codec_private: vec![],
            timescale: 48_000,
            durations: vec![1024; 468],
        };
        let same = file(&[trak(2, 1024), trak(3, 1024)]);
        assert_eq!(
            resolve_audio_edit(&same, &[2, 3], &track).unwrap(),
            Some(AudioEdit { delay: 0, media_start: 1024, media_end: Some(481_024) })
        );
        let differ = file(&[trak(2, 1024), trak(3, 2048)]);
        let err = resolve_audio_edit(&differ, &[2, 3], &track).unwrap_err();
        assert!(format!("{err:#}").contains("2 audio tracks with different edit lists"), "{err:#}");
        let untouched = file(&[trak(2, 0)]);
        assert_eq!(resolve_audio_edit(&untouched, &[2], &track).unwrap(), None);
    }
}
