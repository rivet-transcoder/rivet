//! Integration tests for `tx3g` subtitle passthrough (`-c:s copy`).
//!
//! The unit tests in `mux::subtitle_track` cover sample encoding and gap
//! filling in isolation; what these check is the part that can only go wrong
//! once a third track shares the file — that the `moov` gains a well-formed
//! subtitle `trak`, that its `stco` offset actually points at the subtitle
//! bytes inside the `mdat`, and that adding it doesn't disturb the video and
//! audio tracks that were already there.

use bytes::Bytes;
use frame::EncodedPacket;
use container::AudioInfo;
use container::demux::subtitle::SubtitleCue;
use container::mux::Av1Mp4Muxer;

/// Minimal AV1 OBU_SEQUENCE_HEADER with `obu_has_size_field=1` — enough for
/// `extract_sequence_header` to succeed during finalize.
fn minimal_av1_first_packet() -> Bytes {
    let header: u8 = (1 << 3) | (1 << 1);
    let payload = [0u8; 5];
    let mut out = Vec::with_capacity(2 + payload.len());
    out.push(header);
    out.push(payload.len() as u8);
    out.extend_from_slice(&payload);
    Bytes::from(out)
}

fn push_minimal_video(muxer: &mut Av1Mp4Muxer, frames: usize) {
    muxer
        .add_packet(EncodedPacket { data: minimal_av1_first_packet(), pts: 0, is_keyframe: true })
        .expect("first packet");
    for i in 1..frames {
        muxer
            .add_packet(EncodedPacket {
                data: Bytes::from(vec![0xAAu8; 128]),
                pts: i as u64,
                is_keyframe: false,
            })
            .expect("packet");
    }
}

fn cue(start: u64, duration: u32, text: &str) -> SubtitleCue {
    SubtitleCue { start, duration, text: text.into() }
}

fn sample_cues() -> Vec<SubtitleCue> {
    vec![
        cue(1_000, 2_000, "First line"),
        cue(4_000, 1_500, "Second line"),
        cue(10_000, 2_000, "Third, with a comma"),
    ]
}

fn find_fourcc(data: &[u8], tag: &[u8; 4]) -> Option<usize> {
    data.windows(4).position(|w| w == tag)
}

fn count_fourcc(data: &[u8], tag: &[u8; 4]) -> usize {
    data.windows(4).filter(|w| *w == tag).count()
}

/// Read the single chunk offset out of the subtitle track's `stco`. The
/// subtitle `stco` is the last one in the file (video, then audio, then
/// subtitles), and it always has exactly one entry.
fn last_stco_single_offset(mp4: &[u8]) -> u64 {
    let pos = mp4
        .windows(4)
        .enumerate()
        .filter(|(_, w)| *w == b"stco")
        .map(|(i, _)| i)
        .next_back()
        .expect("an stco box");
    // stco: [size(4)][type(4)][version+flags(4)][entry_count(4)][offsets…]
    let entry_count = u32::from_be_bytes(mp4[pos + 8..pos + 12].try_into().unwrap());
    assert_eq!(entry_count, 1, "the subtitle track should be a single chunk");
    u32::from_be_bytes(mp4[pos + 12..pos + 16].try_into().unwrap()) as u64
}

#[test]
fn subtitles_add_a_tx3g_trak_to_the_moov() {
    let mut muxer = Av1Mp4Muxer::new(640, 480, 30.0).unwrap();
    push_minimal_video(&mut muxer, 30);
    muxer.add_subtitle_track(&sample_cues(), 1_000, "eng").unwrap();
    let mp4 = muxer.finalize().unwrap();

    assert!(find_fourcc(&mp4, b"tx3g").is_some(), "no tx3g sample entry");
    assert!(find_fourcc(&mp4, b"sbtl").is_some(), "no sbtl handler");
    assert!(find_fourcc(&mp4, b"nmhd").is_some(), "no nmhd media header");
    assert!(find_fourcc(&mp4, b"ftab").is_some(), "no font table");
    // Video + subtitles = 2 traks.
    assert_eq!(count_fourcc(&mp4, b"trak"), 2, "expected a video and a subtitle trak");
}

#[test]
fn subtitle_bytes_land_where_the_chunk_offset_says() {
    // The offset bookkeeping is the part most likely to be subtly wrong: the
    // subtitle chunk is written after all the interleaved video/audio bytes,
    // and its stco has to agree.
    let mut muxer = Av1Mp4Muxer::new(640, 480, 30.0).unwrap();
    push_minimal_video(&mut muxer, 30);
    muxer.add_subtitle_track(&sample_cues(), 1_000, "eng").unwrap();
    let mp4 = muxer.finalize().unwrap();

    let offset = last_stco_single_offset(&mp4) as usize;
    assert!(offset < mp4.len(), "subtitle chunk offset past end of file");

    // The first sample there is the leading gap (cue 0 starts at t=1000), so
    // it's an empty sample: a zero length prefix. The next sample must be the
    // first cue's text.
    assert_eq!(&mp4[offset..offset + 2], &[0x00, 0x00], "expected a leading empty sample");
    let first_len = u16::from_be_bytes(mp4[offset + 2..offset + 4].try_into().unwrap()) as usize;
    let text = std::str::from_utf8(&mp4[offset + 4..offset + 4 + first_len]).unwrap();
    assert_eq!(text, "First line");

    // And every cue's text is somewhere in the payload, comma included.
    for c in sample_cues() {
        assert!(
            mp4.windows(c.text.len()).any(|w| w == c.text.as_bytes()),
            "cue text {:?} missing from the mdat",
            c.text
        );
    }
}

#[test]
fn subtitles_coexist_with_an_audio_track() {
    // Three tracks in one file: the case that exercises next_track_ID, the
    // movie duration max, and the mdat payload accounting all at once.
    let mut muxer = Av1Mp4Muxer::new(640, 480, 30.0).unwrap();
    push_minimal_video(&mut muxer, 30);
    muxer.with_audio(AudioInfo::aac_lc(48_000, 2, vec![0x11, 0x90])).unwrap();
    for _ in 0..40 {
        muxer.add_audio_sample(&[0x21, 0x00, 0x03], 0, 1024).unwrap();
    }
    muxer.add_subtitle_track(&sample_cues(), 1_000, "eng").unwrap();
    let mp4 = muxer.finalize().unwrap();

    assert_eq!(count_fourcc(&mp4, b"trak"), 3, "video + audio + subtitles");
    assert!(find_fourcc(&mp4, b"tx3g").is_some());
    assert!(find_fourcc(&mp4, b"mp4a").is_some(), "audio track must survive");

    // next_track_ID in mvhd is the last u32 of the box; with three tracks it
    // must be 4, or a player appending a track would collide with the
    // subtitle track's ID.
    let mvhd = find_fourcc(&mp4, b"mvhd").expect("mvhd");
    let size = u32::from_be_bytes(mp4[mvhd - 4..mvhd].try_into().unwrap()) as usize;
    let end = mvhd - 4 + size;
    assert_eq!(u32::from_be_bytes(mp4[end - 4..end].try_into().unwrap()), 4);

    // The subtitle chunk still resolves, sitting after the audio payload.
    let offset = last_stco_single_offset(&mp4) as usize;
    assert_eq!(&mp4[offset..offset + 2], &[0x00, 0x00]);
}

#[test]
fn no_subtitles_leaves_the_file_byte_identical() {
    // Regression guard: adding the feature must not perturb the two-track
    // layout when no cues are supplied.
    let build = |with_empty_call: bool| {
        let mut muxer = Av1Mp4Muxer::new(640, 480, 30.0).unwrap();
        push_minimal_video(&mut muxer, 30);
        if with_empty_call {
            // An empty cue list is a no-op, not an empty trak.
            muxer.add_subtitle_track(&[], 1_000, "eng").unwrap();
        }
        muxer.finalize().unwrap()
    };
    assert_eq!(build(false), build(true));
    assert!(find_fourcc(&build(true), b"tx3g").is_none(), "empty cue list must emit no trak");
}

/// Every `stco` in the file, in order, as (entry_count, first offset).
fn all_stco(mp4: &[u8]) -> Vec<(u32, u64)> {
    mp4.windows(4)
        .enumerate()
        .filter(|(_, w)| *w == b"stco")
        .map(|(pos, _)| {
            let n = u32::from_be_bytes(mp4[pos + 8..pos + 12].try_into().unwrap());
            let first = u32::from_be_bytes(mp4[pos + 12..pos + 16].try_into().unwrap()) as u64;
            (n, first)
        })
        .collect()
}

/// Every `tkhd` track_ID in the file, in trak order (version 0 boxes).
fn all_track_ids(mp4: &[u8]) -> Vec<u32> {
    mp4.windows(4)
        .enumerate()
        .filter(|(_, w)| *w == b"tkhd")
        .map(|(pos, _)| u32::from_be_bytes(mp4[pos + 16..pos + 20].try_into().unwrap()))
        .collect()
}

#[test]
fn two_subtitle_tracks_get_two_traks_with_their_own_ids_languages_and_chunks() {
    let mut muxer = Av1Mp4Muxer::new(640, 480, 30.0).unwrap();
    push_minimal_video(&mut muxer, 30);
    muxer.with_audio(AudioInfo::aac_lc(48_000, 2, vec![0x11, 0x90])).unwrap();
    for _ in 0..40 {
        muxer.add_audio_sample(&[0x21, 0x00, 0x03], 0, 1024).unwrap();
    }
    let eng = sample_cues();
    let deu = vec![cue(2_000, 1_000, "Hallo Welt"), cue(6_000, 500, "Zweite")];
    muxer.add_subtitle_track(&eng, 1_000, "eng").unwrap();
    muxer.add_subtitle_track(&deu, 1_000, "deu").unwrap();
    let mp4 = muxer.finalize().unwrap();

    assert_eq!(count_fourcc(&mp4, b"trak"), 4, "video + audio + two subtitle traks");
    assert_eq!(count_fourcc(&mp4, b"tx3g"), 2);
    assert_eq!(all_track_ids(&mp4), vec![1, 2, 3, 4], "subtitle IDs follow on from the first");

    // next_track_ID is one past the last subtitle track.
    let mvhd = find_fourcc(&mp4, b"mvhd").expect("mvhd");
    let size = u32::from_be_bytes(mp4[mvhd - 4..mvhd].try_into().unwrap()) as usize;
    let end = mvhd - 4 + size;
    assert_eq!(u32::from_be_bytes(mp4[end - 4..end].try_into().unwrap()), 5);

    // Each subtitle track is one chunk, and the second's chunk starts where
    // the first's bytes end: the offsets tile the subtitle tail of the mdat.
    let stcos = all_stco(&mp4);
    let subs: Vec<(u32, u64)> = stcos[stcos.len() - 2..].to_vec();
    assert_eq!(subs[0].0, 1);
    assert_eq!(subs[1].0, 1);
    let (o1, o2) = (subs[0].1 as usize, subs[1].1 as usize);
    assert!(o1 < o2 && o2 < mp4.len());
    // Walk track 1's samples (length-prefixed) from o1; they must end at o2.
    let mut pos = o1;
    let mut texts = Vec::new();
    while pos < o2 {
        let len = u16::from_be_bytes(mp4[pos..pos + 2].try_into().unwrap()) as usize;
        if len > 0 {
            texts.push(std::str::from_utf8(&mp4[pos + 2..pos + 2 + len]).unwrap().to_string());
        }
        pos += 2 + len;
    }
    assert_eq!(pos, o2, "track 1's samples tile exactly up to track 2's chunk");
    assert_eq!(texts, eng.iter().map(|c| c.text.clone()).collect::<Vec<_>>());
    // Track 2's first sample is its leading gap, then "Hallo Welt".
    assert_eq!(&mp4[o2..o2 + 2], &[0x00, 0x00]);
    let len = u16::from_be_bytes(mp4[o2 + 2..o2 + 4].try_into().unwrap()) as usize;
    assert_eq!(std::str::from_utf8(&mp4[o2 + 4..o2 + 4 + len]).unwrap(), "Hallo Welt");

    // Both languages are in their mdhd boxes (packed ISO-639-2).
    let pack = |l: &str| -> [u8; 2] {
        let b = l.as_bytes();
        let v = ((b[0] as u16 - 0x60) << 10) | ((b[1] as u16 - 0x60) << 5) | (b[2] as u16 - 0x60);
        v.to_be_bytes()
    };
    let mdhd_langs: Vec<[u8; 2]> = mp4
        .windows(4)
        .enumerate()
        .filter(|(_, w)| *w == b"mdhd")
        .map(|(pos, _)| [mp4[pos + 24], mp4[pos + 25]])
        .collect();
    assert_eq!(mdhd_langs.len(), 4);
    assert_eq!(mdhd_langs[2], pack("eng"));
    assert_eq!(mdhd_langs[3], pack("deu"));
}

#[test]
fn zero_timescale_is_rejected() {
    let mut muxer = Av1Mp4Muxer::new(640, 480, 30.0).unwrap();
    assert!(muxer.add_subtitle_track(&sample_cues(), 0, "eng").is_err());
}

#[test]
fn declared_box_sizes_span_the_whole_file() {
    // Walk the top-level box chain. If any size field disagrees with the
    // bytes actually written the walk won't land exactly on the end — the
    // cheapest structural check that a third track didn't corrupt the layout.
    let mut muxer = Av1Mp4Muxer::new(640, 480, 30.0).unwrap();
    push_minimal_video(&mut muxer, 30);
    muxer.with_audio(AudioInfo::aac_lc(48_000, 2, vec![0x11, 0x90])).unwrap();
    for _ in 0..40 {
        muxer.add_audio_sample(&[0x21, 0x00, 0x03], 0, 1024).unwrap();
    }
    muxer.add_subtitle_track(&sample_cues(), 1_000, "eng").unwrap();
    let mp4 = muxer.finalize().unwrap();

    let mut pos = 0usize;
    let mut seen: Vec<String> = Vec::new();
    while pos + 8 <= mp4.len() {
        let size = u32::from_be_bytes(mp4[pos..pos + 4].try_into().unwrap()) as usize;
        let tag = String::from_utf8_lossy(&mp4[pos + 4..pos + 8]).to_string();
        let advance = if size == 1 {
            u64::from_be_bytes(mp4[pos + 8..pos + 16].try_into().unwrap()) as usize
        } else {
            size
        };
        assert!(advance >= 8, "box {tag} declared an impossible size {advance}");
        seen.push(tag);
        pos += advance;
    }
    assert_eq!(pos, mp4.len(), "top-level box sizes must tile the file exactly: {seen:?}");
    assert_eq!(seen, vec!["ftyp", "moov", "mdat"]);
}
