//! B pictures through the two writers and back through the crate's own
//! demuxer.
//!
//! Packets reach a muxer in decode order carrying presentation timestamps;
//! the muxer must put every sample back where it is *shown*. Both writers
//! are checked the same way: mux a decode-order sequence, read the file with
//! `demux_streaming` (the reader every job runs), and require the samples'
//! presentation times to come back as the ranks of the timestamps that went
//! in. The control — in-order packets — must produce no table at all.
//!
//! The sequence used throughout is one IDR then two mini-GOPs of an anchor
//! followed by the two B pictures it releases, so display order 0 3 1 2 6 4
//! 5: every anchor is presented two frames late and every B one frame early.

use bytes::Bytes;
use container::cmaf::{CmafVideoMuxer, CmafVideoMuxerOptions};
use container::mux::Av1Mp4Muxer;
use container::streaming::{Sample, StreamingDemuxer, demux_streaming};
use frame::{ColorMetadata, EncodedPacket, VideoCodec};

/// Decode order of one IDR + two mini-GOPs with two B pictures each.
const DECODE_ORDER: [u64; 7] = [0, 3, 1, 2, 6, 4, 5];

/// A synthetic OBU_SEQUENCE_HEADER with `obu_has_size_field=1`, which is all
/// the AV1 muxers read of a first packet.
fn av1_first_packet() -> Vec<u8> {
    let header: u8 = (1 << 3) | (1 << 1);
    let payload = [0u8; 5];
    let mut out = vec![header, payload.len() as u8];
    out.extend_from_slice(&payload);
    out
}

/// Later packets are opaque bytes to the muxer; sized by frame number so a
/// misplaced sample is visible by its length too.
fn av1_packet(n: u64) -> Vec<u8> {
    vec![0xA5; 64 + n as usize]
}

fn packet(pts: u64) -> EncodedPacket {
    EncodedPacket {
        data: Bytes::from(if pts == 0 { av1_first_packet() } else { av1_packet(pts) }),
        pts,
        is_keyframe: pts == 0,
    }
}

fn drain(d: &mut dyn StreamingDemuxer) -> Vec<Sample> {
    let mut out = Vec::new();
    while let Some(s) = d.next_video_sample().expect("next_video_sample") {
        out.push(s);
    }
    out
}

/// Offset of the first occurrence of a box type, or `None`.
fn find_fourcc(data: &[u8], tag: &[u8; 4]) -> Option<usize> {
    data.windows(4).position(|w| w == tag)
}

fn mux_mp4(order: &[u64]) -> Bytes {
    let mut m = Av1Mp4Muxer::new_with_codec(64, 64, 30.0, VideoCodec::Av1).unwrap();
    for &pts in order {
        m.add_packet(packet(pts)).unwrap();
    }
    m.finalize().unwrap()
}

#[test]
fn mp4_ctts_puts_every_sample_back_where_it_is_shown() {
    let bytes = mux_mp4(&DECODE_ORDER);

    // The table is there, and it is the signed (version 1) form.
    let ctts = find_fourcc(&bytes, b"ctts").expect("a reordered track writes ctts");
    assert_eq!(bytes[ctts + 4], 1, "ctts version 1 (signed offsets)");
    // Decode order is preserved on disk: sample 2 (display 3) is the
    // third-shortest packet, not the second.
    let mut d = demux_streaming(&bytes).expect("our demuxer reads what we wrote");
    let samples = drain(d.as_mut());
    assert_eq!(samples.len(), DECODE_ORDER.len(), "one sample per packet");
    let sizes: Vec<usize> = samples.iter().map(|s| s.data.len()).collect();
    let want_sizes: Vec<usize> = DECODE_ORDER
        .iter()
        .map(|&n| if n == 0 { av1_first_packet().len() } else { 64 + n as usize })
        .collect();
    assert_eq!(sizes, want_sizes, "samples stay in decode order in the file");

    // And each sample's presentation time is its display rank on the 90 kHz
    // / 30 fps grid — exactly the timestamp order that went in.
    let pts: Vec<i64> = samples.iter().map(|s| s.pts_ticks).collect();
    let want: Vec<i64> = DECODE_ORDER.iter().map(|&n| n as i64 * 3000).collect();
    assert_eq!(pts, want, "presentation times follow the input order");
    // Decode times are the fixed grid, so the first packet is shown at zero
    // and nothing is shown twice.
    assert_eq!(pts[0], 0);
    let mut sorted = pts.clone();
    sorted.sort_unstable();
    assert_eq!(sorted, (0..7).map(|k| k * 3000).collect::<Vec<i64>>());
}

#[test]
fn mp4_in_order_packets_write_no_ctts_at_all() {
    let bytes = mux_mp4(&[0, 1, 2, 3, 4, 5, 6]);
    assert!(find_fourcc(&bytes, b"ctts").is_none(), "no table without reordering");
    let mut d = demux_streaming(&bytes).unwrap();
    let pts: Vec<i64> = drain(d.as_mut()).iter().map(|s| s.pts_ticks).collect();
    assert_eq!(pts, (0..7).map(|k| k * 3000).collect::<Vec<i64>>());
}

#[test]
fn mp4_refuses_two_samples_with_one_timestamp() {
    let mut m = Av1Mp4Muxer::new_with_codec(64, 64, 30.0, VideoCodec::Av1).unwrap();
    for &pts in &[0u64, 2, 1, 1] {
        m.add_packet(packet(pts)).unwrap();
    }
    let err = m.finalize().err().expect("a duplicated pts has no display rank");
    assert!(format!("{err:#}").contains("appears on two samples"), "{err:#}");
}

fn mux_cmaf(order: &[u64]) -> (Vec<u8>, Vec<u8>) {
    let dir = tempfile::tempdir().unwrap();
    let mut m = CmafVideoMuxer::new_with_codec_options(
        dir.path(),
        64,
        64,
        30_000,
        ColorMetadata::default(),
        VideoCodec::Av1,
        CmafVideoMuxerOptions::default(),
    )
    .unwrap();
    for &pts in order {
        let p = packet(pts);
        m.add_packet(p.data.to_vec(), 1000, p.is_keyframe, p.pts).unwrap();
    }
    let seg = m.flush_segment().unwrap().expect("one segment");
    let manifest = m.finalize().unwrap();
    (std::fs::read(&manifest.init_path).unwrap(), std::fs::read(&seg.path).unwrap())
}

#[test]
fn cmaf_trun_v1_puts_every_sample_back_where_it_is_shown() {
    let (init, seg) = mux_cmaf(&DECODE_ORDER);

    let trun = find_fourcc(&seg, b"trun").expect("trun");
    assert_eq!(seg[trun + 4], 1, "trun version 1 (signed composition offsets)");
    let flags = u32::from_be_bytes([0, seg[trun + 5], seg[trun + 6], seg[trun + 7]]);
    assert_ne!(flags & 0x800, 0, "sample-composition-time-offsets-present");

    // init + segment is a complete fragmented MP4 for the streaming demuxer,
    // which walks moof/traf/trun itself.
    let mut file = init;
    file.extend_from_slice(&seg);
    let mut d = demux_streaming(&file).expect("our demuxer reads what we wrote");
    let samples = drain(d.as_mut());
    assert_eq!(samples.len(), DECODE_ORDER.len());
    let pts: Vec<i64> = samples.iter().map(|s| s.pts_ticks).collect();
    let want: Vec<i64> = DECODE_ORDER.iter().map(|&n| n as i64 * 1000).collect();
    assert_eq!(pts, want, "presentation times follow the input order");
    assert_eq!(pts[0], 0, "the sync sample opens the segment's presentation too");
}

#[test]
fn cmaf_in_order_packets_write_a_version_0_trun_without_offsets() {
    let (init, seg) = mux_cmaf(&[0, 1, 2, 3, 4, 5, 6]);
    let trun = find_fourcc(&seg, b"trun").expect("trun");
    assert_eq!(seg[trun + 4], 0, "trun version 0 without reordering");
    let flags = u32::from_be_bytes([0, seg[trun + 5], seg[trun + 6], seg[trun + 7]]);
    assert_eq!(flags & 0x800, 0, "no composition-offset column");
    let mut file = init;
    file.extend_from_slice(&seg);
    let mut d = demux_streaming(&file).unwrap();
    let pts: Vec<i64> = drain(d.as_mut()).iter().map(|s| s.pts_ticks).collect();
    assert_eq!(pts, (0..7).map(|k| k * 1000).collect::<Vec<i64>>());
}

/// A segment must stand alone: its sync sample has to be the earliest
/// picture it presents. A picture shown before the IDR but coded after it
/// is a reorder leaking across the boundary, and the writer refuses it.
#[test]
fn cmaf_refuses_a_segment_whose_sync_sample_is_not_its_earliest() {
    let dir = tempfile::tempdir().unwrap();
    let mut m = CmafVideoMuxer::new_with_codec_options(
        dir.path(),
        64,
        64,
        30_000,
        ColorMetadata::default(),
        VideoCodec::Av1,
        CmafVideoMuxerOptions::default(),
    )
    .unwrap();
    // Sync sample at pts 1; the picture at pts 0 arrives after it.
    m.add_packet(av1_first_packet(), 1000, true, 1).unwrap();
    m.add_packet(av1_packet(0), 1000, false, 0).unwrap();
    m.add_packet(av1_packet(2), 1000, false, 2).unwrap();
    let err = m.flush_segment().err().expect("refused");
    assert!(format!("{err:#}").contains("earliest-presented"), "{err:#}");
}
