//! Shared CMAF/HLS helpers used by the job engine and the multi-GPU
//! orchestrator: segment-boundary flushing, per-rung contribution merging,
//! bandwidth measurement, and AV1 codec-string extraction.

use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};

use codec::codec_strings::{av1_codec_string, avc_codec_string, hevc_codec_string};
use codec::encode::EncodedPacket;
use codec::pixel_format::{H264SpsInfo, HevcSpsInfo, parse_av1_sequence_header};
use container::cmaf::{CmafAudioMuxer, CmafTrackManifest, CmafVideoMuxer, SegmentInfo};

/// Keyframe interval (frames) for a target segment length at a frame rate.
pub fn keyframe_interval_for_segment(segment_duration_seconds: f64, frame_rate: f64) -> u32 {
    ((segment_duration_seconds * frame_rate).round() as u32).max(1)
}

/// Number of CMAF segments a rung will produce (ceil division).
pub fn total_segments_for_rung(total_input_frames: u64, keyframe_interval: u32) -> u32 {
    if total_input_frames == 0 || keyframe_interval == 0 {
        return 0;
    }
    let ki = keyframe_interval as u64;
    let segs = total_input_frames.div_ceil(ki);
    segs.min(u32::MAX as u64) as u32
}

/// Add one encoded video packet to a [`CmafVideoMuxer`], flushing the prior
/// segment first when the next packet is a keyframe and the buffered duration
/// has reached the segment target (so each segment opens on an IDR).
///
/// Returns the segment this call closed, if it closed one. That is the point
/// of the return value rather than a detail of it: a segment is a finished,
/// self-contained file the moment it is flushed, and handing it back here is
/// what lets a caller get it off this host before the rest of the rung is
/// done — losing a worker then costs the segment in flight instead of every
/// segment encoded so far.
pub fn add_packet_with_segment_flush(
    muxer: &mut CmafVideoMuxer,
    packet: &EncodedPacket,
    duration_ticks: u32,
    segment_target_ticks: u64,
) -> Result<Option<SegmentInfo>> {
    let mut flushed = None;
    if packet.is_keyframe
        && muxer.pending_duration_ticks() >= segment_target_ticks
        && muxer.first_pending_is_keyframe()
    {
        flushed = muxer.flush_segment().context("flush CMAF video segment")?;
    }
    muxer.add_packet(packet.data.to_vec(), duration_ticks, packet.is_keyframe)?;
    Ok(flushed)
}

/// Add one audio sample to a [`CmafAudioMuxer`] with segment flushing on the
/// same time grid. Returns the segment this call closed, if any.
pub fn add_audio_sample_with_segment_flush(
    muxer: &mut CmafAudioMuxer,
    payload: Vec<u8>,
    duration_ticks: u32,
    segment_target_ticks: u64,
) -> Result<Option<SegmentInfo>> {
    let mut flushed = None;
    if muxer.pending_duration_ticks() >= segment_target_ticks {
        flushed = muxer.flush_segment().context("flush CMAF audio segment")?;
    }
    muxer.add_packet(payload, duration_ticks)?;
    Ok(flushed)
}

/// One encoder worker's contribution to a rung (a slice of its segments).
#[derive(Debug, Clone)]
pub struct RungContribution {
    pub width: u32,
    pub height: u32,
    pub relative_dir: String,
    pub manifest: CmafTrackManifest,
}

/// Merge several workers' segment lists for one rung into a single ordered
/// manifest, detecting duplicate segment numbers and internal gaps.
pub fn merge_rung_contributions(contributions: Vec<RungContribution>) -> Result<RungContribution> {
    if contributions.is_empty() {
        bail!("merge_rung_contributions: at least one contribution required");
    }
    let first = &contributions[0];
    let width = first.width;
    let height = first.height;
    let relative_dir = first.relative_dir.clone();
    let timescale = first.manifest.timescale;
    let init_path = first.manifest.init_path.clone();

    for c in &contributions[1..] {
        if c.width != width || c.height != height {
            bail!(
                "contributors disagree on dimensions: first={width}x{height}, other={}x{}",
                c.width,
                c.height
            );
        }
        if c.relative_dir != relative_dir {
            bail!("contributors disagree on relative_dir");
        }
        if c.manifest.timescale != timescale {
            bail!("contributors disagree on timescale");
        }
    }

    let mut all_segments: Vec<SegmentInfo> = contributions
        .into_iter()
        .flat_map(|c| c.manifest.segments)
        .collect();
    all_segments.sort_by_key(|s| s.sequence_number);

    for w in all_segments.windows(2) {
        if w[0].sequence_number == w[1].sequence_number {
            bail!(
                "duplicate segment number {} in merged manifest (paths: {:?}, {:?})",
                w[0].sequence_number,
                w[0].path,
                w[1].path
            );
        }
    }
    if let (Some(first), Some(last)) = (all_segments.first(), all_segments.last()) {
        let expected = last.sequence_number - first.sequence_number + 1;
        if all_segments.len() as u32 != expected {
            bail!(
                "internal gap in merged segments: range {}..={} expects {} segments, got {}",
                first.sequence_number,
                last.sequence_number,
                expected,
                all_segments.len()
            );
        }
    }

    Ok(RungContribution {
        width,
        height,
        relative_dir,
        manifest: CmafTrackManifest {
            init_path,
            segments: all_segments,
            timescale,
        },
    })
}

/// (average, peak) bandwidth in bits/sec across a manifest's segments.
pub fn measure_bandwidth(manifest: &CmafTrackManifest) -> (u32, u32) {
    if manifest.segments.is_empty() {
        return (0, 0);
    }
    let total_bytes: u64 = manifest.segments.iter().map(|s| s.byte_size).sum();
    let total_ticks: u64 = manifest.segments.iter().map(|s| s.duration_ticks).sum();
    let total_seconds = total_ticks as f64 / manifest.timescale.max(1) as f64;
    let avg_bps = if total_seconds > 0.0 {
        ((total_bytes as f64 * 8.0) / total_seconds) as u32
    } else {
        0
    };
    let mut peak_bps: u32 = 0;
    for seg in &manifest.segments {
        let secs = seg.duration_ticks as f64 / manifest.timescale.max(1) as f64;
        if secs > 0.0 {
            let bps = ((seg.byte_size as f64 * 8.0) / secs) as u32;
            peak_bps = peak_bps.max(bps);
        }
    }
    (avg_bps, peak_bps.max(avg_bps))
}

/// Parse the HLS `CODECS=` string for a rendition from its init segment,
/// dispatching on the visual sample entry: `av01` → AV1 sequence header,
/// `avc1`/`avc3` → `avcC` profile/level, `hvc1`/`hev1` → `hvcC` profile-tier-level.
pub fn codec_string_from_init(init_path: &Path) -> Result<String> {
    let bytes = std::fs::read(init_path)
        .with_context(|| format!("reading init segment {}", init_path.display()))?;
    let entries =
        stsd_sample_entries(&bytes).ok_or_else(|| anyhow!("stsd entries not found in init"))?;
    if entries.len() < 8 {
        bail!("init segment sample entry truncated");
    }
    let fourcc: [u8; 4] = entries[4..8].try_into().unwrap();
    let entry = find_box(entries, &fourcc).ok_or_else(|| anyhow!("sample entry box not found"))?;
    // Visual sample entry: 8-byte box header + 78-byte VisualSampleEntry header,
    // then the codec config box (av1C / avcC / hvcC).
    let children = entry.get(8 + 78..).unwrap_or(&[]);
    let fcc_str = std::str::from_utf8(&fourcc).unwrap_or("");
    match &fourcc {
        b"av01" => {
            let av1c = find_box(children, b"av1C").ok_or_else(|| anyhow!("av1C box missing"))?;
            let obus = av1c.get(8 + 4..).ok_or_else(|| anyhow!("av1C truncated"))?;
            let seq = parse_av1_sequence_header(obus)
                .ok_or_else(|| anyhow!("could not parse AV1 sequence header from av1C"))?;
            Ok(av1_codec_string(&seq))
        }
        b"avc1" | b"avc3" => {
            let avcc = find_box(children, b"avcC").ok_or_else(|| anyhow!("avcC box missing"))?;
            // avcC body: [0]=version [1]=profile_idc [2]=constraint [3]=level_idc.
            let body = avcc.get(8..).ok_or_else(|| anyhow!("avcC truncated"))?;
            if body.len() < 4 {
                bail!("avcC profile/level truncated");
            }
            let sps = H264SpsInfo {
                profile_idc: body[1],
                constraint_set_flags: body[2],
                level_idc: body[3],
                ..Default::default()
            };
            Ok(avc_codec_string(fcc_str, &sps))
        }
        b"hvc1" | b"hev1" => {
            let hvcc = find_box(children, b"hvcC").ok_or_else(|| anyhow!("hvcC box missing"))?;
            // hvcC body: [0]=version [1]=space|tier|profile_idc [2..6]=compat
            // [6..12]=constraint flags [12]=level_idc.
            let body = hvcc.get(8..).ok_or_else(|| anyhow!("hvcC truncated"))?;
            if body.len() < 13 {
                bail!("hvcC profile-tier-level truncated");
            }
            let b1 = body[1];
            let constraint = ((body[6] as u64) << 40)
                | ((body[7] as u64) << 32)
                | ((body[8] as u64) << 24)
                | ((body[9] as u64) << 16)
                | ((body[10] as u64) << 8)
                | (body[11] as u64);
            let sps = HevcSpsInfo {
                general_profile_space: b1 >> 6,
                tier_flag: (b1 >> 5) & 1 == 1,
                profile_idc: b1 & 0x1F,
                profile_compatibility_flags: u32::from_be_bytes([body[2], body[3], body[4], body[5]]),
                general_constraint_flags: constraint,
                level_idc: body[12],
                ..Default::default()
            };
            Ok(hevc_codec_string(fcc_str, &sps))
        }
        other => bail!("unsupported video sample entry fourcc {other:?} in init segment"),
    }
}

fn stsd_sample_entries(buf: &[u8]) -> Option<&[u8]> {
    let moov = find_box(buf, b"moov")?;
    let trak = find_child_box(moov, b"trak")?;
    let mdia = find_child_box(trak, b"mdia")?;
    let minf = find_child_box(mdia, b"minf")?;
    let stbl = find_child_box(minf, b"stbl")?;
    let stsd = find_child_box(stbl, b"stsd")?;
    if stsd.len() < 16 {
        return None;
    }
    // Skip the stsd 8-byte box header + 4-byte version/flags + 4-byte entry_count.
    Some(&stsd[16..])
}

fn find_child_box<'a>(parent: &'a [u8], box_type: &[u8; 4]) -> Option<&'a [u8]> {
    if parent.len() < 8 {
        return None;
    }
    find_box(&parent[8..], box_type)
}

fn find_box<'a>(buf: &'a [u8], box_type: &[u8; 4]) -> Option<&'a [u8]> {
    let mut pos = 0;
    while pos + 8 <= buf.len() {
        let size = u32::from_be_bytes(buf[pos..pos + 4].try_into().ok()?) as usize;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn total_segments_ceil() {
        assert_eq!(total_segments_for_rung(100, 48), 3);
        assert_eq!(total_segments_for_rung(96, 48), 2);
        assert_eq!(total_segments_for_rung(0, 48), 0);
        assert_eq!(total_segments_for_rung(100, 0), 0);
    }

    fn contribution(start: u32, end: u32) -> RungContribution {
        let segments = (start..=end)
            .map(|s| SegmentInfo {
                sequence_number: s,
                path: format!("/tmp/seg-{s:05}.m4s").into(),
                byte_size: 1024,
                duration_ticks: 3000,
            })
            .collect();
        RungContribution {
            width: 1280,
            height: 720,
            relative_dir: "video/720p".into(),
            manifest: CmafTrackManifest {
                init_path: "/tmp/init.mp4".into(),
                segments,
                timescale: 30000,
            },
        }
    }

    #[test]
    fn merge_orders_and_dedups() {
        let merged = merge_rung_contributions(vec![contribution(3, 5), contribution(1, 2)]).unwrap();
        let seqs: Vec<u32> = merged.manifest.segments.iter().map(|s| s.sequence_number).collect();
        assert_eq!(seqs, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn merge_detects_duplicate() {
        assert!(merge_rung_contributions(vec![contribution(1, 3), contribution(3, 4)]).is_err());
    }

    #[test]
    fn bandwidth_nonzero() {
        let c = contribution(1, 4);
        let (avg, peak) = measure_bandwidth(&c.manifest);
        assert!(avg > 0);
        assert!(peak >= avg);
    }

    // The merge is what turns N workers' slices of a rung into the rung; the
    // ways it can be wrong are the ways a stitched rendition plays wrong.

    #[test]
    fn merge_single_contribution_passes_through() {
        let c = contribution(1, 30);
        let merged = merge_rung_contributions(vec![c.clone()]).unwrap();
        assert_eq!((merged.width, merged.height), (c.width, c.height));
        assert_eq!(merged.relative_dir, c.relative_dir);
        assert_eq!(merged.manifest.segments.len(), 30);
        assert_eq!(merged.manifest.segments[0].sequence_number, 1);
        assert_eq!(merged.manifest.segments[29].sequence_number, 30);
    }

    #[test]
    fn merge_is_input_order_independent() {
        let a = contribution(1, 15);
        let b = contribution(16, 30);
        let seqs = |m: &RungContribution| -> Vec<u32> {
            m.manifest.segments.iter().map(|s| s.sequence_number).collect()
        };
        let ab = merge_rung_contributions(vec![a.clone(), b.clone()]).unwrap();
        let ba = merge_rung_contributions(vec![b, a]).unwrap();
        assert_eq!(seqs(&ab), (1..=30).collect::<Vec<_>>());
        assert_eq!(seqs(&ab), seqs(&ba));
    }

    #[test]
    fn merge_three_slices_are_strictly_consecutive() {
        let merged =
            merge_rung_contributions(vec![contribution(1, 10), contribution(11, 20), contribution(21, 30)]).unwrap();
        assert_eq!(merged.manifest.segments.len(), 30);
        assert!(merged.manifest.segments.windows(2).all(|w| w[0].sequence_number + 1 == w[1].sequence_number));
    }

    #[test]
    fn merge_rejects_internal_gap() {
        // [1..=5] and [10..=15]: the merge cannot know whether 6..=9 are
        // missing or meant to be, so it refuses to publish a sparse manifest.
        let err = merge_rung_contributions(vec![contribution(1, 5), contribution(10, 15)]).unwrap_err();
        assert!(err.to_string().contains("internal gap"), "{err}");
    }

    #[test]
    fn merge_rejects_disagreeing_contributors() {
        let mut b = contribution(11, 20);
        b.width = 1920;
        b.height = 1080;
        let err = merge_rung_contributions(vec![contribution(1, 10), b]).unwrap_err();
        assert!(err.to_string().contains("disagree on dimensions"), "{err}");

        let mut b = contribution(11, 20);
        b.relative_dir = "video/1080p".into();
        let err = merge_rung_contributions(vec![contribution(1, 10), b]).unwrap_err();
        assert!(err.to_string().contains("disagree on relative_dir"), "{err}");

        let mut b = contribution(11, 20);
        b.manifest.timescale = 90000;
        let err = merge_rung_contributions(vec![contribution(1, 10), b]).unwrap_err();
        assert!(err.to_string().contains("disagree on timescale"), "{err}");
    }

    #[test]
    fn merge_empty_bails_and_init_path_comes_from_first() {
        assert!(merge_rung_contributions(Vec::new()).unwrap_err().to_string().contains("at least one contribution"));
        let mut a = contribution(1, 10);
        a.manifest.init_path = "/tmp/rung-a/init.mp4".into();
        let mut b = contribution(11, 20);
        b.manifest.init_path = "/tmp/rung-b/init.mp4".into();
        let merged = merge_rung_contributions(vec![a, b]).unwrap();
        assert_eq!(merged.manifest.init_path, std::path::PathBuf::from("/tmp/rung-a/init.mp4"));
    }

    #[test]
    fn total_segments_edge_cases() {
        assert_eq!(total_segments_for_rung(300, 60), 5, "exact multiple");
        assert_eq!(total_segments_for_rung(301, 60), 6, "one trailing frame is a segment");
        assert_eq!(total_segments_for_rung(359, 60), 6);
        assert_eq!(total_segments_for_rung(1, 60), 1, "a one-frame source is one segment");
        assert_eq!(total_segments_for_rung(1_296_000, 120), 10_800, "six hours at 60 fps");
    }

    #[test]
    fn keyframe_interval_rounds_to_nearest_frame() {
        assert_eq!(keyframe_interval_for_segment(4.0, 30.0), 120);
        assert_eq!(keyframe_interval_for_segment(4.0, 29.97), 120, "119.88 rounds up");
        assert_eq!(keyframe_interval_for_segment(2.0, 60.0), 120);
        assert_eq!(keyframe_interval_for_segment(6.0, 24.0), 144);
        assert_eq!(keyframe_interval_for_segment(4.0, 23.976), 96);
        assert_eq!(keyframe_interval_for_segment(1.0, 30.0), 30);
        assert_eq!(keyframe_interval_for_segment(0.5, 30.0), 15);
        assert_eq!(keyframe_interval_for_segment(0.3, 30.0), 9);
        assert_eq!(keyframe_interval_for_segment(0.001, 30.0), 1, "never shorter than a frame");
    }

    #[test]
    fn video_flush_happens_before_the_keyframe_that_crosses_the_target() {
        let dir = tempfile::tempdir().unwrap();
        let mut muxer =
            CmafVideoMuxer::new(dir.path(), 1280, 720, 30000, codec::frame::ColorMetadata::default()).unwrap();
        // A synthetic OBU sequence header so the muxer's init sniff is happy;
        // it must be in the first packet only.
        let mut kf_payload = vec![(1u8 << 3) | (1 << 1), 0x01, 0xAA];
        kf_payload.extend_from_slice(&[0xDE, 0xAD]);
        let kf = EncodedPacket { data: bytes::Bytes::from(kf_payload), pts: 0, is_keyframe: true };
        let p = EncodedPacket { data: bytes::Bytes::from(vec![0xBE, 0xEF]), pts: 0, is_keyframe: false };
        let target = 3000; // two 1500-tick frames

        // keyframe at t=0: nothing buffered, no flush.
        assert!(add_packet_with_segment_flush(&mut muxer, &kf, 1500, target).unwrap().is_none());
        assert_eq!(muxer.pending_duration_ticks(), 1500);
        // p-frame at t=1500: buffered == target but not a keyframe, no flush.
        assert!(add_packet_with_segment_flush(&mut muxer, &p, 1500, target).unwrap().is_none());
        assert_eq!(muxer.pending_duration_ticks(), 3000);
        // keyframe at t=3000: buffered at target AND sync → the segment closes
        // BEFORE this packet is added, and is handed back.
        let flushed = add_packet_with_segment_flush(&mut muxer, &kf, 1500, target).unwrap();
        assert!(flushed.is_some(), "the closed segment is returned");
        assert_eq!(muxer.pending_duration_ticks(), 1500);
        assert!(dir.path().join("seg-00001.m4s").exists());
        assert!(dir.path().join("init.mp4").exists());
    }

    #[test]
    fn audio_flush_happens_when_the_target_is_reached() {
        let info = container::AudioInfo::aac_lc(48000, 2, vec![0x11, 0x90]);
        let dir = tempfile::tempdir().unwrap();
        let mut muxer = CmafAudioMuxer::new(dir.path(), info).unwrap();
        // 4 AAC frames of 1024 ticks = 4096 = the target; the fifth add
        // flushes first.
        for _ in 0..4 {
            assert!(add_audio_sample_with_segment_flush(&mut muxer, vec![0xCC; 256], 1024, 4096).unwrap().is_none());
        }
        let flushed = add_audio_sample_with_segment_flush(&mut muxer, vec![0xCC; 256], 1024, 4096).unwrap();
        assert!(flushed.is_some());
        assert_eq!(muxer.pending_duration_ticks(), 1024, "post-flush, just the new sample");
        assert!(dir.path().join("seg-00001.m4s").exists());
    }
}
