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
}
