//! Stream probe — analyze video files without FFmpeg.
//!
//! Extracts codec, resolution, frame rate, duration, and container
//! metadata from MP4/MOV files using the mp4 crate.

use anyhow::{Context, Result};
use std::io::Cursor;

use crate::frame::{
    ColorMetadata, ColorSpace, ContentLightLevel, MasteringDisplay, PixelFormat, StreamInfo,
};
use crate::hevc_sei;

#[derive(Debug, Clone)]
pub struct ProbeResult {
    pub stream_info: StreamInfo,
    pub container: String,
    pub audio_codec: Option<String>,
    pub audio_sample_rate: Option<u32>,
    pub audio_channels: Option<u16>,
    pub file_size: u64,
    pub metadata: std::collections::HashMap<String, String>,
}

pub fn probe_mp4(data: &[u8]) -> Result<ProbeResult> {
    let size = data.len() as u64;
    let cursor = Cursor::new(data);
    let reader =
        mp4::Mp4Reader::read_header(cursor, size).context("reading MP4 header for probe")?;

    let video_track = reader
        .tracks()
        .values()
        .find(|t| t.track_type().ok() == Some(mp4::TrackType::Video))
        .context("no video track")?;

    let codec = match video_track.media_type() {
        Ok(mp4::MediaType::H264) => "h264",
        Ok(mp4::MediaType::H265) => "h265",
        Ok(mp4::MediaType::VP9) => "vp9",
        _ => "unknown",
    };

    let width = video_track.width() as u32;
    let height = video_track.height() as u32;
    let sample_count = video_track.sample_count();
    let duration = video_track.duration().as_secs_f64();
    let frame_rate = if duration > 0.0 {
        sample_count as f64 / duration
    } else {
        30.0
    };
    let bitrate = video_track.bitrate() as u64;

    let audio_track = reader
        .tracks()
        .values()
        .find(|t| t.track_type().ok() == Some(mp4::TrackType::Audio));

    let audio_codec = audio_track.and_then(|t| t.media_type().ok().map(|mt| format!("{mt:?}")));
    let audio_sample_rate: Option<u32> = None;
    let audio_channels: Option<u16> = None;

    // Squad-21: extract HDR static-metadata boxes (`mdcv`, `clli`) from
    // the visual sample entry. These are the canonical container-side
    // HDR10 carriers — without surfacing them, the muxer can't write
    // them on the output and Apple devices fall back to BT.709 limited.
    let probe_color = probe_mp4_visual_color_metadata(data);
    let color_metadata = ColorMetadata {
        mastering_display: probe_color.mastering_display,
        content_light_level: probe_color.content_light_level,
        ..ColorMetadata::default()
    };

    let stream_info = StreamInfo {
        codec: codec.to_string(),
        width,
        height,
        frame_rate,
        duration,
        pixel_format: PixelFormat::Yuv420p,
        color_space: ColorSpace::Bt709,
        total_frames: sample_count as u64,
        bitrate,
        color_metadata,
    };

    Ok(ProbeResult {
        stream_info,
        container: "mp4".to_string(),
        audio_codec,
        audio_sample_rate,
        audio_channels,
        file_size: size,
        metadata: std::collections::HashMap::new(),
    })
}

/// Squad-21: HDR static metadata pulled from MP4 visual sample-entry
/// boxes (`mdcv`, `clli`). Returns `None` for SDR sources or non-MP4
/// inputs.
#[derive(Debug, Default, Clone, Copy)]
struct ProbeMp4VisualColorMetadata {
    mastering_display: Option<MasteringDisplay>,
    content_light_level: Option<ContentLightLevel>,
}

fn probe_mp4_visual_color_metadata(data: &[u8]) -> ProbeMp4VisualColorMetadata {
    let path: &[&[u8; 4]] = &[b"moov", b"trak", b"mdia", b"minf", b"stbl", b"stsd"];
    let Some(stsd_body) = find_box_body(data, path) else {
        return ProbeMp4VisualColorMetadata::default();
    };
    if stsd_body.len() < 16 {
        return ProbeMp4VisualColorMetadata::default();
    }

    let mut pos = 8;
    while pos + 8 <= stsd_body.len() {
        let entry_size = u32::from_be_bytes([
            stsd_body[pos],
            stsd_body[pos + 1],
            stsd_body[pos + 2],
            stsd_body[pos + 3],
        ]) as usize;
        if entry_size < 8 || pos.saturating_add(entry_size) > stsd_body.len() {
            break;
        }
        let entry_type: [u8; 4] = match stsd_body[pos + 4..pos + 8].try_into() {
            Ok(v) => v,
            Err(_) => break,
        };
        let is_visual = matches!(
            &entry_type,
            b"av01"
                | b"avc1"
                | b"avc3"
                | b"hvc1"
                | b"hev1"
                | b"hvc2"
                | b"hev2"
                | b"dvh1"
                | b"dvhe"
                | b"vp08"
                | b"vp09"
                | b"apcn"
                | b"apch"
                | b"apcs"
                | b"apco"
                | b"ap4h"
                | b"ap4x"
        );
        if !is_visual {
            pos = pos.saturating_add(entry_size);
            continue;
        }
        let end = pos.saturating_add(entry_size);
        let child_start = pos + 8 + 78;
        if child_start >= end {
            return ProbeMp4VisualColorMetadata::default();
        }
        let children = &stsd_body[child_start..end];
        let mut out = ProbeMp4VisualColorMetadata::default();
        if let Some(mdcv) = find_direct_child(children, b"mdcv")
            && mdcv.len() >= 24
        {
            let u16be = |o: usize| u16::from_be_bytes([mdcv[o], mdcv[o + 1]]);
            let u32be =
                |o: usize| u32::from_be_bytes([mdcv[o], mdcv[o + 1], mdcv[o + 2], mdcv[o + 3]]);
            out.mastering_display = Some(MasteringDisplay {
                primaries_g_x: u16be(0),
                primaries_g_y: u16be(2),
                primaries_b_x: u16be(4),
                primaries_b_y: u16be(6),
                primaries_r_x: u16be(8),
                primaries_r_y: u16be(10),
                white_point_x: u16be(12),
                white_point_y: u16be(14),
                max_luminance: u32be(16),
                min_luminance: u32be(20),
            });
        }
        if let Some(clli) = find_direct_child(children, b"clli")
            && clli.len() >= 4
        {
            out.content_light_level = Some(ContentLightLevel {
                max_cll: u16::from_be_bytes([clli[0], clli[1]]),
                max_fall: u16::from_be_bytes([clli[2], clli[3]]),
            });
        }
        return out;
    }
    ProbeMp4VisualColorMetadata::default()
}

/// Walk the ISOBMFF box tree following `path` and return the body bytes
/// of the deepest box (or None if any hop is missing). Local helper —
/// duplicates `container::demux::find_box_body` to avoid making
/// `codec` depend on `container`.
fn find_box_body<'a>(data: &'a [u8], path: &[&[u8; 4]]) -> Option<&'a [u8]> {
    let mut slice = data;
    for (i, target) in path.iter().enumerate() {
        let found = find_direct_child(slice, target)?;
        if i + 1 == path.len() {
            return Some(found);
        }
        slice = found;
    }
    None
}

fn find_direct_child<'a>(data: &'a [u8], target: &[u8; 4]) -> Option<&'a [u8]> {
    let mut pos = 0;
    while pos + 8 <= data.len() {
        let size =
            u32::from_be_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        let btype = &data[pos + 4..pos + 8];
        if size < 8 || pos.checked_add(size).is_none_or(|end| end > data.len()) {
            return None;
        }
        if btype == target {
            return Some(&data[pos + 8..pos + size]);
        }
        pos += size;
    }
    None
}

/// Re-export of the SEI parser entry-point for downstream callers
/// that want to scan an Annex-B HEVC bitstream directly without
/// constructing a decoder. The decoder also folds this internally
/// (see `decode::hevc_de265::De265Decoder::new`).
pub use hevc_sei::parse_annexb as parse_hevc_hdr_sei;

pub fn detect_container(data: &[u8]) -> &'static str {
    if data.len() < 12 {
        return "unknown";
    }
    // MP4/MOV: ftyp box at offset 4
    if &data[4..8] == b"ftyp" {
        return "mp4";
    }
    // MKV/WebM: EBML header
    if data[0] == 0x1A && data[1] == 0x45 && data[2] == 0xDF && data[3] == 0xA3 {
        return "mkv";
    }
    // AVI: RIFF header
    if &data[0..4] == b"RIFF" && data.len() > 11 && &data[8..12] == b"AVI " {
        return "avi";
    }
    "unknown"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_container_mp4() {
        let mut data = vec![0u8; 16];
        data[4..8].copy_from_slice(b"ftyp");
        assert_eq!(detect_container(&data), "mp4");
    }

    #[test]
    fn test_detect_container_mkv() {
        let data = vec![
            0x1A, 0x45, 0xDF, 0xA3, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        ];
        assert_eq!(detect_container(&data), "mkv");
    }

    #[test]
    fn test_detect_container_avi() {
        let mut data = vec![0u8; 16];
        data[0..4].copy_from_slice(b"RIFF");
        data[8..12].copy_from_slice(b"AVI ");
        assert_eq!(detect_container(&data), "avi");
    }

    #[test]
    fn test_detect_container_unknown() {
        let data = vec![0xFF; 16];
        assert_eq!(detect_container(&data), "unknown");
    }

    #[test]
    fn test_detect_container_short() {
        let data = vec![0u8; 4];
        assert_eq!(detect_container(&data), "unknown");
    }
}
