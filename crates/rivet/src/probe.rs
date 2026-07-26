//! Inspect an input without transcoding it.
//!
//! Demuxes just the container header + audio track metadata and reports the
//! video codec, dimensions, frame rate, pixel format, and audio stream
//! shape. Works across every container the [`container`] crate supports
//! (MP4/MOV, MKV/WebM, AVI, MPEG-TS).

use std::path::Path;

use anyhow::{Context, Result};

use container::streaming;

/// Probed media metadata.
#[derive(Debug, Clone)]
pub struct MediaInfo {
    /// Detected container label: `"mp4"`, `"mkv"`, `"avi"`, or `"ts"`.
    pub container: String,
    /// Lower-cased video codec label (e.g. `"h264"`, `"hevc"`, `"av1"`).
    pub video_codec: String,
    /// Video width in pixels (0 if the container did not record it).
    pub width: u32,
    /// Video height in pixels (0 if the container did not record it).
    pub height: u32,
    /// Frame rate in frames per second.
    pub frame_rate: f64,
    /// Duration in seconds (0.0 if the container did not record it).
    pub duration: f64,
    /// Pixel format, e.g. `"Yuv420p"` / `"Yuv420p10le"`.
    pub pixel_format: String,
    /// Audio stream metadata, if present.
    pub audio: Option<AudioStreamInfo>,
}

/// Audio stream metadata.
#[derive(Debug, Clone)]
pub struct AudioStreamInfo {
    /// Lower-cased audio codec label (e.g. `"aac"`, `"opus"`, `"mp3"`).
    pub codec: String,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Channel count.
    pub channels: u16,
}

/// Probe an input file.
pub fn probe_file(input: impl AsRef<Path>) -> Result<MediaInfo> {
    let input = input.as_ref();
    let bytes = std::fs::read(input)
        .with_context(|| format!("reading input file {}", input.display()))?;
    probe_bytes(&bytes)
}

/// Probe an in-memory input buffer.
pub fn probe_bytes(input: &[u8]) -> Result<MediaInfo> {
    probe_bytes_shared(bytes::Bytes::copy_from_slice(input))
}

/// [`probe_bytes`] over a buffer the caller already owns — no copy. Worth
/// using whenever the same bytes are about to be transcoded as well.
pub fn probe_bytes_shared(input: bytes::Bytes) -> Result<MediaInfo> {
    let container = detect_container(&input).to_string();
    let demuxer = streaming::demux_streaming_shared(input).context("demux")?;
    let header = demuxer.header();

    let audio = demuxer.audio().map(|t| AudioStreamInfo {
        codec: t.codec.to_ascii_lowercase(),
        sample_rate: t.sample_rate,
        channels: t.channels,
    });

    Ok(MediaInfo {
        container,
        video_codec: header.codec.to_ascii_lowercase(),
        width: header.info.width,
        height: header.info.height,
        frame_rate: header.info.frame_rate,
        duration: header.info.duration,
        pixel_format: format!("{:?}", header.info.pixel_format),
        audio,
    })
}

/// Magic-byte container detector — mirrors the dispatch in
/// [`container::streaming::demux_streaming`] so the reported label matches
/// the demuxer that was actually used.
fn detect_container(data: &[u8]) -> &'static str {
    if data.len() < 12 {
        return "unknown";
    }
    if &data[4..8] == b"ftyp" || &data[4..8] == b"moov" || &data[4..8] == b"mdat" {
        return "mp4";
    }
    if data[0] == 0x1A && data[1] == 0x45 && data[2] == 0xDF && data[3] == 0xA3 {
        return "mkv";
    }
    if &data[..4] == b"RIFF" && &data[8..12] == b"AVI " {
        return "avi";
    }
    if data[0] == 0x47
        && data.len() > 188
        && data[188] == 0x47
        && (data.len() <= 376 || data[376] == 0x47)
    {
        return "ts";
    }
    "unknown"
}
