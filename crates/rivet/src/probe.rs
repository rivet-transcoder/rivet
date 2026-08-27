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
    /// Video width in pixels **as displayed** (0 if the container did not
    /// record it). The container's rotation is already applied: a source
    /// stored 1920×1080 with a 90° matrix probes as 1080×1920, because that is
    /// the picture every output is sized against. `stored_width` keeps the
    /// value as recorded.
    pub width: u32,
    /// Video height in pixels as displayed — see `width`.
    pub height: u32,
    /// Width as stored in the container, before rotation.
    pub stored_width: u32,
    /// Height as stored in the container, before rotation.
    pub stored_height: u32,
    /// Clockwise rotation the container asks a player to apply: 0, 90, 180 or
    /// 270. rivet applies it while transcoding, so the output plays upright
    /// with no rotation metadata of its own.
    pub rotation_degrees: u32,
    /// Frame rate in frames per second.
    pub frame_rate: f64,
    /// Duration in seconds (0.0 if the container did not record it).
    pub duration: f64,
    /// Pixel format, e.g. `"Yuv420p"` / `"Yuv420p10le"`.
    pub pixel_format: String,
    /// Audio stream metadata, if present.
    pub audio: Option<AudioStreamInfo>,
    /// Text subtitle tracks rivet can carry, in source order — what
    /// `--subtitles <lang,...>` selects from. Bitmap tracks are not listed.
    pub subtitles: Vec<SubtitleStreamInfo>,
}

/// Text subtitle track metadata.
#[derive(Debug, Clone)]
pub struct SubtitleStreamInfo {
    /// Source format label: `subrip`, `ass`, `webvtt`, `tx3g`.
    pub codec: String,
    /// ISO-639-2 language, or `und`.
    pub language: String,
    /// Number of cues with text.
    pub cues: usize,
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
    let container = container::sniff_container(&input).label().to_string();
    let demuxer = streaming::demux_streaming_shared(input).context("demux")?;
    let header = demuxer.header();

    let audio = demuxer.audio().map(|t| AudioStreamInfo {
        codec: t.codec.to_ascii_lowercase(),
        sample_rate: t.sample_rate,
        channels: t.channels,
    });

    let subtitles = demuxer
        .subtitles()
        .iter()
        .map(|t| SubtitleStreamInfo {
            codec: t.codec.clone(),
            language: t.language.clone(),
            cues: t.cues.len(),
        })
        .collect();

    let (width, height) = header.upright_dims();
    Ok(MediaInfo {
        container,
        video_codec: header.codec.to_ascii_lowercase(),
        width,
        height,
        stored_width: header.info.width,
        stored_height: header.info.height,
        rotation_degrees: header.rotation_degrees,
        frame_rate: header.info.frame_rate,
        duration: header.info.duration,
        pixel_format: format!("{:?}", header.info.pixel_format),
        audio,
        subtitles,
    })
}

