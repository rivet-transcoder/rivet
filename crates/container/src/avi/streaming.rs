//! `AviStreamingDemuxer` — pull-based streaming AVI demuxer with two
//! backends: legacy cursor walk (`Backend::Cursor`) and precomputed
//! OpenDML index walk (`Backend::OpenDml`).

use anyhow::{Context, Result, bail};
use codec::frame::{ColorSpace, PixelFormat, StreamInfo};

use crate::demux::AudioTrack;
use crate::streaming::{DemuxHeader, Sample, StreamingDemuxer};

use super::opendml::{locate_stream_indx, parse_ix_chunk, read_avih_total_frames,
                     read_dmlh_total_frames};
use super::riff::{VideoStream, ascii, find_video_stream, fourcc_to_codec,
                  scan_top_level_records};

// ---------------------------------------------------------------------------
// Backend enum
// ---------------------------------------------------------------------------

pub(super) enum Backend {
    /// Walk one or more `LIST movi` records linearly. The Vec is
    /// initialised with one entry per top-level movi LIST in file
    /// order; `rec ` sub-LISTs push additional frames during walk and
    /// pop at EOF. We always operate on the LAST entry (top of stack).
    Cursor(Vec<(usize, usize)>),
    /// Precomputed (absolute_offset_of_chunk_data, data_size) list
    /// drawn from the indx → ix## chain. `cursor` indexes into it.
    OpenDml {
        samples: Vec<(usize, usize)>,
        cursor: usize,
    },
}

// ---------------------------------------------------------------------------
// AviStreamingDemuxer
// ---------------------------------------------------------------------------

/// Streaming AVI demuxer. Owns the input bytes and walks the `movi`
/// LIST(s) one chunk at a time. Two backends:
/// - **Legacy single-movi cursor walk** (`Backend::Cursor`): a stack of
///   (pos, end) frames over a single `LIST movi`. `rec ` sub-LISTs push
///   a new frame; we pop on EOF to resume the parent.
/// - **OpenDML index walk** (`Backend::OpenDml`): a precomputed list of
///   `(absolute byte offset, size)` sample chunks assembled from the
///   stream's `indx` superindex + each `ix##` sub-index. `next_video_sample`
///   advances `cursor` and reads `data[offset..offset+size]`.
/// The streaming impl never holds more than the current sample's bytes
/// regardless of backend.
pub struct AviStreamingDemuxer {
    data: Vec<u8>,
    pub(super) header: DemuxHeader,
    pub(super) backend: Backend,
    /// Two-character stream prefix derived from the video stream's
    /// index. e.g. stream 0 → "00". Only used by the cursor backend.
    prefix: [u8; 2],
    /// Frame index — used as a synthetic monotonic PTS in samples-since-
    /// start. AVI doesn't carry per-sample PTS at the container layer.
    next_idx: u64,
    /// Lazily set on first sample: `pixel_format::detect` is one-shot
    /// against the first sample, so we patch `header.info.pixel_format`
    /// in place once and skip the probe thereafter.
    pixel_format_detected: bool,
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

pub(crate) fn demux_avi_streaming_init(data: &[u8]) -> Result<AviStreamingDemuxer> {
    if data.len() < 12 || &data[..4] != b"RIFF" || &data[8..12] != b"AVI " {
        bail!("not a RIFF/AVI file");
    }
    let owned = data.to_vec();

    let mut hdrl: Option<(usize, usize)> = None;
    let mut movi_lists: Vec<(usize, usize)> = Vec::new();
    scan_top_level_records(&owned, &mut hdrl, &mut movi_lists);

    let (hdrl_start, hdrl_end) = hdrl.context("AVI: missing hdrl LIST")?;
    if movi_lists.is_empty() {
        bail!("AVI: missing movi LIST");
    }

    let video: VideoStream = find_video_stream(&owned[hdrl_start..hdrl_end])
        .context("AVI: no video stream found in hdrl")?;
    let codec = fourcc_to_codec(&video.handler)
        .or_else(|| fourcc_to_codec(&video.compression))
        .with_context(|| {
            format!(
                "AVI: unsupported video fourcc {:?}/{:?}",
                ascii(&video.handler),
                ascii(&video.compression)
            )
        })?;

    let stream_idx = video.stream_index;
    let prefix_str = format!("{:02}", stream_idx);
    let prefix_bytes = prefix_str.as_bytes();
    if prefix_bytes.len() != 2 {
        bail!("AVI: stream index out of range");
    }
    let prefix = [prefix_bytes[0], prefix_bytes[1]];

    // OpenDML detection: look for an `indx` superindex inside the
    // chosen stream's `LIST strl`. Presence triggers the ix##-walking
    // backend; absence falls back to the legacy cursor walk over each
    // `LIST movi` LIST in order.
    let backend =
        if let Some(ix_refs) = locate_stream_indx(&owned[hdrl_start..hdrl_end], stream_idx) {
            // Each `qwOffset` in ix_refs is an absolute file offset to an
            // `ix##` chunk's 8-byte header. Parse each in turn and append
            // its sample chunks to one big list, in superindex order.
            let mut samples: Vec<(usize, usize)> = Vec::new();
            for (ix_off, ix_size) in ix_refs {
                parse_ix_chunk(&owned, ix_off, ix_size, &prefix, &mut samples);
            }
            Backend::OpenDml { samples, cursor: 0 }
        } else {
            Backend::Cursor(movi_lists)
        };

    // total_frames priority for the OpenDML era:
    //   1. `dmlh.dwTotalFrames` inside `LIST hdrl > LIST odml > dmlh`
    //      — the spec-mandated 32-bit count for files that may have
    //      wrapped `avih.dwTotalFrames` (>1 GiB / very long clips).
    //   2. `avih.dwTotalFrames` for legacy single-RIFF files.
    //   3. 0 — same "unknown" sentinel as TS (pipeline tolerates).
    let total_frames = read_dmlh_total_frames(&owned[hdrl_start..hdrl_end])
        .or_else(|| read_avih_total_frames(&owned[hdrl_start..hdrl_end]))
        .unwrap_or(0);
    // Derive duration from total_frames + frame_rate when both are
    // populated — saves the legacy `samples.len() as f64 / frame_rate`
    // computation that needed the materialized Vec.
    let duration = if total_frames > 0 && video.frame_rate > 0.0 {
        total_frames as f64 / video.frame_rate
    } else {
        0.0
    };

    let info = StreamInfo {
        codec: codec.clone(),
        width: video.width,
        height: video.height,
        frame_rate: video.frame_rate,
        duration,
        pixel_format: PixelFormat::Yuv420p,
        color_space: ColorSpace::Bt709,
        color_metadata: Default::default(),
        total_frames,
        bitrate: 0,
    };

    Ok(AviStreamingDemuxer {
        data: owned,
        header: DemuxHeader { codec, info },
        backend,
        prefix,
        next_idx: 0,
        pixel_format_detected: false,
    })
}

// ---------------------------------------------------------------------------
// StreamingDemuxer impl
// ---------------------------------------------------------------------------

impl StreamingDemuxer for AviStreamingDemuxer {
    fn header(&self) -> &DemuxHeader {
        &self.header
    }

    fn next_video_sample(&mut self) -> Result<Option<Sample>> {
        let payload_range = match &mut self.backend {
            Backend::OpenDml { samples, cursor } => {
                loop {
                    if *cursor >= samples.len() {
                        return Ok(None);
                    }
                    let (off, size) = samples[*cursor];
                    *cursor += 1;
                    let end = off
                        .checked_add(size)
                        .ok_or_else(|| anyhow::anyhow!("AVI: ix## entry overflows usize"))?;
                    if end > self.data.len() {
                        // Truncated tail — skip rather than bail; matches
                        // the cursor-walk's "stop on EOF" posture.
                        continue;
                    }
                    break Some((off, end));
                }
            }
            Backend::Cursor(walk) => {
                loop {
                    // Pop empty frames off the walk stack.
                    while let Some(&(pos, end)) = walk.last() {
                        if pos + 8 <= end {
                            break;
                        }
                        walk.pop();
                    }
                    let Some(&mut (ref mut pos, end)) = walk.last_mut() else {
                        return Ok(None);
                    };

                    let fcc: [u8; 4] = self.data[*pos..*pos + 4].try_into()?;
                    let size = u32::from_le_bytes([
                        self.data[*pos + 4],
                        self.data[*pos + 5],
                        self.data[*pos + 6],
                        self.data[*pos + 7],
                    ]) as usize;
                    let payload_start = *pos + 8;
                    let payload_end = payload_start + size;
                    if payload_end > end || payload_end > self.data.len() {
                        // Truncated — pop this frame and resume parent.
                        walk.pop();
                        continue;
                    }

                    // Advance past this chunk on the cursor for the NEXT call.
                    *pos = payload_end + (payload_end & 1);

                    if &fcc == b"LIST" && payload_start + 4 <= payload_end {
                        let list_type: [u8; 4] =
                            self.data[payload_start..payload_start + 4].try_into()?;
                        if &list_type == b"rec " {
                            // Push the inner walk frame and recurse.
                            walk.push((payload_start + 4, payload_end));
                            continue;
                        }
                        continue; // unknown LIST — skip
                    }

                    if fcc[0] != self.prefix[0] || fcc[1] != self.prefix[1] {
                        continue; // wrong stream
                    }
                    let kind = fcc[3];
                    if kind != b'c' && kind != b'b' {
                        continue; // not a video sample chunk
                    }
                    break Some((payload_start, payload_end));
                }
            }
        };
        let Some((start, end)) = payload_range else {
            return Ok(None);
        };

        let pts_ticks = self.next_idx as i64;
        self.next_idx += 1;
        let data = self.data[start..end].to_vec();
        if !self.pixel_format_detected {
            let detected =
                codec::pixel_format::detect(&self.header.codec, std::slice::from_ref(&data));
            self.header.info.pixel_format = detected;
            self.pixel_format_detected = true;
        }
        Ok(Some(Sample {
            data,
            pts_ticks,
            duration_ticks: 0,
        }))
    }

    fn audio(&self) -> Option<&AudioTrack> {
        // AVI audio passthrough is not supported (the legacy path also
        // returns audio: None) — out of scope for this sprint.
        None
    }
}
