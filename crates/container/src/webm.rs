//! The WebM muxer: VP8 or VP9 video, with optional Opus audio, in one
//! Matroska file (`DocType` `webm`).
//!
//! Written from the Matroska specification (RFC 9559, and the EBML of RFC
//! 8794) and the WebM container guidelines: an EBML header, then one
//! Segment holding a SeekHead, Info, Tracks, the Clusters and Cues. Every
//! video key frame opens a Cluster and gets a CuePoint, so a player can seek
//! to it; audio blocks go into the Cluster their time falls in. Blocks are
//! `SimpleBlock`s timed in milliseconds (`TimestampScale` 1 000 000 ns),
//! without lacing.
//!
//! The file is built in memory: rivet's single-file outputs are handed back
//! as bytes anyway, and building it whole lets every size and position be
//! written exactly, with no placeholder to patch and no unknown-size element
//! for a reader to guess at.
//!
//! Opus follows the WebM / Matroska codec mapping: `CodecPrivate` is the
//! `OpusHead` (RFC 7845 §5.1, with its magic), `CodecDelay` the pre-skip and
//! `SeekPreRoll` 80 ms. VP8 and VP9 need no `CodecPrivate`.

use anyhow::{Context, Result, bail};
use frame::{ColorMetadata, EncodedPacket, VideoCodec};

use crate::AudioInfo;
use crate::edit::TrackEdit;

const TIMESTAMP_SCALE_NS: u64 = 1_000_000;
/// Longest span of a cluster: a block's timestamp is a signed 16-bit offset
/// from its cluster's, and five seconds keeps clusters a seekable size even
/// where key frames are far apart.
const MAX_CLUSTER_MS: u64 = 5_000;

/// A WebM file under construction. See the [module docs](self).
pub struct WebmMuxer {
    width: u32,
    height: u32,
    frame_rate: f64,
    codec: VideoCodec,
    color: ColorMetadata,
    /// Video frames in arrival order: data, pts, key.
    video: Vec<(bytes::Bytes, u64, bool)>,
    audio: Option<AudioState>,
    /// Empty time before the first video frame, in milliseconds.
    video_delay_ms: u64,
}

struct AudioState {
    info: AudioInfo,
    edit: TrackEdit,
    /// Samples and their durations, in ticks of `info.timescale`.
    samples: Vec<(Vec<u8>, u32)>,
}

impl WebmMuxer {
    /// A WebM muxer for `codec` (VP8 or VP9) frames at `width` x `height` and
    /// `frame_rate`.
    pub fn new(width: u32, height: u32, frame_rate: f64, codec: VideoCodec) -> Result<Self> {
        if !matches!(codec, VideoCodec::Vp8 | VideoCodec::Vp9) {
            bail!("WebM carries VP8 or VP9 video, not {}", codec.label());
        }
        if !(frame_rate.is_finite() && frame_rate > 0.0) {
            bail!("WebM mux: frame rate {frame_rate} is not a positive number");
        }
        Ok(Self {
            width,
            height,
            frame_rate,
            codec,
            color: ColorMetadata::default(),
            video: Vec::new(),
            audio: None,
            video_delay_ms: 0,
        })
    }

    /// The colour the Video element's `Colour` describes.
    pub fn set_color_metadata(&mut self, color: ColorMetadata) -> &mut Self {
        self.color = color;
        self
    }

    /// Start the video late: `delay` ticks of `timescale` before its first
    /// frame (a source whose video started late, carried through).
    pub fn set_video_delay(&mut self, delay: u64, timescale: u32) -> &mut Self {
        self.video_delay_ms = if timescale == 0 { 0 } else { delay * 1000 / u64::from(timescale) };
        self
    }

    /// Whether `info` can go into a WebM file: Opus, the one audio codec WebM
    /// takes that rivet writes (Vorbis is the other).
    pub fn check_audio(info: &AudioInfo) -> Result<()> {
        if !info.codec.eq_ignore_ascii_case("opus") {
            bail!("WebM carries Opus or Vorbis audio, not {}", info.codec);
        }
        if info.codec_private.len() < 11 {
            bail!("Opus audio without an OpusHead ({} bytes of codec private)", info.codec_private.len());
        }
        if info.timescale != 48_000 {
            bail!("Opus audio is timed at 48 kHz, not {}", info.timescale);
        }
        Ok(())
    }

    /// Add an Opus audio track.
    pub fn with_audio(&mut self, info: AudioInfo) -> Result<&mut Self> {
        Self::check_audio(&info)?;
        self.audio = Some(AudioState { info, edit: TrackEdit::default(), samples: Vec::new() });
        Ok(self)
    }

    /// The audio track's presentation edit: its `delay` starts the audio
    /// late; its `media_time` must be the Opus pre-skip, which `CodecDelay`
    /// carries.
    pub fn set_audio_edit(&mut self, edit: TrackEdit) -> &mut Self {
        if let Some(a) = self.audio.as_mut() {
            a.edit = edit;
        }
        self
    }

    /// One audio packet, `duration` ticks of the track's timescale long.
    pub fn add_audio_sample(&mut self, sample: &[u8], duration: u32) -> Result<()> {
        let a = self.audio.as_mut().context("add_audio_sample before with_audio")?;
        a.samples.push((sample.to_vec(), duration));
        Ok(())
    }

    /// One video frame, in decode order. VP8 and VP9 frames come in display
    /// order, one per packet.
    pub fn add_packet(&mut self, packet: EncodedPacket) -> Result<()> {
        let key = match self.codec {
            VideoCodec::Vp9 => crate::vpx::vp9_frame_info(&packet.data).map(|i| i.key_frame),
            _ => crate::vpx::vp8_frame_info(&packet.data).map(|i| i.key_frame),
        }
        .unwrap_or(packet.is_keyframe);
        if self.video.is_empty() && !key {
            bail!("WebM mux: the first {} frame is not a key frame", self.codec.label());
        }
        self.video.push((packet.data, packet.pts, key));
        Ok(())
    }

    /// The finished file.
    pub fn finalize(self) -> Result<Vec<u8>> {
        if self.video.is_empty() {
            bail!("cannot finalize a WebM file with no video frames");
        }
        let frame_ns = 1e9 / self.frame_rate;
        // Presentation order: a frame's place among the timestamps (VP8 / VP9
        // never reorder, so this is arrival order).
        let mut order: Vec<usize> = (0..self.video.len()).collect();
        order.sort_by_key(|&i| self.video[i].1);
        let mut rank = vec![0usize; self.video.len()];
        for (r, &i) in order.iter().enumerate() {
            rank[i] = r;
        }
        let delay = self.video_delay_ms;
        let video_ms = |i: usize| delay + ((rank[i] as f64 * frame_ns) / TIMESTAMP_SCALE_NS as f64).round() as u64;
        let video_duration_ms = delay + (self.video.len() as f64 * frame_ns / TIMESTAMP_SCALE_NS as f64).round() as u64;

        // Every block, in time order: (ms, track, key, data).
        let mut blocks: Vec<(u64, u8, bool, &[u8])> =
            self.video.iter().enumerate().map(|(i, (d, _, k))| (video_ms(i), 1u8, *k, d.as_ref())).collect();
        let mut audio_duration_ms = 0u64;
        if let Some(a) = &self.audio {
            let ts = u64::from(a.info.timescale.max(1));
            let mut t = a.edit.delay;
            for (data, dur) in &a.samples {
                blocks.push((t * 1000 / ts, 2u8, true, data.as_slice()));
                t += u64::from(*dur);
            }
            audio_duration_ms = t * 1000 / ts;
        }
        // Stable: at one millisecond, video before audio, each in order.
        blocks.sort_by_key(|b| (b.0, b.1));

        // Clusters: a new one at each video key frame, and whenever the
        // current one has run MAX_CLUSTER_MS.
        let mut clusters: Vec<(u64, Vec<u8>)> = Vec::new();
        let mut cue_times: Vec<(u64, usize)> = Vec::new(); // (time, cluster index)
        for (ms, track, key, data) in blocks {
            let video_key = track == 1 && key;
            let open_new = match clusters.last() {
                None => true,
                Some((start, _)) => video_key || ms.saturating_sub(*start) >= MAX_CLUSTER_MS,
            };
            if open_new {
                let mut body = Vec::new();
                put_uint(&mut body, 0xE7, ms); // Timestamp
                clusters.push((ms, body));
            }
            let idx = clusters.len() - 1;
            if video_key && cue_times.last().is_none_or(|(_, c)| *c != idx) {
                cue_times.push((ms, idx));
            }
            let (start, body) = clusters.last_mut().expect("a cluster");
            let rel = i16::try_from(ms as i64 - *start as i64).context("block timestamp outside its cluster")?;
            let mut sb = Vec::with_capacity(data.len() + 4);
            sb.push(0x80 | track); // track number as a one-byte vint
            sb.extend_from_slice(&rel.to_be_bytes());
            sb.push(if key { 0x80 } else { 0x00 });
            sb.extend_from_slice(data);
            put_element(body, 0xA3, &sb); // SimpleBlock
        }
        let clusters: Vec<Vec<u8>> = clusters.into_iter().map(|(_, b)| element(0x1F43B675, &b)).collect();

        let info = {
            let mut b = Vec::new();
            put_uint(&mut b, 0x2AD7B1, TIMESTAMP_SCALE_NS);
            put_float(&mut b, 0x4489, video_duration_ms.max(audio_duration_ms) as f64);
            put_string(&mut b, 0x4D80, "rivet");
            put_string(&mut b, 0x5741, "rivet");
            element(0x1549A966, &b)
        };
        let tracks = element(0x1654AE6B, &self.track_entries());

        // Layout of the Segment's body: SeekHead, Info, Tracks, Clusters,
        // Cues. The SeekHead is a fixed size (8-byte positions), so every
        // position is known before anything is written.
        let seek_head_len = element(0x114D9B74, &seek_head_body(&[(0x1549A966, 0), (0x1654AE6B, 0), (0x1C53BB6B, 0)])).len();
        let info_pos = seek_head_len as u64;
        let tracks_pos = info_pos + info.len() as u64;
        let mut cluster_pos = Vec::with_capacity(clusters.len());
        let mut pos = tracks_pos + tracks.len() as u64;
        for c in &clusters {
            cluster_pos.push(pos);
            pos += c.len() as u64;
        }
        let cues_pos = pos;
        let cues = {
            let mut b = Vec::new();
            for (ms, idx) in &cue_times {
                let mut tp = Vec::new();
                put_uint(&mut tp, 0xF7, 1); // CueTrack
                put_uint(&mut tp, 0xF1, cluster_pos[*idx]); // CueClusterPosition
                let mut cp = Vec::new();
                put_uint(&mut cp, 0xB3, *ms); // CueTime
                put_element(&mut cp, 0xB7, &tp); // CueTrackPositions
                put_element(&mut b, 0xBB, &cp); // CuePoint
            }
            element(0x1C53BB6B, &b)
        };
        let seek_head =
            element(0x114D9B74, &seek_head_body(&[(0x1549A966, info_pos), (0x1654AE6B, tracks_pos), (0x1C53BB6B, cues_pos)]));
        debug_assert_eq!(seek_head.len(), seek_head_len);

        let mut segment_body =
            Vec::with_capacity(seek_head.len() + info.len() + tracks.len() + clusters.iter().map(Vec::len).sum::<usize>() + cues.len());
        segment_body.extend_from_slice(&seek_head);
        segment_body.extend_from_slice(&info);
        segment_body.extend_from_slice(&tracks);
        for c in &clusters {
            segment_body.extend_from_slice(c);
        }
        segment_body.extend_from_slice(&cues);

        let mut out = ebml_header();
        put_element(&mut out, 0x18538067, &segment_body);
        Ok(out)
    }

    /// The `TrackEntry` elements: video as track 1, audio as track 2.
    fn track_entries(&self) -> Vec<u8> {
        let mut video = Vec::new();
        put_uint(&mut video, 0xD7, 1); // TrackNumber
        put_uint(&mut video, 0x73C5, 1); // TrackUID
        put_uint(&mut video, 0x83, 1); // TrackType: video
        put_uint(&mut video, 0x9C, 0); // FlagLacing
        put_string(&mut video, 0x86, if self.codec == VideoCodec::Vp9 { "V_VP9" } else { "V_VP8" });
        put_uint(&mut video, 0x23E383, (1e9 / self.frame_rate).round() as u64); // DefaultDuration
        let mut v = Vec::new();
        put_uint(&mut v, 0xB0, u64::from(self.width)); // PixelWidth
        put_uint(&mut v, 0xBA, u64::from(self.height)); // PixelHeight
        put_element(&mut v, 0x55B0, &colour_body(&self.color)); // Colour
        put_element(&mut video, 0xE0, &v); // Video
        let mut out = element(0xAE, &video);

        if let Some(a) = &self.audio {
            let mut audio = Vec::new();
            put_uint(&mut audio, 0xD7, 2);
            put_uint(&mut audio, 0x73C5, 2);
            put_uint(&mut audio, 0x83, 2); // TrackType: audio
            put_uint(&mut audio, 0x9C, 0);
            put_string(&mut audio, 0x86, "A_OPUS");
            let mut head = b"OpusHead".to_vec();
            head.extend_from_slice(&a.info.codec_private);
            head[8] = 1; // the OpusHead version (a dOps-sourced body says 0)
            put_element(&mut audio, 0x63A2, &head); // CodecPrivate
            let pre_skip = u16::from_le_bytes([a.info.codec_private[2], a.info.codec_private[3]]);
            put_uint(&mut audio, 0x56AA, u64::from(pre_skip) * 1_000_000_000 / 48_000); // CodecDelay
            put_uint(&mut audio, 0x56BB, 80_000_000); // SeekPreRoll
            let mut au = Vec::new();
            put_float(&mut au, 0xB5, 48_000.0); // SamplingFrequency
            put_uint(&mut au, 0x9F, u64::from(a.info.channels)); // Channels
            put_element(&mut audio, 0xE1, &au); // Audio
            out.extend_from_slice(&element(0xAE, &audio));
        }
        out
    }
}

/// The `Colour` element's body: matrix, range, transfer, primaries (H.273
/// codes), and the HDR10 static metadata when there is some.
fn colour_body(c: &ColorMetadata) -> Vec<u8> {
    let mut b = Vec::new();
    put_uint(&mut b, 0x55B1, u64::from(c.matrix_coefficients)); // MatrixCoefficients
    put_uint(&mut b, 0x55B9, if c.full_range { 2 } else { 1 }); // Range
    put_uint(&mut b, 0x55BA, u64::from(crate::mux::transfer_to_h273(c.transfer))); // TransferCharacteristics
    put_uint(&mut b, 0x55BB, u64::from(c.colour_primaries)); // Primaries
    if let Some(cll) = c.content_light_level {
        put_uint(&mut b, 0x55BC, u64::from(cll.max_cll)); // MaxCLL
        put_uint(&mut b, 0x55BD, u64::from(cll.max_fall)); // MaxFALL
    }
    if let Some(m) = c.mastering_display {
        let chroma = |v: u16| f64::from(v) * 0.00002;
        let mut md = Vec::new();
        put_float(&mut md, 0x55D1, chroma(m.primaries_r_x));
        put_float(&mut md, 0x55D2, chroma(m.primaries_r_y));
        put_float(&mut md, 0x55D3, chroma(m.primaries_g_x));
        put_float(&mut md, 0x55D4, chroma(m.primaries_g_y));
        put_float(&mut md, 0x55D5, chroma(m.primaries_b_x));
        put_float(&mut md, 0x55D6, chroma(m.primaries_b_y));
        put_float(&mut md, 0x55D7, chroma(m.white_point_x));
        put_float(&mut md, 0x55D8, chroma(m.white_point_y));
        put_float(&mut md, 0x55D9, f64::from(m.max_luminance) * 0.0001);
        put_float(&mut md, 0x55DA, f64::from(m.min_luminance) * 0.0001);
        put_element(&mut b, 0x55D0, &md); // MasteringMetadata
    }
    b
}

/// The EBML header of a WebM file.
fn ebml_header() -> Vec<u8> {
    let mut b = Vec::new();
    put_uint(&mut b, 0x4286, 1); // EBMLVersion
    put_uint(&mut b, 0x42F7, 1); // EBMLReadVersion
    put_uint(&mut b, 0x42F2, 4); // EBMLMaxIDLength
    put_uint(&mut b, 0x42F3, 8); // EBMLMaxSizeLength
    put_string(&mut b, 0x4282, "webm"); // DocType
    put_uint(&mut b, 0x4287, 4); // DocTypeVersion
    put_uint(&mut b, 0x4285, 2); // DocTypeReadVersion
    element(0x1A45DFA3, &b)
}

/// A SeekHead body with one Seek per `(element id, position)`, positions
/// written as 8-byte integers so the element's size does not depend on them.
fn seek_head_body(entries: &[(u32, u64)]) -> Vec<u8> {
    let mut b = Vec::new();
    for &(id, pos) in entries {
        let mut seek = Vec::new();
        put_element(&mut seek, 0x53AB, &id_bytes(id)); // SeekID
        put_element(&mut seek, 0x53AC, &pos.to_be_bytes()); // SeekPosition
        put_element(&mut b, 0x4DBB, &seek); // Seek
    }
    b
}

fn id_bytes(id: u32) -> Vec<u8> {
    let bytes = id.to_be_bytes();
    let skip = bytes.iter().position(|&b| b != 0).unwrap_or(3);
    bytes[skip..].to_vec()
}

/// The EBML variable-size integer for a data size: the shortest form.
fn size_vint(size: u64) -> Vec<u8> {
    for len in 1..=8u32 {
        // All ones is reserved (unknown size), so the largest value of a
        // length is 2^(7·len) − 2.
        if size < (1u64 << (7 * len)) - 1 {
            let marked = size | (1u64 << (7 * len));
            return marked.to_be_bytes()[(8 - len as usize)..].to_vec();
        }
    }
    panic!("EBML size {size} needs more than 8 bytes")
}

fn element(id: u32, body: &[u8]) -> Vec<u8> {
    let mut out = id_bytes(id);
    out.extend_from_slice(&size_vint(body.len() as u64));
    out.extend_from_slice(body);
    out
}

fn put_element(out: &mut Vec<u8>, id: u32, body: &[u8]) {
    out.extend_from_slice(&element(id, body));
}

fn put_uint(out: &mut Vec<u8>, id: u32, v: u64) {
    let bytes = v.to_be_bytes();
    let skip = bytes.iter().position(|&b| b != 0).unwrap_or(7);
    put_element(out, id, &bytes[skip..]);
}

fn put_float(out: &mut Vec<u8>, id: u32, v: f64) {
    put_element(out, id, &v.to_be_bytes());
}

fn put_string(out: &mut Vec<u8>, id: u32, s: &str) {
    put_element(out, id, s.as_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_are_the_shortest_vint() {
        assert_eq!(size_vint(0), vec![0x80]);
        assert_eq!(size_vint(126), vec![0xFE]);
        assert_eq!(size_vint(127), vec![0x40, 0x7F]);
        assert_eq!(size_vint(16_382), vec![0x7F, 0xFE]);
        assert_eq!(size_vint(16_383), vec![0x20, 0x3F, 0xFF]);
    }

    #[test]
    fn uints_are_minimal_and_ids_keep_their_marker() {
        let mut b = Vec::new();
        put_uint(&mut b, 0xD7, 1);
        assert_eq!(b, vec![0xD7, 0x81, 0x01]);
        let mut b = Vec::new();
        put_uint(&mut b, 0x2AD7B1, 1_000_000);
        assert_eq!(b, vec![0x2A, 0xD7, 0xB1, 0x83, 0x0F, 0x42, 0x40]);
        let mut b = Vec::new();
        put_uint(&mut b, 0x9C, 0);
        assert_eq!(b, vec![0x9C, 0x81, 0x00]);
    }

    #[test]
    fn only_vp8_and_vp9_are_taken() {
        assert!(WebmMuxer::new(64, 48, 30.0, VideoCodec::Vp9).is_ok());
        assert!(WebmMuxer::new(64, 48, 30.0, VideoCodec::Vp8).is_ok());
        assert!(WebmMuxer::new(64, 48, 30.0, VideoCodec::H264).is_err());
        assert!(WebmMuxer::check_audio(&AudioInfo::aac_lc(48_000, 2, vec![0x11, 0x90])).is_err());
    }
}
