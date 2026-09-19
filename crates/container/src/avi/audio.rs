//! The first audio stream of an AVI: its `strh` / `strf` (WAVEFORMATEX),
//! its `##wb` chunks, and the timeline ffmpeg reads from them.
//!
//! AVI stamps no audio packet with a time. A chunk's time is where it sits
//! in the stream: the stream starts `dwStart` units in, each unit lasts
//! `dwScale / dwRate` seconds, and a chunk spans as many units as
//! ffmpeg's `get_duration` (libavformat/avidec.c) counts —
//!
//! - `dwSampleSize > 0` (constant bitrate — PCM, byte-run MP3): its bytes
//!   over the block size (`nBlockAlign` when it is set, which ffmpeg prefers
//!   when the two disagree);
//! - `dwSampleSize == 0` (one or more whole frames a chunk — AAC, AC-3, VBR
//!   MP3): its bytes over `nBlockAlign`, rounded up, so an empty chunk spans
//!   nothing; with no `nBlockAlign`, one unit, empty or not.
//!
//! ffmpeg's muxer writes empty audio chunks when it rounds the first
//! timestamps onto the stream's units; under the rule above they take no
//! time, and a stream starts where `dwStart` says.
//!
//! What rivet's audio path takes: AAC (the raw form ffmpeg writes, with the
//! AudioSpecificConfig in the WAVEFORMATEX extra bytes), AC-3 / E-AC-3 and
//! DTS one frame to a chunk (passthrough, or decoded to Opus), MP3 / MP2
//! and linear PCM (decoded to Opus). Anything else is surfaced by name
//! with no packets, and the audio stage drops it saying which codec it was.

use crate::demux::AudioTrack;
use crate::edit::AudioEdit;

/// One audio stream's `strh` and `strf`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct AudioStream {
    pub(super) stream_index: u32,
    /// `wFormatTag`; for `WAVE_FORMAT_EXTENSIBLE` the sub-format's tag.
    pub(super) format_tag: u16,
    pub(super) channels: u16,
    pub(super) sample_rate: u32,
    pub(super) block_align: u16,
    pub(super) bits_per_sample: u16,
    /// The bytes after the 18-byte WAVEFORMATEX (`cbSize` of them): the
    /// AudioSpecificConfig for AAC.
    pub(super) extra: Vec<u8>,
    pub(super) scale: u32,
    pub(super) rate: u32,
    pub(super) start: u32,
    pub(super) sample_size: u32,
}

/// `WAVE_FORMAT_EXTENSIBLE`: the real tag is the first two bytes of the
/// sub-format GUID, 8 bytes into the extra bytes.
const WAVE_FORMAT_EXTENSIBLE: u16 = 0xFFFE;

/// The first `auds` stream in `hdrl`, with its index among the streams.
pub(super) fn find_audio_stream(hdrl: &[u8]) -> Option<AudioStream> {
    let mut index = 0u32;
    for (fcc, body) in riff_chunks(hdrl) {
        if fcc != *b"LIST" || body.len() < 4 || &body[..4] != b"strl" {
            continue;
        }
        let mut strh = None;
        let mut strf = None;
        for (fcc, body) in riff_chunks(&body[4..]) {
            match &fcc {
                b"strh" => strh = Some(body),
                b"strf" => strf = Some(body),
                _ => {}
            }
        }
        if let (Some(strh), Some(strf)) = (strh, strf)
            && strh.len() >= 48
            && &strh[..4] == b"auds"
            && strf.len() >= 16
        {
            let u32_at = |b: &[u8], at: usize| u32::from_le_bytes(b[at..at + 4].try_into().unwrap());
            let u16_at = |b: &[u8], at: usize| u16::from_le_bytes([b[at], b[at + 1]]);
            let cb_size = if strf.len() >= 18 { usize::from(u16_at(strf, 16)) } else { 0 };
            let extra = strf.get(18..(18 + cb_size).min(strf.len())).unwrap_or_default().to_vec();
            let mut format_tag = u16_at(strf, 0);
            if format_tag == WAVE_FORMAT_EXTENSIBLE && extra.len() >= 10 {
                format_tag = u16_at(&extra, 6);
            }
            return Some(AudioStream {
                stream_index: index,
                format_tag,
                channels: u16_at(strf, 2),
                sample_rate: u32_at(strf, 4),
                block_align: u16_at(strf, 12),
                bits_per_sample: u16_at(strf, 14),
                extra,
                scale: u32_at(strh, 20),
                rate: u32_at(strh, 24),
                start: u32_at(strh, 28),
                sample_size: u32_at(strh, 44),
            });
        }
        index += 1;
    }
    None
}

/// The `(fourcc, body)` of every chunk in a RIFF list body.
fn riff_chunks(mut data: &[u8]) -> impl Iterator<Item = ([u8; 4], &[u8])> {
    std::iter::from_fn(move || {
        if data.len() < 8 {
            return None;
        }
        let fcc: [u8; 4] = data[..4].try_into().unwrap();
        let size = u32::from_le_bytes(data[4..8].try_into().unwrap()) as usize;
        let end = 8usize.checked_add(size).filter(|&e| e <= data.len())?;
        let body = &data[8..end];
        data = &data[(end + (end & 1)).min(data.len())..];
        Some((fcc, body))
    })
}

/// Every `##wb` chunk of stream `index` in the `LIST movi` bodies, in file
/// order, empty ones included (`rec ` lists walked into).
pub(super) fn collect_audio_chunks<'a>(data: &'a [u8], movi_lists: &[(usize, usize)], index: u32) -> Vec<&'a [u8]> {
    fn walk<'a>(body: &'a [u8], prefix: &[u8; 2], out: &mut Vec<&'a [u8]>) {
        for (fcc, payload) in riff_chunks(body) {
            if &fcc == b"LIST" && payload.len() >= 4 && &payload[..4] == b"rec " {
                walk(&payload[4..], prefix, out);
            } else if fcc[..2] == prefix[..] && &fcc[2..] == b"wb" {
                out.push(payload);
            }
        }
    }
    let digits = format!("{index:02}");
    let prefix: [u8; 2] = digits.as_bytes()[..2].try_into().unwrap();
    let mut out = Vec::new();
    for &(start, end) in movi_lists {
        walk(&data[start..end.min(data.len())], &prefix, &mut out);
    }
    out
}

/// The audio a demuxer hands the pipeline: the track, and the delay before
/// it when the stream starts late.
pub(super) struct AviAudio {
    pub(super) track: AudioTrack,
    pub(super) edit: Option<AudioEdit>,
}

/// The codec name for a `wFormatTag` (and the bits for PCM), or `Err` with
/// a name for one rivet has no path for.
fn codec_for(stream: &AudioStream) -> Result<&'static str, String> {
    Ok(match (stream.format_tag, stream.bits_per_sample) {
        (0x0001, 8) => "pcm_u8",
        (0x0001, 16) => "pcm_s16le",
        (0x0001, 24) => "pcm_s24le",
        (0x0001, 32) => "pcm_s32le",
        (0x0001, bits) => return Err(format!("pcm_{bits}bit")),
        (0x0003, 32) => "pcm_f32le",
        (0x0003, 64) => "pcm_f64le",
        (0x0003, bits) => return Err(format!("pcm_float_{bits}bit")),
        // MPEG-1/2 Layer III, and Layers I/II: one decoder (minimp3) reads all
        // three, which is why Matroska's A_MPEG/L2 is surfaced as `mp3` too.
        (0x0055 | 0x0050, _) => "mp3",
        (0x2000, _) => "ac3",
        (0x2001, _) => "dts",
        // ffmpeg writes AAC as 0x00FF with the ASC in the extra bytes; the
        // other tags seen for raw AAC in the wild carry it the same way.
        (0x00FF | 0x706D | 0x4143 | 0xA106, _) => "aac",
        (0x0002, _) => return Err("adpcm_ms".into()),
        (0x0006, _) => return Err("pcm_alaw".into()),
        (0x0007, _) => return Err("pcm_mulaw".into()),
        (0x0011, _) => return Err("adpcm_ima_wav".into()),
        (0x0160, _) => return Err("wmav1".into()),
        (0x0161, _) => return Err("wmav2".into()),
        (0x0162, _) => return Err("wmapro".into()),
        (0x0163, _) => return Err("wmalossless".into()),
        (0x1600 | 0x1610, _) => return Err("aac_adts".into()),
        (tag, _) => return Err(format!("avi_audio_0x{tag:04x}")),
    })
}

/// A track the pipeline cannot use, named: no packets, and the audio stage
/// drops it saying which codec it was (and refuses an `--audio-filter` on
/// it by name).
fn unusable(name: String, stream: &AudioStream, why: &str) -> AviAudio {
    tracing::warn!(codec = %name, format_tag = format!("0x{:04x}", stream.format_tag), "AVI audio: {why}");
    AviAudio {
        track: AudioTrack {
            codec: name,
            samples: Vec::new(),
            sample_rate: stream.sample_rate,
            channels: stream.channels,
            asc: Vec::new(),
            codec_private: Vec::new(),
            timescale: stream.sample_rate.max(1),
            durations: Vec::new(),
        },
        edit: None,
    }
}

/// The first audio stream of the AVI whose `hdrl` and `movi` bodies are
/// given, as the pipeline takes it; `None` when the file has no audio
/// stream or no audio chunks.
pub(super) fn read_audio(data: &[u8], hdrl: &[u8], movi_lists: &[(usize, usize)]) -> Option<AviAudio> {
    let stream = find_audio_stream(hdrl)?;
    let chunks = collect_audio_chunks(data, movi_lists, stream.stream_index);
    if chunks.iter().all(|c| c.is_empty()) {
        return None;
    }
    let codec = match codec_for(&stream) {
        Ok(codec) => codec,
        Err(name) => return Some(unusable(name, &stream, "no passthrough form and no decoder for this format")),
    };
    if stream.scale == 0 || stream.rate == 0 {
        return Some(unusable(codec.into(), &stream, "strh dwScale / dwRate unset: the stream has no timeline"));
    }

    let mut sample_rate = stream.sample_rate;
    let mut channels = stream.channels;
    let mut asc = Vec::new();
    let mut codec_private = Vec::new();
    let first = chunks.iter().find(|c| !c.is_empty()).copied().unwrap_or_default();
    let mut codec = codec.to_string();
    match codec.as_str() {
        "aac" => {
            // The raw form needs its ASC; ADTS-framed AAC carries none here.
            let Some(parsed) = crate::aac_asc::parse_aac_asc(&stream.extra) else {
                return Some(unusable("aac_adts".into(), &stream, "AAC without an AudioSpecificConfig (ADTS framing) is not supported"));
            };
            channels = crate::aac_asc::effective_output_channels(&parsed);
            sample_rate = parsed.sbr_sample_rate.unwrap_or(parsed.sample_rate);
            asc = stream.extra.clone();
        }
        "ac3" => {
            // Passthrough needs whole syncframes, one to a sample.
            if chunks.iter().any(|c| !c.is_empty() && !c.starts_with(&[0x0B, 0x77])) {
                return Some(unusable(codec, &stream, "AC-3 chunks that are not whole syncframes are not supported"));
            }
            match crate::ac3_sync::parse_sync_info(first) {
                Ok(crate::ac3_sync::SyncInfo::Ac3(s)) => {
                    codec_private = crate::mux::dac3_body_from_sync(&s).to_vec();
                    (sample_rate, channels) =
                        crate::demux::audio::ac3_sample_rate_channels_from_dac3(&codec_private)?;
                }
                Ok(crate::ac3_sync::SyncInfo::Eac3(s)) => {
                    sample_rate = crate::ac3_sync::eac3_sample_rate_hz(s.fscod, s.fscod2);
                    channels = crate::ac3_sync::channel_count(s.acmod, s.lfeon);
                    let spf = u64::from(crate::ac3_sync::eac3_samples_per_frame(s.numblkscod));
                    let frame_bytes = (u64::from(s.frmsiz) + 1) * 2;
                    let kbps = (frame_bytes * 8 * u64::from(sample_rate)).checked_div(spf).map_or(0, |bps| bps / 1000);
                    codec_private = crate::mux::dec3_body_from_sync(&s, kbps.div_ceil(2) as u16).to_vec();
                    codec = "eac3".into();
                }
                Err(e) => return Some(unusable(codec, &stream, &format!("first AC-3 frame: {e}"))),
            }
        }
        "dts" => {
            if chunks.iter().any(|c| !c.is_empty() && !c.starts_with(&[0x7F, 0xFE, 0x80, 0x01])) {
                return Some(unusable(codec, &stream, "DTS chunks that are not whole core frames are not supported"));
            }
            match crate::dts_sync::parse_core_sync(first) {
                Ok(core) => {
                    let hd = crate::dts_sync::has_hd_extension(first, &core);
                    codec_private = crate::mux::ddts_body_from_sync(&core, hd);
                    sample_rate = core.sample_rate;
                    channels = core.channels;
                }
                Err(e) => return Some(unusable(codec, &stream, &format!("first DTS frame: {e}"))),
            }
        }
        _ => {}
    }
    if sample_rate == 0 || channels == 0 {
        return Some(unusable(codec, &stream, "no sample rate or channel count"));
    }

    // Each chunk's extent in stream units (ffmpeg's `get_duration`), then
    // every non-empty chunk's start on the track's timescale: the units
    // before it, `scale / rate` seconds each, from `dwStart`.
    let block = if stream.sample_size > 0 && stream.block_align > 0 {
        u64::from(stream.block_align)
    } else {
        u64::from(stream.sample_size)
    };
    let units = |len: usize| -> u64 {
        let len = len as u64;
        if stream.sample_size > 0 {
            len / block.max(1)
        } else if stream.block_align > 0 {
            len.div_ceil(u64::from(stream.block_align))
        } else {
            1
        }
    };
    let timescale = sample_rate;
    let ticks =
        |units: u64| (u128::from(units) * u128::from(stream.scale) * u128::from(timescale) / u128::from(stream.rate)) as u64;
    let mut samples = Vec::new();
    let mut starts = Vec::new();
    let mut at = u64::from(stream.start);
    for chunk in &chunks {
        if !chunk.is_empty() {
            starts.push(ticks(at));
            samples.push(chunk.to_vec());
        }
        at += units(chunk.len());
    }
    let end = ticks(at);
    let delay = starts[0];
    let durations: Vec<u32> = starts
        .iter()
        .zip(starts.iter().skip(1).chain(std::iter::once(&end)))
        .map(|(a, b)| (b - a).max(1) as u32)
        .collect();
    let edit = (delay > 0).then_some(AudioEdit { delay, media_start: 0, media_end: None });
    Some(AviAudio {
        track: AudioTrack { codec, samples, sample_rate, channels, asc, codec_private, timescale, durations },
        edit,
    })
}
