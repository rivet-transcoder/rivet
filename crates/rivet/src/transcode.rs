//! Single-file transcode: arbitrary input → AV1 + audio MP4.
//!
//! Pipeline shape (no S3 / SQS / multi-variant — this is the single-shot
//! path; for segmented CMAF-HLS or an ABR ladder, drive the `container`
//! and `codec` crates directly):
//!
//! ```text
//! input bytes → demux_streaming → header/audio extraction
//!             → create_decoder (GPU dispatch: NVDEC / QSV)
//!             → for each video sample: push_sample → decode_next loop
//!                 → colorspace::convert_to_yuv420p_bt709
//!                 → encoder.send_frame → receive_packet → muxer.add_packet
//!             → drain decoder → flush encoder → muxer.finalize
//!             → output bytes
//! ```
//!
//! Audio is handled per source codec: AAC / Opus / AC-3 / E-AC-3 pass
//! through verbatim; MP3 / Vorbis are transcoded to Opus (mono through 7.1 —
//! surround goes out over Opus's channel-mapping family 1); anything else is
//! dropped (video-only output) with a warning.

use std::path::Path;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};

use codec::audio::{
    AudioCodec, AudioEncoderConfig, create_decoder as audio_decoder,
    create_encoder as audio_encoder,
};
use codec::colorspace;
use codec::decode;
use codec::encode::{self, EncoderBackend, EncoderConfig};
use container::AudioInfo;
use container::demux::AudioTrack;
use container::mux::Av1Mp4Muxer;
use container::streaming;

/// Outcome of a single in-memory transcode.
#[derive(Debug, Clone)]
pub struct TranscodeOutcome {
    /// Lower-cased input video codec label (e.g. `"h264"`, `"hevc"`, `"av1"`).
    pub input_codec: String,
    /// Lower-cased input audio codec label, if the source carried audio.
    pub input_audio_codec: Option<String>,
    /// Source video dimensions `(width, height)` in pixels.
    pub input_dims: (u32, u32),
    /// Source frame rate in frames per second.
    pub input_frame_rate: f64,
    /// Size of the input buffer in bytes.
    pub input_bytes: usize,
    /// The encoded AV1/MP4 output buffer.
    pub output_bytes: Vec<u8>,
    /// Number of decoded video frames fed to the encoder.
    pub frames_processed: u64,
    /// Number of AV1 packets emitted by the encoder.
    pub packets_emitted: u64,
    /// How the audio track was handled.
    pub audio_handling: AudioHandling,
    /// Wall-clock time spent transcoding.
    pub elapsed: Duration,
}

/// What happened to the source audio track.
#[derive(Debug, Clone)]
pub enum AudioHandling {
    /// No audio track in the source.
    None,
    /// Codec carried through verbatim (AAC / Opus / AC-3 / E-AC-3).
    Passthrough(String),
    /// Source decoded and re-encoded to Opus (MP3 / Vorbis).
    TranscodedToOpus(String),
    /// Source audio dropped — codec unsupported or too many channels.
    Dropped(String),
}

impl AudioHandling {
    /// Human-readable one-line summary.
    pub fn label(&self) -> String {
        match self {
            Self::None => "no audio track".into(),
            Self::Passthrough(c) => format!("{c} passthrough"),
            Self::TranscodedToOpus(c) => format!("{c} → opus transcode"),
            Self::Dropped(c) => format!("{c} dropped (unsupported)"),
        }
    }
}

/// Read `input`, transcode to AV1/MP4, and write the result to `output`.
///
/// Returns the [`TranscodeOutcome`]; `outcome.output_bytes` also holds the
/// bytes that were written to disk.
pub fn transcode_file(input: impl AsRef<Path>, output: impl AsRef<Path>) -> Result<TranscodeOutcome> {
    let input = input.as_ref();
    let output = output.as_ref();
    let bytes = std::fs::read(input)
        .with_context(|| format!("reading input file {}", input.display()))?;
    let outcome = transcode_bytes(&bytes)?;
    std::fs::write(output, &outcome.output_bytes)
        .with_context(|| format!("writing output file {}", output.display()))?;
    Ok(outcome)
}

/// Transcode an in-memory input buffer to an AV1/MP4 output buffer.
///
/// This is the primary library entry point.
pub fn transcode_bytes(input: &[u8]) -> Result<TranscodeOutcome> {
    let started = Instant::now();
    let input_bytes = input.len();

    let mut demuxer = streaming::demux_streaming(input).context("demux")?;
    let header = demuxer.header().clone();
    let codec_lower = header.codec.to_ascii_lowercase();
    // As seen: a 90°/270° source swaps its stored width and height once the
    // decoder below turns the frames upright.
    let input_dims = header.upright_dims();
    let input_frame_rate = header.info.frame_rate;

    // GPU-only dispatch: NVDEC for NVIDIA, QSV for Intel, hard-fail otherwise.
    let decoder: Box<dyn codec::decode::Decoder> =
        decode::create_decoder(&header.codec, header.info.clone()).context("create_decoder")?;
    // Honour the container's rotation, so the output plays the way the source
    // does rather than the way it was stored. 0 is a pass-through.
    let mut decoder = decode::RotatingDecoder::new(decoder, header.rotation_degrees);
    tracing::debug!(codec = %header.codec, rotation_degrees = header.rotation_degrees, "decoder constructed");

    let (target_width, target_height) = input_dims;
    let frame_rate = if header.info.frame_rate > 0.0 {
        header.info.frame_rate.min(60.0)
    } else {
        30.0
    };

    let config = EncoderConfig {
        width: target_width,
        height: target_height,
        frame_rate,
        keyframe_interval: (frame_rate * 2.0) as u32,
        pixel_format: header.info.pixel_format,
        color_metadata: header.info.color_metadata,
        ..EncoderConfig::default()
    };

    // GPU-first encoders. Dev override: set
    // `TRANSCODE_ENCODER_BACKEND=nvenc|amf|qsv|h26x|rav1e` to force a backend;
    // otherwise the auto-select chain (NVENC → AMF → QSV → software, if the
    // build allows it) runs.
    let backend_override = std::env::var("TRANSCODE_ENCODER_BACKEND")
        .ok()
        .and_then(|s| match s.to_ascii_lowercase().as_str() {
            "nvenc" => Some(EncoderBackend::Nvenc),
            "amf" => Some(EncoderBackend::Amf),
            "qsv" => Some(EncoderBackend::Qsv),
            "h26x" => Some(EncoderBackend::H26x),
            "rav1e" => Some(EncoderBackend::Rav1e),
            _ => None,
        });
    tracing::debug!(?backend_override, "encoder backend selection");
    let mut encoder = encode::select_encoder(config, backend_override).context("select_encoder")?;

    let mut muxer =
        Av1Mp4Muxer::new(target_width, target_height, frame_rate).context("Av1Mp4Muxer::new")?;
    muxer.set_color_metadata(header.info.color_metadata);

    let audio_track = demuxer.audio().cloned();
    let input_audio_codec = audio_track.as_ref().map(|t| t.codec.to_ascii_lowercase());
    let audio_handling = wire_audio(&mut muxer, audio_track.as_ref())?;

    let mut frames_processed: u64 = 0;
    let mut packets_emitted: u64 = 0;

    loop {
        match demuxer.next_video_sample().context("next_video_sample")? {
            Some(sample) => {
                decoder.push_sample(&sample.data).context("push_sample")?;
                while let Some(frame) = decoder.decode_next().context("decode_next")? {
                    pump_frame(&mut encoder, &mut muxer, frame, &mut packets_emitted)?;
                    frames_processed += 1;
                }
            }
            None => {
                decoder.finish().context("decoder.finish")?;
                while let Some(frame) = decoder.decode_next().context("decode_next drain")? {
                    pump_frame(&mut encoder, &mut muxer, frame, &mut packets_emitted)?;
                    frames_processed += 1;
                }
                encoder.flush().context("encoder.flush")?;
                while let Some(pkt) = encoder.receive_packet().context("receive_packet drain")? {
                    muxer.add_packet(pkt).context("muxer.add_packet drain")?;
                    packets_emitted += 1;
                }
                break;
            }
        }
    }

    tracing::debug!(
        frames_processed,
        packets_emitted,
        "decode loop complete"
    );
    let output_bytes = muxer.finalize().context("muxer.finalize")?.to_vec();

    Ok(TranscodeOutcome {
        input_codec: codec_lower,
        input_audio_codec,
        input_dims,
        input_frame_rate,
        input_bytes,
        output_bytes,
        frames_processed,
        packets_emitted,
        audio_handling,
        elapsed: started.elapsed(),
    })
}

fn pump_frame(
    encoder: &mut Box<dyn encode::Encoder>,
    muxer: &mut Av1Mp4Muxer,
    frame: codec::frame::VideoFrame,
    packets_out: &mut u64,
) -> Result<()> {
    let normalized =
        colorspace::convert_to_yuv420p_bt709(&frame).context("colorspace conversion")?;
    encoder
        .send_frame(&normalized)
        .context("encoder.send_frame")?;
    while let Some(pkt) = encoder.receive_packet().context("receive_packet")? {
        muxer.add_packet(pkt).context("muxer.add_packet")?;
        *packets_out += 1;
    }
    Ok(())
}

fn wire_audio(muxer: &mut Av1Mp4Muxer, track: Option<&AudioTrack>) -> Result<AudioHandling> {
    let Some(track) = track else {
        return Ok(AudioHandling::None);
    };
    let codec_lower = track.codec.to_ascii_lowercase();

    match codec_lower.as_str() {
        "aac" | "opus" | "ac3" | "eac3" | "dts" => {
            let info = build_passthrough_info(&codec_lower, track);
            if let Err(e) = muxer.with_audio(info) {
                tracing::warn!("with_audio rejected ({e}); emitting video-only");
                return Ok(AudioHandling::Dropped(codec_lower));
            }
            for (sample, dur) in track.samples.iter().zip(track.durations.iter().copied()) {
                muxer
                    .add_audio_sample(sample, 0, dur)
                    .context("muxer.add_audio_sample")?;
            }
            Ok(AudioHandling::Passthrough(codec_lower))
        }
        "mp3" | "vorbis" => {
            let extra: Option<&[u8]> = if track.codec_private.is_empty() {
                None
            } else {
                Some(track.codec_private.as_slice())
            };
            let mut dec =
                audio_decoder(&codec_lower, extra, track.sample_rate, track.channels as u8)
                    .context("codec::audio::create_decoder")?;
            let mut enc = audio_encoder(AudioEncoderConfig {
                codec: AudioCodec::Opus,
                sample_rate: track.sample_rate,
                channels: track.channels as u8,
                // 0 = the encoder's layout-derived default (64k mono, 96k
                // stereo, 320k 5.1). This zero-config entry point has no knob
                // to override it — `run_transcode_job` with an `OutputSpec`
                // does, via `audio_bitrate`.
                bitrate: 0,
            })
            .context("codec::audio::create_encoder (opus)")?;

            let mut out: Vec<(Vec<u8>, u32)> = Vec::new();
            let mut pts: i64 = 0;
            for packet in &track.samples {
                for frame in dec.decode(packet, pts).context("mp3/vorbis decode")? {
                    pts = pts.saturating_add(
                        (frame.samples.len() as i64) / frame.channels.max(1) as i64,
                    );
                    for pkt in enc.encode(&frame).context("opus encode")? {
                        out.push((pkt.data, pkt.duration as u32));
                    }
                }
            }
            for frame in dec.flush().context("mp3/vorbis flush")? {
                for pkt in enc.encode(&frame).context("opus encode (flush)")? {
                    out.push((pkt.data, pkt.duration as u32));
                }
            }
            for pkt in enc.flush().context("opus encoder flush")? {
                out.push((pkt.data, pkt.duration as u32));
            }
            let info = AudioInfo {
                codec: "opus".into(),
                sample_rate: 48_000,
                channels: track.channels,
                timescale: 48_000,
                asc_bytes: Vec::new(),
                codec_private: enc.extra_data(),
            };
            if let Err(e) = muxer.with_audio(info) {
                tracing::warn!("with_audio rejected ({e}); emitting video-only");
                return Ok(AudioHandling::Dropped(codec_lower));
            }
            for (sample, dur) in out {
                muxer
                    .add_audio_sample(&sample, 0, dur)
                    .context("muxer.add_audio_sample (opus)")?;
            }
            Ok(AudioHandling::TranscodedToOpus(codec_lower))
        }
        other => Ok(AudioHandling::Dropped(other.into())),
    }
}

fn build_passthrough_info(codec_lower: &str, track: &AudioTrack) -> AudioInfo {
    let timescale = if codec_lower == "opus" {
        48_000
    } else {
        track.timescale
    };
    AudioInfo {
        codec: codec_lower.into(),
        sample_rate: track.sample_rate,
        channels: track.channels,
        timescale,
        asc_bytes: if codec_lower == "aac" {
            track.asc.clone()
        } else {
            Vec::new()
        },
        codec_private: if codec_lower == "aac" {
            Vec::new()
        } else {
            track.codec_private.clone()
        },
    }
}
