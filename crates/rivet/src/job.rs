//! The transcode job engine.
//!
//! [`run_job`] takes an input buffer and an [`OutputSpec`] and drives the
//! whole pipeline: demux → shared decode pump (decode once) → fan out to N
//! per-rung workers (scale + encode + mux) → assemble the requested output
//! mode (single MP4 per rung, or a CMAF/HLS package). Progress is streamed
//! through a [`ProgressSink`] as a uniform [`RungProgress`] per rung.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use bytes::Bytes;

use codec::audio::{
    AudioCodec, AudioEncoderConfig, create_decoder as audio_decoder,
    create_encoder as audio_encoder,
};
use codec::codec_strings::av1_codec_string;
use codec::encode::{self, EncodedPacket, EncoderBackend, EncoderConfig};
use codec::frame::{ColorMetadata, PixelFormat, VideoFrame};
use codec::pixel_format::parse_av1_sequence_header;
use codec::{codec_strings, colorspace};
use container::cmaf::{CmafAudioMuxer, CmafTrackManifest, CmafVideoMuxer};
use container::demux::AudioTrack;
use container::hls::{AudioVariantSpec, VideoVariantSpec, write_hls_package};
use container::mux::Av1Mp4Muxer;
use container::{AudioInfo, streaming};

use crate::decode_pump::{DecodePumpConfig, run_shared_decode_pump_blocking};
use crate::progress::{JobEvent, ProgressSink, RungProgress, RungStatus};
use crate::spec::{AudioPolicy, OutputMode, OutputSpec, Rung};
use crate::validate::needs_chroma_downsample;

/// Bounded per-rung frame channel — backpressures the decode pump.
const FRAME_CHANNEL_CAPACITY: usize = 8;

/// The artifact one rung produced.
#[derive(Debug)]
pub enum RungArtifact {
    /// A single self-contained file (MP4 bytes).
    File(Vec<u8>),
    /// An HLS rendition: a directory of CMAF segments + a media playlist,
    /// referenced from the master playlist by `relative_dir`.
    HlsRendition {
        dir: PathBuf,
        relative_dir: String,
    },
}

/// Result for one completed rung.
#[derive(Debug)]
pub struct RungOutput {
    pub label: String,
    pub width: u32,
    pub height: u32,
    pub frames: u64,
    pub bytes: u64,
    pub artifact: RungArtifact,
}

/// The full job result.
#[derive(Debug)]
pub struct JobOutput {
    /// One entry per rung that completed successfully (failed rungs are
    /// reported via the progress sink with [`RungStatus::Failed`]).
    pub rungs: Vec<RungOutput>,
    /// HLS mode only: the asset root directory containing `master.m3u8`,
    /// `audio/`, and `video/<label>/`.
    pub hls_root: Option<PathBuf>,
    /// HLS mode only: path to the master playlist.
    pub master_playlist: Option<PathBuf>,
    /// Source video codec.
    pub source_codec: String,
    /// Source dimensions.
    pub source_dims: (u32, u32),
    /// Source frame rate.
    pub source_frame_rate: f64,
    /// How the audio was handled (`"aac passthrough"`, `"mp3 → opus"`, ...).
    pub audio_handling: String,
    /// Wall-clock time.
    pub elapsed: Duration,
}

/// Run a transcode job. Async — call from within a Tokio runtime.
///
/// For [`OutputMode::Hls`], `output_dir` is the asset root the HLS package is
/// written under; pass `None` to use a fresh temp directory (returned in
/// [`JobOutput::hls_root`], which the caller then owns). For
/// [`OutputMode::SingleFile`] `output_dir` is ignored (bytes are returned).
pub async fn run_job(
    input: Bytes,
    spec: &OutputSpec,
    output_dir: Option<&Path>,
    sink: Arc<dyn ProgressSink>,
) -> Result<JobOutput> {
    let started = Instant::now();
    spec.validate().context("invalid OutputSpec")?;

    // --- Demux header + audio track ---
    let (header, audio_track) = {
        let demuxer = streaming::demux_streaming(&input).context("demux")?;
        (demuxer.header().clone(), demuxer.audio().cloned())
    };
    let source_codec = header.codec.to_ascii_lowercase();
    let source_dims = (header.info.width, header.info.height);
    let source_frame_rate = header.info.frame_rate;

    sink.on_event(JobEvent::Started {
        rungs: spec.rungs.len(),
    });
    sink.on_event(JobEvent::Probed {
        codec: source_codec.clone(),
        width: header.info.width,
        height: header.info.height,
        frame_rate: header.info.frame_rate,
        audio_codec: audio_track.as_ref().map(|t| t.codec.to_ascii_lowercase()),
    });

    // Effective frame rate: source, clamped by the spec cap.
    let frame_rate = {
        let mut fr = if header.info.frame_rate > 0.0 {
            header.info.frame_rate
        } else {
            30.0
        };
        if let Some(cap) = spec.max_frame_rate {
            fr = fr.min(cap);
        }
        fr
    };

    let frames_total = if header.info.total_frames > 0 {
        Some(header.info.total_frames)
    } else {
        None
    };

    // --- Prepare audio once (shared across rungs / used for the audio rendition) ---
    let prepared_audio = prepare_audio(audio_track.as_ref(), spec.audio)
        .context("preparing audio")?;
    let audio_handling = prepared_audio
        .as_ref()
        .map(|a| a.handling.clone())
        .unwrap_or_else(|| "none".to_string());

    // Base encoder config common to every rung (per-rung sets dims + quality).
    let backend_override = encoder_backend_override();
    let base_cfg = EncoderConfig {
        frame_rate,
        // The decode pump tonemaps + normalizes to 8-bit SDR BT.709, so every
        // rung encodes Yuv420p with default SDR color metadata.
        pixel_format: PixelFormat::Yuv420p,
        color_metadata: ColorMetadata::default(),
        gpu_index: spec.gpu_index,
        ..EncoderConfig::default()
    };

    let pump_cfg = DecodePumpConfig {
        codec_name: header.codec.clone(),
        info_for_decoder: header.info.clone(),
        source_color_metadata: header.info.color_metadata,
        source_pixel_format: header.info.pixel_format,
        needs_downsample: needs_chroma_downsample(header.info.pixel_format),
        gpu_index: spec.gpu_index,
    };

    let rt = tokio::runtime::Handle::current();

    // --- Spawn per-rung workers + the shared decode pump ---
    let mut senders = Vec::with_capacity(spec.rungs.len());
    let mut handles = Vec::with_capacity(spec.rungs.len());

    // HLS asset root (only used in HLS mode).
    let hls_root: Option<PathBuf> = match &spec.mode {
        OutputMode::Hls { .. } => Some(match output_dir {
            Some(d) => d.to_path_buf(),
            None => {
                let tmp = tempfile::Builder::new()
                    .prefix("rivet-hls-")
                    .tempdir()
                    .context("creating HLS temp dir")?;
                tmp.keep()
            }
        }),
        OutputMode::SingleFile => None,
    };

    for (idx, rung) in spec.rungs.iter().cloned().enumerate() {
        let (tx, rx) = tokio::sync::mpsc::channel::<VideoFrame>(FRAME_CHANNEL_CAPACITY);
        senders.push(tx);

        let sink = Arc::clone(&sink);
        let base_cfg = base_cfg.clone();
        let mode = spec.mode.clone();
        let audio = prepared_audio.clone();
        let hls_root = hls_root.clone();

        let handle = tokio::task::spawn_blocking(move || {
            let result = match &mode {
                OutputMode::SingleFile => encode_rung_single_file(
                    idx,
                    &rung,
                    rx,
                    base_cfg,
                    backend_override,
                    frame_rate,
                    frames_total,
                    audio.as_ref(),
                    sink.as_ref(),
                ),
                OutputMode::Hls { segment_seconds } => {
                    let root = hls_root.as_ref().expect("hls_root set in HLS mode");
                    encode_rung_hls(
                        idx,
                        &rung,
                        rx,
                        base_cfg,
                        backend_override,
                        frame_rate,
                        *segment_seconds,
                        frames_total,
                        root,
                        sink.as_ref(),
                    )
                }
            };
            (idx, rung, result)
        });
        handles.push(handle);
    }

    // Shared decode pump on a blocking thread, fanning frames to all rungs.
    let pump_handle = {
        let input = input.clone();
        let rt = rt.clone();
        tokio::task::spawn_blocking(move || {
            run_shared_decode_pump_blocking(pump_cfg, input, senders, rt)
        })
    };

    // Collect rung results.
    let mut rung_outputs: Vec<RungOutput> = Vec::new();
    let mut video_specs: Vec<VideoVariantSpec> = Vec::new();
    for handle in handles {
        let (idx, rung, result) = handle.await.context("rung worker task panicked")?;
        match result {
            Ok((out, manifest)) => {
                if let Some(manifest) = manifest {
                    // HLS rung: build its master-playlist spec.
                    match build_video_variant_spec(&rung, &out, frame_rate, manifest) {
                        Ok(spec) => video_specs.push(spec),
                        Err(e) => {
                            tracing::warn!(rung = %rung.label, error = %e, "could not build HLS variant spec");
                            report_failed(sink.as_ref(), idx, &rung, &e.to_string());
                            continue;
                        }
                    }
                }
                rung_outputs.push(out);
            }
            Err(e) => {
                tracing::warn!(rung = %rung.label, error = %e, "rung failed");
                report_failed(sink.as_ref(), idx, &rung, &e.to_string());
            }
        }
    }

    let frames_pushed = pump_handle
        .await
        .context("decode pump task panicked")?
        .context("decode pump failed")?;
    tracing::debug!(frames_pushed, "decode pump finished");

    if rung_outputs.is_empty() {
        bail!("all {} rung(s) failed", spec.rungs.len());
    }

    // --- Assemble HLS package (audio rendition + playlists) ---
    let mut master_playlist = None;
    if let (OutputMode::Hls { segment_seconds }, Some(root)) = (&spec.mode, hls_root.as_ref()) {
        let audio_spec = match prepared_audio.as_ref() {
            Some(a) => build_audio_rendition(root, a, *segment_seconds)
                .context("building HLS audio rendition")?,
            None => None,
        };
        let target_duration = segment_seconds.ceil() as u32;
        let paths = write_hls_package(root, &video_specs, audio_spec.as_ref(), target_duration)
            .context("writing HLS package")?;
        master_playlist = Some(paths.master_path);
    }

    let completed = rung_outputs.len();
    let failed = spec.rungs.len() - completed;
    sink.on_event(JobEvent::Finished {
        rungs_completed: completed,
        rungs_failed: failed,
    });

    Ok(JobOutput {
        rungs: rung_outputs,
        hls_root,
        master_playlist,
        source_codec,
        source_dims,
        source_frame_rate,
        audio_handling,
        elapsed: started.elapsed(),
    })
}

/// Synchronous wrapper that builds a multi-threaded Tokio runtime. For CLI /
/// non-async callers.
pub fn run_job_blocking(
    input: &[u8],
    spec: &OutputSpec,
    output_dir: Option<&Path>,
    sink: Arc<dyn ProgressSink>,
) -> Result<JobOutput> {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("building Tokio runtime")?;
    rt.block_on(run_job(Bytes::copy_from_slice(input), spec, output_dir, sink))
}

// ---------------------------------------------------------------------------
// Per-rung workers
// ---------------------------------------------------------------------------

type RungResult = Result<(RungOutput, Option<CmafTrackManifest>)>;

#[allow(clippy::too_many_arguments)]
fn encode_rung_single_file(
    rung_index: usize,
    rung: &Rung,
    mut rx: tokio::sync::mpsc::Receiver<VideoFrame>,
    mut cfg: EncoderConfig,
    backend: Option<EncoderBackend>,
    frame_rate: f64,
    frames_total: Option<u64>,
    audio: Option<&PreparedAudio>,
    sink: &dyn ProgressSink,
) -> RungResult {
    cfg.width = rung.width;
    cfg.height = rung.height;
    rung.quality.apply(&mut cfg, frame_rate);

    let mut encoder = encode::select_encoder(cfg, backend)
        .with_context(|| format!("creating encoder for rung {}", rung.label))?;
    let mut muxer = Av1Mp4Muxer::new(rung.width, rung.height, frame_rate)
        .context("Av1Mp4Muxer::new")?;
    muxer.set_color_metadata(ColorMetadata::default());

    // Attach audio up front (interleaved at finalize).
    if let Some(a) = audio {
        if let Err(e) = muxer.with_audio(a.info.clone()) {
            tracing::warn!(rung = %rung.label, "audio rejected ({e}); video-only");
        } else {
            for (sample, dur) in &a.samples {
                muxer
                    .add_audio_sample(sample, 0, *dur)
                    .context("add_audio_sample")?;
            }
        }
    }

    let mut frames: u64 = 0;
    report(sink, rung_index, rung, RungStatus::Running, 0, frames_total, 0, 0);

    while let Some(frame) = rx.blocking_recv() {
        let scaled = colorspace::scale_frame(&frame, rung.width, rung.height)
            .context("scale_frame")?;
        encoder.send_frame(&scaled).context("send_frame")?;
        while let Some(pkt) = encoder.receive_packet().context("receive_packet")? {
            muxer.add_packet(pkt).context("add_packet")?;
        }
        frames += 1;
        if frames % 30 == 0 {
            report(sink, rung_index, rung, RungStatus::Running, frames, frames_total, 0, 0);
        }
    }

    encoder.flush().context("encoder flush")?;
    while let Some(pkt) = encoder.receive_packet().context("receive_packet drain")? {
        muxer.add_packet(pkt).context("add_packet drain")?;
    }

    report(sink, rung_index, rung, RungStatus::Finalizing, frames, frames_total, 0, 0);
    let bytes = muxer.finalize().context("finalize")?.to_vec();
    let nbytes = bytes.len() as u64;
    report(sink, rung_index, rung, RungStatus::Completed, frames, frames_total, 0, nbytes);

    Ok((
        RungOutput {
            label: rung.label.clone(),
            width: rung.width,
            height: rung.height,
            frames,
            bytes: nbytes,
            artifact: RungArtifact::File(bytes),
        },
        None,
    ))
}

#[allow(clippy::too_many_arguments)]
fn encode_rung_hls(
    rung_index: usize,
    rung: &Rung,
    mut rx: tokio::sync::mpsc::Receiver<VideoFrame>,
    mut cfg: EncoderConfig,
    backend: Option<EncoderBackend>,
    frame_rate: f64,
    segment_seconds: f32,
    frames_total: Option<u64>,
    asset_root: &Path,
    sink: &dyn ProgressSink,
) -> RungResult {
    // CMAF timing grid (matches the reference transcoder).
    let timescale: u32 = (frame_rate * 1000.0).round().max(1.0) as u32;
    let per_frame_ticks: u32 = (timescale as f64 / frame_rate.max(1.0)).round().max(1.0) as u32;
    let keyframe_interval = keyframe_interval_for_segment(segment_seconds as f64, frame_rate);
    let segment_target_ticks = (keyframe_interval as u64) * (per_frame_ticks as u64);

    cfg.width = rung.width;
    cfg.height = rung.height;
    rung.quality.apply(&mut cfg, frame_rate);
    // Force the encoder GOP to the segment cadence so segments break on
    // keyframes (CMAF §7.3.2.1 — each segment opens with an IDR).
    cfg.keyframe_interval = keyframe_interval;

    let relative_dir = format!("video/{}", rung.label);
    let rung_dir = asset_root.join(&relative_dir);

    let mut encoder = encode::select_encoder(cfg, backend)
        .with_context(|| format!("creating encoder for rung {}", rung.label))?;
    let mut muxer = CmafVideoMuxer::new(
        &rung_dir,
        rung.width,
        rung.height,
        timescale,
        ColorMetadata::default(),
    )
    .context("CmafVideoMuxer::new")?;

    let mut frames: u64 = 0;
    report(sink, rung_index, rung, RungStatus::Running, 0, frames_total, 0, 0);

    while let Some(frame) = rx.blocking_recv() {
        let scaled = colorspace::scale_frame(&frame, rung.width, rung.height)
            .context("scale_frame")?;
        encoder.send_frame(&scaled).context("send_frame")?;
        while let Some(pkt) = encoder.receive_packet().context("receive_packet")? {
            add_packet_with_segment_flush(&mut muxer, &pkt, per_frame_ticks, segment_target_ticks)?;
        }
        frames += 1;
        if frames % 30 == 0 {
            let segs = muxer.segments().len() as u32;
            report(sink, rung_index, rung, RungStatus::Running, frames, frames_total, segs, 0);
        }
    }

    encoder.flush().context("encoder flush")?;
    while let Some(pkt) = encoder.receive_packet().context("receive_packet drain")? {
        add_packet_with_segment_flush(&mut muxer, &pkt, per_frame_ticks, segment_target_ticks)?;
    }
    // Flush the trailing partial segment.
    muxer.flush_segment().context("final flush_segment")?;

    report(sink, rung_index, rung, RungStatus::Finalizing, frames, frames_total, 0, 0);
    let manifest = muxer.finalize().context("CmafVideoMuxer finalize")?;
    let nbytes = dir_size(&rung_dir);
    let segs = manifest.segments.len() as u32;
    report(sink, rung_index, rung, RungStatus::Completed, frames, frames_total, segs, nbytes);

    Ok((
        RungOutput {
            label: rung.label.clone(),
            width: rung.width,
            height: rung.height,
            frames,
            bytes: nbytes,
            artifact: RungArtifact::HlsRendition {
                dir: rung_dir,
                relative_dir,
            },
        },
        Some(manifest),
    ))
}

// ---------------------------------------------------------------------------
// Audio
// ---------------------------------------------------------------------------

/// Audio prepared once and reused across rungs / for the audio rendition.
#[derive(Clone)]
struct PreparedAudio {
    info: AudioInfo,
    /// (sample bytes, duration ticks).
    samples: Vec<(Vec<u8>, u32)>,
    handling: String,
}

fn prepare_audio(track: Option<&AudioTrack>, policy: AudioPolicy) -> Result<Option<PreparedAudio>> {
    let Some(track) = track else {
        return Ok(None);
    };
    if policy == AudioPolicy::Drop {
        return Ok(None);
    }
    let codec = track.codec.to_ascii_lowercase();

    let passthrough_ok = matches!(codec.as_str(), "aac" | "opus" | "ac3" | "eac3");
    let force_opus = policy == AudioPolicy::ForceOpus;

    if passthrough_ok && !(force_opus && codec != "opus") {
        let info = passthrough_info(&codec, track);
        let samples = track
            .samples
            .iter()
            .cloned()
            .zip(track.durations.iter().copied())
            .collect();
        return Ok(Some(PreparedAudio {
            info,
            samples,
            handling: format!("{codec} passthrough"),
        }));
    }

    // Transcode to Opus (mp3 / vorbis, or anything when ForceOpus).
    if matches!(codec.as_str(), "mp3" | "vorbis") || force_opus {
        if track.channels > 2 {
            tracing::warn!(codec, channels = track.channels, "multichannel audio dropped");
            return Ok(Some(dropped(format!("{codec} ({}ch)", track.channels))));
        }
        if !matches!(codec.as_str(), "mp3" | "vorbis") {
            // ForceOpus on a codec we can't decode → drop.
            tracing::warn!(codec, "cannot transcode to opus; dropping audio");
            return Ok(Some(dropped(codec)));
        }
        let extra: Option<&[u8]> = if track.codec_private.is_empty() {
            None
        } else {
            Some(track.codec_private.as_slice())
        };
        let mut dec = audio_decoder(&codec, extra, track.sample_rate, track.channels as u8)
            .context("audio decoder")?;
        let bitrate = if track.channels == 1 { 64_000 } else { 96_000 };
        let mut enc = audio_encoder(AudioEncoderConfig {
            codec: AudioCodec::Opus,
            sample_rate: track.sample_rate,
            channels: track.channels as u8,
            bitrate,
        })
        .context("opus encoder")?;

        let mut samples: Vec<(Vec<u8>, u32)> = Vec::new();
        let mut pts: i64 = 0;
        for packet in &track.samples {
            for frame in dec.decode(packet, pts).context("audio decode")? {
                pts = pts.saturating_add((frame.samples.len() as i64) / frame.channels.max(1) as i64);
                for pkt in enc.encode(&frame).context("opus encode")? {
                    samples.push((pkt.data, pkt.duration as u32));
                }
            }
        }
        for frame in dec.flush().context("audio flush")? {
            for pkt in enc.encode(&frame).context("opus encode flush")? {
                samples.push((pkt.data, pkt.duration as u32));
            }
        }
        for pkt in enc.flush().context("opus encoder flush")? {
            samples.push((pkt.data, pkt.duration as u32));
        }
        let info = AudioInfo::opus(48_000, track.channels, enc.extra_data());
        return Ok(Some(PreparedAudio {
            info,
            samples,
            handling: format!("{codec} → opus"),
        }));
    }

    // Unsupported codec under Auto policy.
    Ok(Some(dropped(codec)))
}

fn dropped(codec: String) -> PreparedAudio {
    PreparedAudio {
        info: AudioInfo::aac_lc(48_000, 2, Vec::new()),
        samples: Vec::new(),
        handling: format!("{codec} dropped"),
    }
}

fn passthrough_info(codec: &str, track: &AudioTrack) -> AudioInfo {
    match codec {
        "aac" => AudioInfo::aac_lc(track.sample_rate, track.channels, track.asc.clone()),
        "opus" => AudioInfo::opus(track.sample_rate, track.channels, track.codec_private.clone()),
        "ac3" => AudioInfo::ac3(track.sample_rate, track.channels, track.codec_private.clone()),
        "eac3" => AudioInfo::eac3(track.sample_rate, track.channels, track.codec_private.clone()),
        _ => AudioInfo::aac_lc(track.sample_rate, track.channels, track.asc.clone()),
    }
}

impl PreparedAudio {
    fn has_samples(&self) -> bool {
        !self.samples.is_empty()
    }
}

/// Build the single shared CMAF audio rendition for the HLS package. Returns
/// `None` when there are no audio samples (video-only).
fn build_audio_rendition(
    asset_root: &Path,
    audio: &PreparedAudio,
    segment_seconds: f32,
) -> Result<Option<AudioVariantSpec>> {
    if !audio.has_samples() {
        return Ok(None);
    }
    let audio_dir = asset_root.join("audio");
    let seg_target_ticks = (segment_seconds as f64 * audio.info.timescale as f64).round() as u64;
    let mut muxer = CmafAudioMuxer::new(&audio_dir, audio.info.clone())
        .context("CmafAudioMuxer::new")?;
    for (payload, dur) in &audio.samples {
        add_audio_sample_with_segment_flush(&mut muxer, payload.clone(), *dur, seg_target_ticks)?;
    }
    muxer.flush_segment().context("final audio flush_segment")?;
    let manifest = muxer.finalize().context("CmafAudioMuxer finalize")?;

    let codec_string = match audio.info.codec.as_str() {
        "opus" => "opus".to_string(),
        _ => codec_strings::AAC_LC_CODEC_STRING.to_string(),
    };
    Ok(Some(AudioVariantSpec {
        codec_string,
        channels: audio.info.channels,
        sample_rate: audio.info.sample_rate,
        relative_dir: "audio".to_string(),
        language: "und".to_string(),
        name: "Audio".to_string(),
        manifest,
    }))
}

// ---------------------------------------------------------------------------
// HLS helpers (ported from the reference transcoder's cmaf module)
// ---------------------------------------------------------------------------

fn keyframe_interval_for_segment(segment_duration_seconds: f64, frame_rate: f64) -> u32 {
    ((segment_duration_seconds * frame_rate).round() as u32).max(1)
}

fn add_packet_with_segment_flush(
    muxer: &mut CmafVideoMuxer,
    packet: &EncodedPacket,
    duration_ticks: u32,
    segment_target_ticks: u64,
) -> Result<()> {
    if packet.is_keyframe
        && muxer.pending_duration_ticks() >= segment_target_ticks
        && muxer.first_pending_is_keyframe()
    {
        muxer.flush_segment().context("flush CMAF video segment")?;
    }
    muxer.add_packet(packet.data.to_vec(), duration_ticks, packet.is_keyframe)?;
    Ok(())
}

fn add_audio_sample_with_segment_flush(
    muxer: &mut CmafAudioMuxer,
    payload: Vec<u8>,
    duration_ticks: u32,
    segment_target_ticks: u64,
) -> Result<()> {
    if muxer.pending_duration_ticks() >= segment_target_ticks {
        muxer.flush_segment().context("flush CMAF audio segment")?;
    }
    muxer.add_packet(payload, duration_ticks)?;
    Ok(())
}

fn build_video_variant_spec(
    rung: &Rung,
    out: &RungOutput,
    frame_rate: f64,
    manifest: CmafTrackManifest,
) -> Result<VideoVariantSpec> {
    let relative_dir = match &out.artifact {
        RungArtifact::HlsRendition { relative_dir, .. } => relative_dir.clone(),
        _ => format!("video/{}", rung.label),
    };
    let codec_string = av1_codec_string_from_init(&manifest.init_path)
        .unwrap_or_else(|_| "av01.0.08M.08.0.110.01.01.01.0".to_string());
    let duration = manifest.duration_seconds().max(0.001);
    let bandwidth = ((out.bytes as f64 * 8.0) / duration).round() as u32;
    Ok(VideoVariantSpec {
        width: rung.width,
        height: rung.height,
        frame_rate,
        average_bandwidth_bps: bandwidth,
        bandwidth_bps: bandwidth,
        codec_string,
        supplemental_codecs: None,
        video_range: None,
        relative_dir,
        manifest,
    })
}

/// Parse the AV1 codec string from a rendition's init segment.
fn av1_codec_string_from_init(init_path: &Path) -> Result<String> {
    let bytes = std::fs::read(init_path)
        .with_context(|| format!("reading init segment {}", init_path.display()))?;
    let obus = find_av1c_config_obus(&bytes)
        .ok_or_else(|| anyhow!("av1C box not found in init segment"))?;
    let seq = parse_av1_sequence_header(obus)
        .ok_or_else(|| anyhow!("could not parse AV1 sequence header from av1C"))?;
    Ok(av1_codec_string(&seq))
}

fn find_av1c_config_obus(buf: &[u8]) -> Option<&[u8]> {
    let moov = find_box(buf, b"moov")?;
    let trak = find_child_box(moov, b"trak")?;
    let mdia = find_child_box(trak, b"mdia")?;
    let minf = find_child_box(mdia, b"minf")?;
    let stbl = find_child_box(minf, b"stbl")?;
    let stsd = find_child_box(stbl, b"stsd")?;
    if stsd.len() < 16 {
        return None;
    }
    let after_header_and_count = &stsd[8 + 8..];
    let av01 = find_box(after_header_and_count, b"av01")?;
    if av01.len() < 8 + 78 {
        return None;
    }
    let av01_children = &av01[8 + 78..];
    let av1c = find_box(av01_children, b"av1C")?;
    if av1c.len() < 8 + 4 {
        return None;
    }
    Some(&av1c[8 + 4..])
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

// ---------------------------------------------------------------------------
// Misc helpers
// ---------------------------------------------------------------------------

fn encoder_backend_override() -> Option<EncoderBackend> {
    std::env::var("TRANSCODE_ENCODER_BACKEND")
        .ok()
        .and_then(|s| match s.to_ascii_lowercase().as_str() {
            "nvenc" => Some(EncoderBackend::Nvenc),
            "amf" => Some(EncoderBackend::Amf),
            "qsv" => Some(EncoderBackend::Qsv),
            _ => None,
        })
}

fn dir_size(dir: &Path) -> u64 {
    let mut total = 0;
    if let Ok(entries) = std::fs::read_dir(dir) {
        for e in entries.flatten() {
            if let Ok(meta) = e.metadata() {
                if meta.is_file() {
                    total += meta.len();
                }
            }
        }
    }
    total
}

#[allow(clippy::too_many_arguments)]
fn report(
    sink: &dyn ProgressSink,
    rung_index: usize,
    rung: &Rung,
    status: RungStatus,
    frames_done: u64,
    frames_total: Option<u64>,
    segments: u32,
    bytes_out: u64,
) {
    let percent = match status {
        RungStatus::Completed => 100.0,
        RungStatus::Pending => 0.0,
        _ => match frames_total {
            Some(total) if total > 0 => ((frames_done as f32 / total as f32) * 100.0).min(99.0),
            _ => {
                if frames_done == 0 {
                    1.0
                } else {
                    50.0
                }
            }
        },
    };
    sink.on_rung(RungProgress {
        rung_index,
        label: rung.label.clone(),
        width: rung.width,
        height: rung.height,
        status,
        percent,
        frames_done,
        frames_total,
        segments_written: segments,
        bytes_out,
        message: None,
    });
}

fn report_failed(sink: &dyn ProgressSink, rung_index: usize, rung: &Rung, message: &str) {
    sink.on_rung(RungProgress {
        rung_index,
        label: rung.label.clone(),
        width: rung.width,
        height: rung.height,
        status: RungStatus::Failed,
        percent: 0.0,
        frames_done: 0,
        frames_total: None,
        segments_written: 0,
        bytes_out: 0,
        message: Some(message.to_string()),
    });
}
