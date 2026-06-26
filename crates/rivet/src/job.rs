//! The transcode job engine.
//!
//! [`run_job`] takes an input buffer and an [`OutputSpec`] and drives the
//! whole pipeline: demux → shared decode pump (decode once) → fan out to per-
//! rung work → assemble the requested output mode. Progress is streamed
//! through a [`ProgressSink`] as a uniform [`RungProgress`] per rung.
//!
//! - **SingleFile** mode: the decode pump fans frames to one per-rung worker
//!   that scales + encodes + muxes a self-contained MP4.
//! - **Hls** mode: the [`crate::multigpu`] orchestrator decodes once and
//!   schedules every rung's CMAF segments across all GPUs (fair lease pool +
//!   mid-flight helper dispatch + cross-vendor codec invariant), then this
//!   module assembles the HLS package (audio rendition + playlists).

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use bytes::Bytes;

use codec::audio::{
    AudioCodec, AudioEncoderConfig, create_decoder as audio_decoder,
    create_encoder as audio_encoder,
};
use codec::encode::{self, EncoderBackend, EncoderConfig};
use codec::frame::{ColorMetadata, VideoFrame};
use codec::colorspace;
use container::cmaf::CmafAudioMuxer;
use container::demux::AudioTrack;
use container::hls::{AudioVariantSpec, VideoVariantSpec, write_hls_package};
use container::mux::Av1Mp4Muxer;
use container::streaming::{self, DemuxHeader};
use container::AudioInfo;

use crate::cmaf_util::{self, add_audio_sample_with_segment_flush, keyframe_interval_for_segment};
use crate::decode_pump::{DecodePumpConfig, run_shared_decode_pump_blocking};
use crate::multigpu::{self, MultiGpuParams, RungManifest, RungPackets};
use crate::progress::{JobEvent, ProgressSink, RungProgress, RungStatus};
use crate::spec::{AudioPolicy, EncodePolicy, OutputMode, OutputSpec, Rung};
use crate::validate::needs_chroma_downsample;

/// Bounded per-rung frame channel — backpressures the decode pump.
const FRAME_CHANNEL_CAPACITY: usize = 8;

/// The artifact one rung produced.
#[derive(Debug)]
pub enum RungArtifact {
    /// A single self-contained file (MP4 bytes).
    File(Vec<u8>),
    /// An HLS rendition: a directory of CMAF segments + a media playlist.
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
    /// HLS mode only: the asset root directory.
    pub hls_root: Option<PathBuf>,
    /// HLS mode only: path to the master playlist.
    pub master_playlist: Option<PathBuf>,
    pub source_codec: String,
    pub source_dims: (u32, u32),
    pub source_frame_rate: f64,
    /// How the audio was handled.
    pub audio_handling: String,
    pub elapsed: Duration,
}

/// Run a transcode job. Async — call from within a Tokio runtime.
///
/// For [`OutputMode::Hls`], `output_dir` is the asset root the HLS package is
/// written under; `None` uses a fresh temp directory (returned in
/// [`JobOutput::hls_root`]). For [`OutputMode::SingleFile`] `output_dir` is
/// ignored (bytes are returned).
pub async fn run_job(
    input: Bytes,
    spec: &OutputSpec,
    output_dir: Option<&Path>,
    sink: Arc<dyn ProgressSink>,
) -> Result<JobOutput> {
    let started = Instant::now();
    spec.validate().context("invalid OutputSpec")?;

    let (header, audio_track) = {
        let demuxer = streaming::demux_streaming(&input).context("demux")?;
        (demuxer.header().clone(), demuxer.audio().cloned())
    };
    let source_codec = header.codec.to_ascii_lowercase();
    let source_dims = (header.info.width, header.info.height);
    let source_frame_rate = header.info.frame_rate;

    sink.on_event(JobEvent::Started { rungs: spec.rungs.len() });
    sink.on_event(JobEvent::Probed {
        codec: source_codec.clone(),
        width: header.info.width,
        height: header.info.height,
        frame_rate: header.info.frame_rate,
        audio_codec: audio_track.as_ref().map(|t| t.codec.to_ascii_lowercase()),
    });

    let frame_rate = {
        let mut fr = if header.info.frame_rate > 0.0 { header.info.frame_rate } else { 30.0 };
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

    let prepared_audio = prepare_audio(audio_track.as_ref(), spec.audio).context("preparing audio")?;
    let audio_handling = prepared_audio
        .as_ref()
        .map(|a| a.handling.clone())
        .unwrap_or_else(|| "none".to_string());

    let (rungs, hls_root, master_playlist) = match &spec.mode {
        OutputMode::SingleFile => {
            let rungs = run_single_file(
                input.clone(),
                spec,
                &header,
                frame_rate,
                frames_total,
                prepared_audio.as_ref(),
                Arc::clone(&sink),
            )
            .await?;
            (rungs, None, None)
        }
        OutputMode::Hls { segment_seconds } => {
            run_hls(
                input.clone(),
                spec,
                *segment_seconds,
                &header,
                frame_rate,
                prepared_audio.as_ref(),
                output_dir,
                Arc::clone(&sink),
            )
            .await?
        }
    };

    let completed = rungs.len();
    sink.on_event(JobEvent::Finished {
        rungs_completed: completed,
        rungs_failed: spec.rungs.len().saturating_sub(completed),
    });

    Ok(JobOutput {
        rungs,
        hls_root,
        master_playlist,
        source_codec,
        source_dims,
        source_frame_rate,
        audio_handling,
        elapsed: started.elapsed(),
    })
}

/// Synchronous wrapper that builds a multi-threaded Tokio runtime.
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
// SingleFile: decode-once fan-out to per-rung MP4 workers
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
async fn run_single_file(
    input: Bytes,
    spec: &OutputSpec,
    header: &DemuxHeader,
    frame_rate: f64,
    frames_total: Option<u64>,
    audio: Option<&PreparedAudio>,
    sink: Arc<dyn ProgressSink>,
) -> Result<Vec<RungOutput>> {
    // When the frame count is known and the host has more than one GPU, run the
    // multi-GPU engine for single-file too: decode once, chunk each rung at
    // GOP boundaries, encode the chunks across all GPUs (fair lease pool +
    // helper dispatch + cross-vendor codec invariant), then stitch the packets,
    // in segment order, into one MP4 per rung. On a single-GPU host (or unknown
    // frame count) the serial path below is used unchanged — no chunk overhead.
    let total_input_frames = if header.info.total_frames > 0 {
        header.info.total_frames
    } else {
        (header.info.duration * frame_rate).round().max(0.0) as u64
    };
    let gpu_pool = multigpu::gpu_pool_for_policy(spec.encode_policy);
    if matches!(
        spec.encode_policy,
        EncodePolicy::AllGpus | EncodePolicy::Family(_)
    ) && total_input_frames > 0
        && gpu_pool.capacity() > 1
        // `ChunkSeamMode::Serial` forces one encoder (seam-free) even on a
        // multi-GPU host — skip the chunk-and-stitch path entirely.
        && spec.chunk_seam_mode != crate::spec::ChunkSeamMode::Serial
    {
        return run_single_file_multigpu(
            input,
            spec,
            header,
            frame_rate,
            total_input_frames,
            audio,
            gpu_pool,
            sink,
        )
        .await;
    }

    // Serial path: encode on the policy's GPU (the vendor's first device for
    // Family, the pinned index for SingleGpu, auto for AllGpus); decode follows
    // the explicit decode_gpu override, else the same GPU as encode.
    let encode_gpu = multigpu::serial_gpu_for_policy(spec.encode_policy);
    let decode_gpu = spec.decode_gpu.or(encode_gpu);
    let (output_color_metadata, output_pixel_format) =
        spec.resolve_output(header.info.color_metadata, header.info.pixel_format);
    let backend_override = encoder_backend_override();
    let base_cfg = EncoderConfig {
        frame_rate,
        pixel_format: output_pixel_format,
        color_metadata: output_color_metadata,
        gpu_index: encode_gpu,
        ..EncoderConfig::default()
    };
    let pump_cfg = DecodePumpConfig {
        codec_name: header.codec.clone(),
        info_for_decoder: header.info.clone(),
        source_color_metadata: header.info.color_metadata,
        source_pixel_format: header.info.pixel_format,
        needs_downsample: needs_chroma_downsample(header.info.pixel_format),
        tonemap_to_sdr: spec.tonemaps(),
        gpu_index: decode_gpu,
    };
    let rt = tokio::runtime::Handle::current();

    let mut senders = Vec::with_capacity(spec.rungs.len());
    let mut handles = Vec::with_capacity(spec.rungs.len());
    for (idx, rung) in spec.rungs.iter().cloned().enumerate() {
        let (tx, rx) = tokio::sync::mpsc::channel::<VideoFrame>(FRAME_CHANNEL_CAPACITY);
        senders.push(tx);
        let sink = Arc::clone(&sink);
        let base_cfg = base_cfg.clone();
        let audio = audio.cloned();
        let handle = tokio::task::spawn_blocking(move || {
            let r = encode_rung_single_file(
                idx, &rung, rx, base_cfg, backend_override, frame_rate, frames_total,
                audio.as_ref(), sink.as_ref(),
            );
            (idx, rung, r)
        });
        handles.push(handle);
    }

    let pump_handle = {
        let input = input.clone();
        let rt = rt.clone();
        tokio::task::spawn_blocking(move || {
            run_shared_decode_pump_blocking(pump_cfg, input, senders, rt)
        })
    };

    let mut outputs = Vec::new();
    for handle in handles {
        let (idx, rung, r) = handle.await.context("rung worker task panicked")?;
        match r {
            Ok(out) => outputs.push(out),
            Err(e) => {
                tracing::warn!(rung = %rung.label, error = %e, "rung failed");
                report_failed(sink.as_ref(), idx, &rung, &e.to_string());
            }
        }
    }
    let _ = pump_handle.await.context("decode pump panicked")?.context("decode pump failed")?;
    if outputs.is_empty() {
        bail!("all {} rung(s) failed", spec.rungs.len());
    }
    Ok(outputs)
}

/// Single-file via the multi-GPU engine: chunk each rung across GPUs, then
/// stitch the packets into one MP4 per rung (no disk round-trip — packets stay
/// in memory). Chunk length is a 2 s GOP so each chunk is an independently
/// decodable IDR sequence; the cross-vendor codec invariant keeps every chunk's
/// `av1C` contract identical so cross-GPU/-vendor stitching is bit-safe.
#[allow(clippy::too_many_arguments)]
async fn run_single_file_multigpu(
    input: Bytes,
    spec: &OutputSpec,
    header: &DemuxHeader,
    frame_rate: f64,
    total_input_frames: u64,
    audio: Option<&PreparedAudio>,
    gpu_pool: Arc<crate::gpu_pool::GpuPool>,
    sink: Arc<dyn ProgressSink>,
) -> Result<Vec<RungOutput>> {
    const CHUNK_SECONDS: f64 = 2.0;
    let timescale = (frame_rate * 1000.0).round().max(1.0) as u32;
    let per_frame_ticks = (timescale as f64 / frame_rate.max(1.0)).round().max(1.0) as u32;
    let keyframe_interval = keyframe_interval_for_segment(CHUNK_SECONDS, frame_rate);
    let segment_target_ticks = (keyframe_interval as u64) * (per_frame_ticks as u64);

    let (output_color_metadata, output_pixel_format) =
        spec.resolve_output(header.info.color_metadata, header.info.pixel_format);
    let params = MultiGpuParams {
        input,
        rungs: &spec.rungs,
        header: header.clone(),
        source_color_metadata: header.info.color_metadata,
        source_pixel_format: header.info.pixel_format,
        tonemap_to_sdr: spec.tonemaps(),
        output_color_metadata,
        output_pixel_format,
        needs_downsample: needs_chroma_downsample(header.info.pixel_format),
        frame_rate,
        gpu_pool,
        gpu_indices: multigpu::policy_gpu_indices(spec.encode_policy),
        decode_gpu: spec.decode_gpu,
        // Chunk workers collect packets in memory; output_root is unused.
        output_root: std::env::temp_dir(),
        timescale,
        per_frame_ticks,
        keyframe_interval,
        segment_target_ticks,
        total_input_frames,
        // ParallelConstQp ⇒ force constant-QP chunks so stitched seams are flat.
        constant_qp: spec.chunk_seam_mode == crate::spec::ChunkSeamMode::ParallelConstQp,
    };
    let rung_packets = multigpu::run_multigpu_single_file(params, Arc::clone(&sink)).await?;

    let mut outputs = Vec::new();
    for rp in rung_packets.into_iter().flatten() {
        let label = rp.label.clone();
        match mux_rung_packets_to_mp4(rp, frame_rate, output_color_metadata, audio) {
            Ok(out) => outputs.push(out),
            Err(e) => tracing::warn!(rung = %label, error = %e, "stitching rung MP4 failed"),
        }
    }
    if outputs.is_empty() {
        bail!("multi-GPU single-file: no rung produced a stitched MP4");
    }
    Ok(outputs)
}

/// Stitch one rung's ordered AV1 packets (+ optional audio) into an MP4.
fn mux_rung_packets_to_mp4(
    rp: RungPackets,
    frame_rate: f64,
    color_metadata: ColorMetadata,
    audio: Option<&PreparedAudio>,
) -> Result<RungOutput> {
    let mut muxer =
        Av1Mp4Muxer::new(rp.width, rp.height, frame_rate).context("Av1Mp4Muxer::new")?;
    muxer.set_color_metadata(color_metadata);
    if let Some(a) = audio {
        if let Err(e) = muxer.with_audio(a.info.clone()) {
            tracing::warn!(rung = %rp.label, "audio rejected ({e}); video-only");
        } else {
            for (sample, dur) in &a.samples {
                muxer.add_audio_sample(sample, 0, *dur).context("add_audio_sample")?;
            }
        }
    }
    let frames = rp.packets.len() as u64;
    for pkt in rp.packets {
        muxer.add_packet(pkt).context("add_packet")?;
    }
    let bytes = muxer.finalize().context("finalize")?.to_vec();
    let nbytes = bytes.len() as u64;
    Ok(RungOutput {
        label: rp.label,
        width: rp.width,
        height: rp.height,
        frames,
        bytes: nbytes,
        artifact: RungArtifact::File(bytes),
    })
}

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
) -> Result<RungOutput> {
    cfg.width = rung.width;
    cfg.height = rung.height;
    rung.quality.apply(&mut cfg, frame_rate);

    let out_color = cfg.color_metadata;
    let mut encoder = encode::select_encoder(cfg, backend)
        .with_context(|| format!("creating encoder for rung {}", rung.label))?;
    let mut muxer = Av1Mp4Muxer::new(rung.width, rung.height, frame_rate).context("Av1Mp4Muxer::new")?;
    muxer.set_color_metadata(out_color);

    if let Some(a) = audio {
        if let Err(e) = muxer.with_audio(a.info.clone()) {
            tracing::warn!(rung = %rung.label, "audio rejected ({e}); video-only");
        } else {
            for (sample, dur) in &a.samples {
                muxer.add_audio_sample(sample, 0, *dur).context("add_audio_sample")?;
            }
        }
    }

    let mut frames: u64 = 0;
    report(sink, rung_index, rung, RungStatus::Running, 0, frames_total, 0, 0);
    while let Some(frame) = rx.blocking_recv() {
        let scaled = colorspace::scale_frame(&frame, rung.width, rung.height).context("scale_frame")?;
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

    Ok(RungOutput {
        label: rung.label.clone(),
        width: rung.width,
        height: rung.height,
        frames,
        bytes: nbytes,
        artifact: RungArtifact::File(bytes),
    })
}

// ---------------------------------------------------------------------------
// Hls: multi-GPU orchestrator + package assembly
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
async fn run_hls(
    input: Bytes,
    spec: &OutputSpec,
    segment_seconds: f32,
    header: &DemuxHeader,
    frame_rate: f64,
    audio: Option<&PreparedAudio>,
    output_dir: Option<&Path>,
    sink: Arc<dyn ProgressSink>,
) -> Result<(Vec<RungOutput>, Option<PathBuf>, Option<PathBuf>)> {
    let root = match output_dir {
        Some(d) => d.to_path_buf(),
        None => tempfile::Builder::new()
            .prefix("rivet-hls-")
            .tempdir()
            .context("creating HLS temp dir")?
            .keep(),
    };

    let timescale = (frame_rate * 1000.0).round().max(1.0) as u32;
    let per_frame_ticks = (timescale as f64 / frame_rate.max(1.0)).round().max(1.0) as u32;
    let keyframe_interval = keyframe_interval_for_segment(segment_seconds as f64, frame_rate);
    let segment_target_ticks = (keyframe_interval as u64) * (per_frame_ticks as u64);
    let total_input_frames = if header.info.total_frames > 0 {
        header.info.total_frames
    } else {
        (header.info.duration * frame_rate).round().max(0.0) as u64
    };

    let gpu_pool = multigpu::gpu_pool_for_policy(spec.encode_policy);
    let (output_color_metadata, output_pixel_format) =
        spec.resolve_output(header.info.color_metadata, header.info.pixel_format);
    let params = MultiGpuParams {
        input,
        rungs: &spec.rungs,
        header: header.clone(),
        source_color_metadata: header.info.color_metadata,
        source_pixel_format: header.info.pixel_format,
        tonemap_to_sdr: spec.tonemaps(),
        output_color_metadata,
        output_pixel_format,
        needs_downsample: needs_chroma_downsample(header.info.pixel_format),
        frame_rate,
        gpu_pool,
        gpu_indices: multigpu::policy_gpu_indices(spec.encode_policy),
        decode_gpu: spec.decode_gpu,
        output_root: root.clone(),
        timescale,
        per_frame_ticks,
        keyframe_interval,
        segment_target_ticks,
        total_input_frames,
        // HLS segments are independent files — no stitched seams to flatten.
        constant_qp: false,
    };
    let manifests = multigpu::run_multigpu_hls(params, Arc::clone(&sink)).await?;

    let mut rung_outputs = Vec::new();
    let mut video_specs = Vec::new();
    for (idx, m) in manifests.into_iter().enumerate() {
        match m {
            Some(rm) => {
                let dir = root.join(&rm.relative_dir);
                let bytes = dir_size(&dir);
                video_specs.push(build_video_variant_spec(&rm, frame_rate, bytes));
                rung_outputs.push(RungOutput {
                    label: rm.label.clone(),
                    width: rm.width,
                    height: rm.height,
                    frames: total_input_frames,
                    bytes,
                    artifact: RungArtifact::HlsRendition {
                        dir,
                        relative_dir: rm.relative_dir,
                    },
                });
            }
            None => {
                if let Some(rung) = spec.rungs.get(idx) {
                    report_failed(sink.as_ref(), idx, rung, "rung produced no segments");
                }
            }
        }
    }
    if rung_outputs.is_empty() {
        bail!("all {} rung(s) failed", spec.rungs.len());
    }

    let audio_spec = match audio {
        Some(a) => build_audio_rendition(&root, a, segment_seconds).context("building HLS audio rendition")?,
        None => None,
    };
    let target_duration = segment_seconds.ceil() as u32;
    let paths = write_hls_package(&root, &video_specs, audio_spec.as_ref(), target_duration)
        .context("writing HLS package")?;

    Ok((rung_outputs, Some(root), Some(paths.master_path)))
}

fn build_video_variant_spec(rm: &RungManifest, frame_rate: f64, bytes: u64) -> VideoVariantSpec {
    let codec_string = cmaf_util::av1_codec_string_from_init(&rm.manifest.init_path)
        .unwrap_or_else(|_| "av01.0.08M.08.0.110.01.01.01.0".to_string());
    let (_avg, peak) = cmaf_util::measure_bandwidth(&rm.manifest);
    let bandwidth = if peak > 0 {
        peak
    } else {
        let dur = rm.manifest.duration_seconds().max(0.001);
        ((bytes as f64 * 8.0) / dur) as u32
    };
    VideoVariantSpec {
        width: rm.width,
        height: rm.height,
        frame_rate,
        average_bandwidth_bps: bandwidth,
        bandwidth_bps: bandwidth,
        codec_string,
        supplemental_codecs: None,
        video_range: None,
        relative_dir: rm.relative_dir.clone(),
        manifest: rm.manifest.clone(),
    }
}

// ---------------------------------------------------------------------------
// Audio
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct PreparedAudio {
    info: AudioInfo,
    samples: Vec<(Vec<u8>, u32)>,
    handling: String,
}

impl PreparedAudio {
    fn has_samples(&self) -> bool {
        !self.samples.is_empty()
    }
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

    if matches!(codec.as_str(), "mp3" | "vorbis") || force_opus {
        if track.channels > 2 {
            tracing::warn!(codec, channels = track.channels, "multichannel audio dropped");
            return Ok(Some(dropped(format!("{codec} ({}ch)", track.channels))));
        }
        if !matches!(codec.as_str(), "mp3" | "vorbis") {
            tracing::warn!(codec, "cannot transcode to opus; dropping audio");
            return Ok(Some(dropped(codec)));
        }
        let extra: Option<&[u8]> =
            if track.codec_private.is_empty() { None } else { Some(track.codec_private.as_slice()) };
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
    let mut muxer = CmafAudioMuxer::new(&audio_dir, audio.info.clone()).context("CmafAudioMuxer::new")?;
    for (payload, dur) in &audio.samples {
        add_audio_sample_with_segment_flush(&mut muxer, payload.clone(), *dur, seg_target_ticks)?;
    }
    muxer.flush_segment().context("final audio flush_segment")?;
    let manifest = muxer.finalize().context("CmafAudioMuxer finalize")?;

    let codec_string = match audio.info.codec.as_str() {
        "opus" => "opus".to_string(),
        _ => codec::codec_strings::AAC_LC_CODEC_STRING.to_string(),
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
                if frames_done == 0 { 1.0 } else { 50.0 }
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
