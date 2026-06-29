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
use crate::decode_pump::DecodePumpConfig;
use crate::multigpu::{self, MultiGpuParams, RungManifest, RungPackets};
use crate::progress::{JobEvent, ProgressSink, RungProgress, RungStatus};
use crate::spec::{AudioCodecPolicy, EncodePolicy, OutputMode, OutputSpec, Rung};
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

    // Prepare the video filter chain once (loads any overlay images), then share
    // the Arc with every decode pump / multi-GPU param built below.
    let filter_chain = Arc::new(
        codec::filter::FilterChain::prepare(&spec.filters).context("preparing video filters")?,
    );

    let (rungs, hls_root, master_playlist) = match &spec.mode {
        OutputMode::SingleFile => {
            let rungs = run_single_file(
                input.clone(),
                spec,
                &header,
                frame_rate,
                frames_total,
                prepared_audio.as_ref(),
                Arc::clone(&filter_chain),
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
                Arc::clone(&filter_chain),
                output_dir,
                Arc::clone(&sink),
                // Single input: run_hls builds the (optionally trimmed) plan
                // from spec.trim itself.
                Vec::new(),
                None,
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

/// One clip of a [splice](run_splice_job): an input plus an optional
/// `[start, end)` trim window in seconds (either bound `None` = open).
#[derive(Clone)]
pub struct Clip {
    pub input: Bytes,
    pub start: Option<f64>,
    pub end: Option<f64>,
}

impl Clip {
    /// A whole clip, no trim.
    pub fn new(input: impl Into<Bytes>) -> Self {
        Self { input: input.into(), start: None, end: None }
    }

    /// A clip trimmed to `[start, end)` seconds (either bound `None` = open).
    pub fn trimmed(input: impl Into<Bytes>, start: Option<f64>, end: Option<f64>) -> Self {
        Self { input: input.into(), start, end }
    }
}

/// **Splice**: concatenate (and per-clip trim) one or more inputs into a single
/// continuous, re-encoded MP4 per rung. Each clip is decoded with its own
/// decoder, trimmed to its `[start, end)`, and the kept frames are fed to the
/// shared encoder back-to-back. Because the muxer numbers output frames by
/// count, the join is gap-free and the timeline is zero-based — no PTS
/// rewriting. Audio is trimmed per clip and concatenated to match.
///
/// Output config (frame rate, color) follows the **first** clip; inputs are
/// re-encoded to the spec's uniform output, so they may differ in codec /
/// resolution / color. A one-clip `Vec` is a plain (optionally trimmed)
/// transcode. Honors the spec's [`OutputMode`]: `SingleFile` writes one MP4 per
/// rung; `Hls` writes a CMAF/HLS package (the spliced frame stream feeds the
/// multi-GPU HLS engine, so segments are keyframe-aligned across the join).
pub async fn run_splice_job(
    clips: Vec<Clip>,
    spec: &OutputSpec,
    output_dir: Option<&Path>,
    sink: Arc<dyn ProgressSink>,
) -> Result<JobOutput> {
    let started = Instant::now();
    spec.validate().context("invalid OutputSpec")?;
    if clips.is_empty() {
        bail!("splice requires at least one clip");
    }

    // Probe each clip + prepare its audio. The first clip drives output config.
    struct ClipPrep {
        header: DemuxHeader,
        audio: Option<PreparedAudio>,
        src_audio_codec: Option<String>,
    }
    let mut preps = Vec::with_capacity(clips.len());
    for (i, clip) in clips.iter().enumerate() {
        let demuxer = streaming::demux_streaming(&clip.input)
            .with_context(|| format!("demuxing splice clip {i}"))?;
        let header = demuxer.header().clone();
        let src_audio_codec = demuxer.audio().map(|t| t.codec.to_ascii_lowercase());
        let audio = prepare_audio(demuxer.audio(), spec.audio)
            .with_context(|| format!("preparing audio for splice clip {i}"))?;
        preps.push(ClipPrep { header, audio, src_audio_codec });
    }

    let primary = preps[0].header.clone();
    let source_codec = primary.codec.to_ascii_lowercase();
    let source_dims = (primary.info.width, primary.info.height);
    let source_frame_rate = primary.info.frame_rate;
    let frame_rate = {
        let mut fr = if primary.info.frame_rate > 0.0 { primary.info.frame_rate } else { 30.0 };
        if let Some(cap) = spec.max_frame_rate {
            fr = fr.min(cap);
        }
        fr
    };

    sink.on_event(JobEvent::Started { rungs: spec.rungs.len() });
    sink.on_event(JobEvent::Probed {
        codec: source_codec.clone(),
        width: primary.info.width,
        height: primary.info.height,
        frame_rate: primary.info.frame_rate,
        audio_codec: preps[0].src_audio_codec.clone(),
    });

    let filter_chain = Arc::new(
        codec::filter::FilterChain::prepare(&spec.filters).context("preparing video filters")?,
    );
    let encode_gpu = multigpu::serial_gpu_for_policy(spec.encode_policy);
    let decode_gpu = spec.decode_gpu.or(encode_gpu);
    let (output_color_metadata, output_pixel_format) =
        spec.resolve_output(primary.info.color_metadata, primary.info.pixel_format);
    let base_cfg = EncoderConfig {
        frame_rate,
        pixel_format: output_pixel_format,
        color_metadata: output_color_metadata,
        gpu_index: encode_gpu,
        codec: spec.video_codec.codec(),
        ..EncoderConfig::default()
    };

    // One decode source per clip (own decoder cfg + trim range); concatenate the
    // trimmed audio and sum the expected frame total across clips.
    let mut clip_sources = Vec::with_capacity(clips.len());
    let mut combined_audio: Option<PreparedAudio> = None;
    let mut effective_total: u64 = 0;
    let mut total_known = true;
    for (clip, prep) in clips.iter().zip(preps.iter()) {
        let cfps = if prep.header.info.frame_rate > 0.0 {
            prep.header.info.frame_rate
        } else {
            frame_rate
        };
        let start_frame = trim_frame(clip.start, cfps).unwrap_or(0);
        let end_frame = trim_frame(clip.end, cfps);
        match end_frame {
            Some(e) => effective_total += e.saturating_sub(start_frame),
            None if prep.header.info.total_frames > 0 => {
                effective_total += prep.header.info.total_frames.saturating_sub(start_frame)
            }
            None => total_known = false,
        }
        if let Some(a) = trim_audio(prep.audio.as_ref(), clip.start, clip.end) {
            if let Some(c) = combined_audio.as_mut() {
                c.extend(&a);
            } else {
                combined_audio = Some(a);
            }
        }
        let pump_cfg = DecodePumpConfig {
            codec_name: prep.header.codec.clone(),
            info_for_decoder: prep.header.info.clone(),
            source_color_metadata: prep.header.info.color_metadata,
            source_pixel_format: prep.header.info.pixel_format,
            needs_downsample: needs_chroma_downsample(prep.header.info.pixel_format),
            tonemap_to_sdr: spec.tonemaps(),
            gpu_index: decode_gpu,
            filters: Arc::clone(&filter_chain),
        };
        clip_sources.push(crate::decode_pump::ClipSource {
            cfg: pump_cfg,
            input: clip.input.clone(),
            start_frame,
            end_frame,
        });
    }
    let effective_total = total_known.then_some(effective_total);
    let audio_handling = combined_audio
        .as_ref()
        .map(|a| a.handling.clone())
        .unwrap_or_else(|| "none".to_string());

    let (rungs, hls_root, master_playlist) = match &spec.mode {
        OutputMode::SingleFile => {
            let rungs = run_serial_single_file(
                clip_sources,
                spec,
                base_cfg,
                frame_rate,
                effective_total,
                combined_audio,
                Arc::clone(&sink),
            )
            .await?;
            (rungs, None, None)
        }
        OutputMode::Hls { segment_seconds } => {
            // Concat through the multi-GPU HLS engine: the spliced pump feeds the
            // joined frame stream, segments form at keyframe boundaries on the
            // output timeline, so the join is segment-aligned like any ladder.
            run_hls(
                clips[0].input.clone(),
                spec,
                *segment_seconds,
                &primary,
                frame_rate,
                combined_audio.as_ref(),
                Arc::clone(&filter_chain),
                output_dir,
                Arc::clone(&sink),
                clip_sources,
                effective_total,
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

/// Blocking wrapper for [`run_splice_job`].
pub fn run_splice_job_blocking(
    clips: Vec<Clip>,
    spec: &OutputSpec,
    output_dir: Option<&Path>,
    sink: Arc<dyn ProgressSink>,
) -> Result<JobOutput> {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("building Tokio runtime")?;
    rt.block_on(run_splice_job(clips, spec, output_dir, sink))
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
    filter_chain: Arc<codec::filter::FilterChain>,
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
    let gpu_pool = multigpu::gpu_pool_for_policy(spec.encode_policy, spec.video_codec.codec());
    if matches!(
        spec.encode_policy,
        EncodePolicy::AllGpus | EncodePolicy::Family(_)
    ) && total_input_frames > 0
        && gpu_pool.capacity() > 1
        // `ChunkSeamMode::Serial` forces one encoder (seam-free) even on a
        // multi-GPU host — skip the chunk-and-stitch path entirely.
        && spec.chunk_seam_mode != crate::spec::ChunkSeamMode::Serial
        // Trim/splice jobs take the serial path: the multi-GPU chunker sizes its
        // chunks from the full source frame count, which a trim invalidates.
        && spec.trim_start.is_none()
        && spec.trim_end.is_none()
    {
        // The chunk-and-stitch path's codec invariant now handles av1C / avcC /
        // hvcC, so AV1, H.264, and H.265 all chunk across GPUs. Each chunk is a
        // closed GOP (first frame an IDR), so stitched H.264/H.265 streams reset
        // refs cleanly at every chunk boundary.
        return run_single_file_multigpu(
            input,
            spec,
            header,
            frame_rate,
            total_input_frames,
            audio,
            gpu_pool,
            filter_chain,
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
    let base_cfg = EncoderConfig {
        frame_rate,
        pixel_format: output_pixel_format,
        color_metadata: output_color_metadata,
        gpu_index: encode_gpu,
        codec: spec.video_codec.codec(),
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
        filters: Arc::clone(&filter_chain),
    };
    // Splice trim: seconds → source frame indices at the output cadence, as a
    // half-open `[start_frame, end_frame)`. `ceil` makes the bounds exact for
    // any (possibly non-integer) detected fps — keep frame n iff
    // `start <= n/fps < end`. The pump drops out-of-range frames and the muxer
    // re-numbers the kept frames from zero (trimmed + rebased).
    let start_frame = trim_frame(spec.trim_start, frame_rate).unwrap_or(0);
    let end_frame = trim_frame(spec.trim_end, frame_rate);
    // Progress is reported against the trimmed length, not the full source.
    let effective_total = match (end_frame, frames_total) {
        (Some(end), _) => Some(end.saturating_sub(start_frame)),
        (None, Some(t)) => Some(t.saturating_sub(start_frame)),
        (None, None) => None,
    };
    // Trim the prepared audio to the same window so A/V stay aligned.
    let trimmed_audio = trim_audio(audio, spec.trim_start, spec.trim_end);
    let clip = crate::decode_pump::ClipSource { cfg: pump_cfg, input, start_frame, end_frame };
    run_serial_single_file(vec![clip], spec, base_cfg, frame_rate, effective_total, trimmed_audio, sink)
        .await
}

/// Convert a trim time (seconds) to a half-open source frame index at `fps`
/// (`ceil`, so `[start,end)` is exact for non-integer fps). `None` → `None`.
fn trim_frame(sec: Option<f64>, fps: f64) -> Option<u64> {
    sec.map(|s| (s.max(0.0) * fps).ceil() as u64)
}

/// Serial single-file encode of one or more (pre-trimmed) clips: the spliced
/// decode pump concatenates the clips' kept frames into one continuous stream,
/// and each rung worker encodes that stream into one MP4. Shared by the
/// single-input trim path and `run_splice_job` (multi-clip concat).
async fn run_serial_single_file(
    clips: Vec<crate::decode_pump::ClipSource>,
    spec: &OutputSpec,
    base_cfg: EncoderConfig,
    frame_rate: f64,
    effective_total: Option<u64>,
    audio: Option<PreparedAudio>,
    sink: Arc<dyn ProgressSink>,
) -> Result<Vec<RungOutput>> {
    let backend_override = encoder_backend_override();
    let rt = tokio::runtime::Handle::current();

    let mut senders = Vec::with_capacity(spec.rungs.len());
    let mut handles = Vec::with_capacity(spec.rungs.len());
    for (idx, rung) in spec.rungs.iter().cloned().enumerate() {
        let (tx, rx) = tokio::sync::mpsc::channel::<VideoFrame>(FRAME_CHANNEL_CAPACITY);
        senders.push(tx);
        let sink = Arc::clone(&sink);
        let base_cfg = base_cfg.clone();
        let audio = audio.clone();
        let handle = tokio::task::spawn_blocking(move || {
            let r = encode_rung_single_file(
                idx, &rung, rx, base_cfg, backend_override, frame_rate, effective_total,
                audio.as_ref(), sink.as_ref(),
            );
            (idx, rung, r)
        });
        handles.push(handle);
    }

    let pump_handle = {
        let rt = rt.clone();
        tokio::task::spawn_blocking(move || {
            crate::decode_pump::run_spliced_decode_pump_blocking(clips, senders, rt)
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
    filter_chain: Arc<codec::filter::FilterChain>,
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
        // Single-file multi-GPU is never spliced (trimmed/concat single-file
        // takes the serial path) — empty plan ⇒ the pump decodes from `input`.
        spliced_clips: Vec::new(),
        codec: spec.video_codec.codec(),
        rungs: &spec.rungs,
        header: header.clone(),
        source_color_metadata: header.info.color_metadata,
        source_pixel_format: header.info.pixel_format,
        tonemap_to_sdr: spec.tonemaps(),
        output_color_metadata,
        output_pixel_format,
        needs_downsample: needs_chroma_downsample(header.info.pixel_format),
        filters: Arc::clone(&filter_chain),
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
    // Multi-GPU stitch: chunks come from independent encoders (possibly
    // different vendors), so keep parameter sets inline per access unit
    // (avc3/hev1 for H.264/H.265). AV1 ignores the flag (it stores OBUs verbatim).
    let mut muxer = Av1Mp4Muxer::new_with_codec_inline(rp.width, rp.height, frame_rate, rp.codec)
        .context("Av1Mp4Muxer::new_with_codec_inline")?;
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
    let out_codec = cfg.codec;
    let mut encoder = encode::select_encoder(cfg, backend)
        .with_context(|| format!("creating encoder for rung {}", rung.label))?;
    let mut muxer = Av1Mp4Muxer::new_with_codec(rung.width, rung.height, frame_rate, out_codec)
        .context("Av1Mp4Muxer::new_with_codec")?;
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
/// Decode-pump config for one clip: codec/info/color from the header, tonemap +
/// filters from the spec. `gpu` is a placeholder for the splice plan — the
/// multi-GPU `clip_sources_for` overrides it per pump.
fn pump_cfg_for(
    header: &DemuxHeader,
    spec: &OutputSpec,
    filters: Arc<codec::filter::FilterChain>,
    gpu: Option<u32>,
) -> DecodePumpConfig {
    DecodePumpConfig {
        codec_name: header.codec.clone(),
        info_for_decoder: header.info.clone(),
        source_color_metadata: header.info.color_metadata,
        source_pixel_format: header.info.pixel_format,
        needs_downsample: needs_chroma_downsample(header.info.pixel_format),
        tonemap_to_sdr: spec.tonemaps(),
        gpu_index: gpu,
        filters,
    }
}

#[allow(clippy::too_many_arguments)]
async fn run_hls(
    input: Bytes,
    spec: &OutputSpec,
    segment_seconds: f32,
    header: &DemuxHeader,
    frame_rate: f64,
    audio: Option<&PreparedAudio>,
    filter_chain: Arc<codec::filter::FilterChain>,
    output_dir: Option<&Path>,
    sink: Arc<dyn ProgressSink>,
    // Splice plan: explicit clips (concat). Empty ⇒ single `input`, trimmed to
    // the spec's `[trim_start, trim_end)` window if set, else un-spliced.
    spliced_clips: Vec<crate::decode_pump::ClipSource>,
    // Pre-summed trimmed/concat frame total; `None` ⇒ derive from the source.
    effective_total: Option<u64>,
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

    // Resolve the decode plan. Concat clips win; otherwise a single input honors
    // the spec trim window (empty plan ⇒ the multi-GPU pump's input fallback).
    let start_frame = trim_frame(spec.trim_start, frame_rate).unwrap_or(0);
    let end_frame = trim_frame(spec.trim_end, frame_rate);
    let spliced_clips = if !spliced_clips.is_empty() {
        spliced_clips
    } else if start_frame == 0 && end_frame.is_none() {
        Vec::new()
    } else {
        vec![crate::decode_pump::ClipSource {
            cfg: pump_cfg_for(header, spec, Arc::clone(&filter_chain), None),
            input: input.clone(),
            start_frame,
            end_frame,
        }]
    };

    let source_total = if header.info.total_frames > 0 {
        header.info.total_frames
    } else {
        (header.info.duration * frame_rate).round().max(0.0) as u64
    };
    let total_input_frames = effective_total.unwrap_or_else(|| match end_frame {
        Some(end) => end.saturating_sub(start_frame),
        None => source_total.saturating_sub(start_frame),
    });

    let gpu_pool = multigpu::gpu_pool_for_policy(spec.encode_policy, spec.video_codec.codec());
    let (output_color_metadata, output_pixel_format) =
        spec.resolve_output(header.info.color_metadata, header.info.pixel_format);
    let params = MultiGpuParams {
        input,
        spliced_clips,
        codec: spec.video_codec.codec(),
        rungs: &spec.rungs,
        header: header.clone(),
        source_color_metadata: header.info.color_metadata,
        source_pixel_format: header.info.pixel_format,
        tonemap_to_sdr: spec.tonemaps(),
        output_color_metadata,
        output_pixel_format,
        needs_downsample: needs_chroma_downsample(header.info.pixel_format),
        filters: Arc::clone(&filter_chain),
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
    let codec_string = cmaf_util::codec_string_from_init(&rm.manifest.init_path)
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

    /// Append another track's samples after this one (for splice concat). The
    /// muxer re-times from the running duration, so the joined audio is gap-free.
    fn extend(&mut self, other: &PreparedAudio) {
        self.samples.extend(other.samples.iter().cloned());
    }
}

/// Trim a prepared audio track to the window `[start, end)` seconds, dropping
/// packets outside it. Kept packets retain their explicit durations, so the
/// muxer re-times them from zero — aligning with the trimmed, rebased video.
/// Cut points land on packet boundaries (≤ ~20 ms), which is fine for A/V sync.
/// `None`/`None` returns the track unchanged.
fn trim_audio(
    audio: Option<&PreparedAudio>,
    start: Option<f64>,
    end: Option<f64>,
) -> Option<PreparedAudio> {
    let a = audio?;
    if start.is_none() && end.is_none() {
        return Some(a.clone());
    }
    let ticks_per_sec = a.info.timescale.max(1) as f64;
    let start_tick = (start.unwrap_or(0.0).max(0.0) * ticks_per_sec) as u64;
    let end_tick = end.map(|e| (e.max(0.0) * ticks_per_sec) as u64);
    let mut acc: u64 = 0;
    let mut kept = Vec::new();
    for (payload, dur) in &a.samples {
        let sample_start = acc;
        acc += *dur as u64;
        if sample_start < start_tick {
            continue;
        }
        if end_tick.is_some_and(|et| sample_start >= et) {
            break;
        }
        kept.push((payload.clone(), *dur));
    }
    Some(PreparedAudio { info: a.info.clone(), samples: kept, handling: a.handling.clone() })
}

fn prepare_audio(track: Option<&AudioTrack>, policy: AudioCodecPolicy) -> Result<Option<PreparedAudio>> {
    let Some(track) = track else {
        return Ok(None);
    };
    if policy == AudioCodecPolicy::Drop {
        return Ok(None);
    }
    let codec = track.codec.to_ascii_lowercase();
    let passthrough_ok = matches!(codec.as_str(), "aac" | "opus" | "ac3" | "eac3");
    let force_opus = policy == AudioCodecPolicy::ForceOpus;

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

#[cfg(test)]
mod splice_tests {
    use super::*;

    #[test]
    fn trim_frame_is_half_open_exact() {
        // `[start, end)` must be exact even at a non-integer detected fps: a frame
        // whose time is < end_sec is kept (ceil), regardless of rounding.
        // 29.9 fps: 7 s = frame 209.3, so the exclusive end is 210 → frame 209
        // (at 6.99 s) IS kept.
        assert_eq!(trim_frame(Some(7.0), 29.9), Some(210));
        assert_eq!(trim_frame(Some(2.0), 29.9), Some(60)); // ceil(59.8)
        // 30 fps exact boundaries.
        assert_eq!(trim_frame(Some(2.0), 30.0), Some(60));
        assert_eq!(trim_frame(Some(5.0), 30.0), Some(150));
        // Open bound and zero.
        assert_eq!(trim_frame(None, 30.0), None);
        assert_eq!(trim_frame(Some(0.0), 30.0), Some(0));
        // Negative time clamps to 0.
        assert_eq!(trim_frame(Some(-3.0), 30.0), Some(0));
    }

    #[test]
    fn trim_audio_keeps_window_and_concat_appends() {
        // 8 packets, 1000 ticks each, timescale 1000 → one packet per second.
        let info = AudioInfo {
            codec: "opus".into(),
            sample_rate: 48000,
            channels: 2,
            timescale: 1000,
            asc_bytes: Vec::new(),
            codec_private: Vec::new(),
        };
        let mk = |n: usize| PreparedAudio {
            info: info.clone(),
            samples: (0..n).map(|i| (vec![i as u8], 1000u32)).collect(),
            handling: "passthrough".into(),
        };
        let a = mk(8);
        // Trim [2s, 5s) keeps packets starting at t=2,3,4 → indices 2,3,4.
        let t = trim_audio(Some(&a), Some(2.0), Some(5.0)).unwrap();
        assert_eq!(t.samples.len(), 3);
        assert_eq!(t.samples[0].0, vec![2u8]);
        assert_eq!(t.samples[2].0, vec![4u8]);
        // Open start keeps from 0; open end keeps to the end.
        assert_eq!(trim_audio(Some(&a), None, Some(3.0)).unwrap().samples.len(), 3);
        assert_eq!(trim_audio(Some(&a), Some(6.0), None).unwrap().samples.len(), 2);
        // No bounds → unchanged.
        assert_eq!(trim_audio(Some(&a), None, None).unwrap().samples.len(), 8);
        // Concat appends.
        let mut joined = mk(3);
        joined.extend(&mk(2));
        assert_eq!(joined.samples.len(), 5);
    }
}
