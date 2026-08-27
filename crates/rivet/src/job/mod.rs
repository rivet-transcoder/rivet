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
//!   module assembles the HLS package (audio rendition + WebVTT subtitle
//!   renditions + playlists).

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use bytes::Bytes;

use codec::encode::EncoderConfig;
use container::demux::subtitle::SubtitleTrack;
use container::streaming::{self, DemuxHeader};

use crate::decode_pump::{ClipSource, DecodePumpConfig};
use crate::multigpu;
use crate::progress::{JobEvent, ProgressSink, RungProgress, RungStatus};
use crate::spec::{OutputMode, OutputSpec, Rung};
use crate::validate::needs_chroma_downsample;

mod audio;
mod pump;
mod run;
mod splice;
mod subtitles;
#[cfg(test)]
mod tests;

pub use splice::Clip;

use self::audio::{PreparedAudio, prepare_audio};
use self::pump::run_hls;
use self::run::{run_serial_single_file, run_single_file};
use self::splice::{trim_audio, trim_frame};
use self::subtitles::{append_clip_subtitles, trim_subtitles};

/// Bounded per-rung frame channel — backpressures the decode pump.
pub(super) const FRAME_CHANNEL_CAPACITY: usize = 8;

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
    // Per-rung knobs by ladder position, folded into each rung up front so
    // nothing downstream has to know the ladder's shape.
    let policy_resolved = spec.with_rung_policy_resolved();
    let spec = &policy_resolved;

    let (header, audio_track, subtitle_tracks) = {
        let demuxer = streaming::demux_streaming_shared(input.clone()).context("demux")?;
        (
            demuxer.header().clone(),
            demuxer.audio().cloned(),
            demuxer.subtitles().to_vec(),
        )
    };
    // `-c:s copy` equivalent: carry the selected text tracks. A trim re-bases
    // them the way it re-bases the audio — cues clipped to the kept window
    // and moved to zero — so they line up with the re-numbered frames.
    let subtitles: Vec<SubtitleTrack> =
        trim_subtitles(&spec.subtitles.select(&subtitle_tracks), spec.trim_start, spec.trim_end);
    if !subtitle_tracks.is_empty() {
        tracing::info!(
            source = ?subtitle_tracks.iter().map(|t| format!("{}:{}", t.language, t.codec)).collect::<Vec<_>>(),
            carried = ?subtitles.iter().map(|t| t.language.as_str()).collect::<Vec<_>>(),
            policy = ?spec.subtitles,
            "subtitle tracks selected"
        );
    }
    let source_codec = header.codec.to_ascii_lowercase();
    // As seen, not as stored: the pump turns every frame upright, so a 90°/270°
    // source arrives with its stored width and height swapped.
    let source_dims = header.upright_dims();
    let source_frame_rate = header.info.frame_rate;
    if header.rotation_degrees != 0 {
        tracing::info!(
            rotation_degrees = header.rotation_degrees,
            stored = %format!("{}x{}", header.info.width, header.info.height),
            upright = %format!("{}x{}", source_dims.0, source_dims.1),
            "source carries a rotation; every rung will be turned upright"
        );
    }

    // `DecodePolicy::FastestGpu`: benchmark each decode-capable GPU on a short
    // prefix of the input and resolve the policy to `SpecificGpu(fastest)`.
    // A no-op when fewer than two candidates exist (nothing to choose). Rebinds
    // `spec` to a clone carrying the resolved policy; everything downstream
    // reads `spec.decode_policy.gpu_index()`.
    let resolved_spec;
    let spec = if spec.decode_policy.is_fastest() {
        let candidates = codec::decode::decode_capable_gpu_indices(&source_codec);
        if candidates.len() > 1 {
            match crate::decode_pump::fastest_decode_gpu(
                &source_codec,
                &header.info,
                &input,
                &candidates,
                crate::decode_pump::DECODE_BENCH_FRAMES,
            ) {
                Some(gpu) => {
                    let mut s = spec.clone();
                    s.decode_policy = crate::spec::DecodePolicy::SpecificGpu(gpu);
                    resolved_spec = s;
                    &resolved_spec
                }
                None => spec,
            }
        } else {
            tracing::info!(
                candidates = candidates.len(),
                "decode-with-fastest: fewer than two decode-capable GPUs; nothing to benchmark"
            );
            spec
        }
    } else {
        spec
    };

    sink.on_event(JobEvent::Started { rungs: spec.rungs.len() });
    sink.on_event(JobEvent::Probed {
        codec: source_codec.clone(),
        width: source_dims.0,
        height: source_dims.1,
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

    // An audio filter that reaches no audio is a mistake worth stopping for.
    // The demuxer drops a track it can neither pass through nor decode (DTS,
    // TrueHD, …) and hands us `None`, which would otherwise make `--audio-filter`
    // and `--audio-bitrate` evaporate into a warning buried in the log while the
    // output silently ships with no audio at all.
    if audio_track.is_none() && !spec.audio_filters.is_empty() {
        bail!(
            "audio filters were requested ({}) but this input has no usable audio track — \
             either it has none, or its codec can be neither passed through (AAC / Opus / \
             AC-3 / E-AC-3) nor decoded (Vorbis / MP3). Check the demux warning above for \
             the codec, and drop `--audio-filter` to continue without it.",
            codec::audio::filter::chain_to_string(&spec.audio_filters)
        );
    }

    let prepared_audio = prepare_audio(
        audio_track.as_ref(),
        spec.audio,
        spec.audio_bitrate,
        &spec.audio_filters,
    )
    .context("preparing audio")?;
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
                &subtitles,
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
                &subtitles,
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

/// [`run_job_blocking`] over a buffer the caller already owns.
///
/// The slice form has to copy — the job outlives the borrow — which on a
/// multi-gigabyte source is a second full allocation before a single frame is
/// decoded. Callers holding the input as `Bytes` (the CLI, which reads the file
/// once) should use this and pay nothing.
pub fn run_job_blocking_owned(
    input: Bytes,
    spec: &OutputSpec,
    output_dir: Option<&Path>,
    sink: Arc<dyn ProgressSink>,
) -> Result<JobOutput> {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("building Tokio runtime")?;
    rt.block_on(run_job(input, spec, output_dir, sink))
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
/// transcode. Text subtitles are trimmed per clip, re-based onto the joined
/// timeline by the length of the clips before them, and merged by language. Honors the spec's [`OutputMode`]: `SingleFile` writes one MP4 per
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
    let policy_resolved = spec.with_rung_policy_resolved();
    let spec = &policy_resolved;

    // Probe each clip + prepare its audio. The first clip drives output config.
    struct ClipPrep {
        header: DemuxHeader,
        audio: Option<PreparedAudio>,
        src_audio_codec: Option<String>,
        subtitles: Vec<SubtitleTrack>,
    }
    let mut preps = Vec::with_capacity(clips.len());
    for (i, clip) in clips.iter().enumerate() {
        let demuxer = streaming::demux_streaming_shared(clip.input.clone())
            .with_context(|| format!("demuxing splice clip {i}"))?;
        let header = demuxer.header().clone();
        let src_audio_codec = demuxer.audio().map(|t| t.codec.to_ascii_lowercase());
        let audio = prepare_audio(
            demuxer.audio(),
            spec.audio,
            spec.audio_bitrate,
            &spec.audio_filters,
        )
        .with_context(|| format!("preparing audio for splice clip {i}"))?;
        let subtitles = demuxer.subtitles().to_vec();
        preps.push(ClipPrep { header, audio, src_audio_codec, subtitles });
    }

    let primary = preps[0].header.clone();
    let source_codec = primary.codec.to_ascii_lowercase();
    let source_dims = primary.upright_dims();
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
        width: source_dims.0,
        height: source_dims.1,
        frame_rate: primary.info.frame_rate,
        audio_codec: preps[0].src_audio_codec.clone(),
    });

    // Concat re-encodes every clip to one uniform output that follows the FIRST
    // clip. Resolution differences are handled (each frame is scaled to the
    // rung), but frame rate is NOT converted — a clip with a different fps keeps
    // its frames and is timed at the output rate, which shifts its playback
    // speed. Warn so the operator can pre-normalise fps if that matters.
    for (i, prep) in preps.iter().enumerate().skip(1) {
        let dims = prep.header.upright_dims();
        let fps = prep.header.info.frame_rate;
        let fps_differs = fps > 0.0
            && primary.info.frame_rate > 0.0
            && (fps - primary.info.frame_rate).abs() > 0.5;
        if dims != source_dims || fps_differs {
            tracing::warn!(
                clip_index = i,
                clip = %format!("{}x{} @ {:.3} fps", dims.0, dims.1, fps),
                output = %format!(
                    "{}x{} @ {:.3} fps",
                    source_dims.0, source_dims.1, primary.info.frame_rate
                ),
                fps_differs,
                "splice clip differs from the first clip: resolution is scaled to \
                 the output; frame rate is NOT converted (a differing fps shifts \
                 this clip's timing)"
            );
        }
    }

    let filter_chain = Arc::new(
        codec::filter::FilterChain::prepare(&spec.filters).context("preparing video filters")?,
    );
    let encode_gpu = multigpu::serial_gpu_for_policy(spec.encode_policy);
    // `--decode-with-fastest`: benchmark decode-capable GPUs on the first clip
    // and prefer the quickest for the pump (the same decode GPU is used for
    // every clip). Falls through to the explicit override / policy GPU.
    let fastest_decode = if spec.decode_policy.is_fastest() {
        let candidates = codec::decode::decode_capable_gpu_indices(&primary.codec);
        if candidates.len() > 1 {
            crate::decode_pump::fastest_decode_gpu(
                &primary.codec,
                &primary.info,
                &clips[0].input,
                &candidates,
                crate::decode_pump::DECODE_BENCH_FRAMES,
            )
        } else {
            None
        }
    } else {
        None
    };
    let decode_gpu = spec.decode_policy.gpu_index().or(fastest_decode).or(encode_gpu);
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
    // Subtitles join by language; `offset_seconds` is where the next clip
    // starts on the output timeline, from the frames kept so far.
    let mut combined_subtitles: Vec<SubtitleTrack> = Vec::new();
    let mut offset_seconds: f64 = 0.0;
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
        // The clip's cues, clipped to its window, moved to where the clip
        // starts in the output. The clip's length on the output timeline is
        // its kept frames at the output rate — the same arithmetic that
        // numbers the video frames — so the cues stay with their pictures.
        let clip_subs = trim_subtitles(&spec.subtitles.select(&prep.subtitles), clip.start, clip.end);
        append_clip_subtitles(&mut combined_subtitles, &clip_subs, offset_seconds);
        let kept_frames = match end_frame {
            Some(e) => e.saturating_sub(start_frame),
            None => {
                let total = if prep.header.info.total_frames > 0 {
                    prep.header.info.total_frames
                } else {
                    (prep.header.info.duration * cfps).round().max(0.0) as u64
                };
                total.saturating_sub(start_frame)
            }
        };
        offset_seconds += kept_frames as f64 / frame_rate.max(1.0);
        let pump_cfg = DecodePumpConfig {
            codec_name: prep.header.codec.clone(),
            info_for_decoder: prep.header.info.clone(),
            source_color_metadata: prep.header.info.color_metadata,
            source_pixel_format: prep.header.info.pixel_format,
            needs_downsample: needs_chroma_downsample(prep.header.info.pixel_format),
            output_pixel_format,
            tonemap_to_sdr: spec.tonemaps(),
            gpu_index: decode_gpu,
            sample_range: None,
            rotation_degrees: prep.header.rotation_degrees,
            filters: Arc::clone(&filter_chain),
        };
        clip_sources.push(ClipSource {
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
                combined_subtitles,
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
                &combined_subtitles,
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
// Shared helpers used across submodules
// ---------------------------------------------------------------------------

pub(super) fn report_failed(sink: &dyn ProgressSink, rung_index: usize, rung: &Rung, message: &str) {
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
