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
//!   cross-vendor codec invariant), then this
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
mod audio_only;
mod file_mux;
mod pump;
mod run;
mod splice;
mod subtitles;
#[cfg(test)]
mod audio_tests;
#[cfg(test)]
mod lossless_tests;
#[cfg(test)]
mod metadata_tests;
#[cfg(test)]
mod sample_entry_tests;
#[cfg(test)]
mod tests;

pub use splice::Clip;

use self::audio::{AudioRequest, PreparedAudio, audio_codec_string, fit_single_file, prepare_audio};
use self::pump::run_hls;
use self::run::{run_serial_single_file, run_single_file};
use self::splice::{trim_audio_to_video, trim_frame};
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
    /// The RFC 6381 `codecs` value of the output's audio (`opus`,
    /// `mp4a.40.2`, `mp3`, `ac-3`, …) — what a `<source type=…>` or an HLS
    /// `CODECS` attribute names it by. `None` when the output has no audio.
    pub audio_codecs: Option<String>,
    /// What fitting made of each rung asked for, in the order they were asked
    /// for: the output size, and for a rung dropped as the same as another
    /// (a source smaller than both boxes, upscale off), which one. `rungs`
    /// holds the produced ones, each under its `label` here.
    pub renditions: Vec<crate::fit::FittedRung>,
    pub elapsed: Duration,
    /// What the spec's hooks said ([`crate::hooks`]); empty with no hooks.
    pub hooks: crate::hooks::HookReport,
}

/// Run a transcode job. Async — call from within a Tokio runtime.
///
/// For [`OutputMode::Hls`], `output_dir` is the asset root the HLS package is
/// written under; `None` uses a fresh temp directory (returned in
/// [`JobOutput::hls_root`]). For [`OutputMode::SingleFile`] `output_dir` is
/// ignored (bytes are returned).
///
/// The spec's [hooks](crate::hooks) run each at its own point: source hooks
/// before the source is parsed, probe hooks once it is demuxed, decoded-frame
/// and encoder-frame hooks from the decode pumps, artifact hooks for each
/// output, then completed or failed hooks. A hook's
/// rejection is the job's error ([`crate::hooks::rejection_of`] finds it).
pub async fn run_job(
    input: Bytes,
    spec: &OutputSpec,
    output_dir: Option<&Path>,
    sink: Arc<dyn ProgressSink>,
) -> Result<JobOutput> {
    if spec.hooks.is_empty() {
        return run_job_inner(input, spec, output_dir, sink).await;
    }
    let kind = if spec.mode == OutputMode::AudioOnly {
        crate::hooks::JobKind::AudioOnly
    } else {
        crate::hooks::JobKind::Transcode
    };
    let hooks = spec.hooks.ensure_session(kind);
    let spec = spec.clone().with_hooks(hooks.clone());
    let run = async {
        let source = input.clone();
        hooks.offload(move |h| h.emit_source(0, &source)).await?;
        run_job_inner(input, &spec, output_dir, sink).await
    };
    let mut out = hooks.run(artifact_events, run).await?;
    out.hooks = hooks.report();
    Ok(out)
}

/// One artifact event per output of `out`: each single-file rung's bytes,
/// each HLS rendition's directory, the master playlist.
fn artifact_events(out: &JobOutput) -> Vec<crate::hooks::ArtifactEvent> {
    use crate::hooks::{ArtifactData, ArtifactEvent, ArtifactKind};
    let mut events: Vec<ArtifactEvent> = out
        .rungs
        .iter()
        .map(|r| match &r.artifact {
            RungArtifact::File(bytes) => ArtifactEvent {
                kind: if single_file_media_type(bytes).starts_with("audio/") {
                    ArtifactKind::Audio
                } else {
                    ArtifactKind::Video
                },
                label: r.label.clone(),
                media_type: single_file_media_type(bytes).to_string(),
                width: r.width,
                height: r.height,
                data: ArtifactData::Bytes(Bytes::copy_from_slice(bytes)),
            },
            RungArtifact::HlsRendition { dir, .. } => ArtifactEvent {
                kind: ArtifactKind::Rendition,
                label: r.label.clone(),
                media_type: "application/vnd.apple.mpegurl".into(),
                width: r.width,
                height: r.height,
                data: ArtifactData::Directory { path: dir.clone(), files: files_in(dir) },
            },
        })
        .collect();
    if let Some(master) = &out.master_playlist {
        events.push(ArtifactEvent {
            kind: ArtifactKind::Playlist,
            label: "master".into(),
            media_type: "application/vnd.apple.mpegurl".into(),
            width: 0,
            height: 0,
            data: ArtifactData::File(master.clone()),
        });
    }
    events
}

/// The files directly in `dir`, sorted.
fn files_in(dir: &Path) -> Vec<String> {
    let mut files: Vec<String> = std::fs::read_dir(dir)
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .filter(|e| e.file_type().is_ok_and(|t| t.is_file()))
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .collect()
        })
        .unwrap_or_default();
    files.sort();
    files
}

/// A single-file artifact's media type: an audio-only output is an `.mp3`,
/// a `.flac`, an `.m4a` or an `.ogg`, everything else an MP4.
pub fn single_file_media_type(data: &[u8]) -> &'static str {
    match container::sniff_container(data) {
        container::ContainerKind::Mp3 => "audio/mpeg",
        container::ContainerKind::Flac => "audio/flac",
        container::ContainerKind::Ogg => "audio/ogg",
        container::ContainerKind::Matroska => "video/webm",
        _ if data.get(4..12) == Some(b"ftypM4A ") => "audio/mp4",
        _ if data.get(4..12) == Some(b"ftypqt  ") => "video/quicktime",
        _ => "video/mp4",
    }
}

/// A single-file artifact's extension, by the same sniff as
/// [`single_file_media_type`]: `mp3`, `flac`, `m4a`, `opus` (Ogg Opus),
/// `ogg`, `webm`, `mov` or `mp4`.
pub fn single_file_extension(data: &[u8]) -> &'static str {
    match single_file_media_type(data) {
        "audio/mpeg" => "mp3",
        "audio/flac" => "flac",
        // The first page holds the identification header alone: 27 bytes of
        // page header and one lacing value before it.
        "audio/ogg" if data.get(28..36) == Some(b"OpusHead") => "opus",
        "audio/ogg" => "ogg",
        "audio/mp4" => "m4a",
        "video/webm" => "webm",
        "video/quicktime" => "mov",
        _ => "mp4",
    }
}

async fn run_job_inner(
    input: Bytes,
    spec: &OutputSpec,
    output_dir: Option<&Path>,
    sink: Arc<dyn ProgressSink>,
) -> Result<JobOutput> {
    let started = Instant::now();
    spec.validate().context("invalid OutputSpec")?;
    // Per-rung knobs by ladder position, folded into each rung up front so
    // nothing downstream has to know the ladder's shape.
    if spec.mode == OutputMode::AudioOnly {
        return audio_only::run(input, spec, sink, started).await;
    }

    let (header, audio_track, audio_edit, audio_gaps, video_delay, subtitle_tracks) = {
        let demuxer = match streaming::demux_streaming_shared(input.clone()) {
            Ok(d) => d,
            // An input with no video (a bare MP3, an M4A, an audio-only
            // Matroska) has nothing for a ladder; a single-file job becomes
            // its audio-only form, and says so.
            Err(e) => match audio_only::as_audio_only(&input, spec) {
                Some(audio_spec) => {
                    tracing::info!("the input has no video: writing its audio alone (audio-only output)");
                    let audio_spec = audio_spec.context("the input has no video, and audio-only output")?;
                    return audio_only::run(input, &audio_spec, sink, started).await;
                }
                None => return Err(e).context("demux"),
            },
        };
        (
            demuxer.header().clone(),
            demuxer.audio().cloned(),
            demuxer.audio_edit(),
            demuxer.audio_gaps().to_vec(),
            video_delay_of(demuxer.as_ref()),
            demuxer.subtitles().to_vec(),
        )
    };
    // Each rung's box becomes its output size for this source; the rung
    // policy (which reads sizes) is resolved on those.
    let (fitted, renditions) = fit_to(spec, &header);
    let policy_resolved = fitted.with_rung_policy_resolved();
    let spec = &policy_resolved;
    let sink = remap_rung_indices(sink, &renditions);
    // A colour policy that could only re-tag this source is refused before a
    // frame is decoded (an HDR policy on the other HDR transfer, or on an SDR
    // source the BT.2408 mapping cannot take).
    spec.check_source_colour(&header.info.color_metadata)?;
    // What the source makes of the output — a 10-bit source kept at its
    // depth, an HDR one passed through — needs an encoder for the codec that
    // takes it: refused here, by name, before anything is decoded.
    spec.check_source(header.info.color_metadata, header.info.pixel_format)
        .context("invalid OutputSpec")?;
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
    spec.hooks.emit_probe(
        0,
        crate::hooks::MediaSummary::of_header(
            container::sniff_container(&input).label(),
            &header,
            audio_track.as_ref().map(|t| t.codec.to_ascii_lowercase()),
        ),
    )?;
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
    // A constant-rate rung (`rate=cbr`) with no rate of its own takes the
    // default for its codec, size and this output frame rate, here, so every
    // encoder and the HLS playlist see the rate it is coded at.
    let rates_resolved = spec.with_constant_rates_resolved(frame_rate);
    let spec = &rates_resolved;
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

    let prepared_audio = prepare_audio(audio_track.as_ref(), audio_edit, &audio_gaps, AudioRequest::of(spec))
        .context("preparing audio")?;
    let prepared_audio = match spec.mode {
        OutputMode::SingleFile => fit_single_file(prepared_audio, spec.container),
        OutputMode::Hls { .. } | OutputMode::AudioOnly => prepared_audio,
    };
    let stereo_fallback = stereo_fallback(spec, prepared_audio.as_ref(), || {
        prepare_audio(audio_track.as_ref(), audio_edit, &audio_gaps, stereo_request(spec))
    });
    let audio_handling = describe_audio(prepared_audio.as_ref(), stereo_fallback.as_ref());
    let audio_codecs = prepared_audio.as_ref().filter(|a| a.has_samples()).map(|a| audio_codec_string(&a.info));

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
                video_delay,
            )
            .await?;
            let mut rungs = rungs;
            keep_metadata(&input, spec, &mut rungs)?;
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
                stereo_fallback.as_ref().and_then(|r| r.as_ref().ok()),
                &subtitles,
                Arc::clone(&filter_chain),
                output_dir,
                Arc::clone(&sink),
                // Single input: run_hls builds the (optionally trimmed) plan
                // from spec.trim itself.
                Vec::new(),
                None,
                video_delay,
            )
            .await?
        }
        OutputMode::AudioOnly => unreachable!("an audio-only job returned above"),
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
        audio_codecs,
        renditions,
        elapsed: started.elapsed(),
        hooks: crate::hooks::HookReport::default(),
    })
}

/// Write the source metadata `spec.metadata_keep` names into each file
/// output, in the place its container has for it. Nothing is written — and
/// the source is not read — when it names none, so an output carries only
/// what the muxer wrote: no location, device, time or tags.
pub(super) fn keep_metadata(input: &[u8], spec: &OutputSpec, rungs: &mut [RungOutput]) -> Result<()> {
    use crate::spec::Container;
    use container::metadata::{self, write};
    if spec.metadata_keep.is_empty() {
        return Ok(());
    }
    let kept = metadata::read(input).kept(spec.metadata_keep);
    tracing::info!(
        keep = %spec.metadata_keep,
        found = %kept.categories(),
        "carrying the source metadata asked for into the output"
    );
    for rung in rungs {
        let RungArtifact::File(bytes) = &mut rung.artifact else { continue };
        let written = match spec.container {
            // A QuickTime movie is the same box tree: `udta` / `meta` alike.
            Container::Mp4 | Container::M4a | Container::Mov => {
                write::mp4(bytes, &kept).context("writing the kept metadata")?
            }
            Container::WebM => bail!("metadata-keep is not available for WebM output"),
            Container::Ogg => bail!("metadata-keep is not available for Ogg output"),
            Container::Flac => write::flac(bytes, &kept).context("writing the kept metadata")?,
            Container::Mp3 => write::mp3(bytes, &kept),
            Container::Cmaf => bail!("metadata-keep is not available for HLS output"),
        };
        rung.bytes = written.len() as u64;
        *bytes = written;
    }
    Ok(())
}

/// `spec` with its rungs fitted to the source `header` describes: upright,
/// through the size-changing filters, with its sample shape. See
/// [`crate::fit`].
fn fit_to(spec: &OutputSpec, header: &DemuxHeader) -> (OutputSpec, Vec<crate::fit::FittedRung>) {
    let (width, height) = header.upright_dims();
    let upright = crate::fit::SourceShape { width, height, sample_aspect: header.upright_sample_aspect() };
    let shape = crate::fit::filtered_shape(upright, &spec.filters);
    let (fitted, renditions) = spec.with_rungs_fitted(shape);
    for (r, f) in renditions.iter().zip(&spec.rungs) {
        if r.duplicate_of.is_none() && r.requested != r.output {
            tracing::info!(
                rung = %r.label,
                requested = %format!("{}x{}", r.requested.0, r.requested.1),
                output = %format!("{}x{}", r.output.0, r.output.1),
                fit = %f.fit.unwrap_or(spec.fit),
                source = %format!("{}x{} ({}:{} samples)", shape.width, shape.height, shape.sample_aspect.0, shape.sample_aspect.1),
                "rung fitted to the source"
            );
        }
    }
    (fitted, renditions)
}

/// A sink that reports each rung under its position in the request rather
/// than in the fitted ladder, which lacks the rungs fitting dropped — so a
/// caller's progress lines up with the rungs it asked for. The sink itself
/// when nothing was dropped.
fn remap_rung_indices(sink: Arc<dyn ProgressSink>, renditions: &[crate::fit::FittedRung]) -> Arc<dyn ProgressSink> {
    let requested: Vec<usize> =
        renditions.iter().enumerate().filter(|(_, r)| r.duplicate_of.is_none()).map(|(i, _)| i).collect();
    if requested.len() == renditions.len() {
        return sink;
    }
    struct Remap {
        inner: Arc<dyn ProgressSink>,
        requested: Vec<usize>,
    }
    impl ProgressSink for Remap {
        fn on_rung(&self, mut update: RungProgress) {
            update.rung_index = self.requested.get(update.rung_index).copied().unwrap_or(update.rung_index);
            self.inner.on_rung(update);
        }
        fn on_event(&self, event: JobEvent) {
            self.inner.on_event(event);
        }
        fn on_rung_complete(&self, manifest: &crate::multigpu::RungManifest) {
            let mut manifest = manifest.clone();
            manifest.rung_index = self.requested.get(manifest.rung_index).copied().unwrap_or(manifest.rung_index);
            self.inner.on_rung_complete(&manifest);
        }
    }
    Arc::new(Remap { inner: sink, requested })
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
/// shared encoder back-to-back. The muxer times output frames by count and
/// orders them by timestamp, and the pump carries each clip's timestamps on
/// from the clip before, so the join is gap-free and the timeline is
/// zero-based. Audio is trimmed per clip and concatenated to match.
///
/// Output config (frame rate, color) follows the **first** clip; inputs are
/// re-encoded to the spec's uniform output, so they may differ in codec /
/// resolution / color. A one-clip `Vec` is a plain (optionally trimmed)
/// transcode. Text subtitles are trimmed per clip, re-based onto the joined
/// timeline by the length of the clips before them, and merged by language. Honors the spec's [`OutputMode`]: `SingleFile` writes one MP4 per
/// rung; `Hls` writes a CMAF/HLS package (the spliced frame stream feeds the
/// multi-GPU HLS engine, so segments are keyframe-aligned across the join).
///
/// Hooks run as for [`run_job`]; each clip is its own input, its events
/// carrying its position (`clip`).
pub async fn run_splice_job(
    clips: Vec<Clip>,
    spec: &OutputSpec,
    output_dir: Option<&Path>,
    sink: Arc<dyn ProgressSink>,
) -> Result<JobOutput> {
    if spec.hooks.is_empty() {
        return run_splice_job_inner(clips, spec, output_dir, sink).await;
    }
    let hooks = spec.hooks.ensure_session(crate::hooks::JobKind::Splice);
    let spec = spec.clone().with_hooks(hooks.clone());
    let run = async {
        let inputs: Vec<Bytes> = clips.iter().map(|c| c.input.clone()).collect();
        hooks
            .offload(move |h| inputs.iter().enumerate().try_for_each(|(i, b)| h.emit_source(i, b)))
            .await?;
        run_splice_job_inner(clips, &spec, output_dir, sink).await
    };
    let mut out = hooks.run(artifact_events, run).await?;
    out.hooks = hooks.report();
    Ok(out)
}

async fn run_splice_job_inner(
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
    if spec.mode == OutputMode::AudioOnly {
        bail!("audio-only output is not available for a splice: run the clips as single-file jobs");
    }
    if !spec.metadata_keep.is_empty() {
        bail!("metadata-keep is not available for a splice: its clips can each say something different");
    }
    // Probe each clip + prepare its audio. The first clip drives output config.
    struct ClipPrep {
        header: DemuxHeader,
        audio: Option<PreparedAudio>,
        /// The clip's stereo downmix, for an HLS stereo fallback.
        audio_stereo: Option<Result<PreparedAudio, String>>,
        src_audio_codec: Option<String>,
        subtitles: Vec<SubtitleTrack>,
        video_delay: (u64, u32),
    }
    let mut preps = Vec::with_capacity(clips.len());
    for (i, clip) in clips.iter().enumerate() {
        let demuxer = streaming::demux_streaming_shared(clip.input.clone())
            .with_context(|| format!("demuxing splice clip {i}"))?;
        let header = demuxer.header().clone();
        spec.hooks.emit_probe(
            i,
            crate::hooks::MediaSummary::of_header(
                container::sniff_container(&clip.input).label(),
                &header,
                demuxer.audio().map(|t| t.codec.to_ascii_lowercase()),
            ),
        )?;
        spec.check_source_colour(&header.info.color_metadata)
            .with_context(|| format!("splice clip {i}"))?;
        if i == 0 {
            // The output follows the first clip: what it makes of the output
            // (a 10-bit source kept at its depth, an HDR one passed through)
            // needs an encoder that takes it, refused before anything decodes.
            spec.check_source(header.info.color_metadata, header.info.pixel_format)
                .context("invalid OutputSpec")?;
        }
        let src_audio_codec = demuxer.audio().map(|t| t.codec.to_ascii_lowercase());
        let audio = prepare_audio(demuxer.audio(), demuxer.audio_edit(), demuxer.audio_gaps(), AudioRequest::of(spec))
            .with_context(|| format!("preparing audio for splice clip {i}"))?;
        let audio_stereo = stereo_fallback(spec, audio.as_ref(), || {
            prepare_audio(demuxer.audio(), demuxer.audio_edit(), demuxer.audio_gaps(), stereo_request(spec))
        });
        let subtitles = demuxer.subtitles().to_vec();
        let video_delay = video_delay_of(demuxer.as_ref());
        if i > 0 && video_delay.0 != 0 {
            tracing::info!(
                clip_index = i,
                delay_ticks = video_delay.0,
                timescale = video_delay.1,
                "splice: this clip's video starts late (an empty edit); only the first clip's start \
                 delay can be written, so the join is gap-free and the clip's audio joins from where \
                 its video starts"
            );
        }
        preps.push(ClipPrep { header, audio, audio_stereo, src_audio_codec, subtitles, video_delay });
    }

    let primary = preps[0].header.clone();
    // The rungs fit the first clip, as the rest of the output follows it; a
    // later clip of another shape is fitted into the same outputs frame by
    // frame (see `Placement::apply`).
    let (fitted, renditions) = fit_to(spec, &primary);
    let policy_resolved = fitted.with_rung_policy_resolved();
    let spec = &policy_resolved;
    let sink = remap_rung_indices(sink, &renditions);
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
    // A constant-rate rung (`rate=cbr`) with no rate of its own takes the
    // default for its codec, size and this output frame rate, here, so every
    // encoder and the HLS playlist see the rate it is coded at.
    let rates_resolved = spec.with_constant_rates_resolved(frame_rate);
    let spec = &rates_resolved;

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
    // The policy's pool, for two things: the check it does — a pin the host
    // cannot serve is refused here, by name, before a clip is decoded — and
    // where a serial encode lands under it (the pool's first card, pinned by
    // index and vendor for a policy that names silicon; auto otherwise).
    let encode_pool = multigpu::gpu_pool_for_serial_job(
        spec,
        spec.resolve_output(primary.info.color_metadata, primary.info.pixel_format).1,
    )?;
    let (encode_gpu, encode_vendor) = multigpu::serial_target(spec.encode_policy, &encode_pool);
    // Average-rate rungs are coded by the software encoder only, constant-rate
    // ones by the cards (see `multigpu::check_rate_pool`); a splice is serial,
    // so the pin counts.
    multigpu::check_rate_pool(
        spec,
        &encode_pool,
        spec.resolve_output(primary.info.color_metadata, primary.info.pixel_format).1,
        run::encoder_backend_override(),
    )?;
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
        gpu_vendor: encode_vendor,
        codec: spec.video_codec.codec(),
        ..EncoderConfig::default()
    };

    // One decode source per clip (own decoder cfg + trim range); concatenate the
    // trimmed audio and sum the expected frame total across clips.
    let mut clip_sources = Vec::with_capacity(clips.len());
    let mut combined_audio: Option<PreparedAudio> = None;
    // A stereo fallback is joined like the audio, and only when every clip
    // has one: a clip whose downmix failed leaves the package without it.
    let mut combined_stereo: Option<Result<PreparedAudio, String>> = None;
    let stereo_everywhere = preps.iter().all(|p| p.audio_stereo.is_some());
    // Subtitles join by language; `offset_seconds` is where the next clip
    // starts on the output timeline, from the frames kept so far.
    let mut combined_subtitles: Vec<SubtitleTrack> = Vec::new();
    let mut offset_seconds: f64 = 0.0;
    let mut effective_total: u64 = 0;
    let mut total_known = true;
    for (i, (clip, prep)) in clips.iter().zip(preps.iter()).enumerate() {
        let cfps = if prep.header.info.frame_rate > 0.0 {
            prep.header.info.frame_rate
        } else {
            frame_rate
        };
        let start_frame = trim_frame(clip.start, cfps).unwrap_or(0);
        let end_frame = trim_frame(clip.end, cfps);
        // A frame-rate cap below this clip's rate drops its frames; totals
        // and offsets count what reaches the output.
        let clip_decimate = crate::decode_pump::decimation(prep.header.info.frame_rate, spec.max_frame_rate);
        match end_frame {
            Some(e) => {
                effective_total += crate::decode_pump::output_frames(e.saturating_sub(start_frame), clip_decimate)
            }
            None if prep.header.info.total_frames > 0 => {
                effective_total += crate::decode_pump::output_frames(
                    prep.header.info.total_frames.saturating_sub(start_frame),
                    clip_decimate,
                )
            }
            None => total_known = false,
        }
        // The first clip's late video start is written as the output's; a
        // later clip's video joins with none, and its audio follows it.
        let video_delay = if i == 0 { (0, 1) } else { prep.video_delay };
        let clip_audio =
            trim_audio_to_video(prep.audio.as_ref(), video_delay, clip.start, clip.end);
        if let Some(a) = clip_audio {
            if let Some(c) = combined_audio.as_mut() {
                c.extend(&a);
            } else {
                combined_audio = Some(a);
            }
        }
        match (stereo_everywhere, prep.audio_stereo.as_ref()) {
            (true, Some(Ok(s))) => {
                if let Some(a) = trim_audio_to_video(Some(s), video_delay, clip.start, clip.end) {
                    match combined_stereo.as_mut() {
                        Some(Ok(c)) => c.extend(&a),
                        Some(Err(_)) => {}
                        None => combined_stereo = Some(Ok(a)),
                    }
                }
            }
            (true, Some(Err(why))) => combined_stereo = Some(Err(why.clone())),
            _ => {}
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
        offset_seconds +=
            crate::decode_pump::output_frames(kept_frames, clip_decimate) as f64 / frame_rate.max(1.0);
        let pump_cfg = DecodePumpConfig {
            codec_name: prep.header.codec.clone(),
            info_for_decoder: prep.header.info.clone(),
            source_color_metadata: prep.header.info.color_metadata,
            source_pixel_format: prep.header.info.pixel_format,
            needs_downsample: needs_chroma_downsample(prep.header.info.pixel_format),
            chroma_downsample: spec.chroma_downsample,
            output_pixel_format,
            tonemap_to_sdr: spec.tonemaps(),
            // Against the first clip's output colour: an SDR clip joined to an
            // HDR one under passthrough is mapped into the output's HDR.
            sdr_to_hdr: crate::spec::sdr_into_hdr(
                spec.tonemaps(),
                &prep.header.info.color_metadata,
                &output_color_metadata,
            ),
            gpu_index: decode_gpu,
            sample_range: None,
            rotation_degrees: prep.header.rotation_degrees,
            filters: Arc::clone(&filter_chain),
            decimate: clip_decimate,
            hooks: spec.hooks.clone(),
        };
        clip_sources.push(ClipSource {
            cfg: pump_cfg,
            input: clip.input.clone(),
            start_frame,
            end_frame,
        });
    }
    let effective_total = total_known.then_some(effective_total);
    let combined_audio = match spec.mode {
        OutputMode::SingleFile => fit_single_file(combined_audio, spec.container),
        OutputMode::Hls { .. } | OutputMode::AudioOnly => combined_audio,
    };
    let audio_handling = describe_audio(combined_audio.as_ref(), combined_stereo.as_ref());
    let audio_codecs = combined_audio.as_ref().filter(|a| a.has_samples()).map(|a| audio_codec_string(&a.info));

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
                preps[0].video_delay,
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
                combined_stereo.as_ref().and_then(|r| r.as_ref().ok()),
                &combined_subtitles,
                Arc::clone(&filter_chain),
                output_dir,
                Arc::clone(&sink),
                clip_sources,
                effective_total,
                preps[0].video_delay,
            )
            .await?
        }
        OutputMode::AudioOnly => unreachable!("refused above"),
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
        audio_codecs,
        renditions,
        elapsed: started.elapsed(),
        hooks: crate::hooks::HookReport::default(),
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

/// The source's late video start as `(ticks, ticks per second)`, `(0, 1)` for
/// none — what every output writes as its video track's delay (an empty edit
/// in an MP4, the first `tfdt` of a CMAF rendition).
pub(super) fn video_delay_of(demuxer: &dyn container::streaming::StreamingDemuxer) -> (u64, u32) {
    demuxer.video_presentation().map_or((0, 1), |p| (p.delay_ticks, p.delay_timescale))
}

/// Report a rung that failed with `error`: the warning and the rung's
/// progress message carry the whole chain. The outermost context alone was
/// all they said — "finalize" — which hid why every two-clip splice failed.
pub(super) fn report_rung_error(sink: &dyn ProgressSink, rung_index: usize, rung: &Rung, error: &anyhow::Error) {
    let error = format!("{error:#}");
    tracing::warn!(rung = %rung.label, %error, "rung failed");
    report_failed(sink, rung_index, rung, &error);
}

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

/// The request for an HLS stereo fallback: the spec's, downmixed to stereo.
fn stereo_request(spec: &OutputSpec) -> AudioRequest<'_> {
    AudioRequest { channels: crate::spec::AudioChannels::Stereo, ..AudioRequest::of(spec) }
}

/// The stereo downmix to put beside `main` in an HLS package, when the spec
/// asks for one and `main` is surround: `prepare` builds it. `Err` carries
/// why it could not be made (an AAC track cannot be decoded to downmix), so
/// the job reports it rather than quietly shipping the surround alone.
fn stereo_fallback(
    spec: &OutputSpec,
    main: Option<&PreparedAudio>,
    prepare: impl FnOnce() -> Result<Option<PreparedAudio>>,
) -> Option<Result<PreparedAudio, String>> {
    let main = main?;
    if !spec.audio_stereo_fallback || !main.has_samples() || main.info.channels <= 2 {
        return None;
    }
    Some(match prepare() {
        Ok(Some(stereo)) if stereo.has_samples() => Ok(stereo),
        Ok(_) => Err("the downmix produced no audio".into()),
        Err(e) => {
            tracing::warn!(error = %format!("{e:#}"), "no stereo fallback rendition: the surround one goes alone");
            Err(format!("{e:#}"))
        }
    })
}

/// The job's audio handling, with the stereo fallback's when there is one.
fn describe_audio(main: Option<&PreparedAudio>, stereo: Option<&Result<PreparedAudio, String>>) -> String {
    let main = main.map_or_else(|| "none".to_string(), |a| a.handling.clone());
    match stereo {
        None => main,
        Some(Ok(s)) => format!("{main}; stereo fallback: {}", s.handling),
        Some(Err(why)) => format!("{main}; no stereo fallback ({why})"),
    }
}
