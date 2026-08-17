use std::sync::Arc;

use anyhow::{Context, Result, bail};
use bytes::Bytes;

use codec::colorspace;
use codec::encode::{self, EncoderBackend, EncoderConfig};
use codec::frame::{ColorMetadata, VideoFrame};
use container::demux::subtitle::SubtitleTrack;
use container::mux::Av1Mp4Muxer;
use container::streaming::DemuxHeader;

use crate::decode_pump::{self, ClipSource};
use crate::multigpu::{self, MultiGpuParams, RungPackets};
use crate::progress::{ProgressSink, RungProgress, RungStatus};
use crate::spec::{OutputSpec, Rung};
use crate::validate::needs_chroma_downsample;

use super::{RungArtifact, RungOutput, FRAME_CHANNEL_CAPACITY, report_failed};
use super::audio::PreparedAudio;
use super::splice::{trim_frame, trim_audio};

// ---------------------------------------------------------------------------
// SingleFile: decode-once fan-out to per-rung MP4 workers
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
pub(super) async fn run_single_file(
    input: Bytes,
    spec: &OutputSpec,
    header: &DemuxHeader,
    frame_rate: f64,
    frames_total: Option<u64>,
    audio: Option<&PreparedAudio>,
    subtitles: Option<&SubtitleTrack>,
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
    if spec.encode_policy.spreads()
        && total_input_frames > 0
        && gpu_pool.capacity() > 1
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
            subtitles,
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
    let decode_gpu = spec.decode_policy.gpu_index().or(encode_gpu);
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
    let pump_cfg = crate::decode_pump::DecodePumpConfig {
        codec_name: header.codec.clone(),
        info_for_decoder: header.info.clone(),
        source_color_metadata: header.info.color_metadata,
        source_pixel_format: header.info.pixel_format,
        needs_downsample: needs_chroma_downsample(header.info.pixel_format),
        tonemap_to_sdr: spec.tonemaps(),
        gpu_index: decode_gpu,
        sample_range: None,
        rotation_degrees: header.rotation_degrees,
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
    let clip = ClipSource { cfg: pump_cfg, input, start_frame, end_frame };
    run_serial_single_file(
        vec![clip],
        spec,
        base_cfg,
        frame_rate,
        effective_total,
        trimmed_audio,
        subtitles.cloned(),
        sink,
    )
    .await
}

/// Serial single-file encode of one or more (pre-trimmed) clips: the spliced
/// decode pump concatenates the clips' kept frames into one continuous stream,
/// and each rung worker encodes that stream into one MP4. Shared by the
/// single-input trim path and `run_splice_job` (multi-clip concat).
pub(super) async fn run_serial_single_file(
    clips: Vec<ClipSource>,
    spec: &OutputSpec,
    base_cfg: EncoderConfig,
    frame_rate: f64,
    effective_total: Option<u64>,
    audio: Option<PreparedAudio>,
    subtitles: Option<SubtitleTrack>,
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
        let subtitles = subtitles.clone();
        let handle = tokio::task::spawn_blocking(move || {
            let r = encode_rung_single_file(
                idx, &rung, rx, base_cfg, backend_override, frame_rate, effective_total,
                audio.as_ref(), subtitles.as_ref(), sink.as_ref(),
            );
            (idx, rung, r)
        });
        handles.push(handle);
    }

    let pump_handle = {
        let rt = rt.clone();
        tokio::task::spawn_blocking(move || {
            decode_pump::run_spliced_decode_pump_blocking(clips, senders, rt)
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
    subtitles: Option<&SubtitleTrack>,
    gpu_pool: Arc<crate::gpu_pool::GpuPool>,
    filter_chain: Arc<codec::filter::FilterChain>,
    sink: Arc<dyn ProgressSink>,
) -> Result<Vec<RungOutput>> {
    let timescale = (frame_rate * 1000.0).round().max(1.0) as u32;
    let per_frame_ticks = (timescale as f64 / frame_rate.max(1.0)).round().max(1.0) as u32;
    // The GOP is the chunk grid: a chunk is a whole number of GOPs, so this is
    // the spec's `gop` when set, else two seconds.
    let keyframe_interval = spec.gop_frames(frame_rate);
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
        decode: spec.decode_policy,
        encode: spec.encode_policy,
        // Chunk workers collect packets in memory; output_root is unused.
        output_root: std::env::temp_dir(),
        timescale,
        per_frame_ticks,
        keyframe_interval,
        segment_target_ticks,
        total_input_frames,
        // ParallelConstQp ⇒ force constant-QP chunks so stitched seams are flat.
        constant_qp: spec.chunk_seam_mode == crate::spec::ChunkSeamMode::ParallelConstQp,
        cancel: None,
    };
    let rung_packets = multigpu::run_multigpu_single_file(params, Arc::clone(&sink)).await?;

    let mut outputs = Vec::new();
    for rp in rung_packets.into_iter().flatten() {
        let label = rp.label.clone();
        match mux_rung_packets_to_mp4(rp, frame_rate, output_color_metadata, audio, subtitles) {
            Ok(out) => outputs.push(out),
            Err(e) => tracing::warn!(rung = %label, error = %e, "stitching rung MP4 failed"),
        }
    }
    if outputs.is_empty() {
        bail!("multi-GPU single-file: no rung produced a stitched MP4");
    }
    Ok(outputs)
}

/// Attach the source's text subtitles to a single-file muxer as a `tx3g`
/// track. A rejection is logged and the rung continues without them — losing
/// subtitles is a worse outcome than failing the whole encode only in theory;
/// in practice a player with no subtitle track still plays.
fn attach_subtitles(
    muxer: &mut Av1Mp4Muxer,
    subtitles: Option<&SubtitleTrack>,
    label: &str,
) {
    let Some(s) = subtitles else { return };
    if let Err(e) = muxer.with_subtitles(&s.cues, s.timescale, &s.language) {
        tracing::warn!(rung = %label, "subtitles rejected ({e}); continuing without them");
    }
}

/// Stitch one rung's ordered AV1 packets (+ optional audio) into an MP4.
fn mux_rung_packets_to_mp4(
    rp: RungPackets,
    frame_rate: f64,
    color_metadata: ColorMetadata,
    audio: Option<&PreparedAudio>,
    subtitles: Option<&SubtitleTrack>,
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
    attach_subtitles(&mut muxer, subtitles, &rp.label);
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
    subtitles: Option<&SubtitleTrack>,
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

    attach_subtitles(&mut muxer, subtitles, &rung.label);

    let mut frames: u64 = 0;
    // Running total of encoded payload so the CLI can show size to date and
    // project a finished size, matching what the chunked path reports.
    let mut bytes_encoded: u64 = 0;
    report(sink, rung_index, rung, RungStatus::Running, 0, frames_total, 0, 0);
    while let Some(frame) = rx.blocking_recv() {
        let scaled = colorspace::scale_frame(&frame, rung.width, rung.height).context("scale_frame")?;
        encoder.send_frame(&scaled).context("send_frame")?;
        while let Some(pkt) = encoder.receive_packet().context("receive_packet")? {
            bytes_encoded += pkt.data.len() as u64;
            muxer.add_packet(pkt).context("add_packet")?;
        }
        frames += 1;
        if frames % 30 == 0 {
            report(sink, rung_index, rung, RungStatus::Running, frames, frames_total, 0, bytes_encoded);
        }
    }
    encoder.flush().context("encoder flush")?;
    while let Some(pkt) = encoder.receive_packet().context("receive_packet drain")? {
        muxer.add_packet(pkt).context("add_packet drain")?;
    }
    report(sink, rung_index, rung, RungStatus::Finalizing, frames, frames_total, 0, bytes_encoded);
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
// Misc helpers local to this file
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
