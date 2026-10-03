//! Audio-only output: the input's audio, alone, as one file — an `.mp3`, an
//! `.m4a`, an Ogg file (Opus or Vorbis), or for FLAC a native `.flac`.
//!
//! [`OutputMode::AudioOnly`] asks for it outright (`mode=audio`); a
//! single-file job whose input has no video (a bare MP3, an M4A, an Ogg, an
//! audio-only Matroska) becomes one, since there is nothing for a ladder.
//! No video is decoded or encoded. The track goes through the same
//! [`prepare_audio`] as any other output, asked for the file's codec: a
//! source already in it passes through, anything else is decoded, laid out
//! and encoded. An `.mp3` is the frames behind an `Info` frame — the
//! encoder's own when the encode was this job's, whose LAME-style extension
//! carries its delay and end padding, so a gapless player presents exactly
//! the source's samples. An Ogg file's granule positions do the same.

use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use bytes::Bytes;

use container::mp3::Gapless;
use container::streaming;

use super::audio::{AudioRequest, PreparedAudio, audio_codec_string, prepare_audio};
use super::{JobOutput, RungArtifact, RungOutput};
use crate::progress::{JobEvent, ProgressSink, RungProgress, RungStatus};
use crate::spec::{Container, OutputMode, OutputSpec};

/// The label of the one output an audio-only job has.
pub const AUDIO_ONLY_LABEL: &str = "audio";

/// When `input` has no video and `spec` is a single-file job: the spec's
/// audio-only form, in the file its codec goes in (an `.ogg` for Opus, an
/// `.m4a` for AAC, …), validated. `None` when the input has video, or no
/// audio this crate reads, or the spec is not single-file — the caller's own
/// error stands then.
pub(super) fn as_audio_only(input: &Bytes, spec: &OutputSpec) -> Option<Result<OutputSpec>> {
    if spec.mode != OutputMode::SingleFile {
        return None;
    }
    match streaming::demux_audio(input.clone()) {
        Ok(Some(src)) if !src.has_video => {}
        _ => return None,
    }
    let audio = OutputSpec {
        audio: spec.audio,
        audio_bitrate: spec.audio_bitrate,
        audio_quality: spec.audio_quality,
        audio_filters: spec.audio_filters.clone(),
        audio_channels: spec.audio_channels,
        audio_bit_depth: spec.audio_bit_depth,
        he_aac: spec.he_aac,
        audio_decode_deny: spec.audio_decode_deny,
        metadata_keep: spec.metadata_keep,
        flac_level: spec.flac_level,
        trim_start: spec.trim_start,
        trim_end: spec.trim_end,
        hooks: spec.hooks.clone(),
        ..OutputSpec::audio_only_in(OutputSpec::audio_only_container(spec.audio))
    };
    Some(audio.validate().map(|()| audio))
}

pub(super) async fn run(
    input: Bytes,
    spec: &OutputSpec,
    sink: Arc<dyn ProgressSink>,
    started: Instant,
) -> Result<JobOutput> {
    spec.validate().context("invalid OutputSpec")?;
    if spec.trim_start.is_some() || spec.trim_end.is_some() {
        bail!("a trim is not available for audio-only output");
    }
    let input_head = input.slice(..input.len().min(64));
    let src = streaming::demux_audio(input.clone())
        .context("demux")?
        .context("the input has no audio track this build reads")?;
    let source_codec = src.track.codec.to_ascii_lowercase();
    spec.hooks.emit_probe(
        0,
        crate::hooks::MediaSummary {
            container: container::sniff_container(&input_head).label().to_string(),
            audio_codec: Some(source_codec.clone()),
            ..Default::default()
        },
    )?;
    sink.on_event(JobEvent::Started { rungs: 1 });
    sink.on_event(JobEvent::Probed {
        codec: "none".into(),
        width: 0,
        height: 0,
        frame_rate: 0.0,
        audio_codec: Some(source_codec.clone()),
    });
    let report = |status, frames: u64, bytes: u64| {
        sink.on_rung(RungProgress {
            status,
            percent: if status == RungStatus::Completed { 100.0 } else { 0.0 },
            frames_done: frames,
            frames_total: None,
            bytes_out: bytes,
            ..RungProgress::pending(0, AUDIO_ONLY_LABEL, 0, 0)
        })
    };
    report(RungStatus::Running, 0, 0);
    if let Some(e) = src.edit
        && e.delay > 0
    {
        tracing::info!(delay = e.delay, "audio-only output: the source's late start has no picture to wait for; dropped");
    }
    let edit = src.edit.map(|e| container::edit::AudioEdit { delay: 0, ..e });
    let prepared = prepare_audio(Some(&src.track), edit, &src.gaps, AudioRequest::of(spec))
        .context("preparing audio")?
        .filter(|a| a.has_samples())
        .with_context(|| format!("the {source_codec} track came out empty; there is no audio to write"))?;
    let codec = prepared.info.codec.to_ascii_lowercase();
    let bytes = match spec.container {
        Container::Mp3 => write_mp3(&prepared)?,
        Container::Flac if codec == "flac" => {
            if !prepared.edit.is_identity() && prepared.edit.duration != Some(total_ticks(&prepared)) {
                // A native stream has no edit list; a copy cut to a source's
                // edit plays to its frame edges.
                tracing::info!(edit = ?prepared.edit, "a native FLAC file has no edit list; it plays whole frames");
            }
            container::mux::write_native_flac(&prepared.info.codec_private, &prepared.samples)
                .context("writing the .flac")?
        }
        Container::M4a => container::mux::write_audio_mp4(&prepared.info, &prepared.samples, prepared.edit)
            .with_context(|| format!("writing {codec} to an .m4a"))?,
        Container::Ogg => container::ogg::write_audio(&prepared.info, &prepared.samples, prepared.edit)
            .with_context(|| format!("writing {codec} to an Ogg file"))?,
        other => bail!(
            "a {other:?} audio-only output cannot hold the audio as it came out: {} ({})",
            prepared.info.codec,
            prepared.handling
        ),
    };
    let packets = prepared.samples.len() as u64;
    let mut rungs = vec![RungOutput {
        label: AUDIO_ONLY_LABEL.into(),
        width: 0,
        height: 0,
        frames: packets,
        bytes: bytes.len() as u64,
        artifact: RungArtifact::File(bytes),
    }];
    super::keep_metadata(&input, spec, &mut rungs)?;
    let nbytes = rungs[0].bytes;
    report(RungStatus::Completed, packets, nbytes);
    sink.on_event(JobEvent::Finished { rungs_completed: 1, rungs_failed: 0 });
    tracing::info!(handling = %prepared.handling, bytes = nbytes, "audio-only output written");
    Ok(JobOutput {
        rungs,
        hls_root: None,
        master_playlist: None,
        source_codec: "none".into(),
        source_dims: (0, 0),
        source_frame_rate: 0.0,
        audio_codecs: Some(audio_codec_string(&prepared.info)),
        audio_handling: prepared.handling,
        renditions: Vec::new(),
        elapsed: started.elapsed(),
        hooks: crate::hooks::HookReport::default(),
    })
}

fn total_ticks(a: &PreparedAudio) -> u64 {
    a.samples.iter().map(|(_, d)| u64::from(*d)).sum()
}

/// The `.mp3` file: the frames behind an `Info` frame.
fn write_mp3(prepared: &PreparedAudio) -> Result<Vec<u8>> {
    if !prepared.info.codec.eq_ignore_ascii_case("mp3") {
        bail!("an .mp3 file holds MP3, and the audio came out as {} ({})", prepared.info.codec, prepared.handling);
    }
    // This job's encode: the encoder's own tag frame, which knows its delay
    // and padding exactly (and names `rivetmp3`).
    if let Some(header) = &prepared.file_header {
        let mut out = Vec::with_capacity(header.len() + prepared.samples.iter().map(|(f, _)| f.len()).sum::<usize>());
        out.extend_from_slice(header);
        for (f, _) in &prepared.samples {
            out.extend_from_slice(f);
        }
        return Ok(out);
    }
    // A passthrough: the source tag's delay and padding (the edit cut to the
    // source's presentation), under its encoder's name. A source that stated
    // none gets none.
    let gapless = prepared.encoder.as_ref().and_then(|_| {
        let delay = prepared.edit.media_time.checked_sub(u64::from(codec::audio::MP3_DECODER_DELAY))?;
        Some(Gapless { encoder_delay: delay as u32, samples: prepared.edit.duration? })
    });
    let frames: Vec<Vec<u8>> = prepared.samples.iter().map(|(f, _)| f.clone()).collect();
    container::mp3::write_file(&frames, gapless, prepared.encoder.as_deref()).context("writing the .mp3")
}
