//! Audio-only output: the input's audio, alone, as one file — an `.mp3`,
//! or for lossless audio a native `.flac` or an `.m4a`.
//!
//! [`OutputMode::AudioOnly`] asks for it outright (`mode=audio`); a
//! single-file job whose input has no video (a bare MP3, an M4A, an
//! audio-only Matroska) becomes one, since there is nothing for a ladder.
//! No video is decoded or encoded. The track goes through the same
//! [`prepare_audio`] as any other output, asked for MP3: an MP3 source
//! passes through, anything else is decoded, laid out (mono or stereo) and
//! encoded. The file is the frames behind an `Info` frame, whose LAME
//! extension carries the encoder delay and end padding when the encode was
//! this job's, so a gapless player presents exactly the source's samples.

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
/// audio-only form, validated (`audio=opus` has no audio-only form, and says
/// so). `None` when the input has video, or no audio this crate reads, or
/// the spec is not single-file — the caller's own error stands then.
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
        audio_filters: spec.audio_filters.clone(),
        audio_channels: spec.audio_channels,
        audio_bit_depth: spec.audio_bit_depth,
        flac_level: spec.flac_level,
        trim_start: spec.trim_start,
        trim_end: spec.trim_end,
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
    let src = streaming::demux_audio(input)
        .context("demux")?
        .context("the input has no audio track this build reads")?;
    let source_codec = src.track.codec.to_ascii_lowercase();
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
        other => bail!(
            "a {other:?} audio-only output cannot hold the audio as it came out: {} ({})",
            prepared.info.codec,
            prepared.handling
        ),
    };
    let packets = prepared.samples.len() as u64;
    let nbytes = bytes.len() as u64;
    report(RungStatus::Completed, packets, nbytes);
    sink.on_event(JobEvent::Finished { rungs_completed: 1, rungs_failed: 0 });
    tracing::info!(handling = %prepared.handling, bytes = nbytes, "audio-only output written");
    Ok(JobOutput {
        rungs: vec![RungOutput {
            label: AUDIO_ONLY_LABEL.into(),
            width: 0,
            height: 0,
            frames: packets,
            bytes: nbytes,
            artifact: RungArtifact::File(bytes),
        }],
        hls_root: None,
        master_playlist: None,
        source_codec: "none".into(),
        source_dims: (0, 0),
        source_frame_rate: 0.0,
        audio_codecs: Some(audio_codec_string(&prepared.info)),
        audio_handling: prepared.handling,
        elapsed: started.elapsed(),
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
    // The LAME extension's delay and padding: this job's encode's, or a
    // passthrough's from its source's tag (the edit cut to the source's
    // presentation). A source that stated none gets none.
    let gapless = prepared.encoder.as_ref().and_then(|_| {
        let delay = prepared.edit.media_time.checked_sub(u64::from(codec::audio::MP3_DECODER_DELAY))?;
        Some(Gapless { encoder_delay: delay as u32, samples: prepared.edit.duration? })
    });
    let frames: Vec<Vec<u8>> = prepared.samples.iter().map(|(f, _)| f.clone()).collect();
    container::mp3::write_file(&frames, gapless, prepared.encoder.as_deref()).context("writing the .mp3")
}
