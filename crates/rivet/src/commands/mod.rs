//! Subcommand implementations for the `rivet` CLI.

pub mod capabilities;
pub mod devices;
pub mod pipe;
pub mod probe;
pub mod progress;
pub mod splice;
pub mod transcode;

#[cfg(feature = "batch")]
pub mod batch;
#[cfg(feature = "ipc")]
pub mod ipc;
#[cfg(feature = "server")]
pub mod serve;

use std::sync::Arc;

use anyhow::{bail, Context, Result};
use rivet::progress::RungProgress;
use rivet::{RungArtifact, TranscodeSettings};

/// Convert a [`rivet::progress::RungStatus`] to a short display label.
pub(crate) fn status_str(s: rivet::progress::RungStatus) -> &'static str {
    match s {
        rivet::progress::RungStatus::Pending => "pend",
        rivet::progress::RungStatus::Running => "run",
        rivet::progress::RungStatus::Finalizing => "final",
        rivet::progress::RungStatus::Completed => "done",
        rivet::progress::RungStatus::Failed => "FAIL",
    }
}

/// JSON-escape a bare string value (no surrounding quotes).
pub(crate) fn esc(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

/// Transcode `input` honouring `settings`; returns `(mp4_bytes, frame_count, audio_label)`.
///
/// All-default settings take the fast [`rivet::transcode_bytes`] path; any set field
/// routes through [`rivet::TranscodeSettings::into_spec`] + the full `run_job` engine.
pub(crate) fn stream_transcode(
    input: &[u8],
    settings: &TranscodeSettings,
) -> Result<(Vec<u8>, u64, String)> {
    if settings.is_empty() {
        let out = rivet::transcode_bytes(input).context("transcoding")?;
        return Ok((
            out.output_bytes,
            out.frames_processed,
            out.audio_handling.label(),
        ));
    }
    let probed = rivet::probe_bytes(input).context("probing input")?;
    let spec = settings
        .clone()
        .into_spec(probed.width, probed.height)
        .context("invalid settings")?;
    if matches!(spec.mode, rivet::OutputMode::Hls { .. }) {
        bail!(
            "HLS/segmented output isn't supported over pipe/ipc (a single stream) — \
             use `rivet transcode -o <dir>` or the HTTP API"
        );
    }
    let sink = Arc::new(rivet::fn_sink(|_p: RungProgress| {}));
    let out = rivet::run_job_blocking(input, &spec, None, sink).context("transcoding")?;
    let audio = out.audio_handling.clone();
    for r in out.rungs {
        let frames = r.frames;
        if let RungArtifact::File(bytes) = r.artifact {
            return Ok((bytes, frames, audio));
        }
    }
    bail!("no single-file output produced")
}
