//! Writing hooks: each kind hooks into one specific point of a job.
//!
//! ```text
//! cargo run --example hooks -- input.mp4 output.mp4
//! ```
//!
//! Registers source-material hashing (a digest of the source bytes and a
//! perceptual fingerprint of the decoded source frames), a probe gate, an
//! integration on the decoded frames, an artifact digest of the video output,
//! and a completed hook — then runs a single-file transcode and prints what
//! the hooks recorded. Each is a different trait, registered with its own
//! method; the same `Hooks` value can be handed to the HTTP server with
//! `rivet::server::serve_with_hooks` (the `server` feature).

use std::sync::Arc;

use anyhow::{Context, Result};
use rivet::hooks::{
    ArtifactDigest, ArtifactKind, CompletedEvent, CompletedHook, DecodedFrameHook, DigestAlgorithm, FrameEvent,
    FrameSampling, HookContext, HookOutcome, HookPolicy, Hooks, PerceptualAlgorithm, PerceptualFingerprint,
    ProbeEvent, ProbeHook, SourceDigest,
};

/// A probe hook: refuses sources larger than it accepts, before any decoding.
struct SizeGate {
    max_pixels: u64,
}

impl ProbeHook for SizeGate {
    fn on_probe(&self, _ctx: &HookContext, probe: &ProbeEvent) -> Result<HookOutcome> {
        let m = &probe.media;
        Ok(if u64::from(m.width) * u64::from(m.height) > self.max_pixels {
            HookOutcome::reject(format!("{}x{} is over the limit", m.width, m.height))
        } else {
            HookOutcome::proceed()
        })
    }

    fn describe(&self) -> String {
        format!("size gate ({} pixels)", self.max_pixels)
    }
}

/// A decoded-frame hook: where an integration would hand source frames to its
/// own system. Here it records each sampled frame's mean luma.
struct FrameIntegration;

impl DecodedFrameHook for FrameIntegration {
    fn sampling(&self) -> FrameSampling {
        FrameSampling::every_seconds(2.0).max_frames(50)
    }

    fn on_decoded_frame(&self, _ctx: &HookContext, frame: &FrameEvent) -> Result<HookOutcome> {
        let luma = rivet::hooks::frame::luma8(&frame.frame)?;
        let mean = luma.iter().map(|&v| u64::from(v)).sum::<u64>() / luma.len().max(1) as u64;
        Ok(HookOutcome::proceed().annotate("mean_luma", mean))
    }

    fn describe(&self) -> String {
        "example frame integration".into()
    }
}

/// A completed hook: the job's last gate, and a place to announce it.
struct Announce;

impl CompletedHook for Announce {
    fn on_completed(&self, ctx: &HookContext, done: &CompletedEvent) -> Result<HookOutcome> {
        eprintln!("job {} made {} artifact(s) in {:?}", ctx.job_id, done.artifacts, done.elapsed);
        Ok(HookOutcome::proceed())
    }
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let input = args.next().context("usage: hooks <input> <output.mp4>")?;
    let output = args.next().context("usage: hooks <input> <output.mp4>")?;

    let hooks = Hooks::new()
        // Source-material hashing: the bytes, and the decoded source frames.
        .source("source-digest", SourceDigest::new(&DigestAlgorithm::ALL))
        .decoded_frames(
            "source-fingerprint",
            PerceptualFingerprint::new(&[PerceptualAlgorithm::PHash, PerceptualAlgorithm::DHash])
                .sampling(FrameSampling::every_seconds(1.0)),
        )
        .probe("size-gate", SizeGate { max_pixels: 8192 * 4320 })
        // Background: on the session's worker thread, off the decode path.
        .decoded_frames_with("frame-integration", FrameIntegration, HookPolicy::background())
        .artifacts("output-digest", ArtifactDigest::new(&[DigestAlgorithm::Sha256]).kinds(&[ArtifactKind::Video]))
        .completed("announce", Announce);
    // A session the caller keeps, so the report is readable even if a hook
    // rejects the job.
    let session = hooks.session("example-job", rivet::hooks::JobKind::Transcode);

    let data = bytes::Bytes::from(std::fs::read(&input).with_context(|| format!("reading {input}"))?);
    let info = rivet::probe_bytes(&data)?;
    let spec = rivet::OutputSpec::single_file(vec![rivet::Rung::new(info.width, info.height)]).with_hooks(session.clone());
    let result = rivet::run_job_blocking_owned(data, &spec, None, Arc::new(rivet::progress::NullSink));

    println!("{}", serde_json::to_string_pretty(&session.report().to_json())?);
    let out = match result {
        Ok(out) => out,
        Err(e) => match rivet::hooks::rejection_of(&e) {
            Some(r) => anyhow::bail!("{r}"),
            None => return Err(e),
        },
    };
    if let Some(rivet::RungArtifact::File(bytes)) = out.rungs.first().map(|r| &r.artifact) {
        std::fs::write(&output, bytes).with_context(|| format!("writing {output}"))?;
    }
    Ok(())
}
