//! Job progress reporting.
//!
//! Every transcode job reports progress through a [`ProgressSink`] — a small
//! trait that receives a **uniform** [`RungProgress`] struct (status,
//! percentage, frame/segment/byte counters) for each rung as the job runs,
//! plus coarse [`JobEvent`]s for job-level lifecycle.
//!
//! The sink methods are synchronous but are called *as the job progresses*
//! (i.e. asynchronously with respect to completion). To bridge into async
//! code — e.g. forward progress to a websocket/SQS reporter — wrap a
//! `tokio::sync::mpsc::Sender` with [`channel_sink`], or implement
//! [`ProgressSink`] and `try_send` into your own channel.

use std::sync::Arc;

/// Lifecycle status of a single rung (one rendition of the ABR ladder).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RungStatus {
    /// Queued; no frames processed yet.
    Pending,
    /// Actively decoding + scaling + encoding frames.
    Running,
    /// Finalizing the container / writing playlists.
    Finalizing,
    /// Done — `percent == 100`.
    Completed,
    /// Errored out — see [`RungProgress::message`].
    Failed,
}

/// A uniform progress update for one rung. Emitted repeatedly over the life
/// of the job. Consumers can render a per-rung progress bar straight from
/// these fields without knowing anything about the output mode.
#[derive(Debug, Clone)]
pub struct RungProgress {
    /// Index into the job's `rungs` list.
    pub rung_index: usize,
    /// Human label, e.g. `"720p"`.
    pub label: String,
    /// Target width in pixels.
    pub width: u32,
    /// Target height in pixels.
    pub height: u32,
    /// Current status.
    pub status: RungStatus,
    /// Completion fraction in `0.0..=100.0`. Derived from
    /// `frames_done / frames_total` when the total is known, else a coarse
    /// stage estimate.
    pub percent: f32,
    /// Frames encoded so far for this rung.
    pub frames_done: u64,
    /// Total frames expected, if known up front (from the container header).
    pub frames_total: Option<u64>,
    /// Segments written (HLS/CMAF mode only; `0` for single-file).
    pub segments_written: u32,
    /// Output bytes produced so far for this rung.
    pub bytes_out: u64,
    /// Optional human message — error text on `Failed`, notes otherwise.
    pub message: Option<String>,
}

impl RungProgress {
    /// Construct a `Pending` update for a rung at job start.
    pub fn pending(rung_index: usize, label: impl Into<String>, width: u32, height: u32) -> Self {
        Self {
            rung_index,
            label: label.into(),
            width,
            height,
            status: RungStatus::Pending,
            percent: 0.0,
            frames_done: 0,
            frames_total: None,
            segments_written: 0,
            bytes_out: 0,
            message: None,
        }
    }
}

/// Job-level lifecycle events, independent of any single rung.
#[derive(Debug, Clone)]
pub enum JobEvent {
    /// Job accepted; `rungs` renditions will be produced.
    Started { rungs: usize },
    /// Source probed.
    Probed {
        codec: String,
        width: u32,
        height: u32,
        frame_rate: f64,
        audio_codec: Option<String>,
    },
    /// Job finished.
    Finished {
        rungs_completed: usize,
        rungs_failed: usize,
    },
}

/// Receiver for job progress. Implement to consume updates; or use
/// [`channel_sink`] / [`fn_sink`] for the common cases.
pub trait ProgressSink: Send + Sync {
    /// Called with a fresh [`RungProgress`] each time a rung advances.
    fn on_rung(&self, update: RungProgress);
    /// Called for job-level lifecycle events. Default: ignore.
    fn on_event(&self, _event: JobEvent) {}
}

/// A sink that drops every update. Useful as a default.
pub struct NullSink;

impl ProgressSink for NullSink {
    fn on_rung(&self, _update: RungProgress) {}
}

/// Wraps a closure as a [`ProgressSink`].
pub struct FnSink<F>(F);

impl<F: Fn(RungProgress) + Send + Sync> ProgressSink for FnSink<F> {
    fn on_rung(&self, update: RungProgress) {
        (self.0)(update)
    }
}

/// Build a [`ProgressSink`] from a closure: `fn_sink(|p| println!("{}", p.percent))`.
pub fn fn_sink<F: Fn(RungProgress) + Send + Sync>(f: F) -> FnSink<F> {
    FnSink(f)
}

/// A sink that forwards every [`RungProgress`] into a Tokio mpsc channel,
/// turning the callback into an async stream the caller can `.recv().await`.
/// Sends are non-blocking (`try_send`); if the channel is full or closed the
/// update is dropped (progress is advisory, never load-bearing).
pub struct ChannelSink {
    tx: tokio::sync::mpsc::Sender<RungProgress>,
}

impl ChannelSink {
    pub fn new(tx: tokio::sync::mpsc::Sender<RungProgress>) -> Self {
        Self { tx }
    }
}

impl ProgressSink for ChannelSink {
    fn on_rung(&self, update: RungProgress) {
        let _ = self.tx.try_send(update);
    }
}

/// Convenience: wrap an mpsc sender as an `Arc<dyn ProgressSink>`.
pub fn channel_sink(tx: tokio::sync::mpsc::Sender<RungProgress>) -> Arc<dyn ProgressSink> {
    Arc::new(ChannelSink::new(tx))
}
