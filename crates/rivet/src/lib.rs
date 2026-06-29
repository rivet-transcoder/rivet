//! # rivet
//!
//! A modular, GPU-accelerated video transcoding library.
//!
//! `rivet` bundles two lower-level crates — [`codec`] (decode / encode /
//! colorspace / probe) and [`container`] (demux / mux / CMAF / HLS) — behind
//! a small, ergonomic facade for the common case: take an arbitrary input
//! file and produce a single AV1 + Opus MP4.
//!
//! ## Output policy
//!
//! The output codec is **AV1** (video) + **Opus / AAC passthrough** (audio)
//! muxed into **MP4**. This is a deliberate, royalty-clean target — see the
//! project README. Input may be any container/codec the [`container`] and
//! [`codec`] crates can demux + decode (H.264, HEVC, VP8/VP9, AV1, MPEG-2,
//! MPEG-4, ProRes; MP4/MOV/MKV/WebM/MPEG-TS/AVI).
//!
//! ## Quick start
//!
//! ```no_run
//! // Transcode a file on disk to an AV1/Opus MP4.
//! let outcome = rivet::transcode_file("input.mkv", "output.mp4")?;
//! println!("{} frames, {} bytes out", outcome.frames_processed, outcome.output_bytes.len());
//!
//! // Or inspect an input without transcoding it.
//! let info = rivet::probe_file("input.mkv")?;
//! println!("{}x{} {} @ {:.3} fps", info.width, info.height, info.video_codec, info.frame_rate);
//! # Ok::<(), anyhow::Error>(())
//! ```
//!
//! ## Lower-level access
//!
//! The component crates are re-exported for callers that need finer control
//! than the facade offers (custom encoder configs, segment-level CMAF
//! output, per-frame access, etc.):
//!
//! ```
//! use rivet::codec::encode::EncoderConfig;
//! use rivet::container::mux::Av1Mp4Muxer;
//! ```

pub mod cmaf_util;
pub mod decode_pump;
pub mod encoder_worker;
pub mod frame_queue;
pub mod gpu_pool;
pub mod job;
pub mod ladder;
/// Batch manifest DSL (YAML/JSON), opt-in `batch` feature.
#[cfg(feature = "batch")]
pub mod manifest;
pub mod multigpu;
pub mod probe;
pub mod progress;
pub mod rung_scaler;
/// HTTP transcode API (opt-in `server` feature).
#[cfg(feature = "server")]
pub mod server;
pub mod settings;
pub mod spec;
#[cfg(feature = "thumbnail")]
pub mod thumbnail;
pub mod transcode;
pub mod validate;

// Re-export the component crates so downstream consumers can depend on a
// single `rivet` crate and still reach the full lower-level API.
pub use codec;
pub use container;

// Flatten the most common entry points to the crate root.
pub use gpu_pool::{GpuLease, GpuPool};
pub use job::{
    Clip, JobOutput, RungArtifact, RungOutput, run_job, run_job_blocking, run_splice_job,
    run_splice_job_blocking,
};
pub use ladder::standard_ladder;
pub use multigpu::{MultiGpuParams, RungManifest, detect_gpu_pool, run_multigpu_hls};
pub use probe::{AudioStreamInfo, MediaInfo, probe_bytes, probe_file};
pub use progress::{JobEvent, ProgressSink, RungProgress, RungStatus, channel_sink, fn_sink};
#[cfg(feature = "batch")]
pub use manifest::{
    BatchReport, Format as ManifestFormat, JobOutcome, JobStatus, Manifest, run_manifest_file,
};
pub use settings::{Mode, TranscodeSettings};
pub use spec::{
    AudioCodecPolicy, BitDepth, ColorPolicy, Container, EncodePolicy, GpuFamily, Muxer, OutputMode,
    OutputSpec, Quality, Rung, VideoCodec, VideoCodecPolicy,
};
#[allow(deprecated)]
pub use spec::AudioPolicy;
pub use transcode::{AudioHandling, TranscodeOutcome, transcode_bytes, transcode_file};
