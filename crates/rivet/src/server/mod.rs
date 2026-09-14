//! HTTP transcode API (`rivet serve`, behind the `server` feature).
//!
//! A small [axum] webserver so another application can **signal rivet to
//! transcode** something over the network: it POSTs media bytes plus an output
//! spec, rivet runs the job on the same configurable engine the CLI uses, and
//! reports progress + serves the output artifacts.
//!
//! Endpoints (all under `/v1`):
//! - `GET  /v1/health` — liveness + detected GPUs + build capabilities.
//! - `POST /v1/probe` — body = media bytes → JSON [`MediaInfo`](crate::probe::MediaInfo).
//! - `POST /v1/transcode` — body = media bytes, spec from query params. Returns
//!   `202 { job_id }` and runs asynchronously; pass `?sync=true` to block and
//!   get the (single-file, single-rung) MP4 back directly.
//! - `GET  /v1/jobs/{id}` — job status + per-rung progress + output list.
//! - `GET  /v1/jobs/{id}/artifacts/{label}` — download a single-file rung's MP4.
//! - `GET  /v1/jobs/{id}/files/{*path}` — fetch a file from an HLS job's output
//!   tree (e.g. `master.m3u8`, `video/720p/seg-00001.m4s`).
//!
//! The job registry is in-memory; completed single-file artifacts are held in
//! RAM until the process exits (fine for a sidecar/worker, not a public CDN —
//! a production deployment would offload to object storage from a `ProgressSink`
//! watching `RungStatus::Completed`).

use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, RwLock};

use anyhow::{Context, Result};
use axum::Router;
use axum::body::Bytes;
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::extract::DefaultBodyLimit;
use serde_json::{Value, json};
use uuid::Uuid;

use crate::progress::{ProgressSink, RungProgress, RungStatus};

mod handlers;
mod spec;
mod docs;
#[cfg(test)]
mod tests;

// Re-export the public items so `rivet::server::X` paths resolve.
pub use docs::openapi_spec;

/// 4 GiB upload ceiling — large enough for long source files.
pub(super) const MAX_UPLOAD: usize = 4 * 1024 * 1024 * 1024;

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct AppState {
    pub(super) jobs: Arc<RwLock<HashMap<Uuid, Arc<JobHandle>>>>,
}

impl AppState {
    fn new() -> Self {
        Self {
            jobs: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Phase {
    Queued,
    Running,
    Completed,
    Failed,
}

impl Phase {
    pub(super) fn as_str(self) -> &'static str {
        match self {
            Phase::Queued => "queued",
            Phase::Running => "running",
            Phase::Completed => "completed",
            Phase::Failed => "failed",
        }
    }
}

pub(super) struct ArtifactEntry {
    pub(super) label: String,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) frames: u64,
    pub(super) bytes: u64,
    /// In-memory MP4 bytes for a single-file rung held in RAM; `None` for an
    /// HLS rendition or when the bytes were written to `output_path`.
    pub(super) data: Option<Bytes>,
    /// Server-side path the artifact was written to (when the request supplied
    /// `output.path`); surfaced in the status JSON.
    pub(super) output_path: Option<String>,
}

pub(super) struct JobHandle {
    pub(super) id: Uuid,
    mode: String,
    pub(super) phase: Mutex<Phase>,
    progress: Mutex<Vec<RungProgress>>,
    pub(super) artifacts: Mutex<Vec<ArtifactEntry>>,
    pub(super) error: Mutex<Option<String>>,
    /// HLS output root (a temp dir), if any.
    pub(super) output_dir: Mutex<Option<PathBuf>>,
    pub(super) master_playlist: Mutex<Option<String>>,
}

impl JobHandle {
    pub(super) fn new(id: Uuid, mode: &str) -> Self {
        Self {
            id,
            mode: mode.to_string(),
            phase: Mutex::new(Phase::Queued),
            progress: Mutex::new(Vec::new()),
            artifacts: Mutex::new(Vec::new()),
            error: Mutex::new(None),
            output_dir: Mutex::new(None),
            master_playlist: Mutex::new(None),
        }
    }

    pub(super) fn set_phase(&self, p: Phase) {
        *self.phase.lock().unwrap() = p;
    }

    pub(super) fn status_json(&self) -> Value {
        let phase = *self.phase.lock().unwrap();
        let progress: Vec<Value> = self
            .progress
            .lock()
            .unwrap()
            .iter()
            .map(rung_progress_json)
            .collect();
        let artifacts: Vec<Value> = self
            .artifacts
            .lock()
            .unwrap()
            .iter()
            .map(|a| {
                // Download URL only when bytes are held in RAM; when written to
                // disk (`output_path`) the caller already has the path.
                let url = if a.data.is_some() {
                    Some(format!("/v1/jobs/{}/artifacts/{}", self.id, a.label))
                } else if a.output_path.is_none() {
                    Some(format!("/v1/jobs/{}/files/", self.id))
                } else {
                    None
                };
                json!({
                    "label": a.label,
                    "width": a.width,
                    "height": a.height,
                    "frames": a.frames,
                    "bytes": a.bytes,
                    "url": url,
                    "output_path": a.output_path,
                })
            })
            .collect();
        json!({
            "job_id": self.id.to_string(),
            "mode": self.mode,
            "status": phase.as_str(),
            "progress": progress,
            "artifacts": artifacts,
            "master_playlist": *self.master_playlist.lock().unwrap(),
            "error": *self.error.lock().unwrap(),
        })
    }
}

fn rung_progress_json(p: &RungProgress) -> Value {
    json!({
        "rung_index": p.rung_index,
        "label": p.label,
        "width": p.width,
        "height": p.height,
        "status": rung_status_str(p.status),
        "percent": p.percent,
        "frames_done": p.frames_done,
        // Why a failed rung failed, its whole error chain; null otherwise.
        "message": p.message,
    })
}

fn rung_status_str(s: RungStatus) -> &'static str {
    match s {
        RungStatus::Pending => "pending",
        RungStatus::Running => "running",
        RungStatus::Finalizing => "finalizing",
        RungStatus::Completed => "completed",
        RungStatus::Failed => "failed",
    }
}

/// A [`ProgressSink`] that mirrors per-rung updates into a [`JobHandle`].
pub(super) struct RegistrySink {
    pub(super) handle: Arc<JobHandle>,
}

impl ProgressSink for RegistrySink {
    fn on_rung(&self, update: RungProgress) {
        let mut prog = self.handle.progress.lock().unwrap();
        match prog.iter_mut().find(|p| p.rung_index == update.rung_index) {
            Some(slot) => *slot = update,
            None => prog.push(update),
        }
    }
}

// ---------------------------------------------------------------------------
// Response helpers (shared across handlers)
// ---------------------------------------------------------------------------

/// JSON response wrapper (so handlers can return `Json`).
pub(super) struct Json(pub(super) Value);

impl IntoResponse for Json {
    fn into_response(self) -> Response {
        (
            [(header::CONTENT_TYPE, "application/json")],
            serde_json::to_vec(&self.0).unwrap_or_default(),
        )
            .into_response()
    }
}

/// A JSON error with an HTTP status.
pub(super) struct ApiError {
    pub(super) status: StatusCode,
    pub(super) message: String,
}

impl ApiError {
    pub(super) fn bad_request(e: anyhow::Error) -> Self {
        Self { status: StatusCode::BAD_REQUEST, message: format!("{e:#}") }
    }
    pub(super) fn internal(e: anyhow::Error) -> Self {
        Self { status: StatusCode::INTERNAL_SERVER_ERROR, message: format!("{e:#}") }
    }
    pub(super) fn not_found(what: String) -> Self {
        Self { status: StatusCode::NOT_FOUND, message: format!("{what} not found") }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (
            self.status,
            [(header::CONTENT_TYPE, "application/json")],
            serde_json::to_vec(&json!({ "error": self.message })).unwrap_or_default(),
        )
            .into_response()
    }
}

// ---------------------------------------------------------------------------
// Router entry points
// ---------------------------------------------------------------------------

/// Build the axum router (also the test entry point).
pub fn build_router() -> Router {
    let state = AppState::new();
    Router::new()
        .route("/", get(handlers::landing))
        .route("/openapi.json", get(handlers::openapi_json))
        .route("/swagger", get(handlers::swagger_ui))
        .route("/redoc", get(handlers::redoc_ui))
        .route("/v1/health", get(handlers::health))
        .route("/v1/probe", post(handlers::probe))
        .route("/v1/transcode", post(handlers::transcode))
        .route("/v1/jobs/{id}", get(handlers::job_status))
        .route("/v1/jobs/{id}/artifacts/{label}", get(handlers::artifact))
        .route("/v1/jobs/{id}/files/{*path}", get(handlers::hls_file))
        .layer(DefaultBodyLimit::max(MAX_UPLOAD))
        .with_state(state)
}

/// Run the server, blocking until shutdown.
pub async fn serve(addr: SocketAddr) -> Result<()> {
    let app = build_router();
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .with_context(|| format!("binding {addr}"))?;
    tracing::info!(%addr, "rivet transcode API listening");
    axum::serve(listener, app).await.context("axum serve")?;
    Ok(())
}
