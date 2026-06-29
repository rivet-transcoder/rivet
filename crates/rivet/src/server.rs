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
use axum::extract::{DefaultBodyLimit, Path as AxPath, Query, State};
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, post};
use serde::Deserialize;
use serde_json::{Value, json};
use uuid::Uuid;

use crate::progress::{ProgressSink, RungProgress, RungStatus};
use crate::settings::TranscodeSettings;
use crate::spec::OutputSpec;

/// 4 GiB upload ceiling — large enough for long source files.
const MAX_UPLOAD: usize = 4 * 1024 * 1024 * 1024;

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct AppState {
    jobs: Arc<RwLock<HashMap<Uuid, Arc<JobHandle>>>>,
}

impl AppState {
    fn new() -> Self {
        Self {
            jobs: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    Queued,
    Running,
    Completed,
    Failed,
}

impl Phase {
    fn as_str(self) -> &'static str {
        match self {
            Phase::Queued => "queued",
            Phase::Running => "running",
            Phase::Completed => "completed",
            Phase::Failed => "failed",
        }
    }
}

struct ArtifactEntry {
    label: String,
    width: u32,
    height: u32,
    frames: u64,
    bytes: u64,
    /// In-memory MP4 bytes for a single-file rung held in RAM; `None` for an
    /// HLS rendition or when the bytes were written to `output_path`.
    data: Option<Bytes>,
    /// Server-side path the artifact was written to (when the request supplied
    /// `output.path`); surfaced in the status JSON.
    output_path: Option<String>,
}

struct JobHandle {
    id: Uuid,
    mode: String,
    phase: Mutex<Phase>,
    progress: Mutex<Vec<RungProgress>>,
    artifacts: Mutex<Vec<ArtifactEntry>>,
    error: Mutex<Option<String>>,
    /// HLS output root (a temp dir), if any.
    output_dir: Mutex<Option<PathBuf>>,
    master_playlist: Mutex<Option<String>>,
}

impl JobHandle {
    fn new(id: Uuid, mode: &str) -> Self {
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

    fn set_phase(&self, p: Phase) {
        *self.phase.lock().unwrap() = p;
    }

    fn status_json(&self) -> Value {
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
struct RegistrySink {
    handle: Arc<JobHandle>,
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
// Request → OutputSpec
// ---------------------------------------------------------------------------

#[derive(Deserialize, Default, Clone)]
struct TranscodeParams {
    /// `single` (default) or `hls`.
    mode: Option<String>,
    /// Output video codec: `av1` (default), `h264`, or `h265`.
    codec: Option<String>,
    /// Comma-separated `WxH` list, e.g. `1280x720,640x360`. Omit to use the
    /// source resolution (or set `ladder=true`).
    rungs: Option<String>,
    /// Derive a standard ABR ladder from the source instead of explicit rungs.
    ladder: Option<bool>,
    max_short_side: Option<u32>,
    segment_seconds: Option<f32>,
    crf: Option<u8>,
    speed: Option<u8>,
    /// `auto` (default), `opus`, or `drop`.
    audio: Option<String>,
    /// `sdr` (default), `hdr10`, `hlg`, or `passthrough`.
    color: Option<String>,
    /// `auto` (default), `8bit`, or `10bit`.
    pixel_format: Option<String>,
    /// Multi-GPU single-file chunk seam handling: `parallel` (default),
    /// `constqp`, or `serial`.
    seam: Option<String>,
    max_fps: Option<f64>,
    gpu: Option<u32>,
    /// Video filter chain, e.g. `crop=1280:720,hflip`.
    filter: Option<String>,
    /// Block until the job finishes and return the artifact directly.
    sync: Option<bool>,
}

// ---------------------------------------------------------------------------
// Structured JSON request body
// ---------------------------------------------------------------------------

/// A `POST /v1/transcode` body sent as `application/json`. The spec is a
/// structured object (not query params); the media comes from a server-side
/// **file path** or **inline base64** instead of a streamed binary body, and
/// the output can be written to a server **file path** instead of held in RAM.
#[derive(Deserialize)]
struct TranscodeRequest {
    /// Where to read the input media from (`path` or `base64`).
    input: InputSource,
    /// Optional: write the result to a server path instead of keeping it in
    /// memory. A file for single-rung single-file; a directory for multi-rung
    /// or HLS.
    #[serde(default)]
    output: Option<OutputTarget>,
    /// The output spec (structured form of the query params).
    #[serde(default)]
    spec: SpecBody,
    /// Block until the job finishes (stream/summarize the result) instead of
    /// returning a job id immediately.
    #[serde(default)]
    sync: bool,
}

/// The media source for a JSON request: exactly one of `path` / `base64`.
#[derive(Deserialize)]
struct InputSource {
    /// A file path on the **server** to read the media from.
    #[serde(default)]
    path: Option<String>,
    /// The media inline, base64-encoded (standard alphabet).
    #[serde(default)]
    base64: Option<String>,
}

/// Where to write the result of a JSON request.
#[derive(Deserialize)]
struct OutputTarget {
    /// A file path (single-file single-rung) or directory (multi-rung / HLS)
    /// on the **server**.
    path: String,
}

/// The structured spec body (mirrors [`TranscodeParams`] but with `rungs` as a
/// real array). Converts into `TranscodeParams` so it reuses [`build_spec`].
#[derive(Deserialize, Default)]
struct SpecBody {
    mode: Option<String>,
    /// Output video codec: `av1` (default), `h264`, or `h265`.
    codec: Option<String>,
    /// Explicit rungs as `["1280x720", "640x360"]`.
    #[serde(default)]
    rungs: Vec<String>,
    ladder: Option<bool>,
    max_short_side: Option<u32>,
    segment_seconds: Option<f32>,
    crf: Option<u8>,
    speed: Option<u8>,
    audio: Option<String>,
    color: Option<String>,
    /// `auto` | `8bit` | `10bit` (accepts the legacy key `pixel_format` too).
    #[serde(alias = "pixel_format")]
    bit_depth: Option<String>,
    seam: Option<String>,
    max_fps: Option<f64>,
    gpu: Option<u32>,
    /// Video filters — a chain string (`"crop=1280:720,hflip"`) or a structured
    /// list of objects (`[{"crop":{"w":1280,"h":720}},"hflip"]`).
    filter: Option<codec::filter::FilterSpec>,
}

impl SpecBody {
    fn into_params(self) -> TranscodeParams {
        TranscodeParams {
            mode: self.mode,
            codec: self.codec,
            rungs: (!self.rungs.is_empty()).then(|| self.rungs.join(",")),
            ladder: self.ladder,
            max_short_side: self.max_short_side,
            segment_seconds: self.segment_seconds,
            crf: self.crf,
            speed: self.speed,
            audio: self.audio,
            color: self.color,
            pixel_format: self.bit_depth,
            seam: self.seam,
            max_fps: self.max_fps,
            gpu: self.gpu,
            // Collapse the structured-or-string FilterSpec to the chain string
            // (TranscodeParams is the string-keyed query form; into_settings
            // re-parses it). Round-trips losslessly via Display.
            filter: self.filter.map(|f| f.to_chain()),
            sync: None,
        }
    }
}

/// Read the media for a JSON request from its `path` or `base64` field.
fn read_input(src: &InputSource) -> Result<Bytes, ApiError> {
    match (&src.path, &src.base64) {
        (Some(p), None) => {
            let path = resolve_path(p, true)?;
            let bytes = std::fs::read(&path)
                .map_err(|e| ApiError::bad_request(anyhow::anyhow!("reading input {p}: {e}")))?;
            Ok(Bytes::from(bytes))
        }
        (None, Some(b)) => {
            let bytes = base64_decode(b.trim())
                .map_err(|e| ApiError::bad_request(anyhow::anyhow!("input.base64: {e}")))?;
            Ok(Bytes::from(bytes))
        }
        (Some(_), Some(_)) => Err(ApiError::bad_request(anyhow::anyhow!(
            "input: set exactly one of `path` or `base64`"
        ))),
        (None, None) => Err(ApiError::bad_request(anyhow::anyhow!(
            "input: set `path` or `base64`"
        ))),
    }
}

/// Resolve a request-supplied file path. When `RIVET_FILE_ROOT` is set, the
/// path must canonicalize **under** that root (sandbox); otherwise any path is
/// allowed (the server binds localhost by default — treat it as trusted-local).
/// `must_exist` requires an existing file (input); else only the parent dir
/// must exist (output).
fn resolve_path(p: &str, must_exist: bool) -> Result<PathBuf, ApiError> {
    let path = PathBuf::from(p);
    let root = std::env::var_os("RIVET_FILE_ROOT").map(PathBuf::from);

    let resolved = if must_exist {
        std::fs::canonicalize(&path)
            .map_err(|_| ApiError::bad_request(anyhow::anyhow!("input path not found: {p}")))?
    } else {
        let parent = path.parent().filter(|s| !s.as_os_str().is_empty());
        let file = path
            .file_name()
            .ok_or_else(|| ApiError::bad_request(anyhow::anyhow!("invalid output path: {p}")))?;
        let cparent = match parent {
            Some(par) => std::fs::canonicalize(par).map_err(|_| {
                ApiError::bad_request(anyhow::anyhow!("output directory not found: {}", par.display()))
            })?,
            None => std::env::current_dir()
                .map_err(|e| ApiError::internal(anyhow::anyhow!("cwd: {e}")))?,
        };
        cparent.join(file)
    };

    if let Some(root) = root {
        let croot = std::fs::canonicalize(&root).unwrap_or(root);
        if !resolved.starts_with(&croot) {
            return Err(ApiError::bad_request(anyhow::anyhow!(
                "path escapes RIVET_FILE_ROOT sandbox"
            )));
        }
    }
    Ok(resolved)
}

/// Minimal standard-alphabet base64 decoder (no padding required). Avoids a
/// dependency for the JSON `input.base64` convenience.
fn base64_decode(s: &str) -> Result<Vec<u8>> {
    fn val(c: u8) -> Option<u8> {
        match c {
            b'A'..=b'Z' => Some(c - b'A'),
            b'a'..=b'z' => Some(c - b'a' + 26),
            b'0'..=b'9' => Some(c - b'0' + 52),
            b'+' => Some(62),
            b'/' => Some(63),
            _ => None,
        }
    }
    let mut out = Vec::with_capacity(s.len() / 4 * 3);
    let mut acc: u32 = 0;
    let mut bits = 0u32;
    for &c in s.as_bytes() {
        if c == b'=' || c.is_ascii_whitespace() {
            continue;
        }
        let v = val(c).context("invalid base64 character")? as u32;
        acc = (acc << 6) | v;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
        }
    }
    Ok(out)
}

impl TranscodeParams {
    /// Map the (string) query/JSON params onto the canonical
    /// [`TranscodeSettings`] using the shared `settings::parse_*` vocabulary —
    /// so the API doesn't carry its own copy of the field/spec logic.
    fn into_settings(&self) -> Result<TranscodeSettings> {
        use crate::settings::{
            parse_audio, parse_bit_depth, parse_color, parse_mode, parse_rung, parse_seam,
            parse_video_codec,
        };
        let mut s = TranscodeSettings::default();
        if let Some(m) = &self.mode {
            s.mode = Some(parse_mode(m)?);
        }
        if let Some(c) = &self.codec {
            s.video_codec = Some(parse_video_codec(c)?);
        }
        if let Some(r) = &self.rungs {
            for part in r.split(',').map(str::trim).filter(|p| !p.is_empty()) {
                s.rungs.push(parse_rung(part)?);
            }
        }
        s.ladder = self.ladder.unwrap_or(false);
        s.max_short_side = self.max_short_side;
        s.segment_seconds = self.segment_seconds;
        s.crf = self.crf;
        s.speed = self.speed;
        if let Some(a) = &self.audio {
            s.audio = Some(parse_audio(a)?);
        }
        if let Some(c) = &self.color {
            s.color = Some(parse_color(c)?);
        }
        if let Some(p) = &self.pixel_format {
            s.bit_depth = Some(parse_bit_depth(p)?);
        }
        if let Some(sm) = &self.seam {
            s.seam = Some(parse_seam(sm)?);
        }
        s.max_fps = self.max_fps;
        s.gpu = self.gpu;
        if let Some(f) = &self.filter {
            s.filters = codec::filter::parse_chain(f).context("parsing filter")?;
        }
        Ok(s)
    }
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

/// Build the axum router (also the test entry point).
pub fn build_router() -> Router {
    let state = AppState::new();
    Router::new()
        .route("/", get(landing))
        .route("/openapi.json", get(openapi_json))
        .route("/swagger", get(swagger_ui))
        .route("/redoc", get(redoc_ui))
        .route("/v1/health", get(health))
        .route("/v1/probe", post(probe))
        .route("/v1/transcode", post(transcode))
        .route("/v1/jobs/{id}", get(job_status))
        .route("/v1/jobs/{id}/artifacts/{label}", get(artifact))
        .route("/v1/jobs/{id}/files/{*path}", get(hls_file))
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

async fn health() -> Json {
    let gpus: Vec<Value> = codec::gpu::detect_gpus()
        .into_iter()
        .map(|g| json!({ "index": g.index, "vendor": format!("{:?}", g.vendor), "name": g.name }))
        .collect();
    let caps = codec::encode::build_output_caps();
    Json(json!({
        "status": "ok",
        "service": "rivet",
        "gpus": gpus,
        "output_caps": { "max_bit_depth": caps.max_bit_depth, "hdr": caps.hdr },
    }))
}

async fn probe(body: Bytes) -> Result<Json, ApiError> {
    let info = crate::probe::probe_bytes(&body).map_err(ApiError::bad_request)?;
    Ok(Json(json!({
        "video_codec": info.video_codec,
        "width": info.width,
        "height": info.height,
        "frame_rate": info.frame_rate,
        "duration": info.duration,
    })))
}

async fn transcode(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(params): Query<TranscodeParams>,
    body: Bytes,
) -> Result<Response, ApiError> {
    let is_json = headers
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(|ct| ct.starts_with("application/json"))
        .unwrap_or(false);

    // Two ways to submit: a structured JSON body (file path / inline base64,
    // optional server-side output path) or a streamed binary body + query spec.
    let (media, spec_params, output_path, sync) = if is_json {
        let req: TranscodeRequest = serde_json::from_slice(&body)
            .map_err(|e| ApiError::bad_request(anyhow::anyhow!("invalid JSON body: {e}")))?;
        let media = read_input(&req.input)?;
        let output_path = match &req.output {
            Some(o) => Some(resolve_path(&o.path, false)?),
            None => None,
        };
        (media, req.spec.into_params(), output_path, req.sync)
    } else {
        if body.is_empty() {
            return Err(ApiError::bad_request(anyhow::anyhow!(
                "empty request body — POST media bytes (binary), or send `application/json` with input.path / input.base64"
            )));
        }
        let sync = params.sync.unwrap_or(false);
        (body, params, None, sync)
    };

    if media.is_empty() {
        return Err(ApiError::bad_request(anyhow::anyhow!("no input media")));
    }

    // Probe the source so `ladder`/source-resolution rungs and validation work.
    let info = crate::probe::probe_bytes(&media).map_err(ApiError::bad_request)?;
    let settings = spec_params.into_settings().map_err(ApiError::bad_request)?;
    let spec = settings
        .into_spec(info.width, info.height)
        .map_err(ApiError::bad_request)?;

    let id = Uuid::new_v4();
    let mode = if matches!(spec.mode, crate::spec::OutputMode::Hls { .. }) {
        "hls"
    } else {
        "single"
    };
    let handle = Arc::new(JobHandle::new(id, mode));
    state.jobs.write().unwrap().insert(id, Arc::clone(&handle));

    let task = run_job_task(Arc::clone(&handle), media, spec, output_path);

    if sync {
        task.await; // run inline
        return sync_response(&handle);
    }
    tokio::spawn(task);
    Ok((
        StatusCode::ACCEPTED,
        Json(json!({ "job_id": id.to_string(), "status": "queued" })),
    )
        .into_response())
}

/// Write one single-file rung's MP4 to a server path: the path itself for a
/// lone rung, or `<dir>/<label>.mp4` when there are several. Returns the path.
fn write_single_file(bytes: &[u8], output: &std::path::Path, label: &str, multi: bool) -> Result<String, String> {
    let dest = if multi {
        std::fs::create_dir_all(output).map_err(|e| format!("creating {}: {e}", output.display()))?;
        output.join(format!("{label}.mp4"))
    } else {
        output.to_path_buf()
    };
    std::fs::write(&dest, bytes).map_err(|e| format!("writing {}: {e}", dest.display()))?;
    Ok(dest.display().to_string())
}

/// The actual transcode future (shared by the async + sync paths). When
/// `output_path` is set, artifacts are written to the server filesystem
/// (single-file MP4 bytes, or the HLS tree as the asset root) instead of being
/// held in RAM.
fn run_job_task(
    handle: Arc<JobHandle>,
    body: Bytes,
    spec: OutputSpec,
    output_path: Option<PathBuf>,
) -> impl std::future::Future<Output = ()> {
    async move {
        handle.set_phase(Phase::Running);
        let is_hls = matches!(spec.mode, crate::spec::OutputMode::Hls { .. });

        // HLS needs an on-disk asset root: honor `output_path` if given, else a
        // tempdir we keep alive for the process. Single-file keeps bytes in RAM
        // unless `output_path` is set (then it's written below).
        let mut tmp_guard = None;
        let out_dir: Option<PathBuf> = if is_hls {
            if let Some(p) = &output_path {
                if let Err(e) = std::fs::create_dir_all(p) {
                    *handle.error.lock().unwrap() =
                        Some(format!("creating output dir {}: {e}", p.display()));
                    handle.set_phase(Phase::Failed);
                    return;
                }
                *handle.output_dir.lock().unwrap() = Some(p.clone());
                Some(p.clone())
            } else {
                match tempfile::Builder::new().prefix("rivet-api-").tempdir() {
                    Ok(d) => {
                        let path = d.path().to_path_buf();
                        *handle.output_dir.lock().unwrap() = Some(path.clone());
                        tmp_guard = Some(d);
                        Some(path)
                    }
                    Err(e) => {
                        *handle.error.lock().unwrap() = Some(format!("tempdir: {e}"));
                        handle.set_phase(Phase::Failed);
                        return;
                    }
                }
            }
        } else {
            None
        };

        let sink: Arc<dyn ProgressSink> = Arc::new(RegistrySink {
            handle: Arc::clone(&handle),
        });
        let result = crate::job::run_job(body, &spec, out_dir.as_deref(), sink).await;
        match result {
            Ok(out) => {
                let multi = out.rungs.len() > 1;
                let mut write_err: Option<String> = None;
                {
                    let mut arts = handle.artifacts.lock().unwrap();
                    for r in out.rungs {
                        let (data, written) = match r.artifact {
                            crate::job::RungArtifact::File(bytes) => {
                                if let Some(p) = &output_path {
                                    match write_single_file(&bytes, p, &r.label, multi) {
                                        Ok(dest) => (None, Some(dest)),
                                        Err(e) => {
                                            write_err.get_or_insert(e);
                                            (Some(Bytes::from(bytes)), None)
                                        }
                                    }
                                } else {
                                    (Some(Bytes::from(bytes)), None)
                                }
                            }
                            crate::job::RungArtifact::HlsRendition { .. } => (None, None),
                        };
                        arts.push(ArtifactEntry {
                            label: r.label,
                            width: r.width,
                            height: r.height,
                            frames: r.frames,
                            bytes: r.bytes,
                            data,
                            output_path: written,
                        });
                    }
                }
                if out.master_playlist.is_some() {
                    *handle.master_playlist.lock().unwrap() =
                        Some(format!("/v1/jobs/{}/files/master.m3u8", handle.id));
                }
                if let Some(e) = write_err {
                    *handle.error.lock().unwrap() = Some(e);
                    handle.set_phase(Phase::Failed);
                } else {
                    handle.set_phase(Phase::Completed);
                }
            }
            Err(e) => {
                *handle.error.lock().unwrap() = Some(format!("{e:#}"));
                handle.set_phase(Phase::Failed);
            }
        }
        // Keep the HLS tempdir alive for the process lifetime so /files works.
        if let Some(d) = tmp_guard {
            std::mem::forget(d);
        }
    }
}

fn sync_response(handle: &Arc<JobHandle>) -> Result<Response, ApiError> {
    if *handle.phase.lock().unwrap() == Phase::Failed {
        let msg = handle.error.lock().unwrap().clone().unwrap_or_default();
        return Err(ApiError::internal(anyhow::anyhow!(msg)));
    }
    // Extract any in-RAM single-file bytes, then DROP the lock — `status_json()`
    // below re-locks `artifacts`, and std `Mutex` isn't reentrant (holding it
    // here would deadlock the handler; this is the path output.path takes).
    let streamable = {
        let arts = handle.artifacts.lock().unwrap();
        arts.iter().find_map(|a| a.data.clone())
    };
    if let Some(data) = streamable {
        return Ok((StatusCode::OK, [(header::CONTENT_TYPE, "video/mp4")], data).into_response());
    }
    // output.path / multi-rung / HLS: return the status JSON (paths + progress).
    Ok(Json(handle.status_json()).into_response())
}

async fn job_status(
    State(state): State<AppState>,
    AxPath(id): AxPath<String>,
) -> Result<Json, ApiError> {
    let handle = lookup(&state, &id)?;
    Ok(Json(handle.status_json()))
}

async fn artifact(
    State(state): State<AppState>,
    AxPath((id, label)): AxPath<(String, String)>,
) -> Result<Response, ApiError> {
    let handle = lookup(&state, &id)?;
    let arts = handle.artifacts.lock().unwrap();
    let entry = arts
        .iter()
        .find(|a| a.label == label && a.data.is_some())
        .ok_or_else(|| ApiError::not_found(format!("artifact '{label}'")))?;
    let data = entry.data.clone().unwrap();
    Ok((StatusCode::OK, [(header::CONTENT_TYPE, "video/mp4")], data).into_response())
}

async fn hls_file(
    State(state): State<AppState>,
    AxPath((id, path)): AxPath<(String, String)>,
) -> Result<Response, ApiError> {
    let handle = lookup(&state, &id)?;
    let root = handle
        .output_dir
        .lock()
        .unwrap()
        .clone()
        .ok_or_else(|| ApiError::not_found("HLS output".into()))?;
    // Path-traversal guard: no `..`, no absolute components.
    if path.split(['/', '\\']).any(|c| c == ".." || c.is_empty()) {
        return Err(ApiError::bad_request(anyhow::anyhow!("invalid path")));
    }
    let full = root.join(&path);
    let data = std::fs::read(&full).map_err(|_| ApiError::not_found(path.clone()))?;
    let ct = content_type_for(&path);
    Ok((StatusCode::OK, [(header::CONTENT_TYPE, ct)], data).into_response())
}

fn content_type_for(path: &str) -> &'static str {
    match path.rsplit('.').next() {
        Some("m3u8") => "application/vnd.apple.mpegurl",
        Some("m4s") => "video/iso.segment",
        Some("mp4") => "video/mp4",
        _ => "application/octet-stream",
    }
}

fn lookup(state: &AppState, id: &str) -> Result<Arc<JobHandle>, ApiError> {
    let uuid = Uuid::parse_str(id).map_err(|_| ApiError::not_found("job".into()))?;
    state
        .jobs
        .read()
        .unwrap()
        .get(&uuid)
        .cloned()
        .ok_or_else(|| ApiError::not_found(format!("job '{id}'")))
}

// ---------------------------------------------------------------------------
// API documentation — OpenAPI 3.0 + Swagger UI + Redoc
// ---------------------------------------------------------------------------

async fn landing() -> Html<&'static str> {
    Html(LANDING_HTML)
}

async fn openapi_json() -> Json {
    Json(openapi_spec())
}

async fn swagger_ui() -> Html<&'static str> {
    Html(SWAGGER_HTML)
}

async fn redoc_ui() -> Html<&'static str> {
    Html(REDOC_HTML)
}

/// String query parameter for the transcode endpoint.
fn qp(name: &str, ty: &str, desc: &str) -> Value {
    json!({
        "name": name, "in": "query", "required": false,
        "schema": { "type": ty }, "description": desc
    })
}

/// The hand-authored OpenAPI 3.0 document describing the API. Hand-authored
/// (rather than derived) because the JSON responses are dynamic.
pub fn openapi_spec() -> Value {
    json!({
        "openapi": "3.0.3",
        "info": {
            "title": "rivet transcode API",
            "version": env!("CARGO_PKG_VERSION"),
            "description": "HTTP API for the rivet GPU video transcoder. POST media \
                            and an output spec; rivet transcodes to AV1 (single-file \
                            MP4 or CMAF/HLS) and reports per-rung progress.",
            "license": { "name": "Open Encoding Attribution License v1.0", "url": "https://github.com/elyerinfox/rivet/blob/develop/LICENSE.md" }
        },
        "servers": [ { "url": "/", "description": "this server" } ],
        "tags": [
            { "name": "status", "description": "Health + media inspection" },
            { "name": "jobs", "description": "Submit + track transcode jobs" }
        ],
        "paths": {
            "/v1/health": {
                "get": {
                    "tags": ["status"],
                    "summary": "Liveness, detected GPUs, and build output capabilities",
                    "responses": { "200": {
                        "description": "ok",
                        "content": { "application/json": { "schema": { "$ref": "#/components/schemas/Health" } } }
                    } }
                }
            },
            "/v1/probe": {
                "post": {
                    "tags": ["status"],
                    "summary": "Probe media without transcoding",
                    "requestBody": { "required": true, "content": {
                        "application/octet-stream": { "schema": { "type": "string", "format": "binary" } }
                    } },
                    "responses": {
                        "200": { "description": "media info",
                                 "content": { "application/json": { "schema": { "$ref": "#/components/schemas/MediaInfo" } } } },
                        "400": { "$ref": "#/components/responses/Error" }
                    }
                }
            },
            "/v1/transcode": {
                "post": {
                    "tags": ["jobs"],
                    "summary": "Submit a transcode job (structured JSON body or streamed media)",
                    "description": "Two ways to submit. (1) `application/json`: a structured \
                                    TranscodeRequest — input from a server file `path` or inline \
                                    `base64`, an optional server `output.path`, and a structured \
                                    `spec`. No media upload required. (2) a streamed binary body \
                                    (`application/octet-stream`): the raw media bytes, with the \
                                    spec in the query parameters below. Either way: returns 202 + \
                                    a job id and runs asynchronously, unless sync=true, which \
                                    blocks and returns the MP4 (or a JSON summary when written to \
                                    a path). Query params apply to the binary form only.",
                    "parameters": [
                        qp("mode", "string", "single (default) or hls"),
                        qp("rungs", "string", "Comma-separated WxH, e.g. 1280x720,640x360. Omit for source resolution."),
                        qp("ladder", "boolean", "Derive a standard ABR ladder from the source."),
                        qp("max_short_side", "integer", "Cap the ladder's tallest rung's short side."),
                        qp("segment_seconds", "number", "HLS target segment length (default 4)."),
                        qp("crf", "integer", "Constant rate factor (encoder-native 0..255)."),
                        qp("speed", "integer", "Encoder speed preset."),
                        qp("audio", "string", "auto (default) | opus | drop"),
                        qp("color", "string", "sdr (default) | hdr10 | hlg | passthrough"),
                        qp("pixel_format", "string", "auto (default) | 8bit | 10bit"),
                        qp("seam", "string", "parallel (default) | constqp | serial"),
                        qp("max_fps", "number", "Cap the output frame rate."),
                        qp("gpu", "integer", "Pin encode/decode to this GPU index."),
                        qp("filter", "string", "Video filter chain, e.g. crop=1280:720,hflip."),
                        qp("sync", "boolean", "Block and return the artifact directly.")
                    ],
                    "requestBody": { "required": true, "content": {
                        "application/json": { "schema": { "$ref": "#/components/schemas/TranscodeRequest" } },
                        "application/octet-stream": { "schema": { "type": "string", "format": "binary" } }
                    } },
                    "responses": {
                        "202": { "description": "job accepted",
                                 "content": { "application/json": { "schema": { "$ref": "#/components/schemas/Accepted" } } } },
                        "200": { "description": "sync=true: the MP4 (single-file) or job status JSON",
                                 "content": { "video/mp4": { "schema": { "type": "string", "format": "binary" } } } },
                        "400": { "$ref": "#/components/responses/Error" }
                    }
                }
            },
            "/v1/jobs/{id}": {
                "get": {
                    "tags": ["jobs"],
                    "summary": "Job status + per-rung progress + outputs",
                    "parameters": [ { "name": "id", "in": "path", "required": true, "schema": { "type": "string", "format": "uuid" } } ],
                    "responses": {
                        "200": { "description": "job status",
                                 "content": { "application/json": { "schema": { "$ref": "#/components/schemas/JobStatus" } } } },
                        "404": { "$ref": "#/components/responses/Error" }
                    }
                }
            },
            "/v1/jobs/{id}/artifacts/{label}": {
                "get": {
                    "tags": ["jobs"],
                    "summary": "Download a single-file rung's MP4",
                    "parameters": [
                        { "name": "id", "in": "path", "required": true, "schema": { "type": "string", "format": "uuid" } },
                        { "name": "label", "in": "path", "required": true, "schema": { "type": "string" }, "description": "rung label, e.g. 720p" }
                    ],
                    "responses": {
                        "200": { "description": "MP4", "content": { "video/mp4": { "schema": { "type": "string", "format": "binary" } } } },
                        "404": { "$ref": "#/components/responses/Error" }
                    }
                }
            },
            "/v1/jobs/{id}/files/{path}": {
                "get": {
                    "tags": ["jobs"],
                    "summary": "Fetch a file from an HLS job's output tree",
                    "parameters": [
                        { "name": "id", "in": "path", "required": true, "schema": { "type": "string", "format": "uuid" } },
                        { "name": "path", "in": "path", "required": true, "schema": { "type": "string" }, "description": "e.g. master.m3u8 or video/720p/seg-00001.m4s" }
                    ],
                    "responses": {
                        "200": { "description": "the file (m3u8 / m4s / mp4)" },
                        "404": { "$ref": "#/components/responses/Error" }
                    }
                }
            }
        },
        "components": {
            "responses": {
                "Error": { "description": "error",
                           "content": { "application/json": { "schema": { "$ref": "#/components/schemas/Error" } } } }
            },
            "schemas": {
                "Error": { "type": "object", "properties": { "error": { "type": "string" } } },
                "Accepted": { "type": "object", "properties": {
                    "job_id": { "type": "string", "format": "uuid" },
                    "status": { "type": "string", "example": "queued" }
                } },
                "TranscodeRequest": {
                    "type": "object", "required": ["input"],
                    "description": "Structured JSON transcode request (application/json).",
                    "properties": {
                        "input": { "$ref": "#/components/schemas/InputSource" },
                        "output": { "$ref": "#/components/schemas/OutputTarget" },
                        "spec": { "$ref": "#/components/schemas/SpecBody" },
                        "sync": { "type": "boolean", "description": "Block until done and return the result/summary." }
                    }
                },
                "InputSource": {
                    "type": "object",
                    "description": "Media source — set exactly one of path / base64.",
                    "properties": {
                        "path": { "type": "string", "description": "Server-side file path to read the media from." },
                        "base64": { "type": "string", "description": "The media inline, base64-encoded." }
                    }
                },
                "OutputTarget": {
                    "type": "object", "required": ["path"],
                    "properties": {
                        "path": { "type": "string", "description": "Server path to write the result (file for single-file single-rung; directory for multi-rung/HLS)." }
                    }
                },
                "SpecBody": {
                    "type": "object",
                    "description": "Structured output spec (the JSON form of the query params).",
                    "properties": {
                        "mode": { "type": "string", "enum": ["single", "hls"] },
                        "rungs": { "type": "array", "items": { "type": "string", "example": "1280x720" } },
                        "ladder": { "type": "boolean" },
                        "max_short_side": { "type": "integer" },
                        "segment_seconds": { "type": "number" },
                        "crf": { "type": "integer" },
                        "speed": { "type": "integer" },
                        "audio": { "type": "string", "enum": ["auto", "opus", "drop"] },
                        "color": { "type": "string", "enum": ["sdr", "hdr10", "hlg", "passthrough"] },
                        "bit_depth": { "type": "string", "enum": ["auto", "8bit", "10bit"] },
                        "seam": { "type": "string", "enum": ["parallel", "constqp", "serial"] },
                        "max_fps": { "type": "number" },
                        "gpu": { "type": "integer" },
                        "filter": { "type": "string", "example": "crop=1280:720,hflip" }
                    }
                },
                "Health": { "type": "object", "properties": {
                    "status": { "type": "string", "example": "ok" },
                    "service": { "type": "string", "example": "rivet" },
                    "gpus": { "type": "array", "items": { "type": "object", "properties": {
                        "index": { "type": "integer" }, "vendor": { "type": "string" }, "name": { "type": "string" }
                    } } },
                    "output_caps": { "type": "object", "properties": {
                        "max_bit_depth": { "type": "integer" }, "hdr": { "type": "boolean" }
                    } }
                } },
                "MediaInfo": { "type": "object", "properties": {
                    "video_codec": { "type": "string" }, "width": { "type": "integer" }, "height": { "type": "integer" },
                    "frame_rate": { "type": "number" }, "duration": { "type": "number" }
                } },
                "RungProgress": { "type": "object", "properties": {
                    "rung_index": { "type": "integer" }, "label": { "type": "string" },
                    "width": { "type": "integer" }, "height": { "type": "integer" },
                    "status": { "type": "string", "enum": ["pending", "running", "finalizing", "completed", "failed"] },
                    "percent": { "type": "number" }, "frames_done": { "type": "integer" }
                } },
                "Artifact": { "type": "object", "properties": {
                    "label": { "type": "string" }, "width": { "type": "integer" }, "height": { "type": "integer" },
                    "frames": { "type": "integer" }, "bytes": { "type": "integer" }, "url": { "type": "string" }
                } },
                "JobStatus": { "type": "object", "properties": {
                    "job_id": { "type": "string", "format": "uuid" },
                    "mode": { "type": "string" },
                    "status": { "type": "string", "enum": ["queued", "running", "completed", "failed"] },
                    "progress": { "type": "array", "items": { "$ref": "#/components/schemas/RungProgress" } },
                    "artifacts": { "type": "array", "items": { "$ref": "#/components/schemas/Artifact" } },
                    "master_playlist": { "type": "string", "nullable": true },
                    "error": { "type": "string", "nullable": true }
                } }
            }
        }
    })
}

const LANDING_HTML: &str = r#"<!DOCTYPE html><html><head><meta charset="utf-8">
<title>rivet transcode API</title><style>body{font:16px system-ui;margin:3rem auto;max-width:40rem}a{display:block;margin:.5rem 0}</style></head>
<body><h1>rivet transcode API</h1>
<p>Interactive documentation:</p>
<a href="/swagger">Swagger UI</a>
<a href="/redoc">Redoc</a>
<a href="/openapi.json">OpenAPI 3.0 document (JSON)</a>
<p>Quick check: <a href="/v1/health">/v1/health</a></p>
</body></html>"#;

const SWAGGER_HTML: &str = r#"<!DOCTYPE html><html><head><meta charset="utf-8">
<title>rivet API — Swagger UI</title>
<link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist/swagger-ui.css"></head>
<body><div id="swagger-ui"></div>
<script src="https://unpkg.com/swagger-ui-dist/swagger-ui-bundle.js"></script>
<script>window.ui=SwaggerUIBundle({url:'/openapi.json',dom_id:'#swagger-ui'});</script>
</body></html>"#;

const REDOC_HTML: &str = r#"<!DOCTYPE html><html><head><meta charset="utf-8">
<title>rivet API — Redoc</title><meta name="viewport" content="width=device-width,initial-scale=1"></head>
<body><redoc spec-url="/openapi.json"></redoc>
<script src="https://cdn.redoc.ly/redoc/latest/bundles/redoc.standalone.js"></script>
</body></html>"#;

// ---------------------------------------------------------------------------
// Response helpers
// ---------------------------------------------------------------------------

/// JSON response wrapper (so handlers can return `Json`).
struct Json(Value);

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
struct ApiError {
    status: StatusCode,
    message: String,
}

impl ApiError {
    fn bad_request(e: anyhow::Error) -> Self {
        Self { status: StatusCode::BAD_REQUEST, message: format!("{e:#}") }
    }
    fn internal(e: anyhow::Error) -> Self {
        Self { status: StatusCode::INTERNAL_SERVER_ERROR, message: format!("{e:#}") }
    }
    fn not_found(what: String) -> Self {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn query_params_into_settings_defaults() {
        let p = TranscodeParams::default();
        let spec = p.into_settings().unwrap().into_spec(1280, 720).unwrap();
        assert!(matches!(spec.mode, crate::spec::OutputMode::SingleFile));
        assert_eq!(spec.rungs.len(), 1);
        assert_eq!((spec.rungs[0].width, spec.rungs[0].height), (1280, 720));
    }

    #[test]
    fn query_params_explicit_rungs_and_hls() {
        let p = TranscodeParams {
            mode: Some("hls".into()),
            rungs: Some("1920x1080, 1280x720,640x360".into()),
            segment_seconds: Some(6.0),
            crf: Some(28),
            ..Default::default()
        };
        let spec = p.into_settings().unwrap().into_spec(1920, 1080).unwrap();
        assert!(matches!(spec.mode, crate::spec::OutputMode::Hls { .. }));
        assert_eq!(spec.rungs.len(), 3);
        assert_eq!(spec.rungs[1].quality.crf, Some(28));
    }

    #[test]
    fn json_spec_body_into_params_and_settings() {
        // The JSON body uses an array of rungs + a structured spec; it lands on
        // the same TranscodeSettings as the query string.
        let body = serde_json::json!({
            "mode": "hls",
            "rungs": ["1280x720", "640x360"],
            "crf": 30,
            "audio": "opus",
            "pixel_format": "auto"
        });
        let sb: SpecBody = serde_json::from_value(body).unwrap();
        let s = sb.into_params().into_settings().unwrap();
        assert_eq!(s.mode, Some(crate::settings::Mode::Hls));
        assert_eq!(s.rungs, vec![(1280, 720), (640, 360)]);
        assert_eq!(s.crf, Some(30));
        assert_eq!(s.audio, Some(crate::spec::AudioPolicy::ForceOpus));
    }

    #[test]
    fn query_params_reject_bad_values() {
        let bad = TranscodeParams {
            color: Some("ultrahd".into()),
            ..Default::default()
        };
        assert!(bad.into_settings().is_err());
        let bad_rung = TranscodeParams {
            rungs: Some("notarung".into()),
            ..Default::default()
        };
        assert!(bad_rung.into_settings().is_err());
    }

    #[test]
    fn base64_roundtrip() {
        // "rivet" → cml2ZXQ=
        assert_eq!(base64_decode("cml2ZXQ=").unwrap(), b"rivet");
        assert_eq!(base64_decode("").unwrap(), b"");
        assert!(base64_decode("not valid !!!").is_err());
    }
}
