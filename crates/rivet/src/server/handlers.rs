//! Axum route handlers for the rivet HTTP API.

use std::path::PathBuf;
use std::sync::Arc;

use axum::body::Bytes;
use axum::extract::{Path as AxPath, Query, State};
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::{Html, IntoResponse, Response};
use serde_json::json;
use uuid::Uuid;

use crate::progress::ProgressSink;
use crate::spec::OutputSpec;

use super::{
    ApiError, AppState, ArtifactEntry, JobHandle, Json, Phase, RegistrySink,
};
use super::docs::{openapi_spec, LANDING_HTML, REDOC_HTML, SWAGGER_HTML};
use super::spec::{TranscodeParams, TranscodeRequest, read_input, resolve_path};

// ---------------------------------------------------------------------------
// Status / probe handlers
// ---------------------------------------------------------------------------

pub(super) async fn health() -> Json {
    let gpus: Vec<serde_json::Value> = codec::gpu::detect_gpus()
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

pub(super) async fn probe(body: Bytes) -> Result<Json, ApiError> {
    let info = crate::probe::probe_bytes(&body).map_err(ApiError::bad_request)?;
    Ok(Json(json!({
        "video_codec": info.video_codec,
        "width": info.width,
        "height": info.height,
        "frame_rate": info.frame_rate,
        "duration": info.duration,
    })))
}

// ---------------------------------------------------------------------------
// Transcode handler
// ---------------------------------------------------------------------------

pub(super) async fn transcode(
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
    let settings = spec_params.to_settings().map_err(ApiError::bad_request)?;
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

// ---------------------------------------------------------------------------
// Job management helpers
// ---------------------------------------------------------------------------

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
pub(super) async fn run_job_task(
    handle: Arc<JobHandle>,
    body: Bytes,
    spec: OutputSpec,
    output_path: Option<PathBuf>,
) {
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

pub(super) fn sync_response(handle: &Arc<JobHandle>) -> Result<Response, ApiError> {
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

// ---------------------------------------------------------------------------
// Job status + artifact retrieval handlers
// ---------------------------------------------------------------------------

pub(super) async fn job_status(
    State(state): State<AppState>,
    AxPath(id): AxPath<String>,
) -> Result<Json, ApiError> {
    let handle = lookup(&state, &id)?;
    Ok(Json(handle.status_json()))
}

pub(super) async fn artifact(
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

pub(super) async fn hls_file(
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
// Documentation / UI handlers
// ---------------------------------------------------------------------------

pub(super) async fn landing() -> Html<&'static str> {
    Html(LANDING_HTML)
}

pub(super) async fn openapi_json() -> Json {
    Json(openapi_spec())
}

pub(super) async fn swagger_ui() -> Html<&'static str> {
    Html(SWAGGER_HTML)
}

pub(super) async fn redoc_ui() -> Html<&'static str> {
    Html(REDOC_HTML)
}
