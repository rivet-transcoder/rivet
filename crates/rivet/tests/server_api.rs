//! HTTP-layer tests for the `server` feature. Drives the axum router directly
//! via `ServiceExt::oneshot` — no socket, no client, no media needed for the
//! routing/error paths exercised here.
#![cfg(feature = "server")]

use axum::body::{Body, to_bytes};
use axum::http::{Request, StatusCode};
use rivet::server::build_router;
use serde_json::Value;
use tower::ServiceExt; // for `oneshot`

async fn get(uri: &str) -> (StatusCode, Value) {
    let app = build_router();
    let resp = app
        .oneshot(Request::builder().uri(uri).body(Body::empty()).unwrap())
        .await
        .unwrap();
    json(resp).await
}

async fn post_empty(uri: &str) -> (StatusCode, Value) {
    let app = build_router();
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(uri)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    json(resp).await
}

async fn json(resp: axum::response::Response) -> (StatusCode, Value) {
    let status = resp.status();
    let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let v: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, v)
}

async fn get_raw(uri: &str) -> (StatusCode, String) {
    let app = build_router();
    let resp = app
        .oneshot(Request::builder().uri(uri).body(Body::empty()).unwrap())
        .await
        .unwrap();
    let status = resp.status();
    let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    (status, String::from_utf8_lossy(&bytes).into_owned())
}

#[tokio::test]
async fn health_reports_ok_and_capabilities() {
    let (status, body) = get("/v1/health").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["status"], "ok");
    assert_eq!(body["service"], "rivet");
    // Always present (possibly empty) — proves the GPU + caps probe ran.
    assert!(body["gpus"].is_array());
    assert!(body["output_caps"]["max_bit_depth"].is_number());
}

#[tokio::test]
async fn transcode_rejects_empty_body() {
    let (status, body) = post_empty("/v1/transcode").await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(body["error"].is_string());
}

#[tokio::test]
async fn probe_rejects_non_media_body() {
    let app = build_router();
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/probe")
                .body(Body::from("not a media file"))
                .unwrap(),
        )
        .await
        .unwrap();
    let (status, body) = json(resp).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(body["error"].is_string());
}

#[tokio::test]
async fn unknown_job_is_404() {
    let (status, _) = get("/v1/jobs/00000000-0000-0000-0000-000000000000").await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn malformed_job_id_is_404() {
    let (status, _) = get("/v1/jobs/not-a-uuid").await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn openapi_document_is_served() {
    let (status, body) = get("/openapi.json").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["openapi"], "3.0.3");
    assert!(body["paths"]["/v1/transcode"].is_object());
    assert!(body["paths"]["/v1/health"].is_object());
    assert!(body["components"]["schemas"]["JobStatus"].is_object());
}

#[tokio::test]
async fn swagger_redoc_and_landing_render() {
    let (s, b) = get_raw("/swagger").await;
    assert_eq!(s, StatusCode::OK);
    assert!(b.contains("swagger-ui"));
    let (s, b) = get_raw("/redoc").await;
    assert_eq!(s, StatusCode::OK);
    assert!(b.contains("redoc"));
    let (s, b) = get_raw("/").await;
    assert_eq!(s, StatusCode::OK);
    assert!(b.contains("/openapi.json"));
}
