use super::spec::{SpecBody, TranscodeParams, base64_decode};

/// `/v1/health`'s `output_caps` carries `by_codec`, each output codec's own
/// answer over the backends, and a codec-agnostic `max_bit_depth` / `hdr`
/// that every codec meets. A software-H.26x-only set is the case the old
/// union got wrong: it said 10-bit HDR, and this set has no AV1 encoder at all.
#[test]
fn health_output_caps_carry_each_codecs_own_answer() {
    use codec::encode::EncoderBackend::{Av1, H26x, Nvenc};
    use crate::spec::{CodecOutputCaps, OUTPUT_CODECS};
    use super::handlers::output_caps_json;

    let over = |set: &[codec::encode::EncoderBackend]| -> Vec<CodecOutputCaps> {
        OUTPUT_CODECS.iter().map(|&c| CodecOutputCaps::over(c, set)).collect()
    };
    let want: serde_json::Value = serde_json::from_str(
        r#"{"max_bit_depth":8,"hdr":false,"by_codec":[
            {"codec":"av1","max_bit_depth":8,"hdr":false,"backends":[]},
            {"codec":"h264","max_bit_depth":10,"hdr":true,"backends":[{"backend":"h26x","max_bit_depth":10,"hdr":true}]},
            {"codec":"h265","max_bit_depth":10,"hdr":true,"backends":[{"backend":"h26x","max_bit_depth":10,"hdr":true}]},
            {"codec":"vp9","max_bit_depth":8,"hdr":false,"backends":[]},
            {"codec":"vp8","max_bit_depth":8,"hdr":false,"backends":[]},
            {"codec":"mpeg2","max_bit_depth":8,"hdr":false,"backends":[]},
            {"codec":"mpeg4","max_bit_depth":8,"hdr":false,"backends":[]},
            {"codec":"prores","max_bit_depth":8,"hdr":false,"backends":[]}]}"#,
    )
    .unwrap();
    assert_eq!(output_caps_json(&over(&[H26x])), want);
    // NVENC alone: 10-bit HDR AV1 and H.265, 8-bit SDR H.264 — not every codec.
    let nvenc = output_caps_json(&over(&[Nvenc]));
    assert_eq!((nvenc["max_bit_depth"].clone(), nvenc["hdr"].clone()), (8.into(), false.into()));

    // The same block `rivet capabilities --json` prints as `encode.by_codec`
    // for this set (its test in commands/capabilities.rs pins the string).
    let cli_by_codec: serde_json::Value = serde_json::from_str(
        "[{\"codec\":\"av1\",\"max_bit_depth\":10,\"hdr\":true,\"backends\":[\
         {\"backend\":\"nvenc\",\"max_bit_depth\":10,\"hdr\":true},\
         {\"backend\":\"av1\",\"max_bit_depth\":10,\"hdr\":true}]},\
         {\"codec\":\"h264\",\"max_bit_depth\":10,\"hdr\":true,\"backends\":[\
         {\"backend\":\"nvenc\",\"max_bit_depth\":8,\"hdr\":false},\
         {\"backend\":\"h26x\",\"max_bit_depth\":10,\"hdr\":true}]},\
         {\"codec\":\"h265\",\"max_bit_depth\":10,\"hdr\":true,\"backends\":[\
         {\"backend\":\"nvenc\",\"max_bit_depth\":10,\"hdr\":true},\
         {\"backend\":\"h26x\",\"max_bit_depth\":10,\"hdr\":true}]},\
         {\"codec\":\"vp9\",\"max_bit_depth\":8,\"hdr\":false,\"backends\":[]},\
         {\"codec\":\"vp8\",\"max_bit_depth\":8,\"hdr\":false,\"backends\":[]},\
         {\"codec\":\"mpeg2\",\"max_bit_depth\":8,\"hdr\":false,\"backends\":[]},\
         {\"codec\":\"mpeg4\",\"max_bit_depth\":8,\"hdr\":false,\"backends\":[]},\
         {\"codec\":\"prores\",\"max_bit_depth\":8,\"hdr\":false,\"backends\":[]}]",
    )
    .unwrap();
    let got = output_caps_json(&over(&[Nvenc, Av1, H26x]));
    assert_eq!(got["by_codec"], cli_by_codec);
    // Every codec is 10-bit HDR on this set, so the codec-agnostic fields say
    // so; the same keys as ever, and nothing else is added.
    assert_eq!(got["max_bit_depth"], 10);
    assert_eq!(got["hdr"], true);
    assert_eq!(got.as_object().unwrap().len(), 3);
}

/// The handler reports this build: `by_codec` agrees with
/// `build_output_caps_for` per codec, and the codec-agnostic fields with what
/// every codec meets — the lowest depth, HDR only if every codec has it.
#[test]
fn health_reports_this_builds_caps_per_codec() {
    use crate::spec::{OUTPUT_CODECS, output_codec_label};
    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
    let super::Json(v) = rt.block_on(super::handlers::health());
    let caps = &v["output_caps"];
    // The codec-agnostic fields are over the web set (AV1, H.264, H.265), as
    // `every_codec_output_caps` takes them.
    let each: Vec<_> =
        OUTPUT_CODECS.iter().filter(|c| c.is_web_set()).map(|&c| codec::encode::build_output_caps_for(c)).collect();
    assert_eq!(caps["max_bit_depth"], each.iter().map(|c| c.max_bit_depth).min().unwrap());
    assert_eq!(caps["hdr"], each.iter().all(|c| c.hdr));
    let by_codec = caps["by_codec"].as_array().expect("by_codec is an array");
    assert_eq!(by_codec.len(), OUTPUT_CODECS.len());
    for (entry, &codec) in by_codec.iter().zip(OUTPUT_CODECS.iter()) {
        let want = codec::encode::build_output_caps_for(codec);
        assert_eq!(entry["codec"], output_codec_label(codec));
        assert_eq!(entry["max_bit_depth"], want.max_bit_depth, "{codec:?}");
        assert_eq!(entry["hdr"], want.hdr, "{codec:?}");
    }
}

#[test]
fn query_params_into_settings_defaults() {
    let p = TranscodeParams::default();
    let spec = p.to_settings().unwrap().into_spec(1280, 720).unwrap();
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
    let spec = p.to_settings().unwrap().into_spec(1920, 1080).unwrap();
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
    let s = sb.into_params().to_settings().unwrap();
    assert_eq!(s.mode, Some(crate::settings::Mode::Hls));
    assert_eq!(s.rungs, vec![(1280, 720).into(), (640, 360).into()]);
    assert_eq!(s.crf, Some(30));
    assert_eq!(s.audio, Some(crate::spec::AudioCodecPolicy::ForceOpus));
}

/// Rates on both HTTP forms: a rung's `@RATE` in either rung list, and
/// `video_bitrate` / `video_buffer` as strings, through the same readers as
/// the CLI. A bad rate or a unit-less buffer is refused.
#[test]
fn video_rates_on_both_http_forms() {
    let p = TranscodeParams {
        codec: Some("h265".into()),
        rungs: Some("1280x720@2.5M,640x360".into()),
        video_bitrate: Some("800k".into()),
        video_buffer: Some("1s".into()),
        ..Default::default()
    };
    let s = p.to_settings().unwrap();
    assert_eq!(s.rungs[0].bitrate, Some(2_500_000));
    assert_eq!((s.video_bitrate, s.video_buffer_ms), (Some(800_000), Some(1000)));
    let spec = s.into_spec(1280, 720).unwrap().with_rung_policy_resolved();
    let rates: Vec<_> = spec.rungs.iter().map(|r| r.quality.overrides.bitrate).collect();
    assert_eq!(rates, vec![Some(2_500_000), Some(800_000)]);
    let sb: SpecBody = serde_json::from_value(serde_json::json!({
        "codec": "h264", "rungs": ["1920x1080@5M"], "video_buffer": "500ms"
    }))
    .unwrap();
    let s = sb.into_params().to_settings().unwrap();
    assert_eq!((s.rungs[0].bitrate, s.video_buffer_ms), (Some(5_000_000), Some(500)));
    let bad = TranscodeParams { video_buffer: Some("1000".into()), ..Default::default() };
    assert!(bad.to_settings().is_err(), "a buffer needs a unit");
    let bad = TranscodeParams { rungs: Some("1280x720@fast".into()), ..Default::default() };
    assert!(bad.to_settings().is_err(), "not a rate");
}

/// The `subtitles` key means the same thing on the query string and in the
/// JSON body as on the CLI: it reaches `settings::parse_subtitles`.
#[test]
fn subtitles_key_is_the_shared_vocabulary_on_both_http_forms() {
    use crate::spec::SubtitlePolicy;
    let p = TranscodeParams { subtitles: Some("eng,deu".into()), ..Default::default() };
    let s = p.to_settings().unwrap();
    assert_eq!(s.subtitles, Some(SubtitlePolicy::Only(vec!["eng".into(), "deu".into()])));
    let sb: SpecBody = serde_json::from_value(serde_json::json!({ "subtitles": "none" })).unwrap();
    assert_eq!(sb.into_params().to_settings().unwrap().subtitles, Some(SubtitlePolicy::Drop));
    let bad = TranscodeParams { subtitles: Some("english".into()), ..Default::default() };
    assert!(bad.to_settings().is_err(), "not a language code");
}

#[test]
fn query_params_reject_bad_values() {
    let bad = TranscodeParams {
        color: Some("ultrahd".into()),
        ..Default::default()
    };
    assert!(bad.to_settings().is_err());
    let bad_rung = TranscodeParams {
        rungs: Some("notarung".into()),
        ..Default::default()
    };
    assert!(bad_rung.to_settings().is_err());
}

#[test]
fn base64_roundtrip() {
    // "rivet" → cml2ZXQ=
    assert_eq!(base64_decode("cml2ZXQ=").unwrap(), b"rivet");
    assert_eq!(base64_decode("").unwrap(), b"");
    assert!(base64_decode("not valid !!!").is_err());
}

/// A failed rung's status carries why it failed — the whole error chain the
/// job layer reported, not just its outermost context — and a rung that has
/// not failed carries `null`.
#[test]
fn a_failed_rungs_status_carries_its_error_chain() {
    use crate::progress::{RungProgress, RungStatus};
    let rung = |status, message: Option<&str>| RungProgress {
        rung_index: 0,
        label: "360p".into(),
        width: 640,
        height: 360,
        status,
        percent: 0.0,
        frames_done: 0,
        frames_total: None,
        segments_written: 0,
        bytes_out: 0,
        message: message.map(str::to_string),
    };
    let chain = "finalize: placing video samples by presentation order: composition offsets: \
                 presentation timestamp 30 appears on two samples; a display order is undefined";
    let failed = super::rung_progress_json(&rung(RungStatus::Failed, Some(chain)));
    assert_eq!(failed["status"], "failed");
    assert_eq!(failed["message"], chain);
    let running = super::rung_progress_json(&rung(RungStatus::Running, None));
    assert!(running["message"].is_null());
}

/// `rate_mode` on both HTTP forms, through the settings vocabulary; a word
/// it does not know is refused.
#[test]
fn rate_mode_on_both_http_forms() {
    use codec::encode::tuning::RateMode;
    let p = TranscodeParams { codec: Some("av1".into()), rate_mode: Some("cbr".into()), ..Default::default() };
    assert_eq!(p.to_settings().unwrap().rate_mode, Some(RateMode::Constant));
    let sb: SpecBody = serde_json::from_value(serde_json::json!({
        "codec": "h264", "rungs": ["1280x720"], "rate_mode": "constant", "video_bitrate": "2M"
    }))
    .unwrap();
    let s = sb.into_params().to_settings().unwrap();
    assert_eq!((s.rate_mode, s.video_bitrate), (Some(RateMode::Constant), Some(2_000_000)));
    let spec = s.into_spec(1280, 720).unwrap().with_constant_rates_resolved(30.0);
    assert_eq!(spec.rungs[0].quality.overrides.rate_mode, Some(RateMode::Constant));
    assert_eq!(spec.rungs[0].quality.overrides.bitrate, Some(2_000_000));
    let bad = TranscodeParams { rate_mode: Some("vbr".into()), ..Default::default() };
    assert!(bad.to_settings().is_err(), "not a rate mode");
}

// ---------------------------------------------------------------------------
// Hooks
// ---------------------------------------------------------------------------

fn hooked_router() -> axum::Router {
    use crate::hooks::{DigestAlgorithm, HookContext, HookOutcome, HookPolicy, Hooks, ProbeEvent, ProbeHook, SourceDigest};
    struct Gate;
    impl ProbeHook for Gate {
        fn on_probe(&self, _: &HookContext, _: &ProbeEvent) -> anyhow::Result<HookOutcome> {
            Ok(HookOutcome::reject("held for review"))
        }
    }
    let hooks = Hooks::new()
        .source("digest", SourceDigest::new(&[DigestAlgorithm::Sha256]))
        .probe_with("gate", Gate, HookPolicy::default().optional());
    super::build_router_with_hooks(hooks)
}

async fn call(router: axum::Router, req: axum::http::Request<axum::body::Body>) -> (u16, serde_json::Value) {
    use tower::ServiceExt;
    let resp = router.oneshot(req).await.unwrap();
    let status = resp.status().as_u16();
    let body = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    (status, serde_json::from_slice(&body).unwrap_or(serde_json::Value::Null))
}

/// `GET /v1/hooks` lists the configured hooks with their policies.
#[tokio::test]
async fn hooks_endpoint_lists_the_configured_hooks() {
    let req = axum::http::Request::get("/v1/hooks").body(axum::body::Body::empty()).unwrap();
    let (status, v) = call(hooked_router(), req).await;
    assert_eq!(status, 200);
    let names: Vec<&str> = v["hooks"].as_array().unwrap().iter().map(|h| h["name"].as_str().unwrap()).collect();
    assert_eq!(names, ["digest", "gate"]);
    assert_eq!(v["hooks"][0]["kind"], "source");
    assert_eq!(v["hooks"][1]["kind"], "probe");
    assert_eq!(v["hooks"][1]["required"], false);
}

/// A request naming a hook that is not configured is refused.
#[tokio::test]
async fn naming_an_unknown_hook_is_a_bad_request() {
    let Ok(media) = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/../../test_media/bbb_h264_360p_short.mp4")) else {
        return;
    };
    let req = axum::http::Request::post("/v1/transcode?hooks=nope&sync=true").body(axum::body::Body::from(media)).unwrap();
    let (status, v) = call(hooked_router(), req).await;
    assert_eq!(status, 400);
    assert!(v["error"].as_str().unwrap().contains("nope"));
}

/// An optional hook the request opts into rejects the job: a sync request gets
/// 422, and the job's status says `rejected` and carries the hook report.
#[tokio::test(flavor = "multi_thread")]
async fn a_rejected_job_is_422_and_reports_its_hooks() {
    let Ok(media) = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/../../test_media/bbb_h264_360p_short.mp4")) else {
        return;
    };
    let router = hooked_router();
    let req = axum::http::Request::post("/v1/transcode?hooks=gate&sync=true")
        .body(axum::body::Body::from(media))
        .unwrap();
    let (status, v) = call(router.clone(), req).await;
    assert_eq!(status, 422, "{v}");
    assert!(v["error"].as_str().unwrap().contains("held for review"));

    // The async form: the status of the job reports the rejection and the
    // required digest hook's record.
    let media = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/../../test_media/bbb_h264_360p_short.mp4")).unwrap();
    let req = axum::http::Request::post("/v1/transcode?hooks=gate").body(axum::body::Body::from(media)).unwrap();
    let (status, v) = call(router.clone(), req).await;
    assert_eq!(status, 202);
    let id = v["job_id"].as_str().unwrap().to_string();
    let mut job = serde_json::Value::Null;
    for _ in 0..200 {
        let req = axum::http::Request::get(format!("/v1/jobs/{id}")).body(axum::body::Body::empty()).unwrap();
        job = call(router.clone(), req).await.1;
        if job["status"] != "queued" && job["status"] != "running" {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
    }
    assert_eq!(job["status"], "rejected", "{job}");
    assert_eq!(job["hooks"]["job_id"], id);
    assert_eq!(job["hooks"]["rejection"]["hook"], "gate");
    assert_eq!(job["hooks"]["rejection"]["kind"], "probe");
    let records = job["hooks"]["records"].as_array().unwrap();
    assert!(records.iter().any(|r| r["hook"] == "digest" && r["kind"] == "source" && r["annotations"]["sha256"].is_string()));
}

/// The words that state a default reach the HTTP forms too: `gop` in frames
/// or seconds, `max_fps` / `max_short_side` as a number or a word, the
/// bitrates as `standard`, on the query string and in the JSON body.
#[test]
fn explicit_default_words_on_both_http_forms() {
    let query: TranscodeParams =
        axum::extract::Query::try_from_uri(&"http://x/?gop=2s&max_fps=source&max_short_side=standard&audio_bitrate=standard&video_bitrate=standard".parse().unwrap())
            .unwrap()
            .0;
    let s = query.to_settings().unwrap();
    assert!(s.is_empty(), "every value stated is the default");
    let query: TranscodeParams =
        axum::extract::Query::try_from_uri(&"http://x/?gop=48&max_fps=29.97&max_short_side=720".parse().unwrap()).unwrap().0;
    let s = query.to_settings().unwrap();
    assert_eq!((s.gop, s.max_fps, s.max_short_side), (Some(48), Some(29.97), Some(720)));
    let query: TranscodeParams = axum::extract::Query::try_from_uri(&"http://x/?gop=1.5s".parse().unwrap()).unwrap().0;
    assert_eq!(query.to_settings().unwrap().gop_seconds, Some(1.5));
    let query: TranscodeParams = axum::extract::Query::try_from_uri(&"http://x/?gop=0s".parse().unwrap()).unwrap().0;
    assert!(query.to_settings().is_err());

    let body = |v: serde_json::Value| serde_json::from_value::<SpecBody>(v).unwrap().into_params().to_settings();
    let s = body(serde_json::json!({ "gop": 48, "max_fps": 29.97, "max_short_side": 720 })).unwrap();
    assert_eq!((s.gop, s.max_fps, s.max_short_side), (Some(48), Some(29.97), Some(720)), "numbers as before");
    let s = body(serde_json::json!({ "gop": "2s", "max_fps": "source", "max_short_side": "standard", "audio_bitrate": "standard", "video_bitrate": "standard" })).unwrap();
    assert!(s.is_empty());
    let s = body(serde_json::json!({ "gop": "0.5s", "max_fps": 30 })).unwrap();
    assert_eq!((s.gop, s.gop_seconds, s.max_fps), (None, Some(0.5), Some(30.0)));
    assert!(body(serde_json::json!({ "gop": "-1s" })).is_err());
    assert!(body(serde_json::json!({ "max_fps": "fast" })).is_err());
    assert!(serde_json::from_value::<SpecBody>(serde_json::json!({ "gop": [2] })).is_err(), "neither a number nor a word");
    let s = body(serde_json::json!({ "codec": "h264", "rungs": ["1920x1080@standard", "1280x720"], "video_bitrate": "2M" })).unwrap();
    assert!(s.rungs[0].standard_rate && !s.rungs[1].standard_rate);
}

// ---------------------------------------------------------------------------
// Synchronous responses
// ---------------------------------------------------------------------------

/// A completed single-file job whose rungs are held in RAM, one artifact per
/// label, each holding its label's bytes.
fn completed_single_file_job(labels: &[&str]) -> std::sync::Arc<super::JobHandle> {
    let handle = std::sync::Arc::new(super::JobHandle::new(uuid::Uuid::new_v4(), "single"));
    {
        let mut arts = handle.artifacts.lock().unwrap();
        for label in labels {
            let data = axum::body::Bytes::from(format!("bytes of {label}"));
            arts.push(super::ArtifactEntry {
                label: label.to_string(),
                width: 640,
                height: 360,
                frames: 1,
                bytes: data.len() as u64,
                data: Some(data),
                output_path: None,
            });
        }
    }
    handle.set_phase(super::Phase::Completed);
    handle
}

async fn sync_body(handle: &std::sync::Arc<super::JobHandle>) -> (u16, Option<String>, axum::body::Bytes) {
    let resp = super::handlers::sync_response(handle).map_err(|_| "sync_response failed").unwrap();
    let status = resp.status().as_u16();
    let ct = resp.headers().get(axum::http::header::CONTENT_TYPE).map(|v| v.to_str().unwrap().to_string());
    (status, ct, axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap())
}

/// A sync request for one single-file rung gets that file back.
#[tokio::test]
async fn sync_with_one_rung_returns_the_file() {
    let handle = completed_single_file_job(&["360p"]);
    let (status, ct, body) = sync_body(&handle).await;
    assert_eq!(status, 200);
    assert_ne!(ct.as_deref(), Some("application/json"));
    assert_eq!(&body[..], b"bytes of 360p");
}

/// A sync request for several single-file rungs gets the job status JSON,
/// listing every rung's download URL — not the first rung's file.
#[tokio::test]
async fn sync_with_several_rungs_returns_the_status_json() {
    let handle = completed_single_file_job(&["720p", "360p"]);
    let (status, ct, body) = sync_body(&handle).await;
    assert_eq!(status, 200);
    assert_eq!(ct.as_deref(), Some("application/json"));
    let v: serde_json::Value = serde_json::from_slice(&body).expect("the status JSON");
    assert_eq!(v["status"], "completed");
    let urls: Vec<&str> = v["artifacts"].as_array().unwrap().iter().map(|a| a["url"].as_str().unwrap()).collect();
    assert_eq!(urls, [format!("/v1/jobs/{}/artifacts/720p", handle.id), format!("/v1/jobs/{}/artifacts/360p", handle.id)]);
}

/// End to end through the router (when the test media is present, on a build
/// with the software H.264 encoder): one rung comes back as an MP4, two rungs
/// as the status JSON whose artifacts can be fetched.
#[cfg(feature = "h26x-fallback")]
#[tokio::test(flavor = "multi_thread")]
async fn sync_transcode_one_rung_is_the_file_and_several_are_the_status() {
    use tower::ServiceExt;
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../test_media/bbb_h264_360p_short.mp4");
    let Ok(media) = std::fs::read(path) else {
        return;
    };
    let router = super::build_router_with_hooks(crate::hooks::Hooks::default());

    let req = axum::http::Request::post("/v1/transcode?sync=true&codec=h264&rungs=320x180")
        .body(axum::body::Body::from(media.clone()))
        .unwrap();
    let resp = router.clone().oneshot(req).await.unwrap();
    let status = resp.status().as_u16();
    let ct = resp.headers()[axum::http::header::CONTENT_TYPE].clone();
    let body = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    assert_eq!(status, 200, "{}", String::from_utf8_lossy(&body));
    assert_eq!(ct, "video/mp4");
    assert_eq!(&body[4..8], b"ftyp");

    let req = axum::http::Request::post("/v1/transcode?sync=true&codec=h264&rungs=320x180,160x90")
        .body(axum::body::Body::from(media))
        .unwrap();
    let (status, v) = call(router.clone(), req).await;
    assert_eq!(status, 200, "{v}");
    assert_eq!(v["status"], "completed", "{v}");
    let arts = v["artifacts"].as_array().unwrap();
    assert_eq!(arts.len(), 2, "{v}");
    for a in arts {
        let url = a["url"].as_str().unwrap();
        let req = axum::http::Request::get(url).body(axum::body::Body::empty()).unwrap();
        let resp = router.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status().as_u16(), 200, "{url}");
    }
}
