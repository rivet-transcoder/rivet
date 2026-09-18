use super::spec::{SpecBody, TranscodeParams, base64_decode};

/// `/v1/health`'s `output_caps` carries `by_codec`, each output codec's own
/// answer over the backends, and a codec-agnostic `max_bit_depth` / `hdr`
/// that every codec meets. A software-H.26x-only set is the case the old
/// union got wrong: it said 10-bit HDR, and this set has no AV1 encoder at all.
#[test]
fn health_output_caps_carry_each_codecs_own_answer() {
    use codec::encode::EncoderBackend::{H26x, Nvenc, Rav1e};
    use crate::spec::{CodecOutputCaps, OUTPUT_CODECS};
    use super::handlers::output_caps_json;

    let over = |set: &[codec::encode::EncoderBackend]| -> Vec<CodecOutputCaps> {
        OUTPUT_CODECS.iter().map(|&c| CodecOutputCaps::over(c, set)).collect()
    };
    let want: serde_json::Value = serde_json::from_str(
        r#"{"max_bit_depth":8,"hdr":false,"by_codec":[
            {"codec":"av1","max_bit_depth":8,"hdr":false,"backends":[]},
            {"codec":"h264","max_bit_depth":10,"hdr":true,"backends":[{"backend":"h26x","max_bit_depth":10,"hdr":true}]},
            {"codec":"h265","max_bit_depth":10,"hdr":true,"backends":[{"backend":"h26x","max_bit_depth":10,"hdr":true}]}]}"#,
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
         {\"backend\":\"rav1e\",\"max_bit_depth\":8,\"hdr\":false}]},\
         {\"codec\":\"h264\",\"max_bit_depth\":10,\"hdr\":true,\"backends\":[\
         {\"backend\":\"nvenc\",\"max_bit_depth\":8,\"hdr\":false},\
         {\"backend\":\"h26x\",\"max_bit_depth\":10,\"hdr\":true}]},\
         {\"codec\":\"h265\",\"max_bit_depth\":10,\"hdr\":true,\"backends\":[\
         {\"backend\":\"nvenc\",\"max_bit_depth\":10,\"hdr\":true},\
         {\"backend\":\"h26x\",\"max_bit_depth\":10,\"hdr\":true}]}]",
    )
    .unwrap();
    let got = output_caps_json(&over(&[Nvenc, Rav1e, H26x]));
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
    let each: Vec<_> = OUTPUT_CODECS.iter().map(|&c| codec::encode::build_output_caps_for(c)).collect();
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
    assert_eq!(s.rungs, vec![(1280, 720), (640, 360)]);
    assert_eq!(s.crf, Some(30));
    assert_eq!(s.audio, Some(crate::spec::AudioCodecPolicy::ForceOpus));
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
