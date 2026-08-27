use super::spec::{SpecBody, TranscodeParams, base64_decode};

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
    assert_eq!(s.audio, Some(crate::spec::AudioCodecPolicy::ForceOpus));
}

/// The `subtitles` key means the same thing on the query string and in the
/// JSON body as on the CLI: it reaches `settings::parse_subtitles`.
#[test]
fn subtitles_key_is_the_shared_vocabulary_on_both_http_forms() {
    use crate::spec::SubtitlePolicy;
    let p = TranscodeParams { subtitles: Some("eng,deu".into()), ..Default::default() };
    let s = p.into_settings().unwrap();
    assert_eq!(s.subtitles, Some(SubtitlePolicy::Only(vec!["eng".into(), "deu".into()])));
    let sb: SpecBody = serde_json::from_value(serde_json::json!({ "subtitles": "none" })).unwrap();
    assert_eq!(sb.into_params().into_settings().unwrap().subtitles, Some(SubtitlePolicy::Drop));
    let bad = TranscodeParams { subtitles: Some("english".into()), ..Default::default() };
    assert!(bad.into_settings().is_err(), "not a language code");
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
