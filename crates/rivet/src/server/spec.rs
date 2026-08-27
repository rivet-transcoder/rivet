//! Request parameter types, JSON body types, and spec-parsing helpers.
//!
//! Converts HTTP query parameters and the structured JSON request body into the
//! canonical [`TranscodeSettings`] used by the rest of the rivet engine.

use anyhow::{Context, Result};
use axum::body::Bytes;
use serde::Deserialize;

use crate::settings::TranscodeSettings;

use super::ApiError;

// ---------------------------------------------------------------------------
// Query-parameter struct
// ---------------------------------------------------------------------------

#[derive(Deserialize, Default, Clone)]
pub(super) struct TranscodeParams {
    /// `single` (default) or `hls`.
    pub(super) mode: Option<String>,
    /// Output video codec: `av1` (default), `h264`, or `h265`.
    pub(super) codec: Option<String>,
    /// Comma-separated `WxH` list, e.g. `1280x720,640x360`. Omit to use the
    /// source resolution (or set `ladder=true`).
    pub(super) rungs: Option<String>,
    /// Derive a standard ABR ladder from the source instead of explicit rungs.
    pub(super) ladder: Option<bool>,
    pub(super) max_short_side: Option<u32>,
    pub(super) segment_seconds: Option<f32>,
    pub(super) crf: Option<u8>,
    /// Perceptual quality target: `visually_lossless`, `high`, `standard`,
    /// `low`, or `vmaf=N`. Not consulted when `crf` is set.
    pub(super) target: Option<String>,
    /// GOP length in frames (default: two seconds).
    pub(super) gop: Option<u32>,
    /// `auto` (default), `opus`, or `drop`.
    pub(super) audio: Option<String>,
    /// Target Opus bitrate for transcoded audio, e.g. `240k`.
    pub(super) audio_bitrate: Option<String>,
    /// Audio filter chain, e.g. `channelmap=FL-FL|FR-FR:stereo`.
    pub(super) audio_filter: Option<String>,
    /// Subtitle tracks to carry: `all` (default), `none`, or a language list
    /// such as `eng,deu`.
    pub(super) subtitles: Option<String>,
    /// `sdr` (default), `hdr10`, `hlg`, or `passthrough`.
    pub(super) color: Option<String>,
    /// 4:4:4 → 4:2:0 chroma filter: `box` (default) or `lanczos`.
    pub(super) chroma_downsample: Option<String>,
    /// `auto` (default), `8bit`, or `10bit`.
    pub(super) pixel_format: Option<String>,
    /// Multi-GPU single-file chunk seam quality: `parallel` (default) or
    /// `constqp`. (`serial` still parses, as the older spelling of
    /// `encode=single`.)
    pub(super) seam: Option<String>,
    pub(super) max_fps: Option<f64>,
    pub(super) gpu: Option<u32>,
    /// The encode plan: `all` (default), `per-rung`, `single`, `gpu:N`,
    /// `family:nvidia|amd|intel`. Wins over `gpu`.
    pub(super) encode: Option<String>,
    /// The decode plan: `auto` (default), `whole`, `fastest`, `gpu:N`,
    /// `ranges:N`.
    pub(super) decode: Option<String>,
    /// Video filter chain, e.g. `crop=1280:720,hflip`.
    pub(super) filter: Option<String>,
    /// Block until the job finishes and return the artifact directly.
    pub(super) sync: Option<bool>,
}

impl TranscodeParams {
    /// Map the (string) query/JSON params onto the canonical
    /// [`TranscodeSettings`] using the shared `settings::parse_*` vocabulary —
    /// so the API doesn't carry its own copy of the field/spec logic.
    pub(super) fn into_settings(&self) -> Result<TranscodeSettings> {
        use crate::settings::{
            parse_audio, parse_bit_depth, parse_color, parse_decode_plan, parse_encode_plan,
            parse_mode, parse_quality_target, parse_rung,
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
        if let Some(t) = &self.target {
            s.target = Some(parse_quality_target(t)?);
        }
        s.gop = self.gop;
        if let Some(a) = &self.audio {
            s.audio = Some(parse_audio(a)?);
        }
        if let Some(b) = &self.audio_bitrate {
            s.audio_bitrate = Some(crate::settings::parse_bitrate(b)?);
        }
        if let Some(f) = &self.audio_filter {
            s.audio_filters =
                codec::audio::filter::parse_chain(f).context("parsing audio_filter")?;
        }
        if let Some(sel) = &self.subtitles {
            s.subtitles = Some(crate::settings::parse_subtitles(sel)?);
        }
        if let Some(c) = &self.color {
            s.color = Some(parse_color(c)?);
        }
        if let Some(f) = &self.chroma_downsample {
            s.chroma_downsample = Some(crate::settings::parse_chroma_downsample(f)?);
        }
        if let Some(p) = &self.pixel_format {
            s.bit_depth = Some(parse_bit_depth(p)?);
        }
        if let Some(sm) = &self.seam {
            s.apply_seam(sm)?;
        }
        s.max_fps = self.max_fps;
        s.gpu = self.gpu;
        if let Some(e) = &self.encode {
            s.encode = Some(parse_encode_plan(e)?);
        }
        if let Some(d) = &self.decode {
            s.decode_policy = parse_decode_plan(d)?;
        }
        if let Some(f) = &self.filter {
            s.filters = codec::filter::parse_chain(f).context("parsing filter")?;
        }
        Ok(s)
    }
}

// ---------------------------------------------------------------------------
// Structured JSON request body
// ---------------------------------------------------------------------------

/// A `POST /v1/transcode` body sent as `application/json`. The spec is a
/// structured object (not query params); the media comes from a server-side
/// **file path** or **inline base64** instead of a streamed binary body, and
/// the output can be written to a server **file path** instead of held in RAM.
#[derive(Deserialize)]
pub(super) struct TranscodeRequest {
    /// Where to read the input media from (`path` or `base64`).
    pub(super) input: InputSource,
    /// Optional: write the result to a server path instead of keeping it in
    /// memory. A file for single-rung single-file; a directory for multi-rung
    /// or HLS.
    #[serde(default)]
    pub(super) output: Option<OutputTarget>,
    /// The output spec (structured form of the query params).
    #[serde(default)]
    pub(super) spec: SpecBody,
    /// Block until the job finishes (stream/summarize the result) instead of
    /// returning a job id immediately.
    #[serde(default)]
    pub(super) sync: bool,
}

/// The media source for a JSON request: exactly one of `path` / `base64`.
#[derive(Deserialize)]
pub(super) struct InputSource {
    /// A file path on the **server** to read the media from.
    #[serde(default)]
    path: Option<String>,
    /// The media inline, base64-encoded (standard alphabet).
    #[serde(default)]
    base64: Option<String>,
}

/// Where to write the result of a JSON request.
#[derive(Deserialize)]
pub(super) struct OutputTarget {
    /// A file path (single-file single-rung) or directory (multi-rung / HLS)
    /// on the **server**.
    pub(super) path: String,
}

/// The structured spec body (mirrors [`TranscodeParams`] but with `rungs` as a
/// real array). Converts into `TranscodeParams` so it reuses [`build_spec`].
#[derive(Deserialize, Default)]
pub(super) struct SpecBody {
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
    target: Option<String>,
    gop: Option<u32>,
    audio: Option<String>,
    /// Target Opus bitrate for transcoded audio, e.g. `"240k"`.
    audio_bitrate: Option<String>,
    /// Audio filter chain, e.g. `"channelmap=FL-FL|FR-FR:stereo"`.
    audio_filter: Option<String>,
    /// Subtitle tracks to carry: `"all"` (default), `"none"`, or `"eng,deu"`.
    subtitles: Option<String>,
    color: Option<String>,
    /// `auto` | `8bit` | `10bit` (accepts the legacy key `pixel_format` too).
    #[serde(alias = "pixel_format")]
    bit_depth: Option<String>,
    seam: Option<String>,
    max_fps: Option<f64>,
    gpu: Option<u32>,
    encode: Option<String>,
    decode: Option<String>,
    /// Video filters — a chain string (`"crop=1280:720,hflip"`) or a structured
    /// list of objects (`[{"crop":{"w":1280,"h":720}},"hflip"]`).
    filter: Option<codec::filter::FilterSpec>,
}

impl SpecBody {
    pub(super) fn into_params(self) -> TranscodeParams {
        TranscodeParams {
            mode: self.mode,
            codec: self.codec,
            rungs: (!self.rungs.is_empty()).then(|| self.rungs.join(",")),
            ladder: self.ladder,
            max_short_side: self.max_short_side,
            segment_seconds: self.segment_seconds,
            crf: self.crf,
            target: self.target,
            gop: self.gop,
            audio: self.audio,
            audio_bitrate: self.audio_bitrate,
            audio_filter: self.audio_filter,
            subtitles: self.subtitles,
            color: self.color,
            pixel_format: self.bit_depth,
            seam: self.seam,
            max_fps: self.max_fps,
            gpu: self.gpu,
            encode: self.encode,
            decode: self.decode,
            // Collapse the structured-or-string FilterSpec to the chain string
            // (TranscodeParams is the string-keyed query form; into_settings
            // re-parses it). Round-trips losslessly via Display.
            filter: self.filter.map(|f| f.to_chain()),
            sync: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Input helpers
// ---------------------------------------------------------------------------

/// Read the media for a JSON request from its `path` or `base64` field.
pub(super) fn read_input(src: &InputSource) -> Result<Bytes, ApiError> {
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
pub(super) fn resolve_path(p: &str, must_exist: bool) -> Result<std::path::PathBuf, ApiError> {
    let path = std::path::PathBuf::from(p);
    let root = std::env::var_os("RIVET_FILE_ROOT").map(std::path::PathBuf::from);

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
pub(super) fn base64_decode(s: &str) -> Result<Vec<u8>> {
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
