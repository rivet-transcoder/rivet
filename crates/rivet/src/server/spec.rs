//! Request parameter types, JSON body types, and spec-parsing helpers.
//!
//! Converts HTTP query parameters and the structured JSON request body into the
//! canonical [`TranscodeSettings`] used by the rest of the rivet engine.

use anyhow::{Context, Result};
use axum::body::Bytes;
use serde::Deserialize;

use crate::settings::{SettingValue, TranscodeSettings};

use super::ApiError;

// ---------------------------------------------------------------------------
// Query-parameter struct
// ---------------------------------------------------------------------------

#[derive(Deserialize, Default, Clone)]
pub(super) struct TranscodeParams {
    /// `single` (default), `hls`, or `audio` (the audio alone; see
    /// `audio_container`).
    pub(super) mode: Option<String>,
    /// Output video codec: `av1` (default), `h264`, `h265`, `vp9`, `vp8`,
    /// `mpeg2`, `mpeg4`, `prores` or `prores-<profile>`.
    pub(super) codec: Option<String>,
    /// Comma-separated `WxH` list, e.g. `1280x720,640x360`; `WxH@RATE`
    /// (`1280x720@3M`) codes that rung to a bitrate. Omit to use the source
    /// resolution (or set `ladder=true`).
    pub(super) rungs: Option<String>,
    /// How the source meets each rung's box (a maximum, not the output
    /// size): `contain` (default), `cover`, `pad` or `stretch`. A rung's own
    /// `:FIT` wins (`1080x1920:cover:fixed`).
    pub(super) fit: Option<String>,
    /// `auto` (default: a box turns to the source's orientation) or `fixed`.
    pub(super) orientation: Option<String>,
    /// Let a rung be larger than the source (default `false`).
    pub(super) upscale: Option<bool>,
    /// Derive a standard ABR ladder from the source instead of explicit rungs.
    pub(super) ladder: Option<bool>,
    /// The ladder's largest short side: pixels, or `standard` (1080, the
    /// default).
    pub(super) max_short_side: Option<SettingValue>,
    pub(super) segment_seconds: Option<f32>,
    pub(super) crf: Option<u8>,
    /// Perceptual quality target: `visually_lossless`, `high`, `standard`,
    /// `low`, or `vmaf=N`. Not consulted when `crf` is set.
    pub(super) target: Option<String>,
    /// GOP length: frames (`48`) or seconds of output (`2s`, `1.5s`);
    /// default two seconds, which `2s` states.
    pub(super) gop: Option<SettingValue>,
    /// Video bitrate for every rung without its own `@RATE`, e.g. `3M`: code
    /// to a rate rather than to `target` (software H.264 / H.265).
    pub(super) video_bitrate: Option<String>,
    /// Coded picture buffer for the bitrate rungs, e.g. `1s` / `500ms`, `0`
    /// for none.
    pub(super) video_buffer: Option<String>,
    /// Rate mode for the bitrate rungs: `average` (default, `abr`) or `cbr`
    /// (`constant`) — a constant rate within the buffer, coded by the GPU
    /// encoders and the software H.264 / H.265 encoder; a `cbr` rung with no rate of its own takes `video_bitrate`,
    /// else the engine's default for its codec, size and frame rate.
    pub(super) rate_mode: Option<String>,
    /// `auto` (default), `opus`, `mp3`, `aac`, `he-aac`, `he-aacv2`,
    /// `vorbis`, `ac3`, `eac3`, `dts`, `flac`, `alac`, or `drop`.
    pub(super) audio: Option<String>,
    /// Target bitrate for transcoded audio, e.g. `240k` (MP3: one of the
    /// MPEG-1 Layer III rates, 32k..320k; AC-3 and DTS: their tables').
    pub(super) audio_bitrate: Option<String>,
    /// Vorbis quality, -1 to 10 (default 5).
    pub(super) audio_quality: Option<String>,
    /// Output channel layout: `source` (default), `mono`, `stereo`, `5.1`,
    /// `7.1`. Downmixes; never upmixes.
    pub(super) audio_channels: Option<String>,
    /// HLS: a stereo downmix rendition beside a surround one.
    pub(super) audio_stereo_fallback: Option<bool>,
    /// Bit depth of FLAC / ALAC output: `source` (default), `16` or `24`.
    pub(super) audio_bit_depth: Option<String>,
    /// An HE-AAC source: `auto` (default), `passthrough` or `core`.
    pub(super) he_aac: Option<String>,
    /// Source audio codecs that may not be decoded, comma-separated
    /// (`aac`, `mp3`, …). Empty / absent restricts nothing.
    pub(super) audio_decode_deny: Option<String>,
    /// Source metadata to carry into the output, comma-separated
    /// (`location`, `device`, `capture_time`, `descriptive`).
    pub(super) metadata_keep: Option<String>,
    /// FLAC compression effort: `fast`, `default` or `best`.
    pub(super) flac_compression: Option<String>,
    /// The file of an audio-only output: `auto` (default: follows the
    /// codec), `mp3`, `flac`, `mp4` or `ogg`.
    pub(super) audio_container: Option<String>,
    /// The file of a single-file output: `mp4`, `mov` or `webm` (default:
    /// the codec's own — `mov` for ProRes, `webm` for VP8 / VP9).
    pub(super) container: Option<String>,
    /// The ProRes profile with `codec=prores`: `proxy`, `lt`, `422`, `hq`,
    /// `4444`, `4444xq`.
    pub(super) prores_profile: Option<String>,
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
    /// Cap the output frame rate: a rate, or `source` (default: no cap).
    pub(super) max_fps: Option<SettingValue>,
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
    /// Optional hooks this job runs besides the required ones, by name,
    /// comma-separated (`GET /v1/hooks` lists them).
    pub(super) hooks: Option<String>,
}

/// `a,b` → `["a", "b"]`, blanks dropped.
pub(super) fn hook_names(list: Option<&str>) -> Vec<String> {
    list.map(|l| l.split(',').map(str::trim).filter(|n| !n.is_empty()).map(str::to_string).collect())
        .unwrap_or_default()
}

impl TranscodeParams {
    /// Map the (string) query/JSON params onto the canonical
    /// [`TranscodeSettings`] using the shared `settings::parse_*` vocabulary —
    /// so the API doesn't carry its own copy of the field/spec logic.
    pub(super) fn to_settings(&self) -> Result<TranscodeSettings> {
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
        if let Some(f) = &self.fit {
            s.apply_kv("fit", f)?;
        }
        if let Some(o) = &self.orientation {
            s.apply_kv("orientation", o)?;
        }
        s.upscale = self.upscale.unwrap_or(false);
        s.ladder = self.ladder.unwrap_or(false);
        if let Some(v) = &self.max_short_side {
            s.apply_kv("max-short-side", v.as_str())?;
        }
        s.segment_seconds = self.segment_seconds;
        s.crf = self.crf;
        if let Some(t) = &self.target {
            s.target = Some(parse_quality_target(t)?);
        }
        if let Some(v) = &self.gop {
            s.apply_kv("gop", v.as_str())?;
        }
        if let Some(a) = &self.audio {
            s.audio = Some(parse_audio(a)?);
        }
        if let Some(b) = &self.audio_bitrate {
            s.audio_bitrate = crate::settings::parse_bitrate_or_standard(b).context("audio_bitrate")?;
        }
        if let Some(c) = &self.audio_channels {
            s.audio_channels = Some(crate::settings::parse_audio_channels(c)?);
        }
        s.audio_stereo_fallback = self.audio_stereo_fallback.unwrap_or(false);
        for (key, value) in [
            ("audio-bit-depth", &self.audio_bit_depth),
            ("he-aac", &self.he_aac),
            ("audio-decode-deny", &self.audio_decode_deny),
            ("metadata-keep", &self.metadata_keep),
            ("flac-compression", &self.flac_compression),
            ("audio-container", &self.audio_container),
            ("audio-quality", &self.audio_quality),
            ("container", &self.container),
            ("prores-profile", &self.prores_profile),
        ] {
            if let Some(v) = value {
                s.apply_kv(key, v)?;
            }
        }
        if let Some(b) = &self.video_bitrate {
            s.video_bitrate = crate::settings::parse_bitrate_or_standard(b).context("video_bitrate")?;
        }
        if let Some(b) = &self.video_buffer {
            s.video_buffer_ms = Some(crate::settings::parse_buffer(b).context("video_buffer")?);
        }
        if let Some(m) = &self.rate_mode {
            s.rate_mode = Some(crate::settings::parse_rate_mode(m)?);
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
        if let Some(v) = &self.max_fps {
            s.apply_kv("max-fps", v.as_str())?;
        }
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
    /// Optional hooks this job runs besides the required ones, by name.
    #[serde(default)]
    pub(super) hooks: Vec<String>,
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
    /// Output video codec: `av1` (default), `h264`, `h265`, `vp9`, `vp8`,
    /// `mpeg2`, `mpeg4`, `prores` or `prores-<profile>`.
    codec: Option<String>,
    /// Explicit rungs as `["1280x720", "640x360"]`; `"1280x720@3M"` codes
    /// that rung to a bitrate.
    #[serde(default)]
    rungs: Vec<String>,
    /// `"contain"` (default), `"cover"`, `"pad"` or `"stretch"`.
    fit: Option<String>,
    /// `"auto"` (default) or `"fixed"`.
    orientation: Option<String>,
    /// Let a rung be larger than the source (default `false`).
    upscale: Option<bool>,
    ladder: Option<bool>,
    max_short_side: Option<SettingValue>,
    segment_seconds: Option<f32>,
    crf: Option<u8>,
    target: Option<String>,
    /// Frames (`48`) or seconds of output (`"2s"`).
    gop: Option<SettingValue>,
    /// Video bitrate for every rung without its own `@RATE`, e.g. `"3M"`.
    video_bitrate: Option<String>,
    /// Coded picture buffer for the bitrate rungs, e.g. `"1s"`.
    video_buffer: Option<String>,
    /// Rate mode for the bitrate rungs: `"average"` (default) or `"cbr"`.
    rate_mode: Option<String>,
    audio: Option<String>,
    /// Target bitrate for transcoded audio, e.g. `"240k"`.
    audio_bitrate: Option<String>,
    /// Vorbis quality, -1 to 10: `"6"` (or a number).
    audio_quality: Option<SettingValue>,
    /// Output channel layout: `"source"` (default), `"mono"`, `"stereo"`,
    /// `"5.1"`, `"7.1"`.
    audio_channels: Option<String>,
    /// HLS: a stereo downmix rendition beside a surround one.
    audio_stereo_fallback: Option<bool>,
    /// FLAC / ALAC bit depth: `"source"` (default), `"16"` or `"24"`.
    audio_bit_depth: Option<String>,
    /// An HE-AAC source: `"auto"` (default), `"passthrough"` or `"core"`.
    he_aac: Option<String>,
    /// Source audio codecs that may not be decoded: `"aac"`, `"aac,mp3"`, ….
    audio_decode_deny: Option<String>,
    /// Source metadata to carry: `"location,device"`, ….
    metadata_keep: Option<String>,
    /// FLAC compression effort: `"fast", `"default"` or `"best"`.
    flac_compression: Option<String>,
    /// The file of an audio-only output: `"auto"`, `"mp3"`, `"flac"`, `"mp4"`
    /// or `"ogg"`.
    audio_container: Option<String>,
    /// The file of a single-file output: `"mp4"`, `"mov"` or `"webm"`.
    container: Option<String>,
    /// The ProRes profile with `"codec": "prores"`.
    prores_profile: Option<String>,
    /// Audio filter chain, e.g. `"channelmap=FL-FL|FR-FR:stereo"`.
    audio_filter: Option<String>,
    /// Subtitle tracks to carry: `"all"` (default), `"none"`, or `"eng,deu"`.
    subtitles: Option<String>,
    color: Option<String>,
    /// `auto` | `8bit` | `10bit` (accepts the legacy key `pixel_format` too).
    #[serde(alias = "pixel_format")]
    bit_depth: Option<String>,
    seam: Option<String>,
    /// A frame rate cap, or `"source"`.
    max_fps: Option<SettingValue>,
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
            fit: self.fit,
            orientation: self.orientation,
            upscale: self.upscale,
            ladder: self.ladder,
            max_short_side: self.max_short_side,
            segment_seconds: self.segment_seconds,
            crf: self.crf,
            target: self.target,
            gop: self.gop,
            video_bitrate: self.video_bitrate,
            video_buffer: self.video_buffer,
            rate_mode: self.rate_mode,
            audio: self.audio,
            audio_bitrate: self.audio_bitrate,
            audio_quality: self.audio_quality.map(|q| q.0),
            audio_channels: self.audio_channels,
            audio_stereo_fallback: self.audio_stereo_fallback,
            audio_bit_depth: self.audio_bit_depth,
            he_aac: self.he_aac,
            audio_decode_deny: self.audio_decode_deny,
            metadata_keep: self.metadata_keep,
            flac_compression: self.flac_compression,
            audio_container: self.audio_container,
            container: self.container,
            prores_profile: self.prores_profile,
            audio_filter: self.audio_filter,
            subtitles: self.subtitles,
            color: self.color,
            pixel_format: self.bit_depth,
            // The JSON body has no chroma key: `None` is the default (box),
            // exactly what every other surface does when the key is absent.
            chroma_downsample: None,
            seam: self.seam,
            max_fps: self.max_fps,
            gpu: self.gpu,
            encode: self.encode,
            decode: self.decode,
            // Collapse the structured-or-string FilterSpec to the chain string
            // (TranscodeParams is the string-keyed query form; to_settings
            // re-parses it). Round-trips losslessly via Display.
            filter: self.filter.map(|f| f.to_chain()),
            sync: None,
            hooks: None,
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
