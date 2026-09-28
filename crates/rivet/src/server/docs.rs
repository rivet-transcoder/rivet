//! OpenAPI 3.0 specification document, HTML landing page, and documentation UI
//! constants served by the rivet HTTP API.

use serde_json::{Value, json};

// ---------------------------------------------------------------------------
// HTML constants
// ---------------------------------------------------------------------------

pub(super) const LANDING_HTML: &str = r#"<!DOCTYPE html><html><head><meta charset="utf-8">
<title>rivet transcode API</title><style>body{font:16px system-ui;margin:3rem auto;max-width:40rem}a{display:block;margin:.5rem 0}</style></head>
<body><h1>rivet transcode API</h1>
<p>Interactive documentation:</p>
<a href="/swagger">Swagger UI</a>
<a href="/redoc">Redoc</a>
<a href="/openapi.json">OpenAPI 3.0 document (JSON)</a>
<p>Quick check: <a href="/v1/health">/v1/health</a></p>
</body></html>"#;

pub(super) const SWAGGER_HTML: &str = r#"<!DOCTYPE html><html><head><meta charset="utf-8">
<title>rivet API — Swagger UI</title>
<link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist/swagger-ui.css"></head>
<body><div id="swagger-ui"></div>
<script src="https://unpkg.com/swagger-ui-dist/swagger-ui-bundle.js"></script>
<script>window.ui=SwaggerUIBundle({url:'/openapi.json',dom_id:'#swagger-ui'});</script>
</body></html>"#;

pub(super) const REDOC_HTML: &str = r#"<!DOCTYPE html><html><head><meta charset="utf-8">
<title>rivet API — Redoc</title><meta name="viewport" content="width=device-width,initial-scale=1"></head>
<body><redoc spec-url="/openapi.json"></redoc>
<script src="https://cdn.redoc.ly/redoc/latest/bundles/redoc.standalone.js"></script>
</body></html>"#;

// ---------------------------------------------------------------------------
// OpenAPI helpers
// ---------------------------------------------------------------------------

/// String query parameter for the transcode endpoint.
fn qp(name: &str, ty: &str, desc: &str) -> Value {
    json!({
        "name": name, "in": "query", "required": false,
        "schema": { "type": ty }, "description": desc
    })
}

/// `Health.output_caps.by_codec`: each output codec's capabilities on this
/// build and the backends behind them. Its own `json!`, because nested inside
/// the document it takes `json!` past the compiler's recursion limit.
fn health_by_codec_schema() -> Value {
    json!({ "type": "array", "items": { "type": "object", "properties": {
        "codec": { "type": "string", "enum": ["av1", "h264", "h265"] },
        "max_bit_depth": { "type": "integer" }, "hdr": { "type": "boolean" },
        "backends": { "type": "array", "items": { "type": "object", "properties": {
            "backend": { "type": "string", "enum": ["nvenc", "amf", "qsv", "rav1e", "h26x"] },
            "max_bit_depth": { "type": "integer" }, "hdr": { "type": "boolean" }
        } } }
    } } })
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
            "license": { "name": "Open Encoding Attribution License v1.0", "url": "https://github.com/rivet-transcoder/rivet/blob/develop/LICENSE.md" }
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
                        qp("mode", "string", "single (default), hls, or audio (the audio alone as one file: an .mp3, or for lossless audio a .flac or an .m4a; also what a single-file job of an input with no video becomes)"),
                        qp("rungs", "string", "Comma-separated WxH, e.g. 1280x720,640x360; WxH@RATE (1280x720@3M) codes that rung to a bitrate. Each size is a maximum box the source is fitted into (see fit), and may end in the rung's own :FIT, :auto|:fixed and :upscale|:no-upscale (1080x1920:cover:fixed). Omit for source resolution."),
                        qp("fit", "string", "How the source meets each rung's box: contain (default; inside the box, keeping the source's shape), cover (fill the box, centre-cropping the overflow), pad (contain, then black bars to exactly the box) or stretch (exactly the box, distorting the picture)."),
                        qp("orientation", "string", "auto (default): a box turns to the source's orientation, so 1920x1080 on a portrait source is 1080x1920; fixed: boxes are used as written."),
                        qp("upscale", "boolean", "Let a rung be larger than the source. Default false: a smaller source comes out at its own size, and rungs that collapse onto the same size are merged."),
                        qp("ladder", "boolean", "Derive a standard ABR ladder from the source."),
                        qp("max_short_side", "integer", "Cap the ladder's tallest rung's short side."),
                        qp("segment_seconds", "number", "HLS target segment length (default 4)."),
                        qp("crf", "integer", "Constant rate factor (encoder-native 0..255)."),
                        qp("speed", "integer", "Encoder speed preset."),
                        qp("video_bitrate", "string", "Bitrate for every rung without its own @RATE, e.g. 3M. An average rate (the default rate_mode) is coded by the software H.264 / H.265 encoder; a constant one (rate_mode=cbr) by the GPU encoders and the software H.264 / H.265 encoder."),
                        qp("video_buffer", "string", "Coded picture buffer for the bitrate rungs, e.g. 1s or 500ms (0 for none; one second by default). A cbr rung needs one."),
                        qp("rate_mode", "string", "average (default; abr) | cbr (constant): how the bitrate rungs spend their rate. cbr is a constant rate - the rate is also the maximum, an HRD buffer is declared and the encoder holds the rate, with filler where it pads - coded by QSV, NVENC and AMF for AV1, H.264 and H.265, and by the software H.264 / H.265 encoder (not rav1e). A cbr rung with no rate of its own takes video_bitrate, else a default by codec, short side and frame rate (H.264 at 30 fps: 2160p 16M, 1440p 9M, 1080p 5M, 720p 3M, 480p 1.2M, 360p 0.8M, 240p 0.4M; interpolated between; H.265 0.65x, AV1 0.5x; above 30 fps x(1 + (fps/30 - 1)/2), capped at 120 fps). An HLS cbr rendition's BANDWIDTH is its rate plus the audio. Refused beside crf, seam=constqp or video_buffer=0."),
                        qp("audio", "string", "auto (default) | opus | mp3 | aac | flac | alac | drop. mp3 is CBR MP3 for single-file MP4 and audio-only output (not HLS), and needs a build with the lame feature to encode; aac is AAC-LC from rivet's own encoder, for single-file MP4 and HLS. flac / alac are lossless: a source already in that codec is copied, anything decodable is encoded"),
                        qp("audio_bit_depth", "string", "source (default) | 16 | 24: bit depth of flac / alac output. source is 16 for a 16-bit or lossy source, else 24"),
                        qp("flac_compression", "string", "fast | default (default) | best: FLAC compression effort"),
                        qp("audio_container", "string", "auto (default) | mp3 | flac | mp4: the file of an audio-only output. auto is a native FLAC stream for audio=flac, an .m4a for audio=alac, else an .mp3"),
                        qp("audio_bitrate", "string", "Target for transcoded audio, e.g. 240k. Default: Opus from the channel layout (64k mono, 96k stereo, 320k 5.1, 416k 7.1); MP3 128k stereo, 64k mono (MP3 takes 32k..320k on the MPEG-1 ladder)."),
                        qp("audio_channels", "string", "source (default) | mono | stereo | 5.1 | 7.1. Downmixes (ITU-R BS.775, LFE dropped, normalised); asking for more channels than the source has is an error"),
                        qp("audio_stereo_fallback", "boolean", "HLS: add a stereo downmix rendition beside a surround one, in the same audio group (CHANNELS 2 and 6)"),
                        qp("audio_filter", "string", "Audio filter chain, e.g. channelmap=FL-FL|FR-FR:stereo"),
                        qp("subtitles", "string", "all (default) | none | a language list such as eng,deu"),
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
                        "mode": { "type": "string", "enum": ["single", "hls", "audio"] },
                        "rungs": { "type": "array", "items": { "type": "string", "example": "1280x720@3M" } },
                        "fit": { "type": "string", "enum": ["contain", "cover", "pad", "stretch"] },
                        "orientation": { "type": "string", "enum": ["auto", "fixed"] },
                        "upscale": { "type": "boolean" },
                        "ladder": { "type": "boolean" },
                        "max_short_side": { "type": "integer" },
                        "segment_seconds": { "type": "number" },
                        "crf": { "type": "integer" },
                        "speed": { "type": "integer" },
                        "video_bitrate": { "type": "string", "example": "3M" },
                        "video_buffer": { "type": "string", "example": "1s" },
                        "rate_mode": { "type": "string", "enum": ["average", "abr", "cbr", "constant"] },
                        "audio": { "type": "string", "enum": ["auto", "opus", "mp3", "aac", "flac", "alac", "drop"] },
                        "audio_bit_depth": { "type": "string", "enum": ["source", "16", "24"] },
                        "flac_compression": { "type": "string", "enum": ["fast", "default", "best"] },
                        "audio_container": { "type": "string", "enum": ["auto", "mp3", "flac", "mp4"] },
                        "audio_bitrate": { "type": "string", "example": "240k" },
                        "audio_channels": { "type": "string", "enum": ["source", "mono", "stereo", "5.1", "7.1"] },
                        "audio_stereo_fallback": { "type": "boolean" },
                        "audio_filter": { "type": "string", "example": "channelmap=FL-FL|FR-FR|FC-FC|LFE-LFE|SL-BL|SR-BR:5.1" },
                        "subtitles": { "type": "string", "example": "eng,deu" },
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
                        "max_bit_depth": { "type": "integer",
                            "description": "The bit depth every output codec reaches on this build (the lowest across by_codec)" },
                        "hdr": { "type": "boolean",
                            "description": "Whether every output codec produces HDR on this build; by_codec has each codec's answer" },
                        "by_codec": health_by_codec_schema()
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
                    "percent": { "type": "number" }, "frames_done": { "type": "integer" },
                    "message": { "type": "string", "nullable": true,
                        "description": "Why a failed rung failed: the whole error chain" }
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
