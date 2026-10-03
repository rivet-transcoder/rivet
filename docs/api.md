# rivet HTTP API reference

A small axum webserver (behind the `server` feature) so another application can
**signal rivet to transcode** over the network. It runs the same configurable
engine as the [CLI](cli.md) — see [pipeline & architecture](pipeline.md) for how
that engine works internally: POST media + an output spec, rivet transcodes to
AV1 and reports per-rung progress, and you fetch the artifacts.

```sh
cargo build --release --features server,nvidia   # the API + an AV1 encoder
rivet serve --addr 0.0.0.0:8080
```

Interactive documentation is served live:

| Path | What |
|------|------|
| `/` | landing page linking the three below |
| `/swagger` | Swagger UI |
| `/redoc` | Redoc |
| `/openapi.json` | the OpenAPI 3.0 document (source of truth) |

(The UI pages load Swagger/Redoc JS from a CDN; the spec itself is served
locally. An airgapped deployment can vendor the JS.)

---

## Concepts

- **Submit → poll → fetch.** `POST /v1/transcode` returns `202 { job_id }` and
  runs asynchronously. Poll `GET /v1/jobs/{id}` for status + per-rung progress,
  then download the artifact(s). Pass `"sync": true` in a JSON body (or
  `?sync=true` with a binary body) to block and get the result in one call.
- **Two request shapes** (pick whichever fits):
  - **Structured JSON body** (`Content-Type: application/json`) — a
    [`TranscodeRequest`](#json-body): `input` from a server **file path** or inline
    **base64**, an optional server **`output.path`**, and a structured **`spec`**.
    Streaming the media is *not* required.
  - **Streamed binary body** (`application/octet-stream`) — the raw media bytes
    (up to 4 GiB), with the output spec in the **query params**
    ([table](#transcode-query-parameters)). This is the streaming option.
  - With a JSON body, everything comes from the body: query parameters
    (`sync` and `hooks` included) are ignored.
- **Server-side file I/O.** When the JSON body names an input/output `path`, the
  server reads/writes its own filesystem (no upload/download). Set
  `RIVET_FILE_ROOT` to sandbox those paths to a directory; otherwise any path is
  allowed (the server binds localhost by default — treat it as trusted-local).
- The output spec — JSON `spec`, query params, the CLI flags, and the IPC
  `key=value` header — are all the same canonical knob set, so they map 1:1.

---

## Endpoints

### `GET /v1/health`

Liveness, detected GPUs, and this build's output capabilities.

```sh
curl -s http://localhost:8080/v1/health
```
```json
{
  "status": "ok",
  "service": "rivet",
  "gpus": [{ "index": 0, "vendor": "Nvidia", "name": "NVIDIA GeForce RTX 3090" }],
  "output_caps": {
    "max_bit_depth": 8, "hdr": false,
    "by_codec": [
      { "codec": "av1",  "max_bit_depth": 10, "hdr": true,
        "backends": [{ "backend": "nvenc", "max_bit_depth": 10, "hdr": true }] },
      { "codec": "h264", "max_bit_depth": 8,  "hdr": false,
        "backends": [{ "backend": "nvenc", "max_bit_depth": 8, "hdr": false }] },
      { "codec": "h265", "max_bit_depth": 10, "hdr": true,
        "backends": [{ "backend": "nvenc", "max_bit_depth": 10, "hdr": true }] }
    ]
  }
}
```

A job is validated against the caps for **its own codec**: `by_codec` is that
answer, and the one to read. AV1 is 10-bit HDR on `nvidia` / `amd` / `qsv`
and on the software AV1 tier (backend `av1`, `av1-sw-fallback`: it writes the
colour description into the sequence header and the HDR10 metadata into
metadata OBUs), H.264 only on `h26x-fallback` (no hardware backend has a
10-bit H.264 encoder). `rivet capabilities --json` reports the
same block under `encode.by_codec`.

`max_bit_depth` / `hdr` beside it are what **every** output codec meets — the
lowest depth in `by_codec`, and `hdr` only when every codec has it — so a job
asking for no more passes validation whichever codec it names.

> **Changed 2026-09-14.** These two fields used to be the union over the
> compiled encoders: 10-bit HDR when any codec had it. That claimed 10-bit HDR
> AV1 on a build whose only encoder is `h26x` (no AV1 encoder at all) and
> 10-bit H.264 on an `nvidia` build. On such builds they now read 8 / `false`;
> on a build where every codec is 10-bit HDR they read 10 / `true`, as before.
> A client that wants one codec's answer reads `by_codec`.

### `POST /v1/probe`

Body = media bytes → JSON media info (no transcode).

```sh
curl -s -X POST --data-binary @input.mkv http://localhost:8080/v1/probe
```
```json
{ "video_codec": "h264", "width": 1920, "height": 1080, "frame_rate": 30.0, "duration": 12.5 }
```

### `POST /v1/transcode`

Returns `202 { "job_id", "status": "queued" }` and runs asynchronously. With
`sync` it blocks and returns `200` with the single-file artifact itself
(`video/mp4`, `video/quicktime` or `video/webm`, or `audio/mpeg` / `audio/flac` / `audio/mp4` for audio-only
output) when the job made exactly one file held in memory, or else the
[job status](#get-v1jobsid) JSON: several rungs (fetch each from its
`artifacts[].url`), output written to `output.path`, or an HLS package.

<a id="json-body"></a>
**JSON body** (`application/json`) — point at a server file, no upload:

```sh
curl -s -X POST http://localhost:8080/v1/transcode \
  -H 'Content-Type: application/json' \
  -d '{
        "input":  { "path": "/data/in.mkv" },
        "output": { "path": "/data/out.mp4" },
        "spec":   { "mode": "single", "rungs": ["1280x720"], "crf": 28, "color": "sdr" },
        "sync":   true
      }'
```

Body fields:

| Field | Notes |
|-------|-------|
| `input.path` | a file path **on the server** to read the media from |
| `input.base64` | …or the media inline, base64-encoded (set exactly one of path/base64) |
| `output.path` | optional: write the result to a server path (a file for single-rung single-file; a directory for multi-rung / HLS). Omit to keep it in memory / stream it back |
| `spec` | the structured output spec — the query params below, except `sync` and `hooks` (top-level fields here) and `chroma_downsample` (query only). `rungs` is an array (`["1280x720", "640x360@1M"]`); bit depth is `bit_depth` (`pixel_format` is accepted too); `gop`, `max_fps` and `max_short_side` take a number or a word; `filter` takes a chain string or a structured list (see [Video filters](filters/README.md)). Unknown keys are ignored |
| `sync` | `true`: block until done; returns the artifact (no `output.path`) or the job status JSON |
| `hooks` | optional server hooks to run on this job, by name: `["a", "b"]`. Required hooks always run. See [`GET /v1/hooks`](#get-v1hooks) |

**Binary body** (`application/octet-stream`) — stream the media, spec in the query:

```sh
job=$(curl -s --data-binary @input.mkv \
      "http://localhost:8080/v1/transcode?mode=single&crf=28&audio=opus" \
      | jq -r .job_id)
```

#### Transcode query parameters

| Param | Values / default | Notes |
|-------|------------------|-------|
| `mode` | `single` *(default)*, `hls`, `audio` | output shape; `audio` is the audio alone as one file — an `.mp3` (`audio/mpeg`), a `.flac`, an `.m4a` or an Ogg file (`audio/ogg`), the codec's own unless `audio_container` says (also what a `single` job of an input with no video becomes). `image` is refused `400`: stills are [`rivet image`](cli.md#rivet-image) |
| `codec` | `av1` *(default)*, `h264`, `h265`, `vp9`, `vp8`, `mpeg2`, `mpeg4`, `prores` / `prores-proxy` / `-lt` / `-422` / `-hq` / `-4444` / `-4444xq` | output video codec; `vp9`…`prores` are rivet's own software encoders (VP9 also as HLS, the rest single-file only) |
| `container` | `mp4`, `mov`, `webm`; default the codec's own | the file of a single-file output (`video/mp4`, `video/quicktime` or `video/webm` in a synced response); a codec in a file that does not carry it is refused (400) |
| `prores_profile` | `proxy`, `lt`, `422`, `hq`, `4444`, `4444xq` | the ProRes profile with `codec=prores` |
| `rungs` | `WxH,WxH…` | comma-separated, e.g. `1280x720,640x360`. Each size is a maximum box the source is fitted into (see `fit`). Omit for source resolution. `WxH@RATE` (`1280x720@3M`) codes that rung to a bitrate, `WxH@standard` gives it the rate it would have with none named anywhere; `:FIT`, `:auto`/`:fixed` and `:upscale`/`:no-upscale` set that rung's own fitting (`1080x1920:cover:fixed`). |
| `fit` | `contain`/`cover`/`pad`/`stretch` | how the source meets each box — keep its shape inside (default), fill and centre-crop, black bars to exactly the box, or stretch to it |
| `orientation` | `auto`/`fixed` | `auto` (default): a box turns to the source's orientation |
| `upscale` | `true`/`false` | let a rung be larger than the source (default `false`). The job status's `renditions` lists each requested rung's box, its output size, and which rung it merged into when it came out the same as another |
| `ladder` | `true`/`false` | derive a standard ABR ladder instead of `rungs` |
| `max_short_side` | integer or `standard` | cap the ladder's short side; `standard` states the default, 1080 |
| `segment_seconds` | number (default `4`) | HLS segment length |
| `crf` | integer | constant rate factor (names the quantiser; `target` is then not consulted) |
| `target` | `visually_lossless`, `high`, `standard` *(default)*, `low`, `vmaf=N` | perceptual quality target for every rung — same words and meaning as the CLI's `--target` |
| `gop` | integer or string | GOP length for every rung: frames (`48`) or seconds of output (`"2s"`, `"1.5s"`); default two seconds, which `"2s"` states; same meaning as the CLI's `--gop` |
| `video_bitrate` | string | bitrate for every rung without its own `@RATE`, e.g. `3M`: the rung is coded to a rate, not to `target`; `standard` states the default, none. An average rate (the default `rate_mode`) is coded by the software encoders only (H.264 / H.265, AV1, VP9, MPEG-2, MPEG-4 Part 2) — a job whose encode pool is GPUs is refused by name; a constant one by the GPU encoders and the software H.264 / H.265 encoder. Same meaning as the CLI's `--video-bitrate` |
| `video_buffer` | string | coded picture buffer for the bitrate rungs, e.g. `500ms` (`0` for none; default `1s`); as the CLI's `--video-buffer` |
| `rate_mode` | `average` *(default; `abr`)*, `cbr` *(`constant`)* | how the bitrate rungs are coded: `cbr` is a constant rate within the buffer (QSV, NVENC, AMF — AV1 included — and the software H.264 / H.265 encoder; not the software AV1 or VP9 encoders, which refuse it by name). A `cbr` rung with no rate of its own takes `video_bitrate`, else a default for its codec, size and frame rate; as the CLI's `--rate-mode` |
| `video_speed` | `draft`, `standard` *(default)*, `archive` | encoder effort for every rung, mapped by each encoder onto its own presets (NVENC P5 / P6 / P7; software VP9 from fixed partitions to a searched one; software AV1 its effort, speed 8 / 6 / 4, and motion search range); an `encode-policy` `speed=` word wins. The same field in the JSON `spec`. As the CLI's `--video-speed` |
| `audio` | `auto` *(default)*, `opus`, `mp3`, `aac`, `he-aac`, `he-aacv2`, `vorbis`, `ac3`, `eac3`, `dts`, `flac`, `alac`, `drop` | audio policy, every encoder rivet's own (`opus`: MP4 / MOV / WebM, HLS, an Ogg file; `mp3`: CBR MP3, single-file or `audio` mode, not HLS; `aac` / `he-aac` / `he-aacv2`: AAC-LC, HE-AAC, HE-AAC v2 — single-file, HLS, an `.m4a`; `vorbis`: WebM or an Ogg file, by `audio_quality`; `ac3` / `eac3` / `dts`: up to 5.1, single-file, HLS, an `.m4a`; `flac` / `alac`: [lossless](lossless-audio.md)) |
| `audio_quality` | number or string, `-1` … `10` | Vorbis quality (default 5); `audio=vorbis` only, which takes no `audio_bitrate` |
| `audio_bit_depth` | `source` *(default)*, `16`, `24` | bit depth of `flac` / `alac` output |
| `he_aac` | `auto` *(default)*, `passthrough`, `core` | an HE-AAC source, which rivet decodes in full (SBR at the full rate, parametric stereo to two channels): `auto` treats it as any AAC track (passed through where it can be, decoded where the job needs PCM or another codec); `passthrough` never decodes it; `core` decodes only its AAC-LC core (half the rate, lower bandwidth) |
| `audio_decode_deny` | comma list of `aac`, `ac3`, `alac`, `dts`, `eac3`, `flac`, `mp2`, `mp3`, `opus`, `pcm`, `vorbis` | source audio codecs that may not be decoded (default none): a denied track is passed through where the output can carry it, and the job refused, naming the setting, where it needs the PCM; as the CLI's `--audio-decode-deny` |
| `metadata_keep` | comma list of `location` (or `location:approximate`), `capture_time` (or `capture_time:date`), `device` (or `device:all`), `descriptive`, `all`, `none` | the source's identifying metadata to carry into the output (default none: nothing identifying is written). Single-file and `audio`-mode output; HLS refuses it. As the CLI's `--metadata-keep` |
| `flac_compression` | `fast`, `default` *(default)*, `best` | FLAC compression effort |
| `audio_container` | `auto` *(default)*, `mp3`, `flac`, `mp4`, `ogg` | the file of an `audio`-mode output (`audio/mpeg`, `audio/flac`, `audio/mp4` or `audio/ogg` in the response); `auto` follows the codec: `.flac` for `flac`, Ogg for `opus` / `vorbis`, `.m4a` for `alac`, `aac`, `he-aac`, `he-aacv2`, `ac3`, `eac3`, `dts`, else `.mp3` |
| `audio_bitrate` | string | target for transcoded audio, e.g. `240k` (MP3: 32k … 320k on the MPEG-1 ladder; AC-3: A/52 Table 5.18's rates, 32k … 640k; E-AC-3 32k … 6144k; DTS: ETSI TS 102 114 Table 5-7's, 32k … 1536k); `standard` states the default, the codec's rate for the output layout |
| `audio_channels` | `source` *(default)*, `mono`, `stereo`, `5.1`, `7.1` | output channel layout; downmixes, never upmixes |
| `audio_stereo_fallback` | bool | HLS: a stereo downmix rendition beside a surround one |
| `audio_filter` | string | audio filter chain, e.g. `channelmap=FL-FL\|FR-FR:stereo` |
| `subtitles` | string | `all` (default) \| `none` \| a language list such as `eng,deu` — text subtitle tracks to carry (tx3g tracks in an MP4, WebVTT renditions in HLS) |
| `color` | `sdr` *(default)*, `hdr10`, `hlg`, `passthrough` | color / tonemap policy |
| `pixel_format` | `auto` *(default)*, `8bit`, `10bit` | output bit depth (`bit_depth` in the JSON `spec`) |
| `chroma_downsample` | `box` *(default)*, `lanczos` | 4:4:4 → 4:2:0 chroma filter for 4:4:4 sources. Query only: the JSON `spec` has no such field |
| `seam` | `parallel` *(default)*, `constqp` | multi-GPU single-file chunk-seam *quality* (`serial` still parses, as the older spelling of `encode=single`) |
| `encode` | `all` *(default)*, `per-rung`, `single`, `gpu:N`, `family:nvidia\|amd\|intel` | the encode plan — which cards, and how the work is laid across them; wins over `gpu`. Same words and meaning as the CLI's `--encode`: every surface interprets through [`rivet::settings`](../crates/rivet/src/settings.rs) |
| `decode` | `auto` *(default)*, `whole`, `fastest`, `gpu:N`, `ranges:N` | the decode plan — which card(s), and whether the decode is one pump or split into ranges |
| `max_fps` | number or string | cap output frame rate; `"source"` states the default, no cap |
| `gpu` | integer | pin encode/decode to a GPU index |
| `filter` | string | video filter chain, e.g. `crop=1280:720,hflip` (the JSON `spec` body also accepts a structured list — see [Video filters](filters/README.md)) |
| `sync` | `true`/`false` | block and return the artifact directly |
| `hooks` | string | optional server hooks to run on this job, comma-separated (JSON body: `"hooks": [...]`, top level). Required hooks always run; a name the server has not configured is `400`. See [`GET /v1/hooks`](#get-v1hooks) |

A request that the build can't satisfy (e.g. `color=hdr10` on a build with no
10-bit hardware encoder) is rejected `400` at submit time.

### `GET /v1/jobs/{id}`

Job status + per-rung progress + the output list.

```sh
curl -s "http://localhost:8080/v1/jobs/$job"
```
```json
{
  "job_id": "30a2c394-…",
  "mode": "single",
  "status": "completed",
  "progress": [
    { "rung_index": 0, "label": "720p", "width": 1280, "height": 720,
      "status": "completed", "percent": 100.0, "frames_done": 300, "message": null }
  ],
  "artifacts": [
    { "label": "720p", "width": 1280, "height": 720, "frames": 300,
      "bytes": 1048576, "url": "/v1/jobs/30a2c394-…/artifacts/720p", "output_path": null }
  ],
  "renditions": [
    { "label": "720p", "requested": { "width": 1280, "height": 720 },
      "output": { "width": 1280, "height": 720 }, "fit": "contain", "duplicate_of": null }
  ],
  "master_playlist": null,
  "error": null,
  "hooks": null
}
```

`mode` is `single` or `hls` (an audio-only job reports `single`). Each
`progress` entry's `status` is `pending`, `running`, `finalizing`, `completed`
or `failed`, and its `message` is why a failed rung failed (`null`
otherwise). An artifact's `url` is its download link when the bytes are held
in memory, the job's `files/` root for an HLS rendition, and `null` when it
was written to `output.path`, which `output_path` then names. `renditions` has
one entry per requested rung, in request order: the box asked for, the size
produced, and the rung it merged into when it came out the same as another.

`status` is `queued` → `running` → `completed` | `failed` | `rejected`. On
failure, `error` carries the message (e.g. "no AV1 encoder available on this
host"). `rejected` means a [hook](hooks.md) stopped the job. `error` names the
hook, the stage and the reason, and a `?sync=true` request gets `422`.

`hooks` is the job's hook report, readable while the job runs and after it
ends whatever the outcome: `job_id`, `rejected`, `rejection`, and `records`,
one per hook per event (`hook`, `kind`, `stage`, `subject`, `verdict`,
`reason`, `annotations`, `error`, `elapsed_ms`, `background`). It is `null`
when the job runs no hooks.

### `GET /v1/hooks`

The hooks this server runs (configured by the program embedding the server,
see [hooks.md](hooks.md#http-api)): `{ "hooks": [ { "name", "kind",
"description", "stages", "mode", "on_error", "required", "frames",
"artifact_kinds" } ] }`. `kind` is where the hook hooks in: `source`, `probe`,
`decoded-frame`, `encoder-frame`, `still`, `artifact`, `completed`, `failed`,
or `event` for a general hook at the stages it lists. `stages` lists
them; `mode` is `blocking` or `background`; `on_error` is `continue` or
`reject`; `frames` is the frame sampling (`every_frames`, `every_seconds`,
`max_frames`) for a frame hook and `null` otherwise; `artifact_kinds` lists
the artifacts an artifact hook sees and is `null` otherwise. A job runs every
`required` hook, plus the optional ones it names with `hooks=`.

### `GET /v1/jobs/{id}/artifacts/{label}`

Download a single-file rung's MP4 (`Content-Type: video/mp4`), or an
audio-only job's file (`audio/mpeg`, `audio/flac` or `audio/mp4`). Only
artifacts held in memory are served; one written to `output.path` is `404`.

```sh
curl -so 720p.mp4 "http://localhost:8080/v1/jobs/$job/artifacts/720p"
```

### `GET /v1/jobs/{id}/files/{*path}`

For HLS jobs, fetch a file from the output tree — the playlist and segments:

```sh
curl -s "http://localhost:8080/v1/jobs/$job/files/master.m3u8"
curl -so seg.m4s "http://localhost:8080/v1/jobs/$job/files/video/720p/seg-00001.m4s"
```

Served with the right content type (`application/vnd.apple.mpegurl`,
`video/iso.segment`, `video/mp4`; anything else, WebVTT included, as
`application/octet-stream`). A path with a `..` or empty component is
rejected `400`.

---

## Examples

Async (submit, poll, download):

```sh
curl -s http://localhost:8080/v1/health
job=$(curl -s --data-binary @input.mkv \
      "http://localhost:8080/v1/transcode?mode=single&crf=28" | jq -r .job_id)
# poll until status == completed
curl -s "http://localhost:8080/v1/jobs/$job" | jq .status
curl -so out.mp4 "http://localhost:8080/v1/jobs/$job/artifacts/720p"
```

Synchronous (single-file, single rung — the file comes back; with several
rungs the response is the job status JSON, whose `artifacts[].url` fetch each):

```sh
curl -s --data-binary @input.mkv \
     "http://localhost:8080/v1/transcode?sync=true" -o out.mp4
```

HLS ladder:

```sh
job=$(curl -s --data-binary @input.mkv \
      "http://localhost:8080/v1/transcode?mode=hls&ladder=true&segment_seconds=4" \
      | jq -r .job_id)
# after completion:
curl -s "http://localhost:8080/v1/jobs/$job/files/master.m3u8"
```

---

## Errors

JSON errors with the appropriate HTTP status:

```json
{ "error": "h264 at 10 bits (color=Hdr10, bit_depth=Auto) cannot be encoded: this build encodes h264 with nvenc (8-bit SDR). h264 at 10 bits needs the software tier (build with `h26x-fallback`) …" }
```

- `400 Bad Request` — empty body, non-media body, invalid JSON body, bad query
  params, a spec the build can't produce, a hook name the server has not
  configured, an input/output path that is missing or escapes
  `RIVET_FILE_ROOT`, or a `files/` path with `..`.
- `404 Not Found` — unknown/malformed job id, missing artifact or file.
- `422 Unprocessable Entity` — a job a hook rejected under `sync`.
- `500 Internal Server Error` — a job that failed under `sync`.

---

## Operational notes

- **In-memory registry.** Jobs and completed single-file artifacts are held in
  RAM until the process exits — this is a sidecar/worker, not a public CDN. For
  durable output, layer an uploader on top by watching `RungStatus::Completed`
  from a `ProgressSink` (object storage, a status queue, …) and run the engine
  via the library API directly.
- **GPU-only encode by default.** A host with no encode silicon for the chosen
  codec and no software fallback (`av1-sw-fallback` for AV1, `h26x-fallback` for
  H.264 / H.265) will accept jobs and report them `failed` with the encoder
  error. Check `/v1/health` `output_caps` first.
- **Pair with an encode feature.** `--features server` alone has no encoder;
  build `--features server,nvidia` (or `amd` / `qsv`, or `av1-sw-fallback` /
  `h26x-fallback` for software AV1 / H.264 / H.265) for your target.
