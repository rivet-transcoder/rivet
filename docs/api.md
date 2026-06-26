# rivet HTTP API reference

A small axum webserver (behind the `server` feature) so another application can
**signal rivet to transcode** over the network. It runs the same configurable
engine as the [CLI](cli.md): POST media + an output spec, rivet transcodes to
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
  then download the artifact(s). Pass `?sync=true` to block and get a
  single-file MP4 back in one call.
- **The output spec is query params** on `/v1/transcode`, mirroring the CLI
  flags exactly (see [the table](#transcode-query-parameters)).
- **The media is the request body** (`application/octet-stream`) — raw bytes,
  up to a 4 GiB ceiling.

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
  "output_caps": { "max_bit_depth": 8, "hdr": false }
}
```

`output_caps` reflects what the build can actually encode (8-bit unless the
`ffmpeg` feature is on) — useful for a client to decide whether to request HDR.

### `POST /v1/probe`

Body = media bytes → JSON media info (no transcode).

```sh
curl -s -X POST --data-binary @input.mkv http://localhost:8080/v1/probe
```
```json
{ "video_codec": "h264", "width": 1920, "height": 1080, "frame_rate": 30.0, "duration": 12.5 }
```

### `POST /v1/transcode`

Body = media bytes; output spec = query params. Returns `202 { job_id, status }`
and runs asynchronously (or, with `?sync=true`, blocks and returns the MP4).

```sh
job=$(curl -s --data-binary @input.mkv \
      "http://localhost:8080/v1/transcode?mode=single&crf=28&audio=opus" \
      | jq -r .job_id)
```

#### Transcode query parameters

| Param | Values / default | Notes |
|-------|------------------|-------|
| `mode` | `single` *(default)*, `hls` | output shape |
| `rungs` | `WxH,WxH…` | comma-separated, e.g. `1280x720,640x360`. Omit for source resolution. |
| `ladder` | `true`/`false` | derive a standard ABR ladder instead of `rungs` |
| `max_short_side` | integer | cap the ladder's short side |
| `segment_seconds` | number (default `4`) | HLS segment length |
| `crf` | integer | constant rate factor |
| `speed` | integer | encoder speed preset |
| `audio` | `auto` *(default)*, `opus`, `drop` | audio policy |
| `color` | `sdr` *(default)*, `hdr10`, `hlg`, `passthrough` | color / tonemap policy |
| `pixel_format` | `auto` *(default)*, `8bit`, `10bit` | output bit depth |
| `seam` | `parallel` *(default)*, `constqp`, `serial` | multi-GPU single-file chunk-seam handling (see the CLI's [Color & bit depth / GPU notes](cli.md#gpu-selection)) |
| `max_fps` | number | cap output frame rate |
| `gpu` | integer | pin encode/decode to a GPU index |
| `sync` | `true`/`false` | block and return the artifact directly |

A request that the build can't satisfy (e.g. `color=hdr10` on a build without
the `ffmpeg` feature) is rejected `400` at submit time.

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
      "status": "completed", "percent": 100.0, "frames_done": 300 }
  ],
  "artifacts": [
    { "label": "720p", "width": 1280, "height": 720, "frames": 300,
      "bytes": 1048576, "url": "/v1/jobs/30a2c394-…/artifacts/720p" }
  ],
  "master_playlist": null,
  "error": null
}
```

`status` is `queued` → `running` → `completed` | `failed`. On failure, `error`
carries the message (e.g. "no AV1 encoder available on this host").

### `GET /v1/jobs/{id}/artifacts/{label}`

Download a single-file rung's MP4 (`Content-Type: video/mp4`).

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
`video/iso.segment`, `video/mp4`). Path traversal (`..`) is rejected.

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

Synchronous (single-file, single rung):

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
{ "error": "10-bit output requested … build with the `nvidia`, `amd`, or `ffmpeg` feature" }
```

- `400 Bad Request` — empty body, non-media body, bad query params, or a spec
  the build can't produce.
- `404 Not Found` — unknown/malformed job id, missing artifact or file.
- `500 Internal Server Error` — a job that failed under `?sync=true`.

---

## Operational notes

- **In-memory registry.** Jobs and completed single-file artifacts are held in
  RAM until the process exits — this is a sidecar/worker, not a public CDN. For
  durable output, layer an uploader on top by watching `RungStatus::Completed`
  from a `ProgressSink` (object storage, a status queue, …) and run the engine
  via the library API directly.
- **GPU-only encode by default.** A host with no AV1-encode silicon and no
  `ffmpeg` feature will accept jobs and report them `failed` with the encoder
  error. Check `/v1/health` `output_caps` first.
- **Pair with an encode feature.** `--features server` alone has no encoder;
  build `--features server,nvidia` (or `amd` / `qsv` / `ffmpeg`) for your target.
