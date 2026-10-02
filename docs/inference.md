# Running models on a job's pictures

This guide is about running a machine-learning model (a detector, a
classifier, an embedding model, an OCR or moderation model) on the pictures
rivet decodes, while it transcodes them. rivet doesn't ship or depend on any
model or inference runtime. What it provides is the place to run one (the
[hook engine](hooks.md)), the pixels in a form models take
(`rivet::hooks::frame`), and the report your results land in.

| Read | For |
|------|-----|
| This page | How to design an inference integration: which point, which frames, getting pixels in, running the model, answering, getting results out, performance, deployment, testing. |
| [hooks.md](hooks.md) | The hook engine's reference: every kind, policy and type. |
| [hooks-cookbook.md](hooks-cookbook.md) | Sixteen small recipes, one per common job. |
| [hooks-yolo.md](hooks-yolo.md) | A complete worked example: YOLO object detection through ONNX Runtime on CPU, CUDA, DirectML and OpenVINO, with measurements. The code is [`examples/yolo`](../examples/yolo). |

The examples below are in Rust and assume:

```rust
use rivet::hooks::*;
use anyhow::Result;
```

## Contents

1. [The shape of an integration](#1-the-shape-of-an-integration)
2. [Where to run the model](#2-where-to-run-the-model)
3. [Which frames](#3-which-frames)
4. [Getting pixels into a model](#4-getting-pixels-into-a-model)
5. [Running the model](#5-running-the-model)
6. [Answering: annotations and verdicts](#6-answering-annotations-and-verdicts)
7. [Getting results out](#7-getting-results-out)
8. [Performance](#8-performance)
9. [Deployment](#9-deployment)
10. [Testing](#10-testing)
11. [Checklist](#11-checklist)

## 1. The shape of an integration

An integration is a type that holds the loaded model and implements one or
more hook traits. rivet calls it with each picture it selects, and it answers
with values to record and a verdict:

```rust
pub struct Classifier {
    model: std::sync::Mutex<MyModel>,      // your runtime's session
}

impl DecodedFrameHook for Classifier {
    fn sampling(&self) -> FrameSampling {
        FrameSampling::every_seconds(2.0)
    }

    fn on_decoded_frame(&self, _ctx: &HookContext, f: &FrameEvent) -> Result<HookOutcome> {
        let (input, _letterbox) = frame::planar_f32_letterboxed(&f.frame, 224, 224, [0, 0, 0])?;
        let scores = self.model.lock().unwrap().run(&input)?;
        let (label, score) = top1(&scores);
        Ok(HookOutcome::proceed().annotate("label", label).annotate("score", score))
    }
}

let hooks = Hooks::new().decoded_frames_with("classify", Classifier::load("model.onnx")?, HookPolicy::background());
let spec = rivet::OutputSpec::single_file(rungs).with_hooks(hooks);
```

Three things follow from that shape:

- **The model loads once**, when you build the hook, not per job or per
  frame. A `Hooks` value is shared by every job it's handed to (including
  every job on the HTTP server), so one loaded model serves all of them.
- **The hook is called from rivet's threads.** For video that's the decode
  thread, or several decode threads at once when a source is decoded in ranges
  on several GPUs, or the session's background worker. Hooks are
  `Send + Sync`; keep mutable state behind a `Mutex` or atomics.
- **Everything a hook says lands in the job's report**, one record per hook
  per picture, keyed by the picture's position in the source.

## 2. Where to run the model

The kinds that see pictures:

| Kind | Register with | Gets | Use it when |
|------|---------------|------|-------------|
| decoded frame | `Hooks::decoded_frames` | Video frames as decoded, turned upright, in the decoder's pixel format. Before tonemapping, the spec's filters and any scaling. | You want to know what the **source** shows, whatever the job outputs. One set of detections describes every rendition, in the source's own pixel coordinates. The usual choice. |
| encoder frame | `Hooks::encoder_frames` | Video frames as the encoders receive them: after the colour pipeline (HDR→SDR tonemapping, re-matrixing, bit depth) and the spec's filters (crop, pad, overlay, denoise), before each rung's scaling. Always `yuv420p` or `yuv420p10le`. | You want to judge what's **published**, after crops and overlays. Or the source is HDR and your model was trained on SDR pictures (see [HDR sources](#hdr-sources)). |
| still | `Hooks::stills` | Every picture of an image job: the decoded image, or each still taken from a video. Upright 8-bit RGBA. | Photos and thumbnails. A video job never calls it, and an image job never calls the frame kinds, so register both when you ingest both. |
| artifact | `Hooks::artifacts` | Each finished output: bytes, an HLS directory, or a file. | You want to check the **encoded** result (for instance, decode an image output and run a quality model on it). |

One value can be registered at several points. An `Arc` of any hook kind is
that kind, so share one loaded model:

```rust
let model = std::sync::Arc::new(Classifier::load("model.onnx")?);
let hooks = Hooks::new()
    .decoded_frames("classify", std::sync::Arc::clone(&model))
    .stills("classify-stills", model);
```

The probe and source kinds run before anything is decoded. They're the place
for cheap gates that save a model run, such as refusing a 4-hour upload before
decoding a single frame ([cookbook recipe 4](hooks-cookbook.md#4-limits-on-what-the-source-is)).

### HDR sources

`VideoFrame` carries its matrix (`color_space`: BT.601, BT.709 or BT.2020)
but not its transfer function. A decoded frame from an HDR source is 10-bit
BT.2020, encoded with PQ or HLG. `frame::rgb8` and the scaling helpers convert
the matrix and the range, not the transfer curve, so such a picture comes out
flat and dim. A model trained on SDR pictures will see something unlike its
training data. Either register on `encoder_frames`, which come after rivet's
HDR→SDR tonemapping when the output is SDR (the default), or apply a transfer
curve yourself. Neither the frame nor the probe event says whether a source is
HDR; a 10-bit `ColorSpace::Bt2020` frame almost always is.

## 3. Which frames

A frame hook picks its frames with `sampling()`:

```rust
FrameSampling::every_seconds(1.0)               // the first frame of each second (the default)
FrameSampling::every_frames(10)                 // every 10th frame
FrameSampling::every_seconds(0.5).max_frames(300)  // and stop after 300
FrameSampling::all()                            // every frame
```

- The first frame is always selected.
- `every_seconds` and `every_frames` combine as OR when both are set.
- `max_frames` is counted per hook, per job. Once every frame hook has had its
  `max_frames`, `Hooks::frames_exhausted()` is true.
- Indices are presentation order in the source, before any trim or
  frame-rate cap. A frame's `seconds` is its time from the start of its source.
  In a splice, `clip` says which source.
- A frame that no hook selects is never converted or copied. A selected frame
  is handed over by reference count, not by copying pixels.
- With a source decoded in ranges on several GPUs, frames reach hooks from
  several threads at once and out of order.

Pick the interval from what you need to know:

| Need | Sampling |
|------|----------|
| What a video is about (tags, an index, a thumbnail choice) | Every 1–5 s, with `max_frames` as a cost cap. |
| Whether something ever appears (moderation, a gate) | As dense as the shortest appearance you must not miss: an object on screen for half a second needs `every_seconds(0.25)` or denser. A sampled detector never sees what's between samples. |
| Tracking, counting, or per-frame analytics | `all()`, on a GPU (see [Performance](#8-performance)). |

## 4. Getting pixels into a model

Decoders produce planar YUV (4:2:0, 4:2:2 or 4:4:4, 8 to 12 bits), NV12 or
NV21, and image jobs RGBA. `rivet::hooks::frame` turns any of them into what
models take:

| Helper | Returns | Notes |
|--------|---------|-------|
| `planar_f32_letterboxed(frame, w, h, fill)` | `(Vec<f32>, Letterbox)`: the `[1, 3, h, w]` NCHW tensor in `0.0..=1.0`, RGB order, the picture fitted inside `w × h` with its aspect ratio kept and the rest filled with `fill` | Most detection models (YOLO among them) take this. Fastest path: it reads only the source pixels the output needs, straight from the decoder's planes, and writes the tensor directly. |
| `rgb8_letterboxed(frame, w, h, fill)` | `(Vec<u8>, Letterbox)`: interleaved 8-bit RGB | The same picture, for a model that wants bytes or NHWC. |
| `rgb8_resized(frame, w, h)` | `Vec<u8>`: interleaved 8-bit RGB, stretched to exactly `w × h` | For models trained on stretched pictures (many classifiers). |
| `rgb8_to_planar_f32(rgb, w, h)` | `Vec<f32>`: planar, `0.0..=1.0` | Interleaved RGB to NCHW. |
| `rgb8(frame)` / `luma8(frame)` | Full-resolution 8-bit RGB / luma | For your own resizing, or for models on luma. Cost grows with the frame's size. |
| `encode(frame, FrameFormat::Ppm \| Pgm \| Raw)` | An image file's bytes | To hand a picture to a tool or service that reads files. |

The scaling helpers are bilinear, in one pass: a 4K frame costs about the
same as a 1080p one to prepare (about 2–3 ms for a 640×640 tensor). YUV is
converted with the frame's own matrix, limited range.

### Normalisation and layout

The tensor helpers give RGB in `0..=1`. Models differ:

```rust
// ImageNet-style mean / std, per channel, on the planar tensor.
let (mut t, _) = frame::planar_f32_letterboxed(&f.frame, 224, 224, [0, 0, 0])?;
let n = 224 * 224;
for (c, (mean, std)) in [(0.485, 0.229), (0.456, 0.224), (0.406, 0.225)].into_iter().enumerate() {
    for v in &mut t[c * n..(c + 1) * n] {
        *v = (*v - mean) / std;
    }
}

// BGR order (OpenCV-trained models): swap the R and B planes.
let (r, rest) = t.split_at_mut(n);
r.swap_with_slice(&mut rest[n..2 * n]);

// NHWC, 0..=255 bytes (some TFLite-style models): use the interleaved form.
let (nhwc, _) = frame::rgb8_letterboxed(&f.frame, 320, 320, [0, 0, 0])?;
```

For `f16` inputs, convert each value with `half::f16::from_f32`.

### Coordinates

A letterbox scales and pads the picture, so a box the model finds is in the
model's input coordinates. `Letterbox` maps it back onto the frame, clamped
to it:

```rust
let (x, y, w, h) = letterbox.box_to_source(bx, by, bw, bh);   // a box
let (sx, sy) = letterbox.to_source(px, py);                   // a point
```

For `rgb8_resized`, scale by `frame.width / w` and `frame.height / h` instead.
Decoded frames are already upright (the container's rotation is applied), so
coordinates are in the picture as viewers see it, in the source's pixels.

## 5. Running the model

### Choosing a runtime

rivet works with any runtime that can be called from Rust:

| Runtime | Devices | Notes |
|---------|---------|-------|
| [ONNX Runtime](https://onnxruntime.ai) through [`ort`](https://crates.io/crates/ort) | CPU; NVIDIA (CUDA, TensorRT); any DirectX 12 GPU (DirectML); Intel GPU, NPU and CPU (OpenVINO); Apple (CoreML) | The broadest model coverage: most frameworks export ONNX. What [`examples/yolo`](../examples/yolo) uses. |
| [candle](https://github.com/huggingface/candle) | CPU, CUDA, Metal | Pure Rust. Already in this workspace for `denoise=dpir`. |
| [tract](https://github.com/sonos/tract) | CPU | Pure Rust, no native libraries; slower. |
| A model server (Triton, TorchServe, a cloud API) | Wherever it runs | Call it from a background hook, so network latency never stalls decoding. |

Keep the runtime **out of rivet's own dependency graph**. `examples/yolo` is
a crate of its own (`rivet-yolo-example`), which depends on rivet; rivet
doesn't depend on it. Your integration can do the same.

**ONNX Runtime and linking.** This workspace links the MSVC C runtime
statically (`+crt-static`, in `.cargo/config.toml`), and ONNX Runtime's
prebuilt static library needs the dynamic one, so it can't be linked in. The
example uses `ort`'s `load-dynamic` feature instead: the shared library is
loaded at start-up from a path you give (`ort::init_from`). That also means
the binary has no build-time dependency on a particular ONNX Runtime build:
the same binary runs the CPU, CUDA, DirectML or OpenVINO build of ONNX
Runtime, whichever library it's pointed at.

### Loading, warm-up and sessions

- **Load at start-up and fail early.** Read the model's input shape, input
  type (`f32` or `f16`) and output layout when loading, and refuse a model that
  doesn't fit then, rather than on a job's first frame.
- **Warm up.** A GPU's first run is slow: CUDA creates its context and cuDNN
  searches for the fastest algorithm for each layer, which took about 280 ms
  on an RTX 3090. Run the model once on a blank input when loading so that
  happens before the first job.
- **Sessions and threads.** Many runtimes' run call needs exclusive access
  (`ort`'s `Session::run` takes `&mut self`). One session behind a `Mutex`
  serialises pictures. Since frames from one decode thread arrive one at a
  time, that's usually enough. When frames arrive concurrently (ranged
  multi-GPU decode, several jobs on one server), keep a small pool and hand
  each picture to the first free session. `examples/yolo`'s `Worker` does this.
- **CUDA graphs** record a model once and replay it with one launch, which
  helps small models whose many small kernels cost as much to launch as to
  run: 5.0 ms to 3.9 ms of inference for YOLOv8n. They need fixed shapes, and
  a graph must be captured and replayed on the **same thread**. Captured on one
  thread and replayed from rivet's background worker, a graph crashed the job
  5 runs in 6, most likely because the provider captured it again mid-job while
  the decoder and encoder were using the GPU. Give each graph session a thread
  of its own and send it the pictures (see `Worker::Own` in
  [`hook.rs`](../examples/yolo/src/hook.rs)).

### Sharing the GPU with the transcode

Inference and rivet's hardware decode and encode (NVDEC/NVENC, AMF, QSV) can
share a card. Video decode and encode run mostly on the card's dedicated video
engines, not the compute units a model runs on, so the two contend mainly for
memory bandwidth and VRAM. Pictures
reach a hook in CPU memory, so each one is copied up to the GPU, and its
output copied back: about 5 MB up and 3 MB down for a 640×640 YOLO model.
To keep them on different cards, give the runtime a device index
(`cuda:1`) and rivet a decode and encode plan that avoids that card
(`--decode gpu:N`, `--encode`; see [cli.md](cli.md)).

## 6. Answering: annotations and verdicts

Each call returns a `HookOutcome`:

```rust
HookOutcome::proceed()                         // carry on
    .annotate("label", "cat")                  // record key = any JSON value
    .annotate("score", 0.93)
HookOutcome::reject("nudity at 12.0s")         // stop the job
HookOutcome::proceed().annotate(..).rejecting("reason")   // record, and stop
```

**Annotations** are what your hook found, recorded per picture. Keep them
compact and machine-readable: numbers rounded to a sensible precision,
labels as strings, boxes as `[x, y, w, h]` arrays in the source's pixels.
Record timings (`inference_ms`) too: they cost nothing and answer most
performance questions later. A long run with `FrameSampling::all()` makes one
record per frame, so a 10-minute 30 fps video is 18,000 records; keep each
small, or aggregate in the hook (`Mutex<State>`) and record a summary from a
`CompletedHook`.

**Verdicts** turn a model into a gate (content moderation, quality control):

- The first rejection wins. Once a hook rejects, every decode thread stops at
  its next check, and the job returns an error that `rejection_of` finds. On
  the HTTP API the job ends `rejected`, and a `?sync=true` request gets `422`.
- Reject on a threshold you chose deliberately, and say what was found and
  where in the reason (`"`person` detected at 3.00s (frame 90) (score 0.91)"`).
- Register a gate **fail closed** (`HookPolicy::default().fail_closed()`): a
  model that errors then rejects the job instead of letting it through
  unchecked. A model whose results only inform can stay fail open (the
  default): its errors are recorded, and the job carries on.

**Blocking or background:**

| | Blocking (default) | Background (`HookPolicy::background()`) |
|---|---|---|
| Runs on | the decode thread that reached the frame | the session's worker thread |
| Decoding while the model runs | waits | carries on, up to a queue of 64 events |
| A rejection | stops the job at once | stops it at its next point, at the latest before it returns |
| The job returns | after the last call | after the worker drains, so the report is complete either way |

Background suits models slower than decoding whose verdict doesn't need to
stop the job at that exact frame. The queue is bounded, so a model that can't
keep up slows the job down rather than buffering frames in memory.

**Never panic in a hook.** A hook's `Err` is handled by its policy, but a
panic isn't: rivet's release profile is built with `panic = "abort"`, so a
panicking hook ends the process. Return an error instead.

## 7. Getting results out

**After the job.** On success, the report is `JobOutput::hooks` (or
`ImageJobOutput::hooks`). To have it whatever happens, rejection included,
start the session yourself and keep a clone:

```rust
let session = hooks.session("upload-1234", JobKind::Transcode);
let spec = rivet::OutputSpec::single_file(rungs).with_hooks(session.clone());
let result = rivet::run_job_blocking_owned(input, &spec, None, sink);

let report = session.report();
for record in report.by_hook("classify") {
    if let Subject::Frame { seconds, .. } = record.subject {
        let label = record.annotation("label");
        // index, store, forward ...
    }
}
let json = report.to_json();     // the whole report, as the API serves it
```

`HookReport` also has `by_kind`, `annotations(key)` (every record carrying
that key), `errors()` and `rejection`.

**During the job.** To send results somewhere as they're found (a queue, a
database, a search index), do it inside the hook, or hand them to a consumer
thread of your own and register the hook in the background so a slow consumer
never stalls decoding ([cookbook recipe 8](hooks-cookbook.md#8-forward-to-your-own-system-without-blocking)).
`ctx.job_id` is the key to correlate on: the HTTP API's job id, or the id you
gave `Hooks::session`.

**On the HTTP API.** Pass your hooks to `rivet::server::serve_with_hooks`. A
model that should run only when asked is registered `optional()`, and a
request opts in by name:

```rust
let hooks = Hooks::new()
    .decoded_frames_with("classify", std::sync::Arc::clone(&model), HookPolicy::background().optional())
    .stills_with("classify-stills", model, HookPolicy::default().optional());
rivet::server::serve_with_hooks("0.0.0.0:8080".parse()?, hooks).await?;
```

```sh
curl -s -X POST --data-binary @upload.mp4 "localhost:8080/v1/transcode?hooks=classify"
curl -s localhost:8080/v1/jobs/$JOB | jq '[.hooks.records[] | select(.hook == "classify") | {t: .subject.seconds, label: .annotations.label}]'
```

The job status carries the report while the job runs and after it ends.
`GET /v1/hooks` lists what's configured. Requests choose only among
configured hooks: naming an unknown one is a `400`, and nothing in a request
can define a hook. See [api.md](api.md).

## 8. Performance

Measured with YOLOv8n (640×640) on every frame of a 10-second 1080p30 clip,
transcoding to H.264 at the same time. Full details are in
[hooks-yolo.md](hooks-yolo.md#on-a-gpu).

| Where | Prepare | Inference | Per picture | Job (transcode alone) |
|-------|---------|-----------|-------------|-----------------------|
| RTX 3090 host, CPU (Ryzen 9 9950X) | 3.0 ms | 13.5 ms | 17.8 ms | 8.0 s (4.75 s) |
| RTX 3090, CUDA | 2.5–2.8 ms | 4.8–5.5 ms | 8.5–9.5 ms | 5.4–6.0 s |
| RTX 3090, CUDA graph | 2.7 ms | 3.9–4.0 ms | 7.7–7.9 ms | 5.3–5.4 s |
| RTX 3090, DirectML | 2.4 ms | 3.6 ms | 7.1–7.2 ms | 4.9–5.0 s |
| Arc A380 host, CPU (Ryzen 5 5600X), ONNX Runtime | 4.5 ms | 31.4 ms | 37.6 ms | 14.9 s (3.8 s) |
| Arc A380 host, CPU, OpenVINO | 6.5 ms | 24.2 ms | 32.8 ms | 14.9 s |

What makes the difference, roughly in order:

1. **How many frames.** One frame a second is 30 times less work than every
   frame. Sample unless you need every frame.
2. **Where the model runs.** On a GPU, detecting on every 1080p frame added
   about a tenth to the job. On a six-core CPU, it tripled it.
3. **Blocking or background.** In the background, decoding carries on while
   the model runs.
4. **Warm-up and CUDA graphs**, for GPU runtimes (above).
5. **Preparation** costs 2–3 ms whatever the source's size, using
   `planar_f32_letterboxed`. Avoid `rgb8` on large frames: converting a 4K frame
   at full size costs far more than the model needs.

Every hook record carries `elapsed_ms` (the whole call). Record your own
stage timings as annotations (`examples/yolo` records `prepare_ms`,
`inference_ms` and `decode_ms`) to see where time goes.

## 9. Deployment

**Shipping the runtime.** With `ort`'s `load-dynamic`, ship the ONNX Runtime
library (and, for a GPU build, its provider libraries) beside your binary, or
point at it with a flag or `ORT_DYLIB_PATH`. Where to get each build:

| Build | Get it from |
|-------|-------------|
| CPU | [ONNX Runtime releases](https://github.com/microsoft/onnxruntime/releases): `onnxruntime-<os>-x64-<version>` |
| CUDA | The same page: `-gpu_cuda12-` / `-gpu_cuda13-` builds. Also needs the matching CUDA runtime and cuDNN 9 on the library path. |
| DirectML (Windows) | NuGet `Microsoft.ML.OnnxRuntime.DirectML`, plus `DirectML.dll` from `Microsoft.AI.DirectML` beside it |
| OpenVINO | Windows: NuGet `Intel.ML.OnnxRuntime.OpenVino`. Linux: the libraries inside the `onnxruntime-openvino` wheel. Both bundle OpenVINO itself. |

On Windows, pass a full path: a bare `onnxruntime.dll` not beside the program
may load the older copy Windows keeps in System32.

**Containers and Kubernetes.**

- NVIDIA: the NVIDIA Container Toolkit mounts the driver's libraries. CUDA
  and cuDNN user-space libraries must still be in the image.
- Intel GPUs: OpenVINO's GPU plugin needs Intel's compute runtime
  (`intel-opencl-icd`), which QSV doesn't, so an image that encodes with QSV
  may lack it. The compute runtime finds the card through
  `/dev/dri/by-path/pci-*-render`, while VA-API opens `/dev/dri/renderD*`
  directly. A device plugin can give the container's user `renderD128` and
  leave `by-path/` owned by `root` and a host group, and then QSV works while
  OpenVINO sees no GPU. Run `clinfo -l` in the container to check.
- Intel Arc (discrete) cards also need Resizable BAR. Without it, the compute
  runtime prints `WARNING: Small BAR detected` and doesn't expose the card.
  `rivet devices` reports it on its `PCI BAR` line (Linux), and says whether
  the card or the platform is what's in the way:

  ```text
  PCI BAR    : small (256 MiB of 8192 MiB VRAM): the card can resize it to 8192 MiB, but the platform hasn't; enable Above 4G Decoding and Resizable BAR in the firmware
  ```

**Models and licences.** rivet ships no model. Check the licence of the model
you deploy, and of its weights: many popular detectors (Ultralytics' YOLO
among them) are AGPL-3.0 or commercially licensed.

## 10. Testing

- **Test what doesn't need the model on its own.** Pre- and post-processing
  (output decoding, suppression, thresholds) are plain functions. Test them
  with small, hand-written tensors. `examples/yolo`'s `yolo.rs` does that for
  every output layout.
- **Test the hook without running a job.** A session's `emit_*` methods are
  the calls the engine makes. Drive them with a synthetic frame
  ([cookbook recipe 15](hooks-cookbook.md#15-unit-test-a-hook-without-running-a-job)):

  ```rust
  let hooks = Hooks::new().decoded_frames("classify", model).session("test", JobKind::Transcode);
  hooks.emit_decoded_frame(0, 0, 30.0, &frame)?;
  assert_eq!(hooks.report().annotations("label").next().unwrap().1, "cat");
  ```

  A test that needs a model file and a runtime library is better behind an
  environment variable than in the default test run.
- **Look at what the model saw.** Draw the results onto the frames and look
  at them (`examples/yolo` has `--draw DIR`). Most "the model finds nothing"
  problems are a wrong colour order, normalisation, letterbox fill or class
  list, and they're obvious in a picture.
- **Check one known image end to end** against the model's reference output
  (for YOLO, Ultralytics' `bus.jpg`: one bus, four people).

## 11. Checklist

- [ ] The model loads once, at start-up; a model that doesn't fit is refused then.
- [ ] The hook point matches the question: decoded frames for the source,
      encoder frames for what's published or for HDR sources and SDR models,
      stills for images.
- [ ] Sampling is as sparse as the question allows, with `max_frames` as a cap.
- [ ] Input matches training: size, letterbox or stretch, colour order,
      normalisation, layout, precision.
- [ ] Boxes are mapped back through the `Letterbox`.
- [ ] Shared state is behind a `Mutex` or atomics; the hook never panics.
- [ ] A gate is fail closed, with a deliberate threshold and a specific reason.
- [ ] A slow model or consumer runs in the background.
- [ ] On a GPU: warm-up at load; CUDA graphs on their own thread; on Intel,
      `rivet devices` shows a full PCI BAR and `clinfo -l` lists the card.
- [ ] Records are small; timings are recorded.
