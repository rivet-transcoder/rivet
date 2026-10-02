# YOLO object detection with hooks

This guide shows how to run a YOLO detector on the pictures a job decodes,
using the [hook engine](hooks.md). The detector is a hook. It sees sampled
video frames (and every still in an image job), finds objects in them, and
records each box in the job's hook report. It can also reject the job when it
finds a class you've ruled out.

The code is the [`examples/yolo`](../examples/yolo) crate, a small program you
can run as is or copy from:

| File | What it holds |
|------|---------------|
| [`src/yolo.rs`](../examples/yolo/src/yolo.rs) | Reading YOLO's output: the three tensor layouts, non-maximum suppression, class names. Plain Rust, independent of whatever runs the model, with unit tests. |
| [`src/hook.rs`](../examples/yolo/src/hook.rs) | `YoloHook`: loads an ONNX model with ONNX Runtime and implements `DecodedFrameHook` and `StillHook`. |
| [`src/main.rs`](../examples/yolo/src/main.rs) | The `yolo` command: registers the hook, runs a transcode or an image job, prints what was found. |
| [`src/draw.rs`](../examples/yolo/src/draw.rs) | `--draw`: writes each picture with its boxes as a PNG. |

It's a crate of its own, not one of `crates/rivet/examples`, so that ONNX
Runtime never becomes a dependency of rivet itself. rivet only provides the
hook points and the pixel helpers. The model and its runtime are yours to
choose.

## Quick start

You need three things: a YOLO model exported to ONNX, ONNX Runtime, and an
input.

**1. A model.** With [Ultralytics](https://docs.ultralytics.com/modes/export/):

```sh
pip install ultralytics
yolo export model=yolo11n.pt format=onnx          # writes yolo11n.onnx
```

Any detector export of YOLOv5, v7, v8, v9, v10, 11 or 26 works, and so does a
model you trained yourself. See [Models](#models).

**2. ONNX Runtime**, version 1.17 or later. Download the archive for your
platform from the [ONNX Runtime releases](https://github.com/microsoft/onnxruntime/releases)
(`onnxruntime-win-x64-*.zip`, `onnxruntime-linux-x64-*.tgz`,
`onnxruntime-osx-arm64-*.tgz`). Only the shared library in its `lib/` is
needed. The example loads it when it starts, so nothing about ONNX Runtime is
fixed at build time.

**3. Build and run:**

```sh
cargo build --release -p rivet-yolo-example --features rav1e-fallback,image-jobs

# A video: detect on the first frame of every second while it transcodes.
target/release/yolo yolo11n.onnx input.mp4 --ort path/to/onnxruntime.dll -o out.mp4

# A photo, with the boxes drawn so you can check them.
target/release/yolo yolo11n.onnx bus.jpg --ort path/to/onnxruntime.dll --draw boxes/
```

The hook only runs inside a job, so the video case needs an encoder:
`rav1e-fallback` is software AV1 and works anywhere. With a GPU, use its
feature instead (`nvidia`, `amd`, `qsv`). An NVIDIA card older than Ada has
no AV1 encoder, so add `--codec h264`. `image-jobs` adds still images. To save
passing `--ort` each time, set `ORT_DYLIB_PATH`, or put the library next to
the `yolo` binary.

What it prints for [`bus.jpg`](https://ultralytics.com/images/bus.jpg) with
`yolov8n.onnx` on a CPU:

```text
model: yolov8n.onnx (anchors input 640x640 F32, 80 classes) on Cpu, 1 session(s), warmed up in 17 ms
still 0 of source 0  1 bus, 4 person
1 pictures; bus 1, person 4
per picture: 20.91 ms (prepare 4.08, inference 15.29, decode 1.42)
```

And for a two-second video of the same scene:

```text
frame      0     0.00s  1 bus, 4 person
frame     25     1.00s  1 bus, 4 person
2 pictures; bus 2, person 8
per picture: 18.92 ms (prepare 2.10, inference 15.54, decode 1.11)
```

`prepare` is turning the frame into the model's input, `inference` the
model, and `decode` reading its output. [On a GPU](#on-a-gpu) the inference
drops to a few milliseconds.

### Options

| Option | Default | Meaning |
|--------|---------|---------|
| `-o, --output FILE` | | Write the transcoded video, or the image job's first output. |
| `--conf SCORE` | `0.25` | The lowest score a detection is kept at. |
| `--iou IOU` | `0.45` | Non-maximum suppression drops a box that overlaps a better box of the same class by more than this. |
| `--every SECONDS` | `1.0` | Detect on the first frame of each interval this long. `0` detects on every frame. |
| `--max-frames N` | | Stop detecting after N frames. The transcode itself carries on. |
| `--names FILE` | the model's own, else COCO | Class names, one per line, in class order. |
| `--layout LAYOUT` | from the output's shape | `anchors`, `anchors-transposed`, `anchors-objectness`, `end-to-end`. |
| `--device DEVICE` | `cpu` | `cuda[:N]`, `directml[:N]` or `openvino[:TARGET]`. See [On a GPU](#on-a-gpu) and [On Intel hardware](#on-intel-hardware-openvino). |
| `--openvino-cache DIR` | | Keep OpenVINO's compiled models here, so only the first load of a model compiles it. |
| `--cuda-graph` | | On CUDA, capture the model once as a CUDA graph and replay it for each picture. |
| `--sessions N` | `1` | How many pictures the model can take at once. See [Throughput](#throughput). |
| `--no-warm-up` | | Don't run the model once at load. The first picture then pays for the GPU's start-up. |
| `--ort PATH` | `ORT_DYLIB_PATH`, else next to the binary | The ONNX Runtime shared library. |
| `--refuse CLASSES` | | Reject the job when any of these classes (comma separated) is detected. |
| `--refuse-score SCORE` | `--conf` | The score a refused class has to reach to reject. |
| `--background` | | Run on the hook worker thread instead of the decode thread. |
| `--codec CODEC` | `av1` | `av1`, `h264` or `h265`, for the output video. |
| `--draw DIR` | | Write each picture the detector saw, with its boxes, as PNG. |
| `--report FILE` | | Write the job's whole hook report as JSON. |
| `--quiet` | | Print only the totals, not a line per picture. |

## Where it hooks in

The hook goes on the **decoded frames** for video and on the **stills** for
images. One value is registered at both points, so it's shared through an
`Arc`:

```rust
let yolo = Arc::new(YoloHook::load(&model, LoadOptions::default())?);
let hooks = Hooks::new()
    .decoded_frames("yolo", Arc::clone(&yolo))   // video jobs: sampled frames
    .stills("yolo-stills", yolo);                 // image jobs: every still
let spec = spec.with_hooks(hooks);
```

Decoded frames are the source's own pictures, turned upright, before
tonemapping, the spec's filters and any scaling. Boxes found there are in the
source's pixels whatever renditions the job makes, so one set of detections
describes every output. To detect on exactly what gets encoded (after crop,
pad or overlay filters), register it with `.encoder_frames` instead. It
implements both frame traits the same way.

Which frames it sees is the hook's `sampling()`. The example uses
`FrameSampling::every_seconds(--every)`, and `--max-frames` adds `max_frames`.
A frame no hook selects is never converted or copied, so detection costs only
the frames you ask for. A still hook sees every still.

## What it does with each picture

`YoloHook::detect` (in [`hook.rs`](../examples/yolo/src/hook.rs)) takes four
steps:

1. **Letterbox.** `rivet::hooks::frame::planar_f32_letterboxed(frame, 640, 640, [114, 114, 114])`
   makes the `[1, 3, H, W]` tensor in `0..=1` that YOLO takes, straight from
   whatever the decoder produced (YUV at any bit depth, NV12, RGBA). It fits
   the picture inside the model's input with the aspect kept, fills the rest
   with grey 114 like Ultralytics does, and returns the `Letterbox` that maps
   back. It reads only the source pixels the 640×640 output needs, so a 4K
   frame costs about what a 1080p one does: about 2–3 ms. The input size comes
   from the model. A dynamic size is taken as 640.
2. **Infer.** One run of the model, on a free session. `Session::run` takes
   `&mut self`, and frames can reach a hook from several decode threads at
   once (a source decoded in ranges on several GPUs), so each session sits
   behind a `Mutex` and a picture takes the first free one.
3. **Decode.** `yolo::decode` reads the output tensor in its
   [layout](#models), keeping boxes at or above `--conf`. Then `yolo::nms`
   keeps the best box of each overlapping group, class by class. End-to-end
   models have already done this step themselves.
4. **Map back.** `Letterbox::box_to_source` moves each box from the model's
   640×640 into the frame's pixels, and clamps it to the frame.

Then it returns a verdict with its annotations:

```rust
HookOutcome::proceed()
    .annotate("detections", detections)   // [{label, class, score, box: [x, y, w, h]}]
    .annotate("counts", counts)           // {"person": 4, "bus": 1}
    .annotate("prepare_ms", ...)          // step 1
    .annotate("inference_ms", ...)        // step 2
    .annotate("decode_ms", ...)           // steps 3 and 4
```

Each one becomes a record in the job's report. This is one from `--report`:

```json
{
  "hook": "yolo-stills", "kind": "still", "stage": "still",
  "subject": { "type": "still", "clip": 0, "index": 0, "seconds": 0.0 },
  "verdict": "continue", "reason": null, "error": null,
  "annotations": {
    "counts": { "bus": 1, "person": 4 },
    "detections": [
      { "label": "person", "class": 0, "score": 0.889, "box": [670.4, 380.1, 139.5, 499.5] },
      { "label": "person", "class": 0, "score": 0.882, "box": [221.7, 407.5, 122.1, 448.7] },
      { "label": "person", "class": 0, "score": 0.879, "box": [50.6, 397.3, 193.8, 508.2] },
      { "label": "bus",    "class": 5, "score": 0.842, "box": [34.9, 229.6, 762.5, 537.6] },
      { "label": "person", "class": 0, "score": 0.436, "box": [0.5, 549.7, 57.5, 318.7] }
    ],
    "prepare_ms": 4.08, "inference_ms": 15.29, "decode_ms": 1.42
  },
  "elapsed_ms": 20.9, "background": false
}
```

Boxes are `[left, top, width, height]` in the source picture's pixels. For a
video frame, the subject is `{ "type": "frame", "clip", "index", "seconds" }`,
which tells you when in the source each set of boxes was seen.

## Reading the detections

On success, the report is `JobOutput::hooks` (or `ImageJobOutput::hooks`). If
you want it whatever happens, rejection included, start the session yourself
and keep a clone, as `main.rs` does:

```rust
let session = hooks.session("upload-1234", JobKind::Transcode);
let spec = rivet::OutputSpec::single_file(rungs).with_hooks(session.clone());
let result = rivet::run_job_blocking_owned(input, &spec, None, sink);

for record in session.report().by_hook("yolo") {
    let Subject::Frame { seconds, .. } = record.subject else { continue };
    let counts = record.annotation("counts");      // Some({"person": 2, ...})
    let boxes = record.annotation("detections");
    // store, index, forward ...
}
```

To send detections somewhere as they're found rather than after the job, do it
inside the hook, or hand them to a consumer of your own and register the hook
in the background ([cookbook recipe 8](hooks-cookbook.md#8-forward-to-your-own-system-without-blocking)).
`ctx.job_id` is the key to correlate them by.

## Rejecting a job on what it shows

`--refuse person,knife` (in code, `yolo.refuse` and `yolo.refuse_score`)
makes a detection of a refused class reject the job. The outcome keeps its
annotations and gets a reason:

```rust
HookOutcome::proceed()
    .annotate("detections", ...)
    .rejecting(format!("`{label}` detected at {seconds:.2}s (frame {index}) (score {score:.2})"))
```

```text
$ yolo yolov8n.onnx bus.jpg --refuse bus --refuse-score 0.5
still 0 of source 0  1 bus, 4 person
rejected: `bus` detected in the image (score 0.84)
Error: job rejected by hook `yolo-stills`: `bus` detected in the image (score 0.84)
```

Once rejected, every decode thread stops at its next check, and the job
returns an error that `rejection_of` finds. The HTTP API ends the job
`rejected`, and a `?sync=true` request gets a `422`. The example registers the
hook **fail closed** when it's refusing anything. Fail open would let a
picture through whenever the detector errored, which defeats a gate.

A detector only ever samples the frames you let it see. A class that shows up
only between samples is never seen. If the gate matters, sample densely
(`--every 0.2`, or `FrameSampling::every_frames(n)`) and set the threshold
deliberately (`--refuse-score`).

## Blocking or background

| | Blocking (default) | `--background` (`HookPolicy::background()`) |
|---|---|---|
| Runs on | the decode thread that reached the frame | the session's worker thread |
| Decoding while it infers | waits | carries on, up to the worker's queue of 64 events |
| A rejection | stops the job at once | stops it at its next point, at the latest before it returns |
| The job returns | after the last detection | after the worker drains, so the report is complete either way |

Background mode is the right choice when detection is slower than decoding
and nothing needs to stop the job immediately. Because the queue is bounded,
a detector that can't keep up slows the job down instead of piling frames up
in memory.

## Models

`yolo::Layout::infer` works out the output layout from its shape and the
number of classes:

| Layout | Output shape | Each row | Models |
|--------|--------------|----------|--------|
| `anchors` | `[1, 4 + classes, anchors]` | `cx, cy, w, h`, a score per class | YOLOv8, YOLOv9, YOLO11 (Ultralytics' default ONNX export) |
| `anchors-transposed` | `[1, anchors, 4 + classes]` | the same, transposed | some third-party exports of the above |
| `anchors-objectness` | `[1, anchors, 5 + classes]` | `cx, cy, w, h`, objectness, a score per class | YOLOv5, YOLOv7 |
| `end-to-end` | `[1, detections, 6]` | `x1, y1, x2, y2, score, class`, already suppressed | YOLOv10, YOLO26, and Ultralytics exports with `nms=True` |

**Class names** come from `--names`, else from the export's own `names`
metadata (Ultralytics writes it), else the 80 COCO classes. A custom-trained
model exported by Ultralytics brings its names along. For any other export,
pass `--names` with one name per line. The class count is also what tells the
layouts apart, so a model with the wrong names may be reported as "can't tell
the layout". Pass `--names` or `--layout`.

**Input size**: export at the size you'll run (`imgsz=640` is the default).
Larger sizes find smaller objects at a cost that grows with the area.

**Export notes**: the default `format=onnx` export works as is.
`dynamic=True` works too; a dynamic side is taken as 640, though a CUDA graph
needs fixed shapes. `half=True` (fp16 input and output) works: the hook sees
the input's type and converts. Segmentation and pose models have a second
output this example ignores. They still detect, but the masks and keypoints
are left unread.

**Licences.** rivet ships no model. Ultralytics' YOLOv5, v8, 11 and 26 code
and weights are AGPL-3.0, or under an Ultralytics Enterprise licence. YOLOv7,
v9 and v10 come from their authors under their own terms. Check what applies
to the model you deploy. ONNX Runtime is MIT.

## On a GPU

ONNX Runtime picks the device through an execution provider. Build with the
provider's feature, and point `--ort` at an ONNX Runtime build that has it:

| `--device` | Feature | ONNX Runtime build | Also needs |
|------------|---------|--------------------|------------|
| `cuda[:N]` | `cuda` | `onnxruntime-win-x64-gpu_cuda13-*.zip` / `-gpu_cuda12-*` (Windows), `onnxruntime-linux-x64-gpu_cuda13-*.tgz` / `-gpu-*` (Linux), from the [releases](https://github.com/microsoft/onnxruntime/releases) | The CUDA runtime that build was made for (cuBLAS, cuFFT, cudart) and cuDNN 9, on the library path |
| `directml[:N]` | `directml` | `onnxruntime.dll` from the [`Microsoft.ML.OnnxRuntime.DirectML`](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.DirectML) NuGet package (`runtimes/win-x64/native/`) | `DirectML.dll` from [`Microsoft.AI.DirectML`](https://www.nuget.org/packages/Microsoft.AI.DirectML) (`bin/x64-win/`) beside it. Windows, any DirectX 12 GPU, no CUDA. |
| `openvino[:TARGET]` | `openvino` | See [On Intel hardware](#on-intel-hardware-openvino) | |

A `.nupkg` is a zip file. On Windows, CUDA looks like this:

```sh
cargo build --release -p rivet-yolo-example --features nvidia,cuda

# The CUDA toolkit's runtime DLLs, and cuDNN 9 (here from `pip install nvidia-cudnn-cu13`).
export PATH="$CUDA_PATH/bin/x64:$PYTHON/Lib/site-packages/nvidia/cudnn/bin:$PATH"
target/release/yolo yolo11n.onnx input.mp4 --codec h264 \
    --device cuda --cuda-graph --ort onnxruntime-win-x64-gpu_cuda13-1.30.0/lib/onnxruntime.dll
```

The provider is registered with `error_on_failure()`, so a provider that
can't start fails loudly instead of silently falling back to the CPU. The
decode and encode GPUs are rivet's business and inference is ONNX Runtime's.
They can be the same card or different ones (`cuda:1`). Pictures reach the
hook in CPU memory, so each one is copied up to the GPU and its output copied
back. For YOLO that's about 5 MB up and 3 MB down per picture.

**Warm-up.** A GPU's first run is slow: CUDA creates its context and cuDNN
searches for the fastest algorithm for each layer, which takes about 280 ms
here. (DirectML compiles when the session is created, so its warm-up is under
10 ms.) `YoloHook::load` runs each session once on a blank picture (twice for
a CUDA graph), so that happens before the job starts rather than on its first
frame. `--no-warm-up` skips it.

**CUDA graphs** (`--cuda-graph`, `LoadOptions::cuda_graph`). YOLOv8n is a few
hundred small kernels, and on a fast GPU launching them costs as much as
running them. With a CUDA graph, ONNX Runtime records the whole model once
and replays it with one launch per picture. For that, the input and output
have to stay at fixed places on the GPU, so the hook binds a device input and
output to each session (`IoBinding`) and copies each picture in and each
result out. This needs fixed shapes (not a `dynamic=True` export) and every
node on the GPU. A dynamic shape is refused at load, and a node on the CPU
fails the warm-up, which also runs at load.

**A graph stays on one thread.** Captured at load on the main thread and
replayed from the hooks' background worker, a CUDA graph crashed the job in 5
runs out of 6. Captured and replayed on the same thread, it never did. The
likely cause: ONNX Runtime's CUDA provider ties a captured graph to the thread
that ran it, so a run from another thread captures it again in the middle of
the job, while rivet's decoder and encoder are using the same GPU, and
capturing isn't safe alongside them. So each graph session lives on a thread
of its own. It's captured there during warm-up and only ever replayed there,
and a picture from any thread is handed over to it. With that, 10 runs out of
10 succeeded, and the hand-off made no measurable difference to inference
time.

Measured on an RTX 3090 with ONNX Runtime 1.30 (DirectML: 1.24), YOLOv8n at
640×640 on every frame of a 10-second 1080p30 clip (300 frames), transcoding
to H.264 on NVENC. Ranges are over two runs:

| | Prepare | Inference | Decode output | Per picture | Job |
|---|---|---|---|---|---|
| CPU | 3.0 ms | 13.5 ms | 1.2 ms | 17.8 ms | 8.0 s |
| CUDA | 2.5–2.8 ms | 4.8–5.5 ms | 1.1 ms | 8.5–9.5 ms | 5.4–6.0 s |
| CUDA, `--cuda-graph` | 2.7 ms | 3.9–4.0 ms | 1.0 ms | 7.7–7.9 ms | 5.3–5.4 s |
| DirectML | 2.4 ms | 3.6 ms | 1.0 ms | 7.1–7.2 ms | 4.9–5.0 s |

The transcode alone takes about 4.75 s, so on a GPU, detecting on every frame
adds roughly a tenth to the job. The 4K version of the clip prepares in the
same 3 ms. Job times vary by a few tenths of a second between identical runs,
so the GPU rows' differences there, and `--background` (4.9–5.6 s across the
three GPU modes), are within that noise. The per-picture times are steady.

An fp16 export was slower here (7.0 ms of inference on CUDA against 5.0 ms),
because YOLOv8n is too small for half precision to pay for its conversions.
Larger models and TensorRT are where fp16 helps.

## On Intel hardware (OpenVINO)

QSV is Intel's video engine: it decodes and encodes, and rivet uses it for
that (the `qsv` feature). It doesn't run neural networks. On Intel hardware,
the model runs through **OpenVINO**, ONNX Runtime's OpenVINO execution
provider, on the same GPU QSV uses (an Arc card or the integrated GPU), on an
NPU, or on the CPU:

```sh
cargo build --release -p rivet-yolo-example --features qsv,openvino

target/release/yolo yolo11n.onnx input.mp4 --codec h264 \
    --device openvino:GPU --openvino-cache ~/.cache/rivet-openvino --ort path/to/libonnxruntime.so.1.24.1
```

`TARGET` is OpenVINO's own device name: `GPU` (or `GPU.0`, `GPU.1` with
several), `NPU`, `CPU`, or `AUTO`, which lets OpenVINO pick and is the
default. Compiling a model for a GPU takes a while, and `--openvino-cache`
(`LoadOptions::openvino_cache`) keeps the result so that later loads don't
compile again.

**Getting an OpenVINO build of ONNX Runtime.** Microsoft's releases don't
include one. Intel publishes them, each with its own copy of OpenVINO's
runtime and its CPU, GPU and NPU plugins, so nothing else needs installing:

| Platform | Where | What to point `--ort` at |
|----------|-------|--------------------------|
| Windows | the [`Intel.ML.OnnxRuntime.OpenVino`](https://www.nuget.org/packages/Intel.ML.OnnxRuntime.OpenVino) NuGet package (a zip file) | `runtimes/win-x64/native/onnxruntime.dll`, with that folder on `PATH` |
| Linux | the [`onnxruntime-openvino`](https://pypi.org/project/onnxruntime-openvino/) wheel for `manylinux_2_28_x86_64` (also a zip file; any Python version's wheel will do, nothing Python is used) | `onnxruntime/capi/libonnxruntime.so.<version>`, with that folder on `LD_LIBRARY_PATH` |

**An Intel GPU on Linux** also needs:

- Intel's compute runtime, `intel-opencl-icd`, which OpenVINO's GPU plugin
  runs on. QSV doesn't need it (VA-API and oneVPL are separate), so a machine
  that already encodes with QSV may well not have it. `clinfo -l` should list
  the card.
- Read access to the card's `/dev/dri/by-path/pci-*-render` node. The compute
  runtime finds the GPU through it, while VA-API opens `/dev/dri/renderD*`
  directly. In a container, a device plugin can hand the user `renderD128`
  and leave `by-path/` owned by `root` and a host group. QSV then works and
  OpenVINO sees no GPU. Add the user to that group, or have the plugin set
  the ownership of both.
- For an Arc (discrete) card, Resizable BAR. With a small BAR, the compute
  runtime prints `WARNING: Small BAR detected for device ...` and doesn't
  expose the card, on the upstream `i915` driver at least. That's a BIOS
  setting, and with the card passed through to a VM, the hypervisor's too.

**Measured** on an Arc A380 host (Ryzen 5 5600X, Ubuntu 24.04), YOLOv8n on
every frame of the 1080p clip while QSV encoded H.264 on the A380 (ONNX
Runtime 1.24.1 with OpenVINO 2025.4):

| `--device` | Prepare | Inference | Per picture | Job (transcode alone: 3.8 s) |
|---|---|---|---|---|
| `cpu` (ONNX Runtime's own) | 4.5 ms | 31.4 ms | 37.6 ms | 14.9 s |
| `openvino:CPU` | 6.5 ms | 24.2 ms | 32.8 ms | 14.9 s |
| `openvino:GPU` (the A380) | not measured: this host gives the card a small BAR, so the compute runtime won't expose it (see above) | | | |

OpenVINO runs the model a quarter faster than ONNX Runtime's own CPU code on
the same CPU. The detections were identical. The job took the same time
because, at every frame, the six-core CPU is the limit for both.

On Windows, OpenVINO's GPU plugin also drives non-Intel GPUs through OpenCL:
`openvino:GPU` ran on an RTX 3090 there. CUDA or DirectML is the better
choice on such a card.

## Throughput

- **Sample, don't detect every frame**, unless you need to. At one frame a
  second, even the CPU keeps well up with a transcode. On every frame of 30
  fps video, use a GPU (see the table above).
- **Preparing a picture costs about the same whatever the source's size**,
  because `planar_f32_letterboxed` only reads the source pixels the model's
  input needs. Each record's `prepare_ms`, `inference_ms` and `decode_ms`
  show where the time goes. `elapsed_ms` is the whole call.
- **Blocking or background.** Blocking, the decode thread waits for each
  detection. In the background (`--background`), decoding carries on while
  the model runs, up to the worker's queue of 64 events.
- **Sessions.** One session serves one picture at a time. Frames from a single
  decode thread arrive one after another, so a second session doesn't help
  there (measured: 8.9 ms per picture with two against 8.4 ms with one).
  Use `--sessions` when a source is decoded in ranges on several GPUs and
  frames reach the hook concurrently.

## Other runtimes

Only the session handling in [`hook.rs`](../examples/yolo/src/hook.rs)
(`session`, `Bound`, `Worker`, `run`) knows about ONNX Runtime. Everything else is
runtime-independent: `planar_f32_letterboxed` (or `rgb8_letterboxed` for
interleaved RGB) to prepare the input, `yolo::decode` and `yolo::nms` to read the output, and
`Letterbox::box_to_source` to map boxes back. To use another runtime, such as
[tract](https://github.com/sonos/tract) (pure Rust, CPU),
[candle](https://github.com/huggingface/candle) (already in this workspace
for `denoise=dpir`), TensorRT, or a model server, replace those lines and
keep the rest. A remote inference service fits best as a background hook, so
network latency never stalls the decode.

## On the HTTP API

The same `Hooks` value serves the HTTP API. Register the detector as
optional, and requests can opt into it by name:

```rust
let hooks = Hooks::new()
    .decoded_frames_with("yolo", Arc::clone(&yolo), HookPolicy::background().optional())
    .stills_with("yolo-stills", yolo, HookPolicy::default().optional());
rivet::server::serve_with_hooks(addr, hooks).await?;
```

```sh
curl -s -X POST --data-binary @upload.mp4 "localhost:8080/v1/transcode?hooks=yolo"
curl -s localhost:8080/v1/jobs/$JOB | jq '[.hooks.records[] | select(.hook == "yolo") | {t: .subject.seconds, counts: .annotations.counts}]'
```

`GET /v1/hooks` lists it with its `describe()`, for example `YOLO detection
(anchors, 640x640 F32, 80 classes, 1 session(s))`, and its sampling.

## Testing

`cargo test -p rivet-yolo-example` covers decoding every layout, NMS and the
names parser without a model or ONNX Runtime. To test the hook itself inside
a session without running a job, drive `emit_decoded_frame` / `emit_still`
([cookbook recipe 15](hooks-cookbook.md#15-unit-test-a-hook-without-running-a-job)).
That needs a model file, so keep such a test behind an environment variable.

## Troubleshooting

| Symptom | Cause |
|---------|-------|
| `loading ONNX Runtime from onnxruntime.dll` ... `expected version >= '1.17.x'` | An older ONNX Runtime was found first. On Windows that's often the copy in System32. Pass `--ort` with the full path. |
| `failed to load from ...` | The path is wrong, or (for a GPU build) the CUDA, cuDNN or DirectML libraries it depends on aren't on the library path. |
| `--device cuda` fails while loading the model | `onnxruntime_providers_cuda.dll` (beside `onnxruntime.dll`) couldn't load its CUDA or cuDNN libraries. Put the CUDA runtime that ONNX Runtime build was made for (12 or 13) and cuDNN 9 on `PATH` / `LD_LIBRARY_PATH`. |
| `has a dynamic shape ...; a CUDA graph needs fixed shapes` | The model was exported with `dynamic=True`. Export it again without, or leave out `--cuda-graph`. |
| `--cuda-graph` fails while warming up | A node of the model runs on the CPU, which a CUDA graph can't capture. Leave out `--cuda-graph`. |
| `OpenVINO couldn't use `GPU`` with `[OpenVINO] Device GPU is not available` | OpenVINO sees no Intel GPU. Check `clinfo -l`. If it lists nothing, see the three requirements under [On Intel hardware](#on-intel-hardware-openvino): the compute runtime, `/dev/dri/by-path` access, and Resizable BAR for an Arc card. |
| `DirectML.dll` errors, or an old DirectML | Windows has its own `DirectML.dll` in System32. Put the one from `Microsoft.AI.DirectML` beside `onnxruntime.dll`. |
| `can't tell the layout of a [1, a, b] output` | The class names don't match the model. Pass `--names`, or `--layout`. |
| Nothing found where there should be something | Run with `--draw` and look at what the detector saw. Check `--conf`, and check that the model's names are the classes you expect. |
| Boxes in the wrong place | A model that expects a stretched input rather than a letterboxed one (rare for YOLO). Use `rgb8_resized` and scale the boxes by the two ratios. |
| The video job fails before any frame | No encoder. Build with `rav1e-fallback`, or a GPU feature, or use `--codec h264` with `h26x-fallback`. |
