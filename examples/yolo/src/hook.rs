//! A YOLO detector as a rivet hook: a decoded-frame hook for video jobs and a
//! still hook for image jobs, one value registered at both.
//!
//! Each picture it's handed is letterboxed to the model's input size
//! (`rivet::hooks::frame::rgb8_letterboxed`), run through ONNX Runtime, and
//! decoded ([`crate::yolo`]); the boxes are mapped back onto the frame and
//! recorded in the job's hook report. Optionally, a detection of a class it is
//! told to refuse rejects the job.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, mpsc};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use ort::session::Session;
use half::f16;
use ort::session::IoBinding;
use ort::value::{DynTensor, DynTensorValueType, Tensor, TensorElementType, TensorValueTypeMarker};
use rivet::codec::frame::VideoFrame;
use rivet::hooks::frame::{Letterbox, planar_f32_letterboxed};
use rivet::hooks::{DecodedFrameHook, FrameEvent, FrameSampling, HookContext, HookOutcome, StillEvent, StillHook};
use serde_json::{Value, json};

use crate::yolo::{self, Detection, Layout};

/// Where inference runs.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum Device {
    #[default]
    Cpu,
    /// NVIDIA, through ONNX Runtime's CUDA execution provider (the `cuda`
    /// feature, an ONNX Runtime GPU build, and the CUDA and cuDNN 9 libraries
    /// that build was made for on the library path).
    Cuda(i32),
    /// Any DirectX 12 GPU on Windows (the `directml` feature).
    DirectMl(i32),
    /// Intel hardware through OpenVINO (the `openvino` feature, and an ONNX
    /// Runtime build with OpenVINO): the device in OpenVINO's own words, `GPU`
    /// (an Arc card or the integrated GPU, which QSV also runs on), `GPU.1`,
    /// `NPU`, `CPU`, or `AUTO` to let OpenVINO pick.
    OpenVino(String),
}

/// Loads ONNX Runtime from `path`, else `ORT_DYLIB_PATH`, else the platform's
/// library name next to the running program. Call once, before
/// [`YoloHook::load`]. An explicit path matters on Windows, where a bare
/// `onnxruntime.dll` not beside the program finds the copy Windows ships in
/// System32, which may be older than this needs.
pub fn load_runtime(path: Option<&Path>) -> Result<()> {
    let default = if cfg!(windows) {
        "onnxruntime.dll"
    } else if cfg!(target_os = "macos") {
        "libonnxruntime.dylib"
    } else {
        "libonnxruntime.so"
    };
    let path = match path {
        Some(p) => p.to_path_buf(),
        None => std::env::var_os("ORT_DYLIB_PATH").filter(|p| !p.is_empty()).map_or_else(|| PathBuf::from(default), PathBuf::from),
    };
    ort::init_from(&path)
        .with_context(|| format!("loading ONNX Runtime from {} (see --ort)", path.display()))?
        .with_name("rivet-yolo")
        .commit();
    Ok(())
}

/// How [`YoloHook::load`] sets the detector up.
#[derive(Debug, Clone)]
pub struct LoadOptions {
    pub device: Device,
    /// Sessions to keep: how many pictures can be in the model at once. Frames
    /// reach the hook from several decode threads when a source is decoded in
    /// ranges on several GPUs; one session makes them take turns.
    pub sessions: usize,
    /// Class names; `None` reads the export's own, else COCO.
    pub names: Option<Vec<String>>,
    /// The output layout; `None` works it out from the output's shape.
    pub layout: Option<Layout>,
    /// Run each session once on a blank picture before the job starts, so the
    /// first frame doesn't pay for the GPU's start-up (CUDA's context, cuDNN's
    /// algorithm search): hundreds of milliseconds.
    pub warm_up: bool,
    /// On CUDA, capture the model as a CUDA graph and replay it for each
    /// picture: one launch instead of one per layer. The input and output then
    /// live at fixed places on the GPU, copied to and from for each picture.
    /// Needs a model whose every node runs on the GPU and whose shapes are
    /// fixed.
    pub cuda_graph: bool,
    /// On OpenVINO, where to keep compiled models. Compiling for a GPU takes
    /// seconds; with a cache, only the first load of a model pays it.
    pub openvino_cache: Option<PathBuf>,
}

impl Default for LoadOptions {
    fn default() -> Self {
        LoadOptions {
            device: Device::Cpu,
            sessions: 1,
            names: None,
            layout: None,
            warm_up: true,
            cuda_graph: false,
            openvino_cache: None,
        }
    }
}

/// What the model takes its pixels as. A `half=True` export takes `f16`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    F32,
    F16,
}

/// Where one picture's time went.
#[derive(Debug, Clone, Copy, Default)]
pub struct Timing {
    /// Letterboxing the frame into the model's input tensor.
    pub prepare_ms: f64,
    /// The model itself, input upload and output download included.
    pub inference_ms: f64,
    /// Reading the output and suppressing overlaps.
    pub decode_ms: f64,
}

/// A YOLO detector, ready to register as a hook.
pub struct YoloHook {
    /// `Session::run` takes `&mut self`, so each session serves one picture at
    /// a time; a picture takes the first free one.
    sessions: Vec<Worker>,
    next: AtomicUsize,
    input: Input,
    layout: Layout,
    names: Vec<String>,
    warm_up: Option<Duration>,
    pub min_score: f32,
    pub iou: f32,
    pub max_detections: usize,
    pub sampling: FrameSampling,
    /// Classes whose detection rejects the job, by name.
    pub refuse: BTreeSet<String>,
    /// The score a refused class must reach to reject (defaults to `min_score`).
    pub refuse_score: Option<f32>,
    /// Write each picture with its boxes drawn here, as PNG.
    pub draw_to: Option<PathBuf>,
}

/// What a run of the model needs to know about its input.
#[derive(Debug, Clone)]
struct Input {
    name: String,
    /// Width, height.
    size: (u32, u32),
    precision: Precision,
}

type Output = Result<(Vec<i64>, Vec<f32>)>;
type Request = (Vec<f32>, mpsc::SyncSender<Output>);

/// A session, and where it runs.
enum Worker {
    /// On whichever thread hands it a picture.
    Here(Box<Mutex<Slot>>),
    /// On a thread of its own, which a CUDA graph needs. Captured on one
    /// thread and replayed from another (the hooks' background worker, say),
    /// a graph crashed the job in 5 runs of 6: most likely ONNX Runtime's CUDA
    /// provider captures it again for the new thread, in the middle of the
    /// job, while the decoder and encoder are using the GPU. Owned by one
    /// thread, the graph is captured once, at load, and only ever replayed.
    /// The lock marks it busy.
    Own(Mutex<mpsc::Sender<Request>>),
}

/// A worker claimed for one picture.
enum Claimed<'a> {
    Here(MutexGuard<'a, Slot>),
    Own(MutexGuard<'a, mpsc::Sender<Request>>),
}

impl Worker {
    fn new(slot: Slot, input: &Input) -> Result<Worker> {
        if slot.bound.is_none() {
            return Ok(Worker::Here(Box::new(Mutex::new(slot))));
        }
        let (tx, rx) = mpsc::channel::<Request>();
        let input = input.clone();
        std::thread::Builder::new().name("yolo-cuda-graph".into()).spawn(move || {
            let mut slot = slot;
            for (planar, reply) in rx {
                let _ = reply.send(run(&input, &mut slot, planar));
            }
            // The session, and its graph, go with the thread that made them.
        })?;
        Ok(Worker::Own(Mutex::new(tx)))
    }

    /// This worker for one picture: if it's free, or with `wait`, once it is.
    fn claim(&self, wait: bool) -> Option<Claimed<'_>> {
        fn take<T>(m: &Mutex<T>, wait: bool) -> Option<MutexGuard<'_, T>> {
            if wait { Some(m.lock().unwrap_or_else(|e| e.into_inner())) } else { m.try_lock().ok() }
        }
        match self {
            Worker::Here(m) => take(m, wait).map(Claimed::Here),
            Worker::Own(m) => take(m, wait).map(Claimed::Own),
        }
    }

    /// Runs it captures a CUDA graph in, at load.
    fn warm_up_runs(&self) -> usize {
        match self {
            Worker::Here(_) => 1,
            // The graph is captured on a run after the first.
            Worker::Own(_) => 2,
        }
    }
}

impl Claimed<'_> {
    fn run(self, input: &Input, planar: Vec<f32>) -> Output {
        match self {
            Claimed::Here(mut slot) => run(input, &mut slot, planar),
            Claimed::Own(tx) => {
                let (reply, answer) = mpsc::sync_channel(1);
                let gone = || anyhow::anyhow!("the inference thread is gone");
                tx.send((planar, reply)).map_err(|_| gone())?;
                answer.recv().map_err(|_| gone())?
            }
        }
    }
}

/// One session, and when it replays a CUDA graph, the input and output that
/// stay put on the GPU for it.
struct Slot {
    session: Session,
    bound: Option<Bound>,
}

/// A session's input and output on the device, bound once.
struct Bound {
    binding: IoBinding,
    input: DynTensor,
    /// Where each picture is written before it's copied up, and where its
    /// output is copied back to: allocated once, reused for every picture.
    host_input: DynTensor,
    host_output: DynTensor,
    /// What the device tensors were allocated from, which must outlive them
    /// (fields drop in order, so this goes last).
    #[cfg(feature = "cuda")]
    _gpu: ort::memory::Allocator,
}

impl Bound {
    /// Device tensors for `session`'s one input and one output on CUDA device
    /// `id`, bound to it.
    #[cfg(feature = "cuda")]
    fn new(session: &Session, id: i32) -> Result<Bound> {
        use ort::memory::{AllocationDevice, Allocator, AllocatorType, MemoryInfo, MemoryType};
        let fixed = |outlet: &ort::value::Outlet| -> Result<(TensorElementType, Vec<i64>)> {
            let ty = outlet.dtype().tensor_type().context("not a tensor")?;
            let shape = outlet.dtype().tensor_shape().context("not a tensor")?.to_vec();
            if shape.iter().any(|&d| d < 0) {
                bail!("`{}` has a dynamic shape {shape:?}; a CUDA graph needs fixed shapes (export without `dynamic=True`)", outlet.name());
            }
            Ok((ty, shape))
        };
        let (input, output) = (&session.inputs()[0], &session.outputs()[0]);
        let ((in_ty, in_shape), (out_ty, out_shape)) = (fixed(input)?, fixed(output)?);
        let gpu = Allocator::new(session, MemoryInfo::new(AllocationDevice::CUDA, id, AllocatorType::Device, MemoryType::Default)?)?;
        let in_shape_host = in_shape.clone();
        let device_input = DynTensor::new(&gpu, in_ty, in_shape)?;
        let device_output = DynTensor::new(&gpu, out_ty, out_shape.clone())?;
        let mut binding = session.create_binding()?;
        binding.bind_input(input.name(), &device_input)?;
        binding.bind_output(output.name(), device_output)?;
        Ok(Bound {
            binding,
            input: device_input,
            host_input: DynTensor::new(&Allocator::default(), in_ty, in_shape_host)?,
            host_output: DynTensor::new(&Allocator::default(), out_ty, out_shape)?,
            _gpu: gpu,
        })
    }

    #[cfg(not(feature = "cuda"))]
    fn new(_session: &Session, _id: i32) -> Result<Bound> {
        bail!("a CUDA graph needs this example built with the `cuda` feature")
    }
}

/// One session of `model` on the device `options` names.
fn session(model: &Path, options: &LoadOptions) -> Result<Slot> {
    let (device, cuda_graph) = (&options.device, options.cuda_graph);
    if cuda_graph && !matches!(device, Device::Cuda(_)) {
        bail!("a CUDA graph needs `--device cuda`");
    }
    if options.openvino_cache.is_some() && !matches!(device, Device::OpenVino(_)) {
        bail!("an OpenVINO cache needs `--device openvino`");
    }
    let builder = Session::builder()?;
    let mut builder = match device {
        Device::Cpu => builder,
        #[cfg(feature = "cuda")]
        Device::Cuda(id) => builder
            .with_execution_providers([ort::ep::CUDA::default()
                .with_device_id(*id)
                .with_cuda_graph(cuda_graph)
                .build()
                .error_on_failure()])
            .map_err(ort::Error::<()>::from)?,
        // DirectML runs one graph at a time and plans its own memory.
        #[cfg(feature = "directml")]
        Device::DirectMl(id) => builder
            .with_memory_pattern(false)
            .map_err(ort::Error::<()>::from)?
            .with_parallel_execution(false)
            .map_err(ort::Error::<()>::from)?
            .with_execution_providers([ort::ep::DirectML::default().with_device_id(*id).build().error_on_failure()])
            .map_err(ort::Error::<()>::from)?,
        #[cfg(feature = "openvino")]
        Device::OpenVino(target) => {
            let mut ep = ort::ep::OpenVINO::default().with_device_type(target);
            if let Some(dir) = &options.openvino_cache {
                std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
                ep = ep.with_cache_dir(dir.to_string_lossy());
            }
            builder.with_execution_providers([ep.build().error_on_failure()]).map_err(ort::Error::<()>::from).with_context(|| {
                format!(
                    "OpenVINO couldn't use `{target}` (ONNX Runtime's log line above says why). On Linux, an Intel GPU \
                     also needs Intel's compute runtime (intel-opencl-icd), read access to /dev/dri/by-path/*-render, \
                     and, for an Arc card, Resizable BAR (`rivet devices` reports it)"
                )
            })?
        }
        #[allow(unreachable_patterns)]
        other => bail!("{other:?} needs this example built with its feature (`cuda` / `directml` / `openvino`)"),
    };
    let session = builder.commit_from_file(model).with_context(|| format!("loading {}", model.display()))?;
    let bound = match device {
        Device::Cuda(id) if cuda_graph => Some(Bound::new(&session, *id)?),
        _ => None,
    };
    Ok(Slot { session, bound })
}

impl YoloHook {
    /// Loads `model` (an ONNX export).
    pub fn load(model: &Path, options: LoadOptions) -> Result<YoloHook> {
        let first_slot = session(model, &options)?;
        let first = &first_slot.session;

        let [input] = first.inputs() else { bail!("a YOLO model has one input; this one has {}", first.inputs().len()) };
        let input_name = input.name().to_string();
        let precision = match input.dtype().tensor_type() {
            Some(TensorElementType::Float32) => Precision::F32,
            Some(TensorElementType::Float16) => Precision::F16,
            other => bail!("the model's input is {other:?}; YOLO takes f32 (or f16) pixels"),
        };
        // [1, 3, H, W]; a dynamic side (-1) is taken as 640.
        let shape = input.dtype().tensor_shape().context("the model's input isn't a tensor")?;
        let side = |d: i64| if d > 0 { d as u32 } else { 640 };
        let input_size = match shape[..] {
            [_, 3, h, w] => (side(w), side(h)),
            _ => bail!("expected a [1, 3, H, W] input; this model's is {:?}", &shape[..]),
        };

        let names = match options.names.clone() {
            Some(n) => n,
            None => first
                .metadata()
                .ok()
                .and_then(|m| m.custom("names"))
                .and_then(|s| yolo::parse_names(&s))
                .unwrap_or_else(|| yolo::COCO.iter().map(|s| s.to_string()).collect()),
        };
        let layout = match options.layout {
            Some(l) => l,
            None => {
                let out = first.outputs().first().context("the model has no output")?;
                let shape = out.dtype().tensor_shape().context("the model's output isn't a tensor")?;
                Layout::infer(shape, names.len())?
            }
        };

        let input = Input { name: input_name, size: input_size, precision };
        let mut sessions = vec![Worker::new(first_slot, &input)?];
        for _ in 1..options.sessions.max(1) {
            sessions.push(Worker::new(session(model, &options)?, &input)?);
        }
        let mut hook = YoloHook {
            sessions,
            next: AtomicUsize::new(0),
            input,
            layout,
            names,
            warm_up: None,
            min_score: 0.25,
            iou: 0.45,
            max_detections: 300,
            sampling: FrameSampling::every_seconds(1.0),
            refuse: BTreeSet::new(),
            refuse_score: None,
            draw_to: None,
        };
        if options.warm_up {
            let started = Instant::now();
            let (w, h) = hook.input.size;
            let blank = vec![0.5f32; (3 * w * h) as usize];
            for worker in &hook.sessions {
                for _ in 0..worker.warm_up_runs() {
                    worker.claim(true).expect("waits").run(&hook.input, blank.clone())?;
                }
            }
            hook.warm_up = Some(started.elapsed());
        }
        Ok(hook)
    }

    pub fn layout(&self) -> Layout {
        self.layout
    }

    pub fn input_size(&self) -> (u32, u32) {
        self.input.size
    }

    pub fn precision(&self) -> Precision {
        self.input.precision
    }

    pub fn sessions(&self) -> usize {
        self.sessions.len()
    }

    /// How long warming the sessions up took, if they were.
    pub fn warm_up_time(&self) -> Option<Duration> {
        self.warm_up
    }

    pub fn names(&self) -> &[String] {
        &self.names
    }

    /// Runs the model on a free session, or waits for one.
    fn infer(&self, planar: Vec<f32>) -> Output {
        let n = self.sessions.len();
        let start = self.next.fetch_add(1, Ordering::Relaxed);
        let free = (0..n).find_map(|i| self.sessions[(start + i) % n].claim(false));
        let claimed = free.unwrap_or_else(|| self.sessions[start % n].claim(true).expect("waits"));
        claimed.run(&self.input, planar)
    }

    /// The detections in `frame`, in its own pixels, and where the time went.
    pub fn detect(&self, frame: &VideoFrame) -> Result<(Vec<Detection>, Timing)> {
        let ms = |since: Instant| since.elapsed().as_secs_f64() * 1000.0;
        let started = Instant::now();
        let (w, h) = self.input.size;
        let (planar, letterbox) = planar_f32_letterboxed(frame, w, h, [114, 114, 114])?;
        let prepare_ms = ms(started);

        let started = Instant::now();
        let (shape, data) = self.infer(planar)?;
        let inference_ms = ms(started);

        let started = Instant::now();
        let found = yolo::decode(self.layout, &shape, &data, self.names.len(), self.min_score)?;
        let found = match self.layout {
            Layout::EndToEnd => found,
            _ => yolo::nms(found, self.iou, self.max_detections),
        };
        let found = found.into_iter().map(|d| to_source(d, &letterbox)).collect();
        Ok((found, Timing { prepare_ms, inference_ms, decode_ms: ms(started) }))
    }

    fn label(&self, class: usize) -> &str {
        self.names.get(class).map_or("?", String::as_str)
    }

    /// Detects, records, and judges one picture. `at` names it in a rejection.
    fn handle(&self, frame: &VideoFrame, at: &str, file_stem: &str) -> Result<HookOutcome> {
        let (found, timing) = self.detect(frame)?;

        let mut counts: BTreeMap<&str, u64> = BTreeMap::new();
        for d in &found {
            *counts.entry(self.label(d.class)).or_default() += 1;
        }
        let detections: Vec<Value> = found
            .iter()
            .map(|d| {
                json!({
                    "label": self.label(d.class),
                    "class": d.class,
                    "score": round(d.score, 3),
                    "box": [round(d.x, 1), round(d.y, 1), round(d.w, 1), round(d.h, 1)],
                })
            })
            .collect();
        if let Some(dir) = &self.draw_to {
            crate::draw::boxes(frame, &found, &dir.join(format!("{file_stem}.png")))?;
        }

        let outcome = HookOutcome::proceed()
            .annotate("detections", detections)
            .annotate("counts", json!(counts))
            .annotate("prepare_ms", round(timing.prepare_ms as f32, 2))
            .annotate("inference_ms", round(timing.inference_ms as f32, 2))
            .annotate("decode_ms", round(timing.decode_ms as f32, 2));
        let threshold = self.refuse_score.unwrap_or(self.min_score);
        Ok(match found.iter().find(|d| d.score >= threshold && self.refuse.contains(self.label(d.class))) {
            Some(d) => outcome.rejecting(format!("`{}` detected {at} (score {:.2})", self.label(d.class), d.score)),
            None => outcome,
        })
    }
}

/// One run of the model on a planar `[1, 3, H, W]` picture; the output's
/// shape and values, as `f32` whatever the model computes in.
fn run(input: &Input, slot: &mut Slot, planar: Vec<f32>) -> Output {
    let (w, h) = input.size;
    let shape = [1usize, 3, h as usize, w as usize];
    let Slot { session, bound } = slot;
    match bound {
        // Into the host input, up to the bound one, replay, back from the
        // bound output.
        Some(bound) => {
            match input.precision {
                Precision::F32 => bound.host_input.try_extract_tensor_mut::<f32>()?.1.copy_from_slice(&planar),
                Precision::F16 => {
                    for (to, from) in bound.host_input.try_extract_tensor_mut::<f16>()?.1.iter_mut().zip(&planar) {
                        *to = f16::from_f32(*from);
                    }
                }
            }
            bound.host_input.copy_into(&mut bound.input)?;
            let ran = session.run_binding(&bound.binding)?;
            ran[0].downcast_ref::<DynTensorValueType>()?.copy_into(&mut bound.host_output)?;
            as_f32(&bound.host_output)
        }
        None => {
            let tensor: DynTensor = match input.precision {
                Precision::F32 => Tensor::from_array((shape, planar))?.upcast(),
                Precision::F16 => Tensor::from_array((shape, planar.into_iter().map(f16::from_f32).collect::<Vec<_>>()))?.upcast(),
            };
            let outputs = session.run(ort::inputs![input.name.as_str() => tensor])?;
            as_f32(&outputs[0])
        }
    }
}

/// An output tensor's shape and values, as `f32` whatever it holds.
fn as_f32<T: TensorValueTypeMarker + ?Sized>(output: &ort::value::Value<T>) -> Result<(Vec<i64>, Vec<f32>)> {
    Ok(match output.try_extract_tensor::<f32>() {
        Ok((shape, data)) => (shape.to_vec(), data.to_vec()),
        Err(_) => {
            let (shape, data) = output.try_extract_tensor::<f16>()?;
            (shape.to_vec(), data.iter().map(|v| v.to_f32()).collect())
        }
    })
}

/// A detection in the model's input → the frame, through the letterbox.
fn to_source(d: Detection, letterbox: &Letterbox) -> Detection {
    let (x, y, w, h) = letterbox.box_to_source(d.x, d.y, d.w, d.h);
    Detection { x, y, w, h, ..d }
}

fn round(v: f32, places: i32) -> f64 {
    let p = 10f64.powi(places);
    (f64::from(v) * p).round() / p
}

impl DecodedFrameHook for YoloHook {
    fn sampling(&self) -> FrameSampling {
        self.sampling
    }

    fn on_decoded_frame(&self, _ctx: &HookContext, f: &FrameEvent) -> Result<HookOutcome> {
        self.handle(&f.frame, &format!("at {:.2}s (frame {})", f.seconds, f.index), &format!("clip{}-frame{:06}", f.clip, f.index))
    }

    fn describe(&self) -> String {
        format!(
            "YOLO detection ({}, {}x{} {:?}, {} classes, {} session(s))",
            self.layout.as_str(),
            self.input.size.0,
            self.input.size.1,
            self.input.precision,
            self.names.len(),
            self.sessions.len()
        )
    }
}

impl StillHook for YoloHook {
    fn on_still(&self, _ctx: &HookContext, s: &StillEvent) -> Result<HookOutcome> {
        let at = if s.from_video { format!("in the still at {:.2}s", s.seconds) } else { "in the image".to_string() };
        self.handle(&s.frame, &at, &format!("still{:03}", s.index))
    }

    fn describe(&self) -> String {
        DecodedFrameHook::describe(self)
    }
}
