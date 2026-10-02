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
use std::sync::Mutex;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use ort::session::Session;
use ort::value::Tensor;
use rivet::codec::frame::VideoFrame;
use rivet::hooks::frame::{Letterbox, rgb8_letterboxed, rgb8_to_planar_f32};
use rivet::hooks::{DecodedFrameHook, FrameEvent, FrameSampling, HookContext, HookOutcome, StillEvent, StillHook};
use serde_json::{Value, json};

use crate::yolo::{self, Detection, Layout};

/// Where inference runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Device {
    #[default]
    Cpu,
    /// NVIDIA, through ONNX Runtime's CUDA execution provider (the `cuda`
    /// feature, and CUDA 12 + cuDNN 9 on the host).
    Cuda(i32),
    /// Any DirectX 12 GPU on Windows (the `directml` feature).
    DirectMl(i32),
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

/// A YOLO detector, ready to register as a hook.
pub struct YoloHook {
    /// `Session::run` takes `&mut self`; frames may arrive from several decode
    /// threads at once, so they take turns. For more throughput, keep a pool
    /// of sessions instead.
    session: Mutex<Session>,
    input_name: String,
    input: (u32, u32),
    layout: Layout,
    names: Vec<String>,
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

impl YoloHook {
    /// Loads `model` (an ONNX export). The class names come from `names` if
    /// given, else the export's own `names` metadata, else COCO.
    pub fn load(model: &Path, device: Device, names: Option<Vec<String>>, layout: Option<Layout>) -> Result<YoloHook> {
        let mut builder = Session::builder()?;
        builder = match device {
            Device::Cpu => builder,
            #[cfg(feature = "cuda")]
            Device::Cuda(id) => builder
                .with_execution_providers([ort::ep::CUDA::default().with_device_id(id).build().error_on_failure()])
                .map_err(ort::Error::<()>::from)?,
            #[cfg(feature = "directml")]
            Device::DirectMl(id) => builder
                .with_execution_providers([ort::ep::DirectML::default().with_device_id(id).build().error_on_failure()])
                .map_err(ort::Error::<()>::from)?,
            #[allow(unreachable_patterns)]
            other => bail!("{other:?} needs this example built with its feature (`cuda` / `directml`)"),
        };
        let session = builder.commit_from_file(model).with_context(|| format!("loading {}", model.display()))?;

        let [input] = session.inputs() else { bail!("a YOLO model has one input; this one has {}", session.inputs().len()) };
        let input_name = input.name().to_string();
        // [1, 3, H, W]; a dynamic side (-1) is taken as 640.
        let shape = input.dtype().tensor_shape().context("the model's input isn't a tensor")?;
        let side = |d: i64| if d > 0 { d as u32 } else { 640 };
        let input_size = match shape[..] {
            [_, 3, h, w] => (side(w), side(h)),
            _ => bail!("expected a [1, 3, H, W] input; this model's is {:?}", &shape[..]),
        };

        let names = match names {
            Some(n) => n,
            None => session
                .metadata()
                .ok()
                .and_then(|m| m.custom("names"))
                .and_then(|s| yolo::parse_names(&s))
                .unwrap_or_else(|| yolo::COCO.iter().map(|s| s.to_string()).collect()),
        };
        let layout = match layout {
            Some(l) => l,
            None => {
                let out = session.outputs().first().context("the model has no output")?;
                let shape = out.dtype().tensor_shape().context("the model's output isn't a tensor")?;
                Layout::infer(shape, names.len())?
            }
        };

        Ok(YoloHook {
            session: Mutex::new(session),
            input_name,
            input: input_size,
            layout,
            names,
            min_score: 0.25,
            iou: 0.45,
            max_detections: 300,
            sampling: FrameSampling::every_seconds(1.0),
            refuse: BTreeSet::new(),
            refuse_score: None,
            draw_to: None,
        })
    }

    pub fn layout(&self) -> Layout {
        self.layout
    }

    pub fn input_size(&self) -> (u32, u32) {
        self.input
    }

    pub fn names(&self) -> &[String] {
        &self.names
    }

    /// The detections in `frame`, in its own pixels.
    pub fn detect(&self, frame: &VideoFrame) -> Result<Vec<Detection>> {
        let (w, h) = self.input;
        let (rgb, letterbox) = rgb8_letterboxed(frame, w, h, [114, 114, 114])?;
        let planar = rgb8_to_planar_f32(&rgb, w, h);
        let tensor = Tensor::from_array(([1usize, 3, h as usize, w as usize], planar))?;

        let found = {
            let mut session = self.session.lock().unwrap_or_else(|e| e.into_inner());
            let outputs = session.run(ort::inputs![self.input_name.as_str() => tensor])?;
            let (shape, data) = outputs[0].try_extract_tensor::<f32>()?;
            yolo::decode(self.layout, shape, data, self.names.len(), self.min_score)?
        };
        let found = match self.layout {
            Layout::EndToEnd => found,
            _ => yolo::nms(found, self.iou, self.max_detections),
        };
        Ok(found.into_iter().map(|d| to_source(d, &letterbox)).collect())
    }

    fn label(&self, class: usize) -> &str {
        self.names.get(class).map_or("?", String::as_str)
    }

    /// Detects, records, and judges one picture. `at` names it in a rejection.
    fn handle(&self, frame: &VideoFrame, at: &str, file_stem: &str) -> Result<HookOutcome> {
        let started = Instant::now();
        let found = self.detect(frame)?;
        let ms = started.elapsed().as_secs_f64() * 1000.0;

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
            .annotate("inference_ms", round(ms as f32, 1));
        let threshold = self.refuse_score.unwrap_or(self.min_score);
        Ok(match found.iter().find(|d| d.score >= threshold && self.refuse.contains(self.label(d.class))) {
            Some(d) => outcome.rejecting(format!("`{}` detected {at} (score {:.2})", self.label(d.class), d.score)),
            None => outcome,
        })
    }
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
        format!("YOLO detection ({}, {}x{}, {} classes)", self.layout.as_str(), self.input.0, self.input.1, self.names.len())
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
