//! YOLO object detection on rivet's hooks.
//!
//! ```text
//! cargo run --release -p rivet-yolo-example --features rav1e-fallback -- yolo11n.onnx input.mp4
//! cargo run --release -p rivet-yolo-example --features image-jobs -- yolo11n.onnx photo.jpg --draw boxes/
//! ```
//!
//! Registers a YOLO detector as a decoded-frame hook (video jobs) and a still
//! hook (image jobs), runs the job, and prints what it found. See
//! `docs/hooks-yolo.md`.

mod draw;
mod hook;
mod yolo;

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use rivet::hooks::{FrameSampling, HookPolicy, HookReport, Hooks, JobKind, Subject, rejection_of};

use hook::{Device, LoadOptions, YoloHook};
use yolo::Layout;

const USAGE: &str = "\
usage: yolo <model.onnx> <input> [options]

  -o, --output FILE      write the transcoded video (or the first image) here
  --conf SCORE           least score a detection is kept at (0.25)
  --iou IOU              overlap above which non-maximum suppression drops a box (0.45)
  --every SECONDS        detect on the first frame of each interval (1.0); 0 for every frame
  --max-frames N         stop detecting after N frames
  --names FILE           class names, one a line (default: the model's own, else COCO)
  --layout LAYOUT        anchors | anchors-transposed | anchors-objectness | end-to-end (default: from the output shape)
  --device DEVICE        cpu | cuda[:N] | directml[:N] (cpu)
  --sessions N           pictures the model can take at once (1)
  --cuda-graph           on CUDA, replay the model as a CUDA graph (fixed shapes, every node on the GPU)
  --no-warm-up           don't run the model once before the job starts
  --quiet                print the totals only, not a line per picture
  --ort PATH             the ONNX Runtime library (default: ORT_DYLIB_PATH, else onnxruntime.dll /
                         libonnxruntime.so / libonnxruntime.dylib next to this program)
  --refuse CLASSES       reject the job when any of these classes is detected (comma separated)
  --refuse-score SCORE   the score a refused class must reach (default: --conf)
  --background           run on the hook worker instead of the decode thread
  --codec CODEC          av1 | h264 | h265, the output video codec (av1)
  --draw DIR             write each picture it saw, with its boxes, to DIR as PNG
  --report FILE          write the job's whole hook report as JSON";

struct Args {
    model: PathBuf,
    input: PathBuf,
    output: Option<PathBuf>,
    conf: f32,
    iou: f32,
    every: f64,
    max_frames: Option<u64>,
    names: Option<PathBuf>,
    layout: Option<Layout>,
    device: Device,
    sessions: usize,
    cuda_graph: bool,
    warm_up: bool,
    quiet: bool,
    ort: Option<PathBuf>,
    refuse: BTreeSet<String>,
    refuse_score: Option<f32>,
    background: bool,
    codec: rivet::VideoCodecPolicy,
    draw: Option<PathBuf>,
    report: Option<PathBuf>,
}

fn parse_args() -> Result<Args> {
    let mut positional = Vec::new();
    let mut args = Args {
        model: PathBuf::new(),
        input: PathBuf::new(),
        output: None,
        conf: 0.25,
        iou: 0.45,
        every: 1.0,
        max_frames: None,
        names: None,
        layout: None,
        device: Device::Cpu,
        sessions: 1,
        cuda_graph: false,
        warm_up: true,
        quiet: false,
        ort: None,
        refuse: BTreeSet::new(),
        refuse_score: None,
        background: false,
        codec: rivet::VideoCodecPolicy::Av1,
        draw: None,
        report: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut value = || it.next().with_context(|| format!("{arg} needs a value\n\n{USAGE}"));
        match arg.as_str() {
            "-h" | "--help" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            "-o" | "--output" => args.output = Some(value()?.into()),
            "--conf" => args.conf = value()?.parse()?,
            "--iou" => args.iou = value()?.parse()?,
            "--every" => args.every = value()?.parse()?,
            "--max-frames" => args.max_frames = Some(value()?.parse()?),
            "--names" => args.names = Some(value()?.into()),
            "--layout" => args.layout = Some(value()?.parse()?),
            "--device" => args.device = parse_device(&value()?)?,
            "--sessions" => args.sessions = value()?.parse()?,
            "--no-warm-up" => args.warm_up = false,
            "--cuda-graph" => args.cuda_graph = true,
            "--quiet" => args.quiet = true,
            "--ort" => args.ort = Some(value()?.into()),
            "--refuse" => args.refuse.extend(value()?.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty())),
            "--refuse-score" => args.refuse_score = Some(value()?.parse()?),
            "--background" => args.background = true,
            "--codec" => {
                args.codec = match value()?.as_str() {
                    "av1" => rivet::VideoCodecPolicy::Av1,
                    "h264" => rivet::VideoCodecPolicy::H264,
                    "h265" | "hevc" => rivet::VideoCodecPolicy::H265,
                    other => bail!("unknown codec `{other}` (av1, h264, h265)"),
                }
            }
            "--draw" => args.draw = Some(value()?.into()),
            "--report" => args.report = Some(value()?.into()),
            flag if flag.starts_with('-') => bail!("unknown option {flag}\n\n{USAGE}"),
            _ => positional.push(PathBuf::from(arg)),
        }
    }
    let [model, input] = <[PathBuf; 2]>::try_from(positional).map_err(|_| anyhow::anyhow!("{USAGE}"))?;
    args.model = model;
    args.input = input;
    Ok(args)
}

fn parse_device(s: &str) -> Result<Device> {
    let (name, id) = s.split_once(':').unwrap_or((s, "0"));
    let id: i32 = id.parse().with_context(|| format!("bad device index in `{s}`"))?;
    Ok(match name {
        "cpu" => Device::Cpu,
        "cuda" => Device::Cuda(id),
        "directml" | "dml" => Device::DirectMl(id),
        _ => bail!("unknown device `{s}` (cpu, cuda[:N], directml[:N])"),
    })
}

fn main() -> Result<()> {
    let args = parse_args()?;
    hook::load_runtime(args.ort.as_deref())?;

    // The detector. One value, registered at two points below, so it is an Arc.
    let names = match &args.names {
        Some(path) => Some(
            std::fs::read_to_string(path)
                .with_context(|| format!("reading {}", path.display()))?
                .lines()
                .map(str::trim)
                .filter(|l| !l.is_empty())
                .map(String::from)
                .collect(),
        ),
        None => None,
    };
    let options = LoadOptions {
        device: args.device,
        sessions: args.sessions,
        names,
        layout: args.layout,
        warm_up: args.warm_up,
        cuda_graph: args.cuda_graph,
    };
    let mut yolo = YoloHook::load(&args.model, options)?;
    yolo.min_score = args.conf;
    yolo.iou = args.iou;
    yolo.sampling = if args.every > 0.0 { FrameSampling::every_seconds(args.every) } else { FrameSampling::all() };
    if let Some(n) = args.max_frames {
        yolo.sampling = yolo.sampling.max_frames(n);
    }
    for class in &args.refuse {
        if !yolo.names().contains(class) {
            bail!("--refuse `{class}`: the model has no such class");
        }
    }
    yolo.refuse = args.refuse.clone();
    yolo.refuse_score = args.refuse_score;
    yolo.draw_to = args.draw.clone();
    eprintln!(
        "model: {} ({} input {}x{} {:?}, {} classes) on {:?}, {} session(s){}",
        args.model.display(),
        yolo.layout().as_str(),
        yolo.input_size().0,
        yolo.input_size().1,
        yolo.precision(),
        yolo.names().len(),
        args.device,
        yolo.sessions(),
        yolo.warm_up_time().map(|t| format!(", warmed up in {} ms", t.as_millis())).unwrap_or_default()
    );
    let yolo = Arc::new(yolo);

    // A refused class must stop the job, so a detector error does too.
    let mut policy = if args.background { HookPolicy::background() } else { HookPolicy::default() };
    if !args.refuse.is_empty() {
        policy = policy.fail_closed();
    }
    let hooks = Hooks::new()
        .decoded_frames_with("yolo", Arc::clone(&yolo), policy)
        .stills_with("yolo-stills", yolo, policy);

    let input = bytes::Bytes::from(std::fs::read(&args.input).with_context(|| format!("reading {}", args.input.display()))?);

    #[cfg(feature = "image-jobs")]
    if rivet::image::sniff(&input).is_some() {
        let spec = rivet::image::ImageSpec { formats: vec![rivet::image::ImageFormat::Png], ..Default::default() };
        let session = hooks.session("yolo-image", JobKind::Image);
        let result = rivet::image::run_image_job_with_hooks(&input, &spec, &session);
        finish(&session.report(), &args)?;
        let out = result.map_err(explain)?;
        if let (Some(path), Some(first)) = (&args.output, out.artifacts.first()) {
            std::fs::write(path, &first.bytes)?;
            eprintln!("wrote {}", path.display());
        }
        return Ok(());
    }

    let info = rivet::probe_bytes(&input)?;
    let session = hooks.session("yolo-video", JobKind::Transcode);
    let spec = rivet::OutputSpec::single_file(vec![rivet::Rung::new(info.width, info.height)])
        .with_video_codec(args.codec)
        .with_hooks(session.clone());
    let result = rivet::run_job_blocking_owned(input, &spec, None, Arc::new(rivet::progress::NullSink));
    finish(&session.report(), &args)?;
    let out = result.map_err(explain)?;
    if let Some(path) = &args.output {
        match out.rungs.first().map(|r| &r.artifact) {
            Some(rivet::job::RungArtifact::File(bytes)) => {
                std::fs::write(path, bytes)?;
                eprintln!("wrote {}", path.display());
            }
            _ => bail!("the job produced no single-file output"),
        }
    }
    Ok(())
}

/// A rejection said plainly; anything else as it came.
fn explain(err: anyhow::Error) -> anyhow::Error {
    match rejection_of(&err) {
        Some(r) => anyhow::anyhow!("job rejected by hook `{}`: {}", r.hook, r.reason),
        None => err,
    }
}

/// Prints one line per picture the detector saw, then the totals.
fn finish(report: &HookReport, args: &Args) -> Result<()> {
    let mut totals: BTreeMap<String, u64> = BTreeMap::new();
    let mut pictures = 0;
    // prepare, inference, decode, and the whole call
    let mut ms = [0.0f64; 4];
    for record in report.by_hook("yolo").chain(report.by_hook("yolo-stills")) {
        let at = match &record.subject {
            Subject::Frame { index, seconds, .. } => format!("frame {index:>6} {seconds:>8.2}s"),
            other => format!("{other:<22}"),
        };
        if let Some(error) = &record.error {
            println!("{at}  error: {error}");
            continue;
        }
        pictures += 1;
        for (i, key) in ["prepare_ms", "inference_ms", "decode_ms"].into_iter().enumerate() {
            ms[i] += record.annotation(key).and_then(|v| v.as_f64()).unwrap_or(0.0);
        }
        ms[3] += record.elapsed.as_secs_f64() * 1000.0;
        let counts = record.annotation("counts").and_then(|v| v.as_object()).cloned().unwrap_or_default();
        let mut line = Vec::new();
        for (label, n) in &counts {
            let n = n.as_u64().unwrap_or(0);
            *totals.entry(label.clone()).or_default() += n;
            line.push(format!("{n} {label}"));
        }
        if !args.quiet {
            println!("{at}  {}", if line.is_empty() { "-".to_string() } else { line.join(", ") });
        }
    }
    if pictures > 0 {
        let summary: Vec<String> = totals.iter().map(|(label, n)| format!("{label} {n}")).collect();
        let each = ms.map(|t| t / pictures as f64);
        println!("{pictures} pictures; {}", if summary.is_empty() { "nothing found".into() } else { summary.join(", ") });
        println!(
            "per picture: {:.2} ms (prepare {:.2}, inference {:.2}, decode {:.2})",
            each[3], each[0], each[1], each[2]
        );
    }
    if let Some(r) = &report.rejection {
        println!("rejected: {}", r.reason);
    }
    if let Some(path) = &args.report {
        std::fs::write(path, serde_json::to_string_pretty(&report.to_json())?)?;
        eprintln!("report: {}", path.display());
    }
    Ok(())
}
