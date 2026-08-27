//! Batch manifest DSL — convert many files from one YAML or JSON file.
//!
//! A **manifest** is a small declarative document: a list of `jobs`, each a
//! file (or glob) to convert plus any per-job settings, on top of optional
//! shared `defaults`. Every settings field is the same canonical knob set the
//! CLI flags / HTTP API / IPC header use ([`crate::settings::TranscodeSettings`]),
//! so a job is just "an input, an output, and a spec." The engine reads the
//! manifest, expands globs, merges defaults, builds an [`OutputSpec`] per job
//! via [`TranscodeSettings::into_spec`], runs it on the same job engine
//! ([`crate::run_job_blocking`]), and writes the outputs.
//!
//! ```yaml
//! defaults:
//!   crf: 28
//!   color: sdr
//! jobs:
//!   - input: in/a.mkv
//!     output: out/a.mp4
//!     crf: 24            # per-job override
//!   - input: in/b.mov
//!     output: out/b      # HLS asset root (a directory)
//!     mode: hls
//!     ladder: true
//!   - input: "clips/*.mp4"   # glob -> one job per match
//!     output: out/           # directory: each file -> out/<stem>.mp4
//! ```
//!
//! Relative `input`/`output`/`output_dir` paths are resolved **relative to the
//! manifest file's directory**, so a manifest plus its media is portable. The
//! full DSL reference is in `docs/batch.md`.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use serde::Deserialize;

use crate::job::RungArtifact;
use crate::settings::{
    TranscodeSettings, parse_audio, parse_bit_depth, parse_color, parse_decode_plan,
    parse_encode_plan, parse_gpu_family, parse_mode, parse_quality_target, parse_rung,
    parse_video_codec,
};
use crate::spec::OutputMode;

/// One job (or the shared `defaults`): an optional input/output plus every
/// transcode knob. Field names + string values mirror the CLI flags and the
/// HTTP `spec` exactly. Unknown keys are rejected so typos surface immediately.
#[derive(Debug, Default, Clone, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "snake_case")]
pub struct JobSpec {
    /// Input file path or glob (e.g. `clips/*.mp4`). Required on a job, ignored
    /// in `defaults`.
    pub input: Option<String>,
    /// Output file (single-file single-rung) or directory (HLS / multi-rung,
    /// or a trailing slash). Optional — see `output_dir` and the path rules.
    pub output: Option<String>,

    // ── the transcode spec (same vocabulary everywhere) ──
    pub mode: Option<String>,
    /// Output video codec: `av1` (default), `h264`, or `h265`.
    pub codec: Option<String>,
    #[serde(default)]
    pub rungs: Option<Vec<String>>,
    pub ladder: Option<bool>,
    pub max_short_side: Option<u32>,
    pub segment_seconds: Option<f32>,
    pub crf: Option<u8>,
    /// Perceptual quality target: `visually_lossless` | `high` | `standard` |
    /// `low` | `vmaf=N`. Not consulted for a job that sets `crf`.
    pub target: Option<String>,
    /// GOP length in frames (default: two seconds).
    pub gop: Option<u32>,
    pub audio: Option<String>,
    /// Target Opus bitrate for transcoded audio, e.g. `"240k"` or `240000`.
    pub audio_bitrate: Option<String>,
    /// Audio filter chain applied before the Opus encoder, e.g.
    /// `"channelmap=FL-FL|FR-FR|FC-FC|LFE-LFE|SL-BL|SR-BR:5.1"`.
    pub audio_filter: Option<String>,
    /// Subtitle tracks to carry: `all` (default), `none`, or a language list
    /// such as `eng,deu`.
    pub subtitles: Option<String>,
    pub color: Option<String>,
    #[serde(alias = "pixel_format")]
    pub bit_depth: Option<String>,
    pub seam: Option<String>,
    pub max_fps: Option<f64>,
    /// The encode plan as one value: `all` (default), `per-rung`, `single`,
    /// `gpu:N`, `family:nvidia|amd|intel`. Wins over `gpu` / `gpu_family` /
    /// `single_gpu`, which are the older per-field spellings and still work.
    pub encode: Option<String>,
    pub gpu: Option<u32>,
    pub gpu_family: Option<String>,
    pub single_gpu: Option<bool>,
    /// The decode plan as one value: `auto` (default), `whole`, `fastest`,
    /// `gpu:N`, `ranges:N`. Wins over `decode_gpu`, the older numeric pin.
    pub decode: Option<String>,
    pub decode_gpu: Option<u32>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    /// Video filters applied before scaling. Either a chain string
    /// (`"crop=1280:720,hflip"`) or a structured list of objects
    /// (`[{crop: {w: 1280, h: 720}}, hflip]`). See [`codec::filter::FilterSpec`].
    pub filter: Option<codec::filter::FilterSpec>,
}

impl JobSpec {
    /// Merge `self` (a job) over `base` (the defaults): the job wins per field,
    /// the defaults fill the gaps. `input`/`output` come from the job only.
    fn over(&self, base: &JobSpec) -> JobSpec {
        macro_rules! pick {
            ($f:ident) => {
                self.$f.clone().or_else(|| base.$f.clone())
            };
        }
        JobSpec {
            input: self.input.clone(),
            output: self.output.clone(),
            mode: pick!(mode),
            codec: pick!(codec),
            rungs: pick!(rungs),
            ladder: pick!(ladder),
            max_short_side: pick!(max_short_side),
            segment_seconds: pick!(segment_seconds),
            crf: pick!(crf),
            target: pick!(target),
            gop: pick!(gop),
            audio: pick!(audio),
            audio_bitrate: pick!(audio_bitrate),
            audio_filter: pick!(audio_filter),
            subtitles: pick!(subtitles),
            color: pick!(color),
            bit_depth: pick!(bit_depth),
            seam: pick!(seam),
            max_fps: pick!(max_fps),
            encode: pick!(encode),
            gpu: pick!(gpu),
            gpu_family: pick!(gpu_family),
            single_gpu: pick!(single_gpu),
            decode: pick!(decode),
            decode_gpu: pick!(decode_gpu),
            width: pick!(width),
            height: pick!(height),
            filter: pick!(filter),
        }
    }

    /// Convert the (string) spec fields into the canonical [`TranscodeSettings`],
    /// reusing the shared `settings::parse_*` vocabulary.
    pub fn to_settings(&self) -> Result<TranscodeSettings> {
        let mut s = TranscodeSettings::default();
        if let Some(m) = &self.mode {
            s.mode = Some(parse_mode(m)?);
        }
        if let Some(c) = &self.codec {
            s.video_codec = Some(parse_video_codec(c)?);
        }
        if let Some(rungs) = &self.rungs {
            for r in rungs {
                s.rungs.push(parse_rung(r)?);
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
            s.audio_filters = codec::audio::filter::parse_chain(f)?;
        }
        if let Some(sel) = &self.subtitles {
            s.subtitles = Some(crate::settings::parse_subtitles(sel)?);
        }
        if let Some(c) = &self.color {
            s.color = Some(parse_color(c)?);
        }
        if let Some(b) = &self.bit_depth {
            s.bit_depth = Some(parse_bit_depth(b)?);
        }
        if let Some(sm) = &self.seam {
            s.apply_seam(sm)?;
        }
        s.max_fps = self.max_fps;
        s.gpu = self.gpu;
        if let Some(f) = &self.gpu_family {
            s.gpu_family = Some(parse_gpu_family(f)?);
        }
        s.single_gpu = self.single_gpu.unwrap_or(false);
        if let Some(e) = &self.encode {
            s.encode = Some(parse_encode_plan(e)?);
        }
        // `decode` is the whole plan; the older numeric `decode_gpu` pin still
        // works when it is absent (absent ⇒ Auto).
        s.decode_policy = match &self.decode {
            Some(d) => parse_decode_plan(d)?,
            None => self
                .decode_gpu
                .map_or(crate::spec::DecodePolicy::Auto, crate::spec::DecodePolicy::SpecificGpu),
        };
        s.width = self.width;
        s.height = self.height;
        if let Some(f) = &self.filter {
            s.filters = f.resolve().context("resolving filter")?;
        }
        Ok(s)
    }
}

/// A batch manifest: shared defaults + a list of jobs.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "snake_case")]
pub struct Manifest {
    /// Optional schema version (informational; only `1` is defined).
    #[serde(default)]
    pub version: Option<u32>,
    /// Base directory for outputs that don't give an explicit `output`
    /// (relative to the manifest's directory). Defaults to each input's folder.
    #[serde(default)]
    pub output_dir: Option<String>,
    /// `continue` (default) keeps going after a failed job; `stop` aborts.
    #[serde(default)]
    pub on_error: Option<String>,
    /// Settings applied to every job (each job can override per field).
    #[serde(default)]
    pub defaults: JobSpec,
    /// The jobs to run, in order.
    pub jobs: Vec<JobSpec>,
}

/// Manifest serialization format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    Yaml,
    Json,
}

impl Format {
    /// Pick a format from a file extension; defaults to YAML.
    pub fn from_path(path: &Path) -> Format {
        match path.extension().and_then(|e| e.to_str()) {
            Some(e) if e.eq_ignore_ascii_case("json") => Format::Json,
            _ => Format::Yaml, // .yaml / .yml / anything else
        }
    }
}

/// Parse a manifest from text in the given format.
pub fn parse_manifest(text: &str, format: Format) -> Result<Manifest> {
    let m: Manifest = match format {
        Format::Json => serde_json::from_str(text).context("parsing JSON manifest")?,
        Format::Yaml => serde_yaml_ng::from_str(text).context("parsing YAML manifest")?,
    };
    if m.jobs.is_empty() {
        bail!("manifest has no `jobs`");
    }
    Ok(m)
}

/// What happened to one converted file.
#[derive(Debug)]
pub struct JobOutcome {
    pub input: PathBuf,
    pub output: Option<PathBuf>,
    pub frames: u64,
    pub bytes: u64,
    pub status: JobStatus,
}

#[derive(Debug)]
pub enum JobStatus {
    Ok,
    Failed(String),
}

impl JobStatus {
    pub fn is_ok(&self) -> bool {
        matches!(self, JobStatus::Ok)
    }
}

/// The result of running a whole manifest.
#[derive(Debug, Default)]
pub struct BatchReport {
    pub outcomes: Vec<JobOutcome>,
}

impl BatchReport {
    pub fn ok_count(&self) -> usize {
        self.outcomes.iter().filter(|o| o.status.is_ok()).count()
    }
    pub fn failed_count(&self) -> usize {
        self.outcomes.len() - self.ok_count()
    }
    pub fn all_ok(&self) -> bool {
        self.failed_count() == 0
    }
}

/// One planned conversion (after defaults-merge + glob expansion). Used by the
/// dry-run preview and as the unit of work.
#[derive(Debug)]
pub struct PlannedJob {
    pub input: PathBuf,
    pub spec: JobSpec,
}

/// Expand a manifest into the concrete list of `(input file, merged spec)` jobs
/// — resolving globs and merging defaults — without running anything. This is
/// the dry-run / preview surface.
pub fn plan_manifest(manifest: &Manifest, base_dir: &Path) -> Result<Vec<PlannedJob>> {
    let mut planned = Vec::new();
    for (i, job) in manifest.jobs.iter().enumerate() {
        let merged = job.over(&manifest.defaults);
        let input = merged
            .input
            .clone()
            .with_context(|| format!("job #{} has no `input`", i + 1))?;
        for path in expand_input(&input, base_dir)? {
            planned.push(PlannedJob {
                input: path,
                spec: merged.clone(),
            });
        }
    }
    Ok(planned)
}

/// Run a manifest from a file. Relative paths resolve against the manifest's
/// directory. Returns a [`BatchReport`]; emits one `tracing::info!` per job.
pub fn run_manifest_file(path: &Path) -> Result<BatchReport> {
    let text = fs::read_to_string(path)
        .with_context(|| format!("reading manifest {}", path.display()))?;
    let manifest = parse_manifest(&text, Format::from_path(path))?;
    let base_dir = path.parent().unwrap_or(Path::new(".")).to_path_buf();
    run_manifest(&manifest, &base_dir)
}

/// Run an already-parsed manifest, resolving relative paths against `base_dir`.
pub fn run_manifest(manifest: &Manifest, base_dir: &Path) -> Result<BatchReport> {
    let stop_on_error = matches!(manifest.on_error.as_deref(), Some("stop"));
    let manifest_out_dir = manifest.output_dir.as_ref().map(|d| base_dir.join(d));

    let planned = plan_manifest(manifest, base_dir)?;
    tracing::info!(jobs = planned.len(), "batch: starting");

    let mut report = BatchReport::default();
    for (i, job) in planned.iter().enumerate() {
        let n = i + 1;
        let total = planned.len();
        tracing::info!(
            "batch: [{n}/{total}] {} -> converting",
            job.input.display()
        );
        let outcome = match run_one(&job.input, &job.spec, manifest_out_dir.as_deref(), base_dir) {
            Ok((output, frames, bytes)) => {
                tracing::info!(
                    "batch: [{n}/{total}] {} -> {} ({} frames, {} bytes)",
                    job.input.display(),
                    output.display(),
                    frames,
                    bytes
                );
                JobOutcome {
                    input: job.input.clone(),
                    output: Some(output),
                    frames,
                    bytes,
                    status: JobStatus::Ok,
                }
            }
            Err(e) => {
                let msg = format!("{e:#}");
                tracing::error!("batch: [{n}/{total}] {} FAILED: {msg}", job.input.display());
                JobOutcome {
                    input: job.input.clone(),
                    output: None,
                    frames: 0,
                    bytes: 0,
                    status: JobStatus::Failed(msg),
                }
            }
        };
        let failed = !outcome.status.is_ok();
        report.outcomes.push(outcome);
        if failed && stop_on_error {
            tracing::warn!("batch: on_error=stop — aborting after a failed job");
            break;
        }
    }
    Ok(report)
}

/// Convert a single input file. Returns `(output_path, frames, bytes)`.
fn run_one(
    input: &Path,
    spec: &JobSpec,
    manifest_out_dir: Option<&Path>,
    base_dir: &Path,
) -> Result<(PathBuf, u64, u64)> {
    let bytes = fs::read(input).with_context(|| format!("reading {}", input.display()))?;
    let info = crate::probe_bytes(&bytes).context("probing input")?;
    let settings = spec.to_settings()?;
    let mut output_spec = settings
        .into_spec(info.width, info.height)
        .context("building output spec")?;

    // Overlay image paths resolve relative to the manifest file, like
    // `input`/`output` (absolute paths pass through unchanged).
    for f in &mut output_spec.filters {
        if let codec::filter::VideoFilter::Overlay { image, .. } = f {
            *image = join_rel(base_dir, image).to_string_lossy().into_owned();
        }
    }

    let is_hls = matches!(output_spec.mode, OutputMode::Hls { .. });
    let multi = output_spec.rungs.len() > 1;
    let plan = resolve_output(
        spec.output.as_deref(),
        manifest_out_dir,
        base_dir,
        input,
        is_hls,
        multi,
    );

    let sink = Arc::new(crate::fn_sink(|_p| {}));
    match plan {
        OutputPlan::Directory(dir) => {
            fs::create_dir_all(&dir)
                .with_context(|| format!("creating output dir {}", dir.display()))?;
            let out = crate::run_job_blocking(&bytes, &output_spec, Some(&dir), sink)
                .with_context(|| format!("transcoding {}", input.display()))?;
            let mut frames = 0u64;
            let mut written = 0u64;
            for r in out.rungs {
                frames += r.frames;
                // Multi-rung single-file artifacts come back in memory; write
                // them as <label>.mp4. HLS renditions are already on disk.
                if let RungArtifact::File(b) = r.artifact {
                    written += b.len() as u64;
                    let f = dir.join(format!("{}.mp4", r.label));
                    fs::write(&f, &b).with_context(|| format!("writing {}", f.display()))?;
                }
            }
            Ok((dir, frames, written))
        }
        OutputPlan::SingleFile(target) => {
            if let Some(parent) = target.parent() {
                fs::create_dir_all(parent)
                    .with_context(|| format!("creating {}", parent.display()))?;
            }
            let out = crate::run_job_blocking(&bytes, &output_spec, None, sink)
                .with_context(|| format!("transcoding {}", input.display()))?;
            let (data, frames) = out
                .rungs
                .into_iter()
                .find_map(|r| match r.artifact {
                    RungArtifact::File(b) => Some((b, r.frames)),
                    _ => None,
                })
                .context("no single-file output produced")?;
            fs::write(&target, &data).with_context(|| format!("writing {}", target.display()))?;
            Ok((target, frames, data.len() as u64))
        }
    }
}

enum OutputPlan {
    /// Write a single MP4 to this exact path.
    SingleFile(PathBuf),
    /// HLS asset root, or a directory for multi-rung `<label>.mp4` files.
    Directory(PathBuf),
}

/// Resolve where a job's output goes. Rules:
/// - HLS / multi-rung always go to a **directory**; single-file single-rung is a
///   **file**.
/// - explicit `output` ending in `/` is a *container* dir → a `<stem>` subdir
///   (HLS/multi) or `<stem>.mp4` (single) is placed inside it.
/// - explicit `output` without a trailing slash is used verbatim (the file, or
///   the directory).
/// - no `output` → `output_dir` (or the input's folder) + `<stem>.mp4` /
///   `<stem>/`.
fn resolve_output(
    job_output: Option<&str>,
    manifest_out_dir: Option<&Path>,
    base_dir: &Path,
    input: &Path,
    is_hls: bool,
    multi: bool,
) -> OutputPlan {
    let stem = input
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "output".into());
    let wants_dir = is_hls || multi;

    if let Some(o) = job_output {
        let looks_dir = o.ends_with('/') || o.ends_with('\\');
        let p = join_rel(base_dir, o);
        if wants_dir {
            OutputPlan::Directory(if looks_dir { p.join(&stem) } else { p })
        } else if looks_dir {
            OutputPlan::SingleFile(p.join(format!("{stem}.mp4")))
        } else {
            OutputPlan::SingleFile(p)
        }
    } else {
        let base = manifest_out_dir
            .map(|d| d.to_path_buf())
            .unwrap_or_else(|| input.parent().map(Path::to_path_buf).unwrap_or_default());
        if wants_dir {
            OutputPlan::Directory(base.join(&stem))
        } else {
            OutputPlan::SingleFile(base.join(format!("{stem}.mp4")))
        }
    }
}

/// Join a possibly-relative path onto `base_dir` (absolute paths pass through).
fn join_rel(base_dir: &Path, p: &str) -> PathBuf {
    let pb = PathBuf::from(p);
    if pb.is_absolute() {
        pb
    } else {
        base_dir.join(pb)
    }
}

/// Expand an `input` (a literal path or a glob) into concrete files, relative to
/// the manifest dir. A literal path must exist; a glob may match zero files
/// (returns empty, the caller treats it as nothing to do).
fn expand_input(input: &str, base_dir: &Path) -> Result<Vec<PathBuf>> {
    let has_glob = input.contains(['*', '?', '[']);
    if !has_glob {
        let p = join_rel(base_dir, input);
        if !p.is_file() {
            bail!("input not found: {}", p.display());
        }
        return Ok(vec![p]);
    }
    let pattern = join_rel(base_dir, input);
    let pattern = pattern.to_string_lossy();
    let mut out = Vec::new();
    for entry in glob::glob(&pattern).with_context(|| format!("bad glob: {input}"))? {
        match entry {
            Ok(p) if p.is_file() => out.push(p),
            Ok(_) => {}
            Err(e) => tracing::warn!("batch: glob entry error: {e}"),
        }
    }
    out.sort();
    if out.is_empty() {
        tracing::warn!("batch: glob matched no files: {input}");
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::settings;

    const YAML: &str = r#"
output_dir: out
defaults:
  crf: 28
  color: sdr
jobs:
  - input: in/a.mkv
    output: out/a.mp4
    crf: 24
  - input: in/b.mov
    mode: hls
    ladder: true
    pixel_format: 10bit
"#;

    #[test]
    fn parses_yaml_and_merges_defaults() {
        let m = parse_manifest(YAML, Format::Yaml).unwrap();
        assert_eq!(m.jobs.len(), 2);
        assert_eq!(m.output_dir.as_deref(), Some("out"));
        // job 1 overrides crf, inherits color from defaults
        let j1 = m.jobs[0].over(&m.defaults);
        assert_eq!(j1.crf, Some(24));
        assert_eq!(j1.color.as_deref(), Some("sdr"));
        // job 2 inherits crf from defaults; pixel_format aliases bit_depth
        let j2 = m.jobs[1].over(&m.defaults);
        assert_eq!(j2.crf, Some(28));
        assert_eq!(j2.bit_depth.as_deref(), Some("10bit"));
        assert_eq!(j2.mode.as_deref(), Some("hls"));
    }

    #[test]
    fn parses_equivalent_json() {
        let json = r#"
        { "defaults": { "crf": 28 },
          "jobs": [ { "input": "a.mkv", "output": "a.mp4", "crf": 24 } ] }"#;
        let m = parse_manifest(json, Format::Json).unwrap();
        assert_eq!(m.jobs.len(), 1);
        assert_eq!(m.jobs[0].over(&m.defaults).crf, Some(24));
    }

    #[test]
    fn settings_build_from_spec() {
        let m = parse_manifest(YAML, Format::Yaml).unwrap();
        let s = m.jobs[1].over(&m.defaults).to_settings().unwrap();
        assert_eq!(s.mode, Some(settings::Mode::Hls));
        assert!(s.ladder);
        assert_eq!(s.crf, Some(28));
    }

    /// `subtitles` on a job (and in `defaults`) reaches the shared
    /// `settings::parse_subtitles`, like every other worded key.
    #[test]
    fn subtitles_field_selects_tracks_through_the_shared_vocabulary() {
        use crate::spec::SubtitlePolicy;
        let yaml = "defaults:
  subtitles: none
jobs:
  - input: a.mkv
    output: a.mp4
    subtitles: eng,deu
  - input: b.mkv
    output: b.mp4
";
        let m = parse_manifest(yaml, Format::Yaml).unwrap();
        let a = m.jobs[0].over(&m.defaults).to_settings().unwrap();
        assert_eq!(a.subtitles, Some(SubtitlePolicy::Only(vec!["eng".into(), "deu".into()])));
        let b = m.jobs[1].over(&m.defaults).to_settings().unwrap();
        assert_eq!(b.subtitles, Some(SubtitlePolicy::Drop), "the default applies where the job is silent");
        let bad = "jobs:
  - input: a.mkv
    subtitles: english
";
        assert!(parse_manifest(bad, Format::Yaml).unwrap().jobs[0].to_settings().is_err());
    }

    #[test]
    fn unknown_field_is_rejected() {
        let bad = "jobs:\n  - input: a.mkv\n    crff: 24\n";
        assert!(parse_manifest(bad, Format::Yaml).is_err());
    }

    #[test]
    fn codec_field_selects_output_codec() {
        let yaml = "jobs:\n  - input: a.mkv\n    output: a.mp4\n    codec: h265\n";
        let s = parse_manifest(yaml, Format::Yaml).unwrap().jobs[0].to_settings().unwrap();
        assert_eq!(s.video_codec, Some(crate::spec::VideoCodecPolicy::H265));
        // omitted → None → AV1 default at spec-build time
        let plain = "jobs:\n  - input: a.mkv\n    output: a.mp4\n";
        let s2 = parse_manifest(plain, Format::Yaml).unwrap().jobs[0].to_settings().unwrap();
        assert_eq!(s2.video_codec, None);
    }

    #[test]
    fn filter_structured_objects_and_string_resolve_equal() {
        use codec::filter::VideoFilter::{Crop, HFlip, Rotate};
        let expect = vec![Crop { w: 1280, h: 720, x: None, y: None }, HFlip, Rotate(90)];
        // structured object list (the DSL benefit) — block-style YAML
        let structured = "jobs:\n  - input: a.mkv\n    output: a.mp4\n    filter:\n      - crop:\n          w: 1280\n          h: 720\n      - hflip\n      - rotate: 90\n";
        let s = parse_manifest(structured, Format::Yaml).unwrap().jobs[0].to_settings().unwrap();
        assert_eq!(s.filters, expect);
        // the equivalent chain string (interop) resolves identically
        let string = "jobs:\n  - input: a.mkv\n    output: a.mp4\n    filter: \"crop=1280:720,hflip,rotate=90\"\n";
        let s2 = parse_manifest(string, Format::Yaml).unwrap().jobs[0].to_settings().unwrap();
        assert_eq!(s2.filters, expect);
        // a bogus structured filter is rejected
        let bad = "jobs:\n  - input: a.mkv\n    filter:\n      - rotate: 45\n";
        assert!(parse_manifest(bad, Format::Yaml).unwrap().jobs[0].to_settings().is_err());
    }

    #[test]
    fn empty_jobs_rejected() {
        assert!(parse_manifest("jobs: []", Format::Yaml).is_err());
    }

    #[test]
    fn format_from_extension() {
        assert_eq!(Format::from_path(Path::new("m.json")), Format::Json);
        assert_eq!(Format::from_path(Path::new("m.yaml")), Format::Yaml);
        assert_eq!(Format::from_path(Path::new("m.yml")), Format::Yaml);
    }

    #[test]
    fn output_rules() {
        let base = Path::new("/b");
        let inp = Path::new("/b/clip.mkv");
        // single-file, explicit file
        assert!(matches!(
            resolve_output(Some("out/a.mp4"), None, base, inp, false, false),
            OutputPlan::SingleFile(p) if p.ends_with("out/a.mp4")
        ));
        // single-file, trailing-slash dir -> <stem>.mp4 inside
        assert!(matches!(
            resolve_output(Some("out/"), None, base, inp, false, false),
            OutputPlan::SingleFile(p) if p.ends_with("clip.mp4")
        ));
        // hls -> directory verbatim
        assert!(matches!(
            resolve_output(Some("out/hls"), None, base, inp, true, false),
            OutputPlan::Directory(p) if p.ends_with("out/hls")
        ));
        // no output -> output_dir + <stem>.mp4
        assert!(matches!(
            resolve_output(None, Some(Path::new("/out")), base, inp, false, false),
            OutputPlan::SingleFile(p) if p == Path::new("/out/clip.mp4")
        ));
    }

    /// The shipped samples have to parse, in both formats.
    ///
    /// A sample manifest that the parser rejects is worse than no sample: it is
    /// the first thing a new user runs, and it teaches them the DSL. These
    /// files also drifted behind the schema — they demonstrated six fields of
    /// the twenty-odd that exist — so this pins them to something the parser
    /// will actually accept as the DSL grows.
    #[test]
    fn the_shipped_examples_parse() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples");

        for (name, format) in [("batch.yaml", Format::Yaml), ("batch.json", Format::Json)] {
            let path = dir.join(name);
            let text = std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
            let manifest = parse_manifest(&text, format)
                .unwrap_or_else(|e| panic!("{name} does not parse: {e}"));

            assert!(!manifest.jobs.is_empty(), "{name} declares no jobs");

            // Parsing only proves the *shape*. Enum-valued fields — bit depth,
            // seam mode, colour, GPU family — are strings until `to_settings`
            // reads them, so a sample can parse cleanly and still be teaching
            // a spelling the CLI rejects. It did: `bit_depth: 10` parses as a
            // string in YAML and is not one of `auto|8bit|10bit`.
            for (index, job) in manifest.jobs.iter().enumerate() {
                job.over(&manifest.defaults)
                    .to_settings()
                    .unwrap_or_else(|e| panic!("{name} job {index} has invalid settings: {e}"));
            }
        }
    }

    /// And they have to stay the same manifest in two notations.
    ///
    /// They are presented as equivalents, so a field demonstrated in one and
    /// not the other is a reader learning a different DSL depending on which
    /// file they opened.
    #[test]
    fn the_two_sample_formats_agree() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples");
        let yaml = parse_manifest(
            &std::fs::read_to_string(dir.join("batch.yaml")).expect("batch.yaml"),
            Format::Yaml,
        )
        .expect("batch.yaml parses");
        let json = parse_manifest(
            &std::fs::read_to_string(dir.join("batch.json")).expect("batch.json"),
            Format::Json,
        )
        .expect("batch.json parses");

        assert_eq!(yaml.jobs.len(), json.jobs.len(), "the samples describe different job counts");
    }
}
