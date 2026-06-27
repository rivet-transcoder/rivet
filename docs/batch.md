# Batch manifest DSL (YAML / JSON)

Convert many files in one run from a single declarative file. You list the files
you need converted (and how); rivet does them. The manifest is the same spec
vocabulary as everything else — each job is just **an input, an output, and a
transcode spec** — so anything you can express as CLI flags or an HTTP `spec`,
you can express here.

> Needs the **`batch`** feature: `cargo build --release --features batch`
> (pulls a YAML/JSON parser + glob). Then `rivet batch <manifest>`.

```sh
rivet batch jobs.yaml                 # run it
rivet batch jobs.yaml --dry-run       # parse + expand globs + list, convert nothing
rivet batch jobs.json --stop-on-error # abort on the first failure
```

---

## The shape

A manifest is a list of **`jobs`** on top of optional shared **`defaults`**:

```yaml
# every relative path is resolved against THIS file's directory
output_dir: out            # optional: base dir for jobs without an explicit output
on_error: continue         # continue (default) | stop
defaults:                  # optional: applied to every job; each job can override
  crf: 28
  color: sdr
  audio: auto
jobs:
  - input: in/movie.mkv    # a file
    output: out/movie.mp4
    crf: 24                # override just for this job

  - input: in/promo.mov
    output: out/promo      # a directory → HLS asset root
    mode: hls
    ladder: true
    max_short_side: 1080

  - input: "clips/*.mp4"   # a glob → one job per matching file
    output: out/           # trailing slash = directory; each → out/<name>.mp4
```

The exact-same manifest in **JSON**:

```json
{
  "output_dir": "out",
  "on_error": "continue",
  "defaults": { "crf": 28, "color": "sdr", "audio": "auto" },
  "jobs": [
    { "input": "in/movie.mkv", "output": "out/movie.mp4", "crf": 24 },
    { "input": "in/promo.mov", "output": "out/promo", "mode": "hls", "ladder": true, "max_short_side": 1080 },
    { "input": "clips/*.mp4", "output": "out/" }
  ]
}
```

Format is chosen by extension (`.json` → JSON, `.yaml`/`.yml` → YAML).

---

## Top-level keys

| Key | Type | Meaning |
|-----|------|---------|
| `version` | int | Optional schema version (only `1` is defined; informational). |
| `output_dir` | string | Base directory for jobs that omit `output` (relative to the manifest). Defaults to each input's own folder. |
| `on_error` | `continue` \| `stop` | `continue` (default) records the failure and keeps going; `stop` aborts. The `--stop-on-error` flag forces `stop`. |
| `defaults` | spec | Settings applied to **every** job; a job overrides them field-by-field. |
| `jobs` | list | The conversions, run in order. Required, non-empty. |

## Per-job keys

Each job is an `input`, an optional `output`, and any of the spec fields. **Unknown
keys are rejected** (with the line/column and the list of valid fields), so a typo
like `crff: 24` fails loudly instead of being silently ignored.

| Key | Values | Notes |
|-----|--------|-------|
| `input` | path or glob | **Required.** A literal file (must exist), or a glob (`*` `?` `[…]`) that expands to one job per match. |
| `output` | path | File or directory — see [output rules](#output-rules). Optional (derived from `output_dir`). |
| `mode` | `single` \| `hls` | Output shape (default `single`). |
| `rungs` | list of `WxH` | Explicit renditions, e.g. `["1280x720", "640x360"]`. |
| `ladder` | bool | Derive a standard ABR ladder from the source. |
| `max_short_side` | int | Cap the ladder's tallest rung. |
| `segment_seconds` | number | HLS segment length (default 4). |
| `crf` | int | Constant rate factor. |
| `speed` | int | Encoder speed preset. |
| `audio` | `auto` \| `opus` \| `drop` | Audio policy. |
| `color` | `sdr` \| `hdr10` \| `hlg` \| `passthrough` | Color / tonemap policy. |
| `bit_depth` | `auto` \| `8bit` \| `10bit` | Output bit depth (alias: `pixel_format`). |
| `seam` | `parallel` \| `constqp` \| `serial` | Multi-GPU single-file chunk-seam handling. |
| `max_fps` | number | Cap the output frame rate. |
| `gpu` | int | Pin encode to a GPU index. |
| `gpu_family` | `nvidia` \| `amd` \| `intel` | Restrict encode to a vendor. |
| `single_gpu` | bool | Use one GPU (serial). |
| `decode_gpu` | int | Pin the decode pump to a GPU. |
| `width`, `height` | int | Scale a single-rung output (ignored when `rungs`/`ladder` is set). |
| `filter` | string **or** list | Video filters — a chain string `"crop=1280:720,hflip"`, or a structured list of objects (below). See [Video filters](filters.md) for the full set. |

A job's `filter` accepts either a string or a list of filter objects — both
resolve to the same thing and are validated up front:

```yaml
jobs:
  - input: in/clip.mov
    output: out/clip.mp4
    filter:                      # structured objects
      - crop:
          w: 1920
          h: 1080                # x/y optional → centred
      - hflip
      - rotate: 90
  - input: in/other.mov
    output: out/other.mp4
    filter: "crop=1920:1080,hflip"   # …or the equivalent string
```

These are exactly the knobs in the [`OutputSpec` guide](output-spec.md) — read it
for what each one does and the valid combinations. `validate()` still runs per
job (e.g. an `hdr10` job on a build with no 10-bit encoder fails that job).

---

## Output rules

`output` is interpreted per job, and HLS / multi-rung always produce a
**directory** while a single-file single-rung job produces a **file**:

| `output` | single-file (1 rung) | HLS / multi-rung |
|----------|----------------------|------------------|
| `out/a.mp4` (a file path) | written to `out/a.mp4` | n/a — give a directory |
| `out/dir` (no trailing slash) | written to the file `out/dir` | the directory `out/dir` is the asset root |
| `out/` (trailing slash) | `out/<input-stem>.mp4` | `out/<input-stem>/` |
| *(omitted)* | `<output_dir>/<stem>.mp4` | `<output_dir>/<stem>/` |

Multi-rung single-file jobs write `<dir>/<label>.mp4` per rung (e.g.
`720p.mp4`). HLS jobs write the usual `master.m3u8` + `audio/` + `video/<h>p/`
tree into the directory. Parent directories are created as needed.

---

## How it runs

- **Defaults merge per field.** A job inherits every `defaults` value it doesn't
  set itself; `input`/`output` come from the job only.
- **Globs expand to jobs.** `clips/*.mp4` becomes one job per matching file (the
  per-job settings apply to each). A glob that matches nothing logs a warning and
  contributes no jobs.
- **Relative paths are manifest-relative.** `input`, `output`, and `output_dir`
  resolve against the manifest file's directory, so a manifest + its media move
  together. Absolute paths pass through.
- **Sequential, fail-soft.** Jobs run one at a time (the GPU is the bottleneck and
  the [GPU pool](pipeline.md#4-the-multi-gpu-lease-engine--the-rung-benefit)
  already parallelizes a single job across devices). A failed job is recorded and
  — unless `on_error: stop` — the run continues; the command exits non-zero if any
  job failed, after printing a per-job summary.

`--dry-run` does everything except the conversion: it parses, validates, expands
globs, merges defaults, and prints the planned jobs with their resolved settings —
the fast way to check a manifest before committing GPU time.

---

## Library API

The engine is also a library (same `batch` feature):

```rust
let report = rivet::run_manifest_file("jobs.yaml".as_ref())?;
println!("{} ok, {} failed", report.ok_count(), report.failed_count());
for outcome in &report.outcomes {
    // outcome.input / .output / .frames / .bytes / .status
}
```

`rivet::manifest::{parse_manifest, plan_manifest, run_manifest}` give finer
control (parse once, preview the plan, then run).
