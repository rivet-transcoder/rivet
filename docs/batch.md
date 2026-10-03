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
| `input` | path or glob | **Required.** A literal file (must exist: a missing one fails the run before any job starts), or a glob (`*` `?` `[…]`) that expands to one job per match. |
| `output` | path | File or directory — see [output rules](#output-rules). Optional (derived from `output_dir`). |
| `mode` | `single` \| `hls` \| `audio` | Output shape (default `single`). `audio` writes the audio alone as `<stem>.mp3` (`.flac`, `.m4a`, `.opus` / `.ogg` as the codec or `audio_container` has it), as does a `single` job whose input has no video. `image` is refused: stills are [`rivet image`](cli.md#rivet-image). |
| `codec` | `av1` \| `h264` \| `h265` \| `vp9` \| `vp8` \| `mpeg2` \| `mpeg4` \| `prores` \| `prores-<profile>` | Output video codec (default `av1`). The last five are rivet's own software encoders; see [output spec](output-spec.md#the-other-codecs-vp9-vp8-mpeg-2-mpeg-4-part-2-prores). |
| `container` | `mp4` \| `mov` \| `webm` | The file of a single-file output (default: the codec's own — `mov` for ProRes, `webm` for VP8 / VP9, `mp4` otherwise); a multi-rung directory gets `<label>.<ext>`. |
| `prores_profile` | `proxy` \| `lt` \| `422` \| `hq` \| `4444` \| `4444xq` | The ProRes profile with `codec: prores`. |
| `rungs` | list of `WxH` | Explicit renditions, e.g. `["1280x720", "640x360"]` — each a maximum box the source is fitted into; `"1280x720@3M"` codes that rung to a bitrate (`"1280x720@standard"`: the rate it would have with none named anywhere), `"1080x1920:cover:fixed"` sets its own fitting. |
| `fit` | string | `contain` (default), `cover`, `pad` or `stretch` — as the CLI's `--fit`. |
| `orientation` | string | `auto` (default) or `fixed` — as the CLI's `--orientation`. |
| `upscale` | bool | Let a rung be larger than the source (default `false`). |
| `ladder` | bool | Derive a standard ABR ladder from the source. |
| `max_short_side` | int or string | Cap the ladder's tallest rung's short side; `standard` states the default, 1080. |
| `segment_seconds` | number | HLS segment length (default 4). |
| `crf` | int | Constant rate factor (names the quantiser; `target` is then not consulted). |
| `target` | `visually_lossless` \| `high` \| `standard` \| `low` \| `vmaf=N` | Perceptual quality target for every rung — as the CLI's `--target`. |
| `gop` | int or string | GOP length for every rung: frames (`48`) or seconds (`"2s"`, `"1.5s"`); default two seconds, which `"2s"` states — as the CLI's `--gop`. |
| `video_bitrate` | string | Bitrate for every rung without its own `@RATE`, e.g. `"3M"`; `standard` states the default, none — as the CLI's `--video-bitrate`. An average rate is coded by the software encoders only (H.264 / H.265, AV1, VP9, MPEG-2, MPEG-4 Part 2); a constant one (`rate_mode: cbr`) by the GPU encoders and the software H.264 / H.265 encoder. |
| `video_buffer` | string | Coded picture buffer for the bitrate rungs, e.g. `"500ms"` (`"0"` for none; default one second) — as the CLI's `--video-buffer`. |
| `rate_mode` | `average` \| `cbr` | How the bitrate rungs are coded: `average` (default; also `abr`) or `cbr` (also `constant`), a constant rate within the buffer, coded by QSV, NVENC, AMF and the software H.264 / H.265 encoder (not the software AV1 or VP9 encoders, which refuse it by name). A `cbr` rung with no rate of its own takes `video_bitrate`, else a default for its codec, size and frame rate — as the CLI's `--rate-mode`. |
| `video_speed` | `draft` \| `standard` \| `archive` | Encoder effort for every rung (default `standard`), mapped by each encoder onto its own presets (NVENC P5 / P6 / P7; software VP9 from fixed partitions to a searched one; software AV1 its effort, speed 8 / 6 / 4, and motion search range); an `encode_policy` `speed=` word wins — as the CLI's `--video-speed`. |
| `audio` | `auto` \| `opus` \| `mp3` \| `aac` \| `he-aac` \| `he-aacv2` \| `vorbis` \| `ac3` \| `eac3` \| `dts` \| `flac` \| `alac` \| `drop` | Audio policy, as the CLI's `--audio`. See [lossless audio](lossless-audio.md). |
| `audio_quality` | number, `-1` … `10` | Vorbis quality (`audio: vorbis`; default 5), as the CLI's `--audio-quality`. |
| `audio_bit_depth` | `source` \| `16` \| `24` | Bit depth of FLAC / ALAC output. Default `source`. |
| `he_aac` | `auto` \| `passthrough` \| `core` | An HE-AAC source, which rivet decodes in full: `auto` (default) treats it as any AAC track; `passthrough` never decodes it; `core` decodes only its AAC-LC core (half the rate, lower bandwidth). |
| `audio_decode_deny` | string, e.g. `"aac"` or `"aac,mp3"` | Source audio codecs that may not be decoded (`aac`, `ac3`, `alac`, `dts`, `eac3`, `flac`, `mp2`, `mp3`, `opus`, `pcm`, `vorbis`; default none): passed through where the output can carry them, the job refused where it needs their PCM. |
| `metadata_keep` | string, e.g. `"location:approximate,device"` | The source's identifying metadata to carry into the output: `location` (or `location:approximate`), `capture_time` (or `capture_time:date`), `device` (or `device:all`), `descriptive`, `all`, `none`; default none. Single-file and `audio`-mode output; HLS refuses it. As the CLI's `--metadata-keep`. |
| `flac_compression` | `fast` \| `default` \| `best` | FLAC compression effort. |
| `audio_container` | `auto` \| `mp3` \| `flac` \| `mp4` \| `ogg` | The file of an `audio`-mode output; the output path gets its extension (`.mp3` / `.flac` / `.m4a` / `.opus` / `.ogg`). `auto` follows the codec. |
| `audio_bitrate` | string | Target for transcoded audio, e.g. `"240k"`; `standard` states the default: Opus from the channel layout, AAC 64k mono / 128k stereo / 384k 5.1 / 512k 7.1, HE-AAC 48k stereo, HE-AAC v2 32k, MP3 128k stereo / 64k mono, AC-3 192k stereo / 448k 5.1, E-AC-3 192k / 384k / 512k 7.1, DTS 1536k. |
| `audio_channels` | `source` \| `mono` \| `stereo` \| `5.1` \| `7.1` | Output channel layout; downmixes, never upmixes. |
| `audio_stereo_fallback` | bool | HLS: a stereo downmix rendition beside a surround one. |
| `audio_filter` | string | Audio filter chain, e.g. `"channelmap=FL-FL\|FR-FR:stereo"`. See [audio filters](audio-filters.md). |
| `subtitles` | string | `all` (default), `none`, or a language list such as `eng,deu`. Text subtitles → a tx3g track per language (MP4) or a WebVTT rendition per language (HLS). |
| `color` | `sdr` \| `hdr10` \| `hlg` \| `passthrough` | Color / tonemap policy. |
| `bit_depth` | `auto` \| `8bit` \| `10bit` | Output bit depth (alias: `pixel_format`). |
| `chroma_downsample` | `box` \| `lanczos` | 4:4:4 → 4:2:0 chroma filter for 4:4:4 sources (default `box`; alias: `chroma_filter`). |
| `seam` | `parallel` \| `constqp` | Multi-GPU single-file chunk-seam *quality*. (`serial` still parses, as the older spelling of `encode: single`.) |
| `encode` | `all` \| `per-rung` \| `single` \| `gpu:N` \| `family:nvidia\|amd\|intel` | The encode plan: which cards, and how the work is laid across them. Wins over `gpu` / `gpu_family` / `single_gpu`. Same words and meaning as the CLI's `--encode` — every surface interprets through [`rivet::settings`](../crates/rivet/src/settings.rs). |
| `decode` | `auto` \| `whole` \| `fastest` \| `gpu:N` \| `ranges:N` | The decode plan: which card(s), and whether the decode is one pump or split into ranges. Wins over `decode_gpu`. |
| `max_fps` | number or string | Cap the output frame rate; `source` states the default, no cap. |
| `gpu` | int | Pin encode to a GPU index. |
| `gpu_family` | `nvidia` \| `amd` \| `intel` | Restrict encode to a vendor. |
| `single_gpu` | bool | Use one GPU (serial). |
| `decode_gpu` | int | Pin the decode pump to a GPU. |
| `width`, `height` | int | Scale a single-rung output (ignored when `rungs`/`ladder` is set). |
| `filter` | string **or** list | Video filters — a chain string `"crop=1280:720,hflip"`, or a structured list of objects (below). See [Video filters](filters/README.md) for the full set. |

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

An audio-only job (`mode: audio`, or an input with no video) is a single file
whose derived name ends in `.mp3`, `.flac` or `.m4a` instead of `.mp4`.

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
- **Relative paths are manifest-relative.** `input`, `output`, `output_dir`, and
  a `filter` overlay's `image` path all resolve against the manifest file's
  directory, so a manifest + its media (including overlay logos) move together.
  Absolute paths pass through.
- **Sequential, fail-soft.** Jobs run one at a time (the GPU is the bottleneck and
  the [GPU pool](pipeline.md#4-the-multi-gpu-lease-engine--the-rung-benefit)
  already parallelizes a single job across devices). A failed job is recorded and
  — unless `on_error: stop` — the run continues; the command exits non-zero if any
  job failed, after printing a per-job summary.

`--dry-run` parses the manifest (unknown keys fail here), merges defaults,
expands globs, and prints each planned job with its `mode`, `ladder`, `rungs`,
`crf`, `color` and `output` — the fast way to check a manifest before
committing GPU time. Setting values are not interpreted until a job runs, so a
misspelled value (`color: hdr11`) passes the dry run and fails its job.

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
