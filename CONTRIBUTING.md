# Contributing to rivet

Thanks for your interest! rivet is an **opinionated, web-first** video transcoder,
and the most useful contributions sharpen that focus rather than broaden it.
Please skim the **Scope** section before starting a large change — it saves us
both time.

## The north star: get video onto the web, well

rivet exists to turn an arbitrary input file into video that **plays great on the
web** — browser-decodable, ABR-ready, royalty-clean, and fast to produce. Every
feature is judged against that goal. rivet is **not trying to be FFmpeg**: it is
not a universal media-conversion Swiss-army knife, and *"FFmpeg supports it"* is
not, by itself, a reason for rivet to.

Concretely, "web-first" means:

- **Output codecs are the web set** — AV1 (the default, royalty-clean), H.264, and
  H.265: the codecs browsers and devices actually decode. 4:2:0, 8- and 10-bit.
- **Output containers are what streams** — faststart MP4 and segment-aligned
  CMAF/HLS.
- **Color is web-correct** — BT.709 SDR by default; HDR (PQ/HLG) tonemapped or
  signalled so it renders right in a browser.
- **Audio is web audio** — AAC / Opus passthrough; MP3 / Vorbis → Opus.

Ingest is deliberately **broad** (you transcode whatever users upload); output is
deliberately **narrow** (the web). Keep that asymmetry in mind.

## Scope: in vs out

**In scope** — PRs very welcome:

- Improving the **web output path**: encoder quality / speed / correctness for
  AV1 / H.264 / H.265, the MP4 / CMAF / HLS muxers, playlist + `CODECS=` string
  correctness, browser/device compatibility fixes.
- The **job / service layer**: the engine, progress reporting, the CLI / HTTP /
  batch surfaces, multi-GPU scheduling.
- **Cross-vendor GPU** encode/decode (NVENC / AMF / QSV) correctness and hardware
  verification.
- Ingesting **common, real-world uploads** better — the formats people actually
  have (the current decode set + mainstream containers).
- **Color / HDR** correctness, **performance** (decode-once, AVX2 kernels, bounded
  memory), **docs**, and **tests**.

**Out of scope** — likely to be declined (open a discussion first if you disagree):

- **New *output* codecs beyond the web set** (VP9 output, ProRes output, "codec X
  because FFmpeg has it"). The web doesn't need them; AV1 is the future-proof,
  royalty-clean target.
- **Niche / legacy *input* formats** that aren't real-world uploads — dead or
  obscure codecs (Theora, RealVideo, Cinepak, …) and exotic containers nobody
  streams. We ingest what users actually upload, not the long tail.
- **Professional / broadcast features** unrelated to web delivery — 4:4:4 / 12-bit
  mastering pipelines, SDI, frame-accurate editorial workflows, exotic pro
  containers, and the like.
- **FFmpeg-completeness for its own sake.** rivet stays small and focused on
  purpose; breadth is a non-goal, not a missing feature.

> The filter question for any feature: **"does this make video play better on the
> web, for real users?"** If yes, it's probably in. If the honest answer is *"it
> makes rivet more like FFmpeg,"* it's probably out.

Not sure where your idea falls? **Open an issue or discussion before writing
code** — especially for anything that touches scope. We'd rather say "yes, here's
how" up front than decline a finished PR.

## Development

The default build links native libraries, so it needs a C toolchain plus:

- **nasm** — x86 assembly for the codec stack (rav1d, openh264).
- **CMake** + a C/C++ compiler — builds libopus (and Intel oneVPL with `qsv`).

```sh
cargo build                     # default (no hardware encoder)
cargo build --features nvidia   # + NVENC encode / NVDEC decode (hand-rolled FFI; Win + Linux)
cargo build --features ffmpeg   # libavcodec decode tier (needs FFmpeg ≥7.0 dev libs + LLVM)
```

On Windows the project links the static MSVC CRT; with CMake 4.x, set
`CMAKE_POLICY_VERSION_MINIMUM=3.5` so libopus's older `CMakeLists.txt` configures.
See [README → Building](README.md#building) and [`docs/`](docs/) for the full map.

## Before you submit

- **Tests pass.** Run the lib suites and add tests for new behaviour:
  ```sh
  cargo test -p rivet-codec      --lib --features serde
  cargo test -p rivet-container  --lib
  cargo test -p rivet-transcoder --lib --features server,batch,ipc,thumbnail
  ```
  CI runs these on Linux on every PR — keep it green.
- **Refactors change no behaviour.** If a PR claims to be a pure refactor, the
  tests must be unchanged and still pass (same `#[test]` set, same assertions).
- **Match the surrounding code** — naming, comment density, idioms. When a vendored
  library replaces a hand-rolled scaffold, **delete the scaffold** (don't keep dead
  code "for reference").
- **One concern per PR**, with a clear description of the *why*.
- **Hardware-touching code:** NVENC / AMF / QSV changes you can't verify on your own
  silicon should say so, and describe how to verify on the target hardware.

## Conventions

- **Fail fast, typed.** A host that can't do something (no AV1-encode silicon, an
  unsupported pixel format) returns a clear, typed error — it never silently
  degrades to a slow or wrong path.
- **GPU FFI is hand-rolled in-tree**, mirroring the vendor SDK headers — no
  third-party GPU wrapper crates, no bindgen, no build-time SDK link (so it builds
  on Windows MSVC *and* Linux). New vendor work follows that pattern.
- **Encode is GPU-first**, with `ffmpeg` (software) as the explicit fallback tier —
  not the default.

## License

rivet is released under the **Open Encoding Attribution License** (source-available,
royalty-free for every use; see [LICENSE.md](LICENSE.md)). By submitting a
contribution you agree it is licensed to the project under those same terms
(inbound = outbound), and that you have the right to contribute it.
