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

- **Output codecs start from the web set** — AV1 (the default, royalty-clean),
  H.264, and H.265: the codecs browsers and devices actually decode. 4:2:0, 8-
  and 10-bit. Beside them, every codec rivet decodes can be written too, by
  rivet's own clean-room encoders: VP9 and VP8 (WebM, MP4; VP9 in HLS as
  well), MPEG-2 and MPEG-4 Part 2 (MP4, QuickTime), and ProRes (QuickTime).
  The owner asked for an encoder for every codec rivet reads; the web set
  stays the default and the reason rivet exists.
- **Output containers are what streams** — faststart MP4 and segment-aligned
  CMAF/HLS; a WebM for VP8 / VP9 and a QuickTime movie (`.mov`) for ProRes;
  for audio-only output, a bare `.mp3`, a native `.flac` or an `.m4a`.
- **Color is web-correct** — BT.709 SDR by default; HDR (PQ/HLG) tonemapped or
  signalled so it renders right in a browser.
- **Audio is web audio** — Opus (the default transcode target, mono to 7.1),
  AAC-LC (rivet's own encoder, for players that cannot take Opus), AAC / AC-3 /
  E-AC-3 / DTS / Opus passthrough, and **MP3**: universally browser- and
  device-playable, and the standard audio-only deliverable (podcasts,
  previews), so it is an output in its own right — into an MP4, or alone as
  a bare `.mp3`. Layouts are downmixed to what the output carries and never
  upmixed. FLAC / ALAC on request, for lossless delivery: they play from MP4
  and HLS in the browser (see [lossless audio](docs/lossless-audio.md)).
- **Pictures are web media too** — AVIF (the default, AV1 again), WebP, JPEG
  and PNG, at the sizes a `srcset` asks for, upright, sRGB and stripped of
  metadata: posters and stills from a video, and the photos people upload
  (JPEG, PNG, WebP, AVIF, HEIC, GIF, TIFF, ...). Every still-image codec is the workspace's own. See [decisions §28](docs/decisions.md#28-still-images-are-web-media-and-get-the-webs-formats).

Ingest is deliberately **broad** (you transcode whatever users upload); output is
deliberately **narrow** (the web). Keep that asymmetry in mind.

## Scope: in vs out

**In scope** — PRs very welcome:

- Improving the **web output path**: encoder quality / speed / correctness for
  AV1 / H.264 / H.265 and the web audio set (Opus, AAC, MP3, FLAC / ALAC), the MP4 / CMAF / HLS
  muxers, playlist + `CODECS=` string correctness, channel-layout handling,
  browser/device compatibility fixes.
- The **other output codecs** — VP9, VP8, MPEG-2, MPEG-4 Part 2, ProRes — and
  their files (WebM, QuickTime): quality and speed of the clean-room encoders
  (in their own repositories: `crates/{vp9,vp8,mpeg2,mpeg4,prores}`), the
  adapters in `crates/codec/src/encode/`, the muxers, and the tests that read
  every output back with rivet's own demuxers and decoders.
- The **job / service layer**: the engine, progress reporting, the CLI / HTTP /
  batch / IPC surfaces, the hooks, multi-GPU scheduling.
- **Cross-vendor GPU** encode/decode (NVENC / AMF / QSV) correctness and hardware
  verification.
- Ingesting **common, real-world uploads** better — the formats people actually
  have (the current decode set + mainstream containers).
- **Color / HDR** correctness, **performance** (decode-once, AVX2 kernels, bounded
  memory), **docs**, and **tests**.

**Out of scope** — likely to be declined (open a discussion first if you disagree):

- **Output codecs with no clean-room implementation here** ("codec X because
  FFmpeg has it"). Every codec rivet decodes now has an encoder; a new one
  comes in as a codec written from its specification, in a repository of its
  own ([decisions §34](docs/decisions.md#34-codecs-we-dont-have-we-write-clean-room-each-in-its-own-repository)),
  not through a C library or FFmpeg. AV1 remains the default and the
  future-proof, royalty-clean target.
- **Niche / legacy *input* formats** that aren't real-world uploads — dead or
  obscure codecs (Theora, RealVideo, Cinepak, …) and exotic containers nobody
  streams. We ingest what users actually upload, not the long tail.
- **Professional / broadcast features** unrelated to web delivery — 4:4:4 / 12-bit
  mastering pipelines, SDI, frame-accurate editorial workflows, exotic pro
  containers, and the like. (ProRes and MPEG-2 *output* are in: rivet writes
  them from its 8- / 10-bit 4:2:0 pipeline. A 4:4:4 / 12-bit path through
  that pipeline is not.)
- **FFmpeg-completeness for its own sake.** rivet stays small and focused on
  purpose; breadth is a non-goal, not a missing feature.

> The filter question for any feature: **"does this make video play better on the
> web, for real users?"** If yes, it's probably in. If the honest answer is *"it
> makes rivet more like FFmpeg,"* it's probably out.

Not sure where your idea falls? **Open an issue or discussion before writing
code** — especially for anything that touches scope. We'd rather say "yes, here's
how" up front than decline a finished PR.

## Development

The default build is Rust only — every audio and video codec in it is a
workspace crate, so there is no C library to build and nothing to install:

- **Rust 1.99** or newer (the workspace's `rust-version`, which CI's MSRV job
  holds).
- No feature needs a C compiler or an assembler.
- The submodules: `git submodule update --init` (`crates/h26x`, `crates/av1`,
  `crates/aac`, `crates/ac3`, `crates/dts`, `crates/opus`, `crates/mp3`,
  `crates/vorbis`, `crates/lossless`, `crates/prores`, `crates/vp8`,
  `crates/vp9`, `crates/mpeg2`, `crates/mpeg4`, `crates/png`, `crates/jpeg`,
  `crates/webp`, `crates/imagecodecs`). Each is a repository of its own: change it there
  (commit and push inside the submodule, after `git pull --rebase` on its
  `develop`), then commit the new pointer here. `crates/imagecodecs` (GIF,
  BMP, TIFF) is a cargo workspace of its own, not a member of rivet's: test it
  with `cargo test --manifest-path crates/imagecodecs/Cargo.toml --workspace
  --release`.

The GPU features (`nvidia`, `amd`, `qsv`) `dlopen` the vendor runtime, so they
need no SDK at build time.

```sh
cargo build                     # default (no hardware encoder)
cargo build --features nvidia   # + NVENC encode / NVDEC decode (hand-rolled FFI; Win + Linux)
cargo build --features av1-sw-fallback  # + software AV1 encode fallback (pure Rust, no system libs)
cargo build --features h26x-fallback  # + software H.264 / H.265 encode (pure Rust)
cargo build -p rivet-transcoder --features image  # + still images (`mode=image`, `rivet image`)
```

The front-end features of `rivet-transcoder` are `server` (`rivet serve`),
`batch` (`rivet batch`), `ipc` (`rivet ipc`), `thumbnail` and `image`; `dpir`
(`dpir-cuda`, `dpir-cudnn`) adds the deep denoiser. `examples/yolo`
(`rivet-yolo-example`) is a workspace member too.

On Windows the project links the static MSVC CRT.
See [README → Building](README.md#building) and [`docs/`](docs/) for the full map.

## Before you submit

- **Tests pass.** Add tests for new behaviour, and run the gate in
  [docs/testing.md](docs/testing.md) — every target of every crate under each
  feature set it lists. The quick subset is the lib suites:
  ```sh
  cargo test -p rivet-codec      --lib --features serde
  cargo test -p rivet-container  --lib
  cargo test -p rivet-transcoder --lib --features server,batch,ipc,thumbnail,image
  ```
  CI runs these on Linux on every PR, with the MP3, AAC, lossless-audio and
  still-image tests that need outside tools — keep it green.
- **Refactors change no behaviour.** If a PR claims to be a pure refactor, the
  tests must be unchanged and still pass (same `#[test]` set, same assertions).
- **Match the surrounding code** — naming, comment density, idioms. When a vendored
  library replaces a hand-rolled scaffold, **delete the scaffold** (don't keep dead
  code "for reference").
- **One concern per PR**, with a clear description of the *why*.
- **Hardware-touching code:** NVENC / AMF / QSV changes you can't verify on your own
  silicon should say so, and describe how to verify on the target hardware.

## Conventions

- **Fail fast, and never *silently* fall back.** Encode and decode are GPU-only
  *by default*: a host with no encode silicon for the chosen codec, or no
  decoder for the input, **hard-fails at construction with a clear error**
  rather than degrading to a slow or wrong path.

  There are software tiers — `av1-sw-fallback` (AV1 encode)
  and `h26x-fallback` (H.264 / H.265 encode) — and they are **off by default**, which is the load-bearing
  half. (The workspace's own *decoders* are the exception: H.264 / HEVC
  (`crates/h26x`), AV1, ProRes, VP8, VP9, MPEG-1 / MPEG-2 and MPEG-4 Part 2
  (`crates/{av1,prores,vp8,vp9,mpeg2,mpeg4}`) sit in the decode chain
  unconditionally, below the hardware tiers.) They sit *below* the whole vendor
  chain, so they are a floor and never a preference, and enabling one is a
  build-time statement that slow output beats no output. A throughput fleet
  degrading silently into an encoder one to two orders of magnitude slower
  reads as a capacity problem rather than as the missing driver it actually
  is; that is the failure the default prevents, not software encoding itself. Format rejects (e.g. an unsupported chroma
  subsampling) surface as a *typed* error (`NvdecError`); missing-capability
  failures are a descriptive `anyhow` error. The multi-GPU encode pool goes
  further — it *probes* each card with `encode_capable` (building a throwaway
  encoder) and drops the encode-incapable ones, keeping them for decode. (Decode
  has no equivalent per-device probe yet — capability there is a per-vendor codec
  table, `decode_capable_gpu_indices`.)
- **GPU FFI is hand-rolled in-tree**, mirroring the vendor SDK headers — no
  third-party GPU wrapper crates, no bindgen, no build-time SDK link (so it builds
  on Windows MSVC *and* Linux). New vendor work follows that pattern.
- **Encode is GPU-first**, with `av1-sw-fallback` (software AV1) and
  `h26x-fallback` (software H.264 / H.265) as the explicit fallback tiers — not
  the default. VP9, VP8, MPEG-2, MPEG-4 Part 2 and ProRes are the exception:
  no hardware backend here encodes them, so rivet's own software encoder is
  the encoder for them in every build — there is no faster tier to fall back
  from ([decisions §35](docs/decisions.md#35-every-codec-rivet-decodes-it-can-encode-in-software-in-every-build)).
- **No FFmpeg, in any build.** No `ffmpeg-next`, no libav\* linkage, and no
  feature that adds them; see [No FFmpeg](README.md#no-ffmpeg) for what covers
  it in-tree. An opt-in libavcodec decode tier existed from 2026-08-14 to
  2026-10-02 and was removed for good (`crates/codec/Cargo.toml` records why).
  Don't reintroduce it, opt-in or otherwise.

## License

rivet is released under the **Open Encoding Attribution License** (source-available,
royalty-free for every use; see [LICENSE.md](LICENSE.md)). By submitting a
contribution you agree it is licensed to the project under those same terms
(inbound = outbound), and that you have the right to contribute it.
