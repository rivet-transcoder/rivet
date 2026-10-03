# Testing rivet

What a merge gate has to run so that a red test cannot hide, and why each
line is there.

## Why this page exists

Until 2026-09-14 the merge gate ran only the `--lib` test binaries. Three
integration tests had been red on develop for weeks without anyone seeing
them, and `cargo test -p rivet-codec` could not run at all without
`--features nvidia`: `tests/nvdec_smoke.rs` failed to compile, and a test
target that does not compile stops cargo before *any* target in the crate
runs. A regression in any rivet-codec integration test was invisible.

The fixes, and what each red turned out to be, are in the commits that
introduced this page. This page is the rule that keeps it from recurring.

## Environment

```sh
export CARGO_TARGET_DIR=D:/rust-target/<worktree>   # any dir; C: is small on the dev box
export CMAKE_POLICY_VERSION_MINIMUM=3.5              # CMake 4 refuses audiopus_sys's opus otherwise
git -c protocol.file.allow=always submodule update --init   # crates/{h26x,aac,ac3,dts,lossless,prores,vp8,vp9,mpeg2,mpeg4} must not be empty
```

The CMake line matters on a host with CMake 4.x: a fresh build directory fails
while configuring the opus that `audiopus_sys` bundles, because its
`CMakeLists.txt` asks for a policy version CMake 4 no longer accepts. A target
directory that already holds a built `audiopus_sys` does not rebuild it, so the
failure only shows on a new worktree or after a `cargo clean`.

## The gate

Every test target of every crate — unit tests, integration tests and doc
tests — under each feature set below. No `--lib`, no `--test` lists: every
target compiles without its hardware feature now, so a plain `cargo test -p`
runs all of them, and a new test file is in the gate the moment it exists.

```sh
cargo test --no-fail-fast -p rivet-frame
cargo test --no-fail-fast -p rivet-container
cargo test --no-fail-fast -p rivet-aac --release
cargo test --no-fail-fast -p rivet-ac3 --release
cargo test --no-fail-fast -p rivet-dts --release
cargo test --no-fail-fast -p rivet-lossless --release
cargo test --no-fail-fast -p rivet-prores -p rivet-vp8 -p rivet-vp9 -p rivet-mpeg2 -p rivet-mpeg4 --release

cargo test --no-fail-fast -p rivet-codec
cargo test --no-fail-fast -p rivet-codec --features serde,lame
cargo test --no-fail-fast -p rivet-codec --features h26x-fallback
cargo test --no-fail-fast -p rivet-codec --features rav1e-fallback,rav1d-fallback,h26x-fallback
cargo test --no-fail-fast -p rivet-codec --features nvidia
cargo test --no-fail-fast -p rivet-codec --features amd

cargo test --no-fail-fast -p rivet-transcoder
cargo test --no-fail-fast -p rivet-transcoder --features h26x-fallback
cargo test --no-fail-fast -p rivet-transcoder --features rav1e-fallback,rav1d-fallback,h26x-fallback
cargo test --no-fail-fast -p rivet-transcoder --features nvidia
cargo test --no-fail-fast -p rivet-transcoder --features nvidia,rav1e-fallback,rav1d-fallback,h26x-fallback
cargo test --no-fail-fast -p rivet-transcoder --features server,ipc,batch,thumbnail,image,lame
cargo test --no-fail-fast -p rivet-transcoder --features image,rav1d-fallback

cargo test --no-fail-fast -p rivet-yolo-example --features cuda,directml,openvino,image-jobs
```

Judge each command by three things, never by the absence of a `FAILED` line:

1. the exit code is 0,
2. it printed one `test result: ok.` line per target. A target that did not
   compile prints no `test result:` line at all — it looks like silence, not
   like a failure, and
3. its output has no `warning: ` line from the compiler (the `generated N
   warnings` summary lines count as the same warning). Test targets are
   compiled with `cfg(test)`, so an import or item used only outside tests
   warns here and nowhere else: `cargo build` of the same crate stays clean.
   The nvidia lib-test build carried one such warning (an unused
   `ColorMetadata` import in `encode/nvenc/mod.rs`) past a gate that only
   counted `cargo build` warnings.

`--no-fail-fast` matters: without it the first red target stops the run and
every target after it goes unreported.

## What each feature set adds

| Feature set | Why it is in the gate |
|---|---|
| *(none)* | Everything that needs no feature. |
| `h26x-fallback` | The software H.264 / H.265 encode tier becomes a dispatch fallback. |
| `rav1e-fallback,rav1d-fallback,h26x-fallback` | **The only set in which the software round-trip tests actually round-trip.** |
| `nvidia` | Compiles and runs `nvdec_smoke`, `nvenc_caps`, `nvenc_reset`, and the NVENC / NVDEC arms of dispatch. |
| `nvidia` + software | NVDEC decoding what rav1e encoded: the dispatch order a GPU host with the fallbacks on really runs. The only set that caught NVDEC decoding no AV1 at all (the parser was told the stream was AV1 Annex B); no other set reaches that path, because without `nvidia` rav1d decodes and without `rav1e-fallback` the AV1 tests skip. |
| `amd` | Compiles `amf_decode_pixels` and the AMF arms. |
| `serde,lame` (rivet-codec) | The structured (serde) forms of the filter types, and the MP3 encoder through the run-time-loaded LAME. |
| `server,ipc,batch,thumbnail,image,lame` | Compiles and runs `server_api` (`#![cfg(feature = "server")]`) and the unit tests behind the front-end features: the HTTP API, IPC, the batch manifest, thumbnails, still images (`mode=image`, metadata-keep into stills) and MP3 output. `ipc` serves only on Unix but compiles and tests everywhere. |
| `image,rav1d-fallback` | Still images with a software AV1 decoder, so AVIF input is decoded rather than skipped on a host whose GPU decodes no AV1. |
| `rivet-aac`, `--release` | The AAC encoder and decoder (the `crates/aac` submodule), including both against faad2's decoder (a black box). The ISO/IEC 14496-26 conformance streams run in the crate's own CI (`AAC_CONFORMANCE_DIR`). In release, as CI runs it. |
| `rivet-ac3`, `--release` | The AC-3 / E-AC-3 decoder (the `crates/ac3` submodule): its table and unit tests, and the committed 5.1 vector (made by aften) against liba52's decode of it. The full vector sweep needs `RIVET_AC3_VECTORS` (below). |
| `rivet-dts`, `--release` | The DTS codec (the `crates/dts` submodule): its unit tests, round trips through its own encoder against the known source, and the encoder's streams decoded by libdca's `dcadec` (a black box) against this decoder. |
| `rivet-lossless`, `--release` | The FLAC and ALAC encoders and decoders (the `crates/lossless` submodule): round trips, the format pieces, and both codecs against the `flac` CLI and Apple's `alacconvert` (built from Apple's open-source ALAC release by `crates/lossless/tools/build-alacconvert.sh`). rivet-codec's `lossless_oracle` runs the same checks through rivet's adapters. |
| `rivet-prores`, `rivet-vp8`, `rivet-vp9`, `rivet-mpeg2`, `rivet-mpeg4`, `--release` | The video decoders in rivet's decode chain, and their encoders (the `crates/{prores,vp8,vp9,mpeg2,mpeg4}` submodules): spec-derived unit tests, round trips through each crate's encoder, property tests on malformed input, and the conformance material each crate commits — VP8's 18 comprehensive vectors, fourteen small VP9 vectors. Release, because the vector and round-trip tests decode real pictures. The larger suites are fetched, not committed, and skip without them (below). rivet-codec's own tests cover the adapters (`decode/*_sw.rs`) and `prores_dispatch`. |
| *(none)*, both crates: `native_codec_containers` (rivet-codec), `new_codecs_e2e` (rivet-transcoder) | **The output path of VP9, VP8, MPEG-2, MPEG-4 Part 2 and ProRes**, with no feature (their encoders are in every build). `native_codec_containers` encodes a synthetic clip with each codec's adapter, muxes it into each file it goes in (WebM, MP4, QuickTime), demuxes it with the streaming demuxer and decodes it with the decoder `create_decoder` picks — codec label, size, frame count, presentation timestamps one frame apart, luma PSNR per frame against the source — and builds the files the demux mappings need (MPEG-4 with its VOL only in the `esds` / Matroska `CodecPrivate` / a `V_MS/VFW/FOURCC` header, `V_MPEG1` / `V_MPEG2`, `V_PRORES` without its frame header, MPEG-1 video in a TS, an MPEG-2 + AC-3 program stream). `new_codecs_e2e` does the same through `run_job_blocking`: a synthetic H.264 clip, the committed H.264 + AAC / MPEG-2 / VP9 fixtures and `test_media/bbb_h264_360p_short.mp4` (skipped when absent) to every codec × file, VP9 as HLS (the joined init + segments decode to every frame; `CODECS="vp09…"`), an `.mpg` source, and the audio each file carries. No other implementation is run: the oracle is the source. About a minute in a debug build (VP9 is the slow one). |
| `rivet-yolo-example` with `cuda,directml,openvino,image-jobs` | Compiles every inference backend of the YOLO hooks example and its image-job path. |

### Tests that skip, and why the software set is not optional

The round-trip tests in `crates/rivet/tests` (`fidelity_*`, `e2e`) build
their encoder through `tests/common::try_av1_encoder`. When the build has no
AV1 encoder — no NVENC AV1 silicon, no `rav1e-fallback` — they print
`SKIP: ...` to stderr and **pass**. Cargo captures stderr of a passing test,
so the `SKIP` is not visible either.

On the dev box (RTX 3090: no AV1 NVENC) a default build's
`fidelity_pattern` finishes in half a second, having encoded nothing; with
the software set it encodes and decodes all 24 frames (about a minute in a
debug build). A green default run is not evidence about those tests. To see
which tests skipped, add `-- --nocapture` and look for `SKIP:`.

Other tests skip, and pass, when a tool they compare against is missing. Each
has a variable that turns the skip into a failure, which CI sets:

| Tests | Needs | Required by |
|---|---|---|
| `crates/aac/tests/faad_oracle.rs` and the encoder's faad tests | faad2's `faad` (`FAAD` names it) | `AAC_REQUIRE_FAAD=1` |
| `crates/aac/tests/conformance.rs` | the ISO/IEC 14496-26 AAC-LC and HE-AAC streams, fetched by `crates/aac/tools/fetch_conformance.py`, in `AAC_CONFORMANCE_DIR` | `AAC_REQUIRE_CONFORMANCE=1` (the crate's CI) |
| `crates/ac3/tests/ac3_decode_vectors.rs`, the full sweep | the vectors `crates/ac3/tools/make_vectors.sh` makes (aften streams, liba52 references) and Dolby's own streams (`crates/ac3/tools/fetch_dolby_kit.sh`), in the directory `RIVET_AC3_VECTORS` names | `RIVET_AC3_REQUIRE_VECTORS=1` (the crate's CI; the committed 5.1 fixture runs regardless) |
| `crates/dts/tests/dcadec.rs` | libdca's `dcadec` (`DCADEC` names it) | `DTS_REQUIRE_DCADEC=1` |
| `crates/dts/tests/samples.rs` | the public DTS streams `crates/dts/tools/fetch_samples.sh` fetches (VideoLAN's archive), in `DTS_SAMPLES_DIR` | `DTS_REQUIRE_SAMPLES=1` (the crate's CI) |
| `crates/lossless/tests/oracle.rs`, `crates/codec/tests/lossless_oracle.rs` | the `flac` CLI, MKVToolNix (`mkvmerge`, `mkvextract`), Apple's `alacconvert` (`ALACCONVERT` names it) | `RIVET_REQUIRE_LOSSLESS_ORACLES=1` |
| the MP3 encoder tests (`lame` feature) | LAME (`libmp3lame`) | `RIVET_REQUIRE_LAME=1` |
| `crates/rivet/tests/fidelity_mediainfo.rs` | MediaArea's `mediainfo` (`MEDIAINFO` names it) | `RIVET_REQUIRE_MEDIAINFO=1` |
| `crates/rivet/tests/fit_e2e.rs`, `hls_rates.rs`, `dts_audio.rs` | an H.264 encoder (a GPU, or `h26x-fallback`); their sources are made by `tests/common/synth.rs` | — |
| `crates/prores/tests/sample.rs` | Apple-encoded ProRes frames: the first 4 MB of Probe.dev's `AppleProRes422.mov` (the curl line is at the top of the test), its path in `PRORES_SAMPLE` | — (skips when unset; the crate's CI sets it) |
| `crates/vp9/tests/vectors.rs` | the 353 public VP9 test vectors, about 34 MB, fetched into `crates/vp9/tests/vectors` by `crates/vp9/tools/fetch-vectors.sh` (`VP9_VECTOR=<substring>` narrows the run) | `VP9_REQUIRE_VECTORS=1` |
| `crates/mpeg2/tests/conformance.rs` | the ISO/IEC 13818-4 video conformance bitstreams, fetched by `crates/mpeg2/tools/fetch-conformance.sh [dir]` (default `target/conformance`), that directory in `MPEG2_CONFORMANCE_DIR` | `MPEG2_REQUIRE_CONFORMANCE=1` |
| `crates/mpeg4/tests/conformance.rs` | ITU-T H.263 and ISO/IEC 14496-4 streams (`crates/mpeg4/tools/fetch-conformance.sh`) and Xvid 1.3.7 as a black box (`crates/mpeg4/tools/xvid-vectors.sh`), in `crates/mpeg4/tests/conformance` or `MPEG4_CONFORMANCE` | `MPEG4_REQUIRE_CONFORMANCE=1` (the crate's CI) |

### Test inputs and oracles: no FFmpeg

No test, fixture or CI step runs FFmpeg (or anything linking libav*), and no
test data is fetched from FFmpeg's hosting. Test inputs are made by this
workspace's own encoders and muxers (`crates/rivet/tests/common/synth.rs`;
`cargo run --example synth_clip` writes one to disk for the CLI jobs), or are
published conformance material fetched from the standards bodies and codec
owners (ITU-T, ISO, Dolby, VideoLAN's sample archive, Xiph). Where an
independent implementation is the oracle it is run as a black box and never
read: the `flac` CLI, Apple's `alacconvert`, faad2, libdca, liba52, aften,
MKVToolNix, MediaInfo, Xvid, and the reference decoders the `h26x` crate's
tools use.

## Known failures

Reds that are already known, so a run that shows them isn't mistaken for a
new regression. Remove an entry when its fix lands.

| Test | Since | What happens |
|---|---|---|
| `fit_e2e::an_odd_sized_source_is_evened_down_with_its_colour_in_place` (`-p rivet-transcoder`, any feature set with an H.264 encoder) | seen on develop at a2ef743 (2026-10-02); first bad commit not bisected | The job fails: `shared decode pump colorspace convert (HDR-aware): BT.601→BT.709 requires even dimensions for 4:2:0 subsampling; got 853x480`. The 853×480 source reaches the BT.601→BT.709 conversion before it's evened down. Reproduced on a clean checkout of develop at a2ef743. |

## Traps

- **One `CARGO_TARGET_DIR` per worktree.** Two checkouts of this workspace
  that share a target directory reuse each other's builds of the workspace
  crates: the artifact names match, and cargo judges them fresh by the other
  tree's file times. A branch build here silently linked the `rivet-h26x` of a
  checkout at an older develop (1c9ff0a) and failed on its API; a build that had compiled
  would have tested the wrong code. Give each worktree its own directory, or
  `cargo clean -p` the workspace crates when switching.
- **Tests run in parallel inside a binary.** A fixed or pid-named temp path
  is shared by every test in that binary; `dts_audio` failed about one run in
  five because one test removed the directory the other was writing into.
  Use `tempfile::tempdir()` per test.
- **A red that only one feature set shows is still a red.** Rerun it in
  isolation before calling it a flake, and if it is one, find the race.
- **rav1d is built without debug assertions in dev and test builds, on
  purpose.** Its debug-only `DisjointMut` borrow tracker flags a 2-byte
  over-borrow in the fallback CDEF `padding()` that is never read, so it is a
  false positive. Depending on which worker borrows second, the test process
  either aborts (`0xc0000409`, "panic in a function that cannot unwind") or
  loses a worker and hangs forever in `dav1d_get_picture` / `dav1d_send_data`.
  Release builds never compile the tracker. The workspace `Cargo.toml` sets
  `[profile.dev.package.rav1d] debug-assertions = false` (overflow checks stay
  on). `crates/codec/tests/software_av1_decode_stress.rs` guards it: it pins
  itself to 2 cores (on Windows only; unpinned elsewhere, where it is weaker)
  and takes about 15 s. With the tracker forced back on
  (`cargo test --config 'profile.dev.package.rav1d.debug-assertions=true'`) it
  aborts. Don't remove the override to "see more checks".

## Not in the gate, and why

| Feature | Reason |
|---|---|
| `qsv` | No Intel GPU on the dev box; builds everywhere. |
| `dpir`, `dpir-cuda`, `dpir-cudnn` | A 130 MB model download; CUDA toolkit at build time for the GPU variants. |
| `rav1e-asm`, `rav1d-asm` | Need NASM on the build host. |
| `rivet-h26x` | The codec submodule has its own gate (conformance suites and encode sweeps, `crates/h26x/tools`), run when the submodule moves. |
