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
git -c protocol.file.allow=always submodule update --init   # crates/h26x must not be empty
```

## The gate

Every test target of every crate — unit tests, integration tests and doc
tests — under each feature set below. No `--lib`, no `--test` lists: every
target compiles without its hardware feature now, so a plain `cargo test -p`
runs all of them, and a new test file is in the gate the moment it exists.

```sh
cargo test --no-fail-fast -p rivet-frame
cargo test --no-fail-fast -p rivet-container

cargo test --no-fail-fast -p rivet-codec
cargo test --no-fail-fast -p rivet-codec --features h26x-fallback
cargo test --no-fail-fast -p rivet-codec --features rav1e-fallback,rav1d-fallback,h26x-fallback
cargo test --no-fail-fast -p rivet-codec --features nvidia
cargo test --no-fail-fast -p rivet-codec --features amd

cargo test --no-fail-fast -p rivet-transcoder
cargo test --no-fail-fast -p rivet-transcoder --features h26x-fallback
cargo test --no-fail-fast -p rivet-transcoder --features rav1e-fallback,rav1d-fallback,h26x-fallback
cargo test --no-fail-fast -p rivet-transcoder --features nvidia
cargo test --no-fail-fast -p rivet-transcoder --features nvidia,rav1e-fallback,rav1d-fallback,h26x-fallback
cargo test --no-fail-fast -p rivet-transcoder --features server,batch
```

Judge each command by two things, never by the absence of a `FAILED` line:

1. the exit code is 0, and
2. it printed one `test result: ok.` line per target. A target that did not
   compile prints no `test result:` line at all — it looks like silence, not
   like a failure.

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
| `server,batch` | Compiles and runs `server_api` (`#![cfg(feature = "server")]`). |

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

## Not in the gate, and why

| Feature | Reason |
|---|---|
| `ffmpeg` | Needs FFmpeg >= 7 development libraries (found through pkg-config or vcpkg) and libclang. The Windows dev box has neither: `ffmpeg-sys-next` fails its build script ("Could not find ffmpeg with vcpkg", "The pkg-config command could not be found"). Run `cargo test -p rivet-codec --features ffmpeg` where they exist — it is the only build that exercises `prores_dispatch`'s ffmpeg half. |
| `qsv` | No Intel GPU on the dev box; builds everywhere. |
| `dpir`, `dpir-cuda`, `dpir-cudnn` | A 130 MB model download; CUDA toolkit at build time for the GPU variants. |
| `rav1e-asm`, `rav1d-asm` | Need NASM on the build host. |
| `ipc` | Unix-only at run time. |
| `rivet-h26x` | The codec submodule has its own gate (conformance suites and encode sweeps, `crates/h26x/tools`), run when the submodule moves. |
