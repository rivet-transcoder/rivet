#!/bin/sh
# Score each encoded rung against the source it came from, with VMAF and SSIM.
#
#   ./score-ladder.sh SOURCE.mp4 RUNGS_DIR [--window SECONDS]
#
# RUNGS_DIR is rivet's output as it is: `video/<label>/{init.mp4,seg-*.m4s}`
# (HLS) or `<label>.mp4` files (single-file). The scorer is
# crates/rivet/examples/bench_score.rs: rivet decodes both, upscales each rung
# to the source's size (bicubic), picks a window from the middle that misses
# black stretches, and computes SSIM itself; VMAF is Netflix's `vmaf` tool
# (libvmaf's release binary, https://github.com/Netflix/vmaf/releases), run
# as a black box on Y4M — VMAF names it when it is not on PATH. No FFmpeg.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
exec cargo run --release --quiet --manifest-path "$HERE/../Cargo.toml" -p rivet-transcoder \
  --example bench_score -- "$@"
