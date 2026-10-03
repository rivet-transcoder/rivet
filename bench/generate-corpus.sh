#!/bin/sh
# The bench corpus: four 1080p30 clips (grain, flat, motion, dark), each
# opening on a 3-second fade from black, made by rivet's own H.264 encoder
# from generators in crates/rivet/examples/bench_corpus.rs. No FFmpeg.
#
#   ./generate-corpus.sh [OUT_DIR] [--seconds 20] [--size 1920x1080]
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${1:-"$HERE/corpus"}
[ $# -gt 0 ] && shift
cargo run --release --quiet --manifest-path "$HERE/../Cargo.toml" -p rivet-transcoder \
  --example bench_corpus -- "$OUT" "$@"
