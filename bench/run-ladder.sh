#!/usr/bin/env bash
#
# One command from a source clip to a scored ladder: run `rivet transcode` on
# it, then score every rung against the source with VMAF and SSIM.
#
#   ./run-ladder.sh corpus/flat.mp4 baseline
#   ./run-ladder.sh corpus/flat.mp4 recommended --encode-policy recommended
#   ./run-ladder.sh corpus/flat.mp4 vmaf93 --target vmaf=93
#
# The label names the result file; everything after it is passed to
# `rivet transcode` verbatim, so any knob — `--encode-policy`, `--target`,
# `--gop`, `--crf`, `--decode`, `--encode`, `--codec` — is a data point.
#
# The rule this exists to enforce: a ladder change is not a result until it has
# been scored against a source. Reasoning about encoder settings from published
# BD-rate figures has been wrong, in production, more than once — a policy
# tuned that way spent 23% of the largest rung buying 0.62 VMAF, and a per-title
# sweep certified a shift against a floor it then missed. Score it.
set -euo pipefail

CLIP="${1:?usage: run-ladder.sh <clip.mp4> <label> [rivet transcode flags…]}"
LABEL="${2:?usage: run-ladder.sh <clip.mp4> <label> [rivet transcode flags…]}"
shift 2

HERE=$(cd "$(dirname "$0")" && pwd)
RIVET="${RIVET:-rivet}"
MODE="${MODE:-hls}"
SEGMENT_SECONDS="${SEGMENT_SECONDS:-4}"
WORK="$HERE/work-$LABEL"

rm -rf "$WORK"
mkdir -p "$WORK/rungs"
cp "$CLIP" "$WORK/source.mp4"
cp "$HERE/score-ladder.sh" "$HERE/Dockerfile" "$WORK/"

echo "== encoding: $RIVET transcode $CLIP --mode $MODE --ladder $*"
if [ "$MODE" = hls ]; then
    "$RIVET" transcode "$CLIP" --mode hls --ladder --segment-seconds "$SEGMENT_SECONDS" \
        -o "$WORK/rungs" "$@"
else
    "$RIVET" transcode "$CLIP" --mode single --ladder -o "$WORK/rungs" "$@"
fi

echo "== scoring (VMAF + SSIM, mid-clip window, rungs upscaled to source)"
docker build -q -t "rivet-vmaf-$LABEL" "$WORK" >/dev/null
docker run --rm "rivet-vmaf-$LABEL" | tee "$HERE/result-$LABEL.txt"
echo "== written to $HERE/result-$LABEL.txt"
