#!/usr/bin/env bash
# The Matroska rotation fixture for `demux::mkv::ebml`'s rotation tests: a file whose
# Video > Projection > ProjectionPoseRoll an independent writer put there, so the element ids the
# scanner looks for are checked against a file it did not build itself.
#
#   MKVMERGE=/path/to/mkvmerge bash make_fixtures.sh
#
# roll_minus90.mkv  ../colour/h264_601.mp4's video remuxed by MKVToolNix's mkvmerge with
#                   --projection-pose-roll -90: a counter-clockwise roll of -90 degrees (RFC 9559,
#                   ProjectionPoseRoll), i.e. 90 degrees clockwise.
set -euo pipefail
cd "$(dirname "$0")"
"${MKVMERGE:-mkvmerge}" -q --no-date -o roll_minus90.mkv --projection-pose-roll 0:-90 ../colour/h264_601.mp4
