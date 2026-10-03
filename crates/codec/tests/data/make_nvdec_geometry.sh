#!/usr/bin/env bash
# The two NVDEC geometry clips of tests/nvdec_smoke.rs, with no FFmpeg: three frames of a black
# picture with a white stripe on its last 8 display rows, written as raw I420 by the Python below,
# encoded and muxed by GStreamer (rawvideoparse, x264enc / x265enc, h264parse / h265parse,
# mp4mux; no gst-libav element).
#
#   GST=/path/to/gstreamer/bin bash make_nvdec_geometry.sh
#
# nvdec_geometry_h264_640x360.mp4  H.264 High, coded 368 rows (frame cropping), no B pictures
# nvdec_geometry_hevc_640x354.mp4  HEVC Main in an hvc1 sample entry, a conformance window, no B
set -euo pipefail
cd "$(dirname "$0")"
if [ -n "${GST:-}" ]; then PATH="$GST:$PATH"; fi
PY="${PYTHON:-python}"
stripe() { # <width> <height> <out.yuv>: limited-range black, rows h-8..h-1 white, neutral chroma
  "$PY" -c "
import sys
w, h = int(sys.argv[1]), int(sys.argv[2])
y = bytes([16]) * (w * (h - 8)) + bytes([235]) * (w * 8)
c = bytes([128]) * ((w // 2) * ((h + 1) // 2) * 2)
open(sys.argv[3], 'wb').write((y + c) * 3)
" "$@"
}
raw() { echo "filesrc location=$3 ! rawvideoparse format=i420 width=$1 height=$2 framerate=30/1"; }
stripe 640 360 g264.yuv
gst-launch-1.0 -q $(raw 640 360 g264.yuv) ! x264enc threads=1 bframes=0 key-int-max=3 quantizer=18 pass=quant \
  ! video/x-h264,profile=high ! h264parse ! mp4mux ! filesink location=nvdec_geometry_h264_640x360.mp4
stripe 640 354 g265.yuv
gst-launch-1.0 -q $(raw 640 354 g265.yuv) ! x265enc option-string=pools=1:frame-threads=1:bframes=0:crf=18 \
  ! h265parse ! video/x-h265,stream-format=hvc1 ! mp4mux ! filesink location=nvdec_geometry_hevc_640x354.mp4
rm -f g264.yuv g265.yuv
