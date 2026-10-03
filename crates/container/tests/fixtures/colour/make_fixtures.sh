#!/usr/bin/env bash
# Regenerate the colour-fallback fixtures: two-frame 64x64 clips whose colour description lives
# ONLY in the bitstream (SPS VUI and SEI 137/144 for H.264 / HEVC, the sequence header and the
# HDR metadata OBUs for AV1, a keyframe's color_space for VP9, the sequence_display_extension for
# MPEG-2), muxed by third-party muxers so the demuxers read files they did not write.
#
#   GST=/path/to/gstreamer/bin H26XENC=/path/to/h26xenc bash make_fixtures.sh
#
# No FFmpeg and no libav*: every byte comes from GStreamer's own elements (videotestsrc, x264enc,
# x265enc, svtav1enc, vp9enc, h264parse / h265parse / av1parse, mp4mux, matroskamux, webmmux,
# mpegtsmux, avimux — never a gst-libav `av*` element; delete gstlibav from the plugin directory
# to be sure), rivet's own h26x encoder (crates/h26x, example h26xenc; the 10-bit H.264, which
# GStreamer's x264enc build does not offer) and the Python beside this script (mpeg2_es.py writes
# the MPEG-2 elementary streams, cut_ts.py cuts the mid-GOP transport streams, strip_colour.py
# removes the containers' copy of the colour).
#
# Names: <codec>_<case>.<container>
#   codec  h264 (x264; the 10-bit PQ one rivet's h26xenc) | hevc (x265)
#   case   601 — VUI smpte170m/smpte170m/smpte170m, limited range, 8-bit
#          pq  — VUI bt2020/smpte2084/bt2020nc, 10-bit, mastering display + CLL as SEI only
#   container mp4 (colr/mdcv/clli renamed `free`), mkv (Colour voided), ts (carries none),
#             avi (H.264 only: rivet's AVI reader knows no HEVC fourcc)
# `strip_colour.py dump` then prints what each container still says (it must say nothing).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PYTHON:-python}"
if [ -n "${GST:-}" ]; then PATH="$GST:$PATH"; fi
H26XENC="${H26XENC:-h26xenc}"
cd "$HERE"

launch() { gst-launch-1.0 -q "$@"; }

MD='G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(40000000,50)'
# Two frames of the test pattern, 64x64 at 25 fps. The caps' colorimetry is what the muxers copy
# into colr / Colour (and strip_colour.py takes out again); the encoders get the VUI explicitly.
src() { # <frames> <format> <colorimetry>
  echo "videotestsrc num-buffers=$1 ! video/x-raw,format=$2,width=64,height=64,framerate=25/1,colorimetry=$3"
}
VUI601='colorprim=smpte170m:transfer=smpte170m:colormatrix=smpte170m'
X264_601="x264enc threads=1 bframes=0 option-string=$VUI601"
X265_601="x265enc option-string=log-level=error:pools=1:frame-threads=1:$VUI601:range=limited"
X265_PQ="x265enc option-string=log-level=error:pools=1:frame-threads=1:hdr10=1:master-display=$MD:max-cll=1234,567:colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc"

mux_for() {
  case "$1" in
    mp4) echo "mp4mux" ;;
    mkv) echo "matroskamux" ;;
    webm) echo "webmmux" ;;
    ts) echo "mpegtsmux" ;;
    avi) echo "avimux" ;;
  esac
}

# The 10-bit H.264 PQ stream: rivet's encoder writes the Annex-B, GStreamer muxes it. h264parse
# gives a raw Annex-B file decode times only, so it goes through a transport stream first and
# takes its presentation times from there (no B pictures: they are the same).
h264_pq_es() {
  launch $(src 2 I420_10LE bt2100-pq) ! filesink location=pq_src.yuv
  "$H26XENC" --codec h264 --input pq_src.yuv --output h264_pq.264 --size 64x64 --format 420 \
    --depth 10 --fps 25 --qp 26 --color 9:16:9 --mastering-display "$MD" --content-light 1234,567 > /dev/null
  launch pushfilesrc location=h264_pq.264 time-segment=true initial-timestamp=0 ! h264parse \
    ! video/x-h264,alignment=au ! mpegtsmux ! filesink location=h264_pq_es.ts
  rm -f pq_src.yuv h264_pq.264
}

encode() { # <name> <container>
  local name=$1 ext=$2 mux
  mux=$(mux_for "$ext")
  case "$name" in
    h264_601) launch $(src 2 I420 bt601) ! $X264_601 ! h264parse ! $mux ! filesink location="$name.$ext" ;;
    hevc_601) launch $(src 2 I420 bt601) ! $X265_601 ! h265parse ! $mux ! filesink location="$name.$ext" ;;
    hevc_pq)  launch $(src 2 I420_10LE bt2100-pq) ! $X265_PQ ! h265parse ! $mux ! filesink location="$name.$ext" ;;
    h264_pq)  launch filesrc location=h264_pq_es.ts ! tsdemux ! h264parse ! $mux ! filesink location="$name.$ext" ;;
  esac
}

if [ "${ONLY:-}" != midgop ] && [ "${ONLY:-}" != inband ]; then
  h264_pq_es
  for name in h264_601 hevc_601 h264_pq hevc_pq; do
    for ext in mp4 mkv ts avi; do
      if [ "$ext" = avi ] && [ "${name%%_*}" = hevc ]; then continue; fi
      encode "$name" "$ext"
      case "$ext" in
        mp4|mkv)
          echo "-- as muxed:"; "$PY" strip_colour.py dump "$name.$ext"
          "$PY" strip_colour.py strip "$name.$ext"
          echo "-- after strip:"; "$PY" strip_colour.py dump "$name.$ext" ;;
      esac
    done
  done
  rm -f h264_pq_es.ts
fi

# Mid-GOP transport streams (<codec>_<case>_midgop.ts): eight frames in GOPs of four, the first
# two video PES packets cut off (cut_ts.py), so the first access unit carries no SPS and the
# colour, the dimensions and the pixel format are only in the IDR / CRA two frames later. x265
# needs repeat-headers for its SEIs to come again there. `ONLY=midgop` makes just these two.
if [ "${ONLY:-}" != inband ]; then
  launch $(src 8 I420 bt601) ! x264enc threads=1 bframes=0 key-int-max=4 option-string="$VUI601:min-keyint=4:scenecut=0" \
    ! h264parse ! mpegtsmux ! filesink location=midgop_src.ts
  "$PY" cut_ts.py midgop_src.ts h264_601_midgop.ts 2
  launch $(src 8 I420_10LE bt2100-pq) ! x265enc option-string="log-level=error:pools=1:frame-threads=1:keyint=4:min-keyint=4:scenecut=0:bframes=0:repeat-headers=1:hdr10=1:master-display=$MD:max-cll=1234,567:colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc" \
    ! h265parse ! mpegtsmux ! filesink location=midgop_src.ts
  "$PY" cut_ts.py midgop_src.ts hevc_pq_midgop.ts 2
  rm -f midgop_src.ts
fi

# In-band colour of the codecs whose header rivet reads besides H.264 / HEVC (`ONLY=inband` makes
# just these): AV1 (SVT-AV1: sequence header color_config, and for the PQ one the HDR10 metadata
# OBUs, METADATA_TYPE_HDR_MDCV / _HDR_CLL, with the values of the HEVC pair), VP9 (libvpx: a
# keyframe's color_space, CS_SMPTE_170 = 3) and MPEG-2 (a sequence_display_extension). The colour
# goes in through the raw caps' colorimetry; the container's copy is stripped (WebM Colour voided,
# the MP4 colr renamed); a TS has none. `*_none`: the bitstream states nothing either (AV1
# color_description_present_flag 0, VP9 CS_UNKNOWN, no display extension), for the
# standard-definition default.
# GStreamer fills an unknown colorimetry with a default and its encoders write what the caps say,
# so the statements the caps cannot carry are made in the bitstream by the Python beside this
# script, from the codecs' specifications: av1_ivf.py (the AV1 temporal units into IVF, removing
# the colour description or adding the HDR10 metadata OBUs; GStreamer's ivfparse reads it back),
# vp9_colour_space.py (the keyframes' color_space, in place) and mpeg2_es.py (the MPEG-2
# elementary streams, written from the syntax alone).
SVT="svtav1enc preset=12 parameters-string=lp=1"
OBU="av1parse ! video/x-av1,stream-format=obu-stream,alignment=tu"
VP9="vp9enc deadline=1 cpu-used=8 threads=1"
launch $(src 2 I420 bt601) ! $SVT ! av1parse ! webmmux ! filesink location=av1_601.webm
launch $(src 2 I420 bt601) ! $SVT ! av1parse ! mp4mux ! filesink location=av1_601.mp4
launch $(src 2 I420 bt601) ! $SVT ! $OBU ! filesink location=av1_none.obu
"$PY" av1_ivf.py av1_none.obu av1_none.ivf --no-colour-description
launch $(src 2 I420_10LE bt2100-pq) ! $SVT ! $OBU ! filesink location=av1_pq.obu
"$PY" av1_ivf.py av1_pq.obu av1_pq.ivf --hdr10 34000,16000,13250,34500,7500,3000,15635,16450,40000000,50 1234,567
for n in av1_none av1_pq; do
  launch filesrc location=$n.ivf ! ivfparse ! av1parse ! webmmux ! filesink location=$n.webm
  rm -f $n.obu $n.ivf
done
launch $(src 2 I420 bt601) ! $VP9 ! webmmux ! filesink location=vp9_601.webm
"$PY" vp9_colour_space.py vp9_601.webm 3
launch $(src 2 I420 bt601) ! $VP9 ! webmmux ! filesink location=vp9_none.webm
"$PY" vp9_colour_space.py vp9_none.webm 0
"$PY" mpeg2_es.py mpeg2_601.m2v --colour 6,6,6
"$PY" mpeg2_es.py mpeg2_none.m2v
for n in mpeg2_601 mpeg2_none; do
  launch pushfilesrc location=$n.m2v time-segment=true initial-timestamp=0 ! mpegvideoparse ! mpegtsmux ! filesink location=$n.ts
  rm -f $n.m2v
done
for f in av1_601.webm av1_601.mp4 av1_none.webm av1_pq.webm vp9_601.webm vp9_none.webm; do
  echo "-- $f as muxed:"; "$PY" strip_colour.py dump "$f"
  "$PY" strip_colour.py strip "$f"
  echo "-- after strip:"; "$PY" strip_colour.py dump "$f"
done
