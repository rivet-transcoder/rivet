#!/usr/bin/env bash
# Regenerate the colour-fallback fixtures: two-frame 64x64 H.264 / HEVC clips whose colour
# description lives ONLY in the bitstream (SPS VUI, and SEI 137/144 for the HDR10 pair).
#
#   FFMPEG=/path/to/ffmpeg bash make_fixtures.sh
#
# Names: <codec>_<case>.<container>
#   codec  h264 (libx264) | hevc (libx265)
#   case   601 — VUI smpte170m/smpte170m/smpte170m, limited range, 8-bit
#          pq  — VUI bt2020/smpte2084/bt2020nc, 10-bit, mastering display + CLL as SEI only
#   container mp4 (colr/mdcv/clli renamed `free`), mkv (Colour voided), ts (carries none),
#             avi (H.264 only: rivet's AVI reader knows no HEVC fourcc)
# `strip_colour.py dump` then prints what each container still says (it must say nothing).
set -euo pipefail
FF="${FFMPEG:-ffmpeg}"
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PYTHON:-python}"
cd "$HERE"

MD='G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(40000000,50)'
SRC=(-f lavfi -i testsrc2=size=64x64:rate=25:duration=0.08)
# The VUI triple goes in through the encoders' own parameters: ffmpeg's
# -color_primaries / -color_trc did not reach the VUI (it wrote 2 / 2 / 6).
VUI601='colorprim=smpte170m:transfer=smpte170m:colormatrix=smpte170m:range=limited'

args_for() {
  case "$1" in
    h264_601) echo "-pix_fmt yuv420p -c:v libx264 -bf 0 -x264-params $VUI601" ;;
    hevc_601) echo "-pix_fmt yuv420p -c:v libx265 -x265-params log-level=error:$VUI601" ;;
    h264_pq)  echo "-pix_fmt yuv420p10le -c:v libx264 -bf 0 -x264-params mastering-display=$MD:cll=1234,567:colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc" ;;
    hevc_pq)  echo "-pix_fmt yuv420p10le -c:v libx265 -x265-params log-level=error:hdr10=1:master-display=$MD:max-cll=1234,567:colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc" ;;
  esac
}

for name in h264_601 hevc_601 h264_pq hevc_pq; do
  for ext in mp4 mkv ts avi; do
    if [ "$ext" = avi ] && [ "${name%%_*}" = hevc ]; then continue; fi
    extra=()
    [ "$ext" = mkv ] && extra=(-write_crc32 0)
    # Encoded straight into each container (one thread, so the same bitstream every time):
    # a stream copy from a raw Annex-B file has no timestamps and Matroska refuses it.
    # shellcheck disable=SC2046
    "$FF" -v error -y "${SRC[@]}" $(args_for "$name") -threads 1 "${extra[@]}" "$name.$ext"
    case "$ext" in
      mp4|mkv)
        echo "-- as muxed:"; "$PY" strip_colour.py dump "$name.$ext"
        "$PY" strip_colour.py strip "$name.$ext"
        echo "-- after strip:"; "$PY" strip_colour.py dump "$name.$ext" ;;
    esac
  done
done
ls -la
