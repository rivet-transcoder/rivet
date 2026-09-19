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
  case "${ONLY:-}" in midgop|inband) break ;; esac
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

# Mid-GOP transport streams (<codec>_<case>_midgop.ts): eight frames in GOPs of four, cut two
# frames in with `-copyinkf` (which keeps the non-keyframes ffmpeg would otherwise drop), so the
# first access unit carries no SPS and the colour, the dimensions and the pixel format are only in
# the IDR / CRA two frames later. x265 needs repeat-headers for its SEIs to come again there.
# `ONLY=midgop` makes just these two.
if [ "${ONLY:-}" != inband ]; then
"$FF" -v error -y "${SRC[@]/duration=0.08/duration=0.32}" -pix_fmt yuv420p -c:v libx264 -bf 0 -g 4 \
  -x264-params "$VUI601" -threads 1 midgop_src.ts
"$FF" -v error -y -i midgop_src.ts -ss 0.08 -c copy -copyinkf h264_601_midgop.ts
"$FF" -v error -y "${SRC[@]/duration=0.08/duration=0.32}" -pix_fmt yuv420p10le -c:v libx265 \
  -x265-params "log-level=error:keyint=4:min-keyint=4:scenecut=0:bframes=0:repeat-headers=1:hdr10=1:master-display=$MD:max-cll=1234,567:colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc" \
  -threads 1 midgop_src.ts
"$FF" -v error -y -i midgop_src.ts -ss 0.08 -c copy -copyinkf hevc_pq_midgop.ts
rm -f midgop_src.ts
fi

# In-band colour of the codecs whose header rivet reads besides H.264 / HEVC (`ONLY=inband` makes
# just these): AV1 (SVT-AV1: sequence header color_config, and for the PQ one the HDR10 metadata
# OBUs, METADATA_TYPE_HDR_MDCV / _HDR_CLL, with the values of the HEVC pair), VP9 (libvpx: a
# keyframe's color_space, CS_SMPTE_170 = 3) and MPEG-2 (a sequence_display_extension). The colour
# goes in through setparams — ffmpeg's -color_* output options did not reach these bitstreams —
# and the container's copy is stripped (WebM Colour voided, the MP4 colr renamed); a TS has none.
# WebM is muxed bit-exact (no random UIDs) and without CRC-32, so the strip keeps it valid.
# `*_none`: the bitstream states nothing either (AV1 color_description_present_flag 0, VP9
# CS_UNKNOWN, no display extension), for the standard-definition default.
P601="setparams=colorspace=smpte170m:color_primaries=smpte170m:color_trc=smpte170m:range=tv"
PPQ="setparams=colorspace=bt2020nc:color_primaries=bt2020:color_trc=smpte2084:range=tv"
AV1MD='mastering-display=G(0.265,0.690)B(0.150,0.060)R(0.680,0.320)WP(0.3127,0.3290)L(4000,0.005):content-light=1234,567'
AV1=(-c:v libsvtav1 -preset 12 -svtav1-params "lp=1")
VP9=(-c:v libvpx-vp9 -deadline realtime -cpu-used 8 -threads 1)
"$FF" -v error -y "${SRC[@]}" -vf "format=yuv420p,$P601" "${AV1[@]}" -write_crc32 0 -fflags +bitexact av1_601.webm
"$FF" -v error -y "${SRC[@]}" -vf "format=yuv420p,$P601" "${AV1[@]}" av1_601.mp4
"$FF" -v error -y "${SRC[@]}" -vf "format=yuv420p" "${AV1[@]}" -write_crc32 0 -fflags +bitexact av1_none.webm
"$FF" -v error -y "${SRC[@]}" -vf "format=yuv420p10le,$PPQ" -c:v libsvtav1 -preset 12 -svtav1-params "lp=1:$AV1MD" -write_crc32 0 -fflags +bitexact av1_pq.webm
"$FF" -v error -y "${SRC[@]}" -vf "format=yuv420p,$P601" "${VP9[@]}" -write_crc32 0 -fflags +bitexact vp9_601.webm
"$FF" -v error -y "${SRC[@]}" -vf "format=yuv420p" "${VP9[@]}" -write_crc32 0 -fflags +bitexact vp9_none.webm
"$FF" -v error -y "${SRC[@]}" -vf "format=yuv420p,$P601" -c:v mpeg2video -threads 1 mpeg2_601.ts
"$FF" -v error -y "${SRC[@]}" -vf "format=yuv420p" -c:v mpeg2video -threads 1 mpeg2_none.ts
for f in av1_601.webm av1_601.mp4 av1_none.webm av1_pq.webm vp9_601.webm vp9_none.webm; do
  echo "-- $f as muxed:"; "$PY" strip_colour.py dump "$f"
  "$PY" strip_colour.py strip "$f"
  echo "-- after strip:"; "$PY" strip_colour.py dump "$f"
done
ls -la
