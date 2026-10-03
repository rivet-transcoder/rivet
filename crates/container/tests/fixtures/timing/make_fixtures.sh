#!/usr/bin/env bash
# Regenerate the TS program-clock fixtures: 64x64 H.264 at 25 fps with an audio track, each
# placing the video and the audio differently on the program's 90 kHz clock.
#
#   GST=/path/to/gstreamer/bin H26XENC=/path/to/h26xenc bash make_fixtures.sh
#
# No FFmpeg and no libav*: GStreamer's own elements (videotestsrc, audiotestsrc and its
# timestamp-offset, x264enc, x265enc, voaacenc, ac3parse, mpegtsmux with the video on PID 0x100
# and the audio on 0x101 — never a gst-libav `av*` element), rivet's own h26x encoder for the PAFF
# stream, and the Python beside this script, which cuts, shifts and (for the field-coded stream)
# muxes. ts_times.py prints, per file, what an independent reader of the PES headers sees.
#
#   video_first.ts      AAC 60 ms after the video (the audio source's timestamp-offset)
#   audio_first.ts      the video 100 ms after the AAC
#   ac3_video_first.ts  as video_first.ts with AC-3 (1536-sample frames) 80 ms after the video
#   midgop.ts           GOPs of four with two B pictures, its first five video PES packets (an IDR
#                       GOP's I, P, B, B and the next IDR) cut off with the audio before the cut
#                       (../colour/cut_ts.py): opens on three access units before its IDR (a P and
#                       two B, in reordered time), AAC beside them
#   wrap.ts             video_first.ts moved along the clock (shift_ts.py) so its first video PTS
#                       is 2^33 - 19592 ticks: both streams cross the 33-bit PTS wrap
#                       (2^33 ticks = 95443.717689 s)
#
# Frame counting (a transport stream has no frame count; ts::pictures counts one):
#   paff_fields.ts      64x64 H.264 PAFF from rivet's own encoder (the crates/h26x example h26xenc,
#                       --field-coding field; H26XENC=path): 8 frames at 25 fps as 16 field pictures,
#                       one PES per field 1800 ticks apart, muxed by mux_fields_ts.py
#   paff_pairs.ts       the same fields, both of a frame in one PES
#   rasl_cut.ts         64x64 HEVC open GOP (x265, CRA + RASL), cut at its CRA (cut_at_cra.py): it
#                       opens on the CRA with three RASL pictures after it that a decoder starting
#                       there does not output
#
# Robustness (a hole, a dropout, a splice; robust_ts.py derives them):
#   robust_src.ts       64x64 H.264 at 25 fps for 1.6 s, IDR every 10 frames, 76 AAC frames one to a
#                       PES 60 ms after it
#   robust_hole.ts      its audio PES with PTS in [0.6 s, 0.9 s) dropped
#   robust_dropout.ts   its video GOP from the IDR at 0.8 s to the one at 1.2 s, and the audio beside
#                       it, dropped
#   robust_splice.ts    it twice, the second copy's clock moved on by its length and 2 s, flagged with
#                       discontinuity_indicator
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PYTHON:-python}"
if [ -n "${GST:-}" ]; then PATH="$GST:$PATH"; fi
cd "$HERE"

launch() { gst-launch-1.0 -q "$@"; }

ms() { echo $(($1 * 1000000)); }
# <frames> [offset ms]: the test pattern, 64x64 at 25 fps.
video() { echo "videotestsrc num-buffers=$1 timestamp-offset=$(ms "${2:-0}") ! video/x-raw,format=I420,width=64,height=64,framerate=25/1"; }
# <milliseconds> [offset ms]: a 440 Hz sine, 48 kHz mono, in 1024-sample buffers (one AAC frame
# each). The buffers' nanosecond times are truncated (1024 samples = 21333333.3 ns) and the muxer
# truncates again to 90 kHz, which put every third frame a tick early; half a tick (5555 ns) on
# the offset makes every PTS the exact 1920-tick multiple.
audio_frames() { echo "audiotestsrc wave=sine freq=440 samplesperbuffer=1024 num-buffers=$1 timestamp-offset=$(( $(ms "${2:-0}") + 5555 )) ! audio/x-raw,format=S16LE,rate=48000,channels=1"; }
audio() { audio_frames $(( ($1 * 48 + 1023) / 1024 )) "${2:-0}"; }
# <GOP length>: x264 with two B pictures between anchors, an IDR every GOP.
x264() { echo "x264enc threads=1 speed-preset=veryfast bframes=2 b-adapt=false key-int-max=$1 option-string=min-keyint=$1:scenecut=0"; }
X264=$(x264 4)
AAC="voaacenc bitrate=32000 ! aacparse"
# <out> <video pipeline> <audio pipeline>
mux() { launch mpegtsmux name=m ! filesink location="$1" $2 ! h264parse ! m.sink_256 $3 ! m.sink_257; }

mux video_first.ts "$(video 10) ! $X264" "$(audio 400 60) ! $AAC"
mux audio_first.ts "$(video 10 100) ! $X264" "$(audio 400) ! $AAC"
# AC-3: GStreamer's only AC-3 encoder is gst-libav's, so the elementary stream is the aften-made
# crates/rivet/tests/data/audio/tones_51.ac3 (5.1, 448 kbit/s, 0.5 s; its make_fixtures.sh), or
# $AC3. pushfilesrc gives ac3parse a time segment to stamp the frames on; shift_ts.py then moves
# the AC-3 PES 80 ms on (7200 ticks), the muxer having put both streams at the program's start.
AC3="${AC3:-../../../../rivet/tests/data/audio/tones_51.ac3}"
launch mpegtsmux name=m ! filesink location=ac3_src.ts $(video 10) ! $X264 ! h264parse ! m.sink_256 \
  pushfilesrc location="$AC3" time-segment=true initial-timestamp=0 ! ac3parse ! m.sink_257
"$PY" shift_ts.py ac3_src.ts ac3_video_first.ts --pid 0x101 7200
rm -f ac3_src.ts
mux midgop_src.ts "$(video 16) ! $X264" "$(audio 640) ! $AAC"
"$PY" ../colour/cut_ts.py midgop_src.ts midgop.ts 5 --audio-from-video-dts
rm -f midgop_src.ts
"$PY" shift_ts.py video_first.ts wrap.ts $(( (1 << 33) - 19592 ))

launch $(video 8) ! filesink location=paff.yuv
"${H26XENC:-h26xenc}" --input paff.yuv --size 64x64 --format 420 --output paff.264 --interlace tff \
  --field-coding field --fps 25 --gop 8 --qp 26 > /dev/null
launch $(audio 320) ! $AAC ! audio/mpeg,stream-format=adts ! filesink location=paff.aac
"$PY" mux_fields_ts.py paff.264 paff.aac paff_fields.ts 1800
"$PY" mux_fields_ts.py paff.264 paff.aac paff_pairs.ts 1800 126000 --pairs
rm -f paff.yuv paff.264 paff.aac

launch mpegtsmux name=m ! filesink location=rasl_src.ts \
  $(video 20) ! x265enc option-string=log-level=error:pools=1:frame-threads=1:keyint=11:min-keyint=11:scenecut=0:open-gop=1:bframes=3:b-adapt=0:repeat-headers=1 \
  ! h265parse ! m.sink_256 $(audio 800) ! $AAC ! m.sink_257
"$PY" cut_at_cra.py rasl_src.ts rasl_cut.ts
rm -f rasl_src.ts

# 76 AAC frames, one more than 1.6 s holds: the audio runs a frame past the video's end, so where
# the stream is joined to itself the second copy's first audio frame overlaps the first's last.
mux robust_src.ts "$(video 40) ! $(x264 10)" "$(audio_frames 76 60) ! $AAC"
"$PY" robust_ts.py robust_src.ts

for f in video_first.ts audio_first.ts ac3_video_first.ts midgop.ts wrap.ts paff_fields.ts paff_pairs.ts rasl_cut.ts \
  robust_src.ts robust_hole.ts robust_dropout.ts robust_splice.ts; do
  [ -f "$f" ] && "$PY" ts_times.py "$f"
done
