#!/usr/bin/env bash
# mp3_tone_48k_mono.mp3 for tests/mp3_packets.rs, with no FFmpeg: 0.5 s of a 1 kHz sine at 48 kHz
# mono, encoded by LAME through GStreamer's lamemp3enc at a constant 64 kbit/s (192 bytes a
# frame). lamemp3enc writes bare frames: no ID3 tag and no Xing / Info frame.
#
#   GST=/path/to/gstreamer/bin bash make_mp3_tone.sh
set -euo pipefail
cd "$(dirname "$0")"
if [ -n "${GST:-}" ]; then PATH="$GST:$PATH"; fi
gst-launch-1.0 -q audiotestsrc wave=sine freq=1000 samplesperbuffer=1000 num-buffers=24 \
  ! audio/x-raw,format=S16LE,rate=48000,channels=1 \
  ! lamemp3enc target=bitrate bitrate=64 cbr=true ! filesink location=mp3_tone_48k_mono.mp3
