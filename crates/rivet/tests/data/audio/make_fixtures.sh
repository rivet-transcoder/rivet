#!/bin/sh
# Audio fixtures for the job layer's surround / MP3 tests (`src/job/audio_tests.rs`).
# Each 5.1 fixture carries one tone per channel so a swapped or folded
# channel shows up as a frequency in the wrong place:
#   FL 400 Hz, FR 600, FC 800, LFE 50, SL 1000, SR 1200, each at 0.25.
# Run from this directory with ffmpeg on PATH (or FFMPEG=...).
set -eu
FF="${FFMPEG:-ffmpeg}"
TONES="aevalsrc=0.25*sin(2*PI*400*t)|0.25*sin(2*PI*600*t)|0.25*sin(2*PI*800*t)|0.25*sin(2*PI*50*t)|0.25*sin(2*PI*1000*t)|0.25*sin(2*PI*1200*t):s=48000:c=5.1(side):d=0.5"
# AC-3 5.1(side), 448 kbit/s: bare elementary stream and audio-only Matroska.
"$FF" -v error -y -f lavfi -i "$TONES" -c:a ac3 -b:a 448k tones_51.ac3
"$FF" -v error -y -i tones_51.ac3 -c copy tones_51_ac3.mka
# AAC-LC 5.1 in an audio-only MP4. Back surrounds, so the encoder writes
# channelConfiguration 6 (from 5.1(side) it writes a PCE).
"$FF" -v error -y -f lavfi -i "$(echo "$TONES" | sed 's/c=5.1(side)/c=5.1/')" -c:a aac -b:a 192k     -movflags +faststart tones_51_aac.m4a
