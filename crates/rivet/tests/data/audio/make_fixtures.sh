#!/bin/sh
# Audio fixtures for the job layer's surround / MP3 tests (`src/job/audio_tests.rs`)
# and the container's Matroska / MP4 audio tests, made with no FFmpeg. Each
# 5.1 fixture carries one tone per channel so a swapped or folded channel
# shows up as a frequency in the wrong place:
#   FL 400 Hz, FR 600, FC 800, LFE 50, SL (or BL) 1000, SR (or BR) 1200, each at 0.25.
#
# - tones_51.ac3: 48 kHz, 0.5 s (`tones51` in crates/ac3/tools/ac3_signals.py),
#   encoded by aften (an independent AC-3 encoder) as 3/2 + LFE at 448 kbit/s.
# - tones_51_ac3.mka: the same stream in an audio-only Matroska file, by
#   mkvmerge (MKVToolNix).
# - tones_51_aac.m4a: the same tones as 5.1 with back surrounds
#   (channelConfiguration 6), AAC-LC at 192 kbit/s by rivet's own encoder, in
#   rivet's audio-only MP4 (`synth::tones_51_aac_m4a`).
#
#   sudo apt-get install aften mkvtoolnix; sh make_fixtures.sh
# Run from this directory.
set -eu
PY="${PYTHON:-python3}"
"$PY" ../../../../ac3/tools/ac3_signals.py tones51 tones_51.wav 0.5
aften -v 0 -b 448 -chconfig 3/2+LFE tones_51.wav tones_51.ac3
rm tones_51.wav
mkvmerge -q -o tones_51_ac3.mka tones_51.ac3
cargo run -q --release -p rivet-transcoder --example synth_clip -- tones_51_aac.m4a --tones51-aac
