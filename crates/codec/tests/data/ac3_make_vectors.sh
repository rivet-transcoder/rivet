#!/bin/bash
# Generate AC-3 / E-AC-3 cross-check vectors with the real ffmpeg binary:
# for each case an elementary stream (.ac3/.eac3) plus libavcodec's f32le
# decode of it (with and without dynrng), and — with STRIP set to the built
# `ac3_strip_dither` example — a `<name>_nodith` copy of the stream whose
# dither flags are cleared, so both decoders are deterministic and the
# comparison is to float rounding. Everything lands in $OUT (default: the
# scratchpad's ac3_vectors dir).
#
#   cargo build -p rivet-codec --example ac3_strip_dither
#   STRIP=<target>/debug/examples/ac3_strip_dither.exe bash ac3_make_vectors.sh
#
# ffmpeg's encoders never write a `dynrng` word, never switch blocks and
# never use E-AC-3's AHT or spectral extension; those paths are covered by
# real Dolby-encoded streams (see the sweep report) and, for block
# switching, by `ac3_make_blksw_vector.py`.
set -e
FF="C:/Users/elyci/scoop/apps/ffmpeg/current/bin/ffmpeg.exe"
OUT="${OUT:-C:/Users/elyci/AppData/Local/Temp/claude/C--Users-elyci-RustroverProjects-rivet/3f7a59d1-cd44-4b54-b19e-aa48826ca495/scratchpad/ac3_vectors}"
mkdir -p "$OUT"
DUR=3

refs() {
  local name=$1 ext=$2
  "$FF" -v error -y -i "$OUT/$name.$ext" -f f32le "$OUT/$name.drc1.f32"
  "$FF" -v error -y -drc_scale 0 -i "$OUT/$name.$ext" -f f32le "$OUT/$name.drc0.f32"
}

# name | lavfi source | channels | codec | bitrate | extra encoder opts
gen() {
  local name=$1 src=$2 ch=$3 codec=$4 br=$5; shift 5
  local ext=ac3; [ "$codec" = eac3 ] && ext=eac3
  "$FF" -v error -y -f lavfi -i "$src" -t $DUR -ac $ch -c:a $codec -b:a $br "$@" "$OUT/$name.$ext"
  refs "$name" "$ext"
  if [ -n "$STRIP" ]; then
    "$STRIP" "$OUT/$name.$ext" "$OUT/${name}_nodith.$ext" >/dev/null
    refs "${name}_nodith" "$ext"
  fi
  echo "made $name"
}

# Multi-tone + noise sources so every band carries energy (exercises coupling,
# rematrixing and dither); a transient burst to force block switching.
TONES="aevalsrc=0.4*sin(2*PI*440*t)+0.2*sin(2*PI*3000*t)+0.1*sin(2*PI*9000*t)+0.05*sin(2*PI*15000*t):s=48000:c=stereo"
TONES6="aevalsrc=0.4*sin(2*PI*440*t)|0.3*sin(2*PI*880*t)+0.1*sin(2*PI*7000*t)|0.3*sin(2*PI*1320*t)|0.4*sin(2*PI*60*t)|0.2*sin(2*PI*5000*t)+0.1*sin(2*PI*12000*t)|0.2*sin(2*PI*6500*t)+0.1*sin(2*PI*14000*t):s=48000:c=5.1"
NOISE="anoisesrc=color=pink:amplitude=0.3:sample_rate=48000:seed=7"
CLICKS="aevalsrc=0.3*sin(2*PI*500*t)+0.8*gt(mod(t\,0.25)\,0.24)*sin(2*PI*4000*t):s=48000:c=mono"

gen mono_64k     "$CLICKS"                       1 ac3 64k
gen stereo_192k  "$TONES"                        2 ac3 192k
gen stereo_noise "$NOISE"                        2 ac3 128k
gen surround_448k "$TONES6"                      6 ac3 448k
gen surround_192k "$TONES6"                      6 ac3 192k
gen surround_noise "anoisesrc=color=white:amplitude=0.25:sample_rate=48000:seed=3,aformat=channel_layouts=5.1" 6 ac3 384k
gen quad_256k    "anoisesrc=color=brown:amplitude=0.3:sample_rate=48000:seed=11,aformat=channel_layouts=quad" 4 ac3 256k
gen stereo_44k   "aevalsrc=0.4*sin(2*PI*440*t)+0.2*sin(2*PI*5000*t):s=44100:c=stereo" 2 ac3 160k
gen stereo_32k   "aevalsrc=0.4*sin(2*PI*440*t)+0.2*sin(2*PI*5000*t):s=32000:c=stereo" 2 ac3 96k
gen eac3_stereo  "$TONES"                        2 eac3 128k
gen eac3_surround "$TONES6"                      6 eac3 256k
gen eac3_surround_noise "anoisesrc=color=pink:amplitude=0.3:sample_rate=48000:seed=5,aformat=channel_layouts=5.1" 6 eac3 192k
gen eac3_mono_clicks "$CLICKS"                   1 eac3 48k

# Block-switched variants of two vectors (ffmpeg's encoder never sets blksw).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for v in stereo_192k surround_448k; do
  python "$HERE/ac3_make_blksw_vector.py" "$OUT/$v.ac3" "$OUT/blksw_$v.ac3" >/dev/null
  refs "blksw_$v" ac3
  if [ -n "$STRIP" ]; then
    "$STRIP" "$OUT/blksw_$v.ac3" "$OUT/blksw_${v}_nodith.ac3" >/dev/null
    refs "blksw_${v}_nodith" ac3
  fi
  echo "made blksw_$v"
done
ls "$OUT" | grep -c '\.\(ac3\|eac3\)$' | sed 's/^/streams: /'
