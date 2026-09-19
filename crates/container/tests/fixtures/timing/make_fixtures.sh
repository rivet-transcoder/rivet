#!/usr/bin/env bash
# Regenerate the TS program-clock fixtures: 64x64 H.264 at 25 fps with an audio track, each
# placing the video and the audio differently on the program's 90 kHz clock.
#
#   FFMPEG=/path/to/ffmpeg FFPROBE=/path/to/ffprobe bash make_fixtures.sh
#
#   video_first.ts      AAC 60 ms after the video (-itsoffset 0.06 on the audio input)
#   audio_first.ts      the video 100 ms after the AAC (-itsoffset 0.1 on the video input)
#   ac3_video_first.ts  as video_first.ts with AC-3 (1536-sample frames) and 80 ms
#   midgop.ts           GOPs of four with B pictures, cut inside one with -copyinkf: opens on three
#                       access units before its IDR (a P and two B, in reordered time), AAC beside them
#   wrap.ts             video_first.ts's shape moved up to 0.2 s before the 33-bit PTS wrap
#                       (2^33 ticks = 95443.717689 s), so both streams cross it
#
# Frame counting (a transport stream has no frame count; ts::pictures counts one):
#   paff_fields.ts      64x64 H.264 PAFF from rivet's own encoder (the crates/h26x example h26xenc,
#                       --field-coding field; H26XENC=path): 8 frames at 25 fps as 16 field pictures, one
#                       PES per field 1800 ticks apart — the carriage ffmpeg writes. ffmpeg's own raw-PAFF
#                       timestamps drift, so mux_fields_ts.py muxes it.
#   paff_pairs.ts       the same fields, both of a frame in one PES
#   rasl_cut.ts         64x64 HEVC open GOP (x265, CRA + RASL), cut at its CRA (cut_at_cra.py: ffmpeg's
#                       -ss -c copy wrote no video from it): it opens on the CRA with three RASL pictures
#                       after it that a decoder starting there does not output
#
# Then prints, per file, what ffprobe reads: each stream's first packet PTS (ticks) and flags.
set -euo pipefail
FF="${FFMPEG:-ffmpeg}"
FP="${FFPROBE:-ffprobe}"
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

V=(-f lavfi -i testsrc2=size=64x64:rate=25:duration=0.4)
A=(-f lavfi -i sine=frequency=440:sample_rate=48000:duration=0.4)
VENC=(-c:v libx264 -preset veryfast -bf 2 -g 4 -keyint_min 4 -sc_threshold 0 -threads 1)
AAC=(-c:a aac -ac 1 -b:a 32k)

"$FF" -v error -y "${V[@]}" -itsoffset 0.06 "${A[@]}" "${VENC[@]}" "${AAC[@]}" video_first.ts
"$FF" -v error -y -itsoffset 0.1 "${V[@]}" "${A[@]}" "${VENC[@]}" "${AAC[@]}" audio_first.ts
"$FF" -v error -y "${V[@]}" -itsoffset 0.08 "${A[@]}" "${VENC[@]}" -c:a ac3 -ac 1 -b:a 64k ac3_video_first.ts
"$FF" -v error -y "${V[@]/duration=0.4/duration=0.64}" "${A[@]/duration=0.4/duration=0.64}" "${VENC[@]}" "${AAC[@]}" midgop_src.ts
"$FF" -v error -y -i midgop_src.ts -ss 0.28 -c copy -copyinkf midgop.ts
rm -f midgop_src.ts
"$FF" -v error -y "${V[@]}" -itsoffset 0.06 "${A[@]}" "${VENC[@]}" "${AAC[@]}" -output_ts_offset 95442.1 wrap.ts

"$FF" -v error -y -f lavfi -i testsrc2=size=64x64:rate=25:duration=0.32 -f rawvideo -pix_fmt yuv420p paff.yuv
"${H26XENC:-h26xenc}" --input paff.yuv --size 64x64 --format 420 --output paff.264 --interlace tff   --field-coding field --fps 25 --gop 8 --qp 26 > /dev/null
"$FF" -v error -y "${A[@]/duration=0.4/duration=0.32}" "${AAC[@]}" -f adts paff.aac
python mux_fields_ts.py paff.264 paff.aac paff_fields.ts 1800
python mux_fields_ts.py paff.264 paff.aac paff_pairs.ts 1800 126000 --pairs
rm -f paff.yuv paff.264 paff.aac
"$FF" -v error -y "${V[@]/duration=0.4/duration=0.8}" "${A[@]/duration=0.4/duration=0.8}" -c:v libx265   -x265-params log-level=error:keyint=11:min-keyint=11:scenecut=0:open-gop=1:bframes=3 -threads 1 "${AAC[@]}" rasl_src.ts
python cut_at_cra.py rasl_src.ts rasl_cut.ts
rm -f rasl_src.ts

for f in video_first.ts audio_first.ts ac3_video_first.ts midgop.ts wrap.ts paff_fields.ts paff_pairs.ts rasl_cut.ts; do
  echo "--- $f ($(stat -c %s "$f") bytes)"
  for s in v a; do
    echo "  $s first packets (pts ticks, flags): $("$FP" -v error -select_streams $s:0 -show_entries packet=pts,flags -of csv=p=0 "$f" | head -4 | tr '\n' ' ')"
  done
  echo "  video: $("$FP" -v error -count_frames -count_packets -select_streams v:0 -show_entries stream=nb_read_frames,nb_read_packets,field_order -of default=nw=1 "$f" | tr '
' ' ')"
done
