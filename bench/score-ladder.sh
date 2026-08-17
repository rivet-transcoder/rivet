#!/bin/bash
# Score each encoded rung against the source it came from, with VMAF and SSIM.
#
# Runs inside the `linuxserver/ffmpeg` image (which carries libvmaf); see the
# Dockerfile and README. Expects `/work/source.mp4` and `/work/rungs/` holding
# either `video/<label>/{init.mp4,seg-*.m4s}` (rivet HLS output) or
# `<label>.mp4` files (rivet single-file output).
#
# A CMAF rung is an init segment carrying the codec configuration and media
# segments carrying the frames. None of them plays alone, so they are
# concatenated back into one file first — that is what a player does too.
# *Every* segment, in order: scoring only `seg-00001.m4s` measures the first
# few seconds of the clip and calls it the ladder.
#
# The rung is upscaled to the source's dimensions before comparison because
# that is what a viewer sees: a 240p rung is not watched at 240p, it is watched
# stretched to the screen. Scoring at native resolution would flatter every
# small rung by hiding exactly the loss the ladder is trading away.
#
# # Which seconds get scored
#
# A window from the middle, not the whole clip, and not the opening.
#
# Black frames score close to perfect against black frames. A clip that fades
# in from black therefore carries a stretch of free VMAF, and averaging across
# the whole clip mixes it into the result — every rung looks better than it is,
# and the small rungs look better by more, because the fade is the one part
# they reproduce perfectly. That is the opposite of what this harness is for.
#
# So: find the black stretches, pick a window in the middle that misses them,
# and score there. `blackdetect` is what finds them; the window walks forward
# if the middle happens to land in one.
set -u
FF=${FF:-/usr/local/bin/ffmpeg}
PROBE=${PROBE:-/usr/local/bin/ffprobe}
SRC=${SRC:-/work/source.mp4}
WINDOW=${WINDOW:-10}

SW=$($PROBE -v error -select_streams v:0 -show_entries stream=width  -of default=nw=1:nk=1 "$SRC" | head -1)
SH=$($PROBE -v error -select_streams v:0 -show_entries stream=height -of default=nw=1:nk=1 "$SRC" | head -1)
DUR=$($PROBE -v error -show_entries format=duration -of default=nw=1:nk=1 "$SRC" | head -1)
DUR=${DUR%.*}

# Black intervals, as "start end" pairs. `pic_th=0.98` means "98% of the
# picture is below the black threshold", which catches a fade's dark frames
# without flagging a merely dim scene.
mapfile -t BLACK < <($FF -hide_banner -nostats -i "$SRC" \
    -vf blackdetect=d=0.05:pic_th=0.98 -f null - 2>&1 \
    | grep -oE "black_start:[0-9.]+ black_end:[0-9.]+" \
    | sed -E 's/black_start:([0-9.]+) black_end:([0-9.]+)/\1 \2/')

overlaps_black() {
    local s=$1 e=$2 bs be
    for interval in "${BLACK[@]:-}"; do
        [ -z "$interval" ] && continue
        bs=${interval% *}
        be=${interval#* }
        # Any intersection at all disqualifies the window: a sample that is
        # part fade is still part fade.
        awk -v s="$s" -v e="$e" -v bs="$bs" -v be="$be" \
            'BEGIN { exit !(s < be && bs < e) }' && return 0
    done
    return 1
}

# The whole clip when it is shorter than a window — there is nothing to choose
# between, and trimming would leave nothing to score.
if [ "${DUR:-0}" -le "$WINDOW" ]; then
    START=0
    WINDOW=$DUR
    echo "source ${SW}x${SH}, ${DUR}s — shorter than a window, scoring all of it"
else
    START=$(( (DUR - WINDOW) / 2 ))
    tries=0
    while overlaps_black "$START" "$((START + WINDOW))" && [ $tries -lt 4 ]; do
        echo "  window at ${START}s is black; moving on"
        START=$((START + WINDOW))
        tries=$((tries + 1))
        # Past the end — fall back to the middle and say so, rather than
        # scoring nothing.
        if [ $((START + WINDOW)) -gt "$DUR" ]; then
            START=$(( (DUR - WINDOW) / 2 ))
            echo "  every window tried was black; scoring the middle anyway"
            break
        fi
    done
    echo "source ${SW}x${SH}, ${DUR}s — scoring ${WINDOW}s from ${START}s"
fi

END=$((START + WINDOW))
# `trim` in the graph rather than `-ss` on the inputs: input seeking lands on a
# keyframe, and two files whose keyframes differ would be compared a frame or
# two out of step — which reads as a quality loss that is not there.
TRIM="trim=start=${START}:end=${END},setpts=PTS-STARTPTS"

printf '%-10s %10s %10s %8s\n' rung bytes vmaf ssim

# A rung is either a CMAF directory (`<label>/init.mp4` + `seg-*.m4s`, as
# `rivet transcode --mode hls` writes under `video/`) or a single MP4
# (`<label>.mp4`, as `--mode single` writes). Both are scored the same way.
RUNGS=${RUNGS:-/work/rungs}
[ -d "$RUNGS/video" ] && RUNGS="$RUNGS/video"

for entry in "$RUNGS"/*; do
    if [ -d "$entry" ]; then
        name=$(basename "$entry")
        [ -f "$entry/init.mp4" ] || continue
        # Every segment, in order, so the rung is the whole clip and the
        # window above lands on the same content in both files.
        segs=("$entry"/seg-*.m4s)
        cat "$entry/init.mp4" "${segs[@]}" > "/tmp/$name.mp4" 2>/dev/null
        bytes=$(cat "${segs[@]}" | wc -c)
    else
        case "$entry" in *.mp4) ;; *) continue ;; esac
        name=$(basename "$entry" .mp4)
        cp "$entry" "/tmp/$name.mp4"
        bytes=$(stat -c%s "$entry")
    fi

    out=$($FF -hide_banner -nostats -i "/tmp/$name.mp4" -i "$SRC" -lavfi \
        "[0:v]${TRIM},scale=${SW}:${SH}:flags=bicubic[d];[1:v]${TRIM}[r];[d][r]libvmaf=n_threads=4" \
        -f null - 2>&1)
    vmaf=$(echo "$out" | grep -oE "VMAF score: [0-9.]+" | tail -1 | grep -oE "[0-9.]+$")

    out2=$($FF -hide_banner -nostats -i "/tmp/$name.mp4" -i "$SRC" -lavfi \
        "[0:v]${TRIM},scale=${SW}:${SH}:flags=bicubic[d];[1:v]${TRIM}[r];[d][r]ssim" \
        -f null - 2>&1)
    ssim=$(echo "$out2" | grep -oE "All:[0-9.]+" | tail -1 | cut -d: -f2)

    printf '%-10s %10s %10s %8s\n' "$name" "$bytes" "${vmaf:-n/a}" "${ssim:-n/a}"
done
