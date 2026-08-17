# The scoring harness: ffmpeg with libvmaf, plus score-ladder.sh. A build
# context rather than a bind mount, because Docker Desktop on some hosts
# shares no drives and `-v` of a real directory silently mounts an empty one.
#
#   docker build -t rivet-vmaf . && docker run --rm rivet-vmaf
#
# expects, in this directory: source.mp4 and rungs/ (see README.md).
FROM linuxserver/ffmpeg:latest
COPY . /work
ENTRYPOINT ["/bin/bash", "/work/score-ladder.sh"]
