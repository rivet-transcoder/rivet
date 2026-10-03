#!/usr/bin/env python3
"""Set the color_space of every VP9 profile-0 keyframe in a file, in place (VP9 bitstream
specification 6.2 / 7.2: the uncompressed header opens frame_marker, profile, show_existing_frame
= 0, frame_type = KEY_FRAME, show_frame, error_resilient_mode, then frame_sync_code 0x49 0x83 0x42
and, for profile 0, color_space in the next three bits). Same length, so no container offset moves.

    python vp9_colour_space.py <file.webm> <color_space>

color_space: 0 CS_UNKNOWN, 1 CS_BT_601, 2 CS_BT_709, 3 CS_SMPTE_170, 4 CS_SMPTE_240,
5 CS_BT_2020, 6 CS_RESERVED, 7 CS_RGB (an RGB one changes the syntax after it; not allowed here).

GStreamer's vp9enc takes the colour space from the raw caps, where BT.601 is CS_BT_601 and an
unknown colorimetry is filled with a default, so CS_SMPTE_170 and CS_UNKNOWN are set here.
"""
import sys

path, cs = sys.argv[1], int(sys.argv[2])
assert 0 <= cs <= 5, "color_space 0..5"
buf = bytearray(open(path, "rb").read())
# frame_marker 2, profile 0 (low 0, high 0), show_existing_frame 0, frame_type 0 (key),
# show_frame 1, error_resilient_mode 0 -> 0b1000_0010; then the sync code.
key = bytes([0x82, 0x49, 0x83, 0x42])
n, at = 0, buf.find(key)
while at >= 0:
    old = buf[at + 4] >> 5
    buf[at + 4] = (buf[at + 4] & 0x1F) | (cs << 5)
    print(f"{path}: keyframe @{at}: color_space {old} -> {cs}")
    n += 1
    at = buf.find(key, at + 4)
assert n, "no VP9 profile-0 keyframe found"
open(path, "wb").write(buf)
