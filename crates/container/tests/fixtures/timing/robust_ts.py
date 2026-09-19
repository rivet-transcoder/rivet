#!/usr/bin/env python3
"""The TS robustness fixtures, from robust_src.ts (64x64 H.264 at 25 fps, IDR every 10 frames, on PID 0x100
with the PCR; AAC one frame per PES on 0x101):

  robust_hole.ts      the audio PES packets with PTS in [0.6 s, 0.9 s) past the first video PTS dropped:
                      a hole in the audio alone
  robust_dropout.ts   the video from its IDR at 0.8 s to the one at 1.2 s dropped (a whole GOP), and
                      the audio PES in the same span: both streams lost together
  robust_splice.ts    robust_src.ts twice, the second copy's timestamps moved on by its length and
                      2 s: a forward step in the clock that no jump test flags, with the
                      discontinuity_indicator set on the first packet of each PID after the join that
                      has an adaptation field (the PCR packet of 0x100)

(robust_src.ts twice, byte for byte (the clock jumping back at the join) is built by the test.)

    python robust_ts.py robust_src.ts
"""
import sys

TS = 188
VIDEO, AUDIO = 0x100, 0x101


def pid(p):
    return ((p[1] & 0x1F) << 8) | p[2]


def payload_off(p):
    return 4 + (1 + p[4] if p[3] & 0x20 else 0)


def pes_pts(p):
    if not p[1] & 0x40:
        return None
    b = p[payload_off(p):]
    if len(b) < 14 or b[0:3] != b"\x00\x00\x01" or not b[7] & 0x80:
        return None
    return ((b[9] >> 1) & 7) << 30 | ((b[10] << 7 | b[11] >> 1) & 0x7FFF) << 15 | ((b[12] << 7 | b[13] >> 1) & 0x7FFF)


def set_ts(b, at, value):
    b[at] = (b[at] & 0xF0) | ((value >> 29) & 0x0E) | 1
    b[at + 1] = (value >> 22) & 0xFF
    b[at + 2] = ((value >> 14) & 0xFE) | 1
    b[at + 3] = (value >> 7) & 0xFF
    b[at + 4] = ((value << 1) & 0xFE) | 1


def shift(p, ticks):
    """A packet's PCR and its PES's PTS / DTS moved on by `ticks` (90 kHz)."""
    q = bytearray(p)
    if q[3] & 0x20 and q[4] > 0 and q[5] & 0x10:
        base = q[6] << 25 | q[7] << 17 | q[8] << 9 | q[9] << 1 | q[10] >> 7
        base = (base + ticks) % (1 << 33)
        q[6], q[7], q[8], q[9] = base >> 25 & 0xFF, base >> 17 & 0xFF, base >> 9 & 0xFF, base >> 1 & 0xFF
        q[10] = (q[10] & 0x7F) | (base & 1) << 7
    if q[1] & 0x40 and pid(q) in (VIDEO, AUDIO):
        o = payload_off(q)
        if q[o:o + 3] == b"\x00\x00\x01":
            flags = q[o + 7] >> 6
            if flags & 2:
                set_ts(q, o + 9, (pes_pts(q) + ticks) % (1 << 33))
            if flags == 3:
                b = q[o + 14:o + 19]
                d = ((b[0] >> 1) & 7) << 30 | ((b[1] << 7 | b[2] >> 1) & 0x7FFF) << 15 | ((b[3] << 7 | b[4] >> 1) & 0x7FFF)
                set_ts(q, o + 14, (d + ticks) % (1 << 33))
    return q


def main():
    data = open(sys.argv[1], "rb").read()
    pkts = [bytearray(data[i:i + TS]) for i in range(0, len(data), TS)]
    video_ptses = [pes_pts(p) for p in pkts if pid(p) == VIDEO and pes_pts(p) is not None]
    first = min(video_ptses)
    at = lambda s: first + round(s * 90000)

    # robust_hole.ts
    out, dropping = [], False
    for p in pkts:
        if pid(p) == AUDIO:
            pts = pes_pts(p)
            if pts is not None:
                dropping = at(0.6) <= pts < at(0.9)
            if dropping:
                continue
        out.append(p)
    open("robust_hole.ts", "wb").write(b"".join(out))

    # robust_dropout.ts: the video in stream order from the PES with PTS at(0.8) (an IDR) to the one with
    # PTS at(1.2) (the next); the audio by PTS over the same span.
    out, vdrop, adrop = [], False, False
    for p in pkts:
        q, pts = pid(p), pes_pts(p)
        if q == VIDEO and pts is not None:
            if pts == at(0.8):
                vdrop = True
            elif pts == at(1.2):
                vdrop = False
        if q == AUDIO and pts is not None:
            adrop = at(0.8) <= pts < at(1.2)
        if (q == VIDEO and vdrop) or (q == AUDIO and adrop):
            continue
        out.append(p)
    open("robust_dropout.ts", "wb").write(b"".join(out))

    # robust_splice.ts
    length = max(video_ptses) - first + 3600
    second, seen = [], set()
    for p in pkts:
        q = shift(p, length + 2 * 90000)
        if pid(q) in (VIDEO, AUDIO) and pid(q) not in seen and q[3] & 0x20 and q[4] > 0:
            q[5] |= 0x80
            seen.add(pid(q))
        second.append(q)
    open("robust_splice.ts", "wb").write(b"".join(pkts + second))
    print(f"first video PTS {first}, length {length} ticks, discontinuity_indicator on {sorted(hex(x) for x in seen)}")


main()
