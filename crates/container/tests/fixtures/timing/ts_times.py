#!/usr/bin/env python3
"""What a transport stream's PES headers say, read straight from the packets (ISO/IEC 13818-1
2.4.3.6 / 2.4.3.7) with no demuxer in between: per PID, the number of PES packets, the first
PTSes and DTSes (90 kHz ticks) in stream order, and the smallest PTS. The timing tests' expected
values are these numbers.

    python ts_times.py <file.ts>
"""
import sys

TS = 188


def pid(p):
    return ((p[1] & 0x1F) << 8) | p[2]


def payload(p):
    return p[4 + (1 + p[4] if p[3] & 0x20 else 0):]


def ts_field(b):
    return ((b[0] >> 1) & 7) << 30 | ((b[1] << 7 | b[2] >> 1) & 0x7FFF) << 15 | ((b[3] << 7 | b[4] >> 1) & 0x7FFF)


def main():
    data = open(sys.argv[1], "rb").read()
    per = {}
    for i in range(0, len(data), TS):
        p = data[i:i + TS]
        if not p[1] & 0x40 or not p[3] & 0x10:
            continue
        b = payload(p)
        if b[:3] != b"\x00\x00\x01" or b[3] < 0xBD:
            continue
        flags = b[7] >> 6
        pts = ts_field(b[9:14]) if flags & 2 else None
        dts = ts_field(b[14:19]) if flags == 3 else None
        per.setdefault(pid(p), []).append((pts, dts))
    print(f"--- {sys.argv[1]} ({len(data)} bytes)")
    for k in sorted(per):
        v = per[k]
        ptses = [p for p, _ in v if p is not None]
        head = " ".join(f"{p}/{d}" if d is not None else f"{p}" for p, d in v[:4])
        print(f"  PID 0x{k:x}: {len(v)} PES, first PTS[/DTS] {head}, smallest PTS {min(ptses)}")


if __name__ == "__main__":
    main()
