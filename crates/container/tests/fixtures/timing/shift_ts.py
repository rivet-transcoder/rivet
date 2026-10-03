#!/usr/bin/env python3
"""Move a transport stream along the 33-bit program clock: every PCR and every PES's PTS / DTS
shifted by the same amount (mod 2^33), so the smallest video PTS (PID 0x100) lands on `first`.
Packet layout, payloads and everything else unchanged.

    python shift_ts.py <in.ts> <out.ts> <first>
    python shift_ts.py <in.ts> <out.ts> --pid <PID> <ticks>

`first` is in 90 kHz ticks; 2^33 - 19592 puts the streams 0.218 s before the wrap. With --pid,
only that PID's PES timestamps move, by `ticks` (the PCR stays): one stream placed later on the
program's clock than its muxer put it.
"""
import sys

TS = 188
WRAP = 1 << 33


def pid(p):
    return ((p[1] & 0x1F) << 8) | p[2]


def payload_off(p):
    return 4 + (1 + p[4] if p[3] & 0x20 else 0)


def get_ts(b, at):
    return ((b[at] >> 1) & 7) << 30 | ((b[at + 1] << 7 | b[at + 2] >> 1) & 0x7FFF) << 15 | ((b[at + 3] << 7 | b[at + 4] >> 1) & 0x7FFF)


def set_ts(b, at, value):
    b[at] = (b[at] & 0xF0) | ((value >> 29) & 0x0E) | 1
    b[at + 1] = (value >> 22) & 0xFF
    b[at + 2] = ((value >> 14) & 0xFE) | 1
    b[at + 3] = (value >> 7) & 0xFF
    b[at + 4] = ((value << 1) & 0xFE) | 1


def pes_at(p):
    """Offset of the PES header this packet opens, or None."""
    if not p[1] & 0x40 or not p[3] & 0x10:
        return None
    o = payload_off(p)
    return o if p[o:o + 3] == b"\x00\x00\x01" and p[o + 3] >= 0xBD else None


def shift(p, ticks, pcr=True):
    q = bytearray(p)
    if pcr and q[3] & 0x20 and q[4] > 0 and q[5] & 0x10:
        base = q[6] << 25 | q[7] << 17 | q[8] << 9 | q[9] << 1 | q[10] >> 7
        base = (base + ticks) % WRAP
        q[6], q[7], q[8], q[9] = base >> 25 & 0xFF, base >> 17 & 0xFF, base >> 9 & 0xFF, base >> 1 & 0xFF
        q[10] = (q[10] & 0x7F) | (base & 1) << 7
    o = pes_at(q)
    if o is not None:
        flags = q[o + 7] >> 6
        if flags & 2:
            set_ts(q, o + 9, (get_ts(q, o + 9) + ticks) % WRAP)
        if flags == 3:
            set_ts(q, o + 14, (get_ts(q, o + 14) + ticks) % WRAP)
    return bytes(q)


def main():
    src, dst = sys.argv[1], sys.argv[2]
    data = open(src, "rb").read()
    pkts = [data[i:i + TS] for i in range(0, len(data), TS)]
    if sys.argv[3] == "--pid":
        which, ticks = int(sys.argv[4], 0), int(sys.argv[5])
        open(dst, "wb").write(b"".join(shift(p, ticks, pcr=False) if pid(p) == which else p for p in pkts))
        print(f"{dst}: PID 0x{which:x} moved on by {ticks} ticks")
        return
    first = int(sys.argv[3])
    video = [get_ts(p, pes_at(p) + 9) for p in pkts if pid(p) == 0x100 and pes_at(p) is not None]
    ticks = (first - min(video)) % WRAP
    open(dst, "wb").write(b"".join(shift(p, ticks) for p in pkts))
    print(f"{dst}: shifted by {ticks} ticks, first video PTS {first}")


if __name__ == "__main__":
    main()
