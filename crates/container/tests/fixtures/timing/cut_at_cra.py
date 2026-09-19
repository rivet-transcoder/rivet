#!/usr/bin/env python3
"""Cut an HEVC transport stream at its first CRA after the start, as a broadcast recording starts:
the PAT and PMT, then every packet from the one that opens the CRA's PES on. ffmpeg's -ss with
-c copy would not start a 64x64 open-GOP stream on its CRA (it wrote no video at all).

    python cut_at_cra.py <in.ts> <out.ts>

Video on PID 0x100 (ffmpeg's mpegts default). The PES that holds a CRA (nal_unit_type 21) is found
by reassembling each video PES.
"""
import sys

TS = 188
data = open(sys.argv[1], "rb").read()
pkts = [data[i:i + TS] for i in range(0, len(data), TS)]


def pid(p):
    return ((p[1] & 0x1F) << 8) | p[2]


def payload(p):
    off = 4
    if p[3] & 0x20:
        off += 1 + p[4]
    return p[off:]


def nal_types(es):
    return [es[i + 3] >> 1 & 0x3F for i in range(len(es) - 3) if es[i:i + 3] == b"\x00\x00\x01"]


psi = [p for p in pkts if pid(p) in (0x0000, 0x1000)][:2]
starts = [i for i, p in enumerate(pkts) if pid(p) == 0x100 and p[1] & 0x40]


def pes_bytes(k):
    end = starts[k + 1] if k + 1 < len(starts) else len(pkts)
    return b"".join(payload(p) for p in pkts[starts[k]:end] if pid(p) == 0x100)


cra = next(starts[k] for k in range(1, len(starts)) if 21 in nal_types(pes_bytes(k)))
open(sys.argv[2], "wb").write(b"".join(psi + pkts[cra:]))
print(f"{sys.argv[2]}: cut at packet {cra} of {len(pkts)} (the CRA's PES)")
