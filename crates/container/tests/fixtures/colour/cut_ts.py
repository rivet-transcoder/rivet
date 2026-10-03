#!/usr/bin/env python3
"""Cut a transport stream inside a GOP, as a recording that starts mid-stream does: drop the
first N PES packets of the video stream, keep everything else (PAT, PMT, the packets of any other
PID) byte for byte. The kept stream then opens on access units that reference pictures it does
not carry.

    python cut_ts.py <in.ts> <out.ts> <N> [--audio-from-video-dts]

`--audio-from-video-dts` also drops every PES of the other elementary streams whose PTS is
before the first kept video PES's DTS, so the audio starts where the cut video does.

The video PID is the first H.264 / HEVC / MPEG-2 video stream (stream_type 0x1B / 0x24 / 0x02)
the PMT lists; the PMT's PID is read from the PAT.
"""
import sys

TS = 188
VIDEO_TYPES = {0x01, 0x02, 0x1B, 0x24}


def pid(p):
    return ((p[1] & 0x1F) << 8) | p[2]


def payload(p):
    off = 4 + (1 + p[4] if p[3] & 0x20 else 0)
    return p[off:] if p[3] & 0x10 else b""


def section(p):
    b = payload(p)
    return b[1 + b[0]:]  # pointer_field


def streams(pkts):
    """{pid: stream_type} from the first PMT."""
    pat = next(section(p) for p in pkts if pid(p) == 0 and p[1] & 0x40)
    length = ((pat[1] & 0x0F) << 8) | pat[2]
    pmt_pid = None
    for i in range(8, 3 + length - 4, 4):
        if (pat[i] << 8 | pat[i + 1]) != 0:
            pmt_pid = ((pat[i + 2] & 0x1F) << 8) | pat[i + 3]
            break
    pmt = next(section(p) for p in pkts if pid(p) == pmt_pid and p[1] & 0x40)
    length = ((pmt[1] & 0x0F) << 8) | pmt[2]
    info = ((pmt[10] & 0x0F) << 8) | pmt[11]
    out, i = {}, 12 + info
    while i < 3 + length - 4:
        out[((pmt[i + 1] & 0x1F) << 8) | pmt[i + 2]] = pmt[i]
        i += 5 + (((pmt[i + 3] & 0x0F) << 8) | pmt[i + 4])
    return out


def ts_field(b):
    return ((b[0] >> 1) & 7) << 30 | ((b[1] << 7 | b[2] >> 1) & 0x7FFF) << 15 | ((b[3] << 7 | b[4] >> 1) & 0x7FFF)


def pes_times(p):
    """(PTS, DTS) of the PES this packet opens, or None."""
    if not p[1] & 0x40:
        return None
    b = payload(p)
    if b[:3] != b"\x00\x00\x01" or len(b) < 14:
        return None
    flags = b[7] >> 6
    pts = ts_field(b[9:14]) if flags & 2 else None
    dts = ts_field(b[14:19]) if flags == 3 else pts
    return pts, dts


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    audio_too = "--audio-from-video-dts" in sys.argv
    src, dst, n = args[0], args[1], int(args[2])
    data = open(src, "rb").read()
    pkts = [data[i:i + TS] for i in range(0, len(data), TS)]
    es = streams(pkts)
    video = next(p for p, t in es.items() if t in VIDEO_TYPES)
    starts = [i for i, p in enumerate(pkts) if pid(p) == video and pes_times(p)]
    first_kept = starts[n]
    cut_dts = pes_times(pkts[first_kept])[1]
    out, dropping = [], {}
    for i, p in enumerate(pkts):
        q = pid(p)
        if q == video:
            if i < first_kept:
                continue
        elif audio_too and q in es:
            t = pes_times(p)
            if t is not None:
                dropping[q] = t[0] < cut_dts
            if dropping.get(q, True):
                continue
        out.append(p)
    open(dst, "wb").write(b"".join(out))
    print(f"{dst}: dropped the first {n} of {len(starts)} video PES (PID 0x{video:x}); "
          f"{len(out)} of {len(pkts)} packets kept")


if __name__ == "__main__":
    main()
