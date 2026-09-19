"""Regenerate the multiple-parameter-set fixtures for tests/multi_pps_mux.rs.

    FFMPEG=/path/to/ffmpeg python make_fixtures.py [SOURCES_DIR]

64x64 testsrc2 at 25 fps, 12 frames, an IDR every 6, access-unit delimiters
and the parameter sets repeated at every IDR. ffmpeg's encoders write one
PPS; this rewrites the Annex-B streams they produce so they carry more:

  two_pps.h264   H.264 Main, CAVLC, no B pictures. PPS 1 is PPS 0 under
                 id 1. Every odd picture's slice names PPS 1, and its access
                 unit re-sends PPS 1 in-band before the slice; an IDR's
                 access unit sends PPS 1 before PPS 0. Decodes exactly as the
                 source does (CAVLC slice data is not byte-aligned, so moving
                 the slice's pic_parameter_set_id from `1` to `010` shifts
                 the rest of the slice by two bits and nothing else).
  conflict.h264  the source with PPS 0 re-sent in-band before picture 8
                 with pic_init_qp_minus26 raised by 6: the same id, different
                 contents. A decoder of the Annex-B takes the new set from
                 picture 8 on; an avc1 sample entry can carry only one.
  two_pps.h265   H.265 Main with PPS 1 (PPS 0 under id 1, which no slice
                 names) sent before PPS 0 in every IRAP access unit.

With SOURCES_DIR, ffmpeg's unmodified streams are kept there as
source.h264 / source.h265 for decoding the fixtures against.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FF = os.environ.get("FFMPEG", "ffmpeg")
SRC = ["-f", "lavfi", "-i", "testsrc2=size=64x64:rate=25:duration=0.48"]


def encode(args, fmt):
    return subprocess.run([FF, "-v", "error", *SRC, *args, "-threads", "1", "-f", fmt, "-"],
                          check=True, capture_output=True).stdout


def split_nals(data):
    """NAL units between 00 00 01 prefixes, trailing zero bytes dropped."""
    out, i, starts = [], 0, []
    while True:
        j = data.find(b"\x00\x00\x01", i)
        if j < 0:
            break
        starts.append(j + 3)
        i = j + 3
    for k, s in enumerate(starts):
        e = starts[k + 1] - 3 if k + 1 < len(starts) else len(data)
        out.append(data[s:e].rstrip(b"\x00"))
    return out


def unescape(b):
    out, zeros = bytearray(), 0
    for x in b:
        if zeros >= 2 and x == 3:
            zeros = 0
            continue
        zeros = zeros + 1 if x == 0 else 0
        out.append(x)
    return bytes(out)


def escape(b):
    out, zeros = bytearray(), 0
    for x in b:
        if zeros >= 2 and x <= 3:
            out.append(3)
            zeros = 0
        zeros = zeros + 1 if x == 0 else 0
        out.append(x)
    return bytes(out)


def payload_bits(nal, header):
    """The RBSP's bits before its stop bit."""
    bits = [(x >> (7 - i)) & 1 for x in unescape(nal[header:]) for i in range(8)]
    while bits and bits[-1] == 0:
        bits.pop()
    bits.pop()  # rbsp_stop_one_bit
    return bits


def nal_from_bits(header, bits):
    bits = bits + [1]
    bits += [0] * (-len(bits) % 8)
    body = bytes(int("".join(map(str, bits[i:i + 8])), 2) for i in range(0, len(bits), 8))
    return header + escape(body)


class Reader:
    def __init__(self, bits):
        self.bits, self.pos = bits, 0

    def u(self, n):
        v = 0
        for _ in range(n):
            v = v << 1 | self.bits[self.pos]
            self.pos += 1
        return v

    def ue(self):
        zeros = 0
        while self.u(1) == 0:
            zeros += 1
        return (1 << zeros) - 1 + self.u(zeros)

    def se(self):
        k = self.ue()
        return (k + 1) // 2 if k % 2 else -(k // 2)


def ue_bits(v):
    code = bin(v + 1)[2:]
    return [0] * (len(code) - 1) + [int(c) for c in code]


def se_bits(v):
    return ue_bits(2 * v - 1 if v > 0 else -2 * v)


def with_first_ue(nal, header_len, value):
    """The NAL with its first ue(v) field set to `value`."""
    r = Reader(payload_bits(nal, header_len))
    r.ue()
    return nal_from_bits(nal[:header_len], ue_bits(value) + r.bits[r.pos:])


def h264_pps_with_qp(pps, delta):
    """PPS with pic_init_qp_minus26 moved by `delta` (FMO off, as x264 writes)."""
    r = Reader(payload_bits(pps, 1))
    head = []
    for kind in ["ue", "ue", 1, 1, "ue", "ue", "ue", 1, 2]:
        start = r.pos
        v = r.ue() if kind == "ue" else r.u(kind)
        head += r.bits[start:r.pos]
    qp = r.se()
    return nal_from_bits(pps[:1], head + se_bits(qp + delta) + r.bits[r.pos:])


def h264_slice_on_pps(nal, pps_id):
    """A slice NAL whose header names `pps_id` (first_mb, slice_type, then the id)."""
    r = Reader(payload_bits(nal, 1))
    r.ue()
    r.ue()
    at = r.pos
    r.ue()
    return nal_from_bits(nal[:1], r.bits[:at] + ue_bits(pps_id) + r.bits[r.pos:])


def access_units(nals, is_aud):
    units = []
    for n in nals:
        if is_aud(n) or not units:
            units.append([])
        units[-1].append(n)
    return units


def annexb(units):
    return b"".join(b"\x00\x00\x00\x01" + n for u in units for n in u)


def main():
    keep = sys.argv[1] if len(sys.argv) > 1 else None
    h264 = encode(["-c:v", "libx264", "-profile:v", "main", "-x264-params",
                   "cabac=0:bframes=0:aud=1:keyint=6:min-keyint=6:scenecut=0:repeat-headers=1"], "h264")
    h265 = encode(["-c:v", "libx265", "-x265-params",
                   "bframes=0:aud=1:keyint=6:min-keyint=6:scenecut=0:repeat-headers=1:log-level=error"], "hevc")
    if keep:
        os.makedirs(keep, exist_ok=True)
        open(os.path.join(keep, "source.h264"), "wb").write(h264)
        open(os.path.join(keep, "source.h265"), "wb").write(h265)

    units = access_units(split_nals(h264), lambda n: n[0] & 0x1F == 9)
    assert len(units) == 12, len(units)
    pps0 = next(n for u in units for n in u if n[0] & 0x1F == 8)
    pps1 = with_first_ue(pps0, 1, 1)
    two = []
    for i, unit in enumerate(units):
        out = []
        for n in unit:
            t = n[0] & 0x1F
            if t == 8:
                out.append(pps1)
            if t == 1 and i % 2 == 1:
                out += [pps1, h264_slice_on_pps(n, 1)]
                continue
            out.append(n)
        two.append(out)
    conflict = [list(u) for u in units]
    conflict[8].insert(1, h264_pps_with_qp(pps0, 6))

    units265 = access_units(split_nals(h265), lambda n: (n[0] >> 1) & 0x3F == 35)
    assert len(units265) == 12, len(units265)
    pps265 = next(n for u in units265 for n in u if (n[0] >> 1) & 0x3F == 34)
    pps265_1 = with_first_ue(pps265, 2, 1)
    two265 = []
    for unit in units265:
        out = []
        for n in unit:
            if (n[0] >> 1) & 0x3F == 34:
                out.append(pps265_1)
            out.append(n)
        two265.append(out)

    for name, data in [("two_pps.h264", annexb(two)), ("conflict.h264", annexb(conflict)),
                       ("two_pps.h265", annexb(two265))]:
        open(os.path.join(HERE, name), "wb").write(data)
        print(name, len(data), "bytes")


if __name__ == "__main__":
    main()
