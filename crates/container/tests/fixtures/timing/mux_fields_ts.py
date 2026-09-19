#!/usr/bin/env python3
"""Mux an H.264 Annex-B elementary stream into MPEG-TS with ONE PES PACKET PER PICTURE — for a
field-coded (PAFF) stream, one per field, each with its own PTS — plus an ADTS AAC track.

    python mux_fields_ts.py <in.264> <audio.aac|-> <out.ts> <picture_ticks> [first_pts] [--pairs]

`picture_ticks` is the 90 kHz spacing of consecutive pictures in decoding order (1500 for the
fields of 30 fps interlaced video; 3000 for its frames). The stream must have no reordering
(I/P only): PTS = DTS = first_pts + n * picture_ticks. A picture starts at a slice NAL (1, 5)
whose first_mb_in_slice is 0 (the first slice-header bit is 1); SPS / PPS / SEI before it travel
with it. AAC frames (1024 samples at 48 kHz = 1920 ticks) start at the same first PTS, one per PES.
PCR rides the first video packet of each PES, 0.7 s behind its DTS. Deterministic.
`--pairs` carries two consecutive pictures (a field pair) in each PES instead, with the first's PTS.
"""
import sys

TS = 188


def nals(es):
    i, n, starts = 0, len(es), []
    while i + 3 <= n:
        if es[i] == 0 and es[i + 1] == 0 and es[i + 2] == 1:
            starts.append(i + 3)
            i += 3
        else:
            i += 1
    out = []
    for k, s in enumerate(starts):
        e = starts[k + 1] - 3 if k + 1 < len(starts) else n
        while e > s and es[e - 1] == 0:
            e -= 1
        out.append(es[s:e])
    return out


def pictures(es):
    pics, pending = [], []
    for nal in nals(es):
        t = nal[0] & 0x1F
        starts_picture = t in (1, 5) and len(nal) > 1 and nal[1] & 0x80
        if starts_picture and any((m[0] & 0x1F) in (1, 5) for m in pending):
            # close the picture being built: keep its non-VCL tail with the next one
            vcl_end = max(i for i, m in enumerate(pending) if (m[0] & 0x1F) in (1, 5)) + 1
            pics.append(pending[:vcl_end])
            pending = pending[vcl_end:]
        pending.append(nal)
    if pending:
        pics.append(pending)
    return [b"".join(b"\x00\x00\x00\x01" + m for m in p) for p in pics]


def adts_frames(aac):
    out, i = [], 0
    while i + 7 <= len(aac):
        assert aac[i] == 0xFF and aac[i + 1] & 0xF0 == 0xF0, "ADTS sync"
        ln = ((aac[i + 3] & 3) << 11) | (aac[i + 4] << 3) | (aac[i + 5] >> 5)
        out.append(aac[i:i + ln])
        i += ln
    return out


def pts_bytes(prefix, pts):
    return bytes([prefix << 4 | ((pts >> 29) & 0x0E) | 1, (pts >> 22) & 0xFF, ((pts >> 14) & 0xFE) | 1,
                  (pts >> 7) & 0xFF, ((pts << 1) & 0xFE) | 1])


def pes(stream_id, pts, payload):
    hdr = bytes([0x80, 0x80, 5]) + pts_bytes(2, pts)
    size = len(hdr) + len(payload) if stream_id < 0xE0 else 0  # video: 0 = unbounded
    return bytes([0, 0, 1, stream_id, (size >> 8) & 0xFF, size & 0xFF]) + hdr + payload


class Muxer:
    def __init__(self):
        self.cc = {}
        self.out = bytearray()

    def packets(self, pid, data, pcr=None):
        first = True
        while first or data:
            cc = self.cc.get(pid, 0)
            self.cc[pid] = (cc + 1) & 15
            has_af, af = False, b""
            if first and pcr is not None:
                has_af = True
                af = bytes([0x10, (pcr >> 25) & 0xFF, (pcr >> 17) & 0xFF, (pcr >> 9) & 0xFF,
                            (pcr >> 1) & 0xFF, ((pcr & 1) << 7) | 0x7E, 0])
            room = TS - 4 - (1 + len(af) if has_af else 0)
            if len(data) < room:
                stuff = room - len(data)
                if has_af:
                    af += bytes([0xFF]) * stuff
                else:
                    has_af = True
                    af = b"" if stuff == 1 else bytes([0x00]) + bytes([0xFF]) * (stuff - 2)
                room = len(data)
            chunk, data = data[:room], data[room:]
            hdr = bytes([0x47, (0x40 if first else 0) | (pid >> 8), pid & 0xFF, (0x30 if has_af else 0x10) | cc])
            self.out += hdr + (bytes([len(af)]) + af if has_af else b"") + chunk
            assert len(self.out) % TS == 0
            first = False

    def psi(self, pid, section):
        crc = crc32(section)
        self.packets(pid, bytes([0]) + section + crc.to_bytes(4, "big"))


def crc32(data):
    crc = 0xFFFFFFFF
    for b in data:
        crc ^= b << 24
        for _ in range(8):
            crc = ((crc << 1) ^ 0x04C11DB7) & 0xFFFFFFFF if crc & 0x80000000 else (crc << 1) & 0xFFFFFFFF
    return crc


def main():
    es = open(sys.argv[1], "rb").read()
    aac = b"" if sys.argv[2] == "-" else open(sys.argv[2], "rb").read()
    out, step = sys.argv[3], int(sys.argv[4])
    pairs = "--pairs" in sys.argv
    args = [a for a in sys.argv if a != "--pairs"]
    first = int(args[5]) if len(args) > 5 else 126000
    pics, frames = pictures(es), adts_frames(aac)
    if pairs:
        pics = [pics[i] + (pics[i + 1] if i + 1 < len(pics) else b"") for i in range(0, len(pics), 2)]
        step *= 2
    m = Muxer()
    pat = bytes([0x00, 0xB0, 13, 0x00, 0x01, 0xC1, 0x00, 0x00, 0x00, 0x01, 0xF0, 0x00])
    streams = bytes([0x1B, 0xE1, 0x00, 0xF0, 0x00])
    if frames:
        streams += bytes([0x0F, 0xE1, 0x01, 0xF0, 0x00])
    pmt_body = bytes([0x00, 0x01, 0xC1, 0x00, 0x00, 0xE1, 0x00, 0xF0, 0x00]) + streams
    pmt = bytes([0x02, 0xB0 | ((len(pmt_body) + 4) >> 8), (len(pmt_body) + 4) & 0xFF]) + pmt_body
    m.psi(0x0000, pat)
    m.psi(0x1000, pmt)
    a = 0
    for n, pic in enumerate(pics):
        pts = first + n * step
        # audio due before this picture
        while a < len(frames) and first + a * 1920 <= pts:
            m.packets(0x101, pes(0xC0, first + a * 1920, frames[a]))
            a += 1
        m.packets(0x100, pes(0xE0, pts, pic), pcr=pts - 63000)
    while a < len(frames):
        m.packets(0x101, pes(0xC0, first + a * 1920, frames[a]))
        a += 1
    open(out, "wb").write(m.out)
    print(f"{out}: {len(pics)} video PES ({step} ticks apart), {len(frames)} audio PES, {len(m.out) // TS} packets")


main()
