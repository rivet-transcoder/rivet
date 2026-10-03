#!/usr/bin/env python3
"""Turn an AV1 low-overhead OBU stream (temporal units opened by temporal delimiters, as
GStreamer's av1parse writes with stream-format=obu-stream,alignment=tu) into an IVF file for
GStreamer's ivfparse, editing the bitstream on the way with the AV1 specification's own syntax:

    python av1_ivf.py <in.obu> <out.ivf> [--no-colour-description] [--hdr10 MDCV CLL]

--no-colour-description  rewrite the sequence header with color_description_present_flag = 0
                         (5.5.2 color_config): the stream states no colour at all. The colour
                         description is the only thing removed; the bits after it do not depend
                         on it for a 4:2:0 stream that is not BT.709 / sRGB / identity.
--hdr10 MDCV CLL         add a METADATA_TYPE_HDR_MDCV and a METADATA_TYPE_HDR_CLL metadata OBU
                         (5.8.2, 5.8.3) to the first temporal unit, after its sequence header.
                         MDCV: Rx,Ry,Gx,Gy,Bx,By,WPx,WPy,Lmax,Lmin in the HEVC SEI's units
                         (0.00002 for the chromaticities, 0.0001 cd/m2 for the luminances),
                         converted here to AV1's fixed point (0.16, 24.8 and 18.14);
                         CLL: MaxCLL,MaxFALL.

GStreamer's encoders give an AV1 stream the colour of the raw caps, and GStreamer fills an
unknown colorimetry with a default, so neither of these can be had from the encoders alone.
IVF: 25 fps, one temporal unit per frame.
"""
import struct
import sys


def leb128(v):
    out = bytearray()
    while True:
        b = v & 0x7F
        v >>= 7
        out.append(b | (0x80 if v else 0))
        if not v:
            return bytes(out)


def read_leb128(d, i):
    v = s = 0
    while True:
        b = d[i]
        v |= (b & 0x7F) << s
        s += 7
        i += 1
        if b < 0x80:
            return v, i


def obus(d):
    """(type, header bytes, payload) for each OBU; every one must carry a size field."""
    i, out = 0, []
    while i < len(d):
        h = d[i]
        ext = (h >> 2) & 1
        assert (h >> 1) & 1, "OBU without obu_size"
        hdr = d[i:i + 1 + ext]
        n, j = read_leb128(d, i + 1 + ext)
        out.append(((h >> 3) & 0xF, hdr, d[j:j + n]))
        i = j + n
    return out


def obu(kind_hdr, payload):
    return kind_hdr + leb128(len(payload)) + payload


class Reader:
    def __init__(self, data):
        self.bits = [(b >> (7 - k)) & 1 for b in data for k in range(8)]
        self.pos = 0

    def f(self, n):
        v = 0
        for _ in range(n):
            v = v << 1 | self.bits[self.pos]
            self.pos += 1
        return v

    def uvlc(self):
        zeros = 0
        while not self.f(1):
            zeros += 1
        return (1 << zeros) - 1 + self.f(zeros) if zeros < 32 else (1 << 32) - 1


def colour_description_at(payload):
    """The bit position of color_description_present_flag in a sequence_header_obu (5.5)."""
    r = Reader(payload)
    seq_profile = r.f(3)
    r.f(1)  # still_picture
    reduced = r.f(1)
    decoder_model = False
    if reduced:
        r.f(5)
    else:
        if r.f(1):  # timing_info_present_flag
            r.f(32)
            r.f(32)
            if r.f(1):  # equal_picture_interval
                r.uvlc()
            decoder_model = bool(r.f(1))
            if decoder_model:
                buffer_delay_length = r.f(5) + 1
                r.f(32)
                r.f(5)
                r.f(5)
        initial_display_delay = r.f(1)
        for _ in range(r.f(5) + 1):
            r.f(12)
            if r.f(5) > 7:
                r.f(1)
            if decoder_model and r.f(1):
                r.f(buffer_delay_length)
                r.f(buffer_delay_length)
                r.f(1)
            if initial_display_delay and r.f(1):
                r.f(4)
    wbits, hbits = r.f(4) + 1, r.f(4) + 1
    r.f(wbits)
    r.f(hbits)
    if not reduced and r.f(1):  # frame_id_numbers_present_flag
        r.f(4)
        r.f(3)
    r.f(3)  # use_128x128_superblock, enable_filter_intra, enable_intra_edge_filter
    if not reduced:
        r.f(4)  # interintra, masked compound, warped motion, dual filter
        order_hint = r.f(1)
        if order_hint:
            r.f(2)  # jnt_comp, ref_frame_mvs
        force_sct = 2 if r.f(1) else r.f(1)
        if force_sct > 0 and not r.f(1):  # seq_choose_integer_mv
            r.f(1)
        if order_hint:
            r.f(3)
    r.f(3)  # enable_superres, enable_cdef, enable_restoration
    high_bitdepth = r.f(1)
    if seq_profile == 2 and high_bitdepth:
        r.f(1)
    if seq_profile != 1:
        assert not r.f(1), "monochrome"
    return r.pos


def without_colour_description(payload):
    at = colour_description_at(payload)
    bits = Reader(payload).bits
    if not bits[at]:
        return payload
    while bits[-1] == 0:  # trailing_bits: drop the zeros and the one before them
        bits.pop()
    bits.pop()
    bits = bits[:at] + [0] + bits[at + 25:] + [1]
    bits += [0] * (-len(bits) % 8)
    return bytes(int("".join(map(str, bits[k:k + 8])), 2) for k in range(0, len(bits), 8))


def metadata(kind, body):
    return obu(bytes([5 << 3 | 1 << 1]), leb128(kind) + body + b"\x80")


def hdr10_obus(mdcv, cll):
    rx, ry, gx, gy, bx, by, wx, wy, lmax, lmin = mdcv
    chroma = lambda v: round(v * 65536 / 50000)  # 0.00002 -> 0.16
    body = b"".join(struct.pack(">HH", chroma(x), chroma(y)) for x, y in [(rx, ry), (gx, gy), (bx, by)])
    body += struct.pack(">HH", chroma(wx), chroma(wy))
    body += struct.pack(">II", round(lmax * 256 / 10000), round(lmin * 16384 / 10000))
    return metadata(2, body) + metadata(1, struct.pack(">HH", *cll))


def main():
    args = sys.argv[1:]
    src, dst = args[0], args[1]
    strip = "--no-colour-description" in args
    extra = b""
    if "--hdr10" in args:
        k = args.index("--hdr10")
        extra = hdr10_obus([int(x) for x in args[k + 1].split(",")], [int(x) for x in args[k + 2].split(",")])
    units, width, height = [], 0, 0
    for kind, hdr, payload in obus(open(src, "rb").read()):
        if kind == 2:  # OBU_TEMPORAL_DELIMITER opens a temporal unit
            units.append(bytearray())
        if kind == 1:  # OBU_SEQUENCE_HEADER
            if strip:
                payload = without_colour_description(payload)
            units[-1] += obu(hdr, payload)
            if len(units) == 1:
                units[-1] += extra
            continue
        units[-1] += obu(hdr, payload)
    with open(dst, "wb") as f:
        f.write(b"DKIF" + struct.pack("<HH4sHHIIII", 0, 32, b"AV01", 64, 64, 25, 1, len(units), 0))
        for pts, u in enumerate(units):
            f.write(struct.pack("<IQ", len(u), pts) + u)
    print(f"{dst}: {len(units)} temporal units" + (", no colour description" if strip else "")
          + (", HDR10 metadata OBUs" if extra else ""))


if __name__ == "__main__":
    main()
