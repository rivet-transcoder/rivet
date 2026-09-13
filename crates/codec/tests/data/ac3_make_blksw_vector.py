"""Make a block-switched AC-3 cross-check vector out of an ordinary one.

ffmpeg's AC-3 encoder never sets `blksw`, so the 256-point transform
(A/52 §7.9.4.2) has no cross-check vector of its own. `blksw[ch]` is the
first bit per channel of every `audblk()` and nothing in the syntax depends
on it — it only selects the inverse transform — so setting it turns any
stream into a (partly) short-block one that both decoders must still agree
on. crc1 and crc2 are re-solved so the frames stay valid (§7.10.1: crc1
covers the first 5/8 of the frame minus the syncword, crc2 the whole frame
minus the syncword, generator x^16 + x^15 + x^2 + 1, register must read
zero).

    python ac3_make_blksw_vector.py in.ac3 out.ac3 [--channels 0,1] [--block N] [--every K]

By default block 0 of every syncframe is switched for every channel. The
options switch only the listed channels (bitstream order), block N of the
frame (only block 0's start is known from `bsi()` alone, so N > 0 takes
the block boundary from the in-tree decoder's trace — see
`ac3_blksw_offsets`), and only every K-th frame. Switching a single
channel is how a real encoder does it (a transient in one channel), and
it is where libavcodec's overlap-add departs from §7.9.4 step 6: a channel
that is not switched must not change when another one is.
"""
import sys

FRMSIZE_48K = [64, 64, 80, 80, 96, 96, 112, 112, 128, 128, 160, 160, 192, 192, 224, 224, 256, 256,
               320, 320, 384, 384, 448, 448, 512, 512, 640, 640, 768, 768, 896, 896, 1024, 1024,
               1152, 1152, 1280, 1280, 1536, 1536]
FRMSIZE_441 = [69, 70, 87, 88, 104, 105, 121, 122, 139, 140, 174, 175, 208, 209, 243, 244, 278,
               279, 348, 349, 417, 418, 487, 488, 557, 558, 696, 697, 835, 836, 975, 976, 1114,
               1115, 1253, 1254, 1393, 1394, 1672, 1673]
NFCHANS = [2, 1, 2, 3, 3, 4, 4, 5]


def crc16(data, init=0):
    crc = init
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x8005) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def solve_prefix_crc(body):
    """The 16-bit word w such that crc16(w.to_bytes(2) + body) == 0.
    crc16 is linear over GF(2): crc(w||body) = crc(w||0^n) ^ crc(body)."""
    target = crc16(body)
    zeros = bytes(len(body))
    cols = [crc16((1 << i).to_bytes(2, 'big') + zeros) for i in range(16)]
    # Gaussian elimination: find w with XOR of chosen cols == target.
    rows = [(cols[i], 1 << i) for i in range(16)]
    w = 0
    for bit in range(15, -1, -1):
        piv = next((r for r in rows if r[0] & (1 << bit)), None)
        if piv is None:
            continue
        rows.remove(piv)
        rows = [(r[0] ^ piv[0], r[1] ^ piv[1]) if r[0] & (1 << bit) else r for r in rows]
        if target & (1 << bit):
            target ^= piv[0]
            w ^= piv[1]
    assert target == 0, "crc prefix not solvable"
    assert crc16(w.to_bytes(2, 'big') + body) == 0
    return w


class Bits:
    def __init__(self, data):
        self.data = bytearray(data)
        self.pos = 0

    def read(self, n):
        v = 0
        for _ in range(n):
            v = (v << 1) | ((self.data[self.pos >> 3] >> (7 - (self.pos & 7))) & 1)
            self.pos += 1
        return v

    def set_bit(self, pos, val):
        byte = pos >> 3
        mask = 1 << (7 - (pos & 7))
        if val:
            self.data[byte] |= mask
        else:
            self.data[byte] &= ~mask & 0xFF


def audblk0_start(frame):
    """Bit offset of audblk 0 = end of bsi() (Table 5.2)."""
    br = Bits(frame)
    br.read(16); br.read(16)  # syncword, crc1
    fscod = br.read(2); br.read(6)
    br.read(5); br.read(3)
    acmod = br.read(3)
    if (acmod & 1) and acmod != 1:
        br.read(2)
    if acmod & 4:
        br.read(2)
    if acmod == 2:
        br.read(2)
    br.read(1)  # lfeon
    br.read(5)  # dialnorm
    if br.read(1):
        br.read(8)
    if br.read(1):
        br.read(8)
    if br.read(1):
        br.read(7)
    if acmod == 0:
        br.read(5)
        if br.read(1):
            br.read(8)
        if br.read(1):
            br.read(8)
        if br.read(1):
            br.read(7)
    br.read(2)
    if br.read(1):
        br.read(14)
    if br.read(1):
        br.read(14)
    if br.read(1):
        n = br.read(6)
        br.read((n + 1) * 8)
    return br.pos, acmod, fscod


def frames_of(data):
    pos = 0
    while pos + 8 <= len(data):
        assert data[pos] == 0x0B and data[pos + 1] == 0x77, f"no sync at {pos}"
        assert data[pos + 5] >> 3 <= 8, "AC-3 only"
        fscod = data[pos + 4] >> 6
        frmsizecod = data[pos + 4] & 0x3F
        words = {0: FRMSIZE_48K[frmsizecod], 1: FRMSIZE_441[frmsizecod], 2: FRMSIZE_48K[frmsizecod] * 3 // 2}[fscod]
        if pos + words * 2 > len(data):
            break
        yield pos, words
        pos += words * 2


def resolve_crcs(frame, words):
    # crc1 first (crc2 covers it): solve the prefix so bytes 2..5/8 read zero.
    five8 = ((words >> 1) + (words >> 3)) * 2
    frame[2:4] = solve_prefix_crc(bytes(frame[4:five8])).to_bytes(2, 'big')
    # crc2: append the remainder so the register reads zero over bytes 2..
    frame[-2:] = crc16(bytes(frame[2:-2])).to_bytes(2, 'big')
    assert crc16(bytes(frame[2:five8])) == 0 and crc16(bytes(frame[2:])) == 0


def main(argv):
    src, dst = argv[0], argv[1]
    channels = None
    block = 0
    every = 1
    offsets = None
    i = 2
    while i < len(argv):
        if argv[i] == '--channels':
            channels = [int(c) for c in argv[i + 1].split(',')]
        elif argv[i] == '--block':
            block = int(argv[i + 1])
        elif argv[i] == '--every':
            every = int(argv[i + 1])
        elif argv[i] == '--offsets':
            # "frame:blk:bitpos" lines from the decoder's block trace, for --block > 0
            offsets = {}
            for line in open(argv[i + 1]):
                f, b, p = line.split()[:3]
                offsets[(int(f), int(b))] = int(p)
        else:
            raise SystemExit(f"unknown option {argv[i]}")
        i += 2
    if block > 0 and offsets is None:
        raise SystemExit("--block N > 0 needs --offsets <file> (frame blk bitpos per line)")
    data = open(src, 'rb').read()
    out = bytearray()
    frames = 0
    switched = 0
    for pos, words in frames_of(data):
        frame = bytearray(data[pos:pos + words * 2])
        if frames % every == 0:
            start0, acmod, _ = audblk0_start(frame)
            start = start0 if block == 0 else offsets[(frames, block)]
            b = Bits(frame)
            for ch in (channels if channels is not None else range(NFCHANS[acmod])):
                b.set_bit(start + ch, 1)
                switched += 1
            frame = b.data
            resolve_crcs(frame, words)
        out += frame
        frames += 1
    open(dst, 'wb').write(out)
    print(f"{frames} frames, {switched} channel-blocks switched to short blocks (block {block}, every {every}) -> {dst}")


if __name__ == '__main__':
    main(sys.argv[1:])
