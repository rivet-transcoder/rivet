#!/usr/bin/env python3
"""Remove a container's colour description so the only colour left is the bitstream's.

    python strip_colour.py strip <file.mp4|file.mkv>   rewrite in place, same length
    python strip_colour.py dump  <file.mp4|file.mkv>   print what the container says

MP4 / MOV: every `colr`, `mdcv` and `clli` box directly inside a visual sample entry
(`avc1`/`avc3`/`hvc1`/`hev1`) is renamed `free`. Same size, so no offset moves.

Matroska: every `Colour` element (0x55B0, which holds `MasteringMetadata`, `MaxCLL` and
`MaxFALL` too) inside `Video` is overwritten with a `Void` element (0xEC) of the same total
length. Mux with `-write_crc32 0` or the level-1 CRC-32 of `Tracks` goes stale.

`dump` prints the colour boxes / elements it finds, or "none", so a fixture's provenance can be
shown rather than asserted.
"""
import struct
import sys

MP4_CONTAINERS = {b"moov", b"trak", b"mdia", b"minf", b"stbl"}
VISUAL_ENTRIES = {b"avc1", b"avc3", b"hvc1", b"hev1"}
COLOUR_BOXES = {b"colr", b"mdcv", b"clli"}


def mp4_walk(buf, start, end, found, in_entry=False):
    pos = start
    while pos + 8 <= end:
        size = struct.unpack(">I", buf[pos:pos + 4])[0]
        kind = bytes(buf[pos + 4:pos + 8])
        header = 8
        if size == 1:
            size = struct.unpack(">Q", buf[pos + 8:pos + 16])[0]
            header = 16
        elif size == 0:
            size = end - pos
        if size < header or pos + size > end:
            break
        if in_entry and kind in COLOUR_BOXES:
            found.append((pos, kind, bytes(buf[pos + header:pos + size])))
        if kind in MP4_CONTAINERS:
            mp4_walk(buf, pos + header, pos + size, found)
        elif kind == b"stsd":
            mp4_walk(buf, pos + header + 8, pos + size, found)
        elif kind in VISUAL_ENTRIES:
            mp4_walk(buf, pos + 8 + 78, pos + size, found, in_entry=True)
        pos += size


def ebml_vint(buf, pos, keep_marker):
    first = buf[pos]
    width = 1
    mask = 0x80
    while width <= 8 and not first & mask:
        width += 1
        mask >>= 1
    if width > 8:
        raise ValueError(f"bad EBML vint at {pos}")
    value = first if keep_marker else first & (mask - 1)
    for b in buf[pos + 1:pos + width]:
        value = (value << 8) | b
    unknown = not keep_marker and value == (1 << (7 * width)) - 1
    return value, width, unknown


MKV_DESCEND = {0x18538067, 0x1654AE6B, 0xAE, 0xE0}  # Segment, Tracks, TrackEntry, Video
MKV_COLOUR = 0x55B0


def mkv_walk(buf, start, end, found, parent=None):
    pos = start
    while pos < end:
        eid, id_w, _ = ebml_vint(buf, pos, True)
        size, size_w, unknown = ebml_vint(buf, pos + id_w, False)
        body = pos + id_w + size_w
        stop = end if unknown else body + size
        if eid == MKV_COLOUR and parent == 0xE0:
            found.append((pos, stop - pos, bytes(buf[body:stop])))
        elif eid in MKV_DESCEND:
            mkv_walk(buf, body, stop, found, eid)
            if eid == 0x1654AE6B:
                return  # Tracks done; the clusters hold no Colour
        pos = stop


def find(buf):
    if buf[4:8] in (b"ftyp", b"moov", b"mdat"):
        found = []
        mp4_walk(buf, 0, len(buf), found)
        return "mp4", found
    if buf[:4] == b"\x1a\x45\xdf\xa3":
        found = []
        mkv_walk(buf, 0, len(buf), found)
        return "mkv", found
    raise SystemExit("neither ISO BMFF nor Matroska")


def main():
    if len(sys.argv) != 3 or sys.argv[1] not in ("strip", "dump"):
        raise SystemExit(__doc__)
    path = sys.argv[2]
    buf = bytearray(open(path, "rb").read())
    kind, found = find(buf)
    if sys.argv[1] == "dump":
        if not found:
            print(f"{path}: {kind}: no container colour description")
        for item in found:
            if kind == "mp4":
                pos, box, body = item
                print(f"{path}: mp4 box {box.decode()} @{pos} body={body.hex()}")
            else:
                pos, total, body = item
                print(f"{path}: mkv Colour @{pos} len={total} body={body.hex()}")
        return
    for item in found:
        if kind == "mp4":
            pos, box, _ = item
            buf[pos + 4:pos + 8] = b"free"
            print(f"{path}: renamed mp4 {box.decode()} @{pos} -> free")
        else:
            pos, total, _ = item
            if total - 2 < 127:  # one-byte size vint
                void = bytes([0xEC, 0x80 | (total - 2)]) + bytes(total - 2)
            else:  # eight-byte size vint
                void = bytes([0xEC, 0x01]) + (total - 9).to_bytes(7, "big") + bytes(total - 9)
            assert len(void) == total
            buf[pos:pos + total] = void
            print(f"{path}: voided mkv Colour @{pos} ({total} bytes)")
    if not found:
        print(f"{path}: {kind}: nothing to strip")
    open(path, "wb").write(buf)


if __name__ == "__main__":
    main()
