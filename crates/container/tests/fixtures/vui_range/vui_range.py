#!/usr/bin/env python3
"""The SPS test vectors of `demux::hdr`'s `a_vui_range_without_a_colour_description_is_read`:
an encoder's real SPS with its VUI's video signal type rewritten, everything else kept bit for bit.

    python vui_range.py h264|h265 <SPS NAL hex>

prints three variants: `full` (video_signal_type_present_flag 1, video_full_range_flag 1,
colour_description_present_flag 0), `limited` (the same, range 0) and `none`
(video_signal_type_present_flag 0). The SPS is parsed per H.264 7.3.2.1.1 / E.1.1 and H.265
7.3.2.2 / E.2.1 up to the signal type; the parsers stop (assert) on the syntax this does not
need (scaling lists, PCM, short-term RPS in the SPS, sub-layers, POC type 1).

The sources are x264's and x265's SPS through GStreamer — GStreamer's x264enc / x265enc always
write the caps' colour description, which is what this takes out — the parameter sets from
h264parse / h265parse's codec_data:

    gst-launch-1.0 -v videotestsrc num-buffers=1 ! video/x-raw,format=I420,width=64,height=64,framerate=25/1 \
      ! x264enc threads=1 ! video/x-h264,profile=high ! h264parse ! video/x-h264,stream-format=avc ! fakesink
    (x265enc ! h265parse ! video/x-h265,stream-format=hvc1 for H.265)
"""
import sys
def unescape(b):
    out, z = bytearray(), 0
    for x in b:
        if z >= 2 and x == 3: z = 0; continue
        z = z + 1 if x == 0 else 0; out.append(x)
    return bytes(out)
def escape(b):
    out, z = bytearray(), 0
    for x in b:
        if z >= 2 and x <= 3: out.append(3); z = 0
        z = z + 1 if x == 0 else 0; out.append(x)
    return bytes(out)
class R:
    def __init__(s, b): s.b = [(x >> (7 - i)) & 1 for x in b for i in range(8)]; s.p = 0
    def u(s, n):
        v = 0
        for _ in range(n): v = v << 1 | s.b[s.p]; s.p += 1
        return v
    def ue(s):
        z = 0
        while not s.u(1): z += 1
        return (1 << z) - 1 + s.u(z)
def h264_vui_at(r):
    prof = r.u(8); r.u(16); r.ue()
    if prof in (100, 110, 122, 244, 44, 83, 86, 118, 128, 138, 139, 134, 135):
        cf = r.ue()
        if cf == 3: r.u(1)
        r.ue(); r.ue(); r.u(1)
        assert not r.u(1), "scaling matrix"
    r.ue(); poc = r.ue()
    if poc == 0: r.ue()
    assert poc != 1
    r.ue(); r.u(1); r.ue(); r.ue()
    if not r.u(1): r.u(1)
    r.u(1)
    if r.u(1): r.ue(); r.ue(); r.ue(); r.ue()
    assert r.u(1), "no VUI"
    if r.u(1):
        if r.u(8) == 255: r.u(32)
    if r.u(1): r.u(1)
    return r.p
def h265_vui_at(r):
    r.u(4); msl = r.u(3); r.u(1)
    assert msl == 0, "sub-layers"
    r.u(96)
    r.ue(); cf = r.ue()
    if cf == 3: r.u(1)
    r.ue(); r.ue()
    if r.u(1): r.ue(); r.ue(); r.ue(); r.ue()
    r.ue(); r.ue(); r.ue()
    r.u(1); r.ue(); r.ue(); r.ue()
    r.ue(); r.ue(); r.ue(); r.ue(); r.ue(); r.ue()
    assert not r.u(1), "scaling lists"
    r.u(2)
    assert not r.u(1), "pcm"
    assert r.ue() == 0, "short-term RPS in the SPS"
    assert not r.u(1), "long-term refs"
    r.u(2)
    assert r.u(1), "no VUI"
    if r.u(1):
        if r.u(8) == 255: r.u(32)
    if r.u(1): r.u(1)
    return r.p
def variants(codec, nal):
    hdr = 1 if codec == 'h264' else 2
    r = R(unescape(nal[hdr:]))
    at = (h264_vui_at if codec == 'h264' else h265_vui_at)(r)
    bs = r.b[:]
    while bs[-1] == 0: bs.pop()
    bs.pop()
    assert bs[at] == 1 and bs[at + 5] == 1, "signal type with a colour description expected"
    colour = [int(''.join(map(str, bs[at + 6 + 8*k: at + 14 + 8*k])), 2) for k in range(3)]
    fmt = bs[at+1:at+4]
    rest = bs[at + 30:]
    def make(sig):
        b = bs[:at] + sig + rest + [1]
        b += [0] * (-len(b) % 8)
        return nal[:hdr] + escape(bytes(int(''.join(map(str, b[i:i+8])), 2) for i in range(0, len(b), 8)))
    print('source colour', colour, 'format', fmt, 'range', bs[at+4])
    return {'full': make([1] + fmt + [1, 0]), 'limited': make([1] + fmt + [0, 0]), 'none': make([0])}
if __name__ == "__main__":
    codec = sys.argv[1]
    for k, x in variants(codec, bytes.fromhex(sys.argv[2])).items():
        print(k, x.hex())
