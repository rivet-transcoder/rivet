"""Channel-identity check for a multichannel transcode to Opus: decode the
output with an independent decoder and check that each output channel carries
the tone its source channel carried.

    python opus_channel_identity.py out.mp4 [channels=6] [tone Hz per channel ...]

The sources this is run on carry one tone per channel (the AC-3 / E-AC-3 / AAC
e2e and tests/data/audio fixtures: FL 400 Hz, FR 600, FC 800, LFE 50, SL 1000,
SR 1200), so a swapped or folded channel shows up as a tone in the wrong place.
Pass the tones in the source's channel order for any other layout (7.1 has no
default).

The output is decoded by GStreamer (qtdemux, opusparse, opusdec — libopus —
and audioconvert to interleaved float in GStreamer's channel order, which for
5.1 and 7.1 is the WAVE order FL FR FC LFE (BL BR) SL SR), never by FFmpeg;
GST names GStreamer's bin directory if gst-launch-1.0 is not on PATH. Each
output channel's power at every source tone is measured (Goertzel) and printed
as a matrix; every channel must be dominated by its own tone.

This is what caught the Opus encoder feeding libopus's channel-mapping family 1
in the wrong order instead of RFC 7845's (2026-09-13): the stream's header
claimed a perfect 5.1 track whose FC and FR were swapped and whose LFE / SL / SR
were rotated.
"""
import math
import os
import struct
import subprocess
import sys
import tempfile

SECS = 2.5
DEFAULT_TONES = {6: [400, 600, 800, 50, 1000, 1200]}
NAMES = {6: ['FL', 'FR', 'FC', 'LFE', 'SL', 'SR'], 8: ['FL', 'FR', 'FC', 'LFE', 'BL', 'BR', 'SL', 'SR']}


def decode(path, ch):
    """The output's first SECS seconds as `ch` lists of float samples at 48 kHz."""
    gst = os.environ.get('GST')
    launch = os.path.join(gst, 'gst-launch-1.0') if gst else 'gst-launch-1.0'
    fd, tmp = tempfile.mkstemp(suffix='.f32')
    os.close(fd)
    try:
        subprocess.run([launch, '-q', 'filesrc', 'location=' + path.replace(os.sep, '/'), '!', 'qtdemux', '!',
                        'opusparse', '!', 'opusdec', '!', 'audioconvert', '!', 'audioresample', '!',
                        'audio/x-raw,format=F32LE,rate=48000,channels=%d' % ch, '!',
                        'filesink', 'location=' + tmp.replace(os.sep, '/')], check=True)
        data = open(tmp, 'rb').read()
    finally:
        os.remove(tmp)
    n = min(len(data) // 4, int(SECS * 48000) * ch)
    pcm = struct.unpack('<%df' % n, data[:n * 4])
    return [pcm[c::ch] for c in range(ch)]


def power(x, hz, rate=48000):
    """Goertzel power of `x` at `hz`, normalised by its length."""
    w = 2 * math.pi * hz / rate
    c = 2 * math.cos(w)
    s1 = s2 = 0.0
    for v in x:
        s1, s2 = v + c * s1 - s2, s1
    return (s1 * s1 + s2 * s2 - c * s1 * s2) / max(len(x), 1) ** 2


def main():
    path = sys.argv[1]
    ch = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    tones = [float(t) for t in sys.argv[3:]] or DEFAULT_TONES.get(ch)
    if not tones or len(tones) != ch:
        sys.exit('give one tone per channel for %d channels' % ch)
    out = decode(path, ch)
    label = NAMES.get(ch, ['ch%d' % i for i in range(ch)])
    m = [[power(out[i], tones[j]) for j in range(ch)] for i in range(ch)]
    print('rows = output channel, cols = source channel tone; share of the row\'s tone power')
    print('      ' + ' '.join('%6s' % n for n in label))
    ok = True
    for i in range(ch):
        total = sum(m[i]) or 1e-30
        print('%5s ' % label[i] + ' '.join('%6.3f' % (v / total) for v in m[i]))
        j = max(range(ch), key=lambda k: m[i][k])
        if j != i or m[i][i] / total < 0.8:
            ok = False
    print('channel identity: %s' % ('OK' if ok else 'MISMATCH'))
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
