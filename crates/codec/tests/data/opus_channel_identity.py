"""Channel-identity check for a multichannel transcode: decode the output
(Opus) and the source with ffmpeg — both to N-channel f32 at 48 kHz — find
the lag that best aligns them, and print the N x N correlation matrix.
Every output channel must correlate best with the same-numbered source
channel; feed a source whose channels carry distinct tones (the AC-3 /
E-AC-3 e2e sources do) so a swap shows as a lost correlation.

    python opus_channel_identity.py out.mp4 src.mkv [channels=6]

This is what caught the Opus encoder feeding libopus's channel-mapping
family 1 in ffmpeg's native order instead of RFC 7845's (2026-09-13):
`ffprobe` reported a perfect 5.1 track whose FC and FR were swapped and
whose LFE / SL / SR were rotated. Set FFMPEG to the ffmpeg binary.
"""
import os
import struct
import subprocess
import sys
import tempfile

FF = os.environ.get('FFMPEG', 'C:/Users/elyci/scoop/apps/ffmpeg/current/bin/ffmpeg.exe')
SECS = 2.5


def decode(path, ch):
    fd, tmp = tempfile.mkstemp(suffix='.f32')
    os.close(fd)
    subprocess.run([FF, '-v', 'error', '-y', '-i', path, '-t', str(SECS), '-ac', str(ch), '-ar', '48000',
                    '-f', 'f32le', tmp], check=True)
    data = open(tmp, 'rb').read()
    os.remove(tmp)
    n = len(data) // 4
    pcm = struct.unpack('<%df' % n, data[:n * 4])
    return [pcm[c::ch] for c in range(ch)]


def corr(a, b, lag):
    n = min(len(a), len(b)) - abs(lag)
    if lag >= 0:
        a, b = a[lag:lag + n], b[:n]
    else:
        a, b = a[:n], b[-lag:-lag + n]
    sa = sum(x * x for x in a) ** 0.5 or 1e-30
    sb = sum(y * y for y in b) ** 0.5 or 1e-30
    return sum(x * y for x, y in zip(a, b)) / (sa * sb)


def main():
    ch = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    out = decode(sys.argv[1], ch)
    src = decode(sys.argv[2], ch)
    diag = lambda l: sum(corr(out[c], src[c], l) for c in range(ch))
    best = max(range(-1200, 1201, 8), key=diag)
    best = max(range(best - 8, best + 9), key=diag)
    m = [[corr(out[i], src[j], best) for j in range(ch)] for i in range(ch)]
    names = {6: ['FL', 'FR', 'FC', 'LFE', 'SL', 'SR'], 8: ['FL', 'FR', 'FC', 'LFE', 'BL', 'BR', 'SL', 'SR']}
    label = names.get(ch, ['ch%d' % i for i in range(ch)])
    print('lag %d samples; rows = output channel, cols = source channel' % best)
    print('      ' + ' '.join('%6s' % n for n in label))
    ok = True
    for i in range(ch):
        print('%5s ' % label[i] + ' '.join('%6.3f' % v for v in m[i]))
        j = max(range(ch), key=lambda k: m[i][k])
        if j != i or m[i][i] < 0.8:
            ok = False
    print('channel identity: %s' % ('OK' if ok else 'MISMATCH'))
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
