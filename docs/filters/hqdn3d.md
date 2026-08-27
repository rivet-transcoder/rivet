# `hqdn3d`

**Temporal denoise** — the classic "high-quality 3D denoiser". Every sample is
low-passed three times: along its row, down its column, and then **against the
same sample of the previous output frame**. Each stage's strength adapts to the
difference it is smoothing — a small difference (noise) is averaged away, a
large one (an edge, motion) passes through — so noise that flickers from frame
to frame is averaged over time while a static picture converges to its clean
value. Applied to luma + chroma; 8-bit `Yuv420p` only.

The parameters, the coefficient tables and the 16-bit intermediate arithmetic
follow ffmpeg's `vf_hqdn3d`, so an ffmpeg command line means the same thing
here.

## Syntax

```text
hqdn3d                                   # ffmpeg's defaults: 4:3:6:4.5
hqdn3d=4:3:6:4.5                         # luma_spatial:chroma_spatial:luma_tmp:chroma_tmp
hqdn3d=8                                 # luma_spatial=8, the rest derived: 8:6:12:9
hqdn3d=luma_spatial=4:luma_tmp=10        # by key; short keys ls / cs / lt / ct
```

```yaml
- hqdn3d: { luma_spatial: 4, chroma_spatial: 3, luma_tmp: 6, chroma_tmp: 4.5 }
```

## Parameters

| Param | Key | Type | Default | Meaning |
|-------|-----|------|---------|---------|
| `luma_spatial` | `ls` | `f32 >= 0` | `4.0` | Spatial strength for luma. |
| `chroma_spatial` | `cs` | `f32 >= 0` | `3·ls/4` | Spatial strength for chroma. |
| `luma_tmp` | `lt` | `f32 >= 0` | `6·ls/4` | Temporal strength for luma. |
| `chroma_tmp` | `ct` | `f32 >= 0` | `lt·cs/ls` | Temporal strength for chroma. |

A value of `0` (or one that is omitted) **derives from the others**, exactly as
ffmpeg's init does. The parser applies the derivation, so a parsed filter
always carries the effective strengths and `hqdn3d=8` displays — and
round-trips — as `hqdn3d=8:6:12:9`.

A strength is "the difference that keeps a quarter of itself": at difference
`s` the stage pulls the sample 25 % of the way toward its neighbour; smaller
differences are pulled harder (a difference of `s/2` about 50 %), larger ones
less, and past ~`255` not at all. So `luma_tmp=6` means a flicker of ±6 is
mostly averaged across frames, while a change of ±40 is treated as motion and
left alone.

## Why it needs state — and what carries it

The temporal stage reads the **previous output frame**, so this filter cannot
run through the stateless per-frame `apply`. The chain is split in two:

- [`FilterChain`](../../crates/codec/src/filter/mod.rs) holds what is shared
  and immutable — for `hqdn3d`, the four coefficient tables. It is prepared
  once per job and cloned (`Arc`) into every decode pump, as before.
- [`FilterInstance`](../../crates/codec/src/filter/mod.rs) —
  `FilterChain::instantiate()` — holds **one stream's** history. The decode
  pump makes one per clip, so:
  - parallel rungs never share a history (they are fed by one pump, after the
    filter);
  - a splice cut starts fresh (each clip is its own stream);
  - two decode pumps (two GPUs, two ranges) each have their own.

`FilterChain::apply` refuses a chain containing `hqdn3d` with an error rather
than silently filtering without history; the stateless filters behave
identically through either path (proven byte-for-byte, see below).

**Range-parallel decode is disabled for a temporal chain.** A decode range
starts with no history, so the frames at every range boundary would differ
from a whole decode. When `DecodePolicy::Ranges(n)` is combined with `hqdn3d`
the job decodes whole and logs
`decode ranges: the filter chain is temporal (frame history); decoding whole`.

The first frame of a stream is filtered against itself (ffmpeg's convention),
so it gets the spatial stages only; the temporal effect builds over the
following frames.

## How well does it work?

Measured end to end with `rivet transcode --codec h264 --filter 'hqdn3d=4:3:6:4.5'`
on a 640×360, 120-frame `testsrc2` clip with ffmpeg's `noise=alls=20:allf=t+u`
added and encoded at `libx264 -crf 12`, comparing every decoded frame to the
*clean* `testsrc2` source by index (PSNR over Y+U+V, higher is closer to clean):

| | mean PSNR vs clean | first 5 frames | last 5 frames |
|---|---|---|---|
| noisy source, decoded | 32.83 dB | 32.83 | 32.82 |
| rivet, no filter (control, same encoder) | 34.03 dB | 33.89 | 34.01 |
| **rivet, `hqdn3d=4:3:6:4.5`** | **36.39 dB** | 35.67 | 36.39 |
| rivet `hqdn3d`, unencoded | 34.07 dB | 33.81 | 34.07 |
| ffmpeg `-vf hqdn3d=4:3:6:4.5`, unencoded | 34.07 dB | 33.81 | 34.07 |

+2.4 dB over the same encode without the filter, +3.6 dB over the source. The
first-frames / last-frames columns show the temporal stage converging: frame 0
(33.5 dB) only has the spatial stages, and the residual falls as history
accumulates. The two unencoded rows are not merely equal to two decimals —
**rivet's output and ffmpeg's are byte-identical on all 120 frames** (zero
MSE), so the encoded figure above is what ffmpeg's filter would give through
the same encoder. The encoded result is better than the unencoded one because
the smoother picture encodes more faithfully at the same QP than the noisy
one does.

## Cost

Scalar, one thread per plane (the three planes run in parallel). The row and
column stages are first-order IIR recurrences — each sample depends on the
one before it — so they do not vectorise across a row the way the spatial
denoisers do; the whole filter is ~3 table lookups per sample.

| | 720p | 1080p |
|---|---|---|
| ms / frame | 3.8 – 6.3 | 6.3 – 10.3 (7.6 – 8.8 with the planes serialised) |

Three runs of 10 frames, same clip and machine as the
[denoise cost table](denoise.md#cost), on a machine shared with other builds
(hence the spread). The output is identical whatever the thread or SIMD
setting.

## Examples

```text
hqdn3d                       # ffmpeg's defaults, a light clean-up
hqdn3d=4:3:6:4.5             # the same, spelled out
hqdn3d=2:1.5:8:6             # gentle spatially, stronger over time (static camera)
hqdn3d=8                     # heavy: 8:6:12:9
```

```sh
rivet transcode noisy.mkv -o clean.mp4 --filter 'hqdn3d=4:3:6:4.5'
```

Combine with a spatial method for stubborn noise — `denoise=bilateral:0.4,hqdn3d`
runs the edge-preserving spatial pass first and the temporal pass on its
output.

## Notes / limits

- **8-bit SDR only** — a 10-bit / HDR frame is rejected rather than mishandled.
- A frame whose size differs from the history (a splice clip of another
  resolution) restarts the history rather than filtering against the wrong
  frame.
- `FilterInstance::reset()` forgets the history explicitly (for a caller that
  knows a cut is coming).
- Bit-exact with ffmpeg: the tables are checked against values computed
  outside the code, and on the 120-frame clip above rivet's output matches
  `ffmpeg -vf hqdn3d=4:3:6:4.5` byte for byte.

Source: [`crates/codec/src/filter/denoise/hqdn3d.rs`](../../crates/codec/src/filter/denoise/hqdn3d.rs).
