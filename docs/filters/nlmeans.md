# `nlmeans`

**Non-local means** with its real parameters exposed, matching
`ffmpeg -vf nlmeans`. Applied to luma + chroma; 8-bit `Yuv420p` only.

> There are two ways into this algorithm. [`denoise=nlmeans:STRENGTH`](denoise.md)
> runs it at a fixed internal setting behind a uniform strength dial — the right
> control when you're choosing *between* denoisers and want the same number to
> mean the same amount of denoising whichever you pick. This page is the other
> one: for tuning nlmeans *itself*.

## Syntax

```text
nlmeans                              # ffmpeg's defaults: s=1, p=7, r=15
nlmeans=s=1:p=7:pc=5:r=3:rc=3        # keys, in any order
nlmeans=1:7:5:3:3                    # positional, in declaration order
```

```yaml
- nlmeans: { s: 1.0, p: 7, pc: 5, r: 3, rc: 3 }
```

## Parameters

| Param | Type | Default | Meaning |
|-------|------|---------|---------|
| `s` | `f32` `1.0..=30.0` | `1.0` | Denoising strength (σ). Higher denoises harder. |
| `p` | `u32` `0..=99`, odd | `7` | **Patch size** — how much context defines "these two places look alike". |
| `pc` | `u32` `0..=99`, odd | `0` = same as `p` | Patch size for the chroma planes. |
| `r` | `u32` `0..=99`, odd | `15` | **Research window** — how far afield to look for lookalikes. |
| `rc` | `u32` `0..=99`, odd | `0` = same as `r` | Research window for the chroma planes. |

Sizes are in **samples**, as on the ffmpeg command line, and are forced odd
(`size | 1`) — so `0` and `1` both mean a 1×1 window, and `4` means 5.

## How it works

For every offset in the `r`×`r` research window, each candidate sample is
weighted by how similar its `p`×`p` surroundings are to the centre's:

```text
weight = exp(−SSD / (s·10)²)
```

`SSD` is the *sum* of squared differences across the patch — not the mean. That
is what gives `s` its range: summing over a 7×7 patch makes the exponent fall
away quickly, so `s=1` is a light touch and `s=30` is heavy. It's also why the
two knobs interact: enlarging `p` at fixed `s` denoises **less**, because more
samples contribute to the same sum.

Because the algorithm matches *surroundings* rather than individual samples, a
repeating texture reads as signal and survives — see the numbers in
[denoise.md](denoise.md#how-well-does-it-work), where nlmeans tops the table.

## Choosing values

- **`s`** is the dial you reach for first. Start at ffmpeg's `1.0`; go up until
  the noise goes and stop before the detail does.
- **`p`** larger = more conservative (needs a better match to average), smaller
  = more aggressive.
- **`r`** larger = more places to find a match, and quadratically more work.
- **`pc` / `rc`** let chroma be denoised harder than luma, which is usually the
  right trade: chroma noise is more visible and chroma detail less so.

## Cost

The per-offset patch distance goes through a summed-area table, so the cost is
`O(r² · w · h)` — **the patch size is free**, and only the research window
drives the time. `r` is therefore the only parameter that buys speed, and `r=3`
(9 offsets) is the floor: `r=1` degenerates to a 1×1 window, which is the
identity.

The kernel runs **across all cores, with AVX2**. The plane splits into row
bands, each rebuilding the `pr`-row halo its patch windows reach into, so there
is nothing to synchronise and nothing to reduce. Within a band, both inner row
loops — building the summed-area table and accumulating the weighted samples —
have AVX2 forms. The table is built per band rather than per plane, which keeps
it in cache: a full-plane table at 1080p is 16 MiB per offset.

Every one of those paths is **bit-identical** to the plain scalar one. Same
weight table, same truncation, and a separate multiply and add rather than an
FMA, because `sum += wt * v` rounds twice where a fused multiply-add rounds
once. A file's checksum must not depend on how many cores encoded it or which
instructions the host had.

Measured over 1439 frames of 1080p at `s=1:p=7:pc=5:r=3:rc=3`, transcoding to
HEVC at `--crf 22` on a 6-core / 12-thread Ryzen with 3× Arc:

| | wall | throughput | filter's own share |
|---|---|---|---|
| single-threaded (original) | 321.7 s | 4.5 fps | 303.6 s |
| row bands across cores | 87.6 s | 16.4 fps | 69.4 s |
| + AVX2 row kernels | **43.6 s** | **33.0 fps** | **25.5 s** |
| no filter, same pipeline | 18.2 s | 79.3 fps | — |

7.4× end to end, and 11.9× on the filter itself. All four rows produce the same
output byte for byte.

The fixed-parameter `denoise=nlmeans` (7×7 window, 3×3 patch) now runs on
the same machinery — summed-area table, row bands, SSE4.1 / AVX2 row kernels
— and went from 2.1 s to 32 ms per 1080p frame (947 ms → 15 ms at 720p) with
its output unchanged byte for byte; the direct per-sample loop it replaced is
kept as the test reference. `RIVET_DENOISE_MAX_SIMD=avx2|sse41|none` caps the
tier for both entry points (this kernel has AVX2 and scalar forms only; a cap
to `sse41` runs it scalar), `RIVET_DENOISE_THREADS=n` caps the bands. See the
[denoise cost table](denoise.md#cost) for the same-clip comparison.

That leaves nlmeans costing about 1.4× the rest of the pipeline put together, so
it is still the expensive filter here — and still offline-tier at ffmpeg's
default `r=15`, which is 225 offsets, 25× the work of `r=3`. If you need more
than this, [`denoise=bilateral`](denoise.md) is edge-preserving too and costs a
fraction; the PSNR table on that page puts it at +4.6 dB against nlmeans'
+5.2 dB.

## Examples

```text
nlmeans=s=1:p=7:pc=5:r=3:rc=3   # light, fast: small research window
nlmeans=s=10:p=5:r=9            # a real clean-up pass
nlmeans=s=3:p=7:r=5:rc=9        # gentle on luma, harder on chroma
nlmeans                          # ffmpeg's defaults — slow (r=15)
```

```sh
rivet transcode noisy.mkv -o clean.mp4 --filter 'nlmeans=s=1:p=7:pc=5:r=3:rc=3'
```

## Notes / limits

- **Spatial, single-frame only** — for the temporal dimension, chain
  [`hqdn3d`](hqdn3d.md) after it.
- **8-bit SDR only** — a 10-bit / HDR frame is rejected rather than mishandled.
- Border addressing is edge-replicate.
- This is rivet's own implementation of the published algorithm, not a port. It
  follows ffmpeg's **parameter semantics** — same names, same units, same
  defaults, same weighting formula — so a command line transfers and means the
  same thing. It is not bit-exact with ffmpeg's output.

Source: [`crates/codec/src/filter/denoise/nlmeans.rs`](../../crates/codec/src/filter/denoise/nlmeans.rs).
