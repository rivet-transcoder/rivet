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

The per-offset patch distance is evaluated through a summed-area table, so the
cost is `O(r² · w · h)` — **the patch size is free**, and only the research
window drives the time. Even so this is an offline filter: ffmpeg's default
`r=15` is 225 offsets, i.e. 450 passes over each plane.

Shrinking `r` is the effective lever. The example command below uses `r=3`
(9 offsets), which is ~25× less work than the default.

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

- **Spatial, single-frame only** — same constraint as the rest of the
  [denoise](denoise.md) family: the filter chain is stateless and shared across
  rungs, so temporal denoising needs per-stream frame history.
- **8-bit SDR only** — a 10-bit / HDR frame is rejected rather than mishandled.
- Border addressing is edge-replicate.
- This is rivet's own implementation of the published algorithm, not a port. It
  follows ffmpeg's **parameter semantics** — same names, same units, same
  defaults, same weighting formula — so a command line transfers and means the
  same thing. It is not bit-exact with ffmpeg's output.

Source: [`crates/codec/src/filter/denoise/nlmeans.rs`](../../crates/codec/src/filter/denoise/nlmeans.rs).
