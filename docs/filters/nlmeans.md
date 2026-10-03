# `nlmeans`

**Non-local means** denoise with its real parameters exposed. Each output
sample is a weighted average of the samples in a window around it, where a
candidate's weight depends on how much the small *patch* around it looks like
the patch around the sample being denoised. Repeating structure — texture,
text, hatching — finds lookalikes and averages its noise away without being
blurred into its neighbours. Applied to luma + chroma at full weight (there
is no blend; `s` is the strength). 8-bit `Yuv420p` only.

[`denoise=nlmeans[:STRENGTH]`](denoise.md) runs the same kernel at one fixed
setting (3×3 patch, 9×9 window, σ = 10) behind the family's uniform blend
dial; use this filter when you want to tune non-local means itself.

## Syntax

The option syntax is compatible with the `nlmeans` filter on the FFmpeg
command line (same option names, order, defaults and ranges); the
implementation is rivet's own (see [Provenance](#provenance)), so outputs are
not expected to match any other implementation.

```text
nlmeans                              # defaults: s=1, p=7, r=15
nlmeans=s=10:p=7:r=15                # keys, in any order
nlmeans=10:7:5:15:9                  # positional: s:p:pc:r:rc
```

```yaml
- nlmeans: { s: 10.0, p: 7, pc: 5, r: 15, rc: 9 }
```

## Parameters

| Param | Type | Default | Meaning |
|-------|------|---------|---------|
| `s` | `f32` `1.0..=30.0` | `1.0` | Strength: the standard deviation of the noise to remove, in 8-bit code values. |
| `p` | `u32` `0..=99` | `7` | **Patch size** (samples, square) — how much context decides whether two places look alike. |
| `pc` | `u32` `0..=99` | `0` = same as `p` | Patch size for the chroma planes. |
| `r` | `u32` `0..=99` | `15` | **Research window** (samples, square) — how far to look for lookalikes. |
| `rc` | `u32` `0..=99` | `0` = same as `r` | Research window for the chroma planes. |

Sizes are odd; an even size counts as the next odd one (`4` means 5), and `0`
and `1` both mean a single sample. A research window of one sample leaves the
plane untouched. Windows larger than the plane are clipped to it.

## How it works

For a sample `p` and a candidate `q` in the research window around it:

```text
d²(p, q) = mean over the patch of (I(p+u) − I(q+u))²
w(p, q)  = exp(−max(d² − σ², 0) / σ²)        σ = s
out(p)   = Σ_q w(p, q)·I(q) / Σ_q w(p, q)
```

Two patches that differ by less than the noise (`d² ≤ σ²` per sample) count as
"the same" and weigh 1; beyond that the weight decays exponentially, so a
patch that differs by real structure contributes almost nothing. The centre
sample is its own candidate with weight 1. This is the estimator of Buades,
Coll and Morel (2005) with the thresholded weight form of their IPOL article
(2011); our free zone (`σ²`) and decay (`h = σ`) are gentler than the article's
recommended `2σ²` / `0.4σ`, chosen by measurement (below) so that `s` peaks at
the actual noise level and degrades gradually either side of it.

**Borders.** Patches read an edge-replicated copy of the plane; candidates are
always real samples — the research window is clipped at the plane edge, never
padded.

**Speed.** Evaluated directly, every sample costs `patch² × window²`
operations. The kernel instead loops over *offsets*: for one offset `o` it
forms the squared-difference image `(I(x) − I(x+o))²` and box-sums it over the
patch with running column and row sums, so each sample's patch distance costs
O(1) whatever the patch size (the offset-major / integral-image technique of
Wang et al. 2006 and Darbon et al. 2008). The distance is symmetric, so the
weight map for `+o` also serves `−o`, halving the work. Rows are split into
bands across cores (`RIVET_DENOISE_THREADS` caps them).

**Exactness.** Patch distances are exact integer sums; every output sample
accumulates the same terms in the same order however the plane is banded, so
the output is bit-identical for any thread count. A unit test holds the fast
kernel to a direct evaluation of the formula above on random and edge-case
planes, bit for bit.

## Choosing values

- **`s` ≈ the noise σ.** Below it the filter barely acts (patches never look
  alike through the noise); well above it, distinct textures start to count as
  lookalikes and fine detail softens.
- **`p`**: 3–5 for light noise and fine detail, 7 (default) for heavier noise.
  Bigger patches are more robust to noise but match fewer places.
- **`r`**: the cost scales with `r²`. 7–9 is a good fast setting; 15 (default)
  finds more lookalikes and costs ~3× as much as 9.
- **Chroma** is usually noisier and less detailed: a larger `rc`, or the same
  settings, both work.

## How well does it work?

From `cargo test -p rivet-codec --lib denoise_quality -- --ignored --nocapture`:
a 128×96 synthetic picture (gradient, hard-edged rectangle and disk, a band of
sinusoidal texture) under additive Gaussian noise, `p=7 r=15`, PSNR in dB
against the clean picture:

| noise σ | input | s=1 | s=3 | s=5 | s=10 | s=15 | s=20 | s=30 |
|--------:|------:|----:|----:|----:|-----:|-----:|-----:|-----:|
| 5 | 34.03 | 34.03 | 36.72 | **40.76** | 38.08 | 36.07 | 34.93 | 31.76 |
| 10 | 28.15 | 28.15 | 28.15 | 29.37 | **36.91** | 36.30 | 35.03 | 31.90 |
| 20 | 22.19 | 22.19 | 22.19 | 22.19 | 23.98 | 32.47 | **34.24** | 32.26 |

Gains at the matched strength: **+6.7, +8.8 and +12.1 dB**. Edges stay sharp:
on a 60 → 180 step under σ = 10 noise, the row-averaged 10–90 % rise after
`nlmeans=s=10` measures 0.79 samples (an ideal step measures 0.80; a 3×3 box
blur widens it to 2.37) with the full contrast retained.

The suite also asserts: PSNR gain at each noise level with `s` matched, PSNR
rising with `s` up to the noise level, edge width and contrast, flat planes
unchanged, repeating texture preserved at `s=1`, determinism, odd and
degenerate sizes (1×1, 1×N, N×1), banding independence, and `pc`/`rc`
independence from `p`/`r`.

## Cost

Release build, Ryzen 9 9950X (16C/32T), ms per frame (luma + chroma),
median of the frames; scalar code (no hand-written SIMD — the integer box
sums auto-vectorise):

| setting | 720p, 1 thread | 720p, all threads | 1080p, 1 thread | 1080p, all threads |
|---------|---------------:|------------------:|----------------:|-------------------:|
| `denoise=nlmeans` (3×3 patch, 9×9 window) | 145 | 17 | 327 | 37 |
| `nlmeans` (defaults: p=7, r=15) | — | — | — | 88 |

## Examples

```text
nlmeans=s=4:p=5:r=9             # light, fast clean-up of low noise
nlmeans=s=10:p=7:r=15           # a thorough pass on σ≈10 noise
nlmeans=s=6:p=7:r=9:rc=15       # luma moderate, chroma searched wider
```

```sh
rivet transcode noisy.mkv -o clean.mp4 --filter 'nlmeans=s=8:p=7:r=11'
```

## Notes / limits

- 8-bit `Yuv420p` only; a 10-bit frame is refused.
- Spatial only — for noise that changes frame to frame on a static scene,
  chain the temporal [`hqdn3d`](hqdn3d.md) after it.

## Provenance

This filter was rewritten clean-room in October 2026. The previous
implementation carried comments naming internals of another project's
filter, which the project's [clean-room policy](../decisions.md) does not
allow; it was deleted without its body being consulted and replaced by this
one, written only from the published papers:

- A. Buades, B. Coll, J.-M. Morel, "A non-local algorithm for image
  denoising", CVPR 2005.
- A. Buades, B. Coll, J.-M. Morel, "Non-Local Means Denoising", *Image
  Processing On Line* 1 (2011).
- J. Wang, Y. Guo, Y. Ying, Y. Liu, Q. Peng, "Fast non-local algorithm for
  image denoising", ICIP 2006; J. Darbon, A. Cunha, T. Chan, S. Osher,
  G. Jensen, "Fast nonlocal filtering applied to electron cryomicroscopy",
  ISBI 2008 — the offset-major / integral-image speed-up.

and the public user documentation of the command-line options, for option
compatibility only.
