# `denoise`

Spatial denoise with a **selectable algorithm** and a uniform strength dial.
"Denoise" is a family, not one filter — different noise wants different math — so
`denoise` exposes six classical algorithms and lets you pick. Applied to luma +
chroma; 8-bit `Yuv420p` only (the default SDR output).

## Syntax

```text
denoise                  # bilateral, strength 0.5 (defaults)
denoise=METHOD           # named method, strength 0.5
denoise=STRENGTH         # default method (bilateral), given strength
denoise=METHOD:STRENGTH  # both — order-free
```

```yaml
- denoise: { method: bilateral, strength: 0.5 }
```

The two string args are **order-free**: a token that parses as a number is the
strength, anything else is the method (so `denoise=0.7` and `denoise=median`
both work, as does `denoise=0.3:gaussian`). `nr` is an alias for `denoise`.

## Parameters

| Param | Type | Meaning |
|-------|------|---------|
| `method` | enum | The algorithm — see the table below. Default `bilateral`. |
| `strength` | `f32` `0.0..=1.0` | Blend of the filtered result with the source: `out = src·(1−s) + filtered·s`. `0` = off, `1` = fully filtered. Default `0.5`. |

`strength` is deliberately **uniform across methods**: each algorithm runs at a
fixed, moderate internal setting and the strength only controls the blend, so the
same number means the same amount of denoising whichever method you pick.

## The methods

| `method` | Aliases | Best for | Edge-preserving | Speed (720p) |
|----------|---------|----------|:---:|--------------|
| `bilateral` | `bl` | sensor / Gaussian noise — the default | ✅ | 4 ms/frame |
| `gaussian` | `gauss`, `gs` | aggressive smoothing of soft content | ❌ | 2.4 ms/frame |
| `median` | `md` | salt-and-pepper / impulse noise | ✅ | 2.6 ms/frame |
| `mean` | `box`, `average` | cheap blur | ❌ | 1.7 ms/frame |
| `nlmeans` | `nlm` | highest quality; texture without blur | ✅ | 15 ms/frame |
| `anisotropic` | `pm`, `diffusion` | edge-preserving, alternative to bilateral | ✅ | ~100 ms/frame |

(Production configuration — see [Cost](#cost) for the clip, the machine and
the scalar / SSE4.1 / AVX2 breakdown.)

### `bilateral` — edge-preserving (default)

A 5×5 weighted average where each neighbour's weight is `spatial(distance) ×
range(|intensity − centre|)`. The range term collapses across a strong intensity
step, so an **edge barely mixes** while flat noise averages out. The
general-purpose choice for real-world (sensor / compression) noise.

### `gaussian` — plain low-pass

Separable 5-tap blur (`[1,4,6,4,1]/16`). Smooths *everything*, so it softens fine
detail along with the noise — a blunt instrument. Good when content is soft or at
low strength; **can reduce quality on detailed content** (see the numbers below).

### `median` — impulse remover

Replaces each sample with the median of its 3×3 neighbourhood, which deletes
isolated outliers (a stuck-bright/dark pixel) outright while leaving edges intact.
The right tool for salt-and-pepper noise; it does *not* smooth fine Gaussian noise.

### `mean` — box blur

A 3×3 box (separable). The cheapest smoother; same "blurs detail too" caveat as
gaussian, a touch blunter.

### `nlmeans` — non-local means

For each sample, averages a 7×7 search window weighted by how similar each
candidate's 3×3 patch is to the centre's. Because it matches *surroundings*, it
denoises repeating texture without blurring it — the **highest classical quality**.
It is evaluated through a summed-area table of the patch differences (so the
patch is free and the 49 offsets are the cost), on row bands across the cores
with AVX2 row kernels — bit-identical to the direct per-sample loop it
replaced, which is kept as the test reference. Still the most expensive of the
six, at ~4× the bilateral.

> Those window sizes are fixed here, because `strength` is meant to mean the
> same thing across every method on this page. To choose them yourself — patch
> size, research window, separate chroma values, an ffmpeg-compatible σ — use
> the dedicated [`nlmeans`](nlmeans.md) filter instead.

### `anisotropic` — Perona–Malik diffusion

Iterates `u += λ·Σ g(∇)·∇` over the 4-neighbour gradients (8 iterations), where
the conduction `g(∇) = exp(−(∇/κ)²)` falls to ~0 at strong gradients, so the
image diffuses inside flat regions but the flow **stops at edges**. Edge-preserving
like bilateral, with a smoother, more "painterly" character.

## Examples

```text
denoise                    # bilateral 0.5 — sensible default
denoise=bilateral:0.7      # stronger edge-preserving denoise
denoise=median             # clean up salt-and-pepper
denoise=nlmeans:0.6        # best quality, offline render
denoise=anisotropic:0.8    # heavy edge-preserving smoothing
```

## How well does it work?

Measured by adding noise to a clip, denoising, and comparing each frame to the
*clean* source (PSNR — higher is closer to clean; noisy baseline ≈ 31 dB):

| Method (strength 0.8) | PSNR vs clean | vs baseline |
|-----------------------|---------------|-------------|
| `nlmeans` | 36.2 dB | **+5.2** |
| `bilateral` | 35.6 dB | **+4.6** |
| `anisotropic` | 35.1 dB | **+4.0** |
| `gaussian` | 27.5 dB | **−3.5** |

The edge-preserving methods recover real signal. **`gaussian` scored *worse* than
the noisy input** on this sharp synthetic content — that's expected, not a bug:
plain blur trades detail for noise, and on high-detail footage the detail loss
dominates. Use gaussian/mean on soft content or at low strength; reach for
bilateral / nlmeans / anisotropic to actually recover detail. `median` isn't in
the table because the test noise is Gaussian-type — median is for impulse noise.

## Cost

Every method's inner loop runs at one of three tiers — scalar, 128-bit SSE4.1,
256-bit AVX2 — chosen once per process from what the CPU advertises. The
kernels are **bit-identical** to the scalar reference: same tables, same
operation order, no fused multiply-add, and per-kernel tests hold every tier
the host has to the scalar output on random and edge-case planes (widths on
and off every lane multiple, 1×1, flat, 0/255, checkerboards, hard edges,
impulses). `RIVET_DENOISE_MAX_SIMD=avx2|sse41|none` caps the tier;
`RIVET_DENOISE_THREADS=n` caps the row bands the bilateral, median and
nlmeans split across cores. Anisotropic has no lane kernel: its conduction is
`exp` of a non-integer, which no vector `exp` reproduces bit for bit against
the host's libm — so it stays scalar rather than become machine-dependent.

Measured on a 10-frame `testsrc2` clip with ffmpeg's
`noise=all_seed=123:alls=20:allf=t+u` (deterministic), `denoise=METHOD:0.8`,
release build, Ryzen 9 9950X (16C/32T), **ms/frame, median of 10 frames**; one
binary, the tier and thread count switched by environment; "before" is the
pre-kernel binary, "production" is the default configuration (AVX2, all
threads). The last column is the median of per-frame *paired* ratios,
scalar 1 thread → AVX2 1 thread — the SIMD gain alone.

**1080p**

| method | before | scalar 1T | SSE4.1 1T | AVX2 1T | production | before → production | SIMD alone |
|---|---|---|---|---|---|---|---|
| `bilateral` | 129 | 144 | 46.7 | 39.2 | **7.8** | 14× | 5.6× |
| `gaussian` | 22.1 | 21.6 | 12.4 | 8.8 | **6.4** | 2.8× | 3.4× |
| `median` | 176 | 193 | 7.1 | 8.1 | **4.8** | 34× | 29× |
| `mean` | 13.5 | 15.8 | 8.2 | 6.4 | **5.2** | 2.5× | 2.9× |
| `nlmeans` | 2131 | 666 | 478 | 340 | **32** | 70× | 2.0× (+3.6× from the SAT) |
| `anisotropic` | 309 | 446 | — | — | 373 | (scalar; run-to-run noise) | — |

**720p**

| method | before | scalar 1T | SSE4.1 1T | AVX2 1T | production | before → production | SIMD alone |
|---|---|---|---|---|---|---|---|
| `bilateral` | 41.7 | 68.3 | 13.5 | 11.8 | **4.0** | 10× | 5.9× |
| `gaussian` | 7.2 | 9.1 | 3.4 | 2.5 | **2.4** | 3.0× | 3.9× |
| `median` | 69.3 | 69.2 | 2.8 | 3.4 | **2.6** | 27× | 21× |
| `mean` | 4.7 | 4.8 | 2.2 | 2.0 | **1.7** | 2.8× | 2.4× |
| `nlmeans` | 947 | 159 | 138 | 103 | **14.6** | 66× | 1.6× (+6× from the SAT) |
| `anisotropic` | 138 | 137 | — | — | 99 | 1.4× (noise) | — |

Two things the table is honest about. The restructured scalar path is
**slower** than the old monolithic loop for bilateral (the reference is now
a per-row call through a table struct: 0.6–0.7× at one thread) — it is the
specification and the fallback, not the production path. And the machine was
shared with other builds during the run, so the multi-threaded column moved
between runs by up to 2× on the cheap kernels (a quieter first run gave
gaussian 7.6 → 6.4 and mean 7.1 → 5.2 after the band thresholds were tuned);
the single-thread columns were stable to ~10 %. All 14 outputs (7 methods ×
2 resolutions, 10 frames) hashed identical before and after.

## Notes / limits

- **Spatial, single-frame only.** For noise that flickers between frames, the
  temporal [`hqdn3d`](hqdn3d.md) filter averages across time; chain it after a
  spatial method (`denoise=bilateral:0.4,hqdn3d`) for both.
- **8-bit SDR only** — a 10-bit / HDR frame is rejected rather than mishandled.
- Each algorithm lives in its own file under
  [`crates/codec/src/filter/denoise/`](../../crates/codec/src/filter/denoise/).

## Deep denoise (DPIR) — roadmap

The classical methods top out at non-local means; the next tier is a *learned*
denoiser — [**DPIR** (Deep Plug-and-Play Image Restoration)](https://github.com/cszn/DPIR),
whose **DRUNet** CNN is a state-of-the-art Gaussian denoiser. The plan for a
`denoise=dpir` method:

- **Runtime.** Run DRUNet via ONNX — `tract` (pure-Rust, no C dependency, matching
  rivet's hand-rolled-FFI ethos, but CPU-only) or `ort` (onnxruntime, with CUDA /
  DirectML GPU back-ends — much faster, at the cost of a C dependency). Video needs
  GPU inference for real throughput, so `ort` is the likely pick.
- **Model.** Export DRUNet (`drunet_gray` / `drunet_color`) from PyTorch to ONNX
  once and vendor it (~32 MB). It takes the noisy image **plus a noise-level
  channel** (σ), so the filter's `strength` maps to σ.
- **Where it fits.** Exactly the existing **resource-filter** pattern (like
  [overlay](overlay.md)): load the model once in `FilterChain::prepare`, then infer
  per frame. Luma-only with `drunet_gray` is the simplest first cut; a full
  YUV→RGB→DRUNet→YUV colour path is a refinement.
- **Cost.** A U-Net per frame is GPU-bound and not real-time on CPU — an opt-in,
  quality-first, offline tier.

A self-contained sprint (model export + asset + an inference dependency + tensor
plumbing), tracked in [`TODO.md`](../../TODO.md). The classical family above
covers the no-extra-dependency need today.

Source: [`crates/codec/src/filter/denoise/`](../../crates/codec/src/filter/denoise/).
