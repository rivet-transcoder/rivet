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
| `bilateral` | `bl` | sensor / Gaussian noise — the default | ✅ | fast |
| `gaussian` | `gauss`, `gs` | aggressive smoothing of soft content | ❌ | fastest |
| `median` | `md` | salt-and-pepper / impulse noise | ✅ | fast |
| `mean` | `box`, `average` | cheap blur | ❌ | fastest |
| `nlmeans` | `nlm` | highest quality; texture without blur | ✅ | **~0.8 s/frame** |
| `anisotropic` | `pm`, `diffusion` | edge-preserving, alternative to bilateral | ✅ | medium |

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
The cost is ~`49 × 9` ops per sample, ~10× the others — **offline only**.

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

## Notes / limits

- **Spatial, single-frame only.** The chain is stateless and shared across rungs,
  so temporal denoisers (hqdn3d, NLM-temporal) need per-stream frame history and
  are a follow-up.
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
