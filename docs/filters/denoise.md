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
- **8-bit SDR only** for the classical methods — a 10-bit / HDR frame is rejected
  rather than mishandled. (`dpir` below takes 8-bit and 10-bit.)
- Each algorithm lives in its own file under
  [`crates/codec/src/filter/denoise/`](../../crates/codec/src/filter/denoise/).

## `dpir` — deep denoise

The classical methods top out at non-local means; the next tier is a *learned*
denoiser. `denoise=dpir` runs **DRUNet** from
[DPIR](https://github.com/cszn/DPIR) (Zhang et al., *Plug-and-Play Image
Restoration with Deep Denoiser Prior*, MIT licence): a residual U-Net trained as
a Gaussian denoiser for noise levels σ ∈ [0, 50], which takes the image plus a
constant channel holding σ — one network for every strength. It is an **opt-in,
offline** tier: a Cargo feature, a tensor library, and a 130 MB model file.

```text
denoise=dpir              # grayscale model on luma, σ = 15
denoise=dpir:25           # σ = 25 (8-bit code values)
denoise=dpir:7:color      # RGB model on all three planes
```

| Param | Meaning |
|-------|---------|
| `SIGMA` | The **noise level in 8-bit code values**, `0..=50` (default `15`). Not the classical `0..=1` blend: it tells the network how much noise the footage carries. Too high over-smooths, too low under-denoises. |
| `gray` / `color` | `gray` (default) runs `drunet_gray` on **luma only** and copies chroma; `color` converts 4:2:0 → limited-range R'G'B' (the frame's own BT.601/709/2020 matrix), runs `drunet_color`, and converts back. |

The σ channel is `SIGMA / 255` on the gray path (luma fed as `code / max`) and
`SIGMA / 219` on the colour path (one 8-bit luma code value is 1/219 of the
limited R'G'B' range), so `SIGMA` means the same thing on both. 8-bit *and*
10-bit 4:2:0 are accepted (10-bit is normalised to `[0, 1]`; σ stays on the
8-bit scale).

### Building it

```text
cargo build --features dpir          # CPU (candle, pure Rust)
cargo build --features dpir-cuda     # NVIDIA GPU — needs nvcc at build time
cargo build --features dpir-cudnn    # + cuDNN convolutions (3x faster; needs cudnn.lib / cudnn64_9.dll)
```

Inference runs on [candle](https://github.com/huggingface/candle). The route
was picked by measurement, not preference (drunet_gray, σ=25, f32,
seconds per frame, RTX 3090 + 32-thread Ryzen, CUDA 13.3):

| Runtime (whole frame, bench) | 720p | 1080p |
|---------|-----:|------:|
| candle CPU | 16.5 | 40.8 |
| tract CPU (ONNX, pure Rust) | ~3x slower than candle at 360p | — |
| candle CUDA | 0.95 | 2.2 |
| candle CUDA + cuDNN | **0.31** | **0.70** |

(fp16 on CUDA bought only ~20 % and is not used: the time goes into im2col
traffic, which cuDNN's implicit-GEMM kernels remove.)

What the pipeline actually pays per frame — `rivet transcode … --filter
denoise=dpir:25 --codec h264` with `RUST_LOG=codec::filter::dpir=debug`, which
logs every frame's cost; 512-px tiles, the software H.264 encoder running on
the same box, mean of the frames after the first:

| Feature | 720p | 1080p |
|---------|-----:|------:|
| `dpir` (CPU, 32 threads) | CPU720 | 67 s |
| `dpir-cuda` | CUDA720 | 2.4 s |
| `dpir-cudnn` | CUDNN720 | 0.95 s |

Building the GPU features on **Windows**: CUDA 13 wants
`NVCC_APPEND_FLAGS="-Xcompiler /Zc:preprocessor"` and an MSVC 2022 `cl.exe` on
PATH for candle's kernels; cuDNN 9 can come from `pip install
nvidia-cudnn-cu13` (its `bin/` on PATH at run time), but that wheel ships no
import library, so make one from the DLL — `dumpbin -exports cudnn64_9.dll` →
a `.def` → `lib -def:… -machine:x64 -out:cudnn.lib` — and point the linker at
it with `RUSTFLAGS="-L <dir>"`. Linux needs libcudnn.so.9 and its dev package.

Device: `RIVET_DPIR_DEVICE=cpu|cuda[:N]`; the default is CUDA when the build has
it and a device opens (a warning says why when it falls back), else CPU. The
prepare log line names what ran:
`dpir: DRUNet (gray) loaded model=… device=cuda:0 sigma=25.0 tile=512`.

### The model file

The weights are the upstream release assets, read **directly** in their legacy
`torch.save` layout — nothing to convert, no Python. Download each once:

```text
curl -L --create-dirs -o ~/.cache/rivet/models/drunet_gray.pth  https://github.com/cszn/KAIR/releases/download/v1.0/drunet_gray.pth
curl -L --create-dirs -o ~/.cache/rivet/models/drunet_color.pth https://github.com/cszn/KAIR/releases/download/v1.0/drunet_color.pth
```

Looked up in `$RIVET_DPIR_MODEL` (a file, or a directory holding both), else
`%LOCALAPPDATA%\rivet\models` on Windows / `$XDG_CACHE_HOME/rivet/models` or
`~/.cache/rivet/models` elsewhere. A missing file is an error that prints the
exact `curl` line. `.safetensors` files are accepted too.

### Tiling

Frames are cut into 512-pixel tiles with 32 pixels of context on every side,
edge-replicated up to a multiple of 8 (three stride-2 stages), and only each
tile's own interior is kept — so memory is bounded at any frame size (the
largest activation is ~85 MB). `RIVET_DPIR_TILE=N` changes the tile edge
(`0` = whole frame). On the bench, tiled vs whole-frame 1080p cost 10 % more
and scored within 0.05 dB of each other.

### How well does it work?

Same recipe as the table above (`testsrc2`, ffmpeg `noise=alls=25:allf=t+u`,
per-frame luma PSNR against the *clean* source, 30 frames at 720p unless
noted). Note that ffmpeg's `alls=25` is **not** σ = 25: the noise it adds has
an RMS of about 7 code values, so σ ≈ 7 is the honest DPIR setting for it.

Every row is the same 30-frame clip through the same `rivet transcode …
--codec h264` (software H.264, default quality), so the encoder's own loss is
in every number; the *no filter* row is the baseline.

| `--filter` | luma PSNR vs clean | vs no filter |
|--------|-------------------:|-------------:|
| *(none)* | 32.14 dB | — |
| `denoise=bilateral:0.8` | 41.53 dB | +9.4 |
| `denoise=nlmeans:0.8` | 42.45 dB | +10.3 |
| `denoise=dpir:25` | 37.98 dB | +5.8 |
| `denoise=dpir:15` | 40.26 dB | +8.1 |
| `denoise=dpir:10` | 43.64 dB | +11.5 |
| **`denoise=dpir:7`** (σ ≈ the real noise) | **44.94 dB** | **+12.8** |
| `denoise=dpir:7:color` | COLOR7 | |

(`dpir:25` on CPU over the first 4 frames: 38.07 dB — the two devices agree to
the tolerance below. 1080p, `dpir:25`, CUDA: 38.48 dB.)

Take-away: **DPIR wins when σ matches the noise**; asked for σ=25 on σ≈7
content it over-smooths (the network trusts the number it is given) and the
classical methods, which cannot over-commit, come out ahead. Measure your
noise before picking σ.

### Bit-exactness

Not available across devices — the CPU and the GPU reduce in different
orders. Measured on the release model, six 160×96 synthetic frames, σ=25:
CPU vs CUDA (cuDNN) **max abs diff 1 code value**, 342 / 92 160 samples
(0.37 %) differ. `release_cpu_vs_cuda_within_tolerance` pins a tolerance of
2. The CPU path itself is deterministic: `release_gray_cpu_golden_hash` pins
the FNV-1a of the output luma (`0x210e7cc2e15489ab`), identical at 1, 8 and
32 threads. Both are `#[ignore]` (they need the model):
`RIVET_DPIR_MODEL=… cargo test -p rivet-codec --lib --features dpir-cudnn -- --ignored`.

### Limits

- Spatial, per frame; no temporal model.
- One CUDA worker thread per prepared chain; the network runs one tile at a
  time. Throughput is the network's cost above — an offline tier.
- Without the `dpir` feature the filter still parses and displays, and
  `FilterChain::prepare` says which feature to build.

Source: [`crates/codec/src/filter/dpir/`](../../crates/codec/src/filter/dpir/)
(`mod.rs` options / tiling / colour, `pth.rs` the legacy torch reader,
`net.rs` DRUNet, `run.rs` the prepared filter); the classical methods in
[`crates/codec/src/filter/denoise/`](../../crates/codec/src/filter/denoise/).
