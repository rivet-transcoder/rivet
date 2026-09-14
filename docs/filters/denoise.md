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

### Method

DRUNet is `UNetRes(in_nc, out_nc, nc=[64,128,256,512], nb=4)` from KAIR,
transcribed layer for layer in `net.rs`: a 3×3 head, three stages of four
residual blocks (`x + conv(relu(conv(x)))`) each followed by a stride-2 2×2
convolution, a four-block body, three transposed-convolution up stages that
add the matching skip, and a 3×3 tail; no biases, no normalisation. The
network's shape is *inferred* from the state dict's tensor shapes, so the
unit tests run a reduced-width copy (`tests/fixtures/dpir_tiny_*.pth`,
generated by `dpir_tiny_pth.py`) and the release file loads unchanged. The
weights are loaded once in `FilterChain::prepare` and shared read-only by
every decode stream (a `FilterInstance` borrows the chain); `apply` is per
frame. On CUDA the network is owned by one dedicated thread that never
exits: candle keeps its cuDNN handle in a thread-local whose destructor
runs during Windows' `LdrShutdownThread`, after the CUDA DLLs have
detached, and would abort the process *after* every frame was filtered.

### Building it

```text
cargo build --features dpir          # CPU (candle, pure Rust)
cargo build --features dpir-cuda     # NVIDIA GPU — needs nvcc at build time
cargo build --features dpir-cudnn    # + cuDNN convolutions (~2x faster again; needs cudnn.lib / cudnn64_9.dll)
```

Inference runs on [candle](https://github.com/huggingface/candle). The route
was picked by measurement, not preference (drunet_gray, σ=25, f32, whole
frame, seconds per frame on a bench binary; RTX 3090 + 32-thread Ryzen 9
9950X, CUDA 13.3):

| Runtime (whole frame, bench) | 720p | 1080p |
|---------|-----:|------:|
| candle CPU | 16.5 | 40.8 |
| tract CPU (ONNX, pure Rust) | ~3x slower than candle at 360p | — |
| candle CUDA | 0.95 | 2.2 |
| candle CUDA + cuDNN | **0.31** | **0.70** |

(fp16 on CUDA bought only ~20 % and is not used: the time goes into im2col
traffic, which cuDNN's implicit-GEMM kernels remove.)

What the pipeline actually pays per frame — `rivet transcode … --filter
denoise=dpir:7 --codec h264` on the 30-frame clip below, with
`RUST_LOG=codec::filter::dpir=debug`, which logs every frame's cost
(`dpir: frame denoised width=… height=… ms=…`); the software H.264 encoder
running on the same box; **mean of the frames after the first**, default
tiling (CPU 512-px tiles, GPU whole frame — see [Tiling](#tiling)):

| Feature | 720p | 1080p |
|---------|-----:|------:|
| `dpir` (CPU, 32 threads, 512-px tiles) | 22.8 s | 57 s |
| `dpir-cuda` | 0.84 s | 1.98 s |
| `dpir-cudnn` | **0.28 s** | **0.61 s** |

(CPU rows: 4 frames, each within ±10 % of the mean; the software encoder's
32 threads share the cores with candle, which is why they sit above the
whole-frame bench figures. The box was shared with other builds during the
first CPU run, which read 27–180 s/frame — the table is the quiet re-run.)
The first frame of a process costs 0.3–1 s more than the rest on the GPU
(cuDNN's per-shape algorithm search); the very first cuDNN run after a
build on this box spent 13 s on it (cuDNN's runtime-compiled engine cache
filling), never seen again. The CPU path's cost is why the tier is offline:
the network is about 4.7 MFLOP per pixel (four scales, two stages of four
64–512-channel 3×3 residual blocks each), so a 720p frame is ~4.3 TFLOP and
a 1080p frame ~9.8 TFLOP — 0.28 s at 720p is the RTX 3090 sustaining
~15 TFLOP/s, and the CPU rows are the Ryzen's ~0.2–0.3 TFLOP/s while it also
runs the encoder.

Building the GPU features on **Windows** (what worked on this box): CUDA
13.3's `nvcc` compiles candle's kernels and needs an MSVC 2022 `cl.exe` on
PATH (`…\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64`) plus
`NVCC_APPEND_FLAGS="-Xcompiler /Zc:preprocessor"`; the first kernel build
takes several minutes, after which a feature change rebuilds in about a
minute. cuDNN 9 can come from `pip install nvidia-cudnn-cu13` — its
`site-packages/nvidia/cudnn/bin` holds `cudnn64_9.dll` and goes on PATH at
run time — but that wheel ships no import library, so make one from the
DLL (`dumpbin /exports cudnn64_9.dll` → a `.def` → `lib /def:cudnn.def
/machine:x64 /out:cudnn.lib`) and point the linker at it with
`RUSTFLAGS="-L <dir>"`. `CMAKE_POLICY_VERSION_MINIMUM=3.5` for the
workspace's C dependencies under CMake 4. Linux needs `libcudnn.so.9` and
its dev package.

Device: `RIVET_DPIR_DEVICE=cpu|cuda[:N]`; the default is CUDA when the build has
it and a device opens (a warning says why when it falls back), else CPU. The
prepare log line names what ran:
`dpir: DRUNet (gray) loaded model=… device=cuda:0 sigma=7.0 tile=2048`.

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

Frames are cut into tiles with 32 pixels of context on every side,
edge-replicated up to a multiple of 8 (three stride-2 stages), and only each
tile's own interior is kept, so memory is bounded at any frame size. The
default edge is **512 on the CPU** (largest activation ~85 MB) and **2048 on
a GPU** (720p and 1080p go through whole; a 4K frame still tiles; largest
activation ~1.1 GB). `RIVET_DPIR_TILE=N` overrides (`0` = whole frame).

The GPU default is measured, not memory-driven: every tile pays its overlap,
so 512-px tiles feed the network about twice the frame's pixels. cuDNN
build, `dpir:7`, mean per frame after the first, same clip; the PSNR column
is the luma PSNR of the transcode against the clean source:

| `RIVET_DPIR_TILE` | 720p | 1080p | PSNR 720p / 1080p |
|---|---:|---:|---|
| 512 | 0.46 s | 1.01 s | 44.80 / 44.94 dB |
| 1024 | 0.32 s | 0.82 s | 44.80 / 44.94 dB |
| 0 (whole) | **0.28 s** | **0.68 s** | 44.79 / 44.94 dB |

### How well does it work?

Same recipe as the classical table above: `testsrc2` 1280×720, 30 frames,
ffmpeg `noise=all_seed=123:alls=25:allf=t+u`, encoded near-losslessly
(`libx264 -crf 6`); per-plane PSNR of the transcode against the *clean*
source. ffmpeg's `alls=25` is **not** σ = 25: the noise it adds has an RMS of
about 7 code values (the noisy clip scores 30.99 dB), so σ ≈ 7 is the honest
DPIR setting for it. Every row is the same clip through the same `rivet
transcode … --codec h264` (software H.264, default quality), so the encoder's
own loss is in every number; the *no filter* row is the baseline. Cost is the
per-frame filter time on this box, from the classical
[Cost](#cost) table for the classical rows (same source, production
configuration) and from the debug log for `dpir` (cuDNN, whole frame).

| `--filter` | luma PSNR vs clean | vs no filter | cost / frame |
|--------|-------------------:|-------------:|------------:|
| *(none)* | 32.09 dB | — | — |
| `denoise=bilateral:0.8` | 41.30 dB | +9.2 | 4 ms |
| `denoise=nlmeans:0.8` | 42.27 dB | +10.2 | 15 ms |
| `denoise=bilateral:1.0` | 40.98 dB | +8.9 | 4 ms |
| `denoise=nlmeans:1.0` | 42.46 dB | +10.4 | 15 ms |
| `denoise=dpir:25` | 37.95 dB | +5.9 | 0.28 s |
| `denoise=dpir:15` | 40.26 dB | +8.2 | 0.28 s |
| `denoise=dpir:10` | 43.57 dB | +11.5 | 0.28 s |
| **`denoise=dpir:7`** (σ ≈ the real noise) | **44.80 dB** | **+12.7** | 0.28 s |

1080p, same recipe: no filter 32.08 dB, `dpir:7` 44.94 dB, `dpir:25` 38.46 dB.
The CPU, CUDA and cuDNN builds agree on the output to the tolerance below:
`dpir:7` at 720p scores 44.79 dB (cuDNN), 44.78 dB (CUDA) and 44.79 dB
(CPU) on the same 30 frames. The classical rows are at their best strength
on this content (bilateral peaks at 0.8, nlmeans at 1.0).

Take-away: **DPIR wins when σ matches the noise**; asked for σ=25 on σ≈7
content it over-smooths (the network trusts the number it is given) and the
classical methods, which cannot over-commit, come out ahead. Measure your
noise before picking σ.

### The colour path

`denoise=dpir:7:color` on the clip above scores 38.14 dB luma (U 37.21,
V 32.84; unfiltered 32.09 / 32.38 / 32.41): every plane is denoised, but
luma is well short of the gray model and the Cr plane barely moves. The clip
is the reason, and it is a real-world one: ffmpeg wrote `testsrc2` with
BT.601 and no tag, and an untagged source is BT.709 to the pipeline, so the
R'G'B' the network sees is built with the *wrong* matrix — most of the
saturated bars land outside the RGB cube (red at R' ≈ 1.2), which DRUNet was
never trained on. The conversion itself is lossless either way (it does not
clamp; a clamp cost 12 dB here), but what the network makes of out-of-range
input is not something a tag can fix afterwards.

The same pixels re-encoded with their true `smpte170m` tag tell the real
story. (rivet's SDR output policy converts a BT.601-tagged source to BT.709,
so these rows are measured against the BT.709 rendering of the same content;
the unfiltered transcode is 31.62 dB.) Per-plane PSNR, 30 frames, cuDNN:

| `--filter` | Y | U | V |
|---|---:|---:|---:|
| *(none)* | 31.62 | 32.80 | 32.78 |
| `denoise=dpir:7` (gray) | **42.00** | 33.14 | 33.12 |
| `denoise=dpir:7:color` | 38.57 | 37.20 | 37.90 |
| `denoise=dpir:10:color` | 41.30 | 39.16 | 39.51 |
| **`denoise=dpir:13:color`** | 41.69 | **40.19** | **39.79** |
| `denoise=dpir:15:color` | 41.49 | 40.44 | 39.76 |
| `denoise=dpir:20:color` | 39.94 | 40.18 | 39.12 |
| `denoise=nlmeans:1.0` | 40.88 | 39.68 | 39.40 |

In-cube, the colour model denoises chroma by +7 dB over the gray path
(which copies it) and matches the gray model's luma within 0.3 dB — but at
**σ ≈ 13, not 7**. That is not a mismatch in the σ mapping: `color` wants
the noise level *of R'G'B'*, and 4:2:0 chroma noise lands there amplified
(R' = Y' + 1.575·Cr', so with equal noise on every plane the R'G'B' noise is
≈ 1.9× the luma noise — 7 → 13 on this clip). Rule of thumb: for `color`,
use the luma σ when only luma is noisy and about twice it when chroma is as
noisy as luma. Tag your sources; on an untagged BT.601 file the gray model
is the safe choice, since luma never leaves `[0, 1]`.

### Bit-exactness

Not available across devices — the CPU and the GPU reduce in different
orders, and cuDNN picks its own convolution algorithms. Measured on the
release model at σ=25 over six whole-frame 160×96 synthetic frames plus one
640×360 frame through 256-px tiles (so the seams are in the sample; 322 560
luma samples), CPU vs GPU output in 8-bit code values:

| build | max abs diff | samples differing |
|---|---:|---:|
| `dpir-cudnn` | 1 | 1 188 (0.368 %) |
| `dpir-cuda` | 1 | 5 (0.002 %) |

`release_cpu_vs_cuda_within_tolerance` pins a tolerance of **2** — the
measured ceiling with headroom — and prints the histogram it measured. The
CPU path itself is deterministic: `release_gray_cpu_golden_hash` pins the
FNV-1a of the output luma (`0x210e7cc2e15489ab`), identical at 1, 8 and 32
threads and reproduced by the CPU path of all three builds. Both tests are
`#[ignore]` (they need the model):
`RIVET_DPIR_MODEL=… cargo test -p rivet-codec --lib --features dpir-cudnn -- --ignored`.
The pin is not decorative: dropping the network's head skip connection
moves the hash to `0x356fa9f4af78719a` and the test's MSE from 15.5 to
16 305; the reduced-fixture tests, which check shapes and behaviour rather
than values, all still pass on that mutation.

### Measuring it yourself

Two ffmpeg `psnr` traps cost an afternoon here, so: (1) rivet's H.264
output carries BT.709 VUI while a `y4m` reference carries no tags, and
ffmpeg (6.1+) inserts a colour-matrix conversion between mismatched `psnr`
inputs — a 48 dB transcode reads as 22 dB, uniformly, with flat patches
still identical. Pin both sides. (2) The filter's end-of-stream default
*repeats* the shorter input's last frame, so a 4-frame output against a
30-frame reference averages in 26 bogus comparisons; ask for `shortest=1`.

```text
ffmpeg -i out.mp4 -i clean.y4m -lavfi "[0:v]setparams=colorspace=bt709:range=tv:color_primaries=bt709:color_trc=bt709[a];[1:v]setparams=colorspace=bt709:range=tv:color_primaries=bt709:color_trc=bt709[b];[a][b]psnr=shortest=1" -f null -
```

### Limits

- Spatial, per frame; no temporal model.
- One CUDA worker thread per prepared chain; the network runs one tile at a
  time. Throughput is the network's cost above — an offline tier.
- The colour path is only as good as the source's colour tag (above).
- Without the `dpir` feature the filter still parses and displays, and
  `FilterChain::prepare` says which feature to build.

Source: [`crates/codec/src/filter/dpir/`](../../crates/codec/src/filter/dpir/)
(`mod.rs` options / tiling / colour, `pth.rs` the legacy torch reader,
`net.rs` DRUNet, `run.rs` the prepared filter); the classical methods in
[`crates/codec/src/filter/denoise/`](../../crates/codec/src/filter/denoise/).
