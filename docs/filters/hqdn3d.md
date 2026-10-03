# `hqdn3d`

**Spatio-temporal ("3D") denoise.** A spatial edge-preserving low-pass within
each frame, then a temporal one against the previous output frame: on a static
area noise averages out over time, while edges and motion pass through. The
one **stateful** filter — it needs the previous frame. 8-bit `Yuv420p` only.

## Syntax

The option syntax is compatible with the `hqdn3d` filter on the FFmpeg command
line (same option names, positional order, defaults and derivation of omitted
values); the implementation is rivet's own (see [Provenance](#provenance)), so
outputs are not expected to match any other implementation.

```text
hqdn3d                                   # defaults: 4:3:6:4.5
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

A value of `0`, or one that is omitted, is **derived from the others** as the
defaults column says. The parser applies the derivation, so a parsed filter
always carries the effective strengths: `hqdn3d=8` displays — and round-trips
— as `hqdn3d=8:6:12:9`.

A strength is roughly a noise amplitude in 8-bit code values: a strength `S`
smooths differences up to about `1.5·S` and lets larger ones (edges, motion)
through. The best spatial strength is about 1.3–2× the noise's standard
deviation (see the measurements below); the default `4` suits light noise
(σ ≈ 2–3).

## How it works

Every stage is the same edge-preserving first-order recursive filter. With
`x` the incoming sample, `y` the filter's running value and `S` the stage's
strength:

```text
y ← x + k(|y − x|) · (y − x)
k(d) = k_max · exp(−(d / τ)²),   k_max = S / (S + 4),   τ = 1.5 · S
```

`k` is how much of the running value is kept. A small difference is mostly
noise, so up to `k_max` of the running value is kept and the noise averages
out; a difference well beyond `τ` is structure, so `k` falls to ~0 and the
sample passes through untouched. A stronger `S` both averages longer (`k_max`
→ 1) and tolerates larger differences (`τ`). `k_max < 1` always, so the filter
never freezes on a value.

- **Spatial** (`ls`, `cs`): along every row left→right then right→left, then
  down every column top→bottom then bottom→top. Each backward sweep cancels the
  forward sweep's lag, so the smoothing is centred — no smear in one direction.
- **Temporal** (`lt`, `ct`): each spatially filtered sample is blended into
  the previous *output* frame's sample with the same rule. A static area
  converges towards its long-run average; a moving one differs by more than
  `τ` and passes through at once.

The history is kept unrounded (`f32`), so slow convergence is not lost to
rounding; each output frame is rounded once. A frame whose samples are all
equal comes out unchanged. All arithmetic is deterministic and independent of
the thread count.

## Why it needs state — and what carries it

The temporal stage needs the previous output frame of *the same stream*. So
`hqdn3d` cannot run through the stateless paths (`apply`,
`FilterChain::apply`) — they refuse it. Build the chain once with
`FilterChain::prepare` (which tabulates the four strengths' curves; the
prepared chain is immutable and shareable), then `instantiate()` it once per
decode stream; the `FilterInstance` holds that stream's history. Two streams
through one instance would filter each against the other's frames.

The history is dropped — the next frame is filtered as a first frame — on
`FilterInstance::reset()` (call it at a cut) and automatically when the frame
size changes.

## How well does it work?

From `cargo test -p rivet-codec --lib denoise_quality -- --ignored --nocapture`:
a 128×96 synthetic picture (gradient, hard-edged rectangle and disk, a band of
sinusoidal texture) under additive Gaussian noise, PSNR in dB against the
clean picture.

**One frame** (spatial stage only — no history yet), `hqdn3d=LS`:

| noise σ | input | ls=1 | ls=2 | ls=4 | ls=8 | ls=12 | ls=16 |
|--------:|------:|-----:|-----:|-----:|-----:|------:|------:|
| 3 | 38.40 | 38.40 | 39.14 | **43.60** | 42.91 | 39.62 | 38.33 |
| 6 | 32.49 | 32.49 | 32.58 | 33.89 | **40.29** | 39.81 | 38.36 |
| 10 | 28.13 | 28.13 | 28.15 | 28.47 | 31.38 | 36.41 | **37.95** |

**Static scene**, the same picture under fresh noise every frame, PSNR at
frames 1 / 5 / 20:

| noise σ | input | `hqdn3d=2` | `hqdn3d` (4) | `hqdn3d=8` | `hqdn3d=12` |
|--------:|------:|-----------:|-------------:|-----------:|------------:|
| 3 | 38.5 | 39.35 / 40.30 / 40.26 | **43.95 / 47.57 / 47.74** | 42.74 / 43.29 / 43.42 | 39.53 / 39.64 / 39.67 |
| 6 | 32.6 | 32.67 / 32.85 / 32.81 | 34.00 / 35.40 / 35.34 | **40.39 / 43.73 / 44.20** | 39.75 / 40.36 / 40.61 |
| 10 | 28.1 | 28.16 / 28.22 / 28.17 | 28.49 / 28.85 / 28.81 | 31.39 / 33.98 / 33.91 | **36.32 / 39.74 / 40.47** |

At a well-matched strength the temporal stage adds a further 3–4 dB on top of
the spatial stage within a few frames, and frame-to-frame flicker on the
static scene falls to under a third of the input's (asserted by the suite).
Edges stay sharp: a 60 → 180 step under σ = 4 noise keeps a 10–90 % rise of
0.80 samples (an ideal step) after the default `hqdn3d`, where a 3×3 box blur
widens it to 2.4; a bright square that jumps 32 samples between frames
appears at its new place at once with no ghost (max error ≤ 2).

As with any denoiser, a strength far above the noise oversmooths: `ls=16` on
σ = 3 noise loses the texture band and ends slightly below the noisy input.

## Cost

Release build, Ryzen 9 9950X (16C/32T), default `hqdn3d`, ms per frame (luma
+ chroma), median of the frames; scalar code:

| | 720p | 1080p |
|---|---:|---:|
| 1 thread | 26 | 58 |
| all threads (row sweeps split across cores) | 11 | 21 |

## Examples

```text
hqdn3d                       # defaults, a light clean-up
hqdn3d=4:3:6:4.5             # the same, spelled out
hqdn3d=2:1.5:8:6             # gentle spatially, stronger over time (static camera)
hqdn3d=8                     # heavier: 8:6:12:9
```

```sh
rivet transcode noisy.mkv -o clean.mp4 --filter 'hqdn3d=4:3:6:4.5'
```

Combine with a spatial method for stubborn noise — `denoise=bilateral:0.4,hqdn3d`
runs the edge-preserving spatial pass first and this filter on its output.

## Notes / limits

- 8-bit `Yuv420p` only; a 10-bit frame is refused.
- Needs a `FilterInstance` (see above); every rivet front-end creates one per
  decode stream for you.
- A multi-GPU ladder normally splits the decode into ranges; with a temporal
  filter in the chain it decodes the source whole instead, so no frame loses
  its history at a range boundary.

## Provenance

This filter was rewritten clean-room in October 2026. The previous
implementation carried comments naming internals of another project's filter
of the same name, which the project's [clean-room policy](../decisions.md)
does not allow; it was deleted without its body being consulted and replaced
by this design. There is no paper for this filter: the design above is our
own, written from the public, user-facing description of what such a filter
does (an edge-preserving spatial + temporal low-pass with four strengths) and
the public option documentation (names, defaults, derivation of omitted
values), which is the only part it shares with anything else.
