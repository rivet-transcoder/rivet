# Video filters

Per-frame transforms applied to the decoded source **once**, before fan-out +
per-rung scaling — so a filter applies to every rendition. They transform the
*source*; the per-rung scaler then resizes the result to each rung. (So if a
crop changes the aspect ratio, set the rung dimensions to match.)

A chain is a list of [`codec::filter::VideoFilter`](../../crates/codec/src/filter/mod.rs)
values. The implementation mirrors this catalog — **each filter is its own file**
under [`crates/codec/src/filter/`](../../crates/codec/src/filter/), and each has
its own page here.

## Catalog

**Geometry** — pure sample rearrangement, any bit depth (8- or 10-bit):

| Filter | Page | Effect |
|--------|------|--------|
| `crop` | [crop.md](crop.md) | Cut out a `w×h` region (centred or at `x,y`). |
| `pad` | [pad.md](pad.md) | Letterbox / pillarbox into a larger canvas. |
| `hflip` | [hflip.md](hflip.md) | Mirror horizontally. |
| `vflip` | [vflip.md](vflip.md) | Mirror vertically. |
| `rotate` | [rotate.md](rotate.md) | Rotate 90 / 180 / 270° clockwise. |
| `grayscale` | [grayscale.md](grayscale.md) | Drop colour. |

**Colour** — 8-bit `Yuv420p` (the default SDR output):

| Filter | Page | Effect |
|--------|------|--------|
| `invert` | [invert.md](invert.md) | Photo-negative. |
| `brightness` | [brightness.md](brightness.md) | Luma offset. |
| `contrast` | [contrast.md](contrast.md) | Luma contrast around mid-grey. |
| `saturation` | [saturation.md](saturation.md) | Chroma intensity. |

**Composite & restore** — 8-bit `Yuv420p`:

| Filter | Page | Effect |
|--------|------|--------|
| `overlay` | [overlay.md](overlay.md) | Alpha-composite a PNG (logo / watermark). |
| `denoise` | [denoise.md](denoise.md) | Spatial denoise, 6 selectable algorithms behind one strength dial. |
| `nlmeans` | [nlmeans.md](nlmeans.md) | Non-local means with its own patch / research-window parameters (ffmpeg-compatible). |

> `denoise=nlmeans:0.6` and `nlmeans=s=6:p=7:r=5` reach the same algorithm from
> opposite directions: the first is a uniform "how much" dial for comparing
> methods, the second exposes the knobs for tuning nlmeans itself.

4:2:0 alignment means crop / pad / overlay sizes round to even. A chain is
**validated when the spec is built** — a bad value like `rotate=45` is rejected
up front, not at encode time. The overlay PNG is opened + decoded at that point
too, so a missing or unreadable image fails the job immediately.

## Two interchangeable forms

Two serializations round-trip exactly (`parse_chain(&chain_to_string(c)) == c`) —
use whichever fits the surface.

### String chain (ffmpeg `-vf` style)

Comma-separated, each `name` or `name=a:b:…`:

```text
crop=W:H[:X:Y]   pad=W:H[:X:Y]   hflip   vflip   rotate=90|180|270   grayscale
overlay=PATH[:X:Y]   invert   brightness=N   contrast=F   saturation=F
denoise[=METHOD][:STRENGTH]
```

e.g. `crop=1280:720,hflip,rotate=90` or `overlay=logo.png:24:24,saturation=1.2`
or `denoise=median:0.6`. Accepted aliases: `gray` = `grayscale`, `transpose` =
`rotate=90`, `negate` = `invert`, `nr` = `denoise` (denoise methods have their
own aliases — see [denoise.md](denoise.md)). An overlay `PATH` can't contain `:`
in the string form — use the structured form for paths that do.

### Structured objects (YAML / JSON)

The batch manifest and the HTTP JSON `spec` body accept the same filters as a
**list of objects** — unit filters as bare strings, parameterised ones as a
tagged object:

```yaml
filter:
  - crop: { w: 1280, h: 720 }   # x/y optional → centred
  - hflip
  - rotate: 90
  - overlay: { image: assets/logo.png, x: 24, y: 24 }
  - saturation: 1.2
  - denoise: { method: bilateral, strength: 0.5 }
```

```json
"filter": [{ "crop": { "w": 1280, "h": 720 } }, "hflip", { "rotate": 90 }]
```

Both forms resolve to the same validated `Vec<VideoFilter>`.

## Per-surface usage

| Surface | How | Forms |
|---------|-----|-------|
| CLI `transcode` / `pipe` | `--filter "crop=1280:720,hflip"` | string |
| Batch manifest | `filter:` ([batch DSL](../batch.md#per-job-keys)) | string **or** object list |
| HTTP query | `?filter=crop=1280:720,hflip` | string |
| HTTP JSON `spec` | `"filter"` ([HTTP API](../api.md)) | string **or** object list |
| IPC header | `#rivet filter=crop=1280:720,hflip` | string |
| Library | `spec.with_filters(…)` | `Vec<VideoFilter>` |

### Library

```rust
use codec::filter::{VideoFilter, parse_chain};

// build the structs directly…
let spec = OutputSpec::single_file(rungs).with_filters(vec![
    VideoFilter::Crop { w: 1920, h: 1080, x: None, y: None },
    VideoFilter::HFlip,
]);
// …or parse the string form
let spec = OutputSpec::single_file(rungs)
    .with_filters(parse_chain("crop=1920:1080,hflip")?);
```

See the [`OutputSpec` guide](../output-spec.md#6-video-filters--with_filters) for
where filters sit among the other job settings. Implementation:
[`codec::filter`](../../crates/codec/src/filter/mod.rs).
