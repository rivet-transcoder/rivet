# Video filters

Per-frame geometric / colour transforms applied to the decoded source **once**,
before fan-out + per-rung scaling — so a filter applies to every rendition. They
run on the normalised 4:2:0 frame (8- or 10-bit), transform the *source*, and the
per-rung scaler then resizes the result to each rung. (So if a crop changes the
aspect ratio, set the rung dimensions to match.)

## The filters

| Filter | Parameters | Effect |
|--------|------------|--------|
| `crop` | `w`, `h`, optional `x`, `y` | Crop a `w×h` region; without `x`/`y` it is centred. |
| `pad` | `w`, `h`, optional `x`, `y` | Letterbox / pillarbox into a `w×h` canvas (centred, neutral black). |
| `hflip` | — | Mirror horizontally. |
| `vflip` | — | Mirror vertically. |
| `rotate` | `90` \| `180` \| `270` | Rotate clockwise; 90 / 270 swap width↔height. |
| `grayscale` | — | Drop chroma. |

4:2:0 alignment means crop / pad sizes round to even. A chain is validated when
the spec is built — a bad value like `rotate=45` is rejected up front, not at
encode time.

## Two interchangeable forms

A chain is a list of [`codec::filter::VideoFilter`](../crates/codec/src/filter.rs)
values with two serializations that round-trip exactly
(`parse_chain(&chain_to_string(c)) == c`) — use whichever fits the surface.

### String chain (ffmpeg `-vf` style)

Comma-separated, each `name` or `name=a:b:…`:

```text
crop=W:H[:X:Y]   pad=W:H[:X:Y]   hflip   vflip   rotate=90|180|270   grayscale
```

e.g. `crop=1280:720,hflip,rotate=90`. `gray` and `transpose` are accepted
aliases (`gray` = `grayscale`, `transpose` = `rotate=90`). This is the form every
**string** surface uses (CLI flag, IPC header, HTTP query string).

### Structured objects (YAML / JSON)

The batch manifest and the HTTP JSON `spec` body also accept the same filters as a
**list of objects** — unit filters as bare strings, parameterised ones as a
tagged object:

```yaml
filter:
  - crop:
      w: 1280
      h: 720          # x/y optional → centred
  - hflip
  - rotate: 90
```

```json
"filter": [{ "crop": { "w": 1280, "h": 720 } }, "hflip", { "rotate": 90 }]
```

Both forms resolve to the same validated `Vec<VideoFilter>`.

## Per-surface usage

| Surface | How | Forms |
|---------|-----|-------|
| CLI `transcode` / `pipe` | `--filter "crop=1280:720,hflip"` | string |
| Batch manifest | `filter:` ([batch DSL](batch.md#per-job-keys)) | string **or** object list |
| HTTP query | `?filter=crop=1280:720,hflip` | string |
| HTTP JSON `spec` | `"filter"` ([HTTP API](api.md)) | string **or** object list |
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

See the [`OutputSpec` guide](output-spec.md#6-video-filters--with_filters) for where
filters sit among the other job settings. Implementation:
[`codec::filter`](../crates/codec/src/filter.rs).
