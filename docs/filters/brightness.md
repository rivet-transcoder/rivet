# `brightness`

Brighten or darken by adding a constant offset to luma. 8-bit `Yuv420p` only.

## Syntax

```text
brightness=N      # N in -255..=255
```

```yaml
- brightness: 20
```

## Parameters

| Param | Type | Meaning |
|-------|------|---------|
| (value) | `i32` | Luma offset, `-255..=255`. Positive brightens, negative darkens. A larger magnitude is not rejected; it clips the same as ±255. |

## Behaviour

`luma = clamp(luma + N, 0, 255)`. **Chroma is untouched**, so only perceived
brightness changes — hue and saturation are preserved. Large offsets clip (a
bright region pinned at 255 stays 255).

## Examples

```text
brightness=20      # lift shadows a little
brightness=-30     # darken
```

## Notes

- For contrast (stretch around mid-grey) rather than a flat lift, see
  [contrast](contrast.md); for colour intensity, [saturation](saturation.md).
- 8-bit SDR only — a 10-bit/HDR frame is rejected.

Source: [`crates/codec/src/filter/brightness.rs`](../../crates/codec/src/filter/brightness.rs).
