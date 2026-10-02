# `rotate`

Rotate the frame clockwise by 90, 180, or 270 degrees. 90 and 270 **swap width ↔
height**. Pure sample rearrangement, so any bit depth (8- or 10-bit). Only
right-angle rotations are supported (arbitrary angles would need interpolation).

## Syntax

```text
rotate=90 | 180 | 270
rotate               # no value: 90
transpose            # alias for rotate=90
```

```yaml
- rotate: 90
```

## Parameters

| Param | Type | Meaning |
|-------|------|---------|
| (value) | `u32` | Clockwise degrees — must be `90`, `180`, or `270`. Any other value is rejected at validation time. |

## Behaviour

- **90°**: `dst(r,c) = src(h−1−c, r)` — output is `h×w`.
- **270°**: `dst(r,c) = src(c, w−1−r)` — output is `h×w`.
- **180°**: composed from [hflip](hflip.md) + [vflip](vflip.md) — output stays `w×h`.

All three planes rotate together; chroma rotates at half resolution.

## Examples

```text
rotate=90              # portrait → landscape (or vice-versa)
transpose              # same as rotate=90
rotate=180             # upside-down
```

90/270 swap the dimensions. Rungs are fitted to the rotated shape, and with
the default `orientation=auto` a rung's box turns with it (a 1920x1080 rung on
the now-portrait picture is read as 1080x1920), so the rung sizes need no
change.

## Notes

- `rotate=45` (or any non-right-angle) is a hard error — there is no resampling
  path. For mirrors, see [hflip](hflip.md) / [vflip](vflip.md).

Source: [`crates/codec/src/filter/rotate.rs`](../../crates/codec/src/filter/rotate.rs).
