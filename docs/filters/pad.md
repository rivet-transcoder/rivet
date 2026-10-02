# `pad`

Letterbox / pillarbox the frame into a larger `w×h` canvas filled with neutral
(limited-range) black. With an explicit `x`/`y` the source is placed there;
without, it's **centred**. The inverse of [crop](crop.md). Pure sample rewrite,
so any bit depth (8- or 10-bit).

## Syntax

```text
pad=W:H           # centred
pad=W:H:X:Y       # source at (X, Y) inside the canvas
```

```yaml
- pad: { w: 1920, h: 1080 }            # centred
- pad: { w: 1920, h: 1080, x: 0, y: 0 }
```

## Parameters

| Param | Type | Meaning |
|-------|------|---------|
| `w` | `u32` | Canvas width (clamped to be ≥ the frame width). |
| `h` | `u32` | Canvas height (clamped to be ≥ the frame height). |
| `x` | `u32?` | Left offset of the source inside the canvas. Omit to centre. |
| `y` | `u32?` | Top offset. |

## Behaviour

- **Centred mode** (no `x`/`y`): the source is placed at `((w − frame_w)/2,
  (h − frame_h)/2)`.
- **Fill colour**: limited-range black — luma 16, chroma 128 (8-bit); luma 64,
  chroma 512 (10-bit) — so the bars are true black on a BT.709 limited display,
  not a grey or a clipped value.
- **Even alignment**: `w`, `h`, `x`, `y` round down to even for 4:2:0 chroma.
- The source must fit: `x + frame_w ≤ w` and `y + frame_h ≤ h`, else it's a hard
  error. The parser does not know the frame size, so this is caught when the
  first frame is filtered, and fails the job there.

## Examples

```text
pad=1920:1080                  # centre a smaller frame on a 1080p black canvas
pad=1280:720:160:0             # pillarbox a 960-wide source, flush top
```

## Notes

- This pads the *source*, once, before every rung. To letterbox each rendition
  to exactly its rung's box, use the rung fit `pad` instead (`fit=pad`, or
  `WxH:pad` on one rung — see
  [fitting the source into a rung](../output-spec.md#fitting-the-source-into-a-rung)).
  To *remove* borders instead, see [crop](crop.md).

Source: [`crates/codec/src/filter/pad.rs`](../../crates/codec/src/filter/pad.rs).
