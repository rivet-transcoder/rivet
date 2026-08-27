//! Single-frame AVIF thumbnail capture.
//!
//! Decodes the source video to a target frame index (default
//! `floor(0.10 * total_frames)`), turns the captured frame upright if
//! the container declared a rotation, converts it — in whatever pixel
//! format the decoder produced, using the matrix and sample range the
//! source declared — to 8-bit RGB, and encodes a still AVIF via
//! `ravif` (which wraps rav1e + a small HEIF box writer). Output is a
//! single `.avif` blob ready to store next to the renditions.
//!
//! # This path has to mirror the ladder, and kept not doing
//!
//! It builds its own decoder off the demuxed header instead of sharing
//! [`crate::decode_pump`], which is the deliberate isolation described
//! below — a thumbnail failure never stops the rung pipeline. The cost
//! is that everything the pump does to a frame has to be done here as
//! well, and three things were not: rotation, non-4:2:0 pixel formats,
//! and reading the source's own colour matrix and range. Each produced
//! a differently-wrong poster (turned, missing, or off-colour) over a
//! video that was fine, with no error anywhere. Anything added to the
//! decode path over there needs a matching thought here.
//!
//! Why a separate decode pass instead of tapping the rung decoders:
//! simpler integration boundary, isolated failure mode (a thumbnail
//! miss never prevents the rung pipeline from finalising), and the
//! cost is bounded — we only decode up to the capture frame, not the
//! full clip.
//!
//! Why AVIF: we already encode video with rav1e (AV1). Reusing AV1
//! for the still gives the same client codec story (every browser
//! that plays our video plays our thumbnail) without adding a JPEG /
//! WebP encoder to the dep graph.

use anyhow::{Context, Result, anyhow};
use bytes::Bytes;

use codec::decode;
use codec::frame::{ColorSpace, PixelFormat, VideoFrame};
use container::streaming;

/// Default offset into the clip — the rule of thumb is "10% in" so
/// the frame is past intros / fade-ins for most content.
pub const DEFAULT_THUMBNAIL_FRACTION: f64 = 0.10;

/// AVIF quality. Tuned for thumbnails: 65 → ~50 KB on a typical 1080p
/// frame, visually indistinguishable from source at thumbnail scale,
/// fast to encode (sub-second on the workspace's rav1e settings).
pub const DEFAULT_THUMBNAIL_QUALITY: f32 = 65.0;

/// rav1e speed knob (via ravif). 8 keeps encode time bounded for the
/// transcode hot path; the quality ceiling at this speed is well past
/// what's perceptible on a thumbnail.
pub const DEFAULT_THUMBNAIL_SPEED: u8 = 8;

#[derive(Debug, Clone)]
pub struct ThumbnailOutput {
    pub bytes: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

/// Capture a frame at `fraction` (0.0..=1.0) of the source's total
/// frames and encode it as AVIF. Returns the encoded bytes and the
/// frame's dimensions.
pub fn generate_thumbnail(
    input_data: &Bytes,
    fraction: f64,
    quality: f32,
    speed: u8,
) -> Result<ThumbnailOutput> {
    let captured = capture_frame_at_fraction(input_data, fraction)
        .context("capturing thumbnail source frame")?;
    let (rgb, width, height) =
        frame_to_rgb8(&captured.frame, captured.color).context("converting YUV → RGB")?;
    let avif =
        encode_avif_rgb(&rgb, width, height, quality, speed).context("encoding AVIF still")?;
    Ok(ThumbnailOutput {
        bytes: avif,
        width,
        height,
    })
}

/// The frame to encode, plus the source colour properties needed to
/// interpret its samples — see [`SourceColor`].
struct CapturedFrame {
    frame: VideoFrame,
    color: SourceColor,
}

/// Decode the source one sample at a time until we've passed the
/// target frame index, then return that frame. Falls back to the last
/// decoded frame when the stream ends before we get there (tiny
/// clips, malformed metadata reporting more frames than the file
/// contains, etc.) so a short or off-by-N stream still produces a
/// thumbnail.
///
/// # The container's rotation is applied here
///
/// A track can declare a rotation, and a poster that ignores it is
/// upside down (180°) or on its side (90°/270°) against a video that
/// plays upright. The decode pump wraps its decoder in
/// [`decode::RotatingDecoder`] for exactly this, and this path — which
/// builds its own decoder rather than sharing that one — did not, so
/// every rotated source produced a wrongly-oriented thumbnail.
///
/// `RotatingDecoder::new` is a pass-through for 0 and for any value
/// that is not 90/180/270, so this costs nothing on the common path.
fn capture_frame_at_fraction(input_data: &Bytes, fraction: f64) -> Result<CapturedFrame> {
    let mut demuxer =
        streaming::demux_streaming(input_data).context("demuxing for thumbnail capture")?;
    let header = demuxer.header().clone();
    let total_frames = header.info.total_frames.max(1);
    let target_idx = ((total_frames as f64) * fraction.clamp(0.0, 0.999)) as u64;

    // Read before `header.info` is moved into the decoder: the sample
    // range is a property of the source, and no `VideoFrame` carries it.
    let color = SourceColor {
        full_range: header.info.color_metadata.full_range,
    };

    if header.rotation_degrees != 0 {
        tracing::debug!(
            rotation_degrees = header.rotation_degrees,
            "thumbnail source carries a rotation; turning the captured frame upright",
        );
    }

    let decoder =
        decode::create_decoder(&header.codec, header.info).context("creating thumbnail decoder")?;
    let mut decoder = decode::RotatingDecoder::new(decoder, header.rotation_degrees);

    let mut current_idx: u64 = 0;
    let mut last_frame: Option<VideoFrame> = None;

    loop {
        match demuxer
            .next_video_sample()
            .context("demuxing next video sample for thumbnail")?
        {
            Some(sample) => {
                decoder
                    .push_sample(&sample.data)
                    .context("pushing sample to thumbnail decoder")?;
                while let Some(frame) = decoder
                    .decode_next()
                    .context("decoding frame for thumbnail")?
                {
                    last_frame = Some(frame);
                    if current_idx >= target_idx {
                        return last_frame
                            .map(|frame| CapturedFrame { frame, color })
                            .ok_or_else(|| anyhow!("frame slot vanished"));
                    }
                    current_idx += 1;
                }
            }
            None => {
                decoder.finish().context("decoder finish for thumbnail")?;
                while let Some(frame) = decoder
                    .decode_next()
                    .context("decoding frame after finish for thumbnail")?
                {
                    last_frame = Some(frame);
                    if current_idx >= target_idx {
                        return last_frame
                            .map(|frame| CapturedFrame { frame, color })
                            .ok_or_else(|| anyhow!("frame slot vanished"));
                    }
                    current_idx += 1;
                }
                break;
            }
        }
    }

    last_frame
        .map(|frame| CapturedFrame { frame, color })
        .ok_or_else(|| anyhow!("source produced no decoded frames"))
}

/// How a planar YUV format lays its chroma out relative to luma.
///
/// `(x_shift, y_shift)` as ffmpeg names them: the number of times to
/// halve luma width/height to get the chroma plane's dimensions. 4:2:0
/// is `(1, 1)`, 4:2:2 is `(1, 0)`, 4:4:4 is `(0, 0)`.
#[derive(Clone, Copy)]
struct Subsampling {
    x_shift: usize,
    y_shift: usize,
}

/// Sample depth of a planar format: one byte per sample, or two
/// little-endian bytes carrying a left-aligned N-bit value.
#[derive(Clone, Copy)]
enum Depth {
    Eight,
    /// `bits` significant, stored LE in a u16 — shift down to 8.
    Wide {
        bits: u32,
    },
}

/// BT.709 limited-range YUV → 8-bit RGB, for **every pixel format the
/// decoder can hand us**, not only 8-bit 4:2:0.
///
/// Chroma is walked nearest-neighbour (no upsample filter), which is
/// fine for a still that the player scales anyway.
///
/// # Why this is not `Yuv420p`-only any more
///
/// It was, and it silently cost roughly a third of all posters. The
/// thumbnail path builds its own decoder straight off the demuxed
/// header (see [`capture_frame_at_fraction`]) rather than going through
/// the decode pump, which is what normalises the main ladder's frames.
/// So this receives the source's **native**
/// format — 10-bit, 4:2:2, 4:4:4, or a hardware decoder's NV12 — while
/// the encode ladder beside it is fed normalised frames and succeeds.
///
/// The result was the worst shape a failure can take: the job finished,
/// reported no error, produced a perfectly good video, and quietly had
/// no poster — a caller that treats a thumbnail miss as non-fatal (the
/// right call) logs a warning and carries on. In one deployment a third
/// of all uploads finished this way, every one of them a plain
/// `video/mp4`, so nothing about the container distinguished them.
///
/// Every format below is converted from the layout
/// `PixelFormat::bytes_per_frame` documents, and the buffer is measured
/// against that same function rather than against arithmetic repeated
/// here — the crate that produces these frames stays the authority on
/// how big they are.
fn frame_to_rgb8(frame: &VideoFrame, color: SourceColor) -> Result<(Vec<u8>, u32, u32)> {
    let w = frame.width as usize;
    let h = frame.height as usize;
    if w == 0 || h == 0 {
        return Err(anyhow!("thumbnail frame has zero dimension"));
    }

    let data = frame.data.as_ref();
    let expected = frame.format.bytes_per_frame(frame.width, frame.height);
    if data.len() < expected {
        return Err(anyhow!(
            "thumbnail frame buffer truncated for {:?}: data={} expected≥{}",
            frame.format,
            data.len(),
            expected
        ));
    }

    let matrix = YuvMatrix::for_source(frame.color_space, color);
    let planar = |sub: Subsampling, depth: Depth| planar_to_rgb8(data, w, h, sub, depth, matrix);

    let rgb = match frame.format {
        PixelFormat::Yuv420p => planar(
            Subsampling {
                x_shift: 1,
                y_shift: 1,
            },
            Depth::Eight,
        ),
        PixelFormat::Yuv420p10le => planar(
            Subsampling {
                x_shift: 1,
                y_shift: 1,
            },
            Depth::Wide { bits: 10 },
        ),
        PixelFormat::Yuv420p12le => planar(
            Subsampling {
                x_shift: 1,
                y_shift: 1,
            },
            Depth::Wide { bits: 12 },
        ),
        PixelFormat::Yuv422p => planar(
            Subsampling {
                x_shift: 1,
                y_shift: 0,
            },
            Depth::Eight,
        ),
        PixelFormat::Yuv422p10le => planar(
            Subsampling {
                x_shift: 1,
                y_shift: 0,
            },
            Depth::Wide { bits: 10 },
        ),
        PixelFormat::Yuv422p12le => planar(
            Subsampling {
                x_shift: 1,
                y_shift: 0,
            },
            Depth::Wide { bits: 12 },
        ),
        PixelFormat::Yuv444p => planar(
            Subsampling {
                x_shift: 0,
                y_shift: 0,
            },
            Depth::Eight,
        ),
        // The alpha plane of `Yuva444p10le` trails Y/Cb/Cr and is simply
        // not read — a poster is opaque, and compositing it against an
        // assumed background would invent a colour the source never had.
        PixelFormat::Yuv444p10le | PixelFormat::Yuva444p10le => planar(
            Subsampling {
                x_shift: 0,
                y_shift: 0,
            },
            Depth::Wide { bits: 10 },
        ),
        PixelFormat::Yuv444p12le => planar(
            Subsampling {
                x_shift: 0,
                y_shift: 0,
            },
            Depth::Wide { bits: 12 },
        ),
        // Semi-planar: full luma plane, then one interleaved chroma
        // plane at 4:2:0. NV12 is Cb,Cr; NV21 is Cr,Cb. This is what a
        // hardware decoder hands back most often, and it was rejected.
        PixelFormat::Nv12 => nv_to_rgb8(data, w, h, true, matrix),
        PixelFormat::Nv21 => nv_to_rgb8(data, w, h, false, matrix),
        // Already RGB — no matrix, just drop any alpha.
        PixelFormat::Rgb24 => data[..w * h * 3].to_vec(),
        PixelFormat::Rgba32 => data[..w * h * 4]
            .chunks_exact(4)
            .flat_map(|px| [px[0], px[1], px[2]])
            .collect(),
    };

    Ok((rgb, frame.width, frame.height))
}

/// One planar YUV frame → RGB, for any subsampling and sample depth.
fn planar_to_rgb8(
    data: &[u8],
    w: usize,
    h: usize,
    sub: Subsampling,
    depth: Depth,
    matrix: YuvMatrix,
) -> Vec<u8> {
    let bytes_per_sample = match depth {
        Depth::Eight => 1,
        Depth::Wide { .. } => 2,
    };
    // `>>` then the plane's own size, matching how the frame was packed.
    let cw = w >> sub.x_shift;
    let ch = h >> sub.y_shift;

    let y_len = w * h * bytes_per_sample;
    let c_len = cw * ch * bytes_per_sample;

    let y_plane = &data[0..y_len];
    let u_plane = &data[y_len..y_len + c_len];
    let v_plane = &data[y_len + c_len..y_len + 2 * c_len];

    // A wide sample carries `bits` significant bits in a u16; shifting
    // by `bits - 8` scales it into the 8-bit domain the matrix below
    // works in, which is what the ladder's own downsample does too.
    let sample = move |plane: &[u8], idx: usize| -> f32 {
        match depth {
            Depth::Eight => plane[idx] as f32,
            Depth::Wide { bits } => {
                let lo = plane[idx * 2] as u16;
                let hi = plane[idx * 2 + 1] as u16;
                ((lo | (hi << 8)) >> (bits - 8)) as f32
            }
        }
    };

    let mut rgb = Vec::with_capacity(w * h * 3);
    for row in 0..h {
        let cy = row >> sub.y_shift;
        for col in 0..w {
            let cx = col >> sub.x_shift;
            let y = sample(y_plane, row * w + col);
            let u = sample(u_plane, cy * cw + cx);
            let v = sample(v_plane, cy * cw + cx);
            matrix.push(&mut rgb, y, u, v);
        }
    }
    rgb
}

/// NV12 / NV21 → RGB. One luma plane, then Cb/Cr interleaved at 4:2:0.
fn nv_to_rgb8(data: &[u8], w: usize, h: usize, cb_first: bool, matrix: YuvMatrix) -> Vec<u8> {
    let y_len = w * h;
    let cw = w / 2;
    let y_plane = &data[0..y_len];
    let c_plane = &data[y_len..];

    let mut rgb = Vec::with_capacity(w * h * 3);
    for row in 0..h {
        let cy = row / 2;
        for col in 0..w {
            let cx = col / 2;
            let base = (cy * cw + cx) * 2;
            let (u, v) = if cb_first {
                (c_plane[base], c_plane[base + 1])
            } else {
                (c_plane[base + 1], c_plane[base])
            };
            matrix.push(&mut rgb, y_plane[row * w + col] as f32, u as f32, v as f32);
        }
    }
    rgb
}

/// Colour properties of the source that no [`VideoFrame`] carries.
#[derive(Clone, Copy)]
struct SourceColor {
    /// H.273 `full_range_flag`. `false` is studio range (Y 16..235,
    /// chroma 16..240); `true` means the samples already span 0..255.
    full_range: bool,
}

/// The YUV → RGB conversion for one source, derived from its declared
/// matrix and sample range.
///
/// # Why this is not a fixed BT.709 matrix any more
///
/// BT.709 is the right thing to **emit** — ravif tags its output with
/// sRGB primaries and transfer, which is what every browser assumes —
/// but that is a statement about the AVIF we produce, not about how to
/// read the source. The matrix here is a *decode* step: it turns the
/// source's Y'CbCr back into RGB, and it has to be the matrix the
/// source was encoded with. Applying BT.709 coefficients to BT.601
/// content does not make the result more standards-compliant, it just
/// decodes it wrong — the error lands mostly on reds and skin tones.
///
/// So: read with whatever the source declares, hand ravif correct RGB,
/// and the AVIF is still sRGB-tagged and web-compliant. The two
/// concerns never actually competed.
///
/// **Known remaining inaccuracy:** BT.601 also specifies different
/// *primaries* (SMPTE 170M / BT.470BG) from BT.709/sRGB. Getting the
/// matrix right removes the large, visible error; a fully correct
/// conversion would additionally transform primaries in linear light,
/// which is a few-percent shift and is not done here. BT.2020 gets its
/// matrix but keeps its wide primaries un-transformed for the same
/// reason, and an HDR transfer function (PQ/HLG) is not tone-mapped at
/// all — an HDR source's poster will look flat and dark. Both are
/// worth doing and neither is why posters were wrong.
#[derive(Clone, Copy)]
struct YuvMatrix {
    /// Luma scale: `255/219` for studio range, `1.0` for full.
    y_scale: f32,
    /// Chroma scale: `255/224` for studio range, `1.0` for full.
    c_scale: f32,
    /// Luma offset subtracted before scaling: 16 studio, 0 full.
    y_offset: f32,
    kr: f32,
    kb: f32,
}

impl YuvMatrix {
    /// ITU-T H.273 luma coefficients for the matrices the decoder can
    /// report, combined with the source's sample range.
    fn for_source(space: ColorSpace, color: SourceColor) -> Self {
        let (kr, kb) = match space {
            ColorSpace::Bt601 => (0.299, 0.114),
            ColorSpace::Bt709 => (0.2126, 0.0722),
            ColorSpace::Bt2020 => (0.2627, 0.0593),
        };

        if color.full_range {
            Self {
                y_scale: 1.0,
                c_scale: 1.0,
                y_offset: 0.0,
                kr,
                kb,
            }
        } else {
            Self {
                y_scale: 255.0 / 219.0,
                c_scale: 255.0 / 224.0,
                y_offset: 16.0,
                kr,
                kb,
            }
        }
    }

    /// One pixel, appended as an RGB triplet.
    ///
    /// The standard inverse: `R = Y + 2(1-Kr)Cr`, `B = Y + 2(1-Kb)Cb`,
    /// and green from the residual via `Kg = 1 - Kr - Kb`. Written out
    /// rather than as baked constants so the coefficients above are the
    /// only thing that varies — `bt709_studio_matches_the_constants_it_replaced`
    /// pins that this reproduces the previous hard-coded numbers.
    fn push(&self, rgb: &mut Vec<u8>, y: f32, u: f32, v: f32) {
        let kg = 1.0 - self.kr - self.kb;

        let y1 = (y - self.y_offset) * self.y_scale;
        let cb = (u - 128.0) * self.c_scale;
        let cr = (v - 128.0) * self.c_scale;

        let r = y1 + 2.0 * (1.0 - self.kr) * cr;
        let b = y1 + 2.0 * (1.0 - self.kb) * cb;
        let g = y1
            - (2.0 * (1.0 - self.kb) * self.kb / kg) * cb
            - (2.0 * (1.0 - self.kr) * self.kr / kg) * cr;

        rgb.push(clamp_u8(r));
        rgb.push(clamp_u8(g));
        rgb.push(clamp_u8(b));
    }
}

fn clamp_u8(v: f32) -> u8 {
    if v <= 0.0 {
        0
    } else if v >= 255.0 {
        255
    } else {
        v.round() as u8
    }
}

/// Encode RGB pixels as AVIF via ravif. RGB ordering is (R, G, B)
/// triplets, row-major, no padding.
fn encode_avif_rgb(
    rgb: &[u8],
    width: u32,
    height: u32,
    quality: f32,
    speed: u8,
) -> Result<Vec<u8>> {
    let w = width as usize;
    let h = height as usize;
    if rgb.len() != w * h * 3 {
        return Err(anyhow!(
            "avif rgb buffer size mismatch: {} vs {}",
            rgb.len(),
            w * h * 3
        ));
    }

    // Build the row-major Img wrapper that ravif's encoder consumes.
    // Casting the u8 triplets to a slice of `rgb::Rgb<u8>` is
    // size/align-compatible: Rgb<u8> is repr(C) with three u8 fields.
    let pixels: &[rgb::Rgb<u8>] =
        unsafe { std::slice::from_raw_parts(rgb.as_ptr() as *const rgb::Rgb<u8>, w * h) };
    let img = ravif::Img::new(pixels, w, h);

    let encoded = ravif::Encoder::new()
        .with_quality(quality)
        .with_speed(speed)
        .encode_rgb(img)
        .map_err(|e| anyhow!("ravif encode failed: {e}"))?;

    Ok(encoded.avif_file)
}

#[cfg(test)]
mod tests {
    use super::*;
    use codec::frame::ColorSpace;

    /// A mid-grey-ish test colour, and what BT.709 limited-range makes
    /// of it. Computed once here so every format below asserts against
    /// the same expected RGB — the point of these tests is that the
    /// *layout* handling agrees across formats, not that the matrix is
    /// what it is.
    const Y8: u8 = 128;
    const U8_: u8 = 100;
    const V8: u8 = 200;

    fn expected_rgb() -> [u8; 3] {
        let mut v = Vec::new();
        studio_bt709().push(&mut v, Y8 as f32, U8_ as f32, V8 as f32);
        [v[0], v[1], v[2]]
    }

    fn studio_bt709() -> YuvMatrix {
        YuvMatrix::for_source(ColorSpace::Bt709, SourceColor { full_range: false })
    }

    fn frame(format: PixelFormat, data: Vec<u8>, w: u32, h: u32) -> VideoFrame {
        VideoFrame {
            data: Bytes::from(data),
            width: w,
            height: h,
            format,
            color_space: ColorSpace::Bt709,
            pts: 0,
        }
    }

    /// Every pixel of a 2×2 frame is the same colour, so whatever the
    /// subsampling and depth, all four output pixels must match.
    fn assert_uniform_2x2(format: PixelFormat, data: Vec<u8>) {
        let (rgb, w, h) = frame_to_rgb8(
            &frame(format, data, 2, 2),
            SourceColor { full_range: false },
        )
        .unwrap_or_else(|e| panic!("{format:?} should convert, got {e}"));

        assert_eq!((w, h), (2, 2));
        assert_eq!(rgb.len(), 2 * 2 * 3, "{format:?} produced the wrong length");

        let want = expected_rgb();
        for (i, px) in rgb.chunks_exact(3).enumerate() {
            assert_eq!(
                px, want,
                "{format:?} pixel {i} disagrees with the 8-bit 4:2:0 baseline",
            );
        }
    }

    /// Widen an 8-bit sample into a `bits`-deep little-endian u16.
    fn wide(sample: u8, bits: u32) -> [u8; 2] {
        ((sample as u16) << (bits - 8)).to_le_bytes()
    }

    #[test]
    fn eight_bit_420_still_converts() {
        // The only format that ever worked. Kept first so a regression
        // here is not hidden among the formats being added.
        assert_uniform_2x2(PixelFormat::Yuv420p, vec![Y8, Y8, Y8, Y8, U8_, V8]);
    }

    #[test]
    fn ten_bit_420_converts_instead_of_erroring() {
        // The shape that silently cost a third of all posters: a job
        // that encodes perfectly and produces no thumbnail.
        let mut data = Vec::new();
        for _ in 0..4 {
            data.extend_from_slice(&wide(Y8, 10));
        }
        data.extend_from_slice(&wide(U8_, 10));
        data.extend_from_slice(&wide(V8, 10));
        assert_uniform_2x2(PixelFormat::Yuv420p10le, data);
    }

    #[test]
    fn twelve_bit_420_converts() {
        let mut data = Vec::new();
        for _ in 0..4 {
            data.extend_from_slice(&wide(Y8, 12));
        }
        data.extend_from_slice(&wide(U8_, 12));
        data.extend_from_slice(&wide(V8, 12));
        assert_uniform_2x2(PixelFormat::Yuv420p12le, data);
    }

    #[test]
    fn four_two_two_converts_at_both_depths() {
        // 4:2:2 halves horizontally only — two chroma samples for a
        // 2×2 frame, not one. Getting `y_shift` wrong here reads the
        // second row's chroma out of the next plane.
        assert_uniform_2x2(PixelFormat::Yuv422p, vec![Y8, Y8, Y8, Y8, U8_, U8_, V8, V8]);

        let mut wide_data = Vec::new();
        for _ in 0..4 {
            wide_data.extend_from_slice(&wide(Y8, 10));
        }
        for _ in 0..2 {
            wide_data.extend_from_slice(&wide(U8_, 10));
        }
        for _ in 0..2 {
            wide_data.extend_from_slice(&wide(V8, 10));
        }
        assert_uniform_2x2(PixelFormat::Yuv422p10le, wide_data);
    }

    #[test]
    fn four_four_four_converts_at_both_depths() {
        assert_uniform_2x2(
            PixelFormat::Yuv444p,
            vec![Y8, Y8, Y8, Y8, U8_, U8_, U8_, U8_, V8, V8, V8, V8],
        );

        let mut wide_data = Vec::new();
        for _ in 0..4 {
            wide_data.extend_from_slice(&wide(Y8, 10));
        }
        for _ in 0..4 {
            wide_data.extend_from_slice(&wide(U8_, 10));
        }
        for _ in 0..4 {
            wide_data.extend_from_slice(&wide(V8, 10));
        }
        assert_uniform_2x2(PixelFormat::Yuv444p10le, wide_data);
    }

    #[test]
    fn alpha_444_ignores_its_alpha_plane_rather_than_refusing() {
        // Y/Cb/Cr then a 16-bit alpha plane. The alpha is deliberately
        // unread — a poster is opaque — but its presence must not shift
        // the chroma planes' offsets.
        let mut data = Vec::new();
        for _ in 0..4 {
            data.extend_from_slice(&wide(Y8, 10));
        }
        for _ in 0..4 {
            data.extend_from_slice(&wide(U8_, 10));
        }
        for _ in 0..4 {
            data.extend_from_slice(&wide(V8, 10));
        }
        for _ in 0..4 {
            data.extend_from_slice(&u16::MAX.to_le_bytes());
        }
        assert_uniform_2x2(PixelFormat::Yuva444p10le, data);
    }

    #[test]
    fn nv12_and_nv21_disagree_only_about_chroma_order() {
        // What a hardware decoder hands back, and what this rejected
        // outright. NV21 is the same bytes with Cr first, so feeding
        // each its own ordering must land on the same colour.
        assert_uniform_2x2(PixelFormat::Nv12, vec![Y8, Y8, Y8, Y8, U8_, V8]);
        assert_uniform_2x2(PixelFormat::Nv21, vec![Y8, Y8, Y8, Y8, V8, U8_]);
    }

    #[test]
    fn rgb_formats_pass_through_without_the_matrix() {
        let (rgb, _, _) = frame_to_rgb8(
            &frame(
                PixelFormat::Rgb24,
                vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                2,
                2,
            ),
            SourceColor { full_range: false },
        )
        .expect("rgb24 converts");
        assert_eq!(rgb, vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);

        let (rgba, _, _) = frame_to_rgb8(
            &frame(
                PixelFormat::Rgba32,
                vec![1, 2, 3, 255, 4, 5, 6, 255, 7, 8, 9, 255, 10, 11, 12, 255],
                2,
                2,
            ),
            SourceColor { full_range: false },
        )
        .expect("rgba32 converts");
        assert_eq!(rgba, vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    }

    #[test]
    fn a_truncated_buffer_is_refused_rather_than_read_past() {
        // The check is against `bytes_per_frame`, so it has to hold for
        // a format whose size this module never computes by hand.
        let err = frame_to_rgb8(
            &frame(PixelFormat::Yuv444p10le, vec![0; 8], 2, 2),
            SourceColor { full_range: false },
        )
        .expect_err("a short buffer must not be read");
        assert!(
            err.to_string().contains("truncated"),
            "unexpected error: {err}"
        );
    }

    /// The generalised matrix must reproduce, exactly, the hard-coded
    /// BT.709 studio-range numbers it replaced. Without this, widening
    /// the conversion to other matrices could quietly shift the colour
    /// of every thumbnail that was already correct.
    #[test]
    fn bt709_studio_matches_the_constants_it_replaced() {
        let m = studio_bt709();

        for (y, u, v) in [
            (16.0, 128.0, 128.0),
            (128.0, 100.0, 200.0),
            (235.0, 240.0, 16.0),
            (77.0, 33.0, 210.0),
        ] {
            // The previous implementation, verbatim.
            let y1 = (y - 16.0) * 1.164_383_5;
            let cb = u - 128.0;
            let cr = v - 128.0;
            let want = [
                clamp_u8(y1 + 1.792_741_1 * cr),
                clamp_u8(y1 - 0.213_248_5 * cb - 0.532_909_3 * cr),
                clamp_u8(y1 + 2.112_401_8 * cb),
            ];

            let mut got = Vec::new();
            m.push(&mut got, y, u, v);

            assert_eq!(got, want, "BT.709 studio drifted at Y={y} U={u} V={v}");
        }
    }

    /// BT.601 and BT.709 must actually differ, or reading the source's
    /// declared matrix would be a no-op dressed up as a fix. Checked on
    /// a saturated red, where the two matrices disagree most.
    #[test]
    fn bt601_is_not_bt709() {
        let bt601 = YuvMatrix::for_source(ColorSpace::Bt601, SourceColor { full_range: false });
        let bt709 = studio_bt709();

        let (y, u, v) = (81.0, 90.0, 240.0);
        let (mut a, mut b) = (Vec::new(), Vec::new());
        bt601.push(&mut a, y, u, v);
        bt709.push(&mut b, y, u, v);

        assert_ne!(
            a, b,
            "decoding with the source's own matrix must change the result,              otherwise BT.601 content is still being read as BT.709",
        );
    }

    /// Full-range sources span 0..255 already. Applying the studio
    /// offset and scale to them crushes blacks and clips whites — the
    /// second half of reading the source correctly, and common in phone
    /// and screen recordings.
    #[test]
    fn full_range_is_not_scaled_like_studio_range() {
        let full = YuvMatrix::for_source(ColorSpace::Bt709, SourceColor { full_range: true });
        let studio = studio_bt709();

        // Full-range black is 0. Studio maths would drive it negative
        // and clamp; full-range maths lands exactly on 0 either way, so
        // check a mid-tone where the scale difference shows.
        let (mut a, mut b) = (Vec::new(), Vec::new());
        full.push(&mut a, 128.0, 128.0, 128.0);
        studio.push(&mut b, 128.0, 128.0, 128.0);
        assert_ne!(a, b, "full range must not use the studio scale");

        // A full-range grey maps straight through, with no 16-offset.
        let mut grey = Vec::new();
        full.push(&mut grey, 200.0, 128.0, 128.0);
        assert_eq!(
            grey,
            vec![200, 200, 200],
            "neutral full-range luma must pass through"
        );
    }

    /// Turn a 180° display matrix into a copy of `data`'s `tkhd`.
    ///
    /// Avoids committing a second multi-hundred-KB sample: the rotated
    /// input is the committed one with nine words rewritten, so the coded
    /// frames are byte-identical by construction and the declared
    /// rotation is the only difference between the two runs.
    ///
    /// ISO/IEC 14496-12 §8.3.2: `tkhd` is `size|type|version|flags`, then
    /// 20 bytes (v0) or 32 (v1) of times/id/duration, then 16 bytes of
    /// reserved/layer/alternate_group/volume, then the 9-word matrix.
    /// A wrong offset here cannot pass silently — the test asserts the
    /// demuxer reads 180 back out.
    fn with_180_display_matrix(data: &[u8]) -> Vec<u8> {
        fn patch(buf: &mut [u8], mut pos: usize, end: usize) -> bool {
            while pos + 8 <= end {
                let size = u32::from_be_bytes(buf[pos..pos + 4].try_into().unwrap()) as usize;
                let kind: [u8; 4] = buf[pos + 4..pos + 8].try_into().unwrap();
                if size < 8 || pos + size > end {
                    return false;
                }
                match &kind {
                    // Containers worth descending into on the way to tkhd.
                    b"moov" | b"trak" => {
                        if patch(buf, pos + 8, pos + size) {
                            return true;
                        }
                    }
                    b"tkhd" => {
                        let version = buf[pos + 8];
                        let fixed = if version == 1 { 32 } else { 20 };
                        let m = pos + 8 + 4 + fixed + 16;
                        if m + 36 > pos + size {
                            return false;
                        }
                        // a=-1, d=-1, w=1.0 — a half turn. Everything else 0.
                        let words: [i32; 9] = [-65536, 0, 0, 0, -65536, 0, 0, 0, 0x4000_0000];
                        for (i, word) in words.iter().enumerate() {
                            buf[m + i * 4..m + i * 4 + 4].copy_from_slice(&word.to_be_bytes());
                        }
                        return true;
                    }
                    _ => {}
                }
                pos += size;
            }
            false
        }

        let mut out = data.to_vec();
        let end = out.len();
        assert!(patch(&mut out, 0, end), "no tkhd found in the sample");
        out
    }

    /// The rotation fix, against a real file decoded by the real decoder.
    ///
    /// Unit tests cover the conversion maths; they cannot cover the thing
    /// that was actually broken — that the decoder is wrapped at all, and
    /// that the container's rotation is read. The bug this pins produced a
    /// poster turned against a video that plays upright.
    ///
    /// The comparison is exact rather than approximate because both runs
    /// decode the same coded frames: only the declared rotation differs,
    /// so a correct implementation must yield one as the 180° turn of the
    /// other with no tolerance at all.
    /// `RIVET_TEST_MEDIA` env override, else the workspace `test_media/`
    /// dir — the same lookup the integration tests use. The corpus is
    /// fetched on demand and never committed, so a missing file is a
    /// skip, not a failure.
    fn read_test_media(name: &str) -> Option<Vec<u8>> {
        let dir = match std::env::var_os("RIVET_TEST_MEDIA") {
            Some(dir) => std::path::PathBuf::from(dir),
            None => std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .parent()?
                .parent()?
                .join("test_media"),
        };
        std::fs::read(dir.join(name)).ok()
    }

    #[test]
    fn a_rotated_source_is_captured_upright() {
        let Some(base) = read_test_media("bbb_h264_360p_short.mp4") else {
            eprintln!("SKIP: test_media/bbb_h264_360p_short.mp4 not present");
            return;
        };
        let base = Bytes::from(base);
        let rotated = Bytes::from(with_180_display_matrix(&base));

        assert_eq!(
            streaming::demux_streaming(&base)
                .unwrap()
                .header()
                .rotation_degrees,
            0,
            "the untouched sample should declare no rotation",
        );
        assert_eq!(
            streaming::demux_streaming(&rotated)
                .unwrap()
                .header()
                .rotation_degrees,
            180,
            "the demuxer must read the patched display matrix — without this the \
             rest of the test would pass for the wrong reason",
        );

        let cap_base = match capture_frame_at_fraction(&base, 0.10) {
            Ok(cap) => cap,
            Err(e) => {
                // A host with no H.264 decoder (no NVDEC/QSV and no
                // `openh264-fallback`) cannot run this; that is a build
                // property, not a thumbnail bug.
                eprintln!("SKIP: no H.264 decoder on this host/build ({e:#})");
                return;
            }
        };
        let cap_rot = capture_frame_at_fraction(&rotated, 0.10).expect("captures");

        let (rgb_base, w, h) = frame_to_rgb8(&cap_base.frame, cap_base.color).expect("converts");
        let (rgb_rot, w2, h2) = frame_to_rgb8(&cap_rot.frame, cap_rot.color).expect("converts");

        assert_eq!(
            (w, h),
            (w2, h2),
            "a 180° turn must not change the dimensions"
        );

        // 180°: (x, y) → (W-1-x, H-1-y), which over a row-major RGB buffer
        // is simply the triplet order reversed.
        let n = (w as usize) * (h as usize);
        let mut expected = Vec::with_capacity(n * 3);
        for i in (0..n).rev() {
            expected.extend_from_slice(&rgb_base[i * 3..i * 3 + 3]);
        }

        assert_eq!(
            rgb_rot, expected,
            "the captured frame was not turned upright — a rotated source still \
             yields a rotated poster",
        );
    }

    #[test]
    fn a_zero_dimension_frame_is_refused() {
        assert!(
            frame_to_rgb8(
                &frame(PixelFormat::Yuv420p, vec![], 0, 0),
                SourceColor { full_range: false }
            )
            .is_err()
        );
    }
}
