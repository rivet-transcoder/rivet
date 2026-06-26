//! Single-frame AVIF thumbnail capture.
//!
//! Decodes the source video to a target frame index (default
//! `floor(0.10 * total_frames)`), normalises the frame to BT.709
//! limited-range Yuv420p (so non-BT.709 sources still render right
//! everywhere), converts YUV → RGB, and encodes a still AVIF via
//! `ravif` (which wraps rav1e + a small HEIF box writer). Output is a
//! single `.avif` blob ready for S3 upload.
//!
//! Why a separate decode pass instead of tapping the existing CMAF
//! variant decoders: simpler integration boundary, isolated failure
//! mode (a thumbnail miss never prevents the variant pipeline from
//! finalising), and the cost is bounded — we only decode up to the
//! capture frame, not the full clip.
//!
//! Why AVIF: we already encode video with rav1e (AV1). Reusing AV1
//! for the still gives the same client codec story (every browser
//! that plays our video plays our thumbnail) without adding a JPEG /
//! WebP encoder to the dep graph.

use anyhow::{Context, Result, anyhow};
use bytes::Bytes;

use codec::decode;
use codec::frame::{PixelFormat, VideoFrame};
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
    let frame = capture_frame_at_fraction(input_data, fraction)
        .context("capturing thumbnail source frame")?;
    let (rgb, width, height) = yuv420p_to_rgb8(&frame).context("converting YUV → RGB")?;
    let avif =
        encode_avif_rgb(&rgb, width, height, quality, speed).context("encoding AVIF still")?;
    Ok(ThumbnailOutput {
        bytes: avif,
        width,
        height,
    })
}

/// Decode the source one sample at a time until we've passed the
/// target frame index, then return that frame. Falls back to the last
/// decoded frame when the stream ends before we get there (tiny
/// clips, malformed metadata reporting more frames than the file
/// contains, etc.) so a short or off-by-N stream still produces a
/// thumbnail.
fn capture_frame_at_fraction(input_data: &Bytes, fraction: f64) -> Result<VideoFrame> {
    let mut demuxer =
        streaming::demux_streaming(input_data).context("demuxing for thumbnail capture")?;
    let header = demuxer.header().clone();
    let total_frames = header.info.total_frames.max(1);
    let target_idx = ((total_frames as f64) * fraction.clamp(0.0, 0.999)) as u64;

    let mut decoder =
        decode::create_decoder(&header.codec, header.info).context("creating thumbnail decoder")?;

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
                        return last_frame.ok_or_else(|| anyhow!("frame slot vanished"));
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
                        return last_frame.ok_or_else(|| anyhow!("frame slot vanished"));
                    }
                    current_idx += 1;
                }
                break;
            }
        }
    }

    last_frame.ok_or_else(|| anyhow!("source produced no decoded frames"))
}

/// BT.709 limited-range YUV → 8-bit RGB. Walks 4:2:0 subsampling
/// (one chroma sample per 2×2 luma block) directly — no upsample
/// filter, just nearest-neighbor — which is fine for a thumbnail
/// where the output ends up scaled by the player anyway.
fn yuv420p_to_rgb8(frame: &VideoFrame) -> Result<(Vec<u8>, u32, u32)> {
    if frame.format != PixelFormat::Yuv420p {
        return Err(anyhow!("thumbnail expects Yuv420p, got {:?}", frame.format));
    }
    let w = frame.width as usize;
    let h = frame.height as usize;
    if w == 0 || h == 0 {
        return Err(anyhow!("thumbnail frame has zero dimension"));
    }

    let y_size = w * h;
    let cw = w / 2;
    let ch = h / 2;
    let c_size = cw * ch;

    let data = frame.data.as_ref();
    if data.len() < y_size + 2 * c_size {
        return Err(anyhow!(
            "thumbnail frame plane buffer truncated: data={} expected≥{}",
            data.len(),
            y_size + 2 * c_size
        ));
    }
    let y_plane = &data[0..y_size];
    let u_plane = &data[y_size..y_size + c_size];
    let v_plane = &data[y_size + c_size..y_size + 2 * c_size];

    let mut rgb = Vec::with_capacity(w * h * 3);
    for row in 0..h {
        let cy = row / 2;
        for col in 0..w {
            let cx = col / 2;
            let y = y_plane[row * w + col] as f32;
            let u = u_plane[cy * cw + cx] as f32;
            let v = v_plane[cy * cw + cx] as f32;

            // BT.709 limited-range matrix (Y in [16,235], C in [16,240]).
            let y1 = (y - 16.0) * 1.164_383_5;
            let cb = u - 128.0;
            let cr = v - 128.0;

            let r = y1 + 1.792_741_1 * cr;
            let g = y1 - 0.213_248_5 * cb - 0.532_909_3 * cr;
            let b = y1 + 2.112_401_8 * cb;

            rgb.push(clamp_u8(r));
            rgb.push(clamp_u8(g));
            rgb.push(clamp_u8(b));
        }
    }

    Ok((rgb, frame.width, frame.height))
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
