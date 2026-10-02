//! VP8 through this workspace's own decoder (`crates/vp8`, the rivet-vp8
//! repository), written clean-room from RFC 6386 and bit-exact on the
//! comprehensive test vectors.
//!
//! The software tier for VP8: NVDEC takes it first where there is one, and
//! this is what decodes it everywhere else. Each sample is one compressed
//! frame (an IVF frame, a WebM block, an MP4 sample); a hidden frame (an
//! altref) produces nothing. Output is 8-bit 4:2:0, BT.601 (VP8 has no other
//! colour space).

use std::collections::VecDeque;

use anyhow::{Result, bail};
use bytes::Bytes;

use super::Decoder;
use crate::frame::{ColorSpace, PixelFormat, StreamInfo, VideoFrame};

/// The codec labels the VP8 tier serves.
pub fn supports(codec_lower: &str) -> bool {
    matches!(codec_lower, "vp8" | "vp08")
}

/// A VP8 decoder behind rivet's [`Decoder`] trait.
pub struct Vp8Decoder {
    inner: vp8::Decoder,
    info: StreamInfo,
    ready: VecDeque<VideoFrame>,
    next_pts: u64,
}

impl Vp8Decoder {
    pub fn new(info: StreamInfo) -> Result<Self> {
        let codec = info.codec.to_ascii_lowercase();
        if !supports(&codec) {
            bail!("the VP8 decoder decodes VP8, not '{codec}'");
        }
        Ok(Self { inner: vp8::Decoder::new(), info, ready: VecDeque::new(), next_pts: 0 })
    }
}

impl Decoder for Vp8Decoder {
    fn stream_info(&self) -> &StreamInfo {
        &self.info
    }

    fn push_sample(&mut self, data: &[u8]) -> Result<()> {
        let decoded = self
            .inner
            .decode(data)
            .map_err(anyhow::Error::new)
            .map_err(|e| e.context("the VP8 decoder could not decode a frame"))?;
        if let Some(frame) = decoded {
            let pts = self.next_pts;
            self.next_pts += 1;
            let (width, height) = (frame.width, frame.height);
            self.ready.push_back(VideoFrame::new(
                Bytes::from(frame.into_packed()),
                width,
                height,
                PixelFormat::Yuv420p,
                ColorSpace::Bt601,
                pts,
            ));
        }
        Ok(())
    }

    fn finish(&mut self) -> Result<()> {
        Ok(())
    }

    fn decode_next(&mut self) -> Result<Option<VideoFrame>> {
        Ok(self.ready.pop_front())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn info(codec: &str) -> StreamInfo {
        StreamInfo {
            codec: codec.to_string(),
            width: 0,
            height: 0,
            frame_rate: 0.0,
            duration: 0.0,
            pixel_format: PixelFormat::Yuv420p,
            color_space: ColorSpace::Bt601,
            total_frames: 0,
            bitrate: 0,
            color_metadata: Default::default(),
        }
    }

    #[test]
    fn only_vp8_constructs() {
        assert!(Vp8Decoder::new(info("vp8")).is_ok());
        assert!(Vp8Decoder::new(info("vp9")).is_err());
    }

    /// Frames from the crate's encoder come back as pipeline frames, in order.
    #[test]
    fn encoded_frames_come_back_in_order() {
        let (w, h) = (64u32, 48u32);
        let mut enc = vp8::Encoder::new(vp8::Config { width: w, height: h, ..vp8::Config::default() })
            .expect("encoder");
        let mut dec = Vp8Decoder::new(info("vp8")).expect("decoder");
        for n in 0..3u32 {
            let mut frame = vp8::Frame::new(w, h).expect("frame");
            for (i, s) in frame.data.iter_mut().enumerate() {
                *s = ((i as u32 + n * 7) % 251) as u8;
            }
            let packet = enc.encode(&frame).expect("encode");
            dec.push_sample(&packet).expect("decode");
        }
        dec.finish().unwrap();
        for pts in 0..3 {
            let f = dec.decode_next().unwrap().expect("a frame");
            assert_eq!((f.width, f.height, f.format, f.pts), (w, h, PixelFormat::Yuv420p, pts));
            assert_eq!(f.data.len(), (w * h * 3 / 2) as usize);
        }
        assert!(dec.decode_next().unwrap().is_none());
    }

    #[test]
    fn garbage_is_an_error() {
        let mut dec = Vp8Decoder::new(info("vp8")).expect("decoder");
        assert!(dec.push_sample(&[0xff, 1, 2]).is_err());
    }
}
